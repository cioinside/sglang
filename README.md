<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)

</div>

# SGLang — SM86 Ampere Fork

> **Fork of [SGLang](https://github.com/sgl-project/sglang) v0.5.9** with patches to enable NVIDIA Ampere (SM80/SM86) GPU support.

> **Patch 1 (SM86→SM90 binary fallback) is based on [grevinden/sglang-kt-sm86](https://github.com/grevinden/sglang-kt-sm86)** — a Docker-based SGLang build for SM86/SM89 GPUs. Their `load_utils_sm86.patch` for `sgl_kernel` was the starting point for our SM86 support. We extended it with a null-safety check (`is not None`) and added two additional patches for CuDNN and AWQ weight loading.

## Problem

SGLang v0.5.9 dropped Ampere GPU support in the `sgl-kernel` package. The upstream only ships prebuilt CUDA binaries for SM90 (Hopper/H100) and SM100 (Blackwell). On Ampere GPUs (RTX 3060, RTX 3070, RTX 3080, RTX 3090, A100, etc.), SGLang crashes at startup:

```
sgl_kernel CRITICAL: Could not load any common_ops library!
GPU Info: compute_capability = 86
```

Additionally, SGLang's CuDNN compatibility check blocks text-only LLM serving (the Conv3d bug only affects multimodal models), and the Qwen3.5 MoE weight loader crashes on AWQ-quantized checkpoints with mixed key formats. Memory-constrained Ampere cards (e.g., 2× RTX 3060 12GB serving Qwen3.6-35B-A3B) also cannot fit long contexts because the bf16 KV cache grows linearly with sequence length — this fork adds a **Q4_0 quantized KV cache** that packs K/V at ~3.5× smaller per-token cost, multiplying available context without any model weight changes.

## Solutions (4 Patches)

### Patch 1: `sgl-kernel` SM86→SM90 Binary Fallback

> **Based on [grevinden/sglang-kt-sm86](https://github.com/grevinden/sglang-kt-sm86)** — original `patches/load_utils_sm86.patch`.

**File**: `sgl-kernel/python/sgl_kernel/load_utils.py`

Adds a fallback path for GPUs with compute capability < 90 (all Ampere cards) to load the SM90 (Hopper) `common_ops` binaries. The SM90 fast-math binaries are compatible with SM86 — same architecture generation, identical instruction set for the operations SGLang uses.

**Effect**: SGLang starts successfully on Ampere GPUs instead of crashing with "Could not load any common_ops library."

### Patch 2: Skip CuDNN Conv3d Compatibility Check

**File**: `python/sglang/srt/server_args.py` (function `_check_cudnn_fa_info`)

Bypasses the PyTorch 2.9.1 / CuDNN < 9.15 compatibility check. This check was added for `nn.Conv3d` performance (multimodal models only). For text-only LLM serving, Conv3d is never used, so the check is irrelevant and blocks startup unnecessarily.

**Change**: Replaces the existing CuDNN version check body with an early `return` — the function still exists for API compatibility but no longer blocks server startup.

**Effect**: Eliminates the `CRITICAL WARNING: PyTorch 2.9.1 & CuDNN Compatibility Issue Detected` error when serving text models.

### Patch 3: AWQ MoE Weight Loader — Skip Unknown Parameters

**File**: `python/sglang/srt/models/qwen3_5.py` (in `Qwen3_5ForConditionalGeneration.load_weights()`)

Relaxes the Qwen3.5 MoE weight loader to skip any parameter not found in `params_dict`, not just those ending with known GPTQ suffixes. AWQ-quantized checkpoints (e.g., `QuantTrio/Qwen3.6-35B-A3B-AWQ`) use mixed key formats — the first layer has standard `.weight` keys (fp16) while subsequent layers use `.qweight/.qzeros/.scales` keys. The upstream code only skips keys ending with specific GPTQ suffixes, causing a `KeyError` on unrecognized parameter names.

**Change**: Extends the `ignore_suffixes` condition to also `continue` (skip) any key not present in `params_dict` — catches all unknown parameters regardless of naming convention.

**Effect**: AWQ-quantized Qwen3.5 MoE models load successfully in SGLang.

### Patch 4: Q4_0 KV Cache Quantization

**Files**:
- `python/sglang/srt/layers/quantization/q40_tensor.py` (new) — `Q40KVQuantizeUtil` (batched quantize/dequantize for GGUF Q4_0 format: 32 values/block, 4-bit packed data + fp16 scales)
- `python/sglang/srt/layers/attention/ops/q40_decode.py` (new) — Q4_0-aware KV-cache **get** path (dequantize → attention K/V)
- `python/sglang/srt/layers/attention/ops/q40_store.py` (new) — Q4_0-aware KV-cache **set** path (attention → quantize → store)
- `python/sglang/srt/mem_cache/memory_pool.py` — `MHATokenToKVPoolQ40` (Q4_0 pool class); `HybridLinearKVPool` dispatches on `kv_cache_dtype_str`
- `python/sglang/srt/mem_cache/kv_cache_dtype.py` — accepts `"q4_0"` (storage dtype stays bf16; Q4_0 applies at set/get)
- `python/sglang/srt/mem_cache/kv_cache_configurator.py` + `model_executor/pool_configurator.py` — Q4_0 cell-size accounting
- `python/sglang/srt/server_args.py` — adds `"q4_0"` to `kv_cache_dtype` choices

**Format**: GGUF Q4_0 — 32-element blocks, each block stores one fp16 scale plus 32 nibbles (4 bits each) packed into 16 bytes. Per-token storage ≈ `(head_dim + v_head_dim) × num_layers × num_kv_heads × 18 / 32` bytes for K (and same for V), versus bf16's `(head_dim + v_head_dim) × num_layers × num_kv_heads × 2`.

**Effect**: `--kv-cache-dtype q4_0` enables the Q4_0 pool. K and V are quantized on write (set) and dequantized on read (get), so the runtime KV footprint drops ~3.5× vs bf16. On Qwen3.6-35B-A3B (hybrid SWA + Mamba) on 2× RTX 3060 12GB, this raises the maximum context from ~16.6K tokens (bf16) to ~59K tokens (auto-sized), with no model weight changes required.

#### Performance (Qwen3.6-35B-A3B, 2× RTX 3060 12GB, TP=2, Q4_0 + CUDA graphs)

| KV dtype | `--max-total-tokens` | K=V size (each) | Decode tok/s (500-gen) | Notes |
|----------|----------------------|-----------------|------------------------|-------|
| bf16     | 16,592 (auto)        | 0.08 GB         | 87.9                   | baseline |
| Q4_0     | 16,000               | 0.02 GB         | 58.2                   | 4× smaller pool, 66% of bf16 speed |
| Q4_0     | 24,000               | 0.03 GB         | ~58                    | 44% more context vs bf16 |
| Q4_0     | 33,000               | 0.04 GB         | ~58                    | 99% more context vs bf16 |

Decode throughput at 50 / 200 / 500 generated tokens (CUDA graphs enabled):
- Q4_0: 36.5 / 57.2 / 58.2 tok/s
- bf16: 52.8 / 83.5 / 87.9 tok/s

Long-context stability: a 6.4K-token prompt + 6.4K-token generation through a full Q4_0 round-trip produces coherent output (90 distinct chars, 0 CJK garbage). No OOM.

CUDA Graph capture works at `bs=[1]` on the first request (~3.2 s capture).

#### Three Q4_0 bugs fixed during bring-up

1. **`HybridLinearKVPool` hardcoded the bf16 pool class.** Hybrid SWA + Mamba models always took this path, which unconditionally used `MHATokenToKVPool` (bf16 buffers). With Q4_0 requested, the pool still allocated bf16, leading to OOM. Fix: `HybridLinearKVPool` now dispatches on `kv_cache_dtype_str` and selects `MHATokenToKVPoolQ40` when `"q4_0"`.

2. **`set_num_tokens_hybrid_swa` used bf16 per-token costs.** The function recomputed `total_memory` from `max_total_num_tokens` using bf16 cell size; with Q4_0 cell size (much smaller), the recomputed memory was inflated ~7×, leading to oversized `full_max_total_num_tokens`. Fix: use Q4_0 packed-format per-token costs (`2 × (n × (k//2) + (n×k//32) × 2)`) when `kv_cache_dtype == "q4_0"`.

3. **Missing `+ 8` offset in `batched_quantize`.** The quantizer did `q = round(value / scale); clamp(0, 15)`, which clamped every negative quantized value to 0. Dequant `(q − 8) × scale` then produced wildly wrong values for all negative K/V entries — manifesting as garbage tokens. Fix: `q = q + 8` after rounding, before clamping. This maps the symmetric range `[−7.5, +7.5]` to unsigned `[0, 15]` — the GGUF Q4_0 convention.

**Bonus**: `batched_dequantize` originally used `float32` intermediates (~90 MB per layer), which OOMed during inference. Switched to `bf16` intermediates (~45 MB) — fits in the ~0.55 GB headroom on a 12 GB card.

#### Caveats

- Currently enabled for hybrid SWA + Mamba models (the model that prompted this work). The pure-MHA path (`MHATokenToKVPool` vs `MHATokenToKVPoolQ40`) is wired through the same `configure_kv_cache_dtype` and pool-configurator paths, but has not been benchmarked end-to-end on a non-hybrid checkpoint.
- Auto-sizing (`profile_max_num_token` returns ~59K) does **not** fit during inference: the full-pool dequant requires enough headroom for ~45 MB bf16 intermediates per layer. On 2× 12 GB cards, `--max-total-tokens 16000–33000` works comfortably; the full 59 K auto-size OOMs on 12 GB cards during inference (would require direct Q4_0 attention kernels that avoid materializing the full dequantized buffer).
- Decode throughput is ~66% of bf16 because of the quantize/dequant on every KV access. Prefill is unaffected (Q4_0 is applied to the cache, not to activations).

## Applicability

| Hardware | Status |
|----------|--------|
| RTX 3060 (SM86) | Tested — works with patches |
| RTX 3070/3080/3090 (SM86) | Expected to work (same SM86 arch) |
| A100 (SM80) | Expected to work (SM80 < SM90 fallback) |
| H100/H200 (SM90) | Unaffected — upstream works natively |
| B100/B200 (SM100) | Unaffected — upstream works natively |

## Known Limitations

- **Memory pressure on 12GB cards**: Large MoE models (e.g., Qwen3.6-35B-A3B AWQ at ~10.7 GB per GPU with TP=2) may fail CUDA graph capture due to insufficient VRAM for KV cache + CUDA graph workspace. Consider `--mem-fraction-static 0.95`, reducing context length, or using `--kv-cache-dtype q4_0` (Patch 4) to trade ~34% decode throughput for ~3.5× larger context on the same hardware.
- **SM86 uses SM90 fast-math binaries**: Minor numerical differences possible vs. native SM86 compilation, but functionally correct for inference.
- **Q4_0 decode is ~66% of bf16 speed**: Every KV read requires a dequantize; this is a property of the current implementation, not a fundamental limit.

## How to Apply (Standalone)

If you want to patch upstream SGLang v0.5.9 without using this fork:

```bash
pip install sglang==0.5.9 sgl-kernel==0.3.21

# Apply patches from patches/sm86/
cd /
patch -p0 < /path/to/patches/sm86/0001-sgl-kernel-sm86-ampere-fallback-to-sm90.binaries.patch
patch -p0 < /path/to/patches/sm86/0002-server_args-skip-cudnn-conv3d-check-for-text-llm.patch
patch -p0 < /path/to/patches/sm86/0003-qwen3_5-skip-unknown-params-for-mixed-awq-checkpoint.patch
```

Or use this fork directly:

```bash
pip install git+https://github.com/cioinside/sglang.git@sm86-ampere-support
```

## Setup for SM86

```bash
pip install sglang[all]==0.5.9
pip install sgl-kernel==0.3.21
pip install flashinfer-python==0.6.3

# Apply patches (or use this fork)
# Then launch:
python -m sglang.launch_server \
  --model-path <your-model> \
  --tp 2 --host 0.0.0.0 --port 8100 \
  --mem-fraction-static 0.95 \
  --quantization awq_marlin \
  --trust-remote-code --dtype half \
  --disable-custom-all-reduce
```

### Q4_0 KV cache (Patch 4) — long context on memory-constrained cards

Same install as above (the Q4_0 implementation is built into the fork). Launch with `--kv-cache-dtype q4_0` and an explicit `--max-total-tokens` between 16 K and 33 K (the full ~59 K auto-size OOMs on 12 GB cards during inference):

```bash
python -m sglang.launch_server \
  --model-path QuantTrio/Qwen3.6-35B-A3B-AWQ \
  --tensor-parallel-size 2 \
  --quantization awq_marlin \
  --dtype bfloat16 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --max-running-requests 1 \
  --context-length 8192 \
  --disable-radix-cache \
  --chunked-prefill-size 1024 \
  --skip-server-warmup \
  --mem-fraction-static 0.95 \
  --kv-cache-dtype q4_0 \
  --max-total-tokens 16000
```

## Patch Files

Individual patches are in [`patches/sm86/`](patches/sm86/) for easy application to any SGLang version.

## Acknowledgments

- **[grevinden/sglang-kt-sm86](https://github.com/grevinden/sglang-kt-sm86)** — Provided the original SM86→SM90 binary fallback approach for `sgl_kernel/load_utils.py` (Patch 1). Their Docker-based build workflow and CI/CD setup for SM86/SM89 GPUs was instrumental in validating the fast-math binary fallback strategy on Ampere hardware.

---

--------------------------------------------------------------------------------

# Original SGLang README

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Documentation</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Join Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Dev Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

## News
- [2026/06] 🔥 The next generation of speculative decoding: DFlash and Spec V2 ([blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)).
- [2026/04] 🔥 DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles ([blog](https://lmsys.org/blog/2026-04-25-deepseek-v4/)).
- [2026/06] SGLang provides day-0 support for latest open models ([Nemotron 3 Ultra](https://lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra/), [Nemotron 3 Super](https://lmsys.org/blog/2026-03-11-run-nvidia-nemotron-3-super/), [Higgs Audio v3 TTS](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)).
- [2026/02] 🔥 Unlocking 25x Inference Performance with SGLang on NVIDIA GB300 NVL72 ([blog](https://lmsys.org/blog/2026-02-20-gb300-inferencex/)).
- [2026/01] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)).
- [2025/12] SGLang provides day-0 support for latest open models ([MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/), [Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/), [Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213), [LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/), [MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)).
- [2025/10] SGLang now runs natively on TPU with the SGLang-Jax backend ([blog](https://lmsys.org/blog/2025-10-29-sglang-jax/)).

<details>
<summary>More</summary>

- [2025/09] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).

- [2025/11] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)).
- [2025/10] PyTorch Conference 2025 SGLang Talk ([slide](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf)).
- [2025/10] SGLang x Nvidia SF Meetup on 10/2 ([recap](https://x.com/lmsysorg/status/1975339501934510231)).
- [2025/08] SGLang provides day-0 support for OpenAI gpt-oss model ([instructions](https://github.com/sgl-project/sglang/issues/8833))
- [2025/06] SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a high-performance serving framework for large language models and multimodal models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark/5090), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 400,000 GPUs worldwide.
- **RL & Post-Training Backbone**: SGLang is a proven rollout backend used for training many frontier models, with native RL integrations and adoption by well-known post-training frameworks such as [**AReaL**](https://github.com/inclusionAI/AReaL), [**Miles**](https://github.com/radixark/miles), [**slime**](https://github.com/THUDM/slime), [**Tunix**](https://github.com/google/tunix), [**verl**](https://github.com/volcengine/verl) and more.

## Getting Started
- [Install SGLang](https://docs.sglang.io/get_started/install.html)
- [Quick Start](https://docs.sglang.io/basic_usage/send_request.html)
- [Backend Tutorial](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [Contribution Guide](https://docs.sglang.io/developer_guide/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [GB200 rack-scale parallelism](https://lmsys.org/blog/2025-09-25-gb200-part-2/), [GB300 long context](https://lmsys.org/blog/2026-02-19-gb300-longctx/).

## Adoption and Sponsorship
SGLang has been deployed at large scale, generating trillions of tokens in production each day. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, Modal, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations.
As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 400,000 GPUs worldwide.
SGLang is currently hosted under the non-profit open-source organization [LMSYS](https://lmsys.org/about/).

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us
For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at [sglang@lmsys.org](mailto:sglang@lmsys.org).

Long-term active SGLang contributors are eligible for coding agent sponsorship, such as Cursor, Claude Code, or OpenAI Codex. Email [sglang@lmsys.org](mailto:sglang@lmsys.org) with your most important commits or pull requests.

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
