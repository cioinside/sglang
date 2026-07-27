# SGLang Deployment for RTX 3060 12GB (SM86)

Deployment scripts for running Qwen3.6-35B-A3B-AWQ on 2x RTX 3060 12GB via SGLang.

## Quick Start

```bash
# 1. Install environment (one-time, ~5-10 min)
bash install-sglang.sh

# 2. Start server
bash sglang-qwen-server-35b.sh --fast

# 3. Test
curl http://localhost:3008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QuantTrio/Qwen3.6-35B-A3B-AWQ",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50,
    "chat_template_kwargs": {"enable_thinking": false}
  }'

# 4. Benchmark
bash sglang-bench.sh
```

## Server Modes

| Mode | Flag | Context | Decode Speed | Notes |
|------|------|---------|-------------|-------|
| **Fast** | `--fast` | ~16K tokens | ~99 tok/s | CUDA graph + torch.compile (default) |
| **Ultra** | `--ultra` | ~16K tokens | ~90 tok/s | CUDA graph only, no Inductor warmup |
| **Long** | `--long` | up to 117K tokens | ~3 tok/s | V2 CPU weight offload, PCIe limited |

## Performance

Measured 2026-07-27, thinking disabled:

| Configuration | SHORT (ch/s) | MEDIUM (ch/s) | LONG (ch/s) | Decode (tok/s) |
|---|---|---|---|---|
| **SGLang + CG + torch.compile** | **326** | **349** | **410** | **99.2** |
| SGLang + CUDA graph | 326 | 332 | 404 | 90.3 |
| SGLang (no CG) | 39.1 | 49.9 | 51.1 | 10.5 |
| llama.cpp | 36.6 | 41.9 | 34.2 | ~12.2 |
| vLLM v0.25.1 | 6.9 | 8.0 | 7.8 | ~2.6 |

CUDA graph alone provides **8.5x** speedup (10.5 → 90.3 tok/s).
torch.compile adds another **10%** (90.3 → 99.2 tok/s).

## Hardware Requirements

- 2x NVIDIA RTX 3060 12GB (SM86, Ampere)
- PCIe 3.0 x8 slots, no P2P
- 62GB+ system RAM (for V2 CPU offloading)
- CUDA toolkit >= 12.4

## Patches (7 total)

All patches are included in this fork on the `sm86-ampere-support` branch:

1. **sgl-kernel SM86→sm90 fallback** — routes SM86 to sm90 binary path
2. **server_args: skip CuDNN conv3d check** — irrelevant for text-only LLM
3. **qwen3_5: skip unknown AWQ params** — handles mixed .weight/.qweight checkpoints
4. **causal_conv1d: Triton fallback** — avoids dtype assertion crash on SM86
5. **offloader: dedup tied params V1** — fixes ValueError in V1 device_state
6. **offloader: marlin-safe V2 offload** — in-place param swap instead of functional_call
7. **qwen3_5: offloader_kwargs** — enables MoE expert submodule offloading

## Known Limitations

- `--dtype bfloat16` required (fp16/bf16 mismatch in Triton kernel)
- FP8 KV cache breaks Mamba causal_conv1d
- HiCache OOM: 10.6GB model leaves no room for HiCache GPU structures on 12GB cards
- Speculative decoding OOM: no VRAM for draft token KV cache
- V2 offload throughput limited by PCIe 3.0 x8 bandwidth (~5 GB/s)
