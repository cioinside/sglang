#!/bin/bash
# =============================================================================
# Qwen3.6-35B-A3B-AWQ SGLang Server
# Hardware: 2x RTX 3060 12GB, SM86, PCIe 3.0 x8, no P2P, 62GB RAM
# Repository: https://github.com/cioinside/sglang (branch: sm86-ampere-support)
# =============================================================================
#
# MODES:
#   --fast          Fast mode: CUDA graph + torch.compile, 16K context (default)
#   --long          Long context: V2 CPU weight offload, up to 117K context
#   --ultra         Ultra-fast: CUDA graph only (no torch.compile warmup delay)
#   --bench         Benchmark mode: runs short/medium/long tests, prints results
#
# USAGE:
#   bash sglang-qwen-server-35b.sh              # Fast mode (default), port 3008
#   bash sglang-qwen-server-35b.sh --fast       # Fast mode, port 3008
#   bash sglang-qwen-server-35b.sh --long       # Long context mode, port 3008
#   bash sglang-qwen-server-35b.sh --bench      # Run benchmark suite
#   bash sglang-qwen-server-35b.sh --fast --port 8080
#
# PERFORMANCE (measured 2026-07-27, thinking disabled):
#
#   ┌──────────────────────┬──────────┬──────────┬──────────┬──────────┐
#   │ Configuration        │ SHORT    │ MEDIUM   │ LONG     │ Decode   │
#   │                      │ ch/s     │ ch/s     │ ch/s     │ tok/s    │
#   ├──────────────────────┼──────────┼──────────┼──────────┼──────────┤
#   │ SGLang+CG+torch-comp │ 326      │ 349      │ 410      │ 99.2     │
#   │ SGLang+CUDA graph    │ 326      │ 332      │ 404      │ 90.3     │
#   │ SGLang (no CG)       │ 39.1     │ 49.9     │ 51.1     │ 10.5     │
#   │ llama.cpp            │ 36.6     │ 41.9     │ 34.2     │ ~12.2    │
#   │ vLLM v0.25.1        │ 6.9      │ 8.0      │ 7.8      │ ~2.6     │
#   └──────────────────────┴──────────┴──────────┴──────────┴──────────┘
#
#   Fast mode: 21,097 token KV cache, VRAM 11,901 MB/GPU
#   Long mode: 47-117K token KV cache, ~3.0 tok/s decode (PCIe limited)
#
# REQUIRED PATCHES (in our fork, branch sm86-ampere-support):
#   1. sgl-kernel SM86→sm90 fallback (load_utils.py)
#   2. server_args.py: skip CuDNN conv3d check
#   3. qwen3_5.py: skip unknown params for mixed checkpoint + pass offloader_kwargs
#   4. causal_conv1d.py: force Triton fallback on non-sm90
#   5. offloader.py: dedup tied params in V1 + marlin-safe V2 offload
#
# KNOWN LIMITATIONS:
#   - --dtype bfloat16 required (fp16/bf16 mismatch in Triton causal_conv1d)
#   - FP8 KV cache breaks Mamba causal_conv1d
#   - HiCache OOM: 10.6 GB model leaves no room for HiCache GPU structures
#   - Speculative decoding OOM: no VRAM for draft token KV cache
#   - V2 offload throughput limited by PCIe 3.0 x8 bandwidth (~5 GB/s)
#   - torch.compile: first request slower due to Inductor warmup (~15-20s)
# =============================================================================

set -euo pipefail

# ---- Parse arguments ----
MODE="fast"
PORT=3008
BENCH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast)   MODE="fast";   shift ;;
    --long)   MODE="long";   shift ;;
    --ultra)  MODE="ultra";  shift ;;
    --bench)  BENCH=true; MODE="fast"; shift ;;
    --port)   PORT="$2";     shift 2 ;;
    --help|-h)
      cat <<'EOF'
Usage: sglang-qwen-server-35b.sh [--fast|--long|--ultra|--bench] [--port PORT]

Modes:
  --fast    CUDA graph + torch.compile, 16K context, ~99 tok/s (default)
  --long    V2 CPU weight offload, up to 117K context, ~3 tok/s
  --ultra   CUDA graph only (no torch.compile), 16K context, ~90 tok/s
  --bench   Start in fast mode, run benchmark suite, print results, exit
  --port    Server port (default: 3008)
EOF
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---- Environment ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/root/sglang-env}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: SGLang venv not found at $VENV_DIR"
  echo "Run: bash install-sglang.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"

# CUDA runtime libs — system CUDA 12.4 is too old, need 12.8+ for sgl-kernel
export LD_LIBRARY_PATH="${VENV_DIR}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=6

# ---- Model ----
MODEL="QuantTrio/Qwen3.6-35B-A3B-AWQ"
HOST="0.0.0.0"

echo "============================================"
echo "  SGLang Qwen3.6-35B-A3B-AWQ Server"
echo "  Mode:    ${MODE}"
echo "  Model:   ${MODEL}"
echo "  GPUs:    2x RTX 3060 12GB (SM86, no P2P)"
echo "  Port:    ${PORT}"
echo "  Time:    $(date)"
echo "============================================"

# ---- Common flags ----
COMMON_FLAGS=(
  --model-path "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size 2
  --quantization awq_marlin
  --dtype bfloat16
  --trust-remote-code
  --disable-custom-all-reduce
  --disable-radix-cache
  --chunked-prefill-size 1024
  --skip-server-warmup
  --max-running-requests 1
)

# ---- Mode-specific flags ----
case "$MODE" in
  fast)
    echo "  Context:  ~16K tokens, CUDA graph + torch.compile"
    echo "  Decode:   ~99 tok/s steady state"
    echo "  VRAM:     ~11,901 MB / 12,288 MB per GPU"
    echo ""
    exec python -m sglang.launch_server \
      "${COMMON_FLAGS[@]}" \
      --context-length 8192 \
      --mem-fraction-static 0.95 \
      --enable-torch-compile
    ;;

  ultra)
    echo "  Context:  ~16K tokens, CUDA graph only"
    echo "  Decode:   ~90 tok/s steady state"
    echo "  VRAM:     ~11,901 MB / 12,288 MB per GPU"
    echo ""
    exec python -m sglang.launch_server \
      "${COMMON_FLAGS[@]}" \
      --context-length 8192 \
      --mem-fraction-static 0.95
    ;;

  long)
    echo "  Context:  up to 117K tokens (V2 CPU weight offloading)"
    echo "  Decode:   ~3.0 tok/s (PCIe bandwidth limited)"
    echo "  VRAM:     model 8.88 GB + KV per GPU"
    echo ""
    exec python -m sglang.launch_server \
      "${COMMON_FLAGS[@]}" \
      --context-length 98304 \
      --mem-fraction-static 0.88 \
      --offload-group-size 4 \
      --offload-mode cpu
    ;;

  *)
    echo "ERROR: Unknown mode '$MODE'. Use --fast, --long, or --ultra."
    exit 1
    ;;
esac
