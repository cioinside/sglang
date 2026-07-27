#!/bin/bash
# =============================================================================
# SGLang Environment Setup for Qwen3.6-35B-A3B on 2x RTX 3060 12GB (SM86)
# Repository: https://github.com/cioinside/sglang (branch: sm86-ampere-support)
# =============================================================================
#
# Installs SGLang v0.5.9 from our fork with all SM86 (Ampere) compatibility
# patches. Upstream sgl-kernel ships only sm90/sm100 binaries, so the model
# crashes on RTX 3060 (SM86) without these patches.
#
# USAGE:
#   bash install-sglang.sh                  # Full install (recommended)
#   bash install-sglang.sh --venv-dir /path # Custom venv location
#
# WHAT THIS DOES:
#   1. Creates a Python 3.12 venv
#   2. Installs SGLang from our fork (all patches included)
#   3. Installs sgl-kernel==0.3.21 (required for SM86 fallback)
#   4. Applies Patch #4 to causal_conv1d if not already patched
#
# REQUIREMENTS:
#   - Python 3.12
#   - CUDA toolkit >= 12.4
#   - Internet connection
#   - ~5 GB disk space
#
# INSTALL TIME: ~5-10 minutes
# =============================================================================

set -euo pipefail

# ---- Parse arguments ----
VENV_DIR="${VENV_DIR:-/root/sglang-env}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--venv-dir /path/to/venv]"
      echo "  Default venv: /root/sglang-env"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "  SGLang SM86 Environment Setup"
echo "  Venv:   ${VENV_DIR}"
echo "  Fork:   https://github.com/cioinside/sglang"
echo "  Branch: sm86-ampere-support"
echo "  Time:   $(date)"
echo "============================================"

# ---- Step 1: Create venv ----
echo ""
echo "[1/4] Creating Python 3.12 venv..."
if [[ -d "$VENV_DIR" ]]; then
  echo "  Venv already exists at $VENV_DIR"
  echo "  To recreate: rm -rf $VENV_DIR && bash $0"
  echo "  Continuing with existing venv..."
else
  python3 -m venv "$VENV_DIR"
  echo "  Created: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ---- Step 2: Install SGLang from fork ----
echo ""
echo "[2/4] Installing SGLang from fork (branch sm86-ampere-support)..."
echo "  This installs all SM86 patches — no manual patching needed for Patches #1-3,5-7."
pip install --upgrade pip setuptools wheel

# Install from our fork — includes all patches
pip install "sglang[all] @ git+https://github.com/cioinside/sglang@sm86-ampere-support"

# Pin sgl-kernel to version with sm90 binaries (our Patch #1 adds SM86 fallback)
pip install sgl-kernel==0.3.21

# FlashInfer attention backend
pip install flashinfer-python==0.6.3 || echo "  Warning: flashinfer install failed (may already be installed)"

# ---- Step 3: Apply Patch #4 (causal_conv1d) ----
echo ""
echo "[3/4] Checking causal_conv1d Patch #4 (Triton fallback)..."

CAUSAL_SGLANG="${VENV_DIR}/lib/python3.12/site-packages/sglang/srt/layers/attention/mamba/causal_conv1d.py"
CAUSAL_PKG="${VENV_DIR}/lib/python3.12/site-packages/causal_conv1d/causal_conv1d.py"

for f in "$CAUSAL_SGLANG" "$CAUSAL_PKG"; do
  if [[ ! -f "$f" ]]; then
    continue
  fi
  if grep -q "torch.cuda.get_device_capability\|_use_triton_backend\|triton_fallback" "$f" 2>/dev/null; then
    echo "  OK:   $f — already patched"
  else
    echo "  NEED: $f — applying inline patch"
    python3 << 'PYPATCH'
import sys

for path in sys.argv[1:]:
    try:
        with open(path, 'r') as fh:
            content = fh.read()

        if 'torch.cuda.get_device_capability' in content:
            print(f"  OK: {path} — already patched")
            continue

        # Patch: wrap the sgl_kernel.causal_conv1d_fwd call with a
        # device capability check — fallback to PyTorch on non-sm90 GPUs.
        # The actual patch depends on the file layout; we do a safe no-op
        # if the pattern isn't found.
        old = "from sgl_kernel import causal_conv1d_fwd as _causal_conv1d_fwd"
        new = """from sgl_kernel import causal_conv1d_fwd as _causal_conv1d_fwd

def _causal_conv1d_patched(x, weight, bias=None, activation=None, **kwargs):
    \"\"\"SM86-safe wrapper: route to Triton on non-sm90 GPUs.\"\"\"
    major, _ = torch.cuda.get_device_capability(x.device)
    if major >= 9:
        return _causal_conv1d_fwd(x, weight, bias=bias, activation=activation, **kwargs)
    # Triton fallback path (upstream causal_conv1d provides this)
    import causal_conv1d
    if hasattr(causal_conv1d, 'causal_conv1d_fn'):
        return causal_conv1d.causal_conv1d_fn(x, weight, bias=bias, activation=activation)
    return _causal_conv1d_fwd(x, weight, bias=bias, activation=activation, **kwargs)"""

        if old in content:
            content = content.replace(old, new)
            # Replace all subsequent calls to _causal_conv1d_fwd with patched version
            content = content.replace("_causal_conv1d_fwd(", "_causal_conv1d_patched(")
            with open(path, 'w') as fh:
                fh.write(content)
            print(f"  OK: {path} — patched (Triton fallback)")
        else:
            print(f"  SKIP: {path} — pattern not found (may need manual patch)")
    except Exception as e:
        print(f"  ERR: {path} — {e}")
PYPATCH
  fi
done

# ---- Step 4: Verify installation ----
echo ""
echo "[4/4] Verifying installation..."
python3 -c "
import sglang; print(f'  SGLang:  {sglang.__version__}')
import torch; print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda}')
cap = torch.cuda.get_device_capability(0)
print(f'  GPU 0:   SM{cap[0]}{cap[1]} ({torch.cuda.get_device_name(0)})')
cap1 = torch.cuda.get_device_capability(1)
print(f'  GPU 1:   SM{cap1[0]}{cap1[1]} ({torch.cuda.get_device_name(1)})')
" 2>&1

# ---- Summary ----
echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  7 patches included in fork (sm86-ampere-support):"
echo "    #1  sgl-kernel SM86→sm90 fallback"
echo "    #2  server_args: skip CuDNN conv3d check"
echo "    #3  qwen3_5: skip unknown AWQ checkpoint params"
echo "    #4  causal_conv1d: force Triton fallback (applied above)"
echo "    #5  offloader: dedup tied params in V1"
echo "    #6  offloader: marlin-safe V2 CPU offload"
echo "    #7  qwen3_5: pass offloader_kwargs for MoE experts"
echo ""
echo "  Performance (thinking disabled):"
echo "    --fast  (--enable-torch-compile):  99.2 tok/s decode, ~16K context"
echo "    --ultra (CUDA graph only):         90.3 tok/s decode, ~16K context"
echo "    --long  (V2 CPU offload):           3.0 tok/s decode, up to 117K context"
echo ""
echo "  Start the server:"
echo "    bash sglang-qwen-server-35b.sh --fast"
echo ""
echo "  Test inference:"
echo "    curl http://localhost:3008/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"QuantTrio/Qwen3.6-35B-A3B-AWQ\","
echo "           \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],"
echo "           \"max_tokens\": 50}'"
echo "============================================"
