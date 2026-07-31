import functools
import os

import torch

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _q40_dequant_kernel(
        quant_ptr,
        scale_ptr,
        out_ptr,
        NUM_BLOCKS: tl.constexpr,
        BYTES_PER_B: tl.constexpr,
        OUT_PER_B: tl.constexpr,
        BLOCK: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // NUM_BLOCKS
        block_idx = pid % NUM_BLOCKS

        q_ptr = quant_ptr + b * BYTES_PER_B + block_idx * (BLOCK // 2)
        s_ptr = scale_ptr + b * NUM_BLOCKS + block_idx
        o_ptr = out_ptr + b * OUT_PER_B + block_idx * BLOCK

        n_offs = tl.arange(0, BLOCK)
        byte_idx = n_offs // 2
        is_high = (n_offs % 2) != 0

        packed = tl.load(q_ptr + byte_idx)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        q = tl.where(is_high, high, low)

        scale = tl.load(s_ptr).to(OUT_DTYPE)
        q_dt = q.to(OUT_DTYPE)
        out_val = (q_dt - 8.0) * scale

        tl.store(o_ptr + n_offs, out_val)

    _TRITON_AVAILABLE = True

    @triton.jit
    def _q40_quant_kernel(
        in_ptr,
        packed_ptr,
        scale_ptr,
        NUM_BLOCKS: tl.constexpr,
        ELEMENTS_PER_B: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // NUM_BLOCKS
        blk = pid % NUM_BLOCKS

        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + b * ELEMENTS_PER_B + blk * BLOCK + offs).to(tl.float32)

        block_max = tl.maximum(tl.max(tl.abs(x)), 1e-6)
        scale = block_max / 7.5

        scaled = x / scale
        # Round half away from zero via floor/csel: scaled + 0.5 then floor
        # then subtract 1 if negative after rounding... simplest: floor(x+0.5) for x >= 0, ceil(x-0.5) for x < 0
        sign = tl.where(scaled >= 0.0, 1.0, -1.0)
        rounded_mag = tl.floor(tl.abs(scaled) + 0.5)
        q_f = sign * rounded_mag
        q = tl.minimum(tl.maximum(q_f + 8.0, 0.0), 15.0).to(tl.int32)

        # Pack 32 int4 values into 16 uint8: byte i = q[2i] | (q[2i+1] << 4).
        # Triton does not support q.reshape(BLOCK//2, 2)[:, 0] indexing, so use
        # element-wise tl.where to mask low/high nibbles into their byte slots.
        low_mask = tl.where((offs % 2) == 0, q, 0)
        high_mask = tl.where((offs % 2) == 1, q << 4, 0)
        packed = low_mask | high_mask

        packed_offs = tl.arange(0, BLOCK // 2)
        tl.store(
            packed_ptr + b * (ELEMENTS_PER_B // 2) + blk * (BLOCK // 2) + packed_offs,
            packed.to(tl.uint8),
        )
        tl.store(scale_ptr + b * NUM_BLOCKS + blk, scale.to(tl.float16))
except Exception:
    _TRITON_AVAILABLE = False
    _q40_dequant_kernel = None
    _q40_quant_kernel = None


_HADAMARD_SIZES = (32, 64, 128, 256, 512)


@functools.lru_cache(maxsize=16)
def _get_hadamard(n: int, device_str: str, dtype):
    H = torch.ones((1, 1), dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    H = H / (n ** 0.5)
    return H.to(device=torch.device(device_str), dtype=dtype)


def _hadamard_enabled() -> bool:
    return os.environ.get("SGLANG_Q40_HADAMARD", "0") == "1"


def _maybe_rotate(tensor: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation to the last dim of `tensor` if enabled.

    Spreads outlier mass uniformly across the row, so block-wise Q4_0 quantization
    preserves quality. Only applied when SGLANG_Q40_HADAMARD=1 and last dim is a
    power-of-two >= 32 (so a Sylvester-construction Hadamard fits exactly).

    Caller contract (asymmetric, matches llama.cpp llama-kv-cache.cpp:319-339):
    - quantize rotates forward: K -> K @ H, then quantized K @ H is stored.
    - dequantize returns K @ H unchanged (NO inverse rotation here).
    - The attention forward must apply H to Q to keep scores equivalent:
      attention(Q @ H, K @ H) == attention(Q, K) since H is orthogonal.
    """
    n = tensor.shape[-1]
    if (
        _hadamard_enabled()
        and n in _HADAMARD_SIZES
        and (n & (n - 1)) == 0
        and n >= 32
    ):
        H = _get_hadamard(n, str(tensor.device), tensor.dtype)
        return torch.matmul(tensor, H.t())
    return tensor


class Q40KVQuantizeUtil:
    BLOCK_SIZE = 32

    @staticmethod
    def batched_quantize_into(tensor, packed_out, scales_out):
        """Quantize K/V to PRE-ALLOCATED output buffers (CUDA graph friendly).

        Unlike batched_quantize, this does NOT allocate new tensors — it writes
        into the provided buffers. Safe to call from inside a CUDA graph capture
        (which forbids torch.empty).

        Args:
            tensor: (b, m, n) input tensor (fp16/bf16)
            packed_out: (b, m, n // 2) pre-allocated uint8 output buffer
            scales_out: (b, num_blocks) pre-allocated fp16 output buffer
        """
        b, m, n = tensor.shape
        tensor = _maybe_rotate(tensor)
        num_blocks = m * n // Q40KVQuantizeUtil.BLOCK_SIZE
        elements_per_b = m * n
        _q40_quant_kernel[(b * num_blocks,)](
            tensor, packed_out, scales_out,
            NUM_BLOCKS=num_blocks,
            ELEMENTS_PER_B=elements_per_b,
            BLOCK=Q40KVQuantizeUtil.BLOCK_SIZE,
            num_warps=1,
        )
        return packed_out, scales_out, (b, m, n)

    @staticmethod
    def batched_quantize(tensor):
        b, m, n = tensor.shape
        tensor = _maybe_rotate(tensor)
        num_blocks = m * n // Q40KVQuantizeUtil.BLOCK_SIZE
        elements_per_b = m * n

        if _TRITON_AVAILABLE and tensor.is_contiguous() and n == elements_per_b // m and False:
            # Triton kernel for Q4_0 quantize exists but is not activated by default:
            #   - Triton 3.5 lacks several APIs needed (tl.round, fp32->int32 rtne cast)
            #   - Workaround for pack uses tl.where with 32-element broadcast, breaks
            #     against 16-element store buffer.
            # To re-enable: fix the kernel for the current Triton version, then
            # also pre-allocate packed/scales buffers (torch.empty is forbidden
            # inside CUDA graph capture).
            try:
                packed = torch.empty(b, m, n // 2, dtype=torch.uint8, device=tensor.device)
                scales = torch.empty(b, num_blocks, dtype=torch.float16, device=tensor.device)
                grid = (b * num_blocks,)
                _q40_quant_kernel[grid](
                    tensor, packed, scales,
                    NUM_BLOCKS=num_blocks,
                    ELEMENTS_PER_B=elements_per_b,
                    BLOCK=Q40KVQuantizeUtil.BLOCK_SIZE,
                    num_warps=1,
                )
                return packed, scales, (b, m, n)
            except Exception:
                pass

        reshaped = tensor.view(b, num_blocks, Q40KVQuantizeUtil.BLOCK_SIZE)
        abs_vals = reshaped.abs()
        block_max = abs_vals.max(dim=-1, keepdim=True).values
        eps = 1e-6
        scale = torch.clamp(block_max / 7.5, min=eps)
        scaled = reshaped / scale
        q = torch.round(scaled).to(torch.int32)
        q = q + 8
        q = torch.clamp(q, 0, 15)
        q = q.view(b, m, n)
        packed = ((q[:, :, 1::2] << 4) | q[:, :, 0::2]).to(torch.uint8)
        scale_factors = scale.squeeze(-1).to(torch.float16)
        return packed, scale_factors, (b, m, n)

    @staticmethod
    def batched_dequantize(quant_tensor, scale_factors, orig_shape, dtype=torch.bfloat16):
        b, m, n = orig_shape
        num_blocks = m * n // Q40KVQuantizeUtil.BLOCK_SIZE

        if _TRITON_AVAILABLE and dtype in (torch.bfloat16, torch.float16):
            try:
                out = torch.empty(b, m, n, dtype=dtype, device=quant_tensor.device)
                grid = (b * num_blocks,)
                _q40_dequant_kernel[grid](
                    quant_tensor, scale_factors, out,
                    num_blocks, m * n // 2, m * n,
                    BLOCK=Q40KVQuantizeUtil.BLOCK_SIZE,
                    OUT_DTYPE=dtype,
                    num_warps=1,
                )
                return out
            except Exception:
                pass

        BLOCK_SIZE = Q40KVQuantizeUtil.BLOCK_SIZE
        unpacked = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)
        unpacked[:, :, 0::2] = quant_tensor & 0x0F
        unpacked[:, :, 1::2] = (quant_tensor >> 4) & 0x0F
        reshaped_dt = unpacked.view(b, num_blocks, BLOCK_SIZE).to(dtype)
        scale = scale_factors.view(b, num_blocks, 1).to(dtype)
        reshaped_dt.sub_(8.0).mul_(scale)
        return reshaped_dt.view(b, m, n)