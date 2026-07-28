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

        scale = tl.load(s_ptr).to(tl.bfloat16)
        q_bf16 = q.to(tl.bfloat16)
        out_val = (q_bf16 - 8.0) * scale

        tl.store(o_ptr + n_offs, out_val)

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False
    _q40_dequant_kernel = None


class Q40KVQuantizeUtil:
    BLOCK_SIZE = 32

    @staticmethod
    def batched_quantize(tensor):
        b, m, n = tensor.shape
        num_blocks = m * n // Q40KVQuantizeUtil.BLOCK_SIZE
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

        if _TRITON_AVAILABLE and dtype == torch.bfloat16:
            try:
                out = torch.empty(b, m, n, dtype=dtype, device=quant_tensor.device)
                grid = (b * num_blocks,)
                _q40_dequant_kernel[grid](
                    quant_tensor, scale_factors, out,
                    num_blocks, m * n // 2, m * n,
                    BLOCK=Q40KVQuantizeUtil.BLOCK_SIZE,
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