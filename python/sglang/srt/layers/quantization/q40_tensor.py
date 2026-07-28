import torch

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
        scale = torch.clamp(block_max / (7.0 + eps), min=eps)
        scaled = reshaped / scale
        q = torch.round(scaled).to(torch.int32)
        q = torch.clamp(q, 0, 15)
        # Reshape back to [B, M, N] before packing
        q = q.view(b, m, n)
        # Pack 2 values per byte: [B, M, N//2]
        packed = ((q[:, :, 1::2] << 4) | q[:, :, 0::2]).to(torch.uint8)
        scale_factors = scale.squeeze(-1).to(torch.float16)
        return packed, scale_factors, (b, m, n)

    @staticmethod
    def batched_dequantize(quant_tensor, scale_factors, orig_shape, dtype=torch.bfloat16):
        b, m, n = orig_shape
        n_half = n // 2
        low = quant_tensor & 0x0F
        high = (quant_tensor >> 4) & 0x0F
        unpacked = torch.stack([low, high], dim=-1)
        unpacked = unpacked.view(b, m, n)
        num_blocks = m * n // Q40KVQuantizeUtil.BLOCK_SIZE
        reshaped = unpacked.view(b, num_blocks, Q40KVQuantizeUtil.BLOCK_SIZE)
        reshaped_dt = reshaped.to(dtype)
        scale = scale_factors.view(b, num_blocks, 1).to(dtype)
        dequantized = (reshaped_dt - 8.0) * scale
        return dequantized.view(b, m, n)
