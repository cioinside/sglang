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
        scale = torch.clamp(block_max / 7.5, min=eps)
        scaled = reshaped / scale
        q = torch.round(scaled).to(torch.int32)
        q = q + 8
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
        num_blocks = m * n // Q40KVQuantizeUtil.BLOCK_SIZE
        BLOCK_SIZE = Q40KVQuantizeUtil.BLOCK_SIZE
        unpacked = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)
        unpacked[:, :, 0::2] = quant_tensor & 0x0F
        unpacked[:, :, 1::2] = (quant_tensor >> 4) & 0x0F
        reshaped_dt = unpacked.view(b, num_blocks, BLOCK_SIZE).to(dtype)
        scale = scale_factors.view(b, num_blocks, 1).to(dtype)
        reshaped_dt.sub_(8.0).mul_(scale)
        return reshaped_dt.view(b, m, n)
