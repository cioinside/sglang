# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Triton fused Q4_0 KV store.

Quantizes K and V to Q4_0 format and stores in KV cache.

Q4_0 format:
- Block of 32 values
- Scale: fp16 (2 bytes)
- Quantized data: 32 × 4 bits = 16 bytes
- Total: 18 bytes per block

Quantization: q[i] = round(x / scale) + 8, clamped to [0, 15]
"""

import math

import torch

from sglang.srt.utils import next_power_of_2

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _q40_quantize_and_store(
        Key_ptr,  # [N, H, D] float16
        Value_ptr,  # [N, H, D] float16
        KV_cache_ptr,  # [num_blocks, block_size, Hk, slot_size] uint8
        Slot_mapping_ptr,  # [N] int32
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        D: tl.constexpr,
        H: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        Q40_BLOCK_SIZE: tl.constexpr,
        Q40_BYTES_PER_BLOCK: tl.constexpr,
        NUM_Q40_BLOCKS: tl.constexpr,
        SLOT_SIZE: tl.constexpr,
        K_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        token_idx = pid // H
        head_idx = pid % H

        slot = tl.load(Slot_mapping_ptr + token_idx)
        if slot < 0:
            return

        blk = (slot // BLOCK_SIZE).to(tl.int64)
        off = (slot % BLOCK_SIZE).to(tl.int64)
        head_idx_i64 = tl.cast(head_idx, tl.int64)
        slot_base = (
            blk * stride_cache_block
            + off * stride_cache_pos
            + head_idx_i64 * stride_cache_head
        )

        base = pid * D
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # Load K values
        k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

        # Quantize K to Q4_0
        for q40_blk in range(NUM_Q40_BLOCKS):
            blk_mask = (d_offs // Q40_BLOCK_SIZE == q40_blk) & d_mask

            # Load values for this block
            blk_vals = tl.load(Key_ptr + base + d_offs, mask=blk_mask, other=0.0).to(tl.float32)

            # Compute scale (max absolute value / 15)
            blk_max = tl.max(tl.abs(tl.where(blk_mask, blk_vals, 0.0)))
            scale = blk_max / 15.0
            scale = tl.maximum(scale, 1e-8)

            # Quantize: q[i] = round(x / scale) + 8, clamped to [0, 15]
            q_vals = tl.minimum(tl.maximum(tl.round(blk_vals / scale) + 8, 0), 15).to(tl.uint8)

            # Pack 4-bit values (2 per byte)
            blk_offs = d_offs % Q40_BLOCK_SIZE
            byte_idx = blk_offs // 2
            value_idx = blk_offs % 2

            # Pack low and high nibbles
            packed = tl.zeros([BLOCK_D // 2], dtype=tl.uint8)
            for i in range(0, BLOCK_D, 2):
                if i < D:
                    low = tl.load(Key_ptr + base + i, mask=i < D, other=0.0).to(tl.float32)
                    high = tl.load(Key_ptr + base + i + 1, mask=i + 1 < D, other=0.0).to(tl.float32)

                    q_low = tl.minimum(tl.maximum(tl.round(low / scale) + 8, 0), 15).to(tl.uint8)
                    q_high = tl.minimum(tl.maximum(tl.round(high / scale) + 8, 0), 15).to(tl.uint8)

                    packed_val = (q_low & 0xF) | ((q_high & 0xF) << 4)
                    packed[i // 2] = packed_val

            # Store scale (fp16)
            blk_base = q40_blk * Q40_BYTES_PER_BLOCK
            scale_f16 = scale.to(tl.float16)
            scale_u16 = scale_f16.to(tl.uint16, bitcast=True)
            tl.store(KV_cache_ptr + slot_base + blk_base, (scale_u16 & 0xFF).to(tl.uint8))
            tl.store(KV_cache_ptr + slot_base + blk_base + 1, ((scale_u16 >> 8) & 0xFF).to(tl.uint8))

            # Store packed data
            data_base = slot_base + blk_base + 2
            for i in range(Q40_BYTES_PER_BLOCK - 2):
                if i < BLOCK_D // 2:
                    tl.store(KV_cache_ptr + data_base + i, packed[i])

        # Load V values
        v_vals = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

        # Quantize V to Q4_0 (same process as K)
        for q40_blk in range(NUM_Q40_BLOCKS):
            blk_mask = (d_offs // Q40_BLOCK_SIZE == q40_blk) & d_mask

            blk_vals = tl.load(Value_ptr + base + d_offs, mask=blk_mask, other=0.0).to(tl.float32)

            blk_max = tl.max(tl.abs(tl.where(blk_mask, blk_vals, 0.0)))
            scale = blk_max / 15.0
            scale = tl.maximum(scale, 1e-8)

            q_vals = tl.minimum(tl.maximum(tl.round(blk_vals / scale) + 8, 0), 15).to(tl.uint8)

            packed = tl.zeros([BLOCK_D // 2], dtype=tl.uint8)
            for i in range(0, BLOCK_D, 2):
                if i < D:
                    low = tl.load(Value_ptr + base + i, mask=i < D, other=0.0).to(tl.float32)
                    high = tl.load(Value_ptr + base + i + 1, mask=i + 1 < D, other=0.0).to(tl.float32)

                    q_low = tl.minimum(tl.maximum(tl.round(low / scale) + 8, 0), 15).to(tl.uint8)
                    q_high = tl.minimum(tl.maximum(tl.round(high / scale) + 8, 0), 15).to(tl.uint8)

                    packed_val = (q_low & 0xF) | ((q_high & 0xF) << 4)
                    packed[i // 2] = packed_val

            blk_base = K_SIZE + q40_blk * Q40_BYTES_PER_BLOCK
            scale_f16 = scale.to(tl.float16)
            scale_u16 = scale_f16.to(tl.uint16, bitcast=True)
            tl.store(KV_cache_ptr + slot_base + blk_base, (scale_u16 & 0xFF).to(tl.uint8))
            tl.store(KV_cache_ptr + slot_base + blk_base + 1, ((scale_u16 >> 8) & 0xFF).to(tl.uint8))

            data_base = slot_base + blk_base + 2
            for i in range(Q40_BYTES_PER_BLOCK - 2):
                if i < BLOCK_D // 2:
                    tl.store(KV_cache_ptr + data_base + i, packed[i])


def q40_store_kv(
    key: torch.Tensor,  # [N, H, D] float16
    value: torch.Tensor,  # [N, H, D] float16
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, slot_size] uint8
    slot_mapping: torch.Tensor,  # [N] int32
    head_dim: int,
    block_size: int,
):
    """Quantize K and V to Q4_0 and store in KV cache."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for Q4_0 store")

    N, H, D = key.shape
    NH = N * H
    BLOCK_D = next_power_of_2(D)

    # Q4_0 layout constants
    Q40_BLOCK_SIZE = 32
    Q40_BYTES_PER_BLOCK = 18
    NUM_Q40_BLOCKS = D // Q40_BLOCK_SIZE
    K_SIZE = NUM_Q40_BLOCKS * Q40_BYTES_PER_BLOCK
    SLOT_SIZE = 2 * K_SIZE  # K + V

    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    grid = (NH,)
    _q40_quantize_and_store[grid](
        key,
        value,
        kv_cache,
        slot_mapping,
        stride_block,
        stride_pos,
        stride_head,
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        Q40_BLOCK_SIZE=Q40_BLOCK_SIZE,
        Q40_BYTES_PER_BLOCK=Q40_BYTES_PER_BLOCK,
        NUM_Q40_BLOCKS=NUM_Q40_BLOCKS,
        SLOT_SIZE=SLOT_SIZE,
        K_SIZE=K_SIZE,
        num_warps=4,
        num_stages=1,
    )
