# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Triton fused Q4_0 decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits).

Q4_0 format (like llama.cpp):
- Block of 32 values
- Scale: fp16 (2 bytes)
- Quantized data: 32 × 4 bits = 16 bytes
- Total: 18 bytes per block

Dequantization: x = (q[i] - 8) * scale
"""

import math
from typing import Any

import torch

from sglang.srt.utils import next_power_of_2

# Try to import triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    # ---------------------------------------------------------------------------
    # Stage 1: Fused Q4_0 score + value accumulation (BLOCK_KV tiled)
    # ---------------------------------------------------------------------------

    @triton.jit
    def _q40_decode_stage1(
        # Precomputed query projection
        Q_ptr,  # [B, Hq, D] float16
        # Compressed KV cache (combined K+V)
        KV_cache_ptr,  # [num_blocks, block_size, Hk, slot_size] uint8
        # Block table and sequence info
        Block_table_ptr,  # [B, max_num_blocks] int32
        Seq_lens_ptr,  # [B] int32
        # Output (intermediate for stage2)
        Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
        # Strides
        stride_qb,
        stride_qh,  # Q strides: [B, Hq, D]
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,  # KV cache
        stride_bt_b,  # block_table stride per batch
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,  # mid_o strides
        # Constexpr dims
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,  # KV cache block_size (pages)
        NUM_KV_SPLITS: tl.constexpr,
        KV_GROUP_SIZE: tl.constexpr,  # Hq // Hk
        # Q4_0 layout constants
        Q40_BLOCK_SIZE: tl.constexpr,  # 32 (values per Q4_0 block)
        Q40_BYTES_PER_BLOCK: tl.constexpr,  # 18 (16 data + 2 scale)
        NUM_Q40_BLOCKS: tl.constexpr,  # HEAD_DIM / Q40_BLOCK_SIZE
        SLOT_SIZE: tl.constexpr,  # total bytes per head per position
        K_SIZE: tl.constexpr,  # bytes for K (NUM_Q40_BLOCKS * Q40_BYTES_PER_BLOCK)
        # Score constants
        ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
        # Block tile sizes
        BLOCK_D: tl.constexpr,  # next_power_of_2(HEAD_DIM)
        BLOCK_KV: tl.constexpr,  # tokens per tile (16)
    ):
        bid = tl.program_id(0)  # batch index
        hid = tl.program_id(1)  # q_head index
        sid = tl.program_id(2)  # kv_split index

        kv_head = hid // KV_GROUP_SIZE

        # Sequence length for this batch
        seq_len = tl.load(Seq_lens_ptr + bid)

        # KV split range
        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = split_len * sid
        split_end = tl.minimum(split_start + split_len, seq_len)

        if split_start >= split_end:
            return

        # Dimension offsets
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        kv_range = tl.arange(0, BLOCK_KV)

        # Load query vector: [BLOCK_D] float16 -> float32
        q_base = bid * stride_qb + hid * stride_qh
        q = tl.load(Q_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

        # Online softmax accumulators
        m_prev = -float("inf")
        l_prev = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        bt_base = bid * stride_bt_b

        # Precompute Q4_0 unpacking indices
        # For each dimension d, we need:
        # - Which Q4_0 block: d // 32
        # - Which byte within block data: (d % 32) // 2
        # - Which 4-bit value: (d % 32) % 2
        q40_block_idx = d_offs // Q40_BLOCK_SIZE
        byte_idx_in_block = (d_offs % Q40_BLOCK_SIZE) // 2
        value_idx_in_byte = (d_offs % Q40_BLOCK_SIZE) % 2

        # ================================================================
        # TILED LOOP: process BLOCK_KV tokens per iteration
        # ================================================================
        for start_n in range(split_start, split_end, BLOCK_KV):
            kv_offs = start_n + kv_range
            kv_mask = kv_offs < split_end

            page_idx = kv_offs // BLOCK_SIZE
            page_off = kv_offs % BLOCK_SIZE
            block_nums = tl.load(
                Block_table_ptr + bt_base + page_idx,
                mask=kv_mask,
                other=0,
            ).to(tl.int64)

            slot_bases = (
                block_nums * stride_cache_block
                + page_off.to(tl.int64) * stride_cache_pos
                + tl.cast(kv_head, tl.int64) * stride_cache_head
            )

            # ============================================================
            # COMPUTE ATTENTION SCORES: [BLOCK_KV]
            # ============================================================
            # Load and dequantize K from Q4_0 format
            # For each token in kv_range, load Q4_0 blocks and dequantize

            # Initialize K values
            k_float = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)

            # Process each Q4_0 block
            for q40_blk in range(NUM_Q40_BLOCKS):
                # Mask for dimensions in this Q4_0 block
                blk_mask = (q40_block_idx == q40_blk) & d_mask

                # Base offset for this Q4_0 block
                blk_base = q40_blk * Q40_BYTES_PER_BLOCK

                # Load scale (fp16 -> float32) for this block
                scale_addrs = slot_bases + blk_base
                scale_lo = tl.load(KV_cache_ptr + scale_addrs, mask=kv_mask, other=0).to(tl.uint16)
                scale_hi = tl.load(KV_cache_ptr + scale_addrs + 1, mask=kv_mask, other=0).to(tl.uint16)
                scale_u16 = scale_lo | (scale_hi << 8)
                scale = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)

                # Load quantized data (16 bytes per block)
                data_base = slot_bases + blk_base + 2  # skip scale
                byte_addrs = data_base + byte_idx_in_block
                raw_bytes = tl.load(
                    KV_cache_ptr + byte_addrs,
                    mask=kv_mask[:, None] & blk_mask[None, :],
                    other=0,
                ).to(tl.uint8)

                # Unpack 4-bit values
                # value_idx_in_byte: 0 = low nibble, 1 = high nibble
                shift = value_idx_in_byte * 4
                q4_vals = ((raw_bytes >> shift) & 0xF).to(tl.float32)

                # Dequantize: x = (q[i] - 8) * scale
                k_dequant = (q4_vals - 8.0) * scale[:, None]

                # Store in k_float
                k_float = tl.where(blk_mask[None, :], k_dequant, k_float)

            # Compute attention scores
            scores = (
                tl.sum(
                    tl.where(d_mask[None, :], q[None, :] * k_float, 0.0),
                    axis=1,
                )
                * ATTN_SCALE
            )
            scores = tl.where(kv_mask, scores, -float("inf"))

            # ============================================================
            # ONLINE SOFTMAX UPDATE (block-level)
            # ============================================================
            n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
            re_scale = tl.exp(m_prev - n_e_max)
            p = tl.exp(scores - n_e_max)

            # ============================================================
            # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
            # ============================================================
            # V is stored after K in the slot
            # Same Q4_0 format as K

            # Initialize V values
            v_float = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)

            # Process each Q4_0 block for V
            for q40_blk in range(NUM_Q40_BLOCKS):
                # Mask for dimensions in this Q4_0 block
                blk_mask = (q40_block_idx == q40_blk) & d_mask

                # Base offset for this Q4_0 block (V starts after K)
                blk_base = K_SIZE + q40_blk * Q40_BYTES_PER_BLOCK

                # Load scale (fp16 -> float32) for this block
                scale_addrs = slot_bases + blk_base
                scale_lo = tl.load(KV_cache_ptr + scale_addrs, mask=kv_mask, other=0).to(tl.uint16)
                scale_hi = tl.load(KV_cache_ptr + scale_addrs + 1, mask=kv_mask, other=0).to(tl.uint16)
                scale_u16 = scale_lo | (scale_hi << 8)
                scale = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)

                # Load quantized data (16 bytes per block)
                data_base = slot_bases + blk_base + 2  # skip scale
                byte_addrs = data_base + byte_idx_in_block
                raw_bytes = tl.load(
                    KV_cache_ptr + byte_addrs,
                    mask=kv_mask[:, None] & blk_mask[None, :],
                    other=0,
                ).to(tl.uint8)

                # Unpack 4-bit values
                shift = value_idx_in_byte * 4
                q4_vals = ((raw_bytes >> shift) & 0xF).to(tl.float32)

                # Dequantize: x = (q[i] - 8) * scale
                v_dequant = (q4_vals - 8.0) * scale[:, None]

                # Store in v_float
                v_float = tl.where(blk_mask[None, :], v_dequant, v_float)

            # ============================================================
            # WEIGHTED VALUE ACCUMULATION
            # ============================================================
            acc = acc * re_scale + tl.sum(p[:, None] * v_float, 0)
            l_prev = l_prev * re_scale + tl.sum(p, 0)
            m_prev = n_e_max

        # Store partial result
        out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
        safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
        tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
        lse = m_prev + tl.log(safe_l)
        tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


    # ---------------------------------------------------------------------------
    # Stage 2: Reuse from triton_decode_attention.py
    # ---------------------------------------------------------------------------

    @triton.jit
    def _fwd_kernel_stage2(
        Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
        Out_ptr,  # [B, Hq, D] float16
        Lse_ptr,  # [B, Hq] float32
        Seq_lens_ptr,  # [B] int32
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        NUM_KV_SPLITS: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        Lv: tl.constexpr,
        OUTPUT_FP16: tl.constexpr,
    ):
        bid = tl.program_id(0)
        hid = tl.program_id(1)

        seq_len = tl.load(Seq_lens_ptr + bid)
        if seq_len == 0:
            return

        d_offs = tl.arange(0, BLOCK_DV)
        d_mask = d_offs < Lv

        m_prev = -float("inf")
        acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
        l_prev = 0.0

        for s in range(NUM_KV_SPLITS):
            split_start = s * tl.cdiv(seq_len, NUM_KV_SPLITS)
            split_end = tl.minimum(split_start + tl.cdiv(seq_len, NUM_KV_SPLITS), seq_len)
            if split_start >= split_end:
                continue

            mid_base = bid * stride_mid_b + hid * stride_mid_h + s * stride_mid_s
            o_scale = tl.exp(m_prev - tl.maximum(m_prev, tl.load(Mid_o_ptr + mid_base + Lv)))
            acc = acc * o_scale + tl.load(Mid_o_ptr + mid_base + d_offs, mask=d_mask, other=0.0)
            l = o_scale * l_prev + tl.load(Mid_o_ptr + mid_base + Lv)
            l_prev = l
            m_prev = tl.maximum(m_prev, tl.load(Mid_o_ptr + mid_base + Lv))

        safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
        out = acc / safe_l

        out_base = bid * stride_out_b + hid * stride_out_h
        if OUTPUT_FP16:
            tl.store(Out_ptr + out_base + d_offs, out.to(tl.float16), mask=d_mask)
        else:
            tl.store(Out_ptr + out_base + d_offs, out, mask=d_mask)

        lse_base = bid * stride_lse_b + hid * stride_lse_h
        tl.store(Lse_ptr + lse_base, m_prev + tl.log(safe_l))


def q40_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] float16
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, slot_size] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    scale: float,
    head_dim: int,
    block_size: int,
    max_num_kv_splits: int = 32,
) -> torch.Tensor:
    """Launch fused Q4_0 decode attention.

    Returns: output tensor [B, Hq, D] in float16.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for Q4_0 decode attention")

    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    slot_size = kv_cache.shape[3]
    kv_group_size = Hq // Hk
    device = query.device

    NUM_KV_SPLITS = max_num_kv_splits
    BLOCK_D = next_power_of_2(D)
    BLOCK_KV = 16

    # Q4_0 layout constants
    Q40_BLOCK_SIZE = 32  # values per Q4_0 block
    Q40_BYTES_PER_BLOCK = 18  # 16 data + 2 scale
    NUM_Q40_BLOCKS = D // Q40_BLOCK_SIZE  # number of Q4_0 blocks per head

    # K size in bytes
    K_SIZE = NUM_Q40_BLOCKS * Q40_BYTES_PER_BLOCK

    # Allocate intermediate buffers
    mid_o = torch.empty(
        B, Hq, NUM_KV_SPLITS, D + 1,
        dtype=torch.float32, device=device,
    )

    # Stage 1: split-KV tiled attention scoring + value accumulation
    grid = (B, Hq, NUM_KV_SPLITS)
    _q40_decode_stage1[grid](
        query,
        kv_cache,
        block_table,
        seq_lens,
        mid_o,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        Q40_BLOCK_SIZE=Q40_BLOCK_SIZE,
        Q40_BYTES_PER_BLOCK=Q40_BYTES_PER_BLOCK,
        NUM_Q40_BLOCKS=NUM_Q40_BLOCKS,
        SLOT_SIZE=slot_size,
        K_SIZE=K_SIZE,
        ATTN_SCALE=scale,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        num_warps=1,
        num_stages=1,
    )

    # Stage 2: Reduce across KV splits
    output = torch.empty(B, Hq, D, dtype=torch.float16, device=device)
    lse = torch.empty(B, Hq, dtype=torch.float32, device=device)

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        lse.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_D,
        Lv=D,
        OUTPUT_FP16=1,
        num_warps=4,
        num_stages=2,
    )

    return output
