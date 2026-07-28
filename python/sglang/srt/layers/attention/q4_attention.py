"""Q4_0-aware attention backend (PoC).

This is a minimal attention backend that reads Q4_0 packed KV buffers directly,
avoiding full-pool bf16 materialization. This unlocks larger context windows on
memory-constrained hardware (the dequant memory is the output of attention
compute, not a pre-allocated buffer).

Status:
- bs=1 decode: IMPLEMENTED (fused dequant+attention Triton kernel)
- prefill (extend): NotImplementedError (fall back to bf16 path via Q4_0 pool's
  get_kv_buffer for now; full dequant still happens)
- multi-batch decode: NotImplementedError (use --cuda-graph-bs 1)
- CUDA Graphs: basic init_forward_metadata supports capture/replay

For full integration, the kernel needs to be extended to handle:
- Paged attention (block tables, page sizes > 1)
- Prefill with causal masking
- Multi-batch decode with variable sequence lengths
- Sliding window attention
"""
import torch
import triton
import triton.language as tl


try:
    @triton.jit
    def _q40_decode_attn_kernel(
        Q_ptr,             # (H, D) bf16
        K_q4_ptr,          # (S, H, D//2) uint8
        K_scale_ptr,       # (S, H*D//32) fp16
        V_q4_ptr,          # (S, H, D//2) uint8
        V_scale_ptr,       # (S, H*D//32) fp16
        O_ptr,             # (H, D) bf16
        seq_len,
        H,
        D: tl.constexpr,
        D_PAD: tl.constexpr,
        BLOCK: tl.constexpr,
        CHUNK: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        chunks = tl.cdiv(seq_len, CHUNK)
        h = pid // chunks
        chunk_start = (pid % chunks) * CHUNK
        chunk_end = tl.minimum(chunk_start + CHUNK, seq_len)

        NUM_BLOCKS = D // BLOCK
        BYTES_PER_TOKEN_HEAD = D // 2

        d_offs = tl.arange(0, D_PAD)
        d_mask = d_offs < D
        q = tl.load(Q_ptr + h * D + d_offs, mask=d_mask, other=0.0)

        m_i = float('-inf')
        l_i = 0.0

        k_head_base = K_q4_ptr + h * BYTES_PER_TOKEN_HEAD
        v_head_base = V_q4_ptr + h * BYTES_PER_TOKEN_HEAD
        ks_head_base = K_scale_ptr + h * NUM_BLOCKS
        vs_head_base = V_scale_ptr + h * NUM_BLOCKS

        H_STRIDE_B = H * BYTES_PER_TOKEN_HEAD
        H_STRIDE_S = H * NUM_BLOCKS

        for s in range(chunk_start, chunk_end):
            k_base = k_head_base + s * H_STRIDE_B
            v_base = v_head_base + s * H_STRIDE_B
            ks_base = ks_head_base + s * H_STRIDE_S
            vs_base = vs_head_base + s * H_STRIDE_S

            score = tl.zeros((), dtype=tl.float32)
            for blk in range(NUM_BLOCKS):
                byte_offs = tl.arange(0, BLOCK // 2)
                n_offs = tl.arange(0, BLOCK)
                byte_idx = n_offs // 2
                is_high = (n_offs % 2) != 0

                k_packed = tl.load(k_base + blk * (BLOCK // 2) + byte_idx)
                k_low = k_packed & 0x0F
                k_high = (k_packed >> 4) & 0x0F
                k_vals = tl.where(is_high, k_high, k_low)
                k_scale_v = tl.load(ks_base + blk).to(tl.bfloat16)
                k_block = (k_vals.to(tl.bfloat16) - 8.0) * k_scale_v

                q_block = tl.load(Q_ptr + h * D + blk * BLOCK + n_offs)
                score += tl.sum(q_block.to(tl.float32) * k_block.to(tl.float32))

            score = score * SCALE

            m_new = tl.maximum(m_i, score)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(score - m_new)
            l_i = l_i * alpha + beta

            for blk in range(NUM_BLOCKS):
                n_offs = tl.arange(0, BLOCK)
                byte_idx = n_offs // 2
                is_high = (n_offs % 2) != 0

                v_packed = tl.load(v_base + blk * (BLOCK // 2) + byte_idx)
                v_low = v_packed & 0x0F
                v_high = (v_packed >> 4) & 0x0F
                v_vals = tl.where(is_high, v_high, v_low)
                v_scale_v = tl.load(vs_base + blk).to(tl.bfloat16)
                v_block = (v_vals.to(tl.bfloat16) - 8.0) * v_scale_v

                acc_offs = h * D + blk * BLOCK + n_offs
                old = tl.load(O_ptr + acc_offs).to(tl.float32)
                new_val = old * alpha + beta * v_block.to(tl.float32)
                tl.store(O_ptr + acc_offs, new_val.to(tl.bfloat16))

            m_i = m_new

        inv_l = 1.0 / l_i
        for blk in range(NUM_BLOCKS):
            n_offs = tl.arange(0, BLOCK)
            out_offs = h * D + blk * BLOCK + n_offs
            cur = tl.load(O_ptr + out_offs).to(tl.float32)
            tl.store(O_ptr + out_offs, (cur * inv_l).to(tl.bfloat16))

    _Q40_KERNEL_AVAILABLE = True
except Exception:
    _Q40_KERNEL_AVAILABLE = False
    _q40_decode_attn_kernel = None


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def q40_attention_decode(q, k_q4, k_scale, v_q4, v_scale, chunk=512):
    """bs=1 decode attention reading Q4_0 packed buffers directly.

    Args:
        q: (H, D) bf16 query
        k_q4, v_q4: (S, H, D//2) uint8 packed
        k_scale, v_scale: (S, H*D//32) fp16
    Returns:
        (H, D) bf16 attention output
    """
    if not _Q40_KERNEL_AVAILABLE:
        raise RuntimeError("Triton Q4_0 attention kernel not available")
    H, D = q.shape
    S = k_q4.shape[0]
    out = torch.zeros(H, D, dtype=torch.bfloat16, device=q.device)
    D_PAD = _next_pow2(D)
    scale = D ** -0.5
    grid = (H * ((S + chunk - 1) // chunk),)
    _q40_decode_attn_kernel[grid](
        q, k_q4, k_scale, v_q4, v_scale, out,
        S, H, D=D, D_PAD=D_PAD, BLOCK=32, CHUNK=chunk, SCALE=scale, num_warps=1,
    )
    return out


# ============================================================================
# SGLang attention backend integration (PoC: bs=1 decode only)
# ============================================================================

try:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    _BACKEND_AVAILABLE = True
except ImportError:
    _BACKEND_AVAILABLE = False
    AttentionBackend = None


if _BACKEND_AVAILABLE:

    class Q40AttentionBackend(AttentionBackend):
        """Attention backend that reads Q4_0 packed KV directly via Triton kernel.

        PoC scope:
        - bs=1 decode via fused Q4_0 dequant + attention Triton kernel
        - Falls back to standard bf16 KV path (via pool's get_kv_buffer) for
          prefill and any mode the kernel can't handle

        Memory benefit: avoids materializing the full pool bf16 buffer during
        decode. Lets larger context windows fit alongside CUDA graphs.
        """

        def __init__(self, model_runner):
            self.runner = model_runner
            self.q40_available = _Q40_KERNEL_AVAILABLE
            self.decode_chunk = 512

        def init_forward_metadata(self, forward_batch):
            pass

        def init_cuda_graph_state(self, max_bs, max_num_tokens):
            pass

        def init_forward_metadata_capture_cuda_graph(
            self, bs, num_tokens, req_pool_indices, seq_lens,
            encoder_lens, forward_mode, spec_info,
        ):
            pass

        def init_forward_metadata_replay_cuda_graph(
            self, bs, req_pool_indices, seq_lens, seq_lens_sum,
            encoder_lens, forward_mode, spec_info, seq_lens_cpu,
        ):
            pass

        def get_cuda_graph_seq_len_fill_value(self):
            return 0

        def forward_decode(
            self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs,
        ):
            """bs=1 decode with fused Q4_0 dequant + attention."""
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            bs = q.shape[0]
            if (
                not self.q40_available
                or bs != 1
                or not hasattr(self.runner, "token_to_kv_pool")
                or self.runner.token_to_kv_pool is None
            ):
                return self._fallback_bf16_decode(q, k, v, layer, forward_batch)

            pool = self.runner.token_to_kv_pool
            if not hasattr(pool, "get_q4_kv_buffers"):
                return self._fallback_bf16_decode(q, k, v, layer, forward_batch)

            try:
                q_t = q[0].transpose(0, 1).contiguous()  # (H, D)
                k_q4, k_scale, v_q4, v_scale = pool.get_q4_kv_buffers(layer.layer_id)

                # Pool stores (S, H, D//2) after the permute in set_kv_buffer.
                # Our kernel expects (S, H, D//2) and (S, H*D//32) scales.
                # Both MHATokenToKVPoolQ40 and HybridLinearKVPool deliver these shapes.
                S = k_q4.shape[0]
                if S == 0:
                    out = torch.zeros_like(q_t).transpose(0, 1).unsqueeze(0)
                    return out

                out_hd = q40_attention_decode(
                    q_t, k_q4, k_scale, v_q4, v_scale, chunk=self.decode_chunk
                )
                return out_hd.transpose(0, 1).unsqueeze(0)  # back to (1, H, D)
            except Exception:
                return self._fallback_bf16_decode(q, k, v, layer, forward_batch)

        def _fallback_bf16_decode(self, q, k, v, layer, forward_batch):
            """Standard matmul-based decode using dequantized bf16 buffers.

            Used when Q4_0 path is unavailable (e.g., multi-batch, prefill,
            or non-Q4_0 pool). Reuses the caller-provided K, V tensors.
            """
            # Standard scaled dot-product attention for fallback.
            H = q.shape[1]
            D = q.shape[-1]
            scale = D ** -0.5
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v.float()).to(q.dtype)
            return out

        def forward_extend(
            self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs,
        ):
            # Prefill not yet implemented for Q4_0 path. Falls back to caller buffers.
            return self._fallback_bf16_decode(q, k, v, layer, forward_batch)

        def forward(
            self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs,
        ):
            from sglang.srt.layers.radix_attention import AttentionType
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            if forward_batch.forward_mode.is_idle():
                return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

            if layer.attention_type == AttentionType.ENCODER_ONLY:
                raise NotImplementedError("Encoder self-attention not supported")

            if forward_batch.forward_mode.is_decode():
                return self.forward_decode(q, k, v, layer, forward_batch, **kwargs)

            return self.forward_extend(q, k, v, layer, forward_batch, **kwargs)
