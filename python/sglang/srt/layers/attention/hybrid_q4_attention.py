"""Hybrid attention backend: flashinfer prefill + Q4_0-aware decode.

Routes by forward mode:
- extend (prefill): flashinfer (paged attention, reads dequantized bf16)
- decode:
  - bs=1 with Q4_0 pool: fused dequant+attention Triton kernel (Q4_0 packed)
  - otherwise: flashinfer (multi-batch decode path)

Memory benefit: the bs=1 decode path reads Q4_0 packed buffers directly,
avoiding the on-the-fly dequant that flashinfer's get_kv_buffer triggers.
The bf16 workspace that flashinfer normally materializes is not needed on
the decode path; only the packed Q4_0 pool is held for decode attention.
"""
import torch
import triton
import triton.language as tl

try:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    _BACKEND_AVAILABLE = True
except ImportError:
    _BACKEND_AVAILABLE = False
    AttentionBackend = None


try:
    @triton.jit
    def _q40_decode_attn_kernel(
        Q_ptr,
        K_q4_ptr,
        K_scale_ptr,
        V_q4_ptr,
        V_scale_ptr,
        O_ptr,
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


if _BACKEND_AVAILABLE:

    class HybridQ4FlashInferAttnBackend(AttentionBackend):
        """Hybrid: flashinfer prefill + Q4_0 Triton decode.

        All metadata (kv_indptr, decode_wrappers, CUDA graph state, etc.) is
        owned by FlashInferAttnBackend; this wrapper only re-routes forward
        decode to the Q4_0 kernel when conditions allow.
        """

        def __init__(self, model_runner):
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            self.full_attn_backend = FlashInferAttnBackend(model_runner)
            self.runner = model_runner
            self.q40_available = _Q40_KERNEL_AVAILABLE
            self.decode_chunk = 512
            self._token_to_kv_pool = None
            self._req_to_token_pool = None
            self._fallback_warned = False

        @property
        def token_to_kv_pool(self):
            pool = self.full_attn_backend.token_to_kv_pool
            if pool is not None and self._token_to_kv_pool is None:
                self._token_to_kv_pool = pool
            return self._token_to_kv_pool

        @property
        def req_to_token_pool(self):
            pool = self.full_attn_backend.req_to_token_pool
            if pool is not None and self._req_to_token_pool is None:
                self._req_to_token_pool = pool
            return self._req_to_token_pool

        def _q40_pool_active(self):
            pool = self.runner.token_to_kv_pool
            return pool is not None and hasattr(pool, "get_q4_kv_buffers")

        def _try_q40_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            """Q4_0 Triton kernel path for bs=1 decode.

            Pool layout: get_q4_kv_buffers returns (k_packed, k_scale,
            v_packed, v_scale) shaped (S, H, D//2) and (S, H*D//32).
            """
            bs = q.shape[0]
            if bs != 1 or not self.q40_available or not self._q40_pool_active():
                return None

            pool = self.runner.token_to_kv_pool
            try:
                from sglang.srt.mem_cache.memory_pool import KVWriteLoc

                if save_kv_cache:
                    cache_loc = (
                        forward_batch.out_cache_loc
                        if not getattr(layer, "is_cross_attention", False)
                        else getattr(forward_batch, "encoder_out_cache_loc", None)
                    )
                    if cache_loc is not None and k is not None and v is not None:
                        pool.set_kv_buffer(
                            layer,
                            KVWriteLoc(cache_loc, None),
                            k,
                            v,
                        )

                q_t = q[0].transpose(0, 1).contiguous()
                k_q4, k_scale, v_q4, v_scale = pool.get_q4_kv_buffers(layer.layer_id)
                S = k_q4.shape[0]
                if S == 0:
                    return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

                out_hd = q40_attention_decode(
                    q_t, k_q4, k_scale, v_q4, v_scale, chunk=self.decode_chunk
                )
                return out_hd.transpose(0, 1).unsqueeze(0)
            except Exception as e:
                if not self._fallback_warned:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Q4_0 decode path failed, falling back to flashinfer: %s",
                        type(e).__name__,
                    )
                    self._fallback_warned = True
                return None

        def init_forward_metadata(self, forward_batch):
            self.full_attn_backend.init_forward_metadata(forward_batch)

        def init_cuda_graph_state(self, max_bs, max_num_tokens):
            self.full_attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

        def init_forward_metadata_capture_cuda_graph(
            self, bs, num_tokens, req_pool_indices, seq_lens,
            encoder_lens, forward_mode, spec_info,
        ):
            self.full_attn_backend.init_forward_metadata_capture_cuda_graph(
                bs, num_tokens, req_pool_indices, seq_lens,
                encoder_lens, forward_mode, spec_info,
            )

        def init_forward_metadata_replay_cuda_graph(
            self, bs, req_pool_indices, seq_lens, seq_lens_sum,
            encoder_lens, forward_mode, spec_info, seq_lens_cpu,
        ):
            self.full_attn_backend.init_forward_metadata_replay_cuda_graph(
                bs, req_pool_indices, seq_lens, seq_lens_sum,
                encoder_lens, forward_mode, spec_info, seq_lens_cpu,
            )

        def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
            if hasattr(self.full_attn_backend, "init_forward_metadata_out_graph"):
                self.full_attn_backend.init_forward_metadata_out_graph(
                    forward_batch, in_capture=in_capture
                )

        def get_cuda_graph_seq_len_fill_value(self):
            return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

        def init_mha_chunk_metadata(self, forward_batch, disable_flashinfer_ragged=False):
            if hasattr(self.full_attn_backend, "init_mha_chunk_metadata"):
                self.full_attn_backend.init_mha_chunk_metadata(
                    forward_batch, disable_flashinfer_ragged=disable_flashinfer_ragged
                )

        def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
            out = self._try_q40_decode(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache)
            if out is not None:
                return out
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
            from sglang.srt.layers.radix_attention import AttentionType

            if forward_batch.forward_mode.is_idle():
                return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            if layer.attention_type == AttentionType.ENCODER_ONLY:
                raise NotImplementedError("Encoder self-attention not supported")
            if forward_batch.forward_mode.is_decode():
                return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
            return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
