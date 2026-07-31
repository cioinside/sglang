"""Hybrid attention backend: flashinfer prefill + Q4_0-aware decode.

Routes by forward mode:
- extend (prefill): flashinfer (paged attention, reads dequantized bf16)
- decode:
  - bs=1 with Q4_0 pool: fused dequant+attention Triton kernel (Q4_0 packed)
  - bs>1 with SGLANG_Q4_BATCHED_DECODE=1:
      ragged-batch Triton kernel via req_to_token pool indices
  - otherwise: flashinfer (multi-batch fallback)

Memory benefit: decode path reads Q4_0 packed buffers directly, avoiding the
on-the-fly dequant that flashinfer's get_kv_buffer triggers. The bf16 workspace
that flashinfer normally materializes is not needed on the decode path; only
the packed Q4_0 pool is held for decode attention.

Hadamard pre-rotation (SGLANG_Q40_HADAMARD=1): if enabled, K stored rotated
(K @ H) and Q is rotated here (Q @ H) before attention. Since H is orthogonal,
scores stay equivalent to un-rotated attention while quantization captures more
information about the original K distribution (outlier-mass spread by H).
"""
import os as _os

import torch
import triton
import triton.language as tl

try:
    from sglang.srt.layers.quantization.q40_tensor import (
        _get_hadamard as _q40_get_hadamard,
    )
    from sglang.srt.layers.quantization.q40_tensor import _hadamard_enabled
    _HADAMARD_IMPORT_OK = True
except Exception:
    _HADAMARD_IMPORT_OK = False
    _q40_get_hadamard = None

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
        swa_start,
        H,
        D: tl.constexpr,
        D_PAD: tl.constexpr,
        BLOCK: tl.constexpr,
        CHUNK: tl.constexpr,
        SCALE: tl.constexpr,
        H_KV: tl.constexpr,
    ):
        pid = tl.program_id(0)
        # SWA: skip leading chunks before the sliding-window start.
        # swa_start is the first KV index inside the window (0 = full attention).
        eff_len = seq_len - swa_start
        chunks = tl.cdiv(eff_len, CHUNK)
        h = pid // chunks
        chunk_start = swa_start + (pid % chunks) * CHUNK
        chunk_end = tl.minimum(chunk_start + CHUNK, seq_len)

        # GQA: map query head h to the shared KV head
        kv_head = h * H_KV // H

        NUM_BLOCKS = D // BLOCK
        BYTES_PER_TOKEN_HEAD = D // 2

        d_offs = tl.arange(0, D_PAD)
        d_mask = d_offs < D
        tl.load(Q_ptr + h * D + d_offs, mask=d_mask, other=0.0)

        m_i = float('-inf')
        l_i = 0.0

        k_head_base = K_q4_ptr + kv_head * BYTES_PER_TOKEN_HEAD
        v_head_base = V_q4_ptr + kv_head * BYTES_PER_TOKEN_HEAD
        ks_head_base = K_scale_ptr + kv_head * NUM_BLOCKS
        vs_head_base = V_scale_ptr + kv_head * NUM_BLOCKS

        H_STRIDE_B = H_KV * BYTES_PER_TOKEN_HEAD
        H_STRIDE_S = H_KV * NUM_BLOCKS

        for s in range(chunk_start, chunk_end):
            k_base = k_head_base + s * H_STRIDE_B
            v_base = v_head_base + s * H_STRIDE_B
            ks_base = ks_head_base + s * H_STRIDE_S
            vs_base = vs_head_base + s * H_STRIDE_S

            score = tl.zeros((), dtype=tl.float32)
            for blk in range(NUM_BLOCKS):
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


try:
    @triton.jit
    def _q40_decode_attn_kernel_batched(
        Q_ptr,
        K_q4_ptr,
        K_scale_ptr,
        V_q4_ptr,
        V_scale_ptr,
        O_ptr,
        token_indices_ptr,
        cu_seqlens_ptr,
        window_size,
        H: tl.constexpr,
        D: tl.constexpr,
        D_PAD: tl.constexpr,
        BLOCK: tl.constexpr,
        CHUNK: tl.constexpr,
        SCALE: tl.constexpr,
        H_KV: tl.constexpr,
    ):
        batch = tl.program_id(0)
        h = tl.program_id(1)

        # GQA: map query head h to the shared KV head
        kv_head = h * H_KV // H

        seq_start = tl.load(cu_seqlens_ptr + batch)
        seq_end = tl.load(cu_seqlens_ptr + batch + 1)

        # SWA: per-batch offset into the sliding window.
        # Tokens before (seq_start + swa_off) are outside the window.
        swa_off = 0
        if window_size > 0:
            swa_off = tl.maximum(0, (seq_end - seq_start) - window_size)
        win_start = seq_start + swa_off

        NUM_BLOCKS = D // BLOCK
        BYTES_PER_TOKEN_HEAD = D // 2

        H_STRIDE_TOK = H_KV * BYTES_PER_TOKEN_HEAD
        H_STRIDE_SC = H_KV * NUM_BLOCKS

        d_offs = tl.arange(0, D_PAD)
        d_mask = d_offs < D
        tl.load(Q_ptr + batch * (H * D) + h * D + d_offs, mask=d_mask, other=0.0)

        m_i = float('-inf')
        l_i = 0.0

        zero_acc = tl.zeros((D_PAD,), dtype=tl.float32)
        acc_offs = h * D + d_offs
        tl.store(O_ptr + batch * (H * D) + acc_offs, zero_acc.to(tl.bfloat16), mask=d_mask)

        for s in range(win_start, seq_end):
            tok_idx = tl.load(token_indices_ptr + s)
            token_base = tok_idx * H_STRIDE_TOK
            scale_base = tok_idx * H_STRIDE_SC

            k_base = K_q4_ptr + token_base + kv_head * BYTES_PER_TOKEN_HEAD
            v_base = V_q4_ptr + token_base + kv_head * BYTES_PER_TOKEN_HEAD
            ks_base = K_scale_ptr + scale_base + kv_head * NUM_BLOCKS
            vs_base = V_scale_ptr + scale_base + kv_head * NUM_BLOCKS

            score = tl.zeros((), dtype=tl.float32)
            for blk in range(NUM_BLOCKS):
                n_offs = tl.arange(0, BLOCK)
                packed_off = n_offs // 2
                is_high = (n_offs % 2) != 0

                k_packed = tl.load(k_base + blk * (BLOCK // 2) + packed_off)
                k_low = k_packed & 0x0F
                k_high = (k_packed >> 4) & 0x0F
                k_vals = tl.where(is_high, k_high, k_low)
                k_scale_v = tl.load(ks_base + blk).to(tl.bfloat16)
                k_block = (k_vals.to(tl.bfloat16) - 8.0) * k_scale_v

                q_block = tl.load(Q_ptr + batch * (H * D) + h * D + blk * BLOCK + n_offs)
                score += tl.sum(q_block.to(tl.float32) * k_block.to(tl.float32))

            score = score * SCALE

            m_new = tl.maximum(m_i, score)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(score - m_new)
            l_i = l_i * alpha + beta

            for blk in range(NUM_BLOCKS):
                n_offs = tl.arange(0, BLOCK)
                packed_off = n_offs // 2
                is_high = (n_offs % 2) != 0

                v_packed = tl.load(v_base + blk * (BLOCK // 2) + packed_off)
                v_low = v_packed & 0x0F
                v_high = (v_packed >> 4) & 0x0F
                v_vals = tl.where(is_high, v_high, v_low)
                v_scale_v = tl.load(vs_base + blk).to(tl.bfloat16)
                v_block = (v_vals.to(tl.bfloat16) - 8.0) * v_scale_v

                cur_off = batch * (H * D) + h * D + blk * BLOCK + n_offs
                old = tl.load(O_ptr + cur_off).to(tl.float32)
                new_val = old * alpha + beta * v_block.to(tl.float32)
                tl.store(O_ptr + cur_off, new_val.to(tl.bfloat16))

            m_i = m_new

        inv_l = 1.0 / l_i
        for blk in range(NUM_BLOCKS):
            n_offs = tl.arange(0, BLOCK)
            cur_off = batch * (H * D) + h * D + blk * BLOCK + n_offs
            cur = tl.load(O_ptr + cur_off).to(tl.float32)
            tl.store(O_ptr + cur_off, (cur * inv_l).to(tl.bfloat16))

    _Q40_BATCHED_KERNEL_AVAILABLE = True
except Exception:
    _Q40_BATCHED_KERNEL_AVAILABLE = False
    _q40_decode_attn_kernel_batched = None


_Q40_BATCHED_ENABLED = _os.environ.get("SGLANG_Q4_BATCHED_DECODE", "0") == "1"

# Static Q4_0 decode path for CUDA graph compatibility.
# When enabled, the decode path is branchless (no Python `if`s on tensor shapes
# or pool state), so the forward pass can be captured by CUDA graph.
# Assumes bs=1 (matches --cuda-graph-max-bs 1) and standard decode invariants
# (cache_loc set, save_kv_cache=True, k/v provided, Q4_0 pool active).
_Q40_STATIC_FOR_GRAPH = _os.environ.get("SGLANG_Q40_STATIC", "0") == "1"


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def q40_attention_decode(q, k_q4, k_scale, v_q4, v_scale, h_kv=None, chunk=512, window_size=0):
    """bs=1 decode attention reading Q4_0 packed buffers directly.

    Args:
        q: (H, D) bf16 query
        k_q4, v_q4: (S, H_KV, D//2) uint8 packed — H_KV = kv heads (GQA)
        k_scale, v_scale: (S, H_KV*D//32) fp16
        h_kv: number of KV heads (defaults to H i.e. MHA)
        chunk: CHUNK size for kernel grid
        window_size: sliding-window size (0 = full attention, >0 = SWA)
    Returns:
        (H, D) bf16 attention output
    """
    if not _Q40_KERNEL_AVAILABLE:
        raise RuntimeError("Triton Q4_0 attention kernel not available")
    H, D = q.shape
    H_KV = h_kv if h_kv is not None else H
    S = k_q4.shape[0]
    if _HADAMARD_IMPORT_OK and _hadamard_enabled() and D in (32, 64, 128, 256) and (D & (D - 1)) == 0:
        H_rot = _q40_get_hadamard(D, str(q.device), q.dtype)
        q = torch.matmul(q, H_rot)
    out = torch.zeros(H, D, dtype=torch.bfloat16, device=q.device)
    D_PAD = _next_pow2(D)
    scale = D ** -0.5
    if window_size and window_size > 0 and window_size < S:
        swa_start = S - int(window_size)
    else:
        swa_start = 0
    eff_len = S - swa_start
    chunks = (eff_len + chunk - 1) // chunk
    grid = (H * chunks,)
    _q40_decode_attn_kernel[grid](
        q, k_q4, k_scale, v_q4, v_scale, out,
        S, swa_start, H, D=D, D_PAD=D_PAD, BLOCK=32, CHUNK=chunk, SCALE=scale, H_KV=H_KV,
        num_warps=1,
    )
    return out


def q40_attention_decode_batched(
    q, k_q4, k_scale, v_q4, v_scale,
    token_indices, cu_seqlens, B,
    h_kv=None, chunk=512, window_size=0,
):
    """Multi-batch paged decode attention reading Q4_0 packed buffers.

    Args:
        q: (B, H, D) bf16 query
        k_q4, v_q4: (max_size, H_KV, D//2) uint8 packed — full pool, H_KV = kv heads
        k_scale, v_scale: (max_size, H_KV*D//32) fp16
        token_indices: (sum_seqlen,) int32 — flat pool indices, concat of per-batch slots
        cu_seqlens: (B+1,) int32 — cumulative sequence lengths
        B: batch size
        h_kv: number of KV heads (defaults to H i.e. MHA)
        chunk: CHUNK size for kernel grid
        window_size: sliding-window size (0 = full attention, >0 = SWA per batch)
    Returns:
        (B, H, D) bf16 attention output
    """
    if not _Q40_BATCHED_KERNEL_AVAILABLE:
        raise RuntimeError("Triton Q4_0 batched attention kernel not available")
    H, D = q.shape[1], q.shape[2]
    H_KV = h_kv if h_kv is not None else H
    if _HADAMARD_IMPORT_OK and _hadamard_enabled() and D in (32, 64, 128, 256) and (D & (D - 1)) == 0:
        H_rot = _q40_get_hadamard(D, str(q.device), q.dtype)
        q = torch.matmul(q, H_rot)
    out = torch.zeros(B, H, D, dtype=torch.bfloat16, device=q.device)
    D_PAD = _next_pow2(D)
    scale = D ** -0.5
    grid = (B, H)
    _q40_decode_attn_kernel_batched[grid](
        q, k_q4, k_scale, v_q4, v_scale, out,
        token_indices, cu_seqlens, int(window_size),
        H=H, D=D, D_PAD=D_PAD, BLOCK=32, CHUNK=chunk, SCALE=scale, H_KV=H_KV,
        num_warps=1,
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
            self.batched_available = _Q40_BATCHED_KERNEL_AVAILABLE
            self.batched_enabled = _Q40_BATCHED_ENABLED
            self.decode_chunk = 512
            self._token_to_kv_pool = None
            self._req_to_token_pool = None
            self._fallback_warned = False
            self._shape_warned = False
            self._q4_0_static = _Q40_STATIC_FOR_GRAPH and self.q40_available

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

        def _build_ragged_indices(self, forward_batch, layer, debug_tag=""):
            """Build (token_indices, cu_seqlens) for ragged-batch Q4_0 decode.

            Uses req_to_token to gather per-request token slots in the Q4_0 pool,
            concatenates into a flat index, and computes cumulative seq_lens.
            """
            req_pool = self.runner.req_to_token_pool
            req_to_token = req_pool.req_to_token  # (max_req, max_seq) int32
            req_indices = forward_batch.req_pool_indices  # (B,)
            seq_lens = forward_batch.seq_lens  # (B,) int32

            B = req_indices.shape[0]
            device = req_to_token.device

            max_seq = req_to_token.shape[1]
            batch_tokens = req_to_token[req_indices]  # (B, max_seq) int32
            positions = torch.arange(max_seq, device=device, dtype=torch.int32).unsqueeze(0)
            valid_mask = positions < seq_lens.unsqueeze(1)

            token_indices = batch_tokens[valid_mask]  # (sum_seqlen,) — flat slot indices
            cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = torch.cumsum(seq_lens.to(torch.int32), dim=0)

            if debug_tag:
                import logging
                log = logging.getLogger(__name__)
                log.info(
                    "Q4B[%s] B=%d req_ids=%s seq_lens=%s tok_indices=%s cu_seqlens=%s pool_sizes=(%d,%d) max_total=%d",
                    debug_tag, B, list(req_indices.cpu().numpy()),
                    list(seq_lens.cpu().numpy()),
                    list(token_indices.cpu().numpy()),
                    list(cu_seqlens.cpu().numpy()),
                    req_to_token.shape[0], req_to_token.shape[1],
                    self.runner.token_to_kv_pool.get_kv_buffers(layer.layer_id)[0].shape[0]
                    if hasattr(self.runner.token_to_kv_pool, "get_kv_buffers") else -1,
                )
            return token_indices, cu_seqlens, B

        def _try_q40_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            """Q4_0 Triton kernel path for decode.

            bs=1: contiguous-token kernel (reads 0..seq_len-1).
            bs>1: ragged-batch kernel via req_to_token pool indices.
            Falls back to flashinfer if conditions not met.
            """
            bs = q.shape[0]
            if not self.q40_available or not self._q40_pool_active():
                return None
            if bs > 1 and (not self.batched_available or not self.batched_enabled):
                return None
            if bs == 0:
                return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

            h_kv = layer.tp_k_head_num
            window_size = max(0, int(getattr(layer, "sliding_window_size", -1) or 0))
            pool = self.runner.token_to_kv_pool
            try:
                if save_kv_cache:
                    cache_loc = (
                        forward_batch.out_cache_loc
                        if not getattr(layer, "is_cross_attention", False)
                        else getattr(forward_batch, "encoder_out_cache_loc", None)
                    )
                    if cache_loc is not None and k is not None and v is not None:
                        pool.set_kv_buffer(layer, cache_loc, k, v)

                k_q4, k_scale, v_q4, v_scale = pool.get_q4_kv_buffers(layer.layer_id)

                if bs == 1:
                    req_pool = self.runner.req_to_token_pool
                    req_to_token = req_pool.req_to_token
                    rp_idx = forward_batch.req_pool_indices[0].item()
                    seq_len = forward_batch.seq_lens[0].item()
                    tok_slots = req_to_token[rp_idx, :seq_len]
                    S = len(tok_slots)
                    if S == 0:
                        return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    H = layer.tp_q_head_num
                    D = layer.head_dim
                    q_t = q[0].view(H, D).contiguous()
                    k_q4_s = k_q4[tok_slots]
                    k_scale_s = k_scale[tok_slots]
                    v_q4_s = v_q4[tok_slots]
                    v_scale_s = v_scale[tok_slots]
                    out_hd = q40_attention_decode(
                        q_t, k_q4_s, k_scale_s, v_q4_s, v_scale_s,
                        h_kv=h_kv, chunk=self.decode_chunk, window_size=window_size,
                    )
                    return out_hd.view(1, H * D).to(torch.float16)

                # bs>1: ragged-batch via req_to_token
                token_indices, cu_seqlens, B = self._build_ragged_indices(forward_batch, layer, debug_tag="mb")
                sum_seqlen = int(token_indices.shape[0])
                if sum_seqlen == 0:
                    return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                H = layer.tp_q_head_num
                D = layer.head_dim
                q_reshaped = q.view(B, H, D).contiguous()
                out = q40_attention_decode_batched(
                    q_reshaped, k_q4, k_scale, v_q4, v_scale,
                    token_indices, cu_seqlens, B,
                    h_kv=h_kv, chunk=self.decode_chunk, window_size=window_size,
                )
                return out.view(B, H * D).to(torch.float16)
            except Exception as e:
                if not self._fallback_warned:
                    import logging
                    import traceback
                    logging.getLogger(__name__).warning(
                        "Q4_0 decode path failed (bs=%d, q.shape=%s), falling back to flashinfer: %s: %s\n%s",
                        q.shape[0], list(q.shape), type(e).__name__, e,
                        traceback.format_exc(),
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
            if self._q4_0_static:
                return self._q4_0_decode_static(q, k, v, layer, forward_batch)
            out = self._try_q40_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )
            if out is not None:
                return out
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        def _q4_0_decode_static(self, q, k, v, layer, forward_batch):
            """Branchless Q4_0 decode for CUDA graph capture (bs=1).

            Assumes standard decode invariants (validated at init via _q4_0_static):
              - bs == 1 (matches --cuda-graph-max-bs 1)
              - pool has Q4_0 buffers
              - k, v are provided
              - forward_batch has out_cache_loc
            """
            h_kv = layer.tp_k_head_num
            window_size = max(0, int(getattr(layer, "sliding_window_size", -1) or 0))
            pool = self.runner.token_to_kv_pool

            cache_loc = forward_batch.out_cache_loc
            pool.set_kv_buffer(layer, cache_loc, k, v)

            k_q4, k_scale, v_q4, v_scale = pool.get_q4_kv_buffers(layer.layer_id)

            req_pool = self.runner.req_to_token_pool
            req_to_token = req_pool.req_to_token
            rp_idx = forward_batch.req_pool_indices[0]
            seq_len = forward_batch.seq_lens[0]
            tok_slots = req_to_token[rp_idx, :seq_len]
            H = layer.tp_q_head_num
            D = layer.head_dim
            q_t = q[0].view(H, D).contiguous()
            k_q4_s = k_q4[tok_slots]
            k_scale_s = k_scale[tok_slots]
            v_q4_s = v_q4[tok_slots]
            v_scale_s = v_scale[tok_slots]
            out_hd = q40_attention_decode(
                q_t, k_q4_s, k_scale_s, v_q4_s, v_scale_s,
                h_kv=h_kv, chunk=self.decode_chunk, window_size=window_size,
            )
            return out_hd.view(1, H * D).to(torch.float16)

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
                return self.forward_decode(
                    q, k, v, layer, forward_batch, save_kv_cache, **kwargs
                )
            return self.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
