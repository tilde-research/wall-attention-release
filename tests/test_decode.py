"""Wall decode parity: single-step decode against a pre-rescaled KV cache must
reproduce the training/prefill forward at the same position."""

import pytest
import torch
from conftest import make_log_gates

from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_global_cumsum

from wall_attn import build_wall_kv_cache, wall_attn, wall_attn_decode

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _decode_at(t, q, k, v, P, scale, C, *, g_scalar_cumsum=None):
    """Decode at position `t` with the cache truncated to [0, t].

    Decode has no intra-cache causal mask (the query is assumed to come after
    everything in the cache), so we truncate the cache to the current position.
    """
    k_c = k[:, : t + 1].contiguous()
    v_c = v[:, : t + 1].contiguous()
    P_c = P[:, : t + 1].contiguous()
    gsc = g_scalar_cumsum[:, : t + 1].contiguous() if g_scalar_cumsum is not None else None

    k_tilde, r_cache = build_wall_kv_cache(k_c, P_c, chunk_size=C)
    o_dec, _ = wall_attn_decode(
        q=q[:, t : t + 1].contiguous(),
        v=v_c,
        p_curr=P[:, t : t + 1].contiguous(),
        k_tilde=k_tilde,
        r_cache=r_cache,
        sink_bias=None,
        scale=scale,
        cache_chunk_size=C,
        g_scalar_cumsum=gsc,
    )
    return o_dec


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,T,H,HQ,K,V,C", [
    (1, 256, 4, 4, 64, 64, 64),   # MHA
    (1, 256, 2, 8, 64, 64, 64),   # GQA, G=4
    (2, 128, 1, 2, 32, 32, 32),   # small
])
def test_decode_matches_training_forward(dtype, B, T, H, HQ, K, V, C):
    """Decode at position t reproduces the training forward output at that row."""
    torch.manual_seed(0)
    device = "cuda"
    scale = K**-0.5
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = make_log_gates(B, T, HQ, K, device=device)

    o_ref = wall_attn(q, k, v, g, scale=scale)
    P = chunk_global_cumsum(g, scale=RCP_LN2)

    for t in (T - 1, (T // 2 // C) * C + C - 1):
        o_dec = _decode_at(t, q, k, v, P, scale, C)
        tol = 3e-2 if dtype == torch.bfloat16 else 5e-3
        torch.testing.assert_close(o_dec, o_ref[:, t : t + 1], rtol=tol, atol=tol)


@requires_cuda
def test_decode_matches_training_forward_long():
    """Long-context stability: per-block reference must keep exp2 finite."""
    dtype = torch.bfloat16
    torch.manual_seed(1)
    B, T, H, HQ, K, V, C = 1, 4096, 2, 4, 64, 64, 128
    device = "cuda"
    scale = K**-0.5
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = make_log_gates(B, T, HQ, K, device=device)

    o_ref = wall_attn(q, k, v, g, scale=scale)
    P = chunk_global_cumsum(g, scale=RCP_LN2)

    t = T - 1
    o_dec = _decode_at(t, q, k, v, P, scale, C)
    assert torch.isfinite(o_dec).all(), "decode output must be finite at long context"
    torch.testing.assert_close(o_dec, o_ref[:, t : t + 1], rtol=4e-2, atol=4e-2)


@requires_cuda
def test_decode_with_scalar_gate():
    """Wall + FoX-style scalar gate: decode matches training forward."""
    torch.manual_seed(2)
    dtype = torch.float32
    B, T, H, HQ, K, V, C = 1, 256, 2, 4, 64, 64, 64
    device = "cuda"
    scale = K**-0.5
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = make_log_gates(B, T, HQ, K, device=device)
    g_scalar = torch.randn(B, T, HQ, device=device, dtype=dtype) * 0.05

    o_ref = wall_attn(q, k, v, g, scale=scale, g_scalar=g_scalar)
    P = chunk_global_cumsum(g, scale=RCP_LN2)
    c = chunk_global_cumsum(g_scalar, scale=RCP_LN2)

    t = T - 1
    o_dec = _decode_at(t, q, k, v, P, scale, C, g_scalar_cumsum=c)
    torch.testing.assert_close(o_dec, o_ref[:, t : t + 1], rtol=5e-3, atol=5e-3)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_streaming_matches_full_forward(dtype):
    """End-to-end serving pattern: prefill the cache, then decode token-by-token,
    appending each new (k_tilde, v) to a pre-allocated buffer, and compare each
    step against the batched training forward at that position.
    """
    torch.manual_seed(123)
    device = "cuda"
    B, T_prefill, T_gen, H, HQ, K, V, C = 1, 96, 16, 2, 4, 64, 64, 32
    T = T_prefill + T_gen
    G = HQ // H
    scale = K**-0.5

    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = make_log_gates(B, T, HQ, K, device=device)

    o_ref = wall_attn(q, k, v, g, scale=scale)
    P = chunk_global_cumsum(g, scale=RCP_LN2)

    NC_max = (T + C - 1) // C
    k_tilde_buf = torch.zeros(B, T, HQ, K, device=device, dtype=dtype)
    v_buf = torch.zeros(B, T, H, V, device=device, dtype=dtype)
    r_cache_buf = torch.zeros(B, NC_max, HQ, K, device=device, dtype=P.dtype)

    # Prefill: bulk-build the cache for the first T_prefill tokens.
    k_tilde_pre, r_cache_pre = build_wall_kv_cache(
        k[:, :T_prefill].contiguous(), P[:, :T_prefill].contiguous(), chunk_size=C,
    )
    NC_pre = r_cache_pre.shape[1]
    k_tilde_buf[:, :T_prefill] = k_tilde_pre
    v_buf[:, :T_prefill] = v[:, :T_prefill]
    r_cache_buf[:, :NC_pre] = r_cache_pre

    tol = 5e-3 if dtype == torch.float32 else 3e-2
    for t in range(T_prefill, T):
        c = t // C
        if t % C == 0:
            r_cache_buf[:, c] = P[:, t]  # new chunk: freeze anchor R_c = P[t]
        R_c = r_cache_buf[:, c : c + 1]
        k_q_t = k[:, t : t + 1].repeat_interleave(G, dim=2)
        k_tilde_buf[:, t : t + 1] = (
            k_q_t.float() * torch.exp2(R_c.float() - P[:, t : t + 1].float())
        ).to(dtype)
        v_buf[:, t : t + 1] = v[:, t : t + 1]

        T_kv = t + 1
        NC_t = (T_kv + C - 1) // C
        o_t, _ = wall_attn_decode(
            q=q[:, t : t + 1].contiguous(),
            v=v_buf[:, :T_kv].contiguous(),
            p_curr=P[:, t : t + 1].contiguous(),
            k_tilde=k_tilde_buf[:, :T_kv].contiguous(),
            r_cache=r_cache_buf[:, :NC_t].contiguous(),
            sink_bias=None,
            scale=scale,
            cache_chunk_size=C,
        )
        torch.testing.assert_close(o_t, o_ref[:, t : t + 1], rtol=tol, atol=tol)


@requires_cuda
def test_decode_cache_layout_shapes():
    """Pre-rescaled cache has documented shapes; r_cache size == ceil(T/C)."""
    torch.manual_seed(3)
    B, T, H, HQ, K = 2, 200, 2, 8, 32
    device = "cuda"
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    g = make_log_gates(B, T, HQ, K, device=device)
    P = chunk_global_cumsum(g, scale=RCP_LN2)
    for C in (32, 64, 128):
        k_tilde, r_cache = build_wall_kv_cache(k, P, chunk_size=C)
        NC = (T + C - 1) // C
        assert k_tilde.shape == (B, T, HQ, K)
        assert r_cache.shape == (B, NC, HQ, K)
