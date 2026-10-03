"""Wall training/prefill parity: Triton `wall_attn` vs the eager reference."""

import math

import pytest
import torch
from conftest import make_log_gates

from wall_attn import wall_attn, wall_attn_reference

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("B,T,H,HQ,K,V", [(1, 48, 2, 4, 32, 16), (2, 31, 1, 1, 24, 8)])
@pytest.mark.parametrize("window_size", [None, 8])
def test_matches_reference_mha(dtype, B, T, H, HQ, K, V, window_size):
    assert HQ % H == 0
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = make_log_gates(B, T, HQ, K, device=device)
    scale = K**-0.5

    o_ref = wall_attn_reference(q, k, v, g, scale=scale, window_size=window_size)
    o = wall_attn(q, k, v, g, scale=scale, window_size=window_size)

    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_gqa_matches_reference():
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 40, 2, 8, 32, 24
    assert HQ // H == 4
    torch.manual_seed(1)
    device = "cuda"
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = make_log_gates(B, T, HQ, K, device=device)
    scale = K**-0.5

    o_ref = wall_attn_reference(q, k, v, g, scale=scale)
    o = wall_attn(q, k, v, g, scale=scale)
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_varlen_packed_matches_reference():
    dtype = torch.float32
    T1, T2 = 17, 23
    T = T1 + T2
    H, HQ, K, V = 1, 2, 16, 12
    torch.manual_seed(2)
    device = "cuda"
    q = torch.randn(1, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype)
    g = make_log_gates(1, T, HQ, K, device=device)
    cu = torch.tensor([0, T1, T], dtype=torch.long, device=device)
    scale = K**-0.5

    o_ref = wall_attn_reference(q, k, v, g, scale=scale, cu_seqlens=cu)
    o = wall_attn(q, k, v, g, scale=scale, cu_seqlens=cu)
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_sink_bias_matches_reference():
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 29, 1, 2, 20, 10
    torch.manual_seed(3)
    device = "cuda"
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = make_log_gates(B, T, HQ, K, device=device)
    sink_bias = torch.randn(HQ, device=device, dtype=dtype) * 0.1
    scale = K**-0.5

    o_ref = wall_attn_reference(q, k, v, g, scale=scale, sink_bias=sink_bias)
    o = wall_attn(q, k, v, g, scale=scale, sink_bias=sink_bias)
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_aggressive_gates_long_seq():
    """Strong per-timestep decay; exact reference stays in fp32, kernel uses per-block R."""
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 512, 1, 1, 32, 32
    device = "cuda"
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.full((B, T, HQ, K), math.log2(0.9), device=device, dtype=dtype)
    scale = K**-0.5

    o_ref = wall_attn_reference(q, k, v, g, scale=scale)
    o = wall_attn(q, k, v, g, scale=scale)
    torch.testing.assert_close(o, o_ref, rtol=3e-2, atol=3e-2)


@requires_cuda
def test_backward_matches_eager_reference():
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 24, 2, 4, 16, 12
    device = "cuda"
    torch.manual_seed(11)
    q0 = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k0 = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v0 = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g0 = make_log_gates(B, T, HQ, K, device=device)
    scale = K**-0.5

    q = q0.clone().requires_grad_(True)
    k = k0.clone().requires_grad_(True)
    v = v0.clone().requires_grad_(True)
    g = g0.clone().requires_grad_(True)

    q2 = q0.clone().requires_grad_(True)
    k2 = k0.clone().requires_grad_(True)
    v2 = v0.clone().requires_grad_(True)
    g2 = g0.clone().requires_grad_(True)

    o = wall_attn(q, k, v, g, scale=scale)
    o_ref = wall_attn_reference(q2, k2, v2, g2, scale=scale)
    go = torch.randn_like(o)
    o.backward(go)
    o_ref.backward(go)

    torch.testing.assert_close(q.grad, q2.grad, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(k.grad, k2.grad, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(v.grad, v2.grad, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(g.grad, g2.grad, rtol=8e-2, atol=8e-2)


@requires_cuda
def test_dg_nonzero_after_backward():
    torch.manual_seed(3)
    B, T, H, HQ, K, V = 1, 16, 1, 1, 8, 8
    device = "cuda"
    q = torch.randn(B, T, HQ, K, device=device, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, requires_grad=True)
    g = make_log_gates(B, T, HQ, K, device=device).requires_grad_(True)
    o = wall_attn(q, k, v, g, scale=K**-0.5)
    o.sum().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()


@requires_cuda
def test_g_gradient_matches_finite_differences():
    """dL/dg for the Triton Wall path vs central finite differences.

    The loss is accumulated in fp64; the step is sized for fp32 softmax logits.
    """
    torch.manual_seed(7)
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 4, 1, 1, 3, 3
    device = "cuda"
    scale = K**-0.5
    eps = 3e-3

    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g0 = make_log_gates(B, T, HQ, K, device=device)
    go = torch.randn(B, T, HQ, V, device=device, dtype=dtype)

    g = g0.clone().requires_grad_(True)
    o = wall_attn(q, k, v, g, scale=scale)
    (o * go).sum().backward()
    assert g.grad is not None
    dg_ana = g.grad.detach().clone()

    g_flat = g0.reshape(-1)
    dg_fd = torch.empty_like(g_flat)
    for i in range(g_flat.numel()):
        gp = g_flat.clone()
        gm = g_flat.clone()
        gp[i] += eps
        gm[i] -= eps
        op = wall_attn(q, k, v, gp.view_as(g0), scale=scale)
        om = wall_attn(q, k, v, gm.view_as(g0), scale=scale)
        Lp = (op * go).sum().double()
        Lm = (om * go).sum().double()
        dg_fd[i] = ((Lp - Lm) / (2.0 * eps)).to(dtype)

    dg_fd = dg_fd.view_as(dg_ana)
    torch.testing.assert_close(dg_ana, dg_fd, rtol=0.22, atol=0.13)


@requires_cuda
@pytest.mark.parametrize("B,T,H,HQ,K,V", [(1, 48, 2, 4, 32, 16), (2, 31, 1, 1, 24, 8)])
def test_scalar_gate_matches_reference(B, T, H, HQ, K, V):
    """Wall + FoX-style additive scalar gate: Triton vs reference."""
    dtype = torch.float32
    device = "cuda"
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = make_log_gates(B, T, HQ, K, device=device)
    g_scalar = torch.randn(B, T, HQ, device=device, dtype=dtype) * 0.1
    scale = K**-0.5

    o_ref = wall_attn_reference(q, k, v, g, scale=scale, g_scalar=g_scalar)
    o = wall_attn(q, k, v, g, scale=scale, g_scalar=g_scalar)
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_scalar_gate_gradient_finite_differences():
    """dL/dg_scalar for Wall + scalar gate via central differences."""
    torch.manual_seed(13)
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 4, 1, 1, 3, 3
    device = "cuda"
    scale = K**-0.5
    eps = 3e-3

    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g0 = make_log_gates(B, T, HQ, K, device=device)
    gs0 = torch.randn(B, T, HQ, device=device, dtype=dtype) * 0.1
    go = torch.randn(B, T, HQ, V, device=device, dtype=dtype)

    gs = gs0.clone().requires_grad_(True)
    o = wall_attn(q, k, v, g0, scale=scale, g_scalar=gs)
    (o * go).sum().backward()
    assert gs.grad is not None
    dgs_ana = gs.grad.detach().clone()

    gs_flat = gs0.reshape(-1)
    dgs_fd = torch.empty_like(gs_flat)
    for i in range(gs_flat.numel()):
        gsp = gs_flat.clone()
        gsm = gs_flat.clone()
        gsp[i] += eps
        gsm[i] -= eps
        op = wall_attn(q, k, v, g0, scale=scale, g_scalar=gsp.view_as(gs0))
        om = wall_attn(q, k, v, g0, scale=scale, g_scalar=gsm.view_as(gs0))
        Lp = (op * go).sum().double()
        Lm = (om * go).sum().double()
        dgs_fd[i] = ((Lp - Lm) / (2.0 * eps)).to(dtype)

    dgs_fd = dgs_fd.view_as(dgs_ana)
    torch.testing.assert_close(dgs_ana, dgs_fd, rtol=0.22, atol=0.13)
