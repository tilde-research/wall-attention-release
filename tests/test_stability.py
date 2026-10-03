"""Tests for soft-bounded gates and padded Triton tiles."""
import pytest
import torch
import torch.nn.functional as F
import triton

from wall_attn import soft_clamp_log_gates, wall_attn, wall_attn_reference
import wall_attn.training as training


def test_soft_clamp_value_and_derivative():
    g = torch.tensor([0., -1e-8, -.1, -8., -100.], dtype=torch.float64, requires_grad=True)
    actual = soft_clamp_log_gates(g)
    torch.testing.assert_close(actual, -.86 * (1 - torch.exp(g / .86)), atol=1e-15, rtol=1e-12)
    actual.sum().backward()
    torch.testing.assert_close(g.grad, torch.exp(g.detach() / .86))
    assert bool(((actual >= -.86) & (actual <= 0)).all())
    assert soft_clamp_log_gates(g.detach().bfloat16()).dtype == torch.float32


@pytest.mark.parametrize('bound', [0, -1, float('inf'), float('nan')])
def test_invalid_bound(bound):
    with pytest.raises(ValueError):
        soft_clamp_log_gates(torch.zeros(1), bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('bt,bs', [(64,32), (64,64), (128,32), (128,64)])
@pytest.mark.parametrize('lengths', [(129,), (257,), (1025,), (129,137)])
def test_soft_gates_partial_tiles(monkeypatch, bt, bs, lengths):
    # Exercise every supported tile shape rather than whichever autotuning selects.
    for kernel in (training.parallel_wall_attn_fwd_kernel, training.parallel_wall_attn_bwd_kernel_dq, training.parallel_wall_attn_bwd_kernel_dkv):
        monkeypatch.setattr(kernel, 'configs', [triton.Config(dict(BT=bt, BS=bs), num_warps=4, num_stages=2)])
        monkeypatch.setattr(kernel, 'cache', {})
    torch.manual_seed(321)
    t = sum(lengths)
    q0 = torch.randn(1, t, 2, 16, device='cuda', dtype=torch.bfloat16)
    k0 = torch.randn(1, t, 1, 16, device='cuda', dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    probe = torch.randn_like(q0, dtype=torch.float32)
    kwargs = dict(scale=.25)
    if len(lengths) > 1:
        kwargs.update(cu_seqlens=torch.tensor([0,129,t], device='cuda', dtype=torch.long), window_size=17,
                      sink_bias=torch.zeros(2, device='cuda'),
                      g_scalar=torch.full((1,t,2), -.01, device='cuda'))
    def run(fn):
        q, k, v = [x.clone().requires_grad_() for x in (q0,k0,v0)]
        logits = torch.full(q.shape, -8., device='cuda', requires_grad=True)
        g = soft_clamp_log_gates(F.logsigmoid(logits))
        g.retain_grad()
        out = fn(q,k,v,g,**kwargs)
        (out.float()*probe).sum().backward()
        return (out, q.grad,k.grad,v.grad,g.grad,logits.grad)
    actual, expected = run(wall_attn), run(wall_attn_reference)
    for i, (a,e) in enumerate(zip(actual,expected)):
        assert bool(torch.isfinite(a).all())
        assert bool(torch.isfinite(e).all())
        tol = .04 if i == 0 else .06
        torch.testing.assert_close(a.float(),e.float(),atol=tol,rtol=tol)
