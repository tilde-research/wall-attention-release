# Copyright (c) 2026 Tilde Research.
"""Eager PyTorch reference for Wall attention (correctness oracle)."""

import torch

from fla.ops.utils.constant import RCP_LN2


def _prefix(g: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    """Differentiable FP32 prefixes, reset at each packed sequence boundary."""
    if cu_seqlens is None:
        return g.float().cumsum(dim=1) * RCP_LN2
    if g.shape[0] != 1:
        raise ValueError("reference varlen expects batch size 1")
    boundaries = cu_seqlens.tolist()
    return torch.cat([
        g[:, a:b].float().cumsum(dim=1)
        for a, b in zip(boundaries[:-1], boundaries[1:])
    ], dim=1) * RCP_LN2


def wall_attention_reference_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    sink_bias: torch.Tensor | None = None,
    g_scalar: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Eager Wall forward (exact logits): per pair :math:`(i,j)`,
    :math:`s_{ij} = \sum_n q_{in} k_{jn} \exp_2(P_{in}-P_{jn}) \cdot \mathrm{scale}\cdot\mathrm{RCP\_LN2}`
    with :math:`P` the prefix cumsum of ``g`` in log₂ (same as the Triton path)."""
    if g.dim() != 4 or g.shape != (*q.shape[:-1], q.shape[-1]):
        raise ValueError("`g` must be [B, T, HQ, K] matching `q`")
    G = q.shape[2] // k.shape[2]
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("HQ must be divisible by H (GQA)")

    P = _prefix(g, cu_seqlens)
    k_exp = k.repeat_interleave(G, dim=2)
    B, T, HQ, _K = q.shape
    i_idx = torch.arange(T, device=q.device, dtype=torch.long).view(1, T, 1)
    j_idx = torch.arange(T, device=q.device, dtype=torch.long).view(1, 1, T)
    valid = j_idx <= i_idx
    if window_size is not None:
        valid = valid & (i_idx - j_idx < window_size)
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("reference varlen expects batch size 1")
        seg = torch.zeros(T, dtype=torch.long, device=q.device)
        for s in range(int(cu_seqlens.numel()) - 1):
            a = int(cu_seqlens[s].item())
            b = int(cu_seqlens[s + 1].item())
            seg[a:b] = s
        seg_i = seg.view(1, T, 1)
        seg_j = seg.view(1, 1, T)
        valid = valid & (seg_i == seg_j)

    diff = P.unsqueeze(2).float() - P.unsqueeze(1).float()
    diff = diff.masked_fill(~valid[..., None, None], float("-inf"))
    scores = (
        q.unsqueeze(2).float() * k_exp.unsqueeze(1).float() * torch.exp2(diff)
    ).sum(-1).permute(0, 3, 1, 2).contiguous()

    valid_hq = valid.unsqueeze(1).expand(B, HQ, T, T)
    scores = scores.masked_fill(~valid_hq, float("-inf")) * (scale * RCP_LN2)

    if g_scalar is not None:
        c = _prefix(g_scalar, cu_seqlens)
        c_hq = c.permute(0, 2, 1).float()
        scores = scores + c_hq.unsqueeze(-1) - c_hq.unsqueeze(-2)

    m = scores.amax(dim=-1, keepdim=True)
    m_stable = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp2(scores - m_stable)
    den = p.sum(dim=-1)
    if sink_bias is not None:
        sink_l2 = (sink_bias * RCP_LN2).view(1, HQ, 1)
        den = den + torch.exp2(sink_l2 - m_stable.squeeze(-1))
    w = p / den.unsqueeze(-1)
    v_h = v.repeat_interleave(G, dim=2).permute(0, 2, 1, 3).contiguous()
    o = torch.matmul(w, v_h.float()).transpose(1, 2).contiguous()
    return o


wall_attn_reference = wall_attention_reference_torch
