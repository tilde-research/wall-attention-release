"""Gate preparation shared by training, prefill, and decode."""

import math

import torch


def soft_clamp_log_gates(log_gates: torch.Tensor, g_max: float = 0.86) -> torch.Tensor:
    """Soft-bound non-positive natural-log gates to [-g_max, 0].

    Apply once to ``F.logsigmoid(logits.float())`` before ``wall_attn`` or building decode prefixes. 
    Inputs must be log-retentions (<= 0), not logits!

    Follows Wall Attention blog post (https://blog.tilderesearch.com/blog/wall-attn).
    Appendix G derives g_max < 0.87 (B_T=128, B_S=64).
    We keep the default g_max as 0.86 to be safe.
    """
    if not math.isfinite(g_max) or g_max <= 0:
        raise ValueError("g_max must be finite and positive")
    gates = log_gates if log_gates.dtype == torch.float64 else log_gates.float()
    return g_max * torch.expm1(gates / g_max)
