"""Shared test helpers."""

import torch
import torch.nn.functional as F

from wall_attn import soft_clamp_log_gates


def make_log_gates(*shape, device, mean_logit: float = 2.5, max_logit: float = 4.5) -> torch.Tensor:
    """Prepared log retention gates, built the way a model produces them.

    ``mean_logit=2.5`` keeps per-token retention ~0.9
    
    ``max_logit=4.5`` bounds the result to ``g <= -0.011`` so finite-difference perturbations of ``eps = 3e-3`` never push a gate positive. 
    
    Only the two tests need ``max_logit=4.5``:
    (``test_training.py::test_g_gradient_matches_finite_differences`` and ``test_scalar_gate_gradient_finite_differences``)
    These tests the forward at ``g +- eps``, and the kernel is only valid for ``g <= 0``. 
    Other tests never perturb ``g``.
    
    Returns float32.
    """
    logits = (torch.randn(*shape, device=device) + mean_logit).clamp(max=max_logit)
    return soft_clamp_log_gates(F.logsigmoid(logits))
