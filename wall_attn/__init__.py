# Copyright (c) 2026 Tilde Research.
"""Wall Attention: per-channel multiplicative-decay attention kernels.

Public API:
    wall_attn            - training / prefill forward+backward (autograd).
    wall_attn_decode     - single-step decode against a pre-rescaled KV cache.
    build_wall_kv_cache  - build the pre-rescaled KV cache from keys + prefix.
    wall_attn_reference  - eager PyTorch reference (correctness oracle).
"""

from wall_attn.gates import soft_clamp_log_gates
from wall_attn.decode import build_wall_kv_cache, wall_attn_decode
from wall_attn.reference import wall_attn_reference
from wall_attn.training import wall_attn

__all__ = [
    "wall_attn",
    "soft_clamp_log_gates",
    "wall_attn_decode",
    "build_wall_kv_cache",
    "wall_attn_reference",
]
