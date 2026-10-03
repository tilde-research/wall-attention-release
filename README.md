# Wall Attention
Wall Attention is an attention variant with a **per-channel, per-timestep multiplicative decay** baked into the QK inner product. Where standard attention scores a pair $(i, j)$ with $\sum_n q_{i,n}\, k_{j,n}$, Wall Attention weights each channel $n$ by a learned decay accumulated between the two positions. This gives each query channel an independent, content-dependent forgetting rate, generalizing scalar gating (FoX) and RoPE-style decays to the full channel dimension. Setting $g = 0$ recovers vanilla softmax attention.

See the blog for more information: 
https://blog.tilderesearch.com/blog/wall-attn  

This repo packages the two kernels used in practice, each on its own:

- **Training / prefill** (`wall_attn`): a fused forward + backward Triton kernel (FlashAttention-style streaming softmax) with analytic gradients for $q, k, v, g$.
- **Decode** (`wall_attn_decode`): a single-step kernel that reads a **pre-rescaled KV cache**, so per-token generation costs one small GEMV-like pass instead of recomputing the prefix.

## Installation

```bash
# Using uv (recommended)
uv sync
source .venv/bin/activate

# or with pip
pip install -e .
```

## Usage

### Training / prefill

```python
import torch
from wall_attn import wall_attn, soft_clamp_log_gates

B, T, H, HQ, K, V = 2, 1024, 4, 8, 64, 64  # GQA: HQ query heads, H kv heads
q = torch.randn(B, T, HQ, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H,  K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H,  V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
gate_logits = torch.randn(B, T, HQ, K, device="cuda", dtype=torch.float32, requires_grad=True)
g = soft_clamp_log_gates(torch.nn.functional.logsigmoid(gate_logits), g_max=0.86)

o = wall_attn(q, k, v, g, scale=K**-0.5)  # [B, T, HQ, V]
o.sum().backward()
```

Optional arguments: `g_scalar` ([B, T, HQ] FoX-style additive gate), `sink_bias` ([HQ] attention sink), `window_size` (sliding window), and `cu_seqlens` (varlen packing, requires `B == 1`).

### Decode (cached generation)

Build the pre-rescaled cache once at prefill, then decode one token at a time:

```python
import torch
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_global_cumsum
from wall_attn import build_wall_kv_cache, wall_attn_decode

C = 64                                  # cache chunk size (anchor granularity)
P = chunk_global_cumsum(g, scale=RCP_LN2)              # [B, T, HQ, K] prefix
k_tilde, r_cache = build_wall_kv_cache(k, P, chunk_size=C)

o, _ = wall_attn_decode(
    q=q[:, -1:],                        # current query [B, 1, HQ, K]
    v=v,                                # cached values [B, T_kv, H, V]
    p_curr=P[:, -1:],                   # prefix at the current row
    k_tilde=k_tilde,                    # pre-rescaled keys [B, T_kv, HQ, K]
    r_cache=r_cache,                    # per-chunk anchors [B, ceil(T_kv/C), HQ, K]
    sink_bias=None,
    scale=K**-0.5,
    cache_chunk_size=C,
)
```

`build_wall_kv_cache` folds the decay into the keys (`k_tilde[j] = k[j] · exp2(R_c − P[j])`) using a per-chunk anchor `R_c`, so the decode kernel never re-accumulates the prefix. See `tests/test_decode.py::test_decode_streaming_matches_full_forward` for the full append-as-you-go serving loop.

## Code structure

```
wall_attn/
├── __init__.py    # public API
├── training.py    # forward/backward Triton kernels + autograd Function + wall_attn()
├── decode.py      # single-step decode kernel + build_wall_kv_cache()
└── reference.py   # eager PyTorch reference (correctness oracle)
tests/
├── test_training.py   # parity + analytic gradients (finite-difference checked)
└── test_decode.py     # decode == prefill forward, streaming, cache shapes
```

## Features

- **GQA**: query heads `HQ` may exceed kv heads `H` (`HQ % H == 0`).
- **Per-channel decay `g`** with exact analytic gradient, plus an optional **scalar gate** `g_scalar`.
- **Attention sink** (`sink_bias`), **sliding window** (`window_size`), and **varlen packing** (`cu_seqlens`).
- **Pre-rescaled decode cache** for cheap autoregressive generation, numerically stable to long context (per-chunk anchors keep `exp2` bounded).
- BF16/FP32 inputs; autotuned block sizes for Hopper / Ampere.

## Testing

```bash
pytest                 # requires a CUDA GPU
```

Every kernel path is checked against the eager `wall_attn_reference`, and the `g` / `g_scalar` gradients are verified against central finite differences. The decode kernel is checked to reproduce the training forward token-for-token, including a streaming generation loop.

## Acknowledgments

The Triton kernels build on the parallel-attention machinery from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT). We thank the FLA team for their excellent work on efficient attention.

## License

MIT, see [LICENSE](LICENSE).
