# Transformation cookbook (allowed actions)

All rewrites are plain PyTorch in the candidate's `Model.forward` (plus `__init__`
parameters and `convert_weights`). TorchInductor generates every kernel.

## Candidate file skeleton
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# optional backend knobs (whitelisted keys only; see cage/task.py INDUCTOR_CONFIG_WHITELIST)
INDUCTOR_CONFIG = {"coordinate_descent_tuning": True}
COMPILE_MODE = None   # or "max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead"

class Model(nn.Module):
    def __init__(self, <same args as G0>):
        super().__init__()
        ...  # parameters in the layout this candidate wants

    def forward(self, <same inputs as G0>):
        ...  # must return the same structure/shapes/dtypes as G0

def convert_weights(ref_state_dict):      # only if parameter names/layouts differ from G0
    sd = dict(ref_state_dict)
    sd["gate_up"] = torch.cat([sd.pop("gate_proj.weight"), sd.pop("up_proj.weight")], 0)
    return sd
```
`convert_weights` runs once, offline, and is not timed. It is how you express weight
pre-layout transformations: concatenation, transposition, folding a per-channel scale into
a weight, and casting.

## Graph rewrite patterns
- **Merge parallel GEMMs**: `F.linear(x, W_cat)` then `.split(sizes, -1)` (views, free).
- **Fold a scale into weights**: `(w * x) @ W.T == x @ (W * w).T` for a per-input-channel
  `w`. Valid when `w` multiplies the GEMM's K dimension. Do it in `convert_weights`.
- **Avoid materialized transposes**: let GEMMs consume transposed views (`x @ W.T`,
  `torch.mm(a, b.t())`), and produce outputs directly in the consumer layout (reshape the
  input before the op instead of permuting the result). `.contiguous()` forces a copy kernel
  unless Inductor fuses it into the producer.
- **Keep reductions on the contiguous dim** so the reduced dim is innermost.
- **Casting**: an fp32 upcast inside a fused kernel is free (registers). It only costs when
  the fp32 tensor is materialized between kernels. Make sure the consumer fuses with the cast.
- **Batch small GEMMs**: reshape [B, S, K] to [B*S, K] for `mm` instead of `bmm`/`einsum`
  over many small matrices; use `baddbmm`/`addmm` to fold bias and scale.
- **Attention**: `F.scaled_dot_product_attention(q, k, v, attn_mask=..., is_causal=..., scale=...)`
  is a graph-level op (flash/efficient kernels) and is allowed. GQA: `enable_gqa=True`, or
  expand KV heads with views instead of `repeat_interleave` copies.
- **Normalizations**: `F.rms_norm` / `F.layer_norm` are fine, but Inductor decomposes them;
  explicit fp32 formulas usually fuse equally well. Fusing the residual add into the norm
  kernel is a common win.
- **Pooling / windowed ops**: overlapping windows fused with an expensive producer cause
  recomputation; materializing the producer once can be faster.
- **Top-k / sort**: these are extern fallbacks (kernel boundaries). Reduce the data they
  see, for example top-k on a pre-reduced score, or a smaller dtype if exact.
- **Indexing / gather**: prefer `index_select`/`gather` on the smallest tensor; avoid
  boolean-mask indexing (dynamic shape leads to graph breaks).

## Inductor knobs (backend-level actions)
Use them after the structure is right. They are cheap to try, but eval them.
- `COMPILE_MODE="max-autotune-no-cudagraphs"`: autotunes GEMM backends (cuBLAS vs Triton
  templates) and enables **epilogue fusion** into Triton GEMM templates. Compile is slower.
- `COMPILE_MODE="reduce-overhead"`: CUDA graphs. Helps when launch-bound.
- `{"coordinate_descent_tuning": True}`: tunes block sizes of generated kernels.
- `{"triton.persistent_reductions": True/False}`, `{"split_reductions": False}`: reduction strategy.
- `{"epilogue_fusion": True}`, `{"aggressive_fusion": True}`, `{"max_fusion_size": 128}`: fusion aggressiveness.
- `{"combo_kernels": True}`: horizontal fusion of independent kernels.
- `{"shape_padding": True}`, `{"comprehensive_padding": True}`: pad GEMM shapes for alignment.
- `{"b2b_gemm_pass": True}`: back-to-back GEMM fusion (motivating example in the CAGE paper).

## Things the legality checker rejects
`import triton`, CUDA/C++ extensions, `torch.library` custom ops, `torch.compile` or
`torch._dynamo`/`torch._inductor` inside the candidate, `global` state, assigning
`self.<attr>` in `forward`, data-dependent graph breaks, output shape/dtype changes, and
numerics outside the task tolerance that are also less accurate than G0 against a float64
oracle.
