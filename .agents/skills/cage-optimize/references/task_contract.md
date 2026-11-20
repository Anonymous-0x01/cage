# Task contract (how to turn model code into G0)

A task file is a self-contained Python module:

```python
import torch
import torch.nn as nn

DTYPE = torch.float16
TOKENS, HIDDEN = 1024, 4096          # realistic LLM shapes

class Model(nn.Module):
    def __init__(self, hidden):       # args come from get_init_inputs()
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden, dtype=DTYPE))   # create params in the target dtype
    def forward(self, x):             # args come from get_inputs()
        ...
        return out                    # tensor or tuple of tensors

def get_init_inputs():
    return [HIDDEN]

def get_inputs():                     # random inputs on CUDA; called under a fixed seed
    return [torch.randn(1, TOKENS, HIDDEN, dtype=DTYPE, device="cuda")]

TOLERANCE = {"atol": 1e-2, "rtol": 1e-2}   # optional; defaults by dtype
```

Rules for extracting a subgraph from LLM code (Hugging Face style modules, etc.):
1. **Keep semantics exact**: copy the math of the original forward (dtype casts, eps,
   scaling, masking). Don't optimize while extracting; the extracted file is G0.
2. **Inline dependencies**: config objects become constants; helper modules become
   `nn.Linear`/`nn.Parameter`; remove caching, dropout (eval mode), hooks, and logging.
3. **Weights**: parameters must be created in `__init__` from the init args and have
   realistic magnitudes. Weights are random, but scale them (e.g. `normal_(0, 0.02)`) so
   fp16 activations don't overflow.
4. **Inputs**: everything the subgraph reads at runtime (hidden states, position ids or
   cos/sin, masks, KV cache slices) comes from `get_inputs()` on `"cuda"`. Integer inputs
   use `torch.randint`. Masks must be valid (e.g. causal).
5. **Shapes**: use the user's batch, sequence and model dims. Default: batch 1, sequence
   1024, the model's hidden/intermediate/head dims, FP16.
6. **Self-check**: `bin/cage init` statically validates the file and verifies it compiles
   and matches itself. Also run `bin/cage ctx graph --why "verify extraction"` and check
   the shapes are what you intended.

Typical LLM subgraphs: RMSNorm(+residual), QKV projection + QK-norm + RoPE, GQA attention
(prefill), O-projection + residual, gated MLP (SwiGLU), LoRA-augmented linear, logits +
sampling top-k.
