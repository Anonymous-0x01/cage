"""LLaMA-3-8B gated MLP with LoRA adapters (PEFT-style) on gate/up/down projections, 2048 tokens.

Each projection is base(x) + scaling * lora_B(lora_A(x)) with rank 16, alpha 32 (dropout off).
Cross-operator structure in G0: six skinny rank-16 GEMMs next to the big ones, gate/up share
their input (and their LoRA A input), the adapter could be merged into the base weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKENS, HIDDEN, INTER, RANK, ALPHA = 2048, 4096, 14336, 16, 32
DTYPE = torch.float16


class LoraLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, alpha: float):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=False, dtype=DTYPE)
        self.lora_A = nn.Linear(d_in, rank, bias=False, dtype=DTYPE)
        self.lora_B = nn.Linear(rank, d_out, bias=False, dtype=DTYPE)
        self.scaling = alpha / rank
        with torch.no_grad():
            self.base.weight.normal_(0, 0.02)
            self.lora_B.weight.normal_(0, 0.02)   # trained adapter (PEFT initializes B to zero)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


class Model(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = LoraLinear(hidden, inter, RANK, ALPHA)
        self.up_proj = LoraLinear(hidden, inter, RANK, ALPHA)
        self.down_proj = LoraLinear(inter, hidden, RANK, ALPHA)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def get_init_inputs():
    return [HIDDEN, INTER]


def get_inputs():
    return [torch.randn(1, TOKENS, HIDDEN, dtype=DTYPE, device="cuda")]
