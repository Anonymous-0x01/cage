"""Legal rewrite of tiny_mlp: one merged gate/up GEMM (weights concatenated offline)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.inter = inter
        self.gate_up = nn.Parameter(torch.empty(2 * inter, hidden, dtype=DTYPE))
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=DTYPE)

    def forward(self, x):
        gate, up = F.linear(x, self.gate_up).split(self.inter, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def convert_weights(sd):
    return {
        "gate_up": torch.cat([sd["gate_proj.weight"], sd["up_proj.weight"]], dim=0),
        "down_proj.weight": sd["down_proj.weight"],
    }
