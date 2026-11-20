"""LLaMA-3 LoRA MLP candidate: adapters merged into the base weights + one gate/up GEMM.

base(x) + s * B(A(x)) = x (W + s * B A)^T, so each adapter is folded into its base weight
offline (computed in fp32, stored in fp16), and gate/up, which share their input, become one GEMM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

RANK, ALPHA = 16, 32
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.inter = inter
        self.gate_up = nn.Parameter(torch.empty(2 * inter, hidden, dtype=DTYPE))
        self.down = nn.Parameter(torch.empty(hidden, inter, dtype=DTYPE))

    def forward(self, x):
        gate, up = F.linear(x, self.gate_up).split(self.inter, dim=-1)
        return F.linear(F.silu(gate) * up, self.down)


def convert_weights(sd):
    scaling = ALPHA / RANK

    def merged(name):
        w = sd[f"{name}.base.weight"].float()
        return (w + scaling * sd[f"{name}.lora_B.weight"].float() @ sd[f"{name}.lora_A.weight"].float()).to(DTYPE)

    return {"gate_up": torch.cat([merged("gate_proj"), merged("up_proj")], 0), "down": merged("down_proj")}
