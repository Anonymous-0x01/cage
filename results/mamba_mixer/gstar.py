"""Mamba mixer, G*: the sequential selective scan as a parallel associative scan.

h_t = dA_t * h_{t-1} + dBu_t is a first-order linear recurrence, i.e. an associative scan with
combine((a1, b1), (a2, b2)) = (a1 * a2, a2 * b1 + b2).  The 512-step Python loop (512 x [elementwise +
bmm] launches) becomes one scan over the sequence dimension -- the same reformulation transformers
ships as its `use_associative_scan` path -- and the C-contraction is one batched reduction.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._higher_order_ops.associative_scan import associative_scan

CONV = 4
DTYPE = torch.bfloat16


def combine_fn(left, right):
    a_left, b_left = left
    a_right, b_right = right
    return a_left * a_right, a_right * b_left + b_right


class Model(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, state_size: int, time_step_rank: int):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.ssm_state_size = state_size
        self.time_step_rank = time_step_rank
        self.conv1d = nn.Conv1d(intermediate_size, intermediate_size, bias=True, kernel_size=CONV,
                                groups=intermediate_size, padding=CONV - 1, dtype=DTYPE)
        self.in_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False, dtype=DTYPE)
        self.x_proj = nn.Linear(intermediate_size, time_step_rank + state_size * 2, bias=False, dtype=DTYPE)
        self.dt_proj = nn.Linear(time_step_rank, intermediate_size, bias=True, dtype=DTYPE)
        self.out_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=DTYPE)
        self.A_log = nn.Parameter(torch.empty(intermediate_size, state_size))
        self.D = nn.Parameter(torch.empty(intermediate_size))

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        seq_len = hidden_states.shape[1]
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        x, gate = projected_states.chunk(2, dim=1)
        x = F.silu(F.conv1d(x, self.conv1d.weight, self.conv1d.bias, padding=CONV - 1,
                            groups=self.intermediate_size)[:, :, :seq_len])               # [B, I, T]
        time_step, B, C = torch.split(self.x_proj(x.transpose(1, 2)),
                                      [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        dt = self.dt_proj.weight @ time_step.transpose(1, 2)
        dt = F.softplus(dt + self.dt_proj.bias.float().to(dt.dtype)[..., None])
        discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])                      # [B, I, T, N]
        deltaB_u = dt[:, :, :, None] * B[:, None, :, :].float() * x[:, :, :, None].float()
        _, states = associative_scan(combine_fn, (discrete_A, deltaB_u), dim=2, combine_mode="pointwise")
        y = (states.to(dtype).float() * C[:, None, :, :].float()).sum(-1).to(dtype)        # [B, I, T]
        y = y + x * self.D.float()[None, :, None]
        y = y * F.silu(gate)
        return self.out_proj(y.transpose(1, 2).to(dtype))
