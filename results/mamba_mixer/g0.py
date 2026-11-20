"""Mamba selective-SSM mixer, full-sequence forward with the PyTorch fallback (Mamba-130m).

Source: transformers 5.23.0, models/mamba/modeling_mamba.py: MambaMixer.forward (no cache, no padding
mask, inference), causal_conv1d_fn, mamba_selective_scan (sequential "Recurrent iteration" branch, the
path taken without the mamba_ssm / causal_conv1d kernels and with use_mambapy = use_associative_scan =
False, the config defaults).  Code is verbatim apart from inlining.
Config state-spaces/mamba-130m-hf: hidden 768, intermediate 1536, state 16, time_step_rank 48,
conv_kernel 4, use_conv_bias True, use_bias False, hidden_act silu.  bf16 weights (A_log, D in fp32).
Batch 4, 512 tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, SEQ, HIDDEN, INTER, STATE, DT_RANK, CONV = 4, 512, 768, 1536, 16, 48, 4
DTYPE = torch.bfloat16


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(hidden_states.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias, padding=padding,
                   groups=hidden_size)[:, :, :seq_len]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def mamba_selective_scan(hidden_states, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    batch_size, intermediate_size, seq_len = hidden_states.shape
    input_dtype = hidden_states.dtype
    if delta_bias is not None:
        dt = dt + delta_bias.to(dt.dtype)[..., None]
    if delta_softplus:
        dt = F.softplus(dt)
    B = B.transpose(1, 2)
    C = C.transpose(1, 2)
    discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
    discrete_B = dt[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    ssm_state = torch.zeros(batch_size, intermediate_size, A.shape[-1], dtype=input_dtype,
                            device=hidden_states.device)
    scan_outputs = []
    for index in range(seq_len):
        ssm_state = discrete_A[:, :, index] * ssm_state + deltaB_u[:, :, index]
        scan_output = torch.matmul(ssm_state.to(input_dtype), C[:, index, :].unsqueeze(-1))
        scan_outputs.append(scan_output[:, :, 0])
    scan_output = torch.stack(scan_outputs, dim=-1)
    if D is not None:
        scan_output = scan_output + hidden_states * D[None, :, None]
    if z is not None:
        scan_output = scan_output * F.silu(z)
    return scan_output


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
        A = torch.arange(1, state_size + 1, dtype=torch.float32)[None, :].expand(intermediate_size, -1)
        self.A_log = nn.Parameter(torch.log(A).contiguous())
        self.D = nn.Parameter(torch.ones(intermediate_size))
        with torch.no_grad():   # MambaMixer.init_mamba_weights (time_step_min 1e-3, max 0.1, floor 1e-4)
            dt_init_std = time_step_rank ** -0.5
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)
            dt = torch.exp(torch.rand(intermediate_size) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            dt = dt.clamp(min=1e-4)
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        hidden_states_B_C, gate = projected_states.chunk(2, dim=1)
        hidden_states_B_C = causal_conv1d_fn(hidden_states_B_C, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                                             activation="silu")
        hidden_states_B_C = hidden_states_B_C.transpose(1, 2)
        time_step, B, C = torch.split(self.x_proj(hidden_states_B_C),
                                      [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
        time_proj_bias = self.dt_proj.bias.float()
        scan_output = mamba_selective_scan(hidden_states_B_C.transpose(1, 2), time_step, A, B.transpose(1, 2),
                                           C.transpose(1, 2), D=self.D.float(), z=gate, delta_bias=time_proj_bias,
                                           delta_softplus=True)
        return self.out_proj(scan_output.transpose(1, 2).to(dtype))


def get_init_inputs():
    return [HIDDEN, INTER, STATE, DT_RANK]


def get_inputs():
    return [torch.randn(BATCH, SEQ, HIDDEN, dtype=DTYPE, device="cuda")]
