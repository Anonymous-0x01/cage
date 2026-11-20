"""Stable Diffusion VAE decoder mid-block self-attention (SD 1.5 AutoencoderKL).

Source: diffusers 0.43.0, models/attention_processor.py: Attention (group_norm, to_q/k/v, to_out,
residual_connection, rescale_output_factor) with the default AttnProcessor2_0.__call__ (verbatim for a
4-D input without encoder_hidden_states / mask / spatial_norm).  UNetMidBlock2D builds it with
query_dim 512, heads 1 (dim_head 512), norm_num_groups 32, eps 1e-6, bias True, residual_connection True,
rescale_output_factor 1.0.  Input: 4 latents at 64x64 (512x512 images), fp16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, CHANNELS, RES, HEADS, GROUPS, EPS = 4, 512, 64, 1, 32, 1e-6
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, channels: int, heads: int, groups: int):
        super().__init__()
        self.heads = heads
        self.rescale_output_factor = 1.0
        self.residual_connection = True
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=groups, eps=EPS, affine=True, dtype=DTYPE)
        self.to_q = nn.Linear(channels, channels, bias=True, dtype=DTYPE)
        self.to_k = nn.Linear(channels, channels, bias=True, dtype=DTYPE)
        self.to_v = nn.Linear(channels, channels, bias=True, dtype=DTYPE)
        self.to_out = nn.Linear(channels, channels, bias=True, dtype=DTYPE)
        with torch.no_grad():
            self.group_norm.weight.normal_(1.0, 0.1)
            self.group_norm.bias.normal_(0.0, 0.1)

    def forward(self, hidden_states):
        residual = hidden_states
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = self.to_q(hidden_states)
        encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                                       is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out(hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if self.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / self.rescale_output_factor
        return hidden_states


def get_init_inputs():
    return [CHANNELS, HEADS, GROUPS]


def get_inputs():
    return [torch.randn(BATCH, CHANNELS, RES, RES, dtype=DTYPE, device="cuda")]
