"""Swin Transformer shifted-window block (Swin-T, stage 1, second block: shift_size = 3).

Source: transformers 5.23.0, models/swin/modeling_swin.py: SwinLayer.forward, SwinAttention,
SwinRelativePositionBias, eager_attention_forward, window_partition / window_reverse,
get_attn_mask / cyclic_shift / maybe_pad (forward code verbatim; dropout / drop-path are identity at
inference).  Config microsoft/swin-tiny-patch4-window7-224: embed_dim 96, num_heads 3 (stage 1),
window_size 7, mlp_ratio 4, qkv_bias True, layer_norm_eps 1e-5, hidden_act gelu.
Input: batch 64, 56x56 tokens, fp16.
"""

import torch
import torch.nn as nn

BATCH, RES, DIM, HEADS, WINDOW, SHIFT, MLP_RATIO, EPS = 64, 56, 96, 3, 7, 3, 4.0, 1e-5
DTYPE = torch.float16


class SwinRelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, window_size: tuple[int, int]):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.relative_position_bias_table = nn.Parameter(
            0.02 * torch.randn((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads, dtype=DTYPE)
        )
        self.register_buffer("relative_position_index", self._create_relative_position_index().view(-1),
                             persistent=False)

    def _create_relative_position_index(self) -> torch.Tensor:
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        return relative_coords.sum(-1)

    def forward(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]
        relative_position_bias = relative_position_bias.view(self.window_area, self.window_area, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)


def eager_attention_forward(query, key, value, attention_mask, scaling):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class SwinAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, window_size: int):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=DTYPE)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=DTYPE)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=DTYPE)
        self.o_proj = nn.Linear(hidden_size, hidden_size, dtype=DTYPE)
        self.relative_position_bias = SwinRelativePositionBias(num_attention_heads, (window_size, window_size))

    def forward(self, hidden_states, attention_mask=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        relative_position_bias = self.relative_position_bias()
        if attention_mask is not None:
            num_windows = attention_mask.shape[0]
            batch_size = input_shape[0] // num_windows
            seq_len = input_shape[1]
            attention_mask = (
                attention_mask.unsqueeze(1).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
                .reshape(-1, 1, seq_len, seq_len)
            )
            combined_mask = relative_position_bias + attention_mask
        else:
            combined_mask = relative_position_bias

        attn_output, _ = eager_attention_forward(query_states, key_states, value_states, combined_mask, self.scaling)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


def window_partition(input_feature, window_size):
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    return input_feature.transpose(2, 3).contiguous().view(-1, window_size, window_size, num_channels)


def window_reverse(windows, window_size, height, width):
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    return windows.transpose(2, 3).contiguous().view(-1, height, width, num_channels)


class Model(nn.Module):
    def __init__(self, dim: int, resolution: int, num_heads: int, window_size: int, shift_size: int):
        super().__init__()
        self.attention = SwinAttention(dim, num_heads, window_size)
        self.layernorm_before = nn.LayerNorm(dim, eps=EPS, dtype=DTYPE)
        self.layernorm_after = nn.LayerNorm(dim, eps=EPS, dtype=DTYPE)
        self.fc1 = nn.Linear(dim, int(MLP_RATIO * dim), dtype=DTYPE)
        self.fc2 = nn.Linear(int(MLP_RATIO * dim), dim, dtype=DTYPE)
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = (resolution, resolution)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape

        hidden_states_windows = window_partition(self.cyclic_shift(hidden_states), self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype,
                                       device=hidden_states_windows.device)

        attention_output = self.attention(hidden_states_windows, attn_mask)

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        attention_windows = self.cyclic_shift(
            window_reverse(attention_windows, self.window_size, height_pad, width_pad), reverse=True
        )
        if pad_values[3] > 0 or pad_values[5] > 0:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = shortcut + attention_windows

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.fc2(nn.functional.gelu(self.fc1(hidden_states)))
        return hidden_states + residual

    def get_attn_mask(self, height, width, dtype, device):
        if self.shift_size <= 0:
            return None
        h_idx = torch.arange(height, device=device)
        w_idx = torch.arange(width, device=device)
        h_region = (h_idx >= height - self.window_size).long() + (h_idx >= height - self.shift_size).long()
        w_region = (w_idx >= width - self.window_size).long() + (w_idx >= width - self.shift_size).long()
        img_mask = (h_region[None, :, None, None] * 3 + w_region[None, None, :, None]).to(dtype)
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def cyclic_shift(self, hidden_states, reverse=False):
        if self.shift_size > 0:
            direction = 1 if reverse else -1
            hidden_states = torch.roll(hidden_states, shifts=(direction * self.shift_size, direction * self.shift_size),
                                       dims=(1, 2))
        return hidden_states


def get_init_inputs():
    return [DIM, RES, HEADS, WINDOW, SHIFT]


def get_inputs():
    return [torch.randn(BATCH, RES * RES, DIM, dtype=DTYPE, device="cuda")]
