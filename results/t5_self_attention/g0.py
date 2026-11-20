"""T5 encoder self-attention layer with relative position bias (T5-base, first encoder layer).

Source: transformers 5.23.0, models/t5/modeling_t5.py: T5LayerSelfAttention.forward, T5LayerNorm,
T5Attention.forward / compute_bias / _relative_position_bucket, eager_attention_forward (verbatim;
encoder: bidirectional buckets, no padding mask, scaling 1.0, dropout identity).
Config google-t5/t5-base: d_model 768, d_kv 64, num_heads 12, relative_attention_num_buckets 32,
relative_attention_max_distance 128, layer_norm_epsilon 1e-6, bf16.  Batch 16, 512 tokens.
"""

import math

import torch
import torch.nn as nn

BATCH, SEQ, D_MODEL, D_KV, N_HEADS, N_BUCKETS, MAX_DIST, EPS = 16, 512, 768, 64, 12, 32, 128, 1e-6
DTYPE = torch.bfloat16


def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )
    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class Model(nn.Module):
    def __init__(self, d_model: int, d_kv: int, n_heads: int):
        super().__init__()
        self.key_value_proj_dim = d_kv
        self.n_heads = n_heads
        self.inner_dim = n_heads * d_kv
        self.layer_norm_weight = nn.Parameter(1 + 0.02 * torch.randn(d_model, dtype=DTYPE))
        self.q = nn.Linear(d_model, self.inner_dim, bias=False, dtype=DTYPE)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False, dtype=DTYPE)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False, dtype=DTYPE)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False, dtype=DTYPE)
        self.relative_attention_bias = nn.Embedding(N_BUCKETS, n_heads, dtype=DTYPE)

    def layer_norm(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + EPS)
        hidden_states = hidden_states.to(self.layer_norm_weight.dtype)
        return self.layer_norm_weight * hidden_states

    def compute_bias(self, query_length, key_length, device):
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = _relative_position_bucket(
            relative_position, bidirectional=True, num_buckets=N_BUCKETS, max_distance=MAX_DIST
        )
        values = self.relative_attention_bias(relative_position_bucket)
        return values.permute([2, 0, 1]).unsqueeze(0)

    def forward(self, hidden_states):
        normed_hidden_states = self.layer_norm(hidden_states)

        input_shape = normed_hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.key_value_proj_dim)
        query_states = self.q(normed_hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k(normed_hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v(normed_hidden_states).view(hidden_shape).transpose(1, 2)
        position_bias = self.compute_bias(input_shape[1], key_states.shape[-2], device=query_states.device)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * 1.0
        attn_weights = attn_weights + position_bias
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attention_output = self.o(attn_output)
        return hidden_states + attention_output


def get_init_inputs():
    return [D_MODEL, D_KV, N_HEADS]


def get_inputs():
    return [torch.randn(BATCH, SEQ, D_MODEL, dtype=DTYPE, device="cuda")]
