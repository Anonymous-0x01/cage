"""BLOOM self-attention half-block with ALiBi (BLOOM-7B1), prefill.

Source: transformers 5.23.0, models/bloom/modeling_bloom.py: BloomModel.forward (unpadded alibi:
`slopes[:, None, None] * arange(kv_length)`, repeated over the batch), _get_alibi_slopes,
BloomBlock.forward (input_layernorm, residual = hidden_states), BloomAttention.forward / _reshape /
_merge_heads (pretraining_tp = 1), dropout_add at inference (verbatim).  The 4-D causal mask is the eager
mask from masking_utils.create_causal_mask for an unpadded batch.
Config bigscience/bloom-7b1: hidden_size 4096, n_head 32, layer_norm_epsilon 1e-5, bf16.
Batch 1, 2048 tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, SEQ, HIDDEN, N_HEAD, EPS = 1, 2048, 4096, 32, 1e-5
DTYPE = torch.bfloat16


def _get_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.full((), 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=device, dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = torch.full(
            (), 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


class Model(nn.Module):
    def __init__(self, hidden_size: int, n_head: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = n_head
        self.head_dim = hidden_size // n_head
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=EPS, dtype=DTYPE)
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size, bias=True, dtype=DTYPE)
        self.dense = nn.Linear(hidden_size, hidden_size, dtype=DTYPE)

    def _reshape(self, fused_qkv):
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        query_layer = fused_qkv[..., 0, :].transpose(1, 2)
        key_layer = fused_qkv[..., 1, :].transpose(1, 2)
        value_layer = fused_qkv[..., 2, :].transpose(1, 2)
        return query_layer, key_layer, value_layer

    def _merge_heads(self, x):
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.shape
        # BloomModel.forward: alibi without padding, eager causal mask
        slopes = _get_alibi_slopes(self.num_heads, hidden_states.device)
        alibi = slopes[:, None, None] * torch.arange(seq_length, device=hidden_states.device)
        alibi = alibi.repeat(batch_size, 1, 1).to(hidden_states.dtype)
        attention_mask = torch.full((seq_length, seq_length), torch.finfo(DTYPE).min, dtype=DTYPE,
                                    device=hidden_states.device).triu(1)[None, None]

        # BloomBlock.forward (attention half)
        layernorm_output = self.input_layernorm(hidden_states)
        residual = hidden_states

        # BloomAttention.forward
        fused_qkv = self.query_key_value(layernorm_output)
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)
        query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key_layer = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim).transpose(-1, -2)
        value_layer = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        attention_scores = alibi.baddbmm(batch1=query_layer, batch2=key_layer, beta=self.beta,
                                         alpha=self.inv_norm_factor)
        attn_weights = attention_scores.view(batch_size, self.num_heads, seq_length, -1)
        attn_weights = attn_weights + attention_mask
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, seq_length, -1)
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)
        context_layer = self._merge_heads(context_layer)
        output_tensor = self.dense(context_layer)
        return residual + output_tensor


def get_init_inputs():
    return [HIDDEN, N_HEAD]


def get_inputs():
    return [torch.randn(BATCH, SEQ, HIDDEN, dtype=DTYPE, device="cuda")]
