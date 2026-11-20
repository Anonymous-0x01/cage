"""Gemma-2 self-attention with attention-logit soft-capping (Gemma-2-2B, eager attention path), prefill.

Source: transformers 5.23.0, models/gemma2/modeling_gemma2.py: Gemma2Attention.forward,
eager_attention_forward (softcap branch), repeat_kv, rotate_half / apply_rotary_pos_emb (verbatim).
The 4-D causal mask is the eager mask produced by masking_utils.create_causal_mask for an unpadded
batch (0 on/below the diagonal, dtype-min above).
Config google/gemma-2-2b: hidden 2304, heads 8, kv heads 4, head_dim 256, query_pre_attn_scalar 256,
attn_logit_softcapping 50.0, attention_bias False, bf16.  Batch 1, 4096 tokens (global-attention layer).
"""

import torch
import torch.nn as nn

BATCH, SEQ, HIDDEN, N_HEADS, N_KV, HEAD_DIM = 1, 4096, 2304, 8, 4, 256
QUERY_PRE_ATTN_SCALAR, SOFTCAP, ROPE_THETA = 256, 50.0, 10000.0
DTYPE = torch.bfloat16


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, softcap):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Model(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = n_heads // n_kv
        self.scaling = QUERY_PRE_ATTN_SCALAR ** -0.5
        self.attn_logit_softcapping = SOFTCAP
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False, dtype=DTYPE)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, dtype=DTYPE)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, dtype=DTYPE)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=DTYPE)

    def forward(self, hidden_states, cos, sin):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        seq = input_shape[1]
        attention_mask = torch.full((seq, seq), torch.finfo(DTYPE).min, dtype=DTYPE, device=hidden_states.device)
        attention_mask = attention_mask.triu(1)[None, None]

        attn_output, _ = eager_attention_forward(self, query_states, key_states, value_states, attention_mask,
                                                 scaling=self.scaling, softcap=self.attn_logit_softcapping)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


def get_init_inputs():
    return [HIDDEN, N_HEADS, N_KV, HEAD_DIM]


def get_inputs():
    x = torch.randn(BATCH, SEQ, HIDDEN, dtype=DTYPE, device="cuda")
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32) / HEAD_DIM))
    freqs = torch.outer(torch.arange(SEQ, device="cuda", dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(DTYPE).expand(BATCH, -1, -1).contiguous()
    sin = emb.sin().to(DTYPE).expand(BATCH, -1, -1).contiguous()
    return [x, cos, sin]
