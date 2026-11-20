"""Qwen3-8B full model, prefill (Hugging Face Qwen3ForCausalLM, eager attention).

Ported from transformers 4.57 `models/qwen3/modeling_qwen3.py` (Qwen3RMSNorm, Qwen3RotaryEmbedding with
default rope, Qwen3Attention with per-head q_norm / k_norm + eager_attention_forward, Qwen3MLP,
Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM) and `masking_utils.eager_mask`.  Parameter names match HF,
so `Model.state_dict()` loads into `Qwen3ForCausalLM` unchanged.  Config: Qwen/Qwen3-8B (36 layers, hidden
4096, 32/8 GQA heads, head_dim 128, MLP 12288, vocab 151936, rope_theta 1e6, untied lm_head, bf16).

What `generate()` runs for the prompt: BATCH x PROMPT_LEN tokens -> embeddings -> 36 decoder layers (fp32
RMSNorm, q/k/v projections, RMSNorm of every q and k head over head_dim, RoPE, repeat_kv, eager attention
with an additive [B,1,S,S] causal mask, o_proj, SwiGLU MLP) -> final norm -> lm_head on the last position
only (`logits_to_keep=1`).  Returns the bf16 last-token logits and the per-layer K/V cache stacked as
[L, B, n_kv, S, head_dim] (handed to the decoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, PROMPT_LEN = 1, 1024
VOCAB, HIDDEN, INTERMEDIATE, N_LAYERS, N_HEADS, N_KV, HEAD_DIM = 151936, 4096, 12288, 36, 32, 8, 128
ROPE_THETA, EPS, INIT_STD = 1000000.0, 1e-6, 0.02
DTYPE = torch.bfloat16
# Eager vs Inductor-compiled G0 differ by max |err| 0.34-0.54 (rel. L2 4.7-7.8%) on logits and K/V over 3 seeds:
# Inductor skips intermediate bf16 roundings inside fused kernels and 36 layers accumulate them (e2e/validation.json).
# The tolerance sits ~1.85x above that floor; semantic bugs give O(1) relative errors.
TOLERANCE = {"atol": 1.0, "rtol": 0.1}


def default_inv_freq(head_dim):
    return 1.0 / (ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=DTYPE))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.register_buffer("inv_freq", default_inv_freq(head_dim), persistent=False)
        self.attention_scaling = 1.0

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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


def eager_attention_forward(module, query, key, value, attention_mask, scaling):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous()


class Qwen3Attention(nn.Module):
    def __init__(self, hidden, n_heads, n_kv, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = n_heads // n_kv
        self.scaling = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False, dtype=DTYPE)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, dtype=DTYPE)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, dtype=DTYPE)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=DTYPE)
        self.q_norm = Qwen3RMSNorm(head_dim)
        self.k_norm = Qwen3RMSNorm(head_dim)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # DynamicCache.update on an empty cache stores and returns key/value states unchanged
        attn_output = eager_attention_forward(self, query_states, key_states, value_states, attention_mask,
                                              self.scaling)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), key_states, value_states


class Qwen3MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=DTYPE)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=DTYPE)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=DTYPE)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden, intermediate, n_heads, n_kv, head_dim):
        super().__init__()
        self.self_attn = Qwen3Attention(hidden, n_heads, n_kv, head_dim)
        self.mlp = Qwen3MLP(hidden, intermediate)
        self.input_layernorm = Qwen3RMSNorm(hidden)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden)

    def forward(self, hidden_states, attention_mask, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, key_states, value_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, key_states, value_states


class Qwen3Model(nn.Module):
    def __init__(self, vocab, hidden, intermediate, n_layers, n_heads, n_kv, head_dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden, dtype=DTYPE)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(hidden, intermediate, n_heads, n_kv, head_dim) for _ in range(n_layers)])
        self.norm = Qwen3RMSNorm(hidden)
        self.rotary_emb = Qwen3RotaryEmbedding(head_dim)


class Model(nn.Module):
    def __init__(self, vocab, hidden, intermediate, n_layers, n_heads, n_kv, head_dim):
        super().__init__()
        with torch.device("cuda"):
            self.model = Qwen3Model(vocab, hidden, intermediate, n_layers, n_heads, n_kv, head_dim)
            self.lm_head = nn.Linear(hidden, vocab, bias=False, dtype=DTYPE)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if not name.endswith("norm.weight"):
                    p.normal_(0.0, INIT_STD)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        inputs_embeds = self.model.embed_tokens(input_ids)
        cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        # masking_utils.eager_mask: causal bool mask -> 0 / finfo.min, shape [B, 1, q, kv]
        kv_idx = torch.arange(seq_len, device=inputs_embeds.device)
        allowed = kv_idx[None, :] <= cache_position[:, None]
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.where(allowed, torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                                  min_dtype)
        causal_mask = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        keys, values = [], []
        for decoder_layer in self.model.layers:
            hidden_states, k, v = decoder_layer(hidden_states, causal_mask, position_embeddings)
            keys.append(k)
            values.append(v)
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states[:, -1:, :])
        return logits, torch.stack(keys), torch.stack(values)


def get_init_inputs():
    return [VOCAB, HIDDEN, INTERMEDIATE, N_LAYERS, N_HEADS, N_KV, HEAD_DIM]


def get_inputs():
    input_ids = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), device="cuda")
    return [input_ids]
