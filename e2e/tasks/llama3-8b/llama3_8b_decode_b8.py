"""LLaMA-3-8B full model, one decode step with a KV cache (Hugging Face LlamaForCausalLM, eager attention
+ DynamicCache), weights.

Ported from transformers 4.57 `models/llama/modeling_llama.py` (LlamaRMSNorm, LlamaRotaryEmbedding with
llama3 rope scaling, LlamaAttention + eager_attention_forward, LlamaMLP, LlamaDecoderLayer, LlamaModel,
LlamaForCausalLM) and `masking_utils.eager_mask`.  Parameter names match HF, so `Model.state_dict()` loads
into `LlamaForCausalLM` unchanged.  Config: meta-llama/Meta-Llama-3-8B (32 layers, hidden 4096, 32/8 GQA
heads, MLP 14336, vocab 128256, bf16).

What `generate()` runs per new token: BATCH x 1 token -> embedding -> 32 decoder layers (fp32 RMSNorm, q/k/v
projections, RoPE at position CACHE_LEN, DynamicCache.update = torch.cat of the cached and new K/V, repeat_kv,
eager attention with the additive [B,1,1,CACHE_LEN+1] mask, o_proj, SwiGLU MLP) -> final norm -> lm_head.
The past K/V arrive stacked as [L, B, n_kv, CACHE_LEN, head_dim]; the step returns the bf16 logits and the new
token's K/V stacked as [L, B, n_kv, 1, head_dim].  The caller owns the cache (it appends the new entries),
so the cat inside attention is G0's own cost, not part of the output contract.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, CACHE_LEN = 8, 1024
VOCAB, HIDDEN, INTERMEDIATE, N_LAYERS, N_HEADS, N_KV, HEAD_DIM = 128256, 4096, 14336, 32, 32, 8, 128
ROPE_THETA, EPS, INIT_STD = 500000.0, 1e-5, 0.02
ROPE_SCALING = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
DTYPE = torch.bfloat16
# Eager vs Inductor-compiled G0 differ by max |err| 0.54-0.89 (rel. L2 5.5-10.5%) on logits and K/V over 3 seeds:
# Inductor skips intermediate bf16 roundings inside fused kernels and 32 layers accumulate them (e2e/validation.json).
# The tolerance sits ~1.2-1.8x above that floor; semantic bugs give O(1) relative errors.
TOLERANCE = {"atol": 1.0, "rtol": 0.1}


def llama3_inv_freq(head_dim):
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
    factor = ROPE_SCALING["factor"]
    low_freq_factor, high_freq_factor = ROPE_SCALING["low_freq_factor"], ROPE_SCALING["high_freq_factor"]
    old_context_len = ROPE_SCALING["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


class LlamaRMSNorm(nn.Module):
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


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.register_buffer("inv_freq", llama3_inv_freq(head_dim), persistent=False)
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


class LlamaAttention(nn.Module):
    def __init__(self, hidden, n_heads, n_kv, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = n_heads // n_kv
        self.scaling = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False, dtype=DTYPE)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, dtype=DTYPE)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, dtype=DTYPE)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=DTYPE)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key, past_value):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # DynamicLayer.update: cat the cached and new states along the sequence dim
        full_keys = torch.cat([past_key, key_states], dim=-2)
        full_values = torch.cat([past_value, value_states], dim=-2)
        attn_output = eager_attention_forward(self, query_states, full_keys, full_values, attention_mask,
                                              self.scaling)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), key_states, value_states


class LlamaMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=DTYPE)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=DTYPE)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=DTYPE)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden, intermediate, n_heads, n_kv, head_dim):
        super().__init__()
        self.self_attn = LlamaAttention(hidden, n_heads, n_kv, head_dim)
        self.mlp = LlamaMLP(hidden, intermediate)
        self.input_layernorm = LlamaRMSNorm(hidden)
        self.post_attention_layernorm = LlamaRMSNorm(hidden)

    def forward(self, hidden_states, attention_mask, position_embeddings, past_key, past_value):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, key_states, value_states = self.self_attn(hidden_states, position_embeddings, attention_mask,
                                                                 past_key, past_value)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, key_states, value_states


class LlamaModel(nn.Module):
    def __init__(self, vocab, hidden, intermediate, n_layers, n_heads, n_kv, head_dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden, dtype=DTYPE)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(hidden, intermediate, n_heads, n_kv, head_dim) for _ in range(n_layers)])
        self.norm = LlamaRMSNorm(hidden)
        self.rotary_emb = LlamaRotaryEmbedding(head_dim)


class Model(nn.Module):
    def __init__(self, vocab, hidden, intermediate, n_layers, n_heads, n_kv, head_dim):
        super().__init__()
        # weights created on the GPU (no checkpoint); HF _init_weights: normal(0, initializer_range)
        with torch.device("cuda"):
            self.model = LlamaModel(vocab, hidden, intermediate, n_layers, n_heads, n_kv, head_dim)
            self.lm_head = nn.Linear(hidden, vocab, bias=False, dtype=DTYPE)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if not name.endswith("layernorm.weight") and name != "model.norm.weight":
                    p.normal_(0.0, INIT_STD)

    def forward(self, input_ids, past_keys, past_values):
        batch, seq_len = input_ids.shape
        past_seen_tokens = past_keys.shape[-2]
        inputs_embeds = self.model.embed_tokens(input_ids)
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        # masking_utils.eager_mask: causal bool mask -> 0 / finfo.min, shape [B, 1, q, kv]
        kv_idx = torch.arange(past_seen_tokens + seq_len, device=inputs_embeds.device)
        allowed = kv_idx[None, :] <= cache_position[:, None]
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.where(allowed, torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                                  min_dtype)
        causal_mask = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        keys, values = [], []
        for i, decoder_layer in enumerate(self.model.layers):
            hidden_states, k, v = decoder_layer(hidden_states, causal_mask, position_embeddings,
                                                past_keys[i], past_values[i])
            keys.append(k)
            values.append(v)
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states[:, -1:, :])
        return logits, torch.stack(keys), torch.stack(values)


def get_init_inputs():
    return [VOCAB, HIDDEN, INTERMEDIATE, N_LAYERS, N_HEADS, N_KV, HEAD_DIM]


def get_inputs():
    input_ids = torch.randint(0, VOCAB, (BATCH, 1), device="cuda")
    past_keys = torch.randn(N_LAYERS, BATCH, N_KV, CACHE_LEN, HEAD_DIM, dtype=DTYPE, device="cuda")
    past_values = torch.randn(N_LAYERS, BATCH, N_KV, CACHE_LEN, HEAD_DIM, dtype=DTYPE, device="cuda")
    return [input_ids, past_keys, past_values]
