"""LLaMA-3-8B prefill, G*: causal GQA SDPA + merged QKV / gate-up GEMMs + last layer on the last token only + max-autotune GEMMs.

CAGE run runs/llama3_8b_prefill_b1 (c004).  Same inputs/outputs as g0.py.  Attention is one causal flash
kernel per layer (no [B,32,S,S] fp32 scores, no repeat_kv).  Only the last position reaches lm_head
(logits_to_keep=1), so the last layer computes K/V for every position (they are outputs) but q, attention,
o_proj and the MLP for the last position only (~1/32 of the FLOPs removed).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, PROMPT_LEN = 1, 1024
VOCAB, HIDDEN, INTERMEDIATE, N_LAYERS, N_HEADS, N_KV, HEAD_DIM = 128256, 4096, 14336, 32, 32, 8, 128
ROPE_THETA, EPS, INIT_STD = 500000.0, 1e-5, 0.02
ROPE_SCALING = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
DTYPE = torch.bfloat16
COMPILE_MODE = "max-autotune-no-cudagraphs"

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
        self.q_size, self.kv_size = n_heads * head_dim, n_kv * head_dim
        self.qkv_proj = nn.Linear(hidden, self.q_size + 2 * self.kv_size, bias=False, dtype=DTYPE)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=DTYPE)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q, k, v = self.qkv_proj(hidden_states).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # causal GQA flash attention: same mask as the additive [S,S] eager mask, KV heads shared by groups
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True,
                                                     scale=self.scaling, enable_gqa=True)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        return self.o_proj(attn_output), key_states, value_states

    def forward_last_query(self, hidden_states, position_embeddings):
        # K/V for every position (they are outputs), the query only for the last one
        b, s, _ = hidden_states.shape
        w = self.qkv_proj.weight
        k, v = F.linear(hidden_states, w[self.q_size:]).split([self.kv_size, self.kv_size], dim=-1)
        q = F.linear(hidden_states[:, -1:], w[:self.q_size])
        query_states = q.view(b, 1, -1, self.head_dim).transpose(1, 2)
        key_states = k.view(b, s, -1, self.head_dim).transpose(1, 2)
        value_states = v.view(b, s, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query_states = query_states * cos[:, -1:].unsqueeze(1) + rotate_half(query_states) * sin[:, -1:].unsqueeze(1)
        key_states = key_states * cos.unsqueeze(1) + rotate_half(key_states) * sin.unsqueeze(1)
        # the last position attends to every key under the causal mask
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, scale=self.scaling,
                                                     enable_gqa=True)
        attn_output = attn_output.transpose(1, 2).reshape(b, 1, -1)
        return self.o_proj(attn_output), key_states, value_states


class LlamaMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.intermediate = intermediate
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False, dtype=DTYPE)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=DTYPE)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).split([self.intermediate, self.intermediate], dim=-1)
        return self.down_proj(F.silu(gate) * up)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden, intermediate, n_heads, n_kv, head_dim):
        super().__init__()
        self.self_attn = LlamaAttention(hidden, n_heads, n_kv, head_dim)
        self.mlp = LlamaMLP(hidden, intermediate)
        self.input_layernorm = LlamaRMSNorm(hidden)
        self.post_attention_layernorm = LlamaRMSNorm(hidden)

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

    def forward_last_token(self, hidden_states, position_embeddings):
        # only the last position's output reaches lm_head (logits_to_keep=1)
        residual = hidden_states[:, -1:]
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, key_states, value_states = self.self_attn.forward_last_query(hidden_states,
                                                                                    position_embeddings)
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

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        inputs_embeds = self.model.embed_tokens(input_ids)
        cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        causal_mask = None

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        keys, values = [], []
        for decoder_layer in self.model.layers[:-1]:
            hidden_states, k, v = decoder_layer(hidden_states, causal_mask, position_embeddings)
            keys.append(k)
            values.append(v)
        hidden_states, k, v = self.model.layers[-1].forward_last_token(hidden_states, position_embeddings)
        keys.append(k)
        values.append(v)
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, torch.stack(keys), torch.stack(values)


def convert_weights(ref_state_dict):
    sd = dict(ref_state_dict)
    for i in range(N_LAYERS):
        a, m = f"model.layers.{i}.self_attn.", f"model.layers.{i}.mlp."
        sd[a + "qkv_proj.weight"] = torch.cat(
            [sd.pop(a + "q_proj.weight"), sd.pop(a + "k_proj.weight"), sd.pop(a + "v_proj.weight")], 0)
        sd[m + "gate_up_proj.weight"] = torch.cat([sd.pop(m + "gate_proj.weight"), sd.pop(m + "up_proj.weight")], 0)
    return sd


def get_init_inputs():
    return [VOCAB, HIDDEN, INTERMEDIATE, N_LAYERS, N_HEADS, N_KV, HEAD_DIM]


def get_inputs():
    input_ids = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), device="cuda")
    return [input_ids]
