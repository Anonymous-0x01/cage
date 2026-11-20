"""Qwen3-8B decode step, G*: CUDA graphs + merged QKV / gate-up GEMVs + GQA attention split over cache and new token (no KV cat, no repeat_kv) + joint q/k head norm.

CAGE run runs/qwen3_8b_decode_b1 (c004).  Same inputs/outputs as g0.py.  Structure per layer: one qkv GEMV
and one gate_up GEMV (weights concatenated offline in convert_weights); q_norm and k_norm run as one per-head
RMSNorm over the 32 q + 8 k heads of the qkv output, followed by one RoPE; queries grouped per KV head
([B, 8, 4*s, D]) are scored against the cached K and the new K separately, only the small score tensor is
concatenated for the fp32 softmax, and the output is P_cache @ V_cache + P_new @ V_new.
COMPILE_MODE="reduce-overhead" (CUDA graphs) removes the launch overhead of ~580 kernels per step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, CACHE_LEN = 1, 1024
VOCAB, HIDDEN, INTERMEDIATE, N_LAYERS, N_HEADS, N_KV, HEAD_DIM = 151936, 4096, 12288, 36, 32, 8, 128
ROPE_THETA, EPS, INIT_STD = 1000000.0, 1e-6, 0.02
DTYPE = torch.bfloat16
COMPILE_MODE = "reduce-overhead"


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
        self.q_size, self.kv_size = n_heads * head_dim, n_kv * head_dim
        self.qkv_proj = nn.Linear(hidden, self.q_size + 2 * self.kv_size, bias=False, dtype=DTYPE)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=DTYPE)
        self.q_norm = Qwen3RMSNorm(head_dim)
        self.k_norm = Qwen3RMSNorm(head_dim)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key, past_value):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        qk, v = self.qkv_proj(hidden_states).split([self.q_size + self.kv_size, self.kv_size], dim=-1)
        value_states = v.view(hidden_shape).transpose(1, 2)
        # q_norm and k_norm share eps and act per head: one RMSNorm over the 32 q + 8 k heads with the per-head
        # weights stacked, then one RoPE for both
        n_q, n_k = self.q_size // self.head_dim, self.kv_size // self.head_dim
        qk_weight = torch.cat([self.q_norm.weight.expand(n_q, -1), self.k_norm.weight.expand(n_k, -1)])
        qk = qk.view(hidden_shape)
        qk32 = qk.to(torch.float32)
        qk32 = qk32 * torch.rsqrt(qk32.pow(2).mean(-1, keepdim=True) + self.q_norm.variance_epsilon)
        qk = (qk_weight * qk32.to(qk.dtype)).transpose(1, 2)
        cos, sin = position_embeddings
        qk = qk * cos.unsqueeze(1) + rotate_half(qk) * sin.unsqueeze(1)
        query_states, key_states = qk.split([n_q, n_k], dim=1)
        # attention over [cache ; new] without materializing the cat or repeat_kv: query head h uses KV head
        # h // groups, so group the queries per KV head; every cached position precedes the new ones (mask 0)
        b, s = input_shape
        n_kv, g, past_len = key_states.shape[1], self.num_key_value_groups, past_key.shape[-2]
        qg = query_states.reshape(b, n_kv, g * s, self.head_dim)
        scores_cache = torch.matmul(qg, past_key.transpose(2, 3)) * self.scaling
        scores_new = torch.matmul(qg, key_states.transpose(2, 3)) * self.scaling + attention_mask
        attn_weights = torch.cat([scores_cache, scores_new], dim=-1)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = (torch.matmul(attn_weights[..., :past_len], past_value)
                       + torch.matmul(attn_weights[..., past_len:], value_states))
        attn_output = attn_output.view(b, n_kv * g, s, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(*input_shape, -1)
        return self.o_proj(attn_output), key_states, value_states


class Qwen3MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.intermediate = intermediate
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False, dtype=DTYPE)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=DTYPE)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).split([self.intermediate, self.intermediate], dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden, intermediate, n_heads, n_kv, head_dim):
        super().__init__()
        self.self_attn = Qwen3Attention(hidden, n_heads, n_kv, head_dim)
        self.mlp = Qwen3MLP(hidden, intermediate)
        self.input_layernorm = Qwen3RMSNorm(hidden)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden)

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

    def forward(self, input_ids, past_keys, past_values):
        batch, seq_len = input_ids.shape
        past_seen_tokens = past_keys.shape[-2]
        inputs_embeds = self.model.embed_tokens(input_ids)
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        # causal mask restricted to the new tokens (rows ordered (group, query) like the grouped queries)
        new_idx = torch.arange(seq_len, device=inputs_embeds.device)
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.where(new_idx[None, :] <= new_idx[:, None],
                                  torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype), min_dtype)
        causal_mask = causal_mask.repeat(N_HEADS // N_KV, 1)

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
    input_ids = torch.randint(0, VOCAB, (BATCH, 1), device="cuda")
    past_keys = torch.randn(N_LAYERS, BATCH, N_KV, CACHE_LEN, HEAD_DIM, dtype=DTYPE, device="cuda")
    past_values = torch.randn(N_LAYERS, BATCH, N_KV, CACHE_LEN, HEAD_DIM, dtype=DTYPE, device="cuda")
    return [input_ids, past_keys, past_values]
