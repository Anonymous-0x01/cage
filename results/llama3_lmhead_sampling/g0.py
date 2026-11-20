"""LLaMA-3-8B output head + sampling distribution (Hugging Face generate logits warpers), decode, 32 sequences.

final RMSNorm -> lm_head (4096 -> 128256) -> fp32 logits -> temperature 0.7 -> top-k 50 (mask the
rest with -inf) -> top-p 0.9 (sort the full vocabulary, cumulative softmax, mask) -> softmax ->
sampling probabilities over the vocabulary.  Cross-operator structure in G0: top-p sorts all
128256 logits although at most 50 survive top-k; several full-vocab fp32 passes.
"""

import torch
import torch.nn as nn

BATCH, HIDDEN, VOCAB = 32, 4096, 128256
TEMPERATURE, TOP_K, TOP_P, EPS = 0.7, 50, 0.9, 1e-5
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.norm_weight = nn.Parameter(1 + 0.02 * torch.randn(hidden, dtype=DTYPE))
        self.lm_head = nn.Linear(hidden, vocab, bias=False, dtype=DTYPE)
        with torch.no_grad():
            self.lm_head.weight.normal_(0, 0.05)

    def forward(self, hidden_states):
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
        h = self.norm_weight * h.to(DTYPE)
        scores = self.lm_head(h)[:, -1, :].float()

        scores = scores / TEMPERATURE
        kth = torch.topk(scores, TOP_K)[0][..., -1, None]
        scores = scores.masked_fill(scores < kth, -float("inf"))

        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_to_remove = cumulative_probs <= (1 - TOP_P)
        sorted_to_remove[..., -1:] = 0
        to_remove = sorted_to_remove.scatter(1, sorted_indices, sorted_to_remove)
        scores = scores.masked_fill(to_remove, -float("inf"))
        return torch.softmax(scores, dim=-1)


def get_init_inputs():
    return [HIDDEN, VOCAB]


def get_inputs():
    return [torch.randn(BATCH, 1, HIDDEN, dtype=DTYPE, device="cuda")]
