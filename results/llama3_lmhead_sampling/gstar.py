"""LLaMA-3 lm_head + sampling, G*: top-p and softmax on the top-k survivors only.

After the top-k mask every other logit is -inf and contributes exp(-inf) = 0, so the
ascending full-vocabulary sort, cumulative softmax and final softmax of G0 only ever see the
<= 50 surviving values: top-p and the final softmax run on the 50 top-k values (ascending
order = reversed top-k order) and the result is scattered back into the vocabulary.
"""

import torch
import torch.nn as nn

TEMPERATURE, TOP_K, TOP_P, EPS = 0.7, 50, 0.9, 1e-5
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.empty(hidden, dtype=DTYPE))
        self.lm_head = nn.Linear(hidden, vocab, bias=False, dtype=DTYPE)

    def forward(self, hidden_states):
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
        h = self.norm_weight * h.to(DTYPE)
        logits = self.lm_head(h)[:, -1, :]
        # x -> float(x) / T is monotonic, so selecting on the fp16 logits picks the same top-k
        top_v, top_i = torch.topk(logits, TOP_K)                       # descending, 16-bit radix select
        top_v = top_v.float() / TEMPERATURE
        asc = top_v.flip(-1)
        cum = asc.softmax(dim=-1).cumsum(dim=-1)
        keep_last = torch.arange(TOP_K, device=logits.device) == TOP_K - 1
        remove = (cum <= (1 - TOP_P)) & ~keep_last
        top_v = top_v.masked_fill(remove.flip(-1), -float("inf"))
        probs = torch.softmax(top_v, dim=-1)
        return torch.zeros(logits.shape, dtype=torch.float32, device=logits.device).scatter(1, top_i, probs)
