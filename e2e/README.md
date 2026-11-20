# End-to-end CAGE experiments: LLaMA-3-8B, Qwen3-8B, and DeepSeek-R1-Distill-Llama-8B, batch 1 and 8

Each full model (every decoder layer, embedding, lm_head, bf16) is optimized with the CAGE workflow as two
graphs, prefill and one decode step, each at batch 1 and 8. Then a complete greedy generation (1024-token
prompt + 128 new tokens) is timed through the original and the optimized graphs. No checkpoint is downloaded;
the experiments measure latency only.

| model | layers | hidden / MLP | heads (q / kv) x head_dim | vocab | specifics |
|---|---|---|---|---|---|
| `llama3-8b` (Meta-Llama-3-8B) | 32 | 4096 / 14336 | 32 / 8 x 128 | 128256 | llama3 rope scaling, theta 5e5 |
| `qwen3-8b` (Qwen3-8B) | 36 | 4096 / 12288 | 32 / 8 x 128 | 151936 | per-head q_norm / k_norm, default rope, theta 1e6 |
| `deepseek-r1-distill-llama-8b` (DeepSeek-R1-Distill-Llama-8B) | 32 | 4096 / 14336 | 32 / 8 x 128 | 128256 | LlamaForCausalLM, llama3 rope scaling, theta 5e5, max position 131072 |

## Reproduce

```bash
source scripts/env.sh                       # conda env `clo`
python e2e/compare.py --list                # model/case and one-line G* descriptions
python e2e/compare.py --models qwen3-8b --check    # forward (4 cases) + generation (batch 1, 8) for one model
python e2e/compare.py --check                      # every model
python e2e/compare.py --mode forward --cases qwen3-8b/decode_b1,decode_b8 --rounds 10
python e2e/compare.py --mode generate --models llama3-8b --batches 8 --rounds 5
python e2e/compare.py --report              # no GPU: cross-model speedup overview from the saved results
python e2e/compare.py --report --baseline compile   # graph-only speedup, one number per cell
python e2e/validate_g0.py --model qwen3-8b  # G0 == HF causal LM, eager-vs-Inductor error floor
```

Each model is a folder under `tasks/` and `results/`. `--models` selects those folders (default: all).
`--cases` accepts `decode_b1` (that case of every selected model) or `qwen3-8b/decode_b1`.

`compare.py` picks the least-utilized GPU with enough free memory (`--gpu N` overrides). The need is estimated
from the weight size in the model's g0 constants. The forward comparison builds four runners, but the compiled
G0 runners alias the eager parameters (they only read those weights); G* keeps its own layout. Resident weights
are about two copies (~32 GB for an 8B model) plus CUDA-graph pools. The picker still asks for four copies plus
headroom, and generation holds two. The H20s are shared, so ratios within one run are reliable while absolute times
drift with other jobs. The first run of a model compiles everything, max-autotune included (~15 min); later
runs hit the Inductor cache in `.cache/` (a full `--check` run of one model takes ~20 min).

**Results.** Every run merges its rows (per forward case / generation batch, each with timestamp, GPU and
rounds) into `results/<model>/summary.{json,md}`, so re-measuring a subset keeps the other results; a failed
row never replaces a good saved one. Then `e2e/summary.{md,json}` is rebuilt as the cross-model overview of
every saved model. Each baseline gets its own forward table (rows = cases, plus a geomean row) and its own
generation table (rows = batch x total / TPOT / TTFT); a cell is one G* speedup, so models can be compared
down a column. The three baselines are eager G0, torch.compile(G0, default), and torch.compile(G0, G* cfg)
(graph-only). `--baseline eager|compile_default|compile` keeps one of them. Per-model detail tables follow.
`--report` prints and rewrites the overview from the saved JSON without measuring (`--models` filters it).

## Layout

| path | content |
|---|---|
| `tasks/<model>/` | one folder per model: `<model>_{prefill,decode}_b{1,8}.py`, the G0 files used by `bin/cage init` (the b8 files differ only in `BATCH`) |
| `results/<model>/<case>/g0.py`, `gstar.py` | original and optimized graph, cases `{prefill,decode}_b{1,8}` |
| `results/<model>/summary.{json,md}` | saved measurements of that model (merged across runs) |
| `validate_g0.py` / `validation.json` | G0 vs Hugging Face, error floor, memory, rough latency (keys `<model>/<case>`) |
| `generate.py` | greedy generation driver (one GPU, one subprocess per batch size) |
| `compare.py` / `summary.{md,json}` | reproduction script and the cross-model overview |
| `runs/<model>_*` (repo root, gitignored) | full CAGE history: candidates, ledger, `plan.md`, `report.md`, trajectory plot |

## Cross-model overview (`python e2e/compare.py --report`, 2026-11-20)

One table per baseline. DeepSeek-R1-Distill-Llama-8B was measured with `--check` on GPU 2; the LLaMA and Qwen
columns are those models' saved results.

### vs eager G0

| case | deepseek-r1-distill-llama-8b | llama3-8b | qwen3-8b |
|---|---|---|---|
| decode_b1 | 3.32x | 3.35x | 4.38x |
| decode_b8 | 3.67x | 3.68x | 4.12x |
| prefill_b1 | 1.34x | 1.34x | 1.38x |
| prefill_b8 | 1.33x | 1.33x | 1.40x |
| **geomean** | **2.16x** | **2.17x** | **2.43x** |

| batch | greedy generation | deepseek-r1-distill-llama-8b | llama3-8b | qwen3-8b |
|---|---|---|---|---|
| 1 | total | 3.06x | 2.89x | 3.75x |
| 1 | TPOT | 3.30x | 3.12x | 4.07x |
| 1 | TTFT | 1.33x | 1.33x | 1.37x |
| 8 | total | 2.68x | 2.69x | 2.92x |
| 8 | TPOT | 3.84x | 3.87x | 4.15x |
| 8 | TTFT | 1.33x | 1.33x | 1.39x |

### vs torch.compile(G0, default)

| case | deepseek-r1-distill-llama-8b | llama3-8b | qwen3-8b |
|---|---|---|---|
| decode_b1 | 1.43x | 1.41x | 1.46x |
| decode_b8 | 1.87x | 1.87x | 1.92x |
| prefill_b1 | 1.11x | 1.11x | 1.08x |
| prefill_b8 | 1.10x | 1.10x | 1.10x |
| **geomean** | **1.35x** | **1.34x** | **1.35x** |

| batch | greedy generation | deepseek-r1-distill-llama-8b | llama3-8b | qwen3-8b |
|---|---|---|---|---|
| 1 | total | 1.28x | 1.29x | 1.32x |
| 1 | TPOT | 1.30x | 1.32x | 1.35x |
| 1 | TTFT | 1.11x | 1.11x | 1.08x |
| 8 | total | 1.61x | 1.62x | 1.67x |
| 8 | TPOT | 2.05x | 2.06x | 2.13x |
| 8 | TTFT | 1.10x | 1.10x | 1.10x |

### vs torch.compile(G0, G* cfg) (graph-only)

| case | deepseek-r1-distill-llama-8b | llama3-8b | qwen3-8b |
|---|---|---|---|
| decode_b1 | 1.23x | 1.22x | 1.27x |
| decode_b8 | 1.76x | 1.76x | 1.82x |
| prefill_b1 | 1.08x | 1.08x | 1.08x |
| prefill_b8 | 1.07x | 1.07x | 1.07x |
| **geomean** | **1.26x** | **1.26x** | **1.28x** |

| batch | greedy generation | deepseek-r1-distill-llama-8b | llama3-8b | qwen3-8b |
|---|---|---|---|---|
| 1 | total | 1.14x | 1.14x | 1.18x |
| 1 | TPOT | 1.14x | 1.14x | 1.19x |
| 1 | TTFT | 1.08x | 1.08x | 1.08x |
| 8 | total | 1.55x | 1.55x | 1.62x |
| 8 | TPOT | 1.97x | 1.97x | 2.07x |
| 8 | TTFT | 1.08x | 1.07x | 1.07x |

DeepSeek-R1-Distill-Llama-8B is the same Llama graph as LLaMA-3-8B, and its graph-only geomean matches that
model (1.26x). The same rewrite families carry over: decode gains come from CUDA graphs plus removing the KV
cat / repeat_kv traffic (largest at batch 8), prefill gains from flash attention and last-layer pruning on
GEMM-bound graphs. Qwen3's eager baseline is slower (36 layers, two extra per-head norms per layer), so its
speedups over eager are larger; against torch.compile the three models are within a few percent. The per-model
reference runs below carry the absolute times.

## LLaMA-3-8B

### G0: Hugging Face LLaMA-3-8B, ported verbatim

G0 is ported from transformers 4.57 `models/llama/modeling_llama.py` (eager attention, `repeat_kv`,
fp32-upcast RMSNorm, llama3 rope scaling) and `masking_utils.eager_mask`. Parameter names are HF's, and
`validate_g0.py` loads the same weights into `LlamaForCausalLM(attn_implementation="eager")`:
logits and K/V match **bit for bit** for all four tasks.

- **prefill**: `input_ids [B,1024]` -> last-token logits `[B,1,128256]` (`logits_to_keep=1`, as `generate()`
  runs it) and the K/V cache stacked as `[32,B,8,1024,128]`.
- **decode**: `input_ids [B,1]` + past K/V `[32,B,8,L,128]` -> logits + the new token's K/V. As in HF's
  DynamicCache, G0 concatenates cache and new K/V inside attention; the caller owns the cache.

**Tolerance.** Eager and Inductor-compiled G0 already differ by max |err| 0.47-0.89 (relative L2 5.5-10.5%)
on logits and K/V. With `emulate_precision_casts` Inductor matches eager bit for bit up to 4 layers, so the
gap comes from Inductor skipping intermediate bf16 roundings inside fused kernels, accumulated over 32
layers. The tasks therefore use `TOLERANCE = {"atol": 1.0, "rtol": 0.1}`, ~1.2-1.8x above that floor (a
semantic bug gives O(1) relative errors). `compare.py --check` also prints the relative L2 error of G* next to
that of compiled G0: G* deviates no more than Inductor's own compilation of G0 does.

### Cases and transformations (G*)

| case | G* transformations | CAGE run (G* candidate) |
|---|---|---|
| decode_b1 | CUDA graphs (G0 is 41% launch-idle, 803 launches/step); merged QKV `[6144,4096]` and gate/up `[28672,4096]` GEMVs (`convert_weights`); GQA attention split over the cache and the new token: queries grouped per KV head `[B,8,4s,D]`, scores against the cached and new K separately, only the tiny score tensor concatenated for the fp32 softmax, output `P_cache V_cache + P_new V_new` (no KV `cat`, no `repeat_kv`) | `llama3_8b_decode_b1` (c003) |
| decode_b8 | same graph (batch-1 G* transferred unchanged); at batch 8 the removed KV cat / repeat_kv copies are ~10 GB per step | `llama3_8b_decode_b8` (c001) |
| prefill_b1 | causal GQA flash attention (`scaled_dot_product_attention(is_causal=True, enable_gqa=True)`, no `[B,32,S,S]` fp32 scores, no repeat_kv); merged QKV and gate/up GEMMs; **last-layer last-token pruning**: only the last position reaches lm_head, so layer 31 computes K/V for every position (they are outputs) but q, attention, o_proj and the MLP for the last position only (~1/32 of the FLOPs); max-autotune GEMMs | `llama3_8b_prefill_b1` (c004) |
| prefill_b8 | same graph (batch-1 G* transferred unchanged) | `llama3_8b_prefill_b8` (c001) |

Tried and rejected (measured): max-autotune GEMV templates for decode (slower than cuBLAS at M=1),
coordinate-descent tuning for decode (within noise), dropping CUDA graphs at batch 8 to avoid copying the
1 GB cache input (launch savings win), flex_attention instead of cuDNN SDPA for prefill (no gain, 14 min
compile). Prefill is GEMM-bound at ~119 TFLOPS (~80% of H20 dense peak) after the rewrites.

### Reference run (`python e2e/compare.py --models llama3-8b --check`, GPU 4, shared H20, 2026-11-16)

Forward passes (10 interleaved rounds, L2 flush, median):

| case | eager G0 (ms) | compile G0 default (ms) | compile G0 w/ G* cfg (ms) | G* (ms) | G* vs eager | G* vs compile(G0, default) | G* vs compile(G0, G* cfg) = graph-only | legal |
|---|---|---|---|---|---|---|---|---|
| decode_b1 | 18.69 | 8.56 | 7.24 | 6.01 | 3.11x | 1.43x | 1.21x | yes |
| decode_b8 | 26.35 | 13.31 | 12.45 | 7.01 | 3.76x | 1.90x | 1.78x | yes |
| prefill_b1 | 166.09 | 136.68 | 133.22 | 123.42 | 1.35x | 1.11x | 1.08x | yes |
| prefill_b8 | 1255.91 | 1034.98 | 1011.32 | 941.75 | 1.33x | 1.10x | 1.07x | yes |
| **geomean** | | | | | **2.14x** | **1.35x** | **1.26x** | |

Numerics against eager G0 (worst output, relative L2 / max abs): decode_b1 compile(G0) 0.082 / 0.54 vs G*
0.081 / 0.57; decode_b8 0.074 / 0.47 vs 0.075 / 0.48; prefill_b1 0.101 / 0.86 vs 0.098 / 0.69; prefill_b8
0.105 / 0.86 vs 0.097 / 0.70.

Greedy generation, 1024-token prompt + 128 new tokens (2 warm-up + 3 timed requests per target, median):

| batch | metric | eager G0 | compile G0 default | compile G0 w/ G* cfg | G* | G* vs eager | G* vs compile(G0, default) | G* vs compile(G0, G* cfg) |
|---|---|---|---|---|---|---|---|---|
| 1 | TTFT (ms) | 167.41 | 139.93 | 134.69 | 125.86 | 1.33x | 1.11x | 1.07x |
| 1 | TPOT (ms/token) | 20.36 | 8.69 | 7.58 | 6.64 | 3.06x | 1.31x | 1.14x |
| 1 | **total (ms)** | 2753.60 | 1243.36 | 1096.79 | 969.70 | **2.84x** | **1.28x** | **1.13x** |
| 8 | TTFT (ms) | 1326.48 | 1094.40 | 1068.01 | 994.47 | 1.33x | 1.10x | 1.07x |
| 8 | TPOT (ms/token) | 34.03 | 18.20 | 17.51 | 8.80 | 3.87x | 2.07x | 1.99x |
| 8 | **total (ms)** | 5648.35 | 3405.88 | 3291.49 | 2112.95 | **2.67x** | **1.61x** | **1.56x** |

Generation notes:
- The driver owns the KV cache (preallocated `[32,B,8,1152,128]`, prefill K/V written once, each decode step
  reads `buf[..., :L, :]` views and writes its new K/V at L). Cache writes and argmax are the same code for
  every target.
- The decode cache length changes every step, so compiled targets mark it dynamic. That is why default-compiled
  G0's batch-8 TPOT (18.2 ms) is above its static-shape forward time (13.3 ms). CUDA-graph modes record one
  replay per length during the two warm-up requests.
- The token-agreement row in `summary.md` is informational only. With weights the top logits are
  nearly flat (top-5 within ~0.16 at a logit std of 1.28), so any bf16 re-association, Inductor's own
  compilation of G0 included, changes the greedy path within the first tokens.

## Qwen3-8B

### G0: Hugging Face Qwen3-8B, ported verbatim

G0 is ported from transformers 4.57 `models/qwen3/modeling_qwen3.py` (eager attention, `repeat_kv`,
fp32-upcast RMSNorm, default rope) and `masking_utils.eager_mask`. Relative to LLaMA, every attention layer
applies a per-head RMSNorm over head_dim to q (`q_norm`) and k (`k_norm`) between the projections and RoPE;
rope_theta is 1e6 without scaling, eps 1e-6, sliding-window attention is off (as in Qwen3-8B's config).
Parameter names are HF's, and `validate_g0.py --model qwen3-8b` loads the same weights into
`Qwen3ForCausalLM(attn_implementation="eager")`: logits and K/V match **bit for bit** for all four tasks.

- **prefill**: `input_ids [B,1024]` -> last-token logits `[B,1,151936]` (`logits_to_keep=1`) and the K/V cache
  stacked as `[36,B,8,1024,128]`.
- **decode**: `input_ids [B,1]` + past K/V `[36,B,8,L,128]` -> logits + the new token's K/V (DynamicCache-style
  cat inside attention; the caller owns the cache).

**Tolerance.** Eager and Inductor-compiled G0 differ by max |err| 0.34-0.54 (relative L2 4.7-7.8%) on logits and
K/V over 3 seeds, for the same reason as LLaMA (Inductor skips intermediate bf16 roundings, 36 layers accumulate
them). The tasks use `TOLERANCE = {"atol": 1.0, "rtol": 0.1}`, ~1.85x above that floor.

### Cases and transformations (G*)

| case | G* transformations | CAGE run (G* candidate) |
|---|---|---|
| decode_b1 | CUDA graphs; merged QKV `[6144,4096]` and gate/up `[24576,4096]` GEMVs (`convert_weights`); GQA attention split over the cache and the new token (queries grouped per KV head `[B,8,4s,D]`, no KV `cat`, no `repeat_kv`); **joint q/k head norm**: `q_norm` and `k_norm` share eps and act per head, so the 32 q + 8 k heads of the qkv output are normalized by one RMSNorm with the per-head weights stacked `[40,128]`, followed by one RoPE | `qwen3_8b_decode_b1` (c004) |
| decode_b8 | same graph (batch-1 G* transferred unchanged); CUDA graphs kept: without them 1.82x vs 1.90x over Inductor G0, the launch savings outweigh the copies of the 1.15 GB cache inputs | `qwen3_8b_decode_b8` (c002) |
| prefill_b1 | causal GQA flash attention (`scaled_dot_product_attention(is_causal=True, enable_gqa=True)`); merged QKV and gate/up GEMMs; joint q/k head norm + RoPE; **last-layer last-token pruning** (layer 35 computes K/V for every position, q / attention / o_proj / MLP for the last one) | `qwen3_8b_prefill_b1` (c003) |
| prefill_b8 | batch-1 G* + `COMPILE_MODE="max-autotune-no-cudagraphs"` (+2.8% at batch 8; at batch 1 the knob gave +0.5%, within noise, and is not used) | `qwen3_8b_prefill_b8` (c003) |

CAGE trajectory (decode_b1, speedup over Inductor G0 in the run's own sessions): CUDA graphs 1.14x, then merged
GEMVs + split attention 1.45x, then joint q/k norm 1.46x. Prefill_b1: SDPA 1.04x, merged GEMMs + joint norm
1.05x, last-layer pruning 1.08x.

Tried and rejected (measured): writing the M=1 GEMVs as fused fp32 reductions so Inductor generates them
(0.79x: the RMSNorm prologue is recomputed per output row and the Triton reductions reach ~1.3 TB/s against
cuBLAS's 3-4 TB/s); max-autotune GEMV templates with CUDA graphs (1.41x vs 1.46x); pre-transposed `[K,N]`
weights (same cuBLAS kernels, split-K included, neutral); graph-level split-K with K-chunked weights in one
`bmm` + fused sum (1.41x); new-token attention terms as broadcast products and `combo_kernels` (neutral);
for prefill, RoPE frequencies as an outer product and coordinate-descent tuning (neutral). After the rewrites
decode is GEMV-bandwidth-bound inside one CUDA graph (GEMVs ~85% of GPU time) and prefill is GEMM-bound at
~137-139 TFLOPS (~93% of H20 dense peak, non-GEMM kernels <4%).

### Reference run (`python e2e/compare.py --models qwen3-8b --check`, GPU 2, shared H20, 2026-11-20)

Forward passes (10 interleaved rounds, L2 flush, median):

| case | eager G0 (ms) | compile G0 default (ms) | compile G0 w/ G* cfg (ms) | G* (ms) | G* vs eager | G* vs compile(G0, default) | G* vs compile(G0, G* cfg) = graph-only | legal |
|---|---|---|---|---|---|---|---|---|
| decode_b1 | 24.97 | 8.36 | 7.24 | 5.70 | 4.38x | 1.46x | 1.27x | yes |
| decode_b8 | 27.75 | 12.95 | 12.27 | 6.74 | 4.12x | 1.92x | 1.82x | yes |
| prefill_b1 | 149.91 | 117.16 | 117.20 | 108.65 | 1.38x | 1.08x | 1.08x | yes |
| prefill_b8 | 1146.99 | 901.97 | 876.21 | 820.47 | 1.40x | 1.10x | 1.07x | yes |
| **geomean** | | | | | **2.43x** | **1.35x** | **1.28x** | |

Numerics against eager G0 (worst output, relative L2 / max abs): decode_b1 compile(G0) 0.073 / 0.43 vs G*
0.072 / 0.44; decode_b8 0.076 / 0.54 vs 0.075 / 0.52; prefill_b1 0.071 / 0.50 vs 0.071 / 0.44; prefill_b8
0.068 / 0.52 vs 0.066 / 0.47.

Greedy generation, 1024-token prompt + 128 new tokens (2 warm-up + 3 timed requests per target, median):

| batch | metric | eager G0 | compile G0 default | compile G0 w/ G* cfg | G* | G* vs eager | G* vs compile(G0, default) | G* vs compile(G0, G* cfg) |
|---|---|---|---|---|---|---|---|---|
| 1 | TTFT (ms) | 150.63 | 118.84 | 118.55 | 110.27 | 1.37x | 1.08x | 1.08x |
| 1 | TPOT (ms/token) | 25.84 | 8.58 | 7.57 | 6.34 | 4.07x | 1.35x | 1.19x |
| 1 | **total (ms)** | 3432.92 | 1208.27 | 1079.47 | 916.00 | **3.75x** | **1.32x** | **1.18x** |
| 8 | TTFT (ms) | 1156.53 | 914.65 | 887.77 | 832.03 | 1.39x | 1.10x | 1.07x |
| 8 | TPOT (ms/token) | 33.85 | 17.41 | 16.89 | 8.16 | 4.15x | 2.13x | 2.07x |
| 8 | **total (ms)** | 5455.75 | 3125.71 | 3033.36 | 1868.00 | **2.92x** | **1.67x** | **1.62x** |

The generation driver, the dynamic decode length and the informational token-agreement row work as described
for LLaMA-3-8B above.

## DeepSeek-R1-Distill-Llama-8B

### G0: Hugging Face DeepSeek-R1-Distill-Llama-8B, ported verbatim

G0 is the LlamaForCausalLM graph: transformers 4.57 `models/llama/modeling_llama.py` (eager attention,
`repeat_kv`, fp32-upcast RMSNorm, llama3 rope scaling) and `masking_utils.eager_mask`. Parameter names are
HF's. Config: deepseek-ai/DeepSeek-R1-Distill-Llama-8B (32 layers, hidden 4096, 32/8 GQA heads, MLP 14336,
vocab 128256, max position 131072, bf16). `validate_g0.py --model deepseek-r1-distill-llama-8b` loads the same
weights into `LlamaForCausalLM(attn_implementation="eager")`: logits and K/V match **bit for bit** for all four
tasks. No checkpoint is downloaded; the run measures latency only.

- **prefill**: `input_ids [B,1024]` -> last-token logits `[B,1,128256]` (`logits_to_keep=1`) and the K/V cache
  stacked as `[32,B,8,1024,128]`.
- **decode**: `input_ids [B,1]` + past K/V `[32,B,8,L,128]` -> logits + the new token's K/V. As in HF's
  DynamicCache, G0 concatenates cache and new K/V inside attention; the caller owns the cache.

**Tolerance.** Eager and Inductor-compiled G0 differ by max |err| 0.41-0.89 (relative L2 5.5-10.5%) on logits
and K/V over 3 seeds, for the same reason as LLaMA (Inductor skips intermediate bf16 roundings, 32 layers
accumulate them). The tasks use `TOLERANCE = {"atol": 1.0, "rtol": 0.1}`.

### Cases and transformations (G*)

| case | G* transformations | CAGE run (G* candidate) |
|---|---|---|
| decode_b1 | CUDA graphs; merged QKV `[6144,4096]` and gate/up `[28672,4096]` GEMVs (`convert_weights`); GQA attention split over the cache and the new token (queries grouped per KV head `[B,8,4s,D]`, no KV `cat`, no `repeat_kv`) | `deepseek_r1_distill_llama_8b_decode_b1` (c003) |
| decode_b8 | same graph with `BATCH=8`; CUDA graphs kept | `deepseek_r1_distill_llama_8b_decode_b8` (c001) |
| prefill_b1 | causal GQA flash attention (`scaled_dot_product_attention(is_causal=True, enable_gqa=True)`); merged QKV and gate/up GEMMs; **last-layer last-token pruning** (layer 31 computes K/V for every position, q / attention / o_proj / MLP for the last one); max-autotune GEMMs | `deepseek_r1_distill_llama_8b_prefill_b1` (c003) |
| prefill_b8 | same graph with `BATCH=8` | `deepseek_r1_distill_llama_8b_prefill_b8` (c001) |

CAGE trajectory (speedup over Inductor G0 in each run's own session): decode_b1 CUDA graphs 1.14x, then merged
GEMVs 1.23x, then split attention 1.43x. Prefill_b1: SDPA 1.04x, merged GEMMs 1.07x, last-layer pruning 1.11x.
decode_b8 transferred the batch-1 graph and measured 1.87x; prefill_b8 transferred it and measured 1.10x. Both
are inside the band of the same structure on LLaMA-3-8B, so no compile-mode variant was measured.

`ctx graph` on decode G0: 64 clones of the repeated KV cache (`[1,8,4,1025,128]` bf16, 8 MB, 668 MB unfused at
batch 1; `[8,8,4,1025,128]`, 64 MB, 5.2 GB unfused at batch 8) plus separate q/k/v and gate/up GEMVs.
Inductor left the repeat_kv copy in place (610 launches at batch 1). On c003 those clones are gone. Prefill G0
materializes `[B,32,1024,1024]` fp32 scores (128 MB at batch 1, 1 GB at batch 8; 27 GB / 221 GB unfused).

### Reference run (`python e2e/compare.py --models deepseek-r1-distill-llama-8b --check`, GPU 2, shared H20, 2026-11-20)

Forward passes (10 interleaved rounds, L2 flush, median):

| case | eager G0 (ms) | compile G0 default (ms) | compile G0 w/ G* cfg (ms) | G* (ms) | G* vs eager | G* vs compile(G0, default) | G* vs compile(G0, G* cfg) = graph-only | legal |
|---|---|---|---|---|---|---|---|---|
| decode_b1 | 17.83 | 7.67 | 6.62 | 5.37 | 3.32x | 1.43x | 1.23x | yes |
| decode_b8 | 22.84 | 11.67 | 10.97 | 6.23 | 3.67x | 1.87x | 1.76x | yes |
| prefill_b1 | 141.94 | 117.77 | 114.31 | 105.64 | 1.34x | 1.11x | 1.08x | yes |
| prefill_b8 | 1084.80 | 895.11 | 873.43 | 813.37 | 1.33x | 1.10x | 1.07x | yes |
| **geomean** | | | | | **2.16x** | **1.35x** | **1.26x** | |

Numerics against eager G0 (worst output, relative L2 / max abs): decode_b1 compile(G0) 0.081 / 0.54 vs G*
0.081 / 0.57; decode_b8 0.074 / 0.47 vs 0.075 / 0.48; prefill_b1 0.101 / 0.90 vs 0.098 / 0.68; prefill_b8
0.105 / 0.86 vs 0.097 / 0.69.

Greedy generation, 1024-token prompt + 128 new tokens (2 warm-up + 3 timed requests per target, median):

| batch | metric | eager G0 | compile G0 default | compile G0 w/ G* cfg | G* | G* vs eager | G* vs compile(G0, default) | G* vs compile(G0, G* cfg) |
|---|---|---|---|---|---|---|---|---|
| 1 | TTFT (ms) | 143.50 | 119.72 | 116.10 | 107.71 | 1.33x | 1.11x | 1.08x |
| 1 | TPOT (ms/token) | 19.67 | 7.74 | 6.81 | 5.95 | 3.30x | 1.30x | 1.14x |
| 1 | **total (ms)** | 2641.72 | 1102.57 | 980.77 | 863.65 | **3.06x** | **1.28x** | **1.14x** |
| 8 | TTFT (ms) | 1088.23 | 899.28 | 878.19 | 816.84 | 1.33x | 1.10x | 1.08x |
| 8 | TPOT (ms/token) | 28.62 | 15.25 | 14.64 | 7.45 | 3.84x | 2.05x | 1.97x |
| 8 | **total (ms)** | 4723.01 | 2836.99 | 2737.54 | 1762.27 | **2.68x** | **1.61x** | **1.55x** |

The generation driver, the dynamic decode length and the informational token-agreement row work as described
for LLaMA-3-8B above.
