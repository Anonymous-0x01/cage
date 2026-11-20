# CAGE optimization results (G0 vs G*)

Each directory holds the original subgraph `g0.py` and the optimized graph `gstar.py` found with
the CAGE workflow (`runs/<case>/` has the full candidate history, ledger, plan and report).
G* files use only graph-level PyTorch rewrites, offline weight re-layout (`convert_weights`) and
Inductor settings; every G* passes the legality checker against G0 (3 seeds, dtype tolerance or
float64-oracle accuracy, zero graph breaks).

## Compare

```bash
source scripts/env.sh
python scripts/compare.py --list                      # cases and one-line G* descriptions
python scripts/compare.py --gpu 0 --rounds 10 --check # all cases, re-verify legality
python scripts/compare.py --cases vit_b16_embed,deepseek_v3_mla_decode
```

The script measures eager G0, `torch.compile` G0 and compiled G* in one process (interleaved
rounds, L2 flush, CUDA events, median) and writes `summary.md` / `summary.json`. The H20s are
shared; ratios within one run are reliable, absolute numbers drift with other jobs.

## Cases and transformations

| case | source model / code path | G* transformations (cross-operator) | vs torch.compile, idle GPU |
|---|---|---|---|
| deepseek_v3_mla_decode | DeepSeek-V3 MLA decode, naive decompress-then-attend over a latent cache | absorb W_UK into the query and W_UV into the output, attend over 512-d latents (cache never decompressed); new token handled separately (no cache cat); CUDA graphs | 21.7x |
| vit_b16_embed | timm ViT-B/16 PatchEmbed + cls + pos + LayerNorm | stride==kernel conv -> patch reshape + GEMM (removes the slow 3-channel cuDNN conv and NCHW->token layout conversion); max-autotune GEMM | 12.0x |
| llama3_attention_decode | HF LLaMA-3 eager attention + DynamicCache, decode | GQA without repeat_kv (q viewed [B,8,4,D]); split attention over cache + new token instead of cat; merged QKV GEMM | 5.5x |
| edge_proj_agg | GNN message passing (gather-concat-project-GELU-scatter-mean) | projection distributivity: project nodes before gathering (79.5 -> 15 GFLOP, no [E,576] concat), merged node GEMM; max-autotune | 3.5x |
| llama3_lmhead_sampling | HF generate: temperature + top-k + top-p warpers | top-p and softmax on the top-k survivors only (no full-vocab sort); top-k on fp16 logits (16-bit radix select) | 2.3x |
| deepseek_v3_moe_gate | DeepSeek-V3 noaux_tc router | split-precision router GEMM (fp32 weight = fp16 hi + scaled fp16 lo, tensor cores, fp32 out); fusible group top-2 / group selection / index sort; expert selection bit-identical | 2.05x |
| pairwise_topk | kNN retrieval + softmax-weighted aggregation | fp16 GEMM-expanded ranking + 16-bit top-32 shortlist + exact fp32 re-rank with G0's formula; CUDA graphs | 1.8x |
| llama3_attention_prefill | HF LLaMA-3 eager attention, prefill | causal GQA flash attention (SDPA, no [B,H,S,S], no repeat_kv); merged QKV GEMM | 1.5x |
| qknorm | Qwen3 QK-norm + RoPE | Inductor coordinate-descent tuning (structural variants measured slower) | 1.34x |
| llama3_lora_mlp | PEFT LoRA (r=16) on gate/up/down | adapters merged into base weights offline, merged gate/up GEMM | 1.06x |
| gated_mlp | LLaMA-3 SwiGLU MLP | max-autotune GEMMs (GEMMs already ~86% of peak) | 1.04x |
| rmsnorm | LLaMA RMSNorm + residual | tuning only (already ~96% of copy bandwidth) | 1.02x |

### Real-model subgraphs extracted verbatim from library source

G0 for these cases is copied from the released library code (transformers 5.23.0, diffusers 0.43.0;
sources downloaded into `.cache/src/`), with model dimensions from the public configs; only training /
cache / padding branches are dropped.  The source file and class are cited in each `g0.py` docstring.

| case | source (file :: class) | G* transformations (cross-operator) | vs torch.compile (G1 eval) |
|---|---|---|---|
| mamba_mixer | transformers `mamba/modeling_mamba.py` :: MambaMixer + mamba_selective_scan (PyTorch fallback), Mamba-130m | 512-step sequential selective scan -> one associative (parallel) scan (the `use_associative_scan` reformulation), C-contraction as one reduction | 16.2x |
| qwen2_vl_patch_embed | transformers `qwen2_vl/modeling_qwen2_vl.py` :: PatchEmbed, Qwen2-VL-7B vision | Conv3d with kernel == stride on one patch per row == GEMM against the flattened kernel (cuDNN implicit 3-D conv removed) | 7.9x |
| sd_vae_mid_attention | diffusers `models/attention_processor.py` :: Attention + AttnProcessor2_0, SD1.5 VAE mid-block | head_dim-512 attention as tensor-core GEMMs with fp32 scores instead of the sm80 memory-efficient SDPA kernel; GroupNorm affine folded into one merged QKV GEMM | 1.69x |
| bloom_attention | transformers `bloom/modeling_bloom.py` :: alibi + BloomBlock/BloomAttention, BLOOM-7B1 | ALiBi as a flex_attention score_mod with a causal block mask (no [B,H,S,S] scores, no alibi / mask tensors, fp32 scores: accepted by the float64 oracle, G0's bf16 alibi error is ~24x larger) | 1.48x |
| gemma2_attention | transformers `gemma2/modeling_gemma2.py` :: Gemma2Attention (eager, soft-capping), Gemma-2-2B | soft-capped causal GQA attention as a flex_attention score_mod + causal block mask (no repeat_kv, no [S,S] tensors); merged QKV; autotuned flex template | 1.40x |
| t5_self_attention | transformers `t5/modeling_t5.py` :: T5LayerSelfAttention / T5Attention.compute_bias, T5-base | relative position bias depends only on kv-q and saturates beyond 91: offline [H, 257] distance table looked up inside one flex_attention kernel (no [S,S] bucket grid / gather); merged QKV | 1.37x |
| swin_layer | transformers `swin/modeling_swin.py` :: SwinLayer (shifted window), Swin-T stage 1 | relative-position bias + cyclic-shift mask folded offline into one [nW, H, 49, 49] bias; fused window attention (SDPA); merged QKV | 1.27x |
| convnext_layer | transformers `convnext/modeling_convnext.py` :: ConvNextLayer, ConvNeXt-T stage 1 | layer-scale folded into pwconv2; GELU fused into the pwconv1 GEMM epilogue via max-autotune templates (depthwise-conv rewrites lost to cuDNN and were abandoned) | 1.08x |

`summary.md` holds the latest `compare.py` run.
