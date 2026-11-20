# Structural goal catalog

Each goal is a **desired graph property**. The trigger column says which context evidence
suggests it. "Typical steps" lists the subgoals, often non-monotonic: an intermediate step
may be slower until the last step lands. Check every predicate against `ctx graph` or
`ctx backend` output, not against a feeling.

| # | Goal (structure) | Trigger evidence | Typical steps (dependencies) | Success predicate | Risk / notes |
|---|---|---|---|---|---|
| 1 | **Eliminate large intermediates** | `largest intermediates` dominated by one tensor (e.g. fp32 upcast of [T, I], attention scores [B,H,S,S]) consumed by a single chain | (a) reorder ops so the reduction/cast happens before the expansion; (b) merge producer+consumer into one expression Inductor can fuse; (c) fold casts | tensor no longer appears as a kernel output in `ctx backend` | changing reduction order changes rounding, so check with `cage check` |
| 2 | **Expose producer-consumer fusion** | `ctx backend` shows two kernels where the second reads exactly what the first wrote; GEMM prologue/epilogue candidates | (a) remove the op that blocks fusion (a `.contiguous()`, view-incompatible reshape, dtype round-trip, extern op); (b) restructure so both sides share an iteration space; (c) optionally enable `max-autotune` templates for GEMM epilogue fusion | kernel count drops; intermediate not materialized | templates trade cuBLAS speed for fusion, so eval it |
| 3 | **Reduce layout conversions** | `layout copies` list; `clone`/`contiguous` of permute/transpose; strided outputs | (a) move the permute to the cheapest tensor (weights via `convert_weights`, or the smallest activation); (b) make producers write the consumer layout directly; (c) push transposes into GEMM operands (`mm` accepts transposed views for free) | no `triton_poi_*clone*` / transpose-copy kernels | the output layout must still match G0 (shape/dtype; strides are free) |
| 4 | **Merge parallel GEMMs** | `parallel GEMMs sharing an input` (QKV, gate/up) | (a) concatenate weights in `convert_weights`; (b) one GEMM plus split views; (c) fuse the post-split pointwise into one kernel | one GEMM per shared input | for skinny M (decode), merged GEMM may pick a worse cuBLAS kernel, so eval |
| 5 | **Avoid recomputation / dedupe** | `duplicated sub-expressions`; same normalization or rotation applied twice; recomputation inside fused kernels (overlapping windows) | (a) CSE in source; (b) hoist shared terms; (c) for pooling/windowed ops, materialize once instead of recomputing per window | duplicate list empty; kernel bytes drop | sometimes recompute is cheaper than a write+read; let measurements decide |
| 6 | **Algebraic simplification** | chains like `x*a*b`, `(x - mean)/std` recomputed, `exp/log` pairs, `softmax` followed by `log`, masked ops on known masks, matmul with a constant / identity part | rewrite with identities (associativity of scalar scaling, folding constants into weights, `log_softmax`, `addmm` for bias, `baddbmm` for scaled scores) | fewer nodes / bytes | floating-point reassociation, so check tolerance |
| 7 | **Move work to the cheaper side of a GEMM** | elementwise scale or bias on a GEMM output whose reduction dim is large; normalization weight multiplied into the activation before the GEMM | fold per-channel scale into the weights (`convert_weights`); apply per-row scale after the GEMM on the smaller tensor | pointwise kernel removed or shrunk | only valid for linear ops; per-row vs per-column matters |
| 8 | **Reduce launch overhead** | `ctx exec` idle fraction > 30%, many tiny kernels, small shapes (decode) | (a) fuse tiny kernels; (b) horizontal fusion of independent small ops (`combo_kernels`); (c) `COMPILE_MODE="reduce-overhead"` (CUDA graphs) | launches/iter down; idle fraction down | CUDA graphs need static shapes (fine here) |
| 9 | **Better reduction strategy** | `ctx backend`: reduction kernel with low waves (<1), 16 warps and XBLOCK=1, or spills; `ctx exec`: reduction at low % of peak BW | split very long reductions or keep rows persistent (`triton.persistent_reductions`, `split_reductions`, `coordinate_descent_tuning`); restructure row layout so the reduced dim is contiguous | achieved GB/s near the copy bandwidth (about 3.8 TB/s measured) | config knobs are backend context; prefer them only after structure is right |
| 10 | **GEMM efficiency** | `ctx exec`: GEMMs dominate and run well below about 130 TFLOPS on H20; odd shapes (not multiples of 64/128), batched small GEMMs | reshape batched GEMMs into one big GEMM ([B*S, K] x [K, N]); pad or align shapes; try `max-autotune` GEMM templates; avoid `bmm` on non-contiguous operands | TFLOPS up in `ctx exec` | usually small gains once shapes are good |
| 11 | **Precision-preserving cast placement** | fp16 to fp32 to fp16 round-trips around cheap ops (`ctx graph` casts) | keep fp32 only where G0 needs accumulation precision (reductions, softmax, norm statistics); do elementwise in the input dtype when G0's result is identical | fewer/lower-width intermediates | must keep G0's numerics within tolerance; the oracle check helps |

## Composing goals (long-horizon examples)
- **Gated MLP**: Goal 4 (merge gate/up). The merged GEMM is about as fast, and the split
  views may first add a copy (temporary slowdown). Then Goal 2/1: make the SiLU-multiply
  read the merged buffer directly, with no fp32 materialization. Then Goal 7: fold the
  RMSNorm weight into the gate/up weights.
- **QK-norm + RoPE**: Goal 3 (write q/k/v once in [B,H,S,D] instead of norm, then
  transpose, then contiguous). Then Goal 2 (single fused norm+rope kernel per q/k reading
  the packed qkv). Then Goal 8 (one horizontal kernel for q, k and v).
- **Attention-like blocks**: Goal 1 (never materialize [B,H,S,S] in fp32: use
  `F.scaled_dot_product_attention`, which is graph-level and allowed) before tuning
  anything else.
