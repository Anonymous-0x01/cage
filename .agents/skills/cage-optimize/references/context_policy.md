# Adaptive multi-level context acquisition policy

Context is not free: it costs wall-clock budget, and too much of it dilutes your attention.
Acquire evidence **when it can change a decision**, cheapest level first. Every query must
carry `--why "<decision it resolves>"`; the ledger counts queries per level.

## Levels and costs
| level | command | contents | cost |
|---|---|---|---|
| graph | `cage ctx graph` | aten ops, shapes, dtypes, strides, edges, intermediates, layout copies, casts, parallel GEMMs, duplicates, GEMM boundaries | ~5 s, no GPU |
| estimate | `cage cost` | simulated kernels, traffic, materialized bytes, est. speedup vs G0 | ~5 s, no GPU |
| backend | `cage ctx backend` | real Inductor kernels: what each fuses, I/O bytes, XBLOCK/R0_BLOCK, warps, stages, regs, spills, smem, grid waves; generated code saved (use `--show-source <kernel>` to read one) | 10-30 s |
| estimate+ | `cage cost --backend` | estimate from Inductor's real kernel list | ~20 s |
| execution | `cage ctx exec` | per-kernel GPU time share, achieved GB/s and % of peak, GPU idle fraction, launches/iter | 20-40 s |
| execution+ | `cage ctx exec --ncu` | Nsight Compute occupancy/throughput/stalls (**not permitted on this machine**) | n/a |
| ground truth | `cage eval` | measured latency vs eager and Inductor G0 (same session) | 20-90 s + 1 eval |

## Decision rules
1. **Start of run**: `ctx graph` always. Add `ctx backend` on G0 when the graph is
   memory-bound or has more than about 5 non-view ops, so you know what Inductor already
   fuses. This decides which goals are real.
2. **Choosing among goals**: use the cost model on G0 (bytes and kernels per group) and the
   graph summary. Only use `ctx exec` on G0 if the cost model cannot tell you which part
   dominates (for example, GEMM-heavy graphs where cuBLAS efficiency is unknown).
3. **Before a long dependent chain** (3 or more steps): sketch the final structure as a
   candidate if you can and `cost` it. If the estimate shows no gain, don't start the chain
   without another reason.
4. **After a structural step**: `cost` (cheap) to confirm the structure changed as intended:
   kernel count, materialized bytes. If the simulated structure doesn't show the change,
   check with `ctx backend`, because Inductor may differ from the simulation.
5. **Eval disagrees with the estimate (>15%) or a step is unexpectedly slow**:
   - `ctx backend` on the candidate: did fusion happen? Are there new copy kernels,
     spills, under-filled grids, or extern fallbacks?
   - then `ctx exec` if the backend looks fine: which kernel got slower, and its achieved GB/s?
   - Record the explanation in `plan.md` before acting on it.
6. **Fusion looks profitable on paper but risky** (large fused reduction, many loads,
   overlapping windows): `ctx backend` to inspect registers/spills **before** spending an
   eval. High regs (>128) or spills mean a likely slowdown.
7. **Launch-bound suspicion** (small tensors, many kernels): `ctx exec` idle fraction
   decides between fusion work and `reduce-overhead`.
8. **Don't**:
   - re-query an unchanged candidate (results are in `runs/<name>/ctx/`)
   - run `ctx exec` on every candidate
   - collect all levels up front "just in case"
