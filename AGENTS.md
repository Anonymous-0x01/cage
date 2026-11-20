# CAGE harness: context-aware, goal-directed computation graph optimization

This repository turns Codex into **CAGE**, an agent that optimizes PyTorch computation
subgraphs (LLM building blocks) through semantics-preserving graph transformations
compiled by TorchInductor. You, the agent, are the planner: you acquire context, set
structural goals, plan multi-step transformation sequences, and use the `cage` tools to
check, estimate and measure candidates. G0 is the original graph; G* is the best graph
measured so far.

**When the user asks you to optimize a subgraph, model block or task file, first read
`.agents/skills/cage-optimize/SKILL.md` and follow that workflow exactly.**

## Environment
- Python env: conda env `clo` (torch 2.13 + CUDA 12.9 + Triton). `bin/cage` activates it
  automatically; in your own shell commands use `source scripts/env.sh` first.
- GPUs: NVIDIA H20-3e (78 SMs, ~4.8 TB/s HBM3e, 148 dense FP16 TFLOPS). The GPUs are
  **shared with other jobs**, so latency is noisy: trust same-session ratios (`cage eval`
  re-measures eager G0 and Inductor G0 next to every candidate), not absolute numbers.
- Nsight Compute (`ncu`) exists but lacks performance-counter permission on this machine;
  `cage ctx exec --ncu` will report that. Use `cage ctx backend` + `cage ctx exec` instead.
- Work only inside this folder. Caches live in `.cache/`, runs in `runs/<name>/`.

## Tools (all via `bin/cage`, the run defaults to the last `cage init`)
| command | what it gives | typical cost |
|---|---|---|
| `bin/cage init <task.py> --name N --budget-min 45 --max-evals 40` | run dir, baselines (eager, Inductor G0), budget | 1-3 min |
| `bin/cage new <slug> [--from cNNN] [--goal G1]` | copies a candidate to edit (becomes the working graph) | instant |
| `bin/cage ctx graph [cand] --why "..."` | **graph context**: aten ops, shapes/dtypes/strides, edges, large intermediates, layout copies, casts, parallel GEMMs, duplicates, GEMM prologue/epilogue boundaries | ~5 s |
| `bin/cage ctx backend [cand] --why "..." [--show-source NAME]` | **backend context**: Inductor kernels, what each fuses, bytes, block sizes, warps, regs, spills, grid waves; full generated code saved | ~10-30 s |
| `bin/cage ctx exec [cand] --why "..." [--ncu]` | **execution context**: per-kernel GPU time share, achieved GB/s, GPU idle fraction | ~20-40 s |
| `bin/cage check [cand] [--fast]` | **legality checker** (static rules + graph breaks + numerics vs G0 on 3 seeds) | ~10-40 s |
| `bin/cage cost [cand] [--backend]` | **cheap cost model** (tier 0: FX roofline with simulated fusion; tier 1: real Inductor kernels) and estimated speedup vs G0 | ~5 s / ~20 s |
| `bin/cage eval [cand] --goal G1 --step 2/3 --intent "..."` | legality, then cost, then on-device measurement; updates G*, charges 1 hardware eval | ~20-90 s |
| `bin/cage goal G1 defined/progress/reached/revised/abandoned "text"` | records plan events (goal history, trajectory annotations) | instant |
| `bin/cage work cNNN --why "..."` | switch/backtrack the working graph | instant |
| `bin/cage status` / `bin/cage report` | candidates, budget, G*; final report with trajectory plot | instant |

Candidates may be named `c003`, `3`, `g0`, `best`, `working`, or by file path.

## Hard rules
1. **Transformation space**: graph-level PyTorch rewrites only, plus `INDUCTOR_CONFIG`
   and `COMPILE_MODE` in the candidate file. No Triton/CUDA/C++ kernels, no custom ops,
   no `torch.compile` inside the candidate, and no stateful caching across calls. The
   legality checker enforces this.
2. Candidates live in `runs/<name>/candidates/` and keep `class Model` with the **same
   constructor arguments and forward signature** as G0. Inputs, init args and tolerance
   always come from G0. Re-laid-out weights (concatenated, transposed, pre-scaled) must
   be produced by `convert_weights(ref_state_dict)`.
3. Never edit `runs/<name>/task.py`, the files under `tasks/` that an active run uses,
   or the harness code in `cage/` during an optimization session.
4. Every candidate you want to keep must pass `bin/cage check`. Only `bin/cage eval` is
   ground truth; cost-model numbers are estimates.
5. Respect the budget. When `cage eval` reports BUDGET EXHAUSTED, stop, run
   `bin/cage report`, and summarize.
6. Keep `runs/<name>/plan.md` up to date. It is the agent's memory: goals, subgoals,
   evidence and revisions.

## Repository map
- `cage/`: harness. `task.py` (contract), `ledger.py` (run state, G*), `context/{graph,backend,execution}.py`,
  `legality.py`, `costmodel.py`, `measure.py`, `cli.py`, `report.py`, `hw/` (H20 specs).
- `tasks/`: example subgraphs (`rmsnorm.py`, `qknorm.py`, `gated_mlp.py`).
- `.agents/skills/cage-optimize/`: the CAGE workflow and its references.
- `prompts/optimize.md`, `scripts/run_codex.sh`: headless launch.
