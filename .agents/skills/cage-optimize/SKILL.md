---
name: cage-optimize
description: Optimize a PyTorch computation subgraph or LLM block with the CAGE workflow, which combines context-aware, goal-directed long-horizon planning with graph rewrites compiled by TorchInductor. Use when asked to optimize, speed up, or rewrite a subgraph, task file, or model module in this repository.
---

# CAGE optimization workflow

You solve a **sequential decision problem**. At each step t the state is `s_t = (G_t, C_t)`,
where `G_t` is the working graph (a candidate file) and `C_t` is the context you have
acquired. Each action is one of two kinds:

- **Transformation action**: write a new candidate (`bin/cage new` plus an edit), which changes `G_t`.
- **Context action**: `bin/cage ctx graph|backend|exec`, `bin/cage cost`, or `bin/cage eval`, which enriches `C_t`.

The objective is the fastest semantically equivalent graph among the graphs you evaluate,
within the budget. Intermediate graphs may be slower; G* is tracked separately, so
exploring a temporarily worse graph never loses the best solution.

Three principles distinguish CAGE from greedy or tree search. Apply all three.
1. **Goal-directed**: decide what the graph should look like (a structural goal) before
   deciding what to try next. Judge each step by its progress toward that structure, not
   by whether it is immediately faster.
2. **Context-aware, on demand**: start from cheap graph context. Escalate to backend or
   execution context only when that evidence would change a decision
   (see `references/context_policy.md`).
3. **Feedback-driven replanning**: measured results update your beliefs. Revise or abandon
   goals when evidence invalidates them, and backtrack to an earlier candidate when needed.

## Step 0: Ingest the target
- If the user gives a task file (KernelBench-style `Model`, `get_inputs`, `get_init_inputs`),
  use it as is.
- If the user gives LLM or model code, or names a module or function, extract the subgraph
  into `tasks/<name>.py` following `references/task_contract.md`. Use the realistic shapes and
  dtype the user states; otherwise use FP16, batch 1, sequence 1024, and the model's config
  dims. Keep the semantics exactly; this file is G0.
- Start the run with `bin/cage init tasks/<name>.py --name <name> --budget-min <B> --max-evals <E>`,
  using the user's budget (default 45 min, 40 evals). Read the baselines it prints.

## Step 1: Parse graph context
- Run `bin/cage ctx graph --why "initial graph context"`.
- Write the key facts into `runs/<name>/plan.md` under "Graph context summary":
  - dominant cost (GEMM FLOPs or bytes moved)
  - largest intermediates
  - layout copies and casts
  - parallel GEMMs that share an input
  - duplicated sub-expressions
  - fusion boundaries around GEMMs
- If G0 is a memory-bound chain, you usually also want to know how Inductor already fuses
  it, because Inductor may already achieve the obvious structure. A single
  `bin/cage ctx backend --why "what does Inductor already fuse?"` is usually worth its cost
  here.

## Step 2: Define goals (structural, not numeric)
Infer desirable graph properties from tensor semantics and bottlenecks, using
`references/goal_catalog.md`. Rank them by expected gain divided by effort and risk. For
each goal you pursue, write the following into `plan.md`:
- **Hypothesis**: which bottleneck it removes and why it should be faster (bytes, kernels,
  FLOPs, launch overhead).
- **Success predicate**: a checkable structure, for example "no fp32 [T,14336]
  intermediate materialized", "gate and up computed by one GEMM", "at most 2 kernels",
  or "no transpose copy before the output".
- **Expected gain**: from roofline reasoning or `bin/cage cost` on a quick sketch.
- **Subgoals with dependencies**: the ordered transformation steps (G1.1, then G1.2
  depending on G1.1, and so on). Note which steps may be **temporarily slower**, for
  example a split that exposes a later fusion.

Then record it with `bin/cage goal G1 defined "<one-line goal + predicate>"`.

## Step 3: Plan and transform (one subgoal step per candidate)
For each step:
1. `bin/cage new <slug> --from <parent> --goal G1` creates a copy; edit only that file.
   See `references/transformations.md` for safe rewrite patterns and Inductor knobs.
2. `bin/cage check` must pass. If it fails, fix the candidate; don't move on with an
   illegal graph.
3. `bin/cage cost --goal G1 --step k/n --intent "..."` runs the cheap estimate. For an
   **intermediate step** it only has to show the structural progress you expected (kernel
   count, materialized bytes, traffic). A predicted slowdown is acceptable if the plan says
   so. Record it in `plan.md` and keep going.
4. Run `bin/cage eval --goal G1 --step k/n --intent "..."` (it costs hardware budget) when:
   - the goal's success predicate is reached (a goal checkpoint), or
   - the cost model is uncertain for this kind of change (GEMM shape changes, layout and
     stride effects, reduction strategy, Inductor config knobs, cuBLAS versus Triton
     choice), or
   - you need ground truth before investing in a long dependent chain.

   Don't spend evals on steps whose only purpose is to set up a later step, unless you
   need the information.
5. After each eval, log progress: `bin/cage goal G1 progress "<what the step showed>" --cand cNNN`.

## Step 4: Acquire context adaptively
Use `references/context_policy.md`. In short:
- If the cost model predicted a gain but `eval` disagrees by more than 15% (eval prints a
  hint), find out why before trying variants. Use `ctx backend` to see what Inductor
  actually fused, block sizes, registers, spills and under-filled grids. Use `ctx exec` to
  see which kernel dominates and its achieved GB/s.
- To decide between two transformation alternatives, use the cheapest context that
  separates them; often that is `cost` on both sketches.
- Always pass `--why` naming the decision the query resolves. Don't re-query unchanged
  candidates; the outputs are saved in `runs/<name>/ctx/`.

## Step 5: Refine, revise, backtrack
- **Goal reached and faster**: run `bin/cage goal G1 reached "..." --cand cNNN`, then choose
  the next goal (Step 2) from the new graph context of the new working graph.
- **Goal reached but not faster**: acquire backend or execution context to explain why.
  Then either add a subgoal (for example, fix a layout copy the restructuring introduced)
  or mark the goal revised or abandoned with the reason. Abandoning keeps G*; continue
  from the best or from an earlier candidate (`bin/cage work cNNN --why ...`).
- **Evidence invalidates an assumption** (for example, Inductor already fuses this, or the
  real bottleneck is elsewhere): log `bin/cage goal G1 revised "<new hypothesis>"`, update
  `plan.md`, and replan. Don't keep polishing a goal whose premise is false.
- Combine goals. Later goals should start from the best structurally sound candidate,
  which is not always G*. A slightly slower candidate that exposes a bigger opportunity is
  a valid parent.

## Termination
Stop when:
- the budget is exhausted (eval refuses), or
- no remaining goal has a credible expected gain above measurement noise (about 2%), or
- the time left cannot fit another goal checkpoint.

Then:
1. Run `bin/cage report`.
2. Finish `plan.md` "Final summary": G* id and file, speedup vs eager and vs Inductor G0,
   the goal history (what worked, what was revised and why), and the transformations in G*.
3. Reply to the user with the same summary plus the path to `runs/<name>/best.py`.

## Quality bar
- Every claim of speedup must come from `bin/cage eval` output in this run.
- Prefer transformations that generalize (they keep working with other batch and sequence
  sizes) over shape-specific hacks, unless the user asks for a fixed shape.
- Keep candidates readable: short comments only where a rewrite's equivalence is not
  obvious (for example, an algebraic identity).
