"""On-device latency measurement (worker side) and GPU selection (parent side).

Targets are measured in interleaved rounds inside one process so that every candidate is
compared against eager G0 (and Inductor-compiled G0) under the same GPU conditions -- the
H20s in this machine are shared, so absolute latencies drift.  Each timed call is preceded
by an L2 flush and timed with CUDA events; we report median and IQR over all samples.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import time

L2_FLUSH_BYTES = 256 * 1024 * 1024


def pick_gpu() -> str:
    """Least-loaded visible GPU by utilization + memory, unless CAGE_GPU is set."""
    if os.environ.get("CAGE_GPU"):
        return os.environ["CAGE_GPU"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        best, score = "0", float("inf")
        for line in out.strip().splitlines():
            idx, util, used, total = [x.strip() for x in line.split(",")]
            s = float(util) + 50.0 * float(used) / max(1.0, float(total))
            if s < score:
                best, score = idx, s
        return best
    except Exception:
        return "0"


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def measure_worker(task_path: str, targets: list[dict], rounds: int = 5, round_ms: float = 150.0,
                   max_iters: int = 200) -> dict:
    """targets: [{"label": str, "path": str, "mode": "eager"|"compiled"}]."""
    import torch

    from .task import (build_candidate, build_reference, compile_model, inductor_config, init_args,
                       load_spec, make_inputs)

    t_start = time.time()
    task = load_spec(task_path)
    ref = build_reference(task)
    inputs = make_inputs(task, 0)
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.int8, device="cuda")

    def alias_reference():
        # Compiled G0 is the same graph and the same weights as eager. Alias those parameters
        # instead of keeping another full copy: four resident 8B models do not fit beside other
        # jobs on one H20, and the forwards only read the weights.
        torch.manual_seed(0)
        model = task.model_cls(*init_args(task)).cuda().eval()
        model.load_state_dict(ref.state_dict(), assign=True)
        torch.cuda.empty_cache()
        return model

    runners = []
    for tg in targets:
        spec = load_spec(tg["path"])
        if spec.path == task.path and tg["mode"] == "eager":
            model = ref
        elif spec.path == task.path:
            model = alias_reference()
        else:
            model = build_candidate(task, spec, ref)
            torch.cuda.empty_cache()
        # optional "cfg_path": take INDUCTOR_CONFIG / COMPILE_MODE from another file (e.g. run G0's
        # graph with G*'s compile settings, so only the graph differs between the two targets)
        cfg_spec = load_spec(tg["cfg_path"]) if tg.get("cfg_path") else spec
        cfg = cfg_spec.inductor_config if tg["mode"] == "compiled" else None
        fn = compile_model(model, cfg_spec) if tg["mode"] == "compiled" else model
        mark = hasattr(torch.compiler, "cudagraph_mark_step_begin") and tg["mode"] == "compiled"

        def call(fn=fn, mark=mark):
            if mark:
                torch.compiler.cudagraph_mark_step_begin()
            return fn(*inputs)

        t0 = time.time()
        with inductor_config(cfg), torch.no_grad():
            for _ in range(3):
                call()
            torch.cuda.synchronize()
        compile_s = time.time() - t0
        # rough per-call time to size the rounds
        with inductor_config(cfg), torch.no_grad():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(5):
                call()
            e.record()
            torch.cuda.synchronize()
        est_ms = max(s.elapsed_time(e) / 5, 1e-3)
        iters = int(min(max_iters, max(5, round_ms / est_ms)))
        runners.append({"label": tg["label"], "call": call, "cfg": cfg, "iters": iters,
                        "samples": [], "round_medians": [], "compile_s": round(compile_s, 2)})

    for _ in range(rounds):
        for r in runners:
            n = r["iters"]
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            with inductor_config(r["cfg"]), torch.no_grad():
                for i in range(n):
                    flush.zero_()
                    starts[i].record()
                    r["call"]()
                    ends[i].record()
                torch.cuda.synchronize()
            ts = [a.elapsed_time(b) for a, b in zip(starts, ends)]
            r["samples"].extend(ts)
            r["round_medians"].append(statistics.median(ts))

    results = {}
    for r in runners:
        xs = r["samples"]
        results[r["label"]] = {
            "median_ms": round(statistics.median(xs), 5),
            "p25_ms": round(_quantile(xs, 0.25), 5),
            "p75_ms": round(_quantile(xs, 0.75), 5),
            "mean_ms": round(statistics.fmean(xs), 5),
            "n": len(xs),
            "round_medians_ms": [round(x, 5) for x in r["round_medians"]],
            "compile_and_warmup_s": r["compile_s"],
        }
    return {"ok": True, "device": torch.cuda.get_device_name(0), "targets": results,
            "wall_s": round(time.time() - t_start, 2)}
