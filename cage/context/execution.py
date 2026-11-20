"""Execution-level context: per-kernel GPU time (torch.profiler) and optional Nsight Compute metrics.

``execution_context`` runs in a worker: compiles the candidate, profiles N iterations and
reports kernel time shares, achieved bandwidth (joined with backend byte counts), GPU busy
vs wall time (launch-bound indicator).  ``run_ncu`` (parent side) re-runs the candidate
under ``ncu`` restricted to the top kernels and parses occupancy / throughput / stall metrics.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import time
from collections import defaultdict

from .backend import compile_and_capture, summarize_backend


def _kernel_events(prof):
    from torch.autograd import DeviceType

    out = []
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            dur = getattr(e, "device_time", None)
            if dur is None:
                dur = e.cuda_time
            out.append((e.name, float(dur)))
    return out


def execution_context(task_path: str, cand_path: str, iters: int = 20, top: int = 10, peak_gbps: float = 4800.0) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    compiled, inputs, codes, compile_s, cand, ctx = compile_and_capture(task_path, cand_path)
    backend = summarize_backend(codes, torch.cuda.get_device_properties(0).multi_processor_count)
    bytes_by_kernel = {k["name"]: k["bytes"] for k in backend["kernels"] if k["kind"] == "triton"}

    def step():
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        return compiled(*inputs)

    with torch.no_grad():
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            step()
        end.record()
        torch.cuda.synchronize()
        wall_us = start.elapsed_time(end) * 1000 / iters
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(iters):
                step()
            torch.cuda.synchronize()
    ctx.__exit__(None, None, None)

    agg = defaultdict(lambda: [0, 0.0])
    for name, dur in _kernel_events(prof):
        agg[name][0] += 1
        agg[name][1] += dur
    total = sum(v[1] for v in agg.values()) / iters
    kernels = []
    for name, (cnt, dur) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        per_iter = dur / iters
        rec = {"name": name, "calls_per_iter": round(cnt / iters, 2), "us_per_iter": round(per_iter, 2),
               "share": round(per_iter / total, 4) if total else 0.0}
        b = bytes_by_kernel.get(name)
        if b and per_iter > 0:
            rec["io_bytes"] = b
            rec["achieved_gbps"] = round(b / (per_iter * 1e-6) / 1e9, 1)
            rec["bw_util"] = round(rec["achieved_gbps"] / peak_gbps, 3)
        kernels.append(rec)
    return {
        "ok": True,
        "iters": iters,
        "compile_s": round(compile_s, 2),
        "gpu_busy_us_per_iter": round(total, 2),
        "wall_us_per_iter": round(wall_us, 2),
        "gpu_idle_fraction": round(max(0.0, 1 - total / wall_us), 3) if wall_us else None,
        "kernel_launches_per_iter": round(sum(v[0] for v in agg.values()) / iters, 2),
        "kernels": kernels[: max(top, 1) * 3],
        "top_kernels": [k["name"] for k in kernels[:top]],
    }


def ncu_target(task_path: str, cand_path: str, iters: int = 1) -> dict:
    """Worker side of ncu profiling: compile + warm up outside the profiled range, then profile."""
    import torch

    compiled, inputs, _, _, _, ctx = compile_and_capture(task_path, cand_path)
    with torch.no_grad():
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(iters):
            compiled(*inputs)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    ctx.__exit__(None, None, None)
    return {"ok": True}


NCU_METRICS = {
    "Duration": "duration",
    "DRAM Throughput": "dram_throughput_pct",
    "Memory Throughput": "memory_throughput_pct",
    "Compute (SM) Throughput": "sm_throughput_pct",
    "Achieved Occupancy": "achieved_occupancy_pct",
    "Theoretical Occupancy": "theoretical_occupancy_pct",
    "Registers Per Thread": "regs_per_thread",
    "Block Size": "block_size",
    "Grid Size": "grid_size",
    "Waves Per SM": "waves_per_sm",
    "Dynamic Shared Memory Per Block": "dyn_smem_per_block",
    "Block Limit Registers": "block_limit_regs",
    "Block Limit Shared Mem": "block_limit_smem",
    "Warp Cycles Per Issued Instruction": "warp_cycles_per_issue",
    "L2 Hit Rate": "l2_hit_rate_pct",
    "L1/TEX Hit Rate": "l1_hit_rate_pct",
}


def parse_ncu_csv(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    per_kernel: dict[tuple, dict] = {}
    for row in reader:
        kname = row.get("Kernel Name", "?")
        key = (row.get("ID", "0"), kname)
        rec = per_kernel.setdefault(key, {"name": kname})
        metric = row.get("Metric Name", "")
        if metric in NCU_METRICS:
            val = row.get("Metric Value", "")
            unit = row.get("Metric Unit", "")
            try:
                val = float(val.replace(",", ""))
            except ValueError:
                pass
            rec[NCU_METRICS[metric]] = val
            if metric == "Duration":
                rec["duration_unit"] = unit
        sec = row.get("Section Name", "")
        if row.get("Rule Name") and row.get("Rule Description") and sec:
            rec.setdefault("hints", []).append(row["Rule Description"][:200])
    return list(per_kernel.values())


def run_ncu(task_path: str, cand_path: str, kernel_names: list[str], gpu: str, log_dir=None,
            timeout: float = 900) -> dict:
    """Parent side: profile the named kernels with Nsight Compute (expensive; use sparingly)."""
    from ..worker import run_worker

    ncu = shutil.which("ncu")
    if not ncu:
        return {"ok": False, "error": "ncu not found on PATH"}
    names = [re.escape(n) for n in kernel_names if n and not n.startswith("Memcpy")]
    if not names:
        return {"ok": False, "error": "no kernels to profile"}
    wrap = [ncu, "--profile-from-start", "off", "--target-processes", "all", "--csv", "--page", "details",
            "-k", "regex:" + "|".join(n[:120] for n in names),
            "--section", "SpeedOfLight", "--section", "Occupancy", "--section", "LaunchStats",
            "--section", "WarpStateStats", "--section", "MemoryWorkloadAnalysis"]
    t0 = time.time()
    res = run_worker("ncu_target", {"task_path": task_path, "cand_path": cand_path}, gpu=gpu, timeout=timeout,
                     log_dir=log_dir, wrap=wrap)
    stdout = res.pop("_stdout", "")
    if "ERR_NVGPUCTRPERM" in stdout or "ERR_NVGPUCTRPERM" in str(res):
        return {"ok": False, "error": "ncu lacks GPU performance-counter permission (ERR_NVGPUCTRPERM)"}
    kernels = parse_ncu_csv(stdout)
    return {"ok": bool(kernels), "kernels": kernels, "wall_s": round(time.time() - t0, 1),
            "error": None if kernels else ("no ncu rows parsed; " + str(res.get("error") or stdout[-500:]))}


def render(ctx: dict, ncu: dict | None = None) -> str:
    out = ["== EXECUTION CONTEXT =="]
    out.append(f"wall/iter={ctx['wall_us_per_iter']}us  gpu_busy/iter={ctx['gpu_busy_us_per_iter']}us  "
               f"idle_fraction={ctx['gpu_idle_fraction']}  launches/iter={ctx['kernel_launches_per_iter']}  "
               f"(iters={ctx['iters']})")
    if ctx.get("gpu_idle_fraction") and ctx["gpu_idle_fraction"] > 0.3:
        out.append("  ! GPU idle > 30% of wall time: launch/CPU-overhead bound (fewer kernels or CUDA graphs help)")
    out.append("\n-- kernels by GPU time --")
    for k in ctx["kernels"]:
        line = f"  {k['share'] * 100:5.1f}%  {k['us_per_iter']:9.2f}us  x{k['calls_per_iter']}  {k['name'][:110]}"
        if "achieved_gbps" in k:
            line += f"  [{k['achieved_gbps']} GB/s, {k['bw_util'] * 100:.0f}% of peak BW]"
            if k["bw_util"] > 1.0:
                line += " (io over-counted: kernel reads a slice of a larger buffer)"
        out.append(line)
    if ncu is not None:
        out.append("\n-- Nsight Compute --")
        if not ncu.get("ok"):
            out.append(f"  ncu unavailable: {ncu.get('error')}")
        for k in ncu.get("kernels", []):
            keys = ["duration", "dram_throughput_pct", "sm_throughput_pct", "achieved_occupancy_pct",
                    "theoretical_occupancy_pct", "regs_per_thread", "block_size", "grid_size", "waves_per_sm",
                    "warp_cycles_per_issue", "l2_hit_rate_pct"]
            vals = "  ".join(f"{kk}={k[kk]}" for kk in keys if kk in k)
            out.append(f"  {k['name'][:100]}\n      {vals}")
            for h in k.get("hints", [])[:3]:
                out.append(f"      hint: {h}")
    return "\n".join(out)
