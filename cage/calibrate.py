"""Calibrate the cost model on the pinned GPU: achieved HBM bandwidth, FP16 GEMM throughput, launch overhead."""

from __future__ import annotations

import time


def _time_ms(fn, iters: int = 20) -> float:
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def calibrate_worker() -> dict:
    import torch

    from .hw import load_hw

    name = torch.cuda.get_device_name(0)
    hw = load_hw(name)
    n = 256 * 1024 * 1024  # 512 MiB fp16
    x = torch.empty(n, dtype=torch.float16, device="cuda").normal_()
    y = torch.empty_like(x)
    ms = _time_ms(lambda: y.copy_(x))
    bw_gbps = 2 * x.numel() * 2 / (ms * 1e-3) / 1e9
    del x, y

    tflops = {}
    for m, k, nn in ((8192, 8192, 8192), (4096, 4096, 14336), (1024, 4096, 14336)):
        a = torch.randn(m, k, dtype=torch.float16, device="cuda")
        b = torch.randn(k, nn, dtype=torch.float16, device="cuda")
        ms = _time_ms(lambda: torch.mm(a, b))
        tflops[f"{m}x{k}x{nn}"] = round(2 * m * k * nn / (ms * 1e-3) / 1e12, 1)
        del a, b
    gemm_tflops = max(tflops.values())

    t = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    reps = 2000
    t0 = time.perf_counter()
    for _ in range(reps):
        t.add_(1)
    torch.cuda.synchronize()
    launch_us = (time.perf_counter() - t0) / reps * 1e6

    values = {
        "bw_eff": round(min(1.0, bw_gbps / hw["hbm_gbps"]), 3),
        "gemm_eff": round(min(1.0, gemm_tflops / hw["fp16_tflops"]), 3),
        "eager_launch_us": round(launch_us, 2),
        "measured_bw_gbps": round(bw_gbps, 1),
        "measured_gemm_tflops": tflops,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {"ok": True, "device": name, "values": values}
