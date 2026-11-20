"""Isolated worker process for anything that compiles or runs on the GPU.

Parent side: ``run_worker(op, args, gpu=..., timeout=...)`` spawns
``python -m cage.worker <op> <args.json> <out.json>`` with the GPU pinned via
CUDA_VISIBLE_DEVICES and in-project caches, so Inductor config patches, compile
caches and crashes never leak between candidates.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from .ledger import ROOT


def worker_env(gpu: str | None) -> dict:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache" / "triton"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache" / "inductor"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONWARNINGS", "ignore")
    # subgraphs have few kernels; spawning Inductor's parallel compile pool costs more than it saves
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    return env


def run_worker(op: str, args: dict, gpu: str | None, timeout: float = 900, log_dir: Path | None = None,
               wrap: list[str] | None = None) -> dict:
    """Run ``op`` in a subprocess. ``wrap`` prefixes the command (e.g. an ncu invocation)."""
    tmp_root = ROOT / ".cache" / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cage_", dir=tmp_root) as td:
        a = Path(td) / "args.json"
        o = Path(td) / "out.json"
        a.write_text(json.dumps(args, default=str))
        cmd = list(wrap or []) + [sys.executable, "-m", "cage.worker", op, str(a), str(o)]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, env=worker_env(gpu), capture_output=True, text=True, timeout=timeout,
                                  cwd=str(ROOT))
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            rc, stdout, stderr = -9, e.stdout or "", (e.stderr or "") + f"\n[cage] timeout after {timeout}s"
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
        wall = time.time() - t0
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{int(t0)}_{op}.log").write_text(
                f"$ {' '.join(cmd)}\n[rc={rc} wall={wall:.1f}s]\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}")
        if o.exists():
            try:
                result = json.loads(o.read_text())
            except json.JSONDecodeError:
                result = {"ok": False, "error": "worker wrote invalid JSON"}
        else:
            tail = "\n".join((stderr or "").strip().splitlines()[-25:])
            result = {"ok": False, "error": f"worker failed (rc={rc})", "stderr_tail": tail}
        result.setdefault("wall_s", round(wall, 2))
        result["_stdout"] = stdout
        return result


# ---------------------------------------------------------------------- child side

def _dispatch(op: str, args: dict) -> dict:
    if op == "check":
        from .legality import dynamic_check
        return dynamic_check(**args)
    if op == "measure":
        from .measure import measure_worker
        return measure_worker(**args)
    if op == "ctx_backend":
        from .context.backend import backend_context
        return backend_context(**args)
    if op == "ctx_exec":
        from .context.execution import execution_context
        return execution_context(**args)
    if op == "ncu_target":
        from .context.execution import ncu_target
        return ncu_target(**args)
    if op == "cost_backend":
        from .costmodel import backend_cost
        return backend_cost(**args)
    if op == "calibrate":
        from .calibrate import calibrate_worker
        return calibrate_worker(**args)
    raise ValueError(f"unknown worker op {op!r}")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    op, args_path, out_path = argv
    args = json.loads(Path(args_path).read_text())
    try:
        result = _dispatch(op, args)
    except Exception as e:  # report the failure as data; the agent reads it
        result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                  "traceback": "".join(traceback.format_exc().splitlines(True)[-30:])}
    Path(out_path).write_text(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
