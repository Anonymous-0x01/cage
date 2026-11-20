"""Backend-level context: compile a candidate with TorchInductor and inspect what it generated.

Reports the generated kernels (Triton / extern library calls / aten fallbacks), the FX
source nodes fused into each kernel, per-kernel buffer traffic, launch configuration
(block sizes, num_warps, num_stages), register usage / spills / shared memory, and a
grid-fill estimate against the SM count.  Runs inside a worker subprocess.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from ..task import (build_candidate, build_reference, compile_model, inductor_config, load_spec,
                    make_inputs)

_DTYPE_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int64": 8, "int32": 4, "int16": 2, "int8": 1,
    "uint8": 1, "bool": 1, "float8_e4m3fn": 1, "float8_e5m2": 1, "complex64": 8,
}
_SIZE_RE = r"\(([^()]*)\)"


def _prod(dims: str) -> int:
    p = 1
    for d in dims.split(","):
        d = d.strip()
        if d:
            try:
                p *= int(d)
            except ValueError:
                return 0
    return p


def parse_wrapper(code: str) -> dict:
    """Parse an Inductor output-code module: buffers, kernel launches and fused source nodes."""
    buffers: dict[str, tuple[int, int]] = {}  # name -> (numel, element size)
    for m in re.finditer(r"(\w+) = rand_strided\(" + _SIZE_RE + r", " + _SIZE_RE + r", device='[^']*', "
                         r"dtype=torch\.(\w+)\)", code):
        buffers[m.group(1)] = (_prod(m.group(2)), _DTYPE_BYTES.get(m.group(4), 2))
    for m in re.finditer(r"(\w+) = empty_strided_\w+\(" + _SIZE_RE + r", " + _SIZE_RE + r", torch\.(\w+)\)", code):
        buffers[m.group(1)] = (_prod(m.group(2)), _DTYPE_BYTES.get(m.group(4), 2))

    for m in re.finditer(r"(\w+) = reinterpret_tensor\((\w+), " + _SIZE_RE + r", " + _SIZE_RE
                         + r", \d+\); del \w+\s+# reuse", code):
        if m.group(2) in buffers:
            buffers[m.group(1)] = (_prod(m.group(3)), buffers[m.group(2)][1])
    for m in re.finditer(r"(\w+) = (\w+); del \w+\s+# reuse", code):
        if m.group(2) in buffers:
            buffers[m.group(1)] = buffers[m.group(2)]

    sources = {}
    signatures = {}
    for m in re.finditer(r"(triton_\w+) = async_compile\.triton\('\w+', '''(.*?)''', device_str", code, re.S):
        sources[m.group(1)] = m.group(2)
        sig = re.search(r"def \w+\(([^)]*)\)", m.group(2))
        if sig:
            signatures[m.group(1)] = [a.split(":")[0].strip() for a in sig.group(1).split(",")]

    call_idx = code.find("def call(")
    body = code[call_idx:] if call_idx >= 0 else code
    end = body.find("\ndef ", 10)
    if end > 0:
        body = body[:end]
    launches = []
    pending_src = None
    pending_aten = None
    for line in body.splitlines():
        s = line.strip()
        m = re.match(r"# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]", s)
        if m:
            pending_src, pending_aten = m.group(1), m.group(2)
            continue
        kind = None
        mk = re.match(r"(triton_\w+)\.run\((.*)\)", s)
        if mk:
            kind, name, argstr = "triton", mk.group(1), mk.group(2)
        else:
            mk = re.match(r"(?:\w+ = )?extern_kernels\.(\w+)\((.*)\)", s)
            if mk:
                kind, name, argstr = "extern", "extern_kernels." + mk.group(1), mk.group(2)
            else:
                mk = re.match(r"(?:\w+ = )?torch\.ops\.(\w+\.\w+(?:\.\w+)?)\((.*)\)", s)
                if mk:
                    kind, name, argstr = "fallback", mk.group(1), mk.group(2)
        if kind is None:
            continue
        argstr = re.sub(r"stream=\w+", "", argstr)
        nbytes = 0
        seen = set()
        for rm in re.finditer(r"reinterpret_tensor\((\w+), " + _SIZE_RE, argstr):
            base = rm.group(1)
            if base in buffers and base not in seen:
                numel, elem = buffers[base]
                nbytes += min(numel, _prod(rm.group(2))) * elem
                seen.add(base)
        params = signatures.get(name, [])
        positional = [a.strip() for a in argstr.split(",")] if kind == "triton" else []
        for tok in re.findall(r"\b(arg\w+|buf\w+|primals_\w+)\b", argstr):
            if tok in buffers and tok not in seen:
                numel, elem = buffers[tok]
                nbytes += numel * elem
                seen.add(tok)
                if tok in positional and positional.index(tok) < len(params) \
                        and params[positional.index(tok)].startswith("in_out_ptr"):
                    nbytes += numel * elem
        launches.append({
            "kind": kind, "name": name, "bytes": int(nbytes), "buffers": sorted(seen),
            "source_nodes": pending_src, "aten": pending_aten,
        })
        pending_src = pending_aten = None
    return {"buffers": {k: n * e for k, (n, e) in buffers.items()}, "launches": launches, "sources": sources}


def _kernel_objects(names: set[str]) -> dict:
    from torch._inductor.codecache import PyCodeCache

    found = {}
    mods = PyCodeCache.modules
    mods = mods.values() if isinstance(mods, dict) else mods
    for mod in mods:
        for k, v in vars(mod).items():
            if k in names and hasattr(v, "launchers"):
                found[k] = v
    return found


def kernel_meta(obj, sm_count: int) -> dict:
    meta = {"size_hints": dict(getattr(obj, "size_hints", {}) or {})}
    im = getattr(obj, "inductor_meta", {}) or {}
    meta["num_load"] = im.get("num_load")
    meta["num_reduction"] = im.get("num_reduction")
    if obj.launchers:
        L = obj.launchers[0]
        cfg = L.config
        meta["config"] = dict(cfg.kwargs)
        meta["num_warps"] = cfg.num_warps
        meta["num_stages"] = cfg.num_stages
        meta["n_regs"] = getattr(L, "n_regs", None)
        meta["n_spills"] = getattr(L, "n_spills", None)
        meta["shared"] = getattr(L, "shared", None)
        programs = 1
        for dim, size in meta["size_hints"].items():
            if dim.startswith("r"):
                continue
            block = cfg.kwargs.get(dim.upper() + "BLOCK", 1)
            programs *= max(1, -(-int(size) // int(block)))
        meta["programs"] = programs
        meta["waves"] = round(programs / max(1, sm_count), 2)
        if meta["n_regs"]:
            threads = cfg.num_warps * 32
            blocks_by_regs = 65536 // max(1, meta["n_regs"] * threads)
            meta["max_blocks_per_sm_by_regs"] = blocks_by_regs
    return meta


def compile_and_capture(task_path: str, cand_path: str, seed: int = 0):
    """Build + compile a candidate, run once, return (compiled_fn, inputs, codes, compile_s, cand)."""
    import torch
    from torch._inductor.utils import run_and_get_code

    task = load_spec(task_path)
    cand = load_spec(cand_path)
    ref = build_reference(task)
    model = build_candidate(task, cand, ref)
    del ref
    inputs = make_inputs(task, seed)
    ctx = inductor_config(cand.inductor_config)
    ctx.__enter__()
    compiled = compile_model(model, cand)
    t0 = time.time()
    with torch.no_grad():
        _, codes = run_and_get_code(compiled, *inputs)
    torch.cuda.synchronize()
    return compiled, inputs, codes, time.time() - t0, cand, ctx


def summarize_backend(codes: list[str], sm_count: int) -> dict:
    kernels = []
    all_sources = {}
    for code in codes:
        parsed = parse_wrapper(code)
        all_sources.update(parsed["sources"])
        kernels.extend(parsed["launches"])
    objs = _kernel_objects({k["name"] for k in kernels if k["kind"] == "triton"})
    for k in kernels:
        if k["kind"] == "triton" and k["name"] in objs:
            k.update(kernel_meta(objs[k["name"]], sm_count))
        k["type"] = _ktype(k)
    return {"kernels": kernels, "sources": all_sources}


def _ktype(k: dict) -> str:
    if k["kind"] != "triton":
        return k["kind"]
    m = re.match(r"triton_(\w+?)_", k["name"])
    return {"poi": "pointwise", "red": "reduction", "per": "persistent_reduction", "tem": "template",
            "for": "foreach", "spl": "split_scan"}.get(m.group(1) if m else "", "triton")


def backend_context(task_path: str, cand_path: str, out_dir: str | None = None, sm_count: int = 78) -> dict:
    import torch
    import torch._inductor.metrics as metrics

    metrics.reset()
    _, _, codes, compile_s, cand, ctx = compile_and_capture(task_path, cand_path)
    try:
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        pass
    info = summarize_backend(codes, sm_count)
    ctx.__exit__(None, None, None)
    code_path = None
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        code_path = str(Path(out_dir) / (Path(cand_path).stem + "_inductor_output.py"))
        Path(code_path).write_text("\n\n# ======== next graph ========\n\n".join(codes))
    warnings = []
    for k in info["kernels"]:
        if k.get("n_spills"):
            warnings.append(f"{k['name']}: {k['n_spills']} register spills")
        if k.get("n_regs") and k["n_regs"] >= 128:
            warnings.append(f"{k['name']}: high register pressure ({k['n_regs']} regs/thread)")
        if k.get("waves") is not None and k["waves"] < 0.5:
            warnings.append(f"{k['name']}: under-filled grid ({k.get('programs')} programs, {k['waves']} waves)")
        if k["kind"] == "fallback":
            warnings.append(f"{k['name']}: aten fallback (not code-generated, cannot fuse)")
    return {
        "ok": True,
        "compile_s": round(compile_s, 2),
        "compile_mode": cand.compile_mode,
        "inductor_config": cand.inductor_config,
        "generated_kernel_count": metrics.generated_kernel_count,
        "launch_count": len(info["kernels"]),
        "kernels": info["kernels"],
        "kernel_sources": info["sources"],
        "output_code": code_path,
        "warnings": warnings,
        "sm_count": sm_count,
    }


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:.1f}{unit}" if unit != "B" else f"{int(b)}B"
        b /= 1024
    return f"{b:.1f}TB"


def render(ctx: dict, show_source: str | None = None) -> str:
    out = ["== BACKEND CONTEXT (TorchInductor) =="]
    out.append(f"compile={ctx['compile_s']}s  mode={ctx.get('compile_mode')}  config={ctx.get('inductor_config')}  "
               f"launches={ctx['launch_count']}  generated_kernels={ctx['generated_kernel_count']}  "
               f"SMs={ctx.get('sm_count')}")
    if ctx.get("output_code"):
        out.append(f"full generated code: {ctx['output_code']}")
    out.append("\n-- kernel launches in execution order --")
    total = 0
    for i, k in enumerate(ctx["kernels"]):
        total += k["bytes"]
        line = f"  [{i}] {k['type']:20s} {k['name']}  io={_fmt_bytes(k['bytes'])}"
        if "config" in k:
            line += (f"  cfg={k['config']} warps={k.get('num_warps')} stages={k.get('num_stages')} "
                     f"regs={k.get('n_regs')} spills={k.get('n_spills')} smem={k.get('shared')} "
                     f"programs={k.get('programs')} waves={k.get('waves')}")
        out.append(line)
        if k.get("source_nodes"):
            out.append(f"        fuses: [{k['source_nodes']}]")
        if k.get("aten"):
            out.append(f"        aten:  [{k['aten']}]")
    out.append(f"  total kernel I/O = {_fmt_bytes(total)}  (io = sizes of buffers touched; upper bound for sliced reads)")
    if ctx["warnings"]:
        out.append("\n-- warnings --")
        out.extend(f"  ! {w}" for w in ctx["warnings"])
    if show_source:
        for name, src in ctx.get("kernel_sources", {}).items():
            if show_source == "all" or show_source in name:
                out.append(f"\n-- source of {name} --\n{src}")
    return "\n".join(out)
