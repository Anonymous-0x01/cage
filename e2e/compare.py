#!/usr/bin/env python
"""End-to-end speedups of every model under e2e/results/: G0 (HF eager / torch.compile) vs the CAGE graph G*.

    source scripts/env.sh
    python e2e/compare.py                                   # forward + generation, every model, batch 1 and 8
    python e2e/compare.py --models qwen3-8b --gpu 2         # one model; GPU 2, otherwise least-utilized with enough free memory
    python e2e/compare.py --mode forward --cases decode_b1,qwen3-8b/prefill_b1 --rounds 10
    python e2e/compare.py --mode generate --models llama3-8b --batches 8 --rounds 5
    python e2e/compare.py --check                            # also re-verify G* == G0 (legality + numerics)
    python e2e/compare.py --report                           # no measurement: cross-model tables from saved results
    python e2e/compare.py --report --baseline compile       # graph-only speedup, one number per cell
    python e2e/compare.py --list

forward:  every case in e2e/results/<model>/<case>/ (g0.py, gstar.py) is measured like scripts/compare.py: eager G0,
          torch.compile G0 (default settings), torch.compile G0 with G*'s INDUCTOR_CONFIG / COMPILE_MODE
          (graph-only baseline) and compiled G*, interleaved in one process (L2 flush, CUDA events, median).
generate: e2e/generate.py runs a greedy 1024-token-prompt + 128-new-token request through the prefill and
          decode modules of each version and reports TTFT, time per output token and total latency.
--check:  cage legality check of G* against G0 (3 seeds, task tolerance, no graph breaks) plus the relative
          L2 error of compiled G* and of compiled G0 against eager G0, so G*'s deviation can be compared with
          the deviation Inductor itself introduces.
--report: reads e2e/results/<model>/summary.json (filtered by --models) and prints the cross-model speedup
          overview plus each model's detail tables, without touching the GPU.  The overview is one table per
          baseline (eager, compile default, compile with G*'s config), each cell a single speedup; --baseline
          keeps one of those tables.

The forward comparison holds four copies of a model and generation two, so the GPU is chosen by free memory,
estimated from the weight size in the model's g0 constants.  Each run merges its rows (per case / batch, with
timestamp, GPU and rounds) into e2e/results/<model>/summary.{json,md}, so re-measuring a subset keeps the other
results; e2e/summary.{md,json} is then rebuilt as the cross-model overview of every saved model.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cage.worker import run_worker, worker_env  # noqa: E402

E2E = ROOT / "e2e"
RESULTS = E2E / "results"
GEN_TARGETS = ("eager", "compile_default", "compile", "gstar")
BASELINES = ("eager", "compile_default", "compile")
GEN_METRICS = (("total_ms", "total (ms)"), ("tpot_ms", "TPOT (ms/token)"), ("ttft_ms", "TTFT (ms)"))


def discover() -> list[tuple[str, str]]:
    """(model, case) pairs under e2e/results/<model>/<case>/{g0,gstar}.py."""
    found = []
    if not RESULTS.is_dir():
        return found
    for model_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        for case_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if (case_dir / "g0.py").is_file() and (case_dir / "gstar.py").is_file():
                found.append((model_dir.name, case_dir.name))
    return found


def case_dir(model: str, case: str) -> Path:
    return RESULTS / model / case


def case_key(model: str, case: str) -> str:
    return f"{model}/{case}"


def describe(path: Path) -> str:
    text = path.read_text().lstrip()
    return text[3:].split("\n")[0].strip().removesuffix('"""').strip() if text.startswith('"""') else ""


def module_constants(path: Path) -> dict:
    """Module-level literal constants of a task file (VOCAB, HIDDEN, ...) without importing it."""
    consts = {}
    for node in ast.parse(path.read_text()).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        names = [getattr(e, "id", None) for e in target.elts] if isinstance(target, ast.Tuple) else [getattr(target, "id", None)]
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        consts.update(zip(names, value if isinstance(target, ast.Tuple) else [value]))
    return consts


def weights_gb(model: str) -> float | None:
    """bf16 weight size (GiB) of a decoder-only model from its g0 constants; None if they are not all present."""
    g0s = sorted((RESULTS / model).glob("*/g0.py"))
    if not g0s:
        return None
    c = module_constants(g0s[0])
    try:
        h, inter, layers, nh, nkv, d, vocab = (c[k] for k in ("HIDDEN", "INTERMEDIATE", "N_LAYERS", "N_HEADS", "N_KV",
                                                               "HEAD_DIM", "VOCAB"))
    except KeyError:
        return None
    per_layer = h * (nh + 2 * nkv) * d + nh * d * h + 3 * h * inter
    return (2 * vocab * h + layers * per_layer) * 2 / 2 ** 30


def need_gb(models: list[str], mode: str) -> float:
    """Free memory needed: four model copies for the forward comparison, two for generation, plus headroom."""
    copies = 4 if mode in ("forward", "all") else 2
    sizes = [weights_gb(m) for m in models]
    if not sizes or any(s is None for s in sizes):
        return 90.0
    return copies * max(sizes) + 24.0


def pick_gpu(need_gb: float) -> str:
    """Least-utilized GPU with at least `need_gb` free memory (the H20s are shared)."""
    if os.environ.get("CAGE_GPU"):
        return os.environ["CAGE_GPU"]
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    rows = []
    for line in out.strip().splitlines():
        idx, util, used, total = [x.strip() for x in line.split(",")]
        rows.append((idx, float(util), (float(total) - float(used)) / 1024))
    fit = [r for r in rows if r[2] >= need_gb]
    if not fit:
        raise SystemExit(f"no GPU with {need_gb:.0f} GB free: {[(i, round(f)) for i, _, f in rows]}; use --gpu")
    return min(fit, key=lambda r: r[1])[0]


def env_for(gpu: str) -> dict:
    env = worker_env(gpu)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "16")
    return env


# ---------------------------------------------------------------------- forward

def numerics_worker(key: str) -> dict:
    """Relative L2 / max error of compiled G0 and compiled G* against eager G0 (seed 0 inputs)."""
    import torch

    from cage.task import (build_candidate, build_reference, compile_model, flatten_outputs, inductor_config,
                           load_spec, make_inputs)

    model, case = key.split("/", 1)
    d = case_dir(model, case)
    g0, gs = load_spec(d / "g0.py"), load_spec(d / "gstar.py")
    ref = build_reference(g0)
    x = make_inputs(g0, 0)
    out = {}
    with torch.no_grad():
        want = [t.clone() for t in flatten_outputs(ref(*x))]
        for label, spec, model in (("compile_g0", g0, ref), ("gstar", gs, None)):
            model = model if model is not None else build_candidate(g0, spec, ref)
            with inductor_config(spec.inductor_config):
                fn = compile_model(model, spec)
                torch.compiler.cudagraph_mark_step_begin()
                got = flatten_outputs(fn(*x))
            errs = []
            for a, b in zip(want, got):
                diff = (a.double() - b.double())
                errs.append({"rel_l2": float(diff.norm() / a.double().norm()), "max_abs": float(diff.abs().max())})
            out[label] = {"rel_l2": max(e["rel_l2"] for e in errs), "max_abs": max(e["max_abs"] for e in errs)}
            del fn
            torch._dynamo.reset()
    return out


def run_numerics(case: str, gpu: str) -> dict:
    with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as td:
        path = Path(td) / "num.json"
        proc = subprocess.run([sys.executable, __file__, "--_numerics", case, "--_out", str(path)],
                              env=env_for(gpu), cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
        if not path.exists():
            return {"error": "\n".join(proc.stderr.strip().splitlines()[-5:])}
        return json.loads(path.read_text())


def compare_case(model: str, case: str, gpu: str, rounds: int, check: bool) -> dict:
    d = case_dir(model, case)
    g0, gs = str(d / "g0.py"), str(d / "gstar.py")
    out: dict = {"model": model, "case": case}
    if check:
        chk = run_worker("check", {"task_path": g0, "cand_path": gs}, gpu=gpu, timeout=3600)
        out["legal"] = bool(chk.get("ok"))
        out["max_abs_err"] = chk.get("max_abs_err")
        out["check_errors"] = chk.get("errors") or ([chk.get("error")] if chk.get("error") else [])
        out["numerics"] = run_numerics(case_key(model, case), gpu)
    targets = [
        {"label": "eager", "path": g0, "mode": "eager"},
        {"label": "compile_default", "path": g0, "mode": "compiled"},
        {"label": "compile", "path": g0, "mode": "compiled", "cfg_path": gs},
        {"label": "gstar", "path": gs, "mode": "compiled"},
    ]
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = env_for(gpu)["TORCHINDUCTOR_COMPILE_THREADS"]
    res = run_worker("measure", {"task_path": g0, "targets": targets, "rounds": rounds}, gpu=gpu, timeout=7200)
    res.pop("_stdout", None)
    if not res.get("ok"):
        out["error"] = f"{res.get('error')} {res.get('stderr_tail', '')}{res.get('traceback', '')}"
        return out
    T = res["targets"]
    out["device"] = res.get("device")
    for k, v in T.items():
        out[f"{k}_ms"] = v["median_ms"]
        out[f"{k}_iqr_pct"] = round(100 * (v["p75_ms"] - v["p25_ms"]) / v["median_ms"], 2)
    out["speedup_vs_eager"] = T["eager"]["median_ms"] / T["gstar"]["median_ms"]
    out["speedup_vs_compile"] = T["compile"]["median_ms"] / T["gstar"]["median_ms"]
    out["speedup_vs_compile_default"] = T["compile_default"]["median_ms"] / T["gstar"]["median_ms"]
    return out


# ---------------------------------------------------------------------- generate

def compare_generate(model: str, batch: int, gpu: str, rounds: int) -> dict:
    with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as td:
        path = Path(td) / "gen.json"
        cmd = [sys.executable, str(E2E / "generate.py"), "--model", model, "--batch", str(batch),
               "--rounds", str(rounds), "--json", str(path)]
        proc = subprocess.Popen(cmd, env=env_for(gpu), cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        log = []
        for line in proc.stdout:
            log.append(line)
            if line.startswith("[b"):
                print("   " + line.rstrip(), flush=True)
        proc.wait()
        if not path.exists():
            return {"model": model, "batch": batch, "error": "".join(log[-15:])}
        res = json.loads(path.read_text())
        res["model"] = model
    T = res["targets"]
    for metric in ("ttft_ms", "tpot_ms", "total_ms"):
        for base in ("eager", "compile_default", "compile"):
            if base in T and "gstar" in T:
                res[f"{metric}_speedup_vs_{base}"] = T[base][metric] / T["gstar"][metric]
    return res


# ---------------------------------------------------------------------- report

def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def _models(rows: list[dict]) -> list[str]:
    return sorted({r["model"] for r in rows if r.get("model")})


def _title(kind: str, rows: list[dict]) -> str:
    names = _models(rows)
    who = names[0] if len(names) == 1 else (", ".join(names) if names else "models")
    return f"{who} {kind}"


def _case_label(row: dict, multi: bool) -> str:
    if multi and row.get("model"):
        return f"{row['model']}/{row['case']}"
    return row.get("case", "")


def _provenance(rows: list[dict], label) -> str:
    seen = [f"{label(r)} {r['measured']} (GPU {r.get('gpu', '?')}, {r.get('rounds', '?')} rounds)"
            for r in rows if r.get("measured")]
    return "Measured: " + "; ".join(seen) + "." if seen else ""


def render(fwd: list[dict], gen: list[dict], note: str) -> str:
    lines = [f"# {_title('end-to-end: G0 vs G*', fwd or gen)}  ({note})", ""]
    fwd_multi = len(_models(fwd)) > 1
    if fwd:
        lines += ["## Forward passes (one prefill / one decode step)", "",
                  "| case | eager G0 (ms) | compile G0 default (ms) | compile G0 w/ G* cfg (ms) | G* (ms) | "
                  "G* vs eager | G* vs compile(G0, default) | G* vs compile(G0, G* cfg) = graph-only | legal |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in fwd:
            if "error" in r:
                lines.append(f"| {_case_label(r, fwd_multi)} | error: {r['error'][:80]} | | | | | | | |")
                continue
            legal = "" if "legal" not in r else ("yes" if r["legal"] else "NO")
            lines.append(f"| {_case_label(r, fwd_multi)} | {r['eager_ms']:.2f} | {r['compile_default_ms']:.2f} | {r['compile_ms']:.2f} | "
                         f"{r['gstar_ms']:.2f} | {r['speedup_vs_eager']:.2f}x | {r['speedup_vs_compile_default']:.2f}x | "
                         f"{r['speedup_vs_compile']:.2f}x | {legal} |")
        ok = [r for r in fwd if "error" not in r]
        if ok:
            lines.append(f"| **geomean** | | | | | **{geomean([r['speedup_vs_eager'] for r in ok]):.2f}x** | "
                         f"**{geomean([r['speedup_vs_compile_default'] for r in ok]):.2f}x** | "
                         f"**{geomean([r['speedup_vs_compile'] for r in ok]):.2f}x** | |")
        num = [r for r in ok if r.get("numerics") and "error" not in r["numerics"]]
        if num:
            lines += ["", "Numerics vs eager G0 (seed 0, worst output): relative L2 error / max abs error", "",
                      "| case | compile(G0) | G* |", "|---|---|---|"]
            for r in num:
                n = r["numerics"]
                lines.append(f"| {_case_label(r, fwd_multi)} | {n['compile_g0']['rel_l2']:.4f} / {n['compile_g0']['max_abs']:.3f} | "
                             f"{n['gstar']['rel_l2']:.4f} / {n['gstar']['max_abs']:.3f} |")
        lines += ["", "`compile(G0, G* cfg)` compiles the original graph with G*'s INDUCTOR_CONFIG / COMPILE_MODE, "
                  "so the last speedup column isolates the gain from graph rewriting."]
        prov = _provenance(fwd, lambda r: _case_label(r, fwd_multi))
        lines += ([prov] if prov else []) + [""]
    if gen:
        gen_multi = len(_models(gen)) > 1
        head = "| model | " if gen_multi else "| "
        sep = "|---|---|" if gen_multi else "|---|"
        lines += ["## Greedy generation (1024-token prompt + 128 new tokens)", "",
                  f"{head}batch | metric | eager G0 | compile G0 default | compile G0 w/ G* cfg | G* | G* vs eager | "
                  "G* vs compile(G0, default) | G* vs compile(G0, G* cfg) |",
                  f"{sep}---|---|---|---|---|---|---|---|"]
        for g in gen:
            prefix = f"{g.get('model', '')} | " if gen_multi else ""
            if "error" in g:
                lines.append(f"| {prefix}{g['batch']} | error: {g['error'][-80:]} | | | | | | | |")
                continue
            T = g["targets"]
            for metric, name in (("ttft_ms", "TTFT (ms)"), ("tpot_ms", "TPOT (ms/token)"), ("total_ms", "total (ms)")):
                cells = [f"{T[t][metric]:.2f}" if t in T else "" for t in GEN_TARGETS]
                sp = [f"{g[f'{metric}_speedup_vs_{b}']:.2f}x" if f"{metric}_speedup_vs_{b}" in g else ""
                      for b in BASELINES]
                lines.append(f"| {prefix}{g['batch']} | {name} | " + " | ".join(cells + sp) + " |")
            agree = ", ".join(f"{t} {T[t]['token_agreement'] * 100:.1f}%" for t in GEN_TARGETS
                              if t in T and "token_agreement" in T[t] and t != "eager")
            lines.append(f"| {prefix}{g['batch']} | greedy tokens = eager G0 (informational) | | | | | {agree} | | |")
        lines += ["", "TTFT = prefill + first token; TPOT = mean decode step; the KV cache is a preallocated buffer owned "
                  "by the driver. Any bf16 re-association "
                  "(including Inductor's own compilation of G0) changes the greedy path within the first tokens."]
        prov = _provenance(gen, lambda g: (f"{g.get('model')} " if gen_multi else "") + f"batch {g['batch']}")
        lines += ([prov] if prov else []) + [""]
    return "\n".join(lines)


def _speedup_text(value: float | None):
    from rich.text import Text

    if value is None:
        return Text("-", style="dim")
    t = Text(f"{value:.2f}x")
    t.stylize("bold green" if value >= 1.05 else ("yellow" if value >= 0.98 else "bold red"))
    return t


def print_tables(fwd: list[dict], gen: list[dict], note: str) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    sp = _speedup_text
    console = Console()
    fwd_multi = len(_models(fwd)) > 1
    if fwd:
        table = Table(title=f"{_title('forward: G0 vs G*', fwd)}    {note}",
                      box=box.ROUNDED, header_style="bold cyan", title_style="bold")
        for name, justify in (("case", "left"), ("G0 eager\n(ms)", "right"), ("G0 compile\ndefault\n(ms)", "right"),
                              ("G0 compile\nusing G* cfg\n(ms)", "right"), ("G*\n(ms)", "right"),
                              ("vs eager", "right"), ("vs G0 compile\n(default)", "right"),
                              ("vs G0 compile\n(same cfg)", "right"), ("legal", "center")):
            table.add_column(name, justify=justify, no_wrap=True)
        for r in fwd:
            if "error" in r:
                table.add_row(_case_label(r, fwd_multi), Text(r["error"][:60], style="red"), "", "", "", "", "", "", "")
                continue
            legal = "" if "legal" not in r else ("yes" if r["legal"] else "NO")
            table.add_row(_case_label(r, fwd_multi), f"{r['eager_ms']:.2f}", f"{r['compile_default_ms']:.2f}", f"{r['compile_ms']:.2f}",
                          f"{r['gstar_ms']:.2f}", sp(r["speedup_vs_eager"]), sp(r["speedup_vs_compile_default"]),
                          sp(r["speedup_vs_compile"]), Text(legal, style="green" if legal == "yes" else "bold red"))
        ok = [r for r in fwd if "error" not in r]
        if ok:
            table.add_section()
            table.add_row(Text("geomean", style="bold"), "", "", "", "", sp(geomean([r["speedup_vs_eager"] for r in ok])),
                          sp(geomean([r["speedup_vs_compile_default"] for r in ok])),
                          sp(geomean([r["speedup_vs_compile"] for r in ok])), "")
        console.print(table)
    if gen:
        gen_multi = len(_models(gen)) > 1
        table = Table(title=f"{_title('greedy generation (1024 + 128 tokens)', gen)}    {note}",
                      box=box.ROUNDED, header_style="bold cyan", title_style="bold")
        cols = [("batch", "right"), ("metric", "left"), ("G0 eager", "right"),
                ("G0 compile\ndefault", "right"), ("G0 compile\nusing G* cfg", "right"), ("G*", "right"),
                ("vs eager", "right"), ("vs G0 compile\n(default)", "right"),
                ("vs G0 compile\n(same cfg)", "right")]
        if gen_multi:
            cols.insert(0, ("model", "left"))
        for name, justify in cols:
            table.add_column(name, justify=justify, no_wrap=True)
        for g in gen:
            lead = [g.get("model", ""), str(g["batch"])] if gen_multi else [str(g["batch"])]
            if "error" in g:
                table.add_row(*lead, Text(g["error"][-60:], style="red"), "", "", "", "", "", "", "")
                continue
            T = g["targets"]
            for metric, name in (("ttft_ms", "TTFT (ms)"), ("tpot_ms", "TPOT (ms/tok)"), ("total_ms", "total (ms)")):
                table.add_row(*lead, name, *[f"{T[t][metric]:.2f}" if t in T else "" for t in GEN_TARGETS],
                              *[sp(g.get(f"{metric}_speedup_vs_{b}")) for b in BASELINES])
            table.add_section()
        console.print(table)


# ---------------------------------------------------------------------- saved results + cross-model overview

def summary_path(model: str) -> Path:
    return RESULTS / model / "summary.json"


def load_saved(model: str) -> dict:
    """{"forward": {case: row}, "generate": {batch: row}} as saved by earlier runs (empty if none)."""
    path = summary_path(model)
    data = json.loads(path.read_text()) if path.is_file() else {}
    return {"forward": data.get("forward", {}), "generate": data.get("generate", {})}


def saved_rows(data: dict) -> tuple[list[dict], list[dict]]:
    fwd = [data["forward"][k] for k in sorted(data["forward"])]
    gen = [data["generate"][k] for k in sorted(data["generate"], key=int)]
    return fwd, gen


def save_results(fwd: list[dict], gen: list[dict], gpu: str, fwd_rounds: int, gen_rounds: int) -> None:
    """Merge this run's rows into e2e/results/<model>/summary.{json,md}; an error never replaces a good saved row."""
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for model in _models(fwd + gen):
        data = load_saved(model)
        for section, rows, key, rounds in (("forward", fwd, "case", fwd_rounds), ("generate", gen, "batch", gen_rounds)):
            for r in rows:
                if r.get("model") != model:
                    continue
                k = str(r[key])
                old = data[section].get(k)
                if "error" in r and old and "error" not in old:
                    print(f"{model} {section} {k}: error, keeping the result saved {old.get('measured')}", flush=True)
                    continue
                data[section][k] = {**r, "measured": stamp, "gpu": gpu, "rounds": rounds}
        summary_path(model).write_text(json.dumps(data, indent=2, default=str))
        (RESULTS / model / "summary.md").write_text(render(*saved_rows(data), f"saved results, updated {stamp}") + "\n")


def overview(saved: dict[str, dict]):
    """Per-case and per-(batch, metric) speedup triples (vs eager, vs compile default, vs compile w/ G* cfg)."""
    fwd: dict[str, dict] = {}
    geo: dict[str, tuple] = {}
    gen: dict[tuple, dict] = {}
    for model, data in saved.items():
        rows = [r for r in data["forward"].values() if "error" not in r]
        for r in rows:
            fwd.setdefault(r["case"], {})[model] = tuple(r.get(f"speedup_vs_{b}") for b in BASELINES)
        if rows:
            geo[model] = tuple(geomean([r[f"speedup_vs_{b}"] for r in rows]) for b in BASELINES)
        for g in data["generate"].values():
            if "error" in g:
                continue
            for metric, _ in GEN_METRICS:
                gen.setdefault((int(g["batch"]), metric), {})[model] = tuple(
                    g.get(f"{metric}_speedup_vs_{b}") for b in BASELINES)
    return dict(sorted(fwd.items())), geo, dict(sorted(gen.items(), key=lambda kv: (kv[0][0], [m for m, _ in GEN_METRICS].index(kv[0][1]))))


BASELINE_TITLES = {
    "eager": "vs eager G0",
    "compile_default": "vs torch.compile(G0, default)",
    "compile": "vs torch.compile(G0, G* cfg) (graph-only)",
}


def selected_baselines(baseline: str) -> tuple[str, ...]:
    return BASELINES if baseline == "all" else (baseline,)


def _one(value) -> str:
    return f"{value:.2f}x" if value else "-"


def render_overview(saved: dict[str, dict], baseline: str = "all") -> str:
    models = sorted(saved)
    fwd, geo, gen = overview(saved)
    names = dict(GEN_METRICS)
    lines = [f"# End-to-end G* speedups: {', '.join(models)}  ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "One table per baseline; each cell is a single G* speedup. "
             "Per-model detail tables and measurement times follow; the source data is "
             "`e2e/results/<model>/summary.json`.", ""]
    for b in selected_baselines(baseline):
        idx = BASELINES.index(b)
        title = BASELINE_TITLES[b]
        if fwd:
            lines += [f"## Forward passes ({title})", "", "| case | " + " | ".join(models) + " |",
                      "|---|" + "---|" * len(models)]
            for case, by_model in fwd.items():
                lines.append("| " + case + " | " + " | ".join(
                    _one(by_model[m][idx]) if m in by_model else "" for m in models) + " |")
            lines.append("| **geomean** | " + " | ".join(
                f"**{_one(geo[m][idx])}**" if m in geo else "" for m in models) + " |")
            lines.append("")
        if gen:
            lines += [f"## Greedy generation, 1024-token prompt + 128 new tokens ({title})", "",
                      "| batch | metric | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
            for (batch, metric), by_model in gen.items():
                lines.append(f"| {batch} | {names[metric]} | " + " | ".join(
                    _one(by_model[m][idx]) if m in by_model else "" for m in models) + " |")
            lines.append("")
    for model in models:
        detail = render(*saved_rows(saved[model]), "saved results")
        lines += ["---", ""] + [("#" + ln) if ln.startswith("#") else ln for ln in detail.splitlines()] + [""]
    return "\n".join(lines)


def print_overview(saved: dict[str, dict], baseline: str = "all") -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    models = sorted(saved)
    fwd, geo, gen = overview(saved)
    names = dict(GEN_METRICS)
    console = Console()
    for b in selected_baselines(baseline):
        idx = BASELINES.index(b)
        title = BASELINE_TITLES[b]
        if fwd:
            table = Table(title=f"forward speedup {title}", box=box.ROUNDED,
                          header_style="bold cyan", title_style="bold")
            table.add_column("case", no_wrap=True)
            for m in models:
                table.add_column(m, justify="center", no_wrap=True)
            for case, by_model in fwd.items():
                table.add_row(case, *[_speedup_text(by_model[m][idx]) if m in by_model else "" for m in models])
            table.add_section()
            table.add_row(Text("geomean", style="bold"),
                          *[_speedup_text(geo[m][idx]) if m in geo else "" for m in models])
            console.print(table)
        if gen:
            table = Table(title=f"greedy generation speedup {title}", box=box.ROUNDED,
                          header_style="bold cyan", title_style="bold")
            table.add_column("batch", justify="right", no_wrap=True)
            table.add_column("metric", no_wrap=True)
            for m in models:
                table.add_column(m, justify="center", no_wrap=True)
            last = None
            for (batch, metric), by_model in gen.items():
                if last is not None and batch != last:
                    table.add_section()
                last = batch
                table.add_row(str(batch), names[metric],
                              *[_speedup_text(by_model[m][idx]) if m in by_model else "" for m in models])
            console.print(table)


def write_overview(saved: dict[str, dict], baseline: str = "all") -> None:
    (E2E / "summary.md").write_text(render_overview(saved, baseline) + "\n")
    (E2E / "summary.json").write_text(json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M"), "models": saved},
                                                 indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("forward", "generate", "all"), default="all")
    ap.add_argument("--models", default="all", help="model folders under e2e/results (default: all)")
    ap.add_argument("--cases", default="all",
                    help="forward cases: name or model/name, comma-separated (default: all)")
    ap.add_argument("--batches", default="1,8", help="generation batch sizes (default: 1,8)")
    ap.add_argument("--gpu", default=None, help="GPU index (default: least-utilized GPU with enough free memory)")
    ap.add_argument("--rounds", type=int, default=None, help="forward rounds (default 10) / generation rounds (default 3)")
    ap.add_argument("--check", action="store_true", help="re-run the legality check and the numerics report")
    ap.add_argument("--report", action="store_true",
                    help="no measurement: print / write the cross-model overview from saved results")
    ap.add_argument("--baseline", choices=("all", *BASELINES), default="all",
                    help="cross-model speedup tables: one baseline, or all three separate tables (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="do not write the per-model and overview summaries")
    ap.add_argument("--_numerics", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._numerics:
        Path(args._out).write_text(json.dumps(numerics_worker(args._numerics)))
        return 0
    found = discover()
    available_models = sorted({m for m, _ in found})
    if args.models == "all":
        scoped = found
    else:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        unknown_models = sorted(wanted - set(available_models))
        if unknown_models:
            print(f"unknown models: {unknown_models}; available: {available_models}")
            return 2
        scoped = [p for p in found if p[0] in wanted]
    if args.list:
        for model, case in scoped:
            print(f"{case_key(model, case):28s} G*: {describe(case_dir(model, case) / 'gstar.py')}")
        return 0
    if args.report:
        saved = {m: load_saved(m) for m in sorted({m for m, _ in scoped}) if summary_path(m).is_file()}
        if not saved:
            print("no saved results under e2e/results/<model>/summary.json; run a measurement first")
            return 1
        for model, data in saved.items():
            print_tables(*saved_rows(data), "saved results")
        print_overview(saved, args.baseline)
        if not args.no_save:
            write_overview(saved, args.baseline)
            print(f"overview written to {(E2E / 'summary.md').relative_to(ROOT)}")
        return 0
    if args.cases == "all":
        selected = scoped
        unknown = []
    else:
        selected, unknown = [], []
        for token in [c.strip() for c in args.cases.split(",") if c.strip()]:
            if "/" in token:
                hit = tuple(token.split("/", 1))
                if hit in scoped:
                    selected.append(hit)
                else:
                    unknown.append(token)
            else:
                hits = [p for p in scoped if p[1] == token]
                if hits:
                    selected.extend(hits)
                else:
                    unknown.append(token)
    if unknown:
        print(f"unknown cases: {unknown}; available: {[case_key(m, c) for m, c in scoped]}")
        return 2
    run_models = sorted({m for m, _ in (selected if args.mode == "forward" else scoped)})
    gpu = args.gpu or pick_gpu(need_gb(run_models, args.mode))
    fwd_rounds = args.rounds or 10
    gen_rounds = args.rounds or 3
    fwd, gen = [], []
    if args.mode in ("forward", "all"):
        for model, case in selected:
            label = case_key(model, case)
            t0 = time.time()
            r = compare_case(model, case, gpu, fwd_rounds, args.check)
            fwd.append(r)
            if "error" in r:
                print(f"{label:28s} ERROR {r['error'][:300]}", flush=True)
            else:
                extra = f"  legal={r['legal']}" if "legal" in r else ""
                print(f"{label:28s} eager {r['eager_ms']:9.3f} ms | compile(default) {r['compile_default_ms']:9.3f} ms | "
                      f"compile(G* cfg) {r['compile_ms']:9.3f} ms | G* {r['gstar_ms']:9.3f} ms | "
                      f"vs eager {r['speedup_vs_eager']:5.2f}x | vs compile(default) {r['speedup_vs_compile_default']:5.2f}x | "
                      f"vs compile(same cfg) {r['speedup_vs_compile']:5.2f}x{extra}  [{time.time() - t0:.0f}s]", flush=True)
    if args.mode in ("generate", "all"):
        gen_models = sorted({m for m, _ in scoped})
        for model in gen_models:
            for b in [int(x) for x in args.batches.split(",")]:
                if not (case_dir(model, f"prefill_b{b}") / "g0.py").is_file() or not (
                        case_dir(model, f"decode_b{b}") / "g0.py").is_file():
                    print(f"generation {model} batch {b}: no prefill_b{b}/decode_b{b}, skipped", flush=True)
                    continue
                t0 = time.time()
                print(f"generation {model} batch {b} ...", flush=True)
                g = compare_generate(model, b, gpu, gen_rounds)
                gen.append(g)
                if "error" in g:
                    print(f"{model} batch {b} ERROR {g['error'][-500:]}", flush=True)
                else:
                    print(f"{model} batch {b}: total G* vs eager {g['total_ms_speedup_vs_eager']:.2f}x, vs compile(default) "
                          f"{g['total_ms_speedup_vs_compile_default']:.2f}x, vs compile(same cfg) "
                          f"{g['total_ms_speedup_vs_compile']:.2f}x  [{time.time() - t0:.0f}s]", flush=True)
    print()
    stamp = f"{time.strftime('%Y-%m-%d %H:%M')}    GPU {gpu}"
    print_tables(fwd, [], f"{stamp}    {fwd_rounds} rounds")
    print_tables([], gen, f"{stamp}    {gen_rounds} rounds")
    if not args.no_save:
        save_results(fwd, gen, gpu, fwd_rounds, gen_rounds)
        saved = {m: load_saved(m) for m in available_models if summary_path(m).is_file()}
        if saved:
            print_overview(saved, args.baseline)
            write_overview(saved, args.baseline)
    bad = [r for r in fwd if "error" in r or not r.get("legal", True)] + [g for g in gen if "error" in g]
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
