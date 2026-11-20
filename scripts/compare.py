#!/usr/bin/env python
"""Compare G0 (PyTorch eager, torch.compile) against the optimized graph G* for every case in results/.

    source scripts/env.sh
    python scripts/compare.py                          # all cases on the least-loaded GPU
    python scripts/compare.py --cases vit_b16_embed,deepseek_v3_mla_decode --gpu 0 --rounds 10
    python scripts/compare.py --check                  # also re-verify G* == G0 numerically
    python scripts/compare.py --list

Each case directory holds g0.py (the original subgraph: Model, get_inputs, get_init_inputs) and
gstar.py (the optimized graph; may define convert_weights / INDUCTOR_CONFIG / COMPILE_MODE).
All targets of a case are measured in one process, in interleaved rounds, with an L2 flush before
every call (CUDA events, median over all samples), so ratios are robust to other jobs on the GPU.
Results are printed and written to results/summary.md and results/summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cage.measure import pick_gpu  # noqa: E402
from cage.worker import run_worker  # noqa: E402

RESULTS = ROOT / "results"


def cases() -> list[str]:
    return sorted(p.name for p in RESULTS.iterdir() if (p / "g0.py").exists() and (p / "gstar.py").exists())


def describe(path: Path) -> str:
    text = path.read_text()
    if text.lstrip().startswith('"""'):
        return text.lstrip()[3:].split("\n")[0].strip().removesuffix('"""').strip()
    return ""


def compare_case(case: str, gpu: str, rounds: int, check: bool, eager_gstar: bool) -> dict:
    d = RESULTS / case
    g0, gs = str(d / "g0.py"), str(d / "gstar.py")
    out: dict = {"case": case}
    if check:
        chk = run_worker("check", {"task_path": g0, "cand_path": gs}, gpu=gpu, timeout=1800)
        chk.pop("_stdout", None)
        out["legal"] = bool(chk.get("ok"))
        out["max_abs_err"] = chk.get("max_abs_err")
        out["check_errors"] = chk.get("errors") or ([chk.get("error")] if chk.get("error") else [])
    targets = [
        {"label": "eager", "path": g0, "mode": "eager"},
        # G0 graph, default torch.compile (what a user gets out of the box)
        {"label": "compile_default", "path": g0, "mode": "compiled"},
        # G0 graph compiled with G*'s INDUCTOR_CONFIG / COMPILE_MODE: the fair baseline
        {"label": "compile", "path": g0, "mode": "compiled", "cfg_path": gs},
        {"label": "gstar", "path": gs, "mode": "compiled"},
    ]
    if eager_gstar:
        targets.append({"label": "gstar_eager", "path": gs, "mode": "eager"})
    res = run_worker("measure", {"task_path": g0, "targets": targets, "rounds": rounds}, gpu=gpu, timeout=3600)
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
    out["compile_vs_eager"] = T["eager"]["median_ms"] / T["compile"]["median_ms"]
    out["compile_cfg_gain"] = T["compile_default"]["median_ms"] / T["compile"]["median_ms"]
    return out


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def render(rows: list[dict], gpu: str, rounds: int) -> str:
    lines = [
        f"# CAGE results: G0 vs G*  ({time.strftime('%Y-%m-%d %H:%M')}, GPU {gpu}, {rounds} rounds)",
        "",
        "| case | eager G0 (ms) | compile G0 default (ms) | compile G0 w/ G* cfg (ms) | G* (ms) | "
        "G* vs eager | G* vs compile(G0, default) | G* vs compile(G0, G* cfg) = graph-only | legal |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['case']} | error: {r['error'][:80]} | | | | | | | |")
            continue
        legal = "" if "legal" not in r else ("yes" if r["legal"] else "NO")
        lines.append(f"| {r['case']} | {r['eager_ms']:.4f} | {r['compile_default_ms']:.4f} | "
                     f"{r['compile_ms']:.4f} | {r['gstar_ms']:.4f} | {r['speedup_vs_eager']:.2f}x | "
                     f"{r['speedup_vs_compile_default']:.2f}x | {r['speedup_vs_compile']:.2f}x | {legal} |")
    ok = [r for r in rows if "error" not in r]
    if ok:
        lines.append(f"| **geomean** | | | | | **{geomean([r['speedup_vs_eager'] for r in ok]):.2f}x** | "
                     f"**{geomean([r['speedup_vs_compile_default'] for r in ok]):.2f}x** | "
                     f"**{geomean([r['speedup_vs_compile'] for r in ok]):.2f}x** | |")
    lines += ["", "`compile(G0, G* cfg)` compiles the original graph with G*'s INDUCTOR_CONFIG / COMPILE_MODE, "
              "so the last speedup column isolates the gain from graph rewriting."]
    return "\n".join(lines)


def print_table(rows: list[dict], gpu: str, rounds: int) -> None:
    """Pretty terminal table. The markdown written to summary.md stays plain."""
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    def speedup(value: float) -> Text:
        text = Text(f"{value:.2f}x")
        text.stylize("bold green" if value >= 1.05 else ("yellow" if value >= 0.98 else "bold red"))
        return text

    table = Table(
        title=f"CAGE results: G0 vs G*    {time.strftime('%Y-%m-%d %H:%M')}    GPU {gpu}    {rounds} rounds",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold",
        pad_edge=True,
    )
    for name, justify in (("case", "left"), ("G0 eager\n(ms)", "right"), ("G0 compile\ndefault\n(ms)", "right"),
                          ("G0 compile\nusing G* cfg\n(ms)", "right"), ("G*\n(ms)", "right"), ("vs eager", "right"),
                          ("vs G0 compile\n(default)", "right"), ("vs G0 compile\n(same cfg)", "right"),
                          ("legal", "center")):
        table.add_column(name, justify=justify, no_wrap=True)
    for r in rows:
        if "error" in r:
            table.add_row(r["case"], Text(r["error"][:60], style="red"), "", "", "", "", "", "", "")
            continue
        legal = "" if "legal" not in r else ("yes" if r["legal"] else "NO")
        table.add_row(r["case"], f"{r['eager_ms']:.4f}", f"{r['compile_default_ms']:.4f}",
                      f"{r['compile_ms']:.4f}", f"{r['gstar_ms']:.4f}",
                      speedup(r["speedup_vs_eager"]), speedup(r["speedup_vs_compile_default"]),
                      speedup(r["speedup_vs_compile"]),
                      Text(legal, style="green" if legal == "yes" else "bold red"))
    ok = [r for r in rows if "error" not in r]
    if ok:
        table.add_section()
        table.add_row(Text("geomean", style="bold"), "", "", "", "",
                      speedup(geomean([r["speedup_vs_eager"] for r in ok])),
                      speedup(geomean([r["speedup_vs_compile_default"] for r in ok])),
                      speedup(geomean([r["speedup_vs_compile"] for r in ok])), "")
    Console().print(table)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="all", help="comma-separated case names (default: all)")
    ap.add_argument("--gpu", default=None, help="GPU index (default: least loaded)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--check", action="store_true", help="re-run the legality check of G* against G0")
    ap.add_argument("--eager-gstar", action="store_true", help="also time G* without torch.compile")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="do not write results/summary.{md,json}")
    args = ap.parse_args()

    all_cases = cases()
    if args.list:
        for c in all_cases:
            print(f"{c:28s} G*: {describe(RESULTS / c / 'gstar.py')}")
        return 0
    selected = all_cases if args.cases == "all" else [c.strip() for c in args.cases.split(",")]
    unknown = [c for c in selected if c not in all_cases]
    if unknown:
        print(f"unknown cases: {unknown}; available: {all_cases}")
        return 2
    gpu = args.gpu or pick_gpu()
    rows = []
    for c in selected:
        t0 = time.time()
        r = compare_case(c, gpu, args.rounds, args.check, args.eager_gstar)
        rows.append(r)
        if "error" in r:
            print(f"{c:28s} ERROR {r['error'][:300]}", flush=True)
        else:
            extra = f"  legal={r['legal']}" if "legal" in r else ""
            print(f"{c:28s} eager {r['eager_ms']:8.4f} ms | compile(default) {r['compile_default_ms']:8.4f} ms | "
                  f"compile(G* cfg) {r['compile_ms']:8.4f} ms | G* {r['gstar_ms']:8.4f} ms | "
                  f"vs eager {r['speedup_vs_eager']:6.2f}x | vs compile(default) {r['speedup_vs_compile_default']:6.2f}x | "
                  f"vs compile(same cfg) {r['speedup_vs_compile']:6.2f}x{extra}  [{time.time() - t0:.0f}s]", flush=True)
    table = render(rows, gpu, args.rounds)
    print()
    print_table(rows, gpu, args.rounds)
    if not args.no_save:
        (RESULTS / "summary.md").write_text(table + "\n")
        (RESULTS / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    return 0 if all("error" not in r and r.get("legal", True) for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
