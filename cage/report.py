"""Final report: best graph, goal history, tool-call overhead, and the optimization trajectory plot."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .ledger import Run


def _plot(run: Run, events: list[dict], out: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    base = run.meta.get("baselines", {})
    ind = base.get("inductor_speedup_vs_eager", 1.0)
    meas_t, meas_y, best_t, best_y = [], [], [], []
    est_t, est_y = [], []
    best = 0.0
    for e in events:
        t = e["elapsed_s"] / 60
        if e["type"] in ("init",) and e.get("speedup_vs_eager"):
            best = 1.0
            meas_t.append(t)
            meas_y.append(best)
            best_t.append(t)
            best_y.append(best)
        elif e["type"] == "eval" and e.get("measured"):
            meas_t.append(t)
            meas_y.append(e["speedup_vs_g0"])
            best = max(best, e["speedup_vs_g0"])
            best_t.append(t)
            best_y.append(best)
        elif e["type"] == "cost" and e.get("est_speedup_vs_g0") and e.get("tier", 0) == 0:
            est_t.append(t)
            est_y.append(e["est_speedup_vs_g0"])
    if not meas_t:
        return False
    end_t = events[-1]["elapsed_s"] / 60
    best_t.append(end_t)
    best_y.append(best)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(meas_t, meas_y, "o-", label="working graph (measured)", color="C0")
    ax.step(best_t, best_y, where="post", label="best G* so far", color="C2", lw=2)
    if est_t:
        ax.scatter(est_t, est_y, facecolors="none", edgecolors="C1", label="cost-model estimate (unmeasured step)")
    ax.axhline(1.0, color="C3", ls="--", lw=1, label="Inductor G0")
    if 1.0 / ind > min(meas_y + est_y) * 0.8:
        ax.axhline(1.0 / ind, color="gray", ls=":", lw=1, label="eager G0")
    for e in events:
        if e["type"] == "goal" and e.get("event") in ("defined", "reached", "revised", "abandoned"):
            t = e["elapsed_s"] / 60
            ax.axvline(t, color="gray", alpha=0.3, lw=0.8)
            ax.text(t, ax.get_ylim()[1], f"{e['goal']}:{e['event'][:3]}", rotation=90, va="top", fontsize=7,
                    alpha=0.7)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("speedup vs Inductor-compiled G0")
    ax.set_title(f"CAGE trajectory: {run.name}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def make_report(run: Run) -> Path:
    events = run.events()
    meta = run.meta
    base = meta.get("baselines", {})
    best = run.best or {}
    cands = run.candidates
    b = run.budget()

    calls = Counter()
    cost_by = defaultdict(float)
    for e in events:
        k = e["type"] + (":" + e["level"] if e["type"] == "ctx" else "")
        calls[k] += 1
        cost_by[k] += float(e.get("cost_s") or 0)
    evals = [e for e in events if e["type"] == "eval" and e.get("measured")]
    regress = 0
    prev = base.get("inductor_speedup_vs_eager")
    for e in evals:
        if prev and e["speedup_vs_eager"] < prev * 0.98:
            regress += 1
        prev = e["speedup_vs_eager"]

    plot_ok = _plot(run, events, run.path / "trajectory.png")
    lines = [f"# CAGE report: {run.name}", ""]
    lines.append(f"- task: `{meta['source_task']}`  device: {base.get('device', meta.get('device'))} (GPU {meta['gpu']})")
    if base:
        lines.append(f"- G0 eager: {base['eager_ms']:.4f} ms; G0 Inductor: {base['inductor_ms']:.4f} ms "
                     f"({base['inductor_speedup_vs_eager']:.3f}x vs eager)")
    if best:
        bc = cands.get(best["candidate"], {})
        lines.append(f"- **best graph G\\*: {best['candidate']}** (`{bc.get('file')}`, copied to `best.py`): "
                     f"{best['latency_ms']:.4f} ms, **{best['speedup_vs_eager']:.3f}x vs eager**, "
                     f"{best.get('speedup_vs_g0', float('nan')):.3f}x vs Inductor G0; found at "
                     f"{best['elapsed_s'] / 60:.1f} min")
    lines.append(f"- budget used: {b['elapsed_min']:.1f}/{b['budget_min']} min, {b['evals']}/{b['max_evals']} "
                 f"hardware evals; measured regressions along the path: {regress}")
    lines.append("")
    lines.append("## Tool calls (count, total seconds)")
    for k in sorted(calls):
        lines.append(f"- {k}: {calls[k]} calls, {cost_by[k]:.1f}s")
    lines.append("")
    lines.append("## Goal history")
    goals = [e for e in events if e["type"] == "goal"]
    if not goals:
        lines.append("(no goal events recorded)")
    for e in goals:
        lines.append(f"- [{e['elapsed_s'] / 60:5.1f} min] **{e['goal']} {e['event']}**: {e.get('text', '')}"
                     + (f" ({e['candidate']})" if e.get("candidate") else ""))
    lines.append("")
    lines.append("## Candidates")
    lines.append("| id | parent | goal | status | est vs G0 | measured vs G0 | measured vs eager |")
    lines.append("|---|---|---|---|---|---|---|")
    for cid, c in sorted(cands.items()):
        def f(x):
            return "" if x is None else f"{x:.3f}x"
        lines.append(f"| {cid} | {c.get('parent') or ''} | {c.get('goal') or ''} | {c.get('status')} | "
                     f"{f(c.get('est_speedup_vs_g0'))} | {f(c.get('speedup_vs_g0'))} | {f(c.get('speedup_vs_eager'))} |")
    lines.append("")
    lines.append("## Measured trajectory")
    for e in evals:
        lines.append(f"- [{e['elapsed_s'] / 60:5.1f} min] {e['candidate']} goal={e.get('goal')} step={e.get('step')}: "
                     f"{e['speedup_vs_eager']:.3f}x vs eager ({e['speedup_vs_g0']:.3f}x vs G0)"
                     + (" NEW BEST" if e.get("new_best") else "") + (f" -- {e['intent']}" if e.get("intent") else ""))
    if plot_ok:
        lines.append("")
        lines.append("![trajectory](trajectory.png)")
    path = run.path / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
