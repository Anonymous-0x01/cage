"""`cage` command line: the tool interface the optimization agent uses.

    cage init <task.py> [--name N] [--budget-min 45] [--max-evals 40] [--gpu G]
    cage new <slug> [--from cNNN] [--goal G1]             # copy a candidate to edit (new working graph)
    cage ctx graph|backend|exec [cand] --why "..."       # context acquisition (graph / backend / execution)
    cage check [cand] [--fast]                            # legality checker
    cage cost [cand] [--backend]                          # cheap cost model (tier 0 / tier 1)
    cage eval [cand] --goal G1 --step 2/3 --intent "..."  # legality -> cost -> measure on GPU, update G*
    cage goal G1 defined|progress|reached|revised|abandoned "text"   # record plan events
    cage work <cand>                                      # switch / backtrack the working graph
    cage status | cage report | cage calibrate

Candidates may be referred to as c003, 3, g0, best, working, or a file path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from .ledger import ROOT, Run


def _digest(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def _pin_gpu(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache" / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache" / "inductor"))


def _emit(args, text: str, data: dict) -> None:
    if getattr(args, "json", False):
        print(json.dumps({k: v for k, v in data.items() if not k.startswith("_")}, indent=2, default=str))
    else:
        print(text)


def _fmt_x(x) -> str:
    return "n/a" if x is None else f"{x:.3f}x"


def _quiet_warnings():
    import warnings

    warnings.filterwarnings("ignore")


def _cache_json(run: Run, key: str, compute):
    d = run.path / "cache"
    d.mkdir(exist_ok=True)
    p = d / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    val = compute()
    if val.get("ok"):
        p.write_text(json.dumps(val, default=str))
    return val


# ---------------------------------------------------------------------- shared steps

def _static(run: Run, cid: str, path: Path) -> dict:
    from .legality import static_check

    return static_check(path, is_task=False)


def _dynamic(run: Run, cid: str, path: Path, fast: bool = False) -> dict:
    from .worker import run_worker

    rec = run.candidates.get(cid, {})
    dig = _digest(path)
    if not fast and rec.get("legal_digest") == dig:
        return {"ok": True, "cached": True, "errors": [], "notes": rec.get("legal_notes", []),
                "graph_breaks": 0, "compiled": True, "seeds": rec.get("legal_seeds"),
                "max_abs_err": rec.get("max_abs_err", 0.0), "compile_s": None, "wall_s": 0.0}
    res = run_worker("check", {"task_path": str(run.task_path), "cand_path": str(path), "compiled": not fast},
                     gpu=run.gpu, timeout=1200, log_dir=run.path / "logs")
    res.pop("_stdout", None)
    if res.get("ok") and not fast:
        run.update_candidate(cid, legal_digest=dig, legal_notes=res.get("notes", []), legal_seeds=res.get("seeds"),
                             max_abs_err=res.get("max_abs_err"))
    return res


def _tier0(run: Run, cid: str, path: Path) -> tuple[dict, dict]:
    from .costmodel import tier0
    from .hw import load_hw

    hw = load_hw(run.meta.get("device"))
    ref = _cache_json(run, "g0_tier0", lambda: _strip(tier0(str(run.task_path), None, hw)))
    rec = run.candidates.get(cid, {})
    dig = _digest(path)
    if rec.get("tier0_digest") == dig and rec.get("tier0"):
        est = rec["tier0"]
    else:
        est = _strip(tier0(str(run.task_path), str(path), hw))
        run.update_candidate(cid, tier0_digest=dig, tier0=est)
    return est, ref


def _strip(est: dict) -> dict:
    est = dict(est)
    est.pop("_n_nodes", None)
    return est


def _est_speedup(est: dict, ref: dict) -> float | None:
    if est and ref and est.get("est_us"):
        return round(ref["est_us"] / est["est_us"], 4)
    return None


# ---------------------------------------------------------------------- commands

def cmd_init(args) -> int:
    from .legality import static_check
    from .measure import pick_gpu

    task = Path(args.task).resolve()
    st = static_check(task, is_task=True)
    if not st["ok"]:
        print("task file is not valid:\n  " + "\n  ".join(st["errors"]))
        return 2
    gpu = args.gpu or pick_gpu()
    name = args.name or task.stem
    run = Run.create(task, name, args.budget_min, args.max_evals, gpu, overwrite=args.overwrite)
    _pin_gpu(gpu)
    print(f"[cage] run '{name}' created at {run.path} (GPU {gpu})")
    from .hw import device_name
    from .worker import run_worker

    run.update_meta(device=device_name())
    cid, path = run.resolve("c000")
    print("[cage] checking G0 compiles (dynamo + inductor) ...", flush=True)
    chk = _dynamic(run, cid, path)
    if not chk.get("ok"):
        from .legality import render

        print(render({"ok": True, "errors": [], "warnings": []}, chk))
        print("[cage] WARNING: G0 does not pass the legality harness against itself; fix the task file.")
    print("[cage] measuring baselines (eager G0, Inductor G0) ...", flush=True)
    res = run_worker("measure", {"task_path": str(run.task_path), "rounds": args.rounds, "targets": [
        {"label": "eager", "path": str(run.task_path), "mode": "eager"},
        {"label": "inductor", "path": str(path), "mode": "compiled"},
    ]}, gpu=gpu, timeout=1800, log_dir=run.path / "logs")
    res.pop("_stdout", None)
    if not res.get("ok"):
        print(f"[cage] baseline measurement failed: {res.get('error')}\n{res.get('stderr_tail', '')}"
              f"{res.get('traceback', '')}")
        return 1
    eager = res["targets"]["eager"]["median_ms"]
    ind = res["targets"]["inductor"]["median_ms"]
    run.update_meta(baselines={"eager_ms": eager, "inductor_ms": ind, "inductor_speedup_vs_eager": eager / ind,
                               "detail": res["targets"], "device": res.get("device")})
    est, ref = _tier0(run, cid, path)
    run.update_candidate(cid, status="measured", latency_ms=ind, speedup_vs_eager=eager / ind, speedup_vs_g0=1.0,
                         est_us=est["est_us"])
    run.maybe_update_best(cid, eager / ind, ind, {"speedup_vs_g0": 1.0})
    run.log("init", candidate=cid, eager_ms=eager, inductor_ms=ind, speedup_vs_eager=round(eager / ind, 4),
            est_us=est["est_us"], cost_s=res.get("wall_s"))
    print(f"""
== RUN INITIALIZED: {name} ==
G0 eager     : {eager:.4f} ms
G0 inductor  : {ind:.4f} ms   (speedup vs eager {eager / ind:.3f}x)   <- candidates are compiled by Inductor
cost model G0: {est['est_us']:.1f} us est, {est['kernels']} simulated kernels
budget       : {args.budget_min} min, {args.max_evals} hardware evals
working graph: c000 ({path.relative_to(ROOT)})
plan file    : {(run.path / 'plan.md').relative_to(ROOT)}

next: cage ctx graph --why "initial graph context"   then define goals in plan.md""")
    return 0


def cmd_new(args) -> int:
    run = Run.open(args.run)
    parent = run.resolve(args.parent)[0] if args.parent else (run.state.get("working") or "c000")
    cid = run.new_candidate(args.slug, parent=parent, goal=args.goal, note=args.note)
    path = run.candidate_path(cid)
    run.log("new", candidate=cid, parent=parent, goal=args.goal, note=args.note)
    print(f"{cid} {path.relative_to(ROOT)}  (copied from {parent}; now the working graph -- edit this file)")
    return 0


def cmd_work(args) -> int:
    run = Run.open(args.run)
    cid, path = run.resolve(args.cand)
    run.set_working(cid)
    run.log("work", candidate=cid, why=args.why)
    print(f"working graph -> {cid} ({path.relative_to(ROOT)})")
    return 0


def cmd_ctx(args) -> int:
    run = Run.open(args.run)
    _pin_gpu(run.gpu)
    _quiet_warnings()
    cid, path = run.resolve(args.cand)
    out_dir = run.path / "ctx"
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    data: dict
    if args.level == "graph":
        from .context.graph import graph_context, render

        data = graph_context(str(run.task_path), str(path), max_nodes=args.max_nodes)
        text = render(data)
    elif args.level == "backend":
        from .context.backend import render
        from .worker import run_worker

        data = run_worker("ctx_backend", {"task_path": str(run.task_path), "cand_path": str(path),
                                          "out_dir": str(out_dir)}, gpu=run.gpu, timeout=1200,
                          log_dir=run.path / "logs")
        data.pop("_stdout", None)
        text = render(data, show_source=args.show_source) if data.get("ok") else _err(data)
    else:
        from .context.execution import render, run_ncu
        from .hw import load_hw
        from .worker import run_worker

        hw = load_hw(run.meta.get("device"))
        data = run_worker("ctx_exec", {"task_path": str(run.task_path), "cand_path": str(path), "top": args.top,
                                       "iters": args.iters, "peak_gbps": hw["hbm_gbps"]},
                          gpu=run.gpu, timeout=1200, log_dir=run.path / "logs")
        data.pop("_stdout", None)
        ncu = None
        if data.get("ok") and args.ncu:
            ncu = run_ncu(str(run.task_path), str(path), data["top_kernels"][: args.ncu_top], run.gpu,
                          log_dir=run.path / "logs")
            data["ncu"] = ncu
        text = render(data, ncu) if data.get("ok") else _err(data)
    cost_s = round(time.time() - t0, 2)
    (out_dir / f"{cid}_{args.level}.txt").write_text(text)
    run.log("ctx", level=args.level, candidate=cid, why=args.why, goal=args.goal, ok=bool(data.get("ok")),
            cost_s=cost_s, ncu=bool(args.ncu) if args.level == "exec" else None)
    _emit(args, text + f"\n[context query cost {cost_s}s; saved to {(out_dir / f'{cid}_{args.level}.txt').relative_to(ROOT)}]",
          data)
    return 0 if data.get("ok") else 1


def _err(d: dict) -> str:
    return f"ERROR: {d.get('error')}\n{d.get('stderr_tail', '')}{d.get('traceback', '')}"


def cmd_check(args) -> int:
    from .legality import render

    run = Run.open(args.run)
    _pin_gpu(run.gpu)
    cid, path = run.resolve(args.cand)
    t0 = time.time()
    st = _static(run, cid, path)
    dyn = _dynamic(run, cid, path, fast=args.fast) if st["ok"] else None
    ok = st["ok"] and bool(dyn and dyn.get("ok"))
    cost_s = round(time.time() - t0, 2)
    if not ok:
        run.update_candidate(cid, status="illegal")
    elif run.candidates.get(cid, {}).get("status") in (None, "new", "illegal"):
        run.update_candidate(cid, status="legal")
    run.log("check", candidate=cid, ok=ok, fast=args.fast, goal=args.goal, step=args.step, cost_s=cost_s,
            errors=(st["errors"] + (dyn.get("errors", []) if dyn else []))[:5])
    _emit(args, render(st, dyn) + f"\n[{cid}; {cost_s}s]", {"static": st, "dynamic": dyn, "ok": ok})
    return 0 if ok else 1


def cmd_cost(args) -> int:
    from .costmodel import render

    run = Run.open(args.run)
    _pin_gpu(run.gpu)
    _quiet_warnings()
    cid, path = run.resolve(args.cand)
    t0 = time.time()
    if args.backend:
        from .hw import load_hw
        from .worker import run_worker

        hw = load_hw(run.meta.get("device"))

        def tier1(p):
            r = run_worker("cost_backend", {"task_path": str(run.task_path), "cand_path": str(p), "hw": hw},
                           gpu=run.gpu, timeout=1200, log_dir=run.path / "logs")
            r.pop("_stdout", None)
            return r

        ref = _cache_json(run, "g0_tier1", lambda: tier1(run.task_path))
        est = tier1(path)
        if not est.get("ok"):
            print(_err(est))
            return 1
    else:
        try:
            est, ref = _tier0(run, cid, path)
        except Exception as e:
            print(f"ERROR: cost model could not trace candidate: {type(e).__name__}: {e}")
            return 1
    sp = _est_speedup(est, ref)
    cost_s = round(time.time() - t0, 2)
    upd = {"est_speedup_vs_g0": sp, "est_us": est["est_us"]} if not args.backend else \
        {"est1_speedup_vs_g0": sp, "est1_us": est["est_us"]}
    run.update_candidate(cid, **upd)
    run.log("cost", candidate=cid, tier=est["tier"], est_us=est["est_us"], est_speedup_vs_g0=sp, goal=args.goal,
            step=args.step, intent=args.intent, cost_s=cost_s)
    _emit(args, render(est, ref, top=args.top) + f"\n[{cid}; {cost_s}s]", {"estimate": est, "g0": ref,
                                                                           "est_speedup_vs_g0": sp})
    return 0


def cmd_eval(args) -> int:
    from .legality import render as render_legal
    from .worker import run_worker

    run = Run.open(args.run)
    _pin_gpu(run.gpu)
    _quiet_warnings()
    cid, path = run.resolve(args.cand)
    b = run.budget()
    if b["exhausted"]:
        print(f"BUDGET EXHAUSTED ({b['elapsed_min']}/{b['budget_min']} min, {b['evals']}/{b['max_evals']} evals). "
              "No more hardware evaluations; run `cage report` and summarize.")
        return 3
    t0 = time.time()
    st = _static(run, cid, path)
    dyn = _dynamic(run, cid, path) if st["ok"] else None
    if not (st["ok"] and dyn and dyn.get("ok")):
        run.update_candidate(cid, status="illegal")
        run.log("eval", candidate=cid, measured=False, ok=False, goal=args.goal, step=args.step, intent=args.intent,
                cost_s=round(time.time() - t0, 2), errors=(st["errors"] + (dyn or {}).get("errors", []))[:5])
        print(render_legal(st, dyn))
        print("eval aborted: candidate is not legal (not measured, budget not charged).")
        return 1
    try:
        est, ref = _tier0(run, cid, path)
        est_sp = _est_speedup(est, ref)
    except Exception:
        est, est_sp = None, None
    targets = [{"label": "eager", "path": str(run.task_path), "mode": "eager"},
               {"label": "cand", "path": str(path), "mode": "compiled"}]
    if not args.no_g0:
        targets.insert(1, {"label": "inductor", "path": str(run.candidate_path("c000")), "mode": "compiled"})
    res = run_worker("measure", {"task_path": str(run.task_path), "targets": targets, "rounds": args.rounds},
                     gpu=run.gpu, timeout=1800, log_dir=run.path / "logs")
    res.pop("_stdout", None)
    if not res.get("ok"):
        run.log("eval", candidate=cid, measured=False, ok=False, goal=args.goal, step=args.step,
                error=res.get("error"), cost_s=round(time.time() - t0, 2))
        print(f"measurement failed: {_err(res)}")
        return 1
    T = res["targets"]
    eager, cand = T["eager"]["median_ms"], T["cand"]["median_ms"]
    sp_eager = eager / cand
    base = run.meta["baselines"]
    if "inductor" in T:
        sp_g0 = T["inductor"]["median_ms"] / cand
    else:
        sp_g0 = sp_eager / base["inductor_speedup_vs_eager"]
    prev_best = run.best
    new_best = run.maybe_update_best(cid, sp_eager, cand, {"speedup_vs_g0": sp_g0, "goal": args.goal})
    best = run.best
    run.update_candidate(cid, status="measured", latency_ms=cand, speedup_vs_eager=round(sp_eager, 4),
                         speedup_vs_g0=round(sp_g0, 4), goal=args.goal or run.candidates[cid].get("goal"),
                         iqr_ms=[T["cand"]["p25_ms"], T["cand"]["p75_ms"]], est_speedup_vs_g0=est_sp)
    run.set_working(cid)
    cost_s = round(time.time() - t0, 2)
    run.log("eval", candidate=cid, measured=True, ok=True, goal=args.goal, step=args.step, intent=args.intent,
            latency_ms=cand, eager_ms=eager, g0_inductor_ms=T.get("inductor", {}).get("median_ms"),
            speedup_vs_eager=round(sp_eager, 4), speedup_vs_g0=round(sp_g0, 4), est_speedup_vs_g0=est_sp,
            new_best=new_best, best=best["candidate"], best_speedup_vs_eager=best["speedup_vs_eager"], cost_s=cost_s)
    b = run.budget()
    pred = ""
    if est_sp:
        err = (est_sp - sp_g0) / sp_g0
        pred = f"cost model predicted {est_sp:.3f}x vs G0 -> measured {sp_g0:.3f}x (error {err * 100:+.0f}%)"
        if abs(err) > 0.15:
            pred += "  ! cost model and hardware disagree: consider `cage ctx backend` / `cage ctx exec`"
    noise = (T["cand"]["p75_ms"] - T["cand"]["p25_ms"]) / cand
    text = f"""== EVAL {cid} ({path.name}) ==
latency     : {cand:.4f} ms  (IQR {T['cand']['p25_ms']:.4f}-{T['cand']['p75_ms']:.4f}, noise {noise * 100:.1f}%)
eager G0    : {eager:.4f} ms   -> speedup vs eager {sp_eager:.3f}x
inductor G0 : {T.get('inductor', {}).get('median_ms', float('nan')):.4f} ms   -> speedup vs G0(inductor) {sp_g0:.3f}x
{pred}
best G*     : {best['candidate']} ({best['speedup_vs_eager']:.3f}x vs eager){'   ** NEW BEST **' if new_best else ''}{'' if new_best or not prev_best else ''}
legality    : max_abs_err={dyn.get('max_abs_err', 0):.3g}{' (cached)' if dyn.get('cached') else ''}{(' notes: ' + '; '.join(dyn['notes'])) if dyn.get('notes') else ''}
budget      : {b['elapsed_min']:.1f}/{b['budget_min']} min, {b['evals']}/{b['max_evals']} evals
working     : {cid}   [measurement {res.get('wall_s')}s, total {cost_s}s]"""
    _emit(args, text, {"candidate": cid, "latency_ms": cand, "speedup_vs_eager": sp_eager, "speedup_vs_g0": sp_g0,
                       "est_speedup_vs_g0": est_sp, "new_best": new_best, "best": best, "budget": b,
                       "measurement": T})
    return 0


def cmd_goal(args) -> int:
    run = Run.open(args.run)
    run.log("goal", goal=args.goal_id, event=args.event, text=args.text, candidate=args.cand)
    print(f"logged goal event {args.goal_id}:{args.event}")
    return 0


def cmd_status(args) -> int:
    run = Run.open(args.run)
    m, s, b = run.meta, run.state, run.budget()
    best = run.best
    base = m.get("baselines", {})
    lines = [f"== STATUS {run.name} ==  task={m['source_task']}  GPU={m['gpu']}"]
    if base:
        lines.append(f"G0 eager {base['eager_ms']:.4f} ms | G0 inductor {base['inductor_ms']:.4f} ms "
                     f"({base['inductor_speedup_vs_eager']:.3f}x)")
    lines.append(f"budget: {b['elapsed_min']:.1f}/{b['budget_min']} min, {b['evals']}/{b['max_evals']} evals"
                 f"{'  EXHAUSTED' if b['exhausted'] else ''}")
    lines.append(f"working: {s.get('working')}   best G*: "
                 f"{(best['candidate'] + ' %.3fx vs eager, %.3fx vs G0' % (best['speedup_vs_eager'], best.get('speedup_vs_g0', float('nan')))) if best else '-'}")
    lines.append("\n-- candidates --")
    lines.append("  id    parent goal   status    est(vsG0) meas(vsG0) meas(vsEager)  file")
    for cid, c in sorted(run.candidates.items()):
        lines.append(f"  {cid:5s} {str(c.get('parent') or '-'):6s} {str(c.get('goal') or '-'):6s} "
                     f"{c.get('status', '?'):9s} {_fmt_x(c.get('est_speedup_vs_g0')):9s} "
                     f"{_fmt_x(c.get('speedup_vs_g0')):10s} {_fmt_x(c.get('speedup_vs_eager')):13s} {c['file']}")
    ev = run.events()
    counts: dict[str, int] = {}
    for e in ev:
        k = e["type"] + (":" + e["level"] if e["type"] == "ctx" else "")
        counts[k] = counts.get(k, 0) + 1
    lines.append(f"\ntool calls: {counts}")
    if args.events:
        lines.append("\n-- last events --")
        for e in ev[-args.events:]:
            lines.append("  " + json.dumps({k: v for k, v in e.items() if k != "ts"}, default=str)[:220])
    _emit(args, "\n".join(lines), {"meta": m, "state": s, "budget": b, "best": best, "candidates": run.candidates})
    return 0


def cmd_report(args) -> int:
    from .report import make_report

    run = Run.open(args.run)
    path = make_report(run)
    print((run.path / "report.md").read_text())
    print(f"\n[report written to {path.relative_to(ROOT)}]")
    return 0


def cmd_calibrate(args) -> int:
    from .hw import save_calibration
    from .measure import pick_gpu
    from .worker import run_worker

    gpu = args.gpu or pick_gpu()
    res = run_worker("calibrate", {}, gpu=gpu, timeout=600)
    res.pop("_stdout", None)
    if not res.get("ok"):
        print(_err(res))
        return 1
    save_calibration(res["device"], res["values"])
    print(json.dumps(res, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cage", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="run name (default: the most recently initialized run)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, cand=True):
        if cand:
            sp.add_argument("cand", nargs="?", default=None, help="candidate (default: working graph)")
        sp.add_argument("--json", action="store_true")
        sp.add_argument("--goal", default=None)
        sp.add_argument("--step", default=None)

    sp = sub.add_parser("init", help="start a run for a task file")
    sp.add_argument("task")
    sp.add_argument("--name")
    sp.add_argument("--budget-min", type=float, default=45.0)
    sp.add_argument("--max-evals", type=int, default=40)
    sp.add_argument("--gpu")
    sp.add_argument("--rounds", type=int, default=5)
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(fn=cmd_init)

    sp = sub.add_parser("new", help="create a new candidate by copying a parent")
    sp.add_argument("slug")
    sp.add_argument("--from", dest="parent", default=None)
    sp.add_argument("--goal", default=None)
    sp.add_argument("--note", default=None)
    sp.set_defaults(fn=cmd_new)

    sp = sub.add_parser("work", help="set the working graph (e.g. backtrack)")
    sp.add_argument("cand")
    sp.add_argument("--why", default=None)
    sp.set_defaults(fn=cmd_work)

    sp = sub.add_parser("ctx", help="acquire graph / backend / execution context")
    sp.add_argument("level", choices=["graph", "backend", "exec"])
    common(sp)
    sp.add_argument("--why", required=True, help="which decision this context should resolve")
    sp.add_argument("--max-nodes", type=int, default=250)
    sp.add_argument("--show-source", default=None, help="backend: print Triton source of kernels matching this "
                                                        "substring ('all' for every kernel)")
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--iters", type=int, default=20)
    sp.add_argument("--ncu", action="store_true", help="exec: also profile top kernels with Nsight Compute")
    sp.add_argument("--ncu-top", type=int, default=3)
    sp.set_defaults(fn=cmd_ctx)

    sp = sub.add_parser("check", help="legality checker")
    common(sp)
    sp.add_argument("--fast", action="store_true", help="eager-only numerical check (no compile)")
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("cost", help="cheap cost model")
    common(sp)
    sp.add_argument("--backend", action="store_true", help="tier 1: use Inductor's real kernels (compiles)")
    sp.add_argument("--intent", default=None)
    sp.add_argument("--top", type=int, default=8)
    sp.set_defaults(fn=cmd_cost)

    sp = sub.add_parser("eval", help="legality + cost + on-device measurement; updates G*")
    common(sp)
    sp.add_argument("--intent", default=None)
    sp.add_argument("--rounds", type=int, default=5)
    sp.add_argument("--no-g0", action="store_true", help="skip re-measuring Inductor G0 in this session")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("goal", help="record a plan event")
    sp.add_argument("goal_id")
    sp.add_argument("event", choices=["defined", "progress", "reached", "revised", "abandoned", "note"])
    sp.add_argument("text")
    sp.add_argument("--cand", default=None)
    sp.set_defaults(fn=cmd_goal)

    sp = sub.add_parser("status", help="run status")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--events", type=int, default=8)
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("report", help="final report + trajectory plot")
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("calibrate", help="calibrate the cost model on a GPU")
    sp.add_argument("--gpu")
    sp.set_defaults(fn=cmd_calibrate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, FileExistsError, KeyError) as e:
        print(f"cage: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
