"""Run state: candidates, ledger of tool calls, budget, working graph and best graph G*.

Layout of ``runs/<name>/``::

    task.py            original graph G0 (inputs / init args / tolerance come from here)
    meta.json          budget, start time, GPU, baselines
    state.json         working-graph pointer, next candidate id
    candidates.json    candidate registry (parent, goal, status, measurements)
    candidates/cNNN_<slug>.py
    best.py, best.json best evaluated graph G* (independent of the working graph)
    ledger.jsonl       one line per tool call / plan event
    plan.md            goals, subgoals, evidence, revisions (maintained by the agent)
    logs/              worker stdout/stderr
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

ROOT = Path(os.environ.get("CAGE_ROOT", Path(__file__).resolve().parent.parent))
RUNS_DIR = ROOT / "runs"
CURRENT_FILE = RUNS_DIR / ".current"

PLAN_TEMPLATE = """# CAGE plan: {name}

Task: `{task}`  |  budget: {budget_min} min, {max_evals} hardware evals  |  GPU: {gpu}

## Baselines
(filled from `cage status`)

## Graph context summary
<!-- key facts from `cage ctx graph`: shapes, dominant ops, large intermediates, layout copies -->

## Goals
<!--
### G1: <structural goal>
- Hypothesis: <why this structure is faster: bottleneck, bytes, kernels>
- Success predicate: <checkable structure, e.g. "no materialized [B,S,I] fp32 tensor", "<= 2 kernels">
- Expected gain: <x.xx from cost model / reasoning>
- Subgoals (with dependencies):
  - G1.1 <step> (depends: -)       -> candidate cNNN, status
  - G1.2 <step> (depends: G1.1)    -> candidate cNNN, status
- Evidence log: <context queries and what they showed>
- Status: active | reached | revised | abandoned
-->

## Revisions
<!-- when evidence invalidates an assumption: what changed and why -->

## Final summary
"""


def _now() -> float:
    return time.time()


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s[:40] or "cand"


class Run:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not (self.path / "meta.json").exists():
            raise FileNotFoundError(f"not a CAGE run directory: {self.path}")

    # ------------------------------------------------------------------ creation / lookup
    @classmethod
    def create(cls, task_path: Path, name: str, budget_min: float, max_evals: int, gpu: str,
               overwrite: bool = False) -> "Run":
        path = RUNS_DIR / name
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"run {name!r} already exists (use --overwrite or a different --name)")
            shutil.rmtree(path)
        (path / "candidates").mkdir(parents=True)
        (path / "logs").mkdir()
        shutil.copy(task_path, path / "task.py")
        meta = {
            "name": name,
            "source_task": str(Path(task_path).resolve()),
            "created": _now(),
            "budget_min": budget_min,
            "max_evals": max_evals,
            "gpu": gpu,
            "baselines": {},
        }
        _write_json(path / "meta.json", meta)
        _write_json(path / "state.json", {"working": None, "next_id": 0})
        _write_json(path / "candidates.json", {})
        (path / "plan.md").write_text(PLAN_TEMPLATE.format(
            name=name, task=meta["source_task"], budget_min=budget_min, max_evals=max_evals, gpu=gpu))
        (path / "ledger.jsonl").touch()
        CURRENT_FILE.write_text(name)
        run = cls(path)
        run.new_candidate("g0", parent=None, source=path / "task.py", note="original graph G0")
        return run

    @classmethod
    def open(cls, name: str | None = None) -> "Run":
        if name is None:
            name = os.environ.get("CAGE_RUN")
        if name is None:
            if not CURRENT_FILE.exists():
                raise FileNotFoundError("no active run: start one with `cage init <task.py>`")
            name = CURRENT_FILE.read_text().strip()
        p = Path(name)
        if not p.is_absolute() and not (p / "meta.json").exists():
            p = RUNS_DIR / name
        return cls(p)

    # ------------------------------------------------------------------ properties
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def task_path(self) -> Path:
        return self.path / "task.py"

    @property
    def meta(self) -> dict:
        return _read_json(self.path / "meta.json", {})

    def update_meta(self, **kw) -> None:
        m = self.meta
        m.update(kw)
        _write_json(self.path / "meta.json", m)

    @property
    def state(self) -> dict:
        return _read_json(self.path / "state.json", {})

    def _update_state(self, **kw) -> None:
        s = self.state
        s.update(kw)
        _write_json(self.path / "state.json", s)

    @property
    def candidates(self) -> dict:
        return _read_json(self.path / "candidates.json", {})

    def _save_candidates(self, c: dict) -> None:
        _write_json(self.path / "candidates.json", c)

    @property
    def best(self) -> dict | None:
        return _read_json(self.path / "best.json", None)

    @property
    def gpu(self) -> str:
        return str(self.meta.get("gpu", "0"))

    # ------------------------------------------------------------------ candidates
    def new_candidate(self, slug: str, parent: str | None, source: Path | None = None,
                      goal: str | None = None, note: str | None = None) -> str:
        state = self.state
        cid = f"c{state['next_id']:03d}"
        fname = f"{cid}_{slugify(slug)}.py"
        dst = self.path / "candidates" / fname
        if source is None:
            source = self.candidate_path(parent) if parent else self.task_path
        shutil.copy(source, dst)
        cands = self.candidates
        cands[cid] = {
            "id": cid, "file": f"candidates/{fname}", "parent": parent, "goal": goal, "note": note,
            "created": _now(), "status": "new",
        }
        self._save_candidates(cands)
        self._update_state(next_id=state["next_id"] + 1, working=cid)
        return cid

    def register_file(self, path: Path) -> str:
        """Register a file written directly by the agent (outside `cage new`)."""
        path = Path(path).resolve()
        for cid, c in self.candidates.items():
            if (self.path / c["file"]).resolve() == path:
                return cid
        state = self.state
        cid = f"c{state['next_id']:03d}"
        dst = self.path / "candidates" / f"{cid}_{slugify(path.stem)}.py"
        if path.parent != (self.path / "candidates").resolve():
            shutil.copy(path, dst)
        else:
            path.rename(dst)
        cands = self.candidates
        cands[cid] = {"id": cid, "file": f"candidates/{dst.name}", "parent": state.get("working"),
                      "goal": None, "note": "registered", "created": _now(), "status": "new"}
        self._save_candidates(cands)
        self._update_state(next_id=state["next_id"] + 1)
        return cid

    def resolve(self, ref: str | None) -> tuple[str, Path]:
        """Resolve 'c003', '3', 'g0', 'best', 'working' or a path to (candidate id, file)."""
        cands = self.candidates
        if ref in (None, "", "working"):
            ref = self.state.get("working") or "c000"
        if ref == "g0":
            ref = "c000"
        if ref == "best":
            b = self.best
            if not b:
                raise KeyError("no best graph yet")
            ref = b["candidate"]
        if re.fullmatch(r"\d+", str(ref)):
            ref = f"c{int(ref):03d}"
        if ref in cands:
            return ref, self.path / cands[ref]["file"]
        p = Path(ref)
        if not p.exists():
            p2 = self.path / ref
            if p2.exists():
                p = p2
            else:
                m = re.match(r"(c\d{3})", p.name)
                if m and m.group(1) in cands:
                    return m.group(1), self.path / cands[m.group(1)]["file"]
                raise FileNotFoundError(f"unknown candidate {ref!r}")
        m = re.match(r"(c\d{3})_", p.name)
        if m and m.group(1) in cands and (self.path / cands[m.group(1)]["file"]).resolve() == p.resolve():
            return m.group(1), p
        cid = self.register_file(p)
        return cid, self.path / self.candidates[cid]["file"]

    def candidate_path(self, cid: str) -> Path:
        return self.resolve(cid)[1]

    def update_candidate(self, cid: str, **kw) -> None:
        cands = self.candidates
        cands.setdefault(cid, {"id": cid}).update(kw)
        self._save_candidates(cands)

    def set_working(self, cid: str) -> None:
        self._update_state(working=cid)

    # ------------------------------------------------------------------ ledger
    def elapsed_s(self) -> float:
        return _now() - self.meta["created"]

    def log(self, kind: str, **fields) -> dict:
        ev = {"ts": _now(), "elapsed_s": round(self.elapsed_s(), 2), "type": kind}
        ev.update({k: v for k, v in fields.items() if v is not None})
        with open(self.path / "ledger.jsonl", "a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        return ev

    def events(self) -> list[dict]:
        out = []
        with open(self.path / "ledger.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    # ------------------------------------------------------------------ budget
    def budget(self) -> dict:
        m = self.meta
        evals = sum(1 for e in self.events() if e["type"] == "eval" and e.get("measured"))
        el_min = self.elapsed_s() / 60.0
        time_left = m["budget_min"] - el_min
        evals_left = m["max_evals"] - evals
        return {
            "elapsed_min": round(el_min, 2), "budget_min": m["budget_min"], "time_left_min": round(time_left, 2),
            "evals": evals, "max_evals": m["max_evals"], "evals_left": evals_left,
            "exhausted": time_left <= 0 or evals_left <= 0,
        }

    # ------------------------------------------------------------------ best graph
    def maybe_update_best(self, cid: str, speedup: float, latency_ms: float, extra: dict | None = None,
                          min_gain: float = 0.005) -> bool:
        b = self.best
        if b is not None and speedup <= b["speedup_vs_eager"] * (1 + min_gain):
            return False
        shutil.copy(self.candidate_path(cid), self.path / "best.py")
        rec = {"candidate": cid, "speedup_vs_eager": speedup, "latency_ms": latency_ms,
               "elapsed_s": round(self.elapsed_s(), 2)}
        rec.update(extra or {})
        _write_json(self.path / "best.json", rec)
        return True
