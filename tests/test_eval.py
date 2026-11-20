import json
import shutil

from cage.cli import main
from cage.ledger import CURRENT_FILE, RUNS_DIR, Run

RUN = "pytest_tiny_dup"


def test_eval_updates_best_and_budget(fix, gpu, capsys):
    prev_current = CURRENT_FILE.read_text() if CURRENT_FILE.exists() else None
    try:
        assert main(["init", str(fix / "tiny_dup.py"), "--name", RUN, "--budget-min", "30", "--max-evals", "2",
                     "--gpu", gpu, "--overwrite", "--rounds", "3"]) == 0
        run = Run.open(RUN)
        assert run.best["candidate"] == "c000"

        assert main(["--run", RUN, "new", "cse", "--goal", "G1"]) == 0
        cid, path = run.resolve("working")
        shutil.copy(fix / "tiny_dup_cse.py", path)
        assert main(["--run", RUN, "check", cid]) == 0
        assert main(["--run", RUN, "cost", cid]) == 0
        capsys.readouterr()
        assert main(["--run", RUN, "eval", cid, "--goal", "G1", "--step", "1/1", "--json"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["speedup_vs_g0"] > 1.2, out          # one GEMM instead of two
        assert run.best["candidate"] == cid
        assert (run.path / "best.py").read_text() == path.read_text()

        # illegal candidates are rejected without charging the budget
        assert main(["--run", RUN, "new", "wrong", "--from", "c000"]) == 0
        bad, bad_path = run.resolve("working")
        bad_path.write_text(bad_path.read_text().replace("b * 0.5", "b * 0.25"))
        assert main(["--run", RUN, "eval", bad]) == 1
        assert run.budget()["evals"] == 1

        # the second measured eval exhausts max_evals=2, after which eval refuses
        assert main(["--run", RUN, "eval", "c000", "--no-g0"]) == 0
        assert main(["--run", RUN, "eval", cid]) == 3

        assert main(["--run", RUN, "report"]) == 0
        assert (run.path / "report.md").exists()
    finally:
        shutil.rmtree(RUNS_DIR / RUN, ignore_errors=True)
        if prev_current is not None:
            CURRENT_FILE.write_text(prev_current)
        elif CURRENT_FILE.exists():
            CURRENT_FILE.unlink()
