from cage.legality import static_check
from cage.worker import run_worker


def test_static_accepts_task_and_rewrite(fix):
    assert static_check(fix / "tiny_mlp.py", is_task=True)["ok"]
    assert static_check(fix / "tiny_mlp_merged.py")["ok"]


def test_static_rejects_triton_import(fix):
    res = static_check(fix / "tiny_mlp_triton.py")
    assert not res["ok"]
    assert any("triton" in e for e in res["errors"])


def test_static_rejects_stateful_forward(fix):
    res = static_check(fix / "tiny_mlp_stateful.py")
    assert not res["ok"]
    assert any("self.last" in e for e in res["errors"])


def test_static_rejects_unknown_inductor_key(fix, tmp_path_factory):
    src = (fix / "tiny_mlp_merged.py").read_text() + "\nINDUCTOR_CONFIG = {'not_a_real_knob': 1}\n"
    p = fix.parent.parent / ".cache" / "pytest_bad_cfg.py"
    p.parent.mkdir(exist_ok=True)
    p.write_text(src)
    res = static_check(p)
    assert not res["ok"]
    assert any("not whitelisted" in e for e in res["errors"])


def test_dynamic_accepts_correct_rewrite(fix, gpu):
    res = run_worker("check", {"task_path": str(fix / "tiny_mlp.py"), "cand_path": str(fix / "tiny_mlp_merged.py")},
                     gpu=gpu, timeout=600)
    assert res["ok"], res
    assert res["graph_breaks"] == 0


def test_dynamic_rejects_wrong_semantics(fix, gpu):
    res = run_worker("check", {"task_path": str(fix / "tiny_mlp.py"), "cand_path": str(fix / "tiny_mlp_wrong.py")},
                     gpu=gpu, timeout=600)
    assert not res["ok"]
    assert any("mismatch" in e for e in res["errors"])
