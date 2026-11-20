from cage.context.graph import analyze, trace
from cage.costmodel import tier0


def test_merged_gemm_ranks_below_separate(fix, gpu):
    g0 = tier0(str(fix / "tiny_mlp.py"))
    merged = tier0(str(fix / "tiny_mlp.py"), str(fix / "tiny_mlp_merged.py"))
    assert merged["kernels"] < g0["kernels"]
    assert merged["est_us"] < g0["est_us"]


def test_cse_rewrite_ranks_below_duplicate(fix, gpu):
    g0 = tier0(str(fix / "tiny_dup.py"))
    cse = tier0(str(fix / "tiny_dup.py"), str(fix / "tiny_dup_cse.py"))
    assert g0["gemm_kernels"] == 2 and cse["gemm_kernels"] == 1
    assert cse["est_us"] < g0["est_us"]


def test_graph_context_finds_opportunities(fix, gpu):
    _, nodes, _, _ = trace(str(fix / "tiny_mlp.py"))
    s = analyze(nodes)
    assert s["parallel_gemms"], "gate/up share the input and should be reported"
    _, nodes, _, _ = trace(str(fix / "tiny_dup.py"))
    assert analyze(nodes)["duplicate_subexpressions"], "duplicated projection should be reported"
