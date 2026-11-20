"""Graph-level context: trace a candidate to an aten FX graph under FakeTensor (no GPU memory, no kernels).

Reports operator semantics, shapes/dtypes/strides, producer-consumer edges, large
intermediates, layout conversions and casts, parallel GEMMs that share an input,
duplicated sub-expressions, and fusion boundaries around GEMMs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

from ..task import init_args, load_spec

GEMM_OPS = {
    "mm", "addmm", "bmm", "baddbmm", "matmul", "linear", "convolution", "_convolution", "conv2d", "conv1d",
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention", "scaled_dot_product_attention", "_scaled_mm", "addbmm", "einsum",
}
REDUCTION_OPS = {
    "sum", "mean", "amax", "amin", "max", "min", "var", "var_mean", "std", "std_mean", "prod", "argmax",
    "argmin", "_softmax", "softmax", "_log_softmax", "log_softmax", "logsumexp", "norm", "linalg_vector_norm",
    "any", "all", "cumsum", "cumprod", "_safe_softmax", "native_layer_norm", "native_group_norm",
    "native_batch_norm", "_native_batch_norm_legit_no_training", "avg_pool2d", "max_pool2d_with_indices",
    "avg_pool1d", "adaptive_avg_pool2d", "_adaptive_avg_pool2d", "scatter_add", "scatter_reduce", "index_add",
    "segment_reduce", "nansum", "count_nonzero",
}
EXTERN_OPS = {"topk", "sort", "argsort", "kthvalue", "median", "nonzero", "unique", "_unique2", "searchsorted",
              "bucketize", "multinomial", "cumsum_", "_embedding_bag"}
COPY_OPS = {"clone", "contiguous", "_to_copy", "copy", "copy_", "cat", "stack", "constant_pad_nd", "repeat",
            "flip", "roll", "index", "index_select", "gather", "scatter", "embedding", "where", "tril", "triu",
            "_unsafe_index", "index_put", "masked_fill", "pad", "fill", "full", "zeros", "ones", "empty",
            "arange", "full_like", "zeros_like", "ones_like", "empty_like", "new_zeros", "new_ones", "new_empty",
            "scalar_tensor", "lift_fresh_copy", "_unsafe_view_copy", "slice_scatter", "select_scatter"}
VIEW_OPS = {"view", "_unsafe_view", "reshape", "expand", "unsqueeze", "squeeze", "permute", "transpose", "t",
            "slice", "select", "split", "split_with_sizes", "alias", "as_strided", "unbind", "chunk", "narrow",
            "view_as_real", "view_as_complex", "diagonal", "unfold", "detach", "_reshape_alias", "unflatten",
            "flatten", "movedim", "squeeze_copy", "expand_as", "view_as", "lift_fresh", "_unsafe_view_"}
LAYOUT_VIEW_OPS = {"permute", "transpose", "t", "expand", "movedim", "as_strided", "diagonal", "unfold"}
CREATION_OPS = {"full", "zeros", "ones", "empty", "arange", "scalar_tensor", "empty_like", "zeros_like",
                "ones_like", "full_like", "new_zeros", "new_ones", "new_empty", "empty_strided"}


@dataclass
class NodeInfo:
    idx: int
    name: str
    op: str                      # short aten name, e.g. "mm"
    target: str                  # full target string
    category: str                # input/param/gemm/pointwise/reduction/view/copy/extern/other/output
    shape: list
    dtype: str
    stride: list
    numel: int
    bytes: int
    contiguous: bool
    args: list = field(default_factory=list)      # producer node names
    users: list = field(default_factory=list)     # consumer node names
    flops: int = 0
    base: str = ""               # storage-owning node for views
    is_output: bool = False
    extra: dict = field(default_factory=dict)


def _short(target) -> str:
    s = str(target)
    if s.startswith("aten."):
        s = s[5:]
    name = getattr(target, "__name__", None)
    if hasattr(target, "_opname"):
        return target._opname
    return s.split(".")[0] if "." in s else (name or s)


def _category(node, op: str) -> str:
    import torch

    if node.op == "placeholder":
        return "input"
    if node.op == "output":
        return "output"
    if node.op == "get_attr":
        return "param"
    tgt = node.target
    if op in GEMM_OPS:
        return "gemm"
    if op in VIEW_OPS or getattr(tgt, "is_view", False):
        return "view"
    if op in EXTERN_OPS:
        return "extern"
    if op in REDUCTION_OPS:
        return "reduction"
    tags = getattr(tgt, "tags", ()) or ()
    if torch.Tag.pointwise in tags:
        return "pointwise"
    if op in COPY_OPS:
        return "copy"
    if op in ("getitem",):
        return "view"
    return "other"


def _tensor_meta(val):
    import torch

    if isinstance(val, torch.Tensor):
        shape = [int(s) for s in val.shape]
        numel = 1
        for s in shape:
            numel *= s
        try:
            stride = [int(s) for s in val.stride()]
            contiguous = bool(val.is_contiguous())
        except Exception:
            stride, contiguous = [], True
        return shape, str(val.dtype).replace("torch.", ""), stride, numel, numel * val.element_size(), contiguous
    if isinstance(val, (list, tuple)) and val and all(isinstance(v, torch.Tensor) for v in val):
        metas = [_tensor_meta(v) for v in val]
        return ([m[0] for m in metas], metas[0][1], [m[2] for m in metas], sum(m[3] for m in metas),
                sum(m[4] for m in metas), all(m[5] for m in metas))
    return [], type(val).__name__, [], 0, 0, True


def trace(task_path, cand_path=None):
    """Trace ``cand_path`` (default: the task) to an aten FX graph under FakeTensor.

    Returns ``(gm, nodes: list[NodeInfo], param_names: dict[placeholder->param name], cand_spec)``.
    """
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.func import functional_call
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.utils.flop_counter import FlopCounterMode

    task = load_spec(task_path)
    cand = load_spec(cand_path) if cand_path else task
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode, torch.no_grad():
        torch.manual_seed(0)
        inputs = list(task.get_inputs())
        dev = next((x.device for x in inputs if isinstance(x, torch.Tensor)), torch.device("cuda"))
        with torch.device(dev):
            model = cand.model_cls(*init_args(task)).eval()
        stray = [k for k, v in list(model.named_parameters()) + list(model.named_buffers()) if v.device != dev]
        if stray:
            raise RuntimeError(f"parameters not on {dev} after construction under torch.device: {stray[:5]}")
        params = {**dict(model.named_parameters()), **dict(model.named_buffers())}

        def fn(p, *args):
            return functional_call(model, p, args)

        gm = make_fx(fn, tracing_mode="real")(params, *inputs)

        flops: dict[str, int] = {}

        class _Interp(torch.fx.Interpreter):
            def run_node(self, n):
                with FlopCounterMode(display=False) as fc:
                    r = super().run_node(n)
                flops[n.name] = int(fc.get_total_flops())
                return r

        try:
            _Interp(gm).run(params, *inputs)
        except Exception:
            pass

    pnames = list(params.keys())
    placeholder_names = {}
    nodes: list[NodeInfo] = []
    by_name: dict[str, NodeInfo] = {}
    ph_i = 0
    output_names: set[str] = set()
    for n in gm.graph.nodes:
        if n.op == "output":
            import torch.utils._pytree as pytree

            for a in pytree.tree_leaves(n.args):
                if isinstance(a, torch.fx.Node):
                    output_names.add(a.name)
    display: dict[str, str] = {}
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            if ph_i < len(pnames):
                display[n.name] = "p_" + pnames[ph_i].replace(".", "_")
            else:
                display[n.name] = f"in{ph_i - len(pnames)}"
            ph_i += 1
    ph_i = 0
    output_names = {display.get(x, x) for x in output_names}
    for i, n in enumerate(gm.graph.nodes):
        if n.op == "output":
            continue
        op = "placeholder" if n.op == "placeholder" else _short(n.target)
        cat = _category(n, op)
        extra = {}
        if n.op == "placeholder":
            if ph_i < len(pnames):
                cat = "param"
                placeholder_names[n.name] = pnames[ph_i]
                extra["param"] = pnames[ph_i]
            else:
                extra["input_index"] = ph_i - len(pnames)
            ph_i += 1
        shape, dtype, stride, numel, nbytes, contig = _tensor_meta(n.meta.get("val"))
        args = [display.get(a.name, a.name) for a in n.all_input_nodes]
        name = display.get(n.name, n.name)
        info = NodeInfo(
            idx=i, name=name, op=op, target=str(n.target), category=cat, shape=shape, dtype=dtype,
            stride=stride, numel=numel, bytes=nbytes, contiguous=contig, args=args,
            users=[u.name for u in n.users if u.op != "output"], flops=flops.get(n.name, 0),
            is_output=name in output_names, extra=extra,
        )
        if n.op == "call_function" and cat in ("pointwise", "reduction", "copy"):
            info.flops = info.flops or numel
        if cat == "copy" and op in CREATION_OPS:
            info.extra["creation"] = True
        if n.op == "call_function":
            non_node = [a for a in n.args if not isinstance(a, torch.fx.Node)]
            if non_node:
                info.extra["consts"] = repr(non_node)[:80]
        nodes.append(info)
        by_name[name] = info
    for info in nodes:
        if info.category == "view" and info.args:
            info.base = by_name[info.args[0]].base or info.args[0]
        else:
            info.base = info.name
    return gm, nodes, placeholder_names, cand


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:.1f}{unit}" if unit != "B" else f"{int(b)}B"
        b /= 1024
    return f"{b:.1f}TB"


def analyze(nodes: list[NodeInfo]) -> dict:
    by_name = {n.name: n for n in nodes}
    compute = [n for n in nodes if n.category not in ("input", "param", "view")]
    inputs = [n for n in nodes if n.category == "input"]
    params = [n for n in nodes if n.category == "param"]
    outputs = [n for n in nodes if n.is_output]

    def producer_chain(name):
        """Walk back through views to the materialized producer."""
        cur = by_name[name]
        chain = []
        while cur.category == "view" and cur.args:
            chain.append(cur.op)
            cur = by_name[cur.args[0]]
        return cur, chain

    intermediates = sorted(
        [n for n in compute if not n.is_output and n.bytes > 0 and not n.extra.get("creation")],
        key=lambda n: -n.bytes)
    layout = []
    casts = []
    for n in compute:
        if n.op in ("clone", "contiguous", "_to_copy", "copy") and n.args:
            src, chain = producer_chain(n.args[0])
            relayout = [c for c in chain if c in LAYOUT_VIEW_OPS]
            if relayout or (n.op == "clone" and not chain):
                layout.append({"node": n.name, "op": n.op, "from": src.name, "views": chain, "bytes": n.bytes,
                               "shape": n.shape})
            if n.op == "_to_copy" and src.dtype != n.dtype:
                casts.append({"node": n.name, "from": src.name, "src_dtype": src.dtype, "dst_dtype": n.dtype,
                              "bytes": n.bytes, "shape": n.shape})

    gemm_groups = defaultdict(list)
    for n in compute:
        if n.category == "gemm" and n.args:
            act_args = [a for a in n.args if by_name[a].base and by_name[by_name[a].base].category != "param"]
            if act_args:
                gemm_groups[by_name[act_args[0]].base].append(n.name)
    parallel_gemms = [{"shared_input": k, "gemms": v} for k, v in gemm_groups.items() if len(v) > 1]

    canon_cache: dict[str, object] = {}

    def canon(name):
        """Structural key that sees through view chains (two `t(W)` views are the same operand)."""
        if name not in canon_cache:
            n = by_name[name]
            if n.category == "view" and n.args:
                canon_cache[name] = (n.op, n.extra.get("consts"), canon(n.args[0]))
            else:
                canon_cache[name] = name
        return canon_cache[name]

    sig = defaultdict(list)
    for n in compute:
        if n.category in ("pointwise", "reduction", "gemm", "copy") and not n.extra.get("creation"):
            sig[(n.target, tuple(canon(a) for a in n.args), n.extra.get("consts"))].append(n.name)
    duplicates = [v for v in sig.values() if len(v) > 1]

    broadcast_materialized = []
    for n in compute:
        if n.op in ("clone", "contiguous", "_to_copy", "copy", "repeat") and n.args:
            src, chain = producer_chain(n.args[0])
            if "expand" in chain and n.numel > 2 * max(src.numel, 1):
                broadcast_materialized.append({"node": n.name, "from": src.name, "bytes": n.bytes,
                                               "blowup": round(n.numel / max(src.numel, 1), 1)})

    epilogue, prologue = [], []
    for n in compute:
        if n.category == "gemm":
            for u in n.users:
                un = by_name[u]
                while un.category == "view" and len(un.users) == 1:
                    un = by_name[un.users[0]]
                if un.category in ("pointwise", "reduction", "copy"):
                    epilogue.append({"gemm": n.name, "consumer": un.name, "op": un.op})
            for a in n.args:
                src, _ = producer_chain(a)
                if src.category in ("pointwise", "copy", "reduction"):
                    prologue.append({"gemm": n.name, "producer": src.name, "op": src.op, "bytes": src.bytes})

    fp32_upcasts = [c for c in casts if c["dst_dtype"] == "float32"]
    total_flops = sum(n.flops for n in nodes if n.category == "gemm")
    return {
        "counts": {
            "nodes": len(nodes), "compute_nodes": len(compute),
            "by_category": {c: sum(1 for n in nodes if n.category == c)
                            for c in ("gemm", "pointwise", "reduction", "copy", "extern", "view", "other")},
        },
        "inputs": [{"name": n.name, "shape": n.shape, "dtype": n.dtype, "bytes": n.bytes} for n in inputs],
        "params": [{"name": n.extra.get("param"), "shape": n.shape, "dtype": n.dtype, "bytes": n.bytes}
                   for n in params],
        "outputs": [{"name": n.name, "shape": n.shape, "dtype": n.dtype, "bytes": n.bytes} for n in outputs],
        "totals": {
            "gemm_flops": total_flops,
            "input_bytes": sum(n.bytes for n in inputs),
            "param_bytes": sum(n.bytes for n in params),
            "output_bytes": sum(n.bytes for n in outputs),
            "intermediate_bytes_unfused": sum(n.bytes for n in intermediates),
        },
        "largest_intermediates": [{"node": n.name, "op": n.op, "shape": n.shape, "dtype": n.dtype,
                                   "bytes": n.bytes, "users": n.users} for n in intermediates[:8]],
        "layout_copies": layout,
        "casts": casts,
        "fp32_upcasts": len(fp32_upcasts),
        "parallel_gemms": parallel_gemms,
        "duplicate_subexpressions": duplicates,
        "broadcast_materialized": broadcast_materialized,
        "gemm_epilogue_candidates": epilogue[:12],
        "gemm_prologue_candidates": prologue[:12],
    }


def graph_context(task_path, cand_path=None, max_nodes: int = 200) -> dict:
    _, nodes, _, cand = trace(task_path, cand_path)
    summary = analyze(nodes)
    return {
        "ok": True,
        "inductor_config": cand.inductor_config,
        "compile_mode": cand.compile_mode,
        "summary": summary,
        "nodes": [asdict(n) for n in nodes[:max_nodes]],
        "truncated": max(0, len(nodes) - max_nodes),
    }


def render(ctx: dict) -> str:
    s = ctx["summary"]
    out = []
    out.append("== GRAPH CONTEXT ==")
    c = s["counts"]
    out.append(f"nodes={c['nodes']} compute={c['compute_nodes']} by_category={c['by_category']}")
    t = s["totals"]
    out.append(f"gemm_flops={t['gemm_flops'] / 1e9:.3f} GFLOP  inputs={_fmt_bytes(t['input_bytes'])}  "
               f"params={_fmt_bytes(t['param_bytes'])}  outputs={_fmt_bytes(t['output_bytes'])}  "
               f"unfused_intermediates={_fmt_bytes(t['intermediate_bytes_unfused'])}")
    if ctx.get("inductor_config") or ctx.get("compile_mode"):
        out.append(f"compile_mode={ctx.get('compile_mode')} INDUCTOR_CONFIG={ctx.get('inductor_config')}")
    out.append("\n-- inputs / params / outputs --")
    for k in ("inputs", "params", "outputs"):
        for x in s[k]:
            out.append(f"  {k[:-1]:6s} {x['name']}: {x['shape']} {x['dtype']} ({_fmt_bytes(x['bytes'])})")
    out.append("\n-- aten graph (idx name = op(args) -> shape dtype [category] flags) --")
    for n in ctx["nodes"]:
        if n["category"] in ("input", "param"):
            continue
        flags = []
        if not n["contiguous"]:
            flags.append(f"strided{tuple(n['stride'])}")
        if n["is_output"]:
            flags.append("OUT")
        if n["flops"] and n["category"] == "gemm":
            flags.append(f"{n['flops'] / 1e9:.2f}GF")
        if n["extra"].get("consts"):
            flags.append(n["extra"]["consts"])
        out.append(f"  {n['idx']:3d} {n['name']} = {n['op']}({', '.join(n['args'])}) -> {n['shape']} {n['dtype']} "
                   f"[{n['category']}] {_fmt_bytes(n['bytes'])} {' '.join(flags)}".rstrip())
    if ctx.get("truncated"):
        out.append(f"  ... {ctx['truncated']} more nodes (use --max-nodes)")
    out.append("\n-- largest intermediates (materialized if not fused) --")
    for x in s["largest_intermediates"]:
        out.append(f"  {x['node']} ({x['op']}) {x['shape']} {x['dtype']} {_fmt_bytes(x['bytes'])} users={x['users']}")
    if s["layout_copies"]:
        out.append("\n-- layout copies (relayout materialization) --")
        for x in s["layout_copies"]:
            out.append(f"  {x['node']} = {x['op']}({x['from']} via {x['views']}) {x['shape']} {_fmt_bytes(x['bytes'])}")
    if s["casts"]:
        out.append(f"\n-- dtype casts ({s['fp32_upcasts']} fp32 upcasts) --")
        for x in s["casts"][:12]:
            out.append(f"  {x['node']}: {x['src_dtype']}->{x['dst_dtype']} {x['shape']} {_fmt_bytes(x['bytes'])}")
    if s["parallel_gemms"]:
        out.append("\n-- parallel GEMMs sharing an input (merge candidates) --")
        for x in s["parallel_gemms"]:
            out.append(f"  input {x['shared_input']}: {x['gemms']}")
    if s["duplicate_subexpressions"]:
        out.append("\n-- duplicated sub-expressions (recomputation) --")
        for x in s["duplicate_subexpressions"]:
            out.append(f"  {x}")
    if s["broadcast_materialized"]:
        out.append("\n-- materialized broadcasts --")
        for x in s["broadcast_materialized"]:
            out.append(f"  {x['node']} from {x['from']} blowup x{x['blowup']} {_fmt_bytes(x['bytes'])}")
    if s["gemm_epilogue_candidates"] or s["gemm_prologue_candidates"]:
        out.append("\n-- GEMM fusion boundaries --")
        for x in s["gemm_prologue_candidates"]:
            out.append(f"  prologue: {x['producer']}({x['op']}, {_fmt_bytes(x['bytes'])}) -> {x['gemm']}")
        for x in s["gemm_epilogue_candidates"]:
            out.append(f"  epilogue: {x['gemm']} -> {x['consumer']}({x['op']})")
    return "\n".join(out)
