"""Cheap cost model: estimate candidate latency without timing anything on the GPU.

Tier 0 (default, ~seconds, no compilation): trace the candidate to an aten graph under
FakeTensor, simulate Inductor-style fusion, and apply a per-kernel roofline

    t_kernel = max(FLOPs / (P_peak * gemm_eff), bytes / (BW * bw_eff)) + t_launch

Fusion simulation: views are free aliases; GEMM/conv and extern ops (topk, sort, ...) are
kernel boundaries; pointwise / reduction / copy nodes join a producer group when their
iteration domain (numel) is compatible and joining creates no cycle; GEMM epilogues are
fused only when the candidate enables Triton GEMM templates (max-autotune).  A group
reads its external inputs once and writes values consumed outside the group.

Tier 1 (``--backend``): compile with Inductor (one untimed call) and use the real kernel
list and per-kernel buffer traffic instead of the simulated fusion.
"""

from __future__ import annotations

from collections import defaultdict

from .context.graph import NodeInfo, trace
from .hw import load_hw

FUSIBLE = {"pointwise", "reduction", "copy", "other"}
MAX_FUSION_SIZE = 64


def _templates_enabled(spec) -> bool:
    cfg = spec.inductor_config or {}
    mode = spec.compile_mode or ""
    return "max-autotune" in mode or bool(cfg.get("max_autotune")) or bool(cfg.get("max_autotune_gemm"))


def _cudagraphs(spec) -> bool:
    cfg = spec.inductor_config or {}
    return (spec.compile_mode == "reduce-overhead") or bool(cfg.get("triton.cudagraphs"))


def _compatible(a: set[int], b: set[int]) -> bool:
    """Iteration domains are compatible when some sizes divide each other (broadcast / slice / reduce)."""
    return any(x % y == 0 or y % x == 0 for x in a for y in b)


def simulate_fusion(nodes: list[NodeInfo], epilogue_fusion: bool = False) -> list[dict]:
    by = {n.name: n for n in nodes}
    parent: dict[str, str] = {}

    def find(g):
        while parent.get(g, g) != g:
            g = parent[g]
        return g

    group_of: dict[str, str] = {}      # node -> group id (pre-find)
    members: dict[str, list[str]] = defaultdict(list)
    domains: dict[str, set[int]] = defaultdict(set)
    kind: dict[str, str] = {}
    anc: dict[str, set[str]] = {}      # node -> ancestor group ids (pre-find)

    def base(name):
        return by[name].base or name

    def node_domains(n: NodeInfo) -> set[int]:
        d = {n.numel}
        if n.category == "reduction":
            d |= {by[a].numel for a in n.args}
        return {x for x in d if x > 1}

    def ancestors_of(name: str) -> set[str]:
        return {find(g) for g in anc.get(name, set())}

    for n in nodes:
        if n.category in ("input", "param"):
            anc[n.name] = set()
            continue
        if n.category == "view":
            anc[n.name] = set(anc.get(n.args[0], set())) if n.args else set()
            if n.args and base(n.args[0]) in group_of:
                anc[n.name].add(group_of[base(n.args[0])])
            continue
        in_bases = [base(a) for a in n.args]
        my_anc: set[str] = set()
        for b in in_bases:
            my_anc |= anc.get(b, set())
            if b in group_of:
                my_anc.add(group_of[b])
        anc[n.name] = my_anc

        joined = None
        if n.category in FUSIBLE or (epilogue_fusion and n.category in ("pointwise", "copy")):
            nd = node_domains(n)
            cands = []
            for b in in_bases:
                if b not in group_of:
                    continue
                g = find(group_of[b])
                k = kind[g]
                if k == "fused" or (k == "gemm" and epilogue_fusion and n.category in ("pointwise", "copy")
                                    and len(by[b].users) == 1 and len(members[g]) < 8):
                    if g not in cands:
                        cands.append(g)
            for g in cands:
                if kind[g] == "fused" and nd and domains[g] and not _compatible(nd, domains[g]):
                    continue
                if len(members[g]) >= MAX_FUSION_SIZE:
                    continue
                # cycle check: an input outside g must not depend on g
                ext = [b for b in in_bases if b not in group_of or find(group_of[b]) != g]
                if any(g in ancestors_of(b) for b in ext):
                    continue
                if joined is None:
                    joined = g
                elif kind[g] == "fused" and kind[joined] == "fused":
                    # horizontal merge of two producer groups through this consumer
                    if find(joined) in {find(x) for x in ancestors_of_group(g, members, anc, find)} or \
                            find(g) in {find(x) for x in ancestors_of_group(joined, members, anc, find)}:
                        continue
                    parent[g] = joined
                    members[joined].extend(members.pop(g))
                    domains[joined] |= domains.pop(g)
            if joined is not None:
                group_of[n.name] = joined
                members[joined].append(n.name)
                domains[joined] |= nd
                continue
        gid = n.name
        parent[gid] = gid
        group_of[n.name] = gid
        members[gid] = [n.name]
        kind[gid] = "gemm" if n.category == "gemm" else ("extern" if n.category == "extern" else "fused")
        domains[gid] = node_domains(n) if kind[gid] == "fused" else set()

    groups = []
    for gid, mem in members.items():
        if find(gid) != gid:
            continue
        mset = set(mem)
        reads: dict[str, int] = {}
        writes = 0
        flops = 0
        for name in mem:
            n = by[name]
            flops += n.flops
            for a in n.args:
                b = base(a)
                if b in mset:
                    continue
                if by[b].category in ("view",):
                    continue
                read = min(by[a].bytes, by[b].bytes) if by[a].bytes else by[b].bytes
                reads[b] = max(reads.get(b, 0), read)
            consumers_outside = False
            stack = list(n.users)
            while stack:
                u = by.get(stack.pop())
                if u is None:
                    continue
                if u.category == "view":
                    if u.is_output:
                        consumers_outside = True
                    stack.extend(u.users)
                elif u.name not in mset:
                    consumers_outside = True
            if n.is_output or consumers_outside:
                writes += n.bytes
        if all(by[m].extra.get("creation") for m in mem) and writes == 0:
            continue
        groups.append({
            "id": gid, "kind": kind[gid], "nodes": mem, "ops": [by[m].op for m in mem],
            "read_bytes": int(sum(reads.values())), "write_bytes": int(writes), "flops": int(flops),
            "gemm_flops": int(sum(by[m].flops for m in mem if by[m].category == "gemm")),
            "dtype": by[mem[-1]].dtype,
        })
    order = {n.name: n.idx for n in nodes}
    groups.sort(key=lambda g: order[g["nodes"][0]])
    return groups


def ancestors_of_group(g, members, anc, find) -> set[str]:
    out = set()
    for m in members.get(g, []):
        out |= {find(x) for x in anc.get(m, set())}
    out.discard(find(g))
    return out


def kernel_time_us(flops: int, gemm_flops: int, nbytes: int, hw: dict, dtype: str = "float16") -> tuple[float, str]:
    bw = hw["hbm_gbps"] * 1e9 * hw["bw_eff"]
    tc = hw["fp16_tflops"] * 1e12 * hw["gemm_eff"]
    if dtype == "float32":
        tc = hw["fp32_tflops"] * 1e12 * hw["gemm_eff"]
    t_mem = nbytes / bw * 1e6
    t_gemm = gemm_flops / tc * 1e6 if gemm_flops else 0.0
    t_vec = (flops - gemm_flops) / (hw["fp32_tflops"] * 1e12 * 0.5) * 1e6
    t = max(t_mem, t_gemm, t_vec)
    bound = "memory" if t == t_mem else ("compute" if t == t_gemm else "vector")
    return t, bound


def tier0(task_path, cand_path=None, hw: dict | None = None) -> dict:
    hw = hw or load_hw()
    _, nodes, _, cand = trace(task_path, cand_path)
    epi = _templates_enabled(cand)
    groups = simulate_fusion(nodes, epilogue_fusion=epi)
    launch = hw["launch_us"] * (0.35 if _cudagraphs(cand) else 1.0)
    total = 0.0
    for g in groups:
        nbytes = g["read_bytes"] + g["write_bytes"]
        t, bound = kernel_time_us(g["flops"], g["gemm_flops"], nbytes, hw, g["dtype"])
        g["est_us"] = round(t + launch, 2)
        g["bound"] = bound
        total += g["est_us"]
    by = {n.name: n for n in nodes}
    materialized = sum(g["write_bytes"] for g in groups) - sum(n.bytes for n in nodes if n.is_output)
    return {
        "ok": True, "tier": 0, "est_us": round(total, 2), "kernels": len(groups),
        "gemm_kernels": sum(1 for g in groups if g["kind"] == "gemm"),
        "total_bytes": int(sum(g["read_bytes"] + g["write_bytes"] for g in groups)),
        "materialized_intermediate_bytes": int(max(0, materialized)),
        "gemm_flops": int(sum(g["gemm_flops"] for g in groups)),
        "epilogue_fusion": epi, "cudagraphs": _cudagraphs(cand),
        "groups": groups, "hw": {k: hw[k] for k in ("name", "device", "calibrated", "hbm_gbps", "bw_eff",
                                                    "fp16_tflops", "gemm_eff", "launch_us")},
        "_n_nodes": len(by),
    }


def backend_cost(task_path: str, cand_path: str, hw: dict | None = None) -> dict:
    """Tier 1 (worker side): use Inductor's actual kernels and buffer traffic."""
    import torch

    from .context.backend import compile_and_capture, summarize_backend

    hw = hw or load_hw()
    _, nodes, _, cand = trace(task_path, cand_path)
    gemm_nodes = [n for n in nodes if n.category == "gemm"]
    _, _, codes, compile_s, cand, ctx = compile_and_capture(task_path, cand_path)
    info = summarize_backend(codes, torch.cuda.get_device_properties(0).multi_processor_count)
    ctx.__exit__(None, None, None)
    launch = hw["launch_us"] * (0.35 if _cudagraphs(cand) else 1.0)
    gi = 0
    total = 0.0
    kernels = []
    for k in info["kernels"]:
        gflops = 0
        if k["kind"] == "extern" or (k["kind"] == "triton" and k["type"] == "template"):
            if gi < len(gemm_nodes):
                gflops = gemm_nodes[gi].flops
                gi += 1
        t, bound = kernel_time_us(gflops, gflops, k["bytes"], hw)
        est = round(t + launch, 2)
        total += est
        kernels.append({"name": k["name"], "type": k["type"], "bytes": k["bytes"], "gemm_flops": gflops,
                        "est_us": est, "bound": bound, "n_spills": k.get("n_spills"), "waves": k.get("waves")})
    return {"ok": True, "tier": 1, "est_us": round(total, 2), "kernels": len(kernels),
            "total_bytes": int(sum(k["bytes"] for k in kernels)), "compile_s": round(compile_s, 2),
            "kernel_list": kernels}


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:.1f}{unit}" if unit != "B" else f"{int(b)}B"
        b /= 1024
    return f"{b:.1f}TB"


def render(est: dict, ref: dict | None = None, top: int = 8) -> str:
    out = [f"== COST MODEL (tier {est['tier']}) =="]
    line = f"est={est['est_us']:.1f}us  kernels={est['kernels']}  traffic={_fmt_bytes(est['total_bytes'])}"
    if est["tier"] == 0:
        line += (f"  materialized_intermediates={_fmt_bytes(est['materialized_intermediate_bytes'])}  "
                 f"gemm={est['gemm_flops'] / 1e9:.1f}GF  epilogue_fusion={est['epilogue_fusion']}  "
                 f"cudagraphs={est['cudagraphs']}")
    out.append(line)
    if ref is not None:
        r = ref["est_us"] / est["est_us"] if est["est_us"] else float("nan")
        out.append(f"vs G0 (same tier): est {ref['est_us']:.1f}us -> {est['est_us']:.1f}us  "
                   f"=> estimated speedup {r:.3f}x  (kernels {ref['kernels']} -> {est['kernels']}, "
                   f"traffic {_fmt_bytes(ref['total_bytes'])} -> {_fmt_bytes(est['total_bytes'])})")
    if est["tier"] == 0:
        hw = est["hw"]
        out.append(f"hw={hw['name']} ({hw['device']}) calibrated={hw['calibrated']} BW={hw['hbm_gbps']}GB/s*"
                   f"{hw['bw_eff']} TC={hw['fp16_tflops']}TF*{hw['gemm_eff']} launch={hw['launch_us']}us")
        out.append("\n-- simulated kernels (most expensive first) --")
        for g in sorted(est["groups"], key=lambda g: -g["est_us"])[:top]:
            ops = ",".join(dict.fromkeys(g["ops"]))
            out.append(f"  {g['est_us']:9.1f}us {g['bound']:7s} {g['kind']:6s} r={_fmt_bytes(g['read_bytes'])} "
                       f"w={_fmt_bytes(g['write_bytes'])} gf={g['gemm_flops'] / 1e9:.1f} "
                       f"nodes={len(g['nodes'])} [{ops[:90]}]")
    else:
        out.append("\n-- Inductor kernels (most expensive first) --")
        for k in sorted(est["kernel_list"], key=lambda k: -k["est_us"])[:top]:
            out.append(f"  {k['est_us']:9.1f}us {k['bound']:7s} {k['type']:20s} io={_fmt_bytes(k['bytes'])} "
                       f"gf={k['gemm_flops'] / 1e9:.1f} {k['name'][:80]}")
    out.append("note: cheap estimate for ranking/pruning; confirm promising candidates with `cage eval`.")
    return "\n".join(out)
