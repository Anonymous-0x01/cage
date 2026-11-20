#!/usr/bin/env python
"""Validate the end-to-end G0 task files (e2e/tasks/<model>/) before optimizing them.

    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=4 python e2e/validate_g0.py                 # llama3-8b, all four tasks
    CUDA_VISIBLE_DEVICES=2 python e2e/validate_g0.py --model qwen3-8b
    CUDA_VISIBLE_DEVICES=4 python e2e/validate_g0.py --model llama3-8b --tasks prefill_b1 --skip-hf

For every task: (1) G0 matches the Hugging Face causal LM (`LlamaForCausalLM` for llama3-8b and
DeepSeek-R1-Distill-Llama-8B, `Qwen3ForCausalLM` for qwen3-8b, `attn_implementation="eager"`) loaded with the
same weights (logits and K/V), (2) the error between eager G0 and Inductor-compiled G0 on 3 seeds, which is
the floor for the task's TOLERANCE, (3) peak memory and a rough eager / compiled latency.  Writes
e2e/validation.json (keys `<model>/<case>`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cage.task import build_reference, flatten_outputs, load_spec, make_inputs, tolerance_for  # noqa: E402

def task_files(model: str) -> dict[str, Path]:
    """Map case name -> task file under e2e/tasks/<model>/.

    `llama3-8b/llama3_8b_prefill_b1.py` is addressed as `prefill_b1` (the folder name, with
    hyphens turned into underscores, is an optional filename prefix).
    """
    folder = ROOT / "e2e" / "tasks" / model
    if not folder.is_dir():
        raise SystemExit(f"no task dir {folder}")
    prefix = model.replace("-", "_") + "_"
    found = {}
    for path in sorted(folder.glob("*.py")):
        stem = path.stem
        found[stem[len(prefix):] if stem.startswith(prefix) else stem] = path
    if not found:
        raise SystemExit(f"no task files in {folder}")
    return found


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def err_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.double() - b.double()).abs().flatten()
    sample = d[torch.randint(0, d.numel(), (min(d.numel(), 1 << 24),), device=d.device)]
    q = torch.quantile(sample, torch.tensor(
        [0.5, 0.999, 0.99999], device=d.device, dtype=d.dtype)).tolist()
    return {"max": float(d.max()), "p50": q[0], "p99.9": q[1], "p99.999": q[2],
            "rel_l2": float(d.norm() / a.double().norm()), "ref_abs_max": float(a.abs().max())}


def hf_config(model: str, mod):
    """Hugging Face config + causal-LM class for the e2e model folder `model`."""
    common = dict(vocab_size=mod.VOCAB, hidden_size=mod.HIDDEN, intermediate_size=mod.INTERMEDIATE,
                  num_hidden_layers=mod.N_LAYERS, num_attention_heads=mod.N_HEADS,
                  num_key_value_heads=mod.N_KV, head_dim=mod.HEAD_DIM, rms_norm_eps=mod.EPS,
                  rope_theta=mod.ROPE_THETA, tie_word_embeddings=False, torch_dtype="bfloat16",
                  attn_implementation="eager")
    if model.startswith("llama") or model.startswith("deepseek-r1-distill-llama"):
        from transformers import LlamaConfig, LlamaForCausalLM

        return LlamaConfig(rope_scaling={"rope_type": "llama3", **mod.ROPE_SCALING},
                           max_position_embeddings=getattr(mod, "MAX_POSITION_EMBEDDINGS", 8192),
                           **common), LlamaForCausalLM
    if model.startswith("qwen3"):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        return Qwen3Config(max_position_embeddings=40960, use_sliding_window=False, attention_bias=False,
                           **common), Qwen3ForCausalLM
    raise SystemExit(f"no Hugging Face reference for model {model}; use --skip-hf")


def hf_model(model, g0, mod):
    cfg, cls = hf_config(model, mod)
    with torch.device("meta"):
        hf = cls(cfg)
    missing, unexpected = hf.load_state_dict(g0.state_dict(), strict=False, assign=True)
    assert not unexpected and not missing, (missing, unexpected)
    # the only non-persistent buffer: recompute HF's own llama3 inv_freq on the GPU
    rot = hf.model.rotary_emb
    inv_freq, rot.attention_scaling = rot.rope_init_fn(cfg, torch.device("cuda"))
    rot.inv_freq = inv_freq
    rot.original_inv_freq = inv_freq
    return hf.eval()


def layer_kv(cache, i):
    layer = cache.layers[i]
    return layer.keys, layer.values


def check_hf(model, name, g0, mod, inputs, out) -> dict:
    from transformers import DynamicCache

    hf = hf_model(model, g0, mod)
    logits, keys, values = out
    with torch.no_grad():
        if name.startswith("prefill"):
            r = hf(input_ids=inputs[0], use_cache=True, logits_to_keep=1)
            hk = torch.stack([layer_kv(r.past_key_values, i)[0] for i in range(mod.N_LAYERS)])
            hv = torch.stack([layer_kv(r.past_key_values, i)[1] for i in range(mod.N_LAYERS)])
        else:
            ids, pk, pv = inputs
            cache = DynamicCache()
            for i in range(mod.N_LAYERS):
                cache.update(pk[i].clone(), pv[i].clone(), i)
            r = hf(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            hk = torch.stack([layer_kv(r.past_key_values, i)[0][:, :, -1:] for i in range(mod.N_LAYERS)])
            hv = torch.stack([layer_kv(r.past_key_values, i)[1][:, :, -1:] for i in range(mod.N_LAYERS)])
    res = {"logits": max_err(logits, r.logits), "keys": max_err(keys, hk), "values": max_err(values, hv),
           "logits_abs_max": float(r.logits.abs().max())}
    res["ok"] = res["logits"] == 0.0 and res["keys"] == 0.0 and res["values"] == 0.0
    del hf, r
    return res


def timed(fn, inputs, iters):
    with torch.no_grad():
        fn(*inputs)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn(*inputs)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def validate(model: str, name: str, path: Path, skip_hf: bool, seeds: int) -> dict:
    task = load_spec(path)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    g0 = build_reference(task)
    res: dict = {"task": name, "build_s": round(time.time() - t0, 1),
                 "weights_gb": round(sum(p.numel() * p.element_size() for p in g0.parameters()) / 1e9, 2)}
    inputs = make_inputs(task, 0)
    with torch.no_grad():
        out = g0(*inputs)
    res["eager_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if not skip_hf:
        res["hf"] = check_hf(model, name, g0, task.module, inputs, out)
        print(f"[{name}] HF eager match: {res['hf']}", flush=True)

    t0 = time.time()
    compiled = torch.compile(g0, fullgraph=True)
    errs = []
    with torch.no_grad():
        for seed in range(seeds):
            x = make_inputs(task, seed)
            ref = flatten_outputs(g0(*x))
            got = flatten_outputs(compiled(*x))
            if seed == 0:
                res["compile_s"] = round(time.time() - t0, 1)
            errs.append([err_stats(a, b) for a, b in zip(ref, got)])
    res["eager_vs_inductor"] = {k: {s: max(e[i][s] for e in errs) for s in errs[0][i]}
                                for i, k in enumerate(("logits", "keys", "values"))}
    res["default_bf16_tol"] = tolerance_for(task, torch.bfloat16)
    res["task_tol"] = task.tolerance
    iters = 3 if name.startswith("prefill") else 20
    res["eager_ms"] = round(timed(g0, inputs, iters), 3)
    res["inductor_ms"] = round(timed(compiled, inputs, iters), 3)
    res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(f"[{name}] {json.dumps({k: v for k, v in res.items() if k != 'hf'})}", flush=True)
    del g0, compiled, out
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3-8b", help="folder under e2e/tasks (default: llama3-8b)")
    ap.add_argument("--tasks", default="all", help="case names, comma-separated (default: all files in the model folder)")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    tasks = task_files(args.model)
    names = list(tasks) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [n for n in names if n not in tasks]
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}; available: {list(tasks)}")
    rows = [validate(args.model, name, tasks[name], args.skip_hf, args.seeds) for name in names]
    out = ROOT / "e2e" / "validation.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update({f"{args.model}/{r['task']}": r for r in rows})
    out.write_text(json.dumps(old, indent=2))
    return 0 if all(r.get("hf", {"ok": True})["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
