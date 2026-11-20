#!/usr/bin/env python
"""End-to-end greedy generation with one e2e model (e2e/results/<model>/): G0 vs G* prefill + decode modules.

    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=4 python e2e/generate.py --model llama3-8b --batch 1 --rounds 3 --json /tmp/gen_b1.json
    CUDA_VISIBLE_DEVICES=2 python e2e/generate.py --model qwen3-8b --batch 8 --rounds 3 --json /tmp/gen_b8.json

One request = prefill of PROMPT_LEN tokens (results/<model>/prefill_b{B}) -> first token, then NEW_TOKENS-1
decode steps (results/<model>/decode_b{B}), greedy argmax in the driver.  The driver owns the KV cache: a preallocated
[L, B, n_kv, PROMPT_LEN + NEW_TOKENS, head_dim] buffer; prefill K/V are written into it, every decode step gets
the views buf[..., :len, :] and its new K/V are written at position len.  The driver code (cache writes,
argmax) is identical for every target, so only the model graphs differ.

Targets: eager G0, torch.compile G0 (default settings), torch.compile G0 with G*'s INDUCTOR_CONFIG /
COMPILE_MODE (isolates the graph rewrite), and compiled G*.  The decode cache length grows every step, so it is
marked dynamic for compiled targets (one graph for all lengths; CUDA-graph modes record one replay per length
during the warm-up request).  Every target generates one warm-up request, then `--rounds` interleaved timed
requests (CUDA events, no syncs inside a request).  Token agreement with eager G0 is reported as a sanity
check: bf16 rounding differs between graphs, so greedy paths may diverge after a near-tie.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cage.task import build_candidate, build_reference, init_args, inductor_config, load_spec  # noqa: E402

RESULTS = ROOT / "e2e" / "results"
PROMPT_LEN, NEW_TOKENS = 1024, 128
TARGETS = ("eager", "compile_default", "compile", "gstar")


def share_weights(model, source) -> bool:
    """Point `model`'s parameters at `source`'s tensors (same layout) without copying."""
    src = source.state_dict()
    if set(src) != set(model.state_dict()):
        return False
    model.load_state_dict(src, assign=True)
    return True


def build_pair(prefill_g0, decode_g0, prefill_spec, decode_spec, p_ref):
    """Prefill and decode models for one graph version, sharing one copy of the weights when possible."""
    if prefill_spec.path == prefill_g0.path:
        prefill = p_ref
    else:
        prefill = build_candidate(prefill_g0, prefill_spec, p_ref)
    torch.manual_seed(0)
    decode = decode_spec.model_cls(*init_args(decode_g0)).cuda().eval()
    if not share_weights(decode, prefill):
        del decode
        torch.cuda.empty_cache()
        decode = build_candidate(decode_g0, decode_spec, p_ref)
    torch.cuda.empty_cache()
    return prefill, decode


class Runner:
    def __init__(self, label, prefill, decode, prefill_cfg_spec=None, decode_cfg_spec=None):
        self.label = label
        self.compiled = prefill_cfg_spec is not None
        if self.compiled:
            self.p_cfg, self.d_cfg = prefill_cfg_spec.inductor_config, decode_cfg_spec.inductor_config
            with inductor_config(self.p_cfg):
                self.prefill = torch.compile(prefill, mode=self._mode(prefill_cfg_spec), fullgraph=True)
            with inductor_config(self.d_cfg):
                self.decode = torch.compile(decode, mode=self._mode(decode_cfg_spec), fullgraph=True)
            self.modes = (prefill_cfg_spec.compile_mode, decode_cfg_spec.compile_mode)
        else:
            self.p_cfg = self.d_cfg = None
            self.prefill, self.decode = prefill, decode
            self.modes = ("eager", "eager")

    @staticmethod
    def _mode(spec):
        return spec.compile_mode if spec.compile_mode not in (None, "default") else None

    def run_prefill(self, ids):
        if self.compiled:
            torch.compiler.cudagraph_mark_step_begin()
        with inductor_config(self.p_cfg):
            return self.prefill(ids)

    def run_decode(self, tok, pk, pv):
        if self.compiled:
            torch.compiler.cudagraph_mark_step_begin()
            torch._dynamo.maybe_mark_dynamic(pk, 3)
            torch._dynamo.maybe_mark_dynamic(pv, 3)
        with inductor_config(self.d_cfg):
            return self.decode(tok, pk, pv)


@torch.no_grad()
def generate(r: Runner, prompt, kbuf, vbuf, new_tokens):
    start, first, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    start.record()
    logits, k, v = r.run_prefill(prompt)
    p = prompt.shape[1]
    kbuf[:, :, :, :p].copy_(k)
    vbuf[:, :, :, :p].copy_(v)
    tok = logits[:, -1].argmax(-1, keepdim=True)
    first.record()
    tokens = [tok]
    for step in range(new_tokens - 1):
        n = p + step
        logits, k, v = r.run_decode(tok, kbuf[:, :, :, :n], vbuf[:, :, :, :n])
        kbuf[:, :, :, n:n + 1].copy_(k)
        vbuf[:, :, :, n:n + 1].copy_(v)
        tok = logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(tok)
    end.record()
    torch.cuda.synchronize()
    ttft, total = start.elapsed_time(first), start.elapsed_time(end)
    return torch.cat(tokens, 1), {"ttft_ms": ttft, "total_ms": total, "tpot_ms": (total - ttft) / (new_tokens - 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3-8b", help="folder under e2e/results (default: llama3-8b)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed requests per target (CUDA-graph trees warm up, then record, each decode length)")
    ap.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--json", default=None, help="write results here")
    args = ap.parse_args()
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    t_start = time.time()

    pdir, ddir = RESULTS / args.model / f"prefill_b{args.batch}", RESULTS / args.model / f"decode_b{args.batch}"
    if not (pdir / "g0.py").is_file() or not (ddir / "g0.py").is_file():
        raise SystemExit(f"missing {pdir} or {ddir}")
    p_g0, d_g0 = load_spec(pdir / "g0.py"), load_spec(ddir / "g0.py")
    p_gs, d_gs = load_spec(pdir / "gstar.py"), load_spec(ddir / "gstar.py")
    mod = p_g0.module
    assert mod.PROMPT_LEN == PROMPT_LEN and mod.BATCH == args.batch

    p_ref = build_reference(p_g0)
    labels = [t.strip() for t in args.targets.split(",")]
    g0_pair = build_pair(p_g0, d_g0, p_g0, d_g0, p_ref)
    runners = []
    for label in labels:
        if label == "eager":
            runners.append(Runner(label, *g0_pair))
        elif label == "compile_default":
            runners.append(Runner(label, *g0_pair, p_g0, d_g0))
        elif label == "compile":
            runners.append(Runner(label, *g0_pair, p_gs, d_gs))
        elif label == "gstar":
            runners.append(Runner(label, *build_pair(p_g0, d_g0, p_gs, d_gs, p_ref), p_gs, d_gs))
        else:
            raise SystemExit(f"unknown target {label}")

    max_len = PROMPT_LEN + args.new_tokens
    shape = (mod.N_LAYERS, args.batch, mod.N_KV, max_len, mod.HEAD_DIM)
    kbuf = torch.zeros(shape, dtype=mod.DTYPE, device="cuda")
    vbuf = torch.zeros(shape, dtype=mod.DTYPE, device="cuda")
    torch.manual_seed(0)
    prompt = torch.randint(0, mod.VOCAB, (args.batch, PROMPT_LEN), device="cuda")

    out: dict = {"model": args.model, "batch": args.batch, "prompt_len": PROMPT_LEN, "new_tokens": args.new_tokens,
                 "rounds": args.rounds, "device": torch.cuda.get_device_name(0), "targets": {}}
    tokens, samples = {}, {r.label: [] for r in runners}
    for r in runners:
        t0 = time.time()
        for _ in range(max(1, args.warmup)):
            tokens[r.label], _ = generate(r, prompt, kbuf, vbuf, args.new_tokens)
        out["targets"][r.label] = {"modes": r.modes, "warmup_s": round(time.time() - t0, 1)}
        print(f"[b{args.batch}] {r.label:16s} {args.warmup} warm-up requests (compile + record) "
              f"{time.time() - t0:7.1f}s", flush=True)
    for i in range(args.rounds):
        for r in runners:
            _, m = generate(r, prompt, kbuf, vbuf, args.new_tokens)
            samples[r.label].append(m)
        print(f"[b{args.batch}] round {i + 1}/{args.rounds}: " + "  ".join(
            f"{r.label} {samples[r.label][-1]['total_ms']:.1f}ms" for r in runners), flush=True)

    for r in runners:
        s = samples[r.label]
        res = out["targets"][r.label]
        for key in ("ttft_ms", "tpot_ms", "total_ms"):
            res[key] = round(statistics.median(x[key] for x in s), 3)
            res[key + "_all"] = [round(x[key], 3) for x in s]
        for ref in ("eager", "compile_default"):
            if ref in tokens and ref != r.label:
                same = tokens[r.label] == tokens[ref]
                suffix = "" if ref == "eager" else "_vs_compile_default"
                res["token_agreement" + suffix] = round(float(same.float().mean()), 4)
                diverge = (~same).int().argmax(1)
                res["first_divergence" + suffix] = [int(d) if not bool(row.all()) else None
                                                    for d, row in zip(diverge, same)]
    out["wall_s"] = round(time.time() - t_start, 1)
    text = json.dumps(out, indent=2)
    if args.json:
        Path(args.json).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
