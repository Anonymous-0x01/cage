"""Task / candidate contract.

A *task* file (the original graph G0) is a Python module that defines:

    class Model(nn.Module)          # the computation subgraph
    def get_init_inputs() -> list   # positional args for Model(...)
    def get_inputs() -> list        # positional args for Model.forward(...)

    TOLERANCE = {"atol": ..., "rtol": ...}   # optional, numerical tolerance

A *candidate* file (a transformed graph G_t) defines ``Model`` with the same
constructor signature, plus optionally:

    INDUCTOR_CONFIG = {...}          # whitelisted torch._inductor.config overrides
    COMPILE_MODE = "max-autotune-no-cudagraphs"   # torch.compile mode
    def convert_weights(ref_state_dict) -> state_dict   # offline weight re-layout

Inputs, init args and tolerance always come from the task (G0), never from the
candidate, so a candidate cannot change the problem it is solving.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ALLOWED_COMPILE_MODES = (
    None,
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)

# torch._inductor.config keys a candidate may override (backend-level transformation knobs).
INDUCTOR_CONFIG_WHITELIST = frozenset(
    {
        "max_autotune",
        "max_autotune_pointwise",
        "max_autotune_gemm",
        "max_autotune_gemm_backends",
        "max_autotune_gemm_search_space",
        "autotune_num_choices_displayed",
        "coordinate_descent_tuning",
        "coordinate_descent_check_all_directions",
        "coordinate_descent_search_radius",
        "epilogue_fusion",
        "epilogue_fusion_first",
        "prologue_fusion",
        "pattern_matcher",
        "aggressive_fusion",
        "max_fusion_size",
        "max_pointwise_cat_inputs",
        "split_reductions",
        "unroll_reductions_threshold",
        "realize_reads_threshold",
        "realize_opcount_threshold",
        "realize_acc_reads_threshold",
        "permute_fusion",
        "shape_padding",
        "comprehensive_padding",
        "pad_channels_last",
        "padding_alignment_bytes",
        "b2b_gemm_pass",
        "benchmark_fusion",
        "benchmark_epilogue_fusion",
        "loop_ordering_after_fusion",
        "score_fusion_memory_threshold",
        "reorder_for_locality",
        "inplace_buffers",
        "allow_buffer_reuse",
        "combo_kernels",
        "benchmark_combo_kernel",
        "group_fusion",
        "batch_fusion",
        "pre_grad_fusion_options",
        "post_grad_fusion_options",
        "decompose_mem_bound_mm",
        "freezing",
        "use_mixed_mm",
        "force_fuse_int_mm_with_mul",
        "triton.cudagraphs",
        "triton.persistent_reductions",
        "triton.multi_kernel",
        "triton.max_tiles",
        "triton.tile_reductions",
        "triton.prefer_nd_tiling",
        "triton.use_block_ptr",
        "triton.cooperative_reductions",
        "triton.enable_persistent_tma_matmul",
        "triton.divisible_by_16",
        "triton.min_split_scan_rblock",
        "triton.mix_order_reduction",
    }
)

DEFAULT_TOLERANCE = {
    "float16": {"atol": 1e-2, "rtol": 1e-2},
    "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
    "float32": {"atol": 1e-4, "rtol": 1e-4},
    "float64": {"atol": 1e-7, "rtol": 1e-7},
}


@dataclass
class Spec:
    path: Path
    module: Any
    model_cls: type
    get_inputs: Callable[[], list] | None
    get_init_inputs: Callable[[], list] | None
    inductor_config: dict = field(default_factory=dict)
    compile_mode: str | None = None
    tolerance: dict | None = None
    convert_weights: Callable[[dict], dict] | None = None

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.path.read_bytes()).hexdigest()[:12]


def load_module(path: str | Path):
    path = Path(path).resolve()
    name = "cage_user_" + hashlib.sha1(str(path).encode()).hexdigest()[:10] + "_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_spec(path: str | Path) -> Spec:
    module = load_module(path)
    if not hasattr(module, "Model"):
        raise AttributeError(f"{path}: missing `class Model(nn.Module)`")
    return Spec(
        path=Path(path).resolve(),
        module=module,
        model_cls=module.Model,
        get_inputs=getattr(module, "get_inputs", None),
        get_init_inputs=getattr(module, "get_init_inputs", None),
        inductor_config=dict(getattr(module, "INDUCTOR_CONFIG", {}) or {}),
        compile_mode=getattr(module, "COMPILE_MODE", None),
        tolerance=getattr(module, "TOLERANCE", None),
        convert_weights=getattr(module, "convert_weights", None),
    )


def _to_device(obj, device):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def make_inputs(task: Spec, seed: int = 0, device: str = "cuda") -> list:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return list(_to_device(task.get_inputs(), device))


def init_args(task: Spec) -> list:
    import torch

    torch.manual_seed(1234)
    return list(task.get_init_inputs()) if task.get_init_inputs else []


def build_reference(task: Spec, device: str = "cuda"):
    import torch

    torch.manual_seed(0)
    model = task.model_cls(*init_args(task)).to(device).eval()
    return model


def build_candidate(task: Spec, cand: Spec, ref_model, device: str = "cuda"):
    """Instantiate a candidate with the reference weights (optionally re-laid-out)."""
    import torch

    torch.manual_seed(0)
    model = cand.model_cls(*init_args(task)).to(device).eval()
    sd = {k: v.detach().clone() for k, v in ref_model.state_dict().items()}
    if cand.convert_weights is not None:
        with torch.no_grad():
            sd = cand.convert_weights(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"candidate state_dict mismatch: missing={list(missing)} unexpected={list(unexpected)}; "
            "define convert_weights(ref_state_dict) to map reference weights to the candidate layout"
        )
    return model


def tolerance_for(task: Spec, dtype) -> dict:
    name = str(dtype).replace("torch.", "")
    tol = dict(DEFAULT_TOLERANCE.get(name, {"atol": 1e-2, "rtol": 1e-2}))
    if isinstance(task.tolerance, dict):
        if "atol" in task.tolerance or "rtol" in task.tolerance:
            tol.update({k: v for k, v in task.tolerance.items() if k in ("atol", "rtol")})
        elif name in task.tolerance:
            tol.update(task.tolerance[name])
    return tol


@contextlib.contextmanager
def inductor_config(cfg: dict | None):
    from torch._inductor import config as icfg

    if not cfg:
        yield
        return
    with icfg.patch(dict(cfg)):
        yield


def compile_model(model, cand: Spec, fullgraph: bool = True):
    """torch.compile with the candidate's mode; call inside ``inductor_config(cand.inductor_config)``."""
    import torch

    mode = cand.compile_mode if cand.compile_mode not in (None, "default") else None
    return torch.compile(model, mode=mode, fullgraph=fullgraph)


def clone_inputs(inputs):
    import torch

    return [x.clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x) for x in inputs]


def flatten_outputs(out) -> list:
    import torch.utils._pytree as pytree

    leaves, _ = pytree.tree_flatten(out)
    return leaves
