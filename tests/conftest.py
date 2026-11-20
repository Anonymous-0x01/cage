import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache" / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache" / "inductor"))


def _gpu() -> str:
    from cage.measure import pick_gpu

    return pick_gpu()


@pytest.fixture(scope="session")
def gpu():
    import subprocess

    if shutil.which("nvidia-smi") is None or subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode:
        pytest.skip("no NVIDIA GPU")
    g = _gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = g
    return g


@pytest.fixture
def fix():
    return FIX
