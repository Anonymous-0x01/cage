#!/usr/bin/env bash
# Verify the `clo` conda environment (PyTorch + TorchInductor + Triton) for CAGE.
# Usage: bash scripts/setup_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

python - <<'EOF'
import importlib
for mod in ("torch", "triton", "numpy", "yaml", "matplotlib", "pytest"):
    m = importlib.import_module(mod)
    print(f"{mod:12s} {getattr(m, '__version__', '?')}")
import torch
print("cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    f = torch.compile(lambda x: torch.relu(x) * 2 + 1)
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(f(x), torch.relu(x) * 2 + 1)
    print("torch.compile OK")
EOF
cage --help >/dev/null && echo "cage CLI OK"
