#!/usr/bin/env bash
# Interactive CAGE session with Codex in this folder.
#   scripts/codex.sh                         # then type e.g. "Optimize tasks/gated_mlp.py with CAGE, 30 min budget"
#   scripts/codex.sh "Optimize tasks/qknorm.py with CAGE"
# Sandboxed modes cannot run commands on this machine (no unprivileged user namespaces for
# bubblewrap), so commands run unsandboxed; approvals stay on-request.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh
exec codex --cd "$ROOT" --sandbox danger-full-access --ask-for-approval on-request \
    -c model_reasoning_effort='"high"' "$@"
