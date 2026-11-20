#!/usr/bin/env bash
# Headless CAGE run with Codex.
#
#   scripts/run_codex.sh <target> [--name NAME] [--budget-min 45] [--max-evals 40] [--model M] [--extra "text"]
#
# <target> is a task file (tasks/gated_mlp.py) or any description of LLM code to optimize,
# e.g. "the LlamaMLP block in path/to/modeling_llama.py, batch 1, seq 1024".
#
# Codex's Linux sandbox (bubblewrap) cannot create user namespaces on this machine, so the
# sandboxed modes cannot run any command (nor reach the GPU); we run with danger-full-access
# and rely on AGENTS.md rules to keep the agent inside this folder.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

TARGET="${1:?usage: scripts/run_codex.sh <target> [--name N] [--budget-min M] [--max-evals E] [--model M] [--extra TEXT]}"
shift
NAME=""
BUDGET_MIN=45
MAX_EVALS=40
MODEL_ARGS=()
EXTRA=""
while [ $# -gt 0 ]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --budget-min) BUDGET_MIN="$2"; shift 2 ;;
        --max-evals) MAX_EVALS="$2"; shift 2 ;;
        --model) MODEL_ARGS=(-m "$2"); shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
if [ -z "$NAME" ]; then
    base="$(basename "${TARGET%% *}")"
    NAME="${base%.py}"
    NAME="$(echo "$NAME" | tr -c 'a-zA-Z0-9_\n' '_')"
fi

PROMPT="$(python - "$TARGET" "$NAME" "$BUDGET_MIN" "$MAX_EVALS" "$EXTRA" <<'EOF'
import sys
from pathlib import Path
target, name, budget, evals, extra = sys.argv[1:6]
tpl = Path("prompts/optimize.md").read_text()
print(tpl.replace("{TARGET}", target).replace("{RUN_NAME}", name).replace("{BUDGET_MIN}", budget)
         .replace("{MAX_EVALS}", evals).replace("{EXTRA}", extra))
EOF
)"

if ! curl -sS -m 10 -o /dev/null https://chatgpt.com/ 2>/dev/null; then
    echo "[cage] WARNING: chatgpt.com is unreachable from this host; Codex will fail with" \
         "'workspace routing discovery timed out'. Export HTTPS_PROXY/HTTP_PROXY (inherited by codex) and retry." >&2
fi

LOG_DIR="$ROOT/.cache/codex"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
echo "[cage] codex run '$NAME' (budget ${BUDGET_MIN} min / ${MAX_EVALS} evals); events -> $LOG_DIR/${NAME}_${STAMP}.jsonl"

codex exec \
    --cd "$ROOT" \
    --skip-git-repo-check \
    --sandbox danger-full-access \
    -c approval_policy='"never"' \
    -c model_reasoning_effort='"high"' \
    "${MODEL_ARGS[@]}" \
    --json \
    -o "$LOG_DIR/${NAME}_${STAMP}_final.md" \
    "$PROMPT" < /dev/null | tee "$LOG_DIR/${NAME}_${STAMP}.jsonl" | python "$ROOT/scripts/codex_events.py"

echo "[cage] final message: $LOG_DIR/${NAME}_${STAMP}_final.md"
if [ -f "runs/$NAME/report.md" ]; then
    echo "[cage] report: runs/$NAME/report.md"
fi
