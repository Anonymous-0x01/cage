# Source this file to activate the CAGE environment: `source scripts/env.sh`
# Uses the `clo` conda env (override with CAGE_CONDA_ENV) and keeps all caches in-project.
CAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CAGE_ROOT
CAGE_CONDA_ENV="${CAGE_CONDA_ENV:-clo}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CAGE_CONDA_ENV" ]; then
    _conda_base="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
    # shellcheck disable=SC1091
    source "$_conda_base/etc/profile.d/conda.sh"
    conda activate "$CAGE_CONDA_ENV"
    unset _conda_base
fi
export TRITON_CACHE_DIR="$CAGE_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$CAGE_ROOT/.cache/inductor"
export TORCH_HOME="$CAGE_ROOT/.cache/torch"
export MPLCONFIGDIR="$CAGE_ROOT/.cache/matplotlib"
export PYTHONPATH="$CAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
case ":$PATH:" in
    *":$CAGE_ROOT/bin:"*) ;;
    *) export PATH="$CAGE_ROOT/bin:$PATH" ;;
esac
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR"
