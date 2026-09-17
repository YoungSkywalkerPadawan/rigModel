#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${UNIRIG_PYTHON:-/root/autodl-tmp/envs/unirig/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
RUN="$ROOT/outputs/unirig_user25_$(date -u +%Y%m%dT%H%M%SZ)"
STATUS=0
"$PYTHON" -u "$ROOT/experiments/unirig/run.py" --output "$RUN" "$@" || STATUS=$?
if [ -f "$RUN/run_config.json" ]; then
  "$PYTHON" "$ROOT/experiments/unirig/score.py" --predictions "$RUN" --output "$RUN/scores"
fi
exit "$STATUS"
