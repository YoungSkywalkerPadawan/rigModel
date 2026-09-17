#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:?Usage: bash scripts/run_rig_baselines.sh riganything|puppeteer [--case CASE]}
shift
case "$model" in riganything|puppeteer) ;; *) echo "Unknown model: $model" >&2; exit 2;; esac
python="/root/autodl-tmp/envs/$model/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
run_dir="$ROOT/outputs/${model}_user25_$(date -u +%Y%m%dT%H%M%SZ)"
status=0
"$python" "$ROOT/experiments/rig_baselines/run.py" --model "$model" --output "$run_dir" "$@" || status=$?
if [ -f "$run_dir/run_config.json" ]; then
  "$python" "$ROOT/experiments/unirig/score.py" --data "$ROOT/data/user25_$model" --predictions "$run_dir" --output "$run_dir/scores"
  if [ -f "$run_dir/summary.json" ]; then
    "$python" "$ROOT/experiments/rig_baselines/audit_run.py" --model "$model" --run "$run_dir" || status=$?
  fi
fi
echo "RUN_DIRECTORY=$run_dir"
exit "$status"
