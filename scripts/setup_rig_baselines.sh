#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE=${RIG_BASE_ENV:-/root/autodl-tmp/envs/unirig}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
mkdir -p "$ROOT/setup"
for model in riganything puppeteer; do
  env_dir="/root/autodl-tmp/envs/$model"
  if [ ! -f "$env_dir/pyvenv.cfg" ]; then
    "$BASE/bin/python" -m venv "$env_dir"
  fi
  # Install overrides before linking shared dependencies, so pip never uninstalls shared package files.
  if [ -f "$env_dir/BASELINES_LINKED" ]; then
    echo "Environment already linked: $env_dir; validating existing packages."
  else
    "$env_dir/bin/python" -m pip install --no-cache-dir --no-deps -r "$ROOT/requirements-$model.txt"
    "$env_dir/bin/python" "$ROOT/scripts/isolate_dependencies.py" --base "$BASE" --target "$env_dir" > "$ROOT/setup/$model-dependency-links.json"
    touch "$env_dir/BASELINES_LINKED"
  fi
  "$env_dir/bin/python" -m pip check | tee "$ROOT/setup/$model-pip-check.txt"
  "$env_dir/bin/python" -m pip freeze > "$ROOT/setup/$model-environment.freeze.txt"
done
echo BASELINE_ENVIRONMENTS_READY
