#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE=${UNIRIG_BASE_ENV:-/root/autodl-tmp/envs/uniphysgen}
ENV=${UNIRIG_ENV:-/root/autodl-tmp/envs/unirig}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$ROOT/setup" "$ROOT/models"
if [ ! -f "$ENV/pyvenv.cfg" ]; then
  "$BASE/bin/python" -m venv --system-site-packages "$ENV"
fi
"$ENV/bin/python" -m pip install --no-cache-dir -r "$ROOT/requirements-unirig.txt"
"$ENV/bin/python" -m pip install --no-cache-dir --no-deps --only-binary=:all: 'torch-cluster==1.6.3+pt24cu124' -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
"$ENV/bin/python" "$ROOT/scripts/isolate_dependencies.py" --base "$BASE" --target "$ENV" > "$ROOT/setup/dependency-links.json"
"$ENV/bin/python" -m pip check | tee "$ROOT/setup/pip-check.txt"
"$ENV/bin/python" -m pip freeze > "$ROOT/setup/environment.freeze.txt"
"$ENV/bin/python" "$ROOT/scripts/download_assets.py" --endpoint "${HF_ENDPOINT:-https://hf-mirror.com}"
echo DEPENDENCIES_AND_ASSETS_READY
