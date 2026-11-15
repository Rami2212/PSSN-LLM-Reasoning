#!/usr/bin/env bash
# From the repository root, after environment, HF access and rclone are configured.
set -euo pipefail
: "${V6_RUN:?Export V6_RUN to the local experiment directory}"
: "${V6_REMOTE:?Export V6_REMOTE, e.g. gdrive:PSSN2/v6/controller-text-100x3}"
test -f v6/controller_common.py
mkdir -p "$V6_RUN/logs" "$V6_RUN/source-v6"
exec 9>"$V6_RUN/.pipeline.lock"
flock -n 9 || { printf 'Another V6 pipeline holds this run lock.\n' >&2; exit 1; }
cp -a v6/. "$V6_RUN/source-v6/"
python -m pip freeze > "$V6_RUN/logs/pip-freeze.txt"
if command -v rocm-smi >/dev/null; then rocm-smi > "$V6_RUN/logs/gpu.txt"; fi
run_stage() {
  local stage="$1"
  shift
  python -u -m "$@" 2>&1 | tee -a "$V6_RUN/logs/$stage.log"
  bash v6/controller_backup.sh "$V6_RUN" "$stage" "$V6_REMOTE"
}
run_stage prepared v6.prepare --output-dir "$V6_RUN/prepared" --rows 100 --test-rows 100
run_stage labels v6.controller_label --prepared-dir "$V6_RUN/prepared" --output-dir "$V6_RUN/labels"
run_stage training v6.controller_train --labels-dir "$V6_RUN/labels" --output-dir "$V6_RUN/training"
run_stage evaluation v6.controller_evaluate --labels-dir "$V6_RUN/labels" \
  --controller "$V6_RUN/training/controller.joblib" --output-dir "$V6_RUN/evaluation"
run_stage comparison v6.controller_test --prepared-dir "$V6_RUN/prepared" \
  --controller "$V6_RUN/training/controller.joblib" --output-dir "$V6_RUN/comparison" --repeats 3
printf 'Finished. Read %s/comparison/comparison.md\n' "$V6_RUN"
