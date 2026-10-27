#!/usr/bin/env bash
# Profile 4ind inference of global_local, simplenet, and simplenet_plus on one
# GPU.  Usage: scripts/profile_gl_4ind.sh OUTPUT_ROOT
# Outputs are written to OUTPUT_ROOT/profile/.
set -euo pipefail

output_root="${1:?Usage: $0 OUTPUT_ROOT}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
python_bin="${PYTHON_BIN:-python3}"
data_root="${FOURIND_DATA_ROOT:-/workspace/4ind}"
results_root="${FOURIND_RESULTS_ROOT:-/workspace/SimpleNet/4ind_result_date_mixed}"
manifest_path="${FOURIND_MANIFEST:-$results_root/fourind_manifest_date_mixed_seed0.csv}"

cd "$repo_dir"
"$python_bin" profile_gl_inference.py \
    --data-root "$data_root" \
    --results-root "$results_root" \
    --manifest-path "$manifest_path" \
    --category KQG27824 \
    --output-dir "$output_root/profile" \
    --image-count 100 \
    --warmup-images 5 \
    --baseline-batch-sizes "${BASELINE_BATCH_SIZES:-1,8}" \
    --num-workers 4 \
    --num-threads 4
