#!/usr/bin/env bash
# DeGLAD-S on the whole 4ind KQG27824 test split on one GPU.  Usage:
#   scripts/run_deglad_s_4ind.sh OUTPUT_ROOT
# Outputs are written to OUTPUT_ROOT/eval/.
set -euo pipefail

output_root="${1:?Usage: $0 OUTPUT_ROOT}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
python_bin="${PYTHON_BIN:-python3}"
data_root="${FOURIND_DATA_ROOT:-/workspace/4ind}"
results_root="${FOURIND_RESULTS_ROOT:-/workspace/SimpleNet/4ind_result_date_mixed}"
manifest_path="${FOURIND_MANIFEST:-$results_root/fourind_manifest_date_mixed_seed0.csv}"
eval_dir="$output_root/eval"

cd "$repo_dir"
mkdir -p "$eval_dir"
nvidia-smi > "$eval_dir/nvidia-smi-start.txt" 2>&1

"$python_bin" eval_deglad_s_4ind.py --data-root "$data_root" --results-root "$results_root" \
    --manifest-path "$manifest_path" --category KQG27824 --output-dir "$eval_dir" \
    --num-workers 4 --num-threads 4

nvidia-smi > "$eval_dir/nvidia-smi-end.txt" 2>&1
