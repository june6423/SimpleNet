#!/usr/bin/env bash
# Profile the old and the fast layer3 interpolation of SimpleNet._embed on 4ind
# KQG27824 on one GPU: four global-local configurations C0 to C3, the _embed
# tensor check, and the two baselines with the flag off and on.  Usage:
#   scripts/profile_gl_4ind_s4fix.sh OUTPUT_ROOT
# Outputs are written to OUTPUT_ROOT/profile/.  The memory use and the
# utilization of the visible GPU are sampled every 15 seconds for the whole run
# to OUTPUT_ROOT/profile/gpu<index>_samples.csv.
set -euo pipefail

output_root="${1:?Usage: $0 OUTPUT_ROOT}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
python_bin="${PYTHON_BIN:-python3}"
data_root="${FOURIND_DATA_ROOT:-/workspace/4ind}"
results_root="${FOURIND_RESULTS_ROOT:-/workspace/SimpleNet/4ind_result_date_mixed}"
manifest_path="${FOURIND_MANIFEST:-$results_root/fourind_manifest_date_mixed_seed0.csv}"
gpu_index="${CUDA_VISIBLE_DEVICES:-0}"
profile_dir="$output_root/profile"
samples="$profile_dir/gpu${gpu_index}_samples.csv"

cd "$repo_dir"
mkdir -p "$profile_dir"

printf 'unix_time,memory_used_mib,utilization_percent\n' > "$samples"
(
    while true; do
        printf '%s,%s\n' "$(date +%s)" \
            "$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits 2>&1 | tr -d ' ')" >> "$samples"
        sleep 15
    done
) &
sampler_pid=$!
trap 'pkill -P "$sampler_pid" 2>/dev/null || true; kill "$sampler_pid" 2>/dev/null || true' EXIT

common=(
    --data-root "$data_root"
    --results-root "$results_root"
    --manifest-path "$manifest_path"
    --category KQG27824
    --output-dir "$profile_dir"
    --image-count 100
    --warmup-images 5
    --num-workers 4
    --num-threads 4
)

run_config() {
    local name="$1"
    shift
    "$python_bin" profile_gl_inference.py "${common[@]}" --variant "$name" \
        --substages --profiler-images 0 "$@"
}

# Tile batch 32 and stride 384 come from args.json, as in revision 4.
run_config C0 --nvidia-smi-start
run_config C1 --fast-interp
run_config C2 --autocast --channels-last
run_config C3 --autocast --channels-last --fast-interp

"$python_bin" profile_gl_inference.py "${common[@]}" --tensor-check
"$python_bin" profile_gl_inference.py "${common[@]}" --s4fix-baselines

"$python_bin" profile_gl_inference.py "${common[@]}" --merge-s4fix C0,C1,C2,C3
nvidia-smi > "$profile_dir/nvidia-smi-end.txt" 2>&1
