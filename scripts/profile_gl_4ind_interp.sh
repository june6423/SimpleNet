#!/usr/bin/env bash
# Profile the S4 interpolation modes of SimpleNet._embed on 4ind KQG27824 on
# one GPU: eight global-local configurations C0 to C7, the _embed tensor check
# of the three modes, and torch.profiler kernel tables for C0 and C1.  Usage:
#   scripts/profile_gl_4ind_interp.sh OUTPUT_ROOT
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
    "$python_bin" profile_gl_inference.py "${common[@]}" --variant "$name" --substages "$@"
}

fp16=(--autocast --channels-last)

# Tile batch 32 comes from args.json; stride 384 from args.json unless given.
run_config C0 --profiler-images 3 --nvidia-smi-start
run_config C1 --profiler-images 3 --s4-mode channels_last
run_config C2 --profiler-images 0 --s4-mode channels_last "${fp16[@]}"
run_config C3 --profiler-images 0 --s4-mode channels_last --s4-fp16 "${fp16[@]}"
run_config C4 --profiler-images 0 --s4-mode before_unfold
run_config C5 --profiler-images 0 --s4-mode before_unfold --s4-fp16 "${fp16[@]}"
run_config C6 --profiler-images 0 --s4-mode channels_last --s4-fp16 "${fp16[@]}" --tile-stride 512
run_config C7 --profiler-images 0 --s4-mode before_unfold --s4-fp16 "${fp16[@]}" --tile-stride 512

"$python_bin" profile_gl_inference.py "${common[@]}" --interp-tensor-check
"$python_bin" profile_gl_inference.py "${common[@]}" --merge-interp C0,C1,C2,C3,C4,C5,C6,C7
nvidia-smi > "$profile_dir/nvidia-smi-end.txt" 2>&1
