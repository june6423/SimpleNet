#!/usr/bin/env bash
# Profile six global-local inference variants and the two baselines (batch
# size 1) on 4ind KQG27824 on one GPU.  Usage:
#   scripts/profile_gl_4ind_variants.sh OUTPUT_ROOT
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

printf 'timestamp,memory.used,utilization.gpu\n' > "$samples"
(
    while true; do
        printf '%s,%s\n' "$(date --iso-8601=seconds)" \
            "$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader 2>&1)" >> "$samples"
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

run_variant() {
    local name="$1"
    shift
    "$python_bin" profile_gl_inference.py "${common[@]}" --variant "$name" "$@"
}

# V0 is the revision 2 configuration: tile batch 32 and stride 384 from args.json.
run_variant V0 --nvidia-smi-start
run_variant V1 --autocast --channels-last
run_variant V2 --tile-batch-size 129
run_variant V3 --tile-stride 512
run_variant V12 --autocast --channels-last --tile-batch-size 129
run_variant V123 --autocast --channels-last --tile-batch-size 64 --tile-stride 512

# Baselines at batch size 1; this process also writes nvidia-smi-end.txt.
"$python_bin" profile_gl_inference.py "${common[@]}" --baselines-only --baseline-batch-sizes 1

"$python_bin" profile_gl_inference.py "${common[@]}" --merge-variants V0,V1,V2,V3,V12,V123
