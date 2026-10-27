#!/usr/bin/env bash
# Time SimpleNet (boundaries A and B), SimpleNet++ (A and B), DeGLAD, and
# DeGLAD-S at batch size 1 on one GPU for 4ind KQG27824, MVTec bottle, carpet,
# screw, and ViSA candle, macaroni1, pcb4.  Usage:
#   scripts/run_speed_compare.sh OUTPUT_ROOT
# Outputs are written to OUTPUT_ROOT/compare/.  The memory use and the
# utilization of the visible GPU are sampled every 15 seconds for the whole run
# to OUTPUT_ROOT/compare/gpu<index>_samples.csv.
set -euo pipefail

output_root="${1:?Usage: $0 OUTPUT_ROOT}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
python_bin="${PYTHON_BIN:-python3}"
fourind_data_root="${FOURIND_DATA_ROOT:-/workspace/4ind}"
fourind_results_root="${FOURIND_RESULTS_ROOT:-/workspace/SimpleNet/4ind_result_date_mixed}"
fourind_manifest="${FOURIND_MANIFEST:-$fourind_results_root/fourind_manifest_date_mixed_seed0.csv}"
mvtec_data_root="${MVTEC_DATA_ROOT:-/workspace/mvtec-anomaly}"
mvtec_checkpoint_root="${MVTEC_CHECKPOINT_ROOT:-/workspace/mvtec_320_checkpoints}"
visa_data_root="${VISA_DATA_ROOT:-/workspace/visa-anomaly}"
visa_checkpoint_root="${VISA_CHECKPOINT_ROOT:-/workspace/visa_320_checkpoints}"
gpu_index="${CUDA_VISIBLE_DEVICES:-0}"
compare_dir="$output_root/compare"
samples="$compare_dir/gpu${gpu_index}_samples.csv"

cd "$repo_dir"
mkdir -p "$compare_dir"
nvidia-smi > "$compare_dir/nvidia-smi-start.txt" 2>&1

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

run_pair() {
    "$python_bin" speed_compare.py run --output-dir "$compare_dir" --image-count 100 \
        --num-workers 4 --num-threads 4 "$@"
}

run_pair --dataset 4ind --category KQG27824 --checkpoint-root "$fourind_results_root" \
    --data-root "$fourind_data_root" --manifest-path "$fourind_manifest"
for category in bottle carpet screw; do
    run_pair --dataset mvtec --category "$category" --checkpoint-root "$mvtec_checkpoint_root" \
        --data-root "$mvtec_data_root" --resize 329 --imagesize 320
done
for category in candle macaroni1 pcb4; do
    run_pair --dataset visa --category "$category" --checkpoint-root "$visa_checkpoint_root" \
        --data-root "$visa_data_root" --imagesize 320
done

"$python_bin" speed_compare.py merge --output-dir "$compare_dir"
nvidia-smi > "$compare_dir/nvidia-smi-end.txt" 2>&1
