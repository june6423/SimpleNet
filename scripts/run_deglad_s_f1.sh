#!/usr/bin/env bash
# Harness entrypoint: DeGLAD-S evaluation of the 27 global_local 320
# checkpoints (15 MVTec, 12 ViSA) inside the persistent container, with the
# standard-path timing per category.  Evaluation only, no training, no docker
# run.
# Usage: scripts/run_deglad_s_f1.sh OUTPUT_ROOT
# Results go to OUTPUT_ROOT/<dataset>/deglad_s/<category>/f1_eval.json,
# OUTPUT_ROOT/deglad_s_f1_summary.csv, and OUTPUT_ROOT/deglad_s_timing.csv.
# /proc/loadavg and nvidia-smi go to OUTPUT_ROOT/logs/monitor.log at the start,
# every 30 minutes, and at the end.
set -euo pipefail

output_root="${1:?Usage: $0 OUTPUT_ROOT}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
monitor_log="$output_root/logs/monitor.log"
mvtec_data_root="${MVTEC_DATA_ROOT:-/workspace/mvtec-anomaly}"
mvtec_checkpoint_root="${MVTEC_CHECKPOINT_ROOT:-/workspace/mvtec_320_checkpoints}"
visa_data_root="${VISA_DATA_ROOT:-/workspace/visa-anomaly}"
visa_checkpoint_root="${VISA_CHECKPOINT_ROOT:-/workspace/visa_320_checkpoints}"
reference_dir="references/deglad_s_f1_001"
IFS=',' read -r -a gpus <<< "${DEGLAD_S_GPUS:-0,1,2,3,4,5,6}"
jobs_per_gpu="${DEGLAD_S_JOBS_PER_GPU:-2}"
threads="${DEGLAD_S_THREADS_PER_JOB:-4}"
mvtec_categories=(hazelnut screw carpet pill grid leather wood zipper tile cable metal_nut capsule transistor bottle toothbrush)
visa_categories=(pcb1 pcb2 pcb3 pcb4 macaroni1 macaroni2 candle capsules chewinggum cashew fryum pipe_fryum)

cd "$repo_dir"
mkdir -p "$output_root/logs"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
export PYTHONUNBUFFERED=1

snapshot() {
    {
        printf '=== %s %s\n' "$1" "$(date --iso-8601=seconds)"
        cat /proc/loadavg
        nvidia-smi || true
    } >> "$monitor_log" 2>&1
}

snapshot start
( while sleep 1800; do snapshot periodic; done ) &
monitor_pid=$!
trap 'pkill -P "$monitor_pid" 2>/dev/null || true; kill "$monitor_pid" 2>/dev/null || true; snapshot end' EXIT

# Load the WRN50 weights once so that the parallel jobs do not download the
# same file at the same time.
python3 -c 'import backbones; backbones.load("wideresnet50")'

jobs=()
for category in "${mvtec_categories[@]}"; do
    jobs+=("mvtec:$category")
done
for category in "${visa_categories[@]}"; do
    jobs+=("visa:$category")
done
slots=$(( ${#gpus[@]} * jobs_per_gpu ))
printf 'DEGLAD_S_F1 jobs=%d gpus=%s jobs_per_gpu=%s slots=%d threads_per_job=%s start=%s\n' \
    "${#jobs[@]}" "${gpus[*]}" "$jobs_per_gpu" "$slots" "$threads" "$(date --iso-8601=seconds)"

run_job() {
    local gpu="$1" dataset="$2" category="$3"
    local log="$output_root/logs/${dataset}_${category}.log"
    local dataset_args
    if [[ -f "$output_root/$dataset/deglad_s/$category/f1_eval.json" ]]; then
        printf 'SKIP_EVAL %s/%s\n' "$dataset" "$category"
        return 0
    fi
    if [[ "$dataset" == mvtec ]]; then
        dataset_args=(--checkpoint-root "$mvtec_checkpoint_root" --data-root "$mvtec_data_root"
            --resize 329 --imagesize 320
            --reference-summary "$reference_dir/mvtec_f1_summary_320.csv")
    else
        dataset_args=(--checkpoint-root "$visa_checkpoint_root" --data-root "$visa_data_root"
            --reference-summary "$reference_dir/visa_f1_summary.csv")
    fi
    printf 'START gpu=%s %s/%s %s\n' "$gpu" "$dataset" "$category" "$(date --iso-8601=seconds)"
    if ! python3 visa_f1_eval_320.py eval --dataset "$dataset" --method global_local \
        --category "$category" --config deglad-s "${dataset_args[@]}" \
        --results-root "$output_root" --gpu "$gpu" --num-workers 4 >> "$log" 2>&1; then
        printf 'EVAL_FAILED gpu=%s %s/%s log=%s\n' "$gpu" "$dataset" "$category" "$log"
        return 1
    fi
    printf 'EVAL_DONE gpu=%s %s/%s %s\n' "$gpu" "$dataset" "$category" "$(date --iso-8601=seconds)"
    tail -1 "$log"
}

worker() {
    local slot="$1" gpu="$2" failures=0
    for (( i = slot; i < ${#jobs[@]}; i += slots )); do
        local job="${jobs[$i]}"
        run_job "$gpu" "${job%%:*}" "${job#*:}" || failures=$((failures + 1))
    done
    return "$failures"
}

pids=()
for (( s = 0; s < slots; s++ )); do
    worker "$s" "${gpus[$(( s % ${#gpus[@]} ))]}" &
    pids+=($!)
done
total_failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || total_failures=$((total_failures + $?))
done

python3 visa_f1_eval_320.py merge --config deglad-s --results-root "$output_root"
printf 'DEGLAD_S_F1 finished failures=%d end=%s\n' "$total_failures" "$(date --iso-8601=seconds)"
exit "$(( total_failures > 0 ))"
