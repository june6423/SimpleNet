#!/usr/bin/env bash
# Harness entrypoint: F1-max evaluation of the 36 VisA 320 best-test
# checkpoints (simplenet, simplenet_plus, global_local x 12 categories) inside
# the persistent container.  Evaluation only, no training, no docker run.
# Usage: scripts/run_visa_f1_320.sh OUTPUT_ROOT
# Results go to OUTPUT_ROOT/visa/<method>/<category>/f1_eval.json and
# OUTPUT_ROOT/visa_f1_summary.csv.  /proc/loadavg and nvidia-smi go to
# OUTPUT_ROOT/logs/monitor.log at the start, every 30 minutes, and at the end.
set -euo pipefail

output_root="${1:?Usage: $0 OUTPUT_ROOT}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
monitor_log="$output_root/logs/monitor.log"
data_root="${VISA_DATA_ROOT:-/workspace/visa-anomaly}"
checkpoint_root="${VISA_CHECKPOINT_ROOT:-/workspace/visa_320_checkpoints}"
IFS=',' read -r -a gpus <<< "${VISA_GPUS:-0,1,2,3,4,5,6}"
jobs_per_gpu="${VISA_JOBS_PER_GPU:-2}"
threads="${VISA_THREADS_PER_JOB:-4}"
methods=(simplenet simplenet_plus global_local)
categories=(pcb1 pcb2 pcb3 pcb4 macaroni1 macaroni2 candle capsules chewinggum cashew fryum pipe_fryum)

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
for category in "${categories[@]}"; do
    for method in "${methods[@]}"; do
        jobs+=("$method:$category")
    done
done
slots=$(( ${#gpus[@]} * jobs_per_gpu ))
printf 'VISA_F1_320 jobs=%d gpus=%s jobs_per_gpu=%s slots=%d threads_per_job=%s start=%s\n' \
    "${#jobs[@]}" "${gpus[*]}" "$jobs_per_gpu" "$slots" "$threads" "$(date --iso-8601=seconds)"

run_job() {
    local gpu="$1" method="$2" category="$3"
    local log="$output_root/logs/${method}_${category}.log"
    if [[ -f "$output_root/visa/$method/$category/f1_eval.json" ]]; then
        printf 'SKIP_EVAL %s/%s\n' "$method" "$category"
        return 0
    fi
    printf 'START gpu=%s %s/%s %s\n' "$gpu" "$method" "$category" "$(date --iso-8601=seconds)"
    if ! python3 visa_f1_eval_320.py eval --dataset visa --method "$method" --category "$category" \
        --checkpoint-root "$checkpoint_root" --data-root "$data_root" \
        --results-root "$output_root" --gpu "$gpu" --num-workers 4 >> "$log" 2>&1; then
        printf 'EVAL_FAILED gpu=%s %s/%s log=%s\n' "$gpu" "$method" "$category" "$log"
        return 1
    fi
    printf 'EVAL_DONE gpu=%s %s/%s %s\n' "$gpu" "$method" "$category" "$(date --iso-8601=seconds)"
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

python3 visa_f1_eval_320.py merge --dataset visa --results-root "$output_root"
printf 'VISA_F1_320 finished failures=%d end=%s\n' "$total_failures" "$(date --iso-8601=seconds)"
exit "$(( total_failures > 0 ))"
