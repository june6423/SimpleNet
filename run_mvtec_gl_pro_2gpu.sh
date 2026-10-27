#!/usr/bin/env bash
set -uo pipefail

data_path="${1:-/workspace/SimpleNet/mvtec-ad}"
source_root="${2:-/workspace/SimpleNet/results_mvtec_gl_besttest_5060ti_320_seed0}"
result_root="${3:-/workspace/SimpleNet/results_mvtec_gl_besttest_pro_5060ti_320_seed0}"

gpu0_categories=(hazelnut pill wood zipper cable capsule bottle)
gpu1_categories=(screw carpet grid leather tile metal_nut transistor toothbrush)

mkdir -p "$result_root/logs"

run_job() {
    local gpu="$1"
    local category="$2"
    local result="$result_root/$category/job_result.json"
    local log="$result_root/logs/${category}.log"
    if [[ -f "$result" ]]; then
        printf 'SKIP gpu=%s category=%s\n' "$gpu" "$category"
        return 0
    fi
    printf 'START gpu=%s category=%s time=%s\n' \
        "$gpu" "$category" "$(date --iso-8601=seconds)"
    if python3 mvtec_gl_pro_benchmark.py \
        --category "$category" \
        --data-path "$data_path" \
        --source-root "$source_root" \
        --results-root "$result_root" \
        --gpu "$gpu" \
        --seed 0 \
        --resize 329 \
        --imagesize 320 \
        --num-workers 4 \
        --meta-epochs 40 \
        --gan-epochs 4 \
        --augmentation 0.1 \
        --num-thresholds 200 >>"$log" 2>&1; then
        printf 'DONE gpu=%s category=%s time=%s\n' \
            "$gpu" "$category" "$(date --iso-8601=seconds)"
        flock "$result_root/.merge.lock" \
            python3 merge_mvtec_gl_pro_results.py "$result_root" --allow-partial \
            >>"$result_root/merge.log" 2>&1 || true
        return 0
    fi
    printf 'FAILED gpu=%s category=%s time=%s\n' \
        "$gpu" "$category" "$(date --iso-8601=seconds)"
    printf '%s,%s\n' "$gpu" "$category" >>"$result_root/failures.csv"
    return 1
}

worker() {
    local gpu="$1"
    shift
    local failures=0
    local category
    for category in "$@"; do
        run_job "$gpu" "$category" || failures=$((failures + 1))
    done
    printf 'WORKER_DONE gpu=%s failures=%s time=%s\n' \
        "$gpu" "$failures" "$(date --iso-8601=seconds)"
    return "$failures"
}

printf '%s\n' "$$" >"$result_root/scheduler.pid"
printf 'SCHEDULER_START pid=%s time=%s\n' "$$" "$(date --iso-8601=seconds)"
worker 0 "${gpu0_categories[@]}" &
pid0=$!
worker 1 "${gpu1_categories[@]}" &
pid1=$!

status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
merge_status=0
python3 merge_mvtec_gl_pro_results.py "$result_root" || merge_status=$?
final_status=$((status0 + status1 + merge_status))
printf 'SCHEDULER_DONE status=%s time=%s\n' \
    "$final_status" "$(date --iso-8601=seconds)"
exit "$final_status"
