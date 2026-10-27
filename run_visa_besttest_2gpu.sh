#!/usr/bin/env bash
set -uo pipefail

data_path="${1:-/workspace/SimpleNet/visa-anomaly-detection}"
result_root="${2:-/workspace/SimpleNet/results_visa_besttest_5060ti_320_seed0}"

gpu0_categories=(candle cashew fryum macaroni1 macaroni2 pcb2)
gpu1_categories=(capsules chewinggum pcb1 pcb3 pcb4 pipe_fryum)
methods=(simplenet simplenet_plus global_local)

mkdir -p "$result_root/logs"

run_job() {
    local gpu="$1"
    local method="$2"
    local category="$3"
    local result="$result_root/$method/$category/job_result.json"
    local log="$result_root/logs/${method}_${category}.log"
    if [[ -f "$result" ]]; then
        printf 'SKIP gpu=%s method=%s category=%s\n' "$gpu" "$method" "$category"
        return 0
    fi

    local attempt
    for attempt in 1 2; do
        printf 'START gpu=%s method=%s category=%s attempt=%s time=%s\n' \
            "$gpu" "$method" "$category" "$attempt" "$(date --iso-8601=seconds)"
        if python3 visa_besttest_benchmark.py \
            --method "$method" \
            --category "$category" \
            --data-path "$data_path" \
            --results-root "$result_root" \
            --gpu "$gpu" \
            --seed 0 \
            --imagesize 320 \
            --batch-size 8 \
            --num-workers 4 \
            --meta-epochs 40 \
            --gan-epochs 4 \
            --pixel-bins 65536 >>"$log" 2>&1; then
            printf 'DONE gpu=%s method=%s category=%s time=%s\n' \
                "$gpu" "$method" "$category" "$(date --iso-8601=seconds)"
            flock "$result_root/.merge.lock" \
                python3 merge_besttest_results.py visa "$result_root" --allow-partial \
                >>"$result_root/merge.log" 2>&1 || true
            return 0
        fi
        printf 'RETRY gpu=%s method=%s category=%s attempt=%s time=%s\n' \
            "$gpu" "$method" "$category" "$attempt" "$(date --iso-8601=seconds)"
        sleep 30
    done
    printf '%s,%s,%s\n' "$gpu" "$method" "$category" \
        >>"$result_root/failures.csv"
    return 1
}

worker() {
    local gpu="$1"
    shift
    local failures=0
    local method category
    for method in "${methods[@]}"; do
        for category in "$@"; do
            run_job "$gpu" "$method" "$category" || failures=$((failures + 1))
        done
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
python3 merge_besttest_results.py visa "$result_root" || merge_status=$?
final_status=$((status0 + status1 + merge_status))
printf 'SCHEDULER_DONE status=%s time=%s\n' \
    "$final_status" "$(date --iso-8601=seconds)"
exit "$final_status"

