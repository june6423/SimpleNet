#!/usr/bin/env bash
set -uo pipefail

data_path="/workspace/SimpleNet/visa-anomaly-detection"
results_root="/workspace/SimpleNet/results_visa_weekend_5060ti"
mkdir -p "$results_root/logs"

# The split is balanced by official training-image counts (4501 vs 4158).
gpu0_categories=(candle cashew fryum macaroni1 macaroni2 pcb2)
gpu1_categories=(capsules chewinggum pcb1 pcb3 pcb4 pipe_fryum)
methods=(simplenet simplenet_plus global_local)

run_job() {
    local gpu="$1"
    local method="$2"
    local category="$3"
    local result="$results_root/$method/$category/job_result.json"
    local log="$results_root/logs/${method}_${category}.log"
    if [[ -f "$result" ]]; then
        printf 'SKIP completed gpu=%s method=%s category=%s\n' "$gpu" "$method" "$category"
        return 0
    fi

    for attempt in 1 2; do
        printf 'START gpu=%s method=%s category=%s attempt=%s time=%s\n' \
            "$gpu" "$method" "$category" "$attempt" "$(date --iso-8601=seconds)"
        if python3 visa_benchmark.py \
            --method "$method" \
            --category "$category" \
            --data-path "$data_path" \
            --results-root "$results_root" \
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
            python3 merge_visa_results.py "$results_root" --allow-partial \
                >>"$results_root/merge.log" 2>&1 || true
            return 0
        fi
        printf 'FAIL gpu=%s method=%s category=%s attempt=%s time=%s\n' \
            "$gpu" "$method" "$category" "$attempt" "$(date --iso-8601=seconds)"
        sleep 30
    done
    printf '%s,%s,%s\n' "$gpu" "$method" "$category" \
        >>"$results_root/failures.csv"
    return 1
}

worker() {
    local gpu="$1"
    shift
    local categories=("$@")
    local failures=0
    for method in "${methods[@]}"; do
        for category in "${categories[@]}"; do
            run_job "$gpu" "$method" "$category" || failures=$((failures + 1))
        done
    done
    printf 'WORKER_DONE gpu=%s failures=%s time=%s\n' \
        "$gpu" "$failures" "$(date --iso-8601=seconds)"
    return "$failures"
}

printf '%s\n' "$$" >"$results_root/scheduler.pid"
printf 'SCHEDULER_START pid=%s time=%s\n' "$$" "$(date --iso-8601=seconds)"
worker 0 "${gpu0_categories[@]}" &
pid0=$!
worker 1 "${gpu1_categories[@]}" &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
python3 merge_visa_results.py "$results_root" || status=1
printf 'SCHEDULER_DONE status=%s time=%s\n' "$status" "$(date --iso-8601=seconds)"
exit "$status"
