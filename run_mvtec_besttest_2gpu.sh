#!/usr/bin/env bash
set -uo pipefail

data_path="${1:-/workspace/SimpleNet/mvtec-ad}"
result_root="${2:-/workspace/SimpleNet/results_mvtec_besttest_5060ti_320_seed0}"

gpu0_categories=(hazelnut pill wood zipper cable capsule bottle)
gpu1_categories=(screw carpet grid leather tile metal_nut transistor toothbrush)
methods=(simplenet simplenet_plus)

mkdir -p "$result_root/logs"

run_job() {
    local gpu_id="$1"
    local method="$2"
    local category="$3"
    local result_file="$result_root/$method/$category/job_result.json"
    local log_file="$result_root/logs/${method}_${category}.log"
    if [[ -f "$result_file" ]]; then
        echo "SKIP gpu=$gpu_id method=$method category=$category"
        return 0
    fi

    local augmentation="0.0"
    if [[ "$method" == "simplenet_plus" ]]; then
        augmentation="0.1"
    fi
    local attempt
    for attempt in 1 2; do
        echo "START gpu=$gpu_id method=$method category=$category attempt=$attempt time=$(date --iso-8601=seconds)"
        if python3 mvtec_besttest_benchmark.py \
            --method "$method" \
            --category "$category" \
            --data-path "$data_path" \
            --results-root "$result_root" \
            --gpu "$gpu_id" \
            --seed 0 \
            --resize 329 \
            --imagesize 320 \
            --batch-size 8 \
            --num-workers 4 \
            --meta-epochs 40 \
            --gan-epochs 4 \
            --augmentation "$augmentation" >"$log_file" 2>&1; then
            echo "DONE gpu=$gpu_id method=$method category=$category time=$(date --iso-8601=seconds)"
            flock "$result_root/.merge.lock" \
                python3 merge_mvtec_besttest_results.py "$result_root" --allow-partial \
                >>"$result_root/merge.log" 2>&1 || true
            return 0
        fi
        echo "RETRY gpu=$gpu_id method=$method category=$category attempt=$attempt time=$(date --iso-8601=seconds)"
    done
    echo "FAILED gpu=$gpu_id method=$method category=$category"
    return 1
}

run_worker() {
    local gpu_id="$1"
    shift
    local failures=0
    local method category
    for method in "${methods[@]}"; do
        for category in "$@"; do
            run_job "$gpu_id" "$method" "$category" || failures=$((failures + 1))
        done
    done
    echo "WORKER_DONE gpu=$gpu_id failures=$failures time=$(date --iso-8601=seconds)"
    return "$failures"
}

echo "$$" >"$result_root/scheduler.pid"
echo "SCHEDULER_START time=$(date --iso-8601=seconds)"
run_worker 0 "${gpu0_categories[@]}" &
pid0=$!
run_worker 1 "${gpu1_categories[@]}" &
pid1=$!

status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?

python3 merge_mvtec_besttest_results.py "$result_root" || merge_status=$?
merge_status=${merge_status:-0}
final_status=$((status0 + status1 + merge_status))
echo "SCHEDULER_DONE status=$final_status time=$(date --iso-8601=seconds)"
exit "$final_status"
