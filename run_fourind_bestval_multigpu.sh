#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
data_root="${1:-${FOURIND_DATA_ROOT:-}}"
results_root="${2:-${script_dir}/results_fourind_bestval}"
gpu_csv="${3:-0,1}"

if [[ -z "$data_root" ]]; then
    printf 'Usage: %s DATA_ROOT [RESULTS_ROOT] [GPU_CSV]\n' "$0" >&2
    printf 'Example: %s /datasets/4ind_202608 /results/4ind 0,1\n' "$0" >&2
    exit 2
fi
if [[ ! -d "$data_root" ]]; then
    printf 'Dataset root does not exist: %s\n' "$data_root" >&2
    exit 2
fi

IFS=',' read -r -a gpus <<<"$gpu_csv"
if [[ ${#gpus[@]} -lt 1 ]]; then
    printf 'At least one GPU id is required.\n' >&2
    exit 2
fi
for gpu in "${gpus[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        printf 'Invalid GPU id: %s\n' "$gpu" >&2
        exit 2
    fi
done

mkdir -p "$results_root/logs"
manifest_path="$results_root/fourind_manifest_seed0.csv"
manifest_summary="$results_root/fourind_manifest_seed0.summary.json"

if [[ ! -f "$manifest_path" ]]; then
    printf 'MANIFEST_START root=%s time=%s\n' \
        "$data_root" "$(date --iso-8601=seconds)"
    if ! "$python_bin" "$script_dir/fourind_manifest.py" \
        --data-root "$data_root" \
        --output "$manifest_path" \
        --summary-path "$manifest_summary" \
        --train-fraction 0.1 \
        --seed 0 \
        >"$results_root/manifest.log" 2>&1; then
        printf 'MANIFEST_FAILED log=%s\n' "$results_root/manifest.log" >&2
        exit 1
    fi
    if [[ ! -f "$manifest_path" || ! -f "$manifest_summary" ]]; then
        printf 'MANIFEST_OUTPUT_MISSING manifest=%s summary=%s\n' \
            "$manifest_path" "$manifest_summary" >&2
        exit 1
    fi
    printf 'MANIFEST_DONE manifest=%s time=%s\n' \
        "$manifest_path" "$(date --iso-8601=seconds)"
else
    printf 'MANIFEST_REUSE manifest=%s\n' "$manifest_path"
fi

# Start the two longest high-resolution jobs first.  Workers claim the next
# item only after their current method/product job has fully finished.
jobs=(
    global_local:KQG27542
    global_local:KQG27824
    simplenet:KQG27542
    simplenet_plus:KQG27542
    simplenet:KQG27824
    simplenet_plus:KQG27824
)

run_job() {
    local gpu="$1"
    local method="$2"
    local category="$3"
    local result="$results_root/$method/$category/job_result.json"
    local log="$results_root/logs/${method}_${category}.log"
    if [[ -f "$result" ]]; then
        printf 'SKIP gpu=%s method=%s category=%s\n' "$gpu" "$method" "$category"
        return 0
    fi

    local attempt
    for attempt in 1 2; do
        printf 'START gpu=%s method=%s category=%s attempt=%s time=%s\n' \
            "$gpu" "$method" "$category" "$attempt" "$(date --iso-8601=seconds)"
        if "$python_bin" "$script_dir/fourind_bestval_benchmark.py" \
            --method "$method" \
            --category "$category" \
            --data-root "$data_root" \
            --manifest-path "$manifest_path" \
            --results-root "$results_root" \
            --gpu "$gpu" \
            --seed 0 \
            --imagesize 320 \
            --batch-size 8 \
            --num-workers 4 \
            --meta-epochs 40 \
            --gan-epochs 4 \
            --augmentation 0.1 \
            --global-height 128 \
            --global-width 2048 \
            --tile-height 512 \
            --tile-width 512 \
            --tile-stride-y 384 \
            --tile-stride-x 384 \
            --tile-batch-size 32 \
            >>"$log" 2>&1; then
            if [[ ! -f "$result" ]]; then
                printf 'OUTPUT_MISSING gpu=%s method=%s category=%s result=%s\n' \
                    "$gpu" "$method" "$category" "$result" >&2
                continue
            fi
            printf 'DONE gpu=%s method=%s category=%s time=%s\n' \
                "$gpu" "$method" "$category" "$(date --iso-8601=seconds)"
            flock "$results_root/.merge.lock" \
                "$python_bin" "$script_dir/merge_fourind_results.py" \
                "$results_root" --allow-partial \
                >>"$results_root/merge.log" 2>&1 || true
            return 0
        fi
        printf 'RETRY gpu=%s method=%s category=%s attempt=%s time=%s\n' \
            "$gpu" "$method" "$category" "$attempt" "$(date --iso-8601=seconds)"
        sleep 30
    done
    printf '%s,%s,%s\n' "$gpu" "$method" "$category" \
        >>"$results_root/failures.csv"
    return 1
}

queue_index_path="$results_root/.queue_index"
queue_lock_path="$results_root/.queue.lock"
printf '0\n' >"$queue_index_path"

claim_job() {
    local claimed_index
    exec 9>"$queue_lock_path"
    flock 9
    read -r claimed_index <"$queue_index_path"
    if ((claimed_index >= ${#jobs[@]})); then
        flock -u 9
        exec 9>&-
        return 1
    fi
    CLAIMED_JOB="${jobs[$claimed_index]}"
    printf '%s\n' "$((claimed_index + 1))" >"$queue_index_path"
    flock -u 9
    exec 9>&-
    return 0
}

worker() {
    local gpu="$1"
    local failures=0
    local job method category
    while claim_job; do
        job="$CLAIMED_JOB"
        method="${job%%:*}"
        category="${job#*:}"
        run_job "$gpu" "$method" "$category" || failures=$((failures + 1))
    done
    printf 'WORKER_DONE gpu=%s failures=%s time=%s\n' \
        "$gpu" "$failures" "$(date --iso-8601=seconds)"
    return "$failures"
}

printf '%s\n' "$$" >"$results_root/scheduler.pid"
printf 'SCHEDULER_START pid=%s gpus=%s jobs=%s time=%s\n' \
    "$$" "$gpu_csv" "${#jobs[@]}" "$(date --iso-8601=seconds)"

pids=()
for index in "${!gpus[@]}"; do
    worker "${gpus[$index]}" &
    pids+=("$!")
done

worker_failures=0
for pid in "${pids[@]}"; do
    status=0
    wait "$pid" || status=$?
    worker_failures=$((worker_failures + status))
done

merge_status=0
"$python_bin" "$script_dir/merge_fourind_results.py" "$results_root" \
    || merge_status=$?
final_status=$((worker_failures + merge_status))
printf 'SCHEDULER_DONE status=%s time=%s\n' \
    "$final_status" "$(date --iso-8601=seconds)"
exit "$final_status"
