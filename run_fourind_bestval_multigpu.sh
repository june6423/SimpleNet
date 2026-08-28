#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
data_root="${1:-${FOURIND_DATA_ROOT:-}}"
results_root="${2:-${script_dir}/results_fourind_bestval}"
gpu_csv="${3:-0,1,2}"
split_strategy="${FOURIND_SPLIT_STRATEGY:-chronological}"

if [[ -z "$data_root" ]]; then
    printf 'Usage: %s DATA_ROOT [RESULTS_ROOT] [GPU_CSV]\n' "$0" >&2
    printf 'Example: %s /datasets/4ind_202608 /results/4ind 0,1,2\n' "$0" >&2
    exit 2
fi
if [[ ! -d "$data_root" ]]; then
    printf 'Dataset root does not exist: %s\n' "$data_root" >&2
    exit 2
fi

IFS=',' read -r -a gpus <<<"$gpu_csv"
if [[ ${#gpus[@]} -ne 3 ]]; then
    printf 'Exactly three GPU ids are required for the 3-GPU x 2-step schedule.\n' >&2
    exit 2
fi
for gpu in "${gpus[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        printf 'Invalid GPU id: %s\n' "$gpu" >&2
        exit 2
    fi
done
if [[ "$split_strategy" != "chronological" \
    && "$split_strategy" != "date-mixed" ]]; then
    printf 'Invalid FOURIND_SPLIT_STRATEGY: %s\n' "$split_strategy" >&2
    exit 2
fi

mkdir -p "$results_root/logs"
split_tag="${split_strategy//-/_}"
manifest_path="$results_root/fourind_manifest_${split_tag}_seed0.csv"
manifest_summary="$results_root/fourind_manifest_${split_tag}_seed0.summary.json"

if [[ ! -f "$manifest_path" ]]; then
    printf 'MANIFEST_START root=%s time=%s\n' \
        "$data_root" "$(date --iso-8601=seconds)"
    if ! "$python_bin" "$script_dir/fourind_manifest.py" \
        --data-root "$data_root" \
        --output "$manifest_path" \
        --summary-path "$manifest_summary" \
        --train-fraction 0.1 \
        --seed 0 \
        --split-strategy "$split_strategy" \
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

# Two explicit steps: all three methods for one product run in parallel.
# Keeping each method on the same GPU across steps makes the schedule easy to
# audit while preserving every training and evaluation argument below.
jobs=(
    global_local:KQG27542
    simplenet:KQG27542
    simplenet_plus:KQG27542
    global_local:KQG27824
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

printf '%s\n' "$$" >"$results_root/scheduler.pid"
printf 'SCHEDULER_START pid=%s gpus=%s jobs=%s steps=2 time=%s\n' \
    "$$" "$gpu_csv" "${#jobs[@]}" "$(date --iso-8601=seconds)"
printf 'SPLIT_STRATEGY=%s manifest=%s\n' "$split_strategy" "$manifest_path"

worker_failures=0
for step in 0 1; do
    printf 'STEP_START step=%s category=%s time=%s\n' \
        "$((step + 1))" "${jobs[$((step * 3))]#*:}" "$(date --iso-8601=seconds)"
    pids=()
    for gpu_index in 0 1 2; do
        job="${jobs[$((step * 3 + gpu_index))]}"
        method="${job%%:*}"
        category="${job#*:}"
        run_job "${gpus[$gpu_index]}" "$method" "$category" &
        pids+=("$!")
    done

    step_failures=0
    for pid in "${pids[@]}"; do
        status=0
        wait "$pid" || status=$?
        step_failures=$((step_failures + status))
    done
    worker_failures=$((worker_failures + step_failures))
    printf 'STEP_DONE step=%s failures=%s time=%s\n' \
        "$((step + 1))" "$step_failures" "$(date --iso-8601=seconds)"
done

merge_status=0
"$python_bin" "$script_dir/merge_fourind_results.py" "$results_root" \
    || merge_status=$?
final_status=$((worker_failures + merge_status))
printf 'SCHEDULER_DONE status=%s time=%s\n' \
    "$final_status" "$(date --iso-8601=seconds)"
exit "$final_status"
