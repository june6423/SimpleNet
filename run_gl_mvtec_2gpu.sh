#!/usr/bin/env bash
set -euo pipefail

data_path="${1:-/workspace/SimpleNet/mvtec-ad}"
result_root="${2:-/workspace/SimpleNet/results_global_local/mvtec_320_aug01_seed0_2gpu}"

gpu0_categories=(hazelnut pill wood zipper cable capsule bottle)
gpu1_categories=(screw carpet grid leather tile metal_nut transistor toothbrush)

mkdir -p "$result_root"

run_group() {
    local gpu_id="$1"
    local run_name="$2"
    shift 2
    python3 main_gl.py \
        --dataset mvtec \
        --data-path "$data_path" \
        --categories "$@" \
        --results-path "$result_root" \
        --run-name "$run_name" \
        --gpu "$gpu_id" \
        --seed 0 \
        --batch-size 8 \
        --num-workers 4 \
        --resize 329 \
        --imagesize 320 \
        --meta-epochs 40 \
        --gan-epochs 4 \
        --brightness 0.1 \
        --contrast 0.1 \
        --saturation 0.1
}

echo "Starting GPU 0: ${gpu0_categories[*]}"
run_group 0 gpu0 "${gpu0_categories[@]}" >"$result_root/gpu0.log" 2>&1 &
pid0=$!

echo "Starting GPU 1: ${gpu1_categories[*]}"
run_group 1 gpu1 "${gpu1_categories[@]}" >"$result_root/gpu1.log" 2>&1 &
pid1=$!

cleanup() {
    kill "$pid0" "$pid1" 2>/dev/null || true
}
trap cleanup INT TERM

status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
trap - INT TERM

if (( status0 != 0 || status1 != 0 )); then
    echo "Parallel run failed: gpu0=$status0 gpu1=$status1" >&2
    exit 1
fi

python3 merge_gl_results.py \
    --input "$result_root/gpu0/results.csv" \
    --input "$result_root/gpu1/results.csv" \
    --output "$result_root/results.csv" \
    --summary "$result_root/summary.json"

echo "Stored merged results in $result_root/results.csv"
