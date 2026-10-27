#!/usr/bin/env bash
# Train SimpleNet and SimpleNet++ on every MVTec AD category, then compute
# image/pixel F1-max from each best-I-AUROC checkpoint.
#
# Host usage:
#   ./run_mvtec_f1_multigpu.sh MVTEC_ROOT RESULTS_ROOT GPU_IDS [JOBS_PER_GPU]
#   e.g. ./run_mvtec_f1_multigpu.sh /home/vision/dongjun/mvtec-anomaly \
#            /home/vision/dongjun/SimpleNet/mvtec_f1_results 4,5,6,7 2
#
# The host side only starts one docker container (image simplenet-a6000:wandb)
# which re-executes this script with MVTEC_F1_INSIDE=1 to schedule the jobs.
# A job whose results.csv already exists is skipped, so the launcher can be
# re-run after an interruption.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- settings
METHODS=(${MVTEC_METHODS:-simplenet simplenet_plus})
# Largest training sets first so the slowest jobs start early.
CATEGORIES=(${MVTEC_CATEGORIES:-hazelnut screw carpet pill grid leather wood zipper tile cable metal_nut capsule transistor bottle toothbrush})
RESIZE="${MVTEC_RESIZE:-256}"
IMAGESIZE="${MVTEC_IMAGESIZE:-224}"
META_EPOCHS="${MVTEC_META_EPOCHS:-40}"
GAN_EPOCHS="${MVTEC_GAN_EPOCHS:-4}"
SEED="${MVTEC_SEED:-0}"
AUG_SIMPLENET="${MVTEC_AUG_SIMPLENET:-0.0}"
AUG_PLUS="${MVTEC_AUG_PLUS:-0.1}"
IMAGE="${SIMPLENET_IMAGE:-simplenet-a6000:wandb}"

# ---------------------------------------------------------------- host side
if [[ "${MVTEC_F1_INSIDE:-0}" != "1" ]]; then
    if [[ $# -lt 3 ]]; then
        printf 'Usage: %s MVTEC_ROOT RESULTS_ROOT GPU_IDS [JOBS_PER_GPU]\n' "$0" >&2
        exit 2
    fi
    mvtec_root="$(cd "$1" && pwd)"
    results_root="$2"
    gpu_ids="$3"
    jobs_per_gpu="${4:-2}"
    mkdir -p "$results_root"
    results_root="$(cd "$results_root" && pwd)"
    model_cache="${SIMPLENET_MODEL_CACHE:-${HOME}/.cache/simplenet/torch}"
    mkdir -p "$model_cache"
    exec docker run --rm --init \
        --name "${SIMPLENET_CONTAINER_NAME:-simplenet-mvtec-f1}" \
        --gpus all --shm-size "${SIMPLENET_SHM_SIZE:-32g}" \
        --workdir /workspace/SimpleNet \
        --env MVTEC_F1_INSIDE=1 \
        --env MVTEC_GPUS="$gpu_ids" \
        --env MVTEC_JOBS_PER_GPU="$jobs_per_gpu" \
        --env MVTEC_METHODS="${METHODS[*]}" \
        --env MVTEC_CATEGORIES="${CATEGORIES[*]}" \
        --env MVTEC_RESIZE="$RESIZE" --env MVTEC_IMAGESIZE="$IMAGESIZE" \
        --env MVTEC_META_EPOCHS="$META_EPOCHS" --env MVTEC_GAN_EPOCHS="$GAN_EPOCHS" \
        --env MVTEC_SEED="$SEED" \
        --env MVTEC_AUG_SIMPLENET="$AUG_SIMPLENET" --env MVTEC_AUG_PLUS="$AUG_PLUS" \
        --env WANDB_MODE=disabled --env PYTHONUNBUFFERED=1 \
        --volume "${script_dir}:/workspace/SimpleNet:rw" \
        --volume "${mvtec_root}:/data/mvtec:ro" \
        --volume "${results_root}:/results:rw" \
        --volume "${model_cache}:/root/.cache/torch:rw" \
        "$IMAGE" bash /workspace/SimpleNet/run_mvtec_f1_multigpu.sh
fi

# ------------------------------------------------------------ container side
data_root=/data/mvtec
results_root=/results
IFS=',' read -r -a gpus <<< "$MVTEC_GPUS"
jobs_per_gpu="${MVTEC_JOBS_PER_GPU:-2}"
mkdir -p "$results_root/logs"

run_job() {
    local gpu="$1" method="$2" category="$3"
    local out="$results_root/mvtec/$method/$category"
    local log="$results_root/logs/${method}_${category}.log"
    local entry aug
    if [[ "$method" == "simplenet" ]]; then entry=main.py; aug="$AUG_SIMPLENET"; else entry=main_ours.py; aug="$AUG_PLUS"; fi
    local aug_flags=()
    if [[ "$aug" != "0" && "$aug" != "0.0" ]]; then
        aug_flags=(--augment --brightness "$aug" --contrast "$aug" --saturation "$aug")
    fi

    if [[ -f "$out/results.csv" && -f "$out/models/0/mvtec_${category}/ckpt.pth" ]]; then
        printf 'SKIP_TRAIN gpu=%s %s/%s\n' "$gpu" "$method" "$category"
    else
        printf 'START gpu=%s %s/%s %s\n' "$gpu" "$method" "$category" "$(date --iso-8601=seconds)"
        rm -rf "$out"
        if ! python3 "$entry" \
            --gpu "$gpu" --seed "$SEED" \
            --log_project mvtec --log_group "$method" --run_name "$category" \
            --results_path "$results_root" \
            net -b wideresnet50 -le layer2 -le layer3 \
            --pretrain_embed_dimension 1536 --target_embed_dimension 1536 \
            --patchsize 3 --meta_epochs "$META_EPOCHS" --embedding_size 256 \
            --gan_epochs "$GAN_EPOCHS" --noise_std 0.015 --dsc_hidden 1024 \
            --dsc_layers 2 --dsc_margin .5 --pre_proj 1 \
            dataset --batch_size 8 --resize "$RESIZE" --imagesize "$IMAGESIZE" \
            "${aug_flags[@]}" -d "$category" mvtec "$data_root" > "$log" 2>&1; then
            printf 'TRAIN_FAILED gpu=%s %s/%s log=%s\n' "$gpu" "$method" "$category" "$log"
            return 1
        fi
        printf 'TRAIN_DONE gpu=%s %s/%s %s\n' "$gpu" "$method" "$category" "$(date --iso-8601=seconds)"
    fi

    if [[ -f "$out/f1_eval.json" ]]; then
        printf 'SKIP_EVAL %s/%s\n' "$method" "$category"
        return 0
    fi
    if ! python3 mvtec_f1_eval.py eval --method "$method" --category "$category" \
        --results-root "$results_root" --data-root "$data_root" --gpu "$gpu" \
        --resize "$RESIZE" --imagesize "$IMAGESIZE" \
        --meta-epochs "$META_EPOCHS" --gan-epochs "$GAN_EPOCHS" >> "$log" 2>&1; then
        printf 'EVAL_FAILED %s/%s log=%s\n' "$method" "$category" "$log"
        return 1
    fi
    tail -1 "$log"
}

# Build the job list and hand it out round-robin to (gpu, slot) workers.
jobs=()
for category in "${CATEGORIES[@]}"; do
    for method in "${METHODS[@]}"; do
        jobs+=("$method:$category")
    done
done
slots=$(( ${#gpus[@]} * jobs_per_gpu ))
printf 'MVTEC_F1 jobs=%d gpus=%s jobs_per_gpu=%s slots=%d start=%s\n' \
    "${#jobs[@]}" "$MVTEC_GPUS" "$jobs_per_gpu" "$slots" "$(date --iso-8601=seconds)"

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

python3 mvtec_f1_eval.py merge --results-root "$results_root"
printf 'MVTEC_F1 finished failures=%d end=%s\n' "$total_failures" "$(date --iso-8601=seconds)"
exit "$(( total_failures > 0 ))"
