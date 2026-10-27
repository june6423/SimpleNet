#!/usr/bin/env bash
# Train SimpleNet and SimpleNet++ on every VisA category, then compute
# image/pixel F1-max from each best-I-AUROC checkpoint.  A copy of
# run_mvtec_f1_multigpu.sh for VisA with the MVTec input geometry
# (datasets/visa_mvtecstyle.py) and a thread limit for each job.
#
# Host usage:
#   ./run_visa_f1_multigpu.sh VISA_ROOT RESULTS_ROOT GPU_IDS [JOBS_PER_GPU]
#   e.g. ./run_visa_f1_multigpu.sh /home/vision/dongjun/visa-anomaly \
#            /home/vision/dongjun/SimpleNet/visa_f1_results 0,1,2,3,4,5,6 4
#
# The host side only starts one docker container (image simplenet-a6000:wandb)
# which re-executes this script with VISA_F1_INSIDE=1 to schedule the jobs.
# Inside an existing container, set VISA_F1_INSIDE=1, VISA_GPUS,
# VISA_DATA_ROOT, and VISA_RESULTS_ROOT and run this script from the
# repository root.  A job whose results.csv already exists is skipped, so the
# launcher can be re-run after an interruption.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- settings
METHODS=(${VISA_METHODS:-simplenet simplenet_plus})
# Largest training sets first so the slowest jobs start early.
CATEGORIES=(${VISA_CATEGORIES:-pcb1 pcb2 pcb3 pcb4 macaroni1 macaroni2 candle capsules chewinggum cashew fryum pipe_fryum})
RESIZE="${VISA_RESIZE:-256}"
IMAGESIZE="${VISA_IMAGESIZE:-224}"
META_EPOCHS="${VISA_META_EPOCHS:-40}"
GAN_EPOCHS="${VISA_GAN_EPOCHS:-4}"
SEED="${VISA_SEED:-0}"
AUG_SIMPLENET="${VISA_AUG_SIMPLENET:-0.0}"
AUG_PLUS="${VISA_AUG_PLUS:-0.1}"
# One training process used 81 threads without a limit.  Each job gets
# THREADS_PER_JOB OpenMP/MKL threads and NUM_WORKERS loader workers.
THREADS_PER_JOB="${VISA_THREADS_PER_JOB:-4}"
NUM_WORKERS="${VISA_NUM_WORKERS:-4}"
# crop: datasets/visa_mvtecstyle.py (Resize(RESIZE), CenterCrop(IMAGESIZE)).
# letterbox: datasets/visa_legacy.py (whole image letterboxed to IMAGESIZE);
# it needs PIXEL_MODE (original-hist or letterbox-exact, see
# visa_letterbox_f1_eval.py).
LOADER="${VISA_LOADER:-crop}"
PIXEL_MODE="${VISA_PIXEL_MODE:-}"
IMAGE="${SIMPLENET_IMAGE:-simplenet-a6000:wandb}"
case "$LOADER" in
    crop) DATASET_NAME=visa ;;
    letterbox)
        DATASET_NAME=visa_letterbox
        if [[ -z "$PIXEL_MODE" ]]; then
            printf 'VISA_LOADER=letterbox needs VISA_PIXEL_MODE.\n' >&2
            exit 2
        fi
        ;;
    *) printf 'Invalid VISA_LOADER: %s\n' "$LOADER" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- host side
if [[ "${VISA_F1_INSIDE:-0}" != "1" ]]; then
    if [[ $# -lt 3 ]]; then
        printf 'Usage: %s VISA_ROOT RESULTS_ROOT GPU_IDS [JOBS_PER_GPU]\n' "$0" >&2
        exit 2
    fi
    visa_root="$(cd "$1" && pwd)"
    results_root="$2"
    gpu_ids="$3"
    jobs_per_gpu="${4:-4}"
    mkdir -p "$results_root"
    results_root="$(cd "$results_root" && pwd)"
    model_cache="${SIMPLENET_MODEL_CACHE:-${HOME}/.cache/simplenet/torch}"
    mkdir -p "$model_cache"
    exec docker run --rm --init \
        --name "${SIMPLENET_CONTAINER_NAME:-simplenet-visa-f1}" \
        --gpus all --shm-size "${SIMPLENET_SHM_SIZE:-32g}" \
        --workdir /workspace/SimpleNet \
        --env VISA_F1_INSIDE=1 \
        --env VISA_GPUS="$gpu_ids" \
        --env VISA_JOBS_PER_GPU="$jobs_per_gpu" \
        --env VISA_METHODS="${METHODS[*]}" \
        --env VISA_CATEGORIES="${CATEGORIES[*]}" \
        --env VISA_RESIZE="$RESIZE" --env VISA_IMAGESIZE="$IMAGESIZE" \
        --env VISA_META_EPOCHS="$META_EPOCHS" --env VISA_GAN_EPOCHS="$GAN_EPOCHS" \
        --env VISA_SEED="$SEED" \
        --env VISA_AUG_SIMPLENET="$AUG_SIMPLENET" --env VISA_AUG_PLUS="$AUG_PLUS" \
        --env VISA_THREADS_PER_JOB="$THREADS_PER_JOB" --env VISA_NUM_WORKERS="$NUM_WORKERS" \
        --env VISA_LOADER="$LOADER" --env VISA_PIXEL_MODE="$PIXEL_MODE" \
        --env WANDB_MODE=disabled --env PYTHONUNBUFFERED=1 \
        --volume "${script_dir}:/workspace/SimpleNet:rw" \
        --volume "${visa_root}:/data/visa:ro" \
        --volume "${results_root}:/results:rw" \
        --volume "${model_cache}:/root/.cache/torch:rw" \
        "$IMAGE" bash /workspace/SimpleNet/run_visa_f1_multigpu.sh
fi

# ------------------------------------------------------------ container side
data_root="${VISA_DATA_ROOT:-/data/visa}"
results_root="${VISA_RESULTS_ROOT:-/results}"
IFS=',' read -r -a gpus <<< "$VISA_GPUS"
jobs_per_gpu="${VISA_JOBS_PER_GPU:-4}"
mkdir -p "$results_root/logs"
export OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB"

run_job() {
    local gpu="$1" method="$2" category="$3"
    local out="$results_root/visa/$method/$category"
    local log="$results_root/logs/${method}_${category}.log"
    local entry aug
    if [[ "$method" == "simplenet" ]]; then entry=main.py; aug="$AUG_SIMPLENET"; else entry=main_ours.py; aug="$AUG_PLUS"; fi
    local aug_flags=()
    if [[ "$aug" != "0" && "$aug" != "0.0" ]]; then
        aug_flags=(--augment --brightness "$aug" --contrast "$aug" --saturation "$aug")
    fi

    if [[ -f "$out/results.csv" && -f "$out/models/0/${DATASET_NAME}_${category}/ckpt.pth" ]]; then
        printf 'SKIP_TRAIN gpu=%s %s/%s\n' "$gpu" "$method" "$category"
    else
        printf 'START gpu=%s %s/%s %s\n' "$gpu" "$method" "$category" "$(date --iso-8601=seconds)"
        rm -rf "$out"
        if ! python3 "$entry" \
            --gpu "$gpu" --seed "$SEED" \
            --log_project visa --log_group "$method" --run_name "$category" \
            --results_path "$results_root" \
            net -b wideresnet50 -le layer2 -le layer3 \
            --pretrain_embed_dimension 1536 --target_embed_dimension 1536 \
            --patchsize 3 --meta_epochs "$META_EPOCHS" --embedding_size 256 \
            --gan_epochs "$GAN_EPOCHS" --noise_std 0.015 --dsc_hidden 1024 \
            --dsc_layers 2 --dsc_margin .5 --pre_proj 1 \
            dataset --batch_size 8 --num_workers "$NUM_WORKERS" \
            --resize "$RESIZE" --imagesize "$IMAGESIZE" \
            "${aug_flags[@]}" -d "$category" "$DATASET_NAME" "$data_root" > "$log" 2>&1; then
            printf 'TRAIN_FAILED gpu=%s %s/%s log=%s\n' "$gpu" "$method" "$category" "$log"
            return 1
        fi
        printf 'TRAIN_DONE gpu=%s %s/%s %s\n' "$gpu" "$method" "$category" "$(date --iso-8601=seconds)"
    fi

    if [[ -f "$out/f1_eval.json" ]]; then
        printf 'SKIP_EVAL %s/%s\n' "$method" "$category"
        return 0
    fi
    local eval_cmd=(visa_f1_eval.py eval --resize "$RESIZE")
    if [[ "$LOADER" == "letterbox" ]]; then
        eval_cmd=(visa_letterbox_f1_eval.py --pixel-mode "$PIXEL_MODE")
    fi
    if ! python3 "${eval_cmd[@]}" --method "$method" --category "$category" \
        --results-root "$results_root" --data-root "$data_root" --gpu "$gpu" \
        --imagesize "$IMAGESIZE" \
        --meta-epochs "$META_EPOCHS" --gan-epochs "$GAN_EPOCHS" --seed "$SEED" >> "$log" 2>&1; then
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
printf 'VISA_F1 jobs=%d gpus=%s jobs_per_gpu=%s slots=%d threads_per_job=%s num_workers=%s loader=%s imagesize=%s pixel_mode=%s start=%s\n' \
    "${#jobs[@]}" "$VISA_GPUS" "$jobs_per_gpu" "$slots" "$THREADS_PER_JOB" "$NUM_WORKERS" \
    "$LOADER" "$IMAGESIZE" "${PIXEL_MODE:-NONE}" "$(date --iso-8601=seconds)"

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

python3 visa_f1_eval.py merge --results-root "$results_root"
printf 'VISA_F1 finished failures=%d end=%s\n' "$total_failures" "$(date --iso-8601=seconds)"
exit "$(( total_failures > 0 ))"
