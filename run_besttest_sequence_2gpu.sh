#!/usr/bin/env bash
set -uo pipefail

workspace="${1:-/workspace/SimpleNet}"
mvtec_root="${2:-$workspace/results_mvtec_gl_besttest_5060ti_320_seed0}"
visa_root="${3:-$workspace/results_visa_besttest_5060ti_320_seed0}"
check_seconds="${GPU_CHECK_SECONDS:-3600}"
sequence_root="$workspace/results_besttest_sequence_5060ti_320_seed0"

mkdir -p "$sequence_root"
exec 9>"$sequence_root/sequence.lock"
if ! flock -n 9; then
    printf 'Another best-test sequence is already active.\n' >&2
    exit 1
fi
printf '%s\n' "$$" >"$sequence_root/scheduler.pid"

gpu_is_available() {
    local gpu="$1"
    local used util uuid
    local app_uuid app_pid app_name app_memory
    if ! read -r used util < <(
        nvidia-smi -i "$gpu" \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' '
    ); then
        return 1
    fi
    if (( used >= 2048 || util > 20 )); then
        return 1
    fi
    uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
    while IFS=',' read -r app_uuid app_pid app_name app_memory; do
        app_uuid="${app_uuid// /}"
        app_name="${app_name# }"
        if [[ -z "$app_uuid" || "$app_uuid" != "$uuid" ]]; then
            continue
        fi
        if [[ "$app_name" =~ (rustdesk|Xorg|gnome-shell|kwin_wayland) ]]; then
            continue
        fi
        return 1
    done < <(
        nvidia-smi \
            --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
            --format=csv,noheader 2>/dev/null || true
    )
    return 0
}

wait_for_both_gpus() {
    local stage="$1"
    while true; do
        if gpu_is_available 0 && gpu_is_available 1; then
            printf 'GPU_CHECK stage=%s status=available time=%s\n' \
                "$stage" "$(date --iso-8601=seconds)"
            return 0
        fi
        printf 'GPU_CHECK stage=%s status=busy retry_seconds=%s time=%s\n' \
            "$stage" "$check_seconds" "$(date --iso-8601=seconds)"
        sleep "$check_seconds"
    done
}

printf 'SEQUENCE_START pid=%s check_seconds=%s time=%s\n' \
    "$$" "$check_seconds" "$(date --iso-8601=seconds)"

wait_for_both_gpus "mvtec_global_local_besttest"
"$workspace/run_mvtec_gl_besttest_2gpu.sh" \
    "$workspace/mvtec-ad" "$mvtec_root" || {
    status=$?
    printf 'SEQUENCE_STOP stage=mvtec status=%s time=%s\n' \
        "$status" "$(date --iso-8601=seconds)"
    exit "$status"
}

wait_for_both_gpus "visa_three_methods_besttest"
"$workspace/run_visa_besttest_2gpu.sh" \
    "$workspace/visa-anomaly-detection" "$visa_root" || {
    status=$?
    printf 'SEQUENCE_STOP stage=visa status=%s time=%s\n' \
        "$status" "$(date --iso-8601=seconds)"
    exit "$status"
}

printf 'SEQUENCE_DONE status=0 time=%s\n' "$(date --iso-8601=seconds)"
