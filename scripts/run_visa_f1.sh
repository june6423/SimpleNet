#!/usr/bin/env bash
# Harness entrypoint for the VisA F1 run inside the persistent container.
# Runs the container side of run_visa_f1_multigpu.sh (no docker run).
# Usage: scripts/run_visa_f1.sh OUTPUT_ROOT
# Results go to OUTPUT_ROOT; /proc/loadavg and nvidia-smi go to
# OUTPUT_ROOT/logs/monitor.log at the start, every 30 minutes, and at the end.
set -euo pipefail

output_root="${1:?Usage: $0 OUTPUT_ROOT}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
monitor_log="$output_root/logs/monitor.log"

cd "$repo_dir"
mkdir -p "$output_root/logs"

snapshot() {
    {
        printf '=== %s %s\n' "$1" "$(date --iso-8601=seconds)"
        cat /proc/loadavg
        nvidia-smi || true
    } >> "$monitor_log" 2>&1
}

snapshot start
( while sleep 1800; do snapshot periodic; done ) &
monitor_pid=$!
trap 'pkill -P "$monitor_pid" 2>/dev/null || true; kill "$monitor_pid" 2>/dev/null || true; snapshot end' EXIT

# Load the WRN50 weights once so that the parallel jobs do not download the
# same file at the same time.
python3 -c 'import backbones; backbones.load("wideresnet50")'

VISA_F1_INSIDE=1 \
VISA_GPUS="${VISA_GPUS:-0,1,2,3,4,5,6}" \
VISA_JOBS_PER_GPU="${VISA_JOBS_PER_GPU:-4}" \
VISA_DATA_ROOT="${VISA_DATA_ROOT:-/workspace/visa-anomaly}" \
VISA_RESULTS_ROOT="$output_root" \
WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
    bash run_visa_f1_multigpu.sh
