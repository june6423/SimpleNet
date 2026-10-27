#!/usr/bin/env bash
set -u

root="/workspace/SimpleNet/results_visa_weekend_5060ti"
printf 'Completed jobs: '
find "$root" -name job_result.json 2>/dev/null | wc -l
printf 'Scheduler processes:\n'
pgrep -af 'run_visa_weekend_2gpu|visa_benchmark.py' || true
printf 'Recent scheduler events:\n'
tail -20 "$root/scheduler.log" 2>/dev/null || true
printf 'GPU state:\n'
nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
