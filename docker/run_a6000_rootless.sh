#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/.." && pwd)"

data_root="${FOURIND_DATA_ROOT:-}"
results_root="${FOURIND_RESULTS_ROOT:-}"

if [[ -z "$data_root" && $# -gt 0 ]]; then
    data_root="$1"
    shift
fi
if [[ -z "$results_root" && $# -gt 0 ]]; then
    results_root="$1"
    shift
fi

if [[ -z "$data_root" || -z "$results_root" ]]; then
    printf 'Usage: %s DATA_ROOT RESULTS_ROOT [COMMAND ...]\n' "$0" >&2
    printf 'Or set FOURIND_DATA_ROOT and FOURIND_RESULTS_ROOT.\n' >&2
    exit 2
fi
if [[ ! -d "$data_root" ]]; then
    printf 'Dataset root does not exist: %s\n' "$data_root" >&2
    exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
    printf 'docker is not available in PATH.\n' >&2
    exit 2
fi
if ! docker info >/dev/null 2>&1; then
    printf 'The Docker daemon is not reachable for the current user.\n' >&2
    exit 2
fi

mkdir -p "$results_root"
default_cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}"
model_cache_root="${SIMPLENET_MODEL_CACHE:-${default_cache_base}/simplenet/torch}"
mkdir -p "$model_cache_root"

# Exact image used by the existing workstation container. Override this with
# SIMPLENET_IMAGE=simplenet-a6000:rebuild to test the Dockerfile rebuild.
image_ref="${SIMPLENET_IMAGE:-june6423/simplenet-rtx50@sha256:9efed934c9ad7f4b66cacc1cffd33c1de6c63bf2b4ae7228e0e32051a98abb6b}"
container_name="${SIMPLENET_CONTAINER_NAME:-simplenet-a6000}"
gpu_spec="${SIMPLENET_DOCKER_GPUS:-all}"
shared_memory="${SIMPLENET_SHM_SIZE:-32g}"

terminal_args=()
if [[ -t 0 && -t 1 ]]; then
    terminal_args=(-it)
fi

lifecycle_args=(--rm)
if [[ "${SIMPLENET_KEEP_CONTAINER:-0}" == "1" ]]; then
    lifecycle_args=()
fi

command_args=("$@")
if [[ ${#command_args[@]} -eq 0 ]]; then
    command_args=(bash)
fi

exec docker run \
    "${terminal_args[@]}" \
    "${lifecycle_args[@]}" \
    --name "$container_name" \
    --gpus "$gpu_spec" \
    --shm-size "$shared_memory" \
    --init \
    --workdir /workspace/SimpleNet \
    --env FOURIND_DATA_ROOT=/data/4ind \
    --env FOURIND_RESULTS_ROOT=/results \
    --env PYTHONUNBUFFERED=1 \
    --volume "${repository_root}:/workspace/SimpleNet:rw" \
    --volume "${data_root}:/data/4ind:ro" \
    --volume "${results_root}:/results:rw" \
    --volume "${model_cache_root}:/root/.cache/torch:rw" \
    "$image_ref" \
    "${command_args[@]}"
