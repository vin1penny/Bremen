#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 IMAGE [RUNNER_ARGS...]" >&2
  exit 2
fi

image="$1"
shift
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_arguments=()
mount_arguments=(--volume "${repository}:${repository}")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_arguments=(--gpus "device=${CUDA_VISIBLE_DEVICES}")
fi

# Cache artifacts may live outside the repository (for example /lyra/cache).
# Mount only the exact artifact/checkpoint and output directory passed through
# the runner contract rather than exposing the entire shared filesystem.
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --input-artifact|--checkpoint|--detector-checkpoint)
      index=$((index + 1))
      path="${arguments[index]}"
      if [[ "${path}" != "${repository}"/* ]]; then
        mount_arguments+=(--volume "${path}:${path}:ro")
      fi
      ;;
    --output-jsonl)
      index=$((index + 1))
      path="$(dirname "${arguments[index]}")"
      if [[ "${path}" != "${repository}"/* ]]; then
        mount_arguments+=(--volume "${path}:${path}")
      fi
      ;;
  esac
done

exec docker run --rm \
  "${gpu_arguments[@]}" \
  "${mount_arguments[@]}" \
  --user "$(id -u):$(id -g)" \
  --workdir "${repository}" \
  --env XDG_CACHE_HOME=/tmp/football-pose-cache \
  --env TORCH_HOME=/tmp/football-pose-torch \
  --env MPLCONFIGDIR=/tmp/football-pose-matplotlib \
  --env YOLO_CONFIG_DIR=/tmp/ultralytics \
  --env "PYTHONPATH=${repository}/src" \
  "${image}" \
  "$@"
