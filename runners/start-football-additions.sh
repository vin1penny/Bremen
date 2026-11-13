#!/usr/bin/env bash
# Run inside tmux, after reserving the physical GPU in the university semaphore.
set -euo pipefail

if [[ $# != 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash runners/start-football-additions.sh RESERVED_PHYSICAL_GPU" >&2
  exit 2
fi
gpu="$1"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository"
test -x .venv/bin/python || { echo "Project .venv is missing" >&2; exit 1; }
export PYTHONPATH="$repository/src"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export YOLO_CONFIG_DIR="/home/vincent/.cache/football-pose/ultralytics"
mkdir -p "$YOLO_CONFIG_DIR" /mnt/storage2/vincent/football-pose/results/logs

# Share the existing COCO launcher's lock to prevent same-account collisions.
# This supplements, and does not replace, the university semaphore.
exec 9>"/home/vincent/.cache/football-pose/coco-gpu-${gpu}.lock"
flock -n 9 || { echo "Another experiment launcher from this account is using GPU $gpu" >&2; exit 1; }
usage="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
IFS=, read -r memory utilization <<< "$usage"
memory="${memory//[[:space:]]/}"
utilization="${utilization//[[:space:]]/}"
if [[ ! "$memory" =~ ^[0-9]+$ || ! "$utilization" =~ ^[0-9]+$ ]]; then
  echo "Unable to read GPU usage: $usage" >&2
  exit 1
fi
if (( memory > 100 || utilization > 0 )); then
  echo "GPU $gpu is now in use ($usage); recheck the semaphore before proceeding." >&2
  exit 1
fi

configuration=configs/lyra-additional-methods.yaml
log="/mnt/storage2/vincent/football-pose/results/logs/football-additions-$(date +%Y%m%dT%H%M%S)-gpu${gpu}.log"
echo "Log: $log"
.venv/bin/python -m football_pose screen-preprocessing "$configuration" --device "$gpu" --check-only 2>&1 | tee "$log"
.venv/bin/python -m football_pose screen-preprocessing "$configuration" --device "$gpu" 2>&1 | tee -a "$log"
