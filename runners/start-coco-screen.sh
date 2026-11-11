#!/usr/bin/env bash
# Run only after reserving the physical GPU in the university semaphore.
set -euo pipefail

if [[ $# != 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash runners/start-coco-screen.sh RESERVED_PHYSICAL_GPU" >&2
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

# This lock prevents two copies from this account using the same GPU.
# The university semaphore remains the allocation authority across users.
exec 9>"/home/vincent/.cache/football-pose/coco-gpu-${gpu}.lock"
flock -n 9 || { echo "Another COCO launcher from this account is using GPU $gpu" >&2; exit 1; }
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

# The new COCO detector is shared by learned cropping and the HRNet adapter.
# Download only when missing, from the official Ultralytics release.
detector=/mnt/storage2/vincent/football-pose/models/coco/yolo11x.pt
mkdir -p "$(dirname "$detector")"
if [[ ! -s "$detector" ]]; then
  temporary="$(mktemp "${detector}.download.XXXXXX")"
  trap 'rm -f "$temporary"' EXIT
  curl -fL --retry 3 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt -o "$temporary"
  mv "$temporary" "$detector"
  trap - EXIT
fi

configuration=configs/lyra-coco-preprocessing-screen.yaml
log="/mnt/storage2/vincent/football-pose/results/logs/coco-screen-$(date +%Y%m%dT%H%M%S)-gpu${gpu}.log"
echo "Log: $log"
{
  .venv/bin/python -m football_pose screen-coco "$configuration" --device "$gpu" --check-only
  # All 36 combinations must pass before starting the full 5,000-image run.
  .venv/bin/python -m football_pose screen-coco "$configuration" --device "$gpu" --limit 20
  .venv/bin/python -m football_pose screen-coco "$configuration" --device "$gpu"
} 2>&1 | tee "$log"
