# Three-model COCO preprocessing experiment

Reserve one physical GPU in the university semaphore before running inference.
The suite contains YOLO26x Pose, OpenPose BODY_25 and HRNet-W32, each evaluated
on the same 5,000 COCO val2017 images under twelve conditions:
baseline, resize x2, CLAHE, gamma darken, gamma brighten, NLM denoise,
bilateral denoise, unsharp mask, motion deblur, FSRCNN super-resolution x2,
2×2 overlapping tiles, and learned person crops. This is 36 model/condition jobs.

## Get the experiment branch

In the **connected Lyra terminal**, fetch and switch to the experiment branch:

```bash
cd /home/vincent/projects/Bremen
git fetch origin codex/coco-preprocessing-screen
git switch -c codex/coco-preprocessing-screen FETCH_HEAD
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m pip install -r requirements/evaluation.txt
```

If Git reports conflicting local changes, stop and inspect them; do not reset
the server checkout or force the switch. If the previous patch was already
applied, keep that work until it has been compared with the branch.
The explicit fetch also works with Lyra's older single-branch clone. For later
updates while on this branch, run `git pull --ff-only origin codex/coco-preprocessing-screen`.

The existing host orchestrator dependencies and all three Docker images are
required. The launcher downloads the new COCO detector `yolo11x.pt` from the
[official Ultralytics release](https://github.com/ultralytics/assets/releases/tag/v8.3.0)
only if missing. Existing pose checkpoints, the pinned FSRCNN_x2 checkpoint,
COCO images and annotations must already be present at the config paths.
Preflight lists missing prerequisites before any model job starts.
Use `server.md` for the existing dataset, pose weights and container setup.

## Launch after reserving GPU 0

```bash
tmux new-session -s coco-preprocessing \
  'cd /home/vincent/projects/Bremen && bash runners/start-coco-screen.sh 0; exec bash'
```

The launcher checks that GPU 0 is still idle, validates assets and Docker images,
then runs all 36 combinations on a 20-image smoke subset. Only if all succeed
does it start the full dataset. Jobs run sequentially on the reserved GPU.
Detach with **Ctrl+B, then D**; reconnect with:

```bash
tmux attach -t coco-preprocessing
```

Smoke results are marked `subset-smoke` and must not be reported as full COCO
benchmark scores. Logs go under `/mnt/storage2/vincent/football-pose/results/logs/`.
Each suite prints its directory under
`/mnt/storage2/vincent/football-pose/results/coco-preprocessing/`, containing:

- `screening.json`: progress, failures, full per-model reports and comparison.
- `comparison.csv` and `comparison.md`: AP, AP50, AP75, AR and baseline AP changes.
- Expanded per-method YAML and per-model predictions/COCO evaluator output.

## Resume an interrupted suite

Re-reserve the same GPU and replace `SUITE_DIRECTORY` with the printed path:

```bash
cd /home/vincent/projects/Bremen
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
python -u -m football_pose screen-coco configs/lyra-coco-preprocessing-screen.yaml \
  --device 0 --resume SUITE_DIRECTORY
```

For a smoke suite, also supply its original `--limit 20`. Resume verifies code,
configuration, dataset and checkpoint identities plus saved prediction hashes.
Completed valid jobs are reused; failed/incomplete jobs are retried. Do not
change the GPU/config/code halfway through a suite and call it the same run.

## Evaluation protocol

This is a controlled preprocessing ablation, not a claim to reproduce published
model scores. All methods use the same sorted image IDs and model settings.
Preprocessing transforms are inverted by the model runners, and COCO scores
are computed in original-image coordinates. Images without crops/predictions
remain in the evaluation. There is no football pitch filter or tracking across
independent images.

HRNet needs person boxes. It uses the same COCO-pretrained YOLO11x detector as
the crop condition (person class 0, confidence 0.3, image size 640, padding 0.1,
minimum crop size 32); shared person crops bypass its detector. Detection misses
therefore count against HRNet's end-to-end score. Detector wall time is included
in HRNet model time; shared crop generation is recorded as preprocessing time.

Predictions retain each runner's native person score. For tile/crop conditions,
cross-region bounding-box NMS at IoU 0.5 removes duplicate detections from
different regions; predictions from the same region are retained. Standard
COCO OKS evaluation uses maxDets=20. AP/AR are recorded on a 0–1 scale,
and baseline AP changes are percentage points. `-1` means a metric is unavailable.
Fixed-size model input rescaling still occurs after preprocessing; doubling
input dimensions does not necessarily double the model's inference resolution.
