# Lyra server command guide

Run **Mac** commands in a local terminal and **Lyra** commands after SSH.
Paste commands only, not the terminal prompt or previous output.
For the experiment design and output definitions, see [experiment.md](experiment.md).

## 1. Connect and start a session

**On the Mac:**

```bash
ssh vincent@lyra
```

Lyra: `lyra.d2ip.tu-berlin.de` · IP: `141.23.38.146` · account: `vincent`.

Saved account password, retained by request:

```text
fkBoDT9a61VZlFE4VSOD
```

If the SSH alias is missing, add this to `~/.ssh/config` on the Mac:

```sshconfig
Host lyra
    HostName lyra.d2ip.tu-berlin.de
    User vincent
    IdentityFile ~/.ssh/lsi_gpu_rsa
    IdentitiesOnly yes
```

Use the university VPN if required. Do not keep an obsolete jump-host address.
A `user@host's password` prompt asks for that server account's password;
`Enter passphrase for key ...` asks for the local key's passphrase.

**On Lyra:** check available GPUs, then create or reconnect to tmux:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
tmux new -A -s football-experiment
```

**Inside tmux**, activate the project environment. Repeat this in every new shell:

```bash
cd /home/vincent/projects/Bremen
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export TMPDIR=/home/vincent/.cache/football-pose/pip-tmp
export PIP_CACHE_DIR=/home/vincent/.cache/pip
export YOLO_CONFIG_DIR=/home/vincent/.cache/football-pose/ultralytics
which python
python --version
```

Expected Python: `/home/vincent/projects/Bremen/.venv/bin/python`, version 3.12.
The system `python` can be Python 2.7. This repo uses `PYTHONPATH`, not
`pip install -e .` (there is no packaging file).

### tmux essentials

| Action | Command / keys |
| --- | --- |
| List server sessions | `tmux ls` |
| Reconnect | `tmux attach -t football-experiment` |
| Leave running | Ctrl+B, then D |
| Scroll previous output | Ctrl+B, then [; arrows/Page Up; Q to return |
| Enable mouse scrolling | `tmux set -g mouse on` |
| End an idle shell/session | `exit` |

Use tmux **on Lyra**, not on the Mac. Detach before closing SSH; do not press Ctrl+C
if the job should continue. Reconnect with SSH and then attach to the same session.
Other useful session names: `pitch-finetune`, `pitch-pipeline-review`,
`coco-evaluation`; substitute the name in the commands above.
Use `tee` below to keep complete logs; tmux scrollback is limited.

## 2. Locations and storage

| What | Lyra path |
| --- | --- |
| Code and tracked YAML | `/home/vincent/projects/Bremen` |
| Private checkpoints | `/home/vincent/football-pose-private/checkpoints` |
| Private config copies | `/home/vincent/football-pose-private/configs` |
| Input videos | `/mnt/storage2/vincent/football-pose/videos/input` |
| Annotated output videos | `/mnt/storage2/vincent/football-pose/videos/output` |
| YOLO26 / preprocessing results | `/mnt/storage2/vincent/football-pose/results/football` |
| Existing OpenPose baseline results | `/home/vincent/football-pose-results/native-resolution` |
| Existing preprocessing artifacts | `/home/vincent/football-pose-cache/artifacts` |
| Existing pitch geometry cache | `/home/vincent/football-pose-cache/pitch` |
| COCO dataset / cache / results | `/mnt/storage2/vincent/football-pose/{datasets/coco,cache/coco,results/coco-keypoints}` |
| New pitch training / validation | `/mnt/storage2/vincent/football-pose/results/{pitch-training,pitch-validation}` |

Put new large datasets, training runs and videos on mounted storage, not in Git or
`/tmp`. Existing home-based caches and the pitch validation dataset have not
been migrated. `/mnt/storage2` is NFS from `141.23.38.152:/raid0_ssd2`;
backup/retention is not confirmed, so keep another copy of irreplaceable files.

**Upload a video from the Mac** (replace the local source filename):

```bash
scp /Users/larry/Downloads/test-video.mp4 \
  vincent@lyra:/mnt/storage2/vincent/football-pose/videos/input/
```

Then change `input:` in the chosen YAML to its absolute server path.
The current sample is `/mnt/storage2/vincent/football-pose/videos/input/sample_30.mov`.

**Check storage on Lyra:**

```bash
df -hT /home/vincent /mnt/storage2
docker system df
```

Do not prune shared Docker storage or delete other users' files/processes.

## 3. Select the GPU before any run

Choose a GPU with no running compute processes and enough free memory, following
the server's reservation policy. **0% utilization does not mean free**: a process
can still occupy most of its memory. Recheck immediately before starting.

Example using physical GPU **4**; it is not permanently reserved for us:

```bash
nvidia-smi -i 4
export CUDA_VISIBLE_DEVICES=4
nano configs/lyra-openpose-full-frame.yaml
```

In **every config you will run**, set both:

```yaml
models:
  - id: openpose-body25
    # Keep all other model settings unchanged.
    devices: [4]
pitch_filter:
  # Keep all other pitch settings unchanged.
  device: "4"
```

This is a settings excerpt, not a replacement config. Save nano with Ctrl+O,
Enter; exit with Ctrl+X. For a config containing two models, change both device
entries. COCO has no pitch filter.

| Where | Value when using physical GPU 4 |
| --- | --- |
| Shell | `CUDA_VISIBLE_DEVICES=4` |
| Football / COCO YAML model devices | `devices: [4]` |
| Football YAML pitch device | `device: "4"` |
| Direct Ultralytics training/validation CLI | `device=0` (the one exposed GPU) |

Setting only the shell variable does **not** override YAML. Config defaults vary;
check them rather than assuming GPU 7 or 0. Optionally copy a YAML to the private
config directory before editing, then run that copy to keep Git clean.

## 4. Run football experiments and render videos

Current setup: **YOLO26x Pose** for people, **OpenPose BODY_25**, and a separate
**YOLOv8 pitch-landmark model at 640**. Pose input settings are 1920 / 1920x1088
in these football configs. The single-step suite below also includes HRNet-W32,
using the stored YOLO player detector for its required person boxes.

The pitch model runs on original frames, maps landmarks back to source-image
coordinates, and estimates same-frame geometry. The pitch is hardcoded as
**105 × 68 m for Weserstadion, Bremen only**. Do not change pitch weights/settings
between person-preprocessing comparisons. Missing acceptable geometry is reported
as unclassified, not a reliable rectangular pitch estimate.

### One complete OpenPose run

With the venv active and the shell/YAML GPU settings matching:

```bash
python -m football_pose validate-config configs/lyra-openpose-full-frame.yaml
mkdir -p /mnt/storage2/vincent/football-pose/results/logs
RUN_LOG="/mnt/storage2/vincent/football-pose/results/logs/openpose-$(date +%Y%m%dT%H%M%S).log"
set -o pipefail
python -m football_pose run configs/lyra-openpose-full-frame.yaml 2>&1 | tee "$RUN_LOG"
```

This saves predictions, pitch decisions, summary and annotated MP4. Detach tmux
with Ctrl+B, D while it runs. A quiet terminal during inference does not prove a
crash; check `nvidia-smi` and `docker ps` from another SSH terminal.

### Manual YOLO/OpenPose configurations

| Experiment | Config in `configs/` | Models run |
| --- | --- | --- |
| Full-frame baseline | `lyra-yolo-full-frame.yaml` | YOLO26 |
| Deterministic tiling | `lyra-yolo-tiled.yaml` | YOLO26 |
| Full-frame baseline | `lyra-openpose-full-frame.yaml` | OpenPose |
| Deterministic tiling | `lyra-openpose-tiled.yaml` | OpenPose |
| Contrast: CLAHE | `lyra-preprocess-clahe.yaml` | Both, sequentially |
| Gamma darken | `lyra-preprocess-gamma-darken.yaml` | Both, sequentially |
| Gamma brighten | `lyra-preprocess-gamma-brighten.yaml` | Both, sequentially |
| Sharpening: unsharp mask | `lyra-preprocess-unsharp.yaml` | Both, sequentially |

Replace the YAML filename in the validate/run commands above. To run one model
from a two-model config, append `--model yolo26-pose` or
`--model openpose-body25`. To prepare only the cached input, append
`--prepare-only`. List available processing operators with
`python -m football_pose list-processors`; additional operators need
an explicit config and are not included in this eight-config screening.

### All single-step conditions: 36 model jobs

Use `configs/lyra-preprocessing-screen.yaml`: baseline plus resize x2, CLAHE,
both gamma variants, NLM, bilateral denoising, sharpening, Wiener motion deblur,
FSRCNN x2 super-resolution, tiling and tracked player cropping. YOLO26, OpenPose
and HRNet run on every condition. HRNet's input adapter uses the existing
YOLO football player detector, not a generic MMDetection detector.

The suite applies `--device` to every model, pitch filter and crop detector;
**you do not edit GPU IDs in twelve separate YAML files**. Edit shared input,
weights and inference settings only in `configs/lyra-preprocessing-base.yaml`.
All new caches, reports and videos use mounted storage.

**Once on Lyra:** create the asset directory and download the pinned SR checkpoint:

```bash
mkdir -p /mnt/storage2/vincent/football-pose/models/super-resolution
SR_CHECKPOINT=/mnt/storage2/vincent/football-pose/models/super-resolution/FSRCNN_x2.pb
test -f "$SR_CHECKPOINT" || curl -fL --retry 3 \
  https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d/models/FSRCNN_x2.pb \
  -o "$SR_CHECKPOINT"
printf '%s  %s\n' \
  366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c \
  "$SR_CHECKPOINT" | sha256sum --check -
```

The checksum must report OK. **On the Mac**, upload the supplied player detector:

```bash
scp /Users/larry/Bremen/FootballTrackingDataGeneration-main/models/player_detection_model_best/best.pt \
  vincent@lyra:/mnt/storage2/vincent/football-pose/models/player-detection-best.pt
```

This is a separate crop detector; do not overwrite the pitch or person-pose weights.
Its verified classes are 0 ball, 1 goalkeeper, 2 player and 3 referee; the crop
condition and HRNet detector include [1, 2, 3]. Its hash is recorded with each screen.

**Once on Lyra:** install HRNet's separate pose checkpoint and build its container.
The YOLO weights supply person boxes; they do not replace HRNet's joint weights:

```bash
HRNET_CHECKPOINT=/mnt/storage2/vincent/football-pose/models/hrnet-w32-coco-256x192.pth
test -f "$HRNET_CHECKPOINT" || curl -fL --retry 3 \
  https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
  -o "$HRNET_CHECKPOINT"
printf '%s  %s\n' 05654824f0c9b6b573ccaac7dc90e6d0 "$HRNET_CHECKPOINT" | md5sum --check -
cd /home/vincent/projects/Bremen
docker build -f containers/Dockerfile.hrnet -t vincent/football-pose-hrnet:dev .
docker run --rm vincent/football-pose-hrnet:dev --help
```

The image pins PyTorch 2.1/CUDA 11.8 and a prebuilt MMCV 2.1 wheel; it also contains
Ultralytics for the existing detector. The official HRNet download is about 122 MB.
Both checkpoints stay on mounted storage and are mounted into the container read-only.
The wrapper runs as your Lyra UID/GID so private NFS files remain accessible and
new output files are owned by you, not container root. Runtime caches use container `/tmp`.

**On Lyra**, check GPU availability, then open tmux:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
tmux new -A -s preprocessing-all
```

Inside tmux, activate the environment as in section 1. Example with an available,
permitted physical GPU 4:

```bash
cd /home/vincent/projects/Bremen
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=4
python -m football_pose screen-preprocessing \
  configs/lyra-preprocessing-screen.yaml --device 4 --check-only
```

After preflight succeeds, validate HRNet on its own before the full screen:

```bash
docker run --rm --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --entrypoint python3 vincent/football-pose-hrnet:dev \
  -c 'import torch, mmpose; from mmcv.ops import get_compiler_version; print(torch.__version__, torch.version.cuda, get_compiler_version())'
python -m football_pose screen-preprocessing \
  configs/lyra-preprocessing-screen.yaml --device 4 --method baseline --model hrnet-w32
```

If native imports and the HRNet baseline succeed, start all 36 jobs:

```bash
mkdir -p /mnt/storage2/vincent/football-pose/results/logs
SCREEN_LOG="/mnt/storage2/vincent/football-pose/results/logs/single-step-$(date +%Y%m%dT%H%M%S).log"
set -o pipefail
python -m football_pose screen-preprocessing \
  configs/lyra-preprocessing-screen.yaml --device 4 2>&1 | tee "$SCREEN_LOG"
```

Detach: Ctrl+B, D. Reconnect: `tmux attach -t preprocessing-all`.
Preflight checks files, SR checksum, processor parameters and container images.
It does not reserve the GPU or prove that a full-resolution model will fit in VRAM.
Denoising, SR and per-person crops can be expensive; allow the tmux job to finish.

A timestamped run folder is printed below
`/mnt/storage2/vincent/football-pose/results/football/single-step`.
Open its `comparison.md` for the automatic table, `comparison.csv` for analysis,
and `screening.json` for full summaries, errors, checkpoint hashes and video paths.
These reports update after each condition. Failures are recorded, not treated as
zero detections; other conditions continue. A new invocation preserves previous
reports and runs fresh pose jobs, while cached preprocessing can be reused.

To test only selected steps, repeat `--method`; the baseline is always included:

```bash
python -m football_pose screen-preprocessing \
  configs/lyra-preprocessing-screen.yaml --device 4 \
  --method nlm-denoise --method bilateral-denoise --model openpose-body25
```

Resize x2 and SR x2 create larger artifacts, but pose input sizes remain fixed at
YOLO 1920 / OpenPose 1920x1088 for this comparison. The table measures yield and
runtime, not football AP/PCK accuracy. Compare cold/warm preprocessing separately.
Per-condition wall time includes all selected models and video rendering. HRNet's
model wall time includes YOLO detection + crops + pose inference; its runner log
also reports the stages separately. HRNet detects people in each processed full
frame/tile, then applies its standard 256x192 per-person pose input. When the common
artifact already contains person crops, it skips detection and reuses those crops.
Keep the same pitch checkpoint at 640. To run the other two models while preparing
HRNet, repeat `--model yolo26-pose --model openpose-body25`.

## 5. Check results and download videos

The terminal's JSON report is also saved as
`<output_dir>/experiments/<experiment_id>/summary.json`.
`jobs[].outputs.video` is the authoritative MP4 path.

**On Lyra**, show the newest OpenPose full-frame summary and its video paths:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('/home/vincent/football-pose-results/native-resolution/openpose-full-frame')
paths = list(root.glob('experiments/*/summary.json'))
if not paths:
    raise SystemExit('No summary found.')
path = max(paths, key=lambda p: p.stat().st_mtime)
report = json.loads(path.read_text())
print('Summary:', path)
print('Success:', report['success'])
print('Pitch coverage:', report.get('pitch_filter', {}))
for job in report['jobs']:
    print('\nModel:', job['model_id'], 'Status:', job['status'])
    print('Outputs:', json.dumps(job.get('outputs', {}), indent=2))
    if job.get('error'):
        print('Error:', job['error'])
PY
```

For another experiment, change `root` to that YAML's `output_dir`.
Newest summary may be a failed run; check success/status before downloading.

| Output | Purpose |
| --- | --- |
| `summary.json` | Config, completion status, counts, timings and file paths |
| `runner/predictions.jsonl`, `archive/predictions.parquet` | Raw person-pose records |
| `archive/manifest.json` | Archive metadata |
| `predictions-on-pitch.parquet` | Accepted, deduplicated on-pitch records |
| `pitch-decisions.parquet` | Per-record inside/outside/unclassified/duplicate decisions |
| `pitch-filter/.../manifest.json` | Pitch-filter counts/settings |
| `runner/attempt-*/shard-*.stdout.log` / `stderr.log` | Model diagnostics |
| MP4 on mounted storage | Poses, predicted pitch landmarks and estimated boundaries |

Video colors: magenta pitch landmarks, cyan boundary, green inside, red outside,
yellow unclassified (`unavailable` in decisions), gray duplicates.
Counts are detection-yield diagnostics, **not accuracy**;
overlapping tiles can create duplicates. `usable_frames / frame_count` measures
accepted pitch-geometry coverage, not landmark accuracy.

**Automated overview on Lyra:**

```bash
python -m football_pose build-overview /mnt/storage2/vincent/football-pose/results/football
cat /mnt/storage2/vincent/football-pose/results/football/results-overview/records.md
python -m football_pose build-overview /home/vincent/football-pose-results/native-resolution
cat /home/vincent/football-pose-results/native-resolution/results-overview/records.md
```

These are separate result roots; an overview only includes summaries under its
root. The older OpenPose baselines are not automatically included in the mounted
YOLO26/preprocessing overview.

**Download on the Mac, in a new local terminal** (not the SSH terminal).
The successful video supplied on 2026-10-10 can be downloaded directly:

```bash
scp vincent@lyra:/mnt/storage2/vincent/football-pose/videos/output/lyra-native-openpose-full-frame-baseline/4ecb8ee7bd7311ea/openpose-body25-bdce5fcb08b065263bb29512-6d68a4e2.mp4 \
  /Users/larry/Downloads/pitch-review-v8n.mp4
open /Users/larry/Downloads/pitch-review-v8n.mp4
```

For future runs, substitute the exact video path printed by their summary.
Download that run's JSON report too:

```bash
scp vincent@lyra:/home/vincent/football-pose-results/native-resolution/openpose-full-frame/experiments/4ecb8ee7bd7311ea/summary.json \
  /Users/larry/Downloads/openpose-summary.json
```

## 6. Pitch validation, fine-tuning and production weights

Use a separate tmux session: `tmux new -A -s pitch-finetune`.
Activate the venv inside it and export the selected physical GPU as in section 1/3.
Here `device=0` means the GPU exposed by that export.

| Checkpoint | Location / role |
| --- | --- |
| Production | `/home/vincent/football-pose-private/checkpoints/pitch-landmarks-best.pt` |
| Preserved nano reference | `/home/vincent/football-pose-private/checkpoints/pitch-v8n-reference.pt` |
| Selected fine-tuned candidate | `/mnt/storage2/vincent/football-pose/results/pitch-training/pitch-v8n-finetune-6402/weights/best.pt` |

The selected YOLOv8n candidate reported pose mAP50 **0.995**, mAP50–95 **0.844**
on 34 validation images; the additional 20 epochs did not improve it.
The supplied PitchVision x checkpoint performed worse on this validation set.
Do **not** replace production with the local `models/pitch_detection_model/weights/best.pt`
merely because it is named `best.pt`; that file contains the different x candidate.

### Validate the current production checkpoint

```bash
mkdir -p /mnt/storage2/vincent/football-pose/results/pitch-validation
set -o pipefail
yolo pose val \
  model=/home/vincent/football-pose-private/checkpoints/pitch-landmarks-best.pt \
  data=/home/vincent/football-pose-private/pitch-training/pitch-production-data.yaml \
  imgsz=640 batch=2 device=0 split=val \
  project=/mnt/storage2/vincent/football-pose/results/pitch-validation \
  name=pitch-v8n-640 \
  2>&1 | tee /mnt/storage2/vincent/football-pose/results/pitch-validation/pitch-v8n-640.txt
```

Read the **Pose** columns, not the first four Box columns. Confirm the printed
model summary is YOLOv8n-pose. The normalized YAML points to the existing
`/home/vincent/football-pose-private/datasets/football-field-detection-15`
dataset. Use the same images, split and settings when comparing checkpoints.

### Start another fine-tuning cycle only when wanted

```bash
mkdir -p /mnt/storage2/vincent/football-pose/results/pitch-training
set -o pipefail
yolo pose train \
  model=/home/vincent/football-pose-private/checkpoints/pitch-v8n-reference.pt \
  data=/home/vincent/football-pose-private/pitch-training/pitch-production-data.yaml \
  imgsz=640 batch=8 epochs=100 device=0 \
  optimizer=AdamW lr0=0.0001 lrf=0.1 patience=20 mosaic=0.0 \
  seed=0 deterministic=True workers=4 \
  project=/mnt/storage2/vincent/football-pose/results/pitch-training \
  name=pitch-v8n-finetune-640 \
  2>&1 | tee /mnt/storage2/vincent/football-pose/results/pitch-training/pitch-v8n-finetune-640.txt
```

Use the printed save directory: repeated runs get numbered suffixes.
Training uses 255 images; validation uses 34; leave the 28 test images held out.
For another 20-epoch cycle, start from the selected candidate, use
`epochs=20 lr0=0.00001 name=pitch-v8n-extra20-640`. Finished checkpoints have
stripped optimizer state, so this is new fine-tuning, not `resume=True`.
Validate candidates and inspect moving-camera video before adopting them.

### Keep the selected candidate in the production slot

This intentionally replaces the production checkpoint; the reference and training
candidate remain intact. Run only when selecting this candidate for production:

```bash
cp -v \
  /mnt/storage2/vincent/football-pose/results/pitch-training/pitch-v8n-finetune-6402/weights/best.pt \
  /home/vincent/football-pose-private/checkpoints/pitch-landmarks-best.pt
sha256sum \
  /mnt/storage2/vincent/football-pose/results/pitch-training/pitch-v8n-finetune-6402/weights/best.pt \
  /home/vincent/football-pose-private/checkpoints/pitch-landmarks-best.pt
```

Hashes must match. Rerun section 4 in `pitch-pipeline-review` tmux to inspect
pitch landmarks/boundaries with OpenPose. No container rebuild is needed for
changed pitch weights. Good validation scores do not guarantee accurate geometry
on the football video.

For a new dataset, upload the full export to mounted `datasets/pitch/`.
The existing [training script](train/train_pitch_server.py) normalizes split paths:
pass explicit `--data`, `--project`, `--model`, `--imgsz 640`,
`--batch 8 --epochs 100` and a new `--name`; do not rely on its 1280 default.

## 7. COCO keypoints evaluation

This checks person-pose models on original COCO images, with **no pitch filter**
or football preprocessing. It is separate from football detection-count screening.

### Download once, on Lyra

```bash
mkdir -p /mnt/storage2/vincent/football-pose/datasets/coco
cd /mnt/storage2/vincent/football-pose/datasets/coco
curl -fL --retry 3 -C - -o val2017.zip \
  http://images.cocodataset.org/zips/val2017.zip
curl -fL --retry 3 -C - -o annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
printf '%s  %s\n' \
  442b8da7639aecaf257c1dceb8ba8c80 val2017.zip \
  f4bbac642086de4f52a3fdda2de5fa2c annotations_trainval2017.zip | md5sum --check -
```

The HTTP endpoint is a workaround for the observed HTTPS certificate mismatch;
do not use `curl -k`. Extract only after both archive checks report OK:

```bash
unzip -n val2017.zip
unzip -n annotations_trainval2017.zip
```

Required files: `val2017/` (5,000 images) and
`annotations/person_keypoints_val2017.json`.

### Run inside tmux

Open `tmux new -A -s coco-evaluation`, activate the project environment,
select an available GPU, and change **both** model device entries in
`configs/lyra-coco-keypoints.yaml` to the physical GPU ID.

```bash
cd /home/vincent/projects/Bremen
python -m pip install -r requirements/evaluation.txt
python -m football_pose evaluate-coco configs/lyra-coco-keypoints.yaml --limit 50
```

After the smoke check succeeds, start the full run:

```bash
mkdir -p /mnt/storage2/vincent/football-pose/results/logs
COCO_LOG="/mnt/storage2/vincent/football-pose/results/logs/coco-$(date +%Y%m%dT%H%M%S).log"
set -o pipefail
python -m football_pose evaluate-coco configs/lyra-coco-keypoints.yaml 2>&1 | tee "$COCO_LOG"
```

Subset scores are not benchmark results. `validate-config` is for football configs,
not this schema.

**Find the newest COCO summary on Lyra:**

```bash
python - <<'PY'
from pathlib import Path
root = Path('/mnt/storage2/vincent/football-pose/results/coco-keypoints')
paths = list(root.glob('*/summary.json'))
if not paths:
    raise SystemExit('No COCO summary found.')
path = max(paths, key=lambda p: p.stat().st_mtime)
print('Summary:', path)
print(path.read_text())
for log in sorted(path.parent.glob('model-*/coco-evaluation.txt')):
    print('\nMetrics:', log)
    print(log.read_text())
PY
```

Each run contains `summary.json` and `model-00/`, `model-01/` metric
reports/predictions. Confirm model IDs, completion and image count. Download the
desired run from the Mac with `scp -r vincent@lyra:/absolute/run/directory /Users/larry/Downloads/`.

## 8. First-time setup and updates

Skip completed steps. The existing server environment should not be recreated
for every experiment.

### Clone and create the venv (Lyra, once)

```bash
mkdir -p /home/vincent/projects
cd /home/vincent/projects
git clone --branch codex/modular-football-pipeline --single-branch \
  https://github.com/vin1penny/Bremen.git
cd /home/vincent/projects/Bremen
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

### Create storage and install host dependencies (Lyra)

```bash
umask 077
mkdir -p /home/vincent/football-pose-private/{checkpoints,configs}
mkdir -p /home/vincent/football-pose-cache/{artifacts,pitch}
mkdir -p /home/vincent/.cache/football-pose/{pip-tmp,ultralytics}
mkdir -p /home/vincent/.cache/pip
mkdir -p /mnt/storage2/vincent/football-pose/videos/{input,output}
mkdir -p /mnt/storage2/vincent/football-pose/datasets/{coco,pitch}
mkdir -p /mnt/storage2/vincent/football-pose/cache/coco
mkdir -p /mnt/storage2/vincent/football-pose/results/{football,coco-keypoints,pitch-training,pitch-validation,logs}
export TMPDIR=/home/vincent/.cache/football-pose/pip-tmp
export PIP_CACHE_DIR=/home/vincent/.cache/pip
export YOLO_CONFIG_DIR=/home/vincent/.cache/football-pose/ultralytics
python -m pip install --upgrade pip
python -m pip install --resume-retries 20 -r requirements/orchestrator.txt
python -m pip install --resume-retries 20 -r requirements/dev.txt
python -m pip install -r requirements/evaluation.txt
python -m pip check
python -m pytest -q
```

The host handles preprocessing, pitch inference, archives, evaluation and rendering.
Containers isolate person-pose model runtimes. Checkpoint architecture is separate
from the installed Ultralytics library version.

### Build model containers (Lyra, repo root)

```bash
docker build -f containers/Dockerfile.yolo -t vincent/football-pose-yolo:dev .
docker build -f containers/Dockerfile.yolo26 -t vincent/football-pose-yolo26:dev .
docker build --build-arg BUILD_JOBS=8 \
  -f containers/Dockerfile.openpose -t vincent/football-pose-openpose:dev .
docker image ls --filter 'reference=vincent/football-pose-*'
docker run --rm vincent/football-pose-yolo26:dev --help
docker run --rm vincent/football-pose-openpose:dev --help
```

YOLO26 builds on the base YOLO image, so build in that order. OpenPose intentionally
clones a pinned CMU source revision and compiles Caffe/OpenPose with cuDNN; this
can take time. BODY_25 weights are supplied separately and inherit OpenPose's
non-commercial license.

The HRNet image and its separate pose-weight download are in section 4; build that
image too before running the default three-model screen.

Rebuild when container dependencies/native code change. The wrapper mounts the
current repo, so host code/config/checkpoint-only changes generally do not need
a rebuild. With a GPU selected, verify the YOLO26 container can see it:

```bash
docker run --rm --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --entrypoint python3 vincent/football-pose-yolo26:dev \
  -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))'
```

### Get missing person-pose weights (Lyra)

Only download if absent; do not overwrite selected pitch weights.

```bash
cd /home/vincent/football-pose-private/checkpoints
test -f yolo26x-pose.pt || curl -fL --retry 3 -o yolo26x-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt
test -f pose_iter_584000.caffemodel || curl -fL --retry 3 -o pose_iter_584000.caffemodel \
  'https://huggingface.co/camenduru/openpose/resolve/f5bb0c0a16060ac8b373472a5456c76bd68eb202/models/pose/body_25/pose_iter_584000.caffemodel?download=true'
sha256sum yolo26x-pose.pt pose_iter_584000.caffemodel
cd /home/vincent/projects/Bremen
```

Expected BODY_25 SHA-256:
`44e3d7ebd8c8b62d4366d67127f1b562611a9e8fd0f4f3cdeeb4bb4a6ed12be6`.
Use the selected trained pitch checkpoint from section 6, not generic COCO weights.

### Update existing code (Lyra, before starting jobs)

```bash
cd /home/vincent/projects/Bremen
git status --short --branch
```

Inspect local changes first, including server GPU edits. If the worktree is clean:

```bash
git switch codex/modular-football-pipeline
git pull --ff-only
```

Activate the venv again; install changed requirements/rebuild changed containers
when needed. Do not discard unfamiliar changes or pull while a job is running.

## 9. Quick troubleshooting

| Problem | What to check / do |
| --- | --- |
| Python 2.7 syntax error / no `football_pose` | Activate repo venv, set `PYTHONPATH="$PWD/src"`; verify `which python` |
| GPU outside reservation | Match shell export, every model's physical devices and pitch device |
| OpenPose OOM at batch 1 | Check free VRAM/processes, use an available permitted GPU; reducing clip length does not reduce single-frame memory |
| Pip fills `/tmp` | Set `TMPDIR` / pip cache from section 1 before installing |
| Docker permission denied | Ask administrator about Docker access; do not change shared daemon permissions |
| SSH key too open | On Mac: `chmod 600 /Users/larry/.ssh/lsi_gpu_rsa` |
| SSH invalid key format | Check the actual key file; this is not fixed by entering its passphrase |
| SSH disconnect | Reconnect and attach tmux; do not restart blindly |
| Missing results/video | Check config's `output_dir`, latest summary status and `outputs.video` |

For a failed model, use its **exact job ID** from the summary:

```bash
JOB_DIR=/home/vincent/football-pose-results/native-resolution/openpose-full-frame/jobs/da9bd668692800b6fdf5077a
find "$JOB_DIR/runner" -type f -name '*.log' -print
tail -n 100 "$JOB_DIR"/runner/attempt-*/shard-*.log
```

That ID is an earlier failed run; replace it with the failed job you are inspecting.
Never kill another user's process to free a GPU. Do not delete caches as a first fix.

More detail: [experiment.md](experiment.md), [architecture](docs/architecture.md),
[TODOs](TODOS.md).
