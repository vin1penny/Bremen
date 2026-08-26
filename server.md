# Lyra server runbook

This is the durable reference for connecting to Lyra, locating the project, testing it safely, and running thesis experiments. Update the path table whenever a location changes.

## Saved connection and repository details

| Item | Saved value |
| --- | --- |
| SSH alias | `lyra` |
| Server hostname | `lyra.d2ip.tu-berlin.de` |
| Server IP snapshot | `141.23.38.146` |
| Server user | `vincent` |
| Server home | `/home/vincent` |
| Server repository | `/home/vincent/projects/Bremen` |
| GitHub repository | `https://github.com/vin1penny/Bremen.git` |
| Working branch | `codex/modular-football-pipeline` |
| Local Mac repository | `/Users/larry/Bremen/FootballTrackingDataGeneration-main` |
| Regenerable server cache (current fallback) | `/home/vincent/football-pose-cache/artifacts` |
| Preferred shared cache (not yet provisioned) | `/lyra/cache/vincent/football-pose/artifacts` |
| Suggested private-data root | `/home/vincent/football-pose-private` |
| Suggested results root | `/home/vincent/football-pose-results` |

Confirm with the Lyra administrator whether `/home/vincent` is backed up and whether it has a quota. Do not treat `/lyra/cache` as backed-up storage.

## Observed Lyra environment

Recorded on 2026-08-17. Recheck before diagnosing a future environment problem.

| Component | Observed value |
| --- | --- |
| Operating system | Debian GNU/Linux 12 (Bookworm) |
| Python | 3.12.4 |
| Git | 2.39.5 |
| Docker | 28.1.1 |
| Docker NVIDIA runtime | Available |
| GPUs | 8 x NVIDIA A100-SXM4 40 GB |
| NVIDIA driver | 575.51.03 |
| Driver-reported CUDA compatibility | 12.9 |
| `nvitop` | Not installed; use `nvidia-smi` |
| `/lyra` snapshot | 2.1 TB total, 226 GB available, 90% used |
| Root filesystem snapshot | 94 GB total, 1.4 GB available, 99% used |
| `/home` snapshot | 4.7 TB total, 224 GB available, 96% used |
| Built YOLO image | `vincent/football-pose-yolo:dev` (9.84 GB) |
| Verified YOLO packages | PyTorch 2.5.1+cu124, Ultralytics 8.4.19, OpenCV 4.10.0, PyAV 14.2.0 |

The CUDA number printed by `nvidia-smi` describes the driver's maximum compatible CUDA version. Each model container supplies its own pinned CUDA runtime.

## 1. Connect to Lyra

Connect to the university VPN first if Lyra is not reachable from the current network. From the Mac Terminal:

```bash
ssh vincent@lyra
```

The Mac SSH alias should resolve directly to the hostname from the server account
email:

```sshconfig
Host lyra
    HostName lyra.d2ip.tu-berlin.de
    User vincent
    IdentityFile ~/.ssh/lsi_gpu_rsa
    IdentitiesOnly yes
```

Do not retain an old `ProxyCommand` or the obsolete internal address `192.168.5.6` in
this host block.

On the first connection, verify the host fingerprint with the administrator before accepting it. Never share a password or private SSH key.

Confirm the session:

```bash
hostname
whoami
pwd
nvidia-smi
```

Expected identity:

```text
lyra
vincent
/home/vincent
```

Disconnect with:

```bash
exit
```

## 2. Clone the project for the first time

On Lyra:

```bash
mkdir -p /home/vincent/projects
cd /home/vincent/projects

git clone \
  --branch codex/modular-football-pipeline \
  --single-branch \
  https://github.com/vin1penny/Bremen.git

cd /home/vincent/projects/Bremen
git status --short --branch
git log -1 --oneline
```

All project commands below assume this repository root is the current directory:

```bash
cd /home/vincent/projects/Bremen
```

## 3. Update an existing server clone

Before pulling, inspect local work:

```bash
cd /home/vincent/projects/Bremen
git status --short --branch
```

If the worktree is clean:

```bash
git switch codex/modular-football-pipeline
git pull --ff-only
```

Do not discard, overwrite, or stash unfamiliar server changes. Commit intentional work or inspect it before pulling.

## 4. Create the server storage layout

Keep private and durable data outside the Git clone. This reduces the risk of committing footage, weights, secrets, or large results.

```bash
mkdir -p /home/vincent/football-pose-private/footage
mkdir -p /home/vincent/football-pose-private/checkpoints
mkdir -p /home/vincent/football-pose-private/configs
mkdir -p /home/vincent/football-pose-results
mkdir -p /home/vincent/football-pose-cache/artifacts
mkdir -p /home/vincent/.cache/football-pose/pip-tmp
mkdir -p /home/vincent/.cache/football-pose/ultralytics
chmod 700 /home/vincent/football-pose-private
```

Creating `/lyra/cache/vincent/...` currently fails with `Permission denied`. Use the
home-based cache above until an administrator provisions a writable directory on
`/lyra/cache`; then update experiment YAML explicitly. The home fallback avoids the
nearly full root filesystem, but it is not proof that the data is backed up.

| Data | Location | Git-backed? | Regenerable? |
| --- | --- | --- | --- |
| Source code and experiment YAML | `/home/vincent/projects/Bremen` | Yes | Yes |
| Private footage | `/home/vincent/football-pose-private/footage` | No | Usually no |
| Unique checkpoints | `/home/vincent/football-pose-private/checkpoints` | No | Usually no |
| Parquet, manifests, logs | `/home/vincent/football-pose-results` | No | Expensive to reproduce |
| Preprocessed artifacts | `/home/vincent/football-pose-cache/artifacts` | No | Yes |

Check space before large jobs:

```bash
df -h / /tmp /home/vincent /lyra
df -i / /tmp /home/vincent /lyra
docker system df
```

The `quota` command is not installed on Lyra. Ask the administrator about any
account or ZFS quota that is not visible through `df`.

Never run `docker system prune` on the shared Docker installation.

## 5. Transfer footage and checkpoints

Run `scp` from the Mac, not from inside the Lyra SSH session. Examples:

```bash
scp /local/path/test-video.mp4 \
  vincent@lyra:/home/vincent/football-pose-private/footage/

scp /local/path/player.pt \
  vincent@lyra:/home/vincent/football-pose-private/checkpoints/

scp /local/path/yolov8x-pose.pt \
  vincent@lyra:/home/vincent/football-pose-private/checkpoints/

scp /local/path/hrnet-w32.pth \
  vincent@lyra:/home/vincent/football-pose-private/checkpoints/
```

Verify on Lyra:

```bash
ls -lh /home/vincent/football-pose-private/footage
ls -lh /home/vincent/football-pose-private/checkpoints
sha256sum /home/vincent/football-pose-private/checkpoints/*
```

Do not put secrets, private footage, or unique weights into Git. The modular inference pipeline does not need a Roboflow API key unless a separate training or Roboflow notebook is used.

The YOLO Pose checkpoint already present in the Mac clone can be transferred with:

```bash
scp /Users/larry/Bremen/FootballTrackingDataGeneration-main/train/yolov8x-pose.pt \
  vincent@lyra:/home/vincent/football-pose-private/checkpoints/
```

## 6. Create the Python environment

On Lyra, from the repository root:

```bash
cd /home/vincent/projects/Bremen
python3.12 -m venv .venv
source .venv/bin/activate

export TMPDIR=/home/vincent/.cache/football-pose/pip-tmp
export PIP_CACHE_DIR=/home/vincent/.cache/pip
export YOLO_CONFIG_DIR=/home/vincent/.cache/football-pose/ultralytics

python -m pip install --upgrade pip
python -m pip install --resume-retries 20 -r requirements/dev.txt

export PYTHONPATH="$PWD/src"
```

Install the smaller development environment first and run the CPU checks in the
next section. Only after those pass, install the heavier host dependencies used by
crop detection and model orchestration:

```bash
python -m pip install --resume-retries 20 -r requirements/orchestrator.txt
python -m pip check
```

PyTorch is a roughly 906 MB download. `TMPDIR` must not point at `/tmp`, because
`/tmp` is on the nearly full root filesystem. `--resume-retries 20` makes an
interrupted download retry without changing package versions.

Reactivate this environment after reconnecting:

```bash
cd /home/vincent/projects/Bremen
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export TMPDIR=/home/vincent/.cache/football-pose/pip-tmp
export PIP_CACHE_DIR=/home/vincent/.cache/pip
export YOLO_CONFIG_DIR=/home/vincent/.cache/football-pose/ultralytics
```

Activating `.venv` selects the project interpreter, but it does not automatically add
the repository's `src/` directory to Python's import path. Both `source` and the
`PYTHONPATH` export are therefore required after every new SSH login. Confirm them
before running tests or experiments:

```bash
which python
echo "$PYTHONPATH"
python -c "import football_pose; print(football_pose.__file__)"
```

The expected values begin with:

```text
/home/vincent/projects/Bremen/.venv/bin/python
/home/vincent/projects/Bremen/src
/home/vincent/projects/Bremen/src/football_pose/__init__.py
```

If Python reports `No module named football_pose`, return to the repository root and
set `PYTHONPATH` again:

```bash
cd /home/vincent/projects/Bremen
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

Do not use `pip install -e .` for this repository. It currently has no `pyproject.toml`
or `setup.py`; `football_pose` is loaded directly from `src/` through `PYTHONPATH`.

## 7. Run the safe CPU checks

These commands do not require a GPU reservation:

```bash
python -m pip check
python -m pytest -q
python -m football_pose list-processors
python -m football_pose validate-config configs/mock.yaml
python -m football_pose run configs/mock.yaml
python -m football_pose run configs/mock.yaml
```

Expected unit-test result after pulling the deterministic-tiling update:

```text
29 passed
```

The earlier server run on the preceding revision passed all 16 tests. Its cold mock
run processed 782 frames successfully, and the second run reported
`"cache_hit": true` with about 0.067 seconds of preprocessing wall time. This checks
video decoding, preprocessing, caching, runner orchestration, validation, Parquet
output, and resume behavior without using CUDA.

## 8. Build and verify the YOLO container

Docker packages the YOLO runner, CUDA runtime, Python, and pinned libraries into one
reproducible image. The host orchestrator remains in `.venv`; it launches the image
through `runners/container-runner.sh` and mounts only the required artifact,
checkpoint, output directory, and repository.

Container builds do not require a GPU reservation, but they use CPU, network, shared
Docker storage, and time. Prefix image names with the Lyra username so they do not
collide with another user's images.

```bash
docker build \
  -f containers/Dockerfile.yolo \
  -t vincent/football-pose-yolo:dev .
```

This image has already built successfully on Lyra. The Python 3.10 container uses
NumPy 2.2.6 because NumPy 2.3+ requires Python 3.11; the version marker is tested in
the repository. Do not build HRNet or OpenPose yet. First complete the YOLO hardware
smoke test below.

Verify the image without reserving a GPU:

```bash
docker image ls vincent/football-pose-yolo

docker run --rm \
  --entrypoint python3 \
  vincent/football-pose-yolo:dev \
  -c "import torch, ultralytics, cv2, av; print('torch:', torch.__version__); print('ultralytics:', ultralytics.__version__); print('opencv:', cv2.__version__); print('pyav:', av.__version__)"

docker run --rm vincent/football-pose-yolo:dev --help
```

Both checks have passed on Lyra. Ultralytics may say that `/root/.config` is not
writable and use `/tmp/Ultralytics` inside the disposable container. That warning is
harmless for this runner because each container is removed after its shard finishes.

## 9. Follow the shared GPU reservation policy

Before every GPU command:

1. Read the shared semaphore/reservation document.
2. Check current GPU state with `nvidia-smi`.
3. Record `vincent`, the physical GPU IDs, and the expected duration in the semaphore.
4. Export exactly the reserved physical IDs through `CUDA_VISIBLE_DEVICES`.
5. Put the same physical IDs in every selected model's YAML `devices` list.
6. Release the reservation when the command finishes or fails.

An apparently idle GPU in `nvidia-smi` is not permission to use it. The semaphore is authoritative.

Because `nvitop` is not installed, monitor with:

```bash
watch -n 2 nvidia-smi
```

Also monitor host resources:

```bash
free -h
df -h / /tmp /home/vincent /lyra
```

## 10. Prepare the tracked YOLO-only smoke test

Pull the current branch, then copy the tracked smoke template outside the repository
so inserting the reserved GPU ID does not create a Git change:

```bash
cd /home/vincent/projects/Bremen
git pull --ff-only

cp configs/lyra-yolo-one-gpu.yaml \
  /home/vincent/football-pose-private/configs/lyra-yolo-one-gpu.yaml
```

The template deliberately tests only YOLO Pose with the tracked 30-second sample,
`resize`, and `clahe`. It does not enable crop detection, HRNet, or OpenPose. This
isolates the first real model and reuses the two preprocessing stages already proven
by the CPU mock run.

Confirm the checkpoint exists:

```bash
ls -lh /home/vincent/football-pose-private/checkpoints/yolov8x-pose.pt
sha256sum /home/vincent/football-pose-private/checkpoints/yolov8x-pose.pt
```

Validate and materialize the lossless artifact. These commands are CPU-only because
this configuration has no `crop` processor, so they do not require a GPU
reservation:

```bash
python -m football_pose validate-config \
  /home/vincent/football-pose-private/configs/lyra-yolo-one-gpu.yaml

python -m football_pose run \
  /home/vincent/football-pose-private/configs/lyra-yolo-one-gpu.yaml \
  --prepare-only
```

Expect `"success": true`, 782 frames, and an FFV1 artifact path under
`/home/vincent/football-pose-cache/artifacts`.

## 11. Reserve one GPU and set the same ID in both places

Follow the semaphore procedure in section 9. After a physical GPU is assigned, edit
the private YAML:

```bash
nano /home/vincent/football-pose-private/configs/lyra-yolo-one-gpu.yaml
```

Replace the placeholder with the assigned physical ID. For example only, if the
semaphore assigned GPU 2:

```yaml
devices: [2]
```

Export the identical ID in the shell:

```bash
export CUDA_VISIBLE_DEVICES=2
```

The example ID is not a recommendation. Do not continue unless it is the GPU actually
assigned in the shared semaphore.

Confirm that Docker can see exactly one A100:

```bash
docker run --rm \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --entrypoint python3 \
  vincent/football-pose-yolo:dev \
  -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('visible GPUs:', torch.cuda.device_count()); print('GPU 0:', torch.cuda.get_device_name(0))"
```

Expected essentials:

- `CUDA available: True`
- `visible GPUs: 1`
- `GPU 0: NVIDIA A100-SXM4-40GB`

If this fails, release the reservation after diagnosis and do not start inference.

## 12. Run YOLO Pose on the short artifact

Run from the repository root while the reservation and export are still active:

```bash
cd /home/vincent/projects/Bremen
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export TMPDIR=/home/vincent/.cache/football-pose/pip-tmp
export PIP_CACHE_DIR=/home/vincent/.cache/pip
export YOLO_CONFIG_DIR=/home/vincent/.cache/football-pose/ultralytics

python -m football_pose run \
  /home/vincent/football-pose-private/configs/lyra-yolo-one-gpu.yaml \
  --model yolo-pose
```

Open a second SSH session to monitor the reserved GPU:

```bash
watch -n 2 nvidia-smi
```

The initial batch is 8. If a CUDA OOM occurs, the orchestrator automatically retries
the complete YOLO attempt with batches 4, 2, and then 1.

The smoke test passes only when all of these are true:

- The command exits with code 0.
- The summary reports `"success": true` and `"cache_hit": true`.
- The YOLO job reports `"status": "COMPLETE"`.
- `predictions.parquet` and `manifest.json` exist.
- The runner stderr log has no unexplained exception.

This is an infrastructure pass. A `COMPLETE` job with `records: "0"` proves the
container and archive contract, but it is not a useful pose result. The first Lyra
run completed this contract and produced zero records because the wide frame reached
YOLO at its default 640-pixel inference size.

Inspect the output without guessing its generated job ID:

```bash
find /home/vincent/football-pose-results/yolo-one-gpu-smoke \
  -type f \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' \
  | sort
```

Release the semaphore reservation immediately after the run finishes or fails.

## 13. Compare full-frame YOLO with deterministic tiling

Pull the tiling implementation and rebuild the YOLO image. Docker should reuse the
large dependency layers:

```bash
cd /home/vincent/projects/Bremen
git pull --ff-only

docker build \
  -f containers/Dockerfile.yolo \
  -t vincent/football-pose-yolo:dev .
```

The two tracked configurations keep the video, official checkpoint, inference size,
confidence, batch, and GPU constant. Their only image-input difference is the `tile`
processor:

```bash
python -m football_pose validate-config configs/lyra-yolo-full-frame.yaml
python -m football_pose validate-config configs/lyra-yolo-tiled.yaml
```

After reserving the configured GPU and exporting the same ID, run:

```bash
python -m football_pose run configs/lyra-yolo-full-frame.yaml
python -m football_pose run configs/lyra-yolo-tiled.yaml
```

The full-frame baseline uses the original 1920 x 1080 frames but fixes YOLO inference
at 640. The tiled run uses a 2 x 2 grid with 10% overlap and the same 640 inference
size. Compare the record count, model time, and logs. Overlap duplicates remain
identified by tile ID; they will be fused in original-frame coordinates before final
metrics.

The observed result was 0 YOLO pose records for 782 full frames and 451 raw records
for 3,128 tiles. Tiling therefore helped, but the upper-bound player-instance recall
is below 3% when at least 20 players are visible per frame. It is not sufficient as
the final pipeline.

## 14. Compare full-frame OpenPose with deterministic tiling

OpenPose is bottom-up and can run on the same two inputs without external boxes. Build
it after the YOLO comparison has been reviewed. The image compiles OpenPose from the
pinned source commit without downloading model weights during the build. It does not
require a GPU reservation. By default the Dockerfile uses eight compiler jobs rather
than all 64 server CPUs:

```bash
cd /home/vincent/projects/Bremen
git pull --ff-only

docker build \
  --build-arg BUILD_JOBS=8 \
  -f containers/Dockerfile.openpose \
  -t vincent/football-pose-openpose:dev .

docker run --rm vincent/football-pose-openpose:dev --help
```

The standard BODY_25 weight is a separate private-server input so its checksum is
recorded with each job. OpenPose's legacy CMU model host is currently unavailable.
Download the file directly on Lyra from this commit-pinned Hugging Face mirror. The
mirror file was verified against the MD5 embedded in OpenPose v1.7.0 and against its
published SHA-256:

```bash
mkdir -p /home/vincent/football-pose-private/checkpoints

curl --fail --location --retry 20 \
  --output /home/vincent/football-pose-private/checkpoints/pose_iter_584000.caffemodel \
  'https://huggingface.co/camenduru/openpose/resolve/f5bb0c0a16060ac8b373472a5456c76bd68eb202/models/pose/body_25/pose_iter_584000.caffemodel?download=true'
```

Verify the file on Lyra. The MD5 value pinned by OpenPose v1.7.0 is shown below; the
experiment manifest additionally records SHA-256:

```bash
md5sum /home/vincent/football-pose-private/checkpoints/pose_iter_584000.caffemodel
sha256sum /home/vincent/football-pose-private/checkpoints/pose_iter_584000.caffemodel
```

Expected hashes:

```text
MD5     78287b57cf85fa89c03f1393d368e5b7
SHA-256 44e3d7ebd8c8b62d4366d67127f1b562611a9e8fd0f4f3cdeeb4bb4a6ed12be6
```

Delete and download the file again if either hash differs. Do not use an unverified
checkpoint in an experiment.

Validate the matched configurations:

```bash
python -m football_pose validate-config configs/lyra-openpose-full-frame.yaml
python -m football_pose validate-config configs/lyra-openpose-tiled.yaml
```

Both configurations fix BODY_25 network resolution at `-1x368` and currently use
physical GPU 7 as their default. After reserving GPU 7 in the shared semaphore,
run them serially:

```bash
export CUDA_VISIBLE_DEVICES=7

python -m football_pose run configs/lyra-openpose-full-frame.yaml
python -m football_pose run configs/lyra-openpose-tiled.yaml
```

They reuse the existing full-frame and tiled artifacts when the source video and
preprocessing implementation are unchanged. Compare raw records and runtime, while
remembering that overlapping-tile records have not yet been deduplicated.

Do not use a full-frame HRNet run to judge whether HRNet is better at finding players.
HRNet-W32 is top-down and requires a person box. Its first valid test comes after a
fixed detector/tracker has produced the shared crop artifact used by all three pose
models. The complete experiment structure is defined in `experiment.md`.

## 15. Screen other full-frame preprocessing methods

Pull the preprocessing-screen configurations and verify the CPU contracts before
reserving a GPU:

```bash
cd /home/vincent/projects/Bremen
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m pytest -q

for config in configs/lyra-preprocess-*.yaml
do
  python -m football_pose validate-config "$config"
done
```

The expected test result for this revision is `38 passed`. The four configurations
each apply exactly one full-frame preprocessing change: CLAHE, gamma 0.8, gamma 1.2,
or mild unsharp masking. Every configuration materializes one lossless artifact and
then runs YOLO Pose followed by OpenPose with unchanged model settings. No container
rebuild is required because these processors run in the host orchestrator.

After confirming and reserving physical GPU 7 in the shared semaphore, start the
screen inside `tmux`:

```bash
tmux new -s preprocessing-screen

cd /home/vincent/projects/Bremen
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=7

for config in \
  configs/lyra-preprocess-clahe.yaml \
  configs/lyra-preprocess-gamma-darken.yaml \
  configs/lyra-preprocess-gamma-brighten.yaml \
  configs/lyra-preprocess-unsharp.yaml
do
  python -m football_pose run "$config"
done
```

Detach with `Ctrl-b`, then `d`; reconnect with `tmux attach -t preprocessing-screen`.
Do not add tiling to these first screening runs: each result must remain attributable
to one isolated preprocessing step. Release GPU 7 after the loop completes or fails.

Compare each model's record count and model runtime against its full-frame baseline.
Do not promote a step into a tiled or combined pipeline until its predictions have
also passed visual inspection and overlapping-tile results have been deduplicated.

## 16. Later: run the full serial, multi-GPU experiment

Only continue after the staged one-GPU tests pass.

1. Copy the one-GPU YAML to a new, descriptive full-run YAML.
2. Change the input and output paths.
3. Reserve the intended physical GPUs in the semaphore.
4. Put exactly those IDs in every model's `devices` list.
5. Export the identical comma-separated IDs.
6. Start the run inside `tmux` so an SSH disconnect does not stop it.

Example only, assuming GPUs 2, 3, 4, and 5 were explicitly reserved:

```bash
tmux new -s thesis-pose

cd /home/vincent/projects/Bremen
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=2,3,4,5

python -m football_pose run configs/server-full-run.yaml
```

Detach from `tmux` with `Ctrl-b`, then `d`. Reconnect later with:

```bash
tmux attach -t thesis-pose
```

The orchestrator runs YOLO Pose, HRNet-W32, and OpenPose serially. Within each model, work is sharded across all configured GPUs. CUDA out-of-memory failures automatically restart the complete model attempt with half the batch size until `min_batch_size` is reached.

## 17. Find outputs and logs

For the suggested external results root:

```text
/home/vincent/football-pose-results/EXPERIMENT/jobs/JOB_ID/archive/predictions.parquet
/home/vincent/football-pose-results/EXPERIMENT/jobs/JOB_ID/archive/manifest.json
/home/vincent/football-pose-results/EXPERIMENT/jobs/JOB_ID/runner/
/home/vincent/football-pose-results/EXPERIMENT/experiments/EXPERIMENT_ID/summary.json
```

List recent outputs:

```bash
find /home/vincent/football-pose-results -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' \
  | sort -r \
  | head -50
```

Inspect a cached artifact using the path printed by the preparation command:

```bash
python -m football_pose inspect-artifact \
  /home/vincent/football-pose-cache/artifacts/ARTIFACT_ID \
  --count-frames
```

Copy important results back to the Mac from the Mac Terminal:

```bash
scp -r \
  vincent@lyra:/home/vincent/football-pose-results/EXPERIMENT \
  /local/backup/destination/
```

## 18. Save server-side code changes

Before editing:

```bash
cd /home/vincent/projects/Bremen
git status --short --branch
git pull --ff-only
```

After editing, review and commit only intended files:

```bash
git status --short
git diff
git add PATH_TO_FILE
git commit -m "Describe the change"
git push
```

Never use `git add .` when private data or generated outputs may be present. Never commit model weights, source footage, secrets, cache artifacts, Parquet results, or runner logs.

## 19. Daily start and finish checklist

At the start:

```bash
ssh vincent@lyra
cd /home/vincent/projects/Bremen
git status --short --branch
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export TMPDIR=/home/vincent/.cache/football-pose/pip-tmp
export PIP_CACHE_DIR=/home/vincent/.cache/pip
export YOLO_CONFIG_DIR=/home/vincent/.cache/football-pose/ultralytics
```

Before GPU work:

- Check and update the semaphore.
- Export only reserved physical GPU IDs.
- Confirm YAML `devices` contains the same IDs.
- Confirm output and cache free space.
- Start with the smallest reproducible input.

At the end:

- Confirm results and manifests exist.
- Copy irreplaceable results to backed-up storage.
- Release all GPU reservations, including after a failure.
- Commit and push intended source/configuration changes.
- Leave cache and shared Docker storage intact unless the administrator authorizes cleanup.

## Troubleshooting

### `Permission denied` during SSH

Check the username, VPN, password, SSH key, and whether Lyra requires a specific domain name.

### `docker info` reports a permission error

Do not use `sudo` as a workaround. Ask the administrator for the supported Docker or Apptainer workflow.

### `No space left on device` while pip downloads PyTorch

Confirm `tempfile.gettempdir()` is not `/tmp`, then retry with the home-based
temporary directory:

```bash
export TMPDIR=/home/vincent/.cache/football-pose/pip-tmp
export PIP_CACHE_DIR=/home/vincent/.cache/pip
python -c "import tempfile; print(tempfile.gettempdir())"
python -m pip install --resume-retries 20 -r requirements/orchestrator.txt
```

Do not delete shared Docker data to solve a pip temporary-file problem.

### Ultralytics says its config directory is not writable

For the host environment, export the user-owned directory documented in section 6.
The same warning inside a disposable YOLO container is harmless because it falls back
to `/tmp/Ultralytics`.

### YOLO image build cannot install `numpy==2.3.5`

Pull commit `677cec8` or later. The CUDA Ubuntu 22.04 image has Python 3.10 and must
select the conditional NumPy 2.2.6 pin from `requirements/yolo.txt`.

### GPU reservation error from the orchestrator

The physical IDs in YAML do not match `CUDA_VISIBLE_DEVICES`. Compare both with the semaphore reservation.

### CUDA out of memory

The runner automatically halves its batch size. If it reaches the minimum, reduce the configured starting batch or test on a shorter input. Do not take another GPU without reserving it.

### Low GPU utilization

Check CPU use, RAM, swap, `/lyra` free space, and artifact I/O. Preprocessing or storage may be starving the model.

### Interrupted SSH session

Use `tmux` for long runs. Completed jobs and artifacts are resumable; rerun the same validated configuration rather than deleting outputs.

### Stale cache lock

First confirm that no process is writing that exact artifact. Do not delete lock files merely because a job is slow. If the original process is confirmed dead, record the exact error before carefully removing only the corresponding stale lock.

## Related project documentation

- [README.md](README.md) - project setup and modular-pipeline overview
- [experiment.md](experiment.md) - experiment stages, comparisons, and fairness rules
- [docs/architecture.md](docs/architecture.md) - contracts, artifacts, recovery, and GPU scheduling
- [configs/lyra-yolo-full-frame.yaml](configs/lyra-yolo-full-frame.yaml) - full-frame YOLO tiling baseline
- [configs/lyra-yolo-tiled.yaml](configs/lyra-yolo-tiled.yaml) - matched deterministic-tiling YOLO run
- [configs/lyra-openpose-full-frame.yaml](configs/lyra-openpose-full-frame.yaml) - full-frame OpenPose baseline
- [configs/lyra-openpose-tiled.yaml](configs/lyra-openpose-tiled.yaml) - matched deterministic-tiling OpenPose run
- [configs/lyra-yolo-one-gpu.yaml](configs/lyra-yolo-one-gpu.yaml) - tracked YOLO-only Lyra smoke-test template
- [configs/server-three-models.yaml](configs/server-three-models.yaml) - tracked three-model configuration template
- [TODOS.md](TODOS.md) - deferred evaluation work

OpenPose retains its upstream academic/non-commercial licensing restrictions. Confirm the thesis use and any distribution of images, weights, or outputs complies with all third-party terms.
