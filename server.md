# Lyra server runbook

This is the durable reference for connecting to Lyra, locating the project, testing it safely, and running thesis experiments. Update the path table whenever a location changes.

## Saved connection and repository details

| Item | Saved value |
| --- | --- |
| Server hostname | `lyra` |
| Server user | `vincent` |
| Server home | `/home/vincent` |
| Server repository | `/home/vincent/projects/Bremen` |
| GitHub repository | `https://github.com/vin1penny/Bremen.git` |
| Working branch | `codex/modular-football-pipeline` |
| Local Mac repository | `/Users/larry/Bremen/FootballTrackingDataGeneration-main` |
| Regenerable server cache | `/lyra/cache/vincent/football-pose/artifacts` |
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

The CUDA number printed by `nvidia-smi` describes the driver's maximum compatible CUDA version. Each model container supplies its own pinned CUDA runtime.

## 1. Connect to Lyra

Connect to the university VPN first if Lyra is not reachable from the current network. From the Mac Terminal:

```bash
ssh vincent@lyra
```

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
mkdir -p /home/vincent/football-pose-results
mkdir -p /lyra/cache/vincent/football-pose/artifacts
chmod 700 /home/vincent/football-pose-private
```

| Data | Location | Git-backed? | Regenerable? |
| --- | --- | --- | --- |
| Source code and experiment YAML | `/home/vincent/projects/Bremen` | Yes | Yes |
| Private footage | `/home/vincent/football-pose-private/footage` | No | Usually no |
| Unique checkpoints | `/home/vincent/football-pose-private/checkpoints` | No | Usually no |
| Parquet, manifests, logs | `/home/vincent/football-pose-results` | No | Expensive to reproduce |
| Preprocessed artifacts | `/lyra/cache/vincent/football-pose/artifacts` | No | Yes |

Check space before large jobs:

```bash
df -h /home/vincent /lyra/cache
quota -s
docker system df
```

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

## 6. Create the Python environment

On Lyra, from the repository root:

```bash
cd /home/vincent/projects/Bremen
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  -r requirements/dev.txt \
  -r requirements/orchestrator.txt

export PYTHONPATH="$PWD/src"
```

Reactivate this environment after reconnecting:

```bash
cd /home/vincent/projects/Bremen
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

## 7. Run the safe CPU checks

These commands do not require a GPU reservation:

```bash
python -m pip check
pytest -q
python -m football_pose list-processors
python -m football_pose validate-config configs/mock.yaml
python -m football_pose run configs/mock.yaml
python -m football_pose run configs/mock.yaml
```

Expected unit-test result at the time this runbook was created:

```text
16 passed
```

The second mock run should report a cache hit and reuse the completed mock-model job. This checks video decoding, preprocessing, caching, runner orchestration, validation, Parquet output, and resume behavior without using CUDA.

## 8. Build the model containers

Container builds do not require a GPU reservation, but they use CPU, network, Docker storage, and time. Check server policy and disk space first.

```bash
docker build \
  -f containers/Dockerfile.yolo \
  -t football-pose-yolo .

docker build \
  -f containers/Dockerfile.hrnet \
  -t football-pose-hrnet .
```

The OpenPose Dockerfile currently compiles with `make -j$(nproc)`, which can use all 64 server CPUs. Do not run that build unchanged until this resource use is approved or the Dockerfile has been changed to support a safe build limit.

Verify images without changing shared Docker state:

```bash
docker image ls | grep football-pose
```

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
df -h /home/vincent /lyra/cache
```

## 10. Create a one-GPU test configuration

Copy the tracked template:

```bash
cp configs/server-three-models.yaml configs/server-one-gpu-test.yaml
```

Edit `configs/server-one-gpu-test.yaml` and use absolute paths such as:

```yaml
input: /home/vincent/football-pose-private/footage/test-video.mp4
output_dir: /home/vincent/football-pose-results/one-gpu-test

cache:
  root: /lyra/cache/vincent/football-pose/artifacts
```

Use these checkpoint locations:

```yaml
detector:
  checkpoint: /home/vincent/football-pose-private/checkpoints/player.pt
```

```yaml
checkpoint: /home/vincent/football-pose-private/checkpoints/yolov8x-pose.pt
```

```yaml
checkpoint: /home/vincent/football-pose-private/checkpoints/hrnet-w32.pth
```

For a reserved physical GPU such as GPU 2, every model must use:

```yaml
devices: [2]
```

Use conservative test batches:

- YOLO Pose: `batch_size: 8`
- HRNet-W32: `batch_size: 8`
- OpenPose: `batch_size: 1`
- All: `min_batch_size: 1`

The example GPU number is not a recommendation. Use only the ID currently assigned in the semaphore.

## 11. Validate and prepare the short experiment

After reserving one GPU, export the same physical ID. Example only:

```bash
export CUDA_VISIBLE_DEVICES=2
```

Validate all configured paths before doing expensive work:

```bash
python -m football_pose validate-config configs/server-one-gpu-test.yaml
```

Prepare and cache the processed short video:

```bash
python -m football_pose run \
  configs/server-one-gpu-test.yaml \
  --prepare-only
```

The crop preprocessor uses the reserved GPU, so preparation also requires a reservation when `crop` is enabled.

## 12. Test each real model on one GPU

Run the three models separately on the short cached artifact:

```bash
python -m football_pose run \
  configs/server-one-gpu-test.yaml \
  --model yolo-pose

python -m football_pose run \
  configs/server-one-gpu-test.yaml \
  --model hrnet-w32

python -m football_pose run \
  configs/server-one-gpu-test.yaml \
  --model openpose-body25
```

Require the following before scaling up:

- The command exits successfully.
- The summary reports `"success": true`.
- The model job reports `COMPLETE`.
- `predictions.parquet` and `manifest.json` exist.
- Shard stderr logs contain no unexplained errors.

## 13. Run the full serial, multi-GPU experiment

Only continue after all one-GPU model tests pass.

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

## 14. Find outputs and logs

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
  /lyra/cache/vincent/football-pose/artifacts/ARTIFACT_ID \
  --count-frames
```

Copy important results back to the Mac from the Mac Terminal:

```bash
scp -r \
  vincent@lyra:/home/vincent/football-pose-results/EXPERIMENT \
  /local/backup/destination/
```

## 15. Save server-side code changes

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

## 16. Daily start and finish checklist

At the start:

```bash
ssh vincent@lyra
cd /home/vincent/projects/Bremen
git status --short --branch
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
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
- [docs/architecture.md](docs/architecture.md) - contracts, artifacts, recovery, and GPU scheduling
- [configs/server-three-models.yaml](configs/server-three-models.yaml) - tracked three-model configuration template
- [TODOS.md](TODOS.md) - deferred evaluation work

OpenPose retains its upstream academic/non-commercial licensing restrictions. Confirm the thesis use and any distribution of images, weights, or outputs complies with all third-party terms.
