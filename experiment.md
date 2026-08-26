# Master thesis experiment

## Objective

Find the preprocessing pipeline that produces the best human-pose estimates from
wide-angle football video. We compare pipelines, not only isolated model accuracy.
An improvement in player detection or crop quality therefore counts as a pipeline
benefit.

## Fixed experiment flow

```text
video
  -> ordered preprocessing configuration
  -> lossless, content-addressed artifact
  -> YOLO Pose / HRNet-W32 / OpenPose
  -> common COCO-17 records in original-frame coordinates
  -> evaluation metrics
```

Each YAML file defines one pipeline. Processor order is execution order. Changing one
processor or parameter creates a different pipeline identifier and cached artifact.
Models run serially, while each model may distribute the artifact across its reserved
GPUs.

## Models

- **YOLO Pose:** official pretrained checkpoint; supports multi-person input.
- **HRNet-W32:** official pretrained top-down checkpoint; requires one person crop.
- **OpenPose BODY_25:** official pretrained checkpoint; BODY_25 output is mapped to
  COCO-17.

The pose checkpoints are standard published weights, not football-specific
fine-tuning. Exact checkpoint hashes, package versions, commands, and parameters are
stored with every run.

## Experiment stages

### 1. Full-frame baseline

Run the unmodified source frames first. This records how each compatible model behaves
without image preprocessing. Inference image size and confidence are fixed and
recorded; they are not silently changed between comparisons.

### 2. Deterministic tiling comparison

Compare the full-frame baseline with a fixed row-by-column tile grid. Tiling is a
normal preprocessing step configured as:

```yaml
processors:
  - type: tile
    params:
      rows: 2
      columns: 2
      overlap_ratio: 0.10
```

Tiles are produced in row-major order. Their IDs, bounds, overlap, and transforms back
to the source frame are stored in the artifact. `overlap_ratio` is the approximate
fraction of a nominal tile shared by neighboring tiles. The full-frame and tiled runs
keep the model checkpoint, inference size, confidence, and video constant.
Overlapping-tile duplicates remain identifiable by tile ID and must be fused in source
coordinates before final metrics are calculated.

This first comparison uses models that can process multi-person regions directly.
It determines whether tiling is useful before adding it to more expensive combined
pipelines.

### 3. Shared learned-crop comparison

A trained football player detector and tracker generates bounding boxes once for a
pipeline. Those boxes are converted into a lossless crop artifact. All three pose
models receive exactly the same crops:

```text
processed frame -> detector/tracker -> shared player crops
                                      -> YOLO Pose
                                      -> HRNet-W32
                                      -> OpenPose
```

HRNet uses its top-down structure correctly because each input contains one bounded
person. YOLO Pose and OpenPose also receive the same crops, so any benefit from learned
cropping is applied equally. The detector/tracker checkpoint, threshold, classes,
padding, and tracking settings are part of the pipeline configuration and provenance.

Tiling and learned cropping are kept separate in the first screening runs. If tiling
is beneficial, tiled detector results will be projected to the source frame, duplicate
boxes fused, and temporal tracking applied before the common crop artifact is created.
We do not treat sequential per-tile tracker calls as independent video frames.

### 4. Preprocessing search

Evaluate individual steps before testing combinations. Candidate steps include:

- resize or full-resolution input
- CLAHE
- gamma correction
- NLM or bilateral denoising
- motion deblurring
- super-resolution
- deterministic tiling
- learned tracked cropping

The first full-frame screening batch uses four single-factor configurations on the
same 30-second clip. Each artifact is shared by YOLO Pose and OpenPose, which run
serially with unchanged model settings:

| Configuration | Isolated change | Purpose |
| --- | --- | --- |
| `lyra-preprocess-clahe.yaml` | CLAHE, clip 2.0 | Normalize local contrast across sun and shadow |
| `lyra-preprocess-gamma-darken.yaml` | gamma 0.8 | Recover contrast in bright regions |
| `lyra-preprocess-gamma-brighten.yaml` | gamma 1.2 | Lift players located in shadow |
| `lyra-preprocess-unsharp.yaml` | mild unsharp mask | Strengthen small player edges before model resizing |

After reserving GPU 7, run the configurations serially:

```bash
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

Each command materializes one lossless artifact and then runs YOLO Pose followed by
OpenPose. Release GPU 7 after the final command completes or immediately after a
failure. No container rebuild is required for these host-side preprocessing changes.

The initial video inspection showed strong local illumination differences but little
obvious sensor noise or consistent motion-blur direction. Denoising and configured
motion deblurring therefore remain second-tier screens rather than being mixed into
the first batch. Super-resolution remains separate because it requires a pinned model
checkpoint and changes both computational cost and image scale.

Promote only beneficial individual steps into combination experiments. Every promoted
combination is compared against its direct parent pipeline so the contribution of each
added step remains visible.

### 5. Final confirmation

Run the best candidate pipelines on the complete evaluation footage with all three
models where the input contract is valid. Preserve the YAML, artifact manifest,
checkpoint hashes, runner logs, prediction Parquet files, and final metrics.

## Outputs and result files

Every configuration writes to its configured `output_dir`. One experiment directory
contains the high-level report, while each model receives its own job directory:

```text
OUTPUT_DIR/
├── experiments/EXPERIMENT_ID/
│   └── summary.json
└── jobs/JOB_ID/
    ├── job.json
    ├── archive/
    │   ├── manifest.json
    │   └── predictions.parquet
    └── runner/
        ├── predictions.jsonl
        └── attempt-01-batch-N/
            ├── shard-000.jsonl
            ├── shard-000.stdout.log
            └── shard-000.stderr.log
```

The IDs prevent results from different pipelines and model settings from colliding.
A two-model configuration therefore produces two `JOB_ID` directories: one for YOLO
Pose and one for OpenPose.

| File | Meaning | Use |
| --- | --- | --- |
| `summary.json` | Exact high-level JSON also printed by the CLI | First file to inspect; configuration, timings, jobs, record counts, and success |
| `job.json` | Latest execution state for one model | Attempts, model ID, batch size, errors, timings, and output paths |
| `archive/predictions.parquet` | Canonical compact prediction table | Primary input for evaluation and statistical analysis |
| `archive/manifest.json` | Archive schema and provenance | Reproducibility, checkpoint and artifact metadata, and validation |
| `runner/predictions.jsonl` | Merged raw runner predictions | Human-readable debugging and conversion source |
| `runner/attempt-*/shard-*.jsonl` | Predictions produced by one execution shard | Diagnose sharding and merge behavior |
| `runner/attempt-*/*.log` | Model-container standard output and errors | Diagnose warnings, crashes, CUDA errors, and dependency problems |

`summary.json` is written atomically, but rerunning the same experiment replaces that
experiment's previous summary with the latest invocation. Preserve invocation-level
reports in a timestamped external folder when cold-cache and warm-cache runs must both
remain available. The Parquet archive is canonical; JSONL is intentionally retained
because it is convenient for auditing. macOS `.DS_Store` files are Finder metadata and
are unrelated to the experiment.

The result download does not contain the usually much larger preprocessed frame
artifact. Artifacts remain under the configured cache root on Lyra and can be
regenerated from the source video, preprocessing configuration, and implementation.
Important downloaded results should retain at least the configuration YAML,
`summary.json`, `job.json`, and the complete `archive/` directory. Logs should also be
kept for thesis auditability.

### Interpreting `records`

One record represents one person-pose prediction, not one video frame and not one
correct detection. Dividing records by the 782 source frames gives raw predictions per
frame. This is a detection-yield diagnostic, not an accuracy metric: false positives
increase it, missed players decrease it, and overlapping tiles can produce duplicate
records. Pose quality can only be established through visual inspection and later
ground-truth metrics.

Current 30-second screening results are:

| Input pipeline | YOLO records | OpenPose records | OpenPose records/frame | OpenPose change from full frame |
| --- | ---: | ---: | ---: | ---: |
| Full frame, no preprocessing | 0 | 826 | 1.06 | reference |
| CLAHE | 0 | 1,154 | 1.48 | +39.7% |
| Gamma 0.8 | 0 | 700 | 0.90 | -15.3% |
| Gamma 1.2 | 0 | 797 | 1.02 | -3.5% |
| Unsharp mask | 0 | 933 | 1.19 | +13.0% |
| 2 x 2 tiling, 10% overlap | 451 | 5,989 | 7.66 | +625.1% raw |

CLAHE is currently the strongest non-tiling transformation by raw OpenPose yield, and
unsharp masking is the only other full-frame transformation that increased it. Neither
gamma variant is promoted. YOLO remaining at zero for every full-frame pixel-level
transformation supports the player-scale bottleneck hypothesis. Tiling remains the
largest gain, but its records must be deduplicated in source-frame coordinates before
being interpreted as detections or compared with ground truth.

## Fair-comparison rules

1. Use the same source frames and timestamps.
2. Change only the factor named by the comparison.
3. Keep model weights and inference parameters fixed within an ablation.
4. Reuse one shared crop artifact across models in learned-crop comparisons.
5. Map every prediction back to original-video coordinates.
6. Report missing detections, including zero-record runs; do not discard them.
7. Record preprocessing, model, and end-to-end runtime separately.
8. Reserve GPUs according to the Lyra semaphore and record the physical device IDs.

## Code map

- `configs/`: one reproducible YAML file per pipeline.
- `src/football_pose/preprocessing/`: interchangeable processing steps and registry.
- `src/football_pose/preprocessing/tiling.py`: deterministic tile generation.
- `src/football_pose/preprocessing/cropping.py`: learned detection, tracking, and crops.
- `src/football_pose/artifacts.py`: immutable lossless inputs shared by the models.
- `runners/`: isolated YOLO Pose, HRNet-W32, and OpenPose adapters.
- `experiment-output/` or the configured server results directory: manifests, logs,
  JSONL, and Parquet results.

The metric implementation follows after the ground-truth format, camera projection,
and timestamp synchronization are confirmed.
