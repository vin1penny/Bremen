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
  -> cross-tile duplicate removal and pitch-region classification
  -> raw records + on-pitch records + auditable decisions
  -> evaluation metrics
```

Each YAML file defines one pipeline. Processor order is execution order. Changing one
processor or parameter creates a different pipeline identifier and cached artifact.
Models run serially, while each model may distribute the artifact across its reserved
GPUs.

The main experiment does not downscale full-frame model inputs. YOLO Pose uses the
fixed target `--imgsz 1920`; the 1920x1080 source therefore retains its spatial
resolution, while a half-frame tile is enlarged to the same target so tiling can
actually increase player scale. OpenPose uses the matching stride-compatible network
resolution, `1920x1088`. Earlier runs at YOLO 640 and OpenPose `-1x368` are retained
as pilot results and are not mixed with the main full-resolution comparisons.

## Models

- **YOLO Pose:** official pretrained checkpoint; supports multi-person input.
- **HRNet-W32:** official pretrained top-down checkpoint; requires one person crop.
- **OpenPose BODY_25:** official pretrained checkpoint; BODY_25 output is mapped to
  COCO-17.

The pose checkpoints are standard published weights, not football-specific
fine-tuning. Exact checkpoint hashes, package versions, commands, and parameters are
stored with every run.

The separate pitch-localization model is a trained football-pitch model. It is not a
pose model and does not change any pose estimate. It runs once per source video and
settings combination, and its checkpoint hash and geometry cache ID are recorded in
the experiment summary. This keeps crowd filtering identical across YOLO Pose,
OpenPose, and later HRNet comparisons.

## Pitch-aware filtering

As of 2026-09-25, the pitch template is explicitly **Weserstadion, Bremen,
105 × 68 metres** (centimetre coordinates, corner origin). Fixed-size markings
retain their physical dimensions: penalty depth 16.5 m, goal area depth 5.5 m,
penalty spot 11 m and centre-circle radius 9.15 m. Other venues require a different
template. This changes calibration and invalidates the previous geometry cache.
Each frame now archives landmark IDs, pixel coordinates and confidence values.
Magenta P0–P31 labels show predictions with confidence ≥0.5 independently of fit
acceptance; the cyan polygon shows the fitted outer boundary.

Pitch training follows the pitch section of `train/train_remote.ipynb`, available
as `train/train_pitch_server.py` for tmux. Use the version-15 dataset and preserve
each training run, checkpoint hash and `pitch-validation.json`. Select using the
saved checkpoint's pose validation metrics; box accuracy is not the selection
criterion. Inspect the new checkpoint on the video before making its versioned
path the production checkpoint. Training and evaluation splits must remain separate.
The production candidate uses the settings that previously achieved strong landmark
validation: YOLOv8n-pose, 640 pixels, batch 2, 50 epochs, and mosaic disabled. A
20-epoch batch-16 server run produced pose mAP50 0.322 and is explicitly rejected.

Correction (2026-09-24): the rectangular fallback is withdrawn. Pitch inference is
performed on every original frame; only a validated transform from that same frame
can classify poses. All four pitch boundaries are tested, and the video draws the
projected pitch polygon clipped to the image. Missing or invalid geometry produces
`unavailable` decisions (yellow), never an assumed on-pitch result. Geometry is not
copied across frames because the camera can move or cut. The detection box is kept
only as diagnostic metadata. Old rectangle-filter results are not valid pitch metrics.

Checkpoint correction: `models/pitch_detection_model_best/best.pt` was exported using
box mAP as the selection criterion and failed the geometry checks on five sampled
frames. `models/pitch_detection_model/weights/best.pt` passed those same five checks.
Use the latter as `pitch-landmarks-best.pt` on Lyra. Its SHA-256 is
`dd216396a9ba8461445e8ddfbafb98f3d9ed45fe99d48033725c85297bced2f3`.
Full-video boundary accuracy and usable-frame coverage still need server validation;
unavailable poses remain preserved in raw predictions.

Raw pose output is always preserved. After inference, the pipeline uses the detected
pitch region to classify the estimated ground point of each person. The ground point
is the midpoint of confident ankle keypoints, one ankle when only one is reliable, or
the bottom-center of that person's bounding box as a fallback.

The pitch model supplies landmarks for each frame. Enough consistent landmark
correspondences are required to estimate the image-to-pitch homography. Homographies
that fail the configured inlier-ratio or reprojection-error checks are rejected;
the model's rectangular bounding box is not used for filtering. Failed geometry is
marked unavailable. Accepted
transforms test both touchlines and both goal lines in pitch coordinates.

Before pitch classification, predictions from different overlapping tiles are
deduplicated by person-box IoU in original-frame coordinates. Predictions produced
inside the same source region are never merged by this step. Each raw prediction is
retained in the decision table as `inside`, `outside`, `duplicate`, or `unavailable`.
Only `inside` records enter the separate on-pitch archive. This makes the automatic
filter reversible and measurable instead of silently deleting detections.

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
serially with unchanged full-resolution model settings:

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

python -m football_pose build-overview \
  /home/vincent/football-pose-results/native-resolution \
  --model yolo-pose \
  --model openpose-body25
```

Each command materializes one lossless artifact and then runs YOLO Pose followed by
OpenPose. Release GPU 7 after the final command completes or immediately after a
failure. No container rebuild is required for these host-side preprocessing changes.
The final command regenerates the objective record-count table at
`/home/vincent/football-pose-results/native-resolution/results-overview/records.md`.

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
    │   ├── predictions.parquet
    │   └── pitch-filter/FILTER_SETTINGS_ID/
    │       ├── manifest.json
    │       ├── pitch-decisions.parquet
    │       └── predictions-on-pitch.parquet
    └── runner/
        ├── predictions.jsonl
        └── attempt-01-batch-N/
            ├── shard-000.jsonl
            ├── shard-000.stdout.log
            └── shard-000.stderr.log

/mnt/storage2/vincent/football-pose/videos/output/
└── CONFIGURATION_NAME/EXPERIMENT_ID/
    └── MODEL_ID-JOB_ID-VIDEO_SETTINGS_ID.mp4
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
| `pitch-filter/*/predictions-on-pitch.parquet` | Deduplicated poses classified inside the pitch | Input for on-pitch player metrics |
| `pitch-filter/*/pitch-decisions.parquet` | One decision for every raw pose | Audit inside/outside/duplicate/unavailable classifications, anchors, and methods |
| `pitch-filter/*/manifest.json` | Filter settings, geometry path, and counts | Reproduce and interpret the filtered archive |
| `runner/predictions.jsonl` | Merged raw runner predictions | Human-readable debugging and conversion source |
| `runner/attempt-*/shard-*.jsonl` | Predictions produced by one execution shard | Diagnose sharding and merge behavior |
| `runner/attempt-*/*.log` | Model-container standard output and errors | Diagnose warnings, crashes, CUDA errors, and dependency problems |
| Annotated `.mp4` | Processed frames with raw boxes and COCO-17 skeletons | Visual quality control for one model and one pipeline configuration |

Every tracked Lyra configuration enables `video_output`. The renderer reconstructs
the cached processed packets at the original video resolution, averages overlapping
tiles, and then overlays predictions in original-frame coordinates. Tile or crop
boundaries are drawn in gray. With pitch filtering enabled, on-pitch poses are green,
outside-pitch poses red, cross-tile duplicates gray, and unavailable classifications
yellow. The cyan polygon shows the accepted frame-specific pitch region. The upper-left
label reports raw, on-pitch, outside, duplicate, and unknown counts. Videos contain no
audio and do not replace the Parquet predictions or later quantitative metrics.

The configuration name, experiment ID, job ID, and video-settings ID in the path keep
videos from different processing pipelines, checkpoints, and rendering settings from
colliding. A configuration containing YOLO Pose and OpenPose therefore creates two
MP4 files. The same mechanism also creates an HRNet video when that model is added.
The resulting path is stored as `jobs[].outputs.video` in both `job.json` and
`summary.json`; render duration is stored as `jobs[].timings.video_render_seconds`.

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
kept for thesis auditability. Annotated videos are reproducible review artifacts and
are stored separately on the mounted NFS volume because they are comparatively large.

### Interpreting `records`

One raw record represents one person-pose prediction, not one video frame and not one
correct detection. `on_pitch_records` is the automatic subset after cross-tile
deduplication and pitch filtering. Dividing either count by the 782 source frames gives
predictions per frame. Both remain detection-yield diagnostics, not accuracy metrics:
false positives can still increase them and missed players decrease them. Pose quality
can only be established through visual inspection and later ground-truth metrics.

Current 30-second screening results are:

An objective baseline-relative table can be regenerated from saved summaries using the
instructions in [`results-overview/README.md`](results-overview/README.md).

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
9. Keep the pitch checkpoint and filter thresholds fixed within every comparison.

## Code map

- `configs/`: one reproducible YAML file per pipeline.
- `src/football_pose/preprocessing/`: interchangeable processing steps and registry.
- `src/football_pose/preprocessing/tiling.py`: deterministic tile generation.
- `src/football_pose/preprocessing/cropping.py`: learned detection, tracking, and crops.
- `src/football_pose/artifacts.py`: immutable lossless inputs shared by the models.
- `src/football_pose/pitch_filter.py`: cached pitch geometry, cross-tile deduplication,
  ground-point classification, and filtered archives.
- `runners/`: isolated YOLO Pose, HRNet-W32, and OpenPose adapters.
- `experiment-output/` or the configured server results directory: manifests, logs,
  JSONL, and Parquet results.

The metric implementation follows after the ground-truth format, camera projection,
and timestamp synchronization are confirmed.
