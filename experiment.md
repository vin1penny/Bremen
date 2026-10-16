# Master thesis experiment

## Current YOLO26 Pose protocol (2026-10-01)

YOLO26x Pose is the active YOLO pose model for the **whole thesis pipeline**, not
just COCO. Full-frame, deterministic-tiling, CLAHE, gamma, unsharp, and one-GPU
smoke configurations all use `yolo26x-pose.pt`, the
`vincent/football-pose-yolo26:dev` container, `yolo26-pose` as the model ID,
and the native `--nms-free` inference head. The combined COCO/OpenPose check
uses it too. New football outputs have separate `yolo26-*` directories on the
mounted storage, preserving YOLOv8 pilot results. Compare processing steps
against the **YOLO26 full-frame baseline**, not an older YOLOv8 run.

The auxiliary pitch detector has 32 landmarks and uses the existing
`pitch-landmarks-best.pt` path. The supplied PitchVision YOLOv8x weights were
uploaded for evaluation, but their version-15 validation failed to match the
previous YOLOv8n checkpoint; they are not an accepted production upgrade.
It is an auxiliary geometry model, not one of the compared person-pose models.
It runs on the original video frames under the same checkpoint and settings for
every compared pose model and preprocessing condition. There is **no need to
retrain the pitch model** for the YOLO26 person-pose migration. Learned cropping
is separate again: it needs a validated player-detector checkpoint before use.
OpenPose and HRNet remain independent model families.
The pitch-weight comparison YAML files remain historical checks of earlier weights.

PitchVision checkpoint SHA-256:
`28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36`.
Its metadata records training on pitch dataset version 9 at 640 pixels. Use the
existing Ultralytics pose validation operation on our version-15 validation set
at 640 to measure its accuracy; this checkpoint is supplied already trained.
Report pose mAP50 and mAP50–95, then inspect pitch landmarks and estimated
boundaries on the experiment video. The high box scores visible in the supplied
screenshot do not establish keypoint accuracy.

The 2026-10-06 server recheck used the same version-15 validation set (34 images),
640 input size and batch 2 for both checkpoints:

| Pitch checkpoint | Pose mAP50 | Pose mAP50–95 |
| --- | ---: | ---: |
| Previous YOLOv8n, repeated validation | 0.984 | 0.605 |
| Supplied PitchVision YOLOv8x | 0.162 | 0.0413 |

These are checkpoint comparisons, not proof that the nano architecture is generally
better. The supplied x checkpoint records version-9 training; our n checkpoint
records version-15 training. The cause of the difference is not yet established.

Next, fine-tune `pitch-v8n-reference.pt` on the existing version-15 training split
(255 images) at 640, batch 8, for up to 100 additional epochs. Start a new optimizer
cycle with AdamW and learning rate 0.0001, mosaic disabled, seed 0 and deterministic
training. Early stopping patience is 20 under Ultralytics' validation-fitness
criterion. Keep the candidate and logs on mounted storage without overwriting the
reference or production weights. Validate the candidate at 640/batch 2, compare
pose metrics with the reference, and inspect landmarks and boundaries on the same
football video before promotion. The 28-image test split stays untouched until final
evaluation. A subsequent x fine-tuning run should use the same dataset and evaluation
protocol; more parameters or more epochs do not guarantee a better result.

The completed 100-epoch nano fine-tuning run, saved as
`pitch-v8n-finetune-6402`, reported pose mAP50 **0.995** and pose mAP50–95
**0.844** on the 34 validation images. Keep this checkpoint as the comparison
reference. The next candidate starts from that run's `weights/best.pt` for 20
additional epochs at 640/batch 8, using AdamW with learning rate 0.00001 and the
same dataset, seed and mosaic setting. The finished checkpoint has no optimizer
state, so this is a new fine-tuning cycle, not exact optimizer-state resumption.
Save separately and compare both candidates at 640/batch 2 before choosing one;
more training can also reduce validation performance. Production remains unchanged.

The additional 20 epochs were reported not to improve the candidate. Retain the
100-epoch `pitch-v8n-finetune-6402/weights/best.pt` as the selected checkpoint for
video review, installed through the existing `pitch-landmarks-best.pt` path.
The next check is the complete full-frame OpenPose pipeline on the same sample
video, including per-frame pitch landmarks, homography boundaries, pose filtering
and the annotated MP4. Inspect alignment throughout camera movement, not only the
validation-set score. Existing person-pose predictions can be reused; the pitch
checkpoint hash invalidates the relevant geometry and video caches automatically.

## COCO model setup validation

The completed first COCO run used YOLOv8x Pose and OpenPose BODY_25. The next
model setup check uses the official COCO-pretrained YOLO26x Pose checkpoint in
`configs/lyra-coco-yolo26.yaml`. It runs at 640 pixels with the YOLO26
`nms=False` head to match the published 71.6% COCO keypoint AP protocol. A
separate image pinned to Ultralytics 8.4.164 and a distinct model ID preserve
the earlier YOLOv8 result. The full-frame YOLO26 configuration is
`configs/lyra-yolo26-full-frame.yaml`; `configs/lyra-yolo-full-frame.yaml` is an
equivalent entry point. Both use the same checkpoint and inference head at 1920.

Before interpreting football results, evaluate the existing YOLO Pose and
OpenPose runners on COCO person keypoints val2017. This is a separate benchmark
path: original images → existing model containers → canonical COCO-17 predictions
→ official `pycocotools` OKS evaluation. No pitch filtering, tracking, cropping,
or image enhancement is applied. Original image IDs/sizes are retained, and
images with zero predictions remain in the evaluation denominator.

`evaluate-coco configs/lyra-coco-keypoints.yaml --limit 50` is a deterministic
smoke test (first 50 image IDs); omit `--limit` for the full supplied annotation
set. Outputs include AP/AP50/AP75, medium/large AP and corresponding AR,
raw detections, checkpoint and dataset hashes, settings, image IDs and logs.
Scores use 0–1, with -1 indicating an unavailable size/category result.
COCO evaluation uses the official keypoint sigmas and maxDets=20.

YOLO uses its box confidence for person ranking; OpenPose uses `poseScores`
and the existing BODY_25 → COCO-17 mapping. Canonical records now optionally
retain `person_score`; COCO evaluation requires it rather than inventing scores.
The initial comparison is YOLO 640/confidence 0.001/IoU 0.7 and OpenPose
single-scale -1x368. These must be aligned with each reference's exact protocol
before asserting that accuracy matches. HRNet is not yet included in the COCO
benchmark: its football runner uses our trained player detector, while any COCO
reference comparison must specify its box protocol (ground-truth boxes and
detected boxes are not equivalent).

Execution success is not an accuracy pass. A reference score, split, checkpoint,
input size, augmentation/NMS protocol and tolerance are needed for that decision.
Published results can use a different validation implementation from our shared
COCO evaluator. Keep the full benchmark distinct from football robustness tests.
Implementation: `src/football_pose/coco_evaluation.py`; server instructions and
download commands: `server.md`, “COCO keypoints setup check”.
The downloaded dataset, lossless artifact cache, raw predictions, and results
live under `/mnt/storage2/vincent/football-pose/`; only code and small
configuration files remain in the home-directory repository.

Evaluator: [official COCO API](https://github.com/cocodataset/cocoapi).
Model reference: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/).
YOLO26 reference: [Ultralytics pose documentation](https://docs.ultralytics.com/tasks/pose/).

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
  -> YOLO26 Pose / HRNet-W32 / OpenPose
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

- **YOLO26 Pose:** official COCO-pretrained checkpoint; supports multi-person input.
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
OpenPose and HRNet comparisons.
Active football configs run pitch landmark inference at `image_size: 640` on every
original video frame, then map the predicted coordinates back to that frame for
geometry fitting and rendering. This does not downscale the separate person-pose
inputs: OpenPose remains at `1920x1088` and YOLO Pose at its configured size.
The historical pitch-weight comparison configs remain pinned to 1280 and must not
be mixed with the new 640-pixel pitch-filter results without noting that change.

## Pitch-aware filtering

As of 2026-09-25, the pitch template is explicitly **Weserstadion, Bremen,
105 × 68 metres** (centimetre coordinates, corner origin). Fixed-size markings
retain their physical dimensions: penalty depth 16.5 m, goal area depth 5.5 m,
penalty spot 11 m and centre-circle radius 9.15 m. Other venues require a different
template. This changes calibration and invalidates the previous geometry cache.
Each frame now archives landmark IDs, pixel coordinates and confidence values.
Magenta P0–P31 labels show predictions with confidence ≥0.5 independently of fit
acceptance; the cyan polygon shows the fitted outer boundary.

Earlier pitch training followed `train/train_remote.ipynb` with a YOLOv8n backbone.
`train/train_pitch_server.py` retains that backbone as its default if pitch
training is ever repeated. The accepted checkpoint stays fixed during the
YOLO26 person-pose comparisons. Preserve its hash and `pitch-validation.json`;
rerunning with the same name can overwrite a training run directory. Select using the
saved checkpoint's pose validation metrics; box accuracy is not the selection
criterion. Inspect the new checkpoint on the video before making its versioned
path the production checkpoint. Training and evaluation splits must remain separate.
The historical high-resolution candidate selected on 2026-09-25 uses YOLOv8n-pose,
1280 pixels, batch 8, 100 epochs, and mosaic disabled. Compare it with the earlier
640-pixel, batch-2, 50-epoch run, which achieved strong landmark validation. A
640-pixel, 20-epoch, batch-16 server run produced pose mAP50 0.322 and is rejected.

Before selecting production weights, run `lyra-pitch-weights-earlier.yaml` and
`lyra-pitch-weights-new-1280.yaml`. Both use pitch inference at 1280 with identical
thresholds on the same video; their checkpoint and output paths are the only pitch
differences. Compare geometry coverage, reprojection diagnostics, classification
counts, and the labelled landmark videos. Validation-set scores and video behavior
are both reported; neither is replaced by raw pose-record yield.

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
This was the earlier `pitch-landmarks-best.pt` on Lyra, superseded by the
PitchVision YOLOv8x replacement above. Its SHA-256 is
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

### 4. Single-step preprocessing screening

Run `screen-preprocessing configs/lyra-preprocessing-screen.yaml --device GPU_ID`.
The suite uses `lyra-preprocessing-base.yaml` for the shared input, pose models,
pitch filter and rendering settings. It expands each condition to an ordinary
experiment YAML: baseline has no processors; every other condition has exactly one.

| Condition | Isolated operation |
| --- | --- |
| Baseline | Original frames |
| Resize x2 | Bicubic enlargement; scale control for super-resolution |
| CLAHE | Local luminance contrast, clip 2.0, grid 8 × 8 |
| Gamma darken / brighten | Separate runs at gamma 0.8 / 1.2 |
| NLM denoising | Colored non-local means, h/h_color 7, windows 7/21 |
| Bilateral denoising | Diameter 7, color/space sigma 50 |
| Unsharp mask | Kernel 5, sigma 1, amount 0.75 |
| Motion deblur | Wiener deconvolution, horizontal length-9 PSF, NSR 0.01 |
| Super-resolution | Pinned FSRCNN x2 checkpoint through OpenCV |
| Deterministic tiling | 2 × 2 regions, 10% overlap |
| Learned tracked cropping | Shared football-detector/ByteTrack crops, 10% padding |

This is **12 conditions × 3 models = 36 model jobs**: YOLO26 Pose, OpenPose and
HRNet-W32, run serially. Identity is represented by the unchanged baseline.
Segmentation, grayscale and dual-illumination estimation are not implemented
experimental conditions. Validate the HRNet container on Lyra before a full screen.

HRNet's fixed top-down input adapter uses our existing YOLO football detector
(`models/player_detection_model_best/best.pt`, uploaded as
`/mnt/storage2/vincent/football-pose/models/player-detection-best.pt`). It detects
people in each processed frame or tile, creates padded person crops, and runs
HRNet-W32 COCO pose weights on each crop. No generic MMDetection human detector is
downloaded or used. Detector settings and both checkpoint hashes are recorded as
part of the HRNet model protocol. YOLO detection weights supply boxes, not HRNet's
keypoint weights; its separate standard HRNet checkpoint remains necessary.

This mandatory adapter is constant within HRNet's comparisons, not an extra
candidate enhancement. Its baseline is detector + HRNet on unenhanced frames;
compare each family to its own baseline. In the explicit shared-crop condition,
all three models reuse the common crops and HRNet does not detect again. The
internal HRNet adapter does not advance ByteTrack separately for each tile; the
shared crop condition retains its existing once-per-video-frame tracking.
Cross-family comparisons must state these different full-frame input protocols.

The supplied crop detector's class map was checked: ball 0, goalkeeper 1,
player 2, referee 3. Cropping uses [1, 2, 3], not the ball. Its checkpoint,
1920-pixel inference size, confidence and tracker settings are fixed and archived;
cropping is a separate detector-assisted condition, not fine-tuning either pose model.
Crops in the shared-crop condition are generated once and shared by all three pose
runners. Host detector VRAM is released before the pose containers start. HRNet's
internal detector, when needed, runs inside its own container.

Super-resolution uses [FSRCNN's published OpenCV-compatible weights](https://github.com/Saafke/FSRCNN_Tensorflow)
at revision `6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d`, SHA-256
`366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c`.
This is learned enhancement, not recovery of guaranteed true detail. Resize x2 is
the non-learned scale control. Both produce enlarged artifacts, but **pose inference
sizes remain fixed at YOLO 1920 and OpenPose 1920x1088**. Enlarged full frames
therefore get fitted back to those model sizes. This measures enhancement at a fixed
pose-input protocol, not unrestricted 4K inference. Test larger model inputs only
as a separate resolution ablation.

Deblurring assumes the specified motion direction/kernel; it does not estimate
camera motion. Denoising or sharpening can remove useful detail or introduce
artifacts. These are reproducible starting parameters, not presumed improvements.

The command sets all model/pitch/crop GPU IDs, checks selected assets and Docker
images before starting, and always includes the same-model baseline. Use repeated
`--method` or `--model` flags for a smaller screen. Missing assets are errors,
not silently skipped methods. See [server.md](server.md) for asset upload/download,
tmux and exact run commands.

Each invocation creates a new timestamped directory below mounted
`/mnt/storage2/vincent/football-pose/results/football/single-step`, containing:

- `configs/*.yaml`: exact expanded conditions, plus checkpoint hashes in the report.
- `screening.json`: status, full per-condition summaries, errors and comparison rows.
- `comparison.csv` / `comparison.md`: raw/on-pitch yields, records per original frame,
  model wall time/source-frame throughput and baseline-relative changes.
- Per-condition experiment/job folders: normal JSONL, Parquet, pitch decisions and logs.

Annotated videos and preprocessing caches also stay on mounted storage. Videos
are isolated by invocation, avoiding reuse of a previous run's rendering.
Reports update after every condition; failed conditions stay visible and remaining
conditions continue. Interrupted runs preserve reports for completed conditions.
A new invocation runs fresh pose jobs and may reuse preprocessing artifacts.

Model wall time includes container startup and decoding, not only network inference;
HRNet's time additionally includes its required YOLO detection and crop generation.
Its runner log reports detector/crop and pose-stage times separately.
Preprocessing and pitch geometry are shared; do not count the same shared cost twice
when comparing models. Condition wall time includes all selected model jobs and rendering.
Cold artifact-processing time and current cache-access wall time are reported
separately: do not claim an acceleration from comparing a cold run with a warm one.
Percentage changes are unavailable when the baseline failed, is missing, or is zero.

This batch measures **yield and runtime**, not pose accuracy. AP/AR/OKS/PCK on
football footage still require aligned annotated ground truth. The existing COCO
benchmark remains an unchanged model-setup check. Promote combinations only after
individual conditions have been compared with their matching baseline.

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
A two-model configuration produces two `JOB_ID` directories. The current screening
base contains three models and therefore produces three job directories per condition.

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
MP4 files. The three-model screen also creates an HRNet video for each condition.
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

Historical low-resolution YOLOv8/OpenPose pilot results (not the current YOLO26 screen):

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

These pilot counts motivated the full-resolution comparisons; they do not establish
which method is best with the current checkpoints and inference sizes. Overlapping
tile records must be deduplicated before being compared with ground truth.

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
- `src/football_pose/screening.py`: single-step suite, preflight and automatic comparison reports.
- `configs/lyra-preprocessing-screen.yaml`: all single-factor conditions; the adjacent
  `lyra-preprocessing-base.yaml` holds shared model and pitch settings.
- `src/football_pose/preprocessing/tiling.py`: deterministic tile generation.
- `src/football_pose/preprocessing/cropping.py`: learned detection, tracking, and crops.
- `src/football_pose/artifacts.py`: immutable lossless inputs shared by the models.
- `src/football_pose/pitch_filter.py`: cached pitch geometry, cross-tile deduplication,
  ground-point classification, and filtered archives.
- `runners/`: isolated YOLO Pose, HRNet-W32, and OpenPose adapters.
- `src/football_pose/hrnet_input.py`: existing YOLO player detector → person crops
  for HRNet, with automatic reuse of shared person-crop artifacts.
- `experiment-output/` or the configured server results directory: manifests, logs,
  JSONL, and Parquet results.

The metric implementation follows after the ground-truth format, camera projection,
and timestamp synchronization are confirmed.
