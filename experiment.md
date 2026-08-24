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

Promote only beneficial individual steps into combination experiments. Every promoted
combination is compared against its direct parent pipeline so the contribution of each
added step remains visible.

### 5. Final confirmation

Run the best candidate pipelines on the complete evaluation footage with all three
models where the input contract is valid. Preserve the YAML, artifact manifest,
checkpoint hashes, runner logs, prediction Parquet files, and final metrics.

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
