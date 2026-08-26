# Experiment results overview

This folder is the quick, human-readable index of completed thesis experiments. It
contains summarized results only. Canonical predictions, manifests, and logs remain in
the configured experiment output directory and are not committed to Git.

## Preprocessing screen: 30-second clip

Scope: `sample_30.mov`, 782 frames, official published model weights, YOLO inference
size 640 with confidence 0.25, and OpenPose BODY_25 network resolution `-1x368`.
Except for the tiling row, every pipeline processes the full frame and changes exactly
one preprocessing factor.

| Processing | YOLO records | OpenPose records | OpenPose change from baseline | Current decision |
| --- | ---: | ---: | ---: | --- |
| Baseline: full frame, unchanged | 0 | 826 | reference | Comparison baseline |
| CLAHE | 0 | 1,154 | +39.7% | Promote for visual validation |
| Gamma brighten, 1.2 | 0 | 797 | -3.5% | Reject |
| Gamma darken, 0.8 | 0 | 700 | -15.3% | Reject |
| Unsharp mask | 0 | 933 | +13.0% | Retain as secondary candidate |
| 2 x 2 tiling, 10% overlap | 451 | 5,989 | +625.1% raw | Strongest yield; deduplicate first |

One record is one person-pose prediction. Record count is a detection-yield diagnostic,
not an accuracy score: false positives increase it, missed players decrease it, and
overlapping tiles can produce duplicate predictions. No pipeline is considered optimal
until visual inspection and ground-truth metrics confirm pose quality.

## Current interpretation

- CLAHE is the strongest non-tiling transformation by raw OpenPose yield.
- Unsharp masking improved OpenPose yield, but less than CLAHE.
- Neither gamma variant is promoted.
- Full-frame pixel transformations did not solve YOLO's player-scale bottleneck.
- Tiling produced the largest increase, but requires source-coordinate deduplication
  before its count can be interpreted as distinct player detections.

## Updating this overview

Add a row only after the corresponding configuration, `summary.json`, job manifests,
Parquet archives, and logs have been preserved. Record the exact source clip, frame
count, model settings, checkpoint hashes, preprocessing parameters, runtime, and whether
the result is raw or deduplicated. Detailed methodology and output-file definitions are
kept in [`experiment.md`](../experiment.md).
