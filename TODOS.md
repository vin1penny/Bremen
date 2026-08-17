# TODOS

## Evaluation

### Implement ground-truth alignment and pose-evaluation metrics

**What:** Build an evaluation layer for mAP/OKS, PCK, detected-keypoint rate, confidence summaries, temporal stability, and runtime comparisons.

**Why:** The thesis needs quantitative comparisons between raw and preprocessed inputs across OpenPose, HRNet-W32, and YOLO Pose.

**Context:** The preprocessing and inference system will first produce validated Parquet predictions in original-frame coordinates using the canonical COCO-17 schema, presentation timestamps, model and pipeline identifiers, and complete provenance. The evaluator should consume those archives without rerunning preprocessing or inference. It must align predictions to the DFL/Werder ground truth, map any differing joint taxonomy, project ground truth into the scouting-camera image plane when necessary, and make missing detections distinguishable from missing or unsynchronized ground truth. Deferring this keeps uncertain external formats out of the initial modular pipeline while preserving the thesis-critical work and its rationale.

**Effort:** L
**Priority:** P2
**Depends on:** Stable canonical prediction schema and validated experiment archives
**Blocked by:** Exact DFL keypoint format, camera calibration/projection data, and synchronization mechanism from Werder Bremen/DFL

## Completed
