# Scientific project presentation

12 main slides, target 10:00. Three backup slides for questions.

## 1. Preprocessing effects on fixed pose models (20 seconds)

My thesis investigates the effect of preprocessing on pose estimation. The scientific intervention is the input transformation. I keep the published pose-model weights unchanged and compare each processed input with its own baseline. Today I will focus on the experimental controls, reference data, evaluation protocol and reproducibility.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Photo: footage/sample_30.mov, frame 140.

## 2. Research question (40 seconds)

The question is whether a defined input operation changes pose quality under a fixed model protocol. Football scouting video gives us small players, shadows and a moving camera. I compare the same images with and without processing, rather than train a model for each condition. The outputs of interest include whether we find the right people, how accurately we locate their joints, and the computational cost. A visual enhancement can help one model and hurt another. The effect we estimate therefore applies to the specified model, data and inference settings.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Photo: footage/sample_30.mov, frame 580.

## 3. Controlled comparison (60 seconds)

The central design is a paired comparison. Both branches start from exactly the same input and reference labels. One leaves the input unchanged. The other applies one processing operation with explicit parameters. Each branch uses the same published pretrained checkpoint, confidence thresholds and inference configuration. We do not update weights during this experiment and do not fine-tune a separate pose model per condition. This means off-the-shelf pretrained inference, rather than randomly initialized, untrained networks. HRNet still needs our existing trained player detector, and the pitch model and learned super-resolution also have existing trained weights. Those auxiliary checkpoints stay fixed too. The estimated effect includes downstream detector and crop changes in a complete pipeline.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Compared pose models: YOLO26x Pose, OpenPose BODY_25 and HRNet-W32. Auxiliary football detector, pitch model and FSRCNN are existing trained components, not newly trained per condition.

## 4. Ground truth for distinct evaluation tasks (65 seconds)

We need different reference data for different questions. COCO already provides person keypoints and visibility labels, and the official evaluator is integrated. Football pose accuracy requires independent body-joint annotations in the source image. We first need to confirm the DFL or Werder schema, timestamps and joint definitions. If suitable joints are unavailable, a reviewed manual annotation sample is the fallback. Tracking x and y coordinates support a later player-position evaluation, but they do not provide ground truth for all body joints. Predicted poses and predicted pitch boundaries are not independent reference labels. I am showing the football annotation and alignment work as pending, rather than claiming that these references already exist in an evaluable form.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- TODOS.md: ground-truth alignment remains blocked by external format, camera/projection and synchronization details.

## 5. Football reference-data protocol (55 seconds)

The proposed football reference workflow starts with an evaluation sample that covers different scales, lighting conditions, occlusion and camera views. The inclusion rule should cover actual target players, including missed detections. We then align timestamps, identities and joint names in original-frame coordinates. Independent review records visibility and uncertain joints, rather than forcing labels where the evidence is inadequate. We freeze the reviewed labels and split by sequence or match so that neighbouring frames do not leak between development and final testing. If reference joints come in three-dimensional coordinates, they require camera projection. A planar pitch homography only supports ground-plane points, not arbitrary elevated body joints.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Proposed study protocol, not a completed annotation or quality-assurance result.

## 6. Pipeline, step by step (70 seconds)

Step one fixes the source images, frame identifiers and metadata. Step two applies exactly one configured processing method and writes a lossless cached artifact. Step three runs the unchanged pose model. For HRNet, the fixed detector supplies person crops. Step four maps crop, resize and tile coordinates back to the original image and removes cross-tile duplicate records. A separate frozen pitch model operates on each original football frame and provides shared geometry for target filtering. Step five matches predictions to independent reference people and eligible joints. Step six computes the chosen metrics and the paired difference against baseline. The implementation already covers preprocessing, inference, remapping and pitch decisions. Official COCO matching and scoring exist. Football reference alignment and accuracy scoring remain the next evaluation layer.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- src/football_pose/screening.py, experiments.py, model_mapping.py, pitch_filter.py and coco_evaluation.py.

## 7. Evaluation metrics (65 seconds)

The main pose benchmark uses OKS-based average precision and average recall. COCO accounts for person scale and joint-specific localization tolerances. PCK measures the rate of sufficiently close joints under a declared normalization. PCK-AUC summarizes the threshold curve. PCP adds a body-part view. Player precision and recall distinguish extra detections from missed target players. Detected-joint coverage describes completeness, so it should be interpreted alongside localization accuracy. For the later tracking task, positional error uses metres. Before evaluating football, we must freeze matching, joint eligibility, score thresholds and normalization. Raw counts from the present screening cannot substitute for these accuracy measures. The supplementary football metrics are a planned evaluation protocol, not results we have already computed.

Sources:
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- MMPose PCK and PCK-AUC reference: https://mmpose.readthedocs.io/en/latest/_modules/mmpose/evaluation/functional/keypoint_eval.html
- Proposed football extension. Official COCO maxDets=20 and standard OKS sigmas apply to the separate benchmark. A football protocol must declare any deviation.

## 8. Estimating the processing effect (40 seconds)

For each model, the processing effect is a paired score difference on the same reference data. We must not evaluate only the players a method happens to detect. Missed people and frames without predictions stay in the evaluation population. We also separate pose localization errors from person detection and pitch-filter failures. Thresholds and candidate combinations should be chosen on development clips and confirmed on held-out sequences. Multiple sequences support uncertainty estimates. Adjacent frames from the current single clip are correlated, so 782 frames do not constitute 782 independent experiments.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Proposed statistical protocol: paired sequence-level summaries and interval estimation after sufficient independent clips are available.

## 9. Reproducibility safeguards (50 seconds)

The pipeline records input and checkpoint hashes, expanded YAML, processing order, code revision, package versions, hardware metadata and logs. It preserves the raw predictions as well as derived pitch decisions. Content-addressed caches separate different inputs and settings, and new invocations preserve earlier result folders. To complete the final-study reproducibility protocol, we should also pin actual container image digests, version the reference labels and evaluator, and repeat a reference run under declared numerical tolerances. A mutable development image tag is not a complete environment identifier. No retraining removes one source of variation, but it does not guarantee bit-identical GPU arithmetic. We should verify repeatability instead of assuming it.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- src/football_pose/provenance.py; src/football_pose/artifacts.py; src/football_pose/screening.py.
- Final-study controls in the right column are proposed checks, not claims that all are currently enforced.

## 10. Experimental matrix (35 seconds)

The current pilot evaluates twelve conditions with three model families. One condition is the unchanged baseline, followed by eleven separate methods. Tiling uses a fixed two-by-two grid with ten percent overlap. The scale comparison includes bicubic resizing and fixed FSRCNN super-resolution at the same factor, which helps distinguish enlargement from a learned restoration effect. We first study individual operations. Combinations come later, after the reference data and primary metrics are fixed. The full COCO preprocessing ablation is a later extension. The implemented COCO path currently checks the unchanged model setup.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- configs/lyra-preprocessing-screen.yaml.

## 11. Current experiment status (60 seconds)

The current football pilot completes thirty-five of thirty-six model jobs. These examples show why model-specific evaluation matters. Bilateral filtering increases YOLO on-pitch records from 357 to 2,115. CLAHE increases OpenPose from 11,647 to 15,338. Bicubic resizing raises HRNet from 3,602 to 15,554. These are exploratory yield observations, not validated gains in localization accuracy. The latest filtering counts changed after the configuration update, while the checkpoint hash still needs confirmation from the detailed report. Three integration cases remain unresolved: HRNet with tiling yields one raw record, YOLO cropping yields zero, and OpenPose cropping fails. Super-resolution produces almost the same HRNet yield as bicubic scaling, so any distinct restoration benefit still needs accuracy evidence. The next scientific milestone is an independently referenced comparison.

Sources:
- User-supplied screening overview for 20261022T181655Z-fefcb4ec, 22 October 2026. Person-pose record counts across 782 frames, not unique players or accuracy.
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- Pitch reference comparison separately showed 782/782 accepted fits for Roboflow versus 306/782 for the retired nano model. Fit coverage does not establish landmark accuracy.

## 12. Next scientific milestones (40 seconds)

The next milestone is agreement on the football reference data and the primary evaluation protocol. We need to confirm whether the supplied data contain suitable body joints, synchronize any tracking reference and decide the manual-annotation fallback. We then freeze metrics, matching rules and development/test sequences. After resolving the runner problems, we can repeat baselines and evaluate all conditions against the same labels. Selected combinations require a final held-out confirmation. Pitch-coordinate tracking comparison follows as a separate downstream evaluation. This makes the thesis a controlled study of input processing with fixed models, rather than a model-training project.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Proposed milestones.

## 13. On-pitch overview 1/2 (backup)

Use this table for discussion. These are record counts across the pilot clip. They do not rank pose accuracy. HRNet tiling and YOLO cropping have suspicious raw outputs, while OpenPose cropping fails. Repeated-frame observations are correlated. The current report mainly has warm-cache timings, so it cannot establish total cold preprocessing cost.

Sources:
- User-supplied screening overview for 20261022T181655Z-fefcb4ec, 22 October 2026. Person-pose record counts across 782 frames, not unique players or accuracy.

## 14. On-pitch overview 2/2 (backup)

Use this table for discussion. These are record counts across the pilot clip. They do not rank pose accuracy. HRNet tiling and YOLO cropping have suspicious raw outputs, while OpenPose cropping fails. Repeated-frame observations are correlated. The current report mainly has warm-cache timings, so it cannot establish total cold preprocessing cost.

Sources:
- User-supplied screening overview for 20261022T181655Z-fefcb4ec, 22 October 2026. Person-pose record counts across 782 frames, not unique players or accuracy.

## 15. Metric conventions (backup)

For the football extension, one proposed normalization is Euclidean joint error divided by the ground-truth box diagonal, with predeclared PCK thresholds and a declared PCK-AUC interval. That choice remains to be agreed rather than silently inferred from a library default. Define the PCP limb list and reference-length tolerance. Missing detections must remain visible in end-to-end recall. If PCK is reported only on matched people, label that conditional result and also report detection recall. Ground-truth visibility determines eligibility. Joint-confidence coverage cannot by itself establish positional correctness. Tracking error requires matched identities, timestamps and coordinate conventions. COCO uses its existing official protocol.

Sources:
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- MMPose PCK and PCK-AUC reference: https://mmpose.readthedocs.io/en/latest/_modules/mmpose/evaluation/functional/keypoint_eval.html
