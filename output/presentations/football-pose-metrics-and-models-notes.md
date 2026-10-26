# Scientific project presentation: metrics and model architectures

16 main slides; target speaking time 10:00. Slides 17–25 are backup material, not part of the timed route. All worked examples are illustrative. Proposed football metric conventions are distinguished from the implemented official COCO evaluator.

## 1. Preprocessing effects on fixed pose models (20 seconds)

This thesis studies how input processing changes pose estimation. I keep published pose checkpoints fixed and compare each processed input with the same model’s unchanged baseline. The focus is controlled evaluation, not training a different model for each condition.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Photo: footage/sample_30.mov, frame 140.

## 2. Research question (30 seconds)

Football video combines small players, shadows, occlusion and a moving camera. The intervention is one input transformation. We measure target detection, joint localization and computational cost. The same enhancement can help one architecture and hurt another, so each condition is compared with that model’s own baseline on the same frames.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Photo: footage/sample_30.mov, frame 580.

## 3. Controlled comparison (40 seconds)

Both branches use the same input and independent labels. Only the configured processing step changes. Pose weights and inference settings remain fixed; pretrained does not mean randomly initialized. HRNet uses our existing trained player detector. The pitch model and learned super-resolution weights also stay fixed. Consequently, the effect includes changes in detection and cropping within the complete pipeline, not only an isolated pose network.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Compared pose models: YOLO26x Pose, OpenPose BODY_25 and HRNet-W32. Auxiliary football detector, pitch model and FSRCNN are existing trained components, not newly trained per condition.

## 4. Three routes to multi-person poses (60 seconds)

Multi-person means that one image can produce several poses; it is not a third alternative to top-down and bottom-up. In our top-down pipeline, a YOLO player detector finds boxes, each crop goes through HRNet, and joint coordinates return to the source image. Missed boxes prevent poses. HRNet preserves high-resolution features with repeated cross-scale fusion. OpenPose predicts joint confidence maps and part-affinity vector fields over the image, then associates body parts into people. It has no separate person-box prerequisite. YOLO26 Pose jointly outputs person detections and their keypoints in one network pipeline; it is not a detector followed by a separate crop-wise HRNet. Different bottlenecks explain why processing affects them differently.

Sources:
- Project source: runners/hrnet/run.py, src/football_pose/hrnet_input.py, model_mapping.py and coco_evaluation.py; configs/lyra-preprocessing-base.yaml at commit 21c1068.
- Sun et al., Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019): https://arxiv.org/abs/1902.09212
- Cao et al., Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (CVPR 2017): https://arxiv.org/abs/1611.08050
- Ultralytics YOLO26 architecture and pose task: https://docs.ultralytics.com/models/yolo26/ ; https://docs.ultralytics.com/tasks/pose/

## 5. Ground truth for distinct evaluation tasks (35 seconds)

Each quantity needs the right reference. COCO provides image-plane joint labels and visibility flags. Football pose accuracy needs independently validated body-joint labels, either supplied or manually reviewed. Tracking x and y coordinates instead support player ground-position error. They cannot validate all body joints. Predicted pitch geometry is not independent ground truth. Football reference alignment remains pending.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- TODOS.md: ground-truth alignment remains blocked by external format, camera/projection and synchronization details.

## 6. Football reference-data protocol (30 seconds)

We first sample different scales, lighting conditions and camera views. We align timestamps, identities and joint definitions, review visibility and uncertain annotations, then freeze sequence-level development and test splits. Include missed target players in the reference population. Neighbouring frames must not leak across splits. A pitch-plane homography cannot project arbitrary elevated body joints.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Proposed study protocol, not a completed annotation or quality-assurance result.

## 7. Pipeline, step by step (45 seconds)

Freeze source frames and metadata. Apply one processor and cache the transformed inputs. Run fixed pose weights, then return all crop, resize and tile coordinates to source pixels and remove cross-region duplicates. A separate fixed pitch model supplies per-frame geometry shared by all conditions. Finally match people and joints to independent labels, score accuracy and compare against baseline. The pipeline and official COCO scoring exist; football reference alignment and accuracy scoring are still the next layer.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- src/football_pose/screening.py, experiments.py, model_mapping.py, pitch_filter.py and coco_evaluation.py.

## 8. Detection: precision and recall (40 seconds)

Define the reference population and matching criterion first. Person detection can use box overlap, while pose evaluation uses OKS. With one-to-one matching, a valid match is a true positive, an unmatched prediction is a false positive, and an unmatched reference is a false negative. A second prediction cannot earn credit for the same ordinary target. Eight matches, two extras and four misses give eighty percent precision but only sixty-six point seven percent recall. Correct boxes do not establish correct joints. COCO additionally applies its official ignore and crowd rules.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- Illustrative arithmetic, not experiment results. Person-box matching and pose-OKS matching must be reported separately.

## 9. OKS: turn joint errors into a pose similarity (50 seconds)

For each labelled joint, measure pixel distance, normalize by person area and a fixed joint-specific tolerance, then apply the exponential. Average those contributions. Perfect localization approaches one; larger normalized errors approach zero. The same pixel error is more severe on a smaller person. COCO uses k equal to twice its stored sigma. Ground-truth visibility selects joints, including labelled occluded joints; prediction confidence does not weight this formula. A worked example is in the backup slides.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

## 10. AP and AR: aggregate matched poses (45 seconds)

Rank poses by person score. At each OKS threshold, match and accumulate true and false positives. AP uses the interpolated precision envelope at 101 recall levels, then averages across ten OKS thresholds. AP fifty and AP seventy-five use only their named threshold. AR averages the maximum achieved recall at those thresholds. Official keypoint evaluation limits retained predictions to twenty per image. AP is neither the average of precision and recall nor the percentage of detected joints. Backup slides show an explicit ranked example.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

## 11. PCK: how many joints are close enough? (40 seconds)

PCK turns localization into a pass or fail at a declared normalized distance. Our proposed scale is the reference person-box diagonal. A diagonal of two hundred pixels and threshold zero point zero five allow ten pixels. Six correct joints out of eight give seventy-five percent PCK. Seven confident predictions give higher coverage, but confidence does not establish accuracy. For proposed end-to-end PCK, missed targets remain incorrect. Declare eligibility, aggregation and thresholds. The backups explain PCP, PCK-AUC and tracking-position error.

Sources:
- MMPose metric definitions (implementations vary in normalization and aggregation): https://mmpose.readthedocs.io/en/latest/_modules/mmpose/evaluation/functional/keypoint_eval.html
- Supplementary football metric conventions are proposed, not yet implemented results. Numeric examples are illustrative, not experimental observations.

## 12. Estimating the processing effect (25 seconds)

The effect is a paired score difference against each model’s own baseline. Keep the same targets and eligible ground-truth joints in every denominator. Separate person detection, joint localization and pitch-filter errors. Choose settings on development clips, confirm on held-out sequences, and account for correlation between neighbouring video frames.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Proposed statistical protocol: paired sequence-level summaries and interval estimation after sufficient independent clips are available.

## 13. Reproducibility safeguards (35 seconds)

We record input and checkpoint hashes, expanded YAML, code revision, package versions, hardware and logs. Raw predictions and pitch decisions remain available. For the final study, also freeze reference-label versions and actual container image digests, repeat a reference run, declare numerical tolerances and separate cold from warm-cache runtime. Fixed weights reduce variation but do not guarantee bit-identical GPU arithmetic.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- src/football_pose/provenance.py; src/football_pose/artifacts.py; src/football_pose/screening.py.
- Final-study controls in the right column are proposed checks, not claims that all are currently enforced.

## 14. Experimental matrix (25 seconds)

The pilot uses one baseline and eleven individual methods, with three pose pipelines on 782 identical frames. Bicubic and learned super-resolution share the same enlargement factor. Start with individual processing effects; only later test selected combinations. COCO currently checks model functionality on unchanged inputs, rather than a complete preprocessing ablation.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- configs/lyra-preprocessing-screen.yaml.

## 15. Current experiment status (50 seconds)

Thirty-five of thirty-six jobs complete. These examples are on-pitch record yields, not accuracy gains: bilateral filtering raises YOLO records, CLAHE raises OpenPose records and doubling scale greatly raises HRNet records. We still need independent labels to learn whether added poses are correct. HRNet tiling yields one raw record, YOLO cropping zero, and OpenPose cropping fails. Those cases need investigation. Super-resolution and bicubic resizing have almost identical HRNet yield. The final claim must come from accuracy and cost, not appearance or counts alone.

Sources:
- User-supplied screening overview for 20261022T181655Z-fefcb4ec, 22 October 2026. Person-pose record counts across 782 frames, not unique players or accuracy.
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- Pitch reference comparison separately showed 782/782 accepted fits for Roboflow versus 306/782 for the retired nano model. Fit coverage does not establish landmark accuracy.

## 16. Next scientific milestones (30 seconds)

Confirm the football reference schema and annotation fallback. Agree on the primary metric, person matching, visibility rules and held-out sequences. Resolve runner anomalies, verify checkpoint identities, repeat baselines and score every condition against the same references. Then confirm selected combinations and separately evaluate synchronized player positions against tracking data.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Proposed milestones.

## 17. On-pitch overview 1/2 (backup)

Use this table for discussion. These are record counts across the pilot clip. They do not rank pose accuracy. HRNet tiling and YOLO cropping have suspicious raw outputs, while OpenPose cropping fails. Repeated-frame observations are correlated. The current report mainly has warm-cache timings, so it cannot establish total cold preprocessing cost.

Sources:
- User-supplied screening overview for 20261022T181655Z-fefcb4ec, 22 October 2026. Person-pose record counts across 782 frames, not unique players or accuracy.

## 18. On-pitch overview 2/2 (backup)

Use this table for discussion. These are record counts across the pilot clip. They do not rank pose accuracy. HRNet tiling and YOLO cropping have suspicious raw outputs, while OpenPose cropping fails. Repeated-frame observations are correlated. The current report mainly has warm-cache timings, so it cannot establish total cold preprocessing cost.

Sources:
- User-supplied screening overview for 20261022T181655Z-fefcb4ec, 22 October 2026. Person-pose record counts across 782 frames, not unique players or accuracy.

## 19. PCP: complete body parts (backup)

For a limb connecting reference joints a and b, compute its ground-truth length L. Under this explicit strict variant, both predicted endpoints must lie within alpha times L of their corresponding labels. Average the binary outcomes over eligible limbs. Freeze a limb list; one proposed set is left and right upper arms, forearms, thighs and lower legs. Both endpoints must have usable labels and L must be nonzero. An unmatched person or missing predicted endpoint fails the limb in the proposed end-to-end score. Alpha equals 0.5 here only for illustration. PCP variants differ, so report this convention rather than implying every published PCP score is directly comparable.

Sources:
- Supplementary football metric conventions are proposed, not yet implemented results. Numeric examples are illustrative, not experimental observations.
- Definition requested in the user’s research notes: both endpoint errors relative to reference limb length.

## 20. PCK-AUC: threshold curve (backup)

Sweep the PCK threshold while keeping the reference set, scale and matching fixed. Integrate the resulting PCK curve over a declared interval and divide by interval width. The normalized value is between zero and one. This three-point illustration uses trapezoidal integration: the two areas are 0.0125 and 0.0375, so their sum divided by 0.10 is 0.50. A real protocol needs a fixed sufficiently dense grid. Sampling and integration conventions differ between libraries; MMPose, for example, uses a discrete threshold average. Do not compare AUC values with different normalizers, ranges or quadrature rules. This is not AP’s precision-recall area.

Sources:
- MMPose metric definitions (implementations vary in normalization and aggregation): https://mmpose.readthedocs.io/en/latest/_modules/mmpose/evaluation/functional/keypoint_eval.html
- Supplementary football metric conventions are proposed, not yet implemented results. Numeric examples are illustrative, not experimental observations.

## 21. Coverage and pitch-position error (backup)

Coverage counts whether a matched eligible reference joint receives a finite prediction above a declared joint-confidence threshold. No localization tolerance appears, so high coverage can coexist with poor PCK. The proposed denominator includes eligible joints of missed people; joints without a reference label are excluded. For tracking, compare synchronized player identities in one metric coordinate system. For a prediction (31,20) and reference (30,18), Euclidean error is square root of five, about 2.24 metres. Average over M matched player-time pairs and report median plus match coverage; missing players are not zero-error observations. A homography maps ground-contact points only. Calibration errors contribute to this downstream error.

Sources:
- Supplementary football metric conventions are proposed, not yet implemented results. Numeric examples are illustrative, not experimental observations.
- Project source: runners/hrnet/run.py, src/football_pose/hrnet_input.py, model_mapping.py and coco_evaluation.py; configs/lyra-preprocessing-base.yaml at commit 21c1068.
- All numeric examples are illustrative; independent tracking alignment remains pending.

## 22. AP worked example (backup)

These illustrative outcomes assume one-to-one matching has already been performed at a fixed OKS threshold. Cumulate TP and FP in descending person-score order. Replace precision at a recall by the best precision attained at that or any greater recall. Among 101 recall levels, 34 are at or below one third and receive 1.00; the remaining 67 receive 0.75. AP at this threshold is 0.834. Maximum recall is 1.00. The late false positive does not lower the envelope here. Repeat matching at every OKS threshold before averaging; these are not measured project results.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- Illustrative list and arithmetic authored for this presentation.

## 23. OKS worked example (backup)

The first column changes localization error while holding area and tolerance fixed. Averaging the three joint contributions gives approximately 0.581. Reducing person area increases the penalty for the same pixel error. The example uses a hypothetical k for arithmetic clarity; production COCO has fixed per-joint sigmas and k equals twice sigma. Ground-truth visibility zero is excluded, while labelled occluded and visible joints both contribute. Predicted joint confidence is neither ground-truth visibility nor a direct multiplicative weight in official OKS. Person confidence is used separately for ranking.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- Numeric example is illustrative; it is not the reported accuracy of any checkpoint.

## 24. Architecture mechanisms and bottlenecks (backup)

HRNet is a single-person pose network inside our multi-person detector-plus-crop system. It repeatedly fuses parallel-resolution features and predicts joint heatmaps. The project uses a COCO-trained W32 checkpoint with 256 by 192 pose input. OpenPose predicts evidence across the full image and uses part-affinity fields to associate limbs without first requiring a person detection box. Its convolutional work is largely shared, though grouping still depends on detected candidates. YOLO26 Pose jointly predicts instance boxes and keypoints using a task-specific head, without an external detector-to-HRNet pass. Preprocessing may change box detection, joint evidence or associations. None of these labels guarantees higher accuracy on this football video. BODY_25 is converted to the common COCO17 order before comparison.

Sources:
- Sun et al., Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019): https://arxiv.org/abs/1902.09212
- Cao et al., Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (CVPR 2017): https://arxiv.org/abs/1611.08050
- Ultralytics YOLO26 architecture and pose task: https://docs.ultralytics.com/models/yolo26/ ; https://docs.ultralytics.com/tasks/pose/
- Project source: runners/hrnet/run.py, src/football_pose/hrnet_input.py, model_mapping.py and coco_evaluation.py; configs/lyra-preprocessing-base.yaml at commit 21c1068.

## 25. Freeze the calculation protocol (backup)

For supplementary football PCK, this deck proposes micro-averaging over eligible reference joints with reference-box diagonal normalization and a less-than-or-equal distance threshold. Missing predictions count incorrect; missing reference labels are excluded. A matched-only analysis can isolate localization, but must be labelled conditional and accompanied by detection recall. Define matching independently and freeze it across conditions. PCP needs the explicit both-endpoints rule and limb list. Confidence coverage needs a declared joint threshold. AP needs a reproducible person-score definition and must retain sufficient low-score predictions; a high inference cutoff can truncate recall. Use official ignore rules and maximum detections for COCO. Football crowd scenes need their own declared protocol rather than silently reusing the twenty-pose cap.

Sources:
- Project source: runners/hrnet/run.py, src/football_pose/hrnet_input.py, model_mapping.py and coco_evaluation.py; configs/lyra-preprocessing-base.yaml at commit 21c1068.
- Supplementary football metric conventions are proposed, not yet implemented results. Numeric examples are illustrative, not experimental observations.
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- MMPose metric definitions (implementations vary in normalization and aggregation): https://mmpose.readthedocs.io/en/latest/_modules/mmpose/evaluation/functional/keypoint_eval.html
