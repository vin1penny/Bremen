# Thesis presentation aligned to the exposé

18 main slides, target 10:00. Slides 19–29 are backup material. Calculations use illustrative examples. The exposé defines the scope. Current implementation status comes from repository documentation and the supplied pilot results.

Scope correction: pitch-coordinate player tracking, radar display and tracking-position error are downstream extensions outside the current thesis evaluation. Multi-camera skeletal labels can serve as reference data for single-camera 2D predictions.

## 1. Visual preprocessing for robust football pose estimation (15 seconds)

The exposé defines this thesis as Optimization of Visual Data Preprocessing for Robust Keypoint Detection in Single-Camera Football Scouting Feed. I investigate existing input-processing methods with fixed pretrained pose systems and compare quality against computational cost.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Repository experiment.md, TODOS.md and configs/lyra-preprocessing-screen.yaml at commit 21c1068. Status also uses the user-supplied 20261022T181655Z-fefcb4ec screening results.

## 2. Research question and scientific contribution (30 seconds)

The scientific contribution is a systematic transfer study. Methods established in image enhancement, restoration and object detection may behave differently when we use them for skeletal keypoints of small football players. We ask which individual methods help, whether selected combinations add value, how effects differ across pose architectures, and what runtime they require. The output should be evidence-based preprocessing guidelines for this video setting.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Repository experiment.md, TODOS.md and configs/lyra-preprocessing-screen.yaml at commit 21c1068. Status also uses the user-supplied 20261022T181655Z-fefcb4ec screening results.

## 3. Thesis scope (35 seconds)

The exposé limits the task to two-dimensional body poses from one pre-recorded football camera. We vary visual processing and hold the pose checkpoints fixed. Architecture development, pose-model retraining, three-dimensional inference, tracking or event-data fusion, and camera-hardware changes stay outside the evaluation. Runtime still matters, although a real-time product is not the target. Multi-camera DFL joint labels may provide the evaluation reference without becoming extra inference inputs. Earlier pitch-coordinate tracking ideas are downstream extensions, outside this thesis plan.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf

## 4. Controlled comparison (35 seconds)

Both branches use the same input and independent labels. Only the configured processing step changes. Pose weights and inference settings remain fixed; pretrained does not mean randomly initialized. HRNet uses our existing trained player detector. The pitch model and learned super-resolution weights also stay fixed. Consequently, the effect includes changes in detection and cropping within the complete pipeline, not only an isolated pose network.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- Compared pose models: YOLO26x Pose, OpenPose BODY_25 and HRNet-W32. Auxiliary football detector, pitch model and FSRCNN are existing trained components, not newly trained per condition.

## 5. Three routes to multi-person poses (50 seconds)

Multi-person means that one image can produce several poses; it is not a third alternative to top-down and bottom-up. In our top-down pipeline, a YOLO player detector finds boxes, each crop goes through HRNet, and joint coordinates return to the source image. Missed boxes prevent poses. HRNet preserves high-resolution features with repeated cross-scale fusion. OpenPose predicts joint confidence maps and part-affinity vector fields over the image, then associates body parts into people. It has no separate person-box prerequisite. YOLO26 Pose jointly outputs person detections and their keypoints in one network pipeline; it is not a detector followed by a separate crop-wise HRNet. Different bottlenecks explain why processing affects them differently.

Sources:
- Project source: runners/hrnet/run.py, src/football_pose/hrnet_input.py, model_mapping.py and coco_evaluation.py; configs/lyra-preprocessing-base.yaml at commit 21c1068.
- Sun et al., Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019): https://arxiv.org/abs/1902.09212
- Cao et al., Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (CVPR 2017): https://arxiv.org/abs/1611.08050
- Ultralytics YOLO26 architecture and pose task: https://docs.ultralytics.com/models/yolo26/ ; https://docs.ultralytics.com/tasks/pose/

## 6. Evaluation data and ground truth (45 seconds)

The main input is a single wide-angle football video. The exposé intends to obtain independent skeletal labels through Werder Bremen and the DFL. We still need the exact joint schema, timestamps and any camera parameters needed to project labels into the scouting image. Access and alignment are requirements, not completed evaluation. A pitch homography only maps ground-plane points and cannot project all body joints. COCO already supports model setup checks. PoseTrack remains optional. External joint labels serve as references only. Tracking coordinates and predicted pitch boundaries cannot replace body-joint ground truth.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Repository experiment.md, TODOS.md and configs/lyra-preprocessing-screen.yaml at commit 21c1068. Status also uses the user-supplied 20261022T181655Z-fefcb4ec screening results.
- TODOS.md records the pending DFL format, projection and synchronization contract.

## 7. Pipeline, step by step (35 seconds)

Freeze source frames and metadata. Apply one processor and cache the transformed inputs. Run fixed pose weights, then return all crop, resize and tile coordinates to source pixels and remove cross-region duplicates. A separate fixed pitch model supplies per-frame geometry shared by all conditions. Finally match people and joints to independent labels, score accuracy and compare against baseline. The pipeline and official COCO scoring exist; football reference alignment and accuracy scoring are still the next layer.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- src/football_pose/screening.py, experiments.py, model_mapping.py, pitch_filter.py and coco_evaluation.py.

## 8. Detection: precision and recall (35 seconds)

Define the reference population and matching criterion first. Person detection can use box overlap, while pose evaluation uses OKS. With one-to-one matching, a valid match is a true positive, an unmatched prediction is a false positive, and an unmatched reference is a false negative. A second prediction cannot earn credit for the same ordinary target. Eight matches, two extras and four misses give eighty percent precision but only sixty-six point seven percent recall. Correct boxes do not establish correct joints. COCO additionally applies its official ignore and crowd rules.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- Illustrative arithmetic, not experiment results. Person-box matching and pose-OKS matching must be reported separately.

## 9. OKS: joint errors and pose similarity (45 seconds)

For each labelled joint, measure pixel distance, normalize by person area and a fixed joint-specific tolerance, then apply the exponential. Average those contributions. Perfect localization approaches one; larger normalized errors approach zero. The same pixel error is more severe on a smaller person. COCO uses k equal to twice its stored sigma. Ground-truth visibility selects joints, including labelled occluded joints; prediction confidence does not weight this formula. A worked example is in the backup slides.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

## 10. Average Precision and Average Recall (35 seconds)

Rank poses by person score. At each OKS threshold, match and accumulate true and false positives. AP uses the interpolated precision envelope at 101 recall levels, then averages across ten OKS thresholds. AP fifty and AP seventy-five use only their named threshold. AR averages the maximum achieved recall at those thresholds. Official keypoint evaluation limits retained predictions to twenty per image. AP is neither the average of precision and recall nor the percentage of detected joints. Backup slides show an explicit ranked example.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

## 11. PCK: normalized joint localization (35 seconds)

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

## 13. Reproducibility safeguards (30 seconds)

We record input and checkpoint hashes, expanded YAML, code revision, package versions, hardware and logs. Raw predictions and pitch decisions remain available. For the final study, also freeze reference-label versions and actual container image digests, repeat a reference run, declare numerical tolerances and separate cold from warm-cache runtime. Fixed weights reduce variation but do not guarantee bit-identical GPU arithmetic.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- src/football_pose/provenance.py; src/football_pose/artifacts.py; src/football_pose/screening.py.
- Final-study controls in the right column are proposed checks, not claims that all are currently enforced.

## 14. Common uses: contrast and denoising (25 seconds)

These are typical applications, not claims that any method must improve football pose accuracy. CLAHE has a medical-imaging contrast-enhancement history. Gamma is common for tonal and exposure adjustment. Non-local means and bilateral filters address noise, with bilateral filtering preserving edges. Unsharp masking increases local edge contrast. We transfer each operation separately and measure whether it improves actual joint localization.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Pizer et al., Adaptive Histogram Equalization and Its Variations (1987), medical imaging application and clipped AHE: https://doi.org/10.1016/S0734-189X(87)80186-X ; implementation: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- OpenCV, brightness correction including gamma: https://docs.opencv.org/5.0/tutorials/core/basic_linear_transform/basic_linear_transform.html
- OpenCV, computational photography image denoising: https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html
- OpenCV, edge-preserving bilateral smoothing: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- Adobe, photographic sharpening and Unsharp Mask: https://helpx.adobe.com/photoshop/desktop/effects-filters/smart-filters/sharpening-overview.html
- Gamma darken and brighten are two separate screening conditions. Common uses are illustrative, not a frequency ranking.

## 15. Common uses: scale, regions and restoration (25 seconds)

Bicubic resizing provides a non-learned scale control. FSRCNN estimates a larger image using fixed learned weights, so we compare it against bicubic at the same factor. Wiener deblurring assumes a motion kernel. It is our classical restoration baseline, not a reproduction of the deep deblurring method cited in the exposé. Tiling comes from small-object inference, including aerial imagery. Person cropping is standard in top-down pose systems.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Project config fixes bicubic scale=2 as the non-learned scale control. Input resampling is a standard CV preparation operation.
- Dong et al., Accelerating the Super-Resolution Convolutional Neural Network (ECCV 2016), FSRCNN: https://arxiv.org/abs/1608.00367
- OpenCV, Motion Deblur Filter: https://docs.opencv.org/4.x/d1/dfd/tutorial_motion_deblur_filter.html
- Akyon et al., Slicing Aided Hyper Inference (ICIP 2022), aerial small-object detection: https://arxiv.org/abs/2202.06934
- Sun et al., HRNet and top-down pose estimation: https://arxiv.org/abs/1902.09212 ; project src/football_pose/hrnet_input.py
- The SAHI paper motivates regional inference. This project uses its own deterministic 2×2 tiling, not an exact SAHI reproduction.

## 16. Current exploratory results (40 seconds)

Thirty-five of thirty-six jobs complete. These examples are on-pitch record yields, not accuracy gains: bilateral filtering raises YOLO records, CLAHE raises OpenPose records and doubling scale greatly raises HRNet records. We still need independent labels to learn whether added poses are correct. HRNet tiling yields one raw record, YOLO cropping zero, and OpenPose cropping fails. Those cases need investigation. Super-resolution and bicubic resizing have almost identical HRNet yield. The final claim must come from accuracy and cost, not appearance or counts alone.

Sources:
- User-supplied screening overview for 20261022T181655Z-fefcb4ec, 22 October 2026. Person-pose record counts across 782 frames, not unique players or accuracy.
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- Pitch reference comparison separately showed 782/782 accepted fits for Roboflow versus 306/782 for the retired nano model. Fit coverage does not establish landmark accuracy.

## 17. Progress against the exposé (35 seconds)

The implemented system covers interchangeable processing, model runners, coordinate remapping, provenance, reports and videos. The football pilot completes thirty-five of thirty-six jobs, with unresolved crop and tiling cases. Official COCO evaluation is integrated and historical YOLOv8x and OpenPose checks exist, but I am not presenting them as a confirmed YOLO26 score. The key remaining thesis deliverable is football accuracy evaluation against aligned independent joint labels. Selected combinations, broader held-out testing and final preprocessing guidelines follow that step. Confidence and temporal-stability summaries are also not yet completed study results.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Repository experiment.md, TODOS.md and configs/lyra-preprocessing-screen.yaml at commit 21c1068. Status also uses the user-supplied 20261022T181655Z-fefcb4ec screening results.
- Historical user-supplied COCO outputs: YOLOv8x AP=0.684, OpenPose AP=0.296. These are not YOLO26 results.

## 18. Remaining thesis work (25 seconds)

Next we need the football reference-data contract and reproducible accuracy scoring. Resolve the suspicious runner cases before comparing methods, then evaluate individual steps and selected combinations on held-out material. The thesis should conclude with model-specific evidence about transferability and the quality-cost tradeoff. Tracking-coordinate and radar applications remain possible later work, outside this evaluation.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Repository experiment.md, TODOS.md and configs/lyra-preprocessing-screen.yaml at commit 21c1068. Status also uses the user-supplied 20261022T181655Z-fefcb4ec screening results.

## 19. Football reference-data protocol (backup)

The planned primary reference is the DFL/Werder skeletal data named in the exposé. Confirm joint names, visibility, timestamps and camera calibration before choosing projection and alignment procedures. Sample varied conditions and freeze sequence-level development and test splits. Include targets missed by the models. Reference visibility determines eligibility. Independent review should record uncertain labels and alignment errors. Reviewed manual joint annotation would be a contingency requiring agreement if the planned reference is unavailable, rather than a completed or promised replacement. Manual tactical match annotations are outside the exposé scope. Ground-plane pitch calibration cannot project arbitrary three-dimensional body joints.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Repository experiment.md, TODOS.md and configs/lyra-preprocessing-screen.yaml at commit 21c1068. Status also uses the user-supplied 20261022T181655Z-fefcb4ec screening results.

## 20. Experimental matrix (backup)

The pilot uses one baseline and eleven individual methods, with three pose pipelines on 782 identical frames. Bicubic and learned super-resolution share the same enlargement factor. Start with individual processing effects; only later test selected combinations. COCO currently checks model functionality on unchanged inputs, rather than a complete preprocessing ablation.

Sources:
- Repository experiment.md, TODOS.md, configs/lyra-preprocessing-screen.yaml and configs/lyra-preprocessing-base.yaml at commit 21c1068.
- User-supplied Master Thesis.pdf, 10 pages, especially pp. 7–9. These are research/literature notes. Latest screening results supersede older implementation-status notes on p. 9.
- configs/lyra-preprocessing-screen.yaml.

## 21. On-pitch overview 1/2 (backup)

Use this table for discussion. These are record counts across the pilot clip. They do not rank pose accuracy. HRNet tiling and YOLO cropping have suspicious raw outputs, while OpenPose cropping fails. Repeated-frame observations are correlated. The current report mainly has warm-cache timings, so it cannot establish total cold preprocessing cost.

Sources:
- User-supplied screening overview for 20261022T181655Z-fefcb4ec, 22 October 2026. Person-pose record counts across 782 frames, not unique players or accuracy.

## 22. On-pitch overview 2/2 (backup)

Use this table for discussion. These are record counts across the pilot clip. They do not rank pose accuracy. HRNet tiling and YOLO cropping have suspicious raw outputs, while OpenPose cropping fails. Repeated-frame observations are correlated. The current report mainly has warm-cache timings, so it cannot establish total cold preprocessing cost.

Sources:
- User-supplied screening overview for 20261022T181655Z-fefcb4ec, 22 October 2026. Person-pose record counts across 782 frames, not unique players or accuracy.

## 23. PCP: complete body parts (backup)

For a limb connecting reference joints a and b, compute its ground-truth length L. Under this explicit strict variant, both predicted endpoints must lie within alpha times L of their corresponding labels. Average the binary outcomes over eligible limbs. Freeze a limb list; one proposed set is left and right upper arms, forearms, thighs and lower legs. Both endpoints must have usable labels and L must be nonzero. An unmatched person or missing predicted endpoint fails the limb in the proposed end-to-end score. Alpha equals 0.5 here only for illustration. PCP variants differ, so report this convention rather than implying every published PCP score is directly comparable.

Sources:
- Supplementary football metric conventions are proposed, not yet implemented results. Numeric examples are illustrative, not experimental observations.
- Definition requested in the user’s research notes: both endpoint errors relative to reference limb length.

## 24. PCK-AUC: threshold curve (backup)

Sweep the PCK threshold while keeping the reference set, scale and matching fixed. Integrate the resulting PCK curve over a declared interval and divide by interval width. The normalized value is between zero and one. This three-point illustration uses trapezoidal integration: the two areas are 0.0125 and 0.0375, so their sum divided by 0.10 is 0.50. A real protocol needs a fixed sufficiently dense grid. Sampling and integration conventions differ between libraries; MMPose, for example, uses a discrete threshold average. Do not compare AUC values with different normalizers, ranges or quadrature rules. This is not AP’s precision-recall area.

Sources:
- MMPose metric definitions (implementations vary in normalization and aggregation): https://mmpose.readthedocs.io/en/latest/_modules/mmpose/evaluation/functional/keypoint_eval.html
- Supplementary football metric conventions are proposed, not yet implemented results. Numeric examples are illustrative, not experimental observations.

## 25. Proxy metrics and runtime (backup)

Average joint confidence is a diagnostic over K valid predicted joint scores. Different models need not calibrate those scores identically, so higher confidence is not proof of better localization. Coverage additionally requires matched independent labels and a fixed confidence threshold; N includes all eligible reference joints in the proposed end-to-end convention. Existing tables report person-pose record yield rather than detected-joint rate. Temporal stability remains a proposed metric. It requires consistent associations and a way to distinguish true motion and camera motion from jitter. Over-smoothed but wrong poses can look stable. Association used solely to compute this diagnostic does not turn player-tracking or radar evaluation into a thesis objective.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Repository experiment.md, TODOS.md and configs/lyra-preprocessing-screen.yaml at commit 21c1068. Status also uses the user-supplied 20261022T181655Z-fefcb4ec screening results.
- TODOS.md lists confidence summaries, detected-keypoint rate and temporal stability among pending evaluator work.

## 26. AP worked example (backup)

These illustrative outcomes assume one-to-one matching has already been performed at a fixed OKS threshold. Cumulate TP and FP in descending person-score order. Replace precision at a recall by the best precision attained at that or any greater recall. Among 101 recall levels, 34 are at or below one third and receive 1.00; the remaining 67 receive 0.75. AP at this threshold is 0.834. Maximum recall is 1.00. The late false positive does not lower the envelope here. Repeat matching at every OKS threshold before averaging; these are not measured project results.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- Illustrative list and arithmetic authored for this presentation.

## 27. OKS worked example (backup)

The first column changes localization error while holding area and tolerance fixed. Averaging the three joint contributions gives approximately 0.581. Reducing person area increases the penalty for the same pixel error. The example uses a hypothetical k for arithmetic clarity; production COCO has fixed per-joint sigmas and k equals twice sigma. Ground-truth visibility zero is excluded, while labelled occluded and visible joints both contribute. Predicted joint confidence is neither ground-truth visibility nor a direct multiplicative weight in official OKS. Person confidence is used separately for ranking.

Sources:
- Official COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- Numeric example is illustrative; it is not the reported accuracy of any checkpoint.

## 28. Architecture mechanisms and bottlenecks (backup)

HRNet is a single-person pose network inside our multi-person detector-plus-crop system. It repeatedly fuses parallel-resolution features and predicts joint heatmaps. The project uses a COCO-trained W32 checkpoint with 256 by 192 pose input. OpenPose predicts evidence across the full image and uses part-affinity fields to associate limbs without first requiring a person detection box. Its convolutional work is largely shared, though grouping still depends on detected candidates. YOLO26 Pose jointly predicts instance boxes and keypoints using a task-specific head, without an external detector-to-HRNet pass. Preprocessing may change box detection, joint evidence or associations. None of these labels guarantees higher accuracy on this football video. BODY_25 is converted to the common COCO17 order before comparison.

Sources:
- Sun et al., Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019): https://arxiv.org/abs/1902.09212
- Cao et al., Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (CVPR 2017): https://arxiv.org/abs/1611.08050
- Ultralytics YOLO26 architecture and pose task: https://docs.ultralytics.com/models/yolo26/ ; https://docs.ultralytics.com/tasks/pose/
- Project source: runners/hrnet/run.py, src/football_pose/hrnet_input.py, model_mapping.py and coco_evaluation.py; configs/lyra-preprocessing-base.yaml at commit 21c1068.

## 29. Evaluation protocol decisions (backup)

Keep the official COCO benchmark unchanged. For football, agree on reference projection, timestamps and visibility, then freeze matching and the evaluation population. This deck proposes micro-averaged PCK normalized by the reference-box diagonal, with missing predictions counted incorrect and missing labels excluded. Declare that convention before scoring and distinguish end-to-end from matched-only analysis. PCP needs the explicit both-endpoints rule and limb list. AP needs a defined person score and sufficient low-score predictions. Football scenes with more than twenty targets need their own declared prediction cap, rather than silently adopting the COCO limit. Use sequence-level splits and uncertainty estimates. DFL labels serve evaluation only, with no tracking/event/sensor fusion into inference.

Sources:
- User-supplied Exposé (1).pdf, 6 pages: pp. 1–3 motivation and transferability, p. 4 evaluation methodology, pp. 4–5 exclusions. /Users/larry/Downloads/Exposé (1).pdf
- Repository experiment.md, TODOS.md and configs/lyra-preprocessing-screen.yaml at commit 21c1068. Status also uses the user-supplied 20261022T181655Z-fefcb4ec screening results.
