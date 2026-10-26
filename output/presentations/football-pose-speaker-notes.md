# Professor progress presentation

11 main slides, target 10:00. Three backup slides are for questions.

## 1. Pose estimation in football video (25 seconds)

This project studies how image preprocessing changes human pose estimation in football scouting video. Today I will explain the pipeline, how the experiments run on Lyra, and what the latest single-step screening tells us. I will also separate the results we can already use from the integration problems that still need work.

Sources:
- Repository experiment.md and configs/lyra-preprocessing-base.yaml, commit 21c1068.
- Cover photograph: original footage/sample_30.mov, frame 140.

## 2. Research question (55 seconds)

In scouting footage, a player can occupy only a small part of a wide image. We also have strong lighting differences, shadows, motion and many people outside the pitch. The camera moves, so a fixed region cannot reliably separate players from the crowd. My question is which preprocessing operations help under these conditions, and whether the effect depends on the pose model. The aim is to compare complete pipelines, rather than assume that one image enhancement helps every model. Later, we want to compare pose-derived player positions with tracking data. First, the detection and calibration stages need to behave consistently.

Sources:
- Repository experiment.md and configs/lyra-preprocessing-base.yaml, commit 21c1068.
- Source photograph: original footage/sample_30.mov, frame 580.

## 3. Pipeline structure (75 seconds)

The pipeline starts from the original video. The upper branch applies the selected preprocessing method, saves a lossless artifact and sends the resulting images to the pose models. Every output maps back to the original frame coordinates. HRNet needs a person detector to provide its crops. The lower branch runs the pitch model on each original frame, independently of the experimental preprocessing. It predicts field landmarks and fits a homography against the known pitch template. We then merge the branches: remove cross-tile duplicate records and classify each estimated player ground point against the pitch boundaries. Invalid geometry stays unavailable instead of silently counting a detection as inside. Raw records remain intact so that every filtering decision can be inspected.

Sources:
- Repository experiment.md and configs/lyra-preprocessing-base.yaml, commit 21c1068.
- src/football_pose/experiments.py; src/football_pose/pitch_filter.py; src/football_pose/visualization.py.

## 4. Model setup (60 seconds)

We compare three model families. YOLO26x performs multi-person pose estimation. OpenPose is bottom-up and assembles people from detected body parts. HRNet is top-down, so our existing football-trained YOLO player detector supplies its person crops. HRNet then estimates pose at its normal 256 by 192 crop size. Full-frame YOLO uses a 1920 target, and OpenPose uses 1920 by 1088. These settings stay fixed across conditions, so tiling can genuinely change the apparent player scale. All models export a common COCO-17 representation. The pitch model remains YOLOv8x and does not replace or fine-tune any of these person-pose checkpoints.

Sources:
- Repository experiment.md and configs/lyra-preprocessing-base.yaml, commit 21c1068.
- runners/hrnet/run.py; src/football_pose/hrnet_input.py.

## 5. Single-step experimental design (60 seconds)

The current screening uses one approximately thirty-second clip, with 782 frames at 1920 by 1080. There is an unchanged baseline and eleven single processing operations. We compare image scaling, contrast and brightness changes, denoising, sharpening, classical motion deblurring, learned super-resolution, deterministic tiling and detector-guided cropping. The two-by-two tiling condition has ten percent overlap. The super-resolution model is FSRCNN at two-times scale, with bicubic two-times resizing as a useful control. Each condition runs all three pose models. We hold checkpoints and model settings constant, and use the same original-frame pitch calibration throughout. Combinations of processing steps belong to a later stage.

Sources:
- Repository experiment.md and configs/lyra-preprocessing-base.yaml, commit 21c1068.
- configs/lyra-preprocessing-screen.yaml; source clip metadata and supplied overview.

## 6. Lyra execution and reproducibility (50 seconds)

On Lyra, Python coordinates the experiment and each pose model runs in its own container. The models run serially on the selected physical GPU. The screening command applies that device consistently to pose inference, pitch inference and the crop detector. A tmux session lets the run continue after an SSH disconnect. Large inputs, models, caches and output videos live on mounted storage. Each invocation creates a new result directory, while content-based caches can reuse preprocessing safely. The outputs include expanded configurations, checkpoint hashes, logs, raw and filtered predictions, summary tables and annotated videos. This also lets us identify outdated configuration, as happened in the first attempted Roboflow rerun.

Sources:
- Repository server.md and src/football_pose/screening.py, commit 21c1068.
- Mounted root: /mnt/storage2/vincent/football-pose/. No credentials are included in this deck.

## 7. Roboflow improved pitch-fit coverage (65 seconds)

Pitch filtering became important because the pose models also detect people in the crowd and on the sidelines. We compared our earlier nano checkpoint with the reference checkpoint used by Roboflow sports. The comparison kept inference size, confidence thresholds, geometry checks and the Bremen pitch dimensions fixed. Our checkpoint produced accepted geometry on 306 of 782 frames. The reference produced accepted geometry on every frame, and the visual comparison also looked clearly better. Interestingly, it detected slightly fewer confident landmarks on average. More points alone did not imply a better fit. We therefore selected the reference checkpoint in the active configurations. This is evidence about coverage and observed alignment, not a ground-truth accuracy result.

Sources:
- Repository experiment.md and configs/lyra-preprocessing-base.yaml, commit 21c1068.
- Pitch comparison run 20261021T145819Z-1713c4a9: ours 306/782 accepted fits, Roboflow 782/782; visible landmarks/frame 12.49 vs 11.97.
- Reference repository: https://github.com/roboflow/sports/tree/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/examples/soccer
- Production SHA256 28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36. Latest full-screen actual checkpoint hash still requires its JSON; supplied table alone does not verify the hash.

## 8. Processing effects depend on the model (85 seconds)

The latest screening shows very different responses across models. On the unchanged baseline, the pitch filter retains 357 YOLO records, 11,647 OpenPose records and 3,602 HRNet records across the video. CLAHE increases OpenPose yield to 15,338, about thirty-two percent above baseline. Bilateral filtering raises YOLO yield to 2,115, almost six times baseline, though its absolute yield remains low. Tiling gives the highest on-pitch counts for YOLO and OpenPose, but HRNet produces only one record in that condition, which needs investigation. Resizing strongly increases HRNet yield. These are counts of person-pose records across frames, not unique players. A higher count may reflect better coverage, false positives, or remaining duplicate or integration effects. We cannot label these changes as improved accuracy without annotated reference data. The full condition tables are in the backup slides.

Sources:
- User-supplied single-step screening overview, run 20261022T181655Z-fefcb4ec, 22 October 2026, pasted in this conversation on 23 October 2026. Counts are person-pose records, not unique people or ground-truth accuracy. The screening JSON attachment could not be read because of local file-access restrictions.
- Percentages use the same model baseline. Tile postprocessing performs cross-region IoU deduplication, whose behaviour should be verified visually.

## 9. HRNet: resizing explains almost all of the SR yield (60 seconds)

Super-resolution initially appears to be a very strong result for HRNet: on-pitch yield rises from 3,602 to 15,565 records. But ordinary bicubic resizing reaches 15,554, just eleven fewer. That is a difference of about 0.07 percent. At this stage, enlargement explains almost all of the observed yield benefit, while a distinct contribution from learned super-resolution remains unproven. Both conditions take roughly four times the baseline model wall time. Most preprocessing artifacts came from cache, so the tiny preprocessing times in the latest table represent retrieval rather than the cost of computing super-resolution or denoising. Repeated cold and warm measurements are needed before making runtime claims.

Sources:
- User-supplied single-step screening overview, run 20261022T181655Z-fefcb4ec, 22 October 2026, pasted in this conversation on 23 October 2026. Counts are person-pose records, not unique people or ground-truth accuracy. The screening JSON attachment could not be read because of local file-access restrictions.
- Calculations: (15565-15554)/15554 = 0.07072%; model times 352.20 s, 1409.08 s, 1366.36 s.

## 10. Current status and unresolved issues (35 seconds)

The infrastructure now runs nearly the entire suite, but execution success is not enough. HRNet with tiling returns one raw record, YOLO with shared cropping returns zero, and OpenPose fails on cropping. The first two are suspicious even though their jobs report completion. These are raw inference results, so switching the pitch filter cannot explain or repair them. We still need the detailed crop error and targeted checks of crop dimensions, detector handling and coordinate transformations before ranking those conditions.

Sources:
- User-supplied single-step screening overview, run 20261022T181655Z-fefcb4ec, 22 October 2026, pasted in this conversation on 23 October 2026. Counts are person-pose records, not unique people or ground-truth accuracy. The screening JSON attachment could not be read because of local file-access restrictions.
- The latest detailed screening JSON was not readable locally. No OpenPose crop exception or root cause is available in the supplied table.

## 11. Next experiment stage (30 seconds)

The immediate priority is to fix and rerun only the affected integrations. Then we need an annotated football sample and a clearly defined primary accuracy metric, supported by the COCO setup check. After that we can test promising combinations and measure total runtime fairly. Pitch-coordinate player positions, the radar view and comparison with tracking data follow once the geometry and pose stages are reliable. My main discussion point today is which annotation sample and primary metric we should use.

Sources:
- Repository experiment.md and configs/lyra-preprocessing-base.yaml, commit 21c1068.
- Planned accuracy evaluation: OKS-based AP/AR and an explicitly defined football PCK protocol. No football ground-truth scores are claimed.

## 12. Complete on-pitch overview 1/2 (backup)

Use this slide during questions. The table includes all conditions across two backup slides. Changes compare each model with its own baseline. Counts refer to person-pose records across frames. They do not measure accuracy or count unique players.

Sources:
- User-supplied single-step screening overview, run 20261022T181655Z-fefcb4ec, 22 October 2026, pasted in this conversation on 23 October 2026. Counts are person-pose records, not unique people or ground-truth accuracy. The screening JSON attachment could not be read because of local file-access restrictions.

## 13. Complete on-pitch overview 2/2 (backup)

Use this slide during questions. The table includes all conditions across two backup slides. Changes compare each model with its own baseline. Counts refer to person-pose records across frames. They do not measure accuracy or count unique players.

Sources:
- User-supplied single-step screening overview, run 20261022T181655Z-fefcb4ec, 22 October 2026, pasted in this conversation on 23 October 2026. Counts are person-pose records, not unique people or ground-truth accuracy. The screening JSON attachment could not be read because of local file-access restrictions.

## 14. COCO setup validation (backup)

This is the completed earlier COCO check. It evaluated YOLOv8x and OpenPose on the 5,000 validation images using official COCO keypoint evaluation. The AP values were 0.684 and 0.296. These numbers should not be assigned to the current YOLO26 model. The repository includes a YOLO26 benchmark configuration, but no completed YOLO26 COCO score is available in the supplied results. HRNet evaluation also needs a specified detected-box or ground-truth-box protocol before comparison with a reference.

Sources:
- Repository experiment.md and configs/lyra-preprocessing-base.yaml, commit 21c1068.
- User-supplied COCO evaluation logs earlier in this conversation. Protocol: COCO keypoints OKS, maxDets=20, values on 0–1 scale.
