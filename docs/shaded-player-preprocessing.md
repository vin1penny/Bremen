# Shaded-player enhancement experiment

The `shaded-players` condition applies **gamma brightening → CLAHE → unsharp
masking** before pose inference. It is one composite condition consisting of three
ordered processors. It processes the entire frame, including sunlit regions; it
does not detect shadows or select individual players.

## Mechanism and hypothesis

| Step | Settings | Purpose and hypothesis |
| --- | --- | --- |
| Gamma brightening | gamma 1.2; output = 255 × (input / 255)^(1 / gamma), applied to each BGR channel | Lift dark and middle intensities so shaded clothing and limbs are more visible to the model. In this implementation, gamma greater than 1 brightens. |
| CLAHE | Lab luminance only; clip limit 2.0; grid 8 × 8 | Redistribute local contrast after brightening, potentially separating a player's silhouette from nearby grass. The grid counts regions, not pixels. |
| Unsharp masking | Gaussian kernel 5 × 5, sigma 1.0; amount 0.75 | Reinforce existing edges after the contrast changes: output = 1.75 × image − 0.75 × blurred image, clipped to the image range. |

The combined hypothesis is that complementary brightness, contrast and edge
changes improve detection of shaded players. This is a hypothesis to test, not an
established improvement. None of these steps recovers missing spatial detail.
The combination can amplify noise and compression artefacts, create halos, and
clip highlights on sunlit players. Processing order matters and intermediate
images use the existing uint8 implementations.

The frame size, frame count and mapping back to source coordinates stay unchanged.
There is no new trained model, checkpoint or dependency for this condition.

## Comparison and execution

[The example suite](../configs/lyra-shaded-players.example.yaml) compares baseline,
gamma alone, CLAHE alone, unsharp alone and the combined condition across the same
three models: **15 model jobs**. The settings match the existing single-step
suite, making the component comparisons interpretable. This is an initial
comparison, not a full factorial analysis: it does not isolate every pairwise
interaction or establish an optimal order.

On Lyra, after checking the base configuration and reserving the physical GPU,
replace `N` with that GPU's index:

```shell
python -m football_pose screen-preprocessing configs/lyra-shaded-players.example.yaml --device N --check-only
python -m football_pose screen-preprocessing configs/lyra-shaded-players.example.yaml --device N
```

The suite saves expanded configurations, the usual per-model predictions and
annotated videos, and baseline-relative comparison CSV/JSON/Markdown files.
Combination-run videos use the `composite` subdirectory. Processing time remains
an additional metric; each processing stage also has timing in the run summary
when preprocessing is executed. A warm cache does not remeasure the stages.
All three stages and their order participate in the preprocessing cache key.

Inspect shaded and sunlit players separately when reviewing outputs. More raw or
on-pitch predictions alone cannot establish improved player recall, correct
keypoints or fewer hallucinations; the planned reference-based evaluation is
still needed. No detection improvement has been measured for this preset yet.

## Direct comparison with jersey enhancement

For the requested comparison of the two added methods, use
[lyra-additional-methods.yaml](../configs/lyra-additional-methods.yaml): baseline,
`jersey-colors` and `shaded-players`, each with YOLO26, OpenPose and HRNet (**nine
jobs**). The complete `sample_30.mov` is processed. All three conditions share
model thresholds, checkpoints, pitch filtering and the source clip. This focused
run does not rerun the separate gamma/CLAHE/unsharp component controls.

The jersey condition uses strength 0.35 and the four green/white RGB samples from
the earlier local preview. Sample coordinates are recorded in metadata. These
colours are specific to that clip; resample the palette if the footage changes.
The jersey diagnostics export frames 0, 260, 520 and 781 independently of model
inference. Baseline and shade conditions do not generate jersey masks. Unknown
lighting metadata stays empty and has no processing effect.

First push the updated pipeline, configuration and launcher from the local
checkout and pull them on Lyra. Then reserve a GPU in the university semaphore,
create or attach a tmux session, and run:

```shell
bash runners/start-football-additions.sh N
```

Replace `N` with the reserved physical GPU index. The launcher checks that GPU's
current usage, acquires the same per-account lock as the existing COCO launcher,
checks assets and containers, and runs the comparison with a persistent log.
Detach using Ctrl-b, then d; the run continues inside tmux. A usage check does
not grant a university GPU reservation.

Results are under
`/mnt/storage2/vincent/football-pose/results/football/additional-methods/<run-id>/`.
The launcher prints the exact run path. Review `comparison.csv` for raw/on-pitch
prediction counts and processing/model times, `comparison.md` for the compact
table, and `screening.json` for full status and provenance. Failed jobs remain
missing results. Jersey diagnostics and annotated videos are linked from the
condition summaries. More predictions are not automatically better detections;
check videos for spurious outputs and retain the reference-evaluation limitation.

## Optional lighting metadata

The suite includes optional `metadata.lighting` fields for weather, actual
recording start time with UTC offset, venue coordinates, pitch orientation,
camera side/initial heading and observed shadow notes. Fill only known values.
These descriptive values are saved with the experiment configuration. They
**do not change pixels, enable a mask, alter processing strength or invalidate
the preprocessing cache**. Jersey-color metadata remains separately supported.

Weather and camera direction are useful context but cannot by themselves locate
shaded players. Date, time and geographic position determine the sun's direction
([NOAA solar-position equations](https://gml.noaa.gov/grad/solcalc/solareqns.PDF)).
Predicting cast shadows also needs the geometry of objects blocking sunlight,
such as stadium roofs and stands. A camera heading describes the view; it does
not describe those obstructions. Panning, tilting and zooming also change how a
pitch region maps to image pixels.

A practical later extension would be a manually reviewed or image-estimated
shadow region, with a soft transition at its boundary, evaluated as a separate
condition against this global one. Dark pixels alone are insufficient: dark
material can resemble shadow, as discussed by
[Guo et al., CVPR 2011](https://dhoiem.cs.illinois.edu/publications/cvpr11_shadow.pdf).
Their method is related shadow-detection research, not evidence for this exact
three-step enhancement chain. A pitch homography can help map ground regions,
but it does not identify which above-ground body parts are shaded. This extension
is proposed and is not implemented by the current preset.

## Configuration format

Football screening accepts either a single `processor` or a nonempty ordered
`processors` list per condition. Do not specify both. Only `baseline` may omit
processing entirely. Existing single-step suites retain their settings and job
counts. `screen-coco` remains a single-step comparison and rejects composite
conditions explicitly. The separate jersey diagnostic exporter still requires
standalone jersey enhancement and is not enabled for this suite.
