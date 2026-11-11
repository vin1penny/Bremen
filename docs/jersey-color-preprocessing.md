# Jersey-color preprocessing

`jersey_color_enhance` changes the input image before pose inference. It selects
pixels close to a match's jersey-color samples and blends in local luminance
contrast (CLAHE) and a modest chroma boost. Each frame still produces one frame
with the same dimensions and coordinate mapping. All three existing model
runners can consume the resulting lossless preprocessing artifact, including
HRNet's person detector.

```text
Original frame + match palette
    → soft color-similarity mask
    → selective contrast/chroma enhancement
    → YOLO26 / OpenPose / HRNet
```

The method selects colors, not people. Advertising, spectators, pitch lines and
other objects with matching colors can also be enhanced. Green kits can overlap
with grass; white kits can overlap with pitch lines. Effects on detection and
pose accuracy must be measured on annotated football footage.

## Configure a match

Add the following processor to an experiment's `processors` list. The two RGB
hex values below are **examples**, not colors measured from the included clip.
Quote the values because `#` otherwise starts a YAML comment.

```yaml
processors:
  - type: jersey_color_enhance
    params:
      colors: ["#B53238", "#315EA8"]
      color_tolerance: 25.0
      strength: 0.35
      chroma_gain: 1.15
      clip_limit: 2.0
      tile_grid: [8, 8]
```

Sample the visible jerseys from several representative frames using a color
picker. Include both teams and any goalkeepers/referees you want to enhance.
Use samples from sun and shadow when necessary, avoiding badges, skin and
background. Several samples for one kit are allowed; overlapping matches do
not increase the enhancement beyond the chosen strength. Nominal club colors
are only starting guesses because the footage's exposure and white balance
change their appearance.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `colors` | Required | Non-empty list of RGB `#RRGGBB` samples from the clip. |
| `color_tolerance` | `25.0` | Maximum Euclidean distance in CIE Lab; `(0, 100]`. Selection fades smoothly to zero at this distance. A wider tolerance selects more background too. |
| `strength` | `0.35` | Maximum blend weight, from `0` (exact identity) to `1`. Pixels with weaker color similarity receive a smaller weight. |
| `chroma_gain` | `1.15` | Multiplier on Lab a/b channels, from `1` (no chroma boost) to `2`. Local contrast still applies at `1`. |
| `clip_limit` | `2.0` | Positive CLAHE contrast limit. |
| `tile_grid` | `[8, 8]` | Positive integer column/row tile counts for CLAHE. |

The implementation uses floating-point
[OpenCV CIE Lab conversion](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html),
with L in 0–100 and signed a/b channels. Lightness participates in color matching,
so white and black samples are distinguishable. This also means a strongly shaded
jersey may need a separate sample. The mask is calculated from the incoming frame
before enhancement; its weights express color similarity, not detection confidence.

CLAHE calculates a luminance enhancement across the frame, but its output and the
chroma boost are blended in only through the mask. Pixels outside the selected
color range remain byte-for-byte unchanged. Gamut clipping and output rounding
can affect selected pixels. The original input array is never modified.

For a first comparison, use this as the only preprocessing step. Other steps
before it can change the colors used for matching. Palette and parameter changes
are recorded in provenance and produce distinct preprocessing cache keys.

## Compare the three models on Lyra

Use this separate football suite after the current COCO run has finished and the
new code is available on Lyra. Keep the running experiment's checkout in place
while its jobs are active.

```bash
cp configs/lyra-jersey-colors.example.yaml configs/lyra-jersey-colors.yaml
```

Edit `colors` in the copied file. Its YAML anchor shares that palette between the
two enhancement conditions. The base config supplies the football clip, model
checkpoints and pitch settings; to use a different match, copy that base config,
change its `input`, and point `base_config` at the copy. Paths are relative to the
configuration file unless absolute.

The suite runs baseline, `jersey-mild` (`strength: 0.25`) and `jersey-moderate`
(`strength: 0.50`), each with YOLO26, OpenPose and HRNet: nine model jobs. All other
model and pitch settings are shared. After reserving an idle GPU, use its physical
index in both commands below (example GPU 0):

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -m football_pose screen-preprocessing configs/lyra-jersey-colors.yaml --device 0 --check-only
.venv/bin/python -m football_pose screen-preprocessing configs/lyra-jersey-colors.yaml --device 0
```

Run the experiment inside tmux as with the existing suite. Results go under
`/mnt/storage2/vincent/football-pose/results/football/jersey-colors`; the screening
report records detection counts, runtimes, expanded configs and output videos.
Those counts measure yield, not accuracy. Compare recall, precision and pose
metrics against annotations before deciding whether the filter helps, and choose
the palette/tuning on separate calibration frames rather than the evaluation set.

This match-specific processor is opt-in. The generic football and COCO suites
retain their original twelve conditions. No new packages or model retraining
are required.
