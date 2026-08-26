# Automated experiment results overview

The quick comparison table is generated from saved `summary.json` files rather than
maintained manually. Canonical predictions, manifests, and logs remain in the configured
experiment output directory and are not committed to Git.

Generate or refresh the table on Lyra with:

```bash
python -m football_pose build-overview \
  /home/vincent/football-pose-results \
  --model yolo-pose \
  --model openpose-body25
```

The command recursively finds experiment summaries, groups results with the same input
and preprocessing configuration, merges model jobs, locates the full-frame baseline,
and writes:

```text
/home/vincent/football-pose-results/results-overview/records.md
```

The generated table contains only objective fields:

- preprocessing configuration
- record count for each selected model
- percentage change from that model's full-frame baseline

No promotion, rejection, or quality decision is generated. A record is one raw
person-pose prediction, not a confirmed correct detection. Overlapping tiles can create
duplicates, so the table is a detection-yield overview rather than an accuracy table.

Omit the `--model` options to include every model ID found in the summaries. Future
models, including HRNet, receive their own record and baseline columns automatically.
Use `--output PATH` to choose another Markdown destination. Detailed methodology and
output-file definitions remain in [`experiment.md`](../experiment.md).
