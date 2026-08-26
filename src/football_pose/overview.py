from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any
import uuid


MODEL_LABELS = {
    "yolo-pose": "YOLO",
    "openpose-body25": "OpenPose",
    "hrnet-w32": "HRNet",
}
MODEL_ORDER = {model_id: index for index, model_id in enumerate(MODEL_LABELS)}


@dataclass(frozen=True, slots=True)
class ModelResult:
    records: int | None
    status: str
    updated_at_unix: float


@dataclass(slots=True)
class PipelineResult:
    input_path: str
    processors: list[dict[str, Any]]
    models: dict[str, ModelResult] = field(default_factory=dict)


def _pipeline_key(input_path: str, processors: list[dict[str, Any]]) -> tuple[str, str]:
    return input_path, json.dumps(processors, sort_keys=True, separators=(",", ":"))


def _model_result(job: dict[str, Any]) -> ModelResult:
    outputs = job.get("outputs")
    records: int | None = None
    if isinstance(outputs, dict) and outputs.get("records") is not None:
        try:
            records = int(outputs["records"])
        except (TypeError, ValueError):
            records = None
    return ModelResult(
        records=records,
        status=str(job.get("status", "UNKNOWN")),
        updated_at_unix=float(job.get("updated_at_unix", 0.0)),
    )


def collect_results(
    root: Path,
    *,
    selected_models: set[str] | None = None,
) -> dict[str, list[PipelineResult]]:
    summaries = sorted(root.rglob("summary.json"))
    if not summaries:
        raise ValueError(f"no summary.json files found below {root}")

    pipelines: dict[tuple[str, str], PipelineResult] = {}
    for summary_path in summaries:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read experiment summary {summary_path}: {error}") from error

        configuration = payload.get("configuration")
        jobs = payload.get("jobs")
        if not isinstance(configuration, dict) or not isinstance(jobs, list):
            raise ValueError(f"invalid experiment summary structure: {summary_path}")
        input_path = str(configuration.get("input", "unknown"))
        processors = configuration.get("processors", [])
        if not isinstance(processors, list) or not all(
            isinstance(processor, dict) for processor in processors
        ):
            raise ValueError(f"invalid processor list in experiment summary: {summary_path}")

        key = _pipeline_key(input_path, processors)
        pipeline = pipelines.setdefault(
            key,
            PipelineResult(input_path=input_path, processors=processors),
        )
        for job in jobs:
            if not isinstance(job, dict):
                raise ValueError(f"invalid job in experiment summary: {summary_path}")
            model_id = str(job.get("model_id", "unknown"))
            if selected_models and model_id not in selected_models:
                continue
            candidate = _model_result(job)
            current = pipeline.models.get(model_id)
            if current is None or candidate.updated_at_unix >= current.updated_at_unix:
                pipeline.models[model_id] = candidate

    grouped: dict[str, list[PipelineResult]] = {}
    for pipeline in pipelines.values():
        if pipeline.models:
            grouped.setdefault(pipeline.input_path, []).append(pipeline)
    if not grouped:
        requested = ", ".join(sorted(selected_models or []))
        raise ValueError(f"no jobs found for selected model(s): {requested}")
    return grouped


def _processor_label(processors: list[dict[str, Any]]) -> str:
    if not processors:
        return "Baseline: full frame, unchanged"
    labels: list[str] = []
    for processor in processors:
        processor_type = str(processor.get("type", "unknown"))
        params = processor.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if processor_type == "clahe":
            labels.append("CLAHE")
        elif processor_type == "gamma":
            labels.append(f"Gamma {params.get('gamma', '?')}")
        elif processor_type == "unsharp_mask":
            labels.append("Unsharp mask")
        elif processor_type == "tile":
            rows = params.get("rows", "?")
            columns = params.get("columns", "?")
            overlap = params.get("overlap_ratio")
            overlap_label = "?"
            if isinstance(overlap, (int, float)):
                overlap_label = f"{overlap * 100:g}%"
            labels.append(f"{rows} x {columns} tiling ({overlap_label} overlap)")
        else:
            details = ", ".join(
                f"{key}={json.dumps(value, sort_keys=True)}"
                for key, value in sorted(params.items())
            )
            labels.append(f"{processor_type} ({details})" if details else processor_type)
    return " -> ".join(labels)


def _result_cell(result: ModelResult | None) -> str:
    if result is None:
        return "—"
    if result.records is not None:
        return f"{result.records:,}"
    return result.status


def _delta_cell(result: ModelResult | None, baseline: ModelResult | None) -> str:
    if result is None or result.records is None:
        return "—"
    if baseline is None or baseline.records is None or baseline.records == 0:
        return "n/a"
    change = (result.records - baseline.records) / baseline.records * 100
    return f"{change:+.1f}%"


def render_overview(
    grouped: dict[str, list[PipelineResult]],
    *,
    source_root: Path,
) -> str:
    model_ids = sorted(
        {
            model_id
            for pipelines in grouped.values()
            for pipeline in pipelines
            for model_id in pipeline.models
        },
        key=lambda model_id: (MODEL_ORDER.get(model_id, len(MODEL_ORDER)), model_id),
    )
    lines = [
        "# Automated experiment result overview",
        "",
        f"Generated from `summary.json` files below `{source_root}`.",
        "Values are raw person-pose record counts; no manual decisions are included.",
        "",
    ]
    for input_path in sorted(grouped):
        pipelines = grouped[input_path]
        baseline = next((pipeline for pipeline in pipelines if not pipeline.processors), None)
        pipelines.sort(
            key=lambda pipeline: (
                bool(pipeline.processors),
                _processor_label(pipeline.processors),
            )
        )
        lines.extend([f"## Input: `{input_path}`", ""])
        headers = ["Processing"]
        alignments = ["---"]
        for model_id in model_ids:
            label = MODEL_LABELS.get(model_id, model_id)
            headers.extend([f"{label} records", f"{label} vs baseline"])
            alignments.extend(["---:", "---:"])
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(alignments) + " |")
        for pipeline in pipelines:
            cells = [_processor_label(pipeline.processors).replace("|", "\\|")]
            for model_id in model_ids:
                result = pipeline.models.get(model_id)
                baseline_result = baseline.models.get(model_id) if baseline is not None else None
                delta = "reference" if not pipeline.processors and result is not None else _delta_cell(
                    result, baseline_result
                )
                cells.extend([_result_cell(result), delta])
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "Record counts are detection-yield diagnostics, not accuracy metrics.",
            "Overlapping-tile results can contain duplicate predictions.",
            "",
        ]
    )
    return "\n".join(lines)


def write_overview(
    root: Path,
    *,
    output: Path | None = None,
    selected_models: set[str] | None = None,
) -> Path:
    root = root.resolve()
    grouped = collect_results(root, selected_models=selected_models)
    output_path = output or root / "results-overview" / "records.md"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(render_overview(grouped, source_root=root), encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path
