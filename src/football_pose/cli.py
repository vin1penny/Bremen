from __future__ import annotations

import json
from pathlib import Path

import typer

from football_pose.artifacts import iter_artifact, load_artifact_manifest
from football_pose.configuration import load_config
from football_pose.experiments import ExperimentRunner
from football_pose.overview import write_overview
from football_pose.preprocessing import REGISTRY


app = typer.Typer(no_args_is_help=True, help="Run modular football pose experiments.")


@app.command("validate-config")
def validate_config(
    config: Path,
    check_paths: bool = typer.Option(True, help="Require input/checkpoint files to exist."),
) -> None:
    loaded = load_config(config, check_paths=check_paths)
    typer.echo(loaded.model_dump_json(indent=2))


@app.command("list-processors")
def list_processors() -> None:
    for name in sorted((*REGISTRY, "crop")):
        typer.echo(name)


@app.command("build-overview")
def build_overview(
    root: Path,
    output: Path | None = typer.Option(
        None,
        help="Markdown destination; defaults to ROOT/results-overview/records.md.",
    ),
    models: list[str] | None = typer.Option(
        None,
        "--model",
        help="Include only this model ID; repeat to select multiple models.",
    ),
) -> None:
    if not root.is_dir():
        raise typer.BadParameter(f"result root does not exist: {root}")
    try:
        output_path = write_overview(
            root,
            output=output,
            selected_models=set(models or []),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(output_path)


@app.command("inspect-artifact")
def inspect_artifact(path: Path, count_frames: bool = False) -> None:
    manifest = load_artifact_manifest(path)
    payload = manifest.model_dump(mode="json")
    if count_frames:
        payload["decoded_frame_count"] = sum(1 for _ in iter_artifact(path))
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("run")
def run_experiment(
    config: Path,
    models: list[str] | None = typer.Option(
        None, "--model", help="Run only this model id; repeat to select multiple models."
    ),
    prepare_only: bool = typer.Option(
        False, help="Decode, preprocess, and cache the input without running a model."
    ),
) -> None:
    loaded = load_config(config, check_paths=True)
    requested = set(models or [])
    if requested:
        available = {model.id for model in loaded.models}
        unknown = requested - available
        if unknown:
            raise typer.BadParameter(
                f"unknown model id(s): {', '.join(sorted(unknown))}; "
                f"available: {', '.join(sorted(available))}"
            )
        loaded.models = [model for model in loaded.models if model.id in requested]
    runner = ExperimentRunner(loaded)
    if prepare_only:
        prepared = runner.prepare()
        result = {
            "experiment_id": prepared.experiment_id,
            "pipeline_id": prepared.pipeline_id,
            "artifact": str(prepared.artifact_path),
            "cache_hit": prepared.cache_hit,
            "frame_count": prepared.artifact_manifest.frame_count,
            "format": prepared.artifact_manifest.format,
            "preprocessing_stage_timings": prepared.stage_timings,
            "success": True,
        }
    else:
        result = runner.run()
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result["success"]:
        raise typer.Exit(code=1)
