"""Run TOML-defined experiment matrices through uv-managed CLI entry points."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class GridRun:
    """One fully expanded experiment command."""

    experiment: str
    context: dict[str, Any]
    command: tuple[str, ...]
    env: dict[str, str]
    cwd: Path

    @property
    def run_id(self) -> str:
        payload = json.dumps(
            {
                "experiment": self.experiment,
                "context": self.context,
                "command": self.command,
            },
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", self.experiment).strip("-")
        return f"{name}-{digest}"


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a TOML table")
    return value


def _render(value: object, context: Mapping[str, Any]) -> str:
    if isinstance(value, str):
        try:
            return value.format_map(context)
        except KeyError as exc:
            raise ValueError(f"Unknown matrix placeholder {exc}") from exc
    if isinstance(value, (int, float)):
        return str(value)
    raise ValueError(f"Expected a string or number, got {type(value).__name__}")


def _expand_matrix(matrix: Mapping[str, object]) -> Iterator[dict[str, Any]]:
    if not matrix:
        yield {}
        return

    keys = list(matrix)
    choices: list[list[object]] = []
    for key in keys:
        values = matrix[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"matrix.{key} must be a non-empty TOML array")
        choices.append(values)

    for combination in itertools.product(*choices):
        context: dict[str, Any] = {}
        for key, value in zip(keys, combination, strict=True):
            if isinstance(value, dict):
                context.update(value)
            else:
                context[key] = value
        yield context


def _arguments_to_cli(arguments: Mapping[str, object], context: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for raw_name, raw_value in arguments.items():
        option = raw_name if raw_name.startswith("-") else f"--{raw_name.replace('_', '-')}"
        if isinstance(raw_value, bool):
            if raw_value:
                result.append(option)
            continue
        if raw_value is None:
            continue
        if isinstance(raw_value, list):
            for item in raw_value:
                result.extend((option, _render(item, context)))
            continue

        rendered = _render(raw_value, context)
        if rendered:
            result.extend((option, rendered))
    return result


def _command(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("experiment.command must be a string or non-empty array of strings")


def build_runs(config: Mapping[str, Any], base_cwd: Path) -> list[GridRun]:
    """Expand every experiment matrix in a parsed TOML configuration."""
    experiments = config.get("experiment")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Configuration must contain at least one [[experiment]] table")

    settings = _mapping(config.get("settings", {}), "settings")
    defaults = _mapping(settings.get("defaults", {}), "settings.defaults")
    default_arguments = _mapping(defaults.get("arguments", {}), "settings.defaults.arguments")
    default_env = _mapping(defaults.get("env", {}), "settings.defaults.env")
    default_cwd = defaults.get("cwd", ".")

    runs: list[GridRun] = []
    names: set[str] = set()
    for raw_experiment in experiments:
        experiment = _mapping(raw_experiment, "experiment")
        name = experiment.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Every experiment must define a non-empty name")
        if name in names:
            raise ValueError(f"Duplicate experiment name: {name}")
        names.add(name)

        command = _command(experiment.get("command"))
        matrix = _mapping(experiment.get("matrix", {}), f"experiment.{name}.matrix")
        arguments = {
            **default_arguments,
            **_mapping(experiment.get("arguments", {}), f"experiment.{name}.arguments"),
        }
        env = {
            **default_env,
            **_mapping(experiment.get("env", {}), f"experiment.{name}.env"),
        }
        cwd = Path(str(experiment.get("cwd", default_cwd)))
        if not cwd.is_absolute():
            cwd = (base_cwd / cwd).resolve()

        for context in _expand_matrix(matrix):
            rendered_env = {str(key): _render(value, context) for key, value in env.items()}
            rendered_command = tuple(command + _arguments_to_cli(arguments, context))
            runs.append(
                GridRun(
                    experiment=name,
                    context=context,
                    command=rendered_command,
                    env=rendered_env,
                    cwd=cwd,
                )
            )

    ids = [run.run_id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment configuration contains duplicate expanded runs")
    return runs


def _read_config(source: str) -> tuple[str, str, dict[str, Any]]:
    candidate = Path(source)
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
        label = str(candidate.resolve())
        stem = candidate.stem
    else:
        name = source if source.endswith(".toml") else f"{source}.toml"
        resource = files("simplexlora.experiments.configs").joinpath(name)
        if not resource.is_file():
            raise ValueError(f"Unknown grid configuration: {source}")
        text = resource.read_text(encoding="utf-8")
        label = f"package:{name}"
        stem = Path(name).stem
    return label, stem, tomllib.loads(text)


def _select_runs(
    runs: Sequence[GridRun],
    selectors: Sequence[str],
) -> list[GridRun]:
    if not selectors:
        return list(runs)
    selected = [run for run in runs if any(fnmatch.fnmatchcase(run.experiment, pattern) for pattern in selectors)]
    if not selected:
        raise ValueError(f"No experiments matched selectors: {', '.join(selectors)}")
    return selected


def _manifest_path(explicit: Path | None, settings: Mapping[str, Any], stem: str, cwd: Path) -> Path:
    if explicit is not None:
        return explicit.resolve()
    configured = settings.get("manifest", f"outputs/grid/{stem}.json")
    return (cwd / str(configured)).resolve()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "runs": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("runs"), dict):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _show_run(run: GridRun) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(run.env.items()))
    command = shlex.join(run.command)
    return f"{prefix} {command}".strip()


def _parse_selectors(values: Sequence[str]) -> list[str]:
    return [item for value in values for item in value.split(",") if item]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Built-in config name or a path to a TOML file")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Run matching experiment names; repeat or use comma-separated patterns",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments, ignoring settings.default_select",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--resume", action="store_true", help="Skip runs marked successful in the manifest")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed command")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit the number of selected runs")
    parser.add_argument("--manifest", type=Path, default=None, help="Override the JSON manifest path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    label, stem, config = _read_config(args.config)
    settings = _mapping(config.get("settings", {}), "settings")
    base_cwd = Path.cwd()
    runs = build_runs(config, base_cwd)

    selectors = _parse_selectors(args.select)
    if not selectors and not args.all and settings.get("default_select"):
        raw_default = settings["default_select"]
        if not isinstance(raw_default, list) or not all(isinstance(item, str) for item in raw_default):
            raise ValueError("settings.default_select must be an array of strings")
        selectors = raw_default
    runs = _select_runs(runs, selectors)
    if args.max_runs is not None:
        if args.max_runs < 1:
            raise ValueError("--max-runs must be positive")
        runs = runs[: args.max_runs]

    if args.dry_run:
        for number, run in enumerate(runs, start=1):
            print(f"[{number}/{len(runs)}] [{run.run_id}] {_show_run(run)}")
        return 0

    manifest_path = _manifest_path(args.manifest, settings, stem, base_cwd)
    manifest = _read_manifest(manifest_path) if args.resume else {"version": 1, "runs": {}}
    manifest.update({"config": label, "updated_at": _timestamp()})
    records = manifest["runs"]
    failures: list[str] = []
    stop_on_error = bool(settings.get("stop_on_error", True)) and not args.keep_going

    for number, run in enumerate(runs, start=1):
        previous = records.get(run.run_id)
        if args.resume and isinstance(previous, dict) and previous.get("status") == "success":
            print(f"[{number}/{len(runs)}] Skipping successful run {run.run_id}")
            continue

        print(f"[{number}/{len(runs)}] Starting {run.run_id}")
        print(f"  {_show_run(run)}")
        started_at = _timestamp()
        record = {
            "experiment": run.experiment,
            "context": run.context,
            "command": list(run.command),
            "env": run.env,
            "cwd": str(run.cwd),
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "returncode": None,
        }
        records[run.run_id] = record
        _write_manifest(manifest_path, manifest)

        process_env = {**os.environ, **run.env}
        try:
            result = subprocess.run(run.command, cwd=run.cwd, env=process_env, check=False)
            record["returncode"] = result.returncode
            record["status"] = "success" if result.returncode == 0 else "failed"
        except OSError as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"Failed to start command: {exc}")

        record["finished_at"] = _timestamp()
        _write_manifest(manifest_path, manifest)
        if record["status"] != "success":
            failures.append(run.run_id)
            if stop_on_error:
                break

    manifest["updated_at"] = _timestamp()
    _write_manifest(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")
    if failures:
        print(f"Failed runs: {', '.join(failures)}")
        return 1
    print(f"Completed {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
