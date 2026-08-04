"""Run the B1-Enhanced contact-aware compressed downlink baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.b1_enhanced import (  # noqa: E402
    B1EnhancedSimulator,
    build_fairness_manifest,
    load_b1_config,
    write_b1_outputs,
)
from src.common.config import load_project_configs, load_yaml  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "_baseline_b1_enhanced_contact_compressed_v1"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_configs(config_path: Path | None, seed: int | None) -> dict[str, dict[str, Any]]:
    configs = load_project_configs(PROJECT_ROOT)
    if config_path is not None:
        override = load_yaml(config_path)
        if "baseline_b1_enhanced" in override:
            configs = _deep_update(configs, override)
        else:
            configs["baseline_b1_enhanced"] = _deep_update(configs.get("baseline_b1_enhanced", {}), override)
    configs.setdefault("baseline_b1_enhanced", {})
    if seed is not None:
        configs.setdefault("dataset", {})["seed"] = int(seed)
        configs.setdefault("task_stream", {})["seed"] = int(seed)
    configs["simulator"]["slot_seconds"] = float(configs["timeline"].get("slot_seconds", configs["simulator"]["slot_seconds"]))
    if "battery_capacity_wh" in configs["simulator"]:
        configs["simulator"]["battery_capacity_j"] = float(configs["simulator"]["battery_capacity_wh"]) * 3600.0
    return configs


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unavailable"


def _num_orbits(configs: dict[str, dict[str, Any]], timeline_length: int) -> int:
    slot_seconds = float(configs["timeline"].get("slot_seconds", configs["simulator"].get("slot_seconds", 60.0)))
    orbit_period_s = float(configs["timeline"].get("orbit_period_s", float(configs["timeline"].get("orbit_period_min", 90.0)) * 60.0))
    return int(math.ceil((timeline_length * slot_seconds) / max(orbit_period_s, 1.0e-9)))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def validate_outputs(output_dir: Path) -> dict[str, Any]:
    required = [
        "summary_metrics.csv",
        "summary_metrics.json",
        "per_task_delivery_trace.csv",
        "queue_trace.csv",
        "soc_trace.csv",
        "contact_window_schedule.csv",
        "compression_trace.csv",
        "lifecycle_counts.csv",
        "fairness_manifest.json",
        "config_snapshot.yaml",
        "b1_enhanced_validation_report.md",
        "paper_baseline_mapping.md",
        "implementation_notes.md",
        "compression_parameter_audit.csv",
        "ground_model_profile_audit.csv",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    task_rows = _read_csv(output_dir / "per_task_delivery_trace.csv") if not missing else []
    numeric_errors: list[str] = []
    seen_completed: set[str] = set()
    for row in task_rows:
        task_id = row.get("task_id", "")
        for key in [
            "onboard_compression_latency_s",
            "compression_latency_s",
            "queue_wait_latency_s",
            "contact_wait_latency_s",
            "transmission_latency_s",
            "ground_decode_latency_s",
            "ground_inference_latency_s",
            "capture_to_ground_latency_s",
            "compression_energy_wh",
        ]:
            value = float(row.get(key) or 0.0)
            if math.isnan(value) or math.isinf(value) or value < -1.0e-9:
                numeric_errors.append(f"{task_id}:{key}={value}")
        if int(float(row.get("ground_result_available", 0) or 0)):
            if task_id in seen_completed:
                numeric_errors.append(f"{task_id}:duplicate_ground_completion")
            seen_completed.add(task_id)
            completion = float(row.get("ground_result_available_time_s") or -1.0)
            capture = float(row.get("capture_time_s") or 0.0)
            minimum = capture + float(row.get("compression_latency_s") or 0.0)
            if completion + 1.0e-9 < minimum:
                numeric_errors.append(f"{task_id}:impossible_ground_completion_time")
            component_sum = float(row.get("latency_component_sum_s") or 0.0)
            latency = float(row.get("capture_to_ground_latency_s") or 0.0)
            if abs(component_sum - latency) > 1.0e-6:
                numeric_errors.append(f"{task_id}:latency_component_sum_mismatch")
    return {
        "missing_files": missing,
        "numeric_errors": numeric_errors,
        "passed": not missing and not numeric_errors,
    }


def run(
    output_dir: Path,
    config_path: Path | None = None,
    seed: int | None = None,
    task_manifest: Path | None = None,
    contact_trace: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists; use --overwrite explicitly: {output_dir}")
    configs = _load_configs(config_path, seed)
    b1_config = load_b1_config(configs.get("baseline_b1_enhanced", {}))
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs.get("task_stream", {})).build()
    simulator = B1EnhancedSimulator(b1_config, configs["simulator"], configs["model_pool"])
    result = simulator.run(timeline)
    fairness_manifest = build_fairness_manifest(
        timeline=timeline,
        config=b1_config,
        seed=int(configs.get("dataset", {}).get("seed", seed or 0)),
        git_commit=_git_commit(),
        num_orbits=_num_orbits(configs, len(timeline)),
        task_manifest_path=task_manifest,
        contact_trace_path=contact_trace,
    )
    write_b1_outputs(output_dir, result, b1_config, configs, fairness_manifest)
    validation = validate_outputs(output_dir)
    (output_dir / "output_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    if not validation["passed"]:
        raise RuntimeError(f"B1 output validation failed: {validation}")
    return {"output_dir": output_dir, "summary": result["summary"], "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B1-Enhanced contact-aware compressed downlink baseline.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-manifest", type=Path, default=None)
    parser.add_argument("--contact-trace", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    result = run(
        output_dir=args.output_dir,
        config_path=args.config,
        seed=args.seed,
        task_manifest=args.task_manifest,
        contact_trace=args.contact_trace,
        overwrite=args.overwrite,
    )
    print(f"output_dir={result['output_dir']}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
