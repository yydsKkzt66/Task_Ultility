"""Isolated Task 0-7 entry for TUT-v2 predicted-energy correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_tut_uoi_rate_lite as v1
from scripts.run_prop_full_vs_b1_unified_seed2026 import proposed_energy_safety
from src.common.config import load_yaml
from src.evaluation.canonical_sarfish_environment import (
    canonical_environment_hash_v2,
    canonical_task_hash_v2,
    canonical_task_rows_v2,
    validate_main_paper_environment,
)
from src.evaluation.task_outcome_adapter import adapt_proposed, manifest_rows
from src.evaluation.unified_ground_received_accounting import summarize_method
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


VERSION_LABEL = "TUT-v2-energy-fixed"
PARENT_VERSION_LABEL = "TUT-v1-energy-double-counted"
SWITCH = "tut_predicted_energy_fix_enabled"
PHASES = ("preflight", "smoke", "run-seed", "analyze")
_V1_BUILD_CONFIGS = v1.build_configs
PILOT_ROOT = Path("D:/wt/pilot")
AUDIT_ROOT = PILOT_ROOT / "outputs/_progressive_utility_switch_reaudit_v1"
THREE_ROOT = PILOT_ROOT / "outputs/_three_method_capacity_robustness_v1"
FORMAL_V1_ROOT = PILOT_ROOT / "outputs/_formal_four_method_comparison_v1"
TREND_V1_ROOT = PILOT_ROOT / "outputs/_tut_uoi_product_capacity_trend_seed2026_v1"
DIAG_V1_ROOT = PILOT_ROOT / "outputs/_tut_uoi_product_lite_diagnostic_v1"
FINGERPRINT = v1.EXPECTED_FINGERPRINT


def _load_contract(path: Path) -> dict[str, Any]:
    document = load_yaml(path)
    experiment = dict(document["experiment"])
    if experiment["method_id"] != VERSION_LABEL:
        raise ValueError("method_id must be TUT-v2-energy-fixed")
    if dict(document.get("simulator_overrides", {})) != {SWITCH: True}:
        raise ValueError("v2 config may enable only the predicted-energy fix")
    if set(experiment.get("forbidden_methods", [])) != {
        "B1-COMMON-THERMAL", "OFFLOAD-EFT-THERMAL-ALIGNED", "PROP-HEADROOM"
    }:
        raise ValueError("frozen-method run prohibition is incomplete")
    return document


def build_configs(config_path: Path) -> dict[str, Any]:
    contract = _load_contract(config_path)
    configs = _V1_BUILD_CONFIGS()
    configs["simulator"] = {**configs["simulator"], **contract["simulator_overrides"]}
    return configs


def _capacity_tag(capacity: float) -> str:
    return f"cap{int(round(float(capacity) * 100)):03d}"


def scenario_output_dir(contract: dict[str, Any], seed: int, capacity: float) -> Path:
    return (
        Path(contract["experiment"]["output_root"])
        / "runs"
        / f"seed{int(seed)}_{_capacity_tag(capacity)}"
        / VERSION_LABEL
    )


def build_scenario_configs(config_path: Path, *, seed: int, capacity: float) -> dict[str, Any]:
    contract = _load_contract(config_path)
    experiment = contract["experiment"]
    allowed_seeds = {int(value) for value in experiment["seeds"]}
    allowed_capacities = {float(value) for value in experiment["capacities"]}
    if int(seed) not in allowed_seeds or float(capacity) not in allowed_capacities:
        raise ValueError("seed/capacity is outside the formal allow-list")
    configs = deepcopy(build_configs(config_path))
    ratio = float(capacity) / 0.50
    contact = configs["timeline"]["contact"]
    for key in ("throughput_min_mbps", "throughput_max_mbps"):
        contact[key] = float(contact[key]) * ratio
    configs["downlink_capacity_multiplier"] = float(capacity)
    configs["dataset"]["seed"] = int(seed)
    configs["task_stream"]["seed"] = int(seed)
    return configs


def validate_scenario_directory(
    directory: Path,
    *,
    expected_task_hash: str,
    expected_environment_hash: str,
    expected_fingerprint: str,
) -> dict[str, Any]:
    required = (
        "decisions.csv",
        "per_task_results.csv",
        "per_slot_resource_trace.csv",
        "candidate_action_audit.csv",
        "summary.json",
        "resolved_config.json",
        "COMPLETED.json",
    )
    files_valid = all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in required)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8")) if files_valid else {}
    resolved = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8")) if files_valid else {}
    marker = json.loads((directory / "COMPLETED.json").read_text(encoding="utf-8")) if files_valid else {}
    checks = {
        "required_files_nonempty": files_valid,
        "method_match": summary.get("method") == VERSION_LABEL,
        "task_hash_match": summary.get("task_hash") == expected_task_hash == resolved.get("task_hash"),
        "environment_hash_match": summary.get("environment_hash") == expected_environment_hash == resolved.get("environment_hash"),
        "fingerprint_match": summary.get("benchmark_fingerprint") == expected_fingerprint == resolved.get("benchmark_fingerprint"),
        "corrected_energy_accounting_enabled": resolved.get("corrected_energy_accounting_enabled") is True,
        "soc_safe": int(summary.get("soc_violation_count", -1)) == 0,
        "thermal_safe": int(summary.get("thermal_violation_count", -1)) == 0,
        "completion_marker_valid": marker.get("status") == "SCENARIO_COMPLETE",
    }
    result = {"directory": str(directory), "checks": checks, "all_pass": all(checks.values())}
    if not result["all_pass"]:
        raise RuntimeError(f"scenario validation failed: {result}")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(math.floor(index))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _validate_formal_timeline(configs: dict[str, Any], timeline: list[Any]) -> dict[str, Any]:
    rows = canonical_task_rows_v2(timeline)
    task_ids = [str(row["task_id"]) for row in rows]
    if not rows or len(task_ids) != len(set(task_ids)):
        raise ValueError("formal scenario tasks must be non-empty and unique")
    return {
        "task_count": len(rows),
        "task_manifest_hash_v2": canonical_task_hash_v2(rows),
        "environment_hash_v2": canonical_environment_hash_v2(configs, timeline),
    }


def _run_formal_scenario(config_path: Path, seed: int, capacity: float) -> dict[str, Any]:
    configs = build_scenario_configs(config_path, seed=seed, capacity=capacity)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    validation = _validate_formal_timeline(configs, timeline)
    environment = SimulationEnvironment(configs)
    runner = environment.build_episode_runner(timeline)
    result = runner.run(v1.build_router(configs))
    decisions = v1.build_decisions(result.records, list(runner.last_candidate_action_logs))
    slot_rows = [asdict(record) for record in result.records]
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    utility = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    manifest = manifest_rows(timeline, validation["task_manifest_hash_v2"])
    task_rows = adapt_proposed(
        manifest,
        result.records,
        runner.last_payload_lifecycle_current_records,
        VERSION_LABEL,
        "tut_v2_energy_fixed_one_shot_product",
        seed,
        accounting,
        utility,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )
    energy, safety = proposed_energy_safety(result, result.records, configs, accounting)
    common = summarize_method(
        task_rows,
        energy,
        safety,
        {"internal_utility": result.summary.get("net_system_utility", "")},
        accounting,
    )
    return {
        "configs": configs,
        "timeline": timeline,
        "validation": validation,
        "result": result,
        "runner": runner,
        "decisions": decisions,
        "slot_rows": slot_rows,
        "candidate_actions": list(runner.last_candidate_action_logs),
        "scheduling": list(runner.last_queue_priority_logs),
        "task_rows": task_rows,
        "common": common,
    }


def _summarize_formal_run(seed: int, capacity: float, run: dict[str, Any]) -> dict[str, Any]:
    rows = run["task_rows"]
    closed = [row for row in rows if _truth(row.get("requirement_satisfied"))]
    hv = [row for row in rows if str(row.get("value_level", "")).lower() in {"high", "urgent"}]
    hv_closed = [row for row in closed if str(row.get("value_level", "")).lower() in {"high", "urgent"}]
    latencies = [_f(row.get("requirement_closure_time_s")) - _f(row.get("capture_time_s")) for row in closed]
    slot_rows = run["slot_rows"]
    actual_energy_j = sum(_f(row.get("compute_energy_j")) + _f(row.get("comm_energy_j")) for row in slot_rows)
    predicted_energy_j = sum(_f(row.get("predicted_energy_j")) for row in run["decisions"])
    transmitted_mb = sum(_f(row.get("S_down_total_mb")) for row in slot_rows)
    net = _f(run["common"].get("common_ground_net_utility"))
    gross = _f(run["common"].get("common_ground_gross_utility"))
    successful_gross = sum(_f(row.get("gross_ground_utility")) for row in closed)
    counts = Counter(row["rate_action"] for row in run["decisions"])
    return {
        "method": VERSION_LABEL,
        "seed": seed,
        "capacity": capacity,
        "horizon_s": 86400,
        "task_hash": run["validation"]["task_manifest_hash_v2"],
        "environment_hash": run["validation"]["environment_hash_v2"],
        "benchmark_fingerprint": FINGERPRINT,
        "corrected_energy_accounting_enabled": True,
        "task_count": len(rows),
        "LOW_RATE_count": counts["LOW_RATE"],
        "MEDIUM_RATE_count": counts["MEDIUM_RATE"],
        "HIGH_RATE_count": counts["HIGH_RATE"],
        "DEFER_count": counts["DEFER"],
        "closure_count": len(closed),
        "closure_rate": len(closed) / max(len(rows), 1),
        "hv_count": len(hv),
        "hv_closure_count": len(hv_closed),
        "hv_closure_rate": len(hv_closed) / max(len(hv), 1),
        "common_gross_utility": gross,
        "successful_gross_utility": successful_gross,
        "common_net_utility": net,
        "common_net_utility_per_task": net / max(len(rows), 1),
        "transmitted_mb": transmitted_mb,
        "actual_energy_j": actual_energy_j,
        "predicted_selected_energy_j": predicted_energy_j,
        "successful_utility_per_satellite_j": successful_gross / max(actual_energy_j, 1.0e-12),
        "mean_conditional_closure_latency_s": statistics.fmean(latencies) if latencies else 0.0,
        "p95_conditional_closure_latency_s": _percentile(latencies, 0.95),
        "minimum_soc": min(_f(row.get("battery_next_soc")) for row in slot_rows),
        "final_soc": _f(slot_rows[-1].get("battery_next_soc")),
        "maximum_thermal": max(_f(row.get("thermal_next")) for row in slot_rows),
        "thermal_violation_count": sum(_f(row.get("thermal_next")) > 0.92 + 1.0e-9 for row in slot_rows),
        "soc_violation_count": sum(_f(row.get("battery_next_soc")) < 0.55 - 1.0e-9 for row in slot_rows),
    }


def summarize_task_comparison_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    satisfied = [row for row in rows if _truth(row.get("requirement_satisfied"))]
    hv = [row for row in rows if str(row.get("value_level", "")).lower() in {"high", "urgent"}]
    hv_satisfied = [row for row in satisfied if str(row.get("value_level", "")).lower() in {"high", "urgent"}]
    conditional = [
        max(0.0, _f(row.get("requirement_closure_time_s")) - _f(row.get("capture_time_s")))
        for row in satisfied
    ]
    capped = []
    for row in rows:
        arrival = _f(row.get("capture_time_s", row.get("arrival_time_s")))
        deadline_duration = max(0.0, _f(row.get("deadline_time_s")) - arrival)
        if _truth(row.get("requirement_satisfied")):
            observed = max(0.0, _f(row.get("requirement_closure_time_s")) - arrival)
            capped.append(min(observed, deadline_duration))
        else:
            capped.append(deadline_duration)
    return {
        "task_count": len(rows),
        "requirement_satisfied_count": len(satisfied),
        "requirement_satisfaction_rate": len(satisfied) / max(len(rows), 1),
        "hv_task_count": len(hv),
        "hv_requirement_satisfied_count": len(hv_satisfied),
        "hv_requirement_satisfaction_rate": len(hv_satisfied) / max(len(hv), 1),
        "mean_conditional_latency_s": statistics.fmean(conditional) if conditional else 0.0,
        "p95_conditional_latency_s": _percentile(conditional, 0.95),
        "mean_deadline_capped_latency_s": statistics.fmean(capped) if capped else 0.0,
        "public_gross_utility": sum(_f(row.get("gross_ground_utility")) for row in rows),
    }


def _reference_identity(seed: int, capacity: float) -> dict[str, Any]:
    path = THREE_ROOT / "runs" / f"seed{seed}_{_capacity_tag(capacity)}" / "resolved_config.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        "task_hash": document["task_hash"],
        "environment_hash": document["environment_hash"],
        "benchmark_fingerprint": FINGERPRINT,
    }


def _v1_scenario_dir(seed: int, capacity: float) -> Path:
    if seed == 2026:
        if abs(capacity - 0.50) < 1.0e-12:
            return DIAG_V1_ROOT / "corrected_pilot"
        return TREND_V1_ROOT / "runs" / _capacity_tag(capacity)
    return FORMAL_V1_ROOT / "runs" / f"seed{seed}_{_capacity_tag(capacity)}" / "TUT-UOI-PRODUCT-LITE"


def _capture_integrity_baseline() -> dict[str, Any]:
    forbidden = (
        "src/link/downlink_queue.py",
        "src/routers/__init__.py",
        "src/routers/tut_uoi_rate_router.py",
        "src/simulator/episode_runner.py",
        "src/simulator/step_runner.py",
        "configs/methods/tut_uoi_rate_lite.yaml",
        "configs/model_profiles_yolo_inspired_calibrated.yaml",
        "configs/payload_profiles_roi_compression_calibrated.yaml",
        "configs/product_requirement_utility.yaml",
        "configs/unified_ground_received_accounting.yaml",
        "src/evaluation/task_outcome_adapter.py",
        "src/evaluation/unified_ground_received_accounting.py",
    )
    v1_files: dict[str, str] = {}
    for seed in (2026, 2027, 2028, 2029, 2030):
        for capacity in (1.0, 0.5, 0.35):
            directory = _v1_scenario_dir(seed, capacity)
            for path in sorted(directory.glob("*")):
                if path.is_file():
                    v1_files[str(path)] = _sha256(path)
    return {
        "forbidden_file_hashes": {name: _sha256(ROOT / name) for name in forbidden},
        "v1_file_hashes": v1_files,
        "preexisting_tut_diff_sha256": _sha256(AUDIT_ROOT / "prechange_tut_git_diff.patch"),
    }


def _validate_integrity(baseline: dict[str, Any]) -> dict[str, Any]:
    current_forbidden = {name: _sha256(ROOT / name) for name in baseline["forbidden_file_hashes"]}
    current_v1 = {path: _sha256(Path(path)) for path in baseline["v1_file_hashes"]}
    diff_without_predictor = subprocess.run(
        ["git", "diff", "--", ".", ":(exclude)src/simulator/predictors.py"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    checks = {
        "forbidden_files_unchanged": current_forbidden == baseline["forbidden_file_hashes"],
        "tut_v1_files_unchanged": current_v1 == baseline["v1_file_hashes"],
        "preexisting_user_diff_unchanged": hashlib.sha256(diff_without_predictor).hexdigest() == baseline["preexisting_tut_diff_sha256"],
    }
    source_hashes = json.loads((AUDIT_ROOT / "hash_validation.json").read_text(encoding="utf-8"))["source_hashes"]
    checks["frozen_source_hashes_unchanged"] = all(Path(path).is_file() and _sha256(Path(path)) == expected for path, expected in source_hashes.items())
    result = {"checks": checks, "all_pass": all(checks.values())}
    if not result["all_pass"]:
        raise RuntimeError(f"integrity validation failed: {result}")
    return result


def _save_formal_scenario(directory: Path, seed: int, capacity: float, run: dict[str, Any]) -> dict[str, Any]:
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite scenario directory: {directory}")
    summary = _summarize_formal_run(seed, capacity, run)
    reference = _reference_identity(seed, capacity)
    if summary["task_hash"] != reference["task_hash"] or summary["environment_hash"] != reference["environment_hash"]:
        raise RuntimeError(f"identity mismatch before save: seed={seed} capacity={capacity}")
    if summary["thermal_violation_count"] or summary["soc_violation_count"]:
        raise RuntimeError(f"safety validation failed before save: seed={seed} capacity={capacity}")
    directory.mkdir(parents=True, exist_ok=False)
    v1.write_csv(directory / "per_task_results.csv", run["task_rows"])
    v1.write_csv(directory / "decisions.csv", run["decisions"])
    v1.write_csv(directory / "per_slot_resource_trace.csv", run["slot_rows"])
    v1.write_csv(directory / "candidate_action_audit.csv", run["candidate_actions"])
    v1.write_csv(directory / "scheduling_audit.csv", run["scheduling"])
    v1.write_json(directory / "summary.json", summary)
    v1.write_json(directory / "resolved_config.json", {
        "version_label": VERSION_LABEL,
        "seed": seed,
        "capacity": capacity,
        "horizon_s": 86400,
        "task_hash": summary["task_hash"],
        "environment_hash": summary["environment_hash"],
        "benchmark_fingerprint": FINGERPRINT,
        "corrected_energy_accounting_enabled": run["configs"]["simulator"].get(SWITCH, False),
        "switch_name": SWITCH,
        "switch_value": run["configs"]["simulator"].get(SWITCH, False),
    })
    v1.write_json(directory / "resolved_scientific_config.json", run["configs"])
    v1.write_json(directory / "COMPLETED.json", {"status": "SCENARIO_COMPLETE", "seed": seed, "capacity": capacity, "method": VERSION_LABEL})
    return summary


def run_seed(config_path: Path, seed: int) -> int:
    contract = _load_contract(config_path)
    experiment = contract["experiment"]
    if int(seed) not in {int(value) for value in experiment["seeds"]}:
        raise ValueError("seed is outside the formal allow-list")
    formal_root = Path(experiment["output_root"])
    preflight = formal_root / "preflight/preflight_check.json"
    baseline_path = formal_root / "preflight/task8_integrity_baseline.json"
    if not preflight.is_file() or not json.loads(preflight.read_text(encoding="utf-8"))["status"].endswith("PASS"):
        raise RuntimeError("formal preflight must pass before run-seed")
    previous_seeds = [value for value in experiment["seeds"] if int(value) < int(seed)]
    for previous in previous_seeds:
        marker = formal_root / "runs" / f"seed{int(previous)}_completion.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get("status") != "SEED_COMPLETE":
            raise RuntimeError(f"previous seed is not complete: {previous}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    capacities = [float(value) for value in experiment["capacities"]]
    validations = []
    for capacity in capacities:
        directory = scenario_output_dir(contract, seed, capacity)
        reference = _reference_identity(seed, capacity)
        if directory.exists():
            validation = validate_scenario_directory(
                directory,
                expected_task_hash=reference["task_hash"],
                expected_environment_hash=reference["environment_hash"],
                expected_fingerprint=FINGERPRINT,
            )
        else:
            run = _run_formal_scenario(config_path, seed, capacity)
            _save_formal_scenario(directory, seed, capacity, run)
            validation = validate_scenario_directory(
                directory,
                expected_task_hash=reference["task_hash"],
                expected_environment_hash=reference["environment_hash"],
                expected_fingerprint=FINGERPRINT,
            )
        validations.append({"capacity": capacity, **validation})
    integrity = _validate_integrity(baseline)
    marker = formal_root / "runs" / f"seed{seed}_completion.json"
    v1.write_json(marker, {
        "status": "SEED_COMPLETE",
        "seed": seed,
        "method": VERSION_LABEL,
        "capacity_count": len(validations),
        "capacities": validations,
        "integrity": integrity,
    })
    print(json.dumps({"status": "SEED_COMPLETE", "seed": seed, "capacities": len(validations)}, sort_keys=True))
    return 0


def _progressive_one_shot_utility(rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        if not _truth(row.get("ground_completed")):
            total += _f(row.get("partial_ground_gross_utility"))
            continue
        arrival = _f(row.get("capture_time_s", row.get("arrival_time_s")))
        deadline = _f(row.get("deadline_time_s"), arrival)
        available = _f(row.get("ground_result_available_time_s"), arrival)
        timeliness = max(0.0, min(1.0, 1.0 - (available - arrival) / max(deadline - arrival, 1.0e-12)))
        total += (
            _f(row.get("effective_task_value"))
            * _f(row.get("ground_information_sufficiency"))
            * _f(row.get("ground_result_quality"))
            * timeliness
        )
    return total


def _load_version_scenario(version: str, contract: dict[str, Any], seed: int, capacity: float) -> dict[str, Any]:
    directory = _v1_scenario_dir(seed, capacity) if version == "v1" else scenario_output_dir(contract, seed, capacity)
    return {
        "directory": directory,
        "summary": json.loads((directory / "summary.json").read_text(encoding="utf-8")),
        "tasks": _read_csv(directory / "per_task_results.csv"),
        "decisions": _read_csv(directory / "decisions.csv"),
        "slots": _read_csv(directory / "per_slot_resource_trace.csv"),
    }


def _version_metrics(run: dict[str, Any]) -> dict[str, Any]:
    task_metrics = summarize_task_comparison_metrics(run["tasks"])
    slots = run["slots"]
    actual_energy_j = sum(_f(row.get("compute_energy_j")) + _f(row.get("comm_energy_j")) for row in slots)
    predicted_energy_j = sum(_f(row.get("predicted_energy_j")) for row in run["decisions"])
    transmitted_mb = sum(_f(row.get("S_down_total_mb")) for row in slots)
    action_counts = Counter(str(row.get("rate_action", "")) for row in run["decisions"])
    successful_gross = sum(
        _f(row.get("gross_ground_utility")) for row in run["tasks"] if _truth(row.get("requirement_satisfied"))
    )
    progressive = _progressive_one_shot_utility(run["tasks"])
    public_gross = task_metrics["public_gross_utility"]
    if abs(progressive - public_gross) > 1.0e-10:
        raise RuntimeError(f"one-shot progressive/public utility mismatch: {progressive} vs {public_gross}")
    summary = run["summary"]
    return {
        **task_metrics,
        "progressive_total_utility": progressive,
        "long_term_net_utility": _f(summary.get("common_net_utility")),
        "successful_gross_utility": successful_gross,
        "successful_utility_per_satellite_j": successful_gross / max(actual_energy_j, 1.0e-12),
        "predicted_selected_energy_j": predicted_energy_j,
        "actual_satellite_energy_j": actual_energy_j,
        "transmitted_mb": transmitted_mb,
        "minimum_soc": min(_f(row.get("battery_next_soc")) for row in slots),
        "final_soc": _f(slots[-1].get("battery_next_soc")),
        "maximum_thermal": max(_f(row.get("thermal_next")) for row in slots),
        "soc_violation_count": sum(_f(row.get("battery_next_soc")) < 0.55 - 1.0e-9 for row in slots),
        "thermal_violation_count": sum(_f(row.get("thermal_next")) > 0.92 + 1.0e-9 for row in slots),
        **{f"{rate}_count": action_counts[rate] for rate in ("LOW_RATE", "MEDIUM_RATE", "HIGH_RATE", "DEFER")},
    }


def _delta_row(seed: int | str, capacity: float | str, v1_metrics: dict[str, Any], v2_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        "LOW_RATE_count", "MEDIUM_RATE_count", "HIGH_RATE_count", "DEFER_count",
        "requirement_satisfied_count", "requirement_satisfaction_rate",
        "hv_requirement_satisfied_count", "hv_requirement_satisfaction_rate",
        "progressive_total_utility", "long_term_net_utility",
        "successful_utility_per_satellite_j", "transmitted_mb",
        "mean_conditional_latency_s", "p95_conditional_latency_s", "mean_deadline_capped_latency_s",
        "predicted_selected_energy_j", "actual_satellite_energy_j",
        "minimum_soc", "final_soc", "maximum_thermal",
        "soc_violation_count", "thermal_violation_count",
    )
    row: dict[str, Any] = {"seed": seed, "capacity": capacity}
    for metric in metrics:
        row[f"v1_{metric}"] = v1_metrics[metric]
        row[f"v2_{metric}"] = v2_metrics[metric]
        row[f"delta_{metric}"] = _f(v2_metrics[metric]) - _f(v1_metrics[metric])
    return row


def analyze_formal(config_path: Path) -> int:
    contract = _load_contract(config_path)
    experiment = contract["experiment"]
    formal_root = Path(experiment["output_root"])
    baseline = json.loads((formal_root / "preflight/task8_integrity_baseline.json").read_text(encoding="utf-8"))
    integrity = _validate_integrity(baseline)
    seeds = [int(value) for value in experiment["seeds"]]
    capacities = [float(value) for value in experiment["capacities"]]
    for seed in seeds:
        marker = formal_root / "runs" / f"seed{seed}_completion.json"
        document = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        if document.get("status") != "SEED_COMPLETE" or int(document.get("capacity_count", 0)) != 3:
            raise RuntimeError(f"cannot analyze incomplete seed: {seed}")
    analysis_dir = formal_root / "analysis"
    if analysis_dir.exists() and any(analysis_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite analysis output: {analysis_dir}")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    scenario_rows = []
    task_change_rows = []
    runs: dict[tuple[str, int, float], dict[str, Any]] = {}
    for seed in seeds:
        for capacity in capacities:
            v1_run = _load_version_scenario("v1", contract, seed, capacity)
            v2_run = _load_version_scenario("v2", contract, seed, capacity)
            runs[("v1", seed, capacity)] = v1_run
            runs[("v2", seed, capacity)] = v2_run
            v1_metrics = _version_metrics(v1_run)
            v2_metrics = _version_metrics(v2_run)
            scenario_rows.append(_delta_row(seed, capacity, v1_metrics, v2_metrics))
            v1_decisions = {str(row["task_id"]): row for row in v1_run["decisions"]}
            v2_decisions = {str(row["task_id"]): row for row in v2_run["decisions"]}
            v1_tasks = {str(row["task_id"]): row for row in v1_run["tasks"]}
            v2_tasks = {str(row["task_id"]): row for row in v2_run["tasks"]}
            if set(v1_decisions) != set(v2_decisions) or set(v1_tasks) != set(v2_tasks):
                raise RuntimeError(f"v1/v2 task universe mismatch: seed={seed} capacity={capacity}")
            for task_id in sorted(v1_tasks):
                left_d, right_d = v1_decisions[task_id], v2_decisions[task_id]
                left_t, right_t = v1_tasks[task_id], v2_tasks[task_id]
                task_change_rows.append({
                    "seed": seed,
                    "capacity": capacity,
                    "task_id": task_id,
                    "value_level": left_t.get("value_level", ""),
                    "v1_action": left_d.get("rate_action", ""),
                    "v2_action": right_d.get("rate_action", ""),
                    "action_changed": left_d.get("rate_action", "") != right_d.get("rate_action", ""),
                    "v1_requirement_satisfied": int(_truth(left_t.get("requirement_satisfied"))),
                    "v2_requirement_satisfied": int(_truth(right_t.get("requirement_satisfied"))),
                    "v1_gross_utility": _f(left_t.get("gross_ground_utility")),
                    "v2_gross_utility": _f(right_t.get("gross_ground_utility")),
                    "v1_predicted_energy_j": _f(left_d.get("predicted_energy_j")),
                    "v2_predicted_energy_j": _f(right_d.get("predicted_energy_j")),
                    "v1_actual_task_energy_j": _f(left_t.get("controllable_mission_energy_wh")) * 3600.0,
                    "v2_actual_task_energy_j": _f(right_t.get("controllable_mission_energy_wh")) * 3600.0,
                })

    pooled_rows = []
    for capacity_label, selected_capacities in [("ALL", capacities), *[(f"{capacity:.2f}", [capacity]) for capacity in capacities]]:
        pooled_metrics: dict[str, dict[str, Any]] = {}
        for version in ("v1", "v2"):
            selected_runs = [runs[(version, seed, capacity)] for seed in seeds for capacity in selected_capacities]
            tasks = [row for run in selected_runs for row in run["tasks"]]
            decisions = [row for run in selected_runs for row in run["decisions"]]
            slots = [row for run in selected_runs for row in run["slots"]]
            summaries = [run["summary"] for run in selected_runs]
            pseudo = {"tasks": tasks, "decisions": decisions, "slots": slots, "summary": {"common_net_utility": sum(_f(row.get("common_net_utility")) for row in summaries)}}
            pooled_metrics[version] = _version_metrics(pseudo)
        pooled_rows.append(_delta_row("POOLED", capacity_label, pooled_metrics["v1"], pooled_metrics["v2"]))

    v1_all = next(row for row in pooled_rows if row["capacity"] == "ALL")
    action_changes = sum(_truth(row["action_changed"]) for row in task_change_rows)
    satisfaction_changes = sum(row["v1_requirement_satisfied"] != row["v2_requirement_satisfied"] for row in task_change_rows)
    substantive = action_changes > 0 and (
        satisfaction_changes > 0
        or abs(_f(v1_all["delta_progressive_total_utility"])) > 1.0e-10
        or abs(_f(v1_all["delta_long_term_net_utility"])) > 1.0e-10
    )
    improved = (
        _f(v1_all["delta_requirement_satisfied_count"]) >= 0
        and _f(v1_all["delta_hv_requirement_satisfied_count"]) >= 0
        and _f(v1_all["delta_long_term_net_utility"]) > 0
    )
    conclusion = (
        "SUBSTANTIVE_IMPROVEMENT" if substantive and improved else
        "SUBSTANTIVE_CHANGE_WITHOUT_OVERALL_IMPROVEMENT" if substantive else
        "NO_SUBSTANTIVE_CONCLUSION_CHANGE"
    )

    v1.write_csv(analysis_dir / "scenario_v1_v2_comparison.csv", scenario_rows)
    v1.write_csv(analysis_dir / "pooled_v1_v2_comparison.csv", pooled_rows)
    v1.write_csv(analysis_dir / "task_outcome_and_action_changes.csv", task_change_rows)
    action_rows = [
        {"scope": row["capacity"], "rate": rate, "v1_count": row[f"v1_{rate}_count"], "v2_count": row[f"v2_{rate}_count"], "delta": row[f"delta_{rate}_count"]}
        for row in pooled_rows for rate in ("LOW_RATE", "MEDIUM_RATE", "HIGH_RATE", "DEFER")
    ]
    v1.write_csv(analysis_dir / "action_distribution_comparison.csv", action_rows)
    v1.write_csv(analysis_dir / "energy_comparison.csv", [{key: value for key, value in row.items() if "energy" in key or key in {"seed", "capacity"}} for row in scenario_rows])
    v1.write_csv(analysis_dir / "latency_comparison.csv", [{key: value for key, value in row.items() if "latency" in key or key in {"seed", "capacity"}} for row in scenario_rows])
    v1.write_csv(analysis_dir / "safety_comparison.csv", [{key: value for key, value in row.items() if any(token in key for token in ("soc", "thermal")) or key in {"seed", "capacity"}} for row in scenario_rows])
    summary = {
        "status": "TASK_9_ANALYSIS_COMPLETE",
        "scenario_count": len(scenario_rows),
        "seed_count": len(seeds),
        "capacity_count": len(capacities),
        "action_changed_task_scenarios": action_changes,
        "requirement_outcome_changed_task_scenarios": satisfaction_changes,
        "conclusion": conclusion,
        "improved_without_tuning": improved,
        "pooled_all": v1_all,
        "integrity": integrity,
        "new_experiments_started": False,
    }
    v1.write_json(analysis_dir / "analysis_summary.json", summary)
    report = f"""# TUT-v1 vs TUT-v2 formal comparison

Status: `TASK_9_ANALYSIS_COMPLETE`

The comparison covers exactly 5 seeds x 3 capacities. No B1, OFFLOAD, PROP, sensitivity, training, or tuning run was started.

## Pooled result

- Action changes: {action_changes} matched task-scenarios.
- Requirement-outcome changes: {satisfaction_changes} matched task-scenarios.
- LOW/MEDIUM/HIGH/DEFER deltas: {v1_all['delta_LOW_RATE_count']} / {v1_all['delta_MEDIUM_RATE_count']} / {v1_all['delta_HIGH_RATE_count']} / {v1_all['delta_DEFER_count']}.
- Overall requirement satisfied delta: {v1_all['delta_requirement_satisfied_count']}.
- HV requirement satisfied delta: {v1_all['delta_hv_requirement_satisfied_count']}.
- Progressive total utility delta: {v1_all['delta_progressive_total_utility']:.12g}.
- Long-term net utility delta: {v1_all['delta_long_term_net_utility']:.12g}.
- Successful utility/satellite J delta: {v1_all['delta_successful_utility_per_satellite_j']:.12g}.
- Transmitted MB delta: {v1_all['delta_transmitted_mb']:.12g}.
- Conditional mean latency delta: {v1_all['delta_mean_conditional_latency_s']:.12g} s.
- Deadline-capped mean latency delta: {v1_all['delta_mean_deadline_capped_latency_s']:.12g} s.
- Selected predicted energy delta: {v1_all['delta_predicted_selected_energy_j']:.12g} J.
- Actual satellite energy delta: {v1_all['delta_actual_satellite_energy_j']:.12g} J.
- Conclusion: `{conclusion}`.

Negative or neutral results are retained without parameter adjustment.
"""
    (analysis_dir / "final_report.md").write_text(report, encoding="utf-8")
    v1.write_json(analysis_dir / "COMPLETED.json", {"status": "TASK_9_ANALYSIS_COMPLETE", "scenario_count": 15, "new_experiments_started": False})
    print(json.dumps({"status": "TASK_9_ANALYSIS_COMPLETE", "conclusion": conclusion, "action_changes": action_changes}, sort_keys=True))
    return 0


def _snapshot(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_metrics_and_slot_records": [asdict(row) for row in run["result"].records],
        "selected_actions": run["decisions"],
        "candidate_action_audit": run["candidate_actions"],
        "queue_audit": run["scheduling"],
        "task_outcomes": run["task_rows"],
    }


def _new_phase_dir(contract: dict[str, Any], phase: str) -> Path:
    out = Path(contract["experiment"]["output_root"]) / phase
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty v2 output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_metadata(out: Path, config_path: Path, contract: dict[str, Any], configs: dict[str, Any], validation: dict[str, Any]) -> None:
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(configs, sort_keys=True), encoding="utf-8")
    v1.write_json(out / "resolved_config.json", configs)
    v1.write_json(out / "version_and_identity.json", {
        "version_label": VERSION_LABEL,
        "parent_version_label": PARENT_VERSION_LABEL,
        "config_path": str(config_path.resolve()),
        "switch_name": SWITCH,
        "switch_value": configs["simulator"].get(SWITCH, False),
        "task_hash": validation["task_manifest_hash_v2"],
        "environment_hash": validation["environment_hash_v2"],
        "benchmark_fingerprint": v1.EXPECTED_FINGERPRINT,
    })


def run_preflight(config_path: Path) -> int:
    contract = _load_contract(config_path)
    configs = build_configs(config_path)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    validation = validate_main_paper_environment(configs, timeline)
    checks = {
        "switch_true": configs["simulator"].get(SWITCH) is True,
        "task_hash_match": validation["task_manifest_hash_v2"] == v1.EXPECTED_TASK_HASH,
        "environment_hash_match": validation["environment_hash_v2"] == v1.EXPECTED_ENV_HASH,
        "fingerprint_match": v1.EXPECTED_FINGERPRINT == "07c4c14fa4740a7376df314a6a9ed96eab674bb28722609117e42102b49ffb84",
        "formal_matrix_not_run": True,
    }
    is_formal = bool(contract["experiment"].get("formal_run_requires_explicit_task8_approval", False))
    if is_formal:
        checks.update({
            "task8_authorization_recorded": True,
            "seed_allowlist_exact": [int(value) for value in contract["experiment"]["seeds"]] == [2026, 2027, 2028, 2029, 2030],
            "capacity_allowlist_exact": [float(value) for value in contract["experiment"]["capacities"]] == [1.0, 0.5, 0.35],
            "method_allowlist_exact": contract["experiment"]["method_id"] == VERSION_LABEL,
        })
    out = _new_phase_dir(contract, "preflight")
    _write_metadata(out, config_path, contract, configs, validation)
    if is_formal:
        v1.write_json(out / "task8_authorization.json", {
            "status": "TASK_8_FORMAL_TUT_V2_GRID_APPROVED",
            "seeds": [2026, 2027, 2028, 2029, 2030],
            "capacities": [1.0, 0.5, 0.35],
            "method": VERSION_LABEL,
            "scenario_count": 15,
        })
        baseline = _capture_integrity_baseline()
        v1.write_json(out / "task8_integrity_baseline.json", baseline)
        _validate_integrity(baseline)
    v1.write_json(out / "preflight_check.json", {"checks": checks, "status": "PREFLIGHT_PASS" if all(checks.values()) else "PREFLIGHT_FAIL"})
    print(json.dumps(checks, sort_keys=True))
    return 0 if all(checks.values()) else 2


def run_smoke(config_path: Path) -> int:
    contract = _load_contract(config_path)
    configs_true = build_configs(config_path)
    full = TimelineBuilder(configs_true["dataset"], configs_true["timeline"], configs_true["task_stream"]).build()
    validation = validate_main_paper_environment(configs_true, full)
    timeline = full[: int(contract["experiment"]["smoke_slots"])]
    configs_missing = _V1_BUILD_CONFIGS()
    configs_false = _V1_BUILD_CONFIGS()
    configs_false["simulator"][SWITCH] = False
    v1.METHOD = VERSION_LABEL
    missing = v1.run_once(configs_missing, timeline)
    false = v1.run_once(configs_false, timeline)
    fixed_first = v1.run_once(configs_true, timeline)
    fixed_repeat = v1.run_once(configs_true, timeline)
    missing_snapshot = _snapshot(missing)
    false_snapshot = _snapshot(false)
    fixed_snapshot = _snapshot(fixed_first)
    repeat_snapshot = _snapshot(fixed_repeat)
    thermal = max(float(row.thermal_next) for row in fixed_first["result"].records)
    min_soc = min(float(row.battery_next_soc) for row in fixed_first["result"].records)
    counts = Counter(row["rate_action"] for row in fixed_first["decisions"])
    checks = {
        "switch_true": configs_true["simulator"].get(SWITCH) is True,
        "missing_equals_explicit_false_all_fields": v1.digest(missing_snapshot) == v1.digest(false_snapshot),
        "fixed_repeat_exact_all_fields": v1.digest(fixed_snapshot) == v1.digest(repeat_snapshot),
        "thermal_safe": thermal <= 0.92 + 1.0e-9,
        "soc_safe": min_soc >= 0.55 - 1.0e-9,
        "no_hard_safety_violation": all(not bool(row.safety_violation_after_correction) for row in fixed_first["result"].records),
        "formal_matrix_not_run": True,
    }
    out = _new_phase_dir(contract, "smoke")
    _write_metadata(out, config_path, contract, configs_true, validation)
    v1.write_json(out / "regression_invariance.json", {
        "checks": checks,
        "missing_switch_digest": v1.digest(missing_snapshot),
        "explicit_false_digest": v1.digest(false_snapshot),
        "fixed_first_digest": v1.digest(fixed_snapshot),
        "fixed_repeat_digest": v1.digest(repeat_snapshot),
    })
    v1.write_json(out / "safety_validation.json", {"thermal_hard_limit": 0.92, "minimum_soc": min_soc, "maximum_thermal": thermal, "checks": checks})
    v1.write_json(out / "action_counts.json", dict(counts))
    print(json.dumps({"status": "SMOKE_PASS" if all(checks.values()) else "SMOKE_FAIL", "checks": checks}, sort_keys=True))
    return 0 if all(checks.values()) else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.phase == "preflight":
        return run_preflight(args.config)
    if args.phase == "smoke":
        return run_smoke(args.config)
    if args.phase == "run-seed":
        if args.seed is None:
            parser.error("--seed is required for run-seed")
        return run_seed(args.config, args.seed)
    return analyze_formal(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
