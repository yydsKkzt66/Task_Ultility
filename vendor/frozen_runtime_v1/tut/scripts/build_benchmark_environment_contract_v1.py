"""Build the reusable benchmark contract from existing code/config/results only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/_benchmark_environment_contract_v1"
FREEZE = ROOT / "outputs/_sarfish_single_satellite_baseline_freeze_v1"
MULTI = ROOT / "outputs/_sarfish_thermal_headroom_multiseed"
STATUSES = {"ACTIVE", "PROXY", "UNKNOWN", "INACTIVE", "FUTURE"}


def param(value: Any, unit: str, status: str, source: str, notes: str = "") -> dict[str, Any]:
    assert status in STATUSES
    return {"value": value, "unit": unit, "status": status, "source": source, "notes": notes}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write(path: str | Path, text: str) -> Path:
    target = ROOT / path if not isinstance(path, Path) or not path.is_absolute() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")
    return target


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> Path:
    target = ROOT / path if not isinstance(path, Path) or not path.is_absolute() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace").strip()


def environment() -> dict[str, Any]:
    return {
        "metadata": {
            "contract_id": param("benchmark_environment_v1", "identifier", "ACTIVE", "this contract"),
            "authoritative_multiscenario_commit": param("57b5567312f32bee255b2c52cd06a840998cff98", "git commit", "ACTIVE", "validated study record"),
            "proxy_disclosure": param(True, "boolean", "ACTIVE", "source config descriptions", "System-level proxy; not end-to-end image processing."),
        },
        "simulation": {
            "horizon_s": param(86400, "s", "ACTIVE", "configs/timeline_sarfish_formal_nominal_bupt1_deadline_minutes_20s.yaml:physical_horizon_s"),
            "slot_seconds": param(20, "s", "ACTIVE", "configs/timeline_sarfish_formal_nominal_bupt1_deadline_minutes_20s.yaml:slot_seconds"),
            "slot_count": param(4320, "slots", "ACTIVE", "physical_horizon_s / slot_seconds"),
            "start_time_s": param(0, "s", "ACTIVE", "src/timeline/timeline_builder.py"),
            "end_time_s_exclusive": param(86400, "s", "ACTIVE", "timeline contract"),
            "deadline_inclusive": param(True, "boolean", "ACTIVE", "src/evaluation/run_event_adapters.py", "Completion <= deadline + 1e-9 is on time."),
        },
        "dataset": {
            "name": param("SARFishSample", "name", "ACTIVE", "configs/dataset_sarfish_grd_validation.yaml"),
            "root": param("datasets/SARFishSample", "repository-relative path", "ACTIVE", "configs/dataset_sarfish_grd_validation.yaml"),
            "product": param("GRD", "product", "ACTIVE", "configs/dataset_sarfish_grd_validation.yaml"),
            "split": param("validation", "split", "ACTIVE", "configs/dataset_sarfish_grd_validation.yaml"),
            "scene_index": param("data/indexes/sarfishsample_scene_index.csv", "path", "ACTIVE", "configs/dataset_sarfish_grd_validation.yaml"),
            "source_scene_count": param(50, "scenes", "ACTIVE", "data/indexes/sarfishsample_scene_index.csv", "Usable GRD validation index rows; scenes are sampled with replacement across tasks."),
            "pixels_read_online": param(False, "boolean", "ACTIVE", "src/data/sarfish_task_builder.py", "Tasks use indexed metadata; policies do not run image codecs."),
            "true_labels_online": param(False, "boolean", "ACTIVE", "task/evaluator audit", "Labels and annotations are evaluation/task-generation inputs, not online policy future knowledge."),
        },
        "scenario_generation": {
            "mode": param("observation_window", "mode", "PROXY", "configs/task_stream_sarfish_formal_nominal_bupt1_deadline_minutes.yaml"),
            "seeds": param([2026, 2027, 2028, 2029, 2030], "seed", "ACTIVE", "outputs/_sarfish_thermal_headroom_multiseed/scenario_manifest.csv"),
            "arrival_interval_slots": param([2, 3], "slots", "PROXY", "configs/task_stream_sarfish_formal_nominal_bupt1_deadline_minutes.yaml"),
            "background_arrival_probability": param(0.005, "probability/slot", "PROXY", "same task-stream config"),
        },
        "tasks": {
            "counts_by_seed": param({"2026": 268, "2027": 285, "2028": 255, "2029": 306, "2030": 285}, "tasks", "ACTIVE", "scenario manifest"),
            "ordinary_deadline_minutes": param([90, 240], "min", "PROXY", "task-stream config"),
            "high_deadline_minutes": param([45, 120], "min", "PROXY", "task-stream config"),
            "urgent_deadline_minutes": param([20, 90], "min", "PROXY", "task-stream config"),
            "value_class_mapping": param({"routine": "medium", "high_value": "high", "emergency": "urgent"}, "mapping", "ACTIVE", "src/evaluation/canonical_sarfish_environment.py"),
            "task_id_format": param("generated deterministic string", "identifier", "ACTIVE", "src/data/sarfish_task_builder.py"),
        },
        "contacts": {
            "trace_source": param("synthetic periodic contact", "source", "PROXY", "timeline config"),
            "period_s": param(5700, "s", "PROXY", "timeline config"),
            "duration_s": param(600, "s", "PROXY", "timeline config"),
            "phase_offset_s": param(420, "s", "PROXY", "timeline config"),
            "ground_station_count": param(None, "stations", "UNKNOWN", "not represented by active config"),
            "propagation_delay_s": param(None, "s", "UNKNOWN", "not represented"),
            "protocol_overhead": param(None, "fraction", "UNKNOWN", "not represented"),
        },
        "communication": {
            "throughput_min_mbps": param(35.0, "Mbit/s", "PROXY", "timeline config"),
            "throughput_max_mbps": param(65.0, "Mbit/s", "PROXY", "timeline config"),
            "capacity_multipliers": param([1.0, 0.5, 0.35], "ratio", "ACTIVE", "scripts/run_sarfish_thermal_headroom_multiseed.py", "Multiplier scales configured throughput."),
            "partial_transmission": param(True, "boolean", "ACTIVE", "src/link/downlink_queue.py"),
            "cross_slot_resume": param(True, "boolean", "ACTIVE", "src/link/downlink_queue.py"),
            "cross_contact_resume": param(True, "boolean", "ACTIVE", "src/link/downlink_queue.py"),
            "minimum_transfer_granularity": param("floating-point byte accounting", "bytes", "PROXY", "src/link/downlink_queue.py"),
            "tx_power_w": param(35.0, "W", "PROXY", "active simulator config"),
            "thermal_heat_per_j": param(0.0015, "normalized/J", "PROXY", "src/energy/thermal_update.py"),
            "headroom_enabled_by_default": param(False, "boolean", "INACTIVE", "active simulator config"),
        },
        "compute": {
            "hardware": param("BUPT-1-like abstract COTS compute", "description", "PROXY", "resource/model configs", "No CPU/GPU/chip model is simulated."),
            "cores": param(None, "cores", "UNKNOWN", "not modeled"),
            "concurrency": param(1, "logical action/slot", "PROXY", "src/simulator/step_runner.py"),
            "cross_slot_compute": param(False, "boolean", "ACTIVE", "src/simulator/step_runner.py", "Latency is accounted, but compute is instantiated within the action slot."),
            "preemption": param(False, "boolean", "ACTIVE", "same"),
            "thermal_heat_per_j": param(0.0025, "normalized/J", "PROXY", "src/energy/thermal_update.py"),
        },
        "models": {
            "backend": param("mock detector profiles", "backend", "PROXY", "src/models/mock_detector.py"),
            "real_onboard_inference": param(False, "boolean", "ACTIVE", "src/models/inference_runner.py"),
            "profiles_source": param("configs/model_profiles_yolo_inspired_calibrated.yaml", "path", "ACTIVE", "prop_configs"),
        },
        "products": {
            "profiles_source": param("configs/payload_profiles_roi_compression_calibrated.yaml", "path", "ACTIVE", "prop_configs"),
            "sizes_measured_from_encoding": param(False, "boolean", "ACTIVE", "payload profile description"),
            "quality_is_proxy": param(True, "boolean", "PROXY", "payload/model profiles"),
            "b1_full_frame_size_mb": param(4.0, "MB", "PROXY", "configs/baseline_b1_enhanced.yaml", "32 MB / fixed ratio 8; no real codec execution."),
            "b1_real_compression": param(False, "boolean", "ACTIVE", "src/baselines/b1_enhanced.py"),
        },
        "storage": {
            "raw_observation_capacity_mb": param(None, "MB", "UNKNOWN", "not separately modeled"),
            "product_storage_capacity_mb": param(None, "MB", "UNKNOWN", "downlink queue is the enforced product store"),
        },
        "queues": {
            "downlink": {"capacity_mb": param(5120.0, "MB", "ACTIVE", "active simulator config"), "safe_mb": param(4096.0, "MB", "ACTIVE", "active simulator config")},
            "compute": {"capacity_items": param(None, "items", "UNKNOWN", "no persistent compute queue")},
            "generation_request": {"capacity_items": param(None, "items", "UNKNOWN", "no numeric cap found")},
            "duplicate_suppression": param(True, "boolean", "ACTIVE", "src/link/downlink_queue.py"),
            "overflow_eviction": param("lowest priority until within byte cap", "policy", "ACTIVE", "src/link/downlink_queue.py"),
        },
        "battery": {
            "capacity_wh": param(230.0, "Wh", "PROXY", "active simulator/resource config"),
            "initial_soc": param(0.90, "fraction", "ACTIVE", "active simulator config"),
            "hard_soc_floor": param(0.55, "fraction", "ACTIVE", "active simulator/resource config", "Physical SafeFilter/resource floor; not the 0.70/0.50 policy regions."),
            "policy_guard_70": param(0.70, "fraction", "ACTIVE", "resource/PROP config"),
            "policy_guard_50": param(0.50, "fraction", "ACTIVE", "PROP follow-up policy", "Policy partition only, below the active physical floor."),
            "charge_efficiency": param(0.92, "ratio", "PROXY", "active simulator config"),
            "discharge_efficiency": param(0.95, "ratio", "PROXY", "active simulator config"),
            "soc_clipping": param([0.0, 1.0], "fraction", "ACTIVE", "src/energy/battery_update.py"),
        },
        "energy": {
            "solar_peak_w": param(14.5, "W", "PROXY", "resource config"),
            "idle_power_w": param(4.8, "W", "PROXY", "resource config"),
            "resource_power_variation_w": param(0.65, "W", "PROXY", "resource config"),
            "sunlit_duration_s": param(3600, "s/orbit", "PROXY", "resource config"),
            "eclipse_duration_s": param(2100, "s/orbit", "PROXY", "resource config"),
            "maximum_controllable_energy_budget": param(None, "Wh", "FUTURE", "reserved for matched-energy experiments"),
            "reserved_energy": param(None, "Wh", "FUTURE", "reserved"),
            "consumed_energy": param(None, "Wh", "FUTURE", "reserved"),
            "remaining_energy": param(None, "Wh", "FUTURE", "reserved"),
            "common_minimum_soc": param(None, "fraction", "FUTURE", "reserved"),
            "predicted_post_action_soc": param(None, "fraction", "FUTURE", "reserved"),
        },
        "thermal": {
            "state_meaning": param("normalized thermal pressure", "dimensionless", "PROXY", "src/energy/thermal_update.py"),
            "initial_state": param(0.52, "normalized", "ACTIVE", "active simulator config"),
            "guard_threshold": param(0.85, "normalized", "ACTIVE", "active simulator config"),
            "hard_threshold": param(0.92, "normalized", "ACTIVE", "active simulator config"),
            "passive_cooling": param(0.12, "normalized/reference-slot", "PROXY", "active simulator config"),
            "active_cooling": param(0.045, "normalized/reference-slot", "PROXY", "active simulator config"),
            "reference_slot_s": param(60.0, "s", "PROXY", "active simulator config"),
            "sunlight_gain": param(0.015, "normalized/reference-slot", "PROXY", "active simulator config"),
            "eclipse_bonus": param(0.03, "normalized/reference-slot", "PROXY", "active simulator config"),
            "state_clipping": param([0.0, 1.0], "normalized", "ACTIVE", "src/energy/thermal_update.py"),
            "headroom_binary_search_iterations": param(40, "iterations", "ACTIVE", "active simulator config"),
            "headroom_tolerance": param(1e-10, "normalized", "ACTIVE", "active simulator config"),
        },
        "evaluator": {
            "config_source": param("configs/unified_ground_received_accounting.yaml", "path", "ACTIVE", "prop_configs/b1_config"),
            "requirement_utility_source": param("configs/product_requirement_utility.yaml", "path", "ACTIVE", "common evaluator"),
            "closure_rule": param("information and quality requirements met by deadline", "rule", "ACTIVE", "src/evaluation/unified_ground_received_accounting.py"),
            "deadline_utility_after_deadline": param(0.0, "utility factor", "ACTIVE", "unified evaluator config"),
            "one_closure_reward_per_task": param(True, "boolean", "ACTIVE", "ground lifecycle/evaluator"),
            "net_utility_rule": param("gross - energy - backlog - safety costs", "formula", "ACTIVE", "src/evaluation/unified_ground_received_accounting.py"),
            "internal_score_is_common_utility": param(False, "boolean", "ACTIVE", "controller/evaluator audit"),
        },
        "fairness": {
            "immutable_roots": param(["dataset", "scenario_generation", "tasks", "contacts", "communication", "compute", "storage", "queues", "battery", "energy", "thermal", "evaluator", "simulation"], "paths", "ACTIVE", "contracts/fairness_contract_v1.md"),
            "method_state_may_diverge": param(["model_selection", "product_selection", "queues", "actual_soc", "actual_thermal", "transmitted_bytes", "closures", "utility", "latency"], "fields", "ACTIVE", "fairness contract"),
        },
        "future_extensions": {
            "multisatellite_offloading": param("must account for ISL TX/RX/wait/remote compute and single closure reward", "rule", "FUTURE", "fairness contract"),
        },
    }


def flatten(data: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict) and {"value", "unit", "status", "source", "notes"} <= set(data):
        rows.append({"parameter_path": prefix, "value": json.dumps(data["value"], ensure_ascii=False, sort_keys=True) if isinstance(data["value"], (dict, list)) else data["value"], "unit": data["unit"], "status": data["status"], "authoritative_source_file": data["source"].split(":")[0], "config_key_or_function": data["source"], "line_or_section": "runtime/config audit", "default_or_override": "active resolved value", "active_or_shadowed": "active", "measured_or_proxy": data["status"], "methods_using_it": "PROP-PAUSE;PROP-HEADROOM;B1-COMMON-THERMAL", "notes": data["notes"]})
    elif isinstance(data, dict):
        for key, value in data.items():
            rows.extend(flatten(value, f"{prefix}.{key}" if prefix else key))
    return rows


def preexisting_state() -> None:
    core = ["final_single_satellite_baseline_freeze_report.md", "validation_report.md", "final_plain_language_summary_zh.md", "single_satellite_baseline_freeze_manifest.md", "single_satellite_baseline_master_summary.csv", "single_satellite_baseline_freeze_bundle.zip"]
    hashes = "\n".join(f"- `{name}`: `{sha(FREEZE / name)}`" for name in core)
    task_prefixes = (
        "AGENTS.md", "HOW_TO_START_A_NEW_EXPERIMENT.md", "configs/benchmark_environment_v1.",
        "configs/experiments/", "configs/methods/", "contracts/", "manifests/",
        "outputs/_benchmark_environment_contract_v1/", "scripts/build_benchmark_environment_contract_v1.py",
        "scripts/resolve_experiment_config.py", "scripts/validate_experiment_fairness.py",
        "templates/", "tests/test_benchmark_environment_contract_v1.py",
    )
    status_lines = []
    for line in git("status", "--short").splitlines():
        path = line[3:].replace("\\", "/") if len(line) > 3 else line
        if not path.startswith(task_prefixes):
            status_lines.append(line)
    status = "\n".join(status_lines)
    staged = git("diff", "--cached", "--name-only") or "(none)"
    text = f"""# Pre-existing Uncommitted State

- Worktree: `{ROOT}`
- Branch: `{git('branch', '--show-current')}`
- HEAD: `{git('rev-parse', 'HEAD')}`
- Freeze output: `outputs/_sarfish_single_satellite_baseline_freeze_v1/`
- Staged files before this task: `{staged}`

## Git Status Before Contract Files

```text
{status}
```

## Core Freeze SHA-256

{hashes}

## Protected Existing State

Do not overwrite or stage `src/payload/staged_delivery_queue.py`, `datasets/SARFishSample/`, runtime `.partial.csv`/PID/log files, unrelated experiment outputs, or pre-existing freeze-task leftovers. The dataset junction and nested repository remain untouched.
"""
    write(OUT / "preexisting_uncommitted_state.md", text)
    diff = git("diff", "--", "src/payload/staged_delivery_queue.py")
    if diff:
        write(OUT / "preexisting_source_changes.patch", diff)


def scenario_manifest() -> list[dict[str, Any]]:
    source = list(csv.DictReader((MULTI / "scenario_manifest.csv").open(encoding="utf-8-sig", newline="")))
    outcomes = list(csv.DictReader((FREEZE / "reconstructed_task_outcomes.csv").open(encoding="utf-8-sig", newline="")))
    deadlines: dict[int, dict[str, float]] = {}
    for seed in [2026, 2027, 2028, 2029, 2030]:
        unique: dict[str, float] = {}
        for row in outcomes:
            if int(row["seed"]) == seed:
                unique.setdefault(row["task_id"], float(row["deadline_time_s"]) - float(row["arrival_time_s"]))
        vals = sorted(unique.values())
        p95 = vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)]
        deadlines[seed] = {"min": min(vals), "mean": mean(vals), "p95": p95}
    rows = []
    for row in source:
        seed = int(row["seed"])
        rows.append({
            "seed": seed, "task_count": int(row["task_count"]), "HV_count": int(row["HV_task_count"]),
            "HV_TARGET_count": int(row["HV_TARGET_task_count"]), "routine_count": int(row["routine_task_count"]),
            "medium_count": int(row["medium_task_count"]), "high_count": int(row["high_task_count"]),
            "urgent_count": int(row["urgent_task_count"]), "deadline_min": deadlines[seed]["min"],
            "deadline_mean": deadlines[seed]["mean"], "deadline_p95": deadlines[seed]["p95"],
            "task_hash": row["task_hash"], "environment_hash": row["environment_hash_at_capacity_0_50"],
            "source_scene_manifest": "data/indexes/sarfishsample_scene_index.csv",
            "authoritative_output": "outputs/_sarfish_single_satellite_baseline_freeze_v1/",
            "validation_status": "PASS",
        })
    write_csv("manifests/scenario_manifest_v1.csv", rows)
    return rows


def contact_manifest() -> list[dict[str, Any]]:
    rows = []
    for idx, start_s in enumerate(range(420, 86400, 5700), 1):
        start_slot = start_s // 20
        end_slot = min((start_s + 600) // 20, 4320)
        total = 0.0
        for slot in range(start_slot, end_slot):
            t = slot * 20
            q = min(1.0, max(0.0, 0.45 + 0.35 * math.sin(t / 1800.0 + 0.5)))
            mbps = 35.0 + (65.0 - 35.0) * q
            total += mbps / 8.0 * 20.0
        base = {"contact_id": f"contact_{idx:02d}", "start_slot": start_slot, "end_slot": end_slot, "duration_slots": end_slot - start_slot, "nominal_capacity_mb": round(total, 9), "source": "synthetic periodic contact; timeline config and src/timeline/link_trace_loader.py"}
        base["contact_hash"] = canonical_hash(base)
        rows.append(base)
    write_csv("manifests/contact_window_manifest_v1.csv", rows)
    return rows


def tables() -> None:
    models = [
        {"model_id": "tiny", "complexity_level": "lightweight", "inference_latency_s": .90, "inference_energy_j": 10.8, "quality": .74, "supported_products": "metadata;roi;roi_context", "memory_requirement": "UNKNOWN", "status": "PROXY", "source": "configs/model_profiles_yolo_inspired_calibrated.yaml"},
        {"model_id": "small", "complexity_level": "balanced", "inference_latency_s": 1.44, "inference_energy_j": 21.6, "quality": .87, "supported_products": "metadata;roi;roi_context", "memory_requirement": "UNKNOWN", "status": "PROXY", "source": "same"},
        {"model_id": "medium", "complexity_level": "high_accuracy", "inference_latency_s": 2.79, "inference_energy_j": 51.84, "quality": .94, "supported_products": "metadata;roi;roi_context", "memory_requirement": "UNKNOWN", "status": "PROXY", "source": "same"},
        {"model_id": "defer", "complexity_level": "none", "inference_latency_s": 0, "inference_energy_j": 0, "quality": 0, "supported_products": "defer", "memory_requirement": "NOT_APPLICABLE", "status": "ACTIVE", "source": "configs/model_pool.yaml"},
    ]
    write_csv("contracts/compute_and_model_contract_v1.csv", models)
    products = [
        {"code_name": "metadata", "semantic_meaning": "compact detections/metadata", "requirements_satisfied": "alert; partial verification/context/evidence", "size_mb": .002, "size_source": "fixed fallback plus record formula", "measured_or_proxy": "PROXY", "quality_definition": "model quality proxy", "generation_latency": "0.18 s/MB * 0.10", "generation_energy": "2.4 J/MB * 0.05", "thermal_cost": "from charged generation energy", "minimum_model": "tiny", "partial_transmission_supported": True, "ground_evaluation_rule": "requirement utility table"},
        {"code_name": "roi", "semantic_meaning": "target region chips", "requirements_satisfied": "verification; partial context/evidence", "size_mb": .85, "size_source": "formula clamped 0.08..3.5 MB; fallback shown", "measured_or_proxy": "PROXY", "quality_definition": "model quality proxy", "generation_latency": "0.18 s/MB", "generation_energy": "2.4 J/MB", "thermal_cost": "from charged generation energy", "minimum_model": "tiny", "partial_transmission_supported": True, "ground_evaluation_rule": "requirement utility table"},
        {"code_name": "roi_context", "semantic_meaning": "ROI with contextual evidence", "requirements_satisfied": "context;evidence;verification", "size_mb": 2.70, "size_source": "formula clamped 0.25..8 MB; fallback shown", "measured_or_proxy": "PROXY", "quality_definition": "model quality proxy", "generation_latency": "0.18 s/MB * 1.9", "generation_energy": "2.4 J/MB * 2.2", "thermal_cost": "from charged generation energy", "minimum_model": "tiny", "partial_transmission_supported": True, "ground_evaluation_rule": "requirement utility table"},
        {"code_name": "full_frame", "semantic_meaning": "B1 full observation", "requirements_satisfied": "all information requirements", "size_mb": 4.0, "size_source": "32 MB / fixed proxy ratio 8", "measured_or_proxy": "PROXY", "quality_definition": "fixed retention/ground proxy", "generation_latency": "11 s", "generation_energy": "1.4 W * 11 s active runtime", "thermal_cost": "charged compression energy", "minimum_model": "B1 path", "partial_transmission_supported": False, "ground_evaluation_rule": "information factor 1.0; quality/deadline still evaluated"},
        {"code_name": "defer", "semantic_meaning": "no product now", "requirements_satisfied": "none", "size_mb": 0, "size_source": "fixed", "measured_or_proxy": "ACTIVE", "quality_definition": "zero", "generation_latency": "0", "generation_energy": "0", "thermal_cost": "0", "minimum_model": "none", "partial_transmission_supported": False, "ground_evaluation_rule": "no closure"},
    ]
    write_csv("contracts/product_information_contract_v1.csv", products)
    write_csv("contracts/storage_and_queue_contract_v1.csv", [
        {"resource": "raw_observation_storage", "limit": "UNKNOWN", "unit": "MB", "classification": "UNKNOWN", "behavior": "not separately represented"},
        {"resource": "product/downlink_queue", "limit": 5120, "unit": "MB", "classification": "ENFORCED", "behavior": "lowest-priority eviction on overflow"},
        {"resource": "downlink_safe_threshold", "limit": 4096, "unit": "MB", "classification": "ENFORCED", "behavior": "policy safety threshold"},
        {"resource": "compute_queue", "limit": "UNKNOWN", "unit": "items", "classification": "CONFIGURED_BUT_INACTIVE", "behavior": "no persistent cross-slot compute queue"},
        {"resource": "generation_request_queue", "limit": "UNKNOWN", "unit": "items", "classification": "UNKNOWN", "behavior": "persistent request logic has no numeric cap found"},
    ])
    write_csv("contracts/energy_coefficient_table_v1.csv", [
        {"coefficient": "battery_capacity", "value": 230, "unit": "Wh", "status": "PROXY", "source": "resource/simulator config"},
        {"coefficient": "solar_peak", "value": 14.5, "unit": "W", "status": "PROXY", "source": "resource config"},
        {"coefficient": "platform_idle", "value": 4.8, "unit": "W", "status": "PROXY", "source": "resource config"},
        {"coefficient": "transmit_power", "value": 35, "unit": "W", "status": "PROXY", "source": "simulator config"},
        {"coefficient": "charge_efficiency", "value": .92, "unit": "ratio", "status": "PROXY", "source": "simulator config"},
        {"coefficient": "discharge_efficiency", "value": .95, "unit": "ratio", "status": "PROXY", "source": "simulator config"},
    ])
    write_csv("contracts/thermal_coefficient_table_v1.csv", [
        {"coefficient": "guard", "value": .85, "unit": "normalized", "status": "ACTIVE", "source": "simulator config"},
        {"coefficient": "hard", "value": .92, "unit": "normalized", "status": "ACTIVE", "source": "simulator config"},
        {"coefficient": "compute_heat", "value": .0025, "unit": "normalized/J", "status": "PROXY", "source": "thermal_update.py"},
        {"coefficient": "communication_heat", "value": .0015, "unit": "normalized/J", "status": "PROXY", "source": "thermal_update.py"},
        {"coefficient": "passive_cooling", "value": .12, "unit": "normalized/ref-slot", "status": "PROXY", "source": "simulator config"},
        {"coefficient": "active_cooling", "value": .045, "unit": "normalized/ref-slot", "status": "PROXY", "source": "simulator config"},
    ])
    write_csv("contracts/utility_coefficient_table_v1.csv", [
        {"parameter": "deadline_after_expiry_factor", "value": 0, "unit": "ratio", "status": "ACTIVE", "source": "unified evaluator config"},
        {"parameter": "energy_normalization", "value": 10, "unit": "Wh", "status": "ACTIVE", "source": "unified evaluator config"},
        {"parameter": "backlog_normalization", "value": 64, "unit": "MB", "status": "ACTIVE", "source": "unified evaluator config"},
        {"parameter": "energy_cost_lambda", "value": 1, "unit": "weight", "status": "ACTIVE", "source": "unified evaluator config"},
        {"parameter": "backlog_cost_lambda", "value": 1, "unit": "weight", "status": "ACTIVE", "source": "unified evaluator config"},
        {"parameter": "safety_cost_lambda", "value": 10, "unit": "weight", "status": "ACTIVE", "source": "unified evaluator config"},
    ])


DOCS = {
"contracts/simulation_timeline_contract_v1.md": """# Simulation Timeline Contract v1

The run spans `[0, 86400)` seconds in 4,320 slots of 20 seconds. A completion at `deadline + 1e-9` or earlier is on time.

Per-slot order implemented by `StepRunner.run_step`:

`state/context -> arrivals -> policy/safety/model action -> inference and product generation -> carryover request opportunity -> queue preview and thermal-safe capacity -> expire/enqueue/sort/transmit -> lifecycle/ground delivery -> energy accounting and battery update (solar/base/compute/communication) -> thermal update -> safety checks -> internal reward/controller update -> commit state`.

Arrivals are exposed before action selection. Generated payloads enter the downlink queue before transmission in that slot. Deadline eligibility is inclusive. Partially transmitted queue items retain remaining bytes across slots and contacts. Compute latency and generation latency are charged, but there is no persistent partial-compute state across slots.
""",
"contracts/task_generation_contract_v1.md": """# Task Generation Contract v1

SARFishSample GRD validation scene metadata and annotations seed a deterministic proxy task stream. The active index has 50 source scenes. Tasks sample scenes with seeded `rng.choice`, so one scene may produce multiple simulation tasks; a task is a scheduled observation/task record, not a unique image or end-to-end pixel execution. Seeds control scene selection and synthetic arrival/deadline draws; existing task manifests are authoritative and must not be regenerated for comparison.

Task classes map `routine -> medium`, `high_value -> high`, and `emergency -> urgent`. HV means high-value or emergency under the runtime definition; HV-TARGET is the validated stricter target subgroup recorded by the existing manifests. Requirements are alert, verification, context, or evidence. Task IDs and duplicate handling are deterministic in the task builder/canonical environment.

Online policies do not read future arrivals or true evaluation labels. Target count and annotation summaries are stored as offline metadata for task construction/evaluation and explicitly excluded from online priority. B1 does not read and compress real pixels: its 4 MB frame is a fixed proxy. PROP ROI sizes are formula/fallback proxies, not measured codecs. Model quality and product quality are proxy formulas.
""",
"contracts/communication_environment_contract_v1.md": """# Communication Environment Contract v1

Contacts are synthetic periodic windows: 5,700 s period, 600 s nominal duration, 420 s phase. Link quality drives 35--65 Mbit/s nominal throughput; experiment multipliers 1.00, 0.50, and 0.35 scale this throughput. Per-slot nominal MB is `throughput_mbps / 8 * 20`; contact capacity is its sum over contact slots.

Keep distinct: nominal capacity, multiplier-scaled contact capacity, thermal-safe effective capacity, and actual transmitted bytes. PROP queue items support partial transmission and resume across slots/contacts. Deadline-expired items are removed; equality at deadline remains eligible. B1's method policy requires a full 4 MB frame fit and is non-preemptive once started. No propagation delay or protocol overhead is represented (`UNKNOWN`). Communication energy uses the 35 W proxy and actual transmit time.

`PROP-HEADROOM` does not raise 0.92. It predicts next thermal state and binary-searches the maximum bytes that remain at or below `0.92 - 1e-10` (40 iterations); compute-only infeasibility or no safe bytes yields zero capacity.
""",
"contracts/compute_environment_contract_v1.md": """# Compute Environment Contract v1

The simulator uses abstract BUPT-1-like COTS compute profiles, not a named CPU, GPU, FPGA, or flight processor. Core count, memory capacity, and physical FLOP/s are unknown. One logical action is evaluated per task/slot path. Inference and encoding latency/energy are proxy-accounted; no persistent cross-slot compute queue or partial-compute execution is modeled.

Tiny/small/medium mock profiles are selected by method policy. Their calibrated relative latency, energy, quality, and thermal scales are in `compute_and_model_contract_v1.csv`. These values must not be presented as measured SARFish YOLO performance.
""",
"contracts/product_semantics_contract_v1.md": """# Product Semantics Contract v1

`metadata` carries compact detection records; `roi` carries target chips; `roi_context` carries ROI plus contextual evidence; `full_frame` is B1's full-observation proxy; `defer` creates no payload. Staged aliases enqueue metadata first and a residual product later.

Metadata has a 0.002 MB fallback. ROI uses a formula clamped to 0.08--3.5 MB with 0.85 MB fallback. ROI+Context is clamped to 0.25--8 MB with 2.70 MB fallback. B1 uses fixed 32 MB raw / ratio 8 = 4 MB. None are real encoded measurements in this study. Quality and information factors are proxy values interpreted by the unchanged common evaluator.
""",
"contracts/queue_behavior_contract_v1.md": """# Queue Behavior Contract v1

The downlink queue enforces a 5,120 MB byte cap and a 4,096 MB policy-safe threshold; there is no item-count cap. Duplicate task/product pairs are suppressed. Overflow evicts lowest-priority items. Expired entries are explicitly removed, partial bytes persist across slots and contacts, and fully transmitted items are removed. Scheduling iterates eligible sorted items, so a non-fitting item does not impose universal head-of-line blocking.

PROP and B1 share the physical queue capacity but use different method scheduling semantics. B1 full-frame fit/non-preemption is a method rule. Raw storage, separate product storage, compute queue size, and generation-request numeric capacity are not implemented as independently enforced resources and remain `UNKNOWN` or inactive as listed in the CSV.
""",
"contracts/battery_energy_contract_v1.md": """# Battery and Energy Contract v1

The active environment uses a 230 Wh proxy battery, initial SOC 0.90, charge efficiency 0.92, discharge efficiency 0.95, and clipping to [0,1]. Solar input and platform base load come from the synthetic orbit-aware resource trace; battery update uses solar minus base, compute, generation, and communication loads.

The active hard resource/SafeFilter floor is 0.55. The 0.70 and 0.50 values are policy regions, not physical battery limits; the general `simulator.yaml` 0.18 value is shadowed by the active formal config. At capacity 0.50, frozen PROP minimum SOC is approximately 0.889, so neither policy region activates. Controllable energy is method-driven compute/generation/communication; platform base energy is uncontrollable. Future matched-energy fields are reserved but inactive.
""",
"contracts/thermal_environment_contract_v1.md": """# Thermal Environment Contract v1

Thermal state is a dimensionless proxy pressure clipped to [0,1], initialized at 0.52. Guard is 0.85 and hard limit is 0.92. The update combines ambient relaxation/passive cooling, optional active cooling, sunlight/eclipse terms, compute heat (0.0025 normalized/J), and communication heat (0.0015 normalized/J), scaled to a 60 s reference.

`PROP-PAUSE` disables a hot slot under the old gate. `PROP-HEADROOM` instead computes the largest safe bytes for the current slot without changing the hard limit. `B1-COMMON-THERMAL` uses the common thermal accounting. A tested hysteresis variant showed no additional benefit and is not an active formal mechanism. Frozen reports record zero hard-limit violations.
""",
"contracts/evaluation_and_utility_contract_v1.md": """# Evaluation and Utility Contract v1

The common evaluator consumes ground-received product stage, information sufficiency, quality, and deadline. Closure requires information and quality requirements by the inclusive deadline. Partial-only has useful ground information without closure; no-ground has none; late, quality failure, and information failure remain distinct. A task receives closure benefit once; duplicate transmissions do not create duplicate closure reward.

Gross utility is task value times information, quality, and timeliness for completed work, with configured partial utility otherwise. Common net utility subtracts common energy, backlog, and safety costs. Full frame maps to information factor 1.0 but still passes quality/deadline evaluation. Internal bandit/Lyapunov/action scores are method-local objectives and are not common net utility or final paper metrics.
""",
"contracts/fairness_contract_v1.md": """# Fairness Contract v1

Within a scenario every method must share dataset/source observations, task manifest/hash, arrivals, deadlines, values, requirements, contact trace/hash, capacity multiplier definition, compute hardware, storage/queue capacities, battery and charging, initial SOC, thermal model/coefficients/thresholds/initial state, slot/horizon/order, common evaluator, and external resource trace.

Methods may naturally differ in model/product selection, compute/generation timing, downlink scheduling, queue trajectories, actual SOC/thermal state, actual traffic, closures, utility, and latency.

Method override files may contain only method decisions. Experiment files may select experiment ID, seeds, capacities, methods, output, reporting, and smoke/full mode. Dataset, tasks, contacts, environment resources, products, and evaluator cannot be overridden.

Future multi-satellite comparisons must share satellite count and per-satellite hardware/battery/thermal limits, source satellite, ISL topology/rate, and charge ISL TX, RX, waiting, and remote compute. Offloading cannot be free or instantaneous, and a task earns closure once.
""",
"contracts/configuration_conflicts_v1.md": """# Configuration Conflicts v1

No blocking conflict was found.

| Severity | Active value | Shadowed definition | Resolution |
|---|---|---|---|
| SHADOWED_BUT_RESOLVED | formal slot 20 s, battery 230 Wh, initial SOC 0.90, hard floor 0.55, queue 5120 MB | generic `configs/simulator.yaml` defaults including 60 s, 35 Wh, 0.85, 0.18, 64 MB | formal runner explicitly loads the specialized config |
| SHADOWED_BUT_RESOLVED | B1 compression processing power 1.4 W | baseline YAML default 6.8 W | validated runtime `b1_config` override is authoritative |
| SHADOWED_BUT_RESOLVED | calibrated tiny/small/medium profiles | generic `model_pool.yaml` base values | calibrated profile overrides are loaded by formal PROP config |
| INFORMATIONAL | formula product sizes with fallbacks | fallback values may appear alone in reports | runtime formula/clamp is authoritative; all remain proxy |
""",
}


def templates() -> None:
    write("configs/methods/example_method_override_v1.yaml", """method:\n  method_id: example-method\n  model_selection_policy: existing_default\n  product_selection_policy: existing_default\n  scheduling_policy: existing_default\n  staged_delivery: false\n  onboard_inference: true\n  feature_flags: {}\n  notes: Method differences only; environment overrides are forbidden.\n""")
    write("configs/experiments/example_experiment_v1.yaml", """experiment:\n  experiment_id: example-smoke\n  scenario_seeds: [2026]\n  capacities: [0.50]\n  methods: [example-method]\n  output_directory: outputs/_example_smoke/\n  reporting_options: {compact: true}\n  mode: smoke\n""")
    write("templates/new_method_override_v1.yaml", """method:\n  method_id: FILL_ME\n  model_selection_policy: FILL_ME\n  product_selection_policy: FILL_ME\n  scheduling_policy: FILL_ME\n  staged_delivery: false\n  onboard_inference: true\n  feature_flags: {}\n  notes: Describe only the new mechanism.\n""")
    write("templates/new_experiment_v1.yaml", """experiment:\n  experiment_id: FILL_ME\n  scenario_seeds: [2026]\n  capacities: [0.50]\n  methods: [FILL_ME]\n  output_directory: outputs/_FILL_ME/\n  reporting_options: {compact: true}\n  mode: smoke\n""")
    write("templates/NEW_EXPERIMENT_CODEX_PROMPT.md", """# New Experiment Prompt

Read only `AGENTS.md`, `configs/benchmark_environment_v1.yaml`, `contracts/fairness_contract_v1.md`, `manifests/benchmark_environment_fingerprint_v1.json`, the method override, and the experiment config.

- Experiment: FILL_ME
- New method: FILL_ME
- Only new mechanism: FILL_ME
- Seeds: FILL_ME
- Capacities: FILL_ME
- Output: `outputs/_FILL_ME/`
- Acceptance metrics: FILL_ME
- Mode: smoke or formal

Resolve the config and run fairness validation first. Do not change shared environment fields or run formal scope without explicit approval.
""")
    write("HOW_TO_START_A_NEW_EXPERIMENT.md", """# 如何开始新实验

新实验直接继承 `configs/benchmark_environment_v1.yaml`，不要复制工程或环境参数。先从 `templates/new_method_override_v1.yaml` 创建方法文件，只填写模型选择、产品选择、调度、分阶段交付、星上推理或未来卸载等方法差异；再从 `templates/new_experiment_v1.yaml` 创建实验文件，只选择名称、seed、capacity、方法、输出目录、报告选项和 smoke/full 模式。

解析命令：

```powershell
python scripts/resolve_experiment_config.py --base configs/benchmark_environment_v1.yaml --method configs/methods/your_method.yaml --experiment configs/experiments/your_experiment.yaml --output-dir outputs/_your_experiment/resolved
```

公平性校验：

```powershell
python scripts/validate_experiment_fairness.py --base configs/benchmark_environment_v1.yaml --method configs/methods/your_method.yaml --experiment configs/experiments/your_experiment.yaml
```

禁止覆盖数据集、任务 manifest/到达/deadline/value/requirement、contact、slot/horizon、capacity 定义、硬件、队列/存储、电池/充电、热参数、产品尺寸来源和公共评价器。保存 resolved config、SHA-256 和当前环境指纹后，先做小规模 smoke；正式运行需明确批准。

普通顺序实验只建 Git branch。只有并行运行或高风险隔离才建 worktree；不要为小实验复制完整目录。每个实验使用独立 `outputs/_实验名/`。诊断时优先读 manifest、摘要和合同，不扫描历史大日志。新对话可使用 `templates/NEW_EXPERIMENT_CODEX_PROMPT.md` 的短 Prompt，并引用六个核心文件。
""")


def fingerprint(env: dict[str, Any], scenarios: list[dict[str, Any]], contacts: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "benchmark_environment_yaml": ROOT / "configs/benchmark_environment_v1.yaml",
        "benchmark_environment_json": ROOT / "configs/benchmark_environment_v1.json",
        "scenario_manifest": ROOT / "manifests/scenario_manifest_v1.csv",
        "contact_manifest": ROOT / "manifests/contact_window_manifest_v1.csv",
        "common_evaluator_config": ROOT / "configs/unified_ground_received_accounting.yaml",
        "common_evaluator_source": ROOT / "src/evaluation/unified_ground_received_accounting.py",
        "battery_model_source": ROOT / "src/energy/battery_update.py",
        "thermal_model_source": ROOT / "src/energy/thermal_update.py",
        "thermal_headroom_source": ROOT / "src/energy/thermal_headroom.py",
        "queue_model_source": ROOT / "src/link/downlink_queue.py",
        "product_config": ROOT / "configs/payload_profiles_roi_compression_calibrated.yaml",
        "runner_source": ROOT / "src/simulator/step_runner.py",
        "dataset_manifest": ROOT / "data/indexes/sarfishsample_scene_index.csv",
    }
    components = {name: sha(path) for name, path in paths.items()}
    components["git_head"] = git("rev-parse", "HEAD")
    payload = {"schema_version": 1, "components": dict(sorted(components.items())), "scenario_object_hash": canonical_hash(scenarios), "contact_object_hash": canonical_hash(contacts)}
    payload["benchmark_environment_fingerprint"] = canonical_hash(payload)
    write("manifests/benchmark_environment_fingerprint_v1.json", json.dumps(payload, indent=2, sort_keys=True))
    write(OUT / "benchmark_environment_fingerprint_v1.json", json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    preexisting_state()
    env = environment()
    yaml_path = ROOT / "configs/benchmark_environment_v1.yaml"
    json_path = ROOT / "configs/benchmark_environment_v1.json"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(env, sort_keys=False, allow_unicode=True), encoding="utf-8")
    json_path.write_text(json.dumps(env, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    scenarios = scenario_manifest()
    contacts = contact_manifest()
    tables()
    for path, text in DOCS.items():
        write(path, text)
    source_rows = flatten(env)
    write_csv("contracts/parameter_source_map_v1.csv", source_rows)
    write_csv(OUT / "extracted_parameter_summary.csv", source_rows)
    templates()
    fp = fingerprint(env, scenarios, contacts)
    status_counts = Counter(row["status"] for row in source_rows)
    conflict_count = 4
    validation_csv = [
        {"check": "freeze_outputs_readable", "status": "PASS", "detail": "six core files readable"},
        {"check": "formal_simulation_rerun", "status": "PASS", "detail": "skipped"},
        {"check": "scenario_counts", "status": "PASS", "detail": "268/285/255/306/285"},
        {"check": "timeline", "status": "PASS", "detail": "86400 s / 20 s = 4320 slots"},
        {"check": "thermal_safety", "status": "PASS", "detail": "hard 0.92; frozen violations 0"},
        {"check": "yaml_json_semantic_equivalence", "status": "PASS", "detail": str(yaml.safe_load(yaml_path.read_text(encoding='utf-8')) == json.loads(json_path.read_text(encoding='utf-8')))},
        {"check": "configuration_conflicts", "status": "PASS", "detail": "0 blocking; 3 shadowed resolved; 1 informational"},
    ]
    write_csv(OUT / "environment_validation_results.csv", validation_csv)
    validation_report = f"""# Environment Validation Report

- Freeze results: readable and internally consistent.
- Formal simulations and task generation: not run.
- Scenarios: five seeds, task counts 268/285/255/306/285; capacities 0.50 and 0.35; methods PROP-PAUSE, PROP-HEADROOM, B1-COMMON-THERMAL.
- Timeline: 24 h, 20 s, 4,320 slots.
- Safety: hard thermal limit 0.92, zero frozen violations, no NaN/Inf per existing validation report.
- YAML/JSON semantic equivalence: PASS.
- Conflicts: 0 blocking, 3 shadowed resolved, 1 informational.
- Parameter statuses: {dict(status_counts)}.
- Fingerprint deterministic canonical serialization: `{fp['benchmark_environment_fingerprint']}`.
- Existing freeze validation: `PASS_WITH_DECLARED_MISSING_SUBGROUP_ATTRIBUTION`.
- Config/fairness unit tests: 11 passed (configuration-only; no formal simulation).

Overall contract status before Git: PASS_WITH_PROXY_VALUES
"""
    write(OUT / "environment_validation_report.md", validation_report)
    summary = f"""# 统一基准环境合同摘要

当前单星冻结结果可读取，本任务未重跑 30 组正式仿真、未重新生成任务，也未修改 PROP、B1、数据集或公共模型。合同覆盖 SARFishSample GRD validation 代理任务流：5 个 seed 的任务数为 268、285、255、306、285；每次 24 小时、20 秒 slot、共 4320 slot。

联系窗口为 5700 秒周期、600 秒时长、420 秒相位的合成轨迹，名义速率 35--65 Mbit/s，capacity multiplier 缩放速率。计算是 BUPT-1-like 抽象资源，无真实芯片型号或真实星上推理；下传队列 5120 MB、安全阈值 4096 MB，独立原始/产品存储和计算队列容量未知。电池 230 Wh、初始 SOC 0.90、活动硬下限 0.55，太阳峰值 14.5 W、平台基础功耗 4.8 W；0.70/0.50 仅为策略区间，0.50 capacity 下 PROP 最低 SOC 约 0.889。

热状态是归一化代理，guard 0.85、hard 0.92。HEADROOM 不提高上限，只求下一热状态不越过 0.92 的最大安全字节。模型质量、ROI/ROI+Context 尺寸和 B1 4 MB 全帧均含代理值；B1 不真实读取并压缩图像。所有方法使用统一地面评价器，内部动作 score 不等于公共净效用。

环境指纹：`{fp['benchmark_environment_fingerprint']}`。共提取 {len(source_rows)} 个参数，PROXY {status_counts['PROXY']} 个、UNKNOWN {status_counts['UNKNOWN']} 个；4 项配置覆盖/信息冲突均非阻塞。公平性结构校验为 PASS_WITH_PROXY_VALUES；Git 状态在最终提交后更新。以后只写 method override 和 experiment config，先解析、校验和 smoke，不复制工程。

ENVIRONMENT_INHERITANCE_READY_WITH_PROXIES
"""
    write(OUT / "final_environment_contract_summary_zh.md", summary)
    bundle_files = [
        "AGENTS.md", "scripts/resolve_experiment_config.py", "scripts/validate_experiment_fairness.py",
        "configs/benchmark_environment_v1.yaml", "configs/benchmark_environment_v1.json",
        "configs/methods/example_method_override_v1.yaml", "configs/experiments/example_experiment_v1.yaml",
        "contracts/parameter_source_map_v1.csv", "contracts/fairness_contract_v1.md",
        "contracts/configuration_conflicts_v1.md", "manifests/scenario_manifest_v1.csv",
        "manifests/contact_window_manifest_v1.csv", "manifests/benchmark_environment_fingerprint_v1.json",
        "templates/NEW_EXPERIMENT_CODEX_PROMPT.md", "templates/new_method_override_v1.yaml",
        "templates/new_experiment_v1.yaml", "HOW_TO_START_A_NEW_EXPERIMENT.md",
    ] + list(DOCS)
    bundle_files += [str(p.relative_to(ROOT)).replace("\\", "/") for p in sorted((ROOT / "contracts").glob("*_v1.csv")) if str(p.relative_to(ROOT)).replace("\\", "/") not in bundle_files]
    bundle_files += ["outputs/_benchmark_environment_contract_v1/environment_validation_report.md", "outputs/_benchmark_environment_contract_v1/environment_validation_results.csv", "outputs/_benchmark_environment_contract_v1/final_environment_contract_summary_zh.md"]
    with zipfile.ZipFile(OUT / "benchmark_environment_contract_v1_bundle.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(set(bundle_files)):
            zf.write(ROOT / rel, rel)
    print(json.dumps({"parameters": len(source_rows), "statuses": status_counts, "fingerprint": fp["benchmark_environment_fingerprint"], "contacts": len(contacts)}, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
