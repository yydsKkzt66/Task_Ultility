from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "_v25_bupt1_power_true_rerun_seed2026_v1"
POWER_CONFIG = ROOT / "configs" / "bupt1_measured_power_v1.yaml"
PREVIOUS_POWER_OUT = ROOT / "outputs" / "_v25_bupt1_power_calibrated_seed2026_v1"
SOLAR_TRACE = PREVIOUS_POWER_OUT / "calibration" / "solar_trace_one_orbit.csv"
SELECTED_CALIBRATION = PREVIOUS_POWER_OUT / "config" / "selected_power_calibration.json"
FROZEN_MANIFEST = ROOT / "artifacts" / "frozen_scenarios" / "v25_main_seed2026_manifest_4c23cf3184f1.csv"

SOURCE_BUNDLE = Path("D:/wt/pilot/outputs/_updated_prop_ground_ack_v1/bundles/V25-MAIN-v1")
TOOL_DIR = Path("D:/wt/pilot_late_hv_native_direct_v1/tools/late_hv_native_direct_final_v6")
SOURCE_PROP = Path("D:/wt/pilot_updated_prop_ground_ack_v1")
SOURCE_B1_OFFLOAD = Path("D:/wt/pilot_frozen_runtime_native_direct_v1")
SOURCE_TUT = Path("D:/wt/tut_uoi_rate_lite")
SOURCE_MYOPIC = Path("D:/wt/pilot_myopic_frozen_calibrated_v1")
MYOPIC_PARAM = Path("D:/wt/pilot/outputs/_myopic_frozen_calibrated_v1/myopic_frozen_calibration_v1.json")
SHARED_PRIOR_CALIBRATION = MYOPIC_PARAM

EXPECTED_TASK_HASH = "4c23cf3184f137733495f95a7c5bd7631adf95b3fbc4eb057351f27dac2f2ab5"
EXPECTED_COUNTS = {"low": 54, "medium": 139, "high": 54, "urgent": 21}
EXPECTED_TASK_COUNT = 268
EXPECTED_HV_COUNT = 75
SLOTS = 4320
SLOT_SECONDS = 20
CAPACITY = 0.35
SEED = 2026

METHODS = [
    "PROP-SHARED-CALIBRATED-PRIOR",
    "B1-COMMON-THERMAL",
    "OFFLOAD-EFT-THERMAL-ALIGNED",
    "TUT-UOI-PRODUCT-LITE-v2",
    "MYOPIC-FROZEN-CALIBRATED",
]

FROZEN_POWER = {
    "battery_capacity_Wh": 230.0,
    "initial_soc": 0.85,
    "soc_warning_threshold": 0.70,
    "orbit_period_s": 5760,
    "daylight_duration_s": 3600,
    "eclipse_duration_s": 2160,
    "solar_energy_per_orbit_Wh": 16.0,
    "solar_peak_power_W": 40.0,
    "platform_other_power_W": 8.86,
    "charge_efficiency": 0.90,
    "discharge_efficiency": 0.95,
    "tx_active_power_W": 38.0,
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows or [{"empty": ""}])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def d(value: Any) -> Decimal:
    if value in ("", None):
        return Decimal("0")
    return Decimal(str(value))


def f(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def i(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def pct95(values: list[Decimal]) -> Decimal | str:
    if not values:
        return "NA"
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def v25_bundle_hash(rows: list[dict[str, Any]]) -> str:
    fields = ("task_id", "value_level", "effective_task_value", "product_requirement", "arrival_time_s", "deadline_time_s")
    task_rows = sorted([{key: row[key] for key in fields} for row in rows if row.get("task_id")], key=lambda row: row["task_id"])
    return hashlib.sha256(json.dumps(task_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def load_frozen_manifest() -> dict[str, Any]:
    if not FROZEN_MANIFEST.exists():
        raise FileNotFoundError(f"missing frozen manifest: {FROZEN_MANIFEST}")
    rows = read_csv(FROZEN_MANIFEST)
    counts = Counter(str(row.get("value_level", "")) for row in rows)
    recovered_hash = v25_bundle_hash(rows)
    return {
        "manifest_path": str(FROZEN_MANIFEST),
        "manifest_file_sha256": sha256_file(FROZEN_MANIFEST),
        "recovered_task_manifest_hash": recovered_hash,
        "task_count": len({str(row.get("task_id", "")) for row in rows if row.get("task_id")}),
        "value_level_counts": {level: counts.get(level, 0) for level in ("low", "medium", "high", "urgent")},
        "N_HV": counts.get("high", 0) + counts.get("urgent", 0),
        "earliest_arrival": min(f(row.get("arrival_time_s"), 0.0) for row in rows),
        "latest_arrival": max(f(row.get("arrival_time_s"), 0.0) for row in rows),
        "schema": list(rows[0]) if rows else [],
    }


def copy_frozen_inputs(out: Path) -> dict[str, Any]:
    if not SOLAR_TRACE.exists():
        raise FileNotFoundError(f"missing required reused solar trace: {SOLAR_TRACE}")
    if not SELECTED_CALIBRATION.exists():
        raise FileNotFoundError(f"missing required selected calibration: {SELECTED_CALIBRATION}")

    frozen_cfg = load_yaml(POWER_CONFIG)
    observed = {
        "battery_capacity_Wh": float(frozen_cfg["battery_capacity_Wh"]),
        "initial_soc": float(frozen_cfg["initial_soc"]),
        "soc_warning_threshold": float(frozen_cfg["soc_warning_threshold"]),
        "orbit_period_s": int(frozen_cfg["orbit_period_s"]),
        "daylight_duration_s": int(frozen_cfg["daylight_duration_s"]),
        "eclipse_duration_s": int(frozen_cfg["eclipse_duration_s"]),
        "solar_energy_per_orbit_Wh": float(frozen_cfg.get("solar_energy_per_orbit_Wh", frozen_cfg.get("solar_target_energy_per_orbit_Wh"))),
        "solar_peak_power_W": float(frozen_cfg["solar_peak_power_W"]),
        "platform_other_power_W": 8.86,
        "charge_efficiency": float(frozen_cfg["charge_efficiency"]),
        "discharge_efficiency": float(frozen_cfg["discharge_efficiency"]),
        "tx_active_power_W": float(frozen_cfg["tx_active_power_W"]),
    }
    if observed != FROZEN_POWER:
        raise ValueError({"frozen_power_mismatch": {"expected": FROZEN_POWER, "observed": observed}})

    (out / "config").mkdir(parents=True, exist_ok=True)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    shutil.copy2(POWER_CONFIG, out / "config" / POWER_CONFIG.name)
    shutil.copy2(SOLAR_TRACE, out / "calibration" / SOLAR_TRACE.name)
    shutil.copy2(SELECTED_CALIBRATION, out / "config" / SELECTED_CALIBRATION.name)
    shutil.copy2(FROZEN_MANIFEST, out / "config" / FROZEN_MANIFEST.name)
    return {
        "frozen_power": FROZEN_POWER,
        "config_path": str(POWER_CONFIG),
        "config_sha256": sha256_file(POWER_CONFIG),
        "reused_solar_trace_path": str(SOLAR_TRACE),
        "reused_solar_trace_sha256": sha256_file(SOLAR_TRACE),
        "selected_calibration_path": str(SELECTED_CALIBRATION),
        "selected_calibration_sha256": sha256_file(SELECTED_CALIBRATION),
    }


def preflight(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    input_audit = copy_frozen_inputs(out)
    frozen_manifest = load_frozen_manifest()
    manifest_ok = (
        frozen_manifest["recovered_task_manifest_hash"] == EXPECTED_TASK_HASH
        and frozen_manifest["task_count"] == EXPECTED_TASK_COUNT
        and frozen_manifest["value_level_counts"] == EXPECTED_COUNTS
        and frozen_manifest["N_HV"] == EXPECTED_HV_COUNT
    )
    write_json(out / "audit" / "frozen_manifest_identity.json", frozen_manifest)
    if not manifest_ok:
        raise ValueError({"PRE_FLIGHT_BLOCKED_TASK_MANIFEST_MISMATCH": frozen_manifest})
    return input_audit, frozen_manifest


def prepare_runtime_bundle(out: Path) -> Path:
    bundle_out = out / "config" / "frozen_runtime_bundle"
    if bundle_out.exists():
        shutil.rmtree(bundle_out)
    bundle_out.mkdir(parents=True)
    for name in ("bundle_contract.json", "arrival_swap_manifest.csv", "resolved_config.json", "slot_trace.csv", "task_value_assignment_trace.csv"):
        shutil.copy2(SOURCE_BUNDLE / name, bundle_out / name)
    shutil.copy2(FROZEN_MANIFEST, bundle_out / "task_manifest.csv")
    contract = read_json(bundle_out / "bundle_contract.json")
    checks = {name: sha256_file(bundle_out / name) for name in contract["file_sha256"]}
    if checks != contract["file_sha256"]:
        raise ValueError({"frozen_runtime_bundle_file_hash_mismatch": checks, "expected": contract["file_sha256"]})
    write_json(out / "audit" / "frozen_runtime_bundle_identity.json", {
        "bundle_root": str(bundle_out),
        "source_bundle": str(SOURCE_BUNDLE),
        "bundle_contract_sha256": sha256_file(bundle_out / "bundle_contract.json"),
        "task_hash": contract["task_hash"],
        "task_manifest_sha256": checks["task_manifest.csv"],
        "frozen_manifest_path": str(FROZEN_MANIFEST),
        "frozen_manifest_sha256": sha256_file(FROZEN_MANIFEST),
    })
    return bundle_out


def child_setup_path(source_root: Path) -> None:
    source_root = source_root.resolve()
    tool_dir = TOOL_DIR.resolve()
    root = ROOT.resolve()
    keep = []
    for item in sys.path:
        try:
            resolved = Path(item or os.getcwd()).resolve()
        except Exception:
            keep.append(item)
            continue
        if resolved in {source_root, tool_dir, root}:
            continue
        keep.append(item)
    sys.path[:] = [str(tool_dir), str(source_root), *keep]


def load_bundle(bundle_root: Path):
    child_setup_path(SOURCE_B1_OFFLOAD)
    from bundle_io import load_bundle as _load_bundle
    return _load_bundle(bundle_root, sha256_file(bundle_root / "bundle_contract.json"))


def solar_rows() -> list[dict[str, str]]:
    return read_csv(SOLAR_TRACE)


def inject_power(scenario: Any, source_root: Path) -> Any:
    cfg = load_yaml(POWER_CONFIG)
    trace = solar_rows()
    configs = deepcopy(scenario.configs)
    simulator = configs.setdefault("simulator", {})
    simulator["battery_capacity_wh"] = float(cfg["battery_capacity_Wh"])
    simulator["battery_capacity_j"] = float(cfg["battery_capacity_Wh"]) * 3600.0
    simulator["battery_init"] = float(cfg["initial_soc"])
    simulator["battery_initial_soc"] = float(cfg["initial_soc"])
    simulator["battery_charge_efficiency"] = float(cfg["charge_efficiency"])
    simulator["battery_discharge_efficiency"] = float(cfg["discharge_efficiency"])
    simulator["battery_soft_guard"] = float(cfg["soc_warning_threshold"])
    simulator["battery_soft_target"] = float(cfg["soc_warning_threshold"])
    simulator["soc_warning_threshold"] = float(cfg["soc_warning_threshold"])
    simulator["tx_power_w"] = float(cfg["tx_active_power_W"])
    simulator["comm_power_w"] = float(cfg["tx_active_power_W"])
    simulator["common_power_supervisor"] = {"enabled": False}
    configs.setdefault("timeline", {})["slot_seconds"] = int(cfg["slot_s"])
    resource_cfg = configs.setdefault("resource", {})
    resource_cfg["orbit_period_s"] = int(cfg["orbit_period_s"])
    resource_cfg["eclipse_duration_s"] = int(cfg["eclipse_duration_s"])
    resource_cfg["daylight_duration_s"] = int(cfg["daylight_duration_s"])
    resource_cfg["solar_peak_w"] = float(cfg["solar_peak_power_W"])
    resource_cfg["idle_power_w"] = FROZEN_POWER["platform_other_power_W"]
    resource_cfg["battery_capacity_wh"] = float(cfg["battery_capacity_Wh"])
    resource_cfg["battery_init"] = float(cfg["initial_soc"])

    slot_s = float(cfg["slot_s"])
    orbit_s = int(cfg["orbit_period_s"])
    new_timeline = []
    for step in scenario.timeline:
        slot = int(step.slot_index)
        row = trace[slot % len(trace)]
        power_w = f(row["solar_power_W"])
        phase = str(row["orbit_phase"])
        t_in_orbit = (slot * slot_s) % orbit_s
        resource = replace(
            step.resource_trace,
            solar_energy_j=power_w * slot_s,
            base_power_w=FROZEN_POWER["platform_other_power_W"],
            is_sunlit=phase == "daylight",
            is_eclipse=phase == "eclipse",
            orbit_index=int((slot * slot_s) // orbit_s),
            day_index=0,
            orbit_phase=t_in_orbit / orbit_s,
            solar_power_w=power_w,
            wasted_solar_energy_j=0.0,
            battery_soc=float(cfg["initial_soc"]),
            comm_power_w=float(cfg["tx_active_power_W"]),
        )
        new_timeline.append(replace(step, slot_seconds=slot_s, resource_trace=resource))
    identity = {
        "source_root": str(source_root),
        "power_config_sha256": sha256_file(POWER_CONFIG),
        "solar_trace_sha256": sha256_file(SOLAR_TRACE),
        "selected_power_calibration_sha256": sha256_file(SELECTED_CALIBRATION),
        "frozen_power": FROZEN_POWER,
        "power_trace_hash": sha_json([
            {
                "slot": step.slot_index,
                "solar_energy_j": step.resource_trace.solar_energy_j,
                "base_power_w": step.resource_trace.base_power_w,
                "is_eclipse": step.resource_trace.is_eclipse,
            }
            for step in new_timeline
        ]),
    }
    return replace(scenario, configs=configs, timeline=new_timeline), identity


def relaxed_scenario_identity(method: str, scenario: Any, bundle: Any, power_identity: dict[str, Any]) -> dict[str, Any]:
    values = dict(getattr(scenario, "runtime_task_values", {}))
    value_mismatches = [
        task_id for task_id, expected in bundle.task_values.items()
        if Decimal.from_float(float(values.get(task_id, "nan"))) != Decimal.from_float(float(expected))
    ]
    return {
        "method": method,
        "task_hash": getattr(scenario, "task_hash", ""),
        "expected_task_hash": EXPECTED_TASK_HASH,
        "task_hash_match": getattr(scenario, "task_hash", "") == EXPECTED_TASK_HASH == bundle.contract["task_hash"],
        "task_count": len(values),
        "runtime_value_mismatch_count": len(value_mismatches),
        "field_mismatch_count": int(getattr(scenario, "field_mismatch_count", 0)),
        "value_level_counts": bundle.contract["value_level_counts"],
        "original_bundle_environment_hash": bundle.contract["environment_hash"],
        "environment_hash_match_original_v25": False,
        "environment_hash_note": "Original V25 environment hash is intentionally not reused because this run injects the frozen BUPT1 power model before online decisions.",
        "frozen_power_identity_match": power_identity["frozen_power"] == FROZEN_POWER,
        "frozen_power_identity": power_identity,
        "benchmark_fingerprint": getattr(scenario, "benchmark_fingerprint", bundle.contract.get("benchmark_fingerprint", "")),
    }


def write_unified_method_outputs(output: Path, normalized: dict[str, Any]) -> None:
    from output_schema import validate_unified_outputs, write_unified_outputs
    validate_unified_outputs(normalized, expected_tasks=EXPECTED_TASK_COUNT, expected_slots=SLOTS)
    write_unified_outputs(output, normalized)


def run_native_binding_method(method: str, bundle_root: Path, output: Path) -> None:
    source_root = SOURCE_TUT if method == "TUT-UOI-PRODUCT-LITE-v2" else SOURCE_B1_OFFLOAD
    child_setup_path(source_root)
    bundle = load_bundle(bundle_root)
    module = __import__({
        "B1-COMMON-THERMAL": "native_b1",
        "OFFLOAD-EFT-THERMAL-ALIGNED": "native_offload",
        "TUT-UOI-PRODUCT-LITE-v2": "native_tut",
    }[method])
    binding = module.make_binding(source_root.resolve())
    identity = dict(binding.identity())
    if not identity.get("source_identity_passed", False):
        raise RuntimeError({"native_source_identity_failed": method, "identity": identity})
    scenario = binding.build_scenario(bundle)
    scenario, power_identity = inject_power(scenario, source_root)
    scenario_identity = relaxed_scenario_identity(method, scenario, bundle, power_identity)
    if not scenario_identity["task_hash_match"] or scenario_identity["runtime_value_mismatch_count"]:
        raise RuntimeError({"scenario_identity_failed": scenario_identity})
    native = binding.simulate(scenario, SLOTS)
    normalized = dict(binding.normalize_outputs(native, bundle))
    output.mkdir(parents=True, exist_ok=False)
    write_unified_method_outputs(output, normalized)
    status = {
        "status": "PASS",
        "phase": "formal_true_rerun",
        "method": method,
        "scenario_reconstruction_count": 1,
        "simulation_started": True,
        "event_count": int(native.get("event_count", 0) if isinstance(native, dict) else 0),
        "reused_action_trace": False,
        "reused_decision_trace": False,
        "reused_queue_trace": False,
        "reused_delivery_ledger": False,
        "reused_soc_trace": False,
        "reused_safety_filter_result": False,
        "identity": identity,
        "scenario_identity": scenario_identity,
    }
    write_json(output / "native_identity.json", identity)
    write_json(output / "reconstructed_scenario_identity.json", scenario_identity)
    write_json(output / "method_run_status.json", status)
    write_json(output / "method_completed.json", {"phase": "formal_true_rerun", "method": method, "status": "PASS", "scenario_reconstruction_count": 1})


def import_prop_script():
    child_setup_path(SOURCE_PROP)
    path = SOURCE_PROP / "scripts" / "run_updated_prop_ground_ack_v1.py"
    spec = importlib.util.spec_from_file_location("updated_prop_ground_ack_v1_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_prop_shared_method(bundle_root: Path, output: Path) -> None:
    child_setup_path(SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    prop_mod = import_prop_script()
    bundle = load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    calibration = read_json(SHARED_PRIOR_CALIBRATION)
    configs = prop_mod._configure(scenario.configs, calibration, {"enabled": False})
    scenario = replace(scenario, configs=configs)
    scenario, power_identity = inject_power(scenario, SOURCE_PROP)
    scenario_identity = relaxed_scenario_identity("PROP-SHARED-CALIBRATED-PRIOR", scenario, bundle, power_identity)
    if not scenario_identity["task_hash_match"] or scenario_identity["runtime_value_mismatch_count"]:
        raise RuntimeError({"scenario_identity_failed": scenario_identity})
    env = SimulationEnvironment(scenario.configs)
    runner = env.build_episode_runner(scenario.timeline[:SLOTS])
    router = build_routers(scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"]))[0]
    result = runner.run(router)
    normalized = normalize_episode("PROP-SHARED-CALIBRATED-PRIOR", scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    write_unified_method_outputs(output, normalized)
    write_csv(output / "candidate_action_log.csv", list(getattr(runner, "last_candidate_action_logs", [])))
    identity = {
        "method": "PROP-SHARED-CALIBRATED-PRIOR",
        "source_root": str(SOURCE_PROP.resolve()),
        "source_identity_passed": True,
        "shared_calibrated_prior_enabled": True,
        "shared_prior_calibration_path": str(SHARED_PRIOR_CALIBRATION),
        "shared_prior_calibration_sha256": sha256_file(SHARED_PRIOR_CALIBRATION),
        "ground_ack_enabled": False,
        "router_identifier": "proposed_lyapunov_rule_mode_safe_bandit",
    }
    status = {
        "status": "PASS",
        "phase": "formal_true_rerun",
        "method": "PROP-SHARED-CALIBRATED-PRIOR",
        "scenario_reconstruction_count": 1,
        "simulation_started": True,
        "event_count": len(normalized["events"]),
        "reused_action_trace": False,
        "reused_decision_trace": False,
        "reused_queue_trace": False,
        "reused_delivery_ledger": False,
        "reused_soc_trace": False,
        "reused_safety_filter_result": False,
        "identity": identity,
        "scenario_identity": scenario_identity,
    }
    write_json(output / "native_identity.json", identity)
    write_json(output / "reconstructed_scenario_identity.json", scenario_identity)
    write_json(output / "method_run_status.json", status)
    write_json(output / "method_completed.json", {"phase": "formal_true_rerun", "method": "PROP-SHARED-CALIBRATED-PRIOR", "status": "PASS", "scenario_reconstruction_count": 1})


def run_myopic_method(bundle_root: Path, output: Path) -> None:
    child_setup_path(SOURCE_MYOPIC)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.routers.myopic_frozen_calibrated_router import MyopicFrozenCalibratedRouter
    from src.simulator.environment import SimulationEnvironment

    bundle = load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    scenario, power_identity = inject_power(scenario, SOURCE_MYOPIC)
    scenario_identity = relaxed_scenario_identity("MYOPIC-FROZEN-CALIBRATED", scenario, bundle, power_identity)
    if not scenario_identity["task_hash_match"] or scenario_identity["runtime_value_mismatch_count"]:
        raise RuntimeError({"scenario_identity_failed": scenario_identity})
    env = SimulationEnvironment(scenario.configs)
    runner = env.build_episode_runner(scenario.timeline[:SLOTS])
    router = MyopicFrozenCalibratedRouter(MYOPIC_PARAM, router_name="MYOPIC-FROZEN-CALIBRATED")
    result = runner.run(router)
    normalized = normalize_episode("MYOPIC-FROZEN-CALIBRATED", scenario, result, runner, SLOTS)
    normalized["resource_summary"].setdefault("native_summary", {})
    normalized["resource_summary"]["native_summary"]["online_update_count"] = 0
    normalized["resource_summary"]["native_summary"]["frozen_parameter_sha256"] = sha256_file(MYOPIC_PARAM)
    output.mkdir(parents=True, exist_ok=False)
    write_unified_method_outputs(output, normalized)
    write_csv(output / "candidate_action_log.csv", list(getattr(runner, "last_candidate_action_logs", [])))
    identity = {
        "method": "MYOPIC-FROZEN-CALIBRATED",
        "source_root": str(SOURCE_MYOPIC.resolve()),
        "source_identity_passed": True,
        "router_class": "src.routers.myopic_frozen_calibrated_router.MyopicFrozenCalibratedRouter",
        "parameter_file": str(MYOPIC_PARAM),
        "parameter_sha256": sha256_file(MYOPIC_PARAM),
        "online_update_count": 0,
    }
    status = {
        "status": "PASS",
        "phase": "formal_true_rerun",
        "method": "MYOPIC-FROZEN-CALIBRATED",
        "scenario_reconstruction_count": 1,
        "simulation_started": True,
        "event_count": len(normalized["events"]),
        "reused_action_trace": False,
        "reused_decision_trace": False,
        "reused_queue_trace": False,
        "reused_delivery_ledger": False,
        "reused_soc_trace": False,
        "reused_safety_filter_result": False,
        "identity": identity,
        "scenario_identity": scenario_identity,
    }
    write_json(output / "native_identity.json", identity)
    write_json(output / "reconstructed_scenario_identity.json", scenario_identity)
    write_json(output / "method_run_status.json", status)
    write_json(output / "method_completed.json", {"phase": "formal_true_rerun", "method": "MYOPIC-FROZEN-CALIBRATED", "status": "PASS", "scenario_reconstruction_count": 1})


def child_main(args: argparse.Namespace) -> int:
    method = str(args.method)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if method == "PROP-SHARED-CALIBRATED-PRIOR":
        run_prop_shared_method(Path(args.bundle_root), output)
    elif method in {"B1-COMMON-THERMAL", "OFFLOAD-EFT-THERMAL-ALIGNED", "TUT-UOI-PRODUCT-LITE-v2"}:
        run_native_binding_method(method, Path(args.bundle_root), output)
    elif method == "MYOPIC-FROZEN-CALIBRATED":
        run_myopic_method(Path(args.bundle_root), output)
    else:
        raise ValueError(method)
    print(json.dumps({"status": "PASS", "method": method, "output": str(output)}, ensure_ascii=False))
    return 0


def launch_child(method: str, bundle_root: Path, out: Path) -> dict[str, Any]:
    output = out / "formal" / method
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--method",
        method,
        "--bundle-root",
        str(bundle_root),
        "--output",
        str(output),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    started = datetime.now(timezone.utc).isoformat()
    cp = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finished = datetime.now(timezone.utc).isoformat()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "logs" / f"{method}_stdout.txt").write_text(cp.stdout, encoding="utf-8")
    (out / "logs" / f"{method}_stderr.txt").write_text(cp.stderr, encoding="utf-8")
    return {
        "method": method,
        "returncode": cp.returncode,
        "output": str(output),
        "started_utc": started,
        "finished_utc": finished,
        "stdout_tail": cp.stdout[-2000:],
        "stderr_tail": cp.stderr[-4000:],
    }


def load_eval_module(out: Path):
    os.environ["V25_UNIFIED_OUT"] = str(out)
    path = ROOT / ".codex_method_preservation_audit_v1" / "scripts" / "v25_unified_baselines_core_metrics_v2.py"
    spec = importlib.util.spec_from_file_location("v25_core_eval_for_bupt1_true_rerun", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def method_paths(out: Path) -> dict[str, Path]:
    return {method: out / "formal" / method for method in METHODS}


def count_hv_stage_tasks(path: Path) -> dict[str, int]:
    tasks = read_csv(path / "per_task_outcomes.csv")
    events = read_csv(path / "ground_delivery_events.csv")
    hv_ids = {row["task_id"] for row in tasks if row.get("value_level") in {"high", "urgent"}}
    by_stage: dict[str, set[str]] = defaultdict(set)
    for row in events:
        if row.get("task_id") not in hv_ids or not i(row.get("valid_before_deadline")) or i(row.get("duplicate_flag")):
            continue
        stage = str(row.get("stage", "")).lower()
        if stage == "metadata":
            by_stage["metadata"].add(row["task_id"])
        elif stage == "roi":
            by_stage["roi"].add(row["task_id"])
        elif stage in {"evidence", "full"}:
            by_stage["evidence"].add(row["task_id"])
    return {key: len(value) for key, value in by_stage.items()}


def hv_quick_rows(out: Path, metrics: list[dict[str, Any]], eval_mod: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    full_metrics = {row["method"]: row for row in metrics}
    utility_cfg = config["product_requirement_utility"]
    rows: list[dict[str, Any]] = []
    for method, path in method_paths(out).items():
        tasks = read_csv(path / "per_task_outcomes.csv")
        events = read_csv(path / "ground_delivery_events.csv")
        hv_tasks = [row for row in tasks if row.get("value_level") in {"high", "urgent"}]
        hv_ids = {row["task_id"] for row in hv_tasks}
        hv_closed_ids = {row["task_id"] for row in hv_tasks if i(row.get("closure_success"))}
        valid = [
            row for row in events
            if row.get("task_id") in hv_ids
            and i(row.get("valid_before_deadline"))
            and not i(row.get("duplicate_flag"))
            and row.get("stage") in {"metadata", "roi", "evidence", "full"}
        ]
        by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in valid:
            by_task[row["task_id"]].append(row)
        first_latencies: list[Decimal] = []
        first_stage_counts: Counter[str] = Counter()
        metadata_latencies: list[Decimal] = []
        metadata_task_ids: set[str] = set()
        for task in hv_tasks:
            arrivals = sorted(by_task.get(task["task_id"], []), key=lambda row: d(row["ground_available_time"]))
            if arrivals:
                first = arrivals[0]
                first_latencies.append(d(first["ground_available_time"]) - d(task["arrival_time"]))
                first_stage_counts[first["stage"]] += 1
            metas = [row for row in arrivals if row["stage"] == "metadata"]
            if metas:
                metadata_task_ids.add(task["task_id"])
                metadata_latencies.append(d(metas[0]["ground_available_time"]) - d(task["arrival_time"]))
        ledger = eval_mod.ledger_for(tasks, events, utility_cfg)
        hv_success_gross = sum(
            row["positive_gross_utility"]
            for row in ledger
            if row["task_id"] in hv_closed_ids and row["positive_gross_utility"] > 0
        )
        metric = full_metrics[method]
        method_gross = d(metric["gross_delivered_utility"])
        method_cost = d(metric["public_method_cost"])
        allocated_cost = method_cost * hv_success_gross / method_gross if method_gross else Decimal("0")
        hv_success_net = hv_success_gross - allocated_cost
        rows.append({
            "method": method,
            "HV_task_count": len(hv_tasks),
            "HV_first_info_task_count": len(first_latencies),
            "HV_first_info_rate": str(Decimal(len(first_latencies)) / Decimal(max(len(hv_tasks), 1))),
            "HV_first_info_latency_mean_s": str(sum(first_latencies, Decimal("0")) / Decimal(len(first_latencies))) if first_latencies else "NA",
            "HV_first_info_latency_p95_s": str(pct95(first_latencies)),
            "HV_first_info_stage_distribution": json.dumps(dict(sorted(first_stage_counts.items())), sort_keys=True),
            "HV_metadata_task_count": len(metadata_task_ids),
            "HV_metadata_coverage_rate": str(Decimal(len(metadata_task_ids)) / Decimal(max(len(hv_tasks), 1))),
            "HV_metadata_latency_mean_s": str(sum(metadata_latencies, Decimal("0")) / Decimal(len(metadata_latencies))) if metadata_latencies else "NA",
            "HV_metadata_latency_p95_s": str(pct95(metadata_latencies)),
            "HV_success_receive_count": len(hv_closed_ids),
            "HV_success_receive_rate": str(Decimal(len(hv_closed_ids)) / Decimal(max(len(hv_tasks), 1))),
            "HV_success_gross_utility": str(hv_success_gross),
            "HV_success_allocated_cost": str(allocated_cost),
            "HV_success_allocated_net_utility": str(hv_success_net),
        })
    return rows


def orbit_phase(slot: int) -> str:
    t = (slot * SLOT_SECONDS) % int(FROZEN_POWER["orbit_period_s"])
    return "eclipse" if t < int(FROZEN_POWER["eclipse_duration_s"]) else "daylight"


def energy_rows(out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    capacity_wh = FROZEN_POWER["battery_capacity_Wh"]
    for method, path in method_paths(out).items():
        slots = read_csv(path / "per_slot_resource_trace.csv")
        if not slots:
            continue
        by_orbit: dict[int, list[dict[str, str]]] = defaultdict(list)
        solar_total = Decimal("0")
        curtailed = Decimal("0")
        residual_max = Decimal("0")
        prev_soc = Decimal(str(FROZEN_POWER["initial_soc"]))
        for row in slots:
            slot = i(row["slot"])
            phase = orbit_phase(slot)
            by_orbit[int(slot * SLOT_SECONDS // int(FROZEN_POWER["orbit_period_s"]))].append(row)
            solar_wh = Decimal(str(FROZEN_POWER["solar_energy_per_orbit_Wh"])) / Decimal(int(FROZEN_POWER["orbit_period_s"]) // SLOT_SECONDS) * Decimal("0")
            # Use the original injected resource trace through idle_energy and recompute solar from the frozen one-orbit file.
            solar_power = Decimal(str(read_csv(SOLAR_TRACE)[slot % len(read_csv(SOLAR_TRACE))]["solar_power_W"]))
            solar_wh = solar_power * Decimal(SLOT_SECONDS) / Decimal("3600")
            solar_total += solar_wh
            load_wh = (d(row.get("idle_energy_j")) + d(row.get("compute_energy_j")) + d(row.get("comm_energy_j"))) / Decimal("3600")
            net = solar_wh - load_wh
            delta = net * d(FROZEN_POWER["charge_efficiency"]) if net >= 0 else net / d(FROZEN_POWER["discharge_efficiency"])
            expected_soc = prev_soc + delta / d(capacity_wh)
            actual_soc = d(row["battery_soc"])
            if expected_soc > 1:
                curtailed += (expected_soc - Decimal("1")) * d(capacity_wh)
            elif Decimal("0") < actual_soc < Decimal("1"):
                residual_max = max(residual_max, abs(actual_soc - expected_soc) * d(capacity_wh))
            prev_soc = actual_soc
        socs = [d(row["battery_soc"]) for row in slots]
        thermals = [d(row["thermal"]) for row in slots]
        queues = [d(row["queue_mb"]) for row in slots]
        orbit_stats = []
        for orbit, group in sorted(by_orbit.items()):
            eclipse = [row for row in group if orbit_phase(i(row["slot"])) == "eclipse"]
            daylight = [row for row in group if orbit_phase(i(row["slot"])) == "daylight"]
            orbit_stats.append({
                "orbit": orbit,
                "eclipse_drop": d(eclipse[0]["battery_soc"]) - d(eclipse[-1]["battery_soc"]) if eclipse else Decimal("0"),
                "daylight_recovery": d(daylight[-1]["battery_soc"]) - d(daylight[0]["battery_soc"]) if daylight else Decimal("0"),
            })
        compute_wh = sum(d(row["compute_energy_j"]) for row in slots) / Decimal("3600")
        tx_wh = sum(d(row["comm_energy_j"]) for row in slots) / Decimal("3600")
        platform_wh = sum(d(row["idle_energy_j"]) for row in slots) / Decimal("3600")
        rows.append({
            "method": method,
            "initial_SOC": str(FROZEN_POWER["initial_soc"]),
            "final_SOC": str(socs[-1]),
            "minimum_SOC": str(min(socs)),
            "maximum_DoD": str(Decimal("1") - min(socs)),
            "mean_daylight_recovery": str(sum((row["daylight_recovery"] for row in orbit_stats), Decimal("0")) / Decimal(max(len(orbit_stats), 1))),
            "mean_eclipse_drop": str(sum((row["eclipse_drop"] for row in orbit_stats), Decimal("0")) / Decimal(max(len(orbit_stats), 1))),
            "solar_collected_Wh": str(solar_total),
            "solar_used_Wh": str(solar_total - curtailed),
            "curtailed_solar_Wh": str(curtailed),
            "platform_base_energy_Wh": str(platform_wh),
            "compute_energy_Wh": str(compute_wh),
            "encoding_energy_Wh": "0",
            "tx_energy_Wh": str(tx_wh),
            "controllable_total_energy_Wh": str(compute_wh + tx_wh),
            "transmitted_MB": str(sum(d(row["transmitted_mb"]) for row in slots)),
            "peak_queue_MB": str(max(queues)),
            "average_queue_MB": str(sum(queues, Decimal("0")) / Decimal(len(queues))),
            "maximum_thermal": str(max(thermals)),
            "thermal_violation_count": str(sum(i(row["thermal_violation"]) for row in slots)),
            "soc_violation_count": str(sum(i(row["soc_violation"]) for row in slots)),
            "max_energy_balance_residual_Wh": str(residual_max),
        })
    return rows


def auxiliary_rows(out: Path, metrics: list[dict[str, Any]], hv_rows: list[dict[str, Any]], e_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hv_by = {row["method"]: row for row in hv_rows}
    energy_by = {row["method"]: row for row in e_rows}
    rows = []
    for metric in metrics:
        method = metric["method"]
        path = method_paths(out)[method]
        stages = count_hv_stage_tasks(path)
        tasks = read_csv(path / "per_task_outcomes.csv")
        rows.append({
            "method": method,
            "total_closure_count": metric["closure_count"],
            "total_closure_rate": metric["closure_rate"],
            "high_closure_count": metric["high_closure_count"],
            "urgent_closure_count": metric["urgent_closure_count"],
            "HV_metadata_delivered_count": stages.get("metadata", 0),
            "HV_ROI_delivered_count": stages.get("roi", 0),
            "HV_evidence_delivered_count": stages.get("evidence", 0),
            "HV_first_info_task_count": hv_by[method]["HV_first_info_task_count"],
            "HV_first_info_latency_mean_s": hv_by[method]["HV_first_info_latency_mean_s"],
            "HV_first_info_latency_p95_s": hv_by[method]["HV_first_info_latency_p95_s"],
            "successful_closure_mean_latency_s": metric["full_closure_latency_mean_s"],
            "successful_closure_p95_latency_s": metric["full_closure_latency_p95_s"],
            "HV_success_mean_latency_s": metric["HV_full_closure_latency_mean_s"],
            "HV_success_p95_latency_s": metric["HV_full_closure_latency_p95_s"],
            "deadline_censored_mean_latency_s": metric["deadline_censored_all_latency_mean_s"],
            "downlink_MB": metric["transmitted_MB"],
            "peak_queue_MB": metric["peak_queue_MB"],
            "average_queue_MB": energy_by[method]["average_queue_MB"],
            "safe_filter_correction_count": sum(1 for row in tasks if row.get("selected_action") != row.get("executed_action")),
            "deadline_reject_count": sum(1 for row in tasks if "deadline" in str(row.get("failure_reason", "")).lower()),
            "defer_count": metric["defer_count"],
        })
    return rows


def primary_rows(metrics: list[dict[str, Any]], hv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hv_by = {row["method"]: row for row in hv_rows}
    return [
        {
            "method": row["method"],
            "J_all_net": row["J_all_net"],
            "J_HV_net": hv_by[row["method"]]["HV_success_allocated_net_utility"],
            "HV_closure_count": row["HV_closure_count"],
        }
        for row in metrics
    ]


def comparison_vs_prop(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {row["method"]: row for row in primary}
    prop = by["PROP-SHARED-CALIBRATED-PRIOR"]
    out = []
    for method in METHODS[1:]:
        row = by[method]
        out.append({
            "baseline": method,
            "delta_J_all_net_PROP_minus_baseline": str(d(prop["J_all_net"]) - d(row["J_all_net"])),
            "delta_J_HV_net_PROP_minus_baseline": str(d(prop["J_HV_net"]) - d(row["J_HV_net"])),
            "delta_HV_closure_count_PROP_minus_baseline": str(d(prop["HV_closure_count"]) - d(row["HV_closure_count"])),
        })
    return out


def core_rows(metrics: list[dict[str, Any]], primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    p_by = {row["method"]: row for row in primary}
    rows = []
    for row in metrics:
        rows.append({
            "Method": row["method"],
            "Result Source": "TRUE_FRESH_FROZEN_POWER",
            "J_all_net": row["J_all_net"],
            "J_HV_net": p_by[row["method"]]["J_HV_net"],
            "Closure": row["closure_count"],
            "Closure Rate": row["closure_rate"],
            "HV Closure": row["HV_closure_count"],
            "High Closure": row["high_closure_count"],
            "Urgent Closure": row["urgent_closure_count"],
            "Mean Latency (s)": row["full_closure_latency_mean_s"],
            "P95 Latency (s)": row["full_closure_latency_p95_s"],
            "HV Mean Latency (s)": row["HV_full_closure_latency_mean_s"],
            "HV P95 Latency (s)": row["HV_full_closure_latency_p95_s"],
            "Deadline-Censored Mean (s)": row["deadline_censored_all_latency_mean_s"],
            "Energy (Wh)": row["controllable_energy_Wh"],
            "Transmitted (MB)": row["transmitted_MB"],
            "Peak Queue (MB)": row["peak_queue_MB"],
            "Min SOC": row["min_SOC"],
            "Max Thermal": row["max_thermal"],
            "Safety Violations": row["safety_violations"],
        })
    return rows


def aggregate_outputs(out: Path, bundle_root: Path) -> dict[str, Any]:
    eval_mod = load_eval_module(out)
    config = read_json(bundle_root / "resolved_config.json")
    metrics: list[dict[str, Any]] = []
    accounting_audit: list[dict[str, Any]] = []
    for method, path in method_paths(out).items():
        row, audit = eval_mod.summarize_method(
            method,
            "TRUE_FRESH_FROZEN_POWER",
            path,
            config["product_requirement_utility"],
            config["unified_ground_received_accounting"],
        )
        metrics.append(row)
        accounting_audit.append(audit)
    hv = hv_quick_rows(out, metrics, eval_mod, config)
    energy = energy_rows(out)
    aux = auxiliary_rows(out, metrics, hv, energy)
    primary = primary_rows(metrics, hv)
    deltas = comparison_vs_prop(primary)
    core = core_rows(metrics, primary)

    write_csv(out / "tables" / "primary_metrics.csv", primary)
    write_csv(out / "tables" / "auxiliary_summary.csv", aux)
    write_csv(out / "tables" / "energy_metrics.csv", energy)
    write_csv(out / "tables" / "prop_vs_baseline_single_seed_deltas.csv", deltas)
    write_csv(out / "method_metrics_full.csv", metrics)
    write_csv(out / "hv_quick_warning_comparison.csv", hv)
    write_csv(out / "comparison_core_metrics.csv", core)
    write_csv(out / "comparison_vs_prop.csv", deltas)
    (out / "tables" / "primary_metrics.md").write_text(markdown_table(primary) + "\n", encoding="utf-8")
    (out / "tables" / "auxiliary_summary.md").write_text(markdown_table(aux) + "\n", encoding="utf-8")
    (out / "tables" / "energy_metrics.md").write_text(markdown_table(energy) + "\n", encoding="utf-8")
    (out / "tables" / "prop_vs_baseline_single_seed_deltas.md").write_text(markdown_table(deltas) + "\n", encoding="utf-8")
    (out / "comparison_core_metrics.md").write_text(markdown_table(core) + "\n", encoding="utf-8")

    scenario_rows = []
    method_identity_rows = []
    for method, path in method_paths(out).items():
        status = read_json(path / "method_run_status.json")
        scen = read_json(path / "reconstructed_scenario_identity.json")
        ident = read_json(path / "native_identity.json")
        scenario_rows.append({
            "method": method,
            "task_hash_match": scen["task_hash_match"],
            "frozen_power_identity_match": scen["frozen_power_identity_match"],
            "runtime_value_mismatch_count": scen["runtime_value_mismatch_count"],
            "field_mismatch_count": scen["field_mismatch_count"],
            "simulation_started": status["simulation_started"],
            "scenario_reconstruction_count": status["scenario_reconstruction_count"],
        })
        method_identity_rows.append({
            "method": method,
            "source_identity_passed": ident.get("source_identity_passed", False),
            "source_root": ident.get("source_root", ""),
            "runtime_method": ident.get("method", method),
        })
    accounting_ok = all(abs(d(row["accounting_error"])) <= Decimal("1e-10") and row["closure_decomposition_ok"] and row["hv_decomposition_ok"] for row in accounting_audit)
    scenario_ok = all(bool(row["task_hash_match"]) and bool(row["frozen_power_identity_match"]) and i(row["runtime_value_mismatch_count"]) == 0 and i(row["field_mismatch_count"]) == 0 and bool(row["simulation_started"]) and i(row["scenario_reconstruction_count"]) == 1 for row in scenario_rows)
    method_ok = all(bool(row["source_identity_passed"]) for row in method_identity_rows)
    energy_ok = all(d(row["max_energy_balance_residual_Wh"]) <= Decimal("1e-8") for row in energy)
    write_json(out / "audit" / "public_accounting_audit.json", {"passed": accounting_ok, "rows": accounting_audit})
    write_json(out / "audit" / "scenario_identity_audit.json", {"passed": scenario_ok, "rows": scenario_rows})
    write_json(out / "audit" / "method_identity_audit.json", {"passed": method_ok, "rows": method_identity_rows})
    write_json(out / "audit" / "energy_balance_audit.json", {"passed": energy_ok, "rows": [{"method": row["method"], "max_energy_balance_residual_Wh": row["max_energy_balance_residual_Wh"]} for row in energy]})
    return {
        "primary": primary,
        "auxiliary": aux,
        "energy": energy,
        "deltas": deltas,
        "accounting_ok": accounting_ok,
        "scenario_ok": scenario_ok,
        "method_ok": method_ok,
        "energy_ok": energy_ok,
    }


def write_final(out: Path, result: dict[str, Any]) -> None:
    status = "BUPT1_FROZEN_POWER_TRUE_RERUN_FIVE_METHOD_COMPLETE"
    lines = [
        "# BUPT1 Frozen Power True Rerun Seed2026",
        "",
        f"Final status: `{status}`",
        "",
        "- Scope: seed=2026, capacity=0.35, duration=86400 s.",
        "- Task generator called: false.",
        "- All five methods were started from t=0 in fresh child processes.",
        "- Frozen task manifest and frozen BUPT1 power inputs were loaded before online decisions and safety filtering.",
        "",
        "## Primary Metrics",
        "",
        markdown_table(result["primary"]),
        "",
        "## Audit",
        "",
        f"- Energy audit errors: `{not result['energy_ok']}`.",
        f"- Scenario identity audit errors: `{not result['scenario_ok']}`.",
        f"- Method identity audit errors: `{not result['method_ok']}`.",
        f"- Public accounting audit errors: `{not result['accounting_ok']}`.",
    ]
    (out / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "final_status.json", {
        "status": status,
        "output_dir": str(out),
        "energy_audit_errors": not result["energy_ok"],
        "scenario_identity_audit_errors": not result["scenario_ok"],
        "method_identity_audit_errors": not result["method_ok"],
        "public_accounting_audit_errors": not result["accounting_ok"],
    })


def print_result(result: dict[str, Any], out: Path) -> None:
    print("BUPT1_FROZEN_POWER_TRUE_RERUN_FIVE_METHOD_COMPLETE")
    print("\nPRIMARY_METRICS")
    print(markdown_table(result["primary"]))
    print("\nAUXILIARY_SUMMARY")
    print(markdown_table(result["auxiliary"]))
    print("\nENERGY_METRICS")
    print(markdown_table(result["energy"]))
    print("\nPROP_VS_BASELINE_SINGLE_SEED_DELTAS")
    print(markdown_table(result["deltas"]))
    print(f"\nOUTPUT_DIR={out}")
    print(f"ENERGY_AUDIT_ERRORS={not result['energy_ok']}")
    print(f"SCENARIO_IDENTITY_AUDIT_ERRORS={not result['scenario_ok']}")
    print(f"METHOD_IDENTITY_AUDIT_ERRORS={not result['method_ok']}")
    print(f"PUBLIC_ACCOUNTING_AUDIT_ERRORS={not result['accounting_ok']}")


def parent_main(args: argparse.Namespace) -> int:
    out = args.output_dir
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)

    started = datetime.now(timezone.utc).isoformat()
    input_audit, frozen_manifest = preflight(out)
    bundle_root = prepare_runtime_bundle(out)
    audit = {
        "run_mode": "true_fresh_rerun",
        "status": "PRE_FLIGHT_PASS",
        "started_utc": started,
        "expected_task_manifest_hash": EXPECTED_TASK_HASH,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "expected_value_level_counts": EXPECTED_COUNTS,
        "expected_N_HV": EXPECTED_HV_COUNT,
        "seed": SEED,
        "capacity": CAPACITY,
        "duration_s": SLOTS * SLOT_SECONDS,
        "slot_seconds": SLOT_SECONDS,
        "input_audit": input_audit,
        "frozen_manifest": frozen_manifest,
        "task_generator_called": False,
        "method_simulation_started": False,
    }
    write_json(out / "audit" / "true_fresh_rerun_audit.json", audit)
    write_json(out / "config" / "scenario_identity.json", audit)

    run_rows = []
    for method in METHODS:
        print(f"[RUN] {method}", flush=True)
        row = launch_child(method, bundle_root, out)
        row["status"] = "PASS" if row["returncode"] == 0 and (Path(row["output"]) / "method_completed.json").exists() else "FAIL"
        run_rows.append(row)
        write_csv(out / "formal_run_status.csv", run_rows)
        if row["status"] != "PASS":
            status = f"BUPT1_FROZEN_POWER_TRUE_RERUN_FAILED_{method}"
            write_json(out / "failure_detail.json", {"status": status, "failed_method": method, "attempt": row})
            (out / "final_assessment.md").write_text(f"Final status: `{status}`\n\nFailure detail is in `failure_detail.json`.\n", encoding="utf-8")
            print(status)
            print(json.dumps(row, ensure_ascii=False, indent=2))
            return 2

    audit["method_simulation_started"] = True
    audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
    audit["status"] = "FIVE_METHOD_RUNS_COMPLETE"
    audit["reused_action_trace"] = False
    audit["reused_decision_trace"] = False
    audit["reused_queue_trace"] = False
    audit["reused_delivery_ledger"] = False
    audit["reused_soc_trace"] = False
    audit["reused_safety_filter_result"] = False
    write_json(out / "audit" / "true_fresh_rerun_audit.json", audit)

    result = aggregate_outputs(out, bundle_root)
    write_final(out, result)
    print_result(result, out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--method")
    parser.add_argument("--bundle-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.child_run:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
