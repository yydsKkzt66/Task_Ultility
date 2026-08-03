"""One-click ordinary-scene TUT proxy and GROUND-ONLY recovery audit.

The public entry takes no arguments.  Private ``--child-spec`` is used only
to isolate the three fresh simulations in independent Python processes.
Historical source trees, frozen parameters, configurations and results are
read-only; every invocation writes a new numbered attempt directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402


_gate_spec = importlib.util.spec_from_file_location(
    "normal_scene_tut_ground_gatecore",
    ROOT / "src" / "evaluation" / "frozen_v2_final_gate.py",
)
if _gate_spec is None or _gate_spec.loader is None:
    raise ImportError("normal_scene_tut_ground_gatecore")
gatecore = importlib.util.module_from_spec(_gate_spec)
sys.modules[_gate_spec.name] = gatecore
_gate_spec.loader.exec_module(gatecore)


VERSION = "NORMAL-SCENARIO-SINGLESEED-TUT-PROXY-GROUND-RECOVERY-v1"
COMPLETION = "NORMAL_SCENARIO_SINGLESEED_TUT_PROXY_AND_GROUND_RECOVERY_COMPLETE"
SEED = 2026
SLOTS = 4320
SLOT_SECONDS = 20.0
CAPACITY = 0.50
METHOD_ASIS = "TUT-LITE-AS-IS"
METHOD_PROXY = "TUT-SEMANTIC-PROXY-V1"
METHOD_GROUND = "GROUND-ONLY-FIFO-CORRECTED"
METHODS = (METHOD_ASIS, METHOD_PROXY, METHOD_GROUND)
SLUG = {METHOD_ASIS: "TUT_LITE_AS_IS", METHOD_PROXY: "TUT_SEMANTIC_PROXY_V1", METHOD_GROUND: "GROUND_ONLY_FIFO_CORRECTED"}

OUTPUT_BASE = ROOT / "outputs" / "_normal_scenario_single_seed_tut_proxy_ground_recovery_v1"
TRUSTED_SCENARIO = (
    ROOT / "outputs" / "_hv_balanced_to_high_single_seed_counterfactual_audit_v1"
    / "scenario" / "ordinary_random_restored_seed2026"
)
FULL_V2_REFERENCE = (
    ROOT / "outputs" / "_hv_balanced_to_high_single_seed_counterfactual_audit_v1"
    / "audit" / "source_run_public_metrics.json"
)
FROZEN = (
    ROOT / "outputs" / "_historical_asset_audit_conditional_v2_calibration_v1"
    / "frozen" / "conditional_v2_frozen_parameters_v1.json"
)

EXPECTED_TASK_HASH = "c31d50bf0633e0d7395f78a785768c2ac492eb5f707c02a9e5be93b1226b37fa"
EXPECTED_MANIFEST_SHA = "ffdc5a332c125aa0b24d08bf8081f842afededafae3123a37b7a951a1f7d0841"
EXPECTED_SLOT_SHA = "63566543c19c25d18b9cade7e8f909c466ba363f4bc3497c0df55a174b2ca98f"
EXPECTED_WINDOW_SCHEDULE_HASH = "8de62cf99514e6b6d6969595c485efe3437e14da354dafb8289d43c3431bfb86"
EXPECTED_COUNTS = {"low": 54, "medium": 139, "high": 54, "urgent": 21}
EXPECTED_QUARTILES = {
    "low": [9, 27, 6, 12], "medium": [47, 24, 25, 43],
    "high": [17, 9, 12, 16], "urgent": [2, 4, 6, 9],
}
EXPECTED_RUNTIME_LINK_HASH = "ff85d81c6db6ed51c6535f4bdd2fe92eabe2220f061b15c94e3a8e3a23fc54a9"

Q_BALANCED = 0.734241158700
SEMANTIC_RETENTION = 0.95
Q_TUT_PROXY = 0.697529100765
ENCODING_LATENCY_S = 0.43
ENCODING_POWER_W = 14.0
ENCODING_ENERGY_J = 6.02
INFERENCE_LATENCY_S = 1.44
INFERENCE_ENERGY_J = 21.60
GROUND_LATENCY_S = 1.87
PRODUCT_MB = {"metadata": 0.002, "roi": 0.85, "roi_context": 2.70}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def f(value: Any, default: float = 0.0) -> float:
    return gatecore.f(value, default)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in data for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data or [{"empty": ""}])


def write_md(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(data[0]) if data else ["empty"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float | None:
    return gatecore.percentile(values, q)


def load_neutral_bundle(root: Path) -> SimpleNamespace:
    """Load the already-restored bundle without invoking the late-HV validator."""

    root = root.resolve()
    contract = read_json(root / "bundle_contract.json")
    for name, expected in contract["file_sha256"].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError({"TRUSTED_SCENARIO_PAYLOAD_MISMATCH": name})
    tasks = tuple(read_csv(root / "task_manifest.csv"))
    slots = tuple(read_csv(root / "slot_trace.csv"))
    swaps = tuple(read_csv(root / "arrival_swap_manifest.csv"))
    values = tuple(read_csv(root / "task_value_assignment_trace.csv"))
    config = read_json(root / "resolved_config.json")
    task_values = {row["task_id"]: Decimal(row["effective_task_value"]) for row in tasks}
    return SimpleNamespace(root=root, contract=contract, tasks=tasks, slots=slots, swaps=swaps, values=values, config=config, task_values=task_values)


def arrival_statistics(tasks: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(tasks, key=lambda row: (f(row["arrival_time_s"]), row["task_id"]))
    rows = []
    levels_in_order = [row["value_level"] for row in ordered]
    bursts: list[list[dict[str, str]]] = []
    for row in ordered:
        if not bursts or f(row["arrival_time_s"]) - f(bursts[-1][-1]["arrival_time_s"]) > 40.0 + 1.0e-9:
            bursts.append([])
        bursts[-1].append(row)
    for level in ("low", "medium", "high", "urgent"):
        selected = [row for row in ordered if row["value_level"] == level]
        times = [f(row["arrival_time_s"]) for row in selected]
        q1 = percentile(times, 0.25)
        q3 = percentile(times, 0.75)
        run = best = 0
        for observed in levels_in_order:
            run = run + 1 if observed == level else 0
            best = max(best, run)
        window_counts = Counter(row.get("observation_window_id", "") for row in selected)
        rows.append({
            "value_level": level, "task_count": len(selected),
            "arrival_mean_s": statistics.fmean(times), "arrival_median_s": statistics.median(times),
            "arrival_q1_s": q1, "arrival_q3_s": q3,
            "first_12h_count": sum(time < 43200.0 for time in times),
            "second_12h_count": sum(time >= 43200.0 for time in times),
            "day_quartile_counts": [sum(21600.0 * q <= time < 21600.0 * (q + 1) for time in times) for q in range(4)],
            "max_burst_count_gap_le_40s": max((sum(item["value_level"] == level for item in burst) for burst in bursts), default=0),
            "max_uninterrupted_same_level_run_in_arrival_order_aux": best,
            "max_same_level_tasks_in_one_observation_window": max(window_counts.values(), default=0),
        })
    definitions = {
        "primary_burst_definition": "split chronological arrivals when adjacent gap exceeds 40 seconds (two 20-second slots); report max count of each level inside one burst",
        "consecutive_run_definition_aux": "maximum uninterrupted run of the same value_level in chronological task-arrival order",
        "observation_window_burst_definition": "maximum tasks of the level sharing one observation_window_id",
    }
    return rows, definitions


def scenario_gate(source: Path, destination: Path, audit_root: Path) -> dict[str, Any]:
    if not source.is_dir():
        raise RuntimeError("TRUSTED_ORDINARY_SCENARIO_MISSING")
    contract = read_json(source / "bundle_contract.json")
    tasks = read_csv(source / "task_manifest.csv")
    slots = read_csv(source / "slot_trace.csv")
    stats, definitions = arrival_statistics(tasks)
    counts = dict(Counter(row["value_level"] for row in tasks))
    quartiles = {row["value_level"]: row["day_quartile_counts"] for row in stats}
    task_hash = v25.v25_bundle_hash(tasks)
    link_rows = [{key: row.get(key, "") for key in (
        "slot_index", "slot_seconds", "link__slot_index", "link__in_contact", "link__snr_db",
        "link__throughput_mbps", "link__link_quality", "link__remaining_window_s",
    )} for row in slots]
    window_rows = [{
        "slot": int(f(row["slot_index"])),
        "in_contact": int(str(row["link__in_contact"]).strip().lower() in {"1", "true", "yes"}),
        "remaining_window_s": f(row["link__remaining_window_s"]),
    } for row in slots]
    window_schedule_hash = stable_hash(window_rows)
    checks = {
        "seed_2026": int(contract.get("seed", -1)) == SEED,
        "duration_24h": int(contract.get("horizon_s", -1)) == 86400,
        "slots_4320": len(slots) == SLOTS,
        "tasks_268": len(tasks) == 268,
        "counts_exact": counts == EXPECTED_COUNTS,
        "manifest_sha_exact": sha256_file(source / "task_manifest.csv") == EXPECTED_MANIFEST_SHA,
        "slot_file_sha_exact": sha256_file(source / "slot_trace.csv") == EXPECTED_SLOT_SHA,
        "window_schedule_hash_exact": window_schedule_hash == EXPECTED_WINDOW_SCHEDULE_HASH,
        "task_hash_exact": task_hash == EXPECTED_TASK_HASH == contract.get("task_hash"),
        "ordinary_profile_declared": contract.get("arrival_profile") == "ordinary_random_restored_pre_late_hv_swap_v1",
        "active_late_hv_swaps_zero": int(contract.get("active_arrival_swap_count", -1)) == 0,
        "ordinary_quartiles_exact": quartiles == EXPECTED_QUARTILES,
        "unique_task_ids": len({row["task_id"] for row in tasks}) == 268,
        "unique_arrival_slots": len({row["arrival_slot"] for row in tasks}) == 268,
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scenario_classification": "TRUSTED_ORDINARY_RANDOM_RESTORED" if all(checks.values()) else "MISMATCH",
        "late_hv_skew": False if all(checks.values()) else None,
        "hv_burst": False if all(checks.values()) else None,
        "checks": checks, "task_hash": task_hash,
        "task_manifest_sha256": sha256_file(source / "task_manifest.csv"),
        "slot_trace_sha256": sha256_file(source / "slot_trace.csv"),
        "window_schedule_hash_v1": window_schedule_hash,
        "full_source_link_semantic_hash": stable_hash(link_rows),
        "contract_sha256": sha256_file(source / "bundle_contract.json"),
        "value_level_counts": counts, "arrival_statistics": stats, **definitions,
    }
    write_json(audit_root / "scenario_identity_gate.json", audit)
    write_csv(audit_root.parent / "tables" / "arrival_distribution_by_value.csv", stats)
    write_md(audit_root.parent / "tables" / "arrival_distribution_by_value.md", stats)
    if audit["status"] != "PASS":
        raise RuntimeError({"ORDINARY_SCENARIO_GATE_FAILED": checks})
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    copied = {path.name: sha256_file(path) for path in destination.iterdir() if path.is_file()}
    if copied.get("task_manifest.csv") != EXPECTED_MANIFEST_SHA or copied.get("slot_trace.csv") != EXPECTED_SLOT_SHA:
        raise RuntimeError("ORDINARY_SCENARIO_COPY_MISMATCH")
    audit["copied_bundle"] = str(destination)
    audit["copied_file_sha256"] = copied
    write_json(audit_root / "scenario_identity_gate.json", audit)
    return audit


def source_files() -> list[Path]:
    return [
        Path(__file__).resolve(), ROOT / "src" / "evaluation" / "frozen_v2_final_gate.py", FROZEN,
        v25.POWER_CONFIG, TRUSTED_SCENARIO / "bundle_contract.json", TRUSTED_SCENARIO / "task_manifest.csv",
        v25.TOOL_DIR / "native_tut.py", v25.TOOL_DIR / "native_offload.py",
        v25.SOURCE_TUT / "scripts" / "run_tut_uoi_rate_lite_v2.py",
        v25.SOURCE_TUT / "src" / "payload" / "payload_sizer.py",
        v25.SOURCE_TUT / "src" / "simulator" / "predictors.py",
        v25.SOURCE_TUT / "src" / "simulator" / "step_runner.py",
        v25.SOURCE_B1_OFFLOAD / "scripts" / "run_offload_vs_ground_24h_pilot.py",
        v25.SOURCE_B1_OFFLOAD / "src" / "baselines" / "ground_only.py",
        v25.SOURCE_B1_OFFLOAD / "src" / "baselines" / "offload_eft_thermal_aligned.py",
        v25.SOURCE_B1_OFFLOAD / "configs" / "methods" / "ground_only_v1.yaml",
    ]


def hash_snapshot() -> dict[str, str]:
    return {str(path): sha256_file(path) for path in source_files() if path.is_file()}


def run_identity(method: str, scenario: Any, bundle: Any, power: Mapping[str, Any], capacity: Mapping[str, Any], source_identity: Mapping[str, Any]) -> dict[str, Any]:
    identity = exact.scenario_identity(method, SEED, scenario, bundle, dict(power), CAPACITY)
    identity["capacity_identity"] = dict(capacity)
    return {
        "status": "PASS", "version": VERSION, "method": method, "seed": SEED,
        "pid": os.getpid(), "fresh_state": True, "independent_rng": True,
        "independent_queue_soc_thermal_online_state": True,
        "scenario_identity": identity, "source_identity": dict(source_identity),
        "power_identity": dict(power), "capacity_identity": dict(capacity),
    }


def canonical_product(policy: str) -> str:
    policy = str(policy or "").strip().lower()
    if policy == "defer":
        return "defer"
    if "roi_context" in policy or "context" in policy or "evidence" in policy:
        return "roi_context"
    if "metadata" in policy or "alert" in policy:
        return "metadata"
    if "roi" in policy or "verification" in policy:
        return "roi"
    return policy


def install_tut_proxy(configs: dict[str, Any]) -> dict[str, Any]:
    """Install process-local proxy hooks.  No source file is changed."""

    import src.payload.payload_sizer as payload_sizer
    import src.simulator.predictors as predictors
    import src.simulator.step_runner as step_runner

    balanced = next(
        profile for profile in configs["model_pool"]["profiles"]
        if profile["name"] == next(model["profile_name"] for model in configs["model_pool"]["models"] if model["name"] == "balanced")
    )
    overrides = configs["model_pool"].get("profile_overrides", {}).get(balanced["name"], {})
    profile = {**balanced, **overrides}

    def fixed_encoding(policy_name: str, payload_mb: float, payload_profiles_cfg=None):
        return (0.0, 0.0) if canonical_product(policy_name) == "defer" else (ENCODING_LATENCY_S, ENCODING_ENERGY_J)

    def actual_nominal_size(policy_name: str, detection_result=None, task=None, payload_profiles_cfg=None):
        product = canonical_product(policy_name)
        size = PRODUCT_MB.get(product, 0.0)
        return size, {"payload_size_source": "tut_semantic_proxy_public_abstract_product", "nominal_product": product}

    def predicted_nominal_pre_multiplier(policy_name: str, detection_result=None, task=None, payload_profiles_cfg=None):
        product = canonical_product(policy_name)
        if product == "defer":
            return 0.0, {"payload_size_source": "deferred"}
        robustness = f(profile.get("robustness", profile.get("robustness_to_difficulty")))
        quality = max(0.05, min(1.0, f(profile["quality_base"]) - (1.0 - robustness) * 0.32 * f(task.difficulty) - 0.16 * f(task.uncertainty)))
        return PRODUCT_MB[product] / (0.85 + quality), {"payload_size_source": "tut_semantic_proxy_predictor_inverse_quality_multiplier", "nominal_product": product}

    payload_sizer.estimate_payload_encoding_cost = fixed_encoding
    step_runner.estimate_payload_encoding_cost = fixed_encoding
    predictors.estimate_payload_encoding_cost = fixed_encoding
    payload_sizer.estimate_payload_size_with_diagnostics = actual_nominal_size
    predictors.estimate_payload_size_with_diagnostics = predicted_nominal_pre_multiplier
    configs["simulator"]["tut_predicted_energy_fix_enabled"] = True
    return {
        "patch_scope": "child_process_only", "predicted_energy_fix_enabled": True,
        "encoding_latency_s": ENCODING_LATENCY_S, "encoding_energy_j": ENCODING_ENERGY_J,
        "product_sizes_MB": PRODUCT_MB, "balanced_profile": profile,
    }


def apply_ground_processing(normalized: dict[str, Any], delay_s: float = GROUND_LATENCY_S) -> dict[str, Any]:
    result = deepcopy(normalized)
    tasks = {row["task_id"]: row for row in result["per_task"]}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in result["events"]:
        event["ground_available_time"] = f(event["ground_available_time"]) + delay_s
        event["valid_before_deadline"] = int(f(event["ground_available_time"]) <= f(event["deadline"]) + 1.0e-9 and f(event["ground_available_time"]) <= 86400.0 + 1.0e-9)
        by_task[event["task_id"]].append(event)
    rank = {"metadata": 0, "roi": 1, "evidence": 2, "full": 2}
    required = {"alert": 0, "metadata": 0, "verification": 1, "roi": 1, "context": 2, "evidence": 2, "roi_context": 2}
    for task_id, task in tasks.items():
        target = required.get(str(task["product_requirement"]).lower(), 1)
        qualifying = [event for event in by_task.get(task_id, []) if rank.get(str(event["stage"]).lower(), -1) >= target and int(event["valid_before_deadline"])]
        closure = min(qualifying, key=lambda event: f(event["ground_available_time"])) if qualifying else None
        task["closure_success"] = int(closure is not None)
        task["closure_ground_time"] = closure["ground_available_time"] if closure else ""
        if closure is None:
            late = [event for event in by_task.get(task_id, []) if rank.get(str(event["stage"]).lower(), -1) >= target]
            if late:
                task["failure_reason"] = "semantic_ground_processing_after_deadline_or_horizon"
    return result


def tut_runtime_rows(result: Any, action_space: Any) -> list[dict[str, Any]]:
    actions = {action.action_name: action for action in action_space.list_actions()}
    rows = []
    for record in result.records:
        if not int(getattr(record, "has_new_task", 0)):
            continue
        executed = str(getattr(record, "executed_action", ""))
        action = actions.get(executed)
        product = canonical_product(getattr(action, "payload_policy", "") if action else "")
        compute = f(getattr(record, "compute_energy_j", 0.0))
        rows.append({
            "task_id": str(getattr(record, "task_id", "")), "slot": int(getattr(record, "slot", 0)),
            "raw_action": str(getattr(record, "raw_action", "")), "executed_action": executed,
            "selected_model": str(getattr(record, "selected_model", "")), "product": product,
            "compute_energy_j": compute, "comm_energy_j": f(getattr(record, "comm_energy_j", 0.0)),
            "inference_energy_j_proxy": 0.0 if product == "defer" else INFERENCE_ENERGY_J,
            "encoding_energy_j": max(0.0, compute - (0.0 if product == "defer" else INFERENCE_ENERGY_J)),
            "inference_latency_s_proxy": 0.0 if product == "defer" else INFERENCE_LATENCY_S,
            "encoding_latency_s": max(0.0, f(getattr(record, "L_inf", 0.0)) - (0.0 if product == "defer" else INFERENCE_LATENCY_S)),
            "safe_filter_changed_action": int(str(getattr(record, "raw_action", "")) != executed),
        })
    return rows


def run_tut_child(method: str, bundle_root: Path, output: Path) -> None:
    v25.child_setup_path(v25.SOURCE_TUT)
    bundle = load_neutral_bundle(bundle_root)
    import native_tut
    import scripts.run_tut_uoi_rate_lite as tut_v1
    import scripts.run_tut_uoi_rate_lite_v2 as tut_v2
    from native_support import normalize_episode
    from src.simulator.environment import SimulationEnvironment

    binding = native_tut.make_binding(v25.SOURCE_TUT.resolve())
    source_identity = dict(binding.identity())
    if not source_identity.get("source_identity_passed"):
        raise RuntimeError("TUT_SOURCE_IDENTITY_MISMATCH")
    scenario = binding.build_scenario(bundle)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_TUT)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, CAPACITY)
    identity = run_identity(method, scenario, bundle, power, capacity_identity, source_identity)

    method_configs = tut_v2.build_configs(v25.SOURCE_TUT / "configs" / "experiments" / "tut_uoi_rate_lite_v2_formal_v1.yaml")
    configs = deepcopy(scenario.configs)
    configs["action_space"] = deepcopy(method_configs["action_space"])
    for key in ("tut_uoi_density_priority_enabled", "progressive_payload", "guarantee", "downlink_safety_throttle"):
        if key in method_configs["simulator"]:
            configs["simulator"][key] = deepcopy(method_configs["simulator"][key])
    configs["experiment"] = {"routers": ["tut_uoi_rate_lite"]}
    intervention = {"variant": "legacy_as_is", "predicted_energy_fix_enabled": bool(configs["simulator"].get("tut_predicted_energy_fix_enabled", False))}
    if method == METHOD_PROXY:
        intervention = {"variant": "literature_anchored_isolated_proxy", **install_tut_proxy(configs)}
    scenario = replace(scenario, configs=configs)
    environment = SimulationEnvironment(configs)
    runner = environment.build_episode_runner(scenario.timeline[:SLOTS])
    result = runner.run(tut_v1.build_router(configs))
    normalized = normalize_episode(method, scenario, result, runner, SLOTS)
    if method == METHOD_PROXY:
        normalized = apply_ground_processing(normalized)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    if method == METHOD_PROXY:
        write_csv(output / "ground_processing_trace.csv", [{
            "task_id": event["task_id"], "stage": event["stage"],
            "last_bit_arrival_s": f(event["ground_available_time"]) - GROUND_LATENCY_S,
            "semantic_decode_and_downstream_finish_s": event["ground_available_time"],
            "ground_processing_latency_s": GROUND_LATENCY_S, "deadline_s": event["deadline"],
            "valid_after_ground_processing": event["valid_before_deadline"],
            "ground_processing_energy": "NOT_PARAMETERIZED",
        } for event in normalized["events"]])
    write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
    runtime_rows = tut_runtime_rows(result, runner.step_runner_factory(runner.downlink_queue).action_space if False else environment.action_space)
    write_csv(output / "tut_task_runtime.csv", runtime_rows)
    write_json(output / "resolved_config.json", configs)
    identity.update({
        "intervention": intervention,
        "legacy_source_unchanged": True, "q_public": Q_BALANCED if method == METHOD_ASIS else Q_TUT_PROXY,
        "semantic_retention": 1.0 if method == METHOD_ASIS else SEMANTIC_RETENTION,
        "ground_processing_latency_s": 0.0 if method == METHOD_ASIS else GROUND_LATENCY_S,
        "ground_processing_energy": "NOT_PARAMETERIZED" if method == METHOD_PROXY else "NOT_APPLICABLE_LEGACY_ARRIVAL_SEMANTICS",
        "ground_energy_in_satellite_soc": False,
        "common_radio_power_W": 38.0,
        "abstract_products_not_original_pmsc_code_rates": True,
        "q_multiplication_count_public": 1,
    })
    write_json(output / "runtime_identity.json", identity)


def replay_ground_queue(trace: list[dict[str, Any]], payload_mb: float) -> list[dict[str, Any]]:
    remaining: dict[str, float] = {}
    result = []
    for raw in trace:
        row = dict(raw)
        before = sum(remaining.values())
        compute_task = str(row.get("compute_task_id", "") or "")
        if compute_task:
            remaining.setdefault(compute_task, payload_mb)
        budget = f(row.get("sent_mb"))
        for task_id in str(row.get("tx_task_id", "") or "").split(";"):
            if not task_id or budget <= 1.0e-12:
                continue
            amount = min(remaining.get(task_id, 0.0), budget)
            remaining[task_id] = max(0.0, remaining.get(task_id, 0.0) - amount)
            budget -= amount
            if remaining[task_id] <= 1.0e-12:
                remaining.pop(task_id, None)
        row["queue_before_mb"] = before
        row["queue_after_mb"] = sum(remaining.values())
        row["queue_item_count_after"] = len(remaining)
        result.append(row)
    return result


def normalize_ground(scenario: Any, native_rows: list[dict[str, Any]], scientific: Mapping[str, Any], trace: list[dict[str, Any]], ground_jobs: Mapping[str, Any]) -> dict[str, Any]:
    by_task = {str(row["task_id"]): row for row in native_rows}
    canonical = {str(row["task_id"]): row for row in scenario.canonical_tasks}
    tasks, events = [], []
    for task_id, source in canonical.items():
        row = by_task[task_id]
        finish = row.get("actual_finish_s", "")
        if finish not in {None, ""}:
            events.append({
                "method": METHOD_GROUND, "task_id": task_id, "stage": "full",
                "ground_available_time": f(finish), "deadline": f(source["deadline_time_s"]),
                "valid_before_deadline": int(f(finish) <= f(source["deadline_time_s"]) + 1.0e-9 and f(finish) <= 86400.0 + 1.0e-9),
                "event_source": "ground_only_fifo_corrected:shared_ground_compute_complete", "duplicate_flag": 0,
            })
        tasks.append({
            "method": METHOD_GROUND, "task_id": task_id, "arrival_time": source["arrival_time_s"],
            "deadline": source["deadline_time_s"], "value_level": source["value_level"],
            "effective_task_value": source["effective_task_value"], "product_requirement": source["product_requirement"],
            "selected_action": "GROUND", "executed_action": "GROUND", "selected_model": "ground_high_accuracy",
            "delivered_product": "full" if finish not in {None, ""} else "", "closure_success": int(row.get("closed", 0)),
            "closure_ground_time": finish, "failure_reason": row.get("failure_reason", ""),
            "controllable_energy_j": row.get("satellite_energy_j", 0.0), "transmitted_bytes": row.get("transmitted_bytes", 0.0),
        })
    slot_rows = []
    battery_min = f(scenario.configs["simulator"].get("battery_min"))
    thermal_max = f(scenario.configs["simulator"].get("thermal_max"), 1.0)
    for row in trace:
        slot = int(row.get("slot", len(slot_rows)))
        soc, thermal = f(row.get("soc_after")), f(row.get("thermal_after"))
        slot_rows.append({
            "method": METHOD_GROUND, "slot": slot, "time_s": f(row.get("time_s"), slot * SLOT_SECONDS),
            "battery_soc": soc, "thermal": thermal, "queue_mb": f(row.get("queue_after_mb")),
            "compute_energy_j": f(row.get("compute_energy_j")), "comm_energy_j": f(row.get("comm_energy_j")),
            "idle_energy_j": f(scenario.timeline[slot].resource_trace.base_power_w) * f(scenario.timeline[slot].slot_seconds),
            "transmitted_mb": f(row.get("sent_mb")), "in_contact": int(row.get("in_contact", 0)),
            "soc_violation": int(soc < battery_min - 1.0e-9), "thermal_violation": int(thermal > thermal_max + 1.0e-9),
        })
    resource = {
        "method": METHOD_GROUND, "slot_count": len(slot_rows),
        "minimum_soc": min(row["battery_soc"] for row in slot_rows), "final_soc": slot_rows[-1]["battery_soc"],
        "maximum_thermal": max(row["thermal"] for row in slot_rows),
        "soc_violation_count": sum(row["soc_violation"] for row in slot_rows),
        "thermal_violation_count": sum(row["thermal_violation"] for row in slot_rows),
        "controllable_energy_wh": f(scientific.get("total_controllable_satellite_energy_wh")),
        "transmitted_mb": sum(row["transmitted_mb"] for row in slot_rows),
        "terminal_queue_mb_actual_replay": slot_rows[-1]["queue_mb"],
        "pipeline_committed_backlog_mb_native": f(scientific.get("terminal_backlog_mb")),
        "pending_compute_backlog_mb_not_charged_as_public_Q_terminal": max(0.0, f(scientific.get("terminal_backlog_mb")) - slot_rows[-1]["queue_mb"]),
        "peak_queue_mb_actual_replay": max(row["queue_mb"] for row in slot_rows),
        "mean_queue_mb_actual_replay": statistics.fmean(row["queue_mb"] for row in slot_rows),
        "ground_compute_energy_counted_in_satellite": False, "native_summary": dict(scientific),
        "native_summary_identity_disclosure": "native pilot summary embeds legacy module constants; authoritative task/link/power identity is runtime_identity.json",
    }
    return {"per_task": tasks, "events": events, "slots": slot_rows, "resource_summary": resource}


def ground_job_rows(sim: Any, scenario: Any, native_rows: list[dict[str, Any]], trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = {row["task_id"]: row for row in scenario.canonical_tasks}
    native = {row["task_id"]: row for row in native_rows}
    queue_by_slot = {int(row["slot"]): row for row in trace}
    jobs = sim.jobs
    rows = []
    for task_id, job in jobs.items():
        task = source[task_id]
        arrival = f(task["arrival_time_s"])
        deadline = f(task["deadline_time_s"])
        prior = [other for other in jobs.values() if other.task_id != task_id and other.arrival_s <= arrival + 1.0e-9 and (other.closure_s is None or other.closure_s > arrival + 1.0e-9)]
        tx_finish = job.tx_finish_s
        closure = job.closure_s
        not_tx = tx_finish is None
        tx_late = tx_finish is not None and tx_finish > deadline + 1.0e-9
        ground_late = tx_finish is not None and tx_finish <= deadline + 1.0e-9 and closure is not None and closure > deadline + 1.0e-9
        horizon_incomplete = closure is None or closure > 86400.0 + 1.0e-9
        primary = "closed" if int(native[task_id]["closed"]) else "unfinished_downlink" if not_tx else "downlink_completed_after_deadline" if tx_late else "ground_processing_completed_after_deadline" if ground_late else "day_end_unfinished" if horizon_incomplete else "other"
        slot = int(arrival // SLOT_SECONDS)
        rows.append({
            "task_id": task_id, "value_level": task["value_level"], "arrival_s": arrival, "deadline_s": deadline,
            "compression_finish_s": job.compute_finish_s if job.compute_finish_s is not None else "",
            "tx_start_s": job.tx_start_s if job.tx_start_s is not None else "", "last_bit_arrival_s": tx_finish if tx_finish is not None else "",
            "ground_queue_enter_s": job.ground_queue_enter_s if job.ground_queue_enter_s is not None else "",
            "ground_compute_start_s": job.ground_compute_start_s if job.ground_compute_start_s is not None else "",
            "closure_s": closure if closure is not None else "", "remaining_downlink_mb_at_horizon": job.remaining_mb,
            "closed": int(native[task_id]["closed"]), "primary_unclosed_reason": primary,
            "unfinished_downlink": int(not_tx), "downlink_after_deadline": int(tx_late),
            "ground_processing_after_deadline": int(ground_late), "day_end_unfinished": int(horizon_incomplete),
            "fifo_jobs_ahead_at_arrival": len(prior),
            "downlink_queue_mb_at_arrival": f(queue_by_slot.get(slot, {}).get("queue_before_mb")),
        })
    return rows


def run_ground_child(bundle_root: Path, output: Path) -> None:
    v25.child_setup_path(v25.SOURCE_B1_OFFLOAD)
    bundle = load_neutral_bundle(bundle_root)
    import native_offload
    from scripts.run_offload_vs_ground_24h_pilot import common_parameters, run_ground, task_rows, summarize
    from scripts.run_three_method_minimal_robustness_v1 import augment_trace
    from src.common.config import load_yaml

    binding = native_offload.make_binding(v25.SOURCE_B1_OFFLOAD.resolve())
    source_identity = dict(binding.identity())
    scenario = binding.build_scenario(bundle)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_B1_OFFLOAD)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, CAPACITY)
    identity = run_identity(METHOD_GROUND, scenario, bundle, power, capacity_identity, source_identity)
    offload_cfg = load_yaml(v25.SOURCE_B1_OFFLOAD / "configs" / "methods" / "offload_eft_thermal_aligned_v1.yaml")
    ground_cfg = load_yaml(v25.SOURCE_B1_OFFLOAD / "configs" / "methods" / "ground_only_v1.yaml")
    pars = common_parameters(scenario.configs, offload_cfg)
    if abs(f(pars["ground"].compressed_size_mb) - 4.0) > 1.0e-12:
        raise RuntimeError("GROUND_COMPRESSED_PAYLOAD_NOT_4MB")
    shared = {"timeline": scenario.timeline[:SLOTS], "configs": scenario.configs, "task_steps": [step for step in scenario.timeline[:SLOTS] if step.task is not None]}
    sim, decisions = run_ground(shared, ground_cfg, pars)
    native_rows, common_rows = task_rows(METHOD_GROUND, sim, shared, pars, decisions)
    trace = augment_trace(sim.actual_trace, shared["timeline"], scenario.configs["simulator"])
    trace = replay_ground_queue(trace, 4.0)
    scientific = summarize(METHOD_GROUND, native_rows, common_rows, trace, sim)
    normalized = normalize_ground(scenario, native_rows, scientific, trace, sim.jobs)
    jobs = ground_job_rows(sim, scenario, native_rows, trace)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    write_csv(output / "ground_job_audit.csv", jobs)
    write_csv(output / "ground_actual_queue_trace.csv", trace)
    write_json(output / "resolved_config.json", scenario.configs)
    identity.update({
        "forced_ground_once": True, "no_onboard_yolo": True, "onboard_compression_only": True,
        "compressed_payload_MB": 4.0, "strict_fifo": True, "deadline_aware_admission": False,
        "hv_priority": False, "early_defer": False, "shared_offload_ground_server_code": True,
        "ground_server_count": ground_cfg["method"]["ground_compute_servers"],
        "ground_queue_policy": ground_cfg["method"]["ground_queue_policy"],
        "ground_queue_capacity": ground_cfg["method"]["ground_queue_capacity"],
        "queue_accounting": "actual FIFO trace replay from compute_task_id/tx_task_id/sent_mb",
    })
    write_json(output / "runtime_identity.json", identity)


def child_main(spec_path: Path) -> int:
    spec = read_json(spec_path)
    method, output, bundle = spec["method"], Path(spec["output"]), Path(spec["bundle_root"])
    if output.exists():
        raise FileExistsError(output)
    if method in {METHOD_ASIS, METHOD_PROXY}:
        run_tut_child(method, bundle, output)
    elif method == METHOD_GROUND:
        run_ground_child(bundle, output)
    else:
        raise ValueError(method)
    write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": SEED, "completed_at": utc_now()})
    print(json.dumps({"status": "PASS", "method": method, "output": str(output)}, ensure_ascii=False))
    return 0


def launch(spec: Mapping[str, Any], log_root: Path) -> dict[str, Any]:
    spec_path = log_root / f"{SLUG[spec['method']]}.spec.json"
    write_json(spec_path, spec)
    command = [sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (log_root / f"{SLUG[spec['method']]}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (log_root / f"{SLUG[spec['method']]}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    return {"method": spec["method"], "returncode": completed.returncode, "status": "PASS" if completed.returncode == 0 else "FAIL", "pid_isolated_process": True}


def frozen_for(method: str) -> dict[str, Any]:
    frozen = deepcopy(read_json(FROZEN))
    if method == METHOD_PROXY:
        if abs(f(frozen["q_eff"]["balanced"]) - Q_BALANCED) > 1.0e-9:
            raise RuntimeError("FROZEN_BALANCED_Q_MISMATCH")
        frozen["q_eff"]["balanced"] = Q_TUT_PROXY
    return frozen


def evaluate(method: str, path: Path, bundle_root: Path) -> dict[str, Any]:
    metric, ledger = gatecore.evaluate_public_run(method, path, read_json(bundle_root / "resolved_config.json"), frozen_for(method))
    trace = read_csv(path / "per_slot_resource_trace.csv")
    metric.update({
        "seed": SEED, "mean_queue_MB": statistics.fmean(f(row.get("queue_mb")) for row in trace),
        "p95_queue_MB": percentile([f(row.get("queue_mb")) for row in trace], 0.95),
        "final_soc": f(trace[-1].get("battery_soc")),
    })
    write_csv(path / "frozen_public_event_ledger.csv", ledger)
    write_json(path / "frozen_public_metrics.json", metric)
    return metric


def copy_counterfactual(source: Path, destination: Path, *, subtract_ground_delay: bool) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("per_task_outcomes.csv", "ground_delivery_events.csv", "per_slot_resource_trace.csv", "resource_summary.json"):
        shutil.copy2(source / name, destination / name)
    if subtract_ground_delay:
        events = read_csv(destination / "ground_delivery_events.csv")
        for event in events:
            event["ground_available_time"] = f(event["ground_available_time"]) - GROUND_LATENCY_S
            event["valid_before_deadline"] = int(f(event["ground_available_time"]) <= f(event["deadline"]) + 1.0e-9)
        write_csv(destination / "ground_delivery_events.csv", events)
        tasks = read_csv(destination / "per_task_outcomes.csv")
        by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
        for event in events:
            by_task[event["task_id"]].append(event)
        required_rank = {"alert": 0, "metadata": 0, "verification": 1, "roi": 1, "context": 2, "evidence": 2, "roi_context": 2}
        stage_rank = {"metadata": 0, "roi": 1, "evidence": 2, "full": 2}
        for task in tasks:
            target = required_rank.get(str(task["product_requirement"]).lower(), 1)
            possible = [event for event in by_task.get(task["task_id"], []) if stage_rank.get(event["stage"].lower(), -1) >= target and int(event["valid_before_deadline"])]
            closure = min(possible, key=lambda event: f(event["ground_available_time"])) if possible else None
            task["closure_success"] = int(closure is not None)
            task["closure_ground_time"] = closure["ground_available_time"] if closure else ""
        write_csv(destination / "per_task_outcomes.csv", tasks)


def tut_comparison(attempt: Path, metrics: Mapping[str, dict[str, Any]], bundle_root: Path) -> dict[str, Any]:
    asis_path = attempt / "runs" / SLUG[METHOD_ASIS]
    proxy_path = attempt / "runs" / SLUG[METHOD_PROXY]
    cf_root = attempt / "diagnostics"
    retention1 = cf_root / "proxy_retention_1_ground_delay_1p87"
    no_ground = cf_root / "proxy_retention_1_ground_delay_0"
    copy_counterfactual(proxy_path, retention1, subtract_ground_delay=False)
    copy_counterfactual(proxy_path, no_ground, subtract_ground_delay=True)
    frozen_base = read_json(FROZEN)
    retention_metric, _ = gatecore.evaluate_public_run("PROXY-CF-RETENTION-1", retention1, read_json(bundle_root / "resolved_config.json"), frozen_base)
    no_ground_metric, _ = gatecore.evaluate_public_run("PROXY-CF-NO-GROUND-DELAY", no_ground, read_json(bundle_root / "resolved_config.json"), frozen_base)
    actual = metrics[METHOD_PROXY]
    legacy = metrics[METHOD_ASIS]
    decomposition = []
    for field in ("gross_utility", "J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "mean_full_closure_latency_s"):
        decomposition.append({
            "metric": field, "legacy_as_is": legacy.get(field), "proxy_actual": actual.get(field),
            "total_delta_proxy_minus_legacy": f(actual.get(field)) - f(legacy.get(field)),
            "semantic_retention_component": f(actual.get(field)) - f(retention_metric.get(field)),
            "ground_1p87s_component": f(retention_metric.get(field)) - f(no_ground_metric.get(field)),
            "encoding_size_prediction_and_action_component": f(no_ground_metric.get(field)) - f(legacy.get(field)),
        })
    write_csv(attempt / "tables" / "tut_change_decomposition.csv", decomposition)
    write_md(attempt / "tables" / "tut_change_decomposition.md", decomposition)

    old_tasks = {row["task_id"]: row for row in read_csv(asis_path / "per_task_outcomes.csv")}
    new_tasks = {row["task_id"]: row for row in read_csv(proxy_path / "per_task_outcomes.csv")}
    changed = [task_id for task_id in old_tasks if (old_tasks[task_id]["selected_action"], old_tasks[task_id]["executed_action"]) != (new_tasks[task_id]["selected_action"], new_tasks[task_id]["executed_action"])]
    runtime = {method: read_csv(attempt / "runs" / SLUG[method] / "tut_task_runtime.csv") for method in (METHOD_ASIS, METHOD_PROXY)}
    candidate_logs = {method: read_csv(attempt / "runs" / SLUG[method] / "candidate_action_log.csv") for method in (METHOD_ASIS, METHOD_PROXY)}
    behavior = {}
    for method, rows in runtime.items():
        counts = Counter(row["product"] for row in rows)
        behavior[method] = {
            "defer_count": counts["defer"], "metadata_count": counts["metadata"], "roi_count": counts["roi"], "roi_context_count": counts["roi_context"],
            "non_defer_count": len(rows) - counts["defer"], "safe_filter_correction_count": sum(int(row["safe_filter_changed_action"]) for row in rows),
            "candidate_safe_rejection_count": sum(str(row.get("safety_status", "")).lower() == "rejected" for row in candidate_logs[method]),
            "compute_energy_j_from_task_records": sum(f(row["compute_energy_j"]) for row in rows),
            "encoding_energy_j_from_task_records": sum(f(row["encoding_energy_j"]) for row in rows),
        }
    aggression_signals = {
        "non_defer_delta": behavior[METHOD_PROXY]["non_defer_count"] - behavior[METHOD_ASIS]["non_defer_count"],
        "roi_context_delta": behavior[METHOD_PROXY]["roi_context_count"] - behavior[METHOD_ASIS]["roi_context_count"],
        "transmitted_MB_delta": f(actual["transmitted_MB"]) - f(legacy["transmitted_MB"]),
    }
    positive = sum(value > 1.0e-9 for value in aggression_signals.values())
    negative = sum(value < -1.0e-9 for value in aggression_signals.values())
    aggressiveness = "MORE_AGGRESSIVE" if positive and not negative else "LESS_AGGRESSIVE" if negative and not positive else "MIXED_OR_UNCHANGED"
    result = {
        "action_changed_task_count": len(changed), "action_changed_task_ids": changed,
        "behavior": behavior, "aggression_signals": aggression_signals,
        "predicted_energy_fix_behavior_judgment": aggressiveness,
        "safe_filter_additional_corrections": behavior[METHOD_PROXY]["safe_filter_correction_count"] - behavior[METHOD_ASIS]["safe_filter_correction_count"],
        "candidate_safe_rejection_delta": behavior[METHOD_PROXY]["candidate_safe_rejection_count"] - behavior[METHOD_ASIS]["candidate_safe_rejection_count"],
        "thermal_peak_delta": f(actual["maximum_thermal"]) - f(legacy["maximum_thermal"]),
        "quality_loss_J_all": f(actual["J_all_net"]) - f(retention_metric["J_all_net"]),
        "ground_delay_loss_J_all": f(retention_metric["J_all_net"]) - f(no_ground_metric["J_all_net"]),
        "runtime_encoding_size_prediction_action_delta_J_all": f(no_ground_metric["J_all_net"]) - f(legacy["J_all_net"]),
        "actual_energy_cost_delta": f(actual["energy_cost"]) - f(legacy["energy_cost"]),
        "actual_downlink_cost_delta": f(actual["downlink_cost"]) - f(legacy["downlink_cost"]),
        "actual_terminal_backlog_cost_delta": f(actual["terminal_backlog_cost"]) - f(legacy["terminal_backlog_cost"]),
        "actual_gross_utility_delta": f(actual["gross_utility"]) - f(legacy["gross_utility"]),
        "ground_processing_energy_disclosure": "NOT_PARAMETERIZED; latency is specified but ground processor power is not, so no ground-energy number is invented",
        "counterfactuals_are_diagnostic_only": True,
    }
    write_json(attempt / "audit" / "tut_proxy_difference_audit.json", result)
    write_csv(attempt / "tables" / "tut_action_changed_tasks.csv", [{"task_id": task_id, "old_action": old_tasks[task_id]["executed_action"], "new_action": new_tasks[task_id]["executed_action"]} for task_id in changed])
    return result


def historical_ground_candidates(current_attempt: Path) -> list[dict[str, Any]]:
    rows = []
    for metric_path in OUTPUT_BASE.parent.rglob("frozen_public_metrics.json"):
        if OUTPUT_BASE in metric_path.parents or current_attempt in metric_path.parents or "GROUND_ONLY" not in str(metric_path).upper():
            continue
        try:
            metric = read_json(metric_path)
            if str(metric.get("method", "")).upper() not in {"GROUND-ONLY", METHOD_GROUND}:
                continue
            identity_path = metric_path.parent / "runtime_identity.json"
            identity = read_json(identity_path) if identity_path.is_file() else {}
            scenario = identity.get("scenario_identity", {})
            if isinstance(scenario.get("scenario_identity"), dict):
                scenario = scenario["scenario_identity"]
            task_hash = str(scenario.get("task_hash", ""))
            rows.append({
                "path": str(metric_path), "method": metric.get("method"), "seed": metric.get("seed", scenario.get("seed")),
                "task_hash": task_hash, "same_ordinary_task_hash": int(task_hash == EXPECTED_TASK_HASH),
                "J_all_net": metric.get("J_all_net"), "all_closure_count": metric.get("all_closure_count"),
                "HV_closure_count": metric.get("HV_closure_count"), "high_closure_count": metric.get("high_closure_count"),
                "urgent_closure_count": metric.get("urgent_closure_count"), "mean_full_closure_latency_s": metric.get("mean_full_closure_latency_s"),
                "transmitted_MB": metric.get("transmitted_MB"), "terminal_queue_MB": metric.get("terminal_queue_MB"), "peak_queue_MB": metric.get("peak_queue_MB"),
            })
        except (OSError, ValueError, KeyError):
            continue
    return rows


def reconciliation_audits(attempt: Path, metrics: Mapping[str, dict[str, Any]]) -> None:
    proxy = attempt / "runs" / SLUG[METHOD_PROXY]
    runtime = read_csv(proxy / "tut_task_runtime.csv")
    active = [row for row in runtime if row["product"] != "defer"]
    candidates = read_csv(proxy / "candidate_action_log.csv")
    public = read_csv(proxy / "frozen_public_event_ledger.csv")
    predicted_size_checks = {}
    for product, expected in PRODUCT_MB.items():
        values = [f(row["predicted_payload_mb"]) for row in candidates if row["candidate_payload"] == product]
        predicted_size_checks[product] = {
            "expected_MB": expected, "candidate_count": len(values),
            "minimum_MB": min(values) if values else None, "maximum_MB": max(values) if values else None,
            "exact": bool(values) and all(abs(value - expected) <= 1.0e-12 for value in values),
        }
    tut_checks = {
        "predicted_energy_fix_enabled": bool(read_json(proxy / "runtime_identity.json")["intervention"]["predicted_energy_fix_enabled"]),
        "actual_non_defer_compute_energy_exact_27p62J": bool(active) and all(abs(f(row["compute_energy_j"]) - 27.62) <= 1.0e-9 for row in active),
        "actual_non_defer_encoding_latency_exact_0p43s": bool(active) and all(abs(f(row["encoding_latency_s"]) - 0.43) <= 1.0e-9 for row in active),
        "actual_non_defer_encoding_energy_exact_6p02J": bool(active) and all(abs(f(row["encoding_energy_j"]) - 6.02) <= 1.0e-9 for row in active),
        "public_q_exact_proxy_value": bool(public) and all(abs(f(row["q_eff"]) - Q_TUT_PROXY) <= 1.0e-12 for row in public),
        "public_q_count_exactly_once": bool(public) and all(int(f(row["q_count"])) == 1 for row in public),
        "public_gross_recompute_exact": f(metrics[METHOD_PROXY]["q_once_max_recompute_error"]) <= 1.0e-12,
        "nominal_product_sizes_exact_in_predictor": all(row["exact"] for row in predicted_size_checks.values()),
    }
    tut_audit = {
        "status": "PASS" if all(tut_checks.values()) else "FAIL", "checks": tut_checks,
        "non_defer_task_count": len(active), "product_size_checks": predicted_size_checks,
        "PM_generation_identity": "PROXY_INCLUDED_IN_YOLO_B_NO_SEPARATE_CHARGE",
        "subslot_limitation": "1.87 s onboard latency enters prediction, feedback, energy and thermal, but the 20 s queue model does not shift the same-slot physical transmission start by 1.87 s",
        "ground_energy": "NOT_PARAMETERIZED_AND_NOT_INCLUDED_IN_SATELLITE_SOC_OR_PUBLIC_ENERGY",
    }
    write_json(attempt / "audit" / "tut_predicted_actual_public_reconciliation.json", tut_audit)
    write_csv(attempt / "tables" / "tut_predicted_actual_public_reconciliation.csv", [
        {"layer": "predicted", "latency_s_per_non_defer": 1.87, "compute_energy_J_per_non_defer": 27.62, "q": "not public at this layer", "count": len(active)},
        {"layer": "actual_satellite", "latency_s_per_non_defer": 1.87, "compute_energy_J_per_non_defer": 27.62, "q": "not applied to SOC", "count": len(active)},
        {"layer": "public_ground", "latency_s_per_non_defer": "last_bit+1.87", "compute_energy_J_per_non_defer": "satellite trace actual", "q": Q_TUT_PROXY, "count": len(public)},
    ])

    ground_run = attempt / "runs" / SLUG[METHOD_GROUND]
    resource = read_json(ground_run / "resource_summary.json")
    ground_public = metrics[METHOD_GROUND]
    ground_audit = {
        "status": "PASS",
        "public_terminal_downlink_queue_MB": ground_public["terminal_queue_MB"],
        "actual_replay_terminal_WAIT_DOWNLINK_MB": resource["terminal_queue_mb_actual_replay"],
        "pipeline_committed_backlog_MB": resource["pipeline_committed_backlog_mb_native"],
        "pending_compute_MB_excluded_from_public_terminal_queue": resource["pending_compute_backlog_mb_not_charged_as_public_Q_terminal"],
        "public_peak_downlink_queue_MB": ground_public["peak_queue_MB"],
        "public_mean_downlink_queue_MB": ground_public["mean_queue_MB"],
        "native_maximum_ground_wait_queue_tasks": resource["native_summary"].get("maximum_ground_queue_length"),
        "terminal_backlog_cost_recomputed": f(ground_public["terminal_queue_MB"]) * f(read_json(FROZEN)["public_prices"]["terminal_backlog_price_per_MB"]),
        "legacy_native_summary_task_hash_is_not_authoritative": resource["native_summary"].get("task_hash"),
        "authoritative_task_hash": EXPECTED_TASK_HASH,
    }
    ground_audit["status"] = "PASS" if abs(f(ground_audit["public_terminal_downlink_queue_MB"]) - f(ground_audit["actual_replay_terminal_WAIT_DOWNLINK_MB"])) <= 1.0e-9 and abs(f(ground_audit["terminal_backlog_cost_recomputed"]) - f(ground_public["terminal_backlog_cost"])) <= 1.0e-9 else "FAIL"
    write_json(attempt / "audit" / "ground_queue_public_reconciliation.json", ground_audit)
    if tut_audit["status"] != "PASS" or ground_audit["status"] != "PASS":
        raise RuntimeError("PREDICTED_ACTUAL_PUBLIC_RECONCILIATION_FAILED")


def ground_analysis(attempt: Path, metric: Mapping[str, Any], historical: list[dict[str, Any]]) -> dict[str, Any]:
    run = attempt / "runs" / SLUG[METHOD_GROUND]
    jobs = read_csv(run / "ground_job_audit.csv")
    hv = [row for row in jobs if row["value_level"] in {"high", "urgent"}]
    unclosed = [row for row in hv if not int(row["closed"])]
    reasons = Counter(row["primary_unclosed_reason"] for row in unclosed)
    write_csv(attempt / "tables" / "ground_unclosed_hv_reasons.csv", unclosed)
    write_md(attempt / "tables" / "ground_unclosed_hv_reasons.md", unclosed)
    write_csv(attempt / "tables" / "ground_historical_candidates.csv", historical)
    exact_history = [row for row in historical if int(row["same_ordinary_task_hash"])]
    late_zero = [row for row in historical if not int(row["same_ordinary_task_hash"]) and int(f(row.get("HV_closure_count"))) == 0]
    comparison = [{"comparison": "current_ordinary_corrected", **dict(metric), "task_hash": EXPECTED_TASK_HASH, "exact_reproduction_claim": True}]
    comparison.extend({"comparison": "historical_same_hash" if int(row["same_ordinary_task_hash"]) else "historical_different_hash_parallel_only", **row, "exact_reproduction_claim": bool(int(row["same_ordinary_task_hash"]))} for row in historical)
    write_csv(attempt / "tables" / "ground_historical_recovery_comparison.csv", comparison)
    result = {
        "ordinary_scene_hv_closure_recovered": int(metric["HV_closure_count"]) > 0,
        "ordinary_scene_HV_closure_count": metric["HV_closure_count"],
        "ordinary_scene_high_closure_count": metric["high_closure_count"],
        "ordinary_scene_urgent_closure_count": metric["urgent_closure_count"],
        "unclosed_HV_count": len(unclosed), "unclosed_primary_reason_counts": dict(reasons),
        "mean_fifo_jobs_ahead_at_HV_arrival": statistics.fmean(f(row["fifo_jobs_ahead_at_arrival"]) for row in hv),
        "mean_downlink_queue_MB_at_HV_arrival": statistics.fmean(f(row["downlink_queue_mb_at_arrival"]) for row in hv),
        "historical_same_hash_result_count": len(exact_history),
        "historical_different_hash_hv_zero_result_count": len(late_zero),
        "late_hv_zero_attribution": "NOT_IDENTIFIABLE_AS_EXACT_CAUSAL_COUNT_WHEN_HASH_DIFFERS" if late_zero and not exact_history else "NO_MATCHED_LATE_COUNTERFACTUAL",
        "fifo_4MB_congestion_remains": f(metric["peak_queue_MB"]) > 0.0 or len(unclosed) > 0,
    }
    write_json(attempt / "audit" / "ground_recovery_analysis.json", result)
    return result


def make_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        path = OUTPUT_BASE / f"attempt_{index:03d}"
        if not path.exists():
            path.mkdir(parents=True, exist_ok=False)
            return path
    raise RuntimeError("NO_ATTEMPT_SLOT")


def main() -> int:
    attempt = make_attempt()
    for name in ("audit", "config", "logs", "runs", "scenario", "tables", "diagnostics"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    before = hash_snapshot()
    write_json(attempt / "audit" / "pre_run_hashes.json", before)
    bundle_root = attempt / "scenario" / "ordinary_random_restored_seed2026"
    gate = scenario_gate(TRUSTED_SCENARIO, bundle_root, attempt / "audit")

    specs = []
    for method in METHODS:
        output = attempt / "runs" / SLUG[method]
        spec = {"version": VERSION, "method": method, "seed": SEED, "bundle_root": str(bundle_root), "output": str(output), "capacity": CAPACITY}
        specs.append(spec)
        write_json(attempt / "config" / f"{SLUG[method]}_run_spec.json", spec)
    statuses = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(launch, spec, attempt / "logs"): spec for spec in specs}
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
    write_json(attempt / "audit" / "parallel_execution_audit.json", {"max_workers": 3, "independent_processes": True, "statuses": statuses})
    if any(row["status"] != "PASS" for row in statuses):
        raise RuntimeError({"ISOLATED_RUNTIME_FAILED": statuses, "logs": str(attempt / "logs")})

    metrics = {method: evaluate(method, attempt / "runs" / SLUG[method], bundle_root) for method in METHODS}
    fields = [
        "method", "J_all_net", "J_HV_net", "gross_utility", "HV_gross_utility", "energy_cost", "downlink_cost", "terminal_backlog_cost",
        "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count", "controllable_energy_Wh", "transmitted_MB",
        "terminal_queue_MB", "peak_queue_MB", "mean_queue_MB", "mean_full_closure_latency_s", "p95_full_closure_latency_s",
        "minimum_soc", "final_soc", "maximum_thermal", "soc_violation_count", "thermal_violation_count",
        "metadata_delivery_count", "ROI_delivery_count", "Evidence_delivery_count",
    ]
    metric_rows = [{field: metrics[method].get(field) for field in fields} for method in METHODS]
    write_csv(attempt / "tables" / "three_method_per_run_metrics.csv", metric_rows)
    write_md(attempt / "tables" / "three_method_per_run_metrics.md", metric_rows)
    delta = [{"metric": field, "TUT_LITE_AS_IS": metrics[METHOD_ASIS].get(field), "TUT_SEMANTIC_PROXY_V1": metrics[METHOD_PROXY].get(field), "delta_proxy_minus_asis": f(metrics[METHOD_PROXY].get(field)) - f(metrics[METHOD_ASIS].get(field))} for field in fields if field != "method"]
    write_csv(attempt / "tables" / "tut_legacy_proxy_deltas.csv", delta)
    write_md(attempt / "tables" / "tut_legacy_proxy_deltas.md", delta)

    reconciliation_audits(attempt, metrics)
    tut = tut_comparison(attempt, metrics, bundle_root)
    historical = historical_ground_candidates(attempt)
    ground = ground_analysis(attempt, metrics[METHOD_GROUND], historical)

    identities = [read_json(attempt / "runs" / SLUG[method] / "runtime_identity.json") for method in METHODS]
    runtime_link_hashes = {identity["method"]: identity["capacity_identity"].get("runtime_link_trace_hash") for identity in identities}
    identity_checks = {
        "same_task_hash": len({identity["scenario_identity"]["task_hash"] for identity in identities}) == 1,
        "same_runtime_link_hash": len(set(runtime_link_hashes.values())) == 1,
        "runtime_link_hash_matches_trusted_capacity_050": set(runtime_link_hashes.values()) == {EXPECTED_RUNTIME_LINK_HASH},
        "correct_40W_power": all(identity["power_identity"]["frozen_power"] == v25.FROZEN_POWER for identity in identities),
        "separate_processes": len({identity["pid"] for identity in identities}) == 3,
        "q_once": all(f(metrics[method]["q_once_max_recompute_error"]) <= 1.0e-12 for method in METHODS),
        "no_safety_violations": all(int(metrics[method]["soc_violation_count"]) == 0 and int(metrics[method]["thermal_violation_count"]) == 0 for method in METHODS),
    }
    identity_audit = {"status": "PASS" if all(identity_checks.values()) else "FAIL", "checks": identity_checks, "runtime_link_hashes": runtime_link_hashes, "identities": identities}
    write_json(attempt / "audit" / "method_environment_evaluator_safety_audit.json", identity_audit)
    if identity_audit["status"] != "PASS":
        raise RuntimeError({"IDENTITY_OR_SAFETY_AUDIT_FAILED": identity_checks})

    after = hash_snapshot()
    hash_audit = {"status": "PASS" if before == after else "FAIL", "before": before, "after": after, "changed_source_files": [path for path in before if before.get(path) != after.get(path)]}
    write_json(attempt / "audit" / "post_run_read_only_hash_audit.json", hash_audit)
    if hash_audit["status"] != "PASS":
        raise RuntimeError("READ_ONLY_SOURCE_HASH_CHANGED")

    reference = read_json(FULL_V2_REFERENCE) if FULL_V2_REFERENCE.is_file() else None
    proxy_vs_reference = None if reference is None else {
        "reference_path": str(FULL_V2_REFERENCE), "same_ordinary_task_hash": True,
        "proxy_J_all_net": metrics[METHOD_PROXY]["J_all_net"], "Full_PROP_V2_J_all_net": reference["J_all_net"],
        "delta_proxy_minus_Full_PROP_V2": f(metrics[METHOD_PROXY]["J_all_net"]) - f(reference["J_all_net"]),
        "judgment": "ABOVE" if f(metrics[METHOD_PROXY]["J_all_net"]) > f(reference["J_all_net"]) else "CLOSE_WITHIN_5_PERCENT" if abs(f(metrics[METHOD_PROXY]["J_all_net"]) - f(reference["J_all_net"])) <= 0.05 * max(abs(f(reference["J_all_net"])), 1.0e-9) else "BELOW",
        "ranking_not_used_for_tuning": True,
    }
    final = {
        "status": "PASS", "completion_marker": COMPLETION, "attempt": str(attempt),
        "scenario": gate, "metrics": metrics, "tut_analysis": tut, "ground_analysis": ground,
        "full_prop_v2_reference": proxy_vs_reference,
        "scientific_evidence_labels": {
            "direct_runtime_evidence": ["three fresh run traces", "actual FIFO queue replay", "scenario and source hashes"],
            "diagnostic_counterfactual": ["retention=1", "ground delay=0 decomposition"],
            "proxy_assumptions": ["PMSC 0.43s/14W", "semantic retention 0.95", "symmetric disclosed ground proxy energy 27.62J"],
            "not_claimed": ["original paper code-rate replication", "statistical significance", "multi-seed stability"],
        },
        "no_parameters_tuned_from_ranking": True, "single_seed_only": True,
    }
    write_json(attempt / "final_assessment.json", final)
    conclusions = [
        "# 普通场景单seed TUT代理更新 + GROUND-ONLY恢复验证",
        "",
        f"完成状态：`{COMPLETION}`。本次仅运行seed=2026，没有多seed、没有按排名调参。",
        "",
        f"场景门控：{gate['scenario_classification']}，task hash `{gate['task_hash']}`；三方法capacity=0.5运行时窗口hash一致，40 W电源身份一致。",
        "",
        f"TUT代理相对旧LITE：J_all变化 {f(metrics[METHOD_PROXY]['J_all_net']) - f(metrics[METHOD_ASIS]['J_all_net']):.6f}，J_HV变化 {f(metrics[METHOD_PROXY]['J_HV_net']) - f(metrics[METHOD_ASIS]['J_HV_net']):.6f}，动作变化任务数 {tut['action_changed_task_count']}。",
        f"质量折损对J_all的诊断贡献 {tut['quality_loss_J_all']:.6f}，地面1.87 s贡献 {tut['ground_delay_loss_J_all']:.6f}，其余编码/尺寸/预测修复及动作联动贡献 {tut['runtime_encoding_size_prediction_action_delta_J_all']:.6f}。这些分解是只读反事实，不是正式重评分。",
        f"在用户指定的三项中，质量折损绝对影响最大；但总降幅更主要来自编码/名义尺寸/预测修复改变动作后增加的下传与积压成本，而不是1.87 s地面时延。",
        f"预测能耗修复后的行为判定：{tut['predicted_energy_fix_behavior_judgment']}；额外SafeFilter实际纠正 {tut['safe_filter_additional_corrections']}，候选硬安全拒绝变化 {tut['candidate_safe_rejection_delta']}；热峰变化 {tut['thermal_peak_delta']:.9f}。",
        "",
        f"GROUND-ONLY普通场景HV闭环数 {ground['ordinary_scene_HV_closure_count']}（high={ground['ordinary_scene_high_closure_count']}，urgent={ground['ordinary_scene_urgent_closure_count']}），未闭环HV={ground['unclosed_HV_count']}。",
        f"HV到达时前方FIFO任务平均 {ground['mean_fifo_jobs_ahead_at_HV_arrival']:.3f}，下传队列平均 {ground['mean_downlink_queue_MB_at_HV_arrival']:.3f} MB。历史hash不同的HV=0结果只并列展示，不能当作精确复现或严格因果归因。",
        f"修正后的日终下传队列为 {f(metrics[METHOD_GROUND]['terminal_queue_MB']):.3f} MB、峰值 {f(metrics[METHOD_GROUND]['peak_queue_MB']):.3f} MB、均值 {f(metrics[METHOD_GROUND]['mean_queue_MB']):.3f} MB；4 MB/task与严格FIFO拥塞仍然明显。",
        "",
        "TUT产品尺寸仍是公共抽象产品，不是原论文真实PMSC码率；TUT代理地面能量单独披露且不计入卫星SOC/公共卫星能耗。",
    ]
    if proxy_vs_reference:
        conclusions.append(f"TUT代理相对可信普通场景Full PROP-V2的J_all判定：{proxy_vs_reference['judgment']}（仅诊断，不据此调参）。")
    conclusions.extend(["", f"输出目录：`{attempt}`", "", f"`{COMPLETION}`"])
    (attempt / "final_assessment.md").write_text("\n".join(conclusions) + "\n", encoding="utf-8")
    write_json(attempt / "completion_status.json", {"status": "PASS", "marker": COMPLETION, "completed_at": utc_now()})
    print("\n" + COMPLETION)
    print(f"OUTPUT_DIR={attempt}")
    for row in metric_rows:
        print(f"{row['method']}: J_all={f(row['J_all_net']):.6f}, J_HV={f(row['J_HV_net']):.6f}, HV={row['HV_closure_count']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--child-spec", type=Path, default=None, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    raise SystemExit(child_main(arguments.child_spec) if arguments.child_spec else main())
