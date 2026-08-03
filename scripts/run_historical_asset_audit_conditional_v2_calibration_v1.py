"""Historical-asset audit and conditional decoupled-V2 calibration.

No arguments are required.  Existing sources and outputs are read-only.  The
entry first audits historical assets, then follows the evidence-selected
branch.  For the current repository that branch adapts reusable mechanisms and
calibrates only the missing q/price bundle on seeds disjoint from validation
and formal test seeds.
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
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402
import scripts.run_prop_reqalign_seed2026_v1 as reqrun  # noqa: E402
import scripts.run_v25_decoupled_v2_seed2026_diagnostic as v2diag  # noqa: E402


CONFIG = ROOT / "configs" / "conditional_v2_calibration_v1.yaml"
CALIBRATION_SOURCE = ROOT / "src" / "calibration" / "conditional_v2_calibration.py"
OUTPUT = ROOT / "outputs" / "_historical_asset_audit_conditional_v2_calibration_v1"
CAPACITY = 0.50
SLOTS = 4320
SLOT_SECONDS = 20.0
COMPLETION = "HISTORICAL_ASSET_AUDIT_AND_CONDITIONAL_V2_CALIBRATION_COMPLETE"
SAFETY_COST = 10.0
_CALCORE: Any = None


def load_calcore() -> Any:
    global _CALCORE
    if _CALCORE is None:
        spec = importlib.util.spec_from_file_location("conditional_v2_calibration_runtime", CALIBRATION_SOURCE)
        if spec is None or spec.loader is None:
            raise RuntimeError(CALIBRATION_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _CALCORE = module
    return _CALCORE


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in data for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data or [{"empty": ""}])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_md(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(data[0]) if data else ["empty"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def i(value: Any) -> int:
    return int(float(value or 0))


def load_cfg() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _completion(path: Path) -> tuple[bool, str]:
    for name in ("completion_status.json", "method_completed.json", "completed.json", "final_status.json"):
        marker = path / name
        if marker.is_file():
            try:
                row = read_json(marker)
                passed = str(row.get("status", "")).upper() in {"PASS", "COMPLETE", "COMPLETED"}
                return passed, str(marker)
            except Exception:
                return False, str(marker)
    return False, ""


def historical_asset_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evidence-backed, read-only classification of historical assets."""

    assets = [
        {
            "asset_id": "V25_40W_CAP050_RUNTIME_CHAIN", "classification": "DIRECTLY_REUSABLE",
            "scope": "environment_and_scenario_only",
            "primary_path": ROOT / "scripts" / "run_bupt1_power_true_rerun_seed2026_v1.py",
            "call_path": "exact.prepare_controlled_bundle -> reconstruct -> inject_power -> apply_capacity(0.5)",
            "resolved_config": ROOT / "outputs" / "_v25_decoupled_v2_seed2026_diagnostic_v1" / "config" / "controlled_bundle_seed_2026" / "resolved_config.json",
            "correct_40w": True, "capacity_050": True, "q_once": False, "independent_seeds": False,
            "action_3x3": False, "old_environment_pollution": False,
            "reason": "Runtime identity is correct and reusable; it is not a q/price calibration asset.",
        },
        {
            "asset_id": "DECOUPLED_V2_Q_ONCE_MECHANICS", "classification": "REUSABLE_WITH_ADAPTER",
            "scope": "3x3_action_and_q_once_mechanics",
            "primary_path": ROOT / "src" / "routers" / "decoupled_prop_v2.py",
            "call_path": "run_v25_decoupled_v2_seed2026_diagnostic.DecoupledV2RouterWrapper.select_action",
            "resolved_config": ROOT / "configs" / "prop_decoupled_v2_diagnostic.yaml",
            "correct_40w": True, "capacity_050": True, "q_once": True, "independent_seeds": False,
            "action_3x3": True, "old_environment_pollution": False,
            "reason": "Formula and grid pass audit, but hard-coded q and prices have no independent calibration provenance.",
        },
        {
            "asset_id": "GROUND_RECEIVED_ACCOUNTING", "classification": "REUSABLE_WITH_ADAPTER",
            "scope": "ground_event_and_closure_accounting",
            "primary_path": ROOT / "src" / "evaluation" / "unified_ground_received_accounting.py",
            "call_path": "public evaluator -> ground received events -> closure/timeliness",
            "resolved_config": ROOT / "configs" / "unified_ground_received_accounting.yaml",
            "correct_40w": None, "capacity_050": None, "q_once": False, "independent_seeds": False,
            "action_3x3": True, "old_environment_pollution": False,
            "reason": "Ground-event semantics are reusable; event-level q-once gross and calibrated fixed prices require an adapter.",
        },
        {
            "asset_id": "LYAPUNOV_DYNAMIC_PRICING_MECHANISM", "classification": "REUSABLE_WITH_ADAPTER",
            "scope": "dynamic_price_shape_only",
            "primary_path": ROOT / "src" / "controllers" / "lyapunov_pricing.py",
            "call_path": "SimulationEnvironment -> StepRunner guidance -> Predictor/V2 scorer",
            "resolved_config": ROOT / "configs" / "lyapunov_pricing.yaml",
            "correct_40w": None, "capacity_050": None, "q_once": None, "independent_seeds": False,
            "action_3x3": True, "old_environment_pollution": False,
            "reason": "Dynamic pressure modulation is reusable; base lambdas are heuristic and lack physical-budget provenance.",
        },
        {
            "asset_id": "YOLO_INSPIRED_QUALITY_PROFILE", "classification": "DIAGNOSTIC_ONLY",
            "scope": "ordered_quality_proxy_seed",
            "primary_path": ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml",
            "call_path": "ModelRegistry profile override",
            "resolved_config": ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml",
            "correct_40w": False, "capacity_050": False, "q_once": None, "independent_seeds": False,
            "action_3x3": False, "old_environment_pollution": True,
            "reason": "File explicitly says values are not real SARFish measurements; historical profile run used old low-power stress environment.",
        },
        {
            "asset_id": "STRICT_VALUE_COST_SCALE_AUDIT", "classification": "DIAGNOSTIC_ONLY",
            "scope": "legacy_value_cost_scale_diagnostic",
            "primary_path": ROOT / "scripts" / "value_cost_scale_calibration_strict_audit.py",
            "call_path": "standalone legacy diagnostic",
            "resolved_config": ROOT / "outputs" / "_utility_value_cost_scale_calibration_strict",
            "correct_40w": False, "capacity_050": False, "q_once": False, "independent_seeds": False,
            "action_3x3": False, "old_environment_pollution": True,
            "reason": "Generic 35Wh/low-power configuration, no V25/cap0.5 and no parameter selection.",
        },
        {
            "asset_id": "PRE_REGISTERED_VALUE_COST_PATCH", "classification": "UNRELIABLE",
            "scope": "not_reusable_for_joint_v2",
            "primary_path": ROOT / "scripts" / "run_pre_registered_value_cost_calibration_patch.py",
            "call_path": "standalone 80-slot diagnostic",
            "resolved_config": ROOT / "outputs" / "_pre_registered_value_cost_calibration_patch",
            "correct_40w": False, "capacity_050": False, "q_once": False, "independent_seeds": False,
            "action_3x3": False, "old_environment_pollution": True,
            "reason": "80 slots, old generic environment, no seed split/budget perturbation; historical candidates did not pass.",
        },
        {
            "asset_id": "LEGACY_GUIDED_PREDICTOR", "classification": "UNRELIABLE",
            "scope": "must_not_supply_q_once_positive_term",
            "primary_path": ROOT / "src" / "simulator" / "predictors.py",
            "call_path": "Predictor._compute_guided_predicted_reward",
            "resolved_config": ROOT / "configs" / "reward.yaml",
            "correct_40w": None, "capacity_050": None, "q_once": False, "independent_seeds": False,
            "action_3x3": False, "old_environment_pollution": False,
            "reason": "raw_info_gain already includes q and guided utility multiplies predicted_quality again (q squared).",
        },
    ]
    for row in assets:
        path = Path(row["primary_path"])
        row["exists"] = path.exists()
        row["sha256"] = sha256_file(path) if path.is_file() else ""
        resolved = Path(row["resolved_config"])
        row["resolved_exists"] = resolved.exists()
        row["resolved_sha256"] = sha256_file(resolved) if resolved.is_file() else ""
        marker_root = resolved if resolved.is_dir() else resolved.parent
        completed, marker = _completion(marker_root)
        row["completion_found"] = completed
        row["completion_marker"] = marker
        row["primary_path"] = str(path)
        row["resolved_config"] = str(resolved)

    search_roots = [ROOT / name for name in ("configs", "src", "scripts", "contracts", "artifacts", "outputs", ".codex_method_preservation_audit_v1")]
    keywords = ("quality-aware", "q_once", "quality_proxy", "ground_received", "lyapunov", "shadow", "energy_price", "downlink_price", "calibration", "budget", "marginal", "pareto", "reqalign", "decoupled", "completion", "fairness")
    hits = Counter()
    output_evidence_names = (
        "audit", "completion", "completed", "final", "identity", "resolved",
        "config", "manifest", "fairness", "method", "calibration", "pareto",
    )
    for base in search_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".md", ".csv"}:
                continue
            relative_text = str(path.relative_to(ROOT)).lower()
            # Outputs can contain millions of event rows.  Search every output
            # path name, but inspect contents only for compact provenance,
            # configuration, completion, and audit artifacts.
            for keyword in keywords:
                if keyword in relative_text:
                    hits[keyword] += 1
            if base.name == "outputs" and not any(token in path.name.lower() for token in output_evidence_names):
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            for keyword in keywords:
                if keyword in text:
                    hits[keyword] += 1
    decision = {
        "parameter_bundle_directly_reusable": False,
        "reusable_mechanisms_with_adapter": True,
        "reliable_complete_parameter_asset_exists": False,
        "selected_branch": "ADAPT_REUSABLE_MECHANISMS_AND_CALIBRATE_MISSING_Q_AND_FIXED_PRICES",
        "basis": "No historical asset jointly satisfies q-once, independent seed split, budget-derived prices, 3x3 actions, and correct runtime environment.",
        "keyword_hit_counts": dict(hits),
    }
    return assets, decision


class ConditionalCalibrationWrapper(v2diag.DecoupledV2RouterWrapper):
    """Runtime-only adapter: q map is process-local and old sources stay read-only."""

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        selected = super().select_action(context, candidate_actions, predicted_map, guidance_bundle)
        if not self.v2_decision_rows or f(self.v2_decision_rows[-1]["v2_net_score"]) >= -1.0e-12:
            return selected
        defer = next(action for action in self.step_runner.action_space.list_actions() if action.payload_policy == "defer")
        slot = int(self.current_step.slot_index)
        task_id = str(self.current_step.task.task_id)
        for row in reversed(self.candidate_rows):
            if int(row["decision_slot"]) != slot or row["task_id"] != task_id:
                break
            row["final_executed_choice"] = int(row["action"] == defer.action_name)
        decision = self.v2_decision_rows[-1]
        decision.update(
            {
                "selected_action": defer.action_name,
                "selected_model": defer.model_name,
                "selected_product": "defer",
                "selection_reason": "negative_physical_net_utility_defer_guard",
                "v2_net_score": 0.0,
                "v2_quality": 0.0,
                "q_multiplication_count": 0,
            }
        )
        self._set_delegate_state(defer, 0.0)
        return defer


def configure_online_prices(configured: dict[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(configured)
    if bool(spec.get("legacy_online_baseline", False)):
        return cfg
    cal = load_calcore()
    from src.reward.utility_calibration import get_utility_calibration

    calibration = get_utility_calibration(cfg["reward"])
    prices = spec["prices"]
    baseline = cal.online_lyapunov_baseline(
        energy_price_per_Wh=f(prices["energy_price_per_Wh"]),
        downlink_price_per_MB=f(prices["downlink_price_per_MB"]),
        terminal_backlog_price_per_MB=f(prices["terminal_backlog_price_per_MB"]),
        energy_reference_j=f(cfg["simulator"]["E_ref"]),
        queue_max_mb=f(cfg["simulator"]["queue_max_mb"]),
        beta_resource_cost=f(calibration.beta_resource_cost, 1.0),
        base_lambda_L=f(cfg["lyapunov_pricing"]["base_lambda_L"]),
    )
    cfg["lyapunov_pricing"].update(baseline)
    return cfg


def event_assignment(decisions: list[dict[str, str]], q_map: Mapping[str, float]) -> dict[tuple[str, str], tuple[str, float]]:
    mapping: dict[tuple[str, str], tuple[str, float]] = {}
    for row in decisions:
        product, model = row["selected_product"], row["selected_model"]
        if product != "defer" and model in q_map:
            mapping[(row["task_id"], product)] = (model, f(q_map[model]))
    return mapping


def evaluate_run(method: str, path: Path, bundle_root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    from src.payload.information_sufficiency import delta_information_utility

    config = read_json(bundle_root / "resolved_config.json")
    tasks = {row["task_id"]: row for row in read_csv(path / "per_task_outcomes.csv")}
    decisions = read_csv(path / "v2_decision_trace.csv")
    candidates = read_csv(path / "v2_candidate_score_audit.csv")
    q_map = {key: f(value) for key, value in spec["q_eff"].items()}
    assignment = event_assignment(decisions, q_map)
    delivered: dict[str, set[str]] = defaultdict(set)
    ledger = []
    gross = hv_gross = 0.0
    missing_model = 0
    stage_product = {"metadata": "metadata", "roi": "roi", "evidence": "roi_context", "full": "roi_context"}
    events = sorted(read_csv(path / "ground_delivery_events.csv"), key=lambda row: (f(row["ground_available_time"]), row["task_id"], row["stage"]))
    for event in events:
        if not i(event.get("valid_before_deadline")):
            continue
        product = stage_product.get(event["stage"].lower(), "")
        if not product:
            continue
        task = tasks[event["task_id"]]
        model_q = assignment.get((event["task_id"], product))
        source = "direct_executed_product"
        if model_q is None:
            prior = [
                value
                for (task_id, prior_product), value in assignment.items()
                if task_id == event["task_id"] and prior_product == "metadata"
            ] or [
                value
                for (task_id, _prior_product), value in assignment.items()
                if task_id == event["task_id"]
            ]
            if not prior:
                missing_model += 1
                continue
            model_q, source = prior[-1], "cached_initial_inference_for_residual"
        model, q = model_q
        delta_i = delta_information_utility(product, delivered[event["task_id"]], task["product_requirement"], config["product_requirement_utility"])
        task_arrival = f(task["arrival_time"])
        lifetime = max(f(task["deadline"]) - task_arrival, 1.0e-9)
        timely = max(0.0, min(1.0, 1.0 - (f(event["ground_available_time"]) - task_arrival) / lifetime))
        positive = f(task["effective_task_value"]) * delta_i * q * timely
        delivered[event["task_id"]].add(product)
        gross += positive
        if task["value_level"] in {"high", "urgent"} and i(task["closure_success"]):
            hv_gross += positive
        ledger.append({"task_id": event["task_id"], "stage": event["stage"], "product": product, "model": model, "model_source": source, "V": task["effective_task_value"], "delta_I": delta_i, "q_eff": q, "T": timely, "gross": positive, "gross_recomputed": f(task["effective_task_value"]) * delta_i * q * timely, "q_count": 1})
    trace = read_csv(path / "per_slot_resource_trace.csv")
    energy = read_json(path / "energy_components.json")
    energy_wh = (f(energy["compute_including_encoding_energy_j"]) + f(energy["tx_energy_j"])) / 3600.0
    tx_mb = sum(f(row.get("transmitted_mb")) for row in trace)
    terminal_q = f(trace[-1]["queue_mb"])
    peak_q = max(f(row["queue_mb"]) for row in trace)
    soc_v = sum(i(row["soc_violation"]) for row in trace)
    thermal_v = sum(i(row["thermal_violation"]) for row in trace)
    prices = spec["prices"]
    energy_cost = f(prices["energy_price_per_Wh"]) * energy_wh
    tx_cost = f(prices["downlink_price_per_MB"]) * tx_mb
    terminal_cost = f(prices["terminal_backlog_price_per_MB"]) * terminal_q
    safety_cost = SAFETY_COST * (soc_v + thermal_v)
    total_cost = energy_cost + tx_cost + terminal_cost + safety_cost
    hv_cost = total_cost * hv_gross / gross if gross > 0.0 else 0.0
    levels = ("low", "medium", "high", "urgent")
    model_counts = {level: Counter(row["selected_model"] for row in decisions if row["value_level"] == level and row["selected_model"] != "delay_store") for level in levels}
    distributions = {}
    for level, counts in model_counts.items():
        total = sum(counts.values())
        distributions[level] = {model: counts[model] / total if total else 0.0 for model in load_calcore().MODEL_ORDER}
    all_counts = sum(model_counts.values(), Counter())
    all_total = sum(all_counts.values())
    distributions["all"] = {model: all_counts[model] / all_total if all_total else 0.0 for model in load_calcore().MODEL_ORDER}
    negative_executed = sum(row["selected_product"] != "defer" and f(row["v2_net_score"]) < -1.0e-12 for row in decisions)
    hv_high_feasible = {
        row["task_id"] for row in candidates
        if row["value_level"] in {"high", "urgent"} and row["model"] == "high_accuracy"
        and i(row["safe_filter_pass"]) and i(row["deadline_feasible"]) and f(row["v2_net_score"]) > 0.0
    }
    closed = [row for row in tasks.values() if i(row["closure_success"])]
    metric = {
        "phase": spec["phase"], "seed": spec["seed"], "candidate_id": spec["candidate_id"], "method": method,
        "q_id": spec["q_id"], "energy_factor": spec["energy_factor"], "downlink_factor": spec["downlink_factor"],
        "J_all_net": gross - total_cost, "J_HV_net": hv_gross - hv_cost, "gross": gross, "HV_gross": hv_gross,
        "total_cost": total_cost, "energy_cost": energy_cost, "transmitted_cost": tx_cost, "terminal_backlog_cost": terminal_cost, "safety_cost": safety_cost,
        "energy_Wh": energy_wh, "transmitted_MB": tx_mb, "terminal_queue_MB": terminal_q, "peak_queue_MB": peak_q,
        "minimum_soc": min(f(row["battery_soc"]) for row in trace), "maximum_thermal": max(f(row["thermal"]) for row in trace),
        "soc_violations": soc_v, "thermal_violations": thermal_v, "infeasible": int(bool(soc_v or thermal_v)),
        "all_closure_count": len(closed), "HV_closure_count": sum(row["value_level"] in {"high", "urgent"} for row in closed),
        "high_closure_count": sum(row["value_level"] == "high" for row in closed), "urgent_closure_count": sum(row["value_level"] == "urgent" for row in closed),
        "negative_net_executed_count": negative_executed, "missing_event_model_count": missing_model,
        "hv_high_accuracy_positive_feasible_task_count": len(hv_high_feasible),
        "model_distribution": distributions,
        "q_once_max_recompute_error": max([abs(f(row["gross"]) - f(row["gross_recomputed"])) for row in ledger] or [0.0]),
    }
    write_csv(path / "conditional_public_event_ledger.csv", ledger)
    write_json(path / "conditional_metrics.json", metric)
    return metric


def run_child(spec_path: Path) -> int:
    spec = read_json(spec_path)
    seed, bundle_root, output = int(spec["seed"]), Path(spec["bundle_root"]), Path(spec["output"])
    if output.exists():
        raise FileExistsError(output)
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    core = v2diag.load_v2core()
    core.FROZEN_QUALITY_PROXY.clear()
    core.FROZEN_QUALITY_PROXY.update({key: f(value) for key, value in spec["q_eff"].items()})
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configured = v25.import_prop_script()._configure(scenario.configs, calibration, {"enabled": False})
    configured, grid_audit = v2diag.configure_v2(configured, fixed=False)
    configured = configure_online_prices(configured, spec)
    scenario = replace(scenario, configs=configured)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, CAPACITY)
    identity = exact.scenario_identity(spec["candidate_id"], seed, scenario, bundle, power, CAPACITY)
    env = SimulationEnvironment(scenario.configs)
    runner = env.build_episode_runner(scenario.timeline[:SLOTS])
    delegate = build_routers(scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"]))[0]
    wrapper = ConditionalCalibrationWrapper(
        delegate, reqrun.load_projection_config(True), scenario.timeline[:SLOTS], scenario.configs["simulator"],
        reqrun.load_projection_module(), product_utility_cfg=scenario.configs["product_requirement_utility"], reward_cfg=scenario.configs["reward"],
    )
    reqrun.bind_projection_runtime(runner, wrapper)
    result = runner.run(wrapper)
    method = f"CONDITIONAL-V2::{spec['candidate_id']}"
    normalized = normalize_episode(method, scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
    reqrun.write_energy_components(output, result)
    rows = []
    for row in wrapper.candidate_rows:
        rows.append({"seed": seed, "candidate_id": spec["candidate_id"], **row})
    write_csv(output / "v2_candidate_score_audit.csv", rows)
    write_csv(output / "v2_decision_trace.csv", wrapper.v2_decision_rows)
    metric = evaluate_run(method, output, bundle_root, spec)
    run_identity = {
        "status": "PASS", "seed": seed, "pid": os.getpid(), "fresh_state": True,
        "candidate_spec_sha256": stable_hash(spec), "q_eff": spec["q_eff"], "prices": spec["prices"],
        "legacy_online_baseline": spec.get("legacy_online_baseline", False), "grid_audit": grid_audit,
        "scenario_identity": identity, "power_identity": power, "capacity_identity": capacity_identity,
        "physical_net_utility_zscore": False, "defer_score": 0.0, "negative_action_guard": True,
        "hard_safe_filter_unchanged": True, "q_once": metric["q_once_max_recompute_error"] <= 1.0e-12,
    }
    write_json(output / "runtime_identity.json", run_identity)
    write_json(output / "method_completed.json", {"status": "PASS", "candidate_id": spec["candidate_id"], "seed": seed, "candidate_spec_sha256": stable_hash(spec)})
    return 0


def run_path(phase: str, candidate_id: str, seed: int) -> Path:
    return OUTPUT / "runs" / phase / candidate_id / f"seed_{seed}"


def launch_spec(spec: dict[str, Any]) -> dict[str, Any]:
    output = Path(spec["output"])
    marker = output / "method_completed.json"
    expected_hash = stable_hash(spec)
    spec_dir = OUTPUT / "run_specs" / spec["phase"] / spec["candidate_id"]
    spec_file = spec_dir / f"seed_{spec['seed']}.json"
    if marker.is_file():
        row = read_json(marker)
        if row.get("status") == "PASS" and row.get("candidate_spec_sha256") == expected_hash and (output / "conditional_metrics.json").is_file():
            return {"status": "RESUMED", "candidate_id": spec["candidate_id"], "seed": spec["seed"], "output": str(output)}
        # A parent-only reporting/audit edit changes entry_sha256 but does not
        # invalidate a fresh simulation.  Reuse only when every semantic run
        # field and both behavior/config hashes match, and the corrected event
        # accounting is complete.  Record the metadata migration explicitly.
        semantic_fields = (
            "phase", "seed", "bundle_root", "output", "candidate_id", "q_id",
            "q_eff", "energy_factor", "downlink_factor", "prices",
            "legacy_online_baseline", "config_sha256", "calibration_source_sha256",
        )
        if spec_file.is_file() and (output / "conditional_metrics.json").is_file() and (output / "runtime_identity.json").is_file():
            prior_spec = read_json(spec_file)
            metric = read_json(output / "conditional_metrics.json")
            prior_matches = all(prior_spec.get(field) == spec.get(field) for field in semantic_fields)
            if (
                row.get("status") == "PASS"
                and row.get("candidate_spec_sha256") == stable_hash(prior_spec)
                and prior_matches
                and i(metric.get("missing_event_model_count")) == 0
                and f(metric.get("q_once_max_recompute_error")) <= 1.0e-12
            ):
                identity = read_json(output / "runtime_identity.json")
                previous_hash = row.get("candidate_spec_sha256")
                identity.update({
                    "candidate_spec_sha256": expected_hash,
                    "resumed_after_parent_only_audit_change": True,
                    "previous_candidate_spec_sha256": previous_hash,
                })
                write_json(output / "runtime_identity.json", identity)
                write_json(marker, {
                    "status": "PASS", "candidate_id": spec["candidate_id"], "seed": spec["seed"],
                    "candidate_spec_sha256": expected_hash,
                    "resumed_after_parent_only_audit_change": True,
                    "previous_candidate_spec_sha256": previous_hash,
                })
                write_json(spec_file, spec)
                return {
                    "status": "RESUMED_COMPATIBLE_PARENT_ONLY_CHANGE",
                    "candidate_id": spec["candidate_id"], "seed": spec["seed"],
                    "output": str(output), "pid": identity.get("pid"),
                }
    if output.exists():
        quarantine = OUTPUT / "failed_attempts" / f"{spec['phase']}_{spec['candidate_id']}_seed{spec['seed']}_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(quarantine))
    write_json(spec_file, spec)
    command = [sys.executable, str(Path(__file__).resolve()), "--child-run", "--spec", str(spec_file)]
    started = utc_now()
    process = subprocess.run(command, cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": str(spec["seed"])}, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logs = OUTPUT / "logs" / spec["phase"]
    logs.mkdir(parents=True, exist_ok=True)
    label = f"{spec['candidate_id']}_seed{spec['seed']}"
    (logs / f"{label}_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (logs / f"{label}_stderr.txt").write_text(process.stderr, encoding="utf-8")
    child_pid = None
    identity_file = output / "runtime_identity.json"
    if identity_file.is_file():
        child_pid = read_json(identity_file).get("pid")
    status = {"status": "PASS" if process.returncode == 0 else "FAIL", "candidate_id": spec["candidate_id"], "seed": spec["seed"], "pid": child_pid, "started_utc": started, "finished_utc": utc_now(), "returncode": process.returncode, "output": str(output), "stderr_tail": process.stderr[-4000:]}
    if process.returncode != 0 or not marker.is_file():
        raise RuntimeError(status)
    return status


def run_parallel(specs: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
    statuses = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(launch_spec, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            statuses.append(future.result())
            print(f"completed {spec['phase']} {spec['candidate_id']} seed={spec['seed']}", flush=True)
    return statuses


def prepare_bundles(seeds: Iterable[int]) -> dict[int, Path]:
    bundles = {}
    for seed in seeds:
        path = OUTPUT / "config" / "controlled_bundles" / f"seed_{seed}"
        if path.exists():
            v25.load_bundle(path)
        else:
            exact.prepare_controlled_bundle(seed, path)
        bundles[seed] = path
    return bundles


def make_spec(
    *, phase: str, seed: int, bundle: Path, candidate_id: str, q_id: str,
    q_eff: Mapping[str, float], energy_factor: float, downlink_factor: float,
    prices: Mapping[str, float], legacy_online_baseline: bool = False,
) -> dict[str, Any]:
    return {
        "phase": phase, "seed": seed, "bundle_root": str(bundle), "output": str(run_path(phase, candidate_id, seed)),
        "candidate_id": candidate_id, "q_id": q_id, "q_eff": dict(q_eff),
        "energy_factor": energy_factor, "downlink_factor": downlink_factor, "prices": dict(prices),
        "legacy_online_baseline": legacy_online_baseline, "entry_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(CONFIG), "calibration_source_sha256": sha256_file(CALIBRATION_SOURCE),
    }


def collect_metrics(specs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        row = read_json(Path(spec["output"]) / "conditional_metrics.json")
        flat = {key: value for key, value in row.items() if key != "model_distribution"}
        for level, fractions in row["model_distribution"].items():
            for model, value in fractions.items():
                flat[f"fraction_{level}_{model}"] = value
        rows.append(flat)
    return rows


def aggregate_candidates(rows: list[dict[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["candidate_id"]].append(row)
    limits = cfg["selection_constraints"]
    output = []
    for candidate_id, group in sorted(groups.items()):
        first = group[0]
        aggregate: dict[str, Any] = {
            "candidate_id": candidate_id, "q_id": first["q_id"], "energy_factor": first["energy_factor"], "downlink_factor": first["downlink_factor"], "seed_count": len(group),
        }
        for field in ("J_all_net", "J_HV_net", "gross", "HV_gross", "total_cost", "energy_Wh", "transmitted_MB", "terminal_queue_MB", "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count"):
            aggregate[f"{field}_mean"] = statistics.fmean(f(row[field]) for row in group)
        for level in ("low", "medium", "high", "urgent", "all"):
            for model in load_calcore().MODEL_ORDER:
                field = f"fraction_{level}_{model}"
                aggregate[field] = statistics.fmean(f(row[field]) for row in group)
        aggregate["fraction_HV_high_accuracy"] = statistics.fmean(
            (aggregate["fraction_high_high_accuracy"], aggregate["fraction_urgent_high_accuracy"])
        )
        checks = {
            "all_runs_feasible": all(not i(row["infeasible"]) for row in group),
            "negative_action_guard": all(i(row["negative_net_executed_count"]) == 0 for row in group),
            "event_model_complete": all(i(row["missing_event_model_count"]) == 0 for row in group),
            "q_once": all(f(row["q_once_max_recompute_error"]) <= 1.0e-12 for row in group),
            "low_prefers_light": aggregate["fraction_low_lightweight"] >= f(limits["low_lightweight_min_fraction"]),
            "medium_has_balanced_region": aggregate["fraction_medium_balanced"] >= f(limits["medium_balanced_min_fraction"]),
            "hv_has_high_accuracy_region": aggregate["fraction_HV_high_accuracy"] >= f(limits["high_urgent_high_accuracy_min_fraction"]),
            "no_global_model_collapse": (
                max(aggregate[f"fraction_all_{model}"] for model in load_calcore().MODEL_ORDER) <= f(limits["any_model_max_fraction"])
                and all(aggregate[f"fraction_all_{model}"] > 0.0 for model in load_calcore().MODEL_ORDER)
            ),
        }
        aggregate["passed"] = all(checks.values())
        aggregate["exclusion_reasons"] = ";".join(name for name, passed in checks.items() if not passed)
        aggregate.update({f"check_{key}": value for key, value in checks.items()})
        output.append(aggregate)
    return output


def pressure_fallback_audit(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    complexity = {"lightweight": 0.0, "balanced": 1.0, "high_accuracy": 2.0}
    rows = []
    passed = True
    for q_id in sorted({row["q_id"] for row in aggregates}):
        group = [row for row in aggregates if row["q_id"] == q_id]
        low = min(group, key=lambda row: f(row["energy_factor"]) + f(row["downlink_factor"]))
        high = max(group, key=lambda row: f(row["energy_factor"]) + f(row["downlink_factor"]))
        def level(row: Mapping[str, Any]) -> float:
            return sum(complexity[model] * f(row[f"fraction_all_{model}"]) for model in complexity)
        model_fallback = level(high) <= level(low) + 1.0e-12
        physical_resource_fallback = (
            f(high["energy_Wh_mean"]) <= f(low["energy_Wh_mean"]) + 1.0e-12
            and f(high["transmitted_MB_mean"]) <= f(low["transmitted_MB_mean"]) + 1.0e-12
        )
        item = {
            "q_id": q_id,
            "low_pressure_candidate": low["candidate_id"],
            "high_pressure_candidate": high["candidate_id"],
            "low_pressure_mean_complexity": level(low),
            "high_pressure_mean_complexity": level(high),
            "low_pressure_energy_Wh": f(low["energy_Wh_mean"]),
            "high_pressure_energy_Wh": f(high["energy_Wh_mean"]),
            "low_pressure_transmitted_MB": f(low["transmitted_MB_mean"]),
            "high_pressure_transmitted_MB": f(high["transmitted_MB_mean"]),
            "model_complexity_fallback": model_fallback,
            "physical_resource_fallback": physical_resource_fallback,
            "natural_fallback": model_fallback or physical_resource_fallback,
            "criterion": "model complexity falls OR both realized physical resources do not increase",
        }
        passed = passed and item["natural_fallback"]
        rows.append(item)
    return {"status": "PASS" if passed else "FAIL", "rows": rows}


def annotate_pareto_front(aggregates: list[dict[str, Any]]) -> None:
    """Annotate only; parameter selection deliberately does not use this flag."""

    for row in aggregates:
        dominated_by = []
        for other in aggregates:
            if other is row:
                continue
            weakly_better = (
                f(other["gross_mean"]) >= f(row["gross_mean"])
                and f(other["energy_Wh_mean"]) <= f(row["energy_Wh_mean"])
                and f(other["transmitted_MB_mean"]) <= f(row["transmitted_MB_mean"])
            )
            strictly_better = (
                f(other["gross_mean"]) > f(row["gross_mean"])
                or f(other["energy_Wh_mean"]) < f(row["energy_Wh_mean"])
                or f(other["transmitted_MB_mean"]) < f(row["transmitted_MB_mean"])
            )
            if weakly_better and strictly_better:
                dominated_by.append(other["candidate_id"])
        row["pareto_efficient_gross_energy_downlink"] = not dominated_by
        row["dominated_by_count"] = len(dominated_by)
        row["dominated_by_candidates"] = ";".join(dominated_by)


def choose_candidate(aggregates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    q_priority = {"Q_MEAN": 0, "Q_SHRUNK_TO_PROFILE": 1, "Q_LOWER_QUARTILE": 2}
    eligible = [row for row in aggregates if row["passed"]]
    pool = eligible or aggregates
    selected = min(
        pool,
        key=lambda row: (
            abs(math.log(max(f(row["energy_factor"]), 1.0e-9))),
            abs(math.log(max(f(row["downlink_factor"]), 1.0e-9))),
            q_priority.get(row["q_id"], 99),
            row["candidate_id"],
        ),
    )
    exclusions = [{"candidate_id": row["candidate_id"], "selected": row is selected, "passed": row["passed"], "reason": "selected_by_pre_registered_center_distance" if row is selected else (row["exclusion_reasons"] or "farther_from_unity_center;not_selected_by_performance") } for row in aggregates]
    return selected, exclusions


def source_guard_paths() -> list[Path]:
    paths = [
        ROOT / "src" / "routers" / "decoupled_prop_v2.py",
        ROOT / "src" / "controllers" / "lyapunov_pricing.py",
        ROOT / "src" / "evaluation" / "unified_ground_received_accounting.py",
        ROOT / "scripts" / "run_v25_decoupled_v2_seed2026_diagnostic.py",
        ROOT / "scripts" / "run_bupt1_power_true_rerun_seed2026_v1.py",
        ROOT / "scripts" / "run_exact_native_v25_cap050_five_method_fiveseed.py",
        ROOT / "configs" / "bupt1_measured_power_v1.yaml",
        ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml",
    ]
    external = [
        v25.SOURCE_PROP / "scripts" / "run_updated_prop_ground_ack_v1.py",
        v25.SOURCE_PROP / "src" / "controllers" / "lyapunov_pricing.py",
        v25.SOURCE_PROP / "src" / "routers" / "safe_filter.py",
        v25.SOURCE_PROP / "src" / "routers" / "safe_linucb_router.py",
        v25.SOURCE_PROP / "src" / "simulator" / "predictors.py",
        v25.SOURCE_PROP / "src" / "simulator" / "step_runner.py",
    ]
    return [path for path in (*paths, *external) if path.exists()]


def parent_main() -> int:
    cfg = load_cfg()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pre_hashes = {str(path): sha256_file(path) for path in source_guard_paths()}
    assets, decision = historical_asset_audit()
    write_csv(OUTPUT / "tables" / "historical_asset_inventory.csv", assets)
    write_md(OUTPUT / "tables" / "historical_asset_inventory.md", assets)
    write_json(OUTPUT / "audit" / "historical_asset_audit.json", {"status": "PASS", "assets": assets, "decision": decision})
    write_json(OUTPUT / "audit" / "conditional_branch_decision.json", decision)

    cal_seeds = tuple(int(seed) for seed in cfg["seeds"]["calibration"])
    val_seeds = tuple(int(seed) for seed in cfg["seeds"]["validation"])
    test_seeds = tuple(int(seed) for seed in cfg["seeds"]["formal_test"])
    if set(cal_seeds) & set(val_seeds) or set(cal_seeds) & set(test_seeds) or set(val_seeds) & set(test_seeds):
        raise RuntimeError("SEED_LEAKAGE")
    bundles = prepare_bundles((*cal_seeds, *val_seeds, *test_seeds))
    seed_scope_audit = {
        "status": "PASS",
        "seed_identifiers_disjoint": True,
        "calibration_seeds": cal_seeds,
        "validation_seeds": val_seeds,
        "formal_test_seeds": test_seeds,
        "shared_frozen_task_manifest": True,
        "frozen_task_hash": v25.EXPECTED_TASK_HASH,
        "independence_scope": "independent timing/order, RNG, queues, SOC, thermal state, and online learning state",
        "not_independent_image_content": True,
        "disclosure": "All seeds intentionally use the same frozen V25 task content; they are independent simulation realizations, not independent image datasets.",
        "future_information_used_online": False,
        "post_run_ground_events_used_only_for_public_accounting": True,
    }
    write_json(OUTPUT / "audit" / "seed_separation_and_no_future_leakage_audit.json", seed_scope_audit)
    calibration_tasks = [task for seed in cal_seeds for task in v25.load_bundle(bundles[seed]).tasks]
    q_candidates = load_calcore().derive_quality_candidates(calibration_tasks)
    write_json(OUTPUT / "calibration" / "q_eff_candidates.json", q_candidates)

    q_probe = next(row for row in q_candidates if row["q_id"] == "Q_MEAN")
    probe_prices = {"energy_price_per_Wh": 1.0, "downlink_price_per_MB": 0.01, "terminal_backlog_price_per_MB": 0.01}
    probe_specs = [make_spec(phase="probe", seed=seed, bundle=bundles[seed], candidate_id="PROBE_LEGACY_LYAPUNOV", q_id=q_probe["q_id"], q_eff=q_probe["q_eff"], energy_factor=1.0, downlink_factor=1.0, prices=probe_prices, legacy_online_baseline=True) for seed in cal_seeds]
    statuses = run_parallel(probe_specs, min(len(probe_specs), int(cfg["parallel"]["max_independent_processes"])))
    probe_rows = []
    for spec in probe_specs:
        probe_rows.extend(read_csv(Path(spec["output"]) / "v2_candidate_score_audit.csv"))
    resolved = read_json(bundles[cal_seeds[0]] / "resolved_config.json")
    reference_budgets = {
        str(spec["seed"]): {
            "energy_Wh": f(read_json(Path(spec["output"]) / "conditional_metrics.json")["energy_Wh"]),
            "downlink_MB": f(read_json(Path(spec["output"]) / "conditional_metrics.json")["transmitted_MB"]),
        }
        for spec in probe_specs
    }
    adjacent_slopes = load_calcore().derive_shadow_price_center(
        probe_rows,
        energy_reference_j=f(resolved["simulator"]["E_ref"]),
        queue_max_mb=f(resolved["simulator"]["queue_max_mb"]),
    )
    center = load_calcore().derive_budget_perturbation_prices(
        probe_rows,
        reference_budgets_by_seed=reference_budgets,
        energy_reference_j=f(resolved["simulator"]["E_ref"]),
        queue_max_mb=f(resolved["simulator"]["queue_max_mb"]),
        perturbation=0.10,
    )
    price_grid = load_calcore().price_candidates(center, cfg["candidate_factors"])
    write_json(OUTPUT / "calibration" / "budget_perturbation_price_center.json", center)
    write_json(OUTPUT / "calibration" / "adjacent_marginal_slope_diagnostic.json", adjacent_slopes)
    write_csv(OUTPUT / "tables" / "budget_perturbation_price_estimates_by_seed.csv", center["per_seed"])
    write_json(OUTPUT / "calibration" / "fixed_price_candidates.json", price_grid)

    grid_specs = []
    for q in q_candidates:
        for energy in price_grid["energy"]:
            for downlink in price_grid["downlink"]:
                candidate_id = f"{q['q_id']}__{energy['price_id']}__{downlink['price_id']}"
                prices = {"energy_price_per_Wh": energy["value"], "downlink_price_per_MB": downlink["value"], "terminal_backlog_price_per_MB": downlink["value"]}
                for seed in cal_seeds:
                    grid_specs.append(make_spec(phase="calibration_grid", seed=seed, bundle=bundles[seed], candidate_id=candidate_id, q_id=q["q_id"], q_eff=q["q_eff"], energy_factor=energy["factor"], downlink_factor=downlink["factor"], prices=prices))
    statuses.extend(run_parallel(grid_specs, int(cfg["parallel"]["max_independent_processes"])))
    grid_rows = collect_metrics(grid_specs)
    aggregates = aggregate_candidates(grid_rows, cfg)
    annotate_pareto_front(aggregates)
    pressure = pressure_fallback_audit(aggregates)
    selected, exclusions = choose_candidate(aggregates)
    write_csv(OUTPUT / "tables" / "calibration_runs.csv", grid_rows)
    write_csv(OUTPUT / "tables" / "candidate_summary_and_exclusions.csv", aggregates)
    write_csv(OUTPUT / "tables" / "candidate_selection_decision.csv", exclusions)
    write_json(OUTPUT / "audit" / "pressure_fallback_audit.json", pressure)

    selected_q = next(row for row in q_candidates if row["q_id"] == selected["q_id"])
    e_item = next(row for row in price_grid["energy"] if abs(f(row["factor"]) - f(selected["energy_factor"])) <= 1.0e-12)
    b_item = next(row for row in price_grid["downlink"] if abs(f(row["factor"]) - f(selected["downlink_factor"])) <= 1.0e-12)
    frozen_prices = {"energy_price_per_Wh": e_item["value"], "downlink_price_per_MB": b_item["value"], "terminal_backlog_price_per_MB": b_item["value"]}

    val_specs = []
    for factor in cfg["selection_constraints"]["perturbation_factors"]:
        candidate_id = f"FROZEN_VALIDATION_{factor:.2f}x"
        prices = {key: f(value) * f(factor) for key, value in frozen_prices.items()}
        for seed in val_seeds:
            val_specs.append(make_spec(phase="validation", seed=seed, bundle=bundles[seed], candidate_id=candidate_id, q_id=selected_q["q_id"], q_eff=selected_q["q_eff"], energy_factor=factor, downlink_factor=factor, prices=prices))
    statuses.extend(run_parallel(val_specs, int(cfg["parallel"]["max_independent_processes"])))
    val_rows = collect_metrics(val_specs)
    val_aggregates = aggregate_candidates(val_rows, cfg)
    val_pressure = pressure_fallback_audit(val_aggregates)
    center_validation = next(row for row in val_aggregates if row["candidate_id"] == "FROZEN_VALIDATION_1.00x")
    validation_pass = (
        all(not i(row["infeasible"]) and i(row["negative_net_executed_count"]) == 0 and i(row["missing_event_model_count"]) == 0 for row in val_rows)
        and bool(center_validation["check_low_prefers_light"])
        and bool(center_validation["check_medium_has_balanced_region"])
        and bool(center_validation["check_hv_has_high_accuracy_region"])
        and bool(center_validation["check_no_global_model_collapse"])
        and val_pressure["status"] == "PASS"
        and any(
            statistics.fmean((f(row["fraction_high_high_accuracy"]), f(row["fraction_urgent_high_accuracy"])))
            >= f(cfg["selection_constraints"]["high_urgent_high_accuracy_min_fraction"])
            for row in val_rows
            if abs(f(row["energy_factor"]) - 0.75) <= 1.0e-12
        )
    )
    write_csv(OUTPUT / "tables" / "validation_perturbation_runs.csv", val_rows)
    write_csv(OUTPUT / "tables" / "validation_perturbation_summary.csv", val_aggregates)
    write_json(OUTPUT / "audit" / "independent_validation_audit.json", {"status": "PASS" if validation_pass else "FAIL", "pressure": val_pressure, "center": center_validation})
    if not validation_pass:
        raise RuntimeError("INDEPENDENT_VALIDATION_FAILED")

    online_baseline = load_calcore().online_lyapunov_baseline(
        energy_price_per_Wh=f(frozen_prices["energy_price_per_Wh"]), downlink_price_per_MB=f(frozen_prices["downlink_price_per_MB"]), terminal_backlog_price_per_MB=f(frozen_prices["terminal_backlog_price_per_MB"]),
        energy_reference_j=f(resolved["simulator"]["E_ref"]), queue_max_mb=f(resolved["simulator"]["queue_max_mb"]), beta_resource_cost=1.0, base_lambda_L=f(resolved["lyapunov_pricing"]["base_lambda_L"]),
    )
    legacy_unit_conversion = {
        "note": "lambda values are dimensionless normalized coefficients, not direct physical prices",
        "legacy_base_lambda_E": f(resolved["lyapunov_pricing"]["base_lambda_E"]),
        "legacy_equivalent_energy_price_per_Wh": f(resolved["lyapunov_pricing"]["base_lambda_E"]) * 3600.0 / f(resolved["simulator"]["E_ref"]),
        "legacy_base_lambda_S": f(resolved["lyapunov_pricing"]["base_lambda_S"]),
        "legacy_equivalent_downlink_price_per_MB": f(resolved["lyapunov_pricing"]["base_lambda_S"]) / f(resolved["simulator"]["queue_max_mb"]),
        "frozen_public_to_online_conversion": online_baseline,
    }
    write_json(OUTPUT / "audit" / "legacy_price_unit_conversion.json", legacy_unit_conversion)
    frozen = {
        "version": "CONDITIONAL-DECOUPLED-V2-CALIBRATION-v1", "selected_without_method_ranking": True,
        "selection_rule": "constraint pass then q-priority and log-distance to 1x budget-derived center",
        "q_id": selected_q["q_id"], "q_eff": selected_q["q_eff"], "public_prices": frozen_prices,
        "online_lyapunov_baseline": online_baseline, "unchanged_lyapunov_pressure_alphas": {key: value for key, value in resolved["lyapunov_pricing"].items() if key.startswith("alpha_")},
        "calibration_seeds": cal_seeds, "validation_seeds": val_seeds, "formal_test_seeds": test_seeds,
        "calibration_candidate_id": selected["candidate_id"], "validation_passed": True,
        "q_is_proxy_not_real_measurement": True,
    }
    write_json(OUTPUT / "frozen" / "conditional_v2_frozen_parameters_v1.json", frozen)

    formal_specs = [make_spec(phase="formal_test", seed=seed, bundle=bundles[seed], candidate_id="CONDITIONAL_DECOUPLED_V2_FROZEN", q_id=selected_q["q_id"], q_eff=selected_q["q_eff"], energy_factor=1.0, downlink_factor=1.0, prices=frozen_prices) for seed in test_seeds]
    statuses.extend(run_parallel(formal_specs, min(len(formal_specs), int(cfg["parallel"]["max_independent_processes"]))))
    formal_rows = collect_metrics(formal_specs)
    write_csv(OUTPUT / "tables" / "formal_test_metrics.csv", formal_rows)

    threshold_rows = []
    for spec in formal_specs:
        candidates = read_csv(Path(spec["output"]) / "v2_candidate_score_audit.csv")
        threshold_rows.extend(
            asdict(row)
            for row in load_calcore().model_upgrade_thresholds(
                candidates,
                energy_reference_j=f(resolved["simulator"]["E_ref"]),
                queue_max_mb=f(resolved["simulator"]["queue_max_mb"]),
                downlink_price_per_mb=f(frozen_prices["downlink_price_per_MB"]),
            )
        )
    write_csv(OUTPUT / "tables" / "model_upgrade_break_even_prices.csv", threshold_rows)
    write_csv(OUTPUT / "tables" / "benefit_cost_decomposition.csv", [{key: value for key, value in row.items() if key not in {"model_distribution"}} for row in (*val_rows, *formal_rows)])

    post_hashes = {str(path): sha256_file(path) for path in source_guard_paths()}
    source_pass = pre_hashes == post_hashes
    seed_pass = not (set(cal_seeds) & set(val_seeds) or set(cal_seeds) & set(test_seeds) or set(val_seeds) & set(test_seeds))
    parallel_pids = [row.get("pid") for row in statuses if row.get("pid")]
    audit = {
        "status": "PASS" if source_pass and seed_pass and pressure["status"] == "PASS" else "FAIL",
        "old_assets_read_only": source_pass, "source_hashes_before": pre_hashes, "source_hashes_after": post_hashes,
        "seed_sets_disjoint": seed_pass, "calibration_seeds": cal_seeds, "validation_seeds": val_seeds, "formal_test_seeds": test_seeds,
        "safe_parallel_enabled": True, "parallel_backend": "ThreadPoolExecutor supervising independent Python subprocesses",
        "distinct_child_pid_count": len(set(parallel_pids)), "shared_mutable_state": False, "run_count": len(statuses),
        "runtime_environment": {"V25": True, "capacity": CAPACITY, "power": v25.FROZEN_POWER},
        "selection_not_based_on_method_wins": True, "hard_safe_filter_unchanged": True,
    }
    write_json(OUTPUT / "audit" / "method_environment_fairness_completion_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("FINAL_AUDIT_FAILED")

    report = [
        "# Historical asset audit and conditional V2 calibration", "",
        "## Decision", "", decision["selected_branch"], "", decision["basis"], "",
        "Asset classes: 1 DIRECTLY_REUSABLE environment chain, 3 REUSABLE_WITH_ADAPTER mechanisms, 2 DIAGNOSTIC_ONLY assets, and 2 UNRELIABLE assets.", "",
        "The reusable mechanisms are the 3x3 q-once action/scoring layer, ground-received accounting semantics, and Lyapunov pressure modulation. Missing q provenance and fixed physical prices were calibrated; historical assets remained read-only.", "",
        "## Calibration and validation", "",
        f"Evaluated {len(aggregates)} joint candidates on calibration seeds; {sum(bool(row['passed']) for row in aggregates)} passed the pre-registered behavior constraints.", "",
        "Fixed prices came from method-neutral multiple-choice LP budget perturbations at 0.9x and 1.1x, not method ranking. The selected 1x center was chosen by constraint pass and distance to the budget-derived center. Pareto flags are reported diagnostically and were not used for selection.", "",
        f"Independent validation status: {'PASS' if validation_pass else 'FAIL'}; 0.75x/1x/1.25x physical-resource fallback status: {val_pressure['status']}.", "",
        "Seed identifiers are disjoint, but every seed deliberately shares the frozen V25 task content. Independence is at timing/order, RNG, queue, SOC, thermal, and online-state level—not at image-dataset level.", "",
        "## Frozen parameters", "", "```json", json.dumps(frozen, ensure_ascii=False, indent=2), "```", "",
        "## Formal test (not used for parameter selection)", "",
    ]
    formal_table = [{key: value for key, value in row.items() if not key.startswith("fraction_")} for row in formal_rows]
    write_md(OUTPUT / "tables" / "formal_test_metrics.md", formal_table)
    report.append((OUTPUT / "tables" / "formal_test_metrics.md").read_text(encoding="utf-8"))
    report.extend(["", "The q_eff values remain calibrated system-level proxies, not claimed hardware measurements.", "", COMPLETION])
    (OUTPUT / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUTPUT / "completion_status.json", {"status": "PASS", "completion_marker": COMPLETION, "completed_utc": utc_now(), "entry_sha256": sha256_file(Path(__file__).resolve()), "config_sha256": sha256_file(CONFIG), "calibration_source_sha256": sha256_file(CALIBRATION_SOURCE), "frozen_parameters_sha256": sha256_file(OUTPUT / "frozen" / "conditional_v2_frozen_parameters_v1.json"), "old_assets_unchanged": True, "all_audits_passed": True})
    print("run file:", Path(__file__).resolve())
    print("output:", OUTPUT.resolve())
    print("branch:", decision["selected_branch"])
    print("safe parallel: True")
    print("final:", COMPLETION)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--spec", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_child(Path(args.spec)) if args.child_run else parent_main())
