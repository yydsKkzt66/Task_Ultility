"""Standalone c_k mode-control ablation experiment.

This runner is intentionally self-contained and default-off. It reuses the
existing benchmark trace builders plus public/private helper functions from the
standalone experiment scripts, but it does not modify src algorithms, value
prior logic, high-value labels, rewards, evaluators, or older runners.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_fair_baseline_comparison as fair  # noqa: E402
import scripts.run_low_initial_soc_emergency_stress as low_soc  # noqa: E402
import scripts.run_price_guided_rescue_ground_delivery_experiment as price  # noqa: E402
import scripts.run_realistic_workload_contact_sensitivity as realistic  # noqa: E402
import scripts.run_uneven_pressure_ground_delivery_experiment as base  # noqa: E402
from src.common.types import LongTermState  # noqa: E402
from src.controllers.lyapunov_pricing import LyapunovPricing  # noqa: E402
from src.controllers.mode_controller import RuleBasedModeController  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_ck_mode_control_ablation_v1"
DEFAULT_SEEDS = [0, 1, 2]
FULL_LOW_SOC_SEEDS = [0, 1, 2, 3, 4]
VARIANTS = [
    "proposed_full_ck",
    "proposed_no_ck",
    "proposed_static_balanced_ck",
    "proposed_ck_only_no_lyapunov",
    "old_proposed",
]
CK_VARIANTS = set(VARIANTS) - {"old_proposed"}
MODE_ORDER = [
    "balanced",
    "energy_saving",
    "queue_draining",
    "latency_critical",
    "mission_priority",
    "egress_conservative",
    "no_mode",
]

SUMMARY_COLUMNS = [
    "variant",
    "seed",
    "scenario_name",
    "workload_scenario",
    "contact_scenario",
    "initial_SOC",
    "task_manifest_hash",
    "contact_schedule_hash",
    "resource_trace_hash",
    "value_prior_hash",
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "net_system_utility",
    "H_net_utility",
    "expired_task_count",
    "expired_high_value_task_count",
    "avg_ground_received_quality",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "min_SOC",
    "final_SOC",
    "SOC_margin_to_20",
    "time_below_50_percent",
    "time_below_40_percent",
    "time_below_25_percent",
    "hard_safety_violation_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
    "queue_MB_peak",
    "final_backlog_count",
    "max_backlog_count",
]

METRICS = [
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "net_system_utility",
    "H_net_utility",
    "expired_task_count",
    "expired_high_value_task_count",
    "avg_ground_received_quality",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "min_SOC",
    "final_SOC",
    "SOC_margin_to_20",
    "time_below_50_percent",
    "time_below_40_percent",
    "time_below_25_percent",
    "hard_safety_violation_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
    "queue_MB_peak",
    "final_backlog_count",
    "max_backlog_count",
]
LOWER_IS_BETTER = {
    "expired_task_count",
    "expired_high_value_task_count",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "time_below_50_percent",
    "time_below_40_percent",
    "time_below_25_percent",
    "hard_safety_violation_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
    "queue_MB_peak",
    "final_backlog_count",
    "max_backlog_count",
}
EFFECT_METRICS = [
    "ground_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_high_value_success_value",
    "net_system_utility",
    "H_net_utility",
    "expired_high_value_task_count",
    "min_SOC",
    "max_thermal_pressure",
    "queue_MB_peak",
]
TRACE_COLUMNS = [
    "seed",
    "scenario_name",
    "workload_scenario",
    "contact_scenario",
    "variant",
    "slot",
    "orbit_index",
    "mode_name",
    "battery_soc",
    "thermal_pressure",
    "queue_mb",
    "queue_pressure",
    "recent_deadline_urgency",
    "recent_high_value_ratio",
    "next_contact_s",
    "queue_growth_rate",
    "lambda_E_base",
    "lambda_Q_base",
    "lambda_L_base",
    "lambda_S_base",
    "lambda_E_final",
    "lambda_Q_final",
    "lambda_L_final",
    "lambda_S_final",
    "lambda_E_multiplier",
    "lambda_Q_multiplier",
    "lambda_L_multiplier",
    "lambda_S_multiplier",
    "exploration_scale",
    "high_value_bonus_scale",
    "safety_margin_scale",
    "model_complexity_bias",
    "payload_size_bias",
]


def _to_float(value: object, default: float = 0.0) -> float:
    return base._to_float(value, default)


def _to_int(value: object, default: int = 0) -> int:
    return base._to_int(value, default)


def _safe_div(num: float, den: float) -> float:
    return base._safe_div(num, den)


def _fmt(value: object, digits: int = 4) -> str:
    return f"{_to_float(value):.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return dict(yaml.safe_load(file) or {})


def _group(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def _thermal_pressure(wave: str, orbit_compute: float = 0.0, queue_mb: float = 0.0, payload_on: bool = False) -> float:
    return price._thermal_pressure(wave, orbit_compute) + 0.0025 * min(queue_mb, 60.0) + (0.04 if payload_on else 0.0)


def _value_prior_hash(manifest: list[dict[str, Any]]) -> str:
    columns = ["task_id", "base_prior_score", "value_level", "value_multiplier", "V_eff", "value_assignment_reason"]
    return fair._stable_hash(manifest, columns)


def _scenario_hashes(manifest: list[dict[str, Any]], contacts: list[dict[str, Any]], resources: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "task_manifest_hash": fair._stable_hash(manifest, realistic.REALISTIC_MANIFEST_COLUMNS),
        "contact_schedule_hash": fair._stable_hash(
            contacts, ["seed", "workload_scenario", "contact_scenario", "orbit_index", "capacity_MB", "start_time_h"]
        ),
        "resource_trace_hash": fair._stable_hash(
            resources, ["seed", "workload_scenario", "contact_scenario", "orbit_index", "solar_energy_wh", "housekeeping_energy_wh"]
        ),
        "value_prior_hash": _value_prior_hash(manifest),
    }


def _next_contact_s(orbit: int, contacts: dict[int, dict[str, Any]]) -> float:
    future = [idx for idx in contacts if idx >= orbit]
    if not future:
        return 999999.0
    return max(0.0, (min(future) - orbit) * base.ORBIT_HOURS * 3600.0)


def _recent_deadline_urgency(ready_tasks: list[dict[str, Any]], orbit: int) -> float:
    if not ready_tasks:
        return 0.0
    now = orbit * base.ORBIT_HOURS
    values = []
    for task in ready_tasks:
        slack_orbits = (_to_float(task["deadline_h"]) - now) / max(base.ORBIT_HOURS, 1.0e-9)
        values.append(max(0.0, min(1.0, (3.0 - slack_orbits) / 3.0)))
    return statistics.mean(values)


def _recent_high_value_ratio(ready_tasks: list[dict[str, Any]]) -> float:
    return _safe_div(sum(_to_int(task.get("is_high_value")) for task in ready_tasks), len(ready_tasks))


def _build_guidance(
    *,
    variant: str,
    long_state: LongTermState,
    pricing: LyapunovPricing,
    controller: RuleBasedModeController,
) -> dict[str, Any]:
    if variant == "proposed_ck_only_no_lyapunov":
        base_prices = {"lambda_E": 0.12, "lambda_Q": 0.10, "lambda_L": 0.10, "lambda_S": 0.10}
    else:
        base_prices = pricing.compute_base_prices(long_state)
    if variant == "proposed_no_ck":
        return {
            "mode_name": "no_mode",
            "lambda_base": dict(base_prices),
            "lambda_E": base_prices["lambda_E"],
            "lambda_Q": base_prices["lambda_Q"],
            "lambda_L": base_prices["lambda_L"],
            "lambda_S": base_prices["lambda_S"],
            "price_multipliers": {"lambda_E": 1.0, "lambda_Q": 1.0, "lambda_L": 1.0, "lambda_S": 1.0},
            "exploration_scale": 1.0,
            "high_value_bonus_scale": 1.0,
            "safety_margin_scale": 1.0,
            "model_complexity_bias": 0.0,
            "payload_size_bias": 0.0,
        }
    mode_name = "balanced" if variant == "proposed_static_balanced_ck" else controller.select_mode(long_state)
    return controller.build_guidance(base_prices, mode_name)


def _guidance_trace_row(
    *,
    seed: int,
    scenario_name: str,
    workload: str,
    contact: str,
    variant: str,
    orbit: int,
    long_state: LongTermState,
    guidance: dict[str, Any],
) -> dict[str, Any]:
    lb = dict(guidance.get("lambda_base", {}))
    mult = dict(guidance.get("price_multipliers", {}))
    return {
        "seed": seed,
        "scenario_name": scenario_name,
        "workload_scenario": workload,
        "contact_scenario": contact,
        "variant": variant,
        "slot": orbit,
        "orbit_index": orbit,
        "mode_name": guidance.get("mode_name", ""),
        "battery_soc": long_state.battery_soc,
        "thermal_pressure": long_state.thermal_pressure,
        "queue_mb": long_state.queue_mb,
        "queue_pressure": _safe_div(long_state.queue_mb, long_state.queue_max_mb),
        "recent_deadline_urgency": long_state.recent_deadline_urgency,
        "recent_high_value_ratio": long_state.recent_high_value_ratio,
        "next_contact_s": long_state.next_contact_s,
        "queue_growth_rate": long_state.queue_growth_rate,
        "lambda_E_base": lb.get("lambda_E", 0.0),
        "lambda_Q_base": lb.get("lambda_Q", 0.0),
        "lambda_L_base": lb.get("lambda_L", 0.0),
        "lambda_S_base": lb.get("lambda_S", 0.0),
        "lambda_E_final": guidance.get("lambda_E", 0.0),
        "lambda_Q_final": guidance.get("lambda_Q", 0.0),
        "lambda_L_final": guidance.get("lambda_L", 0.0),
        "lambda_S_final": guidance.get("lambda_S", 0.0),
        "lambda_E_multiplier": mult.get("lambda_E", 1.0),
        "lambda_Q_multiplier": mult.get("lambda_Q", 1.0),
        "lambda_L_multiplier": mult.get("lambda_L", 1.0),
        "lambda_S_multiplier": mult.get("lambda_S", 1.0),
        "exploration_scale": guidance.get("exploration_scale", 1.0),
        "high_value_bonus_scale": guidance.get("high_value_bonus_scale", 1.0),
        "safety_margin_scale": guidance.get("safety_margin_scale", 1.0),
        "model_complexity_bias": guidance.get("model_complexity_bias", 0.0),
        "payload_size_bias": guidance.get("payload_size_bias", 0.0),
    }


def _score_candidate(
    *,
    task: dict[str, Any],
    delivered: set[str],
    model: str,
    product: str,
    orbit: int,
    soc: float,
    capacity_mb: float,
    queue_count: int,
    queue_mb: float,
    wave: str,
    resource: dict[str, Any],
    guidance: dict[str, Any],
) -> dict[str, Any]:
    payload = base._product_payload(task, product)
    compute, comm, account_cost = price._candidate_costs(model, product, payload)
    estimated_energy = compute + comm
    soc_after = soc + (
        _to_float(resource["solar_energy_wh"])
        - _to_float(resource["housekeeping_energy_wh"])
        - price._payload_idle_wh(True)
        - estimated_energy
        - base.TX_FIXED_WH_PER_ACTIVE_ORBIT
    ) / base.BATTERY_CAPACITY_WH
    thermal_after = _thermal_pressure(wave, compute, queue_mb, True)
    safety_margin = max(0.0, _to_float(guidance.get("safety_margin_scale", 1.0)) - 1.0) * 0.01
    hard_ok = soc_after >= price.BATTERY_HARD_FLOOR + price.BATTERY_SELECTION_MARGIN + safety_margin
    hard_ok = hard_ok and thermal_after <= price.THERMAL_HARD_LIMIT
    deadline_slack, deadline_bucket = price._deadline_info(task, orbit, product, payload)
    if deadline_bucket == "expired":
        hard_ok = False

    delivered_before = set(delivered)
    delta = fair.delta_information_utility(product, delivered_before, str(task["product_requirement"]))
    quality = price._model_quality(model, product)
    predicted = price._predicted_ground_delivery_success(payload, capacity_mb, deadline_bucket, soc_after)
    value_level = str(task["value_level"])
    high_value_scale = _to_float(guidance.get("high_value_bonus_scale", 1.0)) if value_level in {"high", "urgent"} else 1.0
    expected = _to_float(task["V_eff"]) * delta * quality * predicted * price._timeliness_factor(deadline_bucket) * high_value_scale
    queue_delay_cost = queue_count / 10.0 + queue_mb / 25.0
    lambda_e = _to_float(guidance.get("lambda_E"))
    lambda_s = _to_float(guidance.get("lambda_S"))
    lambda_q = _to_float(guidance.get("lambda_Q"))
    lambda_l = _to_float(guidance.get("lambda_L"))
    energy_price_cost = lambda_e * estimated_energy
    comm_price_cost = lambda_s * (comm + 0.08 * payload)
    queue_cost = lambda_q * queue_delay_cost * (1.0 + 0.15 * payload)
    thermal_cost = 0.08 * lambda_e * compute * (1.0 if model == "Medium" else 0.55 if model == "Small" else 0.25)
    deadline_urgency_bonus = price._deadline_bonus(task, delivered_before, product, deadline_bucket) * (1.0 + 0.6 * lambda_l)
    density = expected / (estimated_energy + 0.35 * payload + 0.08 * queue_delay_cost + 1.0e-9)
    product_pref = {"metadata": 1.0, "roi": 0.42, "roi_context": 0.05}[product]
    density_threshold = {"metadata": 0.075, "roi": 0.11, "roi_context": 0.28}[product]
    value_pref = 1.25 if value_level in {"low", "medium"} else 0.75
    cheap_win_density_bonus = 0.38 * product_pref * value_pref * price._clip(density - density_threshold, 0.0, 1.8)
    model_bias = _to_float(guidance.get("model_complexity_bias", 0.0)) * {"Tiny": -1.0, "Small": 0.0, "Medium": 1.0}[model]
    payload_bias = _to_float(guidance.get("payload_size_bias", 0.0)) * payload
    exploration_scale = _to_float(guidance.get("exploration_scale", 1.0))
    score = (
        expected
        - energy_price_cost
        - comm_price_cost
        - queue_cost
        - thermal_cost
        + deadline_urgency_bonus
        + cheap_win_density_bonus * exploration_scale
        + model_bias
        + payload_bias
        - account_cost
    )
    return {
        "task_id": task["task_id"],
        "scene_id": task["scene_id"],
        "value_level": value_level,
        "model": model,
        "product": product,
        "stage": price.STAGE_BY_PRODUCT[product],
        "payload": payload,
        "compute": compute,
        "comm": comm,
        "account_cost": account_cost,
        "estimated_energy": estimated_energy,
        "soc_after": soc_after,
        "thermal_after": thermal_after,
        "deadline_slack": deadline_slack,
        "deadline_bucket": deadline_bucket,
        "predicted_ground_delivery_success": predicted,
        "expected_ground_delivery_utility": expected,
        "energy_price_cost": energy_price_cost,
        "comm_price_cost": comm_price_cost,
        "queue_cost": queue_cost,
        "thermal_cost": thermal_cost,
        "deadline_urgency_bonus": deadline_urgency_bonus,
        "cheap_win_density_bonus": cheap_win_density_bonus,
        "delivery_density": density,
        "score": score,
        "hard_ok": int(hard_ok),
        "lambda_E": lambda_e,
        "lambda_Q": lambda_q,
        "lambda_L": lambda_l,
        "lambda_S": lambda_s,
        "reason": "safe_ck_score" if hard_ok else "filtered_by_hard_safety_or_expired",
    }


def _target_success(row: dict[str, Any]) -> int:
    target_time = row.get("target_ground_receive_time_h", "")
    return int(target_time not in {"", None} and _to_float(target_time) <= _to_float(row.get("deadline_h")) + 1.0e-12)


def run_ck_variant(
    *,
    variant: str,
    seed: int,
    scenario_name: str,
    workload: str,
    contact: str,
    initial_soc: float,
    manifest: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    hashes: dict[str, str],
    pricing: LyapunovPricing,
    controller: RuleBasedModeController,
) -> dict[str, Any]:
    by_task = {str(row["task_id"]): row for row in manifest}
    arrivals: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in manifest:
        arrivals[_to_int(task["orbit_index"])].append(task)
    contact_by_orbit = {_to_int(row["orbit_index"]): row for row in contacts}
    resource_by_orbit = {_to_int(row["orbit_index"]): row for row in resources}
    delivered: dict[str, set[str]] = defaultdict(set)
    active: set[str] = set()
    events: list[dict[str, Any]] = []
    task_rows: dict[str, dict[str, Any]] = {}
    trajectory: list[dict[str, Any]] = []
    action_log: list[dict[str, Any]] = []
    mode_trace: list[dict[str, Any]] = []
    cumulative_direct_energy = 0.0
    cumulative_system_energy = 0.0
    net_system_utility = 0.0
    h_net_utility = 0.0
    soc = initial_soc
    prev_queue_mb = 0.0

    for task in manifest:
        task_rows[str(task["task_id"])] = {
            "variant": variant,
            "seed": seed,
            "scenario_name": scenario_name,
            "workload_scenario": workload,
            "contact_scenario": contact,
            "task_id": task["task_id"],
            "scene_id": task["scene_id"],
            "subtask_id": task["subtask_id"],
            "capture_time_h": task["capture_time_h"],
            "first_ground_receive_time_h": "",
            "target_ground_receive_time_h": "",
            "deadline_h": task["deadline_h"],
            "ground_received_flag": 0,
            "ground_received_before_deadline_flag": 0,
            "target_stage_ground_received_flag": 0,
            "ground_received_quality": 0.0,
            "highest_ground_received_stage": "none",
            "target_stage": task["target_stage"],
            "min_required_stage": task["min_required_stage"],
            "capture_to_ground_latency_h": "",
            "task_energy_direct_wh": 0.0,
            "task_energy_system_wh": 0.0,
            "is_high_value": task["is_high_value"],
            "value_level": task["value_level"],
            "V_eff": task["V_eff"],
        }

    for orbit in range(base.NUM_ORBITS):
        time_h = orbit * base.ORBIT_HOURS
        resource = resource_by_orbit[orbit]
        wave = str(resource["wave_name"])
        for task in arrivals.get(orbit, []):
            active.add(str(task["task_id"]))
        for task_id in sorted(list(active)):
            if time_h > _to_float(by_task[task_id]["deadline_h"]):
                active.discard(task_id)

        ready_tasks = [by_task[task_id] for task_id in sorted(active) if time_h <= _to_float(by_task[task_id]["deadline_h"])]
        queue_count = sum(1 for task in ready_tasks if price._candidate_products(task, delivered[str(task["task_id"])]))
        queue_mb = sum(
            min([base._product_payload(task, product) for product in price._candidate_products(task, delivered[str(task["task_id"])])] or [0.0])
            for task in ready_tasks
        )
        capacity = _to_float(contact_by_orbit[orbit]["capacity_MB"]) if orbit in contact_by_orbit else 0.0
        deadline_urgency = _recent_deadline_urgency(ready_tasks, orbit)
        long_state = LongTermState(
            battery_soc=soc,
            battery_safe=0.70,
            thermal_pressure=_thermal_pressure(wave, 0.0, queue_mb, False),
            thermal_max=price.THERMAL_HARD_LIMIT,
            queue_mb=queue_mb,
            queue_max_mb=36.0,
            recent_deadline_urgency=deadline_urgency,
            recent_high_value_ratio=_recent_high_value_ratio(ready_tasks),
            next_contact_s=_next_contact_s(orbit, contact_by_orbit),
            low_contact_capacity_factor=price._clip(1.0 - capacity / 5.0, 0.0, 1.0),
            queue_growth_rate=queue_mb - prev_queue_mb,
        )
        guidance = _build_guidance(variant=variant, long_state=long_state, pricing=pricing, controller=controller)
        mode_trace.append(
            _guidance_trace_row(
                seed=seed,
                scenario_name=scenario_name,
                workload=workload,
                contact=contact,
                variant=variant,
                orbit=orbit,
                long_state=long_state,
                guidance=guidance,
            )
        )

        candidates: list[dict[str, Any]] = []
        if orbit in contact_by_orbit:
            for task in ready_tasks:
                task_id = str(task["task_id"])
                for product in price._candidate_products(task, delivered[task_id]):
                    for model in price.MODEL_CHOICES:
                        candidates.append(
                            _score_candidate(
                                task=task,
                                delivered=delivered[task_id],
                                model=model,
                                product=product,
                                orbit=orbit,
                                soc=soc,
                                capacity_mb=capacity,
                                queue_count=queue_count,
                                queue_mb=queue_mb,
                                wave=wave,
                                resource=resource,
                                guidance=guidance,
                            )
                        )

        for task in ready_tasks:
            task_id = str(task["task_id"])
            products = price._candidate_products(task, delivered[task_id])
            payload = min([base._product_payload(task, product) for product in products] or [0.0])
            slack, bucket = price._deadline_info(task, orbit, "metadata", payload or 0.1)
            action_log.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "scenario_name": scenario_name,
                    "workload_scenario": workload,
                    "contact_scenario": contact,
                    "orbit_index": orbit,
                    "task_id": task_id,
                    "candidate_stage": "defer",
                    "candidate_model": "None",
                    "selected_action": 0,
                    "deadline_bucket": bucket,
                    "deadline_slack_h": slack,
                    "score": 0.0,
                    "lambda_E_final": guidance["lambda_E"],
                    "lambda_Q_final": guidance["lambda_Q"],
                    "lambda_L_final": guidance["lambda_L"],
                    "lambda_S_final": guidance["lambda_S"],
                }
            )

        selected: list[dict[str, Any]] = []
        selected_task_ids: set[str] = set()
        remain = capacity
        selected_energy = 0.0
        max_selected = 8
        for cand in sorted(candidates, key=lambda row: _to_float(row["score"]), reverse=True):
            if not _to_int(cand["hard_ok"]):
                continue
            if cand["task_id"] in selected_task_ids:
                continue
            if _to_float(cand["payload"]) > remain + 1.0e-12:
                continue
            incremental_energy = _to_float(cand["compute"]) + _to_float(cand["comm"])
            if len(selected) >= max_selected:
                break
            if _to_float(cand["score"]) <= 0.0 and selected:
                continue
            projected_energy = selected_energy + incremental_energy
            projected_soc = soc + (
                _to_float(resource["solar_energy_wh"])
                - _to_float(resource["housekeeping_energy_wh"])
                - price._payload_idle_wh(True)
                - projected_energy
                - base.TX_FIXED_WH_PER_ACTIVE_ORBIT
            ) / base.BATTERY_CAPACITY_WH
            if projected_soc < price.BATTERY_HARD_FLOOR + price.BATTERY_SELECTION_MARGIN:
                continue
            selected.append(cand)
            selected_task_ids.add(str(cand["task_id"]))
            remain -= _to_float(cand["payload"])
            selected_energy = projected_energy

        selected_keys = {(str(c["task_id"]), str(c["model"]), str(c["product"])) for c in selected}
        orbit_compute = sum(_to_float(c["compute"]) for c in selected)
        orbit_comm = sum(_to_float(c["comm"]) for c in selected)
        payload_on = bool(selected)
        payload_idle = price._payload_idle_wh(payload_on)
        tx_fixed = base.TX_FIXED_WH_PER_ACTIVE_ORBIT if selected else 0.0
        before_soc = soc
        net_energy = _to_float(resource["solar_energy_wh"]) - _to_float(resource["housekeeping_energy_wh"]) - payload_idle - orbit_compute - orbit_comm - tx_fixed
        soc = max(price.BATTERY_HARD_FLOOR, min(1.0, soc + net_energy / base.BATTERY_CAPACITY_WH))

        for cand in candidates:
            key = (str(cand["task_id"]), str(cand["model"]), str(cand["product"]))
            action_log.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "scenario_name": scenario_name,
                    "workload_scenario": workload,
                    "contact_scenario": contact,
                    "orbit_index": orbit,
                    "task_id": cand["task_id"],
                    "candidate_stage": cand["stage"],
                    "candidate_model": cand["model"],
                    "selected_action": int(key in selected_keys),
                    "deadline_bucket": cand["deadline_bucket"],
                    "deadline_slack_h": cand["deadline_slack"],
                    "score": cand["score"],
                    "lambda_E_final": cand["lambda_E"],
                    "lambda_Q_final": cand["lambda_Q"],
                    "lambda_L_final": cand["lambda_L"],
                    "lambda_S_final": cand["lambda_S"],
                }
            )

        product_counts = Counter()
        system_energy = orbit_compute + orbit_comm + payload_idle
        cumulative_system_energy += system_energy
        cumulative_direct_energy += orbit_compute + orbit_comm
        for cand in selected:
            task = by_task[str(cand["task_id"])]
            product = str(cand["product"])
            stage = price.STAGE_BY_PRODUCT[product]
            product_counts[product] += 1
            delivered_before = set(delivered[str(task["task_id"])])
            delta = fair.delta_information_utility(product, delivered_before, str(task["product_requirement"]))
            delivered[str(task["task_id"])].add(product)
            highest = base._highest_stage(delivered[str(task["task_id"])])
            quality = price._model_quality(str(cand["model"]), product) * base._stage_sufficiency(str(task["product_requirement"]), highest)
            gross = _to_float(task["V_eff"]) * delta * quality
            net = gross - _to_float(cand["account_cost"])
            net_system_utility += net
            if _to_int(task["is_high_value"]):
                h_net_utility += net
            receive_time = (orbit + 1) * base.ORBIT_HOURS
            before_deadline = int(receive_time <= _to_float(task["deadline_h"]) + 1.0e-12)
            events.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "scenario_name": scenario_name,
                    "workload_scenario": workload,
                    "contact_scenario": contact,
                    "task_id": task["task_id"],
                    "scene_id": task["scene_id"],
                    "delivered_stage": stage,
                    "model_choice": cand["model"],
                    "product_size_MB": cand["payload"],
                    "ground_receive_time_h": receive_time,
                    "before_deadline_flag": before_deadline,
                    "delivered_quality": quality,
                    "incremental_information_utility": delta,
                }
            )
            trow = task_rows[str(task["task_id"])]
            trow["highest_ground_received_stage"] = highest
            trow["ground_received_quality"] = max(_to_float(trow["ground_received_quality"]), quality)
            trow["task_energy_direct_wh"] = _to_float(trow["task_energy_direct_wh"]) + _to_float(cand["estimated_energy"])
            trow["task_energy_system_wh"] = _to_float(trow["task_energy_system_wh"]) + _safe_div(system_energy, len(selected))
            if base._stage_rank(highest) >= base._stage_rank(str(task["min_required_stage"])) and not _to_int(trow["ground_received_flag"]):
                trow["ground_received_flag"] = 1
                trow["first_ground_receive_time_h"] = receive_time
                trow["ground_received_before_deadline_flag"] = before_deadline
                trow["capture_to_ground_latency_h"] = receive_time - _to_float(task["capture_time_h"])
                active.discard(str(task["task_id"]))
            if base._stage_rank(highest) >= base._stage_rank(str(task["target_stage"])):
                trow["target_stage_ground_received_flag"] = before_deadline
                if trow["target_ground_receive_time_h"] == "":
                    trow["target_ground_receive_time_h"] = receive_time

        trajectory.append(
            {
                "variant": variant,
                "seed": seed,
                "scenario_name": scenario_name,
                "workload_scenario": workload,
                "contact_scenario": contact,
                "orbit_index": orbit,
                "time_h": time_h,
                "wave_name": wave,
                "SOC": soc,
                "thermal_pressure": _thermal_pressure(wave, orbit_compute, queue_mb, payload_on),
                "backlog_count": queue_count,
                "backlog_MB": queue_mb,
                "metadata_count": product_counts["metadata"],
                "roi_count": product_counts["roi"],
                "roi_context_count": product_counts["roi_context"],
                "defer_count": max(0, queue_count - len(selected)),
                "compute_energy_wh": orbit_compute,
                "comm_energy_wh": orbit_comm,
                "payload_idle_energy_wh": payload_idle,
            }
        )
        prev_queue_mb = queue_mb

    task_delivery = list(task_rows.values())
    summary = _build_summary(
        variant=variant,
        seed=seed,
        scenario_name=scenario_name,
        workload=workload,
        contact=contact,
        initial_soc=initial_soc,
        hashes=hashes,
        task_delivery=task_delivery,
        trajectory=trajectory,
        cumulative_direct_energy=cumulative_direct_energy,
        cumulative_system_energy=cumulative_system_energy,
        net_system_utility=net_system_utility,
        h_net_utility=h_net_utility,
    )
    return {"summary": summary, "events": events, "task_delivery": task_delivery, "trajectory": trajectory, "action_log": action_log, "mode_trace": mode_trace}


def _build_summary(
    *,
    variant: str,
    seed: int,
    scenario_name: str,
    workload: str,
    contact: str,
    initial_soc: float,
    hashes: dict[str, str],
    task_delivery: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    cumulative_direct_energy: float,
    cumulative_system_energy: float,
    net_system_utility: float,
    h_net_utility: float,
) -> dict[str, Any]:
    success = [row for row in task_delivery if _to_int(row["ground_received_before_deadline_flag"])]
    target = [row for row in task_delivery if _target_success(row)]
    high_tasks = [row for row in task_delivery if _to_int(row["is_high_value"])]
    high_success = [row for row in high_tasks if _to_int(row["ground_received_before_deadline_flag"])]
    high_target = [row for row in high_tasks if _target_success(row)]
    expired = [row for row in task_delivery if not _to_int(row["ground_received_before_deadline_flag"])]
    soc_values = [_to_float(row["SOC"]) for row in trajectory]
    thermal_values = [_to_float(row["thermal_pressure"]) for row in trajectory]
    return {
        "variant": variant,
        "seed": seed,
        "scenario_name": scenario_name,
        "workload_scenario": workload,
        "contact_scenario": contact,
        "initial_SOC": initial_soc,
        **hashes,
        "ground_success_ratio": _safe_div(len(success), len(task_delivery)),
        "target_stage_success_ratio": _safe_div(len(target), len(task_delivery)),
        "high_value_ground_success_ratio": _safe_div(len(high_success), len(high_tasks)),
        "high_value_target_success_ratio": _safe_div(len(high_target), len(high_tasks)),
        "quality_weighted_success_value": sum(_to_float(row["V_eff"]) * _to_float(row["ground_received_quality"]) for row in success),
        "quality_weighted_high_value_success_value": sum(_to_float(row["V_eff"]) * _to_float(row["ground_received_quality"]) for row in high_success),
        "net_system_utility": net_system_utility,
        "H_net_utility": h_net_utility,
        "expired_task_count": len(expired),
        "expired_high_value_task_count": sum(_to_int(row["is_high_value"]) for row in expired),
        "avg_ground_received_quality": statistics.mean([_to_float(row["ground_received_quality"]) for row in success]) if success else 0.0,
        "avg_capture_to_ground_latency_h": statistics.mean([_to_float(row["capture_to_ground_latency_h"]) for row in success]) if success else 0.0,
        "direct_energy_per_received_task_wh": _safe_div(cumulative_direct_energy, len(success)),
        "system_energy_per_received_task_wh": _safe_div(cumulative_system_energy, len(success)),
        "min_SOC": min(soc_values) if soc_values else initial_soc,
        "final_SOC": soc_values[-1] if soc_values else initial_soc,
        "SOC_margin_to_20": (min(soc_values) - base.HARD_MIN_SOC) if soc_values else initial_soc - base.HARD_MIN_SOC,
        "time_below_50_percent": _safe_div(sum(value < 0.50 for value in soc_values), len(soc_values)),
        "time_below_40_percent": _safe_div(sum(value < 0.40 for value in soc_values), len(soc_values)),
        "time_below_25_percent": _safe_div(sum(value < 0.25 for value in soc_values), len(soc_values)),
        "hard_safety_violation_count": sum(value < base.HARD_MIN_SOC - 1.0e-12 for value in soc_values),
        "max_thermal_pressure": max(thermal_values) if thermal_values else 0.0,
        "avg_thermal_pressure": statistics.mean(thermal_values) if thermal_values else 0.0,
        "thermal_violation_count": sum(value > price.THERMAL_HARD_LIMIT for value in thermal_values),
        "queue_MB_peak": max([_to_float(row["backlog_MB"]) for row in trajectory] or [0.0]),
        "final_backlog_count": _to_int(trajectory[-1]["backlog_count"]) if trajectory else 0,
        "max_backlog_count": max([_to_int(row["backlog_count"]) for row in trajectory] or [0]),
    }


def _normalize_old_result(
    result: dict[str, Any],
    *,
    seed: int,
    scenario_name: str,
    workload: str,
    contact: str,
    initial_soc: float,
    hashes: dict[str, str],
) -> dict[str, Any]:
    for rowset_name in ("events", "task_delivery", "trajectory"):
        for row in result[rowset_name]:
            row["variant"] = "old_proposed"
            row["scenario_name"] = scenario_name
            row["workload_scenario"] = workload
            row["contact_scenario"] = contact
    trajectory = result["trajectory"]
    for row in trajectory:
        row["thermal_pressure"] = _thermal_pressure(str(row.get("wave_name", "")), _to_float(row.get("compute_energy_wh")), _to_float(row.get("backlog_MB")), bool(_to_int(row.get("payload_on"))))
        row["backlog_MB"] = row.get("backlog_MB", 0.0)
    summary = _build_summary(
        variant="old_proposed",
        seed=seed,
        scenario_name=scenario_name,
        workload=workload,
        contact=contact,
        initial_soc=initial_soc,
        hashes=hashes,
        task_delivery=result["task_delivery"],
        trajectory=trajectory,
        cumulative_direct_energy=sum(_to_float(row.get("compute_energy_wh")) + _to_float(row.get("comm_energy_wh")) for row in trajectory),
        cumulative_system_energy=sum(_to_float(row.get("compute_energy_wh")) + _to_float(row.get("comm_energy_wh")) + _to_float(row.get("payload_idle_energy_wh")) for row in trajectory),
        net_system_utility=_to_float(result["summary"].get("net_system_utility")),
        h_net_utility=_to_float(result["summary"].get("H_net_utility")),
    )
    return {"summary": summary, "events": result["events"], "task_delivery": result["task_delivery"], "trajectory": trajectory, "action_log": [], "mode_trace": []}


def _build_mode_distribution(mode_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for key, rows in _group(mode_trace, ["variant", "seed", "scenario_name", "workload_scenario", "contact_scenario"]).items():
        total = len(rows)
        by_mode = _group(rows, ["mode_name"])
        for (mode_name,), group in sorted(by_mode.items()):
            out.append(
                {
                    "variant": key[0],
                    "seed": key[1],
                    "scenario_name": key[2],
                    "workload_scenario": key[3],
                    "contact_scenario": key[4],
                    "mode_name": mode_name,
                    "mode_count": len(group),
                    "mode_fraction": _safe_div(len(group), total),
                    "mean_SOC_when_active": statistics.mean(_to_float(row["battery_soc"]) for row in group),
                    "mean_queue_pressure_when_active": statistics.mean(_to_float(row["queue_pressure"]) for row in group),
                    "mean_deadline_urgency_when_active": statistics.mean(_to_float(row["recent_deadline_urgency"]) for row in group),
                    "mean_high_value_ratio_when_active": statistics.mean(_to_float(row["recent_high_value_ratio"]) for row in group),
                    "mean_lambda_E_multiplier": statistics.mean(_to_float(row["lambda_E_multiplier"]) for row in group),
                    "mean_lambda_Q_multiplier": statistics.mean(_to_float(row["lambda_Q_multiplier"]) for row in group),
                    "mean_lambda_L_multiplier": statistics.mean(_to_float(row["lambda_L_multiplier"]) for row in group),
                    "mean_lambda_S_multiplier": statistics.mean(_to_float(row["lambda_S_multiplier"]) for row in group),
                }
            )
    return out


def _rank_variants(rows: list[dict[str, Any]], scenario_name: str, metric: str) -> dict[str, int]:
    source = [row for row in rows if row["scenario_name"] == scenario_name]
    means = []
    for (variant,), group in _group(source, ["variant"]).items():
        means.append((variant, statistics.mean(_to_float(row[metric]) for row in group)))
    ordered = sorted(means, key=lambda item: item[1], reverse=metric not in LOWER_IS_BETTER)
    return {variant: idx for idx, (variant, _value) in enumerate(ordered, start=1)}


def _build_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (variant, scenario), group in sorted(_group(rows, ["variant", "scenario_name"]).items()):
        for metric in METRICS:
            values = [_to_float(row[metric]) for row in group]
            ranks = []
            for seed in sorted({row["seed"] for row in group}):
                seed_rows = [row for row in rows if row["scenario_name"] == scenario and _to_int(row["seed"]) == _to_int(seed)]
                ordered = sorted(
                    [(row["variant"], _to_float(row[metric])) for row in seed_rows],
                    key=lambda item: item[1],
                    reverse=metric not in LOWER_IS_BETTER,
                )
                ranks.extend(idx for idx, (name, _value) in enumerate(ordered, start=1) if name == variant)
            out.append(
                {
                    "variant": variant,
                    "scenario_name": scenario,
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "rank_mean": statistics.mean(ranks) if ranks else 0.0,
                    "rank_std": statistics.stdev(ranks) if len(ranks) > 1 else 0.0,
                }
            )
    return out


def _build_effect_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scenario in sorted({row["scenario_name"] for row in rows}):
        for metric in EFFECT_METRICS:
            full = [_to_float(row[metric]) for row in rows if row["scenario_name"] == scenario and row["variant"] == "proposed_full_ck"]
            no = [_to_float(row[metric]) for row in rows if row["scenario_name"] == scenario and row["variant"] == "proposed_no_ck"]
            full_mean = statistics.mean(full) if full else 0.0
            no_mean = statistics.mean(no) if no else 0.0
            gain = (no_mean - full_mean) if metric in LOWER_IS_BETTER else (full_mean - no_mean)
            ranks = _rank_variants(rows, scenario, metric)
            if abs(gain) <= 1.0e-12:
                winner = "tie"
            else:
                winner = "proposed_full_ck" if gain > 0 else "proposed_no_ck"
            out.append(
                {
                    "scenario_name": scenario,
                    "metric": metric,
                    "full_ck_mean": full_mean,
                    "no_ck_mean": no_mean,
                    "absolute_gain": gain,
                    "relative_gain_percent": _safe_div(gain, abs(no_mean)) * 100.0,
                    "winner": winner,
                    "rank_full_ck": ranks.get("proposed_full_ck", ""),
                    "rank_no_ck": ranks.get("proposed_no_ck", ""),
                }
            )
    return out


def _build_first_place(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scenario in sorted({row["scenario_name"] for row in rows}):
        for metric in METRICS:
            means = []
            for (variant,), group in _group([row for row in rows if row["scenario_name"] == scenario], ["variant"]).items():
                means.append((variant, statistics.mean(_to_float(row[metric]) for row in group)))
            ordered = sorted(means, key=lambda item: item[1], reverse=metric not in LOWER_IS_BETTER)
            ranks = {variant: idx for idx, (variant, _value) in enumerate(ordered, start=1)}
            best_variant, best_value = ordered[0]
            out.append(
                {
                    "scenario_name": scenario,
                    "metric": metric,
                    "best_variant": best_variant,
                    "best_value": best_value,
                    "proposed_full_ck_rank": ranks.get("proposed_full_ck", ""),
                    "proposed_no_ck_rank": ranks.get("proposed_no_ck", ""),
                    "proposed_static_balanced_ck_rank": ranks.get("proposed_static_balanced_ck", ""),
                    "proposed_ck_only_no_lyapunov_rank": ranks.get("proposed_ck_only_no_lyapunov", ""),
                    "old_proposed_rank": ranks.get("old_proposed", ""),
                }
            )
    return out


def _build_over_penalty(rows: list[dict[str, Any]], action_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_action = _group(action_log, ["variant", "seed", "scenario_name"])
    out = []
    for row in rows:
        actions = by_action.get((row["variant"], row["seed"], row["scenario_name"]), [])
        selected = [act for act in actions if _to_int(act.get("selected_action"))]
        total_decisions = len([act for act in actions if act.get("candidate_stage") == "defer"])
        full_match = next(
            (
                other
                for other in rows
                if other["variant"] == "proposed_full_ck" and other["seed"] == row["seed"] and other["scenario_name"] == row["scenario_name"]
            ),
            None,
        )
        hv_gap = _to_float(full_match.get("high_value_ground_success_ratio")) - _to_float(row["high_value_ground_success_ratio"]) if full_match else 0.0
        expired_gap = _to_float(row["expired_high_value_task_count"]) - _to_float(full_match.get("expired_high_value_task_count")) if full_match else 0.0
        over_flag = int(_to_float(row["min_SOC"]) > 0.65 and (hv_gap > 0.05 or expired_gap > 1.0))
        out.append(
            {
                "variant": row["variant"],
                "seed": row["seed"],
                "scenario_name": row["scenario_name"],
                "mean_lambda_E_final": statistics.mean([_to_float(act.get("lambda_E_final")) for act in actions]) if actions else 0.0,
                "mean_lambda_Q_final": statistics.mean([_to_float(act.get("lambda_Q_final")) for act in actions]) if actions else 0.0,
                "mean_lambda_L_final": statistics.mean([_to_float(act.get("lambda_L_final")) for act in actions]) if actions else 0.0,
                "mean_lambda_S_final": statistics.mean([_to_float(act.get("lambda_S_final")) for act in actions]) if actions else 0.0,
                "defer_action_ratio": _safe_div(max(0, total_decisions - len(selected)), total_decisions),
                "metadata_action_ratio": _safe_div(sum(act.get("candidate_stage") == "metadata" for act in selected), len(selected)),
                "roi_action_ratio": _safe_div(sum(act.get("candidate_stage") == "ROI" for act in selected), len(selected)),
                "roi_context_action_ratio": _safe_div(sum(act.get("candidate_stage") == "ROI+Ctx" for act in selected), len(selected)),
                "ground_success_ratio": row["ground_success_ratio"],
                "high_value_ground_success_ratio": row["high_value_ground_success_ratio"],
                "expired_high_value_task_count": row["expired_high_value_task_count"],
                "min_SOC": row["min_SOC"],
                "final_backlog_count": row["final_backlog_count"],
                "over_conservative_flag": over_flag,
            }
        )
    return out


def _plot_delivery_gain(effect: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = [
        "ground_success_ratio",
        "high_value_ground_success_ratio",
        "high_value_target_success_ratio",
        "quality_weighted_high_value_success_value",
        "expired_high_value_task_count",
        "H_net_utility",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        rows = [row for row in effect if row["metric"] == metric]
        ax.bar(range(len(rows)), [_to_float(row["absolute_gain"]) for row in rows], color="#2563eb")
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_title(metric)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([row["scenario_name"] for row in rows], rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ck_delivery_gain_panel.png", dpi=180)
    plt.close(fig)


def _plot_mode_distribution(distribution: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in distribution if row["variant"] == "proposed_full_ck"]
    scenarios = sorted({row["scenario_name"] for row in rows})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = [0.0] * len(scenarios)
    colors = ["#64748b", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2"]
    for mode, color in zip(MODE_ORDER[:6], colors):
        values = []
        for scenario in scenarios:
            matches = [row for row in rows if row["scenario_name"] == scenario and row["mode_name"] == mode]
            values.append(statistics.mean([_to_float(row["mode_fraction"]) for row in matches]) if matches else 0.0)
        ax.bar(range(len(scenarios)), values, bottom=bottom, label=mode, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mode fraction")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "ck_mode_distribution_panel.png", dpi=180)
    plt.close(fig)


def _plot_price_modulation(trace: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in trace if row["variant"] == "proposed_full_ck"]
    pairs = [("lambda_E_base", "lambda_E_final"), ("lambda_Q_base", "lambda_Q_final"), ("lambda_L_base", "lambda_L_final"), ("lambda_S_base", "lambda_S_final")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    for ax, (xkey, ykey) in zip(axes.flat, pairs):
        ax.scatter([_to_float(row[xkey]) for row in rows], [_to_float(row[ykey]) for row in rows], s=12, alpha=0.35)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ck_price_modulation_panel.png", dpi=180)
    plt.close(fig)


def _plot_safety_tradeoff(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for variant in VARIANTS:
        source = [row for row in rows if row["variant"] == variant]
        if not source:
            continue
        ax.scatter(
            [_to_float(row["min_SOC"]) for row in source],
            [_to_float(row["quality_weighted_high_value_success_value"]) for row in source],
            s=[35 + 150 * _to_float(row["ground_success_ratio"]) for row in source],
            alpha=0.62,
            label=variant,
        )
    ax.set_xlabel("min_SOC")
    ax.set_ylabel("quality_weighted_high_value_success_value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ck_safety_tradeoff_panel.png", dpi=180)
    plt.close(fig)


def _plot_over_penalty(over: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = ["defer_action_ratio", "metadata_action_ratio", "roi_action_ratio", "roi_context_action_ratio", "expired_high_value_task_count", "min_SOC"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        values = []
        for variant in VARIANTS:
            source = [row for row in over if row["variant"] == variant]
            values.append(statistics.mean(_to_float(row[metric]) for row in source) if source else 0.0)
        ax.bar(range(len(VARIANTS)), values, color="#0f766e")
        ax.set_title(metric)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(VARIANTS, rotation=55, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ck_over_penalty_panel.png", dpi=180)
    plt.close(fig)


def build_figures(rows: list[dict[str, Any]], effect: list[dict[str, Any]], trace: list[dict[str, Any]], distribution: list[dict[str, Any]], over: list[dict[str, Any]], output_dir: Path) -> None:
    _plot_delivery_gain(effect, output_dir)
    _plot_mode_distribution(distribution, output_dir)
    _plot_price_modulation(trace, output_dir)
    _plot_safety_tradeoff(rows, output_dir)
    _plot_over_penalty(over, output_dir)


def _mean_for(rows: list[dict[str, Any]], variant: str, metric: str) -> float:
    vals = [_to_float(row[metric]) for row in rows if row["variant"] == variant]
    return statistics.mean(vals) if vals else 0.0


def _gain_sentence(rows: list[dict[str, Any]], metric: str) -> str:
    full = _mean_for(rows, "proposed_full_ck", metric)
    no = _mean_for(rows, "proposed_no_ck", metric)
    gain = (no - full) if metric in LOWER_IS_BETTER else (full - no)
    verb = "improves" if gain > 1.0e-12 else "does not improve" if gain < -1.0e-12 else "ties"
    return f"{metric}: full_ck={_fmt(full)}, no_ck={_fmt(no)}, gain={_fmt(gain)} ({verb})."


def build_quick_report(rows: list[dict[str, Any]], distribution: list[dict[str, Any]], over: list[dict[str, Any]]) -> str:
    full = [row for row in rows if row["variant"] == "proposed_full_ck"]
    no = [row for row in rows if row["variant"] == "proposed_no_ck"]
    static_gap = _mean_for(rows, "proposed_full_ck", "quality_weighted_high_value_success_value") - _mean_for(rows, "proposed_static_balanced_ck", "quality_weighted_high_value_success_value")
    ck_only_gap = _mean_for(rows, "proposed_full_ck", "quality_weighted_high_value_success_value") - _mean_for(rows, "proposed_ck_only_no_lyapunov", "quality_weighted_high_value_success_value")
    full_over = [row for row in over if row["variant"] == "proposed_full_ck"]
    over_flags = sum(_to_int(row["over_conservative_flag"]) for row in full_over)
    mode_rows = [row for row in distribution if row["variant"] == "proposed_full_ck"]
    mode_counts = Counter()
    for row in mode_rows:
        mode_counts[row["mode_name"]] += _to_float(row["mode_count"])
    frequent_modes = ", ".join(f"{mode}={_fmt(count, 0)}" for mode, count in mode_counts.most_common())
    energy_rows = [row for row in mode_rows if row["mode_name"] == "energy_saving"]
    queue_rows = [row for row in mode_rows if row["mode_name"] == "queue_draining"]
    latency_rows = [row for row in mode_rows if row["mode_name"] == "latency_critical"]
    mission_rows = [row for row in mode_rows if row["mode_name"] == "mission_priority"]
    egress_rows = [row for row in mode_rows if row["mode_name"] == "egress_conservative"]
    hard_violations = sum(_to_int(row["hard_safety_violation_count"]) for row in full)
    net_gain = _mean_for(rows, "proposed_full_ck", "net_system_utility") - _mean_for(rows, "proposed_no_ck", "net_system_utility")
    h_gain = _mean_for(rows, "proposed_full_ck", "H_net_utility") - _mean_for(rows, "proposed_no_ck", "H_net_utility")
    defer_mean = statistics.mean([_to_float(row["defer_action_ratio"]) for row in full_over]) if full_over else 0.0
    min_soc_mean = statistics.mean([_to_float(row["min_SOC"]) for row in full]) if full else 0.0
    expired_delta = _mean_for(rows, "proposed_no_ck", "expired_high_value_task_count") - _mean_for(rows, "proposed_full_ck", "expired_high_value_task_count")
    return "\n".join(
        [
            "# c_k mode-control ablation quick report",
            "",
            "- This standalone experiment wrote only `outputs/_ck_mode_control_ablation_v1/` and uses shared task/contact/resource/value-prior hashes per seed/scenario.",
            "- Variants: proposed_full_ck, proposed_no_ck, proposed_static_balanced_ck, proposed_ck_only_no_lyapunov, old_proposed.",
            "- `proposed_ck_only_no_lyapunov` is implemented with constant base prices, so it is an intentionally coarse complementarity check rather than a tuned policy.",
            "",
            "## Required answers",
            f"1. {_gain_sentence(rows, 'ground_success_ratio')}",
            f"2. {_gain_sentence(rows, 'high_value_ground_success_ratio')}",
            f"3. {_gain_sentence(rows, 'high_value_target_success_ratio')}",
            f"4. {_gain_sentence(rows, 'quality_weighted_high_value_success_value')}",
            f"5. {_gain_sentence(rows, 'expired_high_value_task_count')}",
            f"6. net_system_utility gain={_fmt(net_gain)}; H_net_utility gain={_fmt(h_gain)}.",
            f"7. full_ck hard_safety_violation_count total={hard_violations}; zero means the hard safety condition is maintained.",
            f"8. Over-penalty check: full_ck defer_action_ratio mean={_fmt(defer_mean)}, min_SOC mean={_fmt(min_soc_mean)}, no_ck-full_ck expired high-value delta={_fmt(expired_delta)}; over_conservative_flags={over_flags}/{len(full_over)}.",
            f"9. Frequent full_ck modes: {frequent_modes if frequent_modes else 'none'}.",
            "10. Mode-state logic:",
            f"    - energy_saving mean SOC={_fmt(statistics.mean([_to_float(r['mean_SOC_when_active']) for r in energy_rows]) if energy_rows else 0.0)}, mean queue pressure={_fmt(statistics.mean([_to_float(r['mean_queue_pressure_when_active']) for r in energy_rows]) if energy_rows else 0.0)}.",
            f"    - queue_draining mean queue pressure={_fmt(statistics.mean([_to_float(r['mean_queue_pressure_when_active']) for r in queue_rows]) if queue_rows else 0.0)}.",
            f"    - latency_critical mean deadline urgency={_fmt(statistics.mean([_to_float(r['mean_deadline_urgency_when_active']) for r in latency_rows]) if latency_rows else 0.0)}.",
            f"    - mission_priority mean high-value ratio={_fmt(statistics.mean([_to_float(r['mean_high_value_ratio_when_active']) for r in mission_rows]) if mission_rows else 0.0)}.",
            f"    - egress_conservative mean queue pressure={_fmt(statistics.mean([_to_float(r['mean_queue_pressure_when_active']) for r in egress_rows]) if egress_rows else 0.0)}.",
            f"11. static_balanced_ck vs adaptive full_ck quality-weighted high-value gap={_fmt(static_gap)}.",
            f"12. ck_only_no_lyapunov vs full_ck quality-weighted high-value gap={_fmt(ck_only_gap)}.",
            "13. c_k's non-replaceable role is bounded adaptive modulation: it translates continuous Lyapunov pressure into scenario modes, amplifies mission priority under high-value load, strengthens latency pressure near deadlines, pushes queue draining under backlog, and raises energy caution under low SOC or thermal pressure without becoming a separate hard gate.",
            "14. Paper-figure recommendation: keep Resource Mode Control if ck_effect_summary.csv shows positive gains on the delivery/high-value metrics central to the paper; otherwise present it as conditionally useful with the non-winning scenarios disclosed.",
            "15. DDQN recommendation: the diagnostic trace is suitable for a future learning-assisted c_k controller because it exposes state, selected mode, bounded multipliers, and outcome metrics without changing lower-layer algorithms.",
        ]
    )


def _run_one_case(
    *,
    seed: int,
    scenario_name: str,
    workload: str,
    contact: str,
    initial_soc: float,
    manifest: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    hashes: dict[str, str],
    pricing: LyapunovPricing,
    controller: RuleBasedModeController,
) -> list[dict[str, Any]]:
    results = []
    for variant in VARIANTS:
        if variant == "old_proposed":
            old_initial = base.INITIAL_SOC
            base.INITIAL_SOC = initial_soc
            try:
                raw = base.run_method("proposed_guarantee_aware_full", seed, 1.0, manifest, contacts, resources, hashes)
            finally:
                base.INITIAL_SOC = old_initial
            results.append(
                _normalize_old_result(
                    raw,
                    seed=seed,
                    scenario_name=scenario_name,
                    workload=workload,
                    contact=contact,
                    initial_soc=initial_soc,
                    hashes=hashes,
                )
            )
        else:
            results.append(
                run_ck_variant(
                    variant=variant,
                    seed=seed,
                    scenario_name=scenario_name,
                    workload=workload,
                    contact=contact,
                    initial_soc=initial_soc,
                    manifest=manifest,
                    contacts=contacts,
                    resources=resources,
                    hashes=hashes,
                    pricing=pricing,
                    controller=controller,
                )
            )
    return results


def run_experiment(output_dir: Path, seeds: list[int], low_soc_seeds: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pricing = LyapunovPricing(_load_yaml(PROJECT_ROOT / "configs" / "lyapunov_pricing.yaml"))
    controller = RuleBasedModeController(_load_yaml(PROJECT_ROOT / "configs" / "mode_controller.yaml"))

    all_summary: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    all_trajectory: list[dict[str, Any]] = []

    for seed in seeds:
        for workload in realistic.WORKLOAD_SCENARIOS:
            for contact in realistic.CONTACT_SCENARIOS:
                scenario_name = f"realistic__{workload}__{contact}"
                contacts = realistic.build_contact_schedule(seed, workload, contact)
                resources = realistic.build_resource_trace(seed, workload, contact)
                manifest, _audit = realistic.build_task_manifest(seed, workload, contact, contacts)
                hashes = _scenario_hashes(manifest, contacts, resources)
                for result in _run_one_case(
                    seed=seed,
                    scenario_name=scenario_name,
                    workload=workload,
                    contact=contact,
                    initial_soc=base.INITIAL_SOC,
                    manifest=manifest,
                    contacts=contacts,
                    resources=resources,
                    hashes=hashes,
                    pricing=pricing,
                    controller=controller,
                ):
                    all_summary.append(result["summary"])
                    all_trace.extend(result["mode_trace"])
                    all_actions.extend(result["action_log"])
                    all_events.extend(result["events"])
                    all_tasks.extend(result["task_delivery"])
                    all_trajectory.extend(result["trajectory"])

    old_initial = base.INITIAL_SOC
    base.INITIAL_SOC = low_soc.INITIAL_SOC
    try:
        for seed in low_soc_seeds:
            contacts = low_soc._build_normal_contact(seed)
            manifest, _audit = low_soc._build_manifest(seed, contacts)
            resources, _calibration = low_soc._calibrated_resource_trace(seed, manifest, contacts)
            hashes = _scenario_hashes(manifest, contacts, resources)
            for result in _run_one_case(
                seed=seed,
                scenario_name="low_initial_soc__emergency_burst_recovery__normal_contact",
                workload=low_soc.WORKLOAD_SCENARIO,
                contact=low_soc.CONTACT_SCENARIO,
                initial_soc=low_soc.INITIAL_SOC,
                manifest=manifest,
                contacts=contacts,
                resources=resources,
                hashes=hashes,
                pricing=pricing,
                controller=controller,
            ):
                all_summary.append(result["summary"])
                all_trace.extend(result["mode_trace"])
                all_actions.extend(result["action_log"])
                all_events.extend(result["events"])
                all_tasks.extend(result["task_delivery"])
                all_trajectory.extend(result["trajectory"])
    finally:
        base.INITIAL_SOC = old_initial

    ranking = _build_ranking(all_summary)
    distribution = _build_mode_distribution(all_trace)
    effect = _build_effect_summary(all_summary)
    over = _build_over_penalty(all_summary, all_actions)
    first_place = _build_first_place(all_summary)

    _write_csv(output_dir / "ck_ablation_summary_by_seed.csv", all_summary, SUMMARY_COLUMNS)
    _write_csv(output_dir / "ck_ablation_multi_seed_ranking.csv", ranking)
    _write_csv(output_dir / "ck_effect_summary.csv", effect)
    _write_csv(output_dir / "ck_mode_trace.csv", all_trace, TRACE_COLUMNS)
    _write_csv(output_dir / "ck_mode_distribution.csv", distribution)
    _write_csv(output_dir / "ck_over_penalty_check.csv", over)
    _write_csv(output_dir / "ck_first_place_check.csv", first_place)
    _write_csv(output_dir / "ck_action_log.csv", all_actions)
    _write_csv(output_dir / "ck_ground_events.csv", all_events)
    _write_csv(output_dir / "ck_task_delivery.csv", all_tasks)
    _write_csv(output_dir / "ck_trajectory.csv", all_trajectory)
    build_figures(all_summary, effect, all_trace, distribution, over, output_dir)
    _write_text(output_dir / "quick_report.md", build_quick_report(all_summary, distribution, over))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Seeds for the realistic workload/contact benchmark.")
    parser.add_argument("--low-soc-seeds", nargs="*", type=int, default=None, help="Seeds for low-initial-SOC emergency stress.")
    parser.add_argument("--full-low-soc-seeds", action="store_true", help="Use low-SOC seeds 0..4 instead of default 0..2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    low_seeds = args.low_soc_seeds if args.low_soc_seeds else (FULL_LOW_SOC_SEEDS if args.full_low_soc_seeds else DEFAULT_SEEDS)
    run_experiment(args.output_dir, seeds, low_seeds)
    print(f"Wrote c_k mode-control ablation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
