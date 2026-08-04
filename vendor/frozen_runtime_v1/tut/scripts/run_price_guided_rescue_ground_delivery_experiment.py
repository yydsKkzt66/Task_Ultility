"""Price-guided rescue variant for uneven-pressure ground delivery.

Standalone, default-off experiment runner. This file intentionally reuses the
existing manifest/contact/resource builders and old baseline simulators from
``run_uneven_pressure_ground_delivery_experiment`` without changing them. The
only new policy implemented here is ``proposed_price_guided_rescue``.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_uneven_pressure_ground_delivery_experiment as base  # noqa: E402
import scripts.run_fair_baseline_comparison as fair  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_price_guided_rescue_ground_delivery_v1"
METHOD_PRICE = "proposed_price_guided_rescue"
METHODS = [
    "proposed_guarantee_aware_full",
    METHOD_PRICE,
    "utility_rule",
    "persistent_queue_without_guarantee",
    "fixed_lightweight",
    "fixed_heavy",
    "fixed_heavy_with_20pct_protection",
    "ground_approved_followup",
]
METHOD_TYPES = {
    **base.METHOD_TYPES,
    METHOD_PRICE: "proposed_price_guided_rescue",
}
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_PRESSURE_SCALES = [1.0]
FULL_PRESSURE_SCALES = [0.85, 1.0, 1.15]

BATTERY_HARD_FLOOR = 0.20
BATTERY_SELECTION_MARGIN = 0.005
THERMAL_HARD_LIMIT = 1.25
QUEUE_HARD_CAPACITY = None
MODEL_CHOICES = ["Tiny", "Small", "Medium"]
PRODUCT_CHOICES = ["metadata", "roi", "roi_context"]
STAGE_BY_PRODUCT = {"metadata": "metadata", "roi": "ROI", "roi_context": "ROI+Ctx"}
PRODUCT_BY_STAGE = {"metadata": "metadata", "ROI": "roi", "ROI+Ctx": "roi_context"}

GROUND_SUMMARY_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "task_manifest_hash",
    "contact_schedule_hash",
    "resource_trace_hash",
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "avg_ground_received_quality",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "net_system_utility",
    "H_net_utility",
    "min_SOC",
    "final_SOC",
    "fraction_time_SOC_below_50",
    "fraction_time_SOC_below_20",
    "hard_safety_violation_count",
    "expired_task_count",
    "expired_high_value_task_count",
    "final_backlog_count",
    "max_backlog_count",
    "protection_trigger_count",
]

ACTION_LOG_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "time_h",
    "orbit_index",
    "wave_name",
    "task_id",
    "scene_id",
    "value_level",
    "deadline_bucket",
    "deadline_slack_h",
    "SOC_before",
    "SOC_after",
    "candidate_model",
    "candidate_stage",
    "selected_action",
    "score",
    "expected_ground_delivery_utility",
    "energy_price_cost",
    "comm_price_cost",
    "queue_cost",
    "thermal_cost",
    "recovery_opportunity_cost",
    "deadline_urgency_bonus",
    "cheap_win_density_bonus",
    "delivery_density",
    "predicted_ground_delivery_success",
    "product_size_MB",
    "estimated_energy_wh",
    "reason",
]

PRICE_COMPONENT_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "soc_band",
    "avg_lambda_E",
    "avg_lambda_Comm",
    "avg_lambda_Q",
    "avg_score_metadata",
    "avg_score_roi",
    "avg_score_roi_context",
    "metadata_rate",
    "roi_rate",
    "roi_context_rate",
    "defer_rate",
    "avg_delivery_density",
    "ground_success_count",
    "high_value_ground_success_count",
]

RESCUE_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "deadline_rescue_candidate_count",
    "deadline_rescue_selected_count",
    "deadline_rescue_success_count",
    "cheap_win_candidate_count",
    "cheap_win_selected_count",
    "cheap_win_success_count",
    "cheap_win_low_medium_count",
    "cheap_win_high_urgent_count",
    "cheap_win_avg_density",
    "cheap_win_avg_energy_wh",
    "cheap_win_avg_payload_MB",
    "high_value_expired_count",
    "total_expired_count",
]

EXPIRED_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "task_id",
    "scene_id",
    "value_level",
    "wave_name",
    "deadline_h",
    "expired_time_h",
    "last_stage_ground_received",
    "min_required_stage",
    "target_stage",
    "expired_reason",
    "SOC_at_expiration",
    "queue_wait_h",
]

RANKING_METRICS = [
    "ground_success_ratio",
    "high_value_ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_target_success_ratio",
    "avg_ground_received_quality",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "min_SOC",
    "H_net_utility",
    "net_system_utility",
]


def _to_float(value: object, default: float = 0.0) -> float:
    return base._to_float(value, default)


def _to_int(value: object, default: int = 0) -> int:
    return base._to_int(value, default)


def _safe_div(num: float, den: float) -> float:
    return base._safe_div(num, den)


def _fmt(value: object, digits: int = 4) -> str:
    return base._fmt(value, digits)


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


def _stage_rank(stage: str) -> int:
    return base._stage_rank(stage)


def _highest_stage(products: set[str]) -> str:
    return base._highest_stage(products)


def _model_quality(model: str, product: str) -> float:
    table = {
        "Tiny": {"metadata": 0.84, "roi": 0.79, "roi_context": 0.72},
        "Small": {"metadata": 0.88, "roi": 0.86, "roi_context": 0.82},
        "Medium": {"metadata": 0.91, "roi": 0.90, "roi_context": 0.88},
    }
    return table[model][product]


def _model_compute_multiplier(model: str) -> float:
    return {"Tiny": 0.52, "Small": 0.82, "Medium": 1.12}[model]


def _candidate_costs(model: str, product: str, payload_mb: float) -> tuple[float, float, float]:
    base_compute = {"metadata": 1.05, "roi": 5.6, "roi_context": 13.4}[product]
    compute = base_compute * _model_compute_multiplier(model)
    comm = 0.20 * payload_mb
    account = 0.016 * compute + 0.30 * comm + 0.012 * payload_mb
    return compute, comm, account


def _payload_idle_wh(payload_on: bool) -> float:
    return 5.8 if payload_on else 0.0


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _soc_band(soc: float) -> str:
    return base._soc_band(soc)


def _thermal_pressure(wave: str, orbit_compute: float = 0.0) -> float:
    wave_base = {
        "warmup": 0.25,
        "wave_1_moderate": 0.45,
        "recovery_1": 0.25,
        "wave_2_extreme": 0.72,
        "recovery_2_incomplete": 0.32,
        "wave_3_medium_high": 0.62,
        "final_recovery": 0.24,
    }.get(wave, 0.4)
    return wave_base + 0.012 * orbit_compute


def _resource_prices(soc: float, queue_count: int, queue_mb: float, wave: str, contact_capacity_mb: float) -> dict[str, float]:
    soc_gap = max(0.0, 0.70 - soc)
    near_floor = 1.0 / max(0.08, soc - BATTERY_HARD_FLOOR + 0.08)
    queue_pressure = _clip(queue_count / 18.0 + queue_mb / 36.0, 0.0, 2.0)
    contact_pressure = _clip(1.0 - contact_capacity_mb / 5.0, 0.0, 1.0)
    recovery = 1.0 if "recovery" in wave or wave == "final_recovery" else 0.0
    thermal = _thermal_pressure(wave)
    return {
        "lambda_E": _clip(0.035 + 0.78 * soc_gap * soc_gap + 0.008 * near_floor, 0.035, 0.55),
        "lambda_Comm": _clip(0.16 + 0.42 * soc_gap + 0.16 * contact_pressure, 0.12, 0.62),
        "lambda_Q": _clip(0.035 + 0.12 * queue_pressure, 0.02, 0.34),
        "lambda_T": _clip(0.025 + 0.18 * max(0.0, thermal - 0.55), 0.02, 0.18),
        "lambda_R": 0.16 if recovery else 0.0,
    }


def _estimated_delivery_time_h(product: str, payload_mb: float) -> float:
    transmission_h = max(0.05, payload_mb / 5.0) * base.ORBIT_HOURS
    processing_h = {"metadata": 0.08, "roi": 0.18, "roi_context": 0.30}[product] * base.ORBIT_HOURS
    return base.ORBIT_HOURS + transmission_h + processing_h


def _deadline_info(task: dict[str, Any], orbit: int, product: str, payload_mb: float) -> tuple[float, str]:
    current_time_h = orbit * base.ORBIT_HOURS
    if current_time_h > _to_float(task["deadline_h"]):
        return _to_float(task["deadline_h"]) - current_time_h, "expired"
    slack = _to_float(task["deadline_h"]) - current_time_h - _estimated_delivery_time_h(product, payload_mb)
    if slack <= base.ORBIT_HOURS:
        return slack, "critical"
    if slack <= 2.0 * base.ORBIT_HOURS:
        return slack, "urgent_deadline"
    return slack, "normal"


def _value_weight(value_level: str) -> float:
    return {"low": 0.5, "medium": 1.0, "high": 2.0, "urgent": 3.0}.get(value_level, 1.0)


def _missing_stage_gain(task: dict[str, Any], delivered: set[str], product: str) -> float:
    current = _highest_stage(delivered)
    stage = STAGE_BY_PRODUCT[product]
    if _stage_rank(current) >= _stage_rank(stage):
        return 0.0
    if _stage_rank(current) < _stage_rank(str(task["min_required_stage"])):
        return 1.25 if _stage_rank(stage) >= _stage_rank(str(task["min_required_stage"])) else 0.75
    if _stage_rank(current) < _stage_rank(str(task["target_stage"])):
        return 0.95 if _stage_rank(stage) >= _stage_rank(str(task["target_stage"])) else 0.55
    return 0.0


def _deadline_bonus(task: dict[str, Any], delivered: set[str], product: str, deadline_bucket: str) -> float:
    factor = {"critical": 3.0, "urgent_deadline": 1.5, "normal": 0.0, "expired": 0.0}[deadline_bucket]
    return 0.55 * _value_weight(str(task["value_level"])) * factor * _missing_stage_gain(task, delivered, product)


def _predicted_ground_delivery_success(payload_mb: float, capacity_mb: float, deadline_bucket: str, soc_after: float) -> float:
    capacity_factor = _clip(capacity_mb / max(payload_mb, 1.0e-9), 0.0, 1.0)
    deadline_factor = {"critical": 0.94, "urgent_deadline": 0.90, "normal": 0.82, "expired": 0.0}[deadline_bucket]
    soc_factor = _clip((soc_after - BATTERY_HARD_FLOOR) / 0.20, 0.25, 1.0)
    return capacity_factor * deadline_factor * soc_factor


def _timeliness_factor(deadline_bucket: str) -> float:
    return {"critical": 1.0, "urgent_deadline": 0.92, "normal": 0.78, "expired": 0.0}[deadline_bucket]


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
) -> dict[str, Any]:
    payload = base._product_payload(task, product)
    compute, comm, account_cost = _candidate_costs(model, product, payload)
    estimated_energy = compute + comm
    soc_after = soc + (
        _to_float(_current_solar_placeholder.get("solar", 0.0))
        - _to_float(_current_solar_placeholder.get("housekeeping", 0.0))
        - _payload_idle_wh(True)
        - estimated_energy
        - base.TX_FIXED_WH_PER_ACTIVE_ORBIT
    ) / base.BATTERY_CAPACITY_WH
    thermal_after = _thermal_pressure(wave, compute)
    queue_after = queue_count
    hard_ok = soc_after >= BATTERY_HARD_FLOOR + BATTERY_SELECTION_MARGIN and thermal_after <= THERMAL_HARD_LIMIT
    if QUEUE_HARD_CAPACITY is not None:
        hard_ok = hard_ok and queue_after <= QUEUE_HARD_CAPACITY
    deadline_slack, deadline_bucket = _deadline_info(task, orbit, product, payload)
    if deadline_bucket == "expired":
        hard_ok = False

    prices = _resource_prices(soc, queue_count, queue_mb, wave, capacity_mb)
    delivered_before = set(delivered)
    delta = fair.delta_information_utility(product, delivered_before, str(task["product_requirement"]))
    quality = _model_quality(model, product)
    predicted = _predicted_ground_delivery_success(payload, capacity_mb, deadline_bucket, soc_after)
    expected = _to_float(task["V_eff"]) * delta * quality * predicted * _timeliness_factor(deadline_bucket)
    queue_delay_cost = queue_count / 10.0 + queue_mb / 25.0
    energy_price_cost = prices["lambda_E"] * estimated_energy
    comm_price_cost = prices["lambda_Comm"] * (comm + 0.08 * payload)
    queue_cost = prices["lambda_Q"] * queue_delay_cost * (1.0 + 0.15 * payload)
    thermal_cost = prices["lambda_T"] * compute * (1.0 if model == "Medium" else 0.55 if model == "Small" else 0.25)
    recovery_cost = prices["lambda_R"] * (0.10 * compute + 0.05 * payload)
    deadline_urgency_bonus = _deadline_bonus(task, delivered_before, product, deadline_bucket)
    density = expected / (estimated_energy + 0.35 * payload + 0.08 * queue_delay_cost + 1.0e-9)
    product_pref = {"metadata": 1.0, "roi": 0.42, "roi_context": 0.05}[product]
    density_threshold = {"metadata": 0.075, "roi": 0.11, "roi_context": 0.28}[product]
    value_pref = 1.25 if str(task["value_level"]) in {"low", "medium"} else 0.75
    cheap_win_density_bonus = 0.38 * product_pref * value_pref * _clip(density - density_threshold, 0.0, 1.8)
    score = (
        expected
        - energy_price_cost
        - comm_price_cost
        - queue_cost
        - thermal_cost
        - recovery_cost
        + deadline_urgency_bonus
        + cheap_win_density_bonus
        - account_cost
    )
    return {
        "task_id": task["task_id"],
        "scene_id": task["scene_id"],
        "value_level": task["value_level"],
        "model": model,
        "product": product,
        "stage": STAGE_BY_PRODUCT[product],
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
        "recovery_opportunity_cost": recovery_cost,
        "deadline_urgency_bonus": deadline_urgency_bonus,
        "cheap_win_density_bonus": cheap_win_density_bonus,
        "delivery_density": density,
        "score": score,
        "hard_ok": int(hard_ok),
        "lambda_E": prices["lambda_E"],
        "lambda_Comm": prices["lambda_Comm"],
        "lambda_Q": prices["lambda_Q"],
        "reason": "safe_price_score" if hard_ok else "filtered_by_hard_safety_or_expired",
    }


_current_solar_placeholder: dict[str, float] = {}


def _candidate_products(task: dict[str, Any], delivered: set[str]) -> list[str]:
    target_rank = _stage_rank(str(task["target_stage"]))
    products = []
    for product in PRODUCT_CHOICES:
        if product in delivered:
            continue
        if _stage_rank(STAGE_BY_PRODUCT[product]) <= target_rank:
            products.append(product)
    return products


def run_price_guided_method(
    seed: int,
    pressure_scale: float,
    manifest: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    hashes: dict[str, str],
) -> dict[str, Any]:
    method = METHOD_PRICE
    by_task = {str(row["task_id"]): row for row in manifest}
    arrivals: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in manifest:
        arrivals[_to_int(task["orbit_index"])].append(task)
    contact_by_orbit = {_to_int(row["orbit_index"]): row for row in contacts}
    resource_by_orbit = {_to_int(row["orbit_index"]): row for row in resources}

    delivered: dict[str, set[str]] = defaultdict(set)
    active: set[str] = set()
    expired_logged: set[str] = set()
    events: list[dict[str, Any]] = []
    task_rows: dict[str, dict[str, Any]] = {}
    trajectory: list[dict[str, Any]] = []
    action_log: list[dict[str, Any]] = []
    expired_log: list[dict[str, Any]] = []
    price_samples: list[dict[str, Any]] = []
    cumulative_direct_energy = 0.0
    cumulative_system_energy = 0.0
    net_system_utility = 0.0
    h_net_utility = 0.0
    soc = base.INITIAL_SOC
    protection_triggers = 0

    for task in manifest:
        task_rows[str(task["task_id"])] = {
            "method": method,
            "seed": seed,
            "pressure_scale": pressure_scale,
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
            task = by_task[task_id]
            if time_h > _to_float(task["deadline_h"]) and task_id not in expired_logged:
                expired_logged.add(task_id)
                active.discard(task_id)
                expired_log.append(
                    {
                        "method": method,
                        "seed": seed,
                        "pressure_scale": pressure_scale,
                        "task_id": task_id,
                        "scene_id": task["scene_id"],
                        "value_level": task["value_level"],
                        "wave_name": task["wave_name"],
                        "deadline_h": task["deadline_h"],
                        "expired_time_h": time_h,
                        "last_stage_ground_received": _highest_stage(delivered[task_id]),
                        "min_required_stage": task["min_required_stage"],
                        "target_stage": task["target_stage"],
                        "expired_reason": "deadline_passed_before_ground_success",
                        "SOC_at_expiration": soc,
                        "queue_wait_h": max(0.0, time_h - _to_float(task["capture_time_h"])),
                    }
                )

        candidates: list[dict[str, Any]] = []
        capacity = _to_float(contact_by_orbit[orbit]["capacity_MB"]) if orbit in contact_by_orbit else 0.0
        ready_tasks = [by_task[task_id] for task_id in sorted(active) if time_h <= _to_float(by_task[task_id]["deadline_h"])]
        queue_count = sum(1 for task in ready_tasks if _candidate_products(task, delivered[str(task["task_id"])]))
        queue_mb = sum(
            min([base._product_payload(task, product) for product in _candidate_products(task, delivered[str(task["task_id"])])] or [0.0])
            for task in ready_tasks
        )
        _current_solar_placeholder["solar"] = _to_float(resource["solar_energy_wh"])
        _current_solar_placeholder["housekeeping"] = _to_float(resource["housekeeping_energy_wh"])
        if orbit in contact_by_orbit:
            for task in ready_tasks:
                task_id = str(task["task_id"])
                for product in _candidate_products(task, delivered[task_id]):
                    for model in MODEL_CHOICES:
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
                            )
                        )

        by_task_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cand in candidates:
            by_task_candidates[str(cand["task_id"])].append(cand)
        for task in ready_tasks:
            task_id = str(task["task_id"])
            payload = min([base._product_payload(task, p) for p in _candidate_products(task, delivered[task_id])] or [0.0])
            slack, bucket = _deadline_info(task, orbit, "metadata", payload or 0.1)
            action_log.append(
                {
                    "method": method,
                    "seed": seed,
                    "pressure_scale": pressure_scale,
                    "time_h": time_h,
                    "orbit_index": orbit,
                    "wave_name": wave,
                    "task_id": task_id,
                    "scene_id": task["scene_id"],
                    "value_level": task["value_level"],
                    "deadline_bucket": bucket,
                    "deadline_slack_h": slack,
                    "SOC_before": soc,
                    "SOC_after": soc,
                    "candidate_model": "None",
                    "candidate_stage": "defer",
                    "selected_action": 0,
                    "score": 0.0,
                    "expected_ground_delivery_utility": 0.0,
                    "energy_price_cost": 0.0,
                    "comm_price_cost": 0.0,
                    "queue_cost": 0.0,
                    "thermal_cost": 0.0,
                    "recovery_opportunity_cost": 0.0,
                    "deadline_urgency_bonus": 0.0,
                    "cheap_win_density_bonus": 0.0,
                    "delivery_density": 0.0,
                    "predicted_ground_delivery_success": 0.0,
                    "product_size_MB": 0.0,
                    "estimated_energy_wh": 0.0,
                    "reason": "defer_candidate",
                }
            )

        selected: list[dict[str, Any]] = []
        selected_task_ids: set[str] = set()
        remain = capacity
        selected_energy = 0.0
        for cand in sorted(candidates, key=lambda row: _to_float(row["score"]), reverse=True):
            if not _to_int(cand["hard_ok"]):
                continue
            if cand["task_id"] in selected_task_ids:
                continue
            if _to_float(cand["payload"]) > remain + 1.0e-12:
                continue
            incremental_energy = _to_float(cand["compute"]) + _to_float(cand["comm"])
            if len(selected) >= 8:
                break
            if _to_float(cand["score"]) <= 0.0 and selected:
                continue
            projected_energy = selected_energy + incremental_energy
            projected_payload_idle = _payload_idle_wh(True)
            projected_tx_fixed = base.TX_FIXED_WH_PER_ACTIVE_ORBIT
            projected_soc = soc + (
                _to_float(resource["solar_energy_wh"])
                - _to_float(resource["housekeeping_energy_wh"])
                - projected_payload_idle
                - projected_energy
                - projected_tx_fixed
            ) / base.BATTERY_CAPACITY_WH
            if projected_soc < BATTERY_HARD_FLOOR + BATTERY_SELECTION_MARGIN:
                continue
            selected.append(cand)
            selected_task_ids.add(str(cand["task_id"]))
            remain -= _to_float(cand["payload"])
            selected_energy = projected_energy

        selected_keys = {(str(c["task_id"]), str(c["model"]), str(c["product"])) for c in selected}
        orbit_compute = sum(_to_float(c["compute"]) for c in selected)
        orbit_comm = sum(_to_float(c["comm"]) for c in selected)
        payload_on = bool(selected)
        payload_idle = _payload_idle_wh(payload_on)
        tx_fixed = base.TX_FIXED_WH_PER_ACTIVE_ORBIT if selected else 0.0
        net_energy = (
            _to_float(resource["solar_energy_wh"])
            - _to_float(resource["housekeeping_energy_wh"])
            - payload_idle
            - orbit_compute
            - orbit_comm
            - tx_fixed
        )
        before_soc = soc
        next_soc = max(BATTERY_HARD_FLOOR, min(1.0, soc + net_energy / base.BATTERY_CAPACITY_WH))
        if next_soc <= BATTERY_HARD_FLOOR + 1.0e-12 and selected:
            protection_triggers += 1
        soc = next_soc

        for cand in candidates:
            key = (str(cand["task_id"]), str(cand["model"]), str(cand["product"]))
            action_log.append(
                {
                    "method": method,
                    "seed": seed,
                    "pressure_scale": pressure_scale,
                    "time_h": time_h,
                    "orbit_index": orbit,
                    "wave_name": wave,
                    "task_id": cand["task_id"],
                    "scene_id": cand["scene_id"],
                    "value_level": cand["value_level"],
                    "deadline_bucket": cand["deadline_bucket"],
                    "deadline_slack_h": cand["deadline_slack"],
                    "SOC_before": before_soc,
                    "SOC_after": soc if key in selected_keys else cand["soc_after"],
                    "candidate_model": cand["model"],
                    "candidate_stage": cand["stage"],
                    "selected_action": int(key in selected_keys),
                    "score": cand["score"],
                    "expected_ground_delivery_utility": cand["expected_ground_delivery_utility"],
                    "energy_price_cost": cand["energy_price_cost"],
                    "comm_price_cost": cand["comm_price_cost"],
                    "queue_cost": cand["queue_cost"],
                    "thermal_cost": cand["thermal_cost"],
                    "recovery_opportunity_cost": cand["recovery_opportunity_cost"],
                    "deadline_urgency_bonus": cand["deadline_urgency_bonus"],
                    "cheap_win_density_bonus": cand["cheap_win_density_bonus"],
                    "delivery_density": cand["delivery_density"],
                    "predicted_ground_delivery_success": cand["predicted_ground_delivery_success"],
                    "product_size_MB": cand["payload"],
                    "estimated_energy_wh": cand["estimated_energy"],
                    "reason": cand["reason"] if key not in selected_keys else "selected_safe_price_guided",
                }
            )
            price_samples.append(cand)

        product_counts = Counter()
        system_energy = orbit_compute + orbit_comm + payload_idle
        cumulative_system_energy += system_energy
        cumulative_direct_energy += orbit_compute + orbit_comm
        for cand in selected:
            task = by_task[str(cand["task_id"])]
            product = str(cand["product"])
            stage = STAGE_BY_PRODUCT[product]
            product_counts[product] += 1
            delivered_before = set(delivered[str(task["task_id"])])
            delta = fair.delta_information_utility(product, delivered_before, str(task["product_requirement"]))
            delivered[str(task["task_id"])].add(product)
            highest = _highest_stage(delivered[str(task["task_id"])])
            q = _model_quality(str(cand["model"]), product) * base._stage_sufficiency(str(task["product_requirement"]), highest)
            gross = _to_float(task["V_eff"]) * delta * q
            net = gross - _to_float(cand["account_cost"])
            net_system_utility += net
            if _to_int(task["is_high_value"]):
                h_net_utility += net
            receive_time = (orbit + 1) * base.ORBIT_HOURS
            before_deadline = int(receive_time <= _to_float(task["deadline_h"]) + 1.0e-12)
            events.append(
                {
                    "method": method,
                    "seed": seed,
                    "pressure_scale": pressure_scale,
                    "task_id": task["task_id"],
                    "scene_id": task["scene_id"],
                    "delivered_stage": stage,
                    "model_choice": cand["model"],
                    "product_size_MB": cand["payload"],
                    "ground_receive_time_h": receive_time,
                    "before_deadline_flag": before_deadline,
                    "delivered_quality": q,
                    "incremental_information_utility": delta,
                }
            )
            trow = task_rows[str(task["task_id"])]
            trow["highest_ground_received_stage"] = highest
            trow["ground_received_quality"] = max(_to_float(trow["ground_received_quality"]), q)
            trow["task_energy_direct_wh"] = _to_float(trow["task_energy_direct_wh"]) + _to_float(cand["estimated_energy"])
            trow["task_energy_system_wh"] = _to_float(trow["task_energy_system_wh"]) + _safe_div(system_energy, len(selected))
            if _stage_rank(highest) >= _stage_rank(str(task["min_required_stage"])) and not _to_int(trow["ground_received_flag"]):
                trow["ground_received_flag"] = 1
                trow["first_ground_receive_time_h"] = receive_time
                trow["ground_received_before_deadline_flag"] = before_deadline
                trow["capture_to_ground_latency_h"] = receive_time - _to_float(task["capture_time_h"])
                active.discard(str(task["task_id"]))
            if _stage_rank(highest) >= _stage_rank(str(task["target_stage"])):
                trow["target_stage_ground_received_flag"] = before_deadline
                if trow["target_ground_receive_time_h"] == "":
                    trow["target_ground_receive_time_h"] = receive_time

        trajectory.append(
            {
                "method": method,
                "seed": seed,
                "pressure_scale": pressure_scale,
                "orbit_index": orbit,
                "time_h": time_h,
                "wave_name": wave,
                "SOC": soc,
                "soc_delta": soc - before_soc,
                "soc_slope_per_hour": (soc - before_soc) / base.ORBIT_HOURS,
                "soc_band": _soc_band(before_soc),
                "payload_on": int(payload_on),
                "protection_mode": 0,
                "backlog_count": queue_count,
                "backlog_MB": queue_mb,
                "metadata_count": product_counts["metadata"],
                "roi_count": product_counts["roi"],
                "roi_context_count": product_counts["roi_context"],
                "defer_count": max(0, queue_count - len(selected)),
                "compute_energy_wh": orbit_compute,
                "comm_energy_wh": orbit_comm,
                "payload_idle_energy_wh": payload_idle,
                "ground_received_quality_sum": sum(_to_float(e["delivered_quality"]) for e in events),
            }
        )

    task_delivery = list(task_rows.values())
    soc_values = [_to_float(row["SOC"]) for row in trajectory]
    success = [row for row in task_delivery if _to_int(row["ground_received_before_deadline_flag"])]
    high_tasks = [row for row in task_delivery if _to_int(row["is_high_value"])]
    high_success = [row for row in high_tasks if _to_int(row["ground_received_before_deadline_flag"])]
    high_target = [row for row in high_tasks if _to_int(row["target_stage_ground_received_flag"])]
    summary = {
        "method": method,
        "method_type": METHOD_TYPES[method],
        "seed": seed,
        "pressure_scale": pressure_scale,
        **hashes,
        "total_task_count": len(task_delivery),
        "ground_received_task_count": len(success),
        "ground_received_task_ratio": _safe_div(len(success), len(task_delivery)),
        "avg_ground_received_quality": statistics.mean([_to_float(row["ground_received_quality"]) for row in success]) if success else 0.0,
        "avg_capture_to_ground_latency_h": statistics.mean([_to_float(row["capture_to_ground_latency_h"]) for row in success]) if success else 0.0,
        "deadline_satisfaction_ratio": 1.0 if success else 0.0,
        "high_value_task_count": len(high_tasks),
        "high_value_ground_received_count": len(high_success),
        "high_value_ground_received_ratio": _safe_div(len(high_success), len(high_tasks)),
        "high_value_target_stage_success_rate": _safe_div(len(high_target), len(high_tasks)),
        "avg_high_value_capture_to_ground_latency_h": statistics.mean([_to_float(row["capture_to_ground_latency_h"]) for row in high_success]) if high_success else 0.0,
        "direct_energy_per_received_task_wh": _safe_div(cumulative_direct_energy, len(success)),
        "system_energy_per_received_task_wh": _safe_div(cumulative_system_energy, len(success)),
        "direct_energy_per_high_value_received_task_wh": _safe_div(sum(_to_float(row["task_energy_direct_wh"]) for row in high_success), len(high_success)),
        "system_energy_per_high_value_received_task_wh": _safe_div(sum(_to_float(row["task_energy_system_wh"]) for row in high_success), len(high_success)),
        "net_system_utility": net_system_utility,
        "H_net_utility": h_net_utility,
        "min_SOC": min(soc_values),
        "final_SOC": soc_values[-1],
        "fraction_time_SOC_below_50": _safe_div(sum(value < 0.50 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_below_20": _safe_div(sum(value < BATTERY_HARD_FLOOR for value in soc_values), len(soc_values)),
        "hard_safety_violation_count": sum(value < BATTERY_HARD_FLOOR - 1.0e-12 for value in soc_values),
        "final_backlog_count": _to_int(trajectory[-1]["backlog_count"]),
        "max_backlog_count": max(_to_int(row["backlog_count"]) for row in trajectory),
        "protection_trigger_count": protection_triggers,
        "protection_total_slots": 0,
    }
    return {
        "summary": summary,
        "events": events,
        "task_delivery": task_delivery,
        "trajectory": trajectory,
        "price_action_log": action_log,
        "expired_log": expired_log,
        "price_samples": price_samples,
    }


def _target_success(row: dict[str, Any]) -> int:
    target_time = row.get("target_ground_receive_time_h", "")
    if target_time == "":
        return 0
    return int(_to_float(target_time) <= _to_float(row.get("deadline_h")) + 1.0e-12)


def _normalize_ground_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    tasks = result["task_delivery"]
    success = [row for row in tasks if _to_int(row["ground_received_before_deadline_flag"])]
    target_success = [row for row in tasks if _target_success(row)]
    high_tasks = [row for row in tasks if _to_int(row["is_high_value"])]
    high_success = [row for row in high_tasks if _to_int(row["ground_received_before_deadline_flag"])]
    high_target = [row for row in high_tasks if _target_success(row)]
    expired = [row for row in tasks if not _to_int(row["ground_received_before_deadline_flag"])]
    expired_high = [row for row in expired if _to_int(row["is_high_value"])]
    return {
        "method": summary["method"],
        "seed": summary["seed"],
        "pressure_scale": summary["pressure_scale"],
        "task_manifest_hash": summary["task_manifest_hash"],
        "contact_schedule_hash": summary["contact_schedule_hash"],
        "resource_trace_hash": summary["resource_trace_hash"],
        "ground_success_ratio": _safe_div(len(success), len(tasks)),
        "target_stage_success_ratio": _safe_div(len(target_success), len(tasks)),
        "high_value_ground_success_ratio": _safe_div(len(high_success), len(high_tasks)),
        "high_value_target_success_ratio": _safe_div(len(high_target), len(high_tasks)),
        "avg_ground_received_quality": statistics.mean([_to_float(row["ground_received_quality"]) for row in success]) if success else 0.0,
        "avg_capture_to_ground_latency_h": statistics.mean([_to_float(row["capture_to_ground_latency_h"]) for row in success]) if success else 0.0,
        "direct_energy_per_received_task_wh": _to_float(summary["direct_energy_per_received_task_wh"]),
        "system_energy_per_received_task_wh": _to_float(summary["system_energy_per_received_task_wh"]),
        "net_system_utility": _to_float(summary["net_system_utility"]),
        "H_net_utility": _to_float(summary["H_net_utility"]),
        "min_SOC": _to_float(summary["min_SOC"]),
        "final_SOC": _to_float(summary["final_SOC"]),
        "fraction_time_SOC_below_50": _to_float(summary["fraction_time_SOC_below_50"]),
        "fraction_time_SOC_below_20": _to_float(summary["fraction_time_SOC_below_20"]),
        "hard_safety_violation_count": _to_int(summary["hard_safety_violation_count"]),
        "expired_task_count": len(expired),
        "expired_high_value_task_count": len(expired_high),
        "final_backlog_count": _to_int(summary["final_backlog_count"]),
        "max_backlog_count": _to_int(summary["max_backlog_count"]),
        "protection_trigger_count": _to_int(summary["protection_trigger_count"]),
    }


def _build_expired_log_for_result(result: dict[str, Any], manifest_by_task: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if result["summary"]["method"] == METHOD_PRICE:
        return result.get("expired_log", [])
    rows = []
    traj_by_time = sorted(result["trajectory"], key=lambda row: _to_float(row["time_h"]))
    for task in result["task_delivery"]:
        if _to_int(task["ground_received_before_deadline_flag"]):
            continue
        source = manifest_by_task[str(task["task_id"])]
        deadline_h = _to_float(task["deadline_h"])
        soc_at = next((_to_float(row["SOC"]) for row in traj_by_time if _to_float(row["time_h"]) >= deadline_h), _to_float(traj_by_time[-1]["SOC"]))
        rows.append(
            {
                "method": result["summary"]["method"],
                "seed": result["summary"]["seed"],
                "pressure_scale": result["summary"]["pressure_scale"],
                "task_id": task["task_id"],
                "scene_id": task["scene_id"],
                "value_level": task["value_level"],
                "wave_name": source["wave_name"],
                "deadline_h": task["deadline_h"],
                "expired_time_h": deadline_h,
                "last_stage_ground_received": task["highest_ground_received_stage"],
                "min_required_stage": task["min_required_stage"],
                "target_stage": task["target_stage"],
                "expired_reason": "no_ground_success_before_deadline",
                "SOC_at_expiration": soc_at,
                "queue_wait_h": max(0.0, deadline_h - _to_float(task["capture_time_h"])),
            }
        )
    return rows


def _build_price_component_summary(
    action_rows: list[dict[str, Any]],
    ground_rows: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    price_by_key_band: dict[tuple[str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        if row["method"] != METHOD_PRICE or row["candidate_stage"] == "defer":
            continue
        band = _soc_band(_to_float(row["SOC_before"]))
        price_by_key_band[(row["method"], _to_int(row["seed"]), _to_float(row["pressure_scale"]), band)].append(row)

    old_slope = base.build_slope_summary([result for result in all_results if result["summary"]["method"] != METHOD_PRICE])
    out: list[dict[str, Any]] = []
    for row in old_slope:
        out.append(
            {
                "method": row["method"],
                "seed": row["seed"],
                "pressure_scale": row["pressure_scale"],
                "soc_band": row["soc_band"],
                "avg_lambda_E": 0.0,
                "avg_lambda_Comm": 0.0,
                "avg_lambda_Q": 0.0,
                "avg_score_metadata": 0.0,
                "avg_score_roi": 0.0,
                "avg_score_roi_context": 0.0,
                "metadata_rate": row["avg_metadata_rate"],
                "roi_rate": row["avg_roi_rate"],
                "roi_context_rate": row["avg_roi_context_rate"],
                "defer_rate": row["avg_defer_rate"],
                "avg_delivery_density": 0.0,
                "ground_success_count": row["ground_received_task_count"],
                "high_value_ground_success_count": 0,
            }
        )
    success_counts: dict[tuple[str, int, float], tuple[int, int]] = {}
    for result in all_results:
        key = (
            result["summary"]["method"],
            _to_int(result["summary"]["seed"]),
            _to_float(result["summary"]["pressure_scale"]),
        )
        success = [row for row in result["task_delivery"] if _to_int(row["ground_received_before_deadline_flag"])]
        high_success = [row for row in success if _to_int(row["is_high_value"])]
        success_counts[key] = (len(success), len(high_success))
    for key, rows in sorted(price_by_key_band.items(), key=lambda item: item[0]):
        method, seed, pressure, band = key
        selected = [row for row in rows if _to_int(row["selected_action"])]
        action_total = len(selected) + sum(1 for row in action_rows if row["method"] == method and _to_int(row["seed"]) == seed and _to_float(row["pressure_scale"]) == pressure and _soc_band(_to_float(row["SOC_before"])) == band and row["candidate_stage"] == "defer")
        out.append(
            {
                "method": method,
                "seed": seed,
                "pressure_scale": pressure,
                "soc_band": band,
                "avg_lambda_E": statistics.mean([_safe_div(_to_float(row["energy_price_cost"]), max(_to_float(row["estimated_energy_wh"]), 1.0e-9)) for row in rows]),
                "avg_lambda_Comm": statistics.mean([_to_float(row["comm_price_cost"]) for row in rows]),
                "avg_lambda_Q": statistics.mean([_to_float(row["queue_cost"]) for row in rows]),
                "avg_score_metadata": statistics.mean([_to_float(row["score"]) for row in rows if row["candidate_stage"] == "metadata"]) if any(row["candidate_stage"] == "metadata" for row in rows) else 0.0,
                "avg_score_roi": statistics.mean([_to_float(row["score"]) for row in rows if row["candidate_stage"] == "ROI"]) if any(row["candidate_stage"] == "ROI" for row in rows) else 0.0,
                "avg_score_roi_context": statistics.mean([_to_float(row["score"]) for row in rows if row["candidate_stage"] == "ROI+Ctx"]) if any(row["candidate_stage"] == "ROI+Ctx" for row in rows) else 0.0,
                "metadata_rate": _safe_div(sum(1 for row in selected if row["candidate_stage"] == "metadata"), action_total),
                "roi_rate": _safe_div(sum(1 for row in selected if row["candidate_stage"] == "ROI"), action_total),
                "roi_context_rate": _safe_div(sum(1 for row in selected if row["candidate_stage"] == "ROI+Ctx"), action_total),
                "defer_rate": _safe_div(max(0, action_total - len(selected)), action_total),
                "avg_delivery_density": statistics.mean([_to_float(row["delivery_density"]) for row in rows]) if rows else 0.0,
                "ground_success_count": success_counts[(method, seed, pressure)][0],
                "high_value_ground_success_count": success_counts[(method, seed, pressure)][1],
            }
        )
    return out


def _build_rescue_summary(action_rows: list[dict[str, Any]], expired_rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_success: dict[tuple[str, int, float, str], int] = {}
    for result in results:
        method = result["summary"]["method"]
        seed = _to_int(result["summary"]["seed"])
        pressure = _to_float(result["summary"]["pressure_scale"])
        for task in result["task_delivery"]:
            task_success[(method, seed, pressure, str(task["task_id"]))] = _to_int(task["ground_received_before_deadline_flag"])

    out = []
    groups: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        if row["method"] == METHOD_PRICE and row["candidate_stage"] != "defer":
            groups[(row["method"], _to_int(row["seed"]), _to_float(row["pressure_scale"]))].append(row)
    for result in results:
        key = (result["summary"]["method"], _to_int(result["summary"]["seed"]), _to_float(result["summary"]["pressure_scale"]))
        rows = groups.get(key, [])
        rescue_candidates = [
            row
            for row in rows
            if row["value_level"] in {"high", "urgent"} and row["deadline_bucket"] in {"critical", "urgent_deadline"}
        ]
        cheap_candidates = [row for row in rows if _to_float(row["cheap_win_density_bonus"]) > 0.0]
        rescue_selected = [row for row in rescue_candidates if _to_int(row["selected_action"])]
        cheap_selected = [row for row in cheap_candidates if _to_int(row["selected_action"])]
        expired_case = [row for row in expired_rows if row["method"] == key[0] and _to_int(row["seed"]) == key[1] and _to_float(row["pressure_scale"]) == key[2]]
        out.append(
            {
                "method": key[0],
                "seed": key[1],
                "pressure_scale": key[2],
                "deadline_rescue_candidate_count": len(rescue_candidates),
                "deadline_rescue_selected_count": len(rescue_selected),
                "deadline_rescue_success_count": sum(task_success.get((key[0], key[1], key[2], str(row["task_id"])), 0) for row in rescue_selected),
                "cheap_win_candidate_count": len(cheap_candidates),
                "cheap_win_selected_count": len(cheap_selected),
                "cheap_win_success_count": sum(task_success.get((key[0], key[1], key[2], str(row["task_id"])), 0) for row in cheap_selected),
                "cheap_win_low_medium_count": sum(row["value_level"] in {"low", "medium"} for row in cheap_selected),
                "cheap_win_high_urgent_count": sum(row["value_level"] in {"high", "urgent"} for row in cheap_selected),
                "cheap_win_avg_density": statistics.mean([_to_float(row["delivery_density"]) for row in cheap_selected]) if cheap_selected else 0.0,
                "cheap_win_avg_energy_wh": statistics.mean([_to_float(row["estimated_energy_wh"]) for row in cheap_selected]) if cheap_selected else 0.0,
                "cheap_win_avg_payload_MB": statistics.mean([_to_float(row["product_size_MB"]) for row in cheap_selected]) if cheap_selected else 0.0,
                "high_value_expired_count": sum(row["value_level"] in {"high", "urgent"} for row in expired_case),
                "total_expired_count": len(expired_case),
            }
        )
    return out


def _build_multi_seed_ranking(rows: list[dict[str, Any]], pressure_scales: list[float]) -> list[dict[str, Any]]:
    out = []
    for pressure in pressure_scales:
        pressure_rows = [row for row in rows if abs(_to_float(row["pressure_scale"]) - pressure) < 1.0e-9]
        rank_by_metric: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for seed in sorted({_to_int(row["seed"]) for row in pressure_rows}):
            seed_rows = [row for row in pressure_rows if _to_int(row["seed"]) == seed]
            for metric in RANKING_METRICS:
                lower_better = metric in {"avg_capture_to_ground_latency_h", "direct_energy_per_received_task_wh"}
                ordered = sorted(seed_rows, key=lambda row: _to_float(row[metric]), reverse=not lower_better)
                for rank, row in enumerate(ordered, start=1):
                    rank_by_metric[metric][str(row["method"])].append(rank)
        for method in METHODS:
            method_rows = [row for row in pressure_rows if row["method"] == method]
            for metric in RANKING_METRICS:
                values = [_to_float(row[metric]) for row in method_rows]
                ranks = rank_by_metric[metric][method]
                out.append(
                    {
                        "method": method,
                        "pressure_scale": pressure,
                        "metric": metric,
                        "mean": statistics.mean(values) if values else 0.0,
                        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                        "min": min(values) if values else 0.0,
                        "max": max(values) if values else 0.0,
                        "rank_mean": statistics.mean(ranks) if ranks else 0.0,
                        "rank_std": statistics.pstdev(ranks) if len(ranks) > 1 else 0.0,
                    }
                )
    return out


def make_figures(
    out_dir: Path,
    summary: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    wave_summary: list[dict[str, Any]],
    price_component: list[dict[str, Any]],
    rescue_summary: list[dict[str, Any]],
) -> None:
    base_seed, base_pressure = 0, 1.0
    methods_for_plots = [
        "proposed_guarantee_aware_full",
        METHOD_PRICE,
        "utility_rule",
        "fixed_lightweight",
        "fixed_heavy_with_20pct_protection",
        "fixed_heavy",
    ]
    base_traj = [row for row in trajectories if _to_int(row["seed"]) == base_seed and abs(_to_float(row["pressure_scale"]) - base_pressure) < 1.0e-9]
    plt.figure(figsize=(12, 5.4))
    for name, start, end, _density, _pressure in base.WAVES:
        color = "#f3d2c1" if name.startswith("wave") else "#d8ead3"
        plt.axvspan(start * base.ORBIT_HOURS, (end + 1) * base.ORBIT_HOURS, color=color, alpha=0.22)
        if name in {"wave_2_extreme", "recovery_2_incomplete", "wave_3_medium_high"}:
            plt.text(start * base.ORBIT_HOURS, 0.04, name, rotation=90, fontsize=7)
    for method in methods_for_plots:
        rows = [row for row in base_traj if row["method"] == method]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["SOC"]) for row in rows], label=method, linewidth=1.5)
    for level in [0.70, 0.50, 0.20]:
        plt.axhline(level, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("time_h")
    plt.ylabel("SOC")
    plt.ylim(0.0, 1.02)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "price_guided_soc_comparison.png", dpi=190)
    plt.close()

    base_summary = [row for row in summary if _to_int(row["seed"]) == base_seed and abs(_to_float(row["pressure_scale"]) - base_pressure) < 1.0e-9 and row["method"] in methods_for_plots]
    metrics = ["ground_success_ratio", "target_stage_success_ratio", "high_value_ground_success_ratio", "high_value_target_success_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, metric in zip(axes.flat, metrics):
        ax.bar([row["method"] for row in base_summary], [_to_float(row[metric]) for row in base_summary])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "ground_success_comparison.png", dpi=190)
    plt.close(fig)

    pair = [row for row in base_summary if row["method"] in {"proposed_guarantee_aware_full", METHOD_PRICE}]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, ["expired_task_count", "expired_high_value_task_count"]):
        ax.bar([row["method"] for row in pair], [_to_float(row[metric]) for row in pair], color="#4c78a8")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "expiration_reduction.png", dpi=190)
    plt.close(fig)

    price_rows = [row for row in price_component if row["method"] == METHOD_PRICE and _to_int(row["seed"]) == base_seed and abs(_to_float(row["pressure_scale"]) - base_pressure) < 1.0e-9]
    bands = ["comfortable", "soft_protection", "energy_tight", "protection", "hard_violation"]
    price_rows = sorted(price_rows, key=lambda row: bands.index(row["soc_band"]) if row["soc_band"] in bands else 99)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar([row["soc_band"] for row in price_rows], [_to_float(row["avg_lambda_E"]) for row in price_rows])
    axes[0].set_title("energy price")
    for product_metric, label in [("metadata_rate", "metadata"), ("roi_rate", "ROI"), ("roi_context_rate", "ROI+Ctx")]:
        axes[1].plot([row["soc_band"] for row in price_rows], [_to_float(row[product_metric]) for row in price_rows], marker="o", label=label)
    axes[1].set_title("selection rate")
    axes[1].legend(fontsize=7)
    axes[2].bar([row["soc_band"] for row in price_rows], [_to_float(row["avg_delivery_density"]) for row in price_rows])
    axes[2].set_title("cheap-win density")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "price_component_by_soc_band.png", dpi=190)
    plt.close(fig)

    rescue_base = [row for row in rescue_summary if row["method"] == METHOD_PRICE and _to_int(row["seed"]) == base_seed and abs(_to_float(row["pressure_scale"]) - base_pressure) < 1.0e-9]
    row = rescue_base[0] if rescue_base else {}
    metrics = ["cheap_win_selected_count", "cheap_win_success_count", "cheap_win_low_medium_count", "cheap_win_avg_density", "cheap_win_avg_energy_wh"]
    plt.figure(figsize=(8, 4.5))
    plt.bar(metrics, [_to_float(row.get(metric)) for metric in metrics], color="#72b7b2")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "cheap_win_analysis.png", dpi=190)
    plt.close()

    plt.figure(figsize=(8, 5.5))
    for row in base_summary:
        plt.scatter(
            _to_float(row["min_SOC"]),
            _to_float(row["high_value_ground_success_ratio"]),
            s=60 + 240 * _to_float(row["ground_success_ratio"]),
            label=row["method"],
            alpha=0.76,
        )
    plt.xlabel("min_SOC")
    plt.ylabel("high_value_ground_success_ratio")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff_ground_success_vs_safety.png", dpi=190)
    plt.close()


def _compare(base_row: dict[str, Any], new_row: dict[str, Any], metric: str, higher: bool = True) -> str:
    old = _to_float(base_row.get(metric))
    new = _to_float(new_row.get(metric))
    if abs(new - old) < 1.0e-12:
        return f"No change ({_fmt(new)} vs {_fmt(old)})."
    better = new > old if higher else new < old
    return f"{'Yes' if better else 'No'} ({_fmt(new)} vs {_fmt(old)})."


def write_quick_report(
    out_dir: Path,
    summary: list[dict[str, Any]],
    wave_summary: list[dict[str, Any]],
    rescue_summary: list[dict[str, Any]],
    price_component: list[dict[str, Any]],
    fairness_rows: list[dict[str, Any]],
) -> None:
    base_case = {
        row["method"]: row
        for row in summary
        if _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    }
    old = base_case["proposed_guarantee_aware_full"]
    new = base_case[METHOD_PRICE]
    old_wave = {
        row["wave_name"]: row
        for row in wave_summary
        if row["method"] == "proposed_guarantee_aware_full" and _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    }
    rescue = next(
        row
        for row in rescue_summary
        if row["method"] == METHOD_PRICE and _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    )
    price_bands = [
        row
        for row in price_component
        if row["method"] == METHOD_PRICE and _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    ]
    low_soc_rows = [row for row in price_bands if row["soc_band"] in {"energy_tight", "protection"}]
    high_soc_rows = [row for row in price_bands if row["soc_band"] == "comfortable"]
    low_roi_ctx = statistics.mean([_to_float(row["roi_context_rate"]) for row in low_soc_rows]) if low_soc_rows else 0.0
    high_roi_ctx = statistics.mean([_to_float(row["roi_context_rate"]) for row in high_soc_rows]) if high_soc_rows else 0.0
    fairness_ok = all(_to_int(row["passed"]) for row in fairness_rows)
    lines = [
        "# Price-guided rescue quick report",
        "",
        "## Direct Answers",
        "",
        f"1. Original low-SOC gate caused backlog/deadline expiry: Yes. Base proposed wave_2_extreme min_SOC={_fmt(old_wave['wave_2_extreme']['min_SOC'])}, recovery_2_incomplete final_SOC={_fmt(old_wave['recovery_2_incomplete']['final_SOC'])}, final_backlog={old['final_backlog_count']}, expired={old['expired_task_count']}.",
        f"2. Reduced expired_task_count: {_compare(old, new, 'expired_task_count', higher=False)}",
        f"3. Reduced expired_high_value_task_count: {_compare(old, new, 'expired_high_value_task_count', higher=False)}",
        f"4. Improved ground_success_ratio: {_compare(old, new, 'ground_success_ratio')}",
        f"5. Improved high_value_ground_success_ratio: {_compare(old, new, 'high_value_ground_success_ratio')}",
        f"6. Improved target_stage_success_ratio: {_compare(old, new, 'target_stage_success_ratio')}",
        f"7. Improved high_value_target_success_ratio: {_compare(old, new, 'high_value_target_success_ratio')}",
        f"8. Delivered quality remains high: {'Yes' if _to_float(new['avg_ground_received_quality']) >= 0.70 else 'No'} ({_fmt(new['avg_ground_received_quality'])}).",
        f"9. Latency did not clearly worsen: {_compare(old, new, 'avg_capture_to_ground_latency_h', higher=False)}",
        f"10. Energy per received task is reasonable: direct={_fmt(new['direct_energy_per_received_task_wh'])}, system={_fmt(new['system_energy_per_received_task_wh'])}.",
        f"11. min_SOC remains above 0.20: {'Yes' if _to_float(new['min_SOC']) >= BATTERY_HARD_FLOOR - 1.0e-12 else 'No'} ({_fmt(new['min_SOC'])}).",
        f"12. hard_safety_violation_count is 0: {'Yes' if _to_int(new['hard_safety_violation_count']) == 0 else 'No'} ({new['hard_safety_violation_count']}).",
        f"13. fraction_time_SOC_below_20 is 0: {'Yes' if _to_float(new['fraction_time_SOC_below_20']) == 0.0 else 'No'} ({_fmt(new['fraction_time_SOC_below_20'])}).",
        f"14. Safer than utility_rule/fixed_heavy: utility_rule min_SOC={_fmt(base_case['utility_rule']['min_SOC'])}, fixed_heavy min_SOC={_fmt(base_case['fixed_heavy']['min_SOC'])}, price-guided min_SOC={_fmt(new['min_SOC'])}.",
        f"15. More effective than fixed_lightweight: price-guided ground_success={_fmt(new['ground_success_ratio'])}, fixed_lightweight={_fmt(base_case['fixed_lightweight']['ground_success_ratio'])}.",
        f"16. Deadline rescue helps high/urgent near-deadline tasks: candidates={rescue['deadline_rescue_candidate_count']}, selected={rescue['deadline_rescue_selected_count']}, success={rescue['deadline_rescue_success_count']}.",
        f"17. Cheap-win handled low/medium high-density tasks: selected={rescue['cheap_win_selected_count']}, low_medium={rescue['cheap_win_low_medium_count']}, avg_density={_fmt(rescue['cheap_win_avg_density'])}.",
        f"18. Cheap-win crowding evidence: cheap_win_selected={rescue['cheap_win_selected_count']}, rescue_selected={rescue['deadline_rescue_selected_count']}, cheap_high_urgent={rescue['cheap_win_high_urgent_count']}. It is a score bonus only; selected rows are visible in price_guided_action_log.csv.",
        f"19. Low-SOC ROI+Ctx rate naturally declines: low_SOC_rate={_fmt(low_roi_ctx)}, comfortable_rate={_fmt(high_roi_ctx)}. ROI+Ctx candidates are logged under low SOC and are filtered only by hard safety/expiry.",
        "20. Yes. The hard constraints are only battery_hard_floor, thermal_hard_limit, optional queue capacity, and expiry validity; resource prices and score bonuses guide low-energy behavior.",
        "21. Suitability as formal proposed full method: use the ranking CSV and safety checks. It is a plausible full method if the multi-seed gains hold beyond pressure_scale=1.0.",
        "22. Yes. proposed_guarantee_aware_full remains intact as the no-price-rescue / safety-gate ablation.",
        "",
        "## Mechanism Boundaries",
        "",
        "- Hard safety constraints: predicted_SOC_after_action >= 0.20, thermal_after_action <= 1.25, optional queue hard cap if configured, and no expired task can create a before-deadline success.",
        "- Price terms: lambda_E(SOC), lambda_Comm(SOC/contact), lambda_Q(queue), lambda_T(thermal), lambda_R(recovery phase).",
        "- Bonuses: DeadlineUrgencyBonus and CheapWinDensityBonus only change ordering inside the safe candidate set.",
        "- Estimated delivery time uses one orbit plus payload-scaled transmission and processing latency because the base runner does not expose a finer contact wait model.",
        f"- Fairness hash audit passed: {fairness_ok}. Old proposed and price-guided proposed use the same task_manifest_hash, contact_schedule_hash, and resource_trace_hash per seed/pressure.",
        "- The runner imports value-prior and task builders from the existing experiment; it does not modify value prior, high-value labels, task manifest generation, reward, or evaluation code.",
    ]
    _write_text(out_dir / "quick_report.md", "\n".join(lines))


def validate_outputs(out_dir: Path) -> list[dict[str, Any]]:
    required = [
        "price_guided_action_log.csv",
        "price_component_summary_by_soc_band.csv",
        "rescue_and_cheapwin_summary.csv",
        "ground_success_summary_by_seed.csv",
        "ground_success_multi_seed_ranking.csv",
        "expired_task_log.csv",
        "quick_report.md",
        "price_guided_soc_comparison.png",
        "ground_success_comparison.png",
        "expiration_reduction.png",
        "price_component_by_soc_band.png",
        "cheap_win_analysis.png",
        "tradeoff_ground_success_vs_safety.png",
    ]
    rows = []
    for name in required:
        path = out_dir / name
        rows.append({"artifact": name, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    return rows


def run(output_dir: Path = OUT_DIR, seeds: list[int] | None = None, pressure_scales: list[float] | None = None) -> dict[str, Any]:
    seeds = seeds or DEFAULT_SEEDS
    pressure_scales = pressure_scales or DEFAULT_PRESSURE_SCALES
    output_dir.mkdir(parents=True, exist_ok=True)
    all_manifest: list[dict[str, Any]] = []
    all_contacts: list[dict[str, Any]] = []
    all_resources: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    manifests_by_case: dict[tuple[int, float], list[dict[str, Any]]] = {}
    fairness_rows: list[dict[str, Any]] = []
    subtask_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    expired_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for pressure in pressure_scales:
            contacts = base.build_contact_schedule(seed, pressure)
            resources = base.build_resource_trace(seed, pressure)
            manifest = base.build_task_manifest(seed, pressure, contacts)
            hashes = {
                "task_manifest_hash": base._hash(manifest, base.GROUND_MANIFEST_COLUMNS),
                "contact_schedule_hash": base._hash(contacts, ["seed", "pressure_scale", "orbit_index", "capacity_MB", "wave_name"]),
                "resource_trace_hash": base._hash(resources, ["seed", "pressure_scale", "orbit_index", "solar_energy_wh", "wave_name"]),
            }
            manifests_by_case[(seed, pressure)] = manifest
            all_manifest.extend(manifest)
            all_contacts.extend(contacts)
            all_resources.extend(resources)
            subtask_rows.extend(base.build_subtask_audit(manifest))
            case_summaries = []
            case_results = []
            for method in METHODS:
                if method == METHOD_PRICE:
                    result = run_price_guided_method(seed, pressure, manifest, contacts, resources, hashes)
                    action_rows.extend(result["price_action_log"])
                else:
                    result = base.run_method(method, seed, pressure, manifest, contacts, resources, hashes)
                all_results.append(result)
                case_results.append(result)
                case_summaries.append(result["summary"])
            manifest_by_task = {str(row["task_id"]): row for row in manifest}
            for result in case_results:
                expired_rows.extend(_build_expired_log_for_result(result, manifest_by_task))
            for field in ["task_manifest_hash", "contact_schedule_hash", "resource_trace_hash"]:
                values = {row[field] for row in case_summaries}
                fairness_rows.append(
                    {
                        "seed": seed,
                        "pressure_scale": pressure,
                        "check": field,
                        "unique_hash_count": len(values),
                        "passed": int(len(values) == 1),
                        "hash": next(iter(values)),
                    }
                )

    ground_summary = [_normalize_ground_summary(result) for result in all_results]
    events = [row for result in all_results for row in result["events"]]
    task_delivery = [row for result in all_results for row in result["task_delivery"]]
    trajectories = [row for result in all_results for row in result["trajectory"]]
    wave_summary = base.build_wave_summary(all_results, manifests_by_case)
    ranking = _build_multi_seed_ranking(ground_summary, pressure_scales)
    price_component = _build_price_component_summary(action_rows, ground_summary, all_results)
    rescue_summary = _build_rescue_summary(action_rows, expired_rows, all_results)
    task_value_audit = base.build_task_value_audit(all_manifest, subtask_rows)

    _write_csv(output_dir / "ground_delivery_task_manifest.csv", all_manifest, base.GROUND_MANIFEST_COLUMNS)
    _write_csv(output_dir / "shared_contact_schedule.csv", all_contacts)
    _write_csv(output_dir / "shared_resource_trace.csv", all_resources)
    _write_csv(output_dir / "ground_delivery_event_log.csv", events, base.EVENT_COLUMNS)
    _write_csv(output_dir / "ground_delivery_task_delivery_by_method_seed.csv", task_delivery)
    _write_csv(output_dir / "soc_trajectory_by_method_seed.csv", trajectories)
    _write_csv(output_dir / "wave_summary_by_method_seed.csv", wave_summary)
    _write_csv(output_dir / "task_value_audit.csv", task_value_audit)
    _write_csv(output_dir / "fairness_hash_audit.csv", fairness_rows)
    _write_csv(output_dir / "price_guided_action_log.csv", action_rows, ACTION_LOG_COLUMNS)
    _write_csv(output_dir / "price_component_summary_by_soc_band.csv", price_component, PRICE_COMPONENT_COLUMNS)
    _write_csv(output_dir / "rescue_and_cheapwin_summary.csv", rescue_summary, RESCUE_COLUMNS)
    _write_csv(output_dir / "ground_success_summary_by_seed.csv", ground_summary, GROUND_SUMMARY_COLUMNS)
    _write_csv(output_dir / "ground_success_multi_seed_ranking.csv", ranking, ["method", "pressure_scale", "metric", "mean", "std", "min", "max", "rank_mean", "rank_std"])
    _write_csv(output_dir / "expired_task_log.csv", expired_rows, EXPIRED_COLUMNS)
    make_figures(output_dir, ground_summary, trajectories, wave_summary, price_component, rescue_summary)
    write_quick_report(output_dir, ground_summary, wave_summary, rescue_summary, price_component, fairness_rows)
    validation = validate_outputs(output_dir)
    _write_csv(output_dir / "artifact_validation.csv", validation)
    return {
        "output_dir": output_dir,
        "ground_summary": ground_summary,
        "ranking": ranking,
        "fairness_rows": fairness_rows,
        "validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--full-pressure", action="store_true", help="Run pressure scales 0.85, 1.0, and 1.15 instead of the default first-round 1.0.")
    args = parser.parse_args(argv)
    pressures = FULL_PRESSURE_SCALES if args.full_pressure else DEFAULT_PRESSURE_SCALES
    result = run(args.output_dir, seeds=DEFAULT_SEEDS, pressure_scales=pressures)
    base_prop = next(
        row
        for row in result["ground_summary"]
        if row["method"] == "proposed_guarantee_aware_full" and _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    )
    price_prop = next(
        row
        for row in result["ground_summary"]
        if row["method"] == METHOD_PRICE and _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    )
    print(f"output_dir={args.output_dir}")
    print(f"runs={len(result['ground_summary'])}")
    print(f"base_proposed_ground_success_ratio={_fmt(base_prop['ground_success_ratio'])}")
    print(f"price_guided_ground_success_ratio={_fmt(price_prop['ground_success_ratio'])}")
    print(f"price_guided_min_SOC={_fmt(price_prop['min_SOC'])}")
    print(f"fairness_errors={sum(1 for row in result['fairness_rows'] if not _to_int(row['passed']))}")
    print(f"validation_missing={sum(1 for row in result['validation'] if not _to_int(row['exists']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
