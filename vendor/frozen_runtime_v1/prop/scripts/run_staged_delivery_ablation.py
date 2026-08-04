"""Standalone high-value staged delivery ablation.

This runner builds the realistic workload/contact benchmark and evaluates
local policy variants for high-value staged residual delivery. It is intended
to be removable as a single script plus one output directory. It does not edit
src algorithms, value-prior logic, high-value labels, reward/evaluator code, or
older runners.
"""

from __future__ import annotations

import argparse
import csv
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

import scripts.run_fair_baseline_comparison as fair  # noqa: E402
import scripts.run_price_guided_rescue_ground_delivery_experiment as price  # noqa: E402
import scripts.run_realistic_workload_contact_sensitivity as realistic  # noqa: E402
import scripts.run_uneven_pressure_ground_delivery_experiment as base  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_staged_delivery_ablation_v1"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
VARIANTS = [
    "full_staged_delivery",
    "w/o_staged_delivery",
    "ordinary_priority_only",
    "w/o_future_contact_feasibility",
    "w/o_guarantee_debt",
    "ground_approved_followup",
    "no_progressive_payload",
]
STAGED_VARIANTS = {
    "full_staged_delivery",
    "w/o_future_contact_feasibility",
    "w/o_guarantee_debt",
    "ground_approved_followup",
}
SUMMARY_COLUMNS = [
    "variant",
    "seed",
    "workload_scenario",
    "contact_scenario",
    "scenario",
    "task_manifest_hash",
    "contact_schedule_hash",
    "resource_trace_hash",
    "value_prior_hash",
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "ground_received_task_count",
    "high_value_ground_received_count",
    "avg_ground_received_quality",
    "avg_high_value_ground_received_quality",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "net_system_utility",
    "H_net_utility",
    "expired_task_count",
    "expired_high_value_task_count",
    "avg_capture_to_ground_latency_h",
    "avg_high_value_capture_to_ground_latency_h",
    "staged_request_count",
    "staged_request_enqueued_count",
    "staged_request_delivered_count",
    "staged_request_expired_count",
    "staged_request_blocked_count",
    "residual_payload_delivered_count",
    "metadata_only_success_count",
    "roi_success_count",
    "roi_context_success_count",
    "high_value_metadata_only_count",
    "high_value_roi_or_context_count",
    "target_stage_delivered_count",
    "future_contact_blocked_count",
    "guarantee_assisted_candidate_count",
    "guarantee_priority_improved_rank_count",
    "min_SOC",
    "final_SOC",
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
    "ground_received_task_count",
    "high_value_ground_received_count",
    "avg_ground_received_quality",
    "avg_high_value_ground_received_quality",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "net_system_utility",
    "H_net_utility",
    "expired_task_count",
    "expired_high_value_task_count",
    "avg_capture_to_ground_latency_h",
    "avg_high_value_capture_to_ground_latency_h",
    "staged_request_count",
    "staged_request_enqueued_count",
    "staged_request_delivered_count",
    "staged_request_expired_count",
    "staged_request_blocked_count",
    "residual_payload_delivered_count",
    "metadata_only_success_count",
    "roi_success_count",
    "roi_context_success_count",
    "high_value_metadata_only_count",
    "high_value_roi_or_context_count",
    "target_stage_delivered_count",
    "future_contact_blocked_count",
    "guarantee_assisted_candidate_count",
    "guarantee_priority_improved_rank_count",
    "min_SOC",
    "final_SOC",
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
    "avg_high_value_capture_to_ground_latency_h",
    "staged_request_expired_count",
    "staged_request_blocked_count",
    "future_contact_blocked_count",
    "hard_safety_violation_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
    "queue_MB_peak",
    "final_backlog_count",
    "max_backlog_count",
}
EFFECT_METRICS = [
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_high_value_success_value",
    "expired_high_value_task_count",
    "target_stage_delivered_count",
    "staged_request_expired_count",
    "queue_MB_peak",
]
STAGE_BY_PRODUCT = {"metadata": "metadata", "roi": "ROI", "roi_context": "ROI+Ctx"}
PRODUCT_BY_STAGE = {"metadata": "metadata", "ROI": "roi", "ROI+Ctx": "roi_context"}


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


def _group(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def _hashes(manifest: list[dict[str, Any]], contacts: list[dict[str, Any]], resources: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "task_manifest_hash": fair._stable_hash(manifest, realistic.REALISTIC_MANIFEST_COLUMNS),
        "contact_schedule_hash": fair._stable_hash(
            contacts, ["seed", "workload_scenario", "contact_scenario", "orbit_index", "capacity_MB", "start_time_h"]
        ),
        "resource_trace_hash": fair._stable_hash(
            resources, ["seed", "workload_scenario", "contact_scenario", "orbit_index", "solar_energy_wh", "housekeeping_energy_wh"]
        ),
        "value_prior_hash": fair._stable_hash(
            manifest, ["task_id", "base_prior_score", "value_level", "value_multiplier", "V_eff", "value_assignment_reason"]
        ),
    }


def _thermal_pressure(wave: str, orbit_compute: float = 0.0, backlog_mb: float = 0.0, payload_on: bool = False) -> float:
    return price._thermal_pressure(wave, orbit_compute) + 0.0025 * min(backlog_mb, 60.0) + (0.04 if payload_on else 0.0)


def _target_product(task: dict[str, Any]) -> str:
    return PRODUCT_BY_STAGE.get(str(task["target_stage"]), "metadata")


def _highest_stage(delivered: set[str]) -> str:
    return base._highest_stage(delivered)


def _next_progressive_product(task: dict[str, Any], delivered: set[str]) -> str | None:
    target_rank = base._stage_rank(str(task["target_stage"]))
    for product in ("metadata", "roi", "roi_context"):
        if product in delivered:
            continue
        if base._stage_rank(STAGE_BY_PRODUCT[product]) <= target_rank:
            return product
    return None


def _one_shot_product(task: dict[str, Any], delivered: set[str], generated_once: set[str]) -> str | None:
    if str(task["task_id"]) in generated_once or delivered:
        return None
    return _target_product(task)


def _deadline_bucket(task: dict[str, Any], orbit: int, product: str) -> tuple[float, str]:
    payload = base._product_payload(task, product)
    return price._deadline_info(task, orbit, product, payload)


def _future_contact_feasible(
    *,
    product: str,
    task: dict[str, Any],
    current_orbit: int,
    contacts: dict[int, dict[str, Any]],
    include_current: bool = False,
) -> bool:
    payload = base._product_payload(task, product)
    start_orbit = current_orbit if include_current else current_orbit + 1
    for orbit, contact in sorted(contacts.items()):
        if orbit < start_orbit:
            continue
        receive_time = (orbit + 1) * base.ORBIT_HOURS
        if receive_time > _to_float(task["deadline_h"]) + 1.0e-12:
            break
        if _to_float(contact["capacity_MB"]) >= payload:
            return True
    return False


def _candidate_score(
    *,
    task: dict[str, Any],
    delivered: set[str],
    product: str,
    model: str,
    orbit: int,
    soc: float,
    capacity_mb: float,
    queue_mb: float,
    high_value_bonus: float,
    guarantee_debt: float = 0.0,
) -> dict[str, Any]:
    payload = base._product_payload(task, product)
    compute, comm, account_cost = price._candidate_costs(model, product, payload)
    delta = fair.delta_information_utility(product, delivered, str(task["product_requirement"]))
    quality = price._model_quality(model, product)
    deadline_slack, bucket = _deadline_bucket(task, orbit, product)
    soc_factor = max(0.2, min(1.0, (soc - base.HARD_MIN_SOC) / 0.45))
    capacity_factor = min(1.0, capacity_mb / max(payload, 1.0e-9))
    deadline_factor = {"critical": 1.15, "urgent_deadline": 1.05, "normal": 0.85, "expired": 0.0}[bucket]
    high = _to_int(task["is_high_value"]) > 0
    bonus = high_value_bonus if high else 1.0
    expected = _to_float(task["V_eff"]) * delta * quality * capacity_factor * soc_factor * deadline_factor * bonus
    energy = compute + comm
    density = expected / max(payload + 0.3 * energy, 1.0e-9)
    stage_bonus = {"metadata": 0.10, "roi": 0.22, "roi_context": 0.32}[product]
    urgency_bonus = price._deadline_bonus(task, delivered, product, bucket)
    score = expected + stage_bonus + urgency_bonus + 0.55 * guarantee_debt - 0.055 * energy - 0.05 * payload - 0.025 * queue_mb - account_cost
    return {
        "task_id": task["task_id"],
        "product": product,
        "stage": STAGE_BY_PRODUCT[product],
        "model": model,
        "payload": payload,
        "compute": compute,
        "comm": comm,
        "energy": energy,
        "account_cost": account_cost,
        "quality": quality,
        "delta": delta,
        "deadline_slack": deadline_slack,
        "deadline_bucket": bucket,
        "score": score,
        "density": density,
    }


def _request_priority(request: dict[str, Any], task: dict[str, Any], orbit: int, guarantee_enabled: bool) -> tuple[float, float]:
    payload = _to_float(request["payload"])
    residual = _to_float(request["residual_utility"])
    deadline_orbits = max(0.0, (_to_float(task["deadline_h"]) - orbit * base.ORBIT_HOURS) / max(base.ORBIT_HOURS, 1.0e-9))
    urgency = 1.0 / max(deadline_orbits, 0.35)
    value_level = str(task["value_level"])
    value_factor = {"urgent": 2.0, "high": 1.65, "medium": 1.0, "low": 0.7}.get(value_level, 1.0)
    base_score = residual * value_factor / max(payload, 0.05) + 1.2 * urgency + 0.12 * _to_int(request.get("age", 0))
    with_guarantee = base_score + (0.85 * _to_float(request.get("guarantee_debt", 0.0)) if guarantee_enabled else 0.0)
    return base_score, with_guarantee


def run_variant(
    *,
    variant: str,
    seed: int,
    workload: str,
    contact: str,
    manifest: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    hashes: dict[str, str],
) -> dict[str, Any]:
    staged_enabled = variant in STAGED_VARIANTS
    future_gate = variant != "w/o_future_contact_feasibility"
    guarantee_enabled = variant == "full_staged_delivery" or variant == "w/o_future_contact_feasibility"
    ground_followup = variant == "ground_approved_followup"
    one_shot = variant == "no_progressive_payload"
    high_value_bonus = {
        "full_staged_delivery": 1.15,
        "w/o_staged_delivery": 1.85,
        "ordinary_priority_only": 1.45,
        "w/o_future_contact_feasibility": 1.15,
        "w/o_guarantee_debt": 1.15,
        "ground_approved_followup": 1.15,
        "no_progressive_payload": 1.20,
    }[variant]

    by_task = {str(row["task_id"]): row for row in manifest}
    arrivals: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in manifest:
        arrivals[_to_int(task["orbit_index"])].append(task)
    contact_by_orbit = {_to_int(row["orbit_index"]): row for row in contacts}
    resource_by_orbit = {_to_int(row["orbit_index"]): row for row in resources}
    delivered: dict[str, set[str]] = defaultdict(set)
    generated_once: set[str] = set()
    active: set[str] = set()
    staged_requests: dict[tuple[str, str], dict[str, Any]] = {}
    task_rows: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    request_trace: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    stats = Counter()
    cumulative_direct_energy = 0.0
    cumulative_system_energy = 0.0
    net_system_utility = 0.0
    h_net_utility = 0.0
    soc = base.INITIAL_SOC

    for task in manifest:
        task_rows[str(task["task_id"])] = {
            "variant": variant,
            "seed": seed,
            "workload_scenario": workload,
            "contact_scenario": contact,
            "task_id": task["task_id"],
            "scene_id": task["scene_id"],
            "subtask_id": task["subtask_id"],
            "capture_time_h": task["capture_time_h"],
            "deadline_h": task["deadline_h"],
            "first_ground_receive_time_h": "",
            "target_ground_receive_time_h": "",
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

    def create_staged_request(task: dict[str, Any], product: str, orbit: int) -> None:
        task_id = str(task["task_id"])
        key = (task_id, product)
        if key in staged_requests or product in delivered[task_id]:
            return
        stats["staged_request_count"] += 1
        feasible = _future_contact_feasible(product=product, task=task, current_orbit=orbit, contacts=contact_by_orbit)
        if future_gate and not feasible:
            stats["staged_request_blocked_count"] += 1
            stats["future_contact_blocked_count"] += 1
            request_trace.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "workload_scenario": workload,
                    "contact_scenario": contact,
                    "task_id": task_id,
                    "product_type": product,
                    "orbit_index": orbit,
                    "action": "blocked_no_future_contact",
                    "future_contact_feasible": 0,
                    "guarantee_debt": 0.0,
                }
            )
            return
        delivered_before = set(delivered[task_id])
        delta = fair.delta_information_utility(product, delivered_before, str(task["product_requirement"]))
        residual_utility = _to_float(task["V_eff"]) * delta
        if residual_utility <= 0.0:
            stats["staged_request_blocked_count"] += 1
            return
        earliest_orbit = orbit + (2 if ground_followup else 1)
        staged_requests[key] = {
            "task_id": task_id,
            "product": product,
            "created_orbit": orbit,
            "earliest_orbit": earliest_orbit,
            "payload": base._product_payload(task, product),
            "residual_utility": residual_utility,
            "guarantee_debt": 0.0,
            "age": 0,
            "source": "ground_followup" if ground_followup else "onboard_staged",
        }
        stats["staged_request_enqueued_count"] += 1
        request_trace.append(
            {
                "variant": variant,
                "seed": seed,
                "workload_scenario": workload,
                "contact_scenario": contact,
                "task_id": task_id,
                "product_type": product,
                "orbit_index": orbit,
                "action": "enqueued",
                "future_contact_feasible": int(feasible),
                "guarantee_debt": 0.0,
            }
        )

    def deliver_product(cand: dict[str, Any], orbit: int, source: str, system_energy_share: float) -> None:
        nonlocal net_system_utility, h_net_utility, cumulative_direct_energy
        task = by_task[str(cand["task_id"])]
        task_id = str(task["task_id"])
        product = str(cand["product"])
        delivered_before = set(delivered[task_id])
        delivered[task_id].add(product)
        highest = _highest_stage(delivered[task_id])
        quality = _to_float(cand["quality"]) * base._stage_sufficiency(str(task["product_requirement"]), highest)
        delta = fair.delta_information_utility(product, delivered_before, str(task["product_requirement"]))
        gross = _to_float(task["V_eff"]) * delta * quality
        net = gross - _to_float(cand["account_cost"])
        net_system_utility += net
        if _to_int(task["is_high_value"]):
            h_net_utility += net
        cumulative_direct_energy += _to_float(cand["energy"])
        receive_time = (orbit + 1) * base.ORBIT_HOURS
        before_deadline = int(receive_time <= _to_float(task["deadline_h"]) + 1.0e-12)
        events.append(
            {
                "variant": variant,
                "seed": seed,
                "workload_scenario": workload,
                "contact_scenario": contact,
                "task_id": task_id,
                "scene_id": task["scene_id"],
                "delivered_stage": STAGE_BY_PRODUCT[product],
                "product_type": product,
                "source": source,
                "model_choice": cand["model"],
                "product_size_MB": cand["payload"],
                "ground_receive_time_h": receive_time,
                "before_deadline_flag": before_deadline,
                "delivered_quality": quality,
                "incremental_information_utility": delta,
            }
        )
        row = task_rows[task_id]
        row["highest_ground_received_stage"] = highest
        row["ground_received_quality"] = max(_to_float(row["ground_received_quality"]), quality)
        row["task_energy_direct_wh"] = _to_float(row["task_energy_direct_wh"]) + _to_float(cand["energy"])
        row["task_energy_system_wh"] = _to_float(row["task_energy_system_wh"]) + system_energy_share
        if base._stage_rank(highest) >= base._stage_rank(str(task["min_required_stage"])) and not _to_int(row["ground_received_flag"]):
            row["ground_received_flag"] = 1
            row["first_ground_receive_time_h"] = receive_time
            row["ground_received_before_deadline_flag"] = before_deadline
            row["capture_to_ground_latency_h"] = receive_time - _to_float(task["capture_time_h"])
        if base._stage_rank(highest) >= base._stage_rank(str(task["target_stage"])):
            row["target_stage_ground_received_flag"] = before_deadline
            if row["target_ground_receive_time_h"] == "":
                row["target_ground_receive_time_h"] = receive_time
        if source.startswith("staged"):
            stats["staged_request_delivered_count"] += 1
            stats["residual_payload_delivered_count"] += 1
        if _to_int(task["is_high_value"]) and staged_enabled and not one_shot:
            next_product = _next_progressive_product(task, delivered[task_id])
            if next_product is not None:
                create_staged_request(task, next_product, orbit)

    for orbit in range(base.NUM_ORBITS):
        time_h = orbit * base.ORBIT_HOURS
        resource = resource_by_orbit[orbit]
        wave = str(resource["wave_name"])
        for task in arrivals.get(orbit, []):
            active.add(str(task["task_id"]))
        for task_id in list(active):
            if time_h > _to_float(by_task[task_id]["deadline_h"]):
                active.discard(task_id)

        for key, request in list(staged_requests.items()):
            task = by_task[str(request["task_id"])]
            request["age"] = max(0, orbit - _to_int(request["created_orbit"]))
            if time_h > _to_float(task["deadline_h"]):
                stats["staged_request_expired_count"] += 1
                request_trace.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "workload_scenario": workload,
                        "contact_scenario": contact,
                        "task_id": request["task_id"],
                        "product_type": request["product"],
                        "orbit_index": orbit,
                        "action": "expired",
                        "future_contact_feasible": 0,
                        "guarantee_debt": request["guarantee_debt"],
                    }
                )
                del staged_requests[key]

        ready_tasks = [by_task[task_id] for task_id in sorted(active) if time_h <= _to_float(by_task[task_id]["deadline_h"])]
        selected: list[dict[str, Any]] = []
        remain = _to_float(contact_by_orbit[orbit]["capacity_MB"]) if orbit in contact_by_orbit else 0.0
        selected_energy = 0.0
        staged_candidates: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
        ordinary_candidates: list[dict[str, Any]] = []
        backlog_mb = sum(_to_float(req["payload"]) for req in staged_requests.values()) + sum(
            base._product_payload(task, _next_progressive_product(task, delivered[str(task["task_id"])]) or "metadata")
            for task in ready_tasks
            if _next_progressive_product(task, delivered[str(task["task_id"])])
        )

        if orbit in contact_by_orbit:
            for request in staged_requests.values():
                if orbit < _to_int(request["earliest_orbit"]):
                    continue
                task = by_task[str(request["task_id"])]
                if future_gate and not _future_contact_feasible(
                    product=str(request["product"]), task=task, current_orbit=orbit, contacts=contact_by_orbit, include_current=True
                ):
                    continue
                base_score, final_score = _request_priority(request, task, orbit, guarantee_enabled)
                cand = _candidate_score(
                    task=task,
                    delivered=delivered[str(task["task_id"])],
                    product=str(request["product"]),
                    model="Small" if request["product"] != "roi_context" else "Medium",
                    orbit=orbit,
                    soc=soc,
                    capacity_mb=remain,
                    queue_mb=backlog_mb,
                    high_value_bonus=1.0,
                    guarantee_debt=_to_float(request.get("guarantee_debt")),
                )
                cand["score"] = final_score - 0.045 * _to_float(cand["energy"])
                staged_candidates.append((base_score, final_score, request, cand))

            if staged_candidates and guarantee_enabled:
                base_order = [id(item[2]) for item in sorted(staged_candidates, key=lambda item: item[0], reverse=True)]
                final_order = [id(item[2]) for item in sorted(staged_candidates, key=lambda item: item[1], reverse=True)]
                for item in staged_candidates:
                    if _to_float(item[2].get("guarantee_debt")) > 0.0:
                        stats["guarantee_assisted_candidate_count"] += 1
                        if final_order.index(id(item[2])) < base_order.index(id(item[2])):
                            stats["guarantee_priority_improved_rank_count"] += 1

            for _base_score, _final_score, request, cand in sorted(staged_candidates, key=lambda item: item[1], reverse=True):
                if _to_float(cand["payload"]) > remain + 1.0e-12:
                    if guarantee_enabled:
                        request["guarantee_debt"] = min(6.0, _to_float(request.get("guarantee_debt")) + 1.0)
                    continue
                projected_energy = selected_energy + _to_float(cand["energy"])
                projected_soc = soc + (
                    _to_float(resource["solar_energy_wh"])
                    - _to_float(resource["housekeeping_energy_wh"])
                    - price._payload_idle_wh(True)
                    - projected_energy
                    - base.TX_FIXED_WH_PER_ACTIVE_ORBIT
                ) / base.BATTERY_CAPACITY_WH
                if projected_soc < base.HARD_MIN_SOC + 0.005:
                    if guarantee_enabled:
                        request["guarantee_debt"] = min(6.0, _to_float(request.get("guarantee_debt")) + 1.0)
                    continue
                selected.append(dict(cand, source="staged_residual"))
                remain -= _to_float(cand["payload"])
                selected_energy = projected_energy
                staged_requests.pop((str(request["task_id"]), str(request["product"])), None)
                request_trace.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "workload_scenario": workload,
                        "contact_scenario": contact,
                        "task_id": request["task_id"],
                        "product_type": request["product"],
                        "orbit_index": orbit,
                        "action": "delivered",
                        "future_contact_feasible": 1,
                        "guarantee_debt": request["guarantee_debt"],
                    }
                )

            for task in ready_tasks:
                task_id = str(task["task_id"])
                if one_shot:
                    product = _one_shot_product(task, delivered[task_id], generated_once)
                elif staged_enabled and _to_int(task["is_high_value"]) and delivered[task_id]:
                    product = None
                else:
                    product = _next_progressive_product(task, delivered[task_id])
                if product is None:
                    continue
                if one_shot:
                    generated_once.add(task_id)
                for model in price.MODEL_CHOICES:
                    ordinary_candidates.append(
                        _candidate_score(
                            task=task,
                            delivered=delivered[task_id],
                            product=product,
                            model=model,
                            orbit=orbit,
                            soc=soc,
                            capacity_mb=remain,
                            queue_mb=backlog_mb,
                            high_value_bonus=high_value_bonus,
                        )
                    )

            for cand in sorted(ordinary_candidates, key=lambda row: _to_float(row["score"]), reverse=True):
                if any(row["task_id"] == cand["task_id"] for row in selected):
                    continue
                if _to_float(cand["payload"]) > remain + 1.0e-12:
                    continue
                if cand["deadline_bucket"] == "expired":
                    continue
                projected_energy = selected_energy + _to_float(cand["energy"])
                projected_soc = soc + (
                    _to_float(resource["solar_energy_wh"])
                    - _to_float(resource["housekeeping_energy_wh"])
                    - price._payload_idle_wh(True)
                    - projected_energy
                    - base.TX_FIXED_WH_PER_ACTIVE_ORBIT
                ) / base.BATTERY_CAPACITY_WH
                if projected_soc < base.HARD_MIN_SOC + 0.005:
                    continue
                if _to_float(cand["score"]) <= 0.0 and selected:
                    continue
                selected.append(dict(cand, source="ordinary_queue"))
                remain -= _to_float(cand["payload"])
                selected_energy = projected_energy
                if len(selected) >= 8:
                    break

        orbit_compute = sum(_to_float(cand["compute"]) for cand in selected)
        orbit_comm = sum(_to_float(cand["comm"]) for cand in selected)
        payload_on = bool(selected)
        payload_idle = price._payload_idle_wh(payload_on)
        tx_fixed = base.TX_FIXED_WH_PER_ACTIVE_ORBIT if selected else 0.0
        net_energy = _to_float(resource["solar_energy_wh"]) - _to_float(resource["housekeeping_energy_wh"]) - payload_idle - orbit_compute - orbit_comm - tx_fixed
        before_soc = soc
        soc = max(base.HARD_MIN_SOC, min(1.0, soc + net_energy / base.BATTERY_CAPACITY_WH))
        system_energy = orbit_compute + orbit_comm + payload_idle
        cumulative_system_energy += system_energy
        share = _safe_div(system_energy, len(selected))
        for cand in selected:
            deliver_product(cand, orbit, str(cand["source"]), share)

        trajectory.append(
            {
                "variant": variant,
                "seed": seed,
                "workload_scenario": workload,
                "contact_scenario": contact,
                "orbit_index": orbit,
                "time_h": orbit * base.ORBIT_HOURS,
                "SOC": soc,
                "soc_delta": soc - before_soc,
                "thermal_pressure": _thermal_pressure(wave, orbit_compute, backlog_mb, payload_on),
                "backlog_count": len(staged_requests) + len(ready_tasks),
                "backlog_MB": backlog_mb,
                "metadata_count": sum(1 for cand in selected if cand["product"] == "metadata"),
                "roi_count": sum(1 for cand in selected if cand["product"] == "roi"),
                "roi_context_count": sum(1 for cand in selected if cand["product"] == "roi_context"),
            }
        )

    task_delivery = list(task_rows.values())
    summary = _build_summary(
        variant=variant,
        seed=seed,
        workload=workload,
        contact=contact,
        hashes=hashes,
        task_delivery=task_delivery,
        trajectory=trajectory,
        stats=stats,
        cumulative_direct_energy=cumulative_direct_energy,
        cumulative_system_energy=cumulative_system_energy,
        net_system_utility=net_system_utility,
        h_net_utility=h_net_utility,
    )
    return {
        "summary": summary,
        "events": events,
        "task_delivery": task_delivery,
        "trajectory": trajectory,
        "request_trace": request_trace,
    }


def _build_summary(
    *,
    variant: str,
    seed: int,
    workload: str,
    contact: str,
    hashes: dict[str, str],
    task_delivery: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    stats: Counter,
    cumulative_direct_energy: float,
    cumulative_system_energy: float,
    net_system_utility: float,
    h_net_utility: float,
) -> dict[str, Any]:
    success = [row for row in task_delivery if _to_int(row["ground_received_before_deadline_flag"])]
    target = [row for row in task_delivery if _to_int(row["target_stage_ground_received_flag"])]
    high_tasks = [row for row in task_delivery if _to_int(row["is_high_value"])]
    high_success = [row for row in high_tasks if _to_int(row["ground_received_before_deadline_flag"])]
    high_target = [row for row in high_tasks if _to_int(row["target_stage_ground_received_flag"])]
    expired = [row for row in task_delivery if not _to_int(row["ground_received_before_deadline_flag"])]
    high_expired = [row for row in expired if _to_int(row["is_high_value"])]
    metadata_success = [row for row in success if row["highest_ground_received_stage"] == "metadata"]
    roi_success = [row for row in success if row["highest_ground_received_stage"] == "ROI"]
    roi_context_success = [row for row in success if row["highest_ground_received_stage"] == "ROI+Ctx"]
    high_metadata = [row for row in high_success if row["highest_ground_received_stage"] == "metadata"]
    high_roi_context = [row for row in high_success if row["highest_ground_received_stage"] in {"ROI", "ROI+Ctx"}]
    soc_values = [_to_float(row["SOC"]) for row in trajectory]
    thermal_values = [_to_float(row["thermal_pressure"]) for row in trajectory]
    scenario = f"{workload}__{contact}"
    return {
        "variant": variant,
        "seed": seed,
        "workload_scenario": workload,
        "contact_scenario": contact,
        "scenario": scenario,
        **hashes,
        "ground_success_ratio": _safe_div(len(success), len(task_delivery)),
        "target_stage_success_ratio": _safe_div(len(target), len(task_delivery)),
        "high_value_ground_success_ratio": _safe_div(len(high_success), len(high_tasks)),
        "high_value_target_success_ratio": _safe_div(len(high_target), len(high_tasks)),
        "ground_received_task_count": len(success),
        "high_value_ground_received_count": len(high_success),
        "avg_ground_received_quality": statistics.mean([_to_float(row["ground_received_quality"]) for row in success]) if success else 0.0,
        "avg_high_value_ground_received_quality": statistics.mean([_to_float(row["ground_received_quality"]) for row in high_success]) if high_success else 0.0,
        "quality_weighted_success_value": sum(_to_float(row["V_eff"]) * _to_float(row["ground_received_quality"]) for row in success),
        "quality_weighted_high_value_success_value": sum(_to_float(row["V_eff"]) * _to_float(row["ground_received_quality"]) for row in high_success),
        "net_system_utility": net_system_utility,
        "H_net_utility": h_net_utility,
        "expired_task_count": len(expired),
        "expired_high_value_task_count": len(high_expired),
        "avg_capture_to_ground_latency_h": statistics.mean([_to_float(row["capture_to_ground_latency_h"]) for row in success]) if success else 0.0,
        "avg_high_value_capture_to_ground_latency_h": statistics.mean([_to_float(row["capture_to_ground_latency_h"]) for row in high_success]) if high_success else 0.0,
        "staged_request_count": stats["staged_request_count"],
        "staged_request_enqueued_count": stats["staged_request_enqueued_count"],
        "staged_request_delivered_count": stats["staged_request_delivered_count"],
        "staged_request_expired_count": stats["staged_request_expired_count"],
        "staged_request_blocked_count": stats["staged_request_blocked_count"],
        "residual_payload_delivered_count": stats["residual_payload_delivered_count"],
        "metadata_only_success_count": len(metadata_success),
        "roi_success_count": len(roi_success),
        "roi_context_success_count": len(roi_context_success),
        "high_value_metadata_only_count": len(high_metadata),
        "high_value_roi_or_context_count": len(high_roi_context),
        "target_stage_delivered_count": len(target),
        "future_contact_blocked_count": stats["future_contact_blocked_count"],
        "guarantee_assisted_candidate_count": stats["guarantee_assisted_candidate_count"],
        "guarantee_priority_improved_rank_count": stats["guarantee_priority_improved_rank_count"],
        "min_SOC": min(soc_values) if soc_values else base.INITIAL_SOC,
        "final_SOC": soc_values[-1] if soc_values else base.INITIAL_SOC,
        "hard_safety_violation_count": sum(value < base.HARD_MIN_SOC - 1.0e-12 for value in soc_values),
        "max_thermal_pressure": max(thermal_values) if thermal_values else 0.0,
        "avg_thermal_pressure": statistics.mean(thermal_values) if thermal_values else 0.0,
        "thermal_violation_count": sum(value > price.THERMAL_HARD_LIMIT for value in thermal_values),
        "queue_MB_peak": max([_to_float(row["backlog_MB"]) for row in trajectory] or [0.0]),
        "final_backlog_count": _to_int(trajectory[-1]["backlog_count"]) if trajectory else 0,
        "max_backlog_count": max([_to_int(row["backlog_count"]) for row in trajectory] or [0]),
    }


def _build_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (variant, workload, contact), group in sorted(_group(rows, ["variant", "workload_scenario", "contact_scenario"]).items()):
        for metric in METRICS:
            values = [_to_float(row[metric]) for row in group]
            ranks = []
            for seed in sorted({row["seed"] for row in group}):
                seed_rows = [
                    row
                    for row in rows
                    if row["workload_scenario"] == workload and row["contact_scenario"] == contact and _to_int(row["seed"]) == _to_int(seed)
                ]
                ordered = sorted(
                    [(row["variant"], _to_float(row[metric])) for row in seed_rows],
                    key=lambda item: item[1],
                    reverse=metric not in LOWER_IS_BETTER,
                )
                ranks.extend(idx for idx, (name, _value) in enumerate(ordered, start=1) if name == variant)
            out.append(
                {
                    "variant": variant,
                    "workload_scenario": workload,
                    "contact_scenario": contact,
                    "scenario": f"{workload}__{contact}",
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


def _rank_map(rows: list[dict[str, Any]], workload: str, contact: str, metric: str) -> dict[str, int]:
    source = [row for row in rows if row["workload_scenario"] == workload and row["contact_scenario"] == contact]
    means = []
    for (variant,), group in _group(source, ["variant"]).items():
        means.append((variant, statistics.mean(_to_float(row[metric]) for row in group)))
    ordered = sorted(means, key=lambda item: item[1], reverse=metric not in LOWER_IS_BETTER)
    return {variant: idx for idx, (variant, _value) in enumerate(ordered, start=1)}


def _build_effect_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (workload, contact), _group_rows in sorted(_group(rows, ["workload_scenario", "contact_scenario"]).items()):
        scenario = f"{workload}__{contact}"
        for ablation in [variant for variant in VARIANTS if variant != "full_staged_delivery"]:
            for metric in EFFECT_METRICS:
                full_vals = [
                    _to_float(row[metric])
                    for row in rows
                    if row["workload_scenario"] == workload and row["contact_scenario"] == contact and row["variant"] == "full_staged_delivery"
                ]
                abl_vals = [
                    _to_float(row[metric])
                    for row in rows
                    if row["workload_scenario"] == workload and row["contact_scenario"] == contact and row["variant"] == ablation
                ]
                full_mean = statistics.mean(full_vals) if full_vals else 0.0
                abl_mean = statistics.mean(abl_vals) if abl_vals else 0.0
                gain = (abl_mean - full_mean) if metric in LOWER_IS_BETTER else (full_mean - abl_mean)
                winner = "tie" if abs(gain) <= 1.0e-12 else ("full_staged_delivery" if gain > 0 else ablation)
                out.append(
                    {
                        "scenario": scenario,
                        "workload_scenario": workload,
                        "contact_scenario": contact,
                        "ablation_variant": ablation,
                        "metric": metric,
                        "full_mean": full_mean,
                        "ablation_mean": abl_mean,
                        "absolute_gain": gain,
                        "relative_gain_percent": _safe_div(gain, abs(abl_mean)) * 100.0,
                        "winner": winner,
                    }
                )
    return out


def _build_funnel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (variant, seed, workload, contact), group in sorted(_group(rows, ["variant", "seed", "workload_scenario", "contact_scenario"]).items()):
        high_tasks = [row for row in group if _to_int(row["is_high_value"])]
        high_success = [row for row in high_tasks if _to_int(row["ground_received_before_deadline_flag"])]
        out.append(
            {
                "variant": variant,
                "seed": seed,
                "workload_scenario": workload,
                "contact_scenario": contact,
                "scenario": f"{workload}__{contact}",
                "high_value_task_count": len(high_tasks),
                "metadata_delivered_count": sum(row["highest_ground_received_stage"] in {"metadata", "ROI", "ROI+Ctx"} for row in high_tasks),
                "roi_delivered_count": sum(row["highest_ground_received_stage"] in {"ROI", "ROI+Ctx"} for row in high_tasks),
                "roi_context_delivered_count": sum(row["highest_ground_received_stage"] == "ROI+Ctx" for row in high_tasks),
                "target_stage_delivered_count": sum(_to_int(row["target_stage_ground_received_flag"]) for row in high_tasks),
                "high_value_ground_success_count": len(high_success),
                "high_value_expired_count": len(high_tasks) - len(high_success),
                "metadata_only_success_count": sum(row["highest_ground_received_stage"] == "metadata" for row in high_success),
                "roi_or_context_success_count": sum(row["highest_ground_received_stage"] in {"ROI", "ROI+Ctx"} for row in high_success),
            }
        )
    return out


def _build_first_place(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for metric in METRICS:
        means = []
        for (variant,), group in _group(rows, ["variant"]).items():
            means.append((variant, statistics.mean(_to_float(row[metric]) for row in group)))
        ordered = sorted(means, key=lambda item: item[1], reverse=metric not in LOWER_IS_BETTER)
        rank = {variant: idx for idx, (variant, _value) in enumerate(ordered, start=1)}
        best_variant, best_value = ordered[0]
        out.append(
            {
                "metric": metric,
                "best_variant": best_variant,
                "best_value": best_value,
                "full_staged_delivery_rank": rank.get("full_staged_delivery", ""),
                "w/o_staged_delivery_rank": rank.get("w/o_staged_delivery", ""),
                "ordinary_priority_only_rank": rank.get("ordinary_priority_only", ""),
                "w/o_future_contact_feasibility_rank": rank.get("w/o_future_contact_feasibility", ""),
                "w/o_guarantee_debt_rank": rank.get("w/o_guarantee_debt", ""),
                "ground_approved_followup_rank": rank.get("ground_approved_followup", ""),
                "no_progressive_payload_rank": rank.get("no_progressive_payload", ""),
            }
        )
    return out


def _mean(rows: list[dict[str, Any]], variant: str, metric: str) -> float:
    values = [_to_float(row[metric]) for row in rows if row["variant"] == variant]
    return statistics.mean(values) if values else 0.0


def _gain(rows: list[dict[str, Any]], variant: str, metric: str) -> float:
    full = _mean(rows, "full_staged_delivery", metric)
    other = _mean(rows, variant, metric)
    return (other - full) if metric in LOWER_IS_BETTER else (full - other)


def _plot_high_value(rows: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = [
        "high_value_ground_success_ratio",
        "high_value_target_success_ratio",
        "quality_weighted_high_value_success_value",
        "expired_high_value_task_count",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        values = [_mean(rows, variant, metric) for variant in VARIANTS]
        axis.bar(range(len(VARIANTS)), values, color="#2563eb")
        axis.set_title(metric)
        axis.set_xticks(range(len(VARIANTS)))
        axis.set_xticklabels(VARIANTS, rotation=60, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "staged_ablation_high_value_panel.png", dpi=180)
    plt.close(fig)


def _plot_funnel(funnel: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = [
        "high_value_task_count",
        "metadata_delivered_count",
        "roi_delivered_count",
        "roi_context_delivered_count",
        "target_stage_delivered_count",
        "high_value_expired_count",
    ]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = range(len(metrics))
    width = 0.11
    for idx, variant in enumerate(VARIANTS):
        values = []
        source = [row for row in funnel if row["variant"] == variant]
        for metric in metrics:
            values.append(statistics.mean([_to_float(row[metric]) for row in source]) if source else 0.0)
        ax.bar([pos + idx * width for pos in x], values, width=width, label=variant)
    ax.set_xticks([pos + width * 3 for pos in x])
    ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "staged_delivery_funnel_panel.png", dpi=180)
    plt.close(fig)


def _plot_followup(rows: list[dict[str, Any]], output_dir: Path) -> None:
    variants = ["full_staged_delivery", "ground_approved_followup", "ordinary_priority_only"]
    metrics = ["high_value_target_success_ratio", "avg_high_value_capture_to_ground_latency_h", "expired_high_value_task_count"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        axis.bar(range(len(variants)), [_mean(rows, variant, metric) for variant in variants], color="#0f766e")
        axis.set_title(metric)
        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "staged_vs_ground_followup_panel.png", dpi=180)
    plt.close(fig)


def _plot_resource(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for variant in VARIANTS:
        source = [row for row in rows if row["variant"] == variant]
        ax.scatter(
            [_to_float(row["min_SOC"]) for row in source],
            [_to_float(row["quality_weighted_high_value_success_value"]) for row in source],
            s=[35 + 160 * _to_float(row["high_value_ground_success_ratio"]) for row in source],
            alpha=0.62,
            label=variant,
        )
    ax.set_xlabel("min_SOC")
    ax.set_ylabel("quality_weighted_high_value_success_value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "staged_resource_tradeoff_panel.png", dpi=180)
    plt.close(fig)


def build_figures(rows: list[dict[str, Any]], funnel: list[dict[str, Any]], output_dir: Path) -> None:
    _plot_high_value(rows, output_dir)
    _plot_funnel(funnel, output_dir)
    _plot_followup(rows, output_dir)
    _plot_resource(rows, output_dir)


def build_quick_report(rows: list[dict[str, Any]], first_place: list[dict[str, Any]]) -> str:
    full_first = [row["metric"] for row in first_place if row["best_variant"] == "full_staged_delivery"]
    not_first = [f"{row['metric']}->{row['best_variant']}" for row in first_place if row["best_variant"] != "full_staged_delivery"]
    hard_violations = sum(_to_int(row["hard_safety_violation_count"]) for row in rows if row["variant"] == "full_staged_delivery")
    thermal_gain = _gain(rows, "w/o_staged_delivery", "max_thermal_pressure")
    soc_gain = _gain(rows, "w/o_staged_delivery", "min_SOC")
    metadata_gap = _gain(rows, "w/o_staged_delivery", "high_value_metadata_only_count")
    roi_gap = _gain(rows, "w/o_staged_delivery", "high_value_roi_or_context_count")
    return "\n".join(
        [
            "# High-value staged delivery ablation quick report",
            "",
            "- This is a standalone realistic workload/contact benchmark runner. It only writes `outputs/_staged_delivery_ablation_v1/`.",
            "- All variants share the same task manifest, contact schedule, resource trace, and value-prior hash for each seed/workload/contact case.",
            "",
            "## Required answers",
            f"1. high-value ground success gain vs w/o_staged_delivery: {_fmt(_gain(rows, 'w/o_staged_delivery', 'high_value_ground_success_ratio'))}.",
            f"2. high-value target-stage success gain vs w/o_staged_delivery: {_fmt(_gain(rows, 'w/o_staged_delivery', 'high_value_target_success_ratio'))}.",
            f"3. quality-weighted high-value success gain vs w/o_staged_delivery: {_fmt(_gain(rows, 'w/o_staged_delivery', 'quality_weighted_high_value_success_value'))}.",
            f"4. expired high-value reduction vs w/o_staged_delivery: {_fmt(_gain(rows, 'w/o_staged_delivery', 'expired_high_value_task_count'))}.",
            f"5. Ordinary high-value bonus sufficiency: {'not sufficient' if _gain(rows, 'w/o_staged_delivery', 'quality_weighted_high_value_success_value') > 0 else 'not proven by this run'}; compare staged_ablation_effect_summary.csv for paired cases.",
            f"6. Future contact feasibility effect: full vs w/o_future_contact_feasibility expired staged gain={_fmt(_gain(rows, 'w/o_future_contact_feasibility', 'staged_request_expired_count'))}, queue peak gain={_fmt(_gain(rows, 'w/o_future_contact_feasibility', 'queue_MB_peak'))}.",
            f"7. Guarantee debt effect: full vs w/o_guarantee_debt high-value target gain={_fmt(_gain(rows, 'w/o_guarantee_debt', 'high_value_target_success_ratio'))}, guarantee-assisted candidates mean={_fmt(_mean(rows, 'full_staged_delivery', 'guarantee_assisted_candidate_count'))}.",
            f"8. Ground follow-up latency/expiration: latency gain over ground_approved_followup={_fmt(_gain(rows, 'ground_approved_followup', 'avg_high_value_capture_to_ground_latency_h'))}, expired gain={_fmt(_gain(rows, 'ground_approved_followup', 'expired_high_value_task_count'))}.",
            f"9. Metadata stacking check: high-value metadata-only gain={_fmt(metadata_gap)}, high-value ROI/ROI+Ctx gain={_fmt(roi_gap)}. Positive ROI/ROI+Ctx gain means the result is not only metadata stacking.",
            f"10. Resource/safety: hard_safety_violation_count total for full={hard_violations}; min_SOC gain vs w/o staged={_fmt(soc_gain)}; max thermal lower-is-better gain={_fmt(thermal_gain)}.",
            "11. Magazine framework recommendation: keep Module 4 as a dual channel if the high-value target/quality gains above are positive in the scenarios used for the paper.",
            "12. Innovation recommendation: staged delivery is supported as a core mechanism when the effect summary shows full_staged_delivery beating ordinary priority and ground-follow-up on high-value target or quality-weighted metrics.",
            "",
            f"Full_staged_delivery first-place metrics: {', '.join(full_first) if full_first else 'none'}.",
            f"Metrics where full_staged_delivery is not first: {', '.join(not_first[:25]) if not_first else 'none'}{' ...' if len(not_first) > 25 else ''}.",
        ]
    )


def run_experiment(output_dir: Path, seeds: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    all_trajectory: list[dict[str, Any]] = []
    all_requests: list[dict[str, Any]] = []

    for seed in seeds:
        for workload in realistic.WORKLOAD_SCENARIOS:
            for contact in realistic.CONTACT_SCENARIOS:
                contacts = realistic.build_contact_schedule(seed, workload, contact)
                resources = realistic.build_resource_trace(seed, workload, contact)
                manifest, _audit = realistic.build_task_manifest(seed, workload, contact, contacts)
                hashes = _hashes(manifest, contacts, resources)
                for variant in VARIANTS:
                    result = run_variant(
                        variant=variant,
                        seed=seed,
                        workload=workload,
                        contact=contact,
                        manifest=manifest,
                        contacts=contacts,
                        resources=resources,
                        hashes=hashes,
                    )
                    all_summary.append(result["summary"])
                    all_events.extend(result["events"])
                    all_tasks.extend(result["task_delivery"])
                    all_trajectory.extend(result["trajectory"])
                    all_requests.extend(result["request_trace"])

    ranking = _build_ranking(all_summary)
    effect = _build_effect_summary(all_summary)
    funnel = _build_funnel(all_tasks)
    first_place = _build_first_place(all_summary)
    _write_csv(output_dir / "staged_ablation_summary_by_seed.csv", all_summary, SUMMARY_COLUMNS)
    _write_csv(output_dir / "staged_ablation_multi_seed_ranking.csv", ranking)
    _write_csv(output_dir / "staged_ablation_effect_summary.csv", effect)
    _write_csv(output_dir / "staged_delivery_funnel.csv", funnel)
    _write_csv(output_dir / "staged_delivery_first_place_check.csv", first_place)
    _write_csv(output_dir / "staged_delivery_request_trace.csv", all_requests)
    _write_csv(output_dir / "staged_ground_events.csv", all_events)
    _write_csv(output_dir / "staged_task_delivery.csv", all_tasks)
    _write_csv(output_dir / "staged_trajectory.csv", all_trajectory)
    build_figures(all_summary, funnel, output_dir)
    _write_text(output_dir / "quick_report.md", build_quick_report(all_summary, first_place))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    run_experiment(args.output_dir, seeds)
    print(f"Wrote staged delivery ablation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
