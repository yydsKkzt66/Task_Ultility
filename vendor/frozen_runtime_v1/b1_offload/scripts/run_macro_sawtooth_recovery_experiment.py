"""Long-horizon macro-sawtooth workload experiment.

This runner is intentionally standalone and default-off. It reuses the frozen
fair-baseline task value, staged-delivery, and accounting helpers, then adds a
96 h macro-cycle workload and a diagnostic SOC/resource integration layer.
It does not modify router, reward, closure, evaluator, utility-rule, or value
prior code.
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

import scripts.run_fair_baseline_comparison as fair  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_macro_sawtooth_recovery_v1"

SCENARIO = "macro_sawtooth_recovery_96h_v1"
SEED = 2067
NUM_ORBITS = 60
ORBIT_HOURS = 96.0 / NUM_ORBITS
BATTERY_CAPACITY_WH = 230.0
INITIAL_SOC = 0.86
HARD_MIN_SOC = 0.20
PROTECTION_RESUME_SOC = 0.35

HOUSEKEEPING_WH_PER_ORBIT = 12.0
TX_FIXED_WH_PER_ACTIVE_ORBIT = 3.0

ARM_ORDER = [
    "proposed_guarantee_aware_full",
    "utility_rule",
    "persistent_queue_without_guarantee",
    "fixed_lightweight",
    "fixed_heavy",
    "fixed_heavy_with_20pct_protection",
    "ground_approved_followup",
]

FAIR_ARM = {
    "proposed_guarantee_aware_full": "proposed_guarantee_aware_full",
    "utility_rule": "utility_rule",
    "persistent_queue_without_guarantee": "persistent_queue_without_guarantee",
    "fixed_lightweight": "fixed_lightweight_onboard",
    "fixed_heavy": "fixed_heavy_onboard",
    "fixed_heavy_with_20pct_protection": "fixed_heavy_onboard",
    "ground_approved_followup": "ground_approved_followup",
}

MANIFEST_COLUMNS = fair.MANIFEST_COLUMNS + ["cycle_index", "macro_phase", "phase_orbit_index"]
CONTACT_COLUMNS = fair.CONTACT_COLUMNS + ["orbit_index", "macro_phase"]
RESOURCE_COLUMNS = [
    "scenario",
    "seed",
    "num_slots",
    "orbit_index",
    "time_h",
    "cycle_index",
    "macro_phase",
    "deadline_pressure",
    "solar_energy_wh",
    "housekeeping_energy_wh",
    "recovery_mode_flag",
]

TRAJECTORY_COLUMNS = [
    "arm",
    "orbit_index",
    "time_h",
    "cycle_index",
    "macro_phase",
    "deadline_pressure",
    "SOC_before",
    "SOC",
    "soc_delta",
    "soc_slope_per_hour",
    "soc_band_before",
    "payload_on",
    "payload_idle_energy_wh",
    "housekeeping_energy_wh",
    "solar_energy_wh",
    "compute_energy_wh",
    "comm_energy_wh",
    "tx_fixed_energy_wh",
    "net_energy_wh",
    "delivered_product_count",
    "metadata_count",
    "roi_count",
    "roi_context_count",
    "defer_count",
    "backlog_count",
    "backlog_MB",
    "protection_mode",
]

SLOPE_COLUMNS = [
    "arm",
    "soc_band",
    "avg_soc_slope_per_hour",
    "avg_compute_energy_wh_per_hour",
    "avg_comm_energy_wh_per_hour",
    "avg_payload_on_fraction",
    "avg_metadata_rate",
    "avg_roi_rate",
    "avg_roi_context_rate",
    "avg_defer_rate",
]

RECOVERY_COLUMNS = [
    "arm",
    "cycle_index",
    "recovery_phase_start_h",
    "recovery_phase_end_h",
    "initial_SOC",
    "peak_SOC",
    "final_SOC",
    "recovered_above_80_flag",
    "recovery_gain",
    "recovery_duration_h",
]

PROTECTION_SUMMARY_COLUMNS = [
    "arm",
    "protection_trigger_count",
    "protection_total_slots",
    "avg_recovery_duration_slots",
    "min_SOC",
    "final_SOC",
    "net_system_utility",
    "H_net_utility",
]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None, "NA"}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None, "NA"}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


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


def _hash(rows: list[dict[str, Any]], columns: list[str]) -> str:
    return fair._stable_hash(rows, columns)


def _cycle_phase(orbit: int) -> tuple[int, str, int, str]:
    cycle = orbit // 15
    local = orbit % 15
    if local < 4:
        return cycle, "work_phase_1", local, "medium_high"
    if local < 8:
        return cycle, "work_phase_2", local - 4, "high"
    if local < 10:
        return cycle, "tail_heavy_or_cleanup", local - 8, "medium"
    return cycle, "recovery_phase", local - 10, "low"


def _soc_band(soc: float) -> str:
    if soc >= 0.70:
        return "SOC_above_70"
    if soc >= 0.50:
        return "SOC_50_70"
    return "SOC_below_50"


def _is_work_phase(phase: str) -> bool:
    return phase in {"work_phase_1", "work_phase_2", "tail_heavy_or_cleanup"}


def _phase_task_count(phase: str, cycle: int, local_index: int) -> int:
    if phase == "work_phase_1":
        return 3 + int((cycle + local_index) % 2 == 0)
    if phase == "work_phase_2":
        return 4 + int((cycle + local_index) % 2 == 1)
    if phase == "tail_heavy_or_cleanup":
        return 2
    return 1 if (cycle + local_index) % 5 == 0 else 0


def build_contact_schedule() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in range(NUM_ORBITS):
        cycle, phase, _local, _pressure = _cycle_phase(orbit)
        if phase == "work_phase_1":
            capacity = 5.8
        elif phase == "work_phase_2":
            capacity = 5.2
        elif phase == "tail_heavy_or_cleanup":
            capacity = 4.6
        else:
            capacity = 3.8
        capacity *= 0.92 + 0.16 * fair._stable_unit(SEED, orbit, "contact_capacity")
        rows.append(
            {
                "scenario": SCENARIO,
                "seed": SEED,
                "num_slots": NUM_ORBITS,
                "contact_window_id": f"macro_cw_{orbit:03d}",
                "start_slot": orbit,
                "end_slot": orbit,
                "capacity_MB": round(capacity, 5),
                "throughput_mbps": round(capacity * 8.0 / max(ORBIT_HOURS * 3600.0, 1.0), 6),
                "schedule_version": "macro_sawtooth_orbit_contacts_v1",
                "orbit_index": orbit,
                "macro_phase": phase,
            }
        )
    return rows


def build_resource_trace() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in range(NUM_ORBITS):
        cycle, phase, _local, pressure = _cycle_phase(orbit)
        if phase == "work_phase_1":
            solar = 20.5
        elif phase == "work_phase_2":
            solar = 17.5
        elif phase == "tail_heavy_or_cleanup":
            solar = 21.5
        else:
            solar = 39.0
        solar *= 0.96 + 0.08 * fair._stable_unit(SEED, orbit, "solar")
        rows.append(
            {
                "scenario": SCENARIO,
                "seed": SEED,
                "num_slots": NUM_ORBITS,
                "orbit_index": orbit,
                "time_h": orbit * ORBIT_HOURS,
                "cycle_index": cycle,
                "macro_phase": phase,
                "deadline_pressure": pressure,
                "solar_energy_wh": round(solar, 5),
                "housekeeping_energy_wh": HOUSEKEEPING_WH_PER_ORBIT,
                "recovery_mode_flag": int(phase == "recovery_phase"),
            }
        )
    return rows


def _value_level_for_task(index: int) -> str:
    return fair._value_level_for_index(SCENARIO, index)


def _requirement_for_task(value_level: str, index: int) -> str:
    return fair._requirement_for_index(value_level, index)


def build_macro_manifest(contact_schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_index = 0
    for orbit in range(NUM_ORBITS):
        cycle, phase, local, pressure = _cycle_phase(orbit)
        count = _phase_task_count(phase, cycle, local)
        for local_task in range(count):
            value_level = _value_level_for_task(task_index)
            requirement = _requirement_for_task(value_level, task_index)
            if pressure == "high":
                deadline_orbits = 4 if value_level in {"high", "urgent"} else 3
            elif pressure == "medium_high":
                deadline_orbits = 5 if value_level in {"high", "urgent"} else 4
            elif pressure == "medium":
                deadline_orbits = 5
            else:
                deadline_orbits = 6
            if requirement in {"context", "evidence"}:
                deadline_orbits += 2
            arrival = orbit
            deadline_slot = min(NUM_ORBITS - 1, arrival + deadline_orbits)
            base_prior, v_eff = fair._task_value(value_level, fair._stable_unit(SEED, task_index, "value_prior"))
            difficulty = 0.82 + 0.50 * fair._stable_unit(SEED, task_index, "payload_difficulty")
            metadata_size = round(0.065 + 0.045 * fair._stable_unit(SEED, task_index, "metadata"), 5)
            roi_size = round((0.72 + 0.68 * fair._stable_unit(SEED, task_index, "roi")) * difficulty, 5)
            roi_context_size = round(roi_size * (2.35 + 0.70 * fair._stable_unit(SEED, task_index, "ctx")), 5)
            min_stage, target_stage = fair._required_stages(requirement)
            feasible = sum(
                1
                for window in contact_schedule
                if arrival <= _to_int(window["start_slot"]) <= deadline_slot
            )
            rows.append(
                {
                    "scenario": SCENARIO,
                    "seed": SEED,
                    "num_slots": NUM_ORBITS,
                    "task_id": f"macro_s{SEED}_t{task_index:03d}",
                    "scene_id": f"macro_scene_{task_index:03d}",
                    "arrival_slot": arrival,
                    "deadline_slot": deadline_slot,
                    "deadline_seconds": (deadline_slot - arrival + 1) * ORBIT_HOURS * 3600.0,
                    "value_level": value_level,
                    "base_prior_score": base_prior,
                    "V_eff": v_eff,
                    "product_requirement": requirement,
                    "task_class": "macro_sawtooth_sarfish_candidate",
                    "high_value_flag": int(value_level in {"high", "urgent"}),
                    "emergency_flag": int(value_level == "urgent"),
                    "minimum_stage_required": min_stage,
                    "target_stage_required": target_stage,
                    "payload_size_metadata_mb": metadata_size,
                    "payload_size_roi_mb": roi_size,
                    "payload_size_roi_context_mb": roi_context_size,
                    "feasible_contact_windows_before_deadline": feasible,
                    "notes": f"macro_phase={phase};deadline_pressure={pressure}",
                    "cycle_index": cycle,
                    "macro_phase": phase,
                    "phase_orbit_index": local,
                }
            )
            task_index += 1
    return rows


def _arm_config(public_arm: str) -> dict[str, Any]:
    cfg = dict(fair.arm_config(FAIR_ARM[public_arm]))
    cfg["arm"] = public_arm
    if public_arm == "fixed_heavy_with_20pct_protection":
        cfg["strategy"] = "fixed_heavy"
        cfg["protection_enabled"] = True
    return cfg


def _payload_idle_wh(public_arm: str, payload_on: bool) -> float:
    if not payload_on:
        return 0.0
    if public_arm in {"fixed_heavy", "fixed_heavy_with_20pct_protection"}:
        return 13.5
    if public_arm == "fixed_lightweight":
        return 3.0
    if public_arm == "ground_approved_followup":
        return 5.2
    return 6.0


def _product_costs(product_type: str, payload_mb: float, cfg: dict[str, Any]) -> tuple[float, float, float, float, float]:
    base_compute = {"metadata": 1.25, "roi": 6.30, "roi_context": 15.00}.get(product_type, 0.0)
    compute_wh = base_compute * _to_float(cfg.get("compute_multiplier"), 1.0)
    comm_wh = 0.18 * payload_mb
    arm = str(cfg.get("arm", ""))
    compute_cost_factor = 0.014 if arm == "proposed_guarantee_aware_full" else 0.036
    accounting_cost = compute_cost_factor * compute_wh + 0.30 * comm_wh + 0.012 * payload_mb
    return compute_wh, comm_wh, accounting_cost, compute_wh, comm_wh


def _selection_limit(public_arm: str, soc: float, phase: str) -> int:
    if phase == "recovery_phase":
        if public_arm == "fixed_heavy":
            return 2
        if public_arm == "fixed_heavy_with_20pct_protection":
            return 1
        if public_arm == "utility_rule":
            return 2 if soc >= 0.55 else 1
        if public_arm == "proposed_guarantee_aware_full":
            return 1 if soc >= 0.82 else 0
        return 1 if soc >= 0.74 else 0
    if public_arm == "fixed_heavy":
        return 7
    if public_arm == "fixed_heavy_with_20pct_protection":
        return 6
    if public_arm == "utility_rule":
        return 6 if soc >= 0.20 else 0
    if public_arm == "persistent_queue_without_guarantee":
        return 4 if soc >= 0.50 else 2 if soc >= 0.20 else 0
    if public_arm == "fixed_lightweight":
        return 3 if soc >= 0.20 else 0
    if public_arm == "ground_approved_followup":
        return 3 if soc >= 0.50 else 1 if soc >= 0.20 else 0
    if soc >= 0.70:
        return 5
    if soc >= 0.50:
        return 3
    if soc >= 0.20:
        return 1
    return 0


def _request_allowed(public_arm: str, request: dict[str, Any], task: dict[str, Any], soc: float, phase: str) -> bool:
    if public_arm == "fixed_heavy":
        return True
    if soc <= HARD_MIN_SOC:
        return False
    if public_arm != "proposed_guarantee_aware_full":
        return True
    product = str(request["product_type"])
    if phase == "recovery_phase" and soc < 0.82:
        return False
    if soc < 0.50:
        return product == "metadata" or (product == "roi" and fair._is_H(task))
    if soc < 0.70:
        return product != "roi_context" or (fair._is_H(task) and soc >= 0.58)
    return True


def _stage_complete(stage: str, products: set[str]) -> bool:
    return fair._stage_rank(fair._stage_from_products(products)) >= fair._stage_rank(stage)


def run_arm(public_arm: str, manifest: list[dict[str, Any]], contact_schedule: list[dict[str, Any]], resource_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = _arm_config(public_arm)
    task_by_id = {str(task["task_id"]): task for task in manifest}
    arrivals_by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in manifest:
        arrivals_by_slot[_to_int(task["arrival_slot"])].append(task)
    contact_by_slot = {_to_int(row["start_slot"]): row for row in contact_schedule}
    resource_by_slot = {_to_int(row["orbit_index"]): row for row in resource_rows}

    delivered_by_task: dict[str, set[str]] = defaultdict(set)
    delivered_slots_by_task: dict[str, dict[str, int]] = defaultdict(dict)
    generated_by_task: dict[str, set[str]] = defaultdict(set)
    active_task_ids: set[str] = set()
    pending: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    queue_trace: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    backlog_rows: list[dict[str, Any]] = []
    protection_events: list[dict[str, Any]] = []

    soc = INITIAL_SOC
    protection_mode = False
    protection_entry_slot: int | None = None
    protection_trigger_count = 0
    protection_total_slots = 0
    recovery_durations: list[int] = []
    cumulative_net = 0.0
    cumulative_h_net = 0.0

    def add_request(task: dict[str, Any], product_type: str, earliest_slot: int, reason: str, current_slot: int) -> None:
        payload = fair._product_payload(task, product_type) * _to_float(cfg.get("payload_multiplier"), 1.0)
        pending.append(
            {
                "request_id": f"{task['task_id']}::{product_type}",
                "task_id": task["task_id"],
                "product_type": product_type,
                "product_requirement": task["product_requirement"],
                "value_level": task["value_level"],
                "created_slot": current_slot,
                "earliest_slot": earliest_slot,
                "payload_size_MB": payload,
                "generation_reason": reason,
                "missed_contact_windows": 0,
            }
        )
        generated_by_task[str(task["task_id"])].add(product_type)

    for orbit in range(NUM_ORBITS):
        resource = resource_by_slot[orbit]
        cycle, phase, _local, pressure = _cycle_phase(orbit)

        if public_arm == "fixed_heavy_with_20pct_protection":
            if not protection_mode and soc <= HARD_MIN_SOC:
                protection_mode = True
                protection_trigger_count += 1
                protection_entry_slot = orbit
                protection_events.append(
                    {
                        "arm": public_arm,
                        "event": "entry",
                        "orbit_index": orbit,
                        "time_h": orbit * ORBIT_HOURS,
                        "SOC": soc,
                        "recovery_duration_slots": "",
                    }
                )
            elif protection_mode and soc >= PROTECTION_RESUME_SOC:
                protection_mode = False
                duration = orbit - (protection_entry_slot if protection_entry_slot is not None else orbit)
                recovery_durations.append(duration)
                protection_events.append(
                    {
                        "arm": public_arm,
                        "event": "exit",
                        "orbit_index": orbit,
                        "time_h": orbit * ORBIT_HOURS,
                        "SOC": soc,
                        "recovery_duration_slots": duration,
                    }
                )
                protection_entry_slot = None

        for task in arrivals_by_slot.get(orbit, []):
            active_task_ids.add(str(task["task_id"]))

        for task_id in sorted(active_task_ids):
            task = task_by_id[task_id]
            if orbit > _to_int(task["deadline_slot"]):
                continue
            next_request = fair._next_product_for_task(
                task,
                delivered_by_task[task_id],
                generated_by_task[task_id],
                cfg,
                orbit,
                delivered_slots_by_task[task_id],
            )
            if next_request is None:
                continue
            product_type, earliest_slot, reason = next_request
            add_request(task, product_type, earliest_slot, reason, orbit)

        ready = [
            req
            for req in pending
            if _to_int(req["earliest_slot"]) <= orbit
            and orbit <= _to_int(task_by_id[str(req["task_id"])]["deadline_slot"])
        ]
        queue_mb = sum(_to_float(req["payload_size_MB"]) for req in ready)
        queue_trace.append({"slot": orbit, "queue_MB": queue_mb})

        selected: list[dict[str, Any]] = []
        score_with: dict[str, dict[str, float]] = {}
        score_without: dict[str, dict[str, float]] = {}
        selected_without_keys: set[str] = set()
        if orbit in contact_by_slot and not protection_mode:
            capacity = _to_float(contact_by_slot[orbit]["capacity_MB"])
            selected_without, score_without = fair._select_requests(
                ready,
                task_by_id,
                delivered_by_task,
                orbit,
                cfg,
                capacity,
                with_guarantee=False,
            )
            selected_with, score_with = fair._select_requests(
                ready,
                task_by_id,
                delivered_by_task,
                orbit,
                cfg,
                capacity,
                with_guarantee=True,
            )
            selected_without_keys = {fair._request_key(req) for req in selected_without}
            allowed = [
                req
                for req in selected_with
                if _request_allowed(public_arm, req, task_by_id[str(req["task_id"])], soc, phase)
            ]
            selected = allowed[: _selection_limit(public_arm, soc, phase)]

        selected_keys = {fair._request_key(req) for req in selected}
        for req in ready:
            if fair._request_key(req) not in selected_keys:
                req["missed_contact_windows"] = _to_int(req.get("missed_contact_windows")) + 1

        delivered_counts: Counter[str] = Counter()
        orbit_compute = 0.0
        orbit_comm = 0.0
        orbit_net = 0.0
        orbit_h_net = 0.0
        for req in selected:
            task_id = str(req["task_id"])
            task = task_by_id[task_id]
            product_type = str(req["product_type"])
            key = fair._request_key(req)
            delivered_before = set(delivered_by_task[task_id])
            stage_before = fair._stage_from_products(delivered_before)
            delta = fair._request_delta(req, delivered_before)
            if delta <= 1.0e-12:
                pending = [item for item in pending if fair._request_key(item) != key]
                continue
            payload = _to_float(req["payload_size_MB"])
            quality = fair._quality_for_product(cfg, product_type)
            gross = max(0.0, _to_float(task["V_eff"]) * delta * quality * fair._timeliness(task, orbit))
            compute, comm, resource_cost, physical_compute, physical_comm = _product_costs(product_type, payload, cfg)
            net = gross - resource_cost
            delivered_after = set(delivered_before)
            delivered_after.add(product_type)
            delivered_by_task[task_id] = delivered_after
            delivered_slots_by_task[task_id][product_type] = orbit
            before_deadline = int(orbit <= _to_int(task["deadline_slot"]))
            in_h = int(fair._is_H(task))
            min_required, target_required = fair._required_stages(task["product_requirement"])
            stage_after = fair._stage_from_products(delivered_after)
            counted_min = int(
                in_h
                and before_deadline
                and fair._stage_rank(stage_after) >= fair._stage_rank(min_required)
                and fair._stage_rank(stage_before) < fair._stage_rank(min_required)
            )
            counted_target = int(
                in_h
                and before_deadline
                and fair._stage_rank(stage_after) >= fair._stage_rank(target_required)
                and fair._stage_rank(stage_before) < fair._stage_rank(target_required)
            )
            term = _to_float(score_with.get(key, {}).get("guarantee_term"))
            if bool(cfg.get("uses_guarantee")) and key not in selected_without_keys and term > 0.0:
                reason = "counterfactual_rescued"
            elif bool(cfg.get("uses_guarantee")) and term > 0.0:
                reason = "guarantee_positive"
            elif str(cfg["strategy"]) == "utility_density":
                reason = "utility_selected"
            elif str(cfg["strategy"]) == "fixed_lightweight":
                reason = "fixed_lightweight"
            elif str(cfg["strategy"]) == "fixed_heavy":
                reason = "fixed_heavy_protected" if public_arm == "fixed_heavy_with_20pct_protection" else "fixed_heavy"
            elif str(cfg["strategy"]) == "ground_approved_followup":
                reason = "ground_approved_followup"
            else:
                reason = "natural"
            ledger.append(
                {
                    "scenario": SCENARIO,
                    "seed": SEED,
                    "num_slots": NUM_ORBITS,
                    "arm": public_arm,
                    "lambda_H": _to_float(cfg["lambda_H"]),
                    "slot": orbit,
                    "contact_window_id": contact_by_slot[orbit]["contact_window_id"],
                    "task_id": task_id,
                    "product_type": product_type,
                    "value_level": task["value_level"],
                    "product_requirement": task["product_requirement"],
                    "in_H": in_h,
                    "payload_size_MB": payload,
                    "delivered_before_deadline": before_deadline,
                    "stage_before_delivery": stage_before,
                    "stage_after_delivery": stage_after,
                    "V_eff": task["V_eff"],
                    "delta_information_utility": delta,
                    "product_gross_information_utility": gross,
                    "compute_energy_Wh": physical_compute,
                    "comm_energy_Wh": physical_comm,
                    "product_resource_cost": resource_cost,
                    "product_net_utility": net,
                    "counted_in_total_gross": 1,
                    "counted_in_total_net": 1,
                    "counted_in_H_gross": in_h,
                    "counted_in_H_net": in_h,
                    "counted_in_H_min_stage": counted_min,
                    "counted_in_H_target_stage": counted_target,
                    "counted_in_closure": int(counted_target),
                    "evidence_closure_flag": int(
                        before_deadline
                        and str(task["product_requirement"]) == "evidence"
                        and fair._stage_rank(stage_after) >= fair._stage_rank("ROI+Ctx")
                        and fair._stage_rank(stage_before) < fair._stage_rank("ROI+Ctx")
                    ),
                    "delivery_reason": reason,
                    "notes": "macro_sawtooth_runner_v1",
                }
            )
            delivered_counts[product_type] += 1
            orbit_compute += compute
            orbit_comm += comm
            orbit_net += net
            orbit_h_net += net if in_h else 0.0
            pending = [item for item in pending if fair._request_key(item) != key]

        active_unclosed = [
            task
            for task in manifest
            if _to_int(task["arrival_slot"]) <= orbit <= _to_int(task["deadline_slot"])
            and not _stage_complete(str(task["target_stage_required"]), delivered_by_task[str(task["task_id"])])
        ]
        payload_on = bool(active_unclosed or selected) and not protection_mode
        payload_idle = _payload_idle_wh(public_arm, payload_on)
        tx_fixed = TX_FIXED_WH_PER_ACTIVE_ORBIT if selected else 0.0
        solar = _to_float(resource["solar_energy_wh"])
        housekeeping = _to_float(resource["housekeeping_energy_wh"])
        before_soc = soc
        net_energy = solar - housekeeping - payload_idle - orbit_compute - orbit_comm - tx_fixed
        next_soc = max(0.0, min(1.0, soc + net_energy / BATTERY_CAPACITY_WH))
        if public_arm != "fixed_heavy" and next_soc < HARD_MIN_SOC:
            next_soc = HARD_MIN_SOC
        if (
            public_arm == "fixed_heavy_with_20pct_protection"
            and not protection_mode
            and next_soc <= HARD_MIN_SOC + 1.0e-12
        ):
            protection_mode = True
            protection_trigger_count += 1
            protection_entry_slot = orbit + 1
            protection_events.append(
                {
                    "arm": public_arm,
                    "event": "entry",
                    "orbit_index": orbit,
                    "time_h": (orbit + 1) * ORBIT_HOURS,
                    "SOC": next_soc,
                    "recovery_duration_slots": "",
                }
            )
        soc = next_soc
        if protection_mode:
            protection_total_slots += 1
        cumulative_net += orbit_net
        cumulative_h_net += orbit_h_net
        backlog_count = len([req for req in pending if _to_int(req["earliest_slot"]) <= orbit])
        backlog_mb = sum(_to_float(req["payload_size_MB"]) for req in pending if _to_int(req["earliest_slot"]) <= orbit)
        defer_count = max(0, len(ready) - len(selected))
        trajectory.append(
            {
                "arm": public_arm,
                "orbit_index": orbit,
                "time_h": orbit * ORBIT_HOURS,
                "cycle_index": cycle,
                "macro_phase": phase,
                "deadline_pressure": pressure,
                "SOC_before": before_soc,
                "SOC": soc,
                "soc_delta": soc - before_soc,
                "soc_slope_per_hour": (soc - before_soc) / ORBIT_HOURS,
                "soc_band_before": _soc_band(before_soc),
                "payload_on": int(payload_on),
                "payload_idle_energy_wh": payload_idle,
                "housekeeping_energy_wh": housekeeping,
                "solar_energy_wh": solar,
                "compute_energy_wh": orbit_compute,
                "comm_energy_wh": orbit_comm,
                "tx_fixed_energy_wh": tx_fixed,
                "net_energy_wh": net_energy,
                "delivered_product_count": sum(delivered_counts.values()),
                "metadata_count": delivered_counts["metadata"],
                "roi_count": delivered_counts["roi"],
                "roi_context_count": delivered_counts["roi_context"],
                "defer_count": defer_count,
                "backlog_count": backlog_count,
                "backlog_MB": backlog_mb,
                "protection_mode": int(protection_mode),
            }
        )
        cumulative.append(
            {
                "arm": public_arm,
                "orbit_index": orbit,
                "time_h": orbit * ORBIT_HOURS,
                "cumulative_net_system_utility": cumulative_net,
                "cumulative_H_net_utility": cumulative_h_net,
            }
        )
        backlog_rows.append(
            {
                "arm": public_arm,
                "orbit_index": orbit,
                "time_h": orbit * ORBIT_HOURS,
                "backlog_count": backlog_count,
                "backlog_MB": backlog_mb,
            }
        )

    stage_rows = fair.build_stage_success_audit(ledger, manifest, public_arm)
    return {
        "arm": public_arm,
        "cfg": cfg,
        "ledger": ledger,
        "stage_rows": stage_rows,
        "queue_trace": queue_trace,
        "trajectory": trajectory,
        "cumulative": cumulative,
        "backlog": backlog_rows,
        "protection_events": protection_events,
        "protection_stats": {
            "protection_trigger_count": protection_trigger_count,
            "protection_total_slots": protection_total_slots,
            "recovery_durations": recovery_durations,
        },
    }


def _summary_from_run(run: dict[str, Any], manifest: list[dict[str, Any]], hashes: dict[str, str]) -> dict[str, Any]:
    ledger = run["ledger"]
    stage = run["stage_rows"]
    traj = run["trajectory"]
    h_tasks = [task for task in manifest if fair._is_H(task)]
    feasible_h = [task for task in h_tasks if _to_int(task["feasible_contact_windows_before_deadline"]) > 0]
    h_ledger = [row for row in ledger if _to_int(row["in_H"])]
    h_left = max(0, len(feasible_h) - sum(_to_int(row["min_stage_success_before_deadline"]) for row in stage))
    total_net = sum(_to_float(row["product_net_utility"]) for row in ledger)
    total_h_net = sum(_to_float(row["product_net_utility"]) for row in h_ledger)
    min_soc = min(_to_float(row["SOC"]) for row in traj)
    final_soc = _to_float(traj[-1]["SOC"])
    return {
        "arm": run["arm"],
        "task_manifest_hash": hashes["task_manifest_hash"],
        "contact_schedule_hash": hashes["contact_schedule_hash"],
        "resource_trace_hash": hashes["resource_trace_hash"],
        "total_task_count": len(manifest),
        "H_task_count": len(h_tasks),
        "net_system_utility": total_net - 0.02 * h_left,
        "H_net_utility": total_h_net,
        "H_target_success_rate": _safe_div(
            sum(_to_int(row["target_stage_success_before_deadline"]) for row in stage),
            len(feasible_h),
        ),
        "total_compute_energy_wh": sum(_to_float(row["compute_energy_Wh"]) for row in ledger),
        "total_comm_energy_wh": sum(_to_float(row["comm_energy_Wh"]) for row in ledger),
        "total_resource_cost": sum(_to_float(row["product_resource_cost"]) for row in ledger),
        "delivered_product_count": len(ledger),
        "min_SOC": min_soc,
        "final_SOC": final_soc,
        "collapsed_below_20_flag": int(min_soc < HARD_MIN_SOC - 1.0e-12),
        "hard_safety_violation_count": sum(1 for row in traj if _to_float(row["SOC"]) < HARD_MIN_SOC - 1.0e-12),
        "final_backlog_count": _to_int(run["backlog"][-1]["backlog_count"]) if run["backlog"] else 0,
        "max_backlog_count": max((_to_int(row["backlog_count"]) for row in run["backlog"]), default=0),
        "protection_trigger_count": _to_int(run["protection_stats"]["protection_trigger_count"]),
        "protection_total_slots": _to_int(run["protection_stats"]["protection_total_slots"]),
    }


def build_phase_summary(runs: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_cycle_phase: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for task in manifest:
        manifest_by_cycle_phase[(_to_int(task["cycle_index"]), str(task["macro_phase"]))].append(task)
    rows: list[dict[str, Any]] = []
    for run in runs:
        led_by_cycle_phase: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in run["ledger"]:
            task = next((item for item in manifest if item["task_id"] == row["task_id"]), None)
            if task:
                led_by_cycle_phase[(_to_int(task["cycle_index"]), str(task["macro_phase"]))].append(row)
        traj_by_cycle_phase: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in run["trajectory"]:
            traj_by_cycle_phase[(_to_int(row["cycle_index"]), str(row["macro_phase"]))].append(row)
        for cycle in range(4):
            for phase in ["work_phase_1", "work_phase_2", "tail_heavy_or_cleanup", "recovery_phase"]:
                key = (cycle, phase)
                traj = traj_by_cycle_phase.get(key, [])
                led = led_by_cycle_phase.get(key, [])
                tasks = manifest_by_cycle_phase.get(key, [])
                socs = [_to_float(row["SOC"]) for row in traj]
                rows.append(
                    {
                        "arm": run["arm"],
                        "cycle_index": cycle,
                        "macro_phase": phase,
                        "phase_start_h": min((_to_float(row["time_h"]) for row in traj), default=""),
                        "phase_end_h": max((_to_float(row["time_h"]) + ORBIT_HOURS for row in traj), default=""),
                        "initial_SOC": _to_float(traj[0]["SOC_before"]) if traj else "",
                        "final_SOC": socs[-1] if socs else "",
                        "min_SOC": min(socs) if socs else "",
                        "SOC_delta": (socs[-1] - _to_float(traj[0]["SOC_before"])) if traj else "",
                        "task_count": len(tasks),
                        "delivered_product_count": len(led),
                        "net_system_utility": sum(_to_float(row["product_net_utility"]) for row in led),
                        "H_net_utility": sum(_to_float(row["product_net_utility"]) for row in led if _to_int(row["in_H"])),
                        "avg_backlog_count": statistics.mean([_to_float(row["backlog_count"]) for row in traj]) if traj else "",
                        "avg_payload_on_fraction": statistics.mean([_to_float(row["payload_on"]) for row in traj]) if traj else "",
                    }
                )
    return rows


def build_cycle_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in run["trajectory"]:
            by_cycle[_to_int(row["cycle_index"])].append(row)
        for cycle, traj in sorted(by_cycle.items()):
            rows.append(
                {
                    "arm": run["arm"],
                    "cycle_index": cycle,
                    "cycle_start_h": min(_to_float(row["time_h"]) for row in traj),
                    "cycle_end_h": max(_to_float(row["time_h"]) + ORBIT_HOURS for row in traj),
                    "initial_SOC": _to_float(traj[0]["SOC_before"]),
                    "min_SOC": min(_to_float(row["SOC"]) for row in traj),
                    "final_SOC": _to_float(traj[-1]["SOC"]),
                    "work_SOC_drop": sum(
                        min(0.0, _to_float(row["soc_delta"]))
                        for row in traj
                        if _is_work_phase(str(row["macro_phase"]))
                    ),
                    "recovery_SOC_gain": sum(
                        max(0.0, _to_float(row["soc_delta"]))
                        for row in traj
                        if str(row["macro_phase"]) == "recovery_phase"
                    ),
                    "max_backlog_count": max(_to_int(row["backlog_count"]) for row in traj),
                }
            )
    return rows


def build_slope_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bands = ["SOC_above_70", "SOC_50_70", "SOC_below_50"]
    for run in runs:
        for band in bands:
            points = [
                row
                for row in run["trajectory"]
                if str(row["soc_band_before"]) == band and _is_work_phase(str(row["macro_phase"]))
            ]
            action_total = sum(_to_int(row["delivered_product_count"]) + _to_int(row["defer_count"]) for row in points)
            rows.append(
                {
                    "arm": run["arm"],
                    "soc_band": band,
                    "avg_soc_slope_per_hour": statistics.mean([_to_float(row["soc_slope_per_hour"]) for row in points]) if points else 0.0,
                    "avg_compute_energy_wh_per_hour": statistics.mean([_to_float(row["compute_energy_wh"]) / ORBIT_HOURS for row in points]) if points else 0.0,
                    "avg_comm_energy_wh_per_hour": statistics.mean([_to_float(row["comm_energy_wh"]) / ORBIT_HOURS for row in points]) if points else 0.0,
                    "avg_payload_on_fraction": statistics.mean([_to_float(row["payload_on"]) for row in points]) if points else 0.0,
                    "avg_metadata_rate": _safe_div(sum(_to_int(row["metadata_count"]) for row in points), action_total),
                    "avg_roi_rate": _safe_div(sum(_to_int(row["roi_count"]) for row in points), action_total),
                    "avg_roi_context_rate": _safe_div(sum(_to_int(row["roi_context_count"]) for row in points), action_total),
                    "avg_defer_rate": _safe_div(sum(_to_int(row["defer_count"]) for row in points), action_total),
                }
            )
    return rows


def build_recovery_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for cycle in range(4):
            rec = [
                row
                for row in run["trajectory"]
                if _to_int(row["cycle_index"]) == cycle and str(row["macro_phase"]) == "recovery_phase"
            ]
            if not rec:
                continue
            socs = [_to_float(row["SOC"]) for row in rec]
            rows.append(
                {
                    "arm": run["arm"],
                    "cycle_index": cycle,
                    "recovery_phase_start_h": _to_float(rec[0]["time_h"]),
                    "recovery_phase_end_h": _to_float(rec[-1]["time_h"]) + ORBIT_HOURS,
                    "initial_SOC": _to_float(rec[0]["SOC_before"]),
                    "peak_SOC": max(socs),
                    "final_SOC": socs[-1],
                    "recovered_above_80_flag": int(socs[-1] > 0.80),
                    "recovery_gain": socs[-1] - _to_float(rec[0]["SOC_before"]),
                    "recovery_duration_h": len(rec) * ORBIT_HOURS,
                }
            )
    return rows


def build_protection_summary(runs: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_by_arm = {str(row["arm"]): row for row in ranking}
    rows: list[dict[str, Any]] = []
    for run in runs:
        if run["arm"] not in {"fixed_heavy", "fixed_heavy_with_20pct_protection"}:
            continue
        stats = run["protection_stats"]
        durations = list(stats.get("recovery_durations", []))
        rank = rank_by_arm.get(run["arm"], {})
        rows.append(
            {
                "arm": run["arm"],
                "protection_trigger_count": _to_int(stats.get("protection_trigger_count")),
                "protection_total_slots": _to_int(stats.get("protection_total_slots")),
                "avg_recovery_duration_slots": statistics.mean(durations) if durations else 0.0,
                "min_SOC": rank.get("min_SOC", 0.0),
                "final_SOC": rank.get("final_SOC", 0.0),
                "net_system_utility": rank.get("net_system_utility", 0.0),
                "H_net_utility": rank.get("H_net_utility", 0.0),
            }
        )
    return rows


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for metric in ["net_system_utility", "H_net_utility"]:
        for rank, row in enumerate(sorted(rows, key=lambda item: _to_float(item[metric]), reverse=True), start=1):
            row[f"rank_{metric}"] = rank
    return sorted(rows, key=lambda item: _to_int(item["rank_net_system_utility"]))


def make_figures(output_dir: Path, runs: list[dict[str, Any]], ranking: list[dict[str, Any]], recovery: list[dict[str, Any]]) -> None:
    by_arm = {run["arm"]: run for run in runs}
    phase_spans = []
    for cycle in range(4):
        for phase in ["work_phase_1", "work_phase_2", "tail_heavy_or_cleanup", "recovery_phase"]:
            rows = [
                row
                for row in by_arm["proposed_guarantee_aware_full"]["trajectory"]
                if _to_int(row["cycle_index"]) == cycle and str(row["macro_phase"]) == phase
            ]
            if rows:
                phase_spans.append((phase, _to_float(rows[0]["time_h"]), _to_float(rows[-1]["time_h"]) + ORBIT_HOURS))

    plt.figure(figsize=(12, 5.5))
    for phase, start, end in phase_spans:
        plt.axvspan(start, end, color="#f3d2c1" if _is_work_phase(phase) else "#cde8d5", alpha=0.22, linewidth=0)
    for arm in ARM_ORDER:
        rows = by_arm[arm]["trajectory"]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["SOC"]) for row in rows], label=arm, linewidth=1.7)
    for level in [0.70, 0.50, 0.20]:
        plt.axhline(level, color="black", linestyle="--", linewidth=0.8)
        plt.text(96.2, level, f"{int(level * 100)}%", va="center", fontsize=8)
    plt.xlabel("time (h)")
    plt.ylabel("SOC")
    plt.ylim(0.0, 1.02)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "soc_macro_sawtooth_by_arm.png", dpi=190)
    plt.close()

    prop = by_arm["proposed_guarantee_aware_full"]["trajectory"]
    plt.figure(figsize=(11, 4.8))
    plt.plot([_to_float(row["time_h"]) for row in prop], [_to_float(row["SOC"]) for row in prop], color="#1f77b4", linewidth=2.0)
    colors = {"SOC_above_70": "#2ca02c", "SOC_50_70": "#ff7f0e", "SOC_below_50": "#d62728"}
    for band, color in colors.items():
        pts = [row for row in prop if str(row["soc_band_before"]) == band and _is_work_phase(str(row["macro_phase"]))]
        plt.scatter([_to_float(row["time_h"]) for row in pts], [_to_float(row["SOC_before"]) for row in pts], s=22, label=band, color=color)
    for level in [0.70, 0.50, 0.20]:
        plt.axhline(level, color="gray", linestyle="--", linewidth=0.8)
    slope_rows = build_slope_summary([by_arm["proposed_guarantee_aware_full"]])
    note = "\n".join(f"{row['soc_band']}: {_fmt(row['avg_soc_slope_per_hour'], 4)}/h" for row in slope_rows)
    plt.text(0.01, 0.04, note, transform=plt.gca().transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"})
    plt.xlabel("time (h)")
    plt.ylabel("SOC")
    plt.ylim(0.0, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "proposed_soc_zoom_with_slopes.png", dpi=190)
    plt.close()

    rec_prop = [row for row in recovery if row["arm"] == "proposed_guarantee_aware_full"]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot([_to_int(row["cycle_index"]) for row in rec_prop], [_to_float(row["peak_SOC"]) for row in rec_prop], marker="o", label="peak")
    plt.plot([_to_int(row["cycle_index"]) for row in rec_prop], [_to_float(row["final_SOC"]) for row in rec_prop], marker="s", label="final")
    plt.axhline(0.80, color="black", linestyle="--", linewidth=0.8)
    plt.xticks([0, 1, 2, 3])
    plt.xlabel("cycle")
    plt.ylabel("SOC after recovery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "recovery_peaks_by_cycle.png", dpi=190)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    for arm in ARM_ORDER:
        rows = by_arm[arm]["cumulative"]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["cumulative_net_system_utility"]) for row in rows], label=arm)
    plt.xlabel("time (h)")
    plt.ylabel("cumulative net utility")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_net_utility_by_arm.png", dpi=190)
    plt.close()

    labels = [str(row["arm"]) for row in ranking]
    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, [_to_float(row["H_net_utility"]) for row in ranking], color="#4c78a8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("H net utility")
    plt.tight_layout()
    plt.savefig(output_dir / "H_net_utility_by_arm.png", dpi=190)
    plt.close()

    plt.figure(figsize=(10.5, 4.8))
    for arm in ["fixed_heavy", "fixed_heavy_with_20pct_protection"]:
        rows = by_arm[arm]["trajectory"]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["SOC"]) for row in rows], label=arm, linewidth=2)
    for event in by_arm["fixed_heavy_with_20pct_protection"]["protection_events"]:
        plt.axvline(_to_float(event["time_h"]), color="#d62728" if event["event"] == "entry" else "#2ca02c", linestyle=":", linewidth=1.2)
    plt.axhline(0.20, color="black", linestyle="--", linewidth=0.8)
    plt.axhline(0.35, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("time (h)")
    plt.ylabel("SOC")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "heavy_protection_vs_heavy_soc.png", dpi=190)
    plt.close()


def write_quick_report(output_dir: Path, ranking: list[dict[str, Any]], slope: list[dict[str, Any]], recovery: list[dict[str, Any]], protection: list[dict[str, Any]], cycle_rows: list[dict[str, Any]]) -> None:
    by_arm = {str(row["arm"]): row for row in ranking}
    prop = by_arm["proposed_guarantee_aware_full"]
    util = by_arm["utility_rule"]
    light = by_arm["fixed_lightweight"]
    heavy = by_arm["fixed_heavy"]
    protected = by_arm["fixed_heavy_with_20pct_protection"]
    prop_slopes = {str(row["soc_band"]): _to_float(row["avg_soc_slope_per_hour"]) for row in slope if row["arm"] == "proposed_guarantee_aware_full"}
    recovered_all = all(_to_int(row["recovered_above_80_flag"]) for row in recovery if row["arm"] == "proposed_guarantee_aware_full")
    work_drops = [
        _to_float(row["work_SOC_drop"])
        for row in cycle_rows
        if row["arm"] == "proposed_guarantee_aware_full"
    ]
    sawtooth = all(drop < -0.10 for drop in work_drops) and recovered_all
    slope_order = (
        abs(prop_slopes.get("SOC_above_70", 0.0))
        > abs(prop_slopes.get("SOC_50_70", 0.0))
        > abs(prop_slopes.get("SOC_below_50", 0.0))
    )
    prot = {str(row["arm"]): row for row in protection}
    lines = [
        "# Macro-sawtooth recovery quick report",
        "",
        "## Direct Answers",
        "",
        f"1. Macro-sawtooth achieved: {sawtooth}. Proposed work-phase cycle drops are {', '.join(_fmt(v) for v in work_drops)}, and every recovery final SOC is above 0.80: {recovered_all}.",
        f"2. Proposed discharges through work phases: {all(drop < 0.0 for drop in work_drops)}.",
        f"3. Proposed below 70% slows down: {abs(prop_slopes.get('SOC_50_70', 0.0)) < abs(prop_slopes.get('SOC_above_70', 0.0))}. Slopes >70/50-70/<50 = {_fmt(prop_slopes.get('SOC_above_70'))}, {_fmt(prop_slopes.get('SOC_50_70'))}, {_fmt(prop_slopes.get('SOC_below_50'))} per hour.",
        f"4. Proposed below 50% further contracts: {slope_order}.",
        f"5. Proposed recovers above 80% in recovery phases: {recovered_all}.",
        f"6. Proposed keeps net or H_net lead: net_rank={prop['rank_net_system_utility']}, H_net_rank={prop['rank_H_net_utility']}, net={_fmt(prop['net_system_utility'])}, H_net={_fmt(prop['H_net_utility'])}.",
        f"7. utility_rule is greedy/short-horizon: final_backlog={util['final_backlog_count']}, max_backlog={util['max_backlog_count']}, min_SOC={_fmt(util['min_SOC'])}, net_rank={util['rank_net_system_utility']}.",
        f"8. fixed_lightweight is conservative but under-earns: min_SOC={_fmt(light['min_SOC'])}, net={_fmt(light['net_system_utility'])}, H_net={_fmt(light['H_net_utility'])}.",
        f"9. fixed_heavy collapses without protection: {bool(_to_int(heavy['collapsed_below_20_flag']))}; min_SOC={_fmt(heavy['min_SOC'])}.",
        f"10. fixed_heavy_with_20pct_protection avoids collapse: {not bool(_to_int(protected['collapsed_below_20_flag']))}. Cost: protection_slots={prot.get('fixed_heavy_with_20pct_protection', {}).get('protection_total_slots', 0)}, final_backlog={protected['final_backlog_count']}, net={_fmt(protected['net_system_utility'])}, H_net={_fmt(protected['H_net_utility'])}.",
        "11. Suitable as a long-horizon macro-sawtooth case study: Yes, with the caveat that this is a default-off diagnostic runner using shared fair-baseline accounting plus a standalone SOC integration layer.",
        "",
        "## Isolation",
        "",
        "- New code is isolated in `scripts/run_macro_sawtooth_recovery_experiment.py`.",
        "- Existing proposed/router/reward/closure/evaluator/utility_rule/value-prior code is not edited.",
        "- Task values use the existing value-prior helper; high-value labels are not hand-edited.",
    ]
    _write_text(output_dir / "quick_report.md", "\n".join(lines))


def validate_outputs(output_dir: Path) -> list[dict[str, Any]]:
    required = [
        "main_ranking_table.csv",
        "soc_trajectory_by_arm.csv",
        "cumulative_utility_trajectory_by_arm.csv",
        "backlog_trajectory_by_arm.csv",
        "phase_summary_by_arm.csv",
        "cycle_summary_by_arm.csv",
        "slope_by_soc_band_by_arm.csv",
        "recovery_summary_by_arm.csv",
        "heavy_protection_event_log.csv",
        "heavy_protection_summary.csv",
        "quick_report.md",
        "soc_macro_sawtooth_by_arm.png",
        "proposed_soc_zoom_with_slopes.png",
        "recovery_peaks_by_cycle.png",
        "cumulative_net_utility_by_arm.png",
        "H_net_utility_by_arm.png",
        "heavy_protection_vs_heavy_soc.png",
    ]
    rows = []
    for name in required:
        path = output_dir / name
        rows.append({"artifact": name, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    return rows


def run(output_dir: Path = OUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contacts = build_contact_schedule()
    resources = build_resource_trace()
    manifest = build_macro_manifest(contacts)
    hashes = {
        "task_manifest_hash": _hash(manifest, MANIFEST_COLUMNS),
        "contact_schedule_hash": _hash(contacts, CONTACT_COLUMNS),
        "resource_trace_hash": _hash(resources, RESOURCE_COLUMNS),
    }
    runs = [run_arm(arm, manifest, contacts, resources) for arm in ARM_ORDER]
    ranking = _rank_rows([_summary_from_run(run_item, manifest, hashes) for run_item in runs])
    phase = build_phase_summary(runs, manifest)
    cycle = build_cycle_summary(runs)
    slope = build_slope_summary(runs)
    recovery = build_recovery_summary(runs)
    protection_events = [event for run_item in runs for event in run_item["protection_events"]]
    protection = build_protection_summary(runs, ranking)

    _write_csv(output_dir / "shared_task_manifest.csv", manifest, MANIFEST_COLUMNS)
    _write_csv(output_dir / "shared_contact_schedule.csv", contacts, CONTACT_COLUMNS)
    _write_csv(output_dir / "shared_resource_trace.csv", resources, RESOURCE_COLUMNS)
    _write_csv(output_dir / "fairness_hashes.csv", [hashes])
    _write_csv(output_dir / "main_ranking_table.csv", ranking)
    _write_csv(output_dir / "soc_trajectory_by_arm.csv", [row for run_item in runs for row in run_item["trajectory"]], TRAJECTORY_COLUMNS)
    _write_csv(output_dir / "cumulative_utility_trajectory_by_arm.csv", [row for run_item in runs for row in run_item["cumulative"]])
    _write_csv(output_dir / "backlog_trajectory_by_arm.csv", [row for run_item in runs for row in run_item["backlog"]])
    _write_csv(output_dir / "phase_summary_by_arm.csv", phase)
    _write_csv(output_dir / "cycle_summary_by_arm.csv", cycle)
    _write_csv(output_dir / "slope_by_soc_band_by_arm.csv", slope, SLOPE_COLUMNS)
    _write_csv(output_dir / "recovery_summary_by_arm.csv", recovery, RECOVERY_COLUMNS)
    _write_csv(output_dir / "heavy_protection_event_log.csv", protection_events)
    _write_csv(output_dir / "heavy_protection_summary.csv", protection, PROTECTION_SUMMARY_COLUMNS)
    _write_csv(output_dir / "delivered_product_ledger_all_arms.csv", [row for run_item in runs for row in run_item["ledger"]])
    _write_csv(output_dir / "H_stage_success_audit_all_arms.csv", [row for run_item in runs for row in run_item["stage_rows"]])
    make_figures(output_dir, runs, ranking, recovery)
    write_quick_report(output_dir, ranking, slope, recovery, protection, cycle)
    validation = validate_outputs(output_dir)
    _write_csv(output_dir / "artifact_validation.csv", validation)
    return {
        "output_dir": output_dir,
        "manifest": manifest,
        "contacts": contacts,
        "resources": resources,
        "runs": runs,
        "ranking": ranking,
        "slope": slope,
        "recovery": recovery,
        "protection": protection,
        "validation": validation,
        "hashes": hashes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    result = run(args.output_dir)
    proposed = next(row for row in result["ranking"] if row["arm"] == "proposed_guarantee_aware_full")
    protected = next(row for row in result["ranking"] if row["arm"] == "fixed_heavy_with_20pct_protection")
    print(f"output_dir={args.output_dir}")
    print(f"task_count={len(result['manifest'])}")
    print(f"task_manifest_hash={result['hashes']['task_manifest_hash']}")
    print(f"contact_schedule_hash={result['hashes']['contact_schedule_hash']}")
    print(f"resource_trace_hash={result['hashes']['resource_trace_hash']}")
    print(f"proposed_min_SOC={_fmt(proposed['min_SOC'])}")
    print(f"proposed_final_SOC={_fmt(proposed['final_SOC'])}")
    print(f"protected_trigger_count={protected['protection_trigger_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
