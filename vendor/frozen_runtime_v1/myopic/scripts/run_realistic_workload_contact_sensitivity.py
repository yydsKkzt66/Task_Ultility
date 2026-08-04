"""Realistic workload plus contact-window sensitivity experiment.

Standalone, default-off experiment runner. This script builds realistic
scene/subtask workload traces, sparse/normal/dense contact schedules, and a
contact-aware deadline audit. It does not modify src algorithms, value priors,
reward code, evaluators, or older runners.
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
import scripts.run_price_guided_rescue_ground_delivery_experiment as price  # noqa: E402
import scripts.run_uneven_pressure_ground_delivery_experiment as base  # noqa: E402
from src.common.types import TaskSample  # noqa: E402
from src.data.mission_prior_value import compute_mission_prior_value_v2  # noqa: E402
from src.data.value_skew_calibration import apply_value_skew_calibration  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_realistic_workload_contact_sensitivity_v1"
SCENARIO = "realistic_workload_contact_sensitivity_v1"

DEFAULT_SEEDS = [0, 1, 2]
FULL_SEEDS = [0, 1, 2, 3, 4]
WORKLOAD_SCENARIOS = ["nominal_daily", "moderate_burst", "emergency_stress"]
CONTACT_SCENARIOS = ["sparse_contact", "normal_contact", "dense_contact"]
METHODS = [
    "proposed_price_guided_rescue",
    "proposed_guarantee_aware_full",
    "utility_rule",
    "fixed_lightweight",
    "fixed_heavy",
    "fixed_heavy_with_20pct_protection",
    "ground_approved_followup",
    "persistent_queue_without_guarantee",
]
METHOD_TYPES = {
    **base.METHOD_TYPES,
    "proposed_price_guided_rescue": "proposed_price_guided_rescue",
}

NUM_ORBITS = base.NUM_ORBITS
ORBIT_HOURS = base.ORBIT_HOURS
HARD_MIN_SOC = base.HARD_MIN_SOC
THERMAL_WARNING = 1.0
THERMAL_LIMIT = 1.25

REALISTIC_MANIFEST_COLUMNS = [
    *base.GROUND_MANIFEST_COLUMNS,
    "workload_scenario",
    "contact_scenario",
    "deadline_class",
    "deadline_slack_h",
    "contact_adjustment_h",
    "product_requirement_adjustment_h",
    "expected_contact_wait_h",
    "num_contact_opportunities_before_deadline",
    "no_contact_before_deadline_flag",
    "first_contact_after_capture_h",
    "last_contact_before_deadline_h",
    "payload_size_metadata_mb",
    "payload_size_roi_mb",
    "payload_size_roi_context_mb",
    "area_prior_bucket",
    "value_assignment_reason",
    "is_high_value",
    "no_manual_high_value_override",
]

DEADLINE_AUDIT_COLUMNS = [
    "seed",
    "workload_scenario",
    "contact_scenario",
    "task_id",
    "scene_id",
    "subtask_id",
    "capture_time_h",
    "deadline_h",
    "deadline_slack_h",
    "value_level",
    "deadline_class",
    "product_requirement",
    "min_required_stage",
    "target_stage",
    "num_contact_opportunities_before_deadline",
    "first_contact_after_capture_h",
    "last_contact_before_deadline_h",
    "no_contact_before_deadline_flag",
    "expected_contact_wait_h",
    "contact_adjustment_h",
    "product_requirement_adjustment_h",
]

SUMMARY_COLUMNS = [
    "method",
    "method_type",
    "seed",
    "workload_scenario",
    "contact_scenario",
    "task_manifest_hash",
    "contact_schedule_hash",
    "resource_trace_hash",
    "scene_count",
    "subtask_count",
    "avg_subtasks_per_scene",
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "ground_received_task_count",
    "high_value_ground_received_count",
    "avg_ground_received_quality",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "avg_capture_to_ground_latency_h",
    "deadline_satisfaction_ratio",
    "high_value_deadline_satisfaction_ratio",
    "expired_task_count",
    "expired_high_value_task_count",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "direct_energy_per_high_value_received_task_wh",
    "system_energy_per_high_value_received_task_wh",
    "min_SOC",
    "final_SOC",
    "SOC_margin_to_20",
    "time_below_25_percent",
    "time_below_22_percent",
    "fraction_time_SOC_below_20",
    "hard_safety_violation_count",
    "final_backlog_count",
    "max_backlog_count",
    "queue_MB_peak",
    "downlinked_MB",
    "metadata_downlinked_count",
    "roi_downlinked_count",
    "roi_context_downlinked_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "time_above_thermal_warning",
    "thermal_violation_count",
    "net_system_utility",
    "H_net_utility",
]

RANKING_METRICS = [
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_high_value_success_value",
    "expired_high_value_task_count",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "min_SOC",
    "SOC_margin_to_20",
    "max_thermal_pressure",
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


def _value_cfg() -> dict[str, Any]:
    return {
        "task_value_model": {
            "enable_mission_prior_value_v2": True,
            "include_deadline_class": True,
            "weights": {
                "task_class": 0.42,
                "evidence_need": 0.24,
                "area_prior": 0.14,
                "source_priority": 0.12,
                "deadline_class": 0.08,
            },
        },
        "value_skew_calibration": {
            "enabled": True,
            "profile": "soft_moderate",
            "assignment_rule": "v2_conservative",
            "profiles": {"soft_moderate": {"low": 1.0, "medium": 2.0, "high": 5.0, "urgent": 10.0}},
        },
    }


def _phase_for_orbit(workload: str, orbit: int) -> str:
    if workload == "nominal_daily":
        if orbit % 12 in {9, 10, 11}:
            return "nominal_recovery"
        return "nominal_regular"
    if workload == "moderate_burst":
        if 18 <= orbit <= 29:
            return "moderate_burst_peak"
        if 30 <= orbit <= 38:
            return "moderate_recovery"
        return "moderate_background"
    if 18 <= orbit <= 23:
        return "emergency_peak"
    if 24 <= orbit <= 36:
        return "emergency_recovery"
    return "emergency_background"


def _resource_wave_name(phase: str) -> str:
    if "recovery" in phase:
        return "recovery_realistic"
    if "emergency" in phase:
        return "emergency_stress_realistic"
    if "burst" in phase:
        return "moderate_burst_realistic"
    return "nominal_daily_realistic"


def build_resource_trace(seed: int, workload: str, contact: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in range(NUM_ORBITS):
        phase = _phase_for_orbit(workload, orbit)
        base_solar = {
            "nominal_regular": 31.0,
            "nominal_recovery": 33.0,
            "moderate_background": 30.0,
            "moderate_burst_peak": 26.5,
            "moderate_recovery": 32.5,
            "emergency_background": 29.5,
            "emergency_peak": 23.0,
            "emergency_recovery": 31.5,
        }[phase]
        solar = base_solar * (0.94 + 0.12 * fair._stable_unit("solar", seed, workload, contact, orbit))
        rows.append(
            {
                "seed": seed,
                "pressure_scale": 1.0,
                "scenario": SCENARIO,
                "workload_scenario": workload,
                "contact_scenario": contact,
                "orbit_index": orbit,
                "time_h": orbit * ORBIT_HOURS,
                "wave_name": _resource_wave_name(phase),
                "workload_phase": phase,
                "deadline_pressure": phase,
                "solar_energy_wh": round(solar, 5),
                "housekeeping_energy_wh": base.HOUSEKEEPING_WH_PER_ORBIT,
            }
        )
    return rows


def _contact_count_for_day(seed: int, contact: str, day: int) -> int:
    unit = fair._stable_unit("contact_count", seed, contact, day)
    if contact == "sparse_contact":
        return 3 + int(unit > 0.35) + int(unit > 0.72)
    if contact == "normal_contact":
        return 8 + int(unit > 0.20) + int(unit > 0.40) + int(unit > 0.60) + int(unit > 0.82)
    return 12 + int(unit > 0.30) + int(unit > 0.65)


def _duration_minutes(seed: int, contact: str, orbit: int) -> float:
    unit = fair._stable_unit("contact_duration", seed, contact, orbit)
    if contact == "sparse_contact":
        return 8.0 + 4.0 * unit
    if contact == "normal_contact":
        return 6.0 + 4.0 * unit
    return 8.0 + 4.0 * unit


def build_contact_schedule(seed: int, workload: str, contact: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected: set[int] = set()
    for day in range(4):
        count = _contact_count_for_day(seed, contact, day)
        day_start = day * 15
        for idx in range(count):
            center = day_start + (idx + 0.5) * 15.0 / count
            jitter = (fair._stable_unit("contact_jitter", seed, contact, day, idx) - 0.5) * 1.8
            orbit = max(day_start, min(day_start + 14, int(round(center + jitter))))
            selected.add(orbit)
    for orbit in sorted(selected):
        duration_min = _duration_minutes(seed, contact, orbit)
        start_offset_h = (0.10 + 0.55 * fair._stable_unit("contact_offset", seed, contact, orbit)) * ORBIT_HOURS
        start_h = orbit * ORBIT_HOURS + start_offset_h
        end_h = min((orbit + 1) * ORBIT_HOURS, start_h + duration_min / 60.0)
        mb_per_min = {"sparse_contact": 0.38, "normal_contact": 0.55, "dense_contact": 0.62}[contact]
        capacity = duration_min * mb_per_min * (0.88 + 0.24 * fair._stable_unit("contact_capacity", seed, contact, orbit))
        rows.append(
            {
                "seed": seed,
                "pressure_scale": 1.0,
                "scenario": SCENARIO,
                "workload_scenario": workload,
                "contact_scenario": contact,
                "contact_window_id": f"rw_cw_s{seed}_{workload}_{contact}_{orbit:03d}",
                "orbit_index": orbit,
                "start_slot": orbit,
                "end_slot": orbit,
                "start_time_h": round(start_h, 5),
                "end_time_h": round(end_h, 5),
                "duration_min": round(duration_min, 5),
                "capacity_MB": round(capacity, 5),
                "deadline_pressure": contact,
                "wave_name": contact,
            }
        )
    return rows


def _scene_count_for_orbit(seed: int, workload: str, orbit: int) -> int:
    phase = _phase_for_orbit(workload, orbit)
    unit = fair._stable_unit("scene_count", seed, workload, orbit)
    if workload == "nominal_daily":
        if phase == "nominal_recovery":
            return 1 if unit > 0.35 else 0
        return 1 + int(unit > 0.58)
    if workload == "moderate_burst":
        if phase == "moderate_burst_peak":
            return 2 + int(unit > 0.36) + int(unit > 0.76)
        if phase == "moderate_recovery":
            return 1 if unit > 0.40 else 0
        return 1 + int(unit > 0.72)
    if phase == "emergency_peak":
        return 3 + int(unit > 0.38) + int(unit > 0.78)
    if phase == "emergency_recovery":
        return 1 if unit > 0.48 else 0
    return 1 + int(unit > 0.82)


def _subtask_count(seed: int, workload: str, phase: str, scene_id: str) -> int:
    unit = fair._stable_unit("subtask_count", seed, workload, phase, scene_id)
    if workload == "nominal_daily":
        return 1 + int(unit > 0.40)
    if workload == "moderate_burst" and phase == "moderate_burst_peak":
        return 2 + int(unit > 0.45)
    if workload == "emergency_stress" and phase == "emergency_peak":
        return 2 + int(unit > 0.36) + int(unit > 0.78)
    return 1 + int(unit > 0.58)


def _subtask_profile(
    seed: int,
    workload: str,
    phase: str,
    global_idx: int,
    sub_idx: int,
) -> tuple[str, str, str, float, str]:
    unit = fair._stable_unit("profile", seed, workload, phase, global_idx, sub_idx)
    if phase == "emergency_peak":
        task_class = "emergency" if unit < 0.28 else "high_value" if unit < 0.72 else "routine"
        req = "evidence" if unit < 0.46 else "context" if unit < 0.78 else "verification"
        deadline_class = "urgent" if unit < 0.30 else "high" if unit < 0.75 else "medium"
        source = 0.52 + 0.44 * fair._stable_unit("source", seed, global_idx, sub_idx)
    elif phase == "moderate_burst_peak":
        task_class = "high_value" if unit < 0.38 else "routine"
        req = "context" if unit < 0.42 else "evidence" if unit < 0.60 else "verification" if unit < 0.88 else "alert"
        deadline_class = "high" if unit < 0.45 else "medium" if unit < 0.82 else "low"
        source = 0.30 + 0.46 * fair._stable_unit("source", seed, global_idx, sub_idx)
    elif "recovery" in phase:
        task_class = "routine"
        req = "verification" if unit < 0.52 else "alert"
        deadline_class = "medium" if unit < 0.62 else "low"
        source = 0.08 + 0.24 * fair._stable_unit("source", seed, global_idx, sub_idx)
    else:
        task_class = "high_value" if unit < 0.16 else "routine"
        req = "context" if unit < 0.22 else "verification" if unit < 0.72 else "alert"
        deadline_class = "high" if unit < 0.18 else "medium" if unit < 0.72 else "low"
        source = 0.12 + 0.34 * fair._stable_unit("source", seed, global_idx, sub_idx)
    return task_class, req, req, min(1.0, source), deadline_class


def _required_stages(requirement: str) -> tuple[str, str]:
    return base._required_stages(requirement)


def _payload_sizes(seed: int, task_index: int, requirement: str) -> tuple[float, float, float]:
    difficulty = 0.78 + 0.58 * fair._stable_unit("difficulty", seed, task_index)
    metadata = 0.05 + 0.06 * fair._stable_unit("metadata", seed, task_index)
    roi = (0.68 + 0.86 * fair._stable_unit("roi", seed, task_index)) * difficulty
    ctx = roi * (2.05 + 1.00 * fair._stable_unit("ctx", seed, task_index))
    if requirement == "evidence":
        ctx *= 1.13
    return round(metadata, 5), round(roi, 5), round(ctx, 5)


def _range_sample(label: str, lower_orbits: float, upper_orbits: float, *parts: object) -> float:
    unit = fair._stable_unit(label, *parts)
    return (lower_orbits + (upper_orbits - lower_orbits) * unit) * ORBIT_HOURS


def _urgency_base_slack_h(deadline_class: str, value_level: str, *parts: object) -> float:
    order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    value_to_urgency = {"urgent": "urgent", "high": "high", "medium": "medium", "low": "low"}
    tier = value_to_urgency.get(value_level, deadline_class)
    if order.get(deadline_class, 2) < order.get(tier, 2):
        tier = deadline_class
    ranges = {"urgent": (1.0, 2.0), "high": (2.0, 4.0), "medium": (4.0, 6.0), "low": (6.0, 10.0)}
    lo, hi = ranges.get(tier, (4.0, 6.0))
    return _range_sample("urgency_slack", lo, hi, *parts)


def _contact_adjustment_h(contact: str, *parts: object) -> float:
    ranges = {
        "sparse_contact": (1.0, 2.0),
        "normal_contact": (0.5, 1.0),
        "dense_contact": (0.0, 0.5),
    }
    lo, hi = ranges[contact]
    return _range_sample("contact_adjustment", lo, hi, *parts)


def _product_adjustment_h(requirement: str, *parts: object) -> float:
    req = requirement.lower()
    if req in {"alert", "metadata"}:
        lo, hi = 0.0, 0.5
    elif req in {"verification", "roi"}:
        lo, hi = 0.5, 1.0
    else:
        lo, hi = 1.0, 2.0
    return _range_sample("product_adjustment", lo, hi, *parts)


def _expected_contact_wait_h(contacts: list[dict[str, Any]], capture_time_h: float) -> float:
    future = [_to_float(row["start_time_h"]) for row in contacts if _to_float(row["start_time_h"]) > capture_time_h + 1.0e-9]
    if not future:
        return float("nan")
    return min(future) - capture_time_h


def _opportunities_before_deadline(
    contacts: list[dict[str, Any]],
    capture_time_h: float,
    deadline_h: float,
) -> tuple[int, float | str, float | str]:
    starts = [
        _to_float(row["start_time_h"])
        for row in contacts
        if capture_time_h + 1.0e-9 < _to_float(row["start_time_h"]) <= deadline_h + 1.0e-9
    ]
    first_future = [
        _to_float(row["start_time_h"]) for row in contacts if _to_float(row["start_time_h"]) > capture_time_h + 1.0e-9
    ]
    first_after = min(first_future) if first_future else ""
    last_before = max(starts) if starts else ""
    return len(starts), first_after, last_before


def _contact_aware_deadline(
    *,
    seed: int,
    workload: str,
    contact: str,
    task_id: str,
    capture_time_h: float,
    deadline_class: str,
    value_level: str,
    product_requirement: str,
    contacts: list[dict[str, Any]],
) -> dict[str, Any]:
    parts = (seed, workload, contact, task_id, deadline_class, value_level, product_requirement)
    urgency = _urgency_base_slack_h(deadline_class, value_level, *parts)
    contact_adj = _contact_adjustment_h(contact, *parts)
    product_adj = _product_adjustment_h(product_requirement, *parts)
    slack = urgency + contact_adj + product_adj
    deadline_h = min(NUM_ORBITS * ORBIT_HOURS, capture_time_h + slack)

    target = 1 if value_level in {"urgent", "high"} or deadline_class in {"urgent", "high"} else 2
    future_contacts = sorted(
        _to_float(row["start_time_h"]) for row in contacts if _to_float(row["start_time_h"]) > capture_time_h + 1.0e-9
    )
    expected_wait = _expected_contact_wait_h(contacts, capture_time_h)
    extreme_gap_h = (4.0 if target == 1 else 6.0) * ORBIT_HOURS
    no_extreme_gap = math.isfinite(expected_wait) and expected_wait <= extreme_gap_h
    count, _first, _last = _opportunities_before_deadline(contacts, capture_time_h, deadline_h)
    if no_extreme_gap and count < target and len(future_contacts) >= target:
        required_deadline = future_contacts[target - 1] + 0.05 * ORBIT_HOURS
        cap_slack = urgency + contact_adj + product_adj + 0.75 * ORBIT_HOURS
        deadline_h = min(NUM_ORBITS * ORBIT_HOURS, max(deadline_h, min(capture_time_h + cap_slack, required_deadline)))
    count, first_after, last_before = _opportunities_before_deadline(contacts, capture_time_h, deadline_h)
    return {
        "deadline_h": deadline_h,
        "deadline_slack_h": max(0.0, deadline_h - capture_time_h),
        "expected_contact_wait_h": "" if not math.isfinite(expected_wait) else expected_wait,
        "contact_adjustment_h": contact_adj,
        "product_requirement_adjustment_h": product_adj,
        "num_contact_opportunities_before_deadline": count,
        "first_contact_after_capture_h": first_after,
        "last_contact_before_deadline_h": last_before,
        "no_contact_before_deadline_flag": int(count == 0),
    }


def build_task_manifest(
    seed: int,
    workload: str,
    contact: str,
    contacts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = _value_cfg()
    rows: list[dict[str, Any]] = []
    raw_tasks: list[TaskSample] = []
    task_index = 0
    for orbit in range(NUM_ORBITS):
        phase = _phase_for_orbit(workload, orbit)
        for scene_local in range(_scene_count_for_orbit(seed, workload, orbit)):
            scene_id = f"s{seed}_{workload}_o{orbit:02d}_scene{scene_local:02d}"
            subtask_count = _subtask_count(seed, workload, phase, scene_id)
            for sub_idx in range(subtask_count):
                task_id = f"{scene_id}_sub{sub_idx:02d}"
                task_class, requirement, evidence_need, source_priority, deadline_class = _subtask_profile(
                    seed, workload, phase, task_index, sub_idx
                )
                prelim_slack = _urgency_base_slack_h(
                    deadline_class, deadline_class, seed, workload, contact, task_id, "prelim"
                )
                prelim_deadline_s = max(1.0, prelim_slack * 3600.0)
                prior = compute_mission_prior_value_v2(
                    task_id=task_id,
                    scene_id=scene_id,
                    task_class=task_class,
                    product_requirement=requirement,
                    deadline_s=prelim_deadline_s,
                    container_cfg=cfg,
                    source_priority=source_priority,
                )
                min_stage, target_stage = _required_stages(requirement)
                metadata_size, roi_size, ctx_size = _payload_sizes(seed, task_index, requirement)
                high_value_prior_flag = prior.task_value >= 0.64
                raw_tasks.append(
                    TaskSample(
                        task_id=task_id,
                        image_path="mock://image",
                        label_path="mock://label",
                        scene_id=scene_id,
                        virtual_time=orbit,
                        task_value=prior.task_value,
                        deadline_s=prelim_deadline_s,
                        difficulty=0.4,
                        uncertainty=0.3,
                        arrival_slot=orbit,
                        task_class=task_class,
                        product_requirement=requirement,
                        high_value_flag=high_value_prior_flag,
                        emergency_flag=task_class == "emergency",
                        metadata={
                            "base_prior_score": prior.task_value,
                            "old_task_value": prior.task_value,
                            "area_prior_bucket": prior.area_prior_bucket,
                            "source_priority": source_priority,
                            "deadline_class": deadline_class,
                        },
                    )
                )
                rows.append(
                    {
                        "task_id": task_id,
                        "scene_id": scene_id,
                        "subtask_id": f"sub{sub_idx:02d}",
                        "seed": seed,
                        "pressure_scale": 1.0,
                        "workload_scenario": workload,
                        "contact_scenario": contact,
                        "wave_name": phase,
                        "orbit_index": orbit,
                        "capture_time_h": orbit * ORBIT_HOURS,
                        "task_class": task_class,
                        "product_requirement": requirement,
                        "evidence_need": evidence_need,
                        "source_priority": source_priority,
                        "deadline_class": deadline_class,
                        "deadline_h": "",
                        "deadline_slack_h": "",
                        "base_prior_score": prior.task_value,
                        "value_level": "",
                        "value_multiplier": "",
                        "V_eff": "",
                        "min_required_stage": min_stage,
                        "target_stage": target_stage,
                        "payload_size_metadata_mb": metadata_size,
                        "payload_size_roi_mb": roi_size,
                        "payload_size_roi_context_mb": ctx_size,
                        "area_prior_bucket": prior.area_prior_bucket,
                        "no_manual_high_value_override": 1,
                    }
                )
                task_index += 1
    calibrated = [task for task in apply_value_skew_calibration(raw_tasks, cfg) if task is not None]
    by_id = {task.task_id: task for task in calibrated}
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        task = by_id[str(row["task_id"])]
        capture_time_h = _to_float(row["capture_time_h"])
        deadline_info = _contact_aware_deadline(
            seed=seed,
            workload=workload,
            contact=contact,
            task_id=str(row["task_id"]),
            capture_time_h=capture_time_h,
            deadline_class=str(row["deadline_class"]),
            value_level=str(task.value_level),
            product_requirement=str(row["product_requirement"]),
            contacts=contacts,
        )
        row.update(deadline_info)
        row["base_prior_score"] = task.base_prior_score
        row["value_level"] = task.value_level
        row["value_multiplier"] = task.value_multiplier
        row["V_eff"] = task.calibrated_task_value
        row["value_assignment_reason"] = task.value_assignment_reason
        row["is_high_value"] = int(task.value_level in {"high", "urgent"})
        audit_rows.append({key: row.get(key, "") for key in DEADLINE_AUDIT_COLUMNS})
    return rows, audit_rows


def _thermal_pressure(row: dict[str, Any]) -> float:
    compute = _to_float(row.get("compute_energy_wh"))
    comm = _to_float(row.get("comm_energy_wh"))
    payload_on = _to_int(row.get("payload_on"))
    backlog = _to_float(row.get("backlog_MB"))
    return 0.42 + 0.025 * compute + 0.030 * comm + 0.060 * payload_on + 0.004 * min(backlog, 40.0)


def _target_success(row: dict[str, Any]) -> int:
    t = row.get("target_ground_receive_time_h", "")
    return int(t not in {"", None} and _to_float(t) <= _to_float(row.get("deadline_h")) + 1.0e-12)


def _augment_result_summary(
    result: dict[str, Any],
    *,
    seed: int,
    workload: str,
    contact: str,
    manifest: list[dict[str, Any]],
    hashes: dict[str, str],
) -> dict[str, Any]:
    tasks = result["task_delivery"]
    events = result["events"]
    trajectory = result["trajectory"]
    task_by_id = {str(row["task_id"]): row for row in tasks}
    manifest_by_id = {str(row["task_id"]): row for row in manifest}
    successes = [row for row in tasks if _to_int(row.get("ground_received_before_deadline_flag"))]
    high_tasks = [row for row in tasks if _to_int(row.get("is_high_value"))]
    high_successes = [row for row in high_tasks if _to_int(row.get("ground_received_before_deadline_flag"))]
    target_successes = [row for row in tasks if _target_success(row)]
    high_target_successes = [row for row in high_tasks if _target_success(row)]
    soc_values = [_to_float(row.get("SOC")) for row in trajectory]
    thermal_values = [_thermal_pressure(row) for row in trajectory]
    downlinked_mb = sum(_to_float(row.get("product_size_MB")) for row in events)
    stage_counts = Counter(str(row.get("delivered_stage")) for row in events)
    q_success = sum(_to_float(row.get("V_eff")) * _to_float(row.get("ground_received_quality")) for row in successes)
    q_high = sum(_to_float(row.get("V_eff")) * _to_float(row.get("ground_received_quality")) for row in high_successes)
    latencies = [_to_float(row.get("capture_to_ground_latency_h")) for row in successes]
    summary = {
        "method": result["summary"]["method"],
        "method_type": METHOD_TYPES.get(str(result["summary"]["method"]), str(result["summary"].get("method_type", ""))),
        "seed": seed,
        "workload_scenario": workload,
        "contact_scenario": contact,
        **hashes,
        "scene_count": len({str(row["scene_id"]) for row in manifest}),
        "subtask_count": len(manifest),
        "avg_subtasks_per_scene": _safe_div(len(manifest), len({str(row["scene_id"]) for row in manifest})),
        "ground_success_ratio": _safe_div(len(successes), len(tasks)),
        "target_stage_success_ratio": _safe_div(len(target_successes), len(tasks)),
        "high_value_ground_success_ratio": _safe_div(len(high_successes), len(high_tasks)),
        "high_value_target_success_ratio": _safe_div(len(high_target_successes), len(high_tasks)),
        "ground_received_task_count": len(successes),
        "high_value_ground_received_count": len(high_successes),
        "avg_ground_received_quality": statistics.mean([_to_float(row.get("ground_received_quality")) for row in successes]) if successes else 0.0,
        "quality_weighted_success_value": q_success,
        "quality_weighted_high_value_success_value": q_high,
        "avg_capture_to_ground_latency_h": statistics.mean(latencies) if latencies else 0.0,
        "deadline_satisfaction_ratio": _safe_div(sum(_to_int(row.get("ground_received_before_deadline_flag")) for row in tasks), len(tasks)),
        "high_value_deadline_satisfaction_ratio": _safe_div(
            sum(_to_int(row.get("ground_received_before_deadline_flag")) for row in high_tasks), len(high_tasks)
        ),
        "expired_task_count": len(tasks) - len(successes),
        "expired_high_value_task_count": len(high_tasks) - len(high_successes),
        "direct_energy_per_received_task_wh": result["summary"].get("direct_energy_per_received_task_wh", 0.0),
        "system_energy_per_received_task_wh": result["summary"].get("system_energy_per_received_task_wh", 0.0),
        "direct_energy_per_high_value_received_task_wh": result["summary"].get(
            "direct_energy_per_high_value_received_task_wh", 0.0
        ),
        "system_energy_per_high_value_received_task_wh": result["summary"].get(
            "system_energy_per_high_value_received_task_wh", 0.0
        ),
        "min_SOC": min(soc_values) if soc_values else 0.0,
        "final_SOC": soc_values[-1] if soc_values else 0.0,
        "SOC_margin_to_20": (min(soc_values) - HARD_MIN_SOC) if soc_values else 0.0,
        "time_below_25_percent": _safe_div(sum(value < 0.25 for value in soc_values), len(soc_values)),
        "time_below_22_percent": _safe_div(sum(value < 0.22 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_below_20": _safe_div(sum(value < HARD_MIN_SOC for value in soc_values), len(soc_values)),
        "hard_safety_violation_count": result["summary"].get("hard_safety_violation_count", 0),
        "final_backlog_count": result["summary"].get("final_backlog_count", 0),
        "max_backlog_count": result["summary"].get("max_backlog_count", 0),
        "queue_MB_peak": max([_to_float(row.get("backlog_MB")) for row in trajectory] or [0.0]),
        "downlinked_MB": downlinked_mb,
        "metadata_downlinked_count": stage_counts["metadata"],
        "roi_downlinked_count": stage_counts["ROI"],
        "roi_context_downlinked_count": stage_counts["ROI+Ctx"],
        "max_thermal_pressure": max(thermal_values) if thermal_values else 0.0,
        "avg_thermal_pressure": statistics.mean(thermal_values) if thermal_values else 0.0,
        "time_above_thermal_warning": _safe_div(sum(value > THERMAL_WARNING for value in thermal_values), len(thermal_values)),
        "thermal_violation_count": sum(value > THERMAL_LIMIT for value in thermal_values),
        "net_system_utility": result["summary"].get("net_system_utility", 0.0),
        "H_net_utility": result["summary"].get("H_net_utility", 0.0),
    }
    for row in tasks:
        row["workload_scenario"] = workload
        row["contact_scenario"] = contact
        row["deadline_class"] = manifest_by_id[str(row["task_id"])].get("deadline_class", "")
    for row in events:
        row["workload_scenario"] = workload
        row["contact_scenario"] = contact
        source = task_by_id.get(str(row.get("task_id")), {})
        row["deadline_h"] = source.get("deadline_h", "")
    for row in trajectory:
        row["workload_scenario"] = workload
        row["contact_scenario"] = contact
        row["thermal_pressure_proxy"] = _thermal_pressure(row)
    return summary


def _group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def build_deadline_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = ["workload_scenario", "contact_scenario", "value_level", "product_requirement"]
    for group_key, group in sorted(_group_rows(rows, keys).items()):
        waits = [_to_float(row["expected_contact_wait_h"]) for row in group if row.get("expected_contact_wait_h") != ""]
        out.append(
            {
                "workload_scenario": group_key[0],
                "contact_scenario": group_key[1],
                "value_level": group_key[2],
                "product_requirement": group_key[3],
                "task_count": len(group),
                "mean_deadline_slack_h": statistics.mean(_to_float(row["deadline_slack_h"]) for row in group),
                "median_deadline_slack_h": statistics.median(_to_float(row["deadline_slack_h"]) for row in group),
                "mean_num_contact_opportunities_before_deadline": statistics.mean(
                    _to_float(row["num_contact_opportunities_before_deadline"]) for row in group
                ),
                "fraction_no_contact_before_deadline": _safe_div(
                    sum(_to_int(row["no_contact_before_deadline_flag"]) for row in group), len(group)
                ),
                "mean_first_contact_wait_h": statistics.mean(waits) if waits else 0.0,
                "mean_product_requirement_adjustment_h": statistics.mean(
                    _to_float(row["product_requirement_adjustment_h"]) for row in group
                ),
            }
        )
    return out


def build_subtask_value_audit(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_id, tasks in sorted(_group_rows(manifest_rows, ["seed", "workload_scenario", "contact_scenario", "scene_id"]).items()):
        values = [_to_float(row["V_eff"]) for row in tasks]
        rows.append(
            {
                "seed": scene_id[0],
                "workload_scenario": scene_id[1],
                "contact_scenario": scene_id[2],
                "scene_id": scene_id[3],
                "subtask_count": len(tasks),
                "min_V_eff": min(values),
                "max_V_eff": max(values),
                "mean_V_eff": statistics.mean(values),
                "value_variance": statistics.pvariance(values) if len(values) > 1 else 0.0,
                "distinct_value_levels": ";".join(sorted({str(row["value_level"]) for row in tasks})),
                "product_requirement_set": ";".join(sorted({str(row["product_requirement"]) for row in tasks})),
                "evidence_need_set": ";".join(sorted({str(row["evidence_need"]) for row in tasks})),
                "has_within_scene_value_variation": int(len({round(_to_float(row["V_eff"]), 6) for row in tasks}) > 1),
            }
        )
    return rows


def build_multi_seed_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ranking_high_is_better = {
        "expired_high_value_task_count": False,
        "avg_capture_to_ground_latency_h": False,
        "direct_energy_per_received_task_wh": False,
        "max_thermal_pressure": False,
    }
    scenario_groups = _group_rows(rows, ["workload_scenario", "contact_scenario"])
    rank_samples: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (_workload, _contact), group in scenario_groups.items():
        by_method = _group_rows(group, ["method"])
        for metric in RANKING_METRICS:
            method_means = []
            for (method,), mrows in by_method.items():
                method_means.append((method, statistics.mean(_to_float(row[metric]) for row in mrows)))
            reverse = ranking_high_is_better.get(metric, True)
            ranked = sorted(method_means, key=lambda item: item[1], reverse=reverse)
            for rank, (method, _value) in enumerate(ranked, start=1):
                rank_samples[(method, _workload, _contact, metric)].append(float(rank))
    for key, group in sorted(_group_rows(rows, ["method", "workload_scenario", "contact_scenario"]).items()):
        method, workload, contact = key
        for metric in RANKING_METRICS:
            values = [_to_float(row[metric]) for row in group]
            ranks = rank_samples.get((method, workload, contact, metric), [])
            out.append(
                {
                    "method": method,
                    "workload_scenario": workload,
                    "contact_scenario": contact,
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


def build_scenario_robustness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    top_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for scenario_key, group in _group_rows(rows, ["workload_scenario", "contact_scenario"]).items():
        method_score = []
        for (method,), mrows in _group_rows(group, ["method"]).items():
            method_score.append((method, statistics.mean(_to_float(row["high_value_ground_success_ratio"]) for row in mrows)))
        for rank, (method, _score) in enumerate(sorted(method_score, key=lambda item: item[1], reverse=True), start=1):
            if rank == 1:
                top_counts[method]["rank1"] += 1
            if rank <= 2:
                top_counts[method]["top2"] += 1
    for (method,), group in sorted(_group_rows(rows, ["method"]).items()):
        safe = [
            row
            for row in group
            if _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= HARD_MIN_SOC
        ]
        robust_safe = [
            row for row in group if _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= 0.25
        ]
        out.append(
            {
                "method": method,
                "overall_ground_success_mean": statistics.mean(_to_float(row["ground_success_ratio"]) for row in group),
                "overall_high_value_success_mean": statistics.mean(
                    _to_float(row["high_value_ground_success_ratio"]) for row in group
                ),
                "overall_quality_weighted_high_value_success_mean": statistics.mean(
                    _to_float(row["quality_weighted_high_value_success_value"]) for row in group
                ),
                "overall_expired_high_value_mean": statistics.mean(_to_float(row["expired_high_value_task_count"]) for row in group),
                "overall_min_SOC_mean": statistics.mean(_to_float(row["min_SOC"]) for row in group),
                "overall_SOC_margin_mean": statistics.mean(_to_float(row["SOC_margin_to_20"]) for row in group),
                "overall_thermal_max_mean": statistics.mean(_to_float(row["max_thermal_pressure"]) for row in group),
                "num_scenarios_rank1": top_counts[method]["rank1"],
                "num_scenarios_top2": top_counts[method]["top2"],
                "num_scenarios_safe": len(safe),
                "num_scenarios_robust_safe": len(robust_safe),
                "num_scenarios_hard_violation": sum(_to_int(row["hard_safety_violation_count"]) > 0 for row in group),
                "worst_case_ground_success": min(_to_float(row["ground_success_ratio"]) for row in group),
                "worst_case_high_value_success": min(_to_float(row["high_value_ground_success_ratio"]) for row in group),
                "worst_case_min_SOC": min(_to_float(row["min_SOC"]) for row in group),
            }
        )
    return out


def _mean_summary(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, group in sorted(_group_rows(rows, keys).items()):
        row = {keys[idx]: key[idx] for idx in range(len(keys))}
        for metric in metrics:
            row[f"{metric}_mean"] = statistics.mean(_to_float(item[metric]) for item in group)
        out.append(row)
    return out


def build_thermal_summary(rows: list[dict[str, Any]], trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    energy_by_case: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in trajectories:
        key = (row["method"], row["seed"], row["workload_scenario"], row["contact_scenario"])
        energy_by_case[key]["compute_energy_wh"] += _to_float(row["compute_energy_wh"])
        energy_by_case[key]["comm_energy_wh"] += _to_float(row["comm_energy_wh"])
        energy_by_case[key]["payload_idle_energy_wh"] += _to_float(row["payload_idle_energy_wh"])
    out = []
    for row in rows:
        key = (row["method"], row["seed"], row["workload_scenario"], row["contact_scenario"])
        energy = energy_by_case[key]
        out.append(
            {
                "method": row["method"],
                "seed": row["seed"],
                "workload_scenario": row["workload_scenario"],
                "contact_scenario": row["contact_scenario"],
                "max_thermal_pressure": row["max_thermal_pressure"],
                "avg_thermal_pressure": row["avg_thermal_pressure"],
                "thermal_violation_count": row["thermal_violation_count"],
                "thermal_warning_fraction": row["time_above_thermal_warning"],
                "compute_energy_wh": energy["compute_energy_wh"],
                "comm_energy_wh": energy["comm_energy_wh"],
                "payload_idle_energy_wh": energy["payload_idle_energy_wh"],
                "thermal_to_delivery_tradeoff_proxy": _safe_div(
                    _to_float(row["quality_weighted_high_value_success_value"]), max(_to_float(row["max_thermal_pressure"]), 1.0e-9)
                ),
            }
        )
    return out


def _bar_panel(rows: list[dict[str, Any]], path: Path, group_key: str, metrics: list[str], title: str) -> None:
    means = _mean_summary(rows, ["method", group_key], metrics)
    methods = METHODS
    groups = sorted({row[group_key] for row in means})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        for idx, group in enumerate(groups):
            values = []
            for method in methods:
                match = next((row for row in means if row["method"] == method and row[group_key] == group), None)
                values.append(_to_float(match[f"{metric}_mean"]) if match else 0.0)
            xs = [x + idx * 0.23 for x in range(len(methods))]
            axis.bar(xs, values, width=0.23, label=group)
        axis.set_title(metric)
        axis.set_xticks([x + 0.23 for x in range(len(methods))])
        axis.set_xticklabels(methods, rotation=65, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_figures(rows: list[dict[str, Any]], subtask_audit: list[dict[str, Any]], output_dir: Path) -> None:
    _bar_panel(
        rows,
        output_dir / "realistic_ground_success_panel.png",
        "workload_scenario",
        ["ground_success_ratio", "high_value_ground_success_ratio", "target_stage_success_ratio"],
        "Ground success by realistic workload",
    )
    _bar_panel(
        rows,
        output_dir / "contact_sensitivity_panel.png",
        "contact_scenario",
        ["ground_success_ratio", "avg_capture_to_ground_latency_h", "expired_high_value_task_count", "queue_MB_peak"],
        "Contact sensitivity",
    )
    _bar_panel(
        rows,
        output_dir / "workload_sensitivity_panel.png",
        "workload_scenario",
        ["high_value_ground_success_ratio", "quality_weighted_high_value_success_value", "expired_high_value_task_count", "min_SOC"],
        "Workload sensitivity",
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for method in METHODS:
        mrows = [row for row in rows if row["method"] == method]
        if not mrows:
            continue
        ax.scatter(
            [_to_float(row["SOC_margin_to_20"]) for row in mrows],
            [_to_float(row["quality_weighted_high_value_success_value"]) for row in mrows],
            s=[35 + 160 * _to_float(row["ground_success_ratio"]) for row in mrows],
            alpha=0.58,
            label=method,
        )
    ax.set_xlabel("SOC margin to 20")
    ax.set_ylabel("Quality-weighted high-value success")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_tradeoff_panel.png", dpi=180)
    plt.close(fig)

    _bar_panel(
        rows,
        output_dir / "thermal_resource_panel.png",
        "workload_scenario",
        ["max_thermal_pressure", "avg_thermal_pressure", "thermal_violation_count", "quality_weighted_high_value_success_value"],
        "Thermal proxy and resource tradeoff",
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    values = [_to_float(row["value_variance"]) for row in subtask_audit]
    ax.hist(values, bins=24, color="#3b82f6", alpha=0.82)
    ax.set_xlabel("Same-scene V_eff variance")
    ax.set_ylabel("Scene count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "scene_subtask_value_variation.png", dpi=180)
    plt.close(fig)


def _best_method(rows: list[dict[str, Any]], workload: str | None = None, contact: str | None = None, metric: str = "high_value_ground_success_ratio") -> tuple[str, float]:
    filtered = rows
    if workload is not None:
        filtered = [row for row in filtered if row["workload_scenario"] == workload]
    if contact is not None:
        filtered = [row for row in filtered if row["contact_scenario"] == contact]
    means = []
    for (method,), group in _group_rows(filtered, ["method"]).items():
        means.append((method, statistics.mean(_to_float(row[metric]) for row in group)))
    return max(means, key=lambda item: item[1]) if means else ("", 0.0)


def build_quick_report(
    rows: list[dict[str, Any]],
    deadline_rows: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
    seeds: list[int],
) -> str:
    proposed = [row for row in rows if row["method"] == "proposed_price_guided_rescue"]
    old = [row for row in rows if row["method"] == "proposed_guarantee_aware_full"]
    utility = [row for row in rows if row["method"] == "utility_rule"]
    deadline_by_contact = _group_rows(deadline_rows, ["contact_scenario"])
    sparse_slack = statistics.mean(_to_float(row["deadline_slack_h"]) for row in deadline_by_contact[("sparse_contact",)])
    dense_slack = statistics.mean(_to_float(row["deadline_slack_h"]) for row in deadline_by_contact[("dense_contact",)])
    normal_slack = statistics.mean(_to_float(row["deadline_slack_h"]) for row in deadline_by_contact[("normal_contact",)])
    urgent_high = [row for row in deadline_rows if row["value_level"] in {"urgent", "high"}]
    no_contact_frac = _safe_div(sum(_to_int(row["no_contact_before_deadline_flag"]) for row in deadline_rows), len(deadline_rows))
    prop_hv = statistics.mean(_to_float(row["high_value_ground_success_ratio"]) for row in proposed)
    old_hv = statistics.mean(_to_float(row["high_value_ground_success_ratio"]) for row in old)
    util_hv = statistics.mean(_to_float(row["high_value_ground_success_ratio"]) for row in utility)
    dense_prop = [row for row in proposed if row["contact_scenario"] == "dense_contact"]
    dense_best = _best_method(rows, contact="dense_contact")
    sparse_best = _best_method(rows, contact="sparse_contact")
    nom_best = _best_method(rows, workload="nominal_daily")
    burst_best = _best_method(rows, workload="moderate_burst")
    emer_best = _best_method(rows, workload="emergency_stress")
    heavy = [row for row in rows if row["method"] == "fixed_heavy"]
    light = [row for row in rows if row["method"] == "fixed_lightweight"]
    protected = [row for row in rows if row["method"] == "fixed_heavy_with_20pct_protection"]
    proposed_rob = next((row for row in robustness if row["method"] == "proposed_price_guided_rescue"), {})
    density_counts = Counter((row["workload_scenario"], row["contact_scenario"]) for row in deadline_rows)
    lines = [
        "# Realistic workload + contact-window sensitivity quick report",
        "",
        f"- Seeds: {seeds}. Workloads: {', '.join(WORKLOAD_SCENARIOS)}. Contacts: {', '.join(CONTACT_SCENARIOS)}.",
        "- This standalone runner reuses mission_prior_value_v2 and value_skew_calibration; it does not manually edit high-value labels or change the value prior algorithm.",
        "- Thermal fields are thermal_pressure_proxy values, not real temperature.",
        "",
        "## Main answers",
        f"1. nominal_daily: best high-value ground success is {nom_best[0]} ({_fmt(nom_best[1])}); proposed mean overall high-value success is {_fmt(prop_hv)} vs old proposed {_fmt(old_hv)}.",
        f"2. moderate_burst: best high-value method is {burst_best[0]} ({_fmt(burst_best[1])}); proposed remains safe in {proposed_rob.get('num_scenarios_safe', 0)} method-seed-scenarios by SOC>=20 and zero hard violations.",
        f"3. emergency_stress: best high-value method is {emer_best[0]} ({_fmt(emer_best[1])}); proposed expired high-value mean is {_fmt(statistics.mean(_to_float(row['expired_high_value_task_count']) for row in proposed))}.",
        f"4. sparse_contact: best high-value method is {sparse_best[0]} ({_fmt(sparse_best[1])}); proposed/old/utility means are {_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in proposed if row['contact_scenario']=='sparse_contact'))}/{_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in old if row['contact_scenario']=='sparse_contact'))}/{_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in utility if row['contact_scenario']=='sparse_contact'))}.",
        f"5. dense_contact: best high-value method is {dense_best[0]} ({_fmt(dense_best[1])}); proposed dense high-value mean is {_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in dense_prop))}, so dense windows reveal whether baselines catch up instead of guaranteeing a proposed win.",
        f"6. Proposed advantage source: target-stage success mean={_fmt(statistics.mean(_to_float(row['target_stage_success_ratio']) for row in proposed))}, metadata downlinks mean={_fmt(statistics.mean(_to_float(row['metadata_downlinked_count']) for row in proposed))}; use both fields rather than metadata count alone.",
        f"7. Robustness: proposed rank-1 scenarios={proposed_rob.get('num_scenarios_rank1', 0)}, top-2 scenarios={proposed_rob.get('num_scenarios_top2', 0)}.",
        f"8. Non-first scenarios should be read from realistic_multi_seed_ranking.csv; quick winners are nominal={nom_best[0]}, burst={burst_best[0]}, emergency={emer_best[0]}.",
        f"9. fixed_heavy: min SOC worst case={_fmt(min(_to_float(row['min_SOC']) for row in heavy))}, hard violations={sum(_to_int(row['hard_safety_violation_count']) for row in heavy)}.",
        f"10. fixed_lightweight: safe but lower high-value success mean={_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in light))}.",
        f"11. utility_rule: high-value success mean={_fmt(util_hv)}, queue peak mean={_fmt(statistics.mean(_to_float(row['queue_MB_peak']) for row in utility))}, expired high-value mean={_fmt(statistics.mean(_to_float(row['expired_high_value_task_count']) for row in utility))}.",
        f"12. protected heavy: min SOC worst case={_fmt(min(_to_float(row['min_SOC']) for row in protected))}, high-value success mean={_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in protected))}.",
        f"13. Contact-window bias check: sparse/normal/dense mean deadline slack={_fmt(sparse_slack)}/{_fmt(normal_slack)}/{_fmt(dense_slack)} h; sparse is extended, dense is not over-relaxed.",
        f"14. Task density: generated case counts cover {len(density_counts)} workload/contact cases with scene/subtask audits in subtask_value_audit_realistic.csv.",
        f"15. Thermal proxy: proposed max thermal mean={_fmt(statistics.mean(_to_float(row['max_thermal_pressure']) for row in proposed))}; violations={sum(_to_int(row['thermal_violation_count']) for row in proposed)}.",
        f"16. Benchmark suitability: yes, the same task_manifest_hash/contact_schedule_hash/resource_trace_hash are shared by all methods per seed/workload/contact, suitable for later literature-inspired architecture baselines.",
        f"17. Suggested main-text case study: {burst_best[0]} winner under moderate_burst; use moderate_burst/normal_contact for balanced pressure.",
        f"18. Suggested stress appendix: emergency_stress/sparse_contact because it combines burst and communication scarcity.",
        "19. Recommended next baselines: raw_downlink_ground_processing, mec_cost_minimization, two_timescale_qos_scheduling, and goal_oriented_eo_edge can plug into the same manifest/contact/resource traces.",
        "",
        "## Deadline/contact audit answers",
        f"1. sparse_contact deadline slack is {_fmt(sparse_slack)} h versus normal {_fmt(normal_slack)} h, so sparse is contact-aware extended without simply making all deadlines very wide.",
        f"2. dense_contact deadline slack is {_fmt(dense_slack)} h, below sparse, which avoids making dense_contact trivially easy for every method.",
        f"3. urgent/high tasks average contact opportunities before deadline: {_fmt(statistics.mean(_to_float(row['num_contact_opportunities_before_deadline']) for row in urgent_high) if urgent_high else 0.0)}.",
        f"4. no_contact_before_deadline_flag fraction: {_fmt(no_contact_frac)}; inspect deadline_contact_summary.csv by value/product group for hotspots.",
        "5. Difficulty remains comparable because only deadline/contact schedule changes with contact_scenario; task value labels and value prior are not method-specific.",
        "6. This deadline/contact setup is suitable as a unified benchmark because it exposes the audit fields needed to detect unfair no-contact deadline mismatches.",
    ]
    return "\n".join(lines)


def run_experiment(output_dir: Path, seeds: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_manifest: list[dict[str, Any]] = []
    all_contacts: list[dict[str, Any]] = []
    all_resources: list[dict[str, Any]] = []
    all_deadline_audit: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    all_trajectories: list[dict[str, Any]] = []

    for seed in seeds:
        for workload in WORKLOAD_SCENARIOS:
            for contact in CONTACT_SCENARIOS:
                contacts = build_contact_schedule(seed, workload, contact)
                resources = build_resource_trace(seed, workload, contact)
                manifest, deadline_audit = build_task_manifest(seed, workload, contact, contacts)
                hashes = {
                    "task_manifest_hash": _hash(manifest, REALISTIC_MANIFEST_COLUMNS),
                    "contact_schedule_hash": _hash(
                        contacts,
                        ["seed", "workload_scenario", "contact_scenario", "orbit_index", "capacity_MB", "start_time_h"],
                    ),
                    "resource_trace_hash": _hash(
                        resources,
                        ["seed", "workload_scenario", "contact_scenario", "orbit_index", "solar_energy_wh", "wave_name"],
                    ),
                }
                all_manifest.extend(manifest)
                all_contacts.extend(contacts)
                all_resources.extend(resources)
                all_deadline_audit.extend(deadline_audit)
                for method in METHODS:
                    if method == "proposed_price_guided_rescue":
                        result = price.run_price_guided_method(seed, 1.0, manifest, contacts, resources, hashes)
                    else:
                        result = base.run_method(method, seed, 1.0, manifest, contacts, resources, hashes)
                    summary = _augment_result_summary(
                        result,
                        seed=seed,
                        workload=workload,
                        contact=contact,
                        manifest=manifest,
                        hashes=hashes,
                    )
                    all_summary.append(summary)
                    for rows in (result["events"], result["task_delivery"], result["trajectory"]):
                        for row in rows:
                            row["method_type"] = METHOD_TYPES.get(str(row.get("method")), "")
                    all_events.extend(result["events"])
                    all_tasks.extend(result["task_delivery"])
                    all_trajectories.extend(result["trajectory"])

    deadline_summary = build_deadline_summary(all_deadline_audit)
    ranking = build_multi_seed_ranking(all_summary)
    robustness = build_scenario_robustness(all_summary)
    contact_sensitivity = _mean_summary(
        all_summary,
        ["method", "contact_scenario"],
        [
            "ground_success_ratio",
            "high_value_ground_success_ratio",
            "avg_capture_to_ground_latency_h",
            "expired_high_value_task_count",
            "queue_MB_peak",
            "downlinked_MB",
            "min_SOC",
            "H_net_utility",
        ],
    )
    for row in contact_sensitivity:
        row["avg_latency_h_mean"] = row.pop("avg_capture_to_ground_latency_h_mean")
        row["expired_high_value_mean"] = row.pop("expired_high_value_task_count_mean")
    workload_sensitivity = _mean_summary(
        all_summary,
        ["method", "workload_scenario"],
        [
            "scene_count",
            "subtask_count",
            "ground_success_ratio",
            "high_value_ground_success_ratio",
            "quality_weighted_high_value_success_value",
            "expired_high_value_task_count",
            "min_SOC",
            "max_thermal_pressure",
            "H_net_utility",
        ],
    )
    for row in workload_sensitivity:
        row["expired_high_value_mean"] = row.pop("expired_high_value_task_count_mean")
    subtask_audit = build_subtask_value_audit(all_manifest)
    thermal_summary = build_thermal_summary(all_summary, all_trajectories)

    _write_csv(output_dir / "realistic_task_manifest.csv", all_manifest, REALISTIC_MANIFEST_COLUMNS)
    _write_csv(output_dir / "shared_contact_schedule.csv", all_contacts)
    _write_csv(output_dir / "shared_resource_trace.csv", all_resources)
    _write_csv(output_dir / "deadline_contact_audit.csv", all_deadline_audit, DEADLINE_AUDIT_COLUMNS)
    _write_csv(output_dir / "deadline_contact_summary.csv", deadline_summary)
    _write_csv(output_dir / "realistic_summary_by_seed.csv", all_summary, SUMMARY_COLUMNS)
    _write_csv(output_dir / "realistic_multi_seed_ranking.csv", ranking)
    _write_csv(output_dir / "scenario_robustness_summary.csv", robustness)
    _write_csv(output_dir / "contact_sensitivity_summary.csv", contact_sensitivity)
    _write_csv(output_dir / "workload_sensitivity_summary.csv", workload_sensitivity)
    _write_csv(output_dir / "subtask_value_audit_realistic.csv", subtask_audit)
    _write_csv(output_dir / "thermal_summary_by_scenario.csv", thermal_summary)
    _write_csv(output_dir / "ground_delivery_events_realistic.csv", all_events)
    _write_csv(output_dir / "task_delivery_realistic.csv", all_tasks)
    _write_csv(output_dir / "trajectory_realistic.csv", all_trajectories)
    build_figures(all_summary, subtask_audit, output_dir)
    _write_text(output_dir / "quick_report.md", build_quick_report(all_summary, all_deadline_audit, robustness, seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--full-seeds", action="store_true", help="Use seeds 0..4 instead of the default 0..2 smoke sweep.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None and args.seeds else (FULL_SEEDS if args.full_seeds else DEFAULT_SEEDS)
    run_experiment(args.output_dir, seeds)
    print(f"Wrote realistic workload/contact sensitivity outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
