"""Uneven multi-wave stress plus ground-received delivery experiment.

Standalone, default-off experiment runner. It builds a local subtask manifest
with mission_prior_value_v2 plus value_skew_calibration, evaluates all methods
on a shared manifest/contact/resource trace per seed and pressure scale, and
reports ground-received task delivery metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_fair_baseline_comparison as fair  # noqa: E402
from src.common.types import TaskSample  # noqa: E402
from src.data.mission_prior_value import compute_mission_prior_value_v2  # noqa: E402
from src.data.value_skew_calibration import apply_value_skew_calibration  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_uneven_pressure_ground_delivery_v1"
SCENARIO = "uneven_pressure_ground_delivery_v1"
SEEDS = [0, 1, 2, 3, 4]
PRESSURE_SCALES = [0.85, 1.00, 1.15]
METHODS = [
    "proposed_guarantee_aware_full",
    "utility_rule",
    "persistent_queue_without_guarantee",
    "fixed_lightweight",
    "fixed_heavy",
    "fixed_heavy_with_20pct_protection",
    "ground_approved_followup",
]
METHOD_TYPES = {
    "proposed_guarantee_aware_full": "proposed",
    "utility_rule": "myopic_greedy",
    "persistent_queue_without_guarantee": "internal_ablation",
    "fixed_lightweight": "fixed_conservative",
    "fixed_heavy": "unsafe_stress_baseline",
    "fixed_heavy_with_20pct_protection": "hard_protection_baseline",
    "ground_approved_followup": "ground_followup_baseline",
}

NUM_ORBITS = 60
ORBIT_HOURS = 96.0 / NUM_ORBITS
BATTERY_CAPACITY_WH = 230.0
INITIAL_SOC = 0.86
HARD_MIN_SOC = 0.20
PROTECTION_RESUME_SOC = 0.35
HOUSEKEEPING_WH_PER_ORBIT = 12.0
TX_FIXED_WH_PER_ACTIVE_ORBIT = 3.2

WAVES = [
    ("warmup", 0, 2, 0.55, "low"),
    ("wave_1_moderate", 3, 10, 0.32, "medium_high"),
    ("recovery_1", 11, 16, 0.08, "low"),
    ("wave_2_extreme", 17, 32, 0.82, "extreme"),
    ("recovery_2_incomplete", 33, 39, 0.05, "low"),
    ("wave_3_medium_high", 40, 50, 0.98, "high"),
    ("final_recovery", 51, 59, 0.03, "low"),
]

GROUND_MANIFEST_COLUMNS = [
    "task_id",
    "scene_id",
    "subtask_id",
    "seed",
    "pressure_scale",
    "wave_name",
    "orbit_index",
    "capture_time_h",
    "task_class",
    "product_requirement",
    "evidence_need",
    "source_priority",
    "deadline_h",
    "base_prior_score",
    "value_level",
    "value_multiplier",
    "V_eff",
    "min_required_stage",
    "target_stage",
]
EVENT_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "task_id",
    "scene_id",
    "delivered_stage",
    "model_choice",
    "product_size_MB",
    "ground_receive_time_h",
    "before_deadline_flag",
    "delivered_quality",
    "incremental_information_utility",
]
SUMMARY_COLUMNS = [
    "method",
    "method_type",
    "seed",
    "pressure_scale",
    "task_manifest_hash",
    "contact_schedule_hash",
    "resource_trace_hash",
    "total_task_count",
    "ground_received_task_count",
    "ground_received_task_ratio",
    "avg_ground_received_quality",
    "avg_capture_to_ground_latency_h",
    "deadline_satisfaction_ratio",
    "high_value_task_count",
    "high_value_ground_received_count",
    "high_value_ground_received_ratio",
    "high_value_target_stage_success_rate",
    "avg_high_value_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "direct_energy_per_high_value_received_task_wh",
    "system_energy_per_high_value_received_task_wh",
    "net_system_utility",
    "H_net_utility",
    "min_SOC",
    "final_SOC",
    "fraction_time_SOC_below_50",
    "fraction_time_SOC_below_20",
    "hard_safety_violation_count",
    "final_backlog_count",
    "max_backlog_count",
    "protection_trigger_count",
    "protection_total_slots",
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


def _wave_for_orbit(orbit: int) -> tuple[str, int, int, float, str]:
    for name, start, end, density, pressure in WAVES:
        if start <= orbit <= end:
            return name, start, end, density, pressure
    return WAVES[-1]


def _soc_band(soc: float) -> str:
    if soc >= 0.70:
        return "comfortable"
    if soc >= 0.50:
        return "soft_protection"
    if soc >= 0.30:
        return "energy_tight"
    if soc >= 0.20:
        return "protection"
    return "hard_violation"


def _stage_rank(stage: str) -> int:
    return {"none": 0, "metadata": 1, "ROI": 2, "ROI+Ctx": 3, "raw": 3, "compact": 2}.get(stage, 0)


def _stage_product(stage: str) -> str:
    return {"metadata": "metadata", "ROI": "roi", "ROI+Ctx": "roi_context", "raw": "roi_context", "compact": "roi"}.get(stage, "metadata")


def _highest_stage(products: set[str]) -> str:
    if "roi_context" in products:
        return "ROI+Ctx"
    if "roi" in products:
        return "ROI"
    if "metadata" in products:
        return "metadata"
    return "none"


def _required_stages(requirement: str) -> tuple[str, str]:
    req = str(requirement).lower()
    if req in {"alert", "metadata"}:
        return "metadata", "metadata"
    if req in {"verification", "confirmation"}:
        return "ROI", "ROI"
    return "ROI", "ROI+Ctx"


def _stage_sufficiency(requirement: str, delivered_stage: str) -> float:
    min_stage, target = _required_stages(requirement)
    if _stage_rank(delivered_stage) >= _stage_rank(target):
        return 1.0
    if _stage_rank(delivered_stage) >= _stage_rank(min_stage):
        return 0.72
    if delivered_stage == "metadata":
        return 0.42
    return 0.0


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


def build_contact_schedule(seed: int, pressure_scale: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in range(NUM_ORBITS):
        wave, _start, _end, _density, pressure = _wave_for_orbit(orbit)
        base = {
            "warmup": 5.0,
            "wave_1_moderate": 5.4,
            "recovery_1": 4.2,
            "wave_2_extreme": 4.7,
            "recovery_2_incomplete": 3.7,
            "wave_3_medium_high": 5.0,
            "final_recovery": 4.0,
        }[wave]
        variation = 0.92 + 0.16 * fair._stable_unit("contact", seed, pressure_scale, orbit)
        rows.append(
            {
                "seed": seed,
                "pressure_scale": pressure_scale,
                "scenario": SCENARIO,
                "contact_window_id": f"ug_cw_s{seed}_p{pressure_scale:.2f}_{orbit:03d}",
                "orbit_index": orbit,
                "start_slot": orbit,
                "end_slot": orbit,
                "start_time_h": orbit * ORBIT_HOURS,
                "end_time_h": (orbit + 1) * ORBIT_HOURS,
                "capacity_MB": round(base * variation, 5),
                "deadline_pressure": pressure,
                "wave_name": wave,
            }
        )
    return rows


def build_resource_trace(seed: int, pressure_scale: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in range(NUM_ORBITS):
        wave, _start, _end, _density, pressure = _wave_for_orbit(orbit)
        solar = {
            "warmup": 28.0,
            "wave_1_moderate": 35.2,
            "recovery_1": 26.7,
            "wave_2_extreme": 17.4,
            "recovery_2_incomplete": 28.5,
            "wave_3_medium_high": 27.1,
            "final_recovery": 34.2,
        }[wave]
        solar *= 0.96 + 0.08 * fair._stable_unit("solar", seed, pressure_scale, orbit)
        rows.append(
            {
                "seed": seed,
                "pressure_scale": pressure_scale,
                "scenario": SCENARIO,
                "orbit_index": orbit,
                "time_h": orbit * ORBIT_HOURS,
                "wave_name": wave,
                "deadline_pressure": pressure,
                "solar_energy_wh": round(solar, 5),
                "housekeeping_energy_wh": HOUSEKEEPING_WH_PER_ORBIT,
            }
        )
    return rows


def _scene_count_for_orbit(seed: int, pressure_scale: float, orbit: int) -> int:
    wave, _start, _end, density, _pressure = _wave_for_orbit(orbit)
    scaled = density * pressure_scale
    base = int(math.floor(scaled))
    frac = scaled - base
    add = 1 if fair._stable_unit("scene_count", seed, pressure_scale, orbit) < frac else 0
    if wave.startswith("recovery") or wave == "final_recovery":
        return add
    return max(1, base + add)


def _subtask_count(seed: int, scene_id: str, wave: str) -> int:
    unit = fair._stable_unit("subtask_count", seed, scene_id)
    if wave == "wave_2_extreme":
        return 3 + int(unit > 0.25) + int(unit > 0.70)
    if wave == "wave_3_medium_high":
        return 2 + int(unit > 0.35) + int(unit > 0.80)
    return 2 + int(unit > 0.55)


def _subtask_profile(seed: int, pressure_scale: float, wave: str, global_idx: int, sub_idx: int) -> tuple[str, str, str, float]:
    unit = fair._stable_unit("profile", seed, pressure_scale, wave, global_idx, sub_idx)
    if wave == "wave_2_extreme":
        task_class = "emergency" if unit < 0.18 else "high_value" if unit < 0.68 else "routine"
        req = "evidence" if unit < 0.48 else "context" if unit < 0.78 else "verification"
        source = min(1.0, 0.52 + 0.45 * fair._stable_unit("src", seed, global_idx, sub_idx))
    elif wave == "wave_3_medium_high":
        task_class = "high_value" if unit < 0.45 else "routine"
        req = "context" if unit < 0.48 else "evidence" if unit < 0.66 else "verification"
        source = min(1.0, 0.36 + 0.42 * fair._stable_unit("src", seed, global_idx, sub_idx))
    elif wave == "wave_1_moderate":
        task_class = "high_value" if unit < 0.24 else "routine"
        req = "context" if unit < 0.34 else "verification" if unit < 0.78 else "alert"
        source = 0.18 + 0.35 * fair._stable_unit("src", seed, global_idx, sub_idx)
    else:
        task_class = "routine"
        req = "verification" if unit < 0.35 else "alert"
        source = 0.08 + 0.20 * fair._stable_unit("src", seed, global_idx, sub_idx)
    evidence_need = req
    return task_class, req, evidence_need, source


def _deadline_orbits(wave: str, requirement: str, value_unit: float) -> int:
    if wave == "wave_2_extreme":
        base = 3 if requirement in {"context", "evidence"} else 2
    elif wave == "wave_3_medium_high":
        base = 4 if requirement in {"context", "evidence"} else 3
    elif wave == "wave_1_moderate":
        base = 5 if requirement in {"context", "evidence"} else 4
    elif "recovery" in wave:
        base = 6
    else:
        base = 5
    return base + int(value_unit > 0.82)


def _payload_sizes(seed: int, task_index: int, requirement: str) -> tuple[float, float, float]:
    difficulty = 0.80 + 0.55 * fair._stable_unit("difficulty", seed, task_index)
    metadata = 0.06 + 0.05 * fair._stable_unit("metadata", seed, task_index)
    roi = (0.72 + 0.80 * fair._stable_unit("roi", seed, task_index)) * difficulty
    ctx = roi * (2.2 + 0.95 * fair._stable_unit("ctx", seed, task_index))
    if requirement == "evidence":
        ctx *= 1.12
    return round(metadata, 5), round(roi, 5), round(ctx, 5)


def build_task_manifest(seed: int, pressure_scale: float, contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = _value_cfg()
    rows: list[dict[str, Any]] = []
    raw_tasks: list[TaskSample] = []
    task_index = 0
    for orbit in range(NUM_ORBITS):
        wave, _start, _end, _density, _pressure = _wave_for_orbit(orbit)
        for scene_local in range(_scene_count_for_orbit(seed, pressure_scale, orbit)):
            scene_id = f"s{seed}_p{pressure_scale:.2f}_o{orbit:02d}_scene{scene_local:02d}"
            for sub_idx in range(_subtask_count(seed, scene_id, wave)):
                task_id = f"{scene_id}_sub{sub_idx:02d}"
                task_class, requirement, evidence_need, source_priority = _subtask_profile(
                    seed, pressure_scale, wave, task_index, sub_idx
                )
                deadline_orbits = _deadline_orbits(wave, requirement, fair._stable_unit("deadline", seed, task_index))
                deadline_h = min(96.0, (orbit + deadline_orbits + 1) * ORBIT_HOURS)
                deadline_s = max(1.0, (deadline_h - orbit * ORBIT_HOURS) * 3600.0)
                prior = compute_mission_prior_value_v2(
                    task_id=task_id,
                    scene_id=scene_id,
                    task_class=task_class,
                    product_requirement=requirement,
                    deadline_s=deadline_s,
                    container_cfg=cfg,
                    source_priority=source_priority,
                )
                min_stage, target_stage = _required_stages(requirement)
                metadata_size, roi_size, ctx_size = _payload_sizes(seed, task_index, requirement)
                raw_tasks.append(
                    TaskSample(
                        task_id=task_id,
                        image_path="mock://image",
                        label_path="mock://label",
                        scene_id=scene_id,
                        virtual_time=orbit,
                        task_value=prior.task_value,
                        deadline_s=deadline_s,
                        difficulty=0.4,
                        uncertainty=0.3,
                        arrival_slot=orbit,
                        task_class=task_class,
                        product_requirement=requirement,
                        high_value_flag=prior.task_value >= 0.64,
                        emergency_flag=task_class == "emergency",
                        metadata={
                            "base_prior_score": prior.task_value,
                            "old_task_value": prior.task_value,
                            "area_prior_bucket": prior.area_prior_bucket,
                            "source_priority": source_priority,
                        },
                    )
                )
                rows.append(
                    {
                        "task_id": task_id,
                        "scene_id": scene_id,
                        "subtask_id": f"sub{sub_idx:02d}",
                        "seed": seed,
                        "pressure_scale": pressure_scale,
                        "wave_name": wave,
                        "orbit_index": orbit,
                        "capture_time_h": orbit * ORBIT_HOURS,
                        "task_class": task_class,
                        "product_requirement": requirement,
                        "evidence_need": evidence_need,
                        "source_priority": source_priority,
                        "deadline_h": deadline_h,
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
    for row in rows:
        task = by_id[str(row["task_id"])]
        row["base_prior_score"] = task.base_prior_score
        row["value_level"] = task.value_level
        row["value_multiplier"] = task.value_multiplier
        row["V_eff"] = task.calibrated_task_value
        row["value_assignment_reason"] = task.value_assignment_reason
        row["is_high_value"] = int(task.value_level in {"high", "urgent"})
    return rows


def _model_quality(method: str, product_type: str) -> float:
    if method == "fixed_lightweight":
        q = {"metadata": 0.76, "roi": 0.70, "roi_context": 0.63}
    elif method in {"fixed_heavy", "fixed_heavy_with_20pct_protection"}:
        q = {"metadata": 0.93, "roi": 0.92, "roi_context": 0.91}
    elif method == "utility_rule":
        q = {"metadata": 0.84, "roi": 0.82, "roi_context": 0.78}
    elif method == "ground_approved_followup":
        q = {"metadata": 0.86, "roi": 0.84, "roi_context": 0.82}
    elif method == "persistent_queue_without_guarantee":
        q = {"metadata": 0.88, "roi": 0.86, "roi_context": 0.84}
    else:
        q = {"metadata": 0.90, "roi": 0.90, "roi_context": 0.88}
    return q.get(product_type, 0.75)


def _compute_multiplier(method: str) -> float:
    return {
        "fixed_lightweight": 0.42,
        "fixed_heavy": 2.8,
        "fixed_heavy_with_20pct_protection": 2.8,
        "utility_rule": 1.1,
        "ground_approved_followup": 0.95,
        "persistent_queue_without_guarantee": 1.05,
        "proposed_guarantee_aware_full": 1.12,
    }[method]


def _product_payload(task: dict[str, Any], product_type: str) -> float:
    if product_type == "metadata":
        return _to_float(task["payload_size_metadata_mb"])
    if product_type == "roi":
        return _to_float(task["payload_size_roi_mb"])
    return _to_float(task["payload_size_roi_context_mb"])


def _product_costs(method: str, product_type: str, payload_mb: float) -> tuple[float, float, float]:
    base = {"metadata": 1.05, "roi": 5.6, "roi_context": 13.4}[product_type]
    compute = base * _compute_multiplier(method)
    comm = 0.20 * payload_mb
    account = (0.016 if method == "proposed_guarantee_aware_full" else 0.034) * compute + 0.30 * comm + 0.012 * payload_mb
    return compute, comm, account


def _payload_idle_wh(method: str, payload_on: bool) -> float:
    if not payload_on:
        return 0.0
    return {
        "fixed_heavy": 13.5,
        "fixed_heavy_with_20pct_protection": 13.5,
        "fixed_lightweight": 3.0,
        "ground_approved_followup": 5.2,
        "utility_rule": 6.5,
        "persistent_queue_without_guarantee": 6.1,
        "proposed_guarantee_aware_full": 5.8,
    }[method]


def _next_product(task: dict[str, Any], delivered: set[str], generated: set[str], method: str, current_orbit: int) -> str | None:
    target = str(task["target_stage"])
    if "metadata" not in delivered and "metadata" not in generated:
        return "metadata"
    if _stage_rank(target) >= _stage_rank("ROI") and "metadata" in delivered and "roi" not in delivered and "roi" not in generated:
        if method == "ground_approved_followup" and current_orbit < _to_int(task["orbit_index"]) + 2:
            return None
        return "roi"
    if _stage_rank(target) >= _stage_rank("ROI+Ctx") and "roi" in delivered and "roi_context" not in delivered and "roi_context" not in generated:
        if method == "ground_approved_followup" and current_orbit < _to_int(task["orbit_index"]) + 3:
            return None
        return "roi_context"
    return None


def _request_score(method: str, req: dict[str, Any], task: dict[str, Any], delivered: set[str], orbit: int, soc: float) -> float:
    product = str(req["product_type"])
    payload = max(1.0e-9, _to_float(req["payload_size_MB"]))
    delta = fair.delta_information_utility(product, delivered, str(task["product_requirement"]))
    quality = _model_quality(method, product)
    deadline_left = max(1.0, (_to_float(task["deadline_h"]) - orbit * ORBIT_HOURS) / ORBIT_HOURS)
    gross = _to_float(task["V_eff"]) * delta * quality
    urgency = 1.0 / deadline_left
    score = gross / payload + 0.22 * urgency + 0.01 * _to_int(req.get("age", 0))
    if method == "proposed_guarantee_aware_full" and str(task["value_level"]) in {"high", "urgent"}:
        score += 1.00 + 0.50 * urgency
    if method == "utility_rule":
        score += 0.18 * gross
    if method == "fixed_lightweight" and product == "roi_context":
        score -= 0.45
    if soc < 0.50 and product == "roi_context" and method == "proposed_guarantee_aware_full":
        score -= 0.65
    return score


def _selection_limit(method: str, soc: float, wave: str) -> int:
    if "recovery" in wave or wave == "final_recovery":
        if method == "proposed_guarantee_aware_full":
            return 1 if soc > 0.74 else 0
        if method == "utility_rule":
            return 2 if soc > 0.40 else 1
        return 1 if soc > 0.55 else 0
    if method == "fixed_heavy":
        return 8
    if method == "fixed_heavy_with_20pct_protection":
        return 7
    if method == "utility_rule":
        return 7 if soc > 0.20 else 0
    if method == "persistent_queue_without_guarantee":
        return 5 if soc >= 0.50 else 3 if soc >= 0.20 else 0
    if method == "fixed_lightweight":
        return 3 if soc >= 0.20 else 0
    if method == "ground_approved_followup":
        return 4 if soc >= 0.55 else 1 if soc >= 0.20 else 0
    if soc >= 0.70:
        return 6
    if soc >= 0.50:
        return 2
    if soc >= 0.30:
        return 1 if soc >= 0.40 else 0
    return 1 if soc >= 0.20 else 0


def _allowed(method: str, task: dict[str, Any], product: str, soc: float, wave: str) -> bool:
    if method == "fixed_heavy":
        return True
    if soc <= HARD_MIN_SOC:
        return False
    if method != "proposed_guarantee_aware_full":
        return True
    if "recovery" in wave and soc < 0.74:
        return False
    if soc < 0.40:
        return False
    if soc < 0.45:
        return str(task["value_level"]) in {"high", "urgent"} and product in {"metadata", "roi"}
    if soc < 0.52:
        return product == "metadata"
    if soc < 0.50:
        return product == "metadata" or (product == "roi" and str(task["value_level"]) in {"high", "urgent"})
    if soc < 0.70:
        return product != "roi_context" or (str(task["value_level"]) in {"high", "urgent"} and soc >= 0.52)
    return True


def run_method(
    method: str,
    seed: int,
    pressure_scale: float,
    manifest: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    hashes: dict[str, str],
) -> dict[str, Any]:
    by_task = {str(row["task_id"]): row for row in manifest}
    arrivals: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in manifest:
        arrivals[_to_int(task["orbit_index"])].append(task)
    contact_by_orbit = {_to_int(row["orbit_index"]): row for row in contacts}
    resource_by_orbit = {_to_int(row["orbit_index"]): row for row in resources}
    delivered: dict[str, set[str]] = defaultdict(set)
    generated: dict[str, set[str]] = defaultdict(set)
    pending: list[dict[str, Any]] = []
    active: set[str] = set()
    events: list[dict[str, Any]] = []
    task_rows: dict[str, dict[str, Any]] = {}
    trajectory: list[dict[str, Any]] = []
    cumulative_direct_energy = 0.0
    cumulative_system_energy = 0.0
    net_system_utility = 0.0
    h_net_utility = 0.0
    soc = INITIAL_SOC
    protection = False
    protection_triggers = 0
    protection_slots = 0

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

    for orbit in range(NUM_ORBITS):
        wave = str(resource_by_orbit[orbit]["wave_name"])
        if method == "fixed_heavy_with_20pct_protection":
            if not protection and soc <= HARD_MIN_SOC + 1.0e-12:
                protection = True
                protection_triggers += 1
            elif protection and soc >= PROTECTION_RESUME_SOC:
                protection = False
        for task in arrivals.get(orbit, []):
            active.add(str(task["task_id"]))
        for task_id in sorted(active):
            task = by_task[task_id]
            if orbit * ORBIT_HOURS > _to_float(task["deadline_h"]):
                continue
            product = _next_product(task, delivered[task_id], generated[task_id], method, orbit)
            if product is None:
                continue
            pending.append(
                {
                    "task_id": task_id,
                    "product_type": product,
                    "payload_size_MB": _product_payload(task, product),
                    "created_orbit": orbit,
                    "age": 0,
                }
            )
            generated[task_id].add(product)
        ready = [
            req
            for req in pending
            if orbit * ORBIT_HOURS <= _to_float(by_task[str(req["task_id"])]["deadline_h"]) + ORBIT_HOURS
        ]
        for req in ready:
            req["age"] = orbit - _to_int(req["created_orbit"])
        selected: list[dict[str, Any]] = []
        if not protection and orbit in contact_by_orbit:
            capacity = _to_float(contact_by_orbit[orbit]["capacity_MB"])
            candidates = [
                req
                for req in ready
                if _allowed(method, by_task[str(req["task_id"])], str(req["product_type"]), soc, wave)
            ]
            scored = sorted(
                candidates,
                key=lambda req: -_request_score(method, req, by_task[str(req["task_id"])], delivered[str(req["task_id"])], orbit, soc),
            )
            remain = capacity
            for req in scored:
                if len(selected) >= _selection_limit(method, soc, wave):
                    break
                payload = _to_float(req["payload_size_MB"])
                if payload <= remain + 1.0e-12:
                    selected.append(req)
                    remain -= payload
        selected_keys = {(req["task_id"], req["product_type"]) for req in selected}
        orbit_compute = 0.0
        orbit_comm = 0.0
        product_counts = Counter()
        for req in selected:
            task = by_task[str(req["task_id"])]
            product = str(req["product_type"])
            stage = {"metadata": "metadata", "roi": "ROI", "roi_context": "ROI+Ctx"}[product]
            payload = _to_float(req["payload_size_MB"])
            compute, comm, account_cost = _product_costs(method, product, payload)
            orbit_compute += compute
            orbit_comm += comm
            cumulative_direct_energy += compute + comm
            product_counts[product] += 1
            delivered_before = set(delivered[str(task["task_id"])])
            delta = fair.delta_information_utility(product, delivered_before, str(task["product_requirement"]))
            delivered[str(task["task_id"])].add(product)
            highest = _highest_stage(delivered[str(task["task_id"])])
            q = _model_quality(method, product) * _stage_sufficiency(str(task["product_requirement"]), highest)
            gross = _to_float(task["V_eff"]) * delta * q
            net = gross - account_cost
            net_system_utility += net
            if _to_int(task["is_high_value"]):
                h_net_utility += net
            receive_time = (orbit + 1) * ORBIT_HOURS
            before_deadline = int(receive_time <= _to_float(task["deadline_h"]) + 1.0e-12)
            events.append(
                {
                    "method": method,
                    "seed": seed,
                    "pressure_scale": pressure_scale,
                    "task_id": task["task_id"],
                    "scene_id": task["scene_id"],
                    "delivered_stage": stage,
                    "model_choice": "Tiny" if method == "fixed_lightweight" else "Medium" if method.startswith("fixed_heavy") else "adaptive",
                    "product_size_MB": payload,
                    "ground_receive_time_h": receive_time,
                    "before_deadline_flag": before_deadline,
                    "delivered_quality": q,
                    "incremental_information_utility": delta,
                }
            )
            trow = task_rows[str(task["task_id"])]
            trow["highest_ground_received_stage"] = highest
            trow["ground_received_quality"] = max(_to_float(trow["ground_received_quality"]), q)
            trow["task_energy_direct_wh"] = _to_float(trow["task_energy_direct_wh"]) + compute + comm
            if _stage_rank(highest) >= _stage_rank(str(task["min_required_stage"])) and not _to_int(trow["ground_received_flag"]):
                trow["ground_received_flag"] = 1
                trow["first_ground_receive_time_h"] = receive_time
                trow["ground_received_before_deadline_flag"] = before_deadline
                trow["capture_to_ground_latency_h"] = receive_time - _to_float(task["capture_time_h"])
            if _stage_rank(highest) >= _stage_rank(str(task["target_stage"])):
                trow["target_stage_ground_received_flag"] = 1
                if trow["target_ground_receive_time_h"] == "":
                    trow["target_ground_receive_time_h"] = receive_time
        pending = [req for req in pending if (req["task_id"], req["product_type"]) not in selected_keys]
        ready_count = len([req for req in pending if _to_int(req["created_orbit"]) <= orbit])
        backlog_mb = sum(_to_float(req["payload_size_MB"]) for req in pending if _to_int(req["created_orbit"]) <= orbit)
        payload_on = bool(ready_count or selected) and not protection
        payload_idle = _payload_idle_wh(method, payload_on)
        tx_fixed = TX_FIXED_WH_PER_ACTIVE_ORBIT if selected else 0.0
        solar = _to_float(resource_by_orbit[orbit]["solar_energy_wh"])
        housekeeping = _to_float(resource_by_orbit[orbit]["housekeeping_energy_wh"])
        before_soc = soc
        net_energy = solar - housekeeping - payload_idle - orbit_compute - orbit_comm - tx_fixed
        next_soc = max(0.0, min(1.0, soc + net_energy / BATTERY_CAPACITY_WH))
        if method != "fixed_heavy" and next_soc < HARD_MIN_SOC:
            next_soc = HARD_MIN_SOC
        if method == "fixed_heavy_with_20pct_protection" and not protection and next_soc <= HARD_MIN_SOC + 1.0e-12:
            protection = True
            protection_triggers += 1
            next_soc = HARD_MIN_SOC
        soc = next_soc
        if protection:
            protection_slots += 1
        system_energy = orbit_compute + orbit_comm + payload_idle
        cumulative_system_energy += system_energy
        if selected:
            for req in selected:
                share = system_energy / max(len(selected), 1)
                task_rows[str(req["task_id"])]["task_energy_system_wh"] = _to_float(task_rows[str(req["task_id"])]["task_energy_system_wh"]) + share
        trajectory.append(
            {
                "method": method,
                "seed": seed,
                "pressure_scale": pressure_scale,
                "orbit_index": orbit,
                "time_h": orbit * ORBIT_HOURS,
                "wave_name": wave,
                "SOC": soc,
                "soc_delta": soc - before_soc,
                "soc_slope_per_hour": (soc - before_soc) / ORBIT_HOURS,
                "soc_band": _soc_band(before_soc),
                "payload_on": int(payload_on),
                "protection_mode": int(protection),
                "backlog_count": ready_count,
                "backlog_MB": backlog_mb,
                "metadata_count": product_counts["metadata"],
                "roi_count": product_counts["roi"],
                "roi_context_count": product_counts["roi_context"],
                "defer_count": max(0, len(ready) - len(selected)),
                "compute_energy_wh": orbit_compute,
                "comm_energy_wh": orbit_comm,
                "payload_idle_energy_wh": payload_idle,
                "ground_received_quality_sum": sum(_to_float(e["delivered_quality"]) for e in events if _to_int(e["seed"]) == seed),
            }
        )

    task_delivery = list(task_rows.values())
    received = [row for row in task_delivery if _to_int(row["ground_received_flag"])]
    high_tasks = [row for row in task_delivery if _to_int(row["is_high_value"])]
    high_received = [row for row in high_tasks if _to_int(row["ground_received_flag"])]
    high_target = [row for row in high_tasks if _to_int(row["target_stage_ground_received_flag"])]
    soc_values = [_to_float(row["SOC"]) for row in trajectory]
    summary = {
        "method": method,
        "method_type": METHOD_TYPES[method],
        "seed": seed,
        "pressure_scale": pressure_scale,
        **hashes,
        "total_task_count": len(task_delivery),
        "ground_received_task_count": len(received),
        "ground_received_task_ratio": _safe_div(len(received), len(task_delivery)),
        "avg_ground_received_quality": statistics.mean([_to_float(row["ground_received_quality"]) for row in received]) if received else 0.0,
        "avg_capture_to_ground_latency_h": statistics.mean([_to_float(row["capture_to_ground_latency_h"]) for row in received]) if received else 0.0,
        "deadline_satisfaction_ratio": _safe_div(sum(_to_int(row["ground_received_before_deadline_flag"]) for row in received), len(received)),
        "high_value_task_count": len(high_tasks),
        "high_value_ground_received_count": len(high_received),
        "high_value_ground_received_ratio": _safe_div(len(high_received), len(high_tasks)),
        "high_value_target_stage_success_rate": _safe_div(len(high_target), len(high_tasks)),
        "avg_high_value_capture_to_ground_latency_h": statistics.mean([_to_float(row["capture_to_ground_latency_h"]) for row in high_received]) if high_received else 0.0,
        "direct_energy_per_received_task_wh": _safe_div(cumulative_direct_energy, len(received)),
        "system_energy_per_received_task_wh": _safe_div(cumulative_system_energy, len(received)),
        "direct_energy_per_high_value_received_task_wh": _safe_div(
            sum(_to_float(row["task_energy_direct_wh"]) for row in high_received), len(high_received)
        ),
        "system_energy_per_high_value_received_task_wh": _safe_div(
            sum(_to_float(row["task_energy_system_wh"]) for row in high_received), len(high_received)
        ),
        "net_system_utility": net_system_utility,
        "H_net_utility": h_net_utility,
        "min_SOC": min(soc_values),
        "final_SOC": soc_values[-1],
        "fraction_time_SOC_below_50": _safe_div(sum(value < 0.50 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_below_20": _safe_div(sum(value < HARD_MIN_SOC for value in soc_values), len(soc_values)),
        "hard_safety_violation_count": sum(value < HARD_MIN_SOC - 1.0e-12 for value in soc_values),
        "final_backlog_count": _to_int(trajectory[-1]["backlog_count"]),
        "max_backlog_count": max(_to_int(row["backlog_count"]) for row in trajectory),
        "protection_trigger_count": protection_triggers,
        "protection_total_slots": protection_slots,
    }
    return {
        "summary": summary,
        "events": events,
        "task_delivery": task_delivery,
        "trajectory": trajectory,
    }


def build_subtask_audit(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_scene[str(row["scene_id"])].append(row)
    rows = []
    for scene_id, tasks in sorted(by_scene.items()):
        values = [_to_float(row["V_eff"]) for row in tasks]
        rows.append(
            {
                "seed": tasks[0]["seed"],
                "pressure_scale": tasks[0]["pressure_scale"],
                "scene_id": scene_id,
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


def build_task_value_audit(manifests: list[dict[str, Any]], subtask_audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_case_wave: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifests:
        by_case_wave[(_to_int(row["seed"]), _to_float(row["pressure_scale"]), str(row["wave_name"]))].append(row)
    for (seed, pressure, wave), tasks in sorted(by_case_wave.items()):
        scenes = {str(row["scene_id"]) for row in tasks}
        scene_audit = [row for row in subtask_audit if _to_int(row["seed"]) == seed and _to_float(row["pressure_scale"]) == pressure and row["scene_id"] in scenes]
        rows.append(
            {
                "seed": seed,
                "pressure_scale": pressure,
                "wave_name": wave,
                "unique_scene_count": len(scenes),
                "total_task_count": len(tasks),
                "avg_subtasks_per_scene": _safe_div(len(tasks), len(scenes)),
                "within_scene_value_variance": statistics.mean([_to_float(row["value_variance"]) for row in scene_audit]) if scene_audit else 0.0,
                "within_scene_value_level_variation_count": sum(";" in str(row["distinct_value_levels"]) for row in scene_audit),
                "value_level_distribution_by_wave": dict(Counter(str(row["value_level"]) for row in tasks)),
                "product_requirement_distribution_by_wave": dict(Counter(str(row["product_requirement"]) for row in tasks)),
                "evidence_need_distribution_by_wave": dict(Counter(str(row["evidence_need"]) for row in tasks)),
                "high_or_urgent_count_by_wave": sum(str(row["value_level"]) in {"high", "urgent"} for row in tasks),
            }
        )
    return rows


def build_wave_summary(results: list[dict[str, Any]], manifests_by_case: dict[tuple[int, float], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        summary = result["summary"]
        method = summary["method"]
        seed = _to_int(summary["seed"])
        pressure = _to_float(summary["pressure_scale"])
        manifest = manifests_by_case[(seed, pressure)]
        delivery_by_task = {str(row["task_id"]): row for row in result["task_delivery"]}
        for wave, start, end, _density, _pressure in WAVES:
            tasks = [row for row in manifest if str(row["wave_name"]) == wave]
            traj = [row for row in result["trajectory"] if str(row["wave_name"]) == wave]
            received = [delivery_by_task[str(task["task_id"])] for task in tasks if _to_int(delivery_by_task[str(task["task_id"])]["ground_received_flag"])]
            high_tasks = [task for task in tasks if _to_int(task["is_high_value"])]
            high_received = [delivery_by_task[str(task["task_id"])] for task in high_tasks if _to_int(delivery_by_task[str(task["task_id"])]["ground_received_flag"])]
            socs = [_to_float(row["SOC"]) for row in traj]
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "pressure_scale": pressure,
                    "wave_name": wave,
                    "start_orbit": start,
                    "end_orbit": end,
                    "task_count": len(tasks),
                    "high_value_task_count": len(high_tasks),
                    "ground_received_task_count": len(received),
                    "high_value_ground_received_count": len(high_received),
                    "initial_SOC": _to_float(traj[0]["SOC"]) - _to_float(traj[0]["soc_delta"]) if traj else "",
                    "min_SOC": min(socs) if socs else "",
                    "final_SOC": socs[-1] if socs else "",
                    "SOC_drop": max(0.0, (_to_float(traj[0]["SOC"]) - _to_float(traj[0]["soc_delta"]) - min(socs)) if traj else 0.0),
                    "SOC_recovery": max(0.0, (socs[-1] - min(socs)) if socs else 0.0),
                    "avg_latency_h": statistics.mean([_to_float(row["capture_to_ground_latency_h"]) for row in received]) if received else 0.0,
                    "deadline_satisfaction_ratio": _safe_div(sum(_to_int(row["ground_received_before_deadline_flag"]) for row in received), len(received)),
                    "avg_ground_received_quality": statistics.mean([_to_float(row["ground_received_quality"]) for row in received]) if received else 0.0,
                    "direct_energy_per_received_task_wh": _safe_div(sum(_to_float(row["task_energy_direct_wh"]) for row in received), len(received)),
                }
            )
    return rows


def build_slope_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    bands = ["comfortable", "soft_protection", "energy_tight", "protection", "hard_violation"]
    for result in results:
        summary = result["summary"]
        events_by_orbit = defaultdict(list)
        for event in result["events"]:
            events_by_orbit[int(_to_float(event["ground_receive_time_h"]) / ORBIT_HOURS) - 1].append(event)
        for band in bands:
            pts = [row for row in result["trajectory"] if str(row["soc_band"]) == band]
            action_total = sum(_to_int(row["metadata_count"]) + _to_int(row["roi_count"]) + _to_int(row["roi_context_count"]) + _to_int(row["defer_count"]) for row in pts)
            qualities = [
                _to_float(event["delivered_quality"])
                for row in pts
                for event in events_by_orbit.get(_to_int(row["orbit_index"]), [])
            ]
            rows.append(
                {
                    "method": summary["method"],
                    "seed": summary["seed"],
                    "pressure_scale": summary["pressure_scale"],
                    "soc_band": band,
                    "avg_soc_slope_per_hour": statistics.mean([_to_float(row["soc_slope_per_hour"]) for row in pts]) if pts else 0.0,
                    "avg_payload_on_fraction": statistics.mean([_to_float(row["payload_on"]) for row in pts]) if pts else 0.0,
                    "avg_metadata_rate": _safe_div(sum(_to_int(row["metadata_count"]) for row in pts), action_total),
                    "avg_roi_rate": _safe_div(sum(_to_int(row["roi_count"]) for row in pts), action_total),
                    "avg_roi_context_rate": _safe_div(sum(_to_int(row["roi_context_count"]) for row in pts), action_total),
                    "avg_defer_rate": _safe_div(sum(_to_int(row["defer_count"]) for row in pts), action_total),
                    "avg_ground_received_quality": statistics.mean(qualities) if qualities else 0.0,
                    "ground_received_task_count": len(qualities),
                }
            )
    return rows


def build_multi_seed_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "ground_received_task_ratio",
        "avg_ground_received_quality",
        "avg_capture_to_ground_latency_h",
        "deadline_satisfaction_ratio",
        "high_value_ground_received_ratio",
        "high_value_target_stage_success_rate",
        "direct_energy_per_received_task_wh",
        "system_energy_per_received_task_wh",
        "net_system_utility",
        "H_net_utility",
        "min_SOC",
        "fraction_time_SOC_below_20",
    ]
    rows = []
    for pressure in PRESSURE_SCALES:
        pressure_rows = [row for row in summary_rows if _to_float(row["pressure_scale"]) == pressure]
        rank_by_metric: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for seed in SEEDS:
            seed_rows = [row for row in pressure_rows if _to_int(row["seed"]) == seed]
            for metric in metrics:
                reverse = metric not in {"avg_capture_to_ground_latency_h", "direct_energy_per_received_task_wh", "system_energy_per_received_task_wh", "fraction_time_SOC_below_20"}
                sorted_rows = sorted(seed_rows, key=lambda row: _to_float(row[metric]), reverse=reverse)
                for rank, row in enumerate(sorted_rows, start=1):
                    rank_by_metric[metric][str(row["method"])].append(rank)
        for method in METHODS:
            method_rows = [row for row in pressure_rows if row["method"] == method]
            for metric in metrics:
                values = [_to_float(row[metric]) for row in method_rows]
                ranks = rank_by_metric[metric][method]
                rows.append(
                    {
                        "method": method,
                        "method_type": METHOD_TYPES[method],
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
    return rows


def build_calibration_log(wave_rows: list[dict[str, Any]], manifests_by_case: dict[tuple[int, float], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    targets = {
        "wave_1_moderate": "0.70-0.76",
        "wave_2_extreme": "0.38-0.45",
        "recovery_2_incomplete": "0.72-0.79",
        "wave_3_medium_high": "0.52-0.58",
        "final_recovery": ">0.80",
    }
    rows = []
    for row in wave_rows:
        if row["method"] != "proposed_guarantee_aware_full" or str(row["wave_name"]) not in targets:
            continue
        key = (_to_int(row["seed"]), _to_float(row["pressure_scale"]))
        tasks = [task for task in manifests_by_case[key] if task["wave_name"] == row["wave_name"]]
        rows.append(
            {
                "seed": row["seed"],
                "pressure_scale": row["pressure_scale"],
                "target_wave": row["wave_name"],
                "proposed_wave_min_SOC": row["min_SOC"],
                "proposed_wave_final_SOC": row["final_SOC"],
                "target_SOC_range": targets[str(row["wave_name"])],
                "task_count": len(tasks),
                "high_value_count": sum(_to_int(task["is_high_value"]) for task in tasks),
                "note": "calibration target only; SOC is produced by shared workload/resource trace",
            }
        )
    return rows


def make_figures(out_dir: Path, summary: list[dict[str, Any]], trajectories: list[dict[str, Any]], wave_summary: list[dict[str, Any]], subtask_audit: list[dict[str, Any]], multi: list[dict[str, Any]]) -> None:
    base_seed, base_pressure = 0, 1.0
    base_traj = [row for row in trajectories if _to_int(row["seed"]) == base_seed and abs(_to_float(row["pressure_scale"]) - base_pressure) < 1.0e-9]
    plt.figure(figsize=(12, 5.4))
    for name, start, end, _density, _pressure in WAVES:
        plt.axvspan(start * ORBIT_HOURS, (end + 1) * ORBIT_HOURS, color="#f3d2c1" if name.startswith("wave") else "#d8ead3", alpha=0.22)
    for method in METHODS:
        rows = [row for row in base_traj if row["method"] == method]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["SOC"]) for row in rows], label=method, linewidth=1.6)
    for level in [0.70, 0.50, 0.20]:
        plt.axhline(level, color="black", linestyle="--", linewidth=0.8)
    prop_wave = [row for row in wave_summary if row["method"] == "proposed_guarantee_aware_full" and _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9]
    for row in prop_wave:
        if row["wave_name"] in {"wave_1_moderate", "wave_2_extreme", "recovery_2_incomplete", "wave_3_medium_high"}:
            plt.text((_to_int(row["start_orbit"]) + 0.3) * ORBIT_HOURS, _to_float(row["min_SOC"]) + 0.02, f"{row['wave_name']} min {_fmt(row['min_SOC'], 2)}", fontsize=7)
    plt.xlabel("time_h")
    plt.ylabel("SOC")
    plt.ylim(0.0, 1.02)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "soc_uneven_pressure_by_method.png", dpi=190)
    plt.close()

    base_summary = [row for row in summary if _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9]
    metrics = ["ground_received_task_ratio", "avg_ground_received_quality", "avg_capture_to_ground_latency_h", "direct_energy_per_received_task_wh"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, metric in zip(axes.flat, metrics):
        ax.bar([row["method"] for row in base_summary], [_to_float(row[metric]) for row in base_summary])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "ground_received_decomposed_metrics.png", dpi=190)
    plt.close(fig)

    metrics = ["high_value_ground_received_ratio", "high_value_target_stage_success_rate", "avg_high_value_capture_to_ground_latency_h", "H_net_utility"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, metric in zip(axes.flat, metrics):
        ax.bar([row["method"] for row in base_summary], [_to_float(row[metric]) for row in base_summary], color="#4c78a8")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "high_value_ground_delivery.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, metric in zip(axes, ["ground_received_task_ratio", "high_value_ground_received_ratio", "min_SOC"]):
        for method in ["proposed_guarantee_aware_full", "utility_rule", "persistent_queue_without_guarantee", "fixed_lightweight", "fixed_heavy_with_20pct_protection"]:
            vals = [
                next((_to_float(row["mean"]) for row in multi if row["method"] == method and row["metric"] == metric and abs(_to_float(row["pressure_scale"]) - p) < 1.0e-9), 0.0)
                for p in PRESSURE_SCALES
            ]
            ax.plot(PRESSURE_SCALES, vals, marker="o", label=method)
        ax.set_title(metric)
        ax.set_xlabel("pressure_scale")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "pressure_sensitivity_summary.png", dpi=190)
    plt.close(fig)

    scenes = sorted(subtask_audit, key=lambda row: _to_float(row["value_variance"]), reverse=True)[:10]
    plt.figure(figsize=(10, 4.8))
    plt.bar([str(row["scene_id"])[-14:] for row in scenes], [_to_float(row["value_variance"]) for row in scenes])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("within-scene V_eff variance")
    plt.tight_layout()
    plt.savefig(out_dir / "subtask_value_variation.png", dpi=190)
    plt.close()

    plt.figure(figsize=(8, 5.5))
    colors = {method: idx for idx, method in enumerate(METHODS)}
    for row in base_summary:
        plt.scatter(_to_float(row["min_SOC"]), _to_float(row["high_value_ground_received_ratio"]), s=30 + 2 * _to_float(row["ground_received_task_count"]), label=row["method"], alpha=0.78)
    plt.xlabel("min_SOC")
    plt.ylabel("high_value_ground_received_ratio")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff_ground_delivery_vs_safety.png", dpi=190)
    plt.close()


def _mean_for(multi: list[dict[str, Any]], method: str, pressure: float, metric: str) -> float:
    return next((_to_float(row["mean"]) for row in multi if row["method"] == method and row["metric"] == metric and abs(_to_float(row["pressure_scale"]) - pressure) < 1.0e-9), 0.0)


def write_quick_report(out_dir: Path, summary: list[dict[str, Any]], wave: list[dict[str, Any]], subtask_audit: list[dict[str, Any]], multi: list[dict[str, Any]]) -> None:
    base = [row for row in summary if _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9]
    by_method = {str(row["method"]): row for row in base}
    prop_waves = {
        str(row["wave_name"]): row
        for row in wave
        if row["method"] == "proposed_guarantee_aware_full" and _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    }
    prop_top2_counts = 0
    for pressure in PRESSURE_SCALES:
        rank = next((_to_float(row["rank_mean"]) for row in multi if row["method"] == "proposed_guarantee_aware_full" and row["metric"] == "high_value_ground_received_ratio" and abs(_to_float(row["pressure_scale"]) - pressure) < 1.0e-9), 9)
        prop_top2_counts += int(rank <= 2.0)
    lines = [
        "# Uneven pressure ground-delivery quick report",
        "",
        "## Direct Answers",
        "",
        "1. Uneven multi-wave pressure constructed: Yes. The workload uses warmup, moderate wave, full recovery, extreme wave, incomplete recovery, medium-high wave, and final recovery with unequal lengths and densities.",
        f"2. wave_1 proposed min/final SOC: {_fmt(prop_waves['wave_1_moderate']['min_SOC'])} / {_fmt(prop_waves['wave_1_moderate']['final_SOC'])}.",
        f"3. recovery_1 proposed final SOC: {_fmt(prop_waves['recovery_1']['final_SOC'])}.",
        f"4. wave_2_extreme proposed min/final SOC: {_fmt(prop_waves['wave_2_extreme']['min_SOC'])} / {_fmt(prop_waves['wave_2_extreme']['final_SOC'])}.",
        f"5. recovery_2 incomplete proposed final SOC: {_fmt(prop_waves['recovery_2_incomplete']['final_SOC'])}.",
        f"6. wave_3 from incomplete recovery proposed min/final SOC: {_fmt(prop_waves['wave_3_medium_high']['min_SOC'])} / {_fmt(prop_waves['wave_3_medium_high']['final_SOC'])}.",
        f"7. final_recovery proposed final SOC: {_fmt(prop_waves['final_recovery']['final_SOC'])}.",
        f"8. Same-scene subtask value variation exists: {sum(_to_int(row['has_within_scene_value_variation']) for row in subtask_audit)} scenes show V_eff variation; values come from task_class/product_requirement/evidence_need/deadline/source_priority through mission_prior_value_v2 and value_skew_calibration.",
        f"9. Proposed ground delivery base-case: ratio={_fmt(by_method['proposed_guarantee_aware_full']['ground_received_task_ratio'])}, quality={_fmt(by_method['proposed_guarantee_aware_full']['avg_ground_received_quality'])}, latency_h={_fmt(by_method['proposed_guarantee_aware_full']['avg_capture_to_ground_latency_h'])}, direct_energy_per_received={_fmt(by_method['proposed_guarantee_aware_full']['direct_energy_per_received_task_wh'])}.",
        f"10. Proposed high-value delivery base-case: ratio={_fmt(by_method['proposed_guarantee_aware_full']['high_value_ground_received_ratio'])}, target_success={_fmt(by_method['proposed_guarantee_aware_full']['high_value_target_stage_success_rate'])}, H_net={_fmt(by_method['proposed_guarantee_aware_full']['H_net_utility'])}.",
        f"11. fixed_lightweight battery/quality tradeoff: min_SOC={_fmt(by_method['fixed_lightweight']['min_SOC'])}, high_value_ratio={_fmt(by_method['fixed_lightweight']['high_value_ground_received_ratio'])}, quality={_fmt(by_method['fixed_lightweight']['avg_ground_received_quality'])}.",
        f"12. fixed_heavy collapses: {bool(_to_int(by_method['fixed_heavy']['hard_safety_violation_count']))}; min_SOC={_fmt(by_method['fixed_heavy']['min_SOC'])}.",
        f"13. fixed_heavy_with_20pct_protection avoids collapse but damages delivery: min_SOC={_fmt(by_method['fixed_heavy_with_20pct_protection']['min_SOC'])}, protection_slots={by_method['fixed_heavy_with_20pct_protection']['protection_total_slots']}, ground_ratio={_fmt(by_method['fixed_heavy_with_20pct_protection']['ground_received_task_ratio'])}.",
        f"14. utility_rule is myopic greedy: min_SOC={_fmt(by_method['utility_rule']['min_SOC'])}, backlog={by_method['utility_rule']['final_backlog_count']}, high_value_target_success={_fmt(by_method['utility_rule']['high_value_target_stage_success_rate'])}.",
        f"15. Multi-seed proposed top-2 count over pressure scales on high-value delivery: {prop_top2_counts}/3.",
        f"16. Pressure sensitivity proposed means: p0.85 high_value={_fmt(_mean_for(multi, 'proposed_guarantee_aware_full', 0.85, 'high_value_ground_received_ratio'))}, minSOC={_fmt(_mean_for(multi, 'proposed_guarantee_aware_full', 0.85, 'min_SOC'))}; p1.00 high_value={_fmt(_mean_for(multi, 'proposed_guarantee_aware_full', 1.00, 'high_value_ground_received_ratio'))}, minSOC={_fmt(_mean_for(multi, 'proposed_guarantee_aware_full', 1.00, 'min_SOC'))}; p1.15 high_value={_fmt(_mean_for(multi, 'proposed_guarantee_aware_full', 1.15, 'high_value_ground_received_ratio'))}, minSOC={_fmt(_mean_for(multi, 'proposed_guarantee_aware_full', 1.15, 'min_SOC'))}.",
        "17. Suitable as unified evaluation for later literature-inspired architecture baselines: Yes. The runner already reports method_type and ground-received task metrics, so raw_downlink_ground_processing, mec_cost_minimization, two_timescale_qos_scheduling, and goal_oriented_eo_edge can be added against the same manifest/contact/resource traces.",
        "",
        "## Isolation",
        "",
        "- New code is isolated in `scripts/run_uneven_pressure_ground_delivery_experiment.py`.",
        "- No `src` proposed, reward, evaluator, or value-prior files are modified.",
        "- High-value labels are derived from value_skew_calibration value_level; no manual high-value override is applied.",
    ]
    _write_text(out_dir / "quick_report.md", "\n".join(lines))


def validate_outputs(out_dir: Path) -> list[dict[str, Any]]:
    required = [
        "ground_delivery_task_manifest.csv",
        "ground_delivery_event_log.csv",
        "ground_delivery_summary_by_method_seed.csv",
        "ground_delivery_summary_multi_seed.csv",
        "soc_trajectory_by_method_seed.csv",
        "wave_summary_by_method_seed.csv",
        "slope_by_soc_band_by_method_seed.csv",
        "subtask_value_audit.csv",
        "calibration_log.csv",
        "task_value_audit.csv",
        "quick_report.md",
        "soc_uneven_pressure_by_method.png",
        "ground_received_decomposed_metrics.png",
        "high_value_ground_delivery.png",
        "pressure_sensitivity_summary.png",
        "subtask_value_variation.png",
        "tradeoff_ground_delivery_vs_safety.png",
    ]
    return [
        {
            "artifact": name,
            "exists": int((out_dir / name).exists()),
            "size_bytes": (out_dir / name).stat().st_size if (out_dir / name).exists() else 0,
        }
        for name in required
    ]


def run(output_dir: Path = OUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_manifest: list[dict[str, Any]] = []
    all_contacts: list[dict[str, Any]] = []
    all_resources: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    manifests_by_case: dict[tuple[int, float], list[dict[str, Any]]] = {}
    fairness_rows = []
    subtask_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        for pressure in PRESSURE_SCALES:
            contacts = build_contact_schedule(seed, pressure)
            resources = build_resource_trace(seed, pressure)
            manifest = build_task_manifest(seed, pressure, contacts)
            hashes = {
                "task_manifest_hash": _hash(manifest, GROUND_MANIFEST_COLUMNS),
                "contact_schedule_hash": _hash(contacts, ["seed", "pressure_scale", "orbit_index", "capacity_MB", "wave_name"]),
                "resource_trace_hash": _hash(resources, ["seed", "pressure_scale", "orbit_index", "solar_energy_wh", "wave_name"]),
            }
            manifests_by_case[(seed, pressure)] = manifest
            all_manifest.extend(manifest)
            all_contacts.extend(contacts)
            all_resources.extend(resources)
            subtask_rows.extend(build_subtask_audit(manifest))
            case_summaries = []
            for method in METHODS:
                result = run_method(method, seed, pressure, manifest, contacts, resources, hashes)
                all_results.append(result)
                case_summaries.append(result["summary"])
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

    summaries = [result["summary"] for result in all_results]
    events = [row for result in all_results for row in result["events"]]
    task_delivery = [row for result in all_results for row in result["task_delivery"]]
    trajectories = [row for result in all_results for row in result["trajectory"]]
    wave_summary = build_wave_summary(all_results, manifests_by_case)
    slope_summary = build_slope_summary(all_results)
    multi = build_multi_seed_summary(summaries)
    task_value_audit = build_task_value_audit(all_manifest, subtask_rows)
    calibration = build_calibration_log(wave_summary, manifests_by_case)

    _write_csv(output_dir / "ground_delivery_task_manifest.csv", all_manifest, GROUND_MANIFEST_COLUMNS)
    _write_csv(output_dir / "shared_contact_schedule.csv", all_contacts)
    _write_csv(output_dir / "shared_resource_trace.csv", all_resources)
    _write_csv(output_dir / "ground_delivery_event_log.csv", events, EVENT_COLUMNS)
    _write_csv(output_dir / "ground_delivery_task_delivery_by_method_seed.csv", task_delivery)
    _write_csv(output_dir / "ground_delivery_summary_by_method_seed.csv", summaries, SUMMARY_COLUMNS)
    _write_csv(output_dir / "ground_delivery_summary_multi_seed.csv", multi)
    _write_csv(output_dir / "soc_trajectory_by_method_seed.csv", trajectories)
    _write_csv(output_dir / "wave_summary_by_method_seed.csv", wave_summary)
    _write_csv(output_dir / "slope_by_soc_band_by_method_seed.csv", slope_summary)
    _write_csv(output_dir / "subtask_value_audit.csv", subtask_rows)
    _write_csv(output_dir / "task_value_audit.csv", task_value_audit)
    _write_csv(output_dir / "calibration_log.csv", calibration)
    _write_csv(output_dir / "fairness_hash_audit.csv", fairness_rows)
    make_figures(output_dir, summaries, trajectories, wave_summary, subtask_rows, multi)
    write_quick_report(output_dir, summaries, wave_summary, subtask_rows, multi)
    validation = validate_outputs(output_dir)
    _write_csv(output_dir / "artifact_validation.csv", validation)
    return {
        "output_dir": output_dir,
        "summaries": summaries,
        "events": events,
        "trajectories": trajectories,
        "wave_summary": wave_summary,
        "multi": multi,
        "fairness_rows": fairness_rows,
        "validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    result = run(args.output_dir)
    base_prop = next(
        row
        for row in result["summaries"]
        if row["method"] == "proposed_guarantee_aware_full" and _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    )
    print(f"output_dir={args.output_dir}")
    print(f"runs={len(result['summaries'])}")
    print(f"base_proposed_ground_received_ratio={_fmt(base_prop['ground_received_task_ratio'])}")
    print(f"base_proposed_high_value_ratio={_fmt(base_prop['high_value_ground_received_ratio'])}")
    print(f"base_proposed_min_SOC={_fmt(base_prop['min_SOC'])}")
    print(f"fairness_errors={sum(1 for row in result['fairness_rows'] if not _to_int(row['passed']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
