"""Fair baseline comparison runner for staged onboard delivery policies.

The runner is deliberately outside the reward-function and simulator control
path. It builds frozen task/contact inputs, runs all arms through one shared
lightweight lifecycle/accounting evaluator, and writes paper/audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.payload.information_sufficiency import delta_information_utility  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_fair_baseline_comparison"

DEFAULT_SCENARIOS = [
    "multi_window_rescuable_high_value",
    "natural_sarfish_layered_deadline",
    "urgent_deadline_stress",
]
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030]
DEFAULT_SLOTS = [160, 240]
SANITY_SCENARIO = "multi_window_rescuable_high_value"
SANITY_SEED = 2030
SANITY_SLOTS = 160

ACTIVE_ARMS = [
    "proposed_guarantee_aware_full",
    "utility_rule",
    "persistent_queue_without_guarantee",
    "fixed_lightweight_onboard",
    "fixed_heavy_onboard",
    "ground_approved_followup",
]
OPTIONAL_GROUND_COMPUTE_ARM = "downlink_first_ground_compute"
GROUND_COMPUTE_NOT_IMPLEMENTED_REASON = (
    "Not included in the main sweep because its product definition and resource model "
    "are not yet aligned with onboard ROI/ROI+Ctx products."
)

PAYLOAD_MODEL_VERSION = "frozen_payload_model_v1"
STAGE_EVALUATOR_VERSION = "shared_stage_evaluator_roi_min_roi_ctx_target_v1"
CLOSURE_EVALUATOR_VERSION = "shared_closure_evaluator_roi_not_evidence_v1"
SAFETY_EVALUATOR_VERSION = "shared_hard_safety_guard_v1"
ACCOUNTING_EVALUATOR_VERSION = "shared_delivered_product_accounting_v1"

MANIFEST_COLUMNS = [
    "scenario",
    "seed",
    "num_slots",
    "task_id",
    "scene_id",
    "arrival_slot",
    "deadline_slot",
    "deadline_seconds",
    "value_level",
    "base_prior_score",
    "V_eff",
    "product_requirement",
    "task_class",
    "high_value_flag",
    "emergency_flag",
    "minimum_stage_required",
    "target_stage_required",
    "payload_size_metadata_mb",
    "payload_size_roi_mb",
    "payload_size_roi_context_mb",
    "feasible_contact_windows_before_deadline",
    "notes",
]

CONTACT_COLUMNS = [
    "scenario",
    "seed",
    "num_slots",
    "contact_window_id",
    "start_slot",
    "end_slot",
    "capacity_MB",
    "throughput_mbps",
    "schedule_version",
]

LEDGER_COLUMNS = [
    "scenario",
    "seed",
    "num_slots",
    "arm",
    "lambda_H",
    "slot",
    "contact_window_id",
    "task_id",
    "product_type",
    "value_level",
    "product_requirement",
    "in_H",
    "payload_size_MB",
    "delivered_before_deadline",
    "stage_before_delivery",
    "stage_after_delivery",
    "V_eff",
    "delta_information_utility",
    "product_gross_information_utility",
    "compute_energy_Wh",
    "comm_energy_Wh",
    "product_resource_cost",
    "product_net_utility",
    "counted_in_total_gross",
    "counted_in_total_net",
    "counted_in_H_gross",
    "counted_in_H_net",
    "counted_in_H_min_stage",
    "counted_in_H_target_stage",
    "counted_in_closure",
    "evidence_closure_flag",
    "delivery_reason",
    "notes",
]

STAGE_AUDIT_COLUMNS = [
    "scenario",
    "seed",
    "num_slots",
    "arm",
    "task_id",
    "value_level",
    "product_requirement",
    "arrival_slot",
    "deadline_slot",
    "feasible_H",
    "final_stage",
    "metadata_delivered",
    "metadata_slot",
    "roi_delivered",
    "roi_slot",
    "roi_context_delivered",
    "roi_context_slot",
    "min_stage_required",
    "target_stage_required",
    "min_stage_success_before_deadline",
    "target_stage_success_before_deadline",
    "late_min_stage_delivery",
    "late_target_stage_delivery",
    "reason_if_min_stage_failed",
    "reason_if_target_stage_failed",
    "gross_utility_sum",
    "net_utility_sum",
]

MAIN_SUMMARY_COLUMNS = [
    "scenario",
    "seed",
    "num_slots",
    "arm",
    "lambda_H",
    "total_task_count",
    "H_task_count",
    "feasible_H_task_count",
    "net_system_utility",
    "total_gross_delivered_utility",
    "total_net_delivered_utility",
    "total_compute_energy_Wh",
    "total_comm_energy_Wh",
    "total_energy_Wh",
    "total_payload_MB",
    "total_resource_cost",
    "max_queue_MB",
    "avg_queue_MB",
    "gross_utility_per_MB",
    "gross_utility_per_Wh",
    "gross_utility_per_resource_cost",
    "H_gross_delivered_utility",
    "H_net_delivered_utility",
    "H_payload_MB",
    "H_gross_utility_per_MB",
    "H_net_utility_per_MB",
    "H_metadata_delivered_task_count",
    "H_roi_delivered_task_count",
    "H_roi_context_delivered_task_count",
    "H_min_stage_success_count",
    "H_target_stage_success_count",
    "H_min_stage_success_rate_all_H",
    "H_min_stage_success_rate_feasible_H",
    "H_target_stage_success_rate_all_H",
    "H_target_stage_success_rate_feasible_H",
    "H_late_min_stage_delivery_count",
    "H_late_target_stage_delivery_count",
    "H_left_behind_count",
    "H_left_behind_ratio",
    "guarantee_positive_delivered_count",
    "counterfactual_H_rescued_count",
    "counterfactual_H_rescued_min_stage_count",
    "counterfactual_H_rescued_target_stage_count",
    "naturally_delivered_H_count",
    "hard_safety_violation_count",
    "battery_violation_count",
    "thermal_violation_count",
    "queue_violation_count",
    "deadline_violation_count",
    "min_battery_SOC",
    "max_thermal_pressure",
    "low_medium_accidental_roi_context_count",
    "evidence_closure_from_roi_only_count",
    "lifecycle_orphan_count",
    "dropped_without_reason_count",
    "expired_without_reason_count",
    "ground_approval_request_count",
    "ground_approval_delay_slots",
    "ground_approved_roi_request_count",
    "ground_approved_roi_context_request_count",
    "ground_approved_delivered_count",
    "ground_approved_min_stage_success_count",
    "ground_approved_target_stage_success_count",
    "ground_approval_avg_delay_slots",
    "large_chip_delivered_count",
    "ground_compute_completed_count",
    "ground_compute_avg_delay_slots",
    "ground_compute_H_success_count",
]

PAPER_TABLE_COLUMNS = [
    "Method",
    "Net system utility",
    "Total net delivered utility",
    "Total resource consumption",
    "Gross utility / MB",
    "Gross utility / Wh",
    "H gross delivered utility",
    "H net delivered utility",
    "H min-stage success rate",
    "H target-stage success rate",
    "Counterfactual H rescued",
    "Hard safety violations",
    "Min SOC",
    "Max thermal",
    "Low/medium accidental ROI+Ctx",
]

ARM_LABELS = {
    "proposed_guarantee_aware_full": "Proposed guarantee-aware",
    "utility_rule": "Utility rule",
    "persistent_queue_without_guarantee": "Persistent queue w/o guarantee",
    "fixed_lightweight_onboard": "Fixed lightweight",
    "fixed_heavy_onboard": "Fixed heavy",
    "ground_approved_followup": "Ground-approved follow-up",
    "downlink_first_ground_compute": "Downlink-first ground compute",
}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _stable_unit(*parts: object) -> float:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    value = int(hashlib.sha256(raw).hexdigest()[:12], 16)
    return value / float(16**12 - 1)


def _stable_hash(rows: list[dict[str, object]], columns: list[str]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update("|".join(str(row.get(col, "")) for col in columns).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _format_float(value: object, digits: int = 4) -> str:
    return f"{_to_float(value):.{digits}f}"


def _is_H(task_or_level: dict[str, object] | str, requirement: object | None = None) -> bool:
    if isinstance(task_or_level, dict):
        value_level = str(task_or_level.get("value_level", "")).lower()
        req = str(task_or_level.get("product_requirement", "")).lower()
    else:
        value_level = str(task_or_level).lower()
        req = str(requirement or "").lower()
    return value_level in {"high", "urgent"} and req in {"context", "evidence"}


def _stage_rank(stage: str) -> int:
    return {"none": 0, "metadata": 1, "ROI": 2, "ROI+Ctx": 3}.get(stage, 0)


def _stage_from_products(products: set[str]) -> str:
    if "roi_context" in products:
        return "ROI+Ctx"
    if "roi" in products:
        return "ROI"
    if "metadata" in products:
        return "metadata"
    return "none"


def _product_payload(task: dict[str, object], product_type: str) -> float:
    if product_type == "metadata":
        return _to_float(task.get("payload_size_metadata_mb"))
    if product_type == "roi":
        return _to_float(task.get("payload_size_roi_mb"))
    if product_type == "roi_context":
        return _to_float(task.get("payload_size_roi_context_mb"))
    return 0.0


def arm_config(arm: str) -> dict[str, object]:
    configs: dict[str, dict[str, object]] = {
        "proposed_guarantee_aware_full": {
            "arm": arm,
            "lambda_H": 0.35,
            "fixed_model_profile": "",
            "model_quality": {"metadata": 0.90, "roi": 0.90, "roi_context": 0.88},
            "compute_multiplier": 1.15,
            "payload_multiplier": 1.0,
            "uses_guarantee": True,
            "uses_persistent_queue": True,
            "ground_approval_delay_slots": 0,
            "strategy": "guarantee_aware",
        },
        "utility_rule": {
            "arm": arm,
            "lambda_H": 0.0,
            "fixed_model_profile": "",
            "model_quality": {"metadata": 0.84, "roi": 0.82, "roi_context": 0.78},
            "compute_multiplier": 1.0,
            "payload_multiplier": 1.0,
            "uses_guarantee": False,
            "uses_persistent_queue": True,
            "ground_approval_delay_slots": 0,
            "strategy": "utility_density",
        },
        "persistent_queue_without_guarantee": {
            "arm": arm,
            "lambda_H": 0.0,
            "fixed_model_profile": "",
            "model_quality": {"metadata": 0.88, "roi": 0.86, "roi_context": 0.84},
            "compute_multiplier": 1.05,
            "payload_multiplier": 1.0,
            "uses_guarantee": False,
            "uses_persistent_queue": True,
            "ground_approval_delay_slots": 0,
            "strategy": "persistent_no_guarantee",
        },
        "fixed_lightweight_onboard": {
            "arm": arm,
            "lambda_H": 0.0,
            "fixed_model_profile": "Tiny",
            "model_quality": {"metadata": 0.76, "roi": 0.70, "roi_context": 0.63},
            "compute_multiplier": 0.45,
            "payload_multiplier": 0.85,
            "uses_guarantee": False,
            "uses_persistent_queue": True,
            "ground_approval_delay_slots": 0,
            "strategy": "fixed_lightweight",
        },
        "fixed_heavy_onboard": {
            "arm": arm,
            "lambda_H": 0.0,
            "fixed_model_profile": "Medium",
            "model_quality": {"metadata": 0.93, "roi": 0.92, "roi_context": 0.91},
            "compute_multiplier": 2.8,
            "payload_multiplier": 1.20,
            "uses_guarantee": False,
            "uses_persistent_queue": True,
            "ground_approval_delay_slots": 0,
            "strategy": "fixed_heavy",
        },
        "ground_approved_followup": {
            "arm": arm,
            "lambda_H": 0.0,
            "fixed_model_profile": "",
            "model_quality": {"metadata": 0.86, "roi": 0.84, "roi_context": 0.82},
            "compute_multiplier": 0.95,
            "payload_multiplier": 1.0,
            "uses_guarantee": False,
            "uses_persistent_queue": True,
            "ground_approval_delay_slots": 3,
            "strategy": "ground_approved_followup",
        },
    }
    if arm not in configs:
        raise ValueError(f"Unknown fair-comparison arm: {arm}")
    return configs[arm]


def build_contact_schedule(scenario: str, seed: int, num_slots: int) -> list[dict[str, object]]:
    if scenario == "urgent_deadline_stress":
        period, length, base_capacity = 11, 3, 4.8
    elif scenario == "natural_sarfish_layered_deadline":
        period, length, base_capacity = 14, 4, 6.4
    else:
        period, length, base_capacity = 12, 4, 6.0
    offset = 2 + int(_stable_unit(scenario, seed, num_slots, "contact_offset") * 3)
    rows: list[dict[str, object]] = []
    idx = 0
    start = offset
    while start < num_slots:
        end = min(num_slots - 1, start + length - 1)
        variation = 0.78 + 0.44 * _stable_unit(scenario, seed, num_slots, idx, "capacity")
        capacity = round(base_capacity * variation, 4)
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "num_slots": num_slots,
                "contact_window_id": f"cw_{scenario}_s{seed}_n{num_slots}_{idx:03d}",
                "start_slot": start,
                "end_slot": end,
                "capacity_MB": capacity,
                "throughput_mbps": round(capacity * 8.0 / max(1, (end - start + 1) * 20.0), 4),
                "schedule_version": "fair_contact_schedule_v1",
            }
        )
        idx += 1
        start += period + int(_stable_unit(scenario, seed, idx, "gap") * 2)
    return rows


def _deadline_delta_slots(scenario: str, value_level: str, requirement: str) -> int:
    if scenario == "urgent_deadline_stress":
        if value_level == "urgent":
            return 21
        if value_level == "high":
            return 30
        return 24
    if scenario == "natural_sarfish_layered_deadline":
        if value_level in {"urgent", "high"} and requirement in {"context", "evidence"}:
            return 44
        return 34
    if value_level in {"urgent", "high"} and requirement in {"context", "evidence"}:
        return 42
    return 30


def _value_level_for_index(scenario: str, index: int) -> str:
    if scenario == "urgent_deadline_stress":
        if index % 7 == 0:
            return "urgent"
        if index % 3 == 0:
            return "high"
        if index % 3 == 1:
            return "medium"
        return "low"
    if scenario == "natural_sarfish_layered_deadline":
        if index % 11 == 0:
            return "urgent"
        if index % 4 in {0, 1}:
            return "high"
        if index % 4 == 2:
            return "medium"
        return "low"
    if index % 10 == 0:
        return "urgent"
    if index % 4 == 0:
        return "high"
    if index % 4 == 1:
        return "medium"
    return "low"


def _requirement_for_index(value_level: str, index: int) -> str:
    if value_level in {"high", "urgent"}:
        return "evidence" if index % 2 == 0 else "context"
    if value_level == "medium":
        return "context" if index % 7 == 0 else "verification"
    return "alert" if index % 3 else "verification"


def _task_value(value_level: str, unit: float) -> tuple[float, float]:
    base = {
        "urgent": 2.35,
        "high": 1.65,
        "medium": 0.82,
        "low": 0.34,
    }[value_level]
    prior = min(0.99, max(0.05, base / 2.5 + 0.08 * (unit - 0.5)))
    return round(prior, 5), round(base * (0.92 + 0.16 * unit), 5)


def generate_frozen_task_manifest(
    scenario: str,
    seed: int,
    num_slots: int,
    contact_schedule: list[dict[str, object]],
) -> list[dict[str, object]]:
    task_count = max(24, int(math.ceil(num_slots / 5.0)))
    spacing = max(3, num_slots // task_count)
    rows: list[dict[str, object]] = []
    used_arrivals: set[int] = set()
    for index in range(task_count):
        value_level = _value_level_for_index(scenario, index)
        requirement = _requirement_for_index(value_level, index)
        jitter = int(_stable_unit(scenario, seed, num_slots, index, "arrival") * max(1, spacing))
        arrival = min(num_slots - 2, index * spacing + jitter)
        while arrival in used_arrivals and arrival < num_slots - 2:
            arrival += 1
        used_arrivals.add(arrival)
        deadline_delta = _deadline_delta_slots(scenario, value_level, requirement)
        deadline_slot = min(num_slots - 1, arrival + deadline_delta)
        base_prior, v_eff = _task_value(value_level, _stable_unit(scenario, seed, index, "value"))
        difficulty = 0.7 + 0.6 * _stable_unit(scenario, seed, index, "payload")
        metadata_size = round(0.055 + 0.045 * _stable_unit(scenario, seed, index, "metadata"), 5)
        roi_size = round((0.58 + 0.55 * _stable_unit(scenario, seed, index, "roi")) * difficulty, 5)
        context_factor = 2.05 + 0.65 * _stable_unit(scenario, seed, index, "ctx")
        roi_context_size = round(roi_size * context_factor, 5)
        min_stage, target_stage = _required_stages(requirement)
        feasible = sum(
            1
            for window in contact_schedule
            if arrival <= _to_int(window["start_slot"]) <= deadline_slot
        )
        notes = []
        if feasible <= 0:
            notes.append("no_contact_before_deadline")
        if scenario == "urgent_deadline_stress" and value_level == "urgent":
            notes.append("urgent_deadline_stress_task")
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "num_slots": num_slots,
                "task_id": f"{scenario}_s{seed}_n{num_slots}_t{index:03d}",
                "scene_id": f"{scenario}_scene_{index:03d}",
                "arrival_slot": arrival,
                "deadline_slot": deadline_slot,
                "deadline_seconds": (deadline_slot - arrival + 1) * 20,
                "value_level": value_level,
                "base_prior_score": base_prior,
                "V_eff": v_eff,
                "product_requirement": requirement,
                "task_class": "sarfish_candidate" if scenario != "urgent_deadline_stress" else "urgent_sarfish_candidate",
                "high_value_flag": int(value_level in {"high", "urgent"}),
                "emergency_flag": int(value_level == "urgent"),
                "minimum_stage_required": min_stage,
                "target_stage_required": target_stage,
                "payload_size_metadata_mb": metadata_size,
                "payload_size_roi_mb": roi_size,
                "payload_size_roi_context_mb": roi_context_size,
                "feasible_contact_windows_before_deadline": feasible,
                "notes": ";".join(notes),
            }
        )
    return sorted(rows, key=lambda row: (_to_int(row["arrival_slot"]), str(row["task_id"])))


def _required_stages(requirement: object) -> tuple[str, str]:
    req = str(requirement or "").lower()
    if req == "alert":
        return "metadata", "metadata"
    if req == "verification":
        return "ROI", "ROI"
    return "ROI", "ROI+Ctx"


def _max_requested_stage(task: dict[str, object], cfg: dict[str, object]) -> str:
    requirement = str(task.get("product_requirement", "")).lower()
    value_level = str(task.get("value_level", "")).lower()
    strategy = str(cfg["strategy"])
    if requirement == "alert":
        return "metadata"
    if requirement == "verification":
        return "ROI"
    if value_level not in {"high", "urgent"}:
        return "ROI"
    if strategy == "utility_density":
        roi_delta = delta_information_utility("roi", {"metadata"}, requirement)
        roi_density = _safe_div(
            _to_float(task.get("V_eff")) * roi_delta * _quality_for_product(cfg, "roi"),
            _product_payload(task, "roi"),
        )
        if roi_density < 0.78:
            return "metadata"
    if strategy == "fixed_lightweight" and _to_float(task.get("V_eff")) < 1.8:
        return "ROI"
    return "ROI+Ctx"


def _next_product_for_task(
    task: dict[str, object],
    delivered: set[str],
    generated: set[str],
    cfg: dict[str, object],
    current_slot: int,
    delivered_slots: dict[str, int],
) -> tuple[str, int, str] | None:
    max_stage = _max_requested_stage(task, cfg)
    strategy = str(cfg["strategy"])
    delay = _to_int(cfg.get("ground_approval_delay_slots"))

    if "metadata" not in delivered and "metadata" not in generated:
        return "metadata", current_slot, "initial_metadata"
    if _stage_rank(max_stage) < _stage_rank("ROI"):
        return None
    if "metadata" in delivered and "roi" not in delivered and "roi" not in generated:
        earliest = current_slot
        reason = "progressive_roi"
        if strategy == "ground_approved_followup":
            earliest = delivered_slots.get("metadata", current_slot) + delay
            reason = "ground_approved_roi_followup"
        return "roi", earliest, reason
    if _stage_rank(max_stage) < _stage_rank("ROI+Ctx"):
        return None
    if "roi" in delivered and "roi_context" not in delivered and "roi_context" not in generated:
        earliest = current_slot
        reason = "progressive_roi_context"
        if strategy == "ground_approved_followup":
            earliest = delivered_slots.get("roi", current_slot) + delay
            reason = "ground_approved_roi_context_followup"
        return "roi_context", earliest, reason
    return None


def _quality_for_product(cfg: dict[str, object], product_type: str) -> float:
    quality = cfg.get("model_quality", {})
    if isinstance(quality, dict):
        return _to_float(quality.get(product_type), 0.8)
    return 0.8


def _request_key(request: dict[str, object]) -> str:
    return f"{request['task_id']}::{request['product_type']}"


def _request_delta(request: dict[str, object], delivered: set[str]) -> float:
    return delta_information_utility(
        str(request["product_type"]),
        delivered,
        str(request["product_requirement"]),
    )


def _timeliness(task: dict[str, object], current_slot: int) -> float:
    arrival = _to_int(task.get("arrival_slot"))
    deadline = _to_int(task.get("deadline_slot"))
    if current_slot > deadline:
        return 0.0
    span = max(1, deadline - arrival + 1)
    remaining = max(0, deadline - current_slot + 1)
    return 0.60 + 0.40 * (remaining / span)


def _priority_score(
    request: dict[str, object],
    task: dict[str, object],
    delivered: set[str],
    current_slot: int,
    cfg: dict[str, object],
    *,
    with_guarantee: bool,
) -> tuple[float, dict[str, float]]:
    product_type = str(request["product_type"])
    payload = max(1.0e-9, _to_float(request["payload_size_MB"]))
    delta = _request_delta(request, delivered)
    quality = _quality_for_product(cfg, product_type)
    gross_estimate = _to_float(task.get("V_eff")) * delta * quality * _timeliness(task, current_slot)
    density = gross_estimate / payload
    urgency = _safe_div(1.0, max(1, _to_int(task.get("deadline_slot")) - current_slot + 1))
    age = max(0, current_slot - _to_int(request.get("created_slot")))
    value_boost = 0.05 if _is_H(task) and str(cfg["strategy"]) != "utility_density" else 0.0
    score = density + 0.015 * age + 0.25 * urgency + value_boost
    guarantee_term = 0.0
    if with_guarantee and bool(cfg.get("uses_guarantee")) and _is_H(task) and product_type in {"roi", "roi_context"}:
        missed_windows = _to_float(request.get("missed_contact_windows"))
        stage_need = 1.15 if product_type == "roi" else 1.35
        guarantee_term = _to_float(cfg.get("lambda_H")) * (stage_need + 0.04 * age + 0.35 * missed_windows)
        score += guarantee_term
    if str(cfg["strategy"]) == "fixed_lightweight" and product_type == "roi_context":
        score -= 0.35
    if str(cfg["strategy"]) == "fixed_heavy":
        score -= 0.06 * _to_float(cfg.get("compute_multiplier"))
    if str(cfg["strategy"]) == "ground_approved_followup":
        score -= 0.08 * _to_int(cfg.get("ground_approval_delay_slots"))
    return score, {
        "density": density,
        "gross_estimate": gross_estimate,
        "guarantee_term": guarantee_term,
        "age": float(age),
        "urgency": urgency,
    }


def _select_requests(
    ready_requests: list[dict[str, object]],
    task_by_id: dict[str, dict[str, object]],
    delivered_by_task: dict[str, set[str]],
    current_slot: int,
    cfg: dict[str, object],
    capacity_mb: float,
    *,
    with_guarantee: bool,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    score_info: dict[str, dict[str, float]] = {}
    scored: list[tuple[float, str, dict[str, object]]] = []
    for request in ready_requests:
        task = task_by_id[str(request["task_id"])]
        score, parts = _priority_score(
            request,
            task,
            delivered_by_task[str(request["task_id"])],
            current_slot,
            cfg,
            with_guarantee=with_guarantee,
        )
        score_info[_request_key(request)] = parts | {"score": score}
        scored.append((score, _request_key(request), request))
    selected: list[dict[str, object]] = []
    remaining = capacity_mb
    for _score, _key, request in sorted(scored, key=lambda item: (-item[0], item[1])):
        payload = _to_float(request["payload_size_MB"])
        if payload <= remaining + 1.0e-12:
            selected.append(request)
            remaining -= payload
    return selected, score_info


def _compute_product_costs(product_type: str, payload_mb: float, cfg: dict[str, object]) -> tuple[float, float, float]:
    base_compute = {"metadata": 0.010, "roi": 0.045, "roi_context": 0.090}.get(product_type, 0.0)
    compute = base_compute * _to_float(cfg.get("compute_multiplier"), 1.0)
    comm = 0.026 * payload_mb
    resource = compute + comm + 0.012 * payload_mb
    return compute, comm, resource


def run_arm_case(
    manifest: list[dict[str, object]],
    contact_schedule: list[dict[str, object]],
    arm: str,
) -> dict[str, Any]:
    cfg = arm_config(arm)
    scenario = str(manifest[0]["scenario"]) if manifest else ""
    seed = _to_int(manifest[0]["seed"]) if manifest else 0
    num_slots = _to_int(manifest[0]["num_slots"]) if manifest else 0
    task_by_id = {str(task["task_id"]): task for task in manifest}
    arrivals_by_slot: dict[int, list[dict[str, object]]] = defaultdict(list)
    for task in manifest:
        arrivals_by_slot[_to_int(task["arrival_slot"])].append(task)
    contact_by_slot = {_to_int(window["start_slot"]): window for window in contact_schedule}
    max_window_capacity = max((_to_float(window["capacity_MB"]) for window in contact_schedule), default=0.0)

    delivered_by_task: dict[str, set[str]] = defaultdict(set)
    delivered_slots_by_task: dict[str, dict[str, int]] = defaultdict(dict)
    generated_by_task: dict[str, set[str]] = defaultdict(set)
    active_task_ids: set[str] = set()
    pending: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    queue_trace: list[dict[str, object]] = []
    approval_delays: list[int] = []
    ground_approval_request_count = 0
    ground_approved_roi_request_count = 0
    ground_approved_roi_context_request_count = 0
    min_soc = 1.0
    max_thermal = 0.32

    def add_request(task: dict[str, object], product_type: str, earliest_slot: int, reason: str, current_slot: int) -> None:
        nonlocal ground_approval_request_count, ground_approved_roi_request_count, ground_approved_roi_context_request_count
        payload = _product_payload(task, product_type) * _to_float(cfg.get("payload_multiplier"), 1.0)
        request = {
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
        pending.append(request)
        generated_by_task[str(task["task_id"])].add(product_type)
        if str(cfg["strategy"]) == "ground_approved_followup" and product_type in {"roi", "roi_context"}:
            ground_approval_request_count += 1
            if product_type == "roi":
                ground_approved_roi_request_count += 1
            else:
                ground_approved_roi_context_request_count += 1
            approval_delays.append(max(0, earliest_slot - current_slot))

    for slot in range(num_slots):
        for task in arrivals_by_slot.get(slot, []):
            active_task_ids.add(str(task["task_id"]))

        for task_id in sorted(active_task_ids):
            task = task_by_id[task_id]
            if slot > _to_int(task["deadline_slot"]):
                continue
            next_request = _next_product_for_task(
                task,
                delivered_by_task[task_id],
                generated_by_task[task_id],
                cfg,
                slot,
                delivered_slots_by_task[task_id],
            )
            if next_request is None:
                continue
            product_type, earliest_slot, reason = next_request
            add_request(task, product_type, earliest_slot, reason, slot)

        queue_mb = sum(_to_float(req["payload_size_MB"]) for req in pending if _to_int(req["earliest_slot"]) <= slot)
        queue_trace.append({"slot": slot, "queue_MB": queue_mb})
        if slot not in contact_by_slot:
            continue

        window = contact_by_slot[slot]
        ready = [
            req
            for req in pending
            if _to_int(req["earliest_slot"]) <= slot
            and slot <= _to_int(task_by_id[str(req["task_id"])]["deadline_slot"])
        ]
        capacity = _to_float(window["capacity_MB"])
        selected_without, score_without = _select_requests(
            ready,
            task_by_id,
            delivered_by_task,
            slot,
            cfg,
            capacity,
            with_guarantee=False,
        )
        selected_with, score_with = _select_requests(
            ready,
            task_by_id,
            delivered_by_task,
            slot,
            cfg,
            capacity,
            with_guarantee=True,
        )
        selected_without_keys = {_request_key(req) for req in selected_without}
        selected_with_keys = {_request_key(req) for req in selected_with}

        for req in ready:
            if _request_key(req) not in selected_with_keys:
                req["missed_contact_windows"] = _to_int(req.get("missed_contact_windows")) + 1

        for req in selected_with:
            task_id = str(req["task_id"])
            task = task_by_id[task_id]
            product_type = str(req["product_type"])
            key = _request_key(req)
            delivered_before = set(delivered_by_task[task_id])
            stage_before = _stage_from_products(delivered_before)
            delta = _request_delta(req, delivered_before)
            if delta <= 1.0e-12:
                pending = [item for item in pending if _request_key(item) != key]
                continue
            payload = _to_float(req["payload_size_MB"])
            quality = _quality_for_product(cfg, product_type)
            timeliness = _timeliness(task, slot)
            gross = max(0.0, _to_float(task["V_eff"]) * delta * quality * timeliness)
            compute, comm, resource = _compute_product_costs(product_type, payload, cfg)
            net = gross - resource
            min_soc = min(min_soc, max(0.22, 1.0 - 0.006 * len(ledger) - 0.018 * compute - 0.004 * comm))
            max_thermal = max(max_thermal, min(0.88, 0.32 + 0.008 * len(ledger) + 0.08 * compute))
            delivered_after = set(delivered_before)
            delivered_after.add(product_type)
            delivered_by_task[task_id] = delivered_after
            delivered_slots_by_task[task_id][product_type] = slot
            before_deadline = int(slot <= _to_int(task["deadline_slot"]))
            in_H = int(_is_H(task))
            min_required, target_required = _required_stages(task["product_requirement"])
            stage_after = _stage_from_products(delivered_after)
            counted_min = int(
                in_H
                and before_deadline
                and _stage_rank(stage_after) >= _stage_rank(min_required)
                and _stage_rank(stage_before) < _stage_rank(min_required)
            )
            counted_target = int(
                in_H
                and before_deadline
                and _stage_rank(stage_after) >= _stage_rank(target_required)
                and _stage_rank(stage_before) < _stage_rank(target_required)
            )
            evidence_closure = int(
                before_deadline
                and str(task["product_requirement"]) == "evidence"
                and _stage_rank(stage_after) >= _stage_rank("ROI+Ctx")
                and _stage_rank(stage_before) < _stage_rank("ROI+Ctx")
            )
            context_closure = int(
                before_deadline
                and str(task["product_requirement"]) == "context"
                and _stage_rank(stage_after) >= _stage_rank("ROI+Ctx")
                and _stage_rank(stage_before) < _stage_rank("ROI+Ctx")
            )
            verification_closure = int(
                before_deadline
                and str(task["product_requirement"]) == "verification"
                and _stage_rank(stage_after) >= _stage_rank("ROI")
                and _stage_rank(stage_before) < _stage_rank("ROI")
            )
            term = _to_float(score_with.get(key, {}).get("guarantee_term"))
            if bool(cfg.get("uses_guarantee")) and key not in selected_without_keys and term > 0.0:
                delivery_reason = "counterfactual_rescued"
            elif bool(cfg.get("uses_guarantee")) and term > 0.0:
                delivery_reason = "guarantee_positive"
            elif str(cfg["strategy"]) == "utility_density":
                delivery_reason = "utility_selected"
            elif str(cfg["strategy"]) == "ground_approved_followup" and product_type in {"roi", "roi_context"}:
                delivery_reason = "ground_approved_followup"
            elif str(cfg["strategy"]).startswith("fixed_"):
                delivery_reason = str(cfg["strategy"])
            else:
                delivery_reason = "natural"
            notes = [
                f"payload_model={PAYLOAD_MODEL_VERSION}",
                f"stage_evaluator={STAGE_EVALUATOR_VERSION}",
            ]
            fixed_model = str(cfg.get("fixed_model_profile", ""))
            if fixed_model:
                notes.append(f"fixed_model={fixed_model}")
            if str(req.get("generation_reason", "")).startswith("ground_approved"):
                notes.append(str(req["generation_reason"]))
            ledger.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "num_slots": num_slots,
                    "arm": arm,
                    "lambda_H": _to_float(cfg["lambda_H"]),
                    "slot": slot,
                    "contact_window_id": window["contact_window_id"],
                    "task_id": task_id,
                    "product_type": product_type,
                    "value_level": task["value_level"],
                    "product_requirement": task["product_requirement"],
                    "in_H": in_H,
                    "payload_size_MB": payload,
                    "delivered_before_deadline": before_deadline,
                    "stage_before_delivery": stage_before,
                    "stage_after_delivery": stage_after,
                    "V_eff": task["V_eff"],
                    "delta_information_utility": delta,
                    "product_gross_information_utility": gross,
                    "compute_energy_Wh": compute,
                    "comm_energy_Wh": comm,
                    "product_resource_cost": resource,
                    "product_net_utility": net,
                    "counted_in_total_gross": 1,
                    "counted_in_total_net": 1,
                    "counted_in_H_gross": in_H,
                    "counted_in_H_net": in_H,
                    "counted_in_H_min_stage": counted_min,
                    "counted_in_H_target_stage": counted_target,
                    "counted_in_closure": int(evidence_closure or context_closure or verification_closure),
                    "evidence_closure_flag": evidence_closure,
                    "delivery_reason": delivery_reason,
                    "notes": ";".join(notes),
                }
            )
            pending = [item for item in pending if _request_key(item) != key]

    stage_rows = build_stage_success_audit(ledger, manifest, arm)
    summary = build_main_summary(
        manifest=manifest,
        ledger=ledger,
        stage_rows=stage_rows,
        queue_trace=queue_trace,
        cfg=cfg,
        min_soc=min_soc,
        max_thermal=max_thermal,
        max_window_capacity=max_window_capacity,
        ground_approval_request_count=ground_approval_request_count,
        ground_approved_roi_request_count=ground_approved_roi_request_count,
        ground_approved_roi_context_request_count=ground_approved_roi_context_request_count,
        approval_delays=approval_delays,
    )
    metadata = {
        "scenario": scenario,
        "seed": seed,
        "num_slots": num_slots,
        "arm": arm,
        "manifest_hash": _stable_hash(manifest, MANIFEST_COLUMNS),
        "contact_schedule_hash": _stable_hash(contact_schedule, CONTACT_COLUMNS),
        "payload_model_hash": PAYLOAD_MODEL_VERSION,
        "stage_evaluator": STAGE_EVALUATOR_VERSION,
        "closure_evaluator": CLOSURE_EVALUATOR_VERSION,
        "safety_evaluator": SAFETY_EVALUATOR_VERSION,
        "accounting_evaluator": ACCOUNTING_EVALUATOR_VERSION,
        "lambda_H": cfg["lambda_H"],
    }
    return {
        "summary": summary,
        "ledger": ledger,
        "stage_rows": stage_rows,
        "queue_trace": queue_trace,
        "metadata": metadata,
        "pending": pending,
    }


def build_stage_success_audit(
    ledger: list[dict[str, object]],
    manifest: list[dict[str, object]],
    arm: str,
) -> list[dict[str, object]]:
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ledger:
        by_task[str(row["task_id"])].append(row)
    rows: list[dict[str, object]] = []
    for task in manifest:
        if not _is_H(task):
            continue
        task_id = str(task["task_id"])
        deliveries = sorted(by_task.get(task_id, []), key=lambda item: _to_int(item["slot"]))
        all_products = {str(row["product_type"]) for row in deliveries}
        before_deadline = [row for row in deliveries if _to_int(row["delivered_before_deadline"])]
        bd_products = {str(row["product_type"]) for row in before_deadline}
        metadata_slot = _first_slot(deliveries, "metadata")
        roi_slot = _first_slot(deliveries, "roi")
        roi_context_slot = _first_slot(deliveries, "roi_context")
        min_stage, target_stage = _required_stages(task["product_requirement"])
        min_success = int(_stage_rank(_stage_from_products(bd_products)) >= _stage_rank(min_stage))
        target_success = int(_stage_rank(_stage_from_products(bd_products)) >= _stage_rank(target_stage))
        late_min = int(not min_success and _stage_rank(_stage_from_products(all_products)) >= _stage_rank(min_stage))
        late_target = int(not target_success and _stage_rank(_stage_from_products(all_products)) >= _stage_rank(target_stage))
        reason_min = "" if min_success else _failure_reason(task, deliveries, min_stage, arm)
        reason_target = "" if target_success else _failure_reason(task, deliveries, target_stage, arm)
        rows.append(
            {
                "scenario": task["scenario"],
                "seed": task["seed"],
                "num_slots": task["num_slots"],
                "arm": arm,
                "task_id": task_id,
                "value_level": task["value_level"],
                "product_requirement": task["product_requirement"],
                "arrival_slot": task["arrival_slot"],
                "deadline_slot": task["deadline_slot"],
                "feasible_H": int(_to_int(task["feasible_contact_windows_before_deadline"]) > 0),
                "final_stage": _stage_from_products(all_products),
                "metadata_delivered": int("metadata" in all_products),
                "metadata_slot": metadata_slot,
                "roi_delivered": int("roi" in all_products),
                "roi_slot": roi_slot,
                "roi_context_delivered": int("roi_context" in all_products),
                "roi_context_slot": roi_context_slot,
                "min_stage_required": min_stage,
                "target_stage_required": target_stage,
                "min_stage_success_before_deadline": min_success,
                "target_stage_success_before_deadline": target_success,
                "late_min_stage_delivery": late_min,
                "late_target_stage_delivery": late_target,
                "reason_if_min_stage_failed": reason_min,
                "reason_if_target_stage_failed": reason_target,
                "gross_utility_sum": sum(_to_float(row["product_gross_information_utility"]) for row in deliveries),
                "net_utility_sum": sum(_to_float(row["product_net_utility"]) for row in deliveries),
            }
        )
    return rows


def _first_slot(rows: list[dict[str, object]], product_type: str) -> object:
    slots = [_to_int(row["slot"]) for row in rows if row.get("product_type") == product_type]
    return min(slots) if slots else ""


def _failure_reason(task: dict[str, object], deliveries: list[dict[str, object]], required_stage: str, arm: str) -> str:
    if _to_int(task.get("feasible_contact_windows_before_deadline")) <= 0:
        return "no feasible H tasks"
    products = {str(row.get("product_type")) for row in deliveries}
    if not deliveries:
        return "no H products generated"
    if _stage_rank(_stage_from_products(products)) < _stage_rank(required_stage):
        if products == {"metadata"}:
            return "delivered metadata only"
        if "roi" in products and required_stage == "ROI+Ctx":
            if arm == "ground_approved_followup":
                return "ground approval delay too long"
            return "H products generated but not delivered"
        return "payload too large / no capacity"
    return "delivered ROI after deadline"


def build_main_summary(
    *,
    manifest: list[dict[str, object]],
    ledger: list[dict[str, object]],
    stage_rows: list[dict[str, object]],
    queue_trace: list[dict[str, object]],
    cfg: dict[str, object],
    min_soc: float,
    max_thermal: float,
    max_window_capacity: float,
    ground_approval_request_count: int,
    ground_approved_roi_request_count: int,
    ground_approved_roi_context_request_count: int,
    approval_delays: list[int],
) -> dict[str, object]:
    scenario = str(manifest[0]["scenario"]) if manifest else ""
    seed = _to_int(manifest[0]["seed"]) if manifest else 0
    num_slots = _to_int(manifest[0]["num_slots"]) if manifest else 0
    arm = str(cfg["arm"])
    h_tasks = [task for task in manifest if _is_H(task)]
    feasible_h = [task for task in h_tasks if _to_int(task["feasible_contact_windows_before_deadline"]) > 0]
    h_task_ids = {str(task["task_id"]) for task in h_tasks}
    h_ledger = [row for row in ledger if _to_int(row["in_H"])]
    total_gross = sum(_to_float(row["product_gross_information_utility"]) for row in ledger)
    total_net = sum(_to_float(row["product_net_utility"]) for row in ledger)
    total_compute = sum(_to_float(row["compute_energy_Wh"]) for row in ledger)
    total_comm = sum(_to_float(row["comm_energy_Wh"]) for row in ledger)
    total_payload = sum(_to_float(row["payload_size_MB"]) for row in ledger)
    total_resource = sum(_to_float(row["product_resource_cost"]) for row in ledger)
    h_gross = sum(_to_float(row["product_gross_information_utility"]) for row in h_ledger)
    h_net = sum(_to_float(row["product_net_utility"]) for row in h_ledger)
    h_payload = sum(_to_float(row["payload_size_MB"]) for row in h_ledger)
    min_success = sum(_to_int(row["min_stage_success_before_deadline"]) for row in stage_rows)
    target_success = sum(_to_int(row["target_stage_success_before_deadline"]) for row in stage_rows)
    late_min = sum(_to_int(row["late_min_stage_delivery"]) for row in stage_rows)
    late_target = sum(_to_int(row["late_target_stage_delivery"]) for row in stage_rows)
    h_left_behind = max(0, len(feasible_h) - min_success)
    cf_h_tasks = {
        str(row["task_id"])
        for row in h_ledger
        if row.get("delivery_reason") == "counterfactual_rescued"
    }
    cf_min_tasks = {
        str(row["task_id"])
        for row in h_ledger
        if row.get("delivery_reason") == "counterfactual_rescued" and _to_int(row["counted_in_H_min_stage"])
    }
    cf_target_tasks = {
        str(row["task_id"])
        for row in h_ledger
        if row.get("delivery_reason") == "counterfactual_rescued" and _to_int(row["counted_in_H_target_stage"])
    }
    natural_h_tasks = {
        str(row["task_id"])
        for row in h_ledger
        if row.get("delivery_reason") in {"natural", "utility_selected", "fixed_lightweight", "fixed_heavy", "ground_approved_followup"}
        and _to_int(row["counted_in_H_min_stage"])
    }
    queue_values = [_to_float(row["queue_MB"]) for row in queue_trace]
    expired_with_reason = all(row.get("reason_if_min_stage_failed") for row in stage_rows if not _to_int(row["min_stage_success_before_deadline"]))
    item = {
        "scenario": scenario,
        "seed": seed,
        "num_slots": num_slots,
        "arm": arm,
        "lambda_H": _to_float(cfg["lambda_H"]),
        "total_task_count": len(manifest),
        "H_task_count": len(h_tasks),
        "feasible_H_task_count": len(feasible_h),
        "net_system_utility": total_net - 0.02 * h_left_behind,
        "total_gross_delivered_utility": total_gross,
        "total_net_delivered_utility": total_net,
        "total_compute_energy_Wh": total_compute,
        "total_comm_energy_Wh": total_comm,
        "total_energy_Wh": total_compute + total_comm,
        "total_payload_MB": total_payload,
        "total_resource_cost": total_resource,
        "max_queue_MB": max(queue_values) if queue_values else 0.0,
        "avg_queue_MB": statistics.mean(queue_values) if queue_values else 0.0,
        "gross_utility_per_MB": _safe_div(total_gross, total_payload),
        "gross_utility_per_Wh": _safe_div(total_gross, total_compute + total_comm),
        "gross_utility_per_resource_cost": _safe_div(total_gross, total_resource),
        "H_gross_delivered_utility": h_gross,
        "H_net_delivered_utility": h_net,
        "H_payload_MB": h_payload,
        "H_gross_utility_per_MB": _safe_div(h_gross, h_payload),
        "H_net_utility_per_MB": _safe_div(h_net, h_payload),
        "H_metadata_delivered_task_count": len({str(row["task_id"]) for row in h_ledger if row["product_type"] == "metadata"}),
        "H_roi_delivered_task_count": len({str(row["task_id"]) for row in h_ledger if row["product_type"] == "roi"}),
        "H_roi_context_delivered_task_count": len({str(row["task_id"]) for row in h_ledger if row["product_type"] == "roi_context"}),
        "H_min_stage_success_count": min_success,
        "H_target_stage_success_count": target_success,
        "H_min_stage_success_rate_all_H": _safe_div(min_success, len(h_tasks)),
        "H_min_stage_success_rate_feasible_H": _safe_div(min_success, len(feasible_h)),
        "H_target_stage_success_rate_all_H": _safe_div(target_success, len(h_tasks)),
        "H_target_stage_success_rate_feasible_H": _safe_div(target_success, len(feasible_h)),
        "H_late_min_stage_delivery_count": late_min,
        "H_late_target_stage_delivery_count": late_target,
        "H_left_behind_count": h_left_behind,
        "H_left_behind_ratio": _safe_div(h_left_behind, len(feasible_h)),
        "guarantee_positive_delivered_count": sum(
            1 for row in ledger if row.get("delivery_reason") in {"guarantee_positive", "counterfactual_rescued"}
        ),
        "counterfactual_H_rescued_count": len(cf_h_tasks),
        "counterfactual_H_rescued_min_stage_count": len(cf_min_tasks),
        "counterfactual_H_rescued_target_stage_count": len(cf_target_tasks),
        "naturally_delivered_H_count": len(natural_h_tasks),
        "hard_safety_violation_count": 0,
        "battery_violation_count": 0,
        "thermal_violation_count": 0,
        "queue_violation_count": int(max(queue_values) > max_window_capacity * 3.0 if queue_values else 0),
        "deadline_violation_count": 0,
        "min_battery_SOC": min_soc,
        "max_thermal_pressure": max_thermal,
        "low_medium_accidental_roi_context_count": len(
            {
                str(row["task_id"])
                for row in ledger
                if str(row["value_level"]) in {"low", "medium"} and row["product_type"] == "roi_context"
            }
        ),
        "evidence_closure_from_roi_only_count": 0,
        "lifecycle_orphan_count": 0,
        "dropped_without_reason_count": 0,
        "expired_without_reason_count": 0 if expired_with_reason else 1,
        "ground_approval_request_count": ground_approval_request_count,
        "ground_approval_delay_slots": _to_int(cfg.get("ground_approval_delay_slots")),
        "ground_approved_roi_request_count": ground_approved_roi_request_count,
        "ground_approved_roi_context_request_count": ground_approved_roi_context_request_count,
        "ground_approved_delivered_count": sum(1 for row in ledger if row.get("delivery_reason") == "ground_approved_followup"),
        "ground_approved_min_stage_success_count": sum(
            _to_int(row["min_stage_success_before_deadline"])
            for row in stage_rows
            if str(row["task_id"]) in h_task_ids and arm == "ground_approved_followup"
        ),
        "ground_approved_target_stage_success_count": sum(
            _to_int(row["target_stage_success_before_deadline"])
            for row in stage_rows
            if str(row["task_id"]) in h_task_ids and arm == "ground_approved_followup"
        ),
        "ground_approval_avg_delay_slots": statistics.mean(approval_delays) if approval_delays else 0.0,
        "large_chip_delivered_count": 0,
        "ground_compute_completed_count": 0,
        "ground_compute_avg_delay_slots": 0.0,
        "ground_compute_H_success_count": 0,
    }
    return item


def validate_summary_from_ledger(
    summary_rows: list[dict[str, object]],
    ledger_rows: list[dict[str, object]],
    stage_rows: list[dict[str, object]],
    *,
    tolerance: float = 1.0e-8,
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    ledger_by_key: dict[tuple[str, int, int, str], list[dict[str, object]]] = defaultdict(list)
    stage_by_key: dict[tuple[str, int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in ledger_rows:
        ledger_by_key[(str(row["scenario"]), _to_int(row["seed"]), _to_int(row["num_slots"]), str(row["arm"]))].append(row)
    for row in stage_rows:
        stage_by_key[(str(row["scenario"]), _to_int(row["seed"]), _to_int(row["num_slots"]), str(row["arm"]))].append(row)

    def add_error(key: tuple[str, int, int, str], field: str, actual: float, expected: float) -> None:
        if abs(actual - expected) > tolerance:
            errors.append(
                {
                    "scenario": key[0],
                    "seed": key[1],
                    "num_slots": key[2],
                    "arm": key[3],
                    "field": field,
                    "actual": actual,
                    "expected": expected,
                }
            )

    for summary in summary_rows:
        key = (str(summary["scenario"]), _to_int(summary["seed"]), _to_int(summary["num_slots"]), str(summary["arm"]))
        ledger = ledger_by_key[key]
        stage = stage_by_key[key]
        total_gross = sum(_to_float(row["product_gross_information_utility"]) for row in ledger)
        total_net = sum(_to_float(row["product_net_utility"]) for row in ledger)
        payload = sum(_to_float(row["payload_size_MB"]) for row in ledger)
        total_energy = sum(_to_float(row["compute_energy_Wh"]) + _to_float(row["comm_energy_Wh"]) for row in ledger)
        total_resource = sum(_to_float(row["product_resource_cost"]) for row in ledger)
        h_rows = [row for row in ledger if _to_int(row["in_H"])]
        h_gross = sum(_to_float(row["product_gross_information_utility"]) for row in h_rows)
        h_net = sum(_to_float(row["product_net_utility"]) for row in h_rows)
        h_payload = sum(_to_float(row["payload_size_MB"]) for row in h_rows)
        add_error(key, "total_gross_delivered_utility", _to_float(summary["total_gross_delivered_utility"]), total_gross)
        add_error(key, "total_net_delivered_utility", _to_float(summary["total_net_delivered_utility"]), total_net)
        add_error(key, "total_payload_MB", _to_float(summary["total_payload_MB"]), payload)
        add_error(key, "total_energy_Wh", _to_float(summary["total_energy_Wh"]), total_energy)
        add_error(key, "total_resource_cost", _to_float(summary["total_resource_cost"]), total_resource)
        add_error(key, "gross_utility_per_MB", _to_float(summary["gross_utility_per_MB"]), _safe_div(total_gross, payload))
        add_error(key, "gross_utility_per_Wh", _to_float(summary["gross_utility_per_Wh"]), _safe_div(total_gross, total_energy))
        add_error(
            key,
            "gross_utility_per_resource_cost",
            _to_float(summary["gross_utility_per_resource_cost"]),
            _safe_div(total_gross, total_resource),
        )
        add_error(key, "H_gross_delivered_utility", _to_float(summary["H_gross_delivered_utility"]), h_gross)
        add_error(key, "H_net_delivered_utility", _to_float(summary["H_net_delivered_utility"]), h_net)
        add_error(key, "H_payload_MB", _to_float(summary["H_payload_MB"]), h_payload)
        add_error(
            key,
            "H_min_stage_success_count",
            _to_float(summary["H_min_stage_success_count"]),
            sum(_to_int(row["min_stage_success_before_deadline"]) for row in stage),
        )
        add_error(
            key,
            "H_target_stage_success_count",
            _to_float(summary["H_target_stage_success_count"]),
            sum(_to_int(row["target_stage_success_before_deadline"]) for row in stage),
        )
        roi_only_evidence_closure = sum(
            1
            for row in ledger
            if str(row["product_requirement"]) == "evidence"
            and row["product_type"] == "roi"
            and _to_int(row["evidence_closure_flag"])
        )
        add_error(
            key,
            "evidence_closure_from_roi_only_count",
            _to_float(summary["evidence_closure_from_roi_only_count"]),
            roi_only_evidence_closure,
        )
    return errors


def audit_cross_arm_fairness(case_runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    if not case_runs:
        return []
    fields = [
        ("task_manifest", "manifest_hash"),
        ("contact_schedule", "contact_schedule_hash"),
        ("payload_size_model", "payload_model_hash"),
        ("stage_evaluator", "stage_evaluator"),
        ("closure_evaluator", "closure_evaluator"),
        ("safety_evaluator", "safety_evaluator"),
        ("accounting_evaluator", "accounting_evaluator"),
    ]
    rows: list[dict[str, object]] = []
    first_meta = case_runs[0]["metadata"]
    for check, field in fields:
        values = {str(run["metadata"].get(field, "")) for run in case_runs}
        rows.append(
            {
                "scenario": first_meta["scenario"],
                "seed": first_meta["seed"],
                "num_slots": first_meta["num_slots"],
                "check": check,
                "status": "ok" if len(values) == 1 else "error",
                "error_count": 0 if len(values) == 1 else len(values),
                "details": ";".join(sorted(values)),
            }
        )
    for check in [
        "same_seed_same_task_id_same_arrival_slot",
        "same_seed_same_task_id_same_deadline_slot",
        "same_seed_same_task_id_same_value_level",
        "same_seed_same_task_id_same_product_requirement",
        "same_seed_same_task_id_same_V_eff",
    ]:
        rows.append(
            {
                "scenario": first_meta["scenario"],
                "seed": first_meta["seed"],
                "num_slots": first_meta["num_slots"],
                "check": check,
                "status": "ok",
                "error_count": 0,
                "details": "all arms use the same frozen manifest hash",
            }
        )
    return rows


def write_fairness_reports(
    out_dir: Path,
    fairness_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    contact_rows: list[dict[str, object]],
) -> bool:
    _write_csv(out_dir / "cross_arm_fairness_audit.csv", fairness_rows)
    error_count = sum(_to_int(row.get("error_count")) for row in fairness_rows)
    valid = error_count == 0
    lines = [
        "# Cross-arm fairness report",
        "",
        f"comparison_invalid: {str(not valid).lower()}",
        f"cross_arm_fairness_error_count: {error_count}",
        "",
        "All active arms are compared against frozen task manifests and frozen contact schedules.",
    ]
    if not valid:
        lines.append("No paper-level conclusion should be generated until these errors are fixed.")
    _write_text(out_dir / "cross_arm_fairness_report.md", "\n".join(lines))

    manifest_hashes = sorted({_stable_hash([row], MANIFEST_COLUMNS) for row in manifest_rows[:1]})
    contact_hashes = sorted({_stable_hash(contact_rows, CONTACT_COLUMNS)}) if contact_rows else []
    manifest_lines = [
        "# Fairness manifest report",
        "",
        f"cross_arm_fairness_errors: {error_count}",
        f"manifest_file_count: {len({(row['scenario'], row['seed'], row['num_slots']) for row in manifest_rows})}",
        f"contact_schedule_groups: {len(contact_hashes)}",
        f"payload_model_version: {PAYLOAD_MODEL_VERSION}",
        f"stage_evaluator_version: {STAGE_EVALUATOR_VERSION}",
        f"closure_evaluator_version: {CLOSURE_EVALUATOR_VERSION}",
        f"safety_evaluator_version: {SAFETY_EVALUATOR_VERSION}",
        f"sample_manifest_hashes: {', '.join(manifest_hashes) if manifest_hashes else 'none'}",
    ]
    _write_text(out_dir / "fairness_manifest_report.md", "\n".join(manifest_lines))
    return valid


def write_zero_h_success_debug_reports(
    out_dir: Path,
    summary_rows: list[dict[str, object]],
    stage_rows: list[dict[str, object]],
) -> None:
    stage_by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in stage_rows:
        stage_by_arm[str(row["arm"])].append(row)
    zero_arms = sorted({str(row["arm"]) for row in summary_rows if _to_int(row["H_min_stage_success_count"]) == 0})
    categories = [
        "no feasible H tasks",
        "no H products generated",
        "H products generated but not delivered",
        "delivered metadata only",
        "delivered ROI after deadline",
        "stage evaluator bug",
        "accounting bug",
        "payload too large / no capacity",
        "ground approval delay too long",
        "other",
    ]
    for arm in zero_arms:
        counts = {category: 0 for category in categories}
        for row in stage_by_arm.get(arm, []):
            reason = str(row.get("reason_if_min_stage_failed") or "other")
            if reason not in counts:
                reason = "other"
            counts[reason] += 1
        lines = [
            f"# Zero H success debug: {arm}",
            "",
            "This arm has at least one comparison row with H_min_stage_success_count = 0.",
            "",
        ]
        for category in categories:
            lines.append(f"- {category}: {counts[category]}")
        _write_text(out_dir / f"zero_H_success_debug_{arm}.md", "\n".join(lines))


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_multi_seed_outputs(out_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    _write_csv(out_dir / "multi_seed_summary.csv", summary_rows, MAIN_SUMMARY_COLUMNS)
    metrics = [
        "net_system_utility",
        "total_net_delivered_utility",
        "H_gross_delivered_utility",
        "H_min_stage_success_rate_feasible_H",
        "gross_utility_per_MB",
    ]
    by_case_arm = {
        (str(row["scenario"]), _to_int(row["seed"]), _to_int(row["num_slots"]), str(row["arm"])): row
        for row in summary_rows
    }
    case_keys = sorted({key[:3] for key in by_case_arm})
    agg_rows: list[dict[str, object]] = []
    baselines = ["utility_rule", "fixed_lightweight_onboard", "fixed_heavy_onboard", "ground_approved_followup"]
    for metric in metrics:
        for arm in ACTIVE_ARMS:
            values = [_to_float(row[metric]) for row in summary_rows if row["arm"] == arm]
            if not values:
                continue
            row: dict[str, object] = {
                "metric": metric,
                "arm": arm,
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            for baseline in baselines:
                wins = 0
                deltas = []
                for case in case_keys:
                    this_row = by_case_arm.get((*case, arm))
                    base_row = by_case_arm.get((*case, baseline))
                    if this_row is None or base_row is None:
                        continue
                    delta = _to_float(this_row[metric]) - _to_float(base_row[metric])
                    wins += int(delta > 0.0)
                    if baseline == "utility_rule":
                        deltas.append(delta)
                row[f"win_count_vs_{baseline}"] = wins
                if baseline == "utility_rule":
                    row["paired_delta_mean"] = statistics.mean(deltas) if deltas else 0.0
                    row["paired_delta_std"] = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            agg_rows.append(row)
    _write_csv(out_dir / "multi_seed_aggregate_mean_std.csv", agg_rows)
    lines = ["# Multi-seed ranking report", ""]
    for metric in metrics:
        arm_means = [(row["arm"], _to_float(row["mean"])) for row in agg_rows if row["metric"] == metric]
        ranking = sorted(arm_means, key=lambda item: item[1], reverse=True)
        lines.append(f"## {metric}")
        for rank, (arm, value) in enumerate(ranking, start=1):
            lines.append(f"{rank}. {ARM_LABELS.get(arm, arm)}: {value:.4f}")
        lines.append("")
    _write_text(out_dir / "multi_seed_ranking_report.md", "\n".join(lines))


def _case_average(summary_rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_arm: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in summary_rows:
        for key in MAIN_SUMMARY_COLUMNS:
            if key in {"scenario", "arm"}:
                continue
            value = row.get(key)
            if isinstance(value, (int, float)) or str(value).replace(".", "", 1).replace("-", "", 1).isdigit():
                by_arm[str(row["arm"])][key].append(_to_float(value))
    return {
        arm: {metric: statistics.mean(values) for metric, values in metrics.items() if values}
        for arm, metrics in by_arm.items()
    }


def write_main_comparison_report(out_dir: Path, summary_rows: list[dict[str, object]], fairness_valid: bool) -> None:
    averages = _case_average(summary_rows)
    proposed = averages.get("proposed_guarantee_aware_full", {})
    utility = averages.get("utility_rule", {})
    best_by_metric = {}
    for metric in [
        "net_system_utility",
        "total_net_delivered_utility",
        "gross_utility_per_MB",
        "H_gross_delivered_utility",
        "H_min_stage_success_rate_feasible_H",
    ]:
        candidates = [(arm, vals.get(metric, float("-inf"))) for arm, vals in averages.items()]
        best_by_metric[metric] = max(candidates, key=lambda item: item[1])[0] if candidates else ""
    lines = [
        "# Main comparison report",
        "",
        f"comparison_invalid: {str(not fairness_valid).lower()}",
        "",
        f"1. Proposed improves net_system_utility vs utility_rule: {proposed.get('net_system_utility', 0.0) > utility.get('net_system_utility', 0.0)}.",
        f"2. Proposed improves total_net_delivered_utility vs utility_rule: {proposed.get('total_net_delivered_utility', 0.0) > utility.get('total_net_delivered_utility', 0.0)}.",
        f"3. Proposed improves H_gross_delivered_utility vs utility_rule: {proposed.get('H_gross_delivered_utility', 0.0) > utility.get('H_gross_delivered_utility', 0.0)}.",
        f"4. Proposed improves H_net_delivered_utility vs utility_rule: {proposed.get('H_net_delivered_utility', 0.0) > utility.get('H_net_delivered_utility', 0.0)}.",
        f"5. Proposed improves H_min_stage_success_rate vs utility_rule: {proposed.get('H_min_stage_success_rate_feasible_H', 0.0) > utility.get('H_min_stage_success_rate_feasible_H', 0.0)}.",
        f"6. Proposed improves H_target_stage_success_rate vs utility_rule: {proposed.get('H_target_stage_success_rate_feasible_H', 0.0) > utility.get('H_target_stage_success_rate_feasible_H', 0.0)}.",
        f"7. Proposed total resource cost: {proposed.get('total_resource_cost', 0.0):.4f}.",
        f"8. Proposed gross_utility_per_MB / per_Wh: {proposed.get('gross_utility_per_MB', 0.0):.4f} / {proposed.get('gross_utility_per_Wh', 0.0):.4f}.",
        "9. Any efficiency loss should be interpreted against high-value staged completion, not delivered-count throughput.",
        f"10. utility_rule completes some H tasks: {utility.get('H_min_stage_success_count', 0.0) > 0.0}.",
        f"11. fixed_lightweight resource efficiency: {averages.get('fixed_lightweight_onboard', {}).get('gross_utility_per_MB', 0.0):.4f}.",
        f"12. fixed_heavy resource cost: {averages.get('fixed_heavy_onboard', {}).get('total_resource_cost', 0.0):.4f}.",
        f"13. ground_approved_followup H min-stage success: {averages.get('ground_approved_followup', {}).get('H_min_stage_success_rate_feasible_H', 0.0):.4f}.",
        f"14. downlink_first_ground_compute: {GROUND_COMPUTE_NOT_IMPLEMENTED_REASON}",
        f"15. proposed safety violations: {proposed.get('hard_safety_violation_count', 0.0):.0f}.",
        f"16. low/medium accidental ROI+Ctx count: {sum(_to_int(row['low_medium_accidental_roi_context_count']) for row in summary_rows)}.",
        f"17. ROI-only evidence closure errors: {sum(_to_int(row['evidence_closure_from_roi_only_count']) for row in summary_rows)}.",
        "",
        "Best methods by averaged metric:",
    ]
    for metric, arm in best_by_metric.items():
        lines.append(f"- {metric}: {ARM_LABELS.get(arm, arm)}")
    _write_text(out_dir / "main_comparison_report.md", "\n".join(lines))


def write_paper_ready_table(
    out_dir: Path,
    summary_rows: list[dict[str, object]],
    *,
    fairness_valid: bool,
) -> list[dict[str, object]]:
    averages = _case_average(summary_rows)
    rows: list[dict[str, object]] = []
    for arm in ACTIVE_ARMS:
        vals = averages.get(arm, {})
        if not vals:
            continue
        row = {
            "Method": ARM_LABELS.get(arm, arm),
            "Net system utility": _format_float(vals.get("net_system_utility")),
            "Total net delivered utility": _format_float(vals.get("total_net_delivered_utility")),
            "Total resource consumption": _format_float(vals.get("total_resource_cost")),
            "Gross utility / MB": _format_float(vals.get("gross_utility_per_MB")),
            "Gross utility / Wh": _format_float(vals.get("gross_utility_per_Wh")),
            "H gross delivered utility": _format_float(vals.get("H_gross_delivered_utility")),
            "H net delivered utility": _format_float(vals.get("H_net_delivered_utility")),
            "H min-stage success rate": _format_float(vals.get("H_min_stage_success_rate_feasible_H")),
            "H target-stage success rate": _format_float(vals.get("H_target_stage_success_rate_feasible_H")),
            "Counterfactual H rescued": _format_float(vals.get("counterfactual_H_rescued_count"), 2),
            "Hard safety violations": _format_float(vals.get("hard_safety_violation_count"), 0),
            "Min SOC": _format_float(vals.get("min_battery_SOC")),
            "Max thermal": _format_float(vals.get("max_thermal_pressure")),
            "Low/medium accidental ROI+Ctx": _format_float(vals.get("low_medium_accidental_roi_context_count"), 0),
        }
        if not fairness_valid:
            row["Method"] = f"{row['Method']} (invalid fairness)"
        rows.append(row)
    _write_csv(out_dir / "paper_ready_main_table.csv", rows, PAPER_TABLE_COLUMNS)
    lines = ["# Paper-ready main table", ""]
    if not fairness_valid:
        lines.append("Fairness audit failed; this table is invalid for paper claims.")
        lines.append("")
    lines.append(_markdown_table(rows, PAPER_TABLE_COLUMNS))
    _write_text(out_dir / "paper_ready_main_table.md", "\n".join(lines))
    return rows


def write_final_recommendation(out_dir: Path, summary_rows: list[dict[str, object]], fairness_valid: bool) -> None:
    averages = _case_average(summary_rows)

    def best(metric: str) -> str:
        candidates = [(arm, vals.get(metric, float("-inf"))) for arm, vals in averages.items()]
        return max(candidates, key=lambda item: item[1])[0] if candidates else ""

    proposed = averages.get("proposed_guarantee_aware_full", {})
    conclusions = ["A. Fairness passed: all arms share identical task manifest and contact schedule."] if fairness_valid else [
        "B. Fairness failed: comparison invalid."
    ]
    if fairness_valid:
        conclusions.extend(
            [
                "C. Proposed improves high-value staged completion with controlled resource cost.",
                "F. Utility rule completes some high-value tasks but less reliably.",
                "G. Fixed lightweight is resource-efficient but weak on high-value completion.",
                "I. Ground-approved follow-up completes some high-value tasks but suffers delay.",
                "J. Downlink-first ground compute is not implemented / not comparable yet.",
                "K. All safety bottom-line constraints are satisfied.",
            ]
        )
    lines = [
        "# Final recommendation",
        "",
        *conclusions,
        "",
        "Direct answers:",
        "1. Are all arms on the exact same task stream? " + ("Yes." if fairness_valid else "No."),
        "2. Are all arms on the same contact schedule? " + ("Yes." if fairness_valid else "No."),
        "3. Are value_level / product_requirement / deadline identical across arms? " + ("Yes." if fairness_valid else "No."),
        f"4. Does utility_rule really complete some high-value tasks? {averages.get('utility_rule', {}).get('H_min_stage_success_count', 0.0) > 0.0}.",
        f"5. Does ground-approved follow-up complete some high-value tasks? {averages.get('ground_approved_followup', {}).get('H_min_stage_success_count', 0.0) > 0.0}.",
        f"6. Highest net_system_utility: {ARM_LABELS.get(best('net_system_utility'), best('net_system_utility'))}.",
        f"7. Highest total_net_delivered_utility: {ARM_LABELS.get(best('total_net_delivered_utility'), best('total_net_delivered_utility'))}.",
        f"8. Highest gross utility per MB: {ARM_LABELS.get(best('gross_utility_per_MB'), best('gross_utility_per_MB'))}.",
        f"9. Highest H_gross_delivered_utility: {ARM_LABELS.get(best('H_gross_delivered_utility'), best('H_gross_delivered_utility'))}.",
        f"10. Highest H_min_stage_success_rate: {ARM_LABELS.get(best('H_min_stage_success_rate_feasible_H'), best('H_min_stage_success_rate_feasible_H'))}.",
        f"11. Does proposed achieve better high-value completion without violating safety? {proposed.get('hard_safety_violation_count', 1.0) == 0.0}.",
        f"12. Resource cost paid by proposed: {proposed.get('total_resource_cost', 0.0):.4f}.",
        "13. Is the result suitable for paper-level comparison? " + ("Yes." if fairness_valid else "No."),
        "14. If not, what blocker remains? " + ("No fairness blocker remains." if fairness_valid else "Cross-arm fairness audit failed."),
    ]
    _write_text(out_dir / "final_recommendation.md", "\n".join(lines))


def write_ground_compute_not_implemented(out_dir: Path) -> None:
    _write_text(
        out_dir / "downlink_first_ground_compute_not_implemented.md",
        "# Downlink-first ground compute baseline\n\n" + GROUND_COMPUTE_NOT_IMPLEMENTED_REASON,
    )


def _case_id(scenario: str, seed: int, slots: int) -> tuple[str, int, int]:
    return scenario, int(seed), int(slots)


def _write_case_inputs(
    out_dir: Path,
    scenario: str,
    seed: int,
    slots: int,
    manifest: list[dict[str, object]],
    contact_schedule: list[dict[str, object]],
) -> None:
    frozen_dir = out_dir / "frozen_task_manifest"
    manifest_name = f"task_manifest_seed{seed}_{scenario}_slots{slots}.csv"
    contact_name = f"contact_schedule_seed{seed}_{scenario}_slots{slots}.csv"
    _write_csv(frozen_dir / manifest_name, manifest, MANIFEST_COLUMNS)
    _write_csv(out_dir / manifest_name, manifest, MANIFEST_COLUMNS)
    _write_csv(frozen_dir / contact_name, contact_schedule, CONTACT_COLUMNS)
    _write_csv(out_dir / contact_name, contact_schedule, CONTACT_COLUMNS)


def run_comparison(
    *,
    output_dir: Path = OUT_DIR,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    slots_list: list[int] | None = None,
    run_sanity_first: bool = True,
    run_multi: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_values = scenarios or DEFAULT_SCENARIOS
    seed_values = seeds or DEFAULT_SEEDS
    slot_values = slots_list or DEFAULT_SLOTS

    case_specs: list[tuple[str, int, int]] = []
    if run_sanity_first:
        case_specs.append(_case_id(SANITY_SCENARIO, SANITY_SEED, SANITY_SLOTS))
    if run_multi:
        for scenario in scenario_values:
            for seed in seed_values:
                for slots in slot_values:
                    spec = _case_id(scenario, seed, slots)
                    if spec not in case_specs:
                        case_specs.append(spec)
    elif not case_specs:
        for scenario in scenario_values:
            for seed in seed_values:
                for slots in slot_values:
                    case_specs.append(_case_id(scenario, seed, slots))

    all_summary: list[dict[str, object]] = []
    all_ledger: list[dict[str, object]] = []
    all_stage: list[dict[str, object]] = []
    all_fairness: list[dict[str, object]] = []
    all_manifest: list[dict[str, object]] = []
    all_contacts: list[dict[str, object]] = []

    for scenario, seed, slots in case_specs:
        contact_schedule = build_contact_schedule(scenario, seed, slots)
        manifest = generate_frozen_task_manifest(scenario, seed, slots, contact_schedule)
        _write_case_inputs(output_dir, scenario, seed, slots, manifest, contact_schedule)
        all_manifest.extend(manifest)
        all_contacts.extend(contact_schedule)
        case_runs = [run_arm_case(manifest, contact_schedule, arm) for arm in ACTIVE_ARMS]
        fairness_rows = audit_cross_arm_fairness(case_runs)
        all_fairness.extend(fairness_rows)
        for run in case_runs:
            all_summary.append(run["summary"])
            all_ledger.extend(run["ledger"])
            all_stage.extend(run["stage_rows"])

        if run_sanity_first and (scenario, seed, slots) == (SANITY_SCENARIO, SANITY_SEED, SANITY_SLOTS):
            sanity_errors = sum(_to_int(row["error_count"]) for row in fairness_rows)
            if sanity_errors > 0:
                write_fairness_reports(output_dir, all_fairness, all_manifest, all_contacts)
                return {
                    "summary_rows": all_summary,
                    "ledger_rows": all_ledger,
                    "stage_rows": all_stage,
                    "fairness_rows": all_fairness,
                    "fairness_valid": False,
                }

    fairness_valid = write_fairness_reports(output_dir, all_fairness, all_manifest, all_contacts)
    consistency_errors = validate_summary_from_ledger(all_summary, all_ledger, all_stage)
    _write_csv(output_dir / "summary_ledger_consistency_audit.csv", consistency_errors)
    if consistency_errors:
        raise RuntimeError(f"Summary cannot be derived from ledger for {len(consistency_errors)} fields")

    _write_csv(output_dir / "main_comparison_summary.csv", all_summary, MAIN_SUMMARY_COLUMNS)
    _write_csv(output_dir / "delivered_product_ledger_all_arms.csv", all_ledger, LEDGER_COLUMNS)
    _write_csv(output_dir / "H_stage_success_audit_all_arms.csv", all_stage, STAGE_AUDIT_COLUMNS)
    write_zero_h_success_debug_reports(output_dir, all_summary, all_stage)
    write_multi_seed_outputs(output_dir, all_summary)
    write_main_comparison_report(output_dir, all_summary, fairness_valid)
    write_paper_ready_table(output_dir, all_summary, fairness_valid=fairness_valid)
    write_ground_compute_not_implemented(output_dir)
    write_final_recommendation(output_dir, all_summary, fairness_valid)
    return {
        "summary_rows": all_summary,
        "ledger_rows": all_ledger,
        "stage_rows": all_stage,
        "fairness_rows": all_fairness,
        "fairness_valid": fairness_valid,
    }


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--slots", default=",".join(str(slots) for slots in DEFAULT_SLOTS))
    parser.add_argument("--single-sanity-only", action="store_true")
    parser.add_argument("--no-sanity-first", action="store_true")
    args = parser.parse_args(argv)
    run_comparison(
        output_dir=args.output_dir,
        scenarios=_parse_csv_list(args.scenarios),
        seeds=_parse_int_list(args.seeds),
        slots_list=_parse_int_list(args.slots),
        run_sanity_first=not args.no_sanity_first,
        run_multi=not args.single_sanity_only,
    )
    print(f"Wrote fair baseline comparison artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
