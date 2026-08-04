"""Passive Phase-1 reconciliation and high-value comparison helpers."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable


REQUIREMENT_QUALITY_THRESHOLDS = {
    "alert": 0.95,
    "verification": 0.96,
    "context": 0.92,
    "evidence": 0.95,
}
ALL_FULL_FRAME_CAPABILITIES = frozenset(REQUIREMENT_QUALITY_THRESHOLDS)
HV_LEVELS = {"high", "urgent", "high_value", "emergency"}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _b(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _capabilities(task: dict[str, Any]) -> frozenset[str]:
    raw = task.get("full_frame_information_capabilities")
    if raw in (None, ""):
        return ALL_FULL_FRAME_CAPABILITIES
    if isinstance(raw, str):
        return frozenset(part.strip().lower() for part in raw.split(";") if part.strip())
    return frozenset(str(part).strip().lower() for part in raw)


@dataclass(frozen=True)
class B1FrameEvaluation:
    full_frame_generated: bool
    full_frame_received: bool
    ground_processing_success: bool
    frame_contains_required_information: bool
    retained_compression_quality: float
    quality_threshold: float
    quality_pass: bool
    requirement_satisfied: bool
    on_time_success: bool
    failure_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_b1_frame(record: dict[str, Any], task: dict[str, Any]) -> B1FrameEvaluation:
    """Apply explicit full-frame, processing, information, and quality gates."""

    state = str(record.get("state", "")).strip().lower()
    generated = state not in {"compression_pending", "compressing", "blocked"}
    received = state == "ground_result_available" and _b(record.get("ground_result_available", 0))
    completion = _f(record.get("ground_result_available_time_s"), -1.0)
    processing = received and completion >= 0.0 and _f(record.get("ground_decode_latency_s"), -1.0) >= 0.0 and _f(record.get("ground_inference_latency_s"), -1.0) >= 0.0
    requirement = str(task.get("product_requirement", "verification")).strip().lower()
    contains = requirement in _capabilities(task)
    retained = _f(record.get("reconstructed_image_quality_proxy"), 0.0)
    threshold = REQUIREMENT_QUALITY_THRESHOLDS.get(requirement, REQUIREMENT_QUALITY_THRESHOLDS["verification"])
    quality_pass = retained + 1.0e-12 >= threshold
    satisfied = received and processing and contains and quality_pass
    deadline = _f(task.get("deadline_time_s"), float("inf"))
    on_time = satisfied and completion <= deadline + 1.0e-9
    if not generated:
        reason = "compression_incomplete"
    elif not received:
        reason = "transmission_incomplete"
    elif not processing:
        reason = "ground_processing_incomplete"
    elif not contains:
        reason = "lower_than_required_product"
    elif not quality_pass:
        reason = "quality_gap"
    elif not on_time:
        reason = "ground_processing_late"
    else:
        reason = "success"
    return B1FrameEvaluation(generated, received, processing, contains, retained, threshold, quality_pass, satisfied, on_time, reason)


def is_hv_all(row: dict[str, Any]) -> bool:
    return str(row.get("value_level", "")).strip().lower() in HV_LEVELS


def is_hv_target(row: dict[str, Any]) -> bool:
    return is_hv_all(row) and str(row.get("product_requirement", "")).strip().lower() in {"context", "evidence"}


def classify_outcome(row: dict[str, Any]) -> str:
    if _b(row.get("on_time_task_success")):
        return "on_time_requirement_success"
    if _b(row.get("requirement_satisfied")):
        return "late_requirement_satisfied"
    if _b(row.get("ground_completed")) or _b(row.get("partial_ground_delivery")):
        return "partial_only"
    return "no_ground"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def matched_latency(
    left_rows: Iterable[dict[str, Any]],
    right_rows: Iterable[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    bootstrap_seed: int = 2026,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    left = {str(row["task_id"]): row for row in left_rows if predicate(row)}
    right = {str(row["task_id"]): row for row in right_rows if predicate(row)}
    ids = sorted(left.keys() & right.keys())
    differences = [_f(left[task_id].get("capture_to_ground_latency_s")) - _f(right[task_id].get("capture_to_ground_latency_s")) for task_id in ids]
    faster = sum(value < -1.0e-9 for value in differences)
    slower = sum(value > 1.0e-9 for value in differences)
    equal = len(differences) - faster - slower
    ci_low = ci_high = 0.0
    if differences:
        rng = random.Random(bootstrap_seed)
        means = sorted(_mean([rng.choice(differences) for _ in differences]) for _ in range(bootstrap_samples))
        ci_low, ci_high = _percentile(means, 0.025), _percentile(means, 0.975)
    return {
        "matched_task_count": len(ids),
        "matched_task_ids": ";".join(ids),
        "mean_paired_latency_difference_s": _mean(differences),
        "median_paired_latency_difference_s": statistics.median(differences) if differences else 0.0,
        "p90_paired_latency_difference_s": _percentile(differences, 0.90),
        "p95_paired_latency_difference_s": _percentile(differences, 0.95),
        "left_faster_count": faster,
        "equal_count": equal,
        "left_slower_count": slower,
        "bootstrap_mean_difference_ci95_low_s": ci_low,
        "bootstrap_mean_difference_ci95_high_s": ci_high,
    }


def validate_method_rows(method: str, rows: list[dict[str, Any]]) -> None:
    ids = [str(row.get("task_id", "")) for row in rows]
    if len(rows) != 370 or len(set(ids)) != 370 or "" in ids:
        raise ValueError(f"{method}: expected 370 unique task IDs")
    for row in rows:
        if _b(row.get("on_time_task_success")) and not _b(row.get("requirement_satisfied")):
            raise ValueError(f"{method}: on-time closure without requirement satisfaction")
    if method == "PROP-NO-STAGE":
        closures = sum(_b(row.get("requirement_satisfied")) for row in rows)
        if closures != 72:
            raise ValueError(f"authoritative PROP-NO-STAGE closure count is 72, got {closures}")


def summarize_group(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    system_summary: dict[str, Any],
    direct_ids: set[str] | None = None,
    staged_ids: set[str] | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    arrived = len(selected)
    ground = [row for row in selected if _b(row.get("ground_completed"))]
    closures = [row for row in selected if _b(row.get("requirement_satisfied"))]
    on_time = [row for row in selected if _b(row.get("on_time_task_success"))]
    categories = {name: 0 for name in ("on_time_requirement_success", "late_requirement_satisfied", "partial_only", "no_ground")}
    for row in selected:
        categories[classify_outcome(row)] += 1
    all_lat = [_f(row.get("capture_to_ground_latency_s")) for row in closures]
    on_lat = [_f(row.get("capture_to_ground_latency_s")) for row in on_time]
    slack = [_f(row.get("deadline_time_s")) - _f(row.get("requirement_closure_time_s", row.get("ground_result_available_time_s"))) for row in closures]
    late_miss = [-value for value in slack if value < 0.0]
    quality_ground = [_f(row.get("ground_result_quality")) for row in ground]
    quality_closure = [_f(row.get("ground_result_quality")) for row in closures]
    energy = _f(system_summary.get("controllable_mission_energy_wh"))
    delivered_mb = _f(system_summary.get("delivered_mb"))
    group_success = len(on_time)
    gross = sum(_f(row.get("gross_ground_utility")) for row in selected)
    all_gross = sum(_f(row.get("gross_ground_utility")) for row in rows)
    direct_ids = direct_ids or set()
    staged_ids = staged_ids or set()
    closure_ids = {str(row["task_id"]) for row in closures}
    return {
        "arrived": arrived,
        "any_ground_product": len(ground),
        "partial_only": categories["partial_only"],
        "requirement_satisfied": len(closures),
        "on_time_requirement_success": len(on_time),
        "late_requirement_satisfied": categories["late_requirement_satisfied"],
        "no_ground": categories["no_ground"],
        "direct_closures": len(closure_ids & direct_ids) if direct_ids or staged_ids else "N/A",
        "staged_closures": len(closure_ids & staged_ids) if direct_ids or staged_ids else "N/A",
        "ground_coverage_rate": len(ground) / max(arrived, 1),
        "requirement_closure_rate": len(closures) / max(arrived, 1),
        "on_time_closure_rate": len(on_time) / max(arrived, 1),
        "closure_per_any_ground": len(closures) / max(len(ground), 1),
        "oracle_feasible_normalized_success": "pending_phase_2_oracle",
        "conditional_quality_ground_received": _mean(quality_ground),
        "arrival_normalized_ground_quality": sum(quality_ground) / max(arrived, 1),
        "quality_among_requirement_closures": _mean(quality_closure),
        "average_information_sufficiency": _mean([_f(row.get("ground_information_sufficiency")) for row in ground]),
        "mean_closure_latency_s": _mean(all_lat),
        "median_closure_latency_s": statistics.median(all_lat) if all_lat else 0.0,
        "p90_closure_latency_s": _percentile(all_lat, 0.90),
        "p95_closure_latency_s": _percentile(all_lat, 0.95),
        "mean_on_time_closure_latency_s": _mean(on_lat),
        "median_on_time_closure_latency_s": statistics.median(on_lat) if on_lat else 0.0,
        "p95_on_time_closure_latency_s": _percentile(on_lat, 0.95),
        "mean_deadline_slack_at_completion_s": _mean(slack),
        "mean_deadline_miss_for_late_closures_s": _mean(late_miss),
        "partial_ground_utility": sum(_f(row.get("partial_ground_gross_utility")) for row in selected),
        "closure_ground_utility": sum(_f(row.get("gross_ground_utility")) for row in closures),
        "common_gross_utility": gross,
        "common_net_utility_method_level": _f(system_summary.get("common_ground_net_utility")),
        "group_gross_utility_share": gross / max(all_gross, 1.0e-12),
        "controllable_mission_energy_wh": energy,
        "delivered_mb_method_level": delivered_mb,
        "system_energy_per_G_success_wh": energy / max(group_success, 1),
        "system_MB_per_G_success": delivered_mb / max(group_success, 1),
        "efficiency_denominator_G_on_time_success": group_success,
        "efficiency_scope": "system_level_not_task_attribution",
        "energy_per_HV_success_system_level_wh": energy / max(group_success, 1),
        "mb_per_HV_success_system_level": delivered_mb / max(group_success, 1),
        "gross_utility_per_wh": gross / max(energy, 1.0e-12),
        "gross_utility_per_mb": gross / max(delivered_mb, 1.0e-12),
    }
