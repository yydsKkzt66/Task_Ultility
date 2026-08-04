"""Report-only lifecycle and time shadow metrics from trajectory rows."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median
from typing import Any

from src.reporting.lifecycle_taxonomy import (
    is_direct_roi_or_context,
    is_metadata_only,
    normalize_product_type,
    stage_credit_for_product,
)


EPSILON = 1.0e-9

SHADOW_WEIGHT_SETS = {
    "balanced": {"alert": 0.30, "confirmation": 0.30, "evidence": 0.40},
    "alert_heavy": {"alert": 0.50, "confirmation": 0.25, "evidence": 0.25},
    "evidence_heavy": {"alert": 0.20, "confirmation": 0.30, "evidence": 0.50},
}

DEFAULT_DECAY = {"alert": 1.0, "confirmation": 1.0, "evidence": 1.0}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _valid_task_id(row: dict[str, Any]) -> bool:
    task_id = str(row.get("task_id", "")).strip()
    return bool(task_id) and task_id.lower() not in {"none", "idle"}


def _arrival_time(row: dict[str, Any]) -> float:
    value = to_float(row.get("task_arrival_time_s"), default=-1.0)
    if value >= 0.0:
        return value
    return to_float(row.get("t"), default=to_float(row.get("time_s"), 0.0))


def _deadline_time(row: dict[str, Any]) -> float:
    value = to_float(row.get("deadline_time_s"), default=-1.0)
    if value >= 0.0:
        return value
    return _arrival_time(row) + to_float(row.get("deadline_s"), default=0.0)


def _task_value(row: dict[str, Any]) -> float:
    if "V_cal" in row and str(row.get("V_cal", "")).strip() != "":
        return to_float(row.get("V_cal"))
    if "V_t" in row and str(row.get("V_t", "")).strip() != "":
        return to_float(row.get("V_t"))
    return to_float(row.get("task_value"))


def _min_positive(values: list[float]) -> float:
    positives = [value for value in values if value >= 0.0]
    return min(positives) if positives else -1.0


def _event_product(row: dict[str, Any]) -> str:
    product = normalize_product_type(row.get("selected_product_type", ""))
    if product:
        return product
    return normalize_product_type(row.get("selected_payload", row.get("payload_policy", "")))


def _delivered_before_deadline(row: dict[str, Any]) -> bool:
    if str(row.get("delivery_status", "")).strip().lower() != "delivered":
        return False
    delivery_time = to_float(row.get("payload_delivery_time_s"), default=-1.0)
    if delivery_time < 0.0:
        return False
    return delivery_time <= _deadline_time(row) + EPSILON


def build_per_task_lifecycle_rows(
    trajectory_rows: list[dict[str, Any]],
    *,
    seed: int,
    variant: str,
    router_name: str,
) -> list[dict[str, Any]]:
    """Reconstruct lifecycle shadow rows from trajectory CSV rows."""

    tasks: dict[str, dict[str, Any]] = {}
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in trajectory_rows:
        if not _valid_task_id(row):
            continue
        task_id = str(row["task_id"]).strip()
        if task_id not in tasks:
            tasks[task_id] = {
                "seed": seed,
                "variant": variant,
                "router_name": router_name,
                "task_id": task_id,
                "task_value": _task_value(row),
                "deadline_s": to_float(row.get("deadline_s")),
                "arrival_time_s": _arrival_time(row),
                "deadline_time_s": _deadline_time(row),
                "task_class": row.get("task_class", ""),
                "first_selected_action": row.get("raw_action", row.get("selected_action", "")),
                "first_selected_payload": row.get("selected_payload", row.get("payload_policy", "")),
                "canonical_effective_utility_sum": 0.0,
                "canonical_delivered_utility_sum": 0.0,
            }
        tasks[task_id]["canonical_effective_utility_sum"] += to_float(row.get("effective_utility"))
        tasks[task_id]["canonical_delivered_utility_sum"] += to_float(row.get("delivered_only_incremental_utility"))
        if _delivered_before_deadline(row):
            events[task_id].append(
                {
                    "product_type": _event_product(row),
                    "delivery_time_s": to_float(row.get("payload_delivery_time_s")),
                    "selected_payload_plan": row.get("selected_payload_plan", ""),
                }
            )

    per_task: list[dict[str, Any]] = []
    for task_id, base in sorted(tasks.items(), key=lambda item: item[1]["arrival_time_s"]):
        arrival = float(base["arrival_time_s"])
        deadline_s = max(float(base["deadline_s"]), EPSILON)
        deadline_time = float(base["deadline_time_s"])
        stage_times = {"alert": -1.0, "confirmation": -1.0, "evidence": -1.0}
        delivered_products: list[str] = []
        metadata_alert = 0
        direct_roi_alert_confirmation = 0
        for event in sorted(events.get(task_id, []), key=lambda item: float(item["delivery_time_s"])):
            product = normalize_product_type(event["product_type"])
            delivery_time = float(event["delivery_time_s"])
            if delivery_time > deadline_time + EPSILON:
                continue
            delivered_products.append(product)
            credit = stage_credit_for_product(product)
            if credit.alert and stage_times["alert"] < 0.0:
                stage_times["alert"] = delivery_time
            if credit.confirmation and stage_times["confirmation"] < 0.0:
                stage_times["confirmation"] = delivery_time
            if credit.evidence and stage_times["evidence"] < 0.0:
                stage_times["evidence"] = delivery_time
            metadata_alert = max(metadata_alert, int(is_metadata_only(product) and credit.alert))
            direct_roi_alert_confirmation = max(
                direct_roi_alert_confirmation,
                int(is_direct_roi_or_context(product) and credit.alert and credit.confirmation),
            )

        stage_latencies = {
            stage: (-1.0 if time_s < 0.0 else max(0.0, time_s - arrival)) for stage, time_s in stage_times.items()
        }
        stage_indicators = {stage: int(time_s >= 0.0) for stage, time_s in stage_times.items()}
        shadow_values: dict[str, float] = {}
        for name, weights in SHADOW_WEIGHT_SETS.items():
            total = 0.0
            for stage, weight in weights.items():
                latency = stage_latencies[stage]
                if latency >= 0.0:
                    phi = math.exp(-DEFAULT_DECAY[stage] * latency / max(deadline_s, EPSILON))
                    total += weight * phi
            shadow_values[f"lifecycle_shadow_utility_{name}"] = float(base["task_value"]) * total
        first_actionable_latency = stage_latencies["alert"]
        time_shadow = (
            float(base["task_value"]) * math.exp(-first_actionable_latency / max(deadline_s, EPSILON))
            if first_actionable_latency >= 0.0
            else 0.0
        )
        per_task.append(
            {
                **base,
                "delivered_products_shadow": ";".join(delivered_products),
                "alert_achieved": stage_indicators["alert"],
                "confirmation_achieved": stage_indicators["confirmation"],
                "evidence_achieved": stage_indicators["evidence"],
                "alert_latency_s": stage_latencies["alert"],
                "confirmation_latency_s": stage_latencies["confirmation"],
                "evidence_latency_s": stage_latencies["evidence"],
                "first_actionable_latency_s": first_actionable_latency,
                "expired_without_alert": int(stage_indicators["alert"] == 0),
                "metadata_only_alert": int(metadata_alert and not stage_indicators["confirmation"]),
                "direct_roi_alert_confirmation": direct_roi_alert_confirmation,
                "evidence_closure": stage_indicators["evidence"],
                "time_shadow_utility": time_shadow,
                **shadow_values,
            }
        )
    return per_task


def summarize_lifecycle_rows(per_task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(per_task_rows), 1)

    def achieved_latency(stage: str) -> list[float]:
        key = f"{stage}_latency_s"
        return [to_float(row.get(key)) for row in per_task_rows if to_float(row.get(key), -1.0) >= 0.0]

    alert_latencies = achieved_latency("alert")
    confirmation_latencies = achieved_latency("confirmation")
    evidence_latencies = achieved_latency("evidence")
    first_actionable_latencies = [
        to_float(row.get("first_actionable_latency_s"))
        for row in per_task_rows
        if to_float(row.get("first_actionable_latency_s"), -1.0) >= 0.0
    ]
    summary = {
        "task_count": len(per_task_rows),
        "first_actionable_delivered_rate": len(first_actionable_latencies) / total,
        "alert_rate": sum(to_int(row.get("alert_achieved")) for row in per_task_rows) / total,
        "confirmation_rate": sum(to_int(row.get("confirmation_achieved")) for row in per_task_rows) / total,
        "evidence_rate": sum(to_int(row.get("evidence_achieved")) for row in per_task_rows) / total,
        "alert_latency_mean": mean(alert_latencies) if alert_latencies else 0.0,
        "alert_latency_median": median(alert_latencies) if alert_latencies else 0.0,
        "alert_latency_p90": percentile(alert_latencies, 0.90),
        "confirmation_latency_mean": mean(confirmation_latencies) if confirmation_latencies else 0.0,
        "evidence_latency_mean": mean(evidence_latencies) if evidence_latencies else 0.0,
        "first_actionable_latency_p50": median(first_actionable_latencies) if first_actionable_latencies else 0.0,
        "first_actionable_latency_p90": percentile(first_actionable_latencies, 0.90),
        "expired_without_alert_rate": sum(to_int(row.get("expired_without_alert")) for row in per_task_rows) / total,
        "metadata_only_alert_rate": sum(to_int(row.get("metadata_only_alert")) for row in per_task_rows) / total,
        "direct_roi_alert_confirmation_rate": sum(
            to_int(row.get("direct_roi_alert_confirmation")) for row in per_task_rows
        )
        / total,
        "evidence_closure_rate": sum(to_int(row.get("evidence_closure")) for row in per_task_rows) / total,
        "time_shadow_utility": sum(to_float(row.get("time_shadow_utility")) for row in per_task_rows),
    }
    for weight_name in SHADOW_WEIGHT_SETS:
        key = f"lifecycle_shadow_utility_{weight_name}"
        summary[key] = sum(to_float(row.get(key)) for row in per_task_rows)
    return summary
