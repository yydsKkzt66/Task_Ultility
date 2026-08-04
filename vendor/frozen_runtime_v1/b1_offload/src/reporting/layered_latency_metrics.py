"""Report-only layered latency events and survivor-bias-resistant metrics.

This module is deliberately disconnected from routing, scheduling, reward, and
the common evaluator.  Event-log emission is disabled unless the caller passes
``enabled=True``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

from src.evaluation.task_outcome_adapter import (
    proposed_delivery_events,
    requirement_satisfied_by_products,
)


EPSILON = 1.0e-9
STAGE_PRODUCTS = {
    "metadata": ("alert",),
    "roi": ("alert", "verification"),
    "roi_context": ("alert", "verification", "evidence"),
    "complete_result": ("alert", "verification", "evidence"),
}


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _latency(event_time: float | None, arrival_time: float) -> float | None:
    return None if event_time is None else max(0.0, event_time - arrival_time)


def _event_row(
    *,
    task_id: str,
    arrival_time: float,
    deadline_time: float,
    requirement: str,
    metadata_time: float | None,
    roi_time: float | None,
    evidence_time: float | None,
    closure_time: float | None,
    source: str,
) -> dict[str, Any]:
    alert_latency = _latency(metadata_time, arrival_time)
    verification_latency = _latency(roi_time, arrival_time)
    evidence_latency = _latency(evidence_time, arrival_time)
    closure_latency = _latency(closure_time, arrival_time)
    return {
        "task_id": task_id,
        "task_arrival_time_s": arrival_time,
        "deadline_time_s": deadline_time,
        "product_requirement": requirement,
        "metadata_ground_arrival_time_s": metadata_time,
        "roi_ground_arrival_time_s": roi_time,
        "evidence_ground_arrival_time_s": evidence_time,
        "full_closure_time_s": closure_time,
        "alert_latency_s": alert_latency,
        "verification_latency_s": verification_latency,
        "evidence_latency_s": evidence_latency,
        "closure_latency_s": closure_latency,
        "alert_to_verification_gap_s": (
            None if metadata_time is None or roi_time is None else max(0.0, roi_time - metadata_time)
        ),
        "verification_to_evidence_gap_s": (
            None if roi_time is None or evidence_time is None else max(0.0, evidence_time - roi_time)
        ),
        "alert_to_evidence_gap_s": (
            None if metadata_time is None or evidence_time is None else max(0.0, evidence_time - metadata_time)
        ),
        "alert_on_time": int(metadata_time is not None and metadata_time <= deadline_time + EPSILON),
        "verification_on_time": int(roi_time is not None and roi_time <= deadline_time + EPSILON),
        "evidence_on_time": int(evidence_time is not None and evidence_time <= deadline_time + EPSILON),
        "closed_on_time": int(closure_time is not None and closure_time <= deadline_time + EPSILON),
        "event_source": source,
    }


def build_nonprogressive_event_rows(
    per_task_rows: Iterable[Mapping[str, Any]], *, source: str
) -> list[dict[str, Any]]:
    """Record only public closure for a non-progressive complete-result path.

    One-shot methods have no independent metadata, ROI, or evidence delivery
    stages.  Those fields therefore remain missing instead of copying the
    complete-result timestamp into three events that never occurred.
    """

    output: list[dict[str, Any]] = []
    for row in per_task_rows:
        arrival = _float(row.get("arrival_s"))
        deadline = _float(row.get("deadline_s"))
        if arrival is None or deadline is None:
            raise ValueError("Non-progressive row is missing arrival_s or deadline_s")
        finish = _float(row.get("actual_finish_s"))
        closed = str(row.get("closed", "0")).strip().lower() in {"1", "true", "yes"}
        closure = finish if closed else None
        output.append(
            _event_row(
                task_id=str(row["task_id"]),
                arrival_time=arrival,
                deadline_time=deadline,
                requirement="complete_result",
                metadata_time=None,
                roi_time=None,
                evidence_time=None,
                closure_time=closure,
                source=source,
            )
        )
    return output


def build_prop_event_rows(
    manifest: Mapping[str, Mapping[str, Any]],
    records: Iterable[Any],
    lifecycle_rows: Iterable[Mapping[str, Any]],
    *,
    slot_seconds: float,
    source: str = "PROP-HEADROOM",
) -> list[dict[str, Any]]:
    """Build exact stage timestamps from delivered product events."""

    events = proposed_delivery_events(records, lifecycle_rows, slot_seconds=slot_seconds)
    output: list[dict[str, Any]] = []
    for task_id, task in manifest.items():
        arrival = float(task["arrival_time_s"])
        deadline = float(task["deadline_time_s"])
        requirement = str(task.get("product_requirement", "verification"))
        stage_times: dict[str, float | None] = {"alert": None, "verification": None, "evidence": None}
        observed: set[str] = set()
        closure_time: float | None = None
        for event in sorted(events.get(task_id, []), key=lambda item: float(item["ground_available_time_s"])):
            product = str(event["product_type"])
            event_time = float(event["ground_available_time_s"])
            for stage in STAGE_PRODUCTS.get(product, ()):
                if stage_times[stage] is None:
                    stage_times[stage] = event_time
            observed.add(product)
            if (
                closure_time is None
                and requirement_satisfied_by_products(observed, requirement)
                and event_time <= deadline + EPSILON
            ):
                closure_time = event_time
        evidence_time = (
            stage_times["evidence"] if requirement.strip().lower() in {"context", "evidence"} else None
        )
        output.append(
            _event_row(
                task_id=task_id,
                arrival_time=arrival,
                deadline_time=deadline,
                requirement=requirement,
                metadata_time=stage_times["alert"],
                roi_time=stage_times["verification"],
                evidence_time=evidence_time,
                closure_time=closure_time,
                source=source,
            )
        )
    return output


def write_event_log(path: Path, rows: list[dict[str, Any]], *, enabled: bool = False) -> bool:
    """Write the sidecar only when explicitly enabled; return whether written."""

    if not enabled:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["task_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def summarize_layered_latency(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize rates, achieved-task latencies, and all-task restricted time."""

    task_count = len(rows)
    closure_latencies = [float(row["closure_latency_s"]) for row in rows if row.get("closure_latency_s") not in (None, "")]
    restricted: list[float] = []
    for row in rows:
        arrival = float(row["task_arrival_time_s"])
        deadline = float(row["deadline_time_s"])
        closure = _float(row.get("full_closure_time_s"))
        restricted.append(
            max(0.0, min(closure, deadline) - arrival) if closure is not None else max(0.0, deadline - arrival)
        )

    summary: dict[str, Any] = {
        "task_count": task_count,
        "closure_count": len(closure_latencies),
        "closure_rate": len(closure_latencies) / task_count if task_count else None,
        "closure_latency_mean_s": mean(closure_latencies) if closure_latencies else None,
        "closure_latency_p50_s": median(closure_latencies) if closure_latencies else None,
        "closure_latency_p95_s": _percentile(closure_latencies, 0.95),
        "deadline_truncated_restricted_mean_time_to_closure_s": mean(restricted) if restricted else None,
    }
    for stage in ("alert", "verification", "evidence"):
        latency_key = f"{stage}_latency_s"
        values = [float(row[latency_key]) for row in rows if row.get(latency_key) not in (None, "")]
        on_time_key = f"{stage}_on_time"
        summary[f"{stage}_available_count"] = len(values)
        on_time_values = [
            int(row[on_time_key]) for row in rows if row.get(on_time_key) not in (None, "")
        ]
        summary[f"{stage}_on_time_rate"] = (
            sum(on_time_values) / task_count if task_count and on_time_values else None
        )
        summary[f"{stage}_latency_mean_s"] = mean(values) if values else None
        summary[f"{stage}_latency_p50_s"] = median(values) if values else None
        summary[f"{stage}_latency_p95_s"] = _percentile(values, 0.95)
    return summary


def paired_closure_latency(
    left_rows: list[Mapping[str, Any]], right_rows: list[Mapping[str, Any]], *, left_name: str, right_name: str
) -> list[dict[str, Any]]:
    """Return strict task-ID pairs where both methods closed on time."""

    left = {str(row["task_id"]): row for row in left_rows}
    right = {str(row["task_id"]): row for row in right_rows}
    output: list[dict[str, Any]] = []
    for task_id in sorted(left.keys() & right.keys()):
        left_latency = _float(left[task_id].get("closure_latency_s"))
        right_latency = _float(right[task_id].get("closure_latency_s"))
        if left_latency is None or right_latency is None:
            continue
        output.append(
            {
                "task_id": task_id,
                "left_method": left_name,
                "right_method": right_name,
                "left_closure_latency_s": left_latency,
                "right_closure_latency_s": right_latency,
                "right_minus_left_s": right_latency - left_latency,
            }
        )
    return output
