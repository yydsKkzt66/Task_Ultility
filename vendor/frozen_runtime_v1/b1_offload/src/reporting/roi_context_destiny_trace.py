"""Read-only ROI+context staged-delivery destiny diagnostics."""

from __future__ import annotations

import csv
import math
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PERSISTENT_ARM_DIR = "persistent_queue"
ROUTER_TRAJECTORY = "proposed_lyapunov_rule_mode_safe_bandit.csv"
STAGES = (
    "metadata_generation",
    "metadata_delivery",
    "roi_request_creation",
    "roi_delivery",
    "roi_context_request_creation",
    "roi_context_admission",
    "roi_context_feasibility",
    "roi_context_delivery",
    "final_expiration",
)
BLOCKERS = (
    "metadata_not_generated",
    "roi_request_not_created",
    "roi_context_request_not_created",
    "no_current_contact",
    "no_future_contact_before_deadline",
    "insufficient_future_capacity_before_deadline",
    "metadata_not_delivered",
    "roi_not_delivered",
    "residual_delta_too_low",
    "eriv_below_threshold",
    "value_density_below_threshold",
    "semantic_requirement_not_context_or_evidence",
    "value_level_not_high_or_urgent",
    "top_k_budget_exceeded",
    "contact_window_budget_exceeded",
    "queued_budget_exceeded",
    "duplicate_lifecycle",
    "task_closed",
    "task_expired",
    "deadline_infeasible",
    "capacity_infeasible",
    "queue_infeasible",
    "battery_infeasible",
    "thermal_infeasible",
    "downlink_throttled",
    "ordinary_queue_consumed_capacity",
    "unknown",
)
DESTINY_FIELDS = (
    "scenario",
    "seed",
    "arm",
    "task_id",
    "scene_id",
    "patch_id",
    "task_class",
    "product_requirement",
    "value_level",
    "calibrated_task_value",
    "effective_task_value",
    "deadline_s",
    "arrival_slot",
    "deadline_slot",
    "in_contact_at_arrival",
    "next_contact_slot_after_arrival",
    "contact_capacity_before_deadline_mb",
    "number_of_contact_slots_before_deadline",
    "total_contact_duration_before_deadline_s",
    "metadata_generated_slot",
    "metadata_enqueued_slot",
    "metadata_delivered_slot",
    "metadata_blocker_reason",
    "roi_request_created_slot",
    "roi_request_admitted_slot",
    "roi_enqueued_slot",
    "roi_first_feasible_slot",
    "roi_delivered_slot",
    "roi_blocker_reason",
    "roi_context_request_created_slot",
    "roi_context_request_admitted_slot",
    "roi_context_pending_first_slot",
    "roi_context_enqueued_slot",
    "roi_context_first_feasible_slot",
    "roi_context_delivered_slot",
    "roi_context_blocker_reason",
    "roi_context_payload_size_mb",
    "roi_context_priority_score_max",
    "roi_context_priority_rank_best",
    "roi_context_competed_with_lower_value_payload_count",
    "final_stage_reached",
    "final_status",
    "final_failure_reason",
    "was_roi_context_ever_created",
    "was_roi_context_ever_feasible",
    "was_roi_context_feasible_but_not_delivered",
    "was_low_value_payload_delivered_while_this_feasible",
    "destiny_summary_sentence",
)


@dataclass
class RunTrace:
    scenario: str
    seed: int
    arm: str
    run_dir: Path
    trajectory: list[dict[str, str]]
    staged_requests: list[dict[str, str]]
    staged_priority: list[dict[str, str]]
    scheduling: list[dict[str, str]]
    queue_priority: list[dict[str, str]]
    carryover_events: list[dict[str, str]]
    same_slot_vs_queued: list[dict[str, str]]
    future_contact: list[dict[str, str]]
    backlog: list[dict[str, str]]


@dataclass
class DestinyAnalysis:
    funnel_rows: list[dict[str, Any]]
    blocker_rows: list[dict[str, Any]]
    destiny_rows: list[dict[str, Any]]
    product_rows: list[dict[str, Any]]
    contact_rows: list[dict[str, Any]]
    case_specs: list[dict[str, Any]]
    slot_trace_rows_by_case: dict[str, list[dict[str, Any]]]
    report_markdown: str


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def bool01(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y", "on"})
    return int(bool(value))


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if abs(float(den)) > 1.0e-12 else 0.0


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value not in {None, ""}:
            return str(value)
    return ""


def product_set(raw: Any) -> set[str]:
    return {item.strip() for item in str(raw or "").replace(",", ";").split(";") if item.strip()}


def split_reasons(raw: Any) -> list[str]:
    return [item.strip() for item in str(raw or "").replace(",", ";").split(";") if item.strip()]


def first_slot(rows: list[dict[str, str]], predicate) -> str:
    slots = sorted(to_int(row.get("slot"), 10**9) for row in rows if predicate(row))
    return "" if not slots else str(slots[0])


def min_slot(*slots: Any) -> str:
    parsed = [to_int(slot, 10**9) for slot in slots if slot not in {"", None}]
    return "" if not parsed else str(min(parsed))


def category_from_reason(reason: Any) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "unknown"
    if "metadata_not_generated" in text:
        return "metadata_not_generated"
    if "roi_context_request_not_created" in text:
        return "roi_context_request_not_created"
    if "roi_request_not_created" in text:
        return "roi_request_not_created"
    if "no_future" in text:
        return "no_future_contact_before_deadline"
    if "no_contact" in text or "contact_opportunity" in text:
        return "no_current_contact"
    if "future_capacity" in text:
        return "insufficient_future_capacity_before_deadline"
    if "metadata_not_delivered" in text:
        return "metadata_not_delivered"
    if "roi_not_delivered" in text:
        return "roi_not_delivered"
    if "residual_delta" in text or "zero_increment" in text:
        return "residual_delta_too_low"
    if "eriv" in text:
        return "eriv_below_threshold"
    if "value_density" in text:
        return "value_density_below_threshold"
    if "semantic" in text or "requirement_not_context" in text:
        return "semantic_requirement_not_context_or_evidence"
    if "value_level" in text:
        return "value_level_not_high_or_urgent"
    if "top_k" in text or "rank_budget" in text:
        return "top_k_budget_exceeded"
    if "contact_window" in text or "window_budget" in text:
        return "contact_window_budget_exceeded"
    if "staged_budget" in text or "queued_budget" in text:
        return "queued_budget_exceeded"
    if "duplicate" in text:
        return "duplicate_lifecycle"
    if "closed" in text:
        return "task_closed"
    if "expired" in text:
        return "task_expired"
    if "deadline" in text:
        return "deadline_infeasible"
    if "capacity" in text or "payload_exceeds" in text:
        return "capacity_infeasible"
    if "queue" in text:
        return "queue_infeasible"
    if "battery" in text or "energy" in text:
        return "battery_infeasible"
    if "thermal" in text:
        return "thermal_infeasible"
    if "throttle" in text:
        return "downlink_throttled"
    if "ordinary" in text:
        return "ordinary_queue_consumed_capacity"
    return "unknown"


def scenario_from_dir(name: str) -> str:
    return "targeted_resource_scarcity" if name == "scarcity" else name


def infer_run_dirs(source_dir: Path, arm_dir: str = PERSISTENT_ARM_DIR) -> list[Path]:
    runs_root = source_dir / "runs"
    if not runs_root.exists():
        return []
    return sorted(path for path in runs_root.glob(f"* /s*/{arm_dir}".replace(" ", "")) if path.is_dir())


def load_run(run_dir: Path) -> RunTrace:
    seed_text = run_dir.parent.name.lstrip("s")
    scenario = scenario_from_dir(run_dir.parent.parent.name)
    return RunTrace(
        scenario=scenario,
        seed=to_int(seed_text),
        arm="soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue",
        run_dir=run_dir,
        trajectory=read_csv(run_dir / "trajectories" / ROUTER_TRAJECTORY),
        staged_requests=read_csv(run_dir / "staged_delivery_request_trace.csv"),
        staged_priority=read_csv(run_dir / "staged_delivery_priority_trace.csv"),
        scheduling=read_csv(run_dir / "contact_window_scheduling_trace.csv"),
        queue_priority=read_csv(run_dir / "queue_priority_trace.csv"),
        carryover_events=read_csv(run_dir / "carryover_lifecycle_events.csv"),
        same_slot_vs_queued=read_csv(run_dir / "same_slot_vs_queued_followup_trace.csv"),
        future_contact=read_csv(run_dir / "future_contact_feasibility_trace.csv"),
        backlog=read_csv(run_dir / "backlog_composition_trace.csv"),
    )


def task_identity(row: dict[str, str]) -> str:
    return str(row.get("task_id", ""))


def is_real_task(row: dict[str, str]) -> bool:
    return bool01(row.get("has_task")) == 1 and task_identity(row) not in {"", "__idle__"}


def is_high_urgent(row: dict[str, str]) -> bool:
    return str(row.get("value_level", "")).strip().lower() in {"high", "urgent"}


def is_context_evidence(row: dict[str, str]) -> bool:
    return str(row.get("product_requirement", "")).strip().lower() in {"context", "evidence"}


def is_high_urgent_context_evidence(row: dict[str, str]) -> bool:
    return is_high_urgent(row) and is_context_evidence(row)


def unique_task_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    tasks: dict[str, dict[str, str]] = {}
    for row in rows:
        if is_real_task(row):
            tasks[task_identity(row)] = row
    return tasks


def slot_seconds_for_run(run: RunTrace) -> float:
    samples = [(to_int(row.get("slot")), to_float(row.get("t"), -1.0)) for row in run.trajectory if row.get("slot") not in {"", None}]
    for slot, t_value in samples:
        if slot > 0 and t_value > 0:
            return max(1.0, t_value / slot)
    return 20.0


def index_by_task(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    indexed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id:
            indexed[task_id].append(row)
    for task_rows in indexed.values():
        task_rows.sort(key=lambda item: to_int(item.get("slot"), 10**9))
    return dict(indexed)


def queue_delivery_slot(task_rows: list[dict[str, str]], product_type: str) -> str:
    return first_slot(
        task_rows,
        lambda row: str(row.get("payload_type", row.get("product_type", ""))) == product_type
        and bool01(row.get("delivered_this_slot")) == 1,
    )


def queue_seen_slot(task_rows: list[dict[str, str]], product_type: str) -> str:
    return first_slot(task_rows, lambda row: str(row.get("payload_type", row.get("product_type", ""))) == product_type)


def queue_expired_slot(task_rows: list[dict[str, str]], product_type: str) -> str:
    return first_slot(
        task_rows,
        lambda row: str(row.get("payload_type", row.get("product_type", ""))) == product_type
        and bool01(row.get("expired_this_slot")) == 1,
    )


def lifecycle_event_slot(event_rows: list[dict[str, str]], product_type: str, event_kind: str) -> str:
    def matches(row: dict[str, str]) -> bool:
        if str(row.get("residual_product_type", "")) != product_type:
            return False
        event_type = str(row.get("event_type", ""))
        removal = str(row.get("removal_reason", ""))
        if event_kind == "created":
            return event_type in {"candidate_added", "same_slot_candidate_re_evaluated"}
        if event_kind == "feasible":
            return bool01(row.get("proactive_flag")) == 1 or bool01(row.get("ERIV_above_threshold_flag")) == 1
        if event_kind == "enqueued":
            return (
                "enqueued" in event_type
                or "enqueued" in removal
                or bool01(row.get("proactive_enqueue_flag")) == 1
            )
        if event_kind == "delivered":
            return (
                "delivered" in event_type
                or removal == "residual_delivered"
                or bool01(row.get("proactive_delivery_flag")) == 1
            )
        return False

    return first_slot(event_rows, matches)


def request_slot(request_rows: list[dict[str, str]], product_type: str, actions: set[str]) -> str:
    return first_slot(
        request_rows,
        lambda row: str(row.get("product_type", "")) == product_type and str(row.get("action", "")) in actions,
    )


def max_priority(priority_rows: list[dict[str, str]], product_type: str) -> str:
    values = [
        to_float(first_nonempty(row.get("component_priority_score"), row.get("priority_score")), math.nan)
        for row in priority_rows
        if str(row.get("product_type", "")) == product_type
    ]
    values = [value for value in values if not math.isnan(value)]
    return "" if not values else f"{max(values):.12g}"


def best_rank(priority_rows: list[dict[str, str]], product_type: str) -> str:
    ranks = [to_int(row.get("queue_order"), 10**9) for row in priority_rows if str(row.get("product_type", "")) == product_type]
    ranks = [rank for rank in ranks if rank < 10**9]
    return "" if not ranks else str(min(ranks) + 1)


def contact_slots(run: RunTrace) -> dict[int, dict[str, Any]]:
    slots: dict[int, dict[str, Any]] = {}
    for row in run.backlog:
        slot = to_int(row.get("slot"), -1)
        if slot >= 0:
            slots.setdefault(slot, {})["in_contact"] = bool01(row.get("in_contact"))
            slots[slot]["queue_mb"] = to_float(row.get("queue_mb"))
    for row in run.queue_priority:
        slot = to_int(row.get("slot"), -1)
        if slot < 0:
            continue
        entry = slots.setdefault(slot, {})
        if bool01(row.get("contact_available")):
            entry["in_contact"] = 1
        if row.get("link_rate_bps") not in {"", None}:
            entry["link_rate_bps"] = max(to_float(entry.get("link_rate_bps")), to_float(row.get("link_rate_bps")))
    for row in run.scheduling:
        slot = to_int(row.get("slot"), -1)
        if slot >= 0 and bool01(row.get("in_contact")):
            slots.setdefault(slot, {})["in_contact"] = 1
    return slots


def estimate_slot_capacity_mb(slot_entry: dict[str, Any], slot_seconds: float) -> float:
    link_rate_bps = to_float(slot_entry.get("link_rate_bps"))
    if link_rate_bps <= 0:
        return 0.0
    return link_rate_bps * slot_seconds / 8.0 / 1_000_000.0


def contact_context(
    run: RunTrace,
    arrival_slot: int,
    deadline_slot: int,
    slot_seconds: float,
) -> tuple[int, str, float, int, float]:
    contacts = contact_slots(run)
    in_contact_at_arrival = bool01(contacts.get(arrival_slot, {}).get("in_contact"))
    future = [slot for slot, entry in contacts.items() if slot >= arrival_slot and slot <= deadline_slot and bool01(entry.get("in_contact"))]
    next_slot = "" if not future else str(min(future))
    capacity = sum(estimate_slot_capacity_mb(contacts[slot], slot_seconds) for slot in future)
    return in_contact_at_arrival, next_slot, capacity, len(future), len(future) * slot_seconds


def reason_from_rows(rows: list[dict[str, str]], product_type: str) -> str:
    reasons: list[str] = []
    non_blockers = {
        "admitted_to_downlink_queue",
        "queued_followup_registered",
        "queued_followup_payload_ready",
        "same_slot_enqueued_via_downlink_queue",
    }
    for row in rows:
        if product_type and str(row.get("product_type", row.get("payload_type", ""))) not in {product_type, ""}:
            continue
        for reason in split_reasons(row.get("blocker_reason")):
            if reason not in non_blockers:
                reasons.append(reason)
        for reason in split_reasons(row.get("reason_code")):
            if reason not in non_blockers:
                reasons.append(reason)
        for reason in split_reasons(row.get("blockers")):
            if reason not in non_blockers:
                reasons.append(reason)
    return reasons[0] if reasons else ""


def row_products_delivered_by_lifecycle(task_row: dict[str, str], event_rows: list[dict[str, str]]) -> set[str]:
    delivered = product_set(task_row.get("delivered_products")) | product_set(task_row.get("delivered_product_types"))
    for row in event_rows:
        if str(row.get("event_type", "")).endswith("delivered"):
            delivered.add(str(row.get("residual_product_type", "")))
        if str(row.get("removal_reason", "")) == "residual_delivered":
            delivered.add(str(row.get("residual_product_type", "")))
        delivered |= product_set(row.get("delivered_products_after_eval"))
        delivered |= product_set(row.get("delivered_products_before_eval"))
    return {item for item in delivered if item}


def build_task_destiny_rows(run: RunTrace) -> list[dict[str, Any]]:
    task_rows = unique_task_rows(run.trajectory)
    queue_by_task = index_by_task(run.queue_priority)
    request_by_task = index_by_task(run.staged_requests)
    priority_by_task = index_by_task(run.staged_priority)
    schedule_by_task = index_by_task(run.scheduling)
    event_by_task = index_by_task(run.carryover_events)
    same_slot_by_task = index_by_task(run.same_slot_vs_queued)
    slot_seconds = slot_seconds_for_run(run)
    rows: list[dict[str, Any]] = []

    for task_id, task in sorted(task_rows.items(), key=lambda item: to_int(item[1].get("slot"))):
        if not is_high_urgent_context_evidence(task):
            continue
        queue_rows = queue_by_task.get(task_id, [])
        request_rows = request_by_task.get(task_id, [])
        priority_rows = priority_by_task.get(task_id, [])
        schedule_rows = schedule_by_task.get(task_id, [])
        same_rows = same_slot_by_task.get(task_id, [])
        event_rows = event_by_task.get(task_id, [])
        arrival_slot = to_int(task.get("slot"))
        deadline_time_s = to_float(task.get("deadline_time_s"), -1.0)
        deadline_slot = to_int(math.floor(deadline_time_s / slot_seconds), arrival_slot + max(0, to_int(to_float(task.get("deadline_s")) / slot_seconds))) if deadline_time_s >= 0 else arrival_slot + max(0, to_int(to_float(task.get("deadline_s")) / slot_seconds))
        selected_product = str(task.get("selected_product_type", task.get("payload_policy", ""))).strip()
        selected_plan = str(task.get("selected_payload_plan", "")).strip()
        delivered_products = row_products_delivered_by_lifecycle(task, event_rows)

        metadata_generated = ""
        metadata_enqueued = ""
        if selected_product == "metadata" or selected_plan == "metadata" or "metadata" in selected_plan:
            metadata_generated = str(arrival_slot)
            if str(task.get("delivery_status", "")) in {"queued", "delivered"}:
                metadata_enqueued = str(arrival_slot)
        metadata_delivered = queue_delivery_slot(queue_rows, "metadata")
        if not metadata_delivered and "metadata" in delivered_products:
            metadata_delivered = str(arrival_slot)
        metadata_enqueued = min_slot(metadata_enqueued, queue_seen_slot(queue_rows, "metadata"))

        direct_roi_generated = str(arrival_slot) if selected_product == "roi" or selected_plan == "roi" else ""
        roi_request_created = min_slot(
            direct_roi_generated,
            request_slot(request_rows, "roi", {"created_pending", "blocked_no_future_contact", "blocked_value_density", "blocked_eriv"}),
            lifecycle_event_slot(event_rows, "roi", "created"),
        )
        roi_admitted = min_slot(
            str(arrival_slot) if selected_product == "roi" and str(task.get("delivery_status", "")) in {"queued", "delivered"} else "",
            first_slot(schedule_rows, lambda row: str(row.get("product_type", "")) == "roi" and str(row.get("decision", "")) == "enqueued"),
            lifecycle_event_slot(event_rows, "roi", "enqueued"),
        )
        roi_enqueued = min_slot(
            roi_admitted,
            request_slot(request_rows, "roi", {"enqueued"}),
            queue_seen_slot(queue_rows, "roi"),
            lifecycle_event_slot(event_rows, "roi", "enqueued"),
        )
        roi_delivered = queue_delivery_slot(queue_rows, "roi")
        if not roi_delivered and "roi" in delivered_products:
            roi_delivered = min_slot(lifecycle_event_slot(event_rows, "roi", "delivered"), roi_enqueued)
        roi_first_feasible = min_slot(
            roi_delivered,
            first_slot(
                queue_rows,
                lambda row: str(row.get("payload_type", "")) == "roi" and bool01(row.get("contact_available")) == 1,
            ),
            first_slot(schedule_rows, lambda row: str(row.get("product_type", "")) == "roi" and str(row.get("decision", "")) == "enqueued"),
            lifecycle_event_slot(event_rows, "roi", "feasible"),
        )
        if roi_first_feasible and not (roi_enqueued or roi_delivered):
            roi_first_feasible = ""

        direct_context_generated = str(arrival_slot) if selected_product == "roi_context" else ""
        roi_context_created = min_slot(
            direct_context_generated,
            request_slot(
                request_rows,
                "roi_context",
                {"created_pending", "blocked_no_future_contact", "blocked_value_density", "blocked_eriv"},
            ),
            lifecycle_event_slot(event_rows, "roi_context", "created"),
        )
        roi_context_pending = request_slot(request_rows, "roi_context", {"created_pending"})
        roi_context_admitted = min_slot(
            first_slot(schedule_rows, lambda row: str(row.get("product_type", "")) == "roi_context" and str(row.get("decision", "")) == "enqueued"),
            lifecycle_event_slot(event_rows, "roi_context", "enqueued"),
        )
        roi_context_enqueued = min_slot(
            roi_context_admitted,
            request_slot(request_rows, "roi_context", {"enqueued"}),
            queue_seen_slot(queue_rows, "roi_context"),
            lifecycle_event_slot(event_rows, "roi_context", "enqueued"),
        )
        roi_context_delivered = queue_delivery_slot(queue_rows, "roi_context")
        if not roi_context_delivered and "roi_context" in delivered_products:
            roi_context_delivered = min_slot(
                lifecycle_event_slot(event_rows, "roi_context", "delivered"),
                roi_context_enqueued,
            )
        roi_context_first_feasible = min_slot(
            roi_context_delivered,
            first_slot(
                queue_rows,
                lambda row: str(row.get("payload_type", "")) == "roi_context" and bool01(row.get("contact_available")) == 1,
            ),
            first_slot(
                schedule_rows,
                lambda row: str(row.get("product_type", "")) == "roi_context" and str(row.get("decision", "")) == "enqueued",
            ),
            first_slot(
                priority_rows,
                lambda row: str(row.get("product_type", "")) == "roi_context" and bool01(row.get("selected_for_admission")) == 1,
            ),
            lifecycle_event_slot(event_rows, "roi_context", "feasible"),
        )
        if roi_context_first_feasible and not (roi_context_enqueued or roi_context_delivered):
            roi_context_first_feasible = ""
        context_payload_size = first_nonempty(
            next((row.get("payload_size_mb") for row in request_rows if row.get("product_type") == "roi_context"), ""),
            next((row.get("payload_size_mb") for row in queue_rows if row.get("payload_type") == "roi_context"), ""),
            next((row.get("payload_size_mb") for row in same_rows if row.get("payload_size_mb") not in {"", None}), ""),
        )

        in_contact, next_contact, capacity_mb, contact_count, contact_duration = contact_context(
            run,
            arrival_slot,
            deadline_slot,
            slot_seconds,
        )
        feasible_slots = {
            to_int(roi_context_first_feasible)
        } if roi_context_first_feasible not in {"", None} else set()
        low_value_competitors = 0
        low_value_competitor_flag = 0
        for queue_row in run.queue_priority:
            slot = to_int(queue_row.get("slot"))
            if slot not in feasible_slots:
                continue
            if str(queue_row.get("task_id")) == task_id or bool01(queue_row.get("delivered_this_slot")) != 1:
                continue
            if to_float(queue_row.get("task_value")) < to_float(task.get("calibrated_task_value")):
                low_value_competitors += 1
                low_value_competitor_flag = 1

        metadata_blocker = ""
        if not metadata_generated:
            metadata_blocker = "metadata_not_generated"
        elif not metadata_delivered:
            metadata_blocker = reason_from_rows(queue_rows + schedule_rows + request_rows, "metadata") or "metadata_not_delivered"

        roi_blocker = ""
        if not roi_request_created:
            roi_blocker = "metadata_not_delivered" if not metadata_delivered and not direct_roi_generated else "roi_request_not_created"
        elif not roi_delivered:
            roi_blocker = reason_from_rows(request_rows + schedule_rows + queue_rows, "roi") or "roi_not_delivered"

        context_blocker = ""
        if not roi_context_created:
            context_blocker = "roi_not_delivered" if not roi_delivered else "roi_context_request_not_created"
        elif not roi_context_first_feasible:
            context_blocker = reason_from_rows(request_rows + schedule_rows + same_rows, "roi_context") or "no_future_contact_before_deadline"
        elif not roi_context_delivered:
            context_blocker = reason_from_rows(queue_rows + schedule_rows, "roi_context") or "ordinary_queue_consumed_capacity"

        if roi_context_delivered:
            final_stage = "roi_context"
            final_status = "delivered"
            final_failure = ""
        elif roi_delivered:
            final_stage = "roi"
            final_status = "expired" if context_blocker in {"expired", "task_expired"} or roi_context_created else "pending"
            final_failure = context_blocker
        elif metadata_delivered:
            final_stage = "metadata"
            final_status = "expired" if roi_blocker in {"expired", "task_expired"} else "pending"
            final_failure = roi_blocker
        else:
            final_stage = "none"
            final_status = "dropped" if selected_product == "defer" else "expired" if bool01(task.get("deadline_missed")) else "pending"
            final_failure = metadata_blocker or roi_blocker or context_blocker

        summary = destiny_sentence(
            task_id,
            str(task.get("value_level", "")),
            str(task.get("product_requirement", "")),
            metadata_delivered,
            roi_delivered,
            roi_context_created,
            roi_context_delivered,
            final_failure,
        )
        rows.append(
            {
                "scenario": run.scenario,
                "seed": run.seed,
                "arm": run.arm,
                "task_id": task_id,
                "scene_id": task.get("scene_id", ""),
                "patch_id": task.get("patch_id", ""),
                "task_class": task.get("task_class", ""),
                "product_requirement": task.get("product_requirement", ""),
                "value_level": task.get("value_level", ""),
                "calibrated_task_value": task.get("calibrated_task_value", ""),
                "effective_task_value": first_nonempty(task.get("effective_task_value"), task.get("task_value"), task.get("calibrated_task_value")),
                "deadline_s": task.get("deadline_s", ""),
                "arrival_slot": arrival_slot,
                "deadline_slot": deadline_slot,
                "in_contact_at_arrival": in_contact,
                "next_contact_slot_after_arrival": next_contact,
                "contact_capacity_before_deadline_mb": f"{capacity_mb:.6f}",
                "number_of_contact_slots_before_deadline": contact_count,
                "total_contact_duration_before_deadline_s": f"{contact_duration:.6f}",
                "metadata_generated_slot": metadata_generated,
                "metadata_enqueued_slot": metadata_enqueued,
                "metadata_delivered_slot": metadata_delivered,
                "metadata_blocker_reason": metadata_blocker,
                "roi_request_created_slot": roi_request_created,
                "roi_request_admitted_slot": roi_admitted,
                "roi_enqueued_slot": roi_enqueued,
                "roi_first_feasible_slot": roi_first_feasible,
                "roi_delivered_slot": roi_delivered,
                "roi_blocker_reason": roi_blocker,
                "roi_context_request_created_slot": roi_context_created,
                "roi_context_request_admitted_slot": roi_context_admitted,
                "roi_context_pending_first_slot": roi_context_pending,
                "roi_context_enqueued_slot": roi_context_enqueued,
                "roi_context_first_feasible_slot": roi_context_first_feasible,
                "roi_context_delivered_slot": roi_context_delivered,
                "roi_context_blocker_reason": context_blocker,
                "roi_context_payload_size_mb": context_payload_size,
                "roi_context_priority_score_max": max_priority(priority_rows, "roi_context"),
                "roi_context_priority_rank_best": best_rank(priority_rows, "roi_context"),
                "roi_context_competed_with_lower_value_payload_count": low_value_competitors,
                "final_stage_reached": final_stage,
                "final_status": final_status,
                "final_failure_reason": final_failure,
                "was_roi_context_ever_created": int(bool(roi_context_created)),
                "was_roi_context_ever_feasible": int(bool(roi_context_first_feasible)),
                "was_roi_context_feasible_but_not_delivered": int(bool(roi_context_first_feasible) and not bool(roi_context_delivered)),
                "was_low_value_payload_delivered_while_this_feasible": low_value_competitor_flag,
                "destiny_summary_sentence": summary,
            }
        )
    return rows


def destiny_sentence(
    task_id: str,
    value_level: str,
    requirement: str,
    metadata_delivered: str,
    roi_delivered: str,
    roi_context_created: str,
    roi_context_delivered: str,
    failure: str,
) -> str:
    prefix = f"Task {task_id} arrived as {value_level}/{requirement}."
    if roi_context_delivered:
        return f"{prefix} Metadata delivered at slot {metadata_delivered or 'not recorded'}, ROI delivered at slot {roi_delivered or 'not recorded'}, and ROI+Ctx delivered at slot {roi_context_delivered}."
    if roi_context_created:
        return f"{prefix} Metadata slot {metadata_delivered or 'not delivered'}, ROI slot {roi_delivered or 'not delivered'}, ROI+Ctx request was created at slot {roi_context_created} but did not deliver because {failure or 'unknown'}."
    if not roi_delivered:
        return f"{prefix} It never generated ROI+Ctx because ROI was not delivered before the residual stage."
    return f"{prefix} ROI was delivered at slot {roi_delivered}, but ROI+Ctx request was not created ({failure or 'unknown'})."


def summarize_funnel_for_run(run: RunTrace, destiny_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = unique_task_rows(run.trajectory)
    total_tasks = len(tasks)
    high_count = sum(1 for row in tasks.values() if is_high_urgent(row))
    context_count = sum(1 for row in tasks.values() if is_context_evidence(row))
    huce_count = len(destiny_rows)
    row = {
        "scenario": run.scenario,
        "seed": run.seed,
        "arm": run.arm,
        "total_tasks": total_tasks,
        "high_urgent_task_count": high_count,
        "context_evidence_task_count": context_count,
        "high_urgent_context_evidence_task_count": huce_count,
        "metadata_generated_count": count_nonempty(destiny_rows, "metadata_generated_slot"),
        "metadata_enqueued_count": count_nonempty(destiny_rows, "metadata_enqueued_slot"),
        "metadata_delivered_count": count_nonempty(destiny_rows, "metadata_delivered_slot"),
        "metadata_missed_count": count_where(destiny_rows, lambda item: bool(item.get("metadata_generated_slot")) and not bool(item.get("metadata_delivered_slot"))),
        "metadata_expired_count": count_where(destiny_rows, lambda item: category_from_reason(item.get("metadata_blocker_reason")) == "task_expired"),
        "roi_request_created_count": count_nonempty(destiny_rows, "roi_request_created_slot"),
        "roi_request_admitted_count": count_nonempty(destiny_rows, "roi_request_admitted_slot"),
        "roi_request_enqueued_count": count_nonempty(destiny_rows, "roi_enqueued_slot"),
        "roi_request_feasible_count": count_nonempty(destiny_rows, "roi_first_feasible_slot"),
        "roi_delivered_count": count_nonempty(destiny_rows, "roi_delivered_slot"),
        "roi_missed_count": count_where(destiny_rows, lambda item: bool(item.get("roi_request_created_slot")) and not bool(item.get("roi_delivered_slot"))),
        "roi_expired_count": count_where(destiny_rows, lambda item: category_from_reason(item.get("roi_blocker_reason")) == "task_expired"),
        "roi_context_request_created_count": count_nonempty(destiny_rows, "roi_context_request_created_slot"),
        "roi_context_request_admitted_count": count_nonempty(destiny_rows, "roi_context_request_admitted_slot"),
        "roi_context_request_pending_count": count_nonempty(destiny_rows, "roi_context_pending_first_slot"),
        "roi_context_request_enqueued_count": count_nonempty(destiny_rows, "roi_context_enqueued_slot"),
        "roi_context_request_feasible_count": count_nonempty(destiny_rows, "roi_context_first_feasible_slot"),
        "roi_context_delivered_count": count_nonempty(destiny_rows, "roi_context_delivered_slot"),
        "roi_context_missed_count": count_where(destiny_rows, lambda item: bool(item.get("roi_context_request_created_slot")) and not bool(item.get("roi_context_delivered_slot"))),
        "roi_context_expired_count": count_where(destiny_rows, lambda item: category_from_reason(item.get("roi_context_blocker_reason")) == "task_expired" or str(item.get("final_status")) == "expired"),
        "stopped_at_metadata_count": count_where(destiny_rows, lambda item: item.get("final_stage_reached") == "metadata"),
        "stopped_at_roi_count": count_where(destiny_rows, lambda item: item.get("final_stage_reached") == "roi"),
        "stopped_before_roi_count": count_where(destiny_rows, lambda item: not bool(item.get("roi_request_created_slot"))),
        "stopped_before_roi_context_count": count_where(destiny_rows, lambda item: bool(item.get("roi_delivered_slot")) and not bool(item.get("roi_context_request_created_slot"))),
        "delivered_roi_context_count": count_nonempty(destiny_rows, "roi_context_delivered_slot"),
        "context_closure_count": count_where(destiny_rows, lambda item: item.get("product_requirement") == "context" and bool(item.get("roi_context_delivered_slot"))),
        "evidence_closure_count": count_where(destiny_rows, lambda item: item.get("product_requirement") == "evidence" and bool(item.get("roi_context_delivered_slot"))),
    }
    row.update(
        {
            "metadata_delivery_rate_among_high_urgent_context_evidence": safe_div(row["metadata_delivered_count"], huce_count),
            "roi_request_creation_rate_after_metadata": safe_div(
                count_where(destiny_rows, lambda item: bool(item.get("metadata_delivered_slot")) and bool(item.get("roi_request_created_slot"))),
                row["metadata_delivered_count"],
            ),
            "roi_delivery_rate_after_roi_request": safe_div(
                count_where(destiny_rows, lambda item: bool(item.get("roi_request_created_slot")) and bool(item.get("roi_delivered_slot"))),
                row["roi_request_created_count"],
            ),
            "roi_context_request_creation_rate_after_roi_delivered": safe_div(
                count_where(destiny_rows, lambda item: bool(item.get("roi_delivered_slot")) and bool(item.get("roi_context_request_created_slot"))),
                row["roi_delivered_count"],
            ),
            "roi_context_feasible_rate_after_roi_context_request": safe_div(
                count_where(destiny_rows, lambda item: bool(item.get("roi_context_request_created_slot")) and bool(item.get("roi_context_first_feasible_slot"))),
                row["roi_context_request_created_count"],
            ),
            "roi_context_delivery_rate_after_roi_context_feasible": safe_div(
                count_where(destiny_rows, lambda item: bool(item.get("roi_context_first_feasible_slot")) and bool(item.get("roi_context_delivered_slot"))),
                row["roi_context_request_feasible_count"],
            ),
            "end_to_end_metadata_to_roi_context_rate": safe_div(
                count_where(destiny_rows, lambda item: bool(item.get("metadata_delivered_slot")) and bool(item.get("roi_context_delivered_slot"))),
                row["metadata_delivered_count"],
            ),
        }
    )
    return row


def count_nonempty(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) not in {"", None})


def count_where(rows: list[dict[str, Any]], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def build_blocker_rows_for_run(run: RunTrace, destiny_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_to_reasons: dict[str, Counter[str]] = {stage: Counter() for stage in STAGES}
    for row in destiny_rows:
        if not row.get("metadata_generated_slot"):
            stage_to_reasons["metadata_generation"][category_from_reason(row.get("metadata_blocker_reason"))] += 1
        if row.get("metadata_generated_slot") and not row.get("metadata_delivered_slot"):
            stage_to_reasons["metadata_delivery"][category_from_reason(row.get("metadata_blocker_reason"))] += 1
        if row.get("metadata_delivered_slot") and not row.get("roi_request_created_slot"):
            stage_to_reasons["roi_request_creation"][category_from_reason(row.get("roi_blocker_reason"))] += 1
        if row.get("roi_request_created_slot") and not row.get("roi_delivered_slot"):
            stage_to_reasons["roi_delivery"][category_from_reason(row.get("roi_blocker_reason"))] += 1
        if row.get("roi_delivered_slot") and not row.get("roi_context_request_created_slot"):
            stage_to_reasons["roi_context_request_creation"][category_from_reason(row.get("roi_context_blocker_reason"))] += 1
        if row.get("roi_context_request_created_slot") and not row.get("roi_context_enqueued_slot"):
            stage_to_reasons["roi_context_admission"][category_from_reason(row.get("roi_context_blocker_reason"))] += 1
        if row.get("roi_context_request_created_slot") and not row.get("roi_context_first_feasible_slot"):
            stage_to_reasons["roi_context_feasibility"][category_from_reason(row.get("roi_context_blocker_reason"))] += 1
        if row.get("roi_context_first_feasible_slot") and not row.get("roi_context_delivered_slot"):
            stage_to_reasons["roi_context_delivery"][category_from_reason(row.get("roi_context_blocker_reason"))] += 1
        if row.get("final_status") == "expired":
            stage_to_reasons["final_expiration"][category_from_reason(row.get("final_failure_reason"))] += 1
    output: list[dict[str, Any]] = []
    for stage in STAGES:
        for blocker in BLOCKERS:
            output.append(
                {
                    "scenario": run.scenario,
                    "seed": run.seed,
                    "arm": run.arm,
                    "stage": stage,
                    "blocker_category": blocker,
                    "count": stage_to_reasons[stage].get(blocker, 0),
                }
            )
    return output


def build_product_generation_rows(run: RunTrace, destiny_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue_by_task = index_by_task(run.queue_priority)
    request_by_task = index_by_task(run.staged_requests)
    event_by_task = index_by_task(run.carryover_events)
    task_rows = unique_task_rows(run.trajectory)
    rows: list[dict[str, Any]] = []
    for destiny in destiny_rows:
        task_id = str(destiny["task_id"])
        task = task_rows.get(task_id, {})
        queue_rows = queue_by_task.get(task_id, [])
        request_rows = request_by_task.get(task_id, [])
        event_rows = event_by_task.get(task_id, [])
        lifecycle_products = row_products_delivered_by_lifecycle(task, event_rows)
        for product_type in ("metadata", "roi", "roi_context"):
            generated_slot = product_generated_slot(destiny, task, request_rows, queue_rows, product_type)
            staged_created_slot = request_slot(
                request_rows,
                product_type,
                {"created_pending", "blocked_no_future_contact", "blocked_value_density", "blocked_eriv"},
            )
            enqueued_slot = product_enqueued_slot(destiny, request_rows, queue_rows, product_type)
            appeared = bool(queue_seen_slot(queue_rows, product_type))
            delivered_slot = product_delivered_slot(destiny, queue_rows, product_type)
            lifecycle_recorded = product_type in lifecycle_products or bool(delivered_slot and appeared)
            rows.append(
                {
                    "scenario": run.scenario,
                    "seed": run.seed,
                    "arm": run.arm,
                    "task_id": task_id,
                    "product_type": product_type,
                    "target_exists_flag": int(product_target_exists(destiny, product_type)),
                    "product_generated_flag": int(bool(generated_slot)),
                    "generated_slot": generated_slot,
                    "staged_request_created_flag": int(bool(staged_created_slot)),
                    "staged_request_created_slot": staged_created_slot,
                    "pending_flag": int(bool(product_pending_flag(destiny, request_rows, product_type))),
                    "enqueued_flag": int(bool(enqueued_slot)),
                    "enqueued_slot": enqueued_slot,
                    "appeared_in_downlink_queue_flag": int(appeared),
                    "delivered_flag": int(bool(delivered_slot)),
                    "delivered_slot": delivered_slot,
                    "lifecycle_recorded_flag": int(bool(lifecycle_recorded)),
                    "missing_transition_reason": missing_transition_reason(
                        target_exists=product_target_exists(destiny, product_type),
                        generated=bool(generated_slot),
                        staged_created=bool(staged_created_slot),
                        enqueued=bool(enqueued_slot),
                        appeared=appeared,
                        delivered=bool(delivered_slot),
                        lifecycle_recorded=bool(lifecycle_recorded),
                    ),
                }
            )
    return rows


def product_target_exists(destiny: dict[str, Any], product_type: str) -> bool:
    requirement = str(destiny.get("product_requirement", ""))
    if product_type == "metadata":
        return True
    if product_type == "roi":
        return requirement in {"verification", "context", "evidence"}
    return requirement in {"context", "evidence"} and str(destiny.get("value_level")) in {"high", "urgent"}


def product_generated_slot(
    destiny: dict[str, Any],
    task: dict[str, str],
    request_rows: list[dict[str, str]],
    queue_rows: list[dict[str, str]],
    product_type: str,
) -> str:
    if product_type == "metadata":
        return str(destiny.get("metadata_generated_slot", ""))
    if product_type == "roi":
        return str(destiny.get("roi_request_created_slot", ""))
    return str(destiny.get("roi_context_request_created_slot", ""))


def product_enqueued_slot(
    destiny: dict[str, Any],
    request_rows: list[dict[str, str]],
    queue_rows: list[dict[str, str]],
    product_type: str,
) -> str:
    if product_type == "metadata":
        return str(destiny.get("metadata_enqueued_slot", ""))
    if product_type == "roi":
        return str(destiny.get("roi_enqueued_slot", ""))
    return str(destiny.get("roi_context_enqueued_slot", ""))


def product_delivered_slot(destiny: dict[str, Any], queue_rows: list[dict[str, str]], product_type: str) -> str:
    if product_type == "metadata":
        return str(destiny.get("metadata_delivered_slot", ""))
    if product_type == "roi":
        return str(destiny.get("roi_delivered_slot", ""))
    return str(destiny.get("roi_context_delivered_slot", ""))


def product_pending_flag(destiny: dict[str, Any], request_rows: list[dict[str, str]], product_type: str) -> bool:
    if product_type == "roi_context":
        return bool(destiny.get("roi_context_pending_first_slot"))
    return any(row.get("product_type") == product_type and row.get("status") == "pending" for row in request_rows)


def missing_transition_reason(
    *,
    target_exists: bool,
    generated: bool,
    staged_created: bool,
    enqueued: bool,
    appeared: bool,
    delivered: bool,
    lifecycle_recorded: bool,
) -> str:
    if not target_exists:
        return "target_not_required"
    if not generated:
        return "target_exists_but_product_not_generated"
    if staged_created and not enqueued and not delivered:
        return "staged_request_created_but_no_queue_item"
    if generated and not enqueued and not appeared and not delivered:
        return "product_generated_but_never_enqueued"
    if enqueued and not appeared and not delivered:
        return "enqueued_but_missing_from_downlink_queue_trace"
    if appeared and not delivered:
        return "appeared_in_downlink_queue_but_not_delivered"
    if delivered and not lifecycle_recorded:
        return "delivered_but_lifecycle_not_recorded"
    return ""


def build_contact_timeline_rows(run: RunTrace, destiny_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slot_seconds = slot_seconds_for_run(run)
    contacts = contact_slots(run)
    queue_by_slot: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in run.queue_priority:
        queue_by_slot[to_int(row.get("slot"), -1)].append(row)
    staged_by_slot: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in run.staged_priority:
        staged_by_slot[to_int(row.get("slot"), -1)].append(row)
    scheduling_by_slot: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in run.scheduling:
        scheduling_by_slot[to_int(row.get("slot"), -1)].append(row)
    rows: list[dict[str, Any]] = []
    window_id = ""
    last_contact = False
    window_seq = 0
    for slot in sorted(contacts):
        entry = contacts[slot]
        in_contact = bool01(entry.get("in_contact")) == 1
        if not in_contact:
            continue
        if not last_contact:
            window_seq += 1
            window_id = f"contact_window_{slot}"
        last_contact = True
        qrows = sorted(queue_by_slot.get(slot, []), key=lambda row: to_int(first_nonempty(row.get("delivery_order"), row.get("queue_order_before_downlink")), 10**9))
        delivered = [row for row in qrows if bool01(row.get("delivered_this_slot")) == 1]
        total_mb = sum(to_float(row.get("payload_size_mb")) for row in delivered)
        staged_delivered = [row for row in delivered if row_is_staged(row)]
        ordinary_delivered = [row for row in delivered if not row_is_staged(row)]
        staged_pending = [
            row
            for row in staged_by_slot.get(slot, [])
            if str(row.get("product_requirement")) in {"context", "evidence"}
            and str(row.get("value_level")) in {"high", "urgent"}
        ]
        feasible_staged = [row for row in staged_pending if bool01(row.get("selected_for_admission")) == 1]
        low_value_ordinary = [row for row in ordinary_delivered if bool01(row.get("high_value_flag")) == 0]
        priority_inversion = bool(low_value_ordinary and feasible_staged and not staged_delivered)
        reason = ""
        if not staged_delivered:
            if not staged_pending:
                reason = "no_high_value_staged_pending"
            elif not feasible_staged:
                reason = reason_from_rows(scheduling_by_slot.get(slot, []), "") or "no_feasible_staged_request"
            elif ordinary_delivered:
                reason = "ordinary_queue_consumed_capacity"
            else:
                reason = "no_delivery_this_contact_slot"
        rows.append(
            {
                "scenario": run.scenario,
                "seed": run.seed,
                "arm": run.arm,
                "slot": slot,
                "contact_window_id": first_nonempty(
                    next((item.get("contact_window_id") for item in scheduling_by_slot.get(slot, []) if item.get("contact_window_id")), ""),
                    window_id,
                ),
                "available_capacity_mb": f"{estimate_slot_capacity_mb(entry, slot_seconds):.6f}",
                "total_delivered_mb": f"{total_mb:.6f}",
                "staged_delivered_mb": f"{sum(to_float(row.get('payload_size_mb')) for row in staged_delivered):.6f}",
                "ordinary_delivered_mb": f"{sum(to_float(row.get('payload_size_mb')) for row in ordinary_delivered):.6f}",
                "delivered_payload_order": ";".join(f"{row.get('task_id')}:{row.get('payload_type')}" for row in delivered),
                "delivered_task_values_order": ";".join(str(row.get("task_value", "")) for row in delivered),
                "delivered_product_types_order": ";".join(str(row.get("payload_type", "")) for row in delivered),
                "highest_pending_staged_priority": f"{max([to_float(first_nonempty(row.get('component_priority_score'), row.get('priority_score'))) for row in staged_pending] or [0.0]):.12g}",
                "highest_delivered_priority": f"{max([to_float(row.get('queue_priority_total')) for row in delivered] or [0.0]):.12g}",
                "high_value_staged_pending_count": len(staged_pending),
                "high_value_staged_feasible_count": len(feasible_staged),
                "low_value_ordinary_delivered_count": len(low_value_ordinary),
                "priority_inversion_flag": int(priority_inversion),
                "reason_if_no_staged_delivered": reason,
            }
        )
    return rows


def row_is_staged(row: dict[str, str]) -> bool:
    if bool01(row.get("is_staged_delivery_queue")):
        return True
    if row.get("staged_priority_total") not in {"", None}:
        return True
    return str(row.get("followup_trigger", "")) == "value_aware_staged_delivery_queue"


def select_case_studies(destiny_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[tuple[int, str, Any]] = [
        (1, "ROI+Ctx delivered", lambda row: bool(row.get("roi_context_delivered_slot"))),
        (2, "stopped at ROI", lambda row: row.get("final_stage_reached") == "roi"),
        (3, "ROI not delivered", lambda row: not bool(row.get("roi_delivered_slot"))),
        (4, "ROI+Ctx request created but never feasible", lambda row: bool(row.get("roi_context_request_created_slot")) and not bool(row.get("roi_context_first_feasible_slot"))),
        (5, "ROI+Ctx feasible but not delivered", lambda row: bool01(row.get("was_roi_context_feasible_but_not_delivered")) == 1),
        (6, "had contact but not delivered", lambda row: to_int(row.get("number_of_contact_slots_before_deadline")) > 0 and not bool(row.get("roi_context_delivered_slot"))),
        (7, "no current contact then pending", lambda row: bool01(row.get("in_contact_at_arrival")) == 0 and bool(row.get("roi_context_pending_first_slot"))),
        (8, "expired before future contact", lambda row: row.get("final_status") == "expired" and not row.get("next_contact_slot_after_arrival")),
        (9, "payload or capacity insufficient", lambda row: category_from_reason(row.get("roi_context_blocker_reason")) == "capacity_infeasible"),
        (10, "highest calibrated value without ROI+Ctx", lambda row: not bool(row.get("roi_context_delivered_slot"))),
    ]
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int, str]] = set()
    for case_id, label, predicate in cases:
        candidates = [row for row in destiny_rows if predicate(row)]
        if case_id == 10:
            candidates = sorted(candidates, key=lambda row: to_float(row.get("calibrated_task_value")), reverse=True)
        else:
            candidates = sorted(candidates, key=lambda row: (-to_float(row.get("calibrated_task_value")), to_int(row.get("arrival_slot"))))
        for row in candidates:
            key = (str(row.get("scenario")), to_int(row.get("seed")), str(row.get("task_id")))
            if key in selected_keys and len(selected) >= 6:
                continue
            selected_keys.add(key)
            selected.append({"case_id": case_id, "case_label": label, **row})
            break
        if len(selected) >= 10:
            break
    if len(selected) < 6:
        for row in sorted(destiny_rows, key=lambda item: (-to_float(item.get("calibrated_task_value")), to_int(item.get("arrival_slot")))):
            key = (str(row.get("scenario")), to_int(row.get("seed")), str(row.get("task_id")))
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append({"case_id": 90 + len(selected), "case_label": "additional failed high-value task", **row})
            if len(selected) >= 6:
                break
    return selected[:10]


def build_slot_trace_for_case(run: RunTrace, case: dict[str, Any]) -> list[dict[str, Any]]:
    task_id = str(case["task_id"])
    arrival = to_int(case.get("arrival_slot"))
    deadline = to_int(case.get("deadline_slot"), arrival)
    slot_seconds = slot_seconds_for_run(run)
    queue_by_slot: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in run.queue_priority:
        queue_by_slot[to_int(row.get("slot"), -1)].append(row)
    sched_by_slot: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in run.scheduling:
        sched_by_slot[to_int(row.get("slot"), -1)].append(row)
    priority_by_slot: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in run.staged_priority:
        priority_by_slot[to_int(row.get("slot"), -1)].append(row)
    contacts = contact_slots(run)
    traj_by_slot = {to_int(row.get("slot")): row for row in run.trajectory}
    delivered_so_far: set[str] = set()
    out: list[dict[str, Any]] = []
    for slot in range(arrival, min(deadline, max(deadline, arrival) + 1) + 1):
        qrows = queue_by_slot.get(slot, [])
        own_delivered = [
            str(row.get("payload_type", ""))
            for row in qrows
            if str(row.get("task_id")) == task_id and bool01(row.get("delivered_this_slot")) == 1
        ]
        delivered_so_far.update(item for item in own_delivered if item)
        task_priority_rows = [row for row in priority_by_slot.get(slot, []) if str(row.get("task_id")) == task_id]
        blockers = []
        for row in sched_by_slot.get(slot, []):
            if str(row.get("task_id")) == task_id:
                blockers.extend(split_reasons(row.get("reason_code")))
                blockers.extend(split_reasons(row.get("blockers")))
        delivered_payloads = [f"{row.get('task_id')}:{row.get('payload_type')}" for row in qrows if bool01(row.get("delivered_this_slot"))]
        lower_value = any(
            bool01(row.get("delivered_this_slot")) == 1
            and str(row.get("task_id")) != task_id
            and to_float(row.get("task_value")) < to_float(case.get("calibrated_task_value"))
            for row in qrows
        )
        stage = stage_at_slot(case, slot)
        trow = traj_by_slot.get(slot, {})
        contact_entry = contacts.get(slot, {})
        out.append(
            {
                "slot": slot,
                "time_s": f"{slot * slot_seconds:.6f}",
                "in_contact": bool01(contact_entry.get("in_contact")),
                "remaining_window_s": "",
                "available_capacity_mb": f"{estimate_slot_capacity_mb(contact_entry, slot_seconds):.6f}",
                "queue_mb": first_nonempty(contact_entry.get("queue_mb"), trow.get("queue_mb")),
                "battery_soc": trow.get("battery_soc", ""),
                "thermal_pressure": first_nonempty(trow.get("thermal_pressure"), trow.get("Theta_t")),
                "task_stage_at_slot": stage,
                "delivered_products_so_far": ";".join(sorted(delivered_so_far)),
                "pending_products": pending_products_at_slot(case, slot),
                "staged_request_status": staged_status_at_slot(case, slot),
                "priority_score": "" if not task_priority_rows else max_priority(task_priority_rows, str(task_priority_rows[0].get("product_type", ""))),
                "priority_rank": "" if not task_priority_rows else str(min(to_int(row.get("queue_order"), 10**9) for row in task_priority_rows) + 1),
                "feasible_this_slot": int(bool(task_priority_rows and any(bool01(row.get("selected_for_admission")) for row in task_priority_rows))),
                "blockers_this_slot": ";".join(dict.fromkeys(blockers)),
                "delivered_payloads_this_slot": ";".join(delivered_payloads),
                "lower_value_payload_delivered_this_slot": int(lower_value),
                "reason_not_delivered_this_slot": "" if own_delivered else (";".join(dict.fromkeys(blockers)) or "not_selected_or_no_target_delivery"),
            }
        )
    return out


def stage_at_slot(case: dict[str, Any], slot: int) -> str:
    if case.get("roi_context_delivered_slot") not in {"", None} and slot >= to_int(case.get("roi_context_delivered_slot")):
        return "roi_context_delivered"
    if case.get("roi_context_request_created_slot") not in {"", None} and slot >= to_int(case.get("roi_context_request_created_slot")):
        return "roi_context_request"
    if case.get("roi_delivered_slot") not in {"", None} and slot >= to_int(case.get("roi_delivered_slot")):
        return "roi_delivered"
    if case.get("metadata_delivered_slot") not in {"", None} and slot >= to_int(case.get("metadata_delivered_slot")):
        return "metadata_delivered"
    if case.get("metadata_generated_slot") not in {"", None} and slot >= to_int(case.get("metadata_generated_slot")):
        return "metadata_generated"
    return "none"


def pending_products_at_slot(case: dict[str, Any], slot: int) -> str:
    pending = []
    if case.get("metadata_enqueued_slot") not in {"", None} and not case.get("metadata_delivered_slot"):
        pending.append("metadata")
    if case.get("roi_enqueued_slot") not in {"", None} and not case.get("roi_delivered_slot"):
        pending.append("roi")
    if case.get("roi_context_request_created_slot") not in {"", None} and slot >= to_int(case.get("roi_context_request_created_slot")) and not case.get("roi_context_delivered_slot"):
        pending.append("roi_context")
    return ";".join(pending)


def staged_status_at_slot(case: dict[str, Any], slot: int) -> str:
    if case.get("roi_context_delivered_slot") not in {"", None} and slot >= to_int(case.get("roi_context_delivered_slot")):
        return "delivered"
    if case.get("roi_context_enqueued_slot") not in {"", None} and slot >= to_int(case.get("roi_context_enqueued_slot")):
        return "enqueued"
    if case.get("roi_context_request_created_slot") not in {"", None} and slot >= to_int(case.get("roi_context_request_created_slot")):
        return "pending"
    return ""


def validate_funnel_counts_are_monotonic(funnel_rows: list[dict[str, Any]]) -> None:
    for row in funnel_rows:
        assert to_int(row["metadata_delivered_count"]) <= to_int(row["metadata_enqueued_count"]) <= to_int(row["metadata_generated_count"]) <= to_int(row["high_urgent_context_evidence_task_count"])
        assert to_int(row["roi_delivered_count"]) <= to_int(row["roi_request_feasible_count"]) <= to_int(row["roi_request_enqueued_count"]) <= to_int(row["roi_request_created_count"]) <= to_int(row["high_urgent_context_evidence_task_count"])
        assert to_int(row["roi_context_delivered_count"]) <= to_int(row["roi_context_request_feasible_count"]) <= to_int(row["roi_context_request_enqueued_count"]) <= to_int(row["roi_context_request_created_count"]) <= to_int(row["high_urgent_context_evidence_task_count"])


def analyze_runs(runs: list[RunTrace]) -> DestinyAnalysis:
    all_destiny: list[dict[str, Any]] = []
    all_funnel: list[dict[str, Any]] = []
    all_blockers: list[dict[str, Any]] = []
    all_products: list[dict[str, Any]] = []
    all_contacts: list[dict[str, Any]] = []
    run_lookup: dict[tuple[str, int], RunTrace] = {}
    for run in runs:
        destiny = build_task_destiny_rows(run)
        all_destiny.extend(destiny)
        all_funnel.append(summarize_funnel_for_run(run, destiny))
        all_blockers.extend(build_blocker_rows_for_run(run, destiny))
        all_products.extend(build_product_generation_rows(run, destiny))
        all_contacts.extend(build_contact_timeline_rows(run, destiny))
        run_lookup[(run.scenario, run.seed)] = run
    validate_funnel_counts_are_monotonic(all_funnel)
    cases = select_case_studies(all_destiny)
    slot_traces: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        run = run_lookup.get((str(case["scenario"]), to_int(case["seed"])))
        if run is None:
            continue
        slot_traces[case_key(case)] = build_slot_trace_for_case(run, case)
    return DestinyAnalysis(
        funnel_rows=all_funnel,
        blocker_rows=all_blockers,
        destiny_rows=all_destiny,
        product_rows=all_products,
        contact_rows=all_contacts,
        case_specs=cases,
        slot_trace_rows_by_case=slot_traces,
        report_markdown=build_report(all_funnel, all_blockers, all_destiny, all_products, all_contacts, cases),
    )


def case_key(case: dict[str, Any]) -> str:
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case.get("task_id", "task")))
    return f"case_{to_int(case.get('case_id')):02d}_task_{safe_task}"


def build_report(
    funnel_rows: list[dict[str, Any]],
    blocker_rows: list[dict[str, Any]],
    destiny_rows: list[dict[str, Any]],
    product_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> str:
    totals = aggregate_funnel(funnel_rows)
    losses = [
        ("task -> metadata generated", totals["high_urgent_context_evidence_task_count"] - totals["metadata_generated_count"]),
        ("metadata generated -> metadata delivered", totals["metadata_generated_count"] - totals["metadata_delivered_count"]),
        ("metadata delivered -> ROI request", totals["metadata_delivered_count"] - totals["roi_request_created_count"]),
        ("ROI request -> ROI delivered", totals["roi_request_created_count"] - totals["roi_delivered_count"]),
        ("ROI delivered -> ROI+Ctx request", totals["roi_delivered_count"] - totals["roi_context_request_created_count"]),
        ("ROI+Ctx request -> feasible", totals["roi_context_request_created_count"] - totals["roi_context_request_feasible_count"]),
        ("ROI+Ctx feasible -> delivered", totals["roi_context_request_feasible_count"] - totals["roi_context_delivered_count"]),
    ]
    biggest_loss = max(losses, key=lambda item: item[1]) if losses else ("none", 0)
    blocker_totals = Counter()
    stage_totals = Counter()
    for row in blocker_rows:
        count = to_int(row.get("count"))
        blocker_totals[str(row.get("blocker_category"))] += count
        stage_totals[str(row.get("stage"))] += count
    main_stage, _ = stage_totals.most_common(1)[0] if stage_totals else ("unknown", 0)
    top_blockers = [item for item in blocker_totals.most_common(5) if item[1] > 0]
    conclusion = choose_conclusion(totals, biggest_loss[0], contact_rows, product_rows)
    lifecycle_misses = sum(1 for row in product_rows if bool01(row.get("delivered_flag")) and not bool01(row.get("lifecycle_recorded_flag")))
    generated_no_queue = sum(1 for row in product_rows if row.get("missing_transition_reason") == "product_generated_but_never_enqueued")
    request_no_queue = sum(1 for row in product_rows if row.get("missing_transition_reason") == "staged_request_created_but_no_queue_item")
    feasible_not_delivered = sum(1 for row in destiny_rows if bool01(row.get("was_roi_context_feasible_but_not_delivered")))
    low_value_while_feasible = sum(1 for row in destiny_rows if bool01(row.get("was_low_value_payload_delivered_while_this_feasible")))
    lines = [
        "# ROI+Ctx Destiny Trace Diagnosis",
        "",
        "## Conclusion",
        "",
        f"{conclusion}",
        "",
        f"The largest funnel loss is `{biggest_loss[0]}` with {biggest_loss[1]} tasks. The most active blocker stage is `{main_stage}`.",
        "",
        "## Required Answers",
        "",
        f"1. high-value context/evidence tasks: {totals['high_urgent_context_evidence_task_count']}.",
        f"2. metadata generated: {totals['metadata_generated_count']}.",
        f"3. metadata delivered: {totals['metadata_delivered_count']}.",
        f"4. ROI requests generated: {totals['roi_request_created_count']}.",
        f"5. ROI delivered: {totals['roi_delivered_count']}.",
        f"6. ROI+Ctx requests generated: {totals['roi_context_request_created_count']}.",
        f"7. ROI+Ctx feasible: {totals['roi_context_request_feasible_count']}.",
        f"8. ROI+Ctx delivered: {totals['roi_context_delivered_count']}.",
        f"9. largest loss: {biggest_loss[0]}.",
        "10. representative task fates show whether each task stopped before metadata, before ROI, after ROI, at ROI+Ctx creation, or after feasibility.",
        f"11. high-value feasible-but-not-delivered examples: {'yes' if feasible_not_delivered else 'no'} ({feasible_not_delivered}).",
        f"12. product generated but queue did not catch it: {'yes' if generated_no_queue or request_no_queue else 'no'} (generated-no-queue={generated_no_queue}, request-no-queue={request_no_queue}).",
        f"13. delivered but lifecycle not recorded: {'yes' if lifecycle_misses else 'no'} ({lifecycle_misses}).",
        f"14. next step: {'change scenario/contact/deadline/task stream first' if conclusion.startswith(('A.', 'B.', 'C.', 'D.')) else 'inspect mechanism plumbing first'}; do not tune reward or relax hard safety based on this report alone.",
        "",
        "## Stage Blockers",
        "",
    ]
    for blocker, count in top_blockers[:3]:
        lines.append(f"- {blocker}: {count}")
    lines += [
        "",
        "## Direct Diagnosis",
        "",
        f"- Main stuck stage: `{main_stage}`.",
        f"- Top blockers at all stages: {', '.join(f'{name}={count}' for name, count in top_blockers[:3]) if top_blockers else 'none'}.",
        f"- Request not generated vs infeasible vs feasible-not-delivered: ROI+Ctx not-created={totals['roi_delivered_count'] - totals['roi_context_request_created_count']}, created-not-feasible={totals['roi_context_request_created_count'] - totals['roi_context_request_feasible_count']}, feasible-not-delivered={totals['roi_context_request_feasible_count'] - totals['roi_context_delivered_count']}.",
        f"- Low delivery is primarily predecessor-stage caused: {'yes' if totals['roi_delivered_count'] < totals['high_urgent_context_evidence_task_count'] / 2 else 'no'}.",
        f"- Low-value payload while feasible high-value ROI+Ctx existed: {'yes' if low_value_while_feasible else 'no'} ({low_value_while_feasible}).",
        "",
        "## Case Studies",
        "",
    ]
    for case in cases:
        lines.append(
            f"- Case {to_int(case.get('case_id'))}: {case.get('case_label')} / {case.get('scenario')} s{case.get('seed')} task {case.get('task_id')} -> {case.get('final_failure_reason') or case.get('final_status')}"
        )
    return "\n".join(lines)


def aggregate_funnel(rows: list[dict[str, Any]]) -> dict[str, int]:
    numeric_keys = [
        "total_tasks",
        "high_urgent_task_count",
        "context_evidence_task_count",
        "high_urgent_context_evidence_task_count",
        "metadata_generated_count",
        "metadata_enqueued_count",
        "metadata_delivered_count",
        "roi_request_created_count",
        "roi_request_feasible_count",
        "roi_delivered_count",
        "roi_context_request_created_count",
        "roi_context_request_feasible_count",
        "roi_context_delivered_count",
    ]
    return {key: sum(to_int(row.get(key)) for row in rows) for key in numeric_keys}


def choose_conclusion(
    totals: dict[str, int],
    biggest_loss_label: str,
    contact_rows: list[dict[str, Any]],
    product_rows: list[dict[str, Any]],
) -> str:
    if totals["high_urgent_context_evidence_task_count"] == 0:
        return "A. ROI+Ctx is scarce mainly because high/urgent context/evidence tasks are insufficient."
    if "metadata" in biggest_loss_label or "ROI request -> ROI delivered" in biggest_loss_label or "task -> metadata" in biggest_loss_label:
        return "B. ROI+Ctx is scarce mainly because metadata or ROI predecessor stages are not delivered in time."
    if "ROI delivered -> ROI+Ctx request" in biggest_loss_label:
        return "E. ROI+Ctx is scarce mainly because admission/request creation blocks after ROI delivery."
    if "request -> feasible" in biggest_loss_label:
        return "C. ROI+Ctx is scarce mainly because contact/capacity before deadline is missing after ROI delivery."
    if "feasible -> delivered" in biggest_loss_label:
        inversions = sum(to_int(row.get("priority_inversion_flag")) for row in contact_rows)
        if inversions:
            return "H. ROI+Ctx is scarce mainly because the priority scheduler is not prioritizing staged queue items."
        return "C. ROI+Ctx is scarce mainly because feasible windows still do not leave enough contact/capacity before deadline."
    lifecycle_misses = sum(1 for row in product_rows if bool01(row.get("delivered_flag")) and not bool01(row.get("lifecycle_recorded_flag")))
    if lifecycle_misses:
        return "G. ROI+Ctx is scarce mainly because lifecycle/accounting records are inconsistent."
    return "I. ROI+Ctx scarcity is multi-factor; predecessor delivery and post-ROI contact feasibility dominate in that order."


def case_markdown(case: dict[str, Any]) -> str:
    success_change = suggested_change(case)
    return "\n".join(
        [
            f"# Case {to_int(case.get('case_id'))}: {case.get('task_id')}",
            "",
            f"- Why high value: value_level={case.get('value_level')}, calibrated_task_value={case.get('calibrated_task_value')}.",
            f"- Required product: {case.get('product_requirement')} task, target final product is ROI+Ctx after metadata/ROI predecessors.",
            f"- Actual deepest layer: {case.get('final_stage_reached')} ({case.get('final_status')}).",
            f"- Metadata slot: generated={case.get('metadata_generated_slot')}, delivered={case.get('metadata_delivered_slot')}.",
            f"- ROI slot: request={case.get('roi_request_created_slot')}, feasible={case.get('roi_first_feasible_slot')}, delivered={case.get('roi_delivered_slot')}.",
            f"- ROI+Ctx slot: request={case.get('roi_context_request_created_slot')}, feasible={case.get('roi_context_first_feasible_slot')}, delivered={case.get('roi_context_delivered_slot')}.",
            f"- Root cause: {case.get('final_failure_reason') or case.get('roi_context_blocker_reason') or 'none recorded'}.",
            f"- To make it succeed: {success_change}.",
            "",
            case.get("destiny_summary_sentence", ""),
        ]
    )


def suggested_change(case: dict[str, Any]) -> str:
    reason = category_from_reason(case.get("final_failure_reason") or case.get("roi_context_blocker_reason") or case.get("roi_blocker_reason"))
    if reason in {"no_current_contact", "no_future_contact_before_deadline", "deadline_infeasible", "task_expired"}:
        return "longer deadline or earlier ROI so the ROI+Ctx request exists before the contact window."
    if reason in {"capacity_infeasible", "insufficient_future_capacity_before_deadline", "queue_infeasible"}:
        return "larger contact capacity or smaller ROI+Ctx payload."
    if reason in {"metadata_not_delivered", "roi_not_delivered"}:
        return "improve predecessor delivery timing; changing ROI+Ctx scheduling alone will not fix this case."
    if reason in {"eriv_below_threshold", "value_density_below_threshold", "queued_budget_exceeded", "top_k_budget_exceeded"}:
        return "inspect mechanism/admission budget with this trace; do not change reward based only on this case."
    return "inspect task stream/contact geometry before changing the mechanism."


def write_analysis_outputs(
    analysis: DestinyAnalysis,
    out_dir: Path,
    *,
    source_dir: Path,
    command: str,
    project_root: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "staged_delivery_funnel_summary.csv", analysis.funnel_rows)
    write_csv(out_dir / "stage_blocker_summary.csv", analysis.blocker_rows)
    write_csv(out_dir / "high_value_task_destiny_trace.csv", analysis.destiny_rows, list(DESTINY_FIELDS))
    write_csv(out_dir / "product_generation_trace.csv", analysis.product_rows)
    write_csv(out_dir / "contact_slot_delivery_timeline.csv", analysis.contact_rows)
    write_text(out_dir / "final_diagnosis_report.md", analysis.report_markdown)
    write_text(
        out_dir / "README.md",
        "# ROI+Ctx Destiny Trace\n\nRead-only diagnostic package built from persistent value-aware staged delivery queue traces.",
    )
    write_text(
        out_dir / "command_log.md",
        "\n".join(
            [
                "# Command Log",
                "",
                f"- {command}",
                "- python -m pytest tests/test_roi_context_destiny_trace.py -q",
                "- python -m pytest -q",
                f"- Source: {source_dir.as_posix()}",
                "",
            ]
        ),
    )
    write_text(
        out_dir / "run_manifest.yaml",
        "\n".join(
            [
                f"command: {command}",
                f"source_dir: {source_dir.as_posix()}",
                "analysis_only: true",
                "reward_function_changed: false",
                "closure_definition_changed: false",
                "gate_changed: false",
                "hard_safety_relaxed: false",
            ]
        ),
    )
    write_text(out_dir / "git_status.txt", git_text(project_root, ["status", "--short"]))
    write_text(out_dir / "git_diff.patch", git_text(project_root, ["diff", "--", "src/reporting/roi_context_destiny_trace.py", "scripts/run_roi_context_destiny_trace.py", "tests/test_roi_context_destiny_trace.py"]))
    case_dir = out_dir / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    for case in analysis.case_specs:
        key = case_key(case)
        write_text(case_dir / f"{key}.md", case_markdown(case))
        write_csv(case_dir / f"{key}_slot_trace.csv", analysis.slot_trace_rows_by_case.get(key, []))
    archive_manifest = []
    for path in sorted(item for item in out_dir.rglob("*") if item.is_file()):
        archive_manifest.append(path.relative_to(out_dir).as_posix())
    write_text(out_dir / "archive_manifest.txt", "\n".join(archive_manifest))
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(item for item in out_dir.rglob("*") if item.is_file()):
            zf.write(path, path.relative_to(out_dir.parent))


def git_text(project_root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = completed.stdout
    if completed.stderr:
        text += "\nSTDERR:\n" + completed.stderr
    return text.strip()
