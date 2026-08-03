"""Audit high-value staged activation from existing seed=2026 traces only."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_hv_phase1_seed2026 import OUT, PROP_SOURCE, read_csv, write_csv


REPORT = OUT / "high_value_staged_activation_audit.md"
TASK_CSV = OUT / "high_value_staged_activation_task_level.csv"


def _i(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _f(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_join(values: list[str]) -> str:
    return ";".join(sorted({value for value in values if value}))


def build_task_audit(
    outcomes: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    staged_trace: list[dict[str, Any]],
    *,
    runtime_level_evidence: str = "medium",
) -> list[dict[str, Any]]:
    ledger_by_task: dict[str, list[dict[str, Any]]] = {}
    request_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in ledger:
        ledger_by_task.setdefault(str(row.get("task_id", "")), []).append(row)
    for row in staged_trace:
        request_by_task.setdefault(str(row.get("task_id", "")), []).append(row)

    output: list[dict[str, Any]] = []
    for outcome in sorted(outcomes, key=lambda row: str(row["task_id"])):
        task_id = str(outcome["task_id"])
        events = ledger_by_task.get(task_id, [])
        requests = request_by_task.get(task_id, [])

        def generated(product: str) -> bool:
            return any(row.get("product_type") == product and row.get("current_status") == "created" for row in events)

        def delivered(product: str) -> bool:
            return any(row.get("product_type") == product and row.get("current_status") == "delivered" for row in events)

        metadata_generated = generated("metadata")
        metadata_delivered = delivered("metadata")
        roi_generated = generated("roi")
        roi_delivered = delivered("roi")
        roi_context_generated = generated("roi_context")
        roi_context_delivered = delivered("roi_context")
        staged_events = [row for row in events if row.get("source") == "staged_delivery_queue"]
        staged_created = any(row.get("current_status") == "created" for row in staged_events) or any(
            row.get("action") == "created_pending" for row in requests
        )
        request_levels = [str(row.get("value_level", "")) for row in staged_events + requests if row.get("value_level")]
        runtime_level = request_levels[0] if request_levels else runtime_level_evidence
        requirement = str(outcome.get("product_requirement", ""))
        eligible_roi = metadata_delivered and requirement in {"context", "evidence"}
        roi_context_before_value = roi_delivered and requirement in {"context", "evidence"} and not roi_context_delivered
        eligible_roi_context = roi_context_before_value and runtime_level in {"high", "urgent"}
        blocker_reasons = _unique_join(
            [str(row.get("blocker_reason", "")) for row in staged_events + requests]
        )
        staged_statuses = _unique_join(
            [str(row.get("current_status", row.get("status", ""))) for row in staged_events]
            + [str(row.get("action", "")) for row in requests]
        )
        direct_closed = roi_context_delivered and bool(_i(outcome.get("requirement_satisfied")))

        if direct_closed:
            failed_condition = "none_direct_roi_context_closure"
        elif staged_created:
            if blocker_reasons:
                failed_condition = blocker_reasons
            elif "expired" in staged_statuses:
                failed_condition = "staged_request_expired_before_delivery"
            else:
                failed_condition = "staged_request_created_not_delivered"
        elif roi_context_before_value and runtime_level not in {"high", "urgent"}:
            failed_condition = "runtime_value_level_not_high_or_urgent"
        elif metadata_delivered and not staged_created:
            failed_condition = "metadata_delivered_but_staged_roi_request_absent"
        elif metadata_generated and not metadata_delivered:
            failed_condition = "metadata_not_delivered"
        elif roi_generated and not roi_delivered:
            failed_condition = "roi_not_delivered"
        elif roi_context_generated and not roi_context_delivered:
            failed_condition = "direct_roi_context_not_delivered"
        else:
            failed_condition = "no_metadata_or_roi_trigger_generated"

        any_expired = bool(_i(outcome.get("expired"))) or any(row.get("current_status") == "expired" for row in events)
        any_blocked = any(row.get("current_status") == "blocked" for row in events) or bool(blocker_reasons)
        deferred = str(outcome.get("failure_reason", "")) == "deferred"
        lacks_contact = any(
            token in blocker_reasons
            for token in ("no_contact_opportunity", "insufficient_contact_capacity", "no_future_contact")
        )
        output.append({
            "task_id": task_id,
            "reported_value_level": outcome.get("value_level", ""),
            "runtime_staged_value_level": runtime_level,
            "runtime_value_level_source": "task-specific staged trace" if request_levels else "inferred from uniform seed-2026 staged-trace mapping; not persisted for this task",
            "product_requirement": requirement,
            "metadata_generated": int(metadata_generated),
            "metadata_delivered": int(metadata_delivered),
            "roi_generated": int(roi_generated),
            "roi_delivered": int(roi_delivered),
            "roi_context_generated": int(roi_context_generated),
            "roi_context_delivered": int(roi_context_delivered),
            "eligible_for_staged_roi": int(eligible_roi),
            "eligible_for_staged_roi_context_before_value_gate": int(roi_context_before_value),
            "eligible_for_staged_roi_context_after_value_gate": int(eligible_roi_context),
            "staged_request_created": int(staged_created),
            "staged_request_products": _unique_join([str(row.get("product_type", "")) for row in staged_events]),
            "staged_request_statuses": staged_statuses,
            "exact_failed_condition": failed_condition,
            "blocker_reasons": blocker_reasons,
            "closed_directly": int(direct_closed),
            "requirement_satisfied": _i(outcome.get("requirement_satisfied")),
            "on_time_success": _i(outcome.get("on_time_task_success")),
            "closure_time_s": outcome.get("requirement_closure_time_s", ""),
            "expired": int(any_expired),
            "blocked": int(any_blocked),
            "deferred": int(deferred),
            "lacked_contact_opportunity": int(lacks_contact),
            "final_failure_reason": outcome.get("failure_reason", ""),
            "final_transmitted_products": outcome.get("transmitted_product_type", ""),
        })
    return output


def capacity_overlap_audit(
    high_created: list[dict[str, Any]],
    ordinary_delivered: list[dict[str, Any]],
    high_recompetition: list[dict[str, Any]],
    ordinary_recompetition: list[dict[str, Any]],
) -> dict[str, Any]:
    lifetime_pairs = []
    for high in high_created:
        created = _i(high.get("created_slot"))
        deadline = _i(high.get("deadline_slot"))
        for ordinary in ordinary_delivered:
            delivered = _i(ordinary.get("delivered_slot"))
            if created <= delivered <= deadline:
                lifetime_pairs.append((high.get("task_id"), ordinary.get("task_id"), delivered))
    window_pairs = []
    for high in high_recompetition:
        for ordinary in ordinary_recompetition:
            if (
                high.get("contact_window_id")
                and high.get("contact_window_id") == ordinary.get("contact_window_id")
                and high.get("slot") == ordinary.get("slot")
                and _i(ordinary.get("delivered"))
            ):
                window_pairs.append((high.get("task_id"), ordinary.get("task_id"), high.get("slot")))
    return {
        "active_lifetime_overlap_pair_count": len(lifetime_pairs),
        "active_lifetime_overlap_high_task_count": len({row[0] for row in lifetime_pairs}),
        "same_window_recompetition_pair_count": len(window_pairs),
        "same_window_recompetition_high_task_count": len({row[0] for row in window_pairs}),
        "ordinary_staged_delivered_count": len(ordinary_delivered),
        "ordinary_staged_delivered_mb": sum(_f(row.get("payload_size_mb")) for row in ordinary_delivered),
        "displacement_supported_by_trace": int(bool(lifetime_pairs or window_pairs)),
    }


def run() -> None:
    if REPORT.exists() or TASK_CSV.exists():
        raise FileExistsError("high-value staged activation audit outputs already exist")
    outcome_path = PROP_SOURCE / "task_level_ground_outcomes.csv"
    ledger_path = PROP_SOURCE / "prop_full_lifecycle_ledger_trace.csv"
    request_path = PROP_SOURCE / "prop_full_staged_request_trace.csv"
    recompetition_path = PROP_SOURCE / "prop_full_recompetition_trace.csv"
    paths = [outcome_path, ledger_path, request_path, recompetition_path]
    before = {str(path): _hash(path) for path in paths}

    all_outcomes = read_csv(outcome_path)
    outcomes = [
        row for row in all_outcomes
        if row.get("method") == "PROP-FULL"
        and row.get("value_level") in {"high_value", "emergency"}
        and row.get("product_requirement") in {"context", "evidence"}
    ]
    if len(outcomes) != 111 or len({row["task_id"] for row in outcomes}) != 111:
        raise AssertionError("expected 111 unique high/urgent context/evidence tasks")
    task_ids = {row["task_id"] for row in outcomes}
    ledger = read_csv(ledger_path)
    requests = read_csv(request_path)
    recompetition = read_csv(recompetition_path)
    task_rows = build_task_audit(outcomes, ledger, requests, runtime_level_evidence="medium")

    high_created = [
        row for row in ledger
        if row.get("task_id") in task_ids
        and row.get("source") == "staged_delivery_queue"
        and row.get("current_status") == "created"
    ]
    ordinary_delivered = [
        row for row in ledger
        if row.get("task_id") not in task_ids
        and row.get("source") == "staged_delivery_queue"
        and row.get("current_status") == "delivered"
    ]
    high_recompetition = [row for row in recompetition if row.get("task_id") in task_ids and row.get("product_type") == "roi"]
    ordinary_recompetition = [row for row in recompetition if row.get("task_id") not in task_ids and row.get("product_type") == "roi"]
    capacity = capacity_overlap_audit(high_created, ordinary_delivered, high_recompetition, ordinary_recompetition)

    request_levels = Counter(
        str(row.get("value_level", ""))
        for row in requests
        if row.get("action") == "created_pending"
    )
    conditions = Counter(row["exact_failed_condition"] for row in task_rows)
    blockers = Counter(
        reason
        for row in task_rows
        for reason in str(row["blocker_reasons"]).split(";")
        if reason
    )
    checks = {
        "111_unique_target_tasks": len(task_rows) == len({row["task_id"] for row in task_rows}) == 111,
        "all_85_created_requests_medium": request_levels == Counter({"medium": 85}),
        "27_target_staged_roi_requests": sum(_i(row["staged_request_created"]) for row in task_rows) == 27,
        "zero_target_staged_deliveries": not any(
            row.get("task_id") in task_ids and row.get("source") == "staged_delivery_queue" and row.get("current_status") == "delivered"
            for row in ledger
        ),
        "no_trace_supported_ordinary_displacement": capacity["displacement_supported_by_trace"] == 0,
        "no_nan_inf": all(
            math.isfinite(float(value))
            for row in task_rows
            for value in row.values()
            if isinstance(value, (int, float))
        ),
    }
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])

    write_csv(TASK_CSV, task_rows)
    lines = [
        "# High-Value Staged Activation Audit, seed=2026",
        "",
        "This is a passive audit of existing traces only. No policy, simulator, workload, or historical trace was modified or rerun.",
        "",
        "## Root cause",
        "",
        "The apparent contradiction comes from two different value labels. The final task table calls these 111 tasks `high_value`, but every staged request persisted the internal `task.value_level` as `medium`. The staged constructor and ROI→ROI+Context gate use this internal value level, not the final task-class label. Consequently all 85 created staged requests are recorded as medium, and no ROI→ROI+Context request can pass the `{high, urgent}` gate.",
        "",
        "This is a runtime label-propagation/semantic mismatch, not evidence that the high-value task population was absent.",
        "",
        "## High/urgent context/evidence funnel",
        "",
        f"- Unique target tasks: {len(task_rows)}",
        f"- Metadata generated/delivered: {sum(_i(r['metadata_generated']) for r in task_rows)}/{sum(_i(r['metadata_delivered']) for r in task_rows)}",
        f"- ROI generated/delivered: {sum(_i(r['roi_generated']) for r in task_rows)}/{sum(_i(r['roi_delivered']) for r in task_rows)}",
        f"- ROI+Context generated/delivered directly: {sum(_i(r['roi_context_generated']) for r in task_rows)}/{sum(_i(r['roi_context_delivered']) for r in task_rows)}",
        f"- Eligible for staged ROI after metadata: {sum(_i(r['eligible_for_staged_roi']) for r in task_rows)}",
        f"- Staged ROI requests created: {sum(_i(r['staged_request_created']) for r in task_rows)}; delivered: 0; expired: {sum('expired' in r['staged_request_statuses'] for r in task_rows)}",
        f"- ROI deliveries reaching the ROI+Context pre-value gate: {sum(_i(r['eligible_for_staged_roi_context_before_value_gate']) for r in task_rows)}",
        f"- ROI deliveries passing the internal high/urgent value gate: {sum(_i(r['eligible_for_staged_roi_context_after_value_gate']) for r in task_rows)}",
        f"- Direct ROI+Context closures: {sum(_i(r['closed_directly']) for r in task_rows)}",
        "",
        "## Exact failed-condition counts",
        "",
        *[f"- `{name}`: {count}" for name, count in sorted(conditions.items())],
        "",
        "Observed staged blocker reasons:",
        "",
        *[f"- `{name}`: {count}" for name, count in sorted(blockers.items())],
        "",
        "## Ordinary staged capacity impact",
        "",
        f"Ordinary staged requests delivered {capacity['ordinary_staged_delivered_count']} payloads totaling {capacity['ordinary_staged_delivered_mb']:.1f} MB. However, there were {capacity['active_lifetime_overlap_pair_count']} ordinary-delivery/high-request active-lifetime overlaps and {capacity['same_window_recompetition_pair_count']} same-slot/contact-window recompetition overlaps. Therefore existing traces do **not** support a claim that ordinary staged traffic delayed, displaced, or consumed a contact opportunity usable by an eligible high-value staged request.",
        "",
        "This is a trace-supported conclusion, not a counterfactual proof: unlogged alternative schedules were not simulated. The observed high-value staged failures are instead explained by very short request lifetimes plus `followup_hard_safety_blocked_deadline` and `no_contact_opportunity`, while ROI→ROI+Context activation is suppressed by the medium internal value label.",
        "",
        "## Consistency checks",
        "",
        *[f"- {'PASS' if passed else 'FAIL'} `{name}`" for name, passed in checks.items()],
        "",
        "## Input hashes",
        "",
        *[f"- `{Path(path).relative_to(ROOT)}`: `{digest}`" for path, digest in before.items()],
        "",
        f"Task-level evidence: `{TASK_CSV.name}`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {str(path): _hash(path) for path in paths}
    if after != before:
        raise AssertionError("an input trace changed during the passive audit")
    print(json.dumps({"tasks": len(task_rows), "checks": checks, "capacity": capacity}, indent=2))


if __name__ == "__main__":
    run()
