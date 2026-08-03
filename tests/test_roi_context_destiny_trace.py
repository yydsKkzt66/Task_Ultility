from __future__ import annotations

from pathlib import Path

from src.reporting.roi_context_destiny_trace import (
    RunTrace,
    analyze_runs,
    build_product_generation_rows,
    build_task_destiny_rows,
    select_case_studies,
    summarize_funnel_for_run,
    validate_funnel_counts_are_monotonic,
)


def _task(task_id: str = "T1", *, selected: str = "metadata") -> dict[str, str]:
    return {
        "router_name": "proposed_lyapunov_rule_mode_safe_bandit",
        "slot": "1",
        "t": "20",
        "task_id": task_id,
        "has_task": "1",
        "scene_id": "scene",
        "task_class": "science",
        "product_requirement": "context",
        "value_level": "high",
        "selected_payload_plan": selected,
        "selected_product_type": selected,
        "delivery_status": "queued",
        "deadline_missed": "0",
        "delivered_products": "",
        "pending_products": selected,
        "deadline_s": "100",
        "deadline_time_s": "120",
        "calibrated_task_value": "2.0",
        "task_value": "2.0",
    }


def _queue(slot: int, product: str, *, delivered: bool = True, task_id: str = "T1", staged: bool = False) -> dict[str, str]:
    return {
        "slot": str(slot),
        "time_s": str(slot * 20),
        "task_id": task_id,
        "payload_type": product,
        "payload_plan": product,
        "product_requirement": "context",
        "payload_size_mb": "1.0",
        "task_value": "2.0",
        "high_value_flag": "1",
        "contact_available": "1",
        "link_rate_bps": "8000000",
        "selected_for_downlink": "1",
        "delivered_this_slot": "1" if delivered else "0",
        "expired_this_slot": "0",
        "queue_priority_total": "1.0",
        "queue_order_before_downlink": "0",
        "staged_priority_total": "1.0" if staged else "",
        "followup_trigger": "value_aware_staged_delivery_queue" if staged else "",
    }


def _request(slot: int, product: str, action: str = "created_pending") -> dict[str, str]:
    return {
        "slot": str(slot),
        "request_id": f"req_{product}",
        "task_id": "T1",
        "product_type": product,
        "parent_product_type": "roi" if product == "roi_context" else "metadata",
        "product_requirement": "context",
        "value_level": "high",
        "status": "pending",
        "payload_size_mb": "1.0",
        "priority_score": "1.0",
        "deadline_slot": "6",
        "created_slot": str(slot),
        "action": action,
    }


def _run(*, include_context_queue: bool = True, context_delivered: bool = True) -> RunTrace:
    staged_requests = [
        _request(2, "roi"),
        _request(2, "roi", "enqueued"),
        _request(3, "roi_context"),
    ]
    if include_context_queue:
        staged_requests.append(_request(3, "roi_context", "enqueued"))
    queue = [_queue(1, "metadata"), _queue(2, "roi")]
    if include_context_queue:
        queue.append(_queue(3, "roi_context", delivered=context_delivered, staged=True))
    return RunTrace(
        scenario="unit",
        seed=1,
        arm="persistent",
        run_dir=Path("."),
        trajectory=[_task("T1"), _task("T1"), {**_task("low"), "value_level": "low"}],
        staged_requests=staged_requests,
        staged_priority=[
            {
                "slot": "3",
                "task_id": "T1",
                "product_type": "roi_context",
                "product_requirement": "context",
                "value_level": "high",
                "selected_for_admission": "1" if include_context_queue else "0",
                "queue_order": "0",
                "priority_score": "1.0",
            }
        ],
        scheduling=[
            {
                "slot": "3",
                "task_id": "T1",
                "product_type": "roi_context",
                "decision": "enqueued" if include_context_queue else "left_pending",
                "reason_code": "admitted_to_downlink_queue" if include_context_queue else "no_contact_opportunity",
                "in_contact": "1" if include_context_queue else "0",
            }
        ],
        queue_priority=queue,
        carryover_events=[],
        same_slot_vs_queued=[],
        future_contact=[],
        backlog=[
            {"slot": "1", "in_contact": "1", "queue_mb": "0"},
            {"slot": "2", "in_contact": "1", "queue_mb": "0"},
            {"slot": "3", "in_contact": "1" if include_context_queue else "0", "queue_mb": "0"},
        ],
    )


def test_destiny_trace_counts_unique_tasks():
    run = _run()
    destiny = build_task_destiny_rows(run)
    summary = summarize_funnel_for_run(run, destiny)

    assert len(destiny) == 1
    assert summary["total_tasks"] == 2
    assert summary["high_urgent_context_evidence_task_count"] == 1


def test_funnel_counts_are_monotonic_by_stage():
    run = _run()
    destiny = build_task_destiny_rows(run)
    summary = summarize_funnel_for_run(run, destiny)

    validate_funnel_counts_are_monotonic([summary])


def test_product_generation_trace_detects_missing_transition():
    run = _run(include_context_queue=False)
    destiny = build_task_destiny_rows(run)
    products = build_product_generation_rows(run, destiny)
    context_row = next(row for row in products if row["task_id"] == "T1" and row["product_type"] == "roi_context")

    assert context_row["missing_transition_reason"] == "staged_request_created_but_no_queue_item"


def test_case_study_selection_includes_failed_high_value_task():
    failed = build_task_destiny_rows(_run(include_context_queue=False))
    delivered = build_task_destiny_rows(_run())
    cases = select_case_studies(failed + delivered)

    assert any(not case.get("roi_context_delivered_slot") for case in cases)


def test_no_feasible_high_value_marked_missed_when_infeasible():
    run = _run(include_context_queue=False)
    destiny = build_task_destiny_rows(run)

    assert destiny[0]["was_roi_context_ever_feasible"] == 0
    assert destiny[0]["was_roi_context_feasible_but_not_delivered"] == 0


def test_lifecycle_delivered_matches_product_generation_trace():
    analysis = analyze_runs([_run()])
    context_row = next(row for row in analysis.product_rows if row["task_id"] == "T1" and row["product_type"] == "roi_context")

    assert context_row["delivered_flag"] == 1
    assert context_row["lifecycle_recorded_flag"] == 1
