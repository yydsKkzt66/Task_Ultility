from __future__ import annotations

from types import SimpleNamespace

from src.common.types import CommState, GeneratedPayload, LinkTrace, ProductLifecycleState, ResourceState, ResourceTrace, TaskSample, TimelineStep
from src.link.downlink_queue import DownlinkQueue
from src.payload.payload_lifecycle_ledger import PayloadLifecycleLedger
from src.payload.staged_delivery_queue import StagedDeliveryRequest, compute_staged_priority, staged_delivery_queue_config
from src.simulator.step_runner import StepRunner


def _payload(
    task_id: str,
    product_type: str,
    *,
    payload_mb: float = 0.4,
    task_value: float = 0.9,
    created_slot: int = 0,
    deadline_s: float = 100.0,
    is_residual: bool = False,
) -> GeneratedPayload:
    return GeneratedPayload(
        task_id=task_id,
        payload_policy=product_type,
        payload_mb=payload_mb,
        information_content=0.5,
        priority_score=0.8,
        value_score=task_value,
        task_value=task_value,
        uncertainty=0.2,
        remaining_utility=task_value * 0.5,
        is_metadata=product_type == "metadata",
        is_high_value=task_value >= 0.75,
        deadline_s=deadline_s,
        created_slot=created_slot,
        metadata={"value_level": "high" if task_value >= 0.75 else "low", "product_requirement": "context"},
        payload_plan=product_type,
        product_type=product_type,
        product_requirement="context",
        incremental_utility_estimate=task_value * 0.5,
        incremental_information_utility=0.5,
        is_residual=is_residual,
        duplicate_identity_key=f"{task_id}:{product_type}",
    )


def _link(slot: int, in_contact: bool, throughput_mbps: float = 8.0) -> LinkTrace:
    return LinkTrace(slot, in_contact, 10.0, throughput_mbps if in_contact else 0.0, 1.0, 10.0 if in_contact else 0.0)


def _queue() -> tuple[DownlinkQueue, PayloadLifecycleLedger]:
    ledger = PayloadLifecycleLedger(slot_seconds=10.0)
    return DownlinkQueue(queue_max_mb=5.0, slot_seconds=10.0, lifecycle_ledger=ledger), ledger


def _cfg() -> dict[str, object]:
    return {
        "epsilon": 1.0e-9,
        "battery_min": 0.18,
        "thermal_max": 0.92,
        "queue_max_mb": 10.0,
        "slot_seconds": 10.0,
        "battery_capacity_j": 100.0,
        "battery_charge_efficiency": 1.0,
        "battery_discharge_efficiency": 1.0,
        "comm_power_w": 5.0,
        "tx_power_w": 5.0,
        "comm_alpha_gamma": 0.0,
        "downlink_efficiency": 1.0,
        "cooling_per_slot": 0.0,
        "thermal_reference_slot_seconds": 10.0,
        "thermal_passive_cooling": 0.0,
        "thermal_sunlight_gain": 0.0,
        "thermal_eclipse_cooling_bonus": 0.0,
        "thermal_reference_min_c": -10.0,
        "thermal_reference_max_c": 30.0,
        "high_value_threshold": 0.75,
        "progressive_payload": {
            "value_aware_staged_delivery_queue": {
                "enabled": True,
                "full_method_aliases": ["proposed_lyapunov_rule_mode_safe_bandit"],
                "residual_utility_threshold": 0.0,
                "require_future_contact_before_deadline": True,
                "allow_pending_without_future_contact": True,
                "max_admissions_per_slot": 2,
                "max_admissions_per_contact_window": 4,
            }
        },
    }


def _task(task_id: str = "hv") -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_path="mock://image",
        label_path="mock://label",
        scene_id=f"scene_{task_id}",
        virtual_time=0,
        task_value=0.9,
        deadline_s=100.0,
        difficulty=0.1,
        uncertainty=0.2,
        high_value_flag=True,
        product_requirement="context",
        value_level="high",
        calibrated_task_value=1.2,
    )


def _timeline(task: TaskSample | None, slot: int, *, in_contact: bool) -> TimelineStep:
    return TimelineStep(
        slot_index=slot,
        slot_seconds=10.0,
        task=task,
        resource_trace=ResourceTrace(slot, 0.0, 0.0, 10.0, is_sunlit=False),
        link_trace=_link(slot, in_contact),
    )


def _runner() -> StepRunner:
    runner = StepRunner.__new__(StepRunner)
    runner.simulator_cfg = _cfg()
    runner.reward_cfg = {"high_value_threshold": 0.75}
    runner.payload_profiles_cfg = None
    runner.downlink_queue = DownlinkQueue(10.0, 10.0)
    runner.product_lifecycle = {}
    runner.future_link_traces = [_link(0, False), _link(1, True), _link(2, True)]
    runner._roi_context_budget_slot_index = -1
    runner._roi_context_slot_selected_count = 0
    runner._roi_context_queued_budget_slot_index = -1
    runner._roi_context_queued_selected_count = 0
    runner._roi_context_contact_window_id = ""
    runner._roi_context_contact_window_selected_count = 0
    runner._roi_context_window_selected_count = 0
    runner.last_future_contact_feasibility_trace = []
    runner.last_duplicate_suppression_trace = []
    runner._ensure_staged_delivery_state()
    runner._reset_staged_delivery_step_logs()
    return runner


def _staged_request(task_id: str = "hv") -> StagedDeliveryRequest:
    request = StagedDeliveryRequest(
        request_id=f"req_{task_id}",
        task_id=task_id,
        product_type="roi_context",
        parent_product_type="roi",
        product_requirement="context",
        value_level="high",
        calibrated_task_value=1.2,
        residual_delta_information_utility=0.4,
        residual_utility_remaining=0.4,
        detection_quality=1.0,
        timeliness_factor=1.0,
        payload_size_mb=0.4,
        deadline_slot=2,
        created_slot=0,
        last_evaluated_slot=0,
        earliest_delivery_slot=1,
        status="blocked",
        duplicate_key=f"{task_id}:roi_context",
        deadline_s=100.0,
    )
    request.priority_score, request.priority_components = compute_staged_priority(request, current_slot=0, config=_cfg())
    return request


def test_generated_payload_always_has_lifecycle_record():
    queue, ledger = _queue()
    queue.process_slot(0, _payload("t1", "metadata"), _link(0, False))
    assert any(row["task_id"] == "t1" and row["product_type"] == "metadata" for row in ledger.transition_trace)


def test_pending_payload_recompetes_next_contact_window():
    queue, ledger = _queue()
    queue.process_slot(0, _payload("t2", "metadata"), _link(0, False), contact_window_id="")
    result = queue.process_slot(1, None, _link(1, True), contact_window_id="contact_window_1")
    assert result.delivered_item_count == 1
    assert any(row["slot"] == 1 and row["task_id"] == "t2" and row["reevaluated_flag"] for row in ledger.recompetition_trace)


def test_blocked_payload_not_dropped_without_reason():
    queue, ledger = _queue()
    queue.process_slot(0, _payload("t3", "metadata"), _link(0, False))
    record = next(record for record in ledger.records.values() if record.task_id == "t3")
    assert record.current_status == "blocked"
    assert record.blocker_reason == "no_contact_opportunity"


def test_expired_payload_has_expiration_reason():
    queue, ledger = _queue()
    queue.process_slot(0, _payload("t4", "metadata", deadline_s=10.0), _link(0, False))
    queue.process_slot(2, None, _link(2, False))
    record = next(record for record in ledger.records.values() if record.task_id == "t4")
    assert record.current_status == "expired"
    assert record.transition_reason == "deadline_passed"


def test_delivered_payload_updates_lifecycle_once():
    queue, ledger = _queue()
    queue.process_slot(0, _payload("t5", "metadata", payload_mb=0.2), _link(0, True), contact_window_id="contact_window_0")
    delivered = [
        row
        for row in ledger.transition_trace
        if row["task_id"] == "t5" and row["product_type"] == "metadata" and row["current_status"] == "delivered"
    ]
    assert len(delivered) == 1


def test_duplicate_task_product_does_not_overwrite_lifecycle():
    queue, ledger = _queue()
    first_payload = _payload("t6", "metadata")
    queue.process_slot(0, first_payload, _link(0, False))
    first_record_id = next(record.lifecycle_id for record in ledger.records.values() if record.task_id == "t6")
    queue.process_slot(1, _payload("t6", "metadata"), _link(1, False))
    assert ledger.records[first_record_id].current_status == "blocked"
    dropped_duplicates = [record for record in ledger.records.values() if record.task_id == "t6" and record.current_status == "dropped"]
    assert dropped_duplicates


def test_low_value_payload_can_persist_but_lower_priority():
    queue, ledger = _queue()
    queue.process_slot(0, _payload("low", "roi_context", payload_mb=1.0, task_value=0.2), _link(0, False))
    queue.process_slot(0, _payload("high", "metadata", payload_mb=0.4, task_value=0.95), _link(0, False))
    result = queue.process_slot(1, None, _link(1, True), capacity_override_mb=0.4, contact_window_id="contact_window_1")
    assert "high" in result.delivered_task_ids
    assert any(item.task_id == "low" for item in queue.items)
    low_record = next(record for record in ledger.records.values() if record.task_id == "low")
    assert low_record.current_status in {"blocked", "queued"}


def test_high_value_payload_persists_across_no_contact():
    queue, ledger = _queue()
    queue.process_slot(0, _payload("t8", "roi_context", is_residual=True), _link(0, False))
    assert any(item.task_id == "t8" for item in queue.items)
    record = next(record for record in ledger.records.values() if record.task_id == "t8")
    assert record.current_status == "blocked"


def test_next_window_scans_pending_staged_requests():
    runner = _runner()
    task = _task("staged")
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle[task.task_id] = ProductLifecycleState(
        task.task_id,
        product_requirement="context",
        delivered_products={"metadata", "roi"},
        pending_products={"roi_context"},
    )
    request = _staged_request("staged")
    runner.staged_delivery_requests[request.duplicate_key] = request
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(None, 1, in_contact=True),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name="proposed_lyapunov_rule_mode_safe_bandit"),
    )
    assert any(row["task_id"] == "staged" and row["slot"] == 1 for row in runner.lifecycle_ledger.recompetition_trace)
    assert runner.staged_delivery_requests["staged:roi_context"].status == "enqueued"


def test_orphan_lifecycle_check_detects_missing_queue_item():
    ledger = PayloadLifecycleLedger(slot_seconds=10.0)
    record = ledger.create_record(task_id="orphan", product_type="metadata", slot=0, payload_size_mb=0.3)
    ledger.transition(record.lifecycle_id, "queued", slot=0, reason="test_missing_queue_item")
    rows = ledger.orphan_check_rows(queue_items=[], staged_requests=[])
    assert any(row["orphan_flag"] and row["task_id"] == "orphan" for row in rows)


def test_utility_rule_unchanged():
    runner = _runner()
    assert not runner._staged_delivery_enabled(SimpleNamespace(router_name="utility_rule"))


def test_legacy_default_unchanged():
    assert staged_delivery_queue_config({})["enabled"] is False
