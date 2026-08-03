from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.common.types import CommState, LinkTrace, ProductLifecycleState, ResourceState, ResourceTrace, TaskSample, TimelineStep
from src.link.downlink_queue import DownlinkQueue
from src.payload.staged_delivery_queue import StagedDeliveryRequest, compute_staged_priority, staged_delivery_queue_config
from src.simulator.step_runner import StepRunner


def _cfg(**queue_overrides) -> dict[str, object]:
    staged_cfg = {
        "enabled": True,
        "full_method_aliases": ["proposed_lyapunov_rule_mode_safe_bandit"],
        "residual_utility_threshold": 0.0,
        "require_future_contact_before_deadline": True,
        "max_admissions_per_slot": 2,
        "max_admissions_per_contact_window": 4,
        "deadline_urgency_lambda": 0.2,
    }
    staged_cfg.update(queue_overrides)
    return {
        "epsilon": 1.0e-9,
        "battery_min": 0.18,
        "thermal_max": 0.92,
        "queue_max_mb": 10.0,
        "slot_seconds": 20.0,
        "battery_capacity_j": 100.0,
        "battery_charge_efficiency": 1.0,
        "battery_discharge_efficiency": 1.0,
        "comm_power_w": 20.0,
        "tx_power_w": 20.0,
        "comm_alpha_gamma": 0.0,
        "downlink_efficiency": 1.0,
        "cooling_per_slot": 0.0,
        "thermal_reference_slot_seconds": 20.0,
        "thermal_passive_cooling": 0.0,
        "thermal_sunlight_gain": 0.0,
        "thermal_eclipse_cooling_bonus": 0.0,
        "thermal_reference_min_c": -10.0,
        "thermal_reference_max_c": 30.0,
        "high_value_threshold": 0.75,
        "progressive_payload": {"value_aware_staged_delivery_queue": staged_cfg},
    }


def _runner(cfg: dict[str, object] | None = None, future: list[LinkTrace] | None = None) -> StepRunner:
    runner = StepRunner.__new__(StepRunner)
    runner.simulator_cfg = cfg or _cfg()
    runner.reward_cfg = {"high_value_threshold": 0.75}
    runner.payload_profiles_cfg = None
    runner.downlink_queue = DownlinkQueue(10.0, 20.0)
    runner.product_lifecycle = {}
    runner.future_link_traces = future or [LinkTrace(i, True, 10.0, 80.0, 1.0, 20.0) for i in range(8)]
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


def _task(task_id: str, *, value_level: str = "high", requirement: str = "context", deadline_s: float = 120.0) -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_path="mock://image",
        label_path="mock://label",
        scene_id=f"scene_{task_id}",
        virtual_time=0,
        task_value=0.8,
        deadline_s=deadline_s,
        difficulty=0.1,
        uncertainty=0.2,
        high_value_flag=value_level in {"high", "urgent"},
        emergency_flag=value_level == "urgent",
        product_requirement=requirement,
        value_level=value_level,
        calibrated_task_value=1.5 if value_level in {"high", "urgent"} else 0.4,
    )


def _timeline(task: TaskSample | None, slot: int, *, in_contact: bool) -> TimelineStep:
    return TimelineStep(
        slot_index=slot,
        slot_seconds=20.0,
        task=task,
        resource_trace=ResourceTrace(slot, 0.0, 0.0, 10.0, is_sunlit=False),
        link_trace=LinkTrace(slot, in_contact, 10.0, 80.0 if in_contact else 0.0, 1.0, 20.0 if in_contact else 0.0),
    )


def _router(name: str = "proposed_lyapunov_rule_mode_safe_bandit"):
    return SimpleNamespace(router_name=name)


def _request(task_id: str, *, value: float = 1.0, delta: float = 0.3, payload: float = 1.0, deadline_slot: int = 6):
    request = StagedDeliveryRequest(
        request_id=f"req_{task_id}",
        task_id=task_id,
        product_type="roi_context",
        parent_product_type="roi",
        product_requirement="context",
        value_level="high",
        calibrated_task_value=value,
        residual_delta_information_utility=delta,
        residual_utility_remaining=delta,
        detection_quality=1.0,
        timeliness_factor=1.0,
        payload_size_mb=payload,
        deadline_slot=deadline_slot,
        created_slot=0,
        last_evaluated_slot=0,
        earliest_delivery_slot=1,
        duplicate_key=f"{task_id}:roi_context",
        deadline_s=120.0,
    )
    request.priority_score, request.priority_components = compute_staged_priority(request, current_slot=0, config=_cfg())
    return request


def test_high_value_roi_context_request_persists_without_current_contact():
    task = _task("hv")
    future = [LinkTrace(0, False, 10.0, 0.0, 1.0, 0.0), LinkTrace(1, True, 10.0, 80.0, 1.0, 20.0)]
    runner = _runner(future=future)
    runner.task_snapshots[task.task_id] = task
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"metadata", "roi"})
    runner.product_lifecycle[task.task_id] = state
    runner._maybe_create_staged_delivery_request_after_delivery(
        timeline_step=_timeline(task, 0, in_contact=False),
        state=state,
        task=task,
        delivered_product_type="roi",
        delivered_metadata={"detection_quality": 0.9},
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
    )
    request = runner.staged_delivery_requests["hv:roi_context"]
    assert request.status == "pending"
    assert request.value_level == "high"


def test_no_contact_never_same_slot_delivers():
    task = _task("hv")
    runner = _runner(future=[LinkTrace(i, False, 10.0, 0.0, 1.0, 0.0) for i in range(4)])
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle[task.task_id] = ProductLifecycleState(task.task_id, product_requirement="context")
    runner.staged_delivery_requests["hv:roi_context"] = _request("hv")
    result = runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(None, 1, in_contact=False),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=_router(),
    )
    assert result is None
    assert not runner.downlink_queue.items
    assert runner.last_contact_window_scheduling_trace[-1]["reason_code"] == "no_contact_opportunity"


def test_next_contact_prioritizes_high_value_roi_context():
    high = _task("high")
    low = _task("low", value_level="medium")
    runner = _runner()
    runner.task_snapshots.update({high.task_id: high, low.task_id: low})
    runner.product_lifecycle[high.task_id] = ProductLifecycleState(high.task_id, product_requirement="context")
    runner.product_lifecycle[low.task_id] = ProductLifecycleState(low.task_id, product_requirement="context")
    runner.staged_delivery_requests["low:roi_context"] = _request("low", value=0.2, delta=0.3)
    runner.staged_delivery_requests["high:roi_context"] = _request("high", value=2.0, delta=0.4)
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(None, 1, in_contact=True),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=_router(),
    )
    assert runner.downlink_queue.items[0].task_id == "high"


def test_low_medium_not_allowed_to_roi_context():
    task = _task("m", value_level="medium", requirement="context")
    runner = _runner()
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"metadata", "roi"})
    runner._maybe_create_staged_delivery_request_after_delivery(
        timeline_step=_timeline(task, 0, in_contact=True),
        state=state,
        task=task,
        delivered_product_type="roi",
        delivered_metadata={},
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
    )
    assert "m:roi_context" not in runner.staged_delivery_requests


def test_verification_task_stops_at_roi_when_marginal_roi_context_low():
    task = _task("v", value_level="high", requirement="verification")
    runner = _runner()
    state = ProductLifecycleState(task.task_id, product_requirement="verification", delivered_products={"metadata", "roi"})
    runner._maybe_create_staged_delivery_request_after_delivery(
        timeline_step=_timeline(task, 0, in_contact=True),
        state=state,
        task=task,
        delivered_product_type="roi",
        delivered_metadata={},
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
    )
    assert not runner.staged_delivery_requests


def test_context_task_can_progress_roi_to_roi_context():
    task = _task("c", value_level="high", requirement="context")
    runner = _runner()
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"metadata", "roi"})
    runner._maybe_create_staged_delivery_request_after_delivery(
        timeline_step=_timeline(task, 0, in_contact=True),
        state=state,
        task=task,
        delivered_product_type="roi",
        delivered_metadata={},
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
    )
    assert "c:roi_context" in runner.staged_delivery_requests


def test_evidence_task_can_progress_roi_to_roi_context():
    task = _task("e", value_level="urgent", requirement="evidence")
    runner = _runner()
    state = ProductLifecycleState(task.task_id, product_requirement="evidence", delivered_products={"metadata", "roi"})
    runner._maybe_create_staged_delivery_request_after_delivery(
        timeline_step=_timeline(task, 0, in_contact=True),
        state=state,
        task=task,
        delivered_product_type="roi",
        delivered_metadata={},
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
    )
    assert "e:roi_context" in runner.staged_delivery_requests


def test_priority_queue_uses_task_value_and_residual_delta():
    low = _request("a", value=1.0, delta=0.1)
    high = _request("b", value=2.0, delta=0.4)
    assert high.priority_score > low.priority_score


def test_priority_queue_uses_deadline_urgency():
    request = _request("a", value=1.0, delta=0.2, deadline_slot=10)
    early, _ = compute_staged_priority(request, current_slot=1, config=_cfg())
    late, _ = compute_staged_priority(request, current_slot=9, config=_cfg())
    assert late > early


def test_priority_queue_respects_payload_size():
    small = _request("small", value=1.0, delta=0.3, payload=0.5)
    large = _request("large", value=1.0, delta=0.3, payload=2.0)
    assert small.priority_score > large.priority_score


def test_priority_queue_respects_hard_safety_guard():
    task = _task("hv")
    runner = _runner()
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle[task.task_id] = ProductLifecycleState(task.task_id, product_requirement="context")
    request = _request("hv", payload=2.0)
    runner.staged_delivery_requests[request.duplicate_key] = request
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(None, 1, in_contact=True),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(9.5, 10.0),
        router=_router(),
    )
    assert runner.last_contact_window_scheduling_trace[-1]["decision"] == "blocked_by_hard_safety_guard"


def test_expired_request_is_dropped():
    runner = _runner()
    request = _request("old", deadline_slot=0)
    runner.staged_delivery_requests[request.duplicate_key] = request
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(None, 2, in_contact=True),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=_router(),
    )
    assert "old:roi_context" not in runner.staged_delivery_requests
    assert runner.staged_delivery_history["old:roi_context"].status == "expired"


def test_duplicate_task_product_request_suppressed():
    task = _task("dup")
    runner = _runner()
    existing = _request("dup")
    runner.staged_delivery_requests[existing.duplicate_key] = existing
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"metadata", "roi"})
    runner._maybe_create_staged_delivery_request_after_delivery(
        timeline_step=_timeline(task, 0, in_contact=True),
        state=state,
        task=task,
        delivered_product_type="roi",
        delivered_metadata={},
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
    )
    assert len(runner.staged_delivery_requests) == 1


def test_utility_rule_unchanged():
    runner = _runner()
    assert not runner._staged_delivery_enabled(_router("utility_rule"))


def test_legacy_default_unchanged():
    cfg = staged_delivery_queue_config({})
    assert cfg["enabled"] is False
