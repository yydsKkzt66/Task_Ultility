from __future__ import annotations

from src.common.types import (
    DownlinkResult,
    DetectionResult,
    GeneratedPayload,
    LinkTrace,
    CommState,
    ObservedFeedback,
    ProductLifecycleState,
    ResourceState,
    ResourceTrace,
    TaskSample,
    TimelineStep,
)
from src.link.downlink_queue import DownlinkQueue
from src.payload.continue_request import evaluate_virtual_continue_request
from src.simulator.step_runner import StepRunner


def _cfg(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "residual_utility_threshold": 0.08,
        "queue_pressure_threshold": 0.65,
        "min_remaining_deadline_s": 60.0,
        "battery_guard_threshold": 0.60,
        "thermal_guard_threshold": 0.85,
        "continue_request_priority_bonus": 0.05,
    }


def _state(**kwargs) -> ProductLifecycleState:
    state = ProductLifecycleState(
        task_id="task_phase5",
        product_requirement=kwargs.pop("product_requirement", "verification"),
        delivered_products=set(kwargs.pop("delivered_products", {"metadata"})),
        pending_products=set(kwargs.pop("pending_products", {"roi"})),
        residual_utility_remaining=kwargs.pop("residual_utility_remaining", 0.20),
    )
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def _system(**kwargs) -> dict:
    values = {
        "queue_mb": 1.0,
        "queue_max_mb": 64.0,
        "queued_products": set(),
        "remaining_deadline_s": 180.0,
        "battery_soc": 0.85,
        "thermal_pressure": 0.25,
        "task_value": 0.9,
        "high_value_flag": True,
        "emergency_flag": False,
        "uncertainty": 0.4,
    }
    values.update(kwargs)
    return values


def _payload(task_id: str, product_type: str, utility: float, is_residual: bool = False, bonus: float = 0.0):
    return GeneratedPayload(
        task_id=task_id,
        payload_policy=product_type,
        payload_mb=1.0,
        information_content=0.5,
        priority_score=0.5,
        value_score=0.8,
        task_value=0.8,
        uncertainty=0.2,
        remaining_utility=utility,
        is_metadata=(product_type == "metadata"),
        is_high_value=True,
        deadline_s=180.0,
        created_slot=0,
        payload_plan=product_type,
        product_type=product_type,
        product_requirement="verification",
        incremental_utility_estimate=utility,
        is_residual=is_residual,
        duplicate_identity_key=f"{task_id}:{product_type}",
        continue_request_priority_bonus=bonus,
    )


def _feedback() -> ObservedFeedback:
    return ObservedFeedback(
        S_gen_current_mb=1.0,
        S_down_total_mb=1.0,
        S_down_current_task_mb=1.0,
        queue_next_mb=0.0,
        delta_queue_mb=0.0,
        E_comp=0.0,
        E_comm=0.0,
        L_inf=0.0,
        L_queue=0.0,
        L_comm=0.0,
        delivery_status="delivered",
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=0.0,
        dropped_current_task_mb=0.0,
        queue_item_count=0,
        detection_quality=1.0,
    )


def _timeline_step(task: TaskSample) -> TimelineStep:
    return TimelineStep(
        slot_index=1,
        slot_seconds=60.0,
        task=task,
        resource_trace=ResourceTrace(
            slot_index=1,
            solar_energy_j=0.0,
            base_power_w=0.0,
            temperature_reference_c=0.0,
        ),
        link_trace=LinkTrace(1, True, 20.0, 8.0, 1.0, 60.0),
    )


def test_virtual_continue_request_false_before_metadata_delivered():
    decision = evaluate_virtual_continue_request(
        _state(delivered_products=set()),
        _system(),
        _cfg(),
    )
    assert not decision.request_flag
    assert "metadata_not_delivered" in decision.blockers


def test_virtual_continue_request_false_if_no_residual_pending():
    decision = evaluate_virtual_continue_request(
        _state(pending_products=set()),
        _system(),
        _cfg(),
    )
    assert not decision.request_flag
    assert "no_pending_residual_product" in decision.blockers


def test_virtual_continue_request_false_if_residual_utility_below_threshold():
    decision = evaluate_virtual_continue_request(
        _state(residual_utility_remaining=0.08),
        _system(),
        _cfg(),
    )
    assert not decision.request_flag
    assert "residual_utility_below_threshold" in decision.blockers


def test_virtual_continue_request_false_if_task_expired():
    decision = evaluate_virtual_continue_request(
        _state(),
        _system(remaining_deadline_s=30.0),
        _cfg(),
    )
    assert not decision.request_flag
    assert "no_useful_remaining_deadline" in decision.blockers


def test_virtual_continue_request_false_if_duplicate_residual_queued():
    decision = evaluate_virtual_continue_request(
        _state(),
        _system(queued_products={"task_phase5:roi"}),
        _cfg(),
    )
    assert not decision.request_flag
    assert "duplicate_residual_already_queued" in decision.blockers


def test_virtual_continue_request_true_for_high_value_verification_task():
    decision = evaluate_virtual_continue_request(
        _state(product_requirement="verification"),
        _system(high_value_flag=True, uncertainty=0.5),
        _cfg(),
    )
    assert decision.request_flag
    assert decision.residual_product_type == "roi"
    assert decision.leakage_safe_flag


def test_virtual_continue_request_does_not_use_forbidden_ground_truth_fields():
    decision = evaluate_virtual_continue_request(_state(), _system(), _cfg())
    forbidden = {
        "sarfish_label",
        "bbox_count",
        "target_count",
        "object_class",
        "detector_ground_truth",
        "future_delivered_outcome",
        "future_utility_realization",
    }
    assert forbidden.isdisjoint(set(decision.fields_used))


def test_phase5_disabled_is_phase0_compatible_no_request():
    decision = evaluate_virtual_continue_request(_state(), _system(), _cfg(enabled=False))
    assert not decision.request_flag
    assert decision.reason_code == "phase5_disabled"


def test_duplicate_virtual_requests_do_not_enqueue_duplicate_residuals():
    queue = DownlinkQueue(queue_max_mb=4.0, slot_seconds=10.0)
    no_contact = LinkTrace(0, False, -3.0, 0.0, 0.0, 0.0)
    first = queue.process_slot(0, _payload("task_dup_residual", "roi", 0.2, is_residual=True), no_contact)
    duplicate = queue.process_slot(1, _payload("task_dup_residual", "roi", 0.2, is_residual=True), no_contact)
    assert first.residual_enqueue_count == 1
    assert duplicate.duplicate_residual_suppression_count == 1
    assert duplicate.queue_item_count == 1


def test_residual_product_gives_incremental_utility_only_once():
    state = _state(delivered_products={"metadata", "roi"}, pending_products={"roi"})
    decision = evaluate_virtual_continue_request(state, _system(), _cfg())
    assert not decision.request_flag
    assert "product_already_delivered" in decision.blockers


def test_queue_priority_uses_incremental_utility_per_mb_and_bonus_not_utility():
    queue = DownlinkQueue(queue_max_mb=4.0, slot_seconds=10.0, continue_request_priority_bonus=0.05)
    item = queue._make_queue_item(_payload("task_priority", "roi", 0.4, is_residual=True, bonus=0.05))
    components = queue._priority_components(item, 0.0)
    assert components["utility_density_raw"] == item.remaining_utility / item.remaining_mb
    assert components["continue_request_bonus_component"] == 0.05
    assert item.incremental_utility == 0.4


def test_residual_delivery_updates_lifecycle_and_incremental_utility():
    task = TaskSample(
        "task_lifecycle",
        "a",
        "b",
        "scene",
        0,
        1.0,
        240.0,
        0.2,
        0.4,
        product_requirement="verification",
    )
    runner = StepRunner(None, None, None, None, None, None, None, {"high_value_threshold": 0.75}, {"epsilon": 1.0e-9})
    state = runner._get_product_lifecycle(task)
    state.delivered_products.add("metadata")
    state.pending_products.add("roi")
    state.first_alert_delivered_time_s = 0.0
    downlink_result = DownlinkResult(
        S_down_total_mb=1.0,
        S_down_current_task_mb=1.0,
        generated_payload_current_mb=1.0,
        dropped_current_task_mb=0.0,
        queue_next_mb=0.0,
        delta_queue_mb=0.0,
        queue_latency_s=0.0,
        queue_item_count=0,
        current_task_delivery_status="delivered",
        delivered_task_ids=[task.task_id],
        delivered_product_types=["roi"],
        delivered_incremental_utilities=[0.55],
        delivered_task_event_times_s=[60.0],
        residual_delivery_count=1,
        residual_queue_wait_times_s=[10.0],
    )
    feedback = _feedback()
    runner._apply_product_lifecycle_updates(
        _timeline_step(task),
        _payload(task.task_id, "roi", 0.55, True),
        downlink_result,
        feedback,
    )
    assert state.delivered_products == {"metadata", "roi"}
    assert "roi" not in state.pending_products
    assert feedback.passive_followup_incremental_utility > 0.0
    assert feedback.residual_delivery_count == 1


def test_residual_expiration_clears_pending_product():
    task = TaskSample("task_expire", "a", "b", "scene", 0, 1.0, 240.0, 0.2, 0.4)
    runner = StepRunner(None, None, None, None, None, None, None, {"high_value_threshold": 0.75}, {"epsilon": 1.0e-9})
    state = runner._get_product_lifecycle(task)
    state.delivered_products.add("metadata")
    state.pending_products.add("roi")
    downlink_result = DownlinkResult(
        S_down_total_mb=0.0,
        S_down_current_task_mb=0.0,
        generated_payload_current_mb=0.0,
        dropped_current_task_mb=1.0,
        queue_next_mb=0.0,
        delta_queue_mb=0.0,
        queue_latency_s=0.0,
        queue_item_count=0,
        current_task_delivery_status="expired",
        expired_task_ids=[task.task_id],
        expired_product_types=["roi"],
        residual_expired_count=1,
    )
    feedback = _feedback()
    runner._apply_product_lifecycle_updates(_timeline_step(task), None, downlink_result, feedback)
    assert "roi" not in state.pending_products
    assert state.residual_expired_count == 1


def test_enabled_phase5_runner_builds_and_enqueues_residual_payload():
    task = TaskSample(
        "task_enabled_followup",
        "a",
        "b",
        "scene",
        0,
        1.0,
        240.0,
        0.2,
        0.4,
        product_requirement="verification",
        high_value_flag=True,
    )
    queue = DownlinkQueue(queue_max_mb=4.0, slot_seconds=60.0)
    runner = StepRunner(
        None,
        None,
        None,
        None,
        None,
        queue,
        None,
        {"high_value_threshold": 0.75},
        {
            "epsilon": 1.0e-9,
            "progressive_payload": {
                "phase5_virtual_continue": {
                    "enabled": True,
                    "residual_utility_threshold": 0.08,
                    "queue_pressure_threshold": 0.65,
                    "min_remaining_deadline_s": 60.0,
                    "battery_guard_threshold": 0.60,
                    "thermal_guard_threshold": 0.85,
                    "continue_request_priority_bonus": 0.05,
                }
            },
        },
    )
    state = runner._get_product_lifecycle(task)
    state.delivered_products.add("metadata")
    state.pending_products.add("roi")
    state.first_alert_delivered_time_s = 0.0
    state.residual_utility_remaining = 0.20
    detection = DetectionResult([], 0.9, False, [], 0.2, 1.0, 0.9, {})
    residual_payload = runner._maybe_build_residual_payload(
        _timeline_step(task),
        ResourceState(battery_soc=0.85, ai_energy_budget=1.0, thermal_pressure=0.2),
        CommState(queue_mb=0.0, queue_max_mb=4.0),
        detection,
        _payload(task.task_id, "metadata", 0.35),
        state,
    )
    result = queue.process_slot(1, residual_payload, LinkTrace(1, True, 20.0, 8.0, 1.0, 60.0))
    assert residual_payload.is_residual
    assert residual_payload.product_type == "roi"
    assert result.residual_enqueue_count == 1
