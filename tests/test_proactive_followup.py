from __future__ import annotations

from src.common.types import (
    CommState,
    DetectionResult,
    GeneratedPayload,
    LinkTrace,
    ObservedFeedback,
    ProductLifecycleState,
    ResourceState,
    ResourceTrace,
    TaskSample,
    TimelineStep,
)
from src.link.downlink_queue import DownlinkQueue
from src.payload.proactive_followup import (
    compute_expected_residual_information_value,
    evaluate_proactive_followup_eligibility,
)
from src.simulator.step_runner import StepRunner


class _Router:
    router_name = "proposed_lyapunov_rule_mode_safe_bandit"
    use_lyapunov_pricing = True
    use_mode_controller = True


class _UtilRouter:
    router_name = "utility_rule"
    use_lyapunov_pricing = False
    use_mode_controller = False


def _cfg(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "eriv_threshold": 0.25,
        "residual_utility_threshold": 0.08,
        "max_proactive_residuals_per_slot": 1,
        "max_proactive_residuals_per_contact": 2,
        "queue_pressure_threshold": 0.65,
        "min_remaining_deadline_s": 60.0,
        "battery_guard_threshold": 0.60,
        "thermal_guard_threshold": 0.85,
        "require_contact_opportunity": True,
        "proactive_queue_bonus": 0.05,
    }


def _phase_cfg(
    phase6_enabled: bool = True,
    phase5_enabled: bool = True,
    carryover_enabled: bool = False,
    same_slot_enabled: bool = False,
    queue_fallback_enabled: bool = False,
) -> dict:
    return {
        "epsilon": 1.0e-9,
        "progressive_payload": {
            "phase5_virtual_continue": {
                "enabled": phase5_enabled,
                "residual_utility_threshold": 0.08,
                "queue_pressure_threshold": 0.65,
                "min_remaining_deadline_s": 60.0,
                "battery_guard_threshold": 0.60,
                "thermal_guard_threshold": 0.85,
                "continue_request_priority_bonus": 0.05,
            },
            "phase6_proactive_followup": _cfg(enabled=phase6_enabled),
            "phase6_residual_candidate_carryover": {
                "enabled": carryover_enabled,
                "re_evaluate_once_per_slot": True,
                "max_candidates_per_slot": 2,
                "max_candidate_age_s": 600,
                "require_metadata_delivered": True,
                "require_pending_residual": True,
                "require_positive_residual_utility": True,
                "remove_on_expired": True,
                "remove_on_closed": True,
                "remove_on_residual_delivered": True,
                "remove_on_duplicate_enqueued": False,
            },
            "phase6_same_slot_carryover": {
                "enabled": same_slot_enabled,
                "only_after_lifecycle_update": True,
                "reuse_existing_guards": True,
                "max_same_slot_eval_per_slot": 1,
                "enable_queue_fallback_on_insufficient_capacity": queue_fallback_enabled,
                "enforce_no_reward_change": True,
                "enforce_no_queue_policy_change": True,
                "strict_mode": True,
            },
        },
    }


def _state(**kwargs) -> ProductLifecycleState:
    state = ProductLifecycleState(
        task_id="task_phase6",
        product_requirement=kwargs.pop("product_requirement", "verification"),
        delivered_products=set(kwargs.pop("delivered_products", {"metadata"})),
        pending_products=set(kwargs.pop("pending_products", {"roi"})),
        residual_utility_remaining=kwargs.pop("residual_utility_remaining", 0.30),
    )
    state.first_alert_delivered_time_s = kwargs.pop("first_alert_delivered_time_s", 0.0)
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def _system(**kwargs) -> dict:
    values = {
        "variant": "proposed_lyapunov_rule_mode_safe_bandit",
        "router_name": "proposed_lyapunov_rule_mode_safe_bandit",
        "use_lyapunov_pricing": True,
        "use_mode_controller": True,
        "queue_mb": 0.0,
        "queue_max_mb": 64.0,
        "queue_pressure": 0.0,
        "queued_products": set(),
        "remaining_deadline_s": 180.0,
        "deadline_s": 240.0,
        "battery_soc": 0.85,
        "thermal_pressure": 0.25,
        "task_value": 1.0,
        "high_value_flag": True,
        "emergency_flag": False,
        "uncertainty": 0.4,
        "model_quality_factor": 0.95,
        "contact_available": True,
        "contact_remaining_window_s": 60.0,
        "payload_size_mb": 0.8,
        "z_k_guided_followup_permission": True,
    }
    values.update(kwargs)
    return values


def _task() -> TaskSample:
    return TaskSample(
        "task_phase6",
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


def _timeline_step(task: TaskSample, slot_index: int = 1) -> TimelineStep:
    return TimelineStep(
        slot_index=slot_index,
        slot_seconds=60.0,
        task=task,
        resource_trace=ResourceTrace(slot_index=slot_index, solar_energy_j=0.0, base_power_w=0.0, temperature_reference_c=0.0),
        link_trace=LinkTrace(slot_index, True, 20.0, 8.0, 1.0, 60.0),
    )


def _payload(task_id: str, product_type: str, utility: float, *, is_residual: bool = False, trigger: str = ""):
    return GeneratedPayload(
        task_id=task_id,
        payload_policy=product_type,
        payload_mb=1.0,
        information_content=0.5,
        priority_score=0.5,
        value_score=1.0,
        task_value=1.0,
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
        continue_request_priority_bonus=0.05,
        followup_trigger=trigger,
        metadata={"followup_trigger": trigger} if trigger else {},
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


def test_eriv_uses_only_allowed_fields_and_no_forbidden_sarfish_future_fields():
    eriv = compute_expected_residual_information_value(_state(), _system(), _cfg())
    forbidden = {
        "sarfish_label",
        "bbox_count",
        "target_count",
        "object_class",
        "detector_ground_truth",
        "future_delivered_outcome",
        "future_utility_realization",
        "future_passive_virtual_continue_request",
    }
    assert eriv.leakage_safe_flag
    assert forbidden.isdisjoint(set(eriv.fields_used))


def test_eriv_increases_with_residual_utility_and_decreases_with_payload_size():
    low = compute_expected_residual_information_value(_state(residual_utility_remaining=0.10), _system(residual_delta_u_I=0.10), _cfg())
    high = compute_expected_residual_information_value(_state(residual_utility_remaining=0.45), _system(residual_delta_u_I=0.45), _cfg())
    large_payload = compute_expected_residual_information_value(
        _state(residual_utility_remaining=0.45),
        _system(residual_delta_u_I=0.45, payload_size_mb=3.0),
        _cfg(),
    )
    assert high.eriv_score > low.eriv_score
    assert large_payload.eriv_score < high.eriv_score


def test_proactive_false_cases_cover_disabled_util_duplicates_safety_deadline_contact_and_passive():
    assert not evaluate_proactive_followup_eligibility(_state(), _system(), _cfg(enabled=False)).proactive_flag
    assert "variant_not_full_proposed" in evaluate_proactive_followup_eligibility(
        _state(), _system(variant="utility_rule", router_name="utility_rule", use_lyapunov_pricing=False, use_mode_controller=False), _cfg()
    ).blockers
    assert "metadata_not_delivered" in evaluate_proactive_followup_eligibility(
        _state(delivered_products=set()), _system(), _cfg()
    ).blockers
    assert "no_pending_residual_product" in evaluate_proactive_followup_eligibility(
        _state(pending_products=set()), _system(), _cfg()
    ).blockers
    assert "duplicate_residual_already_queued" in evaluate_proactive_followup_eligibility(
        _state(), _system(queued_products={"task_phase6:roi"}), _cfg()
    ).blockers
    assert "passive_request_already_enqueued_residual" in evaluate_proactive_followup_eligibility(
        _state(virtual_continue_request_flag=True), _system(), _cfg()
    ).blockers
    assert "battery_unsafe" in evaluate_proactive_followup_eligibility(_state(), _system(battery_soc=0.50), _cfg()).blockers
    assert "thermal_unsafe" in evaluate_proactive_followup_eligibility(_state(), _system(thermal_pressure=0.90), _cfg()).blockers
    assert "queue_pressure_too_high" in evaluate_proactive_followup_eligibility(
        _state(), _system(queue_mb=50.0, queue_max_mb=64.0, queue_pressure=0.78), _cfg()
    ).blockers
    assert "no_useful_remaining_deadline" in evaluate_proactive_followup_eligibility(
        _state(), _system(remaining_deadline_s=30.0), _cfg()
    ).blockers
    assert "no_contact_opportunity" in evaluate_proactive_followup_eligibility(
        _state(), _system(contact_available=False, contact_remaining_window_s=0.0), _cfg()
    ).blockers
    assert "eriv_below_threshold" in evaluate_proactive_followup_eligibility(
        _state(residual_utility_remaining=0.09), _system(payload_size_mb=5.0), _cfg()
    ).blockers


def test_proactive_true_for_full_proposed_high_value_metadata_residual_case():
    decision = evaluate_proactive_followup_eligibility(_state(), _system(), _cfg())
    assert decision.proactive_flag
    assert decision.residual_product_type == "roi"
    assert decision.reason_code == "eligible"
    assert decision.eriv_score >= decision.threshold


def test_proactive_enqueue_uses_queue_admission_bonus_and_reward_is_utility_only():
    queue = DownlinkQueue(queue_max_mb=4.0, slot_seconds=10.0)
    residual = _payload("task_phase6", "roi", 0.40, is_residual=True, trigger="proactive_followup")
    result = queue.process_slot(0, residual, LinkTrace(0, False, -3.0, 0.0, 0.0, 0.0))
    item = queue.items[0]
    components = queue._priority_components(item, 0.0)
    assert result.proactive_followup_enqueue_count == 1
    assert components["proactive_queue_bonus_component"] == 0.05
    assert item.incremental_utility == 0.40


def test_duplicate_proactive_request_is_suppressed_without_second_positive_utility():
    queue = DownlinkQueue(queue_max_mb=4.0, slot_seconds=10.0)
    no_contact = LinkTrace(0, False, -3.0, 0.0, 0.0, 0.0)
    first = queue.process_slot(0, _payload("task_dup_phase6", "roi", 0.2, is_residual=True, trigger="proactive_followup"), no_contact)
    duplicate = queue.process_slot(1, _payload("task_dup_phase6", "roi", 0.2, is_residual=True, trigger="proactive_followup"), no_contact)
    assert first.proactive_followup_enqueue_count == 1
    assert duplicate.proactive_duplicate_suppression_count == 1
    assert duplicate.proactive_followup_enqueue_count == 0


def test_step_runner_phase6_disabled_preserves_phase5_passive_trigger():
    task = _task()
    queue = DownlinkQueue(queue_max_mb=4.0, slot_seconds=60.0)
    runner = StepRunner(None, None, None, None, None, queue, None, {"high_value_threshold": 0.75}, _phase_cfg(False, True))
    state = runner._get_product_lifecycle(task)
    state.delivered_products.add("metadata")
    state.pending_products.add("roi")
    state.residual_utility_remaining = 0.30
    payload = runner._maybe_build_residual_payload(
        _timeline_step(task),
        ResourceState(0.85, 1.0, 0.2),
        CommState(0.0, 4.0),
        DetectionResult([], 0.9, False, [], 0.2, 1.0, 0.95, {}),
        _payload(task.task_id, "metadata", 0.35),
        state,
        _Router(),
    )
    assert payload.is_residual
    assert payload.followup_trigger == "passive_virtual_continue_request"
    assert runner.last_proactive_followup_decision is None


def test_step_runner_full_proposed_proactive_before_passive_then_later_passive_suppressed():
    task = _task()
    queue = DownlinkQueue(queue_max_mb=4.0, slot_seconds=60.0)
    runner = StepRunner(None, None, None, None, None, queue, None, {"high_value_threshold": 0.75}, _phase_cfg(True, True))
    runner.last_guidance_bundle = {"mode_name": "nominal", "lambda_after_z": {"lambda_Q": 0.1}}
    state = runner._get_product_lifecycle(task)
    state.delivered_products.add("metadata")
    state.pending_products.add("roi")
    state.residual_utility_remaining = 0.30
    detection = DetectionResult([], 0.9, False, [], 0.2, 1.0, 0.95, {})
    proactive_payload = runner._maybe_build_residual_payload(
        _timeline_step(task),
        ResourceState(0.85, 1.0, 0.2),
        CommState(0.0, 4.0),
        detection,
        _payload(task.task_id, "metadata", 0.35),
        state,
        _Router(),
    )
    assert proactive_payload.is_residual
    assert proactive_payload.followup_trigger == "proactive_followup"
    enqueue_result = queue.process_slot(1, proactive_payload, LinkTrace(1, False, -3.0, 0.0, 0.0, 0.0))
    runner._apply_product_lifecycle_updates(_timeline_step(task), proactive_payload, enqueue_result, _feedback())
    later_payload = runner._maybe_build_residual_payload(
        _timeline_step(task, slot_index=2),
        ResourceState(0.85, 1.0, 0.2),
        CommState(queue.queue_mb, 4.0),
        detection,
        _payload(task.task_id, "metadata", 0.35),
        state,
        _Router(),
    )
    assert not later_payload.is_residual
    assert runner.last_passive_request_suppressed_by_prior_proactive == 1


def test_step_runner_util_has_no_phase6_proactive_enqueue():
    task = _task()
    runner = StepRunner(None, None, None, None, None, DownlinkQueue(4.0, 60.0), None, {"high_value_threshold": 0.75}, _phase_cfg(True, True))
    runner.last_guidance_bundle = {"mode_name": "nominal"}
    state = runner._get_product_lifecycle(task)
    state.delivered_products.add("metadata")
    state.pending_products.add("roi")
    state.residual_utility_remaining = 0.30
    payload = runner._maybe_build_residual_payload(
        _timeline_step(task),
        ResourceState(0.85, 1.0, 0.2),
        CommState(0.0, 4.0),
        DetectionResult([], 0.9, False, [], 0.2, 1.0, 0.95, {}),
        _payload(task.task_id, "metadata", 0.35),
        state,
        _UtilRouter(),
    )
    assert payload.followup_trigger == "passive_virtual_continue_request"
    assert "variant_not_full_proposed" in runner.last_proactive_followup_decision.blockers


def _carryover_runner(
    *,
    phase6_enabled: bool = True,
    carryover_enabled: bool = True,
    same_slot_enabled: bool = False,
    queue_fallback_enabled: bool = False,
    queue: DownlinkQueue | None = None,
):
    runner = StepRunner(
        None,
        None,
        None,
        None,
        None,
        queue or DownlinkQueue(8.0, 60.0),
        None,
        {"high_value_threshold": 0.75},
        _phase_cfg(phase6_enabled, True, carryover_enabled, same_slot_enabled, queue_fallback_enabled),
    )
    runner.last_guidance_bundle = {"mode_name": "nominal", "lambda_after_z": {"lambda_Q": 0.1}}
    return runner


def _ready_carryover_state(runner: StepRunner, task: TaskSample | None = None) -> tuple[TaskSample, ProductLifecycleState]:
    task = task or _task()
    state = runner._get_product_lifecycle(task)
    state.delivered_products.add("metadata")
    state.pending_products.add("roi")
    state.residual_utility_remaining = 0.30
    state.first_alert_delivered_time_s = 60.0
    state.metadata_delivery_slot = 1
    state.pending_residual_registered_slot = 1
    state.residual_parent_payload_plan = "metadata_then_roi"
    return task, state


def _detection() -> DetectionResult:
    return DetectionResult([], 0.9, False, [], 0.2, 1.0, 0.95, {})


def _register_candidate(runner: StepRunner, task: TaskSample | None = None):
    task, state = _ready_carryover_state(runner, task)
    payload = _payload(task.task_id, "metadata", 0.35)
    payload.payload_plan = "metadata_then_roi"
    runner._add_residual_candidate_after_lifecycle(_timeline_step(task, 1), state, payload, _detection(), _Router())
    return task, state


def test_carryover_candidate_added_only_after_metadata_delivered_and_pending_residual_exists():
    runner = _carryover_runner()
    _register_candidate(runner)
    assert "task_phase6:roi" in runner.residual_candidate_pool
    assert runner.last_carryover_pool_log["candidate_added_count"] == 1


def test_same_slot_carryover_default_off_preserves_candidate_for_tplus1():
    queue = DownlinkQueue(queue_max_mb=8.0, slot_seconds=60.0)
    runner = _carryover_runner(queue=queue)
    task = _task()
    step = _timeline_step(task, 1)
    payload = _payload(task.task_id, "metadata", 0.35)
    payload.payload_plan = "metadata_then_roi"
    payload.residual_product_type = "roi"
    result = queue.process_slot(1, payload, step.link_trace)
    feedback = _feedback()
    merged = runner._apply_product_lifecycle_updates(
        step,
        payload,
        result,
        feedback,
        router=_Router(),
        detection_result=_detection(),
        resource_state=ResourceState(0.85, 1.0, 0.2),
        comm_state=CommState(0.0, 8.0),
    )
    assert merged.proactive_followup_enqueue_count == 0
    assert feedback.same_slot_candidate_detected_count == 0
    assert "task_phase6:roi" in runner.residual_candidate_pool


def test_same_slot_carryover_enabled_reuses_existing_queue_path_and_removes_candidate():
    queue = DownlinkQueue(queue_max_mb=8.0, slot_seconds=60.0)
    runner = _carryover_runner(queue=queue, same_slot_enabled=True)
    task = _task()
    step = _timeline_step(task, 1)
    payload = _payload(task.task_id, "metadata", 0.35)
    payload.payload_plan = "metadata_then_roi"
    payload.residual_product_type = "roi"
    result = queue.process_slot(1, payload, step.link_trace)
    feedback = _feedback()
    merged = runner._apply_product_lifecycle_updates(
        step,
        payload,
        result,
        feedback,
        router=_Router(),
        detection_result=_detection(),
        resource_state=ResourceState(0.85, 1.0, 0.2),
        comm_state=CommState(0.0, 8.0),
    )
    assert merged.proactive_followup_enqueue_count == 1
    assert feedback.same_slot_candidate_detected_count == 1
    assert feedback.same_slot_candidate_evaluated_count == 1
    assert feedback.same_slot_carryover_enqueue_count == 1
    assert feedback.same_slot_opportunity_saved_from_tplus1_count == 1
    assert "task_phase6:roi" not in runner.residual_candidate_pool
    assert runner.product_lifecycle[task.task_id].residual_enqueued_count == 1
    assert runner.last_same_slot_carryover_trace
    assert any(row["event_type"] == "same_slot_proactive_enqueued" for row in runner.last_same_slot_carryover_trace)


def test_same_slot_insufficient_capacity_queue_fallback_is_default_off_then_enqueues_when_enabled():
    def run_case(enabled: bool):
        queue = DownlinkQueue(queue_max_mb=8.0, slot_seconds=60.0)
        runner = _carryover_runner(queue=queue, same_slot_enabled=True, queue_fallback_enabled=enabled)
        task = _task()
        step = TimelineStep(
            slot_index=1,
            slot_seconds=60.0,
            task=task,
            resource_trace=ResourceTrace(slot_index=1, solar_energy_j=0.0, base_power_w=0.0, temperature_reference_c=0.0),
            link_trace=LinkTrace(1, True, 20.0, 0.02, 1.0, 60.0),
        )
        payload = _payload(task.task_id, "metadata", 0.35)
        payload.payload_mb = 0.1
        payload.payload_plan = "metadata_then_roi"
        payload.residual_product_type = "roi"
        result = queue.process_slot(1, payload, step.link_trace)
        assert result.current_task_delivery_status == "delivered"
        feedback = _feedback()
        merged = runner._apply_product_lifecycle_updates(
            step,
            payload,
            result,
            feedback,
            router=_Router(),
            detection_result=_detection(),
            resource_state=ResourceState(0.85, 1.0, 0.2),
            comm_state=CommState(0.0, 8.0),
        )
        return runner, feedback, merged

    runner_off, feedback_off, merged_off = run_case(False)
    assert merged_off.proactive_followup_enqueue_count == 0
    assert feedback_off.ordinary_queue_fallback_attempt_count == 0
    assert "task_phase6:roi" in runner_off.residual_candidate_pool
    assert any(
        row["event_type"] == "same_slot_fallback_to_tplus1"
        and row["reason_code"] == "residual_payload_exceeds_remaining_contact_capacity"
        for row in runner_off.last_same_slot_carryover_trace
    )

    runner_on, feedback_on, merged_on = run_case(True)
    assert merged_on.proactive_followup_enqueue_count == 1
    assert merged_on.proactive_followup_delivery_count == 0
    assert feedback_on.ordinary_queue_fallback_attempt_count == 1
    assert feedback_on.ordinary_queue_fallback_success_count == 1
    assert feedback_on.same_slot_fallback_to_ordinary_queue_count == 1
    assert "task_phase6:roi" not in runner_on.residual_candidate_pool
    assert any(
        row["event_type"] == "ordinary_queue_fallback_enqueued"
        and row["reason_code"] == "insufficient_same_slot_capacity_but_ordinary_queue_enabled"
        for row in runner_on.last_same_slot_carryover_trace
    )


def test_carryover_candidate_not_added_if_metadata_not_delivered():
    runner = _carryover_runner()
    task, state = _ready_carryover_state(runner)
    state.delivered_products.clear()
    runner._add_residual_candidate_after_lifecycle(_timeline_step(task, 1), state, _payload(task.task_id, "metadata", 0.35), _detection(), _Router())
    assert not runner.residual_candidate_pool


def test_carryover_candidate_not_added_if_no_pending_residual():
    runner = _carryover_runner()
    task, state = _ready_carryover_state(runner)
    state.pending_products.clear()
    runner._add_residual_candidate_after_lifecycle(_timeline_step(task, 1), state, _payload(task.task_id, "metadata", 0.35), _detection(), _Router())
    assert not runner.residual_candidate_pool


def test_carryover_candidate_not_added_if_residual_utility_below_threshold():
    runner = _carryover_runner()
    task, state = _ready_carryover_state(runner)
    state.residual_utility_remaining = 0.0
    runner._add_residual_candidate_after_lifecycle(_timeline_step(task, 1), state, _payload(task.task_id, "metadata", 0.35), _detection(), _Router())
    assert not runner.residual_candidate_pool


def test_carryover_candidate_not_added_if_task_expired_or_closed():
    runner = _carryover_runner()
    task, state = _ready_carryover_state(runner)
    state.task_closed_flag = True
    runner._add_residual_candidate_after_lifecycle(_timeline_step(task, 1), state, _payload(task.task_id, "metadata", 0.35), _detection(), _Router())
    assert not runner.residual_candidate_pool

    runner = _carryover_runner()
    task, state = _ready_carryover_state(runner)
    runner._add_residual_candidate_after_lifecycle(_timeline_step(task, 6), state, _payload(task.task_id, "metadata", 0.35), _detection(), _Router())
    assert not runner.residual_candidate_pool


def test_carryover_duplicate_candidate_not_added_twice():
    runner = _carryover_runner()
    task, _state_obj = _register_candidate(runner)
    runner._add_residual_candidate_after_lifecycle(_timeline_step(task, 1), runner.product_lifecycle[task.task_id], _payload(task.task_id, "metadata", 0.35), _detection(), _Router())
    assert len(runner.residual_candidate_pool) == 1
    assert runner.last_carryover_pool_log["candidate_duplicate_suppressed_count"] == 1


def test_carryover_candidate_re_evaluated_in_later_idle_slot_and_uses_existing_phase6_logic():
    runner = _carryover_runner()
    _register_candidate(runner)
    payload, feedback_task = runner._maybe_build_carryover_payload(
        _timeline_step(_task(), 2).__class__(
            slot_index=2,
            slot_seconds=60.0,
            task=None,
            resource_trace=ResourceTrace(slot_index=2, solar_energy_j=0.0, base_power_w=0.0, temperature_reference_c=0.0),
            link_trace=LinkTrace(2, True, 20.0, 8.0, 1.0, 60.0),
        ),
        ResourceState(0.85, 1.0, 0.2),
        CommState(0.0, 8.0),
        _Router(),
    )
    assert payload is not None
    assert feedback_task is not None
    assert payload.followup_trigger == "proactive_followup"
    assert payload.metadata["followup_source"] == "phase6_residual_candidate_carryover"
    assert runner.last_carryover_pool_log["candidate_re_evaluated_count"] == 1
    assert runner.last_proactive_followup_decision.proactive_flag


def test_carryover_candidate_removed_after_residual_delivery_or_expiration():
    runner = _carryover_runner()
    task, state = _register_candidate(runner)
    state.delivered_products.add("roi")
    runner._maybe_build_carryover_payload(
        TimelineStep(
            slot_index=2,
            slot_seconds=60.0,
            task=None,
            resource_trace=ResourceTrace(slot_index=2, solar_energy_j=0.0, base_power_w=0.0, temperature_reference_c=0.0),
            link_trace=LinkTrace(2, True, 20.0, 8.0, 1.0, 60.0),
        ),
        ResourceState(0.85, 1.0, 0.2),
        CommState(0.0, 8.0),
        _Router(),
    )
    assert not runner.residual_candidate_pool
    assert runner.last_carryover_pool_log["candidate_removed_count"] == 1

    runner = _carryover_runner()
    _register_candidate(runner)
    runner._maybe_build_carryover_payload(
        TimelineStep(
            slot_index=20,
            slot_seconds=60.0,
            task=None,
            resource_trace=ResourceTrace(slot_index=20, solar_energy_j=0.0, base_power_w=0.0, temperature_reference_c=0.0),
            link_trace=LinkTrace(20, True, 20.0, 8.0, 1.0, 60.0),
        ),
        ResourceState(0.85, 1.0, 0.2),
        CommState(0.0, 8.0),
        _Router(),
    )
    assert not runner.residual_candidate_pool
    assert runner.last_carryover_pool_log["candidate_expired_count"] == 1


def test_carryover_disabled_phase6_disabled_and_util_do_not_produce_carryover_payload():
    for runner, router in [
        (_carryover_runner(phase6_enabled=False, carryover_enabled=True), _Router()),
        (_carryover_runner(phase6_enabled=True, carryover_enabled=False), _Router()),
        (_carryover_runner(phase6_enabled=True, carryover_enabled=True), _UtilRouter()),
    ]:
        _register_candidate(runner)
        payload, _feedback_task = runner._maybe_build_carryover_payload(
            TimelineStep(
                slot_index=2,
                slot_seconds=60.0,
                task=None,
                resource_trace=ResourceTrace(slot_index=2, solar_energy_j=0.0, base_power_w=0.0, temperature_reference_c=0.0),
                link_trace=LinkTrace(2, True, 20.0, 8.0, 1.0, 60.0),
            ),
            ResourceState(0.85, 1.0, 0.2),
            CommState(0.0, 8.0),
            router,
        )
        assert payload is None


def test_carryover_queue_bonus_not_delivered_utility_and_no_forbidden_fields_used():
    runner = _carryover_runner()
    _register_candidate(runner)
    payload, _feedback_task = runner._maybe_build_carryover_payload(
        TimelineStep(
            slot_index=2,
            slot_seconds=60.0,
            task=None,
            resource_trace=ResourceTrace(slot_index=2, solar_energy_j=0.0, base_power_w=0.0, temperature_reference_c=0.0),
            link_trace=LinkTrace(2, True, 20.0, 8.0, 1.0, 60.0),
        ),
        ResourceState(0.85, 1.0, 0.2),
        CommState(0.0, 8.0),
        _Router(),
    )
    assert payload.continue_request_priority_bonus == 0.05
    assert payload.incremental_utility_estimate == payload.task_value * payload.incremental_information_utility
    forbidden = {"bbox_count", "target_count", "object_class", "future_delivered_outcome", "future_utility_realization"}
    assert forbidden.isdisjoint(set(runner.last_proactive_followup_decision.fields_used))


def test_carryover_duplicate_residual_enqueue_suppressed_and_passive_duplicate_has_no_positive_utility():
    queue = DownlinkQueue(queue_max_mb=8.0, slot_seconds=60.0)
    runner = _carryover_runner(queue=queue)
    task, state = _register_candidate(runner)
    idle_step = TimelineStep(
        slot_index=2,
        slot_seconds=60.0,
        task=None,
        resource_trace=ResourceTrace(slot_index=2, solar_energy_j=0.0, base_power_w=0.0, temperature_reference_c=0.0),
        link_trace=LinkTrace(2, True, 20.0, 0.0, 1.0, 60.0),
    )
    payload, feedback_task = runner._maybe_build_carryover_payload(idle_step, ResourceState(0.85, 1.0, 0.2), CommState(0.0, 8.0), _Router())
    result = queue.process_slot(2, payload, idle_step.link_trace)
    feedback = _feedback()
    runner._apply_product_lifecycle_updates(idle_step, payload, result, feedback, feedback_task=feedback_task, router=_Router(), detection_result=None)
    assert result.proactive_followup_enqueue_count == 1

    duplicate = queue.process_slot(3, payload, idle_step.link_trace)
    assert duplicate.proactive_duplicate_suppression_count == 1
    assert duplicate.proactive_followup_enqueue_count == 0

    later_payload = runner._maybe_build_residual_payload(
        _timeline_step(task, slot_index=3),
        ResourceState(0.85, 1.0, 0.2),
        CommState(queue.queue_mb, 8.0),
        _detection(),
        _payload(task.task_id, "metadata", 0.35),
        state,
        _Router(),
    )
    assert not later_payload.is_residual
    assert runner.last_passive_request_suppressed_by_prior_proactive == 1


def test_carryover_integration_metadata_then_next_slot_enqueue_delivery_counted_once():
    queue = DownlinkQueue(queue_max_mb=8.0, slot_seconds=60.0)
    runner = _carryover_runner(queue=queue)
    task, _state_obj = _register_candidate(runner)
    idle_step = TimelineStep(
        slot_index=2,
        slot_seconds=60.0,
        task=None,
        resource_trace=ResourceTrace(slot_index=2, solar_energy_j=0.0, base_power_w=0.0, temperature_reference_c=0.0),
        link_trace=LinkTrace(2, True, 20.0, 8.0, 1.0, 60.0),
    )
    payload, feedback_task = runner._maybe_build_carryover_payload(idle_step, ResourceState(0.85, 1.0, 0.2), CommState(0.0, 8.0), _Router())
    assert payload is not None
    result = queue.process_slot(2, payload, idle_step.link_trace)
    feedback = _feedback()
    runner._apply_product_lifecycle_updates(idle_step, payload, result, feedback, feedback_task=feedback_task, router=_Router(), detection_result=None)
    assert feedback.carryover_proactive_enqueue_count == 1
    assert feedback.carryover_proactive_delivery_count == 1
    assert feedback.carryover_proactive_incremental_utility > 0.0
    assert feedback.delivered_only_incremental_utility == feedback.carryover_proactive_incremental_utility
    assert runner.product_lifecycle[task.task_id].residual_delivered_count == 1
    assert runner.product_lifecycle[task.task_id].carryover_proactive_delivery_count == 1
