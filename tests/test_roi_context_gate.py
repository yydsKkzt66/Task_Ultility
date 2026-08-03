from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.common.types import (
    CommState,
    DetectionResult,
    DownlinkResult,
    JointAction,
    LinkTrace,
    PredictedMetrics,
    ProductLifecycleState,
    ResidualCandidateState,
    ResourceState,
    ResourceTrace,
    TaskSample,
    TimelineStep,
)
from src.data.value_skew_calibration import task_effective_value
from src.link.downlink_queue import DownlinkQueue
from src.payload.information_sufficiency import residual_information_utility
from src.payload.roi_context_gate import evaluate_roi_context_gate, roi_context_gate_config
from src.simulator.step_runner import PayloadExecutionFeasibilityDecision, StepRunner


def _cfg(*, allow_queued: bool = True, top_k_per_contact_window: int = 3) -> dict[str, object]:
    return {
        "epsilon": 1.0e-9,
        "battery_min": 0.18,
        "thermal_max": 0.92,
        "queue_max_mb": 10.0,
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
        "progressive_payload": {
            "phase6_proactive_followup": {
                "enabled": True,
                "eriv_threshold": 0.0,
                "residual_utility_threshold": 0.0,
                "queue_pressure_threshold": 1.0,
                "min_remaining_deadline_s": 0.0,
                "battery_guard_threshold": 0.0,
                "thermal_guard_threshold": 1.0,
                "require_contact_opportunity": True,
                "full_method_aliases": ["proposed_lyapunov_rule_mode_safe_bandit"],
            },
            "phase6_residual_candidate_carryover": {"enabled": True, "max_candidates_per_slot": 2},
            "phase6_same_slot_carryover": {
                "enabled": True,
                "max_same_slot_eval_per_slot": 2,
                "enable_queue_fallback_on_insufficient_capacity": False,
            },
            "roi_context_gate": {
                "enabled": True,
                "profile": "soft_moderate",
                "profiles": {
                    "soft_moderate": {
                        "eriv_threshold": 0.0,
                        "min_value_density": 0.0,
                        "eligible_product_requirements": ["context", "evidence"],
                        "eligible_value_levels": ["high", "urgent"],
                        "queue_pressure_threshold": 1.0,
                        "battery_guard_threshold": 0.0,
                        "thermal_guard_threshold": 1.0,
                        "min_remaining_deadline_s": 0.0,
                        "require_contact_opportunity": True,
                        "require_current_contact_for_same_slot": True,
                        "allow_queued_followup_without_current_contact": allow_queued,
                        "require_future_contact_before_deadline_for_queue": True,
                        "top_k_per_slot": 2,
                        "top_k_per_contact_window": top_k_per_contact_window,
                        "queued_followup_budget_per_slot": 1,
                    }
                },
            },
            "roi_context_after_roi": {"enabled": True, "residual_utility_threshold": 0.0},
        },
    }


def _runner(cfg: dict[str, object] | None = None, future: list[LinkTrace] | None = None) -> StepRunner:
    runner = StepRunner.__new__(StepRunner)
    runner.simulator_cfg = cfg or _cfg()
    runner.reward_cfg = {"high_value_threshold": 0.75}
    runner.payload_profiles_cfg = None
    runner.downlink_queue = DownlinkQueue(10.0, 20.0)
    runner.product_lifecycle = {}
    runner.residual_candidate_pool = {}
    runner.residual_candidate_history = {}
    runner.last_carryover_pool_log = StepRunner._empty_carryover_pool_log()
    runner.last_guidance_bundle = {"enabled": True}
    runner.future_link_traces = future or []
    runner.last_carryover_lifecycle_events = []
    runner.last_same_slot_carryover_trace = []
    runner.last_followup_hard_safety_guard_trace = []
    runner.last_roi_context_window_budget_trace = []
    runner.last_roi_context_budget_trace = runner.last_roi_context_window_budget_trace
    runner.last_value_density_trace = []
    runner.last_same_slot_vs_queued_followup_trace = []
    runner.last_future_contact_feasibility_trace = []
    runner.last_duplicate_suppression_trace = []
    runner._roi_context_budget_slot_index = -1
    runner._roi_context_slot_selected_count = 0
    runner._roi_context_queued_budget_slot_index = -1
    runner._roi_context_queued_selected_count = 0
    runner._roi_context_contact_window_id = ""
    runner._roi_context_contact_window_selected_count = 0
    runner._roi_context_window_selected_count = 0
    return runner


def _task(task_id: str = "task_a", *, value_level: str = "high", requirement: str = "context") -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_path="mock://image",
        label_path="mock://label",
        scene_id=f"scene_{task_id}",
        virtual_time=0,
        task_value=0.9,
        deadline_s=120.0,
        difficulty=0.1,
        uncertainty=0.2,
        high_value_flag=value_level in {"high", "urgent"},
        emergency_flag=value_level == "urgent",
        product_requirement=requirement,
        value_level=value_level,
        calibrated_task_value=1.8,
    )


def _timeline(task: TaskSample | None, slot: int = 1, *, in_contact: bool = True) -> TimelineStep:
    return TimelineStep(
        slot_index=slot,
        slot_seconds=20.0,
        task=task,
        resource_trace=ResourceTrace(slot, 0.0, 0.0, 10.0, is_sunlit=False),
        link_trace=LinkTrace(slot, in_contact, 10.0, 8.0, 1.0, 20.0),
    )


def _detection(quality: float = 0.8) -> DetectionResult:
    return DetectionResult([], 0.9, True, [], 0.1, 0.1, quality, {})


def _predicted(info: float = 0.37, quality: float = 0.8, delivery: float = 0.5, size: float = 2.0) -> PredictedMetrics:
    return PredictedMetrics(
        predicted_S_gen_mb=size,
        predicted_S_down_mb=size,
        predicted_E_comp=0.0,
        predicted_E_comm=0.0,
        predicted_L_inf=0.0,
        predicted_L_queue=0.0,
        predicted_L_comm=0.0,
        predicted_Q_next_mb=0.0,
        predicted_B_next=0.9,
        predicted_E_AI_next=1.0,
        predicted_Theta_next=0.1,
        predicted_delivery_factor=delivery,
        predicted_safety_violation=0.0,
        predicted_reward=info,
        predicted_status="ok",
        predicted_detection_quality=quality,
        predicted_incremental_information_utility=info,
    )


def _state(task_id: str = "task_a") -> ProductLifecycleState:
    return ProductLifecycleState(
        task_id=task_id,
        product_requirement="context",
        delivered_products={"metadata", "roi"},
        pending_products={"roi_context"},
        residual_utility_remaining=0.37,
        metadata_delivery_slot=0,
        first_alert_delivered_time_s=0.0,
    )


def _candidate(task: TaskSample) -> ResidualCandidateState:
    return ResidualCandidateState(
        task_id=task.task_id,
        scene_id=task.scene_id,
        product_requirement=task.product_requirement,
        task_value=task_effective_value(task),
        high_value_flag=True,
        emergency_flag=False,
        payload_plan="metadata_then_roi_then_roi_context",
        residual_product_type="roi_context",
        metadata_delivery_slot=0,
        metadata_delivery_time_s=0.0,
        pending_residual_registered_slot=0,
        residual_utility_remaining=0.37,
        residual_delta_u_I=0.37,
        deadline_s=120.0,
        created_slot=0,
        delivered_products_snapshot={"metadata", "roi"},
        pending_products_snapshot={"roi_context"},
        duplicate_identity_key=f"{task.task_id}:roi_context",
        task_snapshot=task,
        detection_result_snapshot=_detection(),
        value_level="high",
    )


def test_value_density_consistent_between_action_and_second_stage():
    task = _task()
    runner = _runner(future=[LinkTrace(i, True, 10.0, 8.0, 1.0, 20.0) for i in range(5)])
    action = JointAction("ctx", "model", "roi_context", "immediate", 1.0, 1.0)
    runner._apply_roi_context_gate_to_actions(
        timeline_step=_timeline(task, 1, in_contact=True),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        candidate_actions=[action],
        predicted_map={"ctx": _predicted()},
    )
    action_trace = runner.last_value_density_trace[-1]
    expected_action_density = task_effective_value(task) * 0.37 * 0.8 * 0.5 / 2.0
    assert float(action_trace["value_density"]) == pytest.approx(expected_action_density)

    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"})
    runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline(task, 1, in_contact=True),
        lifecycle_state=state,
        task=task,
        detection_result=_detection(0.8),
        router=None,
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
    )
    second_trace = runner.last_value_density_trace[-1]
    payload_size = float(second_trace["payload_size_mb"])
    timeliness = float(second_trace["timeliness_factor"])
    residual = residual_information_utility({"roi"}, "context")
    assert float(second_trace["value_density"]) == pytest.approx(task_effective_value(task) * residual * 0.8 * timeliness / payload_size)


def test_information_density_and_value_density_are_distinct():
    task = _task()
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"})
    runner = _runner(future=[LinkTrace(i, True, 10.0, 8.0, 1.0, 20.0) for i in range(5)])
    assert runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline(task, 1, in_contact=True),
        lifecycle_state=state,
        task=task,
        detection_result=_detection(0.7),
        router=None,
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
    )
    trace = runner.last_value_density_trace[-1]
    assert float(trace["information_density"]) > 0.0
    assert float(trace["value_density"]) != pytest.approx(float(trace["information_density"]))


def test_no_contact_does_not_same_slot_deliver_roi_context(monkeypatch):
    task = _task()
    runner = _runner()
    candidate = _candidate(task)
    candidate.created_slot = 1
    runner.residual_candidate_pool[candidate.duplicate_identity_key] = candidate
    runner.product_lifecycle[task.task_id] = _state(task.task_id)
    monkeypatch.setattr(runner.downlink_queue, "process_slot", lambda *args, **kwargs: pytest.fail("must not deliver without contact"))
    result = runner._maybe_process_same_slot_carryover(
        timeline_step=_timeline(task, 1, in_contact=False),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name="proposed_lyapunov_rule_mode_safe_bandit", use_lyapunov_pricing=True, use_mode_controller=True),
        downlink_result=DownlinkResult(0, 0, 0, 0, 0, 0, 0, 0, "empty"),
        candidate_key=candidate.duplicate_identity_key,
    )
    assert result is None
    assert runner.last_same_slot_vs_queued_followup_trace[-1]["reason_code"] == "no_contact_opportunity"


def test_no_contact_can_enqueue_roi_context_if_future_contact_before_deadline():
    task = _task()
    future = [LinkTrace(0, False, 10.0, 0.0, 1.0, 20.0), LinkTrace(1, False, 10.0, 0.0, 1.0, 20.0), LinkTrace(2, True, 10.0, 80.0, 1.0, 20.0)]
    runner = _runner(future=future)
    candidate = _candidate(task)
    runner.residual_candidate_pool[candidate.duplicate_identity_key] = candidate
    runner.product_lifecycle[task.task_id] = _state(task.task_id)
    payload, feedback_task = runner._maybe_build_carryover_payload(
        timeline_step=_timeline(None, 1, in_contact=False),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name="proposed_lyapunov_rule_mode_safe_bandit", use_lyapunov_pricing=True, use_mode_controller=True),
    )
    assert payload is not None
    assert feedback_task == task
    assert runner.last_future_contact_feasibility_trace[-1]["future_contact_feasible"] == 1


def test_no_future_contact_blocks_queued_roi_context():
    task = _task()
    future = [LinkTrace(i, False, 10.0, 0.0, 1.0, 20.0) for i in range(5)]
    runner = _runner(future=future)
    candidate = _candidate(task)
    runner.residual_candidate_pool[candidate.duplicate_identity_key] = candidate
    runner.product_lifecycle[task.task_id] = _state(task.task_id)
    payload, _feedback_task = runner._maybe_build_carryover_payload(
        timeline_step=_timeline(None, 1, in_contact=False),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name="proposed_lyapunov_rule_mode_safe_bandit", use_lyapunov_pricing=True, use_mode_controller=True),
    )
    assert payload is None
    assert "no_contact_opportunity" in runner.last_carryover_lifecycle_events[-1]["proactive_blockers"]


def test_queued_roi_context_checks_queue_hard_safety():
    task = _task()
    future = [LinkTrace(i, True, 10.0, 80.0, 1.0, 20.0) for i in range(5)]
    runner = _runner(future=future)
    candidate = _candidate(task)
    runner.residual_candidate_pool[candidate.duplicate_identity_key] = candidate
    runner.product_lifecycle[task.task_id] = _state(task.task_id)
    payload, _feedback_task = runner._maybe_build_carryover_payload(
        timeline_step=_timeline(None, 1, in_contact=False),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(9.9, 10.0),
        router=SimpleNamespace(router_name="proposed_lyapunov_rule_mode_safe_bandit", use_lyapunov_pricing=True, use_mode_controller=True),
    )
    assert payload is None
    assert "followup_hard_safety_blocked_queue" in runner.last_followup_hard_safety_guard_trace[-1]["blockers"]


def test_same_slot_roi_context_checks_capacity_and_hard_safety(monkeypatch):
    task = _task()
    runner = _runner()
    candidate = _candidate(task)
    runner.residual_candidate_pool[candidate.duplicate_identity_key] = candidate
    runner.product_lifecycle[task.task_id] = _state(task.task_id)
    monkeypatch.setattr(
        runner,
        "_evaluate_and_build_carryover_payload",
        lambda *args, **kwargs: (SimpleNamespace(payload_mb=99.0, product_type="roi_context"), task, SimpleNamespace(proactive_flag=True, reason_code="eligible", blockers=())),
    )
    result = runner._maybe_process_same_slot_carryover(
        timeline_step=_timeline(task, 1, in_contact=True),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name="proposed_lyapunov_rule_mode_safe_bandit", use_lyapunov_pricing=True, use_mode_controller=True),
        downlink_result=DownlinkResult(0, 0, 0, 0, 0, 0, 0, 0, "empty"),
        candidate_key=candidate.duplicate_identity_key,
    )
    assert result is None


def test_contact_window_budget_accumulates_selected_count():
    task = _task()
    future = [LinkTrace(0, True, 10.0, 80.0, 1.0, 20.0), LinkTrace(1, True, 10.0, 80.0, 1.0, 20.0)]
    runner = _runner(future=future)
    action = JointAction("ctx", "model", "roi_context", "immediate", 1.0, 1.0)
    for slot in [0, 1]:
        runner._apply_roi_context_gate_to_actions(
            timeline_step=_timeline(task, slot, in_contact=True),
            resource_state=ResourceState(0.9, 1.0, 0.1),
            comm_state=CommState(0.0, 10.0),
            candidate_actions=[action],
            predicted_map={"ctx": _predicted()},
        )
    assert runner._roi_context_contact_window_selected_count == 2


def test_budget_resets_on_new_contact_window():
    task = _task()
    future = [
        LinkTrace(0, True, 10.0, 80.0, 1.0, 20.0),
        LinkTrace(1, False, 10.0, 0.0, 1.0, 20.0),
        LinkTrace(2, True, 10.0, 80.0, 1.0, 20.0),
    ]
    runner = _runner(future=future)
    action = JointAction("ctx", "model", "roi_context", "immediate", 1.0, 1.0)
    for slot, contact in [(0, True), (2, True)]:
        runner._apply_roi_context_gate_to_actions(
            timeline_step=_timeline(task, slot, in_contact=contact),
            resource_state=ResourceState(0.9, 1.0, 0.1),
            comm_state=CommState(0.0, 10.0),
            candidate_actions=[action],
            predicted_map={"ctx": _predicted()},
        )
    assert runner._roi_context_contact_window_id == "contact_window_2"
    assert runner._roi_context_contact_window_selected_count == 1


def test_low_medium_value_still_blocked_from_roi_context():
    decision = evaluate_roi_context_gate(
        product_requirement="context",
        value_level="medium",
        value_density=10.0,
        system_state={"contact_available": True, "remaining_deadline_s": 100.0},
        config=_cfg(),
        eriv_score=10.0,
    )
    assert not decision.allowed
    assert "value_level_not_high_or_urgent" in decision.blockers


def test_high_urgent_context_evidence_can_pass_with_safe_resources():
    decision = evaluate_roi_context_gate(
        product_requirement="evidence",
        value_level="urgent",
        value_density=1.0,
        system_state={"contact_available": True, "remaining_deadline_s": 100.0, "battery_soc": 0.9, "thermal_pressure": 0.1},
        config=_cfg(),
        eriv_score=1.0,
    )
    assert decision.allowed


def test_roi_context_after_roi_success_path():
    task = _task()
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"})
    runner = _runner(future=[LinkTrace(i, True, 10.0, 80.0, 1.0, 20.0) for i in range(5)])
    registered = runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline(task, 1, in_contact=True),
        lifecycle_state=state,
        task=task,
        detection_result=_detection(),
        router=None,
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
    )
    assert registered
    assert "roi_context" in state.pending_products


def test_legacy_default_unchanged():
    cfg = roi_context_gate_config({})
    assert cfg["enabled"] is False
    decision = evaluate_roi_context_gate(
        product_requirement="verification",
        value_level="low",
        value_density=0.0,
        system_state={},
        config={},
        eriv_score=0.0,
    )
    assert decision.allowed
    assert decision.reason_code == "gate_disabled"
