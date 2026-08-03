from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.common.types import (
    CommState,
    DetectionResult,
    DownlinkResult,
    GeneratedPayload,
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
from src.link.downlink_queue import DownlinkQueue
from src.simulator.step_runner import PayloadExecutionFeasibilityDecision, StepRunner


def _cfg(**overrides) -> dict[str, object]:
    cfg: dict[str, object] = {
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
                "min_remaining_deadline_s": 5.0,
                "full_method_aliases": ["proposed_lyapunov_rule_mode_safe_bandit"],
            },
            "phase6_residual_candidate_carryover": {"enabled": True},
            "phase6_same_slot_carryover": {
                "enabled": True,
                "only_after_lifecycle_update": True,
                "reuse_existing_guards": True,
                "enforce_no_reward_change": True,
                "enforce_no_queue_policy_change": True,
                "strict_mode": True,
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
                        "require_contact_opportunity": False,
                        "top_k_per_window": 2,
                        "window_budget": 2,
                    }
                },
            },
            "roi_context_after_roi": {"enabled": True, "residual_utility_threshold": 0.0},
        },
    }
    cfg.update(overrides)
    return cfg


def _runner(cfg: dict[str, object] | None = None) -> StepRunner:
    runner = StepRunner.__new__(StepRunner)
    runner.simulator_cfg = cfg or _cfg()
    runner.reward_cfg = {"high_value_threshold": 0.75}
    runner.payload_profiles_cfg = None
    runner.downlink_queue = DownlinkQueue(10.0, 20.0)
    runner.product_lifecycle = {}
    runner.residual_candidate_pool = {}
    runner.residual_candidate_history = {}
    runner.last_carryover_pool_log = StepRunner._empty_carryover_pool_log()
    runner.last_same_slot_carryover_trace = []
    runner.last_carryover_lifecycle_events = []
    runner.last_followup_hard_safety_guard_trace = []
    runner.last_roi_context_window_budget_trace = []
    runner.last_duplicate_suppression_trace = []
    runner._roi_context_window_selected_count = 0
    runner.last_guidance_bundle = {"enabled": True}
    return runner


def _task(task_id: str = "task_a", requirement: str = "context") -> TaskSample:
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
        high_value_flag=True,
        product_requirement=requirement,
        value_level="high",
        calibrated_task_value=0.9,
    )


def _timeline(task: TaskSample | None = None, *, in_contact: bool = True) -> TimelineStep:
    return TimelineStep(
        slot_index=1,
        slot_seconds=20.0,
        task=task,
        resource_trace=ResourceTrace(1, 0.0, 0.0, 10.0, is_sunlit=False),
        link_trace=LinkTrace(1, in_contact, 10.0, 8.0, 1.0, 20.0),
    )


def _payload(task_id: str = "task_a", product_type: str = "roi_context", size_mb: float = 1.0) -> GeneratedPayload:
    return GeneratedPayload(
        task_id=task_id,
        payload_policy=product_type,
        payload_mb=size_mb,
        information_content=0.9,
        priority_score=1.0,
        value_score=0.9,
        task_value=0.9,
        uncertainty=0.2,
        remaining_utility=0.3,
        is_metadata=False,
        is_high_value=True,
        deadline_s=100.0,
        created_slot=1,
        product_type=product_type,
        product_requirement="context",
        incremental_utility_estimate=0.3,
        incremental_information_utility=0.3,
        is_residual=True,
        duplicate_identity_key=f"{task_id}:{product_type}",
        followup_trigger="proactive_followup",
    )


def _state(task_id: str = "task_a") -> ProductLifecycleState:
    return ProductLifecycleState(
        task_id=task_id,
        product_requirement="context",
        delivered_products={"roi"},
        pending_products={"roi_context"},
        residual_utility_remaining=0.4,
    )


def test_followup_guard_blocks_battery_violation():
    runner = _runner()
    decision = runner._followup_hard_safety_guard(
        timeline_step=_timeline(_task()),
        resource_state=ResourceState(0.181, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        payload=_payload(size_mb=2.0),
        lifecycle_state=_state(),
        task=_task(),
        operation="test",
        require_same_slot_delivery=True,
        remaining_contact_capacity_mb=2.0,
    )
    assert not decision.allowed
    assert "followup_hard_safety_blocked_battery" in decision.blockers


def test_followup_guard_blocks_thermal_violation():
    runner = _runner(_cfg(tx_power_w=200.0, comm_power_w=200.0))
    decision = runner._followup_hard_safety_guard(
        timeline_step=_timeline(_task()),
        resource_state=ResourceState(0.9, 1.0, 0.91),
        comm_state=CommState(0.0, 10.0),
        payload=_payload(size_mb=1.0),
        lifecycle_state=_state(),
        task=_task(),
        operation="test",
        require_same_slot_delivery=True,
        remaining_contact_capacity_mb=1.0,
    )
    assert not decision.allowed
    assert "followup_hard_safety_blocked_thermal" in decision.blockers


def test_followup_guard_blocks_queue_overflow():
    runner = _runner()
    decision = runner._followup_hard_safety_guard(
        timeline_step=_timeline(_task()),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(9.5, 10.0),
        payload=_payload(size_mb=1.0),
        lifecycle_state=_state(),
        task=_task(),
        operation="test",
        require_same_slot_delivery=False,
        predict_immediate_downlink=False,
    )
    assert not decision.allowed
    assert "followup_hard_safety_blocked_queue" in decision.blockers


def test_followup_guard_blocks_downlink_throttled_same_slot():
    runner = _runner()
    decision = runner._followup_hard_safety_guard(
        timeline_step=_timeline(_task()),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        payload=_payload(size_mb=0.5),
        lifecycle_state=_state(),
        task=_task(),
        operation="test",
        require_same_slot_delivery=True,
        remaining_contact_capacity_mb=1.0,
        downlink_throttled=True,
    )
    assert not decision.allowed
    assert "followup_hard_safety_blocked_downlink_throttled" in decision.blockers


def test_followup_guard_allows_safe_roi_context():
    runner = _runner()
    decision = runner._followup_hard_safety_guard(
        timeline_step=_timeline(_task()),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        payload=_payload(size_mb=0.5),
        lifecycle_state=_state(),
        task=_task(),
        operation="test",
        require_same_slot_delivery=True,
        remaining_contact_capacity_mb=1.0,
    )
    assert decision.allowed
    assert decision.reason_code == "followup_hard_safety_allowed"


def test_same_slot_carryover_calls_hard_safety_guard_before_process_slot(monkeypatch):
    runner = _runner()
    task = _task()
    candidate = ResidualCandidateState(
        task_id=task.task_id,
        scene_id=task.scene_id,
        product_requirement="context",
        task_value=0.9,
        high_value_flag=True,
        emergency_flag=False,
        payload_plan="metadata_then_roi_context",
        residual_product_type="roi_context",
        metadata_delivery_slot=1,
        metadata_delivery_time_s=20.0,
        pending_residual_registered_slot=1,
        residual_utility_remaining=0.4,
        residual_delta_u_I=0.4,
        deadline_s=120.0,
        created_slot=1,
        duplicate_identity_key="task_a:roi_context",
        task_snapshot=task,
        detection_result_snapshot=DetectionResult([], 0.9, True, [], 0.1, 0.1, 0.9, {}),
        value_level="high",
    )
    runner.residual_candidate_pool[candidate.duplicate_identity_key] = candidate
    runner.product_lifecycle[task.task_id] = _state(task.task_id)
    router = SimpleNamespace(
        router_name="proposed_lyapunov_rule_mode_safe_bandit",
        use_lyapunov_pricing=True,
        use_mode_controller=True,
    )
    proactive_decision = SimpleNamespace(
        proactive_flag=True,
        blockers=(),
        eriv_score=1.0,
        threshold=0.0,
        reason_code="eligible",
        leakage_safe_flag=True,
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_and_build_carryover_payload",
        lambda *args, **kwargs: (_payload(size_mb=0.5), task, proactive_decision),
    )
    guard_called = {"value": False}

    def _blocked_guard(**kwargs):
        guard_called["value"] = True
        return PayloadExecutionFeasibilityDecision(
            allowed=False,
            reason_code="followup_hard_safety_blocked_battery",
            blockers=("followup_hard_safety_blocked_battery",),
        )

    monkeypatch.setattr(runner, "_followup_hard_safety_guard", _blocked_guard)
    monkeypatch.setattr(
        runner.downlink_queue,
        "process_slot",
        lambda *args, **kwargs: pytest.fail("process_slot must not be called after guard block"),
    )
    result = runner._maybe_process_same_slot_carryover(
        timeline_step=_timeline(task),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=router,
        downlink_result=DownlinkResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "empty"),
        candidate_key=candidate.duplicate_identity_key,
    )
    assert result is None
    assert guard_called["value"]


def test_no_duplicate_roi_context_for_same_task():
    runner = _runner()
    task = _task("task_a")
    runner.downlink_queue.enqueue(_payload("task_a", "roi_context", 0.5))
    registered = runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline(task),
        lifecycle_state=ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"}),
        task=task,
        detection_result=DetectionResult([], 0.9, True, [], 0.1, 0.1, 0.9, {}),
        router=None,
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.5, 10.0),
    )
    assert not registered
    assert runner.last_duplicate_suppression_trace[-1]["duplicate_identity_key"] == "task_a:roi_context"


def test_other_task_roi_context_does_not_block_current_task():
    runner = _runner()
    task = _task("task_a")
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"})
    runner.downlink_queue.enqueue(_payload("other_task", "roi_context", 0.5))
    registered = runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline(task),
        lifecycle_state=state,
        task=task,
        detection_result=DetectionResult([], 0.9, True, [], 0.1, 0.1, 0.9, {}),
        router=None,
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.5, 10.0),
    )
    assert registered
    assert "roi_context" in state.pending_products


def _predicted(reward: float, size: float = 1.0) -> PredictedMetrics:
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
        predicted_delivery_factor=1.0,
        predicted_safety_violation=0.0,
        predicted_reward=reward,
        predicted_status="ok",
        predicted_detection_quality=1.0,
        predicted_incremental_information_utility=reward,
    )


def test_topk_window_budget_counts_real_selected_candidates():
    runner = _runner()
    task = _task()
    actions = [
        JointAction(f"a{i}", "tiny", "roi_context", "immediate", 1.0, 1.0)
        for i in range(3)
    ]
    filtered = runner._apply_roi_context_gate_to_actions(
        timeline_step=_timeline(task),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        candidate_actions=actions,
        predicted_map={action.action_name: _predicted(3.0 - index) for index, action in enumerate(actions)},
    )
    assert [action.action_name for action in filtered] == ["a0", "a1"]
    assert runner._roi_context_window_selected_count == 2
    assert runner.last_roi_context_window_budget_trace[-1]["reason_code"] == "top_k_budget_exceeded"
