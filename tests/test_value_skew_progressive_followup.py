from __future__ import annotations

import inspect

import pytest

from src.common.types import (
    CommState,
    DetectionResult,
    LinkTrace,
    ProductLifecycleState,
    ResourceState,
    ResourceTrace,
    TaskSample,
    TimelineStep,
)
from src.context.context_builder import ContextBuilder
from src.data.task_stream import TaskStreamBuilder
from src.data.value_skew_calibration import apply_value_skew_calibration, task_effective_value
from src.link.downlink_queue import DownlinkQueue
from src.payload.information_sufficiency import residual_information_utility
from src.payload.payload_sizer import generate_payload
from src.payload.proactive_followup import evaluate_proactive_followup_eligibility
from src.payload.roi_context_gate import evaluate_roi_context_gate, load_gate_profile
from src.simulator.step_runner import StepRunner


def _task(
    task_id: str,
    *,
    value: float = 0.8,
    task_class: str = "high_value",
    requirement: str = "context",
    high: bool = True,
    emergency: bool = False,
    value_level: str = "",
    calibrated: float | None = None,
) -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_path="mock://image",
        label_path="mock://label",
        scene_id=f"scene_{task_id}",
        virtual_time=0,
        task_value=value,
        deadline_s=120.0,
        difficulty=0.2,
        uncertainty=0.3,
        metadata={"old_task_value": value, "area_prior_bucket": "high"},
        task_class=task_class,
        high_value_flag=high,
        emergency_flag=emergency,
        product_requirement=requirement,
        value_level=value_level,
        calibrated_task_value=calibrated,
    )


def _value_skew_cfg() -> dict[str, object]:
    return {
        "task_value_model": {"enable_mission_prior_value_v2": True},
        "value_skew_calibration": {
            "enabled": True,
            "profile": "soft_moderate",
            "assignment_rule": "v2_conservative",
            "profiles": {"soft_moderate": {"low": 1, "medium": 2, "high": 5, "urgent": 10}},
        },
    }


def _gate_cfg(*, progressive: bool = True) -> dict[str, object]:
    return {
        "progressive_payload": {
            "roi_context_gate": {
                "enabled": True,
                "profile": "soft_moderate",
                "profiles": {
                    "soft_moderate": {
                        "eriv_threshold": 0.0,
                        "min_value_density": 0.0,
                        "eligible_product_requirements": ["context", "evidence"],
                        "eligible_value_levels": ["high", "urgent"],
                        "queue_pressure_threshold": 0.95,
                        "battery_guard_threshold": 0.10,
                        "thermal_guard_threshold": 0.99,
                        "min_remaining_deadline_s": 1.0,
                        "require_contact_opportunity": True,
                        "top_k_per_window": 2,
                        "window_budget": 2,
                    }
                },
            },
            "roi_context_after_roi": {"enabled": progressive, "residual_utility_threshold": 0.01},
        },
        "epsilon": 1.0e-9,
    }


def test_legacy_unchanged_when_value_skew_disabled():
    dataset_cfg = {"dataset_name": "mock", "seed": 2026}
    task_cfg = {"mode": "one_task_per_slot", "value_skew_calibration": {"enabled": False}}
    base = TaskStreamBuilder(dataset_cfg, {"mode": "one_task_per_slot"}).build(8, 20.0)
    disabled = TaskStreamBuilder(dataset_cfg, task_cfg).build(8, 20.0)
    assert [(t.task_id, t.task_value, t.calibrated_task_value) for t in base if t] == [
        (t.task_id, t.task_value, t.calibrated_task_value) for t in disabled if t
    ]


def test_mean_preserving_value_skew_does_not_overwrite_task_value():
    stream = [_task("a", value=0.4, task_class="routine", requirement="verification", high=False), _task("b", value=0.9)]
    calibrated = [task for task in apply_value_skew_calibration(stream, _value_skew_cfg()) if task]
    assert sum(task.calibrated_task_value for task in calibrated) / len(calibrated) == pytest.approx((0.4 + 0.9) / 2)
    assert [task.task_value for task in calibrated] == [0.4, 0.9]
    for task in calibrated:
        assert task.base_prior_score is not None
        assert task.value_level in {"low", "medium", "high", "urgent"}
        assert task.value_skew_profile == "soft_moderate"
        assert task.normalization_alpha > 0.0
        assert task.value_assignment_reason


def test_value_skew_uses_no_label_or_delivery_leakage():
    source = inspect.getsource(apply_value_skew_calibration)
    forbidden = ["label", "bbox", "delivery", "delivered", "closure", "target_count", "future"]
    assert not [token for token in forbidden if token in source]


def test_module_reward_payload_and_eriv_share_calibrated_value():
    task = _task("shared", value=0.5, calibrated=1.25, value_level="high")
    context = ContextBuilder(queue_max_mb=10.0).build(
        task,
        ResourceState(battery_soc=0.8, ai_energy_budget=0.9, thermal_pressure=0.2),
        CommState(queue_mb=0.0, queue_max_mb=10.0),
        LinkTrace(0, True, 10.0, 80.0, 1.0, 20.0),
    )
    detection = DetectionResult([], 0.9, True, [], 0.2, 0.1, 0.9, {})
    payload = generate_payload(task, detection, "roi_context", 0, delivered_products=set())
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"}, pending_products={"roi_context"}, residual_utility_remaining=0.3)
    decision = evaluate_proactive_followup_eligibility(
        state,
        {
            "variant": "proposed_lyapunov_rule_mode_safe_bandit",
            "router_name": "proposed_lyapunov_rule_mode_safe_bandit",
            "use_lyapunov_pricing": True,
            "use_mode_controller": True,
            "task_value": task_effective_value(task),
            "value_level": "high",
            "remaining_deadline_s": 100.0,
            "contact_available": True,
            "battery_soc": 0.8,
            "thermal_pressure": 0.2,
            "payload_size_mb": 1.0,
            "z_k_guided_followup_permission": True,
        },
        {**_gate_cfg(), "enabled": True, "eriv_threshold": 0.0, "residual_utility_threshold": 0.0},
    )
    assert context.V_t == pytest.approx(1.25)
    assert payload.task_value == pytest.approx(1.25)
    assert decision.eriv_breakdown.task_value == pytest.approx(1.25)


def test_profile_specific_gate_loading_and_semantic_low_value_block():
    enabled, profile, data = load_gate_profile(_gate_cfg())
    assert enabled
    assert profile == "soft_moderate"
    assert data["eligible_value_levels"] == ["high", "urgent"]
    decision = evaluate_roi_context_gate(
        product_requirement="verification",
        value_level="low",
        value_density=10.0,
        system_state={"contact_available": True, "remaining_deadline_s": 100.0},
        config=_gate_cfg(),
        eriv_score=10.0,
    )
    assert not decision.allowed
    assert "semantic_requirement_not_context_or_evidence" in decision.blockers
    assert "value_level_not_high_or_urgent" in decision.blockers


def test_top_k_gate_blocks_out_of_budget_candidate():
    decision = evaluate_roi_context_gate(
        product_requirement="context",
        value_level="high",
        value_density=10.0,
        system_state={"contact_available": True, "remaining_deadline_s": 100.0, "candidate_rank": 3},
        config=_gate_cfg(),
        eriv_score=10.0,
    )
    assert not decision.allowed
    assert "top_k_budget_exceeded" in decision.blockers


def _runner_for_second_stage(enabled: bool = True) -> StepRunner:
    runner = StepRunner.__new__(StepRunner)
    runner.simulator_cfg = _gate_cfg(progressive=enabled)
    runner.payload_profiles_cfg = None
    runner.downlink_queue = DownlinkQueue(10.0, 20.0)
    runner.residual_candidate_pool = {}
    runner.last_proactive_followup_decision = None
    return runner


def _timeline_step(task: TaskSample) -> TimelineStep:
    return TimelineStep(
        slot_index=1,
        slot_seconds=20.0,
        task=task,
        resource_trace=ResourceTrace(1, 0.0, 0.0, 0.0),
        link_trace=LinkTrace(1, True, 10.0, 80.0, 1.0, 20.0),
    )


def test_second_stage_roi_context_pending_registration_and_no_duplicate():
    task = _task("stage", value=0.9, value_level="high", calibrated=0.9)
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"})
    detection = DetectionResult([], 0.9, True, [], 0.2, 0.1, 0.9, {})
    runner = _runner_for_second_stage(True)
    registered = runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline_step(task),
        lifecycle_state=state,
        task=task,
        detection_result=detection,
        router=None,
        resource_state=ResourceState(0.8, 0.9, 0.2),
        comm_state=CommState(0.0, 10.0),
    )
    duplicate = runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline_step(task),
        lifecycle_state=state,
        task=task,
        detection_result=detection,
        router=None,
        resource_state=ResourceState(0.8, 0.9, 0.2),
        comm_state=CommState(0.0, 10.0),
    )
    assert registered
    assert not duplicate
    assert "roi_context" in state.pending_products


def test_progressive_followup_disabled_under_legacy():
    task = _task("legacy_stage", value=0.9, value_level="high", calibrated=0.9)
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"})
    detection = DetectionResult([], 0.9, True, [], 0.2, 0.1, 0.9, {})
    runner = _runner_for_second_stage(False)
    assert not runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline_step(task),
        lifecycle_state=state,
        task=task,
        detection_result=detection,
        router=None,
        resource_state=ResourceState(0.8, 0.9, 0.2),
        comm_state=CommState(0.0, 10.0),
    )


def test_integrated_closure_rules_for_verification_context_and_evidence():
    runner = _runner_for_second_stage(True)
    verification = ProductLifecycleState("v", product_requirement="verification", delivered_products={"roi"})
    context = ProductLifecycleState("c", product_requirement="context", delivered_products={"roi"})
    evidence = ProductLifecycleState("e", product_requirement="evidence", delivered_products={"roi", "roi_context"})
    assert runner._integrated_branch_closure_satisfied(verification)
    assert not runner._integrated_branch_closure_satisfied(context)
    assert runner._integrated_branch_closure_satisfied(evidence)
    assert residual_information_utility({"roi"}, "evidence") > 0.0
