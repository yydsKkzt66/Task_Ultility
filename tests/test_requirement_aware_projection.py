from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import src.routers.requirement_aware_projection as requirement_projection
from scripts.run_prop_reqalign_seed2026_v1 import RequirementAwareRouterWrapper
from src.routers.requirement_aware_projection import (
    ContactCapacitySlot,
    ProjectionCandidate,
    choose_competing_task,
    closure_credit_due,
    immediate_product,
    maybe_project_requirement_aware,
    predict_ground_arrival,
    product_rank,
    product_satisfies,
    project_requirement_aware_action,
    residual_required_product,
)


@dataclass(frozen=True)
class Action:
    action_name: str
    model_name: str
    payload_policy: str


DEFER = Action("delay/store", "none", "defer")


def wrapper_delegate() -> SimpleNamespace:
    return SimpleNamespace(
        router_name="proposed_lyapunov_rule_mode_safe_bandit",
        use_lyapunov_pricing=True,
        use_mode_controller=True,
        use_guided_utility=True,
    )


def lifecycle(requirement: str = "evidence") -> SimpleNamespace:
    return SimpleNamespace(
        product_requirement=requirement,
        delivered_products=set(),
        pending_products=set(),
        task_closed_flag=False,
        residual_parent_payload_plan="",
        pending_residual_registered_slot=None,
        residual_requested_count=0,
        residual_request_time_s=None,
        residual_utility_remaining=0.0,
        residual_expired_count=0,
    )


@dataclass(frozen=True)
class RetryStep:
    slot_index: int
    slot_seconds: float
    task: object | None
    link_trace: object
    resource_trace: object | None = None


def candidate(
    action: Action,
    *,
    score: float,
    arrival: float | None = 50.0,
    safe: bool = True,
    deadline: bool = True,
    above_defer: bool = True,
    reasons: tuple[str, ...] = (),
    contact: bool = True,
) -> ProjectionCandidate:
    return ProjectionCandidate(
        action=action,
        action_name=action.action_name,
        model_name=action.model_name,
        product=immediate_product(action.payload_policy),
        original_score=score,
        predicted_ground_arrival_s=arrival,
        hard_safe=safe,
        deadline_feasible=deadline,
        score_above_defer=above_defer,
        safety_reasons=reasons,
        contact_found=contact,
    )


def project(
    selected: Action,
    requirement: str,
    candidates: list[ProjectionCandidate],
    *,
    level: str,
):
    rows = list(candidates)
    if not any(row.action_name == DEFER.action_name for row in rows):
        rows.append(candidate(DEFER, score=0.0, arrival=0.0))
    return project_requirement_aware_action(
        selected_action=selected,
        selected_product=immediate_product(selected.payload_policy),
        selected_model=selected.model_name,
        residual_requirement=requirement,
        candidates=rows,
        value_level=level,
    )


def test_product_aliases_follow_public_evaluator_order() -> None:
    assert product_rank("metadata") < product_rank("verification") < product_rank("evidence")
    assert product_satisfies("ROI+Context", "evidence")
    assert product_satisfies("roi", "verification")
    assert not product_satisfies("metadata", "verification")


@pytest.mark.parametrize("level", ["low", "medium"])
def test_ordinary_original_action_already_satisfies_requirement_is_unchanged(level: str) -> None:
    original = Action("balanced+roi", "balanced", "roi")
    result = project(original, "verification", [candidate(original, score=2.0)], level=level)
    assert result.final_action is original
    assert result.projection_reason == "already_aligned"
    assert result.one_shot_delivery
    assert not result.progressive_reentry


@pytest.mark.parametrize("level", ["low", "medium"])
def test_ordinary_same_model_required_product_is_one_shot(level: str) -> None:
    original = Action("balanced+metadata", "balanced", "metadata")
    required = Action("balanced+roi", "balanced", "roi")
    result = project(
        original,
        "verification",
        [candidate(original, score=3.0), candidate(required, score=2.0)],
        level=level,
    )
    assert result.final_action is required
    assert result.projection_reason == "requirement_projected"
    assert result.one_shot_delivery
    assert not result.progressive_reentry
    assert result.residual_requirement_after_action == ""


@pytest.mark.parametrize("level", ["low", "medium"])
def test_ordinary_infeasible_required_product_defers_without_metadata_or_residual(level: str) -> None:
    original = Action("balanced+metadata", "balanced", "metadata")
    required = Action("balanced+roi", "balanced", "roi")
    result = project(
        original,
        "verification",
        [
            candidate(original, score=3.0),
            candidate(required, score=2.0, safe=False, reasons=("thermal pressure violation",)),
        ],
        level=level,
    )
    assert result.final_action is DEFER
    assert result.projection_reason == "ordinary_required_product_infeasible"
    assert result.closure_status_after == "deferred"
    assert result.residual_requirement_after_action == ""
    assert not result.progressive_reentry


@pytest.mark.parametrize("level", ["high", "urgent"])
def test_high_value_direct_final_product_is_preferred_over_metadata(level: str) -> None:
    original = Action("balanced+metadata", "balanced", "metadata")
    direct = Action("balanced+roi_context", "balanced", "roi_context")
    result = project(
        original,
        "evidence",
        [candidate(original, score=4.0), candidate(direct, score=2.0)],
        level=level,
    )
    assert result.final_action is direct
    assert result.projection_reason == "requirement_projected"
    assert result.direct_closure_feasible
    assert result.one_shot_delivery


@pytest.mark.parametrize("level", ["high", "urgent"])
def test_high_value_metadata_first_only_when_final_is_not_directly_feasible(level: str) -> None:
    metadata = Action("balanced+metadata", "balanced", "metadata")
    final = Action("balanced+roi_context", "balanced", "roi_context")
    result = project(
        metadata,
        "evidence",
        [
            candidate(metadata, score=1.0),
            candidate(final, score=2.0, deadline=False, arrival=101.0),
        ],
        level=level,
    )
    assert result.final_action is metadata
    assert result.projection_reason == "metadata_first_progressive_fallback"
    assert result.closure_status_after == "incomplete"
    assert result.residual_requirement_after_action == "evidence"
    assert result.progressive_reentry
    assert not result.one_shot_delivery


def test_metadata_then_roi_does_not_close_evidence_or_clear_residual() -> None:
    assert residual_required_product("evidence", {"metadata"}) == "evidence"
    assert residual_required_product("evidence", {"metadata", "roi"}) == "evidence"
    assert not closure_credit_due("evidence", {"metadata"}, {"metadata", "roi"})


def test_later_roi_context_closes_once_and_only_once() -> None:
    before = {"metadata", "roi"}
    after = {"metadata", "roi", "roi_context"}
    assert closure_credit_due("evidence", before, after)
    assert not closure_credit_due("evidence", after, after | {"roi_context"})
    assert residual_required_product("evidence", after) == ""


@pytest.mark.parametrize("level", ["high", "urgent"])
def test_high_value_defers_and_retains_residual_when_metadata_and_final_are_infeasible(level: str) -> None:
    original = Action("balanced+roi", "balanced", "roi")
    metadata = Action("balanced+metadata", "balanced", "metadata")
    final = Action("balanced+roi_context", "balanced", "roi_context")
    result = project(
        original,
        "evidence",
        [
            candidate(original, score=3.0),
            candidate(metadata, score=1.0, safe=False, reasons=("battery reserve violation",)),
            candidate(final, score=2.0, safe=False, reasons=("thermal pressure violation",)),
        ],
        level=level,
    )
    assert result.final_action is DEFER
    assert result.projection_reason == "high_value_defer_recheck"
    assert result.residual_requirement_after_action == "evidence"
    assert result.progressive_reentry
    assert not result.direct_closure_feasible


def test_high_value_defer_recheck_is_registered_in_existing_lifecycle_after_execution() -> None:
    task = SimpleNamespace(
        task_id="synthetic_high_retry",
        value_level="high",
        metadata={},
        product_requirement="evidence",
    )
    state = lifecycle("evidence")
    runner = SimpleNamespace(
        product_lifecycle={task.task_id: state},
        _get_product_lifecycle=lambda _task: state,
        _minimum_closing_residual_product=lambda _requirement, _delivered: "roi_context",
    )
    wrapper = RequirementAwareRouterWrapper(
        wrapper_delegate(),
        {},
        [],
        {},
        requirement_projection,
    )
    wrapper.bind_step_runner(runner)
    wrapper.bind_step(
        RetryStep(10, 20.0, task, SimpleNamespace(in_contact=False)),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    row = {"projection_reason": "high_value_defer_recheck"}
    assert wrapper._register_high_value_defer_recheck(task, row)
    assert state.residual_parent_payload_plan == wrapper._DEFER_RECHECK_MARKER
    assert state.pending_products == {"roi_context"}
    assert state.residual_requested_count == 1
    assert row["defer_recheck_registered"] == 1


def test_ordinary_defer_never_registers_progressive_reentry() -> None:
    task = SimpleNamespace(
        task_id="synthetic_ordinary",
        value_level="medium",
        metadata={},
        product_requirement="verification",
    )
    state = lifecycle("verification")
    runner = SimpleNamespace(
        product_lifecycle={task.task_id: state},
        _get_product_lifecycle=lambda _task: state,
        _minimum_closing_residual_product=lambda _requirement, _delivered: "roi",
    )
    wrapper = RequirementAwareRouterWrapper(wrapper_delegate(), {}, [], {}, requirement_projection)
    wrapper.bind_step_runner(runner)
    wrapper.bind_step(
        RetryStep(10, 20.0, task, SimpleNamespace(in_contact=False)),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    assert not wrapper._register_high_value_defer_recheck(
        task,
        {"projection_reason": "ordinary_required_product_infeasible"},
    )
    assert state.pending_products == set()
    assert state.residual_parent_payload_plan == ""


def test_physically_infeasible_urgent_reentry_does_not_preempt_feasible_high() -> None:
    urgent = SimpleNamespace(
        task_id="synthetic_urgent_retry",
        value_level="urgent",
        metadata={},
        product_requirement="evidence",
        deadline_s=200.0,
        arrival_slot=0,
    )
    high = SimpleNamespace(
        task_id="synthetic_high_retry",
        value_level="high",
        metadata={},
        product_requirement="evidence",
        deadline_s=200.0,
        arrival_slot=0,
    )
    urgent_state = lifecycle("evidence")
    high_state = lifecycle("evidence")
    for state in (urgent_state, high_state):
        state.pending_products.add("roi_context")
        state.residual_parent_payload_plan = RequirementAwareRouterWrapper._DEFER_RECHECK_MARKER
    runner = SimpleNamespace(
        product_lifecycle={urgent.task_id: urgent_state, high.task_id: high_state},
        task_snapshots={urgent.task_id: urgent, high.task_id: high},
        staged_delivery_requests={},
        _queued_product_keys=lambda: set(),
        _contact_window_id_for_slot=lambda _slot: "contact_synthetic",
        _minimum_closing_residual_product=lambda _requirement, _delivered: "roi_context",
    )
    wrapper = RequirementAwareRouterWrapper(wrapper_delegate(), {}, [], {}, requirement_projection)
    wrapper.bind_step_runner(runner)

    def physical(task, _step, _resource, _comm):
        feasible = task.task_id == high.task_id
        return {
            "task": task,
            "task_id": task.task_id,
            "effective_task_value": 25.0 if task is urgent else 10.0,
            "remaining_slack_s": 100.0,
            "original_score": 2.0,
            "hard_safe": feasible,
            "deadline_feasible": feasible,
        }

    wrapper._reentry_physical_feasibility = physical
    idle = RetryStep(5, 20.0, None, SimpleNamespace(in_contact=True))
    effective = wrapper.prepare_step_for_reentry(idle, SimpleNamespace(), SimpleNamespace())
    assert effective.task is high
    assert wrapper.active_reentry_task_id == high.task_id
    assert urgent_state.pending_products == {"roi_context"}
    assert high_state.pending_products == set()


def test_minimum_satisfying_product_and_same_model_precede_original_score() -> None:
    original = Action("balanced+metadata", "balanced", "metadata")
    same_model = Action("balanced+roi", "balanced", "roi")
    higher_product = Action("balanced+roi_context", "balanced", "roi_context")
    other_model = Action("high_accuracy+roi", "high_accuracy", "roi")
    result = project(
        original,
        "verification",
        [
            candidate(same_model, score=1.0),
            candidate(higher_product, score=10.0),
            candidate(other_model, score=9.0),
        ],
        level="medium",
    )
    assert result.final_action is same_model


def test_below_defer_direct_product_is_not_forced() -> None:
    metadata = Action("balanced+metadata", "balanced", "metadata")
    final = Action("balanced+roi_context", "balanced", "roi_context")
    result = project(
        metadata,
        "evidence",
        [candidate(metadata, score=1.0), candidate(final, score=-1.0, above_defer=False)],
        level="urgent",
    )
    assert result.final_action is metadata
    assert result.projection_reason == "metadata_first_progressive_fallback"
    assert "negative_score" in result.blocked_reasons


def test_urgent_wins_when_both_tasks_are_feasible_and_value_is_higher() -> None:
    high = {
        "task_id": "synthetic_high",
        "effective_task_value": 10.0,
        "remaining_slack_s": 100.0,
        "original_score": 2.0,
        "hard_safe": True,
        "deadline_feasible": True,
    }
    urgent = {
        "task_id": "synthetic_urgent",
        "effective_task_value": 25.0,
        "remaining_slack_s": 110.0,
        "original_score": 1.99,
        "hard_safe": True,
        "deadline_feasible": True,
    }
    assert choose_competing_task([high, urgent]) is urgent


def test_infeasible_urgent_does_not_preempt_closable_high() -> None:
    high = {
        "task_id": "synthetic_high",
        "effective_task_value": 10.0,
        "remaining_slack_s": 100.0,
        "original_score": 2.0,
        "hard_safe": True,
        "deadline_feasible": True,
    }
    urgent = {
        "task_id": "synthetic_urgent",
        "effective_task_value": 25.0,
        "remaining_slack_s": -1.0,
        "original_score": 3.0,
        "hard_safe": True,
        "deadline_feasible": False,
    }
    assert choose_competing_task([urgent, high]) is high


def test_disabled_configuration_does_not_call_projector() -> None:
    calls = 0

    def forbidden_projector() -> Action:
        nonlocal calls
        calls += 1
        raise AssertionError("projection must be fully bypassed")

    original = Action("balanced+roi", "balanced", "roi")
    assert maybe_project_requirement_aware(
        enabled=False,
        selected_action=original,
        projector=forbidden_projector,
    ) is original
    assert calls == 0


def test_configuration_separates_ordinary_and_high_value_policies() -> None:
    cfg = yaml.safe_load(Path("configs/prop_requirement_projection_v1.yaml").read_text(encoding="utf-8"))
    alignment = cfg["requirement_alignment"]
    assert alignment["enabled"] is False
    assert alignment["ordinary_policy"] == {
        "value_levels": ["low", "medium"],
        "one_shot_required_product": True,
        "allow_metadata_fallback": False,
        "allow_progressive_reentry": False,
        "infeasible_action": "defer",
    }
    assert alignment["high_value_policy"]["value_levels"] == ["high", "urgent"]
    assert alignment["high_value_policy"]["prefer_direct_required_delivery"] is True
    assert alignment["urgent_policy"]["urgent_always_wins"] is False
    assert alignment["urgent_soft_cost_multiplier"] == 1.0


def test_baseline_child_branches_do_not_load_requirement_alignment() -> None:
    source = Path("scripts/run_prop_reqalign_seed2026_v1.py").read_text(encoding="utf-8")
    start = source.index("def child_main(")
    end = source.index("\ndef launch_child(", start)
    child = source[start:end]
    baseline_branch = child[child.index("elif args.label in BASELINE_LABELS") :]
    assert "load_projection_module" not in baseline_branch
    assert "RequirementAwareRouterWrapper" not in baseline_branch


def test_ground_arrival_forecast_includes_compute_queue_wait_contact_and_tx() -> None:
    prediction = predict_ground_arrival(
        current_time_s=0.0,
        compute_and_encode_s=5.0,
        backlog_mb=4.0,
        payload_mb=2.0,
        contact_slots=[ContactCapacitySlot(20.0, 30.0, 1.0)],
        deadline_s=40.0,
    )
    assert prediction.compute_finish_s == pytest.approx(5.0)
    assert prediction.downlink_start_s == pytest.approx(24.0)
    assert prediction.ground_arrival_s == pytest.approx(26.0)
    assert prediction.queue_delay_s == pytest.approx(19.0)
    assert prediction.transmission_time_s == pytest.approx(2.0)


def test_no_future_contact_has_no_predicted_ground_arrival() -> None:
    prediction = predict_ground_arrival(
        current_time_s=0.0,
        compute_and_encode_s=1.0,
        backlog_mb=0.0,
        payload_mb=1.0,
        contact_slots=[],
        deadline_s=10.0,
    )
    assert prediction.ground_arrival_s is None
    assert not prediction.contact_found
