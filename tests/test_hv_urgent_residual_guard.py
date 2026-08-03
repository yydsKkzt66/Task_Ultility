from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.routers.hv_urgent_residual_guard import (
    ActiveResidual,
    ParetoActionCandidate,
    ResourceDemand,
    dynamic_minimum_reservation,
    pareto_nondominated,
    promotion_status,
    released_resources,
    requires_cross_task_guard,
    select_pareto_guarded_action,
    shadow_displacement,
)
from src.routers.requirement_aware_projection import (
    ProjectionCandidate,
    closure_credit_due,
    immediate_product,
    project_requirement_aware_action,
    residual_required_product,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Action:
    action_name: str
    model_name: str = "balanced"
    payload_policy: str = "roi"


DEFER = Action("delay/store", "delay_store", "defer")


def residual(
    task_id: str,
    level: str,
    *,
    feasible: bool = True,
    safe: bool = True,
    utility: float = 2.0,
    demand: ResourceDemand | None = None,
) -> ActiveResidual:
    return ActiveResidual(
        task_id=task_id,
        value_level=level,
        effective_task_value=25.0 if level == "urgent" else 10.0,
        delivered_products=("metadata",),
        residual_required_product="evidence",
        absolute_deadline_s=1000.0,
        remaining_slack_s=500.0,
        demand=demand or ResourceDemand(10.0, 1.0, 1.0, 1_000_000.0, 0.05),
        residual_compute_requirement_s=10.0,
        residual_communication_energy_wh=0.2,
        earliest_generation_completion_s=510.0 if feasible else None,
        candidate_contact_windows=("contact_1",) if feasible else (),
        earliest_ground_arrival_s=600.0 if feasible else None,
        closure_feasible_before_deadline=feasible,
        predicted_marginal_hv_net_utility=utility,
        predicted_closure_increment=1 if feasible else 0,
        hard_safe=safe,
        original_score=utility,
    )


def capacity() -> ResourceDemand:
    return ResourceDemand(100.0, 10.0, 10.0, 10_000_000.0, 1.0)


def action_candidate(
    name: str,
    *,
    level: str = "low",
    j_all: float = 0.0,
    j_hv: float = 0.0,
    closure: int = 0,
    urgent_loss: int = 0,
    high_loss: int = 0,
    score: float = 0.0,
    safe: bool = True,
    deadline: bool = True,
    defer: bool = False,
) -> ParetoActionCandidate:
    action = DEFER if defer else Action(name)
    return ParetoActionCandidate(
        action=action,
        action_name=action.action_name,
        original_score=score,
        effective_task_value={"urgent": 25.0, "high": 10.0}.get(level, 1.0),
        remaining_slack_s=100.0,
        task_id=f"synthetic_{level}",
        value_level=level,
        delta_j_all_net_hat=j_all,
        delta_j_hv_net_hat=j_hv,
        delta_n_hv_closure_hat=closure,
        urgent_closure_displacement=urgent_loss,
        high_closure_displacement=high_loss,
        hard_safe=safe,
        deadline_feasible=deadline,
        is_defer=defer,
    )


def projection_candidate(
    action: Action,
    *,
    score: float,
    feasible: bool = True,
) -> ProjectionCandidate:
    return ProjectionCandidate(
        action=action,
        action_name=action.action_name,
        model_name=action.model_name,
        product=immediate_product(action.payload_policy),
        original_score=score,
        predicted_ground_arrival_s=50.0 if feasible else None,
        hard_safe=True,
        deadline_feasible=feasible,
        score_above_defer=True,
        contact_found=feasible,
    )


def test_01_ordinary_policy_configuration_is_unchanged() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "hv_urgent_residual_protection_v1.yaml").read_text(encoding="utf-8")
    )["hv_urgent_residual_guard"]
    assert config["ordinary_policy"] == {"unchanged": True, "allow_progressive": False}
    assert not config["change_task_values"]
    assert not config["relax_hard_safety"]


def test_02_low_never_enters_progressive_delivery() -> None:
    metadata = Action("balanced+metadata", payload_policy="metadata")
    result = project_requirement_aware_action(
        selected_action=metadata,
        selected_product="metadata",
        selected_model="balanced",
        residual_requirement="verification",
        candidates=[
            projection_candidate(metadata, score=2.0),
            projection_candidate(DEFER, score=0.0),
        ],
        value_level="low",
    )
    assert result.final_action is DEFER
    assert not result.progressive_reentry
    assert result.residual_requirement_after_action == ""


def test_03_medium_never_enters_progressive_delivery() -> None:
    metadata = Action("balanced+metadata", payload_policy="metadata")
    result = project_requirement_aware_action(
        selected_action=metadata,
        selected_product="metadata",
        selected_model="balanced",
        residual_requirement="verification",
        candidates=[
            projection_candidate(metadata, score=2.0),
            projection_candidate(DEFER, score=0.0),
        ],
        value_level="medium",
    )
    assert result.final_action is DEFER
    assert not result.progressive_reentry


def test_04_direct_feasible_urgent_closes_without_metadata_first() -> None:
    metadata = Action("balanced+metadata", payload_policy="metadata")
    evidence = Action("balanced+roi_context", payload_policy="roi_context")
    result = project_requirement_aware_action(
        selected_action=metadata,
        selected_product="metadata",
        selected_model="balanced",
        residual_requirement="evidence",
        candidates=[
            projection_candidate(metadata, score=3.0),
            projection_candidate(evidence, score=2.0),
            projection_candidate(DEFER, score=0.0),
        ],
        value_level="urgent",
    )
    assert result.final_action is evidence
    assert result.direct_closure_feasible
    assert not result.progressive_reentry


def test_05_direct_infeasible_urgent_can_use_metadata_first() -> None:
    metadata = Action("balanced+metadata", payload_policy="metadata")
    evidence = Action("balanced+roi_context", payload_policy="roi_context")
    result = project_requirement_aware_action(
        selected_action=metadata,
        selected_product="metadata",
        selected_model="balanced",
        residual_requirement="evidence",
        candidates=[
            projection_candidate(metadata, score=2.0),
            projection_candidate(evidence, score=3.0, feasible=False),
            projection_candidate(DEFER, score=0.0),
        ],
        value_level="urgent",
    )
    assert result.final_action is metadata
    assert result.progressive_reentry
    assert result.residual_requirement_after_action == "evidence"


def test_06_metadata_preserves_autonomous_residual_for_reassessment() -> None:
    assert residual_required_product("evidence", {"metadata"}) == "evidence"
    source = (ROOT / "scripts" / "run_prop_reqalign_seed2026_v1.py").read_text(encoding="utf-8")
    assert "prepare_step_for_reentry" in source
    assert "_register_high_value_defer_recheck" in source


def test_07_low_is_rejected_when_it_displaces_an_urgent_closure() -> None:
    low = action_candidate("balanced+roi", j_all=5.0, urgent_loss=1)
    defer = action_candidate("defer", defer=True)
    decision = select_pareto_guarded_action([low, defer])
    assert decision.selected.is_defer
    assert low in decision.rejected_by_displacement


def test_08_low_can_run_when_it_does_not_displace_hv() -> None:
    low = action_candidate("balanced+roi", j_all=5.0, score=2.0)
    defer = action_candidate("defer", defer=True)
    assert select_pareto_guarded_action([low, defer]).selected is low
    assert not requires_cross_task_guard(low)


def test_09_physically_infeasible_urgent_is_not_reserved() -> None:
    urgent = residual("urgent_infeasible", "urgent", feasible=False)
    reservation = dynamic_minimum_reservation([urgent], capacity())
    assert reservation.protected_task_ids == ()
    assert reservation.infeasible_urgent_task_ids == ("urgent_infeasible",)


def test_10_infeasible_urgent_does_not_block_feasible_high() -> None:
    urgent = residual("urgent_infeasible", "urgent", feasible=False)
    high = residual("high_feasible", "high")
    reservation = dynamic_minimum_reservation([urgent, high], capacity())
    assert reservation.protected_task_ids == ("high_feasible",)


def test_11_closed_or_invalid_reservation_is_released_immediately() -> None:
    previous = residual("urgent_open", "urgent").demand
    release = released_resources(previous, ResourceDemand())
    assert release.released == previous
    assert release.increased == ResourceDemand()


def test_12_high_residual_is_not_consumed_by_an_ordinary_action() -> None:
    high = residual("high_open", "high")
    displaced = shadow_displacement(
        [high],
        capacity_before=high.demand,
        action_demand=high.demand,
    )
    assert displaced.high_closure_displacement == 1
    ordinary = action_candidate("ordinary", j_all=3.0, high_loss=1)
    defer = action_candidate("defer", defer=True)
    assert select_pareto_guarded_action([ordinary, defer]).selected.is_defer


def test_13_roi_delivery_does_not_close_an_evidence_requirement() -> None:
    assert residual_required_product("evidence", {"metadata", "roi"}) == "evidence"
    assert not closure_credit_due("evidence", {"metadata"}, {"metadata", "roi"})


def test_14_roi_context_closure_credit_is_counted_once() -> None:
    before = {"metadata", "roi"}
    after = before | {"roi_context"}
    assert closure_credit_due("evidence", before, after)
    assert not closure_credit_due("evidence", after, after)


def test_15_pareto_dominated_action_is_removed() -> None:
    better = action_candidate("better", j_all=2.0, j_hv=1.0, closure=1)
    worse = action_candidate("worse", j_all=1.0, j_hv=1.0, closure=1)
    nondominated, dominated = pareto_nondominated([better, worse])
    assert nondominated == [better]
    assert dominated == [worse]


def test_16_equal_pareto_candidates_still_use_original_prop_score() -> None:
    lower_score = action_candidate("lower", j_all=2.0, score=1.0)
    higher_score = action_candidate("higher", j_all=2.0, score=3.0)
    decision = select_pareto_guarded_action([lower_score, higher_score])
    assert decision.selected is higher_score


def test_17_disabled_guard_is_default_and_not_imported_by_off_variants() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "hv_urgent_residual_protection_v1.yaml").read_text(encoding="utf-8")
    )["hv_urgent_residual_guard"]
    runner = (ROOT / "scripts" / "run_hv_urgent_residual_protection_v1.py").read_text(encoding="utf-8")
    assert config["enabled"] is False
    assert '"REQALIGN_GUARD_OFF": (True, False' in runner
    assert "if guard_enabled:" in runner


def test_18_baselines_have_no_guard_runtime_binding() -> None:
    runner = (ROOT / "scripts" / "run_hv_urgent_residual_protection_v1.py").read_text(encoding="utf-8")
    tree = ast.parse(runner)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "run_native_binding_method" not in called_attributes
    assert "run_myopic_method" not in called_attributes
    assert '"baseline_runtime_runs_started": False' in runner


def test_19_unsafe_or_late_candidate_cannot_be_selected() -> None:
    unsafe = action_candidate("unsafe", j_all=10.0, safe=False)
    late = action_candidate("late", j_all=9.0, deadline=False)
    defer = action_candidate("defer", defer=True)
    assert select_pareto_guarded_action([unsafe, late, defer]).selected.is_defer


def test_20_no_duplicate_delivery_or_duplicate_utility_path() -> None:
    source = (ROOT / "src" / "routers" / "hv_urgent_residual_guard.py").read_text(encoding="utf-8")
    assert "delivered_products.add" not in source
    assert "observed_reward" not in source
    assert closure_credit_due("evidence", {"metadata"}, {"metadata", "roi_context"})
    assert not closure_credit_due("evidence", {"metadata", "roi_context"}, {"metadata", "roi_context"})


def test_hv_replacement_requires_no_urgent_loss_and_nonnegative_hv_delta() -> None:
    allowed = action_candidate(
        "high_closure",
        level="high",
        j_all=2.0,
        j_hv=1.0,
        closure=0,
        high_loss=1,
    )
    urgent_loss = action_candidate(
        "high_displaces_urgent",
        level="high",
        j_all=3.0,
        j_hv=2.0,
        closure=0,
        urgent_loss=1,
    )
    decision = select_pareto_guarded_action([allowed, urgent_loss])
    assert decision.selected is allowed
    assert urgent_loss in decision.rejected_by_displacement


def test_current_urgent_may_replace_one_urgent_when_net_count_and_hv_utility_do_not_drop() -> None:
    current_urgent = action_candidate(
        "current_urgent",
        level="urgent",
        j_all=1.0,
        j_hv=1.0,
        closure=0,
        urgent_loss=1,
    )
    defer = action_candidate("defer", level="urgent", defer=True)
    assert current_urgent.delta_n_urgent_closure_hat == 0
    assert select_pareto_guarded_action([current_urgent, defer]).selected is current_urgent


def test_cross_task_guard_does_not_become_a_second_utility_gate() -> None:
    reqalign = action_candidate("reqalign", level="medium", j_all=-5.0, score=2.0)
    assert not requires_cross_task_guard(reqalign)
    displaced = action_candidate("displaced", level="medium", j_all=5.0, urgent_loss=1)
    assert requires_cross_task_guard(displaced)


def test_preregistered_promotion_rules_are_exact() -> None:
    old = {"J_all_net": 1, "J_HV_net": 1, "HV_closure_count": 1, "urgent_closure_count": 1}
    corrected = {"J_all_net": 2, "J_HV_net": 2, "HV_closure_count": 2, "urgent_closure_count": 2}
    strict = {"J_all_net": 3, "J_HV_net": 3, "HV_closure_count": 3, "urgent_closure_count": 2}
    pareto = {"J_all_net": 2, "J_HV_net": 2, "HV_closure_count": 3, "urgent_closure_count": 2}
    failed = {"J_all_net": 2, "J_HV_net": 1, "HV_closure_count": 3, "urgent_closure_count": 2}
    assert promotion_status(
        old_prop=old,
        corrected_reqalign=corrected,
        guarded=strict,
        safety_violations=0,
        ordinary_logic_unchanged=True,
    ).endswith("STRICT_THREE_METRIC_IMPROVEMENT")
    assert promotion_status(
        old_prop=old,
        corrected_reqalign=corrected,
        guarded=pareto,
        safety_violations=0,
        ordinary_logic_unchanged=True,
    ).endswith("PARETO_NONDEGRADING_CANDIDATE")
    assert promotion_status(
        old_prop=old,
        corrected_reqalign=corrected,
        guarded=failed,
        safety_violations=0,
        ordinary_logic_unchanged=True,
    ).endswith("NO_PROMOTION")
