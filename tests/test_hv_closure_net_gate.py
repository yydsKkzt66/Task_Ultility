from __future__ import annotations

from src.routers.hv_closure_net_gate import (
    HVClosureNetGateFacts,
    evaluate_hv_closure_net_gate,
)


def facts(**overrides):
    values = dict(
        enabled=True,
        value_level="urgent",
        closure_satisfying=True,
        original_score_above_defer=False,
        hard_safe=True,
        deadline_feasible=True,
        contact_found=True,
        eligible_by_existing_filters=True,
        capture_time_s=100.0,
        deadline_duration_s=1000.0,
        predicted_ground_arrival_s=500.0,
        effective_task_value=25.0,
        information_sufficiency=1.0,
        predicted_quality=0.8,
        public_energy_cost=0.1,
    )
    values.update(overrides)
    return HVClosureNetGateFacts(**values)


def test_disabled_gate_is_exact_original_gate_bypass():
    blocked = evaluate_hv_closure_net_gate(facts(enabled=False))
    passed = evaluate_hv_closure_net_gate(
        facts(enabled=False, original_score_above_defer=True)
    )
    assert not blocked.allowed
    assert passed.allowed
    assert blocked.reason == passed.reason == "disabled_exact_bypass"


def test_ordinary_task_is_never_rescued():
    result = evaluate_hv_closure_net_gate(facts(value_level="medium"))
    assert not result.allowed
    assert result.reason == "ordinary_exact_bypass"


def test_positive_public_net_rescues_negative_original_hv_score():
    result = evaluate_hv_closure_net_gate(facts())
    assert result.allowed
    assert result.rescued_negative_original_score
    assert result.predicted_public_net_utility > 0.0
    assert result.reason == "positive_public_closure_net_rescue"


def test_nonclosing_action_is_never_rescued():
    result = evaluate_hv_closure_net_gate(facts(closure_satisfying=False))
    assert not result.allowed
    assert result.reason == "not_a_closure_action"


def test_hard_safety_and_deadline_remain_hard_blocks():
    assert not evaluate_hv_closure_net_gate(facts(hard_safe=False)).allowed
    assert not evaluate_hv_closure_net_gate(facts(deadline_feasible=False)).allowed


def test_nonpositive_public_net_is_not_rescued():
    result = evaluate_hv_closure_net_gate(
        facts(public_energy_cost=100.0)
    )
    assert not result.allowed
    assert result.reason == "nonpositive_public_closure_net"


def test_existing_positive_score_is_preserved_without_reclassification():
    result = evaluate_hv_closure_net_gate(
        facts(original_score_above_defer=True, hard_safe=False)
    )
    assert result.allowed
    assert not result.rescued_negative_original_score
    assert result.reason == "existing_score_gate_passed"
