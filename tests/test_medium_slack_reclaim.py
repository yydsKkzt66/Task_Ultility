from src.routers.medium_slack_reclaim import (
    MediumReclaimFacts,
    decide_medium_slack_reclaim,
)


def eligible(**overrides):
    values = {
        "value_level": "medium",
        "projection_reason": "ordinary_required_product_infeasible",
        "reqalign_is_defer": True,
        "original_action_available": True,
        "original_hard_safe": True,
        "original_deadline_feasible": True,
        "original_score": 1.0,
        "defer_score": 0.2,
        "predicted_reward": 0.5,
        "urgent_closure_displacement": 0,
        "high_closure_displacement": 0,
    }
    values.update(overrides)
    return MediumReclaimFacts(**values)


def test_eligible_medium_defer_is_reclaimed():
    decision = decide_medium_slack_reclaim(eligible())
    assert decision.reclaim
    assert decision.reason == "medium_slack_reclaimed"


def test_non_medium_is_never_reclaimed():
    assert not decide_medium_slack_reclaim(
        eligible(value_level="low")
    ).reclaim
    assert not decide_medium_slack_reclaim(
        eligible(value_level="high")
    ).reclaim


def test_wrong_projection_reason_is_never_reclaimed():
    decision = decide_medium_slack_reclaim(
        eligible(projection_reason="already_aligned")
    )
    assert not decision.reclaim
    assert decision.reason == "projection_reason_not_eligible"


def test_hard_safety_and_deadline_are_required():
    assert not decide_medium_slack_reclaim(
        eligible(original_hard_safe=False)
    ).reclaim
    assert not decide_medium_slack_reclaim(
        eligible(original_deadline_feasible=False)
    ).reclaim


def test_positive_existing_score_is_required():
    assert not decide_medium_slack_reclaim(
        eligible(original_score=0.1, defer_score=0.2)
    ).reclaim
    assert not decide_medium_slack_reclaim(
        eligible(predicted_reward=0.0)
    ).reclaim


def test_existing_score_only_mode_does_not_add_raw_reward_gate():
    decision = decide_medium_slack_reclaim(
        eligible(predicted_reward=-1.0),
        require_positive_predicted_reward=False,
    )
    assert decision.reclaim


def test_any_hv_closure_displacement_blocks_reclaim():
    urgent = decide_medium_slack_reclaim(
        eligible(urgent_closure_displacement=1)
    )
    high = decide_medium_slack_reclaim(
        eligible(high_closure_displacement=1)
    )
    assert not urgent.reclaim
    assert urgent.reason == "urgent_closure_displacement"
    assert not high.reclaim
    assert high.reason == "high_closure_displacement"
