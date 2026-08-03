"""Pure eligibility rule for reclaiming a deferred medium action.

The rule does not rescore actions or alter safety.  It is evaluated after
corrected REQALIGN and may restore only the original selector action when an
online shadow check predicts no high/urgent closure displacement.
"""

from __future__ import annotations

from dataclasses import dataclass


EPSILON = 1.0e-12


@dataclass(frozen=True)
class MediumReclaimFacts:
    value_level: str
    projection_reason: str
    reqalign_is_defer: bool
    original_action_available: bool
    original_hard_safe: bool
    original_deadline_feasible: bool
    original_score: float
    defer_score: float
    predicted_reward: float
    urgent_closure_displacement: int
    high_closure_displacement: int


@dataclass(frozen=True)
class MediumReclaimDecision:
    reclaim: bool
    reason: str


def decide_medium_slack_reclaim(
    facts: MediumReclaimFacts,
    *,
    require_positive_predicted_reward: bool = True,
) -> MediumReclaimDecision:
    """Return a deterministic decision using online information only."""

    checks = (
        (
            str(facts.value_level).lower() == "medium",
            "not_medium",
        ),
        (
            facts.projection_reason
            == "ordinary_required_product_infeasible",
            "projection_reason_not_eligible",
        ),
        (bool(facts.reqalign_is_defer), "reqalign_not_defer"),
        (
            bool(facts.original_action_available),
            "original_action_unavailable",
        ),
        (bool(facts.original_hard_safe), "original_not_hard_safe"),
        (
            bool(facts.original_deadline_feasible),
            "original_not_deadline_feasible",
        ),
        (
            float(facts.original_score)
            >= float(facts.defer_score) - EPSILON,
            "original_score_below_defer",
        ),
        (
            (
                not require_positive_predicted_reward
                or float(facts.predicted_reward) > EPSILON
            ),
            "non_positive_predicted_reward",
        ),
        (
            int(facts.urgent_closure_displacement) == 0,
            "urgent_closure_displacement",
        ),
        (
            int(facts.high_closure_displacement) == 0,
            "high_closure_displacement",
        ),
    )
    for passed, reason in checks:
        if not passed:
            return MediumReclaimDecision(False, reason)
    return MediumReclaimDecision(True, "medium_slack_reclaimed")
