"""Resource-nonincreasing net-utility projection for ordinary tasks.

The corrected REQALIGN action is the reference action.  High and urgent tasks
hard-bypass this module.  Low and medium tasks may change only to an already
safe, on-time action that uses the same model and no more payload, energy, or
compute time than the reference action.  This makes the layer a downlink
granularity projection rather than a replacement scorer or selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Sequence


EPSILON = 1.0e-12
ORDINARY_LEVELS = frozenset({"low", "medium"})
HIGH_VALUE_LEVELS = frozenset({"high", "urgent"})


@dataclass(frozen=True)
class OrdinaryNetCandidate:
    action: Any
    action_name: str
    model_name: str
    product: str
    predicted_net_utility: float
    predicted_gross_utility: float
    predicted_energy_cost: float
    payload_mb: float
    total_energy_j: float
    compute_latency_s: float
    predicted_ground_arrival_s: float | None
    hard_safe: bool
    deadline_feasible: bool
    eligible_by_existing_filters: bool = True
    is_defer: bool = False


@dataclass(frozen=True)
class OrdinaryNetDecision:
    final_action: Any
    reference_action: Any
    reason: str
    changed: bool
    selected: OrdinaryNetCandidate
    unconstrained_winner: OrdinaryNetCandidate
    blocked_resource_increasing_winner: bool


@dataclass(frozen=True)
class SameSlotSurplusFacts:
    """Online certificate facts for using otherwise-idle current contact."""

    value_level: str
    reference_is_defer: bool
    current_in_contact: bool
    current_queue_mb: float
    pending_high_value_payload: bool
    candidate_hard_safe: bool
    candidate_deadline_feasible: bool
    candidate_eligible_by_existing_filters: bool
    candidate_is_staged_followup: bool
    candidate_predicted_net_utility: float
    predicted_ground_arrival_s: float | None
    slot_end_s: float
    absolute_deadline_s: float
    conservative_payload_upper_bound_mb: float
    remaining_same_slot_capacity_mb: float
    candidate_model_matches_original_selector: bool


@dataclass(frozen=True)
class SameSlotSurplusDecision:
    eligible: bool
    reason: str


def same_slot_surplus_certificate(
    facts: SameSlotSurplusFacts,
    *,
    epsilon: float = EPSILON,
) -> SameSlotSurplusDecision:
    """Admit only positive-net payloads that cannot spill into a future slot."""

    checks = (
        (str(facts.value_level).lower() in ORDINARY_LEVELS, "not_ordinary"),
        (facts.reference_is_defer, "reference_not_defer"),
        (facts.current_in_contact, "not_in_contact"),
        (float(facts.current_queue_mb) <= epsilon, "queue_not_empty"),
        (not facts.pending_high_value_payload, "pending_high_value_payload"),
        (facts.candidate_hard_safe, "candidate_not_hard_safe"),
        (facts.candidate_deadline_feasible, "candidate_misses_deadline"),
        (
            facts.candidate_eligible_by_existing_filters,
            "candidate_not_in_original_safe_set",
        ),
        (not facts.candidate_is_staged_followup, "staged_followup_forbidden"),
        (
            facts.candidate_model_matches_original_selector,
            "model_differs_from_original_selector",
        ),
        (
            float(facts.candidate_predicted_net_utility) > epsilon,
            "non_positive_predicted_net",
        ),
        (
            facts.predicted_ground_arrival_s is not None,
            "no_predicted_ground_arrival",
        ),
        (
            facts.predicted_ground_arrival_s is not None
            and float(facts.predicted_ground_arrival_s)
            <= min(float(facts.slot_end_s), float(facts.absolute_deadline_s))
            + epsilon,
            "not_delivered_in_same_slot",
        ),
        (
            float(facts.conservative_payload_upper_bound_mb)
            <= float(facts.remaining_same_slot_capacity_mb) + epsilon,
            "conservative_payload_exceeds_same_slot_capacity",
        ),
    )
    for passed, reason in checks:
        if not passed:
            return SameSlotSurplusDecision(False, reason)
    return SameSlotSurplusDecision(True, "same_slot_surplus_certified")


def resource_nonincreasing(
    candidate: OrdinaryNetCandidate,
    reference: OrdinaryNetCandidate,
    *,
    epsilon: float = EPSILON,
) -> bool:
    """Return whether every protected resource dimension is nonincreasing."""

    if candidate.is_defer:
        return True
    return (
        candidate.model_name == reference.model_name
        and candidate.payload_mb <= reference.payload_mb + epsilon
        and candidate.total_energy_j <= reference.total_energy_j + epsilon
        and candidate.compute_latency_s <= reference.compute_latency_s + epsilon
    )


def _eligible(candidate: OrdinaryNetCandidate) -> bool:
    if not candidate.eligible_by_existing_filters or not candidate.hard_safe:
        return False
    if candidate.is_defer:
        return True
    return (
        candidate.deadline_feasible
        and candidate.predicted_ground_arrival_s is not None
        and isfinite(float(candidate.predicted_ground_arrival_s))
        and isfinite(float(candidate.predicted_net_utility))
    )


def _selection_key(
    candidate: OrdinaryNetCandidate,
    reference: OrdinaryNetCandidate,
) -> tuple[float, int, float, float, str]:
    """Deterministic net-first order; exact ties retain the reference."""

    return (
        -float(candidate.predicted_net_utility),
        0 if candidate.action_name == reference.action_name else 1,
        float(candidate.payload_mb),
        float(candidate.total_energy_j),
        str(candidate.action_name),
    )


def choose_ordinary_net_utility_action(
    *,
    value_level: str,
    reference_action: Any,
    candidates: Sequence[OrdinaryNetCandidate],
    require_positive_net: bool = True,
    require_resource_nonincrease: bool = True,
) -> OrdinaryNetDecision:
    """Choose the highest predicted-net ordinary granularity under HV guards."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    reference_name = str(getattr(reference_action, "action_name", reference_action))
    reference = next(
        (row for row in candidates if row.action_name == reference_name),
        None,
    )
    if reference is None:
        raise ValueError("reference action is absent from candidates")

    level = str(value_level or "").strip().lower()
    if level in HIGH_VALUE_LEVELS or level not in ORDINARY_LEVELS:
        return OrdinaryNetDecision(
            final_action=reference.action,
            reference_action=reference.action,
            reason="high_value_bypass" if level in HIGH_VALUE_LEVELS else "unsupported_level_bypass",
            changed=False,
            selected=reference,
            unconstrained_winner=reference,
            blocked_resource_increasing_winner=False,
        )

    feasible = [row for row in candidates if _eligible(row)]
    if reference not in feasible:
        feasible.append(reference)
    unconstrained = sorted(feasible, key=lambda row: _selection_key(row, reference))[0]
    protected = (
        [row for row in feasible if resource_nonincreasing(row, reference)]
        if require_resource_nonincrease
        else feasible
    )
    if reference not in protected:
        protected.append(reference)
    if require_positive_net:
        positive_or_defer = [
            row
            for row in protected
            if row.is_defer or row.predicted_net_utility > EPSILON
        ]
        if positive_or_defer:
            protected = positive_or_defer
    selected = sorted(protected, key=lambda row: _selection_key(row, reference))[0]
    changed = selected.action_name != reference.action_name
    blocked = (
        unconstrained.action_name != selected.action_name
        and not resource_nonincreasing(unconstrained, reference)
    )
    return OrdinaryNetDecision(
        final_action=selected.action,
        reference_action=reference.action,
        reason=(
            "ordinary_net_projected"
            if changed
            else "resource_increasing_winner_blocked"
            if blocked
            else "reference_net_max"
        ),
        changed=changed,
        selected=selected,
        unconstrained_winner=unconstrained,
        blocked_resource_increasing_winner=blocked,
    )
