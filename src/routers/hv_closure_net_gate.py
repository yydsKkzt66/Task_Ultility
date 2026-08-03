"""Default-off public-accounting gate for high-value closure candidates.

The gate is deliberately narrower than a new scorer.  It never chooses an
action and never changes the frozen MYOPIC ranking.  It only allows an
otherwise score-blocked high/urgent *closing* action to enter corrected
REQALIGN when the action is already hard-safe, predicted to reach the ground
before the deadline, and has positive predicted net utility under the public
ground-accounting formula.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping


EPSILON = 1.0e-12
HIGH_VALUE_LEVELS = frozenset({"high", "urgent"})


@dataclass(frozen=True)
class HVClosureNetGateFacts:
    enabled: bool
    value_level: str
    closure_satisfying: bool
    original_score_above_defer: bool
    hard_safe: bool
    deadline_feasible: bool
    contact_found: bool
    eligible_by_existing_filters: bool
    capture_time_s: float
    deadline_duration_s: float
    predicted_ground_arrival_s: float | None
    effective_task_value: float
    information_sufficiency: float
    predicted_quality: float
    public_energy_cost: float


@dataclass(frozen=True)
class HVClosureNetGateDecision:
    allowed: bool
    rescued_negative_original_score: bool
    reason: str
    predicted_timeliness: float
    predicted_public_gross_utility: float
    predicted_public_net_utility: float


def evaluate_hv_closure_net_gate(
    facts: HVClosureNetGateFacts,
    *,
    epsilon: float = EPSILON,
) -> HVClosureNetGateDecision:
    """Evaluate the narrow post-selector admission certificate."""

    def decision(
        allowed: bool,
        reason: str,
        *,
        timeliness: float = 0.0,
        gross: float = 0.0,
        net: float = 0.0,
    ) -> HVClosureNetGateDecision:
        return HVClosureNetGateDecision(
            allowed=bool(allowed),
            rescued_negative_original_score=bool(
                allowed and not facts.original_score_above_defer
            ),
            reason=reason,
            predicted_timeliness=float(timeliness),
            predicted_public_gross_utility=float(gross),
            predicted_public_net_utility=float(net),
        )

    if not facts.enabled:
        return decision(facts.original_score_above_defer, "disabled_exact_bypass")
    if str(facts.value_level).strip().lower() not in HIGH_VALUE_LEVELS:
        return decision(facts.original_score_above_defer, "ordinary_exact_bypass")
    if facts.original_score_above_defer:
        return decision(True, "existing_score_gate_passed")
    if not facts.closure_satisfying:
        return decision(False, "not_a_closure_action")
    if not facts.eligible_by_existing_filters:
        return decision(False, "outside_original_safe_candidate_set")
    if not facts.hard_safe:
        return decision(False, "hard_safety_blocked")
    if not facts.contact_found or facts.predicted_ground_arrival_s is None:
        return decision(False, "no_predicted_ground_arrival")
    if not facts.deadline_feasible:
        return decision(False, "deadline_blocked")

    numeric = (
        facts.capture_time_s,
        facts.deadline_duration_s,
        facts.predicted_ground_arrival_s,
        facts.effective_task_value,
        facts.information_sufficiency,
        facts.predicted_quality,
        facts.public_energy_cost,
    )
    if not all(isfinite(float(value)) for value in numeric):
        return decision(False, "nonfinite_prediction")
    if float(facts.deadline_duration_s) <= epsilon:
        return decision(False, "nonpositive_deadline")

    latency = max(
        0.0,
        float(facts.predicted_ground_arrival_s) - float(facts.capture_time_s),
    )
    timeliness = max(
        0.0,
        min(1.0, 1.0 - latency / float(facts.deadline_duration_s)),
    )
    information = max(0.0, float(facts.information_sufficiency))
    quality = max(0.0, min(1.0, float(facts.predicted_quality)))
    gross = (
        max(0.0, float(facts.effective_task_value))
        * information
        * quality
        * timeliness
    )
    net = gross - max(0.0, float(facts.public_energy_cost))
    if net <= epsilon:
        return decision(
            False,
            "nonpositive_public_closure_net",
            timeliness=timeliness,
            gross=gross,
            net=net,
        )
    return decision(
        True,
        "positive_public_closure_net_rescue",
        timeliness=timeliness,
        gross=gross,
        net=net,
    )


class HVClosureNetGateProjectionFacade:
    """Projection-module facade that changes only candidate gate admission."""

    def __init__(
        self,
        base_projection: Any,
        delegate: Any,
        config: Mapping[str, Any],
    ) -> None:
        self.base_projection = base_projection
        self.delegate = delegate
        self.config = dict(config)
        self.current_task: Any = None
        self.capture_time_s = 0.0
        self.candidate_audit_rows: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_projection, name)

    def bind_step(self, task: Any, *, slot_seconds: float) -> None:
        self.current_task = task
        self.capture_time_s = (
            float(getattr(task, "arrival_slot", 0) or 0) * float(slot_seconds)
            if task is not None
            else 0.0
        )

    def ProjectionCandidate(self, **kwargs: Any) -> Any:  # noqa: N802
        """Create the unchanged candidate unless the certificate rescues it."""

        task = self.current_task
        action_name = str(kwargs.get("action_name", ""))
        components = dict(
            getattr(self.delegate, "last_all_score_components", {}).get(
                action_name, {}
            )
        )
        original_pass = bool(kwargs.get("score_above_defer", False))
        value_level = "" if task is None else str(
            getattr(task, "value_level", "")
            or (getattr(task, "metadata", {}) or {}).get("value_level", "")
        ).lower()
        required_product = "" if task is None else str(
            getattr(task, "product_requirement", "") or ""
        )
        facts = HVClosureNetGateFacts(
            enabled=bool(self.config.get("enabled", False)),
            value_level=value_level,
            closure_satisfying=self.base_projection.product_satisfies(
                kwargs.get("product", ""), required_product
            ),
            original_score_above_defer=original_pass,
            hard_safe=bool(kwargs.get("hard_safe", False)),
            deadline_feasible=bool(kwargs.get("deadline_feasible", False)),
            contact_found=bool(kwargs.get("contact_found", False)),
            eligible_by_existing_filters=bool(
                kwargs.get("eligible_by_existing_filters", False)
            ),
            capture_time_s=self.capture_time_s,
            deadline_duration_s=(
                0.0 if task is None else float(getattr(task, "deadline_s", 0.0) or 0.0)
            ),
            predicted_ground_arrival_s=kwargs.get("predicted_ground_arrival_s"),
            effective_task_value=float(components.get("effective_task_value", 0.0) or 0.0),
            information_sufficiency=float(
                components.get("information_sufficiency", 0.0) or 0.0
            ),
            predicted_quality=float(
                components.get("profile_result_quality", 0.0) or 0.0
            ),
            public_energy_cost=float(
                components.get("public_immediate_controllable_cost", 0.0) or 0.0
            ),
        )
        gate = evaluate_hv_closure_net_gate(facts)
        kwargs["score_above_defer"] = bool(original_pass or gate.allowed)
        candidate = self.base_projection.ProjectionCandidate(**kwargs)
        self.candidate_audit_rows.append(
            {
                "task_id": "" if task is None else str(getattr(task, "task_id", "")),
                "value_level": value_level,
                "required_product": required_product,
                "action_name": action_name,
                "candidate_product": str(kwargs.get("product", "")),
                "original_score": kwargs.get("original_score", ""),
                "original_score_above_defer": int(original_pass),
                "gate_score_above_defer": int(kwargs["score_above_defer"]),
                **asdict(gate),
            }
        )
        return candidate
