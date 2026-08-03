"""Pure cross-task HV residual protection and Pareto action filtering.

The module consumes predictions and lifecycle-derived residual facts.  It does
not score PROP actions, update SafeLinUCB, change task values, or relax safety.
All reservations are recomputed from current feasible residual demand; no fixed
quota or bandwidth share is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Any, Iterable, Mapping, Sequence


EPSILON = 1.0e-12


@dataclass(frozen=True)
class ResourceDemand:
    compute_s: float = 0.0
    energy_wh: float = 0.0
    queue_mb: float = 0.0
    contact_bytes: float = 0.0
    thermal_increment: float = 0.0

    def nonnegative(self) -> "ResourceDemand":
        return ResourceDemand(
            compute_s=max(0.0, float(self.compute_s)),
            energy_wh=max(0.0, float(self.energy_wh)),
            queue_mb=max(0.0, float(self.queue_mb)),
            contact_bytes=max(0.0, float(self.contact_bytes)),
            thermal_increment=max(0.0, float(self.thermal_increment)),
        )

    def __add__(self, other: "ResourceDemand") -> "ResourceDemand":
        return ResourceDemand(
            self.compute_s + other.compute_s,
            self.energy_wh + other.energy_wh,
            self.queue_mb + other.queue_mb,
            self.contact_bytes + other.contact_bytes,
            self.thermal_increment + other.thermal_increment,
        )

    def __sub__(self, other: "ResourceDemand") -> "ResourceDemand":
        return ResourceDemand(
            self.compute_s - other.compute_s,
            self.energy_wh - other.energy_wh,
            self.queue_mb - other.queue_mb,
            self.contact_bytes - other.contact_bytes,
            self.thermal_increment - other.thermal_increment,
        )

    def fits(self, capacity: "ResourceDemand", epsilon: float = EPSILON) -> bool:
        demand = self.nonnegative()
        available = capacity.nonnegative()
        return (
            demand.compute_s <= available.compute_s + epsilon
            and demand.energy_wh <= available.energy_wh + epsilon
            and demand.queue_mb <= available.queue_mb + epsilon
            and demand.contact_bytes <= available.contact_bytes + epsilon
            and demand.thermal_increment <= available.thermal_increment + epsilon
        )

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}compute_s": float(self.compute_s),
            f"{prefix}energy_wh": float(self.energy_wh),
            f"{prefix}queue_mb": float(self.queue_mb),
            f"{prefix}contact_bytes": float(self.contact_bytes),
            f"{prefix}thermal_increment": float(self.thermal_increment),
        }


@dataclass(frozen=True)
class ActiveResidual:
    task_id: str
    value_level: str
    effective_task_value: float
    delivered_products: tuple[str, ...]
    residual_required_product: str
    absolute_deadline_s: float
    remaining_slack_s: float
    demand: ResourceDemand
    residual_compute_requirement_s: float
    residual_communication_energy_wh: float
    earliest_generation_completion_s: float | None
    candidate_contact_windows: tuple[str, ...]
    earliest_ground_arrival_s: float | None
    closure_feasible_before_deadline: bool
    predicted_marginal_hv_net_utility: float
    predicted_closure_increment: int
    hard_safe: bool
    original_score: float = 0.0
    blocker_reason: str = ""

    @property
    def protectable(self) -> bool:
        return (
            self.value_level in {"high", "urgent"}
            and self.closure_feasible_before_deadline
            and self.hard_safe
            and self.predicted_marginal_hv_net_utility > 0.0
            and self.remaining_slack_s >= -EPSILON
            and self.earliest_ground_arrival_s is not None
            and bool(self.candidate_contact_windows)
        )


@dataclass(frozen=True)
class ReservationDecision:
    protected_task_ids: tuple[str, ...]
    urgent_task_ids: tuple[str, ...]
    high_task_ids: tuple[str, ...]
    infeasible_urgent_task_ids: tuple[str, ...]
    infeasible_high_task_ids: tuple[str, ...]
    unreserved_feasible_task_ids: tuple[str, ...]
    total: ResourceDemand
    per_task: Mapping[str, ResourceDemand] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceRelease:
    released: ResourceDemand
    increased: ResourceDemand


@dataclass(frozen=True)
class DisplacementResult:
    urgent_task_ids: tuple[str, ...]
    high_task_ids: tuple[str, ...]
    urgent_utility_displacement: float
    hv_utility_displacement: float
    before: ReservationDecision
    after: ReservationDecision

    @property
    def urgent_closure_displacement(self) -> int:
        return len(self.urgent_task_ids)

    @property
    def high_closure_displacement(self) -> int:
        return len(self.high_task_ids)

    @property
    def hv_closure_displacement(self) -> int:
        return self.urgent_closure_displacement + self.high_closure_displacement


@dataclass(frozen=True)
class ParetoActionCandidate:
    action: Any
    action_name: str
    original_score: float
    effective_task_value: float
    remaining_slack_s: float
    task_id: str
    value_level: str
    delta_j_all_net_hat: float
    delta_j_hv_net_hat: float
    delta_n_hv_closure_hat: int
    urgent_closure_displacement: int
    high_closure_displacement: int
    demand: ResourceDemand = ResourceDemand()
    predicted_ground_arrival_s: float | None = None
    hard_safe: bool = True
    deadline_feasible: bool = True
    is_defer: bool = False
    rejection_reason: str = ""

    @property
    def pareto_vector(self) -> tuple[float, float, int]:
        return (
            float(self.delta_j_all_net_hat),
            float(self.delta_j_hv_net_hat),
            int(self.delta_n_hv_closure_hat),
        )

    @property
    def current_hv_closure_increment_hat(self) -> int:
        """Recover the current action's closure increment before displacement."""

        return max(
            0,
            int(self.delta_n_hv_closure_hat)
            + int(self.urgent_closure_displacement)
            + int(self.high_closure_displacement),
        )

    @property
    def delta_n_urgent_closure_hat(self) -> int:
        """Net urgent closure change, including the current urgent task."""

        current_urgent = (
            self.current_hv_closure_increment_hat
            if str(self.value_level).lower() == "urgent"
            else 0
        )
        return current_urgent - int(self.urgent_closure_displacement)


@dataclass(frozen=True)
class ParetoGuardDecision:
    selected: ParetoActionCandidate
    nondominated: tuple[ParetoActionCandidate, ...]
    rejected_by_displacement: tuple[ParetoActionCandidate, ...]
    pareto_dominated: tuple[ParetoActionCandidate, ...]


def _reservation_priority(residual: ActiveResidual) -> tuple[float, float, float, str]:
    return (
        -float(residual.effective_task_value),
        float(residual.remaining_slack_s),
        -float(residual.original_score),
        str(residual.task_id),
    )


def active_residuals(
    rows: Iterable[ActiveResidual],
    *,
    now_s: float,
) -> list[ActiveResidual]:
    """Return the current high/urgent residual set without parallel task state."""

    return sorted(
        [
            row
            for row in rows
            if row.value_level in {"high", "urgent"}
            and bool(row.residual_required_product)
            and row.absolute_deadline_s + EPSILON >= float(now_s)
        ],
        key=lambda row: (
            0 if row.value_level == "urgent" else 1,
            *_reservation_priority(row),
        ),
    )


def dynamic_minimum_reservation(
    rows: Sequence[ActiveResidual],
    capacity: ResourceDemand,
) -> ReservationDecision:
    """Reserve minimum predicted resources, urgent first and then high.

    The function has no configurable percentage, quota, or static bandwidth
    share.  A residual is protected only when it remains feasible, hard-safe,
    positive-net, and fits the currently available multi-resource envelope.
    """

    active = list(rows)
    infeasible_urgent = tuple(
        sorted(row.task_id for row in active if row.value_level == "urgent" and not row.protectable)
    )
    infeasible_high = tuple(
        sorted(row.task_id for row in active if row.value_level == "high" and not row.protectable)
    )
    urgent = sorted(
        [row for row in active if row.value_level == "urgent" and row.protectable],
        key=_reservation_priority,
    )
    high = sorted(
        [row for row in active if row.value_level == "high" and row.protectable],
        key=_reservation_priority,
    )
    total = ResourceDemand()
    per_task: dict[str, ResourceDemand] = {}
    unreserved: list[str] = []
    urgent_ids: list[str] = []
    high_ids: list[str] = []
    for residual in [*urgent, *high]:
        proposed = total + residual.demand.nonnegative()
        if not proposed.fits(capacity):
            unreserved.append(residual.task_id)
            continue
        total = proposed
        per_task[residual.task_id] = residual.demand.nonnegative()
        if residual.value_level == "urgent":
            urgent_ids.append(residual.task_id)
        else:
            high_ids.append(residual.task_id)
    return ReservationDecision(
        protected_task_ids=tuple([*urgent_ids, *high_ids]),
        urgent_task_ids=tuple(urgent_ids),
        high_task_ids=tuple(high_ids),
        infeasible_urgent_task_ids=infeasible_urgent,
        infeasible_high_task_ids=infeasible_high,
        unreserved_feasible_task_ids=tuple(unreserved),
        total=total,
        per_task=per_task,
    )


def released_resources(previous: ResourceDemand, current: ResourceDemand) -> ResourceRelease:
    """Return immediately released and newly required reservation volumes."""

    delta = previous - current
    increase = current - previous
    return ResourceRelease(released=delta.nonnegative(), increased=increase.nonnegative())


def shadow_displacement(
    rows: Sequence[ActiveResidual],
    *,
    capacity_before: ResourceDemand,
    action_demand: ResourceDemand,
    exclude_task_id: str = "",
) -> DisplacementResult:
    """Lightweight action-before/action-after reservation feasibility check."""

    considered = [row for row in rows if row.task_id != exclude_task_id]
    before = dynamic_minimum_reservation(considered, capacity_before)
    after_capacity = (capacity_before - action_demand.nonnegative()).nonnegative()
    after = dynamic_minimum_reservation(considered, after_capacity)
    lost = set(before.protected_task_ids) - set(after.protected_task_ids)
    by_id = {row.task_id: row for row in considered}
    urgent_ids = tuple(sorted(task_id for task_id in lost if by_id[task_id].value_level == "urgent"))
    high_ids = tuple(sorted(task_id for task_id in lost if by_id[task_id].value_level == "high"))
    urgent_loss = sum(by_id[task_id].predicted_marginal_hv_net_utility for task_id in urgent_ids)
    hv_loss = sum(by_id[task_id].predicted_marginal_hv_net_utility for task_id in lost)
    return DisplacementResult(
        urgent_task_ids=urgent_ids,
        high_task_ids=high_ids,
        urgent_utility_displacement=float(urgent_loss),
        hv_utility_displacement=float(hv_loss),
        before=before,
        after=after,
    )


def pareto_dominates(left: ParetoActionCandidate, right: ParetoActionCandidate) -> bool:
    """True when left is no worse on all three metrics and better on one."""

    a = left.pareto_vector
    b = right.pareto_vector
    return all(x >= y - EPSILON for x, y in zip(a, b)) and any(x > y + EPSILON for x, y in zip(a, b))


def pareto_nondominated(
    candidates: Sequence[ParetoActionCandidate],
) -> tuple[list[ParetoActionCandidate], list[ParetoActionCandidate]]:
    nondominated: list[ParetoActionCandidate] = []
    dominated: list[ParetoActionCandidate] = []
    for candidate in candidates:
        if any(
            other is not candidate and pareto_dominates(other, candidate)
            for other in candidates
        ):
            dominated.append(candidate)
        else:
            nondominated.append(candidate)
    return nondominated, dominated


def _displacement_allowed(candidate: ParetoActionCandidate) -> bool:
    level = str(candidate.value_level).lower()
    if candidate.is_defer:
        return True
    if level in {"low", "medium"}:
        return (
            candidate.urgent_closure_displacement == 0
            and candidate.high_closure_displacement == 0
        )
    if level in {"high", "urgent"}:
        return (
            candidate.delta_n_urgent_closure_hat >= 0
            and
            candidate.delta_n_hv_closure_hat >= 0
            and candidate.delta_j_hv_net_hat >= -EPSILON
        )
    return candidate.urgent_closure_displacement == 0


def requires_cross_task_guard(candidate: ParetoActionCandidate) -> bool:
    """Return whether the corrected REQALIGN action needs guard intervention."""

    return (
        not candidate.is_defer
        and (
            int(candidate.urgent_closure_displacement) > 0
            or int(candidate.high_closure_displacement) > 0
        )
    )


def select_pareto_guarded_action(
    candidates: Sequence[ParetoActionCandidate],
) -> ParetoGuardDecision:
    """Apply displacement rejection, Pareto filtering, then deterministic order."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    eligible = [
        candidate
        for candidate in candidates
        if candidate.hard_safe
        and (candidate.is_defer or candidate.deadline_feasible)
        and _displacement_allowed(candidate)
    ]
    rejected = [candidate for candidate in candidates if candidate not in eligible]
    if not eligible:
        defer = next((candidate for candidate in candidates if candidate.is_defer), None)
        if defer is None:
            raise ValueError("no displacement-safe candidate and no DEFER")
        eligible = [defer]
    nondominated, dominated = pareto_nondominated(eligible)
    selected = sorted(
        nondominated,
        key=lambda candidate: (
            int(candidate.urgent_closure_displacement),
            int(candidate.urgent_closure_displacement + candidate.high_closure_displacement),
            -float(candidate.delta_j_hv_net_hat),
            -int(candidate.delta_n_hv_closure_hat),
            -float(candidate.delta_j_all_net_hat),
            -float(candidate.original_score),
            -float(candidate.effective_task_value),
            float(candidate.remaining_slack_s if candidate.remaining_slack_s is not None else inf),
            str(candidate.task_id),
            str(candidate.action_name),
        ),
    )[0]
    return ParetoGuardDecision(
        selected=selected,
        nondominated=tuple(nondominated),
        rejected_by_displacement=tuple(rejected),
        pareto_dominated=tuple(dominated),
    )


def promotion_status(
    *,
    old_prop: Mapping[str, float],
    corrected_reqalign: Mapping[str, float],
    guarded: Mapping[str, float],
    safety_violations: int,
    ordinary_logic_unchanged: bool,
) -> str:
    """Apply the preregistered three-level promotion rule exactly."""

    metrics = ("J_all_net", "J_HV_net", "HV_closure_count")
    strict = all(
        float(guarded[key]) > float(old_prop[key]) + EPSILON
        and float(guarded[key]) > float(corrected_reqalign[key]) + EPSILON
        for key in metrics
    )
    urgent_ok = (
        float(guarded["urgent_closure_count"])
        >= max(
            float(old_prop["urgent_closure_count"]),
            float(corrected_reqalign["urgent_closure_count"]),
        )
        - EPSILON
    )
    safe = int(safety_violations) == 0 and bool(ordinary_logic_unchanged)
    if strict and urgent_ok and safe:
        return "HV_URGENT_GUARD_STRICT_THREE_METRIC_IMPROVEMENT"
    nondegrading = all(
        float(guarded[key]) >= float(old_prop[key]) - EPSILON
        and float(guarded[key]) >= float(corrected_reqalign[key]) - EPSILON
        for key in metrics
    )
    strictly_better_somewhere = any(
        float(guarded[key]) > float(old_prop[key]) + EPSILON
        or float(guarded[key]) > float(corrected_reqalign[key]) + EPSILON
        for key in metrics
    )
    if nondegrading and strictly_better_somewhere and urgent_ok and safe:
        return "HV_URGENT_GUARD_PARETO_NONDEGRADING_CANDIDATE"
    return "HV_URGENT_GUARD_NO_PROMOTION"
