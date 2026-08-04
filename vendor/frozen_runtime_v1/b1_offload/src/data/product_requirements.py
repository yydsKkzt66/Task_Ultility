"""Mission-prior product requirement assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.types import TaskSample


ALLOWED_PRODUCT_REQUIREMENTS = {"alert", "verification", "context", "evidence"}


@dataclass(frozen=True)
class ProductRequirementV2Result:
    """Diagnostic-only v2 requirement assignment result.

    The v2 result deliberately separates ideal mission information need from
    the payload plan that may be feasible under resource pressure.
    """

    ideal_product_requirement: str
    requirement_scores: dict[str, float]
    requirement_source: str
    diagnostic_reason: str
    missing_fields: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProductRequirementV21Result:
    """Diagnostic-only v2.1 mission-semantics requirement assignment result."""

    ideal_product_requirement: str
    requirement_scores: dict[str, float]
    requirement_source: str
    diagnostic_reason: str
    initial_alert_required: bool = False
    allowed_requirements: tuple[str, ...] = field(default_factory=tuple)
    missing_fields: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PayloadPlanRecommendation:
    """Diagnostic-only payload plan recommendation."""

    recommended_payload_plan: str
    recommendation_source: str
    diagnostic_reason: str
    missing_fields: tuple[str, ...] = field(default_factory=tuple)


def assign_product_requirement(
    *,
    task_value: float,
    deadline_s: float,
    uncertainty: float,
    high_value_flag: bool = False,
    emergency_flag: bool = False,
    task_class: str = "routine",
) -> tuple[str, str]:
    """Assign requirement from mission priors only.

    This deliberately does not inspect labels, object counts, bbox statistics, or
    detector ground truth.
    """

    if emergency_flag or task_class == "emergency" or deadline_s <= 45.0:
        return "alert", "mission_prior_proxy"
    if high_value_flag and (task_value >= 0.82 or uncertainty >= 0.55):
        return "evidence", "mission_prior_proxy"
    if task_class == "high_value" or high_value_flag or task_value >= 0.75:
        return "context", "mission_prior_proxy"
    return "verification", "mission_prior_proxy"


def _clip01(value: float | int | bool | None, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _optional_float(
    value: float | int | None,
    *,
    field_name: str,
    default: float,
    missing_fields: list[str],
) -> float:
    if value is None:
        missing_fields.append(field_name)
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        missing_fields.append(field_name)
        return default


def _moderate_deadline_score(urgency: float) -> float:
    """Peak at medium urgency and fall off for very low or extreme urgency."""

    urgency = _clip01(urgency)
    return _clip01(1.0 - abs(urgency - 0.5) / 0.5)


def assign_product_requirement_v2(
    *,
    task_value: float,
    deadline_s: float,
    uncertainty: float,
    remaining_deadline_s: float | None = None,
    high_value_flag: bool = False,
    emergency_flag: bool = False,
    task_class: str = "routine",
    mission_source: str | None = None,
    scene_prior: float | None = None,
    product_use_case: str | None = None,
    queue_mb: float | None = None,
    queue_max_mb: float | None = None,
    known_contact_capacity_before_deadline: float | None = None,
    required_capacity_for_roi_context: float | None = None,
    battery_slack: float | None = None,
    thermal_slack: float | None = None,
    queue_slack: float | None = None,
    contact_slack: float | None = None,
    deadline_slack: float | None = None,
    high_value_threshold: float = 0.75,
    epsilon: float = 1.0e-9,
) -> ProductRequirementV2Result:
    """Assign ideal mission-level product requirement using decision-time fields.

    This function is not called by the current task stream unless explicitly
    wired by a separate default-off experiment. It does not inspect labels,
    object counts, bbox statistics, detector outputs, future delivery, or
    realized future utility.
    """

    missing_fields: list[str] = []
    value = _clip01(task_value)
    uncertainty_score = _clip01(uncertainty)
    deadline = max(float(deadline_s), epsilon)
    remaining_deadline = _optional_float(
        remaining_deadline_s,
        field_name="remaining_deadline_s",
        default=deadline,
        missing_fields=missing_fields,
    )
    urgency = _clip01(1.0 - remaining_deadline / max(deadline, epsilon))
    emergency = 1.0 if emergency_flag or str(task_class).lower() == "emergency" else 0.0
    high_value = 1.0 if high_value_flag or value >= high_value_threshold else 0.0
    emergency_or_high_value = max(emergency, high_value)

    queue_capacity = _optional_float(
        queue_max_mb,
        field_name="queue_max_mb",
        default=1.0,
        missing_fields=missing_fields,
    )
    queue_used = _optional_float(queue_mb, field_name="queue_mb", default=0.0, missing_fields=missing_fields)
    queue_pressure = _clip01(queue_used / max(queue_capacity, epsilon))
    queue_slack_score = _clip01(
        _optional_float(
            queue_slack,
            field_name="queue_slack",
            default=1.0 - queue_pressure,
            missing_fields=missing_fields,
        )
    )

    contact_capacity = _optional_float(
        known_contact_capacity_before_deadline,
        field_name="known_contact_capacity_before_deadline",
        default=0.0,
        missing_fields=missing_fields,
    )
    required_capacity = _optional_float(
        required_capacity_for_roi_context,
        field_name="required_capacity_for_roi_context",
        default=1.0,
        missing_fields=missing_fields,
    )
    contact_pressure = _clip01(1.0 - contact_capacity / max(required_capacity, epsilon))
    contact_slack_score = _clip01(
        _optional_float(
            contact_slack,
            field_name="contact_slack",
            default=1.0 - contact_pressure,
            missing_fields=missing_fields,
        )
    )

    battery_slack_score = _clip01(
        _optional_float(battery_slack, field_name="battery_slack", default=1.0, missing_fields=missing_fields)
    )
    thermal_slack_score = _clip01(
        _optional_float(thermal_slack, field_name="thermal_slack", default=1.0, missing_fields=missing_fields)
    )
    deadline_slack_score = _clip01(
        _optional_float(deadline_slack, field_name="deadline_slack", default=1.0 - urgency, missing_fields=missing_fields)
    )
    enough_slack = min(
        battery_slack_score,
        thermal_slack_score,
        queue_slack_score,
        contact_slack_score,
        deadline_slack_score,
    )

    if scene_prior is None:
        missing_fields.append("scene_prior")
        scene_context_need = 0.5
    else:
        scene_context_need = _clip01(scene_prior, default=0.5)
    if mission_source is None:
        missing_fields.append("mission_source")
    if product_use_case is None:
        missing_fields.append("product_use_case")

    normal_priority = 1.0 - high_value
    moderate_deadline = _moderate_deadline_score(urgency)
    scores = {
        "alert": _clip01(0.35 * urgency + 0.25 * emergency + 0.20 * contact_pressure + 0.20 * queue_pressure),
        "verification": _clip01(
            0.35 * uncertainty_score + 0.25 * value + 0.20 * normal_priority + 0.20 * moderate_deadline
        ),
        "context": _clip01(0.35 * value + 0.25 * uncertainty_score + 0.20 * scene_context_need + 0.20 * enough_slack),
        "evidence": _clip01(
            0.40 * value + 0.25 * emergency_or_high_value + 0.20 * uncertainty_score + 0.15 * enough_slack
        ),
    }

    ideal = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
    reasons = [f"score_argmax={ideal}"]

    task_class_key = str(task_class).strip().lower()
    evidence_guard = (
        emergency_or_high_value >= 1.0
        and value >= max(0.82, high_value_threshold)
        and uncertainty_score >= 0.50
    )
    context_guard = (high_value >= 1.0 or task_class_key == "high_value") and value >= high_value_threshold
    if evidence_guard and ideal == "alert":
        ideal = "evidence"
        reasons.append("mission_level_guard_preserved_evidence_need_under_pressure")
    elif context_guard and ideal == "alert":
        ideal = "context"
        reasons.append("mission_level_guard_preserved_context_need_under_pressure")

    if missing_fields:
        reasons.append("safe_defaults_used_for=" + ";".join(sorted(set(missing_fields))))

    return ProductRequirementV2Result(
        ideal_product_requirement=ideal,
        requirement_scores=scores,
        requirement_source="mission_prior_product_requirement_v2_shadow",
        diagnostic_reason=" | ".join(reasons),
        missing_fields=tuple(sorted(set(missing_fields))),
    )


def _contains_any(value: str | None, needles: set[str]) -> bool:
    haystack = str(value or "").strip().lower()
    return any(needle in haystack for needle in needles)


def assign_product_requirement_v2_1(
    *,
    task_value: float,
    uncertainty: float,
    emergency_flag: bool = False,
    high_value_flag: bool = False,
    task_class: str = "routine",
    mission_source: str | None = None,
    scene_context_need: float | None = None,
    product_use_case: str | None = None,
    timeliness_class: str | None = None,
    high_value_threshold: float = 0.75,
) -> ProductRequirementV21Result:
    """Assign ideal product requirement using mission/task-side semantics only.

    v2.1 intentionally excludes queue/contact/battery/thermal/payload-size and
    future-feasibility fields from ideal requirement scoring. Resource pressure
    belongs in the shadow payload-plan recommendation, not here.
    """

    missing_fields: list[str] = []
    value = _clip01(task_value)
    uncertainty_score = _clip01(uncertainty)
    class_key = str(task_class or "routine").strip().lower()

    if mission_source is None:
        missing_fields.append("mission_source")
    if product_use_case is None:
        missing_fields.append("product_use_case")
    if timeliness_class is None:
        missing_fields.append("timeliness_class")
    if scene_context_need is None:
        missing_fields.append("scene_context_need")
        context_need = 0.5
    else:
        context_need = _clip01(scene_context_need, default=0.5)

    emergency = bool(emergency_flag) or class_key == "emergency"
    high_value = bool(high_value_flag) or class_key == "high_value" or value >= high_value_threshold
    low_value = value < 0.45
    medium_value = 0.45 <= value < high_value_threshold
    high_uncertainty = uncertainty_score >= 0.60
    low_uncertainty = uncertainty_score < 0.30
    medium_uncertainty = 0.30 <= uncertainty_score < 0.60

    explicit_evidence = _contains_any(
        product_use_case,
        {"evidence", "closure", "forensic", "audit", "proof", "legal", "incident"},
    )
    explicit_context = context_need >= 0.70 or _contains_any(
        product_use_case,
        {"context", "scene", "maritime", "situational", "review"},
    )
    alert_use_case = _contains_any(product_use_case, {"alert", "tip", "notify", "heartbeat"})
    urgent_timeliness = _contains_any(
        timeliness_class,
        {"urgent", "immediate", "short", "tight", "critical", "asap"},
    )
    initial_alert_required = emergency or urgent_timeliness or alert_use_case

    scores = {
        "alert": _clip01(0.45 * (1.0 - value) + 0.25 * (1.0 - uncertainty_score) + 0.30 * float(initial_alert_required)),
        "verification": _clip01(
            0.45 * uncertainty_score
            + 0.25 * (1.0 - value)
            + 0.20 * float(medium_value or low_value)
            + 0.10 * float(emergency)
        ),
        "context": _clip01(0.40 * value + 0.25 * uncertainty_score + 0.25 * context_need + 0.10 * float(high_value)),
        "evidence": _clip01(
            0.40 * value
            + 0.30 * uncertainty_score
            + 0.20 * float(emergency and high_value)
            + 0.10 * float(explicit_evidence)
        ),
    }

    reasons: list[str] = []
    if emergency and high_value and high_uncertainty:
        allowed = {"evidence"}
        reasons.append("gate_emergency_high_value_high_uncertainty")
    elif emergency and high_value:
        allowed = {"context"}
        reasons.append("gate_emergency_high_value")
    elif emergency:
        allowed = {"context"} if (medium_value and high_uncertainty or explicit_context) else {"verification"}
        reasons.append("gate_emergency_alert_required_verification_need")
    elif high_value and high_uncertainty:
        allowed = {"evidence"} if (explicit_evidence or value >= 0.85 or context_need >= 0.80) else {"context"}
        reasons.append("gate_high_value_high_uncertainty")
    elif high_value and low_uncertainty:
        allowed = {"context"} if (explicit_context or value >= 0.82) else {"verification"}
        reasons.append("gate_high_value_low_uncertainty")
    elif high_value and medium_uncertainty:
        allowed = {"context"}
        reasons.append("gate_high_value_medium_uncertainty")
    elif medium_value and high_uncertainty:
        allowed = {"context"} if explicit_context else {"verification"}
        reasons.append("gate_medium_value_high_uncertainty")
    elif medium_value:
        allowed = {"context"} if explicit_context and uncertainty_score >= 0.45 else {"verification"}
        reasons.append("gate_medium_value")
    elif low_value and high_uncertainty:
        allowed = {"verification"}
        reasons.append("gate_low_value_high_uncertainty")
    elif low_value and low_uncertainty:
        allowed = {"alert"} if (alert_use_case or value < 0.25 and uncertainty_score < 0.20) else {"verification"}
        reasons.append("gate_low_value_low_uncertainty")
    else:
        allowed = {"verification"}
        reasons.append("gate_low_value_medium_uncertainty")

    ideal = max(allowed, key=lambda requirement: (scores[requirement], RICHNESS_ORDER[requirement]))
    reasons.append(f"tie_break_within_allowed={ideal}")
    if initial_alert_required and ideal != "alert":
        reasons.append("initial_alert_required_separate_from_ideal_requirement")
    if missing_fields:
        reasons.append("safe_defaults_used_for=" + ";".join(sorted(set(missing_fields))))

    return ProductRequirementV21Result(
        ideal_product_requirement=ideal,
        requirement_scores=scores,
        requirement_source="mission_prior_product_requirement_v2_1_shadow",
        diagnostic_reason=" | ".join(reasons),
        initial_alert_required=initial_alert_required,
        allowed_requirements=tuple(sorted(allowed, key=lambda item: RICHNESS_ORDER[item])),
        missing_fields=tuple(sorted(set(missing_fields))),
    )


RICHNESS_ORDER = {"alert": 0, "verification": 1, "context": 2, "evidence": 3}


def _infer_feasible(
    explicit: bool | None,
    *,
    field_name: str,
    missing_fields: list[str],
    default: bool,
) -> bool:
    if explicit is None:
        missing_fields.append(field_name)
        return default
    return bool(explicit)


def recommend_payload_plan_from_requirement(
    *,
    ideal_product_requirement: str,
    metadata_feasible: bool | None = None,
    roi_feasible: bool | None = None,
    roi_context_feasible: bool | None = None,
    metadata_then_roi_feasible: bool | None = None,
    metadata_then_roi_context_feasible: bool | None = None,
) -> PayloadPlanRecommendation:
    """Recommend a payload plan in shadow mode without forcing router behavior."""

    missing_fields: list[str] = []
    metadata_ok = _infer_feasible(
        metadata_feasible,
        field_name="metadata_feasible",
        missing_fields=missing_fields,
        default=True,
    )
    roi_ok = _infer_feasible(roi_feasible, field_name="roi_feasible", missing_fields=missing_fields, default=True)
    roi_context_ok = _infer_feasible(
        roi_context_feasible,
        field_name="roi_context_feasible",
        missing_fields=missing_fields,
        default=True,
    )
    staged_roi_ok = _infer_feasible(
        metadata_then_roi_feasible,
        field_name="metadata_then_roi_feasible",
        missing_fields=missing_fields,
        default=metadata_ok and roi_ok,
    )
    staged_context_ok = _infer_feasible(
        metadata_then_roi_context_feasible,
        field_name="metadata_then_roi_context_feasible",
        missing_fields=missing_fields,
        default=metadata_ok and roi_context_ok,
    )

    requirement = str(ideal_product_requirement).strip().lower()
    if requirement == "alert":
        plan = "metadata" if metadata_ok else "defer"
    elif requirement == "verification":
        if roi_ok:
            plan = "roi"
        elif staged_roi_ok:
            plan = "metadata_then_roi"
        elif metadata_ok:
            plan = "metadata"
        else:
            plan = "defer"
    elif requirement == "context":
        if roi_context_ok:
            plan = "roi_context"
        elif staged_context_ok:
            plan = "metadata_then_roi_context"
        elif roi_ok:
            plan = "roi"
        elif metadata_ok:
            plan = "metadata"
        else:
            plan = "defer"
    elif requirement == "evidence":
        if roi_context_ok:
            plan = "roi_context"
        elif staged_context_ok:
            plan = "metadata_then_roi_context"
        elif metadata_ok:
            plan = "metadata"
        else:
            plan = "defer"
    else:
        plan = "defer"

    reason = f"shadow_mapping_for_{requirement}"
    if missing_fields:
        reason += " | safe_defaults_used_for=" + ";".join(sorted(set(missing_fields)))
    return PayloadPlanRecommendation(
        recommended_payload_plan=plan,
        recommendation_source="product_requirement_v2_shadow_payload_plan",
        diagnostic_reason=reason,
        missing_fields=tuple(sorted(set(missing_fields))),
    )


def assign_requirement_for_task(task: TaskSample) -> tuple[str, str]:
    return assign_product_requirement(
        task_value=task.task_value,
        deadline_s=task.deadline_s,
        uncertainty=task.uncertainty,
        high_value_flag=task.high_value_flag,
        emergency_flag=task.emergency_flag,
        task_class=task.task_class,
    )
