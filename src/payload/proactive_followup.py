"""Phase 6 ERIV-guided proactive residual follow-up.

This module is intentionally pure.  It only reads decision-time task,
product-lifecycle, resource, queue, contact, and z_k-like guidance fields.
It does not inspect SARFish labels, detector correctness, future delivery
outcomes, future utility, or future passive continue-request values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.payload.continue_request import RESIDUAL_PRODUCT_TYPES, _as_set, _queued_residual_duplicate
from src.payload.information_sufficiency import delta_information_utility, normalize_product_requirement
from src.payload.roi_context_gate import evaluate_roi_context_gate


DEFAULT_FULL_METHOD_ALIASES = {
    "proposed_lyapunov_rule_mode_safe_bandit",
    "proposed_calibrated_payload_plus_z_guided",
    "proposed_payload_repaired_combined",
}

DEFAULT_CARRYOVER_CONFIG = {
    "enabled": False,
    "re_evaluate_once_per_slot": True,
    "max_candidates_per_slot": 2,
    "max_candidate_age_s": 600,
    "require_metadata_delivered": True,
    "require_pending_residual": True,
    "require_positive_residual_utility": True,
    "remove_on_expired": True,
    "remove_on_closed": True,
    "remove_on_residual_delivered": True,
    "remove_on_duplicate_enqueued": False,
}


DEFAULT_SAME_SLOT_CARRYOVER_CONFIG = {
    "enabled": False,
    "only_after_lifecycle_update": True,
    "reuse_existing_guards": True,
    "max_same_slot_eval_per_slot": 1,
    "enable_queue_fallback_on_insufficient_capacity": False,
    "enforce_no_reward_change": True,
    "enforce_no_queue_policy_change": True,
    "strict_mode": True,
}


@dataclass(frozen=True)
class ERIVBreakdown:
    eriv_score: float
    residual_delta_u_I: float
    task_value: float
    model_quality_factor: float
    timeliness_factor: float
    contact_opportunity_score: float
    z_k_followup_gate: float
    payload_size_mb: float
    normalized_utility_per_mb: float
    information_density: float = 0.0
    value_density_per_mb: float = 0.0
    fields_used: tuple[str, ...] = field(default_factory=tuple)
    blockers: tuple[str, ...] = field(default_factory=tuple)
    leakage_safe_flag: bool = True


@dataclass(frozen=True)
class ProactiveFollowupDecision:
    proactive_flag: bool
    residual_product_type: str = ""
    eriv_score: float = 0.0
    threshold: float = 0.0
    reason_code: str = "not_evaluated"
    blockers: tuple[str, ...] = field(default_factory=tuple)
    fields_used: tuple[str, ...] = field(default_factory=tuple)
    leakage_safe_flag: bool = True
    eriv_breakdown: ERIVBreakdown | None = None


def phase6_proactive_config(simulator_cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    simulator_cfg = simulator_cfg or {}
    progressive = simulator_cfg.get("progressive_payload") if isinstance(simulator_cfg, Mapping) else None
    if isinstance(progressive, Mapping):
        nested = progressive.get("phase6_proactive_followup")
        if isinstance(nested, Mapping):
            return nested
    nested = simulator_cfg.get("phase6_proactive_followup") if isinstance(simulator_cfg, Mapping) else None
    if isinstance(nested, Mapping):
        return nested
    return simulator_cfg


def phase6_residual_candidate_carryover_config(simulator_cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    simulator_cfg = simulator_cfg or {}
    cfg = dict(DEFAULT_CARRYOVER_CONFIG)
    progressive = simulator_cfg.get("progressive_payload") if isinstance(simulator_cfg, Mapping) else None
    nested = None
    if isinstance(progressive, Mapping):
        nested = progressive.get("phase6_residual_candidate_carryover")
    if not isinstance(nested, Mapping) and isinstance(simulator_cfg, Mapping):
        nested = simulator_cfg.get("phase6_residual_candidate_carryover")
    if isinstance(nested, Mapping):
        cfg.update(nested)
    return cfg


def phase6_same_slot_carryover_config(simulator_cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    simulator_cfg = simulator_cfg or {}
    cfg = dict(DEFAULT_SAME_SLOT_CARRYOVER_CONFIG)
    progressive = simulator_cfg.get("progressive_payload") if isinstance(simulator_cfg, Mapping) else None
    nested = None
    if isinstance(progressive, Mapping):
        nested = progressive.get("phase6_same_slot_carryover")
    if not isinstance(nested, Mapping) and isinstance(simulator_cfg, Mapping):
        nested = simulator_cfg.get("phase6_same_slot_carryover")
    if isinstance(nested, Mapping):
        cfg.update(nested)
    return cfg


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _residual_product(task_state: Any) -> str:
    pending_products = _as_set(_get(task_state, "pending_products", set()))
    residual_pending = sorted(product for product in pending_products if product in RESIDUAL_PRODUCT_TYPES)
    return residual_pending[0] if len(residual_pending) == 1 else ""


def _full_method_allowed(system_state: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    variant = str(system_state.get("variant", system_state.get("router_name", ""))).strip()
    aliases = set(DEFAULT_FULL_METHOD_ALIASES)
    configured = cfg.get("full_method_aliases", ())
    if isinstance(configured, str):
        aliases.update(item.strip() for item in configured.split(",") if item.strip())
    else:
        try:
            aliases.update(str(item) for item in configured)
        except TypeError:
            pass
    if variant in aliases:
        return True
    if variant.startswith("proposed_") and _as_bool(system_state.get("use_lyapunov_pricing")) and _as_bool(
        system_state.get("use_mode_controller")
    ):
        return True
    return False


def _contact_score(system_state: Mapping[str, Any]) -> float:
    if "contact_opportunity_score" in system_state:
        return _clamp(_as_float(system_state.get("contact_opportunity_score"), 0.0))
    if _as_bool(system_state.get("contact_available"), False):
        remaining = _as_float(system_state.get("contact_remaining_window_s"), 60.0)
        return _clamp(remaining / 60.0)
    return 0.0


def _z_k_followup_gate(system_state: Mapping[str, Any]) -> float:
    if "z_k_guided_followup_permission" in system_state:
        return 1.0 if _as_bool(system_state.get("z_k_guided_followup_permission")) else 0.0
    if not (_as_bool(system_state.get("use_lyapunov_pricing")) and _as_bool(system_state.get("use_mode_controller"))):
        return 0.0
    queue_pressure = _as_float(system_state.get("queue_pressure"), 0.0)
    battery_soc = _as_float(system_state.get("battery_soc"), 1.0)
    thermal_pressure = _as_float(system_state.get("thermal_pressure"), 0.0)
    resource_slack = min(_clamp((battery_soc - 0.60) / 0.40), _clamp((0.85 - thermal_pressure) / 0.85))
    queue_slack = _clamp(1.0 - queue_pressure)
    return _clamp(0.5 + 0.5 * min(resource_slack, queue_slack))


def compute_expected_residual_information_value(
    task_state: Any,
    system_state: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> ERIVBreakdown:
    cfg = phase6_proactive_config(config)
    system_state = system_state or {}
    fields_used = [
        "phase6_proactive_followup.enabled",
        "task_id",
        "product_requirement",
        "delivered_products",
        "pending_products",
        "task_closed_flag",
        "residual_utility_remaining",
        "residual_delta_u_I",
        "residual_enqueued_count",
        "residual_delivered_count",
        "virtual_continue_request_flag",
        "queue_mb",
        "queue_max_mb",
        "queued_products",
        "remaining_deadline_s",
        "battery_soc",
        "thermal_pressure",
        "task_value",
        "high_value_flag",
        "emergency_flag",
        "uncertainty",
        "model_quality_factor",
        "contact_available",
        "contact_opportunity_score",
        "payload_size_mb",
        "use_lyapunov_pricing",
        "use_mode_controller",
        "z_k_guided_followup_permission",
    ]

    delivered_products = _as_set(_get(task_state, "delivered_products", set()))
    residual_product_type = _residual_product(task_state)
    product_requirement = normalize_product_requirement(str(_get(task_state, "product_requirement", "verification")))
    residual_delta = 0.0
    if residual_product_type:
        residual_delta = delta_information_utility(residual_product_type, delivered_products, product_requirement)
    residual_remaining = _as_float(_get(task_state, "residual_utility_remaining", 0.0), 0.0)
    if "residual_delta_u_I" in system_state:
        residual_delta = _as_float(system_state.get("residual_delta_u_I"), residual_delta)
    residual_delta = max(residual_delta, residual_remaining)
    task_value = _as_float(system_state.get("task_value", _get(task_state, "task_value", 1.0)), 1.0)
    model_quality_factor = _clamp(_as_float(system_state.get("model_quality_factor", system_state.get("detection_quality", 1.0)), 1.0))
    remaining_deadline_s = _as_float(system_state.get("remaining_deadline_s", 0.0), 0.0)
    deadline_scale_s = max(_as_float(system_state.get("deadline_s", _get(task_state, "deadline_s", 240.0)), 240.0), 1.0)
    timeliness_factor = _clamp(remaining_deadline_s / deadline_scale_s)
    contact_opportunity_score = _contact_score(system_state)
    z_k_followup_gate = _z_k_followup_gate(system_state)
    payload_size_mb = max(_as_float(system_state.get("payload_size_mb", 1.0), 1.0), 1.0e-9)
    information_density = residual_delta / payload_size_mb
    value_density_per_mb = (
        max(0.0, task_value)
        * max(0.0, residual_delta)
        * model_quality_factor
        * timeliness_factor
        / payload_size_mb
    )
    eriv_score = (
        max(0.0, task_value)
        * max(0.0, residual_delta)
        * model_quality_factor
        * timeliness_factor
        * contact_opportunity_score
        * z_k_followup_gate
        / payload_size_mb
    )
    return ERIVBreakdown(
        eriv_score=max(0.0, eriv_score),
        residual_delta_u_I=max(0.0, residual_delta),
        task_value=max(0.0, task_value),
        model_quality_factor=model_quality_factor,
        timeliness_factor=timeliness_factor,
        contact_opportunity_score=contact_opportunity_score,
        z_k_followup_gate=z_k_followup_gate,
        payload_size_mb=payload_size_mb,
        normalized_utility_per_mb=max(0.0, information_density),
        information_density=max(0.0, information_density),
        value_density_per_mb=max(0.0, value_density_per_mb),
        fields_used=tuple(fields_used),
        blockers=(),
        leakage_safe_flag=True,
    )


def evaluate_proactive_followup_eligibility(
    task_state: Any,
    system_state: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> ProactiveFollowupDecision:
    cfg = phase6_proactive_config(config)
    system_state = system_state or {}
    eriv = compute_expected_residual_information_value(task_state, system_state, cfg)
    fields_used = eriv.fields_used + ("variant", "router_name")
    threshold = _as_float(cfg.get("eriv_threshold", 0.25), 0.25)
    residual_threshold = _as_float(cfg.get("residual_utility_threshold", 0.08), 0.08)
    queue_pressure_threshold = _as_float(cfg.get("queue_pressure_threshold", 0.65), 0.65)
    min_remaining_deadline_s = _as_float(cfg.get("min_remaining_deadline_s", 60.0), 60.0)
    battery_guard = _as_float(cfg.get("battery_guard_threshold", 0.60), 0.60)
    thermal_guard = _as_float(cfg.get("thermal_guard_threshold", 0.85), 0.85)
    require_contact = _as_bool(cfg.get("require_contact_opportunity", True), True)

    task_id = str(_get(task_state, "task_id", ""))
    delivered_products = _as_set(_get(task_state, "delivered_products", set()))
    pending_products = _as_set(_get(task_state, "pending_products", set()))
    residual_pending = sorted(product for product in pending_products if product in RESIDUAL_PRODUCT_TYPES)
    residual_product_type = residual_pending[0] if len(residual_pending) == 1 else ""
    product_requirement = normalize_product_requirement(str(_get(task_state, "product_requirement", "verification")))
    queued_products = system_state.get("queued_products", set())
    queue_mb = _as_float(system_state.get("queue_mb", 0.0), 0.0)
    queue_max_mb = max(_as_float(system_state.get("queue_max_mb", 1.0), 1.0), 1.0e-9)
    queue_pressure = _as_float(system_state.get("queue_pressure", queue_mb / queue_max_mb), queue_mb / queue_max_mb)
    remaining_deadline_s = _as_float(system_state.get("remaining_deadline_s", 0.0), 0.0)
    battery_soc = _as_float(system_state.get("battery_soc", 1.0), 1.0)
    thermal_pressure = _as_float(system_state.get("thermal_pressure", 0.0), 0.0)
    value_level = str(system_state.get("value_level", _get(task_state, "value_level", "")) or "")

    blockers: list[str] = []
    if not _as_bool(cfg.get("enabled", False)):
        blockers.append("phase6_disabled")
    if not _full_method_allowed(system_state, cfg):
        blockers.append("variant_not_full_proposed")
    if "metadata" not in delivered_products:
        blockers.append("metadata_not_delivered")
    if len(residual_pending) != 1:
        blockers.append("no_pending_residual_product" if not residual_pending else "multiple_pending_residual_products")
    if bool(_get(task_state, "task_closed_flag", False)):
        blockers.append("task_closed")
    if residual_product_type and residual_product_type in delivered_products:
        blockers.append("product_already_delivered")
    if int(_get(task_state, "residual_enqueued_count", 0) or 0) > 0:
        blockers.append("residual_already_enqueued")
    if int(_get(task_state, "residual_delivered_count", 0) or 0) > 0:
        blockers.append("residual_already_delivered")
    if bool(_get(task_state, "virtual_continue_request_flag", False)) or int(_get(task_state, "residual_requested_count", 0) or 0) > 0:
        blockers.append("passive_request_already_enqueued_residual")
    if residual_product_type and _queued_residual_duplicate(task_id, residual_product_type, queued_products):
        blockers.append("duplicate_residual_already_queued")
    if eriv.residual_delta_u_I <= residual_threshold:
        blockers.append("residual_utility_below_threshold")
    if eriv.eriv_score < threshold:
        blockers.append("eriv_below_threshold")
    if battery_soc < battery_guard:
        blockers.append("battery_unsafe")
    if thermal_pressure > thermal_guard:
        blockers.append("thermal_unsafe")
    if queue_pressure >= queue_pressure_threshold:
        blockers.append("queue_pressure_too_high")
    if remaining_deadline_s <= min_remaining_deadline_s:
        blockers.append("no_useful_remaining_deadline")
    if require_contact and eriv.contact_opportunity_score <= 0.0:
        blockers.append("no_contact_opportunity")
    if eriv.z_k_followup_gate <= 0.0:
        blockers.append("z_k_guided_followup_permission_blocked")
    roi_ctx_gate = evaluate_roi_context_gate(
        product_requirement=product_requirement,
        value_level=value_level,
        value_density=eriv.value_density_per_mb,
        system_state=system_state,
        config=config,
        eriv_score=eriv.eriv_score,
        residual_product_type=residual_product_type,
    )
    if not roi_ctx_gate.allowed:
        blockers.extend(roi_ctx_gate.blockers)

    return ProactiveFollowupDecision(
        proactive_flag=not blockers,
        residual_product_type=residual_product_type,
        eriv_score=eriv.eriv_score,
        threshold=threshold,
        reason_code="eligible" if not blockers else blockers[0],
        blockers=tuple(blockers),
        fields_used=tuple(fields_used),
        leakage_safe_flag=True,
        eriv_breakdown=eriv,
    )
