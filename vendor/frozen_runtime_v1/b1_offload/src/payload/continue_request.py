"""Phase 5 virtual continue-request decision rule.

The rule is intentionally pure and limited to mission-prior/product lifecycle
state available at decision time. It does not inspect detector labels, object
classes, future delivery outcomes, or realized future utility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.common.types import ProductLifecycleState
from src.payload.information_sufficiency import delta_information_utility, normalize_product_requirement


RESIDUAL_PRODUCT_TYPES = {"roi", "roi_context"}
FOLLOWUP_REQUIREMENTS = {"verification", "context", "evidence"}


@dataclass(frozen=True)
class ContinueRequestDecision:
    request_flag: bool
    residual_product_type: str = ""
    residual_utility_remaining: float = 0.0
    reason_code: str = "not_evaluated"
    blockers: tuple[str, ...] = field(default_factory=tuple)
    fields_used: tuple[str, ...] = field(default_factory=tuple)
    leakage_safe_flag: bool = True


def _cfg(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    config = config or {}
    nested = config.get("phase5_virtual_continue") if isinstance(config, Mapping) else None
    if isinstance(nested, Mapping):
        return nested
    return config


def phase5_continue_config(simulator_cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the Phase 5 config block, accepting either nested or direct input."""

    simulator_cfg = simulator_cfg or {}
    progressive = simulator_cfg.get("progressive_payload") if isinstance(simulator_cfg, Mapping) else None
    if isinstance(progressive, Mapping):
        return _cfg(progressive)
    return _cfg(simulator_cfg)


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(";") if item.strip()}
    try:
        return {str(item) for item in value if str(item)}
    except TypeError:
        return {str(value)}


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _queued_residual_duplicate(task_id: str, residual_product_type: str, queued_products: Any) -> bool:
    keys = _as_set(queued_products)
    identity_key = f"{task_id}:{residual_product_type}"
    return identity_key in keys


def _decision(
    request_flag: bool,
    residual_product_type: str,
    residual_utility_remaining: float,
    reason_code: str,
    blockers: list[str],
    fields_used: list[str],
) -> ContinueRequestDecision:
    return ContinueRequestDecision(
        request_flag=request_flag,
        residual_product_type=residual_product_type if request_flag else residual_product_type,
        residual_utility_remaining=max(0.0, float(residual_utility_remaining)),
        reason_code=reason_code,
        blockers=tuple(blockers),
        fields_used=tuple(fields_used),
        leakage_safe_flag=True,
    )


def evaluate_virtual_continue_request(
    task_state: ProductLifecycleState | Mapping[str, Any],
    system_state: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> ContinueRequestDecision:
    """Evaluate whether a staged task may enqueue one passive residual product."""

    cfg = phase5_continue_config(config)
    fields_used = [
        "phase5_virtual_continue.enabled",
        "task_id",
        "product_requirement",
        "delivered_products",
        "pending_products",
        "task_closed_flag",
        "residual_utility_remaining",
        "residual_requested_count",
        "residual_enqueued_count",
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
    ]
    if not bool(cfg.get("enabled", False)):
        return _decision(False, "", 0.0, "phase5_disabled", ["phase5_disabled"], fields_used)

    system_state = system_state or {}
    task_id = str(_get(task_state, "task_id", ""))
    delivered_products = _as_set(_get(task_state, "delivered_products", set()))
    pending_products = _as_set(_get(task_state, "pending_products", set()))
    residual_pending = sorted(product for product in pending_products if product in RESIDUAL_PRODUCT_TYPES)
    product_requirement = normalize_product_requirement(str(_get(task_state, "product_requirement", "verification")))
    residual_utility_remaining = float(_get(task_state, "residual_utility_remaining", 0.0) or 0.0)
    residual_threshold = float(cfg.get("residual_utility_threshold", 0.08))
    queued_products = system_state.get("queued_products", set())
    queue_mb = float(system_state.get("queue_mb", 0.0) or 0.0)
    queue_max_mb = float(system_state.get("queue_max_mb", 1.0) or 1.0)
    queue_pressure = queue_mb / max(queue_max_mb, 1.0e-9)
    queue_pressure_threshold = float(cfg.get("queue_pressure_threshold", 0.65))
    remaining_deadline_s = float(system_state.get("remaining_deadline_s", 0.0) or 0.0)
    min_remaining_deadline_s = float(cfg.get("min_remaining_deadline_s", 60.0))
    battery_soc = float(system_state.get("battery_soc", 1.0) or 0.0)
    thermal_pressure = float(system_state.get("thermal_pressure", 0.0) or 0.0)
    battery_guard = float(cfg.get("battery_guard_threshold", 0.60))
    thermal_guard = float(cfg.get("thermal_guard_threshold", 0.85))
    task_value = float(system_state.get("task_value", 0.0) or 0.0)
    high_value_flag = bool(system_state.get("high_value_flag", False))
    emergency_flag = bool(system_state.get("emergency_flag", False))
    uncertainty = float(system_state.get("uncertainty", 0.0) or 0.0)
    uncertainty_threshold = float(cfg.get("high_value_uncertainty_threshold", 0.25))
    high_value_threshold = float(cfg.get("high_value_threshold", 0.75))

    blockers: list[str] = []
    residual_product_type = residual_pending[0] if residual_pending else ""

    if "metadata" not in delivered_products:
        blockers.append("metadata_not_delivered")
    if len(residual_pending) != 1:
        blockers.append("no_pending_residual_product" if not residual_pending else "multiple_pending_residual_products")
    if bool(_get(task_state, "task_closed_flag", False)):
        blockers.append("task_closed")
    if residual_product_type and residual_product_type in delivered_products:
        blockers.append("product_already_delivered")
    if remaining_deadline_s <= min_remaining_deadline_s:
        blockers.append("no_useful_remaining_deadline")
    if residual_utility_remaining <= residual_threshold:
        blockers.append("residual_utility_below_threshold")
    if residual_product_type and _queued_residual_duplicate(task_id, residual_product_type, queued_products):
        blockers.append("duplicate_residual_already_queued")
    if product_requirement not in FOLLOWUP_REQUIREMENTS:
        allowed_by_priority = (high_value_flag or emergency_flag or task_value >= high_value_threshold) and uncertainty >= uncertainty_threshold
        if not allowed_by_priority:
            blockers.append("product_requirement_not_followup_eligible")
    if queue_pressure >= queue_pressure_threshold:
        blockers.append("queue_pressure_too_high")
    if battery_soc < battery_guard:
        blockers.append("battery_unsafe")
    if thermal_pressure > thermal_guard:
        blockers.append("thermal_unsafe")

    if residual_product_type and delta_information_utility(residual_product_type, delivered_products, product_requirement) <= 0.0:
        blockers.append("residual_incremental_utility_not_positive")

    return _decision(
        not blockers,
        residual_product_type,
        residual_utility_remaining,
        "eligible" if not blockers else blockers[0],
        blockers,
        fields_used,
    )
