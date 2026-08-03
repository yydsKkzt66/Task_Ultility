"""Persistent value-aware staged delivery request queue.

The queue is intentionally decision-time only. It uses task metadata, product
lifecycle state, resource/contact state, and configured product/value weights.
It does not inspect ground-truth labels, future delivery outcomes, future
closure, or realized future reward.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


DEFAULT_STAGED_DELIVERY_QUEUE_CFG: dict[str, Any] = {
    "enabled": False,
    "full_method_aliases": ["proposed_lyapunov_rule_mode_safe_bandit"],
    "allow_utility_rule": False,
    "residual_utility_threshold": 0.01,
    "min_value_density": 0.0,
    "eriv_threshold": 0.0,
    "require_future_contact_before_deadline": True,
    "allow_pending_without_future_contact": False,
    "max_admissions_per_slot": 2,
    "max_admissions_per_contact_window": 4,
    "deadline_urgency_lambda": 0.15,
    "timeliness_floor": 0.05,
    "priority_inversion_fix_enabled": False,
    "preemption_priority_floor": 1000.0,
    "product_requirement_weight": {
        "alert": {"metadata": 1.0, "roi": 0.25, "roi_context": 0.0},
        "verification": {"metadata": 0.45, "roi": 1.0, "roi_context": 0.15},
        "context": {"metadata": 0.25, "roi": 0.65, "roi_context": 1.0},
        "evidence": {"metadata": 0.20, "roi": 0.70, "roi_context": 1.20},
    },
    "value_level_factor": {
        "low": {"metadata": 1.0, "roi": 0.75, "roi_context": 0.0},
        "medium": {"metadata": 1.0, "roi": 0.90, "roi_context": 0.0},
        "high": {"metadata": 1.0, "roi": 1.0, "roi_context": 1.0},
        "urgent": {"metadata": 1.1, "roi": 1.1, "roi_context": 1.3},
    },
    "fixed_bonus_debug": {"enabled": False, "bonus": 0.0},
}


STAGED_REQUEST_STATUSES = {"pending", "enqueued", "delivered", "expired", "blocked", "dropped"}


@dataclass
class StagedDeliveryRequest:
    request_id: str
    task_id: str
    product_type: str
    parent_product_type: str
    product_requirement: str
    value_level: str
    calibrated_task_value: float
    residual_delta_information_utility: float
    residual_utility_remaining: float
    detection_quality: float
    timeliness_factor: float
    payload_size_mb: float
    deadline_slot: int
    created_slot: int
    last_evaluated_slot: int
    earliest_delivery_slot: int
    status: str = "pending"
    duplicate_key: str = ""
    priority_score: float = 0.0
    priority_components: dict[str, float] = field(default_factory=dict)
    blocker_reason: str = ""
    leakage_safe_flag: bool = True
    scene_id: str = ""
    patch_id: str = ""
    deadline_s: float = 0.0
    created_time_s: float = 0.0
    enqueued_slot: int | None = None
    delivered_slot: int | None = None
    expired_slot: int | None = None
    lifecycle_id: str = ""

    def __post_init__(self) -> None:
        if not self.duplicate_key:
            self.duplicate_key = f"{self.task_id}:{self.product_type}"
        if self.status not in STAGED_REQUEST_STATUSES:
            self.status = "pending"


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _nested_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key, {})
    return value if isinstance(value, Mapping) else {}


def staged_delivery_queue_config(container_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    container_cfg = container_cfg or {}
    progressive = container_cfg.get("progressive_payload") if isinstance(container_cfg, Mapping) else None
    raw = None
    if isinstance(progressive, Mapping):
        raw = progressive.get("value_aware_staged_delivery_queue")
    if not isinstance(raw, Mapping) and isinstance(container_cfg, Mapping):
        raw = container_cfg.get("value_aware_staged_delivery_queue")

    cfg = dict(DEFAULT_STAGED_DELIVERY_QUEUE_CFG)
    cfg["product_requirement_weight"] = {
        key: dict(value)
        for key, value in DEFAULT_STAGED_DELIVERY_QUEUE_CFG["product_requirement_weight"].items()
    }
    cfg["value_level_factor"] = {
        key: dict(value)
        for key, value in DEFAULT_STAGED_DELIVERY_QUEUE_CFG["value_level_factor"].items()
    }
    cfg["fixed_bonus_debug"] = dict(DEFAULT_STAGED_DELIVERY_QUEUE_CFG["fixed_bonus_debug"])
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if key in {"product_requirement_weight", "value_level_factor"} and isinstance(value, Mapping):
                nested = {
                    outer: dict(inner)
                    for outer, inner in cfg[key].items()
                }
                for outer, inner in value.items():
                    merged = dict(nested.get(str(outer), {}))
                    if isinstance(inner, Mapping):
                        merged.update({str(k): float(v) for k, v in inner.items()})
                    nested[str(outer)] = merged
                cfg[key] = nested
            elif key == "fixed_bonus_debug" and isinstance(value, Mapping):
                merged = dict(cfg["fixed_bonus_debug"])
                merged.update(value)
                cfg[key] = merged
            else:
                cfg[key] = value
    return cfg


def staged_delivery_enabled_for_router(
    container_cfg: Mapping[str, Any] | None,
    router_name: str,
) -> bool:
    cfg = staged_delivery_queue_config(container_cfg)
    if not _as_bool(cfg.get("enabled"), False):
        return False
    router = str(router_name or "")
    if router == "utility_rule" and not _as_bool(cfg.get("allow_utility_rule"), False):
        return False
    aliases = cfg.get("full_method_aliases", ())
    if isinstance(aliases, str):
        allowed = {item.strip() for item in aliases.split(",") if item.strip()}
    else:
        try:
            allowed = {str(item) for item in aliases}
        except TypeError:
            allowed = set()
    return not allowed or router in allowed


def product_requirement_weight(cfg: Mapping[str, Any], requirement: str, product_type: str) -> float:
    table = _nested_mapping(cfg, "product_requirement_weight")
    row = table.get(str(requirement or "verification").strip().lower(), table.get("verification", {}))
    return max(0.0, _as_float(row.get(product_type), 0.0) if isinstance(row, Mapping) else 0.0)


def value_level_factor(cfg: Mapping[str, Any], value_level: str, product_type: str) -> float:
    table = _nested_mapping(cfg, "value_level_factor")
    row = table.get(str(value_level or "medium").strip().lower(), table.get("medium", {}))
    return max(0.0, _as_float(row.get(product_type), 0.0) if isinstance(row, Mapping) else 0.0)


def deadline_urgency(current_slot: int, created_slot: int, deadline_slot: int) -> float:
    total = max(1, int(deadline_slot) - int(created_slot))
    remaining = max(0, int(deadline_slot) - int(current_slot))
    if remaining <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - remaining / total))


def compute_staged_priority(
    request: StagedDeliveryRequest,
    *,
    current_slot: int,
    config: Mapping[str, Any] | None,
) -> tuple[float, dict[str, float]]:
    cfg = staged_delivery_queue_config(config)
    product_weight = product_requirement_weight(cfg, request.product_requirement, request.product_type)
    level_factor = value_level_factor(cfg, request.value_level, request.product_type)
    timeliness = max(_as_float(cfg.get("timeliness_floor"), 0.0), max(0.0, float(request.timeliness_factor)))
    payload = max(float(request.payload_size_mb), 1.0e-9)
    density = (
        max(0.0, float(request.calibrated_task_value))
        * max(0.0, float(request.residual_delta_information_utility))
        * max(0.0, float(request.detection_quality))
        * timeliness
        * product_weight
        * level_factor
        / payload
    )
    urgency = deadline_urgency(current_slot, request.created_slot, request.deadline_slot)
    deadline_bonus = _as_float(cfg.get("deadline_urgency_lambda"), 0.0) * urgency
    fixed_cfg = cfg.get("fixed_bonus_debug", {})
    fixed_bonus = 0.0
    if isinstance(fixed_cfg, Mapping) and _as_bool(fixed_cfg.get("enabled"), False):
        fixed_bonus = _as_float(fixed_cfg.get("bonus"), 0.0)
    priority = density + deadline_bonus + fixed_bonus
    components = {
        "V_eff": max(0.0, float(request.calibrated_task_value)),
        "delta_u_I": max(0.0, float(request.residual_delta_information_utility)),
        "detection_quality": max(0.0, float(request.detection_quality)),
        "timeliness_factor": timeliness,
        "product_requirement_weight": product_weight,
        "value_level_factor": level_factor,
        "payload_size_mb": payload,
        "value_density_priority": density,
        "deadline_urgency": urgency,
        "deadline_bonus": deadline_bonus,
        "fixed_bonus_debug": fixed_bonus,
        "priority_score": priority,
    }
    return max(0.0, priority), components


def request_to_trace_row(request: StagedDeliveryRequest) -> dict[str, object]:
    components = dict(request.priority_components)
    return {
        "request_id": request.request_id,
        "task_id": request.task_id,
        "scene_id": request.scene_id,
        "patch_id": request.patch_id,
        "product_type": request.product_type,
        "parent_product_type": request.parent_product_type,
        "product_requirement": request.product_requirement,
        "value_level": request.value_level,
        "calibrated_task_value": request.calibrated_task_value,
        "residual_delta_information_utility": request.residual_delta_information_utility,
        "residual_utility_remaining": request.residual_utility_remaining,
        "detection_quality": request.detection_quality,
        "timeliness_factor": request.timeliness_factor,
        "payload_size_mb": request.payload_size_mb,
        "deadline_slot": request.deadline_slot,
        "created_slot": request.created_slot,
        "last_evaluated_slot": request.last_evaluated_slot,
        "earliest_delivery_slot": request.earliest_delivery_slot,
        "status": request.status,
        "duplicate_key": request.duplicate_key,
        "priority_score": request.priority_score,
        "priority_components": ";".join(f"{key}={value:.6g}" for key, value in sorted(components.items())),
        "blocker_reason": request.blocker_reason,
        "leakage_safe_flag": int(bool(request.leakage_safe_flag)),
        "enqueued_slot": "" if request.enqueued_slot is None else request.enqueued_slot,
        "delivered_slot": "" if request.delivered_slot is None else request.delivered_slot,
        "expired_slot": "" if request.expired_slot is None else request.expired_slot,
        "lifecycle_id": request.lifecycle_id,
    }
