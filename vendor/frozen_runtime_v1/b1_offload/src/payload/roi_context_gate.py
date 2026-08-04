"""Profile-specific ROI+Ctx gate helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


DEFAULT_GATE_PROFILES: dict[str, dict[str, Any]] = {
    "soft_moderate": {
        "eriv_threshold": 0.03,
        "min_value_density": 0.04,
        "eligible_product_requirements": ["context", "evidence"],
        "eligible_value_levels": ["high", "urgent"],
        "queue_pressure_threshold": 0.70,
        "battery_guard_threshold": 0.55,
        "thermal_guard_threshold": 0.88,
        "min_remaining_deadline_s": 40.0,
        "require_contact_opportunity": True,
        "require_current_contact_for_same_slot": True,
        "allow_queued_followup_without_current_contact": True,
        "require_future_contact_before_deadline_for_queue": True,
        "top_k_per_slot": 1,
        "top_k_per_window": 1,
        "top_k_per_contact_window": 2,
        "queued_followup_budget_per_slot": 1,
        "window_budget": 2,
    }
}

DEFAULT_ROI_CONTEXT_GATE_CFG: dict[str, Any] = {
    "enabled": False,
    "profile": "soft_moderate",
    "profiles": DEFAULT_GATE_PROFILES,
}


@dataclass(frozen=True)
class RoiContextGateDecision:
    allowed: bool
    profile: str = "disabled"
    threshold: float = 0.0
    value_density: float = 0.0
    reason_code: str = "gate_disabled"
    blockers: tuple[str, ...] = field(default_factory=tuple)
    fields_used: tuple[str, ...] = field(default_factory=tuple)
    leakage_safe_flag: bool = True


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


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.replace(",", ";").split(";") if item.strip()}
    try:
        return {str(item) for item in value if str(item)}
    except TypeError:
        return {str(value)}


def roi_context_gate_config(container_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    container_cfg = container_cfg or {}
    progressive = container_cfg.get("progressive_payload") if isinstance(container_cfg, Mapping) else None
    raw = None
    if isinstance(progressive, Mapping):
        raw = progressive.get("roi_context_gate")
    if not isinstance(raw, Mapping):
        raw = container_cfg.get("roi_context_gate") if isinstance(container_cfg, Mapping) else None
    cfg = dict(DEFAULT_ROI_CONTEXT_GATE_CFG)
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping) and isinstance(cfg.get(key), Mapping):
                nested = dict(cfg[key])
                nested.update(value)
                cfg[key] = nested
            else:
                cfg[key] = value
    return cfg


def load_gate_profile(container_cfg: Mapping[str, Any] | None) -> tuple[bool, str, dict[str, Any]]:
    cfg = roi_context_gate_config(container_cfg)
    enabled = bool(cfg.get("enabled", False))
    profile_name = str(cfg.get("profile", "soft_moderate"))
    profiles = cfg.get("profiles", {})
    profile = {}
    if isinstance(profiles, Mapping):
        profile = dict(profiles.get(profile_name, DEFAULT_GATE_PROFILES.get(profile_name, DEFAULT_GATE_PROFILES["soft_moderate"])))
    if not profile:
        profile = dict(DEFAULT_GATE_PROFILES.get(profile_name, DEFAULT_GATE_PROFILES["soft_moderate"]))
    return enabled, profile_name, profile


def evaluate_roi_context_gate(
    *,
    product_requirement: str,
    value_level: str,
    value_density: float,
    system_state: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
    eriv_score: float | None = None,
    residual_product_type: str = "roi_context",
) -> RoiContextGateDecision:
    fields = (
        "roi_context_gate.enabled",
        "roi_context_gate.profile",
        "product_requirement",
        "value_level",
        "value_density",
        "ERIV",
        "queue_pressure",
        "battery_soc",
        "thermal_pressure",
        "remaining_deadline_s",
        "contact_available",
        "future_contact_feasible",
        "followup_mode",
        "candidate_rank",
        "slot_selected_count",
        "contact_window_selected_count",
        "queued_followup_selected_count",
        "top_k_per_slot",
        "top_k_per_contact_window",
        "queued_followup_budget_per_slot",
    )
    if residual_product_type != "roi_context":
        return RoiContextGateDecision(True, reason_code="not_roi_context", fields_used=fields)
    enabled, profile_name, profile = load_gate_profile(config)
    if not enabled:
        return RoiContextGateDecision(True, profile=profile_name, reason_code="gate_disabled", fields_used=fields)
    system_state = system_state or {}
    threshold = _as_float(profile.get("eriv_threshold"), 0.0)
    density_threshold = _as_float(profile.get("min_value_density"), 0.0)
    eligible_requirements = _as_set(profile.get("eligible_product_requirements", {"context", "evidence"}))
    eligible_levels = _as_set(profile.get("eligible_value_levels", {"high", "urgent"}))
    queue_pressure = _as_float(system_state.get("queue_pressure"), 0.0)
    queue_mb = _as_float(system_state.get("queue_mb"), 0.0)
    queue_max_mb = max(_as_float(system_state.get("queue_max_mb"), 1.0), 1.0e-9)
    if "queue_pressure" not in system_state:
        queue_pressure = queue_mb / queue_max_mb
    battery_soc = _as_float(system_state.get("battery_soc"), 1.0)
    thermal_pressure = _as_float(system_state.get("thermal_pressure"), 0.0)
    remaining_deadline_s = _as_float(system_state.get("remaining_deadline_s"), 0.0)
    contact_available = _as_bool(system_state.get("contact_available"), False)
    contact_score = _as_float(system_state.get("contact_opportunity_score"), 1.0 if contact_available else 0.0)
    followup_mode = str(system_state.get("followup_mode", "same_slot")).strip().lower()
    if followup_mode in {"queue", "enqueue", "queued_followup", "carryover"}:
        followup_mode = "queued"
    future_contact_feasible = _as_bool(system_state.get("future_contact_feasible"), False)
    candidate_rank = int(_as_float(system_state.get("candidate_rank"), 1.0))
    slot_selected_count = int(_as_float(system_state.get("slot_selected_count"), system_state.get("window_selected_count", 0.0)))
    contact_window_selected_count = int(
        _as_float(system_state.get("contact_window_selected_count"), system_state.get("window_selected_count", 0.0))
    )
    queued_followup_selected_count = int(
        _as_float(system_state.get("queued_followup_selected_count"), system_state.get("window_selected_count", 0.0))
    )
    blockers: list[str] = []
    if str(product_requirement).strip().lower() not in eligible_requirements:
        blockers.append("semantic_requirement_not_context_or_evidence")
    if str(value_level).strip().lower() not in eligible_levels:
        blockers.append("value_level_not_high_or_urgent")
    score = _as_float(eriv_score, value_density)
    if score < threshold:
        blockers.append("eriv_below_profile_threshold")
    if _as_float(value_density) < density_threshold:
        blockers.append("value_density_below_profile_threshold")
    if queue_pressure >= _as_float(profile.get("queue_pressure_threshold"), 1.0):
        blockers.append("queue_pressure_too_high")
    if battery_soc < _as_float(profile.get("battery_guard_threshold"), 0.0):
        blockers.append("battery_unsafe")
    if thermal_pressure > _as_float(profile.get("thermal_guard_threshold"), 1.0):
        blockers.append("thermal_unsafe")
    if remaining_deadline_s <= _as_float(profile.get("min_remaining_deadline_s"), 0.0):
        blockers.append("no_useful_remaining_deadline")
    if _as_bool(profile.get("require_contact_opportunity"), True) and contact_score <= 0.0:
        if followup_mode == "queued" and _as_bool(profile.get("allow_queued_followup_without_current_contact"), False):
            if _as_bool(profile.get("require_future_contact_before_deadline_for_queue"), True) and not future_contact_feasible:
                blockers.append("no_contact_opportunity")
        elif followup_mode == "same_slot" and _as_bool(profile.get("require_current_contact_for_same_slot"), True):
            blockers.append("no_contact_opportunity")
        else:
            blockers.append("no_contact_opportunity")
    top_k = int(_as_float(profile.get("top_k_per_slot", profile.get("top_k_per_window")), 0.0))
    if top_k > 0 and candidate_rank > top_k:
        blockers.append("top_k_budget_exceeded")
    if followup_mode == "queued":
        window_budget = int(_as_float(profile.get("queued_followup_budget_per_slot"), 0.0))
        selected_count = queued_followup_selected_count
    else:
        window_budget = int(
            _as_float(
                profile.get("top_k_per_contact_window", profile.get("window_budget", profile.get("top_k_per_window"))),
                0.0,
            )
        )
        selected_count = contact_window_selected_count
    if window_budget > 0 and selected_count >= window_budget:
        blockers.append("window_budget_exceeded")
    return RoiContextGateDecision(
        allowed=not blockers,
        profile=profile_name,
        threshold=threshold,
        value_density=max(0.0, _as_float(value_density)),
        reason_code="eligible" if not blockers else blockers[0],
        blockers=tuple(blockers),
        fields_used=fields,
        leakage_safe_flag=True,
    )
