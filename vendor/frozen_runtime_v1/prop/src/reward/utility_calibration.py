"""Default-off value-cost calibration helpers.

Lifecycle utility is intentionally reserved only in this patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_UTILITY_CALIBRATION_CFG: dict[str, Any] = {
    "enable_value_cost_calibration": False,
    "value_scale": {
        "a": 1.0,
        "b": 0.0,
        "clip_min": 0.05,
        "clip_max": 0.95,
    },
    "cost_penalty_scale": {
        "beta_resource_cost": 1.0,
        "beta_requirement_penalty": 1.0,
        "beta_hard_penalty": 1.0,
    },
    "diagnostic_logging": True,
    "shadow_only": False,
}

DEFAULT_LIFECYCLE_UTILITY_CFG: dict[str, Any] = {
    "enable_lifecycle_utility": False,
    "reserved_only": True,
    "stage_weights": {
        "alert": None,
        "confirmation": None,
        "evidence": None,
    },
    "stage_deadlines": {
        "alert_fraction": None,
        "alert_max_seconds": None,
        "confirmation_fraction": None,
        "evidence_fraction": None,
    },
    "notes": "Reserved for future lifecycle utility. Must not affect current results.",
}


@dataclass(frozen=True)
class UtilityCalibration:
    enabled: bool
    value_scale_a: float
    value_scale_b: float
    clip_min: float
    clip_max: float
    beta_resource_cost: float
    beta_requirement_penalty: float
    beta_hard_penalty: float
    diagnostic_logging: bool
    shadow_only: bool


def _merge_nested(defaults: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(defaults)
    if not isinstance(override, dict):
        return result
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            nested = dict(result[key])
            nested.update(value)
            result[key] = nested
        else:
            result[key] = value
    return result


def lifecycle_utility_active(reward_cfg: dict[str, Any]) -> bool:
    cfg = _merge_nested(DEFAULT_LIFECYCLE_UTILITY_CFG, reward_cfg.get("lifecycle_utility", {}))
    return bool(cfg.get("enable_lifecycle_utility", False))


def assert_lifecycle_utility_reserved(reward_cfg: dict[str, Any]) -> None:
    if lifecycle_utility_active(reward_cfg):
        raise NotImplementedError(
            "lifecycle_utility.enable_lifecycle_utility is reserved only in this patch"
        )


def get_utility_calibration(reward_cfg: dict[str, Any]) -> UtilityCalibration:
    cfg = _merge_nested(DEFAULT_UTILITY_CALIBRATION_CFG, reward_cfg.get("utility_calibration", {}))
    value_scale = _merge_nested(
        DEFAULT_UTILITY_CALIBRATION_CFG["value_scale"],
        cfg.get("value_scale", {}),
    )
    penalty_scale = _merge_nested(
        DEFAULT_UTILITY_CALIBRATION_CFG["cost_penalty_scale"],
        cfg.get("cost_penalty_scale", {}),
    )
    enabled = bool(cfg.get("enable_value_cost_calibration", False))
    return UtilityCalibration(
        enabled=enabled,
        value_scale_a=float(value_scale.get("a", 1.0)) if enabled else 1.0,
        value_scale_b=float(value_scale.get("b", 0.0)) if enabled else 0.0,
        clip_min=float(value_scale.get("clip_min", 0.05)) if enabled else 0.05,
        clip_max=float(value_scale.get("clip_max", 0.95)) if enabled else 0.95,
        beta_resource_cost=float(penalty_scale.get("beta_resource_cost", 1.0)) if enabled else 1.0,
        beta_requirement_penalty=float(penalty_scale.get("beta_requirement_penalty", 1.0)) if enabled else 1.0,
        beta_hard_penalty=float(penalty_scale.get("beta_hard_penalty", 1.0)) if enabled else 1.0,
        diagnostic_logging=bool(cfg.get("diagnostic_logging", True)),
        shadow_only=bool(cfg.get("shadow_only", False)),
    )


def calibrate_value(value_raw: float, calibration: UtilityCalibration) -> float:
    if not calibration.enabled:
        return float(value_raw)
    value = calibration.value_scale_a * float(value_raw) + calibration.value_scale_b
    return max(calibration.clip_min, min(calibration.clip_max, value))
