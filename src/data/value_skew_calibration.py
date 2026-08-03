"""Default-off mission-prior value skew calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from src.common.types import TaskSample
from src.data.mission_prior_value import compute_mission_prior_value_v2


LEVELS = ("low", "medium", "high", "urgent")
SOFT_MODERATE_MULTIPLIERS = {"low": 1.0, "medium": 2.0, "high": 5.0, "urgent": 10.0}

DEFAULT_VALUE_SKEW_CFG: dict[str, Any] = {
    "enabled": False,
    "profile": "soft_moderate",
    "assignment_rule": "v2_conservative",
    "normalization": "mean_preserving",
    "profiles": {
        "soft_moderate": SOFT_MODERATE_MULTIPLIERS,
    },
}


@dataclass(frozen=True)
class ValueSkewAssignment:
    base_prior_score: float
    value_level: str
    value_multiplier: float
    value_skew_profile: str
    normalization_alpha: float
    calibrated_task_value: float
    value_assignment_reason: str


def value_skew_config(container_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_VALUE_SKEW_CFG)
    raw = (container_cfg or {}).get("value_skew_calibration", {})
    raw_cfg = raw if isinstance(raw, Mapping) else {}
    for key, value in raw_cfg.items():
        if isinstance(value, Mapping) and isinstance(cfg.get(key), Mapping):
            nested = dict(cfg[key])
            nested.update(value)
            cfg[key] = nested
        else:
            cfg[key] = value
    return cfg


def value_skew_enabled(container_cfg: Mapping[str, Any] | None) -> bool:
    return bool(value_skew_config(container_cfg).get("enabled", False))


def task_effective_value(task: TaskSample) -> float:
    if getattr(task, "calibrated_task_value", None) is not None:
        return float(task.calibrated_task_value)
    return float(task.task_value)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _region_score(value: object) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(value or "").strip().lower(), 0)


def _base_prior(task: TaskSample, container_cfg: Mapping[str, Any] | None) -> float:
    metadata = task.metadata if isinstance(task.metadata, Mapping) else {}
    if "base_prior_score" in metadata:
        return _clamp(_as_float(metadata.get("base_prior_score"), task.task_value))
    if "new_task_value" in metadata:
        return _clamp(_as_float(metadata.get("new_task_value"), task.task_value))
    result = compute_mission_prior_value_v2(
        task_id=task.task_id,
        scene_id=task.scene_id,
        task_class=task.task_class,
        product_requirement=task.product_requirement,
        deadline_s=float(task.deadline_s),
        container_cfg=dict(container_cfg or {}),
        source_priority=None,
    )
    return _clamp(result.task_value)


def _old_value(task: TaskSample) -> float:
    metadata = task.metadata if isinstance(task.metadata, Mapping) else {}
    return _as_float(metadata.get("old_task_value"), float(task.task_value))


def assign_value_level_v2_conservative(
    task: TaskSample,
    *,
    base_prior_score: float,
    q50: float,
    q85: float,
) -> tuple[str, str]:
    metadata = task.metadata if isinstance(task.metadata, Mapping) else {}
    task_class = str(task.task_class or "").strip().lower()
    requirement = str(task.product_requirement or "").strip().lower()
    source = _as_float(metadata.get("h_source_priority", metadata.get("source_priority")), 0.0)
    region = _region_score(metadata.get("area_prior_bucket", metadata.get("region_prior")))
    high_flag = bool(task.high_value_flag)
    emergency = bool(task.emergency_flag) or task_class == "emergency"
    if emergency:
        return "urgent", "emergency_flag_or_class_kept_urgent"
    if high_flag and (region >= 2 or base_prior_score >= q85 or source >= 0.85):
        return "high", "high_value_with_strong_region_source_or_top_prior"
    if task_class in {"routine", "background"} and not high_flag:
        return "low", "routine_or_background_non_emergency"
    if high_flag or requirement in {"context", "evidence"} or base_prior_score >= q50:
        return "medium", "high_value_or_context_without_strong_prior_demoted_to_medium"
    return "medium", "ordinary_verification_medium"


def _multiplier_map(cfg: Mapping[str, Any]) -> dict[str, float]:
    profile = str(cfg.get("profile", "soft_moderate"))
    profiles = cfg.get("profiles", {})
    raw = profiles.get(profile, SOFT_MODERATE_MULTIPLIERS) if isinstance(profiles, Mapping) else SOFT_MODERATE_MULTIPLIERS
    if not isinstance(raw, Mapping):
        raw = SOFT_MODERATE_MULTIPLIERS
    merged = dict(SOFT_MODERATE_MULTIPLIERS if profile == "soft_moderate" else {})
    merged.update({str(key): float(value) for key, value in raw.items()})
    return {level: float(merged.get(level, SOFT_MODERATE_MULTIPLIERS[level])) for level in LEVELS}


def apply_value_skew_calibration(
    stream: list[TaskSample | None],
    container_cfg: Mapping[str, Any] | None,
) -> list[TaskSample | None]:
    cfg = value_skew_config(container_cfg)
    if not bool(cfg.get("enabled", False)):
        return stream
    tasks = [task for task in stream if task is not None]
    if not tasks:
        return stream
    base_by_id = {task.task_id: _base_prior(task, container_cfg) for task in tasks}
    base_values = list(base_by_id.values())
    q50 = _percentile(base_values, 0.50)
    q85 = _percentile(base_values, 0.85)
    profile = str(cfg.get("profile", "soft_moderate"))
    multipliers = _multiplier_map(cfg)
    assignments: dict[str, tuple[str, str, float, float]] = {}
    raw_values: list[float] = []
    for task in tasks:
        base = base_by_id[task.task_id]
        level, reason = assign_value_level_v2_conservative(task, base_prior_score=base, q50=q50, q85=q85)
        multiplier = float(multipliers[level])
        raw = multiplier * (0.8 + 0.4 * base)
        assignments[task.task_id] = (level, reason, multiplier, raw)
        raw_values.append(raw)
    old_mean = sum(_old_value(task) for task in tasks) / max(len(tasks), 1)
    raw_mean = sum(raw_values) / max(len(raw_values), 1)
    alpha = old_mean / max(raw_mean, 1.0e-12)
    calibrated_stream: list[TaskSample | None] = []
    for task in stream:
        if task is None:
            calibrated_stream.append(None)
            continue
        base = base_by_id[task.task_id]
        level, reason, multiplier, raw = assignments[task.task_id]
        calibrated = alpha * raw
        metadata = dict(task.metadata or {})
        metadata.update(
            {
                "base_prior_score": base,
                "value_level": level,
                "value_multiplier": multiplier,
                "value_skew_profile": profile,
                "normalization_alpha": alpha,
                "calibrated_task_value": calibrated,
                "value_assignment_reason": reason,
                "value_skew_normalization": "mean_preserving",
                "value_skew_enabled": True,
                "legacy_task_value": float(task.task_value),
            }
        )
        calibrated_stream.append(
            replace(
                task,
                metadata=metadata,
                base_prior_score=base,
                value_level=level,
                value_multiplier=multiplier,
                value_skew_profile=profile,
                normalization_alpha=alpha,
                calibrated_task_value=calibrated,
                value_assignment_reason=reason,
            )
        )
    return calibrated_stream
