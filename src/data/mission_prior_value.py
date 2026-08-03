"""Leakage-free mission prior value model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any


VALUE_MODEL_VERSION = "mission_prior_value_v2"


DEFAULT_TASK_VALUE_MODEL_CFG: dict[str, Any] = {
    "enable_mission_prior_value_v2": False,
    "clip_min": 0.05,
    "clip_max": 0.95,
    "weights": {
        "task_class": 0.50,
        "evidence_need": 0.25,
        "area_prior": 0.15,
        "source_priority": 0.10,
        "deadline_class": 0.0,
    },
    "task_class_map": {
        "routine": 0.25,
        "high_value": 0.70,
        "emergency": 0.95,
    },
    "evidence_need_map": {
        "alert": 0.25,
        "alert_only": 0.25,
        "metadata": 0.25,
        "verification": 0.65,
        "confirmation": 0.65,
        "context": 0.75,
        "evidence": 0.85,
        "closure": 0.85,
        "emergency_evidence": 0.95,
    },
    "area_prior_mode": "deterministic_scene_hash",
    "area_prior_values": {
        "low": 0.20,
        "medium": 0.50,
        "high": 0.80,
    },
    "source_priority_default": 0.0,
    "include_deadline_class": False,
    "jitter": 0.0,
}


@dataclass(frozen=True)
class MissionPriorValueResult:
    task_value: float
    pi_task_class: float
    r_evidence_need: float
    g_area_prior: float
    h_source_priority: float
    optional_deadline_class_term: float
    area_prior_bucket: str
    value_model_version: str = VALUE_MODEL_VERSION
    no_label_leakage_flag: bool = True


def task_value_model_cfg(container_cfg: dict[str, object] | None) -> dict[str, Any]:
    """Return task value model config with project defaults filled in."""

    cfg = dict(DEFAULT_TASK_VALUE_MODEL_CFG)
    raw = (container_cfg or {}).get("task_value_model", {})
    raw_cfg = raw if isinstance(raw, dict) else {}
    for key, value in raw_cfg.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merged = dict(cfg[key])
            merged.update(value)
            cfg[key] = merged
        else:
            cfg[key] = value
    return cfg


def mission_prior_value_v2_enabled(container_cfg: dict[str, object] | None) -> bool:
    return bool(task_value_model_cfg(container_cfg).get("enable_mission_prior_value_v2", False))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _stable_unit_interval(*parts: object) -> float:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64 - 1)


def _lookup_float(mapping: object, key: str, default: float) -> float:
    if not isinstance(mapping, dict):
        return default
    normalized = key.strip().lower()
    if normalized in mapping:
        return float(mapping[normalized])
    return default


def _area_prior(scene_id: str, cfg: dict[str, Any]) -> tuple[str, float]:
    mode = str(cfg.get("area_prior_mode", "deterministic_scene_hash"))
    values = cfg.get("area_prior_values", {})
    table = cfg.get("scene_area_prior_table", {})
    if mode == "fixed_scene_prior_table" and isinstance(table, dict) and scene_id in table:
        raw = table[scene_id]
        if isinstance(raw, str):
            bucket = raw.strip().lower()
            return bucket, _lookup_float(values, bucket, 0.50)
        value = _clamp(float(raw), 0.0, 1.0)
        if value < 1.0 / 3.0:
            return "low", value
        if value < 2.0 / 3.0:
            return "medium", value
        return "high", value

    unit = _stable_unit_interval("area_prior", scene_id or "__missing_scene__")
    if unit < 1.0 / 3.0:
        bucket = "low"
    elif unit < 2.0 / 3.0:
        bucket = "medium"
    else:
        bucket = "high"
    return bucket, _lookup_float(values, bucket, 0.50)


def _deadline_class_term(deadline_s: float) -> float:
    if deadline_s <= 45.0:
        return 0.80
    if deadline_s <= 90.0:
        return 0.55
    return 0.25


def compute_mission_prior_value_v2(
    *,
    task_id: str,
    scene_id: str,
    task_class: str,
    product_requirement: str,
    deadline_s: float,
    container_cfg: dict[str, object] | None,
    source_priority: float | None = None,
) -> MissionPriorValueResult:
    """Compute prior value from decision-time task fields only."""

    cfg = task_value_model_cfg(container_cfg)
    weights = cfg.get("weights", {})
    task_class_key = task_class.strip().lower()
    requirement_key = product_requirement.strip().lower()
    pi_task_class = _lookup_float(cfg.get("task_class_map"), task_class_key, 0.25)
    r_evidence_need = _lookup_float(cfg.get("evidence_need_map"), requirement_key, 0.65)
    area_prior_bucket, g_area_prior = _area_prior(scene_id, cfg)
    h_source_priority = (
        float(cfg.get("source_priority_default", 0.0)) if source_priority is None else _clamp(float(source_priority), 0.0, 1.0)
    )
    optional_deadline_class_term = _deadline_class_term(deadline_s) if bool(cfg.get("include_deadline_class", False)) else 0.0

    value = (
        float(weights.get("task_class", 0.50)) * pi_task_class
        + float(weights.get("evidence_need", 0.25)) * r_evidence_need
        + float(weights.get("area_prior", 0.15)) * g_area_prior
        + float(weights.get("source_priority", 0.10)) * h_source_priority
        + float(weights.get("deadline_class", 0.0)) * optional_deadline_class_term
    )
    jitter = min(abs(float(cfg.get("jitter", 0.0))), 0.02)
    if jitter > 0.0:
        value += (2.0 * _stable_unit_interval("jitter", task_id, scene_id) - 1.0) * jitter
    task_value = _clamp(value, float(cfg.get("clip_min", 0.05)), float(cfg.get("clip_max", 0.95)))
    return MissionPriorValueResult(
        task_value=task_value,
        pi_task_class=pi_task_class,
        r_evidence_need=r_evidence_need,
        g_area_prior=g_area_prior,
        h_source_priority=h_source_priority,
        optional_deadline_class_term=optional_deadline_class_term,
        area_prior_bucket=area_prior_bucket,
    )


def task_value_component_metadata(
    result: MissionPriorValueResult,
    *,
    old_task_value: float,
) -> dict[str, object]:
    return {
        "old_task_value": old_task_value,
        "new_task_value": result.task_value,
        "pi_task_class": result.pi_task_class,
        "r_evidence_need": result.r_evidence_need,
        "g_area_prior": result.g_area_prior,
        "h_source_priority": result.h_source_priority,
        "optional_deadline_class_term": result.optional_deadline_class_term,
        "area_prior_bucket": result.area_prior_bucket,
        "value_model_version": result.value_model_version,
        "no_label_leakage_flag": result.no_label_leakage_flag,
    }
