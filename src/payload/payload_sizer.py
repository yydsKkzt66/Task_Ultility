"""Payload size, information, and encoding-cost estimation."""

from __future__ import annotations

import math

from src.common.types import DetectionResult, GeneratedPayload, TaskSample
from src.data.value_skew_calibration import task_effective_value
from src.payload.information_sufficiency import delta_information_utility, residual_information_utility
from src.payload.payload_policy import get_payload_policy_profile
from src.payload.roi_builder import build_roi_regions


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def resolve_payload_plan(payload_plan: str) -> tuple[str, str]:
    if payload_plan == "metadata_then_roi":
        return "metadata", "roi"
    if payload_plan == "metadata_then_roi_context":
        return "metadata", "roi_context"
    if payload_plan in {"metadata", "roi", "roi_context", "defer"}:
        return payload_plan, ""
    return payload_plan, ""


def calibrated_profile_enabled(payload_profiles_cfg: dict[str, object] | None) -> bool:
    cfg = payload_profiles_cfg or {}
    calibrated = cfg.get("calibrated_profile", {})
    return isinstance(calibrated, dict) and bool(calibrated.get("enabled", False))


def _calibrated_defaults(payload_profiles_cfg: dict[str, object] | None) -> dict[str, object]:
    cfg = payload_profiles_cfg or {}
    calibrated = cfg.get("calibrated_profile", {})
    return dict(calibrated.get("defaults", {})) if isinstance(calibrated, dict) else {}


def _offline_label_summary(task: TaskSample | None) -> dict[str, object]:
    if task is None or not isinstance(task.metadata, dict):
        return {}
    summary = task.metadata.get("offline_label_summary", {})
    return dict(summary) if isinstance(summary, dict) else {}


def _target_count_from_detection(detection_result: DetectionResult | None) -> int:
    if detection_result is None:
        return 0
    if isinstance(detection_result.generated_metadata, dict):
        count = _as_int(detection_result.generated_metadata.get("bbox_count"), -1)
        if count >= 0:
            return count
    return len(build_roi_regions(detection_result))


def _estimate_label_roi_size_mb(
    profile_params: dict[str, object],
    label_summary: dict[str, object],
) -> tuple[float, dict[str, object]]:
    target_count = _as_int(label_summary.get("target_count"), 0)
    bbox_area_mean = _as_float(label_summary.get("bbox_area_mean"), 0.0)
    bbox_area_max = _as_float(label_summary.get("bbox_area_max"), 0.0)
    small_target_ratio = _clamp(_as_float(label_summary.get("small_target_ratio"), 0.0), 0.0, 1.0)
    if target_count <= 0 or bbox_area_mean <= 0.0:
        return 0.0, {}

    fraction = _as_float(profile_params.get("estimated_roi_count_fraction"), 0.06)
    min_count = _as_int(profile_params.get("min_estimated_roi_count", 1), 1)
    max_count = _as_int(profile_params.get("max_estimated_roi_count", 16), 16)
    estimated_count = _clamp(math.ceil(target_count * fraction), min_count, max_count)

    padding = _as_float(profile_params.get("chip_area_padding_factor"), _as_float(profile_params.get("context_margin_factor"), 1.4))
    min_chip_area_px = _as_float(profile_params.get("min_chip_area_px"), 196608.0)
    max_chip_area_px = _as_float(profile_params.get("max_chip_area_px"), 524288.0)
    area = _clamp(max(bbox_area_mean * padding, min_chip_area_px), min_chip_area_px, max_chip_area_px)
    bytes_per_pixel = _as_float(profile_params.get("bytes_per_pixel"), 2.0)
    compression_ratio = max(_as_float(profile_params.get("compression_ratio"), 8.0), 1.0e-9)
    small_target_multiplier = 1.0 + small_target_ratio * _as_float(profile_params.get("small_target_area_multiplier"), 0.15)
    size_mb = (estimated_count * area * bytes_per_pixel * small_target_multiplier / compression_ratio) / (1024.0 * 1024.0)
    size_mb = _clamp(
        size_mb,
        _as_float(profile_params.get("min_payload_mb"), 0.08),
        _as_float(profile_params.get("max_payload_mb"), 3.5),
    )
    return size_mb, {
        "estimated_target_count": target_count,
        "estimated_roi_count": int(estimated_count),
        "estimated_bbox_area": bbox_area_mean,
        "estimated_bbox_area_max": bbox_area_max,
        "small_target_ratio": small_target_ratio,
        "estimated_roi_size_kb": size_mb * 1000.0,
    }


def estimate_payload_size_with_diagnostics(
    policy_name: str,
    detection_result: DetectionResult | None = None,
    task: TaskSample | None = None,
    payload_profiles_cfg: dict[str, object] | None = None,
) -> tuple[float, dict[str, object]]:
    """Estimate payload size and expose label/fallback diagnostics."""

    product_type, residual_product_type = resolve_payload_plan(policy_name)
    profile = get_payload_policy_profile(product_type, payload_profiles_cfg)
    if product_type == "defer" or (detection_result is not None and detection_result.skipped):
        return 0.0, {
            "payload_size_source": "deferred",
            "estimated_target_count": 0,
            "estimated_roi_count": 0,
            "estimated_bbox_area": 0.0,
            "estimated_roi_size_kb": 0.0,
            "estimated_roi_context_size_kb": 0.0,
        }

    if calibrated_profile_enabled(payload_profiles_cfg):
        params = dict(profile.parameters)
        label_summary = _offline_label_summary(task)
        target_count = _as_int(label_summary.get("target_count"), _target_count_from_detection(detection_result))
        if product_type == "metadata":
            base_size_kb = _as_float(params.get("base_size_kb"), profile.base_size_mb * 1000.0)
            record_bytes = _as_float(params.get("per_detection_record_bytes"), 120.0)
            size_mb = (base_size_kb + target_count * record_bytes / 1000.0) / 1000.0
            return size_mb, {
                "payload_size_source": "label_profile" if label_summary else "fallback",
                "estimated_target_count": target_count,
                "estimated_roi_count": target_count,
                "estimated_bbox_area": _as_float(label_summary.get("bbox_area_mean"), 0.0),
                "estimated_roi_size_kb": 0.0,
                "estimated_roi_context_size_kb": 0.0,
            }
        if product_type in {"roi", "roi_context"}:
            roi_profile = profile if product_type == "roi" else get_payload_policy_profile("roi", payload_profiles_cfg)
            roi_size, diagnostics = _estimate_label_roi_size_mb(dict(roi_profile.parameters), label_summary)
            source = "label_profile" if roi_size > 0.0 else "fallback"
            if roi_size <= 0.0:
                roi_size = _as_float(params.get("fallback_roi_size_mb"), _as_float(params.get("fallback_size_mb"), profile.base_size_mb))
                diagnostics = {
                    "estimated_target_count": target_count,
                    "estimated_roi_count": _target_count_from_detection(detection_result),
                    "estimated_bbox_area": _as_float(label_summary.get("bbox_area_mean"), 0.0),
                    "estimated_bbox_area_max": _as_float(label_summary.get("bbox_area_max"), 0.0),
                    "small_target_ratio": _as_float(label_summary.get("small_target_ratio"), 0.0),
                }
            context_multiplier = _as_float(params.get("context_multiplier"), _as_float(params.get("context_margin_factor"), 1.0))
            if product_type == "roi_context":
                size_mb = roi_size * max(context_multiplier, 1.0)
                size_mb = _clamp(
                    size_mb,
                    _as_float(params.get("min_payload_mb"), 0.20),
                    _as_float(params.get("max_payload_mb"), 8.0),
                )
            else:
                size_mb = roi_size
            diagnostics.update(
                {
                    "payload_size_source": source,
                    "estimated_roi_size_kb": roi_size * 1000.0,
                    "estimated_roi_context_size_kb": (
                        size_mb * 1000.0 if product_type == "roi_context" else roi_size * max(context_multiplier, 1.0) * 1000.0
                    ),
                }
            )
            return size_mb, diagnostics

        return profile.base_size_mb, {
            "payload_size_source": "fallback",
            "estimated_target_count": target_count,
            "estimated_roi_count": _target_count_from_detection(detection_result),
            "estimated_bbox_area": _as_float(label_summary.get("bbox_area_mean"), 0.0),
            "estimated_roi_size_kb": profile.base_size_mb * 1000.0,
            "estimated_roi_context_size_kb": profile.base_size_mb * 1000.0,
        }

    if detection_result is None:
        return profile.base_size_mb, {
            "payload_size_source": "fallback",
            "estimated_target_count": 0,
            "estimated_roi_count": 0,
            "estimated_bbox_area": 0.0,
            "estimated_roi_size_kb": profile.base_size_mb * 1000.0,
            "estimated_roi_context_size_kb": profile.base_size_mb * 1000.0,
        }

    roi_count = len(build_roi_regions(detection_result))
    if product_type == "metadata":
        size = profile.base_size_mb + 0.02 * roi_count
    elif product_type == "roi":
        size = profile.base_size_mb + 0.18 * roi_count
    elif product_type == "roi_context":
        size = profile.base_size_mb + 0.35 * roi_count
    else:
        size = profile.base_size_mb
    return size, {
        "payload_size_source": "legacy_roi_count",
        "estimated_target_count": roi_count,
        "estimated_roi_count": roi_count,
        "estimated_bbox_area": 0.0,
        "estimated_roi_size_kb": size * 1000.0 if product_type == "roi" else 0.0,
        "estimated_roi_context_size_kb": size * 1000.0 if product_type == "roi_context" else 0.0,
    }


def _estimate_payload_size_mb(policy_name: str, detection_result: DetectionResult) -> float:
    size_mb, _diagnostics = estimate_payload_size_with_diagnostics(policy_name, detection_result)
    return size_mb


def estimate_payload_encoding_cost(
    policy_name: str,
    payload_mb: float,
    payload_profiles_cfg: dict[str, object] | None = None,
) -> tuple[float, float]:
    if not calibrated_profile_enabled(payload_profiles_cfg):
        return 0.0, 0.0
    product_type, _residual_product_type = resolve_payload_plan(policy_name)
    profile = get_payload_policy_profile(product_type, payload_profiles_cfg)
    params = dict(profile.parameters)
    defaults = _calibrated_defaults(payload_profiles_cfg)
    latency_per_mb = _as_float(defaults.get("encoding_latency_s_per_mb"), 0.18)
    energy_per_mb = _as_float(defaults.get("encoding_energy_j_per_mb"), 2.4)
    latency = payload_mb * latency_per_mb * _as_float(params.get("encoding_latency_scale"), 1.0)
    energy = payload_mb * energy_per_mb * _as_float(params.get("encoding_energy_scale"), 1.0)
    return latency, energy


def estimate_raw_information_gain(policy_name: str, payload_profiles_cfg: dict[str, object] | None = None) -> float:
    product_type, _residual_product_type = resolve_payload_plan(policy_name)
    profile = get_payload_policy_profile(product_type, payload_profiles_cfg)
    if not calibrated_profile_enabled(payload_profiles_cfg):
        return profile.information_level
    params = dict(profile.parameters)
    if product_type == "defer":
        return 0.0
    raw = _as_float(params.get("raw_info_gain", params.get("information_factor")), profile.information_level)
    if product_type == "roi_context":
        roi_raw = estimate_raw_information_gain("roi", payload_profiles_cfg)
        cap = _as_float(params.get("marginal_info_gain_cap_relative_to_roi"), 0.10)
        raw = min(raw, roi_raw + cap)
    return _clamp(raw, 0.0, 1.0)


def generate_payload(
    task: TaskSample,
    detection_result: DetectionResult,
    payload_policy: str,
    slot_index: int,
    high_value_threshold: float = 0.75,
    payload_profiles_cfg: dict[str, object] | None = None,
    delivered_products: set[str] | None = None,
) -> GeneratedPayload:
    """Instantiate the selected payload."""

    product_type, residual_product_type = resolve_payload_plan(payload_policy)
    profile = get_payload_policy_profile(product_type, payload_profiles_cfg)
    payload_mb, diagnostics = estimate_payload_size_with_diagnostics(
        payload_policy,
        detection_result=detection_result,
        task=task,
        payload_profiles_cfg=payload_profiles_cfg,
    )
    roi_count = len(build_roi_regions(detection_result))
    if calibrated_profile_enabled(payload_profiles_cfg):
        information_content = estimate_raw_information_gain(payload_policy, payload_profiles_cfg)
    else:
        information_bonus = min(0.25, 0.05 * roi_count)
        information_content = min(1.0, profile.information_level + information_bonus)
    product_requirement = task.product_requirement
    effective_task_value = task_effective_value(task)
    delta_u_i = delta_information_utility(product_type, delivered_products or set(), product_requirement)
    residual_remaining = residual_information_utility(
        set(delivered_products or set()) | ({product_type} if product_type != "defer" else set()),
        product_requirement,
    )
    priority_score = (
        0.60 * effective_task_value + 0.25 * (1.0 - min(task.deadline_s / 120.0, 1.0)) + 0.15 * task.uncertainty
    ) * profile.priority_weight
    return GeneratedPayload(
        task_id=task.task_id,
        payload_policy=payload_policy,
        payload_mb=payload_mb,
        information_content=profile.information_level if payload_policy == "defer" else information_content,
        priority_score=min(1.0, priority_score),
        value_score=effective_task_value,
        task_value=effective_task_value,
        uncertainty=task.uncertainty,
        remaining_utility=effective_task_value * delta_u_i,
        is_metadata=(product_type == "metadata"),
        is_high_value=(effective_task_value >= high_value_threshold),
        deadline_s=task.deadline_s,
        created_slot=slot_index,
        metadata={
            "roi_count": roi_count,
            "target_flag": detection_result.target_flag,
            "detection_quality": detection_result.estimated_quality,
            "task_value": effective_task_value,
            "value_level": getattr(task, "value_level", ""),
            "product_requirement": product_requirement,
            "residual_utility_remaining": residual_remaining,
            **diagnostics,
        },
        payload_plan=payload_policy,
        product_type=product_type,
        residual_product_type=residual_product_type,
        staged_parent_task_id=task.task_id if residual_product_type else "",
        product_requirement=product_requirement,
        incremental_utility_estimate=effective_task_value * delta_u_i,
        incremental_information_utility=delta_u_i,
        duplicate_product_attempt=product_type in set(delivered_products or set()),
        zero_increment_product=(delta_u_i <= 1.0e-12),
        duplicate_identity_key=f"{task.task_id}:{product_type}",
    )


def generate_residual_payload(
    task: TaskSample,
    detection_result: DetectionResult,
    residual_product_type: str,
    slot_index: int,
    *,
    parent_payload_plan: str = "",
    high_value_threshold: float = 0.75,
    payload_profiles_cfg: dict[str, object] | None = None,
    delivered_products: set[str] | None = None,
    remaining_deadline_s: float | None = None,
    continue_request_priority_bonus: float = 0.0,
    followup_trigger: str = "passive_virtual_continue_request",
) -> GeneratedPayload:
    """Instantiate a residual product after a passive or proactive trigger."""

    if residual_product_type not in {"roi", "roi_context"}:
        residual_product_type = "roi"
    payload = generate_payload(
        task=task,
        detection_result=detection_result,
        payload_policy=residual_product_type,
        slot_index=slot_index,
        high_value_threshold=high_value_threshold,
        payload_profiles_cfg=payload_profiles_cfg,
        delivered_products=delivered_products,
    )
    payload.payload_plan = parent_payload_plan or f"metadata_then_{residual_product_type}"
    payload.residual_product_type = ""
    payload.staged_parent_task_id = task.task_id
    payload.is_residual = True
    payload.duplicate_identity_key = f"{task.task_id}:{payload.product_type}"
    payload.continue_request_priority_bonus = float(continue_request_priority_bonus)
    payload.followup_trigger = followup_trigger
    if remaining_deadline_s is not None:
        payload.deadline_s = max(0.0, float(remaining_deadline_s))
    payload.metadata = {
        **payload.metadata,
        "residual_followup": True,
        "followup_trigger": followup_trigger,
        "parent_payload_plan": payload.payload_plan,
        "duplicate_identity_key": payload.duplicate_identity_key,
    }
    return payload
