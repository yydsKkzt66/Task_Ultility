"""鎬?reward 璁＄畻銆?"""

from __future__ import annotations

from src.common.types import ContextVector, ObservedFeedback, RewardBreakdown, TaskSample
from src.reward.delivery import compute_delivery_factor
from src.reward.high_value_error_penalty import compute_high_value_error_penalty
from src.reward.mission_utility import (
    compute_dynamic_energy_weight,
    compute_information_requirement_from_context,
    compute_information_sufficiency,
    compute_quality_requirement,
)
from src.reward.safety_penalty import compute_safety_penalty
from src.reward.timeliness import compute_timeliness
from src.reward.utility_calibration import (
    assert_lifecycle_utility_reserved,
    calibrate_value,
    get_utility_calibration,
    lifecycle_utility_active,
)
from src.payload.information_sufficiency import compute_net_incremental_utility


def compute_detection_quality_factor(detection_quality: float) -> float:
    """灏嗘娴嬭川閲忚鍓埌 reward 浣跨敤鐨勫悎娉曡寖鍥淬€?"""

    return max(0.0, min(1.0, detection_quality))


def compute_energy_penalty(total_energy_j: float, E_ref: float) -> float:
    """褰掍竴鍖栨€昏兘鑰椼€?"""

    return total_energy_j / max(E_ref, 1.0e-9)


def compute_latency_penalty(total_latency_s: float, deadline_s: float, epsilon: float) -> float:
    """褰掍竴鍖栫鍒扮鏃跺欢銆?"""

    return total_latency_s / max(deadline_s, epsilon)


def compute_queue_penalty(queue_next_mb: float, queue_max_mb: float) -> float:
    """褰掍竴鍖栭槦鍒楀帇鍔涖€?"""

    return queue_next_mb / max(queue_max_mb, 1.0e-9)


def _reward_cfg_value(reward_cfg: dict[str, float], primary: str, fallback: str, default: float = 0.0) -> float:
    if primary in reward_cfg:
        return float(reward_cfg[primary])
    if fallback in reward_cfg:
        return float(reward_cfg[fallback])
    return float(default)


def compute_reward(
    task: TaskSample,
    context: ContextVector,
    payload_information: float,
    feedback: ObservedFeedback,
    reward_cfg: dict[str, float],
    simulator_cfg: dict[str, float],
) -> RewardBreakdown:
    """璁＄畻 observed reward銆?
    璇ュ嚱鏁板繀椤讳娇鐢?observed feedback锛岃€屼笉鏄?predicted reward 鏉ユ洿鏂?bandit/router銆?"""

    assert_lifecycle_utility_reserved(reward_cfg)
    calibration = get_utility_calibration(reward_cfg)
    value_raw = float(context.V_t)
    value_calibrated = calibrate_value(value_raw, calibration)

    detection_quality = compute_detection_quality_factor(feedback.detection_quality)
    quality_requirement = (
        float(task.quality_requirement)
        if task.quality_requirement is not None
        else compute_quality_requirement(context, reward_cfg)
    )
    information_requirement = (
        float(task.information_requirement)
        if task.information_requirement is not None
        else compute_information_requirement_from_context(context, reward_cfg)
    )
    info_sufficiency = compute_information_sufficiency(context, payload_information, reward_cfg)
    timeliness = compute_timeliness(feedback.total_latency_s, task.deadline_s)
    delivery_factor = compute_delivery_factor(
        feedback.delivery_status,
        delivery_beta=float(simulator_cfg["delivery_beta"]),
        deferred_credit=float(reward_cfg["deferred_delivery_credit"]),
    )
    dynamic_energy_weight = compute_dynamic_energy_weight(context, reward_cfg)
    information_weight = float(reward_cfg.get("information_weight", 1.0))
    delivery_weight = float(reward_cfg.get("delivery_weight", 1.0))
    deadline_miss_penalty = float(reward_cfg.get("deadline_miss_penalty", 0.0))
    high_value_bonus = float(reward_cfg.get("high_value_bonus", 0.0))
    high_value_threshold = float(reward_cfg.get("high_value_threshold", 0.75))

    total_energy = feedback.E_comp + feedback.E_comm
    energy_penalty = compute_energy_penalty(total_energy, float(simulator_cfg["E_ref"]))
    latency_penalty = compute_latency_penalty(
        feedback.total_latency_s,
        task.deadline_s,
        float(simulator_cfg["epsilon"]),
    )
    queue_penalty = compute_queue_penalty(feedback.queue_next_mb, float(simulator_cfg["queue_max_mb"]))
    safety_penalty = compute_safety_penalty(feedback.safety_violation, reward_cfg)
    mission_error_penalty = compute_high_value_error_penalty(
        context=context,
        reward_cfg=reward_cfg,
        miss_signal=feedback.miss_signal,
        false_alarm_signal=feedback.false_alarm_signal,
    )

    effective_utility = (
        value_calibrated
        * detection_quality
        * (information_weight * info_sufficiency)
        * timeliness
        * (delivery_weight * delivery_factor)
    )
    if value_calibrated >= high_value_threshold and feedback.delivery_status == "delivered":
        effective_utility += high_value_bonus * value_calibrated

    mission_utility = max(0.0, value_calibrated)
    resource_penalty_raw = (
        dynamic_energy_weight * energy_penalty
        + _reward_cfg_value(reward_cfg, "lambda_L", "latency_penalty") * latency_penalty
        + _reward_cfg_value(reward_cfg, "lambda_Q", "queue_penalty") * queue_penalty
    )
    quality_gap = max(0.0, quality_requirement - detection_quality)
    information_gap = max(0.0, information_requirement - payload_information)
    quality_gap_penalty_raw = float(reward_cfg.get("quality_gap_penalty", 0.0)) * quality_gap
    information_gap_penalty_raw = float(reward_cfg.get("information_gap_penalty", 0.0)) * information_gap
    quality_gap_penalty = calibration.beta_requirement_penalty * quality_gap_penalty_raw
    information_gap_penalty = calibration.beta_requirement_penalty * information_gap_penalty_raw
    requirement_penalty = quality_gap_penalty + information_gap_penalty
    resource_penalty = calibration.beta_resource_cost * resource_penalty_raw
    deadline_penalty_raw = deadline_miss_penalty if feedback.total_latency_s > task.deadline_s else 0.0
    safety_penalty = calibration.beta_hard_penalty * safety_penalty
    mission_error_penalty = calibration.beta_hard_penalty * mission_error_penalty
    deadline_penalty = calibration.beta_hard_penalty * deadline_penalty_raw
    hard_penalty = safety_penalty + mission_error_penalty + deadline_penalty
    reward = effective_utility - resource_penalty - requirement_penalty - hard_penalty
    gross_incremental_utility = max(0.0, float(feedback.delivered_only_incremental_utility))
    net_incremental_utility = compute_net_incremental_utility(
        gross_incremental_utility,
        resource_cost=resource_penalty,
        safety_penalty=hard_penalty,
    )

    return RewardBreakdown(
        value_raw=value_raw,
        value_calibrated=value_calibrated,
        effective_utility=effective_utility,
        resource_penalty=resource_penalty,
        quality_requirement=quality_requirement,
        information_requirement=information_requirement,
        quality_gap=quality_gap,
        information_gap=information_gap,
        quality_gap_penalty=quality_gap_penalty,
        information_gap_penalty=information_gap_penalty,
        requirement_penalty=requirement_penalty,
        mission_utility=mission_utility,
        detection_quality=detection_quality,
        information_sufficiency=info_sufficiency,
        timeliness=timeliness,
        delivery_factor=delivery_factor,
        dynamic_energy_weight=dynamic_energy_weight,
        energy_penalty=energy_penalty,
        latency_penalty=latency_penalty,
        queue_penalty=queue_penalty,
        safety_penalty=safety_penalty,
        mission_error_penalty=mission_error_penalty,
        deadline_penalty=deadline_penalty,
        hard_penalty=hard_penalty,
        reward=reward,
        gross_incremental_utility=gross_incremental_utility,
        net_incremental_utility=net_incremental_utility,
        resource_cost=resource_penalty,
        net_utility=reward,
        beta_resource_cost=calibration.beta_resource_cost,
        beta_requirement_penalty=calibration.beta_requirement_penalty,
        beta_hard_penalty=calibration.beta_hard_penalty,
        lifecycle_fields_active=lifecycle_utility_active(reward_cfg),
    )
