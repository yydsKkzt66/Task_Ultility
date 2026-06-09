"""总 reward 计算。"""

from __future__ import annotations

from src.common.types import ContextVector, ObservedFeedback, RewardBreakdown, TaskSample
from src.reward.delivery import compute_delivery_factor
from src.reward.high_value_error_penalty import compute_high_value_error_penalty
from src.reward.mission_utility import compute_dynamic_energy_weight, compute_information_sufficiency
from src.reward.safety_penalty import compute_safety_penalty
from src.reward.timeliness import compute_timeliness


def compute_detection_quality_factor(detection_quality: float) -> float:
    """将检测质量裁剪到 reward 使用的合法范围。"""

    return max(0.0, min(1.0, detection_quality))


def compute_energy_penalty(total_energy_j: float, E_ref: float) -> float:
    """归一化总能耗。"""

    return total_energy_j / max(E_ref, 1.0e-9)


def compute_latency_penalty(total_latency_s: float, deadline_s: float, epsilon: float) -> float:
    """归一化端到端时延。"""

    return total_latency_s / max(deadline_s, epsilon)


def compute_queue_penalty(queue_next_mb: float, queue_max_mb: float) -> float:
    """归一化队列压力。"""

    return queue_next_mb / max(queue_max_mb, 1.0e-9)


def compute_reward(
    task: TaskSample,
    context: ContextVector,
    payload_information: float,
    feedback: ObservedFeedback,
    reward_cfg: dict[str, float],
    simulator_cfg: dict[str, float],
) -> RewardBreakdown:
    """计算 observed reward。

    该函数必须使用 observed feedback，而不是 predicted reward 来更新 bandit/router。
    """

    detection_quality = compute_detection_quality_factor(feedback.detection_quality)
    info_sufficiency = compute_information_sufficiency(context, payload_information, reward_cfg)
    timeliness = compute_timeliness(feedback.total_latency_s, task.deadline_s)
    delivery_factor = compute_delivery_factor(
        feedback.delivery_status,
        delivery_beta=float(simulator_cfg["delivery_beta"]),
        deferred_credit=float(reward_cfg["deferred_delivery_credit"]),
    )
    dynamic_energy_weight = compute_dynamic_energy_weight(context, reward_cfg)

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

    reward = (
        context.V_t * detection_quality * info_sufficiency * timeliness * delivery_factor
        - dynamic_energy_weight * energy_penalty
        - float(reward_cfg["lambda_L"]) * latency_penalty
        - float(reward_cfg["lambda_Q"]) * queue_penalty
        - safety_penalty
        - mission_error_penalty
    )
    return RewardBreakdown(
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
        reward=reward,
    )
