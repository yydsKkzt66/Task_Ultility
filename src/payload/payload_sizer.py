"""根据检测结果和策略生成实际 payload。"""

from __future__ import annotations

from src.common.types import DetectionResult, GeneratedPayload, TaskSample
from src.payload.payload_policy import get_payload_policy_profile
from src.payload.roi_builder import build_roi_regions


def _estimate_payload_size_mb(policy_name: str, detection_result: DetectionResult) -> float:
    """估计实际负载大小。

    该函数只依赖 DetectionResult 与 payload policy，
    不感知 detector backend，从而保证模块边界清晰。
    """

    profile = get_payload_policy_profile(policy_name)
    if policy_name == "defer" or detection_result.skipped:
        return 0.0

    roi_count = len(build_roi_regions(detection_result))
    if policy_name == "metadata":
        return profile.base_size_mb + 0.02 * roi_count
    if policy_name == "roi":
        return profile.base_size_mb + 0.18 * roi_count
    if policy_name == "roi_context":
        return profile.base_size_mb + 0.35 * roi_count
    return profile.base_size_mb


def generate_payload(
    task: TaskSample,
    detection_result: DetectionResult,
    payload_policy: str,
    slot_index: int,
) -> GeneratedPayload:
    """实例化实际 payload。"""

    profile = get_payload_policy_profile(payload_policy)
    payload_mb = _estimate_payload_size_mb(payload_policy, detection_result)
    roi_count = len(build_roi_regions(detection_result))
    information_bonus = min(0.25, 0.05 * roi_count)
    information_content = min(1.0, profile.information_level + information_bonus)
    priority_score = 0.60 * task.task_value + 0.25 * (1.0 - min(task.deadline_s / 120.0, 1.0)) + 0.15 * task.uncertainty
    return GeneratedPayload(
        task_id=task.task_id,
        payload_policy=payload_policy,
        payload_mb=payload_mb,
        information_content=0.05 if payload_policy == "defer" else information_content,
        priority_score=min(1.0, priority_score),
        value_score=task.task_value,
        deadline_s=task.deadline_s,
        created_slot=slot_index,
        metadata={"roi_count": roi_count, "target_flag": detection_result.target_flag},
    )
