"""Mock detector。"""

from __future__ import annotations

import random

from src.common.types import DetectionResult, ProfileMetrics, TaskSample
from src.models.detector_base import DetectorBase
from src.models.error_model import estimate_false_alarm_probability, estimate_miss_probability, sample_error_flags


class MockDetector(DetectorBase):
    """使用 profile-calibrated 规则生成 synthetic detection result。"""

    def __init__(self, error_model_cfg: dict[str, float] | None = None) -> None:
        self.error_model_cfg = error_model_cfg

    def detect(self, task: TaskSample, profile: ProfileMetrics) -> DetectionResult:
        seed = f"{task.task_id}:{profile.profile_name}"
        rng = random.Random(seed)
        difficulty_penalty = (1.0 - profile.robustness) * 0.35 * task.difficulty
        uncertainty_penalty = 0.18 * task.uncertainty
        estimated_quality = max(0.05, min(1.0, profile.quality_base - difficulty_penalty - uncertainty_penalty))
        confidence = max(0.05, min(0.99, estimated_quality + rng.uniform(-0.08, 0.08)))
        miss_probability = estimate_miss_probability(task, profile, estimated_quality, self.error_model_cfg)
        false_alarm_probability = estimate_false_alarm_probability(task, estimated_quality, miss_probability, self.error_model_cfg)
        miss_flag, false_alarm_flag = sample_error_flags(seed, miss_probability, false_alarm_probability)
        bbox_count = max(1, int(round(1 + confidence * 3 + task.metadata.get("synthetic_target_density", 1) / 2)))

        bboxes = []
        roi_regions = []
        for index in range(bbox_count):
            x1 = round(0.08 + 0.10 * index, 3)
            y1 = round(0.10 + 0.07 * index, 3)
            x2 = min(0.95, round(x1 + 0.20 + 0.02 * rng.random(), 3))
            y2 = min(0.95, round(y1 + 0.18 + 0.02 * rng.random(), 3))
            box = (x1, y1, x2, y2)
            bboxes.append(box)
            roi_regions.append(box)

        if miss_flag:
            bboxes = []
            roi_regions = []
            bbox_count = 0
            confidence = min(confidence, 0.18)
        elif false_alarm_flag:
            confidence = min(0.55, confidence)

        return DetectionResult(
            bboxes=bboxes,
            confidence=confidence,
            target_flag=(confidence > 0.35 and not miss_flag),
            roi_regions=roi_regions,
            uncertainty=max(0.0, min(1.0, 1.0 - estimated_quality + 0.2 * task.uncertainty)),
            inference_latency_s=profile.latency_s,
            estimated_quality=estimated_quality,
            generated_metadata={
                "detector": "mock",
                "bbox_count": bbox_count,
                "scene_id": task.scene_id,
                "miss_flag": miss_flag,
                "false_alarm_flag": false_alarm_flag,
            },
            miss_probability=miss_probability,
            false_alarm_probability=false_alarm_probability,
            miss_flag=miss_flag,
            false_alarm_flag=false_alarm_flag,
        )
