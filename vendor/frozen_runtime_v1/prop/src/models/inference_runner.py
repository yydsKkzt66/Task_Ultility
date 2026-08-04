"""推理执行器。"""

from __future__ import annotations

from src.common.types import DetectionResult, JointAction, TaskSample
from src.models.model_registry import ModelRegistry


class InferenceRunner:
    """根据选择动作执行推理或跳过。"""

    def __init__(self, model_registry: ModelRegistry) -> None:
        self.model_registry = model_registry

    def run(self, task: TaskSample, action: JointAction) -> DetectionResult:
        """执行所选模型。

        对 `delay/store` 动作，显式跳过重模型执行，并返回一个统一的“空检测结果”。
        """

        if action.model_name == "delay_store":
            return DetectionResult(
                bboxes=[],
                confidence=0.0,
                target_flag=False,
                roi_regions=[],
                uncertainty=task.uncertainty,
                inference_latency_s=0.0,
                estimated_quality=0.0,
                generated_metadata={"detector": "skip"},
                skipped=True,
                skip_reason="delay/store 动作跳过模型推理。",
            )
        profile = self.model_registry.get_profile(action.model_name)
        detector = self.model_registry.get_detector(action.model_name)
        return detector.detect(task, profile)
