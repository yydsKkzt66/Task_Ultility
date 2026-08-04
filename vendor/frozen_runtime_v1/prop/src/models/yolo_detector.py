"""YOLO detector 适配器占位实现。"""

from __future__ import annotations

from src.common.types import DetectionResult, ProfileMetrics, TaskSample
from src.models.detector_base import DetectorBase


class YoloDetector(DetectorBase):
    """可选的 YOLO-family 适配器。

    注意：当前 scaffold 默认不依赖 ultralytics。
    如果未来要接入真实模型，这里应负责完成后端推理并返回统一 DetectionResult。
    """

    def __init__(self) -> None:
        try:
            import ultralytics  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 依赖缺失分支用于运行期提示
            raise ImportError(
                "YoloDetector 需要可选依赖 ultralytics。当前 mock demo 不依赖该包，请继续使用 mock backend。"
            ) from exc

    def detect(self, task: TaskSample, profile: ProfileMetrics) -> DetectionResult:
        raise NotImplementedError("当前版本未集成真实 YOLO 推理，请改用 MockDetector。")
