"""检测器抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.types import DetectionResult, ProfileMetrics, TaskSample


class DetectorBase(ABC):
    """统一 detector 接口。"""

    @abstractmethod
    def detect(self, task: TaskSample, profile: ProfileMetrics) -> DetectionResult:
        """执行检测并返回统一 DetectionResult。"""
