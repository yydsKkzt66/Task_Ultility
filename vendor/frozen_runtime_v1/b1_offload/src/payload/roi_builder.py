"""ROI 构建辅助函数。"""

from __future__ import annotations

from src.common.types import DetectionResult


def build_roi_regions(detection_result: DetectionResult) -> list[tuple[float, float, float, float]]:
    """从检测结果中抽取 ROI。

    当前 mock 版本直接复用 detector 给出的 roi_regions。
    后续若接入真实图像裁剪，这里可以替换为更细的 ROI 扩展与上下文边界逻辑。
    """

    return list(detection_result.roi_regions)
