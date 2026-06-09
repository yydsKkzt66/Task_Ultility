"""模型注册表。"""

from __future__ import annotations

from src.common.types import ProfileMetrics
from src.models.mock_detector import MockDetector
from src.models.profiling_store import ProfilingStore
from src.models.yolo_detector import YoloDetector


class ModelRegistry:
    """管理抽象模型层级、后端 detector 与 profiling 之间的映射关系。"""

    def __init__(self, model_pool_cfg: dict[str, object]) -> None:
        self.model_pool_cfg = model_pool_cfg
        self.profiling_store = ProfilingStore(model_pool_cfg.get("profile_overrides"))
        self.error_model_cfg = model_pool_cfg.get("error_model")
        self._model_table = {}

        for item in model_pool_cfg.get("models", []):
            self._model_table[item["name"]] = dict(item)

        self._model_table["delay_store"] = {
            "name": "delay_store",
            "backend": "mock",
            "profile_name": "defer",
        }

    def get_profile(self, model_name: str) -> ProfileMetrics:
        """根据抽象模型名获取 profile。"""

        spec = self._model_table[model_name]
        return self.profiling_store.get(spec["profile_name"])

    def get_model_path(self, model_name: str) -> str | None:
        """返回模型路径占位字段，供未来真实模型池接入使用。"""

        spec = self._model_table[model_name]
        return spec.get("model_path") or spec.get("path")

    def get_detector(self, model_name: str):
        """按 backend 返回 detector。"""

        backend = self._model_table[model_name]["backend"]
        if backend == "mock":
            return MockDetector(self.error_model_cfg)
        if backend == "yolo":
            return YoloDetector()
        raise ValueError(f"未知 detector backend: {backend}")
