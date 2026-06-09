"""Task stream 构建入口。"""

from __future__ import annotations

from src.data.mock_dataset import MockDatasetAdapter
from src.data.sarfish_adapter import SarfishAdapter


class TaskStreamBuilder:
    """根据配置选择数据集适配器并构建统一任务流。"""

    def __init__(self, dataset_cfg: dict[str, object]) -> None:
        self.dataset_cfg = dataset_cfg

    def build(self, max_tasks: int):
        dataset_name = str(self.dataset_cfg.get("dataset_name", "mock")).lower()
        if dataset_name == "mock":
            return MockDatasetAdapter(self.dataset_cfg).build_task_stream(max_tasks)
        if dataset_name == "sarfish":
            return SarfishAdapter(self.dataset_cfg).build_task_stream(max_tasks)
        raise ValueError(f"不支持的数据集类型: {dataset_name}")
