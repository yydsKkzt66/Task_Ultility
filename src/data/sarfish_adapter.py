"""SARFish 数据集适配器占位实现。"""

from __future__ import annotations

from src.data.base_dataset import DatasetAdapter


class SarfishAdapter(DatasetAdapter):
    """为未来接入 SARFishSample-GRD 预留接口。

    TODO:
    1. 读取 GRD 图像。
    2. 做 tile 切分。
    3. 读取 bbox 标签并转成统一格式。
    4. 自动生成 task_value / deadline / difficulty / uncertainty。
    """

    def __init__(self, dataset_cfg: dict[str, object]) -> None:
        self.dataset_cfg = dataset_cfg

    def build_task_stream(self, max_tasks: int):
        raise NotImplementedError("SarfishAdapter 仅为占位接口，当前版本请使用 MockDatasetAdapter。")
