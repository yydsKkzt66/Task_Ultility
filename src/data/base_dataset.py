"""数据集适配器抽象层。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.types import TaskSample


class DatasetAdapter(ABC):
    """所有数据集适配器的统一抽象接口。"""

    @abstractmethod
    def build_task_stream(self, max_tasks: int) -> list[TaskSample]:
        """生成统一的 TaskSample 序列。"""
