"""Mock 数据集适配器。"""

from __future__ import annotations

import random

from src.common.types import TaskSample
from src.data.base_dataset import DatasetAdapter
from src.data.difficulty import sample_deadline_s, sample_difficulty, sample_task_value, sample_uncertainty


class MockDatasetAdapter(DatasetAdapter):
    """生成 synthetic task stream。

    该适配器保证在没有真实数据集、没有图像文件和没有检测模型的情况下，
    整个 trace-driven closed-loop case study 仍然可以完整跑通。
    """

    def __init__(self, dataset_cfg: dict[str, object]) -> None:
        self.dataset_cfg = dataset_cfg
        self.seed = int(dataset_cfg.get("seed", 2026))

    def build_task_stream(self, max_tasks: int) -> list[TaskSample]:
        rng = random.Random(self.seed)
        tasks: list[TaskSample] = []
        for slot_index in range(max_tasks):
            task_id = f"task_{slot_index:04d}"
            tasks.append(
                TaskSample(
                    task_id=task_id,
                    image_path=f"mock://scene_{slot_index // 12:03d}/tile_{slot_index:04d}.png",
                    label_path=f"mock://scene_{slot_index // 12:03d}/tile_{slot_index:04d}.txt",
                    scene_id=f"scene_{slot_index // 12:03d}",
                    virtual_time=slot_index,
                    task_value=sample_task_value(slot_index, rng),
                    deadline_s=sample_deadline_s(slot_index, rng),
                    difficulty=sample_difficulty(slot_index, rng),
                    uncertainty=sample_uncertainty(slot_index, rng),
                    metadata={
                        "dataset_name": "mock",
                        "synthetic_target_density": 1 + (slot_index % 3),
                    },
                )
            )
        return tasks
