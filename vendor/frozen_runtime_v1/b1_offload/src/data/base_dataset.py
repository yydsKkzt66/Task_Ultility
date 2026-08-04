"""Dataset adapter abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.types import TaskSample


class DatasetAdapter(ABC):
    """Unified interface for dataset-backed task stream builders."""

    @abstractmethod
    def build_task_stream(
        self,
        num_slots: int,
        slot_seconds: float,
        task_stream_cfg: dict[str, object] | None = None,
    ) -> list[TaskSample | None]:
        """Build a slot-aligned task stream."""
