"""SARFish dataset adapter placeholder."""

from __future__ import annotations

from src.data.base_dataset import DatasetAdapter


class SarfishAdapter(DatasetAdapter):
    """Reserved interface for future SARFishSample-GRD integration."""

    def __init__(self, dataset_cfg: dict[str, object]) -> None:
        self.dataset_cfg = dataset_cfg

    def build_task_stream(
        self,
        num_slots: int,
        slot_seconds: float,
        task_stream_cfg: dict[str, object] | None = None,
    ):
        _ = (num_slots, slot_seconds, task_stream_cfg)
        raise NotImplementedError("SarfishAdapter is a placeholder; use MockDatasetAdapter in the current scaffold.")
