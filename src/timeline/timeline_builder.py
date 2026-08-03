"""Timeline 构建器。"""

from __future__ import annotations

from src.common.types import TimelineStep
from src.data.task_stream import TaskStreamBuilder
from src.timeline.link_trace_loader import load_synthetic_link_trace
from src.timeline.resource_trace_loader import load_synthetic_resource_trace
from src.timeline.trace_aligner import align_to_length
from src.timeline.virtual_clock import VirtualClock


class TimelineBuilder:
    """把任务流、资源 trace 和链路 trace 对齐到统一 virtual timeline。"""

    def __init__(
        self,
        dataset_cfg: dict[str, object],
        timeline_cfg: dict[str, object],
        task_stream_cfg: dict[str, object] | None = None,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.timeline_cfg = timeline_cfg
        self.task_stream_cfg = task_stream_cfg or {}

    def build(self) -> list[TimelineStep]:
        slot_seconds = float(self.timeline_cfg.get("slot_seconds", self.task_stream_cfg.get("slot_seconds", 10.0)))
        use_num_days = bool(self.timeline_cfg.get("use_num_days", self.task_stream_cfg.get("use_num_days", False)))
        if use_num_days:
            num_days = float(self.timeline_cfg.get("num_days", self.task_stream_cfg.get("num_days", 1.0)))
            num_slots = int(num_days * 24.0 * 3600.0 / max(slot_seconds, 1.0e-9))
        else:
            num_slots = int(self.timeline_cfg["num_slots"])

        self.timeline_cfg["slot_seconds"] = slot_seconds
        self.timeline_cfg["num_slots"] = num_slots
        self.task_stream_cfg["slot_seconds"] = slot_seconds
        self.task_stream_cfg["num_days"] = self.timeline_cfg.get("num_days", self.task_stream_cfg.get("num_days", 1.0))

        tasks = TaskStreamBuilder(self.dataset_cfg, self.task_stream_cfg).build(num_slots, slot_seconds)
        resources = load_synthetic_resource_trace(num_slots, self.timeline_cfg)
        contact_cfg = self.timeline_cfg.get("contact", {})
        links = load_synthetic_link_trace(num_slots, slot_seconds, contact_cfg if isinstance(contact_cfg, dict) else None)
        tasks = align_to_length(tasks, num_slots)
        resources = align_to_length(resources, num_slots)
        links = align_to_length(links, num_slots)

        timeline: list[TimelineStep] = []
        for slot_index in VirtualClock(num_slots, slot_seconds).iter_slots():
            timeline.append(
                TimelineStep(
                    slot_index=slot_index,
                    slot_seconds=slot_seconds,
                    task=tasks[slot_index],
                    resource_trace=resources[slot_index],
                    link_trace=links[slot_index],
                )
            )
        return timeline
