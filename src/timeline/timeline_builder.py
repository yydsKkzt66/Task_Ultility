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

    def __init__(self, dataset_cfg: dict[str, object], timeline_cfg: dict[str, object]) -> None:
        self.dataset_cfg = dataset_cfg
        self.timeline_cfg = timeline_cfg

    def build(self) -> list[TimelineStep]:
        num_slots = int(self.timeline_cfg["num_slots"])
        slot_seconds = float(self.timeline_cfg["slot_seconds"])
        tasks = TaskStreamBuilder(self.dataset_cfg).build(num_slots)
        resources = load_synthetic_resource_trace(num_slots)
        links = load_synthetic_link_trace(num_slots, slot_seconds)
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
