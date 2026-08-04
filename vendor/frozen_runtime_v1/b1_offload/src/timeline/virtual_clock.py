"""虚拟时钟。"""

from __future__ import annotations


class VirtualClock:
    """为 trace-driven case study 提供统一离散时间槽。"""

    def __init__(self, num_slots: int, slot_seconds: float) -> None:
        self.num_slots = num_slots
        self.slot_seconds = slot_seconds

    def iter_slots(self):
        """按顺序产生时隙编号。"""

        for slot_index in range(self.num_slots):
            yield slot_index
