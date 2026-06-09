"""链路 trace 加载器。"""

from __future__ import annotations

import math

from src.common.types import LinkTrace


def load_synthetic_link_trace(num_slots: int, slot_seconds: float) -> list[LinkTrace]:
    """生成 synthetic link trace。"""

    traces: list[LinkTrace] = []
    for slot_index in range(num_slots):
        window_phase = slot_index % 18
        in_contact = window_phase in {4, 5, 6, 7, 8, 13, 14, 15}
        link_quality = max(0.0, min(1.0, 0.45 + 0.35 * math.sin(slot_index / 8.0 + 0.5)))
        throughput_mbps = (1.5 + 4.5 * link_quality) if in_contact else 0.0
        snr_db = (5.0 + 12.0 * link_quality) if in_contact else -3.0
        remaining_window_s = float((9 - window_phase) * slot_seconds) if in_contact and window_phase <= 8 else (float((16 - window_phase) * slot_seconds) if in_contact else 0.0)
        traces.append(
            LinkTrace(
                slot_index=slot_index,
                in_contact=in_contact,
                snr_db=snr_db,
                throughput_mbps=throughput_mbps,
                link_quality=link_quality if in_contact else 0.0,
                remaining_window_s=max(0.0, remaining_window_s),
            )
        )
    return traces
