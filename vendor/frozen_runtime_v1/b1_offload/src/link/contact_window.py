"""接触窗口相关计算。"""

from __future__ import annotations

from src.common.types import LinkTrace


def compute_available_capacity_mb(link_trace: LinkTrace, slot_seconds: float, downlink_efficiency: float = 1.0) -> float:
    """计算当前时隙的理论下传容量。

    capacity_mb = throughput_mbps * slot_seconds / 8
    """

    if not link_trace.in_contact:
        return 0.0
    return max(0.0, link_trace.throughput_mbps * slot_seconds * downlink_efficiency / 8.0)
