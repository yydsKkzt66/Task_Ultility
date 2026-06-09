"""链路辅助计算。"""

from __future__ import annotations

from src.common.types import LinkTrace


def estimate_rate_mb_per_s(link_trace: LinkTrace) -> float:
    """将 Mbps 转换为 MB/s。"""

    if not link_trace.in_contact:
        return 0.0
    return max(0.0, link_trace.throughput_mbps / 8.0)
