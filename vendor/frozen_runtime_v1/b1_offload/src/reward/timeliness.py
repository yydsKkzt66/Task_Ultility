"""时效性计算。"""

from __future__ import annotations


def compute_timeliness(total_latency_s: float, deadline_s: float) -> float:
    """计算 deadline timeliness。"""

    if deadline_s <= 1.0e-9:
        return 0.0
    return max(0.0, min(1.0, 1.0 - total_latency_s / deadline_s))
