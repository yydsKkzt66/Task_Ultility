"""通信延迟与链路开销计算。"""

from __future__ import annotations

from src.common.types import LinkTrace
from src.link.link_model import estimate_rate_mb_per_s


def estimate_comm_latency_s(payload_mb: float, link_trace: LinkTrace, epsilon: float) -> float:
    """估计通信时延。"""

    rate = estimate_rate_mb_per_s(link_trace)
    if rate <= epsilon:
        return 0.0
    return payload_mb / rate


def estimate_link_overhead(link_trace: LinkTrace, alpha_gamma: float) -> float:
    """估计弱链路带来的额外发射开销。"""

    return 1.0 + alpha_gamma * (1.0 - link_trace.link_quality)
