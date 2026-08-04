"""通信状态辅助函数。"""

from __future__ import annotations

from src.common.types import CommState, LinkTrace


def build_initial_comm_state(simulator_cfg: dict[str, float]) -> CommState:
    """根据配置创建初始通信状态。"""

    return CommState(
        queue_mb=float(simulator_cfg["queue_init_mb"]),
        queue_max_mb=float(simulator_cfg["queue_max_mb"]),
        queue_item_count=0,
        effective_opportunity=0.0,
    )


def compute_effective_downlink_opportunity(link_trace: LinkTrace) -> float:
    """把 contact / SNR / rate / remaining window 压缩成一个 [0, 1] 机会量。

    这正是 9D context 里的 C_t：保留紧凑表征，而非把全部链路物理量都直接塞进上下文。
    """

    if not link_trace.in_contact:
        return 0.0
    window_factor = min(link_trace.remaining_window_s / 10.0, 1.0)
    throughput_factor = min(link_trace.throughput_mbps / 12.0, 1.0)
    return max(0.0, min(1.0, 0.40 * link_trace.link_quality + 0.35 * throughput_factor + 0.25 * window_factor))
