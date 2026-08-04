"""通信能耗模型。"""

from __future__ import annotations

def compute_comm_energy_j(
    comm_power_w: float,
    tx_time_s: float,
    link_quality: float,
    alpha_gamma: float,
) -> float:
    """计算通信能耗。

    E_comm,t = P_comm * tx_time * zeta(gamma_t)
    """

    overhead = 1.0 + alpha_gamma * (1.0 - link_quality)
    return max(0.0, comm_power_w * tx_time_s * overhead)
