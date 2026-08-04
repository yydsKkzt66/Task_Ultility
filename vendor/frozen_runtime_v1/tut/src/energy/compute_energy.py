"""计算能耗模型。"""

from __future__ import annotations


def compute_inference_energy_j(estimated_power_w: float, latency_s: float) -> float:
    """计算推理能耗。

    E_comp,t = P_model(m) * L_inf,t
    """

    return max(0.0, estimated_power_w * latency_s)
