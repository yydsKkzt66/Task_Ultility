"""热状态更新。"""

from __future__ import annotations


def update_thermal_pressure(
    thermal_pressure: float,
    compute_energy_j: float,
    comm_energy_j: float,
    cooling: float,
) -> float:
    """更新热压力。

    当前版本使用简化的一阶离散递推，重点是保证接口清晰与可替换。
    """

    heat_from_compute = 0.0025 * compute_energy_j
    heat_from_comm = 0.0015 * comm_energy_j
    next_value = thermal_pressure + heat_from_compute + heat_from_comm - cooling
    return max(0.0, min(1.0, next_value))
