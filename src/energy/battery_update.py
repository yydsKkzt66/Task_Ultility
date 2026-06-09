"""电池与 AI 能量预算更新。"""

from __future__ import annotations


def update_battery_soc(
    battery_soc: float,
    solar_energy_j: float,
    base_power_w: float,
    slot_seconds: float,
    compute_energy_j: float,
    comm_energy_j: float,
    battery_capacity_j: float,
) -> float:
    """更新电池荷电状态。

    B_next = clip(B_current + (E_solar - E_base - E_comp - E_comm) / C_bat, 0, 1)
    """

    base_energy_j = base_power_w * slot_seconds
    next_value = battery_soc + (solar_energy_j - base_energy_j - compute_energy_j - comm_energy_j) / max(battery_capacity_j, 1.0e-9)
    return max(0.0, min(1.0, next_value))


def update_ai_energy_budget(
    ai_energy_budget: float,
    compute_energy_j: float,
    E_AI_ref: float,
    recharge: float,
) -> float:
    """更新 AI 预算状态。"""

    next_value = ai_energy_budget - compute_energy_j / max(E_AI_ref, 1.0e-9) + recharge
    return max(0.0, min(1.0, next_value))
