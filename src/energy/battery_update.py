"""鐢垫睜涓?AI 鑳介噺棰勭畻鏇存柊銆?"""

from __future__ import annotations


def update_battery_soc(
    battery_soc: float,
    solar_energy_j: float,
    base_power_w: float,
    slot_seconds: float,
    compute_energy_j: float,
    comm_energy_j: float,
    battery_capacity_j: float,
    charge_efficiency: float = 1.0,
    discharge_efficiency: float = 1.0,
) -> float:
    """鏇存柊鐢垫睜鑽风數鐘舵€併€?
    鍦ㄦ湁鍏夌収鏃舵寜 charge efficiency 璁＄畻鍏呯數锛屽湪鏈夎礋杞戒簭绌烘椂鎸?discharge efficiency
    璁＄畻鏀剧數锛屼粠鑰屼綋鐜?multi-orbit / multi-day 鐨勫厖鏀剧數鍔ㄦ€併€?"""

    base_energy_j = base_power_w * slot_seconds
    load_energy_j = base_energy_j + compute_energy_j + comm_energy_j
    net_input_j = solar_energy_j - load_energy_j

    if net_input_j >= 0.0:
        delta_energy_j = net_input_j * max(charge_efficiency, 1.0e-9)
    else:
        delta_energy_j = net_input_j / max(discharge_efficiency, 1.0e-9)

    next_value = battery_soc + delta_energy_j / max(battery_capacity_j, 1.0e-9)
    return max(0.0, min(1.0, next_value))


def update_ai_energy_budget(
    ai_energy_budget: float,
    compute_energy_j: float,
    E_AI_ref: float,
    recharge: float,
) -> float:
    """鏇存柊 AI 棰勭畻鐘舵€併€?"""

    next_value = ai_energy_budget - compute_energy_j / max(E_AI_ref, 1.0e-9) + recharge
    return max(0.0, min(1.0, next_value))
