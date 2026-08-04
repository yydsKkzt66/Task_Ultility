"""资源状态辅助函数。"""

from __future__ import annotations

from src.common.types import ResourceState


def build_initial_resource_state(simulator_cfg: dict[str, float]) -> ResourceState:
    """根据配置创建初始资源状态。"""

    return ResourceState(
        battery_soc=float(simulator_cfg["battery_init"]),
        ai_energy_budget=float(simulator_cfg["ai_budget_init"]),
        thermal_pressure=float(simulator_cfg["thermal_init"]),
    )
