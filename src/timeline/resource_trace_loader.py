"""资源 trace 加载器。"""

from __future__ import annotations

import math

from src.common.types import ResourceTrace


def load_synthetic_resource_trace(num_slots: int) -> list[ResourceTrace]:
    """生成 synthetic resource trace。"""

    traces: list[ResourceTrace] = []
    for slot_index in range(num_slots):
        solar_energy_j = 18.0 + 16.0 * max(0.0, math.sin(slot_index / 19.0 + 0.6))
        base_power_w = 2.8 + 0.6 * (0.5 + 0.5 * math.cos(slot_index / 17.0))
        temperature_reference_c = 10.0 + 5.0 * math.sin(slot_index / 15.0)
        traces.append(
            ResourceTrace(
                slot_index=slot_index,
                solar_energy_j=solar_energy_j,
                base_power_w=base_power_w,
                temperature_reference_c=temperature_reference_c,
            )
        )
    return traces
