"""保留原路径下的指标接口。"""

from __future__ import annotations

from dataclasses import asdict

from src.common.constants import EPSILON
from src.common.types import TrajectoryRecord


def records_to_dicts(records: list[TrajectoryRecord]) -> list[dict]:
    """把 dataclass 记录转为 dict。"""

    return [asdict(record) for record in records]


def summarize_episode(
    router_name: str,
    records: list[TrajectoryRecord],
    queue_max_mb: float,
    final_battery: float,
    max_thermal: float,
) -> dict[str, float | str]:
    """计算 summary.csv 所需字段。"""

    total_utility = sum(record.observed_reward for record in records)
    total_compute_energy = sum(record.E_comp for record in records)
    total_comm_energy = sum(record.E_comm for record in records)
    total_energy = total_compute_energy + total_comm_energy
    total_generated_payload_mb = sum(record.S_gen_current_mb for record in records)
    total_downlinked_payload_mb = sum(record.S_down_total_mb for record in records)
    delivered_count = sum(1 for record in records if record.delivery_status == "delivered")
    avg_queue_pressure = sum(record.queue_next_mb / max(queue_max_mb, EPSILON) for record in records) / max(len(records), 1)
    utility_per_joule = total_utility / max(total_energy, EPSILON)
    utility_per_downlink_bit = total_utility / max(total_downlinked_payload_mb * 8_000_000.0, EPSILON)
    return {
        "router": router_name,
        "total_mission_utility": total_utility,
        "avg_mission_utility": total_utility / max(len(records), 1),
        "utility_per_joule": utility_per_joule,
        "utility_per_downlink_bit": utility_per_downlink_bit,
        "deadline_satisfaction_rate": delivered_count / max(len(records), 1),
        "total_energy": total_energy,
        "total_compute_energy": total_compute_energy,
        "total_comm_energy": total_comm_energy,
        "total_generated_payload_mb": total_generated_payload_mb,
        "total_downlinked_payload_mb": total_downlinked_payload_mb,
        "avg_queue_pressure": avg_queue_pressure,
        "max_queue_mb": max((record.queue_next_mb for record in records), default=0.0),
        "safety_violations": sum(record.safety_violation for record in records),
        "final_battery": final_battery,
        "max_thermal": max_thermal,
    }
