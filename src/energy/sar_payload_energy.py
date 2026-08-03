"""Common SAR acquisition-to-SOC accounting for frozen-method diagnostics.

The module identifies physical acquisitions independently of online methods.
It deliberately does not alter task objects, solar input, platform base power,
thermal state, compute energy or communication energy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class SarPayloadConfig:
    sar_payload_enabled: bool = False
    sar_active_power_w: float = 0.0
    sar_active_duration_s: float = 10.0
    charge_once_per_acquisition: bool = True

    @property
    def energy_per_acquisition_j(self) -> float:
        if not self.sar_payload_enabled:
            return 0.0
        return max(0.0, self.sar_active_power_w) * max(0.0, self.sar_active_duration_s)

    @property
    def energy_per_acquisition_wh(self) -> float:
        return self.energy_per_acquisition_j / 3600.0


def _metadata(task: Mapping[str, Any]) -> dict[str, Any]:
    raw = task.get("metadata_json", task.get("metadata", {}))
    if isinstance(raw, Mapping):
        return dict(raw)
    if raw in (None, ""):
        return {}
    return dict(json.loads(str(raw)))


def acquisition_identity(task: Mapping[str, Any]) -> tuple[str, str]:
    """Return stable physical-acquisition identity with an audited fallback."""
    metadata = _metadata(task)
    for key in ("acquisition_id", "observation_id", "task_group_id"):
        value = task.get(key, metadata.get(key))
        if value not in (None, ""):
            return str(value), key
    value = metadata.get("product_identifier")
    if value not in (None, ""):
        return str(value), "metadata.product_identifier"
    for key in ("scene_id", "image_path", "task_id"):
        value = task.get(key)
        if value not in (None, ""):
            return str(value), key
    raise ValueError("task has no stable acquisition identity")


def build_acquisition_schedule(
    tasks: Iterable[Mapping[str, Any]],
    config: SarPayloadConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    materialized = [dict(task) for task in tasks]
    grouped: dict[str, list[tuple[dict[str, Any], str]]] = {}
    if config.charge_once_per_acquisition:
        for task in materialized:
            acquisition_id, source = acquisition_identity(task)
            grouped.setdefault(acquisition_id, []).append((task, source))
    else:
        for task in materialized:
            grouped[str(task["task_id"])] = [(task, "task_id:charge_each_task")]

    event_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for acquisition_id, members in sorted(grouped.items()):
        ordered = sorted(
            members,
            key=lambda item: (int(float(item[0]["arrival_slot"])), str(item[0]["task_id"])),
        )
        first_task, identity_source = ordered[0]
        event_slot = int(float(first_task["arrival_slot"]))
        event_time_s = float(first_task.get("arrival_time_s", event_slot * 20.0))
        event_rows.append(
            {
                "acquisition_id": acquisition_id,
                "identity_source": identity_source,
                "event_slot": event_slot,
                "event_time_s": event_time_s,
                "derived_task_count": len(ordered),
                "first_task_id": str(first_task["task_id"]),
                "sar_payload_enabled": int(config.sar_payload_enabled),
                "sar_active_power_w": config.sar_active_power_w,
                "sar_active_duration_s": config.sar_active_duration_s,
                "sar_energy_j": config.energy_per_acquisition_j,
                "sar_energy_wh": config.energy_per_acquisition_wh,
            }
        )
        for index, (task, member_source) in enumerate(ordered):
            mapping_rows.append(
                {
                    "task_id": str(task["task_id"]),
                    "acquisition_id": acquisition_id,
                    "identity_source": member_source,
                    "task_arrival_slot": int(float(task["arrival_slot"])),
                    "event_slot": event_slot,
                    "is_charge_trigger_task": int(index == 0),
                    "derived_task_count": len(ordered),
                }
            )
    event_rows.sort(key=lambda row: (int(row["event_slot"]), str(row["acquisition_id"])))
    mapping_rows.sort(key=lambda row: str(row["task_id"]))
    return mapping_rows, event_rows


def schedule_hash(events: Iterable[Mapping[str, Any]]) -> str:
    payload = [
        (
            str(row["acquisition_id"]),
            int(row["event_slot"]),
            float(row["sar_energy_j"]),
        )
        for row in events
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SarSocDebitController:
    """Apply one precomputed SAR debit per real observation event."""

    def __init__(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        battery_capacity_j: float,
        discharge_efficiency: float,
        battery_update: Callable[..., float],
        method: str,
        profile: str,
    ) -> None:
        event_list = [dict(row) for row in events]
        self.events_by_slot = {int(row["event_slot"]): row for row in event_list}
        if len(self.events_by_slot) != len(event_list):
            raise ValueError("multiple physical acquisitions share a debit slot")
        self.battery_capacity_j = float(battery_capacity_j)
        self.discharge_efficiency = float(discharge_efficiency)
        self.battery_update = battery_update
        self.method = method
        self.profile = profile
        self.charged_slots: set[int] = set()
        self.trace: list[dict[str, Any]] = []

    def debit_value(self, battery_soc: float, slot: int) -> float:
        event = self.events_by_slot.get(int(slot))
        if event is None or int(slot) in self.charged_slots:
            return float(battery_soc)
        energy_j = float(event["sar_energy_j"])
        before = float(battery_soc)
        after = self.battery_update(
            battery_soc=before,
            solar_energy_j=0.0,
            base_power_w=0.0,
            slot_seconds=0.0,
            compute_energy_j=energy_j,
            comm_energy_j=0.0,
            battery_capacity_j=self.battery_capacity_j,
            charge_efficiency=1.0,
            discharge_efficiency=self.discharge_efficiency,
        )
        self.charged_slots.add(int(slot))
        self.trace.append(
            {
                "method": self.method,
                "profile": self.profile,
                **event,
                "soc_before_sar": before,
                "soc_after_sar": after,
                "soc_delta": after - before,
            }
        )
        return after

    def debit_resource_state(self, resource_state: Any, slot: int) -> None:
        resource_state.battery_soc = self.debit_value(resource_state.battery_soc, slot)

    def debit_attribute(self, owner: Any, attribute: str, slot: int) -> None:
        setattr(owner, attribute, self.debit_value(getattr(owner, attribute), slot))
