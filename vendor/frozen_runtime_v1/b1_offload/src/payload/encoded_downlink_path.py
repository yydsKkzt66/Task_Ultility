"""Isolated encoded-active physical path used by propagation sentinels.

The canonical legacy downlink queue does not import this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.payload.fixed_product_compression import EncodedPayload, EncodingState


def _event(payload: EncodedPayload, event_type: str, event_time_s: float, **fields: object) -> dict[str, object]:
    return {
        "payload_id": payload.payload_id,
        "task_id": payload.task_id,
        "product_id": payload.product_id,
        "stage_id": payload.stage_id,
        "product_type": payload.product_type,
        "compression_profile_hash": payload.compression_profile_hash,
        "event_type": event_type,
        "event_time_s": float(event_time_s),
        "state": payload.encoding_status.value,
        **fields,
    }


@dataclass
class EncodedResourceLedger:
    initial_soc_Wh: float
    controllable_energy_Wh: float = 0.0
    thermal_compute_energy_Wh: float = 0.0
    _charged_payload_ids: set[str] = field(default_factory=set)

    @property
    def soc_Wh(self) -> float:
        return self.initial_soc_Wh - self.controllable_energy_Wh

    def charge_encoding(self, payload: EncodedPayload, *, thermal_fraction: float) -> None:
        if payload.payload_id in self._charged_payload_ids:
            raise RuntimeError(f"encoding energy already charged payload_id={payload.payload_id}")
        if not 0.0 <= thermal_fraction <= 1.0:
            raise ValueError("thermal_fraction must be in [0, 1]")
        payload.physical_ready_time_s(module_name="EncodedResourceLedger.charge_encoding")
        self.controllable_energy_Wh += payload.encode_energy_Wh
        self.thermal_compute_energy_Wh += payload.encode_energy_Wh * thermal_fraction
        self._charged_payload_ids.add(payload.payload_id)


class EncodedDownlinkQueue:
    """Queue that retains the exact same EncodedPayload objects by reference."""

    def __init__(self, *, compression_physics_active: bool) -> None:
        if not compression_physics_active:
            raise ValueError("EncodedDownlinkQueue requires compression_physics_active=true")
        self.compression_physics_active = True
        self._payloads: dict[str, EncodedPayload] = {}
        self.lineage_events: list[dict[str, object]] = []
        self.communication_energy_Wh = 0.0
        self.delivered_bytes_from_events = 0

    def _payload(self, payload_id: str) -> EncodedPayload:
        try:
            return self._payloads[payload_id]
        except KeyError as exc:
            raise KeyError(f"unknown encoded payload_id={payload_id}") from exc

    @property
    def queue_bytes(self) -> int:
        return sum(
            payload.physical_remaining_bytes(module_name="EncodedDownlinkQueue.queue_bytes")
            for payload in self._payloads.values()
            if payload.encoding_status is not EncodingState.EXPIRED
        )

    def is_available(self, payload: EncodedPayload, *, current_time_s: float) -> bool:
        if payload.encoding_status in {EncodingState.RAW, EncodingState.ENCODING}:
            return False
        return current_time_s >= payload.physical_ready_time_s(module_name="EncodedDownlinkQueue.is_available")

    def enqueue(self, payload: EncodedPayload, *, current_time_s: float, arrival_time_s: float = 0.0) -> None:
        if payload.encoding_status is not EncodingState.ENCODED:
            raise payload._violation(
                module_name="EncodedDownlinkQueue.enqueue",
                requested_field="encoding_status",
                message="queue admission requires state=ENCODED",
            )
        ready = payload.physical_ready_time_s(module_name="EncodedDownlinkQueue.enqueue")
        if current_time_s < ready:
            raise payload._violation(
                module_name="EncodedDownlinkQueue.enqueue",
                requested_field="ready_time_s",
                message="cannot queue before ready_time_s",
            )
        if payload.payload_id in self._payloads:
            raise payload._violation(
                module_name="EncodedDownlinkQueue.enqueue",
                requested_field="payload_id",
                message="duplicate encoded payload",
            )
        payload.mark_queued()
        self._payloads[payload.payload_id] = payload
        start = float(payload.encode_start_time_s or 0.0)
        self.lineage_events.extend(
            [
                _event(payload, "task_arrival", arrival_time_s),
                _event(payload, "product_generated", start),
                _event(payload, "encoding_started", start),
                _event(
                    payload,
                    "encoding_completed",
                    ready,
                    encoded_size_bytes=payload.physical_size_bytes(module_name="EncodedDownlinkQueue.enqueue"),
                    encoded_quality=payload.physical_quality(module_name="EncodedDownlinkQueue.enqueue"),
                ),
                _event(payload, "queued", current_time_s, remaining_encoded_bytes=payload.physical_remaining_bytes(module_name="EncodedDownlinkQueue.enqueue")),
            ]
        )

    def priority(self, payload_id: str, *, base_utility: float) -> float:
        payload = self._payload(payload_id)
        remaining = payload.physical_remaining_bytes(module_name="EncodedDownlinkQueue.priority")
        quality = payload.physical_quality(module_name="EncodedDownlinkQueue.priority")
        value = float(base_utility) * quality / max(remaining, 1)
        self.lineage_events.append(_event(payload, "priority_evaluated", payload.physical_ready_time_s(module_name="EncodedDownlinkQueue.priority"), priority=value))
        return value

    def queue_waiting_s(self, payload_id: str, *, current_time_s: float) -> float:
        payload = self._payload(payload_id)
        ready = payload.physical_ready_time_s(module_name="EncodedDownlinkQueue.queue_waiting_s")
        return max(0.0, float(current_time_s) - ready)

    def transmission_duration_s(self, payload_id: str, *, throughput_bytes_s: float) -> float:
        if throughput_bytes_s <= 0.0:
            raise ValueError("throughput_bytes_s must be positive")
        payload = self._payload(payload_id)
        return payload.physical_remaining_bytes(module_name="EncodedDownlinkQueue.transmission_duration_s") / throughput_bytes_s

    def deadline_feasible(
        self,
        payload_id: str,
        *,
        current_time_s: float,
        deadline_time_s: float,
        throughput_bytes_s: float,
    ) -> bool:
        payload = self._payload(payload_id)
        start = max(current_time_s, payload.physical_ready_time_s(module_name="EncodedDownlinkQueue.deadline_feasible"))
        finish = start + self.transmission_duration_s(payload_id, throughput_bytes_s=throughput_bytes_s)
        feasible = finish <= deadline_time_s
        self.lineage_events.append(
            _event(
                payload,
                "window_selected" if feasible else "window_rejected",
                start,
                predicted_finish_time_s=finish,
                deadline_time_s=float(deadline_time_s),
                feasible=int(feasible),
            )
        )
        return feasible

    def transmit(
        self,
        payload_id: str,
        *,
        current_time_s: float,
        max_bytes: int,
        communication_energy_Wh_per_byte: float,
    ) -> int:
        payload = self._payload(payload_id)
        ready = payload.physical_ready_time_s(module_name="EncodedDownlinkQueue.transmit")
        if current_time_s < ready:
            raise payload._violation(
                module_name="EncodedDownlinkQueue.transmit",
                requested_field="ready_time_s",
                message="transmission before ready_time_s",
            )
        remaining_before = payload.physical_remaining_bytes(module_name="EncodedDownlinkQueue.transmit")
        sent = min(max(0, int(max_bytes)), remaining_before)
        if sent <= 0:
            return 0
        payload.record_transmission(bytes_sent=sent, module_name="EncodedDownlinkQueue.transmit")
        energy = sent * float(communication_energy_Wh_per_byte)
        self.communication_energy_Wh += energy
        self.delivered_bytes_from_events += sent
        self.lineage_events.append(
            _event(
                payload,
                "delivered" if payload.encoding_status is EncodingState.DELIVERED else "partially_transmitted",
                current_time_s,
                bytes_transmitted=sent,
                transmitted_bytes=payload.transmitted_bytes,
                remaining_encoded_bytes=payload.physical_remaining_bytes(module_name="EncodedDownlinkQueue.transmit"),
                communication_energy_Wh=energy,
            )
        )
        return sent

    def closure_utility(self, payload_id: str, *, base_utility: float) -> float:
        payload = self._payload(payload_id)
        if payload.encoding_status is not EncodingState.DELIVERED:
            return 0.0
        utility = float(base_utility) * payload.physical_quality(module_name="EncodedDownlinkQueue.closure_utility")
        delivered_events = [row for row in self.lineage_events if row["payload_id"] == payload_id and row["event_type"] == "delivered"]
        event_time = float(delivered_events[-1]["event_time_s"]) if delivered_events else payload.physical_ready_time_s(module_name="EncodedDownlinkQueue.closure_utility")
        self.lineage_events.append(_event(payload, "closure_utility_evaluated", event_time, closure_status="DELIVERED", utility=utility))
        return utility

    def end_to_end_latency_s(self, payload_id: str, *, arrival_time_s: float) -> float:
        events = [row for row in self.lineage_events if row["payload_id"] == payload_id and row["event_type"] == "delivered"]
        if not events:
            raise RuntimeError(f"payload not delivered payload_id={payload_id}")
        return float(events[-1]["event_time_s"]) - float(arrival_time_s)

    def physical_metric_values(self) -> list[float | int]:
        values: list[float | int] = [self.queue_bytes, self.communication_energy_Wh, self.delivered_bytes_from_events]
        for payload in self._payloads.values():
            values.extend(
                [
                    payload.physical_size_bytes(module_name="EncodedDownlinkQueue.physical_metric_values"),
                    payload.physical_remaining_bytes(module_name="EncodedDownlinkQueue.physical_metric_values"),
                    payload.physical_quality(module_name="EncodedDownlinkQueue.physical_metric_values"),
                    payload.physical_ready_time_s(module_name="EncodedDownlinkQueue.physical_metric_values"),
                    payload.encode_energy_Wh,
                ]
            )
        return values
