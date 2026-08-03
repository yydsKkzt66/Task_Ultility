from __future__ import annotations

import math

import pytest

from src.payload.encoded_downlink_path import EncodedDownlinkQueue, EncodedResourceLedger
from src.payload.fixed_product_compression import CodecAdapter, EncodedPayload, EncodingState


RAW_BYTES = 10_000_000
ENCODED_BYTES = 1_234_567
RAW_QUALITY = 0.97
ENCODED_QUALITY = 0.83
ENCODE_LATENCY_S = 7.25
ENCODE_ENERGY_WH = 0.0042


class _SentinelCodec(CodecAdapter):
    codec_id = "SENTINEL_NOT_REAL_CODEC"
    codec_version = "1"

    def encode(self, product_bytes: bytes, *, profile_id: str) -> bytes:
        return b"x" * ENCODED_BYTES

    def decode(self, encoded_bytes: bytes, *, profile_id: str) -> bytes:
        return encoded_bytes


def _raw_payload() -> EncodedPayload:
    return EncodedPayload(
        task_id="sentinel-task",
        product_id="sentinel-product",
        product_type="roi_context",
        stage_id="deep",
        raw_size_bytes=RAW_BYTES,
        raw_quality=RAW_QUALITY,
        compression_profile_id="sentinel-profile",
        compression_profile_hash="sha256:sentinel-profile",
        codec_mode="sentinel_contract_only",
        is_real_codec_output=False,
        provenance_source_label="deterministic_sentinel",
        compression_physics_active=True,
    )


def _encoded_payload() -> EncodedPayload:
    payload = _raw_payload()
    payload.start_encoding(start_time_s=0.0)
    encoded = _SentinelCodec().encode(b"raw-not-materialized", profile_id="sentinel-profile")
    payload.finish_encoding(
        encoded_bytes=encoded,
        encoded_quality=ENCODED_QUALITY,
        end_time_s=ENCODE_LATENCY_S,
        encode_energy_Wh=ENCODE_ENERGY_WH,
        codec=_SentinelCodec(),
        codec_profile_id="sentinel-profile",
        header_bytes=67,
    )
    return payload


def test_distinctive_sentinel_propagates_end_to_end() -> None:
    payload = _encoded_payload()
    queue = EncodedDownlinkQueue(compression_physics_active=True)
    ledger = EncodedResourceLedger(initial_soc_Wh=100.0)

    assert queue.is_available(payload, current_time_s=7.24) is False
    queue.enqueue(payload, current_time_s=ENCODE_LATENCY_S)
    assert queue.queue_bytes == ENCODED_BYTES
    assert queue.queue_waiting_s(payload.payload_id, current_time_s=ENCODE_LATENCY_S) == 0.0
    assert queue.queue_waiting_s(payload.payload_id, current_time_s=8.0) == pytest.approx(0.75)
    assert queue.priority(payload.payload_id, base_utility=2.0) == pytest.approx(
        2.0 * ENCODED_QUALITY / ENCODED_BYTES
    )
    assert queue.transmission_duration_s(payload.payload_id, throughput_bytes_s=1_000_000.0) == pytest.approx(
        ENCODED_BYTES / 1_000_000.0
    )
    assert queue.deadline_feasible(
        payload.payload_id,
        current_time_s=0.0,
        deadline_time_s=9.0,
        throughput_bytes_s=1_000_000.0,
    )

    ledger.charge_encoding(payload, thermal_fraction=0.5)
    assert ledger.controllable_energy_Wh == pytest.approx(ENCODE_ENERGY_WH)
    assert ledger.soc_Wh == pytest.approx(100.0 - ENCODE_ENERGY_WH)
    assert ledger.thermal_compute_energy_Wh == pytest.approx(ENCODE_ENERGY_WH * 0.5)

    first = queue.transmit(
        payload.payload_id,
        current_time_s=8.0,
        max_bytes=234_567,
        communication_energy_Wh_per_byte=1e-9,
    )
    assert first == 234_567
    assert payload.encoding_status is EncodingState.PARTIALLY_TRANSMITTED
    assert payload.physical_remaining_bytes(module_name="test") == 1_000_000
    second = queue.transmit(
        payload.payload_id,
        current_time_s=9.0,
        max_bytes=2_000_000,
        communication_energy_Wh_per_byte=1e-9,
    )
    assert second == 1_000_000
    assert payload.encoding_status is EncodingState.DELIVERED
    assert payload.transmitted_bytes + payload.physical_remaining_bytes(module_name="test") == ENCODED_BYTES
    assert queue.delivered_bytes_from_events == ENCODED_BYTES
    assert queue.communication_energy_Wh == pytest.approx(ENCODED_BYTES * 1e-9)
    assert queue.closure_utility(payload.payload_id, base_utility=2.0) == pytest.approx(1.66)
    assert queue.end_to_end_latency_s(payload.payload_id, arrival_time_s=0.0) == pytest.approx(9.0)
    assert all(event["payload_id"] == payload.payload_id for event in queue.lineage_events)
    assert all(event["compression_profile_hash"] == payload.compression_profile_hash for event in queue.lineage_events)
    assert {
        "task_arrival",
        "product_generated",
        "encoding_started",
        "encoding_completed",
        "queued",
        "priority_evaluated",
        "window_selected",
        "partially_transmitted",
        "delivered",
        "closure_utility_evaluated",
    } <= {str(event["event_type"]) for event in queue.lineage_events}
    assert RAW_BYTES not in queue.physical_metric_values()


def test_queue_rejects_raw_payload() -> None:
    with pytest.raises(RuntimeError, match="state=RAW"):
        EncodedDownlinkQueue(compression_physics_active=True).enqueue(_raw_payload(), current_time_s=0.0)


def test_direct_raw_access_is_guarded_when_physics_active() -> None:
    payload = _raw_payload()
    with pytest.raises(RuntimeError, match="requested_field=raw_size_bytes"):
        _ = payload.raw_size_bytes


def test_missing_encoded_quality_is_rejected() -> None:
    payload = _raw_payload()
    payload.start_encoding(start_time_s=0.0)
    with pytest.raises(
        RuntimeError,
        match=r"payload_id=.*task_id=sentinel-task.*module=EncodedPayload.finish_encoding.*requested_field=encoded_quality.*state=ENCODING",
    ):
        payload.finish_encoding(
            encoded_bytes=b"encoded",
            encoded_quality=None,
            end_time_s=1.0,
            encode_energy_Wh=0.0,
            codec=_SentinelCodec(),
            codec_profile_id="sentinel-profile",
        )


def test_transmit_before_ready_time_is_rejected() -> None:
    payload = _encoded_payload()
    queue = EncodedDownlinkQueue(compression_physics_active=True)
    with pytest.raises(RuntimeError, match="ready_time_s"):
        queue.enqueue(payload, current_time_s=7.24)


def test_encoding_energy_cannot_be_double_charged() -> None:
    payload = _encoded_payload()
    ledger = EncodedResourceLedger(initial_soc_Wh=100.0)
    ledger.charge_encoding(payload, thermal_fraction=1.0)
    with pytest.raises(RuntimeError, match="already charged"):
        ledger.charge_encoding(payload, thermal_fraction=1.0)


def test_remaining_bytes_cannot_be_reset_from_raw_size() -> None:
    payload = _encoded_payload()
    with pytest.raises(RuntimeError, match="raw_size_bytes"):
        payload.reset_remaining_bytes(source_field="raw_size_bytes", module_name="negative_test")
