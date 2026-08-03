from __future__ import annotations

import pytest

from src.payload.fixed_product_compression import (
    CodecAdapter,
    EncodedPayload,
    EncodingState,
    FixedProductCompressionConfig,
)


class _IdentityCodec(CodecAdapter):
    codec_id = "TEST_IDENTITY"
    codec_version = "1"

    def encode(self, product_bytes: bytes, *, profile_id: str) -> bytes:
        return product_bytes

    def decode(self, encoded_bytes: bytes, *, profile_id: str) -> bytes:
        return encoded_bytes


def _payload() -> EncodedPayload:
    return EncodedPayload(
        task_id="task-1",
        product_id="product-1",
        product_type="roi",
        stage_id="direct",
        raw_size_bytes=8,
        raw_quality=0.9,
        compression_physics_active=True,
    )


def test_feature_flag_defaults_disabled() -> None:
    assert FixedProductCompressionConfig().enable_fixed_product_compression is False


def test_legal_state_path_exposes_encoded_physical_view() -> None:
    payload = _payload()
    payload.start_encoding(start_time_s=20.0)
    payload.finish_encoding(
        encoded_bytes=b"abcd",
        encoded_quality=0.88,
        end_time_s=21.5,
        encode_energy_Wh=0.002,
        codec=_IdentityCodec(),
        codec_profile_id="identity",
    )
    payload.mark_queued()

    view = payload.downstream_physical_view(size_source="encoded_size_bytes")

    assert payload.encoding_status is EncodingState.QUEUED
    assert view.size_bytes == 4
    assert view.remaining_bytes == 4
    assert view.quality == pytest.approx(0.88)
    assert view.ready_time_s == pytest.approx(21.5)


def test_downstream_access_rejects_raw_size_source() -> None:
    payload = _payload()
    payload.start_encoding(start_time_s=0.0)
    payload.finish_encoding(
        encoded_bytes=b"abcd",
        encoded_quality=0.9,
        end_time_s=1.0,
        encode_energy_Wh=0.0,
        codec=_IdentityCodec(),
        codec_profile_id="identity",
    )

    with pytest.raises(AssertionError, match="raw_size_bytes is audit-only"):
        payload.downstream_physical_view(size_source="raw_size_bytes")


def test_payload_cannot_queue_before_encoding_completes() -> None:
    with pytest.raises(RuntimeError, match="ENCODED"):
        _payload().mark_queued()


def test_duplicate_encoding_is_rejected() -> None:
    payload = _payload()
    payload.start_encoding(start_time_s=0.0)
    with pytest.raises(RuntimeError, match="RAW"):
        payload.start_encoding(start_time_s=1.0)


def test_nonpositive_encoded_result_is_rejected() -> None:
    payload = _payload()
    payload.start_encoding(start_time_s=0.0)
    with pytest.raises(ValueError, match="positive"):
        payload.finish_encoding(
            encoded_bytes=b"",
            encoded_quality=0.9,
            end_time_s=1.0,
            encode_energy_Wh=0.0,
            codec=_IdentityCodec(),
            codec_profile_id="identity",
        )
