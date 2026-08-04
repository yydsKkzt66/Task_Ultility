"""Disabled-by-default contract for future real product compression.

This module does not create image products or estimate compression.  It only
defines the physical boundary a real codec integration must satisfy.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class EncodingState(str, Enum):
    RAW = "RAW"
    ENCODING = "ENCODING"
    ENCODED = "ENCODED"
    QUEUED = "QUEUED"
    TRANSMITTING = "TRANSMITTING"
    PARTIALLY_TRANSMITTED = "PARTIALLY_TRANSMITTED"
    DELIVERED = "DELIVERED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class FixedProductCompressionConfig:
    enable_fixed_product_compression: bool = False


class CodecAdapter(ABC):
    """Codec boundary; implementations must return actual encoded bytes."""

    codec_id: str
    codec_version: str

    @abstractmethod
    def encode(self, product_bytes: bytes, *, profile_id: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def decode(self, encoded_bytes: bytes, *, profile_id: str) -> bytes:
        raise NotImplementedError


@dataclass(frozen=True)
class DownstreamEncodedPhysicalView:
    size_bytes: int
    remaining_bytes: int
    quality: float
    ready_time_s: float
    encode_energy_Wh: float


@dataclass
class EncodedPayload:
    task_id: str
    product_id: str
    product_type: str
    stage_id: str
    raw_size_bytes: int
    raw_quality: float
    payload_id: str = ""
    compression_profile_id: str = ""
    compression_profile_hash: str = ""
    codec_id: str = ""
    codec_version: str = ""
    encoded_size_bytes: int = 0
    encoded_quality: float | None = None
    compression_factor: float = 0.0
    header_bytes: int = 0
    encode_start_time_s: float | None = None
    encode_end_time_s: float | None = None
    encode_latency_s: float = 0.0
    encode_energy_Wh: float = 0.0
    ready_time_s: float | None = None
    transmitted_bytes: int = 0
    remaining_encoded_bytes: int = 0
    checksum_sha256: str = ""
    encoding_status: EncodingState = EncodingState.RAW
    codec_mode: str = ""
    is_real_codec_output: bool = False
    provenance_source_label: str = ""
    compression_physics_active: bool = False
    _raw_access_module: str = "direct_downstream_access"

    def __getattribute__(self, name: str):
        if name in {"raw_size_bytes", "raw_quality"}:
            active = object.__getattribute__(self, "__dict__").get("compression_physics_active", False)
            if active:
                state = object.__getattribute__(self, "__dict__").get("encoding_status", EncodingState.RAW)
                payload_id = object.__getattribute__(self, "__dict__").get("payload_id", "unknown")
                task_id = object.__getattribute__(self, "__dict__").get("task_id", "unknown")
                module = object.__getattribute__(self, "__dict__").get("_raw_access_module", "direct_downstream_access")
                raise RuntimeError(
                    f"encoded physics violation payload_id={payload_id} task_id={task_id} "
                    f"module={module} requested_field={name} state={state.value}"
                )
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        raw_size = object.__getattribute__(self, "raw_size_bytes")
        if raw_size <= 0:
            raise ValueError("raw_size_bytes must be positive audit evidence")
        if not self.payload_id:
            identity = f"{self.task_id}|{self.product_id}|{self.stage_id}|{self.product_type}"
            self.payload_id = "encoded-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    @property
    def codec_profile_id(self) -> str:
        return self.compression_profile_id

    def audit_raw_size_bytes(self) -> int:
        return int(object.__getattribute__(self, "raw_size_bytes"))

    def audit_raw_quality(self) -> float:
        return float(object.__getattribute__(self, "raw_quality"))

    def _violation(self, *, module_name: str, requested_field: str, message: str) -> RuntimeError:
        return RuntimeError(
            f"{message} payload_id={self.payload_id} task_id={self.task_id} module={module_name} "
            f"requested_field={requested_field} state={self.encoding_status.value}"
        )

    def start_encoding(self, *, start_time_s: float) -> None:
        if self.encoding_status is not EncodingState.RAW:
            raise RuntimeError("encoding may start only from RAW")
        self.encode_start_time_s = float(start_time_s)
        self.encoding_status = EncodingState.ENCODING

    def finish_encoding(
        self,
        *,
        encoded_bytes: bytes,
        encoded_quality: float,
        end_time_s: float,
        encode_energy_Wh: float,
        codec: CodecAdapter,
        codec_profile_id: str,
        header_bytes: int = 0,
    ) -> None:
        if self.encoding_status is not EncodingState.ENCODING:
            raise RuntimeError("encoding may finish only from ENCODING")
        if not encoded_bytes:
            raise ValueError("encoded_size_bytes must be positive")
        if self.encode_start_time_s is None:
            raise RuntimeError("encode_start_time_s is missing")
        end = float(end_time_s)
        if end < self.encode_start_time_s:
            raise ValueError("encode_end_time_s precedes encode_start_time_s")
        if encoded_quality is None or not math.isfinite(float(encoded_quality)) or not 0.0 <= float(encoded_quality) <= 1.0:
            raise self._violation(
                module_name="EncodedPayload.finish_encoding",
                requested_field="encoded_quality",
                message="encoded_quality must be finite and in [0, 1]",
            )
        if float(encode_energy_Wh) < 0.0:
            raise ValueError("encode_energy_Wh must be non-negative")

        self.codec_id = str(codec.codec_id)
        self.codec_version = str(codec.codec_version)
        self.compression_profile_id = str(codec_profile_id)
        self.encoded_size_bytes = len(encoded_bytes)
        self.encoded_quality = float(encoded_quality)
        raw_size = self.audit_raw_size_bytes()
        self.compression_factor = raw_size / self.encoded_size_bytes
        self.header_bytes = int(header_bytes)
        if self.header_bytes < 0 or self.header_bytes > self.encoded_size_bytes:
            raise ValueError("header_bytes must be within encoded_size_bytes")
        self.encode_end_time_s = end
        self.encode_latency_s = end - self.encode_start_time_s
        self.encode_energy_Wh = float(encode_energy_Wh)
        self.ready_time_s = end
        self.remaining_encoded_bytes = self.encoded_size_bytes
        self.checksum_sha256 = hashlib.sha256(encoded_bytes).hexdigest()
        self.encoding_status = EncodingState.ENCODED

    def mark_queued(self) -> None:
        if self.encoding_status is not EncodingState.ENCODED:
            raise RuntimeError("only ENCODED payloads may enter the queue")
        self.encoding_status = EncodingState.QUEUED

    def physical_size_bytes(self, *, module_name: str) -> int:
        self._require_encoded_field(module_name=module_name, requested_field="encoded_size_bytes")
        return self.encoded_size_bytes

    def physical_quality(self, *, module_name: str) -> float:
        self._require_encoded_field(module_name=module_name, requested_field="encoded_quality")
        if self.encoded_quality is None:
            raise self._violation(module_name=module_name, requested_field="encoded_quality", message="missing encoded quality")
        return float(self.encoded_quality)

    def physical_ready_time_s(self, *, module_name: str) -> float:
        self._require_encoded_field(module_name=module_name, requested_field="ready_time_s")
        if self.ready_time_s is None:
            raise self._violation(module_name=module_name, requested_field="ready_time_s", message="missing ready time")
        return float(self.ready_time_s)

    def physical_remaining_bytes(self, *, module_name: str) -> int:
        self._require_encoded_field(module_name=module_name, requested_field="remaining_encoded_bytes")
        return int(self.remaining_encoded_bytes)

    def _require_encoded_field(self, *, module_name: str, requested_field: str) -> None:
        if not self.compression_physics_active:
            raise self._violation(module_name=module_name, requested_field=requested_field, message="compression physics inactive")
        if self.encoding_status in {EncodingState.RAW, EncodingState.ENCODING, EncodingState.EXPIRED}:
            raise self._violation(module_name=module_name, requested_field=requested_field, message="encoded field unavailable")
        value = getattr(self, requested_field)
        if value is None or isinstance(value, (int, float)) and (not math.isfinite(float(value)) or float(value) < 0.0):
            raise self._violation(module_name=module_name, requested_field=requested_field, message="invalid encoded field")

    def record_transmission(self, *, bytes_sent: int, module_name: str) -> None:
        if self.encoding_status not in {
            EncodingState.QUEUED,
            EncodingState.TRANSMITTING,
            EncodingState.PARTIALLY_TRANSMITTED,
        }:
            raise self._violation(module_name=module_name, requested_field="remaining_encoded_bytes", message="payload not transmissible")
        if bytes_sent <= 0 or bytes_sent > self.remaining_encoded_bytes:
            raise self._violation(module_name=module_name, requested_field="bytes_sent", message="invalid encoded transmission")
        self.encoding_status = EncodingState.TRANSMITTING
        self.transmitted_bytes += int(bytes_sent)
        self.remaining_encoded_bytes -= int(bytes_sent)
        if self.transmitted_bytes + self.remaining_encoded_bytes != self.encoded_size_bytes:
            raise self._violation(module_name=module_name, requested_field="remaining_encoded_bytes", message="encoded byte conservation failed")
        self.encoding_status = (
            EncodingState.DELIVERED if self.remaining_encoded_bytes == 0 else EncodingState.PARTIALLY_TRANSMITTED
        )

    def reset_remaining_bytes(self, *, source_field: str, module_name: str) -> None:
        if source_field != "encoded_size_bytes":
            raise self._violation(module_name=module_name, requested_field=source_field, message="remaining bytes cannot reset from raw_size_bytes")
        raise self._violation(module_name=module_name, requested_field=source_field, message="remaining bytes reset is forbidden after encoding")

    def downstream_physical_view(self, *, size_source: str) -> DownstreamEncodedPhysicalView:
        if size_source == "raw_size_bytes":
            raise AssertionError("raw_size_bytes is audit-only and cannot drive downstream physics")
        if size_source != "encoded_size_bytes":
            raise AssertionError(f"unsupported downstream size source: {size_source}")
        if self.encoding_status not in {
            EncodingState.ENCODED,
            EncodingState.QUEUED,
            EncodingState.TRANSMITTING,
            EncodingState.DELIVERED,
        }:
            raise RuntimeError("downstream physical access requires an ENCODED payload")
        if self.encoded_size_bytes <= 0 or self.remaining_encoded_bytes < 0 or self.ready_time_s is None:
            raise AssertionError("encoded physical fields are incomplete")
        return DownstreamEncodedPhysicalView(
            size_bytes=self.encoded_size_bytes,
            remaining_bytes=self.remaining_encoded_bytes,
            quality=self.physical_quality(module_name="downstream_physical_view"),
            ready_time_s=self.ready_time_s,
            encode_energy_Wh=self.encode_energy_Wh,
        )
