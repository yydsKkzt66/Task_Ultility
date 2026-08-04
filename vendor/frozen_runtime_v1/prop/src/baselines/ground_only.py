"""Runnable GROUND-ONLY baseline built on the frozen aligned slot executor.

Every task is deterministically assigned to the GROUND path once.  Compression,
FIFO downlink, thermal headroom, battery updates, and finite ground compute are
the same components used by OFFLOAD-EFT-THERMAL-ALIGNED's GROUND path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.baselines.b1_enhanced import (
    B1CompressionConfig,
    B1GroundConfig,
    compressed_size_mb,
    compression_energy_wh,
    compression_latency_s,
)
from src.baselines.offload_eft import GROUND
from src.baselines.offload_eft_thermal_aligned import AlignedJob, ThermalAlignedSimulator


@dataclass(frozen=True)
class GroundPathParameters:
    raw_size_mb: float
    compressed_size_mb: float
    compression_latency_s: float
    compression_energy_j: float
    ground_compute_latency_s: float
    ground_quality: float


@dataclass(frozen=True)
class GroundOnlyDecision:
    task_id: str
    selected_path: str = GROUND
    committed_once: bool = True
    migration_count: int = 0
    replication_count: int = 0
    future_tasks_read: int = 0


def _model_profile(configs: dict[str, Any], model_name: str) -> dict[str, float]:
    model = next(row for row in configs["model_pool"]["models"] if row["name"] == model_name)
    base = next(row for row in configs["model_pool"]["profiles"] if row["name"] == model["profile_name"])
    return {**base, **configs["model_pool"].get("profile_overrides", {}).get(model["profile_name"], {})}


def ground_path_parameters(configs: dict[str, Any]) -> GroundPathParameters:
    """Resolve the existing B1/OFFLOAD GROUND parameters without new knobs."""
    compression = B1CompressionConfig()
    ground = B1GroundConfig()
    raw_mb = compression.reference_raw_size_mb
    payload_mb = compressed_size_mb(raw_mb, compression.compression_ratio)
    if abs(payload_mb - compression.reference_compressed_size_mb) > 1.0e-12:
        raise ValueError("B1 compressed-size contract is internally inconsistent")
    latency_s = compression_latency_s(raw_mb, compression)
    energy_j = compression_energy_wh(latency_s, compression.compression_power_w) * 3600.0
    profile = _model_profile(configs, ground.inference_model_name)
    ground_latency_s = ground.decode_latency_s_per_reference_image + float(profile["latency_s"])
    quality = min(compression.reconstructed_ssim_proxy, float(profile["quality_base"]))
    return GroundPathParameters(raw_mb, payload_mb, latency_s, energy_j, ground_latency_s, quality)


def validate_manifest_identity(
    actual_task_hash: str,
    actual_environment_hash: str,
    expected_task_hash: str,
    expected_environment_hash: str,
) -> None:
    if actual_task_hash != expected_task_hash:
        raise ValueError("task_manifest_hash_mismatch")
    if actual_environment_hash != expected_environment_hash:
        raise ValueError("environment_fingerprint_mismatch")


class GroundOnlySimulator(ThermalAlignedSimulator):
    """Force every task through the existing aligned GROUND execution path."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ground_only_decisions: dict[str, GroundOnlyDecision] = {}
        self.migration_count = 0
        self.replication_count = 0
        self.future_tasks_read = 0

    def admit_ground_task(
        self,
        *,
        task_id: str,
        arrival_s: float,
        deadline_s: float,
        product_feasible: bool,
    ) -> GroundOnlyDecision:
        if task_id in self.ground_only_decisions:
            return self.ground_only_decisions[task_id]
        decision = GroundOnlyDecision(task_id=task_id)
        self.ground_only_decisions[task_id] = decision
        self._admit(self._new_job(task_id, GROUND, arrival_s, deadline_s, product_feasible))
        return decision

    def is_closed_at(self, task_id: str, time_s: float) -> bool:
        job = self.jobs.get(task_id)
        return bool(
            job
            and job.product_feasible
            and job.closure_s is not None
            and job.closure_s <= time_s + 1.0e-9
            and job.closure_s <= job.deadline_s + 1.0e-9
        )

    def final_failure_reason(self, task_id: str) -> str:
        job: AlignedJob = self.jobs[task_id]
        if not job.product_feasible:
            return "product_or_requirement_infeasible"
        if job.failure_reason:
            return job.failure_reason
        if job.closure_s is None:
            return "not_completed_before_horizon"
        if job.closure_s > job.deadline_s + 1.0e-9:
            return "deadline"
        return ""
