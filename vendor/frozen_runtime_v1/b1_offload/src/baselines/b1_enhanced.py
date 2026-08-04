"""B1-Enhanced contact-aware compressed downlink baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.common.types import LinkTrace, ResourceState, TaskSample, TimelineStep
from src.energy.battery_update import update_battery_soc
from src.energy.comm_energy import compute_comm_energy_j
from src.energy.compute_energy import compute_inference_energy_j
from src.energy.thermal_update import update_thermal_pressure
from src.energy.thermal_headroom import maximum_safe_downlink_capacity, thermal_headroom_config
from src.models.model_registry import ModelRegistry


METHOD_NAME = "b1_enhanced_contact_compressed"
PAYLOAD_TYPE = "compressed_full_image"
GROUND_RESULT_STATE = "ground_result_available"


@dataclass(frozen=True)
class B1CompressionConfig:
    compression_ratio: float = 8.0
    reference_raw_size_mb: float = 32.0
    reference_onboard_latency_s: float = 11.0
    reference_compressed_size_mb: float = 4.0
    paper_accelerator_power_w: float = 1.4
    reconstructed_ssim_proxy: float = 0.96
    downstream_quality_retention: float = 1.0
    compression_power_w: float = 6.8
    compression_power_assumption_source: str = (
        "engineering_assumption_for_current_simulator_not_reported_phi_sat2_power"
    )
    latency_scaling: str = "linear_by_raw_mb"


@dataclass(frozen=True)
class B1GroundConfig:
    decode_latency_s_per_reference_image: float = 1.5
    inference_model_name: str = "high_accuracy"
    inference_latency_s: float | None = None
    ground_energy_not_counted_in_satellite_soc: bool = True


@dataclass(frozen=True)
class B1SchedulerConfig:
    policy: str = "enhanced"
    require_full_fit_in_current_contact: bool = True
    non_preemptive_once_started: bool = True
    w_density: float = 0.55
    w_urgency: float = 0.25
    w_age: float = 0.10
    w_fit: float = 0.10
    deadline_urgency_tau_s: float = 120.0
    hard_safety_epsilon: float = 1.0e-9


@dataclass(frozen=True)
class B1RawSizeConfig:
    default_raw_size_mb: float = 32.0
    metadata_keys: tuple[str, ...] = ("raw_payload_mb", "raw_size_mb", "image_raw_size_mb", "source_image_mb")


@dataclass(frozen=True)
class B1EnhancedConfig:
    enabled: bool = False
    name: str = "contact_aware_compressed_ground_processing"
    baseline_version: str = "b1_enhanced_contact_compressed_v2"
    compression: B1CompressionConfig = field(default_factory=B1CompressionConfig)
    ground: B1GroundConfig = field(default_factory=B1GroundConfig)
    scheduler: B1SchedulerConfig = field(default_factory=B1SchedulerConfig)
    raw_size: B1RawSizeConfig = field(default_factory=B1RawSizeConfig)


@dataclass
class B1TaskState:
    task: TaskSample
    created_slot: int
    capture_time_s: float
    deadline_time_s: float
    raw_size_mb: float
    compressed_size_mb: float
    compression_latency_s: float
    compression_energy_wh: float
    reconstructed_image_quality_proxy: float
    ground_task_quality: float
    state: str = "compression_pending"
    compression_remaining_s: float = 0.0
    compressed_available_time_s: float | None = None
    queue_enter_time_s: float | None = None
    transmission_start_time_s: float | None = None
    ground_received_compressed_time_s: float | None = None
    ground_decode_start_time_s: float | None = None
    ground_decode_end_time_s: float | None = None
    ground_inference_start_time_s: float | None = None
    ground_result_available_time_s: float | None = None
    remaining_downlink_mb: float = 0.0
    downlinked_mb: float = 0.0
    contact_wait_latency_s: float = 0.0
    transmission_latency_s: float = 0.0
    ground_decode_latency_s: float = 0.0
    ground_inference_latency_s: float = 0.0
    queue_wait_latency_s: float = 0.0
    capture_to_ground_latency_s: float = 0.0
    late_delivery: bool = False
    scheduler_score: float = 0.0
    scheduler_rank: int = 0
    blocked_reason: str = ""
    expired_recorded: bool = False
    window_id: str = ""

    @property
    def high_value(self) -> bool:
        return bool(self.task.high_value_flag or self.task.emergency_flag)


@dataclass(frozen=True)
class B1ScheduleDecision:
    task_id: str
    score: float
    value_density: float
    deadline_urgency: float
    age_s: float
    full_window_fit: int
    reason: str


def load_b1_config(raw_cfg: dict[str, Any] | None) -> B1EnhancedConfig:
    raw_cfg = raw_cfg or {}
    compression = raw_cfg.get("compression", {}) if isinstance(raw_cfg.get("compression", {}), dict) else {}
    ground = raw_cfg.get("ground", {}) if isinstance(raw_cfg.get("ground", {}), dict) else {}
    scheduler = raw_cfg.get("scheduler", {}) if isinstance(raw_cfg.get("scheduler", {}), dict) else {}
    raw_size = raw_cfg.get("raw_size", {}) if isinstance(raw_cfg.get("raw_size", {}), dict) else {}
    return B1EnhancedConfig(
        enabled=bool(raw_cfg.get("enabled", False)),
        name=str(raw_cfg.get("name", "contact_aware_compressed_ground_processing")),
        baseline_version=str(raw_cfg.get("baseline_version", "b1_enhanced_contact_compressed_v2")),
        compression=B1CompressionConfig(**{k: v for k, v in compression.items() if k in B1CompressionConfig.__annotations__}),
        ground=B1GroundConfig(**{k: v for k, v in ground.items() if k in B1GroundConfig.__annotations__}),
        scheduler=B1SchedulerConfig(**{k: v for k, v in scheduler.items() if k in B1SchedulerConfig.__annotations__}),
        raw_size=B1RawSizeConfig(
            default_raw_size_mb=float(raw_size.get("default_raw_size_mb", 32.0)),
            metadata_keys=tuple(raw_size.get("metadata_keys", B1RawSizeConfig().metadata_keys)),
        ),
    )


def compressed_size_mb(raw_size_mb: float, compression_ratio: float) -> float:
    return max(0.0, float(raw_size_mb)) / max(float(compression_ratio), 1.0e-9)


def compression_latency_s(raw_size_mb: float, cfg: B1CompressionConfig) -> float:
    if cfg.latency_scaling != "linear_by_raw_mb":
        raise ValueError(f"Unsupported B1 compression latency scaling: {cfg.latency_scaling}")
    return cfg.reference_onboard_latency_s * max(0.0, raw_size_mb) / max(cfg.reference_raw_size_mb, 1.0e-9)


def compression_energy_wh(latency_s: float, compression_power_w: float) -> float:
    return max(0.0, compression_power_w * latency_s / 3600.0)


def _stable_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_manifest_hash(timeline: list[TimelineStep]) -> str:
    rows = []
    for step in timeline:
        if step.task is None:
            continue
        task = step.task
        row = asdict(task)
        row["slot_index"] = step.slot_index
        rows.append(row)
    return _stable_hash(rows)


def contact_schedule_hash(timeline: list[TimelineStep]) -> str:
    return _stable_hash([asdict(step.link_trace) for step in timeline])


def resource_trace_hash(timeline: list[TimelineStep]) -> str:
    return _stable_hash([asdict(step.resource_trace) for step in timeline])


def _file_sha256(path: Path | None) -> str:
    if path is None:
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ContactAwareCompressedGroundScheduler:
    """Rolling-horizon scheduler for compressed full-image payloads."""

    def __init__(self, cfg: B1SchedulerConfig) -> None:
        self.cfg = cfg
        self.no_fit_count = 0

    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        min_value = min(values.values())
        max_value = max(values.values())
        span = max(max_value - min_value, 1.0e-9)
        return {key: (value - min_value) / span for key, value in values.items()}

    def rank(
        self,
        candidates: list[B1TaskState],
        current_time_s: float,
        link_trace: LinkTrace,
        resource_state: ResourceState,
        simulator_cfg: dict[str, Any],
        current_window_capacity_mb: float,
    ) -> tuple[list[B1ScheduleDecision], list[B1TaskState]]:
        raw_density = {
            item.task.task_id: item.task.task_value / max(item.compressed_size_mb, 1.0e-9) for item in candidates
        }
        raw_urgency = {
            item.task.task_id: math.exp(
                -max(0.0, item.deadline_time_s - current_time_s) / max(self.cfg.deadline_urgency_tau_s, 1.0e-9)
            )
            for item in candidates
        }
        raw_age = {
            item.task.task_id: max(0.0, current_time_s - (item.queue_enter_time_s or current_time_s)) for item in candidates
        }
        density = self._normalize(raw_density)
        urgency = self._normalize(raw_urgency)
        age = self._normalize(raw_age)

        decisions: list[B1ScheduleDecision] = []
        feasible: list[B1TaskState] = []
        for item in candidates:
            full_fit = int(item.remaining_downlink_mb <= current_window_capacity_mb + 1.0e-9)
            reason = "candidate"
            if current_time_s > item.deadline_time_s:
                reason = "expired"
            elif self.cfg.require_full_fit_in_current_contact and not full_fit:
                reason = "no_full_fit_current_contact"
                self.no_fit_count += 1
            elif not self._hard_safety_feasible(item, link_trace, resource_state, simulator_cfg):
                reason = "blocked_by_hard_safety"
            score = (
                self.cfg.w_density * density.get(item.task.task_id, 0.0)
                + self.cfg.w_urgency * urgency.get(item.task.task_id, 0.0)
                + self.cfg.w_age * age.get(item.task.task_id, 0.0)
                + self.cfg.w_fit * full_fit
            )
            item.scheduler_score = score
            decisions.append(
                B1ScheduleDecision(
                    task_id=item.task.task_id,
                    score=score,
                    value_density=raw_density[item.task.task_id],
                    deadline_urgency=raw_urgency[item.task.task_id],
                    age_s=raw_age[item.task.task_id],
                    full_window_fit=full_fit,
                    reason=reason,
                )
            )
            if reason == "candidate":
                feasible.append(item)

        feasible.sort(
            key=lambda item: (
                -item.scheduler_score,
                item.deadline_time_s,
                -(item.task.task_value / max(item.compressed_size_mb, 1.0e-9)),
                -max(0.0, current_time_s - (item.queue_enter_time_s or current_time_s)),
                item.task.task_id,
            )
        )
        for index, item in enumerate(feasible, start=1):
            item.scheduler_rank = index
        return decisions, feasible

    def select(
        self,
        candidates: list[B1TaskState],
        current_time_s: float,
        link_trace: LinkTrace,
        resource_state: ResourceState,
        simulator_cfg: dict[str, Any],
        current_window_capacity_mb: float,
    ) -> tuple[B1TaskState | None, list[B1ScheduleDecision]]:
        decisions, feasible = self.rank(
            candidates, current_time_s, link_trace, resource_state, simulator_cfg, current_window_capacity_mb
        )
        if self.cfg.policy == "fifo":
            fifo_feasible = sorted(
                feasible,
                key=lambda item: (
                    item.queue_enter_time_s if item.queue_enter_time_s is not None else item.capture_time_s,
                    item.task.task_id,
                ),
            )
            selected = fifo_feasible[0] if fifo_feasible else None
            return selected, decisions
        return (feasible[0] if feasible else None), decisions

    def _hard_safety_feasible(
        self,
        item: B1TaskState,
        link_trace: LinkTrace,
        resource_state: ResourceState,
        simulator_cfg: dict[str, Any],
    ) -> bool:
        rate_mb_s = max(0.0, link_trace.throughput_mbps / 8.0 * float(simulator_cfg.get("downlink_efficiency", 1.0)))
        tx_time_s = item.remaining_downlink_mb / max(rate_mb_s, 1.0e-9)
        comm_energy_j = compute_comm_energy_j(
            comm_power_w=float(simulator_cfg.get("comm_power_w", simulator_cfg.get("tx_power_w", 0.0))),
            tx_time_s=tx_time_s,
            link_quality=link_trace.link_quality,
            alpha_gamma=float(simulator_cfg.get("comm_alpha_gamma", 0.0)),
        )
        next_battery = update_battery_soc(
            battery_soc=resource_state.battery_soc,
            solar_energy_j=0.0,
            base_power_w=0.0,
            slot_seconds=0.0,
            compute_energy_j=0.0,
            comm_energy_j=comm_energy_j,
            battery_capacity_j=float(simulator_cfg.get("battery_capacity_j", 1.0)),
            charge_efficiency=float(simulator_cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(simulator_cfg.get("battery_discharge_efficiency", 1.0)),
        )
        next_thermal = update_thermal_pressure(
            thermal_pressure=resource_state.thermal_pressure,
            compute_energy_j=0.0,
            comm_energy_j=comm_energy_j,
            cooling=0.0,
            slot_seconds=tx_time_s,
            reference_slot_seconds=float(simulator_cfg.get("thermal_reference_slot_seconds", simulator_cfg.get("slot_seconds", 60.0))),
        )
        eps = self.cfg.hard_safety_epsilon
        return (
            next_battery >= float(simulator_cfg.get("battery_min", 0.0)) - eps
            and next_thermal <= float(simulator_cfg.get("thermal_max", 1.0)) + eps
        )


class B1EnhancedSimulator:
    """Slot-based B1-Enhanced lifecycle simulator."""

    def __init__(self, config: B1EnhancedConfig, simulator_cfg: dict[str, Any], model_pool_cfg: dict[str, Any]) -> None:
        self.config = config
        self.simulator_cfg = dict(simulator_cfg)
        if "battery_capacity_wh" in self.simulator_cfg and "battery_capacity_j" not in self.simulator_cfg:
            self.simulator_cfg["battery_capacity_j"] = float(self.simulator_cfg["battery_capacity_wh"]) * 3600.0
        self.model_registry = ModelRegistry(model_pool_cfg)
        self.ground_profile = self.model_registry.get_profile(config.ground.inference_model_name)
        self.scheduler = ContactAwareCompressedGroundScheduler(config.scheduler)
        self.tasks: dict[str, B1TaskState] = {}
        self.queue: list[B1TaskState] = []
        self.current_tx: B1TaskState | None = None
        self.completed_window_ids: set[str] = set()
        self.window_utilization: dict[str, dict[str, float]] = {}
        self.rows_per_task: list[dict[str, Any]] = []
        self.queue_trace: list[dict[str, Any]] = []
        self.soc_trace: list[dict[str, Any]] = []
        self.contact_window_schedule: list[dict[str, Any]] = []
        self.scheduler_candidate_trace: list[dict[str, Any]] = []
        self.compression_trace: list[dict[str, Any]] = []
        self.raw_size_mapping_audit: list[dict[str, Any]] = []
        self.lifecycle_rows: list[dict[str, Any]] = []
        self.scheduler_no_fit_count = 0
        self.queue_overflow_count = 0
        self.compression_failed_or_blocked_count = 0
        self.thermal_safe_capacity_trace: list[dict[str, Any]] = []
        self.thermal_hysteresis_trace: list[dict[str, Any]] = []
        self._thermal_headroom_recovery_mode = False

    def _raw_size_mb(self, task: TaskSample) -> tuple[float, str]:
        for key in self.config.raw_size.metadata_keys:
            value = task.metadata.get(key)
            try:
                if value is not None and value != "":
                    return max(0.0, float(value)), f"task.metadata.{key}"
            except (TypeError, ValueError):
                continue
        return self.config.raw_size.default_raw_size_mb, "baseline_b1.raw_size.default_raw_size_mb"

    def _ground_quality(self, task: TaskSample) -> float:
        _ = task
        profile_quality = max(0.0, min(1.0, self.ground_profile.quality_base))
        retained = max(0.0, min(1.0, self.config.compression.downstream_quality_retention))
        return max(0.0, min(1.0, profile_quality * retained))

    def _create_task(self, step: TimelineStep) -> None:
        if step.task is None:
            return
        task = step.task
        raw_mb, source = self._raw_size_mb(task)
        compressed_mb = compressed_size_mb(raw_mb, self.config.compression.compression_ratio)
        latency = compression_latency_s(raw_mb, self.config.compression)
        energy_wh = compression_energy_wh(latency, self.config.compression.compression_power_w)
        current_time_s = step.slot_index * step.slot_seconds
        state = B1TaskState(
            task=task,
            created_slot=step.slot_index,
            capture_time_s=current_time_s,
            deadline_time_s=current_time_s + task.deadline_s,
            raw_size_mb=raw_mb,
            compressed_size_mb=compressed_mb,
            compression_latency_s=latency,
            compression_energy_wh=energy_wh,
            reconstructed_image_quality_proxy=self.config.compression.reconstructed_ssim_proxy,
            ground_task_quality=self._ground_quality(task),
            compression_remaining_s=latency,
            remaining_downlink_mb=compressed_mb,
            ground_decode_latency_s=self.config.ground.decode_latency_s_per_reference_image
            * raw_mb
            / max(self.config.compression.reference_raw_size_mb, 1.0e-9),
            ground_inference_latency_s=float(
                self.config.ground.inference_latency_s
                if self.config.ground.inference_latency_s is not None
                else self.ground_profile.latency_s
            ),
        )
        self.tasks[task.task_id] = state
        self.raw_size_mapping_audit.append(
            {
                "task_id": task.task_id,
                "raw_size_mb": raw_mb,
                "raw_size_source": source,
                "compressed_size_mb": compressed_mb,
                "compression_ratio": self.config.compression.compression_ratio,
                "payload_type": PAYLOAD_TYPE,
            }
        )
        self.compression_trace.append(
            {
                "task_id": task.task_id,
                "slot_index": step.slot_index,
                "state": "compression_pending",
                "raw_size_mb": raw_mb,
                "compressed_size_mb": compressed_mb,
                "compression_latency_s": latency,
                "compression_energy_wh": energy_wh,
                "compression_power_w": self.config.compression.compression_power_w,
                "compression_power_assumption_source": self.config.compression.compression_power_assumption_source,
            }
        )

    def _process_compression(self, step: TimelineStep) -> float:
        current_time_s = step.slot_index * step.slot_seconds
        compute_energy_j = 0.0
        for item in self.tasks.values():
            if item.state not in {"compression_pending", "compressing"}:
                continue
            if current_time_s > item.deadline_time_s:
                item.state = "expired"
                item.expired_recorded = True
                continue
            item.state = "compressing"
            work_s = min(step.slot_seconds, item.compression_remaining_s)
            planned_energy_j = compute_inference_energy_j(self.config.compression.compression_power_w, work_s)
            if not self._compute_safety_feasible(planned_energy_j, step):
                item.state = "blocked"
                item.blocked_reason = "compression_blocked_by_hard_safety"
                self.compression_failed_or_blocked_count += 1
                continue
            item.compression_remaining_s = max(0.0, item.compression_remaining_s - work_s)
            compute_energy_j += planned_energy_j
            self.compression_trace.append(
                {
                    "task_id": item.task.task_id,
                    "slot_index": step.slot_index,
                    "state": "compressing",
                    "compression_work_s": work_s,
                    "compression_energy_wh": planned_energy_j / 3600.0,
                    "remaining_compression_s": item.compression_remaining_s,
                }
            )
            if item.compression_remaining_s <= 1.0e-9:
                item.state = "compressed_queued"
                item.compressed_available_time_s = current_time_s + work_s
                item.queue_enter_time_s = item.compressed_available_time_s
                self._enqueue(item)
        return compute_energy_j

    def _compute_safety_feasible(self, compute_energy_j: float, step: TimelineStep) -> bool:
        battery = update_battery_soc(
            battery_soc=float(getattr(self, "_battery_soc", self.simulator_cfg.get("battery_init", 0.85))),
            solar_energy_j=0.0,
            base_power_w=0.0,
            slot_seconds=0.0,
            compute_energy_j=compute_energy_j,
            comm_energy_j=0.0,
            battery_capacity_j=float(self.simulator_cfg.get("battery_capacity_j", 1.0)),
            charge_efficiency=float(self.simulator_cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(self.simulator_cfg.get("battery_discharge_efficiency", 1.0)),
        )
        thermal = update_thermal_pressure(
            thermal_pressure=float(getattr(self, "_thermal_pressure", self.simulator_cfg.get("thermal_init", 0.25))),
            compute_energy_j=compute_energy_j,
            comm_energy_j=0.0,
            cooling=0.0,
            slot_seconds=step.slot_seconds,
            reference_slot_seconds=float(self.simulator_cfg.get("thermal_reference_slot_seconds", step.slot_seconds)),
            temperature_reference_c=step.resource_trace.temperature_reference_c,
            is_sunlit=step.resource_trace.is_sunlit,
            passive_cooling=float(self.simulator_cfg.get("thermal_passive_cooling", 0.12)),
            sunlight_gain=float(self.simulator_cfg.get("thermal_sunlight_gain", 0.015)),
            eclipse_cooling_bonus=float(self.simulator_cfg.get("thermal_eclipse_cooling_bonus", 0.03)),
            reference_min_c=float(self.simulator_cfg.get("thermal_reference_min_c", -10.0)),
            reference_max_c=float(self.simulator_cfg.get("thermal_reference_max_c", 30.0)),
        )
        return battery >= float(self.simulator_cfg.get("battery_min", 0.0)) and thermal <= float(
            self.simulator_cfg.get("thermal_max", 1.0)
        )

    def _enqueue(self, item: B1TaskState) -> None:
        queue_mb = sum(task.remaining_downlink_mb for task in self.queue)
        if queue_mb + item.remaining_downlink_mb > float(self.simulator_cfg.get("queue_max_mb", 0.0)) + 1.0e-9:
            item.state = "blocked"
            item.blocked_reason = "queue_overflow_blocked"
            self.queue_overflow_count += 1
            self.compression_failed_or_blocked_count += 1
            return
        self.queue.append(item)

    def _window_id(self, step: TimelineStep) -> str:
        if not step.link_trace.in_contact:
            return ""
        start_s = step.slot_index * step.slot_seconds - (
            max(0.0, step.link_trace.remaining_window_s - step.slot_seconds)
        )
        return f"window_{int(round(start_s / max(step.slot_seconds, 1.0e-9))):06d}"

    def _remaining_window_capacity_mb(self, step: TimelineStep) -> float:
        if not step.link_trace.in_contact:
            return 0.0
        rate = step.link_trace.throughput_mbps * float(self.simulator_cfg.get("downlink_efficiency", 1.0)) / 8.0
        return max(0.0, rate * step.link_trace.remaining_window_s)

    def _process_downlink(
        self,
        step: TimelineStep,
        resource_state: ResourceState,
        baseline_compute_energy_j: float = 0.0,
    ) -> tuple[float, float]:
        if not step.link_trace.in_contact:
            self.current_tx = None
            return 0.0, 0.0
        current_time_s = step.slot_index * step.slot_seconds
        window_id = self._window_id(step)
        rate_mb_s = step.link_trace.throughput_mbps * float(self.simulator_cfg.get("downlink_efficiency", 1.0)) / 8.0
        slot_capacity_mb = rate_mb_s * step.slot_seconds
        window_capacity_mb = self._remaining_window_capacity_mb(step)

        for item in list(self.queue):
            if current_time_s > item.deadline_time_s and item.state != "transmitting":
                item.state = "expired"
                item.expired_recorded = True
                self.queue.remove(item)

        if self.current_tx is None or self.current_tx.state != "transmitting":
            candidates = [
                item
                for item in self.queue
                if item.state == "compressed_queued"
                and (item.queue_enter_time_s is None or item.queue_enter_time_s <= current_time_s + 1.0e-9)
            ]
            selected, decisions = self.scheduler.select(
                candidates,
                current_time_s,
                step.link_trace,
                resource_state,
                self.simulator_cfg,
                window_capacity_mb,
            )
            feasible_ids = [decision.task_id for decision in decisions if decision.reason == "candidate"]
            self.scheduler_candidate_trace.append(
                {
                    "slot_index": step.slot_index,
                    "contact_window_id": window_id,
                    "queued_candidate_count_before_filter": sum(
                        item.state == "compressed_queued" for item in self.queue
                    ),
                    "eligible_candidate_count_before_feasibility": len(candidates),
                    "feasible_candidate_count_after_filter": len(feasible_ids),
                    "rejected_expired_count": sum(decision.reason == "expired" for decision in decisions),
                    "rejected_deadline_or_contact_count": sum(
                        decision.reason in {"cannot_finish_before_deadline", "no_full_fit_current_contact"}
                        for decision in decisions
                    ),
                    "rejected_safety_count": sum(decision.reason == "hard_safety_infeasible" for decision in decisions),
                    "candidate_task_ids_before_feasibility": ";".join(item.task.task_id for item in candidates),
                    "feasible_task_ids_after_filter": ";".join(feasible_ids),
                    "selected_task_id": selected.task.task_id if selected is not None else "",
                    "scheduler_policy": self.config.scheduler.policy,
                }
            )
            self.scheduler_no_fit_count = self.scheduler.no_fit_count
            for decision in decisions:
                self.contact_window_schedule.append(
                    {
                        **asdict(decision),
                        "slot_index": step.slot_index,
                        "contact_window_id": window_id,
                        "remaining_window_capacity_mb": window_capacity_mb,
                        "selected": int(selected is not None and decision.task_id == selected.task.task_id),
                    }
                )
            if selected is not None:
                selected.state = "transmitting"
                selected.transmission_start_time_s = current_time_s
                selected.contact_wait_latency_s = max(0.0, current_time_s - (selected.queue_enter_time_s or current_time_s))
                selected.queue_wait_latency_s = 0.0
                selected.window_id = window_id
                self.current_tx = selected

        if self.current_tx is None:
            return 0.0, 0.0

        item = self.current_tx
        sent_mb = min(slot_capacity_mb, item.remaining_downlink_mb)
        headroom_cfg = thermal_headroom_config(self.simulator_cfg)
        thermal_decision = None
        if headroom_cfg["enabled"]:
            previous_recovery = bool(self._thermal_headroom_recovery_mode)
            event = ""
            if headroom_cfg["hysteresis_enabled"]:
                if previous_recovery and self._thermal_pressure < headroom_cfg["guard"]:
                    self._thermal_headroom_recovery_mode = False
                    event = "EXIT_RECOVERY"
                elif self._thermal_pressure >= headroom_cfg["hard"] and not previous_recovery:
                    self._thermal_headroom_recovery_mode = True
                    event = "ENTER_HARD_PAUSE"
            thermal_decision = maximum_safe_downlink_capacity(
                current_thermal=self._thermal_pressure,
                nominal_capacity_mb=sent_mb,
                rate_mb_s=rate_mb_s,
                link_quality=step.link_trace.link_quality,
                compute_energy_j=baseline_compute_energy_j,
                compute_heat_gain_scale=1.0,
                slot_seconds=step.slot_seconds,
                temperature_reference_c=step.resource_trace.temperature_reference_c,
                is_sunlit=step.resource_trace.is_sunlit,
                simulator_cfg=self.simulator_cfg,
                force_hard_pause=bool(headroom_cfg["hysteresis_enabled"] and self._thermal_headroom_recovery_mode),
            )
            if headroom_cfg["hysteresis_enabled"] and thermal_decision.zero_capacity_reason == "COMPUTE_ALONE_EXHAUSTS_HEADROOM":
                self._thermal_headroom_recovery_mode = True
                event = "ENTER_HARD_PAUSE"
            sent_mb = min(sent_mb, thermal_decision.safe_capacity_mb)
            self.thermal_hysteresis_trace.append({"slot":step.slot_index,"current_thermal":self._thermal_pressure,
                "recovery_mode_before":int(previous_recovery),"recovery_mode_after":int(self._thermal_headroom_recovery_mode),"event":event})
            self.thermal_safe_capacity_trace.append({"slot":step.slot_index,"current_thermal":self._thermal_pressure,
                "nominal_capacity_bytes":thermal_decision.nominal_capacity_mb*1_000_000.0,
                "safe_capacity_bytes":thermal_decision.safe_capacity_mb*1_000_000.0,
                "predicted_thermal_at_nominal_capacity":thermal_decision.predicted_nominal_thermal,
                "predicted_thermal_at_safe_capacity":thermal_decision.predicted_safe_thermal,
                "search_iterations":thermal_decision.search_iterations,"zero_capacity_reason":thermal_decision.zero_capacity_reason,
                "monotonic":int(thermal_decision.monotonic),"actual_transmitted_bytes":0.0,"realized_next_thermal":""})
        tx_time_s = sent_mb / max(rate_mb_s, 1.0e-9)
        comm_energy_j = compute_comm_energy_j(
            comm_power_w=float(self.simulator_cfg.get("comm_power_w", self.simulator_cfg.get("tx_power_w", 0.0))),
            tx_time_s=tx_time_s,
            link_quality=step.link_trace.link_quality,
            alpha_gamma=float(self.simulator_cfg.get("comm_alpha_gamma", 0.0)),
        )
        if not self._slot_action_safety_feasible(baseline_compute_energy_j, comm_energy_j, step):
            item.state = "compressed_queued"
            item.blocked_reason = "transmission_blocked_by_hard_safety"
            self.current_tx = None
            return 0.0, 0.0
        item.remaining_downlink_mb = max(0.0, item.remaining_downlink_mb - sent_mb)
        item.downlinked_mb += sent_mb
        item.transmission_latency_s += tx_time_s
        self.window_utilization.setdefault(window_id, {"used_mb": 0.0, "capacity_mb": 0.0})
        self.window_utilization[window_id]["used_mb"] += sent_mb
        self.window_utilization[window_id]["capacity_mb"] += slot_capacity_mb
        if item.remaining_downlink_mb <= 1.0e-9:
            item.state = "ground_received_compressed"
            finish_time_s = current_time_s + tx_time_s
            item.ground_received_compressed_time_s = finish_time_s
            item.ground_decode_start_time_s = finish_time_s
            item.ground_decode_end_time_s = finish_time_s + item.ground_decode_latency_s
            item.ground_inference_start_time_s = item.ground_decode_end_time_s
            item.ground_result_available_time_s = item.ground_inference_start_time_s + item.ground_inference_latency_s
            item.capture_to_ground_latency_s = item.ground_result_available_time_s - item.capture_time_s
            item.queue_wait_latency_s = 0.0
            item.late_delivery = item.ground_result_available_time_s > item.deadline_time_s + 1.0e-9
            item.state = GROUND_RESULT_STATE
            if item in self.queue:
                self.queue.remove(item)
            self.current_tx = None
            self.completed_window_ids.add(window_id)
        return sent_mb, comm_energy_j

    def _slot_action_safety_feasible(self, compute_energy_j: float, comm_energy_j: float, step: TimelineStep) -> bool:
        next_battery = update_battery_soc(
            battery_soc=self._battery_soc,
            solar_energy_j=step.resource_trace.solar_energy_j,
            base_power_w=step.resource_trace.base_power_w,
            slot_seconds=step.slot_seconds,
            compute_energy_j=compute_energy_j,
            comm_energy_j=comm_energy_j,
            battery_capacity_j=float(self.simulator_cfg.get("battery_capacity_j", 1.0)),
            charge_efficiency=float(self.simulator_cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(self.simulator_cfg.get("battery_discharge_efficiency", 1.0)),
        )
        next_thermal = update_thermal_pressure(
            thermal_pressure=self._thermal_pressure,
            compute_energy_j=compute_energy_j,
            comm_energy_j=comm_energy_j,
            cooling=float(self.simulator_cfg.get("cooling_per_slot", 0.05)),
            slot_seconds=step.slot_seconds,
            reference_slot_seconds=float(self.simulator_cfg.get("thermal_reference_slot_seconds", step.slot_seconds)),
            temperature_reference_c=step.resource_trace.temperature_reference_c,
            is_sunlit=step.resource_trace.is_sunlit,
            passive_cooling=float(self.simulator_cfg.get("thermal_passive_cooling", 0.12)),
            sunlight_gain=float(self.simulator_cfg.get("thermal_sunlight_gain", 0.015)),
            eclipse_cooling_bonus=float(self.simulator_cfg.get("thermal_eclipse_cooling_bonus", 0.03)),
            reference_min_c=float(self.simulator_cfg.get("thermal_reference_min_c", -10.0)),
            reference_max_c=float(self.simulator_cfg.get("thermal_reference_max_c", 30.0)),
        )
        return (
            next_battery >= float(self.simulator_cfg.get("battery_min", 0.0))
            and next_thermal <= float(self.simulator_cfg.get("thermal_max", 1.0))
        )

    def run(self, timeline: list[TimelineStep]) -> dict[str, Any]:
        self._battery_soc = float(self.simulator_cfg.get("battery_init", 0.85))
        self._thermal_pressure = float(self.simulator_cfg.get("thermal_init", 0.25))
        min_soc = self._battery_soc
        total_comm_energy_j = 0.0
        total_compute_energy_j = 0.0

        for step in timeline:
            self._create_task(step)
            compute_energy_j = self._process_compression(step)
            resource_state = ResourceState(
                battery_soc=self._battery_soc,
                ai_energy_budget=float(self.simulator_cfg.get("ai_budget_init", 1.0)),
                thermal_pressure=self._thermal_pressure,
            )
            sent_mb, comm_energy_j = self._process_downlink(step, resource_state, compute_energy_j)
            total_compute_energy_j += compute_energy_j
            total_comm_energy_j += comm_energy_j
            self._battery_soc = update_battery_soc(
                battery_soc=self._battery_soc,
                solar_energy_j=step.resource_trace.solar_energy_j,
                base_power_w=step.resource_trace.base_power_w,
                slot_seconds=step.slot_seconds,
                compute_energy_j=compute_energy_j,
                comm_energy_j=comm_energy_j,
                battery_capacity_j=float(self.simulator_cfg.get("battery_capacity_j", 1.0)),
                charge_efficiency=float(self.simulator_cfg.get("battery_charge_efficiency", 1.0)),
                discharge_efficiency=float(self.simulator_cfg.get("battery_discharge_efficiency", 1.0)),
            )
            self._thermal_pressure = update_thermal_pressure(
                thermal_pressure=self._thermal_pressure,
                compute_energy_j=compute_energy_j,
                comm_energy_j=comm_energy_j,
                cooling=float(self.simulator_cfg.get("cooling_per_slot", 0.05)),
                compute_heat_gain_scale=1.0,
                comm_heat_gain_scale=1.0,
                slot_seconds=step.slot_seconds,
                reference_slot_seconds=float(self.simulator_cfg.get("thermal_reference_slot_seconds", step.slot_seconds)),
                temperature_reference_c=step.resource_trace.temperature_reference_c,
                is_sunlit=step.resource_trace.is_sunlit,
                passive_cooling=float(self.simulator_cfg.get("thermal_passive_cooling", 0.12)),
                sunlight_gain=float(self.simulator_cfg.get("thermal_sunlight_gain", 0.015)),
                eclipse_cooling_bonus=float(self.simulator_cfg.get("thermal_eclipse_cooling_bonus", 0.03)),
                reference_min_c=float(self.simulator_cfg.get("thermal_reference_min_c", -10.0)),
                reference_max_c=float(self.simulator_cfg.get("thermal_reference_max_c", 30.0)),
            )
            if self.thermal_safe_capacity_trace and int(self.thermal_safe_capacity_trace[-1]["slot"]) == int(step.slot_index):
                self.thermal_safe_capacity_trace[-1]["actual_transmitted_bytes"] = sent_mb * 1_000_000.0
                self.thermal_safe_capacity_trace[-1]["realized_next_thermal"] = self._thermal_pressure
            min_soc = min(min_soc, self._battery_soc)
            queue_mb = sum(item.remaining_downlink_mb for item in self.queue)
            self.queue_trace.append(
                {
                    "slot_index": step.slot_index,
                    "time_s": step.slot_index * step.slot_seconds,
                    "queue_mb": queue_mb,
                    "queue_item_count": len(self.queue),
                    "in_contact": int(step.link_trace.in_contact),
                    "sent_mb": sent_mb,
                    "current_transmitting_task_id": self.current_tx.task.task_id if self.current_tx else "",
                }
            )
            self.soc_trace.append(
                {
                    "slot_index": step.slot_index,
                    "time_s": step.slot_index * step.slot_seconds,
                    "battery_soc": self._battery_soc,
                    "SOC": self._battery_soc,
                    "timestamp_s": step.slot_index * step.slot_seconds,
                    "battery_energy_wh": self._battery_soc * float(self.simulator_cfg.get("battery_capacity_j", 0.0)) / 3600.0,
                    "thermal_pressure": self._thermal_pressure,
                    "thermal_state": self._thermal_pressure,
                    "temperature_reference_c": step.resource_trace.temperature_reference_c,
                    "active_task_phase": "compression+downlink" if compute_energy_j > 0 and comm_energy_j > 0 else ("compression" if compute_energy_j > 0 else ("downlink" if comm_energy_j > 0 else "idle")),
                    "compression_activity": int(compute_energy_j > 0),
                    "compute_activity": int(compute_energy_j > 0),
                    "downlink_activity": int(comm_energy_j > 0),
                    "compression_energy_wh": compute_energy_j / 3600.0,
                    "communication_energy_wh": comm_energy_j / 3600.0,
                    "base_power_w": step.resource_trace.base_power_w,
                    "solar_energy_j": step.resource_trace.solar_energy_j,
                }
            )

        horizon_end_s = len(timeline) * float(timeline[0].slot_seconds if timeline else self.simulator_cfg.get("slot_seconds", 1.0))
        for item in self.tasks.values():
            if item.state not in {GROUND_RESULT_STATE, "expired", "blocked"}:
                item.state = "expired" if horizon_end_s > item.deadline_time_s else item.state
        self.rows_per_task = [self._task_row(item) for item in sorted(self.tasks.values(), key=lambda x: x.task.task_id)]
        self.lifecycle_rows = self._lifecycle_counts()
        summary = self._summary_metrics(min_soc, total_compute_energy_j, total_comm_energy_j, timeline)
        return {
            "summary": summary,
            "per_task_delivery_trace": self.rows_per_task,
            "queue_trace": self.queue_trace,
            "soc_trace": self.soc_trace,
            "contact_window_schedule": self.contact_window_schedule,
            "scheduler_candidate_trace": self.scheduler_candidate_trace,
            "compression_trace": self.compression_trace,
            "thermal_safe_capacity_trace": self.thermal_safe_capacity_trace,
            "thermal_hysteresis_trace": self.thermal_hysteresis_trace,
            "lifecycle_counts": self.lifecycle_rows,
            "raw_size_mapping_audit": self.raw_size_mapping_audit,
            "compression_parameter_audit": self._compression_parameter_audit(),
            "ground_model_profile_audit": self._ground_model_profile_audit(),
        }

    def _compression_parameter_audit(self) -> list[dict[str, Any]]:
        c = self.config.compression
        return [
            {"parameter": "compression_ratio", "value": c.compression_ratio, "unit": "raw_MB/compressed_MB", "source_type": "paper_measured", "source_description": "Phi-Sat-2 CAE reference compression ratio"},
            {"parameter": "reference_raw_size_mb", "value": c.reference_raw_size_mb, "unit": "MB", "source_type": "paper_measured", "source_description": "reference single-band image size"},
            {"parameter": "reference_compressed_size_mb", "value": c.reference_compressed_size_mb, "unit": "MB", "source_type": "paper_measured", "source_description": "CR=8 reference compressed image size"},
            {"parameter": "reference_onboard_latency_s", "value": c.reference_onboard_latency_s, "unit": "s", "source_type": "paper_measured", "source_description": "reported approximately 10.92/11 s reference latency"},
            {"parameter": "reconstructed_image_quality_proxy", "value": c.reconstructed_ssim_proxy, "unit": "SSIM proxy", "source_type": "paper_measured", "source_description": "image reconstruction quality only; not detector accuracy"},
            {"parameter": "paper_accelerator_power_w", "value": c.paper_accelerator_power_w, "unit": "W", "source_type": "paper_measured", "source_description": "reference accelerator power sensitivity point"},
            {"parameter": "compression_power_w", "value": c.compression_power_w, "unit": "W", "source_type": "engineering_assumed", "source_description": c.compression_power_assumption_source},
        ]

    def _ground_model_profile_audit(self) -> list[dict[str, Any]]:
        return [{"ground_model_name": self.config.ground.inference_model_name, "ground_model_base_quality": self.ground_profile.quality_base, "ground_inference_latency_s": self.config.ground.inference_latency_s if self.config.ground.inference_latency_s is not None else self.ground_profile.latency_s, "downstream_quality_retention": self.config.compression.downstream_quality_retention, "quality_formula": "ground_model_base_quality * downstream_quality_retention", "ground_energy_counted_in_satellite_soc": False}]

    def _task_row(self, item: B1TaskState) -> dict[str, Any]:
        completion_time = item.ground_result_available_time_s
        return {
            "task_id": item.task.task_id,
            "scene_id": item.task.scene_id,
            "task_class": item.task.task_class,
            "value_level": item.task.value_level,
            "product_requirement": item.task.product_requirement,
            "quality_requirement": item.task.quality_requirement,
            "information_requirement": item.task.information_requirement,
            "high_value_flag": int(item.high_value),
            "task_value": item.task.task_value,
            "deadline_s": item.task.deadline_s,
            "capture_time_s": item.capture_time_s,
            "deadline_time_s": item.deadline_time_s,
            "state": item.state,
            "payload_type": PAYLOAD_TYPE,
            "raw_size_mb": item.raw_size_mb,
            "compressed_size_mb": item.compressed_size_mb,
            "downlinked_mb": item.downlinked_mb,
            "compression_ratio": self.config.compression.compression_ratio,
            "onboard_compression_latency_s": item.compression_latency_s,
            "compression_latency_s": item.compression_latency_s,
            "compression_energy_wh": item.compression_energy_wh,
            "queue_wait_latency_s": item.queue_wait_latency_s,
            "contact_wait_latency_s": item.contact_wait_latency_s,
            "queue_and_contact_wait_latency_s": item.queue_wait_latency_s + item.contact_wait_latency_s,
            "transmission_latency_s": item.transmission_latency_s,
            "ground_decode_latency_s": item.ground_decode_latency_s,
            "ground_inference_latency_s": item.ground_inference_latency_s,
            "capture_to_ground_latency_s": item.capture_to_ground_latency_s,
            "latency_component_sum_s": (
                item.compression_latency_s
                + item.queue_wait_latency_s
                + item.contact_wait_latency_s
                + item.transmission_latency_s
                + item.ground_decode_latency_s
                + item.ground_inference_latency_s
            )
            if item.state == GROUND_RESULT_STATE
            else 0.0,
            "ground_received_compressed_time_s": item.ground_received_compressed_time_s if item.ground_received_compressed_time_s is not None else "",
            "ground_result_available_time_s": completion_time if completion_time is not None else "",
            "ground_result_available": int(item.state == GROUND_RESULT_STATE),
            "late_delivery": int(item.late_delivery),
            "deadline_satisfied": int(item.state == GROUND_RESULT_STATE and not item.late_delivery),
            "reconstructed_image_quality_proxy": item.reconstructed_image_quality_proxy,
            "ground_task_quality": item.ground_task_quality,
            "ground_received_quality": item.ground_task_quality,
            "blocked_reason": item.blocked_reason,
            "scheduler_score": item.scheduler_score,
            "scheduler_rank": item.scheduler_rank,
            "window_id": item.window_id,
        }

    def _lifecycle_counts(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for item in self.tasks.values():
            counts[item.state] = counts.get(item.state, 0) + 1
        return [{"lifecycle_state": key, "count": counts[key]} for key in sorted(counts)]

    def _summary_metrics(
        self,
        min_soc: float,
        total_compute_energy_j: float,
        total_comm_energy_j: float,
        timeline: list[TimelineStep],
    ) -> dict[str, Any]:
        tasks = list(self.tasks.values())
        completed = [item for item in tasks if item.state == GROUND_RESULT_STATE]
        high_completed = [item for item in completed if item.high_value]
        high_tasks = [item for item in tasks if item.high_value]
        latencies = [item.capture_to_ground_latency_s for item in completed]
        high_qualities = [item.ground_task_quality for item in high_completed]
        urgent_tasks = [item for item in tasks if item.task.emergency_flag or item.task.deadline_s <= 60.0]
        deadline_ok = [item for item in completed if not item.late_delivery]
        high_deadline_ok = [item for item in high_completed if not item.late_delivery]
        urgent_deadline_ok = [item for item in urgent_tasks if item.state == GROUND_RESULT_STATE and not item.late_delivery]
        total_raw = sum(item.raw_size_mb for item in tasks)
        potential_compressed = sum(item.compressed_size_mb for item in tasks)
        actually_compressed = sum(item.compressed_size_mb for item in tasks if item.compressed_available_time_s is not None)
        total_downlinked = sum(item.downlinked_mb for item in tasks)
        queue_values = [row["queue_mb"] for row in self.queue_trace]
        utilizations = [
            values["used_mb"] / max(values["capacity_mb"], 1.0e-9)
            for values in self.window_utilization.values()
            if values["capacity_mb"] > 0.0
        ]
        delivered_effective_utility = sum(item.task.task_value * item.ground_task_quality for item in completed)
        task_energy_wh = (total_compute_energy_j + total_comm_energy_j) / 3600.0
        base_energy_wh = sum(float(step.resource_trace.base_power_w) * float(step.slot_seconds) for step in timeline) / 3600.0
        total_sat_energy_wh = task_energy_wh + base_energy_wh
        delivered_task_net_utility_sum = delivered_effective_utility
        high_value_delivered_net_utility_sum = sum(item.task.task_value * item.ground_task_quality for item in high_completed)
        on_time_count = len(deadline_ok)
        high_value_first_latency_values = [item.capture_to_ground_latency_s for item in high_completed]
        thermal_violations = sum(
            1 for row in self.soc_trace if float(row["thermal_pressure"]) > float(self.simulator_cfg.get("thermal_max", 1.0))
        )
        hard_safety = thermal_violations + sum(
            1 for row in self.soc_trace if float(row["battery_soc"]) < float(self.simulator_cfg.get("battery_min", 0.0))
        )
        nslots = max(len(self.soc_trace), 1)
        net_system_utility = delivered_effective_utility - 0.01 * total_sat_energy_wh
        return {
            "method_name": METHOD_NAME,
            "scheduler_policy": self.config.scheduler.policy,
            "baseline_version": self.config.baseline_version,
            "total_generated_tasks": len(tasks),
            "ground_received_task_count": len(completed),
            "ground_received_task_ratio": len(completed) / max(len(tasks), 1),
            "ground_received_high_value_task_count": len(high_completed),
            "ground_received_high_value_task_ratio": len(high_completed) / max(len(high_tasks), 1),
            "average_ground_received_quality": statistics.fmean([item.ground_task_quality for item in completed]) if completed else 0.0,
            "average_high_value_received_quality": statistics.fmean(high_qualities) if high_qualities else 0.0,
            "average_capture_to_ground_latency_s": statistics.fmean(latencies) if latencies else 0.0,
            "median_capture_to_ground_latency_s": statistics.median(latencies) if latencies else 0.0,
            "p95_capture_to_ground_latency_s": sorted(latencies)[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)] if latencies else 0.0,
            "deadline_satisfaction_ratio": len(deadline_ok) / max(len(tasks), 1),
            "on_time_ground_received_count": on_time_count,
            "on_time_ratio_over_generated": on_time_count / max(len(tasks), 1),
            "on_time_ratio_among_ground_received": on_time_count / max(len(completed), 1),
            "urgent_task_deadline_satisfaction_ratio": len(urgent_deadline_ok) / max(len(urgent_tasks), 1),
            "high_value_deadline_satisfaction_ratio": len(high_deadline_ok) / max(len(high_tasks), 1),
            "target_stage_success_rate": len(completed) / max(len(tasks), 1),
            "high_value_target_stage_success_rate": len(high_completed) / max(len(high_tasks), 1),
            "high_value_any_useful_received_count": len(high_completed),
            "high_value_target_stage_success_count": len(high_completed),
            "high_value_any_useful_success_ratio": len(high_completed) / max(len(high_tasks), 1),
            "high_value_target_stage_success_ratio": len(high_completed) / max(len(high_tasks), 1),
            "average_high_value_first_useful_latency_s": statistics.fmean(high_value_first_latency_values)
            if high_value_first_latency_values
            else 0.0,
            "average_high_value_target_completion_latency_s": statistics.fmean(high_value_first_latency_values)
            if high_value_first_latency_values
            else 0.0,
            "high_value_delivered_net_utility_sum": high_value_delivered_net_utility_sum,
            "late_ground_result_count": sum(1 for item in completed if item.late_delivery),
            "expired_before_ground_result_count": sum(1 for item in tasks if item.state == "expired"),
            "incomplete_downlink_task_count": sum(1 for item in tasks if item.downlinked_mb > 0.0 and item.state != GROUND_RESULT_STATE),
            "undelivered_by_end_count": sum(1 for item in tasks if item.state != GROUND_RESULT_STATE),
            "total_raw_volume_mb": total_raw,
            "potential_compressed_volume_mb": potential_compressed,
            "actually_compressed_volume_mb": actually_compressed,
            "actually_downlinked_volume_mb": total_downlinked,
            "total_compressed_volume_mb": potential_compressed,
            "potential_compressed_volume_mb": potential_compressed,
            "actually_compressed_volume_mb": actually_compressed,
            "actually_downlinked_volume_mb": total_downlinked,
            "achieved_data_reduction_ratio": total_raw / max(potential_compressed, 1.0e-9),
            "total_downlinked_volume_mb": total_downlinked,
            "average_downlink_queue_mb": statistics.fmean(queue_values) if queue_values else 0.0,
            "max_downlink_queue_mb": max(queue_values) if queue_values else 0.0,
            "queue_overflow_count": self.queue_overflow_count,
            "average_window_utilization": statistics.fmean(utilizations) if utilizations else 0.0,
            "completed_contact_windows": len(self.completed_window_ids),
            "scheduler_no_fit_count": self.scheduler_no_fit_count,
            "compression_completed_count": sum(1 for item in tasks if item.compressed_available_time_s is not None),
            "compression_failed_or_blocked_count": self.compression_failed_or_blocked_count,
            "total_compression_energy_wh": total_compute_energy_j / 3600.0,
            "compression_energy_completed_tasks_wh": sum(item.compression_energy_wh for item in tasks if item.compressed_available_time_s is not None),
            "partial_compression_energy_wh": max(0.0, total_compute_energy_j / 3600.0 - sum(item.compression_energy_wh for item in tasks if item.compressed_available_time_s is not None)),
            "compression_energy_expected_wh": sum(float(row.get("compression_energy_wh", 0.0)) for row in self.compression_trace if row.get("state") == "compressing"),
            "compression_energy_consistency_error_wh": total_compute_energy_j / 3600.0 - sum(float(row.get("compression_energy_wh", 0.0)) for row in self.compression_trace if row.get("state") == "compressing"),
            "total_communication_energy_wh": total_comm_energy_j / 3600.0,
            "total_housekeeping_energy_wh": base_energy_wh,
            "total_satellite_energy_wh": total_sat_energy_wh,
            "total_satellite_task_energy_wh": total_sat_energy_wh,
            "energy_per_ground_received_task_wh": total_sat_energy_wh / max(len(completed), 1),
            "satellite_energy_per_ground_received_task_wh": total_sat_energy_wh / max(len(completed), 1),
            "energy_per_high_value_ground_received_task_wh": (
                total_sat_energy_wh / len(high_completed) if high_completed else 0.0
            ),
            "satellite_energy_per_high_value_ground_received_task_wh": (
                total_sat_energy_wh / len(high_completed) if high_completed else 0.0
            ),
            "initial_soc": float(self.simulator_cfg.get("battery_init", 0.85)),
            "max_soc": max([float(row["battery_soc"]) for row in self.soc_trace] + [float(self.simulator_cfg.get("battery_init", 0.85))]),
            "min_soc": min_soc,
            "final_soc": self._battery_soc,
            "fraction_time_soc_below_70": sum(1 for row in self.soc_trace if float(row["battery_soc"]) < 0.70) / nslots,
            "fraction_time_soc_below_50": sum(1 for row in self.soc_trace if float(row["battery_soc"]) < 0.50) / nslots,
            "fraction_time_soc_below_20": sum(1 for row in self.soc_trace if float(row["battery_soc"]) < 0.20) / nslots,
            "hard_safety_violation_count": hard_safety,
            "thermal_violation_count": thermal_violations,
            "max_thermal_state": max([float(row["thermal_pressure"]) for row in self.soc_trace] + [float(self.simulator_cfg.get("thermal_init", 0.25))]),
            "min_thermal_state": min([float(row["thermal_pressure"]) for row in self.soc_trace] + [float(self.simulator_cfg.get("thermal_init", 0.25))]),
            "ground_model_name": self.config.ground.inference_model_name,
            "ground_model_base_quality": self.ground_profile.quality_base,
            "downstream_quality_retention": self.config.compression.downstream_quality_retention,
            "reconstructed_image_quality_proxy": self.config.compression.reconstructed_ssim_proxy,
            "compression_power_w": self.config.compression.compression_power_w,
            "delivered_effective_utility": delivered_effective_utility,
            "delivered_task_net_utility_sum": delivered_task_net_utility_sum,
            "net_system_utility": net_system_utility,
            "H_net_utility": net_system_utility,
            "evaluation_horizon_h": (
                len(timeline) * float(timeline[0].slot_seconds) / 3600.0 if timeline else 0.0
            ),
        }


def write_b1_outputs(
    output_dir: Path,
    result: dict[str, Any],
    config: B1EnhancedConfig,
    configs_snapshot: dict[str, Any],
    fairness_manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary_metrics.csv", [result["summary"]])
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output_dir / "per_task_delivery_trace.csv", result["per_task_delivery_trace"])
    _write_csv(output_dir / "queue_trace.csv", result["queue_trace"])
    _write_csv(output_dir / "soc_trace.csv", result["soc_trace"])
    _write_csv(output_dir / "contact_window_schedule.csv", result["contact_window_schedule"])
    _write_csv(output_dir / "compression_trace.csv", result["compression_trace"])
    _write_csv(output_dir / "lifecycle_counts.csv", result["lifecycle_counts"])
    _write_csv(output_dir / "raw_size_mapping_audit.csv", result["raw_size_mapping_audit"])
    _write_csv(output_dir / "compression_parameter_audit.csv", result["compression_parameter_audit"])
    _write_csv(output_dir / "ground_model_profile_audit.csv", result["ground_model_profile_audit"])
    (output_dir / "fairness_manifest.json").write_text(
        json.dumps(fairness_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(configs_snapshot, sort_keys=True), encoding="utf-8")
    (output_dir / "b1_enhanced_validation_report.md").write_text(_validation_report(result), encoding="utf-8")
    (output_dir / "paper_baseline_mapping.md").write_text(_paper_mapping(config), encoding="utf-8")
    (output_dir / "implementation_notes.md").write_text(_implementation_notes(config), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_fairness_manifest(
    timeline: list[TimelineStep],
    config: B1EnhancedConfig,
    seed: int,
    git_commit: str,
    num_orbits: int,
    task_manifest_path: Path | None = None,
    contact_trace_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "method_name": METHOD_NAME,
        "task_manifest_hash": _file_sha256(task_manifest_path) if task_manifest_path else task_manifest_hash(timeline),
        "contact_schedule_hash": _file_sha256(contact_trace_path) if contact_trace_path else contact_schedule_hash(timeline),
        "resource_trace_hash": resource_trace_hash(timeline),
        "random_seed": seed,
        "evaluation_horizon_h": len(timeline) * float(timeline[0].slot_seconds) / 3600.0 if timeline else 0.0,
        "num_tasks": sum(1 for step in timeline if step.task is not None),
        "num_orbits": num_orbits,
        "compression_ratio": config.compression.compression_ratio,
        "baseline_version": config.baseline_version,
        "git_commit": git_commit,
    }


def _validation_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    return "\n".join(
        [
            "# B1-Enhanced Validation Report",
            "",
            f"- method_name: `{summary['method_name']}`",
            f"- total_generated_tasks: {summary['total_generated_tasks']}",
            f"- ground_received_task_count: {summary['ground_received_task_count']}",
            f"- late_ground_result_count: {summary['late_ground_result_count']}",
            f"- hard_safety_violation_count: {summary['hard_safety_violation_count']}",
            f"- thermal_violation_count: {summary['thermal_violation_count']}",
            "",
            "Validation scope: architecture-equivalent implementation, not a line-by-line reproduction of the source papers.",
        ]
    ) + "\n"


def _paper_mapping(config: B1EnhancedConfig) -> str:
    return "\n".join(
        [
            "# Paper Baseline Mapping",
            "",
            "- Phi-Sat-2 CAE compression analogue: onboard fixed-rate encoder, CR=8 by default, 32 MB to 4 MB, SSIM proxy recorded separately from task quality.",
            "- EO contact-aware scheduling analogue: rolling-horizon value-density, deadline, age, and full-window-fit scheduling.",
            "- Ground processing analogue: compressed full image must be fully received before decode and inference.",
            "- Not implemented: multi-satellite/multi-station optical or microwave ALNS_TS metaheuristic.",
            "",
            f"- reconstructed_image_quality_proxy: {config.compression.reconstructed_ssim_proxy}",
            f"- downstream_quality_retention: {config.compression.downstream_quality_retention}",
            f"- compression_power_w: {config.compression.compression_power_w}",
            f"- compression_power_assumption_source: {config.compression.compression_power_assumption_source}",
        ]
    ) + "\n"


def _implementation_notes(config: B1EnhancedConfig) -> str:
    return "\n".join(
        [
            "# Implementation Notes",
            "",
            "- B1-Enhanced is independent of proposed mode control, Lyapunov pricing, safe bandit, staged evidence debt, and cheap-win logic.",
            "- Satellite SOC includes onboard compression energy and communication energy only; ground decode and inference energy are not charged to satellite SOC.",
            "- `blocked` and `dropped` are distinct. This implementation blocks infeasible compression or queue admission and does not relabel those cases as dropped.",
            "- `reconstructed_image_quality_proxy` is SSIM-like image quality; it is not detection accuracy.",
            f"- Ground inference model profile: `{config.ground.inference_model_name}`.",
        ]
    ) + "\n"
