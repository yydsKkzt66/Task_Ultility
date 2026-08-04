"""Lightweight one-shot LOCAL/GROUND earliest-finish-time baseline.

The policy only consumes the current task, committed queue state, current
satellite safety state, and the published contact trace.  It never consumes
future task arrivals or labels.  Ground processing is an explicit finite FIFO
queue and is intentionally a documented proxy for smoke validation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from src.energy.battery_update import update_battery_soc
from src.energy.thermal_update import update_thermal_pressure

LOCAL = "LOCAL"
GROUND = "GROUND"
DEFER = "DEFER"


@dataclass(frozen=True)
class ContactSegment:
    start_s: float
    end_s: float
    rate_mb_s: float


@dataclass(frozen=True)
class PathEstimate:
    path: str
    feasible: bool
    finish_time_s: float
    latency_s: float
    tx_bytes: float
    sat_energy_j: float
    infeasible_reason: str = ""
    onboard_start_s: float = 0.0
    onboard_finish_s: float = 0.0
    product_ready_s: float = 0.0
    transmission_start_s: float = 0.0
    ground_received_s: float = 0.0
    ground_compute_start_s: float = 0.0
    ground_compute_end_s: float = 0.0
    compute_energy_j: float = 0.0
    comm_energy_j: float = 0.0
    predicted_soc: float = 0.0
    predicted_thermal: float = 0.0


@dataclass(frozen=True)
class OffloadDecision:
    task_id: str
    selected_path: str
    selection_reason: str
    committed_once: bool
    local: PathEstimate
    ground: PathEstimate


@dataclass(frozen=True)
class GroundJob:
    task_id: str
    queue_entry_time: float
    compute_start_time: float
    compute_end_time: float
    waiting_time: float
    server_id: int
    queue_length_at_entry: int


class FiniteGroundComputeQueue:
    """Arrival-ordered FIFO queue with a fixed number of servers."""

    def __init__(self, servers: int = 1, max_queue_items: int = 32) -> None:
        if servers < 1 or max_queue_items < 0:
            raise ValueError("ground queue requires positive servers and nonnegative capacity")
        self.servers = servers
        self.max_queue_items = max_queue_items
        self.available_s = [0.0] * servers
        self.jobs: list[GroundJob] = []

    def preview(self, entry_s: float, duration_s: float) -> GroundJob | None:
        waiting = sum(job.compute_start_time > entry_s and job.compute_end_time > entry_s for job in self.jobs)
        if waiting >= self.max_queue_items:
            return None
        server = min(range(self.servers), key=lambda i: (self.available_s[i], i))
        start = max(float(entry_s), self.available_s[server])
        return GroundJob("", entry_s, start, start + duration_s, start - entry_s, server, waiting)

    def commit(self, task_id: str, entry_s: float, duration_s: float) -> GroundJob:
        candidate = self.preview(entry_s, duration_s)
        if candidate is None:
            raise OverflowError("ground_compute_queue_full")
        job = GroundJob(task_id, candidate.queue_entry_time, candidate.compute_start_time,
                        candidate.compute_end_time, candidate.waiting_time,
                        candidate.server_id, candidate.queue_length_at_entry)
        self.available_s[job.server_id] = job.compute_end_time
        self.jobs.append(job)
        return job


class OneShotEFTPolicy:
    def __init__(self, tie_tolerance_s: float = 1.0e-6, final_tie_order: str = LOCAL) -> None:
        self.tie_tolerance_s = max(0.0, tie_tolerance_s)
        self.final_tie_order = final_tie_order
        self._decisions: dict[str, OffloadDecision] = {}

    @property
    def decisions(self) -> dict[str, OffloadDecision]:
        return dict(self._decisions)

    def decide(self, task_id: str, local: PathEstimate, ground: PathEstimate) -> OffloadDecision:
        if task_id in self._decisions:
            return self._decisions[task_id]
        if local.feasible and not ground.feasible:
            selected, reason = LOCAL, "only_local_feasible"
        elif ground.feasible and not local.feasible:
            selected, reason = GROUND, "only_ground_feasible"
        elif not local.feasible and not ground.feasible:
            selected, reason = DEFER, "both_infeasible"
        else:
            delta = local.finish_time_s - ground.finish_time_s
            if abs(delta) > self.tie_tolerance_s:
                selected = LOCAL if delta < 0 else GROUND
                reason = "earliest_finish_time"
            elif abs(local.tx_bytes - ground.tx_bytes) > 1.0e-6:
                selected = LOCAL if local.tx_bytes < ground.tx_bytes else GROUND
                reason = "tie_lower_transmitted_bytes"
            elif abs(local.sat_energy_j - ground.sat_energy_j) > 1.0e-9:
                selected = LOCAL if local.sat_energy_j < ground.sat_energy_j else GROUND
                reason = "tie_lower_satellite_energy"
            else:
                selected, reason = self.final_tie_order, "deterministic_final_tie"
        decision = OffloadDecision(task_id, selected, reason, True, local, ground)
        self._decisions[task_id] = decision
        return decision


def forecast_transfer(ready_s: float, size_mb: float, contacts: Iterable[ContactSegment],
                      queue_cursor_s: float = 0.0) -> tuple[float, float, float]:
    """Return transmission start, finish and active TX seconds for FIFO service."""
    remaining = max(0.0, float(size_mb))
    if remaining <= 1.0e-12:
        start = max(float(ready_s), float(queue_cursor_s))
        return start, start, 0.0
    t = max(float(ready_s), float(queue_cursor_s)); first = math.inf; active = 0.0
    for segment in contacts:
        if segment.end_s <= t or segment.rate_mb_s <= 0.0:
            continue
        point = max(t, segment.start_s)
        if point >= segment.end_s:
            continue
        first = min(first, point)
        capacity = (segment.end_s - point) * segment.rate_mb_s
        if capacity + 1.0e-12 >= remaining:
            duration = remaining / segment.rate_mb_s
            return first, point + duration, active + duration
        remaining -= capacity; active += segment.end_s - point; t = segment.end_s
    raise ValueError("insufficient_contact_capacity_before_horizon")


def project_satellite_state(*, soc: float, thermal: float, compute_energy_j: float,
                            comm_energy_j: float, simulator_cfg: dict[str, Any]) -> tuple[float, float]:
    capacity_j = float(simulator_cfg.get("battery_capacity_j", simulator_cfg.get("battery_capacity_wh", 1.0) * 3600.0))
    next_soc = update_battery_soc(
        battery_soc=soc, solar_energy_j=0.0, base_power_w=0.0, slot_seconds=0.0,
        compute_energy_j=compute_energy_j, comm_energy_j=comm_energy_j,
        battery_capacity_j=capacity_j,
        charge_efficiency=float(simulator_cfg.get("battery_charge_efficiency", 1.0)),
        discharge_efficiency=float(simulator_cfg.get("battery_discharge_efficiency", 1.0)),
    )
    next_thermal = update_thermal_pressure(
        thermal_pressure=thermal, compute_energy_j=compute_energy_j, comm_energy_j=comm_energy_j,
        cooling=0.0, slot_seconds=0.0,
        reference_slot_seconds=float(simulator_cfg.get("thermal_reference_slot_seconds", 60.0)),
        temperature_reference_c=0.0, is_sunlit=False, passive_cooling=0.0,
        sunlight_gain=0.0, eclipse_cooling_bonus=0.0, reference_min_c=-10.0, reference_max_c=30.0,
    )
    return next_soc, next_thermal


class OffloadEFTSimulator:
    """Small deterministic event simulator used only by the OFFLOAD-EFT smoke."""

    def __init__(self, *, contacts: list[ContactSegment], simulator_cfg: dict[str, Any],
                 local_compute_latency_s: float, local_compute_energy_j: float,
                 local_product_size_mb: float, local_generation_latency_s: float,
                 local_generation_energy_j: float, ground_payload_size_mb: float,
                 ground_preprocess_latency_s: float, ground_preprocess_energy_j: float,
                 ground_compute_latency_s: float, ground_servers: int = 1,
                 ground_queue_capacity: int = 32, tie_tolerance_s: float = 1.0e-6) -> None:
        self.contacts = contacts; self.simulator_cfg = dict(simulator_cfg)
        self.local_compute_latency_s = local_compute_latency_s
        self.local_compute_energy_j = local_compute_energy_j
        self.local_product_size_mb = local_product_size_mb
        self.local_generation_latency_s = local_generation_latency_s
        self.local_generation_energy_j = local_generation_energy_j
        self.ground_payload_size_mb = ground_payload_size_mb
        self.ground_preprocess_latency_s = ground_preprocess_latency_s
        self.ground_preprocess_energy_j = ground_preprocess_energy_j
        self.ground_compute_latency_s = ground_compute_latency_s
        self.ground_queue = FiniteGroundComputeQueue(ground_servers, ground_queue_capacity)
        self.policy = OneShotEFTPolicy(tie_tolerance_s)
        self.compute_cursor_s = 0.0; self.downlink_cursor_s = 0.0
        self.soc = float(simulator_cfg.get("battery_init", 0.9))
        self.thermal = float(simulator_cfg.get("thermal_init", 0.52))
        self.transfers: list[dict[str, float]] = []; self.decision_trace: list[dict[str, Any]] = []

    def advance_environment(self, *, slot_seconds: float, solar_energy_j: float,
                            base_power_w: float, temperature_reference_c: float,
                            is_sunlit: bool) -> None:
        """Advance shared passive battery/thermal dynamics to the next observable slot."""
        capacity_j = float(self.simulator_cfg.get(
            "battery_capacity_j", self.simulator_cfg.get("battery_capacity_wh", 1.0) * 3600.0
        ))
        self.soc = update_battery_soc(
            battery_soc=self.soc, solar_energy_j=solar_energy_j,
            base_power_w=base_power_w, slot_seconds=slot_seconds,
            compute_energy_j=0.0, comm_energy_j=0.0, battery_capacity_j=capacity_j,
            charge_efficiency=float(self.simulator_cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(self.simulator_cfg.get("battery_discharge_efficiency", 1.0)),
        )
        self.thermal = update_thermal_pressure(
            thermal_pressure=self.thermal, compute_energy_j=0.0, comm_energy_j=0.0,
            cooling=float(self.simulator_cfg.get("cooling_per_slot", 0.0)), slot_seconds=slot_seconds,
            reference_slot_seconds=float(self.simulator_cfg.get("thermal_reference_slot_seconds", 60.0)),
            temperature_reference_c=temperature_reference_c, is_sunlit=is_sunlit,
            passive_cooling=float(self.simulator_cfg.get("thermal_passive_cooling", 0.12)),
            sunlight_gain=float(self.simulator_cfg.get("thermal_sunlight_gain", 0.015)),
            eclipse_cooling_bonus=float(self.simulator_cfg.get("thermal_eclipse_cooling_bonus", 0.03)),
            reference_min_c=float(self.simulator_cfg.get("thermal_reference_min_c", -10.0)),
            reference_max_c=float(self.simulator_cfg.get("thermal_reference_max_c", 30.0)),
        )

    def _estimate(self, *, path: str, arrival_s: float, deadline_s: float,
                  product_feasible: bool) -> PathEstimate:
        reasons: list[str] = []
        latency = self.local_compute_latency_s + self.local_generation_latency_s if path == LOCAL else self.ground_preprocess_latency_s
        compute_j = self.local_compute_energy_j + self.local_generation_energy_j if path == LOCAL else self.ground_preprocess_energy_j
        size_mb = self.local_product_size_mb if path == LOCAL else self.ground_payload_size_mb
        onboard_start = max(arrival_s, self.compute_cursor_s); onboard_finish = onboard_start + latency
        backlog_mb = sum(row["size_mb"] for row in self.transfers if row["finish_s"] > arrival_s)
        if backlog_mb + size_mb > float(self.simulator_cfg.get("queue_max_mb", math.inf)) + 1.0e-9:
            reasons.append("satellite_queue_capacity")
        try:
            tx_start, ground_received, tx_s = forecast_transfer(onboard_finish, size_mb, self.contacts, self.downlink_cursor_s)
        except ValueError:
            tx_start = ground_received = math.inf; tx_s = math.inf; reasons.append("contact_capacity")
        comm_j = float(self.simulator_cfg.get("tx_power_w", 35.0)) * tx_s if math.isfinite(tx_s) else math.inf
        ground_start = ground_end = ground_received
        if path == GROUND and math.isfinite(ground_received):
            preview = self.ground_queue.preview(ground_received, self.ground_compute_latency_s)
            if preview is None:
                reasons.append("ground_queue_capacity")
            else:
                ground_start, ground_end = preview.compute_start_time, preview.compute_end_time
        finish = ground_end
        if not product_feasible: reasons.append("product_or_requirement_infeasible")
        if finish > deadline_s + 1.0e-9: reasons.append("deadline")
        if math.isfinite(comm_j):
            next_soc, next_thermal = project_satellite_state(soc=self.soc, thermal=self.thermal,
                compute_energy_j=compute_j, comm_energy_j=comm_j, simulator_cfg=self.simulator_cfg)
            if next_soc < float(self.simulator_cfg.get("battery_min", 0.0)) - 1.0e-9: reasons.append("soc_floor")
            if next_thermal > float(self.simulator_cfg.get("thermal_max", 1.0)) + 1.0e-9: reasons.append("thermal_hard_limit")
        else:
            next_soc, next_thermal = -math.inf, math.inf
        return PathEstimate(path, not reasons, finish, finish-arrival_s, size_mb*1_000_000.0,
            compute_j+comm_j, ";".join(dict.fromkeys(reasons)), onboard_start, onboard_finish,
            onboard_finish, tx_start, ground_received, ground_start, ground_end, compute_j, comm_j,
            next_soc, next_thermal)

    def process_task(self, *, task_id: str, arrival_s: float, deadline_time_s: float,
                     local_product_feasible: bool, ground_product_feasible: bool) -> OffloadDecision:
        if task_id in self.policy.decisions:
            return self.policy.decisions[task_id]
        local = self._estimate(path=LOCAL, arrival_s=arrival_s, deadline_s=deadline_time_s,
                               product_feasible=local_product_feasible)
        ground = self._estimate(path=GROUND, arrival_s=arrival_s, deadline_s=deadline_time_s,
                                product_feasible=ground_product_feasible)
        decision = self.policy.decide(task_id, local, ground)
        chosen = local if decision.selected_path == LOCAL else ground
        if decision.selected_path != DEFER:
            self.compute_cursor_s = chosen.onboard_finish_s; self.downlink_cursor_s = chosen.ground_received_s
            self.transfers.append({"finish_s": chosen.ground_received_s,
                                   "size_mb": chosen.tx_bytes/1_000_000.0})
            if decision.selected_path == GROUND:
                job = self.ground_queue.commit(task_id, chosen.ground_received_s, self.ground_compute_latency_s)
                if abs(job.compute_end_time-chosen.ground_compute_end_s) > 1.0e-9:
                    raise AssertionError("ground queue prediction/commit mismatch")
            self.soc, self.thermal = chosen.predicted_soc, chosen.predicted_thermal
        finish = chosen.finish_time_s if decision.selected_path != DEFER else math.nan
        row = {
            "task_id": task_id, "arrival_time": arrival_s,
            "predicted_local_latency": local.latency_s, "predicted_ground_latency": ground.latency_s,
            "local_feasible": local.feasible, "ground_feasible": ground.feasible,
            "local_infeasible_reason": local.infeasible_reason, "ground_infeasible_reason": ground.infeasible_reason,
            "selected_path": decision.selected_path, "tie_break_reason": decision.selection_reason,
            "committed_once": True, "final_status": "CLOSED" if decision.selected_path != DEFER else "DEFERRED",
            "actual_closure_latency": finish-arrival_s if math.isfinite(finish) else "",
            "prediction_error": 0.0 if math.isfinite(finish) else "",
            "product_generation_complete": chosen.product_ready_s if decision.selected_path != DEFER else "",
            "ground_arrival": chosen.ground_received_s if decision.selected_path != DEFER else "",
            "ground_compute_start": chosen.ground_compute_start_s if decision.selected_path == GROUND else "",
            "ground_compute_end": chosen.ground_compute_end_s if decision.selected_path == GROUND else "",
            "final_closure": finish if math.isfinite(finish) else "",
            "actual_transmitted_bytes": chosen.tx_bytes if decision.selected_path != DEFER else 0.0,
            "actual_sat_energy": chosen.sat_energy_j if decision.selected_path != DEFER else 0.0,
            "minimum_soc": self.soc, "maximum_thermal_state": self.thermal,
        }
        self.decision_trace.append(row)
        return decision

    def ground_queue_rows(self) -> list[dict[str, Any]]:
        return [asdict(job) for job in self.ground_queue.jobs]
