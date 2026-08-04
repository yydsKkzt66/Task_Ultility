"""Thermally aligned one-shot OFFLOAD-EFT experiment.

The experiment differs from OFFLOAD-EFT only in feasibility prediction and
execution: both use the same slot transition, cooling, battery update, contact
trace, FIFO resources, and exact thermal-headroom calculation.  A dry run is a
deep clone and never consumes future task arrivals or labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from src.baselines.offload_eft import DEFER, GROUND, LOCAL, FiniteGroundComputeQueue
from src.common.types import TimelineStep
from src.energy.battery_update import update_battery_soc
from src.energy.comm_energy import compute_comm_energy_j
from src.energy.thermal_headroom import maximum_safe_downlink_capacity
from src.energy.thermal_update import update_thermal_pressure


@dataclass
class AlignedJob:
    task_id: str
    path: str
    arrival_s: float
    deadline_s: float
    product_feasible: bool
    compute_duration_s: float
    compute_energy_j: float
    payload_mb: float
    ground_compute_duration_s: float
    state: str = "PENDING_COMPUTE"
    compute_start_s: float | None = None
    compute_finish_s: float | None = None
    downlink_queue_enter_s: float | None = None
    tx_start_s: float | None = None
    tx_finish_s: float | None = None
    remaining_mb: float = 0.0
    transmitted_mb: float = 0.0
    comm_energy_j: float = 0.0
    ground_queue_enter_s: float | None = None
    ground_compute_start_s: float | None = None
    ground_compute_finish_s: float | None = None
    closure_s: float | None = None
    failure_reason: str = ""
    compute_blocked_slots: int = 0
    downlink_blocked_slots: int = 0

    def __post_init__(self) -> None:
        if self.remaining_mb <= 0.0:
            self.remaining_mb = self.payload_mb


@dataclass(frozen=True)
class AlignedEstimate:
    path: str
    feasible: bool
    predicted_finish_s: float
    predicted_latency_s: float
    predicted_max_thermal: float
    predicted_min_soc: float
    predicted_tx_bytes: float
    predicted_sat_energy_j: float
    predicted_queue_wait_s: float
    failure_reason: str
    state_isolated: bool
    future_tasks_read: int
    trace: tuple[dict[str, Any], ...] = field(repr=False)


@dataclass(frozen=True)
class AlignedDecision:
    task_id: str
    selected_path: str
    selection_reason: str
    local: AlignedEstimate
    ground: AlignedEstimate


class ThermalAlignedSimulator:
    """Slot executor shared by cloned prediction and independent real execution."""

    def __init__(
        self,
        *,
        timeline: list[TimelineStep],
        simulator_cfg: dict[str, Any],
        local_compute_duration_s: float,
        local_compute_energy_j: float,
        local_payload_mb: float,
        ground_preprocess_duration_s: float,
        ground_preprocess_energy_j: float,
        ground_payload_mb: float,
        ground_compute_duration_s: float,
        ground_servers: int = 1,
        ground_queue_capacity: int = 32,
        tie_tolerance_s: float = 1.0e-6,
    ) -> None:
        self.timeline = timeline
        self.cfg = dict(simulator_cfg)
        if "battery_capacity_j" not in self.cfg:
            self.cfg["battery_capacity_j"] = float(self.cfg.get("battery_capacity_wh", 1.0))*3600.0
        self.local_compute_duration_s = float(local_compute_duration_s)
        self.local_compute_energy_j = float(local_compute_energy_j)
        self.local_payload_mb = float(local_payload_mb)
        self.ground_preprocess_duration_s = float(ground_preprocess_duration_s)
        self.ground_preprocess_energy_j = float(ground_preprocess_energy_j)
        self.ground_payload_mb = float(ground_payload_mb)
        self.ground_compute_duration_s = float(ground_compute_duration_s)
        self.ground_queue = FiniteGroundComputeQueue(ground_servers, ground_queue_capacity)
        self.tie_tolerance_s = max(0.0, float(tie_tolerance_s))
        self.current_index = 0
        self.soc = float(self.cfg.get("battery_init", 0.9))
        self.thermal = float(self.cfg.get("thermal_init", 0.52))
        self.jobs: dict[str, AlignedJob] = {}
        self.compute_queue: list[str] = []
        self.downlink_queue: list[str] = []
        self.decisions: dict[str, AlignedDecision] = {}
        self.actual_trace: list[dict[str, Any]] = []
        self.safety_violations = 0

    def _state_payload(self) -> dict[str, Any]:
        return {
            "current_index": self.current_index,
            "soc": self.soc,
            "thermal": self.thermal,
            "jobs": {k: asdict(v) for k, v in sorted(self.jobs.items())},
            "compute_queue": self.compute_queue,
            "downlink_queue": self.downlink_queue,
            "ground_available": self.ground_queue.available_s,
            "ground_jobs": [asdict(x) for x in self.ground_queue.jobs],
            "decisions": sorted(self.decisions),
        }

    def state_fingerprint(self) -> str:
        raw = json.dumps(self._state_payload(), sort_keys=True, separators=(",", ":"), allow_nan=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _new_job(self, task_id: str, path: str, arrival_s: float, deadline_s: float,
                 product_feasible: bool) -> AlignedJob:
        if path == LOCAL:
            return AlignedJob(task_id, path, arrival_s, deadline_s, product_feasible,
                              self.local_compute_duration_s, self.local_compute_energy_j,
                              self.local_payload_mb, 0.0)
        return AlignedJob(task_id, path, arrival_s, deadline_s, product_feasible,
                          self.ground_preprocess_duration_s, self.ground_preprocess_energy_j,
                          self.ground_payload_mb, self.ground_compute_duration_s)

    def _admit(self, job: AlignedJob) -> None:
        if job.task_id in self.jobs:
            raise ValueError(f"duplicate admitted task: {job.task_id}")
        committed_mb = sum(x.remaining_mb for x in self.jobs.values() if x.state not in {"CLOSED", "FAILED"})
        if committed_mb + job.payload_mb > float(self.cfg.get("queue_max_mb", math.inf)) + 1e-9:
            job.state = "FAILED"; job.failure_reason = "satellite_queue_capacity"
        self.jobs[job.task_id] = job
        if job.state != "FAILED":
            self.compute_queue.append(job.task_id)

    def _transition(self, step: TimelineStep, compute_j: float, comm_j: float) -> tuple[float, float]:
        next_soc = update_battery_soc(
            battery_soc=self.soc, solar_energy_j=step.resource_trace.solar_energy_j,
            base_power_w=step.resource_trace.base_power_w, slot_seconds=step.slot_seconds,
            compute_energy_j=compute_j, comm_energy_j=comm_j,
            battery_capacity_j=float(self.cfg["battery_capacity_j"]),
            charge_efficiency=float(self.cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(self.cfg.get("battery_discharge_efficiency", 1.0)),
        )
        next_thermal = update_thermal_pressure(
            thermal_pressure=self.thermal, compute_energy_j=compute_j, comm_energy_j=comm_j,
            cooling=float(self.cfg.get("cooling_per_slot", 0.0)), slot_seconds=step.slot_seconds,
            reference_slot_seconds=float(self.cfg.get("thermal_reference_slot_seconds", step.slot_seconds)),
            temperature_reference_c=step.resource_trace.temperature_reference_c,
            is_sunlit=step.resource_trace.is_sunlit,
            passive_cooling=float(self.cfg.get("thermal_passive_cooling", 0.12)),
            sunlight_gain=float(self.cfg.get("thermal_sunlight_gain", 0.015)),
            eclipse_cooling_bonus=float(self.cfg.get("thermal_eclipse_cooling_bonus", 0.03)),
            reference_min_c=float(self.cfg.get("thermal_reference_min_c", -10.0)),
            reference_max_c=float(self.cfg.get("thermal_reference_max_c", 30.0)),
        )
        return next_soc, next_thermal

    def _step(self, *, record: bool) -> None:
        if self.current_index >= len(self.timeline):
            return
        step = self.timeline[self.current_index]
        start = step.slot_index*step.slot_seconds; end = start+step.slot_seconds
        before_soc, before_thermal = self.soc, self.thermal
        compute_j = 0.0; comm_j = 0.0; compute_task = ""; tx_task = ""; sent_mb = 0.0
        safe_capacity_mb = 0.0; nominal_capacity_mb = 0.0; zero_reason = ""

        # One shared onboard compute/compression job per slot. Unsafe work waits;
        # it is never executed by violating the hard transition.
        pending = [tid for tid in self.compute_queue
                   if self.jobs[tid].state == "PENDING_COMPUTE" and self.jobs[tid].arrival_s <= start+1e-9]
        if pending:
            tid = pending[0]; job = self.jobs[tid]
            probe_soc, probe_thermal = self._transition(step, job.compute_energy_j, 0.0)
            if probe_soc >= float(self.cfg.get("battery_min", 0.0))-1e-9 and probe_thermal <= float(self.cfg.get("thermal_max", 1.0))+1e-9:
                compute_task = tid; compute_j = job.compute_energy_j
                job.compute_start_s = start; job.compute_finish_s = start+job.compute_duration_s
                job.downlink_queue_enter_s = job.compute_finish_s; job.state = "WAIT_DOWNLINK"
                self.compute_queue.remove(tid); self.downlink_queue.append(tid)
            else:
                job.compute_blocked_slots += 1

        # FIFO downlink. Exact thermal headroom uses the same full slot
        # transition that will be used below for the real state update.
        eligible = [tid for tid in self.downlink_queue
                    if self.jobs[tid].state == "WAIT_DOWNLINK"
                    and (self.jobs[tid].downlink_queue_enter_s or math.inf) <= end+1e-9]
        if step.link_trace.in_contact and eligible:
            first_job = self.jobs[eligible[0]]
            tx_available_start = max(start, float(first_job.downlink_queue_enter_s or start))
            available_s = max(0.0, end-tx_available_start)
            rate = step.link_trace.throughput_mbps*float(self.cfg.get("downlink_efficiency", 1.0))/8.0
            nominal_capacity_mb = min(sum(self.jobs[tid].remaining_mb for tid in eligible), rate*available_s)
            cap = maximum_safe_downlink_capacity(
                current_thermal=self.thermal, nominal_capacity_mb=nominal_capacity_mb,
                rate_mb_s=rate, link_quality=step.link_trace.link_quality,
                compute_energy_j=compute_j, compute_heat_gain_scale=1.0,
                slot_seconds=step.slot_seconds,
                temperature_reference_c=step.resource_trace.temperature_reference_c,
                is_sunlit=step.resource_trace.is_sunlit, simulator_cfg=self.cfg,
            )
            safe_capacity_mb = min(nominal_capacity_mb, cap.safe_capacity_mb)
            zero_reason = cap.zero_capacity_reason
            candidate_comm_j = compute_comm_energy_j(
                comm_power_w=float(self.cfg.get("comm_power_w", self.cfg.get("tx_power_w", 0.0))),
                tx_time_s=safe_capacity_mb/max(rate, 1e-9), link_quality=step.link_trace.link_quality,
                alpha_gamma=float(self.cfg.get("comm_alpha_gamma", 0.0)),
            )
            probe_soc, _ = self._transition(step, compute_j, candidate_comm_j)
            if probe_soc < float(self.cfg.get("battery_min", 0.0))-1e-9:
                safe_capacity_mb = 0.0; candidate_comm_j = 0.0; zero_reason = "SOC_FLOOR"
            if safe_capacity_mb > 1e-12:
                # Consume the one slot-level safe budget continuously in FIFO
                # order. Completing a small head item must not discard the
                # remaining safe bytes in this same slot.
                budget=safe_capacity_mb; cursor=tx_available_start; tx_ids=[]
                for tid in list(eligible):
                    if budget<=1e-12: break
                    job=self.jobs[tid]; amount=min(job.remaining_mb,budget)
                    if amount<=1e-12: continue
                    tx_ids.append(tid)
                    if job.tx_start_s is None: job.tx_start_s=cursor
                    duration=amount/max(rate,1e-9)
                    energy=compute_comm_energy_j(
                        comm_power_w=float(self.cfg.get("comm_power_w",self.cfg.get("tx_power_w",0.0))),
                        tx_time_s=duration,link_quality=step.link_trace.link_quality,
                        alpha_gamma=float(self.cfg.get("comm_alpha_gamma",0.0)))
                    job.remaining_mb=max(0.0,job.remaining_mb-amount)
                    job.transmitted_mb+=amount;job.comm_energy_j+=energy
                    sent_mb+=amount;comm_j+=energy;budget-=amount;cursor+=duration
                    if job.remaining_mb<=1e-9:
                        job.tx_finish_s=cursor;job.ground_queue_enter_s=cursor
                        if job.path==GROUND:
                            ground=self.ground_queue.commit(job.task_id,cursor,job.ground_compute_duration_s)
                            job.ground_compute_start_s=ground.compute_start_time
                            job.ground_compute_finish_s=ground.compute_end_time
                            job.closure_s=ground.compute_end_time
                        else: job.closure_s=cursor
                        job.state="CLOSED" if job.closure_s<=job.deadline_s+1e-9 else "FAILED"
                        if job.state=="FAILED":job.failure_reason="deadline"
                        self.downlink_queue.remove(tid)
                tx_task=";".join(tx_ids)
            else:
                first_job.downlink_blocked_slots += 1

        self.soc, self.thermal = self._transition(step, compute_j, comm_j)
        hard = float(self.cfg.get("thermal_max", 1.0)); floor = float(self.cfg.get("battery_min", 0.0))
        violation = int(self.thermal > hard+1e-9 or self.soc < floor-1e-9)
        self.safety_violations += violation
        row = {
            "slot": step.slot_index, "time_s": start, "soc_before": before_soc,
            "soc_after": self.soc, "thermal_before": before_thermal, "thermal_after": self.thermal,
            "compute_task_id": compute_task, "compute_energy_j": compute_j,
            "tx_task_id": tx_task, "comm_energy_j": comm_j, "sent_mb": sent_mb,
            "nominal_capacity_mb": nominal_capacity_mb, "safe_capacity_mb": safe_capacity_mb,
            "thermal_headroom_zero_reason": zero_reason, "in_contact": int(step.link_trace.in_contact),
            "temperature_reference_c": step.resource_trace.temperature_reference_c,
            "is_sunlit": int(step.resource_trace.is_sunlit), "hard_limit": hard,
            "soc_floor": floor, "safety_violation": violation,
        }
        if record: self.actual_trace.append(row)
        else: self.actual_trace.append(row)  # clone-local prediction trace
        self.current_index += 1

    def advance_one_slot(self) -> None:
        self._step(record=True)

    def dry_run(self, *, task_id: str, path: str, arrival_s: float, deadline_s: float,
                product_feasible: bool) -> AlignedEstimate:
        before = self.state_fingerprint()
        clone: ThermalAlignedSimulator = copy.deepcopy(self)
        trace_start = len(clone.actual_trace)
        job = clone._new_job(task_id, path, arrival_s, deadline_s, product_feasible)
        clone._admit(job)
        while clone.current_index < len(clone.timeline):
            current_s = clone.timeline[clone.current_index].slot_index*clone.timeline[clone.current_index].slot_seconds
            if current_s > deadline_s+1e-9 or job.state in {"CLOSED", "FAILED"}:
                break
            clone._step(record=False)
        prediction_trace = clone.actual_trace[trace_start:]
        max_thermal = max([self.thermal]+[float(x["thermal_after"]) for x in prediction_trace])
        min_soc = min([self.soc]+[float(x["soc_after"]) for x in prediction_trace])
        reasons: list[str] = []
        if not product_feasible: reasons.append("product_or_requirement_infeasible")
        if job.failure_reason: reasons.append(job.failure_reason)
        if job.closure_s is None or job.closure_s > deadline_s+1e-9: reasons.append("deadline")
        if min_soc < float(self.cfg.get("battery_min", 0.0))-1e-9: reasons.append("soc_floor")
        if max_thermal > float(self.cfg.get("thermal_max", 1.0))+1e-9: reasons.append("thermal_hard_limit")
        if job.compute_blocked_slots and job.compute_start_s is None: reasons.append("compute_safety_blocked")
        if job.downlink_blocked_slots and job.tx_start_s is None: reasons.append("downlink_safety_blocked")
        reasons = list(dict.fromkeys(reasons))
        finish = float(job.closure_s) if job.closure_s is not None else math.inf
        queue_wait = 0.0
        if job.compute_start_s is not None: queue_wait += max(0.0, job.compute_start_s-arrival_s)
        if job.downlink_queue_enter_s is not None and job.tx_start_s is not None:
            queue_wait += max(0.0, job.tx_start_s-job.downlink_queue_enter_s)
        estimate = AlignedEstimate(
            path=path, feasible=not reasons, predicted_finish_s=finish,
            predicted_latency_s=finish-arrival_s if math.isfinite(finish) else math.inf,
            predicted_max_thermal=max_thermal, predicted_min_soc=min_soc,
            predicted_tx_bytes=job.payload_mb*1_000_000.0,
            predicted_sat_energy_j=job.compute_energy_j+job.comm_energy_j,
            predicted_queue_wait_s=queue_wait, failure_reason=";".join(reasons),
            state_isolated=(before == self.state_fingerprint()), future_tasks_read=0,
            trace=tuple({**row, "prediction_task_id": task_id, "prediction_path": path} for row in prediction_trace),
        )
        if not estimate.state_isolated:
            raise AssertionError("dry-run mutated real simulator state")
        return estimate

    def decide_task(self, *, task_id: str, arrival_s: float, deadline_s: float,
                    local_product_feasible: bool, ground_product_feasible: bool) -> AlignedDecision:
        if task_id in self.decisions:
            return self.decisions[task_id]
        local = self.dry_run(task_id=task_id, path=LOCAL, arrival_s=arrival_s,
                             deadline_s=deadline_s, product_feasible=local_product_feasible)
        ground = self.dry_run(task_id=task_id, path=GROUND, arrival_s=arrival_s,
                              deadline_s=deadline_s, product_feasible=ground_product_feasible)
        if local.feasible and not ground.feasible: selected, reason = LOCAL, "only_local_feasible"
        elif ground.feasible and not local.feasible: selected, reason = GROUND, "only_ground_feasible"
        elif not local.feasible and not ground.feasible: selected, reason = DEFER, "both_infeasible"
        else:
            delta = local.predicted_finish_s-ground.predicted_finish_s
            if abs(delta) > self.tie_tolerance_s:
                selected, reason = (LOCAL if delta < 0 else GROUND), "earliest_finish_time"
            elif abs(local.predicted_tx_bytes-ground.predicted_tx_bytes) > 1e-6:
                selected, reason = (LOCAL if local.predicted_tx_bytes < ground.predicted_tx_bytes else GROUND), "tie_lower_transmitted_bytes"
            elif abs(local.predicted_sat_energy_j-ground.predicted_sat_energy_j) > 1e-9:
                selected, reason = (LOCAL if local.predicted_sat_energy_j < ground.predicted_sat_energy_j else GROUND), "tie_lower_satellite_energy"
            else: selected, reason = LOCAL, "deterministic_final_tie"
        decision = AlignedDecision(task_id, selected, reason, local, ground)
        self.decisions[task_id] = decision
        if selected != DEFER:
            chosen_product = local_product_feasible if selected == LOCAL else ground_product_feasible
            self._admit(self._new_job(task_id, selected, arrival_s, deadline_s, chosen_product))
        return decision

    def actual_job(self, task_id: str) -> AlignedJob | None:
        return self.jobs.get(task_id)
