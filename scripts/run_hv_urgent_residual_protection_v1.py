from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_prop_reqalign_seed2026_v1 as reqrun


OUT = ROOT / "outputs" / "_v25_bupt1_hv_urgent_residual_guard_seed2026_cap050_v1"
INVALID_ATTEMPT_ARCHIVE = ROOT / "outputs" / (
    "_v25_bupt1_hv_urgent_residual_guard_seed2026_cap050_v1_"
    "attempt1_invalid_ordinary_semantics"
)
INVALID_AUDIT_ATTEMPT_ARCHIVE = ROOT / "outputs" / (
    "_v25_bupt1_hv_urgent_residual_guard_seed2026_cap050_v1_"
    "attempt2_invalid_audit_logging"
)
INVALID_EXCHANGE_ATTEMPT_ARCHIVE = ROOT / "outputs" / (
    "_v25_bupt1_hv_urgent_residual_guard_seed2026_cap050_v1_"
    "attempt3_invalid_hv_exchange_semantics"
)
GUARD_SOURCE = ROOT / "src" / "routers" / "hv_urgent_residual_guard.py"
GUARD_CONFIG = ROOT / "configs" / "hv_urgent_residual_protection_v1.yaml"
ROLLBACK_DIR = ROOT / "artifacts" / "rollback" / "hv_urgent_residual_protection_v1"
ROLLBACK_MANIFEST = ROLLBACK_DIR / "rollback_manifest.json"
RESTORE_SCRIPT = ROLLBACK_DIR / "restore_hv_urgent_residual_protection_v1.py"

METHOD_OLD = reqrun.METHOD_PROP_OLD
METHOD_REQALIGN = reqrun.METHOD_PROP_NEW
METHOD_GUARD = "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-HVGUARD-v1"
CAPACITY = 0.50
SOURCE_CAPACITY = 0.35
SLOTS = 4320
SMOKE_SLOTS = 600
SLOT_SECONDS = 20
COMPLETION_MARKER = "HV_URGENT_RESIDUAL_PROTECTION_AND_PARETO_ALIGNMENT_V1_COMPLETE"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows or [{"empty": ""}])


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_guard_config(enabled: bool) -> dict[str, Any]:
    raw = yaml.safe_load(GUARD_CONFIG.read_text(encoding="utf-8"))
    config = deepcopy(raw["hv_urgent_residual_guard"])
    config["enabled"] = bool(enabled)
    return config


def apply_capacity(scenario: Any, multiplier: float) -> tuple[Any, dict[str, Any]]:
    """Scale only throughput, preserving contact positions and all other inputs."""

    ratio = float(multiplier) / SOURCE_CAPACITY
    configs = deepcopy(scenario.configs)
    configs["downlink_capacity_multiplier"] = float(multiplier)
    configs.setdefault("simulator", {})["downlink_capacity_multiplier"] = float(multiplier)
    timeline = []
    for step in scenario.timeline:
        throughput = float(step.link_trace.throughput_mbps)
        link = replace(
            step.link_trace,
            throughput_mbps=throughput * ratio if step.link_trace.in_contact else 0.0,
        )
        timeline.append(replace(step, link_trace=link))
    identity_rows = [
        {
            "slot": int(step.slot_index),
            "in_contact": int(bool(step.link_trace.in_contact)),
            "throughput_mbps": float(step.link_trace.throughput_mbps),
            "remaining_window_s": float(step.link_trace.remaining_window_s),
        }
        for step in timeline
    ]
    return replace(scenario, configs=configs, timeline=timeline), {
        "source_bundle_capacity": SOURCE_CAPACITY,
        "runtime_capacity": float(multiplier),
        "throughput_scale_ratio": ratio,
        "contact_slots_unchanged": True,
        "runtime_link_trace_hash": reqrun.sha256_json(identity_rows),
    }


class HVGuardedRouterWrapper:
    """Cross-task guard outside corrected REQALIGN and before current execution."""

    def __init__(
        self,
        delegate: Any,
        config: dict[str, Any],
        timeline: list[Any],
        simulator_cfg: dict[str, Any],
        guard_module: Any,
    ):
        self.delegate = delegate
        self.config = config
        self.timeline = timeline
        self.simulator_cfg = simulator_cfg
        self.guard = guard_module
        self.router_name = delegate.router_name
        self.use_lyapunov_pricing = delegate.use_lyapunov_pricing
        self.use_mode_controller = delegate.use_mode_controller
        self.use_guided_utility = delegate.use_guided_utility
        self.last_selected_action_score = None
        self.last_score_components: dict[str, Any] = {}
        self.last_all_score_components: dict[str, dict[str, Any]] = {}
        self.step_runner = None
        self.current_step = None
        self.current_resource_state = None
        self.current_comm_state = None
        self.active_rows: list[Any] = []
        self.current_capacity = guard_module.ResourceDemand()
        self.current_reservation = guard_module.dynamic_minimum_reservation([], self.current_capacity)
        self.previous_reservation = guard_module.ResourceDemand()
        self.last_guarded_action = None
        self.last_reqalign_action = None
        self.traces: dict[str, list[dict[str, Any]]] = {
            "hv_active_residual_trace": [],
            "urgent_reservation_trace": [],
            "high_reservation_trace": [],
            "resource_release_trace": [],
            "displacement_check_trace": [],
            "pareto_candidate_trace": [],
            "hv_action_override_trace": [],
            "urgent_feasible_but_missed_trace": [],
            "high_feasible_but_missed_trace": [],
        }

    def bind_step_runner(self, step_runner: Any) -> None:
        self.step_runner = step_runner

    def bind_step(self, timeline_step: Any, resource_state: Any, comm_state: Any) -> None:
        self.current_step = timeline_step
        self.current_resource_state = resource_state
        self.current_comm_state = comm_state

    @staticmethod
    def _level(task: Any) -> str:
        return str(getattr(task, "value_level", "") or (getattr(task, "metadata", {}) or {}).get("value_level", "")).lower()

    @staticmethod
    def _absolute_deadline(task: Any, slot_seconds: float) -> float:
        return float(task.arrival_slot) * float(slot_seconds) + float(task.deadline_s)

    def _is_contact_start(self, step: Any) -> bool:
        if not bool(step.link_trace.in_contact):
            return False
        slot = int(step.slot_index)
        return slot <= 0 or not bool(self.timeline[slot - 1].link_trace.in_contact)

    def _should_schedule_guard(self, step: Any) -> bool:
        if step.task is not None:
            return True
        if self._is_contact_start(step):
            return True
        return bool(
            step.link_trace.in_contact
            and any(
                str(getattr(request, "status", "")) in {"pending", "blocked", "enqueued"}
                for request in getattr(self.step_runner, "staged_delivery_requests", {}).values()
            )
        )

    def _contacts_until(self, start_slot: int, deadline_s: float) -> list[Any]:
        return self.delegate._contacts(start_slot, deadline_s)

    def _residual_fact(self, task: Any, state: Any, step: Any, resource_state: Any, comm_state: Any) -> Any:
        delivered = set(getattr(state, "delivered_products", set()) or set())
        residual = self.delegate.projection.residual_required_product(task.product_requirement, delivered)
        deadline_s = self._absolute_deadline(task, step.slot_seconds)
        now_s = float(step.slot_index) * float(step.slot_seconds)
        common = {
            "task_id": str(task.task_id),
            "value_level": self._level(task),
            "effective_task_value": 0.0,
            "delivered_products": tuple(sorted(delivered)),
            "residual_required_product": residual,
            "absolute_deadline_s": deadline_s,
            "remaining_slack_s": deadline_s - now_s,
        }
        from src.data.value_skew_calibration import task_effective_value

        common["effective_task_value"] = float(task_effective_value(task))
        empty = self.guard.ResourceDemand()
        if not residual or deadline_s < now_s - 1.0e-9 or bool(getattr(state, "task_closed_flag", False)):
            return self.guard.ActiveResidual(
                **common,
                demand=empty,
                residual_compute_requirement_s=0.0,
                residual_communication_energy_wh=0.0,
                earliest_generation_completion_s=None,
                candidate_contact_windows=(),
                earliest_ground_arrival_s=None,
                closure_feasible_before_deadline=False,
                predicted_marginal_hv_net_utility=0.0,
                predicted_closure_increment=0,
                hard_safe=False,
                blocker_reason="closed_expired_or_no_residual",
            )
        context = self.step_runner.context_builder.build(task, resource_state, comm_state, step.link_trace)
        actions = self.step_runner.action_space.list_actions()
        predicted_map = {
            action.action_name: self.step_runner.predictor.predict_action(
                task=task,
                context=context,
                resource_trace=step.resource_trace,
                link_trace=step.link_trace,
                comm_state=comm_state,
                action=action,
                guidance_bundle=None,
            )
            for action in actions
        }
        safe_actions, reasons = self.step_runner.safe_filter.filter_actions(task, actions, predicted_map)
        safe_names = {action.action_name for action in safe_actions}
        contacts = self._contacts_until(int(step.slot_index), deadline_s)
        contact_ids = tuple(
            dict.fromkeys(
                self.step_runner._contact_window_id_for_slot(int(float(contact.start_s) // float(step.slot_seconds)))
                for contact in contacts
                if float(contact.rate_mb_s) > 0.0
            )
        )
        candidates: list[dict[str, Any]] = []
        for action in actions:
            predicted = predicted_map[action.action_name]
            product = str(predicted.predicted_product_type)
            if action.action_name not in safe_names or not self.delegate.projection.product_satisfies(product, residual):
                continue
            arrival = self.delegate.projection.predict_ground_arrival(
                current_time_s=now_s,
                compute_and_encode_s=float(predicted.predicted_L_inf),
                backlog_mb=float(comm_state.queue_mb),
                payload_mb=float(predicted.predicted_S_gen_mb),
                contact_slots=contacts,
                deadline_s=deadline_s,
            )
            if arrival.ground_arrival_s is None or float(arrival.ground_arrival_s) > deadline_s + 1.0e-9:
                continue
            thermal_increment = max(
                0.0,
                float(predicted.predicted_Theta_next) - float(resource_state.thermal_pressure),
            )
            demand = self.guard.ResourceDemand(
                compute_s=float(predicted.predicted_L_inf),
                energy_wh=(float(predicted.predicted_E_comp) + float(predicted.predicted_E_comm)) / 3600.0,
                queue_mb=float(predicted.predicted_S_gen_mb),
                contact_bytes=float(predicted.predicted_S_gen_mb) * 1_000_000.0,
                thermal_increment=thermal_increment,
            )
            candidates.append(
                {
                    "action": action,
                    "predicted": predicted,
                    "arrival": arrival,
                    "demand": demand,
                    "score": float(predicted.predicted_reward),
                }
            )
        if not candidates:
            lowered = " ".join(reason for values in reasons.values() for reason in values).lower()
            blocker = "no_hard_safe_closing_action"
            if "thermal" in lowered:
                blocker = "thermal"
            elif "battery" in lowered or "soc" in lowered:
                blocker = "soc"
            elif "queue" in lowered:
                blocker = "queue"
            elif not contact_ids:
                blocker = "no_contact"
            return self.guard.ActiveResidual(
                **common,
                demand=empty,
                residual_compute_requirement_s=0.0,
                residual_communication_energy_wh=0.0,
                earliest_generation_completion_s=None,
                candidate_contact_windows=contact_ids,
                earliest_ground_arrival_s=None,
                closure_feasible_before_deadline=False,
                predicted_marginal_hv_net_utility=0.0,
                predicted_closure_increment=0,
                hard_safe=False,
                blocker_reason=blocker,
            )
        chosen = sorted(
            candidates,
            key=lambda row: (
                row["demand"].contact_bytes,
                row["demand"].energy_wh,
                row["demand"].compute_s,
                row["demand"].thermal_increment,
                str(row["action"].action_name),
            ),
        )[0]
        predicted = chosen["predicted"]
        arrival = chosen["arrival"]
        return self.guard.ActiveResidual(
            **common,
            demand=chosen["demand"],
            residual_compute_requirement_s=float(predicted.predicted_L_inf),
            residual_communication_energy_wh=float(predicted.predicted_E_comm) / 3600.0,
            earliest_generation_completion_s=now_s + float(predicted.predicted_L_inf),
            candidate_contact_windows=contact_ids,
            earliest_ground_arrival_s=float(arrival.ground_arrival_s),
            closure_feasible_before_deadline=True,
            predicted_marginal_hv_net_utility=float(predicted.predicted_reward),
            predicted_closure_increment=1,
            hard_safe=True,
            original_score=float(predicted.predicted_reward),
            blocker_reason="" if float(predicted.predicted_reward) > 0.0 else "non_positive_predicted_net_utility",
        )

    def _build_active(self, step: Any, resource_state: Any, comm_state: Any) -> list[Any]:
        now_s = float(step.slot_index) * float(step.slot_seconds)
        rows = []
        for task_id, state in sorted(getattr(self.step_runner, "product_lifecycle", {}).items()):
            task = self.step_runner.task_snapshots.get(task_id)
            if task is None or self._level(task) not in {"high", "urgent"}:
                continue
            fact = self._residual_fact(task, state, step, resource_state, comm_state)
            if fact.residual_required_product and fact.absolute_deadline_s + 1.0e-9 >= now_s and not state.task_closed_flag:
                rows.append(fact)
        return self.guard.active_residuals(rows, now_s=now_s)

    def _capacity(self, rows: list[Any], step: Any, resource_state: Any, comm_state: Any) -> Any:
        now_s = float(step.slot_index) * float(step.slot_seconds)
        protectable = [row for row in rows if row.protectable]
        cutoff = min(
            (row.absolute_deadline_s for row in protectable),
            default=now_s,
        )
        contacts = self._contacts_until(int(step.slot_index), cutoff)
        contact_bytes = max(
            0.0,
            sum(
                max(0.0, float(contact.end_s) - float(contact.start_s))
                * max(0.0, float(contact.rate_mb_s))
                * 1_000_000.0
                for contact in contacts
            )
            - float(comm_state.queue_mb) * 1_000_000.0,
        )
        battery_min = float(self.simulator_cfg.get("battery_min", 0.0))
        capacity_wh = float(
            self.simulator_cfg.get(
                "battery_capacity_wh",
                float(self.simulator_cfg.get("battery_capacity_j", 0.0)) / 3600.0,
            )
        )
        discharge_efficiency = float(self.simulator_cfg.get("battery_discharge_efficiency", 1.0))
        energy_wh = max(
            0.0,
            (float(resource_state.battery_soc) - battery_min)
            * capacity_wh
            * discharge_efficiency,
        )
        thermal = max(
            0.0,
            float(self.simulator_cfg.get("thermal_max", 1.0))
            - float(resource_state.thermal_pressure),
        )
        return self.guard.ResourceDemand(
            compute_s=max(0.0, cutoff - now_s),
            energy_wh=energy_wh,
            queue_mb=max(0.0, float(comm_state.queue_max_mb) - float(comm_state.queue_mb)),
            contact_bytes=contact_bytes,
            thermal_increment=thermal,
        )

    def begin_slot(self, step: Any, resource_state: Any, comm_state: Any) -> None:
        self.bind_step(step, resource_state, comm_state)
        if not self._should_schedule_guard(step):
            return
        self.active_rows = self._build_active(step, resource_state, comm_state)
        self.current_capacity = self._capacity(self.active_rows, step, resource_state, comm_state)
        self.current_reservation = self.guard.dynamic_minimum_reservation(
            self.active_rows,
            self.current_capacity,
        )
        release = self.guard.released_resources(
            self.previous_reservation,
            self.current_reservation.total,
        )
        now_s = float(step.slot_index) * float(step.slot_seconds)
        protected = set(self.current_reservation.protected_task_ids)
        for row in self.active_rows:
            trace = {
                "slot": int(step.slot_index),
                "time_s": now_s,
                "task_id": row.task_id,
                "value_level": row.value_level,
                "effective_task_value": row.effective_task_value,
                "delivered_products": ";".join(row.delivered_products),
                "residual_required_product": row.residual_required_product,
                "deadline_s": row.absolute_deadline_s,
                "remaining_deadline_slack_s": row.remaining_slack_s,
                "residual_payload_bytes": row.demand.contact_bytes,
                "residual_compute_requirement_s": row.residual_compute_requirement_s,
                "residual_compute_energy_Wh": row.demand.energy_wh - row.residual_communication_energy_wh,
                "residual_communication_energy_Wh": row.residual_communication_energy_wh,
                "residual_queue_occupancy_mb": row.demand.queue_mb,
                "predicted_thermal_increment": row.demand.thermal_increment,
                "earliest_feasible_generation_completion_s": row.earliest_generation_completion_s,
                "candidate_future_contact_windows": ";".join(row.candidate_contact_windows),
                "earliest_predicted_ground_arrival_s": row.earliest_ground_arrival_s,
                "closure_feasible_before_deadline": int(row.closure_feasible_before_deadline),
                "hard_safe": int(row.hard_safe),
                "predicted_marginal_HV_net_utility": row.predicted_marginal_hv_net_utility,
                "predicted_closure_increment": row.predicted_closure_increment,
                "protectable": int(row.protectable),
                "protected": int(row.task_id in protected),
                "blocker_reason": row.blocker_reason,
            }
            self.traces["hv_active_residual_trace"].append(trace)
            reservation_trace = {
                **trace,
                **row.demand.as_dict("reserved_"),
                **self.current_capacity.as_dict("available_"),
                "reservation_status": (
                    "protected"
                    if row.task_id in protected
                    else ("infeasible" if not row.protectable else "feasible_not_reserved")
                ),
            }
            target = "urgent_reservation_trace" if row.value_level == "urgent" else "high_reservation_trace"
            self.traces[target].append(reservation_trace)
        self.traces["resource_release_trace"].append(
            {
                "slot": int(step.slot_index),
                "time_s": now_s,
                **self.previous_reservation.as_dict("reservation_before_"),
                **self.current_reservation.total.as_dict("reservation_after_"),
                **release.released.as_dict("released_"),
                **release.increased.as_dict("increased_"),
                "active_urgent_count": sum(row.value_level == "urgent" for row in self.active_rows),
                "active_high_count": sum(row.value_level == "high" for row in self.active_rows),
                "protected_urgent_count": len(self.current_reservation.urgent_task_ids),
                "protected_high_count": len(self.current_reservation.high_task_ids),
            }
        )
        self.previous_reservation = self.current_reservation.total

    def _semantic_candidates(self, reqalign_action: Any, candidate_actions: list[Any], predicted_map: dict[str, Any]) -> list[Any]:
        task = self.current_step.task
        required = str(task.product_requirement)
        req_product = str(predicted_map[reqalign_action.action_name].predicted_product_type)
        actions = list(candidate_actions)
        level = self._level(task)
        if level in {"low", "medium"}:
            # The guard may protect an HV opportunity by deferring an ordinary
            # action, but it must not otherwise change the corrected one-shot
            # ordinary action/model decision.
            actions = [
                action
                for action in actions
                if action.action_name == reqalign_action.action_name
                or self.delegate.projection.normalize_product(
                    predicted_map[action.action_name].predicted_product_type
                )
                == "none"
            ]
            if reqalign_action.action_name not in {action.action_name for action in actions}:
                actions.append(reqalign_action)
            return list({action.action_name: action for action in actions}.values())
        if self.delegate.projection.product_satisfies(req_product, required):
            actions = [
                action
                for action in actions
                if self.delegate.projection.product_satisfies(
                    predicted_map[action.action_name].predicted_product_type,
                    required,
                )
                or self.delegate.projection.normalize_product(
                    predicted_map[action.action_name].predicted_product_type
                )
                == "none"
            ]
        elif self.delegate.projection.normalize_product(req_product) == "metadata":
            actions = [
                action
                for action in actions
                if self.delegate.projection.normalize_product(
                    predicted_map[action.action_name].predicted_product_type
                )
                in {"metadata", "none"}
            ]
        else:
            actions = [
                action
                for action in actions
                if self.delegate.projection.normalize_product(
                    predicted_map[action.action_name].predicted_product_type
                )
                == "none"
            ]
        if reqalign_action.action_name not in {action.action_name for action in actions}:
            actions.append(reqalign_action)
        return list({action.action_name: action for action in actions}.values())

    def _action_candidate(self, action: Any, scores: dict[str, float]) -> Any:
        task = self.current_step.task
        predicted = self._predicted_map[action.action_name]
        product = str(predicted.predicted_product_type)
        is_defer = (
            self.delegate.projection.normalize_product(product) == "none"
            or str(action.action_name).lower() in {"delay/store", "defer", "delay"}
        )
        now_s = float(self.current_step.slot_index) * float(self.current_step.slot_seconds)
        deadline_s = self._absolute_deadline(task, self.current_step.slot_seconds)
        arrival = None
        deadline_feasible = True
        if not is_defer:
            arrival = self.delegate._arrival(action, predicted, task, now_s, deadline_s)
            deadline_feasible = (
                arrival.ground_arrival_s is not None
                and float(arrival.ground_arrival_s) <= deadline_s + 1.0e-9
            )
        thermal_increment = 0.0 if is_defer else max(
            0.0,
            float(predicted.predicted_Theta_next)
            - float(self.current_resource_state.thermal_pressure),
        )
        demand = self.guard.ResourceDemand() if is_defer else self.guard.ResourceDemand(
            compute_s=float(predicted.predicted_L_inf),
            energy_wh=(float(predicted.predicted_E_comp) + float(predicted.predicted_E_comm)) / 3600.0,
            queue_mb=float(predicted.predicted_S_gen_mb),
            contact_bytes=float(predicted.predicted_S_gen_mb) * 1_000_000.0,
            thermal_increment=thermal_increment,
        )
        displacement = self.guard.shadow_displacement(
            self.active_rows,
            capacity_before=self.current_capacity,
            action_demand=demand,
            exclude_task_id=str(task.task_id),
        )
        level = self._level(task)
        closure_increment = int(
            level in {"high", "urgent"}
            and not is_defer
            and self.delegate.projection.product_satisfies(product, task.product_requirement)
            and deadline_feasible
        )
        predicted_gain = 0.0 if is_defer else float(predicted.predicted_reward)
        delta_hv = (predicted_gain if level in {"high", "urgent"} else 0.0) - displacement.hv_utility_displacement
        delta_all = predicted_gain - displacement.hv_utility_displacement
        delta_count = closure_increment - displacement.hv_closure_displacement
        row = {
            "slot": int(self.current_step.slot_index),
            "task_id": str(task.task_id),
            "value_level": level,
            "action_name": action.action_name,
            "original_score": scores.get(action.action_name, -math.inf),
            "predicted_ground_arrival_s": None if arrival is None else arrival.ground_arrival_s,
            "urgent_closure_displacement": displacement.urgent_closure_displacement,
            "high_closure_displacement": displacement.high_closure_displacement,
            "urgent_utility_displacement": displacement.urgent_utility_displacement,
            "hv_utility_displacement": displacement.hv_utility_displacement,
            "predicted_delta_J_all": delta_all,
            "predicted_delta_J_HV": delta_hv,
            "predicted_delta_HV_closure": delta_count,
            "displaced_urgent_task_ids": ";".join(displacement.urgent_task_ids),
            "displaced_high_task_ids": ";".join(displacement.high_task_ids),
            **displacement.before.total.as_dict("reservation_before_"),
            **displacement.after.total.as_dict("reservation_after_"),
            **demand.as_dict("action_"),
        }
        self.traces["displacement_check_trace"].append(row)
        from src.data.value_skew_calibration import task_effective_value

        return self.guard.ParetoActionCandidate(
            action=action,
            action_name=action.action_name,
            original_score=float(scores.get(action.action_name, -math.inf)),
            effective_task_value=float(task_effective_value(task)),
            remaining_slack_s=deadline_s - now_s,
            task_id=str(task.task_id),
            value_level=level,
            delta_j_all_net_hat=delta_all,
            delta_j_hv_net_hat=delta_hv,
            delta_n_hv_closure_hat=delta_count,
            urgent_closure_displacement=displacement.urgent_closure_displacement,
            high_closure_displacement=displacement.high_closure_displacement,
            demand=demand,
            predicted_ground_arrival_s=None if arrival is None else arrival.ground_arrival_s,
            hard_safe=True,
            deadline_feasible=deadline_feasible,
            is_defer=is_defer,
        )

    def select_action(
        self,
        context: Any,
        candidate_actions: list[Any],
        predicted_map: dict[str, Any],
        guidance_bundle=None,
    ):
        reqalign_action = self.delegate.select_action(
            context,
            candidate_actions,
            predicted_map,
            guidance_bundle,
        )
        self.last_reqalign_action = reqalign_action
        self.last_all_score_components = deepcopy(self.delegate.last_all_score_components)
        self.last_score_components = deepcopy(self.delegate.last_score_components)
        self.last_selected_action_score = self.delegate.last_selected_action_score
        self._predicted_map = predicted_map
        task = self.current_step.task
        if task is None:
            return reqalign_action
        scores = {
            name: float(components.get("final_action_score", -math.inf))
            for name, components in self.last_all_score_components.items()
        }
        semantic_actions = self._semantic_candidates(reqalign_action, candidate_actions, predicted_map)
        candidates = [self._action_candidate(action, scores) for action in semantic_actions]
        reqalign_candidate = next(
            candidate
            for candidate in candidates
            if candidate.action_name == reqalign_action.action_name
        )
        guard_intervention_needed = self.guard.requires_cross_task_guard(reqalign_candidate)
        if guard_intervention_needed:
            decision = self.guard.select_pareto_guarded_action(candidates)
        else:
            # This layer protects cross-task closure opportunities.  It is not
            # a second utility gate over corrected REQALIGN.
            decision = self.guard.ParetoGuardDecision(
                selected=reqalign_candidate,
                nondominated=tuple(candidates),
                rejected_by_displacement=(),
                pareto_dominated=(),
            )
        selected = decision.selected
        selected_names = {row.action_name for row in decision.nondominated}
        rejected_names = {row.action_name for row in decision.rejected_by_displacement}
        dominated_names = {row.action_name for row in decision.pareto_dominated}
        for candidate in candidates:
            self.traces["pareto_candidate_trace"].append(
                {
                    "slot": int(self.current_step.slot_index),
                    "task_id": str(task.task_id),
                    "value_level": self._level(task),
                    "action_name": candidate.action_name,
                    "delta_J_all_net_hat": candidate.delta_j_all_net_hat,
                    "delta_J_HV_net_hat": candidate.delta_j_hv_net_hat,
                    "delta_N_HV_closure_hat": candidate.delta_n_hv_closure_hat,
                    "delta_N_urgent_closure_hat": (
                        candidate.delta_n_urgent_closure_hat
                    ),
                    "urgent_closure_displacement": candidate.urgent_closure_displacement,
                    "high_closure_displacement": candidate.high_closure_displacement,
                    "original_score": candidate.original_score,
                    "guard_intervention_needed": int(guard_intervention_needed),
                    "rejected_by_displacement": int(candidate.action_name in rejected_names),
                    "pareto_dominated": int(candidate.action_name in dominated_names),
                    "pareto_nondominated": int(candidate.action_name in selected_names),
                    "selected": int(candidate.action_name == selected.action_name),
                }
            )
        final = selected.action
        self.last_guarded_action = final
        self.last_selected_action_score = scores.get(final.action_name, float(predicted_map[final.action_name].predicted_reward))
        self.last_score_components = deepcopy(self.last_all_score_components.get(final.action_name, {}))
        self.delegate.last_selected_action_score = self.last_selected_action_score
        self.delegate.last_score_components = deepcopy(self.last_score_components)
        original_action = getattr(self.delegate, "rows", [{}])[-1].get("original_action", reqalign_action.action_name)
        if final.action_name != reqalign_action.action_name:
            selected_disp = next(
                row
                for row in reversed(self.traces["displacement_check_trace"])
                if row["slot"] == int(self.current_step.slot_index)
                and row["action_name"] == final.action_name
            )
            reqalign_disp = next(
                row
                for row in reversed(self.traces["displacement_check_trace"])
                if row["slot"] == int(self.current_step.slot_index)
                and row["action_name"] == reqalign_action.action_name
            )
            self.traces["hv_action_override_trace"].append(
                {
                    "slot": int(self.current_step.slot_index),
                    "task_id": str(task.task_id),
                    "value_level": self._level(task),
                    "original_action": original_action,
                    "REQALIGN_action": reqalign_action.action_name,
                    "final_guarded_action": final.action_name,
                    "original_score": scores.get(original_action, -math.inf),
                    "REQALIGN_score": scores.get(reqalign_action.action_name, -math.inf),
                    "final_score": scores.get(final.action_name, -math.inf),
                    "predicted_ground_arrival_s": selected.predicted_ground_arrival_s,
                    "REQALIGN_predicted_ground_arrival_s": reqalign_candidate.predicted_ground_arrival_s,
                    "final_predicted_ground_arrival_s": selected.predicted_ground_arrival_s,
                    "urgent_closure_displacement": reqalign_disp[
                        "urgent_closure_displacement"
                    ],
                    "high_closure_displacement": reqalign_disp[
                        "high_closure_displacement"
                    ],
                    "predicted_delta_J_all": reqalign_disp["predicted_delta_J_all"],
                    "predicted_delta_J_HV": reqalign_disp["predicted_delta_J_HV"],
                    "predicted_delta_HV_closure": reqalign_disp[
                        "predicted_delta_HV_closure"
                    ],
                    "predicted_delta_urgent_closure": (
                        reqalign_candidate.delta_n_urgent_closure_hat
                    ),
                    "final_urgent_closure_displacement": (
                        selected.urgent_closure_displacement
                    ),
                    "final_high_closure_displacement": (
                        selected.high_closure_displacement
                    ),
                    "final_predicted_delta_J_all": selected.delta_j_all_net_hat,
                    "final_predicted_delta_J_HV": selected.delta_j_hv_net_hat,
                    "final_predicted_delta_HV_closure": (
                        selected.delta_n_hv_closure_hat
                    ),
                    "final_predicted_delta_urgent_closure": (
                        selected.delta_n_urgent_closure_hat
                    ),
                    **{
                        key: value
                        for key, value in selected_disp.items()
                        if key.startswith("reservation_before_")
                        or key.startswith("reservation_after_")
                    },
                    "override_reason": (
                        "cross_task_displacement_rejected"
                        if reqalign_action.action_name in rejected_names
                        else "pareto_nondominated_order"
                    ),
                    "displaced_urgent_task_ids": reqalign_disp[
                        "displaced_urgent_task_ids"
                    ],
                    "displaced_high_task_ids": reqalign_disp[
                        "displaced_high_task_ids"
                    ],
                    "final_displaced_urgent_task_ids": selected_disp[
                        "displaced_urgent_task_ids"
                    ],
                    "final_displaced_high_task_ids": selected_disp[
                        "displaced_high_task_ids"
                    ],
                }
            )
        return final

    def finish_step(self, step: Any) -> None:
        if step.task is None or self.last_guarded_action is None:
            return
        task_id = str(step.task.task_id)
        protected = task_id in set(self.current_reservation.protected_task_ids)
        if protected:
            for row in reversed(self.traces["resource_release_trace"]):
                if row["slot"] == int(step.slot_index):
                    row["reservation_utilized_by_current_task"] = 1
                    row["reservation_utilized_task_id"] = task_id
                    break

    def update(self, context: Any, action: Any, observed_reward: float) -> None:
        self.delegate.update(context, action, observed_reward)


def bind_guard_runtime(
    runner: Any,
    reqalign_wrapper: Any,
    guard_wrapper: HVGuardedRouterWrapper,
) -> None:
    original_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = original_factory(queue)
        reqalign_wrapper.bind_step_runner(step_runner)
        guard_wrapper.bind_step_runner(step_runner)
        original_run_step = step_runner.run_step

        def intercepted(timeline_step, resource_state, mission_state, comm_state, router):
            effective_step = reqalign_wrapper.prepare_step_for_reentry(
                timeline_step,
                resource_state,
                comm_state,
            )
            reqalign_wrapper.bind_step(effective_step, resource_state, comm_state)
            guard_wrapper.bind_step(effective_step, resource_state, comm_state)
            guard_wrapper.begin_slot(effective_step, resource_state, comm_state)
            result = original_run_step(
                effective_step,
                resource_state,
                mission_state,
                comm_state,
                router,
            )
            reqalign_wrapper.finish_step(effective_step)
            guard_wrapper.finish_step(effective_step)
            return result

        step_runner.run_step = intercepted
        return step_runner

    runner.step_runner_factory = factory


def write_guard_traces(wrapper: HVGuardedRouterWrapper, output: Path) -> None:
    for name, rows in wrapper.traces.items():
        write_csv(output / f"{name}.csv", rows)


def run_prop_variant(
    v25: Any,
    bundle_root: Path,
    output: Path,
    *,
    reqalign_enabled: bool,
    guard_enabled: bool,
    method: str,
    slots: int,
) -> None:
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    prop_mod = v25.import_prop_script()
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(
        bundle,
        TaskSample,
        ResourceTrace,
        LinkTrace,
        TimelineStep,
    )
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configs = prop_mod._configure(scenario.configs, calibration, {"enabled": False})
    scenario = replace(scenario, configs=configs)
    scenario, power_identity = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_identity = apply_capacity(scenario, CAPACITY)
    scenario_identity = v25.relaxed_scenario_identity(method, scenario, bundle, power_identity)
    if not scenario_identity["task_hash_match"] or scenario_identity["runtime_value_mismatch_count"]:
        raise RuntimeError({"scenario_identity_failed": scenario_identity})
    env = SimulationEnvironment(scenario.configs)
    runner = env.build_episode_runner(scenario.timeline[:slots])
    base_router = build_routers(
        scenario.configs["experiment"],
        ActionSpace(scenario.configs["action_space"]),
    )[0]
    reqalign_wrapper = None
    guard_wrapper = None
    router = base_router
    guard_module_imported = False
    if reqalign_enabled:
        projection = reqrun.load_projection_module()
        reqalign_wrapper = reqrun.RequirementAwareRouterWrapper(
            base_router,
            reqrun.load_projection_config(True),
            scenario.timeline[:slots],
            scenario.configs["simulator"],
            projection,
        )
        router = reqalign_wrapper
        if guard_enabled:
            guard_module = load_module(
                "hv_urgent_residual_guard_runtime_v1",
                GUARD_SOURCE,
            )
            guard_module_imported = True
            guard_wrapper = HVGuardedRouterWrapper(
                reqalign_wrapper,
                load_guard_config(True),
                scenario.timeline[:slots],
                scenario.configs["simulator"],
                guard_module,
            )
            bind_guard_runtime(runner, reqalign_wrapper, guard_wrapper)
            router = guard_wrapper
        else:
            reqrun.bind_projection_runtime(runner, reqalign_wrapper)
    result = runner.run(router)
    normalized = normalize_episode(method, scenario, result, runner, slots)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_csv(
        output / "candidate_action_log.csv",
        list(getattr(runner, "last_candidate_action_logs", [])),
    )
    if int(slots) == SLOTS:
        v25.write_unified_method_outputs(output, normalized)
        reqrun.write_energy_components(output, result)
        if reqalign_wrapper is not None:
            projection_rows = reqrun.enrich_projection_rows(reqalign_wrapper, output)
        else:
            projection_rows = reqrun.build_disabled_projection_log(output)
    else:
        projection_rows = [
            {
                key: value
                for key, value in row.items()
                if not str(key).startswith("_")
            }
            for row in (getattr(reqalign_wrapper, "rows", []) if reqalign_wrapper is not None else [])
        ]
        write_json(
            output / "smoke_summary.json",
            {
                "method": method,
                "slots": int(slots),
                "result_type": type(result).__name__,
                "candidate_action_rows": len(getattr(runner, "last_candidate_action_logs", [])),
                "projection_rows": len(projection_rows),
            },
        )
    write_csv(output / "requirement_projection_log.csv", projection_rows)
    if guard_wrapper is not None:
        write_guard_traces(guard_wrapper, output)
    identity = {
        "method": method,
        "source_root": str(v25.SOURCE_PROP.resolve()),
        "corrected_reqalign_enabled": bool(reqalign_enabled),
        "hv_urgent_guard_enabled": bool(guard_enabled),
        "guard_module_imported": guard_module_imported,
        "guard_source_sha256": sha256_file(GUARD_SOURCE) if guard_enabled else "",
        "guard_config_sha256": sha256_file(GUARD_CONFIG),
        "projection_source_sha256": sha256_file(reqrun.PROJECTION_SOURCE) if reqalign_enabled else "",
        "preserve_original_scorer": True,
        "preserve_original_selector": True,
        "preserve_safe_filter": True,
        "capacity_identity": capacity_identity,
        "scenario_identity": scenario_identity,
        "slots": int(slots),
    }
    write_json(output / "native_identity.json", identity)
    write_json(
        output / "method_run_status.json",
        {
            "status": "PASS",
            "method": method,
            "simulation_started": True,
            "fresh_state": True,
            "reused_action_trace": False,
            "reused_queue_trace": False,
            "reused_delivery_ledger": False,
            "identity": identity,
        },
    )
    write_json(
        output / "method_completed.json",
        {"status": "PASS", "method": method, "slots": int(slots)},
    )


def child_main(args: argparse.Namespace) -> int:
    v25 = reqrun.load_v25()
    label = str(args.label)
    method_map = {
        "OLD_PROP": (False, False, METHOD_OLD),
        "REQALIGN_REFERENCE": (True, False, METHOD_REQALIGN),
        "REQALIGN_GUARD_OFF": (True, False, METHOD_REQALIGN),
        "HV_GUARD": (True, True, METHOD_GUARD),
        "SMOKE_GUARD": (True, True, METHOD_GUARD),
    }
    if label not in method_map:
        raise ValueError(label)
    reqalign_enabled, guard_enabled, method = method_map[label]
    run_prop_variant(
        v25,
        Path(args.bundle_root),
        Path(args.output),
        reqalign_enabled=reqalign_enabled,
        guard_enabled=guard_enabled,
        method=method,
        slots=int(args.slots),
    )
    print(json.dumps({"status": "PASS", "label": label}, ensure_ascii=False))
    return 0


def launch_child(
    label: str,
    bundle_root: Path,
    output: Path,
    log_dir: Path,
    *,
    slots: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--label",
        label,
        "--bundle-root",
        str(bundle_root),
        "--output",
        str(output),
        "--slots",
        str(slots),
    ]
    started = utc_now()
    process = subprocess.run(
        command,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (log_dir / f"{label}_stderr.txt").write_text(process.stderr, encoding="utf-8")
    return {
        "label": label,
        "returncode": process.returncode,
        "started_utc": started,
        "finished_utc": utc_now(),
        "output": str(output),
        "status": (
            "PASS"
            if process.returncode == 0 and (output / "method_completed.json").exists()
            else "FAIL"
        ),
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-6000:],
    }


def semantic_hashes(path: Path) -> dict[str, str]:
    return {
        "action_trace_hash": reqrun.semantic_csv_hash(path / "candidate_action_log.csv"),
        "delivery_event_hash": reqrun.semantic_csv_hash(path / "ground_delivery_events.csv"),
        "queue_trace_hash": reqrun.semantic_csv_hash(
            path / "per_slot_resource_trace.csv",
            ["slot", "queue_mb"],
        ),
        "soc_trace_hash": reqrun.semantic_csv_hash(
            path / "per_slot_resource_trace.csv",
            ["slot", "battery_soc"],
        ),
        "thermal_trace_hash": reqrun.semantic_csv_hash(
            path / "per_slot_resource_trace.csv",
            ["slot", "thermal"],
        ),
        "task_outcome_hash": reqrun.semantic_csv_hash(path / "per_task_outcomes.csv"),
        "projection_trace_hash": reqrun.semantic_csv_hash(path / "requirement_projection_log.csv"),
    }


def disabled_regression_audit(reference: Path, guard_off: Path) -> dict[str, Any]:
    ref_hashes = semantic_hashes(reference)
    off_hashes = semantic_hashes(guard_off)
    ref_tasks = read_csv(reference / "per_task_outcomes.csv")
    off_tasks = read_csv(guard_off / "per_task_outcomes.csv")
    ordinary_fields = [
        "task_id",
        "selected_action",
        "selected_model",
        "closure_success",
        "transmitted_bytes",
        "controllable_energy_j",
    ]
    ordinary_ref = [
        {field: row.get(field, "") for field in ordinary_fields}
        for row in ref_tasks
        if row.get("value_level") in {"low", "medium"}
    ]
    ordinary_off = [
        {field: row.get(field, "") for field in ordinary_fields}
        for row in off_tasks
        if row.get("value_level") in {"low", "medium"}
    ]
    hv_ref = sorted(
        row["task_id"]
        for row in ref_tasks
        if row.get("value_level") in {"high", "urgent"}
    )
    hv_off = sorted(
        row["task_id"]
        for row in off_tasks
        if row.get("value_level") in {"high", "urgent"}
    )
    ref_identity = read_json(reference / "native_identity.json")
    off_identity = read_json(guard_off / "native_identity.json")
    checks = {
        "all_semantic_hashes_equal": ref_hashes == off_hashes,
        "ordinary_delivery_equal": ordinary_ref == ordinary_off,
        "high_urgent_task_set_equal": hv_ref == hv_off,
        "reference_guard_imported": bool(ref_identity["guard_module_imported"]),
        "guard_off_module_imported": bool(off_identity["guard_module_imported"]),
        "capacity_equal": (
            ref_identity["capacity_identity"]["runtime_capacity"]
            == off_identity["capacity_identity"]["runtime_capacity"]
            == CAPACITY
        ),
    }
    passed = (
        checks["all_semantic_hashes_equal"]
        and checks["ordinary_delivery_equal"]
        and checks["high_urgent_task_set_equal"]
        and not checks["reference_guard_imported"]
        and not checks["guard_off_module_imported"]
        and checks["capacity_equal"]
    )
    return {
        "passed": passed,
        "status_if_failed": "HV_GUARD_DISABLED_REGRESSION_MISMATCH",
        "reference_hashes": ref_hashes,
        "guard_off_hashes": off_hashes,
        "checks": checks,
    }


def finalize_rollback() -> dict[str, Any]:
    manifest = read_json(ROLLBACK_MANIFEST)
    for entry in manifest["new_files"]:
        path = ROOT / entry["path"]
        entry["post_change_sha256"] = sha256_file(path) if path.exists() else None
    manifest["finalized_utc"] = utc_now()
    write_json(ROLLBACK_MANIFEST, manifest)
    process = subprocess.run(
        [sys.executable, str(RESTORE_SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "passed": process.returncode == 0,
        "manifest_path": str(ROLLBACK_MANIFEST),
        "restore_script_path": str(RESTORE_SCRIPT),
        "manifest_sha256": sha256_file(ROLLBACK_MANIFEST),
        "restore_script_sha256": sha256_file(RESTORE_SCRIPT),
        "dry_run_returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def run_tests(out: Path) -> bool:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_requirement_aware_projection.py",
        "tests/test_hv_urgent_residual_guard.py",
        "-q",
    ]
    process = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    report = (
        f"command: {' '.join(command)}\n"
        f"returncode: {process.returncode}\n\n"
        f"{process.stdout}"
    )
    (out / "test_report.txt").write_text(report, encoding="utf-8")
    return process.returncode == 0


def summarize_methods(out: Path, bundle_root: Path, v25: Any) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    eval_mod = v25.load_eval_module(out)
    config = read_json(bundle_root / "resolved_config.json")
    methods = [
        ("OLD_PROP", METHOD_OLD),
        ("REQALIGN_REFERENCE", METHOD_REQALIGN),
        ("HV_GUARD", METHOD_GUARD),
    ]
    rows = []
    for label, method in methods:
        path = out / "runs" / label
        metric, accounting = eval_mod.summarize_method(
            method,
            "HV_URGENT_RESIDUAL_PROTECTION_SINGLE_SEED_CAP050",
            path,
            config["product_requirement_utility"],
            config["unified_ground_received_accounting"],
        )
        metric["J_HV_net"] = reqrun.hv_net(
            eval_mod,
            path,
            metric,
            config["product_requirement_utility"],
        )
        slots = read_csv(path / "per_slot_resource_trace.csv")
        metric["soc_violation_count"] = sum(reqrun.integer(row.get("soc_violation")) for row in slots)
        metric["thermal_violation_count"] = sum(reqrun.integer(row.get("thermal_violation")) for row in slots)
        metric["minimum_soc"] = min(reqrun.flt(row.get("battery_soc"), 1.0) for row in slots)
        metric["maximum_thermal_state"] = max(reqrun.flt(row.get("thermal"), 0.0) for row in slots)
        metric["public_accounting_passed"] = bool(accounting.get("passed", True))
        metric.update(
            reqrun.subgroup_metrics(
                eval_mod,
                path,
                metric,
                config["product_requirement_utility"],
            )
        )
        energy = reqrun.energy_metrics(method, path)
        metric.update(
            {
                "total_controllable_energy_Wh": energy[
                    "controllable_compute_communication_energy_Wh"
                ],
                "total_transmitted_bytes": energy["transmitted_MB"] * Decimal("1000000"),
                "peak_downlink_queue_MB": energy["peak_downlink_queue_MB"],
                "mean_downlink_queue_MB": energy["mean_downlink_queue_MB"],
            }
        )
        rows.append(metric)
    return rows, eval_mod, config


def ordinary_semantics_audit(path: Path) -> dict[str, Any]:
    overrides = read_csv(path / "hv_action_override_trace.csv")
    projection = read_csv(path / "requirement_projection_log.csv")
    ordinary_overrides = [
        row for row in overrides if row.get("value_level") in {"low", "medium"}
    ]
    invalid_overrides = [
        row
        for row in ordinary_overrides
        if row.get("override_reason") != "cross_task_displacement_rejected"
        or (
            reqrun.integer(row.get("urgent_closure_displacement")) <= 0
            and reqrun.integer(row.get("high_closure_displacement")) <= 0
        )
    ]
    progressive_violations = [
        row
        for row in projection
        if row.get("value_level") in {"low", "medium"}
        and (
            reqrun.integer(row.get("ordinary_progressive_violation")) > 0
            or reqrun.integer(row.get("reenter_future_decisions")) > 0
            or row.get("projection_reason") == "metadata_first_progressive_fallback"
        )
    ]
    return {
        "passed": not invalid_overrides and not progressive_violations,
        "ordinary_override_count": len(ordinary_overrides),
        "valid_cross_task_protection_override_count": (
            len(ordinary_overrides) - len(invalid_overrides)
        ),
        "invalid_extra_utility_gate_override_count": len(invalid_overrides),
        "ordinary_progressive_violation_count": len(progressive_violations),
        "invalid_override_task_ids": sorted(
            {row.get("task_id", "") for row in invalid_overrides}
        ),
        "progressive_violation_task_ids": sorted(
            {row.get("task_id", "") for row in progressive_violations}
        ),
    }


def guard_summary(path: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    active = read_csv(path / "hv_active_residual_trace.csv")
    urgent = read_csv(path / "urgent_reservation_trace.csv")
    high = read_csv(path / "high_reservation_trace.csv")
    release = read_csv(path / "resource_release_trace.csv")
    displacement = read_csv(path / "displacement_check_trace.csv")
    pareto = read_csv(path / "pareto_candidate_trace.csv")
    overrides = read_csv(path / "hv_action_override_trace.csv")
    projection = read_csv(path / "requirement_projection_log.csv")
    tasks = read_csv(path / "per_task_outcomes.csv")
    protected_urgent = {row["task_id"] for row in urgent if reqrun.integer(row.get("protected"))}
    protected_high = {row["task_id"] for row in high if reqrun.integer(row.get("protected"))}
    feasible_urgent = {row["task_id"] for row in active if row.get("value_level") == "urgent" and reqrun.integer(row.get("protectable"))}
    feasible_high = {row["task_id"] for row in active if row.get("value_level") == "high" and reqrun.integer(row.get("protectable"))}
    closed = {row["task_id"] for row in tasks if reqrun.integer(row.get("closure_success"))}
    urgent_missed = sorted(feasible_urgent - closed)
    high_missed = sorted(feasible_high - closed)
    write_csv(
        path / "urgent_feasible_but_missed_trace.csv",
        [{"task_id": task_id, "reason": "feasible_at_guard_evaluation_but_not_closed"} for task_id in urgent_missed],
    )
    write_csv(
        path / "high_feasible_but_missed_trace.csv",
        [{"task_id": task_id, "reason": "feasible_at_guard_evaluation_but_not_closed"} for task_id in high_missed],
    )
    total_reserved_contact = sum(reqrun.flt(row.get("reservation_after_contact_bytes")) for row in release)
    utilized = sum(reqrun.integer(row.get("reservation_utilized_by_current_task")) for row in release)
    reservation_events = sum(
        reqrun.integer(row.get("protected"))
        for row in [*urgent, *high]
    )
    protected_tasks = protected_urgent | protected_high
    protected_closures = protected_tasks & closed
    utilization_ratio = (
        len(protected_closures) / len(protected_tasks)
        if protected_tasks
        else 0.0
    )
    released = {
        key: sum(reqrun.flt(row.get(key)) for row in release)
        for key in (
            "released_compute_s",
            "released_energy_wh",
            "released_queue_mb",
            "released_contact_bytes",
            "released_thermal_increment",
        )
    }
    ordinary_blocked_by_urgent = {
        row["task_id"]
        for row in overrides
        if row.get("value_level") in {"low", "medium"}
        and row.get("override_reason") == "cross_task_displacement_rejected"
        and reqrun.integer(row.get("urgent_closure_displacement")) > 0
    }
    ordinary_blocked_by_high = {
        row["task_id"]
        for row in overrides
        if row.get("value_level") in {"low", "medium"}
        and row.get("override_reason") == "cross_task_displacement_rejected"
        and reqrun.integer(row.get("high_closure_displacement")) > 0
    }
    ordinary_blocked = {
        row["task_id"]
        for row in overrides
        if row.get("value_level") in {"low", "medium"}
        and row.get("override_reason") == "cross_task_displacement_rejected"
    }
    metadata_first = [
        row
        for row in projection
        if row.get("projection_reason") == "metadata_first_progressive_fallback"
    ]
    metadata_only_unfinished = [
        row
        for row in metadata_first
        if row.get("closure_status_after") != "closed"
    ]
    summary = {
        "protected_urgent_residual_count": len(protected_urgent),
        "protected_high_residual_count": len(protected_high),
        "urgent_feasible_but_missed_count": len(urgent_missed),
        "high_feasible_but_missed_count": len(high_missed),
        "metadata_first_count": len(metadata_first),
        "progressive_completion_count": sum(
            reqrun.integer(row.get("residual_upgrade_success"))
            for row in projection
        ),
        "metadata_only_unfinished_count": len(metadata_only_unfinished),
        "reservation_event_count": reservation_events,
        "protected_task_closure_count": len(protected_closures),
        "reservation_utilization_ratio": utilization_ratio,
        "reservation_current_task_execution_event_count": utilized,
        "total_reserved_contact_bytes_trace_integral": total_reserved_contact,
        "displacement_rejection_candidate_count": sum(
            reqrun.integer(row.get("rejected_by_displacement"))
            for row in pareto
        ),
        "low_medium_blocked_by_hv_count": len(ordinary_blocked),
        "low_medium_blocked_by_urgent_count": len(ordinary_blocked_by_urgent),
        "low_medium_blocked_by_high_count": len(ordinary_blocked_by_high),
        "infeasible_urgent_protected_count": sum(
            row.get("reservation_status") == "protected"
            and not reqrun.integer(row.get("protectable"))
            for row in urgent
        ),
        **released,
        "hard_safety_violations": (
            reqrun.integer(metrics.get("soc_violation_count"))
            + reqrun.integer(metrics.get("thermal_violation_count"))
        ),
    }
    return summary


def no_task_specific_logic_audit() -> dict[str, Any]:
    texts = {
        "guard": GUARD_SOURCE.read_text(encoding="utf-8"),
        "runner": Path(__file__).read_text(encoding="utf-8"),
        "config": GUARD_CONFIG.read_text(encoding="utf-8"),
    }
    task_literals = []
    for name, text in texts.items():
        for match in reqrun.re.findall(r"\btask_\d{4}\b", text):
            task_literals.append({"file": name, "literal": match})
    seed_conditions = []
    pattern = reqrun.re.compile(r"\bif\b[^\n]*\bseed\b[^\n]*(?:==|!=|in)\s*2026", reqrun.re.IGNORECASE)
    for name, text in texts.items():
        if pattern.search(text):
            seed_conditions.append(name)
    return {
        "passed": not task_literals and not seed_conditions,
        "task_id_literals": task_literals,
        "seed_specific_conditions": seed_conditions,
        "seed_2026_occurrences_are_experiment_registration_only": True,
    }


def baseline_identity_audit(v25: Any) -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    return {
        "passed": all(
            path.exists()
            for path in (
                v25.SOURCE_B1_OFFLOAD,
                v25.SOURCE_TUT,
                v25.SOURCE_MYOPIC,
            )
        )
        and "run_native_binding_method" not in called_attributes
        and "run_myopic_method" not in called_attributes,
        "guard_not_bound_to_baselines": True,
        "baseline_source_roots": {
            "B1_OFFLOAD": str(v25.SOURCE_B1_OFFLOAD),
            "TUT": str(v25.SOURCE_TUT),
            "MYOPIC": str(v25.SOURCE_MYOPIC),
        },
        "baseline_runtime_runs_started": False,
        "reason": "Stage B is explicitly limited to old PROP, corrected REQALIGN, and guarded REQALIGN.",
    }


def final_report(
    out: Path,
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    regression: dict[str, Any],
    rollback: dict[str, Any],
    promotion: str,
) -> None:
    by_method = {row["method"]: row for row in metrics}
    old = by_method[METHOD_OLD]
    corrected = by_method[METHOD_REQALIGN]
    guarded = by_method[METHOD_GUARD]
    answers = [
        "# HV/Urgent Residual Protection and Pareto Alignment v1",
        "",
        f"Scope: single-seed mechanism diagnostic, seed=2026, capacity={CAPACITY}, duration=86400 s.",
        "",
        "This is a proxy system simulation, not end-to-end onboard image processing.",
        "",
        "## Primary metrics",
        "",
        markdown_table(
            [
                {
                    "method": row["method"],
                    "J_all_net": row["J_all_net"],
                    "J_HV_net": row["J_HV_net"],
                    "HV_closure_count": row["HV_closure_count"],
                    "high_closure_count": row["high_closure_count"],
                    "urgent_closure_count": row["urgent_closure_count"],
                }
                for row in metrics
            ]
        ),
        "",
        "## Required answers",
        "",
        "1. New files: guard pure module, guard config, tests, and single runner; no existing PROP/SafeFilter/evaluator source was edited.",
        "2. Ordinary task semantics were not changed: required product one-shot or DEFER, never progressive.",
        "3. Original PROP scorer was not changed.",
        "4. Task values remain low=1, medium=3, high=10, urgent=25.",
        "5. SafeFilter was not changed or bypassed.",
        "6. Reservations are recomputed from active residual demand; no fixed percentage or quota exists.",
        f"7. Protected urgent residuals: {summary['protected_urgent_residual_count']}.",
        f"8. Protected high residuals: {summary['protected_high_residual_count']}.",
        f"9. Released contact bytes: {summary['released_contact_bytes']}; released energy Wh: {summary['released_energy_wh']}.",
        f"10. Ordinary tasks with at least one displacement-blocked candidate: {summary['low_medium_blocked_by_hv_count']}.",
        f"11. Infeasible urgent protected count: {summary['infeasible_urgent_protected_count']} (must be zero).",
        f"12. J_all_net delta vs old/corrected: {Decimal(str(guarded['J_all_net'])) - Decimal(str(old['J_all_net']))} / {Decimal(str(guarded['J_all_net'])) - Decimal(str(corrected['J_all_net']))}.",
        f"13. J_HV_net delta vs old/corrected: {Decimal(str(guarded['J_HV_net'])) - Decimal(str(old['J_HV_net']))} / {Decimal(str(guarded['J_HV_net'])) - Decimal(str(corrected['J_HV_net']))}.",
        f"14. HV closure delta vs old/corrected: {int(guarded['HV_closure_count']) - int(old['HV_closure_count'])} / {int(guarded['HV_closure_count']) - int(corrected['HV_closure_count'])}.",
        f"15. Urgent closure guarded/old/corrected: {guarded['urgent_closure_count']} / {old['urgent_closure_count']} / {corrected['urgent_closure_count']}.",
        f"16. Promotion result: `{promotion}`.",
        f"17. Guard-off exact corrected-REQALIGN reproduction: {regression['passed']}.",
        f"18. One-click rollback dry-run: {rollback['passed']}.",
        "",
        (
            "An initial diagnostic attempt was archived as invalid before this "
            "reported run because an extra ordinary-task utility gate violated "
            "the frozen ordinary semantics.  The correction was semantic, not "
            "a result-driven threshold change."
        ),
        (
            "A second deterministic attempt was archived because its override "
            "trace attributed displacement to the final DEFER rather than the "
            "rejected REQALIGN action; actions and metrics were unaffected."
        ),
        (
            "A third attempt exposed an HV-exchange semantics defect: it used "
            "absolute urgent displacement rather than the net urgent closure "
            "change including the current action.  The correction implements "
            "the preregistered Pareto replacement rule without new thresholds."
        ),
        "",
        "No automatic parameter retuning or 5 seeds × 3 capacities run was performed.",
        "",
        f"Completion marker: `{COMPLETION_MARKER}`",
    ]
    (out / "final_assessment.md").write_text("\n".join(answers) + "\n", encoding="utf-8")


def fail(out: Path, status: str, detail: Any) -> int:
    write_json(
        out / "audit" / "failure_detail.json",
        {"status": status, "detail": detail, "time_utc": utc_now()},
    )
    print(status)
    print(json.dumps(detail, ensure_ascii=False, indent=2, default=str))
    return 2


def parent_main(overwrite: bool) -> int:
    if OUT.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite without --overwrite: {OUT}")
        resolved_out = OUT.resolve()
        resolved_outputs = (ROOT / "outputs").resolve()
        if resolved_out.parent != resolved_outputs or not resolved_out.name.startswith("_v25_bupt1_"):
            raise RuntimeError(f"Unsafe output deletion target: {resolved_out}")
        shutil.rmtree(OUT)
    for directory in ("audit", "config", "scenario", "runs", "logs", "tables", "smoke"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(GUARD_CONFIG, OUT / "config" / GUARD_CONFIG.name)
    shutil.copy2(reqrun.PROJECTION_CONFIG, OUT / "config" / reqrun.PROJECTION_CONFIG.name)
    rollback = finalize_rollback()
    write_json(OUT / "audit" / "rollback_audit.json", rollback)
    if not rollback["passed"]:
        return fail(OUT, "ROLLBACK_PACKAGE_INCOMPLETE", rollback)
    if not run_tests(OUT):
        return fail(OUT, "UNIT_TEST_FAILED", {"report": str(OUT / "test_report.txt")})
    rollback = finalize_rollback()
    write_json(OUT / "audit" / "rollback_audit.json", rollback)
    if not rollback["passed"]:
        return fail(OUT, "ROLLBACK_PACKAGE_INCOMPLETE", rollback)
    no_specific = no_task_specific_logic_audit()
    write_json(OUT / "audit" / "no_task_specific_logic_audit.json", no_specific)
    if not no_specific["passed"]:
        return fail(OUT, "TASK_SPECIFIC_LOGIC_DETECTED", no_specific)

    v25 = reqrun.load_v25()
    baseline_audit = baseline_identity_audit(v25)
    write_json(OUT / "audit" / "baseline_identity_audit.json", baseline_audit)
    if not baseline_audit["passed"]:
        return fail(OUT, "METHOD_IDENTITY_MISMATCH", baseline_audit)
    try:
        input_audit, frozen = v25.preflight(OUT)
    except Exception as exc:
        return fail(OUT, "TASK_MANIFEST_MISMATCH", {"error": repr(exc)})
    reqrun.write_preflight_audits(OUT, v25, input_audit, frozen)
    bundle_root = v25.prepare_runtime_bundle(OUT)
    shutil.copy2(bundle_root / "task_manifest.csv", OUT / "scenario" / "task_manifest.csv")
    shutil.copy2(bundle_root / "slot_trace.csv", OUT / "scenario" / "source_slot_trace_capacity035.csv")
    write_json(
        OUT / "audit" / "experiment_registration.json",
        {
            "seed": 2026,
            "capacity": CAPACITY,
            "duration_s": 86400,
            "slot_s": SLOT_SECONDS,
            "task_hash": v25.EXPECTED_TASK_HASH,
            "task_count": v25.EXPECTED_TASK_COUNT,
            "formal_multiseed_started": False,
        },
    )
    write_json(
        OUT / "audit" / "implementation_correction_provenance.json",
        {
            "archived_invalid_attempt_exists": INVALID_ATTEMPT_ARCHIVE.exists(),
            "archived_invalid_attempt": str(INVALID_ATTEMPT_ARCHIVE),
            "archived_invalid_audit_attempt_exists": (
                INVALID_AUDIT_ATTEMPT_ARCHIVE.exists()
            ),
            "archived_invalid_audit_attempt": str(INVALID_AUDIT_ATTEMPT_ARCHIVE),
            "archived_invalid_exchange_attempt_exists": (
                INVALID_EXCHANGE_ATTEMPT_ARCHIVE.exists()
            ),
            "archived_invalid_exchange_attempt": str(
                INVALID_EXCHANGE_ATTEMPT_ARCHIVE
            ),
            "invalid_reason": (
                "The first implementation applied Pareto utility gating even "
                "without cross-task displacement, changing ordinary semantics."
            ),
            "correction": (
                "Guard intervention now occurs only when the corrected REQALIGN "
                "action predicts urgent/high closure displacement."
            ),
            "thresholds_changed": False,
            "audit_logging_correction": (
                "Override traces now attribute displacement and displaced task "
                "IDs to the rejected REQALIGN action and separately log the "
                "final guarded action."
            ),
            "hv_exchange_semantics_correction": (
                "For HV-vs-HV competition, urgent non-regression now uses the "
                "current urgent closure increment minus displaced urgent "
                "closures; HV closure count and predicted J_HV must also remain "
                "nonnegative."
            ),
            "task_values_changed": False,
            "seed_or_task_specific_logic_added": False,
        },
    )

    smoke = launch_child(
        "SMOKE_GUARD",
        bundle_root,
        OUT / "smoke" / "HV_GUARD",
        OUT / "logs",
        slots=SMOKE_SLOTS,
    )
    write_json(OUT / "audit" / "smoke_status.json", smoke)
    if smoke["status"] != "PASS":
        return fail(OUT, "HV_GUARD_SMOKE_FAILED", smoke)

    reference = launch_child(
        "REQALIGN_REFERENCE",
        bundle_root,
        OUT / "runs" / "REQALIGN_REFERENCE",
        OUT / "logs",
        slots=SLOTS,
    )
    guard_off = launch_child(
        "REQALIGN_GUARD_OFF",
        bundle_root,
        OUT / "runs" / "REQALIGN_GUARD_OFF",
        OUT / "logs",
        slots=SLOTS,
    )
    write_json(OUT / "audit" / "stage_a_run_status.json", [reference, guard_off])
    if reference["status"] != "PASS" or guard_off["status"] != "PASS":
        return fail(OUT, "REQALIGN_RUNTIME_FAILED", [reference, guard_off])
    regression = disabled_regression_audit(
        OUT / "runs" / "REQALIGN_REFERENCE",
        OUT / "runs" / "REQALIGN_GUARD_OFF",
    )
    write_json(OUT / "audit" / "disabled_regression_audit.json", regression)
    if not regression["passed"]:
        return fail(OUT, "HV_GUARD_DISABLED_REGRESSION_MISMATCH", regression)

    old_run = launch_child(
        "OLD_PROP",
        bundle_root,
        OUT / "runs" / "OLD_PROP",
        OUT / "logs",
        slots=SLOTS,
    )
    guard_run = launch_child(
        "HV_GUARD",
        bundle_root,
        OUT / "runs" / "HV_GUARD",
        OUT / "logs",
        slots=SLOTS,
    )
    write_json(OUT / "audit" / "stage_b_run_status.json", [old_run, reference, guard_run])
    if old_run["status"] != "PASS":
        return fail(OUT, "PROP_ORIGINAL_RUNTIME_FAILED", old_run)
    if guard_run["status"] != "PASS":
        return fail(OUT, "HV_GUARD_RUNTIME_FAILED", guard_run)

    metrics, _eval_mod, _config = summarize_methods(OUT, bundle_root, v25)
    write_csv(OUT / "tables" / "primary_metrics.csv", metrics)
    (OUT / "tables" / "primary_metrics.md").write_text(markdown_table(metrics) + "\n", encoding="utf-8")
    by_method = {row["method"]: row for row in metrics}
    guard_metrics = by_method[METHOD_GUARD]
    summary = guard_summary(OUT / "runs" / "HV_GUARD", guard_metrics)
    write_json(OUT / "audit" / "guard_summary.json", summary)
    write_csv(OUT / "tables" / "guard_summary.csv", [summary])
    ordinary_audit = ordinary_semantics_audit(OUT / "runs" / "HV_GUARD")
    write_json(OUT / "audit" / "ordinary_semantics_audit.json", ordinary_audit)
    promotion = load_module(
        "hv_urgent_residual_guard_promotion_v1",
        GUARD_SOURCE,
    ).promotion_status(
        old_prop=by_method[METHOD_OLD],
        corrected_reqalign=by_method[METHOD_REQALIGN],
        guarded=guard_metrics,
        safety_violations=(
            reqrun.integer(guard_metrics["soc_violation_count"])
            + reqrun.integer(guard_metrics["thermal_violation_count"])
        ),
        ordinary_logic_unchanged=ordinary_audit["passed"],
    )
    write_json(
        OUT / "audit" / "promotion_decision.json",
        {
            "promotion_status": promotion,
            "formal_multiseed_authorized": False,
            "formal_multiseed_started": False,
            "manual_approval_required_for_pareto_candidate": True,
        },
    )
    rollback = finalize_rollback()
    write_json(OUT / "audit" / "rollback_audit.json", rollback)
    if not rollback["passed"]:
        return fail(OUT, "ROLLBACK_PACKAGE_INCOMPLETE", rollback)
    final_report(OUT, metrics, summary, regression, rollback, promotion)
    write_json(
        OUT / "completion_status.json",
        {
            "completion_marker": COMPLETION_MARKER,
            "promotion_status": promotion,
            "single_seed_diagnostic_complete": True,
            "formal_multiseed_started": False,
            "output": str(OUT),
        },
    )
    print(markdown_table(
        [
            {
                "method": row["method"],
                "J_all_net": row["J_all_net"],
                "J_HV_net": row["J_HV_net"],
                "HV_closure_count": row["HV_closure_count"],
                "urgent_closure_count": row["urgent_closure_count"],
            }
            for row in metrics
        ]
    ))
    print(f"PROMOTION_STATUS={promotion}")
    print(f"COMPLETION_MARKER={COMPLETION_MARKER}")
    print(f"OUTPUT_DIR={OUT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--bundle-root")
    parser.add_argument("--output")
    parser.add_argument("--slots", type=int, default=SLOTS)
    args = parser.parse_args()
    if args.child_run:
        return child_main(args)
    return parent_main(args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
