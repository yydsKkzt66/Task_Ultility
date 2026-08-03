from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_hv_urgent_residual_protection_v1 as guardrun
import scripts.run_prop_reqalign_seed2026_v1 as reqrun


STRICT_OUT = ROOT / "outputs" / "_v25_bupt1_ordinary_net_granularity_seed2026_cap050_v1"
SURPLUS_OUT = ROOT / "outputs" / "_v25_bupt1_ordinary_net_same_slot_surplus_seed2026_cap050_v1"
OUT = STRICT_OUT
SOURCE = ROOT / "src" / "routers" / "ordinary_net_utility_projection.py"
METHOD_REFERENCE = reqrun.METHOD_PROP_NEW
STRICT_METHOD = "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-ORDINARY-NET-v1"
SURPLUS_METHOD = "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-ORDINARY-NET-SAME-SLOT-v1"
METHOD_NEW = STRICT_METHOD
ENABLE_SAME_SLOT_SURPLUS = False
CAPACITY = 0.50
SLOTS = 4320


def configure_variant(enable_same_slot_surplus: bool) -> None:
    global OUT, METHOD_NEW, ENABLE_SAME_SLOT_SURPLUS
    ENABLE_SAME_SLOT_SURPLUS = bool(enable_same_slot_surplus)
    OUT = SURPLUS_OUT if ENABLE_SAME_SLOT_SURPLUS else STRICT_OUT
    METHOD_NEW = SURPLUS_METHOD if ENABLE_SAME_SLOT_SURPLUS else STRICT_METHOD


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class OrdinaryNetGranularityWrapper:
    """Net-utility granularity projection outside corrected REQALIGN."""

    def __init__(
        self,
        delegate: Any,
        projection_module: Any,
        utility_cfg: dict[str, Any],
        accounting_cfg: dict[str, Any],
        *,
        allow_same_slot_surplus: bool = False,
    ) -> None:
        self.delegate = delegate
        self.projection = projection_module
        self.utility_cfg = utility_cfg
        self.accounting_cfg = accounting_cfg
        self.allow_same_slot_surplus = bool(allow_same_slot_surplus)
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
        self.selection_rows: list[dict[str, Any]] = []
        self.candidate_rows: list[dict[str, Any]] = []
        self.active_same_slot_override_task_id = ""
        self.active_same_slot_override_product = ""

    def bind_step_runner(self, step_runner: Any) -> None:
        self.step_runner = step_runner

    def bind_step(self, step: Any, resource_state: Any, comm_state: Any) -> None:
        self.current_step = step
        self.current_resource_state = resource_state
        self.current_comm_state = comm_state

    @staticmethod
    def _level(task: Any) -> str:
        return str(
            getattr(task, "value_level", "")
            or (getattr(task, "metadata", {}) or {}).get("value_level", "")
        ).lower()

    @staticmethod
    def _is_defer(action: Any, product: str) -> bool:
        return (
            str(action.action_name).lower() in {"delay/store", "defer", "delay"}
            or str(product).lower() in {"", "none", "defer"}
        )

    def _pending_high_value_payload_exists(self) -> bool:
        active_status = {"pending", "enqueued", "blocked"}
        for request in getattr(
            self.step_runner, "staged_delivery_requests", {}
        ).values():
            if (
                str(getattr(request, "status", "")) in active_status
                and str(getattr(request, "value_level", "")).lower()
                in {"high", "urgent"}
            ):
                return True
        now_s = float(self.current_step.slot_index) * float(
            self.current_step.slot_seconds
        )
        for task_id, state in getattr(
            self.step_runner, "product_lifecycle", {}
        ).items():
            task = self.step_runner.task_snapshots.get(task_id)
            if task is None or self._level(task) not in {"high", "urgent"}:
                continue
            deadline_s = (
                float(task.arrival_slot) * float(self.current_step.slot_seconds)
                + float(task.deadline_s)
            )
            if (
                bool(getattr(state, "task_closed_flag", False))
                or deadline_s < now_s - 1.0e-9
            ):
                continue
            if (
                bool(set(getattr(state, "pending_products", set()) or set()))
                or float(
                    getattr(state, "residual_utility_remaining", 0.0) or 0.0
                )
                > 1.0e-12
            ):
                return True
        return False

    def _payload_upper_bound_mb(self, action: Any, predicted: Any) -> float:
        from src.payload.payload_policy import get_payload_policy_profile
        from src.payload.payload_sizer import resolve_payload_plan

        product, _residual = resolve_payload_plan(action.payload_policy)
        profile = get_payload_policy_profile(
            product, self.step_runner.payload_profiles_cfg
        )
        parameters = dict(getattr(profile, "parameters", {}) or {})
        return max(
            float(predicted.predicted_S_gen_mb),
            float(
                parameters.get(
                    "max_payload_mb", predicted.predicted_S_gen_mb
                )
            ),
        )

    def _candidate(
        self,
        action: Any,
        predicted: Any,
        task: Any,
        delivered_before: set[str],
        eligible_names: set[str],
        current_time_s: float,
        deadline_s: float,
    ) -> tuple[Any, dict[str, Any]]:
        from src.data.value_skew_calibration import task_effective_value
        from src.payload.information_sufficiency import delta_information_utility

        product = str(predicted.predicted_product_type or "")
        is_defer = self._is_defer(action, product)
        if is_defer:
            arrival = None
            ground_arrival_s = None
            deadline_feasible = True
            timeliness = 0.0
        else:
            arrival = self.delegate._arrival(
                action, predicted, task, current_time_s, deadline_s
            )
            ground_arrival_s = arrival.ground_arrival_s
            deadline_feasible = (
                ground_arrival_s is not None
                and float(ground_arrival_s) <= deadline_s + 1.0e-9
            )
            capture_s = float(task.arrival_slot) * float(self.current_step.slot_seconds)
            latency_s = (
                math.inf
                if ground_arrival_s is None
                else max(0.0, float(ground_arrival_s) - capture_s)
            )
            timeliness = (
                0.0
                if not math.isfinite(latency_s) or float(task.deadline_s) <= 1.0e-9
                else max(0.0, min(1.0, 1.0 - latency_s / float(task.deadline_s)))
            )
        safety_reasons = tuple(
            self.step_runner.safe_filter.evaluate_action(
                task, action, self._predicted_map
            )
        )
        delta_info = delta_information_utility(
            product,
            delivered_before,
            str(task.product_requirement),
            self.utility_cfg,
        )
        gross = (
            0.0
            if is_defer or not deadline_feasible
            else float(task_effective_value(task))
            * max(0.0, float(predicted.predicted_detection_quality))
            * max(0.0, float(delta_info))
            * timeliness
        )
        total_energy_j = max(
            0.0,
            float(predicted.predicted_E_comp)
            + float(predicted.predicted_E_comm),
        )
        energy_normalizer_wh = float(
            self.accounting_cfg.get("energy_normalization", {}).get(
                "controllable_mission_energy_wh", 10.0
            )
        )
        energy_cost = (
            0.0
            if is_defer
            else total_energy_j / 3600.0 / max(energy_normalizer_wh, 1.0e-12)
        )
        net = gross - energy_cost
        from src.payload.payload_sizer import estimate_payload_encoding_cost

        encoding_latency_s, _encoding_energy_j = estimate_payload_encoding_cost(
            action.payload_policy,
            max(0.0, float(predicted.predicted_S_gen_mb)),
            self.step_runner.payload_profiles_cfg,
        )
        slot_end_s = current_time_s + float(self.current_step.slot_seconds)
        ready_s = (
            current_time_s
            + max(0.0, float(predicted.predicted_L_inf))
            + max(0.0, float(encoding_latency_s))
        )
        rate_mb_s = (
            float(self.current_step.link_trace.throughput_mbps)
            * float(
                self.delegate.simulator_cfg.get("downlink_efficiency", 1.0)
            )
            / 8.0
            if bool(self.current_step.link_trace.in_contact)
            else 0.0
        )
        remaining_same_slot_capacity_mb = max(
            0.0, slot_end_s - ready_s
        ) * max(0.0, rate_mb_s)
        payload_upper_bound_mb = self._payload_upper_bound_mb(
            action, predicted
        )
        fact = self.projection.OrdinaryNetCandidate(
            action=action,
            action_name=str(action.action_name),
            model_name=str(action.model_name),
            product=product,
            predicted_net_utility=net,
            predicted_gross_utility=gross,
            predicted_energy_cost=energy_cost,
            payload_mb=max(0.0, float(predicted.predicted_S_gen_mb)),
            total_energy_j=total_energy_j,
            compute_latency_s=max(0.0, float(predicted.predicted_L_inf)),
            predicted_ground_arrival_s=(
                None if ground_arrival_s is None else float(ground_arrival_s)
            ),
            hard_safe=not safety_reasons,
            deadline_feasible=deadline_feasible,
            eligible_by_existing_filters=action.action_name in eligible_names,
            is_defer=is_defer,
        )
        details = {
            "predicted_delta_information_utility": delta_info,
            "predicted_timeliness": timeliness,
            "safety_reasons": ";".join(safety_reasons),
            "contact_found": int(bool(arrival and arrival.contact_found)),
            "encoding_latency_s": encoding_latency_s,
            "ready_time_s": ready_s,
            "slot_end_s": slot_end_s,
            "remaining_same_slot_capacity_mb": (
                remaining_same_slot_capacity_mb
            ),
            "conservative_payload_upper_bound_mb": payload_upper_bound_mb,
            "candidate_is_staged_followup": int(
                "_then_" in str(action.payload_policy).lower()
            ),
        }
        return fact, details

    def select_action(
        self,
        context: Any,
        candidate_actions: list[Any],
        predicted_map: dict[str, Any],
        guidance_bundle=None,
    ) -> Any:
        reference = self.delegate.select_action(
            context, candidate_actions, predicted_map, guidance_bundle
        )
        self.last_all_score_components = deepcopy(
            self.delegate.last_all_score_components
        )
        self.last_score_components = deepcopy(self.delegate.last_score_components)
        self.last_selected_action_score = self.delegate.last_selected_action_score
        self._predicted_map = predicted_map
        task = self.current_step.task
        if task is None or self.step_runner is None:
            return reference
        self.active_same_slot_override_task_id = ""
        self.active_same_slot_override_product = ""
        level = self._level(task)
        if level in {"high", "urgent"}:
            self.selection_rows.append(
                {
                    "slot": int(self.current_step.slot_index),
                    "task_id": str(task.task_id),
                    "value_level": level,
                    "REQALIGN_action": reference.action_name,
                    "final_action": reference.action_name,
                    "changed": 0,
                    "decision_reason": "high_value_bypass",
                }
            )
            return reference

        current_time_s = float(self.current_step.slot_index) * float(
            self.current_step.slot_seconds
        )
        deadline_s = (
            float(task.arrival_slot) * float(self.current_step.slot_seconds)
            + float(task.deadline_s)
        )
        lifecycle = self.step_runner.product_lifecycle.get(task.task_id)
        delivered_before = (
            set() if lifecycle is None else set(lifecycle.delivered_products)
        )
        eligible_names = {action.action_name for action in candidate_actions}
        facts = []
        details_by_name = {}
        for action in self.step_runner.action_space.list_actions():
            fact, details = self._candidate(
                action,
                predicted_map[action.action_name],
                task,
                delivered_before,
                eligible_names,
                current_time_s,
                deadline_s,
            )
            facts.append(fact)
            details_by_name[fact.action_name] = details
        reference_fact = next(
            row for row in facts if row.action_name == reference.action_name
        )
        decision = self.projection.choose_ordinary_net_utility_action(
            value_level=level,
            reference_action=reference,
            candidates=facts,
            require_positive_net=True,
            require_resource_nonincrease=True,
        )
        certificate_by_name: dict[str, Any] = {}
        if (
            self.allow_same_slot_surplus
            and reference_fact.is_defer
            and not decision.changed
        ):
            projection_row = self.delegate.rows[-1]
            original_model = str(projection_row.get("original_model", ""))
            pending_hv = self._pending_high_value_payload_exists()
            for fact in facts:
                detail = details_by_name[fact.action_name]
                certificate_by_name[fact.action_name] = (
                    self.projection.same_slot_surplus_certificate(
                        self.projection.SameSlotSurplusFacts(
                            value_level=level,
                            reference_is_defer=True,
                            current_in_contact=bool(
                                self.current_step.link_trace.in_contact
                            ),
                            current_queue_mb=float(
                                self.current_comm_state.queue_mb
                            ),
                            pending_high_value_payload=pending_hv,
                            candidate_hard_safe=fact.hard_safe,
                            candidate_deadline_feasible=(
                                fact.deadline_feasible
                            ),
                            candidate_eligible_by_existing_filters=(
                                fact.eligible_by_existing_filters
                            ),
                            candidate_is_staged_followup=bool(
                                detail["candidate_is_staged_followup"]
                            ),
                            candidate_predicted_net_utility=(
                                fact.predicted_net_utility
                            ),
                            predicted_ground_arrival_s=(
                                fact.predicted_ground_arrival_s
                            ),
                            slot_end_s=float(detail["slot_end_s"]),
                            absolute_deadline_s=deadline_s,
                            conservative_payload_upper_bound_mb=float(
                                detail[
                                    "conservative_payload_upper_bound_mb"
                                ]
                            ),
                            remaining_same_slot_capacity_mb=float(
                                detail["remaining_same_slot_capacity_mb"]
                            ),
                            candidate_model_matches_original_selector=(
                                bool(original_model)
                                and fact.model_name == original_model
                            ),
                        )
                    )
                )
            surplus = [
                fact
                for fact in facts
                if certificate_by_name[fact.action_name].eligible
            ]
            if surplus:
                selected_surplus = sorted(
                    surplus,
                    key=lambda fact: (
                        -float(fact.predicted_net_utility),
                        float(fact.payload_mb),
                        float(fact.total_energy_j),
                        str(fact.action_name),
                    ),
                )[0]
                decision = self.projection.OrdinaryNetDecision(
                    final_action=selected_surplus.action,
                    reference_action=reference,
                    reason="same_slot_surplus_projected",
                    changed=True,
                    selected=selected_surplus,
                    unconstrained_winner=decision.unconstrained_winner,
                    blocked_resource_increasing_winner=False,
                )
        final = decision.final_action
        if decision.reason == "same_slot_surplus_projected":
            self.active_same_slot_override_task_id = str(task.task_id)
            self.active_same_slot_override_product = str(
                decision.selected.product
            )
        scores = {
            name: float(components.get("final_action_score", -math.inf))
            for name, components in self.last_all_score_components.items()
        }
        if decision.changed:
            self.last_selected_action_score = scores.get(
                final.action_name,
                float(predicted_map[final.action_name].predicted_reward),
            )
            self.last_score_components = deepcopy(
                self.last_all_score_components.get(final.action_name, {})
            )
            self.delegate.last_selected_action_score = self.last_selected_action_score
            self.delegate.last_score_components = deepcopy(self.last_score_components)

        selected = decision.selected
        self.selection_rows.append(
            {
                "slot": int(self.current_step.slot_index),
                "task_id": str(task.task_id),
                "value_level": level,
                "required_product": str(task.product_requirement),
                "REQALIGN_action": reference.action_name,
                "REQALIGN_product": reference_fact.product,
                "REQALIGN_predicted_net": reference_fact.predicted_net_utility,
                "final_action": final.action_name,
                "final_product": selected.product,
                "final_predicted_net": selected.predicted_net_utility,
                "unconstrained_winner": decision.unconstrained_winner.action_name,
                "unconstrained_winner_net": decision.unconstrained_winner.predicted_net_utility,
                "changed": int(decision.changed),
                "decision_reason": decision.reason,
                "blocked_resource_increasing_winner": int(
                    decision.blocked_resource_increasing_winner
                ),
                "payload_delta_mb": selected.payload_mb - reference_fact.payload_mb,
                "energy_delta_j": selected.total_energy_j - reference_fact.total_energy_j,
                "compute_latency_delta_s": (
                    selected.compute_latency_s - reference_fact.compute_latency_s
                ),
                "same_slot_surplus_enabled": int(
                    self.allow_same_slot_surplus
                ),
                "same_slot_surplus_certified": int(
                    decision.reason == "same_slot_surplus_projected"
                ),
            }
        )
        for fact in facts:
            detail = details_by_name[fact.action_name]
            self.candidate_rows.append(
                {
                    "slot": int(self.current_step.slot_index),
                    "task_id": str(task.task_id),
                    "value_level": level,
                    "required_product": str(task.product_requirement),
                    "action_name": fact.action_name,
                    "model_name": fact.model_name,
                    "product": fact.product,
                    "predicted_net_utility": fact.predicted_net_utility,
                    "predicted_gross_utility": fact.predicted_gross_utility,
                    "predicted_energy_cost": fact.predicted_energy_cost,
                    "payload_mb": fact.payload_mb,
                    "total_energy_j": fact.total_energy_j,
                    "compute_latency_s": fact.compute_latency_s,
                    "predicted_ground_arrival_s": fact.predicted_ground_arrival_s,
                    "hard_safe": int(fact.hard_safe),
                    "deadline_feasible": int(fact.deadline_feasible),
                    "eligible_by_existing_filters": int(
                        fact.eligible_by_existing_filters
                    ),
                    "resource_nonincreasing": int(
                        self.projection.resource_nonincreasing(fact, reference_fact)
                    ),
                    "selected": int(fact.action_name == final.action_name),
                    "same_slot_surplus_eligible": int(
                        bool(
                            certificate_by_name.get(fact.action_name)
                            and certificate_by_name[fact.action_name].eligible
                        )
                    ),
                    "same_slot_surplus_reason": (
                        certificate_by_name[fact.action_name].reason
                        if fact.action_name in certificate_by_name
                        else "not_evaluated"
                    ),
                    **detail,
                }
            )
        return final

    def finish_step(self, step: Any) -> None:
        """Prevent a same-slot ordinary partial product from using future slots."""

        task_id = self.active_same_slot_override_task_id
        if not task_id or step.task is None or str(step.task.task_id) != task_id:
            return
        state = self.step_runner.product_lifecycle.get(task_id)
        pending_before = 0
        if state is not None:
            pending_before = len(
                set(getattr(state, "pending_products", set()) or set())
            )
            state.pending_products.clear()
            state.virtual_continue_request_flag = False
            state.pending_residual_registered_slot = None
            state.residual_parent_payload_plan = ""
            state.residual_request_time_s = None
            state.residual_enqueue_time_s = None
        removed_staged = 0
        for key, request in list(
            getattr(self.step_runner, "staged_delivery_requests", {}).items()
        ):
            if str(getattr(request, "task_id", "")) == task_id:
                self.step_runner.staged_delivery_requests.pop(key, None)
                removed_staged += 1
        removed_candidates = 0
        for attr in ("residual_candidate_pool", "residual_candidate_history"):
            container = getattr(self.step_runner, attr, {})
            for key, candidate in list(container.items()):
                if str(getattr(candidate, "task_id", "")) == task_id:
                    container.pop(key, None)
                    removed_candidates += 1
        superseded_lifecycle_records = 0
        ledger = getattr(self.step_runner, "lifecycle_ledger", None)
        if ledger is not None:
            for record in list(getattr(ledger, "records", {}).values()):
                if (
                    str(getattr(record, "task_id", "")) == task_id
                    and str(getattr(record, "product_type", "")).lower()
                    in {"roi", "roi_context"}
                    and str(getattr(record, "current_status", "")).lower()
                    in {"created", "pending", "queued", "scheduled", "blocked"}
                ):
                    ledger.transition(
                        str(record.lifecycle_id),
                        "superseded",
                        slot=int(step.slot_index),
                        reason="ordinary_same_slot_partial_no_followup",
                        blocker_reason="ordinary_same_slot_partial_no_followup",
                    )
                    superseded_lifecycle_records += 1
        queue = self.step_runner.downlink_queue
        queue_before = len(queue.items)
        queue.items = [
            item
            for item in queue.items
            if str(getattr(item, "task_id", "")) != task_id
        ]
        removed_queue_items = queue_before - len(queue.items)
        for row in reversed(self.selection_rows):
            if (
                row.get("task_id") == task_id
                and row.get("decision_reason")
                == "same_slot_surplus_projected"
            ):
                row["ordinary_followup_suppression_applied"] = 1
                row["suppressed_pending_product_count"] = pending_before
                row["suppressed_staged_request_count"] = removed_staged
                row["suppressed_residual_candidate_count"] = (
                    removed_candidates
                )
                row["superseded_lifecycle_record_count"] = (
                    superseded_lifecycle_records
                )
                row["removed_future_queue_item_count"] = removed_queue_items
                break

    def update(self, context: Any, action: Any, observed_reward: float) -> None:
        self.delegate.update(context, action, observed_reward)


def bind_runtime(runner: Any, reqalign: Any, wrapper: Any) -> None:
    original_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = original_factory(queue)
        reqalign.bind_step_runner(step_runner)
        wrapper.bind_step_runner(step_runner)
        original_run_step = step_runner.run_step
        original_add_residual_candidate = (
            step_runner._add_residual_candidate_after_lifecycle
        )
        original_create_staged_request = (
            step_runner._maybe_create_staged_delivery_request_after_delivery
        )

        def guarded_create_staged_request(*args, **kwargs):
            task = kwargs.get("task")
            if task is None and len(args) >= 3:
                task = args[2]
            task_id = str(getattr(task, "task_id", ""))
            if (
                wrapper.allow_same_slot_surplus
                and task_id
                and task_id == wrapper.active_same_slot_override_task_id
            ):
                for row in reversed(wrapper.selection_rows):
                    if (
                        row.get("task_id") == task_id
                        and row.get("decision_reason")
                        == "same_slot_surplus_projected"
                    ):
                        row["suppressed_staged_creation_count"] = (
                            int(row.get("suppressed_staged_creation_count", 0))
                            + 1
                        )
                        break
                return None
            return original_create_staged_request(*args, **kwargs)

        step_runner._maybe_create_staged_delivery_request_after_delivery = (
            guarded_create_staged_request
        )

        def guarded_add_residual_candidate(*args, **kwargs):
            lifecycle_state = kwargs.get("lifecycle_state")
            if lifecycle_state is None and len(args) >= 2:
                lifecycle_state = args[1]
            task_id = str(getattr(lifecycle_state, "task_id", ""))
            if (
                wrapper.allow_same_slot_surplus
                and task_id
                == wrapper.active_same_slot_override_task_id
            ):
                return ""
            return original_add_residual_candidate(*args, **kwargs)

        step_runner._add_residual_candidate_after_lifecycle = (
            guarded_add_residual_candidate
        )

        def intercepted(
            timeline_step, resource_state, mission_state, comm_state, router
        ):
            effective_step = reqalign.prepare_step_for_reentry(
                timeline_step, resource_state, comm_state
            )
            reqalign.bind_step(effective_step, resource_state, comm_state)
            wrapper.bind_step(effective_step, resource_state, comm_state)
            result = original_run_step(
                effective_step,
                resource_state,
                mission_state,
                comm_state,
                router,
            )
            reqalign.finish_step(effective_step)
            wrapper.finish_step(effective_step)
            return result

        step_runner.run_step = intercepted
        return step_runner

    runner.step_runner_factory = factory


def run_new(bundle_root: Path, output: Path) -> None:
    v25 = reqrun.load_v25()
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    projection_module = load_module(
        "ordinary_net_utility_projection_runtime_v1", SOURCE
    )
    prop_mod = v25.import_prop_script()
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(
        bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep
    )
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configs = prop_mod._configure(
        scenario.configs, calibration, {"enabled": False}
    )
    scenario = guardrun.replace(scenario, configs=configs)
    scenario, power_identity = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, CAPACITY)
    scenario_identity = v25.relaxed_scenario_identity(
        METHOD_NEW, scenario, bundle, power_identity
    )
    if (
        not scenario_identity["task_hash_match"]
        or scenario_identity["runtime_value_mismatch_count"]
    ):
        raise RuntimeError({"scenario_identity_failed": scenario_identity})
    env = SimulationEnvironment(scenario.configs)
    runner = env.build_episode_runner(scenario.timeline[:SLOTS])
    base_router = build_routers(
        scenario.configs["experiment"],
        ActionSpace(scenario.configs["action_space"]),
    )[0]
    reqalign = reqrun.RequirementAwareRouterWrapper(
        base_router,
        reqrun.load_projection_config(True),
        scenario.timeline[:SLOTS],
        scenario.configs["simulator"],
        reqrun.load_projection_module(),
    )
    wrapper = OrdinaryNetGranularityWrapper(
        reqalign,
        projection_module,
        scenario.configs["product_requirement_utility"],
        scenario.configs["unified_ground_received_accounting"],
        allow_same_slot_surplus=ENABLE_SAME_SLOT_SURPLUS,
    )
    bind_runtime(runner, reqalign, wrapper)
    result = runner.run(wrapper)
    normalized = normalize_episode(METHOD_NEW, scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(
        output / "candidate_action_log.csv",
        list(getattr(runner, "last_candidate_action_logs", [])),
    )
    reqrun.write_energy_components(output, result)
    reqrun.write_csv(
        output / "requirement_projection_log.csv",
        reqrun.enrich_projection_rows(reqalign, output),
    )
    reqrun.write_csv(
        output / "ordinary_net_selection_trace.csv", wrapper.selection_rows
    )
    reqrun.write_csv(
        output / "ordinary_net_candidate_trace.csv", wrapper.candidate_rows
    )
    identity = {
        "method": METHOD_NEW,
        "parent_method": METHOD_REFERENCE,
        "capacity": CAPACITY,
        "seed": 2026,
        "ordinary_levels_only": ["low", "medium"],
        "high_urgent_hard_bypass": True,
        "same_model_required": True,
        "resource_nonincrease_required": (
            "unless_same_slot_surplus_certified"
            if ENABLE_SAME_SLOT_SURPLUS
            else True
        ),
        "same_slot_surplus_enabled": ENABLE_SAME_SLOT_SURPLUS,
        "same_slot_surplus_rule": (
            "ordinary_defer_only_current_contact_empty_queue_no_pending_hv_same_slot_complete_positive_net"
            if ENABLE_SAME_SLOT_SURPLUS
            else "disabled"
        ),
        "ordinary_same_slot_followup_suppressed": (
            ENABLE_SAME_SLOT_SURPLUS
        ),
        "safe_filter_preserved": True,
        "corrected_reqalign_preserved": True,
        "original_scorer_preserved": True,
        "original_selector_preserved": True,
        "public_predicted_net_utility": True,
        "future_ground_truth_used": False,
        "projection_source": str(SOURCE),
        "projection_source_sha256": reqrun.sha256_file(SOURCE),
        "capacity_identity": capacity_identity,
        "scenario_identity": scenario_identity,
    }
    guardrun.write_json(output / "native_identity.json", identity)
    guardrun.write_json(
        output / "method_run_status.json",
        {
            "status": "PASS",
            "method": METHOD_NEW,
            "simulation_started": True,
            "fresh_state": True,
            "reused_action_trace": False,
            "reused_queue_trace": False,
            "reused_delivery_ledger": False,
            "identity": identity,
        },
    )
    guardrun.write_json(
        output / "method_completed.json", {"status": "PASS", "method": METHOD_NEW}
    )


def launch(label: str, bundle_root: Path, output: Path) -> dict[str, Any]:
    if label == "REFERENCE":
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_hv_urgent_residual_protection_v1.py"),
            "--child-run",
            "--label",
            "REQALIGN_REFERENCE",
            "--bundle-root",
            str(bundle_root),
            "--output",
            str(output),
            "--slots",
            str(SLOTS),
        ]
    else:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-run",
            "--bundle-root",
            str(bundle_root),
            "--output",
            str(output),
        ]
        if ENABLE_SAME_SLOT_SURPLUS:
            command.append("--same-slot-surplus")
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
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / f"{label}_stdout.txt").write_text(
        process.stdout, encoding="utf-8"
    )
    (OUT / "logs" / f"{label}_stderr.txt").write_text(
        process.stderr, encoding="utf-8"
    )
    return {
        "label": label,
        "status": (
            "PASS"
            if process.returncode == 0
            and (output / "method_completed.json").exists()
            else "FAIL"
        ),
        "returncode": process.returncode,
        "started_utc": started,
        "finished_utc": utc_now(),
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-6000:],
    }


def summarize(out: Path, bundle_root: Path) -> list[dict[str, Any]]:
    v25 = reqrun.load_v25()
    eval_mod = v25.load_eval_module(out)
    config = reqrun.read_json(bundle_root / "resolved_config.json")
    rows = []
    for method, path in (
        (METHOD_REFERENCE, out / "runs" / "REFERENCE"),
        (METHOD_NEW, out / "runs" / "ORDINARY_NET"),
    ):
        metric, accounting = eval_mod.summarize_method(
            method,
            "ORDINARY_NET_GRANULARITY_CAP050_SINGLE_SEED",
            path,
            config["product_requirement_utility"],
            config["unified_ground_received_accounting"],
        )
        metric["J_HV_net"] = reqrun.hv_net(
            eval_mod, path, metric, config["product_requirement_utility"]
        )
        metric.update(
            reqrun.subgroup_metrics(
                eval_mod,
                path,
                metric,
                config["product_requirement_utility"],
            )
        )
        metric.update(reqrun.energy_metrics(method, path))
        slots = reqrun.read_csv(path / "per_slot_resource_trace.csv")
        metric["minimum_soc"] = min(
            reqrun.flt(row.get("battery_soc"), 1.0) for row in slots
        )
        metric["maximum_thermal_state"] = max(
            reqrun.flt(row.get("thermal"), 0.0) for row in slots
        )
        metric["public_accounting_passed"] = bool(accounting.get("passed", True))
        rows.append(metric)
    return rows


def table_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "J_all_net",
        "J_HV_net",
        "HV_closure_count",
        "high_closure_count",
        "urgent_closure_count",
        "ordinary_closure_count",
        "closure_count",
        "gross_delivered_utility",
        "public_method_cost",
        "controllable_compute_communication_energy_Wh",
        "transmitted_MB",
        "peak_downlink_queue_MB",
        "mean_downlink_queue_MB",
        "minimum_soc",
        "maximum_thermal_state",
        "soc_violation_count",
        "thermal_violation_count",
        "successful_closure_mean_latency_s",
        "HV_successful_closure_mean_latency_s",
        "urgent_successful_closure_mean_latency_s",
        "public_accounting_passed",
    )
    return [
        {"method": row["method"], **{key: row.get(key, "") for key in keys}}
        for row in metrics
    ]


def paired_closure_changes(reference: Path, new: Path) -> list[dict[str, Any]]:
    old = {
        row["task_id"]: row
        for row in reqrun.read_csv(reference / "per_task_outcomes.csv")
    }
    current = {
        row["task_id"]: row
        for row in reqrun.read_csv(new / "per_task_outcomes.csv")
    }
    output = []
    for task_id in sorted(old):
        before = old[task_id]
        after = current[task_id]
        old_closed = reqrun.integer(before.get("closure_success"))
        new_closed = reqrun.integer(after.get("closure_success"))
        if old_closed == new_closed:
            continue
        output.append(
            {
                "task_id": task_id,
                "value_level": before.get("value_level", ""),
                "required_product": before.get("product_requirement", ""),
                "reference_closure": old_closed,
                "ordinary_net_closure": new_closed,
                "change": "gained" if new_closed > old_closed else "lost",
            }
        )
    return output


def parent_main(overwrite: bool) -> int:
    if OUT.exists():
        if not overwrite:
            raise FileExistsError(OUT)
        resolved = OUT.resolve()
        if (
            resolved.parent != (ROOT / "outputs").resolve()
            or resolved.name != OUT.name
        ):
            raise RuntimeError(f"unsafe output deletion target: {resolved}")
        shutil.rmtree(OUT)
    for name in ("audit", "config", "logs", "runs", "scenario", "tables"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_ordinary_net_utility_projection.py",
            "tests/test_requirement_aware_projection.py",
            "-q",
        ],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (OUT / "test_report.txt").write_text(tests.stdout, encoding="utf-8")
    if tests.returncode:
        return 2
    v25 = reqrun.load_v25()
    input_audit, frozen = v25.preflight(OUT)
    reqrun.write_preflight_audits(OUT, v25, input_audit, frozen)
    bundle_root = v25.prepare_runtime_bundle(OUT)
    statuses = [
        launch("REFERENCE", bundle_root, OUT / "runs" / "REFERENCE"),
        launch("ORDINARY_NET", bundle_root, OUT / "runs" / "ORDINARY_NET"),
    ]
    guardrun.write_json(OUT / "audit" / "run_status.json", statuses)
    if not all(row["status"] == "PASS" for row in statuses):
        return 2
    metrics = summarize(OUT, bundle_root)
    table = table_rows(metrics)
    reqrun.write_csv(OUT / "tables" / "primary_and_auxiliary_metrics.csv", table)
    (OUT / "tables" / "primary_and_auxiliary_metrics.md").write_text(
        guardrun.markdown_table(table) + "\n", encoding="utf-8"
    )
    by_method = {row["method"]: row for row in metrics}
    before = by_method[METHOD_REFERENCE]
    after = by_method[METHOD_NEW]
    deltas = [
        {
            "metric": key,
            "reference": before.get(key, ""),
            "ordinary_net": after.get(key, ""),
            "delta": (
                reqrun.dec(after[key]) - reqrun.dec(before[key])
                if key in after and key in before
                else ""
            ),
        }
        for key in (
            "J_all_net",
            "J_HV_net",
            "HV_closure_count",
            "high_closure_count",
            "urgent_closure_count",
            "ordinary_closure_count",
            "closure_count",
            "gross_delivered_utility",
            "public_method_cost",
            "transmitted_MB",
        )
    ]
    reqrun.write_csv(OUT / "tables" / "old_new_deltas.csv", deltas)
    pairs = paired_closure_changes(
        OUT / "runs" / "REFERENCE", OUT / "runs" / "ORDINARY_NET"
    )
    reqrun.write_csv(OUT / "tables" / "paired_closure_changes.csv", pairs)
    selections = reqrun.read_csv(
        OUT / "runs" / "ORDINARY_NET" / "ordinary_net_selection_trace.csv"
    )
    ordinary = [
        row for row in selections if row.get("value_level") in {"low", "medium"}
    ]
    hv = [
        row for row in selections if row.get("value_level") in {"high", "urgent"}
    ]
    mechanism = {
        "ordinary_decision_count": len(ordinary),
        "ordinary_changed_count": sum(
            reqrun.integer(row.get("changed")) for row in ordinary
        ),
        "resource_increasing_winner_blocked_count": sum(
            reqrun.integer(row.get("blocked_resource_increasing_winner"))
            for row in ordinary
        ),
        "high_urgent_decision_count": len(hv),
        "high_urgent_non_bypass_count": sum(
            row.get("decision_reason") != "high_value_bypass" for row in hv
        ),
        "high_urgent_action_changed_by_new_module": sum(
            reqrun.integer(row.get("changed")) for row in hv
        ),
        "ordinary_payload_delta_mb_sum": sum(
            reqrun.flt(row.get("payload_delta_mb"), 0.0) for row in ordinary
        ),
        "ordinary_energy_delta_j_sum": sum(
            reqrun.flt(row.get("energy_delta_j"), 0.0) for row in ordinary
        ),
    }
    reference_selected = {
        (row.get("slot", ""), row.get("task_id", "")): row.get(
            "candidate_action", ""
        )
        for row in reqrun.read_csv(
            OUT / "runs" / "REFERENCE" / "candidate_action_log.csv"
        )
        if reqrun.integer(row.get("selected_flag"))
    }
    new_selected = {
        (row.get("slot", ""), row.get("task_id", "")): row.get(
            "candidate_action", ""
        )
        for row in reqrun.read_csv(
            OUT / "runs" / "ORDINARY_NET" / "candidate_action_log.csv"
        )
        if reqrun.integer(row.get("selected_flag"))
    }
    selected_keys = set(reference_selected) | set(new_selected)
    mechanism["selected_action_mismatch_count"] = sum(
        reference_selected.get(key) != new_selected.get(key)
        for key in selected_keys
    )
    resource_fields = (
        "battery_soc",
        "thermal",
        "queue_mb",
        "compute_energy_j",
        "comm_energy_j",
        "transmitted_mb",
        "in_contact",
        "soc_violation",
        "thermal_violation",
    )
    reference_resource = reqrun.read_csv(
        OUT / "runs" / "REFERENCE" / "per_slot_resource_trace.csv"
    )
    new_resource = reqrun.read_csv(
        OUT / "runs" / "ORDINARY_NET" / "per_slot_resource_trace.csv"
    )
    mechanism["resource_trace_row_mismatch_count"] = sum(
        any(left.get(field) != right.get(field) for field in resource_fields)
        for left, right in zip(reference_resource, new_resource)
    ) + abs(len(reference_resource) - len(new_resource))
    same_slot_rows = [
        row
        for row in ordinary
        if row.get("decision_reason") == "same_slot_surplus_projected"
    ]
    reference_resource_by_slot = {
        reqrun.integer(row.get("slot")): row for row in reference_resource
    }
    new_resource_by_slot = {
        reqrun.integer(row.get("slot")): row for row in new_resource
    }
    delivery_events = reqrun.read_csv(
        OUT / "runs" / "ORDINARY_NET" / "ground_delivery_events.csv"
    )
    same_slot_delivery_misses = 0
    same_slot_future_delivery_events = 0
    future_queue_increases = 0
    for row in same_slot_rows:
        slot = reqrun.integer(row.get("slot"))
        slot_start = float(slot * 20)
        slot_end = slot_start + 20.0
        delivered_in_slot = any(
            event.get("task_id") == row.get("task_id")
            and slot_start - 1.0e-9
            <= reqrun.flt(event.get("ground_available_time"), math.inf)
            <= slot_end + 1.0e-9
            for event in delivery_events
        )
        same_slot_delivery_misses += int(not delivered_in_slot)
        same_slot_future_delivery_events += sum(
            event.get("task_id") == row.get("task_id")
            and reqrun.flt(event.get("ground_available_time"), -math.inf)
            > slot_end + 1.0e-9
            for event in delivery_events
        )
        reference_next = reference_resource_by_slot.get(slot + 1, {})
        new_next = new_resource_by_slot.get(slot + 1, {})
        future_queue_increases += int(
            reqrun.flt(new_next.get("queue_mb"), 0.0)
            > reqrun.flt(reference_next.get("queue_mb"), 0.0) + 1.0e-9
        )
    mechanism["same_slot_surplus_action_count"] = len(same_slot_rows)
    mechanism["same_slot_realized_delivery_miss_count"] = (
        same_slot_delivery_misses
    )
    mechanism["same_slot_future_delivery_event_count"] = (
        same_slot_future_delivery_events
    )
    mechanism["same_slot_next_slot_queue_increase_count"] = (
        future_queue_increases
    )
    mechanism["same_slot_nonpositive_predicted_net_count"] = sum(
        reqrun.flt(row.get("final_predicted_net"), 0.0) <= 1.0e-12
        for row in same_slot_rows
    )
    mechanism["same_slot_followup_suppression_missing_count"] = sum(
        reqrun.integer(row.get("ordinary_followup_suppression_applied"))
        != 1
        for row in same_slot_rows
    )
    mechanism["suppressed_pending_product_count"] = sum(
        reqrun.integer(row.get("suppressed_pending_product_count"))
        for row in same_slot_rows
    )
    mechanism["suppressed_staged_request_count"] = sum(
        reqrun.integer(row.get("suppressed_staged_request_count"))
        for row in same_slot_rows
    )
    mechanism["suppressed_residual_candidate_count"] = sum(
        reqrun.integer(row.get("suppressed_residual_candidate_count"))
        for row in same_slot_rows
    )
    mechanism["suppressed_staged_creation_count"] = sum(
        reqrun.integer(row.get("suppressed_staged_creation_count"))
        for row in same_slot_rows
    )
    mechanism["superseded_lifecycle_record_count"] = sum(
        reqrun.integer(row.get("superseded_lifecycle_record_count"))
        for row in same_slot_rows
    )
    safety_ok = (
        reqrun.integer(after.get("soc_violation_count")) == 0
        and reqrun.integer(after.get("thermal_violation_count")) == 0
    )
    audit = {
        "engineering_complete": True,
        "single_seed_diagnostic": True,
        "seed": 2026,
        "capacity": CAPACITY,
        "method_identity": METHOD_NEW,
        "high_urgent_hard_bypass_passed": (
            mechanism["high_urgent_non_bypass_count"] == 0
            and mechanism["high_urgent_action_changed_by_new_module"] == 0
        ),
        "resource_nonincrease_observed": (
            mechanism["ordinary_payload_delta_mb_sum"] <= 1.0e-9
            and mechanism["ordinary_energy_delta_j_sum"] <= 1.0e-9
        ),
        "same_slot_surplus_enabled": ENABLE_SAME_SLOT_SURPLUS,
        "same_slot_surplus_execution_audit_passed": (
            not ENABLE_SAME_SLOT_SURPLUS
            or (
                mechanism["same_slot_realized_delivery_miss_count"] == 0
                and mechanism["same_slot_future_delivery_event_count"] == 0
                and mechanism[
                    "same_slot_nonpositive_predicted_net_count"
                ]
                == 0
                and mechanism[
                    "same_slot_followup_suppression_missing_count"
                ]
                == 0
            )
        ),
        "public_accounting_passed": bool(after.get("public_accounting_passed")),
        "hard_safety_passed": safety_ok,
        "mechanism": mechanism,
    }
    guardrun.write_json(
        OUT / "audit" / "ordinary_net_granularity_audit.json", audit
    )
    lines = [
        "# Ordinary-task net-utility granularity ablation",
        "",
        "Seed=2026, capacity=0.50. This is a single-seed mechanism diagnostic.",
        "",
        (
            "High and urgent tasks hard-bypass the new module. Low and medium tasks maximize predicted public marginal net utility; the SAME-SLOT variant may release a REQALIGN defer only with a current-contact, empty-queue, no-pending-HV, conservative same-slot completion certificate."
            if ENABLE_SAME_SLOT_SURPLUS
            else "High and urgent tasks hard-bypass the new module. Low and medium tasks maximize predicted public marginal net utility subject to same-model and componentwise resource nonincrease relative to corrected REQALIGN."
        ),
        "",
        "## Metrics",
        "",
        guardrun.markdown_table(table),
        "",
        "## Deltas",
        "",
        guardrun.markdown_table(deltas),
        "",
        "## Mechanism audit",
        "",
        "```json",
        json.dumps(audit, ensure_ascii=False, indent=2),
        "```",
    ]
    (OUT / "final_assessment.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    guardrun.write_json(
        OUT / "completion_status.json",
        {
            "status": "ORDINARY_NET_GRANULARITY_CAP050_COMPLETE",
            "output": str(OUT),
        },
    )
    print(guardrun.markdown_table(table))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("FINAL_STATUS=ORDINARY_NET_GRANULARITY_CAP050_COMPLETE")
    print(f"OUTPUT_DIR={OUT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--bundle-root")
    parser.add_argument("--output")
    parser.add_argument("--same-slot-surplus", action="store_true")
    args = parser.parse_args()
    configure_variant(args.same_slot_surplus)
    if args.child_run:
        run_new(Path(args.bundle_root), Path(args.output))
        return 0
    return parent_main(args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
