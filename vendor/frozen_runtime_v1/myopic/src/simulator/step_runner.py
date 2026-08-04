"""Single-step closed-loop executor."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.actions.action_space import ActionSpace
from src.common.types import (
    CommState,
    DetectionResult,
    DownlinkResult,
    LinkTrace,
    LongTermState,
    MissionState,
    ObservedFeedback,
    ProductLifecycleState,
    ResidualCandidateState,
    ResourceState,
    RewardBreakdown,
    ShieldDecision,
    TimelineStep,
)
from src.controllers.lyapunov_pricing import LyapunovPricing
from src.controllers.mode_controller import BaseModeController
from src.context.context_builder import ContextBuilder
from src.energy.battery_update import update_ai_energy_budget, update_battery_soc
from src.energy.comm_energy import compute_comm_energy_j
from src.energy.compute_energy import compute_inference_energy_j
from src.energy.soc_band_policy import apply_soc_band_policy
from src.energy.thermal_update import update_thermal_pressure
from src.energy.thermal_headroom import maximum_safe_downlink_capacity, thermal_headroom_config
from src.link.contact_window import compute_available_capacity_mb
from src.link.downlink_queue import DownlinkQueue
from src.link.link_model import estimate_rate_mb_per_s
from src.models.inference_runner import InferenceRunner
from src.models.model_registry import ModelRegistry
from src.payload.payload_sizer import estimate_payload_encoding_cost, estimate_payload_size_with_diagnostics, generate_payload
from src.payload.payload_sizer import generate_residual_payload
from src.payload.continue_request import RESIDUAL_PRODUCT_TYPES, evaluate_virtual_continue_request, phase5_continue_config
from src.payload.proactive_followup import (
    DEFAULT_FULL_METHOD_ALIASES,
    evaluate_proactive_followup_eligibility,
    phase6_proactive_config,
    phase6_residual_candidate_carryover_config,
    phase6_same_slot_carryover_config,
)
from src.payload.information_sufficiency import (
    compute_gross_incremental_utility,
    delta_information_utility,
    normalize_delivered_product_set,
    residual_information_utility,
)
from src.data.value_skew_calibration import task_effective_value
from src.payload.roi_context_gate import evaluate_roi_context_gate, load_gate_profile, roi_context_gate_config
from src.payload.staged_delivery_queue import (
    StagedDeliveryRequest,
    compute_staged_priority,
    request_to_trace_row,
    staged_delivery_enabled_for_router,
    staged_delivery_queue_config,
    value_level_factor,
    product_requirement_weight,
)
from src.payload.bounded_guarantee_queue import (
    debt_increment_for_value,
    guarantee_config,
    guarantee_enabled_for_router,
    guarantee_priority_active,
    is_high_value_guarantee_eligible,
    marginal_utility_floor,
    rank_guarantee_aware_requests,
    stage_from_delivered,
    stage_rank,
)
from src.payload.payload_lifecycle_ledger import PayloadLifecycleLedger
from src.reward.delivery import compute_delivery_factor
from src.reward.reward_function import compute_reward
from src.routers.base_router import BaseRouter
from src.routers.safe_filter import SafeFilter
from src.simulator.feedback import build_trajectory_record
from src.simulator.predictors import Predictor


@dataclass(frozen=True)
class PayloadExecutionFeasibilityDecision:
    allowed: bool
    reason_code: str
    blockers: tuple[str, ...] = field(default_factory=tuple)
    predicted_queue_next_mb: float = 0.0
    predicted_battery_next: float = 0.0
    predicted_thermal_next: float = 0.0
    extra_comm_energy_j: float = 0.0
    extra_payload_mb: float = 0.0
    fields_used: tuple[str, ...] = field(default_factory=tuple)


class StepRunner:
    """Strictly preserves the project mainline execution order."""

    def __init__(
        self,
        action_space: ActionSpace,
        context_builder: ContextBuilder,
        predictor: Predictor,
        safe_filter: SafeFilter,
        inference_runner: InferenceRunner,
        downlink_queue: DownlinkQueue,
        model_registry: ModelRegistry,
        reward_cfg: dict[str, float],
        simulator_cfg: dict[str, float],
        payload_profiles_cfg: dict[str, object] | None = None,
        lyapunov_pricing: LyapunovPricing | None = None,
        mode_controller: BaseModeController | None = None,
        future_link_traces: list[LinkTrace] | None = None,
    ) -> None:
        self.action_space = action_space
        self.context_builder = context_builder
        self.predictor = predictor
        self.safe_filter = safe_filter
        self.inference_runner = inference_runner
        self.downlink_queue = downlink_queue
        self.model_registry = model_registry
        self.reward_cfg = reward_cfg
        self.simulator_cfg = simulator_cfg
        self.payload_profiles_cfg = payload_profiles_cfg
        self.lyapunov_pricing = lyapunov_pricing
        self.mode_controller = mode_controller
        self.future_link_traces = list(future_link_traces or [])
        self.last_feedback = None
        self.last_reward_breakdown = None
        self.last_detection_result = None
        self.last_guidance_bundle = None
        self.last_candidate_action_log: list[dict[str, object]] = []
        self.last_queue_priority_log: list[dict[str, object]] = []
        self.current_mode_name = "disabled"
        self.current_mode_slot = -1
        self.deadline_urgency_history: list[float] = []
        self.high_value_history: list[float] = []
        self.queue_history: list[float] = []
        self.no_contact_streak_slots = 0
        self.product_lifecycle: dict[str, ProductLifecycleState] = {}
        self.last_continue_request_decision = None
        self.last_proactive_followup_decision = None
        self.last_passive_request_suppressed_by_prior_proactive = 0
        self.last_phase6_pre_eval_log: dict[str, object] = {}
        self.last_lifecycle_post_update_log: dict[str, object] = {}
        self.residual_candidate_pool: dict[str, ResidualCandidateState] = {}
        self.residual_candidate_history: dict[str, ResidualCandidateState] = {}
        self.last_carryover_pool_log: dict[str, int | float] = self._empty_carryover_pool_log()
        self.last_carryover_payload_key = ""
        self.last_carryover_feedback_task = None
        self.last_carryover_lifecycle_events: list[dict[str, object]] = []
        self.last_same_slot_carryover_trace: list[dict[str, object]] = []
        self.last_same_slot_carryover_payload = None
        self.last_staged_feasibility_filter_trace: list[dict[str, object]] = []
        self.last_staged_feasibility_filter_summary: dict[str, int] = self._empty_staged_feasibility_filter_summary()
        self.last_followup_hard_safety_guard_trace: list[dict[str, object]] = []
        self.last_roi_context_window_budget_trace: list[dict[str, object]] = []
        self.last_roi_context_budget_trace: list[dict[str, object]] = self.last_roi_context_window_budget_trace
        self.last_value_density_trace: list[dict[str, object]] = []
        self.last_same_slot_vs_queued_followup_trace: list[dict[str, object]] = []
        self.last_future_contact_feasibility_trace: list[dict[str, object]] = []
        self.last_duplicate_suppression_trace: list[dict[str, object]] = []
        self._roi_context_budget_slot_index = -1
        self._roi_context_slot_selected_count = 0
        self._roi_context_queued_budget_slot_index = -1
        self._roi_context_queued_selected_count = 0
        self._roi_context_contact_window_id = ""
        self._roi_context_contact_window_selected_count = 0
        self._roi_context_window_selected_count = 0
        self.task_snapshots: dict[str, object] = {}
        self.lifecycle_ledger = self._init_payload_lifecycle_ledger(downlink_queue)
        self.staged_delivery_requests: dict[str, StagedDeliveryRequest] = {}
        self.staged_delivery_history: dict[str, StagedDeliveryRequest] = {}
        self._staged_delivery_request_seq = 0
        self._staged_delivery_contact_window_id = ""
        self._staged_delivery_contact_window_admission_count = 0
        self.last_staged_delivery_request_trace: list[dict[str, object]] = []
        self.last_staged_delivery_priority_trace: list[dict[str, object]] = []
        self.last_contact_window_scheduling_trace: list[dict[str, object]] = []
        self.last_contact_window_candidate_rank_trace: list[dict[str, object]] = []
        self.last_contact_window_delivery_order_trace: list[dict[str, object]] = []
        self.last_ordinary_queue_vs_staged_queue_capacity_trace: list[dict[str, object]] = []
        self.last_backlog_composition_trace: list[dict[str, object]] = []
        self.last_leftover_tail_value_trace: list[dict[str, object]] = []
        self.guarantee_debt_by_task: dict[str, float] = {}
        self.guarantee_age_debt_by_task: dict[str, float] = {}
        self.guarantee_miss_debt_by_task: dict[str, float] = {}
        self.guarantee_waiting_age_slots_by_task: dict[str, int] = {}
        self._guarantee_last_age_update_slot_by_task: dict[str, int] = {}
        self.guarantee_roi_wait_windows: dict[str, int] = {}
        self._guarantee_contact_window_id = ""
        self._guarantee_window_used_mb = 0.0
        self._guarantee_window_observations: dict[str, dict[str, object]] = {}
        self.last_guarantee_debt_trace: list[dict[str, object]] = []
        self.last_guarantee_debt_window_update_trace: list[dict[str, object]] = []
        self.last_task_debt_trajectory_trace: list[dict[str, object]] = []
        self.last_candidate_priority_trace: list[dict[str, object]] = []
        self.last_thermal_safe_capacity_trace: list[dict[str, object]] = []
        self.last_thermal_hysteresis_trace: list[dict[str, object]] = []
        self._thermal_headroom_recovery_mode = False
        self.last_guarantee_opportunity_trace: list[dict[str, object]] = []
        self.last_guarantee_progress_trace: list[dict[str, object]] = []
        self.last_guarantee_priority_component_trace: list[dict[str, object]] = []
        self.last_guarantee_budget_trace: list[dict[str, object]] = []
        self.last_guarantee_blocker_reason_trace: list[dict[str, object]] = []

    def _init_payload_lifecycle_ledger(self, downlink_queue: DownlinkQueue | None) -> PayloadLifecycleLedger:
        existing = getattr(downlink_queue, "lifecycle_ledger", None) if downlink_queue is not None else None
        if existing is not None:
            return existing
        ledger = PayloadLifecycleLedger(slot_seconds=float(self.simulator_cfg.get("slot_seconds", 1.0)))
        if downlink_queue is not None:
            downlink_queue.lifecycle_ledger = ledger
        return ledger

    @staticmethod
    def _empty_staged_feasibility_filter_summary() -> dict[str, int]:
        return {
            "staged_feasibility_filter_enabled": 0,
            "staged_candidate_evaluated_count": 0,
            "staged_candidate_feasible_count": 0,
            "staged_candidate_infeasible_count": 0,
            "staged_candidate_blocked_by_contact_count": 0,
            "staged_candidate_blocked_by_deadline_count": 0,
            "staged_candidate_blocked_by_queue_count": 0,
            "staged_candidate_blocked_by_energy_count": 0,
            "staged_candidate_blocked_by_thermal_count": 0,
            "staged_candidate_demoted_count": 0,
            "staged_replaced_by_direct_roi_count": 0,
            "staged_replaced_by_metadata_only_count": 0,
            "staged_replaced_by_defer_count": 0,
            "metadata_only_after_filter_count": 0,
            "residual_completion_after_filter_count": 0,
            "passive_would_miss_but_proactive_meets_count": 0,
            "proactive_and_passive_both_miss_count": 0,
        }

    def _staged_feasibility_filter_cfg(self) -> dict[str, object]:
        repair_cfg = self.simulator_cfg.get("policy_repair", {})
        if not isinstance(repair_cfg, dict):
            return {"enabled": False}
        cfg = repair_cfg.get("staged_feasibility_filter", {})
        cfg = dict(cfg) if isinstance(cfg, dict) else {}
        enabled = bool(repair_cfg.get("enable_staged_feasibility_filter", False)) or bool(cfg.get("enabled", False))
        phase6_cfg = phase6_proactive_config(self.simulator_cfg)
        return {
            "enabled": enabled,
            "battery_guard_threshold": float(
                cfg.get("battery_guard_threshold", phase6_cfg.get("battery_guard_threshold", 0.60))
            ),
            "thermal_guard_threshold": float(
                cfg.get("thermal_guard_threshold", phase6_cfg.get("thermal_guard_threshold", 0.85))
            ),
            "passive_nominal_latency_slots": int(cfg.get("passive_nominal_latency_slots", 3)),
        }

    @staticmethod
    def _is_staged_payload(payload_policy: str) -> bool:
        return payload_policy in {"metadata_then_roi", "metadata_then_roi_context"}

    @staticmethod
    def _direct_payload_for_staged_payload(payload_policy: str) -> str:
        return "roi_context" if payload_policy == "metadata_then_roi_context" else "roi"

    def _future_capacity_mb(self, slot_index: int, slot_seconds: float) -> float:
        future_link_traces = getattr(self, "future_link_traces", [])
        if 0 <= slot_index < len(future_link_traces):
            return compute_available_capacity_mb(
                future_link_traces[slot_index],
                slot_seconds,
                float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
            )
        return 0.0

    def _known_capacity_until_deadline(self, start_slot: int, deadline_slot: int, slot_seconds: float) -> float:
        future_link_traces = getattr(self, "future_link_traces", [])
        if not future_link_traces:
            return 0.0
        start = max(0, min(start_slot, len(future_link_traces) - 1))
        end = max(start, min(deadline_slot, len(future_link_traces) - 1))
        return sum(self._future_capacity_mb(slot, slot_seconds) for slot in range(start, end + 1))

    def _earliest_slot_for_demand(
        self,
        start_slot: int,
        deadline_slot: int,
        slot_seconds: float,
        demand_mb: float,
    ) -> int | None:
        if demand_mb <= 1.0e-12:
            return max(0, start_slot)
        future_link_traces = getattr(self, "future_link_traces", [])
        if not future_link_traces:
            return None
        start = max(0, min(start_slot, len(future_link_traces) - 1))
        end = max(start, min(deadline_slot, len(future_link_traces) - 1))
        served_mb = 0.0
        for slot in range(start, end + 1):
            served_mb += self._future_capacity_mb(slot, slot_seconds)
            if served_mb + 1.0e-12 >= demand_mb:
                return slot
        return None

    def _service_before_slot(self, start_slot: int, stop_slot_exclusive: int, slot_seconds: float) -> float:
        future_link_traces = getattr(self, "future_link_traces", [])
        if not future_link_traces or stop_slot_exclusive <= start_slot:
            return 0.0
        start = max(0, min(start_slot, len(future_link_traces)))
        stop = max(start, min(stop_slot_exclusive, len(future_link_traces)))
        return sum(self._future_capacity_mb(slot, slot_seconds) for slot in range(start, stop))

    @staticmethod
    def _slot_meets_deadline(slot_index: int | None, deadline_time_s: float, slot_seconds: float) -> bool:
        return slot_index is not None and float(slot_index) * slot_seconds <= deadline_time_s + 1.0e-9

    def _residual_size_from_direct_action(self, all_actions, predicted_map, staged_payload: str) -> float:
        direct_payload = self._direct_payload_for_staged_payload(staged_payload)
        sizes = [
            float(predicted_map[action.action_name].predicted_S_gen_mb)
            for action in all_actions
            if action.payload_policy == direct_payload and action.action_name in predicted_map
        ]
        if sizes:
            return max(sizes)
        size_mb, _diagnostics = estimate_payload_size_with_diagnostics(
            direct_payload,
            detection_result=None,
            task=None,
            payload_profiles_cfg=self.payload_profiles_cfg,
        )
        return float(size_mb)

    def _evaluate_staged_candidate_feasibility(
        self,
        *,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        action,
        all_actions,
        predicted_map,
        cfg: dict[str, object],
    ) -> dict[str, object]:
        task = timeline_step.task
        slot = int(timeline_step.slot_index)
        slot_seconds = float(timeline_step.slot_seconds)
        deadline_remaining_s = self._remaining_deadline_s(timeline_step)
        deadline_time_s = float(slot) * slot_seconds + deadline_remaining_s
        if self.future_link_traces:
            deadline_slot = max(0, min(len(self.future_link_traces) - 1, int(deadline_time_s // slot_seconds)))
        else:
            deadline_slot = slot
        metadata_size_mb = float(predicted_map[action.action_name].predicted_S_gen_mb)
        residual_size_mb = self._residual_size_from_direct_action(all_actions, predicted_map, action.payload_policy)
        queue_mb = float(comm_state.queue_mb)
        known_capacity_mb = self._known_capacity_until_deadline(slot, deadline_slot, slot_seconds)
        available_service_mb = max(0.0, known_capacity_mb - queue_mb)
        metadata_slot = self._earliest_slot_for_demand(slot, deadline_slot, slot_seconds, queue_mb + metadata_size_mb)
        proactive_slot = self._earliest_slot_for_demand(
            slot,
            deadline_slot,
            slot_seconds,
            queue_mb + metadata_size_mb + residual_size_mb,
        )
        passive_request_slot = None if metadata_slot is None else metadata_slot + int(cfg["passive_nominal_latency_slots"])
        if passive_request_slot is None or passive_request_slot > deadline_slot:
            passive_slot = None
        else:
            served_before_request = self._service_before_slot(slot, passive_request_slot, slot_seconds)
            backlog_at_request = max(0.0, queue_mb + metadata_size_mb - served_before_request)
            passive_slot = self._earliest_slot_for_demand(
                passive_request_slot,
                deadline_slot,
                slot_seconds,
                backlog_at_request + residual_size_mb,
            )
        metadata_feasible = self._slot_meets_deadline(metadata_slot, deadline_time_s, slot_seconds)
        residual_by_deadline = self._slot_meets_deadline(proactive_slot, deadline_time_s, slot_seconds)
        residual_by_contact = proactive_slot is not None and known_capacity_mb > queue_mb + 1.0e-12
        residual_by_queue = available_service_mb + 1.0e-12 >= metadata_size_mb + residual_size_mb
        residual_by_energy = float(resource_state.battery_soc) >= float(cfg["battery_guard_threshold"])
        residual_by_thermal = float(resource_state.thermal_pressure) < float(cfg["thermal_guard_threshold"])
        blockers: list[str] = []
        if not metadata_feasible:
            blockers.append("metadata_alert_misses_deadline_or_contact")
        if not residual_by_deadline:
            blockers.append("residual_followup_misses_deadline")
        if not residual_by_contact:
            blockers.append("insufficient_known_contact_capacity")
        if not residual_by_queue:
            blockers.append("queue_service_before_deadline_insufficient")
        if not residual_by_energy:
            blockers.append("battery_below_followup_guard")
        if not residual_by_thermal:
            blockers.append("thermal_above_followup_guard")
        feasible = not blockers
        proactive_meets = self._slot_meets_deadline(proactive_slot, deadline_time_s, slot_seconds)
        passive_meets = self._slot_meets_deadline(passive_slot, deadline_time_s, slot_seconds)
        return {
            "slot": slot,
            "task_id": "" if task is None else task.task_id,
            "candidate_action": action.action_name,
            "candidate_payload": action.payload_policy,
            "metadata_size_mb_estimate": metadata_size_mb,
            "residual_size_mb_estimate": residual_size_mb,
            "queue_mb": queue_mb,
            "deadline_remaining_s": deadline_remaining_s,
            "deadline_slot": deadline_slot,
            "known_contact_capacity_before_deadline_mb": known_capacity_mb,
            "estimated_available_queue_service_before_deadline_mb": available_service_mb,
            "metadata_delivery_slot_estimate": -1 if metadata_slot is None else metadata_slot,
            "proactive_residual_earliest_slot": -1 if proactive_slot is None else proactive_slot,
            "passive_nominal_residual_earliest_slot": -1 if passive_slot is None else passive_slot,
            "metadata_alert_feasible": int(metadata_feasible),
            "residual_followup_feasible_by_deadline": int(residual_by_deadline),
            "residual_followup_feasible_by_contact": int(residual_by_contact),
            "residual_followup_feasible_by_queue": int(residual_by_queue),
            "residual_followup_feasible_by_energy": int(residual_by_energy),
            "residual_followup_feasible_by_thermal": int(residual_by_thermal),
            "staged_feasible_overall": int(feasible),
            "proactive_meets_nominal_deadline": int(proactive_meets),
            "passive_nominal_meets_deadline": int(passive_meets),
            "passive_would_miss_but_proactive_meets": int(proactive_meets and not passive_meets),
            "proactive_and_passive_both_miss": int((not proactive_meets) and (not passive_meets)),
            "blockers": ";".join(blockers),
        }

    def _apply_staged_feasibility_filter(
        self,
        *,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        actions,
        candidate_actions,
        predicted_map,
    ):
        self.last_staged_feasibility_filter_trace = []
        self.last_staged_feasibility_filter_summary = self._empty_staged_feasibility_filter_summary()
        cfg = self._staged_feasibility_filter_cfg()
        if not bool(cfg.get("enabled", False)):
            return candidate_actions
        self.last_staged_feasibility_filter_summary["staged_feasibility_filter_enabled"] = 1
        if timeline_step.task is None or not self.future_link_traces:
            return candidate_actions

        filtered_actions = []
        demoted_action_names: set[str] = set()
        for action in candidate_actions:
            if not self._is_staged_payload(action.payload_policy):
                filtered_actions.append(action)
                continue
            result = self._evaluate_staged_candidate_feasibility(
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                action=action,
                all_actions=actions,
                predicted_map=predicted_map,
                cfg=cfg,
            )
            self.last_staged_feasibility_filter_trace.append(dict(result, filter_decision="keep" if result["staged_feasible_overall"] else "demote"))
            self.last_staged_feasibility_filter_summary["staged_candidate_evaluated_count"] += 1
            self.last_staged_feasibility_filter_summary["staged_candidate_feasible_count"] += int(result["staged_feasible_overall"])
            self.last_staged_feasibility_filter_summary["staged_candidate_infeasible_count"] += int(
                not result["staged_feasible_overall"]
            )
            self.last_staged_feasibility_filter_summary["staged_candidate_blocked_by_contact_count"] += int(
                not result["residual_followup_feasible_by_contact"]
            )
            self.last_staged_feasibility_filter_summary["staged_candidate_blocked_by_deadline_count"] += int(
                not result["residual_followup_feasible_by_deadline"]
            )
            self.last_staged_feasibility_filter_summary["staged_candidate_blocked_by_queue_count"] += int(
                not result["residual_followup_feasible_by_queue"]
            )
            self.last_staged_feasibility_filter_summary["staged_candidate_blocked_by_energy_count"] += int(
                not result["residual_followup_feasible_by_energy"]
            )
            self.last_staged_feasibility_filter_summary["staged_candidate_blocked_by_thermal_count"] += int(
                not result["residual_followup_feasible_by_thermal"]
            )
            self.last_staged_feasibility_filter_summary["passive_would_miss_but_proactive_meets_count"] += int(
                result["passive_would_miss_but_proactive_meets"]
            )
            self.last_staged_feasibility_filter_summary["proactive_and_passive_both_miss_count"] += int(
                result["proactive_and_passive_both_miss"]
            )
            if result["staged_feasible_overall"]:
                filtered_actions.append(action)
            else:
                demoted_action_names.add(action.action_name)

        if not filtered_actions:
            self.last_staged_feasibility_filter_trace.append(
                {
                    "slot": int(timeline_step.slot_index),
                    "task_id": timeline_step.task.task_id,
                    "candidate_action": "",
                    "candidate_payload": "",
                    "filter_decision": "restore_all_candidates",
                    "blockers": "all_safe_candidates_would_be_demoted",
                }
            )
            return candidate_actions
        self.last_staged_feasibility_filter_summary["staged_candidate_demoted_count"] = len(demoted_action_names)
        return filtered_actions

    def _apply_roi_context_gate_to_actions(
        self,
        *,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        candidate_actions,
        predicted_map,
    ):
        task = timeline_step.task
        gate_cfg = roi_context_gate_config(self.simulator_cfg)
        if task is None or not bool(gate_cfg.get("enabled", False)):
            return candidate_actions
        self._refresh_roi_context_budget_state(timeline_step)
        filtered = []
        value_level = str(getattr(task, "value_level", "") or (task.metadata or {}).get("value_level", ""))
        roi_context_candidates = []
        for action in candidate_actions:
            if action.payload_policy not in {"roi_context", "metadata_then_roi_context"}:
                continue
            predicted = predicted_map[action.action_name]
            residual_delta = max(0.0, float(predicted.predicted_incremental_information_utility))
            detection_quality = max(0.0, float(predicted.predicted_detection_quality))
            delivery_factor = max(0.0, float(predicted.predicted_delivery_factor))
            payload_size_mb = max(float(predicted.predicted_S_gen_mb), 1.0e-9)
            value_density = (
                task_effective_value(task)
                * residual_delta
                * detection_quality
                * delivery_factor
                / payload_size_mb
            )
            self._record_value_density_trace(
                timeline_step=timeline_step,
                task_id=task.task_id,
                stage="module2_candidate_action_gate",
                product_type=action.payload_policy,
                residual_delta_u_i=residual_delta,
                detection_quality=detection_quality,
                timeliness_factor=delivery_factor,
                payload_size_mb=payload_size_mb,
                value_density=value_density,
                information_density=residual_delta / payload_size_mb,
                eriv_score=value_density,
                followup_mode="same_slot",
            )
            roi_context_candidates.append((action, value_density))
        ranked_roi_context = sorted(
            roi_context_candidates,
            key=lambda item: (float(item[1]), float(predicted_map[item[0].action_name].predicted_reward)),
            reverse=True,
        )
        rank_by_name = {action.action_name: index + 1 for index, (action, _density) in enumerate(ranked_roi_context)}
        density_by_name = {action.action_name: density for action, density in roi_context_candidates}
        for action in candidate_actions:
            if action.payload_policy not in {"roi_context", "metadata_then_roi_context"}:
                filtered.append(action)
                continue
            predicted = predicted_map[action.action_name]
            value_density = float(density_by_name.get(action.action_name, 0.0))
            remaining_deadline_s = self._remaining_deadline_s(timeline_step)
            budget_counts = self._roi_context_budget_counts(timeline_step, "same_slot")
            decision = evaluate_roi_context_gate(
                product_requirement=task.product_requirement,
                value_level=value_level,
                value_density=value_density,
                system_state={
                    "queue_mb": comm_state.queue_mb,
                    "queue_max_mb": comm_state.queue_max_mb,
                    "queue_pressure": comm_state.queue_mb / max(comm_state.queue_max_mb, 1.0e-9),
                    "battery_soc": resource_state.battery_soc,
                    "thermal_pressure": resource_state.thermal_pressure,
                    "remaining_deadline_s": remaining_deadline_s,
                    "contact_available": timeline_step.link_trace.in_contact,
                    "contact_opportunity_score": 1.0 if timeline_step.link_trace.in_contact else 0.0,
                    "candidate_rank": rank_by_name.get(action.action_name, 999),
                    **budget_counts,
                },
                config=self.simulator_cfg,
                eriv_score=value_density,
                residual_product_type="roi_context",
            )
            self._record_roi_context_window_budget_trace(
                timeline_step=timeline_step,
                task_id=task.task_id,
                candidate_name=action.action_name,
                candidate_payload=action.payload_policy,
                candidate_rank=rank_by_name.get(action.action_name, 999),
                window_selected_count=int(budget_counts["window_selected_count"]),
                decision=decision,
                stage="module2_candidate_action_gate",
                followup_mode="same_slot",
            )
            if decision.allowed:
                filtered.append(action)
                self._increment_roi_context_budget(timeline_step, "same_slot")
        return filtered or candidate_actions

    def _record_roi_context_window_budget_trace(
        self,
        *,
        timeline_step: TimelineStep,
        task_id: str,
        candidate_name: str,
        candidate_payload: str,
        candidate_rank: int,
        window_selected_count: int,
        decision,
        stage: str = "",
        followup_mode: str = "same_slot",
    ) -> None:
        self._ensure_roi_context_trace_state()
        _enabled, profile_name, profile = load_gate_profile(self.simulator_cfg)
        budget_counts = self._roi_context_budget_counts(timeline_step, followup_mode)
        self.last_roi_context_window_budget_trace.append(
            {
                "slot": int(timeline_step.slot_index),
                "task_id": task_id,
                "stage": stage,
                "followup_mode": followup_mode,
                "candidate_name": candidate_name,
                "candidate_payload": candidate_payload,
                "candidate_rank": int(candidate_rank),
                "window_selected_count": int(window_selected_count),
                "slot_selected_count": int(budget_counts["slot_selected_count"]),
                "contact_window_id": str(budget_counts["contact_window_id"]),
                "contact_window_selected_count": int(budget_counts["contact_window_selected_count"]),
                "queued_followup_selected_count": int(budget_counts["queued_followup_selected_count"]),
                "top_k_per_slot": int(float(profile.get("top_k_per_slot", profile.get("top_k_per_window", 0)) or 0)),
                "top_k_per_window": int(float(profile.get("top_k_per_window", 0) or 0)),
                "top_k_per_contact_window": int(
                    float(profile.get("top_k_per_contact_window", profile.get("window_budget", 0)) or 0)
                ),
                "queued_followup_budget_per_slot": int(float(profile.get("queued_followup_budget_per_slot", 0) or 0)),
                "window_budget": int(float(profile.get("window_budget", 0) or 0)),
                "allowed": int(bool(decision.allowed)),
                "reason_code": str(decision.reason_code),
                "blockers": self._format_products(decision.blockers),
                "profile": profile_name,
                "value_density": f"{float(decision.value_density):.6f}",
                "threshold": f"{float(decision.threshold):.6f}",
            }
        )

    def _record_staged_feasibility_replacement(self, executed_action) -> None:
        summary = self.last_staged_feasibility_filter_summary
        if not summary.get("staged_feasibility_filter_enabled") or summary.get("staged_candidate_demoted_count", 0) <= 0:
            return
        payload_policy = getattr(executed_action, "payload_policy", "")
        if payload_policy in {"roi", "roi_context"}:
            summary["staged_replaced_by_direct_roi_count"] = 1
        elif payload_policy == "metadata":
            summary["staged_replaced_by_metadata_only_count"] = 1
            summary["metadata_only_after_filter_count"] = 1
        elif payload_policy == "defer" or getattr(executed_action, "model_name", "") == "delay_store":
            summary["staged_replaced_by_defer_count"] = 1

    def _copy_staged_feasibility_metrics_to_feedback(self, feedback: ObservedFeedback) -> None:
        summary = self.last_staged_feasibility_filter_summary
        feedback.staged_feasibility_filter_enabled = int(summary.get("staged_feasibility_filter_enabled", 0))
        feedback.staged_candidate_evaluated_count = int(summary.get("staged_candidate_evaluated_count", 0))
        feedback.staged_candidate_feasible_count = int(summary.get("staged_candidate_feasible_count", 0))
        feedback.staged_candidate_infeasible_count = int(summary.get("staged_candidate_infeasible_count", 0))
        feedback.staged_candidate_blocked_by_contact_count = int(summary.get("staged_candidate_blocked_by_contact_count", 0))
        feedback.staged_candidate_blocked_by_deadline_count = int(summary.get("staged_candidate_blocked_by_deadline_count", 0))
        feedback.staged_candidate_blocked_by_queue_count = int(summary.get("staged_candidate_blocked_by_queue_count", 0))
        feedback.staged_candidate_blocked_by_energy_count = int(summary.get("staged_candidate_blocked_by_energy_count", 0))
        feedback.staged_candidate_blocked_by_thermal_count = int(summary.get("staged_candidate_blocked_by_thermal_count", 0))
        feedback.staged_candidate_demoted_count = int(summary.get("staged_candidate_demoted_count", 0))
        feedback.staged_replaced_by_direct_roi_count = int(summary.get("staged_replaced_by_direct_roi_count", 0))
        feedback.staged_replaced_by_metadata_only_count = int(summary.get("staged_replaced_by_metadata_only_count", 0))
        feedback.staged_replaced_by_defer_count = int(summary.get("staged_replaced_by_defer_count", 0))
        feedback.metadata_only_after_filter_count = int(summary.get("metadata_only_after_filter_count", 0))
        feedback.residual_completion_after_filter_count = int(
            bool(summary.get("staged_feasibility_filter_enabled", 0)) and feedback.residual_delivery_count > 0
        )
        feedback.passive_would_miss_but_proactive_meets_count = int(
            summary.get("passive_would_miss_but_proactive_meets_count", 0)
        )
        feedback.proactive_and_passive_both_miss_count = int(summary.get("proactive_and_passive_both_miss_count", 0))

    def _get_product_lifecycle(self, task) -> ProductLifecycleState:
        state = self.product_lifecycle.get(task.task_id)
        if state is None:
            state = ProductLifecycleState(
                task_id=task.task_id,
                product_requirement=task.product_requirement,
            )
            self.product_lifecycle[task.task_id] = state
        return state

    def _build_long_term_state(
        self,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        context,
    ) -> LongTermState:
        history_window = max(1, int(self.mode_controller.update_interval_slots)) if self.mode_controller else 10
        self.deadline_urgency_history.append(float(context.D_t))
        self.high_value_history.append(float(context.V_t >= float(self.reward_cfg.get("high_value_threshold", 0.75))))
        self.queue_history.append(float(comm_state.queue_mb))
        self.deadline_urgency_history = self.deadline_urgency_history[-history_window:]
        self.high_value_history = self.high_value_history[-history_window:]
        self.queue_history = self.queue_history[-history_window:]

        if timeline_step.link_trace.in_contact:
            self.no_contact_streak_slots = 0
        else:
            self.no_contact_streak_slots += 1

        recent_deadline_urgency = sum(self.deadline_urgency_history) / max(len(self.deadline_urgency_history), 1)
        recent_high_value_ratio = sum(self.high_value_history) / max(len(self.high_value_history), 1)
        queue_growth_rate = 0.0
        if len(self.queue_history) >= 2:
            queue_growth_rate = (self.queue_history[-1] - self.queue_history[0]) / max(len(self.queue_history) - 1, 1)

        if timeline_step.link_trace.in_contact:
            next_contact_s = 0.0
            low_contact_capacity_factor = max(
                0.0,
                1.0 - min(timeline_step.link_trace.remaining_window_s / max(timeline_step.slot_seconds * 4.0, 1.0e-9), 1.0),
            )
        else:
            next_contact_s = float(self.no_contact_streak_slots * timeline_step.slot_seconds)
            low_contact_capacity_factor = 1.0

        return LongTermState(
            battery_soc=resource_state.battery_soc,
            battery_safe=float(self.simulator_cfg["battery_min"]),
            thermal_pressure=resource_state.thermal_pressure,
            thermal_max=float(self.simulator_cfg["thermal_max"]),
            queue_mb=comm_state.queue_mb,
            queue_max_mb=comm_state.queue_max_mb,
            recent_deadline_urgency=recent_deadline_urgency,
            recent_high_value_ratio=recent_high_value_ratio,
            next_contact_s=next_contact_s,
            low_contact_capacity_factor=low_contact_capacity_factor,
            queue_growth_rate=queue_growth_rate,
        )

    def _build_guidance_bundle(
        self,
        slot_index: int,
        long_term_state: LongTermState,
        router: BaseRouter,
    ) -> dict[str, object] | None:
        use_lyapunov = bool(router.use_lyapunov_pricing and self.lyapunov_pricing is not None)
        use_mode = bool(router.use_mode_controller and self.mode_controller is not None)

        if not use_lyapunov:
            if not use_mode:
                self.current_mode_name = "disabled"
                return None
            if self.current_mode_slot < 0 or (slot_index - self.current_mode_slot) >= self.mode_controller.update_interval_slots:
                self.current_mode_name = self.mode_controller.select_mode(long_term_state)
                self.current_mode_slot = slot_index
            lambda_base = {"lambda_E": 0.0, "lambda_L": 0.0, "lambda_Q": 0.0, "lambda_S": 0.0}
            guidance_bundle = self.mode_controller.build_guidance(lambda_base, self.current_mode_name)
            guidance_bundle["lambda_E_base"] = 0.0
            guidance_bundle["lambda_L_base"] = 0.0
            guidance_bundle["lambda_Q_base"] = 0.0
            guidance_bundle["lambda_S_base"] = 0.0
            return guidance_bundle

        lambda_base = self.lyapunov_pricing.compute_base_prices(long_term_state)
        if not use_mode:
            self.current_mode_name = "no_mode"
            return {
                "mode_name": self.current_mode_name,
                "lambda_base": dict(lambda_base),
                "lambda_E_base": lambda_base["lambda_E"],
                "lambda_L_base": lambda_base["lambda_L"],
                "lambda_Q_base": lambda_base["lambda_Q"],
                "lambda_S_base": lambda_base["lambda_S"],
                "lambda_E": lambda_base["lambda_E"],
                "lambda_L": lambda_base["lambda_L"],
                "lambda_Q": lambda_base["lambda_Q"],
                "lambda_S": lambda_base["lambda_S"],
                "price_multipliers": {"lambda_E": 1.0, "lambda_L": 1.0, "lambda_Q": 1.0, "lambda_S": 1.0},
                "exploration_scale": 1.0,
                "model_complexity_bias": 0.0,
                "payload_size_bias": 0.0,
                "high_value_bonus_scale": 1.0,
                "safety_margin_scale": 1.0,
                "z_k": dict(lambda_base),
            }

        if self.current_mode_slot < 0 or (slot_index - self.current_mode_slot) >= self.mode_controller.update_interval_slots:
            self.current_mode_name = self.mode_controller.select_mode(long_term_state)
            self.current_mode_slot = slot_index
        guidance_bundle = self.mode_controller.build_guidance(lambda_base, self.current_mode_name)
        guidance_bundle["lambda_E_base"] = lambda_base["lambda_E"]
        guidance_bundle["lambda_L_base"] = lambda_base["lambda_L"]
        guidance_bundle["lambda_Q_base"] = lambda_base["lambda_Q"]
        guidance_bundle["lambda_S_base"] = lambda_base["lambda_S"]
        return guidance_bundle

    def _apply_policy_repair_guidance(
        self,
        timeline_step: TimelineStep,
        long_term_state: LongTermState,
        guidance_bundle: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if guidance_bundle is None:
            return None
        guidance_bundle = dict(guidance_bundle)
        repair_cfg = self.simulator_cfg.get("policy_repair", {})
        guidance_bundle["policy_repair"] = repair_cfg if isinstance(repair_cfg, dict) else {}
        scoring_cfg = self.simulator_cfg.get("module2_product_requirement_aware_scoring", {})
        guidance_bundle["module2_product_requirement_aware_scoring"] = scoring_cfg if isinstance(scoring_cfg, dict) else {}
        if timeline_step.task is not None:
            task_metadata = dict(timeline_step.task.metadata or {})
            guidance_bundle["product_requirement_context"] = {
                "ideal_product_requirement": timeline_step.task.product_requirement,
                "product_requirement_source": timeline_step.task.product_requirement_source,
                "initial_alert_required": bool(task_metadata.get("v21_initial_alert_required", False)),
                "requirement_scores": dict(task_metadata.get("v21_requirement_scores", {}))
                if isinstance(task_metadata.get("v21_requirement_scores", {}), dict)
                else {},
                "task_value": float(task_effective_value(timeline_step.task)),
                "uncertainty": float(timeline_step.task.uncertainty),
                "deadline_s": float(timeline_step.task.deadline_s),
                "high_value_flag": bool(timeline_step.task.high_value_flag),
                "emergency_flag": bool(timeline_step.task.emergency_flag),
                "task_class": timeline_step.task.task_class,
            }
        guidance_bundle["resource_context"] = {
            "is_eclipse": bool(timeline_step.resource_trace.is_eclipse),
            "is_sunlit": bool(timeline_step.resource_trace.is_sunlit),
            "in_contact": bool(timeline_step.link_trace.in_contact),
            "remaining_window_s": float(timeline_step.link_trace.remaining_window_s),
        }
        guidance_bundle["lambda_before_z"] = {
            "lambda_E": float(guidance_bundle.get("lambda_E", 0.0)),
            "lambda_L": float(guidance_bundle.get("lambda_L", 0.0)),
            "lambda_Q": float(guidance_bundle.get("lambda_Q", 0.0)),
            "lambda_S": float(guidance_bundle.get("lambda_S", 0.0)),
        }
        guidance_bundle["beta_before_z"] = float(guidance_bundle.get("exploration_scale", 1.0))
        guidance_bundle["z_guidance_reason"] = "disabled"
        if not isinstance(repair_cfg, dict) or not bool(repair_cfg.get("enable_z_guided_coupling", False)):
            guidance_bundle["lambda_after_z"] = dict(guidance_bundle["lambda_before_z"])
            guidance_bundle["beta_after_z"] = guidance_bundle["beta_before_z"]
            guidance_bundle = apply_soc_band_policy(
                guidance_bundle,
                battery_soc=float(long_term_state.battery_soc),
                cfg=self.simulator_cfg.get("soc_band_policy", {}),
            )
            return guidance_bundle

        cfg = dict(repair_cfg.get("z_guided_coupling", {}))
        reasons: list[str] = []
        dod = max(0.0, 1.0 - float(long_term_state.battery_soc))
        queue_pressure = float(long_term_state.queue_mb) / max(float(long_term_state.queue_max_mb), 1.0e-9)
        thermal_pressure = float(long_term_state.thermal_pressure)
        no_contact = not timeline_step.link_trace.in_contact
        eclipse = timeline_step.resource_trace.is_eclipse

        lambda_E_mult = 1.0
        lambda_Q_mult = 1.0
        lambda_S_mult = 1.0
        lambda_L_mult = 1.0
        beta_mult = 1.0
        if dod >= float(cfg.get("dod_threshold", 0.30)):
            lambda_E_mult *= float(cfg.get("dod_lambda_E_multiplier", 1.35))
            beta_mult *= float(cfg.get("dod_beta_multiplier", 0.85))
            reasons.append("high_dod")
        if eclipse:
            lambda_E_mult *= float(cfg.get("eclipse_lambda_E_multiplier", 1.20))
            lambda_S_mult *= float(cfg.get("eclipse_lambda_S_multiplier", 1.10))
            beta_mult *= float(cfg.get("eclipse_beta_multiplier", 0.90))
            reasons.append("eclipse")
        if thermal_pressure >= float(cfg.get("thermal_threshold", 0.85)):
            lambda_S_mult *= float(cfg.get("thermal_lambda_S_multiplier", 1.35))
            lambda_E_mult *= float(cfg.get("thermal_lambda_E_multiplier", 1.15))
            beta_mult *= float(cfg.get("thermal_beta_multiplier", 0.85))
            reasons.append("high_thermal")
        if queue_pressure >= float(cfg.get("queue_threshold", 0.15)):
            lambda_Q_mult *= float(cfg.get("queue_lambda_Q_multiplier", 1.30))
            lambda_S_mult *= float(cfg.get("queue_lambda_S_multiplier", 1.20))
            reasons.append("queue_pressure")
        if no_contact:
            lambda_S_mult *= float(cfg.get("no_contact_lambda_S_multiplier", 1.25))
            lambda_Q_mult *= float(cfg.get("no_contact_lambda_Q_multiplier", 1.10))
            beta_mult *= float(cfg.get("no_contact_beta_multiplier", 0.90))
            reasons.append("no_contact")

        guidance_bundle["lambda_E"] = float(guidance_bundle.get("lambda_E", 0.0)) * lambda_E_mult
        guidance_bundle["lambda_Q"] = float(guidance_bundle.get("lambda_Q", 0.0)) * lambda_Q_mult
        guidance_bundle["lambda_S"] = float(guidance_bundle.get("lambda_S", 0.0)) * lambda_S_mult
        guidance_bundle["lambda_L"] = float(guidance_bundle.get("lambda_L", 0.0)) * lambda_L_mult
        guidance_bundle["exploration_scale"] = max(
            float(cfg.get("min_beta_scale", 0.35)),
            float(guidance_bundle.get("exploration_scale", 1.0)) * beta_mult,
        )
        guidance_bundle["lambda_after_z"] = {
            "lambda_E": float(guidance_bundle.get("lambda_E", 0.0)),
            "lambda_L": float(guidance_bundle.get("lambda_L", 0.0)),
            "lambda_Q": float(guidance_bundle.get("lambda_Q", 0.0)),
            "lambda_S": float(guidance_bundle.get("lambda_S", 0.0)),
        }
        guidance_bundle["beta_after_z"] = float(guidance_bundle.get("exploration_scale", 1.0))
        guidance_bundle["z_guidance_reason"] = ";".join(reasons) if reasons else "low_risk"
        guidance_bundle = apply_soc_band_policy(
            guidance_bundle,
            battery_soc=float(long_term_state.battery_soc),
            cfg=self.simulator_cfg.get("soc_band_policy", {}),
        )
        return guidance_bundle

    @staticmethod
    def _build_idle_action():
        from src.common.types import JointAction

        return JointAction(
            action_name="no_task",
            model_name="no_task",
            payload_policy="none",
            transmission_mode="idle",
            expected_payload_level=0.0,
            expected_compute_level=0.0,
        )

    @staticmethod
    def _build_zero_reward_breakdown() -> RewardBreakdown:
        return RewardBreakdown(
            effective_utility=0.0,
            resource_penalty=0.0,
            quality_requirement=0.0,
            information_requirement=0.0,
            quality_gap=0.0,
            information_gap=0.0,
            quality_gap_penalty=0.0,
            information_gap_penalty=0.0,
            requirement_penalty=0.0,
            mission_utility=0.0,
            detection_quality=0.0,
            information_sufficiency=0.0,
            timeliness=0.0,
            delivery_factor=0.0,
            dynamic_energy_weight=0.0,
            energy_penalty=0.0,
            latency_penalty=0.0,
            queue_penalty=0.0,
            safety_penalty=0.0,
            mission_error_penalty=0.0,
            reward=0.0,
        )

    @staticmethod
    def _format_task_ids(task_ids: list[str]) -> str:
        return ";".join(task_ids)

    @staticmethod
    def _format_event_times(event_times_s: list[float]) -> str:
        return ";".join(f"{time_s:.6f}".rstrip("0").rstrip(".") for time_s in event_times_s)

    @staticmethod
    def _format_products(products) -> str:
        return ";".join(sorted(str(product) for product in products if str(product)))

    def _queued_product_keys(self) -> set[str]:
        keys: set[str] = set()
        if self.downlink_queue is None:
            return keys
        for item in self.downlink_queue.items:
            product_type = item.product_type or item.payload_type
            keys.add(f"{item.task_id}:{product_type}")
        return keys

    def _ensure_roi_context_trace_state(self) -> None:
        if not hasattr(self, "last_roi_context_window_budget_trace"):
            self.last_roi_context_window_budget_trace = []
        self.last_roi_context_budget_trace = self.last_roi_context_window_budget_trace
        if not hasattr(self, "last_value_density_trace"):
            self.last_value_density_trace = []
        if not hasattr(self, "last_same_slot_vs_queued_followup_trace"):
            self.last_same_slot_vs_queued_followup_trace = []
        if not hasattr(self, "last_future_contact_feasibility_trace"):
            self.last_future_contact_feasibility_trace = []
        if not hasattr(self, "_roi_context_budget_slot_index"):
            self._roi_context_budget_slot_index = -1
        if not hasattr(self, "_roi_context_slot_selected_count"):
            self._roi_context_slot_selected_count = int(getattr(self, "_roi_context_window_selected_count", 0))
        if not hasattr(self, "_roi_context_queued_budget_slot_index"):
            self._roi_context_queued_budget_slot_index = -1
        if not hasattr(self, "_roi_context_queued_selected_count"):
            self._roi_context_queued_selected_count = 0
        if not hasattr(self, "_roi_context_contact_window_id"):
            self._roi_context_contact_window_id = ""
        if not hasattr(self, "_roi_context_contact_window_selected_count"):
            self._roi_context_contact_window_selected_count = int(getattr(self, "_roi_context_window_selected_count", 0))

    def _contact_window_id_for_slot(self, slot_index: int) -> str:
        future_link_traces = getattr(self, "future_link_traces", [])
        if not future_link_traces or slot_index < 0 or slot_index >= len(future_link_traces):
            return f"contact_window_{slot_index}" if slot_index >= 0 else ""
        if not future_link_traces[slot_index].in_contact:
            return ""
        start = slot_index
        while start > 0 and future_link_traces[start - 1].in_contact:
            start -= 1
        return f"contact_window_{start}"

    def _next_contact_window_id(self, start_slot: int, deadline_slot: int) -> str:
        future_link_traces = getattr(self, "future_link_traces", [])
        if not future_link_traces:
            return ""
        start = max(0, min(start_slot, len(future_link_traces) - 1))
        end = max(start, min(deadline_slot, len(future_link_traces) - 1))
        for slot in range(start, end + 1):
            if future_link_traces[slot].in_contact:
                return self._contact_window_id_for_slot(slot)
        return ""

    def _refresh_roi_context_budget_state(self, timeline_step: TimelineStep) -> None:
        self._ensure_roi_context_trace_state()
        slot_index = int(timeline_step.slot_index)
        if self._roi_context_budget_slot_index != slot_index:
            self._roi_context_budget_slot_index = slot_index
            self._roi_context_slot_selected_count = 0
        if self._roi_context_queued_budget_slot_index != slot_index:
            self._roi_context_queued_budget_slot_index = slot_index
            self._roi_context_queued_selected_count = 0
        contact_window_id = self._contact_window_id_for_slot(slot_index) if timeline_step.link_trace.in_contact else ""
        if contact_window_id != self._roi_context_contact_window_id:
            self._roi_context_contact_window_id = contact_window_id
            self._roi_context_contact_window_selected_count = 0
        self._roi_context_window_selected_count = self._roi_context_contact_window_selected_count

    def _roi_context_budget_counts(self, timeline_step: TimelineStep, followup_mode: str) -> dict[str, object]:
        self._refresh_roi_context_budget_state(timeline_step)
        return {
            "followup_mode": followup_mode,
            "slot_selected_count": int(self._roi_context_slot_selected_count),
            "contact_window_id": self._roi_context_contact_window_id,
            "contact_window_selected_count": int(self._roi_context_contact_window_selected_count),
            "queued_followup_selected_count": int(self._roi_context_queued_selected_count),
            "window_selected_count": (
                int(self._roi_context_queued_selected_count)
                if followup_mode == "queued"
                else int(self._roi_context_contact_window_selected_count)
            ),
        }

    def _increment_roi_context_budget(self, timeline_step: TimelineStep, followup_mode: str) -> None:
        self._refresh_roi_context_budget_state(timeline_step)
        self._roi_context_slot_selected_count += 1
        if followup_mode == "queued":
            self._roi_context_queued_selected_count += 1
        else:
            self._roi_context_contact_window_selected_count += 1
            self._roi_context_window_selected_count = self._roi_context_contact_window_selected_count

    def _roi_context_future_contact_feasibility(
        self,
        *,
        timeline_step: TimelineStep,
        task,
        payload_size_mb: float,
        comm_state: CommState,
        start_after_current: bool,
    ) -> dict[str, object]:
        slot = int(timeline_step.slot_index)
        slot_seconds = float(timeline_step.slot_seconds)
        fallback_deadline_s = float(getattr(task, "deadline_s", 0.0) or 0.0)
        remaining_deadline_s = self._remaining_deadline_for_task(task, slot, slot_seconds, fallback_deadline_s)
        deadline_time_s = float(slot) * slot_seconds + remaining_deadline_s
        future_link_traces = getattr(self, "future_link_traces", [])
        if future_link_traces:
            deadline_slot = max(0, min(len(future_link_traces) - 1, int(deadline_time_s // slot_seconds)))
        else:
            deadline_slot = slot
        start_slot = slot + 1 if start_after_current else slot
        known_capacity_mb = self._known_capacity_until_deadline(start_slot, deadline_slot, slot_seconds)
        demand_mb = max(0.0, float(comm_state.queue_mb)) + max(0.0, float(payload_size_mb))
        earliest_slot = self._earliest_slot_for_demand(start_slot, deadline_slot, slot_seconds, demand_mb)
        feasible = (
            remaining_deadline_s > 0.0
            and earliest_slot is not None
            and self._slot_meets_deadline(earliest_slot, deadline_time_s, slot_seconds)
            and known_capacity_mb + float(self.simulator_cfg.get("epsilon", 1.0e-9)) >= demand_mb
        )
        window_id = self._next_contact_window_id(start_slot, deadline_slot)
        result = {
            "slot": slot,
            "task_id": "" if task is None else task.task_id,
            "payload_size_mb": float(payload_size_mb),
            "queue_mb": float(comm_state.queue_mb),
            "queue_max_mb": float(comm_state.queue_max_mb),
            "start_slot": int(start_slot),
            "deadline_slot": int(deadline_slot),
            "remaining_deadline_s": float(remaining_deadline_s),
            "known_contact_capacity_before_deadline_mb": float(known_capacity_mb),
            "required_service_before_deadline_mb": float(demand_mb),
            "future_contact_feasible": int(bool(feasible)),
            "earliest_feasible_delivery_slot": -1 if earliest_slot is None else int(earliest_slot),
            "next_eligible_contact_window_id": window_id,
            "start_after_current": int(bool(start_after_current)),
        }
        self._record_future_contact_feasibility_trace(result)
        return result

    def _record_future_contact_feasibility_trace(self, row: dict[str, object]) -> None:
        self._ensure_roi_context_trace_state()
        self.last_future_contact_feasibility_trace.append(dict(row))

    def _record_same_slot_vs_queued_followup_trace(
        self,
        *,
        timeline_step: TimelineStep,
        task_id: str,
        stage: str,
        followup_mode: str,
        action: str,
        reason_code: str,
        payload_size_mb: float,
        same_slot_delivery_allowed: bool,
        queued_followup_allowed: bool,
        future_contact_feasible: bool,
        enqueue_flag: int = 0,
        delivery_flag: int = 0,
    ) -> None:
        self._ensure_roi_context_trace_state()
        self.last_same_slot_vs_queued_followup_trace.append(
            {
                "slot": int(timeline_step.slot_index),
                "task_id": task_id,
                "stage": stage,
                "followup_mode": followup_mode,
                "action": action,
                "reason_code": reason_code,
                "payload_size_mb": f"{float(payload_size_mb):.6f}",
                "current_in_contact": int(bool(timeline_step.link_trace.in_contact)),
                "same_slot_delivery_allowed": int(bool(same_slot_delivery_allowed)),
                "queued_followup_allowed": int(bool(queued_followup_allowed)),
                "future_contact_feasible": int(bool(future_contact_feasible)),
                "enqueue_flag": int(enqueue_flag),
                "delivery_flag": int(delivery_flag),
            }
        )

    def _record_value_density_trace(
        self,
        *,
        timeline_step: TimelineStep,
        task_id: str,
        stage: str,
        product_type: str,
        residual_delta_u_i: float,
        detection_quality: float,
        timeliness_factor: float,
        payload_size_mb: float,
        value_density: float,
        information_density: float,
        eriv_score: float,
        followup_mode: str = "",
    ) -> None:
        self._ensure_roi_context_trace_state()
        self.last_value_density_trace.append(
            {
                "slot": int(timeline_step.slot_index),
                "task_id": task_id,
                "stage": stage,
                "product_type": product_type,
                "task_effective_value": f"{float(task_effective_value(timeline_step.task)):.6f}"
                if timeline_step.task is not None
                else "",
                "residual_delta_u_i": f"{float(residual_delta_u_i):.6f}",
                "detection_quality": f"{float(detection_quality):.6f}",
                "timeliness_factor": f"{float(timeliness_factor):.6f}",
                "payload_size_mb": f"{float(payload_size_mb):.6f}",
                "information_density": f"{float(information_density):.6f}",
                "value_density": f"{float(value_density):.6f}",
                "ERIV": f"{float(eriv_score):.6f}",
                "followup_mode": followup_mode,
            }
        )

    def _record_duplicate_suppression_trace(
        self,
        *,
        timeline_step: TimelineStep | None,
        task_id: str,
        product_type: str,
        reason_code: str,
        duplicate_source: str,
    ) -> None:
        if not hasattr(self, "last_duplicate_suppression_trace"):
            self.last_duplicate_suppression_trace = []
        self.last_duplicate_suppression_trace.append(
            {
                "slot": -1 if timeline_step is None else int(timeline_step.slot_index),
                "task_id": task_id,
                "product_type": product_type,
                "duplicate_identity_key": self._candidate_key(task_id, product_type),
                "duplicate_source": duplicate_source,
                "reason_code": reason_code,
            }
        )

    def _ensure_staged_delivery_state(self) -> None:
        if not hasattr(self, "lifecycle_ledger"):
            self.lifecycle_ledger = self._init_payload_lifecycle_ledger(getattr(self, "downlink_queue", None))
        if not hasattr(self, "task_snapshots"):
            self.task_snapshots = {}
        if not hasattr(self, "staged_delivery_requests"):
            self.staged_delivery_requests = {}
        if not hasattr(self, "staged_delivery_history"):
            self.staged_delivery_history = {}
        if not hasattr(self, "_staged_delivery_request_seq"):
            self._staged_delivery_request_seq = 0
        if not hasattr(self, "_staged_delivery_contact_window_id"):
            self._staged_delivery_contact_window_id = ""
        if not hasattr(self, "_staged_delivery_contact_window_admission_count"):
            self._staged_delivery_contact_window_admission_count = 0
        if not hasattr(self, "last_staged_delivery_request_trace"):
            self.last_staged_delivery_request_trace = []
        if not hasattr(self, "last_staged_delivery_priority_trace"):
            self.last_staged_delivery_priority_trace = []
        if not hasattr(self, "last_contact_window_scheduling_trace"):
            self.last_contact_window_scheduling_trace = []
        if not hasattr(self, "last_contact_window_candidate_rank_trace"):
            self.last_contact_window_candidate_rank_trace = []
        if not hasattr(self, "last_contact_window_delivery_order_trace"):
            self.last_contact_window_delivery_order_trace = []
        if not hasattr(self, "last_ordinary_queue_vs_staged_queue_capacity_trace"):
            self.last_ordinary_queue_vs_staged_queue_capacity_trace = []
        if not hasattr(self, "last_backlog_composition_trace"):
            self.last_backlog_composition_trace = []
        if not hasattr(self, "last_leftover_tail_value_trace"):
            self.last_leftover_tail_value_trace = []
        if not hasattr(self, "last_guard_admission_trace"):
            self.last_guard_admission_trace = []
        if not hasattr(self, "last_protected_closing_task_trace"):
            self.last_protected_closing_task_trace = []
        if not hasattr(self, "last_contact_capacity_reservation_trace"):
            self.last_contact_capacity_reservation_trace = []
        if not hasattr(self, "_staged_contact_reservations"):
            self._staged_contact_reservations = {}
        self._ensure_guarantee_state()

    def _reset_staged_delivery_step_logs(self) -> None:
        self._ensure_staged_delivery_state()
        self.last_staged_delivery_request_trace = []
        self.last_staged_delivery_priority_trace = []
        self.last_contact_window_scheduling_trace = []
        self.last_contact_window_candidate_rank_trace = []
        self.last_contact_window_delivery_order_trace = []
        self.last_ordinary_queue_vs_staged_queue_capacity_trace = []
        self.last_backlog_composition_trace = []
        self.last_leftover_tail_value_trace = []
        self.last_guard_admission_trace = []
        self.last_protected_closing_task_trace = []
        self.last_contact_capacity_reservation_trace = []
        self._reset_guarantee_step_logs()

    def _staged_delivery_cfg(self) -> dict[str, object]:
        return staged_delivery_queue_config(self.simulator_cfg)

    def _staged_delivery_enabled(self, router: BaseRouter | None) -> bool:
        router_name = "" if router is None else str(getattr(router, "router_name", ""))
        return staged_delivery_enabled_for_router(self.simulator_cfg, router_name)

    def _ensure_guarantee_state(self) -> None:
        if not hasattr(self, "guarantee_debt_by_task"):
            self.guarantee_debt_by_task = {}
        if not hasattr(self, "guarantee_age_debt_by_task"):
            self.guarantee_age_debt_by_task = {}
        if not hasattr(self, "guarantee_miss_debt_by_task"):
            self.guarantee_miss_debt_by_task = {}
        if not hasattr(self, "guarantee_waiting_age_slots_by_task"):
            self.guarantee_waiting_age_slots_by_task = {}
        if not hasattr(self, "_guarantee_last_age_update_slot_by_task"):
            self._guarantee_last_age_update_slot_by_task = {}
        if not hasattr(self, "guarantee_roi_wait_windows"):
            self.guarantee_roi_wait_windows = {}
        if not hasattr(self, "_guarantee_contact_window_id"):
            self._guarantee_contact_window_id = ""
        if not hasattr(self, "_guarantee_window_used_mb"):
            self._guarantee_window_used_mb = 0.0
        if not hasattr(self, "_guarantee_window_observations"):
            self._guarantee_window_observations = {}
        for name in (
            "last_guarantee_debt_trace",
            "last_guarantee_debt_window_update_trace",
            "last_task_debt_trajectory_trace",
            "last_candidate_priority_trace",
            "last_guarantee_opportunity_trace",
            "last_guarantee_progress_trace",
            "last_guarantee_priority_component_trace",
            "last_guarantee_budget_trace",
            "last_guarantee_blocker_reason_trace",
        ):
            if not hasattr(self, name):
                setattr(self, name, [])

    def _reset_guarantee_step_logs(self) -> None:
        self._ensure_guarantee_state()
        self.last_guarantee_debt_trace = []
        self.last_guarantee_debt_window_update_trace = []
        self.last_task_debt_trajectory_trace = []
        self.last_candidate_priority_trace = []
        self.last_guarantee_opportunity_trace = []
        self.last_guarantee_progress_trace = []
        self.last_guarantee_priority_component_trace = []
        self.last_guarantee_budget_trace = []
        self.last_guarantee_blocker_reason_trace = []

    def _guarantee_cfg(self) -> dict[str, object]:
        return guarantee_config(self.simulator_cfg)

    def _guarantee_enabled(self, router: BaseRouter | None) -> bool:
        router_name = "" if router is None else str(getattr(router, "router_name", ""))
        return guarantee_enabled_for_router(self.simulator_cfg, router_name)

    def _guarantee_priority_active(self, router: BaseRouter | None = None) -> bool:
        router_name = "" if router is None else str(getattr(router, "router_name", ""))
        return guarantee_priority_active(self.simulator_cfg, router_name)

    def _guarantee_cap(self, name: str, fallback: float) -> float:
        cfg = self._guarantee_cfg()
        return max(0.0, float(cfg.get(name, cfg.get("z_max", fallback)) or fallback))

    def _sync_guarantee_total_debt(self, task_id: str) -> float:
        self._ensure_guarantee_state()
        task_id = str(task_id)
        legacy = max(0.0, float(self.guarantee_debt_by_task.get(task_id, 0.0) or 0.0))
        has_component_debt = task_id in self.guarantee_age_debt_by_task or task_id in self.guarantee_miss_debt_by_task
        age = max(0.0, float(self.guarantee_age_debt_by_task.get(task_id, 0.0) or 0.0))
        miss = max(0.0, float(self.guarantee_miss_debt_by_task.get(task_id, 0.0) or 0.0))
        if not has_component_debt and age <= 0.0 and miss <= 0.0 and legacy > 0.0:
            miss = min(self._guarantee_cap("z_miss_max", 4.0), legacy)
            self.guarantee_miss_debt_by_task[task_id] = miss
        age = min(self._guarantee_cap("z_age_max", 2.0), age)
        miss = min(self._guarantee_cap("z_miss_max", 4.0), miss)
        total = min(self._guarantee_cap("z_total_max", self._guarantee_cap("z_max", 6.0)), age + miss)
        self.guarantee_age_debt_by_task[task_id] = age
        self.guarantee_miss_debt_by_task[task_id] = miss
        self.guarantee_debt_by_task[task_id] = total
        return total

    def _guarantee_debt_components_by_task(self) -> dict[str, dict[str, float]]:
        self._ensure_guarantee_state()
        task_ids = set(self.guarantee_debt_by_task) | set(self.guarantee_age_debt_by_task) | set(self.guarantee_miss_debt_by_task)
        out: dict[str, dict[str, float]] = {}
        for task_id in task_ids:
            total = self._sync_guarantee_total_debt(task_id)
            out[task_id] = {
                "Z_age": float(self.guarantee_age_debt_by_task.get(task_id, 0.0) or 0.0),
                "Z_miss": float(self.guarantee_miss_debt_by_task.get(task_id, 0.0) or 0.0),
                "Z_total": total,
            }
        return out

    def _clear_guarantee_debt(self, task_id: str) -> tuple[float, float, float]:
        self._ensure_guarantee_state()
        task_id = str(task_id)
        age = float(self.guarantee_age_debt_by_task.get(task_id, 0.0) or 0.0)
        miss = float(self.guarantee_miss_debt_by_task.get(task_id, 0.0) or 0.0)
        total = float(self.guarantee_debt_by_task.get(task_id, 0.0) or 0.0)
        self.guarantee_age_debt_by_task[task_id] = 0.0
        self.guarantee_miss_debt_by_task[task_id] = 0.0
        self.guarantee_debt_by_task[task_id] = 0.0
        self.guarantee_waiting_age_slots_by_task[task_id] = 0
        return age, miss, total

    def _future_contact_before_deadline(self, start_slot: int, deadline_slot: int) -> bool:
        if int(deadline_slot) < int(start_slot):
            return False
        return bool(self._next_contact_window_id(int(start_slot), int(deadline_slot)))

    def _deadline_slot_for_task(self, task_id: str, now_slot: int) -> int:
        task = self.task_snapshots.get(task_id)
        if task is None:
            return int(now_slot)
        return int(
            float(getattr(task, "arrival_slot", now_slot) or now_slot)
            + max(0.0, float(getattr(task, "deadline_s", 0.0) or 0.0)) / max(1.0e-9, float(self.simulator_cfg.get("slot_seconds", 1.0)))
        )

    def _guarantee_waiting_contexts(self, now_slot: int) -> dict[str, dict[str, object]]:
        self._ensure_staged_delivery_state()
        contexts: dict[str, dict[str, object]] = {}
        for request in self.staged_delivery_requests.values():
            if request.status not in {"pending", "blocked", "enqueued"}:
                continue
            task = self.task_snapshots.get(request.task_id)
            value_level = str(request.value_level or getattr(task, "value_level", "") or "").lower()
            product_requirement = str(request.product_requirement or getattr(task, "product_requirement", "") or "").lower()
            contexts[request.task_id] = {
                "task_id": request.task_id,
                "value_level": value_level,
                "product_requirement": product_requirement,
                "lifecycle_status": "queued" if request.status == "enqueued" else request.status,
                "product_type": request.product_type,
                "deadline_slot": int(request.deadline_slot),
                "request_id": request.request_id,
                "duplicate_key": request.duplicate_key,
            }
        for record in getattr(self.lifecycle_ledger, "records", {}).values():
            if record.current_status not in {"pending", "queued", "blocked"}:
                continue
            if record.product_type not in {"roi", "roi_context"}:
                continue
            if record.task_id in contexts:
                continue
            task = self.task_snapshots.get(record.task_id)
            value_level = str(record.value_level or getattr(task, "value_level", "") or "").lower()
            product_requirement = str(record.product_requirement or getattr(task, "product_requirement", "") or "").lower()
            contexts[record.task_id] = {
                "task_id": record.task_id,
                "value_level": value_level,
                "product_requirement": product_requirement,
                "lifecycle_status": record.current_status,
                "product_type": record.product_type,
                "deadline_slot": int(record.deadline_slot if record.deadline_slot >= 0 else self._deadline_slot_for_task(record.task_id, now_slot)),
                "request_id": record.request_id,
                "duplicate_key": self._candidate_key(record.task_id, record.product_type),
            }
        return contexts

    def _archive_terminal_guarantee_debt(self, timeline_step: TimelineStep) -> None:
        self._ensure_guarantee_state()
        now_slot = int(timeline_step.slot_index)
        for task_id in list(set(self.guarantee_debt_by_task) | set(self.guarantee_age_debt_by_task) | set(self.guarantee_miss_debt_by_task)):
            state = self.product_lifecycle.get(task_id)
            stage = stage_from_delivered(set() if state is None else state.delivered_products)
            terminal_reason = ""
            if state is not None and state.task_closed_flag:
                terminal_reason = "closed_debt_cleared"
            elif stage_rank(stage) >= stage_rank("ROI+Ctx"):
                terminal_reason = "target_stage_reached_debt_cleared"
            else:
                for record in getattr(self.lifecycle_ledger, "records", {}).values():
                    if record.task_id == task_id and record.current_status in {"expired", "dropped", "superseded"}:
                        terminal_reason = f"{record.current_status}_with_debt"
                        break
            if not terminal_reason:
                continue
            previous_age, previous_miss, previous_total = self._clear_guarantee_debt(task_id)
            if previous_total <= 0.0 and previous_age <= 0.0 and previous_miss <= 0.0:
                continue
            row = {
                "slot": now_slot,
                "contact_window_id": self._staged_delivery_contact_window_id,
                "task_id": task_id,
                "lifecycle_status": "terminal",
                "stage": stage,
                "Z_age": 0.0,
                "Z_miss": 0.0,
                "Z_total": 0.0,
                "age_debt_increment": 0.0,
                "miss_debt_increment": 0.0,
                "progress_credit": previous_total,
                "debt_update_reason": terminal_reason,
                "future_contact_before_deadline": 0,
                "feasible_opportunity_this_window": 0,
                "progress_made": int("reached" in terminal_reason or "closed" in terminal_reason),
                "selected_product": "",
                "delivered_product": "",
                "final_task_outcome": terminal_reason,
            }
            self.last_task_debt_trajectory_trace.append(row)
            self.last_guarantee_debt_trace.append(dict(row, debt_component="terminal_clear"))

    def _update_guarantee_age_debt_for_waiting_tasks(self, timeline_step: TimelineStep, router: BaseRouter | None) -> None:
        self._ensure_guarantee_state()
        if not self._guarantee_enabled(router):
            return
        cfg = self._guarantee_cfg()
        if not bool(cfg.get("update_age_debt_each_slot", True)):
            return
        now_slot = int(timeline_step.slot_index)
        self._archive_terminal_guarantee_debt(timeline_step)
        contexts = self._guarantee_waiting_contexts(now_slot)
        queued_keys = self._queued_product_keys()
        for task_id, context in contexts.items():
            value_level = str(context.get("value_level", "")).lower()
            product_requirement = str(context.get("product_requirement", "")).lower()
            eligible = is_high_value_guarantee_eligible(value_level, product_requirement, cfg)
            if not eligible:
                if bool(cfg.get("low_medium_debt_forbidden", True)):
                    self.guarantee_age_debt_by_task[str(task_id)] = 0.0
                    self.guarantee_miss_debt_by_task[str(task_id)] = 0.0
                    self.guarantee_debt_by_task[str(task_id)] = 0.0
                continue
            state = self.product_lifecycle.get(str(task_id))
            delivered = set() if state is None else set(state.delivered_products)
            stage = stage_from_delivered(delivered)
            lifecycle_status = str(context.get("lifecycle_status", "pending"))
            deadline_slot = int(context.get("deadline_slot", now_slot))
            future_contact = self._future_contact_before_deadline(now_slot, deadline_slot)
            product_type = str(context.get("product_type", ""))
            duplicate_conflict = bool(product_type and product_type in delivered)
            duplicate_key = str(context.get("duplicate_key", ""))
            if duplicate_key and duplicate_key in queued_keys and lifecycle_status != "queued":
                duplicate_conflict = True
            target_pending = stage_rank(stage) < stage_rank("ROI+Ctx")
            active_status = lifecycle_status in {"pending", "queued", "blocked"}
            last_update = int(self._guarantee_last_age_update_slot_by_task.get(str(task_id), -10**9))
            gamma = float(cfg.get("gamma_age_urgent_per_slot", 0.015) if value_level == "urgent" else cfg.get("gamma_age_high_per_slot", 0.01))
            increment = 0.0
            reason = "age_not_eligible"
            if state is not None and state.task_closed_flag:
                reason = "task_closed"
            elif not target_pending:
                reason = "target_stage_reached"
            elif not active_status:
                reason = "inactive_lifecycle_status"
            elif bool(cfg.get("require_future_contact_for_age_debt", True)) and not future_contact:
                reason = "no_future_contact_before_deadline"
            elif duplicate_conflict:
                reason = "duplicate_lifecycle_conflict"
            elif last_update == now_slot:
                reason = "age_already_updated_this_slot"
            else:
                previous_age = float(self.guarantee_age_debt_by_task.get(str(task_id), 0.0) or 0.0)
                increment = gamma
                self.guarantee_age_debt_by_task[str(task_id)] = min(self._guarantee_cap("z_age_max", 2.0), previous_age + increment)
                self.guarantee_waiting_age_slots_by_task[str(task_id)] = int(self.guarantee_waiting_age_slots_by_task.get(str(task_id), 0)) + 1
                self._guarantee_last_age_update_slot_by_task[str(task_id)] = now_slot
                reason = "waiting_age_debt_increment"
            total = self._sync_guarantee_total_debt(str(task_id))
            row = {
                "slot": now_slot,
                "contact_window_id": self._staged_delivery_contact_window_id,
                "task_id": str(task_id),
                "lifecycle_status": lifecycle_status,
                "stage": stage,
                "Z_age": float(self.guarantee_age_debt_by_task.get(str(task_id), 0.0) or 0.0),
                "Z_miss": float(self.guarantee_miss_debt_by_task.get(str(task_id), 0.0) or 0.0),
                "Z_total": total,
                "age_debt_increment": increment,
                "miss_debt_increment": 0.0,
                "progress_credit": 0.0,
                "debt_update_reason": reason,
                "future_contact_before_deadline": int(future_contact),
                "feasible_opportunity_this_window": 0,
                "progress_made": 0,
                "selected_product": "",
                "delivered_product": "",
                "final_task_outcome": "",
            }
            self.last_task_debt_trajectory_trace.append(row)
            self.last_guarantee_debt_trace.append(dict(row, debt_component="age"))

    def _refresh_staged_delivery_contact_budget(self, timeline_step: TimelineStep) -> None:
        self._ensure_staged_delivery_state()
        contact_window_id = self._contact_window_id_for_slot(int(timeline_step.slot_index)) if timeline_step.link_trace.in_contact else ""
        if contact_window_id != self._staged_delivery_contact_window_id:
            self._staged_delivery_contact_window_id = contact_window_id
            self._staged_delivery_contact_window_admission_count = 0
        if contact_window_id != self._guarantee_contact_window_id:
            self._guarantee_contact_window_id = contact_window_id
            self._guarantee_window_used_mb = 0.0
            self._guarantee_window_observations = {}

    @staticmethod
    def _metadata_get(source: object, name: str, default: object = "") -> object:
        metadata = getattr(source, "metadata", None)
        if isinstance(metadata, dict):
            return metadata.get(name, default)
        return default

    def _staged_delivery_duplicate_exists(self, task_id: str, product_type: str) -> bool:
        self._ensure_staged_delivery_state()
        duplicate_key = self._candidate_key(task_id, product_type)
        active = self.staged_delivery_requests.get(duplicate_key)
        if active is not None and active.status in {"pending", "enqueued", "blocked"}:
            return True
        if duplicate_key in self._queued_product_keys():
            return True
        state = self.product_lifecycle.get(task_id)
        return bool(state is not None and product_type in state.delivered_products)

    @staticmethod
    def _minimum_closing_residual_product(
        requirement: str,
        delivered_products: set[str],
    ) -> str:
        requirement = str(requirement or "verification").strip().lower()
        delivered = {str(product).strip().lower() for product in delivered_products or set()}
        if requirement == "alert":
            return "" if "metadata" in delivered else "metadata"
        if requirement == "verification":
            return "" if ("roi" in delivered or "roi_context" in delivered) else "roi"
        if requirement in {"context", "evidence"}:
            return "" if "roi_context" in delivered else "roi_context"
        return "" if ("roi" in delivered or "roi_context" in delivered) else "roi"

    def _record_staged_request_trace(self, request: StagedDeliveryRequest, action: str, slot: int) -> None:
        self._ensure_staged_delivery_state()
        row = request_to_trace_row(request)
        row.update({"slot": int(slot), "action": action})
        self.last_staged_delivery_request_trace.append(row)

    def _record_staged_priority_trace(
        self,
        request: StagedDeliveryRequest,
        *,
        slot: int,
        queue_order: int,
        selected_for_admission: bool,
    ) -> None:
        self._ensure_staged_delivery_state()
        row = request_to_trace_row(request)
        row.update(
            {
                "slot": int(slot),
                "queue_order": int(queue_order),
                "selected_for_admission": int(bool(selected_for_admission)),
                **{f"component_{key}": value for key, value in request.priority_components.items()},
            }
        )
        self.last_staged_delivery_priority_trace.append(row)
        if hasattr(self, "lifecycle_ledger") and self.lifecycle_ledger is not None:
            self.lifecycle_ledger.record_priority_components(row, source="staged_delivery_queue")

    def _ensure_staged_lifecycle_record(self, request: StagedDeliveryRequest, slot: int) -> str:
        self._ensure_staged_delivery_state()
        record = self.lifecycle_ledger.create_for_staged_request(request, slot=slot, set_active=True)
        if record.current_status == "created":
            self.lifecycle_ledger.transition(
                record.lifecycle_id,
                "pending",
                slot=slot,
                reason="staged_request_waiting_for_contact_window",
                staged_queue_id=request.duplicate_key,
                priority_score=request.priority_score,
                priority_components=request.priority_components,
            )
        return record.lifecycle_id

    def _transition_staged_lifecycle(
        self,
        request: StagedDeliveryRequest,
        status: str,
        *,
        slot: int,
        reason: str,
        blocker_reason: str = "",
        contact_window_id: str = "",
        next_contact_window_id_candidate: str = "",
    ) -> None:
        lifecycle_id = self._ensure_staged_lifecycle_record(request, slot)
        self.lifecycle_ledger.transition(
            lifecycle_id,
            status,
            slot=slot,
            reason=reason,
            blocker_reason=blocker_reason,
            priority_score=request.priority_score,
            priority_components=request.priority_components,
            staged_queue_id=request.duplicate_key,
            contact_window_id=contact_window_id,
            next_contact_window_id_candidate=next_contact_window_id_candidate,
        )

    def _materialize_pending_lifecycle_records(self, timeline_step: TimelineStep) -> None:
        self._ensure_staged_delivery_state()
        now_slot = int(timeline_step.slot_index)
        queued_keys = self._queued_product_keys()
        for record in list(self.lifecycle_ledger.records.values()):
            if record.product_type not in {"roi", "roi_context"}:
                continue
            if record.current_status not in {"pending", "blocked"}:
                continue
            duplicate_key = self._candidate_key(record.task_id, record.product_type)
            if duplicate_key in self.staged_delivery_requests or duplicate_key in queued_keys:
                continue
            task = self.task_snapshots.get(record.task_id)
            state = self.product_lifecycle.get(record.task_id)
            if task is None or state is None:
                continue
            if record.product_type in state.delivered_products:
                self.lifecycle_ledger.transition(
                    record.lifecycle_id,
                    "superseded",
                    slot=now_slot,
                    reason="product_already_delivered",
                    blocker_reason="product_already_delivered",
                )
                continue
            if record.deadline_slot >= 0 and now_slot > record.deadline_slot:
                self.lifecycle_ledger.transition(
                    record.lifecycle_id,
                    "expired",
                    slot=now_slot,
                    reason="deadline_passed_before_recompetition",
                    blocker_reason="deadline_passed",
                )
                continue
            request = StagedDeliveryRequest(
                request_id=record.request_id or f"ledger_{record.lifecycle_id}",
                task_id=record.task_id,
                product_type=record.product_type,
                parent_product_type=record.parent_product_type or "metadata",
                product_requirement=record.product_requirement or state.product_requirement,
                value_level=record.value_level or str(getattr(task, "value_level", "") or "medium").lower(),
                calibrated_task_value=record.calibrated_task_value or float(task_effective_value(task)),
                residual_delta_information_utility=max(0.0, record.residual_delta_information_utility),
                residual_utility_remaining=max(0.0, record.residual_delta_information_utility),
                detection_quality=1.0,
                timeliness_factor=1.0,
                payload_size_mb=max(0.0, record.payload_size_mb),
                deadline_slot=record.deadline_slot if record.deadline_slot >= 0 else self._staged_request_deadline_slot(timeline_step, task),
                created_slot=record.created_slot if record.created_slot >= 0 else now_slot,
                last_evaluated_slot=now_slot,
                earliest_delivery_slot=now_slot,
                status="pending",
                duplicate_key=duplicate_key,
                scene_id=str(getattr(task, "scene_id", "")),
                patch_id=str(self._metadata_get(task, "patch_id", "")),
                deadline_s=float(getattr(task, "deadline_s", 0.0) or 0.0),
                created_time_s=float(record.created_slot if record.created_slot >= 0 else now_slot) * float(timeline_step.slot_seconds),
                lifecycle_id=record.lifecycle_id,
            )
            request.priority_score, request.priority_components = compute_staged_priority(
                request,
                current_slot=now_slot,
                config=self.simulator_cfg,
            )
            self.staged_delivery_requests[duplicate_key] = request
            state.pending_products.add(record.product_type)
            self.lifecycle_ledger.transition(
                record.lifecycle_id,
                "pending",
                slot=now_slot,
                reason="pending_lifecycle_record_materialized_for_recompetition",
                priority_score=request.priority_score,
                priority_components=request.priority_components,
                staged_queue_id=duplicate_key,
                contact_window_id=self._staged_delivery_contact_window_id,
            )

    def _is_high_value_context_residual(self, request: StagedDeliveryRequest) -> bool:
        return (
            request.value_level in {"high", "urgent"}
            and request.product_requirement in {"context", "evidence"}
            and request.product_type in {"roi", "roi_context"}
        )

    def _staged_candidate_feasibility_row(
        self,
        request: StagedDeliveryRequest,
        *,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        queue_order: int,
        available_capacity_mb: float,
    ) -> dict[str, object]:
        now_slot = int(timeline_step.slot_index)
        payload_size_mb = max(0.0, float(request.payload_size_mb))
        deadline_feasible = now_slot <= int(request.deadline_slot)
        min_remaining_deadline_s = float(self._followup_guard_cfg()["min_remaining_deadline_s"])
        remaining_deadline_s = max(0.0, (float(request.deadline_slot) - float(now_slot)) * float(timeline_step.slot_seconds))
        deadline_guard_feasible = remaining_deadline_s > min_remaining_deadline_s
        capacity_feasible = bool(timeline_step.link_trace.in_contact) and payload_size_mb <= available_capacity_mb + float(
            self.simulator_cfg.get("epsilon", 1.0e-9)
        )
        future_contact = self._roi_context_future_contact_feasibility(
            timeline_step=timeline_step,
            task=self.task_snapshots.get(request.task_id),
            payload_size_mb=payload_size_mb,
            comm_state=comm_state,
            start_after_current=not bool(timeline_step.link_trace.in_contact),
        )
        future_capacity_feasible = bool(future_contact.get("future_contact_feasible", 0))
        queue_feasible = float(comm_state.queue_mb) + payload_size_mb <= float(comm_state.queue_max_mb) + float(
            self.simulator_cfg.get("epsilon", 1.0e-9)
        )
        lifecycle_state = self.product_lifecycle.get(request.task_id)
        lifecycle_open = lifecycle_state is None or not bool(lifecycle_state.task_closed_flag)
        duplicate_key = self._candidate_key(request.task_id, request.product_type)
        duplicate_feasible = True
        if lifecycle_state is not None and request.product_type in lifecycle_state.delivered_products:
            duplicate_feasible = False
        if duplicate_key in self._queued_product_keys():
            duplicate_feasible = False
        extra_comm_energy_j = self._predict_extra_payload_comm_energy_j(
            timeline_step=timeline_step,
            payload_mb=payload_size_mb,
            require_same_slot_delivery=True,
            predict_immediate_downlink=True,
            remaining_contact_capacity_mb=available_capacity_mb,
        )
        predicted_battery_next = update_battery_soc(
            battery_soc=resource_state.battery_soc,
            solar_energy_j=timeline_step.resource_trace.solar_energy_j,
            base_power_w=timeline_step.resource_trace.base_power_w,
            slot_seconds=timeline_step.slot_seconds,
            compute_energy_j=0.0,
            comm_energy_j=extra_comm_energy_j,
            battery_capacity_j=float(self.simulator_cfg.get("battery_capacity_j", 1.0e12)),
            charge_efficiency=float(self.simulator_cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(self.simulator_cfg.get("battery_discharge_efficiency", 1.0)),
        )
        predicted_thermal_next = self._estimate_next_thermal(
            timeline_step=timeline_step,
            resource_state=resource_state,
            compute_energy_j=0.0,
            comm_energy_j=extra_comm_energy_j,
            compute_heat_gain_scale=1.0,
        )
        battery_feasible = predicted_battery_next >= float(self.simulator_cfg.get("battery_min", 0.0)) - float(
            self.simulator_cfg.get("epsilon", 1.0e-9)
        )
        thermal_feasible = predicted_thermal_next <= float(self.simulator_cfg.get("thermal_max", 1.0)) + float(
            self.simulator_cfg.get("epsilon", 1.0e-9)
        )
        feasible = (
            bool(timeline_step.link_trace.in_contact)
            and capacity_feasible
            and deadline_feasible
            and deadline_guard_feasible
            and queue_feasible
            and battery_feasible
            and thermal_feasible
            and lifecycle_open
            and duplicate_feasible
        )
        future_generation_feasible = (
            future_capacity_feasible
            and deadline_feasible
            and deadline_guard_feasible
            and queue_feasible
            and battery_feasible
            and thermal_feasible
            and lifecycle_open
            and duplicate_feasible
        )
        return {
            "slot": now_slot,
            "time_s": float(now_slot) * float(timeline_step.slot_seconds),
            "contact_window_id": self._staged_delivery_contact_window_id,
            "request_id": request.request_id,
            "task_id": request.task_id,
            "product_type": request.product_type,
            "value_level": request.value_level,
            "product_requirement": request.product_requirement,
            "calibrated_task_value": float(request.calibrated_task_value),
            "priority_score": float(request.priority_score),
            "payload_size_mb": payload_size_mb,
            "deadline_slot": int(request.deadline_slot),
            "queue_order": int(queue_order),
            "status_before": request.status,
            "in_contact": int(bool(timeline_step.link_trace.in_contact)),
            "available_capacity_mb": float(available_capacity_mb),
            "queue_mb": float(comm_state.queue_mb),
            "capacity_feasible": int(capacity_feasible),
            "future_capacity_feasible": int(future_capacity_feasible),
            "known_contact_capacity_before_deadline_mb": float(
                future_contact.get("known_contact_capacity_before_deadline_mb", 0.0) or 0.0
            ),
            "required_service_before_deadline_mb": float(
                future_contact.get("required_service_before_deadline_mb", 0.0) or 0.0
            ),
            "earliest_feasible_delivery_slot": int(future_contact.get("earliest_feasible_delivery_slot", -1) or -1),
            "next_eligible_contact_window_id": str(future_contact.get("next_eligible_contact_window_id", "")),
            "deadline_feasible": int(deadline_feasible),
            "deadline_guard_feasible": int(deadline_guard_feasible),
            "queue_feasible": int(queue_feasible),
            "battery_feasible": int(battery_feasible),
            "thermal_feasible": int(thermal_feasible),
            "lifecycle_open": int(lifecycle_open),
            "duplicate_feasible": int(duplicate_feasible),
            "feasible": int(feasible),
            "future_generation_feasible": int(future_generation_feasible),
            "high_value_context_residual_flag": int(self._is_high_value_context_residual(request)),
            "predicted_battery_next": float(predicted_battery_next),
            "predicted_thermal_next": float(predicted_thermal_next),
            "remaining_deadline_s": float(remaining_deadline_s),
        }

    @staticmethod
    def _staged_string_set(value: object) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            raw_items = value.replace(";", ",").split(",")
        else:
            try:
                raw_items = list(value)  # type: ignore[arg-type]
            except TypeError:
                raw_items = [value]
        return {str(item).strip() for item in raw_items if str(item).strip()}

    def _request_disable_replay_ids(self) -> set[str]:
        cfg = self._staged_delivery_cfg()
        if not bool(cfg.get("enable_request_disable_replay", False)):
            return set()
        disabled = set()
        disabled.update(self._staged_string_set(cfg.get("request_disable_replay_ids", [])))
        disabled.update(self._staged_string_set(cfg.get("disabled_persistent_request_ids", [])))
        return disabled

    def _disable_staged_request_for_replay(self, request: StagedDeliveryRequest, slot: int) -> None:
        self._ensure_staged_delivery_state()
        previous_status = request.status
        request.status = "dropped"
        request.blocker_reason = "disabled_by_request_replay"
        state = self.product_lifecycle.get(request.task_id)
        if state is not None:
            state.pending_products.discard(request.product_type)
        self._transition_staged_lifecycle(
            request,
            "dropped",
            slot=slot,
            reason="disabled_by_request_replay",
            blocker_reason="disabled_by_request_replay",
            contact_window_id=self._staged_delivery_contact_window_id,
        )
        self.staged_delivery_history[request.duplicate_key] = request
        self.staged_delivery_requests.pop(request.duplicate_key, None)
        self._record_staged_request_trace(request, "disabled_replay", slot)
        row = request_to_trace_row(request)
        row.update(
            {
                "slot": int(slot),
                "time_s": float(slot) * float(self.simulator_cfg.get("slot_seconds", 1.0)),
                "decision": "CANCEL_STALE",
                "rejection_reason": "disabled_by_request_replay",
                "previous_status": previous_status,
                "disabled_request_ids": ";".join(sorted(self._request_disable_replay_ids())),
            }
        )
        self.last_guard_admission_trace.append(row)

    def _deadline_slot_from_time(self, deadline_time_s: float, slot_seconds: float) -> int:
        future_link_traces = getattr(self, "future_link_traces", [])
        raw_slot = int(max(0.0, float(deadline_time_s)) // max(slot_seconds, 1.0e-9))
        if future_link_traces:
            return max(0, min(len(future_link_traces) - 1, raw_slot))
        return raw_slot

    def _protected_closing_tasks(
        self,
        *,
        timeline_step: TimelineStep,
        comm_state: CommState,
    ) -> list[dict[str, object]]:
        self._ensure_staged_delivery_state()
        if self.downlink_queue is None:
            return []
        now_slot = int(timeline_step.slot_index)
        slot_seconds = float(timeline_step.slot_seconds)
        start_slot = now_slot if bool(timeline_step.link_trace.in_contact) else now_slot + 1
        protected: list[dict[str, object]] = []
        for item in getattr(self.downlink_queue, "items", []):
            task_id = str(item.task_id)
            state = self.product_lifecycle.get(task_id)
            task = self.task_snapshots.get(task_id)
            if state is None or task is None or bool(state.task_closed_flag):
                continue
            if float(item.deadline_time_s) < float(now_slot) * slot_seconds - float(self.simulator_cfg.get("epsilon", 1.0e-9)):
                continue
            product_type = str(item.product_type or item.payload_type)
            closing_product = self._minimum_closing_residual_product(state.product_requirement, set(state.delivered_products))
            if not closing_product or product_type != closing_product:
                continue
            remaining_mb = max(0.0, float(item.remaining_mb))
            expected_utility = max(0.0, float(item.remaining_utility or 0.0))
            if remaining_mb <= 0.0 or expected_utility <= 0.0:
                continue
            deadline_slot = self._deadline_slot_from_time(float(item.deadline_time_s), slot_seconds)
            known_capacity_mb = self._known_capacity_until_deadline(start_slot, deadline_slot, slot_seconds)
            earliest_slot = self._earliest_slot_for_demand(start_slot, deadline_slot, slot_seconds, remaining_mb)
            if earliest_slot is None or known_capacity_mb + float(self.simulator_cfg.get("epsilon", 1.0e-9)) < remaining_mb:
                continue
            contact_id = self._next_contact_window_id(start_slot, deadline_slot)
            row = {
                "slot": now_slot,
                "time_s": float(now_slot) * slot_seconds,
                "task_id": task_id,
                "value_level": str(getattr(task, "value_level", "") or self._metadata_get(task, "value_level", "")),
                "deadline_slot": deadline_slot,
                "remaining_closing_bytes": remaining_mb * 1_000_000.0,
                "remaining_closing_mb": remaining_mb,
                "contacts_capable_of_finishing_it": contact_id,
                "expected_closure_utility": expected_utility,
                "protection_start_time_s": float(now_slot) * slot_seconds,
                "protection_end_time_s": float(item.deadline_time_s),
                "known_contact_capacity_before_deadline_mb": known_capacity_mb,
                "earliest_feasible_delivery_slot": earliest_slot,
                "product_type": product_type,
            }
            protected.append(row)
            self.last_protected_closing_task_trace.append(dict(row))
        return protected

    def _candidate_expected_benefit(self, request: StagedDeliveryRequest) -> float:
        return max(0.0, float(request.calibrated_task_value)) * max(0.0, float(request.residual_delta_information_utility))

    def _candidate_direct_comm_cost_j(
        self,
        request: StagedDeliveryRequest,
        *,
        timeline_step: TimelineStep,
        available_capacity_mb: float,
    ) -> float:
        return self._predict_extra_payload_comm_energy_j(
            timeline_step=timeline_step,
            payload_mb=float(request.payload_size_mb),
            require_same_slot_delivery=False,
            predict_immediate_downlink=True,
            remaining_contact_capacity_mb=available_capacity_mb,
        ) / 3600.0

    def _closure_guard_decision(
        self,
        request: StagedDeliveryRequest,
        *,
        timeline_step: TimelineStep,
        comm_state: CommState,
        feasibility_row: dict[str, object],
        available_capacity_mb: float,
    ) -> dict[str, object]:
        cfg = self._staged_delivery_cfg()
        protect_enabled = bool(cfg.get("enable_close_task_protection", False))
        net_guard_enabled = bool(cfg.get("enable_displacement_net_guard", False))
        if not protect_enabled and not net_guard_enabled:
            return {"decision": "ADMIT", "rejection_reason": "", "estimated_net_gain": self._candidate_expected_benefit(request)}
        protected = self._protected_closing_tasks(timeline_step=timeline_step, comm_state=comm_state)
        candidate_contact = str(feasibility_row.get("next_eligible_contact_window_id", "")) or self._staged_delivery_contact_window_id
        affected: list[dict[str, object]] = []
        candidate_bytes_mb = max(0.0, float(request.payload_size_mb))
        for row in protected:
            protected_contact = str(row.get("contacts_capable_of_finishing_it", ""))
            if protected_contact and candidate_contact and protected_contact != candidate_contact:
                continue
            protected_deadline_slot = int(row.get("deadline_slot", int(request.deadline_slot)) or int(request.deadline_slot))
            start_slot = int(timeline_step.slot_index) if bool(timeline_step.link_trace.in_contact) else int(timeline_step.slot_index) + 1
            capacity_mb = self._known_capacity_until_deadline(start_slot, protected_deadline_slot, float(timeline_step.slot_seconds))
            required_without_candidate = max(0.0, float(comm_state.queue_mb))
            required_with_candidate = required_without_candidate + candidate_bytes_mb
            protected_loses_closure = required_without_candidate <= capacity_mb + float(self.simulator_cfg.get("epsilon", 1.0e-9)) and required_with_candidate > capacity_mb + float(self.simulator_cfg.get("epsilon", 1.0e-9))
            if protected_loses_closure:
                affected.append(row)
        displacement_loss = sum(float(row.get("expected_closure_utility", 0.0) or 0.0) for row in affected)
        benefit = self._candidate_expected_benefit(request)
        direct_generation_cost = 0.0
        direct_comm_cost = self._candidate_direct_comm_cost_j(request, timeline_step=timeline_step, available_capacity_mb=available_capacity_mb)
        estimated_net_gain = benefit - direct_generation_cost - direct_comm_cost - displacement_loss
        decision = "ADMIT"
        reason = ""
        if affected and protect_enabled and benefit <= displacement_loss + float(self.simulator_cfg.get("epsilon", 1.0e-9)):
            decision = "TEMPORARILY_BLOCK"
            reason = "close_task_protection_displacement"
        elif net_guard_enabled and estimated_net_gain <= float(self.simulator_cfg.get("epsilon", 1.0e-9)):
            decision = "REJECT_NON_POSITIVE_NET_GAIN"
            reason = "non_positive_estimated_net_gain"
        row = {
            "slot": int(timeline_step.slot_index),
            "time_s": float(timeline_step.slot_index) * float(timeline_step.slot_seconds),
            "contact_window_id": self._staged_delivery_contact_window_id,
            "request_id": request.request_id,
            "task_id": request.task_id,
            "product_type": request.product_type,
            "expected_closing_bytes": candidate_bytes_mb * 1_000_000.0,
            "expected_direct_utility_gain": benefit,
            "expected_generation_cost": direct_generation_cost,
            "expected_communication_cost": direct_comm_cost,
            "protected_tasks_affected": self._format_task_ids([str(row.get("task_id", "")) for row in affected]),
            "estimated_displacement_loss": displacement_loss,
            "estimated_net_gain": estimated_net_gain,
            "contacts_considered": candidate_contact,
            "reserved_bytes": 0.0,
            "decision": decision,
            "rejection_reason": reason,
            "later_reconsideration_result": "",
        }
        self.last_guard_admission_trace.append(row)
        return row
    def _record_guarantee_priority_trace(self, request: StagedDeliveryRequest, *, slot: int, queue_order: int) -> None:
        self._ensure_guarantee_state()
        cfg = self._guarantee_cfg()
        if not bool(cfg.get("enabled", False)):
            return
        row = {
            "slot": int(slot),
            "request_id": request.request_id,
            "task_id": request.task_id,
            "product_type": request.product_type,
            "value_level": request.value_level,
            "product_requirement": request.product_requirement,
            "queue_order": int(queue_order),
            "priority_score": float(request.priority_score),
            "guarantee_debt_z": float(request.priority_components.get("guarantee_debt_z", 0.0) or 0.0),
            "Z_age": float(request.priority_components.get("Z_age", 0.0) or 0.0),
            "Z_miss": float(request.priority_components.get("Z_miss", 0.0) or 0.0),
            "Z_total": float(request.priority_components.get("Z_total", request.priority_components.get("guarantee_debt_z", 0.0)) or 0.0),
            "guarantee_progress_gate": float(request.priority_components.get("guarantee_progress_gate", 0.0) or 0.0),
            "G_i_p": float(request.priority_components.get("G_i_p", request.priority_components.get("guarantee_progress_gate", 0.0)) or 0.0),
            "guarantee_term": float(request.priority_components.get("guarantee_term", 0.0) or 0.0),
            "priority_without_guarantee": float(
                request.priority_components.get("priority_without_guarantee", request.priority_score) or 0.0
            ),
            "final_priority_without_guarantee": float(
                request.priority_components.get("final_priority_without_guarantee", request.priority_components.get("priority_without_guarantee", request.priority_score)) or 0.0
            ),
            "final_priority_with_guarantee": float(
                request.priority_components.get("final_priority_with_guarantee", request.priority_score) or 0.0
            ),
            "rank_without_guarantee": int(request.priority_components.get("rank_without_guarantee", queue_order + 1) or 0),
            "rank_with_guarantee": int(request.priority_components.get("rank_with_guarantee", queue_order + 1) or 0),
            "rank_gain": int(request.priority_components.get("rank_gain", 0) or 0),
            "waiting_age_slots": int(request.priority_components.get("waiting_age_slots", 0) or 0),
            "base_priority_score": float(request.priority_components.get("base_priority_score", request.priority_score) or 0.0),
            "guarantee_assisted_candidate": int(
                bool(request.priority_components.get("guarantee_assisted_candidate", 0))
            ),
            **{f"component_{key}": value for key, value in request.priority_components.items()},
        }
        self.last_guarantee_priority_component_trace.append(row)

    def _record_guarantee_opportunity(
        self,
        request: StagedDeliveryRequest,
        feasibility_row: dict[str, object],
        *,
        slot: int,
    ) -> None:
        self._ensure_guarantee_state()
        cfg = self._guarantee_cfg()
        if not bool(cfg.get("enabled", False)):
            return
        state = self.product_lifecycle.get(request.task_id)
        delivered = set() if state is None else set(state.delivered_products)
        eligible = is_high_value_guarantee_eligible(request.value_level, request.product_requirement, cfg)
        current_stage = stage_from_delivered(delivered)
        feasible = bool(feasibility_row.get("feasible", 0))
        contact = bool(feasibility_row.get("in_contact", 0))
        progress_gate = float(request.priority_components.get("guarantee_progress_gate", 0.0) or 0.0)
        row = {
            "slot": int(slot),
            "contact_window_id": self._staged_delivery_contact_window_id,
            "request_id": request.request_id,
            "task_id": request.task_id,
            "product_type": request.product_type,
            "value_level": request.value_level,
            "product_requirement": request.product_requirement,
            "stage_at_window_eval": current_stage,
            "delivered_products": self._format_products(delivered),
            "eligible_high_value_flag": int(eligible),
            "feasible_i_w": int(feasible),
            "contact_opportunity_flag": int(contact),
            "progress_gate": progress_gate,
            "Z_age": float(self.guarantee_age_debt_by_task.get(request.task_id, 0.0) or 0.0),
            "Z_miss": float(self.guarantee_miss_debt_by_task.get(request.task_id, 0.0) or 0.0),
            "Z_total": float(self.guarantee_debt_by_task.get(request.task_id, 0.0) or 0.0),
            "payload_size_mb": float(request.payload_size_mb),
            "deadline_slot": int(request.deadline_slot),
            "capacity_feasible": int(feasibility_row.get("capacity_feasible", 0)),
            "queue_feasible": int(feasibility_row.get("queue_feasible", 0)),
            "battery_feasible": int(feasibility_row.get("battery_feasible", 0)),
            "thermal_feasible": int(feasibility_row.get("thermal_feasible", 0)),
            "duplicate_feasible": int(feasibility_row.get("duplicate_feasible", 0)),
            "deadline_feasible": int(feasibility_row.get("deadline_feasible", 0)),
        }
        self.last_guarantee_opportunity_trace.append(row)
        if not eligible or not feasible or not contact:
            return
        observed = self._guarantee_window_observations.setdefault(
            request.task_id,
            {
                "task_id": request.task_id,
                "value_level": request.value_level,
                "product_requirement": request.product_requirement,
                "stage_start_rank": stage_rank(current_stage),
                "stage_start": current_stage,
                "min_opportunity": False,
                "target_opportunity": False,
                "opportunity_count": 0,
                "contact_window_id": self._staged_delivery_contact_window_id,
                "deadline_slot": int(request.deadline_slot),
            },
        )
        observed["opportunity_count"] = int(observed.get("opportunity_count", 0)) + 1
        observed["stage_start_rank"] = min(int(observed.get("stage_start_rank", stage_rank(current_stage))), stage_rank(current_stage))
        if stage_rank(current_stage) < stage_rank("ROI") and request.product_type == "roi":
            observed["min_opportunity"] = True
        if stage_rank(current_stage) == stage_rank("ROI") and request.product_type == "roi_context" and progress_gate > 0.0:
            observed["target_opportunity"] = True

    def _record_guarantee_blocker(
        self,
        request: StagedDeliveryRequest,
        *,
        slot: int,
        reason_code: str,
        decision: str,
    ) -> None:
        self._ensure_guarantee_state()
        self.last_guarantee_blocker_reason_trace.append(
            {
                "slot": int(slot),
                "contact_window_id": self._staged_delivery_contact_window_id,
                "request_id": request.request_id,
                "task_id": request.task_id,
                "product_type": request.product_type,
                "reason_code": reason_code,
                "decision": decision,
                "guarantee_debt_z": float(request.priority_components.get("guarantee_debt_z", 0.0) or 0.0),
                "guarantee_progress_gate": float(request.priority_components.get("guarantee_progress_gate", 0.0) or 0.0),
                "guarantee_term": float(request.priority_components.get("guarantee_term", 0.0) or 0.0),
            }
        )

    def _candidate_priority_trace_row(
        self,
        request: StagedDeliveryRequest,
        feasibility_row: dict[str, object],
        *,
        slot: int,
        order: int,
        capacity_cutoff_rank: int,
        selected: bool,
        delivered: bool = False,
        blocker_reason: str = "",
        lifecycle_status_after: str = "",
    ) -> dict[str, object]:
        state = self.product_lifecycle.get(request.task_id)
        delivered_products = set() if state is None else set(state.delivered_products)
        stage_before = stage_from_delivered(delivered_products)
        components = request.priority_components
        z_total = float(components.get("Z_total", components.get("guarantee_debt_z", 0.0)) or 0.0)
        g_i_p = float(components.get("G_i_p", components.get("guarantee_progress_gate", 0.0)) or 0.0)
        delta_information = float(request.residual_delta_information_utility)
        payload_size = max(1.0e-9, float(request.payload_size_mb))
        blocked_by_deadline = not bool(feasibility_row.get("deadline_feasible", 1)) or not bool(
            feasibility_row.get("deadline_guard_feasible", 1)
        )
        blocked_by_capacity = (not bool(feasibility_row.get("capacity_feasible", 1))) or (
            bool(feasibility_row.get("in_contact", 0)) and order + 1 > max(0, int(capacity_cutoff_rank))
        )
        blocked_by_safety = not bool(feasibility_row.get("battery_feasible", 1)) or not bool(
            feasibility_row.get("thermal_feasible", 1)
        )
        blocked_by_duplicate = not bool(feasibility_row.get("duplicate_feasible", 1))
        blocked_by_budget = blocker_reason == "guarantee_budget_limited"
        blocked_by_floor = blocker_reason == "marginal_utility_floor_blocked"
        return {
            "slot": int(slot),
            "contact_window_id": self._staged_delivery_contact_window_id,
            "scheduler_step_id": int(order),
            "task_id": request.task_id,
            "product_type": request.product_type,
            "value_level": request.value_level,
            "product_requirement": request.product_requirement,
            "in_H_flag": int(self._is_high_value_context_residual(request)),
            "lifecycle_status_before": str(feasibility_row.get("status_before", request.status)),
            "stage_before": stage_before,
            "payload_size_mb": float(request.payload_size_mb),
            "delta_information_utility": delta_information,
            "deltaU_total": float(components.get("absolute_delta_utility", delta_information) or 0.0),
            "utility_density_per_MB": delta_information / payload_size,
            "Z_age": float(components.get("Z_age", 0.0) or 0.0),
            "Z_miss": float(components.get("Z_miss", 0.0) or 0.0),
            "Z_total": z_total,
            "G_i_p": g_i_p,
            "guarantee_progress_gate": g_i_p,
            "guarantee_term": float(components.get("guarantee_term", 0.0) or 0.0),
            "waiting_age_slots": int(components.get("waiting_age_slots", 0) or 0),
            "deadline_urgency": float(components.get("deadline_urgency", 0.0) or 0.0),
            "closure_progress_bonus": float(components.get("closure_progress_bonus", components.get("product_requirement_weight", 0.0)) or 0.0),
            "final_priority_without_guarantee": float(
                components.get("final_priority_without_guarantee", components.get("priority_without_guarantee", request.priority_score)) or 0.0
            ),
            "final_priority_with_guarantee": float(
                components.get("final_priority_with_guarantee", request.priority_score) or 0.0
            ),
            "rank_without_guarantee": int(components.get("rank_without_guarantee", order + 1) or 0),
            "rank_with_guarantee": int(components.get("rank_with_guarantee", order + 1) or 0),
            "rank_gain": int(components.get("rank_gain", 0) or 0),
            "capacity_cutoff_rank": int(capacity_cutoff_rank),
            "selected": int(bool(selected)),
            "delivered": int(bool(delivered)),
            "guarantee_assisted_flag": int(bool(components.get("guarantee_assisted_candidate", 0))),
            "blocked_by_safety": int(blocked_by_safety),
            "blocked_by_capacity": int(blocked_by_capacity),
            "blocked_by_deadline": int(blocked_by_deadline),
            "blocked_by_duplicate": int(blocked_by_duplicate),
            "blocked_by_marginal_utility_floor": int(blocked_by_floor),
            "blocked_by_budget": int(blocked_by_budget),
            "blocker_reason": blocker_reason,
            "lifecycle_status_after": lifecycle_status_after or request.status,
            "stage_after": stage_from_delivered(set() if state is None else set(state.delivered_products)),
        }

    def _append_candidate_priority_trace(self, row: dict[str, object]) -> None:
        self._ensure_guarantee_state()
        self.last_candidate_priority_trace.append(row)

    def _block_staged_request_for_guarantee(
        self,
        request: StagedDeliveryRequest,
        *,
        slot: int,
        order: int,
        reason_code: str,
        decision: str,
        feasibility_row: dict[str, object],
    ) -> None:
        previous_status = request.status
        request.status = "blocked"
        request.blocker_reason = reason_code
        self._transition_staged_lifecycle(
            request,
            "blocked",
            slot=slot,
            reason=reason_code,
            blocker_reason=reason_code,
            contact_window_id=self._staged_delivery_contact_window_id,
            next_contact_window_id_candidate=self._next_contact_window_id(slot + 1, int(request.deadline_slot)),
        )
        self.lifecycle_ledger.record_recompetition(
            slot=slot,
            contact_window_id=self._staged_delivery_contact_window_id,
            task_id=request.task_id,
            product_type=request.product_type,
            previous_status=previous_status,
            reevaluated_flag=True,
            feasible_this_window=bool(feasibility_row.get("feasible", 0)),
            priority_score=float(request.priority_score),
            priority_rank=order,
            selected_for_delivery=False,
            delivered=False,
            blocker_reason=reason_code,
            status_after_window=request.status,
        )
        self.last_contact_window_scheduling_trace.append(
            {
                "slot": slot,
                "time_s": float(slot) * float(self.simulator_cfg.get("slot_seconds", 1.0)),
                "contact_window_id": self._staged_delivery_contact_window_id,
                "request_id": request.request_id,
                "task_id": request.task_id,
                "product_type": request.product_type,
                "priority_score": request.priority_score,
                "status_before": previous_status,
                "decision": decision,
                "reason_code": reason_code,
                "in_contact": int(bool(feasibility_row.get("in_contact", 0))),
                "queue_mb": float(feasibility_row.get("queue_mb", 0.0) or 0.0),
                "guarantee_assisted": int(bool(request.priority_components.get("guarantee_assisted_candidate", 0))),
            }
        )
        self._record_guarantee_blocker(request, slot=slot, reason_code=reason_code, decision=decision)

    def _guarantee_budget_limit_mb(self, available_capacity_mb: float) -> float:
        cfg = self._guarantee_cfg()
        return max(0.0, float(available_capacity_mb)) * max(0.0, float(cfg.get("guarantee_window_budget_fraction", 0.30) or 0.0))

    def _contact_window_ends_at_step(self, timeline_step: TimelineStep) -> bool:
        if not bool(timeline_step.link_trace.in_contact):
            return False
        next_slot = int(timeline_step.slot_index) + 1
        future_link_traces = getattr(self, "future_link_traces", [])
        if next_slot >= len(future_link_traces):
            return True
        return not bool(future_link_traces[next_slot].in_contact)

    def _update_guarantee_debt_at_contact_window_end(self, timeline_step: TimelineStep, router: BaseRouter | None) -> None:
        self._ensure_guarantee_state()
        if not self._guarantee_enabled(router):
            return
        if not self._contact_window_ends_at_step(timeline_step):
            return
        cfg = self._guarantee_cfg()
        if not bool(cfg.get("update_miss_debt_at_contact_window_end", True)):
            self._guarantee_window_observations = {}
            return
        now_slot = int(timeline_step.slot_index)
        for task_id, observed in list(self._guarantee_window_observations.items()):
            state = self.product_lifecycle.get(task_id)
            delivered = set() if state is None else set(state.delivered_products)
            stage_after = stage_from_delivered(delivered)
            stage_start_rank = int(observed.get("stage_start_rank", 0))
            after_rank = stage_rank(stage_after)
            min_progress = stage_start_rank < stage_rank("ROI") and after_rank >= stage_rank("ROI")
            target_progress = stage_start_rank < stage_rank("ROI+Ctx") and after_rank >= stage_rank("ROI+Ctx")
            min_miss = bool(observed.get("min_opportunity", False)) and not min_progress
            target_opportunity = bool(observed.get("target_opportunity", False))
            target_miss = target_opportunity and not target_progress
            deadline_slot = int(observed.get("deadline_slot", now_slot))
            future_contact = self._future_contact_before_deadline(now_slot + 1, deadline_slot)
            previous_age = float(self.guarantee_age_debt_by_task.get(task_id, 0.0) or 0.0)
            previous_miss = float(self.guarantee_miss_debt_by_task.get(task_id, 0.0) or 0.0)
            previous_debt = self._sync_guarantee_total_debt(task_id)
            progress_credit = 0.0
            if target_progress:
                progress_credit = float(cfg.get("progress_credit_target_stage", 2.0) or 0.0)
            elif min_progress:
                progress_credit = float(cfg.get("progress_credit_min_stage", 1.5) or 0.0)
            min_weight = float(cfg.get("min_stage_miss_weight", 1.0) or 1.0)
            target_weight = float(cfg.get("target_stage_miss_weight", 0.5) or 0.5)
            miss_weight = (min_weight if min_miss else 0.0) + (target_weight if target_miss else 0.0)
            value_level = str(observed.get("value_level", "high")).lower()
            gamma_miss = float(cfg.get("gamma_miss_urgent", 1.25) if value_level == "urgent" else cfg.get("gamma_miss_high", 1.0))
            increment = gamma_miss * miss_weight if future_contact else 0.0
            new_age = previous_age
            new_miss = min(self._guarantee_cap("z_miss_max", 4.0), previous_miss + increment)
            if min_progress:
                new_miss = max(0.0, new_miss - progress_credit)
                new_age = max(0.0, new_age - 1.0)
                if bool(cfg.get("reset_debt_when_min_stage_reached", False)):
                    new_age = 0.0
                    new_miss = 0.0
            if target_progress:
                new_age = 0.0
                new_miss = 0.0
            if bool(cfg.get("reset_debt_when_task_closed", True)) and (target_progress or (state is not None and state.task_closed_flag)):
                new_age = 0.0
                new_miss = 0.0
            self.guarantee_age_debt_by_task[task_id] = min(self._guarantee_cap("z_age_max", 2.0), new_age)
            self.guarantee_miss_debt_by_task[task_id] = min(self._guarantee_cap("z_miss_max", 4.0), new_miss)
            new_debt = self._sync_guarantee_total_debt(task_id)
            if (min_miss or target_miss) and not future_contact:
                reason = "non_rescuable_no_future_contact"
            elif increment > 0.0:
                reason = "missed_opportunity_debt_increment"
            elif min_progress or target_progress:
                reason = "progress_credit"
            else:
                reason = "no_miss_or_progress"
            update_row = {
                "slot": now_slot,
                "contact_window_id": observed.get("contact_window_id", self._staged_delivery_contact_window_id),
                "task_id": task_id,
                "value_level": observed.get("value_level", ""),
                "product_requirement": observed.get("product_requirement", ""),
                "stage_start": observed.get("stage_start", ""),
                "stage_after": stage_after,
                "min_opportunity": int(bool(observed.get("min_opportunity", False))),
                "target_opportunity": int(target_opportunity),
                "min_stage_progress": int(min_progress),
                "target_stage_progress": int(target_progress),
                "min_stage_miss": int(min_miss),
                "target_stage_miss": int(target_miss),
                "miss_weight": miss_weight,
                "progress_credit": progress_credit,
                "debt_increment": increment,
                "age_debt_before": previous_age,
                "miss_debt_before": previous_miss,
                "Z_age": self.guarantee_age_debt_by_task[task_id],
                "Z_miss": self.guarantee_miss_debt_by_task[task_id],
                "Z_total": new_debt,
                "debt_before": previous_debt,
                "debt_after": new_debt,
                "future_contact_before_deadline": int(future_contact),
                "non_rescuable": int((min_miss or target_miss) and not future_contact),
                "bounded_by_z_max": int(new_debt >= self._guarantee_cap("z_total_max", self._guarantee_cap("z_max", 6.0)) - 1.0e-12),
                "debt_update_reason": reason,
            }
            self.last_guarantee_debt_window_update_trace.append(update_row)
            self.last_guarantee_debt_trace.append(update_row)
            self.last_task_debt_trajectory_trace.append(
                {
                    "slot": now_slot,
                    "contact_window_id": observed.get("contact_window_id", self._staged_delivery_contact_window_id),
                    "task_id": task_id,
                    "lifecycle_status": "pending" if state is None or not state.task_closed_flag else "closed",
                    "stage": stage_after,
                    "Z_age": self.guarantee_age_debt_by_task[task_id],
                    "Z_miss": self.guarantee_miss_debt_by_task[task_id],
                    "Z_total": new_debt,
                    "age_debt_increment": 0.0,
                    "miss_debt_increment": increment,
                    "progress_credit": progress_credit,
                    "debt_update_reason": reason,
                    "future_contact_before_deadline": int(future_contact),
                    "feasible_opportunity_this_window": int(bool(observed.get("min_opportunity", False)) or bool(observed.get("target_opportunity", False))),
                    "progress_made": int(min_progress or target_progress),
                    "selected_product": "",
                    "delivered_product": "ROI+Ctx" if target_progress else ("ROI" if min_progress else ""),
                    "final_task_outcome": "closed" if state is not None and state.task_closed_flag else "",
                }
            )
            if min_progress or target_progress:
                self.last_guarantee_progress_trace.append(
                    {
                        "slot": now_slot,
                        "contact_window_id": observed.get("contact_window_id", self._staged_delivery_contact_window_id),
                        "task_id": task_id,
                        "stage_start": observed.get("stage_start", ""),
                        "stage_after": stage_after,
                        "min_stage_progress": int(min_progress),
                        "target_stage_progress": int(target_progress),
                        "debt_after_progress": new_debt,
                        "Z_age_after_progress": self.guarantee_age_debt_by_task[task_id],
                        "Z_miss_after_progress": self.guarantee_miss_debt_by_task[task_id],
                    }
                )
        self._guarantee_window_observations = {}

    def _staged_request_deadline_slot(self, timeline_step: TimelineStep, task) -> int:
        deadline_time_s = (
            float(getattr(task, "arrival_slot", timeline_step.slot_index)) * float(timeline_step.slot_seconds)
            + float(getattr(task, "deadline_s", 0.0) or 0.0)
        )
        future_link_traces = getattr(self, "future_link_traces", [])
        if future_link_traces:
            return max(0, min(len(future_link_traces) - 1, int(deadline_time_s // float(timeline_step.slot_seconds))))
        return int(deadline_time_s // float(timeline_step.slot_seconds))

    def _build_staged_delivery_request(
        self,
        *,
        timeline_step: TimelineStep,
        state: ProductLifecycleState,
        task,
        product_type: str,
        parent_product_type: str,
        detection_quality: float,
        payload_size_mb: float,
        residual_delta: float,
        residual_remaining: float,
        earliest_delivery_slot: int,
    ) -> StagedDeliveryRequest:
        self._ensure_staged_delivery_state()
        self._staged_delivery_request_seq += 1
        value_level = str(getattr(task, "value_level", "") or self._metadata_get(task, "value_level", "") or "medium").lower()
        remaining_deadline_s = self._remaining_deadline_for_task(
            task,
            timeline_step.slot_index,
            timeline_step.slot_seconds,
            float(getattr(task, "deadline_s", 0.0) or 0.0),
        )
        timeliness = max(0.0, min(1.0, remaining_deadline_s / max(float(getattr(task, "deadline_s", 1.0) or 1.0), 1.0e-9)))
        request = StagedDeliveryRequest(
            request_id=f"staged_{int(timeline_step.slot_index):05d}_{self._staged_delivery_request_seq:06d}",
            task_id=str(task.task_id),
            product_type=product_type,
            parent_product_type=parent_product_type,
            product_requirement=str(state.product_requirement or getattr(task, "product_requirement", "verification")).lower(),
            value_level=value_level,
            calibrated_task_value=float(task_effective_value(task)),
            residual_delta_information_utility=float(residual_delta),
            residual_utility_remaining=float(residual_remaining),
            detection_quality=max(0.0, float(detection_quality)),
            timeliness_factor=timeliness,
            payload_size_mb=max(0.0, float(payload_size_mb)),
            deadline_slot=self._staged_request_deadline_slot(timeline_step, task),
            created_slot=int(timeline_step.slot_index),
            last_evaluated_slot=int(timeline_step.slot_index),
            earliest_delivery_slot=int(earliest_delivery_slot),
            duplicate_key=self._candidate_key(str(task.task_id), product_type),
            scene_id=str(getattr(task, "scene_id", "")),
            patch_id=str(self._metadata_get(task, "patch_id", "")),
            deadline_s=float(getattr(task, "deadline_s", 0.0) or 0.0),
            created_time_s=float(timeline_step.slot_index) * float(timeline_step.slot_seconds),
        )
        priority, components = compute_staged_priority(
            request,
            current_slot=int(timeline_step.slot_index),
            config=self.simulator_cfg,
        )
        request.priority_score = priority
        request.priority_components = components
        return request

    def _maybe_create_staged_delivery_request_after_delivery(
        self,
        *,
        timeline_step: TimelineStep,
        state: ProductLifecycleState,
        task,
        delivered_product_type: str,
        delivered_metadata: dict[str, object] | None,
        resource_state: ResourceState | None,
        comm_state: CommState | None,
    ) -> None:
        self._ensure_staged_delivery_state()
        if task is None or resource_state is None or comm_state is None:
            return
        cfg = self._staged_delivery_cfg()
        if not bool(cfg.get("enabled", False)):
            return
        requirement = str(state.product_requirement or getattr(task, "product_requirement", "verification")).lower()
        value_level = str(getattr(task, "value_level", "") or self._metadata_get(task, "value_level", "") or "medium").lower()
        delivered_products = {str(product).strip().lower() for product in state.delivered_products}
        if bool(cfg.get("enable_persistent_closing_requests", False)):
            if bool(state.task_closed_flag):
                return
            product_type = self._minimum_closing_residual_product(requirement, delivered_products)
            if product_type not in {"roi", "roi_context"}:
                return
            parent_product_type = str(delivered_product_type or stage_from_delivered(delivered_products)).strip().lower()
            if parent_product_type in {"roi+ctx", "roi_context"}:
                return
            if parent_product_type not in {"metadata", "roi"}:
                parent_product_type = "metadata" if "metadata" in delivered_products else "none"
        elif delivered_product_type == "metadata" and requirement in {"verification", "context", "evidence"}:
            product_type = "roi"
            parent_product_type = "metadata"
        elif delivered_product_type == "roi" and requirement in {"context", "evidence"} and value_level in {"high", "urgent"}:
            product_type = "roi_context"
            parent_product_type = "roi"
        else:
            return
        if self._staged_delivery_duplicate_exists(str(task.task_id), product_type):
            return
        residual_delta = delta_information_utility(product_type, state.delivered_products, requirement)
        residual_remaining = residual_information_utility(state.delivered_products, requirement)
        if residual_delta <= float(cfg.get("residual_utility_threshold", 0.01)):
            return
        payload_size_mb, _diag = estimate_payload_size_with_diagnostics(
            product_type,
            detection_result=None,
            task=task,
            payload_profiles_cfg=self.payload_profiles_cfg,
        )
        future_contact = self._roi_context_future_contact_feasibility(
            timeline_step=timeline_step,
            task=task,
            payload_size_mb=payload_size_mb,
            comm_state=comm_state,
            start_after_current=not bool(timeline_step.link_trace.in_contact),
        )
        future_contact_feasible = bool(future_contact.get("future_contact_feasible", 0))
        if (
            bool(cfg.get("require_future_contact_before_deadline", True))
            and not future_contact_feasible
            and not bool(cfg.get("allow_pending_without_future_contact", False))
        ):
            request = self._build_staged_delivery_request(
                timeline_step=timeline_step,
                state=state,
                task=task,
                product_type=product_type,
                parent_product_type=parent_product_type,
                detection_quality=float((delivered_metadata or {}).get("detection_quality", 1.0) or 1.0),
                payload_size_mb=payload_size_mb,
                residual_delta=residual_delta,
                residual_remaining=residual_remaining,
                earliest_delivery_slot=int(future_contact.get("earliest_feasible_delivery_slot", -1)),
            )
            request.status = "blocked"
            request.blocker_reason = "no_contact_opportunity"
            self.staged_delivery_requests[request.duplicate_key] = request
            state.pending_products.add(product_type)
            if state.pending_residual_registered_slot is None:
                state.pending_residual_registered_slot = int(timeline_step.slot_index)
            state.residual_utility_remaining = residual_remaining
            self._transition_staged_lifecycle(
                request,
                "blocked",
                slot=int(timeline_step.slot_index),
                reason="no_contact_opportunity",
                blocker_reason="no_contact_opportunity",
                contact_window_id=self._staged_delivery_contact_window_id,
                next_contact_window_id_candidate=str(future_contact.get("next_eligible_contact_window_id", "")),
            )
            self._record_staged_request_trace(request, "blocked_no_future_contact", int(timeline_step.slot_index))
            return
        request = self._build_staged_delivery_request(
            timeline_step=timeline_step,
            state=state,
            task=task,
            product_type=product_type,
            parent_product_type=parent_product_type,
            detection_quality=float((delivered_metadata or {}).get("detection_quality", 1.0) or 1.0),
            payload_size_mb=payload_size_mb,
            residual_delta=residual_delta,
            residual_remaining=residual_remaining,
            earliest_delivery_slot=int(future_contact.get("earliest_feasible_delivery_slot", -1)),
        )
        density = request.priority_components.get("value_density_priority", 0.0)
        if density < float(cfg.get("min_value_density", 0.0) or 0.0):
            request.status = "dropped"
            request.blocker_reason = "value_density_below_profile_threshold"
            self._transition_staged_lifecycle(
                request,
                "dropped",
                slot=int(timeline_step.slot_index),
                reason="marginal_utility_density_below_threshold",
                blocker_reason="value_density_below_profile_threshold",
                contact_window_id=self._staged_delivery_contact_window_id,
            )
            self.staged_delivery_history[request.duplicate_key] = request
            self._record_staged_request_trace(request, "blocked_value_density", int(timeline_step.slot_index))
            return
        if request.priority_score < float(cfg.get("eriv_threshold", 0.0) or 0.0):
            request.status = "dropped"
            request.blocker_reason = "eriv_below_profile_threshold"
            self._transition_staged_lifecycle(
                request,
                "dropped",
                slot=int(timeline_step.slot_index),
                reason="expected_residual_information_value_below_threshold",
                blocker_reason="eriv_below_profile_threshold",
                contact_window_id=self._staged_delivery_contact_window_id,
            )
            self.staged_delivery_history[request.duplicate_key] = request
            self._record_staged_request_trace(request, "blocked_eriv", int(timeline_step.slot_index))
            return
        if (
            product_type == "roi_context"
            and value_level not in {"high", "urgent"}
            and not bool(cfg.get("enable_persistent_closing_requests", False))
        ):
            request.status = "dropped"
            request.blocker_reason = "value_level_not_high_or_urgent"
            self._transition_staged_lifecycle(
                request,
                "dropped",
                slot=int(timeline_step.slot_index),
                reason="roi_context_not_required_for_value_level",
                blocker_reason="value_level_not_high_or_urgent",
                contact_window_id=self._staged_delivery_contact_window_id,
            )
            self.staged_delivery_history[request.duplicate_key] = request
            self._record_staged_request_trace(request, "blocked_value_level", int(timeline_step.slot_index))
            return
        self.staged_delivery_requests[request.duplicate_key] = request
        state.pending_products.add(product_type)
        if state.pending_residual_registered_slot is None:
            state.pending_residual_registered_slot = int(timeline_step.slot_index)
        state.residual_utility_remaining = residual_remaining
        self._transition_staged_lifecycle(
            request,
            "pending",
            slot=int(timeline_step.slot_index),
            reason="staged_request_created_pending",
            contact_window_id=self._staged_delivery_contact_window_id,
            next_contact_window_id_candidate=str(future_contact.get("next_eligible_contact_window_id", "")),
        )
        self._record_staged_request_trace(request, "created_pending", int(timeline_step.slot_index))

    def _staged_request_to_payload(self, request: StagedDeliveryRequest, timeline_step: TimelineStep):
        task = self.task_snapshots.get(request.task_id)
        if task is None:
            return None
        detection_placeholder = DetectionResult(
            bboxes=[],
            confidence=0.0,
            target_flag=False,
            roi_regions=[],
            uncertainty=float(getattr(task, "uncertainty", 0.0) or 0.0),
            inference_latency_s=0.0,
            estimated_quality=float(request.detection_quality),
            generated_metadata={"staged_delivery_placeholder": True},
        )
        payload = generate_residual_payload(
            task=task,
            detection_result=detection_placeholder,
            residual_product_type=request.product_type,
            slot_index=int(timeline_step.slot_index),
            parent_payload_plan=f"{request.parent_product_type}_then_{request.product_type}",
            high_value_threshold=float(self.reward_cfg.get("high_value_threshold", 0.75)),
            payload_profiles_cfg=self.payload_profiles_cfg,
            delivered_products=set(self.product_lifecycle.get(request.task_id, ProductLifecycleState(request.task_id)).delivered_products),
            remaining_deadline_s=self._remaining_deadline_for_task(
                task,
                timeline_step.slot_index,
                timeline_step.slot_seconds,
                request.deadline_s,
            ),
            continue_request_priority_bonus=0.0,
            followup_trigger="value_aware_staged_delivery_queue",
        )
        cfg = self._staged_delivery_cfg()
        payload.metadata = {
            **payload.metadata,
            "staged_delivery_queue_enabled": True,
            "staged_request_id": request.request_id,
            "staged_duplicate_key": request.duplicate_key,
            "lifecycle_id": request.lifecycle_id,
            "staged_V_eff": request.priority_components.get("V_eff", request.calibrated_task_value),
            "staged_delta_u_I": request.priority_components.get(
                "delta_u_I",
                request.residual_delta_information_utility,
            ),
            "staged_detection_quality": request.priority_components.get("detection_quality", request.detection_quality),
            "staged_timeliness_factor": request.priority_components.get("timeliness_factor", request.timeliness_factor),
            "staged_product_requirement_weight": request.priority_components.get(
                "product_requirement_weight",
                product_requirement_weight(cfg, request.product_requirement, request.product_type),
            ),
            "staged_value_level_factor": request.priority_components.get(
                "value_level_factor",
                value_level_factor(cfg, request.value_level, request.product_type),
            ),
            "staged_payload_size_mb": request.payload_size_mb,
            "staged_deadline_s": request.deadline_s,
            "staged_deadline_urgency_lambda": float(cfg.get("deadline_urgency_lambda", 0.0) or 0.0),
            "staged_fixed_bonus_debug": request.priority_components.get("fixed_bonus_debug", 0.0),
            "guarantee_priority_enabled": bool(guarantee_priority_active(self.simulator_cfg)),
            "guarantee_priority_score": request.priority_score,
            "guarantee_debt_z": request.priority_components.get("guarantee_debt_z", 0.0),
            "guarantee_Z_age": request.priority_components.get("Z_age", 0.0),
            "guarantee_Z_miss": request.priority_components.get("Z_miss", 0.0),
            "guarantee_Z_total": request.priority_components.get("Z_total", request.priority_components.get("guarantee_debt_z", 0.0)),
            "guarantee_progress_gate": request.priority_components.get("guarantee_progress_gate", 0.0),
            "guarantee_G_i_p": request.priority_components.get("G_i_p", request.priority_components.get("guarantee_progress_gate", 0.0)),
            "guarantee_term": request.priority_components.get("guarantee_term", 0.0),
            "guarantee_rank_without": request.priority_components.get("rank_without_guarantee", 0),
            "guarantee_rank_with": request.priority_components.get("rank_with_guarantee", 0),
            "guarantee_assisted_candidate": bool(request.priority_components.get("guarantee_assisted_candidate", 0)),
            "guarantee_priority_without_guarantee": request.priority_components.get(
                "priority_without_guarantee",
                request.priority_score,
            ),
            "staged_priority_inversion_fix_enabled": bool(cfg.get("priority_inversion_fix_enabled", False)),
            "staged_residual_preempts_ordinary": bool(
                cfg.get("priority_inversion_fix_enabled", False) and self._is_high_value_context_residual(request)
            ),
            "staged_preemption_priority_floor": float(cfg.get("preemption_priority_floor", 1000.0) or 0.0),
            "expected_residual_information_value": request.priority_score,
            "ERIV": request.priority_score,
            "followup_source": "value_aware_staged_delivery_queue",
        }
        payload.followup_trigger = "value_aware_staged_delivery_queue"
        payload.duplicate_identity_key = request.duplicate_key
        return payload

    def _record_staged_backlog_composition(self, timeline_step: TimelineStep, comm_state: CommState) -> None:
        self._ensure_staged_delivery_state()
        active = [req for req in self.staged_delivery_requests.values() if req.status in {"pending", "enqueued", "blocked"}]
        high_urgent = [req for req in active if req.value_level in {"high", "urgent"}]
        low_medium = [req for req in active if req.value_level in {"low", "medium"}]
        context_evidence = [req for req in active if req.product_requirement in {"context", "evidence"}]
        high_context = [req for req in high_urgent if req.product_requirement in {"context", "evidence"}]
        value_sum = sum(req.calibrated_task_value for req in active)
        row = {
            "slot": int(timeline_step.slot_index),
            "time_s": float(timeline_step.slot_index) * float(timeline_step.slot_seconds),
            "contact_window_id": self._staged_delivery_contact_window_id,
            "in_contact": int(bool(timeline_step.link_trace.in_contact)),
            "pending_count": sum(1 for req in active if req.status == "pending"),
            "enqueued_count": sum(1 for req in active if req.status == "enqueued"),
            "blocked_count": sum(1 for req in active if req.status == "blocked"),
            "active_count": len(active),
            "value_sum": value_sum,
            "mean_value": value_sum / max(len(active), 1),
            "high_urgent_count": len(high_urgent),
            "high_urgent_context_evidence_count": len(high_context),
            "low_medium_count": len(low_medium),
            "context_evidence_count": len(context_evidence),
            "queue_mb": float(comm_state.queue_mb),
        }
        self.last_backlog_composition_trace.append(row)
        self.last_leftover_tail_value_trace.append(dict(row, tail_is_low_value_flag=int(len(high_context) == 0)))

    def _process_staged_delivery_queue_at_slot(
        self,
        *,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        router: BaseRouter | None,
    ) -> DownlinkResult | None:
        self._ensure_staged_delivery_state()
        self._refresh_staged_delivery_contact_budget(timeline_step)
        if not self._staged_delivery_enabled(router):
            return None
        self._materialize_pending_lifecycle_records(timeline_step)
        cfg = self._staged_delivery_cfg()
        now_slot = int(timeline_step.slot_index)
        for key, request in list(self.staged_delivery_requests.items()):
            self._ensure_staged_lifecycle_record(request, now_slot)
            if request.status not in {"pending", "enqueued", "blocked"}:
                continue
            if now_slot > int(request.deadline_slot):
                request.status = "expired"
                request.expired_slot = now_slot
                request.blocker_reason = "expired"
                self._transition_staged_lifecycle(
                    request,
                    "expired",
                    slot=now_slot,
                    reason="staged_request_deadline_passed",
                    blocker_reason="deadline_passed",
                    contact_window_id=self._staged_delivery_contact_window_id,
                )
                self.staged_delivery_history[key] = request
                self.staged_delivery_requests.pop(key, None)
                self._record_staged_request_trace(request, "expired", now_slot)
        disabled_request_ids = self._request_disable_replay_ids()
        for key, request in list(self.staged_delivery_requests.items()):
            if request.status not in {"pending", "enqueued", "blocked"}:
                continue
            if request.request_id in disabled_request_ids:
                self._disable_staged_request_for_replay(request, now_slot)
        self._update_guarantee_age_debt_for_waiting_tasks(timeline_step, router)
        pending = [req for req in self.staged_delivery_requests.values() if req.status in {"pending", "blocked"}]
        if self._guarantee_priority_active(router):
            delivered_by_task = {
                task_id: set(state.delivered_products)
                for task_id, state in self.product_lifecycle.items()
            }
            queue_pressure = float(comm_state.queue_mb) / max(float(comm_state.queue_max_mb), float(self.simulator_cfg.get("epsilon", 1.0e-9)))
            rank_guarantee_aware_requests(
                pending,
                current_slot=now_slot,
                config=self.simulator_cfg,
                debt_by_task=self.guarantee_debt_by_task,
                delivered_products_by_task=delivered_by_task,
                debt_components_by_task=self._guarantee_debt_components_by_task(),
                waiting_age_slots_by_task=self.guarantee_waiting_age_slots_by_task,
                queue_pressure=queue_pressure,
            )
            for request in pending:
                request.last_evaluated_slot = now_slot
        else:
            for request in pending:
                request.last_evaluated_slot = now_slot
                request.priority_score, request.priority_components = compute_staged_priority(
                    request,
                    current_slot=now_slot,
                    config=self.simulator_cfg,
                )
        ranked = sorted(pending, key=lambda req: (-req.priority_score, req.deadline_slot, req.payload_size_mb, req.request_id))
        max_slot = max(0, int(float(cfg.get("max_admissions_per_slot", 2) or 0)))
        max_window = max(0, int(float(cfg.get("max_admissions_per_contact_window", 4) or 0)))
        priority_inversion_fix_enabled = bool(cfg.get("priority_inversion_fix_enabled", False))
        future_contact_generation_enabled = bool(cfg.get("enable_future_contact_generation", False))
        available_capacity_mb = compute_available_capacity_mb(
            timeline_step.link_trace,
            float(timeline_step.slot_seconds),
            float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
        )
        capacity_cutoff_rank = min(max_slot, max(0, max_window - self._staged_delivery_contact_window_admission_count))
        if not bool(timeline_step.link_trace.in_contact) and not future_contact_generation_enabled:
            capacity_cutoff_rank = 0
        candidate_feasibility: dict[str, dict[str, object]] = {}
        for order, request in enumerate(ranked):
            feasibility_row = self._staged_candidate_feasibility_row(
                request,
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                queue_order=order,
                available_capacity_mb=available_capacity_mb,
            )
            candidate_feasibility[request.request_id] = feasibility_row
            self.last_contact_window_candidate_rank_trace.append(feasibility_row)
            self._record_guarantee_opportunity(request, feasibility_row, slot=now_slot)
        admitted_this_slot = 0
        result: DownlinkResult | None = None
        for order, request in enumerate(ranked):
            feasibility_row = candidate_feasibility.get(request.request_id, {})
            future_generation_feasible = bool(feasibility_row.get("future_generation_feasible", 0))
            current_contact_admission = bool(timeline_step.link_trace.in_contact)
            future_contact_admission = (
                future_contact_generation_enabled
                and not current_contact_admission
                and future_generation_feasible
            )
            high_value_feasible_bypass = (
                priority_inversion_fix_enabled
                and (bool(feasibility_row.get("feasible", 0)) or future_contact_admission)
                and self._is_high_value_context_residual(request)
            )
            selected = (
                (current_contact_admission or future_contact_admission)
                and (
                    high_value_feasible_bypass
                    or (
                        admitted_this_slot < max_slot
                        and self._staged_delivery_contact_window_admission_count < max_window
                    )
                )
            )
            self._record_staged_priority_trace(
                request,
                slot=now_slot,
                queue_order=order,
                selected_for_admission=selected,
            )
            self._record_guarantee_priority_trace(request, slot=now_slot, queue_order=order)
            if not selected:
                if not timeline_step.link_trace.in_contact and not future_contact_generation_enabled:
                    reason = "no_contact_opportunity"
                elif not timeline_step.link_trace.in_contact and not future_generation_feasible:
                    reason = "future_contact_generation_not_feasible"
                else:
                    reason = "staged_budget_exceeded"
                previous_status = request.status
                request.status = "blocked"
                request.blocker_reason = reason
                self._transition_staged_lifecycle(
                    request,
                    "blocked",
                    slot=now_slot,
                    reason=reason,
                    blocker_reason=reason,
                    contact_window_id=self._staged_delivery_contact_window_id,
                    next_contact_window_id_candidate=self._next_contact_window_id(now_slot + 1, int(request.deadline_slot)),
                )
                self.lifecycle_ledger.record_recompetition(
                    slot=now_slot,
                    contact_window_id=self._staged_delivery_contact_window_id,
                    task_id=request.task_id,
                    product_type=request.product_type,
                    previous_status=previous_status,
                    reevaluated_flag=True,
                    feasible_this_window=bool(feasibility_row.get("feasible", 0)),
                    priority_score=float(request.priority_score),
                    priority_rank=order,
                    selected_for_delivery=False,
                    delivered=False,
                    blocker_reason=reason,
                    status_after_window="queued" if request.status == "enqueued" else request.status,
                )
                self.last_contact_window_scheduling_trace.append(
                    {
                        "slot": now_slot,
                        "time_s": float(now_slot) * float(timeline_step.slot_seconds),
                        "contact_window_id": self._staged_delivery_contact_window_id,
                        "request_id": request.request_id,
                        "task_id": request.task_id,
                        "product_type": request.product_type,
                        "priority_score": request.priority_score,
                        "status_before": request.status,
                        "decision": "left_pending",
                        "reason_code": reason,
                        "priority_inversion_fix_enabled": int(priority_inversion_fix_enabled),
                        "high_value_feasible_bypass": int(high_value_feasible_bypass),
                        "in_contact": int(bool(timeline_step.link_trace.in_contact)),
                        "queue_mb": float(comm_state.queue_mb),
                        "future_contact_generation_enabled": int(future_contact_generation_enabled),
                        "future_generation_feasible": int(future_generation_feasible),
                        "admission_mode": "current_contact" if current_contact_admission else "future_contact",
                    }
                )
                self._append_candidate_priority_trace(
                    self._candidate_priority_trace_row(
                        request,
                        feasibility_row,
                        slot=now_slot,
                        order=order,
                        capacity_cutoff_rank=capacity_cutoff_rank,
                        selected=False,
                        blocker_reason=reason,
                        lifecycle_status_after=request.status,
                    )
                )
                continue
            guarantee_assisted = bool(request.priority_components.get("guarantee_assisted_candidate", 0))
            if guarantee_assisted:
                floor = marginal_utility_floor(self._guarantee_cfg())
                if float(request.residual_delta_information_utility) < floor:
                    self._block_staged_request_for_guarantee(
                        request,
                        slot=now_slot,
                        order=order,
                        reason_code="marginal_utility_floor_blocked",
                        decision="marginal_utility_floor_blocked",
                        feasibility_row=feasibility_row,
                    )
                    self.last_guarantee_budget_trace.append(
                        {
                            "slot": now_slot,
                            "contact_window_id": self._staged_delivery_contact_window_id,
                            "request_id": request.request_id,
                            "task_id": request.task_id,
                            "product_type": request.product_type,
                            "decision": "marginal_utility_floor_blocked",
                            "guarantee_assisted": 1,
                            "guarantee_budget_used_MB": self._guarantee_window_used_mb,
                            "guarantee_budget_limit_MB": self._guarantee_budget_limit_mb(available_capacity_mb),
                            "payload_size_mb": float(request.payload_size_mb),
                            "marginal_utility": float(request.residual_delta_information_utility),
                            "marginal_utility_floor": floor,
                        }
                    )
                    self._append_candidate_priority_trace(
                        self._candidate_priority_trace_row(
                            request,
                            feasibility_row,
                            slot=now_slot,
                            order=order,
                            capacity_cutoff_rank=capacity_cutoff_rank,
                            selected=True,
                            blocker_reason="marginal_utility_floor_blocked",
                            lifecycle_status_after=request.status,
                        )
                    )
                    continue
                budget_limit = self._guarantee_budget_limit_mb(available_capacity_mb)
                if self._guarantee_window_used_mb + float(request.payload_size_mb) > budget_limit + float(
                    self.simulator_cfg.get("epsilon", 1.0e-9)
                ):
                    self._block_staged_request_for_guarantee(
                        request,
                        slot=now_slot,
                        order=order,
                        reason_code="guarantee_budget_limited",
                        decision="guarantee_budget_limited",
                        feasibility_row=feasibility_row,
                    )
                    self.last_guarantee_budget_trace.append(
                        {
                            "slot": now_slot,
                            "contact_window_id": self._staged_delivery_contact_window_id,
                            "request_id": request.request_id,
                            "task_id": request.task_id,
                            "product_type": request.product_type,
                            "decision": "guarantee_budget_limited",
                            "guarantee_assisted": 1,
                            "guarantee_budget_used_MB": self._guarantee_window_used_mb,
                            "guarantee_budget_limit_MB": budget_limit,
                            "payload_size_mb": float(request.payload_size_mb),
                        }
                    )
                    self._append_candidate_priority_trace(
                        self._candidate_priority_trace_row(
                            request,
                            feasibility_row,
                            slot=now_slot,
                            order=order,
                            capacity_cutoff_rank=capacity_cutoff_rank,
                            selected=True,
                            blocker_reason="guarantee_budget_limited",
                            lifecycle_status_after=request.status,
                        )
                    )
                    continue
            closure_guard = self._closure_guard_decision(
                request,
                timeline_step=timeline_step,
                comm_state=comm_state,
                feasibility_row=feasibility_row,
                available_capacity_mb=available_capacity_mb,
            )
            if str(closure_guard.get("decision", "ADMIT")) != "ADMIT":
                previous_status = request.status
                request.status = "blocked"
                request.blocker_reason = str(closure_guard.get("rejection_reason", "closure_guard_blocked"))
                self._transition_staged_lifecycle(
                    request,
                    "blocked",
                    slot=now_slot,
                    reason=request.blocker_reason,
                    blocker_reason=request.blocker_reason,
                    contact_window_id=self._staged_delivery_contact_window_id,
                    next_contact_window_id_candidate=self._next_contact_window_id(now_slot + 1, int(request.deadline_slot)),
                )
                self.lifecycle_ledger.record_recompetition(
                    slot=now_slot,
                    contact_window_id=self._staged_delivery_contact_window_id,
                    task_id=request.task_id,
                    product_type=request.product_type,
                    previous_status=previous_status,
                    reevaluated_flag=True,
                    feasible_this_window=bool(feasibility_row.get("feasible", 0)),
                    priority_score=float(request.priority_score),
                    priority_rank=order,
                    selected_for_delivery=False,
                    delivered=False,
                    blocker_reason=request.blocker_reason,
                    status_after_window=request.status,
                )
                self.last_contact_window_scheduling_trace.append(
                    {
                        "slot": now_slot,
                        "time_s": float(now_slot) * float(timeline_step.slot_seconds),
                        "contact_window_id": self._staged_delivery_contact_window_id,
                        "request_id": request.request_id,
                        "task_id": request.task_id,
                        "product_type": request.product_type,
                        "priority_score": request.priority_score,
                        "status_before": previous_status,
                        "decision": str(closure_guard.get("decision", "TEMPORARILY_BLOCK")),
                        "reason_code": request.blocker_reason,
                        "protected_tasks_affected": str(closure_guard.get("protected_tasks_affected", "")),
                        "estimated_displacement_loss": float(closure_guard.get("estimated_displacement_loss", 0.0) or 0.0),
                        "estimated_net_gain": float(closure_guard.get("estimated_net_gain", 0.0) or 0.0),
                        "priority_inversion_fix_enabled": int(priority_inversion_fix_enabled),
                        "high_value_feasible_bypass": int(high_value_feasible_bypass),
                        "in_contact": int(bool(timeline_step.link_trace.in_contact)),
                        "queue_mb": float(comm_state.queue_mb),
                        "future_contact_generation_enabled": int(future_contact_generation_enabled),
                        "future_generation_feasible": int(future_generation_feasible),
                        "admission_mode": "current_contact" if current_contact_admission else "future_contact",
                    }
                )
                self._append_candidate_priority_trace(
                    self._candidate_priority_trace_row(
                        request,
                        feasibility_row,
                        slot=now_slot,
                        order=order,
                        capacity_cutoff_rank=capacity_cutoff_rank,
                        selected=True,
                        blocker_reason=request.blocker_reason,
                        lifecycle_status_after=request.status,
                    )
                )
                continue
            state = self.product_lifecycle.get(request.task_id)
            task = self.task_snapshots.get(request.task_id)
            payload = self._staged_request_to_payload(request, timeline_step)
            if state is None or task is None or payload is None:
                previous_status = request.status
                request.status = "blocked"
                request.blocker_reason = "missing_task_or_lifecycle"
                self._transition_staged_lifecycle(
                    request,
                    "blocked",
                    slot=now_slot,
                    reason="missing_task_or_lifecycle",
                    blocker_reason="missing_task_or_lifecycle",
                    contact_window_id=self._staged_delivery_contact_window_id,
                )
                self.lifecycle_ledger.record_recompetition(
                    slot=now_slot,
                    contact_window_id=self._staged_delivery_contact_window_id,
                    task_id=request.task_id,
                    product_type=request.product_type,
                    previous_status=previous_status,
                    reevaluated_flag=True,
                    feasible_this_window=False,
                    priority_score=float(request.priority_score),
                    priority_rank=order,
                    selected_for_delivery=False,
                    delivered=False,
                    blocker_reason="missing_task_or_lifecycle",
                    status_after_window="queued" if request.status == "enqueued" else request.status,
                )
                self._append_candidate_priority_trace(
                    self._candidate_priority_trace_row(
                        request,
                        feasibility_row,
                        slot=now_slot,
                        order=order,
                        capacity_cutoff_rank=capacity_cutoff_rank,
                        selected=True,
                        blocker_reason="missing_task_or_lifecycle",
                        lifecycle_status_after=request.status,
                    )
                )
                continue
            guard_decision = self._followup_hard_safety_guard(
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                payload=payload,
                lifecycle_state=state,
                task=task,
                operation="value_aware_staged_delivery_queue_admission",
                require_same_slot_delivery=False,
                allow_no_contact_enqueue=bool(future_contact_admission),
                predict_immediate_downlink=False,
                allow_existing_pending=True,
            )
            if not guard_decision.allowed:
                previous_status = request.status
                request.status = "blocked"
                request.blocker_reason = guard_decision.reason_code
                self._transition_staged_lifecycle(
                    request,
                    "blocked",
                    slot=now_slot,
                    reason=guard_decision.reason_code,
                    blocker_reason=guard_decision.reason_code,
                    contact_window_id=self._staged_delivery_contact_window_id,
                    next_contact_window_id_candidate=self._next_contact_window_id(now_slot + 1, int(request.deadline_slot)),
                )
                self.lifecycle_ledger.record_recompetition(
                    slot=now_slot,
                    contact_window_id=self._staged_delivery_contact_window_id,
                    task_id=request.task_id,
                    product_type=request.product_type,
                    previous_status=previous_status,
                    reevaluated_flag=True,
                    feasible_this_window=False,
                    priority_score=float(request.priority_score),
                    priority_rank=order,
                    selected_for_delivery=False,
                    delivered=False,
                    blocker_reason=guard_decision.reason_code,
                    status_after_window="queued" if request.status == "enqueued" else request.status,
                )
                self.last_contact_window_scheduling_trace.append(
                    {
                        "slot": now_slot,
                        "time_s": float(now_slot) * float(timeline_step.slot_seconds),
                        "contact_window_id": self._staged_delivery_contact_window_id,
                        "request_id": request.request_id,
                        "task_id": request.task_id,
                        "product_type": request.product_type,
                        "priority_score": request.priority_score,
                        "status_before": request.status,
                        "decision": "blocked_by_hard_safety_guard",
                        "reason_code": guard_decision.reason_code,
                        "blockers": self._format_products(guard_decision.blockers),
                        "priority_inversion_fix_enabled": int(priority_inversion_fix_enabled),
                        "high_value_feasible_bypass": int(high_value_feasible_bypass),
                        "in_contact": int(bool(timeline_step.link_trace.in_contact)),
                        "queue_mb": float(comm_state.queue_mb),
                        "future_contact_generation_enabled": int(future_contact_generation_enabled),
                        "future_generation_feasible": int(future_generation_feasible),
                        "admission_mode": "current_contact" if current_contact_admission else "future_contact",
                    }
                )
                self._record_guarantee_blocker(
                    request,
                    slot=now_slot,
                    reason_code=guard_decision.reason_code,
                    decision="safety_blocked",
                )
                if guarantee_assisted:
                    self.last_guarantee_budget_trace.append(
                        {
                            "slot": now_slot,
                            "contact_window_id": self._staged_delivery_contact_window_id,
                            "request_id": request.request_id,
                            "task_id": request.task_id,
                            "product_type": request.product_type,
                            "decision": "safety_blocked",
                            "guarantee_assisted": 1,
                            "guarantee_budget_used_MB": self._guarantee_window_used_mb,
                            "guarantee_budget_limit_MB": self._guarantee_budget_limit_mb(available_capacity_mb),
                            "payload_size_mb": float(request.payload_size_mb),
                        }
                    )
                self._append_candidate_priority_trace(
                    self._candidate_priority_trace_row(
                        request,
                        feasibility_row,
                        slot=now_slot,
                        order=order,
                        capacity_cutoff_rank=capacity_cutoff_rank,
                        selected=True,
                        blocker_reason=guard_decision.reason_code,
                        lifecycle_status_after=request.status,
                    )
                )
                continue
            admission_result = self.downlink_queue.process_slot(
                slot_index=timeline_step.slot_index,
                payload=payload,
                link_trace=timeline_step.link_trace,
                capacity_override_mb=0.0,
                contact_window_id=self._staged_delivery_contact_window_id,
            )
            comm_state.queue_mb = admission_result.queue_next_mb
            comm_state.queue_item_count = admission_result.queue_item_count
            admitted_this_slot += int(admission_result.residual_enqueue_count > 0)
            self._staged_delivery_contact_window_admission_count += int(admission_result.residual_enqueue_count > 0)
            if admission_result.residual_enqueue_count:
                previous_status = request.status
                request.status = "enqueued"
                request.enqueued_slot = now_slot
                request.blocker_reason = ""
                if guarantee_assisted:
                    self._guarantee_window_used_mb += float(request.payload_size_mb)
                self._transition_staged_lifecycle(
                    request,
                    "queued",
                    slot=now_slot,
                    reason="admitted_to_downlink_queue",
                    contact_window_id=self._staged_delivery_contact_window_id,
                )
                self._record_staged_request_trace(request, "enqueued", now_slot)
            elif admission_result.duplicate_residual_suppression_count:
                previous_status = request.status
                request.status = "superseded"
                request.blocker_reason = "duplicate_residual_already_queued"
                self._transition_staged_lifecycle(
                    request,
                    "superseded",
                    slot=now_slot,
                    reason="duplicate_residual_already_queued",
                    blocker_reason="duplicate_residual_already_queued",
                    contact_window_id=self._staged_delivery_contact_window_id,
                )
                self.staged_delivery_history[request.duplicate_key] = request
                self.staged_delivery_requests.pop(request.duplicate_key, None)
            else:
                previous_status = request.status
                request.status = "blocked"
                request.blocker_reason = "queue_admission_rejected"
                self._transition_staged_lifecycle(
                    request,
                    "blocked",
                    slot=now_slot,
                    reason="queue_admission_rejected",
                    blocker_reason="queue_admission_rejected",
                    contact_window_id=self._staged_delivery_contact_window_id,
                )
            self.lifecycle_ledger.record_recompetition(
                slot=now_slot,
                contact_window_id=self._staged_delivery_contact_window_id,
                task_id=request.task_id,
                product_type=request.product_type,
                previous_status=previous_status,
                reevaluated_flag=True,
                feasible_this_window=bool(feasibility_row.get("feasible", 0)),
                priority_score=float(request.priority_score),
                priority_rank=order,
                selected_for_delivery=True,
                delivered=False,
                blocker_reason=request.blocker_reason,
                status_after_window="queued" if request.status == "enqueued" else request.status,
            )
            self.last_contact_window_scheduling_trace.append(
                {
                    "slot": now_slot,
                    "time_s": float(now_slot) * float(timeline_step.slot_seconds),
                    "contact_window_id": self._staged_delivery_contact_window_id,
                    "request_id": request.request_id,
                    "task_id": request.task_id,
                    "product_type": request.product_type,
                    "priority_score": request.priority_score,
                    "status_before": "pending",
                    "decision": "enqueued" if admission_result.residual_enqueue_count else "not_enqueued",
                    "reason_code": "admitted_to_downlink_queue"
                    if admission_result.residual_enqueue_count
                    else "queue_admission_rejected",
                    "priority_inversion_fix_enabled": int(priority_inversion_fix_enabled),
                    "high_value_feasible_bypass": int(high_value_feasible_bypass),
                    "in_contact": int(bool(timeline_step.link_trace.in_contact)),
                    "queue_mb": float(comm_state.queue_mb),
                    "queue_item_count": int(comm_state.queue_item_count),
                    "guarantee_assisted": int(guarantee_assisted),
                    "future_contact_generation_enabled": int(future_contact_generation_enabled),
                    "future_generation_feasible": int(future_generation_feasible),
                    "admission_mode": "current_contact" if current_contact_admission else "future_contact",
                    "next_eligible_contact_window_id": str(feasibility_row.get("next_eligible_contact_window_id", "")),
                    "earliest_feasible_delivery_slot": int(feasibility_row.get("earliest_feasible_delivery_slot", -1) or -1),
                    "known_contact_capacity_before_deadline_mb": float(
                        feasibility_row.get("known_contact_capacity_before_deadline_mb", 0.0) or 0.0
                    ),
                    "required_service_before_deadline_mb": float(
                        feasibility_row.get("required_service_before_deadline_mb", 0.0) or 0.0
                    ),
                }
            )
            self.last_guarantee_budget_trace.append(
                {
                    "slot": now_slot,
                    "contact_window_id": self._staged_delivery_contact_window_id,
                    "request_id": request.request_id,
                    "task_id": request.task_id,
                    "product_type": request.product_type,
                    "decision": "guarantee_assisted" if guarantee_assisted and admission_result.residual_enqueue_count else "utility_selected",
                    "guarantee_assisted": int(guarantee_assisted),
                    "guarantee_budget_used_MB": self._guarantee_window_used_mb,
                    "guarantee_budget_limit_MB": self._guarantee_budget_limit_mb(available_capacity_mb),
                    "payload_size_mb": float(request.payload_size_mb),
                }
            )
            self._append_candidate_priority_trace(
                self._candidate_priority_trace_row(
                    request,
                    feasibility_row,
                    slot=now_slot,
                    order=order,
                    capacity_cutoff_rank=capacity_cutoff_rank,
                    selected=True,
                    blocker_reason=request.blocker_reason,
                    lifecycle_status_after=request.status,
                )
            )
            result = admission_result if result is None else self._merge_downlink_results(result, admission_result)
        self._record_staged_backlog_composition(timeline_step, comm_state)
        return result

    def _mark_staged_delivery_from_downlink_result(self, downlink_result: DownlinkResult, timeline_step: TimelineStep) -> None:
        self._ensure_staged_delivery_state()
        def _find_request(metadata: dict[str, object]):
            request_id = str(metadata.get("staged_request_id", ""))
            duplicate_key = str(metadata.get("staged_duplicate_key", ""))
            request = None
            if duplicate_key:
                request = self.staged_delivery_requests.get(duplicate_key) or self.staged_delivery_history.get(duplicate_key)
            if request is None:
                for candidate in self.staged_delivery_requests.values():
                    if candidate.request_id == request_id:
                        request = candidate
                        duplicate_key = candidate.duplicate_key
                        break
            return request, duplicate_key

        for index, metadata in enumerate(getattr(downlink_result, "delivered_payload_metadata", [])):
            request_id = str(metadata.get("staged_request_id", ""))
            duplicate_key = str(metadata.get("staged_duplicate_key", ""))
            if not request_id and not duplicate_key:
                continue
            request, duplicate_key = _find_request(metadata)
            if request is None:
                continue
            request.status = "delivered"
            request.delivered_slot = int(timeline_step.slot_index)
            record = self.lifecycle_ledger.get(request.lifecycle_id)
            if record is None or record.current_status != "delivered":
                self._transition_staged_lifecycle(
                    request,
                    "delivered",
                    slot=int(timeline_step.slot_index),
                    reason="full_payload_downlinked",
                    contact_window_id=self._staged_delivery_contact_window_id,
                )
            self.staged_delivery_history[request.duplicate_key] = request
            self.staged_delivery_requests.pop(request.duplicate_key, None)
            self._record_staged_request_trace(request, "delivered", int(timeline_step.slot_index))
        for metadata in getattr(downlink_result, "expired_payload_metadata", []):
            request_id = str(metadata.get("staged_request_id", ""))
            duplicate_key = str(metadata.get("staged_duplicate_key", ""))
            if not request_id and not duplicate_key:
                continue
            request, _duplicate_key = _find_request(metadata)
            if request is None:
                continue
            request.status = "expired"
            request.expired_slot = int(timeline_step.slot_index)
            request.blocker_reason = "deadline_passed"
            record = self.lifecycle_ledger.get(request.lifecycle_id)
            if record is None or record.current_status != "expired":
                self._transition_staged_lifecycle(
                    request,
                    "expired",
                    slot=int(timeline_step.slot_index),
                    reason="deadline_passed",
                    blocker_reason="deadline_passed",
                    contact_window_id=self._staged_delivery_contact_window_id,
                )
            self.staged_delivery_history[request.duplicate_key] = request
            self.staged_delivery_requests.pop(request.duplicate_key, None)
            self._record_staged_request_trace(request, "expired", int(timeline_step.slot_index))
        for metadata in getattr(downlink_result, "dropped_payload_metadata", []):
            request_id = str(metadata.get("staged_request_id", ""))
            duplicate_key = str(metadata.get("staged_duplicate_key", ""))
            if not request_id and not duplicate_key:
                continue
            request, _duplicate_key = _find_request(metadata)
            if request is None:
                continue
            request.status = "dropped"
            request.blocker_reason = "queue_overflow"
            record = self.lifecycle_ledger.get(request.lifecycle_id)
            if record is None or record.current_status != "dropped":
                self._transition_staged_lifecycle(
                    request,
                    "dropped",
                    slot=int(timeline_step.slot_index),
                    reason="queue_overflow_lowest_priority_evicted",
                    blocker_reason="queue_overflow",
                    contact_window_id=self._staged_delivery_contact_window_id,
                )
            self.staged_delivery_history[request.duplicate_key] = request
            self.staged_delivery_requests.pop(request.duplicate_key, None)
            self._record_staged_request_trace(request, "dropped", int(timeline_step.slot_index))

    def _followup_guard_cfg(self) -> dict[str, object]:
        progressive = self.simulator_cfg.get("progressive_payload", {})
        cfg = progressive.get("followup_hard_safety_guard", {}) if isinstance(progressive, dict) else {}
        if not isinstance(cfg, dict):
            cfg = {}
        phase6_cfg = phase6_proactive_config(self.simulator_cfg)
        phase5_cfg = phase5_continue_config(self.simulator_cfg)
        return {
            "min_remaining_deadline_s": float(
                cfg.get(
                    "min_remaining_deadline_s",
                    phase6_cfg.get("min_remaining_deadline_s", phase5_cfg.get("min_remaining_deadline_s", 0.0)),
                )
                or 0.0
            ),
            "allow_no_contact_enqueue": bool(cfg.get("allow_no_contact_enqueue", True)),
        }

    def _predict_extra_payload_comm_energy_j(
        self,
        *,
        timeline_step: TimelineStep,
        payload_mb: float,
        require_same_slot_delivery: bool,
        predict_immediate_downlink: bool,
        remaining_contact_capacity_mb: float,
    ) -> float:
        if payload_mb <= 0.0 or not timeline_step.link_trace.in_contact:
            return 0.0
        if require_same_slot_delivery:
            tx_payload_mb = min(float(payload_mb), max(0.0, float(remaining_contact_capacity_mb)))
        elif predict_immediate_downlink:
            tx_payload_mb = min(
                float(payload_mb),
                compute_available_capacity_mb(
                    timeline_step.link_trace,
                    timeline_step.slot_seconds,
                    float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
                ),
            )
        else:
            tx_payload_mb = 0.0
        if tx_payload_mb <= 0.0:
            return 0.0
        rate_mb_s = estimate_rate_mb_per_s(
            timeline_step.link_trace,
            float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
        )
        if rate_mb_s <= float(self.simulator_cfg.get("epsilon", 1.0e-9)):
            return 0.0
        return compute_comm_energy_j(
            comm_power_w=float(self.simulator_cfg.get("tx_power_w", self.simulator_cfg.get("comm_power_w", 0.0))),
            tx_time_s=tx_payload_mb / rate_mb_s,
            link_quality=timeline_step.link_trace.link_quality,
            alpha_gamma=float(self.simulator_cfg.get("comm_alpha_gamma", 0.0)),
        )

    def _evaluate_payload_execution_feasibility(
        self,
        *,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        payload,
        lifecycle_state: ProductLifecycleState | None = None,
        task=None,
        operation: str = "",
        require_same_slot_delivery: bool = False,
        remaining_contact_capacity_mb: float = 0.0,
        downlink_throttled: bool = False,
        allow_no_contact_enqueue: bool | None = None,
        predict_immediate_downlink: bool = True,
        allow_existing_pending: bool = True,
        baseline_compute_energy_j: float = 0.0,
        baseline_comm_energy_j: float = 0.0,
        baseline_compute_heat_gain_scale: float = 1.0,
    ) -> PayloadExecutionFeasibilityDecision:
        guard_cfg = self._followup_guard_cfg()
        if allow_no_contact_enqueue is None:
            allow_no_contact_enqueue = bool(guard_cfg["allow_no_contact_enqueue"])
        payload_size_mb = max(0.0, float(getattr(payload, "payload_mb", 0.0) or 0.0))
        product_type = str(getattr(payload, "product_type", "") or getattr(payload, "payload_policy", "") or "")
        task_id = str(getattr(payload, "task_id", "") or getattr(task, "task_id", "") or "")
        queue_max_mb = float(comm_state.queue_max_mb)
        predicted_queue_next_mb = float(comm_state.queue_mb) + payload_size_mb
        extra_comm_energy_j = self._predict_extra_payload_comm_energy_j(
            timeline_step=timeline_step,
            payload_mb=payload_size_mb,
            require_same_slot_delivery=require_same_slot_delivery,
            predict_immediate_downlink=predict_immediate_downlink,
            remaining_contact_capacity_mb=remaining_contact_capacity_mb,
        )
        predicted_battery_next = update_battery_soc(
            battery_soc=resource_state.battery_soc,
            solar_energy_j=timeline_step.resource_trace.solar_energy_j,
            base_power_w=timeline_step.resource_trace.base_power_w,
            slot_seconds=timeline_step.slot_seconds,
            compute_energy_j=max(0.0, float(baseline_compute_energy_j)),
            comm_energy_j=max(0.0, float(baseline_comm_energy_j)) + extra_comm_energy_j,
            battery_capacity_j=float(self.simulator_cfg.get("battery_capacity_j", 1.0e12)),
            charge_efficiency=float(self.simulator_cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(self.simulator_cfg.get("battery_discharge_efficiency", 1.0)),
        )
        predicted_thermal_next = self._estimate_next_thermal(
            timeline_step=timeline_step,
            resource_state=resource_state,
            compute_energy_j=max(0.0, float(baseline_compute_energy_j)),
            comm_energy_j=max(0.0, float(baseline_comm_energy_j)) + extra_comm_energy_j,
            compute_heat_gain_scale=float(baseline_compute_heat_gain_scale),
        )
        fallback_deadline_s = float(getattr(payload, "deadline_s", getattr(task, "deadline_s", 0.0)) or 0.0)
        remaining_deadline_s = self._remaining_deadline_for_task(
            task,
            timeline_step.slot_index,
            timeline_step.slot_seconds,
            fallback_deadline_s,
        )
        min_remaining_deadline_s = float(guard_cfg["min_remaining_deadline_s"])
        queued_keys = self._queued_product_keys()

        blockers: list[str] = []
        if require_same_slot_delivery and downlink_throttled:
            blockers.append("followup_hard_safety_blocked_downlink_throttled")
        if require_same_slot_delivery and not timeline_step.link_trace.in_contact:
            blockers.append("followup_hard_safety_blocked_no_contact")
        if require_same_slot_delivery and payload_size_mb > max(0.0, float(remaining_contact_capacity_mb)) + float(
            self.simulator_cfg.get("epsilon", 1.0e-9)
        ):
            blockers.append("followup_hard_safety_blocked_capacity")
        if not require_same_slot_delivery and not timeline_step.link_trace.in_contact and not allow_no_contact_enqueue:
            blockers.append("followup_hard_safety_blocked_no_contact")
        if predicted_queue_next_mb > queue_max_mb + float(self.simulator_cfg.get("epsilon", 1.0e-9)):
            blockers.append("followup_hard_safety_blocked_queue")
        if predicted_battery_next < float(self.simulator_cfg.get("battery_min", 0.0)) - float(self.simulator_cfg.get("epsilon", 1.0e-9)):
            blockers.append("followup_hard_safety_blocked_battery")
        if predicted_thermal_next > float(self.simulator_cfg.get("thermal_max", 1.0)) + float(self.simulator_cfg.get("epsilon", 1.0e-9)):
            blockers.append("followup_hard_safety_blocked_thermal")
        if remaining_deadline_s <= 0.0:
            blockers.append("followup_hard_safety_blocked_expired")
        elif remaining_deadline_s <= min_remaining_deadline_s:
            blockers.append("followup_hard_safety_blocked_deadline")
        if lifecycle_state is not None and bool(lifecycle_state.task_closed_flag):
            blockers.append("followup_hard_safety_blocked_closed")
        if lifecycle_state is not None and product_type:
            duplicate_sources: list[str] = []
            if product_type in lifecycle_state.delivered_products:
                duplicate_sources.append("delivered")
            if not allow_existing_pending and product_type in lifecycle_state.pending_products:
                duplicate_sources.append("pending")
            if task_id and self._candidate_key(task_id, product_type) in queued_keys:
                duplicate_sources.append("queued")
            if duplicate_sources:
                blockers.append("followup_hard_safety_blocked_duplicate")
                for duplicate_source in duplicate_sources:
                    self._record_duplicate_suppression_trace(
                        timeline_step=timeline_step,
                        task_id=task_id,
                        product_type=product_type,
                        reason_code="followup_hard_safety_blocked_duplicate",
                        duplicate_source=duplicate_source,
                    )

        fields_used = (
            "payload_size_mb",
            "remaining_contact_capacity_mb",
            "queue_mb",
            "queue_max_mb",
            "battery_soc",
            "battery_min",
            "thermal_pressure",
            "thermal_max",
            "remaining_deadline_s",
            "min_remaining_deadline_s",
            "task_closed_flag",
            "delivered_products",
            "pending_products",
            "queued_product_identity_keys",
            "downlink_throttled",
            "contact_available",
            "baseline_compute_energy_j",
            "baseline_comm_energy_j",
        )
        unique_blockers = tuple(dict.fromkeys(blockers))
        decision = PayloadExecutionFeasibilityDecision(
            allowed=not unique_blockers,
            reason_code="followup_hard_safety_allowed" if not unique_blockers else unique_blockers[0],
            blockers=unique_blockers,
            predicted_queue_next_mb=predicted_queue_next_mb,
            predicted_battery_next=predicted_battery_next,
            predicted_thermal_next=predicted_thermal_next,
            extra_comm_energy_j=extra_comm_energy_j,
            extra_payload_mb=payload_size_mb,
            fields_used=fields_used,
        )
        if not hasattr(self, "last_followup_hard_safety_guard_trace"):
            self.last_followup_hard_safety_guard_trace = []
        self.last_followup_hard_safety_guard_trace.append(
            {
                "slot": int(timeline_step.slot_index),
                "operation": operation,
                "task_id": task_id,
                "product_type": product_type,
                "allowed": int(decision.allowed),
                "reason_code": decision.reason_code,
                "blockers": self._format_products(decision.blockers),
                "predicted_queue_next_mb": f"{decision.predicted_queue_next_mb:.6f}",
                "predicted_battery_next": f"{decision.predicted_battery_next:.6f}",
                "predicted_thermal_next": f"{decision.predicted_thermal_next:.6f}",
                "extra_comm_energy_j": f"{decision.extra_comm_energy_j:.6f}",
                "baseline_compute_energy_j": f"{float(baseline_compute_energy_j):.6f}",
                "baseline_comm_energy_j": f"{float(baseline_comm_energy_j):.6f}",
                "extra_payload_mb": f"{decision.extra_payload_mb:.6f}",
                "remaining_contact_capacity_mb": f"{float(remaining_contact_capacity_mb):.6f}",
                "remaining_deadline_s": f"{remaining_deadline_s:.6f}",
                "downlink_throttled": int(bool(downlink_throttled)),
                "contact_available": int(bool(timeline_step.link_trace.in_contact)),
                "fields_used": self._format_products(decision.fields_used),
            }
        )
        return decision

    def _followup_hard_safety_guard(self, **kwargs) -> PayloadExecutionFeasibilityDecision:
        return self._evaluate_payload_execution_feasibility(**kwargs)

    @staticmethod
    def _slot_from_time(time_s: float | None, slot_seconds: float) -> int:
        if time_s is None or slot_seconds <= 0.0:
            return -1
        return int(float(time_s) // float(slot_seconds))

    def _lifecycle_candidate_flags(
        self,
        timeline_step: TimelineStep,
        lifecycle_state: ProductLifecycleState,
        residual_utility_threshold: float,
    ) -> dict[str, object]:
        delivered_products = set(lifecycle_state.delivered_products)
        pending_products = set(lifecycle_state.pending_products)
        residual_pending = sorted(product for product in pending_products if product in RESIDUAL_PRODUCT_TYPES)
        residual_product_type = residual_pending[0] if len(residual_pending) == 1 else ""
        queued_keys = self._queued_product_keys()
        expired = bool(timeline_step.task is not None and self._remaining_deadline_s(timeline_step) <= 0.0)
        metadata_delivered = "metadata" in delivered_products
        roi_delivered_for_second_stage = "roi" in delivered_products and "roi_context" in pending_products
        pending_residual_exists = bool(residual_product_type)
        residual_already_delivered = bool(residual_product_type and residual_product_type in delivered_products)
        residual_already_queued = bool(
            residual_product_type
            and f"{lifecycle_state.task_id}:{residual_product_type}" in queued_keys
        )
        true_candidate = (
            (metadata_delivered or roi_delivered_for_second_stage)
            and pending_residual_exists
            and float(lifecycle_state.residual_utility_remaining) > float(residual_utility_threshold)
            and not expired
            and not bool(lifecycle_state.task_closed_flag)
            and not residual_already_delivered
            and not residual_already_queued
            and int(lifecycle_state.residual_enqueued_count or 0) == 0
            and int(lifecycle_state.residual_delivered_count or 0) == 0
        )
        return {
            "task_expired_flag": int(expired),
            "task_closed_flag": int(bool(lifecycle_state.task_closed_flag)),
            "metadata_delivered_flag": int(metadata_delivered),
            "pending_residual_exists_flag": int(pending_residual_exists),
            "true_candidate_flag": int(true_candidate),
        }

    def _roi_context_after_roi_cfg(self) -> dict[str, object]:
        progressive = self.simulator_cfg.get("progressive_payload", {})
        nested = progressive.get("roi_context_after_roi", {}) if isinstance(progressive, dict) else {}
        if not isinstance(nested, dict):
            nested = {}
        return {
            "enabled": bool(nested.get("enabled", False)),
            "residual_utility_threshold": float(nested.get("residual_utility_threshold", 0.08)),
            "parent_payload_plan": str(nested.get("parent_payload_plan", "metadata_then_roi_then_roi_context")),
        }

    def _maybe_register_roi_context_after_roi(
        self,
        *,
        timeline_step: TimelineStep,
        lifecycle_state: ProductLifecycleState,
        task,
        detection_result,
        router: BaseRouter | None,
        resource_state: ResourceState | None,
        comm_state: CommState | None,
    ) -> bool:
        self._ensure_staged_delivery_state()
        cfg = self._roi_context_after_roi_cfg()
        if not bool(cfg.get("enabled", False)):
            return False
        if task is None or detection_result is None or resource_state is None or comm_state is None:
            return False
        product_requirement = str(lifecycle_state.product_requirement or task.product_requirement).strip().lower()
        value_level = str(getattr(task, "value_level", "") or (task.metadata or {}).get("value_level", "")).strip().lower()
        if product_requirement not in {"context", "evidence"} or value_level not in {"high", "urgent"}:
            return False
        if "roi" not in lifecycle_state.delivered_products or "roi_context" in lifecycle_state.delivered_products:
            if "roi_context" in lifecycle_state.delivered_products:
                self._record_duplicate_suppression_trace(
                    timeline_step=timeline_step,
                    task_id=task.task_id,
                    product_type="roi_context",
                    reason_code="duplicate_residual_already_delivered",
                    duplicate_source="delivered",
                )
            return False
        if "roi_context" in lifecycle_state.pending_products:
            self._record_duplicate_suppression_trace(
                timeline_step=timeline_step,
                task_id=task.task_id,
                product_type="roi_context",
                reason_code="duplicate_residual_already_pending",
                duplicate_source="pending",
            )
            return False
        queued = self._queued_product_keys()
        if self._candidate_key(task.task_id, "roi_context") in queued:
            self._record_duplicate_suppression_trace(
                timeline_step=timeline_step,
                task_id=task.task_id,
                product_type="roi_context",
                reason_code="duplicate_residual_already_queued",
                duplicate_source="queued",
            )
            return False
        remaining_deadline_s = self._remaining_deadline_s(timeline_step)
        if remaining_deadline_s <= 0.0 or lifecycle_state.task_closed_flag:
            return False
        residual_remaining = residual_information_utility(lifecycle_state.delivered_products, product_requirement)
        if residual_remaining <= float(cfg.get("residual_utility_threshold", 0.08)):
            return False
        payload_size_mb, _diag = estimate_payload_size_with_diagnostics(
            "roi_context",
            detection_result=None,
            task=None,
            payload_profiles_cfg=self.payload_profiles_cfg,
        )
        timeliness = max(0.0, min(1.0, remaining_deadline_s / max(float(task.deadline_s), 1.0e-9)))
        future_contact = self._roi_context_future_contact_feasibility(
            timeline_step=timeline_step,
            task=task,
            payload_size_mb=payload_size_mb,
            comm_state=comm_state,
            start_after_current=not bool(timeline_step.link_trace.in_contact),
        )
        future_contact_feasible = bool(future_contact.get("future_contact_feasible", 0))
        _gate_enabled, _profile_name, gate_profile = load_gate_profile(self.simulator_cfg)
        allow_queued_without_current = bool(gate_profile.get("allow_queued_followup_without_current_contact", False))
        contact_score = 1.0 if timeline_step.link_trace.in_contact or (allow_queued_without_current and future_contact_feasible) else 0.0
        detection_quality = float(getattr(detection_result, "estimated_quality", 1.0))
        information_density = residual_remaining / max(payload_size_mb, 1.0e-9)
        value_density = (
            task_effective_value(task)
            * residual_remaining
            * detection_quality
            * timeliness
            / max(payload_size_mb, 1.0e-9)
        )
        eriv_score = (
            task_effective_value(task)
            * residual_remaining
            * detection_quality
            * timeliness
            * contact_score
            / max(payload_size_mb, 1.0e-9)
        )
        self._record_value_density_trace(
            timeline_step=timeline_step,
            task_id=task.task_id,
            stage="roi_context_after_roi_registration",
            product_type="roi_context",
            residual_delta_u_i=residual_remaining,
            detection_quality=detection_quality,
            timeliness_factor=timeliness,
            payload_size_mb=payload_size_mb,
            value_density=value_density,
            information_density=information_density,
            eriv_score=eriv_score,
            followup_mode="queued",
        )
        budget_counts = self._roi_context_budget_counts(timeline_step, "queued")
        gate_decision = evaluate_roi_context_gate(
            product_requirement=product_requirement,
            value_level=value_level,
            value_density=value_density,
            system_state={
                "queue_mb": comm_state.queue_mb,
                "queue_max_mb": comm_state.queue_max_mb,
                "queue_pressure": comm_state.queue_mb / max(comm_state.queue_max_mb, 1.0e-9),
                "battery_soc": resource_state.battery_soc,
                "thermal_pressure": resource_state.thermal_pressure,
                "remaining_deadline_s": remaining_deadline_s,
                "contact_available": timeline_step.link_trace.in_contact,
                "contact_opportunity_score": contact_score,
                "candidate_rank": int(getattr(self, "_roi_context_window_selected_count", 0)) + 1,
                "future_contact_feasible": future_contact_feasible,
                **budget_counts,
            },
            config=self.simulator_cfg,
            eriv_score=eriv_score,
            residual_product_type="roi_context",
        )
        self._record_roi_context_window_budget_trace(
            timeline_step=timeline_step,
            task_id=task.task_id,
            candidate_name="roi_context_after_roi",
            candidate_payload="roi_context",
            candidate_rank=int(getattr(self, "_roi_context_window_selected_count", 0)) + 1,
            window_selected_count=int(budget_counts["window_selected_count"]),
            decision=gate_decision,
            stage="roi_context_after_roi_registration",
            followup_mode="queued",
        )
        if not gate_decision.allowed:
            self._record_same_slot_vs_queued_followup_trace(
                timeline_step=timeline_step,
                task_id=task.task_id,
                stage="roi_context_after_roi_registration",
                followup_mode="queued",
                action="blocked",
                reason_code=gate_decision.reason_code,
                payload_size_mb=payload_size_mb,
                same_slot_delivery_allowed=False,
                queued_followup_allowed=False,
                future_contact_feasible=future_contact_feasible,
            )
            return False
        residual_payload_for_guard = generate_residual_payload(
            task=task,
            detection_result=detection_result,
            residual_product_type="roi_context",
            slot_index=timeline_step.slot_index,
            parent_payload_plan=str(cfg.get("parent_payload_plan", "metadata_then_roi_then_roi_context")),
            high_value_threshold=float(
                getattr(self, "reward_cfg", {}).get("high_value_threshold", self.simulator_cfg.get("high_value_threshold", 0.75))
            ),
            payload_profiles_cfg=self.payload_profiles_cfg,
            delivered_products=set(lifecycle_state.delivered_products),
            remaining_deadline_s=remaining_deadline_s,
            continue_request_priority_bonus=0.0,
            followup_trigger="proactive_followup",
        )
        guard_decision = self._followup_hard_safety_guard(
            timeline_step=timeline_step,
            resource_state=resource_state,
            comm_state=comm_state,
            payload=residual_payload_for_guard,
            lifecycle_state=lifecycle_state,
            task=task,
            operation="roi_context_after_roi_queue_admission",
            require_same_slot_delivery=False,
            allow_no_contact_enqueue=True,
            predict_immediate_downlink=False,
            allow_existing_pending=True,
        )
        if not guard_decision.allowed:
            self._record_same_slot_vs_queued_followup_trace(
                timeline_step=timeline_step,
                task_id=task.task_id,
                stage="roi_context_after_roi_registration",
                followup_mode="queued",
                action="blocked_by_hard_safety_guard",
                reason_code=guard_decision.reason_code,
                payload_size_mb=payload_size_mb,
                same_slot_delivery_allowed=False,
                queued_followup_allowed=False,
                future_contact_feasible=future_contact_feasible,
            )
            return False
        self._increment_roi_context_budget(timeline_step, "queued")
        lifecycle_state.pending_products.add("roi_context")
        lifecycle_state.residual_parent_payload_plan = str(cfg.get("parent_payload_plan", "metadata_then_roi_then_roi_context"))
        lifecycle_state.pending_residual_registered_slot = int(timeline_step.slot_index)
        lifecycle_state.residual_utility_remaining = residual_remaining
        passive_record = self.lifecycle_ledger.create_record(
            task_id=task.task_id,
            product_type="roi_context",
            parent_product_type="roi",
            request_id=f"roi_context_after_roi_{int(timeline_step.slot_index):05d}_{task.task_id}",
            slot=int(timeline_step.slot_index),
            value_level=value_level,
            product_requirement=product_requirement,
            calibrated_task_value=float(task_effective_value(task)),
            effective_task_value=float(task_effective_value(task)),
            residual_delta_information_utility=float(residual_remaining),
            payload_size_mb=float(payload_size_mb),
            deadline_slot=self._staged_request_deadline_slot(timeline_step, task),
            priority_score=float(eriv_score),
            priority_components={
                "marginal_utility_per_mb": information_density,
                "value_density_priority": value_density,
                "deadline_urgency": timeliness,
            },
            staged_queue_id=self._candidate_key(task.task_id, "roi_context"),
            source="roi_context_after_roi_pending_registration",
            set_active=True,
            transition_reason="roi_context_after_roi_request_created",
        )
        self.lifecycle_ledger.transition(
            passive_record.lifecycle_id,
            "pending",
            slot=int(timeline_step.slot_index),
            reason="roi_context_after_roi_registered_pending",
            staged_queue_id=self._candidate_key(task.task_id, "roi_context"),
            priority_score=float(eriv_score),
            priority_components=passive_record.priority_components,
            contact_window_id=self._contact_window_id_for_slot(int(timeline_step.slot_index)),
            next_contact_window_id_candidate=str(future_contact.get("next_eligible_contact_window_id", "")),
        )
        self._record_same_slot_vs_queued_followup_trace(
            timeline_step=timeline_step,
            task_id=task.task_id,
            stage="roi_context_after_roi_registration",
            followup_mode="queued",
            action="registered_pending",
            reason_code="queued_followup_registered",
            payload_size_mb=payload_size_mb,
            same_slot_delivery_allowed=bool(timeline_step.link_trace.in_contact),
            queued_followup_allowed=True,
            future_contact_feasible=future_contact_feasible,
            enqueue_flag=0,
            delivery_flag=0,
        )
        return True

    def _integrated_branch_closure_satisfied(self, lifecycle_state: ProductLifecycleState) -> bool:
        if not bool(self._roi_context_after_roi_cfg().get("enabled", False)):
            return False
        requirement = str(lifecycle_state.product_requirement or "").strip().lower()
        delivered = set(lifecycle_state.delivered_products)
        if requirement == "alert":
            return "metadata" in delivered or "roi" in delivered or "roi_context" in delivered
        if requirement == "verification":
            return "roi" in delivered or "roi_context" in delivered
        if requirement in {"context", "evidence"}:
            return "roi_context" in delivered
        return False

    def _build_phase6_pre_eval_log(
        self,
        timeline_step: TimelineStep,
        lifecycle_state: ProductLifecycleState,
    ) -> dict[str, object]:
        phase6_cfg = phase6_proactive_config(self.simulator_cfg)
        residual_threshold = float(phase6_cfg.get("residual_utility_threshold", 0.0) or 0.0)
        flags = self._lifecycle_candidate_flags(timeline_step, lifecycle_state, residual_threshold)
        return {
            "lifecycle_pre_delivered_products": self._format_products(lifecycle_state.delivered_products),
            "lifecycle_pre_pending_products": self._format_products(lifecycle_state.pending_products),
            "lifecycle_pre_residual_utility_remaining": float(lifecycle_state.residual_utility_remaining),
            "lifecycle_pre_task_expired_flag": int(flags["task_expired_flag"]),
            "lifecycle_pre_task_closed_flag": int(flags["task_closed_flag"]),
            "lifecycle_pre_metadata_delivered_flag": int(flags["metadata_delivered_flag"]),
            "lifecycle_pre_pending_residual_exists_flag": int(flags["pending_residual_exists_flag"]),
            "lifecycle_pre_true_candidate_flag": int(flags["true_candidate_flag"]),
        }

    def _build_lifecycle_post_update_log(
        self,
        timeline_step: TimelineStep,
        lifecycle_state: ProductLifecycleState,
    ) -> dict[str, object]:
        phase6_cfg = phase6_proactive_config(self.simulator_cfg)
        residual_threshold = float(phase6_cfg.get("residual_utility_threshold", 0.0) or 0.0)
        flags = self._lifecycle_candidate_flags(timeline_step, lifecycle_state, residual_threshold)
        return {
            "lifecycle_post_delivered_products": self._format_products(lifecycle_state.delivered_products),
            "lifecycle_post_pending_products": self._format_products(lifecycle_state.pending_products),
            "lifecycle_post_residual_utility_remaining": float(lifecycle_state.residual_utility_remaining),
            "lifecycle_post_metadata_delivery_slot": (
                -1 if lifecycle_state.metadata_delivery_slot is None else int(lifecycle_state.metadata_delivery_slot)
            ),
            "lifecycle_post_pending_residual_registered_slot": (
                -1
                if lifecycle_state.pending_residual_registered_slot is None
                else int(lifecycle_state.pending_residual_registered_slot)
            ),
            "lifecycle_post_true_candidate_flag": int(flags["true_candidate_flag"]),
        }

    @staticmethod
    def _empty_carryover_pool_log() -> dict[str, int | float]:
        return {
            "residual_candidate_pool_size": 0,
            "candidate_added_count": 0,
            "candidate_removed_count": 0,
            "candidate_re_evaluated_count": 0,
            "candidate_expired_count": 0,
            "candidate_closed_count": 0,
            "candidate_duplicate_suppressed_count": 0,
            "residual_candidate_pool_add_count": 0,
            "residual_candidate_pool_size_max": 0,
            "residual_candidate_re_evaluation_count": 0,
            "residual_candidate_re_evaluation_after_metadata_delivery_count": 0,
            "residual_candidate_true_count": 0,
            "residual_candidate_expired_count": 0,
            "residual_candidate_removed_count": 0,
            "residual_candidate_duplicate_suppression_count": 0,
            "residual_candidate_age_s": 0.0,
            "carryover_proactive_attempt_count": 0,
            "carryover_proactive_enqueue_count": 0,
            "carryover_proactive_delivery_count": 0,
            "carryover_proactive_incremental_utility": 0.0,
            "carryover_before_passive_request_count": 0,
            "carryover_passive_request_suppressed_count": 0,
            "carryover_eriv": 0.0,
            "carryover_eriv_selected": 0.0,
            "carryover_contact_block_count": 0,
            "carryover_thermal_block_count": 0,
            "carryover_queue_block_count": 0,
            "carryover_deadline_block_count": 0,
            "carryover_energy_block_count": 0,
            "same_slot_candidate_detected_count": 0,
            "same_slot_candidate_evaluated_count": 0,
            "same_slot_carryover_enqueue_count": 0,
            "same_slot_carryover_delivery_count": 0,
            "same_slot_carryover_duplicate_block_count": 0,
            "same_slot_carryover_contact_block_count": 0,
            "same_slot_opportunity_saved_from_tplus1_count": 0,
            "same_slot_fallback_to_ordinary_queue_count": 0,
            "ordinary_queue_fallback_attempt_count": 0,
            "ordinary_queue_fallback_success_count": 0,
            "duplicate_residual_suppressed_after_fallback_count": 0,
            "zero_size_residual_payload_blocked_count": 0,
            "residual_fallback_blocked_by_expired_deadline_count": 0,
            "residual_fallback_dropped_by_queue_overflow_count": 0,
        }

    def _copy_lifecycle_boundary_logs_to_feedback(
        self,
        feedback: ObservedFeedback,
        post_log: dict[str, object] | None = None,
        pool_log: dict[str, int | float] | None = None,
    ) -> None:
        for key, value in self.last_phase6_pre_eval_log.items():
            setattr(feedback, key, value)
        for key, value in (post_log or {}).items():
            setattr(feedback, key, value)
        for key, value in (pool_log or self._empty_carryover_pool_log()).items():
            setattr(feedback, key, value)

    def _bump_carryover_log(self, key: str, value: int | float = 1) -> None:
        current = self.last_carryover_pool_log.get(key, 0)
        self.last_carryover_pool_log[key] = current + value

    def _set_carryover_log_max(self, key: str, value: int | float) -> None:
        self.last_carryover_pool_log[key] = max(float(self.last_carryover_pool_log.get(key, 0)), float(value))

    def _full_proposed_router(self, router: BaseRouter | None) -> bool:
        if router is None:
            return False
        phase6_cfg = phase6_proactive_config(self.simulator_cfg)
        aliases = set(DEFAULT_FULL_METHOD_ALIASES)
        configured = phase6_cfg.get("full_method_aliases", ())
        if isinstance(configured, str):
            aliases.update(item.strip() for item in configured.split(",") if item.strip())
        else:
            try:
                aliases.update(str(item) for item in configured)
            except TypeError:
                pass
        router_name = str(getattr(router, "router_name", ""))
        if router_name in aliases:
            return True
        return bool(
            router_name.startswith("proposed_")
            and getattr(router, "use_lyapunov_pricing", False)
            and getattr(router, "use_mode_controller", False)
        )

    def _carryover_behavior_enabled(self, router: BaseRouter | None) -> bool:
        phase6_cfg = phase6_proactive_config(self.simulator_cfg)
        carryover_cfg = phase6_residual_candidate_carryover_config(self.simulator_cfg)
        return bool(phase6_cfg.get("enabled", False)) and bool(carryover_cfg.get("enabled", False)) and self._full_proposed_router(router)

    def _same_slot_carryover_enabled(self, router: BaseRouter | None) -> bool:
        same_slot_cfg = phase6_same_slot_carryover_config(self.simulator_cfg)
        if not self._carryover_behavior_enabled(router):
            return False
        if not bool(same_slot_cfg.get("enabled", False)):
            return False
        if not bool(same_slot_cfg.get("only_after_lifecycle_update", True)):
            return False
        if not bool(same_slot_cfg.get("reuse_existing_guards", True)):
            return False
        if not bool(same_slot_cfg.get("enforce_no_reward_change", True)):
            return False
        if not bool(same_slot_cfg.get("enforce_no_queue_policy_change", True)):
            return False
        if not bool(same_slot_cfg.get("strict_mode", True)):
            return False
        return True

    def _same_slot_remaining_capacity(
        self,
        timeline_step: TimelineStep,
        downlink_result: DownlinkResult,
    ) -> tuple[bool, float, float, str]:
        if not timeline_step.link_trace.in_contact:
            return False, 0.0, 0.0, "no_contact_opportunity"
        rate_mb_s = estimate_rate_mb_per_s(
            timeline_step.link_trace,
            float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
        )
        if rate_mb_s <= float(self.simulator_cfg.get("epsilon", 1.0e-9)):
            return False, 0.0, 0.0, "contact_capacity_unavailable"
        full_capacity_mb = compute_available_capacity_mb(
            timeline_step.link_trace,
            timeline_step.slot_seconds,
            float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
        )
        remaining_capacity_mb = max(0.0, full_capacity_mb - float(downlink_result.S_down_total_mb))
        remaining_contact_s = remaining_capacity_mb / max(rate_mb_s, 1.0e-9)
        if remaining_capacity_mb <= float(self.simulator_cfg.get("epsilon", 1.0e-9)):
            return False, 0.0, remaining_contact_s, "no_remaining_contact_capacity"
        return True, remaining_capacity_mb, remaining_contact_s, ""

    def _record_same_slot_carryover_trace(
        self,
        event_type: str,
        timeline_step: TimelineStep,
        candidate: ResidualCandidateState,
        *,
        decision=None,
        remaining_capacity_mb: float = 0.0,
        remaining_contact_s: float = 0.0,
        enqueue_flag: int = 0,
        delivery_flag: int = 0,
        reason_code: str = "",
    ) -> None:
        blockers = () if decision is None else tuple(decision.blockers)
        eriv = 0.0 if decision is None else float(decision.eriv_score)
        threshold = 0.0 if decision is None else float(decision.threshold)
        remaining_deadline_s = self._remaining_deadline_for_task(
            candidate.task_snapshot,
            timeline_step.slot_index,
            timeline_step.slot_seconds,
            candidate.deadline_s,
        )
        self.last_same_slot_carryover_trace.append(
            {
                "slot": int(timeline_step.slot_index),
                "time_s": float(timeline_step.slot_index) * float(timeline_step.slot_seconds),
                "event_type": event_type,
                "candidate_id": candidate.duplicate_identity_key,
                "task_id": candidate.task_id,
                "residual_product_type": candidate.residual_product_type,
                "ERIV": f"{eriv:.6f}",
                "ERIV_threshold": f"{threshold:.6f}",
                "contact_remaining_s": f"{remaining_contact_s:.6f}",
                "contact_remaining_capacity_mb": f"{remaining_capacity_mb:.6f}",
                "remaining_deadline_s": f"{remaining_deadline_s:.6f}",
                "enqueue_decision": int(enqueue_flag),
                "delivery_decision": int(delivery_flag),
                "reason_code": reason_code or ("" if decision is None else decision.reason_code),
                "blockers": self._format_products(blockers),
                "leakage_safe_flag": int(candidate.leakage_safe_flag and (decision is None or decision.leakage_safe_flag)),
            }
        )

    @staticmethod
    def _candidate_key(task_id: str, residual_product_type: str) -> str:
        return f"{task_id}:{residual_product_type}"

    @staticmethod
    def _remaining_deadline_for_task(task, slot_index: int, slot_seconds: float, fallback_deadline_s: float) -> float:
        if task is None:
            return max(0.0, float(fallback_deadline_s))
        elapsed_s = max(0.0, (float(slot_index) - float(task.arrival_slot)) * float(slot_seconds))
        return max(0.0, float(task.deadline_s) - elapsed_s)

    @staticmethod
    def _candidate_age_s(candidate: ResidualCandidateState, slot_index: int, slot_seconds: float) -> float:
        return max(0.0, (float(slot_index) - float(candidate.created_slot)) * float(slot_seconds))

    def _record_carryover_event(
        self,
        event_type: str,
        timeline_step: TimelineStep,
        candidate: ResidualCandidateState,
        *,
        resource_state: ResourceState | None = None,
        comm_state: CommState | None = None,
        decision=None,
        candidate_added_flag: int = 0,
        candidate_re_evaluated_flag: int = 0,
        candidate_removed_flag: int = 0,
        removal_reason: str = "",
        proactive_enqueue_flag: int = 0,
        proactive_delivery_flag: int = 0,
        passive_request_flag: int = 0,
        passive_request_suppressed_flag: int = 0,
        duplicate_suppression_flag: int = 0,
    ) -> None:
        eriv = 0.0 if decision is None else float(decision.eriv_score)
        threshold = 0.0 if decision is None else float(decision.threshold)
        blockers = () if decision is None else tuple(decision.blockers)
        fields_used = () if decision is None else tuple(decision.fields_used)
        remaining_deadline_s = self._remaining_deadline_for_task(
            candidate.task_snapshot,
            timeline_step.slot_index,
            timeline_step.slot_seconds,
            candidate.deadline_s,
        )
        queue_mb = 0.0 if comm_state is None else float(comm_state.queue_mb)
        contact_available = bool(timeline_step.link_trace.in_contact)
        battery_soc = 0.0 if resource_state is None else float(resource_state.battery_soc)
        thermal_pressure = 0.0 if resource_state is None else float(resource_state.thermal_pressure)
        self.last_carryover_lifecycle_events.append(
            {
                "slot": timeline_step.slot_index,
                "task_id": candidate.task_id,
                "scene_id": candidate.scene_id,
                "event_type": event_type,
                "payload_plan": candidate.payload_plan,
                "residual_product_type": candidate.residual_product_type,
                "product_requirement": candidate.product_requirement,
                "task_value": candidate.task_value,
                "high_value_flag": int(candidate.high_value_flag),
                "emergency_flag": int(candidate.emergency_flag),
                "value_level": candidate.value_level,
                "metadata_delivery_slot": candidate.metadata_delivery_slot,
                "pending_residual_registered_slot": candidate.pending_residual_registered_slot,
                "candidate_added_flag": candidate_added_flag,
                "candidate_re_evaluated_flag": candidate_re_evaluated_flag,
                "candidate_removed_flag": candidate_removed_flag,
                "removal_reason": removal_reason,
                "candidate_age_s": f"{self._candidate_age_s(candidate, timeline_step.slot_index, timeline_step.slot_seconds):.6f}",
                "delivered_products_before_eval": self._format_products(candidate.delivered_products_snapshot),
                "pending_products_before_eval": self._format_products(candidate.pending_products_snapshot),
                "residual_utility_remaining": f"{candidate.residual_utility_remaining:.6f}",
                "residual_delta_u_I": f"{candidate.residual_delta_u_I:.6f}",
                "ERIV": f"{eriv:.6f}",
                "ERIV_threshold": f"{threshold:.6f}",
                "ERIV_above_threshold_flag": int(eriv >= threshold and threshold > 0.0),
                "proactive_flag": int(bool(decision is not None and decision.proactive_flag)),
                "proactive_blockers": self._format_products(blockers),
                "proactive_enqueue_flag": proactive_enqueue_flag,
                "proactive_delivery_flag": proactive_delivery_flag,
                "passive_request_flag": passive_request_flag,
                "passive_request_suppressed_flag": passive_request_suppressed_flag,
                "duplicate_suppression_flag": duplicate_suppression_flag,
                "queue_mb": f"{queue_mb:.6f}",
                "contact_available": int(contact_available),
                "remaining_deadline_s": f"{remaining_deadline_s:.6f}",
                "battery_soc": f"{battery_soc:.6f}",
                "thermal_pressure": f"{thermal_pressure:.6f}",
                "leakage_safe_flag": int(candidate.leakage_safe_flag and (decision is None or decision.leakage_safe_flag)),
                "fields_used": self._format_products(fields_used),
            }
        )

    def _candidate_invalid_reason(
        self,
        candidate: ResidualCandidateState,
        timeline_step: TimelineStep,
        cfg: dict[str, object],
    ) -> str:
        state = self.product_lifecycle.get(candidate.task_id)
        if state is None:
            return "task_missing"
        if bool(cfg.get("remove_on_closed", True)) and state.task_closed_flag:
            return "task_closed"
        if bool(cfg.get("remove_on_residual_delivered", True)) and (
            candidate.residual_product_type in state.delivered_products or state.residual_delivered_count > 0
        ):
            return "residual_delivered"
        remaining_deadline_s = self._remaining_deadline_for_task(
            candidate.task_snapshot,
            timeline_step.slot_index,
            timeline_step.slot_seconds,
            candidate.deadline_s,
        )
        if bool(cfg.get("remove_on_expired", True)) and remaining_deadline_s <= 0.0:
            return "candidate_expired"
        max_age_s = float(cfg.get("max_candidate_age_s", 600) or 600)
        if self._candidate_age_s(candidate, timeline_step.slot_index, timeline_step.slot_seconds) > max_age_s:
            return "candidate_age_exceeded"
        if candidate.residual_product_type not in state.pending_products:
            return "pending_residual_missing"
        residual_threshold = float(phase6_proactive_config(self.simulator_cfg).get("residual_utility_threshold", 0.08) or 0.08)
        if bool(cfg.get("require_positive_residual_utility", True)) and state.residual_utility_remaining <= residual_threshold:
            return "residual_utility_zero_or_below_threshold"
        queued_keys = self._queued_product_keys()
        duplicate_queued = (
            candidate.duplicate_identity_key in queued_keys
            or self._candidate_key(candidate.task_id, candidate.residual_product_type) in queued_keys
        )
        if duplicate_queued and bool(cfg.get("remove_on_duplicate_enqueued", False)):
            return "duplicate_residual_already_queued"
        return ""

    def _remove_candidate(
        self,
        key: str,
        reason: str,
        timeline_step: TimelineStep,
        *,
        resource_state: ResourceState | None = None,
        comm_state: CommState | None = None,
    ) -> None:
        candidate = self.residual_candidate_pool.pop(key, None)
        if candidate is None:
            return
        self.residual_candidate_history[key] = candidate
        self._bump_carryover_log("candidate_removed_count")
        self._bump_carryover_log("residual_candidate_removed_count")
        if reason in {"candidate_expired", "candidate_age_exceeded"}:
            self._bump_carryover_log("candidate_expired_count")
            self._bump_carryover_log("residual_candidate_expired_count")
            event_type = "candidate_expired"
        else:
            event_type = "candidate_removed"
        if reason == "task_closed":
            self._bump_carryover_log("candidate_closed_count")
        if "duplicate" in reason:
            self._bump_carryover_log("candidate_duplicate_suppressed_count")
            self._bump_carryover_log("residual_candidate_duplicate_suppression_count")
            event_type = "duplicate_suppressed"
        self._record_carryover_event(
            event_type,
            timeline_step,
            candidate,
            resource_state=resource_state,
            comm_state=comm_state,
            candidate_removed_flag=1,
            removal_reason=reason,
            duplicate_suppression_flag=int("duplicate" in reason),
        )

    def _add_residual_candidate_after_lifecycle(
        self,
        timeline_step: TimelineStep,
        lifecycle_state: ProductLifecycleState,
        payload,
        detection_result,
        router: BaseRouter | None,
    ) -> str:
        flags = self._lifecycle_candidate_flags(
            timeline_step,
            lifecycle_state,
            float(phase6_proactive_config(self.simulator_cfg).get("residual_utility_threshold", 0.08) or 0.08),
        )
        if flags["true_candidate_flag"]:
            self._bump_carryover_log("residual_candidate_true_count")
        if not self._carryover_behavior_enabled(router):
            self._set_carryover_log_max("residual_candidate_pool_size_max", len(self.residual_candidate_pool))
            return ""
        task = timeline_step.task
        if task is None or detection_result is None:
            return ""
        residual_pending = sorted(product for product in lifecycle_state.pending_products if product in RESIDUAL_PRODUCT_TYPES)
        if len(residual_pending) != 1 or not flags["true_candidate_flag"]:
            return ""
        residual_product_type = residual_pending[0]
        key = self._candidate_key(task.task_id, residual_product_type)
        if key in self.residual_candidate_pool:
            self._bump_carryover_log("candidate_duplicate_suppressed_count")
            self._bump_carryover_log("residual_candidate_duplicate_suppression_count")
            self._record_duplicate_suppression_trace(
                timeline_step=timeline_step,
                task_id=task.task_id,
                product_type=residual_product_type,
                reason_code="duplicate_residual_candidate_already_pending",
                duplicate_source="residual_candidate_pool",
            )
            return ""
        residual_delta = delta_information_utility(
            residual_product_type,
            set(lifecycle_state.delivered_products),
            lifecycle_state.product_requirement,
        )
        candidate = ResidualCandidateState(
            task_id=task.task_id,
            scene_id=task.scene_id,
            product_requirement=task.product_requirement,
            task_value=float(task_effective_value(task)),
            high_value_flag=bool(task.high_value_flag),
            emergency_flag=bool(task.emergency_flag),
            payload_plan=lifecycle_state.residual_parent_payload_plan or str(getattr(payload, "payload_plan", "")),
            residual_product_type=residual_product_type,
            metadata_delivery_slot=-1
            if lifecycle_state.metadata_delivery_slot is None
            else int(lifecycle_state.metadata_delivery_slot),
            metadata_delivery_time_s=lifecycle_state.first_alert_delivered_time_s,
            pending_residual_registered_slot=-1
            if lifecycle_state.pending_residual_registered_slot is None
            else int(lifecycle_state.pending_residual_registered_slot),
            residual_utility_remaining=float(lifecycle_state.residual_utility_remaining),
            residual_delta_u_I=float(residual_delta),
            deadline_s=float(task.deadline_s),
            created_slot=int(timeline_step.slot_index),
            delivered_products_snapshot=set(lifecycle_state.delivered_products),
            pending_products_snapshot=set(lifecycle_state.pending_products),
            duplicate_identity_key=key,
            task_snapshot=task,
            detection_result_snapshot=detection_result,
            value_level=str(getattr(task, "value_level", "") or (task.metadata or {}).get("value_level", "")),
        )
        self.residual_candidate_pool[key] = candidate
        self._bump_carryover_log("candidate_added_count")
        self._bump_carryover_log("residual_candidate_pool_add_count")
        self._set_carryover_log_max("residual_candidate_pool_size_max", len(self.residual_candidate_pool))
        self.last_carryover_pool_log["residual_candidate_pool_size"] = len(self.residual_candidate_pool)
        self._record_carryover_event("candidate_added", timeline_step, candidate, candidate_added_flag=1)
        return key

    def _evaluate_and_build_carryover_payload(
        self,
        key: str,
        candidate: ResidualCandidateState,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        router: BaseRouter,
        phase6_cfg,
        *,
        same_slot: bool = False,
    ):
        state = self.product_lifecycle.get(candidate.task_id)
        if state is None:
            return None, None, None
        candidate.delivered_products_snapshot = set(state.delivered_products)
        candidate.pending_products_snapshot = set(state.pending_products)
        candidate.residual_utility_remaining = float(state.residual_utility_remaining)
        candidate.residual_delta_u_I = delta_information_utility(
            candidate.residual_product_type,
            set(state.delivered_products),
            state.product_requirement,
        )
        candidate.candidate_age_s = self._candidate_age_s(candidate, timeline_step.slot_index, timeline_step.slot_seconds)
        candidate.last_evaluated_slot = int(timeline_step.slot_index)
        candidate.evaluation_count += 1

        payload_size_mb, _diagnostics = estimate_payload_size_with_diagnostics(
            candidate.residual_product_type,
            detection_result=None,
            task=None,
            payload_profiles_cfg=self.payload_profiles_cfg,
        )
        remaining_deadline_s = self._remaining_deadline_for_task(
            candidate.task_snapshot,
            timeline_step.slot_index,
            timeline_step.slot_seconds,
            candidate.deadline_s,
        )
        queue_pressure = comm_state.queue_mb / max(comm_state.queue_max_mb, 1.0e-9)
        future_contact = self._roi_context_future_contact_feasibility(
            timeline_step=timeline_step,
            task=candidate.task_snapshot,
            payload_size_mb=payload_size_mb,
            comm_state=comm_state,
            start_after_current=not bool(timeline_step.link_trace.in_contact),
        )
        future_contact_feasible = bool(future_contact.get("future_contact_feasible", 0))
        _gate_enabled, _profile_name, gate_profile = load_gate_profile(self.simulator_cfg)
        allow_queued_without_current = bool(gate_profile.get("allow_queued_followup_without_current_contact", False))
        followup_mode = "same_slot" if same_slot else "queued"
        contact_opportunity_score = (
            1.0
            if timeline_step.link_trace.in_contact
            or (followup_mode == "queued" and allow_queued_without_current and future_contact_feasible)
            else 0.0
        )
        budget_counts = self._roi_context_budget_counts(timeline_step, followup_mode)
        decision = evaluate_proactive_followup_eligibility(
            state,
            {
                "variant": router.router_name,
                "router_name": router.router_name,
                "use_lyapunov_pricing": bool(getattr(router, "use_lyapunov_pricing", False)),
                "use_mode_controller": bool(getattr(router, "use_mode_controller", False)),
                "queue_mb": comm_state.queue_mb,
                "queue_max_mb": comm_state.queue_max_mb,
                "queue_pressure": queue_pressure,
                "queued_products": self._queued_product_keys(),
                "remaining_deadline_s": remaining_deadline_s,
                "deadline_s": candidate.deadline_s,
                "battery_soc": resource_state.battery_soc,
                "thermal_pressure": resource_state.thermal_pressure,
                "task_value": candidate.task_value,
                "high_value_flag": candidate.high_value_flag,
                "emergency_flag": candidate.emergency_flag,
                "value_level": candidate.value_level,
                "uncertainty": getattr(candidate.task_snapshot, "uncertainty", 0.0),
                "model_quality_factor": getattr(candidate.detection_result_snapshot, "estimated_quality", 1.0),
                "contact_available": timeline_step.link_trace.in_contact,
                "contact_remaining_window_s": timeline_step.link_trace.remaining_window_s,
                "contact_opportunity_score": contact_opportunity_score,
                "future_contact_feasible": future_contact_feasible,
                "followup_mode": followup_mode,
                **budget_counts,
                "payload_size_mb": payload_size_mb,
                "z_k_guided_followup_permission": bool(
                    getattr(router, "use_lyapunov_pricing", False)
                    and getattr(router, "use_mode_controller", False)
                    and self.last_guidance_bundle is not None
                ),
            },
            phase6_cfg,
        )
        self.last_proactive_followup_decision = decision
        if decision.eriv_breakdown is not None:
            self._record_value_density_trace(
                timeline_step=timeline_step,
                task_id=candidate.task_id,
                stage="carryover_queued_followup_gate",
                product_type=candidate.residual_product_type,
                residual_delta_u_i=decision.eriv_breakdown.residual_delta_u_I,
                detection_quality=decision.eriv_breakdown.model_quality_factor,
                timeliness_factor=decision.eriv_breakdown.timeliness_factor,
                payload_size_mb=decision.eriv_breakdown.payload_size_mb,
                value_density=decision.eriv_breakdown.value_density_per_mb,
                information_density=decision.eriv_breakdown.information_density,
                eriv_score=decision.eriv_score,
                followup_mode=followup_mode,
            )
        self._bump_carryover_log("candidate_re_evaluated_count")
        self._bump_carryover_log("residual_candidate_re_evaluation_count")
        self._bump_carryover_log("carryover_proactive_attempt_count")
        self._bump_carryover_log("residual_candidate_age_s", candidate.candidate_age_s)
        if same_slot:
            self._bump_carryover_log("same_slot_candidate_evaluated_count")
        if candidate.metadata_delivery_slot >= 0 and int(timeline_step.slot_index) > candidate.metadata_delivery_slot:
            self._bump_carryover_log("residual_candidate_re_evaluation_after_metadata_delivery_count")
        self.last_carryover_pool_log["carryover_eriv"] = decision.eriv_score
        self._bump_carryover_log("carryover_contact_block_count", int("no_contact_opportunity" in decision.blockers))
        self._bump_carryover_log("carryover_thermal_block_count", int("thermal_unsafe" in decision.blockers))
        self._bump_carryover_log("carryover_queue_block_count", int("queue_pressure_too_high" in decision.blockers))
        self._bump_carryover_log("carryover_deadline_block_count", int("no_useful_remaining_deadline" in decision.blockers))
        self._bump_carryover_log("carryover_energy_block_count", int("battery_unsafe" in decision.blockers))
        if same_slot:
            self._bump_carryover_log("same_slot_carryover_contact_block_count", int("no_contact_opportunity" in decision.blockers))
            self._bump_carryover_log(
                "same_slot_carryover_duplicate_block_count",
                int("duplicate_residual_already_queued" in decision.blockers),
            )
        self._record_carryover_event(
            "same_slot_candidate_re_evaluated" if same_slot else "candidate_re_evaluated",
            timeline_step,
            candidate,
            resource_state=resource_state,
            comm_state=comm_state,
            decision=decision,
            candidate_re_evaluated_flag=1,
        )
        if not decision.proactive_flag:
            self._record_same_slot_vs_queued_followup_trace(
                timeline_step=timeline_step,
                task_id=candidate.task_id,
                stage="carryover_queued_followup_gate",
                followup_mode="queued",
                action="blocked",
                reason_code=decision.reason_code,
                payload_size_mb=payload_size_mb,
                same_slot_delivery_allowed=False,
                queued_followup_allowed=False,
                future_contact_feasible=future_contact_feasible,
            )
            self._record_carryover_event(
                "same_slot_proactive_blocked" if same_slot else "proactive_blocked",
                timeline_step,
                candidate,
                resource_state=resource_state,
                comm_state=comm_state,
                decision=decision,
            )
            return None, candidate.task_snapshot, decision

        current_time_s = float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
        if state.first_alert_delivered_time_s is not None:
            state.time_from_metadata_to_followup_s = max(0.0, current_time_s - state.first_alert_delivered_time_s)
        residual_payload = generate_residual_payload(
            task=candidate.task_snapshot,
            detection_result=candidate.detection_result_snapshot,
            residual_product_type=decision.residual_product_type,
            slot_index=timeline_step.slot_index,
            parent_payload_plan=candidate.payload_plan,
            high_value_threshold=float(self.reward_cfg.get("high_value_threshold", 0.75)),
            payload_profiles_cfg=self.payload_profiles_cfg,
            delivered_products=set(state.delivered_products),
            remaining_deadline_s=remaining_deadline_s,
            continue_request_priority_bonus=float(phase6_cfg.get("proactive_queue_bonus", 0.0)),
            followup_trigger="proactive_followup",
        )
        residual_payload.metadata = {
            **residual_payload.metadata,
            "expected_residual_information_value": decision.eriv_score,
            "ERIV": decision.eriv_score,
            "z_k_guided_followup_permission": True,
            "proactive_before_passive_request": state.residual_requested_count == 0,
            "followup_source": "phase6_residual_candidate_carryover",
            "carryover_candidate_key": key,
            "carryover_candidate_age_s": candidate.candidate_age_s,
            "same_slot_carryover": bool(same_slot),
        }
        if not same_slot:
            guard_decision = self._followup_hard_safety_guard(
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                payload=residual_payload,
                lifecycle_state=state,
                task=candidate.task_snapshot,
                operation="carryover_residual_enqueue",
                require_same_slot_delivery=False,
                allow_no_contact_enqueue=True,
                predict_immediate_downlink=bool(timeline_step.link_trace.in_contact),
                allow_existing_pending=True,
            )
            if not guard_decision.allowed:
                self._record_same_slot_vs_queued_followup_trace(
                    timeline_step=timeline_step,
                    task_id=candidate.task_id,
                    stage="carryover_queued_followup_gate",
                    followup_mode="queued",
                    action="blocked_by_hard_safety_guard",
                    reason_code=guard_decision.reason_code,
                    payload_size_mb=payload_size_mb,
                    same_slot_delivery_allowed=False,
                    queued_followup_allowed=False,
                    future_contact_feasible=future_contact_feasible,
                )
                self._record_carryover_event(
                    "proactive_blocked_by_hard_safety_guard",
                    timeline_step,
                    candidate,
                    resource_state=resource_state,
                    comm_state=comm_state,
                    decision=decision,
                    removal_reason=guard_decision.reason_code,
                )
                return None, candidate.task_snapshot, decision
            if candidate.residual_product_type == "roi_context":
                self._increment_roi_context_budget(timeline_step, "queued")
            self._record_same_slot_vs_queued_followup_trace(
                timeline_step=timeline_step,
                task_id=candidate.task_id,
                stage="carryover_queued_followup_gate",
                followup_mode="queued",
                action="enqueue_candidate_payload",
                reason_code="queued_followup_payload_ready",
                payload_size_mb=payload_size_mb,
                same_slot_delivery_allowed=bool(timeline_step.link_trace.in_contact),
                queued_followup_allowed=True,
                future_contact_feasible=future_contact_feasible,
                enqueue_flag=1,
            )
        state.residual_utility_generated += residual_payload.incremental_utility_estimate
        self.last_carryover_payload_key = key
        self.last_carryover_feedback_task = candidate.task_snapshot
        if same_slot:
            self.last_same_slot_carryover_payload = residual_payload
        self.last_carryover_pool_log["carryover_eriv_selected"] = decision.eriv_score
        self._bump_carryover_log("carryover_before_passive_request_count", int(state.residual_requested_count == 0))
        return residual_payload, candidate.task_snapshot, decision

    def _maybe_build_carryover_payload(
        self,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        router: BaseRouter,
    ):
        self.last_carryover_payload_key = ""
        self.last_carryover_feedback_task = None
        if timeline_step.task is not None or not self._carryover_behavior_enabled(router):
            self.last_carryover_pool_log["residual_candidate_pool_size"] = len(self.residual_candidate_pool)
            self._set_carryover_log_max("residual_candidate_pool_size_max", len(self.residual_candidate_pool))
            return None, None

        carryover_cfg = dict(phase6_residual_candidate_carryover_config(self.simulator_cfg))
        phase6_cfg = phase6_proactive_config(self.simulator_cfg)
        max_candidates = max(1, int(carryover_cfg.get("max_candidates_per_slot", 2) or 2))
        evaluated = 0
        for key, candidate in sorted(self.residual_candidate_pool.items(), key=lambda item: (item[1].created_slot, item[0])):
            if evaluated >= max_candidates:
                break
            if int(timeline_step.slot_index) <= int(candidate.created_slot):
                continue
            invalid_reason = self._candidate_invalid_reason(candidate, timeline_step, carryover_cfg)
            if invalid_reason:
                self._remove_candidate(
                    key,
                    invalid_reason,
                    timeline_step,
                    resource_state=resource_state,
                    comm_state=comm_state,
                )
                continue
            evaluated += 1
            residual_payload, feedback_task, decision = self._evaluate_and_build_carryover_payload(
                key,
                candidate,
                timeline_step,
                resource_state,
                comm_state,
                router,
                phase6_cfg,
            )
            if residual_payload is None:
                continue
            return residual_payload, feedback_task

        self.last_carryover_pool_log["residual_candidate_pool_size"] = len(self.residual_candidate_pool)
        self._set_carryover_log_max("residual_candidate_pool_size_max", len(self.residual_candidate_pool))
        return None, None

    def _maybe_process_same_slot_carryover(
        self,
        timeline_step: TimelineStep,
        resource_state: ResourceState | None,
        comm_state: CommState | None,
        router: BaseRouter | None,
        downlink_result: DownlinkResult,
        candidate_key: str,
        *,
        downlink_throttled: bool = False,
        baseline_compute_energy_j: float = 0.0,
        baseline_comm_energy_j: float = 0.0,
        baseline_compute_heat_gain_scale: float = 1.0,
    ) -> DownlinkResult | None:
        if not candidate_key or candidate_key not in self.residual_candidate_pool:
            return None
        candidate = self.residual_candidate_pool[candidate_key]
        if not self._same_slot_carryover_enabled(router):
            return None
        same_slot_cfg = phase6_same_slot_carryover_config(self.simulator_cfg)
        max_eval = max(1, int(same_slot_cfg.get("max_same_slot_eval_per_slot", 1) or 1))
        if int(self.last_carryover_pool_log.get("same_slot_candidate_evaluated_count", 0)) >= max_eval:
            return None
        if int(candidate.created_slot) != int(timeline_step.slot_index):
            return None
        self._bump_carryover_log("same_slot_candidate_detected_count")
        reliable_capacity, remaining_capacity_mb, remaining_contact_s, capacity_reason = self._same_slot_remaining_capacity(
            timeline_step,
            downlink_result,
        )
        if downlink_throttled:
            reliable_capacity = False
            capacity_reason = "downlink_throttled"
            remaining_capacity_mb = 0.0
            remaining_contact_s = 0.0
        self._record_same_slot_carryover_trace(
            "same_slot_candidate_detected",
            timeline_step,
            candidate,
            remaining_capacity_mb=remaining_capacity_mb,
            remaining_contact_s=remaining_contact_s,
            reason_code=capacity_reason,
        )
        if not reliable_capacity:
            if capacity_reason in {"no_contact_opportunity", "no_remaining_contact_capacity", "contact_capacity_unavailable"}:
                self._bump_carryover_log("same_slot_carryover_contact_block_count")
            self._record_same_slot_vs_queued_followup_trace(
                timeline_step=timeline_step,
                task_id=candidate.task_id,
                stage="same_slot_carryover",
                followup_mode="same_slot",
                action="blocked",
                reason_code=capacity_reason or "contact_capacity_unavailable",
                payload_size_mb=0.0,
                same_slot_delivery_allowed=False,
                queued_followup_allowed=False,
                future_contact_feasible=False,
            )
            self._record_same_slot_carryover_trace(
                "same_slot_fallback_to_tplus1",
                timeline_step,
                candidate,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                reason_code=capacity_reason or "contact_capacity_unavailable",
            )
            return None

        same_slot_comm_state = CommState(
            queue_mb=float(downlink_result.queue_next_mb),
            queue_max_mb=(float(comm_state.queue_max_mb) if comm_state is not None else float(self.simulator_cfg.get("queue_max_mb", 1.0))),
            queue_item_count=int(downlink_result.queue_item_count),
            effective_opportunity=(0.0 if comm_state is None else float(comm_state.effective_opportunity)),
        )
        phase6_cfg = phase6_proactive_config(self.simulator_cfg)
        residual_payload, _feedback_task, decision = self._evaluate_and_build_carryover_payload(
            candidate_key,
            candidate,
            timeline_step,
            resource_state or ResourceState(1.0, 1.0, 0.0),
            same_slot_comm_state,
            router,
            phase6_cfg,
            same_slot=True,
        )
        self._record_same_slot_carryover_trace(
            "same_slot_candidate_re_evaluated",
            timeline_step,
            candidate,
            decision=decision,
            remaining_capacity_mb=remaining_capacity_mb,
            remaining_contact_s=remaining_contact_s,
            reason_code="" if decision is None else decision.reason_code,
        )
        if residual_payload is None or decision is None or not decision.proactive_flag:
            return None
        if float(residual_payload.payload_mb) > remaining_capacity_mb + float(self.simulator_cfg.get("epsilon", 1.0e-9)):
            self._record_same_slot_carryover_trace(
                "same_slot_fallback_to_tplus1",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                reason_code="residual_payload_exceeds_remaining_contact_capacity",
            )
            if not bool(same_slot_cfg.get("enable_queue_fallback_on_insufficient_capacity", False)):
                return None
            fallback_result = self._try_same_slot_ordinary_queue_fallback(
                timeline_step,
                candidate_key,
                candidate,
                residual_payload,
                decision,
                resource_state,
                same_slot_comm_state,
                remaining_capacity_mb,
                remaining_contact_s,
                baseline_compute_energy_j=baseline_compute_energy_j,
                baseline_comm_energy_j=baseline_comm_energy_j,
                baseline_compute_heat_gain_scale=baseline_compute_heat_gain_scale,
            )
            return fallback_result

        guard_decision = self._followup_hard_safety_guard(
            timeline_step=timeline_step,
            resource_state=resource_state or ResourceState(1.0, 1.0, 0.0),
            comm_state=same_slot_comm_state,
            payload=residual_payload,
            lifecycle_state=self.product_lifecycle.get(candidate.task_id),
            task=candidate.task_snapshot,
            operation="same_slot_carryover_delivery",
            require_same_slot_delivery=True,
            remaining_contact_capacity_mb=remaining_capacity_mb,
            downlink_throttled=bool(downlink_throttled),
            allow_no_contact_enqueue=False,
            predict_immediate_downlink=True,
            allow_existing_pending=True,
            baseline_compute_energy_j=baseline_compute_energy_j,
            baseline_comm_energy_j=baseline_comm_energy_j,
            baseline_compute_heat_gain_scale=baseline_compute_heat_gain_scale,
        )
        if not guard_decision.allowed:
            self._record_same_slot_vs_queued_followup_trace(
                timeline_step=timeline_step,
                task_id=candidate.task_id,
                stage="same_slot_carryover",
                followup_mode="same_slot",
                action="blocked_by_hard_safety_guard",
                reason_code=guard_decision.reason_code,
                payload_size_mb=float(residual_payload.payload_mb),
                same_slot_delivery_allowed=False,
                queued_followup_allowed=False,
                future_contact_feasible=False,
            )
            self._record_same_slot_carryover_trace(
                "same_slot_hard_safety_blocked",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                reason_code=guard_decision.reason_code,
            )
            return None

        same_slot_result = self.downlink_queue.process_slot(
            slot_index=timeline_step.slot_index,
            payload=residual_payload,
            link_trace=timeline_step.link_trace,
            capacity_override_mb=remaining_capacity_mb,
            contact_window_id=self._contact_window_id_for_slot(int(timeline_step.slot_index)),
        )
        self._bump_carryover_log("same_slot_carryover_enqueue_count", same_slot_result.proactive_followup_enqueue_count)
        self._bump_carryover_log("carryover_proactive_enqueue_count", same_slot_result.proactive_followup_enqueue_count)
        self._bump_carryover_log("same_slot_carryover_duplicate_block_count", same_slot_result.proactive_duplicate_suppression_count)
        if same_slot_result.proactive_followup_delivery_count:
            self._bump_carryover_log("same_slot_carryover_delivery_count", same_slot_result.proactive_followup_delivery_count)
        if same_slot_result.proactive_followup_enqueue_count:
            if candidate.residual_product_type == "roi_context":
                self._increment_roi_context_budget(timeline_step, "same_slot")
            self._bump_carryover_log("same_slot_opportunity_saved_from_tplus1_count")
            self._record_same_slot_vs_queued_followup_trace(
                timeline_step=timeline_step,
                task_id=candidate.task_id,
                stage="same_slot_carryover",
                followup_mode="same_slot",
                action="same_slot_processed",
                reason_code="same_slot_enqueued_via_downlink_queue",
                payload_size_mb=float(residual_payload.payload_mb),
                same_slot_delivery_allowed=True,
                queued_followup_allowed=False,
                future_contact_feasible=False,
                enqueue_flag=1,
                delivery_flag=int(same_slot_result.proactive_followup_delivery_count > 0),
            )
            self._record_carryover_event(
                "same_slot_proactive_enqueued",
                timeline_step,
                candidate,
                resource_state=resource_state,
                comm_state=same_slot_comm_state,
                decision=decision,
                proactive_enqueue_flag=1,
            )
            self._record_same_slot_carryover_trace(
                "same_slot_proactive_enqueued",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                enqueue_flag=1,
                delivery_flag=int(same_slot_result.proactive_followup_delivery_count > 0),
                reason_code="enqueued_via_downlink_queue",
            )
            self._remove_candidate(
                candidate_key,
                "same_slot_proactive_enqueued",
                timeline_step,
                resource_state=resource_state,
                comm_state=same_slot_comm_state,
            )
        elif same_slot_result.proactive_duplicate_suppression_count:
            self._record_same_slot_carryover_trace(
                "same_slot_duplicate_suppressed",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                reason_code="duplicate_residual_already_queued",
            )
        return same_slot_result

    def _try_same_slot_ordinary_queue_fallback(
        self,
        timeline_step: TimelineStep,
        candidate_key: str,
        candidate: ResidualCandidateState,
        residual_payload,
        decision,
        resource_state: ResourceState | None,
        same_slot_comm_state: CommState,
        remaining_capacity_mb: float,
        remaining_contact_s: float,
        *,
        baseline_compute_energy_j: float = 0.0,
        baseline_comm_energy_j: float = 0.0,
        baseline_compute_heat_gain_scale: float = 1.0,
    ) -> DownlinkResult | None:
        self._bump_carryover_log("ordinary_queue_fallback_attempt_count")
        if float(getattr(residual_payload, "payload_mb", 0.0)) <= 0.0:
            self._bump_carryover_log("zero_size_residual_payload_blocked_count")
            self._record_same_slot_carryover_trace(
                "ordinary_queue_fallback_blocked",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                reason_code="zero_size_residual_payload",
            )
            return None
        remaining_deadline_s = self._remaining_deadline_for_task(
            candidate.task_snapshot,
            timeline_step.slot_index,
            timeline_step.slot_seconds,
            candidate.deadline_s,
        )
        if remaining_deadline_s <= 0.0:
            self._bump_carryover_log("residual_fallback_blocked_by_expired_deadline_count")
            self._record_same_slot_carryover_trace(
                "ordinary_queue_fallback_blocked",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                reason_code="expired_deadline",
            )
            return None

        guard_decision = self._followup_hard_safety_guard(
            timeline_step=timeline_step,
            resource_state=resource_state or ResourceState(1.0, 1.0, 0.0),
            comm_state=same_slot_comm_state,
            payload=residual_payload,
            lifecycle_state=self.product_lifecycle.get(candidate.task_id),
            task=candidate.task_snapshot,
            operation="ordinary_queue_fallback_enqueue",
            require_same_slot_delivery=False,
            remaining_contact_capacity_mb=0.0,
            downlink_throttled=False,
            allow_no_contact_enqueue=True,
            predict_immediate_downlink=False,
            allow_existing_pending=True,
            baseline_compute_energy_j=baseline_compute_energy_j,
            baseline_comm_energy_j=baseline_comm_energy_j,
            baseline_compute_heat_gain_scale=baseline_compute_heat_gain_scale,
        )
        if not guard_decision.allowed:
            self._record_same_slot_carryover_trace(
                "ordinary_queue_fallback_hard_safety_blocked",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                reason_code=guard_decision.reason_code,
            )
            return None

        fallback_result = self.downlink_queue.process_slot(
            slot_index=timeline_step.slot_index,
            payload=residual_payload,
            link_trace=timeline_step.link_trace,
            capacity_override_mb=0.0,
            contact_window_id=self._contact_window_id_for_slot(int(timeline_step.slot_index)),
        )
        self._bump_carryover_log("same_slot_fallback_to_ordinary_queue_count")
        self._bump_carryover_log("same_slot_carryover_duplicate_block_count", fallback_result.proactive_duplicate_suppression_count)
        if fallback_result.proactive_duplicate_suppression_count:
            self._bump_carryover_log(
                "duplicate_residual_suppressed_after_fallback_count",
                fallback_result.proactive_duplicate_suppression_count,
            )
        if fallback_result.proactive_followup_enqueue_count:
            self._bump_carryover_log("ordinary_queue_fallback_success_count", fallback_result.proactive_followup_enqueue_count)
            self._bump_carryover_log("carryover_proactive_enqueue_count", fallback_result.proactive_followup_enqueue_count)
            self._record_carryover_event(
                "ordinary_queue_fallback_enqueued",
                timeline_step,
                candidate,
                resource_state=resource_state,
                comm_state=same_slot_comm_state,
                decision=decision,
                proactive_enqueue_flag=1,
            )
            self._record_same_slot_carryover_trace(
                "ordinary_queue_fallback_enqueued",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                enqueue_flag=1,
                reason_code="insufficient_same_slot_capacity_but_ordinary_queue_enabled",
            )
            self._remove_candidate(
                candidate_key,
                "ordinary_queue_fallback_enqueued",
                timeline_step,
                resource_state=resource_state,
                comm_state=same_slot_comm_state,
            )
        elif fallback_result.proactive_duplicate_suppression_count:
            self._record_carryover_event(
                "ordinary_queue_fallback_duplicate_suppressed",
                timeline_step,
                candidate,
                resource_state=resource_state,
                comm_state=same_slot_comm_state,
                decision=decision,
                duplicate_suppression_flag=1,
            )
            self._record_same_slot_carryover_trace(
                "ordinary_queue_fallback_duplicate_suppressed",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                reason_code="duplicate_residual_already_queued",
            )
        else:
            self._record_same_slot_carryover_trace(
                "ordinary_queue_fallback_blocked",
                timeline_step,
                candidate,
                decision=decision,
                remaining_capacity_mb=remaining_capacity_mb,
                remaining_contact_s=remaining_contact_s,
                reason_code="queue_admission_rejected",
            )
        if fallback_result.proactive_followup_dropped_count or fallback_result.residual_dropped_count:
            self._bump_carryover_log(
                "residual_fallback_dropped_by_queue_overflow_count",
                max(fallback_result.proactive_followup_dropped_count, fallback_result.residual_dropped_count),
            )
        return fallback_result

    @staticmethod
    def _merge_downlink_results(primary: DownlinkResult, extra: DownlinkResult) -> DownlinkResult:
        primary.S_down_total_mb += extra.S_down_total_mb
        primary.S_down_current_task_mb += extra.S_down_current_task_mb
        primary.dropped_current_task_mb += extra.dropped_current_task_mb
        primary.queue_next_mb = extra.queue_next_mb
        primary.delta_queue_mb += extra.delta_queue_mb
        primary.queue_latency_s = extra.queue_latency_s
        primary.queue_item_count = extra.queue_item_count
        primary.delivered_item_count += extra.delivered_item_count
        primary.expired_item_count += extra.expired_item_count
        primary.dropped_item_count += extra.dropped_item_count
        primary.metadata_delivered_count += extra.metadata_delivered_count
        primary.high_value_metadata_delivered_count += extra.high_value_metadata_delivered_count
        primary.delivered_high_value_payload_count += extra.delivered_high_value_payload_count
        primary.delivered_waiting_time_s += extra.delivered_waiting_time_s
        primary.transmitted_task_ids.extend(extra.transmitted_task_ids)
        primary.delivered_task_ids.extend(extra.delivered_task_ids)
        primary.expired_task_ids.extend(extra.expired_task_ids)
        primary.dropped_task_ids.extend(extra.dropped_task_ids)
        primary.delivered_task_event_times_s.extend(extra.delivered_task_event_times_s)
        primary.expired_task_event_times_s.extend(extra.expired_task_event_times_s)
        primary.dropped_task_event_times_s.extend(extra.dropped_task_event_times_s)
        primary.delivered_payload_ids.extend(extra.delivered_payload_ids)
        primary.delivered_product_types.extend(extra.delivered_product_types)
        primary.expired_payload_ids.extend(extra.expired_payload_ids)
        primary.expired_product_types.extend(extra.expired_product_types)
        primary.dropped_payload_ids.extend(extra.dropped_payload_ids)
        primary.dropped_product_types.extend(extra.dropped_product_types)
        primary.delivered_incremental_utilities.extend(extra.delivered_incremental_utilities)
        primary.duplicate_product_attempt_count += extra.duplicate_product_attempt_count
        primary.zero_increment_product_count += extra.zero_increment_product_count
        primary.residual_enqueue_count += extra.residual_enqueue_count
        primary.residual_delivery_count += extra.residual_delivery_count
        primary.residual_expired_count += extra.residual_expired_count
        primary.residual_dropped_count += extra.residual_dropped_count
        primary.duplicate_residual_suppression_count += extra.duplicate_residual_suppression_count
        primary.followup_payload_mb += extra.followup_payload_mb
        primary.residual_delivered_incremental_utilities.extend(extra.residual_delivered_incremental_utilities)
        primary.residual_queue_wait_times_s.extend(extra.residual_queue_wait_times_s)
        primary.delivered_followup_triggers.extend(extra.delivered_followup_triggers)
        primary.proactive_followup_enqueue_count += extra.proactive_followup_enqueue_count
        primary.proactive_followup_delivery_count += extra.proactive_followup_delivery_count
        primary.proactive_followup_expired_count += extra.proactive_followup_expired_count
        primary.proactive_followup_dropped_count += extra.proactive_followup_dropped_count
        primary.proactive_duplicate_suppression_count += extra.proactive_duplicate_suppression_count
        primary.proactive_followup_payload_mb += extra.proactive_followup_payload_mb
        primary.proactive_delivered_incremental_utilities.extend(extra.proactive_delivered_incremental_utilities)
        primary.delivered_payload_metadata.extend(extra.delivered_payload_metadata)
        primary.expired_payload_metadata.extend(getattr(extra, "expired_payload_metadata", []))
        primary.dropped_payload_metadata.extend(getattr(extra, "dropped_payload_metadata", []))
        return primary

    def _refresh_feedback_downlink_fields(self, feedback: ObservedFeedback, downlink_result: DownlinkResult) -> None:
        feedback.S_down_total_mb = downlink_result.S_down_total_mb
        feedback.S_down_current_task_mb = downlink_result.S_down_current_task_mb
        feedback.queue_next_mb = downlink_result.queue_next_mb
        feedback.delta_queue_mb = downlink_result.delta_queue_mb
        feedback.L_queue = downlink_result.queue_latency_s
        feedback.dropped_current_task_mb = downlink_result.dropped_current_task_mb
        feedback.queue_item_count = downlink_result.queue_item_count
        feedback.delivered_items = downlink_result.delivered_item_count
        feedback.expired_items = downlink_result.expired_item_count
        feedback.dropped_items = downlink_result.dropped_item_count
        feedback.metadata_delivered_items = downlink_result.metadata_delivered_count
        feedback.high_value_metadata_delivered_items = downlink_result.high_value_metadata_delivered_count
        feedback.delivered_high_value_payloads = downlink_result.delivered_high_value_payload_count
        feedback.average_waiting_time_s = downlink_result.delivered_waiting_time_s / max(downlink_result.delivered_item_count, 1)
        feedback.transmitted_task_ids = self._format_task_ids(downlink_result.transmitted_task_ids)
        feedback.delivered_task_ids = self._format_task_ids(downlink_result.delivered_task_ids)
        feedback.expired_task_ids = self._format_task_ids(downlink_result.expired_task_ids)
        feedback.dropped_task_ids = self._format_task_ids(downlink_result.dropped_task_ids)
        feedback.delivered_payload_ids = self._format_task_ids(downlink_result.delivered_payload_ids)
        feedback.delivered_product_types = self._format_task_ids(downlink_result.delivered_product_types)
        feedback.expired_payload_ids = self._format_task_ids(downlink_result.expired_payload_ids)
        feedback.expired_product_types = self._format_task_ids(downlink_result.expired_product_types)
        feedback.dropped_payload_ids = self._format_task_ids(downlink_result.dropped_payload_ids)
        feedback.dropped_product_types = self._format_task_ids(downlink_result.dropped_product_types)
        feedback.delivered_task_event_times_s = self._format_event_times(downlink_result.delivered_task_event_times_s)
        feedback.expired_task_event_times_s = self._format_event_times(downlink_result.expired_task_event_times_s)
        feedback.dropped_task_event_times_s = self._format_event_times(downlink_result.dropped_task_event_times_s)

    def _apply_same_slot_delivery_effects(
        self,
        timeline_step: TimelineStep,
        feedback: ObservedFeedback,
        same_slot_result: DownlinkResult,
    ) -> dict[str, float]:
        totals = {
            "delivered_only_incremental_utility": 0.0,
            "high_value_delivered_incremental_utility": 0.0,
            "alert_delivered_utility": 0.0,
            "verification_delivered_utility": 0.0,
            "context_evidence_delivered_utility": 0.0,
            "proactive_followup_incremental_utility": 0.0,
            "carryover_proactive_incremental_utility": 0.0,
            "carryover_proactive_delivery_count": 0.0,
            "residual_utility_delivered": 0.0,
        }
        for index, task_id in enumerate(same_slot_result.delivered_task_ids):
            product_type = (
                same_slot_result.delivered_product_types[index]
                if index < len(same_slot_result.delivered_product_types)
                else ""
            )
            delivered_metadata = (
                same_slot_result.delivered_payload_metadata[index]
                if index < len(getattr(same_slot_result, "delivered_payload_metadata", []))
                else {}
            )
            if delivered_metadata.get("followup_source") != "phase6_residual_candidate_carryover":
                continue
            candidate_key = str(delivered_metadata.get("carryover_candidate_key", ""))
            candidate = self.residual_candidate_history.get(candidate_key) or self.residual_candidate_pool.get(candidate_key)
            state = self.product_lifecycle.get(task_id)
            if state is None or candidate is None:
                continue
            utility_task = candidate.task_snapshot
            event_time_s = (
                same_slot_result.delivered_task_event_times_s[index]
                if index < len(same_slot_result.delivered_task_event_times_s)
                else float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
            )
            total_latency_s = max(0.0, event_time_s - float(utility_task.arrival_slot) * float(timeline_step.slot_seconds))
            timeliness = self._timeliness_factor(total_latency_s, utility_task.deadline_s)
            delta_u_i = delta_information_utility(product_type, state.delivered_products, utility_task.product_requirement)
            detection_quality = float(getattr(candidate.detection_result_snapshot, "estimated_quality", feedback.detection_quality))
            increment = compute_gross_incremental_utility(
                task_value=task_effective_value(utility_task),
                detection_quality=detection_quality,
                delta_u_i=delta_u_i,
                timeliness=timeliness,
                rho_res=1.0,
            )
            totals["delivered_only_incremental_utility"] += increment
            if utility_task.high_value_flag or task_effective_value(utility_task) >= float(self.reward_cfg.get("high_value_threshold", 0.75)):
                totals["high_value_delivered_incremental_utility"] += increment
            if utility_task.product_requirement == "alert":
                totals["alert_delivered_utility"] += increment
            elif utility_task.product_requirement == "verification":
                totals["verification_delivered_utility"] += increment
            elif utility_task.product_requirement in {"context", "evidence"}:
                totals["context_evidence_delivered_utility"] += increment
            followup_trigger = (
                same_slot_result.delivered_followup_triggers[index]
                if index < len(same_slot_result.delivered_followup_triggers)
                else str(delivered_metadata.get("followup_trigger", ""))
            )
            if product_type in {"roi", "roi_context"} and followup_trigger == "proactive_followup":
                totals["proactive_followup_incremental_utility"] += increment
                totals["carryover_proactive_incremental_utility"] += increment
                totals["carryover_proactive_delivery_count"] += 1
                totals["residual_utility_delivered"] += increment
                state.proactive_followup_delivery_count += 1
                state.residual_utility_saved_by_proactive += increment
                state.carryover_proactive_delivery_count += 1
                state.carryover_proactive_incremental_utility += increment
                self._record_carryover_event(
                    "same_slot_proactive_delivered",
                    timeline_step,
                    candidate,
                    proactive_delivery_flag=1,
                )
                self._record_same_slot_carryover_trace(
                    "same_slot_proactive_delivered",
                    timeline_step,
                    candidate,
                    decision=self.last_proactive_followup_decision,
                    enqueue_flag=1,
                    delivery_flag=1,
                    reason_code="delivered_via_downlink_queue",
                )
            state.pending_products.discard(product_type)
            state.delivered_products.add(product_type)
            if product_type in {"roi", "roi_context"}:
                state.residual_delivered_count += 1
                state.residual_utility_delivered += increment
                if index < len(same_slot_result.residual_queue_wait_times_s):
                    state.residual_queue_wait_s = same_slot_result.residual_queue_wait_times_s[index]
            state.residual_utility_remaining = residual_information_utility(
                state.delivered_products,
                state.product_requirement,
            )
            if state.residual_utility_remaining <= 1.0e-12 or self._integrated_branch_closure_satisfied(state):
                state.task_closed_flag = True
        return totals

    @staticmethod
    def _timeliness_factor(total_latency_s: float, deadline_s: float) -> float:
        if deadline_s <= 1.0e-9:
            return 0.0
        return max(0.0, min(1.0, 1.0 - total_latency_s / deadline_s))

    def _predicted_resource_cost(
        self,
        predicted,
        task,
        guidance_bundle: dict[str, object] | None,
    ) -> float:
        epsilon = float(self.simulator_cfg["epsilon"])
        total_energy = float(predicted.predicted_E_comp + predicted.predicted_E_comm)
        latency = float(predicted.predicted_L_inf + predicted.predicted_L_queue + predicted.predicted_L_comm)
        lambda_E = (
            float(guidance_bundle.get("lambda_E", 0.0))
            if guidance_bundle
            else float(self.reward_cfg.get("lambda_E_base", self.reward_cfg.get("energy_penalty", 0.0)))
        )
        lambda_L = (
            float(guidance_bundle.get("lambda_L", 0.0))
            if guidance_bundle
            else float(self.reward_cfg.get("lambda_L", self.reward_cfg.get("latency_penalty", 0.0)))
        )
        lambda_Q = (
            float(guidance_bundle.get("lambda_Q", 0.0))
            if guidance_bundle
            else float(self.reward_cfg.get("lambda_Q", self.reward_cfg.get("queue_penalty", 0.0)))
        )
        lambda_S = float(guidance_bundle.get("lambda_S", 0.0)) if guidance_bundle else 0.0
        return (
            lambda_E * total_energy / max(float(self.simulator_cfg["E_ref"]), epsilon)
            + lambda_L * latency / max(float(task.deadline_s), epsilon)
            + lambda_Q * float(predicted.predicted_Q_next_mb) / max(float(self.simulator_cfg["queue_max_mb"]), epsilon)
            + lambda_S * float(predicted.predicted_S_gen_mb) / max(float(self.simulator_cfg["queue_max_mb"]), epsilon)
        )

    def _build_candidate_action_log(
        self,
        timeline_step: TimelineStep,
        context,
        actions,
        candidate_actions,
        predicted_map,
        reasons_by_action: dict[str, list[str]],
        raw_action,
        executed_action,
        selected_action_score: float,
        guidance_bundle: dict[str, object] | None,
        router: BaseRouter,
    ) -> list[dict[str, object]]:
        task = timeline_step.task
        if task is None:
            return []
        candidate_names = {action.action_name for action in candidate_actions}
        score_components_by_action = getattr(router, "last_all_score_components", {}) or {}
        guidance_bundle = guidance_bundle or {}
        rows: list[dict[str, object]] = []
        for action in actions:
            predicted = predicted_map[action.action_name]
            score_components = dict(score_components_by_action.get(action.action_name, {}))
            total_latency = float(predicted.predicted_L_inf + predicted.predicted_L_queue + predicted.predicted_L_comm)
            reasons = reasons_by_action.get(action.action_name, [])
            selected_flag = int(action.action_name == raw_action.action_name)
            candidate_row = {
                "slot": timeline_step.slot_index,
                "time_s": float(timeline_step.slot_index) * float(timeline_step.slot_seconds),
                "task_id": task.task_id,
                "scene_id": task.scene_id,
                "task_value": task_effective_value(task),
                "value_level": str(getattr(task, "value_level", "") or (task.metadata or {}).get("value_level", "")),
                "product_requirement": task.product_requirement,
                "V_t": context.V_t,
                "high_value_flag": int(bool(task.high_value_flag) or context.V_t >= float(self.reward_cfg.get("high_value_threshold", 0.75))),
                "emergency_flag": int(bool(task.emergency_flag)),
                "deadline_s": task.deadline_s,
                "remaining_deadline_s": task.deadline_s,
                "candidate_action": action.action_name,
                "candidate_model": action.model_name,
                "candidate_payload": action.payload_policy,
                "considered_by_router": int(action.action_name in candidate_names),
                "safety_status": "safe" if action.action_name in candidate_names else "rejected",
                "safety_rejection_reason": ";".join(reasons) if reasons else "",
                "predicted_quality": predicted.predicted_detection_quality,
                "predicted_information_factor": predicted.effective_info_gain,
                "predicted_raw_info_gain": predicted.raw_info_gain,
                "predicted_delivery_factor": predicted.predicted_delivery_factor,
                "predicted_timeliness_factor": self._timeliness_factor(total_latency, task.deadline_s),
                "predicted_reward": predicted.predicted_reward,
                "predicted_resource_cost": self._predicted_resource_cost(predicted, task, guidance_bundle),
                "predicted_energy_j": float(predicted.predicted_E_comp + predicted.predicted_E_comm),
                "predicted_latency_s": total_latency,
                "predicted_payload_mb": predicted.predicted_S_gen_mb,
                "predicted_queue_next_mb": predicted.predicted_Q_next_mb,
                "lambda_E": float(guidance_bundle.get("lambda_E", 0.0)),
                "lambda_L": float(guidance_bundle.get("lambda_L", 0.0)),
                "lambda_Q": float(guidance_bundle.get("lambda_Q", 0.0)),
                "lambda_S": float(guidance_bundle.get("lambda_S", 0.0)),
                "base_lyapunov_price_contribution": "",
                "mode_guidance_contribution": "",
                "ucb_bonus": score_components.get("cost_aware_ucb_bonus", score_components.get("raw_ucb_bonus", "")),
                "raw_ucb_bonus": score_components.get("raw_ucb_bonus", ""),
                "risk_discount": score_components.get("risk_discount", ""),
                "guided_score": score_components.get("predicted_reward_prior", 0.20 * predicted.predicted_reward),
                "final_candidate_score": score_components.get(
                    "final_action_score",
                    selected_action_score if selected_flag else predicted.predicted_reward,
                ),
                "selected_flag": selected_flag,
                "executed_flag": int(action.action_name == executed_action.action_name),
                "payload_delivery_risk_reason": predicted.payload_delivery_risk_reason,
                "ideal_product_requirement": score_components.get("ideal_product_requirement", task.product_requirement),
                "product_requirement_aware_enabled": score_components.get("product_requirement_aware_enabled", 0),
                "requirement_fit_score": score_components.get("requirement_fit_score", ""),
                "model_fit_score": score_components.get("model_fit_score", ""),
                "deadline_feasibility_score": score_components.get("deadline_feasibility_score", ""),
                "staged_delivery_feasibility_score": score_components.get("staged_delivery_feasibility_score", ""),
                "product_requirement_aware_bonus": score_components.get("product_requirement_aware_bonus", ""),
                "product_requirement_aware_score_shadow": score_components.get("product_requirement_aware_score_shadow", ""),
            }
            if bool(score_components.get("log_utility_guided_score_fields", False)):
                candidate_row.update(
                    {
                        "scoring_mode": score_components.get("scoring_mode", ""),
                        "utility_prior_raw": score_components.get("utility_prior_raw", ""),
                        "utility_prior_norm": score_components.get("utility_prior_norm", ""),
                        "learned_correction_raw": score_components.get("learned_correction_raw", ""),
                        "learned_correction_norm": score_components.get("learned_correction_norm", ""),
                        "safe_exploration_raw": score_components.get("safe_exploration_raw", ""),
                        "safe_exploration_norm": score_components.get("safe_exploration_norm", ""),
                        "utility_prior_weight": score_components.get("utility_prior_weight", ""),
                        "learned_correction_weight": score_components.get("learned_correction_weight", ""),
                        "exploration_weight": score_components.get("exploration_weight", ""),
                        "score_component_normalization": score_components.get("score_component_normalization", ""),
                        "legacy_score_shadow": score_components.get("legacy_score_shadow", ""),
                        "utility_guided_score_shadow": score_components.get("utility_guided_score_shadow", ""),
                        "selected_by_legacy_shadow": score_components.get("selected_by_legacy_shadow", ""),
                        "selected_by_utility_guided_shadow": score_components.get("selected_by_utility_guided_shadow", ""),
                        "action_flip_vs_legacy_shadow": score_components.get("action_flip_vs_legacy_shadow", ""),
                        "z_lambda_E": score_components.get("z_lambda_E", ""),
                        "z_lambda_L": score_components.get("z_lambda_L", ""),
                        "z_lambda_Q": score_components.get("z_lambda_Q", ""),
                        "z_lambda_S": score_components.get("z_lambda_S", ""),
                        "z_mode_name": score_components.get("z_mode_name", ""),
                        "z_exploration_scale": score_components.get("z_exploration_scale", ""),
                        "z_model_complexity_bias": score_components.get("z_model_complexity_bias", ""),
                        "z_payload_size_bias": score_components.get("z_payload_size_bias", ""),
                        "z_high_value_bonus_scale": score_components.get("z_high_value_bonus_scale", ""),
                        "z_safety_margin_scale": score_components.get("z_safety_margin_scale", ""),
                        "utility_prior_uses_z_pricing": score_components.get("utility_prior_uses_z_pricing", ""),
                        "learned_correction_uses_z": score_components.get("learned_correction_uses_z", ""),
                        "safe_exploration_uses_z": score_components.get("safe_exploration_uses_z", ""),
                    }
                )
            rows.append(candidate_row)
        return rows

    def _estimate_compute_energy_j(self, action, detection_result) -> float:
        if action is None or detection_result is None:
            return 0.0
        profile = self.model_registry.get_profile(action.model_name)
        if action.model_name == "delay_store":
            return 0.0
        return compute_inference_energy_j(profile.estimated_power_w, detection_result.inference_latency_s)

    def _compute_downlink_comm_energy_j(self, timeline_step: TimelineStep, downlink_result) -> float:
        rate_mb_s = estimate_rate_mb_per_s(
            timeline_step.link_trace,
            float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
        )
        epsilon = float(self.simulator_cfg["epsilon"])
        tx_time_total = 0.0 if rate_mb_s <= epsilon else downlink_result.S_down_total_mb / rate_mb_s
        return compute_comm_energy_j(
            comm_power_w=float(self.simulator_cfg.get("tx_power_w", self.simulator_cfg["comm_power_w"])),
            tx_time_s=tx_time_total,
            link_quality=timeline_step.link_trace.link_quality,
            alpha_gamma=float(self.simulator_cfg["comm_alpha_gamma"]),
        )

    def _estimate_next_thermal(
        self,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        compute_energy_j: float,
        comm_energy_j: float,
        compute_heat_gain_scale: float = 1.0,
    ) -> float:
        return update_thermal_pressure(
            thermal_pressure=resource_state.thermal_pressure,
            compute_energy_j=compute_energy_j,
            comm_energy_j=comm_energy_j,
            cooling=float(self.simulator_cfg.get("cooling_per_slot", 0.0)),
            compute_heat_gain_scale=compute_heat_gain_scale,
            slot_seconds=float(timeline_step.slot_seconds),
            reference_slot_seconds=float(self.simulator_cfg.get("thermal_reference_slot_seconds", 60.0)),
            temperature_reference_c=timeline_step.resource_trace.temperature_reference_c,
            is_sunlit=timeline_step.resource_trace.is_sunlit,
            passive_cooling=float(self.simulator_cfg.get("thermal_passive_cooling", 0.12)),
            sunlight_gain=float(self.simulator_cfg.get("thermal_sunlight_gain", 0.015)),
            eclipse_cooling_bonus=float(self.simulator_cfg.get("thermal_eclipse_cooling_bonus", 0.03)),
            reference_min_c=float(self.simulator_cfg.get("thermal_reference_min_c", -10.0)),
            reference_max_c=float(self.simulator_cfg.get("thermal_reference_max_c", 30.0)),
        )

    def _evaluate_downlink_throttle(
        self,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        planned_downlink_mb: float,
        planned_compute_energy_j: float,
        planned_comm_energy_j: float,
        planned_compute_heat_gain_scale: float = 1.0,
    ) -> tuple[int, str]:
        throttle_cfg = self.simulator_cfg.get("downlink_safety_throttle", {})
        if not isinstance(throttle_cfg, dict) or not bool(throttle_cfg.get("enabled", False)):
            return 0, "none"
        if str(throttle_cfg.get("throttle_mode", "pause_when_hot")) != "pause_when_hot":
            return 0, "none"
        if planned_downlink_mb <= float(self.simulator_cfg["epsilon"]):
            return 0, "none"

        thermal_guard = float(throttle_cfg.get("thermal_guard_threshold", self.simulator_cfg["thermal_max"]))
        thermal_hard = float(throttle_cfg.get("thermal_hard_threshold", self.simulator_cfg["thermal_max"]))
        planned_next_thermal = self._estimate_next_thermal(
            timeline_step=timeline_step,
            resource_state=resource_state,
            compute_energy_j=planned_compute_energy_j,
            comm_energy_j=planned_comm_energy_j,
            compute_heat_gain_scale=planned_compute_heat_gain_scale,
        )
        current_thermal = resource_state.thermal_pressure
        epsilon = float(self.simulator_cfg["epsilon"])
        if current_thermal >= thermal_hard - epsilon or planned_next_thermal > thermal_hard + epsilon:
            return 1, "thermal_guard"
        if current_thermal >= thermal_guard and planned_next_thermal > thermal_hard + epsilon:
            return 1, "thermal_guard"
        return 0, "none"

    def _thermal_headroom_capacity_decision(
        self,
        *,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        nominal_capacity_mb: float,
        planned_compute_energy_j: float,
        planned_compute_heat_gain_scale: float,
    ):
        cfg = thermal_headroom_config(self.simulator_cfg)
        current = float(resource_state.thermal_pressure)
        previous_recovery = bool(self._thermal_headroom_recovery_mode)
        event = ""
        if cfg["hysteresis_enabled"]:
            if previous_recovery and current < cfg["guard"]:
                self._thermal_headroom_recovery_mode = False
                event = "EXIT_RECOVERY"
            elif current >= cfg["hard"] and not previous_recovery:
                self._thermal_headroom_recovery_mode = True
                event = "ENTER_HARD_PAUSE"
        force_pause = bool(cfg["hysteresis_enabled"] and self._thermal_headroom_recovery_mode)
        rate_mb_s = estimate_rate_mb_per_s(
            timeline_step.link_trace,
            float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
        )
        decision = maximum_safe_downlink_capacity(
            current_thermal=current,
            nominal_capacity_mb=nominal_capacity_mb,
            rate_mb_s=rate_mb_s,
            link_quality=float(timeline_step.link_trace.link_quality),
            compute_energy_j=planned_compute_energy_j,
            compute_heat_gain_scale=planned_compute_heat_gain_scale,
            slot_seconds=float(timeline_step.slot_seconds),
            temperature_reference_c=float(timeline_step.resource_trace.temperature_reference_c),
            is_sunlit=bool(timeline_step.resource_trace.is_sunlit),
            simulator_cfg=self.simulator_cfg,
            force_hard_pause=force_pause,
        )
        if cfg["hysteresis_enabled"] and not self._thermal_headroom_recovery_mode and decision.zero_capacity_reason == "COMPUTE_ALONE_EXHAUSTS_HEADROOM":
            self._thermal_headroom_recovery_mode = True
            event = "ENTER_HARD_PAUSE"
        self.last_thermal_hysteresis_trace.append(
            {
                "slot": int(timeline_step.slot_index),
                "time_s": float(timeline_step.slot_index) * float(timeline_step.slot_seconds),
                "current_thermal": current,
                "guard_threshold": cfg["guard"],
                "hard_threshold": cfg["hard"],
                "recovery_mode_before": int(previous_recovery),
                "recovery_mode_after": int(self._thermal_headroom_recovery_mode),
                "event": event,
            }
        )
        return decision

    @staticmethod
    def _positive_normalized_square(value: float, guard: float, hard: float) -> float:
        denominator = max(abs(hard - guard), 1.0e-9)
        return max(0.0, (value - guard) / denominator) ** 2

    def _compute_long_term_risk_breakdown(
        self,
        next_battery: float,
        next_thermal: float,
        queue_next_mb: float,
        actual_downlink_mb: float,
    ) -> dict[str, float | int]:
        risk_cfg = self.simulator_cfg.get("long_term_risk_penalty", {})
        if not isinstance(risk_cfg, dict) or not bool(risk_cfg.get("enabled", False)):
            return {
                "battery_next_soc": next_battery,
                "thermal_next": next_thermal,
                "thermal_guard_exceeded": 0,
                "battery_guard_exceeded": 0,
                "thermal_guard_penalty": 0.0,
                "battery_reserve_penalty": 0.0,
                "downlink_burden_penalty": 0.0,
                "queue_pressure_risk_penalty": 0.0,
                "long_term_risk_penalty": 0.0,
            }

        thermal_guard = float(risk_cfg.get("thermal_guard_threshold", 0.85))
        thermal_hard = float(risk_cfg.get("thermal_hard_threshold", self.simulator_cfg["thermal_max"]))
        battery_guard = float(risk_cfg.get("battery_guard_threshold", 0.80))
        battery_min = float(self.simulator_cfg["battery_min"])
        queue_max = float(self.simulator_cfg["queue_max_mb"])

        thermal_risk = self._positive_normalized_square(next_thermal, thermal_guard, thermal_hard)
        battery_denominator = max(battery_guard - battery_min, 1.0e-9)
        battery_risk = max(0.0, (battery_guard - next_battery) / battery_denominator) ** 2
        queue_pressure = queue_next_mb / max(queue_max, 1.0e-9)

        thermal_guard_penalty = float(risk_cfg.get("thermal_risk_weight", 0.0)) * thermal_risk
        battery_reserve_penalty = float(risk_cfg.get("battery_reserve_weight", 0.0)) * battery_risk
        downlink_burden_penalty = float(risk_cfg.get("downlink_burden_weight_per_mb", 0.0)) * actual_downlink_mb
        queue_pressure_risk_penalty = float(risk_cfg.get("queue_risk_weight", 0.0)) * queue_pressure**2
        long_term_risk_penalty = (
            thermal_guard_penalty
            + battery_reserve_penalty
            + downlink_burden_penalty
            + queue_pressure_risk_penalty
        )
        return {
            "battery_next_soc": next_battery,
            "thermal_next": next_thermal,
            "thermal_guard_exceeded": int(next_thermal > thermal_guard),
            "battery_guard_exceeded": int(next_battery < battery_guard),
            "thermal_guard_penalty": thermal_guard_penalty,
            "battery_reserve_penalty": battery_reserve_penalty,
            "downlink_burden_penalty": downlink_burden_penalty,
            "queue_pressure_risk_penalty": queue_pressure_risk_penalty,
            "long_term_risk_penalty": long_term_risk_penalty,
        }

    def _compute_actual_feedback(
        self,
        timeline_step: TimelineStep,
        action,
        detection_result,
        payload,
        downlink_result,
        feedback_task=None,
    ) -> tuple[ObservedFeedback, float, float, float]:
        has_task = timeline_step.task is not None and action is not None and detection_result is not None
        task_for_feedback = feedback_task or timeline_step.task
        payload_encoding_latency_s = 0.0
        payload_encoding_energy_j = 0.0
        if has_task:
            E_comp = self._estimate_compute_energy_j(action, detection_result)
            if payload is not None:
                payload_encoding_latency_s, payload_encoding_energy_j = estimate_payload_encoding_cost(
                    payload.payload_policy,
                    payload.payload_mb,
                    self.payload_profiles_cfg,
                )
            E_comp += payload_encoding_energy_j
            L_inf = detection_result.inference_latency_s + payload_encoding_latency_s
            miss_signal = 1.0 if detection_result.miss_flag else 0.0
            false_alarm_signal = 1.0 if detection_result.false_alarm_flag else 0.0
            detection_quality = detection_result.estimated_quality
        else:
            E_comp = 0.0
            L_inf = 0.0
            miss_signal = 0.0
            false_alarm_signal = 0.0
            detection_quality = 0.0

        rate_mb_s = estimate_rate_mb_per_s(
            timeline_step.link_trace,
            float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
        )
        epsilon = float(self.simulator_cfg["epsilon"])
        L_comm = 0.0 if rate_mb_s <= epsilon else downlink_result.S_down_current_task_mb / rate_mb_s
        E_comm = self._compute_downlink_comm_energy_j(timeline_step, downlink_result)
        L_queue = downlink_result.queue_latency_s
        total_latency = L_inf + L_queue + L_comm
        delivery_factor = compute_delivery_factor(
            downlink_result.current_task_delivery_status,
            delivery_beta=float(self.simulator_cfg["delivery_beta"]),
            deferred_credit=float(self.reward_cfg["deferred_delivery_credit"]),
        )
        feedback = ObservedFeedback(
            S_gen_current_mb=downlink_result.generated_payload_current_mb,
            S_down_total_mb=downlink_result.S_down_total_mb,
            S_down_current_task_mb=downlink_result.S_down_current_task_mb,
            queue_next_mb=downlink_result.queue_next_mb,
            delta_queue_mb=downlink_result.delta_queue_mb,
            E_comp=E_comp,
            E_comm=E_comm,
            L_inf=L_inf,
            L_queue=L_queue,
            L_comm=L_comm,
            delivery_status=downlink_result.current_task_delivery_status,
            delivery_factor=delivery_factor,
            safety_violation=0,
            total_latency_s=total_latency,
            dropped_current_task_mb=downlink_result.dropped_current_task_mb,
            queue_item_count=downlink_result.queue_item_count,
            delivered_items=downlink_result.delivered_item_count,
            expired_items=downlink_result.expired_item_count,
            dropped_items=downlink_result.dropped_item_count,
            metadata_delivered_items=downlink_result.metadata_delivered_count,
            high_value_metadata_delivered_items=downlink_result.high_value_metadata_delivered_count,
            delivered_high_value_payloads=downlink_result.delivered_high_value_payload_count,
            average_waiting_time_s=downlink_result.delivered_waiting_time_s / max(downlink_result.delivered_item_count, 1),
            miss_signal=miss_signal,
            false_alarm_signal=false_alarm_signal,
            detection_quality=detection_quality,
            transmitted_task_ids=self._format_task_ids(downlink_result.transmitted_task_ids),
            delivered_task_ids=self._format_task_ids(downlink_result.delivered_task_ids),
            expired_task_ids=self._format_task_ids(downlink_result.expired_task_ids),
            dropped_task_ids=self._format_task_ids(downlink_result.dropped_task_ids),
            delivered_payload_ids=self._format_task_ids(downlink_result.delivered_payload_ids),
            delivered_product_types=self._format_task_ids(downlink_result.delivered_product_types),
            expired_payload_ids=self._format_task_ids(downlink_result.expired_payload_ids),
            expired_product_types=self._format_task_ids(downlink_result.expired_product_types),
            dropped_payload_ids=self._format_task_ids(downlink_result.dropped_payload_ids),
            dropped_product_types=self._format_task_ids(downlink_result.dropped_product_types),
            delivered_task_event_times_s=self._format_event_times(downlink_result.delivered_task_event_times_s),
            expired_task_event_times_s=self._format_event_times(downlink_result.expired_task_event_times_s),
            dropped_task_event_times_s=self._format_event_times(downlink_result.dropped_task_event_times_s),
            estimated_target_count=int((payload.metadata if payload else {}).get("estimated_target_count", 0)),
            estimated_bbox_area=float((payload.metadata if payload else {}).get("estimated_bbox_area", 0.0)),
            estimated_roi_size_kb=float((payload.metadata if payload else {}).get("estimated_roi_size_kb", 0.0)),
            estimated_roi_context_size_kb=float((payload.metadata if payload else {}).get("estimated_roi_context_size_kb", 0.0)),
            payload_size_source=str((payload.metadata if payload else {}).get("payload_size_source", "unknown")),
            payload_encoding_latency_s=payload_encoding_latency_s,
            payload_encoding_energy_j=payload_encoding_energy_j,
            product_requirement=(task_for_feedback.product_requirement if task_for_feedback is not None else "verification"),
            selected_payload_plan=((payload.payload_plan or payload.payload_policy) if payload else ""),
            selected_product_type=((payload.product_type or payload.payload_policy) if payload else ""),
            residual_product_type=(payload.residual_product_type if payload else ""),
            duplicate_product_attempt_count=downlink_result.duplicate_product_attempt_count,
            zero_increment_product_count=downlink_result.zero_increment_product_count,
        )
        return feedback, E_comp, E_comm, total_latency

    def _remaining_deadline_s(self, timeline_step: TimelineStep) -> float:
        if timeline_step.task is None:
            return 0.0
        elapsed_s = max(
            0.0,
            (float(timeline_step.slot_index) - float(timeline_step.task.arrival_slot)) * float(timeline_step.slot_seconds),
        )
        return max(0.0, float(timeline_step.task.deadline_s) - elapsed_s)

    def _maybe_build_residual_payload(
        self,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        comm_state: CommState,
        detection_result,
        payload,
        lifecycle_state: ProductLifecycleState,
        router: BaseRouter | None = None,
    ):
        task = timeline_step.task
        self.last_proactive_followup_decision = None
        self.last_passive_request_suppressed_by_prior_proactive = 0
        self.last_phase6_pre_eval_log = {}
        if task is None or detection_result is None:
            return payload
        self.last_phase6_pre_eval_log = self._build_phase6_pre_eval_log(timeline_step, lifecycle_state)
        phase6_cfg = phase6_proactive_config(self.simulator_cfg)
        if bool(phase6_cfg.get("enabled", False)):
            residual_product_type = sorted(
                product for product in lifecycle_state.pending_products if product in {"roi", "roi_context"}
            )
            payload_size_mb = 1.0
            if len(residual_product_type) == 1:
                payload_size_mb, _diagnostics = estimate_payload_size_with_diagnostics(
                    residual_product_type[0],
                    detection_result=None,
                    task=None,
                    payload_profiles_cfg=self.payload_profiles_cfg,
                )
            queue_pressure = comm_state.queue_mb / max(comm_state.queue_max_mb, 1.0e-9)
            remaining_deadline_s = self._remaining_deadline_s(timeline_step)
            future_contact = self._roi_context_future_contact_feasibility(
                timeline_step=timeline_step,
                task=task,
                payload_size_mb=payload_size_mb,
                comm_state=comm_state,
                start_after_current=not bool(timeline_step.link_trace.in_contact),
            )
            future_contact_feasible = bool(future_contact.get("future_contact_feasible", 0))
            _gate_enabled, _profile_name, gate_profile = load_gate_profile(self.simulator_cfg)
            allow_queued_without_current = bool(gate_profile.get("allow_queued_followup_without_current_contact", False))
            contact_opportunity_score = (
                1.0 if timeline_step.link_trace.in_contact or (allow_queued_without_current and future_contact_feasible) else 0.0
            )
            budget_counts = self._roi_context_budget_counts(timeline_step, "queued")
            proactive_decision = evaluate_proactive_followup_eligibility(
                lifecycle_state,
                {
                    "variant": "" if router is None else router.router_name,
                    "router_name": "" if router is None else router.router_name,
                    "use_lyapunov_pricing": bool(getattr(router, "use_lyapunov_pricing", False)),
                    "use_mode_controller": bool(getattr(router, "use_mode_controller", False)),
                    "queue_mb": comm_state.queue_mb,
                    "queue_max_mb": comm_state.queue_max_mb,
                    "queue_pressure": queue_pressure,
                    "queued_products": self._queued_product_keys(),
                    "remaining_deadline_s": remaining_deadline_s,
                    "deadline_s": task.deadline_s,
                    "battery_soc": resource_state.battery_soc,
                    "thermal_pressure": resource_state.thermal_pressure,
                    "task_value": task_effective_value(task),
                    "high_value_flag": task.high_value_flag,
                    "emergency_flag": task.emergency_flag,
                    "value_level": str(getattr(task, "value_level", "") or (task.metadata or {}).get("value_level", "")),
                    "uncertainty": task.uncertainty,
                    "model_quality_factor": getattr(detection_result, "estimated_quality", 1.0),
                    "contact_available": timeline_step.link_trace.in_contact,
                    "contact_remaining_window_s": timeline_step.link_trace.remaining_window_s,
                    "contact_opportunity_score": contact_opportunity_score,
                    "future_contact_feasible": future_contact_feasible,
                    "followup_mode": "queued",
                    **budget_counts,
                    "payload_size_mb": payload_size_mb,
                    "z_k_guided_followup_permission": bool(
                        router is not None
                        and getattr(router, "use_lyapunov_pricing", False)
                        and getattr(router, "use_mode_controller", False)
                        and self.last_guidance_bundle is not None
                    ),
                },
                phase6_cfg,
            )
            self.last_proactive_followup_decision = proactive_decision
            if proactive_decision.eriv_breakdown is not None:
                self._record_value_density_trace(
                    timeline_step=timeline_step,
                    task_id=task.task_id,
                    stage="current_task_proactive_residual_gate",
                    product_type=proactive_decision.residual_product_type,
                    residual_delta_u_i=proactive_decision.eriv_breakdown.residual_delta_u_I,
                    detection_quality=proactive_decision.eriv_breakdown.model_quality_factor,
                    timeliness_factor=proactive_decision.eriv_breakdown.timeliness_factor,
                    payload_size_mb=proactive_decision.eriv_breakdown.payload_size_mb,
                    value_density=proactive_decision.eriv_breakdown.value_density_per_mb,
                    information_density=proactive_decision.eriv_breakdown.information_density,
                    eriv_score=proactive_decision.eriv_score,
                    followup_mode="queued",
                )
            if proactive_decision.proactive_flag:
                current_time_s = float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
                if lifecycle_state.first_alert_delivered_time_s is not None:
                    lifecycle_state.time_from_metadata_to_followup_s = max(
                        0.0,
                        current_time_s - lifecycle_state.first_alert_delivered_time_s,
                    )
                residual_payload = generate_residual_payload(
                    task=task,
                    detection_result=detection_result,
                    residual_product_type=proactive_decision.residual_product_type,
                    slot_index=timeline_step.slot_index,
                    parent_payload_plan=lifecycle_state.residual_parent_payload_plan or (payload.payload_plan if payload else ""),
                    high_value_threshold=float(self.reward_cfg.get("high_value_threshold", 0.75)),
                    payload_profiles_cfg=self.payload_profiles_cfg,
                    delivered_products=set(lifecycle_state.delivered_products),
                    remaining_deadline_s=remaining_deadline_s,
                    continue_request_priority_bonus=float(phase6_cfg.get("proactive_queue_bonus", 0.0)),
                    followup_trigger="proactive_followup",
                )
                residual_payload.metadata = {
                    **residual_payload.metadata,
                    "expected_residual_information_value": proactive_decision.eriv_score,
                    "ERIV": proactive_decision.eriv_score,
                    "z_k_guided_followup_permission": True,
                    "proactive_before_passive_request": lifecycle_state.residual_requested_count == 0,
                }
                guard_decision = self._followup_hard_safety_guard(
                    timeline_step=timeline_step,
                    resource_state=resource_state,
                    comm_state=comm_state,
                    payload=residual_payload,
                    lifecycle_state=lifecycle_state,
                    task=task,
                    operation="current_task_proactive_residual",
                    require_same_slot_delivery=False,
                    allow_no_contact_enqueue=True,
                    predict_immediate_downlink=bool(timeline_step.link_trace.in_contact),
                    allow_existing_pending=True,
                )
                if not guard_decision.allowed:
                    self._record_same_slot_vs_queued_followup_trace(
                        timeline_step=timeline_step,
                        task_id=task.task_id,
                        stage="current_task_proactive_residual_gate",
                        followup_mode="queued",
                        action="blocked_by_hard_safety_guard",
                        reason_code=guard_decision.reason_code,
                        payload_size_mb=payload_size_mb,
                        same_slot_delivery_allowed=False,
                        queued_followup_allowed=False,
                        future_contact_feasible=future_contact_feasible,
                    )
                    return payload
                if proactive_decision.residual_product_type == "roi_context":
                    self._increment_roi_context_budget(timeline_step, "queued")
                self._record_same_slot_vs_queued_followup_trace(
                    timeline_step=timeline_step,
                    task_id=task.task_id,
                    stage="current_task_proactive_residual_gate",
                    followup_mode="queued",
                    action="enqueue_candidate_payload",
                    reason_code="queued_followup_payload_ready",
                    payload_size_mb=payload_size_mb,
                    same_slot_delivery_allowed=bool(timeline_step.link_trace.in_contact),
                    queued_followup_allowed=True,
                    future_contact_feasible=future_contact_feasible,
                    enqueue_flag=1,
                )
                lifecycle_state.residual_utility_generated += residual_payload.incremental_utility_estimate
                return residual_payload
            else:
                self._record_same_slot_vs_queued_followup_trace(
                    timeline_step=timeline_step,
                    task_id=task.task_id,
                    stage="current_task_proactive_residual_gate",
                    followup_mode="queued",
                    action="blocked",
                    reason_code=proactive_decision.reason_code,
                    payload_size_mb=payload_size_mb,
                    same_slot_delivery_allowed=False,
                    queued_followup_allowed=False,
                    future_contact_feasible=future_contact_feasible,
                )

        cfg = phase5_continue_config(self.simulator_cfg)
        if not bool(cfg.get("enabled", False)):
            self.last_continue_request_decision = None
            return payload

        remaining_deadline_s = self._remaining_deadline_s(timeline_step)
        decision = evaluate_virtual_continue_request(
            lifecycle_state,
            {
                "queue_mb": comm_state.queue_mb,
                "queue_max_mb": comm_state.queue_max_mb,
                "queued_products": self._queued_product_keys(),
                "remaining_deadline_s": remaining_deadline_s,
                "battery_soc": resource_state.battery_soc,
                "thermal_pressure": resource_state.thermal_pressure,
                "task_value": task_effective_value(task),
                "high_value_flag": task.high_value_flag,
                "emergency_flag": task.emergency_flag,
                "value_level": str(getattr(task, "value_level", "") or (task.metadata or {}).get("value_level", "")),
                "uncertainty": task.uncertainty,
            },
            cfg,
        )
        self.last_continue_request_decision = decision
        if not decision.request_flag:
            if (
                lifecycle_state.proactive_followup_enqueue_count > 0
                and any(blocker == "duplicate_residual_already_queued" for blocker in decision.blockers)
            ):
                self.last_passive_request_suppressed_by_prior_proactive = 1
            return payload

        residual_payload = generate_residual_payload(
            task=task,
            detection_result=detection_result,
            residual_product_type=decision.residual_product_type,
            slot_index=timeline_step.slot_index,
            parent_payload_plan=lifecycle_state.residual_parent_payload_plan or (payload.payload_plan if payload else ""),
            high_value_threshold=float(self.reward_cfg.get("high_value_threshold", 0.75)),
            payload_profiles_cfg=self.payload_profiles_cfg,
            delivered_products=set(lifecycle_state.delivered_products),
            remaining_deadline_s=remaining_deadline_s,
            continue_request_priority_bonus=float(cfg.get("continue_request_priority_bonus", 0.0)),
            followup_trigger="passive_virtual_continue_request",
        )
        guard_decision = self._followup_hard_safety_guard(
            timeline_step=timeline_step,
            resource_state=resource_state,
            comm_state=comm_state,
            payload=residual_payload,
            lifecycle_state=lifecycle_state,
            task=task,
            operation="passive_virtual_continue_residual",
            require_same_slot_delivery=False,
            allow_no_contact_enqueue=None,
            predict_immediate_downlink=True,
            allow_existing_pending=True,
        )
        if not guard_decision.allowed:
            return payload
        current_time_s = float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
        lifecycle_state.virtual_continue_request_flag = True
        lifecycle_state.residual_requested_count += 1
        lifecycle_state.residual_request_time_s = current_time_s
        if lifecycle_state.first_alert_delivered_time_s is not None:
            lifecycle_state.time_from_metadata_to_followup_s = max(
                0.0,
                current_time_s - lifecycle_state.first_alert_delivered_time_s,
            )
        lifecycle_state.residual_utility_generated += residual_payload.incremental_utility_estimate
        return residual_payload

    def _apply_product_lifecycle_updates(
        self,
        timeline_step: TimelineStep,
        payload,
        downlink_result,
        feedback: ObservedFeedback,
        feedback_task=None,
        router: BaseRouter | None = None,
        detection_result=None,
        resource_state: ResourceState | None = None,
        comm_state: CommState | None = None,
        downlink_throttled: bool = False,
        baseline_compute_energy_j: float = 0.0,
        baseline_comm_energy_j: float = 0.0,
        baseline_compute_heat_gain_scale: float = 1.0,
    ) -> DownlinkResult:
        task = feedback_task or timeline_step.task
        current_state = None
        if task is not None:
            current_state = self._get_product_lifecycle(task)
            if payload is not None:
                product_type = payload.product_type or payload.payload_policy
                if product_type and product_type != "defer":
                    current_state.pending_products.add(product_type)
                if payload.residual_product_type:
                    current_state.pending_products.add(payload.residual_product_type)
                    current_state.residual_parent_payload_plan = payload.payload_plan or payload.payload_policy
                    if current_state.pending_residual_registered_slot is None:
                        current_state.pending_residual_registered_slot = int(timeline_step.slot_index)

        delivered_only_incremental_utility = 0.0
        alert_delivered_utility = 0.0
        verification_delivered_utility = 0.0
        context_evidence_delivered_utility = 0.0
        high_value_delivered_incremental_utility = 0.0
        passive_followup_incremental_utility = 0.0
        proactive_followup_incremental_utility = 0.0
        carryover_proactive_incremental_utility = 0.0
        carryover_proactive_delivery_count = 0
        residual_utility_delivered = 0.0

        for index, task_id in enumerate(downlink_result.delivered_task_ids):
            product_type = downlink_result.delivered_product_types[index] if index < len(downlink_result.delivered_product_types) else ""
            state = self.product_lifecycle.get(task_id)
            if state is None:
                snapshot = self.task_snapshots.get(task_id)
                state = ProductLifecycleState(
                    task_id=task_id,
                    product_requirement=str(getattr(snapshot, "product_requirement", "verification")),
                )
                self.product_lifecycle[task_id] = state
            state.pending_products.discard(product_type)
            if product_type:
                state.pending_products.discard(product_type)
            if product_type == "metadata" and state.first_alert_delivered_time_s is None:
                event_time = (
                    downlink_result.delivered_task_event_times_s[index]
                    if index < len(downlink_result.delivered_task_event_times_s)
                    else float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
                )
                state.first_alert_delivered_time_s = event_time
                state.metadata_delivery_slot = int(timeline_step.slot_index)

            delivered_metadata = (
                downlink_result.delivered_payload_metadata[index]
                if index < len(getattr(downlink_result, "delivered_payload_metadata", []))
                else {}
            )
            candidate_for_delivery = None
            if delivered_metadata.get("followup_source") == "phase6_residual_candidate_carryover":
                candidate_for_delivery = self.residual_candidate_history.get(str(delivered_metadata.get("carryover_candidate_key", "")))
            utility_task = task if task is not None and task_id == task.task_id else None
            if utility_task is None and candidate_for_delivery is not None:
                utility_task = candidate_for_delivery.task_snapshot

            increment = 0.0
            if utility_task is not None and task_id == utility_task.task_id:
                delta_u_i = delta_information_utility(product_type, state.delivered_products, utility_task.product_requirement)
                event_time_s = (
                    downlink_result.delivered_task_event_times_s[index]
                    if index < len(downlink_result.delivered_task_event_times_s)
                    else float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
                )
                total_latency_s = max(0.0, event_time_s - float(utility_task.arrival_slot) * float(timeline_step.slot_seconds))
                timeliness = self._timeliness_factor(total_latency_s, utility_task.deadline_s)
                detection_quality = float(feedback.detection_quality)
                if candidate_for_delivery is not None:
                    detection_quality = float(getattr(candidate_for_delivery.detection_result_snapshot, "estimated_quality", detection_quality))
                increment = compute_gross_incremental_utility(
                    task_value=task_effective_value(utility_task),
                    detection_quality=detection_quality,
                    delta_u_i=delta_u_i,
                    timeliness=timeliness,
                    rho_res=1.0,
                )
                delivered_only_incremental_utility += increment
                if product_type in {"roi", "roi_context"} and state.first_alert_delivered_time_s is not None:
                    followup_trigger = (
                        downlink_result.delivered_followup_triggers[index]
                        if index < len(downlink_result.delivered_followup_triggers)
                        else ""
                    )
                    if not followup_trigger:
                        followup_trigger = str(delivered_metadata.get("followup_trigger", ""))
                    if followup_trigger == "proactive_followup":
                        proactive_followup_incremental_utility += increment
                        state.proactive_followup_delivery_count += 1
                        state.residual_utility_saved_by_proactive += increment
                        if delivered_metadata.get("followup_source") == "phase6_residual_candidate_carryover":
                            carryover_proactive_incremental_utility += increment
                            carryover_proactive_delivery_count += 1
                            state.carryover_proactive_delivery_count += 1
                            state.carryover_proactive_incremental_utility += increment
                            if candidate_for_delivery is not None:
                                self._record_carryover_event(
                                    "proactive_delivered",
                                    timeline_step,
                                    candidate_for_delivery,
                                    proactive_delivery_flag=1,
                                )
                    else:
                        passive_followup_incremental_utility += increment
                    residual_utility_delivered += increment
                    state.residual_delivered_count += 1
                    state.residual_utility_delivered += increment
                    if index < len(downlink_result.residual_queue_wait_times_s):
                        state.residual_queue_wait_s = downlink_result.residual_queue_wait_times_s[index]
                if utility_task.product_requirement == "alert":
                    alert_delivered_utility += increment
                elif utility_task.product_requirement == "verification":
                    verification_delivered_utility += increment
                elif utility_task.product_requirement in {"context", "evidence"}:
                    context_evidence_delivered_utility += increment
                if utility_task.high_value_flag or task_effective_value(utility_task) >= float(self.reward_cfg.get("high_value_threshold", 0.75)):
                    high_value_delivered_incremental_utility += increment
            state.delivered_products.add(product_type)
            state.residual_utility_remaining = residual_information_utility(
                state.delivered_products,
                state.product_requirement,
            )
            staged_task = utility_task or self.task_snapshots.get(task_id)
            staged_comm_state = CommState(
                queue_mb=float(downlink_result.queue_next_mb),
                queue_max_mb=float(comm_state.queue_max_mb) if comm_state is not None else float(self.simulator_cfg.get("queue_max_mb", 1.0)),
                queue_item_count=int(downlink_result.queue_item_count),
            )
            self._maybe_create_staged_delivery_request_after_delivery(
                timeline_step=timeline_step,
                state=state,
                task=staged_task,
                delivered_product_type=product_type,
                delivered_metadata=delivered_metadata,
                resource_state=resource_state,
                comm_state=staged_comm_state,
            )
            if state.residual_utility_remaining <= 1.0e-12 or self._integrated_branch_closure_satisfied(state):
                state.task_closed_flag = True

        self._mark_staged_delivery_from_downlink_result(downlink_result, timeline_step)

        for index, task_id in enumerate(downlink_result.expired_task_ids):
            product_type = downlink_result.expired_product_types[index] if index < len(downlink_result.expired_product_types) else ""
            state = self.product_lifecycle.get(task_id)
            if state is not None:
                state.pending_products.discard(product_type)
                if product_type in {"roi", "roi_context"}:
                    state.residual_expired_count += 1
                    state.residual_utility_lost += max(0.0, state.residual_utility_remaining)
                    state.proactive_followup_expired_count += downlink_result.proactive_followup_expired_count
        for index, task_id in enumerate(downlink_result.dropped_task_ids):
            product_type = downlink_result.dropped_product_types[index] if index < len(downlink_result.dropped_product_types) else ""
            state = self.product_lifecycle.get(task_id)
            if state is not None:
                state.pending_products.discard(product_type)
                if product_type in {"roi", "roi_context"}:
                    state.residual_dropped_count += 1
                    state.residual_utility_lost += max(0.0, state.residual_utility_remaining)
                    state.proactive_followup_dropped_count += downlink_result.proactive_followup_dropped_count

        if current_state is not None:
            current_state.residual_utility_remaining = residual_information_utility(
                current_state.delivered_products,
                current_state.product_requirement,
            )
            self._maybe_register_roi_context_after_roi(
                timeline_step=timeline_step,
                lifecycle_state=current_state,
                task=task,
                detection_result=detection_result,
                router=router,
                resource_state=resource_state,
                comm_state=comm_state,
            )
            feedback.delivered_products = self._format_products(current_state.delivered_products)
            feedback.pending_products = self._format_products(current_state.pending_products)
            feedback.first_alert_delivered_time_s = (
                -1.0 if current_state.first_alert_delivered_time_s is None else current_state.first_alert_delivered_time_s
            )
            feedback.residual_utility_remaining = current_state.residual_utility_remaining
            feedback.virtual_continue_request_count = int(
                bool(getattr(payload, "is_residual", False))
                and getattr(payload, "payload_mb", 0.0) > 0.0
                and getattr(payload, "followup_trigger", "") != "proactive_followup"
            )
            feedback.residual_utility_generated = (
                float(getattr(payload, "incremental_utility_estimate", 0.0))
                if bool(getattr(payload, "is_residual", False))
                else 0.0
            )
            proactive_decision = self.last_proactive_followup_decision
            if proactive_decision is not None:
                current_state.proactive_followup_attempt_count += 1
                feedback.proactive_followup_attempt_count = 1
                feedback.ERIV = proactive_decision.eriv_score
                feedback.ERIV_threshold = proactive_decision.threshold
                feedback.proactive_followup_reason_code = proactive_decision.reason_code
                feedback.proactive_followup_blockers = self._format_products(proactive_decision.blockers)
                z_allowed = int(
                    proactive_decision.eriv_breakdown is not None
                    and proactive_decision.eriv_breakdown.z_k_followup_gate > 0.0
                    and "z_k_guided_followup_permission_blocked" not in proactive_decision.blockers
                )
                feedback.z_k_guided_followup_permission = z_allowed
                feedback.z_k_followup_allowed_count = z_allowed
                feedback.z_k_followup_blocked_count = int(not z_allowed)
                feedback.proactive_queue_pressure_block_count = int("queue_pressure_too_high" in proactive_decision.blockers)
                feedback.proactive_energy_block_count = int("battery_unsafe" in proactive_decision.blockers)
                feedback.proactive_thermal_block_count = int("thermal_unsafe" in proactive_decision.blockers)
                feedback.proactive_deadline_block_count = int("no_useful_remaining_deadline" in proactive_decision.blockers)
                feedback.proactive_contact_block_count = int("no_contact_opportunity" in proactive_decision.blockers)
                if proactive_decision.proactive_flag:
                    feedback.ERIV_selected = proactive_decision.eriv_score
            if downlink_result.residual_enqueue_count:
                current_state.residual_enqueued_count += downlink_result.residual_enqueue_count
                current_state.residual_enqueue_time_s = float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
            if downlink_result.proactive_followup_enqueue_count:
                current_state.proactive_followup_enqueue_count += downlink_result.proactive_followup_enqueue_count
                current_state.proactive_residual_enqueue_time_s = float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
                feedback.proactive_before_passive_request_count = int(
                    bool(getattr(payload, "metadata", {}).get("proactive_before_passive_request", False))
                )
                current_state.proactive_before_passive_request_count += feedback.proactive_before_passive_request_count
                if (
                    payload is not None
                    and getattr(payload, "metadata", {}).get("followup_source") == "phase6_residual_candidate_carryover"
                ):
                    current_state.carryover_proactive_enqueue_count += downlink_result.proactive_followup_enqueue_count
                    self._bump_carryover_log("carryover_proactive_enqueue_count", downlink_result.proactive_followup_enqueue_count)
                    candidate_key = str(getattr(payload, "metadata", {}).get("carryover_candidate_key", self.last_carryover_payload_key))
                    candidate = self.residual_candidate_pool.get(candidate_key)
                    if candidate is not None:
                        self._record_carryover_event(
                            "proactive_enqueued",
                            timeline_step,
                            candidate,
                            decision=self.last_proactive_followup_decision,
                            proactive_enqueue_flag=1,
                        )
                    self._remove_candidate(candidate_key, "proactive_enqueued", timeline_step)
            current_state.duplicate_residual_suppression_count += downlink_result.duplicate_residual_suppression_count
            current_state.proactive_duplicate_suppression_count += downlink_result.proactive_duplicate_suppression_count
            if (
                downlink_result.proactive_duplicate_suppression_count
                and payload is not None
                and getattr(payload, "metadata", {}).get("followup_source") == "phase6_residual_candidate_carryover"
            ):
                self._bump_carryover_log(
                    "residual_candidate_duplicate_suppression_count",
                    downlink_result.proactive_duplicate_suppression_count,
                )
                self._bump_carryover_log("candidate_duplicate_suppressed_count", downlink_result.proactive_duplicate_suppression_count)
                candidate_key = str(getattr(payload, "metadata", {}).get("carryover_candidate_key", self.last_carryover_payload_key))
                candidate = self.residual_candidate_pool.get(candidate_key) or self.residual_candidate_history.get(candidate_key)
                if candidate is not None:
                    self._record_carryover_event(
                        "duplicate_suppressed",
                        timeline_step,
                        candidate,
                        decision=self.last_proactive_followup_decision,
                        duplicate_suppression_flag=1,
                        removal_reason="duplicate_residual_already_queued",
                    )
            if self.last_passive_request_suppressed_by_prior_proactive:
                current_state.passive_request_suppressed_by_prior_proactive_count += 1
                feedback.passive_request_suppressed_by_prior_proactive_count = 1
            new_candidate_key = self._add_residual_candidate_after_lifecycle(
                timeline_step=timeline_step,
                lifecycle_state=current_state,
                payload=payload,
                detection_result=detection_result,
                router=router,
            )
            same_slot_result = self._maybe_process_same_slot_carryover(
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                router=router,
                downlink_result=downlink_result,
                candidate_key=new_candidate_key,
                downlink_throttled=downlink_throttled,
                baseline_compute_energy_j=baseline_compute_energy_j,
                baseline_comm_energy_j=baseline_comm_energy_j,
                baseline_compute_heat_gain_scale=baseline_compute_heat_gain_scale,
            )
            if same_slot_result is not None:
                same_slot_delivery_totals = self._apply_same_slot_delivery_effects(
                    timeline_step,
                    feedback,
                    same_slot_result,
                )
                delivered_only_incremental_utility += same_slot_delivery_totals["delivered_only_incremental_utility"]
                high_value_delivered_incremental_utility += same_slot_delivery_totals["high_value_delivered_incremental_utility"]
                alert_delivered_utility += same_slot_delivery_totals["alert_delivered_utility"]
                verification_delivered_utility += same_slot_delivery_totals["verification_delivered_utility"]
                context_evidence_delivered_utility += same_slot_delivery_totals["context_evidence_delivered_utility"]
                proactive_followup_incremental_utility += same_slot_delivery_totals["proactive_followup_incremental_utility"]
                carryover_proactive_incremental_utility += same_slot_delivery_totals["carryover_proactive_incremental_utility"]
                carryover_proactive_delivery_count += int(same_slot_delivery_totals["carryover_proactive_delivery_count"])
                residual_utility_delivered += same_slot_delivery_totals["residual_utility_delivered"]
                if same_slot_result.residual_enqueue_count:
                    current_state.residual_enqueued_count += same_slot_result.residual_enqueue_count
                    current_state.residual_enqueue_time_s = float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
                if same_slot_result.proactive_followup_enqueue_count:
                    current_state.proactive_followup_enqueue_count += same_slot_result.proactive_followup_enqueue_count
                    current_state.proactive_residual_enqueue_time_s = float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
                    current_state.proactive_before_passive_request_count += int(
                        bool(getattr(self.last_same_slot_carryover_payload, "metadata", {}).get("proactive_before_passive_request", False))
                    )
                current_state.carryover_proactive_enqueue_count += same_slot_result.proactive_followup_enqueue_count
                downlink_result = self._merge_downlink_results(downlink_result, same_slot_result)
                self._refresh_feedback_downlink_fields(feedback, downlink_result)
            post_log = self._build_lifecycle_post_update_log(timeline_step, current_state)
            self.last_lifecycle_post_update_log = post_log
            self._copy_lifecycle_boundary_logs_to_feedback(
                feedback,
                post_log=post_log,
                pool_log=self.last_carryover_pool_log,
            )
        else:
            self._copy_lifecycle_boundary_logs_to_feedback(feedback, pool_log=self.last_carryover_pool_log)
        feedback.delivered_only_incremental_utility = delivered_only_incremental_utility
        feedback.high_value_delivered_incremental_utility = high_value_delivered_incremental_utility
        feedback.alert_delivered_utility = alert_delivered_utility
        feedback.verification_delivered_utility = verification_delivered_utility
        feedback.context_evidence_delivered_utility = context_evidence_delivered_utility
        feedback.residual_enqueue_count = downlink_result.residual_enqueue_count
        feedback.residual_delivery_count = downlink_result.residual_delivery_count
        feedback.residual_expired_count = downlink_result.residual_expired_count
        feedback.residual_dropped_count = downlink_result.residual_dropped_count
        feedback.duplicate_residual_suppression_count = downlink_result.duplicate_residual_suppression_count
        feedback.passive_followup_incremental_utility = passive_followup_incremental_utility
        feedback.proactive_followup_enqueue_count = downlink_result.proactive_followup_enqueue_count
        feedback.proactive_followup_delivery_count = downlink_result.proactive_followup_delivery_count
        feedback.proactive_followup_expired_count = downlink_result.proactive_followup_expired_count
        feedback.proactive_followup_dropped_count = downlink_result.proactive_followup_dropped_count
        feedback.proactive_followup_incremental_utility = proactive_followup_incremental_utility
        feedback.proactive_followup_payload_mb = downlink_result.proactive_followup_payload_mb
        feedback.proactive_duplicate_suppression_count = downlink_result.proactive_duplicate_suppression_count
        feedback.carryover_proactive_delivery_count = carryover_proactive_delivery_count
        feedback.carryover_proactive_incremental_utility = carryover_proactive_incremental_utility
        feedback.residual_utility_saved_by_proactive = proactive_followup_incremental_utility
        feedback.high_value_proactive_followup_count = int(
            current_state is not None
            and (task is not None and (task.high_value_flag or task_effective_value(task) >= float(self.reward_cfg.get("high_value_threshold", 0.75))))
            and bool(downlink_result.proactive_followup_delivery_count)
        )
        feedback.high_value_proactive_followup_utility = (
            proactive_followup_incremental_utility if feedback.high_value_proactive_followup_count else 0.0
        )
        feedback.residual_utility_delivered = residual_utility_delivered
        feedback.residual_utility_lost = float(downlink_result.residual_expired_count + downlink_result.residual_dropped_count) * max(
            0.0,
            current_state.residual_utility_remaining if current_state is not None else 0.0,
        )
        feedback.time_from_metadata_to_followup_s = (
            current_state.time_from_metadata_to_followup_s if current_state is not None else 0.0
        )
        feedback.residual_queue_wait_s = (
            sum(downlink_result.residual_queue_wait_times_s) / max(len(downlink_result.residual_queue_wait_times_s), 1)
            if downlink_result.residual_queue_wait_times_s
            else 0.0
        )
        feedback.followup_payload_mb = downlink_result.followup_payload_mb
        delivered_residual_type = (
            downlink_result.delivered_product_types[0] if downlink_result.residual_delivery_count and downlink_result.delivered_product_types else ""
        )
        feedback.metadata_to_roi_success_count = int(delivered_residual_type == "roi")
        feedback.metadata_to_roi_context_success_count = int(delivered_residual_type == "roi_context")
        feedback.alert_to_verification_conversion_count = int(
            current_state is not None
            and current_state.product_requirement == "verification"
            and bool(feedback.residual_delivery_count)
        )
        feedback.context_evidence_followup_count = int(
            current_state is not None
            and current_state.product_requirement in {"context", "evidence"}
            and bool(feedback.residual_delivery_count)
        )
        return downlink_result

    def run_step(
        self,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        mission_state: MissionState,
        comm_state: CommState,
        router: BaseRouter,
    ):
        context = self.context_builder.build(timeline_step.task, resource_state, comm_state, timeline_step.link_trace)
        long_term_state = self._build_long_term_state(timeline_step, resource_state, comm_state, context)
        self.last_guidance_bundle = self._apply_policy_repair_guidance(
            timeline_step,
            long_term_state,
            self._build_guidance_bundle(timeline_step.slot_index, long_term_state, router),
        )
        self.last_candidate_action_log = []
        self.last_thermal_safe_capacity_trace = []
        self.last_thermal_hysteresis_trace = []
        self.last_queue_priority_log = []
        self.last_carryover_lifecycle_events = []
        self.last_carryover_pool_log = self._empty_carryover_pool_log()
        self.last_carryover_payload_key = ""
        self.last_carryover_feedback_task = None
        self.last_proactive_followup_decision = None
        self.last_same_slot_carryover_trace = []
        self.last_same_slot_carryover_payload = None
        self.last_staged_feasibility_filter_trace = []
        self.last_staged_feasibility_filter_summary = self._empty_staged_feasibility_filter_summary()
        self.last_followup_hard_safety_guard_trace = []
        self.last_roi_context_window_budget_trace = []
        self.last_roi_context_budget_trace = self.last_roi_context_window_budget_trace
        self.last_value_density_trace = []
        self.last_same_slot_vs_queued_followup_trace = []
        self.last_future_contact_feasibility_trace = []
        self.last_duplicate_suppression_trace = []
        self._reset_staged_delivery_step_logs()
        self._refresh_roi_context_budget_state(timeline_step)
        has_task = timeline_step.task is not None
        if has_task:
            self._ensure_staged_delivery_state()
            self.task_snapshots[timeline_step.task.task_id] = timeline_step.task
        pre_staged_admission_result = self._process_staged_delivery_queue_at_slot(
            timeline_step=timeline_step,
            resource_state=resource_state,
            comm_state=comm_state,
            router=router,
        )
        selection_guidance_bundle = self.last_guidance_bundle if router.use_guided_utility else None

        actions = self.action_space.list_actions()
        feedback_task = None

        if has_task:
            predicted_map = {
                action.action_name: self.predictor.predict_action(
                    task=timeline_step.task,
                    context=context,
                    resource_trace=timeline_step.resource_trace,
                    link_trace=timeline_step.link_trace,
                    comm_state=comm_state,
                    action=action,
                    guidance_bundle=selection_guidance_bundle,
                )
                for action in actions
            }
            safe_actions, reasons_by_action = self.safe_filter.filter_actions(timeline_step.task, actions, predicted_map)
            candidate_actions = safe_actions or actions
            candidate_actions = self._apply_staged_feasibility_filter(
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                actions=actions,
                candidate_actions=candidate_actions,
                predicted_map=predicted_map,
            )
            candidate_actions = self._apply_roi_context_gate_to_actions(
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                candidate_actions=candidate_actions,
                predicted_map=predicted_map,
            )
            if hasattr(router, "prepare_decision"):
                router.prepare_decision(
                    timeline_step=timeline_step,
                    resource_state=resource_state,
                    comm_state=comm_state,
                    context=context,
                    predicted_map=predicted_map,
                )
            router.last_selected_action_score = None
            raw_action = router.select_action(context, candidate_actions, predicted_map, selection_guidance_bundle)
            selected_action_score = router.last_selected_action_score
            if selected_action_score is None:
                selected_action_score = predicted_map[raw_action.action_name].predicted_reward
            shield_decision = self.safe_filter.shield_or_fallback(
                timeline_step.task,
                raw_action,
                actions,
                predicted_map,
            )
            executed_action = shield_decision.executed_action
            self._record_staged_feasibility_replacement(executed_action)
            self.last_candidate_action_log = self._build_candidate_action_log(
                timeline_step=timeline_step,
                context=context,
                actions=actions,
                candidate_actions=candidate_actions,
                predicted_map=predicted_map,
                reasons_by_action=reasons_by_action,
                raw_action=raw_action,
                executed_action=executed_action,
                selected_action_score=float(selected_action_score),
                guidance_bundle=self.last_guidance_bundle,
                router=router,
            )
            reference_max_payload_mb = max((metrics.predicted_S_gen_mb for metrics in predicted_map.values()), default=0.0)
            detection_result = self.inference_runner.run(timeline_step.task, executed_action)
            lifecycle_state = self._get_product_lifecycle(timeline_step.task)
            payload = generate_payload(
                task=timeline_step.task,
                detection_result=detection_result,
                payload_policy=executed_action.payload_policy,
                slot_index=timeline_step.slot_index,
                high_value_threshold=float(self.reward_cfg.get("high_value_threshold", 0.75)),
                payload_profiles_cfg=self.payload_profiles_cfg,
                delivered_products=set(lifecycle_state.delivered_products),
            )
            payload = self._maybe_build_residual_payload(
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                detection_result=detection_result,
                payload=payload,
                lifecycle_state=lifecycle_state,
                router=router,
            )
        else:
            predicted_map = {}
            raw_action = self._build_idle_action()
            shield_decision = ShieldDecision(
                raw_action=raw_action,
                executed_action=raw_action,
                safety_corrected=0,
                safety_violation_before_correction=0,
                safety_violation_after_correction=0,
            )
            executed_action = raw_action
            reference_max_payload_mb = 0.0
            detection_result = None
            payload, feedback_task = self._maybe_build_carryover_payload(
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                router=router,
            )
            selected_action_score = 0.0

        planned_downlink_result = self.downlink_queue.preview_slot(
            slot_index=timeline_step.slot_index,
            payload=payload,
            link_trace=timeline_step.link_trace,
        )
        planned_compute_energy_j = self._estimate_compute_energy_j(executed_action if has_task else None, detection_result)
        planned_compute_heat_gain_scale = 1.0
        if has_task and executed_action is not None and executed_action.model_name != "delay_store":
            planned_compute_heat_gain_scale = self.model_registry.get_profile(executed_action.model_name).thermal_gain_scale
        planned_comm_energy_j = self._compute_downlink_comm_energy_j(timeline_step, planned_downlink_result)
        thermal_headroom_cfg = thermal_headroom_config(self.simulator_cfg)
        thermal_capacity_decision = None
        if thermal_headroom_cfg["enabled"]:
            thermal_capacity_decision = self._thermal_headroom_capacity_decision(
                timeline_step=timeline_step,
                resource_state=resource_state,
                nominal_capacity_mb=compute_available_capacity_mb(
                    timeline_step.link_trace,
                    float(timeline_step.slot_seconds),
                    float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
                ),
                planned_compute_energy_j=planned_compute_energy_j,
                planned_compute_heat_gain_scale=planned_compute_heat_gain_scale,
            )
            downlink_throttled = int(thermal_capacity_decision.safe_capacity_mb <= float(self.simulator_cfg["epsilon"]) and planned_downlink_result.S_down_total_mb > float(self.simulator_cfg["epsilon"]))
            downlink_throttle_reason = thermal_capacity_decision.zero_capacity_reason or ("thermal_headroom_partial" if thermal_capacity_decision.safe_capacity_mb + float(self.simulator_cfg["epsilon"]) < planned_downlink_result.S_down_total_mb else "none")
        else:
            downlink_throttled, downlink_throttle_reason = self._evaluate_downlink_throttle(
                timeline_step=timeline_step,
                resource_state=resource_state,
                planned_downlink_mb=planned_downlink_result.S_down_total_mb,
                planned_compute_energy_j=planned_compute_energy_j,
                planned_comm_energy_j=planned_comm_energy_j,
                planned_compute_heat_gain_scale=planned_compute_heat_gain_scale,
            )
        downlink_result = self.downlink_queue.process_slot(
            slot_index=timeline_step.slot_index,
            payload=payload,
            link_trace=timeline_step.link_trace,
            capacity_override_mb=(thermal_capacity_decision.safe_capacity_mb if thermal_capacity_decision is not None else (0.0 if downlink_throttled else None)),
            contact_window_id=self._contact_window_id_for_slot(int(timeline_step.slot_index)),
        )
        self.last_queue_priority_log = list(getattr(self.downlink_queue, "last_priority_component_log", []))
        if self._staged_delivery_enabled(router) and timeline_step.link_trace.in_contact:
            available_capacity_mb = compute_available_capacity_mb(
                timeline_step.link_trace,
                float(timeline_step.slot_seconds),
                float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
            )
            remaining_capacity_mb = available_capacity_mb
            staged_capacity_used_mb = 0.0
            ordinary_capacity_used_mb = 0.0
            ordinary_before_staged_mb = 0.0
            staged_seen = False
            for row in sorted(self.last_queue_priority_log, key=lambda item: int(item.get("queue_order_before_downlink", 0))):
                if not int(row.get("selected_for_downlink", 0)):
                    continue
                tx_mb = min(max(0.0, float(row.get("remaining_mb", 0.0) or 0.0)), max(0.0, remaining_capacity_mb))
                remaining_capacity_mb = max(0.0, remaining_capacity_mb - tx_mb)
                is_staged = bool(row.get("is_staged_delivery_queue", 0))
                staged_seen = staged_seen or is_staged
                staged_capacity_used_mb += tx_mb if is_staged else 0.0
                ordinary_capacity_used_mb += 0.0 if is_staged else tx_mb
                if not is_staged and not staged_seen:
                    ordinary_before_staged_mb += tx_mb
                delivery_row = dict(row)
                delivery_row.update(
                    {
                        "slot": int(timeline_step.slot_index),
                        "time_s": float(timeline_step.slot_index) * float(timeline_step.slot_seconds),
                        "contact_window_id": self._staged_delivery_contact_window_id,
                        "delivery_order": int(row.get("queue_order_before_downlink", 0)),
                        "tx_mb_estimate": tx_mb,
                    }
                )
                self.last_contact_window_delivery_order_trace.append(delivery_row)
            self.last_ordinary_queue_vs_staged_queue_capacity_trace.append(
                {
                    "slot": int(timeline_step.slot_index),
                    "time_s": float(timeline_step.slot_index) * float(timeline_step.slot_seconds),
                    "contact_window_id": self._staged_delivery_contact_window_id,
                    "available_capacity_mb": float(available_capacity_mb),
                    "ordinary_queue_capacity_used_before_staged_queue_mb": ordinary_before_staged_mb,
                    "ordinary_queue_capacity_used_mb": ordinary_capacity_used_mb,
                    "staged_queue_capacity_used_mb": staged_capacity_used_mb,
                    "total_capacity_used_mb": float(downlink_result.S_down_total_mb),
                    "remaining_capacity_mb": max(0.0, remaining_capacity_mb),
                    "downlink_throttled": int(bool(downlink_throttled)),
                }
            )
        feedback, E_comp, E_comm, total_latency = self._compute_actual_feedback(
            timeline_step=timeline_step,
            action=executed_action if has_task else None,
            detection_result=detection_result,
            payload=payload,
            downlink_result=downlink_result,
            feedback_task=feedback_task,
        )
        downlink_result = self._apply_product_lifecycle_updates(
            timeline_step=timeline_step,
            payload=payload,
            downlink_result=downlink_result,
            feedback=feedback,
            feedback_task=feedback_task,
            router=router,
            detection_result=detection_result,
            resource_state=resource_state,
            comm_state=comm_state,
            downlink_throttled=downlink_throttled,
            baseline_compute_energy_j=E_comp,
            baseline_comm_energy_j=E_comm,
            baseline_compute_heat_gain_scale=planned_compute_heat_gain_scale,
        )
        self._update_guarantee_debt_at_contact_window_end(timeline_step, router)
        self._copy_staged_feasibility_metrics_to_feedback(feedback)
        E_comm = self._compute_downlink_comm_energy_j(timeline_step, downlink_result)
        feedback.E_comm = E_comm
        rate_mb_s = estimate_rate_mb_per_s(
            timeline_step.link_trace,
            float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
        )
        feedback.L_comm = 0.0 if rate_mb_s <= float(self.simulator_cfg["epsilon"]) else downlink_result.S_down_current_task_mb / rate_mb_s
        feedback.total_latency_s = feedback.L_inf + feedback.L_queue + feedback.L_comm
        feedback.downlink_throttled = downlink_throttled
        feedback.downlink_throttle_reason = downlink_throttle_reason
        feedback.planned_downlink_mb_before_throttle = planned_downlink_result.S_down_total_mb
        feedback.actual_downlink_mb_after_throttle = downlink_result.S_down_total_mb
        feedback.planned_comm_energy_j_before_throttle = planned_comm_energy_j
        feedback.actual_comm_energy_j_after_throttle = E_comm

        if thermal_capacity_decision is not None:
            self.last_thermal_safe_capacity_trace.append(
                {
                    "slot": int(timeline_step.slot_index),
                    "time_s": float(timeline_step.slot_index) * float(timeline_step.slot_seconds),
                    "contact_id": self._contact_window_id_for_slot(int(timeline_step.slot_index)),
                    "current_thermal": float(resource_state.thermal_pressure),
                    "nominal_capacity_bytes": thermal_capacity_decision.nominal_capacity_mb * 1_000_000.0,
                    "safe_capacity_bytes": thermal_capacity_decision.safe_capacity_mb * 1_000_000.0,
                    "actual_transmitted_bytes": float(downlink_result.S_down_total_mb) * 1_000_000.0,
                    "predicted_thermal_at_nominal_capacity": thermal_capacity_decision.predicted_nominal_thermal,
                    "predicted_thermal_at_safe_capacity": thermal_capacity_decision.predicted_safe_thermal,
                    "predicted_compute_only_thermal": thermal_capacity_decision.predicted_compute_only_thermal,
                    "search_iterations": thermal_capacity_decision.search_iterations,
                    "zero_capacity_reason": thermal_capacity_decision.zero_capacity_reason,
                    "monotonic": int(thermal_capacity_decision.monotonic),
                    "queue_bytes_after": float(downlink_result.queue_next_mb) * 1_000_000.0,
                    "queue_bytes_before": (float(downlink_result.queue_next_mb) + float(downlink_result.S_down_total_mb)) * 1_000_000.0,
                }
            )

        next_battery = update_battery_soc(
            battery_soc=resource_state.battery_soc,
            solar_energy_j=timeline_step.resource_trace.solar_energy_j,
            base_power_w=timeline_step.resource_trace.base_power_w,
            slot_seconds=timeline_step.slot_seconds,
            compute_energy_j=E_comp,
            comm_energy_j=E_comm,
            battery_capacity_j=float(self.simulator_cfg["battery_capacity_j"]),
            charge_efficiency=float(self.simulator_cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(self.simulator_cfg.get("battery_discharge_efficiency", 1.0)),
        )
        next_ai_budget = update_ai_energy_budget(
            ai_energy_budget=resource_state.ai_energy_budget,
            compute_energy_j=E_comp,
            E_AI_ref=float(self.simulator_cfg["E_AI_ref"]),
            recharge=float(timeline_step.resource_trace.solar_energy_j / float(self.simulator_cfg["E_AI_ref"]) * 0.05),
        )
        next_thermal = update_thermal_pressure(
            thermal_pressure=resource_state.thermal_pressure,
            compute_energy_j=E_comp,
            comm_energy_j=E_comm,
            cooling=float(self.simulator_cfg["cooling_per_slot"]),
            compute_heat_gain_scale=planned_compute_heat_gain_scale,
            slot_seconds=float(timeline_step.slot_seconds),
            reference_slot_seconds=float(self.simulator_cfg.get("thermal_reference_slot_seconds", 60.0)),
            temperature_reference_c=timeline_step.resource_trace.temperature_reference_c,
            is_sunlit=timeline_step.resource_trace.is_sunlit,
            passive_cooling=float(self.simulator_cfg.get("thermal_passive_cooling", 0.12)),
            sunlight_gain=float(self.simulator_cfg.get("thermal_sunlight_gain", 0.015)),
            eclipse_cooling_bonus=float(self.simulator_cfg.get("thermal_eclipse_cooling_bonus", 0.03)),
            reference_min_c=float(self.simulator_cfg.get("thermal_reference_min_c", -10.0)),
            reference_max_c=float(self.simulator_cfg.get("thermal_reference_max_c", 30.0)),
        )

        actual_resource_safety_reasons: list[str] = []
        if next_battery < float(self.simulator_cfg["battery_min"]):
            actual_resource_safety_reasons.append("battery reserve violation")
        if next_thermal > float(self.simulator_cfg["thermal_max"]):
            actual_resource_safety_reasons.append("thermal pressure violation")
        if downlink_result.queue_next_mb > float(self.simulator_cfg["queue_max_mb"]):
            actual_resource_safety_reasons.append("queue overflow violation")
        resource_safety_violation = int(bool(actual_resource_safety_reasons))
        feedback.safety_violation = resource_safety_violation
        feedback.resource_safety_reasons = actual_resource_safety_reasons
        risk_breakdown = self._compute_long_term_risk_breakdown(
            next_battery=next_battery,
            next_thermal=next_thermal,
            queue_next_mb=downlink_result.queue_next_mb,
            actual_downlink_mb=downlink_result.S_down_total_mb,
        )
        feedback.battery_next_soc = float(risk_breakdown["battery_next_soc"])
        feedback.thermal_next = float(risk_breakdown["thermal_next"])
        feedback.thermal_guard_exceeded = int(risk_breakdown["thermal_guard_exceeded"])
        feedback.battery_guard_exceeded = int(risk_breakdown["battery_guard_exceeded"])
        feedback.thermal_guard_penalty = float(risk_breakdown["thermal_guard_penalty"])
        feedback.battery_reserve_penalty = float(risk_breakdown["battery_reserve_penalty"])
        feedback.downlink_burden_penalty = float(risk_breakdown["downlink_burden_penalty"])
        feedback.queue_pressure_risk_penalty = float(risk_breakdown["queue_pressure_risk_penalty"])
        feedback.long_term_risk_penalty = float(risk_breakdown["long_term_risk_penalty"])

        if has_task:
            reward = compute_reward(
                task=timeline_step.task,
                context=context,
                payload_information=payload.information_content,
                feedback=feedback,
                reward_cfg=self.reward_cfg,
                simulator_cfg=self.simulator_cfg,
            )
            router.update(context, executed_action, reward.reward)
            payload_information_content = payload.information_content
        else:
            reward = self._build_zero_reward_breakdown()
            payload_information_content = 0.0
        feedback.net_system_reward_slot = (
            reward.net_utility
            - reward.beta_hard_penalty * feedback.long_term_risk_penalty
        )

        resource_state.battery_soc = next_battery
        resource_state.ai_energy_budget = next_ai_budget
        resource_state.thermal_pressure = next_thermal
        comm_state.queue_mb = downlink_result.queue_next_mb
        comm_state.queue_item_count = downlink_result.queue_item_count
        comm_state.effective_opportunity = context.C_t
        mission_state.slot_index = timeline_step.slot_index + 1
        mission_state.completed_tasks += int(feedback.delivery_status == "delivered")
        mission_state.safety_violations += resource_safety_violation

        self.last_feedback = feedback
        self.last_reward_breakdown = reward
        self.last_detection_result = detection_result

        record = build_trajectory_record(
            router_name=router.router_name,
            slot_index=timeline_step.slot_index,
            task=timeline_step.task,
            context=context,
            resource_trace=timeline_step.resource_trace,
            slot_seconds=timeline_step.slot_seconds,
            shield=shield_decision,
            predicted=predicted_map.get(executed_action.action_name),
            feedback=feedback,
            reward=reward,
            reference_max_payload_mb=reference_max_payload_mb,
            payload_information_content=payload_information_content,
            guidance_bundle=self.last_guidance_bundle,
            selected_action_score=float(selected_action_score),
            score_components=getattr(router, "last_score_components", {}),
        )
        return record, reward




