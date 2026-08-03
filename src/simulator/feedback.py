"""Build trajectory records."""

from __future__ import annotations

from src.common.types import (
    ContextVector,
    ObservedFeedback,
    PredictedMetrics,
    ResourceTrace,
    RewardBreakdown,
    ShieldDecision,
    TaskSample,
    TrajectoryRecord,
)
from src.simulator.predictors import Predictor
from src.data.value_skew_calibration import task_effective_value


RESOURCE_REASON_LABELS = {
    "predicted battery too low": "battery_violation",
    "battery reserve violation": "battery_violation",
    "predicted thermal too high": "thermal_violation",
    "thermal pressure violation": "thermal_violation",
    "predicted queue overflow": "queue_violation",
    "queue overflow violation": "queue_violation",
    "AI energy budget violation": "battery_violation",
}
SERVICE_FAILURE_PRIORITY = (
    "deferred",
    "deadline_missed",
    "quality_gap",
    "information_gap",
    "battery_violation",
    "thermal_violation",
    "queue_violation",
)


def _normalize_reason_label(reason: str) -> str:
    reason = reason.strip()
    if reason in RESOURCE_REASON_LABELS:
        return RESOURCE_REASON_LABELS[reason]
    return reason.lower().replace("/", "_").replace(" ", "_")


def _format_reason_labels(reasons: list[str]) -> str:
    labels: list[str] = []
    for reason in reasons:
        label = _normalize_reason_label(reason)
        if label and label not in labels:
            labels.append(label)
    return ";".join(labels) if labels else "none"


def _select_service_failure_reason(reasons: list[str]) -> str:
    labels = {_normalize_reason_label(reason) for reason in reasons}
    for reason in SERVICE_FAILURE_PRIORITY:
        if reason in labels:
            return reason
    return "none"


def _split_semicolon(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _current_task_event_time(task_id: str, task_ids: str, event_times_s: str) -> float:
    ids = _split_semicolon(task_ids)
    times = _split_semicolon(event_times_s)
    for index, item in enumerate(ids):
        if item == task_id and index < len(times):
            try:
                return float(times[index])
            except ValueError:
                return -1.0
    return -1.0


def build_trajectory_record(
    router_name: str,
    slot_index: int,
    task: TaskSample | None,
    context: ContextVector,
    resource_trace: ResourceTrace,
    slot_seconds: float,
    shield: ShieldDecision,
    predicted: PredictedMetrics | None,
    feedback: ObservedFeedback,
    reward: RewardBreakdown,
    reference_max_payload_mb: float,
    payload_information_content: float,
    guidance_bundle: dict[str, object] | None,
    selected_action_score: float,
    score_components: dict[str, object] | None = None,
) -> TrajectoryRecord:
    executed_action = shield.executed_action
    predicted_reward = 0.0 if predicted is None else predicted.predicted_reward
    guidance_bundle = guidance_bundle or {}
    score_components = score_components or {}
    lambda_base = dict(guidance_bundle.get("lambda_base", {}))
    lambda_before_z = dict(guidance_bundle.get("lambda_before_z", {}))
    lambda_after_z = dict(guidance_bundle.get("lambda_after_z", {}))
    guidance_summary = Predictor.summarize_guidance_bundle(guidance_bundle)
    task_deferred = int(
        task is not None
        and (
            executed_action.action_name in {"delay/store", "delay_store", "no-op", "noop", "no_op"}
            or executed_action.model_name == "delay_store"
            or executed_action.payload_policy == "defer"
            or feedback.delivery_status == "deferred"
        )
    )
    deadline_missed = int(task is not None and feedback.total_latency_s > (task.deadline_s if task is not None else 0.0))
    service_failure_reasons: list[str] = []
    if task_deferred:
        service_failure_reasons.append("deferred")
    if deadline_missed:
        service_failure_reasons.append("deadline_missed")
    if task is not None and feedback.delivery_status in {"dropped", "expired", "empty"}:
        service_failure_reasons.append(f"delivery_{feedback.delivery_status}")
    if reward.quality_gap > 0.0:
        service_failure_reasons.append("quality_gap")
    if reward.information_gap > 0.0:
        service_failure_reasons.append("information_gap")
    service_failure_reasons.extend(_normalize_reason_label(reason) for reason in feedback.resource_safety_reasons)
    service_failure_reason = _select_service_failure_reason(service_failure_reasons)
    safety_violation_after_correction = int(feedback.safety_violation)
    task_arrival_time_s = -1.0 if task is None else float(task.arrival_slot) * float(slot_seconds)
    payload_generation_time_s = float(slot_index) * float(slot_seconds) if task is not None and feedback.S_gen_current_mb > 0.0 else -1.0
    deadline_time_s = task_arrival_time_s + float(task.deadline_s) if task is not None else -1.0
    payload_delivery_time_s = (
        -1.0
        if task is None
        else _current_task_event_time(task.task_id, feedback.delivered_task_ids, feedback.delivered_task_event_times_s)
    )
    high_value_threshold = 0.75
    high_value_flag = int(task is not None and (bool(task.high_value_flag) or context.V_t >= high_value_threshold))
    task_metadata = dict(task.metadata or {}) if task is not None else {}
    effective_task_value = 0.0 if task is None else task_effective_value(task)
    delivered_late_flag = int(
        task is not None
        and feedback.delivery_status == "delivered"
        and (payload_delivery_time_s > deadline_time_s if payload_delivery_time_s >= 0.0 else deadline_missed)
    )
    delivered_before_deadline = int(
        task is not None
        and feedback.delivery_status == "delivered"
        and not delivered_late_flag
        and not deadline_missed
    )
    delivered_only_utility = reward.effective_utility if delivered_before_deadline else 0.0
    delivered_only_incremental_utility = feedback.delivered_only_incremental_utility
    queued_provisional_utility = reward.effective_utility if feedback.delivery_status == "queued" else 0.0

    return TrajectoryRecord(
        router_name=router_name,
        slot=slot_index,
        t=slot_index,
        task_id=task.task_id if task is not None else "__idle__",
        has_task=int(task is not None),
        has_new_task=int(task is not None),
        scene_id="" if task is None else task.scene_id,
        observation_window_id="" if task is None else task.observation_window_id,
        task_class="idle" if task is None else task.task_class,
        active_observation_window=int(task is not None and bool(task.observation_window_id)),
        raw_action=shield.raw_action.action_name,
        executed_action=executed_action.action_name,
        safety_corrected=shield.safety_corrected,
        safety_violation_before_correction=shield.safety_violation_before_correction,
        raw_safety_violation=shield.safety_violation_before_correction,
        safety_violation_after_correction=safety_violation_after_correction,
        resource_safety_violation=safety_violation_after_correction,
        final_resource_safety_violation=safety_violation_after_correction,
        task_deferred=task_deferred,
        deadline_missed=deadline_missed,
        service_failure_reason=service_failure_reason,
        safety_violation_reason_before=_format_reason_labels(shield.reasons_before_correction),
        safety_violation_reason_after=_format_reason_labels(feedback.resource_safety_reasons),
        selected_action=executed_action.action_name,
        model_name=executed_action.model_name,
        payload_policy=executed_action.payload_policy,
        selected_model=executed_action.model_name,
        selected_payload=executed_action.payload_policy,
        orbit_index=resource_trace.orbit_index,
        day_index=resource_trace.day_index,
        in_sunlight=int(resource_trace.is_sunlit),
        is_sunlit=int(resource_trace.is_sunlit),
        is_eclipse=int(resource_trace.is_eclipse),
        solar_power_w=resource_trace.solar_power_w,
        solar_energy_j=resource_trace.solar_energy_j,
        wasted_solar_energy_j=resource_trace.wasted_solar_energy_j,
        idle_energy_j=resource_trace.base_power_w * slot_seconds,
        base_power_w=resource_trace.base_power_w,
        dod=1.0 - context.B_t,
        dod_guard_violation=int((1.0 - context.B_t) > float(getattr(resource_trace, "dod_guard", 0.30))),
        temperature_reference_c=resource_trace.temperature_reference_c,
        battery_soc=context.B_t,
        B_t=context.B_t,
        E_AI_t=context.E_t_AI,
        Theta_t=context.Theta_t,
        V_t=context.V_t,
        V_raw=reward.value_raw,
        V_cal=reward.value_calibrated,
        base_prior_score=float(task.base_prior_score if task is not None and task.base_prior_score is not None else task_metadata.get("base_prior_score", effective_task_value)),
        value_level="" if task is None else str(task.value_level or task_metadata.get("value_level", "")),
        value_multiplier=float(task.value_multiplier if task is not None else 1.0),
        value_skew_profile="legacy" if task is None else str(task.value_skew_profile or task_metadata.get("value_skew_profile", "legacy")),
        normalization_alpha=float(task.normalization_alpha if task is not None else 1.0),
        calibrated_task_value=effective_task_value,
        value_assignment_reason="" if task is None else str(task.value_assignment_reason or task_metadata.get("value_assignment_reason", "")),
        deadline_s=0.0 if task is None else task.deadline_s,
        D_t=context.D_t,
        H_t=context.H_t,
        U_t=context.U_t,
        C_t=context.C_t,
        Q_t=context.Q_t,
        lambda_E_base=float(guidance_bundle.get("lambda_E_base", lambda_base.get("lambda_E", 0.0))),
        lambda_L_base=float(guidance_bundle.get("lambda_L_base", lambda_base.get("lambda_L", 0.0))),
        lambda_Q_base=float(guidance_bundle.get("lambda_Q_base", lambda_base.get("lambda_Q", 0.0))),
        lambda_S_base=float(guidance_bundle.get("lambda_S_base", lambda_base.get("lambda_S", 0.0))),
        mode_name=str(guidance_bundle.get("mode_name", "disabled")),
        lambda_E=float(guidance_bundle.get("lambda_E", 0.0)),
        lambda_L=float(guidance_bundle.get("lambda_L", 0.0)),
        lambda_Q=float(guidance_bundle.get("lambda_Q", 0.0)),
        lambda_S=float(guidance_bundle.get("lambda_S", 0.0)),
        exploration_scale=float(guidance_bundle.get("exploration_scale", 1.0)),
        model_complexity_bias=float(guidance_bundle.get("model_complexity_bias", 0.0)),
        payload_size_bias=float(guidance_bundle.get("payload_size_bias", 0.0)),
        high_value_bonus_scale=float(guidance_bundle.get("high_value_bonus_scale", 1.0)),
        safety_margin_scale=float(guidance_bundle.get("safety_margin_scale", 1.0)),
        guidance_bundle_summary=guidance_summary,
        selected_action_score=selected_action_score,
        predicted_reward=predicted_reward,
        effective_utility=reward.effective_utility,
        mission_gain=reward.effective_utility,
        mission_utility=reward.mission_utility,
        resource_penalty=reward.resource_penalty,
        resource_cost=reward.resource_cost,
        penalized_reward=reward.reward,
        observed_reward=reward.reward,
        detection_quality=reward.detection_quality,
        payload_information_content=payload_information_content,
        quality_requirement=reward.quality_requirement,
        information_requirement=reward.information_requirement,
        Q_req=reward.quality_requirement,
        I_req=reward.information_requirement,
        predicted_quality=0.0 if predicted is None else predicted.predicted_detection_quality,
        information_factor=reward.information_sufficiency,
        quality_gap=reward.quality_gap,
        information_gap=reward.information_gap,
        quality_gap_penalty=reward.quality_gap_penalty,
        information_gap_penalty=reward.information_gap_penalty,
        requirement_penalty=reward.requirement_penalty,
        safety_penalty=reward.safety_penalty,
        mission_error_penalty=reward.mission_error_penalty,
        hard_penalty=reward.hard_penalty,
        net_utility=reward.net_utility,
        beta_resource_cost=reward.beta_resource_cost,
        beta_requirement_penalty=reward.beta_requirement_penalty,
        beta_hard_penalty=reward.beta_hard_penalty,
        lifecycle_fields_active=reward.lifecycle_fields_active,
        alert_utility_reserved=reward.alert_utility_reserved,
        confirmation_utility_reserved=reward.confirmation_utility_reserved,
        evidence_utility_reserved=reward.evidence_utility_reserved,
        lifecycle_utility_reserved=reward.lifecycle_utility_reserved,
        S_gen_current_mb=feedback.S_gen_current_mb,
        S_down_total_mb=feedback.S_down_total_mb,
        S_down_current_task_mb=feedback.S_down_current_task_mb,
        reference_max_payload_mb=reference_max_payload_mb,
        queue_mb=feedback.queue_next_mb,
        queue_next_mb=feedback.queue_next_mb,
        delta_queue_mb=feedback.delta_queue_mb,
        compute_energy_j=feedback.E_comp,
        comm_energy_j=feedback.E_comm,
        E_comp=feedback.E_comp,
        E_comm=feedback.E_comm,
        L_inf=feedback.L_inf,
        L_queue=feedback.L_queue,
        L_comm=feedback.L_comm,
        queue_item_count=feedback.queue_item_count,
        delivered_items=feedback.delivered_items,
        expired_items=feedback.expired_items,
        dropped_items=feedback.dropped_items,
        metadata_delivered_items=feedback.metadata_delivered_items,
        high_value_metadata_delivered_items=feedback.high_value_metadata_delivered_items,
        delivered_high_value_payloads=feedback.delivered_high_value_payloads,
        average_waiting_time_s=feedback.average_waiting_time_s,
        delivery_status=feedback.delivery_status,
        downlink_throttled=feedback.downlink_throttled,
        downlink_throttle_reason=feedback.downlink_throttle_reason,
        planned_downlink_mb_before_throttle=feedback.planned_downlink_mb_before_throttle,
        actual_downlink_mb_after_throttle=feedback.actual_downlink_mb_after_throttle,
        planned_comm_energy_j_before_throttle=feedback.planned_comm_energy_j_before_throttle,
        actual_comm_energy_j_after_throttle=feedback.actual_comm_energy_j_after_throttle,
        battery_next_soc=feedback.battery_next_soc,
        thermal_next=feedback.thermal_next,
        thermal_guard_exceeded=feedback.thermal_guard_exceeded,
        battery_guard_exceeded=feedback.battery_guard_exceeded,
        thermal_guard_penalty=feedback.thermal_guard_penalty,
        battery_reserve_penalty=feedback.battery_reserve_penalty,
        downlink_burden_penalty=feedback.downlink_burden_penalty,
        queue_pressure_risk_penalty=feedback.queue_pressure_risk_penalty,
        long_term_risk_penalty=feedback.long_term_risk_penalty,
        net_system_reward_slot=feedback.net_system_reward_slot,
        safety_violation=feedback.safety_violation,
        raw_ucb_bonus=float(score_components.get("raw_ucb_bonus", 0.0)),
        risk_discount=float(score_components.get("risk_discount", 1.0)),
        cost_aware_ucb_bonus=float(score_components.get("cost_aware_ucb_bonus", 0.0)),
        ucb_suppression_reason=str(score_components.get("ucb_suppression_reason", "disabled")),
        raw_info_gain=0.0 if predicted is None else predicted.raw_info_gain,
        delivery_proxy_raw=1.0 if predicted is None else predicted.delivery_proxy_raw,
        delivery_proxy_after_floor=1.0 if predicted is None else predicted.delivery_success_proxy,
        delivery_success_proxy=1.0 if predicted is None else predicted.delivery_success_proxy,
        effective_info_gain=0.0 if predicted is None else predicted.effective_info_gain,
        payload_delivery_risk_reason="disabled" if predicted is None else predicted.payload_delivery_risk_reason,
        estimated_target_count=feedback.estimated_target_count,
        estimated_bbox_area=feedback.estimated_bbox_area,
        estimated_roi_size_kb=feedback.estimated_roi_size_kb,
        estimated_roi_context_size_kb=feedback.estimated_roi_context_size_kb,
        payload_size_source=feedback.payload_size_source,
        payload_encoding_latency_s=feedback.payload_encoding_latency_s,
        payload_encoding_energy_j=feedback.payload_encoding_energy_j,
        lambda_E_before_z=float(lambda_before_z.get("lambda_E", guidance_bundle.get("lambda_E", 0.0))),
        lambda_Q_before_z=float(lambda_before_z.get("lambda_Q", guidance_bundle.get("lambda_Q", 0.0))),
        lambda_S_before_z=float(lambda_before_z.get("lambda_S", guidance_bundle.get("lambda_S", 0.0))),
        lambda_L_before_z=float(lambda_before_z.get("lambda_L", guidance_bundle.get("lambda_L", 0.0))),
        lambda_E_after_z=float(lambda_after_z.get("lambda_E", guidance_bundle.get("lambda_E", 0.0))),
        lambda_Q_after_z=float(lambda_after_z.get("lambda_Q", guidance_bundle.get("lambda_Q", 0.0))),
        lambda_S_after_z=float(lambda_after_z.get("lambda_S", guidance_bundle.get("lambda_S", 0.0))),
        lambda_L_after_z=float(lambda_after_z.get("lambda_L", guidance_bundle.get("lambda_L", 0.0))),
        beta_before_z=float(guidance_bundle.get("beta_before_z", guidance_bundle.get("exploration_scale", 1.0))),
        beta_after_z=float(guidance_bundle.get("beta_after_z", guidance_bundle.get("exploration_scale", 1.0))),
        z_guidance_reason=str(guidance_bundle.get("z_guidance_reason", "disabled")),
        rank_before_z=int(score_components.get("rank_before_z", 0)),
        rank_after_z=int(score_components.get("rank_after_z", 0)),
        transmitted_task_ids=feedback.transmitted_task_ids,
        delivered_task_ids=feedback.delivered_task_ids,
        expired_task_ids=feedback.expired_task_ids,
        dropped_task_ids=feedback.dropped_task_ids,
        task_arrival_time_s=task_arrival_time_s,
        payload_generation_time_s=payload_generation_time_s,
        payload_delivery_time_s=payload_delivery_time_s,
        deadline_time_s=deadline_time_s,
        lifetime_status="current_step_delivered" if delivered_before_deadline else feedback.delivery_status,
        high_value_flag=high_value_flag,
        timeliness_at_decision=reward.timeliness,
        timeliness_at_delivery=reward.timeliness if delivered_before_deadline else 0.0,
        effective_utility_provisional=reward.effective_utility,
        delivered_only_utility=delivered_only_utility,
        delivered_before_deadline_utility=delivered_only_utility,
        queued_provisional_utility=queued_provisional_utility,
        delivered_late_flag=delivered_late_flag,
        expired_flag=int(feedback.delivery_status == "expired"),
        deferred_flag=task_deferred,
        delivered_task_event_times_s=feedback.delivered_task_event_times_s,
        expired_task_event_times_s=feedback.expired_task_event_times_s,
        dropped_task_event_times_s=feedback.dropped_task_event_times_s,
        product_requirement=feedback.product_requirement,
        selected_payload_plan=feedback.selected_payload_plan or executed_action.payload_plan or executed_action.payload_policy,
        selected_product_type=feedback.selected_product_type,
        residual_product_type=feedback.residual_product_type,
        delivered_products=feedback.delivered_products,
        pending_products=feedback.pending_products,
        first_alert_delivered_time_s=feedback.first_alert_delivered_time_s,
        residual_utility_remaining=feedback.residual_utility_remaining,
        delivered_only_incremental_utility=delivered_only_incremental_utility,
        high_value_delivered_incremental_utility=feedback.high_value_delivered_incremental_utility,
        alert_delivered_utility=feedback.alert_delivered_utility,
        verification_delivered_utility=feedback.verification_delivered_utility,
        context_evidence_delivered_utility=feedback.context_evidence_delivered_utility,
        duplicate_product_attempt_count=feedback.duplicate_product_attempt_count,
        zero_increment_product_count=feedback.zero_increment_product_count,
        virtual_continue_request_count=feedback.virtual_continue_request_count,
        residual_enqueue_count=feedback.residual_enqueue_count,
        residual_delivery_count=feedback.residual_delivery_count,
        residual_expired_count=feedback.residual_expired_count,
        residual_dropped_count=feedback.residual_dropped_count,
        duplicate_residual_suppression_count=feedback.duplicate_residual_suppression_count,
        passive_followup_incremental_utility=feedback.passive_followup_incremental_utility,
        residual_utility_generated=feedback.residual_utility_generated,
        residual_utility_delivered=feedback.residual_utility_delivered,
        residual_utility_lost=feedback.residual_utility_lost,
        time_from_metadata_to_followup_s=feedback.time_from_metadata_to_followup_s,
        residual_queue_wait_s=feedback.residual_queue_wait_s,
        followup_payload_mb=feedback.followup_payload_mb,
        metadata_to_roi_success_count=feedback.metadata_to_roi_success_count,
        metadata_to_roi_context_success_count=feedback.metadata_to_roi_context_success_count,
        alert_to_verification_conversion_count=feedback.alert_to_verification_conversion_count,
        context_evidence_followup_count=feedback.context_evidence_followup_count,
        proactive_followup_attempt_count=feedback.proactive_followup_attempt_count,
        proactive_followup_enqueue_count=feedback.proactive_followup_enqueue_count,
        proactive_followup_delivery_count=feedback.proactive_followup_delivery_count,
        proactive_followup_expired_count=feedback.proactive_followup_expired_count,
        proactive_followup_dropped_count=feedback.proactive_followup_dropped_count,
        proactive_followup_incremental_utility=feedback.proactive_followup_incremental_utility,
        proactive_followup_payload_mb=feedback.proactive_followup_payload_mb,
        proactive_before_passive_request_count=feedback.proactive_before_passive_request_count,
        passive_request_suppressed_by_prior_proactive_count=feedback.passive_request_suppressed_by_prior_proactive_count,
        proactive_duplicate_suppression_count=feedback.proactive_duplicate_suppression_count,
        ERIV=feedback.ERIV,
        ERIV_selected=feedback.ERIV_selected,
        ERIV_threshold=feedback.ERIV_threshold,
        z_k_followup_allowed_count=feedback.z_k_followup_allowed_count,
        z_k_followup_blocked_count=feedback.z_k_followup_blocked_count,
        proactive_queue_pressure_block_count=feedback.proactive_queue_pressure_block_count,
        proactive_energy_block_count=feedback.proactive_energy_block_count,
        proactive_thermal_block_count=feedback.proactive_thermal_block_count,
        proactive_deadline_block_count=feedback.proactive_deadline_block_count,
        proactive_contact_block_count=feedback.proactive_contact_block_count,
        residual_utility_lost_after_no_proactive=feedback.residual_utility_lost_after_no_proactive,
        residual_utility_saved_by_proactive=feedback.residual_utility_saved_by_proactive,
        high_value_proactive_followup_count=feedback.high_value_proactive_followup_count,
        high_value_proactive_followup_utility=feedback.high_value_proactive_followup_utility,
        proactive_followup_reason_code=feedback.proactive_followup_reason_code,
        proactive_followup_blockers=feedback.proactive_followup_blockers,
        z_k_guided_followup_permission=feedback.z_k_guided_followup_permission,
        lifecycle_pre_delivered_products=feedback.lifecycle_pre_delivered_products,
        lifecycle_pre_pending_products=feedback.lifecycle_pre_pending_products,
        lifecycle_pre_residual_utility_remaining=feedback.lifecycle_pre_residual_utility_remaining,
        lifecycle_pre_task_expired_flag=feedback.lifecycle_pre_task_expired_flag,
        lifecycle_pre_task_closed_flag=feedback.lifecycle_pre_task_closed_flag,
        lifecycle_pre_metadata_delivered_flag=feedback.lifecycle_pre_metadata_delivered_flag,
        lifecycle_pre_pending_residual_exists_flag=feedback.lifecycle_pre_pending_residual_exists_flag,
        lifecycle_pre_true_candidate_flag=feedback.lifecycle_pre_true_candidate_flag,
        lifecycle_post_delivered_products=feedback.lifecycle_post_delivered_products,
        lifecycle_post_pending_products=feedback.lifecycle_post_pending_products,
        lifecycle_post_residual_utility_remaining=feedback.lifecycle_post_residual_utility_remaining,
        lifecycle_post_metadata_delivery_slot=feedback.lifecycle_post_metadata_delivery_slot,
        lifecycle_post_pending_residual_registered_slot=feedback.lifecycle_post_pending_residual_registered_slot,
        lifecycle_post_true_candidate_flag=feedback.lifecycle_post_true_candidate_flag,
        residual_candidate_pool_size=feedback.residual_candidate_pool_size,
        candidate_added_count=feedback.candidate_added_count,
        candidate_removed_count=feedback.candidate_removed_count,
        candidate_re_evaluated_count=feedback.candidate_re_evaluated_count,
        candidate_expired_count=feedback.candidate_expired_count,
        candidate_closed_count=feedback.candidate_closed_count,
        candidate_duplicate_suppressed_count=feedback.candidate_duplicate_suppressed_count,
        residual_candidate_pool_add_count=feedback.residual_candidate_pool_add_count,
        residual_candidate_pool_size_max=feedback.residual_candidate_pool_size_max,
        residual_candidate_re_evaluation_count=feedback.residual_candidate_re_evaluation_count,
        residual_candidate_re_evaluation_after_metadata_delivery_count=feedback.residual_candidate_re_evaluation_after_metadata_delivery_count,
        residual_candidate_true_count=feedback.residual_candidate_true_count,
        residual_candidate_expired_count=feedback.residual_candidate_expired_count,
        residual_candidate_removed_count=feedback.residual_candidate_removed_count,
        residual_candidate_duplicate_suppression_count=feedback.residual_candidate_duplicate_suppression_count,
        residual_candidate_age_s=feedback.residual_candidate_age_s,
        carryover_proactive_attempt_count=feedback.carryover_proactive_attempt_count,
        carryover_proactive_enqueue_count=feedback.carryover_proactive_enqueue_count,
        carryover_proactive_delivery_count=feedback.carryover_proactive_delivery_count,
        carryover_proactive_incremental_utility=feedback.carryover_proactive_incremental_utility,
        carryover_before_passive_request_count=feedback.carryover_before_passive_request_count,
        carryover_passive_request_suppressed_count=feedback.carryover_passive_request_suppressed_count,
        carryover_eriv=feedback.carryover_eriv,
        carryover_eriv_selected=feedback.carryover_eriv_selected,
        carryover_contact_block_count=feedback.carryover_contact_block_count,
        carryover_thermal_block_count=feedback.carryover_thermal_block_count,
        carryover_queue_block_count=feedback.carryover_queue_block_count,
        carryover_deadline_block_count=feedback.carryover_deadline_block_count,
        carryover_energy_block_count=feedback.carryover_energy_block_count,
        same_slot_candidate_detected_count=feedback.same_slot_candidate_detected_count,
        same_slot_candidate_evaluated_count=feedback.same_slot_candidate_evaluated_count,
        same_slot_carryover_enqueue_count=feedback.same_slot_carryover_enqueue_count,
        same_slot_carryover_delivery_count=feedback.same_slot_carryover_delivery_count,
        same_slot_carryover_duplicate_block_count=feedback.same_slot_carryover_duplicate_block_count,
        same_slot_carryover_contact_block_count=feedback.same_slot_carryover_contact_block_count,
        same_slot_opportunity_saved_from_tplus1_count=feedback.same_slot_opportunity_saved_from_tplus1_count,
        staged_feasibility_filter_enabled=feedback.staged_feasibility_filter_enabled,
        staged_candidate_evaluated_count=feedback.staged_candidate_evaluated_count,
        staged_candidate_feasible_count=feedback.staged_candidate_feasible_count,
        staged_candidate_infeasible_count=feedback.staged_candidate_infeasible_count,
        staged_candidate_blocked_by_contact_count=feedback.staged_candidate_blocked_by_contact_count,
        staged_candidate_blocked_by_deadline_count=feedback.staged_candidate_blocked_by_deadline_count,
        staged_candidate_blocked_by_queue_count=feedback.staged_candidate_blocked_by_queue_count,
        staged_candidate_blocked_by_energy_count=feedback.staged_candidate_blocked_by_energy_count,
        staged_candidate_blocked_by_thermal_count=feedback.staged_candidate_blocked_by_thermal_count,
        staged_candidate_demoted_count=feedback.staged_candidate_demoted_count,
        staged_replaced_by_direct_roi_count=feedback.staged_replaced_by_direct_roi_count,
        staged_replaced_by_metadata_only_count=feedback.staged_replaced_by_metadata_only_count,
        staged_replaced_by_defer_count=feedback.staged_replaced_by_defer_count,
        metadata_only_after_filter_count=feedback.metadata_only_after_filter_count,
        residual_completion_after_filter_count=feedback.residual_completion_after_filter_count,
        passive_would_miss_but_proactive_meets_count=feedback.passive_would_miss_but_proactive_meets_count,
        proactive_and_passive_both_miss_count=feedback.proactive_and_passive_both_miss_count,
        product_requirement_distribution=f"{feedback.product_requirement}:1" if task is not None else "",
        delivered_product_set_distribution=feedback.delivered_products or "empty",
    )
