"""淇濈暀鍘熻矾寰勪笅鐨勬寚鏍囨帴鍙ｃ€?"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from src.common.constants import DELIVERY_STATUS_DELIVERED, EPSILON
from src.common.types import TrajectoryRecord


def records_to_dicts(records: list[TrajectoryRecord]) -> list[dict]:
    """鎶?dataclass 璁板綍杞负 dict銆?"""

    return [asdict(record) for record in records]


def _is_deadline_satisfied(record: TrajectoryRecord) -> bool:
    return (
        record.delivery_status == DELIVERY_STATUS_DELIVERED
        and (record.L_inf + record.L_queue + record.L_comm) <= record.deadline_s + EPSILON
    )


def _risk_enabled(risk_cfg: dict | None) -> bool:
    return isinstance(risk_cfg, dict) and bool(risk_cfg.get("enabled", False))


def _positive_normalized_square(value: float, guard: float, hard: float) -> float:
    denominator = max(abs(hard - guard), EPSILON)
    return max(0.0, (value - guard) / denominator) ** 2


def _battery_reserve_risk(value: float, guard: float, minimum: float) -> float:
    denominator = max(guard - minimum, EPSILON)
    return max(0.0, (guard - value) / denominator) ** 2


def _compute_terminal_state_penalty(
    risk_cfg: dict | None,
    final_battery: float,
    final_thermal: float,
    final_queue_mb: float,
    queue_max_mb: float,
    battery_min: float,
    thermal_max: float,
) -> float:
    if not _risk_enabled(risk_cfg):
        return 0.0

    battery_guard = float(risk_cfg.get("battery_guard_threshold", 0.80))
    thermal_guard = float(risk_cfg.get("thermal_guard_threshold", 0.85))
    thermal_hard = float(risk_cfg.get("thermal_hard_threshold", thermal_max))
    terminal_battery = float(risk_cfg.get("terminal_battery_weight", 0.0)) * _battery_reserve_risk(
        final_battery,
        battery_guard,
        battery_min,
    )
    terminal_thermal = float(risk_cfg.get("terminal_thermal_weight", 0.0)) * _positive_normalized_square(
        final_thermal,
        thermal_guard,
        thermal_hard,
    )
    terminal_queue = float(risk_cfg.get("terminal_queue_weight", 0.0)) * (final_queue_mb / max(queue_max_mb, EPSILON)) ** 2
    return terminal_battery + terminal_thermal + terminal_queue


def summarize_episode(
    router_name: str,
    records: list[TrajectoryRecord],
    queue_max_mb: float,
    final_battery: float,
    max_thermal: float,
    battery_min: float,
    thermal_max: float,
    high_value_threshold: float,
    risk_cfg: dict | None = None,
    final_thermal: float | None = None,
    final_queue_mb: float | None = None,
) -> dict[str, float | int | str]:
    """璁＄畻 summary.csv 鎵€闇€瀛楁銆?"""

    total_effective_utility = sum(record.effective_utility for record in records)
    total_mission_utility = sum(record.mission_utility for record in records)
    total_penalized_reward = sum(record.penalized_reward for record in records)
    total_resource_penalty = sum(record.resource_penalty for record in records)
    total_quality_gap_penalty = sum(record.quality_gap_penalty for record in records)
    total_information_gap_penalty = sum(record.information_gap_penalty for record in records)
    total_requirement_penalty = sum(record.requirement_penalty for record in records)
    total_safety_penalty = sum(record.safety_penalty for record in records)
    total_mission_error_penalty = sum(record.mission_error_penalty for record in records)
    total_hard_penalty = sum(record.hard_penalty for record in records)
    gross_task_utility = total_effective_utility
    total_thermal_guard_penalty = sum(record.thermal_guard_penalty for record in records)
    total_battery_reserve_penalty = sum(record.battery_reserve_penalty for record in records)
    total_downlink_burden_penalty = sum(record.downlink_burden_penalty for record in records)
    total_queue_pressure_risk_penalty = sum(record.queue_pressure_risk_penalty for record in records)
    total_long_term_risk_penalty = sum(record.long_term_risk_penalty for record in records)
    total_compute_energy = sum(record.E_comp for record in records)
    total_comm_energy = sum(record.E_comm for record in records)
    total_base_energy = sum(record.idle_energy_j for record in records)
    total_energy = total_compute_energy + total_comm_energy
    total_spacecraft_energy = total_base_energy + total_compute_energy + total_comm_energy
    total_generated_payload_mb = sum(record.S_gen_current_mb for record in records)
    total_downlinked_payload_mb = sum(record.S_down_total_mb for record in records)
    delivered_mb = total_downlinked_payload_mb
    delivered_only_incremental_utility = sum(record.delivered_only_incremental_utility for record in records)
    high_value_delivered_incremental_utility = sum(record.high_value_delivered_incremental_utility for record in records)
    alert_delivered_utility = sum(record.alert_delivered_utility for record in records)
    verification_delivered_utility = sum(record.verification_delivered_utility for record in records)
    context_evidence_delivered_utility = sum(record.context_evidence_delivered_utility for record in records)
    duplicate_product_attempt_count = sum(record.duplicate_product_attempt_count for record in records)
    zero_increment_product_count = sum(record.zero_increment_product_count for record in records)
    virtual_continue_request_count = sum(record.virtual_continue_request_count for record in records)
    residual_enqueue_count = sum(record.residual_enqueue_count for record in records)
    residual_delivery_count = sum(record.residual_delivery_count for record in records)
    residual_expired_count = sum(record.residual_expired_count for record in records)
    residual_dropped_count = sum(record.residual_dropped_count for record in records)
    duplicate_residual_suppression_count = sum(record.duplicate_residual_suppression_count for record in records)
    passive_followup_incremental_utility = sum(record.passive_followup_incremental_utility for record in records)
    residual_utility_generated = sum(record.residual_utility_generated for record in records)
    residual_utility_delivered = sum(record.residual_utility_delivered for record in records)
    residual_utility_lost = sum(record.residual_utility_lost for record in records)
    followup_payload_mb = sum(record.followup_payload_mb for record in records)
    residual_followup_attempts = max(residual_enqueue_count, 1)
    metadata_to_roi_success_count = sum(record.metadata_to_roi_success_count for record in records)
    metadata_to_roi_context_success_count = sum(record.metadata_to_roi_context_success_count for record in records)
    alert_to_verification_conversion_count = sum(record.alert_to_verification_conversion_count for record in records)
    context_evidence_followup_count = sum(record.context_evidence_followup_count for record in records)
    proactive_followup_attempt_count = sum(record.proactive_followup_attempt_count for record in records)
    proactive_followup_enqueue_count = sum(record.proactive_followup_enqueue_count for record in records)
    proactive_followup_delivery_count = sum(record.proactive_followup_delivery_count for record in records)
    proactive_followup_expired_count = sum(record.proactive_followup_expired_count for record in records)
    proactive_followup_dropped_count = sum(record.proactive_followup_dropped_count for record in records)
    proactive_followup_incremental_utility = sum(record.proactive_followup_incremental_utility for record in records)
    proactive_followup_payload_mb = sum(record.proactive_followup_payload_mb for record in records)
    proactive_before_passive_request_count = sum(record.proactive_before_passive_request_count for record in records)
    passive_request_suppressed_by_prior_proactive_count = sum(
        record.passive_request_suppressed_by_prior_proactive_count for record in records
    )
    proactive_duplicate_suppression_count = sum(record.proactive_duplicate_suppression_count for record in records)
    eriv_values = [record.ERIV for record in records if record.proactive_followup_attempt_count > 0]
    eriv_selected_values = [record.ERIV_selected for record in records if record.ERIV_selected > 0.0]
    eriv_threshold_values = [record.ERIV_threshold for record in records if record.ERIV_threshold > 0.0]
    z_k_followup_allowed_count = sum(record.z_k_followup_allowed_count for record in records)
    z_k_followup_blocked_count = sum(record.z_k_followup_blocked_count for record in records)
    proactive_queue_pressure_block_count = sum(record.proactive_queue_pressure_block_count for record in records)
    proactive_energy_block_count = sum(record.proactive_energy_block_count for record in records)
    proactive_thermal_block_count = sum(record.proactive_thermal_block_count for record in records)
    proactive_deadline_block_count = sum(record.proactive_deadline_block_count for record in records)
    proactive_contact_block_count = sum(record.proactive_contact_block_count for record in records)
    residual_utility_lost_after_no_proactive = sum(record.residual_utility_lost_after_no_proactive for record in records)
    residual_utility_saved_by_proactive = sum(record.residual_utility_saved_by_proactive for record in records)
    high_value_proactive_followup_count = sum(record.high_value_proactive_followup_count for record in records)
    high_value_proactive_followup_utility = sum(record.high_value_proactive_followup_utility for record in records)
    residual_candidate_pool_add_count = sum(record.residual_candidate_pool_add_count for record in records)
    residual_candidate_pool_size_max = max([record.residual_candidate_pool_size for record in records] + [0])
    residual_candidate_re_evaluation_count = sum(record.residual_candidate_re_evaluation_count for record in records)
    residual_candidate_re_evaluation_after_metadata_delivery_count = sum(
        record.residual_candidate_re_evaluation_after_metadata_delivery_count for record in records
    )
    residual_candidate_true_count = sum(record.residual_candidate_true_count for record in records)
    residual_candidate_expired_count = sum(record.residual_candidate_expired_count for record in records)
    residual_candidate_removed_count = sum(record.residual_candidate_removed_count for record in records)
    residual_candidate_duplicate_suppression_count = sum(
        record.residual_candidate_duplicate_suppression_count for record in records
    )
    residual_candidate_age_values = [
        record.residual_candidate_age_s for record in records if record.residual_candidate_age_s > 0.0
    ]
    carryover_proactive_attempt_count = sum(record.carryover_proactive_attempt_count for record in records)
    carryover_proactive_enqueue_count = sum(record.carryover_proactive_enqueue_count for record in records)
    carryover_proactive_delivery_count = sum(record.carryover_proactive_delivery_count for record in records)
    carryover_proactive_incremental_utility = sum(record.carryover_proactive_incremental_utility for record in records)
    carryover_before_passive_request_count = sum(record.carryover_before_passive_request_count for record in records)
    carryover_passive_request_suppressed_count = sum(record.carryover_passive_request_suppressed_count for record in records)
    carryover_eriv_values = [record.carryover_eriv for record in records if record.carryover_proactive_attempt_count > 0]
    carryover_eriv_selected_values = [record.carryover_eriv_selected for record in records if record.carryover_eriv_selected > 0.0]
    carryover_contact_block_count = sum(record.carryover_contact_block_count for record in records)
    carryover_thermal_block_count = sum(record.carryover_thermal_block_count for record in records)
    carryover_queue_block_count = sum(record.carryover_queue_block_count for record in records)
    carryover_deadline_block_count = sum(record.carryover_deadline_block_count for record in records)
    carryover_energy_block_count = sum(record.carryover_energy_block_count for record in records)
    same_slot_candidate_detected_count = sum(record.same_slot_candidate_detected_count for record in records)
    same_slot_candidate_evaluated_count = sum(record.same_slot_candidate_evaluated_count for record in records)
    same_slot_carryover_enqueue_count = sum(record.same_slot_carryover_enqueue_count for record in records)
    same_slot_carryover_delivery_count = sum(record.same_slot_carryover_delivery_count for record in records)
    same_slot_carryover_duplicate_block_count = sum(record.same_slot_carryover_duplicate_block_count for record in records)
    same_slot_carryover_contact_block_count = sum(record.same_slot_carryover_contact_block_count for record in records)
    same_slot_opportunity_saved_from_tplus1_count = sum(record.same_slot_opportunity_saved_from_tplus1_count for record in records)
    staged_feasibility_filter_enabled = max((record.staged_feasibility_filter_enabled for record in records), default=0)
    staged_candidate_evaluated_count = sum(record.staged_candidate_evaluated_count for record in records)
    staged_candidate_feasible_count = sum(record.staged_candidate_feasible_count for record in records)
    staged_candidate_infeasible_count = sum(record.staged_candidate_infeasible_count for record in records)
    staged_candidate_blocked_by_contact_count = sum(record.staged_candidate_blocked_by_contact_count for record in records)
    staged_candidate_blocked_by_deadline_count = sum(record.staged_candidate_blocked_by_deadline_count for record in records)
    staged_candidate_blocked_by_queue_count = sum(record.staged_candidate_blocked_by_queue_count for record in records)
    staged_candidate_blocked_by_energy_count = sum(record.staged_candidate_blocked_by_energy_count for record in records)
    staged_candidate_blocked_by_thermal_count = sum(record.staged_candidate_blocked_by_thermal_count for record in records)
    staged_candidate_demoted_count = sum(record.staged_candidate_demoted_count for record in records)
    staged_replaced_by_direct_roi_count = sum(record.staged_replaced_by_direct_roi_count for record in records)
    staged_replaced_by_metadata_only_count = sum(record.staged_replaced_by_metadata_only_count for record in records)
    staged_replaced_by_defer_count = sum(record.staged_replaced_by_defer_count for record in records)
    metadata_only_after_filter_count = sum(record.metadata_only_after_filter_count for record in records)
    residual_completion_after_filter_count = sum(record.residual_completion_after_filter_count for record in records)
    passive_would_miss_but_proactive_meets_count = sum(
        record.passive_would_miss_but_proactive_meets_count for record in records
    )
    proactive_and_passive_both_miss_count = sum(record.proactive_and_passive_both_miss_count for record in records)
    followup_time_values = [record.time_from_metadata_to_followup_s for record in records if record.time_from_metadata_to_followup_s > 0.0]
    residual_wait_values = [record.residual_queue_wait_s for record in records if record.residual_queue_wait_s > 0.0]
    product_requirement_distribution = Counter(
        record.product_requirement for record in records if record.has_task and record.product_requirement
    )
    delivered_product_set_distribution = Counter(
        record.delivered_products if record.delivered_products else "empty" for record in records if record.has_task
    )
    metadata_first_rate = sum(
        1 for record in records if record.has_task and record.selected_product_type == "metadata"
    ) / max(sum(record.has_task for record in records), 1)
    staged_plan_rate = sum(
        1 for record in records if record.has_task and record.selected_payload_plan in {"metadata_then_roi", "metadata_then_roi_context"}
    ) / max(sum(record.has_task for record in records), 1)

    deadline_met_tasks = sum(1 for record in records if record.has_task and _is_deadline_satisfied(record))
    queue_overflow_count = sum(1 for record in records if record.queue_next_mb > queue_max_mb + EPSILON)
    raw_safety_violation_count = sum(record.safety_violation_before_correction for record in records)
    final_resource_safety_violation_count = sum(record.resource_safety_violation for record in records)
    safety_violation_count = final_resource_safety_violation_count
    safety_correction_count = sum(record.safety_corrected for record in records)
    task_deferred_count = sum(record.task_deferred for record in records)
    deadline_missed_count = sum(record.deadline_missed for record in records)
    service_failure_count = sum(1 for record in records if record.service_failure_reason != "none")
    downlink_throttle_count = sum(record.downlink_throttled for record in records)
    total_planned_downlink_mb_before_throttle = sum(record.planned_downlink_mb_before_throttle for record in records)
    total_actual_downlink_mb_after_throttle = sum(record.actual_downlink_mb_after_throttle for record in records)
    downlink_throttle_thermal_count = sum(
        1 for record in records if record.downlink_throttled and record.downlink_throttle_reason == "thermal_guard"
    )
    delivered_items = sum(record.delivered_items for record in records)
    expired_items = sum(record.expired_items for record in records)
    dropped_items = sum(record.dropped_items for record in records)
    metadata_delivered_items = sum(record.metadata_delivered_items for record in records)
    high_value_metadata_delivered_items = sum(record.high_value_metadata_delivered_items for record in records)
    delivered_high_value_payloads = sum(record.delivered_high_value_payloads for record in records)
    delivered_waiting_time_s = sum(record.average_waiting_time_s * record.delivered_items for record in records)
    average_queue_mb = sum(record.queue_next_mb for record in records) / max(len(records), 1)
    avg_queue_pressure = average_queue_mb / max(queue_max_mb, EPSILON)
    final_queue_value = records[-1].queue_next_mb if records else 0.0
    if final_queue_mb is not None:
        final_queue_value = final_queue_mb
    final_thermal_value = max_thermal if final_thermal is None else final_thermal
    terminal_state_penalty = _compute_terminal_state_penalty(
        risk_cfg=risk_cfg,
        final_battery=final_battery,
        final_thermal=final_thermal_value,
        final_queue_mb=final_queue_value,
        queue_max_mb=queue_max_mb,
        battery_min=battery_min,
        thermal_max=thermal_max,
    )
    net_system_utility = (
        gross_task_utility
        - total_resource_penalty
        - total_requirement_penalty
        - total_hard_penalty
        - total_long_term_risk_penalty
        - terminal_state_penalty
    )
    thermal_guard_exceed_count = sum(record.thermal_guard_exceeded for record in records)
    thermal_090_exceed_count = sum(1 for record in records if record.thermal_next > 0.90 + EPSILON)
    thermal_091_exceed_count = sum(1 for record in records if record.thermal_next > 0.91 + EPSILON)
    battery_below_080_count = sum(1 for record in records if record.battery_next_soc < 0.80 - EPSILON)
    battery_below_077_count = sum(1 for record in records if record.battery_next_soc < 0.77 - EPSILON)
    battery_below_075_count = sum(1 for record in records if record.battery_next_soc < 0.75 - EPSILON)
    total_generated_tasks = sum(record.has_new_task for record in records)
    completed_tasks = sum(1 for record in records if record.has_task and record.delivery_status == DELIVERY_STATUS_DELIVERED)
    deadline_met_completed_tasks = deadline_met_tasks
    active_observation_slot_count = sum(record.active_observation_window for record in records)
    idle_slot_count = sum(1 for record in records if record.has_task == 0)
    routine_task_count = sum(1 for record in records if record.has_task and record.task_class == "routine")
    high_value_task_count = sum(1 for record in records if record.has_task and record.task_class == "high_value")
    emergency_task_count = sum(1 for record in records if record.has_task and record.task_class == "emergency")
    observation_window_ids = {record.observation_window_id for record in records if record.observation_window_id}
    total_observation_windows = len(observation_window_ids)
    average_tasks_per_window = total_generated_tasks / max(total_observation_windows, 1)

    utility_per_joule = total_penalized_reward / max(total_energy, EPSILON)
    utility_per_downlink_bit = total_penalized_reward / max(total_downlinked_payload_mb * 8_000_000.0, EPSILON)

    high_value_records = [record for record in records if record.V_t >= high_value_threshold]
    high_value_task_utility = sum(record.effective_utility for record in high_value_records)
    emergency_records = [record for record in records if record.has_task and record.task_class == "emergency"]
    emergency_task_utility = sum(record.effective_utility for record in emergency_records)
    high_value_delivered = sum(1 for record in high_value_records if record.has_task and _is_deadline_satisfied(record))
    high_value_delivery_ratio = high_value_delivered / max(len(high_value_records), 1)
    metadata_delivery_ratio = metadata_delivered_items / max(delivered_items, 1)
    high_value_metadata_delivery_ratio = high_value_metadata_delivered_items / max(delivered_high_value_payloads, 1)
    average_waiting_time = delivered_waiting_time_s / max(delivered_items, 1)

    total_reference_payload_mb = sum(record.reference_max_payload_mb for record in records)
    payload_saving_ratio = 0.0
    if total_reference_payload_mb > EPSILON:
        payload_saving_ratio = max(0.0, min(1.0, 1.0 - total_generated_payload_mb / total_reference_payload_mb))

    feasible_episode_flag = int(
        final_resource_safety_violation_count == 0
        and queue_overflow_count == 0
        and final_battery >= battery_min - EPSILON
        and max_thermal <= thermal_max + EPSILON
    )
    resource_feasible_flag = feasible_episode_flag
    min_battery_soc = min((record.battery_soc for record in records), default=final_battery)
    average_battery_soc = sum(record.battery_soc for record in records) / max(len(records), 1)
    mode_names = [record.mode_name for record in records]
    dominant_mode = Counter(mode_names).most_common(1)[0][0] if mode_names else "unknown"
    mode_switch_count = sum(1 for previous, current in zip(mode_names, mode_names[1:]) if previous != current)

    return {
        "router": router_name,
        "router_name": router_name,
        "total_effective_utility": total_effective_utility,
        "total_penalized_reward": total_penalized_reward,
        "total_mission_utility": total_mission_utility,
        "total_resource_penalty": total_resource_penalty,
        "total_quality_gap_penalty": total_quality_gap_penalty,
        "total_information_gap_penalty": total_information_gap_penalty,
        "total_requirement_penalty": total_requirement_penalty,
        "total_safety_penalty": total_safety_penalty,
        "total_mission_error_penalty": total_mission_error_penalty,
        "total_hard_penalty": total_hard_penalty,
        "delivered_only_incremental_utility": delivered_only_incremental_utility,
        "high_value_delivered_incremental_utility": high_value_delivered_incremental_utility,
        "alert_delivered_utility": alert_delivered_utility,
        "verification_delivered_utility": verification_delivered_utility,
        "context_evidence_delivered_utility": context_evidence_delivered_utility,
        "metadata_first_rate": metadata_first_rate,
        "staged_plan_rate": staged_plan_rate,
        "duplicate_product_attempt_count": duplicate_product_attempt_count,
        "zero_increment_product_count": zero_increment_product_count,
        "virtual_continue_request_count": virtual_continue_request_count,
        "virtual_continue_request_rate": virtual_continue_request_count / max(sum(record.has_task for record in records), 1),
        "residual_enqueue_count": residual_enqueue_count,
        "residual_delivery_count": residual_delivery_count,
        "residual_delivery_rate": residual_delivery_count / residual_followup_attempts,
        "residual_expired_count": residual_expired_count,
        "residual_dropped_count": residual_dropped_count,
        "duplicate_residual_suppression_count": duplicate_residual_suppression_count,
        "passive_followup_incremental_utility": passive_followup_incremental_utility,
        "residual_utility_generated": residual_utility_generated,
        "residual_utility_delivered": residual_utility_delivered,
        "residual_utility_lost": residual_utility_lost,
        "metadata_to_roi_success_rate": metadata_to_roi_success_count / residual_followup_attempts,
        "metadata_to_roi_context_success_rate": metadata_to_roi_context_success_count / residual_followup_attempts,
        "average_time_from_metadata_to_followup_s": sum(followup_time_values) / max(len(followup_time_values), 1),
        "average_residual_queue_wait_s": sum(residual_wait_values) / max(len(residual_wait_values), 1),
        "followup_payload_mb": followup_payload_mb,
        "followup_utility_per_mb": passive_followup_incremental_utility / max(followup_payload_mb, EPSILON),
        "alert_to_verification_conversion_rate": alert_to_verification_conversion_count / residual_followup_attempts,
        "context_evidence_followup_rate": context_evidence_followup_count / residual_followup_attempts,
        "proactive_followup_attempt_count": proactive_followup_attempt_count,
        "proactive_followup_enqueue_count": proactive_followup_enqueue_count,
        "proactive_followup_delivery_count": proactive_followup_delivery_count,
        "proactive_followup_delivery_rate": proactive_followup_delivery_count / max(proactive_followup_enqueue_count, 1),
        "proactive_followup_expired_count": proactive_followup_expired_count,
        "proactive_followup_dropped_count": proactive_followup_dropped_count,
        "proactive_followup_incremental_utility": proactive_followup_incremental_utility,
        "proactive_followup_payload_mb": proactive_followup_payload_mb,
        "proactive_followup_utility_per_mb": proactive_followup_incremental_utility / max(proactive_followup_payload_mb, EPSILON),
        "proactive_before_passive_request_count": proactive_before_passive_request_count,
        "passive_request_suppressed_by_prior_proactive_count": passive_request_suppressed_by_prior_proactive_count,
        "proactive_duplicate_suppression_count": proactive_duplicate_suppression_count,
        "ERIV_mean": sum(eriv_values) / max(len(eriv_values), 1),
        "ERIV_selected_mean": sum(eriv_selected_values) / max(len(eriv_selected_values), 1),
        "ERIV_threshold": sum(eriv_threshold_values) / max(len(eriv_threshold_values), 1),
        "z_k_followup_allowed_count": z_k_followup_allowed_count,
        "z_k_followup_blocked_count": z_k_followup_blocked_count,
        "proactive_queue_pressure_block_count": proactive_queue_pressure_block_count,
        "proactive_energy_block_count": proactive_energy_block_count,
        "proactive_thermal_block_count": proactive_thermal_block_count,
        "proactive_deadline_block_count": proactive_deadline_block_count,
        "proactive_contact_block_count": proactive_contact_block_count,
        "residual_utility_lost_after_no_proactive": residual_utility_lost_after_no_proactive,
        "residual_utility_saved_by_proactive": residual_utility_saved_by_proactive,
        "high_value_proactive_followup_count": high_value_proactive_followup_count,
        "high_value_proactive_followup_utility": high_value_proactive_followup_utility,
        "residual_candidate_pool_add_count": residual_candidate_pool_add_count,
        "residual_candidate_pool_size_max": residual_candidate_pool_size_max,
        "residual_candidate_re_evaluation_count": residual_candidate_re_evaluation_count,
        "residual_candidate_re_evaluation_after_metadata_delivery_count": residual_candidate_re_evaluation_after_metadata_delivery_count,
        "residual_candidate_true_count": residual_candidate_true_count,
        "residual_candidate_expired_count": residual_candidate_expired_count,
        "residual_candidate_removed_count": residual_candidate_removed_count,
        "residual_candidate_duplicate_suppression_count": residual_candidate_duplicate_suppression_count,
        "residual_candidate_age_mean_s": sum(residual_candidate_age_values) / max(len(residual_candidate_age_values), 1),
        "residual_candidate_age_max_s": max(residual_candidate_age_values) if residual_candidate_age_values else 0.0,
        "carryover_proactive_attempt_count": carryover_proactive_attempt_count,
        "carryover_proactive_enqueue_count": carryover_proactive_enqueue_count,
        "carryover_proactive_delivery_count": carryover_proactive_delivery_count,
        "carryover_proactive_incremental_utility": carryover_proactive_incremental_utility,
        "carryover_before_passive_request_count": carryover_before_passive_request_count,
        "carryover_passive_request_suppressed_count": carryover_passive_request_suppressed_count,
        "carryover_eriv_mean": sum(carryover_eriv_values) / max(len(carryover_eriv_values), 1),
        "carryover_eriv_selected_mean": sum(carryover_eriv_selected_values) / max(len(carryover_eriv_selected_values), 1),
        "carryover_contact_block_count": carryover_contact_block_count,
        "carryover_thermal_block_count": carryover_thermal_block_count,
        "carryover_queue_block_count": carryover_queue_block_count,
        "carryover_deadline_block_count": carryover_deadline_block_count,
        "carryover_energy_block_count": carryover_energy_block_count,
        "same_slot_candidate_detected_count": same_slot_candidate_detected_count,
        "same_slot_candidate_evaluated_count": same_slot_candidate_evaluated_count,
        "same_slot_carryover_enqueue_count": same_slot_carryover_enqueue_count,
        "same_slot_carryover_delivery_count": same_slot_carryover_delivery_count,
        "same_slot_carryover_duplicate_block_count": same_slot_carryover_duplicate_block_count,
        "same_slot_carryover_contact_block_count": same_slot_carryover_contact_block_count,
        "same_slot_opportunity_saved_from_tplus1_count": same_slot_opportunity_saved_from_tplus1_count,
        "staged_feasibility_filter_enabled": staged_feasibility_filter_enabled,
        "staged_candidate_evaluated_count": staged_candidate_evaluated_count,
        "staged_candidate_feasible_count": staged_candidate_feasible_count,
        "staged_candidate_infeasible_count": staged_candidate_infeasible_count,
        "staged_candidate_blocked_by_contact_count": staged_candidate_blocked_by_contact_count,
        "staged_candidate_blocked_by_deadline_count": staged_candidate_blocked_by_deadline_count,
        "staged_candidate_blocked_by_queue_count": staged_candidate_blocked_by_queue_count,
        "staged_candidate_blocked_by_energy_count": staged_candidate_blocked_by_energy_count,
        "staged_candidate_blocked_by_thermal_count": staged_candidate_blocked_by_thermal_count,
        "staged_candidate_demoted_count": staged_candidate_demoted_count,
        "staged_replaced_by_direct_roi_count": staged_replaced_by_direct_roi_count,
        "staged_replaced_by_metadata_only_count": staged_replaced_by_metadata_only_count,
        "staged_replaced_by_defer_count": staged_replaced_by_defer_count,
        "metadata_only_after_filter_count": metadata_only_after_filter_count,
        "residual_completion_after_filter_count": residual_completion_after_filter_count,
        "passive_would_miss_but_proactive_meets_count": passive_would_miss_but_proactive_meets_count,
        "proactive_and_passive_both_miss_count": proactive_and_passive_both_miss_count,
        "product_requirement_distribution": ";".join(
            f"{key}:{value}" for key, value in sorted(product_requirement_distribution.items())
        ),
        "delivered_product_set_distribution": ";".join(
            f"{key}:{value}" for key, value in sorted(delivered_product_set_distribution.items())
        ),
        "gross_task_utility": gross_task_utility,
        "total_net_utility": sum(record.net_utility for record in records),
        "net_system_utility": net_system_utility,
        "total_long_term_risk_penalty": total_long_term_risk_penalty,
        "total_thermal_guard_penalty": total_thermal_guard_penalty,
        "total_battery_reserve_penalty": total_battery_reserve_penalty,
        "total_downlink_burden_penalty": total_downlink_burden_penalty,
        "total_queue_pressure_risk_penalty": total_queue_pressure_risk_penalty,
        "terminal_state_penalty": terminal_state_penalty,
        "thermal_guard_exceed_count": thermal_guard_exceed_count,
        "thermal_090_exceed_count": thermal_090_exceed_count,
        "thermal_091_exceed_count": thermal_091_exceed_count,
        "battery_below_080_count": battery_below_080_count,
        "battery_below_077_count": battery_below_077_count,
        "battery_below_075_count": battery_below_075_count,
        "avg_mission_utility": total_mission_utility / max(len(records), 1),
        "utility_per_joule": utility_per_joule,
        "utility_per_downlink_bit": utility_per_downlink_bit,
        "safety_violation_count": safety_violation_count,
        "safety_violations": safety_violation_count,
        "raw_safety_violation_count": raw_safety_violation_count,
        "final_resource_safety_violation_count": final_resource_safety_violation_count,
        "safety_correction_count": safety_correction_count,
        "task_deferred_count": task_deferred_count,
        "deadline_missed_count": deadline_missed_count,
        "service_failure_count": service_failure_count,
        "downlink_throttle_count": downlink_throttle_count,
        "total_planned_downlink_mb_before_throttle": total_planned_downlink_mb_before_throttle,
        "total_actual_downlink_mb_after_throttle": total_actual_downlink_mb_after_throttle,
        "downlink_throttle_thermal_count": downlink_throttle_thermal_count,
        "feasible_episode_flag": feasible_episode_flag,
        "resource_feasible_flag": resource_feasible_flag,
        "deadline_met_tasks": deadline_met_tasks,
        "completed_tasks": completed_tasks,
        "deadline_satisfaction_ratio": deadline_met_tasks / max(total_generated_tasks, 1),
        "completed_deadline_satisfaction_ratio": deadline_met_completed_tasks / max(completed_tasks, 1),
        "deadline_satisfaction_rate": deadline_met_tasks / max(total_generated_tasks, 1),
        "queue_overflow_count": queue_overflow_count,
        "queue_overflow_ratio": queue_overflow_count / max(len(records), 1),
        "average_queue_mb": average_queue_mb,
        "avg_queue_pressure": avg_queue_pressure,
        "max_queue_mb": max((record.queue_next_mb for record in records), default=0.0),
        "total_observation_windows": total_observation_windows,
        "total_generated_tasks": total_generated_tasks,
        "active_observation_slot_ratio": active_observation_slot_count / max(len(records), 1),
        "idle_slot_ratio": idle_slot_count / max(len(records), 1),
        "routine_task_count": routine_task_count,
        "high_value_task_count": high_value_task_count,
        "emergency_task_count": emergency_task_count,
        "average_tasks_per_window": average_tasks_per_window,
        "delivered_items": delivered_items,
        "expired_items": expired_items,
        "dropped_items": dropped_items,
        "metadata_delivery_ratio": metadata_delivery_ratio,
        "high_value_metadata_delivery_ratio": high_value_metadata_delivery_ratio,
        "average_waiting_time": average_waiting_time,
        "delivered_high_value_payloads": delivered_high_value_payloads,
        "final_battery": final_battery,
        "final_battery_soc": final_battery,
        "min_battery_soc": min_battery_soc,
        "average_battery_soc": average_battery_soc,
        "max_thermal": max_thermal,
        "delivered_mb": delivered_mb,
        "total_energy": total_energy,
        "total_base_energy": total_base_energy,
        "total_spacecraft_energy": total_spacecraft_energy,
        "total_compute_energy": total_compute_energy,
        "total_comm_energy": total_comm_energy,
        "total_generated_payload_mb": total_generated_payload_mb,
        "total_downlinked_payload_mb": total_downlinked_payload_mb,
        "high_value_task_utility": high_value_task_utility,
        "emergency_task_utility": emergency_task_utility,
        "high_value_delivery_ratio": high_value_delivery_ratio,
        "payload_saving_ratio": payload_saving_ratio,
        "dominant_mode": dominant_mode,
        "mode_switch_count": mode_switch_count,
    }
