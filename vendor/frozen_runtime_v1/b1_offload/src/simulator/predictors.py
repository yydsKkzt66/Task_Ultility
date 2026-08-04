"""动作选择前的启发式预测器。"""

from __future__ import annotations

import json

from src.common.constants import ACTION_FALLBACK_NAME
from src.common.types import CommState, ContextVector, JointAction, LinkTrace, PredictedMetrics, ResourceTrace, TaskSample
from src.energy.battery_update import update_ai_energy_budget, update_battery_soc
from src.energy.comm_energy import compute_comm_energy_j
from src.energy.compute_energy import compute_inference_energy_j
from src.energy.thermal_update import update_thermal_pressure
from src.link.contact_window import compute_available_capacity_mb
from src.link.link_model import estimate_rate_mb_per_s
from src.models.error_model import estimate_false_alarm_probability, estimate_miss_probability
from src.models.model_registry import ModelRegistry
from src.payload.payload_policy import get_payload_policy_profile
from src.payload.payload_sizer import (
    calibrated_profile_enabled,
    estimate_payload_encoding_cost,
    estimate_payload_size_with_diagnostics,
    estimate_raw_information_gain,
    resolve_payload_plan,
)
from src.reward.delivery import compute_delivery_factor
from src.reward.reward_function import compute_reward
from src.reward.utility_calibration import calibrate_value, get_utility_calibration
from src.data.value_skew_calibration import task_effective_value
from src.common.types import ObservedFeedback


class Predictor:
    """在动作执行前估计 reward 与安全后果。"""

    def __init__(
        self,
        model_registry: ModelRegistry,
        reward_cfg: dict[str, float],
        simulator_cfg: dict[str, float],
        payload_profiles_cfg: dict[str, object] | None = None,
    ) -> None:
        self.model_registry = model_registry
        self.reward_cfg = reward_cfg
        self.simulator_cfg = simulator_cfg
        self.payload_profiles_cfg = payload_profiles_cfg

    @staticmethod
    def summarize_guidance_bundle(guidance_bundle: dict[str, object] | None) -> str:
        if not guidance_bundle:
            return "disabled"
        compact = {
            "mode_name": guidance_bundle.get("mode_name", "unknown"),
            "lambda_E": round(float(guidance_bundle.get("lambda_E", 0.0)), 4),
            "lambda_L": round(float(guidance_bundle.get("lambda_L", 0.0)), 4),
            "lambda_Q": round(float(guidance_bundle.get("lambda_Q", 0.0)), 4),
            "lambda_S": round(float(guidance_bundle.get("lambda_S", 0.0)), 4),
            "exploration_scale": round(float(guidance_bundle.get("exploration_scale", 1.0)), 4),
            "model_complexity_bias": round(float(guidance_bundle.get("model_complexity_bias", 0.0)), 4),
            "payload_size_bias": round(float(guidance_bundle.get("payload_size_bias", 0.0)), 4),
            "high_value_bonus_scale": round(float(guidance_bundle.get("high_value_bonus_scale", 1.0)), 4),
            "safety_margin_scale": round(float(guidance_bundle.get("safety_margin_scale", 1.0)), 4),
        }
        return json.dumps(compact, ensure_ascii=False, sort_keys=True)

    def _delivery_aware_info_gain(
        self,
        raw_info_gain: float,
        predicted_payload_mb: float,
        predicted_delivery_factor: float,
        predicted_Q_next: float,
        predicted_L_queue: float,
        predicted_L_comm: float,
        task: TaskSample,
        context: ContextVector,
        link_trace: LinkTrace,
        guidance_bundle: dict[str, object] | None,
        payload_policy: str = "",
    ) -> tuple[float, float, float, str]:
        repair_cfg = self.simulator_cfg.get("policy_repair", {})
        if not isinstance(repair_cfg, dict):
            return raw_info_gain, 1.0, 1.0, "disabled"
        payload_specific_enabled = bool(repair_cfg.get("enable_payload_specific_delivery_proxy", False))
        if not bool(repair_cfg.get("enable_delivery_aware_info_gain", False)) and not payload_specific_enabled:
            return raw_info_gain, 1.0, 1.0, "disabled"
        if payload_policy == "defer":
            return 0.0, 0.0, 0.0, "deferred_no_immediate_delivery_utility"
        cfg = dict(repair_cfg.get("delivery_aware_info_gain", {}))
        resource_context = dict(guidance_bundle.get("resource_context", {})) if guidance_bundle else {}
        queue_pressure = predicted_Q_next / max(float(self.simulator_cfg["queue_max_mb"]), 1.0e-9)
        payload_pressure = predicted_payload_mb / max(float(cfg.get("payload_ref_mb", 1.6)), 1.0e-9)
        latency_to_deadline = (predicted_L_queue + predicted_L_comm) / max(task.deadline_s, float(self.simulator_cfg["epsilon"]))
        no_contact = 1.0 if not link_trace.in_contact else 0.0
        eclipse = 1.0 if bool(resource_context.get("is_eclipse", False)) else 0.0
        low_soc = max(0.0, float(cfg.get("soc_guard", 0.70)) - context.B_t)
        remaining_contact = link_trace.remaining_window_s if link_trace.in_contact else 0.0
        contact_shortfall = max(0.0, predicted_L_comm - remaining_contact) / max(task.deadline_s, 1.0e-9)
        payload_product_type, _payload_residual_product_type = resolve_payload_plan(payload_policy)
        payload_params = get_payload_policy_profile(payload_product_type, self.payload_profiles_cfg).parameters if payload_policy else {}
        risk_sensitivity = float(payload_params.get("delivery_risk_sensitivity", 1.0)) if payload_specific_enabled else 1.0
        queue_sensitivity = float(payload_params.get("queue_sensitivity", 1.0)) if payload_specific_enabled else 1.0
        deadline_sensitivity = float(payload_params.get("deadline_sensitivity", 1.0)) if payload_specific_enabled else 1.0
        contact_sensitivity = float(payload_params.get("contact_sensitivity", 1.0)) if payload_specific_enabled else 1.0
        risk = risk_sensitivity * (
            float(cfg.get("queue_weight", 0.35)) * queue_sensitivity * queue_pressure
            + float(cfg.get("payload_weight", 0.45)) * payload_pressure
            + float(cfg.get("deadline_weight", 0.45)) * deadline_sensitivity * latency_to_deadline
            + float(cfg.get("no_contact_weight", 0.35)) * contact_sensitivity * no_contact
            + float(cfg.get("contact_shortfall_weight", 0.25)) * contact_sensitivity * contact_shortfall
            + float(cfg.get("eclipse_weight", 0.10)) * eclipse
            + float(cfg.get("low_soc_weight", 0.15)) * low_soc
        )
        proxy = predicted_delivery_factor / (1.0 + max(0.0, risk))
        min_proxy = float(payload_params.get("min_delivery_proxy", cfg.get("min_proxy", 0.20))) if payload_specific_enabled else float(cfg.get("min_proxy", 0.20))
        metadata_floor = float(cfg.get("metadata_floor", 0.70))
        if payload_policy == "metadata" or raw_info_gain <= float(cfg.get("metadata_info_threshold", 0.35)):
            proxy = max(proxy, metadata_floor)
        proxy_raw = min(1.0, proxy)
        proxy = max(min_proxy, proxy_raw)
        reasons = []
        if no_contact:
            reasons.append("no_contact")
        if queue_pressure > float(cfg.get("queue_reason_threshold", 0.20)):
            reasons.append("queue_pressure")
        if payload_pressure > float(cfg.get("payload_reason_threshold", 0.80)):
            reasons.append("large_payload")
        if latency_to_deadline > float(cfg.get("deadline_reason_threshold", 0.50)):
            reasons.append("deadline_risk")
        if eclipse:
            reasons.append("eclipse")
        return raw_info_gain * proxy, proxy_raw, min(1.0, proxy), ";".join(reasons) if reasons else "low_delivery_risk"

    def _compute_guided_predicted_reward(
        self,
        task: TaskSample,
        action: JointAction,
        policy_profile,
        predicted_quality: float,
        predicted_payload: float,
        predicted_delivery_factor: float,
        predicted_E_comp: float,
        predicted_E_comm: float,
        predicted_latency: float,
        predicted_L_queue: float,
        predicted_L_comm: float,
        predicted_Q_next: float,
        predicted_B_next: float,
        predicted_Theta_next: float,
        guidance_bundle: dict[str, object] | None,
        effective_info_gain: float | None = None,
    ) -> float:
        if not guidance_bundle:
            return float("nan")

        lambda_E = float(guidance_bundle.get("lambda_E", 0.0))
        lambda_L = float(guidance_bundle.get("lambda_L", 0.0))
        lambda_Q = float(guidance_bundle.get("lambda_Q", 0.0))
        lambda_S = float(guidance_bundle.get("lambda_S", 0.0))
        model_complexity_bias = float(guidance_bundle.get("model_complexity_bias", 0.0))
        payload_size_bias = float(guidance_bundle.get("payload_size_bias", 0.0))
        high_value_bonus_scale = float(guidance_bundle.get("high_value_bonus_scale", 1.0))
        safety_margin_scale = float(guidance_bundle.get("safety_margin_scale", 1.0))
        calibration = get_utility_calibration(self.reward_cfg)
        task_value = calibrate_value(task_effective_value(task), calibration)

        info_factor = (
            effective_info_gain
            if effective_info_gain is not None
            else policy_profile.information_level * max(predicted_quality, 0.10 if action.payload_policy == "defer" else 0.55)
        )
        effective_utility = task_value * predicted_quality * info_factor * predicted_delivery_factor
        if task_value >= float(self.reward_cfg.get("high_value_threshold", 0.75)):
            effective_utility *= high_value_bonus_scale

        epsilon = float(self.simulator_cfg["epsilon"])
        energy_term = (predicted_E_comp + predicted_E_comm) / max(float(self.simulator_cfg["E_ref"]), epsilon)
        latency_term = (predicted_latency + predicted_L_queue + predicted_L_comm) / max(task.deadline_s, epsilon)
        queue_term = predicted_Q_next / max(float(self.simulator_cfg["queue_max_mb"]), epsilon)
        payload_term = predicted_payload / max(float(self.simulator_cfg["queue_max_mb"]), epsilon)
        resource_cost = (
            lambda_E * energy_term
            + lambda_L * latency_term
            + lambda_Q * queue_term
            + lambda_S * payload_term
        )

        battery_margin = max(0.0, float(self.simulator_cfg["battery_min"]) - predicted_B_next)
        thermal_margin = max(0.0, predicted_Theta_next - float(self.simulator_cfg["thermal_max"]))
        queue_margin = max(0.0, predicted_Q_next - float(self.simulator_cfg["queue_max_mb"])) / max(
            float(self.simulator_cfg["queue_max_mb"]),
            epsilon,
        )
        safety_margin_penalty = safety_margin_scale * (battery_margin + thermal_margin + queue_margin)

        bias_term = (
            model_complexity_bias * action.expected_compute_level
            + payload_size_bias * action.expected_payload_level
        )
        return (
            effective_utility
            - calibration.beta_resource_cost * resource_cost
            + bias_term
            - calibration.beta_hard_penalty * safety_margin_penalty
        )

    def predict_action(
        self,
        task: TaskSample,
        context: ContextVector,
        resource_trace: ResourceTrace,
        link_trace: LinkTrace,
        comm_state: CommState,
        action: JointAction,
        guidance_bundle: dict[str, object] | None = None,
    ) -> PredictedMetrics:
        """为单个动作估计执行后指标。"""

        slot_seconds = float(self.simulator_cfg["slot_seconds"])
        epsilon = float(self.simulator_cfg["epsilon"])
        queue_max = float(self.simulator_cfg["queue_max_mb"])
        comm_power = float(self.simulator_cfg.get("tx_power_w", self.simulator_cfg["comm_power_w"]))
        alpha_gamma = float(self.simulator_cfg["comm_alpha_gamma"])
        downlink_efficiency = float(self.simulator_cfg.get("downlink_efficiency", 1.0))
        profile = self.model_registry.get_profile(action.model_name)
        product_type, residual_product_type = resolve_payload_plan(action.payload_policy)
        policy_profile = get_payload_policy_profile(product_type, self.payload_profiles_cfg)

        if action.action_name == ACTION_FALLBACK_NAME:
            predicted_quality = 0.0
            predicted_latency = 0.0
            predicted_payload = 0.0
            predicted_miss_probability = 0.0
            predicted_false_alarm_probability = 0.0
        else:
            predicted_quality = max(
                0.05,
                min(1.0, profile.quality_base - (1.0 - profile.robustness) * 0.32 * task.difficulty - 0.16 * task.uncertainty),
            )
            predicted_payload, _payload_diag = estimate_payload_size_with_diagnostics(
                action.payload_policy,
                task=task,
                payload_profiles_cfg=self.payload_profiles_cfg,
            )
            predicted_payload *= 0.85 + predicted_quality
            predicted_latency = profile.latency_s
            predicted_miss_probability = estimate_miss_probability(
                task=task,
                profile=profile,
                estimated_quality=predicted_quality,
                error_model_cfg=self.model_registry.error_model_cfg,
            )
            predicted_false_alarm_probability = estimate_false_alarm_probability(
                task=task,
                estimated_quality=predicted_quality,
                miss_probability=predicted_miss_probability,
                error_model_cfg=self.model_registry.error_model_cfg,
            )

        if not calibrated_profile_enabled(self.payload_profiles_cfg):
            if action.payload_policy == "metadata":
                predicted_payload *= 0.35
            elif action.payload_policy == "roi":
                predicted_payload *= 0.90
            elif action.payload_policy == "roi_context":
                predicted_payload *= 1.20

        encoding_latency_s, encoding_energy_j = estimate_payload_encoding_cost(
            action.payload_policy,
            predicted_payload,
            self.payload_profiles_cfg,
        )
        predicted_latency += encoding_latency_s

        capacity_mb = compute_available_capacity_mb(link_trace, slot_seconds, downlink_efficiency)
        backlog = comm_state.queue_mb
        predicted_S_down_total = min(backlog + predicted_payload, capacity_mb)
        predicted_S_down_current = max(0.0, min(predicted_payload, capacity_mb - min(backlog, capacity_mb)))
        predicted_Q_next = max(0.0, backlog + predicted_payload - predicted_S_down_total)

        rate_mb_s = estimate_rate_mb_per_s(link_trace, downlink_efficiency)
        predicted_L_queue = 0.0 if backlog <= 0.0 else slot_seconds * min(1.0, backlog / max(queue_max, epsilon))
        predicted_L_comm = 0.0 if rate_mb_s <= epsilon else predicted_S_down_current / rate_mb_s
        predicted_tx_time_total = 0.0 if rate_mb_s <= epsilon else predicted_S_down_total / rate_mb_s
        predicted_E_comp = compute_inference_energy_j(profile.estimated_power_w, predicted_latency) + encoding_energy_j
        predicted_E_comm = compute_comm_energy_j(comm_power, predicted_tx_time_total, link_trace.link_quality, alpha_gamma)

        predicted_B_next = update_battery_soc(
            battery_soc=context.B_t,
            solar_energy_j=resource_trace.solar_energy_j,
            base_power_w=resource_trace.base_power_w,
            slot_seconds=slot_seconds,
            compute_energy_j=predicted_E_comp,
            comm_energy_j=predicted_E_comm,
            battery_capacity_j=float(self.simulator_cfg["battery_capacity_j"]),
            charge_efficiency=float(self.simulator_cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(self.simulator_cfg.get("battery_discharge_efficiency", 1.0)),
        )
        predicted_E_AI_next = update_ai_energy_budget(
            ai_energy_budget=context.E_t_AI,
            compute_energy_j=predicted_E_comp,
            E_AI_ref=float(self.simulator_cfg["E_AI_ref"]),
            recharge=float(resource_trace.solar_energy_j / float(self.simulator_cfg["E_AI_ref"]) * 0.05),
        )
        predicted_Theta_next = update_thermal_pressure(
            thermal_pressure=context.Theta_t,
            compute_energy_j=predicted_E_comp,
            comm_energy_j=predicted_E_comm,
            cooling=float(self.simulator_cfg["cooling_per_slot"]),
            compute_heat_gain_scale=profile.thermal_gain_scale,
            slot_seconds=slot_seconds,
            reference_slot_seconds=float(self.simulator_cfg.get("thermal_reference_slot_seconds", 60.0)),
            temperature_reference_c=resource_trace.temperature_reference_c,
            is_sunlit=resource_trace.is_sunlit,
            passive_cooling=float(self.simulator_cfg.get("thermal_passive_cooling", 0.12)),
            sunlight_gain=float(self.simulator_cfg.get("thermal_sunlight_gain", 0.015)),
            eclipse_cooling_bonus=float(self.simulator_cfg.get("thermal_eclipse_cooling_bonus", 0.03)),
            reference_min_c=float(self.simulator_cfg.get("thermal_reference_min_c", -10.0)),
            reference_max_c=float(self.simulator_cfg.get("thermal_reference_max_c", 30.0)),
        )

        if action.payload_policy == "defer":
            predicted_status = "deferred"
        elif predicted_payload <= 0.0:
            predicted_status = "empty"
        elif predicted_S_down_current >= predicted_payload - epsilon:
            predicted_status = "delivered"
        else:
            predicted_status = "queued"

        predicted_delivery_factor = compute_delivery_factor(
            predicted_status,
            delivery_beta=float(self.simulator_cfg["delivery_beta"]),
            deferred_credit=float(self.reward_cfg["deferred_delivery_credit"]),
        )
        raw_profile_gain = estimate_raw_information_gain(action.payload_policy, self.payload_profiles_cfg)
        raw_info_gain = raw_profile_gain * max(predicted_quality, 0.10 if action.payload_policy == "defer" else 0.55)
        effective_info_gain, delivery_proxy_raw, delivery_success_proxy, payload_delivery_risk_reason = self._delivery_aware_info_gain(
            raw_info_gain=raw_info_gain,
            predicted_payload_mb=predicted_payload,
            predicted_delivery_factor=predicted_delivery_factor,
            predicted_Q_next=predicted_Q_next,
            predicted_L_queue=predicted_L_queue,
            predicted_L_comm=predicted_L_comm,
            task=task,
            context=context,
            link_trace=link_trace,
            guidance_bundle=guidance_bundle,
            payload_policy=action.payload_policy,
        )

        safety_reasons: list[str] = []
        if predicted_B_next < float(self.simulator_cfg["battery_min"]):
            safety_reasons.append("predicted battery too low")
        if predicted_Theta_next > float(self.simulator_cfg["thermal_max"]):
            safety_reasons.append("predicted thermal too high")
        if predicted_Q_next > queue_max:
            safety_reasons.append("predicted queue overflow")

        pseudo_feedback = ObservedFeedback(
            S_gen_current_mb=predicted_payload,
            S_down_total_mb=predicted_S_down_total,
            S_down_current_task_mb=predicted_S_down_current,
            queue_next_mb=predicted_Q_next,
            delta_queue_mb=predicted_Q_next - backlog,
            E_comp=predicted_E_comp,
            E_comm=predicted_E_comm,
            L_inf=predicted_latency,
            L_queue=predicted_L_queue,
            L_comm=predicted_L_comm,
            delivery_status=predicted_status,
            delivery_factor=predicted_delivery_factor,
            safety_violation=int(bool(safety_reasons)),
            total_latency_s=predicted_latency + predicted_L_queue + predicted_L_comm,
            dropped_current_task_mb=0.0,
            queue_item_count=int(round(predicted_Q_next / max(policy_profile.base_size_mb + 0.01, 0.01))) if predicted_Q_next > 0 else 0,
            miss_signal=predicted_miss_probability,
            false_alarm_signal=predicted_false_alarm_probability,
            detection_quality=predicted_quality,
        )
        reward = compute_reward(
            task=task,
            context=context,
            payload_information=effective_info_gain,
            feedback=pseudo_feedback,
            reward_cfg=self.reward_cfg,
            simulator_cfg=self.simulator_cfg,
        )
        guided_predicted_reward = self._compute_guided_predicted_reward(
            task=task,
            action=action,
            policy_profile=policy_profile,
            predicted_quality=predicted_quality,
            predicted_payload=predicted_payload,
            predicted_delivery_factor=predicted_delivery_factor,
            predicted_E_comp=predicted_E_comp,
            predicted_E_comm=predicted_E_comm,
            predicted_latency=predicted_latency,
            predicted_L_queue=predicted_L_queue,
            predicted_L_comm=predicted_L_comm,
            predicted_Q_next=predicted_Q_next,
            predicted_B_next=predicted_B_next,
            predicted_Theta_next=predicted_Theta_next,
            guidance_bundle=guidance_bundle,
            effective_info_gain=effective_info_gain,
        )
        return PredictedMetrics(
            predicted_S_gen_mb=predicted_payload,
            predicted_S_down_mb=predicted_S_down_current,
            predicted_E_comp=predicted_E_comp,
            predicted_E_comm=predicted_E_comm,
            predicted_L_inf=predicted_latency,
            predicted_L_queue=predicted_L_queue,
            predicted_L_comm=predicted_L_comm,
            predicted_Q_next_mb=predicted_Q_next,
            predicted_B_next=predicted_B_next,
            predicted_E_AI_next=predicted_E_AI_next,
            predicted_Theta_next=predicted_Theta_next,
            predicted_delivery_factor=predicted_delivery_factor,
            predicted_safety_violation=float(bool(safety_reasons)),
            predicted_reward=reward.reward if guidance_bundle is None else guided_predicted_reward,
            predicted_status=predicted_status,
            safety_reasons=safety_reasons,
            predicted_detection_quality=predicted_quality,
            raw_info_gain=raw_info_gain,
            delivery_proxy_raw=delivery_proxy_raw,
            delivery_success_proxy=delivery_success_proxy,
            effective_info_gain=effective_info_gain,
            payload_delivery_risk_reason=payload_delivery_risk_reason,
            predicted_payload_plan=action.payload_plan or action.payload_policy,
            predicted_product_type=product_type,
            predicted_residual_product_type=residual_product_type,
            predicted_incremental_information_utility=effective_info_gain,
        )
