"""动作选择前的启发式预测器。"""

from __future__ import annotations

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
from src.reward.delivery import compute_delivery_factor
from src.reward.reward_function import compute_reward
from src.common.types import ObservedFeedback


class Predictor:
    """在动作执行前估计 reward 与安全后果。"""

    def __init__(
        self,
        model_registry: ModelRegistry,
        reward_cfg: dict[str, float],
        simulator_cfg: dict[str, float],
    ) -> None:
        self.model_registry = model_registry
        self.reward_cfg = reward_cfg
        self.simulator_cfg = simulator_cfg

    def predict_action(
        self,
        task: TaskSample,
        context: ContextVector,
        resource_trace: ResourceTrace,
        link_trace: LinkTrace,
        comm_state: CommState,
        action: JointAction,
    ) -> PredictedMetrics:
        """为单个动作估计执行后指标。"""

        slot_seconds = float(self.simulator_cfg["slot_seconds"])
        epsilon = float(self.simulator_cfg["epsilon"])
        queue_max = float(self.simulator_cfg["queue_max_mb"])
        comm_power = float(self.simulator_cfg["comm_power_w"])
        alpha_gamma = float(self.simulator_cfg["comm_alpha_gamma"])
        profile = self.model_registry.get_profile(action.model_name)
        policy_profile = get_payload_policy_profile(action.payload_policy)

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
            predicted_latency = profile.latency_s
            predicted_payload = policy_profile.base_size_mb * (0.85 + predicted_quality)
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

        if action.payload_policy == "metadata":
            predicted_payload *= 0.35
        elif action.payload_policy == "roi":
            predicted_payload *= 0.90
        elif action.payload_policy == "roi_context":
            predicted_payload *= 1.20

        capacity_mb = compute_available_capacity_mb(link_trace, slot_seconds)
        backlog = comm_state.queue_mb
        predicted_S_down_total = min(backlog + predicted_payload, capacity_mb)
        predicted_S_down_current = max(0.0, min(predicted_payload, capacity_mb - min(backlog, capacity_mb)))
        predicted_Q_next = max(0.0, backlog + predicted_payload - predicted_S_down_total)

        rate_mb_s = estimate_rate_mb_per_s(link_trace)
        predicted_L_queue = 0.0 if backlog <= 0.0 else slot_seconds * min(1.0, backlog / max(queue_max, epsilon))
        predicted_L_comm = 0.0 if rate_mb_s <= epsilon else predicted_S_down_current / rate_mb_s
        predicted_tx_time_total = 0.0 if rate_mb_s <= epsilon else predicted_S_down_total / rate_mb_s
        predicted_E_comp = compute_inference_energy_j(profile.estimated_power_w, predicted_latency)
        predicted_E_comm = compute_comm_energy_j(comm_power, predicted_tx_time_total, link_trace.link_quality, alpha_gamma)

        predicted_B_next = update_battery_soc(
            battery_soc=context.B_t,
            solar_energy_j=resource_trace.solar_energy_j,
            base_power_w=resource_trace.base_power_w,
            slot_seconds=slot_seconds,
            compute_energy_j=predicted_E_comp,
            comm_energy_j=predicted_E_comm,
            battery_capacity_j=float(self.simulator_cfg["battery_capacity_j"]),
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
            payload_information=policy_profile.information_level * max(predicted_quality, 0.10 if action.payload_policy == "defer" else 0.55),
            feedback=pseudo_feedback,
            reward_cfg=self.reward_cfg,
            simulator_cfg=self.simulator_cfg,
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
            predicted_reward=reward.reward,
            predicted_status=predicted_status,
            safety_reasons=safety_reasons,
            predicted_detection_quality=predicted_quality,
        )
