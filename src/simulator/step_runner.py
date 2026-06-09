"""单步闭环执行器。"""

from __future__ import annotations

from src.common.types import CommState, MissionState, ObservedFeedback, ResourceState, TimelineStep
from src.context.context_builder import ContextBuilder
from src.energy.battery_update import update_ai_energy_budget, update_battery_soc
from src.energy.comm_energy import compute_comm_energy_j
from src.energy.compute_energy import compute_inference_energy_j
from src.energy.thermal_update import update_thermal_pressure
from src.link.downlink_queue import DownlinkQueue
from src.link.link_model import estimate_rate_mb_per_s
from src.models.inference_runner import InferenceRunner
from src.models.model_registry import ModelRegistry
from src.payload.payload_sizer import generate_payload
from src.reward.delivery import compute_delivery_factor
from src.reward.reward_function import compute_reward
from src.routers.base_router import BaseRouter
from src.routers.safe_filter import SafeFilter
from src.simulator.feedback import build_trajectory_record
from src.simulator.predictors import Predictor
from src.actions.action_space import ActionSpace


class StepRunner:
    """严格遵循固定主流程执行单步闭环。"""

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
        self.last_feedback = None
        self.last_reward_breakdown = None
        self.last_detection_result = None

    def _compute_actual_feedback(
        self,
        timeline_step: TimelineStep,
        comm_state: CommState,
        action,
        detection_result,
        downlink_result,
    ) -> tuple[ObservedFeedback, float, float, float]:
        """根据真实执行结果计算 observed feedback。"""

        profile = self.model_registry.get_profile(action.model_name)
        E_comp = 0.0 if action.model_name == "delay_store" else compute_inference_energy_j(profile.estimated_power_w, detection_result.inference_latency_s)
        L_inf = detection_result.inference_latency_s
        rate_mb_s = estimate_rate_mb_per_s(timeline_step.link_trace)
        epsilon = float(self.simulator_cfg["epsilon"])
        L_comm = 0.0 if rate_mb_s <= epsilon else downlink_result.S_down_current_task_mb / rate_mb_s
        tx_time_total = 0.0 if rate_mb_s <= epsilon else downlink_result.S_down_total_mb / rate_mb_s
        E_comm = compute_comm_energy_j(
            comm_power_w=float(self.simulator_cfg["comm_power_w"]),
            tx_time_s=tx_time_total,
            link_quality=timeline_step.link_trace.link_quality,
            alpha_gamma=float(self.simulator_cfg["comm_alpha_gamma"]),
        )
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
            miss_signal=1.0 if detection_result.miss_flag else 0.0,
            false_alarm_signal=1.0 if detection_result.false_alarm_flag else 0.0,
            detection_quality=detection_result.estimated_quality,
        )
        return feedback, E_comp, E_comm, total_latency

    def run_step(
        self,
        timeline_step: TimelineStep,
        resource_state: ResourceState,
        mission_state: MissionState,
        comm_state: CommState,
        router: BaseRouter,
    ):
        """执行一个完整时隙。

        固定顺序：
        TimelineStep -> Context -> Candidate Actions -> Predictor -> SafeFilter
        -> Router -> Detector -> Payload -> Contact-Aware Downlink -> Feedback
        -> Reward -> State Update -> Router Update
        """

        context = self.context_builder.build(timeline_step.task, resource_state, comm_state, timeline_step.link_trace)
        actions = self.action_space.list_actions()
        predicted_map = {
            action.action_name: self.predictor.predict_action(
                task=timeline_step.task,
                context=context,
                resource_trace=timeline_step.resource_trace,
                link_trace=timeline_step.link_trace,
                comm_state=comm_state,
                action=action,
            )
            for action in actions
        }
        safe_actions, _unsafe_reasons = self.safe_filter.filter_actions(timeline_step.task, actions, predicted_map)
        selected_action = router.select_action(context, safe_actions, predicted_map)

        detection_result = self.inference_runner.run(timeline_step.task, selected_action)
        payload = generate_payload(
            task=timeline_step.task,
            detection_result=detection_result,
            payload_policy=selected_action.payload_policy,
            slot_index=timeline_step.slot_index,
        )
        downlink_result = self.downlink_queue.process_slot(
            slot_index=timeline_step.slot_index,
            payload=payload,
            link_trace=timeline_step.link_trace,
        )
        feedback, E_comp, E_comm, total_latency = self._compute_actual_feedback(
            timeline_step, comm_state, selected_action, detection_result, downlink_result
        )

        next_battery = update_battery_soc(
            battery_soc=resource_state.battery_soc,
            solar_energy_j=timeline_step.resource_trace.solar_energy_j,
            base_power_w=timeline_step.resource_trace.base_power_w,
            slot_seconds=timeline_step.slot_seconds,
            compute_energy_j=E_comp,
            comm_energy_j=E_comm,
            battery_capacity_j=float(self.simulator_cfg["battery_capacity_j"]),
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
        )

        safety_violation = int(
            next_battery < float(self.simulator_cfg["battery_min"])
            or next_thermal > float(self.simulator_cfg["thermal_max"])
            or downlink_result.queue_next_mb > float(self.simulator_cfg["queue_max_mb"])
            or total_latency > timeline_step.task.deadline_s
        )
        feedback.safety_violation = safety_violation

        reward = compute_reward(
            task=timeline_step.task,
            context=context,
            payload_information=payload.information_content,
            feedback=feedback,
            reward_cfg=self.reward_cfg,
            simulator_cfg=self.simulator_cfg,
        )
        router.update(context, selected_action, reward.reward)

        resource_state.battery_soc = next_battery
        resource_state.ai_energy_budget = next_ai_budget
        resource_state.thermal_pressure = next_thermal
        comm_state.queue_mb = downlink_result.queue_next_mb
        comm_state.queue_item_count = downlink_result.queue_item_count
        comm_state.effective_opportunity = context.C_t
        mission_state.slot_index = timeline_step.slot_index + 1
        mission_state.completed_tasks += int(feedback.delivery_status == "delivered")
        mission_state.safety_violations += safety_violation

        self.last_feedback = feedback
        self.last_reward_breakdown = reward
        self.last_detection_result = detection_result

        record = build_trajectory_record(
            slot_index=timeline_step.slot_index,
            task=timeline_step.task,
            context=context,
            action=selected_action,
            predicted=predicted_map[selected_action.action_name],
            feedback=feedback,
            observed_reward=reward.reward,
        )
        return record, reward
