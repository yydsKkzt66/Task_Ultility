"""轨迹记录构建工具。"""

from __future__ import annotations

from src.common.types import ContextVector, JointAction, ObservedFeedback, PredictedMetrics, TaskSample, TrajectoryRecord


def build_trajectory_record(
    slot_index: int,
    task: TaskSample,
    context: ContextVector,
    action: JointAction,
    predicted: PredictedMetrics,
    feedback: ObservedFeedback,
    observed_reward: float,
) -> TrajectoryRecord:
    """把单步执行结果组装为轨迹 CSV 记录。"""

    return TrajectoryRecord(
        t=slot_index,
        task_id=task.task_id,
        selected_action=action.action_name,
        model_name=action.model_name,
        payload_policy=action.payload_policy,
        B_t=context.B_t,
        E_AI_t=context.E_t_AI,
        Theta_t=context.Theta_t,
        V_t=context.V_t,
        D_t=context.D_t,
        H_t=context.H_t,
        U_t=context.U_t,
        C_t=context.C_t,
        Q_t=context.Q_t,
        predicted_reward=predicted.predicted_reward,
        observed_reward=observed_reward,
        S_gen_current_mb=feedback.S_gen_current_mb,
        S_down_total_mb=feedback.S_down_total_mb,
        S_down_current_task_mb=feedback.S_down_current_task_mb,
        queue_next_mb=feedback.queue_next_mb,
        delta_queue_mb=feedback.delta_queue_mb,
        E_comp=feedback.E_comp,
        E_comm=feedback.E_comm,
        L_inf=feedback.L_inf,
        L_queue=feedback.L_queue,
        L_comm=feedback.L_comm,
        delivery_status=feedback.delivery_status,
        safety_violation=feedback.safety_violation,
    )
