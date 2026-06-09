from src.common.types import ContextVector, ObservedFeedback, TaskSample
from src.reward.reward_function import compute_reward


def test_reward_is_computable():
    task = TaskSample("task_01", "a", "b", "scene", 0, 0.8, 50.0, 0.4, 0.4)
    context = ContextVector(0.8, 0.9, 0.2, 0.8, 0.2, 0.4, 0.4, 0.6, 0.1)
    reward_cfg = {
        "lambda_L": 0.2,
        "lambda_Q": 0.1,
        "lambda_Omega": 2.0,
        "lambda_E_base": 0.3,
        "eta_B": 1.0,
        "eta_Theta": 0.5,
        "info_base": 0.25,
        "info_value_weight": 0.35,
        "info_difficulty_weight": 0.20,
        "info_uncertainty_weight": 0.25,
        "info_urgency_weight": 0.20,
        "deferred_delivery_credit": 0.15,
    }
    simulator_cfg = {
        "delivery_beta": 0.5,
        "E_ref": 100.0,
        "epsilon": 1.0e-9,
        "queue_max_mb": 64.0,
    }
    feedback = ObservedFeedback(
        S_gen_current_mb=1.0,
        S_down_total_mb=1.0,
        S_down_current_task_mb=1.0,
        queue_next_mb=0.2,
        delta_queue_mb=0.1,
        E_comp=8.0,
        E_comm=1.0,
        L_inf=1.5,
        L_queue=0.2,
        L_comm=0.5,
        delivery_status="delivered",
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=2.2,
        dropped_current_task_mb=0.0,
        queue_item_count=1,
        detection_quality=0.9,
    )
    result = compute_reward(task, context, 0.9, feedback, reward_cfg, simulator_cfg)
    assert isinstance(result.reward, float)


def test_reward_increases_with_detection_quality():
    task = TaskSample("task_02", "a", "b", "scene", 0, 0.9, 60.0, 0.3, 0.3)
    context = ContextVector(0.8, 0.9, 0.2, 0.9, 0.2, 0.3, 0.3, 0.7, 0.1)
    reward_cfg = {
        "lambda_L": 0.0,
        "lambda_Q": 0.0,
        "lambda_Omega": 0.0,
        "lambda_E_base": 0.0,
        "eta_B": 1.0,
        "eta_Theta": 0.5,
        "info_base": 0.25,
        "info_value_weight": 0.35,
        "info_difficulty_weight": 0.20,
        "info_uncertainty_weight": 0.25,
        "info_urgency_weight": 0.20,
        "deferred_delivery_credit": 0.15,
    }
    simulator_cfg = {
        "delivery_beta": 0.5,
        "E_ref": 100.0,
        "epsilon": 1.0e-9,
        "queue_max_mb": 64.0,
    }

    low_quality = ObservedFeedback(
        S_gen_current_mb=1.0,
        S_down_total_mb=1.0,
        S_down_current_task_mb=1.0,
        queue_next_mb=0.0,
        delta_queue_mb=0.0,
        E_comp=0.0,
        E_comm=0.0,
        L_inf=1.0,
        L_queue=0.0,
        L_comm=0.0,
        delivery_status="delivered",
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=1.0,
        dropped_current_task_mb=0.0,
        queue_item_count=0,
        detection_quality=0.2,
    )
    high_quality = ObservedFeedback(
        S_gen_current_mb=1.0,
        S_down_total_mb=1.0,
        S_down_current_task_mb=1.0,
        queue_next_mb=0.0,
        delta_queue_mb=0.0,
        E_comp=0.0,
        E_comm=0.0,
        L_inf=1.0,
        L_queue=0.0,
        L_comm=0.0,
        delivery_status="delivered",
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=1.0,
        dropped_current_task_mb=0.0,
        queue_item_count=0,
        detection_quality=0.9,
    )

    low_result = compute_reward(task, context, 0.95, low_quality, reward_cfg, simulator_cfg)
    high_result = compute_reward(task, context, 0.95, high_quality, reward_cfg, simulator_cfg)
    assert high_result.reward > low_result.reward
