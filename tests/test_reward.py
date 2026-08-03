import pytest

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


def test_requirement_gap_penalties_apply_when_quality_and_information_are_insufficient():
    task = TaskSample("task_03", "a", "b", "scene", 0, 0.9, 30.0, 0.7, 0.7)
    context = ContextVector(0.8, 0.9, 0.2, 0.9, 0.7, 0.7, 0.7, 0.6, 0.1)
    reward_cfg = {
        "energy_penalty": 0.0,
        "latency_penalty": 0.0,
        "queue_penalty": 0.0,
        "violation_penalty": 0.0,
        "delivery_weight": 1.0,
        "information_weight": 1.0,
        "quality_base_requirement": 0.40,
        "quality_value_weight": 0.18,
        "quality_difficulty_weight": 0.22,
        "quality_uncertainty_weight": 0.16,
        "quality_urgency_weight": 0.12,
        "quality_gap_penalty": 1.0,
        "information_gap_penalty": 1.0,
        "high_value_bonus": 0.0,
        "high_value_threshold": 0.75,
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
    feedback = ObservedFeedback(
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
    result = compute_reward(task, context, 0.2, feedback, reward_cfg, simulator_cfg)
    assert result.quality_gap > 0.0
    assert result.information_gap > 0.0
    assert result.quality_gap_penalty > 0.0
    assert result.information_gap_penalty > 0.0


def test_reward_breakdown_keeps_mission_effective_and_penalized_terms_separate():
    task = TaskSample("task_04", "a", "b", "scene", 0, 0.8, 40.0, 0.6, 0.6)
    context = ContextVector(0.8, 0.9, 0.2, 0.8, 0.3, 0.6, 0.6, 0.7, 0.2)
    reward_cfg = {
        "lambda_L": 0.2,
        "lambda_Q": 0.1,
        "lambda_Omega": 2.0,
        "lambda_E_base": 0.3,
        "eta_B": 1.0,
        "eta_Theta": 0.5,
        "quality_gap_penalty": 0.6,
        "information_gap_penalty": 0.35,
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
        safety_violation=1,
        total_latency_s=2.2,
        dropped_current_task_mb=0.0,
        queue_item_count=1,
        detection_quality=0.7,
    )
    result = compute_reward(task, context, 0.5, feedback, reward_cfg, simulator_cfg)
    expected_penalized = (
        result.effective_utility
        - result.resource_penalty
        - result.requirement_penalty
        - result.safety_penalty
        - result.mission_error_penalty
    )

    assert result.mission_utility == context.V_t
    assert result.reward == expected_penalized
    assert result.mission_utility != result.reward


def test_value_cost_calibration_disabled_is_neutral_even_with_non_neutral_values():
    task = TaskSample("task_05", "a", "b", "scene", 0, 0.5, 40.0, 0.0, 0.0)
    context = ContextVector(1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0)
    base_reward_cfg = {
        "lambda_L": 0.0,
        "lambda_Q": 0.0,
        "lambda_Omega": 0.0,
        "lambda_E_base": 1.0,
        "eta_B": 0.0,
        "eta_Theta": 0.0,
        "info_base": 1.0,
        "info_value_weight": 0.0,
        "info_difficulty_weight": 0.0,
        "info_uncertainty_weight": 0.0,
        "info_urgency_weight": 0.0,
        "deferred_delivery_credit": 0.15,
    }
    disabled_reward_cfg = {
        **base_reward_cfg,
        "utility_calibration": {
            "enable_value_cost_calibration": False,
            "value_scale": {"a": 9.0, "b": 0.4, "clip_min": 0.05, "clip_max": 0.95},
            "cost_penalty_scale": {
                "beta_resource_cost": 0.2,
                "beta_requirement_penalty": 0.2,
                "beta_hard_penalty": 0.2,
            },
        },
        "lifecycle_utility": {"enable_lifecycle_utility": False, "reserved_only": True},
    }
    simulator_cfg = {"delivery_beta": 0.5, "E_ref": 100.0, "epsilon": 1.0e-9, "queue_max_mb": 64.0}
    feedback = ObservedFeedback(
        S_gen_current_mb=1.0,
        S_down_total_mb=1.0,
        S_down_current_task_mb=1.0,
        queue_next_mb=0.0,
        delta_queue_mb=0.0,
        E_comp=10.0,
        E_comm=0.0,
        L_inf=0.0,
        L_queue=0.0,
        L_comm=0.0,
        delivery_status="delivered",
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=0.0,
        dropped_current_task_mb=0.0,
        queue_item_count=0,
        detection_quality=1.0,
    )

    baseline = compute_reward(task, context, 1.0, feedback, base_reward_cfg, simulator_cfg)
    disabled = compute_reward(task, context, 1.0, feedback, disabled_reward_cfg, simulator_cfg)

    assert disabled.reward == baseline.reward
    assert disabled.value_raw == context.V_t
    assert disabled.value_calibrated == context.V_t
    assert not disabled.lifecycle_fields_active


def test_value_cost_calibration_enabled_applies_affine_value_and_betas():
    task = TaskSample(
        "task_06",
        "a",
        "b",
        "scene",
        0,
        0.5,
        40.0,
        0.0,
        0.0,
        quality_requirement=0.0,
        information_requirement=0.0,
    )
    context = ContextVector(1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0)
    reward_cfg = {
        "lambda_L": 0.0,
        "lambda_Q": 0.0,
        "lambda_Omega": 0.0,
        "lambda_E_base": 1.0,
        "eta_B": 0.0,
        "eta_Theta": 0.0,
        "info_base": 1.0,
        "info_value_weight": 0.0,
        "info_difficulty_weight": 0.0,
        "info_uncertainty_weight": 0.0,
        "info_urgency_weight": 0.0,
        "deferred_delivery_credit": 0.15,
        "utility_calibration": {
            "enable_value_cost_calibration": True,
            "value_scale": {"a": 1.10, "b": 0.03, "clip_min": 0.05, "clip_max": 0.95},
            "cost_penalty_scale": {
                "beta_resource_cost": 0.65,
                "beta_requirement_penalty": 0.50,
                "beta_hard_penalty": 1.00,
            },
        },
        "lifecycle_utility": {"enable_lifecycle_utility": False, "reserved_only": True},
    }
    simulator_cfg = {"delivery_beta": 0.5, "E_ref": 100.0, "epsilon": 1.0e-9, "queue_max_mb": 64.0}
    feedback = ObservedFeedback(
        S_gen_current_mb=1.0,
        S_down_total_mb=1.0,
        S_down_current_task_mb=1.0,
        queue_next_mb=0.0,
        delta_queue_mb=0.0,
        E_comp=10.0,
        E_comm=0.0,
        L_inf=0.0,
        L_queue=0.0,
        L_comm=0.0,
        delivery_status="delivered",
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=0.0,
        dropped_current_task_mb=0.0,
        queue_item_count=0,
        detection_quality=1.0,
    )

    result = compute_reward(task, context, 1.0, feedback, reward_cfg, simulator_cfg)

    assert result.value_calibrated == pytest.approx(0.58)
    assert result.effective_utility == pytest.approx(0.58)
    assert result.resource_penalty == pytest.approx(0.065)
    assert result.requirement_penalty == pytest.approx(0.0)
    assert result.hard_penalty == pytest.approx(0.0)
    assert result.reward == pytest.approx(0.515)
    assert result.beta_hard_penalty == pytest.approx(1.0)


def test_lifecycle_utility_enable_is_reserved_not_implemented():
    task = TaskSample("task_07", "a", "b", "scene", 0, 0.5, 40.0, 0.0, 0.0)
    context = ContextVector(1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0)
    reward_cfg = {
        "lambda_E_base": 0.0,
        "eta_B": 0.0,
        "eta_Theta": 0.0,
        "info_base": 1.0,
        "info_value_weight": 0.0,
        "info_difficulty_weight": 0.0,
        "info_uncertainty_weight": 0.0,
        "info_urgency_weight": 0.0,
        "deferred_delivery_credit": 0.15,
        "lifecycle_utility": {"enable_lifecycle_utility": True, "reserved_only": True},
    }
    simulator_cfg = {"delivery_beta": 0.5, "E_ref": 100.0, "epsilon": 1.0e-9, "queue_max_mb": 64.0}
    feedback = ObservedFeedback(
        S_gen_current_mb=0.0,
        S_down_total_mb=0.0,
        S_down_current_task_mb=0.0,
        queue_next_mb=0.0,
        delta_queue_mb=0.0,
        E_comp=0.0,
        E_comm=0.0,
        L_inf=0.0,
        L_queue=0.0,
        L_comm=0.0,
        delivery_status="delivered",
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=0.0,
        dropped_current_task_mb=0.0,
        queue_item_count=0,
        detection_quality=1.0,
    )

    with pytest.raises(NotImplementedError):
        compute_reward(task, context, 1.0, feedback, reward_cfg, simulator_cfg)
