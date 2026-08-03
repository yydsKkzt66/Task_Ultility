from src.actions.action_space import ActionSpace
from src.common.types import ContextVector, ObservedFeedback, TaskSample
from src.models.model_registry import ModelRegistry
from src.payload.payload_policy import get_payload_policy_profile
from src.reward.reward_function import compute_reward


def test_model_profiles_can_be_changed_via_config():
    model_pool_cfg = {
        "profiles": [
            {
                "name": "tiny",
                "energy_j": 12.0,
                "latency_s": 3.0,
                "quality_base": 0.61,
                "robustness_to_difficulty": 0.77,
                "enabled": True,
            }
        ],
        "models": [
            {
                "name": "lightweight",
                "backend": "mock",
                "profile_name": "tiny",
                "enabled": True,
            }
        ],
    }
    profile = ModelRegistry(model_pool_cfg).get_profile("lightweight")
    assert profile.energy_j == 12.0
    assert profile.latency_s == 3.0
    assert profile.quality_base == 0.61
    assert profile.robustness_to_difficulty == 0.77
    assert profile.estimated_power_w == 4.0


def test_payload_profiles_can_be_changed_via_config():
    payload_cfg = {
        "profiles": [
            {
                "name": "metadata",
                "size_mb": 0.22,
                "information_factor": 0.41,
                "priority_weight": 1.3,
                "enabled": True,
            }
        ]
    }
    profile = get_payload_policy_profile("metadata", payload_cfg)
    assert profile.base_size_mb == 0.22
    assert profile.information_level == 0.41
    assert profile.priority_weight == 1.3


def test_action_space_can_be_changed_via_config():
    action_cfg = {
        "actions": [
            {
                "action_name": "tiny+metadata+relay",
                "model_name": "lightweight",
                "payload_policy": "metadata",
                "transmission_option": "relay",
                "expected_payload_level": 0.1,
                "expected_compute_level": 0.2,
                "enabled": True,
            }
        ]
    }
    action = ActionSpace(action_cfg).list_actions()[0]
    assert action.action_name == "tiny+metadata+relay"
    assert action.transmission_mode == "relay"


def test_reward_alias_weights_can_drive_reward_without_legacy_names():
    task = TaskSample("task_cfg", "a", "b", "scene", 0, 0.8, 40.0, 0.3, 0.2)
    context = ContextVector(0.9, 1.0, 0.1, 0.8, 0.2, 0.3, 0.2, 0.6, 0.1)
    reward_cfg = {
        "energy_penalty": 0.4,
        "latency_penalty": 0.2,
        "queue_penalty": 0.1,
        "violation_penalty": 2.0,
        "deadline_miss_penalty": 0.0,
        "high_value_bonus": 0.0,
        "delivery_weight": 1.0,
        "information_weight": 1.0,
        "eta_B": 1.0,
        "eta_Theta": 0.5,
        "info_base": 0.25,
        "info_value_weight": 0.35,
        "info_difficulty_weight": 0.20,
        "info_uncertainty_weight": 0.25,
        "info_urgency_weight": 0.20,
        "deferred_delivery_credit": 0.15,
        "high_value_threshold": 0.75,
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
        E_comp=6.0,
        E_comm=1.0,
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
    result = compute_reward(task, context, 0.9, feedback, reward_cfg, simulator_cfg)
    assert isinstance(result.reward, float)
