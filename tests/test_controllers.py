from pathlib import Path

from src.common.config import load_project_configs
from src.common.types import LongTermState
from src.controllers.lyapunov_pricing import LyapunovPricing
from src.controllers.mode_controller import RuleBasedModeController
from src.evaluation.baselines import build_routers
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


def test_lyapunov_pricing_returns_all_base_prices():
    pricing = LyapunovPricing(
        {
            "base_lambda_E": 0.30,
            "base_lambda_L": 0.20,
            "base_lambda_Q": 0.15,
            "base_lambda_S": 0.12,
            "alpha_energy": 1.20,
            "alpha_thermal": 0.80,
            "alpha_queue": 0.90,
            "alpha_deadline": 0.75,
            "alpha_egress": 1.10,
        }
    )
    long_state = LongTermState(
        battery_soc=0.2,
        battery_safe=0.3,
        thermal_pressure=0.6,
        thermal_max=1.0,
        queue_mb=20.0,
        queue_max_mb=40.0,
        recent_deadline_urgency=0.7,
        recent_high_value_ratio=0.5,
        next_contact_s=100.0,
        low_contact_capacity_factor=0.8,
        queue_growth_rate=0.5,
    )
    prices = pricing.compute_base_prices(long_state)
    assert set(prices) == {"lambda_E", "lambda_L", "lambda_Q", "lambda_S"}


def test_rule_based_mode_controller_selects_energy_saving_under_low_battery():
    controller = RuleBasedModeController(
        {
            "update_interval_slots": 10,
            "thresholds": {
                "battery_low": 0.30,
                "thermal_high": 0.80,
                "queue_high": 0.65,
                "deadline_pressure_high": 0.65,
                "high_value_ratio_high": 0.55,
                "next_contact_long_s": 120.0,
            },
            "modes": {
                "balanced": {
                    "price_multipliers": {"lambda_E": 1.0, "lambda_L": 1.0, "lambda_Q": 1.0, "lambda_S": 1.0},
                    "exploration_scale": 1.0,
                    "model_complexity_bias": 0.0,
                    "payload_size_bias": 0.0,
                    "high_value_bonus_scale": 1.0,
                    "safety_margin_scale": 1.0,
                },
                "energy_saving": {
                    "price_multipliers": {"lambda_E": 1.4, "lambda_L": 1.0, "lambda_Q": 1.0, "lambda_S": 1.0},
                    "exploration_scale": 0.9,
                    "model_complexity_bias": -0.2,
                    "payload_size_bias": -0.1,
                    "high_value_bonus_scale": 1.0,
                    "safety_margin_scale": 1.2,
                },
            },
        }
    )
    long_state = LongTermState(
        battery_soc=0.2,
        battery_safe=0.3,
        thermal_pressure=0.4,
        thermal_max=1.0,
        queue_mb=5.0,
        queue_max_mb=40.0,
        recent_deadline_urgency=0.2,
        recent_high_value_ratio=0.1,
        next_contact_s=0.0,
        low_contact_capacity_factor=0.0,
        queue_growth_rate=0.0,
    )
    mode_name = controller.select_mode(long_state)
    guidance = controller.build_guidance({"lambda_E": 0.3, "lambda_L": 0.2, "lambda_Q": 0.1, "lambda_S": 0.1}, mode_name)
    assert mode_name == "energy_saving"
    assert "z_k" in guidance
    assert guidance["lambda_E"] > 0.3


def test_environment_can_toggle_lyapunov_pricing_and_mode_controller():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    env = SimulationEnvironment(configs)
    assert env.lyapunov_pricing is not None
    assert env.mode_controller is not None

    configs["lyapunov_pricing"]["enabled"] = False
    configs["mode_controller"]["enabled"] = False
    disabled_env = SimulationEnvironment(configs)
    assert disabled_env.lyapunov_pricing is None
    assert disabled_env.mode_controller is None


def test_episode_runs_when_lyapunov_pricing_is_disabled():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    configs["timeline"]["use_num_days"] = False
    configs["lyapunov_pricing"]["enabled"] = False
    configs["timeline"]["num_slots"] = 4
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers({"routers": ["utility_rule"]}, env.action_space)[0]
    result = env.build_episode_runner(timeline).run(router)
    assert len(result.records) == 4


def test_episode_runs_when_mode_controller_is_disabled():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    configs["timeline"]["use_num_days"] = False
    configs["mode_controller"]["enabled"] = False
    configs["timeline"]["num_slots"] = 4
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers({"routers": ["utility_rule"]}, env.action_space)[0]
    result = env.build_episode_runner(timeline).run(router)
    assert len(result.records) == 4
