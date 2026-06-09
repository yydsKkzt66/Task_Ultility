from src.common.types import ContextVector
from src.reward.mission_utility import compute_dynamic_energy_weight


def test_dynamic_energy_weight_increases_when_battery_is_low():
    reward_cfg = {"lambda_E_base": 0.3, "eta_B": 1.0, "eta_Theta": 0.5}
    low_battery = ContextVector(0.2, 1.0, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    high_battery = ContextVector(0.9, 1.0, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert compute_dynamic_energy_weight(low_battery, reward_cfg) > compute_dynamic_energy_weight(high_battery, reward_cfg)
