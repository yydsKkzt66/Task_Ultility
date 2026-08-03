from src.energy.battery_update import update_battery_soc
from src.energy.soc_band_policy import apply_soc_band_policy, classify_soc_band


CFG = {
    "enabled": True,
    "soft_target": 0.70,
    "strong_guard": 0.50,
    "hard_min": 0.20,
    "soft_lambda_E_multiplier": 1.45,
    "strong_lambda_E_multiplier": 2.20,
    "soft_model_complexity_bias_delta": -0.12,
    "strong_model_complexity_bias_delta": -0.35,
    "soft_payload_size_bias_delta": -0.08,
    "strong_payload_size_bias_delta": -0.30,
}


def _guidance():
    return {
        "lambda_E": 1.0,
        "lambda_Q": 0.3,
        "lambda_S": 0.2,
        "model_complexity_bias": 0.0,
        "payload_size_bias": 0.0,
        "high_value_bonus_scale": 1.0,
        "safety_margin_scale": 1.0,
        "lambda_after_z": {"lambda_E": 1.0, "lambda_Q": 0.3, "lambda_S": 0.2},
    }


def test_soc_bands_preserve_020_as_hard_bound_not_070():
    assert classify_soc_band(0.80) == "normal"
    assert classify_soc_band(0.69) == "soft_pressure"
    assert classify_soc_band(0.49) == "strong_conservation"
    assert classify_soc_band(0.19) == "hard_safety"


def test_soc_below_070_raises_lambda_e_and_below_050_reduces_heavy_tendency_more():
    normal = apply_soc_band_policy(_guidance(), 0.80, CFG)
    soft = apply_soc_band_policy(_guidance(), 0.65, CFG)
    strong = apply_soc_band_policy(_guidance(), 0.45, CFG)

    assert soft["lambda_E"] > normal["lambda_E"]
    assert strong["lambda_E"] > soft["lambda_E"]
    assert soft["model_complexity_bias"] < normal["model_complexity_bias"]
    assert strong["model_complexity_bias"] < soft["model_complexity_bias"]
    assert strong["payload_size_bias"] < soft["payload_size_bias"]


def test_battery_soc_update_is_clamped_to_unit_interval():
    overfull = update_battery_soc(
        battery_soc=0.99,
        solar_energy_j=100000.0,
        base_power_w=0.0,
        slot_seconds=20.0,
        compute_energy_j=0.0,
        comm_energy_j=0.0,
        battery_capacity_j=1000.0,
    )
    empty = update_battery_soc(
        battery_soc=0.01,
        solar_energy_j=0.0,
        base_power_w=1000.0,
        slot_seconds=20.0,
        compute_energy_j=100000.0,
        comm_energy_j=100000.0,
        battery_capacity_j=1000.0,
    )

    assert overfull == 1.0
    assert empty == 0.0
