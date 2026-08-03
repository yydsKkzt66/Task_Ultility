import pytest

from src.calibration.conditional_v2_calibration import (
    derive_quality_candidates,
    online_lyapunov_baseline,
    price_candidates,
)


def test_quality_candidates_are_strictly_ordered() -> None:
    tasks = [
        {"difficulty": 0.2, "uncertainty": 0.3},
        {"difficulty": 0.8, "uncertainty": 0.7},
    ]
    for candidate in derive_quality_candidates(tasks):
        q = candidate["q_eff"]
        assert q["lightweight"] < q["balanced"] < q["high_accuracy"]


def test_price_perturbations_are_exact() -> None:
    rows = price_candidates({"energy_price_per_Wh": 4.0, "downlink_price_per_MB": 0.2})
    assert [item["value"] for item in rows["energy"]] == [3.0, 4.0, 5.0]
    assert [item["value"] for item in rows["downlink"]] == pytest.approx([0.15, 0.2, 0.25])


def test_public_prices_convert_to_dimensionally_consistent_online_lambdas() -> None:
    result = online_lyapunov_baseline(
        energy_price_per_Wh=3.6,
        downlink_price_per_MB=0.01,
        terminal_backlog_price_per_MB=0.02,
        energy_reference_j=100.0,
        queue_max_mb=5000.0,
        beta_resource_cost=1.0,
        base_lambda_L=0.2,
    )
    assert result["base_lambda_E"] == 0.1
    assert result["base_lambda_S"] == 50.0
    assert result["base_lambda_Q"] == 100.0
