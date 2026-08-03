from pathlib import Path

from src.common.config import load_yaml
from src.timeline.resource_trace_loader import load_synthetic_resource_trace
from scripts.run_bupt1_long_horizon_energy_audit import run_long_horizon_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_bupt1_profile_has_eclipse_zero_solar_and_variable_sunlit_power():
    profile = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_bupt1_balanced_long_horizon.yaml")
    traces = load_synthetic_resource_trace(
        360,
        {
            "slot_seconds": 20.0,
            "resource": profile,
        },
    )

    assert profile["battery_min"] == 0.20
    assert any(trace.is_eclipse for trace in traces)
    assert all(trace.solar_energy_j == 0.0 for trace in traces if trace.is_eclipse)

    sunlit_powers = [round(trace.solar_power_w, 6) for trace in traces if trace.is_sunlit]
    assert any(power > 0.0 for power in sunlit_powers)
    assert len(set(sunlit_powers)) > 5


def test_sine_power_shape_matches_bupt1_nominal_orbit_energy():
    traces = load_synthetic_resource_trace(
        288,
        {
            "slot_seconds": 20.0,
            "resource": {
                "orbit_period_min": 96.0,
                "sunlit_duration_min": 60.0,
                "solar_peak_w": 40.0,
                "solar_shape": "sine_power",
                "solar_shape_exponent": 3.0,
                "resource_power_variation_w": 0.0,
                "day_cycle_orbits": 1,
            },
        },
    )

    assert all(trace.solar_energy_j == 0.0 for trace in traces if trace.is_eclipse)
    solar_wh = sum(trace.solar_energy_j for trace in traces) / 3600.0
    assert 16.0 <= solar_wh <= 17.5


def test_bupt1_nominal_and_energy_shadow_stress_profiles_are_separate():
    nominal = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_bupt1_balanced_long_horizon.yaml")
    stress = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")

    assert nominal["profile_name"] == "bupt1_balanced_long_horizon"
    assert stress["profile_name"] == "energy_shadow_stress"
    assert "not nominal" in stress["description"]
    assert nominal["solar_peak_w"] > stress["solar_peak_w"]
    assert nominal["battery_capacity_wh"] == 230.0


def test_long_horizon_audit_smoke_uses_shared_trace_and_reports_hard_safety(tmp_path):
    run_long_horizon_audit(output_dir=tmp_path, num_slots=20, slot_seconds=20.0)

    manifest = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_bupt1_balanced_long_horizon.yaml")
    assert manifest["battery_min"] == 0.20
    assert (tmp_path / "long_horizon_main_summary.csv").exists()
    assert (tmp_path / "hard_safety_audit.csv").exists()
    assert (tmp_path / "arm_resource_task_manifest_audit.csv").exists()

    import csv

    with (tmp_path / "arm_resource_task_manifest_audit.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    task_hashes = {row["task_manifest_sha256"] for row in rows}
    resource_hashes = {row["resource_trace_sha256"] for row in rows}
    assert len(task_hashes) == 1
    assert len(resource_hashes) == 1

    with (tmp_path / "long_horizon_main_summary.csv").open(encoding="utf-8-sig", newline="") as file:
        summary_rows = list(csv.DictReader(file))
    assert summary_rows
    assert "hard_safety_violation_count" in summary_rows[0]
