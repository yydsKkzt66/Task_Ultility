from __future__ import annotations

import csv
import inspect

import scripts.run_resource_use_audit as audit
import scripts.run_final_low_budget_check as final_check


def _rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _small_audit(tmp_path):
    return audit.run_audit(
        output_dir=tmp_path,
        scenarios=["multi_window_rescuable_high_value"],
        seeds=[2030],
        slots_list=[160],
        include_sanity=False,
    )


def test_resource_trajectory_generated(tmp_path):
    result = _small_audit(tmp_path)
    path = result["out_dir"] / "resource_trajectory_by_arm.csv"
    rows = _rows(path)
    assert rows
    assert set(audit.TRAJECTORY_COLUMNS).issubset(rows[0].keys())
    assert {row["arm"] for row in rows} >= {"proposed_guarantee_aware_full", "utility_rule"}


def test_safety_headroom_summary_generated(tmp_path):
    result = _small_audit(tmp_path)
    rows = _rows(result["out_dir"] / "safety_headroom_summary.csv")
    proposed = next(row for row in rows if row["arm"] == "proposed_guarantee_aware_full")
    assert set(audit.SAFETY_COLUMNS).issubset(proposed.keys())
    assert int(float(proposed["hard_safety_violation_count"])) == 0
    assert (result["out_dir"] / "safety_headroom_report.md").exists()


def test_recovery_summary_generated(tmp_path):
    result = _small_audit(tmp_path)
    rows = _rows(result["out_dir"] / "recovery_summary.csv")
    assert rows
    assert set(audit.RECOVERY_COLUMNS).issubset(rows[0].keys())
    assert (result["out_dir"] / "recovery_report.md").exists()


def test_resource_to_value_attribution_generated(tmp_path):
    result = _small_audit(tmp_path)
    rows = _rows(result["out_dir"] / "resource_to_value_attribution.csv")
    proposed = next(row for row in rows if row["arm"] == "proposed_guarantee_aware_full")
    assert set(audit.ATTRIBUTION_COLUMNS).issubset(proposed.keys())
    assert float(proposed["H_resource_cost_fraction"]) > 0.0
    assert float(proposed["H_net_delivered_utility"]) > 0.0
    assert (result["out_dir"] / "resource_to_value_report.md").exists()


def test_opportunity_use_summary_generated(tmp_path):
    result = _small_audit(tmp_path)
    rows = _rows(result["out_dir"] / "opportunity_use_summary.csv")
    assert rows
    assert set(audit.OPPORTUNITY_COLUMNS).issubset(rows[0].keys())
    assert (result["out_dir"] / "opportunity_use_report.md").exists()


def test_figures_generated(tmp_path):
    result = _small_audit(tmp_path)
    fig_dir = result["out_dir"] / "figures"
    expected = [
        "figure1_resource_state_trajectories.png",
        "figure2_proposed_energy_balance.png",
        "figure3_resource_to_value_attribution.png",
        "figure4_H_utility_vs_resource_cost.png",
        "figure5_opportunity_utilization.png",
    ]
    for name in expected:
        assert (fig_dir / name).exists()
        assert (fig_dir / name).stat().st_size > 0
    assert (result["out_dir"] / "figure_readme.md").exists()


def test_no_scheduler_behavior_change():
    source = inspect.getsource(audit)
    forbidden = [
        "SimulationEnvironment",
        "compute_reward",
        "reward_function",
        "src.simulator",
        "src.reward",
        "src.routers",
    ]
    for token in forbidden:
        assert token not in source


def test_final_recommendation_generated(tmp_path):
    result = _small_audit(tmp_path)
    path = result["out_dir"] / "final_recommendation.md"
    text = path.read_text(encoding="utf-8")
    assert "Selected conclusions" in text
    assert "Proposed" in text
    assert "trace" in text


def test_completed_resource_trace_has_physical_fields(tmp_path):
    result = final_check.run_completed_resource_trace(
        output_dir=tmp_path,
        scenarios=["multi_window_rescuable_high_value"],
        seeds=[2030],
        slots_list=[160],
    )
    rows = _rows(result["out_dir"] / "resource_trajectory_completed_by_arm.csv")
    assert rows
    assert set(final_check.COMPLETED_TRACE_COLUMNS).issubset(rows[0].keys())
    for field in ["solar_harvest_Wh", "base_energy_Wh", "cooling_delta", "net_energy_delta_Wh"]:
        assert field in rows[0]


def test_proposed_lyapunov_trace_present_or_reported(tmp_path):
    result = final_check.run_completed_resource_trace(
        output_dir=tmp_path,
        scenarios=["multi_window_rescuable_high_value"],
        seeds=[2030],
        slots_list=[160],
    )
    proposed = [row for row in result["rows"] if row["arm"] == "proposed_guarantee_aware_full"]
    report = (result["out_dir"] / "resource_trace_completion_report.md").read_text(encoding="utf-8")
    assert proposed
    assert all(str(row["lambda_E"]) != "NA" for row in proposed)
    assert "Lyapunov prices present for proposed: true" in report
