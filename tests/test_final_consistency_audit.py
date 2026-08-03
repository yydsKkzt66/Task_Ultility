from __future__ import annotations

import csv

import pytest

import scripts.run_fair_baseline_comparison as fair
import scripts.run_final_low_budget_check as final_check


def _rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


@pytest.fixture(scope="session")
def small_final_audit(tmp_path_factory):
    fair_dir = tmp_path_factory.mktemp("fair_for_final")
    fair.run_comparison(
        output_dir=fair_dir,
        scenarios=["multi_window_rescuable_high_value"],
        seeds=[2030],
        slots_list=[160],
        run_sanity_first=False,
        run_multi=True,
    )
    resource_dir = tmp_path_factory.mktemp("resource_for_final")
    resource_result = final_check.run_completed_resource_trace(
        output_dir=resource_dir,
        scenarios=["multi_window_rescuable_high_value"],
        seeds=[2030],
        slots_list=[160],
    )
    output_dir = tmp_path_factory.mktemp("final_audit")
    result = final_check.run_final_consistency_audit(
        fair_dir=fair_dir,
        utility_dir=fair_dir,
        resource_dir=resource_dir,
        output_dir=output_dir,
    )
    result["resource_result"] = resource_result
    return result


def test_final_metric_consistency_audit_generated(small_final_audit):
    out_dir = small_final_audit["out_dir"]
    assert (out_dir / "final_metric_consistency_report.md").exists()
    rows = _rows(out_dir / "final_metric_consistency_checks.csv")
    assert rows
    assert any(row["check_name"] == "cross_arm_fairness_zero" for row in rows)


def test_final_metric_consistency_no_subset_sum_error(small_final_audit):
    rows = _rows(small_final_audit["out_dir"] / "final_metric_consistency_checks.csv")
    error_rows = [row for row in rows if row["severity"] == "error"]
    assert error_rows == []
    assert any(row["check_name"] == "H_gross_not_above_total_gross" for row in rows)


def test_no_default_behavior_change():
    contacts = fair.build_contact_schedule("multi_window_rescuable_high_value", 2030, 160)
    manifest = fair.generate_frozen_task_manifest("multi_window_rescuable_high_value", 2030, 160, contacts)
    profiled, trace, stats = final_check.apply_value_profiles(manifest)
    assert profiled == manifest
    assert all(row["value_mix_profile"] == "default" for row in trace)
    assert stats["normalization_alpha"] == pytest.approx(1.0)
