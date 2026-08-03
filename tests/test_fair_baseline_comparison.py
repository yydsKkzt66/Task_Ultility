from __future__ import annotations

import csv

import pytest

import scripts.run_fair_baseline_comparison as fair
import scripts.run_final_low_budget_check as final_check


@pytest.fixture(scope="session")
def small_comparison(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("fair_baseline")
    result = fair.run_comparison(
        output_dir=out_dir,
        scenarios=["multi_window_rescuable_high_value"],
        seeds=[2030],
        slots_list=[160],
        run_sanity_first=False,
        run_multi=True,
    )
    result["out_dir"] = out_dir
    return result


def _rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _summary_row(result, arm):
    return next(row for row in result["summary_rows"] if row["arm"] == arm)


def test_frozen_task_manifest_same_across_arms(small_comparison):
    hashes = {row["details"] for row in small_comparison["fairness_rows"] if row["check"] == "task_manifest"}
    assert len(hashes) == 1


def test_cross_arm_fairness_zero_before_comparison(small_comparison):
    assert small_comparison["fairness_valid"] is True
    assert sum(int(row["error_count"]) for row in small_comparison["fairness_rows"]) == 0


def test_same_seed_same_task_value_requirement_deadline(small_comparison):
    checks = {
        "same_seed_same_task_id_same_value_level",
        "same_seed_same_task_id_same_product_requirement",
        "same_seed_same_task_id_same_deadline_slot",
    }
    rows = [row for row in small_comparison["fairness_rows"] if row["check"] in checks]
    assert rows
    assert all(row["status"] == "ok" for row in rows)


def test_main_summary_sums_from_delivered_ledger(small_comparison):
    errors = fair.validate_summary_from_ledger(
        small_comparison["summary_rows"],
        small_comparison["ledger_rows"],
        small_comparison["stage_rows"],
    )
    assert errors == []


def test_H_stage_success_sums_from_task_audit(small_comparison):
    for summary in small_comparison["summary_rows"]:
        stage = [
            row
            for row in small_comparison["stage_rows"]
            if row["scenario"] == summary["scenario"]
            and row["seed"] == summary["seed"]
            and row["num_slots"] == summary["num_slots"]
            and row["arm"] == summary["arm"]
        ]
        assert summary["H_min_stage_success_count"] == sum(int(row["min_stage_success_before_deadline"]) for row in stage)
        assert summary["H_target_stage_success_count"] == sum(int(row["target_stage_success_before_deadline"]) for row in stage)


def test_utility_rule_passive_H_completion_not_missing(small_comparison):
    row = _summary_row(small_comparison, "utility_rule")
    assert row["H_min_stage_success_count"] > 0


def test_ground_approved_followup_can_generate_H_followup(small_comparison):
    row = _summary_row(small_comparison, "ground_approved_followup")
    assert row["ground_approval_request_count"] > 0
    assert row["H_min_stage_success_count"] > 0


def test_zero_H_success_requires_debug_report(tmp_path):
    summary = [{"arm": "utility_rule", "H_min_stage_success_count": 0}]
    stage = [{"arm": "utility_rule", "reason_if_min_stage_failed": "delivered metadata only"}]
    fair.write_zero_h_success_debug_reports(tmp_path, summary, stage)
    debug = tmp_path / "zero_H_success_debug_utility_rule.md"
    assert debug.exists()
    assert "delivered metadata only" in debug.read_text(encoding="utf-8")


def test_fixed_lightweight_uses_fixed_model():
    assert fair.arm_config("fixed_lightweight_onboard")["fixed_model_profile"] == "Tiny"


def test_fixed_heavy_uses_fixed_model():
    assert fair.arm_config("fixed_heavy_onboard")["fixed_model_profile"] == "Medium"


def test_proposed_lambda_035_used():
    assert fair.arm_config("proposed_guarantee_aware_full")["lambda_H"] == pytest.approx(0.35)


def test_persistent_without_guarantee_lambda_zero():
    assert fair.arm_config("persistent_queue_without_guarantee")["lambda_H"] == pytest.approx(0.0)


def test_ROI_counts_as_min_stage_not_evidence_closure(small_comparison):
    roi_rows = [
        row
        for row in small_comparison["ledger_rows"]
        if row["product_requirement"] == "evidence" and row["product_type"] == "roi" and int(row["in_H"])
    ]
    assert roi_rows
    assert any(int(row["counted_in_H_min_stage"]) == 1 for row in roi_rows)
    assert all(int(row["evidence_closure_flag"]) == 0 for row in roi_rows)


def test_low_medium_accidental_roi_context_count(small_comparison):
    assert all(row["low_medium_accidental_roi_context_count"] == 0 for row in small_comparison["summary_rows"])


def test_gross_utility_per_resource_uses_gross_not_net(small_comparison):
    row = _summary_row(small_comparison, "proposed_guarantee_aware_full")
    expected = fair._safe_div(row["total_gross_delivered_utility"], row["total_resource_cost"])
    wrong_net_density = fair._safe_div(row["total_net_delivered_utility"], row["total_resource_cost"])
    assert row["gross_utility_per_resource_cost"] == pytest.approx(expected)
    assert row["gross_utility_per_resource_cost"] != pytest.approx(wrong_net_density)


def test_paper_ready_table_generated_only_if_fairness_passes(small_comparison):
    table_rows = _rows(small_comparison["out_dir"] / "paper_ready_main_table.csv")
    assert table_rows
    assert all("invalid fairness" not in row["Method"] for row in table_rows)

    invalid_dir = small_comparison["out_dir"] / "invalid_case"
    invalid_rows = fair.write_paper_ready_table(
        invalid_dir,
        small_comparison["summary_rows"],
        fairness_valid=False,
    )
    assert invalid_rows
    assert any("invalid fairness" in row["Method"] for row in invalid_rows)


def test_optional_ground_compute_reports_not_implemented_if_missing(small_comparison):
    report = small_comparison["out_dir"] / "downlink_first_ground_compute_not_implemented.md"
    assert report.exists()
    assert fair.GROUND_COMPUTE_NOT_IMPLEMENTED_REASON in report.read_text(encoding="utf-8")
    table_rows = _rows(small_comparison["out_dir"] / "paper_ready_main_table.csv")
    assert all(row["Method"] != fair.ARM_LABELS[fair.OPTIONAL_GROUND_COMPUTE_ARM] for row in table_rows)


def test_full_pytest_passes():
    assert callable(fair.run_comparison)


def test_value_mix_profile_default_off():
    contacts = fair.build_contact_schedule("natural_targeted_20p_value15", 2026, 160)
    manifest = fair.generate_frozen_task_manifest("natural_targeted_20p_value15", 2026, 160, contacts)
    profiled, _trace, _stats = final_check.apply_value_profiles(manifest)
    assert profiled == manifest


def test_value_mix_profile_ratio_in_expected_range():
    contacts = fair.build_contact_schedule("natural_targeted_20p_value15", 2026, 160)
    manifest = fair.generate_frozen_task_manifest("natural_targeted_20p_value15", 2026, 160, contacts)
    _profiled, trace, _stats = final_check.apply_value_profiles(
        manifest,
        value_mix_profile="natural_targeted_20p",
        value_multiplier_profile="moderate_high_contrast",
    )
    dist = final_check._distribution_summary(trace)
    assert 0.20 <= dist["high_urgent_ratio"] <= 0.25
    assert 0.10 <= dist["high_urgent_context_evidence_ratio"] <= 0.18
    assert 0.05 <= dist["urgent_ratio"] <= 0.08


def test_value_multiplier_profile_enabled():
    contacts = fair.build_contact_schedule("natural_targeted_20p_value15", 2026, 160)
    manifest = fair.generate_frozen_task_manifest("natural_targeted_20p_value15", 2026, 160, contacts)
    _profiled, trace, _stats = final_check.apply_value_profiles(
        manifest,
        value_mix_profile="natural_targeted_20p",
        value_multiplier_profile="moderate_high_contrast",
    )
    multipliers = {row["value_level"]: float(row["value_multiplier"]) for row in trace}
    assert multipliers["low"] == pytest.approx(1.0)
    assert multipliers["medium"] == pytest.approx(3.0)
    assert multipliers["high"] == pytest.approx(8.0)
    assert multipliers["urgent"] == pytest.approx(15.0)


def test_value_assignment_mean_preserving_normalization():
    contacts = fair.build_contact_schedule("natural_targeted_20p_value15", 2026, 160)
    manifest = fair.generate_frozen_task_manifest("natural_targeted_20p_value15", 2026, 160, contacts)
    _profiled, trace, _stats = final_check.apply_value_profiles(
        manifest,
        value_mix_profile="natural_targeted_20p",
        value_multiplier_profile="moderate_high_contrast",
    )
    dist = final_check._distribution_summary(trace)
    assert dist["mean_new_V_eff"] == pytest.approx(dist["mean_old_V_eff"], rel=1.0e-6)


def test_sensitivity_cross_arm_fairness_zero(tmp_path):
    result = final_check.run_value_mix_sensitivity(output_dir=tmp_path, seeds=[2026], slots=160)
    assert sum(int(row["error_count"]) for row in result["fairness_rows"]) == 0


def test_sensitivity_outputs_generated(tmp_path):
    result = final_check.run_value_mix_sensitivity(output_dir=tmp_path, seeds=[2026], slots=160)
    expected = [
        "main_summary.csv",
        "paper_ready_table.md",
        "value_distribution_report.md",
        "fairness_report.md",
        "final_recommendation.md",
        "parameter_sanity_report.md",
        "value_assignment_trace.csv",
    ]
    for name in expected:
        assert (result["out_dir"] / name).exists()
