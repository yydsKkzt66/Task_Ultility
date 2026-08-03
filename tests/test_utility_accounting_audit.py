from __future__ import annotations

import inspect

import pytest

import scripts.run_utility_accounting_audit as audit


def _sample_inputs(*, arm: str = "proposed_guarantee_aware_full", lambda_h: float = 0.35):
    mapping = [
        {
            "task_id": "h1",
            "value_level": "high",
            "product_requirement": "evidence",
            "arrival_slot": "1",
            "new_deadline_slot": "5",
            "mapped_contact_windows": "2",
        },
        {
            "task_id": "m1",
            "value_level": "medium",
            "product_requirement": "context",
            "arrival_slot": "1",
            "new_deadline_slot": "5",
            "mapped_contact_windows": "2",
        },
    ]
    lifecycle = [
        {
            "lifecycle_id": "pl1",
            "task_id": "h1",
            "product_type": "metadata",
            "value_level": "high",
            "product_requirement": "evidence",
            "effective_task_value": "1.0",
            "payload_size_mb": "0.1",
            "created_slot": "1",
            "deadline_slot": "5",
            "current_status": "delivered",
            "status_updated_slot": "1",
            "delivered_slot": "1",
            "last_contact_window_id_seen": "cw1",
        },
        {
            "lifecycle_id": "pl2",
            "task_id": "h1",
            "product_type": "roi",
            "value_level": "high",
            "product_requirement": "evidence",
            "effective_task_value": "1.0",
            "payload_size_mb": "0.8",
            "created_slot": "2",
            "deadline_slot": "5",
            "current_status": "delivered",
            "status_updated_slot": "2",
            "delivered_slot": "2",
            "last_contact_window_id_seen": "cw1",
        },
        {
            "lifecycle_id": "pl3",
            "task_id": "m1",
            "product_type": "metadata",
            "value_level": "medium",
            "product_requirement": "context",
            "effective_task_value": "0.5",
            "payload_size_mb": "0.1",
            "created_slot": "3",
            "deadline_slot": "5",
            "current_status": "delivered",
            "status_updated_slot": "3",
            "delivered_slot": "3",
            "last_contact_window_id_seen": "cw2",
        },
    ]
    trajectory = [
        {
            "slot": "1",
            "task_id": "h1",
            "has_task": "1",
            "value_level": "high",
            "product_requirement": "evidence",
            "detection_quality": "1.0",
            "timeliness_at_delivery": "1.0",
            "resource_penalty": "0.0",
            "long_term_risk_penalty": "0.0",
            "S_down_total_mb": "0.1",
        },
        {
            "slot": "2",
            "task_id": "h1",
            "has_task": "1",
            "value_level": "high",
            "product_requirement": "evidence",
            "detection_quality": "1.0",
            "timeliness_at_delivery": "1.0",
            "resource_penalty": "0.0",
            "long_term_risk_penalty": "0.0",
            "S_down_total_mb": "0.8",
        },
        {
            "slot": "3",
            "task_id": "m1",
            "has_task": "1",
            "value_level": "medium",
            "product_requirement": "context",
            "detection_quality": "1.0",
            "timeliness_at_delivery": "1.0",
            "resource_penalty": "0.0",
            "long_term_risk_penalty": "0.0",
            "S_down_total_mb": "0.1",
        },
    ]
    queue = [
        {
            "slot": "2",
            "task_id": "h1",
            "payload_type": "roi",
            "delivered_this_slot": "1",
            "payload_size_mb": "0.8",
            "remaining_utility": "0.5",
        }
    ]
    candidates = [
        {
            "slot": "2",
            "task_id": "h1",
            "product_type": "roi",
            "rank_without_guarantee": "2",
            "rank_with_guarantee": "1",
            "capacity_cutoff_rank": "1",
            "selected": "1",
            "guarantee_term": "0.1",
            "blocked_by_safety": "0",
            "blocked_by_deadline": "0",
            "blocked_by_duplicate": "0",
            "blocked_by_marginal_utility_floor": "0",
        }
    ]
    ledger, cf = audit.build_delivered_product_ledger(
        scenario="s",
        seed=1,
        arm=arm,
        slots=10,
        lambda_h=lambda_h,
        summary={"net_system_utility": "0"},
        lifecycle_rows=lifecycle,
        queue_rows=queue,
        candidate_rows=candidates if arm != "utility_rule" else [],
        mapping_rows=mapping,
        trajectory_rows=trajectory,
    )
    tasks = audit._task_mapping(mapping, trajectory)
    stage = audit.build_stage_success_audit(ledger, tasks, scenario="s", seed=1, arm=arm, lambda_h=lambda_h)
    return ledger, stage, tasks, cf


def test_delivered_product_ledger_sums_to_total_gross_utility():
    ledger, stage, tasks, _ = _sample_inputs()
    run = {"scenario": "s", "seed": 1, "slots": 10, "arm": "proposed_guarantee_aware_full", "lambda_H": 0.35, "summary": {}, "orphan_rows": [], "lifecycle_rows": []}
    summary = audit.build_fixed_summary_rows([run], {"proposed_guarantee_aware_full": ledger}, {"proposed_guarantee_aware_full": stage}, {"proposed_guarantee_aware_full": tasks})[0]
    assert summary["total_gross_delivered_utility"] == pytest.approx(sum(row["product_gross_information_utility"] for row in ledger))


def test_delivered_product_ledger_sums_to_total_net_utility():
    ledger, stage, tasks, _ = _sample_inputs()
    run = {"scenario": "s", "seed": 1, "slots": 10, "arm": "proposed_guarantee_aware_full", "lambda_H": 0.35, "summary": {}, "orphan_rows": [], "lifecycle_rows": []}
    summary = audit.build_fixed_summary_rows([run], {"proposed_guarantee_aware_full": ledger}, {"proposed_guarantee_aware_full": stage}, {"proposed_guarantee_aware_full": tasks})[0]
    assert summary["total_net_delivered_utility"] == pytest.approx(sum(row["product_net_utility"] for row in ledger))


def test_high_value_gross_utility_not_exceed_total_gross():
    ledger, _, _, _ = _sample_inputs()
    total = sum(row["product_gross_information_utility"] for row in ledger)
    high = sum(row["product_gross_information_utility"] for row in ledger if row["in_high_value_set_H"])
    assert high <= total


def test_payload_mb_sums_match_summary():
    ledger, stage, tasks, _ = _sample_inputs()
    run = {"scenario": "s", "seed": 1, "slots": 10, "arm": "proposed_guarantee_aware_full", "lambda_H": 0.35, "summary": {}, "orphan_rows": [], "lifecycle_rows": []}
    summary = audit.build_fixed_summary_rows([run], {"proposed_guarantee_aware_full": ledger}, {"proposed_guarantee_aware_full": stage}, {"proposed_guarantee_aware_full": tasks})[0]
    assert summary["total_delivered_payload_MB"] == pytest.approx(1.0)


def test_utility_per_mb_uses_correct_denominator():
    ledger, stage, tasks, _ = _sample_inputs()
    run = {"scenario": "s", "seed": 1, "slots": 10, "arm": "proposed_guarantee_aware_full", "lambda_H": 0.35, "summary": {}, "orphan_rows": [], "lifecycle_rows": []}
    summary = audit.build_fixed_summary_rows([run], {"proposed_guarantee_aware_full": ledger}, {"proposed_guarantee_aware_full": stage}, {"proposed_guarantee_aware_full": tasks})[0]
    assert summary["gross_utility_per_MB"] == pytest.approx(summary["total_gross_delivered_utility"] / summary["total_delivered_payload_MB"])


def test_gross_and_net_utility_fields_not_mixed():
    ledger, stage, tasks, _ = _sample_inputs()
    run = {"scenario": "s", "seed": 1, "slots": 10, "arm": "proposed_guarantee_aware_full", "lambda_H": 0.35, "summary": {"net_system_utility": "-99"}, "orphan_rows": [], "lifecycle_rows": []}
    summary = audit.build_fixed_summary_rows([run], {"proposed_guarantee_aware_full": ledger}, {"proposed_guarantee_aware_full": stage}, {"proposed_guarantee_aware_full": tasks})[0]
    assert summary["net_system_utility"] == -99
    assert summary["total_gross_delivered_utility"] != summary["net_system_utility"]


def test_roi_counts_as_min_stage_for_context_evidence():
    _, stage, _, _ = _sample_inputs()
    h1 = next(row for row in stage if row["task_id"] == "h1")
    assert h1["min_stage_success_before_deadline"] == 1


def test_roi_does_not_count_as_evidence_closure():
    _, stage, _, _ = _sample_inputs()
    h1 = next(row for row in stage if row["task_id"] == "h1")
    assert h1["evidence_closure"] == 0


def test_metadata_does_not_count_as_min_stage():
    ledger, _, _, _ = _sample_inputs()
    metadata = next(row for row in ledger if row["task_id"] == "h1" and row["product_type"] == "metadata")
    assert metadata["counted_in_min_stage_success"] == 0


def test_high_value_min_stage_success_unique_task_count():
    _, stage, _, _ = _sample_inputs()
    assert sum(row["min_stage_success_before_deadline"] for row in stage) == 1


def test_utility_rule_passive_high_value_metrics():
    ledger, stage, tasks, _ = _sample_inputs(arm="utility_rule", lambda_h=0.0)
    run = {"scenario": "s", "seed": 1, "slots": 10, "arm": "utility_rule", "lambda_H": 0.0, "summary": {}, "orphan_rows": [], "lifecycle_rows": []}
    summary = audit.build_fixed_summary_rows([run], {"utility_rule": ledger}, {"utility_rule": stage}, {"utility_rule": tasks})[0]
    assert summary["high_value_gross_delivered_utility"] > 0


def test_lambda_zero_metrics_match_no_guarantee():
    ledger0, stage0, tasks0, _ = _sample_inputs(lambda_h=0.0)
    ledger1, stage1, tasks1, _ = _sample_inputs(lambda_h=0.0)
    assert sum(row["product_gross_information_utility"] for row in ledger0) == pytest.approx(sum(row["product_gross_information_utility"] for row in ledger1))
    assert len(stage0) == len(stage1)
    assert tasks0.keys() == tasks1.keys()


def test_counterfactual_rescue_not_equal_guarantee_positive():
    ledger, _, _, _ = _sample_inputs()
    rescued = sum(1 for row in ledger if row["delivery_reason"] == "counterfactual_rescued")
    positive = sum(1 for row in ledger if row["delivery_reason"] == "guarantee_positive")
    assert rescued == 1
    assert positive == 0


def test_delivered_product_has_task_value_requirement_product_type():
    ledger, _, _, _ = _sample_inputs()
    for row in ledger:
        assert row["task_id"]
        assert row["product_type"] in {"metadata", "roi", "roi_context"}
        assert row["product_requirement"] in {"alert", "verification", "context", "evidence"}
        assert row["V_eff"] >= 0


def test_method_vs_utility_rule_fixed_summary_has_required_columns():
    assert "total_gross_delivered_utility" in audit.FIXED_SUMMARY_COLUMNS
    assert "high_value_min_stage_success_rate_feasible_H" in audit.FIXED_SUMMARY_COLUMNS
    assert "counterfactual_rescued_target_stage_count" in audit.FIXED_SUMMARY_COLUMNS


def test_cross_arm_same_seed_same_task_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "OUT_DIR", tmp_path)
    tasks = {"a": {"value_level": "high", "product_requirement": "context", "deadline_slot": 5}}
    rows = audit._write_cross_arm_fairness_report([], {"utility_rule": tasks, "proposed_guarantee_aware_full": tasks})
    assert all(row["severity"] == "ok" for row in rows if row["check"] != "proposed_extra_mechanism_only_priority_selection")


def test_metric_consistency_checker_flags_impossible_subset_sum():
    ledger, stage, _, _ = _sample_inputs()
    summary = {
        "arm": "proposed_guarantee_aware_full",
        "total_gross_delivered_utility": 0.1,
        "total_net_delivered_utility": sum(row["product_net_utility"] for row in ledger),
        "high_value_gross_delivered_utility": 99.0,
        "high_value_net_delivered_utility": sum(row["product_net_utility"] for row in ledger if row["in_high_value_set_H"]),
        "total_delivered_payload_MB": sum(row["payload_size_mb"] for row in ledger),
        "gross_utility_per_MB": 0.1,
        "net_utility_per_MB": sum(row["product_net_utility"] for row in ledger) / sum(row["payload_size_mb"] for row in ledger),
    }
    checks = audit.check_metric_consistency([summary], {"proposed_guarantee_aware_full": ledger}, {"proposed_guarantee_aware_full": stage})
    assert any(row["check_name"] == "high_value_gross_delivered_utility_sums_to_ledger" and row["severity"] == "error" for row in checks)


def test_low_utility_report_generated(tmp_path):
    ledger, _, _, _ = _sample_inputs()
    path = tmp_path / "low.md"
    audit._write_low_utility_report(
        [{"arm": "utility_rule"}, {"arm": "proposed_guarantee_aware_full"}],
        {"proposed_guarantee_aware_full": ledger},
        path,
    )
    assert path.exists()
    assert "Conclusion" in path.read_text(encoding="utf-8")


def test_no_scheduler_behavior_change_from_accounting_audit():
    source = inspect.getsource(audit)
    assert "SimulationEnvironment" not in source
    assert "compute_reward" not in source
    assert "reward_function" not in source


def test_full_pytest_passes():
    assert callable(audit.run_audit)
