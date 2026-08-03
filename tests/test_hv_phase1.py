from __future__ import annotations

import pytest

from src.evaluation.hv_phase1 import (
    classify_outcome,
    evaluate_b1_frame,
    is_hv_all,
    is_hv_target,
    matched_latency,
    validate_method_rows,
    summarize_group,
)


def _task(requirement: str = "evidence") -> dict:
    return {
        "task_id": "T1",
        "product_requirement": requirement,
        "deadline_time_s": 200.0,
        "value_level": "high_value",
        "effective_task_value": 0.9,
    }


def _frame(**overrides) -> dict:
    row = {
        "state": "ground_result_available",
        "compressed_size_mb": 4.0,
        "ground_result_available": 1,
        "ground_result_available_time_s": 150.0,
        "ground_task_quality": 0.88,
        "reconstructed_image_quality_proxy": 0.96,
        "ground_decode_latency_s": 1.5,
        "ground_inference_latency_s": 2.8,
    }
    row.update(overrides)
    return row


def test_b1_full_frame_requires_all_four_dominance_conditions():
    success = evaluate_b1_frame(_frame(), _task())
    assert success.full_frame_received
    assert success.ground_processing_success
    assert success.frame_contains_required_information
    assert success.quality_pass
    assert success.requirement_satisfied

    incomplete = evaluate_b1_frame(_frame(state="expired", ground_result_available=0), _task("alert"))
    assert not incomplete.requirement_satisfied
    assert incomplete.failure_reason == "transmission_incomplete"

    missing_information = evaluate_b1_frame(
        _frame(),
        {**_task("evidence"), "full_frame_information_capabilities": "alert;verification;context"},
    )
    assert not missing_information.requirement_satisfied
    assert missing_information.failure_reason == "lower_than_required_product"

    low_quality = evaluate_b1_frame(_frame(reconstructed_image_quality_proxy=0.94), _task("evidence"))
    assert not low_quality.requirement_satisfied
    assert low_quality.failure_reason == "quality_gap"


def test_hv_groups_use_value_and_requirement():
    assert is_hv_all(_task("alert"))
    assert not is_hv_target(_task("alert"))
    assert is_hv_target(_task("context"))
    assert is_hv_target(_task("evidence"))
    assert not is_hv_all({**_task(), "value_level": "routine"})


def test_outcome_partition_is_exclusive():
    assert classify_outcome({"requirement_satisfied": 1, "on_time_task_success": 1}) == "on_time_requirement_success"
    assert classify_outcome({"requirement_satisfied": 1, "on_time_task_success": 0}) == "late_requirement_satisfied"
    assert classify_outcome({"requirement_satisfied": 0, "ground_completed": 1}) == "partial_only"
    assert classify_outcome({"requirement_satisfied": 0, "ground_completed": 0}) == "no_ground"


def test_matched_latency_uses_exact_declared_intersection():
    left = [
        {"task_id": "A", "requirement_satisfied": 1, "capture_to_ground_latency_s": 10.0},
        {"task_id": "B", "requirement_satisfied": 1, "capture_to_ground_latency_s": 30.0},
    ]
    right = [
        {"task_id": "B", "requirement_satisfied": 1, "capture_to_ground_latency_s": 20.0},
        {"task_id": "C", "requirement_satisfied": 1, "capture_to_ground_latency_s": 5.0},
    ]
    result = matched_latency(left, right, lambda row: bool(int(row["requirement_satisfied"])))
    assert result["matched_task_count"] == 1
    assert result["matched_task_ids"] == "B"
    assert result["mean_paired_latency_difference_s"] == 10.0


def test_no_stage_authoritative_count_is_72():
    good = [{"task_id": f"T{i}", "requirement_satisfied": int(i < 72), "on_time_task_success": int(i < 72)} for i in range(370)]
    validate_method_rows("PROP-NO-STAGE", good)
    stale = [{**row, "requirement_satisfied": int(i < 17), "on_time_task_success": int(i < 17)} for i, row in enumerate(good)]
    with pytest.raises(ValueError, match="72"):
        validate_method_rows("PROP-NO-STAGE", stale)


def test_group_efficiency_uses_group_success_denominator():
    rows = [
        {"task_id": "H1", "value_level": "high_value", "product_requirement": "context",
         "requirement_satisfied": 1, "on_time_task_success": 1, "ground_completed": 1,
         "ground_result_quality": 0.9, "ground_information_sufficiency": 1.0,
         "capture_to_ground_latency_s": 10.0, "deadline_time_s": 100.0,
         "requirement_closure_time_s": 10.0, "gross_ground_utility": 1.0},
        {"task_id": "H2", "value_level": "high_value", "product_requirement": "alert",
         "requirement_satisfied": 1, "on_time_task_success": 1, "ground_completed": 1,
         "ground_result_quality": 0.9, "ground_information_sufficiency": 1.0,
         "capture_to_ground_latency_s": 10.0, "deadline_time_s": 100.0,
         "requirement_closure_time_s": 10.0, "gross_ground_utility": 1.0},
    ]
    summary = summarize_group(
        rows,
        lambda row: row["product_requirement"] in {"context", "evidence"},
        system_summary={"controllable_mission_energy_wh": 10.0, "delivered_mb": 20.0},
        direct_ids={"H1", "H2"},
        staged_ids=set(),
    )
    assert summary["system_energy_per_G_success_wh"] == 10.0
    assert summary["system_MB_per_G_success"] == 20.0
    assert summary["direct_closures"] + summary["staged_closures"] == summary["requirement_satisfied"]
    assert summary["efficiency_scope"] == "system_level_not_task_attribution"
