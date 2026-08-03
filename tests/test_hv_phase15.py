from scripts.run_hv_phase15_correction_audit_seed2026 import b1_hv_breakdown, staged_closure_audit


def test_staged_audit_uses_outcome_value_level_and_reconstructed_action():
    outcomes = [{"task_id": "T1", "value_level": "routine", "product_requirement": "verification",
                 "requirement_satisfied": "1", "requirement_closure_time_s": "120",
                 "transmitted_product_type": "metadata+roi"}]
    ledger = [
        {"task_id": "T1", "source": "ordinary_downlink_queue", "current_status": "delivered",
         "product_type": "metadata", "delivered_slot": "1", "request_id": ""},
        {"task_id": "T1", "source": "staged_delivery_queue", "current_status": "delivered",
         "product_type": "roi", "delivered_slot": "2", "request_id": "staged_T1"},
    ]
    rows = staged_closure_audit(outcomes, ledger)
    assert rows[0]["selected_action"] == "metadata_then_roi"
    assert rows[0]["delivered_stages"] == "metadata;roi"
    assert rows[0]["in_HV_ALL"] == 0
    assert rows[0]["in_HV_TARGET"] == 0


def test_b1_hv_breakdown_includes_zero_categories():
    rows = [{"task_id": "T1", "value_level": "high_value", "product_requirement": "context",
             "requirement_satisfied": "1", "on_time_task_success": "1"}]
    output = b1_hv_breakdown({"B1": rows})
    by = {(row["dimension"], row["category"]): row for row in output}
    assert by[("requirement", "context")]["closure_count"] == 1
    assert by[("requirement", "alert")]["closure_count"] == 0
    assert by[("value_level", "urgent")]["closure_count"] == 0
