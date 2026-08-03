from scripts.audit_high_value_staged_activation_seed2026 import build_task_audit, capacity_overlap_audit


def test_roi_delivery_fails_roi_context_gate_when_runtime_level_is_medium():
    outcomes = [{"task_id": "H1", "value_level": "high_value", "product_requirement": "context",
                 "requirement_satisfied": "0", "failure_reason": "quality_gap", "expired": "0"}]
    ledger = [
        {"task_id": "H1", "source": "ordinary_downlink_queue", "product_type": "roi",
         "current_status": "created", "value_level": "", "created_slot": "1"},
        {"task_id": "H1", "source": "ordinary_downlink_queue", "product_type": "roi",
         "current_status": "delivered", "value_level": "", "delivered_slot": "2"},
    ]
    rows = build_task_audit(outcomes, ledger, [], runtime_level_evidence="medium")
    assert rows[0]["roi_delivered"] == 1
    assert rows[0]["eligible_for_staged_roi_context_before_value_gate"] == 1
    assert rows[0]["staged_request_created"] == 0
    assert rows[0]["exact_failed_condition"] == "runtime_value_level_not_high_or_urgent"


def test_capacity_overlap_requires_same_active_lifetime_or_window():
    high_created = [{"task_id": "H1", "created_slot": "10", "deadline_slot": "11",
                     "source": "staged_delivery_queue", "current_status": "created"}]
    ordinary_delivered = [{"task_id": "R1", "delivered_slot": "20", "payload_size_mb": "0.8",
                           "last_contact_window_id_seen": "W2"}]
    result = capacity_overlap_audit(high_created, ordinary_delivered, [], [])
    assert result["active_lifetime_overlap_pair_count"] == 0
    assert result["same_window_recompetition_pair_count"] == 0
    assert result["displacement_supported_by_trace"] == 0
