from scripts.run_outcome_taxonomy_hard_attribution_audit import classify_row


def _base_row(**updates):
    row = {
        "mission_gain": "0.0",
        "C_resource": "0.0",
        "P_requirement": "0.0",
        "P_quality_gap": "0.0",
        "P_info_gap": "0.0",
        "P_hard": "0.0",
        "P_safety": "0.0",
        "P_deadline": "0.0",
        "P_expiration": "0.0",
        "P_infeasible": "0.0",
        "P_terminal_risk": "0.0",
        "hard_failure_flag": "0",
        "success_flag": "0",
        "delivered_before_deadline": "0",
        "action": "delay/store",
        "payload_type": "defer",
    }
    row.update(updates)
    return row


def test_terminal_risk_only_legacy_hard_is_not_true_hard_failure():
    result = classify_row(
        _base_row(P_hard="0.2", P_terminal_risk="0.2", hard_failure_flag="1")
    )

    assert result["true_hard_failure"] == 0
    assert result["legacy_hard_failure_flag"] == 1
    assert result["terminal_risk_note"] == 1
    assert result["taxonomy_label"] == "unattempted_low_value_or_deferred"
    assert result["hard_failure_source"] == "terminal_risk_note_only"


def test_safety_penalty_defines_true_hard_failure():
    result = classify_row(_base_row(P_hard="0.3", P_safety="0.3", hard_failure_flag="1"))

    assert result["true_hard_failure"] == 1
    assert result["terminal_risk_note"] == 0
    assert result["taxonomy_label"] == "true_hard_failure"
    assert result["hard_failure_source"] == "P_safety"


def test_full_success_requires_success_delivery_gain_and_no_true_hard():
    result = classify_row(
        _base_row(
            mission_gain="0.4",
            C_resource="0.03",
            success_flag="1",
            delivered_before_deadline="1",
            action="balanced+roi",
            payload_type="roi",
        )
    )

    assert result["taxonomy_label"] == "full_success"
    assert result["attempted_flag"] == 1


def test_deferred_zero_gain_zero_cost_is_unattempted_even_with_soft_shortfall():
    result = classify_row(_base_row(P_requirement="0.3", P_quality_gap="0.2"))

    assert result["taxonomy_label"] == "unattempted_low_value_or_deferred"
    assert result["soft_shortfall_flag"] == 1
    assert result["unattempted_low_value_flag"] == 1
