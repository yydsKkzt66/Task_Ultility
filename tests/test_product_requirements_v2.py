from src.data.product_requirements import (
    assign_product_requirement,
    assign_product_requirement_v2,
    assign_product_requirement_v2_1,
    recommend_payload_plan_from_requirement,
)


def test_old_product_requirement_rule_is_preserved_for_emergency_deadline():
    assert (
        assign_product_requirement(
            task_value=0.94,
            deadline_s=35.0,
            uncertainty=0.70,
            high_value_flag=True,
            emergency_flag=True,
            task_class="emergency",
        )
        == ("alert", "mission_prior_proxy")
    )


def test_v2_preserves_ideal_evidence_under_resource_pressure():
    result = assign_product_requirement_v2(
        task_value=0.94,
        deadline_s=35.0,
        remaining_deadline_s=3.0,
        uncertainty=0.70,
        high_value_flag=True,
        emergency_flag=True,
        task_class="emergency",
        mission_source="mock",
        scene_prior=0.7,
        product_use_case="mission_review",
        queue_mb=63.0,
        queue_max_mb=64.0,
        known_contact_capacity_before_deadline=0.05,
        required_capacity_for_roi_context=2.7,
        battery_slack=0.4,
        thermal_slack=0.4,
        queue_slack=0.02,
        contact_slack=0.02,
        deadline_slack=0.02,
    )

    assert result.ideal_product_requirement == "evidence"
    assert result.requirement_source == "mission_prior_product_requirement_v2_shadow"
    assert "mission_level_guard_preserved_evidence_need_under_pressure" in result.diagnostic_reason


def test_shadow_payload_plan_handles_feasibility_separately_from_requirement():
    recommendation = recommend_payload_plan_from_requirement(
        ideal_product_requirement="evidence",
        metadata_feasible=True,
        roi_feasible=False,
        roi_context_feasible=False,
        metadata_then_roi_feasible=True,
        metadata_then_roi_context_feasible=True,
    )

    assert recommendation.recommended_payload_plan == "metadata_then_roi_context"
    assert recommendation.recommendation_source == "product_requirement_v2_shadow_payload_plan"


def test_v2_1_low_value_high_uncertainty_stays_verification():
    result = assign_product_requirement_v2_1(
        task_value=0.25,
        uncertainty=0.88,
        mission_source="unit_test",
        scene_context_need=0.5,
        product_use_case="review",
        timeliness_class="normal",
    )

    assert result.ideal_product_requirement == "verification"
    assert result.initial_alert_required is False
    assert result.requirement_source == "mission_prior_product_requirement_v2_1_shadow"


def test_v2_1_emergency_high_value_high_uncertainty_is_evidence_with_alert_flag():
    result = assign_product_requirement_v2_1(
        task_value=0.94,
        uncertainty=0.82,
        high_value_flag=True,
        emergency_flag=True,
        task_class="emergency",
        mission_source="unit_test",
        scene_context_need=0.8,
        product_use_case="incident_evidence",
        timeliness_class="urgent",
    )

    assert result.ideal_product_requirement == "evidence"
    assert result.initial_alert_required is True


def test_v2_1_high_value_low_uncertainty_not_automatically_evidence():
    result = assign_product_requirement_v2_1(
        task_value=0.86,
        uncertainty=0.18,
        high_value_flag=True,
        task_class="high_value",
        mission_source="unit_test",
        scene_context_need=0.5,
        product_use_case="review",
        timeliness_class="normal",
    )

    assert result.ideal_product_requirement in {"verification", "context"}
    assert result.ideal_product_requirement != "evidence"
