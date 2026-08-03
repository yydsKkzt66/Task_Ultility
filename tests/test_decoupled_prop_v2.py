from types import SimpleNamespace

from src.routers.decoupled_prop_v2 import (
    FROZEN_QUALITY_PROXY,
    action_grid_audit,
    build_decoupled_action_rows,
    score_action_q_once,
    timeliness_factor,
)


def test_full_and_fixed_action_grids_are_exact() -> None:
    full = build_decoupled_action_rows()
    fixed = build_decoupled_action_rows(fixed_model="balanced")
    assert action_grid_audit(full)["complete"]
    assert len(full) == 10
    assert action_grid_audit(fixed, fixed_model="balanced")["complete"]
    assert len(fixed) == 4


def test_q_occurs_once_and_uses_frozen_model_proxy() -> None:
    task = SimpleNamespace(
        task_value=10.0,
        calibrated_task_value=None,
        product_requirement="evidence",
        arrival_slot=0,
        deadline_s=100.0,
    )
    action = SimpleNamespace(
        action_name="v2:high_accuracy+roi_context",
        model_name="high_accuracy",
        payload_policy="roi_context",
        expected_compute_level=0.95,
        expected_payload_level=0.95,
    )
    predicted = SimpleNamespace(
        predicted_product_type="roi_context",
        predicted_detection_quality=0.01,
        delivery_success_proxy=0.5,
        predicted_E_comp=0.0,
        predicted_E_comm=0.0,
        predicted_L_inf=0.0,
        predicted_L_queue=0.0,
        predicted_L_comm=0.0,
        predicted_Q_next_mb=0.0,
        predicted_S_gen_mb=0.0,
        predicted_B_next=1.0,
        predicted_Theta_next=0.0,
    )
    score = score_action_q_once(
        task=task,
        action=action,
        predicted=predicted,
        delivered_products=set(),
        product_utility_cfg=None,
        guidance_bundle={"lambda_E": 0, "lambda_L": 0, "lambda_Q": 0, "lambda_S": 0},
        simulator_cfg={"epsilon": 1e-9, "E_ref": 100, "queue_max_mb": 64, "battery_min": 0.2, "thermal_max": 1.0},
        beta_resource_cost=1.0,
        beta_hard_penalty=1.0,
        ground_arrival_s=50.0,
        slot_seconds=20.0,
    )
    assert score.quality == FROZEN_QUALITY_PROXY["high_accuracy"]
    assert score.q_multiplication_count == 1
    assert score.positive_benefit == 10.0 * 0.95 * 0.94 * 0.5 * 0.5


def test_timeliness_is_arrival_relative_and_deadline_censored() -> None:
    assert timeliness_factor(task_arrival_s=100, deadline_duration_s=200, ground_arrival_s=150) == 0.75
    assert timeliness_factor(task_arrival_s=100, deadline_duration_s=200, ground_arrival_s=350) == 0.0


def test_defer_has_no_quality_or_positive_benefit() -> None:
    task = SimpleNamespace(task_value=25.0, calibrated_task_value=None, product_requirement="evidence", arrival_slot=0, deadline_s=100.0)
    action = SimpleNamespace(action_name="delay/store", model_name="delay_store", payload_policy="defer", expected_compute_level=0.0, expected_payload_level=0.0)
    predicted = SimpleNamespace(predicted_product_type="defer", predicted_E_comp=0.0, predicted_E_comm=0.0, predicted_L_inf=0.0, predicted_L_queue=0.0, predicted_L_comm=0.0, predicted_Q_next_mb=0.0, predicted_S_gen_mb=0.0, predicted_B_next=1.0, predicted_Theta_next=0.0)
    score = score_action_q_once(task=task, action=action, predicted=predicted, delivered_products=set(), product_utility_cfg=None, guidance_bundle={}, simulator_cfg={"epsilon": 1e-9, "E_ref": 100, "queue_max_mb": 64, "battery_min": 0.2, "thermal_max": 1.0}, beta_resource_cost=1.0, beta_hard_penalty=1.0, ground_arrival_s=None, slot_seconds=20)
    assert score.positive_benefit == 0.0
    assert score.q_multiplication_count == 0
