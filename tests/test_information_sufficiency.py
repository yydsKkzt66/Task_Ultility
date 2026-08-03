from src.common.types import ContextVector
from src.payload.information_sufficiency import (
    clamp_residual_usefulness_gate,
    compute_gross_incremental_utility,
    compute_information_sufficiency,
    compute_net_incremental_utility,
    delta_information_utility,
    product_information_utility,
    validate_product_utility_table,
)


def test_information_sufficiency_saturates_at_one():
    context = ContextVector(0.8, 0.9, 0.2, 0.3, 0.1, 0.1, 0.1, 0.5, 0.1)
    reward_cfg = {
        "info_base": 0.25,
        "info_value_weight": 0.35,
        "info_difficulty_weight": 0.20,
        "info_uncertainty_weight": 0.25,
        "info_urgency_weight": 0.20,
    }
    assert compute_information_sufficiency(context, 10.0, reward_cfg) == 1.0


def test_metadata_can_be_sufficient_for_easy_task():
    context = ContextVector(0.8, 0.9, 0.2, 0.1, 0.1, 0.05, 0.05, 0.5, 0.1)
    reward_cfg = {
        "info_base": 0.25,
        "info_value_weight": 0.35,
        "info_difficulty_weight": 0.20,
        "info_uncertainty_weight": 0.25,
        "info_urgency_weight": 0.20,
    }
    assert compute_information_sufficiency(context, 0.35, reward_cfg) > 0.8


def test_product_utility_table_is_monotone():
    validate_product_utility_table()


def test_delta_information_utility_and_duplicate_product_zero_increment():
    assert delta_information_utility("metadata", set(), "verification") == 0.35
    assert delta_information_utility("metadata", {"metadata"}, "verification") == 0.0
    assert product_information_utility({"metadata", "roi"}, "verification") == 0.90


def test_gross_incremental_utility_is_nonnegative_before_costs():
    gross = compute_gross_incremental_utility(
        task_value=0.8,
        detection_quality=0.9,
        delta_u_i=0.35,
        timeliness=0.8,
        rho_res=1.0,
    )
    assert gross >= 0.0


def test_resource_cost_is_subtractive_not_multiplied_into_gross_utility():
    gross = compute_gross_incremental_utility(0.8, 0.9, 0.35, 0.8, rho_res=1.0)
    low_cost = compute_net_incremental_utility(gross, resource_cost=0.1)
    high_cost = compute_net_incremental_utility(gross, resource_cost=0.3)
    assert high_cost < low_cost
    assert gross == compute_gross_incremental_utility(0.8, 0.9, 0.35, 0.8, rho_res=1.0)


def test_queue_payload_size_cost_is_subtractive():
    gross = compute_gross_incremental_utility(1.0, 1.0, 0.5, 1.0)
    small_payload_net = compute_net_incremental_utility(gross, queue_cost=0.05)
    large_payload_net = compute_net_incremental_utility(gross, queue_cost=0.20)
    assert large_payload_net < small_payload_net


def test_rho_res_is_bounded():
    assert clamp_residual_usefulness_gate(-1.0) == 0.0
    assert clamp_residual_usefulness_gate(2.0) == 1.0
