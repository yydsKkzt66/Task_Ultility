from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.routers.myopic_e2e_utility_greedy import MyopicE2EUtilityGreedyRouter


def _router(reference: float = 4.0) -> MyopicE2EUtilityGreedyRouter:
    return MyopicE2EUtilityGreedyRouter(
        {"utility_calibration": {"enable_value_cost_calibration": False}},
        {"E_ref": 100.0, "queue_max_mb": 64.0},
        {"utility_table": {}},
        task_value_reference=reference,
    )


def test_v25_value_reference_preserves_ratio_without_exceeding_one() -> None:
    router = _router()
    router.current_step = SimpleNamespace(
        task=SimpleNamespace(calibrated_task_value=4.0),
    )
    assert router._normalized_task_value() == pytest.approx(1.0)
    router.current_step.task.calibrated_task_value = 0.4
    assert router._normalized_task_value() == pytest.approx(0.1)


def test_standalone_identity_disables_long_horizon_guidance() -> None:
    router = _router()
    assert router.use_lyapunov_pricing is False
    assert router.use_mode_controller is False
    assert router.use_guided_utility is False
    assert router.online_update_count == 0


def test_prepare_decision_binds_only_current_observable_state() -> None:
    router = _router()
    step, resource, comm = object(), object(), object()
    router.prepare_decision(
        timeline_step=step,
        resource_state=resource,
        comm_state=comm,
        context=object(),
        predicted_map={},
    )
    assert router.current_step is step
    assert router.current_resource_state is resource
    assert router.current_comm_state is comm


def test_common_predicted_reward_mode_compares_action_and_defer_once() -> None:
    router = MyopicE2EUtilityGreedyRouter(
        {"utility_calibration": {"enable_value_cost_calibration": False}},
        {"E_ref": 100.0, "queue_max_mb": 64.0},
        {"utility_table": {}},
        task_value_reference=4.0,
        score_mode="common_predicted_reward",
    )
    router.current_step = SimpleNamespace(
        task=SimpleNamespace(calibrated_task_value=4.0),
    )
    execute = SimpleNamespace(action_name="execute", payload_policy="roi", model_name="balanced")
    defer = SimpleNamespace(action_name="delay/store", payload_policy="defer", model_name="delay_store")

    def predicted(reward: float, payload: float) -> SimpleNamespace:
        return SimpleNamespace(
            predicted_reward=reward,
            predicted_product_type="roi",
            effective_info_gain=0.5,
            predicted_delivery_factor=0.5,
            predicted_L_inf=1.0,
            predicted_L_queue=2.0,
            predicted_L_comm=3.0,
            predicted_E_comp=4.0,
            predicted_E_comm=5.0,
            predicted_S_gen_mb=payload,
        )

    selected = router.select_action(
        SimpleNamespace(),
        [execute, defer],
        {"execute": predicted(0.1, 1.0), "delay/store": predicted(-0.2, 0.0)},
    )
    assert selected is execute
    assert router.last_selected_action_score == pytest.approx(0.1)
