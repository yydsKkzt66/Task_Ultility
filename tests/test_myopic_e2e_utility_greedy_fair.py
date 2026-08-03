from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.run_myopic_e2e_utility_greedy import (
    DEFAULT_METHODS,
    DEFAULT_VALUE_PROFILE,
    METHOD_NAMES,
    MYOPIC_FROZEN_PARAMETERS,
    MYOPIC_PARAMETER_FILE,
    VALUE_PROFILES,
    _closing_event,
    _parse_methods,
    build_full_closure_rows,
    build_method_router,
    build_shared_scenario,
    build_unified_metrics,
    configure_target_scenario_profile,
)
from scripts.run_prop_reqalign_seed2026_v1 import (
    METHOD_PROP_NEW,
    RequirementAwareRouterWrapper,
)
from src.common.types import (
    ContextVector,
    JointAction,
    LinkTrace,
    PredictedMetrics,
    ResourceTrace,
    TaskSample,
    TimelineStep,
)
from src.routers.myopic_frozen_calibrated_router import (
    MyopicFrozenCalibratedRouter,
)
from src.routers.safe_linucb_router import SafeLinUCBRouter
from src.simulator.environment import SimulationEnvironment


def _action(name: str, payload: str = "roi") -> JointAction:
    return JointAction(name, "balanced", payload, "contact_aware", 0.5, 0.5, payload)


def _metrics(**overrides) -> PredictedMetrics:
    values = {
        "predicted_S_gen_mb": 1.0,
        "predicted_S_down_mb": 1.0,
        "predicted_E_comp": 4.0,
        "predicted_E_comm": 1.0,
        "predicted_L_inf": 2.0,
        "predicted_L_queue": 1.0,
        "predicted_L_comm": 1.0,
        "predicted_Q_next_mb": 2.0,
        "predicted_B_next": 0.8,
        "predicted_E_AI_next": 0.8,
        "predicted_Theta_next": 0.3,
        "predicted_delivery_factor": 1.0,
        "predicted_safety_violation": 0.0,
        "predicted_reward": 0.0,
        "predicted_status": "delivered",
        "predicted_detection_quality": 0.8,
        "raw_info_gain": 0.8,
        "effective_info_gain": 0.8,
        "predicted_product_type": "roi",
    }
    values.update(overrides)
    return PredictedMetrics(**values)


def _router() -> MyopicFrozenCalibratedRouter:
    router = MyopicFrozenCalibratedRouter(MYOPIC_PARAMETER_FILE)
    task = TaskSample(
        task_id="task-1",
        image_path="",
        label_path="",
        scene_id="scene",
        virtual_time=0,
        task_value=0.8,
        deadline_s=5000.0,
        difficulty=0.3,
        uncertainty=0.4,
        arrival_slot=0,
        product_requirement="verification",
    )
    step = TimelineStep(
        slot_index=0,
        slot_seconds=20.0,
        task=task,
        resource_trace=ResourceTrace(0, 0.0, 0.0, 0.0),
        link_trace=LinkTrace(0, True, 10.0, 1.0, 1.0, 20.0),
    )
    router.bind_step(step, None, None)
    return router


def _context() -> ContextVector:
    return ContextVector(0.8, 1.0, 0.2, 0.8, 0.2, 0.3, 0.4, 0.7, 0.1)


def test_myopic_ignores_predicted_reward_and_has_no_bonus():
    router = _router()
    efficient = _action("efficient")
    expensive = _action("expensive")
    selected = router.select_action(
        _context(),
        [expensive, efficient],
        {
            "efficient": _metrics(predicted_reward=-1.0e9, predicted_E_comp=1.0),
            "expensive": _metrics(predicted_reward=1.0e9, predicted_E_comp=80.0),
        },
    )
    assert selected == efficient
    for row in router.last_all_score_components.values():
        assert row.get("exploration_bonus", 0.0) == 0.0
        assert row.get("experience_bonus", 0.0) == 0.0
    router.update(_context(), selected, 999.0)
    assert router.online_update_count == 0


def test_frozen_myopic_preserves_effective_value_above_one():
    router = _router()
    router.current_step = TimelineStep(
        slot_index=0,
        slot_seconds=20.0,
        task=TaskSample(
            task_id="v25-above-one",
            image_path="",
            label_path="",
            scene_id="scene",
            virtual_time=0,
            task_value=0.8,
            deadline_s=120.0,
            difficulty=0.3,
            uncertainty=0.4,
            calibrated_task_value=1.109420044335736,
            value_level="high",
            value_multiplier=10.0,
            product_requirement="verification",
        ),
        resource_trace=ResourceTrace(0, 0.0, 0.0, 0.0),
        link_trace=LinkTrace(0, True, 10.0, 1.0, 1.0, 20.0),
    )
    router.current_task = router.current_step.task
    components = router._score_components(_action("value-check"), _metrics())
    assert components["effective_task_value"] == pytest.approx(1.109420044335736)
    assert components["effective_task_value"] > 1.0
    assert MYOPIC_FROZEN_PARAMETERS["allow_online_update"] is False


def test_default_comparison_is_frozen_five_method_set():
    assert DEFAULT_METHODS == (
        "CORRECTED_REQALIGN",
        "MYOPIC",
        "B1",
        "OFFLOAD",
        "TUT",
    )
    assert METHOD_NAMES["CORRECTED_REQALIGN"] == METHOD_PROP_NEW
    assert METHOD_NAMES["B1"] == "B1-COMMON-THERMAL"
    assert METHOD_NAMES["OFFLOAD"] == "OFFLOAD-EFT-THERMAL-ALIGNED"
    assert METHOD_NAMES["TUT"] == "TUT-UOI-PRODUCT-LITE"
    assert (
        METHOD_NAMES["MYOPIC"]
        == "MYOPIC-FROZEN-CALIBRATED-ON-CORRECTED-REQALIGN"
    )
    assert "PROP" not in METHOD_NAMES
    assert _parse_methods("REQALIGN,MYOPIC") == ("CORRECTED_REQALIGN", "MYOPIC")
    with pytest.raises(ValueError):
        _parse_methods("PROP,MYOPIC")


def test_both_methods_share_corrected_reqalign_backend():
    configs, timeline, _identity = build_shared_scenario(2026)
    delegates = {}
    for method in ("CORRECTED_REQALIGN", "MYOPIC"):
        method_configs = dict(configs)
        environment = SimulationEnvironment(method_configs)
        runner = environment.build_episode_runner(timeline)
        router = build_method_router(
            method,
            method_configs,
            timeline,
            runner,
            environment.action_space,
        )
        assert isinstance(router, RequirementAwareRouterWrapper)
        assert router.config["enabled"] is True
        delegates[method] = router.delegate
    assert isinstance(delegates["CORRECTED_REQALIGN"], SafeLinUCBRouter)
    assert isinstance(delegates["MYOPIC"], MyopicFrozenCalibratedRouter)
    assert delegates["MYOPIC"].params == MYOPIC_FROZEN_PARAMETERS
    assert delegates["MYOPIC"].router_name == "myopic_frozen_calibrated"


def test_default_single_seed_environment_is_frozen_v25_40w_cap050_and_guard_free():
    configs, timeline, identity = build_shared_scenario(2026)
    target = identity["target_environment"]
    assert configs["simulator"]["downlink_capacity_multiplier"] == pytest.approx(0.5)
    assert target["solar"]["configured_peak_w"] == pytest.approx(40.0)
    assert target["solar"]["trace_max_w"] == pytest.approx(40.0)
    assert target["solar"]["sunlit_mean_w"] == pytest.approx(16.0)
    assert target["solar"]["full_day_mean_w"] == pytest.approx(10.0)
    assert target["solar"]["radio_active_draw_w_separate"] == pytest.approx(38.0)
    assert target["communication"]["link_quality_phase_is_capacity_multiplier"] is False
    assert configs["simulator"]["formal_method_extensions"] == {
        "hv_urgent_guard_enabled": False,
        "medium_slack_reclaim_enabled": False,
    }
    assert max(step.link_trace.throughput_mbps for step in timeline) <= 32.5
    task_level_counts = {
        level: sum(
            step.task is not None and step.task.value_level == level
            for step in timeline
        )
        for level in ("low", "medium", "high", "urgent")
    }
    assert task_level_counts == {
        "low": 54,
        "medium": 139,
        "high": 54,
        "urgent": 21,
    }
    task_effective_values = {
        step.task.value_level: step.task.calibrated_task_value
        for step in timeline
        if step.task is not None
    }
    assert task_effective_values == {
        "low": pytest.approx(0.17447916666666666),
        "medium": pytest.approx(0.5234375),
        "high": pytest.approx(1.7447916666666667),
        "urgent": pytest.approx(4.361979166666667),
    }
    assert all(
        step.task.calibrated_task_value is not None
        for step in timeline
        if step.task is not None
    )


def test_value_sensitivity_profiles_only_replace_declared_ratio():
    assert DEFAULT_VALUE_PROFILE == "main"
    assert VALUE_PROFILES["main"]["weights"] == {
        "low": 1.0,
        "medium": 3.0,
        "high": 10.0,
        "urgent": 25.0,
    }
    expected = {
        "mild": {"low": 1.0, "medium": 2.0, "high": 3.0, "urgent": 5.0},
        "stress": {"low": 1.0, "medium": 5.0, "high": 20.0, "urgent": 50.0},
    }
    configure_target_scenario_profile("generated_multiseed")
    try:
        main_configs, main_timeline, main_identity = build_shared_scenario(2026)
        for profile, weights in expected.items():
            configs, timeline, identity = build_shared_scenario(
                2026,
                value_profile=profile,
            )
            assert identity["nominal_value_weights"] == weights
            assert identity["task_manifest_hash_v2"] == main_identity["task_manifest_hash_v2"]
            assert identity["environment_hash_v2"] == main_identity["environment_hash_v2"]
            assert identity["evaluator_identity_hash"] == main_identity["evaluator_identity_hash"]
            assert [
                (step.slot_index, step.task.task_id if step.task else None)
                for step in timeline
            ] == [
                (step.slot_index, step.task.task_id if step.task else None)
                for step in main_timeline
            ]
            assert configs["simulator"] == main_configs["simulator"]
            assert configs["resource"] == main_configs["resource"]
            assert configs["contact"] == main_configs["contact"]
    finally:
        configure_target_scenario_profile("frozen_v25_stress")


def test_metadata_does_not_close_evidence_requirement():
    metadata = {"product_type": "metadata", "ground_available_time_s": 10.0}
    evidence = {"product_type": "roi_context", "ground_available_time_s": 50.0}
    assert _closing_event([metadata], "evidence") is None
    assert _closing_event([metadata, evidence], "evidence") == evidence


def test_full_closure_serializes_compute_generation_and_transmission():
    rows = build_full_closure_rows(
        "MYOPIC-E2E-UTILITY-GREEDY",
        [
            {
                "task_id": "task-1",
                "value_level": "high",
                "product_requirement": "verification",
                "arrival_time_s": 0.0,
                "deadline_time_s": 100.0,
            }
        ],
        {
            "task-1": [
                {
                    "task_id": "task-1",
                    "product_type": "roi",
                    "ground_available_time_s": 0.1,
                }
            ]
        },
        [
            SimpleNamespace(
                task_id="task-1",
                has_new_task=1,
                L_inf=1.5,
                payload_encoding_latency_s=0.2,
            )
        ],
        [
            {
                "lifecycle_id": "pl-1",
                "task_id": "task-1",
                "product_type": "roi",
                "current_status": "delivered",
                "created_slot": 0,
                "delivered_slot": 0,
            }
        ],
        [],
        20.0,
    )
    row = rows[0]
    assert row["closure_success"] == 1
    assert row["required_product_ground_time_s"] == pytest.approx(1.6)
    assert row["full_closure_latency_s"] == pytest.approx(1.6)
    assert row["inference_latency_s"] == pytest.approx(1.3)
    assert row["product_generation_latency_s"] == pytest.approx(0.2)
    assert row["downlink_transmission_s"] == pytest.approx(0.1)
    assert row["component_sum_matches_full_closure"] == 1


def test_unified_metric_latency_uses_full_closure_column_only():
    task_rows = [
        {
            "task_id": "hv",
            "value_level": "high",
            "gross_ground_utility": 2.0,
            "on_time_task_success": 1,
        },
        {
            "task_id": "ordinary",
            "value_level": "medium",
            "gross_ground_utility": 1.0,
            "on_time_task_success": 1,
        },
    ]
    event_rows = [
        {
            "task_id": "hv",
            "product_type": "metadata",
            "valid_before_deadline": 1,
        },
        {
            "task_id": "hv",
            "product_type": "roi_context",
            "valid_before_deadline": 1,
        },
    ]
    closure_rows = [
        {
            "task_id": "hv",
            "closure_success": 1,
            "full_closure_latency_s": 100.0,
        },
        {
            "task_id": "ordinary",
            "closure_success": 1,
            "full_closure_latency_s": 20.0,
        },
    ]
    metrics = build_unified_metrics(
        "MYOPIC-E2E-UTILITY-GREEDY",
        1,
        task_rows,
        event_rows,
        closure_rows,
        {"common_ground_net_utility": 2.7},
        {"controllable_mission_energy_wh": 1.0, "delivered_mb": 2.0},
        {"hard_thermal_violation_count": 0, "hard_battery_violation_count": 0},
    )
    assert metrics["N_HV_closure"] == 1
    assert metrics["N_HV_metadata"] == 1
    assert metrics["N_HV_evidence"] == 1
    assert metrics["mean_closure_latency_s"] == pytest.approx(60.0)
    assert metrics["p95_closure_latency_s"] == pytest.approx(100.0)
    assert metrics["transmitted_bytes"] == pytest.approx(2_000_000.0)
