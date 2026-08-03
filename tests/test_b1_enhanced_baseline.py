from __future__ import annotations

import pytest

from scripts.run_b1_enhanced_contact_compressed_baseline import run as run_b1
from src.baselines.b1_enhanced import (
    B1EnhancedConfig,
    B1EnhancedSimulator,
    B1TaskState,
    ContactAwareCompressedGroundScheduler,
    compression_energy_wh,
    compression_latency_s,
    compressed_size_mb,
)
from src.common.types import LinkTrace, ResourceState, ResourceTrace, TaskSample, TimelineStep
from src.evaluation.baselines import build_routers


def _task(
    task_id: str,
    *,
    value: float = 0.8,
    deadline_s: float = 120.0,
    raw_size_mb: float = 32.0,
    slot: int = 0,
) -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_path=f"mock://{task_id}.png",
        label_path=f"mock://{task_id}.txt",
        scene_id="scene",
        virtual_time=slot,
        task_value=value,
        deadline_s=deadline_s,
        difficulty=0.2,
        uncertainty=0.2,
        metadata={"raw_payload_mb": raw_size_mb},
        arrival_slot=slot,
        high_value_flag=value >= 0.75,
    )


def _step(slot: int, task: TaskSample | None = None, *, in_contact: bool = False, throughput: float = 8.0) -> TimelineStep:
    return TimelineStep(
        slot_index=slot,
        slot_seconds=10.0,
        task=task,
        resource_trace=ResourceTrace(
            slot_index=slot,
            solar_energy_j=100.0,
            base_power_w=0.1,
            temperature_reference_c=5.0,
        ),
        link_trace=LinkTrace(
            slot_index=slot,
            in_contact=in_contact,
            snr_db=10.0 if in_contact else -3.0,
            throughput_mbps=throughput if in_contact else 0.0,
            link_quality=1.0 if in_contact else 0.0,
            remaining_window_s=10.0 if in_contact else 0.0,
        ),
    )


def _sim_cfg() -> dict[str, float]:
    return {
        "battery_init": 0.85,
        "thermal_init": 0.25,
        "battery_min": 0.18,
        "thermal_max": 0.92,
        "queue_max_mb": 64.0,
        "slot_seconds": 10.0,
        "battery_capacity_j": 35.0 * 3600.0,
        "battery_charge_efficiency": 1.0,
        "battery_discharge_efficiency": 1.0,
        "comm_power_w": 8.0,
        "tx_power_w": 8.0,
        "comm_alpha_gamma": 0.0,
        "cooling_per_slot": 0.0,
        "thermal_passive_cooling": 0.0,
        "thermal_sunlight_gain": 0.0,
        "thermal_eclipse_cooling_bonus": 0.0,
    }


def _model_pool() -> dict[str, object]:
    return {
        "profiles": [
            {
                "name": "medium",
                "energy_j": 53.2,
                "latency_s": 2.8,
                "quality_base": 0.88,
                "robustness_to_difficulty": 0.82,
                "enabled": True,
            }
        ],
        "models": [{"name": "high_accuracy", "backend": "mock", "profile_name": "medium", "enabled": True}],
    }


def _state(task_id: str, *, value: float, size: float, deadline: float, queued_at: float = 0.0) -> B1TaskState:
    task = _task(task_id, value=value, deadline_s=deadline, raw_size_mb=size * 8.0)
    return B1TaskState(
        task=task,
        created_slot=0,
        capture_time_s=0.0,
        deadline_time_s=deadline,
        raw_size_mb=size * 8.0,
        compressed_size_mb=size,
        compression_latency_s=1.0,
        compression_energy_wh=0.01,
        reconstructed_image_quality_proxy=0.96,
        ground_task_quality=0.9,
        state="compressed_queued",
        queue_enter_time_s=queued_at,
        remaining_downlink_mb=size,
    )


def test_compression_ratio_direction_and_energy_units():
    cfg = B1EnhancedConfig().compression
    assert compressed_size_mb(32.0, 8.0) == pytest.approx(4.0)
    assert compression_latency_s(32.0, cfg) == pytest.approx(11.0)
    assert compression_energy_wh(11.0, 6.8) == pytest.approx(6.8 * 11.0 / 3600.0)


def test_lifecycle_blocks_queue_and_ground_until_ready():
    simulator = B1EnhancedSimulator(B1EnhancedConfig(), _sim_cfg(), _model_pool())
    result = simulator.run([_step(0, _task("t0"), in_contact=True, throughput=32.0)])
    row = result["per_task_delivery_trace"][0]
    assert row["payload_type"] == "compressed_full_image"
    assert row["ground_result_available"] == 0
    assert row["state"] in {"compressing", "compressed_queued"}


def test_complete_downlink_waits_for_ground_decode_and_inference():
    simulator = B1EnhancedSimulator(B1EnhancedConfig(), _sim_cfg(), _model_pool())
    timeline = [
        _step(0, _task("t0"), in_contact=False),
        _step(1, None, in_contact=True, throughput=64.0),
        _step(2, None, in_contact=True, throughput=64.0),
    ]
    result = simulator.run(timeline)
    row = result["per_task_delivery_trace"][0]
    assert row["ground_result_available"] == 1
    assert float(row["ground_result_available_time_s"]) > float(row["ground_received_compressed_time_s"])
    assert "accuracy" not in row
    assert row["reconstructed_image_quality_proxy"] == pytest.approx(0.96)


def test_scheduler_prefers_high_value_density_then_tight_deadline():
    scheduler = ContactAwareCompressedGroundScheduler(B1EnhancedConfig().scheduler)
    link = LinkTrace(0, True, 10.0, 64.0, 1.0, 10.0)
    resource = ResourceState(0.85, 1.0, 0.25)
    selected, _ = scheduler.select(
        [_state("low_density", value=0.9, size=8.0, deadline=100.0), _state("high_density", value=0.7, size=2.0, deadline=100.0)],
        0.0,
        link,
        resource,
        _sim_cfg(),
        80.0,
    )
    assert selected is not None
    assert selected.task.task_id == "high_density"

    selected, _ = scheduler.select(
        [_state("later", value=0.7, size=2.0, deadline=100.0), _state("earlier", value=0.7, size=2.0, deadline=50.0)],
        0.0,
        link,
        resource,
        _sim_cfg(),
        80.0,
    )
    assert selected is not None
    assert selected.task.task_id == "earlier"


def test_require_full_fit_rejects_candidate():
    scheduler = ContactAwareCompressedGroundScheduler(B1EnhancedConfig().scheduler)
    selected, decisions = scheduler.select(
        [_state("too_big", value=1.0, size=20.0, deadline=100.0)],
        0.0,
        LinkTrace(0, True, 10.0, 8.0, 1.0, 10.0),
        ResourceState(0.85, 1.0, 0.25),
        _sim_cfg(),
        10.0,
    )
    assert selected is None
    assert decisions[0].reason == "no_full_fit_current_contact"


def test_non_preemptive_transmission_does_not_switch():
    simulator = B1EnhancedSimulator(B1EnhancedConfig(), _sim_cfg(), _model_pool())
    timeline = [
        _step(0, _task("first", value=0.9, raw_size_mb=32.0), in_contact=False),
        _step(1, _task("second", value=1.0, raw_size_mb=32.0), in_contact=False),
        _step(2, None, in_contact=True, throughput=16.0),
        _step(2, None, in_contact=True, throughput=16.0),
    ]
    result = simulator.run(timeline)
    schedule = result["contact_window_schedule"]
    selected = [row for row in schedule if row["selected"] == 1]
    assert selected
    assert selected[0]["task_id"] == "first"


def test_candidate_competition_trace_is_observer_only():
    simulator = B1EnhancedSimulator(B1EnhancedConfig(), _sim_cfg(), _model_pool())
    timeline = [
        _step(0, _task("first", value=0.9, raw_size_mb=32.0), in_contact=False),
        _step(1, _task("second", value=1.0, raw_size_mb=32.0), in_contact=False),
        _step(2, None, in_contact=True, throughput=16.0),
    ]
    result = simulator.run(timeline)
    selected = [row for row in result["contact_window_schedule"] if row["selected"] == 1]
    trace = result["scheduler_candidate_trace"]
    assert selected[0]["task_id"] == "first"
    assert trace[0]["queued_candidate_count_before_filter"] == 2
    assert trace[0]["eligible_candidate_count_before_feasibility"] == 1
    assert trace[0]["feasible_candidate_count_after_filter"] == 1
    assert trace[0]["selected_task_id"] == "first"


def test_b1_router_registration_does_not_enable_proposed_controls():
    from src.actions.action_space import ActionSpace

    routers = build_routers({"routers": ["b1_enhanced"]}, ActionSpace())
    router = routers[0]
    assert router.router_name == "b1_enhanced_contact_compressed"
    assert not router.use_lyapunov_pricing
    assert not router.use_mode_controller
    assert not router.use_guided_utility


def test_runner_refuses_to_overwrite_existing_output(tmp_path: pytest.TempPathFactory):
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        run_b1(output_dir=out)


def test_deterministic_repeatability(tmp_path: pytest.TempPathFactory):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    result1 = run_b1(output_dir=out1, seed=2026)
    result2 = run_b1(output_dir=out2, seed=2026)
    assert result1["summary"]["ground_received_task_count"] == result2["summary"]["ground_received_task_count"]
    manifest1 = (out1 / "fairness_manifest.json").read_text(encoding="utf-8")
    manifest2 = (out2 / "fairness_manifest.json").read_text(encoding="utf-8")
    assert '"task_manifest_hash"' in manifest1
    assert '"contact_schedule_hash"' in manifest1
    assert '"resource_trace_hash"' in manifest1
    assert manifest1.replace(str(out1), "") == manifest2.replace(str(out2), "")


def test_energy_accounting_no_double_soc_deduction():
    simulator = B1EnhancedSimulator(B1EnhancedConfig(), _sim_cfg(), _model_pool())
    result = simulator.run([_step(0, _task("t0"), in_contact=False)])
    soc_row = result["soc_trace"][0]
    assert float(soc_row["compression_energy_wh"]) > 0.0
    assert result["summary"]["total_compression_energy_wh"] == pytest.approx(float(soc_row["compression_energy_wh"]))


def test_v2_latency_components_and_audits_are_consistent():
    simulator = B1EnhancedSimulator(B1EnhancedConfig(), _sim_cfg(), _model_pool())
    result = simulator.run([_step(0, _task("t0"), in_contact=False), _step(1, None, in_contact=False), _step(2, None, in_contact=True, throughput=16.0)])
    completed = [row for row in result["per_task_delivery_trace"] if row["ground_result_available"]]
    assert completed
    row = completed[0]
    assert row["capture_to_ground_latency_s"] == pytest.approx(row["latency_component_sum_s"])
    assert row["ground_received_quality"] != row["reconstructed_image_quality_proxy"]
    assert result["summary"]["compression_energy_consistency_error_wh"] == pytest.approx(0.0)
    assert any(row["parameter"] == "compression_power_w" and row["source_type"] == "engineering_assumed" for row in result["compression_parameter_audit"])


def test_fifo_policy_is_a_distinct_scheduler_rule():
    enhanced = ContactAwareCompressedGroundScheduler(B1EnhancedConfig().scheduler)
    fifo_cfg = B1EnhancedConfig().scheduler.__class__(policy="fifo")
    fifo = ContactAwareCompressedGroundScheduler(fifo_cfg)
    first = _state("old_low", value=0.2, size=2.0, deadline=100.0); first.queue_enter_time_s = 0.0
    second = _state("new_high", value=1.0, size=2.0, deadline=100.0); second.queue_enter_time_s = 10.0
    link = LinkTrace(0, True, 10.0, 8.0, 1.0, 10.0); resource = ResourceState(0.85, 1.0, 0.25)
    assert enhanced.select([first, second], 10.0, link, resource, _sim_cfg(), 80.0)[0].task.task_id == "new_high"
    assert fifo.select([first, second], 10.0, link, resource, _sim_cfg(), 80.0)[0].task.task_id == "old_low"
