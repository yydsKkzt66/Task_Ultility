from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.common.types import CommState, LinkTrace, ProductLifecycleState, ResourceState, ResourceTrace, TaskSample, TimelineStep
from src.link.downlink_queue import DownlinkQueue
from src.payload.bounded_guarantee_queue import (
    guarantee_config,
    guarantee_progress_gate,
    rank_guarantee_aware_requests,
)
from src.payload.payload_lifecycle_ledger import PayloadLifecycleLedger
from src.payload.staged_delivery_queue import StagedDeliveryRequest, compute_staged_priority
from src.simulator.step_runner import StepRunner
from scripts.run_deadline_rescue_recheck import (
    LAYERED_DEADLINE_PROFILE,
    apply_layered_deadline_profile,
    counterfactual_rescue_summary_rows,
    counterfactual_selection_trace_rows,
    deadline_audit_rows,
    miss_debt_validation_rows,
)


ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"


def _cfg(**guarantee_overrides) -> dict[str, object]:
    guarantee = {
        "enabled": True,
        "mode": "bounded_virtual_queue",
        "full_method_aliases": [ROUTER],
        "allow_utility_rule": False,
        "eligible_value_levels": ["high", "urgent"],
        "eligible_product_requirements": ["context", "evidence"],
        "lambda_H": 0.35,
        "z_max": 4.0,
        "debt_increment_high": 1.0,
        "debt_increment_urgent": 1.25,
        "progress_credit_min_stage": 1.5,
        "progress_credit_target_stage": 2.0,
        "reset_debt_when_min_stage_reached": True,
        "reset_debt_when_task_closed": True,
        "marginal_utility_floor_main": 0.0,
        "guarantee_window_budget_fraction": 0.30,
        "waiting_limit_windows_for_target_progress": 1,
        "hard_safety_veto": True,
        "low_medium_debt_forbidden": True,
        "roi_cannot_close_evidence": True,
    }
    guarantee.update(guarantee_overrides)
    return {
        "epsilon": 1.0e-9,
        "battery_min": 0.18,
        "thermal_max": 0.92,
        "queue_max_mb": 10.0,
        "slot_seconds": 20.0,
        "battery_capacity_j": 100.0,
        "battery_charge_efficiency": 1.0,
        "battery_discharge_efficiency": 1.0,
        "comm_power_w": 20.0,
        "tx_power_w": 20.0,
        "comm_alpha_gamma": 0.0,
        "downlink_efficiency": 1.0,
        "cooling_per_slot": 0.0,
        "thermal_reference_slot_seconds": 20.0,
        "thermal_passive_cooling": 0.0,
        "thermal_sunlight_gain": 0.0,
        "thermal_eclipse_cooling_bonus": 0.0,
        "thermal_reference_min_c": -10.0,
        "thermal_reference_max_c": 30.0,
        "high_value_threshold": 0.75,
        "guarantee": guarantee,
        "progressive_payload": {
            "value_aware_staged_delivery_queue": {
                "enabled": True,
                "full_method_aliases": [ROUTER],
                "residual_utility_threshold": 0.0,
                "require_future_contact_before_deadline": True,
                "allow_pending_without_future_contact": True,
                "max_admissions_per_slot": 4,
                "max_admissions_per_contact_window": 4,
                "deadline_urgency_lambda": 0.2,
            },
            "roi_context_after_roi": {"enabled": True, "residual_utility_threshold": 0.0},
        },
    }


def _link(slot: int, in_contact: bool = True, throughput_mbps: float = 80.0) -> LinkTrace:
    return LinkTrace(slot, in_contact, 10.0, throughput_mbps if in_contact else 0.0, 1.0, 20.0 if in_contact else 0.0)


def _timeline(slot: int = 0, *, in_contact: bool = True) -> TimelineStep:
    return TimelineStep(
        slot_index=slot,
        slot_seconds=20.0,
        task=None,
        resource_trace=ResourceTrace(slot, 0.0, 0.0, 10.0, is_sunlit=False),
        link_trace=_link(slot, in_contact),
    )


def _task(task_id: str, *, value_level: str = "high", requirement: str = "context") -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_path="mock://image",
        label_path="mock://label",
        scene_id=f"scene_{task_id}",
        virtual_time=0,
        task_value=0.8,
        deadline_s=120.0,
        difficulty=0.1,
        uncertainty=0.2,
        high_value_flag=value_level in {"high", "urgent"},
        emergency_flag=value_level == "urgent",
        product_requirement=requirement,
        value_level=value_level,
        calibrated_task_value=1.5 if value_level in {"high", "urgent"} else 0.4,
    )


def _request(
    task_id: str,
    *,
    product_type: str = "roi",
    value_level: str = "high",
    requirement: str = "context",
    delta: float = 0.4,
    payload: float = 1.0,
) -> StagedDeliveryRequest:
    parent = "metadata" if product_type == "roi" else "roi"
    request = StagedDeliveryRequest(
        request_id=f"req_{task_id}_{product_type}",
        task_id=task_id,
        product_type=product_type,
        parent_product_type=parent,
        product_requirement=requirement,
        value_level=value_level,
        calibrated_task_value=1.2 if value_level in {"high", "urgent"} else 0.4,
        residual_delta_information_utility=delta,
        residual_utility_remaining=max(0.0, delta),
        detection_quality=1.0,
        timeliness_factor=1.0,
        payload_size_mb=payload,
        deadline_slot=2,
        created_slot=0,
        last_evaluated_slot=0,
        earliest_delivery_slot=0,
        duplicate_key=f"{task_id}:{product_type}",
        deadline_s=120.0,
    )
    request.priority_score, request.priority_components = compute_staged_priority(request, current_slot=0, config=_cfg())
    return request


def _runner(cfg: dict[str, object] | None = None) -> StepRunner:
    runner = StepRunner.__new__(StepRunner)
    runner.simulator_cfg = cfg or _cfg()
    runner.reward_cfg = {"high_value_threshold": 0.75}
    runner.payload_profiles_cfg = None
    ledger = PayloadLifecycleLedger(slot_seconds=20.0)
    runner.downlink_queue = DownlinkQueue(10.0, 20.0, lifecycle_ledger=ledger)
    runner.product_lifecycle = {}
    runner.task_snapshots = {}
    runner.future_link_traces = [_link(0, True), _link(1, False), _link(2, True), _link(3, False)]
    runner._roi_context_budget_slot_index = -1
    runner._roi_context_slot_selected_count = 0
    runner._roi_context_queued_budget_slot_index = -1
    runner._roi_context_queued_selected_count = 0
    runner._roi_context_contact_window_id = ""
    runner._roi_context_contact_window_selected_count = 0
    runner._roi_context_window_selected_count = 0
    runner.last_future_contact_feasibility_trace = []
    runner.last_duplicate_suppression_trace = []
    runner._ensure_staged_delivery_state()
    runner._reset_staged_delivery_step_logs()
    return runner


def _feasibility(feasible: bool = True, in_contact: bool = True) -> dict[str, object]:
    return {
        "feasible": int(feasible),
        "in_contact": int(in_contact),
        "capacity_feasible": int(feasible),
        "queue_feasible": int(feasible),
        "battery_feasible": int(feasible),
        "thermal_feasible": int(feasible),
        "duplicate_feasible": int(feasible),
        "deadline_feasible": int(feasible),
        "queue_mb": 0.0,
    }


def _finish_window(runner: StepRunner) -> None:
    runner._update_guarantee_debt_at_contact_window_end(_timeline(0, in_contact=True), SimpleNamespace(router_name=ROUTER))


def test_guarantee_default_off():
    assert guarantee_config({})["enabled"] is False


def test_lambda_zero_matches_no_guarantee_queue():
    cfg = _cfg(lambda_H=0.0)
    req_a = _request("a", delta=0.3, payload=1.0)
    req_b = _request("b", delta=0.5, payload=1.5)
    base = []
    for req in (req_a, req_b):
        priority, components = compute_staged_priority(req, current_slot=1, config=cfg)
        base.append((priority, components["value_density_priority"]))
    rank_guarantee_aware_requests(
        [req_a, req_b],
        current_slot=1,
        config=cfg,
        debt_by_task={"a": 4.0, "b": 4.0},
        delivered_products_by_task={"a": {"metadata"}, "b": {"metadata"}},
    )
    assert [req_a.priority_score, req_b.priority_score] == pytest.approx([row[0] for row in base])


def test_low_medium_tasks_do_not_accumulate_debt():
    runner = _runner()
    request = _request("m", value_level="medium", requirement="context")
    runner.product_lifecycle["m"] = ProductLifecycleState("m", product_requirement="context", delivered_products={"metadata"})
    runner._refresh_staged_delivery_contact_budget(_timeline())
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    _finish_window(runner)
    assert runner.guarantee_debt_by_task.get("m", 0.0) == 0.0


def test_infeasible_high_value_task_does_not_accumulate_debt():
    runner = _runner()
    request = _request("hv")
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner._refresh_staged_delivery_contact_budget(_timeline())
    runner._record_guarantee_opportunity(request, _feasibility(feasible=False), slot=0)
    _finish_window(runner)
    assert runner.guarantee_debt_by_task.get("hv", 0.0) == 0.0


def test_no_contact_slot_does_not_increase_debt():
    runner = _runner()
    request = _request("hv")
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner._refresh_staged_delivery_contact_budget(_timeline(in_contact=False))
    runner._record_guarantee_opportunity(request, _feasibility(in_contact=False), slot=0)
    runner._update_guarantee_debt_at_contact_window_end(_timeline(in_contact=False), SimpleNamespace(router_name=ROUTER))
    assert runner.guarantee_debt_by_task.get("hv", 0.0) == 0.0


def test_debt_grows_after_missed_feasible_contact_window():
    runner = _runner()
    request = _request("hv")
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner._refresh_staged_delivery_contact_budget(_timeline())
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    _finish_window(runner)
    assert runner.guarantee_debt_by_task["hv"] > 0.0


def test_debt_decreases_or_resets_after_progress():
    runner = _runner()
    runner.guarantee_debt_by_task["hv"] = 2.0
    request = _request("hv")
    state = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner.product_lifecycle["hv"] = state
    runner._refresh_staged_delivery_contact_budget(_timeline())
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    state.delivered_products.add("roi")
    _finish_window(runner)
    assert runner.guarantee_debt_by_task["hv"] == 0.0


def test_debt_is_bounded_by_z_max():
    runner = _runner(_cfg(z_max=4.0))
    runner.guarantee_debt_by_task["hv"] = 3.9
    request = _request("hv")
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner._refresh_staged_delivery_contact_budget(_timeline())
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    _finish_window(runner)
    assert runner.guarantee_debt_by_task["hv"] <= 4.0


def test_guarantee_debt_does_not_bypass_hard_safety():
    runner = _runner()
    task = _task("hv")
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner.guarantee_debt_by_task["hv"] = 4.0
    runner.staged_delivery_requests["hv:roi"] = _request("hv", payload=2.0)
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(9.5, 10.0),
        router=SimpleNamespace(router_name=ROUTER),
    )
    assert runner.last_contact_window_scheduling_trace[-1]["decision"] == "blocked_by_hard_safety_guard"


def test_guarantee_candidate_respects_marginal_utility_floor():
    runner = _runner()
    task = _task("hv")
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner.guarantee_debt_by_task["hv"] = 4.0
    runner.staged_delivery_requests["hv:roi"] = _request("hv", delta=-0.1)
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name=ROUTER),
    )
    assert runner.last_contact_window_scheduling_trace[-1]["decision"] == "marginal_utility_floor_blocked"


def test_guarantee_window_budget_cap():
    runner = _runner(_cfg(guarantee_window_budget_fraction=0.05))
    task = _task("hv")
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner.guarantee_debt_by_task["hv"] = 4.0
    runner.staged_delivery_requests["hv:roi"] = _request("hv", payload=1.0)
    low_capacity_step = TimelineStep(
        slot_index=0,
        slot_seconds=20.0,
        task=None,
        resource_trace=ResourceTrace(0, 0.0, 0.0, 10.0, is_sunlit=False),
        link_trace=_link(0, True, throughput_mbps=0.2),
    )
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=low_capacity_step,
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name=ROUTER),
    )
    assert runner.last_contact_window_scheduling_trace[-1]["decision"] == "guarantee_budget_limited"


def test_roi_does_not_close_evidence():
    runner = _runner()
    state = ProductLifecycleState("e", product_requirement="evidence", delivered_products={"roi"})
    assert not runner._integrated_branch_closure_satisfied(state)


def test_verification_task_not_forced_to_roi_context():
    request = _request("v", product_type="roi_context", requirement="verification")
    assert guarantee_progress_gate(request, delivered_products={"roi"}, config=_cfg()) == 0.0


def test_duplicate_product_gets_zero_marginal_information_gain():
    request = _request("dup", product_type="roi_context", requirement="evidence")
    assert guarantee_progress_gate(request, delivered_products={"roi", "roi_context"}, config=_cfg()) == 0.0


def test_utility_rule_unchanged():
    runner = _runner()
    assert not runner._guarantee_enabled(SimpleNamespace(router_name="utility_rule"))


def test_legacy_default_unchanged():
    assert guarantee_config({})["enabled"] is False


def test_lifecycle_orphan_check_still_zero():
    ledger = PayloadLifecycleLedger(slot_seconds=20.0)
    queue = DownlinkQueue(10.0, 20.0, lifecycle_ledger=ledger)
    assert ledger.orphan_check_rows(queue_items=queue.items, staged_requests=[]) == []


def test_pending_payload_still_recompetes_next_window():
    runner = _runner()
    task = _task("hv")
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner.staged_delivery_requests["hv:roi"] = _request("hv")
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name=ROUTER),
    )
    assert any(row["task_id"] == "hv" and row["reevaluated_flag"] for row in runner.lifecycle_ledger.recompetition_trace)


def test_guarantee_trace_has_required_fields():
    runner = _runner()
    request = _request("hv")
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner._refresh_staged_delivery_contact_budget(_timeline())
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    _finish_window(runner)
    row = runner.last_guarantee_debt_window_update_trace[-1]
    for field in {"task_id", "debt_before", "debt_after", "min_stage_miss", "target_stage_miss"}:
        assert field in row


def test_unique_task_metrics_not_only_evaluation_count():
    runner = _runner()
    request = _request("hv")
    runner.product_lifecycle["hv"] = ProductLifecycleState("hv", product_requirement="context", delivered_products={"metadata"})
    runner._refresh_staged_delivery_contact_budget(_timeline())
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    _finish_window(runner)
    assert len(runner.last_guarantee_opportunity_trace) == 2
    assert len(runner.last_guarantee_debt_window_update_trace) == 1


def test_waiting_age_debt_increases_for_pending_high_value_task():
    runner = _runner()
    task = _task("age")
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle["age"] = ProductLifecycleState("age", product_requirement="context", delivered_products={"metadata"})
    runner.staged_delivery_requests["age:roi"] = _request("age")
    runner.future_link_traces = [_link(0, False), _link(1, True), _link(2, False)]
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(0, in_contact=False),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name=ROUTER),
    )
    assert runner.guarantee_age_debt_by_task["age"] > 0.0


def test_waiting_age_debt_does_not_increase_without_future_contact():
    runner = _runner()
    task = _task("nofuture")
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle["nofuture"] = ProductLifecycleState("nofuture", product_requirement="context", delivered_products={"metadata"})
    runner.staged_delivery_requests["nofuture:roi"] = _request("nofuture")
    runner.future_link_traces = [_link(0, False), _link(1, False)]
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(0, in_contact=False),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name=ROUTER),
    )
    assert runner.guarantee_age_debt_by_task.get("nofuture", 0.0) == 0.0


def test_missed_opportunity_debt_increases_after_feasible_window_miss():
    test_debt_grows_after_missed_feasible_contact_window()


def test_no_contact_slot_does_not_increase_missed_opportunity_debt():
    runner = _runner()
    request = _request("miss")
    runner.product_lifecycle["miss"] = ProductLifecycleState("miss", product_requirement="context", delivered_products={"metadata"})
    runner._refresh_staged_delivery_contact_budget(_timeline(in_contact=False))
    runner._record_guarantee_opportunity(request, _feasibility(in_contact=False), slot=0)
    runner._update_guarantee_debt_at_contact_window_end(_timeline(in_contact=False), SimpleNamespace(router_name=ROUTER))
    assert runner.guarantee_miss_debt_by_task.get("miss", 0.0) == 0.0


def test_low_medium_task_never_gets_age_or_miss_debt():
    test_low_medium_tasks_do_not_accumulate_debt()


def test_verification_task_not_forced_to_roi_context_alias():
    test_verification_task_not_forced_to_roi_context()


def test_positive_debt_persists_to_next_window():
    runner = _runner()
    task = _task("carry")
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle["carry"] = ProductLifecycleState("carry", product_requirement="context", delivered_products={"metadata"})
    runner.staged_delivery_requests["carry:roi"] = _request("carry")
    runner._update_guarantee_age_debt_for_waiting_tasks(_timeline(0, in_contact=False), SimpleNamespace(router_name=ROUTER))
    assert runner.guarantee_debt_by_task["carry"] > 0.0
    runner._refresh_staged_delivery_contact_budget(_timeline(2, in_contact=True))
    assert runner.guarantee_debt_by_task["carry"] > 0.0


def test_scheduler_reads_positive_carried_debt():
    cfg = _cfg(lambda_H=0.35)
    req = _request("read", product_type="roi_context", requirement="evidence")
    rank_guarantee_aware_requests(
        [req],
        current_slot=1,
        config=cfg,
        debt_by_task={"read": 1.0},
        debt_components_by_task={"read": {"Z_age": 0.5, "Z_miss": 0.5}},
        waiting_age_slots_by_task={"read": 10},
        delivered_products_by_task={"read": {"roi"}},
    )
    assert req.priority_components["Z_total"] == pytest.approx(1.0)


def test_positive_Z_and_positive_G_create_positive_guarantee_term():
    cfg = _cfg(lambda_H=0.5)
    req = _request("term", product_type="roi_context", requirement="evidence")
    rank_guarantee_aware_requests(
        [req],
        current_slot=1,
        config=cfg,
        debt_by_task={"term": 2.0},
        debt_components_by_task={"term": {"Z_age": 1.0, "Z_miss": 1.0}},
        delivered_products_by_task={"term": {"roi"}},
    )
    assert req.priority_components["guarantee_term"] > 0.0


def test_priority_recomputed_each_scheduler_step():
    cfg = _cfg(lambda_H=1.0)
    req = _request("recompute", product_type="roi_context", requirement="evidence")
    rank_guarantee_aware_requests(
        [req],
        current_slot=1,
        config=cfg,
        debt_by_task={"recompute": 0.0},
        debt_components_by_task={"recompute": {"Z_age": 0.0, "Z_miss": 0.0}},
        delivered_products_by_task={"recompute": {"roi"}},
    )
    before = req.priority_score
    rank_guarantee_aware_requests(
        [req],
        current_slot=2,
        config=cfg,
        debt_by_task={"recompute": 2.0},
        debt_components_by_task={"recompute": {"Z_age": 1.0, "Z_miss": 1.0}},
        delivered_products_by_task={"recompute": {"roi"}},
    )
    assert req.priority_score > before


def test_final_scheduler_uses_priority_with_guarantee_when_enabled():
    cfg = _cfg(lambda_H=2.0)
    low = _request("low", value_level="low", requirement="verification", delta=0.9, payload=0.5)
    high = _request("high", product_type="roi_context", requirement="evidence", delta=0.1, payload=2.0)
    ranked = rank_guarantee_aware_requests(
        [low, high],
        current_slot=1,
        config=cfg,
        debt_by_task={"high": 4.0},
        debt_components_by_task={"high": {"Z_age": 2.0, "Z_miss": 2.0}},
        delivered_products_by_task={"high": {"roi"}, "low": {"metadata"}},
    )
    assert sorted(ranked, key=lambda req: -req.priority_score)[0].task_id == "high"


def test_progress_to_roi_reduces_min_stage_debt():
    runner = _runner(_cfg(reset_debt_when_min_stage_reached=False))
    runner.guarantee_age_debt_by_task["p"] = 1.0
    runner.guarantee_miss_debt_by_task["p"] = 2.0
    request = _request("p")
    state = ProductLifecycleState("p", product_requirement="context", delivered_products={"metadata"})
    runner.product_lifecycle["p"] = state
    runner._refresh_staged_delivery_contact_budget(_timeline())
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    before = runner._sync_guarantee_total_debt("p")
    state.delivered_products.add("roi")
    _finish_window(runner)
    assert runner.guarantee_debt_by_task["p"] < before


def test_progress_to_roi_context_clears_debt():
    runner = _runner()
    runner.guarantee_age_debt_by_task["p2"] = 1.0
    runner.guarantee_miss_debt_by_task["p2"] = 2.0
    request = _request("p2", product_type="roi_context", requirement="evidence")
    state = ProductLifecycleState("p2", product_requirement="evidence", delivered_products={"roi"})
    runner.product_lifecycle["p2"] = state
    runner._refresh_staged_delivery_contact_budget(_timeline())
    rank_guarantee_aware_requests([request], current_slot=0, config=_cfg(), debt_by_task={"p2": 3.0}, delivered_products_by_task={"p2": {"roi"}})
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    state.delivered_products.add("roi_context")
    _finish_window(runner)
    assert runner.guarantee_debt_by_task["p2"] == 0.0


def test_expired_task_archives_debt_with_reason():
    runner = _runner()
    runner.guarantee_age_debt_by_task["exp"] = 1.0
    record = runner.lifecycle_ledger.create_record(task_id="exp", product_type="roi", slot=0, value_level="high", product_requirement="context")
    runner.lifecycle_ledger.transition(record.lifecycle_id, "expired", slot=1, reason="deadline_passed")
    runner._archive_terminal_guarantee_debt(_timeline(1, in_contact=False))
    assert any(row["debt_update_reason"] == "expired_with_debt" for row in runner.last_task_debt_trajectory_trace)


def test_lambda_zero_has_zero_guarantee_term():
    test_lambda_zero_matches_no_guarantee_queue()


def test_roi_counts_as_min_stage_support_but_not_evidence_closure():
    test_roi_does_not_close_evidence()


def test_metric_comparability_between_no_guarantee_and_lambda_zero():
    test_lambda_zero_matches_no_guarantee_queue()


def test_hard_safety_vetoes_guarantee():
    test_guarantee_debt_does_not_bypass_hard_safety()


def test_priority_trace_contains_age_miss_debt_fields():
    runner = _runner()
    task = _task("trace")
    runner.task_snapshots[task.task_id] = task
    runner.product_lifecycle["trace"] = ProductLifecycleState("trace", product_requirement="context", delivered_products={"metadata"})
    runner.staged_delivery_requests["trace:roi"] = _request("trace")
    runner._process_staged_delivery_queue_at_slot(
        timeline_step=_timeline(0),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        router=SimpleNamespace(router_name=ROUTER),
    )
    row = runner.last_candidate_priority_trace[-1]
    assert {"Z_age", "Z_miss", "Z_total"}.issubset(row)


def test_rescuable_opportunity_requires_future_window():
    runner = _runner()
    runner.future_link_traces = [_link(0, True), _link(1, False)]
    request = _request("nr")
    runner.product_lifecycle["nr"] = ProductLifecycleState("nr", product_requirement="context", delivered_products={"metadata"})
    runner._refresh_staged_delivery_contact_budget(_timeline())
    runner._record_guarantee_opportunity(request, _feasibility(), slot=0)
    _finish_window(runner)
    assert runner.last_guarantee_debt_window_update_trace[-1]["non_rescuable"] == 1


def test_self_check_report_identifies_first_zero_in_activation_chain():
    from scripts.run_bounded_guarantee_virtual_queue import _activation_funnel_rows

    run = {
        "scenario": "s",
        "seed": 1,
        "arm": "proposed_guarantee_aware_full",
        "lambda_H": 0.35,
        "beta_H": 0.30,
        "num_slots": 1,
        "duration_s": 20.0,
        "rows": [],
        "task_debt_trajectory": [],
        "candidate_priority": [],
        "budget": [],
    }
    row = _activation_funnel_rows([run])[0]
    assert row["first_zero_activation_stage"] == "high_value_context_evidence_task_count"


def _counterfactual_run(candidate_row: dict[str, object], *, delivered: bool = True) -> dict[str, object]:
    return {
        "scenario": "multi_window_rescuable_high_value",
        "seed": 2026,
        "arm": "proposed_guarantee_aware_full",
        "lambda_H": 0.35,
        "beta_H": 0.30,
        "num_slots": 160,
        "duration_s": 3200.0,
        "deadline_profile": "layered_by_requirement_and_value",
        "candidate_priority": [candidate_row],
        "queue_priority": [
            {
                "task_id": candidate_row.get("task_id", "hv"),
                "payload_type": candidate_row.get("product_type", "roi"),
                "delivered_this_slot": int(delivered),
            }
        ],
        "lifecycle": [],
    }


def _candidate_row(**overrides) -> dict[str, object]:
    row = {
        "slot": 1,
        "contact_window_id": "w1",
        "task_id": "hv",
        "product_type": "roi",
        "value_level": "high",
        "product_requirement": "context",
        "rank_without_guarantee": 1,
        "rank_with_guarantee": 1,
        "capacity_cutoff_rank": 1,
        "selected": 1,
        "guarantee_term": 0.5,
        "deltaU_total": 1.0,
        "blocked_by_safety": 0,
        "blocked_by_deadline": 0,
        "blocked_by_duplicate": 0,
        "blocked_by_marginal_utility_floor": 0,
    }
    row.update(overrides)
    return row


def test_natural_delivery_not_counted_as_guarantee_rescue():
    trace = counterfactual_selection_trace_rows(_counterfactual_run(_candidate_row()))
    assert trace[0]["naturally_delivered"] == 1
    assert trace[0]["counterfactual_guarantee_rescued_delivered"] == 0


def test_counterfactual_rescue_requires_without_false_with_true():
    trace = counterfactual_selection_trace_rows(
        _counterfactual_run(_candidate_row(rank_without_guarantee=2, rank_with_guarantee=1))
    )
    assert trace[0]["selected_without_guarantee"] == 0
    assert trace[0]["selected_with_guarantee"] == 1
    assert trace[0]["counterfactual_guarantee_rescued_delivered"] == 1


def test_guarantee_positive_delivery_not_equal_rescued_delivery():
    trace = counterfactual_selection_trace_rows(_counterfactual_run(_candidate_row()))
    summary = counterfactual_rescue_summary_rows(trace)[0]
    assert summary["guarantee_positive_delivered_count"] == 1
    assert summary["counterfactual_guarantee_rescued_delivered_count"] == 0


def test_rank_improvement_computed_correctly():
    trace = counterfactual_selection_trace_rows(
        _counterfactual_run(_candidate_row(rank_without_guarantee=3, rank_with_guarantee=2, capacity_cutoff_rank=3))
    )
    assert trace[0]["guarantee_rank_improved"] == 1


def test_capacity_cutoff_crossing_computed_correctly():
    trace = counterfactual_selection_trace_rows(
        _counterfactual_run(_candidate_row(rank_without_guarantee=2, rank_with_guarantee=1, capacity_cutoff_rank=1))
    )
    assert trace[0]["guarantee_crossed_capacity_cutoff"] == 1


def _deadline_test_timeline(task: TaskSample) -> list[TimelineStep]:
    contact_slots = {2, 3, 10, 11, 20, 21, 28, 29}
    timeline = []
    for slot in range(32):
        timeline.append(
            TimelineStep(
                slot_index=slot,
                slot_seconds=20.0,
                task=task if slot == 0 else None,
                resource_trace=ResourceTrace(slot, 0.0, 0.0, 10.0, is_sunlit=False),
                link_trace=_link(slot, slot in contact_slots, throughput_mbps=8.0),
            )
        )
    return timeline


def test_deadline_audit_outputs_contact_windows_before_deadline():
    rows = deadline_audit_rows(_deadline_test_timeline(_task("d")), scenario="s", seed=1, num_slots=32, deadline_profile="legacy")
    assert "contact_windows_before_deadline" in rows[0]


def test_layered_deadline_profile_gives_context_evidence_multiple_windows():
    timeline, _mapping = apply_layered_deadline_profile(_deadline_test_timeline(_task("ctx", requirement="context")), LAYERED_DEADLINE_PROFILE)
    rows = deadline_audit_rows(timeline, scenario="multi", seed=1, num_slots=32, deadline_profile="layered")
    assert rows[0]["contact_windows_before_deadline"] >= 2


def test_alert_deadline_can_be_shorter_than_evidence():
    alert_timeline, _ = apply_layered_deadline_profile(_deadline_test_timeline(_task("alert", requirement="alert")), LAYERED_DEADLINE_PROFILE)
    evidence_timeline, _ = apply_layered_deadline_profile(_deadline_test_timeline(_task("evidence", requirement="evidence")), LAYERED_DEADLINE_PROFILE)
    assert alert_timeline[0].task.deadline_s <= evidence_timeline[0].task.deadline_s


def test_window_count_deadline_mapping_no_future_reward_leakage():
    timeline, mapping = apply_layered_deadline_profile(_deadline_test_timeline(_task("map", requirement="evidence")), LAYERED_DEADLINE_PROFILE)
    assert mapping[0]["mapping_source"] == "deterministic_contact_schedule_no_reward_or_delivery_outcome"
    assert timeline[0].task.metadata["deadline_profile_mode"] == "layered_by_requirement_and_value"


def test_multi_window_rescuable_scenario_has_post_miss_future_window():
    timeline, _ = apply_layered_deadline_profile(_deadline_test_timeline(_task("mw", requirement="context")), LAYERED_DEADLINE_PROFILE)
    rows = deadline_audit_rows(timeline, scenario="multi", seed=1, num_slots=32, deadline_profile="layered")
    assert rows[0]["contact_windows_after_first_miss"] > 0


def _miss_validation_fixture():
    run = {
        "scenario": "multi_window_rescuable_high_value",
        "seed": 2026,
        "arm": "proposed_guarantee_aware_full",
        "lambda_H": 0.35,
        "num_slots": 160,
        "deadline_profile": "layered_by_requirement_and_value",
        "summary": {"final_resource_safety_violation_count": 0},
        "debt_update": [
            {
                "task_id": "hv",
                "min_stage_miss": 1,
                "target_stage_miss": 0,
                "future_contact_before_deadline": 1,
                "debt_increment": 1.0,
            }
        ],
        "task_debt_trajectory": [{"task_id": "hv", "Z_miss": 1.0}],
        "candidate_priority": [
            _candidate_row(task_id="hv", product_type="roi", Z_miss=1.0, rank_without_guarantee=2, rank_with_guarantee=1)
        ],
        "queue_priority": [{"task_id": "hv", "payload_type": "roi", "delivered_this_slot": 1}],
        "lifecycle": [],
    }
    global_rows = [
        {
            "scenario": "multi_window_rescuable_high_value",
            "seed": 2026,
            "num_slots": 160,
            "arm": "proposed_persistent_queue_without_guarantee",
            "delivered_effective_utility": 10.0,
        },
        {
            "scenario": "multi_window_rescuable_high_value",
            "seed": 2026,
            "num_slots": 160,
            "arm": "proposed_guarantee_aware_full",
            "lambda_H": 0.35,
            "delivered_effective_utility": 9.8,
        },
    ]
    cf_rows = counterfactual_selection_trace_rows(run)
    return run, cf_rows, global_rows


def test_miss_debt_positive_in_multi_window_scenario():
    run, cf_rows, global_rows = _miss_validation_fixture()
    _trace, summary = miss_debt_validation_rows([run], cf_rows, global_rows)
    assert summary[0]["miss_debt_positive_task_count"] == 1


def test_post_miss_candidate_enters_scheduler():
    run, cf_rows, global_rows = _miss_validation_fixture()
    _trace, summary = miss_debt_validation_rows([run], cf_rows, global_rows)
    assert summary[0]["post_miss_candidate_count"] == 1


def test_miss_debt_can_create_positive_guarantee_term():
    test_positive_Z_and_positive_G_create_positive_guarantee_term()


def test_low_medium_debt_forbidden():
    test_low_medium_task_never_gets_age_or_miss_debt()


def test_roi_min_stage_but_not_evidence_closure():
    test_roi_counts_as_min_stage_support_but_not_evidence_closure()


def test_lambda_zero_equivalence_still_passes():
    test_lambda_zero_matches_no_guarantee_queue()


def test_hard_safety_still_zero():
    test_guarantee_debt_does_not_bypass_hard_safety()
