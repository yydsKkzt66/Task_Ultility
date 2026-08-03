from dataclasses import dataclass
from types import SimpleNamespace

from src.routers.ordinary_net_utility_projection import (
    OrdinaryNetCandidate,
    SameSlotSurplusFacts,
    choose_ordinary_net_utility_action,
    same_slot_surplus_certificate,
)


@dataclass(frozen=True)
class Action:
    action_name: str


def candidate(
    name: str,
    *,
    model: str = "balanced",
    net: float = 1.0,
    payload: float = 1.0,
    energy: float = 10.0,
    latency: float = 2.0,
    safe: bool = True,
    deadline: bool = True,
    defer: bool = False,
) -> OrdinaryNetCandidate:
    action = Action(name)
    return OrdinaryNetCandidate(
        action=action,
        action_name=name,
        model_name=model,
        product="none" if defer else "roi",
        predicted_net_utility=net,
        predicted_gross_utility=max(net, 0.0),
        predicted_energy_cost=max(-net, 0.0),
        payload_mb=payload,
        total_energy_j=energy,
        compute_latency_s=latency,
        predicted_ground_arrival_s=None if defer else 100.0,
        hard_safe=safe,
        deadline_feasible=deadline,
        is_defer=defer,
    )


def test_high_and_urgent_hard_bypass():
    reference = candidate("balanced+roi", net=1.0)
    better = candidate("balanced+roi_context", net=2.0)
    for level in ("high", "urgent"):
        result = choose_ordinary_net_utility_action(
            value_level=level,
            reference_action=reference.action,
            candidates=[reference, better],
        )
        assert result.final_action == reference.action
        assert result.reason == "high_value_bypass"


def test_ordinary_selects_higher_net_resource_reducing_granularity():
    reference = candidate(
        "balanced+roi_context", net=0.8, payload=3.0, energy=20.0, latency=4.0
    )
    roi = candidate("balanced+roi", net=1.1, payload=1.0, energy=12.0, latency=2.0)
    result = choose_ordinary_net_utility_action(
        value_level="medium",
        reference_action=reference.action,
        candidates=[reference, roi],
    )
    assert result.final_action == roi.action
    assert result.reason == "ordinary_net_projected"


def test_resource_increasing_winner_is_blocked():
    reference = candidate("balanced+roi", net=1.0, payload=1.0, energy=10.0)
    larger = candidate("balanced+roi_context", net=2.0, payload=3.0, energy=20.0)
    result = choose_ordinary_net_utility_action(
        value_level="low",
        reference_action=reference.action,
        candidates=[reference, larger],
    )
    assert result.final_action == reference.action
    assert result.blocked_resource_increasing_winner
    assert result.reason == "resource_increasing_winner_blocked"


def test_unsafe_and_late_candidates_are_not_selected():
    reference = candidate("balanced+roi", net=0.5)
    unsafe = candidate("balanced+metadata", net=3.0, safe=False)
    late = candidate("balanced+late", net=4.0, deadline=False)
    result = choose_ordinary_net_utility_action(
        value_level="medium",
        reference_action=reference.action,
        candidates=[reference, unsafe, late],
    )
    assert result.final_action == reference.action


def test_defer_wins_over_negative_net_action():
    reference = candidate("balanced+roi", net=-0.1)
    defer = candidate(
        "delay/store", model="delay_store", net=0.0, payload=0.0,
        energy=0.0, latency=0.0, defer=True,
    )
    result = choose_ordinary_net_utility_action(
        value_level="low",
        reference_action=reference.action,
        candidates=[reference, defer],
    )
    assert result.final_action == defer.action


def test_reference_defer_cannot_be_replaced_by_resource_increasing_payload():
    defer = candidate(
        "delay/store", model="delay_store", net=0.0, payload=0.0,
        energy=0.0, latency=0.0, defer=True,
    )
    roi = candidate("balanced+roi", net=2.0, payload=1.0, energy=10.0)
    result = choose_ordinary_net_utility_action(
        value_level="medium",
        reference_action=defer.action,
        candidates=[defer, roi],
    )
    assert result.final_action == defer.action
    assert result.blocked_resource_increasing_winner


def surplus_facts(**overrides):
    values = {
        "value_level": "medium",
        "reference_is_defer": True,
        "current_in_contact": True,
        "current_queue_mb": 0.0,
        "pending_high_value_payload": False,
        "candidate_hard_safe": True,
        "candidate_deadline_feasible": True,
        "candidate_eligible_by_existing_filters": True,
        "candidate_is_staged_followup": False,
        "candidate_predicted_net_utility": 0.1,
        "predicted_ground_arrival_s": 119.0,
        "slot_end_s": 120.0,
        "absolute_deadline_s": 300.0,
        "conservative_payload_upper_bound_mb": 3.5,
        "remaining_same_slot_capacity_mb": 20.0,
        "candidate_model_matches_original_selector": True,
    }
    values.update(overrides)
    return SameSlotSurplusFacts(**values)


def test_same_slot_surplus_certificate_accepts_idle_positive_net_capacity():
    decision = same_slot_surplus_certificate(surplus_facts())
    assert decision.eligible
    assert decision.reason == "same_slot_surplus_certified"


def test_same_slot_surplus_rejects_queue_or_pending_hv():
    assert not same_slot_surplus_certificate(
        surplus_facts(current_queue_mb=0.1)
    ).eligible
    assert not same_slot_surplus_certificate(
        surplus_facts(pending_high_value_payload=True)
    ).eligible


def test_same_slot_surplus_rejects_future_spill_and_staged_followup():
    assert not same_slot_surplus_certificate(
        surplus_facts(predicted_ground_arrival_s=121.0)
    ).eligible
    assert not same_slot_surplus_certificate(
        surplus_facts(conservative_payload_upper_bound_mb=21.0)
    ).eligible
    assert not same_slot_surplus_certificate(
        surplus_facts(candidate_is_staged_followup=True)
    ).eligible


def test_same_slot_surplus_rejects_nonpositive_net_and_model_change():
    assert not same_slot_surplus_certificate(
        surplus_facts(candidate_predicted_net_utility=0.0)
    ).eligible
    assert not same_slot_surplus_certificate(
        surplus_facts(candidate_model_matches_original_selector=False)
    ).eligible


def test_same_slot_wrapper_suppresses_all_future_ordinary_followups():
    from scripts.run_ordinary_net_granularity_cap050_v1 import (
        OrdinaryNetGranularityWrapper,
    )

    task_id = "ordinary_task"
    state = SimpleNamespace(
        pending_products={"roi_context"},
        virtual_continue_request_flag=True,
        pending_residual_registered_slot=10,
        residual_parent_payload_plan="metadata_then_roi_context",
        residual_request_time_s=200.0,
        residual_enqueue_time_s=201.0,
    )

    class Ledger:
        def __init__(self):
            self.records = {
                "pl_ordinary": SimpleNamespace(
                    lifecycle_id="pl_ordinary",
                    task_id=task_id,
                    product_type="roi_context",
                    current_status="pending",
                )
            }

        def transition(self, lifecycle_id, status, **kwargs):
            self.records[lifecycle_id].current_status = status

    ledger = Ledger()
    runner = SimpleNamespace(
        product_lifecycle={task_id: state},
        staged_delivery_requests={
            "ordinary": SimpleNamespace(task_id=task_id),
            "hv": SimpleNamespace(task_id="hv_task"),
        },
        residual_candidate_pool={
            "ordinary": SimpleNamespace(task_id=task_id)
        },
        residual_candidate_history={},
        lifecycle_ledger=ledger,
        downlink_queue=SimpleNamespace(
            items=[
                SimpleNamespace(task_id=task_id),
                SimpleNamespace(task_id="hv_task"),
            ]
        ),
    )
    wrapper = object.__new__(OrdinaryNetGranularityWrapper)
    wrapper.active_same_slot_override_task_id = task_id
    wrapper.active_same_slot_override_product = "metadata"
    wrapper.step_runner = runner
    wrapper.selection_rows = [
        {
            "task_id": task_id,
            "decision_reason": "same_slot_surplus_projected",
        }
    ]
    wrapper.finish_step(
        SimpleNamespace(slot_index=10, task=SimpleNamespace(task_id=task_id))
    )

    assert not state.pending_products
    assert not state.virtual_continue_request_flag
    assert set(runner.staged_delivery_requests) == {"hv"}
    assert not runner.residual_candidate_pool
    assert [item.task_id for item in runner.downlink_queue.items] == ["hv_task"]
    assert ledger.records["pl_ordinary"].current_status == "superseded"
    assert wrapper.selection_rows[0]["superseded_lifecycle_record_count"] == 1
    assert wrapper.selection_rows[0]["ordinary_followup_suppression_applied"] == 1
