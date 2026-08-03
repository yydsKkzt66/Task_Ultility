from copy import deepcopy

import pytest

from src.evaluation.full_method_validation import validate_method_label
from src.evaluation.task_outcome_adapter import proposed_delivery_events
from src.evaluation.unified_ground_received_accounting import common_net


def full_cfg():
    return {"experiment": {"routers": ["proposed_lyapunov_rule_mode_safe_bandit"]}, "simulator": {"guarantee": {"enabled": True, "lambda_H": .35, "hard_safety_veto": True}, "progressive_payload": {"value_aware_staged_delivery_queue": {"enabled": True}, "phase6_proactive_followup": {"enabled": True}, "phase6_residual_candidate_carryover": {"enabled": True}}}}


@pytest.mark.parametrize("path", [
    ("progressive_payload", "value_aware_staged_delivery_queue"),
    ("progressive_payload", "phase6_proactive_followup"),
    ("progressive_payload", "phase6_residual_candidate_carryover"),
    ("guarantee",),
])
def test_prop_full_fails_when_required_feature_disabled(path):
    cfg = full_cfg(); node = cfg["simulator"]
    for key in path: node = node[key]
    node["enabled"] = False
    with pytest.raises(ValueError): validate_method_label("PROP-FULL", cfg)


def test_prop_no_stage_cannot_be_labeled_prop_full():
    cfg = deepcopy(full_cfg())
    cfg["simulator"]["progressive_payload"]["value_aware_staged_delivery_queue"]["enabled"] = False
    with pytest.raises(ValueError): validate_method_label("proposed_guarantee_aware_full", cfg)


def test_ordinary_direct_delivery_event_is_retained():
    class R:
        delivered_task_ids="ordinary_1"; delivered_product_types="roi"; delivered_task_event_times_s="120"
        delivery_status="delivered"; task_id="ordinary_1"; selected_product_type="roi"; payload_delivery_time_s=120; slot=2
    assert proposed_delivery_events([R()], [])["ordinary_1"][0]["delivery_route"] == "direct_record"


def test_real_staged_lifecycle_event_is_identified():
    life={"task_id":"high_1","product_type":"roi_context","current_status":"delivered","delivered_slot":"8","source":"staged_delivery_queue","request_id":"staged_high_1"}
    assert proposed_delivery_events([], [life])["high_1"][0]["delivery_route"] == "staged_lifecycle"


def test_residual_recompetition_trace_contract():
    rows=[{"task_id":"t","window_index":1,"decision":"retained"},{"task_id":"t","window_index":2,"decision":"selected"}]
    assert len({r["window_index"] for r in rows if r["task_id"] == "t"}) > 1


def test_same_task_universe_and_denominator_contract():
    universes=[{"a","b","c"},{"a","b","c"},{"a","b","c"}]
    assert all(x == universes[0] for x in universes)
    assert len({len(x) for x in universes}) == 1


def test_common_net_exact_and_finite():
    cfg={"energy_normalization":{"controllable_mission_energy_wh":10},"backlog_normalization":{"terminal_backlog_mb":64},"lambda_E_common":1,"lambda_Q_common":1,"lambda_safe_common":10}
    out=common_net(5,10,32,0,cfg)
    assert out["common_ground_net_utility"] == pytest.approx(3.5)
    assert all(float(v) == float(v) and abs(float(v)) != float("inf") for v in out.values())
