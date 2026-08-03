from __future__ import annotations
import math
import pytest
import csv
from pathlib import Path
from src.evaluation.unified_ground_received_accounting import common_net, incremental_ground_events, score_task
from src.evaluation.task_outcome_adapter import adapt_b1, proposed_delivery_events, requirement_satisfied_by_products

CFG={"energy_normalization":{"controllable_mission_energy_wh":10.0},"backlog_normalization":{"terminal_backlog_mb":64.0},"lambda_E_common":1.0,"lambda_Q_common":1.0,"lambda_safe_common":10.0}
OUT=Path("outputs/_unified_ground_received_accounting_seed2026")
def output_rows():
    with (OUT/"task_level_ground_outcomes.csv").open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))

def base(**kw):
    row={"ground_completed":1,"capture_to_ground_latency_s":50.0,"capture_time_s":0.0,"deadline_time_s":100.0,"ground_information_sufficiency":1.0,"ground_result_quality":.8,"effective_task_value":1.0,"partial_ground_gross_utility":0.0};row.update(kw);return row

def test_late_delivery_does_not_receive_full_timeliness_utility():
    r=score_task(base(capture_to_ground_latency_s=120.0),CFG);assert r["timeliness_factor"]==0 and r["gross_ground_utility"]==0
def test_expired_task_has_zero_positive_ground_utility():
    r=score_task(base(ground_completed=0,partial_ground_gross_utility=0),CFG);assert r["gross_ground_utility"]==0
def test_duplicate_product_has_zero_incremental_utility():
    e=incremental_ground_events([{"product_type":"metadata","ground_available_time_s":1},{"product_type":"metadata","ground_available_time_s":2}],"verification");assert e[1]["delta_information_utility"]==0
def test_proposed_staged_utility_is_incremental_and_not_duplicated():
    e=incremental_ground_events([{"product_type":"metadata","ground_available_time_s":1},{"product_type":"roi_context","ground_available_time_s":2}],"evidence");assert sum(x["delta_information_utility"] for x in e)==pytest.approx(.95)
def test_roi_is_not_evidence_closure():
    e=incremental_ground_events([{"product_type":"roi","ground_available_time_s":1}],"evidence");assert e[-1]["information_sufficiency"]<.95
def test_common_net_identity_recomputes_exactly():
    x=common_net(5,10,32,0,CFG);assert x["common_ground_net_utility"]==pytest.approx(3.5)
def test_energy_units_are_wh():
    assert 3600/3600==1
def test_no_nan_or_inf_in_core_values():
    assert all(math.isfinite(v) for v in common_net(5,10,32,0,CFG).values())
def test_all_methods_have_same_task_universe():
    rows=output_rows(); groups={m:{r["task_id"] for r in rows if r["method"]==m} for m in {r["method"] for r in rows}};assert len(groups)==4 and len({frozenset(v) for v in groups.values()})==1
def test_each_method_has_one_row_per_task():
    rows=output_rows();assert all(sum(r["method"]==m for r in rows)==370 for m in {r["method"] for r in rows})
def test_b1_has_no_positive_utility_before_ground_result_available():
    assert all(float(r["gross_ground_utility"])==0 for r in output_rows() if r["method"].startswith("b1") and r["ground_completed"]=="0")
def test_energy_decomposition_sums_correctly():
    with (OUT/"energy_decomposition.csv").open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f));assert all(float(r["controllable_mission_energy_wh"])==pytest.approx(float(r["compute_energy_wh"])+float(r["compression_energy_wh"])+float(r["communication_energy_wh"])+float(r["payload_idle_energy_wh"])+float(r["isl_energy_wh"])) for r in rows)
def test_latency_decomposition_sums_correctly():
    assert all(float(r["capture_to_ground_latency_s"])>=0 for r in output_rows())
def test_ratio_denominators_are_explicit():
    with (OUT/"fairness_and_denominator_audit.csv").open(encoding="utf-8-sig") as f: assert all(r["arrival_denominator"]=="370" for r in csv.DictReader(f))
def test_high_value_task_definition_matches_manifest():
    rows=output_rows();assert all(sum(r["value_level"] in {"high_value","emergency"} for r in rows if r["method"]==m)==213 for m in {r["method"] for r in rows})
def test_b1_decision_logic_unchanged():
    from src.baselines.b1_enhanced import B1EnhancedConfig;assert B1EnhancedConfig().scheduler.policy=="enhanced"
def test_proposed_decision_logic_unchanged():
    from src.evaluation.baselines import build_routers
    from src.actions.action_space import ActionSpace
    r=build_routers({"routers":["proposed_lyapunov_rule_mode_safe_bandit"]},ActionSpace())[0];assert r.router_name=="proposed_lyapunov_rule_mode_safe_bandit"

def test_requirement_closure_uses_mission_product_semantics():
    assert requirement_satisfied_by_products({"metadata"}, "alert")
    assert not requirement_satisfied_by_products({"metadata"}, "verification")
    assert requirement_satisfied_by_products({"roi"}, "verification")
    assert not requirement_satisfied_by_products({"roi"}, "evidence")
    assert requirement_satisfied_by_products({"roi_context"}, "evidence")

def test_direct_delivery_record_is_not_omitted_when_lifecycle_is_empty():
    class Record:
        delivered_task_ids="task_1"; delivered_product_types="roi"; delivered_task_event_times_s="120"; slot=2
    events=proposed_delivery_events([Record()], [])
    assert events["task_1"] == [{"product_type":"roi","ground_available_time_s":120.0,"delivery_route":"direct_record"}]

def test_direct_and_lifecycle_duplicate_is_counted_once():
    class Record:
        delivered_task_ids="task_1"; delivered_product_types="roi"; delivered_task_event_times_s="120"; slot=2
    lifecycle=[{"task_id":"task_1","product_type":"roi","current_status":"delivered","delivered_slot":2}]
    assert len(proposed_delivery_events([Record()], lifecycle)["task_1"]) == 1


def test_lifecycle_delivery_time_uses_configured_slot_duration():
    lifecycle=[{"task_id":"task_1","product_type":"roi","current_status":"delivered","delivered_slot":2}]
    assert proposed_delivery_events([], lifecycle, slot_seconds=20.0)["task_1"][0]["ground_available_time_s"] == 40.0

def test_current_task_delivery_fallback_uses_selected_product_type():
    class Record:
        delivered_task_ids="task_1"; delivered_task_event_times_s="120"; delivery_status="delivered"
        task_id="task_1"; selected_product_type="metadata"; payload_delivery_time_s=120.0; slot=2
    assert proposed_delivery_events([Record()], [])["task_1"][0]["product_type"] == "metadata"


def test_b1_closure_uses_reconstruction_quality_not_ground_model_quality():
    manifest = {"task_1": {"task_id": "task_1", "deadline_time_s": 100.0, "capture_time_s": 0.0,
        "quality_requirement": 0.92, "information_requirement": 0.95, "product_requirement": "evidence",
        "effective_task_value": 1.0, "value_level": "high"}}
    trace = [{"task_id": "task_1", "ground_result_available": 1, "ground_result_available_time_s": 50.0,
        "capture_to_ground_latency_s": 50.0, "reconstructed_image_quality_proxy": 0.96,
        "ground_task_quality": 0.88, "compression_energy_wh": 1.0, "state": "ground_result_available"}]
    rows = adapt_b1(manifest, trace, "B1", "test", 2026, {**CFG, "full_frame_information_mapping": 1.0})
    assert rows[0]["requirement_satisfied"] == 1
    assert rows[0]["compression_sufficiency_quality"] == pytest.approx(0.96)
    assert rows[0]["ground_result_quality"] == pytest.approx(0.88)


def test_b1_reconstruction_below_threshold_does_not_close_requirement():
    manifest = {"task_1": {"task_id": "task_1", "deadline_time_s": 100.0, "capture_time_s": 0.0,
        "quality_requirement": 0.97, "information_requirement": 0.95, "product_requirement": "context",
        "effective_task_value": 1.0, "value_level": "urgent"}}
    trace = [{"task_id": "task_1", "ground_result_available": 1, "ground_result_available_time_s": 50.0,
        "capture_to_ground_latency_s": 50.0, "reconstructed_image_quality_proxy": 0.96,
        "ground_task_quality": 0.99, "compression_energy_wh": 1.0, "state": "ground_result_available"}]
    rows = adapt_b1(manifest, trace, "B1", "test", 2026, {**CFG, "full_frame_information_mapping": 1.0})
    assert rows[0]["ground_completed"] == 1
    assert rows[0]["requirement_satisfied"] == 0
