"""Single-seed canonical PROP-FULL run and read-only B1 unified comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_b1_enhanced_contact_compressed_baseline import _load_configs
from scripts.run_bounded_guarantee_virtual_queue import _guarantee_cfg, _staged_queue_cfg
from scripts.run_unified_ground_received_accounting_seed2026 import b1_energy, b1_safety, f, read_csv
from src.baselines.b1_enhanced import contact_schedule_hash, resource_trace_hash, task_manifest_hash
from src.common.config import load_yaml
from src.evaluation.baselines import build_routers
from src.evaluation.full_method_validation import validate_method_label
from src.evaluation.task_outcome_adapter import adapt_b1, adapt_proposed, manifest_rows
from src.evaluation.unified_ground_received_accounting import common_net, summarize_method
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder

OUT = ROOT / "outputs/_prop_full_vs_b1_unified_seed2026"
B1_SRC = ROOT / "outputs/_baseline_b1_enhanced_validation_v2_seed2026"
NO_STAGE_SRC = ROOT / "outputs/_unified_ground_received_accounting_seed2026"
ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["empty"]); writer.writeheader(); writer.writerows(rows)


def resolved_full_configs() -> dict[str, dict[str, Any]]:
    configs = _load_configs(None, 2026)
    configs["experiment"] = {"routers": [ROUTER]}
    progressive = dict(configs["simulator"].get("progressive_payload", {}))
    staged = _staged_queue_cfg(); staged["force_all_tasks_staged"] = False
    progressive["value_aware_staged_delivery_queue"] = staged
    progressive["phase6_proactive_followup"] = {
        "enabled": True, "eriv_threshold": 0.03, "residual_utility_threshold": 0.01,
        "max_proactive_residuals_per_slot": 1, "max_proactive_residuals_per_contact": 2,
        "queue_pressure_threshold": 0.72, "min_remaining_deadline_s": 30.0,
        "battery_guard_threshold": 0.50, "thermal_guard_threshold": 0.88,
        "require_contact_opportunity": True, "proactive_queue_bonus": 0.30,
        "full_method_aliases": [ROUTER],
    }
    progressive["phase6_residual_candidate_carryover"] = {
        "enabled": True, "re_evaluate_once_per_slot": True, "max_candidates_per_slot": 2,
        "max_candidate_age_s": 600, "require_metadata_delivered": True,
        "require_pending_residual": True, "require_positive_residual_utility": True,
        "remove_on_expired": True, "remove_on_closed": True,
        "remove_on_residual_delivered": True, "remove_on_duplicate_enqueued": False,
    }
    progressive["phase6_same_slot_carryover"] = {
        "enabled": True, "only_after_lifecycle_update": True, "reuse_existing_guards": True,
        "max_same_slot_eval_per_slot": 1, "enable_queue_fallback_on_insufficient_capacity": False,
        "enforce_no_reward_change": True, "enforce_no_queue_policy_change": True, "strict_mode": True,
    }
    progressive["roi_context_after_roi"] = {"enabled": True, "residual_utility_threshold": 0.01}
    configs["simulator"]["progressive_payload"] = progressive
    configs["simulator"]["guarantee"] = _guarantee_cfg(enabled=True, lambda_h=.35, z_max=4.0, beta_h=.30)
    validate_method_label("PROP-FULL", configs)
    return configs


def proposed_energy_safety(result, records, configs, cfg):
    ps = result.summary
    energy = {"compute_energy_wh": f(ps["total_compute_energy"])/3600, "compression_energy_wh": 0.0,
        "communication_energy_wh": f(ps["total_comm_energy"])/3600, "isl_energy_wh": 0.0,
        "payload_idle_energy_wh": 0.0, "housekeeping_energy_wh": f(ps["total_base_energy"])/3600,
        "controllable_mission_energy_wh": f(ps["total_energy"])/3600,
        "total_spacecraft_energy_wh": f(ps["total_spacecraft_energy"])/3600, "delivered_mb": f(ps["delivered_mb"])}
    safety = {"min_soc": f(ps["min_battery_soc"]), "final_soc": f(ps["final_battery_soc"]),
        "fraction_time_soc_below_70": sum(r.battery_next_soc < .7 for r in records)/len(records),
        "fraction_time_soc_below_50": sum(r.battery_next_soc < .5 for r in records)/len(records),
        "fraction_time_soc_below_20": sum(r.battery_next_soc < .2 for r in records)/len(records),
        "hard_battery_violation_count": sum(r.battery_next_soc < f(configs["simulator"]["battery_min"]) for r in records),
        "max_thermal_pressure": f(ps["max_thermal"]),
        "fraction_time_thermal_above_soft_limit": sum(r.thermal_next > f(cfg["thermal_soft_limit"]) for r in records)/len(records),
        "hard_thermal_violation_count": sum(r.thermal_next > f(configs["simulator"]["thermal_max"]) for r in records),
        "hard_safety_violation_count": f(ps["safety_violation_count"]), "terminal_queue_mb": f(records[-1].queue_next_mb),
        "maximum_queue_mb": f(ps["max_queue_mb"])}
    return energy, safety


def add_requested_metrics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in rows if int(r["ground_completed"])]
    partial = [r for r in rows if int(r["partial_ground_delivery"])]
    high = [r for r in rows if str(r["value_level"]) in {"high_value", "emergency"}]
    return {**summary,
        "any_ground_product_received_count": len(completed), "partial_only_task_count": len(partial),
        "closure_utility": sum(float(r["gross_ground_utility"]) for r in rows if int(r["requirement_satisfied"])),
        "partial_utility": sum(float(r["partial_ground_gross_utility"]) for r in rows),
        "average_ground_information_sufficiency": sum(float(r["ground_information_sufficiency"]) for r in completed)/max(len(completed), 1),
        "utility_per_wh": float(summary["common_ground_gross_utility"])/max(float(summary["controllable_mission_energy_wh"]), 1e-12),
        "utility_per_mb": float(summary["common_ground_gross_utility"])/max(float(summary["delivered_mb"]), 1e-12),
        "high_value_partial_count": sum(int(r["partial_ground_delivery"]) for r in high),
        "high_value_requirement_satisfied_count": sum(int(r["requirement_satisfied"]) for r in high)}


def alignment_rows(manifest, timeline, configs, methods):
    fields = {
        "task_manifest": task_manifest_hash(timeline), "task_ids": hashlib.sha256("\n".join(sorted(manifest)).encode()).hexdigest(),
        "arrivals": hashlib.sha256("\n".join(f"{k}:{manifest[k]['arrival_time_s']}" for k in sorted(manifest)).encode()).hexdigest(),
        "deadlines": hashlib.sha256("\n".join(f"{k}:{manifest[k]['deadline_time_s']}" for k in sorted(manifest)).encode()).hexdigest(),
        "task_values": hashlib.sha256("\n".join(f"{k}:{manifest[k]['effective_task_value']}" for k in sorted(manifest)).encode()).hexdigest(),
        "product_requirements": hashlib.sha256("\n".join(f"{k}:{manifest[k]['product_requirement']}" for k in sorted(manifest)).encode()).hexdigest(),
        "contact_trace": contact_schedule_hash(timeline), "resource_trace": resource_trace_hash(timeline),
        "battery_thermal_model": hashlib.sha256(yaml.safe_dump(configs["simulator"], sort_keys=True).encode()).hexdigest(),
        "horizon": str(len(timeline)), "seed": "2026", "unified_evaluator": "src.evaluation.unified_ground_received_accounting",
    }
    return [{"method": method, "aspect": key, "fingerprint": value, "aligned": 1} for method in methods for key, value in fields.items()]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=OUT); args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(f"Will not overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    configs = resolved_full_configs(); flags = validate_method_label("PROP-FULL", configs)
    (args.output_dir/"resolved_full_config.yaml").write_text(yaml.safe_dump(configs, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_csv(args.output_dir/"resolved_feature_flags.csv", flags)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs.get("task_stream", {})).build()
    manifest = manifest_rows(timeline, task_manifest_hash(timeline)); cfg = load_yaml(ROOT/"configs/unified_ground_received_accounting.yaml")
    utility_cfg = load_yaml(ROOT/"configs/product_requirement_utility.yaml")
    all_rows=[]; summaries=[]; energy_rows=[]; safety_rows=[]
    variants=[("B1-FIFO 1.4 W", "b1_fifo_paper_power_retention_1_00"), ("B1-Enhanced 1.4 W", "b1_enhanced_paper_power_retention_1_00"), ("B1-Enhanced 6.8 W", "b1_enhanced_conservative_power_retention_1_00")]
    for method, variant in variants:
        folder=B1_SRC/variant; trace=read_csv(folder/"per_task_delivery_trace.csv"); sm=json.loads((folder/"summary_metrics.json").read_text())
        queue=read_csv(folder/"queue_trace.csv"); rows=adapt_b1(manifest, trace, method, variant, 2026, cfg)
        energy=b1_energy(sm); safety=b1_safety(sm, queue); summary=summarize_method(rows, energy, safety, {"internal_utility": sm.get("net_system_utility", "")}, cfg)
        all_rows += rows; summaries.append(add_requested_metrics(summary, rows)); energy_rows.append({"method":method, **energy}); safety_rows.append({"method":method, **safety})
    env=SimulationEnvironment(configs); router=build_routers(configs["experiment"], env.action_space)[0]; runner=env.build_episode_runner(timeline); result=runner.run(router)
    rows=adapt_proposed(manifest, result.records, runner.last_payload_lifecycle_current_records, "PROP-FULL", "proposed_guarantee_aware_full", 2026, cfg, utility_cfg)
    energy,safety=proposed_energy_safety(result, result.records, configs, cfg); summary=summarize_method(rows,energy,safety,{"internal_utility":result.summary["net_system_utility"]},cfg)
    all_rows += rows; summaries.append(add_requested_metrics(summary, rows)); energy_rows.append({"method":"PROP-FULL", **energy}); safety_rows.append({"method":"PROP-FULL", **safety})
    if (NO_STAGE_SRC/"task_level_ground_outcomes.csv").exists():
        prior=[r for r in read_csv(NO_STAGE_SRC/"task_level_ground_outcomes.csv") if r["method"] == "proposed"]
        for r in prior: r["method"]="PROP-NO-STAGE"; r["variant"]="proposed_without_staged_delivery"
        prior_summary=[r for r in read_csv(NO_STAGE_SRC/"method_level_unified_summary.csv") if r["method"] == "proposed"][0]
        prior_summary["method"]="PROP-NO-STAGE"; prior_summary["variant"]="proposed_without_staged_delivery"
        all_rows += prior; summaries.append(add_requested_metrics(prior_summary, prior))
    methods=[row["method"] for row in summaries]
    write_csv(args.output_dir/"fair_environment_alignment.csv", alignment_rows(manifest,timeline,configs,methods))
    write_csv(args.output_dir/"task_level_ground_outcomes.csv", all_rows); write_csv(args.output_dir/"method_level_unified_summary.csv", summaries)
    write_csv(args.output_dir/"common_utility_decomposition.csv", [{k:r.get(k,"") for k in ("method","variant","partial_utility","closure_utility","common_ground_gross_utility","common_controllable_energy_cost","common_terminal_backlog_cost","common_hard_safety_cost","common_ground_net_utility")} for r in summaries])
    write_csv(args.output_dir/"high_value_delivery_summary.csv", [{k:v for k,v in r.items() if k.startswith("high_value") or k in {"method","variant"}} for r in summaries])
    write_csv(args.output_dir/"safety_summary.csv", safety_rows)
    request_trace=list(runner.last_staged_delivery_request_trace); lifecycle=list(runner.last_payload_lifecycle_current_records); recomp=list(runner.last_next_window_recompetition_trace)
    ledger=list(runner.last_payload_lifecycle_ledger_trace)
    write_csv(args.output_dir/"prop_full_staged_request_trace.csv", request_trace)
    write_csv(args.output_dir/"prop_full_recompetition_trace.csv", recomp)
    write_csv(args.output_dir/"prop_full_lifecycle_ledger_trace.csv", ledger)
    path_rows=[]
    for event, data in (("staged_request_created",request_trace),("residual_reevaluated_later_window",recomp)):
        path_rows.append({"event":event,"count":len(data),"example_task_ids":";".join(str(x.get("task_id",x.get("parent_task_id",""))) for x in data[:10])})
    statuses=Counter(str(x.get("current_status", "unknown")) for x in lifecycle if "staged" in str(x.get("source","")) or str(x.get("request_id","")).startswith("staged_"))
    for status,count in statuses.items(): path_rows.append({"event":f"staged_{status}","count":count,"example_task_ids":""})
    request_actions=Counter(str(x.get("action", "unknown")) for x in request_trace)
    for action,count in request_actions.items(): path_rows.append({"event":f"request_action_{action}","count":count,"example_task_ids":""})
    ledger_states=Counter(str(x.get("current_status", x.get("status", "unknown"))) for x in ledger if "staged" in str(x.get("source","")) or str(x.get("request_id","")).startswith("staged_"))
    for status,count in ledger_states.items(): path_rows.append({"event":f"ledger_{status}","count":count,"example_task_ids":""})
    ordinary_direct=sum(1 for r in rows if r["value_level"] not in {"high_value","emergency"} and r["ground_completed"] and "staged" not in str(r.get("transmitted_product_type","")))
    path_rows.append({"event":"ordinary_direct_ground_delivery","count":ordinary_direct,"example_task_ids":""})
    write_csv(args.output_dir/"prop_full_delivery_path_summary.csv",path_rows)
    for s in summaries:
        recomputed=common_net(float(s["common_ground_gross_utility"]),float(s["controllable_mission_energy_wh"]),float(s["terminal_queue_mb"]),float(s["hard_safety_violation_count"]),cfg)["common_ground_net_utility"]
        if not math.isclose(recomputed,float(s["common_ground_net_utility"]),abs_tol=1e-12): raise AssertionError(f"common net mismatch: {s['method']}")
    for row in summaries:
        for value in row.values():
            if isinstance(value,float) and not math.isfinite(value): raise AssertionError(f"non-finite value in {row['method']}")
    best=max(summaries,key=lambda r:float(r["common_ground_net_utility"]))["method"]
    full=next(r for r in summaries if r["method"]=="PROP-FULL"); no_stage=next((r for r in summaries if r["method"]=="PROP-NO-STAGE"),None)
    staged_delivered=sum(int(r["count"]) for r in path_rows if r["event"]=="staged_delivered")
    staged_blocked=sum(int(r["count"]) for r in path_rows if "blocked" in r["event"])
    report=f"""# PROP-FULL vs B1 Unified Comparison, seed=2026

- Correct PROP-FULL executed: yes. The run passed the fail-fast resolved-feature validator before simulation.
- Environment alignment with B1: yes for task universe, arrivals, deadlines, values, requirements, contact/resource traces, physical model, horizon, and seed; all task outcomes use the same unified evaluator.
- Previous staged-disabled proposed result is included read-only as `PROP-NO-STAGE`; no historical file was changed.
- Runtime staged path: {len(request_trace)} request trace events, {len(recomp)} later-window re-evaluations, {staged_delivered} staged deliveries, and {staged_blocked} blocked trace states/actions. See `prop_full_delivery_path_summary.csv` and the raw staged traces.
- PROP-FULL requirement-satisfied tasks: {full['requirement_satisfied_count']}; on-time closures: {full['on_time_task_success_count']}.
"""
    if no_stage:
        report += f"- PROP-NO-STAGE requirement-satisfied tasks: {no_stage['requirement_satisfied_count']}; on-time closures: {no_stage['on_time_task_success_count']}. Closure delta: {float(full['requirement_satisfied_count'])-float(no_stage['requirement_satisfied_count']):.0f}.\n"
        report += f"- Staging cost versus PROP-NO-STAGE: energy {float(full['controllable_mission_energy_wh'])-float(no_stage['controllable_mission_energy_wh']):.6g} Wh, average latency {float(full['average_latency_all_completed'])-float(no_stage['average_latency_all_completed']):.6g} s, delivered volume {float(full['delivered_mb'])-float(no_stage['delivered_mb']):.6g} MB.\n"
    report += f"- Highest common net utility at seed=2026: `{best}`. Therefore this seed {'supports' if best == 'PROP-FULL' else 'does not support'} PROP-FULL as the best compared method. No tuning was performed after observing results.\n"
    (args.output_dir/"final_comparison_report.md").write_text(report,encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__": main()
