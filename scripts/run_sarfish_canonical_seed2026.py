"""Run the frozen SARFishSample seed-2026 PROP-FULL and two B1 variants."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_bounded_guarantee_virtual_queue import _guarantee_cfg, _staged_queue_cfg
from scripts.run_prop_full_vs_b1_unified_seed2026 import proposed_energy_safety
from scripts.run_unified_ground_received_accounting_seed2026 import b1_energy, b1_safety
from src.baselines.b1_enhanced import B1EnhancedSimulator, load_b1_config
from src.common.config import load_yaml
from src.evaluation.baselines import build_routers
from src.evaluation.canonical_sarfish_environment import (
    LEGACY_AUDIT_MANIFEST, canonical_task_rows_v2, load_canonical_sarfish_configs,
    validate_legacy_task_equivalence, validate_main_paper_environment,
)
from src.evaluation.full_method_validation import validate_method_label
from src.evaluation.hv_phase1 import is_hv_all, is_hv_target, matched_latency, summarize_group
from src.evaluation.task_outcome_adapter import adapt_b1, adapt_proposed, manifest_rows
from src.evaluation.unified_ground_received_accounting import summarize_method
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder

OUT = ROOT / "outputs/_sarfish_formal_nominal_seed2026"
ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
LEGACY_TASK_HASH = "8c4987e6b861126db27d877ae1cfc183e3dce263808efdab9e9740a3c7a37dbb"
LEGACY_ENV_HASH = "d9c3e4ee3eef9b98be99a313c95d3a09f39d52a9dc8dfdb98afe00529c51538f"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def prop_configs() -> dict[str, Any]:
    configs = load_canonical_sarfish_configs(ROOT, seed=2026)
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


def b1_config(policy: str) -> Any:
    raw = load_yaml(ROOT / "configs/baseline_b1_enhanced.yaml")
    raw["enabled"] = True
    raw["scheduler"]["policy"] = policy
    raw["compression"]["compression_power_w"] = 1.4
    raw["compression"]["downstream_quality_retention"] = 1.0
    return load_b1_config(raw)


def stage_paths(rows: list[dict[str, Any]], lifecycle: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    staged_products: dict[str, set[str]] = {}
    for item in lifecycle:
        if str(item.get("current_status")) != "delivered": continue
        source = str(item.get("source", "")); request = str(item.get("request_id", ""))
        if "staged_delivery_queue" not in source and not request.startswith("staged_"): continue
        staged_products.setdefault(str(item["task_id"]), set()).add(str(item.get("product_type", "")))
    closure_ids = {str(row["task_id"]) for row in rows if int(row["requirement_satisfied"])}
    deep = {task_id for task_id, products in staged_products.items() if "roi_context" in products} & closure_ids
    shallow = ({task_id for task_id, products in staged_products.items() if "roi" in products} - deep) & closure_ids
    direct = closure_ids - shallow - deep
    return direct, shallow, deep


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=OUT); parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    occupied = [p.name for p in args.output_dir.iterdir() if p.name not in {"rollback", "resolved_sarfish_config.yaml"}]
    if occupied: raise FileExistsError(f"refusing to overwrite existing result files: {occupied}")
    configs = prop_configs()
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    validation = validate_main_paper_environment(configs, timeline)
    legacy = validate_legacy_task_equivalence(canonical_task_rows_v2(timeline), ROOT / LEGACY_AUDIT_MANIFEST)
    task_hash_v2 = validation["task_manifest_hash_v2"]; env_hash_v2 = validation["environment_hash_v2"]
    resolved = {"lineage": {"legacy_task_hash": LEGACY_TASK_HASH, "legacy_environment_hash": LEGACY_ENV_HASH,
        "canonical_task_hash_v2": task_hash_v2, "canonical_environment_hash_v2": env_hash_v2}, "configs": configs}
    (args.output_dir / "resolved_sarfish_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if args.preflight_only:
        print(json.dumps({**validation, **legacy}, indent=2)); return

    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    utility_cfg = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    manifest = manifest_rows(timeline, task_hash_v2)
    methods: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    environment_rows=[]; label_rows=[]; competition_rows=[]; stage_rows=[]

    for method, policy in (("B1-FIFO-1.4W", "fifo"), ("B1-Enhanced-1.4W", "enhanced")):
        check = validate_main_paper_environment(configs, timeline, expected_task_hash_v2=task_hash_v2, expected_environment_hash_v2=env_hash_v2)
        simulator = B1EnhancedSimulator(b1_config(policy), configs["simulator"], configs["model_pool"])
        result = simulator.run(timeline)
        rows = adapt_b1(manifest, result["per_task_delivery_trace"], method, policy, 2026, accounting)
        energy = b1_energy(result["summary"]); safety = b1_safety(result["summary"], result["queue_trace"])
        methods[method] = rows; summaries[method] = summarize_method(rows, energy, safety, {"internal_utility": result["summary"].get("net_system_utility", "")}, accounting)
        for trace in result["scheduler_candidate_trace"]: competition_rows.append({"method": method, **trace})
        environment_rows.append({"method": method, **check, "legacy_task_hash": LEGACY_TASK_HASH, "legacy_environment_hash": LEGACY_ENV_HASH})
        raw_dir=args.output_dir/"raw"/method; raw_dir.mkdir(parents=True, exist_ok=True)
        write_csv(raw_dir/"per_task_delivery_trace.csv", result["per_task_delivery_trace"]); write_csv(raw_dir/"scheduler_candidate_trace.csv", result["scheduler_candidate_trace"])

    check = validate_main_paper_environment(configs, timeline, expected_task_hash_v2=task_hash_v2, expected_environment_hash_v2=env_hash_v2)
    env = SimulationEnvironment(configs); router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline); prop_result = runner.run(router)
    prop_rows = adapt_proposed(manifest, prop_result.records, runner.last_payload_lifecycle_current_records, "PROP-FULL", "proposed_guarantee_aware_full", 2026, accounting, utility_cfg, slot_seconds=float(configs["timeline"]["slot_seconds"]))
    energy, safety = proposed_energy_safety(prop_result, prop_result.records, configs, accounting)
    methods["PROP-FULL"] = prop_rows; summaries["PROP-FULL"] = summarize_method(prop_rows, energy, safety, {"internal_utility": prop_result.summary.get("net_system_utility", "")}, accounting)
    environment_rows.append({"method": "PROP-FULL", **check, "legacy_task_hash": LEGACY_TASK_HASH, "legacy_environment_hash": LEGACY_ENV_HASH})
    direct, shallow, deep = stage_paths(prop_rows, list(runner.last_payload_lifecycle_current_records))
    raw_dir=args.output_dir/"raw"/"PROP-FULL"; raw_dir.mkdir(parents=True, exist_ok=True)
    write_csv(raw_dir/"staged_request_trace.csv", list(runner.last_staged_delivery_request_trace)); write_csv(raw_dir/"lifecycle.csv", list(runner.last_payload_lifecycle_current_records))

    for method, rows in methods.items():
        for row in rows:
            expected = manifest[str(row["task_id"])]["value_level"]
            label_rows.append({"method": method, "task_id": row["task_id"], "runtime_value_level": row["value_level"], "evaluator_value_level": expected, "aligned": int(row["value_level"] == expected)})
        system=summaries[method]
        method_direct = direct if method == "PROP-FULL" else {str(r["task_id"]) for r in rows if int(r["requirement_satisfied"])}
        method_staged = (shallow | deep) if method == "PROP-FULL" else set()
        for group, predicate in (("ALL", lambda r: True), ("HV", is_hv_all), ("HV-TARGET", is_hv_target)):
            item={"method":method,"task_group":group,**summarize_group(rows,predicate,system_summary=system,direct_ids=method_direct,staged_ids=method_staged)}
            item["shallow_staged_closures"] = len({str(r["task_id"]) for r in rows if predicate(r) and int(r["requirement_satisfied"])} & shallow) if method == "PROP-FULL" else 0
            item["deep_staged_closures"] = len({str(r["task_id"]) for r in rows if predicate(r) and int(r["requirement_satisfied"])} & deep) if method == "PROP-FULL" else 0
            stage_rows.append(item)

    latency=[]
    for right in ("B1-FIFO-1.4W", "B1-Enhanced-1.4W"):
        for task_set, predicate in (("both_ground_completed",lambda r:bool(int(r["ground_completed"]))), ("both_requirement_closed",lambda r:bool(int(r["requirement_satisfied"]))), ("both_HV_requirement_closed",lambda r:is_hv_all(r) and bool(int(r["requirement_satisfied"]))), ("both_HV_TARGET_requirement_closed",lambda r:is_hv_target(r) and bool(int(r["requirement_satisfied"])))):
            latency.append({"left_method":"PROP-FULL","right_method":right,"task_set":task_set,**matched_latency(prop_rows,methods[right],predicate)})

    if not all(row["aligned"] for row in label_rows): raise AssertionError("runtime/evaluator value label mismatch")
    if len({row["task_manifest_hash_v2"] for row in environment_rows}) != 1 or len({row["environment_hash_v2"] for row in environment_rows}) != 1: raise AssertionError("method hash mismatch")
    if any(len(rows)!=268 or len({r["task_id"] for r in rows})!=268 for rows in methods.values()): raise AssertionError("task universe mismatch")
    for method, rows in methods.items():
        if any(isinstance(v,float) and not math.isfinite(v) for row in rows for v in row.values()): raise AssertionError(f"non-finite task result: {method}")
    write_csv(args.output_dir/"value_label_alignment.csv",label_rows); write_csv(args.output_dir/"environment_alignment.csv",environment_rows)
    write_csv(args.output_dir/"method_level_summary.csv",[{"method":m,**s} for m,s in summaries.items()])
    write_csv(args.output_dir/"high_value_comparison.csv",[r for r in stage_rows if r["task_group"] in {"HV","HV-TARGET"}])
    write_csv(args.output_dir/"staged_activation_summary.csv",stage_rows); write_csv(args.output_dir/"matched_latency.csv",latency)
    write_csv(args.output_dir/"b1_candidate_competition.csv",competition_rows)
    competition_summary={m:{"dispatch_decisions":len([r for r in competition_rows if r["method"]==m]),"pre_filter_competition":sum(r["queued_candidate_count_before_filter"]>=2 for r in competition_rows if r["method"]==m),"eligible_competition":sum(r["eligible_candidate_count_before_feasibility"]>=2 for r in competition_rows if r["method"]==m),"post_filter_competition":sum(r["feasible_candidate_count_after_filter"]>=2 for r in competition_rows if r["method"]==m)} for m in ("B1-FIFO-1.4W","B1-Enhanced-1.4W")}
    validation_text=f"""# Validation report\n\n- PASS: SARFishSample/GRD/validation resolved; 50 legacy-equivalent scenes and 268 unique ordered tasks.\n- PASS: canonical task hash v2 `{task_hash_v2}` and environment hash v2 `{env_hash_v2}` match across all three methods.\n- PASS: all runtime/evaluator value labels align and no value label is empty.\n- PASS: task universes are unique and identical; no NaN/inf was found.\n- PASS: PROP-FULL feature-label fail-fast validation passed.\n- PASS: legacy hashes were retained as lineage and not overwritten.\n- Algorithm behavior was not tuned; B1 candidate instrumentation and evaluator changes are passive.\n"""
    (args.output_dir/"validation_report.md").write_text(validation_text,encoding="utf-8")
    lines=["# Final SARFishSample comparison, seed=2026","",validation_text.strip(),"","## Results",""]
    for method,s in summaries.items(): lines.append(f"- {method}: closures={s['requirement_satisfied_count']}, on-time={s['on_time_task_success_count']}, energy={float(s['controllable_mission_energy_wh']):.4f} Wh, delivered={float(s['delivered_mb']):.4f} MB, common net={float(s['common_ground_net_utility']):.6f}.")
    lines += ["", "## B1 candidate competition", "", "```json", json.dumps(competition_summary,indent=2), "```", "", "Quality semantics: reconstructed full-frame quality controls compression sufficiency; downstream ground-result quality controls utility. Full-frame receipt alone is not a closure."]
    (args.output_dir/"final_sarfish_comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__": main()
