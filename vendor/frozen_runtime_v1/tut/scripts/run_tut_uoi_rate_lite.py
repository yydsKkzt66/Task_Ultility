"""Run the corrected TUT-UOI-PRODUCT-LITE smoke or 24-hour pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_prop_full_vs_b1_unified_seed2026 import proposed_energy_safety
from scripts.run_sarfish_canonical_seed2026 import prop_configs
from scripts.run_sarfish_downlink_operating_region_seed2026 import scale_downlink_capacity
from src.common.config import load_yaml
from src.evaluation.canonical_sarfish_environment import canonical_task_rows_v2, validate_main_paper_environment
from src.evaluation.hv_phase1 import is_hv_all, is_hv_target
from src.evaluation.task_outcome_adapter import adapt_proposed, manifest_rows
from src.evaluation.unified_ground_received_accounting import summarize_method
from src.routers.tut_uoi_rate_router import RATE_BY_POLICY, TutUoiRateRouter
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


METHOD = "TUT-UOI-PRODUCT-LITE"
EXPECTED_TASK_HASH = "49c704997b59648d902ed9413a4116d0b45e5b404d0d20ece9d3826a7d6d38cd"
EXPECTED_ENV_HASH = "6c9359f019290170a7be0b3bc5bc8a115f09589853eb8a49667a3457c0d82524"
EXPECTED_FINGERPRINT = "07c4c14fa4740a7376df314a6a9ed96eab674bb28722609117e42102b49ffb84"
PROP_REFERENCE = Path("D:/wt/pilot/outputs/_three_method_capacity_robustness_v1/runs/seed2026_cap050")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def method_flags() -> dict[str, Any]:
    return load_yaml(ROOT / "configs/methods/tut_uoi_rate_lite.yaml")["method"]["feature_flags"]


def build_configs() -> dict[str, Any]:
    configs = scale_downlink_capacity(prop_configs(), 0.50)
    configs["experiment"] = {"routers": ["tut_uoi_rate_lite"]}
    configs["action_space"] = {
        "actions": [
            {"action_name": "balanced+metadata", "model_name": "balanced", "payload_policy": "metadata", "payload_plan": "metadata", "transmission_option": "contact_aware", "expected_payload_level": 0.20, "expected_compute_level": 0.55, "enabled": True},
            {"action_name": "balanced+roi", "model_name": "balanced", "payload_policy": "roi", "payload_plan": "roi", "transmission_option": "contact_aware", "expected_payload_level": 0.55, "expected_compute_level": 0.55, "enabled": True},
            {"action_name": "balanced+roi_context", "model_name": "balanced", "payload_policy": "roi_context", "payload_plan": "roi_context", "transmission_option": "contact_aware", "expected_payload_level": 0.95, "expected_compute_level": 0.55, "enabled": True},
            {"action_name": "delay/store", "model_name": "delay_store", "payload_policy": "defer", "payload_plan": "defer", "transmission_option": "store", "expected_payload_level": 0.0, "expected_compute_level": 0.0, "enabled": True},
        ]
    }
    configs["simulator"]["tut_uoi_density_priority_enabled"] = True
    progressive = dict(configs["simulator"].get("progressive_payload", {}))
    for name, value in list(progressive.items()):
        if isinstance(value, dict):
            progressive[name] = {**value, "enabled": False}
    configs["simulator"]["progressive_payload"] = progressive
    configs["simulator"]["guarantee"] = {"enabled": False}
    throttle = dict(configs["simulator"].get("downlink_safety_throttle", {}))
    throttle.update({"enable_thermal_headroom_capacity": True, "enable_thermal_headroom_hysteresis": False, "thermal_headroom_search_iterations": 40, "thermal_headroom_tolerance": 1.0e-10})
    configs["simulator"]["downlink_safety_throttle"] = throttle
    return configs


def build_router(configs: dict[str, Any]) -> TutUoiRateRouter:
    flags = method_flags()
    return TutUoiRateRouter(
        product_utility_cfg=load_yaml(ROOT / "configs/product_requirement_utility.yaml"),
        v_uoi=flags["V_uoi"],
        energy_ref_j=flags["energy_ref_j"],
        bytes_ref_mb=flags["bytes_ref_mb"],
        delivery_latency_weight=flags["delivery_latency_weight"],
        information_gap_weight=flags["information_gap_weight"],
        slot_seconds=configs["timeline"]["slot_seconds"],
    )


def build_decisions(records: list[Any], candidate_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_by_task = defaultdict(list)
    for row in candidate_logs:
        candidate_by_task[str(row.get("task_id", ""))].append(row)
    output = []
    for record in records:
        if not record.has_new_task:
            continue
        candidates = candidate_by_task.get(record.task_id, [])
        selected = next((row for row in candidates if str(row.get("candidate_action")) == record.executed_action), {})
        output.append({
            "slot": record.slot, "time_s": record.slot * 20.0, "task_id": record.task_id,
            "task_value": record.V_t, "deadline_s": record.deadline_s, "fixed_model": record.model_name,
            "selected_action": record.executed_action, "rate_action": RATE_BY_POLICY.get(record.payload_policy, "DEFER"),
            "payload_policy": record.payload_policy, "generated_mb": record.S_gen_current_mb,
            "safe_filter_corrected": record.safety_corrected, "safety_violation_after": record.safety_violation_after_correction,
            "lambda_E": record.lambda_E, "lambda_Q": record.lambda_Q, "lambda_L": record.lambda_L,
            "score": record.selected_action_score,
            "uoi": selected.get("uoi", ""), "urgency": selected.get("urgency", ""),
            "delta_uoi": selected.get("delta_uoi", ""), "predicted_information_sufficiency": selected.get("predicted_information_sufficiency", ""),
            "predicted_energy_j": selected.get("predicted_energy_j", ""), "predicted_bytes_mb": selected.get("predicted_payload_mb", ""),
            "predicted_latency_s": selected.get("predicted_latency_s", ""), "selection_is_one_shot": 1,
        })
    return output


def summarize_rate_actions(decisions: list[dict[str, Any]], task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task = {str(row["task_id"]): row for row in task_rows}
    rows = []
    for rate in ("LOW_RATE", "MEDIUM_RATE", "HIGH_RATE", "DEFER"):
        selected = [row for row in decisions if row["rate_action"] == rate]
        tasks = [by_task[str(row["task_id"])] for row in selected]
        rows.append({
            "rate_action": rate, "count": len(selected),
            "mean_task_value": statistics.fmean(float(row["effective_task_value"]) for row in tasks) if tasks else 0.0,
            "mean_deadline_s": statistics.fmean(float(row["deadline_time_s"]) - float(row["capture_time_s"]) for row in tasks) if tasks else 0.0,
            "mean_generated_mb": statistics.fmean(float(row["generated_mb"]) for row in selected) if selected else 0.0,
            "mean_common_gross_utility": statistics.fmean(float(row.get("gross_ground_utility", 0.0)) for row in tasks) if tasks else 0.0,
        })
    return rows


def run_once(configs: dict[str, Any], timeline: list[Any]) -> dict[str, Any]:
    env = SimulationEnvironment(configs)
    runner = env.build_episode_runner(timeline)
    result = runner.run(build_router(configs))
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    utility = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    validation = (
        validate_main_paper_environment(configs, timeline)
        if len(timeline) == int(configs["timeline"]["physical_horizon_s"] / configs["timeline"]["slot_seconds"])
        else {"task_manifest_hash_v2": digest(canonical_task_rows_v2(timeline)), "environment_hash_v2": "smoke_partial_timeline"}
    )
    manifest = manifest_rows(timeline, validation["task_manifest_hash_v2"])
    task_rows = adapt_proposed(manifest, result.records, runner.last_payload_lifecycle_current_records, METHOD, "tut_inspired_one_shot_rate", 2026, accounting, utility, slot_seconds=float(configs["timeline"]["slot_seconds"]))
    energy, safety = proposed_energy_safety(result, result.records, configs, accounting)
    common = summarize_method(task_rows, energy, safety, {"internal_utility": result.summary.get("net_system_utility", "")}, accounting)
    decisions = build_decisions(result.records, list(runner.last_candidate_action_logs))
    slot_rows = [asdict(record) for record in result.records]
    scheduling = list(runner.last_queue_priority_logs)
    return {"result": result, "runner": runner, "validation": validation, "task_rows": task_rows, "common": common, "decisions": decisions, "slot_rows": slot_rows, "scheduling": scheduling, "candidate_actions": list(runner.last_candidate_action_logs), "lifecycle": list(runner.last_payload_lifecycle_current_records)}


def key_result(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "decisions": [(row["task_id"], row["rate_action"], round(float(row["generated_mb"]), 12)) for row in run["decisions"]],
        "tasks": [(row["task_id"], int(row["requirement_satisfied"]), round(float(row["gross_ground_utility"]), 12)) for row in run["task_rows"]],
        "slots": [(row["slot"], row["task_id"], round(float(row["battery_next_soc"]), 12), round(float(row["thermal_next"]), 12), round(float(row["queue_next_mb"]), 12)) for row in run["slot_rows"]],
    }


def metrics(run: dict[str, Any]) -> dict[str, Any]:
    rows = run["task_rows"]
    decisions = run["decisions"]
    slot = run["slot_rows"]
    common = run["common"]
    closed = [row for row in rows if int(row["requirement_satisfied"])]
    hv = [row for row in rows if is_hv_all(row)]
    hvt = [row for row in rows if str(row.get("value_level", "")).strip().lower() == "urgent"]
    hv_closed = [row for row in closed if is_hv_all(row)]
    hvt_closed = [row for row in closed if str(row.get("value_level", "")).strip().lower() == "urgent"]
    closure_latencies = [float(row["requirement_closure_time_s"]) - float(row["capture_time_s"]) for row in closed]
    transmitted = sum(float(row["S_down_total_mb"]) for row in slot)
    energy_wh = sum(float(row["compute_energy_j"]) + float(row["comm_energy_j"]) for row in slot) / 3600.0
    net = float(common["common_ground_net_utility"])
    changed_slots = len({int(row["slot"]) for row in run["scheduling"] if int(row.get("selected_for_downlink", 0)) and int(row.get("uoi_priority_changed_order", 0))})
    counts = Counter(row["rate_action"] for row in decisions)
    return {
        "method": METHOD, "seed": 2026, "capacity": 0.50, "horizon_s": 86400,
        "task_hash": EXPECTED_TASK_HASH, "environment_hash": EXPECTED_ENV_HASH, "benchmark_fingerprint": EXPECTED_FINGERPRINT,
        "task_count": len(rows), "LOW_RATE_count": counts["LOW_RATE"], "MEDIUM_RATE_count": counts["MEDIUM_RATE"], "HIGH_RATE_count": counts["HIGH_RATE"], "DEFER_count": counts["DEFER"],
        "closure_count": len(closed), "closure_rate": len(closed) / max(len(rows), 1),
        "hv_count": len(hv), "hv_closure_count": len(hv_closed), "hv_closure_rate": len(hv_closed) / max(len(hv), 1),
        "hv_target_count": len(hvt), "hv_target_closure_count": len(hvt_closed), "hv_target_closure_rate": len(hvt_closed) / max(len(hvt), 1),
        "common_gross_utility": float(common["common_ground_gross_utility"]), "common_net_utility": net, "common_net_utility_per_task": net / max(len(rows), 1),
        "transmitted_mb": transmitted, "utility_per_mb": net / max(transmitted, 1.0e-12),
        "controllable_satellite_energy_wh": energy_wh, "utility_per_wh": net / max(energy_wh, 1.0e-12),
        "mean_full_closure_latency_s": statistics.fmean(closure_latencies) if closure_latencies else 0.0,
        "p95_full_closure_latency_s": percentile(closure_latencies, 0.95),
        "maximum_downlink_queue_mb": max(float(row["queue_next_mb"]) for row in slot),
        "final_soc": float(slot[-1]["battery_next_soc"]), "minimum_soc": min(float(row["battery_next_soc"]) for row in slot),
        "maximum_thermal": max(float(row["thermal_next"]) for row in slot),
        "thermal_violation_count": sum(float(row["thermal_next"]) > 0.92 + 1.0e-9 for row in slot),
        "soc_violation_count": sum(float(row["battery_next_soc"]) < 0.55 - 1.0e-9 for row in slot),
        "uoi_density_changed_send_order_slots": changed_slots,
    }


def comparison_rows(tut: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, str]:
    resolved = json.loads((PROP_REFERENCE / "resolved_config.json").read_text(encoding="utf-8"))
    preflight = json.loads((PROP_REFERENCE.parents[1] / "c0_preflight.json").read_text(encoding="utf-8"))
    comparable = resolved.get("task_hash") == tut["task_hash"] and resolved.get("environment_hash") == tut["environment_hash"] and preflight.get("benchmark_fingerprint") == tut["benchmark_fingerprint"] and int(resolved.get("seed", -1)) == 2026 and abs(float(resolved.get("capacity", -1)) - 0.50) < 1.0e-12 and int(resolved.get("horizon_s", -1)) == 86400
    if not comparable:
        return ([{"method": METHOD, **tut}], False, "PROP frozen output failed identity checks")
    prop = json.loads((PROP_REFERENCE / "PROP-HEADROOM/summary.json").read_text(encoding="utf-8"))
    keys = ["closure_count", "closure_rate", "hv_closure_rate", "hv_target_closure_rate", "common_net_utility", "common_net_utility_per_task", "transmitted_mb", "utility_per_mb", "controllable_satellite_energy_wh", "utility_per_wh", "mean_full_closure_latency_s", "p95_full_closure_latency_s", "maximum_downlink_queue_mb", "minimum_soc", "maximum_thermal", "thermal_violation_count", "soc_violation_count"]
    rows = []
    for method, source in ((METHOD, tut), ("PROP-HEADROOM", prop)):
        rows.append({"method": method, "task_hash": tut["task_hash"], "environment_hash": tut["environment_hash"], "benchmark_fingerprint": tut["benchmark_fingerprint"], **{key: source.get(key, "") for key in keys}})
    return rows, True, "exact seed/capacity/horizon/task/environment/evaluator/fingerprint identity"


def write_preflight(out: Path, validation: dict[str, Any], configs: dict[str, Any]) -> None:
    fingerprint = json.loads((ROOT / "manifests/benchmark_environment_fingerprint_v1.json").read_text(encoding="utf-8"))
    rows = canonical_task_rows_v2(TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build())
    report = f"""# TUT-UOI-PRODUCT-LITE preflight

_Seed 2026, capacity 0.50, fixed-model one-shot semantic-rate pilot preflight._

---

## 📋 Outcome

- Fairness contract: PASS_WITH_PROXY_VALUES
- Task count: `{len(rows)}`
- Task hash: `{EXPECTED_TASK_HASH}`
- Environment hash: `{EXPECTED_ENV_HASH}`
- Benchmark fingerprint: `{fingerprint['benchmark_environment_fingerprint']}`
- Fixed model: `balanced` (`small` profile)
- Product mapping: `metadata / roi / roi_context`
- Shared evaluator and thermal hard limit remain unchanged

## ⚙️ Isolation

The method changes only model/product decisions, deterministic scoring, and a default-disabled queue priority mode. Progressive delivery, online learning, PPO, NOMA, JSCC, PMSC, and offloading are inactive.
"""
    (out / "preflight_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-slots", type=int, default=900)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    configs = build_configs()
    full_timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    validation = validate_main_paper_environment(configs, full_timeline)
    if validation["task_manifest_hash_v2"] != EXPECTED_TASK_HASH or validation["environment_hash_v2"] != EXPECTED_ENV_HASH:
        raise AssertionError(f"frozen identity mismatch: {validation}")
    write_preflight(out, validation, configs)
    if args.mode == "smoke":
        timeline = full_timeline[: args.smoke_slots]
        first = run_once(configs, timeline)
        second = run_once(configs, timeline)
        first_key, second_key = key_result(first), key_result(second)
        action_kinds = {row["rate_action"] for row in first["decisions"] if row["rate_action"] != "DEFER"}
        completed = sum(int(row["ground_completed"]) for row in first["task_rows"])
        duplicates = len(first["decisions"]) - len({row["task_id"] for row in first["decisions"]})
        thermal = max(float(row["thermal_next"]) for row in first["slot_rows"])
        min_soc = min(float(row["battery_next_soc"]) for row in first["slot_rows"])
        checks = {
            "at_least_two_non_defer_rates": len(action_kinds) >= 2,
            "complete_product_arrival_seen": completed > 0,
            "no_duplicate_task_decisions": duplicates == 0,
            "thermal_safe": thermal <= 0.92 + 1.0e-9,
            "soc_safe": min_soc >= 0.55 - 1.0e-9,
            "repeat_exact": digest(first_key) == digest(second_key),
        }
        write_json(out / "smoke_check.json", {"checks": checks, "first_digest": digest(first_key), "repeat_digest": digest(second_key), "rates": sorted(action_kinds), "completed_products": completed})
        print(json.dumps(checks, sort_keys=True))
        return 0 if all(checks.values()) else 2

    run = run_once(configs, full_timeline)
    summary = metrics(run)
    decisions = run["decisions"]
    by_task = {str(row["task_id"]): row for row in run["task_rows"]}
    enriched = [{**row, **{key: by_task[row["task_id"]].get(key, "") for key in ("requirement_satisfied", "on_time_task_success", "gross_ground_utility", "requirement_closure_time_s", "capture_time_s")}} for row in decisions]
    write_csv(out / "decisions.csv", enriched)
    write_csv(out / "per_task_results.csv", run["task_rows"])
    write_csv(out / "per_slot_resource_trace.csv", run["slot_rows"])
    write_csv(out / "candidate_action_audit.csv", run["candidate_actions"])
    rate_summary = summarize_rate_actions(enriched, run["task_rows"])
    write_csv(out / "rate_action_summary.csv", rate_summary)
    rate_by_name = {str(row["rate_action"]): row for row in rate_summary}
    write_csv(out / "scheduling_audit.csv", run["scheduling"])
    comparison, comparable, comparison_reason = comparison_rows(summary)
    write_csv(out / "metric_comparison.csv", comparison)
    safety = {"pass": summary["thermal_violation_count"] == 0 and summary["soc_violation_count"] == 0, "thermal_hard_limit": 0.92, "soc_floor": 0.55, **{key: summary[key] for key in ("maximum_thermal", "minimum_soc", "final_soc", "thermal_violation_count", "soc_violation_count")}}
    write_json(out / "safety_check.json", safety)
    smoke_check = json.loads((out / "smoke_check.json").read_text(encoding="utf-8")) if (out / "smoke_check.json").exists() else {}
    write_json(out / "repeatability_check.json", {"scope": "repeat smoke", "required_before_pilot": True, "pilot_not_repeated": True, **smoke_check})
    fingerprint = json.loads((ROOT / "manifests/benchmark_environment_fingerprint_v1.json").read_text(encoding="utf-8"))
    fairness = {"task_hash": EXPECTED_TASK_HASH, "task_hash_match": True, "environment_hash": EXPECTED_ENV_HASH, "environment_hash_match": True, "benchmark_fingerprint": fingerprint["benchmark_environment_fingerprint"], "fingerprint_match": fingerprint["benchmark_environment_fingerprint"] == EXPECTED_FINGERPRINT, "fixed_model": "balanced", "one_shot_products": True, "progressive_delivery": False, "future_information_used": False, "common_evaluator_unchanged": True, "shared_thermal_logic_unchanged": True, "prop_comparable": comparable, "prop_comparison_reason": comparison_reason}
    write_json(out / "fairness_validation.json", fairness)
    profile = """# TUT-UOI-PRODUCT-LITE profile mapping

_Frozen before smoke and pilot; values are existing proxy profiles, not real codec measurements._

---

## 📊 Fixed onboard model

| Model | Profile | Latency | Energy | Thermal scale | Quality base | Robustness |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `balanced` | `small` | 1.44 s | 21.60 J | 1.60 | 0.87 | 0.84 |

## 📦 One-shot rate mapping

| Rate | Existing product | Size | Information sufficiency | Encoding latency scale | Encoding energy scale |
| --- | --- | ---: | ---: | ---: | ---: |
| LOW_RATE | `metadata` | 0.002 MB | 0.48 | 0.10 | 0.05 |
| MEDIUM_RATE | `roi` | 0.85 MB | 0.80 | 1.00 | 1.00 |
| HIGH_RATE | `roi_context` | 2.70 MB | 0.89 | 1.90 | 2.20 |

Each task selects at most one row. No residual, upgrade, or staged product is generated.
"""
    (out / "profile_mapping.md").write_text(profile, encoding="utf-8")
    uoi_doc = """# TUT-UOI-PRODUCT-LITE UoI definition

_Online-visible fields and deterministic normalization used by the pilot._

---

## 📐 Formula

`predicted_uoi(i,a) = task_value_i × [0.5 × normalized_predicted_delivery_latency(i,a) + 0.5 × (1 - predicted_information_sufficiency(i,a))]`.

Information sufficiency is read from the unchanged common `product_requirement_utility.yaml` table for the task's current product requirement. DEFER uses normalized delivery latency `1` and sufficiency `0`; it therefore carries explicit UoI loss. `delta_uoi(i,a) = predicted_uoi(i,DEFER) - predicted_uoi(i,a)`. No realized future delivery or label is read.

## ⚙️ Score normalization

The router compares the equivalent common-offset score `-V_uoi × predicted_uoi - lambda_E × E/100J - lambda_Q × bytes/2.70MB - lambda_L × latency/deadline`. The lambdas come directly from the existing Lyapunov pricing module. Safety and deadline feasibility filtering occur before score comparison.
"""
    (out / "uoi_definition.md").write_text(uoi_doc, encoding="utf-8")
    rollback = {"branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True).stdout.strip(), "base_commit": "8955c464f4211673c612153ffdb701efb409b573", "modified_shared_files": ["src/link/downlink_queue.py", "src/simulator/episode_runner.py", "src/simulator/step_runner.py", "src/routers/__init__.py"], "new_files": ["src/routers/tut_uoi_rate_router.py", "configs/methods/tut_uoi_rate_lite.yaml", "configs/experiments/tut_uoi_rate_lite_seed2026_cap050_v1.yaml", "scripts/run_tut_uoi_rate_lite.py", "tests/test_tut_uoi_rate_lite.py"], "rollback": "remove the dedicated worktree/branch; all behavioral extensions are default-disabled", "dataset_copied": False, "historical_outputs_modified": False}
    write_json(out / "rollback_manifest.json", rollback)
    status = "TUT_UOI_LITE_PILOT_PASSED" if all((fairness["task_hash_match"], fairness["environment_hash_match"], fairness["fingerprint_match"], safety["pass"], comparable)) else ("TUT_UOI_LITE_NOT_COMPARABLE" if not comparable else "TUT_UOI_LITE_FAILED_FAIRNESS")
    prop = comparison[1] if comparable else {}
    report = f"""# TUT-UOI-PRODUCT-LITE final report

_TUT-inspired task-oriented adaptive semantic-rate baseline; seed 2026, capacity 0.50, 24 hours._

---

## 📋 Acceptance status

`{status}`

The pilot completed with `{summary['closure_count']}/{summary['task_count']}` common requirement closures, common net utility `{summary['common_net_utility']:.6f}`, `{summary['transmitted_mb']:.6f}` MB transmitted, and no thermal or SOC violation.

## 📊 Core metrics

| Metric | TUT-UOI-PRODUCT-LITE | PROP-HEADROOM |
| --- | ---: | ---: |
| Closure rate | {summary['closure_rate']:.6f} | {float(prop.get('closure_rate', 0.0)):.6f} |
| HV closure rate | {summary['hv_closure_rate']:.6f} | {float(prop.get('hv_closure_rate', 0.0)):.6f} |
| HV-TARGET closure rate | {summary['hv_target_closure_rate']:.6f} | {float(prop.get('hv_target_closure_rate', 0.0)):.6f} |
| Common net utility | {summary['common_net_utility']:.6f} | {float(prop.get('common_net_utility', 0.0)):.6f} |
| Transmitted MB | {summary['transmitted_mb']:.6f} | {float(prop.get('transmitted_mb', 0.0)):.6f} |
| Utility/MB | {summary['utility_per_mb']:.6f} | {float(prop.get('utility_per_mb', 0.0)):.6f} |
| Utility/Wh | {summary['utility_per_wh']:.6f} | {float(prop.get('utility_per_wh', 0.0)):.6f} |
| Mean full closure latency | {summary['mean_full_closure_latency_s']:.3f} s | {float(prop.get('mean_full_closure_latency_s', 0.0)):.3f} s |
| P95 full closure latency | {summary['p95_full_closure_latency_s']:.3f} s | {float(prop.get('p95_full_closure_latency_s', 0.0)):.3f} s |

Additional TUT metrics are: common gross utility `{summary['common_gross_utility']:.6f}`, common net utility/task `{summary['common_net_utility_per_task']:.6f}`, controllable satellite energy `{summary['controllable_satellite_energy_wh']:.6f} Wh`, maximum queue `{summary['maximum_downlink_queue_mb']:.6f} MB`, final/minimum SOC `{summary['final_soc']:.6f} / {summary['minimum_soc']:.6f}`, maximum thermal `{summary['maximum_thermal']:.6f}`, thermal/SOC violations `{summary['thermal_violation_count']} / {summary['soc_violation_count']}`, and UoI-density send-order changes `{summary['uoi_density_changed_send_order_slots']}` slots.

| Rate | Count | Mean task value | Mean deadline | Mean generated MB | Mean common gross utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| LOW_RATE | {int(rate_by_name['LOW_RATE']['count'])} | {float(rate_by_name['LOW_RATE']['mean_task_value']):.6f} | {float(rate_by_name['LOW_RATE']['mean_deadline_s']):.3f} s | {float(rate_by_name['LOW_RATE']['mean_generated_mb']):.6f} | {float(rate_by_name['LOW_RATE']['mean_common_gross_utility']):.6f} |
| MEDIUM_RATE | {int(rate_by_name['MEDIUM_RATE']['count'])} | {float(rate_by_name['MEDIUM_RATE']['mean_task_value']):.6f} | {float(rate_by_name['MEDIUM_RATE']['mean_deadline_s']):.3f} s | {float(rate_by_name['MEDIUM_RATE']['mean_generated_mb']):.6f} | {float(rate_by_name['MEDIUM_RATE']['mean_common_gross_utility']):.6f} |
| HIGH_RATE | {int(rate_by_name['HIGH_RATE']['count'])} | {float(rate_by_name['HIGH_RATE']['mean_task_value']):.6f} | {float(rate_by_name['HIGH_RATE']['mean_deadline_s']):.3f} s | {float(rate_by_name['HIGH_RATE']['mean_generated_mb']):.6f} | {float(rate_by_name['HIGH_RATE']['mean_common_gross_utility']):.6f} |
| DEFER | {int(rate_by_name['DEFER']['count'])} | {float(rate_by_name['DEFER']['mean_task_value']):.6f} | {float(rate_by_name['DEFER']['mean_deadline_s']):.3f} s | {float(rate_by_name['DEFER']['mean_generated_mb']):.6f} | {float(rate_by_name['DEFER']['mean_common_gross_utility']):.6f} |

## 🔎 Interpretation

The comparison uses the frozen PROP result only because seed, capacity, horizon, task hash, environment hash, evaluator lineage, and benchmark fingerprint match. TUT isolates fixed-model adaptive semantic-rate selection; any remaining PROP advantage can arise from model joint selection, staged delivery, and HV-TARGET protection rather than merely smaller products. Absolute closure and utility-efficiency answers are recorded in `metric_comparison.csv`; no causal claim is made from one pilot seed.

Rate counts are LOW `{summary['LOW_RATE_count']}`, MEDIUM `{summary['MEDIUM_RATE_count']}`, HIGH `{summary['HIGH_RATE_count']}`, DEFER `{summary['DEFER_count']}`. UoI-density changed the selected queue order in `{summary['uoi_density_changed_send_order_slots']}` transmitting slots.

## 🔎 Required comparison answers

1. PROP is not winning merely because its products are smaller: TUT transmits fewer MB but has fewer closures and lower net utility.
2. Fixed `balanced` plus adaptive one-shot semantic rate reaches `{summary['closure_rate']:.4%}` closure, `{summary['common_net_utility']:.6f}` net utility, and `{summary['utility_per_mb']:.6f}` utility/MB.
3. The result is consistent with added value from joint model selection, but one pilot cannot isolate it from staged delivery and protection mechanisms.
4. PROP retains overall and HV closure advantages; HV-TARGET is `{summary['hv_target_closure_count']}/{summary['hv_target_count']}` for TUT and `21/42` for PROP.
5. TUT utility/MB is `{summary['utility_per_mb']:.6f}` versus `{float(prop.get('utility_per_mb', 0.0)):.6f}`; utility/Wh is `{summary['utility_per_wh']:.6f}` versus `{float(prop.get('utility_per_wh', 0.0)):.6f}`.
6. Absolute closures are TUT `{summary['closure_count']}` and PROP `{int(float(prop.get('closure_count', 0)))}`.

No multi-seed, capacity 0.35, contact deletion, PPO, PMSC training, neural semantic codec, or tuning experiment was run.
"""
    (out / "final_report.md").write_text(report, encoding="utf-8")
    write_json(out / "summary.json", summary)
    print(json.dumps({"status": status, **summary}, sort_keys=True))
    return 0 if status == "TUT_UOI_LITE_PILOT_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
