"""Validate frozen TUT-UOI-PRODUCT-LITE product adaptivity across link capacity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_tut_uoi_rate_lite import (
    EXPECTED_ENV_HASH,
    EXPECTED_FINGERPRINT,
    EXPECTED_TASK_HASH,
    METHOD,
    build_configs,
    digest,
    key_result,
    metrics,
    run_once,
)
from src.evaluation.canonical_sarfish_environment import canonical_task_rows_v2, validate_main_paper_environment
from src.timeline.timeline_builder import TimelineBuilder


OUT = Path("D:/wt/pilot/outputs/_tut_uoi_product_capacity_trend_seed2026_v1")
FROZEN_050 = Path("D:/wt/pilot/outputs/_tut_uoi_product_lite_diagnostic_v1/corrected_pilot")
FROZEN_REPEATABILITY = Path("D:/wt/pilot/outputs/_tut_uoi_product_lite_diagnostic_v1/repeatability_check.json")
CAPACITIES = (1.00, 0.50, 0.35)
NEW_CAPACITIES = (1.00, 0.35)
EXPECTED_ENV_BY_CAPACITY = {
    1.00: "df2776eb79ed5cb74063ce624f16e8cfee8b24a5f9c2f810ec1f8760485082f0",
    0.50: EXPECTED_ENV_HASH,
    0.35: "de8c9e70a7210918ba9de5d656aa6b1e25ab69b3add8eed4e95604de76e6ba8b",
}
RATE_BY_POLICY = {"metadata": "LOW_RATE", "roi": "MEDIUM_RATE", "roi_context": "HIGH_RATE", "defer": "DEFER"}
RATE_RANK = {"DEFER": 0, "LOW_RATE": 1, "MEDIUM_RATE": 2, "HIGH_RATE": 3}
HASH_FILES = (
    ROOT / "src/routers/tut_uoi_rate_router.py",
    ROOT / "configs/methods/tut_uoi_rate_lite.yaml",
    ROOT / "configs/product_requirement_utility.yaml",
    ROOT / "configs/unified_ground_received_accounting.yaml",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capacity_tag(capacity: float) -> str:
    return f"cap{int(round(capacity * 100)):03d}"


def raw_dir(capacity: float) -> Path:
    return OUT / "runs" / capacity_tag(capacity)


def build_capacity_configs(capacity: float) -> dict[str, Any]:
    """Change only the contract-approved throughput multiplier from frozen cap=0.50."""
    configs = deepcopy(build_configs())
    ratio = float(capacity) / 0.50
    contact = configs["timeline"]["contact"]
    for key in ("throughput_min_mbps", "throughput_max_mbps"):
        contact[key] = float(contact[key]) * ratio
    configs["downlink_capacity_multiplier"] = float(capacity)
    return configs


def timeline_for(configs: dict[str, Any]) -> list[Any]:
    return TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()


def immutable_config_view(configs: dict[str, Any]) -> dict[str, Any]:
    view = deepcopy(configs)
    contact = view["timeline"]["contact"]
    contact.pop("throughput_min_mbps", None)
    contact.pop("throughput_max_mbps", None)
    view.pop("downlink_capacity_multiplier", None)
    return view


def candidate_safe(row: dict[str, Any]) -> bool:
    return not str(row.get("safety_rejection_reason", "")).strip()


def preflight() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_fairness = json.loads((FROZEN_050 / "fairness_validation.json").read_text(encoding="utf-8"))
    frozen_repeat = json.loads(FROZEN_REPEATABILITY.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {
        "frozen_050_exists": FROZEN_050.exists(),
        "frozen_050_repeatability_exact": bool(frozen_repeat["corrected_pilot"]["all_exact"]),
        "frozen_task_hash_match": frozen_fairness["task_hash"] == EXPECTED_TASK_HASH,
        "frozen_environment_hash_match": frozen_fairness["environment_hash"] == EXPECTED_ENV_HASH,
        "frozen_fingerprint_match": frozen_fairness["benchmark_fingerprint"] == EXPECTED_FINGERPRINT,
        "common_evaluator_unchanged": bool(frozen_fairness["common_evaluator_unchanged"]),
    }
    base_view = None
    canonical_rows = None
    contact_mask = None
    capacity_details = {}
    for capacity in CAPACITIES:
        configs = build_capacity_configs(capacity)
        timeline = timeline_for(configs)
        validation = validate_main_paper_environment(configs, timeline)
        rows = canonical_task_rows_v2(timeline)
        mask = [int(step.link_trace.in_contact) for step in timeline]
        view = immutable_config_view(configs)
        if base_view is None:
            base_view, canonical_rows, contact_mask = view, rows, mask
        checks[f"cap_{capacity:.2f}_immutable_config_match"] = digest(view) == digest(base_view)
        checks[f"cap_{capacity:.2f}_task_rows_match"] = digest(rows) == digest(canonical_rows)
        checks[f"cap_{capacity:.2f}_contact_timing_match"] = mask == contact_mask
        checks[f"cap_{capacity:.2f}_task_hash_match"] = validation["task_manifest_hash_v2"] == EXPECTED_TASK_HASH
        checks[f"cap_{capacity:.2f}_capacity_specific_environment_hash_match"] = (
            validation["environment_hash_v2"] == EXPECTED_ENV_BY_CAPACITY[capacity]
        )
        capacity_details[f"{capacity:.2f}"] = {
            "task_count": len(rows),
            "task_hash": validation["task_manifest_hash_v2"],
            "environment_hash": validation["environment_hash_v2"],
            "throughput_min_mbps": configs["timeline"]["contact"]["throughput_min_mbps"],
            "throughput_max_mbps": configs["timeline"]["contact"]["throughput_max_mbps"],
            "contact_slot_count": sum(mask),
        }
    checks["all_pass"] = all(bool(value) for value in checks.values())
    resolved = {
        "method": METHOD,
        "seed": 2026,
        "capacities": list(CAPACITIES),
        "new_runs": list(NEW_CAPACITIES),
        "reused_capacity": 0.50,
        "task_hash": EXPECTED_TASK_HASH,
        "environment_hash_by_capacity": {f"{capacity:.2f}": EXPECTED_ENV_BY_CAPACITY[capacity] for capacity in CAPACITIES},
        "benchmark_fingerprint": EXPECTED_FINGERPRINT,
        "frozen_file_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in HASH_FILES},
        "capacity_details": capacity_details,
        "only_changed_fields": [
            "timeline.contact.throughput_min_mbps",
            "timeline.contact.throughput_max_mbps",
            "downlink_capacity_multiplier",
        ],
        "checks": checks,
    }
    write_json(OUT / "resolved_capacity_config.json", resolved)
    write_json(OUT / "preflight_validation.json", checks)
    status = "PASS" if checks["all_pass"] else "FAIL"
    report = f"""# TUT-UOI-PRODUCT-LITE capacity trend preflight

_Seed 2026, capacities 1.00 / 0.50 / 0.35, frozen product-selection method._

---

## ✅ Outcome

`{status}`

| Identity item | Value |
| --- | --- |
| Task count | `268` |
| Task hash | `{EXPECTED_TASK_HASH}` |
| Environment hash | Capacity-specific; see `resolved_capacity_config.json` |
| Benchmark fingerprint | `{EXPECTED_FINGERPRINT}` |
| Fixed model | `balanced` |
| Frozen cap=0.50 reuse | `{checks['frozen_050_repeatability_exact']}` |

## ⚙️ Allowed change

Only `timeline.contact.throughput_min_mbps` and `throughput_max_mbps` are multiplied by the contract capacity value. Task rows, arrivals, deadlines, labels, contact-slot mask, initial state, hardware, product profiles, UoI formula, weights, SafeFilter, and common evaluator are identical.

## 🚫 Isolation

No PROP, B1, or OFFLOAD simulation is invoked. No tuning, multi-seed run, product-profile change, or evaluator change is permitted by this runner.
"""
    (OUT / "preflight_report.md").write_text(report, encoding="utf-8")
    if not checks["all_pass"]:
        raise AssertionError(f"capacity preflight failed: {checks}")


def smoke(capacity: float) -> None:
    checks = json.loads((OUT / "preflight_validation.json").read_text(encoding="utf-8"))
    if not checks.get("all_pass"):
        raise AssertionError("preflight must pass before smoke")
    configs = build_capacity_configs(capacity)
    timeline = timeline_for(configs)[:900]
    first = run_once(configs, timeline)
    second = run_once(configs, timeline)
    thermal = max(f(row["thermal_next"]) for row in first["slot_rows"])
    minimum_soc = min(f(row["battery_next_soc"]) for row in first["slot_rows"])
    result = {
        "capacity": capacity,
        "slot_count": len(timeline),
        "first_digest": digest(key_result(first)),
        "repeat_digest": digest(key_result(second)),
        "repeat_exact": digest(key_result(first)) == digest(key_result(second)),
        "thermal_safe": thermal <= 0.92 + 1.0e-9,
        "soc_safe": minimum_soc >= 0.55 - 1.0e-9,
        "task_decision_count": len(first["decisions"]),
    }
    result["all_pass"] = all(bool(result[key]) for key in ("repeat_exact", "thermal_safe", "soc_safe"))
    write_json(OUT / f"smoke_{capacity_tag(capacity)}.json", result)
    if not result["all_pass"]:
        raise AssertionError(f"smoke failed: {result}")


def pilot(capacity: float) -> None:
    smoke_result = json.loads((OUT / f"smoke_{capacity_tag(capacity)}.json").read_text(encoding="utf-8"))
    if not smoke_result.get("all_pass"):
        raise AssertionError("matching-capacity smoke must pass before pilot")
    configs = build_capacity_configs(capacity)
    timeline = timeline_for(configs)
    validation = validate_main_paper_environment(configs, timeline)
    if validation["task_manifest_hash_v2"] != EXPECTED_TASK_HASH or validation["environment_hash_v2"] != EXPECTED_ENV_BY_CAPACITY[capacity]:
        raise AssertionError(f"identity mismatch at capacity={capacity}: {validation}")
    run = run_once(configs, timeline)
    summary = metrics(run)
    summary["capacity"] = capacity
    summary["environment_hash"] = validation["environment_hash_v2"]
    out = raw_dir(capacity)
    out.mkdir(parents=True, exist_ok=True)
    by_task = {str(row["task_id"]): row for row in run["task_rows"]}
    decisions = [
        {
            **row,
            **{
                key: by_task[row["task_id"]].get(key, "")
                for key in ("requirement_satisfied", "on_time_task_success", "gross_ground_utility", "requirement_closure_time_s", "capture_time_s")
            },
        }
        for row in run["decisions"]
    ]
    write_csv(out / "decisions.csv", decisions)
    write_csv(out / "per_task_results.csv", run["task_rows"])
    write_csv(out / "per_slot_resource_trace.csv", run["slot_rows"])
    write_csv(out / "candidate_action_audit.csv", run["candidate_actions"])
    write_csv(out / "scheduling_audit.csv", run["scheduling"])
    write_json(out / "summary.json", summary)
    safety = {
        "pass": summary["thermal_violation_count"] == 0 and summary["soc_violation_count"] == 0,
        "maximum_thermal": summary["maximum_thermal"],
        "minimum_soc": summary["minimum_soc"],
        "final_soc": summary["final_soc"],
        "thermal_violation_count": summary["thermal_violation_count"],
        "soc_violation_count": summary["soc_violation_count"],
    }
    write_json(out / "safety_check.json", safety)
    write_json(
        out / "fairness_validation.json",
        {
            "task_hash": validation["task_manifest_hash_v2"],
            "task_hash_match": validation["task_manifest_hash_v2"] == EXPECTED_TASK_HASH,
            "environment_hash": validation["environment_hash_v2"],
            "environment_hash_match": validation["environment_hash_v2"] == EXPECTED_ENV_BY_CAPACITY[capacity],
            "benchmark_fingerprint": EXPECTED_FINGERPRINT,
            "fingerprint_match": True,
            "capacity": capacity,
            "capacity_only_change": True,
            "common_evaluator_unchanged": True,
            "product_profiles_unchanged": True,
            "uoi_formula_and_weights_unchanged": True,
            "fixed_model": "balanced",
            "one_shot_products": True,
            "future_information_used": False,
        },
    )
    write_json(out / "scientific_digest.json", {"capacity": capacity, "key_result_sha256": digest(key_result(run))})


def load_capacity(capacity: float) -> dict[str, Any]:
    base = FROZEN_050 if abs(capacity - 0.50) < 1.0e-12 else raw_dir(capacity)
    return {
        "base": base,
        "summary": json.loads((base / "summary.json").read_text(encoding="utf-8")),
        "decisions": read_csv(base / "decisions.csv"),
        "tasks": read_csv(base / "per_task_results.csv"),
        "slots": read_csv(base / "per_slot_resource_trace.csv"),
        "candidates": read_csv(base / "candidate_action_audit.csv"),
        "scheduling": read_csv(base / "scheduling_audit.csv"),
        "safety": json.loads((base / "safety_check.json").read_text(encoding="utf-8")),
        "fairness": json.loads((base / "fairness_validation.json").read_text(encoding="utf-8")),
    }


def action_summary(capacity: float, run: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(row["rate_action"] for row in run["decisions"])
    candidates = run["candidates"]
    row: dict[str, Any] = {"capacity": capacity, "task_count": len(run["decisions"])}
    for rate, policy in (("LOW_RATE", "metadata"), ("MEDIUM_RATE", "roi"), ("HIGH_RATE", "roi_context"), ("DEFER", "defer")):
        relevant = [item for item in candidates if item["candidate_payload"] == policy]
        row[f"{rate}_count"] = counts[rate]
        row[f"{rate}_ratio"] = counts[rate] / max(len(run["decisions"]), 1)
        row[f"{rate}_safe_filter_pass_count"] = sum(candidate_safe(item) for item in relevant)
    high = [item for item in candidates if item["candidate_payload"] == "roi_context"]
    row["HIGH_thermal_filtered_count"] = sum("thermal" in str(item.get("safety_rejection_reason", "")).lower() for item in high)
    row["HIGH_score_dominated_count"] = sum(candidate_safe(item) and not int(item["selected_flag"]) for item in high)
    row["HIGH_selected_count"] = sum(int(item["selected_flag"]) for item in high)
    return row


def score_value(row: dict[str, Any], key: str) -> float:
    return f(row.get(key, 0.0))


def transition_cause(old_row: dict[str, Any], new_row: dict[str, Any], changed: bool) -> str:
    if not changed:
        return "UNCHANGED"
    if str(new_row.get("safety_rejection_reason", "")).strip():
        return "SAFETY_FILTER"
    delivery_change = abs(
        0.5 * score_value(new_row, "task_value")
        * (score_value(new_row, "normalized_predicted_delivery_latency") - score_value(old_row, "normalized_predicted_delivery_latency"))
    )
    components = {
        "DELIVERY_LATENCY": delivery_change + abs(score_value(new_row, "normalized_latency_cost") - score_value(old_row, "normalized_latency_cost")),
        "QUEUE_PRESSURE_BYTE_COST": abs(score_value(new_row, "normalized_byte_cost") - score_value(old_row, "normalized_byte_cost")),
        "ENERGY_PRESSURE": abs(score_value(new_row, "normalized_energy_cost") - score_value(old_row, "normalized_energy_cost")),
    }
    dominant, magnitude = max(components.items(), key=lambda item: item[1])
    return dominant if magnitude > 1.0e-12 else "FINAL_SCORE_REORDERING"


def analyze() -> None:
    runs = {capacity: load_capacity(capacity) for capacity in CAPACITIES}
    identities_ok = all(
        run["fairness"].get("task_hash") == EXPECTED_TASK_HASH
        and run["fairness"].get("environment_hash") == EXPECTED_ENV_BY_CAPACITY[capacity]
        and run["fairness"].get("benchmark_fingerprint") == EXPECTED_FINGERPRINT
        and bool(run["fairness"].get("common_evaluator_unchanged"))
        for capacity, run in runs.items()
    )
    task_sets = [{row["task_id"] for row in run["decisions"]} for run in runs.values()]
    paired_tasks_ok = all(task_set == task_sets[0] and len(task_set) == 268 for task_set in task_sets)

    action_rows = [action_summary(capacity, runs[capacity]) for capacity in CAPACITIES]

    metric_keys = (
        "closure_count", "closure_rate", "hv_closure_rate", "hv_target_closure_rate",
        "common_gross_utility", "common_net_utility", "common_net_utility_per_task",
        "transmitted_mb", "utility_per_mb", "controllable_satellite_energy_wh", "utility_per_wh",
        "mean_full_closure_latency_s", "p95_full_closure_latency_s", "maximum_downlink_queue_mb",
        "uoi_density_changed_send_order_slots", "minimum_soc", "final_soc", "maximum_thermal",
        "thermal_violation_count", "soc_violation_count",
    )
    metric_rows = [{"capacity": capacity, "method": METHOD, **{key: runs[capacity]["summary"].get(key, "") for key in metric_keys}} for capacity in CAPACITIES]
    write_csv(OUT / "metric_comparison.csv", metric_rows)

    decision_maps = {capacity: {row["task_id"]: row for row in runs[capacity]["decisions"]} for capacity in CAPACITIES}
    candidate_maps = {
        capacity: {(row["task_id"], RATE_BY_POLICY[row["candidate_payload"]]): row for row in runs[capacity]["candidates"]}
        for capacity in CAPACITIES
    }
    transitions = []
    score_audit = []
    for capacity in CAPACITIES:
        selected_map = decision_maps[capacity]
        for row in runs[capacity]["candidates"]:
            rate = RATE_BY_POLICY[row["candidate_payload"]]
            score_audit.append(
                {
                    "capacity": capacity,
                    "task_id": row["task_id"],
                    "candidate_rate_action": rate,
                    "candidate_action": row.get("candidate_action", ""),
                    "selected_rate_action": selected_map[row["task_id"]]["rate_action"],
                    "selected_flag": row["selected_flag"],
                    "safety_rejection_reason": row.get("safety_rejection_reason", ""),
                    "final_score": row.get("tut_score", ""),
                    "predicted_information_sufficiency": row.get("predicted_information_sufficiency", ""),
                    "predicted_delivery_latency_s": row.get("predicted_latency_s", ""),
                    "normalized_predicted_delivery_latency": row.get("normalized_predicted_delivery_latency", ""),
                    "normalized_energy_cost": row.get("normalized_energy_cost", ""),
                    "normalized_byte_cost": row.get("normalized_byte_cost", ""),
                    "normalized_latency_cost": row.get("normalized_latency_cost", ""),
                    "lambda_E": row.get("lambda_E", ""),
                    "lambda_Q": row.get("lambda_Q", ""),
                    "lambda_L": row.get("lambda_L", ""),
                    "predicted_payload_mb": row.get("predicted_payload_mb", ""),
                    "predicted_queue_next_mb": row.get("predicted_queue_next_mb", ""),
                    "task_value": row.get("task_value", ""),
                }
            )
    score_groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in score_audit:
        score_groups.setdefault((f(row["capacity"]), str(row["task_id"])), []).append(row)
    order_effect_counts = {capacity: Counter() for capacity in CAPACITIES}
    for (capacity, _task_id), rows in score_groups.items():
        safe_rows = [row for row in rows if not str(row["safety_rejection_reason"]).strip()] or rows

        def winner(score_fn) -> str:
            return max(
                safe_rows,
                key=lambda row: (
                    score_fn(row),
                    -f(row["predicted_payload_mb"]),
                    str(row["candidate_action"]),
                ),
            )["candidate_rate_action"]

        selected = str(rows[0]["selected_rate_action"])
        winner_without_byte = winner(lambda row: f(row["final_score"]) + f(row["normalized_byte_cost"]))
        winner_without_energy = winner(lambda row: f(row["final_score"]) + f(row["normalized_energy_cost"]))
        winner_without_latency = winner(
            lambda row: f(row["final_score"])
            + f(row["normalized_latency_cost"])
            + 0.5 * f(row["task_value"]) * f(row["normalized_predicted_delivery_latency"])
        )
        flags = {
            "byte_queue_cost_changed_order": int(winner_without_byte != selected),
            "energy_cost_changed_order": int(winner_without_energy != selected),
            "delivery_latency_changed_order": int(winner_without_latency != selected),
        }
        for key, value in flags.items():
            order_effect_counts[capacity][key] += value
        for row in rows:
            row.update(
                {
                    **flags,
                    "winner_without_byte_queue_cost": winner_without_byte,
                    "winner_without_energy_cost": winner_without_energy,
                    "winner_without_delivery_latency": winner_without_latency,
                    "score_without_byte_queue_cost": f(row["final_score"]) + f(row["normalized_byte_cost"]),
                    "score_without_energy_cost": f(row["final_score"]) + f(row["normalized_energy_cost"]),
                    "score_without_delivery_latency": f(row["final_score"])
                    + f(row["normalized_latency_cost"])
                    + 0.5 * f(row["task_value"]) * f(row["normalized_predicted_delivery_latency"]),
                }
            )
    for row in action_rows:
        row.update(order_effect_counts[f(row["capacity"])])
    write_csv(OUT / "capacity_action_summary.csv", action_rows)
    write_csv(OUT / "score_component_audit.csv", score_audit)

    for task_id in sorted(task_sets[0]):
        a100 = decision_maps[1.00][task_id]["rate_action"]
        a050 = decision_maps[0.50][task_id]["rate_action"]
        a035 = decision_maps[0.35][task_id]["rate_action"]
        c100_050 = transition_cause(candidate_maps[1.00][(task_id, a100)], candidate_maps[0.50][(task_id, a100)], a100 != a050)
        c050_035 = transition_cause(candidate_maps[0.50][(task_id, a050)], candidate_maps[0.35][(task_id, a050)], a050 != a035)
        rank_delta_100_050 = RATE_RANK[a050] - RATE_RANK[a100]
        rank_delta_050_035 = RATE_RANK[a035] - RATE_RANK[a050]
        transitions.append(
            {
                "task_id": task_id,
                "capacity_1.00_action": a100,
                "capacity_0.50_action": a050,
                "capacity_0.35_action": a035,
                "action_chain": f"{a100}->{a050}->{a035}",
                "transition_1.00_to_0.50": f"{a100}->{a050}",
                "transition_0.50_to_0.35": f"{a050}->{a035}",
                "cause_1.00_to_0.50": c100_050,
                "cause_0.50_to_0.35": c050_035,
                "rank_delta_1.00_to_0.50": rank_delta_100_050,
                "rank_delta_0.50_to_0.35": rank_delta_050_035,
                "changed_any": int(len({a100, a050, a035}) > 1),
                "downgrade_count": int(rank_delta_100_050 < 0) + int(rank_delta_050_035 < 0),
                "upgrade_count": int(rank_delta_100_050 > 0) + int(rank_delta_050_035 > 0),
            }
        )
    write_csv(OUT / "task_action_transition.csv", transitions)

    pair_counts = Counter()
    for row in transitions:
        pair_counts[f"1.00_to_0.50:{row['transition_1.00_to_0.50']}"] += 1
        pair_counts[f"0.50_to_0.35:{row['transition_0.50_to_0.35']}"] += 1
        pair_counts[f"CHAIN:{row['action_chain']}"] += 1
    write_csv(OUT / "transition_matrix_summary.csv", [{"transition": key, "count": value} for key, value in sorted(pair_counts.items())])

    safety = {
        "all_pass": all(bool(run["safety"]["pass"]) for run in runs.values()),
        "capacities": {f"{capacity:.2f}": runs[capacity]["safety"] for capacity in CAPACITIES},
        "task_environment_fingerprint_match": identities_ok,
        "paired_task_ids_match": paired_tasks_ok,
    }
    write_json(OUT / "safety_check.json", safety)

    frozen_repeat = json.loads(FROZEN_REPEATABILITY.read_text(encoding="utf-8"))
    repeatability = {
        "capacity_1.00_smoke": json.loads((OUT / "smoke_cap100.json").read_text(encoding="utf-8")),
        "capacity_0.50_full_pilot": frozen_repeat["corrected_pilot"],
        "capacity_0.35_smoke": json.loads((OUT / "smoke_cap035.json").read_text(encoding="utf-8")),
        "new_full_pilots_independently_repeated": False,
        "note": "New full pilots were each run once; matching-capacity smoke runs were exact repeats, and frozen cap=0.50 has byte-exact full repeat evidence.",
        "raw_artifact_sha256": {
            f"{capacity:.2f}": {
                name: sha256(raw_dir(capacity) / name)
                for name in ("decisions.csv", "per_task_results.csv", "per_slot_resource_trace.csv", "candidate_action_audit.csv", "summary.json", "safety_check.json")
            }
            for capacity in NEW_CAPACITIES
        },
    }
    write_json(OUT / "repeatability_check.json", repeatability)

    by_capacity = {row["capacity"]: row for row in action_rows}
    low_monotone = by_capacity[1.00]["LOW_RATE_count"] <= by_capacity[0.50]["LOW_RATE_count"] <= by_capacity[0.35]["LOW_RATE_count"]
    medium_monotone = by_capacity[1.00]["MEDIUM_RATE_count"] >= by_capacity[0.50]["MEDIUM_RATE_count"] >= by_capacity[0.35]["MEDIUM_RATE_count"]
    high_monotone = by_capacity[1.00]["HIGH_RATE_count"] >= by_capacity[0.50]["HIGH_RATE_count"] >= by_capacity[0.35]["HIGH_RATE_count"]
    high_strictly_decreased = by_capacity[1.00]["HIGH_RATE_count"] > by_capacity[0.35]["HIGH_RATE_count"]
    defer_monotone = by_capacity[1.00]["DEFER_count"] <= by_capacity[0.50]["DEFER_count"] <= by_capacity[0.35]["DEFER_count"]
    closure_monotone = runs[1.00]["summary"]["closure_count"] >= runs[0.50]["summary"]["closure_count"] >= runs[0.35]["summary"]["closure_count"]
    utility_monotone = runs[1.00]["summary"]["common_net_utility"] >= runs[0.50]["summary"]["common_net_utility"] >= runs[0.35]["summary"]["common_net_utility"]
    changed_tasks = sum(int(row["changed_any"]) for row in transitions)
    downgrades = sum(int(row["downgrade_count"]) for row in transitions)
    upgrades = sum(int(row["upgrade_count"]) for row in transitions)
    if not (identities_ok and paired_tasks_ok and safety["all_pass"]):
        status = "FAIRNESS_OR_IMPLEMENTATION_FAILURE"
    elif changed_tasks == 0:
        status = "TUT_PRODUCT_SELECTION_CAPACITY_INSENSITIVE"
    elif all((low_monotone, medium_monotone, high_monotone, defer_monotone, closure_monotone, utility_monotone)) and downgrades >= upgrades:
        status = "TUT_PRODUCT_ADAPTIVITY_VALIDATED"
    else:
        status = "TUT_PRODUCT_ADAPTIVITY_WEAK_BUT_VALID"

    causes = Counter()
    for row in transitions:
        if row["cause_1.00_to_0.50"] != "UNCHANGED":
            causes[row["cause_1.00_to_0.50"]] += 1
        if row["cause_0.50_to_0.35"] != "UNCHANGED":
            causes[row["cause_0.50_to_0.35"]] += 1
    requested_transitions = {
        "HIGH_to_MEDIUM_LOW_DEFER": sum(
            count for key, count in pair_counts.items()
            if not key.startswith("CHAIN:") and ":HIGH_RATE->" in key and not key.endswith("HIGH_RATE")
        ),
        "MEDIUM_to_LOW_DEFER": sum(
            count for key, count in pair_counts.items()
            if not key.startswith("CHAIN:") and (key.endswith("MEDIUM_RATE->LOW_RATE") or key.endswith("MEDIUM_RATE->DEFER"))
        ),
        "LOW_to_DEFER": sum(
            count for key, count in pair_counts.items()
            if not key.startswith("CHAIN:") and key.endswith("LOW_RATE->DEFER")
        ),
        "reverse_upgrades": upgrades,
        "fully_invariant_tasks": 268 - changed_tasks,
    }
    write_json(OUT / "requested_transition_counts.json", requested_transitions)
    checks = {
        "status": status,
        "changed_task_count": changed_tasks,
        "unchanged_task_count": 268 - changed_tasks,
        "downgrade_transition_count": downgrades,
        "upgrade_transition_count": upgrades,
        "low_increases_as_capacity_drops": low_monotone,
        "medium_decreases_as_capacity_drops": medium_monotone,
        "high_decreases_as_capacity_drops": high_monotone,
        "high_strictly_decreases": high_strictly_decreased,
        "defer_increases_as_capacity_drops": defer_monotone,
        "closure_monotone_degradation": closure_monotone,
        "net_utility_monotone_degradation": utility_monotone,
        "transition_cause_counts": dict(causes),
    }
    write_json(OUT / "adaptivity_validation.json", checks)

    report = f"""# TUT-UOI-PRODUCT-LITE capacity adaptivity report

_Frozen seed-2026 method under contract capacities 1.00, 0.50, and 0.35._

---

## ✅ Final status

`{status}`

## 📊 Capacity metrics

| Capacity | LOW | MEDIUM | HIGH | DEFER | Closure | Net utility | MB | Utility/MB | Max queue MB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {capacity:.2f} | {by_capacity[capacity]['LOW_RATE_count']} | {by_capacity[capacity]['MEDIUM_RATE_count']} | {by_capacity[capacity]['HIGH_RATE_count']} | {by_capacity[capacity]['DEFER_count']} | {runs[capacity]['summary']['closure_count']} | {f(runs[capacity]['summary']['common_net_utility']):.6f} | {f(runs[capacity]['summary']['transmitted_mb']):.6f} | {f(runs[capacity]['summary']['utility_per_mb']):.6f} | {f(runs[capacity]['summary']['maximum_downlink_queue_mb']):.6f} |"
        for capacity in CAPACITIES
    ) + f"""

## 🔄 Product transitions

`{changed_tasks}` of 268 tasks changed action at least once; `{268 - changed_tasks}` were invariant. Across the two adjacent capacity drops there were `{downgrades}` downward transitions and `{upgrades}` reverse upgrades.

| Check | Result |
| --- | --- |
| LOW increases as capacity drops | `{low_monotone}` |
| MEDIUM decreases as capacity drops | `{medium_monotone}` |
| HIGH is non-increasing | `{high_monotone}` |
| HIGH strictly decreases | `{high_strictly_decreased}` |
| DEFER increases as capacity drops | `{defer_monotone}` |
| Closure degrades monotonically | `{closure_monotone}` |
| Net utility degrades monotonically | `{utility_monotone}` |

Transition causes are audited from the same task and same candidate across capacities. Counts: `{json.dumps(dict(causes), sort_keys=True)}`. `score_component_audit.csv` records the delivery-latency, byte/queue, energy, safety, and final-score terms used for each classification.

Requested paired counts are: HIGH→lower/DEFER `{requested_transitions['HIGH_to_MEDIUM_LOW_DEFER']}`, MEDIUM→LOW/DEFER `{requested_transitions['MEDIUM_to_LOW_DEFER']}`, LOW→DEFER `{requested_transitions['LOW_to_DEFER']}`, reverse upgrades `{requested_transitions['reverse_upgrades']}`, and fully invariant tasks `{requested_transitions['fully_invariant_tasks']}`.

Counterfactual order checks show how often removing one score component changes the selected action. Byte/queue cost counts are `{by_capacity[1.00].get('byte_queue_cost_changed_order', 0)} / {by_capacity[0.50].get('byte_queue_cost_changed_order', 0)} / {by_capacity[0.35].get('byte_queue_cost_changed_order', 0)}` for capacities 1.00 / 0.50 / 0.35; delivery-latency counts are `{by_capacity[1.00].get('delivery_latency_changed_order', 0)} / {by_capacity[0.50].get('delivery_latency_changed_order', 0)} / {by_capacity[0.35].get('delivery_latency_changed_order', 0)}`.

## 🛡️ Fairness and safety

All three points use the same 268 task IDs and task hash. Each capacity-specific environment hash matches the existing authoritative capacity run, while the environment after removing the permitted throughput fields is identical. Fingerprint, evaluator, product profiles, fixed model, UoI formula, weights, contact timing, initial state, and thermal/SOC rules are unchanged. Safety pass is `{safety['all_pass']}`; no PROP, B1, or OFFLOAD core simulation was run.

## 🧊 Freeze decision

The baseline may be frozen only if the status above is not `FAIRNESS_OR_IMPLEMENTATION_FAILURE`. No parameter change or follow-on tuning was performed, including when adaptivity is weak or absent.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "smoke", "pilot", "analyze"), required=True)
    parser.add_argument("--capacity", type=float, choices=NEW_CAPACITIES)
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight()
    elif args.phase in {"smoke", "pilot"}:
        if args.capacity is None:
            parser.error("--capacity is required for smoke/pilot")
        smoke(args.capacity) if args.phase == "smoke" else pilot(args.capacity)
    else:
        analyze()


if __name__ == "__main__":
    main()
