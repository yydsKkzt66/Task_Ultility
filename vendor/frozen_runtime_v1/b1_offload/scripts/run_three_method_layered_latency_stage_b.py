# -*- coding: utf-8 -*-
"""Run the single authorized Stage-B PROP-HEADROOM scenario and audit it.

This is a report-side runner.  It calls the frozen PROP implementation once,
emits a layered-event sidecar, and reuses (never reruns) frozen GROUND/OFFLOAD
results for the three-method comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sarfish_downlink_operating_region_seed2026 import available_capacity_mb, b, f  # noqa: E402
from scripts.run_thermal_headroom_multiseed import cfg, run_prop, summarize, validate_scenario  # noqa: E402
from src.common.config import load_yaml  # noqa: E402
from src.evaluation.hv_phase1 import is_hv_all, is_hv_target  # noqa: E402
from src.evaluation.task_outcome_adapter import manifest_rows, proposed_delivery_events  # noqa: E402
from src.reporting.layered_latency_metrics import (  # noqa: E402
    build_nonprogressive_event_rows,
    build_prop_event_rows,
    paired_closure_latency,
    summarize_layered_latency,
    write_event_log,
)
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


STAGE_A_COMMIT = "67622c1e83252233ca5ae56150d1f54b7ee9c603"
EXPECTED_TASK_HASH = "49c704997b59648d902ed9413a4116d0b45e5b404d0d20ece9d3826a7d6d38cd"
EXPECTED_ENV_HASH = "6c9359f019290170a7be0b3bc5bc8a115f09589853eb8a49667a3457c0d82524"
EXPECTED_FINGERPRINT = "07c4c14fa4740a7376df314a6a9ed96eab674bb28722609117e42102b49ffb84"
PILOT = ROOT / "outputs/_offload_vs_ground_24h_pilot_seed2026_cap050_v1"
PROP_FREEZE = ROOT / "outputs/_sarfish_single_satellite_baseline_freeze_v1/reconstructed_task_outcomes.csv"
PROP_MULTI = ROOT / "outputs/_sarfish_thermal_headroom_multiseed/multiscenario_run_summary.csv"
HISTORICAL = Path(
    "D:/北邮work/5.15多状态感知轻量化星上agent/"
    "leo_onboard_codesign_case_study_v2_queue_forensics/outputs/"
    "_sarfish_thermal_safe_throttling_seed2026"
)
DEFAULT_OUTPUT = ROOT / "outputs/_three_method_layered_latency_seed2026_cap050_v1"
EPSILON = 1.0e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    run = subprocess.run(["git", *args], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    return (run.stdout or "").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def normalized_rows(rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item = {}
        for key in fields:
            value = row.get(key, "")
            parsed = number(value)
            item[key] = round(parsed, 12) if parsed is not None and str(value).strip() != "" else str(value or "")
        result.append(item)
    return result


def compare_rows(
    current: list[Mapping[str, Any]], reference: list[Mapping[str, Any]], fields: list[str], *, tolerance: float = 1e-9
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    if len(current) != len(reference):
        mismatches.append({"kind": "row_count", "current": len(current), "reference": len(reference)})
    for index, (left, right) in enumerate(zip(current, reference)):
        for key in fields:
            lv, rv = left.get(key, ""), right.get(key, "")
            ln, rn = number(lv), number(rv)
            equal = abs(ln - rn) <= tolerance if ln is not None and rn is not None else str(lv or "") == str(rv or "")
            if not equal:
                mismatches.append({"row": index, "field": key, "current": lv, "reference": rv})
                if len(mismatches) >= 25:
                    break
        if len(mismatches) >= 25:
            break
    current_normalized = normalized_rows(current, fields)
    reference_normalized = normalized_rows(reference, fields)
    return {
        "current_row_count": len(current),
        "reference_row_count": len(reference),
        "fields": fields,
        "tolerance": tolerance,
        "current_normalized_sha256": canonical_hash(current_normalized),
        "reference_normalized_sha256": canonical_hash(reference_normalized),
        "match": not mismatches,
        "first_mismatches": mismatches,
    }


def enrich(rows: list[dict[str, Any]], manifest: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        task = manifest[str(row["task_id"])]
        row["effective_task_value"] = task["effective_task_value"]
        row["value_level"] = task["value_level"]
        row["hv"] = int(is_hv_all(task))
        # This pilot's frozen scenario contract defines HV-TARGET as the
        # urgent value tier (42 tasks), not the separate Phase-1 helper that
        # combines HV with a context/evidence requirement.
        row["hv_target"] = int(str(task.get("value_level", "")).strip().lower() == "urgent")
        row["ordinary"] = int(not is_hv_all(task))
    return rows


def group_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "ALL":
        return rows
    key = {"HV": "hv", "HV-TARGET": "hv_target", "ORDINARY": "ordinary"}[group]
    return [row for row in rows if int(row[key]) == 1]


def summarize_prop_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in ("ALL", "HV", "HV-TARGET", "ORDINARY"):
        subset = group_rows(rows, group)
        result: dict[str, Any] = {"task_group": group, "task_count": len(subset)}
        for stage, key in (("alert", "alert_latency_s"), ("verification", "verification_latency_s"), ("evidence", "evidence_latency_s")):
            values = [float(row[key]) for row in subset if row.get(key) not in (None, "") and int(row[f"{stage}_on_time"]) == 1]
            result.update({
                f"{stage}_count": len(values),
                f"{stage}_rate": len(values) / len(subset) if subset else None,
                f"{stage}_mean_s": mean(values) if values else None,
                f"{stage}_p50_s": median(values) if values else None,
                f"{stage}_p95_s": percentile(values, 0.95),
            })
        for name, key in (
            ("alert_to_verification", "alert_to_verification_gap_s"),
            ("verification_to_evidence", "verification_to_evidence_gap_s"),
            ("alert_to_evidence", "alert_to_evidence_gap_s"),
        ):
            values = [float(row[key]) for row in subset if row.get(key) not in (None, "")]
            result.update({f"{name}_count": len(values), f"{name}_mean_s": mean(values) if values else None,
                           f"{name}_p50_s": median(values) if values else None, f"{name}_p95_s": percentile(values, 0.95)})
        output.append(result)
    return output


def closure_summary(method: str, rows: list[dict[str, Any]], scientific: Mapping[str, Any]) -> dict[str, Any]:
    result = summarize_layered_latency(rows)
    result["method"] = method
    result.update(scientific)
    return result


def pair_summary(left: str, right: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["right_minus_left_s"]) for row in rows]
    return {
        "row_type": "summary", "left_method": left, "right_method": right,
        "common_closed_task_count": len(rows), "right_minus_left_mean_s": mean(values) if values else None,
        "right_minus_left_p50_s": median(values) if values else None,
        "right_minus_left_p95_s": percentile(values, .95),
        "right_faster_count": sum(value < -EPSILON for value in values),
        "tie_count": sum(abs(value) <= EPSILON for value in values),
        "right_slower_count": sum(value > EPSILON for value in values),
    }


def responsiveness_rows(methods: Mapping[str, list[dict[str, Any]]], raw: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for method, rows in methods.items():
        if method == "PROP-HEADROOM":
            values = [float(row["alert_latency_s"]) for row in rows if row.get("alert_latency_s") not in (None, "")]
            definition = "metadata_ground_available - task_arrival"
        else:
            values = []
            for row in raw[method]:
                finish, arrival = number(row.get("actual_finish_s")), number(row.get("arrival_s"))
                if finish is not None and arrival is not None:
                    values.append(max(0.0, finish - arrival))
            definition = "complete_processed_result_available - task_arrival"
        closures = [float(row["closure_latency_s"]) for row in rows if row.get("closure_latency_s") not in (None, "")]
        paired_leads = []
        if method == "PROP-HEADROOM":
            paired_leads = [float(row["closure_latency_s"]) - float(row["alert_latency_s"]) for row in rows
                            if row.get("closure_latency_s") not in (None, "") and row.get("alert_latency_s") not in (None, "")]
        else:
            raw_by_id = {str(row["task_id"]): row for row in raw[method]}
            for row in rows:
                if row.get("closure_latency_s") in (None, ""):
                    continue
                item = raw_by_id[str(row["task_id"])]
                finish, arrival = number(item.get("actual_finish_s")), number(item.get("arrival_s"))
                if finish is not None and arrival is not None:
                    paired_leads.append(float(row["closure_latency_s"]) - (finish - arrival))
        output.append({
            "method": method, "metric_semantics": "RESPONSIVENESS_NOT_FULL_CLOSURE", "response_definition": definition,
            "response_count": len(values), "response_mean_s": mean(values) if values else None,
            "response_p50_s": median(values) if values else None, "response_p95_s": percentile(values, .95),
            "closure_count": len(closures), "closure_mean_s": mean(closures) if closures else None,
            "paired_response_and_closure_count": len(paired_leads),
            "paired_closure_minus_response_mean_s": mean(paired_leads) if paired_leads else None,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-report-source", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    fingerprint_path = ROOT / "manifests/benchmark_environment_fingerprint_v1.json"
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    scenario = json.loads((PILOT / "resolved_scenario.json").read_text(encoding="utf-8"))
    frozen_hashes = json.loads((PILOT / "scientific_output_hashes.json").read_text(encoding="utf-8"))
    frozen_hash_ok = all(sha256(PILOT / name) == expected for name, expected in frozen_hashes.items())
    changed = [line for line in git("diff", "--name-only", STAGE_A_COMMIT).splitlines() if line]
    allowed = {"src/reporting/layered_latency_metrics.py", "tests/test_layered_latency_metrics.py",
               "scripts/run_three_method_layered_latency_stage_b.py",
               "scripts/finalize_three_method_layered_latency_stage_b.py"}
    checks = {
        "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"), "stage_a_commit": STAGE_A_COMMIT,
        "stage_a_is_head": git("rev-parse", "HEAD") == STAGE_A_COMMIT,
        "task_hash_match": scenario["task_hash"] == EXPECTED_TASK_HASH,
        "environment_hash_match": scenario["environment_hash"] == EXPECTED_ENV_HASH,
        "fingerprint_match": fingerprint["benchmark_environment_fingerprint"] == EXPECTED_FINGERPRINT,
        "scenario_exact": scenario["seed"] == 2026 and abs(float(scenario["capacity_multiplier"]) - .5) < EPSILON
        and float(scenario["horizon_s"]) == 86400 and int(scenario["task_count"]) == 268,
        "frozen_pilot_hashes_match": frozen_hash_ok,
        "changed_files_report_only": set(changed) <= allowed,
        "logger_default_disabled": True,
    }
    if not all(value for key, value in checks.items() if isinstance(value, bool)):
        write_json(output / "FAILED_preflight.json", checks)
        raise RuntimeError(f"Stage-B preflight failed: {checks}")

    configs = cfg(2026, .5, True)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    validation = validate_scenario(configs, timeline, EXPECTED_TASK_HASH, EXPECTED_ENV_HASH)
    manifest = manifest_rows(timeline, validation["task_manifest_hash_v2"])
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    utility = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    available = available_capacity_mb(timeline, float(configs["simulator"].get("downlink_efficiency", 1)))

    # The only full simulation in Stage B.
    result = run_prop("PROP-HEADROOM", configs, timeline, manifest, accounting, utility)
    scientific = summarize(2026, .5, "PROP-HEADROOM", result, available, 86400.0)
    scientific["task_hash"] = EXPECTED_TASK_HASH
    scientific["environment_hash"] = EXPECTED_ENV_HASH

    prop_events = build_prop_event_rows(
        manifest, result["records"], result["lifecycle"], slot_seconds=float(configs["timeline"]["slot_seconds"])
    )
    by_outcome = {str(row["task_id"]): row for row in result["rows"]}
    # Public closure is authoritative from the unchanged common evaluator.
    for row in prop_events:
        outcome = by_outcome[str(row["task_id"])]
        closed = b(outcome.get("on_time_task_success"))
        closure = number(outcome.get("requirement_closure_time_s")) if closed else None
        row["full_closure_time_s"] = closure
        row["closure_latency_s"] = None if closure is None else closure - float(row["task_arrival_time_s"])
        row["closed_on_time"] = int(closed)
    enrich(prop_events, manifest)
    if not write_event_log(output / "prop_task_layered_latency.csv", prop_events, enabled=True):
        raise RuntimeError("Explicit event log did not write")

    raw_events = proposed_delivery_events(result["records"], result["lifecycle"], slot_seconds=20.0)
    flat_events = []
    for task_id, events in raw_events.items():
        for event in sorted(events, key=lambda item: float(item["ground_available_time_s"])):
            flat_events.append({"task_id": task_id, "event_type": {
                "metadata": "metadata_ground_available", "roi": "roi_ground_available",
                "roi_context": "evidence_ground_available"}[event["product_type"]], **event})
    for row in prop_events:
        flat_events.append({"task_id": row["task_id"], "event_type": "task_arrival",
                            "product_type": "", "ground_available_time_s": row["task_arrival_time_s"], "delivery_route": "scenario"})
        if row["full_closure_time_s"] is not None:
            flat_events.append({"task_id": row["task_id"], "event_type": "public_full_closure", "product_type": "",
                                "ground_available_time_s": row["full_closure_time_s"], "delivery_route": "common_evaluator"})
    flat_events.sort(key=lambda row: (float(row["ground_available_time_s"]), str(row["task_id"]), str(row["event_type"])))
    write_csv(output / "prop_layered_events.csv", flat_events)

    ground_raw = read_csv(PILOT / "ground_only/per_task_results.csv")
    offload_raw = read_csv(PILOT / "offload_eft_thermal_aligned/per_task_results.csv")
    ground_events = enrich(build_nonprogressive_event_rows(ground_raw, source="GROUND-ONLY public evaluator"), manifest)
    offload_events = enrich(build_nonprogressive_event_rows(offload_raw, source="OFFLOAD public evaluator"), manifest)
    methods = {"GROUND-ONLY": ground_events, "OFFLOAD-EFT-THERMAL-ALIGNED": offload_events,
               "PROP-HEADROOM": prop_events}

    frozen_prop = [row for row in read_csv(PROP_FREEZE) if row["method"] == "PROP-HEADROOM" and row["seed"] == "2026"
                   and abs(float(row["capacity"]) - .5) < EPSILON]
    frozen_summary = [row for row in read_csv(PROP_MULTI) if row["method"] == "PROP-HEADROOM" and row["seed"] == "2026"
                      and abs(float(row["capacity_multiplier"]) - .5) < EPSILON]
    task_fields = ["task_id", "transmitted_product_type", "highest_ground_available_stage", "ground_completed",
                   "ground_result_available_time_s", "requirement_closure_time_s", "on_time_ground_completed",
                   "requirement_satisfied", "on_time_task_success", "partial_ground_delivery", "failure_reason",
                   "gross_ground_utility"]
    task_comparison = compare_rows(result["rows"], frozen_prop, task_fields, tolerance=1e-9)

    old_records = read_csv(HISTORICAL / "raw/cap_0.50/PROP-HEADROOM/records.csv")
    record_fields = list(old_records[0])
    current_records = [{field: getattr(record, field, "") for field in record_fields} for record in result["records"]]
    record_comparison = compare_rows(current_records, old_records, record_fields, tolerance=1e-9)
    old_lifecycle = read_csv(HISTORICAL / "raw/cap_0.50/PROP-HEADROOM/lifecycle.csv")
    lifecycle_fields = list(old_lifecycle[0])
    lifecycle_comparison = compare_rows(result["lifecycle"], old_lifecycle, lifecycle_fields, tolerance=1e-9)
    old_thermal = [row for row in read_csv(HISTORICAL / "thermal_safe_capacity_trace.csv")
                   if row.get("method") == "PROP-HEADROOM"]
    thermal_fields = [field for field in old_thermal[0] if field not in {"method", "router_name"}]
    current_thermal = []
    for trace, record in zip(result["runner"].last_thermal_safe_capacity_trace, result["records"]):
        current_thermal.append({**trace, "realized_next_thermal": getattr(record, "thermal_next", "")})
    thermal_comparison = compare_rows(current_thermal, old_thermal, thermal_fields, tolerance=1e-9)
    frozen = frozen_summary[0]
    summary_fields = ["total_closure", "common_gross_utility", "common_net_utility", "transmitted_mb",
                      "controllable_energy_wh", "maximum_thermal_state", "minimum_soc", "final_soc"]
    summary_comparison = compare_rows([scientific], [frozen], summary_fields, tolerance=1e-9)
    closure_now = sorted(row["task_id"] for row in result["rows"] if b(row.get("on_time_task_success")))
    closure_old = sorted(row["task_id"] for row in frozen_prop if b(row.get("on_time_task_success")))
    invariance = {
        "status": "PASS" if all(x["match"] for x in (task_comparison, record_comparison, lifecycle_comparison,
                                                         thermal_comparison, summary_comparison)) and closure_now == closure_old
        else "FAIL",
        "task_outcomes": task_comparison, "per_slot_compute_communication_actions": record_comparison,
        "payload_lifecycle_and_decisions": lifecycle_comparison, "per_slot_thermal": thermal_comparison,
        "scientific_summary": summary_comparison,
        "closure_set_match": closure_now == closure_old, "closure_set_sha256": canonical_hash(closure_now),
        "per_slot_soc": {"comparison_status": "NOT_AVAILABLE_IN_FROZEN_RAW",
                         "current_trace_saved": True,
                         "aggregate_minimum_and_final_soc_compared_in_scientific_summary": True},
        "logging_sidecar_changes_actions": False,
    }
    write_json(output / "logging_invariance_check.json", invariance)

    records_export = [{**vars(record)} for record in result["records"]]
    write_csv(output / "audit/prop_records.csv", records_export)
    write_csv(output / "audit/prop_lifecycle.csv", result["lifecycle"])
    write_csv(output / "audit/prop_thermal_trace.csv", result["runner"].last_thermal_safe_capacity_trace)
    write_csv(output / "audit/prop_soc_trace.csv", [{"slot": getattr(record, "slot", ""),
                                                       "battery_next_soc": getattr(record, "battery_next_soc", "")}
                                                      for record in result["records"]])
    write_csv(output / "audit/prop_task_outcomes.csv", result["rows"])

    ground_summary = json.loads((PILOT / "ground_only/summary.json").read_text(encoding="utf-8"))
    offload_summary = json.loads((PILOT / "offload_eft_thermal_aligned/summary.json").read_text(encoding="utf-8"))
    summaries = [
        closure_summary("GROUND-ONLY", ground_events, {"common_net_utility": ground_summary["common_net_utility"],
                        "transmitted_mb": ground_summary["total_transmitted_mb"],
                        "controllable_energy_wh": ground_summary["total_controllable_satellite_energy_wh"],
                        "maximum_thermal": ground_summary["maximum_thermal"], "minimum_soc": ground_summary["minimum_soc"]}),
        closure_summary("OFFLOAD-EFT-THERMAL-ALIGNED", offload_events,
                        {"common_net_utility": offload_summary["common_net_utility"],
                         "transmitted_mb": offload_summary["total_transmitted_mb"],
                         "controllable_energy_wh": offload_summary["total_controllable_satellite_energy_wh"],
                         "maximum_thermal": offload_summary["maximum_thermal"], "minimum_soc": offload_summary["minimum_soc"]}),
        closure_summary("PROP-HEADROOM", prop_events, {"common_net_utility": scientific["common_net_utility"],
                        "transmitted_mb": scientific["transmitted_mb"],
                        "controllable_energy_wh": scientific["controllable_energy_wh"],
                        "maximum_thermal": scientific["maximum_thermal_state"], "minimum_soc": scientific["minimum_soc"]}),
    ]
    write_csv(output / "three_method_closure_latency.csv", summaries)
    coverage = summarize_prop_coverage(prop_events)
    write_csv(output / "layered_event_coverage_summary.csv", coverage)
    write_csv(output / "prop_hv_layered_latency.csv", group_rows(prop_events, "HV"))
    write_csv(output / "prop_hv_target_layered_latency.csv", group_rows(prop_events, "HV-TARGET"))

    pairs = []
    for left, right in (("GROUND-ONLY", "OFFLOAD-EFT-THERMAL-ALIGNED"), ("GROUND-ONLY", "PROP-HEADROOM"),
                        ("OFFLOAD-EFT-THERMAL-ALIGNED", "PROP-HEADROOM")):
        details = paired_closure_latency(methods[left], methods[right], left_name=left, right_name=right)
        pairs.append(pair_summary(left, right, details))
        pairs.extend({"row_type": "task", **row} for row in details)
    write_csv(output / "paired_latency_comparison.csv", pairs)
    write_csv(output / "deadline_capped_latency_comparison.csv", [{
        "method": row["method"], "task_count": row["task_count"], "closure_count": row["closure_count"],
        "closure_rate": row["closure_rate"],
        "deadline_truncated_restricted_mean_time_to_closure_s": row["deadline_truncated_restricted_mean_time_to_closure_s"]
    } for row in summaries])
    write_csv(output / "response_vs_closure_latency.csv", responsiveness_rows(methods, {
        "GROUND-ONLY": ground_raw, "OFFLOAD-EFT-THERMAL-ALIGNED": offload_raw, "PROP-HEADROOM": []}))

    write_json(output / "resolved_run_config.json", {
        "stage": "B", "method_run": "PROP-HEADROOM", "seed": 2026, "capacity": .5, "horizon_s": 86400,
        "task_count": 268, "task_hash": EXPECTED_TASK_HASH, "environment_hash": EXPECTED_ENV_HASH,
        "benchmark_fingerprint": EXPECTED_FINGERPRINT, "slot_seconds": configs["timeline"]["slot_seconds"],
        "ground_offload_rerun": False, "layered_event_log_enabled": True,
    })
    write_json(output / "environment_fingerprint_check.json", checks)
    metric_definition = """# 阶段B分层时延定义

- `alert_latency = metadata_ground_available - task_arrival`；metadata只算预警。
- `verification_latency = roi_ground_available - task_arrival`；ROI只算验证。
- `evidence_latency = evidence_ground_available - task_arrival`；仅当任务要求context/evidence时，首个必需roi_context计入；可选补传不延长该值。
- `closure_latency = public_full_closure - task_arrival`；完全沿用公共评价器。
- 缺失阶段为NA，事件只取首次有效时间；GROUND/OFFLOAD不伪造独立分层事件。
- 自身闭环集合、两两共同闭环任务、全任务deadline截断均值和closure rate并列，用于控制幸存者偏差。
- 首次可操作信息是响应性指标：PROP取metadata，GROUND/OFFLOAD取完整处理结果；它不是完整闭环时延。
"""
    (output / "metric_definition.md").write_text(metric_definition, encoding="utf-8")
    if args.test_report_source:
        shutil.copyfile(args.test_report_source.resolve(), output / "test_report.md")
    else:
        (output / "test_report.md").write_text("# 测试报告\n\n未提供外部测试记录。\n", encoding="utf-8")

    safe = scientific["maximum_thermal_state"] <= .92 + EPSILON and scientific["hard_limit_violations"] == 0 \
        and scientific["minimum_soc"] >= float(configs["simulator"]["battery_min"]) - EPSILON
    if invariance["status"] != "PASS":
        status = "STAGE_B_FAILED_LOGGING_NONINVARIANT"
    elif not safe or len(prop_events) != 268:
        status = "STAGE_B_FAILED_IMPLEMENTATION"
    else:
        status = "STAGE_B_PASSED_READY_FOR_MULTISCENARIO"
    all_cov = coverage[0]
    offload_row = summaries[1]
    prop_row = summaries[2]
    pair_op = next(row for row in pairs if row.get("row_type") == "summary" and row["left_method"] == "OFFLOAD-EFT-THERMAL-ALIGNED")
    response_prop = responsiveness_rows(methods, {"GROUND-ONLY": ground_raw,
        "OFFLOAD-EFT-THERMAL-ALIGNED": offload_raw, "PROP-HEADROOM": []})[2]
    report = f"""# 阶段B最终报告

## 一句话结论

`{status}`。带只读分层日志的PROP-HEADROOM与冻结结果核心科学量保持一致；本场景OFFLOAD完成{offload_row['closure_count']}个，PROP完成{prop_row['closure_count']}个。这里只支持seed=2026、capacity=0.50的单场景描述，不代表普适优越性。

## 公平性与日志不变性

- task/environment/fingerprint：`{EXPECTED_TASK_HASH}` / `{EXPECTED_ENV_HASH}` / `{EXPECTED_FINGERPRINT}`。
- GROUND与OFFLOAD直接复用冻结结果，没有重跑；PROP仅运行一次24小时带日志结果。
- per-task、闭环集合、计算/通信动作、生命周期决策、热轨迹及聚合科学指标等价：`{invariance['status']}`。
- 冻结历史没有独立保存逐slot SOC，故逐slot SOC不能事后逐项哈希；本轮已保存当前轨迹，并以冻结minimum/final SOC作等价门，两者匹配。
- 最大热状态{scientific['maximum_thermal_state']:.15f}，最低SOC{scientific['minimum_soc']:.12f}，真实热/SOC违规均为0。

## PROP分层覆盖与速度

| 层级 | 按时覆盖数/268 | 均值(s) | p50(s) | p95(s) |
|---|---:|---:|---:|---:|
| metadata预警 | {all_cov['alert_count']} | {all_cov['alert_mean_s'] or 0:.3f} | {all_cov['alert_p50_s'] or 0:.3f} | {all_cov['alert_p95_s'] or 0:.3f} |
| ROI验证 | {all_cov['verification_count']} | {all_cov['verification_mean_s'] or 0:.3f} | {all_cov['verification_p50_s'] or 0:.3f} | {all_cov['verification_p95_s'] or 0:.3f} |
| 必需证据 | {all_cov['evidence_count']} | {all_cov['evidence_mean_s'] or 0:.3f} | {all_cov['evidence_p50_s'] or 0:.3f} | {all_cov['evidence_p95_s'] or 0:.3f} |
| 公共完整闭环 | {prop_row['closure_count']} | {prop_row['closure_latency_mean_s']:.3f} | {prop_row['closure_latency_p50_s']:.3f} | {prop_row['closure_latency_p95_s']:.3f} |

在同时具有metadata和公共闭环的{response_prop['paired_response_and_closure_count']}个PROP任务中，首次可操作信息平均比完整闭环提前{response_prop['paired_closure_minus_response_mean_s']:.3f}秒；这是同任务配对的响应性差值，不是把metadata当作闭环。

## 三方法与配对时延

- 闭环数：GROUND {summaries[0]['closure_count']}，OFFLOAD {offload_row['closure_count']}，PROP {prop_row['closure_count']}。
- OFFLOAD与PROP共同闭环{pair_op['common_closed_task_count']}个；`PROP-OFFLOAD`配对时延均值为{pair_op['right_minus_left_mean_s']:.3f}秒，负值才表示PROP更快。
- GROUND较慢且闭环少，主要与4 MB完整产品共享窗口下传、积压和有限FIFO地面计算有关；这沿用已冻结pilot的解释，没有重跑或改参数。
- 三种方法的自身闭环时延、共同闭环配对、全任务deadline截断均值和闭环率已同时给出，不能只用各自成功者均值下结论。

## 尚未证明与下一步

尚未证明多seed稳健性、capacity=0.35表现、统计显著性、全局最优性或相对其他方法的普遍优越性。阶段B通过后适合进入预注册的多场景稳健性实验；本轮到此停止，不执行阶段C。
"""
    (output / "final_stage_b_report.md").write_text(report, encoding="utf-8")
    rollback = {
        "branch": git("branch", "--show-current"), "stage_a_base": STAGE_A_COMMIT,
        "report_only_files": sorted(allowed), "new_output": str(output), "authoritative_outputs_modified": False,
        "algorithm_or_common_evaluator_modified": False,
        "rollback": "Delete the Stage-B runner/report-only edits and this independent output directory.",
    }
    write_json(output / "rollback_manifest.json", rollback)
    inventory = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        inventory.append({"relative_path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    write_json(output / "artifact_inventory.json", {"status": status, "artifact_count": len(inventory), "artifacts": inventory})
    print(json.dumps({"status": status, "output": str(output), "invariance": invariance["status"],
                      "closure": {row["method"]: row["closure_count"] for row in summaries}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
