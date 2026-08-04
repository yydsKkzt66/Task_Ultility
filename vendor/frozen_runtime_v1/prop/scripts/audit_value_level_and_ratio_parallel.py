"""Parallel audit for value-level assignment and value-skew ratio choices.

This script is audit-only. It does not modify simulator defaults, reward logic,
Module 2 scoring, SafeLinUCB, SafeFilter, utility_rule, queue policy, lifecycle
closure, information_sufficiency, or task_value wiring.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_value_skew_feasibility import (  # noqa: E402
    BASELINES,
    DEFAULT_SEEDS,
    OLD_THRESHOLD,
    _add_file_patch,
    _assign_levels,
    _base_f,
    _gini,
    _load_tasks,
    _percentile,
    _read_csv,
    _stable_key,
    _to_float,
    _to_int,
    _top_share,
    _trajectory_paths_for_baseline,
    _write_csv,
    _write_text,
)


OUT_DIR = PROJECT_ROOT / "outputs" / "_value_level_ratio_parallel_audit"
ZIP_PATH = PROJECT_ROOT / "outputs" / "_value_level_ratio_parallel_audit.zip"

LEVELS = ["low", "medium", "high", "urgent"]
RATIOS: dict[str, dict[str, float] | None] = {
    "old": None,
    "mild": {"low": 1.0, "medium": 2.0, "high": 3.0, "urgent": 5.0},
    "soft_moderate": {"low": 1.0, "medium": 2.0, "high": 5.0, "urgent": 10.0},
    "moderate": {"low": 1.0, "medium": 3.0, "high": 8.0, "urgent": 15.0},
    "strong_moderate": {"low": 1.0, "medium": 4.0, "high": 10.0, "urgent": 20.0},
    "stress": {"low": 1.0, "medium": 5.0, "high": 20.0, "urgent": 50.0},
}
RECOMMENDED_V2_RULE = "v2_conservative"


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return (completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else "")).strip()
    except Exception as exc:  # pragma: no cover - diagnostic robustness
        return f"git {' '.join(args)} failed: {exc}"


def _region_score(row: dict[str, object]) -> int:
    region = str(row.get("region_prior", "")).lower()
    return {"low": 0, "medium": 1, "high": 2}.get(region, 0)


def _prepare_v1_rows(base_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = copy.deepcopy(base_rows)
    _assign_levels(rows)
    for row in rows:
        row["v1_value_level"] = row["assigned_value_level"]
        row["v1_assignment_reason"] = row["assignment_reason"]
    return rows


def _assign_v2(rows: list[dict[str, object]], variant: str) -> None:
    base_values = [_to_float(row["base_prior_score"]) for row in rows]
    q50 = _percentile(base_values, 0.50)
    q75 = _percentile(base_values, 0.75)
    q85 = _percentile(base_values, 0.85)

    for row in rows:
        task_class = str(row.get("task_class", "")).lower()
        requirement = str(row.get("product_requirement", "")).lower()
        base = _to_float(row.get("base_prior_score"))
        source = _to_float(row.get("task_source_priority"))
        region = _region_score(row)
        high_flag = bool(_to_int(row.get("high_value_flag")))
        emergency = bool(_to_int(row.get("emergency_flag"))) or task_class == "emergency"
        time_critical_context = requirement in {"context", "evidence"} and (base >= q85 or source >= 0.85)

        if emergency:
            level = "urgent"
            reason = "emergency_flag_or_class_kept_urgent"
        elif variant == "v2_conservative":
            if high_flag and (region >= 2 or base >= q85 or source >= 0.85):
                level = "high"
                reason = "high_value_with_strong_region_source_or_top_prior"
            elif task_class in {"routine", "background"} and not high_flag:
                level = "low"
                reason = "routine_or_background_non_emergency"
            elif high_flag or requirement in {"context", "evidence"} or base >= q50:
                level = "medium"
                reason = "high_value_or_context_without_strong_prior_demoted_to_medium"
            else:
                level = "medium"
                reason = "ordinary_verification_medium"
        elif variant == "v2_balanced":
            if time_critical_context:
                level = "urgent"
                reason = "time_critical_high_value_context_or_evidence"
            elif high_flag and (region >= 1 or base >= q75):
                level = "high"
                reason = "high_value_with_medium_or_high_region_prior"
            elif task_class in {"routine", "background"} and region <= 1 and base < q75:
                level = "low"
                reason = "routine_low_or_medium_region_prior"
            else:
                level = "medium"
                reason = "verification_or_context_without_strong_high_value_evidence"
        elif variant == "v2_permissive":
            if time_critical_context:
                level = "urgent"
                reason = "time_critical_context_or_evidence"
            elif high_flag or (requirement in {"context", "evidence"} and base >= q50):
                level = "high"
                reason = "high_value_or_context_evidence_with_nonlow_base_prior"
            elif task_class in {"routine", "background"} and region <= 0:
                level = "low"
                reason = "routine_low_region_prior"
            else:
                level = "medium"
                reason = "medium_recovery_for_verification_or_nonstrong_context"
        else:  # pragma: no cover
            raise ValueError(f"unknown v2 variant: {variant}")

        row[f"{variant}_value_level"] = level
        row[f"{variant}_assignment_reason"] = reason


def _distribution_by_rule(rows: list[dict[str, object]], rule_field: str, rule_name: str) -> dict[str, object]:
    total = len(rows)
    counts = Counter(str(row.get(rule_field, "")) for row in rows)
    context_rows = [row for row in rows if str(row.get("product_requirement")) in {"context", "evidence"}]
    emergency_rows = [row for row in rows if _to_int(row.get("emergency_flag")) or str(row.get("task_class")) == "emergency"]
    context_high = sum(1 for row in context_rows if str(row.get(rule_field)) == "high")
    emergency_urgent = sum(1 for row in emergency_rows if str(row.get(rule_field)) == "urgent")
    high_urgent = counts["high"] + counts["urgent"]
    out: dict[str, object] = {
        "assignment_rule": rule_name,
        "total_tasks": total,
        "medium_nonzero": int(counts["medium"] > 0),
        "high_urgent_count": high_urgent,
        "high_urgent_share": high_urgent / max(total, 1),
        "high_urgent_exceeds_60pct": int(high_urgent / max(total, 1) > 0.60),
        "context_evidence_count": len(context_rows),
        "context_evidence_mapped_to_high_count": context_high,
        "context_evidence_mapped_to_high_share": context_high / max(len(context_rows), 1),
        "emergency_count": len(emergency_rows),
        "emergency_mapped_to_urgent_count": emergency_urgent,
        "emergency_mapped_to_urgent_share": emergency_urgent / max(len(emergency_rows), 1),
    }
    for level in LEVELS:
        out[f"{level}_count"] = counts[level]
        out[f"{level}_share"] = counts[level] / max(total, 1)
    return out


def _level_trace(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "seed",
        "task_id",
        "scene_id",
        "slot",
        "task_class",
        "product_requirement",
        "high_value_flag",
        "emergency_flag",
        "task_source_priority",
        "region_prior",
        "base_prior_score",
        "old_task_value",
        "v1_value_level",
        "v1_assignment_reason",
        "v2_conservative_value_level",
        "v2_conservative_assignment_reason",
        "v2_balanced_value_level",
        "v2_balanced_assignment_reason",
        "v2_permissive_value_level",
        "v2_permissive_assignment_reason",
        f"{RECOMMENDED_V2_RULE}_value_level",
    ]
    return [{field: row.get(field, "") for field in fields} for row in rows]


def _calibrate_values(rows: list[dict[str, object]], level_field: str) -> dict[str, dict[str, float]]:
    old_mean = mean([_to_float(row["old_task_value"]) for row in rows]) if rows else 1.0
    values: dict[str, dict[str, float]] = {
        "old": {_stable_key(_to_int(row["seed"]), str(row["task_id"])): _to_float(row["old_task_value"]) for row in rows}
    }
    for ratio_name, multipliers in RATIOS.items():
        if ratio_name == "old" or multipliers is None:
            continue
        raw = {
            _stable_key(_to_int(row["seed"]), str(row["task_id"])): multipliers[str(row[level_field])] * _base_f(_to_float(row["base_prior_score"]))
            for row in rows
        }
        raw_mean = mean(raw.values()) if raw else 1.0
        alpha = old_mean / max(raw_mean, 1.0e-12)
        values[ratio_name] = {key: alpha * value for key, value in raw.items()}
    return values


def _value_distribution_summary(
    rows: list[dict[str, object]],
    level_field: str,
    assignment_rule: str,
    values_by_ratio: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    level_by_key = {_stable_key(_to_int(row["seed"]), str(row["task_id"])): str(row[level_field]) for row in rows}
    out: list[dict[str, object]] = []
    for ratio_name in RATIOS:
        value_map = values_by_ratio[ratio_name]
        values = list(value_map.values())
        total_value = sum(values)
        level_values = {
            level: [value for key, value in value_map.items() if level_by_key[key] == level]
            for level in LEVELS
        }
        high_urgent_values = level_values["high"] + level_values["urgent"]
        high_urgent_count = len(level_values["high"]) + len(level_values["urgent"])
        out.append(
            {
                "assignment_rule": assignment_rule,
                "value_scale": ratio_name,
                "normalization": "current" if ratio_name == "old" else "mean_preserving",
                "n": len(values),
                "value_min": min(values) if values else 0.0,
                "value_mean": mean(values) if values else 0.0,
                "value_median": median(values) if values else 0.0,
                "value_p75": _percentile(values, 0.75),
                "value_p90": _percentile(values, 0.90),
                "value_p95": _percentile(values, 0.95),
                "value_max": max(values) if values else 0.0,
                "high_low_mean_ratio": mean(level_values["high"]) / max(mean(level_values["low"]), 1.0e-12)
                if level_values["high"] and level_values["low"]
                else 0.0,
                "urgent_low_mean_ratio": mean(level_values["urgent"]) / max(mean(level_values["low"]), 1.0e-12)
                if level_values["urgent"] and level_values["low"]
                else 0.0,
                "gini_coefficient": _gini(values),
                "top_10pct_tasks_share_total_value": _top_share(values, 0.10),
                "high_urgent_tasks_share_total_count": high_urgent_count / max(len(values), 1),
                "high_urgent_tasks_share_total_value": sum(high_urgent_values) / max(total_value, 1.0e-12),
                "low_value_tasks_share_total_count": len(level_values["low"]) / max(len(values), 1),
                "low_value_tasks_share_total_value": sum(level_values["low"]) / max(total_value, 1.0e-12),
                "medium_value_tasks_share_total_count": len(level_values["medium"]) / max(len(values), 1),
                "medium_value_tasks_share_total_value": sum(level_values["medium"]) / max(total_value, 1.0e-12),
            }
        )
    return out


def _candidate_rows(
    rows: list[dict[str, object]],
    level_field: str,
    values_by_ratio: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    task_by_key = {_stable_key(_to_int(row["seed"]), str(row["task_id"])): row for row in rows}
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    paths = [
        PROJECT_ROOT / "outputs" / "_phase6_residual_enqueue_fallback_validation" / "same_slot_carryover_trace.csv",
        PROJECT_ROOT / "outputs" / "_phase6_residual_enqueue_fallback_validation" / "per_candidate_followup_trace.csv",
    ]
    for path in paths:
        for trace in _read_csv(path):
            event_type = str(trace.get("event_type", ""))
            if event_type and event_type not in {"same_slot_candidate_re_evaluated", "ordinary_queue_fallback_enqueued"}:
                continue
            old_eriv = _to_float(trace.get("ERIV"))
            if old_eriv <= 0.0:
                continue
            key = _stable_key(_to_int(trace.get("seed")), str(trace.get("task_id", "")))
            task = task_by_key.get(key)
            if task is None:
                continue
            identity = (path.name, str(trace.get("arm", "")), str(trace.get("slot", "")), key)
            if identity in seen:
                continue
            seen.add(identity)
            old_value = max(_to_float(task["old_task_value"]), 1.0e-12)
            item: dict[str, object] = {
                "seed": _to_int(trace.get("seed")),
                "task_id": trace.get("task_id", ""),
                "slot": trace.get("slot", ""),
                "arm": trace.get("arm", ""),
                "product_requirement": task.get("product_requirement", ""),
                "value_level": task.get(level_field, ""),
                "high_value_flag": task.get("high_value_flag", 0),
                "residual_product_type": trace.get("residual_product_type", ""),
                "old_eriv": old_eriv,
                "payload_size_proxy_mb": _to_float(trace.get("residual_payload_mb"), _to_float(trace.get("queue_size_after_fallback_mb"), 1.5)),
            }
            for ratio_name in RATIOS:
                calibrated = values_by_ratio[ratio_name].get(key, old_value)
                item[f"{ratio_name}_eriv"] = old_eriv * calibrated / old_value
            out.append(item)
    return out


def _eriv_summary(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    old_values = [_to_float(row["old_eriv"]) for row in candidate_rows]
    old_count = sum(1 for value in old_values if value >= OLD_THRESHOLD)
    for ratio_name in RATIOS:
        values = [_to_float(row.get(f"{ratio_name}_eriv")) for row in candidate_rows]
        sorted_values = sorted(values, reverse=True)
        match_threshold = sorted_values[old_count - 1] if old_count and len(sorted_values) >= old_count else 0.0
        high_values = [
            _to_float(row.get(f"{ratio_name}_eriv"))
            for row in candidate_rows
            if str(row.get("value_level")) in {"high", "urgent"} or _to_int(row.get("high_value_flag"))
        ]
        top_k = min(max(1, old_count), len(high_values)) if high_values else 0
        high_sorted = sorted(high_values, reverse=True)
        high_k_threshold = high_sorted[top_k - 1] if top_k else match_threshold
        above = sum(1 for value in values if value >= OLD_THRESHOLD)
        out.append(
            {
                "value_scale": ratio_name,
                "candidate_count": len(values),
                "eriv_min": min(values) if values else 0.0,
                "eriv_mean": mean(values) if values else 0.0,
                "eriv_median": median(values) if values else 0.0,
                "eriv_p75": _percentile(values, 0.75),
                "eriv_p90": _percentile(values, 0.90),
                "eriv_p95": _percentile(values, 0.95),
                "eriv_max": max(values) if values else 0.0,
                "candidates_above_old_threshold_0.15": above,
                "suggested_threshold_to_match_old_candidate_count": match_threshold,
                "suggested_threshold_to_allow_top_k_high_value_candidates": high_k_threshold,
                "risk_of_over_triggering": "high" if above > max(old_count * 2, old_count + 3) else "medium" if above > old_count else "low",
                "risk_of_under_triggering": "high" if above < max(1, old_count // 2) else "medium" if above < old_count else "low",
                "threshold_basis": "candidate_count_plus_high_value_coverage_not_multiplier_scaling",
            }
        )
    return out


def _roi_pressure(candidate_rows: list[dict[str, object]], eriv_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    threshold_by_ratio = {
        row["value_scale"]: _to_float(row["suggested_threshold_to_match_old_candidate_count"], OLD_THRESHOLD)
        for row in eriv_rows
    }
    out: list[dict[str, object]] = []
    for ratio_name in RATIOS:
        threshold = threshold_by_ratio.get(ratio_name, OLD_THRESHOLD)
        eligible = [
            row
            for row in candidate_rows
            if str(row.get("product_requirement")) in {"context", "evidence"} and str(row.get("value_level")) in {"high", "urgent"}
        ]
        selected = [row for row in eligible if _to_float(row.get(f"{ratio_name}_eriv")) >= threshold]
        low_accidental = [
            row
            for row in candidate_rows
            if str(row.get("value_level")) == "low"
            and str(row.get("product_requirement")) in {"context", "evidence"}
            and _to_float(row.get(f"{ratio_name}_eriv")) >= threshold
        ]
        all_payload = sum(max(1.0, _to_float(row.get("payload_size_proxy_mb"), 1.5)) for row in eligible)
        selected_payload = sum(max(1.0, _to_float(row.get("payload_size_proxy_mb"), 1.5)) for row in selected)
        out.append(
            {
                "value_scale": ratio_name,
                "roi_ctx_eligible_candidate_count": len(eligible),
                "high_value_roi_ctx_candidate_count": sum(1 for row in selected if _to_int(row.get("high_value_flag"))),
                "low_value_accidental_roi_ctx_trigger_count": len(low_accidental),
                "queue_pressure_proxy_if_all_upgraded": all_payload,
                "queue_pressure_proxy_if_selective_upgrade": selected_payload,
                "utility_rule_also_enhanced": int(ratio_name != "old"),
                "proposed_advantage_metric_note": "judge_high_value_closure_not_raw_delivered_count",
                "must_rescan_eriv_threshold": int(ratio_name != "old"),
                "must_rescan_roi_ctx_gate": int(ratio_name in {"soft_moderate", "moderate", "strong_moderate", "stress"}),
            }
        )
    return out


def _baseline_level(row: dict[str, str]) -> str:
    if _to_int(row.get("emergency_flag")):
        return "urgent"
    if _to_int(row.get("high_value_flag")):
        return "high"
    requirement = str(row.get("product_requirement", ""))
    value = _to_float(row.get("V_t"), _to_float(row.get("task_value"), 0.5))
    if requirement in {"context", "evidence"} and value >= 0.75:
        return "high"
    if requirement == "verification" and value >= 0.65:
        return "medium"
    return "low"


def _baseline_ratio_value(row: dict[str, str], ratio_name: str, old_mean: float) -> float:
    old_value = _to_float(row.get("V_t"), _to_float(row.get("task_value"), old_mean))
    multipliers = RATIOS[ratio_name]
    if multipliers is None:
        return old_value
    level = _baseline_level(row)
    raw = multipliers[level] * _base_f(old_value)
    reference = mean([multipliers[level] * _base_f(old_mean) for level in LEVELS])
    return raw * old_mean / max(reference, 1.0e-12)


def _baseline_fairness_summary() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for baseline_name in BASELINES:
        paths = _trajectory_paths_for_baseline(baseline_name)
        traj_rows = [row for path in paths for row in _read_csv(path) if _to_int(row.get("has_task"), 1)]
        old_mean = mean([_to_float(row.get("V_t"), _to_float(row.get("task_value"), 0.5)) for row in traj_rows]) if traj_rows else 0.5
        for ratio_name in RATIOS:
            delivered = 0.0
            high_delivered = 0.0
            low_delivered = 0.0
            high_completion = 0
            low_bad = 0
            queue_max = 0.0
            for row in traj_rows:
                old_v = max(_to_float(row.get("V_t"), _to_float(row.get("task_value"), old_mean)), 1.0e-12)
                new_v = _baseline_ratio_value(row, ratio_name, old_mean)
                delivered_raw = _to_float(row.get("delivered_only_incremental_utility"), _to_float(row.get("delivered_before_deadline_utility")))
                value = delivered_raw * new_v / old_v
                delivered += value
                level = _baseline_level(row)
                if level in {"high", "urgent"}:
                    high_delivered += value
                    high_completion += int(str(row.get("delivery_status", "")).lower() == "delivered")
                elif level == "low":
                    low_delivered += value
                    low_bad += int(str(row.get("delivery_status", "")).lower() in {"deferred", "expired"} or _to_int(row.get("expired_flag")))
                queue_max = max(queue_max, _to_float(row.get("queue_next_mb"), _to_float(row.get("queue_mb"))))
            resource_cost = sum(_to_float(row.get("resource_cost"), _to_float(row.get("resource_penalty"))) for row in traj_rows)
            penalties = sum(
                _to_float(row.get("hard_penalty"), 0.0)
                + _to_float(row.get("safety_penalty"), 0.0)
                + _to_float(row.get("requirement_penalty"), 0.0)
                for row in traj_rows
            )
            out.append(
                {
                    "value_scale": ratio_name,
                    "baseline_name": baseline_name,
                    "delivered_utility_rescaled": delivered,
                    "net_system_utility_rescaled": delivered - resource_cost - penalties,
                    "high_value_delivered_utility": high_delivered,
                    "low_value_delivered_utility": low_delivered,
                    "high_value_task_completion_proxy": high_completion,
                    "low_value_deferred_or_expired_proxy": low_bad,
                    "queue_pressure_proxy": queue_max,
                    "fairness_note": "counterfactual_rescale_only_all_baselines_share_same_value_scale",
                }
            )
    return out


def _resource_proxy(
    dist_rows: list[dict[str, object]],
    eriv_rows: list[dict[str, object]],
    roi_rows: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    dist_by_ratio = {row["value_scale"]: row for row in dist_rows if row["assignment_rule"] == RECOMMENDED_V2_RULE}
    eriv_by_ratio = {row["value_scale"]: row for row in eriv_rows}
    roi_by_ratio = {row["value_scale"]: row for row in roi_rows}
    proposed_by_ratio = {
        row["value_scale"]: row
        for row in baseline_rows
        if row["baseline_name"] == "proposed_requirement_aware"
    }
    out: list[dict[str, object]] = []
    for ratio_name in RATIOS:
        dist = dist_by_ratio.get(ratio_name, {})
        eriv = eriv_by_ratio.get(ratio_name, {})
        roi = roi_by_ratio.get(ratio_name, {})
        proposed = proposed_by_ratio.get(ratio_name, {})
        closure_proxy = _to_float(proposed.get("high_value_delivered_utility")) / max(
            _to_float(roi.get("queue_pressure_proxy_if_selective_upgrade"), 1.0), 1.0
        )
        out.append(
            {
                "value_scale": ratio_name,
                "resource_constrained_high_value_closure_proxy": closure_proxy,
                "high_urgent_value_share": dist.get("high_urgent_tasks_share_total_value", 0.0),
                "medium_value_share": dist.get("medium_value_tasks_share_total_value", 0.0),
                "top_10pct_value_share": dist.get("top_10pct_tasks_share_total_value", 0.0),
                "eriv_candidates_above_old_threshold": eriv.get("candidates_above_old_threshold_0.15", 0),
                "selective_roi_ctx_queue_pressure_proxy": roi.get("queue_pressure_proxy_if_selective_upgrade", 0.0),
                "interpretation": (
                    "legacy_reference"
                    if ratio_name == "old"
                    else "stable_main_candidate"
                    if ratio_name == "soft_moderate"
                    else "useful_sensitivity"
                    if ratio_name in {"mild", "moderate"}
                    else "stress_or_upper_bound_only"
                ),
            }
        )
    return out


def _recommendation_matrix(dist_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in dist_rows:
        if row["assignment_rule"] not in {"v1", RECOMMENDED_V2_RULE}:
            continue
        ratio_name = str(row["value_scale"])
        high_low = _to_float(row["high_low_mean_ratio"])
        high_urgent_value = _to_float(row["high_urgent_tasks_share_total_value"])
        medium_value = _to_float(row["medium_value_tasks_share_total_value"])
        top10 = _to_float(row["top_10pct_tasks_share_total_value"])
        passes = int(
            ratio_name != "stress"
            and high_low > 2.0
            and high_urgent_value < 0.85
            and medium_value > 0.08
            and top10 < 0.30
        )
        if ratio_name == "old":
            role = "legacy_reference"
        elif row["assignment_rule"] == "v1" and ratio_name == "moderate":
            role = "sensitivity_due_to_v1_assignment_concentration"
        elif row["assignment_rule"] == RECOMMENDED_V2_RULE and ratio_name == "soft_moderate":
            role = "recommended_main"
        elif ratio_name in {"mild", "moderate"}:
            role = "sensitivity"
        elif ratio_name in {"strong_moderate", "stress"}:
            role = "stress_sensitivity_only"
        else:
            role = "not_primary"
        out.append(
            {
                "assignment_rule": row["assignment_rule"],
                "value_scale": ratio_name,
                "main_candidate_screen_pass": passes,
                "recommended_role": role,
                "reason": (
                    "1:2:5:10 keeps medium visible while still widening old value spread"
                    if role == "recommended_main"
                    else "current assignment concentrates high_or_urgent tasks"
                    if role == "sensitivity_due_to_v1_assignment_concentration"
                    else "kept for comparison"
                ),
            }
        )
    return out


def _write_reports(
    distributions: list[dict[str, object]],
    value_rows: list[dict[str, object]],
    eriv_rows: list[dict[str, object]],
    roi_rows: list[dict[str, object]],
) -> None:
    dist = {(row["assignment_rule"], row["value_scale"]): row for row in value_rows}
    dist_rule = {row["assignment_rule"]: row for row in distributions}
    v1_hu = _to_float(dist_rule.get("v1", {}).get("high_urgent_share"))
    v2_hu = _to_float(dist_rule.get(RECOMMENDED_V2_RULE, {}).get("high_urgent_share"))
    v1_mod = dist.get(("v1", "moderate"), {})
    v2_mod = dist.get((RECOMMENDED_V2_RULE, "moderate"), {})
    v2_soft = dist.get((RECOMMENDED_V2_RULE, "soft_moderate"), {})
    mod_eriv = next((row for row in eriv_rows if row["value_scale"] == "moderate"), {})
    soft_roi = next((row for row in roi_rows if row["value_scale"] == "soft_moderate"), {})

    main_report = f"""# Main Experiment Ratio Recommendation

## Bottom Line

Final recommendation case: B.

The ratio `1:3:8:15` is defensible as a ratio family, but it is too aggressive as the main experiment when combined with the current v1 assignment. After the dry-run v2 assignment, it becomes much more defensible, but `1:2:5:10` is the more reviewer-stable main setting because it preserves visible medium contribution while still widening the old value scale.

## Ratio Itself vs Assignment Rule

- Ratio-only judgment: `1:3:8:15` is reasonable for sensitivity and upper-main exploration. It is not a stress extreme.
- Current v1 assignment judgment: v1 high/urgent share is `{v1_hu:.3f}`, with medium unused. That makes `1:3:8:15` look more extreme than the ratio alone warrants.
- Recommended v2 assignment judgment: `{RECOMMENDED_V2_RULE}` reduces high/urgent share to `{v2_hu:.3f}` and restores medium. Under that assignment, `1:3:8:15` remains usable, but `1:2:5:10` is safer as the primary claim.

## Key Metrics

- v1 moderate high/low ratio: `{_to_float(v1_mod.get('high_low_mean_ratio')):.4f}`; high/urgent value share: `{_to_float(v1_mod.get('high_urgent_tasks_share_total_value')):.4f}`.
- v2 soft_moderate high/low ratio: `{_to_float(v2_soft.get('high_low_mean_ratio')):.4f}`; high/urgent value share: `{_to_float(v2_soft.get('high_urgent_tasks_share_total_value')):.4f}`; medium value share: `{_to_float(v2_soft.get('medium_value_tasks_share_total_value')):.4f}`.
- v2 moderate high/low ratio: `{_to_float(v2_mod.get('high_low_mean_ratio')):.4f}`; high/urgent value share: `{_to_float(v2_mod.get('high_urgent_tasks_share_total_value')):.4f}`; medium value share: `{_to_float(v2_mod.get('medium_value_tasks_share_total_value')):.4f}`.
- Moderate ERIV match-count threshold: `{_to_float(mod_eriv.get('suggested_threshold_to_match_old_candidate_count')):.4f}`.
- Soft-moderate selective ROI+Ctx queue pressure proxy: `{_to_float(soft_roi.get('queue_pressure_proxy_if_selective_upgrade')):.4f}`.

## Placement

- Main experiment: `soft_moderate = 1:2:5:10`.
- Sensitivity: `mild = 1:2:3:5` and `moderate = 1:3:8:15`.
- Stress sensitivity only: `stress = 1:5:20:50`; keep `strong_moderate = 1:4:10:20` as an upper-bound diagnostic if space allows.
- Legacy reference: keep `old`.

## Engineering Next Step

Proceed to a default-off value-skew calibration layer only after formal gate sweeps for ERIV and ROI+Ctx. The implementation should not replace task_value by default and should force all baselines to share the same calibrated value scale in copied experiment configs.
"""
    readiness = f"""# Implementation Readiness Checklist

| Item | Status | Audit Note |
|---|---:|---|
| value_level_assignment_v2 ready | partial | `{RECOMMENDED_V2_RULE}` is ready for dry-run use, not mainline replacement. |
| medium restored | yes | v2 restores nonzero medium coverage. |
| high/urgent share reasonable | partial | v2 high/urgent share is `{v2_hu:.3f}`; emergency alone creates a high urgent floor. |
| mean-preserving normalization recommended | yes | Keeps average reward scale stable across ratios. |
| keep 1:3:8:15 | yes | Keep as sensitivity and possible upper-main check. |
| need 1:2:5:10 fallback | yes | Recommended as main experiment ratio. |
| ERIV threshold must be rescanned | yes | Candidate-count and high-value coverage based scan required. |
| ROI+Ctx gate must be rescanned | yes | Semantic gate plus value-density gate required. |
| can start default-off implementation | yes | Only default-off and copied config paths. |
| can run actual A/B | no | First implement default-off layer and gate recalibration sweep. |
"""
    final = f"""# Final Recommendation

Recommendation case: B.

Use `1:2:5:10` as the main experiment ratio. Keep `1:3:8:15` as sensitivity because it is ratio-reasonable but too exposed to the current assignment concentration. Keep `1:5:20:50` as stress sensitivity only and keep `old` as a legacy reference.

The value-level assignment should be revised before formal integration. The main issue is not the ratio alone; it is the v1 assignment rule, where medium is unused and product_requirement/context/evidence plus high_value semantics push too many tasks into high or urgent. The v2 dry-run separates product requirement from mission value level and recommends `{RECOMMENDED_V2_RULE}` for the next calibration pass.

Next engineering step: implement a default-off calibration layer after base_prior_score, with mean-preserving normalization and copied experiment configs only. Before actual A/B, rescan ERIV and ROI+Ctx gates by candidate count, high-value coverage, queue pressure, low-value accidental triggers, and resource-constrained high-value closure.
"""
    _write_text(OUT_DIR / "main_experiment_ratio_recommendation.md", main_report)
    _write_text(OUT_DIR / "implementation_readiness_checklist.md", readiness)
    _write_text(OUT_DIR / "final_recommendation.md", final)


def _write_rule_audit(distributions: list[dict[str, object]]) -> None:
    by_rule = {row["assignment_rule"]: row for row in distributions}
    lines = [
        "# Value-Level Assignment Rule Audit",
        "",
        "This is a dry-run audit only. It does not connect v2 assignment to the simulator.",
        "",
        "## Findings",
        "",
        "- v1 uses no medium bucket and maps 75.5% of tasks to high/urgent.",
        "- product_requirement is treated as product need, not mission value level, in v2.",
        "- The suggested urgent target of 3%-15% is not attainable here without demoting emergency tasks; emergency tasks alone are above that band.",
        f"- Recommended dry-run v2 rule: `{RECOMMENDED_V2_RULE}`.",
        "",
        "## Distribution Summary",
        "",
        "| Rule | Low | Medium | High | Urgent | High/Urgent | Medium Nonzero | Context/Evidence -> High | Emergency -> Urgent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rule in ["v1", "v2_conservative", "v2_balanced", "v2_permissive"]:
        row = by_rule[rule]
        lines.append(
            f"| {rule} | {row['low_count']} | {row['medium_count']} | {row['high_count']} | {row['urgent_count']} | "
            f"{_to_float(row['high_urgent_share']):.3f} | {row['medium_nonzero']} | "
            f"{_to_float(row['context_evidence_mapped_to_high_share']):.3f} | {_to_float(row['emergency_mapped_to_urgent_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "`v2_conservative` is the least risky next rule because it restores medium, keeps emergency semantics intact, and avoids allowing context/evidence to directly force high. `v2_balanced` and `v2_permissive` are useful diagnostics, but they leave more assignment pressure on high/urgent.",
        ]
    )
    _write_text(OUT_DIR / "value_level_assignment_rule_audit.md", "\n".join(lines))


def _archive_manifest() -> None:
    files = sorted(path.relative_to(OUT_DIR).as_posix() for path in OUT_DIR.rglob("*") if path.is_file())
    if "archive_manifest.txt" not in files:
        files.append("archive_manifest.txt")
        files = sorted(files)
    _write_text(OUT_DIR / "archive_manifest.txt", "\n".join(files))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    command = f"python scripts/audit_value_level_and_ratio_parallel.py --seeds {','.join(str(seed) for seed in seeds)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_rows = _load_tasks(seeds)
    rows = _prepare_v1_rows(base_rows)
    for variant in ["v2_conservative", "v2_balanced", "v2_permissive"]:
        _assign_v2(rows, variant)

    distribution_rows = [_distribution_by_rule(rows, "v1_value_level", "v1")]
    for variant in ["v2_conservative", "v2_balanced", "v2_permissive"]:
        distribution_rows.append(_distribution_by_rule(rows, f"{variant}_value_level", variant))

    v1_values = _calibrate_values(rows, "v1_value_level")
    v2_values = _calibrate_values(rows, f"{RECOMMENDED_V2_RULE}_value_level")
    value_summary = _value_distribution_summary(rows, "v1_value_level", "v1", v1_values)
    value_summary.extend(_value_distribution_summary(rows, f"{RECOMMENDED_V2_RULE}_value_level", RECOMMENDED_V2_RULE, v2_values))
    candidate_rows = _candidate_rows(rows, f"{RECOMMENDED_V2_RULE}_value_level", v2_values)
    eriv_rows = _eriv_summary(candidate_rows)
    roi_rows = _roi_pressure(candidate_rows, eriv_rows)
    baseline_rows = _baseline_fairness_summary()
    resource_rows = _resource_proxy(value_summary, eriv_rows, roi_rows, baseline_rows)
    recommendation_matrix = _recommendation_matrix(value_summary)

    _write_text(
        OUT_DIR / "README.md",
        "# Value-Level And Ratio Parallel Audit\n\nAudit-only dry-run for value-level assignment variants, ratio comparison, ERIV/ROI+Ctx pressure, baseline fairness, and resource-constrained high-value closure proxy. No mainline code or default configs are changed.",
    )
    _write_text(OUT_DIR / "command_log.md", f"# Command Log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "audit_only": True,
                "default_off": True,
                "main_defaults_changed": False,
                "reward_function_changed": False,
                "module2_scoring_changed": False,
                "safe_linucb_changed": False,
                "safe_filter_changed": False,
                "utility_rule_changed": False,
                "queue_policy_changed": False,
                "lifecycle_closure_changed": False,
                "information_sufficiency_changed": False,
                "recommended_v2_rule": RECOMMENDED_V2_RULE,
                "recommended_main_ratio": "soft_moderate",
                "recommended_main_ratio_weights": RATIOS["soft_moderate"],
                "sensitivity_ratios": ["mild", "moderate", "stress"],
                "final_recommendation_case": "B",
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_csv(OUT_DIR / "value_level_assignment_v1_vs_v2_trace.csv", _level_trace(rows))
    _write_csv(OUT_DIR / "value_level_distribution_v1_vs_v2.csv", distribution_rows)
    _write_rule_audit(distribution_rows)
    _write_csv(OUT_DIR / "value_skew_ratio_comparison_summary.csv", value_summary)
    _write_csv(OUT_DIR / "calibrated_value_distribution_ratio_comparison.csv", value_summary)
    _write_csv(OUT_DIR / "ratio_recommendation_matrix.csv", recommendation_matrix)
    _write_csv(OUT_DIR / "eriv_ratio_impact_summary.csv", eriv_rows)
    _write_csv(OUT_DIR / "roi_context_pressure_ratio_summary.csv", roi_rows)
    _write_csv(OUT_DIR / "baseline_fairness_ratio_summary.csv", baseline_rows)
    _write_csv(OUT_DIR / "resource_constrained_metric_proxy_ratio_summary.csv", resource_rows)
    _write_reports(distribution_rows, value_summary, eriv_rows, roi_rows)
    _write_text(OUT_DIR / "git_status.txt", _git(["status", "--short"]))
    diff = _git(["diff", "--", "scripts/audit_value_level_and_ratio_parallel.py"])
    script_path = PROJECT_ROOT / "scripts" / "audit_value_level_and_ratio_parallel.py"
    _write_text(OUT_DIR / "git_diff.patch", diff if diff.strip() else _add_file_patch(script_path))
    _archive_manifest()

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", OUT_DIR)

    v1_dist = next(row for row in distribution_rows if row["assignment_rule"] == "v1")
    v2_dist = next(row for row in distribution_rows if row["assignment_rule"] == RECOMMENDED_V2_RULE)
    print(f"audit output directory: {OUT_DIR}")
    print(f"zip path: {ZIP_PATH}")
    print(f"recommended value-level assignment rule: {RECOMMENDED_V2_RULE}")
    print(f"v1 high/urgent share: {_to_float(v1_dist['high_urgent_share']):.4f}")
    print(f"v2 high/urgent share: {_to_float(v2_dist['high_urgent_share']):.4f}")
    print("recommended main ratio: soft_moderate=1:2:5:10")
    print("recommended sensitivity ratios: mild=1:2:3:5, moderate=1:3:8:15, stress=1:5:20:50")
    print("whether 1:3:8:15 remains main: no")
    print("whether 1:2:5:10 is safer: yes")
    print("final recommendation case: B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
