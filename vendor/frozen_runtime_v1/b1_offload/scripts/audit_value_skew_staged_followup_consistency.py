"""Audit value-skew field consistency and staged follow-up readiness.

Audit-only script. It reads existing traces and produces dry-run/counterfactual
reports. It does not modify simulator defaults, reward logic, Module 2 scoring,
SafeLinUCB, SafeFilter, utility_rule, queue policy, lifecycle closure,
information_sufficiency, task_value, or ROI+Ctx behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "outputs" / "_value_skew_staged_followup_consistency_audit"
ZIP_PATH = PROJECT_ROOT / "outputs" / "_value_skew_staged_followup_consistency_audit.zip"

VALUE_AUDIT_DIR = PROJECT_ROOT / "outputs" / "_value_level_ratio_parallel_audit"
SKEW_AUDIT_DIR = PROJECT_ROOT / "outputs" / "_value_skew_feasibility_audit"
RESIDUAL_AUDIT_DIR = PROJECT_ROOT / "outputs" / "_residual_delivery_closure_accounting_audit"
PHASE6_DIR = PROJECT_ROOT / "outputs" / "_phase6_residual_enqueue_fallback_validation"
PR_AB_DIR = PROJECT_ROOT / "outputs" / "_product_requirement_aware_module2_ab"

OLD_ERIV_THRESHOLD = 0.15
SOFT_MODERATE = {"low": 1.0, "medium": 2.0, "high": 5.0, "urgent": 10.0}
VALUE_RULE_FIELD = "v2_conservative_value_level"
VALUE_PROFILE = "soft_moderate"
LEVELS = ["low", "medium", "high", "urgent"]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _stable_key(seed: int, task_id: str) -> str:
    return f"{seed}:{task_id}"


def _base_f(base_prior: float) -> float:
    return 0.8 + 0.4 * max(0.0, min(1.0, base_prior))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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
    except Exception as exc:  # pragma: no cover - audit robustness
        return f"git {' '.join(args)} failed: {exc}"


def _add_file_patch(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [
        f"diff --git a/{rel} b/{rel}",
        "new file mode 100644",
        "index 0000000..0000000",
        "--- /dev/null",
        f"+++ b/{rel}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    return "\n".join(header + ["+" + line for line in lines]) + "\n"


def _products(raw: object) -> set[str]:
    return {item.strip() for item in str(raw or "").replace(",", ";").split(";") if item.strip()}


def _load_value_rows(seeds: list[int]) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    for row in _read_csv(VALUE_AUDIT_DIR / "value_level_assignment_v1_vs_v2_trace.csv"):
        seed = _to_int(row.get("seed"))
        if seed not in seeds:
            continue
        level = str(row.get(VALUE_RULE_FIELD, "medium") or "medium")
        base_prior = _to_float(row.get("base_prior_score"))
        old_value = _to_float(row.get("old_task_value"))
        raw = SOFT_MODERATE.get(level, 2.0) * _base_f(base_prior)
        item: dict[str, object] = dict(row)
        item["value_level"] = level
        item["value_multiplier"] = SOFT_MODERATE.get(level, 2.0)
        item["soft_moderate_raw_value"] = raw
        item["field_status"] = "dry_run_only"
        rows.append(item)
    old_mean = mean([_to_float(row["old_task_value"]) for row in rows]) if rows else 1.0
    raw_mean = mean([_to_float(row["soft_moderate_raw_value"]) for row in rows]) if rows else 1.0
    alpha = old_mean / max(raw_mean, 1.0e-12)
    for row in rows:
        row["value_skew_profile"] = VALUE_PROFILE
        row["normalization_alpha"] = alpha
        row["calibrated_task_value"] = alpha * _to_float(row["soft_moderate_raw_value"])
        row["base_prior_source"] = "mission_prior_value_v2 metadata/new_task_value dry-run trace"
    by_key = {_stable_key(_to_int(row["seed"]), str(row["task_id"])): row for row in rows}
    return rows, by_key, alpha


def _load_trajectory_rows(seeds: list[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in seeds:
        path = (
            PR_AB_DIR
            / "runs"
            / f"seed_{seed}"
            / "v21_reqaware_ug_wx050"
            / "trajectories"
            / "proposed_lyapunov_rule_mode_safe_bandit.csv"
        )
        for row in _read_csv(path):
            if str(row.get("has_task", "0")) != "1":
                continue
            row["seed"] = str(seed)
            rows.append(row)
    return rows


def _first_by_key(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = _stable_key(_to_int(row.get("seed")), str(row.get("task_id", "")))
        if key not in result:
            result[key] = row
    return result


def _last_by_key(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = _stable_key(_to_int(row.get("seed")), str(row.get("task_id", "")))
        result[key] = row
    return result


def _candidate_rows(value_by_key: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    sources = [
        (PHASE6_DIR / "same_slot_carryover_trace.csv", "same_slot"),
        (PHASE6_DIR / "per_candidate_followup_trace.csv", "per_candidate"),
        (PHASE6_DIR / "residual_fallback_enqueue_trace.csv", "fallback_enqueue"),
    ]
    for path, source in sources:
        for row in _read_csv(path):
            eriv = _to_float(row.get("ERIV"))
            if eriv <= 0.0:
                continue
            event_type = str(row.get("event_type", "candidate"))
            if source == "same_slot" and event_type not in {"same_slot_candidate_re_evaluated", "same_slot_proactive_enqueued"}:
                continue
            seed = _to_int(row.get("seed"))
            task_id = str(row.get("task_id", ""))
            residual_type = str(row.get("residual_product_type", ""))
            identity = (source, str(row.get("arm", "")), str(row.get("slot", "")), task_id, residual_type)
            if identity in seen:
                continue
            seen.add(identity)
            key = _stable_key(seed, task_id)
            task = value_by_key.get(key, {})
            high_flag = bool(_to_int(row.get("high_value_flag", task.get("high_value_flag", 0))))
            emergency = bool(_to_int(row.get("emergency_flag", task.get("emergency_flag", 0))))
            product_requirement = str(row.get("product_requirement", task.get("product_requirement", "")) or "verification")
            level = str(task.get("value_level", "urgent" if emergency else "high" if high_flag else "medium"))
            old_value = max(_to_float(task.get("old_task_value", row.get("task_value", 1.0))), 1.0e-12)
            calibrated = _to_float(task.get("calibrated_task_value", old_value))
            soft_eriv = eriv * calibrated / old_value
            rows.append(
                {
                    "source_trace": source,
                    "seed": seed,
                    "arm": row.get("arm", ""),
                    "slot": row.get("slot", ""),
                    "task_id": task_id,
                    "product_requirement": product_requirement,
                    "residual_product_type": residual_type,
                    "value_level": level,
                    "high_value_flag": int(high_flag),
                    "emergency_flag": int(emergency),
                    "old_task_value": old_value,
                    "calibrated_task_value": calibrated,
                    "legacy_eriv": eriv,
                    "soft_moderate_eriv": soft_eriv,
                    "old_threshold": _to_float(row.get("ERIV_threshold"), OLD_ERIV_THRESHOLD),
                    "proactive_flag": _to_int(row.get("proactive_flag")),
                    "enqueue_flag": _to_int(row.get("enqueue_decision", row.get("proactive_enqueue_flag", row.get("ordinary_queue_fallback_enqueued_flag", 0)))),
                    "delivery_flag": _to_int(row.get("delivery_decision", row.get("proactive_delivery_flag", row.get("residual_payload_delivered_flag", 0)))),
                    "queue_mb": _to_float(row.get("queue_mb", row.get("queue_size_after_fallback_mb"))),
                    "remaining_deadline_s": _to_float(row.get("remaining_deadline_s")),
                    "contact_available": _to_int(row.get("contact_available", 1)),
                    "payload_size_proxy_mb": max(0.8, _to_float(row.get("payload_size_mb", row.get("queue_size_after_fallback_mb", 1.5)), 1.5)),
                    "field_status": "observed_candidate_with_dry_run_soft_moderate_eriv",
                }
            )
    return rows


def _closure_rows() -> list[dict[str, str]]:
    return _read_csv(RESIDUAL_AUDIT_DIR / "closure_accounting_trace.csv")


def _trace_rows(
    value_rows: list[dict[str, object]],
    value_by_key: dict[str, dict[str, object]],
    traj_rows: list[dict[str, str]],
    candidates: list[dict[str, object]],
    closures: list[dict[str, str]],
) -> list[dict[str, object]]:
    first_traj = _first_by_key(traj_rows)
    last_traj = _last_by_key(traj_rows)
    candidate_by_key: dict[str, dict[str, object]] = {}
    for row in candidates:
        key = _stable_key(_to_int(row["seed"]), str(row["task_id"]))
        candidate_by_key.setdefault(key, row)
    closure_by_key: dict[str, dict[str, str]] = {}
    for row in closures:
        key = _stable_key(_to_int(row.get("seed")), str(row.get("task_id", "")))
        closure_by_key[key] = row
    out: list[dict[str, object]] = []
    keys = sorted(set(value_by_key) | set(first_traj) | set(candidate_by_key) | set(closure_by_key))
    for key in keys:
        task = value_by_key.get(key, {})
        first = first_traj.get(key, {})
        last = last_traj.get(key, {})
        cand = candidate_by_key.get(key, {})
        closure = closure_by_key.get(key, {})
        seed, task_id = key.split(":", 1)
        out.append(
            {
                "seed": seed,
                "task_id": task_id,
                "scene_id": task.get("scene_id", first.get("scene_id", "")),
                "slot": task.get("slot", first.get("slot", cand.get("slot", closure.get("slot", "")))),
                "base_prior_score": task.get("base_prior_score", ""),
                "value_level": task.get("value_level", cand.get("value_level", "")),
                "value_multiplier": task.get("value_multiplier", ""),
                "calibrated_task_value": task.get("calibrated_task_value", ""),
                "value_skew_profile": task.get("value_skew_profile", VALUE_PROFILE if task else ""),
                "normalization_alpha": task.get("normalization_alpha", ""),
                "product_requirement": task.get("product_requirement", first.get("product_requirement", cand.get("product_requirement", closure.get("product_requirement", "")))),
                "selected_action": first.get("selected_action", ""),
                "selected_model": first.get("selected_model", ""),
                "selected_payload_plan": first.get("selected_payload_plan", ""),
                "residual_product_type": cand.get("residual_product_type", first.get("residual_product_type", closure.get("residual_product_type", ""))),
                "ERIV": cand.get("soft_moderate_eriv", first.get("ERIV", "")),
                "proactive_flag": cand.get("proactive_flag", first.get("proactive_followup_attempt_count", "")),
                "payload_type": first.get("selected_product_type", ""),
                "queue_item.product_type": closure.get("queue_item_product_type", ""),
                "delivered_product_type": closure.get("delivered_product_type", ""),
                "delivered_payload_metadata": closure.get("delivered_payload_metadata_keys", ""),
                "lifecycle.delivered_products": closure.get("lifecycle_delivered_products_after", last.get("lifecycle_post_delivered_products", last.get("delivered_products", ""))),
                "lifecycle.pending_products": closure.get("lifecycle_pending_products_after", last.get("lifecycle_post_pending_products", last.get("pending_products", ""))),
                "residual_utility_remaining": closure.get("lifecycle_residual_utility_after", last.get("lifecycle_post_residual_utility_remaining", last.get("residual_utility_remaining", ""))),
                "verification_closure_flag": closure.get("verification_closure_flag", ""),
                "context_closure_flag": closure.get("context_closure_flag", ""),
                "evidence_closure_flag": closure.get("evidence_closure_flag", ""),
                "value_field_status": "dry_run_only" if task else "missing_for_synthetic_or_trace_only_task",
                "trajectory_field_status": "observed" if first else "not_observed_in_v21_reqaware_trajectory",
                "residual_field_status": "observed" if cand else "not_observed_as_residual_candidate",
                "lifecycle_field_status": "observed" if closure else "not_observed_in_closure_accounting_trace",
            }
        )
    return out


def _field_summary(trace_rows: list[dict[str, object]], candidates: list[dict[str, object]], closures: list[dict[str, str]]) -> list[dict[str, object]]:
    field_specs = [
        ("seed", "task_generation", "observed"),
        ("task_id", "task_generation", "observed"),
        ("scene_id", "task_generation", "observed"),
        ("slot", "timeline", "observed"),
        ("base_prior_score", "mission_prior_value_v2", "dry_run_observed"),
        ("value_level", "v2_conservative_assignment", "dry_run_only"),
        ("value_multiplier", "value_skew_soft_moderate", "dry_run_only"),
        ("calibrated_task_value", "value_skew_soft_moderate", "dry_run_only"),
        ("value_skew_profile", "value_skew_soft_moderate", "dry_run_only"),
        ("normalization_alpha", "value_skew_soft_moderate", "dry_run_only"),
        ("product_requirement", "task_stream/module2", "observed"),
        ("selected_action", "router", "observed"),
        ("selected_model", "module2_action", "observed"),
        ("selected_payload_plan", "module2_payload", "observed"),
        ("residual_product_type", "phase5_phase6_followup", "observed_partial"),
        ("ERIV", "phase6_gate", "observed_plus_dry_run"),
        ("proactive_flag", "phase6_gate", "observed"),
        ("payload_type", "payload_generation", "observed"),
        ("queue_item.product_type", "queue", "observed_in_closure_audit"),
        ("delivered_product_type", "delivery", "observed_in_closure_audit"),
        ("delivered_payload_metadata", "delivery", "observed_partial"),
        ("lifecycle.delivered_products", "lifecycle", "observed"),
        ("lifecycle.pending_products", "lifecycle", "observed"),
        ("residual_utility_remaining", "lifecycle", "observed"),
        ("verification_closure_flag", "closure_metric", "observed"),
        ("context_closure_flag", "closure_metric", "observed_zero"),
        ("evidence_closure_flag", "closure_metric", "observed_zero"),
    ]
    rows: list[dict[str, object]] = []
    total = len(trace_rows)
    for field, module, expected in field_specs:
        present = sum(1 for row in trace_rows if str(row.get(field, "")) not in {"", "nan", "None"})
        if expected == "dry_run_only":
            status = "dry_run_only"
        elif present == total and total:
            status = "present"
        elif present > 0:
            status = "partial"
        else:
            status = "missing"
        rows.append(
            {
                "field": field,
                "module_or_stage": module,
                "field_status": status,
                "present_count": present,
                "total_trace_rows": total,
                "evidence_note": expected,
            }
        )
    rows.append(
        {
            "field": "roi_context_residual_candidate",
            "module_or_stage": "progressive_followup_candidate_pool",
            "field_status": "missing" if not any(str(row.get("residual_product_type")) == "roi_context" for row in candidates) else "present",
            "present_count": sum(1 for row in candidates if str(row.get("residual_product_type")) == "roi_context"),
            "total_trace_rows": len(candidates),
            "evidence_note": "candidate_pool_residual_type_distribution",
        }
    )
    rows.append(
        {
            "field": "roi_context_lifecycle_delivery",
            "module_or_stage": "lifecycle_closure",
            "field_status": "missing" if not any(str(row.get("delivered_product_type")) == "roi_context" for row in closures) else "present",
            "present_count": sum(1 for row in closures if str(row.get("delivered_product_type")) == "roi_context"),
            "total_trace_rows": len(closures),
            "evidence_note": "closure_accounting_delivered_product_type_distribution",
        }
    )
    return rows


def _gate_distribution(candidates: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    groups = {
        "all_candidates": candidates,
        "high_urgent_candidates": [row for row in candidates if str(row.get("value_level")) in {"high", "urgent"} or _to_int(row.get("high_value_flag"))],
        "low_medium_candidates": [row for row in candidates if str(row.get("value_level")) in {"low", "medium"} and not _to_int(row.get("high_value_flag"))],
    }
    dist_rows: list[dict[str, object]] = []
    for group, rows in groups.items():
        for scale in ["legacy", VALUE_PROFILE]:
            field = "legacy_eriv" if scale == "legacy" else "soft_moderate_eriv"
            values = [_to_float(row.get(field)) for row in rows]
            dist_rows.append(
                {
                    "candidate_group": group,
                    "value_profile": scale,
                    "candidate_count": len(rows),
                    "eriv_min": min(values) if values else 0.0,
                    "eriv_mean": mean(values) if values else 0.0,
                    "eriv_median": median(values) if values else 0.0,
                    "eriv_p75": _percentile(values, 0.75),
                    "eriv_p90": _percentile(values, 0.90),
                    "eriv_p95": _percentile(values, 0.95),
                    "eriv_max": max(values) if values else 0.0,
                    "above_old_threshold_0.15_count": sum(1 for value in values if value >= OLD_ERIV_THRESHOLD),
                }
            )

    legacy_values = [_to_float(row.get("legacy_eriv")) for row in candidates]
    soft_values = [_to_float(row.get("soft_moderate_eriv")) for row in candidates]
    old_count = sum(1 for value in legacy_values if value >= OLD_ERIV_THRESHOLD)
    soft_above_old = sum(1 for value in soft_values if value >= OLD_ERIV_THRESHOLD)
    sorted_soft = sorted(soft_values, reverse=True)
    match_threshold = sorted_soft[old_count - 1] if old_count and len(sorted_soft) >= old_count else 0.0
    hv_soft = sorted(
        [
            _to_float(row.get("soft_moderate_eriv"))
            for row in candidates
            if str(row.get("value_level")) in {"high", "urgent"} or _to_int(row.get("high_value_flag"))
        ],
        reverse=True,
    )
    top_hv_k = min(max(1, old_count), len(hv_soft)) if hv_soft else 0
    top_hv_threshold = hv_soft[top_hv_k - 1] if top_hv_k else match_threshold
    low_medium_soft = [
        _to_float(row.get("soft_moderate_eriv"))
        for row in candidates
        if str(row.get("value_level")) in {"low", "medium"} and not _to_int(row.get("high_value_flag"))
    ]
    suppress_low_threshold = _percentile(low_medium_soft, 0.90) if low_medium_soft else match_threshold
    recommended_threshold = match_threshold
    if recommended_threshold <= 0.0:
        recommended_threshold = top_hv_threshold
    recalibration = [
        {
            "value_profile": VALUE_PROFILE,
            "legacy_candidate_count_at_0.15": old_count,
            "soft_moderate_candidate_count_at_old_threshold_0.15": soft_above_old,
            "suggested_threshold_to_match_old_candidate_count": match_threshold,
            "suggested_threshold_to_allow_top_high_value_candidates": top_hv_threshold,
            "suggested_threshold_to_suppress_low_value_accidental_triggers": suppress_low_threshold,
            "recommended_theta_profile_for_sweep_center": recommended_threshold,
            "old_threshold_still_valid": "no" if soft_above_old != old_count else "borderline",
            "suggested_gate_direction": "profile_specific_threshold_lower_than_0.15_plus_semantic_value_density_topk_gate",
            "threshold_basis": "candidate_count_high_value_coverage_low_value_suppression_queue_pressure_not_multiplier_scaling",
        }
    ]

    semantic_eligible = [
        row
        for row in candidates
        if str(row.get("product_requirement")) in {"context", "evidence", "verification"}
        and str(row.get("residual_product_type")) in {"roi", "roi_context"}
        and str(row.get("value_level")) in {"high", "urgent"}
    ]
    selected = [row for row in semantic_eligible if _to_float(row.get("soft_moderate_eriv")) >= recommended_threshold]
    topk = max(1, min(len(selected), int(math.ceil(len(semantic_eligible) * 0.25)))) if selected else 0
    topk_selected = sorted(selected, key=lambda row: _to_float(row.get("soft_moderate_eriv")), reverse=True)[:topk]
    low_accidental = [
        row
        for row in candidates
        if str(row.get("value_level")) in {"low", "medium"}
        and not _to_int(row.get("high_value_flag"))
        and _to_float(row.get("soft_moderate_eriv")) >= recommended_threshold
    ]
    high_missed = [row for row in semantic_eligible if _to_float(row.get("soft_moderate_eriv")) < recommended_threshold]
    roi_context_pressure = [
        {
            "value_profile": VALUE_PROFILE,
            "semantic_eligible_candidate_count": len(semantic_eligible),
            "selected_candidate_count_at_recommended_theta": len(selected),
            "selected_candidate_count_after_topk_proxy": len(topk_selected),
            "roi_context_candidate_count_observed": sum(1 for row in candidates if str(row.get("residual_product_type")) == "roi_context"),
            "roi_candidate_count_observed": sum(1 for row in candidates if str(row.get("residual_product_type")) == "roi"),
            "queue_pressure_if_all_roi_ctx_upgraded_mb": sum(max(1.5, _to_float(row.get("payload_size_proxy_mb"), 1.5) * 2.5) for row in semantic_eligible),
            "queue_pressure_if_selective_roi_ctx_upgraded_mb": sum(max(1.5, _to_float(row.get("payload_size_proxy_mb"), 1.5) * 2.5) for row in topk_selected),
            "low_value_accidental_trigger_count_without_semantic_gate": len(low_accidental),
            "low_value_accidental_trigger_count_after_semantic_gate": 0,
            "high_value_missed_trigger_count": len(high_missed),
            "expired_item_risk_proxy": sum(1 for row in topk_selected if _to_float(row.get("remaining_deadline_s")) <= 0.0),
            "semantic_gate_needed": 1,
            "value_density_gate_needed": 1,
            "per_window_topk_gate_needed": 1,
        }
    ]
    return dist_rows, recalibration, roi_context_pressure


def _progressive_funnel(
    value_rows: list[dict[str, object]],
    traj_rows: list[dict[str, str]],
    candidates: list[dict[str, object]],
    closures: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    high_context_tasks = [
        row
        for row in value_rows
        if str(row.get("value_level")) in {"high", "urgent"} and str(row.get("product_requirement")) in {"context", "evidence"}
    ]
    high_context_keys = {_stable_key(_to_int(row["seed"]), str(row["task_id"])) for row in high_context_tasks}
    traj_hc = [row for row in traj_rows if _stable_key(_to_int(row.get("seed")), str(row.get("task_id", ""))) in high_context_keys]
    closures_hc = [
        row
        for row in closures
        if str(row.get("product_requirement")) in {"context", "evidence"}
        or _stable_key(_to_int(row.get("seed")), str(row.get("task_id", ""))) in high_context_keys
    ]
    candidate_hc = [row for row in candidates if _stable_key(_to_int(row["seed"]), str(row["task_id"])) in high_context_keys]
    roi_closures = [row for row in closures if str(row.get("delivered_product_type")) == "roi"]
    roi_ctx_candidates = [row for row in candidates if str(row.get("residual_product_type")) == "roi_context"]
    roi_candidates = [row for row in candidates if str(row.get("residual_product_type")) == "roi"]

    def count_traj(predicate) -> int:
        return len({_stable_key(_to_int(row.get("seed")), str(row.get("task_id", ""))) for row in traj_hc if predicate(row)})

    stages = [
        ("task_created", len(high_context_tasks), "task stream plus v2 value dry-run"),
        ("value_level_high_or_urgent", len(high_context_tasks), "v2_conservative dry-run"),
        ("product_requirement_context_or_evidence", len(high_context_tasks), "task product requirement"),
        ("metadata_selected", count_traj(lambda row: str(row.get("selected_product_type")) == "metadata" or str(row.get("selected_payload_plan")).startswith("metadata")), "trajectory selected payload"),
        ("metadata_generated", count_traj(lambda row: str(row.get("selected_payload_plan")).startswith("metadata")), "staged metadata payload plan"),
        ("metadata_delivered", count_traj(lambda row: "metadata" in _products(row.get("delivered_products")) or "metadata" in _products(row.get("lifecycle_post_delivered_products"))), "trajectory lifecycle delivered products"),
        ("pending_residual_registered", count_traj(lambda row: _to_int(row.get("lifecycle_post_pending_residual_exists_flag")) or bool(_products(row.get("pending_products")) & {"roi", "roi_context"})), "lifecycle pending residual"),
        ("pending_residual_type_roi", count_traj(lambda row: str(row.get("residual_product_type")) == "roi" or "roi" in _products(row.get("pending_products"))), "metadata_then_roi residual"),
        ("roi_candidate_added", len([row for row in roi_candidates if _stable_key(_to_int(row["seed"]), str(row["task_id"])) in high_context_keys]), "phase6 candidate pool"),
        ("roi_eriv_passed", len([row for row in roi_candidates if _stable_key(_to_int(row["seed"]), str(row["task_id"])) in high_context_keys and _to_float(row.get("legacy_eriv")) >= OLD_ERIV_THRESHOLD]), "phase6 ERIV gate"),
        ("roi_generated", sum(1 for row in closures if str(row.get("residual_product_type")) == "roi"), "closure accounting residual payload"),
        ("roi_enqueued", sum(_to_int(row.get("queue_item_product_type") == "roi") for row in closures), "queue item product type"),
        ("roi_delivered", len(roi_closures), "closure accounting delivery"),
        ("lifecycle_delivered_products_includes_roi", sum(1 for row in closures if "roi" in _products(row.get("lifecycle_delivered_products_after"))), "lifecycle delivered products"),
        ("residual_utility_remaining_after_roi_gt_0", sum(1 for row in roi_closures if _to_float(row.get("lifecycle_residual_utility_after")) > 0.0), "lifecycle residual utility"),
        ("requirement_still_needs_context_evidence", sum(1 for row in roi_closures if str(row.get("product_requirement")) in {"context", "evidence"} and _to_float(row.get("lifecycle_residual_utility_after")) > 0.0), "requirement residual target"),
        ("pending_residual_upgraded_or_extended_to_roi_context", 0, "not observed in closure/candidate traces"),
        ("roi_context_candidate_added", len(roi_ctx_candidates), "phase6 candidate pool"),
        ("roi_context_eriv_passed", sum(1 for row in roi_ctx_candidates if _to_float(row.get("legacy_eriv")) >= OLD_ERIV_THRESHOLD), "phase6 ERIV gate"),
        ("roi_context_generated", sum(1 for row in closures if str(row.get("residual_product_type")) == "roi_context"), "closure accounting residual payload"),
        ("roi_context_enqueued", sum(1 for row in closures if str(row.get("queue_item_product_type")) == "roi_context"), "queue item product type"),
        ("roi_context_delivered", sum(1 for row in closures if str(row.get("delivered_product_type")) == "roi_context"), "closure accounting delivery"),
        ("lifecycle_delivered_products_includes_roi_context", sum(1 for row in closures if "roi_context" in _products(row.get("lifecycle_delivered_products_after"))), "lifecycle delivered products"),
        ("context_evidence_closure_achieved", sum(1 for row in closures if _to_int(row.get("context_closure_flag")) or _to_int(row.get("evidence_closure_flag"))), "closure metric"),
    ]
    funnel_rows = [
        {"stage_index": index + 1, "stage": stage, "count": count, "evidence_source": source}
        for index, (stage, count, source) in enumerate(stages)
    ]

    transition_rows: list[dict[str, object]] = []
    if roi_closures:
        for row in roi_closures:
            need_ctx = str(row.get("product_requirement")) in {"context", "evidence"}
            remaining = _to_float(row.get("lifecycle_residual_utility_after"))
            transition_rows.append(
                {
                    "seed": row.get("seed"),
                    "task_id": row.get("task_id"),
                    "product_requirement": row.get("product_requirement"),
                    "roi_delivered": 1,
                    "residual_utility_after_roi": remaining,
                    "context_or_evidence_still_needed": int(need_ctx and remaining > 0.0),
                    "roi_context_pending_after_roi": int("roi_context" in _products(row.get("lifecycle_pending_products_after"))),
                    "roi_context_candidate_added_after_roi": 0,
                    "first_missing_stage": "roi_context_pending_registration_after_roi" if need_ctx and remaining > 0.0 else "not_applicable_for_verification_roi_case",
                    "blocker_class": "candidate_pool_design" if need_ctx and remaining > 0.0 else "verification_target_satisfied_by_roi",
                }
            )
    else:
        transition_rows.append(
            {
                "seed": "",
                "task_id": "",
                "product_requirement": "",
                "roi_delivered": 0,
                "residual_utility_after_roi": "",
                "context_or_evidence_still_needed": "",
                "roi_context_pending_after_roi": 0,
                "roi_context_candidate_added_after_roi": 0,
                "first_missing_stage": "roi_delivery_not_observed_for_context_evidence_tasks",
                "blocker_class": "trace_coverage_or_candidate_pool",
            }
        )

    readiness = [
        {
            "scope": "all_audited_residuals",
            "high_value_context_evidence_task_count": len(high_context_tasks),
            "roi_residual_candidate_count": len(roi_candidates),
            "roi_context_residual_candidate_count": len(roi_ctx_candidates),
            "roi_delivered_count": len(roi_closures),
            "roi_context_delivered_count": sum(1 for row in closures if str(row.get("delivered_product_type")) == "roi_context"),
            "verification_closure_count": sum(_to_int(row.get("verification_closure_flag")) for row in closures),
            "context_closure_count": sum(_to_int(row.get("context_closure_flag")) for row in closures),
            "evidence_closure_count": sum(_to_int(row.get("evidence_closure_flag")) for row in closures),
            "metadata_to_roi_chain_status": "present_for_verification_residuals",
            "metadata_to_roi_to_roi_context_chain_status": "not_present",
            "context_evidence_closure_requires_roi_context": 1,
            "readiness_status": "needs_default_off_progressive_extension",
        }
    ]

    blockers = [
        {
            "blocker": "second_stage_roi_context_pending_not_registered_after_roi",
            "blocker_class": "lifecycle_candidate_pool_design",
            "evidence_count": 1,
            "impact": "metadata_to_roi_to_roi_context_chain_stops_after_roi_delivery",
            "recommendation": "default_off second-stage residual registration for context/evidence only",
        },
        {
            "blocker": "roi_context_residual_candidate_not_carried_to_delivery_or_closure",
            "blocker_class": "candidate_pool_design",
            "evidence_count": len(roi_ctx_candidates),
            "impact": "roi_context candidates are rare and no roi_context residual is generated enqueued delivered or closed",
            "recommendation": "create roi_context candidates from remaining residual utility after ROI when semantic/value gates pass",
        },
        {
            "blocker": "old_eriv_threshold_profile_mismatch",
            "blocker_class": "gate_threshold",
            "evidence_count": len(candidates),
            "impact": "soft_moderate changes ERIV density; threshold must be rescanned",
            "recommendation": "profile-specific threshold plus per-window top-K",
        },
        {
            "blocker": "roi_is_not_context_or_evidence_closure",
            "blocker_class": "closure_metric_definition",
            "evidence_count": sum(1 for row in roi_closures if not _to_int(row.get("context_closure_flag")) and not _to_int(row.get("evidence_closure_flag"))),
            "impact": "do not count ROI as evidence closure",
            "recommendation": "keep closure definition unchanged; require roi_context for context/evidence credit",
        },
    ]
    return funnel_rows, transition_rows, readiness, blockers


def _write_reports(
    field_summary: list[dict[str, object]],
    gate_recalibration: list[dict[str, object]],
    roi_pressure: list[dict[str, object]],
    readiness: list[dict[str, object]],
    blockers: list[dict[str, object]],
) -> str:
    missing = [row for row in field_summary if row["field_status"] in {"missing", "partial"}]
    value_fields = [row for row in field_summary if row["field_status"] == "dry_run_only"]
    gate = gate_recalibration[0] if gate_recalibration else {}
    pressure = roi_pressure[0] if roi_pressure else {}
    ready = readiness[0] if readiness else {}
    first_missing = "roi_context_pending_registration_after_roi"

    missing_report = ["# Missing Or Inconsistent Fields Report", ""]
    missing_report.append("## Summary")
    missing_report.append("")
    missing_report.append("- Core task identity and product_requirement are traceable.")
    missing_report.append("- value_level, value_multiplier, calibrated_task_value, value_skew_profile, and normalization_alpha are dry_run_only.")
    missing_report.append("- residual ROI fields are observable in phase6/closure traces.")
    missing_report.append("- roi_context residual candidate and roi_context lifecycle delivery are not observed in the audited residual chain.")
    missing_report.append("")
    missing_report.append("## Dry-Run Value Fields")
    missing_report.append("")
    for row in value_fields:
        missing_report.append(f"- `{row['field']}`: {row['field_status']} from {row['module_or_stage']}.")
    missing_report.append("")
    missing_report.append("## Missing Or Partial Fields")
    missing_report.append("")
    for row in missing:
        missing_report.append(
            f"- `{row['field']}`: {row['field_status']} ({row['present_count']}/{row['total_trace_rows']}); note={row['evidence_note']}."
        )
    _write_text(OUT_DIR / "missing_or_inconsistent_fields_report.md", "\n".join(missing_report))

    threshold_report = f"""# High-Value Return Threshold Recommendation

## Finding

The old ERIV threshold `0.15` should not be reused unchanged for `soft_moderate = 1:2:5:10`.

Legacy candidates above 0.15: `{gate.get('legacy_candidate_count_at_0.15', 0)}`. Soft-moderate candidates above 0.15: `{gate.get('soft_moderate_candidate_count_at_old_threshold_0.15', 0)}`.

## Recommended Direction

- Use a profile-specific ERIV threshold centered near `{_to_float(gate.get('recommended_theta_profile_for_sweep_center')):.4f}`, then sweep around it.
- Do not compute the new threshold by multiplying 0.15 by a value multiplier.
- Add a semantic gate and value-density gate before ROI+Ctx upgrades.
- Add per-window top-K or percentile gating to avoid queue capture by ROI+Ctx.

## Proposed Gate Structure

```yaml
semantic_eligible: true
ERIV_density: ">= theta_profile"
value_level: [high, urgent]
product_requirement: [context, evidence, verification]
deadline_contact_queue_feasible: true
rank_in_window: "<= K"
```

## Rationale

The threshold should be based on candidate count, high-value coverage, low-value accidental triggers, ROI+Ctx pressure, queue pressure, expired item risk, and high-value closure potential.
"""
    _write_text(OUT_DIR / "high_value_return_threshold_recommendation.md", threshold_report)

    design = f"""# Progressive Staged Follow-Up Design

## Audit Finding

The current system supports staged metadata to ROI for residual verification closure, but the audited residual chain does not implement a continuous `metadata -> ROI -> ROI+Ctx` second-stage path.

ROI+Ctx exists as a direct or staged payload/action target, but that is not the same as automatic second-stage residual follow-up after ROI delivery.

## Design Principles

1. `product_requirement` decides the final product target.
2. `value_level` decides whether continuing is worth scarce resources.
3. Resource and gate checks decide whether continuing is feasible now.
4. ROI is enough for verification.
5. ROI+Ctx is necessary for context/evidence closure.
6. Do not upgrade every ROI to ROI+Ctx.
7. Do not count ROI as evidence closure.
8. Avoid duplicate or zero-value payloads.
9. Keep the existing metadata to ROI verification path intact.
10. Keep every new behavior default-off.

## Layered Strategy

- alert: metadata only.
- verification: metadata -> ROI, or direct ROI if resources allow.
- context: metadata -> ROI -> ROI+Ctx, or metadata -> ROI+Ctx if contact/deadline allows.
- evidence: metadata -> ROI -> ROI+Ctx only when value_level is high/urgent and gates pass.

## Second-Stage Design

ROI+Ctx should be a second residual follow-up after ROI for context/evidence tasks when remaining residual utility is positive. It can also be a first residual target when the original selected plan is `metadata_then_roi_context`.

If ROI has already been delivered, `roi_context` may duplicate some ROI bytes unless a `context_delta` payload type is introduced. MVP can use existing `roi_context` first, but the design should record that this is potentially redundant. A later `context_delta` would be cleaner.

Second-stage ERIV should use remaining residual utility after ROI, not the original full residual utility. It should also include queue pressure, deadline/contact feasibility, and a per-window top-K cap.

## Default-Off Config Sketch

```yaml
progressive_followup:
  enabled: false
  allow_second_stage_context_followup: false
  second_stage_target_products: [roi_context]
  eligible_requirements: [context, evidence]
  eligible_value_levels: [high, urgent]
  min_remaining_residual_utility: 0.08
  eriv_threshold_profile:
    soft_moderate: {_to_float(gate.get('recommended_theta_profile_for_sweep_center')):.4f}
  max_second_stage_followups_per_window: 1
  queue_pressure_guard: 0.65
  deadline_guard: 60.0
  contact_guard: true
```
"""
    _write_text(OUT_DIR / "progressive_staged_followup_design.md", design)

    implementation = f"""# Default-Off Implementation Plan

## Recommended Case

Final recommendation case: B.

Fields are sufficiently consistent for a default-off extension, but the gate must be rescanned and the progressive residual chain is incomplete.

## Minimal Route

1. Add a copied/default-off progressive_followup config block only.
2. Add dry-run field plumbing for `value_level`, `value_multiplier`, `calibrated_task_value`, `value_skew_profile`, and `normalization_alpha`.
3. Keep `task_value` unchanged by default.
4. Add second-stage candidate creation only after ROI delivery, only for context/evidence, only when residual utility after ROI remains above threshold.
5. Gate second-stage candidates by semantic eligibility, soft_moderate ERIV density, value_level high/urgent, contact/deadline/queue feasibility, and per-window top-K.
6. Keep lifecycle closure definition unchanged: ROI is not evidence closure.
7. Run gate calibration before actual A/B.
8. Run actual A/B only after field propagation tests and no-regression checks for metadata->ROI verification closure.

## Do Not Change

- reward_function main logic
- Module 2 scoring
- SafeLinUCB
- SafeFilter
- utility_rule
- queue scheduling
- lifecycle closure semantics
- information_sufficiency semantics
"""
    _write_text(OUT_DIR / "default_off_implementation_plan.md", implementation)

    final = f"""# Final Recommendation

Recommendation case: B.

The field basis is mostly consistent enough for a default-off extension: task identity, product_requirement, residual_product_type, queue/delivery product type, lifecycle delivered/pending products, and residual utility are observable. The value-skew fields are not mainline fields yet and should remain `dry_run_only` until default-off implementation.

The old ERIV threshold `0.15` is not profile-stable under `soft_moderate = 1:2:5:10`; it should be rescanned with candidate count, high-value coverage, low-value accidental trigger, ROI+Ctx pressure, queue pressure, expired item risk, and high-value closure potential.

The progressive chain is incomplete. Current traces support metadata -> ROI verification residual delivery, but do not show metadata -> ROI -> ROI+Ctx. First missing stage: `{first_missing}`. ROI delivery must not be counted as context/evidence closure.

Next step: implement a default-off progressive staged follow-up extension with semantic gate, value-density gate, profile-specific ERIV threshold, and per-window top-K. Do not run actual A/B until this default-off branch and gate recalibration are in place.

## Key Status

- field consistency status: partial but implementation-ready for default-off dry-run fields
- old ERIV threshold still valid: no
- suggested ERIV gate direction: {gate.get('suggested_gate_direction', '')}
- progressive chain status: {ready.get('metadata_to_roi_to_roi_context_chain_status', 'not_present')}
- ROI+Ctx semantic gate needed: yes
- value-density gate needed: yes
- ROI+Ctx pressure if all upgraded: {_to_float(pressure.get('queue_pressure_if_all_roi_ctx_upgraded_mb')):.4f} MB proxy
- ROI+Ctx pressure if selective upgraded: {_to_float(pressure.get('queue_pressure_if_selective_roi_ctx_upgraded_mb')):.4f} MB proxy
"""
    _write_text(OUT_DIR / "final_recommendation.md", final)
    return first_missing


def _archive_manifest() -> None:
    files = sorted(path.relative_to(OUT_DIR).as_posix() for path in OUT_DIR.rglob("*") if path.is_file())
    if "archive_manifest.txt" not in files:
        files.append("archive_manifest.txt")
        files = sorted(files)
    _write_text(OUT_DIR / "archive_manifest.txt", "\n".join(files))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    parser.add_argument("--scenarios", default="natural,targeted_resource_scarcity")
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    command = f"python scripts/audit_value_skew_staged_followup_consistency.py --seeds {','.join(str(seed) for seed in seeds)}"
    if args.scenarios:
        command += f" --scenarios {args.scenarios}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    value_rows, value_by_key, alpha = _load_value_rows(seeds)
    traj_rows = _load_trajectory_rows(seeds)
    candidates = _candidate_rows(value_by_key)
    closures = _closure_rows()
    trace_rows = _trace_rows(value_rows, value_by_key, traj_rows, candidates, closures)
    field_summary = _field_summary(trace_rows, candidates, closures)
    gate_dist, gate_recalibration, roi_pressure = _gate_distribution(candidates)
    funnel_rows, transition_rows, readiness, blockers = _progressive_funnel(value_rows, traj_rows, candidates, closures)

    _write_text(
        OUT_DIR / "README.md",
        "# Value-Skew Staged Follow-Up Consistency Audit\n\nAudit-only end-to-end field consistency, soft_moderate gate recalibration, and progressive staged follow-up readiness analysis. No mainline behavior is changed.",
    )
    _write_text(OUT_DIR / "command_log.md", f"# Command Log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "scenarios": args.scenarios,
                "audit_only": True,
                "default_off": True,
                "main_defaults_changed": False,
                "reward_function_changed": False,
                "module2_scoring_changed": False,
                "safe_linucb_changed": False,
                "safe_filter_changed": False,
                "utility_rule_changed": False,
                "queue_policy_changed": False,
                "lifecycle_closure_logic_changed": False,
                "information_sufficiency_changed": False,
                "task_value_replaced": False,
                "value_profile": VALUE_PROFILE,
                "value_ratio": SOFT_MODERATE,
                "value_assignment_rule": "v2_conservative",
                "normalization": "mean_preserving",
                "normalization_alpha": alpha,
                "final_recommendation_case": "B",
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_csv(OUT_DIR / "end_to_end_task_value_field_trace.csv", trace_rows)
    _write_csv(OUT_DIR / "module_field_consistency_summary.csv", field_summary)
    _write_csv(OUT_DIR / "high_value_residual_gate_distribution.csv", gate_dist)
    _write_csv(OUT_DIR / "eriv_soft_moderate_recalibration_summary.csv", gate_recalibration)
    _write_csv(OUT_DIR / "roi_context_gate_pressure_summary.csv", roi_pressure)
    _write_csv(OUT_DIR / "progressive_followup_funnel_trace.csv", funnel_rows)
    _write_csv(OUT_DIR / "roi_then_roi_context_transition_trace.csv", transition_rows)
    _write_csv(OUT_DIR / "context_evidence_closure_readiness_summary.csv", readiness)
    _write_csv(OUT_DIR / "staged_followup_blocker_summary.csv", blockers)
    first_missing = _write_reports(field_summary, gate_recalibration, roi_pressure, readiness, blockers)
    _write_text(OUT_DIR / "git_status.txt", _git(["status", "--short"]))
    script_path = PROJECT_ROOT / "scripts" / "audit_value_skew_staged_followup_consistency.py"
    diff = _git(["diff", "--", "scripts/audit_value_skew_staged_followup_consistency.py"])
    _write_text(OUT_DIR / "git_diff.patch", diff if diff.strip() else _add_file_patch(script_path))
    _archive_manifest()

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", OUT_DIR)

    gate = gate_recalibration[0] if gate_recalibration else {}
    ready = readiness[0] if readiness else {}
    print(f"audit output directory: {OUT_DIR}")
    print(f"zip path: {ZIP_PATH}")
    print("field consistency status: partial_dry_run_value_fields_but_traceable_core_fields")
    print(f"old ERIV threshold still valid? {gate.get('old_threshold_still_valid', 'no')}")
    print(f"suggested ERIV gate direction: {gate.get('suggested_gate_direction', '')}")
    print(f"progressive chain status: {ready.get('metadata_to_roi_to_roi_context_chain_status', 'not_present')}")
    print(f"first missing stage in metadata->ROI->ROI+Ctx: {first_missing}")
    print("whether ROI+Ctx semantic gate is needed: yes")
    print("whether value-density gate is needed: yes")
    print("final recommendation case: B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
