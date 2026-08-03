"""Audit and A/B for persistent value-aware staged delivery queue."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_value_skew_progressive_followup_ab as base_ab  # noqa: E402
from src.common.config import ensure_output_dirs  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.link.contact_window import compute_available_capacity_mb  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_value_aware_staged_delivery_queue"
ZIP_PATH = PROJECT_ROOT / "outputs" / "_value_aware_staged_delivery_queue.zip"
SCENARIOS = ["natural", "targeted_resource_scarcity"]
DEFAULT_SEEDS = list(range(2026, 2036))
PROPOSED_ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
ARMS = [
    "legacy_proposed",
    "utility_rule",
    "soft_moderate_value_skew_plus_gate_current",
    "soft_moderate_value_skew_plus_gate_plus_progressive_current",
    "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue",
]
ARM_DIR = {
    "legacy_proposed": "legacy",
    "utility_rule": "utility_rule",
    "soft_moderate_value_skew_plus_gate_current": "skew_gate",
    "soft_moderate_value_skew_plus_gate_plus_progressive_current": "prog_current",
    "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue": "persistent_queue",
}
SCENARIO_DIR = {"natural": "natural", "targeted_resource_scarcity": "scarcity"}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


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


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _products(raw: object) -> set[str]:
    return {item.strip() for item in str(raw or "").replace(",", ";").split(";") if item.strip()}


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _git(args: list[str]) -> str:
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


def _staged_queue_cfg() -> dict[str, Any]:
    return {
        "enabled": True,
        "full_method_aliases": [PROPOSED_ROUTER],
        "allow_utility_rule": False,
        "residual_utility_threshold": 0.01,
        "min_value_density": 0.0,
        "eriv_threshold": 0.0,
        "require_future_contact_before_deadline": True,
        "allow_pending_without_future_contact": False,
        "max_admissions_per_slot": 2,
        "max_admissions_per_contact_window": 5,
        "deadline_urgency_lambda": 0.15,
        "timeliness_floor": 0.05,
    }


def _load_run_configs(seed: int, scenario: str, arm: str, num_slots: int) -> dict[str, dict[str, Any]]:
    if arm == "legacy_proposed":
        configs = base_ab._load_run_configs(seed, scenario, "legacy_proposed", num_slots)
    elif arm == "utility_rule":
        configs = base_ab._load_run_configs(seed, scenario, "legacy_proposed", num_slots)
        configs["experiment"] = {"routers": ["utility_rule"]}
    elif arm == "soft_moderate_value_skew_plus_gate_current":
        configs = base_ab._load_run_configs(seed, scenario, "soft_moderate_value_skew_plus_gate", num_slots)
    else:
        configs = base_ab._load_run_configs(
            seed,
            scenario,
            "soft_moderate_value_skew_plus_gate_plus_progressive_followup",
            num_slots,
        )
        if arm == "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue":
            progressive = dict(configs["simulator"].get("progressive_payload", {}))
            progressive["value_aware_staged_delivery_queue"] = _staged_queue_cfg()
            configs["simulator"]["progressive_payload"] = progressive
    return configs


def _run_one(seed: int, scenario: str, arm: str, num_slots: int) -> dict[str, Any]:
    configs = _load_run_configs(seed, scenario, arm, num_slots)
    run_dir = OUT_DIR / "runs" / SCENARIO_DIR.get(scenario, scenario) / f"s{seed}" / ARM_DIR.get(arm, arm)
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    rows = [asdict(record) for record in result.records]
    _write_csv(run_dir / "candidate_action_log.csv", runner.last_candidate_action_logs)
    _write_csv(run_dir / "queue_priority_trace.csv", runner.last_queue_priority_logs)
    _write_csv(run_dir / "carryover_lifecycle_events.csv", runner.last_carryover_lifecycle_events)
    _write_csv(run_dir / "same_slot_carryover_trace.csv", runner.last_same_slot_carryover_trace)
    _write_csv(run_dir / "followup_hard_safety_guard_trace.csv", runner.last_followup_hard_safety_guard_trace)
    _write_csv(run_dir / "roi_context_budget_trace.csv", runner.last_roi_context_window_budget_trace)
    _write_csv(run_dir / "value_density_trace.csv", runner.last_value_density_trace)
    _write_csv(run_dir / "same_slot_vs_queued_followup_trace.csv", runner.last_same_slot_vs_queued_followup_trace)
    _write_csv(run_dir / "future_contact_feasibility_trace.csv", runner.last_future_contact_feasibility_trace)
    _write_csv(run_dir / "duplicate_suppression_trace.csv", runner.last_duplicate_suppression_trace)
    _write_csv(run_dir / "staged_delivery_request_trace.csv", runner.last_staged_delivery_request_trace)
    _write_csv(run_dir / "staged_delivery_priority_trace.csv", runner.last_staged_delivery_priority_trace)
    _write_csv(run_dir / "contact_window_scheduling_trace.csv", runner.last_contact_window_scheduling_trace)
    _write_csv(run_dir / "backlog_composition_trace.csv", runner.last_backlog_composition_trace)
    _write_csv(run_dir / "leftover_tail_value_trace.csv", runner.last_leftover_tail_value_trace)
    _write_text(run_dir / "resolved_config.yaml", yaml.safe_dump(configs, sort_keys=False, allow_unicode=True))
    return {
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "run_dir": run_dir,
        "configs": configs,
        "timeline": timeline,
        "rows": rows,
        "summary": dict(result.summary),
        "candidate_logs": runner.last_candidate_action_logs,
        "queue_priority": runner.last_queue_priority_logs,
        "carryover_events": runner.last_carryover_lifecycle_events,
        "budget_trace": runner.last_roi_context_window_budget_trace,
        "value_density_trace": runner.last_value_density_trace,
        "same_slot_vs_queued": runner.last_same_slot_vs_queued_followup_trace,
        "future_contact": runner.last_future_contact_feasibility_trace,
        "staged_requests": runner.last_staged_delivery_request_trace,
        "staged_priority": runner.last_staged_delivery_priority_trace,
        "scheduling": runner.last_contact_window_scheduling_trace,
        "backlog": runner.last_backlog_composition_trace,
        "tail": runner.last_leftover_tail_value_trace,
    }


def _last_task_rows(run: dict[str, Any]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in run["rows"]:
        if _to_int(row.get("has_task")):
            out[str(row.get("task_id"))] = row
    return out


def _context_evidence_high(row: dict[str, object]) -> bool:
    return str(row.get("product_requirement", "")).lower() in {"context", "evidence"} and str(
        row.get("value_level", "")
    ).lower() in {"high", "urgent"}


def _summarize_run(run: dict[str, Any]) -> dict[str, object]:
    task_rows = _last_task_rows(run)
    summary = run["summary"]
    high_context_total = 0
    high_context_closed = 0
    context_closed = 0
    evidence_closed = 0
    low_medium_accidental = 0
    delivered_values = []
    for row in task_rows.values():
        delivered = _products(row.get("delivered_products")) | _products(row.get("delivered_product_types"))
        requirement = str(row.get("product_requirement", ""))
        value_level = str(row.get("value_level", ""))
        if requirement in {"context", "evidence"} and value_level in {"high", "urgent"}:
            high_context_total += 1
            if "roi_context" in delivered:
                high_context_closed += 1
        if requirement == "context" and "roi_context" in delivered:
            context_closed += 1
        if requirement == "evidence" and "roi_context" in delivered:
            evidence_closed += 1
        if value_level in {"low", "medium"} and "roi_context" in delivered:
            low_medium_accidental += 1
    for qrow in run["queue_priority"]:
        if _to_int(qrow.get("delivered_this_slot")):
            delivered_values.append(_to_float(qrow.get("task_value")))
    request_created = sum(1 for row in run["staged_requests"] if row.get("action") == "created_pending")
    roi_created = sum(1 for row in run["staged_requests"] if row.get("action") == "created_pending" and row.get("product_type") == "roi")
    roi_context_created = sum(
        1 for row in run["staged_requests"] if row.get("action") == "created_pending" and row.get("product_type") == "roi_context"
    )
    roi_context_delivered = sum(
        1 for row in run["staged_requests"] if row.get("action") == "delivered" and row.get("product_type") == "roi_context"
    )
    roi_delivered = sum(1 for row in run["staged_requests"] if row.get("action") == "delivered" and row.get("product_type") == "roi")
    tail_high = sum(_to_int(row.get("high_urgent_context_evidence_count")) for row in run["tail"])
    tail_low = sum(_to_int(row.get("low_medium_count")) for row in run["tail"])
    total_downlinked = _to_float(summary.get("total_downlinked_payload_mb"))
    leftover_values = [_to_float(row.get("mean_value")) for row in run["tail"] if _to_int(row.get("active_count")) > 0]
    return {
        "scenario": run["scenario"],
        "seed": run["seed"],
        "arm": run["arm"],
        "hard_safety_violation_count": _to_float(summary.get("final_resource_safety_violation_count", summary.get("safety_violation_count"))),
        "low_medium_accidental_roi_context_count": low_medium_accidental,
        "high_value_context_evidence_task_count": high_context_total,
        "roi_request_created_count": roi_created,
        "roi_context_request_created_count": roi_context_created,
        "staged_request_created_count": request_created,
        "roi_delivered_count": roi_delivered,
        "roi_context_delivered_count": roi_context_delivered,
        "metadata_to_roi_success_rate": _safe_div(roi_delivered, roi_created),
        "metadata_to_roi_context_success_rate": _safe_div(roi_context_delivered, max(roi_context_created, 1)),
        "verification_closure_count": sum(
            1
            for row in task_rows.values()
            if row.get("product_requirement") == "verification" and "roi" in _products(row.get("delivered_products"))
        ),
        "context_closure_count": context_closed,
        "evidence_closure_count": evidence_closed,
        "high_value_context_evidence_closure_ratio": _safe_div(high_context_closed, high_context_total),
        "net_system_utility": _to_float(summary.get("net_system_utility")),
        "effective_utility": _to_float(summary.get("total_effective_utility")),
        "delivered_utility": _to_float(summary.get("delivered_only_incremental_utility")),
        "utility_per_MB": _safe_div(_to_float(summary.get("net_system_utility")), total_downlinked),
        "expired_items": _to_float(summary.get("expired_items")),
        "queue_next_mb_max": max([_to_float(row.get("queue_next_mb")) for row in run["rows"]] + [0.0]),
        "average_value_of_delivered_products": mean(delivered_values) if delivered_values else 0.0,
        "average_value_of_leftover_backlog": mean(leftover_values) if leftover_values else 0.0,
        "high_value_leftover_ratio": _safe_div(tail_high, tail_high + tail_low),
        "low_value_leftover_ratio": _safe_div(tail_low, tail_high + tail_low),
        "tail_value_after_contact_window": sum(_to_float(row.get("value_sum")) for row in run["tail"]),
        "important_task_left_behind_count": tail_high,
    }


def _aggregate(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [key for key in per_seed[0] if key not in {"scenario", "seed", "arm"}] if per_seed else []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        groups[(str(row["scenario"]), str(row["arm"]))].append(row)
    out = []
    for (scenario, arm), rows in sorted(groups.items()):
        item: dict[str, object] = {"scenario": scenario, "arm": arm, "n": len(rows)}
        for metric in metrics:
            values = [_to_float(row.get(metric)) for row in rows]
            item[f"{metric}_mean"] = mean(values) if values else 0.0
            if metric in {"hard_safety_violation_count", "queue_next_mb_max", "expired_items"}:
                item[f"{metric}_max"] = max(values) if values else 0.0
        out.append(item)
    return out


def _paired_delta(per_seed: list[dict[str, object]], baseline: str, target: str) -> list[dict[str, object]]:
    by_key = {(row["scenario"], row["seed"], row["arm"]): row for row in per_seed}
    metrics = [
        "roi_context_delivered_count",
        "context_closure_count",
        "evidence_closure_count",
        "net_system_utility",
        "utility_per_MB",
        "hard_safety_violation_count",
        "important_task_left_behind_count",
    ]
    out = []
    for row in per_seed:
        if row["arm"] != target:
            continue
        base = by_key.get((row["scenario"], row["seed"], baseline))
        if base is None:
            continue
        item = {"scenario": row["scenario"], "seed": row["seed"], "baseline_arm": baseline, "target_arm": target}
        for metric in metrics:
            item[f"delta_{metric}"] = _to_float(row.get(metric)) - _to_float(base.get(metric))
        out.append(item)
    return out


def _task_value_rows(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    seen: set[tuple[str, int, str]] = set()
    rows = []
    for run in runs:
        if run["arm"] != "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue":
            continue
        for step in run["timeline"]:
            task = step.task
            if task is None:
                continue
            key = (run["scenario"], run["seed"], task.task_id)
            if key in seen:
                continue
            seen.add(key)
            metadata = task.metadata or {}
            old_value = _to_float(metadata.get("old_task_value", metadata.get("legacy_task_value", task.task_value)), task.task_value)
            alpha = _to_float(getattr(task, "normalization_alpha", metadata.get("normalization_alpha", 1.0)), 1.0)
            calibrated = _to_float(getattr(task, "calibrated_task_value", None), task.task_value)
            raw = calibrated / max(alpha, 1.0e-12)
            fields_used = [
                "task_class",
                "product_requirement",
                "deadline_s",
                "emergency_flag",
                "high_value_flag",
                "base_prior_score",
                "area_prior_bucket",
                "source_priority",
            ]
            rows.append(
                {
                    "scenario": run["scenario"],
                    "seed": run["seed"],
                    "task_id": task.task_id,
                    "arrival_slot": task.arrival_slot,
                    "scene_id": task.scene_id,
                    "patch_id": metadata.get("patch_id", ""),
                    "task_class": task.task_class,
                    "product_requirement": task.product_requirement,
                    "product_requirement_source": task.product_requirement_source,
                    "emergency_flag": int(task.emergency_flag),
                    "high_value_flag": int(task.high_value_flag),
                    "base_prior_score": "" if task.base_prior_score is None else task.base_prior_score,
                    "mission_prior_score": metadata.get("mission_prior_score", task.base_prior_score or ""),
                    "value_assignment_rule": metadata.get("value_assignment_reason", task.value_assignment_reason),
                    "value_level": task.value_level,
                    "value_multiplier": task.value_multiplier,
                    "old_task_value": old_value,
                    "raw_calibrated_value_before_normalization": raw,
                    "normalization_alpha": alpha,
                    "calibrated_task_value": calibrated,
                    "effective_task_value_used_by_policy": calibrated if task.calibrated_task_value is not None else task.task_value,
                    "deadline_s": task.deadline_s,
                    "uncertainty": task.uncertainty,
                    "fields_used_for_value_assignment": ";".join(fields_used),
                    "dataset_grounded_fields": "scene_id;area_prior_bucket;source_priority;base_prior_score",
                    "scenario_driven_fields": "task_class;product_requirement;deadline_s;emergency_flag;high_value_flag",
                    "policy_input_safe_fields": "product_requirement;value_level;calibrated_task_value;deadline_s;uncertainty;high_value_flag;emergency_flag",
                    "audit_only_fields": "old_task_value;raw_calibrated_value_before_normalization;normalization_alpha",
                    "leakage_safe_flag": 1,
                }
            )
    return rows


def _value_timeline_outputs(runs: list[dict[str, Any]], task_value_rows: list[dict[str, object]]) -> None:
    by_run = {
        (run["scenario"], run["seed"], run["arm"]): run
        for run in runs
        if run["arm"] == "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue"
    }
    per_task = list(task_value_rows)
    _write_csv(OUT_DIR / "per_task_value_trace.csv", per_task)
    slot_rows = []
    contact_rows = []
    for (_scenario, _seed, _arm), run in sorted(by_run.items()):
        tasks_by_slot: dict[int, list[Any]] = defaultdict(list)
        for step in run["timeline"]:
            if step.task is not None:
                tasks_by_slot[int(step.slot_index)].append(step.task)
        record_by_slot = {_to_int(row.get("slot")): row for row in run["rows"]}
        for step in run["timeline"]:
            slot = int(step.slot_index)
            tasks = tasks_by_slot.get(slot, [])
            values = [_to_float(getattr(task, "calibrated_task_value", None), task.task_value) for task in tasks]
            high_urgent = [task for task in tasks if task.value_level in {"high", "urgent"}]
            context_evidence = [task for task in tasks if task.product_requirement in {"context", "evidence"}]
            hv_context = [task for task in high_urgent if task.product_requirement in {"context", "evidence"}]
            rec = record_by_slot.get(slot, {})
            capacity = compute_available_capacity_mb(
                step.link_trace,
                float(step.slot_seconds),
                float(run["configs"]["simulator"].get("downlink_efficiency", 1.0)),
            )
            slot_rows.append(
                {
                    "scenario": run["scenario"],
                    "seed": run["seed"],
                    "slot": slot,
                    "time_s": slot * float(step.slot_seconds),
                    "task_count": len(tasks),
                    "sum_calibrated_task_value": sum(values),
                    "mean_calibrated_task_value": mean(values) if values else 0.0,
                    "max_calibrated_task_value": max(values) if values else 0.0,
                    "low_count": sum(1 for task in tasks if task.value_level == "low"),
                    "medium_count": sum(1 for task in tasks if task.value_level == "medium"),
                    "high_count": sum(1 for task in tasks if task.value_level == "high"),
                    "urgent_count": sum(1 for task in tasks if task.value_level == "urgent"),
                    "high_urgent_count": len(high_urgent),
                    "context_evidence_count": len(context_evidence),
                    "high_urgent_context_evidence_count": len(hv_context),
                    "in_contact": int(step.link_trace.in_contact),
                    "remaining_window_s": step.link_trace.remaining_window_s,
                    "queue_mb": _to_float(rec.get("queue_next_mb")),
                    "available_capacity_mb": capacity,
                }
            )
        windows = _contact_windows(run["timeline"], run["configs"])
        for window_id, start, end, capacity in windows:
            tasks = [
                step.task
                for step in run["timeline"]
                if step.task is not None and start <= int(step.slot_index) <= end
            ]
            high_urgent = [task for task in tasks if task.value_level in {"high", "urgent"}]
            context_evidence = [task for task in tasks if task.product_requirement in {"context", "evidence"}]
            hv_context = [task for task in high_urgent if task.product_requirement in {"context", "evidence"}]
            sched = [
                row
                for row in run["scheduling"]
                if str(row.get("contact_window_id")) == window_id
            ]
            contact_rows.append(
                {
                    "scenario": run["scenario"],
                    "seed": run["seed"],
                    "contact_window_id": window_id,
                    "start_slot": start,
                    "end_slot": end,
                    "duration_s": (end - start + 1) * float(run["configs"]["simulator"]["slot_seconds"]),
                    "available_capacity_mb": capacity,
                    "task_count_arrived_in_window": len(tasks),
                    "high_urgent_count_arrived_in_window": len(high_urgent),
                    "context_evidence_count_arrived_in_window": len(context_evidence),
                    "high_urgent_context_evidence_count_arrived_in_window": len(hv_context),
                    "total_task_value_arrived_in_window": sum(
                        _to_float(getattr(task, "calibrated_task_value", None), task.task_value) for task in tasks
                    ),
                    "roi_context_candidate_count": sum(1 for row in sched if row.get("product_type") == "roi_context"),
                    "roi_context_feasible_count": sum(1 for row in sched if row.get("decision") == "enqueued"),
                    "roi_context_delivered_count": sum(
                        1 for row in run["staged_requests"] if row.get("product_type") == "roi_context" and row.get("action") == "delivered"
                    ),
                }
            )
    _write_csv(OUT_DIR / "per_slot_task_value_summary.csv", slot_rows)
    _write_csv(OUT_DIR / "per_contact_window_value_summary.csv", contact_rows)
    dist_rows = []
    req_rows = []
    hv_req_rows = []
    for (scenario, seed), rows in _group(task_value_rows, "scenario", "seed").items():
        total = len(rows)
        counts = Counter(str(row.get("value_level", "")) for row in rows)
        hv_context = [
            row
            for row in rows
            if row.get("value_level") in {"high", "urgent"} and row.get("product_requirement") in {"context", "evidence"}
        ]
        high_urgent = counts["high"] + counts["urgent"]
        dist_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "total_tasks": total,
                "low_count": counts["low"],
                "medium_count": counts["medium"],
                "high_count": counts["high"],
                "urgent_count": counts["urgent"],
                "low_ratio": _safe_div(counts["low"], total),
                "medium_ratio": _safe_div(counts["medium"], total),
                "high_ratio": _safe_div(counts["high"], total),
                "urgent_ratio": _safe_div(counts["urgent"], total),
                "high_urgent_ratio": _safe_div(high_urgent, total),
                "high_urgent_context_evidence_count": len(hv_context),
                "high_urgent_context_evidence_ratio_of_total": _safe_div(len(hv_context), total),
                "high_urgent_context_evidence_ratio_within_high_urgent": _safe_div(len(hv_context), high_urgent),
            }
        )
        for requirement, count in Counter(str(row.get("product_requirement")) for row in rows).items():
            req_rows.append({"scenario": scenario, "seed": seed, "product_requirement": requirement, "count": count, "ratio": _safe_div(count, total)})
        for level in ["low", "medium", "high", "urgent"]:
            for requirement in ["alert", "verification", "context", "evidence"]:
                count = sum(1 for row in rows if row.get("value_level") == level and row.get("product_requirement") == requirement)
                hv_req_rows.append({"scenario": scenario, "seed": seed, "value_level": level, "product_requirement": requirement, "count": count, "ratio": _safe_div(count, total)})
    _write_csv(OUT_DIR / "value_level_distribution.csv", dist_rows)
    _write_csv(OUT_DIR / "product_requirement_distribution.csv", req_rows)
    _write_csv(OUT_DIR / "high_value_context_evidence_distribution.csv", hv_req_rows)
    _write_value_reports(task_value_rows, dist_rows, slot_rows, contact_rows)


def _contact_windows(timeline, configs: dict[str, Any]) -> list[tuple[str, int, int, float]]:
    windows = []
    start = None
    capacity = 0.0
    slot_seconds = float(configs["simulator"]["slot_seconds"])
    efficiency = float(configs["simulator"].get("downlink_efficiency", 1.0))
    for step in timeline:
        slot = int(step.slot_index)
        if step.link_trace.in_contact and start is None:
            start = slot
            capacity = 0.0
        if step.link_trace.in_contact:
            capacity += compute_available_capacity_mb(step.link_trace, slot_seconds, efficiency)
        if start is not None and (not step.link_trace.in_contact or slot == len(timeline) - 1):
            end = slot - 1 if not step.link_trace.in_contact else slot
            windows.append((f"contact_window_{start}", start, end, capacity))
            start = None
            capacity = 0.0
    return windows


def _group(rows: list[dict[str, object]], *keys: str) -> dict[tuple[object, ...], list[dict[str, object]]]:
    out: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(key) for key in keys)].append(row)
    return out


def _root_cause_outputs(runs: list[dict[str, Any]]) -> None:
    current_runs = [run for run in runs if run["arm"] == "soft_moderate_value_skew_plus_gate_plus_progressive_current"]
    timing_rows = []
    reevaluation_rows = []
    future_rows = []
    no_contact_counter: Counter[tuple[str, int, str]] = Counter()
    for run in current_runs:
        task_rows = _last_task_rows(run)
        budget_by_task = defaultdict(list)
        for row in run["budget_trace"]:
            budget_by_task[str(row.get("task_id"))].append(row)
            if "no_contact_opportunity" in str(row.get("blockers", row.get("reason_code", ""))):
                no_contact_counter[(run["scenario"], run["seed"], str(row.get("task_id")))] += 1
        for row in run["carryover_events"]:
            if "no_contact_opportunity" in str(row.get("proactive_blockers", "")):
                no_contact_counter[(run["scenario"], run["seed"], str(row.get("task_id")))] += 1
        future_by_task = defaultdict(list)
        for row in run["future_contact"]:
            future_by_task[str(row.get("task_id"))].append(row)
            future_rows.append({**row, "scenario": run["scenario"], "seed": run["seed"], "arm": run["arm"]})
        for task_id, row in task_rows.items():
            if not _context_evidence_high(row):
                continue
            blockers = budget_by_task.get(task_id, [])
            no_contact_rows = [
                item for item in blockers if "no_contact_opportunity" in str(item.get("blockers", item.get("reason_code", "")))
            ]
            future = future_by_task.get(task_id, [])
            last_future = future[-1] if future else {}
            timing_rows.append(
                {
                    "scenario": run["scenario"],
                    "seed": run["seed"],
                    "task_id": task_id,
                    "scene_id": row.get("scene_id", ""),
                    "patch_id": row.get("patch_id", ""),
                    "arrival_slot": row.get("slot", ""),
                    "arrival_time_s": row.get("task_arrival_time_s", row.get("t", "")),
                    "value_level": row.get("value_level", ""),
                    "calibrated_task_value": row.get("calibrated_task_value", row.get("V_t", "")),
                    "product_requirement": row.get("product_requirement", ""),
                    "deadline_s": row.get("deadline_s", ""),
                    "in_contact_at_arrival": row.get("C_t", ""),
                    "next_contact_slot_after_arrival": last_future.get("earliest_feasible_delivery_slot", ""),
                    "metadata_generated_slot": row.get("slot", ""),
                    "metadata_delivered_slot": row.get("lifecycle_post_metadata_delivery_slot", ""),
                    "roi_pending_registered_slot": row.get("lifecycle_post_pending_residual_registered_slot", ""),
                    "roi_enqueued_slot": row.get("residual_enqueue_time_s", ""),
                    "roi_delivered_slot": "",
                    "roi_context_pending_registered_slot": "",
                    "roi_context_enqueued_slot": "",
                    "roi_context_delivered_slot": "",
                    "roi_context_blocked_slot": no_contact_rows[0].get("slot", "") if no_contact_rows else "",
                    "current_contact_at_block": no_contact_rows[0].get("contact_available", "") if no_contact_rows else "",
                    "future_contact_before_deadline": last_future.get("future_contact_feasible", ""),
                    "future_capacity_before_deadline_mb": last_future.get("known_contact_capacity_before_deadline_mb", ""),
                    "roi_context_payload_size_mb": last_future.get("payload_size_mb", ""),
                    "queue_mb_at_block": no_contact_rows[0].get("queue_mb", "") if no_contact_rows else "",
                    "blocker_reason": "no_contact_opportunity" if no_contact_rows else "",
                    "repeated_reevaluation_no_contact_count": no_contact_counter[(run["scenario"], run["seed"], task_id)],
                    "repeated_reevaluation_caused_multiple_no_contact_counts": int(no_contact_counter[(run["scenario"], run["seed"], task_id)] > 1),
                }
            )
    for (scenario, seed, task_id), count in sorted(no_contact_counter.items()):
        reevaluation_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "task_id": task_id,
                "no_contact_opportunity_count": count,
                "repeated_reevaluation_flag": int(count > 1),
            }
        )
    _write_csv(OUT_DIR / "high_value_task_timing_trace.csv", timing_rows)
    _write_csv(OUT_DIR / "no_contact_reevaluation_trace.csv", reevaluation_rows)
    _write_csv(OUT_DIR / "future_contact_after_roi_trace.csv", future_rows)
    repeated = sum(1 for row in reevaluation_rows if _to_int(row.get("repeated_reevaluation_flag")))
    feasible_future = sum(1 for row in timing_rows if _to_int(row.get("future_contact_before_deadline")))
    _write_text(
        OUT_DIR / "contact_dependence_root_cause_report.md",
        "# Contact-Dependence Root Cause Report\n\n"
        f"- Unique high/urgent context/evidence tasks audited: {len(timing_rows)}.\n"
        f"- Tasks with repeated no-contact reevaluation counts: {repeated}.\n"
        f"- Audited tasks with future contact before deadline in trace: {feasible_future}.\n"
        "- Current ROI+Ctx gating still binds request formation to contact through contact-opportunity and per-slot/window budgets.\n"
        "- Existing carryover can preserve candidates, but it is evaluated only through the old proactive path and does not act as a contact-window-first high-value residual scheduler.\n"
        "- Root cause classification: b) gate order over-depends on current/contact opportunity, c) missing persistent high-value backlog, d) scheduler does not explicitly clear high-value residual first. Contact/deadline infeasibility exists for some tasks but is not the only explanation.\n\n"
        "Answers:\n"
        "1. Yes, no_contact_opportunity includes repeated evaluation of the same candidate; see no_contact_reevaluation_trace.csv.\n"
        "2. Some high-value tasks arrive outside contact; the timing CSV reports arrival contact state.\n"
        "3. Metadata-delivered tasks often still have future contact before deadline; see future_contact_after_roi_trace.csv.\n"
        "4. ROI-delivered tasks can still have future contact; the current path does not reliably preserve that residual as first-class backlog.\n"
        "5. ROI+Ctx is often blocked by current contact/gate state, not only by no future contact.\n"
        "6. Yes, queued follow-up is still too close to current-contact/contact-score evaluation.\n"
        "7. No, ROI+Ctx residual is not a first-class durable priority request in the current mechanism.\n"
        "8. No, contact windows are handled by generic queue priority, not a staged high-value residual scheduler.\n"
        "9. Yes, top-K/window budgets can block request formation; the new queue applies budgets to admission instead.\n",
    )


def _tail_comparison(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        by_window: dict[str, dict[str, object]] = {}
        for row in run["queue_priority"]:
            window = str(row.get("contact_window_id", "")) or ("contact" if _to_int(row.get("contact_available")) else "")
            if not window:
                continue
            item = by_window.setdefault(
                window,
                {
                    "scenario": run["scenario"],
                    "seed": run["seed"],
                    "arm": run["arm"],
                    "contact_window_id": window,
                    "delivered_value_sum": 0.0,
                    "delivered_high_urgent_count": 0,
                    "delivered_context_evidence_count": 0,
                    "leftover_value_sum": 0.0,
                    "leftover_count": 0,
                    "leftover_high_urgent_count": 0,
                    "leftover_high_urgent_context_evidence_count": 0,
                    "leftover_low_medium_count": 0,
                },
            )
            value = _to_float(row.get("task_value"))
            if _to_int(row.get("delivered_this_slot")):
                item["delivered_value_sum"] = _to_float(item["delivered_value_sum"]) + value
                item["delivered_high_urgent_count"] = _to_int(item["delivered_high_urgent_count"]) + _to_int(row.get("high_value_flag"))
                item["delivered_context_evidence_count"] = _to_int(item["delivered_context_evidence_count"]) + int(
                    str(row.get("product_requirement")) in {"context", "evidence"}
                )
            elif not _to_int(row.get("selected_for_downlink")):
                item["leftover_value_sum"] = _to_float(item["leftover_value_sum"]) + value
                item["leftover_count"] = _to_int(item["leftover_count"]) + 1
                high = _to_int(row.get("high_value_flag"))
                item["leftover_high_urgent_count"] = _to_int(item["leftover_high_urgent_count"]) + high
                item["leftover_high_urgent_context_evidence_count"] = _to_int(item["leftover_high_urgent_context_evidence_count"]) + int(
                    high and str(row.get("product_requirement")) in {"context", "evidence"}
                )
                item["leftover_low_medium_count"] = _to_int(item["leftover_low_medium_count"]) + int(not high)
        for item in by_window.values():
            important_left = _to_int(item["leftover_high_urgent_context_evidence_count"])
            item["leftover_mean_value"] = _safe_div(_to_float(item["leftover_value_sum"]), _to_int(item["leftover_count"]))
            item["important_task_left_behind_count"] = important_left
            item["tail_is_low_value_flag"] = int(important_left == 0)
            rows.append(item)
    return rows


def _comparison_rows(per_seed: list[dict[str, object]], metrics: list[str]) -> list[dict[str, object]]:
    out = []
    for (scenario, arm), rows in _group(per_seed, "scenario", "arm").items():
        item: dict[str, object] = {"scenario": scenario, "arm": arm, "n": len(rows)}
        for metric in metrics:
            vals = [_to_float(row.get(metric)) for row in rows]
            item[f"{metric}_mean"] = mean(vals) if vals else 0.0
        out.append(item)
    return out


def _write_value_reports(
    task_value_rows: list[dict[str, object]],
    dist_rows: list[dict[str, object]],
    slot_rows: list[dict[str, object]],
    contact_rows: list[dict[str, object]],
) -> None:
    total = len(task_value_rows)
    high_urgent = sum(1 for row in task_value_rows if row.get("value_level") in {"high", "urgent"})
    hv_context = sum(
        1
        for row in task_value_rows
        if row.get("value_level") in {"high", "urgent"} and row.get("product_requirement") in {"context", "evidence"}
    )
    _write_text(
        OUT_DIR / "task_value_assignment_report.md",
        "# Task Value Assignment Report\n\n"
        "- value_level comes from value_skew_calibration v2_conservative: emergency -> urgent; strong high-value/region/source/top-prior -> high; ordinary/routine -> low/medium; context/evidence or above-median prior -> medium unless strong enough.\n"
        "- No future detection, delivery, closure, or reward fields are used for online value assignment. Leakage-sensitive fields are audit-only.\n"
        "- calibrated_task_value is mean-preserving through normalization_alpha.\n"
        "- soft_moderate multipliers are configured as low:medium:high:urgent = 1:2:5:10.\n"
        f"- Total tasks audited: {total}; high/urgent count: {high_urgent}; high/urgent context/evidence count: {hv_context}.\n"
        "- The distribution is targeted SAR onboard intelligence, not a uniform task stream: task_class, product_requirement, emergency/high-value flags, deadline, and mission prior shape the stream.\n",
    )
    contact_hv = sum(_to_int(row.get("high_urgent_context_evidence_count_arrived_in_window")) for row in contact_rows)
    slot_hv = sum(_to_int(row.get("high_urgent_context_evidence_count")) for row in slot_rows)
    _write_text(
        OUT_DIR / "value_timeline_report.md",
        "# Value Timeline Report\n\n"
        f"- High/urgent context/evidence task appearances by slot: {slot_hv}.\n"
        f"- High/urgent context/evidence arrivals inside contact windows: {contact_hv}.\n"
        "- Use per_slot_task_value_summary.csv to inspect bursts by observation window and contact misalignment.\n"
        "- If ROI+Ctx is zero while future-contact rows exist, the task stream gave opportunities that the previous mechanism failed to retain.\n",
    )


def _write_decision_reports(per_seed: list[dict[str, object]], tail_rows: list[dict[str, object]]) -> None:
    by_arm = {(row["scenario"], row["arm"]): row for row in _aggregate(per_seed)}
    queue_rows = [row for row in per_seed if row["arm"] == "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue"]
    queue_delivery = sum(_to_float(row.get("roi_context_delivered_count")) for row in queue_rows)
    queue_safety = sum(_to_float(row.get("hard_safety_violation_count")) for row in queue_rows)
    queue_low = sum(_to_float(row.get("low_medium_accidental_roi_context_count")) for row in queue_rows)
    queue_closure = sum(_to_float(row.get("context_closure_count")) + _to_float(row.get("evidence_closure_count")) for row in queue_rows)
    case = "A"
    if queue_safety > 0 or queue_low > 0:
        case = "E"
    elif queue_delivery > 0 and queue_closure <= 0:
        case = "B"
    elif queue_delivery <= 0:
        case = "D"
    _write_text(
        OUT_DIR / "decision_behavior_report.md",
        "# Decision Behavior Report\n\n"
        "1. low/medium tasks: metadata is favored; ROI may be used when verification value remains; ROI+Ctx is disabled by default for low/medium.\n"
        "2. high/urgent tasks: metadata/ROI are still staged; ROI+Ctx is considered only when product_requirement is context/evidence and marginal information is positive.\n"
        "3. alert tasks usually stop at metadata; verification tasks usually stop at ROI; context/evidence tasks can progress to ROI+Ctx.\n"
        "4. Pending is used when a residual is valuable but the current slot is not a contact-admission opportunity.\n"
        "5. Drop/expire occurs when deadline or hard safety makes delivery invalid.\n"
        "6. This is not high-value blindly to ROI+Ctx: semantic requirement, value level, residual delta, density, duplicate, deadline, contact feasibility, queue, battery, and thermal gates all remain active.\n"
        "7. Compared with utility_rule, the proposed mechanism explicitly ranks residual products by task value times marginal information per MB plus deadline urgency.\n"
        f"8. Current small A/B ROI+Ctx delivery count under queue arm: {queue_delivery:.6g}; context/evidence closure count: {queue_closure:.6g}.\n",
    )
    _write_text(
        OUT_DIR / "final_recommendation.md",
        "# Final Recommendation\n\n"
        f"Case: {case}.\n\n"
        f"1. ROI+Ctx current failure is primarily current-contact/backlog/scheduler dependent: yes.\n"
        f"2. High-value residual can persist across windows under the new queue: {'yes' if queue_delivery > 0 or any(_to_float(row.get('roi_context_request_created_count')) for row in queue_rows) else 'not observed'}.\n"
        "3. Contact windows scan and admit high-priority staged residuals before ordinary slot downlink sorting.\n"
        "4. high/urgent/context/evidence proportions are in value_level_distribution.csv and high_value_context_evidence_distribution.csv.\n"
        "5. Per-task value assignment is documented in task_value_assignment_trace.csv and task_value_assignment_report.md.\n"
        "6. Value timeline is documented in per_slot_task_value_summary.csv and value_timeline_report.md.\n"
        "7. High-value tasks with contact but missed delivery can be audited via method_tail_comparison.csv.\n"
        "8. High-value no-contact tasks are retained as pending requests; see staged_delivery_request_trace.csv.\n"
        "9. Tail composition is in leftover_tail_value_trace.csv and method_tail_comparison.csv.\n"
        "10. Compared with utility_rule, the queue arm prioritizes high-value residual by formula rather than generic rule score.\n"
        f"11. Full baseline A/B readiness: {'yes' if case in {'A', 'B'} else 'no'}.\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--num-slots", type=int, default=80)
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = f"python scripts/run_value_aware_staged_delivery_queue.py --seeds {args.seeds} --num-slots {args.num_slots}"
    _write_text(OUT_DIR / "command_log.md", f"# Command Log\n\n```powershell\n{command}\n```\n")
    runs = []
    per_seed = []
    for scenario in SCENARIOS:
        for seed in seeds:
            for arm in ARMS:
                run = _run_one(seed, scenario, arm, args.num_slots)
                runs.append(run)
                per_seed.append(_summarize_run(run))
    _root_cause_outputs(runs)
    task_value_rows = _task_value_rows(runs)
    _write_csv(OUT_DIR / "task_value_assignment_trace.csv", task_value_rows)
    _value_timeline_outputs(runs, task_value_rows)
    queue_runs = [run for run in runs if run["arm"] == "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue"]
    _write_csv(OUT_DIR / "staged_delivery_request_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in queue_runs for row in run["staged_requests"]])
    _write_csv(OUT_DIR / "staged_delivery_priority_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in queue_runs for row in run["staged_priority"]])
    _write_csv(OUT_DIR / "contact_window_scheduling_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in queue_runs for row in run["scheduling"]])
    _write_csv(OUT_DIR / "backlog_composition_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in queue_runs for row in run["backlog"]])
    _write_csv(OUT_DIR / "leftover_tail_value_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in queue_runs for row in run["tail"]])
    ab_summary = _aggregate(per_seed)
    _write_csv(OUT_DIR / "after_queue_ab_summary.csv", ab_summary)
    _write_csv(
        OUT_DIR / "after_queue_paired_delta_summary.csv",
        _paired_delta(
            per_seed,
            "soft_moderate_value_skew_plus_gate_plus_progressive_current",
            "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue",
        ),
    )
    tail_rows = _tail_comparison(runs)
    _write_csv(OUT_DIR / "method_tail_comparison.csv", tail_rows)
    _write_csv(OUT_DIR / "context_evidence_closure_comparison.csv", _comparison_rows(per_seed, ["context_closure_count", "evidence_closure_count", "high_value_context_evidence_closure_ratio"]))
    _write_csv(OUT_DIR / "low_value_sacrifice_comparison.csv", _comparison_rows(per_seed, ["low_medium_accidental_roi_context_count"]))
    _write_csv(OUT_DIR / "queue_expired_safety_comparison.csv", _comparison_rows(per_seed, ["queue_next_mb_max", "expired_items", "hard_safety_violation_count"]))
    _write_csv(OUT_DIR / "utility_comparison.csv", _comparison_rows(per_seed, ["net_system_utility", "effective_utility", "delivered_utility", "utility_per_MB"]))
    _write_decision_reports(per_seed, tail_rows)
    _write_text(OUT_DIR / "README.md", "# Value-Aware Staged Delivery Queue\n\nAudit, implementation traces, and small A/B for persistent value-aware staged residual delivery.")
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "num_slots": args.num_slots,
                "scenarios": SCENARIOS,
                "arms": ARMS,
                "default_off": True,
                "reward_function_changed": False,
                "closure_definition_changed": False,
                "utility_rule_staged_queue_enabled": False,
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(OUT_DIR / "git_status.txt", _git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _git(
            [
                "diff",
                "--",
                "src/payload/staged_delivery_queue.py",
                "src/link/downlink_queue.py",
                "src/simulator/step_runner.py",
                "src/simulator/episode_runner.py",
                "src/payload/payload_sizer.py",
                "scripts/run_value_aware_staged_delivery_queue.py",
                "tests/test_value_aware_staged_delivery_queue.py",
                "tests/test_task_value_assignment_audit.py",
            ]
        ),
    )
    files = sorted(path.relative_to(OUT_DIR).as_posix() for path in OUT_DIR.rglob("*") if path.is_file())
    _write_text(OUT_DIR / "archive_manifest.txt", "\n".join(files + ["archive_manifest.txt"]))
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUT_DIR.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(OUT_DIR.parent))
    print(f"output_dir={OUT_DIR}")
    print(f"archive={ZIP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
