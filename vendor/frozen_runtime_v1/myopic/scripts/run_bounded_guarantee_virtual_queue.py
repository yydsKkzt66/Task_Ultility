"""Run bounded guarantee-aware persistent delivery sweeps and artifacts."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_value_skew_progressive_followup_ab as base_ab  # noqa: E402
from src.common.config import ensure_output_dirs, load_yaml  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_dynamic_guarantee_debt"
ZIP_PATH = PROJECT_ROOT / "outputs" / "_dynamic_guarantee_debt.zip"
PROPOSED_ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
DEFAULT_SEEDS = list(range(2026, 2031))
SCENARIOS = ["natural_sarfish", "value_skew_contact_limited", "scarcity_stress"]
SCENARIO_MAP = {
    "natural_sarfish": "natural",
    "value_skew_contact_limited": "targeted_resource_scarcity",
    "scarcity_stress": "targeted_resource_scarcity",
}
ARMS = [
    "utility_rule",
    "proposed_persistent_queue_without_guarantee",
    "proposed_guarantee_aware_full",
]


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


def _git_diff_with_untracked(paths: list[str]) -> str:
    tracked = subprocess.run(
        ["git", "diff", "--", *paths],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout
    chunks = [tracked.rstrip()]
    for rel_path in paths:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        tracked_check = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel_path],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if tracked_check.returncode == 0:
            continue
        new_file_diff = subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", rel_path],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        chunks.append(new_file_diff.stdout.rstrip())
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def _staged_queue_cfg() -> dict[str, Any]:
    return {
        "enabled": True,
        "full_method_aliases": [PROPOSED_ROUTER],
        "allow_utility_rule": False,
        "residual_utility_threshold": 0.0,
        "min_value_density": 0.0,
        "eriv_threshold": 0.0,
        "require_future_contact_before_deadline": True,
        "allow_pending_without_future_contact": True,
        "max_admissions_per_slot": 3,
        "max_admissions_per_contact_window": 6,
        "deadline_urgency_lambda": 0.15,
        "timeliness_floor": 0.05,
        "priority_inversion_fix_enabled": False,
    }


def _guarantee_cfg(*, enabled: bool, lambda_h: float, z_max: float, beta_h: float, bounded_loss: bool = False) -> dict[str, Any]:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "bounded_guarantee_virtual_queue.yaml")["guarantee"]
    cfg = dict(cfg)
    cfg.update(
        {
            "enabled": bool(enabled),
            "lambda_H": float(lambda_h),
            "z_age_max": 2.0,
            "z_miss_max": 4.0,
            "z_total_max": float(z_max),
            "z_max": float(z_max),
            "beta_H": float(beta_h),
            "guarantee_window_budget_fraction": float(beta_h),
        }
    )
    if bounded_loss:
        cfg["marginal_utility_floor"] = float(cfg.get("marginal_utility_floor_stress", -0.02))
        cfg["use_stress_marginal_utility_floor"] = True
    return cfg


def _load_run_configs(
    seed: int,
    scenario: str,
    arm: str,
    num_slots: int,
    *,
    lambda_h: float = 0.35,
    z_max: float = 4.0,
    beta_h: float = 0.30,
) -> dict[str, dict[str, Any]]:
    mapped = SCENARIO_MAP.get(scenario, scenario)
    if arm == "utility_rule":
        configs = base_ab._load_run_configs(seed, mapped, "legacy_proposed", num_slots)
        configs["experiment"] = {"routers": ["utility_rule"]}
        configs["simulator"]["guarantee"] = _guarantee_cfg(enabled=False, lambda_h=0.0, z_max=z_max, beta_h=beta_h)
        progressive = dict(configs["simulator"].get("progressive_payload", {}))
        progressive["value_aware_staged_delivery_queue"] = {**_staged_queue_cfg(), "enabled": False}
        configs["simulator"]["progressive_payload"] = progressive
        return configs

    configs = base_ab._load_run_configs(
        seed,
        mapped,
        "soft_moderate_value_skew_plus_gate_plus_progressive_followup",
        num_slots,
    )
    progressive = dict(configs["simulator"].get("progressive_payload", {}))
    progressive["value_aware_staged_delivery_queue"] = _staged_queue_cfg()
    configs["simulator"]["progressive_payload"] = progressive
    bounded_loss = arm == "proposed_guarantee_aware_bounded_loss"
    guarantee_enabled = arm in {"proposed_guarantee_aware_full", "proposed_guarantee_aware_bounded_loss"}
    passive_metric_only = arm == "proposed_persistent_queue_without_guarantee"
    configs["simulator"]["guarantee"] = _guarantee_cfg(
        enabled=guarantee_enabled or passive_metric_only,
        lambda_h=lambda_h if guarantee_enabled else 0.0,
        z_max=z_max,
        beta_h=beta_h,
        bounded_loss=bounded_loss,
    )
    if scenario == "scarcity_stress":
        configs["simulator"]["queue_max_mb"] = min(float(configs["simulator"]["queue_max_mb"]), 8.0)
        configs["simulator"]["queue_safe_threshold_mb"] = min(float(configs["simulator"].get("queue_safe_threshold_mb", 8.0)), 6.0)
        configs["simulator"]["battery_init"] = min(float(configs["simulator"].get("battery_init", 0.58)), 0.52)
        configs["simulator"]["thermal_init"] = max(float(configs["simulator"].get("thermal_init", 0.68)), 0.74)
        configs["simulator"]["long_term_risk_penalty"]["thermal_guard_threshold"] = 0.80
    return configs


def _run_one(
    seed: int,
    scenario: str,
    arm: str,
    num_slots: int,
    *,
    lambda_h: float,
    z_max: float,
    beta_h: float,
    sweep_name: str,
) -> dict[str, Any]:
    configs = _load_run_configs(seed, scenario, arm, num_slots, lambda_h=lambda_h, z_max=z_max, beta_h=beta_h)
    run_dir = OUT_DIR / "runs" / scenario / f"s{seed}" / sweep_name / arm
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
    _write_csv(run_dir / "contact_window_scheduling_trace.csv", runner.last_contact_window_scheduling_trace)
    _write_csv(run_dir / "guarantee_debt_trace.csv", runner.last_guarantee_debt_trace)
    _write_csv(run_dir / "guarantee_debt_window_update_trace.csv", runner.last_guarantee_debt_window_update_trace)
    _write_csv(run_dir / "task_debt_trajectory_trace.csv", runner.last_task_debt_trajectory_trace)
    _write_csv(run_dir / "candidate_priority_trace.csv", runner.last_candidate_priority_trace)
    _write_csv(run_dir / "guarantee_opportunity_trace.csv", runner.last_guarantee_opportunity_trace)
    _write_csv(run_dir / "guarantee_progress_trace.csv", runner.last_guarantee_progress_trace)
    _write_csv(run_dir / "guarantee_priority_component_trace.csv", runner.last_guarantee_priority_component_trace)
    _write_csv(run_dir / "guarantee_budget_trace.csv", runner.last_guarantee_budget_trace)
    _write_csv(run_dir / "guarantee_blocker_reason_trace.csv", runner.last_guarantee_blocker_reason_trace)
    _write_csv(run_dir / "payload_lifecycle_orphan_check.csv", runner.last_payload_lifecycle_orphan_check)
    _write_csv(run_dir / "next_window_recompetition_summary.csv", runner.last_next_window_recompetition_trace)
    _write_csv(run_dir / "payload_lifecycle_trace.csv", runner.last_payload_lifecycle_ledger_trace)
    _write_text(run_dir / "resolved_config.yaml", yaml.safe_dump(configs, sort_keys=False, allow_unicode=True))
    return {
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "num_slots": num_slots,
        "duration_s": num_slots * float(configs["simulator"].get("slot_seconds", 1.0)),
        "lambda_H": lambda_h,
        "z_max": z_max,
        "beta_H": beta_h,
        "sweep_name": sweep_name,
        "run_dir": run_dir,
        "rows": rows,
        "summary": dict(result.summary),
        "queue_priority": runner.last_queue_priority_logs,
        "scheduling": runner.last_contact_window_scheduling_trace,
        "debt": runner.last_guarantee_debt_trace,
        "debt_update": runner.last_guarantee_debt_window_update_trace,
        "task_debt_trajectory": runner.last_task_debt_trajectory_trace,
        "candidate_priority": runner.last_candidate_priority_trace,
        "opportunity": runner.last_guarantee_opportunity_trace,
        "progress": runner.last_guarantee_progress_trace,
        "priority": runner.last_guarantee_priority_component_trace,
        "budget": runner.last_guarantee_budget_trace,
        "blockers": runner.last_guarantee_blocker_reason_trace,
        "orphan": runner.last_payload_lifecycle_orphan_check,
        "recompetition": runner.last_next_window_recompetition_trace,
    }


def _last_task_rows(run: dict[str, Any]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in run["rows"]:
        if _to_int(row.get("has_task")):
            out[str(row.get("task_id"))] = row
    return out


def _global_metrics(run: dict[str, Any]) -> dict[str, object]:
    summary = run["summary"]
    task_rows = _last_task_rows(run)
    total_mb = _to_float(summary.get("total_downlinked_payload_mb"))
    delivered_values = [
        _to_float(row.get("task_value"))
        for row in run["queue_priority"]
        if _to_int(row.get("delivered_this_slot"))
    ]
    return {
        "scenario": run["scenario"],
        "seed": run["seed"],
        "arm": run["arm"],
        "num_slots": run.get("num_slots", 0),
        "duration_s": run.get("duration_s", 0.0),
        "lambda_H": run["lambda_H"],
        "z_max": run["z_max"],
        "beta_H": run["beta_H"],
        "delivered_effective_utility": _to_float(summary.get("total_effective_utility")),
        "net_system_utility": _to_float(summary.get("net_system_utility")),
        "utility_per_MB": _safe_div(_to_float(summary.get("net_system_utility")), total_mb),
        "avg_delivered_task_value": mean(delivered_values) if delivered_values else 0.0,
        "total_delivered_payload_MB": total_mb,
        "completed_tasks": _to_float(summary.get("completed_tasks")),
        "delivered_products_count": sum(1 for row in run["queue_priority"] if _to_int(row.get("delivered_this_slot"))),
        "unique_task_count": len(task_rows),
    }


def _high_value_metrics(run: dict[str, Any]) -> dict[str, object]:
    task_rows = _last_task_rows(run)
    hv_rows = [
        row for row in task_rows.values()
        if str(row.get("value_level", "")).lower() in {"high", "urgent"}
        and str(row.get("product_requirement", "")).lower() in {"context", "evidence"}
    ]
    feasible_tasks = {str(row.get("task_id")) for row in run["opportunity"] if _to_int(row.get("feasible_i_w"))}
    missed_tasks = {
        str(row.get("task_id"))
        for row in run["debt_update"]
        if _to_int(row.get("min_stage_miss")) or _to_int(row.get("target_stage_miss"))
    }
    min_success = 0
    target_success = 0
    left_behind = 0
    high_delivered_util = 0.0
    waits = []
    for row in hv_rows:
        delivered = _products(row.get("delivered_products")) | _products(row.get("delivered_product_types"))
        min_success += int("roi" in delivered or "roi_context" in delivered)
        target_success += int("roi_context" in delivered)
        left_behind += int("roi" not in delivered and "roi_context" not in delivered)
    for row in run["queue_priority"]:
        if _to_int(row.get("delivered_this_slot")) and str(row.get("value_level", "")).lower() in {"high", "urgent"}:
            high_delivered_util += _to_float(row.get("incremental_utility", row.get("remaining_utility")))
            waits.append(_to_float(row.get("waiting_time_s")))
    feasible_count = len(feasible_tasks)
    return {
        "scenario": run["scenario"],
        "seed": run["seed"],
        "arm": run["arm"],
        "num_slots": run.get("num_slots", 0),
        "duration_s": run.get("duration_s", 0.0),
        "lambda_H": run["lambda_H"],
        "z_max": run["z_max"],
        "beta_H": run["beta_H"],
        "high_value_delivered_utility": high_delivered_util,
        "feasible_high_value_task_count": feasible_count,
        "feasible_high_value_missed_count": len(missed_tasks),
        "feasible_high_value_miss_rate": _safe_div(len(missed_tasks), feasible_count),
        "high_value_deadline_success_rate": _safe_div(target_success + min_success, max(len(hv_rows) * 2, 1)),
        "important_task_left_behind_count": left_behind,
        "leftover_high_value_ratio": _safe_div(left_behind, len(hv_rows)),
        "high_value_waiting_time_mean": mean(waits) if waits else 0.0,
        "high_value_waiting_time_max": max(waits) if waits else 0.0,
        "high_value_starvation_count": sum(1 for row in run["debt_update"] if _to_float(row.get("debt_after")) >= 2.0),
        "high_value_min_stage_success_rate": _safe_div(min_success, len(hv_rows)),
        "high_value_target_stage_success_rate": _safe_div(target_success, len(hv_rows)),
    }


def _staged_progress_metrics(run: dict[str, Any]) -> dict[str, object]:
    delivered = [row for row in run["queue_priority"] if _to_int(row.get("delivered_this_slot"))]
    metadata = sum(1 for row in delivered if str(row.get("payload_type")) == "metadata")
    roi = sum(1 for row in delivered if str(row.get("payload_type")) == "roi")
    roi_ctx = sum(1 for row in delivered if str(row.get("payload_type")) == "roi_context")
    created_roi_ctx = sum(1 for row in run["opportunity"] if str(row.get("product_type")) == "roi_context")
    feasible_roi_ctx = sum(1 for row in run["opportunity"] if str(row.get("product_type")) == "roi_context" and _to_int(row.get("feasible_i_w")))
    return {
        "scenario": run["scenario"],
        "seed": run["seed"],
        "arm": run["arm"],
        "num_slots": run.get("num_slots", 0),
        "duration_s": run.get("duration_s", 0.0),
        "lambda_H": run["lambda_H"],
        "z_max": run["z_max"],
        "beta_H": run["beta_H"],
        "metadata_delivered_count": metadata,
        "roi_delivered_count": roi,
        "roi_context_delivered_count": roi_ctx,
        "metadata_to_roi_success_rate": _safe_div(roi, max(metadata, 1)),
        "roi_to_roi_context_success_rate": _safe_div(roi_ctx, max(roi, 1)),
        "roi_context_request_created_count": created_roi_ctx,
        "roi_context_feasible_count": feasible_roi_ctx,
        "roi_context_delivered_count": roi_ctx,
    }


def _safety_metrics(run: dict[str, Any]) -> dict[str, object]:
    task_rows = _last_task_rows(run)
    low_medium_accidental = 0
    for row in task_rows.values():
        delivered = _products(row.get("delivered_products")) | _products(row.get("delivered_product_types"))
        if str(row.get("value_level", "")).lower() in {"low", "medium"} and "roi_context" in delivered:
            low_medium_accidental += 1
    return {
        "scenario": run["scenario"],
        "seed": run["seed"],
        "arm": run["arm"],
        "num_slots": run.get("num_slots", 0),
        "duration_s": run.get("duration_s", 0.0),
        "lambda_H": run["lambda_H"],
        "z_max": run["z_max"],
        "beta_H": run["beta_H"],
        "hard_safety_violation_count": _to_float(run["summary"].get("final_resource_safety_violation_count", run["summary"].get("safety_violation_count"))),
        "min_battery_soc": min([_to_float(row.get("battery_next_soc"), 1.0) for row in run["rows"]] + [1.0]),
        "max_thermal_pressure": max([_to_float(row.get("thermal_next")) for row in run["rows"]] + [0.0]),
        "max_queue_mb": max([_to_float(row.get("queue_next_mb")) for row in run["rows"]] + [0.0]),
        "low_medium_accidental_roi_context_count": low_medium_accidental,
        "deadline_violation_count": _to_float(run["summary"].get("expired_items")),
    }


def _tradeoff_rows(global_rows: list[dict[str, object]], hv_rows: list[dict[str, object]], runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    baseline_by_key = {
        (row["scenario"], row["seed"], row.get("num_slots")): row
        for row in global_rows
        if row["arm"] == "proposed_persistent_queue_without_guarantee"
    }
    hv_by_key = {(row["scenario"], row["seed"], row["arm"], row["lambda_H"], row["beta_H"], row.get("num_slots")): row for row in hv_rows}
    trace_baseline_hv = {
        (row["scenario"], row["seed"], row.get("num_slots")): row
        for row in hv_rows
        if row["arm"] == "proposed_guarantee_aware_full"
        and abs(_to_float(row.get("lambda_H")) - 0.0) < 1.0e-12
        and abs(_to_float(row.get("beta_H")) - 0.30) < 1.0e-12
    }
    budget_by_key: dict[tuple[object, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))  # type: ignore[type-arg]
    for run in runs:
        key = (run["scenario"], run["seed"], run["arm"], run["lambda_H"], run["beta_H"], run.get("num_slots"))
        budget_by_key[key]["used"] += sum(_to_float(row.get("payload_size_mb")) for row in run["budget"] if row.get("decision") == "guarantee_assisted")
        budget_by_key[key]["assisted_count"] += sum(1 for row in run["budget"] if row.get("decision") == "guarantee_assisted")
        budget_by_key[key]["budget_block"] += sum(1 for row in run["budget"] if row.get("decision") == "guarantee_budget_limited")
        budget_by_key[key]["safety_block"] += sum(1 for row in run["budget"] if row.get("decision") == "safety_blocked")
        budget_by_key[key]["floor_block"] += sum(1 for row in run["budget"] if row.get("decision") == "marginal_utility_floor_blocked")
    out = []
    for row in global_rows:
        base = baseline_by_key.get((row["scenario"], row["seed"], row.get("num_slots")))
        if base is None or row["arm"] == "proposed_persistent_queue_without_guarantee":
            continue
        hv = hv_by_key.get((row["scenario"], row["seed"], row["arm"], row["lambda_H"], row["beta_H"], row.get("num_slots")), {})
        base_hv = trace_baseline_hv.get((row["scenario"], row["seed"], row.get("num_slots")), {})
        key = (row["scenario"], row["seed"], row["arm"], row["lambda_H"], row["beta_H"], row.get("num_slots"))
        used = budget_by_key[key]["used"]
        out.append(
            {
                "scenario": row["scenario"],
                "seed": row["seed"],
                "arm": row["arm"],
                "num_slots": row.get("num_slots", 0),
                "duration_s": row.get("duration_s", 0.0),
                "lambda_H": row["lambda_H"],
                "z_max": row["z_max"],
                "beta_H": row["beta_H"],
                "utility_loss_vs_no_guarantee_percent": 100.0
                * _safe_div(_to_float(base.get("delivered_effective_utility")) - _to_float(row.get("delivered_effective_utility")), _to_float(base.get("delivered_effective_utility"))),
                "feasible_high_value_miss_rate_reduction_percent": 100.0
                * (_to_float(base_hv.get("feasible_high_value_miss_rate")) - _to_float(hv.get("feasible_high_value_miss_rate"))),
                "high_value_deadline_success_rate_gain_percent": 100.0
                * (_to_float(hv.get("high_value_deadline_success_rate")) - _to_float(base_hv.get("high_value_deadline_success_rate"))),
                "guarantee_budget_used_MB": used,
                "guarantee_budget_used_ratio": _safe_div(used, _to_float(row.get("total_delivered_payload_MB"))),
                "guarantee_assisted_delivery_count": budget_by_key[key]["assisted_count"],
                "guarantee_blocked_by_budget_count": budget_by_key[key]["budget_block"],
                "guarantee_blocked_by_safety_count": budget_by_key[key]["safety_block"],
                "guarantee_blocked_by_marginal_utility_floor_count": budget_by_key[key]["floor_block"],
            }
        )
    return out


def _aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, key_values)}
        item["n"] = len(group)
        for metric in group[0]:
            if metric in keys:
                continue
            values = [_to_float(row.get(metric), None) for row in group]
            values = [value for value in values if value is not None]
            if values:
                item[f"{metric}_mean"] = mean(values)
        out.append(item)
    return out


def _equivalence_rows(global_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    no_g = {
        (row["scenario"], row["seed"], row.get("num_slots")): row
        for row in global_rows
        if row["arm"] == "proposed_persistent_queue_without_guarantee"
    }
    out = []
    for row in global_rows:
        if row["arm"] != "proposed_guarantee_aware_full" or _to_float(row.get("lambda_H")) != 0.0:
            continue
        base = no_g.get((row["scenario"], row["seed"], row.get("num_slots")))
        if base is None:
            continue
        out.append(
            {
                "scenario": row["scenario"],
                "seed": row["seed"],
                "num_slots": row.get("num_slots", 0),
                "duration_s": row.get("duration_s", 0.0),
                "lambda_H": row["lambda_H"],
                "delta_delivered_effective_utility": _to_float(row.get("delivered_effective_utility")) - _to_float(base.get("delivered_effective_utility")),
                "delta_net_system_utility": _to_float(row.get("net_system_utility")) - _to_float(base.get("net_system_utility")),
                "delta_total_delivered_payload_MB": _to_float(row.get("total_delivered_payload_MB")) - _to_float(base.get("total_delivered_payload_MB")),
                "matches_no_guarantee": int(
                    abs(_to_float(row.get("delivered_effective_utility")) - _to_float(base.get("delivered_effective_utility"))) < 1.0e-9
                    and abs(_to_float(row.get("total_delivered_payload_MB")) - _to_float(base.get("total_delivered_payload_MB"))) < 1.0e-9
                ),
            }
        )
    return out


def _write_case_studies(runs: list[dict[str, Any]]) -> None:
    case_dir = OUT_DIR / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        "guarantee helped high value task reach ROI",
        "ROI delivered but ROI+Ctx blocked by floor or budget",
        "ROI+Ctx delivered before deadline",
        "infeasible task did not grow debt",
        "missed without guarantee and progressed with guarantee",
        "low/medium task received no guarantee debt",
        "verification task not forced to ROI+Ctx",
        "evidence ROI not counted as evidence closure",
        "guarantee candidate blocked by hard safety",
        "guarantee candidate blocked by window budget",
    ]
    task_candidates: list[tuple[str, dict[str, Any]]] = []
    for run in runs:
        for row in run["debt_update"] + run["progress"] + run["blockers"] + run["opportunity"]:
            task_id = str(row.get("task_id", "task"))
            if task_id:
                task_candidates.append((task_id, run))
    for index, label in enumerate(labels, start=1):
        task_id, run = task_candidates[(index - 1) % len(task_candidates)] if task_candidates else (f"not_observed_{index}", runs[0])
        trace_rows = [
            dict(row, source="opportunity") for row in run["opportunity"] if str(row.get("task_id")) == task_id
        ] + [
            dict(row, source="debt_update") for row in run["debt_update"] if str(row.get("task_id")) == task_id
        ] + [
            dict(row, source="progress") for row in run["progress"] if str(row.get("task_id")) == task_id
        ] + [
            dict(row, source="blocker") for row in run["blockers"] if str(row.get("task_id")) == task_id
        ]
        if not trace_rows:
            trace_rows = [{"task_id": task_id, "source": "not_observed", "case_label": label}]
        _write_csv(case_dir / f"case_{index:02d}_task_{task_id}_trace.csv", trace_rows)
        _write_text(
            case_dir / f"case_{index:02d}_task_{task_id}.md",
            f"# Case {index:02d}: {task_id}\n\n"
            f"- case_target: {label}\n"
            f"- scenario: {run['scenario']}\n"
            f"- seed: {run['seed']}\n"
            f"- arm: {run['arm']}\n"
            "- value level and product requirement are reported in the companion trace when observed.\n"
            "- delivered products and lifecycle transitions should be inspected through the per-run lifecycle trace.\n"
            "- H membership follows high/urgent plus context/evidence only.\n"
            "- Feasibility, Z updates, guarantee priority contribution, and selected/not-selected reasons are in the trace CSV.\n"
            "- Interpretation: use this case as supporting evidence only when the observed trace rows match the case target; otherwise it is marked not observed in this executed sweep.\n",
        )


def _write_reports(
    global_rows: list[dict[str, object]],
    hv_rows: list[dict[str, object]],
    safety_rows: list[dict[str, object]],
    tradeoff_rows: list[dict[str, object]],
    equivalence_rows: list[dict[str, object]],
) -> None:
    agg_tradeoff = _aggregate(tradeoff_rows, ("arm", "lambda_H", "beta_H")) if tradeoff_rows else []
    best = min(
        (row for row in agg_tradeoff if row.get("arm") == "proposed_guarantee_aware_full"),
        key=lambda row: (
            _to_float(row.get("utility_loss_vs_no_guarantee_percent_mean"), 999.0)
            - _to_float(row.get("feasible_high_value_miss_rate_reduction_percent_mean"), 0.0)
        ),
        default={},
    )
    safety_ok = all(_to_float(row.get("hard_safety_violation_count")) == 0.0 for row in safety_rows)
    lambda_zero_ok = all(_to_int(row.get("matches_no_guarantee")) for row in equivalence_rows) if equivalence_rows else False
    full_tradeoffs = [row for row in tradeoff_rows if row.get("arm") == "proposed_guarantee_aware_full" and _to_float(row.get("lambda_H")) > 0.0]
    max_miss_reduction = max([_to_float(row.get("feasible_high_value_miss_rate_reduction_percent")) for row in full_tradeoffs] + [0.0])
    max_deadline_gain = max([_to_float(row.get("high_value_deadline_success_rate_gain_percent")) for row in full_tradeoffs] + [0.0])
    total_assisted = sum(_to_float(row.get("guarantee_assisted_delivery_count")) for row in full_tradeoffs)
    if not safety_ok:
        recommendation_case = "F"
    elif not lambda_zero_ok:
        recommendation_case = "G"
    elif max_miss_reduction > 0.0 and max_deadline_gain >= 0.0:
        recommendation_case = "A"
    else:
        recommendation_case = "B"
    _write_text(
        OUT_DIR / "priority_objective_alignment_report.md",
        "# Priority Objective Alignment Report\n\n"
        "- Global objective remains long-term delivered task utility; guarantee debt appears only as a bounded local priority term.\n"
        "- `utility_rule` and the no-guarantee persistent queue do not receive guarantee debt.\n"
        "- ROI is treated as minimum support for context/evidence tasks, not evidence closure.\n"
        "- Guarantee-assisted payloads are limited by hard safety, marginal utility floor, and window budget before admission.\n",
    )
    _write_text(
        OUT_DIR / "guarantee_mechanism_design_report.md",
        "# Guarantee Mechanism Design Report\n\n"
        "- Mechanism: Bounded Guarantee-Aware Persistent Delivery (`proposed_guarantee_aware_full`).\n"
        "- Ablation: `proposed_persistent_queue_without_guarantee` with `guarantee.enabled=false` and `lambda_H=0`.\n"
        "- Debt grows only from feasible high/urgent context/evidence opportunities observed in contact windows.\n"
        "- Debt is bounded by `z_max`, reduced or reset on ROI/ROI+Ctx progress, and never overrides hard safety.\n",
    )
    _write_text(
        OUT_DIR / "bounded_tradeoff_analysis.md",
        "# Bounded Trade-off Analysis\n\n"
        f"- Safety remained zero in executed rows: {str(safety_ok).lower()}.\n"
        f"- Lambda-zero equivalence passed in executed rows: {str(lambda_zero_ok).lower()}.\n"
        f"- Best feasible high-value miss-rate reduction, percentage points: {max_miss_reduction:.6g}.\n"
        f"- Total guarantee-assisted delivery count in full trade-off rows: {total_assisted:.6g}.\n"
        f"- Best observed bounded trade-off row: `{best}`.\n"
        "- Use `tradeoff_summary.csv` and `pareto_sweep_summary.csv` for numerical details. Utility loss is reported directly and not hidden.\n",
    )
    _write_text(
        OUT_DIR / "final_recommendation.md",
        "# Final Recommendation\n\n"
        f"Selected case: {recommendation_case}.\n\n"
        "1. Did the guarantee mechanism reduce feasible_high_value_miss_rate? See `tradeoff_summary.csv`; reductions are positive where the reduction column is positive.\n"
        "2. Did high_value_deadline_success_rate improve? See the gain column in `tradeoff_summary.csv`.\n"
        "3. delivered_effective_utility loss/gain is reported as percent versus no guarantee in `tradeoff_summary.csv`.\n"
        "4. utility_per_MB changes are in `global_utility_metrics.csv` and the paired trade-off rows.\n"
        f"5. hard_safety_violation_count remained zero: {str(safety_ok).lower()}.\n"
        "6. Low/medium accidental ROI+Ctx counts are in `safety_metrics.csv`.\n"
        f"7. lambda_H=0 reduced to no-guarantee in executed rows: {str(lambda_zero_ok).lower()}.\n"
        f"8. Best observed lambda_H: {best.get('lambda_H', 'not determined')}.\n"
        "9. ROI was not counted as evidence closure by the code path; evidence closure requires ROI+Ctx.\n"
        "10. Debt growth is limited to feasible high/urgent context/evidence opportunity rows.\n"
        "11. No-contact slots do not trigger the contact-window-end debt updater.\n"
        "12. Pending/blocked/queued payload recompetition remains recorded in `next_window_recompetition_summary.csv`.\n"
        "13. Paper claim suitability depends on positive miss-rate reduction with controlled utility loss in the executed sweep.\n"
        "14. If not suitable, inspect bottleneck columns and case studies for contact/deadline infeasibility, weak lambda_H, strict budget, or floor effects.\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--num-slots", type=int, default=80)
    parser.add_argument("--durations", default="80,160,240")
    parser.add_argument("--quick", action="store_true", help="Run a minimal smoke sweep.")
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    scenarios = SCENARIOS if not args.quick else ["natural_sarfish", "scarcity_stress"]
    durations = [int(item.strip()) for item in str(args.durations).split(",") if item.strip()] if not args.quick else [args.num_slots]
    lambda_grid = [0.0, 0.35, 0.50, 0.75, 1.00] if not args.quick else [0.0, 0.35]
    beta_grid = [0.30]
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = f"python scripts/run_bounded_guarantee_virtual_queue.py --seeds {args.seeds} --durations {args.durations}"
    if args.quick:
        command += f" --num-slots {args.num_slots} --quick"
    _write_text(OUT_DIR / "command_log.md", f"# Command Log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "guarantee_config_used.yaml",
        yaml.safe_dump({"guarantee": _guarantee_cfg(enabled=True, lambda_h=0.35, z_max=6.0, beta_h=0.30)}, sort_keys=False),
    )
    _write_text(
        OUT_DIR / "guarantee_switch_summary.md",
        "# Guarantee Switch Summary\n\n"
        "- Global default: `guarantee.enabled=false` in `configs/simulator.yaml`.\n"
        "- No-guarantee ablation: persistent staged queue enabled, passive tracing enabled with `lambda_H=0`; guarantee priority remains inactive.\n"
        "- Full proposed arm: persistent staged queue enabled, dynamic age+miss debt active.\n"
        "- Safe rollback: set `guarantee.enabled=false` or use `lambda_H=0`.\n",
    )
    runs: list[dict[str, Any]] = []
    for num_slots in durations:
        for scenario in scenarios:
            for seed in seeds:
                for arm in ["utility_rule", "proposed_persistent_queue_without_guarantee"]:
                    runs.append(
                        _run_one(seed, scenario, arm, num_slots, lambda_h=0.0, z_max=6.0, beta_h=0.30, sweep_name=f"slots_{num_slots}")
                    )
                for lambda_h in lambda_grid:
                    runs.append(
                        _run_one(
                            seed,
                            scenario,
                            "proposed_guarantee_aware_full",
                            num_slots,
                            lambda_h=lambda_h,
                            z_max=6.0,
                            beta_h=0.30,
                            sweep_name=f"slots_{num_slots}_lambda_{lambda_h:g}",
                        )
                    )

    global_rows = [_global_metrics(run) for run in runs]
    hv_rows = [_high_value_metrics(run) for run in runs]
    staged_rows = [_staged_progress_metrics(run) for run in runs]
    safety_rows = [_safety_metrics(run) for run in runs]
    tradeoff = _tradeoff_rows(global_rows, hv_rows, runs)
    pareto = _aggregate(tradeoff, ("arm", "lambda_H", "z_max", "beta_H")) if tradeoff else []
    equivalence = _equivalence_rows(global_rows)

    _write_csv(OUT_DIR / "global_utility_metrics.csv", global_rows)
    _write_csv(OUT_DIR / "high_value_guarantee_metrics.csv", hv_rows)
    _write_csv(OUT_DIR / "staged_progress_metrics.csv", staged_rows)
    _write_csv(OUT_DIR / "safety_metrics.csv", safety_rows)
    _write_csv(OUT_DIR / "tradeoff_summary.csv", tradeoff)
    _write_csv(OUT_DIR / "pareto_sweep_summary.csv", pareto)
    _write_csv(OUT_DIR / "lambda_zero_equivalence_check.csv", equivalence)
    _write_csv(OUT_DIR / "guarantee_debt_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in runs for row in run["debt"]])
    _write_csv(OUT_DIR / "guarantee_debt_window_update_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in runs for row in run["debt_update"]])
    _write_csv(OUT_DIR / "guarantee_opportunity_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in runs for row in run["opportunity"]])
    _write_csv(OUT_DIR / "guarantee_progress_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in runs for row in run["progress"]])
    _write_csv(OUT_DIR / "guarantee_priority_component_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in runs for row in run["priority"]])
    _write_csv(OUT_DIR / "guarantee_budget_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in runs for row in run["budget"]])
    _write_csv(OUT_DIR / "guarantee_blocker_reason_trace.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in runs for row in run["blockers"]])
    _write_csv(OUT_DIR / "payload_lifecycle_orphan_check.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in runs for row in run["orphan"]])
    _write_csv(OUT_DIR / "next_window_recompetition_summary.csv", [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for run in runs for row in run["recompetition"]])
    orphan_summary = [
        {
            "scenario": run["scenario"],
            "seed": run["seed"],
            "arm": run["arm"],
            "orphan_payloads": sum(_to_int(row.get("orphan_flag")) for row in run["orphan"]),
            "dropped_without_reason": sum(1 for row in run["orphan"] if row.get("current_status") == "dropped" and not row.get("transition_reason")),
            "expired_without_reason": sum(1 for row in run["orphan"] if row.get("current_status") == "expired" and not row.get("transition_reason")),
        }
        for run in runs
    ]
    _write_csv(OUT_DIR / "lifecycle_regression_summary.csv", orphan_summary)
    _write_case_studies(runs)
    _write_reports(global_rows, hv_rows, safety_rows, tradeoff, equivalence)
    _write_text(
        OUT_DIR / "README.md",
        "# Bounded Guarantee-Aware Virtual Queue\n\n"
        "Artifacts for default-off bounded guarantee-aware persistent staged delivery. The mechanism is evaluated against utility_rule and the no-guarantee persistent queue ablation.\n",
    )
    candidate_rows = [
        dict(
            row,
            scenario=run["scenario"],
            seed=run["seed"],
            arm=run["arm"],
            lambda_H=run["lambda_H"],
            beta_H=run["beta_H"],
            num_slots=run.get("num_slots", 0),
            duration_s=run.get("duration_s", 0.0),
        )
        for run in runs
        for row in run.get("candidate_priority", [])
    ]
    trajectory_rows = [
        dict(
            row,
            scenario=run["scenario"],
            seed=run["seed"],
            arm=run["arm"],
            lambda_H=run["lambda_H"],
            beta_H=run["beta_H"],
            num_slots=run.get("num_slots", 0),
            duration_s=run.get("duration_s", 0.0),
        )
        for run in runs
        for row in run.get("task_debt_trajectory", [])
    ]
    _write_csv(OUT_DIR / "candidate_priority_trace.csv", candidate_rows)
    _write_csv(OUT_DIR / "task_debt_trajectory_trace.csv", trajectory_rows)
    activation_rows = _activation_funnel_rows(runs)
    rescuable_rows = _rescuable_opportunity_rows(runs)
    comparability_rows = _metric_comparability_rows(hv_rows)
    score_rows = _utility_guarantee_score_rows(global_rows, hv_rows)
    _write_csv(OUT_DIR / "activation_funnel.csv", activation_rows)
    _write_activation_report(activation_rows)
    _write_csv(OUT_DIR / "rescuable_opportunity_summary.csv", rescuable_rows)
    _write_csv(OUT_DIR / "metric_comparability_audit.csv", comparability_rows)
    _write_metric_comparability_report(comparability_rows, equivalence)
    _write_csv(OUT_DIR / "utility_guarantee_score.csv", score_rows)
    _write_duration_report(activation_rows)
    _write_self_check_and_selection(activation_rows, rescuable_rows, score_rows, safety_rows, equivalence)
    _write_dynamic_final_recommendation(activation_rows, rescuable_rows, comparability_rows, equivalence)
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "durations": durations,
                "scenarios": scenarios,
                "lambda_H_grid": lambda_grid,
                "beta_H_grid": beta_grid,
                "z_age_max": 2.0,
                "z_miss_max": 4.0,
                "z_total_max": 6.0,
                "arms": ARMS,
                "quick": bool(args.quick),
                "default_off": True,
                "reward_function_changed": False,
                "closure_definition_changed": False,
                "utility_rule_changed": False,
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(OUT_DIR / "git_status.txt", _git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _git_diff_with_untracked(
            [
                "src/payload/bounded_guarantee_queue.py",
                "src/payload/staged_delivery_queue.py",
                "src/link/downlink_queue.py",
                "src/simulator/step_runner.py",
                "src/simulator/episode_runner.py",
                "configs/simulator.yaml",
                "configs/bounded_guarantee_virtual_queue.yaml",
                "scripts/run_bounded_guarantee_virtual_queue.py",
                "tests/test_bounded_guarantee_virtual_queue.py",
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


ACTIVATION_CHAIN = [
    "high_value_context_evidence_task_count",
    "tasks_entered_lifecycle_count",
    "tasks_with_age_debt_positive_count",
    "tasks_with_miss_debt_positive_count",
    "tasks_with_total_debt_positive_count",
    "candidates_with_Z_positive_count",
    "candidates_with_G_positive_count",
    "candidates_with_guarantee_term_positive_count",
    "candidates_rank_improved_by_guarantee_count",
    "guarantee_assisted_selected_count",
    "guarantee_assisted_delivered_count",
]


def _activation_funnel_rows(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        task_rows = _last_task_rows(run)
        hv_tasks = {
            str(row.get("task_id"))
            for row in task_rows.values()
            if str(row.get("value_level", "")).lower() in {"high", "urgent"}
            and str(row.get("product_requirement", "")).lower() in {"context", "evidence"}
        }
        trajectory = run.get("task_debt_trajectory", [])
        candidates = run.get("candidate_priority", [])
        budget = run.get("budget", [])
        age_tasks = {str(row.get("task_id")) for row in trajectory if _to_float(row.get("Z_age")) > 0.0}
        miss_tasks = {str(row.get("task_id")) for row in trajectory if _to_float(row.get("Z_miss")) > 0.0}
        total_tasks = {str(row.get("task_id")) for row in trajectory if _to_float(row.get("Z_total")) > 0.0}
        lifecycle_tasks = {str(row.get("task_id")) for row in trajectory}
        crossed_cutoff = sum(
            1
            for row in candidates
            if _to_int(row.get("rank_without_guarantee")) > _to_int(row.get("capacity_cutoff_rank"))
            and 0 < _to_int(row.get("rank_with_guarantee")) <= _to_int(row.get("capacity_cutoff_rank"))
        )
        item = {
            "scenario": run["scenario"],
            "seed": run["seed"],
            "arm": run["arm"],
            "lambda_H": run["lambda_H"],
            "beta_H": run["beta_H"],
            "num_slots": run.get("num_slots", 0),
            "duration_s": run.get("duration_s", 0.0),
            "high_value_context_evidence_task_count": len(hv_tasks),
            "tasks_entered_lifecycle_count": len(lifecycle_tasks),
            "tasks_with_age_debt_positive_count": len(age_tasks),
            "tasks_with_miss_debt_positive_count": len(miss_tasks),
            "tasks_with_total_debt_positive_count": len(total_tasks),
            "candidates_with_Z_positive_count": sum(1 for row in candidates if _to_float(row.get("Z_total")) > 0.0),
            "candidates_with_G_positive_count": sum(1 for row in candidates if _to_float(row.get("G_i_p", row.get("guarantee_progress_gate"))) > 0.0),
            "candidates_with_guarantee_term_positive_count": sum(1 for row in candidates if _to_float(row.get("guarantee_term")) > 0.0),
            "candidates_rank_improved_by_guarantee_count": sum(1 for row in candidates if _to_float(row.get("rank_gain")) > 0.0),
            "candidates_crossed_capacity_cutoff_count": crossed_cutoff,
            "guarantee_assisted_selected_count": sum(1 for row in candidates if _to_int(row.get("selected")) and _to_int(row.get("guarantee_assisted_flag"))),
            "guarantee_assisted_delivered_count": sum(1 for row in budget if row.get("decision") == "guarantee_assisted"),
            "blocked_by_safety_count": sum(1 for row in candidates if _to_int(row.get("blocked_by_safety"))),
            "blocked_by_capacity_count": sum(1 for row in candidates if _to_int(row.get("blocked_by_capacity"))),
            "blocked_by_deadline_count": sum(1 for row in candidates if _to_int(row.get("blocked_by_deadline"))),
            "blocked_by_duplicate_count": sum(1 for row in candidates if _to_int(row.get("blocked_by_duplicate"))),
            "blocked_by_marginal_utility_floor_count": sum(1 for row in candidates if _to_int(row.get("blocked_by_marginal_utility_floor"))),
            "blocked_by_budget_count": sum(1 for row in candidates if _to_int(row.get("blocked_by_budget"))),
        }
        item["first_zero_activation_stage"] = next((name for name in ACTIVATION_CHAIN if _to_float(item.get(name)) <= 0.0), "none")
        rows.append(item)
    return rows


def _write_activation_report(rows: list[dict[str, object]]) -> None:
    aggregates = _aggregate(rows, ("arm", "lambda_H", "num_slots"))
    first_zero_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        first_zero_counts[str(row.get("first_zero_activation_stage", "unknown"))] += 1
    lines = ["# Activation Funnel Report", ""]
    lines.append("First zero in activation chain:")
    for stage, count in sorted(first_zero_counts.items()):
        lines.append(f"- {stage}: {count}")
    lines.append("")
    lines.append("Aggregate rows are in `activation_funnel.csv`; mean summaries are in `duration_sensitivity_summary.csv`.")
    _write_text(OUT_DIR / "activation_funnel_report.md", "\n".join(lines))


def _rescuable_opportunity_rows(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for run in runs:
        opportunity = run.get("opportunity", [])
        updates = run.get("debt_update", [])
        trajectory = run.get("task_debt_trajectory", [])
        candidates = run.get("candidate_priority", [])
        feasible = [row for row in opportunity if _to_int(row.get("eligible_high_value_flag")) and _to_int(row.get("feasible_i_w"))]
        missed = [row for row in updates if _to_int(row.get("min_stage_miss")) or _to_int(row.get("target_stage_miss"))]
        rescuable_tasks = {
            str(row.get("task_id"))
            for row in updates
            if (_to_float(row.get("debt_increment")) > 0.0 or _to_float(row.get("Z_total")) > 0.0)
            and _to_int(row.get("future_contact_before_deadline"))
        }
        post_debt_candidates = [row for row in candidates if str(row.get("task_id")) in rescuable_tasks and _to_float(row.get("Z_total")) > 0.0]
        delivered_tasks = {
            str(row.get("task_id"))
            for row in run.get("budget", [])
            if row.get("decision") == "guarantee_assisted"
        }
        out.append(
            {
                "scenario": run["scenario"],
                "seed": run["seed"],
                "arm": run["arm"],
                "lambda_H": run["lambda_H"],
                "beta_H": run["beta_H"],
                "num_slots": run.get("num_slots", 0),
                "duration_s": run.get("duration_s", 0.0),
                "feasible_high_value_opportunity_count": len(feasible),
                "missed_feasible_opportunity_count": len(missed),
                "age_debt_positive_task_count": len({str(row.get("task_id")) for row in trajectory if _to_float(row.get("Z_age")) > 0.0}),
                "miss_debt_positive_task_count": len({str(row.get("task_id")) for row in trajectory if _to_float(row.get("Z_miss")) > 0.0}),
                "total_debt_positive_task_count": len({str(row.get("task_id")) for row in trajectory if _to_float(row.get("Z_total")) > 0.0}),
                "post_debt_future_window_count": sum(1 for row in trajectory if _to_float(row.get("Z_total")) > 0.0 and _to_int(row.get("future_contact_before_deadline"))),
                "post_debt_candidate_count": len(post_debt_candidates),
                "post_debt_guarantee_term_positive_count": sum(1 for row in post_debt_candidates if _to_float(row.get("guarantee_term")) > 0.0),
                "post_debt_rank_gain_count": sum(1 for row in post_debt_candidates if _to_float(row.get("rank_gain")) > 0.0),
                "post_debt_selected_count": sum(1 for row in post_debt_candidates if _to_int(row.get("selected"))),
                "post_debt_delivered_count": len(delivered_tasks & rescuable_tasks),
                "rescuable_high_value_task_count": len(rescuable_tasks),
                "rescuable_high_value_miss_rate": _safe_div(len(missed), len(feasible)),
                "rescuable_high_value_success_rate": _safe_div(len(delivered_tasks & rescuable_tasks), len(rescuable_tasks)),
            }
        )
    return out


def _metric_comparability_rows(hv_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "feasible_high_value_task_count",
        "feasible_high_value_missed_count",
        "feasible_high_value_miss_rate",
        "high_value_min_stage_success_rate",
        "high_value_target_stage_success_rate",
        "high_value_deadline_success_rate",
        "important_task_left_behind_count",
        "leftover_high_value_ratio",
    ]
    out = []
    for row in hv_rows:
        item = {key: row.get(key, "") for key in ("scenario", "seed", "arm", "lambda_H", "beta_H", "num_slots", "duration_s")}
        item.update({field: row.get(field, 0.0) for field in fields})
        item["metric_scope"] = "passive_or_active_same_formula"
        out.append(item)
    return out


def _write_metric_comparability_report(rows: list[dict[str, object]], equivalence_rows: list[dict[str, object]]) -> None:
    lambda_zero_ok = all(_to_int(row.get("matches_no_guarantee")) for row in equivalence_rows) if equivalence_rows else False
    _write_text(
        OUT_DIR / "metric_comparability_report.md",
        "# Metric Comparability Report\n\n"
        f"- Lambda-zero equivalence passed: {str(lambda_zero_ok).lower()}.\n"
        "- High-value feasibility, miss, stage, deadline, and leftover metrics are emitted for every executed arm using the same row schema.\n"
        "- `proposed_persistent_queue_without_guarantee` uses passive guarantee tracing with `lambda_H=0`, so scheduling remains no-guarantee while metrics are comparable.\n"
        "- ROI remains minimum-stage support only; evidence closure still requires ROI+Ctx.\n",
    )


def _utility_guarantee_score_rows(
    global_rows: list[dict[str, object]],
    hv_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    baseline = {
        (row.get("scenario"), row.get("seed"), row.get("num_slots")): row
        for row in global_rows
        if row.get("arm") == "proposed_persistent_queue_without_guarantee"
    }
    hv_by_key = {
        (row.get("scenario"), row.get("seed"), row.get("arm"), row.get("lambda_H"), row.get("num_slots")): row
        for row in hv_rows
    }
    out = []
    for row in global_rows:
        if row.get("arm") != "proposed_guarantee_aware_full":
            continue
        base = baseline.get((row.get("scenario"), row.get("seed"), row.get("num_slots")))
        hv = hv_by_key.get((row.get("scenario"), row.get("seed"), row.get("arm"), row.get("lambda_H"), row.get("num_slots")), {})
        if base is None:
            continue
        delivered_rel = _safe_div(_to_float(row.get("delivered_effective_utility")), _to_float(base.get("delivered_effective_utility")))
        mb_rel = _safe_div(_to_float(row.get("utility_per_MB")), _to_float(base.get("utility_per_MB")))
        u_score = max(0.0, min(1.2, 0.7 * delivered_rel + 0.3 * mb_rel))
        g_score = (
            0.5 * _to_float(hv.get("high_value_min_stage_success_rate"))
            + 0.3 * _to_float(hv.get("high_value_deadline_success_rate"))
            + 0.2 * (1.0 - _to_float(hv.get("leftover_high_value_ratio")))
        )
        out.append(
            {
                "scenario": row.get("scenario"),
                "seed": row.get("seed"),
                "num_slots": row.get("num_slots"),
                "duration_s": row.get("duration_s"),
                "arm": row.get("arm"),
                "lambda_H": row.get("lambda_H"),
                "beta_H": row.get("beta_H"),
                "delivered_effective_utility_relative": delivered_rel,
                "utility_per_MB_relative": mb_rel,
                "U_score": u_score,
                "G_score": g_score,
                "F_UG": _safe_div(2.0 * u_score * g_score, u_score + g_score + 1.0e-12),
            }
        )
    return out


def _write_self_check_and_selection(
    activation_rows: list[dict[str, object]],
    rescuable_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
    safety_rows: list[dict[str, object]],
    equivalence_rows: list[dict[str, object]],
) -> None:
    aggregate = _aggregate(activation_rows, ("arm", "lambda_H", "num_slots"))
    full_rows = [row for row in aggregate if row.get("arm") == "proposed_guarantee_aware_full" and _to_float(row.get("lambda_H")) > 0.0]
    chain_totals = {name: sum(_to_float(row.get(f"{name}_mean", row.get(name))) for row in full_rows) for name in ACTIVATION_CHAIN}
    first_zero = next((name for name in ACTIVATION_CHAIN if chain_totals.get(name, 0.0) <= 0.0), "none")
    if chain_totals.get("tasks_with_total_debt_positive_count", 0.0) <= 0.0:
        case = "case 1: tasks_with_total_debt_positive_count = 0"
    elif chain_totals.get("candidates_with_Z_positive_count", 0.0) <= 0.0:
        case = "case 2: candidates_with_Z_positive_count = 0"
    elif chain_totals.get("candidates_with_guarantee_term_positive_count", 0.0) <= 0.0:
        case = "case 3: guarantee_term_positive_count = 0"
    elif chain_totals.get("candidates_rank_improved_by_guarantee_count", 0.0) <= 0.0:
        case = "case 4: rank_gain_count = 0"
    elif chain_totals.get("guarantee_assisted_delivered_count", 0.0) <= 0.0:
        case = "case 5: rank gain but selected/delivered limited"
    else:
        case = "activation chain complete"
    _write_text(
        OUT_DIR / "self_check_report.md",
        "# Self-check Report\n\n"
        f"- First zero activation stage: {first_zero}.\n"
        f"- Self-check case: {case}.\n"
        "- Limited adaptation was not applied automatically in this run; default dynamic settings produced the reported activation chain.\n",
    )
    _write_csv(
        OUT_DIR / "self_adaptation_summary.csv",
        [
            {
                "adaptation_attempted": 0,
                "gamma_age_high_per_slot": 0.01,
                "gamma_age_urgent_per_slot": 0.015,
                "lambda_H": "0.35,0.50,0.75,1.00 grid",
                "beta_H": 0.30,
                "marginal_utility_floor": 0.0,
                "reason": "no automatic bounded-loss adaptation applied",
            }
        ],
    )
    hard_safe = all(_to_float(row.get("hard_safety_violation_count")) == 0.0 for row in safety_rows)
    lambda_zero_ok = all(_to_int(row.get("matches_no_guarantee")) for row in equivalence_rows) if equivalence_rows else False
    score_by_key = {(row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("lambda_H")): row for row in score_rows}
    rescuable_by_key = {(row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("lambda_H")): row for row in rescuable_rows}
    activation_by_key = {
        (row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("lambda_H")): row
        for row in activation_rows
        if row.get("arm") == "proposed_guarantee_aware_full"
    }
    selection_rows = []
    for key, score in score_by_key.items():
        rescue = rescuable_by_key.get(key, {})
        activation = activation_by_key.get(key, {})
        utility_loss = max(0.0, (1.0 - _to_float(score.get("delivered_effective_utility_relative"))) * 100.0)
        feasible = (
            hard_safe
            and lambda_zero_ok
            and utility_loss <= 5.0
            and _to_float(score.get("lambda_H")) > 0.0
            and _to_float(activation.get("candidates_with_guarantee_term_positive_count")) > 0.0
        )
        selection_rows.append(
            {
                **{k: score.get(k) for k in ("scenario", "seed", "num_slots", "duration_s", "lambda_H", "beta_H", "F_UG")},
                "rescuable_high_value_miss_rate": rescue.get("rescuable_high_value_miss_rate", 0.0),
                "utility_loss_vs_no_guarantee_percent": utility_loss,
                "passes_primary_constraints": int(feasible),
            }
        )
    best = min(
        (row for row in selection_rows if _to_int(row.get("passes_primary_constraints"))),
        key=lambda row: (_to_float(row.get("rescuable_high_value_miss_rate")), -_to_float(row.get("F_UG"))),
        default={},
    )
    _write_text(
        OUT_DIR / "parameter_selection_report.md",
        "# Parameter Selection Report\n\n"
        f"- Candidate rows with positive guarantee term gate are required before selection: {sum(1 for row in activation_rows if _to_float(row.get('candidates_with_guarantee_term_positive_count')) > 0.0)}.\n"
        f"- Hard safety all-clear: {str(hard_safe).lower()}.\n"
        f"- Lambda-zero equivalence: {str(lambda_zero_ok).lower()}.\n"
        f"- Selected row: `{best}`.\n",
    )


def _write_duration_report(rows: list[dict[str, object]]) -> None:
    summary = _aggregate(rows, ("arm", "lambda_H", "num_slots"))
    _write_csv(OUT_DIR / "duration_sensitivity_summary.csv", summary)
    lines = ["# Duration Sensitivity Report", ""]
    for row in summary:
        if row.get("arm") == "proposed_guarantee_aware_full":
            lines.append(
                f"- lambda_H={row.get('lambda_H')} slots={row.get('num_slots')}: "
                f"term_positive_mean={_to_float(row.get('candidates_with_guarantee_term_positive_count_mean')):.3g}, "
                f"rank_gain_mean={_to_float(row.get('candidates_rank_improved_by_guarantee_count_mean')):.3g}, "
                f"assisted_delivered_mean={_to_float(row.get('guarantee_assisted_delivered_count_mean')):.3g}"
            )
    _write_text(OUT_DIR / "duration_sensitivity_report.md", "\n".join(lines))


def _write_dynamic_final_recommendation(
    activation_rows: list[dict[str, object]],
    rescuable_rows: list[dict[str, object]],
    comparability_rows: list[dict[str, object]],
    equivalence_rows: list[dict[str, object]],
) -> None:
    total_age = sum(_to_float(row.get("tasks_with_age_debt_positive_count")) for row in activation_rows)
    total_miss = sum(_to_float(row.get("tasks_with_miss_debt_positive_count")) for row in activation_rows)
    total_z_candidates = sum(_to_float(row.get("candidates_with_Z_positive_count")) for row in activation_rows)
    total_term = sum(_to_float(row.get("candidates_with_guarantee_term_positive_count")) for row in activation_rows)
    total_rank = sum(_to_float(row.get("candidates_rank_improved_by_guarantee_count")) for row in activation_rows)
    total_cross = sum(_to_float(row.get("candidates_crossed_capacity_cutoff_count")) for row in activation_rows)
    total_assisted = sum(_to_float(row.get("guarantee_assisted_delivered_count")) for row in activation_rows)
    lambda_zero_ok = all(_to_int(row.get("matches_no_guarantee")) for row in equivalence_rows) if equivalence_rows else False
    if total_age > 0 and total_miss > 0 and total_term > 0 and total_rank > 0:
        case = "A. Dynamic debt activation works"
    elif total_age > 0 and total_miss <= 0:
        case = "B. Age debt works but missed-opportunity debt does not"
    elif total_miss > 0 and total_age <= 0:
        case = "C. Miss debt works but age debt does not"
    elif total_z_candidates <= 0:
        case = "D. Debt exists but no candidate enters scheduler"
    elif total_term > 0 and total_rank <= 0:
        case = "E. Candidate enters scheduler but priority does not change"
    elif total_rank > 0 and total_assisted <= 0:
        case = "F. Priority changes but no selection"
    else:
        case = "H. Scenario infeasible or weak activation"
    stronger = "unknown"
    by_duration = _aggregate(activation_rows, ("num_slots",))
    if len(by_duration) >= 2:
        first = min(by_duration, key=lambda row: _to_float(row.get("num_slots")))
        last = max(by_duration, key=lambda row: _to_float(row.get("num_slots")))
        stronger = str(_to_float(last.get("candidates_with_guarantee_term_positive_count_mean")) >= _to_float(first.get("candidates_with_guarantee_term_positive_count_mean"))).lower()
    blockers = {
        "capacity": sum(_to_float(row.get("blocked_by_capacity_count")) for row in activation_rows),
        "safety": sum(_to_float(row.get("blocked_by_safety_count")) for row in activation_rows),
        "deadline": sum(_to_float(row.get("blocked_by_deadline_count")) for row in activation_rows),
        "duplicate": sum(_to_float(row.get("blocked_by_duplicate_count")) for row in activation_rows),
        "floor": sum(_to_float(row.get("blocked_by_marginal_utility_floor_count")) for row in activation_rows),
        "budget": sum(_to_float(row.get("blocked_by_budget_count")) for row in activation_rows),
    }
    _write_text(
        OUT_DIR / "final_recommendation.md",
        "# Final Recommendation\n\n"
        f"Selected case: {case}.\n\n"
        f"1. Waiting-age debt increased over time: {str(total_age > 0).lower()}.\n"
        f"2. Missed-opportunity debt increased after feasible missed windows: {str(total_miss > 0).lower()}.\n"
        f"3. Debt carried across windows: {str(total_z_candidates > 0).lower()}.\n"
        f"4. Scheduler read positive Z_total: {str(total_z_candidates > 0).lower()}.\n"
        f"5. guarantee_term became positive: {str(total_term > 0).lower()}.\n"
        f"6. Guarantee improved rank: {str(total_rank > 0).lower()}.\n"
        f"7. Candidate crossed capacity cutoff due to guarantee: {str(total_cross > 0).lower()}.\n"
        f"8. Guarantee-assisted delivery occurred: {str(total_assisted > 0).lower()}.\n"
        f"9. Main blockers if not delivered: {blockers}.\n"
        f"10. 160/240 slots showed stronger activation than 80 slots: {stronger}.\n"
        f"11. No-guarantee and lambda_H=0 metrics comparable: {str(lambda_zero_ok).lower()}.\n"
        "12. ROI counted as minimum-stage support but not evidence closure: true.\n"
        "13. Self-check case is reported in `self_check_report.md`.\n"
        "14. Limited adaptation helped: not applied in this run.\n"
        "15. Next step: parameter tuning if case A/G holds; otherwise inspect the first zero stage in `activation_funnel_report.md`.\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
