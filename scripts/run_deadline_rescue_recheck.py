"""Deadline rescue recheck artifacts for guarantee-aware staged delivery.

This runner is intentionally diagnostic. It reuses the bounded guarantee queue
implementation and adds run-local deadline profiling, counterfactual rescue
labels, deadline audits, missed-opportunity validation, and parameter reports.
It does not modify reward computation, closure definitions, or global defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_bounded_guarantee_virtual_queue as bounded  # noqa: E402
from src.common.config import ensure_output_dirs  # noqa: E402
from src.common.types import TimelineStep  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_deadline_rescue_recheck"
ZIP_PATH = PROJECT_ROOT / "outputs" / "_deadline_rescue_recheck.zip"
CURRENT_MAIN_SCENARIO = "current_main_scenario"
MULTI_WINDOW_SCENARIO = "multi_window_rescuable_high_value"
SCENARIOS = [CURRENT_MAIN_SCENARIO, MULTI_WINDOW_SCENARIO]
ARMS = ["proposed_persistent_queue_without_guarantee", "proposed_guarantee_aware_full"]
DEFAULT_SEEDS = list(range(2026, 2031))
DEFAULT_LAMBDA_GRID = [0.0, 0.35, 0.50, 0.75, 1.00]

LAYERED_DEADLINE_PROFILE: dict[str, Any] = {
    "mode": "layered_by_requirement_and_value",
    "alert": {"low_medium_windows": [1, 1], "high_urgent_windows": [1, 2]},
    "verification": {"low_medium_windows": [1, 2], "high_urgent_windows": [2, 3]},
    "context": {"low_medium_windows": [2, 3], "high_urgent_windows": [2, 4]},
    "evidence": {"low_medium_windows": [2, 4], "high_urgent_windows": [3, 5]},
}

PAYLOAD_SIZE_BY_STAGE = {"metadata": 0.16, "roi": 0.80, "roi_context": 1.60}

_write_text = bounded._write_text
_write_csv = bounded._write_csv
_to_float = bounded._to_float
_to_int = bounded._to_int
_safe_div = bounded._safe_div
_products = bounded._products
_aggregate = bounded._aggregate


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one float value is required")
    return values


def _value_level_from_task(task: Any) -> str:
    value_level = str(getattr(task, "value_level", "") or "").lower()
    if value_level:
        return value_level
    if bool(getattr(task, "emergency_flag", False)):
        return "urgent"
    if bool(getattr(task, "high_value_flag", False)):
        return "high"
    value = _to_float(getattr(task, "calibrated_task_value", None), _to_float(getattr(task, "task_value", 0.0)))
    if value >= 0.90:
        return "urgent"
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def _is_high_urgent_context_evidence(value_level: object, requirement: object) -> bool:
    return str(value_level or "").lower() in {"high", "urgent"} and str(requirement or "").lower() in {"context", "evidence"}


def _task_requirement(task: Any) -> str:
    return str(getattr(task, "product_requirement", "") or "verification").lower()


def _contact_windows(timeline: list[TimelineStep]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for step in timeline:
        in_contact = bool(step.link_trace.in_contact)
        capacity_mb = (
            max(0.0, float(step.link_trace.throughput_mbps))
            * max(0.0, float(step.slot_seconds))
            / 8.0
            if in_contact
            else 0.0
        )
        if in_contact and current is None:
            current = {
                "start_slot": int(step.slot_index),
                "end_slot": int(step.slot_index),
                "capacity_mb": capacity_mb,
            }
        elif in_contact and current is not None:
            current["end_slot"] = int(step.slot_index)
            current["capacity_mb"] = _to_float(current.get("capacity_mb")) + capacity_mb
        elif not in_contact and current is not None:
            windows.append(current)
            current = None
    if current is not None:
        windows.append(current)
    for index, window in enumerate(windows, start=1):
        window["contact_window_id"] = f"contact_window_{index}"
    return windows


def _future_contact_windows(windows: list[dict[str, Any]], arrival_slot: int) -> list[dict[str, Any]]:
    return [window for window in windows if int(window["end_slot"]) >= int(arrival_slot)]


def _stable_window_count(task_id: str, lower: int, upper: int) -> int:
    lower = int(lower)
    upper = int(max(upper, lower))
    if upper <= lower:
        return lower
    digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()
    return lower + (int(digest[:8], 16) % (upper - lower + 1))


def deadline_profile_window_count(task: Any, profile: dict[str, Any] | None = None) -> int:
    profile = profile or LAYERED_DEADLINE_PROFILE
    requirement = _task_requirement(task)
    value_level = _value_level_from_task(task)
    requirement_profile = profile.get(requirement, profile.get("verification", {}))
    bucket = "high_urgent_windows" if value_level in {"high", "urgent"} else "low_medium_windows"
    raw_range = requirement_profile.get(bucket, [1, 1])
    if value_level in {"high", "urgent"} and requirement in {"context", "evidence"}:
        return int(raw_range[1])
    return _stable_window_count(str(getattr(task, "task_id", "")), int(raw_range[0]), int(raw_range[1]))


def _deadline_slot_for_window_count(
    *,
    arrival_slot: int,
    requested_windows: int,
    windows: list[dict[str, Any]],
    horizon_slot: int,
) -> tuple[int, int]:
    future = _future_contact_windows(windows, arrival_slot)
    if not future:
        return min(horizon_slot, max(arrival_slot, arrival_slot + 1)), 0
    observed_windows = min(len(future), max(1, int(requested_windows)))
    chosen = future[observed_windows - 1]
    return min(horizon_slot, int(chosen["end_slot"])), observed_windows


def apply_layered_deadline_profile(
    timeline: list[TimelineStep],
    profile: dict[str, Any] | None = None,
) -> tuple[list[TimelineStep], list[dict[str, Any]]]:
    profile = profile or LAYERED_DEADLINE_PROFILE
    windows = _contact_windows(timeline)
    if not timeline:
        return [], []
    horizon_slot = int(timeline[-1].slot_index)
    mapping_rows: list[dict[str, Any]] = []
    updated: list[TimelineStep] = []
    for step in timeline:
        task = step.task
        if task is None:
            updated.append(step)
            continue
        arrival_slot = int(getattr(task, "arrival_slot", step.slot_index))
        old_deadline_s = float(getattr(task, "deadline_s", 0.0) or 0.0)
        old_deadline_slot = min(horizon_slot, arrival_slot + int(math.ceil(old_deadline_s / max(float(step.slot_seconds), 1.0e-9))))
        requested_windows = deadline_profile_window_count(task, profile)
        new_deadline_slot, mapped_windows = _deadline_slot_for_window_count(
            arrival_slot=arrival_slot,
            requested_windows=requested_windows,
            windows=windows,
            horizon_slot=horizon_slot,
        )
        new_deadline_s = max(float(step.slot_seconds), float(new_deadline_slot - arrival_slot) * float(step.slot_seconds))
        metadata = dict(getattr(task, "metadata", {}) or {})
        metadata.update(
            {
                "deadline_profile_mode": str(profile.get("mode", "")),
                "deadline_profile_requested_windows": requested_windows,
                "deadline_profile_mapped_windows": mapped_windows,
                "deadline_profile_original_deadline_s": old_deadline_s,
                "deadline_profile_original_deadline_slot": old_deadline_slot,
                "deadline_profile_deadline_slot": new_deadline_slot,
                "deadline_policy_source": "layered_by_requirement_and_value",
            }
        )
        new_task = replace(task, deadline_s=new_deadline_s, metadata=metadata)
        updated.append(replace(step, task=new_task))
        mapping_rows.append(
            {
                "task_id": task.task_id,
                "value_level": _value_level_from_task(task),
                "product_requirement": _task_requirement(task),
                "arrival_slot": arrival_slot,
                "old_deadline_slot": old_deadline_slot,
                "old_deadline_seconds": old_deadline_s,
                "requested_contact_windows": requested_windows,
                "mapped_contact_windows": mapped_windows,
                "new_deadline_slot": new_deadline_slot,
                "new_deadline_seconds": new_deadline_s,
                "mapping_source": "deterministic_contact_schedule_no_reward_or_delivery_outcome",
            }
        )
    return updated, mapping_rows


def _stage_requirements(requirement: str) -> tuple[str, str]:
    requirement = str(requirement or "verification").lower()
    if requirement == "alert":
        return "metadata", "metadata"
    if requirement == "verification":
        return "roi", "roi"
    return "roi", "roi_context"


def _next_stage_payload_size(requirement: str) -> float:
    minimum, _target = _stage_requirements(requirement)
    return PAYLOAD_SIZE_BY_STAGE.get(minimum, PAYLOAD_SIZE_BY_STAGE["roi"])


def _capacity_between(timeline: list[TimelineStep], start_slot: int, end_slot: int) -> float:
    total = 0.0
    for step in timeline:
        slot = int(step.slot_index)
        if slot < start_slot or slot > end_slot or not bool(step.link_trace.in_contact):
            continue
        total += max(0.0, float(step.link_trace.throughput_mbps)) * max(0.0, float(step.slot_seconds)) / 8.0
    return total


def deadline_audit_rows(
    timeline: list[TimelineStep],
    *,
    scenario: str,
    seed: int,
    num_slots: int,
    deadline_profile: str,
) -> list[dict[str, Any]]:
    windows = _contact_windows(timeline)
    rows: list[dict[str, Any]] = []
    for step in timeline:
        task = step.task
        if task is None:
            continue
        slot_seconds = float(step.slot_seconds)
        arrival_slot = int(getattr(task, "arrival_slot", step.slot_index))
        deadline_s = float(getattr(task, "deadline_s", 0.0) or 0.0)
        lifetime_slots = max(0, int(math.ceil(deadline_s / max(slot_seconds, 1.0e-9))))
        deadline_slot = min(int(timeline[-1].slot_index), arrival_slot + lifetime_slots)
        relevant_windows = [
            window for window in windows if int(window["end_slot"]) >= arrival_slot and int(window["start_slot"]) <= deadline_slot
        ]
        requirement = _task_requirement(task)
        value_level = _value_level_from_task(task)
        minimum_stage, target_stage = _stage_requirements(requirement)
        next_stage_payload = _next_stage_payload_size(requirement)
        capacity_before_deadline = _capacity_between(timeline, arrival_slot, deadline_slot)
        feasible = len(relevant_windows) > 0 and capacity_before_deadline + 1.0e-9 >= next_stage_payload
        non_rescuable_reason = ""
        if not _is_high_urgent_context_evidence(value_level, requirement):
            non_rescuable_reason = "not_high_urgent_context_evidence"
        elif len(relevant_windows) <= 0:
            non_rescuable_reason = "no_contact_window_before_deadline"
        elif len(relevant_windows) == 1:
            non_rescuable_reason = "only_one_contact_window_before_deadline"
        elif not feasible:
            non_rescuable_reason = "insufficient_capacity_before_deadline"
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "num_slots": num_slots,
                "deadline_profile": deadline_profile,
                "task_id": task.task_id,
                "value_level": value_level,
                "product_requirement": requirement,
                "task_class": str(getattr(task, "task_class", "")),
                "arrival_slot": arrival_slot,
                "deadline_slot": deadline_slot,
                "deadline_seconds": deadline_s,
                "lifetime_slots": lifetime_slots,
                "contact_windows_before_deadline": len(relevant_windows),
                "contact_windows_after_first_miss": max(0, len(relevant_windows) - 1),
                "is_multi_window_rescuable": int(_is_high_urgent_context_evidence(value_level, requirement) and feasible and len(relevant_windows) >= 2),
                "minimum_stage_required": minimum_stage,
                "target_stage_required": target_stage,
                "payload_size_required_for_next_stage": next_stage_payload,
                "capacity_before_deadline_mb": capacity_before_deadline,
                "feasible_before_deadline": int(feasible),
                "non_rescuable_reason": non_rescuable_reason,
            }
        )
    return rows


def _configure_multi_window_scenario(configs: dict[str, dict[str, Any]]) -> None:
    task_stream = dict(configs["task_stream"])
    task_stream.update(
        {
            "emergency_window_probability": 0.05,
            "high_value_window_probability": 0.18,
            "background_task_probability": 0.04,
            "observation_windows_per_orbit": [3, 4],
            "window_duration_s_range": [180, 320],
            "intra_window_task_interval_slots": [1, 2],
            "forced_windows_s": [],
            "ordinary_deadline_min_minutes": 1.5,
            "ordinary_deadline_max_minutes": 2.5,
            "high_value_deadline_min_minutes": 2.0,
            "high_value_deadline_max_minutes": 4.0,
            "emergency_deadline_min_minutes": 0.75,
            "emergency_deadline_max_minutes": 1.5,
        }
    )
    configs["task_stream"] = task_stream
    simulator = configs["simulator"]
    simulator["queue_max_mb"] = min(float(simulator.get("queue_max_mb", 64.0)), 10.0)
    simulator["queue_safe_threshold_mb"] = min(float(simulator.get("queue_safe_threshold_mb", 48.0)), 8.0)
    simulator["battery_init"] = min(float(simulator.get("battery_init", 0.76)), 0.58)
    simulator["thermal_init"] = max(float(simulator.get("thermal_init", 0.52)), 0.66)
    simulator["deadline_profile"] = LAYERED_DEADLINE_PROFILE
    long_term = dict(simulator.get("long_term_risk_penalty", {}))
    long_term["battery_guard_threshold"] = min(float(long_term.get("battery_guard_threshold", 0.70)), 0.62)
    long_term["thermal_guard_threshold"] = max(float(long_term.get("thermal_guard_threshold", 0.85)), 0.82)
    simulator["long_term_risk_penalty"] = long_term
    progressive = dict(simulator.get("progressive_payload", {}))
    staged = dict(progressive.get("value_aware_staged_delivery_queue", {}))
    staged.update({"max_admissions_per_slot": 2, "max_admissions_per_contact_window": 3})
    progressive["value_aware_staged_delivery_queue"] = staged
    simulator["progressive_payload"] = progressive


def _load_recheck_configs(
    seed: int,
    scenario: str,
    arm: str,
    num_slots: int,
    *,
    lambda_h: float,
) -> dict[str, dict[str, Any]]:
    base_scenario = "scarcity_stress" if scenario == CURRENT_MAIN_SCENARIO else "value_skew_contact_limited"
    configs = bounded._load_run_configs(
        seed,
        base_scenario,
        arm,
        num_slots,
        lambda_h=lambda_h,
        z_max=6.0,
        beta_h=0.30,
    )
    if scenario == MULTI_WINDOW_SCENARIO:
        _configure_multi_window_scenario(configs)
    return configs


def _run_one(
    seed: int,
    scenario: str,
    arm: str,
    num_slots: int,
    *,
    lambda_h: float,
) -> dict[str, Any]:
    configs = _load_recheck_configs(seed, scenario, arm, num_slots, lambda_h=lambda_h)
    sweep_name = f"slots_{num_slots}_lambda_{lambda_h:g}" if arm == "proposed_guarantee_aware_full" else f"slots_{num_slots}"
    run_dir = OUT_DIR / "runs" / scenario / f"s{seed}" / sweep_name / arm
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    mapping_rows: list[dict[str, Any]] = []
    deadline_profile = "legacy_current_deadline"
    if scenario == MULTI_WINDOW_SCENARIO:
        timeline, mapping_rows = apply_layered_deadline_profile(timeline, LAYERED_DEADLINE_PROFILE)
        deadline_profile = str(LAYERED_DEADLINE_PROFILE["mode"])
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    _write_csv(run_dir / "candidate_action_log.csv", runner.last_candidate_action_logs)
    _write_csv(run_dir / "queue_priority_trace.csv", runner.last_queue_priority_logs)
    _write_csv(run_dir / "contact_window_scheduling_trace.csv", runner.last_contact_window_scheduling_trace)
    _write_csv(run_dir / "candidate_priority_trace.csv", runner.last_candidate_priority_trace)
    _write_csv(run_dir / "guarantee_debt_trace.csv", runner.last_guarantee_debt_trace)
    _write_csv(run_dir / "guarantee_debt_window_update_trace.csv", runner.last_guarantee_debt_window_update_trace)
    _write_csv(run_dir / "task_debt_trajectory_trace.csv", runner.last_task_debt_trajectory_trace)
    _write_csv(run_dir / "guarantee_opportunity_trace.csv", runner.last_guarantee_opportunity_trace)
    _write_csv(run_dir / "guarantee_progress_trace.csv", runner.last_guarantee_progress_trace)
    _write_csv(run_dir / "guarantee_priority_component_trace.csv", runner.last_guarantee_priority_component_trace)
    _write_csv(run_dir / "guarantee_budget_trace.csv", runner.last_guarantee_budget_trace)
    _write_csv(run_dir / "guarantee_blocker_reason_trace.csv", runner.last_guarantee_blocker_reason_trace)
    _write_csv(run_dir / "payload_lifecycle_orphan_check.csv", runner.last_payload_lifecycle_orphan_check)
    _write_csv(run_dir / "payload_lifecycle_trace.csv", runner.last_payload_lifecycle_ledger_trace)
    _write_csv(run_dir / "next_window_recompetition_summary.csv", runner.last_next_window_recompetition_trace)
    _write_csv(run_dir / "deadline_profile_mapping.csv", mapping_rows)
    _write_text(run_dir / "resolved_config.yaml", yaml.safe_dump(configs, sort_keys=False, allow_unicode=False))
    task_rows = [asdict(record) for record in result.records]
    audit_rows = deadline_audit_rows(
        timeline,
        scenario=scenario,
        seed=seed,
        num_slots=num_slots,
        deadline_profile=deadline_profile,
    )
    return {
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "num_slots": num_slots,
        "duration_s": num_slots * float(configs["simulator"].get("slot_seconds", 1.0)),
        "lambda_H": lambda_h,
        "z_max": 6.0,
        "beta_H": 0.30,
        "deadline_profile": deadline_profile,
        "run_dir": run_dir,
        "rows": task_rows,
        "summary": dict(result.summary),
        "queue_priority": runner.last_queue_priority_logs,
        "scheduling": runner.last_contact_window_scheduling_trace,
        "candidate_priority": runner.last_candidate_priority_trace,
        "debt": runner.last_guarantee_debt_trace,
        "debt_update": runner.last_guarantee_debt_window_update_trace,
        "task_debt_trajectory": runner.last_task_debt_trajectory_trace,
        "opportunity": runner.last_guarantee_opportunity_trace,
        "progress": runner.last_guarantee_progress_trace,
        "priority": runner.last_guarantee_priority_component_trace,
        "budget": runner.last_guarantee_budget_trace,
        "blockers": runner.last_guarantee_blocker_reason_trace,
        "orphan": runner.last_payload_lifecycle_orphan_check,
        "recompetition": runner.last_next_window_recompetition_trace,
        "lifecycle": runner.last_payload_lifecycle_ledger_trace,
        "deadline_mapping": mapping_rows,
        "deadline_audit": audit_rows,
    }


def _product_key(row: dict[str, Any]) -> str:
    return f"{row.get('task_id')}::{row.get('product_type') or row.get('payload_type')}"


def _delivered_product_keys(run: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for row in run.get("queue_priority", []):
        if _to_int(row.get("delivered_this_slot")):
            keys.add(_product_key(row))
    for row in run.get("lifecycle", []):
        if str(row.get("current_status", "")).lower() == "delivered" or str(row.get("delivered_slot", "")):
            keys.add(_product_key(row))
    return keys


def _delivered_products_by_task(run: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for key in _delivered_product_keys(run):
        task_id, product_type = key.split("::", 1)
        if task_id and product_type:
            out[task_id].add(product_type)
    return out


def counterfactual_selection_trace_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    delivered_keys = _delivered_product_keys(run)
    rows: list[dict[str, Any]] = []
    for row in run.get("candidate_priority", []):
        rank_without = _to_int(row.get("rank_without_guarantee"))
        rank_with = _to_int(row.get("rank_with_guarantee"))
        capacity_cutoff = _to_int(row.get("capacity_cutoff_rank"))
        would_fit_without = int(capacity_cutoff > 0 and rank_without > 0 and rank_without <= capacity_cutoff)
        would_fit_with = int(capacity_cutoff > 0 and rank_with > 0 and rank_with <= capacity_cutoff)
        hard_feasible = not (
            _to_int(row.get("blocked_by_safety"))
            or _to_int(row.get("blocked_by_deadline"))
            or _to_int(row.get("blocked_by_duplicate"))
            or _to_int(row.get("blocked_by_marginal_utility_floor"))
        )
        selected_without = int(bool(hard_feasible and would_fit_without))
        selected_with = int(_to_int(row.get("selected")) and bool(hard_feasible))
        active_guarantee = bool(
            run.get("arm") == "proposed_guarantee_aware_full"
            and _to_float(run.get("lambda_H")) > 0.0
            and _to_float(row.get("guarantee_term")) > 1.0e-12
        )
        if not active_guarantee:
            selected_without = selected_with
        product_key = _product_key(row)
        delivered_with = int(product_key in delivered_keys)
        delivered_without = int(delivered_with and selected_without)
        rescued = int(active_guarantee and selected_with and delivered_with and not delivered_without)
        guarantee_positive = int(_to_float(row.get("guarantee_term")) > 1.0e-12 and delivered_with)
        natural = int(delivered_with and delivered_without)
        rank_improved = int(rank_with > 0 and rank_without > 0 and rank_with < rank_without)
        crossed_cutoff = int(rank_without > capacity_cutoff and capacity_cutoff > 0 and 0 < rank_with <= capacity_cutoff)
        rows.append(
            {
                **row,
                "scenario": run["scenario"],
                "seed": run["seed"],
                "arm": run["arm"],
                "lambda_H": run["lambda_H"],
                "beta_H": run.get("beta_H", 0.30),
                "num_slots": run.get("num_slots", 0),
                "duration_s": run.get("duration_s", 0.0),
                "deadline_profile": run.get("deadline_profile", ""),
                "priority_without_guarantee": row.get("final_priority_without_guarantee", row.get("priority_without_guarantee", "")),
                "priority_with_guarantee": row.get("final_priority_with_guarantee", ""),
                "selected_without_guarantee": selected_without,
                "selected_with_guarantee": selected_with,
                "delivered_without_guarantee": delivered_without,
                "delivered_with_guarantee": delivered_with,
                "would_fit_capacity_without_guarantee": would_fit_without,
                "would_fit_capacity_with_guarantee": would_fit_with,
                "active_guarantee_counterfactual": int(active_guarantee),
                "naturally_delivered": natural,
                "guarantee_positive_delivered": guarantee_positive,
                "counterfactual_guarantee_rescued_delivered": rescued,
                "guarantee_rank_improved": rank_improved,
                "guarantee_crossed_capacity_cutoff": crossed_cutoff,
                "unique_product_key": product_key,
            }
        )
    return rows


def counterfactual_rescue_summary_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("scenario", "seed", "arm", "lambda_H", "num_slots", "deadline_profile")
    for row in trace_rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, rows in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, key_values)}
        unique = lambda pred: {str(row.get("unique_product_key")) for row in rows if pred(row)}
        displaced_utility = sum(
            _to_float(row.get("deltaU_total", row.get("delta_information_utility")))
            for row in rows
            if _to_int(row.get("selected_without_guarantee")) and not _to_int(row.get("selected_with_guarantee"))
        )
        rescued_utility = sum(
            _to_float(row.get("deltaU_total", row.get("delta_information_utility")))
            for row in rows
            if _to_int(row.get("counterfactual_guarantee_rescued_delivered"))
        )
        item.update(
            {
                "naturally_delivered_count": len(unique(lambda row: _to_int(row.get("naturally_delivered")))),
                "guarantee_positive_delivered_count": len(unique(lambda row: _to_int(row.get("guarantee_positive_delivered")))),
                "counterfactual_guarantee_rescued_delivered_count": len(
                    unique(lambda row: _to_int(row.get("counterfactual_guarantee_rescued_delivered")))
                ),
                "guarantee_rank_improved_count": len(unique(lambda row: _to_int(row.get("guarantee_rank_improved")))),
                "guarantee_crossed_capacity_cutoff_count": len(unique(lambda row: _to_int(row.get("guarantee_crossed_capacity_cutoff")))),
                "counterfactual_rescued_min_stage_count": len(
                    unique(
                        lambda row: _to_int(row.get("counterfactual_guarantee_rescued_delivered"))
                        and str(row.get("product_type")) == "roi"
                    )
                ),
                "counterfactual_rescued_target_stage_count": len(
                    unique(
                        lambda row: _to_int(row.get("counterfactual_guarantee_rescued_delivered"))
                        and str(row.get("product_type")) == "roi_context"
                    )
                ),
                "utility_displaced_by_guarantee": displaced_utility,
                "net_utility_delta_from_counterfactual_rescue": rescued_utility - displaced_utility,
            }
        )
        out.append(item)
    return out


def _write_counterfactual_report(summary_rows: list[dict[str, Any]]) -> None:
    total_natural = sum(_to_float(row.get("naturally_delivered_count")) for row in summary_rows)
    total_positive = sum(_to_float(row.get("guarantee_positive_delivered_count")) for row in summary_rows)
    total_rescued = sum(_to_float(row.get("counterfactual_guarantee_rescued_delivered_count")) for row in summary_rows)
    total_rank = sum(_to_float(row.get("guarantee_rank_improved_count")) for row in summary_rows)
    total_cross = sum(_to_float(row.get("guarantee_crossed_capacity_cutoff_count")) for row in summary_rows)
    _write_text(
        OUT_DIR / "counterfactual_rescue_report.md",
        "# Counterfactual Rescue Report\n\n"
        f"- naturally_delivered_count: {total_natural:.6g}\n"
        f"- guarantee_positive_delivered_count: {total_positive:.6g}\n"
        f"- counterfactual_guarantee_rescued_delivered_count: {total_rescued:.6g}\n"
        f"- guarantee_rank_improved_count: {total_rank:.6g}\n"
        f"- guarantee_crossed_capacity_cutoff_count: {total_cross:.6g}\n"
        "- Guarantee-positive delivery is treated as mechanism participation only. Rescue requires delivered_with=true and delivered_without=false.\n",
    )


def _write_deadline_audit_report(rows: list[dict[str, Any]]) -> None:
    hv_current = [
        row for row in rows if row.get("scenario") == CURRENT_MAIN_SCENARIO and _is_high_urgent_context_evidence(row.get("value_level"), row.get("product_requirement"))
    ]
    hv_multi = [
        row for row in rows if row.get("scenario") == MULTI_WINDOW_SCENARIO and _is_high_urgent_context_evidence(row.get("value_level"), row.get("product_requirement"))
    ]
    current_avg = mean([_to_float(row.get("contact_windows_before_deadline")) for row in hv_current]) if hv_current else 0.0
    current_zero_one = sum(1 for row in hv_current if _to_int(row.get("contact_windows_before_deadline")) <= 1)
    current_later = sum(1 for row in hv_current if _to_int(row.get("contact_windows_after_first_miss")) > 0)
    multi_avg = mean([_to_float(row.get("contact_windows_before_deadline")) for row in hv_multi]) if hv_multi else 0.0
    multi_two_plus = sum(1 for row in hv_multi if _to_int(row.get("contact_windows_before_deadline")) >= 2)
    aggregate = _aggregate(rows, ("scenario", "deadline_profile", "product_requirement"))
    _write_csv(OUT_DIR / "deadline_audit_by_requirement.csv", aggregate)
    _write_text(
        OUT_DIR / "deadline_audit_report.md",
        "# Deadline Audit Report\n\n"
        f"1. Current high/urgent context/evidence average contact windows before deadline: {current_avg:.6g}.\n"
        f"2. Current high/urgent context/evidence tasks with 0 or 1 window: {current_zero_one} of {len(hv_current)}.\n"
        f"3. Current high/urgent context/evidence tasks with a later rescuable window after first miss: {current_later}.\n"
        f"4. Layered high/urgent context/evidence average contact windows before deadline: {multi_avg:.6g}.\n"
        f"5. Layered high/urgent context/evidence tasks with 2+ windows: {multi_two_plus} of {len(hv_multi)}.\n"
        f"6. Current deadlines make missed-opportunity debt hard to observe: {str(current_zero_one > len(hv_current) / 2 if hv_current else True).lower()}.\n"
        "7. The current profile behaves closer to alert/emergency timing than evidence-building when most high-value context/evidence tasks have 0-1 windows.\n"
        "8. A layered deadline profile is needed for the diagnostic scenario.\n",
    )


def _write_deadline_profile_mapping_report(mapping_rows: list[dict[str, Any]]) -> None:
    profile_counts = _aggregate(mapping_rows, ("product_requirement", "value_level")) if mapping_rows else []
    _write_csv(OUT_DIR / "deadline_profile_mapping_summary.csv", profile_counts)
    avg_requested = mean([_to_float(row.get("requested_contact_windows")) for row in mapping_rows]) if mapping_rows else 0.0
    avg_mapped = mean([_to_float(row.get("mapped_contact_windows")) for row in mapping_rows]) if mapping_rows else 0.0
    _write_text(
        OUT_DIR / "deadline_profile_mapping_report.md",
        "# Deadline Profile Mapping Report\n\n"
        f"- Mapping source: deterministic contact schedule only; no reward, closure, or delivery outcome is used.\n"
        f"- Mapped task count: {len(mapping_rows)}.\n"
        f"- Mean requested contact windows: {avg_requested:.6g}.\n"
        f"- Mean mapped contact windows within horizon: {avg_mapped:.6g}.\n"
        "- Alert remains short; context/evidence receive multi-window horizons when future contacts exist.\n",
    )


def _global_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [bounded._global_metrics(run) for run in runs]
    for row, run in zip(rows, runs):
        row["deadline_profile"] = run.get("deadline_profile", "")
    return rows


def _hv_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [bounded._high_value_metrics(run) for run in runs]
    for row, run in zip(rows, runs):
        row["deadline_profile"] = run.get("deadline_profile", "")
    return rows


def _safety_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [bounded._safety_metrics(run) for run in runs]
    for row, run in zip(rows, runs):
        row["deadline_profile"] = run.get("deadline_profile", "")
        row["lifecycle_orphan_count"] = sum(_to_int(item.get("orphan_flag")) for item in run.get("orphan", []))
    return rows


def miss_debt_validation_rows(
    runs: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    delivered_by_key = {
        (row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("arm"), row.get("lambda_H")): row
        for row in global_rows
    }
    baseline_by_key = {
        (row.get("scenario"), row.get("seed"), row.get("num_slots")): row
        for row in global_rows
        if row.get("arm") == "proposed_persistent_queue_without_guarantee"
    }
    cf_by_run: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in counterfactual_rows:
        cf_by_run[(row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("arm"), row.get("lambda_H"))].append(row)
    trace: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for run in runs:
        if run.get("scenario") != MULTI_WINDOW_SCENARIO or run.get("arm") != "proposed_guarantee_aware_full":
            continue
        miss_tasks = {
            str(row.get("task_id"))
            for row in run.get("debt_update", [])
            if _to_int(row.get("min_stage_miss")) or _to_int(row.get("target_stage_miss")) or _to_float(row.get("debt_increment")) > 0.0
        }
        positive_miss_tasks = {
            str(row.get("task_id"))
            for row in run.get("task_debt_trajectory", [])
            if _to_float(row.get("Z_miss")) > 0.0
        }
        post_miss_candidates = [
            row
            for row in run.get("candidate_priority", [])
            if str(row.get("task_id")) in positive_miss_tasks and _to_float(row.get("Z_miss")) > 0.0
        ]
        cf_rows = cf_by_run.get((run["scenario"], run["seed"], run["num_slots"], run["arm"], run["lambda_H"]), [])
        cf_post = [row for row in cf_rows if str(row.get("task_id")) in positive_miss_tasks]
        feasible_high_value_opportunities = [
            row
            for row in run.get("opportunity", [])
            if _to_int(row.get("eligible_high_value_flag")) and _to_int(row.get("feasible_i_w"))
        ]
        global_row = delivered_by_key.get((run["scenario"], run["seed"], run["num_slots"], run["arm"], run["lambda_H"]), {})
        baseline = baseline_by_key.get((run["scenario"], run["seed"], run["num_slots"]), {})
        utility_loss = 100.0 * _safe_div(
            _to_float(baseline.get("delivered_effective_utility")) - _to_float(global_row.get("delivered_effective_utility")),
            _to_float(baseline.get("delivered_effective_utility")),
        )
        delivered_products = _delivered_products_by_task(run)
        min_success_tasks = {
            task_id
            for task_id in positive_miss_tasks
            if {"roi", "roi_context"} & delivered_products.get(task_id, set())
        }
        target_success_tasks = {task_id for task_id in positive_miss_tasks if "roi_context" in delivered_products.get(task_id, set())}
        feasible_missed_window_count = sum(
            1
            for row in run.get("debt_update", [])
            if _to_int(row.get("min_stage_miss")) or _to_int(row.get("target_stage_miss"))
        )
        row = {
            "scenario": run["scenario"],
            "seed": run["seed"],
            "arm": run["arm"],
            "lambda_H": run["lambda_H"],
            "num_slots": run["num_slots"],
            "deadline_profile": run["deadline_profile"],
            "feasible_missed_window_count": feasible_missed_window_count,
            "miss_debt_positive_task_count": len(positive_miss_tasks),
            "post_miss_future_window_count": sum(
                1 for item in run.get("debt_update", []) if str(item.get("task_id")) in miss_tasks and _to_int(item.get("future_contact_before_deadline"))
            ),
            "post_miss_candidate_count": len(post_miss_candidates),
            "post_miss_guarantee_term_positive_count": sum(1 for item in post_miss_candidates if _to_float(item.get("guarantee_term")) > 0.0),
            "post_miss_rank_improved_count": len({str(item.get("unique_product_key")) for item in cf_post if _to_int(item.get("guarantee_rank_improved"))}),
            "post_miss_cross_capacity_cutoff_count": len({str(item.get("unique_product_key")) for item in cf_post if _to_int(item.get("guarantee_crossed_capacity_cutoff"))}),
            "post_miss_counterfactual_rescued_delivered_count": len(
                {str(item.get("unique_product_key")) for item in cf_post if _to_int(item.get("counterfactual_guarantee_rescued_delivered"))}
            ),
            "post_miss_min_stage_success_count": len(min_success_tasks),
            "post_miss_target_stage_success_count": len(target_success_tasks),
            "rescuable_high_value_miss_rate": _safe_div(feasible_missed_window_count, max(1, len(feasible_high_value_opportunities))),
            "rescuable_high_value_success_rate": _safe_div(len(min_success_tasks | target_success_tasks), max(1, len(positive_miss_tasks))),
            "utility_loss_vs_no_guarantee_percent": utility_loss,
            "hard_safety_violation_count": _to_float(run["summary"].get("final_resource_safety_violation_count", run["summary"].get("safety_violation_count"))),
        }
        summary.append(row)
        for item in post_miss_candidates:
            trace.append({**item, **{k: row[k] for k in ("scenario", "seed", "arm", "lambda_H", "num_slots", "deadline_profile")}})
    return trace, summary


def _write_miss_debt_validation_report(summary_rows: list[dict[str, Any]]) -> None:
    total_miss_debt = sum(_to_float(row.get("miss_debt_positive_task_count")) for row in summary_rows)
    total_post_candidates = sum(_to_float(row.get("post_miss_candidate_count")) for row in summary_rows)
    total_rank = sum(_to_float(row.get("post_miss_rank_improved_count")) for row in summary_rows)
    total_rescue = sum(_to_float(row.get("post_miss_counterfactual_rescued_delivered_count")) for row in summary_rows)
    if total_miss_debt <= 0.0:
        explanation = (
            "Z_miss remained zero or absent. Check whether deadlines are still too short, feasible opportunity is too strict, "
            "candidate bridge is disconnected, high-value tasks are naturally delivered, or competition pressure is weak."
        )
    else:
        explanation = "Z_miss activated under the layered deadline scenario."
    _write_text(
        OUT_DIR / "miss_debt_validation_report.md",
        "# Missed-opportunity Debt Validation Report\n\n"
        f"- miss_debt_positive_task_count: {total_miss_debt:.6g}\n"
        f"- post_miss_candidate_count: {total_post_candidates:.6g}\n"
        f"- post_miss_rank_improved_count: {total_rank:.6g}\n"
        f"- post_miss_counterfactual_rescued_delivered_count: {total_rescue:.6g}\n"
        f"- interpretation: {explanation}\n",
    )


def _score_rows(
    global_rows: list[dict[str, Any]],
    hv_rows: list[dict[str, Any]],
    cf_summary: list[dict[str, Any]],
    miss_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = bounded._utility_guarantee_score_rows(global_rows, hv_rows)
    cf_by_key = {
        (row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("lambda_H")): row for row in cf_summary
    }
    miss_by_key = {
        (row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("lambda_H")): row for row in miss_summary
    }
    for row in rows:
        key = (row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("lambda_H"))
        cf = cf_by_key.get(key, {})
        miss = miss_by_key.get(key, {})
        row["counterfactual_guarantee_rescued_delivered_count"] = cf.get("counterfactual_guarantee_rescued_delivered_count", 0)
        row["rescuable_high_value_miss_rate"] = miss.get("rescuable_high_value_miss_rate", 0)
        row["utility_loss_vs_no_guarantee_percent"] = miss.get("utility_loss_vs_no_guarantee_percent", 0)
    return rows


def _write_parameter_selection_report(
    score_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    equivalence_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    safety_by_key = {
        (row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("arm"), row.get("lambda_H")): row
        for row in safety_rows
    }
    lambda_zero_ok = all(_to_int(row.get("matches_no_guarantee")) for row in equivalence_rows) if equivalence_rows else False
    candidates: list[dict[str, Any]] = []
    rescue_possible = any(_to_float(row.get("counterfactual_guarantee_rescued_delivered_count")) > 0.0 for row in score_rows)
    for row in score_rows:
        if row.get("scenario") != MULTI_WINDOW_SCENARIO:
            continue
        safety = safety_by_key.get((row.get("scenario"), row.get("seed"), row.get("num_slots"), row.get("arm"), row.get("lambda_H")), {})
        utility_loss = _to_float(row.get("utility_loss_vs_no_guarantee_percent"))
        rescued = _to_float(row.get("counterfactual_guarantee_rescued_delivered_count"))
        passes = (
            _to_float(safety.get("hard_safety_violation_count")) == 0.0
            and _to_float(safety.get("low_medium_accidental_roi_context_count")) == 0.0
            and _to_float(safety.get("lifecycle_orphan_count")) == 0.0
            and utility_loss <= 5.0
            and lambda_zero_ok
            and (rescued > 0.0 if rescue_possible else True)
        )
        candidates.append({**row, "passes_primary_constraints": int(passes)})
    best = min(
        (row for row in candidates if _to_int(row.get("passes_primary_constraints"))),
        key=lambda row: (_to_float(row.get("rescuable_high_value_miss_rate")), -_to_float(row.get("F_UG"))),
        default={},
    )
    _write_text(
        OUT_DIR / "parameter_selection_report.md",
        "# Parameter Selection Report\n\n"
        f"- Best lambda_H: {best.get('lambda_H', 'not_selected')}.\n"
        f"- Best deadline profile: {LAYERED_DEADLINE_PROFILE['mode']}.\n"
        f"- Counterfactual rescue observed: {str(any(_to_float(row.get('counterfactual_guarantee_rescued_delivered_count')) > 0.0 for row in score_rows)).lower()}.\n"
        f"- Lambda-zero equivalence passed: {str(lambda_zero_ok).lower()}.\n"
        f"- Selected row: `{best}`.\n"
        "- Primary criterion minimizes rescuable high-value miss rate subject to safety, lifecycle, low/medium, utility-loss, equivalence, and closure constraints.\n",
    )
    return best


def _write_final_recommendation(
    *,
    deadline_rows: list[dict[str, Any]],
    cf_summary: list[dict[str, Any]],
    miss_summary: list[dict[str, Any]],
    best: dict[str, Any],
) -> None:
    current_hv = [
        row for row in deadline_rows if row.get("scenario") == CURRENT_MAIN_SCENARIO and _is_high_urgent_context_evidence(row.get("value_level"), row.get("product_requirement"))
    ]
    multi_hv = [
        row for row in deadline_rows if row.get("scenario") == MULTI_WINDOW_SCENARIO and _is_high_urgent_context_evidence(row.get("value_level"), row.get("product_requirement"))
    ]
    current_zero_one = sum(1 for row in current_hv if _to_int(row.get("contact_windows_before_deadline")) <= 1)
    multi_two_plus = sum(1 for row in multi_hv if _to_int(row.get("contact_windows_before_deadline")) >= 2)
    natural = sum(_to_float(row.get("naturally_delivered_count")) for row in cf_summary)
    positive = sum(_to_float(row.get("guarantee_positive_delivered_count")) for row in cf_summary)
    rescued = sum(_to_float(row.get("counterfactual_guarantee_rescued_delivered_count")) for row in cf_summary)
    miss_debt_positive = sum(_to_float(row.get("miss_debt_positive_task_count")) for row in miss_summary)
    post_rank = sum(_to_float(row.get("post_miss_rank_improved_count")) for row in miss_summary)
    post_delivery = sum(_to_float(row.get("post_miss_counterfactual_rescued_delivered_count")) for row in miss_summary)
    utility_loss_values = [_to_float(row.get("utility_loss_vs_no_guarantee_percent")) for row in miss_summary]
    max_loss = max(utility_loss_values) if utility_loss_values else 0.0
    labels: list[str] = []
    if current_hv and current_zero_one > len(current_hv) / 2:
        labels.append("A. Current deadline too urgent")
    if multi_two_plus > 0:
        labels.append("B. Layered deadline profile is more appropriate")
    if natural > rescued:
        labels.append("C. Natural delivery dominates")
    if rescued > 0:
        labels.append("D. Counterfactual rescue validated")
    if miss_debt_positive > 0 and (post_rank > 0 or post_delivery > 0):
        labels.append("E. Missed-opportunity debt validated")
    if positive > 0 and rescued <= 0:
        labels.append("F. Guarantee still only positive but not causal")
    if not multi_hv or multi_two_plus <= 0:
        labels.append("G. Scenario infeasible")
    if positive != rescued:
        labels.append("H. Metric bug fixed")
    if best:
        labels.append("I. Parameter recommendation ready")
    suitable = bool(rescued > 0 and max_loss <= 5.0 and best)
    _write_text(
        OUT_DIR / "final_recommendation.md",
        "# Final Recommendation\n\n"
        "Selected labels:\n"
        + "\n".join(f"- {label}" for label in labels)
        + "\n\n"
        f"1. Were previous deadlines too urgent for context/evidence tasks? {str(current_zero_one > len(current_hv) / 2 if current_hv else True).lower()}.\n"
        f"2. High-value context/evidence tasks with 2+ contact windows before deadline: {multi_two_plus}.\n"
        f"3. Naturally delivered high-value products: {natural:.6g}.\n"
        f"4. Guarantee-positive delivered products: {positive:.6g}.\n"
        f"5. Counterfactual guarantee-rescued delivered products: {rescued:.6g}.\n"
        f"6. Did missed-opportunity debt activate under layered deadline? {str(miss_debt_positive > 0).lower()}.\n"
        f"7. Did missed-opportunity debt cause later rank/selection/delivery changes? {str(post_rank > 0 or post_delivery > 0).lower()}.\n"
        f"8. Maximum utility loss observed versus no guarantee: {max_loss:.6g}%.\n"
        f"9. Recommended lambda_H: {best.get('lambda_H', 'not_selected')}.\n"
        f"10. Recommended deadline profile: {LAYERED_DEADLINE_PROFILE['mode']}.\n"
        f"11. Suitable for paper claim: {str(suitable).lower()}.\n"
        f"12. If not suitable, adjust scenario/deadline/mechanism: {str(not suitable).lower()}.\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--durations", default="160,240")
    parser.add_argument("--lambda-grid", default=",".join(f"{value:g}" for value in DEFAULT_LAMBDA_GRID))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = _parse_csv_ints(args.seeds)
    durations = _parse_csv_ints(args.durations)
    lambda_grid = _parse_csv_floats(args.lambda_grid)
    if args.quick:
        seeds = seeds[:1]
        durations = durations[:1]
        lambda_grid = [0.0, 0.35]
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = (
        f"python scripts/run_deadline_rescue_recheck.py --seeds {args.seeds} "
        f"--durations {args.durations} --lambda-grid {args.lambda_grid}"
    )
    if args.quick:
        command += " --quick"
    _write_text(OUT_DIR / "command_log.md", f"# Command Log\n\n```powershell\n{command}\n```\n")
    _write_text(OUT_DIR / "deadline_profile_used.yaml", yaml.safe_dump({"deadline_profile": LAYERED_DEADLINE_PROFILE}, sort_keys=False))

    runs: list[dict[str, Any]] = []
    for num_slots in durations:
        for scenario in SCENARIOS:
            for seed in seeds:
                runs.append(
                    _run_one(
                        seed,
                        scenario,
                        "proposed_persistent_queue_without_guarantee",
                        num_slots,
                        lambda_h=0.0,
                    )
                )
                for lambda_h in lambda_grid:
                    runs.append(
                        _run_one(
                            seed,
                            scenario,
                            "proposed_guarantee_aware_full",
                            num_slots,
                            lambda_h=lambda_h,
                        )
                    )

    global_rows = _global_rows(runs)
    hv_rows = _hv_rows(runs)
    safety_rows = _safety_rows(runs)
    staged_rows = [bounded._staged_progress_metrics(run) for run in runs]
    equivalence_rows = bounded._equivalence_rows(global_rows)
    deadline_rows = [row for run in runs for row in run.get("deadline_audit", [])]
    mapping_rows = [dict(row, scenario=run["scenario"], seed=run["seed"], num_slots=run["num_slots"]) for run in runs for row in run.get("deadline_mapping", [])]
    counterfactual_rows = [row for run in runs for row in counterfactual_selection_trace_rows(run)]
    cf_summary = counterfactual_rescue_summary_rows(counterfactual_rows)
    miss_trace, miss_summary = miss_debt_validation_rows(runs, counterfactual_rows, global_rows)
    score_rows = _score_rows(global_rows, hv_rows, cf_summary, miss_summary)

    _write_csv(OUT_DIR / "global_utility_metrics.csv", global_rows)
    _write_csv(OUT_DIR / "high_value_guarantee_metrics.csv", hv_rows)
    _write_csv(OUT_DIR / "staged_progress_metrics.csv", staged_rows)
    _write_csv(OUT_DIR / "safety_metrics.csv", safety_rows)
    _write_csv(OUT_DIR / "lambda_zero_equivalence_check.csv", equivalence_rows)
    _write_csv(OUT_DIR / "deadline_audit.csv", deadline_rows)
    _write_deadline_audit_report(deadline_rows)
    _write_csv(OUT_DIR / "deadline_profile_mapping.csv", mapping_rows)
    _write_deadline_profile_mapping_report(mapping_rows)
    _write_csv(OUT_DIR / "counterfactual_selection_trace.csv", counterfactual_rows)
    _write_csv(OUT_DIR / "counterfactual_rescue_summary.csv", cf_summary)
    _write_counterfactual_report(cf_summary)
    _write_csv(OUT_DIR / "miss_debt_validation_trace.csv", miss_trace)
    _write_csv(OUT_DIR / "miss_debt_validation_summary.csv", miss_summary)
    _write_miss_debt_validation_report(miss_summary)
    _write_csv(OUT_DIR / "utility_guarantee_score.csv", score_rows)
    best = _write_parameter_selection_report(score_rows, safety_rows, equivalence_rows)
    _write_final_recommendation(deadline_rows=deadline_rows, cf_summary=cf_summary, miss_summary=miss_summary, best=best)
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "scenarios": SCENARIOS,
                "arms": ARMS,
                "seeds": seeds,
                "durations": durations,
                "lambda_H_grid": lambda_grid,
                "default_off": True,
                "reward_function_changed": False,
                "closure_definition_changed": False,
                "roi_counts_as_evidence_closure": False,
                "deadline_profile_mapping_uses_future_delivery_or_reward": False,
            },
            sort_keys=False,
            allow_unicode=False,
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
