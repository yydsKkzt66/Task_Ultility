"""Blocker-aware ROI+Ctx gate diagnosis, sweep, and after-sweep A/B."""

from __future__ import annotations

import argparse
import csv
import itertools
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
from src.payload.roi_context_gate import DEFAULT_GATE_PROFILES  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_roi_context_blocker_aware_gate_sweep"
ZIP_PATH = PROJECT_ROOT / "outputs" / "_roi_context_blocker_aware_gate_sweep.zip"
PROPOSED_ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
SCENARIOS = ["natural", "targeted_resource_scarcity"]
DEFAULT_SEEDS = list(range(2026, 2036))
RUN_DIR_ALIAS = {
    "legacy_proposed": "legacy",
    "soft_moderate_value_skew_plus_gate_current": "skew_gate",
    "soft_moderate_value_skew_plus_gate_plus_progressive_current": "prog_current",
    "soft_moderate_value_skew_plus_gate_plus_progressive_recommended": "prog_rec",
    "soft_moderate_value_skew_plus_gate_plus_progressive_current": "prog_current",
}
BLOCKER_REASONS = [
    "semantic_requirement_not_context_or_evidence",
    "value_level_not_high_or_urgent",
    "eriv_below_profile_threshold",
    "value_density_below_profile_threshold",
    "queue_pressure_too_high",
    "battery_unsafe",
    "thermal_unsafe",
    "no_useful_remaining_deadline",
    "no_contact_opportunity",
    "top_k_budget_exceeded",
    "window_budget_exceeded",
    "hard_safety_guard_blocked_capacity",
    "hard_safety_guard_blocked_queue",
    "hard_safety_guard_blocked_battery",
    "hard_safety_guard_blocked_thermal",
    "hard_safety_guard_blocked_deadline",
    "hard_safety_guard_blocked_downlink_throttled",
    "duplicate_suppression",
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


def _products(raw: object) -> set[str]:
    return {item.strip() for item in str(raw or "").replace(",", ";").split(";") if item.strip()}


def _gate_profile(params: dict[str, object], *, allow_queued: bool) -> dict[str, object]:
    profile = dict(DEFAULT_GATE_PROFILES["soft_moderate"])
    profile.update(
        {
            "eriv_threshold": float(params.get("eriv_threshold", 0.03)),
            "min_value_density": float(params.get("min_value_density", 0.03)),
            "queue_pressure_threshold": float(params.get("queue_pressure_threshold", 0.72)),
            "battery_guard_threshold": float(params.get("battery_guard_threshold", 0.50)),
            "thermal_guard_threshold": float(params.get("thermal_guard_threshold", 0.88)),
            "min_remaining_deadline_s": float(params.get("min_remaining_deadline_s", 30.0)),
            "require_contact_opportunity": True,
            "require_current_contact_for_same_slot": True,
            "allow_queued_followup_without_current_contact": bool(allow_queued),
            "require_future_contact_before_deadline_for_queue": True,
            "top_k_per_slot": int(params.get("top_k_per_slot", 1)),
            "top_k_per_window": int(params.get("top_k_per_slot", 1)),
            "top_k_per_contact_window": int(params.get("top_k_per_contact_window", 2)),
            "queued_followup_budget_per_slot": int(params.get("queued_followup_budget_per_slot", 0)),
            "window_budget": int(params.get("top_k_per_contact_window", 2)),
        }
    )
    return profile


def _gate_cfg(params: dict[str, object], *, allow_queued: bool) -> dict[str, object]:
    return {"enabled": True, "profile": "soft_moderate", "profiles": {"soft_moderate": _gate_profile(params, allow_queued=allow_queued)}}


def _apply_progressive_profile(configs: dict[str, dict[str, Any]], params: dict[str, object], *, allow_queued: bool) -> None:
    progressive = dict(configs["simulator"].get("progressive_payload", {}))
    progressive["roi_context_gate"] = _gate_cfg(params, allow_queued=allow_queued)
    progressive["phase6_proactive_followup"] = {
        "enabled": True,
        "eriv_threshold": float(params.get("eriv_threshold", 0.03)),
        "residual_utility_threshold": 0.01,
        "queue_pressure_threshold": float(params.get("queue_pressure_threshold", 0.72)),
        "min_remaining_deadline_s": float(params.get("min_remaining_deadline_s", 30.0)),
        "battery_guard_threshold": float(params.get("battery_guard_threshold", 0.50)),
        "thermal_guard_threshold": float(params.get("thermal_guard_threshold", 0.88)),
        "require_contact_opportunity": True,
        "proactive_queue_bonus": 0.30,
        "full_method_aliases": [PROPOSED_ROUTER],
    }
    progressive["phase6_residual_candidate_carryover"] = {
        "enabled": True,
        "max_candidates_per_slot": 2,
        "max_candidate_age_s": 600,
    }
    progressive["phase6_same_slot_carryover"] = {
        "enabled": True,
        "only_after_lifecycle_update": True,
        "reuse_existing_guards": True,
        "max_same_slot_eval_per_slot": int(params.get("top_k_per_slot", 1)),
        "enable_queue_fallback_on_insufficient_capacity": False,
        "enforce_no_reward_change": True,
        "enforce_no_queue_policy_change": True,
        "strict_mode": True,
    }
    progressive["roi_context_after_roi"] = {"enabled": True, "residual_utility_threshold": 0.01}
    configs["simulator"]["progressive_payload"] = progressive


def _load_run_configs(
    seed: int,
    scenario: str,
    arm: str,
    num_slots: int,
    params: dict[str, object] | None = None,
    *,
    allow_queued: bool = False,
) -> dict[str, dict[str, Any]]:
    if arm == "legacy_proposed":
        return base_ab._load_run_configs(seed, scenario, "legacy_proposed", num_slots)
    if arm == "soft_moderate_value_skew_plus_gate_current":
        return base_ab._load_run_configs(seed, scenario, "soft_moderate_value_skew_plus_gate", num_slots)
    configs = base_ab._load_run_configs(seed, scenario, "soft_moderate_value_skew_plus_gate_plus_progressive_followup", num_slots)
    _apply_progressive_profile(configs, params or _current_params(), allow_queued=allow_queued)
    return configs


def _run_one(
    *,
    seed: int,
    scenario: str,
    arm: str,
    num_slots: int,
    params: dict[str, object] | None,
    allow_queued: bool,
    run_root: Path,
    config_id: str = "",
) -> dict[str, Any]:
    configs = _load_run_configs(seed, scenario, arm, num_slots, params, allow_queued=allow_queued)
    scenario_dir = base_ab.SCENARIO_DIR.get(scenario, scenario)
    run_dir_name = RUN_DIR_ALIAS.get(config_id or arm, config_id or arm)
    run_dir = run_root / "runs" / scenario_dir / f"s{seed}" / run_dir_name
    output_dirs = ensure_output_dirs(run_dir)
    for path in output_dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    rows = [asdict(record) for record in result.records]
    _write_csv(run_dir / "candidate_action_log.csv", runner.last_candidate_action_logs)
    _write_csv(run_dir / "carryover_lifecycle_events.csv", runner.last_carryover_lifecycle_events)
    _write_csv(run_dir / "same_slot_carryover_trace.csv", runner.last_same_slot_carryover_trace)
    _write_csv(run_dir / "followup_hard_safety_guard_trace.csv", runner.last_followup_hard_safety_guard_trace)
    _write_csv(run_dir / "roi_context_budget_trace.csv", runner.last_roi_context_window_budget_trace)
    _write_csv(run_dir / "value_density_trace.csv", runner.last_value_density_trace)
    _write_csv(run_dir / "same_slot_vs_queued_followup_trace.csv", runner.last_same_slot_vs_queued_followup_trace)
    _write_csv(run_dir / "future_contact_feasibility_trace.csv", runner.last_future_contact_feasibility_trace)
    _write_csv(run_dir / "duplicate_suppression_trace.csv", runner.last_duplicate_suppression_trace)
    _write_text(run_dir / "resolved_config.yaml", yaml.safe_dump(configs, sort_keys=False, allow_unicode=True))
    return {
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "config_id": config_id,
        "params": dict(params or {}),
        "allow_queued": allow_queued,
        "rows": rows,
        "summary": dict(result.summary),
        "candidate_logs": runner.last_candidate_action_logs,
        "carryover_events": runner.last_carryover_lifecycle_events,
        "followup_guard": runner.last_followup_hard_safety_guard_trace,
        "budget_trace": runner.last_roi_context_window_budget_trace,
        "value_density_trace": runner.last_value_density_trace,
        "same_slot_vs_queued": runner.last_same_slot_vs_queued_followup_trace,
        "future_contact": runner.last_future_contact_feasibility_trace,
        "duplicate_trace": runner.last_duplicate_suppression_trace,
    }


def _summarize_run(run: dict[str, Any]) -> dict[str, object]:
    base = base_ab._summarize_run(run)
    rows = [row for row in run["rows"] if _to_int(row.get("has_task"))]
    last_by_task = {str(row.get("task_id")): row for row in rows}
    high_value_context_evidence_task_count = 0
    low_medium_accidental = 0
    for row in last_by_task.values():
        requirement = str(row.get("product_requirement", ""))
        value_level = str(row.get("value_level", ""))
        delivered = _products(row.get("delivered_products")) | _products(row.get("delivered_product_types"))
        is_high = value_level in {"high", "urgent"} or bool(_to_int(row.get("high_value_flag")))
        if requirement in {"context", "evidence"} and is_high:
            high_value_context_evidence_task_count += 1
        if value_level in {"low", "medium"} and "roi_context" in delivered:
            low_medium_accidental += 1
    blocker_counts = _blocker_counts_for_runs([run])
    top = [reason for reason, _count in Counter(blocker_counts).most_common(3)]
    total_downlinked_mb = _to_float(run["summary"].get("total_downlinked_payload_mb"))
    delivered_utility = _to_float(run["summary"].get("delivered_only_incremental_utility"))
    base.update(
        {
            "config_id": run.get("config_id", ""),
            "allow_queued_followup_without_current_contact": int(bool(run.get("allow_queued"))),
            "high_value_context_evidence_task_count": high_value_context_evidence_task_count,
            "roi_context_candidate_count": sum(
                1
                for row in run["candidate_logs"]
                if str(row.get("candidate_payload")) in {"roi_context", "metadata_then_roi_context"}
            ),
            "roi_context_pending_registered_count": sum(
                1
                for row in run["same_slot_vs_queued"]
                if row.get("stage") == "roi_context_after_roi_registration" and row.get("action") == "registered_pending"
            ),
            "roi_context_enqueued_count": _to_float(run["summary"].get("proactive_followup_enqueue_count")),
            "roi_context_delivered_count": base.get("high_value_roi_context_delivered_count", 0),
            "low_medium_accidental_roi_context_count": low_medium_accidental,
            "effective_utility": _to_float(run["summary"].get("total_effective_utility")),
            "delivered_utility": delivered_utility,
            "utility_per_MB": _safe_div(_to_float(run["summary"].get("net_system_utility")), total_downlinked_mb),
            "blocker_reason_top1": top[0] if len(top) > 0 else "",
            "blocker_reason_top2": top[1] if len(top) > 1 else "",
            "blocker_reason_top3": top[2] if len(top) > 2 else "",
        }
    )
    return base


def _aggregate_by_config(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        key
        for key in per_seed[0]
        if key not in {"seed", "scenario", "arm", "config_id", "blocker_reason_top1", "blocker_reason_top2", "blocker_reason_top3"}
    ] if per_seed else []
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        groups[(str(row.get("scenario", "")), str(row.get("arm", "")), str(row.get("config_id", "")))].append(row)
    out = []
    for (scenario, arm, config_id), rows in sorted(groups.items()):
        item: dict[str, object] = {"scenario": scenario, "arm": arm, "config_id": config_id, "n": len(rows)}
        for metric in metrics:
            values = [_to_float(row.get(metric)) for row in rows]
            item[f"{metric}_mean"] = mean(values) if values else 0.0
            if metric in {"queue_next_mb_max", "hard_safety_violation_count", "expired_items"}:
                item[f"{metric}_max"] = max(values) if values else 0.0
        blockers = []
        for row in rows:
            blockers.extend([str(row.get("blocker_reason_top1", "")), str(row.get("blocker_reason_top2", "")), str(row.get("blocker_reason_top3", ""))])
        top = [reason for reason, _count in Counter(reason for reason in blockers if reason).most_common(3)]
        item["blocker_reason_top1"] = top[0] if len(top) > 0 else ""
        item["blocker_reason_top2"] = top[1] if len(top) > 1 else ""
        item["blocker_reason_top3"] = top[2] if len(top) > 2 else ""
        out.append(item)
    return out


def _split_blockers(raw: object) -> list[str]:
    return [item for item in str(raw or "").replace(",", ";").split(";") if item]


def _map_guard_blocker(reason: str) -> str:
    mapping = {
        "followup_hard_safety_blocked_capacity": "hard_safety_guard_blocked_capacity",
        "followup_hard_safety_blocked_queue": "hard_safety_guard_blocked_queue",
        "followup_hard_safety_blocked_battery": "hard_safety_guard_blocked_battery",
        "followup_hard_safety_blocked_thermal": "hard_safety_guard_blocked_thermal",
        "followup_hard_safety_blocked_deadline": "hard_safety_guard_blocked_deadline",
        "followup_hard_safety_blocked_expired": "hard_safety_guard_blocked_deadline",
        "followup_hard_safety_blocked_downlink_throttled": "hard_safety_guard_blocked_downlink_throttled",
    }
    return mapping.get(reason, reason)


def _blocker_rows_for_runs(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        common = {"scenario": run["scenario"], "seed": run["seed"], "arm": run["arm"], "config_id": run.get("config_id", "")}
        for row in run["budget_trace"]:
            blockers = _split_blockers(row.get("blockers")) or ([str(row.get("reason_code"))] if row.get("allowed") in {0, "0"} else [])
            for blocker in blockers:
                if blocker in BLOCKER_REASONS:
                    rows.append({**common, "stage": row.get("stage", "roi_context_gate"), "reason": blocker, "count": 1})
        for row in run["carryover_events"]:
            for blocker in _split_blockers(row.get("proactive_blockers")):
                mapped = _map_guard_blocker(blocker)
                if mapped in BLOCKER_REASONS:
                    rows.append({**common, "stage": "carryover_queued_followup_gate", "reason": mapped, "count": 1})
        for row in run["followup_guard"]:
            for blocker in _split_blockers(row.get("blockers")):
                mapped = _map_guard_blocker(blocker)
                if mapped in BLOCKER_REASONS:
                    rows.append({**common, "stage": str(row.get("operation", "followup_hard_safety_guard")), "reason": mapped, "count": 1})
        for _row in run["duplicate_trace"]:
            rows.append({**common, "stage": "duplicate_suppression", "reason": "duplicate_suppression", "count": 1})
    return rows


def _blocker_counts_for_runs(runs: list[dict[str, Any]]) -> Counter:
    return Counter(row["reason"] for row in _blocker_rows_for_runs(runs))


def _summarize_blockers(runs: list[dict[str, Any]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    raw = _blocker_rows_for_runs(runs)
    total = Counter(row["reason"] for row in raw)
    summary = [{"reason": reason, "count": total.get(reason, 0)} for reason in BLOCKER_REASONS]
    by_scenario_counter = Counter((row["scenario"], row["reason"]) for row in raw)
    by_scenario = [
        {"scenario": scenario, "reason": reason, "count": by_scenario_counter.get((scenario, reason), 0)}
        for scenario in SCENARIOS
        for reason in BLOCKER_REASONS
    ]
    by_stage_counter = Counter((row["stage"], row["reason"]) for row in raw)
    stages = sorted({str(row["stage"]) for row in raw}) or ["none"]
    by_stage = [
        {"stage": stage, "reason": reason, "count": by_stage_counter.get((stage, reason), 0)}
        for stage in stages
        for reason in BLOCKER_REASONS
    ]
    return summary, by_scenario, by_stage


def _current_params() -> dict[str, object]:
    return {
        "eriv_threshold": 0.03,
        "min_value_density": 0.03,
        "queue_pressure_threshold": 0.72,
        "battery_guard_threshold": 0.50,
        "thermal_guard_threshold": 0.88,
        "min_remaining_deadline_s": 30,
        "top_k_per_slot": 1,
        "top_k_per_contact_window": 2,
        "queued_followup_budget_per_slot": 0,
    }


def _coarse_grid() -> list[dict[str, object]]:
    return [
        {"eriv_threshold": 0.005, "min_value_density": 0.005, "queue_pressure_threshold": 0.85, "min_remaining_deadline_s": 10, "top_k_per_slot": 2, "top_k_per_contact_window": 3, "queued_followup_budget_per_slot": 1},
        {"eriv_threshold": 0.010, "min_value_density": 0.010, "queue_pressure_threshold": 0.80, "min_remaining_deadline_s": 20, "top_k_per_slot": 2, "top_k_per_contact_window": 3, "queued_followup_budget_per_slot": 1},
        {"eriv_threshold": 0.015, "min_value_density": 0.015, "queue_pressure_threshold": 0.75, "min_remaining_deadline_s": 30, "top_k_per_slot": 1, "top_k_per_contact_window": 2, "queued_followup_budget_per_slot": 1},
        {"eriv_threshold": 0.020, "min_value_density": 0.020, "queue_pressure_threshold": 0.75, "min_remaining_deadline_s": 30, "top_k_per_slot": 1, "top_k_per_contact_window": 2, "queued_followup_budget_per_slot": 0},
        {"eriv_threshold": 0.030, "min_value_density": 0.030, "queue_pressure_threshold": 0.70, "min_remaining_deadline_s": 40, "top_k_per_slot": 1, "top_k_per_contact_window": 2, "queued_followup_budget_per_slot": 0},
    ]


def _local_grid(best: dict[str, object] | None) -> list[dict[str, object]]:
    if not best:
        return []
    erivs = sorted({max(0.005, _to_float(best.get("eriv_threshold")) - 0.005), _to_float(best.get("eriv_threshold")), min(0.03, _to_float(best.get("eriv_threshold")) + 0.005)})
    densities = sorted({max(0.005, _to_float(best.get("min_value_density")) - 0.005), _to_float(best.get("min_value_density")), min(0.03, _to_float(best.get("min_value_density")) + 0.005)})
    rows = []
    for eriv, density in itertools.product(erivs, densities):
        row = dict(best)
        row["eriv_threshold"] = eriv
        row["min_value_density"] = density
        rows.append(row)
    return rows[:4]


def _config_id(stage: str, index: int) -> str:
    return f"{stage}_{index:03d}"


def _is_hard_feasible(row: dict[str, object], queue_limit: float) -> bool:
    return (
        _to_float(row.get("hard_safety_violation_count_mean")) == 0.0
        and _to_float(row.get("low_medium_accidental_roi_context_count_mean")) <= 0.0
        and _to_float(row.get("queue_next_mb_max_max", row.get("queue_next_mb_max_mean"))) <= queue_limit + 1.0e-9
    )


def _select_pareto(grid_rows: list[dict[str, object]], queue_limit: float) -> list[dict[str, object]]:
    feasible = [row for row in grid_rows if _is_hard_feasible(row, queue_limit)]
    if not feasible:
        return []
    viable = [
        row
        for row in feasible
        if _to_float(row.get("metadata_to_roi_context_success_rate_mean")) > 0.0
        and _to_float(row.get("roi_context_delivered_count_mean")) > 0.0
    ]
    if not viable:
        return []
    candidates = viable
    candidates = sorted(
        candidates,
        key=lambda row: (
            _to_float(row.get("context_closure_count_mean")) + _to_float(row.get("evidence_closure_count_mean")),
            _to_float(row.get("high_value_context_evidence_closure_ratio_mean")),
            _to_float(row.get("utility_per_MB_mean")),
            _to_float(row.get("net_system_utility_mean")),
            -_to_float(row.get("expired_items_mean")),
        ),
        reverse=True,
    )
    return candidates[:10]


def _write_value_density_report(rows: list[dict[str, object]]) -> None:
    checked = 0
    mismatch = 0
    for row in rows:
        task_value = _to_float(row.get("task_effective_value"), -1.0)
        residual = _to_float(row.get("residual_delta_u_i"))
        quality = _to_float(row.get("detection_quality"))
        timeliness = _to_float(row.get("timeliness_factor"))
        payload = _to_float(row.get("payload_size_mb"))
        if task_value < 0 or payload <= 0:
            continue
        checked += 1
        expected_info = residual / payload
        expected_value = task_value * residual * quality * timeliness / payload
        if abs(_to_float(row.get("information_density")) - expected_info) > 1.0e-4:
            mismatch += 1
        if abs(_to_float(row.get("value_density")) - expected_value) > 1.0e-4:
            mismatch += 1
    _write_text(
        OUT_DIR / "value_density_consistency_report.md",
        "# Value Density Consistency Report\n\n"
        f"- Checked trace rows: {checked}\n"
        f"- Formula mismatches above tolerance: {mismatch}\n"
        "- `value_density` is task-effective-value aware and includes marginal information, detection quality, and timeliness or delivery proxy per MB.\n"
        "- `information_density` remains residual marginal information per MB and is reported separately for interpretation.\n"
        "- ERIV remains the stricter follow-up score; it additionally includes contact opportunity and z_k permission where applicable.\n",
    )


def _write_blocker_report(summary: list[dict[str, object]]) -> None:
    ordered = sorted(summary, key=lambda row: _to_float(row.get("count")), reverse=True)
    top = [row for row in ordered if _to_float(row.get("count")) > 0][:8]
    dominant = top[0]["reason"] if top else "none"
    text = ["# Top Blockers Report", ""]
    text.append(f"Dominant blocker: `{dominant}`.")
    text.append("")
    text.append("| reason | count |")
    text.append("|---|---:|")
    for row in ordered:
        text.append(f"| {row['reason']} | {row['count']} |")
    text.append("")
    text.append("Interpretation categories:")
    text.append("- ERIV/value-density strictness: `eriv_below_profile_threshold`, `value_density_below_profile_threshold`.")
    text.append("- Contact strictness: `no_contact_opportunity` and future-contact infeasibility traces.")
    text.append("- Resource hard guards: queue, battery, thermal, capacity, deadline, downlink-throttle hard-safety blockers.")
    text.append("- Budget strictness: `top_k_budget_exceeded`, `window_budget_exceeded`.")
    text.append("- Duplicate issues: `duplicate_suppression`.")
    _write_text(OUT_DIR / "top_blockers_report.md", "\n".join(text))


def _comparison_rows(per_seed: list[dict[str, object]], metric_names: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        groups[(str(row["scenario"]), str(row["arm"]))].append(row)
    rows = []
    for (scenario, arm), group_rows in sorted(groups.items()):
        item: dict[str, object] = {"scenario": scenario, "arm": arm, "n": len(group_rows)}
        for metric in metric_names:
            values = [_to_float(row.get(metric)) for row in group_rows]
            item[f"{metric}_mean"] = mean(values) if values else 0.0
        rows.append(item)
    return rows


def _paired_delta_rows(per_seed: list[dict[str, object]], baseline_arm: str, target_arm: str) -> list[dict[str, object]]:
    by_key = {(row["scenario"], row["seed"], row["arm"]): row for row in per_seed}
    metrics = ["roi_context_delivered_count", "metadata_to_roi_context_success_rate", "context_closure_count", "evidence_closure_count", "net_system_utility", "utility_per_MB", "queue_next_mb_max", "expired_items", "hard_safety_violation_count"]
    deltas = []
    for row in per_seed:
        if row["arm"] != target_arm:
            continue
        base = by_key.get((row["scenario"], row["seed"], baseline_arm))
        if base is None:
            continue
        item = {"scenario": row["scenario"], "seed": row["seed"], "baseline_arm": baseline_arm, "target_arm": target_arm}
        for metric in metrics:
            item[f"delta_{metric}"] = _to_float(row.get(metric)) - _to_float(base.get(metric))
        deltas.append(item)
    return deltas


def _final_case(recommended: dict[str, object] | None, current_agg: list[dict[str, object]], rec_agg: list[dict[str, object]]) -> str:
    if recommended is None:
        return "C"
    if any(_to_float(row.get("hard_safety_violation_count_mean")) > 0.0 for row in rec_agg):
        return "D"
    if any(_to_float(row.get("low_medium_accidental_roi_context_count_mean")) > 0.0 for row in rec_agg):
        return "D"
    delivered = sum(_to_float(row.get("roi_context_delivered_count_mean")) for row in rec_agg)
    if delivered <= 0.0:
        return "C"
    rec_closure = sum(_to_float(row.get("context_closure_count_mean")) + _to_float(row.get("evidence_closure_count_mean")) for row in rec_agg)
    cur_closure = sum(_to_float(row.get("context_closure_count_mean")) + _to_float(row.get("evidence_closure_count_mean")) for row in current_agg)
    if rec_closure <= cur_closure:
        return "B"
    return "A"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--num-slots", type=int, default=80)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    command = f"python scripts/run_roi_context_blocker_aware_gate_sweep.py --seeds {args.seeds} --num-slots {args.num_slots}"
    _write_text(OUT_DIR / "command_log.md", f"# Command Log\n\n```powershell\n{command}\n```\n")

    current_runs: list[dict[str, Any]] = []
    current_params = _current_params()
    for scenario in SCENARIOS:
        for seed in seeds:
            current_runs.append(
                _run_one(
                    seed=seed,
                    scenario=scenario,
                    arm="soft_moderate_value_skew_plus_gate_plus_progressive_current",
                    num_slots=args.num_slots,
                    params=current_params,
                    allow_queued=False,
                    run_root=OUT_DIR / "current_progressive",
                    config_id="current_progressive",
                )
            )

    blocker_summary, blocker_by_scenario, blocker_by_stage = _summarize_blockers(current_runs)
    _write_csv(OUT_DIR / "blocker_reason_summary.csv", blocker_summary)
    _write_csv(OUT_DIR / "blocker_reason_by_scenario.csv", blocker_by_scenario)
    _write_csv(OUT_DIR / "blocker_reason_by_stage.csv", blocker_by_stage)
    _write_blocker_report(blocker_summary)

    value_density_rows: list[dict[str, object]] = []
    same_slot_vs_queued_rows: list[dict[str, object]] = []
    future_contact_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    for run in current_runs:
        common = {"scenario": run["scenario"], "seed": run["seed"], "arm": run["arm"], "config_id": run.get("config_id", "")}
        value_density_rows.extend(dict(row, **common) for row in run["value_density_trace"])
        same_slot_vs_queued_rows.extend(dict(row, **common) for row in run["same_slot_vs_queued"])
        future_contact_rows.extend(dict(row, **common) for row in run["future_contact"])
        budget_rows.extend(dict(row, **common) for row in run["budget_trace"])
    _write_csv(OUT_DIR / "value_density_trace.csv", value_density_rows)
    _write_value_density_report(value_density_rows)
    _write_csv(OUT_DIR / "same_slot_vs_queued_followup_trace.csv", same_slot_vs_queued_rows)
    _write_csv(OUT_DIR / "future_contact_feasibility_trace.csv", future_contact_rows)
    _write_csv(OUT_DIR / "roi_context_budget_trace.csv", budget_rows)
    contact_summary_counter = Counter((row.get("scenario"), row.get("contact_window_id"), row.get("allowed")) for row in budget_rows)
    _write_csv(
        OUT_DIR / "contact_window_budget_summary.csv",
        [
            {"scenario": scenario, "contact_window_id": window, "allowed": allowed, "count": count}
            for (scenario, window, allowed), count in sorted(contact_summary_counter.items())
        ],
    )

    sweep_per_seed: list[dict[str, object]] = []
    sweep_agg: list[dict[str, object]] = []
    params_by_config: dict[str, dict[str, object]] = {}
    if not args.skip_sweep:
        coarse = _coarse_grid()
        all_stage_runs: list[dict[str, Any]] = []
        for index, params in enumerate(coarse):
            config_id = _config_id("stage1", index)
            params_by_config[config_id] = params
            stage_runs = []
            for scenario in SCENARIOS:
                for seed in seeds:
                    stage_runs.append(
                        _run_one(
                            seed=seed,
                            scenario=scenario,
                            arm="sweep_candidate",
                            num_slots=args.num_slots,
                            params=params,
                            allow_queued=True,
                            run_root=OUT_DIR / "gate_sweep_runs",
                            config_id=config_id,
                        )
                    )
            all_stage_runs.extend(stage_runs)
            sweep_per_seed.extend(_summarize_run(run) for run in stage_runs)
        stage1_agg = _aggregate_by_config(sweep_per_seed)
        stage1_front = _select_pareto(stage1_agg, queue_limit=64.0)
        best_config_id = str(stage1_front[0].get("config_id")) if stage1_front else ""
        local = _local_grid(params_by_config.get(best_config_id))
        for index, params in enumerate(local):
            config_id = _config_id("stage2", index)
            params_by_config[config_id] = params
            stage_runs = []
            for scenario in SCENARIOS:
                for seed in seeds:
                    stage_runs.append(
                        _run_one(
                            seed=seed,
                            scenario=scenario,
                            arm="sweep_candidate",
                            num_slots=args.num_slots,
                            params=params,
                            allow_queued=True,
                            run_root=OUT_DIR / "gate_sweep_runs",
                            config_id=config_id,
                        )
                    )
            all_stage_runs.extend(stage_runs)
            sweep_per_seed.extend(_summarize_run(run) for run in stage_runs)
        sweep_agg = _aggregate_by_config(sweep_per_seed)

    for row in sweep_agg:
        row.update(params_by_config.get(str(row.get("config_id")), {}))
    pareto = _select_pareto(sweep_agg, queue_limit=64.0)
    recommended = dict(pareto[0]) if pareto else None
    recommended_params = params_by_config.get(str(recommended.get("config_id"))) if recommended else None
    hold_params = dict(current_params)
    hold_params["queued_followup_budget_per_slot"] = 1
    profile_to_write = recommended_params or hold_params
    _write_csv(OUT_DIR / "gate_sweep_grid.csv", sweep_agg)
    _write_csv(OUT_DIR / "gate_sweep_pareto_front.csv", pareto)
    _write_text(
        OUT_DIR / "recommended_roi_context_gate_profile.yaml",
        yaml.safe_dump(
            {
                "recommendation_case": "A" if recommended_params else "C",
                "recommended": bool(recommended_params),
                "note": (
                    "Pareto-valid gate found."
                    if recommended_params
                    else "No sweep config delivered ROI+Ctx under safety and low/medium-accidental constraints; this is a safe hold profile, not a deployment recommendation."
                ),
                "roi_context_gate": _gate_cfg(profile_to_write, allow_queued=True),
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(
        OUT_DIR / "gate_sweep_report.md",
        "# Gate Sweep Report\n\n"
        f"- Stage 1 configs: {len(_coarse_grid())}\n"
        f"- Stage 2 configs: {max(0, len(params_by_config) - len(_coarse_grid()))}\n"
        f"- Pareto candidates: {len(pareto)}\n"
        f"- Recommended config id: {'' if recommended is None else recommended.get('config_id')}\n"
        f"- Recommendation status: {'Pareto-valid recommendation found' if recommended_params else 'no Pareto-valid recommendation; Case C'}\n",
    )

    ab_arms = [
        ("legacy_proposed", None, False),
        ("soft_moderate_value_skew_plus_gate_current", None, False),
        ("soft_moderate_value_skew_plus_gate_plus_progressive_current", current_params, False),
        ("soft_moderate_value_skew_plus_gate_plus_progressive_recommended", profile_to_write, True),
    ]
    ab_runs: list[dict[str, Any]] = []
    ab_per_seed: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for seed in seeds:
            for arm, params, allow_queued in ab_arms:
                run = _run_one(
                    seed=seed,
                    scenario=scenario,
                    arm=arm,
                    num_slots=args.num_slots,
                    params=params,
                    allow_queued=allow_queued,
                    run_root=OUT_DIR / "after_sweep_ab_runs",
                    config_id=arm,
                )
                ab_runs.append(run)
                ab_per_seed.append(_summarize_run(run))
    ab_summary = _aggregate_by_config(ab_per_seed)
    _write_csv(OUT_DIR / "after_sweep_ab_summary.csv", ab_summary)
    _write_csv(
        OUT_DIR / "after_sweep_paired_delta_summary.csv",
        _paired_delta_rows(
            ab_per_seed,
            "soft_moderate_value_skew_plus_gate_plus_progressive_current",
            "soft_moderate_value_skew_plus_gate_plus_progressive_recommended",
        ),
    )
    _write_csv(OUT_DIR / "context_evidence_closure_comparison.csv", _comparison_rows(ab_per_seed, ["context_closure_count", "evidence_closure_count", "high_value_context_evidence_closure_ratio"]))
    _write_csv(OUT_DIR / "low_value_sacrifice_comparison.csv", _comparison_rows(ab_per_seed, ["low_medium_accidental_roi_context_count"]))
    _write_csv(OUT_DIR / "queue_expired_safety_comparison.csv", _comparison_rows(ab_per_seed, ["queue_next_mb_max", "expired_items", "hard_safety_violation_count"]))
    _write_csv(OUT_DIR / "utility_comparison.csv", _comparison_rows(ab_per_seed, ["net_system_utility", "utility_per_MB", "effective_utility", "delivered_utility"]))

    current_agg = [row for row in ab_summary if row.get("arm") == "soft_moderate_value_skew_plus_gate_plus_progressive_current"]
    rec_agg = [row for row in ab_summary if row.get("arm") == "soft_moderate_value_skew_plus_gate_plus_progressive_recommended"]
    case = _final_case(recommended if recommended_params else None, current_agg, rec_agg)
    top_blocker = max(blocker_summary, key=lambda row: _to_float(row.get("count")))["reason"] if blocker_summary else "none"
    rec_delivery = sum(_to_float(row.get("roi_context_delivered_count_mean")) for row in rec_agg)
    rec_safety = sum(_to_float(row.get("hard_safety_violation_count_mean")) for row in rec_agg)
    rec_low = sum(_to_float(row.get("low_medium_accidental_roi_context_count_mean")) for row in rec_agg)
    rec_success = sum(_to_float(row.get("metadata_to_roi_context_success_rate_mean")) for row in rec_agg)
    rec_closure = sum(_to_float(row.get("context_closure_count_mean")) + _to_float(row.get("evidence_closure_count_mean")) for row in rec_agg)
    cur_closure = sum(_to_float(row.get("context_closure_count_mean")) + _to_float(row.get("evidence_closure_count_mean")) for row in current_agg)
    _write_text(
        OUT_DIR / "final_recommendation.md",
        "# Final Recommendation\n\n"
        f"Case: {case}.\n\n"
        f"1. Main current blocker: `{top_blocker}`.\n"
        "2. Value-density: unified; `value_density` and `information_density` are traced separately.\n"
        "3. No-contact handling: no-contact cannot same-slot deliver ROI+Ctx; it can only register/enqueue queued follow-up.\n"
        "4. Future contact before deadline: used for queued follow-up feasibility when current contact is absent.\n"
        "5. Budget: slot/contact-window/queued budget counters are active and traced.\n"
        f"6. Recommended gate profile: {'none; see safe hold profile in `recommended_roi_context_gate_profile.yaml`' if not recommended_params else recommended_params}.\n"
        f"7. Recommended hard_safety_violation mean sum: {rec_safety:.6g}.\n"
        f"8. Recommended low/medium accidental ROI+Ctx mean sum: {rec_low:.6g}.\n"
        f"9. Recommended high-value ROI+Ctx delivered mean sum: {rec_delivery:.6g}.\n"
        f"10. Recommended metadata-to-ROI-to-ROI+Ctx success-rate mean sum: {rec_success:.6g}.\n"
        f"11. Context/evidence closure current -> recommended: {cur_closure:.6g} -> {rec_closure:.6g}.\n"
        "12. Queue/expired/net utility: see `queue_expired_safety_comparison.csv` and `utility_comparison.csv`.\n"
        f"13. Full baseline A/B readiness: {'yes' if case in {'A', 'B'} else 'no'}.\n",
    )

    _write_text(
        OUT_DIR / "README.md",
        "# ROI+Ctx Blocker-Aware Gate Sweep\n\n"
        "This package diagnoses current progressive ROI+Ctx blockers, unifies value-density tracing, separates same-slot and queued follow-up, performs a two-stage gate sweep, and reruns a small after-sweep A/B.",
    )
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "num_slots": args.num_slots,
                "scenarios": SCENARIOS,
                "profile": "soft_moderate",
                "coarse_grid_count": len(_coarse_grid()),
                "total_sweep_config_count": len(params_by_config),
                "recommended_params": recommended_params,
                "safe_hold_params": hold_params if recommended_params is None else {},
                "final_case": case,
                "legacy_default_unchanged": True,
                "reward_function_changed": False,
                "closure_definition_changed": False,
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
                "src/payload/roi_context_gate.py",
                "src/payload/proactive_followup.py",
                "src/simulator/step_runner.py",
                "src/simulator/episode_runner.py",
                "scripts/run_roi_context_blocker_aware_gate_sweep.py",
                "tests/test_roi_context_gate.py",
                "tests/test_roi_context_blocker_aware_gate_sweep.py",
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
    print(f"case={case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
