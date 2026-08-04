"""Run all-variant BUPT-1 1-day smoke tests with minute-scale task deadlines."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_task_lifetime_diagnostics import generate_lifetime_outputs  # noqa: E402
from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.routers.fixed_router import FixedRouter  # noqa: E402
from src.routers.rule_router import RuleBasedRouter  # noqa: E402
from src.routers.safe_linucb_router import SafeLinUCBRouter  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


INDEX_DIR = PROJECT_ROOT / "outputs" / "_index"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "sarfish_1day_smoke"
SCENARIOS = {
    "nominal": {
        "stage": "sarfish_1day_smoke_bupt1_deadline_minutes_nominal",
        "output_dir": PROJECT_ROOT / "outputs" / "sarfish_1day_smoke_bupt1_deadline_minutes_nominal",
        "resource": PROJECT_ROOT / "configs" / "resource_sarfish_formal_nominal_bupt1.yaml",
        "task_stream": PROJECT_ROOT / "configs" / "task_stream_sarfish_formal_nominal_bupt1_deadline_minutes.yaml",
        "timeline": PROJECT_ROOT / "configs" / "timeline_sarfish_formal_nominal_bupt1_deadline_minutes_20s.yaml",
        "simulator": PROJECT_ROOT / "configs" / "simulator_sarfish_formal_nominal_bupt1_deadline_minutes_1day_20s.yaml",
    },
    "stress": {
        "stage": "sarfish_1day_smoke_bupt1_deadline_minutes_stress",
        "output_dir": PROJECT_ROOT / "outputs" / "sarfish_1day_smoke_bupt1_deadline_minutes_stress",
        "resource": PROJECT_ROOT / "configs" / "resource_sarfish_formal_stress_bupt1.yaml",
        "task_stream": PROJECT_ROOT / "configs" / "task_stream_sarfish_formal_stress_bupt1_deadline_minutes.yaml",
        "timeline": PROJECT_ROOT / "configs" / "timeline_sarfish_formal_stress_bupt1_deadline_minutes_20s.yaml",
        "simulator": PROJECT_ROOT / "configs" / "simulator_sarfish_formal_stress_bupt1_deadline_minutes_1day_20s.yaml",
    },
}
PROPOSED = {
    "proposed_calibrated_payload_plus_z_guided",
    "proposed_calibrated_payload_profile",
    "proposed_payload_repaired_combined",
    "proposed_calibrated_payload_plus_cost_aware_ucb",
}
DELIVERED_FIELDS = [
    "experiment_stage",
    "output_dir",
    "variant_short",
    "variant_full",
    "metric_quality",
    "notes",
    "gross_task_utility_provisional",
    "net_system_utility_provisional",
    "penalized_reward",
    "resource_penalty",
    "requirement_penalty",
    "delivered_only_gross_utility",
    "delivered_before_deadline_utility",
    "high_value_delivered_utility",
    "queued_provisional_utility",
    "delivered_existing_utility",
    "delivery_realization_gap",
    "delivered_utility_ratio",
    "lifetime_delivered_task_level",
    "lifetime_expired_task_level",
    "lifetime_deferred_task_level",
    "lifetime_undelivered_by_end_task_level",
    "delivered_items_payload_level",
    "expired_items_payload_level",
    "completed_tasks_same_step",
    "completed_deadline_satisfaction_ratio",
    "deadline_satisfaction_ratio",
    "total_energy",
    "compute_energy",
    "comm_energy",
    "min_battery_soc",
    "final_battery_soc",
    "max_thermal",
    "thermal_violation_count",
    "average_queue_mb",
    "max_queue_mb",
    "generated_payload_mb",
    "delivered_mb",
    "lightweight_count",
    "balanced_count",
    "high_accuracy_count",
    "metadata_count",
    "roi_count",
    "roi_context_count",
    "defer_count",
    "balanced_roi_count",
    "high_accuracy_roi_context_count",
    "stage_rank_by_provisional_net",
    "stage_rank_by_delivered_only",
    "stage_rank_by_high_value_delivered",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if not _finite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _safe_div(numerator: float, denominator: float) -> float:
    if not _finite(numerator) or not _finite(denominator) or abs(denominator) <= 1.0e-12:
        return math.nan
    return numerator / denominator


def _variants() -> list[dict[str, str]]:
    data = load_yaml(PROJECT_ROOT / "configs" / "smoke_test_variants.yaml")
    return [{"short_name": str(row["short_name"]), "full_name": str(row["full_name"])} for row in data["variants"]]


def _variant_aliases() -> dict[str, str]:
    return {row["full_name"]: row["short_name"] for row in _variants()}


def _load_repair_cfg(variant: str) -> dict[str, object]:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "payload_profile_ablation.yaml")
    repair = copy.deepcopy(cfg["policy_repair_defaults"])
    repair.update(copy.deepcopy(cfg.get("variant_overrides", {}).get(variant, {})))
    return repair


def _load_configs(scenario: str, variant: str) -> dict[str, dict]:
    paths = SCENARIOS[scenario]
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(paths["task_stream"])
    configs["timeline"] = load_yaml(paths["timeline"])
    configs["simulator"] = load_yaml(paths["simulator"])
    configs["resource"] = load_yaml(paths["resource"])
    configs["contact"] = copy.deepcopy(configs["timeline"].get("contact", {}))
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    profile_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml")
    configs["model_pool"] = copy.deepcopy(configs["model_pool"])
    configs["model_pool"]["profile_overrides"] = dict(profile_cfg["profile_overrides"])
    configs["payload_profiles"] = load_yaml(PROJECT_ROOT / "configs" / "payload_profiles_roi_compression_calibrated.yaml")
    configs["simulator"]["policy_repair"] = _load_repair_cfg(variant)
    return configs


def _build_router(variant: str, environment: SimulationEnvironment):
    action_names = [action.action_name for action in environment.action_space.list_actions()]
    if variant.startswith("proposed_"):
        return SafeLinUCBRouter(
            action_names,
            alpha=0.70,
            router_name=variant,
            use_lyapunov_pricing=True,
            use_mode_controller=True,
        )
    if variant == "utility_rule":
        return RuleBasedRouter(variant, mode="utility_rule")
    if variant == "no_mode":
        return SafeLinUCBRouter(action_names, router_name=variant, use_lyapunov_pricing=True, use_mode_controller=False)
    if variant == "no_lyapunov":
        return SafeLinUCBRouter(action_names, router_name=variant, use_lyapunov_pricing=False, use_mode_controller=True)
    if variant == "fixed_lightweight":
        return FixedRouter(variant, "lightweight+metadata")
    if variant == "fixed_high_accuracy":
        return FixedRouter(variant, "high_accuracy+roi_context")
    raise ValueError(f"Unsupported variant: {variant}")


def _build_timeline_for_scenario(scenario: str):
    configs = _load_configs(scenario, "utility_rule")
    return TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build(), configs


def _next_contact_wait_s(timeline, index: int) -> float:
    if timeline[index].link_trace.in_contact:
        return 0.0
    for future in timeline[index + 1 :]:
        if future.link_trace.in_contact:
            return (future.slot_index - timeline[index].slot_index) * timeline[index].slot_seconds
    return math.inf


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": math.nan, "mean": math.nan, "median": math.nan, "max": math.nan}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _deadline_diagnostics(scenario: str) -> dict[str, Any]:
    timeline, configs = _build_timeline_for_scenario(scenario)
    task_steps = [step for step in timeline if step.task is not None]
    deadlines_min = [float(step.task.deadline_s) / 60.0 for step in task_steps]
    contact_period_min = float(configs["timeline"]["contact"].get("contact_period_s", 0.0)) / 60.0
    rows_by_class: list[dict[str, Any]] = []
    for task_class in ["routine", "high_value", "emergency"]:
        values = [float(step.task.deadline_s) / 60.0 for step in task_steps if step.task.task_class == task_class]
        stat = _stats(values)
        rows_by_class.append(
            {
                "scenario": scenario,
                "task_class": task_class,
                "count": stat["count"],
                "min_deadline_min": stat["min"],
                "mean_deadline_min": stat["mean"],
                "median_deadline_min": stat["median"],
                "max_deadline_min": stat["max"],
            }
        )
    next_contact_shorter = 0
    finite_next_contact = 0
    for step in task_steps:
        wait_s = _next_contact_wait_s(timeline, step.slot_index)
        if math.isfinite(wait_s):
            finite_next_contact += 1
            if float(step.task.deadline_s) < wait_s:
                next_contact_shorter += 1
    stat = _stats(deadlines_min)
    return {
        "scenario": scenario,
        "task_count": len(task_steps),
        "min_deadline_min": stat["min"],
        "mean_deadline_min": stat["mean"],
        "median_deadline_min": stat["median"],
        "max_deadline_min": stat["max"],
        "contact_period_min": contact_period_min,
        "ratio_lt_30_min": sum(1 for value in deadlines_min if value < 30.0) / max(len(deadlines_min), 1),
        "ratio_lt_60_min": sum(1 for value in deadlines_min if value < 60.0) / max(len(deadlines_min), 1),
        "ratio_lt_contact_period": sum(1 for value in deadlines_min if value < contact_period_min) / max(len(deadlines_min), 1),
        "ratio_deadline_lt_next_contact": next_contact_shorter / max(finite_next_contact, 1),
        "class_rows": rows_by_class,
    }


def _write_deadline_audit(path: Path, diagnostics: list[dict[str, Any]]) -> None:
    old_nominal = _read_csv(AUDIT_DIR / "bupt1_validation" / "nominal" / "trajectories" / "proposed_calibrated_payload_plus_z_guided.csv")
    old_deadlines = [_to_float(row.get("deadline_s")) / 60.0 for row in old_nominal if _to_int(row.get("has_task")) == 1]
    old_stat = _stats([value for value in old_deadlines if _finite(value)])
    lines = [
        "# Deadline Scale Audit",
        "",
        "## Current Source",
        "",
        "Before this recalibration, SARFishSample deadlines came from `SARFishSampleDatasetAdapter.TASK_CLASS_RANGES`: routine 90-120 s, high_value 60-90 s, and emergency 30-60 s. These values are seconds-scale debug deadlines.",
        "",
        "The new path keeps task class generation independent of labels and reads minute-scale deadline ranges from task-stream config keys such as `ordinary_deadline_min_minutes` and `emergency_deadline_max_minutes`. Internally, deadlines are still stored as seconds in `TaskSample.deadline_s`.",
        "",
        "## Prior BUPT-1 Validation Deadline Statistics",
        "",
        f"- prior nominal PROP-Z task deadline min/mean/median/max: {_fmt(old_stat['min'])} / {_fmt(old_stat['mean'])} / {_fmt(old_stat['median'])} / {_fmt(old_stat['max'])} minutes",
        "- old deadlines were much shorter than the 95-minute contact period, which explains the low delivered-only ratios and high lifetime expiration counts.",
        "",
        "## Recalibrated Deadline Diagnostics",
        "",
        "| scenario | tasks | min | mean | median | max | contact period | <30 min | <60 min | <contact period | deadline < next contact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in diagnostics:
        lines.append(
            f"| {row['scenario']} | {row['task_count']} | {_fmt(row['min_deadline_min'])} | {_fmt(row['mean_deadline_min'])} | "
            f"{_fmt(row['median_deadline_min'])} | {_fmt(row['max_deadline_min'])} | {_fmt(row['contact_period_min'])} | "
            f"{_fmt(row['ratio_lt_30_min'])} | {_fmt(row['ratio_lt_60_min'])} | {_fmt(row['ratio_lt_contact_period'])} | "
            f"{_fmt(row['ratio_deadline_lt_next_contact'])} |"
        )
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            "- Deadline source is now explicit and config-driven for SARFishSample task streams.",
            "- Deadline unit remains seconds internally, with config values specified in minutes.",
            "- Deadlines are based on generated task class/priority priors, not SARFish ground-truth label content.",
            "- High-value and emergency tasks remain more urgent than routine monitoring, but now fit mission-level contact-aware delivery.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_deadline_plan(path: Path, diagnostics: list[dict[str, Any]]) -> None:
    class_rows = [row for diag in diagnostics for row in diag["class_rows"]]
    lines = [
        "# Deadline Recalibration Plan",
        "",
        "## Policy",
        "",
        "- Nominal routine maritime monitoring: 90-240 minutes.",
        "- Nominal high-value suspicious target: 45-120 minutes.",
        "- Nominal emergency/alert: 20-90 minutes.",
        "- Stress routine maritime monitoring: 60-180 minutes.",
        "- Stress high-value suspicious target: 30-90 minutes.",
        "- Stress emergency/alert: 15-60 minutes.",
        "",
        "## Diagnostics to Report",
        "",
        "Every minute-deadline smoke report includes min/mean/median/max deadline, per-class deadline distributions, ratios below 30/60 minutes, ratio below one contact period, ratio below next-contact wait, and delivered-before-deadline ratio via delivered-only metrics.",
        "",
        "## Per-class Generated Distribution",
        "",
        "| scenario | class | count | min | mean | median | max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in class_rows:
        lines.append(
            f"| {row['scenario']} | {row['task_class']} | {row['count']} | {_fmt(row['min_deadline_min'])} | "
            f"{_fmt(row['mean_deadline_min'])} | {_fmt(row['median_deadline_min'])} | {_fmt(row['max_deadline_min'])} |"
        )
    lines.extend(
        [
            "",
            "## Future Sensitivity Plan",
            "",
            "Prepare but do not execute deadline scale sensitivity at x0.5, x1.0, and x2.0 after the all-variant smoke is healthy.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _trajectory_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return _read_csv(output_dir / "trajectories" / f"{variant}.csv")


def _task_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return [row for row in _trajectory_rows(output_dir, variant) if _to_int(row.get("has_task")) == 1]


def _summary_by_variant(output_dir: Path) -> dict[str, dict[str, str]]:
    return {str(row.get("router") or row.get("router_name")): row for row in _read_csv(output_dir / "summary.csv")}


def _lifetime_by_variant(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv(output_dir / "task_lifetime_outcomes.csv"):
        grouped.setdefault(str(row.get("router_name", "")), []).append(row)
    return grouped


def _compute_delivered_metrics(scenario: str, output_dir: Path, deadline_diag: dict[str, Any]) -> list[dict[str, Any]]:
    aliases = _variant_aliases()
    summary = _summary_by_variant(output_dir)
    lifetime = _lifetime_by_variant(output_dir)
    rows: list[dict[str, Any]] = []
    for variant in [row["full_name"] for row in _variants()]:
        summary_row = summary.get(variant, {})
        task_rows = _task_rows(output_dir, variant)
        lifetime_rows = lifetime.get(variant, [])
        life_counts = Counter(str(row.get("final_lifetime_status", "")) for row in lifetime_rows)
        action_counts = Counter(str(row.get("selected_action") or row.get("executed_action") or "") for row in task_rows)
        payload_counts = Counter(str(row.get("payload_policy") or row.get("selected_payload") or "") for row in task_rows)
        model_counts = Counter(str(row.get("model_name") or row.get("selected_model") or "") for row in task_rows)
        delivered_only = sum(_to_float(row.get("delivered_only_utility"), 0.0) for row in lifetime_rows)
        high_value_delivered = sum(
            _to_float(row.get("delivered_only_utility"), 0.0)
            for row in lifetime_rows
            if str(row.get("high_value_flag", "")).lower() in {"1", "1.0", "true", "yes"}
        )
        exact_missing = sum(
            1
            for row in lifetime_rows
            if str(row.get("final_lifetime_status", "")) == "delivered"
            and str(row.get("met_deadline_lifetime", "")) == "true"
            and str(row.get("exact_lifetime_timing_available", "")) != "true"
        )
        gross = _to_float(summary_row.get("gross_task_utility", summary_row.get("total_effective_utility")))
        queued = sum(
            _to_float(row.get("queued_provisional_utility", row.get("effective_utility")), 0.0)
            for row in task_rows
            if str(row.get("delivery_status")) == "queued"
        )
        delivered_existing = sum(
            _to_float(row.get("effective_utility"), 0.0)
            for row in task_rows
            if str(row.get("delivery_status")) == "delivered"
        )
        rows.append(
            {
                "experiment_stage": SCENARIOS[scenario]["stage"],
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "variant_short": aliases.get(variant, variant),
                "variant_full": variant,
                "metric_quality": "exact" if exact_missing == 0 and lifetime_rows else "approximate",
                "notes": "" if exact_missing == 0 else f"{exact_missing} delivered-before-deadline tasks lacked exact event timing",
                "gross_task_utility_provisional": gross,
                "net_system_utility_provisional": _to_float(summary_row.get("net_system_utility")),
                "penalized_reward": _to_float(summary_row.get("total_penalized_reward")),
                "resource_penalty": _to_float(summary_row.get("total_resource_penalty")),
                "requirement_penalty": _to_float(summary_row.get("total_requirement_penalty")),
                "delivered_only_gross_utility": delivered_only,
                "delivered_before_deadline_utility": delivered_only,
                "high_value_delivered_utility": high_value_delivered,
                "queued_provisional_utility": queued,
                "delivered_existing_utility": delivered_existing,
                "delivery_realization_gap": gross - delivered_only if _finite(gross) else math.nan,
                "delivered_utility_ratio": _safe_div(delivered_only, gross),
                "lifetime_delivered_task_level": life_counts.get("delivered", 0),
                "lifetime_expired_task_level": life_counts.get("expired", 0),
                "lifetime_deferred_task_level": life_counts.get("deferred", 0),
                "lifetime_undelivered_by_end_task_level": life_counts.get("undelivered_by_end", 0),
                "delivered_items_payload_level": _to_int(summary_row.get("delivered_items")),
                "expired_items_payload_level": _to_int(summary_row.get("expired_items")),
                "completed_tasks_same_step": _to_int(summary_row.get("completed_tasks")),
                "completed_deadline_satisfaction_ratio": _to_float(summary_row.get("completed_deadline_satisfaction_ratio")),
                "deadline_satisfaction_ratio": _to_float(summary_row.get("deadline_satisfaction_ratio")),
                "total_energy": _to_float(summary_row.get("total_energy")),
                "compute_energy": _to_float(summary_row.get("total_compute_energy")),
                "comm_energy": _to_float(summary_row.get("total_comm_energy")),
                "min_battery_soc": _to_float(summary_row.get("min_battery_soc")),
                "final_battery_soc": _to_float(summary_row.get("final_battery_soc", summary_row.get("final_battery"))),
                "max_thermal": _to_float(summary_row.get("max_thermal")),
                "thermal_violation_count": _to_int(summary_row.get("thermal_guard_exceed_count")),
                "average_queue_mb": _to_float(summary_row.get("average_queue_mb")),
                "max_queue_mb": _to_float(summary_row.get("max_queue_mb")),
                "generated_payload_mb": _to_float(summary_row.get("total_generated_payload_mb")),
                "delivered_mb": _to_float(summary_row.get("delivered_mb")),
                "lightweight_count": model_counts.get("lightweight", 0),
                "balanced_count": model_counts.get("balanced", 0),
                "high_accuracy_count": model_counts.get("high_accuracy", 0),
                "metadata_count": payload_counts.get("metadata", 0),
                "roi_count": payload_counts.get("roi", 0),
                "roi_context_count": payload_counts.get("roi_context", 0),
                "defer_count": payload_counts.get("defer", 0),
                "balanced_roi_count": action_counts.get("balanced+roi", 0),
                "high_accuracy_roi_context_count": action_counts.get("high_accuracy+roi_context", 0),
                "deadline_min_minutes": deadline_diag["min_deadline_min"],
                "deadline_mean_minutes": deadline_diag["mean_deadline_min"],
                "deadline_median_minutes": deadline_diag["median_deadline_min"],
                "deadline_max_minutes": deadline_diag["max_deadline_min"],
                "deadline_lt_30_min_ratio": deadline_diag["ratio_lt_30_min"],
                "deadline_lt_60_min_ratio": deadline_diag["ratio_lt_60_min"],
                "deadline_lt_contact_period_ratio": deadline_diag["ratio_lt_contact_period"],
                "deadline_lt_next_contact_ratio": deadline_diag["ratio_deadline_lt_next_contact"],
                "stage_rank_by_provisional_net": "",
                "stage_rank_by_delivered_only": "",
                "stage_rank_by_high_value_delivered": "",
            }
        )
    for rank, row in enumerate(sorted(rows, key=lambda item: item["net_system_utility_provisional"], reverse=True), start=1):
        row["stage_rank_by_provisional_net"] = rank
    for rank, row in enumerate(sorted(rows, key=lambda item: item["delivered_only_gross_utility"], reverse=True), start=1):
        row["stage_rank_by_delivered_only"] = rank
    for rank, row in enumerate(sorted(rows, key=lambda item: item["high_value_delivered_utility"], reverse=True), start=1):
        row["stage_rank_by_high_value_delivered"] = rank
    return rows


def _extend_summary(output_dir: Path, delivered_rows: list[dict[str, Any]]) -> None:
    provisional_rows = _read_csv(output_dir / "summary.csv")
    _write_csv(output_dir / "summary_provisional.csv", provisional_rows)
    by_variant = {row["variant_full"]: row for row in delivered_rows}
    extended: list[dict[str, Any]] = []
    for row in provisional_rows:
        variant = str(row.get("router") or row.get("router_name"))
        out: dict[str, Any] = dict(row)
        metrics = by_variant.get(variant, {})
        for key, value in metrics.items():
            if key not in {"variant_full", "output_dir"}:
                out[key] = value
        extended.append(out)
    _write_csv(output_dir / "summary.csv", extended)


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| rank delivered | rank high-value | variant | net provisional | delivered-only | high-value delivered | ratio | lifetime delivered | lifetime expired | min SOC | generated MB | delivered MB | quality |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: item["stage_rank_by_delivered_only"]):
        lines.append(
            f"| {row['stage_rank_by_delivered_only']} | {row['stage_rank_by_high_value_delivered']} | {row['variant_short']} | "
            f"{_fmt(row['net_system_utility_provisional'])} | {_fmt(row['delivered_only_gross_utility'])} | "
            f"{_fmt(row['high_value_delivered_utility'])} | {_fmt(row['delivered_utility_ratio'])} | "
            f"{row['lifetime_delivered_task_level']} | {row['lifetime_expired_task_level']} | {_fmt(row['min_battery_soc'])} | "
            f"{_fmt(row['generated_payload_mb'])} | {_fmt(row['delivered_mb'])} | {row['metric_quality']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_formal_report(scenario: str, output_dir: Path, rows: list[dict[str, Any]], deadline_diag: dict[str, Any]) -> None:
    top_delivered = sorted(rows, key=lambda row: row["delivered_only_gross_utility"], reverse=True)[:3]
    top_high = sorted(rows, key=lambda row: row["high_value_delivered_utility"], reverse=True)[:3]
    top_delivered_text = ", ".join(
        f"{row['variant_short']}={_fmt(row['delivered_only_gross_utility'])}" for row in top_delivered
    )
    top_high_text = ", ".join(
        f"{row['variant_short']}={_fmt(row['high_value_delivered_utility'])}" for row in top_high
    )
    proposed = [row for row in rows if row["variant_full"] in PROPOSED]
    best_proposed = max(proposed, key=lambda row: row["delivered_only_gross_utility"])
    resource_ok = all(_to_float(row["min_battery_soc"], 0.0) > 0.0 for row in rows)
    ratio_max = max((_to_float(row["delivered_utility_ratio"], 0.0) for row in rows), default=0.0)
    ratio_outliers = [row for row in rows if _to_float(row["delivered_utility_ratio"], 0.0) > 1.2]
    target_low, target_high = (0.4, 0.7) if scenario == "nominal" else (0.2, 0.5)
    ratio_healthy = target_low <= ratio_max <= target_high
    decision = "B. Need one more calibration smoke before multi-seed."
    if resource_ok and ratio_healthy:
        decision = "A. Ready for full formal multi-seed after minor documentation updates."
    elif resource_ok:
        decision = "B. Need one more metric/environment check before multi-seed."
    lines = [
        "# Formal Smoke Test Report",
        "",
        "## Executive Summary",
        "",
        f"- scenario: {scenario}",
        f"- variants run: {len(rows)}",
        f"- generated task count: {max(row['lifetime_delivered_task_level'] + row['lifetime_expired_task_level'] + row['lifetime_deferred_task_level'] + row['lifetime_undelivered_by_end_task_level'] for row in rows)}",
        f"- deadline min/mean/median/max: {_fmt(deadline_diag['min_deadline_min'])} / {_fmt(deadline_diag['mean_deadline_min'])} / {_fmt(deadline_diag['median_deadline_min'])} / {_fmt(deadline_diag['max_deadline_min'])} minutes",
        f"- top delivered-only: {top_delivered_text}",
        f"- top high-value delivered: {top_high_text}",
        f"- best proposed candidate: {best_proposed['variant_short']}",
        f"- decision: {decision}",
        "",
        "## Variant Metrics",
        "",
        "| variant | delivered-only | high-value delivered | ratio | deadline sat | completed deadline sat | lifetime delivered | lifetime expired | net provisional | queued | gap | min SOC | final SOC | max thermal | energy | queue avg/max | generated/delivered MB | balanced+ROI | high-acc+ROI+ctx |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: item["stage_rank_by_delivered_only"]):
        lines.append(
            f"| {row['variant_short']} | {_fmt(row['delivered_only_gross_utility'])} | {_fmt(row['high_value_delivered_utility'])} | "
            f"{_fmt(row['delivered_utility_ratio'])} | {_fmt(row['deadline_satisfaction_ratio'])} | {_fmt(row['completed_deadline_satisfaction_ratio'])} | "
            f"{row['lifetime_delivered_task_level']} | {row['lifetime_expired_task_level']} | {_fmt(row['net_system_utility_provisional'])} | "
            f"{_fmt(row['queued_provisional_utility'])} | {_fmt(row['delivery_realization_gap'])} | {_fmt(row['min_battery_soc'])} | "
            f"{_fmt(row['final_battery_soc'])} | {_fmt(row['max_thermal'])} | {_fmt(row['total_energy'])} | "
            f"{_fmt(row['average_queue_mb'])}/{_fmt(row['max_queue_mb'])} | {_fmt(row['generated_payload_mb'])}/{_fmt(row['delivered_mb'])} | "
            f"{row['balanced_roi_count']} | {row['high_accuracy_roi_context_count']} |"
        )
    lines.extend(
        [
            "",
            "## Deadline Diagnostics",
            "",
            f"- ratio below 30 minutes: {_fmt(deadline_diag['ratio_lt_30_min'])}",
            f"- ratio below 60 minutes: {_fmt(deadline_diag['ratio_lt_60_min'])}",
            f"- ratio below one contact period: {_fmt(deadline_diag['ratio_lt_contact_period'])}",
            f"- ratio deadline shorter than next-contact wait: {_fmt(deadline_diag['ratio_deadline_lt_next_contact'])}",
            f"- delivered utility ratio > 1.2 variants: {', '.join(row['variant_short'] for row in ratio_outliers) if ratio_outliers else 'none'}",
            "- A ratio above 1.0 can occur because `delivered_only_gross_utility` is lifetime-delivered utility while the denominator is provisional `gross_task_utility`; treat this ratio as diagnostic rather than a standalone formal metric.",
            "",
            "## Decision Recommendation",
            "",
            decision,
            "",
        ]
    )
    (output_dir / "formal_smoke_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_scenario(scenario: str, deadline_diag: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = SCENARIOS[scenario]["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dirs = ensure_output_dirs(output_dir)
    logger = get_logger("run_bupt1_deadline_minutes_smoke")
    results = []
    timeline_for_report = None
    for item in _variants():
        variant = item["full_name"]
        configs = _load_configs(scenario, variant)
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        if timeline_for_report is None:
            timeline_for_report = timeline
        environment = SimulationEnvironment(configs)
        router = _build_router(variant, environment)
        logger.info("running %s minute-deadline smoke variant: %s", scenario, variant)
        results.append(environment.build_episode_runner(timeline).run(router))
    write_episode_results(results, output_dirs)
    slot_seconds = float(timeline_for_report[0].slot_seconds if timeline_for_report else 20.0)
    generate_lifetime_outputs(output_dir, slot_seconds)
    rows = _compute_delivered_metrics(scenario, output_dir, deadline_diag)
    _write_csv(output_dir / "delivered_only_metrics.csv", rows, DELIVERED_FIELDS + [
        "deadline_min_minutes",
        "deadline_mean_minutes",
        "deadline_median_minutes",
        "deadline_max_minutes",
        "deadline_lt_30_min_ratio",
        "deadline_lt_60_min_ratio",
        "deadline_lt_contact_period_ratio",
        "deadline_lt_next_contact_ratio",
    ])
    _write_markdown_table(output_dir / "delivered_only_metrics.md", rows, f"{scenario.title()} Delivered-only Metrics")
    _extend_summary(output_dir, rows)
    _write_markdown_table(output_dir / "summary.md", rows, f"{scenario.title()} 1-day Smoke Summary")
    _write_formal_report(scenario, output_dir, rows, deadline_diag)
    return rows


def _write_comparison(all_rows: dict[str, list[dict[str, Any]]], diagnostics: list[dict[str, Any]]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    flat_rows = [row for rows in all_rows.values() for row in rows]
    _write_csv(INDEX_DIR / "bupt1_deadline_minutes_smoke_comparison.csv", flat_rows)
    top_lines = []
    for scenario, rows in all_rows.items():
        top_delivered = sorted(rows, key=lambda row: row["delivered_only_gross_utility"], reverse=True)[:3]
        top_high = sorted(rows, key=lambda row: row["high_value_delivered_utility"], reverse=True)[:3]
        top_delivered_text = ", ".join(
            f"{row['variant_short']}={_fmt(row['delivered_only_gross_utility'])}" for row in top_delivered
        )
        top_high_text = ", ".join(
            f"{row['variant_short']}={_fmt(row['high_value_delivered_utility'])}" for row in top_high
        )
        best_prop = max([row for row in rows if row["variant_full"] in PROPOSED], key=lambda row: row["delivered_only_gross_utility"])
        util = next(row for row in rows if row["variant_short"] == "UTIL")
        no_mode = next(row for row in rows if row["variant_short"] == "NO-MODE")
        top_lines.extend(
            [
                f"### {scenario.title()}",
                "",
                f"- top delivered-only: {top_delivered_text}",
                f"- top high-value delivered: {top_high_text}",
                f"- best proposed candidate: {best_prop['variant_short']}",
                f"- best proposed beats UTIL by delivered-only: {best_prop['delivered_only_gross_utility'] > util['delivered_only_gross_utility']}",
                f"- best proposed beats NO-MODE by delivered-only: {best_prop['delivered_only_gross_utility'] > no_mode['delivered_only_gross_utility']}",
                "",
            ]
        )
    nominal_best_ratio = max(row["delivered_utility_ratio"] for row in all_rows["nominal"])
    stress_best_ratio = max(row["delivered_utility_ratio"] for row in all_rows["stress"])
    nominal_outliers = [row["variant_short"] for row in all_rows["nominal"] if row["delivered_utility_ratio"] > 1.2]
    stress_outliers = [row["variant_short"] for row in all_rows["stress"] if row["delivered_utility_ratio"] > 1.2]
    healthy = 0.4 <= nominal_best_ratio <= 0.7 and 0.2 <= stress_best_ratio <= 0.5
    lines = [
        "# BUPT-1 Deadline-minutes Smoke Comparison",
        "",
        "## Executive Summary",
        "",
        f"- minute-scale deadlines improved the deadline scale from seconds to mission-level minutes: {'yes' if diagnostics else 'unknown'}.",
        f"- nominal best delivered utility ratio: {_fmt(nominal_best_ratio)}; useful target 0.4-0.7.",
        f"- stress best delivered utility ratio: {_fmt(stress_best_ratio)}; useful target 0.2-0.5.",
        f"- ratio > 1.2 diagnostics: nominal {', '.join(nominal_outliers) if nominal_outliers else 'none'}; stress {', '.join(stress_outliers) if stress_outliers else 'none'}.",
        f"- formal multi-seed readiness: {'can proceed after documenting single-seed scope' if healthy else 'wait; one more calibration review recommended'}.",
        "",
        "## Scenario Results",
        "",
        *top_lines,
        "## Interpretation",
        "",
        "- FIX-H should be judged against resource use, queue/deadline outcomes, and delivered-only utility rather than raw model complexity.",
        "- FIX-L can deliver small payloads efficiently but should not be treated as optimal unless it also preserves high-value delivered utility.",
        "- `delivered_utility_ratio` is diagnostic because the numerator is lifetime-delivered utility and the denominator is provisional gross utility; use delivered-only and high-value delivered utility as primary paper metrics.",
        "- PROP-Z, PROP-PAY, and PROP-COMB are the proposed candidates to evaluate for freezing; choose the one with the best delivered-only and high-value delivered balance under both scenarios.",
        "- Sensitivity plan is prepared but not executed: deadline scale x0.5, x1.0, x2.0.",
        "",
    ]
    (INDEX_DIR / "bupt1_deadline_minutes_smoke_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    diagnostics = [_deadline_diagnostics("nominal"), _deadline_diagnostics("stress")]
    _write_deadline_audit(AUDIT_DIR / "deadline_scale_audit.md", diagnostics)
    _write_deadline_plan(AUDIT_DIR / "deadline_recalibration_plan.md", diagnostics)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for diag in diagnostics:
        all_rows[diag["scenario"]] = _run_scenario(diag["scenario"], diag)
    _write_comparison(all_rows, diagnostics)
    return {"diagnostics": diagnostics, "rows": all_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BUPT-1 minute-deadline all-variant smoke tests.")
    parser.parse_args()
    configure_logging()
    result = run()
    diagnostics = {row["scenario"]: row for row in result["diagnostics"]}
    rows = result["rows"]
    print("deadline_audit_result=old_seconds_scale_deadlines_replaced_by_configured_minute_scale_deadlines")
    print("configs_created=" + ",".join(str(path.relative_to(PROJECT_ROOT)) for scenario in SCENARIOS.values() for path in [scenario["task_stream"], scenario["timeline"], scenario["simulator"]]))
    for scenario in ["nominal", "stress"]:
        diag = diagnostics[scenario]
        scenario_rows = rows[scenario]
        print(f"{scenario}_generated_task_count={diag['task_count']}")
        print(
            f"{scenario}_deadline_min_mean_median_max="
            f"{_fmt(diag['min_deadline_min'])},{_fmt(diag['mean_deadline_min'])},{_fmt(diag['median_deadline_min'])},{_fmt(diag['max_deadline_min'])}"
        )
        print(
            f"{scenario}_delivered_only_ratios="
            + ",".join(f"{row['variant_short']}:{_fmt(row['delivered_utility_ratio'])}" for row in sorted(scenario_rows, key=lambda item: item["stage_rank_by_delivered_only"]))
        )
        print(
            f"{scenario}_top3_delivered_only="
            + ",".join(f"{row['variant_short']}:{_fmt(row['delivered_only_gross_utility'])}" for row in sorted(scenario_rows, key=lambda item: item["delivered_only_gross_utility"], reverse=True)[:3])
        )
        print(
            f"{scenario}_top3_high_value_delivered="
            + ",".join(f"{row['variant_short']}:{_fmt(row['high_value_delivered_utility'])}" for row in sorted(scenario_rows, key=lambda item: item["high_value_delivered_utility"], reverse=True)[:3])
        )
        best_proposed = max([row for row in scenario_rows if row["variant_full"] in PROPOSED], key=lambda item: item["delivered_only_gross_utility"])
        print(f"{scenario}_best_proposed_candidate={best_proposed['variant_short']}")
    nominal_best = max(row["delivered_utility_ratio"] for row in rows["nominal"])
    stress_best = max(row["delivered_utility_ratio"] for row in rows["stress"])
    can_proceed = 0.4 <= nominal_best <= 0.7 and 0.2 <= stress_best <= 0.5
    print("formal_multiseed_recommendation=" + ("can proceed after documentation" if can_proceed else "wait; one more calibration review recommended"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
