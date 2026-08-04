"""Run a 1-day single-seed SARFish smoke test with delivered-only reporting."""

from __future__ import annotations

import argparse
import copy
import csv
import math
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


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sarfish_1day_smoke"
INDEX_DIR = PROJECT_ROOT / "outputs" / "_index"
STAGE_NAME = "sarfish_1day_smoke"

ACTION_FOCUS = (
    "lightweight+metadata",
    "balanced+metadata",
    "balanced+roi",
    "high_accuracy+roi_context",
    "delay/store",
)

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
    "beats_utility_rule_by_provisional_net",
    "beats_utility_rule_by_delivered_only",
    "beats_no_mode_by_delivered_only",
    "stage_rank_by_provisional_net",
    "stage_rank_by_delivered_only",
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


def _load_variants() -> list[dict[str, str]]:
    data = load_yaml(PROJECT_ROOT / "configs" / "smoke_test_variants.yaml")
    variants = data.get("variants", [])
    if not variants:
        raise ValueError("configs/smoke_test_variants.yaml has no variants")
    return [{"short_name": str(row["short_name"]), "full_name": str(row["full_name"])} for row in variants]


def _load_ablation_cfg() -> dict[str, object]:
    return load_yaml(PROJECT_ROOT / "configs" / "payload_profile_ablation.yaml")


def _load_repair_cfg(variant: str) -> dict[str, object]:
    cfg = _load_ablation_cfg()
    repair = copy.deepcopy(cfg["policy_repair_defaults"])
    repair.update(copy.deepcopy(cfg.get("variant_overrides", {}).get(variant, {})))
    return repair


def _load_configs(variant: str) -> dict[str, dict]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    configs["timeline"] = load_yaml(
        PROJECT_ROOT / "configs" / "timeline_sarfish_energy_shadow_payload_calibrated_1day_smoke_20s.yaml"
    )
    configs["simulator"] = load_yaml(
        PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_payload_calibrated_1day_smoke_20s.yaml"
    )
    configs["resource"] = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")
    configs["contact"] = load_yaml(PROJECT_ROOT / "configs" / "contact_sarfish_physical_20s.yaml")
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
    raise ValueError(f"Unsupported smoke variant: {variant}")


def _variant_aliases() -> dict[str, str]:
    return {row["full_name"]: row["short_name"] for row in _load_variants()}


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


def _action_counts(rows: list[dict[str, str]]) -> Counter:
    return Counter(str(row.get("selected_action") or row.get("executed_action") or "") for row in rows)


def _payload_counts(rows: list[dict[str, str]]) -> Counter:
    return Counter(str(row.get("payload_policy") or row.get("selected_payload") or "") for row in rows)


def _model_counts(rows: list[dict[str, str]]) -> Counter:
    return Counter(str(row.get("model_name") or row.get("selected_model") or "") for row in rows)


def _lifetime_counts(rows: list[dict[str, str]]) -> Counter:
    return Counter(str(row.get("final_lifetime_status", "")) for row in rows)


def compute_delivered_metrics(output_dir: Path) -> list[dict[str, Any]]:
    aliases = _variant_aliases()
    summary = _summary_by_variant(output_dir)
    lifetime = _lifetime_by_variant(output_dir)
    rows: list[dict[str, Any]] = []
    variants = [row["full_name"] for row in _load_variants()]
    for variant in variants:
        summary_row = summary.get(variant, {})
        task_rows = _task_rows(output_dir, variant)
        lifetime_rows = lifetime.get(variant, [])
        life_counts = _lifetime_counts(lifetime_rows)
        action_counts = _action_counts(task_rows)
        payload_counts = _payload_counts(task_rows)
        model_counts = _model_counts(task_rows)

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
        metric_quality = "exact" if exact_missing == 0 and lifetime_rows else "approximate"
        notes = "" if metric_quality == "exact" else f"{exact_missing} delivered-before-deadline tasks lacked exact event timing"

        gross = _to_float(summary_row.get("gross_task_utility", summary_row.get("total_effective_utility")))
        net = _to_float(summary_row.get("net_system_utility"))
        queued = sum(_to_float(row.get("queued_provisional_utility", row.get("effective_utility")), 0.0) for row in task_rows if str(row.get("delivery_status")) == "queued")
        delivered_existing = sum(_to_float(row.get("effective_utility"), 0.0) for row in task_rows if str(row.get("delivery_status")) == "delivered")
        delivery_gap = gross - delivered_only if _finite(gross) else math.nan
        delivered_ratio = _safe_div(delivered_only, gross)

        rows.append(
            {
                "experiment_stage": STAGE_NAME,
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "variant_short": aliases.get(variant, variant),
                "variant_full": variant,
                "metric_quality": metric_quality,
                "notes": notes,
                "gross_task_utility_provisional": gross,
                "net_system_utility_provisional": net,
                "penalized_reward": _to_float(summary_row.get("total_penalized_reward")),
                "resource_penalty": _to_float(summary_row.get("total_resource_penalty")),
                "requirement_penalty": _to_float(summary_row.get("total_requirement_penalty")),
                "delivered_only_gross_utility": delivered_only,
                "delivered_before_deadline_utility": delivered_only,
                "high_value_delivered_utility": high_value_delivered,
                "queued_provisional_utility": queued,
                "delivered_existing_utility": delivered_existing,
                "delivery_realization_gap": delivery_gap,
                "delivered_utility_ratio": delivered_ratio,
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
                "beats_utility_rule_by_provisional_net": "",
                "beats_utility_rule_by_delivered_only": "",
                "beats_no_mode_by_delivered_only": "",
                "stage_rank_by_provisional_net": "",
                "stage_rank_by_delivered_only": "",
            }
        )
    utility_net = next((row["net_system_utility_provisional"] for row in rows if row["variant_full"] == "utility_rule"), math.nan)
    utility_delivered = next((row["delivered_only_gross_utility"] for row in rows if row["variant_full"] == "utility_rule"), math.nan)
    no_mode_delivered = next((row["delivered_only_gross_utility"] for row in rows if row["variant_full"] == "no_mode"), math.nan)
    for row in rows:
        row["beats_utility_rule_by_provisional_net"] = "" if not _finite(utility_net) else int(row["net_system_utility_provisional"] > utility_net)
        row["beats_utility_rule_by_delivered_only"] = "" if not _finite(utility_delivered) else int(row["delivered_only_gross_utility"] > utility_delivered)
        row["beats_no_mode_by_delivered_only"] = "" if not _finite(no_mode_delivered) else int(row["delivered_only_gross_utility"] > no_mode_delivered)
    for rank, row in enumerate(sorted(rows, key=lambda item: item["net_system_utility_provisional"], reverse=True), start=1):
        row["stage_rank_by_provisional_net"] = rank
    for rank, row in enumerate(sorted(rows, key=lambda item: item["delivered_only_gross_utility"], reverse=True), start=1):
        row["stage_rank_by_delivered_only"] = rank
    return rows


def write_markdown_table(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| rank net | rank delivered | short | full | net provisional | delivered-only | ratio | queued | lifetime delivered | lifetime expired | energy | generated MB | delivered MB | quality |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: item["stage_rank_by_provisional_net"]):
        lines.append(
            "| {rank_net} | {rank_del} | {short} | {full} | {net} | {delivered} | {ratio} | {queued} | {life_del} | {life_exp} | {energy} | {gen} | {down} | {quality} |".format(
                rank_net=row["stage_rank_by_provisional_net"],
                rank_del=row["stage_rank_by_delivered_only"],
                short=row["variant_short"],
                full=row["variant_full"],
                net=_fmt(row["net_system_utility_provisional"]),
                delivered=_fmt(row["delivered_only_gross_utility"]),
                ratio=_fmt(row["delivered_utility_ratio"]),
                queued=_fmt(row["queued_provisional_utility"]),
                life_del=row["lifetime_delivered_task_level"],
                life_exp=row["lifetime_expired_task_level"],
                energy=_fmt(row["total_energy"]),
                gen=_fmt(row["generated_payload_mb"]),
                down=_fmt(row["delivered_mb"]),
                quality=row["metric_quality"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extend_summary(output_dir: Path, delivered_rows: list[dict[str, Any]]) -> None:
    summary_path = output_dir / "summary.csv"
    provisional_rows = _read_csv(summary_path)
    _write_csv(output_dir / "summary_provisional.csv", provisional_rows)
    delivered_by_variant = {row["variant_full"]: row for row in delivered_rows}
    extended: list[dict[str, Any]] = []
    for row in provisional_rows:
        variant = str(row.get("router") or row.get("router_name"))
        out: dict[str, Any] = dict(row)
        metrics = delivered_by_variant.get(variant, {})
        for key in DELIVERED_FIELDS:
            if key not in {"variant_full", "output_dir"}:
                out[key] = metrics.get(key, out.get(key, ""))
        extended.append(out)
    _write_csv(summary_path, extended)


def write_smoke_report(output_dir: Path, delivered_rows: list[dict[str, Any]]) -> None:
    top_net = sorted(delivered_rows, key=lambda row: row["net_system_utility_provisional"], reverse=True)[:3]
    top_delivered = sorted(delivered_rows, key=lambda row: row["delivered_only_gross_utility"], reverse=True)[:3]
    proposed = [
        row for row in delivered_rows
        if row["variant_full"] in {
            "proposed_calibrated_payload_plus_z_guided",
            "proposed_calibrated_payload_profile",
            "proposed_payload_repaired_combined",
            "proposed_calibrated_payload_plus_cost_aware_ucb",
        }
    ]
    best_proposed = max(proposed, key=lambda row: row["delivered_only_gross_utility"]) if proposed else None
    exact_count = sum(1 for row in delivered_rows if row["metric_quality"] == "exact")
    generated_tasks = max(
        (
            row["lifetime_delivered_task_level"]
            + row["lifetime_expired_task_level"]
            + row["lifetime_deferred_task_level"]
            + row["lifetime_undelivered_by_end_task_level"]
            for row in delivered_rows
        ),
        default=0,
    )
    max_ratio = max((_to_float(row["delivered_utility_ratio"], 0.0) for row in delivered_rows), default=0.0)
    resource_infeasible = any(
        _to_float(row.get("min_battery_soc"), 1.0) <= 0.18
        or _to_float(row.get("max_thermal"), 0.0) > 0.92
        for row in delivered_rows
    )
    decision = "B. Need one more metric/logging fix before multi-seed."
    if resource_infeasible:
        decision = "D. Need environment/contact/deadline stress recalibration before multi-seed."
    elif exact_count == len(delivered_rows) and max_ratio <= 1.2:
        decision = "A. Ready for full formal multi-seed after minor documentation updates."

    def list_rows(rows: list[dict[str, Any]], field: str) -> str:
        return ", ".join(f"{row['variant_short']}={_fmt(row[field])}" for row in rows)

    lines = [
        "# Formal Smoke Test Report",
        "",
        "## Executive Summary",
        "",
        f"- variants run: {len(delivered_rows)}",
        f"- generated task count: {generated_tasks}",
        f"- exact delivered-only rows: {exact_count}/{len(delivered_rows)}",
        f"- top provisional net: {list_rows(top_net, 'net_system_utility_provisional')}",
        f"- top delivered-only: {list_rows(top_delivered, 'delivered_only_gross_utility')}",
        f"- best proposed candidate by delivered-only utility: {best_proposed['variant_short'] if best_proposed else 'none'}",
        f"- decision: {decision}",
        "",
        "## Interpretation",
        "",
        "- `net_system_utility_provisional` is expected/provisional system utility, not strict delivered utility.",
        "- `delivered_only_gross_utility`, `high_value_delivered_utility`, and lifetime delivered/expired counts are the primary paper-readiness metrics.",
        "- `queued_provisional_utility` and `delivery_realization_gap` diagnose generated information that has not become delivered task utility.",
        "- This single-seed smoke test is not a final paper result.",
        "",
        "## Variant Table",
        "",
        "| short | net provisional | delivered-only | high-value delivered | ratio | queued | lifetime delivered / expired / deferred / end | energy | max thermal | min SOC | balanced+ROI | high-acc+ROI+ctx |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(delivered_rows, key=lambda item: item["stage_rank_by_delivered_only"]):
        lines.append(
            f"| {row['variant_short']} | {_fmt(row['net_system_utility_provisional'])} | {_fmt(row['delivered_only_gross_utility'])} | "
            f"{_fmt(row['high_value_delivered_utility'])} | {_fmt(row['delivered_utility_ratio'])} | {_fmt(row['queued_provisional_utility'])} | "
            f"{row['lifetime_delivered_task_level']} / {row['lifetime_expired_task_level']} / {row['lifetime_deferred_task_level']} / {row['lifetime_undelivered_by_end_task_level']} | "
            f"{_fmt(row['total_energy'])} | {_fmt(row['max_thermal'])} | {_fmt(row['min_battery_soc'])} | "
            f"{row['balanced_roi_count']} | {row['high_accuracy_roi_context_count']} |"
        )
    lines.extend(
        [
            "",
            "## Sanity Check Notes",
            "",
            f"- delivered utility ratio maximum: {_fmt(max_ratio)}",
            f"- resource infeasibility detected: {str(resource_infeasible).lower()}",
            "- `completed_tasks_same_step`, `delivered_items_payload_level`, and lifetime task-level counts are reported separately in `delivered_only_metrics.csv`.",
            "- `generated_payload_mb` and `delivered_mb` remain separate lifecycle-phase MB metrics.",
            "- FIX-L should be read against delivered-only utility and high-value delivered utility, not delivery count alone.",
            "- FIX-H should be read against energy, queue/deadline failures, and low delivered-only utility under stress.",
            "- UTIL remains a strong heuristic baseline and should stay in formal comparisons.",
            "",
            "## Decision Recommendation",
            "",
            decision,
            "",
        ]
    )
    (output_dir / "formal_smoke_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_index_leaderboard(output_dir: Path, delivered_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_csv(INDEX_DIR / "leaderboard_with_delivered_only.csv")
    smoke_rows = []
    for row in delivered_rows:
        smoke_rows.append(
            {
                "experiment_stage": STAGE_NAME,
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "variant_short": row["variant_short"],
                "variant_full": row["variant_full"],
                "enabled_features": "1day_smoke,payload_calibrated,energy_shadow",
                "gross_task_utility": row["gross_task_utility_provisional"],
                "net_system_utility": row["net_system_utility_provisional"],
                "penalized_reward": row["penalized_reward"],
                "resource_penalty": row["resource_penalty"],
                "requirement_penalty": row["requirement_penalty"],
                "total_energy": row["total_energy"],
                "compute_energy": row["compute_energy"],
                "comm_energy": row["comm_energy"],
                "delivered_mb": row["delivered_mb"],
                "generated_payload_mb": row["generated_payload_mb"],
                "expired_items": row["expired_items_payload_level"],
                "lifetime_delivered": row["lifetime_delivered_task_level"],
                "lifetime_expired": row["lifetime_expired_task_level"],
                "high_accuracy_roi_context_count": row["high_accuracy_roi_context_count"],
                "balanced_roi_count": row["balanced_roi_count"],
                "metadata_count": row["metadata_count"],
                "roi_count": row["roi_count"],
                "roi_context_count": row["roi_context_count"],
                "defer_count": row["defer_count"],
                "action_similarity_to_no_mode": "",
                "stage_rank_by_net": row["stage_rank_by_provisional_net"],
                "stage_best": int(row["stage_rank_by_provisional_net"] == 1),
                "beats_utility_rule": row["beats_utility_rule_by_provisional_net"],
                "beats_no_mode": "",
                "notes": "single-seed smoke; not formal result",
                "metric_quality": row["metric_quality"],
                "delivered_only_gross_utility": row["delivered_only_gross_utility"],
                "delivered_before_deadline_utility": row["delivered_before_deadline_utility"],
                "high_value_delivered_utility": row["high_value_delivered_utility"],
                "queued_provisional_utility": row["queued_provisional_utility"],
                "delivery_realization_gap": row["delivery_realization_gap"],
                "delivered_utility_ratio": row["delivered_utility_ratio"],
                "expiration_adjusted_net_utility": "",
                "explicit_size_cost_net_utility": "",
                "lifetime_delivered_task_level": row["lifetime_delivered_task_level"],
                "lifetime_expired_task_level": row["lifetime_expired_task_level"],
            }
        )
    combined = existing + smoke_rows
    fields: list[str] = []
    for item in combined:
        for key in item:
            if key not in fields:
                fields.append(key)
    csv_path = INDEX_DIR / "leaderboard_with_1day_smoke.csv"
    _write_csv(csv_path, combined, fields)
    md_path = INDEX_DIR / "leaderboard_with_1day_smoke.md"
    lines = [
        "# Leaderboard With 1-day Smoke",
        "",
        "Includes previous indexed rows plus the single-seed smoke stage. Cross-stage comparisons remain diagnostic only.",
        "",
        "| stage | rank | short | full | net provisional | delivered-only | ratio | lifetime delivered | lifetime expired | notes |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in combined:
        lines.append(
            f"| {row.get('experiment_stage', '')} | {row.get('stage_rank_by_net', '')} | {row.get('variant_short', '')} | "
            f"{row.get('variant_full', '')} | {_fmt(_to_float(row.get('net_system_utility')))} | "
            f"{_fmt(_to_float(row.get('delivered_only_gross_utility')))} | {_fmt(_to_float(row.get('delivered_utility_ratio')))} | "
            f"{row.get('lifetime_delivered_task_level', row.get('lifetime_delivered', ''))} | "
            f"{row.get('lifetime_expired_task_level', row.get('lifetime_expired', ''))} | {row.get('notes', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def run_smoke(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dirs = ensure_output_dirs(output_dir)
    logger = get_logger("run_sarfish_1day_smoke")
    variants = _load_variants()
    results = []
    timeline_for_report = None
    for item in variants:
        variant = item["full_name"]
        configs = _load_configs(variant)
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        if timeline_for_report is None:
            timeline_for_report = timeline
        environment = SimulationEnvironment(configs)
        router = _build_router(variant, environment)
        episode_runner = environment.build_episode_runner(timeline)
        logger.info("running 1-day smoke variant: %s", variant)
        results.append(episode_runner.run(router))

    write_episode_results(results, output_dirs)
    slot_seconds = float(timeline_for_report[0].slot_seconds if timeline_for_report else 20.0)
    generate_lifetime_outputs(output_dir, slot_seconds)
    delivered_rows = compute_delivered_metrics(output_dir)
    _write_csv(output_dir / "delivered_only_metrics.csv", delivered_rows, DELIVERED_FIELDS)
    write_markdown_table(output_dir / "delivered_only_metrics.md", delivered_rows, "Delivered-only Metrics")
    _extend_summary(output_dir, delivered_rows)
    write_markdown_table(output_dir / "summary.md", delivered_rows, "1-day Smoke Summary")
    write_smoke_report(output_dir, delivered_rows)
    index_csv, index_md = write_index_leaderboard(output_dir, delivered_rows)

    task_count = max(
        (
            row["lifetime_delivered_task_level"]
            + row["lifetime_expired_task_level"]
            + row["lifetime_deferred_task_level"]
            + row["lifetime_undelivered_by_end_task_level"]
            for row in delivered_rows
        ),
        default=0,
    )
    top_net = sorted(delivered_rows, key=lambda row: row["net_system_utility_provisional"], reverse=True)[:3]
    top_delivered = sorted(delivered_rows, key=lambda row: row["delivered_only_gross_utility"], reverse=True)[:3]
    proposed = [
        row for row in delivered_rows
        if row["variant_full"] in {
            "proposed_calibrated_payload_plus_z_guided",
            "proposed_calibrated_payload_profile",
            "proposed_payload_repaired_combined",
            "proposed_calibrated_payload_plus_cost_aware_ucb",
        }
    ]
    best_proposed = max(proposed, key=lambda row: row["delivered_only_gross_utility"]) if proposed else None
    return {
        "variants": [item["short_name"] for item in variants],
        "generated_tasks": task_count,
        "lifecycle_counts": {
            "delivered": max(row["lifetime_delivered_task_level"] for row in delivered_rows),
            "expired": max(row["lifetime_expired_task_level"] for row in delivered_rows),
            "deferred": max(row["lifetime_deferred_task_level"] for row in delivered_rows),
            "undelivered_by_end": max(row["lifetime_undelivered_by_end_task_level"] for row in delivered_rows),
        },
        "top_net": [(row["variant_short"], row["net_system_utility_provisional"]) for row in top_net],
        "top_delivered": [(row["variant_short"], row["delivered_only_gross_utility"]) for row in top_delivered],
        "best_proposed": None if best_proposed is None else best_proposed["variant_short"],
        "all_exact": all(row["metric_quality"] == "exact" for row in delivered_rows),
        "resource_infeasible": any(
            _to_float(row.get("min_battery_soc"), 1.0) <= 0.18
            or _to_float(row.get("max_thermal"), 0.0) > 0.92
            for row in delivered_rows
        ),
        "files": [
            output_dir / "summary.csv",
            output_dir / "summary.md",
            output_dir / "delivered_only_metrics.csv",
            output_dir / "delivered_only_metrics.md",
            output_dir / "task_lifetime_outcomes.csv",
            output_dir / "formal_smoke_test_report.md",
            index_csv,
            index_md,
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SARFish 1-day smoke test.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    configure_logging()
    result = run_smoke(args.output_dir)
    print("files_created=" + ",".join(str(path.relative_to(PROJECT_ROOT)) for path in result["files"]))
    print("variants_run=" + ",".join(result["variants"]))
    print(f"generated_task_count={result['generated_tasks']}")
    counts = result["lifecycle_counts"]
    print(
        "lifecycle_counts="
        f"delivered:{counts['delivered']},expired:{counts['expired']},"
        f"deferred:{counts['deferred']},undelivered_by_end:{counts['undelivered_by_end']}"
    )
    print("top3_provisional_net=" + ",".join(f"{name}:{value:.6g}" for name, value in result["top_net"]))
    print("top3_delivered_only=" + ",".join(f"{name}:{value:.6g}" for name, value in result["top_delivered"]))
    print(f"best_proposed_candidate={result['best_proposed']}")
    print(
        "formal_multiseed_recommendation="
        + (
            "wait; resource/contact/deadline stress recalibration needed"
            if result["resource_infeasible"]
            else (
                "can proceed after minor documentation updates"
                if result["all_exact"]
                else "wait for one more metric/logging fix"
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
