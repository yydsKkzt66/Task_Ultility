"""Standalone 5-seed realistic workload/contact benchmark.

This script extends the existing realistic workload/contact sensitivity
experiment to seeds 0..4 and writes a separate output package. It reuses the
existing task/contact/resource builders and method simulators without changing
src algorithms, value-prior logic, high-value labels, reward/evaluator code, or
older runners.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_price_guided_rescue_ground_delivery_experiment as price  # noqa: E402
import scripts.run_realistic_workload_contact_sensitivity as realistic  # noqa: E402
import scripts.run_uneven_pressure_ground_delivery_experiment as base  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_realistic_workload_contact_sensitivity_5seeds_v1"
SEEDS = [0, 1, 2, 3, 4]
METHODS = realistic.METHODS
METHOD_LABELS = {
    "proposed_price_guided_rescue": "proposed_rank",
    "proposed_guarantee_aware_full": "old_proposed_rank",
    "utility_rule": "utility_rule_rank",
    "fixed_lightweight": "fixed_lightweight_rank",
    "fixed_heavy": "fixed_heavy_rank",
    "fixed_heavy_with_20pct_protection": "protected_heavy_rank",
    "ground_approved_followup": "ground_followup_rank",
    "persistent_queue_without_guarantee": "persistent_queue_rank",
}

SUMMARY_COLUMNS = [
    "method",
    "seed",
    "workload_scenario",
    "contact_scenario",
    "task_manifest_hash",
    "contact_schedule_hash",
    "resource_trace_hash",
    "value_prior_hash",
    "scene_count",
    "subtask_count",
    "avg_subtasks_per_scene",
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "ground_received_task_count",
    "high_value_ground_received_count",
    "avg_ground_received_quality",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "net_system_utility",
    "H_net_utility",
    "avg_capture_to_ground_latency_h",
    "deadline_satisfaction_ratio",
    "expired_task_count",
    "expired_high_value_task_count",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "min_SOC",
    "final_SOC",
    "SOC_margin_to_20",
    "time_below_25_percent",
    "time_below_22_percent",
    "fraction_time_SOC_below_20",
    "hard_safety_violation_count",
    "final_backlog_count",
    "max_backlog_count",
    "queue_MB_peak",
    "downlinked_MB",
    "metadata_downlinked_count",
    "roi_downlinked_count",
    "roi_context_downlinked_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
]
METRICS = [
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "ground_received_task_count",
    "high_value_ground_received_count",
    "avg_ground_received_quality",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "net_system_utility",
    "H_net_utility",
    "avg_capture_to_ground_latency_h",
    "deadline_satisfaction_ratio",
    "expired_task_count",
    "expired_high_value_task_count",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "min_SOC",
    "final_SOC",
    "SOC_margin_to_20",
    "time_below_25_percent",
    "time_below_22_percent",
    "fraction_time_SOC_below_20",
    "hard_safety_violation_count",
    "final_backlog_count",
    "max_backlog_count",
    "queue_MB_peak",
    "downlinked_MB",
    "metadata_downlinked_count",
    "roi_downlinked_count",
    "roi_context_downlinked_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
]
LOWER_IS_BETTER = {
    "avg_capture_to_ground_latency_h",
    "expired_task_count",
    "expired_high_value_task_count",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "time_below_25_percent",
    "time_below_22_percent",
    "fraction_time_SOC_below_20",
    "hard_safety_violation_count",
    "final_backlog_count",
    "max_backlog_count",
    "queue_MB_peak",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
}
FIRST_PLACE_METRICS = [
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_high_value_success_value",
    "H_net_utility",
    "expired_high_value_task_count",
    "min_SOC",
    "max_thermal_pressure",
    "queue_MB_peak",
]


def _to_float(value: object, default: float = 0.0) -> float:
    return base._to_float(value, default)


def _to_int(value: object, default: int = 0) -> int:
    return base._to_int(value, default)


def _safe_div(num: float, den: float) -> float:
    return base._safe_div(num, den)


def _fmt(value: object, digits: int = 4) -> str:
    return f"{_to_float(value):.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
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


def _group(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def _value_prior_hash(manifest: list[dict[str, Any]]) -> str:
    return realistic._hash(
        manifest,
        ["task_id", "base_prior_score", "value_level", "value_multiplier", "V_eff", "value_assignment_reason"],
    )


def _hashes(manifest: list[dict[str, Any]], contacts: list[dict[str, Any]], resources: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "task_manifest_hash": realistic._hash(manifest, realistic.REALISTIC_MANIFEST_COLUMNS),
        "contact_schedule_hash": realistic._hash(
            contacts, ["seed", "workload_scenario", "contact_scenario", "orbit_index", "capacity_MB", "start_time_h"]
        ),
        "resource_trace_hash": realistic._hash(
            resources, ["seed", "workload_scenario", "contact_scenario", "orbit_index", "solar_energy_wh", "wave_name"]
        ),
        "value_prior_hash": _value_prior_hash(manifest),
    }


def _mean_summary(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    out = []
    for key, group in sorted(_group(rows, keys).items()):
        item = {keys[idx]: key[idx] for idx in range(len(keys))}
        for metric in metrics:
            item[f"{metric}_mean"] = statistics.mean(_to_float(row[metric]) for row in group)
        out.append(item)
    return out


def build_multi_seed_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (method, workload, contact), group in sorted(_group(rows, ["method", "workload_scenario", "contact_scenario"]).items()):
        for metric in METRICS:
            values = [_to_float(row[metric]) for row in group]
            ranks = []
            for seed in sorted({row["seed"] for row in group}):
                seed_rows = [
                    row
                    for row in rows
                    if row["workload_scenario"] == workload and row["contact_scenario"] == contact and _to_int(row["seed"]) == _to_int(seed)
                ]
                ordered = sorted(
                    [(row["method"], _to_float(row[metric])) for row in seed_rows],
                    key=lambda item: item[1],
                    reverse=metric not in LOWER_IS_BETTER,
                )
                ranks.extend(rank for rank, (name, _value) in enumerate(ordered, start=1) if name == method)
            out.append(
                {
                    "method": method,
                    "workload_scenario": workload,
                    "contact_scenario": contact,
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "rank_mean": statistics.mean(ranks) if ranks else 0.0,
                    "rank_std": statistics.stdev(ranks) if len(ranks) > 1 else 0.0,
                }
            )
    return out


def _scenario_method_means(rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, str, str], float]:
    out = {}
    for (method, workload, contact), group in _group(rows, ["method", "workload_scenario", "contact_scenario"]).items():
        out[(method, workload, contact)] = statistics.mean(_to_float(row[metric]) for row in group)
    return out


def build_robustness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for (workload, contact), group in _group(rows, ["workload_scenario", "contact_scenario"]).items():
        method_scores = []
        for (method,), method_rows in _group(group, ["method"]).items():
            method_scores.append((method, statistics.mean(_to_float(row["high_value_ground_success_ratio"]) for row in method_rows)))
        for rank, (method, _score) in enumerate(sorted(method_scores, key=lambda item: item[1], reverse=True), start=1):
            if rank == 1:
                rank_counts[method]["rank1"] += 1
            if rank <= 2:
                rank_counts[method]["top2"] += 1
    out = []
    for (method,), group in sorted(_group(rows, ["method"]).items()):
        safe = [row for row in group if _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= base.HARD_MIN_SOC]
        robust_safe = [row for row in group if _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= 0.25]
        out.append(
            {
                "method": method,
                "overall_ground_success_mean": statistics.mean(_to_float(row["ground_success_ratio"]) for row in group),
                "overall_high_value_success_mean": statistics.mean(_to_float(row["high_value_ground_success_ratio"]) for row in group),
                "overall_high_value_target_success_mean": statistics.mean(_to_float(row["high_value_target_success_ratio"]) for row in group),
                "overall_quality_weighted_high_value_success_mean": statistics.mean(
                    _to_float(row["quality_weighted_high_value_success_value"]) for row in group
                ),
                "overall_expired_high_value_mean": statistics.mean(_to_float(row["expired_high_value_task_count"]) for row in group),
                "overall_min_SOC_mean": statistics.mean(_to_float(row["min_SOC"]) for row in group),
                "overall_SOC_margin_mean": statistics.mean(_to_float(row["SOC_margin_to_20"]) for row in group),
                "overall_max_thermal_pressure_mean": statistics.mean(_to_float(row["max_thermal_pressure"]) for row in group),
                "num_scenarios_rank1": rank_counts[method]["rank1"],
                "num_scenarios_top2": rank_counts[method]["top2"],
                "num_scenarios_safe": len(safe),
                "num_scenarios_robust_safe": len(robust_safe),
                "num_scenarios_hard_violation": sum(_to_int(row["hard_safety_violation_count"]) > 0 for row in group),
                "worst_case_ground_success": min(_to_float(row["ground_success_ratio"]) for row in group),
                "worst_case_high_value_success": min(_to_float(row["high_value_ground_success_ratio"]) for row in group),
                "worst_case_high_value_target_success": min(_to_float(row["high_value_target_success_ratio"]) for row in group),
                "worst_case_min_SOC": min(_to_float(row["min_SOC"]) for row in group),
            }
        )
    return out


def build_first_place(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (workload, contact), group in sorted(_group(rows, ["workload_scenario", "contact_scenario"]).items()):
        scenario = f"{workload}__{contact}"
        for metric in FIRST_PLACE_METRICS:
            means = []
            for (method,), method_rows in _group(group, ["method"]).items():
                means.append((method, statistics.mean(_to_float(row[metric]) for row in method_rows)))
            ordered = sorted(means, key=lambda item: item[1], reverse=metric not in LOWER_IS_BETTER)
            rank = {method: idx for idx, (method, _value) in enumerate(ordered, start=1)}
            best_method, best_value = ordered[0]
            item = {"scenario": scenario, "metric": metric, "best_method": best_method, "best_value": best_value}
            for method, label in METHOD_LABELS.items():
                item[label] = rank.get(method, "")
            out.append(item)
    return out


def build_deadline_overall_summary(deadline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (workload, contact, value_level), group in sorted(
        _group(deadline_rows, ["workload_scenario", "contact_scenario", "value_level"]).items()
    ):
        waits = [_to_float(row["expected_contact_wait_h"]) for row in group if row.get("expected_contact_wait_h") != ""]
        out.append(
            {
                "workload_scenario": workload,
                "contact_scenario": contact,
                "value_level": value_level,
                "task_count": len(group),
                "no_contact_before_deadline_flag_ratio": _safe_div(sum(_to_int(row["no_contact_before_deadline_flag"]) for row in group), len(group)),
                "mean_contact_opportunities_before_deadline": statistics.mean(
                    _to_float(row["num_contact_opportunities_before_deadline"]) for row in group
                ),
                "mean_deadline_slack_h": statistics.mean(_to_float(row["deadline_slack_h"]) for row in group),
                "mean_first_contact_wait_h": statistics.mean(waits) if waits else 0.0,
            }
        )
    return out


def _bar_panel(rows: list[dict[str, Any]], path: Path, group_key: str, metrics: list[str], title: str) -> None:
    means = _mean_summary(rows, ["method", group_key], metrics)
    groups = sorted({row[group_key] for row in means})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.9 * len(metrics), 4.7), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        for idx, group in enumerate(groups):
            values = []
            for method in METHODS:
                match = next((row for row in means if row["method"] == method and row[group_key] == group), None)
                values.append(_to_float(match[f"{metric}_mean"]) if match else 0.0)
            xs = [x + idx * 0.22 for x in range(len(METHODS))]
            axis.bar(xs, values, width=0.22, label=group)
        axis.set_title(metric)
        axis.set_xticks([x + 0.22 for x in range(len(METHODS))])
        axis.set_xticklabels(METHODS, rotation=65, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_figures(rows: list[dict[str, Any]], output_dir: Path) -> None:
    _bar_panel(
        rows,
        output_dir / "realistic_5seed_ground_success_panel.png",
        "workload_scenario",
        ["ground_success_ratio", "high_value_ground_success_ratio", "high_value_target_success_ratio"],
        "5-seed ground success by workload",
    )
    _bar_panel(
        rows,
        output_dir / "realistic_5seed_contact_sensitivity_panel.png",
        "contact_scenario",
        ["ground_success_ratio", "high_value_ground_success_ratio", "expired_high_value_task_count", "queue_MB_peak"],
        "5-seed contact sensitivity",
    )
    _bar_panel(
        rows,
        output_dir / "realistic_5seed_workload_sensitivity_panel.png",
        "workload_scenario",
        ["high_value_ground_success_ratio", "high_value_target_success_ratio", "quality_weighted_high_value_success_value", "min_SOC"],
        "5-seed workload sensitivity",
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for method in METHODS:
        mrows = [row for row in rows if row["method"] == method]
        ax.scatter(
            [_to_float(row["min_SOC"]) for row in mrows],
            [_to_float(row["quality_weighted_high_value_success_value"]) for row in mrows],
            s=[35 + 160 * _to_float(row["ground_success_ratio"]) for row in mrows],
            alpha=0.58,
            label=method,
        )
    ax.set_xlabel("min_SOC")
    ax.set_ylabel("quality_weighted_high_value_success_value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "realistic_5seed_robustness_tradeoff_panel.png", dpi=180)
    plt.close(fig)
    _bar_panel(
        rows,
        output_dir / "realistic_5seed_thermal_panel.png",
        "workload_scenario",
        ["max_thermal_pressure", "avg_thermal_pressure", "thermal_violation_count", "quality_weighted_high_value_success_value"],
        "5-seed thermal proxy panel",
    )


def _best_method(rows: list[dict[str, Any]], *, workload: str | None = None, contact: str | None = None, metric: str) -> tuple[str, float]:
    source = rows
    if workload is not None:
        source = [row for row in source if row["workload_scenario"] == workload]
    if contact is not None:
        source = [row for row in source if row["contact_scenario"] == contact]
    means = []
    for (method,), group in _group(source, ["method"]).items():
        means.append((method, statistics.mean(_to_float(row[metric]) for row in group)))
    reverse = metric not in LOWER_IS_BETTER
    return sorted(means, key=lambda item: item[1], reverse=reverse)[0] if means else ("", 0.0)


def build_quick_report(
    rows: list[dict[str, Any]],
    deadline_rows: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
    first_place: list[dict[str, Any]],
    seeds: list[int],
) -> str:
    proposed = [row for row in rows if row["method"] == "proposed_price_guided_rescue"]
    old = [row for row in rows if row["method"] == "proposed_guarantee_aware_full"]
    utility = [row for row in rows if row["method"] == "utility_rule"]
    heavy = [row for row in rows if row["method"] == "fixed_heavy"]
    light = [row for row in rows if row["method"] == "fixed_lightweight"]
    proposed_rob = next(row for row in robustness if row["method"] == "proposed_price_guided_rescue")
    proposed_firsts = [row["metric"] + "@" + row["scenario"] for row in first_place if row["best_method"] == "proposed_price_guided_rescue"]
    proposed_not_first = [
        row["metric"] + "@" + row["scenario"] + "->" + row["best_method"]
        for row in first_place
        if row["best_method"] != "proposed_price_guided_rescue"
    ]
    no_contact_ratio = _safe_div(sum(_to_int(row["no_contact_before_deadline_flag"]) for row in deadline_rows), len(deadline_rows))
    urgent_high = [row for row in deadline_rows if row["value_level"] in {"urgent", "high"}]
    urgent_contacts = statistics.mean(_to_float(row["num_contact_opportunities_before_deadline"]) for row in urgent_high) if urgent_high else 0.0
    slack_by_contact = {
        contact: statistics.mean(_to_float(row["deadline_slack_h"]) for row in deadline_rows if row["contact_scenario"] == contact)
        for contact in realistic.CONTACT_SCENARIOS
    }
    sparse_prop = [row for row in proposed if row["contact_scenario"] == "sparse_contact"]
    sparse_old = [row for row in old if row["contact_scenario"] == "sparse_contact"]
    sparse_util = [row for row in utility if row["contact_scenario"] == "sparse_contact"]
    dense_prop = [row for row in proposed if row["contact_scenario"] == "dense_contact"]
    dense_best = _best_method(rows, contact="dense_contact", metric="high_value_ground_success_ratio")
    emergency_prop = [row for row in proposed if row["workload_scenario"] == "emergency_stress"]
    emergency_old = [row for row in old if row["workload_scenario"] == "emergency_stress"]
    nominal_prop = [row for row in proposed if row["workload_scenario"] == "nominal_daily"]
    proposed_overall_hv = statistics.mean(_to_float(row["high_value_ground_success_ratio"]) for row in proposed)
    best_overall_hv = _best_method(rows, metric="high_value_ground_success_ratio")
    overall_best = best_overall_hv[0] == "proposed_price_guided_rescue"
    return "\n".join(
        [
            "# Realistic workload/contact 5-seed benchmark quick report",
            "",
            f"- Seeds: {seeds}. Runs: {len(rows)} = {len(seeds)} seeds x 3 workloads x 3 contacts x 8 methods.",
            "- Task generation remains scene -> subtasks and uses mission_prior_value_v2 plus value_skew_calibration. No manual high-value labels are introduced.",
            "- Thermal fields are thermal_pressure_proxy values, not real temperature.",
            "",
            "## Required answers",
            f"1. Overall best check: proposed overall high-value success={_fmt(proposed_overall_hv)}; best method is {best_overall_hv[0]} ({_fmt(best_overall_hv[1])}). Proposed is {'overall best' if overall_best else 'not overall best on this metric'}.",
            f"2. proposed rank-1 scenarios: {proposed_rob['num_scenarios_rank1']}/9.",
            f"3. proposed top-2 scenarios: {proposed_rob['num_scenarios_top2']}/9.",
            f"4. proposed worst-case high-value success={_fmt(proposed_rob['worst_case_high_value_success'])}; worst-case high-value target={_fmt(proposed_rob['worst_case_high_value_target_success'])}.",
            f"5. sparse_contact high-value success proposed/old/utility={_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in sparse_prop))}/{_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in sparse_old))}/{_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in sparse_util))}.",
            f"6. dense_contact best high-value method={dense_best[0]} ({_fmt(dense_best[1])}); proposed dense mean={_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in dense_prop))}.",
            f"7. nominal_daily proposed high-value success={_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in nominal_prop))}, queue peak={_fmt(statistics.mean(_to_float(row['queue_MB_peak']) for row in nominal_prop))}; inspect first-place rows for over-complexity.",
            f"8. emergency_stress expired high-value proposed/old={_fmt(statistics.mean(_to_float(row['expired_high_value_task_count']) for row in emergency_prop))}/{_fmt(statistics.mean(_to_float(row['expired_high_value_task_count']) for row in emergency_old))}.",
            f"9. fixed_heavy worst min_SOC={_fmt(min(_to_float(row['min_SOC']) for row in heavy))}, hard violations={sum(_to_int(row['hard_safety_violation_count']) for row in heavy)}, thermal max mean={_fmt(statistics.mean(_to_float(row['max_thermal_pressure']) for row in heavy))}.",
            f"10. fixed_lightweight high-value success mean={_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in light))}, worst min_SOC={_fmt(min(_to_float(row['min_SOC']) for row in light))}.",
            f"11. utility_rule high-value success mean={_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in utility))}, expired high-value mean={_fmt(statistics.mean(_to_float(row['expired_high_value_task_count']) for row in utility))}, queue peak mean={_fmt(statistics.mean(_to_float(row['queue_MB_peak']) for row in utility))}.",
            f"12. proposed thermal proxy max mean={_fmt(statistics.mean(_to_float(row['max_thermal_pressure']) for row in proposed))}, violations={sum(_to_int(row['thermal_violation_count']) for row in proposed)}.",
            f"13. Deadline/contact audit: no-contact-before-deadline ratio={_fmt(no_contact_ratio)}, urgent/high mean contact opportunities before deadline={_fmt(urgent_contacts)}, sparse/normal/dense slack={_fmt(slack_by_contact['sparse_contact'])}/{_fmt(slack_by_contact['normal_contact'])}/{_fmt(slack_by_contact['dense_contact'])} h.",
            "14. Magazine main benchmark suitability: yes, this package gives 5-seed summaries, deadline/contact audit, robustness ranking, and paper-ready panels under shared traces.",
            "15. Literature-inspired baseline reuse: yes, manifest/contact/resource/value-prior hashes are stable per seed/workload/contact and shared across methods.",
            "",
            f"Proposed first-place metric/scenario cells: {', '.join(proposed_firsts[:30]) if proposed_firsts else 'none'}{' ...' if len(proposed_firsts) > 30 else ''}.",
            f"Cells where proposed is not first: {', '.join(proposed_not_first[:30]) if proposed_not_first else 'none'}{' ...' if len(proposed_not_first) > 30 else ''}.",
        ]
    )


def run_experiment(output_dir: Path, seeds: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_manifest: list[dict[str, Any]] = []
    all_contacts: list[dict[str, Any]] = []
    all_resources: list[dict[str, Any]] = []
    all_deadline_audit: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    all_trajectories: list[dict[str, Any]] = []

    for seed in seeds:
        for workload in realistic.WORKLOAD_SCENARIOS:
            for contact in realistic.CONTACT_SCENARIOS:
                contacts = realistic.build_contact_schedule(seed, workload, contact)
                resources = realistic.build_resource_trace(seed, workload, contact)
                manifest, deadline_audit = realistic.build_task_manifest(seed, workload, contact, contacts)
                hashes = _hashes(manifest, contacts, resources)
                all_manifest.extend(manifest)
                all_contacts.extend(contacts)
                all_resources.extend(resources)
                all_deadline_audit.extend(deadline_audit)
                for method in METHODS:
                    if method == "proposed_price_guided_rescue":
                        result = price.run_price_guided_method(seed, 1.0, manifest, contacts, resources, hashes)
                    else:
                        result = base.run_method(method, seed, 1.0, manifest, contacts, resources, hashes)
                    summary = realistic._augment_result_summary(
                        result,
                        seed=seed,
                        workload=workload,
                        contact=contact,
                        manifest=manifest,
                        hashes=hashes,
                    )
                    all_summary.append(summary)
                    for rowset in (result["events"], result["task_delivery"], result["trajectory"]):
                        for row in rowset:
                            row["workload_scenario"] = workload
                            row["contact_scenario"] = contact
                            row["method_type"] = realistic.METHOD_TYPES.get(str(row.get("method")), "")
                    all_events.extend(result["events"])
                    all_tasks.extend(result["task_delivery"])
                    all_trajectories.extend(result["trajectory"])

    deadline_summary = realistic.build_deadline_summary(all_deadline_audit)
    deadline_overall = build_deadline_overall_summary(all_deadline_audit)
    ranking = build_multi_seed_ranking(all_summary)
    robustness = build_robustness(all_summary)
    first_place = build_first_place(all_summary)
    contact_sensitivity = _mean_summary(
        all_summary,
        ["method", "contact_scenario"],
        [
            "ground_success_ratio",
            "high_value_ground_success_ratio",
            "high_value_target_success_ratio",
            "avg_capture_to_ground_latency_h",
            "expired_high_value_task_count",
            "queue_MB_peak",
            "downlinked_MB",
            "min_SOC",
            "H_net_utility",
        ],
    )
    for row in contact_sensitivity:
        row["avg_latency_h_mean"] = row.pop("avg_capture_to_ground_latency_h_mean")
        row["expired_high_value_mean"] = row.pop("expired_high_value_task_count_mean")
    workload_sensitivity = _mean_summary(
        all_summary,
        ["method", "workload_scenario"],
        [
            "scene_count",
            "subtask_count",
            "ground_success_ratio",
            "high_value_ground_success_ratio",
            "high_value_target_success_ratio",
            "quality_weighted_high_value_success_value",
            "expired_high_value_task_count",
            "min_SOC",
            "max_thermal_pressure",
            "H_net_utility",
        ],
    )
    for row in workload_sensitivity:
        row["quality_weighted_high_value_success_mean"] = row.pop("quality_weighted_high_value_success_value_mean")
        row["expired_high_value_mean"] = row.pop("expired_high_value_task_count_mean")
    build_figures(all_summary, output_dir)

    _write_csv(output_dir / "realistic_5seed_summary_by_seed.csv", all_summary, SUMMARY_COLUMNS)
    _write_csv(output_dir / "realistic_5seed_multi_seed_ranking.csv", ranking)
    _write_csv(output_dir / "realistic_5seed_scenario_robustness_summary.csv", robustness)
    _write_csv(output_dir / "realistic_5seed_contact_sensitivity_summary.csv", contact_sensitivity)
    _write_csv(output_dir / "realistic_5seed_workload_sensitivity_summary.csv", workload_sensitivity)
    _write_csv(output_dir / "realistic_5seed_first_place_check.csv", first_place)
    _write_csv(output_dir / "deadline_contact_audit.csv", all_deadline_audit, realistic.DEADLINE_AUDIT_COLUMNS)
    _write_csv(output_dir / "deadline_contact_summary.csv", deadline_summary)
    _write_csv(output_dir / "deadline_contact_overall_summary.csv", deadline_overall)
    _write_csv(output_dir / "realistic_5seed_task_manifest.csv", all_manifest, realistic.REALISTIC_MANIFEST_COLUMNS)
    _write_csv(output_dir / "shared_contact_schedule.csv", all_contacts)
    _write_csv(output_dir / "shared_resource_trace.csv", all_resources)
    _write_csv(output_dir / "realistic_5seed_ground_events.csv", all_events)
    _write_csv(output_dir / "realistic_5seed_task_delivery.csv", all_tasks)
    _write_csv(output_dir / "realistic_5seed_trajectory.csv", all_trajectories)
    _write_text(output_dir / "quick_report.md", build_quick_report(all_summary, all_deadline_audit, robustness, first_place, seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else SEEDS
    run_experiment(args.output_dir, seeds)
    print(f"Wrote realistic workload/contact 5-seed benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
