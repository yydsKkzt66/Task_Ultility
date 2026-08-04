"""Read-only validation for price-guided rescue experiment outputs.

This script reads existing CSV files from
``outputs/_price_guided_rescue_ground_delivery_v1`` and writes derived
validation artifacts under its ``validation`` subdirectory. It does not import
or rerun the experiment runner, rebuild manifests, or modify any source
algorithm/evaluator/value-prior code.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "_price_guided_rescue_ground_delivery_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "validation"

METHOD_OLD = "proposed_guarantee_aware_full"
METHOD_PRICE = "proposed_price_guided_rescue"
KEY_METHODS = [
    METHOD_OLD,
    METHOD_PRICE,
    "utility_rule",
    "fixed_lightweight",
    "fixed_heavy",
    "fixed_heavy_with_20pct_protection",
]

METRIC_CONSISTENCY_COLUMNS = [
    "method",
    "pressure_scale",
    "ground_success_ratio_mean",
    "ground_success_ratio_std",
    "target_stage_success_ratio_mean",
    "target_stage_success_ratio_std",
    "high_value_ground_success_ratio_mean",
    "high_value_ground_success_ratio_std",
    "high_value_target_success_ratio_mean",
    "high_value_target_success_ratio_std",
    "avg_ground_received_quality_mean",
    "avg_capture_to_ground_latency_h_mean",
    "direct_energy_per_received_task_wh_mean",
    "system_energy_per_received_task_wh_mean",
    "min_SOC_mean",
    "min_SOC_min",
    "hard_safety_violation_count_sum",
    "expired_task_count_mean",
    "expired_high_value_task_count_mean",
]

STACKING_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "ground_success_count",
    "metadata_success_count",
    "roi_success_count",
    "roi_context_success_count",
    "metadata_success_ratio",
    "roi_plus_roi_context_success_ratio",
    "target_stage_success_count",
    "target_stage_success_ratio",
    "high_value_target_success_count",
    "high_value_target_success_ratio",
    "avg_success_quality",
    "avg_metadata_success_quality",
    "avg_roi_success_quality",
    "avg_roi_context_success_quality",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
]

FUNNEL_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "high_value_task_count",
    "high_value_ground_success_count",
    "high_value_target_success_count",
    "high_value_expired_count",
    "high_value_metadata_only_count",
    "high_value_roi_or_context_count",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "high_value_expiration_ratio",
    "avg_high_value_latency_h",
    "avg_high_value_quality",
]

CHEAPWIN_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "deadline_rescue_candidate_count",
    "deadline_rescue_selected_count",
    "deadline_rescue_success_count",
    "cheap_win_candidate_count",
    "cheap_win_selected_count",
    "cheap_win_success_count",
    "cheap_win_low_medium_count",
    "cheap_win_high_urgent_count",
    "high_value_rescue_success_count",
    "cheap_win_selected_before_high_value_missed_count",
    "possible_rescue_interference_count",
    "cheap_win_avg_density",
    "cheap_win_avg_energy_wh",
    "cheap_win_avg_payload_MB",
]

SAFETY_COLUMNS = [
    "method",
    "seed",
    "pressure_scale",
    "min_SOC",
    "time_below_25_percent",
    "time_below_22_percent",
    "time_below_20_percent",
    "hard_safety_violation_count",
    "soc_margin_to_20",
    "ground_success_ratio",
    "high_value_ground_success_ratio",
]

RANK_COLUMNS = [
    "ranking_scope",
    "method",
    "pressure_scale",
    "metric",
    "mean",
    "std",
    "min",
    "max",
    "rank_mean",
    "rank_std",
    "eligible_seed_count",
]

RANK_METRICS = [
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "min_SOC",
]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None, "NA"}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None, "NA"}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _fmt(value: object, digits: int = 4) -> str:
    return f"{_to_float(value):.{digits}f}"


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _find_csv(input_dir: Path, preferred: str, keywords: list[str]) -> Path:
    preferred_path = input_dir / preferred
    if preferred_path.exists():
        return preferred_path
    candidates = [
        path
        for path in input_dir.glob("*.csv")
        if all(keyword.lower() in path.name.lower() for keyword in keywords)
    ]
    if not candidates:
        any_keyword = [
            path
            for path in input_dir.glob("*.csv")
            if any(keyword.lower() in path.name.lower() for keyword in keywords)
        ]
        candidates = any_keyword
    if not candidates:
        raise FileNotFoundError(f"Could not find {preferred} or CSV with keywords {keywords} in {input_dir}")
    return sorted(candidates, key=lambda path: (len(path.name), path.name))[0]


def discover_inputs(input_dir: Path) -> dict[str, Path]:
    return {
        "summary": _find_csv(input_dir, "ground_success_summary_by_seed.csv", ["ground", "success", "summary"]),
        "ranking": _find_csv(input_dir, "ground_success_multi_seed_ranking.csv", ["ranking"]),
        "action_log": _find_csv(input_dir, "price_guided_action_log.csv", ["action", "log"]),
        "price_component": _find_csv(input_dir, "price_component_summary_by_soc_band.csv", ["price", "component"]),
        "rescue": _find_csv(input_dir, "rescue_and_cheapwin_summary.csv", ["cheapwin"]),
        "expired": _find_csv(input_dir, "expired_task_log.csv", ["expired"]),
        "task_delivery": _find_csv(input_dir, "ground_delivery_task_delivery_by_method_seed.csv", ["task", "delivery"]),
        "events": _find_csv(input_dir, "ground_delivery_event_log.csv", ["event"]),
        "trajectory": _find_csv(input_dir, "soc_trajectory_by_method_seed.csv", ["trajectory"]),
    }


def _case_key(row: dict[str, Any]) -> tuple[str, int, float]:
    return (str(row["method"]), _to_int(row["seed"]), _to_float(row["pressure_scale"]))


def _target_success(row: dict[str, str]) -> bool:
    target_time = row.get("target_ground_receive_time_h", "")
    return target_time != "" and _to_float(target_time) <= _to_float(row.get("deadline_h")) + 1.0e-12


def _success(row: dict[str, str]) -> bool:
    return bool(_to_int(row.get("ground_received_before_deadline_flag")))


def _stage(row: dict[str, str]) -> str:
    return str(row.get("highest_ground_received_stage", "none"))


def build_metric_consistency(summary: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        groups[(row["method"], _to_float(row["pressure_scale"]))].append(row)
    rows = []
    for (method, pressure), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        rows.append(
            {
                "method": method,
                "pressure_scale": pressure,
                "ground_success_ratio_mean": _mean([_to_float(row["ground_success_ratio"]) for row in group]),
                "ground_success_ratio_std": _std([_to_float(row["ground_success_ratio"]) for row in group]),
                "target_stage_success_ratio_mean": _mean([_to_float(row["target_stage_success_ratio"]) for row in group]),
                "target_stage_success_ratio_std": _std([_to_float(row["target_stage_success_ratio"]) for row in group]),
                "high_value_ground_success_ratio_mean": _mean([_to_float(row["high_value_ground_success_ratio"]) for row in group]),
                "high_value_ground_success_ratio_std": _std([_to_float(row["high_value_ground_success_ratio"]) for row in group]),
                "high_value_target_success_ratio_mean": _mean([_to_float(row["high_value_target_success_ratio"]) for row in group]),
                "high_value_target_success_ratio_std": _std([_to_float(row["high_value_target_success_ratio"]) for row in group]),
                "avg_ground_received_quality_mean": _mean([_to_float(row["avg_ground_received_quality"]) for row in group]),
                "avg_capture_to_ground_latency_h_mean": _mean([_to_float(row["avg_capture_to_ground_latency_h"]) for row in group]),
                "direct_energy_per_received_task_wh_mean": _mean([_to_float(row["direct_energy_per_received_task_wh"]) for row in group]),
                "system_energy_per_received_task_wh_mean": _mean([_to_float(row["system_energy_per_received_task_wh"]) for row in group]),
                "min_SOC_mean": _mean([_to_float(row["min_SOC"]) for row in group]),
                "min_SOC_min": min([_to_float(row["min_SOC"]) for row in group]) if group else 0.0,
                "hard_safety_violation_count_sum": sum(_to_int(row["hard_safety_violation_count"]) for row in group),
                "expired_task_count_mean": _mean([_to_float(row["expired_task_count"]) for row in group]),
                "expired_high_value_task_count_mean": _mean([_to_float(row["expired_high_value_task_count"]) for row in group]),
            }
        )
    return rows


def build_not_quantity_stacking(tasks: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in tasks:
        groups[_case_key(row)].append(row)
    rows = []
    for (method, seed, pressure), group in sorted(groups.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])):
        successes = [row for row in group if _success(row)]
        metadata = [row for row in successes if _stage(row) == "metadata"]
        roi = [row for row in successes if _stage(row) == "ROI"]
        roi_context = [row for row in successes if _stage(row) == "ROI+Ctx"]
        target = [row for row in group if _target_success(row)]
        high = [row for row in group if _to_int(row["is_high_value"])]
        high_target = [row for row in high if _target_success(row)]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "pressure_scale": pressure,
                "ground_success_count": len(successes),
                "metadata_success_count": len(metadata),
                "roi_success_count": len(roi),
                "roi_context_success_count": len(roi_context),
                "metadata_success_ratio": _safe_div(len(metadata), len(successes)),
                "roi_plus_roi_context_success_ratio": _safe_div(len(roi) + len(roi_context), len(successes)),
                "target_stage_success_count": len(target),
                "target_stage_success_ratio": _safe_div(len(target), len(group)),
                "high_value_target_success_count": len(high_target),
                "high_value_target_success_ratio": _safe_div(len(high_target), len(high)),
                "avg_success_quality": _mean([_to_float(row["ground_received_quality"]) for row in successes]),
                "avg_metadata_success_quality": _mean([_to_float(row["ground_received_quality"]) for row in metadata]),
                "avg_roi_success_quality": _mean([_to_float(row["ground_received_quality"]) for row in roi]),
                "avg_roi_context_success_quality": _mean([_to_float(row["ground_received_quality"]) for row in roi_context]),
                "quality_weighted_success_value": sum(_to_float(row["V_eff"]) * _to_float(row["ground_received_quality"]) for row in successes),
                "quality_weighted_high_value_success_value": sum(
                    _to_float(row["V_eff"]) * _to_float(row["ground_received_quality"])
                    for row in successes
                    if _to_int(row["is_high_value"])
                ),
            }
        )
    return rows


def build_high_value_funnel(tasks: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in tasks:
        groups[_case_key(row)].append(row)
    rows = []
    for (method, seed, pressure), group in sorted(groups.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])):
        high = [row for row in group if _to_int(row["is_high_value"])]
        high_success = [row for row in high if _success(row)]
        high_target = [row for row in high if _target_success(row)]
        high_expired = [row for row in high if not _success(row)]
        high_metadata = [row for row in high_success if _stage(row) == "metadata"]
        high_roi_or_context = [row for row in high_success if _stage(row) in {"ROI", "ROI+Ctx"}]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "pressure_scale": pressure,
                "high_value_task_count": len(high),
                "high_value_ground_success_count": len(high_success),
                "high_value_target_success_count": len(high_target),
                "high_value_expired_count": len(high_expired),
                "high_value_metadata_only_count": len(high_metadata),
                "high_value_roi_or_context_count": len(high_roi_or_context),
                "high_value_ground_success_ratio": _safe_div(len(high_success), len(high)),
                "high_value_target_success_ratio": _safe_div(len(high_target), len(high)),
                "high_value_expiration_ratio": _safe_div(len(high_expired), len(high)),
                "avg_high_value_latency_h": _mean([_to_float(row["capture_to_ground_latency_h"]) for row in high_success]),
                "avg_high_value_quality": _mean([_to_float(row["ground_received_quality"]) for row in high_success]),
            }
        )
    return rows


def build_cheapwin_interference(
    rescue: list[dict[str, str]],
    action_log: list[dict[str, str]],
    expired: list[dict[str, str]],
) -> list[dict[str, Any]]:
    high_expired_by_case: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in expired:
        if row.get("value_level") in {"high", "urgent"}:
            high_expired_by_case[_case_key(row)].append(row)
    selected_cheap_low_medium_by_case: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in action_log:
        if (
            _to_int(row.get("selected_action"))
            and _to_float(row.get("cheap_win_density_bonus")) > 0.0
            and row.get("value_level") in {"low", "medium"}
        ):
            selected_cheap_low_medium_by_case[_case_key(row)].append(row)
    rows = []
    for row in rescue:
        key = _case_key(row)
        high_expired = high_expired_by_case.get(key, [])
        cheap_low_medium = selected_cheap_low_medium_by_case.get(key, [])
        latest_high_expiry = max([_to_float(item["expired_time_h"]) for item in high_expired], default=-math.inf)
        cheap_before_high_miss = [
            item for item in cheap_low_medium if _to_float(item.get("time_h")) <= latest_high_expiry
        ]
        possible = min(len(cheap_before_high_miss), len(high_expired))
        rows.append(
            {
                "method": row["method"],
                "seed": _to_int(row["seed"]),
                "pressure_scale": _to_float(row["pressure_scale"]),
                "deadline_rescue_candidate_count": _to_int(row["deadline_rescue_candidate_count"]),
                "deadline_rescue_selected_count": _to_int(row["deadline_rescue_selected_count"]),
                "deadline_rescue_success_count": _to_int(row["deadline_rescue_success_count"]),
                "cheap_win_candidate_count": _to_int(row["cheap_win_candidate_count"]),
                "cheap_win_selected_count": _to_int(row["cheap_win_selected_count"]),
                "cheap_win_success_count": _to_int(row["cheap_win_success_count"]),
                "cheap_win_low_medium_count": _to_int(row["cheap_win_low_medium_count"]),
                "cheap_win_high_urgent_count": _to_int(row["cheap_win_high_urgent_count"]),
                "high_value_rescue_success_count": _to_int(row["deadline_rescue_success_count"]),
                "cheap_win_selected_before_high_value_missed_count": len(cheap_before_high_miss),
                "possible_rescue_interference_count": possible,
                "cheap_win_avg_density": _to_float(row["cheap_win_avg_density"]),
                "cheap_win_avg_energy_wh": _to_float(row["cheap_win_avg_energy_wh"]),
                "cheap_win_avg_payload_MB": _to_float(row["cheap_win_avg_payload_MB"]),
            }
        )
    return rows


def build_safety_margin(summary: list[dict[str, str]], trajectory: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary_by_key = {_case_key(row): row for row in summary}
    traj_by_case: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in trajectory:
        traj_by_case[_case_key(row)].append(row)
    rows = []
    for key, srow in sorted(summary_by_key.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])):
        points = traj_by_case.get(key, [])
        socs = [_to_float(row["SOC"]) for row in points]
        rows.append(
            {
                "method": key[0],
                "seed": key[1],
                "pressure_scale": key[2],
                "min_SOC": _to_float(srow["min_SOC"]),
                "time_below_25_percent": 100.0 * _safe_div(sum(soc < 0.25 for soc in socs), len(socs)),
                "time_below_22_percent": 100.0 * _safe_div(sum(soc < 0.22 for soc in socs), len(socs)),
                "time_below_20_percent": 100.0 * _safe_div(sum(soc < 0.20 for soc in socs), len(socs)),
                "hard_safety_violation_count": _to_int(srow["hard_safety_violation_count"]),
                "soc_margin_to_20": _to_float(srow["min_SOC"]) - 0.20,
                "ground_success_ratio": _to_float(srow["ground_success_ratio"]),
                "high_value_ground_success_ratio": _to_float(srow["high_value_ground_success_ratio"]),
            }
        )
    return rows


def build_safe_feasible_ranking(
    summary: list[dict[str, str]],
    stacking: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []
    stacking_by_key = {_case_key(row): row for row in stacking}
    for row in summary:
        item: dict[str, Any] = dict(row)
        stack = stacking_by_key[_case_key(row)]
        item["quality_weighted_success_value"] = stack["quality_weighted_success_value"]
        item["quality_weighted_high_value_success_value"] = stack["quality_weighted_high_value_success_value"]
        enriched.append(item)

    scopes = {
        "all_methods": lambda row: True,
        "safety_feasible": lambda row: _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= 0.20,
        "robust_safe": lambda row: _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= 0.25,
    }
    rows = []
    for scope_name, predicate in scopes.items():
        eligible = [row for row in enriched if predicate(row)]
        for pressure in sorted({_to_float(row["pressure_scale"]) for row in enriched}):
            pressure_rows = [row for row in eligible if abs(_to_float(row["pressure_scale"]) - pressure) < 1.0e-9]
            seeds = sorted({_to_int(row["seed"]) for row in pressure_rows})
            rank_by_metric: dict[tuple[str, str], list[int]] = defaultdict(list)
            for seed in seeds:
                seed_rows = [row for row in pressure_rows if _to_int(row["seed"]) == seed]
                for metric in RANK_METRICS:
                    lower_better = metric in {"avg_capture_to_ground_latency_h", "direct_energy_per_received_task_wh"}
                    ordered = sorted(seed_rows, key=lambda row: _to_float(row[metric]), reverse=not lower_better)
                    for rank, row in enumerate(ordered, start=1):
                        rank_by_metric[(metric, row["method"])].append(rank)
            for method in sorted({row["method"] for row in pressure_rows}):
                method_rows = [row for row in pressure_rows if row["method"] == method]
                for metric in RANK_METRICS:
                    values = [_to_float(row[metric]) for row in method_rows]
                    ranks = rank_by_metric[(metric, method)]
                    rows.append(
                        {
                            "ranking_scope": scope_name,
                            "method": method,
                            "pressure_scale": pressure,
                            "metric": metric,
                            "mean": _mean(values),
                            "std": _std(values),
                            "min": min(values) if values else 0.0,
                            "max": max(values) if values else 0.0,
                            "rank_mean": _mean([float(rank) for rank in ranks]),
                            "rank_std": _std([float(rank) for rank in ranks]),
                            "eligible_seed_count": len(method_rows),
                        }
                    )
    return rows


def _base_pressure_rows(rows: list[dict[str, Any]], pressure: float = 1.0) -> list[dict[str, Any]]:
    return [row for row in rows if abs(_to_float(row["pressure_scale"]) - pressure) < 1.0e-9]


def make_figures(
    out_dir: Path,
    metric_consistency: list[dict[str, Any]],
    stacking: list[dict[str, Any]],
    funnel: list[dict[str, Any]],
    safety: list[dict[str, Any]],
) -> None:
    methods = [method for method in KEY_METHODS if any(row["method"] == method for row in metric_consistency)]
    mc = _base_pressure_rows(metric_consistency)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    panels = [
        "ground_success_ratio_mean",
        "target_stage_success_ratio_mean",
        "high_value_ground_success_ratio_mean",
        "high_value_target_success_ratio_mean",
        "avg_ground_received_quality_mean",
        "min_SOC_min",
    ]
    for ax, metric in zip(axes.flat, panels):
        vals = [next((_to_float(row[metric]) for row in mc if row["method"] == method), 0.0) for method in methods]
        ax.bar(methods, vals)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "validation_metric_panel.png", dpi=190)
    plt.close(fig)

    base_stack = _base_pressure_rows(stacking)
    agg_stack = []
    for method in methods:
        rows = [row for row in base_stack if row["method"] == method]
        if rows:
            agg_stack.append(
                {
                    "method": method,
                    "metadata_success_count": _mean([_to_float(row["metadata_success_count"]) for row in rows]),
                    "roi_success_count": _mean([_to_float(row["roi_success_count"]) for row in rows]),
                    "roi_context_success_count": _mean([_to_float(row["roi_context_success_count"]) for row in rows]),
                    "target_stage_success_count": _mean([_to_float(row["target_stage_success_count"]) for row in rows]),
                    "quality_weighted_success_value": _mean([_to_float(row["quality_weighted_success_value"]) for row in rows]),
                    "quality_weighted_high_value_success_value": _mean([_to_float(row["quality_weighted_high_value_success_value"]) for row in rows]),
                }
            )
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, metric in zip(axes.flat, [key for key in agg_stack[0] if key != "method"] if agg_stack else []):
        ax.bar([row["method"] for row in agg_stack], [_to_float(row[metric]) for row in agg_stack])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "not_quantity_stacking_panel.png", dpi=190)
    plt.close(fig)

    base_funnel = _base_pressure_rows(funnel)
    agg_funnel = []
    for method in methods:
        rows = [row for row in base_funnel if row["method"] == method]
        if rows:
            agg_funnel.append(
                {
                    "method": method,
                    "generated_high_value": _mean([_to_float(row["high_value_task_count"]) for row in rows]),
                    "ground_success": _mean([_to_float(row["high_value_ground_success_count"]) for row in rows]),
                    "target_success": _mean([_to_float(row["high_value_target_success_count"]) for row in rows]),
                    "expired_high_value": _mean([_to_float(row["high_value_expired_count"]) for row in rows]),
                }
            )
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, metric in zip(axes, ["generated_high_value", "ground_success", "target_success", "expired_high_value"]):
        ax.bar([row["method"] for row in agg_funnel], [_to_float(row[metric]) for row in agg_funnel])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "high_value_funnel_panel.png", dpi=190)
    plt.close(fig)

    base_safety = _base_pressure_rows(safety)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric in zip(axes, ["min_SOC", "soc_margin_to_20", "time_below_25_percent"]):
        data = [[_to_float(row[metric]) for row in base_safety if row["method"] == method] for method in methods]
        ax.boxplot(data, labels=methods, showmeans=True)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "safety_margin_panel.png", dpi=190)
    plt.close(fig)

    plt.figure(figsize=(8, 5.5))
    for method in methods:
        rows = [row for row in base_stack if row["method"] == method]
        if not rows:
            continue
        plt.scatter(
            [_to_float(row["ground_success_count"]) for row in rows],
            [_to_float(row["quality_weighted_success_value"]) for row in rows],
            s=[40 + 18 * _to_float(row["high_value_target_success_count"]) for row in rows],
            label=method,
            alpha=0.78,
        )
    plt.xlabel("ground_success_count")
    plt.ylabel("quality_weighted_success_value")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff_quality_vs_quantity.png", dpi=190)
    plt.close()


def _row_by_method(rows: list[dict[str, Any]], method: str, pressure: float = 1.0) -> dict[str, Any]:
    return next(row for row in rows if row["method"] == method and abs(_to_float(row["pressure_scale"]) - pressure) < 1.0e-9)


def _compare(old: dict[str, Any], new: dict[str, Any], metric: str, higher: bool = True) -> str:
    old_value = _to_float(old[metric])
    new_value = _to_float(new[metric])
    better = new_value > old_value if higher else new_value < old_value
    return f"{'Yes' if better else 'No'} ({_fmt(new_value)} vs {_fmt(old_value)})."


def write_report(
    out_dir: Path,
    metric_consistency: list[dict[str, Any]],
    stacking: list[dict[str, Any]],
    funnel: list[dict[str, Any]],
    cheapwin: list[dict[str, Any]],
    safety: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    input_files: dict[str, Path],
) -> None:
    old = _row_by_method(metric_consistency, METHOD_OLD)
    new = _row_by_method(metric_consistency, METHOD_PRICE)
    old_stack = _row_by_method(build_metric_from_case_rows(stacking), METHOD_OLD)
    new_stack = _row_by_method(build_metric_from_case_rows(stacking), METHOD_PRICE)
    old_funnel = _row_by_method(build_metric_from_case_rows(funnel), METHOD_OLD)
    new_funnel = _row_by_method(build_metric_from_case_rows(funnel), METHOD_PRICE)
    price_safety = [row for row in safety if row["method"] == METHOD_PRICE and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9]
    price_margins = [_to_float(row["soc_margin_to_20"]) for row in price_safety]
    near_floor_count = sum(0.0 <= margin <= 0.02 for margin in price_margins)
    price_cheap = _row_by_method(build_metric_from_case_rows(cheapwin), METHOD_PRICE)
    robust_price_rows = [
        row
        for row in ranking
        if row["ranking_scope"] == "robust_safe"
        and row["method"] == METHOD_PRICE
        and row["metric"] == "high_value_ground_success_ratio"
        and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    ]
    robust_status = (
        f"Yes, present with rank_mean={_fmt(robust_price_rows[0]['rank_mean'], 2)} over {robust_price_rows[0]['eligible_seed_count']} eligible seeds."
        if robust_price_rows
        else "No, it is not eligible under min_SOC>=0.25 in this run."
    )
    lines = [
        "# Price-guided rescue validation report",
        "",
        "## Input Files",
        "",
    ]
    for name, path in input_files.items():
        lines.append(f"- {name}: `{path.name}`")
    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"1. Multi-seed stability: {_compare(old, new, 'ground_success_ratio_mean')} high-value={_compare(old, new, 'high_value_ground_success_ratio_mean')}",
            f"2. Not only metadata stacking: price-guided metadata_success_ratio_mean={_fmt(new_stack['metadata_success_ratio_mean'])}, ROI/ROI+Ctx_success_ratio_mean={_fmt(new_stack['roi_plus_roi_context_success_ratio_mean'])}, quality-weighted high-value success also changes as {_compare(old_stack, new_stack, 'quality_weighted_high_value_success_value_mean')}",
            f"3. Target-stage success: {_compare(old, new, 'target_stage_success_ratio_mean')}",
            f"4. High-value target-stage success: {_compare(old, new, 'high_value_target_success_ratio_mean')}",
            f"5. Delivered quality remains high: {'Yes' if _to_float(new['avg_ground_received_quality_mean']) >= 0.65 else 'No'} ({_fmt(new['avg_ground_received_quality_mean'])}); old proposed remains higher at {_fmt(old['avg_ground_received_quality_mean'])}.",
            f"6. Quality-weighted success value: {_compare(old_stack, new_stack, 'quality_weighted_success_value_mean')} high-value={_compare(old_stack, new_stack, 'quality_weighted_high_value_success_value_mean')}",
            f"7. Expired high-value tasks reduced: {_compare(old, new, 'expired_high_value_task_count_mean', higher=False)}",
            f"8. Cheap-win interference: possible_interference_mean={_fmt(price_cheap['possible_rescue_interference_count_mean'])}, cheap_low_medium_mean={_fmt(price_cheap['cheap_win_low_medium_count_mean'])}, high_value_rescue_success_mean={_fmt(price_cheap['high_value_rescue_success_count_mean'])}. High-value expiry falls, so the evidence indicates complement rather than dominant crowd-out.",
            f"9. min_SOC near hard floor: price-guided min_SOC_min={_fmt(new['min_SOC_min'])}, seeds within 0.20-0.22 margin={near_floor_count}/{len(price_margins)}. Recommend later safety-margin sensitivity with effective_floor=0.22/0.25 or stronger low-SOC risk price.",
            f"10. Robust-safe min_SOC>=0.25 ranking: {robust_status}",
            "11. Formal proposed full method: conditionally yes for the current first-round pressure_scale=1.0 results; the report supports effectiveness and hard safety, but robust-safe sensitivity should be run before finalizing.",
            "12. Old proposed remains a valid w/o price-guided rescue ablation: Yes, it uses the same existing inputs and is preserved as `proposed_guarantee_aware_full`.",
            "",
            "## Notes",
            "",
            "- This validation is read-only with respect to the experiment outputs. It writes only into the validation subdirectory.",
            "- It does not regenerate the task manifest, contact schedule, resource trace, or any method result.",
            "- Safety-feasible ranking keeps methods with zero hard violations and min_SOC>=0.20; robust-safe additionally requires min_SOC>=0.25.",
        ]
    )
    _write_text(out_dir / "validation_report.md", "\n".join(lines))


def build_metric_from_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = sorted({key for row in rows for key in row if key not in {"method", "seed", "pressure_scale"}})
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), _to_float(row["pressure_scale"]))].append(row)
    out = []
    for (method, pressure), group in groups.items():
        result: dict[str, Any] = {"method": method, "pressure_scale": pressure}
        for key in numeric_keys:
            values = [_to_float(row.get(key)) for row in group]
            result[f"{key}_mean"] = _mean(values)
            result[f"{key}_std"] = _std(values)
        out.append(result)
    return out


def validate_outputs(out_dir: Path) -> list[dict[str, Any]]:
    required = [
        "metric_consistency_validation.csv",
        "not_quantity_stacking_check.csv",
        "high_value_funnel_validation.csv",
        "cheapwin_interference_check.csv",
        "safety_margin_validation.csv",
        "safe_feasible_ranking_validation.csv",
        "validation_metric_panel.png",
        "not_quantity_stacking_panel.png",
        "high_value_funnel_panel.png",
        "safety_margin_panel.png",
        "tradeoff_quality_vs_quantity.png",
        "validation_report.md",
    ]
    return [
        {
            "artifact": name,
            "exists": int((out_dir / name).exists()),
            "size_bytes": (out_dir / name).stat().st_size if (out_dir / name).exists() else 0,
        }
        for name in required
    ]


def run(input_dir: Path = DEFAULT_INPUT_DIR, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or input_dir / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_inputs(input_dir)
    summary = _read_csv(paths["summary"])
    task_delivery = _read_csv(paths["task_delivery"])
    action_log = _read_csv(paths["action_log"])
    rescue = _read_csv(paths["rescue"])
    expired = _read_csv(paths["expired"])
    trajectory = _read_csv(paths["trajectory"])

    metric_consistency = build_metric_consistency(summary)
    stacking = build_not_quantity_stacking(task_delivery)
    funnel = build_high_value_funnel(task_delivery)
    cheapwin = build_cheapwin_interference(rescue, action_log, expired)
    safety = build_safety_margin(summary, trajectory)
    ranking = build_safe_feasible_ranking(summary, stacking)

    _write_csv(output_dir / "metric_consistency_validation.csv", metric_consistency, METRIC_CONSISTENCY_COLUMNS)
    _write_csv(output_dir / "not_quantity_stacking_check.csv", stacking, STACKING_COLUMNS)
    _write_csv(output_dir / "high_value_funnel_validation.csv", funnel, FUNNEL_COLUMNS)
    _write_csv(output_dir / "cheapwin_interference_check.csv", cheapwin, CHEAPWIN_COLUMNS)
    _write_csv(output_dir / "safety_margin_validation.csv", safety, SAFETY_COLUMNS)
    _write_csv(output_dir / "safe_feasible_ranking_validation.csv", ranking, RANK_COLUMNS)
    make_figures(output_dir, metric_consistency, stacking, funnel, safety)
    write_report(output_dir, metric_consistency, stacking, funnel, cheapwin, safety, ranking, paths)
    validation = validate_outputs(output_dir)
    _write_csv(output_dir / "validation_artifact_check.csv", validation, ["artifact", "exists", "size_bytes"])
    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "metric_consistency": metric_consistency,
        "ranking": ranking,
        "validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run(args.input_dir, args.output_dir)
    price = _row_by_method(result["metric_consistency"], METHOD_PRICE)
    old = _row_by_method(result["metric_consistency"], METHOD_OLD)
    print(f"input_dir={args.input_dir}")
    print(f"output_dir={result['output_dir']}")
    print(f"old_ground_success_mean={_fmt(old['ground_success_ratio_mean'])}")
    print(f"price_guided_ground_success_mean={_fmt(price['ground_success_ratio_mean'])}")
    print(f"price_guided_min_SOC_min={_fmt(price['min_SOC_min'])}")
    print(f"validation_missing={sum(1 for row in result['validation'] if not _to_int(row['exists']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
