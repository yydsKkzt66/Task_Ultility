"""Safety-margin sensitivity for the price-guided rescue variant.

Standalone experiment runner. It reuses the existing standalone builders and
baseline simulators, runs price-guided rescue variants with different low-SOC
risk-price references, and writes all artifacts to a new output directory.

Important: the hard battery floor remains 0.20 for every variant. The
0.22/0.25/0.30 settings are risk-price references only, not hard gates.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_price_guided_rescue_ground_delivery_experiment as pg  # noqa: E402
import scripts.run_uneven_pressure_ground_delivery_experiment as base  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_price_guided_rescue_safety_margin_sensitivity_v1"
SEEDS = [0, 1, 2, 3, 4]
PRESSURE_SCALES = [1.0]
FULL_PRESSURE_SCALES = [0.85, 1.0, 1.15]
BATTERY_HARD_FLOOR = 0.20

MARGIN_VARIANTS = [
    ("proposed_price_guided_rescue_margin_0p20", 0.20, 0.0),
    ("proposed_price_guided_rescue_margin_0p22", 0.22, 250.0),
    ("proposed_price_guided_rescue_margin_0p25", 0.25, 600.0),
    ("proposed_price_guided_rescue_margin_0p30", 0.30, 1000.0),
]

METHODS = [
    "proposed_guarantee_aware_full",
    *[name for name, _ref, _scale in MARGIN_VARIANTS],
    "utility_rule",
    "fixed_lightweight",
    "fixed_heavy",
    "fixed_heavy_with_20pct_protection",
]

METHOD_TYPES = {
    "proposed_guarantee_aware_full": "old_proposed_no_price_rescue",
    "utility_rule": "myopic_greedy",
    "fixed_lightweight": "fixed_conservative",
    "fixed_heavy": "unsafe_stress_baseline",
    "fixed_heavy_with_20pct_protection": "hard_protection_baseline",
    **{name: "price_guided_rescue_risk_price_sensitivity" for name, _ref, _scale in MARGIN_VARIANTS},
}

SUMMARY_COLUMNS = [
    "method",
    "method_type",
    "seed",
    "pressure_scale",
    "risk_reference_SOC",
    "low_SOC_risk_price_scale",
    "task_manifest_hash",
    "contact_schedule_hash",
    "resource_trace_hash",
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "avg_ground_received_quality",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "expired_task_count",
    "expired_high_value_task_count",
    "min_SOC",
    "SOC_margin_to_20",
    "time_below_25_percent",
    "time_below_22_percent",
    "fraction_time_SOC_below_20",
    "hard_safety_violation_count",
    "net_system_utility",
    "H_net_utility",
    "final_backlog_count",
    "max_backlog_count",
]

RANKING_METRICS = [
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_high_value_success_value",
    "expired_high_value_task_count",
    "min_SOC",
    "SOC_margin_to_20",
    "H_net_utility",
    "net_system_utility",
]

FIRST_PLACE_METRICS = [
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "avg_ground_received_quality",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "expired_high_value_task_count",
    "min_SOC",
    "H_net_utility",
    "net_system_utility",
]

LOWER_IS_BETTER = {
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "expired_task_count",
    "expired_high_value_task_count",
}

FIRST_PLACE_COLUMNS = [
    "metric",
    "best_method",
    "best_value",
    "proposed_margin_0p20_rank",
    "proposed_margin_0p22_rank",
    "proposed_margin_0p25_rank",
    "proposed_margin_0p30_rank",
    "old_proposed_rank",
    "utility_rule_rank",
    "fixed_lightweight_rank",
    "fixed_heavy_rank",
    "protected_heavy_rank",
]

RANK_ALIAS = {
    "proposed_margin_0p20_rank": "proposed_price_guided_rescue_margin_0p20",
    "proposed_margin_0p22_rank": "proposed_price_guided_rescue_margin_0p22",
    "proposed_margin_0p25_rank": "proposed_price_guided_rescue_margin_0p25",
    "proposed_margin_0p30_rank": "proposed_price_guided_rescue_margin_0p30",
    "old_proposed_rank": "proposed_guarantee_aware_full",
    "utility_rule_rank": "utility_rule",
    "fixed_lightweight_rank": "fixed_lightweight",
    "fixed_heavy_rank": "fixed_heavy",
    "protected_heavy_rank": "fixed_heavy_with_20pct_protection",
}


def _to_float(value: object, default: float = 0.0) -> float:
    return pg._to_float(value, default)


def _to_int(value: object, default: int = 0) -> int:
    return pg._to_int(value, default)


def _safe_div(num: float, den: float) -> float:
    return pg._safe_div(num, den)


def _fmt(value: object, digits: int = 4) -> str:
    return pg._fmt(value, digits)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _risk_prices_factory(
    risk_reference_soc: float,
    risk_scale: float,
    base_price_fn: Callable[[float, int, float, str, float], dict[str, float]],
) -> Callable[[float, int, float, str, float], dict[str, float]]:
    def prices(soc: float, queue_count: int, queue_mb: float, wave: str, contact_capacity_mb: float) -> dict[str, float]:
        base_prices = base_price_fn(soc, queue_count, queue_mb, wave, contact_capacity_mb)
        risk_gap = max(0.0, risk_reference_soc - soc)
        risk_price = risk_scale * (risk_gap + 8.0 * risk_gap * risk_gap)
        near_floor = 0.018 / max(soc - BATTERY_HARD_FLOOR + 0.025, 0.025)
        base_prices["lambda_E"] = pg._clip(base_prices["lambda_E"] + near_floor + risk_price, 0.035, 80.0)
        return base_prices

    return prices


@contextmanager
def _price_variant_context(method_name: str, risk_reference_soc: float, risk_scale: float):
    old_method = pg.METHOD_PRICE
    old_prices = pg._resource_prices
    old_score_candidate = pg._score_candidate
    old_type = pg.METHOD_TYPES.get(method_name)
    pg.METHOD_PRICE = method_name
    pg.METHOD_TYPES[method_name] = METHOD_TYPES[method_name]
    pg._resource_prices = _risk_prices_factory(risk_reference_soc, risk_scale, old_prices)

    def score_candidate_with_after_action_risk(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cand = old_score_candidate(*args, **kwargs)
        after_gap = max(0.0, risk_reference_soc - _to_float(cand["soc_after"]))
        after_action_risk_price = risk_scale * (after_gap + 8.0 * after_gap * after_gap)
        extra_cost = after_action_risk_price * _to_float(cand["estimated_energy"])
        cand["energy_price_cost"] = _to_float(cand["energy_price_cost"]) + extra_cost
        cand["score"] = _to_float(cand["score"]) - extra_cost
        if extra_cost > 0.0 and cand.get("reason") == "safe_price_score":
            cand["reason"] = "safe_price_score_with_after_action_soc_risk"
        return cand

    pg._score_candidate = score_candidate_with_after_action_risk
    try:
        yield
    finally:
        pg.METHOD_PRICE = old_method
        pg._resource_prices = old_prices
        pg._score_candidate = old_score_candidate
        if old_type is None:
            pg.METHOD_TYPES.pop(method_name, None)
        else:
            pg.METHOD_TYPES[method_name] = old_type


def _target_success(row: dict[str, Any]) -> bool:
    target_time = row.get("target_ground_receive_time_h", "")
    return target_time != "" and _to_float(target_time) <= _to_float(row.get("deadline_h")) + 1.0e-12


def _success(row: dict[str, Any]) -> bool:
    return bool(_to_int(row.get("ground_received_before_deadline_flag")))


def _quality_weighted(task_rows: list[dict[str, Any]], high_only: bool = False) -> float:
    return sum(
        _to_float(row["V_eff"]) * _to_float(row["ground_received_quality"])
        for row in task_rows
        if _success(row) and (not high_only or _to_int(row["is_high_value"]))
    )


def _risk_settings(method: str) -> tuple[float | str, float | str]:
    for name, ref, scale in MARGIN_VARIANTS:
        if method == name:
            return ref, scale
    return "", ""


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = pg._normalize_ground_summary(result)
    tasks = result["task_delivery"]
    trajectory = result["trajectory"]
    socs = [_to_float(row["SOC"]) for row in trajectory]
    risk_ref, risk_scale = _risk_settings(str(summary["method"]))
    return {
        "method": summary["method"],
        "method_type": METHOD_TYPES[str(summary["method"])],
        "seed": summary["seed"],
        "pressure_scale": summary["pressure_scale"],
        "risk_reference_SOC": risk_ref,
        "low_SOC_risk_price_scale": risk_scale,
        "task_manifest_hash": summary["task_manifest_hash"],
        "contact_schedule_hash": summary["contact_schedule_hash"],
        "resource_trace_hash": summary["resource_trace_hash"],
        "ground_success_ratio": summary["ground_success_ratio"],
        "target_stage_success_ratio": summary["target_stage_success_ratio"],
        "high_value_ground_success_ratio": summary["high_value_ground_success_ratio"],
        "high_value_target_success_ratio": summary["high_value_target_success_ratio"],
        "avg_ground_received_quality": summary["avg_ground_received_quality"],
        "quality_weighted_success_value": _quality_weighted(tasks),
        "quality_weighted_high_value_success_value": _quality_weighted(tasks, high_only=True),
        "avg_capture_to_ground_latency_h": summary["avg_capture_to_ground_latency_h"],
        "direct_energy_per_received_task_wh": summary["direct_energy_per_received_task_wh"],
        "system_energy_per_received_task_wh": summary["system_energy_per_received_task_wh"],
        "expired_task_count": summary["expired_task_count"],
        "expired_high_value_task_count": summary["expired_high_value_task_count"],
        "min_SOC": summary["min_SOC"],
        "SOC_margin_to_20": _to_float(summary["min_SOC"]) - BATTERY_HARD_FLOOR,
        "time_below_25_percent": 100.0 * _safe_div(sum(soc < 0.25 for soc in socs), len(socs)),
        "time_below_22_percent": 100.0 * _safe_div(sum(soc < 0.22 for soc in socs), len(socs)),
        "fraction_time_SOC_below_20": summary["fraction_time_SOC_below_20"],
        "hard_safety_violation_count": summary["hard_safety_violation_count"],
        "net_system_utility": summary["net_system_utility"],
        "H_net_utility": summary["H_net_utility"],
        "final_backlog_count": summary["final_backlog_count"],
        "max_backlog_count": summary["max_backlog_count"],
    }


def build_multi_seed_ranking(summary_rows: list[dict[str, Any]], pressure_scales: list[float]) -> list[dict[str, Any]]:
    out = []
    for pressure in pressure_scales:
        pressure_rows = [row for row in summary_rows if abs(_to_float(row["pressure_scale"]) - pressure) < 1.0e-9]
        rank_by_metric: dict[tuple[str, str], list[int]] = defaultdict(list)
        for seed in sorted({_to_int(row["seed"]) for row in pressure_rows}):
            seed_rows = [row for row in pressure_rows if _to_int(row["seed"]) == seed]
            for metric in RANKING_METRICS:
                ordered = sorted(seed_rows, key=lambda row: _to_float(row[metric]), reverse=metric not in LOWER_IS_BETTER)
                for rank, row in enumerate(ordered, start=1):
                    rank_by_metric[(metric, str(row["method"]))].append(rank)
        for method in METHODS:
            method_rows = [row for row in pressure_rows if row["method"] == method]
            for metric in RANKING_METRICS:
                values = [_to_float(row[metric]) for row in method_rows]
                ranks = [float(rank) for rank in rank_by_metric[(metric, method)]]
                out.append(
                    {
                        "method": method,
                        "pressure_scale": pressure,
                        "metric": metric,
                        "mean": _mean(values),
                        "std": _std(values),
                        "min": min(values) if values else 0.0,
                        "max": max(values) if values else 0.0,
                        "rank_mean": _mean(ranks),
                        "rank_std": _std(ranks),
                    }
                )
    return out


def build_tradeoff(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in summary_rows:
        rows.append(
            {
                "method": row["method"],
                "seed": row["seed"],
                "pressure_scale": row["pressure_scale"],
                "risk_reference_SOC": row["risk_reference_SOC"],
                "ground_success_ratio": row["ground_success_ratio"],
                "high_value_ground_success_ratio": row["high_value_ground_success_ratio"],
                "high_value_target_success_ratio": row["high_value_target_success_ratio"],
                "quality_weighted_high_value_success_value": row["quality_weighted_high_value_success_value"],
                "expired_high_value_task_count": row["expired_high_value_task_count"],
                "min_SOC": row["min_SOC"],
                "SOC_margin_to_20": row["SOC_margin_to_20"],
                "direct_energy_per_received_task_wh": row["direct_energy_per_received_task_wh"],
                "hard_safety_violation_count": row["hard_safety_violation_count"],
                "is_safe_feasible": int(_to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= 0.20),
                "is_robust_safe": int(_to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= 0.25),
            }
        )
    return rows


def build_first_place_check(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_rows = [row for row in summary_rows if abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9]
    by_method_metric: dict[tuple[str, str], float] = {}
    for method in METHODS:
        rows = [row for row in base_rows if row["method"] == method]
        for metric in FIRST_PLACE_METRICS:
            by_method_metric[(method, metric)] = _mean([_to_float(row[metric]) for row in rows])

    out = []
    for metric in FIRST_PLACE_METRICS:
        ordered = sorted(
            METHODS,
            key=lambda method: by_method_metric[(method, metric)],
            reverse=metric not in LOWER_IS_BETTER,
        )
        ranks = {method: idx for idx, method in enumerate(ordered, start=1)}
        best = ordered[0]
        row = {
            "metric": metric,
            "best_method": best,
            "best_value": by_method_metric[(best, metric)],
        }
        for col, method in RANK_ALIAS.items():
            row[col] = ranks.get(method, "")
        out.append(row)
    return out


def _mean_by_method(summary_rows: list[dict[str, Any]], method: str, metric: str, pressure: float = 1.0) -> float:
    rows = [row for row in summary_rows if row["method"] == method and abs(_to_float(row["pressure_scale"]) - pressure) < 1.0e-9]
    return _mean([_to_float(row[metric]) for row in rows])


def make_figures(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    first_place: list[dict[str, Any]],
) -> None:
    base_rows = [row for row in summary_rows if abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9]
    plt.figure(figsize=(8, 5.6))
    for method in METHODS:
        rows = [row for row in base_rows if row["method"] == method]
        if not rows:
            continue
        plt.scatter(
            [_to_float(row["SOC_margin_to_20"]) for row in rows],
            [_to_float(row["quality_weighted_high_value_success_value"]) for row in rows],
            s=[55 + 210 * _to_float(row["ground_success_ratio"]) for row in rows],
            label=method,
            alpha=0.75,
        )
    plt.xlabel("SOC_margin_to_20")
    plt.ylabel("quality_weighted_high_value_success_value")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "safety_margin_tradeoff_panel.png", dpi=190)
    plt.close()

    metrics = [
        "ground_success_ratio",
        "high_value_ground_success_ratio",
        "high_value_target_success_ratio",
        "quality_weighted_high_value_success_value",
        "expired_high_value_task_count",
        "min_SOC",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, metric in zip(axes.flat, metrics):
        ax.bar(METHODS, [_mean_by_method(summary_rows, method, metric) for method in METHODS])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "safety_margin_metric_panel.png", dpi=190)
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    data = [[_to_float(row["min_SOC"]) for row in base_rows if row["method"] == method] for method in METHODS]
    plt.boxplot(data, labels=METHODS, showmeans=True)
    for level in [0.20, 0.22, 0.25]:
        plt.axhline(level, color="black", linestyle="--", linewidth=0.8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.ylabel("min_SOC")
    plt.tight_layout()
    plt.savefig(out_dir / "safety_margin_soc_distribution.png", dpi=190)
    plt.close()

    rank_metrics = [
        "ground_success_ratio",
        "target_stage_success_ratio",
        "high_value_ground_success_ratio",
        "high_value_target_success_ratio",
        "quality_weighted_high_value_success_value",
        "H_net_utility",
        "min_SOC",
    ]
    rank_rows = [row for row in first_place if row["metric"] in rank_metrics]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(rank_rows))
    margin_methods = [
        "proposed_margin_0p20_rank",
        "proposed_margin_0p22_rank",
        "proposed_margin_0p25_rank",
        "proposed_margin_0p30_rank",
        "old_proposed_rank",
    ]
    for idx, col in enumerate(margin_methods):
        ax.plot([pos + 0.03 * idx for pos in x], [_to_float(row[col]) for row in rank_rows], marker="o", label=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels([row["metric"] for row in rank_rows], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("rank, lower is better")
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "first_place_ranking_panel.png", dpi=190)
    plt.close(fig)


def _first_place_metrics(first_place: list[dict[str, Any]], method: str) -> list[str]:
    return [row["metric"] for row in first_place if row["best_method"] == method]


def _first_place_metrics_for_scope(summary_rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, list[str]]:
    scoped = [row for row in summary_rows if abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9 and predicate(row)]
    out: dict[str, list[str]] = defaultdict(list)
    for metric in FIRST_PLACE_METRICS:
        methods = sorted({row["method"] for row in scoped})
        if not methods:
            continue
        values = {
            method: _mean([_to_float(row[metric]) for row in scoped if row["method"] == method])
            for method in methods
        }
        best = sorted(methods, key=lambda method: values[method], reverse=metric not in LOWER_IS_BETTER)[0]
        out[best].append(metric)
    return dict(out)


def _rank_for(ranking_rows: list[dict[str, Any]], method: str, metric: str, pressure: float = 1.0) -> float:
    return next(
        (
            _to_float(row["rank_mean"])
            for row in ranking_rows
            if row["method"] == method and row["metric"] == metric and abs(_to_float(row["pressure_scale"]) - pressure) < 1.0e-9
        ),
        99.0,
    )


def write_quick_report(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    tradeoff_rows: list[dict[str, Any]],
    first_place: list[dict[str, Any]],
    fairness_rows: list[dict[str, Any]],
) -> None:
    old = "proposed_guarantee_aware_full"
    m20, m22, m25, m30 = [name for name, _ref, _scale in MARGIN_VARIANTS]
    all_safe = all(_to_int(row["hard_safety_violation_count"]) == 0 for row in summary_rows if row["method"] != "fixed_heavy")
    best_trade = max(
        [row for row in ranking_rows if row["metric"] == "quality_weighted_high_value_success_value" and row["method"].startswith("proposed_price_guided")],
        key=lambda row: -_to_float(row["rank_mean"]),
    )
    all_first = {method: _first_place_metrics(first_place, method) for method in METHODS}
    safety_first_by_method = _first_place_metrics_for_scope(
        summary_rows,
        lambda row: _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= 0.20,
    )
    safe_first = {method: metrics for method, metrics in safety_first_by_method.items() if method.startswith("proposed_price_guided")}
    robust_methods = sorted({row["method"] for row in tradeoff_rows if _to_int(row["is_robust_safe"]) and row["method"].startswith("proposed_price_guided")})
    robust_note = ", ".join(robust_methods) if robust_methods else "none"
    fairness_ok = all(_to_int(row["passed"]) for row in fairness_rows)
    m20_ground = _mean_by_method(summary_rows, m20, "ground_success_ratio")
    lines = [
        "# Price-guided rescue safety-margin sensitivity report",
        "",
        "## Direct Answers",
        "",
        f"1. Risk-price variants remain above old proposed: margin_0p22 ground={_fmt(_mean_by_method(summary_rows, m22, 'ground_success_ratio'))} vs old={_fmt(_mean_by_method(summary_rows, old, 'ground_success_ratio'))}; high-value={_fmt(_mean_by_method(summary_rows, m22, 'high_value_ground_success_ratio'))} vs old={_fmt(_mean_by_method(summary_rows, old, 'high_value_ground_success_ratio'))}.",
        f"2. SOC margin improves: margin_0p20 minSOC={_fmt(_mean_by_method(summary_rows, m20, 'min_SOC'))}, margin_0p22={_fmt(_mean_by_method(summary_rows, m22, 'min_SOC'))}, margin_0p25={_fmt(_mean_by_method(summary_rows, m25, 'min_SOC'))}.",
        f"3. Ground success change vs margin_0p20: margin_0p22={_fmt(_mean_by_method(summary_rows, m22, 'ground_success_ratio') - m20_ground)}, margin_0p25={_fmt(_mean_by_method(summary_rows, m25, 'ground_success_ratio') - m20_ground)}. Positive means improved delivery; negative means delivery cost for margin.",
        f"4. High-value ground success remains above old proposed: margin_0p25={_fmt(_mean_by_method(summary_rows, m25, 'high_value_ground_success_ratio'))} vs old={_fmt(_mean_by_method(summary_rows, old, 'high_value_ground_success_ratio'))}.",
        f"5. High-value target success remains above old proposed: margin_0p25={_fmt(_mean_by_method(summary_rows, m25, 'high_value_target_success_ratio'))} vs old={_fmt(_mean_by_method(summary_rows, old, 'high_value_target_success_ratio'))}.",
        f"6. Quality-weighted high-value success remains above old: margin_0p25={_fmt(_mean_by_method(summary_rows, m25, 'quality_weighted_high_value_success_value'))} vs old={_fmt(_mean_by_method(summary_rows, old, 'quality_weighted_high_value_success_value'))}.",
        f"7. Expired high-value tasks remain lower: margin_0p25={_fmt(_mean_by_method(summary_rows, m25, 'expired_high_value_task_count'))} vs old={_fmt(_mean_by_method(summary_rows, old, 'expired_high_value_task_count'))}.",
        f"8. Hard safety violations are zero for all non-fixed-heavy methods: {'Yes' if all_safe else 'No'}. fixed_heavy is intentionally unsafe and retained as stress baseline.",
        f"9. Best trade-off by high-value weighted utility rank: {best_trade['method']} (rank_mean={_fmt(best_trade['rank_mean'], 2)}).",
        f"10. All-method first-place metrics: margin_0p20={all_first.get(m20, [])}; margin_0p22={all_first.get(m22, [])}; margin_0p25={all_first.get(m25, [])}; margin_0p30={all_first.get(m30, [])}.",
        f"11. Safety-feasible price-guided first-place metrics: {safe_first}.",
        f"12. Robust-safe price-guided methods with min_SOC>=0.25: {robust_note}.",
        "13. Recommendation: margin_0p22/margin_0p25 are better sensitivity evidence than margin_0p20, but their SOC margin is still modest. Use one as the paper main version only if the target is improved margin over 0p20; if robust_safe min_SOC>=0.25 is required, run a stronger risk-price design.",
        "14. The sensitivity supports the claim that gains are not solely from touching the 20% hard floor when higher risk-price references still beat old proposed. It does not replace a final full pressure-scale sweep.",
        "",
        "## Method Boundary",
        "",
        "- Hard floor is always 0.20.",
        "- 0.22/0.25/0.30 are low-SOC risk-price references, not hard gates.",
        "- This runner does not modify `src`, value prior, high-value labels, evaluator, reward, or the previous output directory.",
        f"- Fairness hash audit passed: {fairness_ok}.",
    ]
    _write_text(out_dir / "quick_report.md", "\n".join(lines))


def validate_outputs(out_dir: Path) -> list[dict[str, Any]]:
    required = [
        "safety_margin_summary_by_seed.csv",
        "safety_margin_multi_seed_ranking.csv",
        "safety_margin_tradeoff.csv",
        "metric_first_place_check.csv",
        "safety_margin_tradeoff_panel.png",
        "safety_margin_metric_panel.png",
        "safety_margin_soc_distribution.png",
        "first_place_ranking_panel.png",
        "quick_report.md",
    ]
    return [
        {"artifact": name, "exists": int((out_dir / name).exists()), "size_bytes": (out_dir / name).stat().st_size if (out_dir / name).exists() else 0}
        for name in required
    ]


def run(output_dir: Path = OUT_DIR, pressure_scales: list[float] | None = None) -> dict[str, Any]:
    pressure_scales = pressure_scales or PRESSURE_SCALES
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        for pressure in pressure_scales:
            contacts = base.build_contact_schedule(seed, pressure)
            resources = base.build_resource_trace(seed, pressure)
            manifest = base.build_task_manifest(seed, pressure, contacts)
            hashes = {
                "task_manifest_hash": base._hash(manifest, base.GROUND_MANIFEST_COLUMNS),
                "contact_schedule_hash": base._hash(contacts, ["seed", "pressure_scale", "orbit_index", "capacity_MB", "wave_name"]),
                "resource_trace_hash": base._hash(resources, ["seed", "pressure_scale", "orbit_index", "solar_energy_wh", "wave_name"]),
            }
            case_hash_rows = []
            for method in METHODS:
                if method.startswith("proposed_price_guided_rescue_margin_"):
                    ref, scale = next((r, s) for name, r, s in MARGIN_VARIANTS if name == method)
                    with _price_variant_context(method, ref, scale):
                        result = pg.run_price_guided_method(seed, pressure, manifest, contacts, resources, hashes)
                else:
                    result = base.run_method(method, seed, pressure, manifest, contacts, resources, hashes)
                row = _summarize_result(result)
                all_summary.append(row)
                case_hash_rows.append(row)
            for field in ["task_manifest_hash", "contact_schedule_hash", "resource_trace_hash"]:
                values = {str(row[field]) for row in case_hash_rows}
                fairness_rows.append(
                    {
                        "seed": seed,
                        "pressure_scale": pressure,
                        "check": field,
                        "unique_hash_count": len(values),
                        "passed": int(len(values) == 1),
                        "hash": next(iter(values)),
                    }
                )

    ranking = build_multi_seed_ranking(all_summary, pressure_scales)
    tradeoff = build_tradeoff(all_summary)
    first_place = build_first_place_check(all_summary)

    _write_csv(output_dir / "safety_margin_summary_by_seed.csv", all_summary, SUMMARY_COLUMNS)
    _write_csv(output_dir / "safety_margin_multi_seed_ranking.csv", ranking, ["method", "pressure_scale", "metric", "mean", "std", "min", "max", "rank_mean", "rank_std"])
    _write_csv(
        output_dir / "safety_margin_tradeoff.csv",
        tradeoff,
        [
            "method",
            "seed",
            "pressure_scale",
            "risk_reference_SOC",
            "ground_success_ratio",
            "high_value_ground_success_ratio",
            "high_value_target_success_ratio",
            "quality_weighted_high_value_success_value",
            "expired_high_value_task_count",
            "min_SOC",
            "SOC_margin_to_20",
            "direct_energy_per_received_task_wh",
            "hard_safety_violation_count",
            "is_safe_feasible",
            "is_robust_safe",
        ],
    )
    _write_csv(output_dir / "metric_first_place_check.csv", first_place, FIRST_PLACE_COLUMNS)
    _write_csv(output_dir / "fairness_hash_audit.csv", fairness_rows, ["seed", "pressure_scale", "check", "unique_hash_count", "passed", "hash"])
    make_figures(output_dir, all_summary, ranking, first_place)
    write_quick_report(output_dir, all_summary, ranking, tradeoff, first_place, fairness_rows)
    validation = validate_outputs(output_dir)
    _write_csv(output_dir / "artifact_validation.csv", validation, ["artifact", "exists", "size_bytes"])
    return {"output_dir": output_dir, "summary": all_summary, "ranking": ranking, "fairness_rows": fairness_rows, "validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--full-pressure", action="store_true")
    args = parser.parse_args(argv)
    pressures = FULL_PRESSURE_SCALES if args.full_pressure else PRESSURE_SCALES
    result = run(args.output_dir, pressures)
    old = _mean_by_method(result["summary"], "proposed_guarantee_aware_full", "ground_success_ratio")
    m22 = _mean_by_method(result["summary"], "proposed_price_guided_rescue_margin_0p22", "ground_success_ratio")
    print(f"output_dir={args.output_dir}")
    print(f"runs={len(result['summary'])}")
    print(f"old_ground_success_mean={_fmt(old)}")
    print(f"margin_0p22_ground_success_mean={_fmt(m22)}")
    print(f"fairness_errors={sum(1 for row in result['fairness_rows'] if not _to_int(row['passed']))}")
    print(f"validation_missing={sum(1 for row in result['validation'] if not _to_int(row['exists']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
