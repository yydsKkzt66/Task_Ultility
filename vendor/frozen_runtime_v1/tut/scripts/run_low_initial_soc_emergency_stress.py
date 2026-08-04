"""Low-initial-SOC emergency burst plus recovery stress experiment.

Standalone, default-off runner. It reuses the existing standalone ground
delivery policy simulators, but builds an independent low-SOC emergency
scenario, output tables, figures, and quick report. It does not modify src
algorithms, value prior logic, high-value labels, evaluator code, reward code,
or older runners.
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

import scripts.run_fair_baseline_comparison as fair  # noqa: E402
import scripts.run_price_guided_rescue_ground_delivery_experiment as price  # noqa: E402
import scripts.run_realistic_workload_contact_sensitivity as realistic  # noqa: E402
import scripts.run_uneven_pressure_ground_delivery_experiment as base  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_low_initial_soc_emergency_stress_v1"
INITIAL_SOC = 0.75
WORKLOAD_SCENARIO = "low_initial_soc_emergency_burst_recovery"
CONTACT_SCENARIO = "normal_contact"
SEEDS = [0, 1, 2, 3, 4]
METHODS = [
    "proposed_price_guided_rescue",
    "proposed_guarantee_aware_full",
    "utility_rule",
    "fixed_lightweight",
    "fixed_heavy",
    "fixed_heavy_with_20pct_protection",
    "ground_approved_followup",
    "persistent_queue_without_guarantee",
]
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
METHOD_TYPES = {**base.METHOD_TYPES, "proposed_price_guided_rescue": "proposed_price_guided_rescue"}
SOC_TARGET_LOW = 0.35
SOC_TARGET_HIGH = 0.45
SOC_TARGET_CENTER = 0.40
THERMAL_WARNING = 1.0
THERMAL_LIMIT = 1.25

SUMMARY_COLUMNS = [
    "method",
    "seed",
    "initial_SOC",
    "workload_scenario",
    "contact_scenario",
    "task_manifest_hash",
    "contact_schedule_hash",
    "resource_trace_hash",
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "net_system_utility",
    "H_net_utility",
    "avg_ground_received_quality",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "expired_task_count",
    "expired_high_value_task_count",
    "min_SOC",
    "final_SOC",
    "SOC_margin_to_20",
    "time_below_50_percent",
    "time_below_40_percent",
    "time_below_25_percent",
    "hard_safety_violation_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
    "final_backlog_count",
    "max_backlog_count",
    "proposed_min_SOC_target_hit_flag",
]
METRICS = [
    "ground_success_ratio",
    "target_stage_success_ratio",
    "high_value_ground_success_ratio",
    "high_value_target_success_ratio",
    "quality_weighted_success_value",
    "quality_weighted_high_value_success_value",
    "net_system_utility",
    "H_net_utility",
    "avg_ground_received_quality",
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "expired_task_count",
    "expired_high_value_task_count",
    "min_SOC",
    "final_SOC",
    "SOC_margin_to_20",
    "time_below_50_percent",
    "time_below_40_percent",
    "time_below_25_percent",
    "hard_safety_violation_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
]
LOWER_IS_BETTER = {
    "avg_capture_to_ground_latency_h",
    "direct_energy_per_received_task_wh",
    "system_energy_per_received_task_wh",
    "expired_task_count",
    "expired_high_value_task_count",
    "time_below_50_percent",
    "time_below_40_percent",
    "time_below_25_percent",
    "hard_safety_violation_count",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "thermal_violation_count",
}


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


def _target_success(row: dict[str, Any]) -> int:
    target_time = row.get("target_ground_receive_time_h", "")
    return int(target_time not in {"", None} and _to_float(target_time) <= _to_float(row.get("deadline_h")) + 1.0e-12)


def _thermal_pressure(row: dict[str, Any]) -> float:
    compute = _to_float(row.get("compute_energy_wh"))
    comm = _to_float(row.get("comm_energy_wh"))
    backlog = min(_to_float(row.get("backlog_MB")), 55.0)
    payload_on = _to_int(row.get("payload_on"))
    return 0.44 + 0.028 * compute + 0.034 * comm + 0.065 * payload_on + 0.0045 * backlog


def _build_normal_contact(seed: int) -> list[dict[str, Any]]:
    contacts = realistic.build_contact_schedule(seed, "emergency_stress", "normal_contact")
    for row in contacts:
        row["workload_scenario"] = WORKLOAD_SCENARIO
        row["contact_scenario"] = CONTACT_SCENARIO
        row["scenario"] = "low_initial_soc_emergency_stress_v1"
    return contacts


def _build_resource_trace(seed: int, burst_solar_wh: float, contact_pressure: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in range(base.NUM_ORBITS):
        if 12 <= orbit <= 21:
            phase = "emergency_burst"
            solar = burst_solar_wh
            housekeeping = 12.8 + contact_pressure
        elif 22 <= orbit <= 36:
            phase = "recovery_period"
            solar = 34.0
            housekeeping = 11.6
        else:
            phase = "background"
            solar = 22.0
            housekeeping = 12.0
        variation = 0.97 + 0.06 * fair._stable_unit("low_soc_solar", seed, orbit)
        rows.append(
            {
                "seed": seed,
                "pressure_scale": 1.0,
                "scenario": "low_initial_soc_emergency_stress_v1",
                "workload_scenario": WORKLOAD_SCENARIO,
                "contact_scenario": CONTACT_SCENARIO,
                "orbit_index": orbit,
                "time_h": orbit * base.ORBIT_HOURS,
                "wave_name": phase,
                "workload_phase": phase,
                "deadline_pressure": phase,
                "solar_energy_wh": round(solar * variation, 5),
                "housekeeping_energy_wh": round(housekeeping, 5),
            }
        )
    return rows


def _build_manifest(seed: int, contacts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest, audit = realistic.build_task_manifest(seed, "emergency_stress", "normal_contact", contacts)
    for row in manifest:
        row["workload_scenario"] = WORKLOAD_SCENARIO
        row["contact_scenario"] = CONTACT_SCENARIO
        row["scenario"] = "low_initial_soc_emergency_stress_v1"
    for row in audit:
        row["workload_scenario"] = WORKLOAD_SCENARIO
        row["contact_scenario"] = CONTACT_SCENARIO
    return manifest, audit


def _hashes(manifest: list[dict[str, Any]], contacts: list[dict[str, Any]], resources: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "task_manifest_hash": fair._stable_hash(manifest, realistic.REALISTIC_MANIFEST_COLUMNS),
        "contact_schedule_hash": fair._stable_hash(
            contacts, ["seed", "workload_scenario", "contact_scenario", "orbit_index", "capacity_MB", "start_time_h"]
        ),
        "resource_trace_hash": fair._stable_hash(
            resources, ["seed", "workload_scenario", "contact_scenario", "orbit_index", "solar_energy_wh", "housekeeping_energy_wh"]
        ),
    }


def _run_proposed_for_calibration(
    seed: int,
    manifest: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    resources: list[dict[str, Any]],
) -> float:
    result = price.run_price_guided_method(
        seed, 1.0, manifest, contacts, resources, {"task_manifest_hash": "x", "contact_schedule_hash": "y", "resource_trace_hash": "z"}
    )
    return _to_float(result["summary"]["min_SOC"])


def _calibrated_resource_trace(
    seed: int,
    manifest: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for burst_solar in [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]:
        for contact_pressure in [0.0, 0.6, 1.2]:
            resources = _build_resource_trace(seed, burst_solar, contact_pressure)
            min_soc = _run_proposed_for_calibration(seed, manifest, contacts, resources)
            candidates.append(
                {
                    "burst_solar_wh": burst_solar,
                    "contact_pressure": contact_pressure,
                    "proposed_min_SOC": min_soc,
                    "target_distance": abs(min_soc - SOC_TARGET_CENTER),
                    "hit": int(SOC_TARGET_LOW <= min_soc <= SOC_TARGET_HIGH),
                    "resources": resources,
                }
            )
    hit_candidates = [row for row in candidates if row["hit"]]
    selected = min(hit_candidates or candidates, key=lambda row: (row["target_distance"], row["burst_solar_wh"]))
    resources = selected.pop("resources")
    return resources, selected


def _augment_summary(
    result: dict[str, Any],
    *,
    seed: int,
    manifest: list[dict[str, Any]],
    hashes: dict[str, str],
    proposed_target_hit: int,
) -> dict[str, Any]:
    tasks = result["task_delivery"]
    events = result["events"]
    trajectory = result["trajectory"]
    successes = [row for row in tasks if _to_int(row.get("ground_received_before_deadline_flag"))]
    high_tasks = [row for row in tasks if _to_int(row.get("is_high_value"))]
    high_successes = [row for row in high_tasks if _to_int(row.get("ground_received_before_deadline_flag"))]
    target_successes = [row for row in tasks if _target_success(row)]
    high_target_successes = [row for row in high_tasks if _target_success(row)]
    soc_values = [_to_float(row.get("SOC")) for row in trajectory]
    thermal_values = [_thermal_pressure(row) for row in trajectory]
    quality_value = sum(_to_float(row.get("V_eff")) * _to_float(row.get("ground_received_quality")) for row in successes)
    high_quality_value = sum(_to_float(row.get("V_eff")) * _to_float(row.get("ground_received_quality")) for row in high_successes)
    for row in trajectory:
        row["initial_SOC"] = INITIAL_SOC
        row["workload_scenario"] = WORKLOAD_SCENARIO
        row["contact_scenario"] = CONTACT_SCENARIO
        row["thermal_pressure_proxy"] = _thermal_pressure(row)
    for row in events:
        row["initial_SOC"] = INITIAL_SOC
        row["workload_scenario"] = WORKLOAD_SCENARIO
        row["contact_scenario"] = CONTACT_SCENARIO
    for row in tasks:
        row["initial_SOC"] = INITIAL_SOC
        row["workload_scenario"] = WORKLOAD_SCENARIO
        row["contact_scenario"] = CONTACT_SCENARIO
    return {
        "method": result["summary"]["method"],
        "seed": seed,
        "initial_SOC": INITIAL_SOC,
        "workload_scenario": WORKLOAD_SCENARIO,
        "contact_scenario": CONTACT_SCENARIO,
        **hashes,
        "ground_success_ratio": _safe_div(len(successes), len(tasks)),
        "target_stage_success_ratio": _safe_div(len(target_successes), len(tasks)),
        "high_value_ground_success_ratio": _safe_div(len(high_successes), len(high_tasks)),
        "high_value_target_success_ratio": _safe_div(len(high_target_successes), len(high_tasks)),
        "quality_weighted_success_value": quality_value,
        "quality_weighted_high_value_success_value": high_quality_value,
        "net_system_utility": result["summary"].get("net_system_utility", 0.0),
        "H_net_utility": result["summary"].get("H_net_utility", 0.0),
        "avg_ground_received_quality": statistics.mean([_to_float(row.get("ground_received_quality")) for row in successes]) if successes else 0.0,
        "avg_capture_to_ground_latency_h": statistics.mean([_to_float(row.get("capture_to_ground_latency_h")) for row in successes]) if successes else 0.0,
        "direct_energy_per_received_task_wh": result["summary"].get("direct_energy_per_received_task_wh", 0.0),
        "system_energy_per_received_task_wh": result["summary"].get("system_energy_per_received_task_wh", 0.0),
        "expired_task_count": len(tasks) - len(successes),
        "expired_high_value_task_count": len(high_tasks) - len(high_successes),
        "min_SOC": min(soc_values) if soc_values else 0.0,
        "final_SOC": soc_values[-1] if soc_values else 0.0,
        "SOC_margin_to_20": (min(soc_values) - base.HARD_MIN_SOC) if soc_values else 0.0,
        "time_below_50_percent": _safe_div(sum(value < 0.50 for value in soc_values), len(soc_values)),
        "time_below_40_percent": _safe_div(sum(value < 0.40 for value in soc_values), len(soc_values)),
        "time_below_25_percent": _safe_div(sum(value < 0.25 for value in soc_values), len(soc_values)),
        "hard_safety_violation_count": sum(value < base.HARD_MIN_SOC - 1.0e-12 for value in soc_values),
        "max_thermal_pressure": max(thermal_values) if thermal_values else 0.0,
        "avg_thermal_pressure": statistics.mean(thermal_values) if thermal_values else 0.0,
        "thermal_violation_count": sum(value > THERMAL_LIMIT for value in thermal_values),
        "final_backlog_count": result["summary"].get("final_backlog_count", 0),
        "max_backlog_count": result["summary"].get("max_backlog_count", 0),
        "proposed_min_SOC_target_hit_flag": proposed_target_hit,
    }


def _rank_methods(rows: list[dict[str, Any]], metric: str, methods: list[str] | None = None) -> list[tuple[str, float, int]]:
    source = [row for row in rows if methods is None or row["method"] in methods]
    means = []
    for (method,), group in _group(source, ["method"]).items():
        means.append((method, statistics.mean(_to_float(row[metric]) for row in group)))
    ranked = sorted(means, key=lambda item: item[1], reverse=metric not in LOWER_IS_BETTER)
    return [(method, value, rank) for rank, (method, value) in enumerate(ranked, start=1)]


def build_multi_seed_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (method,), group in sorted(_group(rows, ["method"]).items()):
        for metric in METRICS:
            values = [_to_float(row[metric]) for row in group]
            per_seed_ranks = []
            for seed in sorted({row["seed"] for row in rows}):
                seed_rows = [row for row in rows if row["seed"] == seed]
                per_seed_ranks.extend(rank for m, _v, rank in _rank_methods(seed_rows, metric) if m == method)
            out.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "rank_mean": statistics.mean(per_seed_ranks) if per_seed_ranks else 0.0,
                    "rank_std": statistics.stdev(per_seed_ranks) if len(per_seed_ranks) > 1 else 0.0,
                }
            )
    return out


def build_metric_first_place(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for metric in METRICS:
        ranked = _rank_methods(rows, metric)
        rank_by_method = {method: rank for method, _value, rank in ranked}
        best_method, best_value, _rank = ranked[0]
        row = {"metric": metric, "best_method": best_method, "best_value": best_value}
        for method, label in METHOD_LABELS.items():
            row[label] = rank_by_method.get(method, "")
        out.append(row)
    return out


def build_safe_method_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = {
        "all_methods": rows,
        "safety_feasible": [
            row for row in rows if _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= base.HARD_MIN_SOC
        ],
        "robust_safe": [
            row for row in rows if _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= 0.25
        ],
    }
    for ranking_scope, scope_rows in groups.items():
        methods = sorted({row["method"] for row in scope_rows})
        for metric in METRICS:
            for method, value, rank in _rank_methods(scope_rows, metric, methods):
                out.append(
                    {
                        "ranking_scope": ranking_scope,
                        "metric": metric,
                        "method": method,
                        "mean": value,
                        "rank": rank,
                    }
                )
    return out


def _bar_panel(rows: list[dict[str, Any]], path: Path, metrics: list[str], title: str) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4.6), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        values = []
        for method in METHODS:
            mrows = [row for row in rows if row["method"] == method]
            values.append(statistics.mean(_to_float(row[metric]) for row in mrows) if mrows else 0.0)
        axis.bar(range(len(METHODS)), values, color="#4f46e5")
        axis.set_title(metric)
        axis.set_xticks(range(len(METHODS)))
        axis.set_xticklabels(METHODS, rotation=65, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_figures(rows: list[dict[str, Any]], ranking: list[dict[str, Any]], output_dir: Path) -> None:
    _bar_panel(
        rows,
        output_dir / "low_initial_soc_metric_panel.png",
        [
            "ground_success_ratio",
            "high_value_ground_success_ratio",
            "high_value_target_success_ratio",
            "quality_weighted_high_value_success_value",
            "net_system_utility",
            "H_net_utility",
        ],
        "Low initial SOC delivery and utility metrics",
    )
    _bar_panel(
        rows,
        output_dir / "low_initial_soc_resource_panel.png",
        ["min_SOC", "final_SOC", "time_below_40_percent", "system_energy_per_received_task_wh", "max_thermal_pressure"],
        "Low initial SOC resource metrics",
    )
    key_rank_metrics = [
        "ground_success_ratio",
        "high_value_ground_success_ratio",
        "quality_weighted_high_value_success_value",
        "net_system_utility",
        "min_SOC",
        "max_thermal_pressure",
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    width = 0.12
    for idx, metric in enumerate(key_rank_metrics):
        values = []
        for method in METHODS:
            match = next((row for row in ranking if row["method"] == method and row["metric"] == metric), None)
            values.append(_to_float(match["rank_mean"]) if match else 0.0)
        xs = [x + idx * width for x in range(len(METHODS))]
        ax.bar(xs, values, width=width, label=metric)
    ax.set_ylabel("Mean rank")
    ax.set_xticks([x + width * 2.5 for x in range(len(METHODS))])
    ax.set_xticklabels(METHODS, rotation=65, ha="right", fontsize=7)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "low_initial_soc_ranking_panel.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for method in METHODS:
        mrows = [row for row in rows if row["method"] == method]
        ax.scatter(
            [_to_float(row["min_SOC"]) for row in mrows],
            [_to_float(row["quality_weighted_high_value_success_value"]) for row in mrows],
            s=[40 + 160 * _to_float(row["ground_success_ratio"]) for row in mrows],
            alpha=0.62,
            label=method,
        )
    ax.axvspan(SOC_TARGET_LOW, SOC_TARGET_HIGH, color="#22c55e", alpha=0.12)
    ax.set_xlabel("min_SOC")
    ax.set_ylabel("Quality-weighted high-value success")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "low_initial_soc_tradeoff_panel.png", dpi=180)
    plt.close(fig)


def _metric_firsts(rows: list[dict[str, Any]], scope_rows: list[dict[str, Any]]) -> list[str]:
    firsts = []
    for metric in METRICS:
        ranked = _rank_methods(scope_rows, metric)
        if ranked and ranked[0][0] == "proposed_price_guided_rescue":
            firsts.append(metric)
    return firsts


def build_quick_report(rows: list[dict[str, Any]], first_place: list[dict[str, Any]], safe_rank: list[dict[str, Any]], calibration: list[dict[str, Any]]) -> str:
    proposed = [row for row in rows if row["method"] == "proposed_price_guided_rescue"]
    old = [row for row in rows if row["method"] == "proposed_guarantee_aware_full"]
    utility = [row for row in rows if row["method"] == "utility_rule"]
    heavy = [row for row in rows if row["method"] == "fixed_heavy"]
    light = [row for row in rows if row["method"] == "fixed_lightweight"]
    safe_rows = [
        row for row in rows if _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= base.HARD_MIN_SOC
    ]
    robust_rows = [
        row for row in rows if _to_int(row["hard_safety_violation_count"]) == 0 and _to_float(row["min_SOC"]) >= 0.25
    ]
    proposed_firsts = [row["metric"] for row in first_place if row["best_method"] == "proposed_price_guided_rescue"]
    safe_firsts = _metric_firsts(rows, safe_rows)
    robust_firsts = _metric_firsts(rows, robust_rows)
    top_metrics = [
        "ground_success_ratio",
        "high_value_ground_success_ratio",
        "high_value_target_success_ratio",
        "quality_weighted_high_value_success_value",
        "net_system_utility",
        "H_net_utility",
    ]
    top_answers = []
    for metric in top_metrics:
        rank = next(row for row in first_place if row["metric"] == metric)["proposed_rank"]
        top_answers.append(f"{metric}=rank{rank}")
    not_first = [row["metric"] for row in first_place if row["best_method"] != "proposed_price_guided_rescue"]
    return "\n".join(
        [
            "# Low initial SOC emergency stress quick report",
            "",
            f"- initial_SOC={INITIAL_SOC}; seeds={SEEDS}; workload={WORKLOAD_SCENARIO}; contact={CONTACT_SCENARIO}.",
            "- Scenario calibration adjusts only burst resource pressure/contact-pressure proxy. It does not modify algorithms, reward, evaluator, value prior, or high-value labels.",
            "- Thermal values are thermal_pressure_proxy, not physical temperature.",
            "",
            "## Required answers",
            f"1. Proposed min_SOC range: {min(_to_float(row['min_SOC']) for row in proposed):.4f} to {max(_to_float(row['min_SOC']) for row in proposed):.4f}; target hits={sum(_to_int(row['proposed_min_SOC_target_hit_flag']) for row in proposed)}/{len(proposed)}.",
            f"2. Proposed hard safety: hard violations={sum(_to_int(row['hard_safety_violation_count']) for row in proposed)}, worst min_SOC={_fmt(min(_to_float(row['min_SOC']) for row in proposed))}.",
            f"3. Proposed first in all_methods metrics: {', '.join(proposed_firsts) if proposed_firsts else 'none'}.",
            f"4. Proposed first among safety_feasible metrics: {', '.join(safe_firsts) if safe_firsts else 'none'}.",
            f"5. Proposed first among robust_safe metrics: {', '.join(robust_firsts) if robust_firsts else 'none'}.",
            f"6. Delivery/high-value top checks: {', '.join(top_answers[:4])}.",
            f"7. Utility top checks: {', '.join(top_answers[4:])}.",
            f"8. fixed_heavy delivery may be high, but worst min_SOC={_fmt(min(_to_float(row['min_SOC']) for row in heavy))}, hard violations={sum(_to_int(row['hard_safety_violation_count']) for row in heavy)}, max thermal mean={_fmt(statistics.mean(_to_float(row['max_thermal_pressure']) for row in heavy))}.",
            f"9. fixed_lightweight is safe with worst min_SOC={_fmt(min(_to_float(row['min_SOC']) for row in light))}, but high-value success mean={_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in light))}.",
            f"10. old proposed high-value expired mean={_fmt(statistics.mean(_to_float(row['expired_high_value_task_count']) for row in old))} versus proposed {_fmt(statistics.mean(_to_float(row['expired_high_value_task_count']) for row in proposed))}.",
            f"11. utility_rule high-value success mean={_fmt(statistics.mean(_to_float(row['high_value_ground_success_ratio']) for row in utility))}, net utility mean={_fmt(statistics.mean(_to_float(row['net_system_utility']) for row in utility))}, hard violations={sum(_to_int(row['hard_safety_violation_count']) for row in utility)}.",
            f"12. Proposed is not first on: {', '.join(not_first) if not_first else 'none'}. Lower-is-better metrics often favor lightweight or protected methods; delivery metrics can favor heavier methods if they spend safety margin aggressively.",
            f"13. Trade-off conclusion: proposed is a strong delivery-safety trade-off if its safety_feasible/robust_safe ranks are top-1/top-2 on delivery and utility; see safe_method_ranking.csv for the exact scoped ranks.",
            "",
            "## Calibration audit",
            "| seed | burst_solar_wh | contact_pressure | proposed_min_SOC | target_hit |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {row['seed']} | {_fmt(row['burst_solar_wh'])} | {_fmt(row['contact_pressure'])} | {_fmt(row['proposed_min_SOC'])} | {row['hit']} |"
                for row in calibration
            ],
        ]
    )


def run_experiment(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    old_initial_soc = base.INITIAL_SOC
    base.INITIAL_SOC = INITIAL_SOC
    try:
        all_summary: list[dict[str, Any]] = []
        all_manifest: list[dict[str, Any]] = []
        all_contacts: list[dict[str, Any]] = []
        all_resources: list[dict[str, Any]] = []
        all_audit: list[dict[str, Any]] = []
        all_events: list[dict[str, Any]] = []
        all_tasks: list[dict[str, Any]] = []
        all_trajectory: list[dict[str, Any]] = []
        calibration_rows: list[dict[str, Any]] = []

        for seed in SEEDS:
            contacts = _build_normal_contact(seed)
            manifest, audit = _build_manifest(seed, contacts)
            resources, calibration = _calibrated_resource_trace(seed, manifest, contacts)
            calibration["seed"] = seed
            calibration_rows.append(calibration)
            hashes = _hashes(manifest, contacts, resources)
            proposed_target_hit = _to_int(calibration["hit"])
            all_manifest.extend(manifest)
            all_contacts.extend(contacts)
            all_resources.extend(resources)
            all_audit.extend(audit)
            for method in METHODS:
                if method == "proposed_price_guided_rescue":
                    result = price.run_price_guided_method(seed, 1.0, manifest, contacts, resources, hashes)
                else:
                    result = base.run_method(method, seed, 1.0, manifest, contacts, resources, hashes)
                summary = _augment_summary(
                    result, seed=seed, manifest=manifest, hashes=hashes, proposed_target_hit=proposed_target_hit
                )
                all_summary.append(summary)
                all_events.extend(result["events"])
                all_tasks.extend(result["task_delivery"])
                all_trajectory.extend(result["trajectory"])

        ranking = build_multi_seed_ranking(all_summary)
        first_place = build_metric_first_place(all_summary)
        safe_rank = build_safe_method_ranking(all_summary)
        _write_csv(output_dir / "low_initial_soc_summary_by_seed.csv", all_summary, SUMMARY_COLUMNS)
        _write_csv(output_dir / "low_initial_soc_multi_seed_ranking.csv", ranking)
        _write_csv(output_dir / "metric_first_place_check.csv", first_place)
        _write_csv(output_dir / "safe_method_ranking.csv", safe_rank)
        _write_csv(output_dir / "low_initial_soc_task_manifest.csv", all_manifest)
        _write_csv(output_dir / "low_initial_soc_contact_schedule.csv", all_contacts)
        _write_csv(output_dir / "low_initial_soc_resource_trace.csv", all_resources)
        _write_csv(output_dir / "low_initial_soc_deadline_contact_audit.csv", all_audit)
        _write_csv(output_dir / "low_initial_soc_ground_events.csv", all_events)
        _write_csv(output_dir / "low_initial_soc_task_delivery.csv", all_tasks)
        _write_csv(output_dir / "low_initial_soc_trajectory.csv", all_trajectory)
        _write_csv(output_dir / "low_initial_soc_calibration_audit.csv", calibration_rows)
        build_figures(all_summary, ranking, output_dir)
        _write_text(output_dir / "quick_report.md", build_quick_report(all_summary, first_place, safe_rank, calibration_rows))
    finally:
        base.INITIAL_SOC = old_initial_soc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.output_dir)
    print(f"Wrote low-initial-SOC emergency stress outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
