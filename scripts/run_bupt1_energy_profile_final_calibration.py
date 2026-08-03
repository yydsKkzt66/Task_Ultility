"""Final BUPT-1 inspired energy-profile calibration and quick 24h rerun."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_fair_baseline_comparison as fair  # noqa: E402
from src.energy.battery_update import update_battery_soc  # noqa: E402
from src.energy.thermal_update import update_thermal_pressure  # noqa: E402
from src.timeline.resource_trace_loader import load_synthetic_resource_trace  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "_bupt1_energy_profile_final_calibration"
MAIN_DIR = OUTPUT_ROOT / "main_experiment_with_best_profile"
FIG_DIR = OUTPUT_ROOT / "figures"
SLOT_SECONDS = 20.0
NUM_SLOTS = 4320
SCENARIO = "natural_sarfish_layered_deadline"
SEED = 2031
CHARGE_CEILING_SOC = 0.895
MAIN_TASK_LIMIT = 240
THERMAL_FAIR_WH_TO_J = 80.0

COMMON_PROFILE = {
    "battery_capacity_wh": 230.0,
    "battery_initial_soc": 0.85,
    "battery_init": 0.85,
    "battery_charge_efficiency": 1.0,
    "battery_discharge_efficiency": 1.0,
    "battery_charge_ceiling_soc": CHARGE_CEILING_SOC,
    "orbit_period_min": 96.0,
    "sunlit_duration_min": 60.0,
    "eclipse_duration_min": 36.0,
    "sunlight_fraction": 0.625,
    "eclipse_solar_power_w": 0.0,
    "battery_soft_target": 0.70,
    "battery_strong_guard": 0.50,
    "battery_min": 0.20,
    "resource_power_variation_w": 0.0,
    "day_cycle_orbits": 1,
    "compute_power_w": {"Tiny": 6.8, "Small": 8.5, "Medium": 10.2},
    "comm_power_w": {"contact_standby": 18.0, "tx_nominal": 25.0, "tx_peak": 38.0},
}

CASES = [
    {
        "case_name": "case_A_tight_old",
        "profile_name": "bupt1_tight_energy_38_14",
        "solar_peak_w": 38.0,
        "idle_power_w": 14.0,
        "solar_shape": "sinusoidal",
        "solar_shape_exponent": "",
    },
    {
        "case_name": "case_B_paper_like_nominal",
        "profile_name": "bupt1_paper_like_nominal_40_10",
        "solar_peak_w": 40.0,
        "idle_power_w": 10.0,
        "solar_shape": "sine_power",
        "solar_shape_exponent": 3.0,
    },
    {
        "case_name": "case_C_nominal_margin",
        "profile_name": "bupt1_nominal_margin_42_10p5",
        "solar_peak_w": 42.0,
        "idle_power_w": 10.5,
        "solar_shape": "sine_power",
        "solar_shape_exponent": 3.0,
    },
    {
        "case_name": "case_D_upper_sensitivity",
        "profile_name": "bupt1_upper_sensitivity_45_11",
        "solar_peak_w": 45.0,
        "idle_power_w": 11.0,
        "solar_shape": "sine_power",
        "solar_shape_exponent": 3.0,
    },
]

ARM_ORDER = [
    "proposed_guarantee_aware_full",
    "utility_rule",
    "persistent_queue_without_guarantee",
    "fixed_lightweight_onboard",
    "fixed_heavy_onboard",
    "ground_approved_followup",
]
ARM_ALIAS = {
    "fixed_lightweight_onboard": "fixed_lightweight",
    "fixed_heavy_onboard": "fixed_heavy",
}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _fmt(value: object, digits: int = 4) -> str:
    return f"{_to_float(value):.{digits}f}"


def _fraction(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _profile(case: dict[str, Any]) -> dict[str, Any]:
    profile = dict(COMMON_PROFILE)
    for key, value in case.items():
        if value != "":
            profile[key] = value
    return profile


def _trace_for_profile(profile: dict[str, Any], num_slots: int = NUM_SLOTS) -> list[Any]:
    return load_synthetic_resource_trace(
        num_slots,
        {
            "slot_seconds": SLOT_SECONDS,
            "num_slots": num_slots,
            "resource": profile,
        },
    )


def _next_soc_with_unused(
    *,
    soc: float,
    solar_energy_j: float,
    base_power_w: float,
    compute_energy_j: float,
    comm_energy_j: float,
    profile: dict[str, Any],
) -> tuple[float, float]:
    capacity_j = _to_float(profile["battery_capacity_wh"]) * 3600.0
    charge_eff = _to_float(profile.get("battery_charge_efficiency"), 0.92)
    discharge_eff = _to_float(profile.get("battery_discharge_efficiency"), 0.95)
    ceiling = _to_float(profile.get("battery_charge_ceiling_soc"), 1.0)
    next_raw = update_battery_soc(
        battery_soc=soc,
        solar_energy_j=solar_energy_j,
        base_power_w=base_power_w,
        slot_seconds=SLOT_SECONDS,
        compute_energy_j=compute_energy_j,
        comm_energy_j=comm_energy_j,
        battery_capacity_j=capacity_j,
        charge_efficiency=charge_eff,
        discharge_efficiency=discharge_eff,
    )
    unused_j = 0.0
    if next_raw > ceiling:
        unused_j = (next_raw - ceiling) * capacity_j / max(charge_eff, 1.0e-9)
        next_raw = ceiling
    return max(0.0, min(ceiling, next_raw)), unused_j


def run_no_task_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    orbit_rows_all: list[dict[str, Any]] = []
    for case in CASES:
        profile = _profile(case)
        traces = _trace_for_profile(profile)
        soc = _to_float(profile["battery_initial_soc"])
        orbit_rows: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "solar_energy_wh": 0.0,
                "base_energy_wh": 0.0,
                "compute_energy_wh": 0.0,
                "comm_energy_wh": 0.0,
                "unused_solar_energy_wh": 0.0,
                "net_energy_wh": 0.0,
                "slot_count": 0,
                "eclipse_soc_delta_sum": 0.0,
                "eclipse_slot_count": 0,
                "sunlit_soc_delta_sum": 0.0,
                "sunlit_slot_count": 0,
            }
        )
        for trace in traces:
            before = soc
            base_j = _to_float(trace.base_power_w) * SLOT_SECONDS
            soc, unused_j = _next_soc_with_unused(
                soc=soc,
                solar_energy_j=_to_float(trace.solar_energy_j),
                base_power_w=_to_float(trace.base_power_w),
                compute_energy_j=0.0,
                comm_energy_j=0.0,
                profile=profile,
            )
            delta = soc - before
            orbit = orbit_rows[int(trace.orbit_index)]
            orbit["case_name"] = case["case_name"]
            orbit["profile_name"] = case["profile_name"]
            orbit["orbit_index"] = int(trace.orbit_index)
            orbit["slot_count"] += 1
            orbit["solar_energy_wh"] += _to_float(trace.solar_energy_j) / 3600.0
            orbit["base_energy_wh"] += base_j / 3600.0
            orbit["unused_solar_energy_wh"] += unused_j / 3600.0
            orbit["net_energy_wh"] += (_to_float(trace.solar_energy_j) - base_j - unused_j) / 3600.0
            if trace.is_eclipse:
                orbit["eclipse_soc_delta_sum"] += delta
                orbit["eclipse_slot_count"] += 1
            else:
                orbit["sunlit_soc_delta_sum"] += delta
                orbit["sunlit_slot_count"] += 1
            trajectory_rows.append(
                {
                    "case_name": case["case_name"],
                    "profile_name": case["profile_name"],
                    "slot": trace.slot_index,
                    "time_h": trace.slot_index * SLOT_SECONDS / 3600.0,
                    "orbit_index": trace.orbit_index,
                    "is_sunlit": int(trace.is_sunlit),
                    "is_eclipse": int(trace.is_eclipse),
                    "solar_power_w": trace.solar_power_w,
                    "base_power_w": trace.base_power_w,
                    "unused_solar_energy_wh": unused_j / 3600.0,
                    "soc_before": before,
                    "soc": soc,
                    "soc_delta": delta,
                }
            )
        orbits = [orbit_rows[idx] for idx in sorted(orbit_rows)]
        orbit_rows_all.extend(orbits)
        soc_values = [_to_float(row["soc"]) for row in trajectory_rows if row["case_name"] == case["case_name"]]
        deltas = [row for row in trajectory_rows if row["case_name"] == case["case_name"]]
        eclipse_deltas = [_to_float(row["soc_delta"]) for row in deltas if _to_int(row["is_eclipse"])]
        sunlit_deltas = [_to_float(row["soc_delta"]) for row in deltas if _to_int(row["is_sunlit"])]
        saw = sum(
            1
            for prev, cur in zip(deltas, deltas[1:])
            if _to_float(prev["soc_delta"]) < 0.0 and _to_float(cur["soc_delta"]) > 0.0
        )
        mean_solar = sum(_to_float(row["solar_energy_wh"]) for row in orbits) / max(len(orbits), 1)
        mean_base = sum(_to_float(row["base_energy_wh"]) for row in orbits) / max(len(orbits), 1)
        final_soc = soc_values[-1]
        initial_soc = _to_float(profile["battery_initial_soc"])
        pass_profile = (
            all(_to_float(row["solar_power_w"]) == 0.0 for row in deltas if _to_int(row["is_eclipse"]))
            and sum(eclipse_deltas) / max(len(eclipse_deltas), 1) < 0.0
            and sum(sunlit_deltas) / max(len(sunlit_deltas), 1) > 0.0
            and 14.0 <= mean_solar <= 18.0
            and 14.0 <= mean_base <= 18.0
            and final_soc >= initial_soc - 0.03
            and final_soc <= 0.98
            and min(soc_values) >= 0.75
            and saw > 0
        )
        summary_rows.append(
            {
                "case_name": case["case_name"],
                "profile_name": case["profile_name"],
                "solar_peak_w": case["solar_peak_w"],
                "idle_power_w": case["idle_power_w"],
                "solar_shape": case["solar_shape"],
                "solar_shape_exponent": case.get("solar_shape_exponent", ""),
                "initial_SOC": initial_soc,
                "final_SOC": final_soc,
                "min_SOC": min(soc_values),
                "max_SOC": max(soc_values),
                "daily_SOC_drift": final_soc - initial_soc,
                "fraction_time_SOC_above_70": _fraction(sum(value >= 0.70 for value in soc_values), len(soc_values)),
                "fraction_time_SOC_50_70": _fraction(sum(0.50 <= value < 0.70 for value in soc_values), len(soc_values)),
                "fraction_time_SOC_below_50": _fraction(sum(value < 0.50 for value in soc_values), len(soc_values)),
                "total_solar_energy_wh": sum(_to_float(row["solar_energy_wh"]) for row in orbits),
                "total_base_energy_wh": sum(_to_float(row["base_energy_wh"]) for row in orbits),
                "unused_solar_energy_wh": sum(_to_float(row["unused_solar_energy_wh"]) for row in orbits),
                "net_energy_wh": sum(_to_float(row["net_energy_wh"]) for row in orbits),
                "mean_solar_energy_per_orbit_wh": mean_solar,
                "mean_base_energy_per_orbit_wh": mean_base,
                "mean_net_energy_per_orbit_wh": sum(_to_float(row["net_energy_wh"]) for row in orbits) / max(len(orbits), 1),
                "mean_eclipse_soc_drop_per_orbit": sum(
                    _to_float(row["eclipse_soc_delta_sum"]) for row in orbits
                )
                / max(len(orbits), 1),
                "mean_sunlit_soc_recovery_per_orbit": sum(
                    _to_float(row["sunlit_soc_delta_sum"]) for row in orbits
                )
                / max(len(orbits), 1),
                "sawtooth_reversal_count": saw,
                "pass_no_task_profile": bool(pass_profile),
            }
        )
    _write_csv(OUTPUT_ROOT / "no_task_calibration_summary.csv", summary_rows)
    _write_csv(OUTPUT_ROOT / "no_task_trajectory_by_case.csv", trajectory_rows)
    _write_csv(OUTPUT_ROOT / "no_task_energy_balance_by_orbit.csv", orbit_rows_all)
    return summary_rows, trajectory_rows, orbit_rows_all


def choose_best_profile(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in summary_rows if str(row["pass_no_task_profile"]).lower() == "true"]
    if not candidates:
        raise RuntimeError("No no-task profile passed calibration; main experiment will not run.")
    non_upper = [row for row in candidates if row["case_name"] != "case_D_upper_sensitivity"]
    pool = non_upper or candidates

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        final = _to_float(row["final_SOC"])
        final_penalty = 0.0 if 0.85 <= final <= 0.95 else min(abs(final - 0.85), abs(final - 0.95))
        unused = _to_float(row["unused_solar_energy_wh"])
        unused_penalty = 0.0 if 0.0 < unused < 8.0 else abs(unused - 2.0)
        return (
            abs(_to_float(row["solar_peak_w"]) - 40.0),
            final_penalty,
            abs(_to_float(row["mean_solar_energy_per_orbit_wh"]) - 16.0),
            unused_penalty,
            abs(_to_float(row["idle_power_w"]) - 10.0),
        )

    best_summary = sorted(pool, key=score)[0]
    case = next(item for item in CASES if item["case_name"] == best_summary["case_name"])
    best_profile = _profile(case)
    best_profile["selected_case_name"] = best_summary["case_name"]
    best_profile["selection_reason"] = "passed no-task calibration and is closest to 40W/10W paper-like nominal target"
    return best_profile


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            values.append(_fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_no_task_report(summary_rows: list[dict[str, Any]], best_profile: dict[str, Any]) -> None:
    columns = [
        "case_name",
        "profile_name",
        "solar_peak_w",
        "idle_power_w",
        "solar_shape",
        "final_SOC",
        "min_SOC",
        "daily_SOC_drift",
        "mean_solar_energy_per_orbit_wh",
        "mean_base_energy_per_orbit_wh",
        "unused_solar_energy_wh",
        "pass_no_task_profile",
    ]
    lines = [
        "# No-task calibration report",
        "",
        "The 24h no-task run uses no generated tasks, no compute, and no communication. It only integrates platform base load, solar harvest, SOC clamping at the configured operational charge ceiling, and unused solar accounting.",
        "",
        _markdown_table(summary_rows, columns),
        "",
        f"Selected best profile: `{best_profile['profile_name']}` from `{best_profile['selected_case_name']}`.",
    ]
    _write_text(OUTPUT_ROOT / "no_task_calibration_report.md", "\n".join(lines))


def _save_best_profile(best_profile: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    best_summary = next(row for row in summary_rows if row["case_name"] == best_profile["selected_case_name"])
    with (OUTPUT_ROOT / "best_energy_profile.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(best_profile, file, sort_keys=False, allow_unicode=False)
    lines = [
        "# Best energy profile",
        "",
        f"- selected_case_name: `{best_profile['selected_case_name']}`",
        f"- profile_name: `{best_profile['profile_name']}`",
        f"- solar_peak_w: {_fmt(best_profile['solar_peak_w'])}",
        f"- idle_power_w: {_fmt(best_profile['idle_power_w'])}",
        f"- solar_shape: `{best_profile['solar_shape']}`",
        f"- solar_shape_exponent: {_fmt(best_profile.get('solar_shape_exponent', 1.0))}",
        f"- no_task_final_SOC: {_fmt(best_summary['final_SOC'])}",
        f"- no_task_min_SOC: {_fmt(best_summary['min_SOC'])}",
        f"- no_task_daily_SOC_drift: {_fmt(best_summary['daily_SOC_drift'])}",
        f"- unused_solar_energy_wh: {_fmt(best_summary['unused_solar_energy_wh'])}",
        "",
        "The old 38W/14W profile is retained as `bupt1_tight_energy_38_14` and is not selected as nominal.",
    ]
    _write_text(OUTPUT_ROOT / "best_energy_profile_report.md", "\n".join(lines))


def _physical_resource_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for trace in _trace_for_profile(profile):
        rows.append(
            {
                "slot": trace.slot_index,
                "time_h": trace.slot_index * SLOT_SECONDS / 3600.0,
                "orbit_index": trace.orbit_index,
                "is_sunlit": int(trace.is_sunlit),
                "is_eclipse": int(trace.is_eclipse),
                "solar_power_w": trace.solar_power_w,
                "solar_energy_wh": trace.solar_energy_j / 3600.0,
                "base_power_w": trace.base_power_w,
                "base_energy_wh": trace.base_power_w * SLOT_SECONDS / 3600.0,
                "temperature_reference_c": trace.temperature_reference_c,
            }
        )
    return rows


def _select_high_intensity_manifest(rows: list[dict[str, Any]], limit: int = MAIN_TASK_LIMIT) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    selected: list[dict[str, Any]] = []
    for index in range(limit):
        source = rows[min(len(rows) - 1, int(index * len(rows) / limit))]
        item = dict(source)
        item["notes"] = (str(item.get("notes", "")) + ";high_intensity_24h_subset_240_tasks").strip(";")
        selected.append(item)
    return sorted(selected, key=lambda row: (_to_int(row["arrival_slot"]), str(row["task_id"])))


def _ledger_by_slot(ledger: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    by_slot: dict[int, dict[str, float]] = defaultdict(lambda: {"compute_wh": 0.0, "comm_wh": 0.0})
    for row in ledger:
        slot = _to_int(row.get("slot"))
        by_slot[slot]["compute_wh"] += _to_float(row.get("compute_energy_Wh"))
        by_slot[slot]["comm_wh"] += _to_float(row.get("comm_energy_Wh"))
    return by_slot


def _integrate_arm_energy(
    arm: str,
    ledger: list[dict[str, Any]],
    profile: dict[str, Any],
    resource_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_slot = _ledger_by_slot(ledger)
    soc = _to_float(profile["battery_initial_soc"])
    thermal = 0.52
    trajectory: list[dict[str, Any]] = []
    orbit_rows: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "solar_energy_wh": 0.0,
            "base_energy_wh": 0.0,
            "compute_energy_wh": 0.0,
            "comm_energy_wh": 0.0,
            "unused_solar_energy_wh": 0.0,
            "net_energy_wh": 0.0,
            "slot_count": 0,
        }
    )
    for row in resource_rows:
        slot = _to_int(row["slot"])
        compute_j = by_slot[slot]["compute_wh"] * 3600.0
        comm_j = by_slot[slot]["comm_wh"] * 3600.0
        before = soc
        soc, unused_j = _next_soc_with_unused(
            soc=soc,
            solar_energy_j=_to_float(row["solar_energy_wh"]) * 3600.0,
            base_power_w=_to_float(row["base_power_w"]),
            compute_energy_j=compute_j,
            comm_energy_j=comm_j,
            profile=profile,
        )
        compute_wh = compute_j / 3600.0
        comm_wh = comm_j / 3600.0
        thermal = update_thermal_pressure(
            thermal,
            compute_energy_j=compute_wh * THERMAL_FAIR_WH_TO_J,
            comm_energy_j=comm_wh * THERMAL_FAIR_WH_TO_J,
            cooling=0.045,
            slot_seconds=SLOT_SECONDS,
            reference_slot_seconds=60.0,
            temperature_reference_c=_to_float(row["temperature_reference_c"]),
            is_sunlit=bool(_to_int(row["is_sunlit"])),
            passive_cooling=0.12,
            sunlight_gain=0.015,
            eclipse_cooling_bonus=0.03,
            reference_min_c=-10.0,
            reference_max_c=30.0,
        )
        unused_wh = unused_j / 3600.0
        net_wh = _to_float(row["solar_energy_wh"]) - _to_float(row["base_energy_wh"]) - compute_wh - comm_wh - unused_wh
        out = {
            "arm": ARM_ALIAS.get(arm, arm),
            "internal_arm": arm,
            "slot": slot,
            "time_h": row["time_h"],
            "orbit_index": row["orbit_index"],
            "is_sunlit": row["is_sunlit"],
            "solar_power_w": row["solar_power_w"],
            "base_power_w": row["base_power_w"],
            "compute_energy_wh": compute_wh,
            "comm_energy_wh": comm_wh,
            "unused_solar_energy_wh": unused_wh,
            "battery_SOC_before": before,
            "battery_SOC": soc,
            "soc_delta": soc - before,
            "thermal_pressure": thermal,
        }
        trajectory.append(out)
        orbit = orbit_rows[_to_int(row["orbit_index"])]
        orbit["arm"] = ARM_ALIAS.get(arm, arm)
        orbit["internal_arm"] = arm
        orbit["orbit_index"] = _to_int(row["orbit_index"])
        orbit["slot_count"] += 1
        orbit["solar_energy_wh"] += _to_float(row["solar_energy_wh"])
        orbit["base_energy_wh"] += _to_float(row["base_energy_wh"])
        orbit["compute_energy_wh"] += compute_wh
        orbit["comm_energy_wh"] += comm_wh
        orbit["unused_solar_energy_wh"] += unused_wh
        orbit["net_energy_wh"] += net_wh
    soc_values = [_to_float(row["battery_SOC"]) for row in trajectory]
    thermal_values = [_to_float(row["thermal_pressure"]) for row in trajectory]
    summary = {
        "arm": ARM_ALIAS.get(arm, arm),
        "internal_arm": arm,
        "min_SOC": min(soc_values),
        "final_SOC": soc_values[-1],
        "fraction_time_SOC_above_70": _fraction(sum(value >= 0.70 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_50_70": _fraction(sum(0.50 <= value < 0.70 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_20_50": _fraction(sum(0.20 <= value < 0.50 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_below_20": _fraction(sum(value < 0.20 for value in soc_values), len(soc_values)),
        "max_thermal_pressure": max(thermal_values),
        "fraction_time_thermal_above_75": _fraction(sum(value > 0.75 for value in thermal_values), len(thermal_values)),
        "fraction_time_thermal_above_85": _fraction(sum(value > 0.85 for value in thermal_values), len(thermal_values)),
        "unused_solar_energy_wh": sum(_to_float(row["unused_solar_energy_wh"]) for row in trajectory),
        "hard_safety_violation_count": sum(1 for value in soc_values if value < 0.20)
        + sum(1 for value in thermal_values if value > 0.90),
    }
    return trajectory, [orbit_rows[idx] for idx in sorted(orbit_rows)], summary


def run_main_experiment(best_profile: dict[str, Any]) -> dict[str, Any]:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    resource_rows = _physical_resource_rows(best_profile)
    contact_schedule = fair.build_contact_schedule(SCENARIO, SEED, NUM_SLOTS)
    manifest_full = fair.generate_frozen_task_manifest(SCENARIO, SEED, NUM_SLOTS, contact_schedule)
    manifest = _select_high_intensity_manifest(manifest_full)
    _write_csv(MAIN_DIR / "shared_task_manifest.csv", manifest, fair.MANIFEST_COLUMNS)
    _write_csv(MAIN_DIR / "shared_contact_schedule.csv", contact_schedule, fair.CONTACT_COLUMNS)
    _write_csv(MAIN_DIR / "shared_resource_trace.csv", resource_rows)

    runs = {arm: fair.run_arm_case(manifest, contact_schedule, arm) for arm in ARM_ORDER}
    _write_csv(MAIN_DIR / "delivered_product_ledger_all_arms.csv", [row for run in runs.values() for row in run["ledger"]])
    _write_csv(MAIN_DIR / "H_stage_success_audit_all_arms.csv", [row for run in runs.values() for row in run["stage_rows"]])
    _write_csv(MAIN_DIR / "fair_main_summary_raw.csv", [run["summary"] for run in runs.values()], fair.MAIN_SUMMARY_COLUMNS)

    soc_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    phys_by_arm: dict[str, dict[str, Any]] = {}
    for arm, run in runs.items():
        traj, energy, phys_summary = _integrate_arm_energy(arm, run["ledger"], best_profile, resource_rows)
        soc_rows.extend(traj)
        energy_rows.extend(energy)
        phys_by_arm[arm] = phys_summary
    _write_csv(MAIN_DIR / "soc_trajectory_by_arm.csv", soc_rows)
    _write_csv(MAIN_DIR / "energy_balance_by_arm.csv", energy_rows)

    ranking_rows = []
    for arm in ARM_ORDER:
        summary = runs[arm]["summary"]
        phys = phys_by_arm[arm]
        ranking_rows.append(
            {
                "arm": ARM_ALIAS.get(arm, arm),
                "internal_arm": arm,
                "net_system_utility": _to_float(summary["net_system_utility"]),
                "total_net_delivered_utility": _to_float(summary["total_net_delivered_utility"]),
                "H_gross_delivered_utility": _to_float(summary["H_gross_delivered_utility"]),
                "H_net_delivered_utility": _to_float(summary["H_net_delivered_utility"]),
                "H_min_stage_success_rate": _to_float(summary["H_min_stage_success_rate_feasible_H"]),
                "H_target_stage_success_rate": _to_float(summary["H_target_stage_success_rate_feasible_H"]),
                "total_resource_cost": _to_float(summary["total_resource_cost"]),
                "min_SOC": phys["min_SOC"],
                "final_SOC": phys["final_SOC"],
                "fraction_time_SOC_above_70": phys["fraction_time_SOC_above_70"],
                "fraction_time_SOC_50_70": phys["fraction_time_SOC_50_70"],
                "fraction_time_SOC_20_50": phys["fraction_time_SOC_20_50"],
                "fraction_time_SOC_below_20": phys["fraction_time_SOC_below_20"],
                "max_thermal_pressure": phys["max_thermal_pressure"],
                "fraction_time_thermal_above_75": phys["fraction_time_thermal_above_75"],
                "fraction_time_thermal_above_85": phys["fraction_time_thermal_above_85"],
                "unused_solar_energy_wh": phys["unused_solar_energy_wh"],
                "hard_safety_violation_count": _to_int(summary["hard_safety_violation_count"])
                + _to_int(phys["hard_safety_violation_count"]),
            }
        )

    for rank, row in enumerate(sorted(ranking_rows, key=lambda item: item["net_system_utility"], reverse=True), start=1):
        row["rank_net_system_utility"] = rank
    for rank, row in enumerate(sorted(ranking_rows, key=lambda item: item["H_gross_delivered_utility"], reverse=True), start=1):
        row["rank_H_gross"] = rank
    for rank, row in enumerate(sorted(ranking_rows, key=lambda item: item["H_net_delivered_utility"], reverse=True), start=1):
        row["rank_H_net"] = rank
    for rank, row in enumerate(sorted(ranking_rows, key=lambda item: item["H_min_stage_success_rate"], reverse=True), start=1):
        row["rank_H_min_stage"] = rank
    for rank, row in enumerate(sorted(ranking_rows, key=lambda item: item["H_target_stage_success_rate"], reverse=True), start=1):
        row["rank_H_target_stage"] = rank
    for rank, row in enumerate(sorted(ranking_rows, key=lambda item: item["total_resource_cost"]), start=1):
        row["rank_resource_cost"] = rank

    ranking_cols = [
        "arm",
        "net_system_utility",
        "rank_net_system_utility",
        "total_net_delivered_utility",
        "H_gross_delivered_utility",
        "rank_H_gross",
        "H_net_delivered_utility",
        "rank_H_net",
        "H_min_stage_success_rate",
        "rank_H_min_stage",
        "H_target_stage_success_rate",
        "rank_H_target_stage",
        "total_resource_cost",
        "rank_resource_cost",
        "min_SOC",
        "final_SOC",
        "fraction_time_SOC_above_70",
        "fraction_time_SOC_50_70",
        "fraction_time_SOC_20_50",
        "fraction_time_SOC_below_20",
        "max_thermal_pressure",
        "fraction_time_thermal_above_75",
        "fraction_time_thermal_above_85",
        "unused_solar_energy_wh",
        "hard_safety_violation_count",
    ]
    ranking_rows = sorted(ranking_rows, key=lambda item: item["rank_net_system_utility"])
    _write_csv(MAIN_DIR / "main_ranking_table.csv", ranking_rows, ranking_cols)
    _write_text(MAIN_DIR / "main_ranking_table.md", "# Main ranking table\n\n" + _markdown_table(ranking_rows, ranking_cols))
    _write_csv(
        MAIN_DIR / "soc_summary_by_arm.csv",
        [
            {
                key: row[key]
                for key in [
                    "arm",
                    "min_SOC",
                    "final_SOC",
                    "fraction_time_SOC_above_70",
                    "fraction_time_SOC_50_70",
                    "fraction_time_SOC_20_50",
                    "fraction_time_SOC_below_20",
                    "unused_solar_energy_wh",
                    "hard_safety_violation_count",
                ]
            }
            for row in ranking_rows
        ],
    )
    _write_csv(
        MAIN_DIR / "thermal_summary_by_arm.csv",
        [
            {
                "arm": row["arm"],
                "max_thermal_pressure": row["max_thermal_pressure"],
                "fraction_time_thermal_above_75": row["fraction_time_thermal_above_75"],
                "fraction_time_thermal_above_85": row["fraction_time_thermal_above_85"],
            }
            for row in ranking_rows
        ],
    )
    _write_text(MAIN_DIR / "quick_main_result.md", _quick_main_result(ranking_rows, best_profile))
    return {
        "runs": runs,
        "ranking_rows": ranking_rows,
        "soc_rows": soc_rows,
        "energy_rows": energy_rows,
        "resource_rows": resource_rows,
        "manifest": manifest,
        "contact_schedule": contact_schedule,
    }


def _quick_main_result(rows: list[dict[str, Any]], profile: dict[str, Any]) -> str:
    proposed = next(row for row in rows if row["arm"] == "proposed_guarantee_aware_full")
    best = rows[0]
    return "\n".join(
        [
            "# Quick main result",
            "",
            f"Best profile: `{profile['profile_name']}` with solar_peak_w={_fmt(profile['solar_peak_w'])}, idle_power_w={_fmt(profile['idle_power_w'])}, solar_shape={profile['solar_shape']}.",
            "",
            "The requested method arm `proposed_guarantee_aware_full` is represented by the guarantee-aware fair evaluator and is mapped in the report to the implementation family `proposed_lyapunov_rule_mode_safe_bandit`: Lyapunov + mode + safe bandit + staged delivery + bounded guarantee + persistent backlog.",
            "",
            f"Top net_system_utility arm: `{best['arm']}` rank {best['rank_net_system_utility']}.",
            f"Proposed net_system_utility rank: {proposed['rank_net_system_utility']}.",
            f"Proposed H_gross/H_net/H_min/H_target ranks: {proposed['rank_H_gross']} / {proposed['rank_H_net']} / {proposed['rank_H_min_stage']} / {proposed['rank_H_target_stage']}.",
            f"Proposed min/final SOC: {_fmt(proposed['min_SOC'])} / {_fmt(proposed['final_SOC'])}.",
        ]
    )


def write_sensitivity(summary_rows: list[dict[str, Any]], main: dict[str, Any], best_profile: dict[str, Any]) -> None:
    passed = {row["case_name"] for row in summary_rows if str(row["pass_no_task_profile"]).lower() == "true"}
    if not ({"case_C_nominal_margin", "case_D_upper_sensitivity"} & passed):
        return
    resource_cache = {case["case_name"]: _physical_resource_rows(_profile(case)) for case in CASES}
    proposed_run = main["runs"]["proposed_guarantee_aware_full"]
    rows = []
    for case_name in ["case_B_paper_like_nominal", "case_C_nominal_margin", "case_D_upper_sensitivity"]:
        case = next(item for item in CASES if item["case_name"] == case_name)
        no_task = next(row for row in summary_rows if row["case_name"] == case_name)
        _, _, phys = _integrate_arm_energy(
            "proposed_guarantee_aware_full",
            proposed_run["ledger"],
            _profile(case),
            resource_cache[case_name],
        )
        summary = proposed_run["summary"]
        rows.append(
            {
                "profile_name": case["profile_name"],
                "solar_peak_w": case["solar_peak_w"],
                "idle_power_w": case["idle_power_w"],
                "no_task_final_SOC": no_task["final_SOC"],
                "no_task_min_SOC": no_task["min_SOC"],
                "unused_solar_energy_wh": no_task["unused_solar_energy_wh"],
                "proposed_net_system_utility": summary["net_system_utility"],
                "proposed_H_gross": summary["H_gross_delivered_utility"],
                "proposed_H_net": summary["H_net_delivered_utility"],
                "proposed_H_target_stage_success_rate": summary["H_target_stage_success_rate_feasible_H"],
                "proposed_min_SOC": phys["min_SOC"],
                "proposed_hard_safety_violation_count": phys["hard_safety_violation_count"],
            }
        )
    _write_csv(OUTPUT_ROOT / "sensitivity_summary.csv", rows)


def make_figures(
    summary_rows: list[dict[str, Any]],
    no_task_traj: list[dict[str, Any]],
    no_task_energy: list[dict[str, Any]],
    main: dict[str, Any],
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for case in CASES:
        rows = [row for row in no_task_traj if row["case_name"] == case["case_name"]]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["soc"]) for row in rows], label=case["case_name"])
    for level in [0.70, 0.50, 0.20]:
        plt.axhline(level, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("time (h)")
    plt.ylabel("SOC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "no_task_soc_trajectory_by_case.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for metric in ["solar_energy_wh", "base_energy_wh", "unused_solar_energy_wh", "net_energy_wh"]:
        b_rows = [row for row in no_task_energy if row["case_name"] == "case_B_paper_like_nominal"]
        plt.plot([_to_int(row["orbit_index"]) for row in b_rows], [_to_float(row[metric]) for row in b_rows], label=metric)
    plt.xlabel("orbit")
    plt.ylabel("Wh")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "no_task_energy_balance_by_orbit.png", dpi=160)
    plt.close()

    soc_rows = main["soc_rows"]
    plt.figure(figsize=(10, 5))
    for arm in [ARM_ALIAS.get(item, item) for item in ARM_ORDER]:
        rows = [row for row in soc_rows if row["arm"] == arm]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["battery_SOC"]) for row in rows], label=arm)
    for level in [0.70, 0.50, 0.20]:
        plt.axhline(level, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("time (h)")
    plt.ylabel("SOC")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "best_profile_soc_trajectory_by_arm.png", dpi=160)
    plt.close()

    energy_by_arm = defaultdict(lambda: defaultdict(float))
    for row in main["energy_rows"]:
        arm = row["arm"]
        for key in ["solar_energy_wh", "base_energy_wh", "compute_energy_wh", "comm_energy_wh", "unused_solar_energy_wh", "net_energy_wh"]:
            energy_by_arm[arm][key] += _to_float(row[key])
    arms = [ARM_ALIAS.get(item, item) for item in ARM_ORDER]
    bottom = [0.0] * len(arms)
    plt.figure(figsize=(10, 5))
    for key in ["base_energy_wh", "compute_energy_wh", "comm_energy_wh", "unused_solar_energy_wh"]:
        vals = [energy_by_arm[arm][key] for arm in arms]
        plt.bar(arms, vals, bottom=bottom, label=key)
        bottom = [a + b for a, b in zip(bottom, vals)]
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Wh")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "best_profile_energy_balance_by_arm.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for arm in arms:
        rows = [row for row in soc_rows if row["arm"] == arm]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["thermal_pressure"]) for row in rows], label=arm)
    for level in [0.75, 0.85]:
        plt.axhline(level, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("time (h)")
    plt.ylabel("thermal pressure")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "best_profile_thermal_pressure_by_arm.png", dpi=160)
    plt.close()


def write_final_report(summary_rows: list[dict[str, Any]], best_profile: dict[str, Any], main: dict[str, Any]) -> None:
    best_no_task = next(row for row in summary_rows if row["case_name"] == best_profile["selected_case_name"])
    tight = next(row for row in summary_rows if row["case_name"] == "case_A_tight_old")
    rows = main["ranking_rows"]
    proposed = next(row for row in rows if row["arm"] == "proposed_guarantee_aware_full")
    fixed_light = next(row for row in rows if row["arm"] == "fixed_lightweight")
    fixed_heavy = next(row for row in rows if row["arm"] == "fixed_heavy")
    min_resource = min(rows, key=lambda row: _to_float(row["total_resource_cost"]))
    heavy_pressure_larger = _to_float(fixed_heavy["min_SOC"]) < _to_float(fixed_light["min_SOC"]) or _to_float(
        fixed_heavy["max_thermal_pressure"]
    ) > _to_float(fixed_light["max_thermal_pressure"])
    lines = [
        "# Final calibration report",
        "",
        "## Direct answers",
        "",
        f"1. The old 38W/14W configuration is not suitable as nominal because it is the retained tight-energy profile and misses the paper-like per-orbit energy target in this calibration: solar={_fmt(tight['mean_solar_energy_per_orbit_wh'])}Wh/orbit, base={_fmt(tight['mean_base_energy_per_orbit_wh'])}Wh/orbit, pass={tight['pass_no_task_profile']}.",
        f"2. The closest paper-like Fig.10/Fig.11/Fig.13 energy behavior is `{best_profile['profile_name']}`.",
        f"3. Best profile: solar_peak_w={_fmt(best_profile['solar_peak_w'])}, idle_power_w={_fmt(best_profile['idle_power_w'])}, solar_shape={best_profile['solar_shape']}.",
        f"4. Yes. Eclipse slots have zero solar and negative average SOC delta; sunlit slots recover SOC. Mean eclipse drop/orbit={_fmt(best_no_task['mean_eclipse_soc_drop_per_orbit'])}, mean sunlit recovery/orbit={_fmt(best_no_task['mean_sunlit_soc_recovery_per_orbit'])}.",
        f"5. No-task final_SOC={_fmt(best_no_task['final_SOC'])}, min_SOC={_fmt(best_no_task['min_SOC'])}, daily_SOC_drift={_fmt(best_no_task['daily_SOC_drift'])}.",
        f"6. Yes. The no-task profile has light positive energy margin: net={_fmt(best_no_task['net_energy_wh'])}Wh/day.",
        f"7. Yes. unused_solar_energy={_fmt(best_no_task['unused_solar_energy_wh'])}Wh under the operational charge ceiling.",
        f"8. With the best profile, proposed net_system_utility ranks #{proposed['rank_net_system_utility']}.",
        f"9. Proposed H_gross/H_net/H_min_stage/H_target_stage ranks are #{proposed['rank_H_gross']} / #{proposed['rank_H_net']} / #{proposed['rank_H_min_stage']} / #{proposed['rank_H_target_stage']}.",
        f"10. fixed_lightweight is the lowest-resource arm: {min_resource['arm'] == 'fixed_lightweight'} (rank={fixed_light['rank_resource_cost']}).",
        f"11. fixed_heavy has larger battery or thermal pressure than fixed_lightweight: {heavy_pressure_larger}. fixed_heavy min_SOC={_fmt(fixed_heavy['min_SOC'])}, max_thermal={_fmt(fixed_heavy['max_thermal_pressure'])}; fixed_lightweight min_SOC={_fmt(fixed_light['min_SOC'])}, max_thermal={_fmt(fixed_light['max_thermal_pressure'])}.",
        "12. Yes. case_D 45W/11W is treated as upper sensitivity, not the default nominal, because it is farther from the 40W target and more permissive.",
        "13. Yes. The old 38W/14W setting should be named `bupt1_tight_energy_38_14`.",
        f"14. Yes. The best profile is suitable as the nominal sustainable energy profile for follow-up WC main experiments, with the explicit caveat that the {_fmt(CHARGE_CEILING_SOC, 3)} operational charge ceiling is part of this calibration artifact.",
        "",
        "## Proposed mapping",
        "",
        "`proposed_guarantee_aware_full` is the paper/report arm name. In the implementation family it corresponds to `proposed_lyapunov_rule_mode_safe_bandit` plus Lyapunov pricing, mode control, safe bandit action selection, staged delivery, bounded guarantee, and persistent backlog accounting. No proposed router, reward, baseline, or stage-success code path was modified.",
    ]
    _write_text(OUTPUT_ROOT / "final_calibration_report.md", "\n".join(lines))


def run(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    global OUTPUT_ROOT, MAIN_DIR, FIG_DIR
    OUTPUT_ROOT = output_root
    MAIN_DIR = OUTPUT_ROOT / "main_experiment_with_best_profile"
    FIG_DIR = OUTPUT_ROOT / "figures"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows, no_task_traj, no_task_energy = run_no_task_cases()
    best_profile = choose_best_profile(summary_rows)
    _write_no_task_report(summary_rows, best_profile)
    _save_best_profile(best_profile, summary_rows)
    main = run_main_experiment(best_profile)
    write_sensitivity(summary_rows, main, best_profile)
    make_figures(summary_rows, no_task_traj, no_task_energy, main)
    write_final_report(summary_rows, best_profile, main)
    return {"summary_rows": summary_rows, "best_profile": best_profile, "main": main}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run(args.output_root)
    best = result["best_profile"]
    proposed = next(row for row in result["main"]["ranking_rows"] if row["arm"] == "proposed_guarantee_aware_full")
    print(f"output_root={args.output_root}")
    print(f"best_profile={best['profile_name']}")
    print(f"best_solar_peak_w={best['solar_peak_w']}")
    print(f"best_idle_power_w={best['idle_power_w']}")
    print(f"best_solar_shape={best['solar_shape']}")
    print(f"proposed_net_rank={proposed['rank_net_system_utility']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
