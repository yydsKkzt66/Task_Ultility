"""Run BUPT-1 inspired no-task energy calibration."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_yaml  # noqa: E402
from src.energy.battery_update import update_battery_soc  # noqa: E402
from src.energy.soc_band_policy import classify_soc_band  # noqa: E402
from src.timeline.resource_trace_loader import load_synthetic_resource_trace  # noqa: E402


DEFAULT_PROFILE = PROJECT_ROOT / "configs" / "resource_sarfish_bupt1_balanced_long_horizon.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "_bupt1_energy_health_check"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fraction(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def run_no_task_calibration(
    profile_path: Path = DEFAULT_PROFILE,
    output_dir: Path = DEFAULT_OUTPUT,
    num_slots: int = 4320,
    slot_seconds: float = 20.0,
) -> dict[str, float]:
    profile = load_yaml(profile_path)
    timeline_cfg = {
        "slot_seconds": slot_seconds,
        "num_slots": num_slots,
        "resource": profile,
    }
    traces = load_synthetic_resource_trace(num_slots, timeline_cfg)

    capacity_j = float(profile["battery_capacity_wh"]) * 3600.0
    charge_eff = float(profile.get("battery_charge_efficiency", 0.92))
    discharge_eff = float(profile.get("battery_discharge_efficiency", 0.95))
    soc = float(profile.get("battery_initial_soc", profile.get("battery_init", 0.85)))

    rows: list[dict[str, object]] = []
    orbit_rows: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "solar_harvest_Wh": 0.0,
            "base_energy_Wh": 0.0,
            "compute_energy_Wh": 0.0,
            "comm_energy_Wh": 0.0,
            "unused_solar_energy_Wh": 0.0,
            "slot_count": 0.0,
        }
    )

    for trace in traces:
        soc_before = soc
        base_energy_j = float(trace.base_power_w) * slot_seconds
        net_input_j = float(trace.solar_energy_j) - base_energy_j
        if net_input_j >= 0.0:
            unclamped_soc = soc + net_input_j * charge_eff / capacity_j
        else:
            unclamped_soc = soc + net_input_j / max(discharge_eff, 1.0e-9) / capacity_j
        soc = update_battery_soc(
            battery_soc=soc,
            solar_energy_j=float(trace.solar_energy_j),
            base_power_w=float(trace.base_power_w),
            slot_seconds=slot_seconds,
            compute_energy_j=0.0,
            comm_energy_j=0.0,
            battery_capacity_j=capacity_j,
            charge_efficiency=charge_eff,
            discharge_efficiency=discharge_eff,
        )
        unused_solar_j = max(0.0, unclamped_soc - 1.0) * capacity_j / max(charge_eff, 1.0e-9)
        band = classify_soc_band(
            soc,
            soft_target=float(profile.get("battery_soft_target", 0.70)),
            strong_guard=float(profile.get("battery_strong_guard", 0.50)),
            hard_min=float(profile.get("battery_min", 0.20)),
        )
        rows.append(
            {
                "slot": trace.slot_index,
                "time_s": float(trace.slot_index) * slot_seconds,
                "orbit_index": trace.orbit_index,
                "is_sunlit": int(trace.is_sunlit),
                "is_eclipse": int(trace.is_eclipse),
                "solar_power_w": trace.solar_power_w,
                "solar_energy_j": trace.solar_energy_j,
                "base_power_w": trace.base_power_w,
                "base_energy_j": base_energy_j,
                "unused_solar_energy_j": unused_solar_j,
                "soc_before": soc_before,
                "soc": soc,
                "soc_delta": soc - soc_before,
                "soc_band": band,
            }
        )
        orbit = orbit_rows[int(trace.orbit_index)]
        orbit["orbit_index"] = float(trace.orbit_index)
        orbit["slot_count"] += 1.0
        orbit["solar_harvest_Wh"] += float(trace.solar_energy_j) / 3600.0
        orbit["base_energy_Wh"] += base_energy_j / 3600.0
        orbit["unused_solar_energy_Wh"] += unused_solar_j / 3600.0
        orbit.setdefault("per_orbit_SOC_start", soc_before)
        orbit["per_orbit_SOC_end"] = soc

    orbit_summary: list[dict[str, object]] = []
    for orbit_index in sorted(orbit_rows):
        orbit = orbit_rows[orbit_index]
        start = float(orbit.get("per_orbit_SOC_start", 0.0))
        end = float(orbit.get("per_orbit_SOC_end", start))
        orbit_summary.append(
            {
                "orbit_index": orbit_index,
                "slot_count": int(orbit["slot_count"]),
                "solar_harvest_Wh": orbit["solar_harvest_Wh"],
                "base_energy_Wh": orbit["base_energy_Wh"],
                "compute_energy_Wh": orbit["compute_energy_Wh"],
                "comm_energy_Wh": orbit["comm_energy_Wh"],
                "unused_solar_energy_Wh": orbit["unused_solar_energy_Wh"],
                "net_energy_delta_Wh": orbit["solar_harvest_Wh"] - orbit["base_energy_Wh"],
                "per_orbit_SOC_start": start,
                "per_orbit_SOC_end": end,
                "per_orbit_SOC_recovery": end - start,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "no_task_soc_trajectory.csv", rows)
    _write_csv(output_dir / "no_task_energy_balance_by_orbit.csv", orbit_summary)

    soc_values = [float(row["soc"]) for row in rows]
    initial_soc = float(rows[0]["soc_before"]) if rows else soc
    final_soc = float(rows[-1]["soc"]) if rows else soc
    duration_days = max(num_slots * slot_seconds / 86400.0, 1.0e-9)
    sawtooth_reversals = sum(
        1
        for previous, current in zip(rows, rows[1:])
        if float(previous["soc_delta"]) < 0.0 and float(current["soc_delta"]) > 0.0
    )
    metrics = {
        "initial_SOC": initial_soc,
        "final_SOC": final_soc,
        "min_SOC": min(soc_values) if soc_values else soc,
        "max_SOC": max(soc_values) if soc_values else soc,
        "daily_SOC_drift": (final_soc - initial_soc) / duration_days,
        "fraction_time_SOC_above_70": _fraction(sum(value >= 0.70 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_50_70": _fraction(sum(0.50 <= value < 0.70 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_20_50": _fraction(sum(0.20 <= value < 0.50 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_below_20": _fraction(sum(value < 0.20 for value in soc_values), len(soc_values)),
        "unused_solar_energy_wh": sum(float(row["unused_solar_energy_j"]) for row in rows) / 3600.0,
        "sawtooth_reversal_count": float(sawtooth_reversals),
        "monotonic_drop_detected": float(all(float(row["soc_delta"]) <= 0.0 for row in rows[1:])),
    }

    report_lines = [
        "# BUPT-1 No-Task Energy Calibration",
        "",
        f"profile: `{profile_path.relative_to(PROJECT_ROOT).as_posix() if profile_path.is_absolute() else profile_path.as_posix()}`",
        f"num_slots: {num_slots}",
        f"slot_seconds: {slot_seconds}",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        report_lines.append(f"| {key} | {value:.6f} |")
    report_lines.extend(
        [
            "",
            "## Judgment",
            f"- no_task_24h_healthy: {metrics['monotonic_drop_detected'] == 0.0 and metrics['fraction_time_SOC_below_20'] == 0.0}",
            f"- SOC_sawtooth_present: {sawtooth_reversals > 0}",
            f"- final_SOC_minus_initial_SOC: {final_soc - initial_soc:.6f}",
        ]
    )
    (output_dir / "no_task_calibration_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BUPT-1 no-task SOC calibration.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-slots", type=int, default=4320)
    parser.add_argument("--slot-seconds", type=float, default=20.0)
    args = parser.parse_args()
    run_no_task_calibration(args.profile, args.output_dir, args.num_slots, args.slot_seconds)


if __name__ == "__main__":
    main()
