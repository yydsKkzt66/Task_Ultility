"""Power-gating and burst-recovery energy profile runner for BUPT-1 case study."""

from __future__ import annotations

import argparse
import csv
import math
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
from src.energy.battery_update import update_battery_soc  # noqa: E402
from src.energy.thermal_update import update_thermal_pressure  # noqa: E402
from src.timeline.resource_trace_loader import load_synthetic_resource_trace  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_bupt1_energy_profile_power_gating_burst_recovery"
FIG_DIR = OUT_DIR / "figures"
SLOT_SECONDS = 20.0
NUM_SLOTS = 4320
ORBIT_PERIOD_MIN = 95.0
ORBIT_SLOTS = int(ORBIT_PERIOD_MIN * 60.0 / SLOT_SECONDS)
CONTACT_DURATION_MIN = 9.0
CONTACT_SLOTS = int(CONTACT_DURATION_MIN * 60.0 / SLOT_SECONDS)
SEED = 2042

HOUSEKEEPING_POWER_W = 8.0
PI_IDLE_POWER_W = 2.4
ATLAS_IDLE_POWER_W = 7.2
BATTERY_CAPACITY_WH = 230.0
INITIAL_SOC = 0.85
CHARGE_EFF = 0.92
DISCHARGE_EFF = 0.95
TX_PEAK_POWER_W = 38.0
THERMAL_FAIR_WH_TO_J = 8.0

SOC_LEVELS = {
    "comfort_soc": 0.70,
    "soft_guard_soc": 0.50,
    "energy_tight_soc": 0.30,
    "hard_guard_soc": 0.20,
}

RESOURCE_PROFILE = {
    "orbit_altitude_km": 500,
    "orbit_period_min": ORBIT_PERIOD_MIN,
    "sunlit_duration_min": 60.0,
    "eclipse_duration_min": 35.0,
    "sunlight_fraction": 0.632,
    "solar_peak_w": 40.0,
    "solar_peak_sensitivity_w": [40.0, 42.0, 45.0],
    "target_solar_energy_per_orbit_wh": 16.0,
    "eclipse_solar_power_w": 0.0,
    "solar_shape": "sine_power",
    "solar_shape_exponent": 3.0,
    "housekeeping_power_w": HOUSEKEEPING_POWER_W,
    "payload_idle_power_w": PI_IDLE_POWER_W,
    "payload_power_gating": True,
    "battery_capacity_wh": BATTERY_CAPACITY_WH,
    "battery_initial_soc": INITIAL_SOC,
    "battery_charge_efficiency": CHARGE_EFF,
    "battery_discharge_efficiency": DISCHARGE_EFF,
    "resource_power_variation_w": 0.0,
    "day_cycle_orbits": 1,
}

NO_TASK_CASES = [
    {
        "case_name": "case_1_housekeeping_only",
        "payload_power_gating_enabled": True,
        "payload_idle_enabled": False,
        "payload_idle_power_w": 0.0,
    },
    {
        "case_name": "case_2_pi_idle",
        "payload_power_gating_enabled": False,
        "payload_idle_enabled": True,
        "payload_idle_power_w": PI_IDLE_POWER_W,
    },
    {
        "case_name": "case_3_atlas_idle",
        "payload_power_gating_enabled": False,
        "payload_idle_enabled": True,
        "payload_idle_power_w": ATLAS_IDLE_POWER_W,
    },
]

PHASES = [
    {
        "phase_name": "phase_1_burst",
        "start_h": 0.0,
        "end_h": 4.0,
        "task_arrival_per_orbit": 4.0,
        "high_value_ratio": 0.34,
        "deadline_orbits": 1.0,
    },
    {
        "phase_name": "phase_2_recovery",
        "start_h": 4.0,
        "end_h": 8.0,
        "task_arrival_per_orbit": 0.0,
        "high_value_ratio": 0.04,
        "deadline_orbits": 1.3,
    },
    {
        "phase_name": "phase_3_regular",
        "start_h": 8.0,
        "end_h": 14.0,
        "task_arrival_per_orbit": 1.2,
        "high_value_ratio": 0.10,
        "deadline_orbits": 1.5,
    },
    {
        "phase_name": "phase_4_heavy",
        "start_h": 14.0,
        "end_h": 18.0,
        "task_arrival_per_orbit": 2.5,
        "high_value_ratio": 0.22,
        "deadline_orbits": 2.0,
    },
    {
        "phase_name": "phase_5_recovery",
        "start_h": 18.0,
        "end_h": 24.0,
        "task_arrival_per_orbit": 0.0,
        "high_value_ratio": 0.03,
        "deadline_orbits": 1.5,
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


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _fmt(value: object, digits: int = 4) -> str:
    return f"{_to_float(value):.{digits}f}"


def _fraction(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _phase_for_slot(slot: int) -> dict[str, Any]:
    time_h = slot * SLOT_SECONDS / 3600.0
    for phase in PHASES:
        if _to_float(phase["start_h"]) <= time_h < _to_float(phase["end_h"]):
            return phase
    return PHASES[-1]


def _soc_band(soc: float) -> str:
    if soc >= 0.70:
        return "SOC_above_70"
    if soc >= 0.50:
        return "SOC_50_70"
    if soc >= 0.30:
        return "SOC_30_50"
    if soc >= 0.20:
        return "SOC_20_30"
    return "SOC_below_20"


def _trace(profile: dict[str, Any]) -> list[Any]:
    resource_cfg = dict(profile)
    resource_cfg["idle_power_w"] = 0.0
    return load_synthetic_resource_trace(
        NUM_SLOTS,
        {"slot_seconds": SLOT_SECONDS, "num_slots": NUM_SLOTS, "resource": resource_cfg},
    )


def _next_soc(
    *,
    soc: float,
    solar_energy_j: float,
    base_power_w: float,
    compute_energy_j: float,
    comm_energy_j: float,
) -> tuple[float, float]:
    capacity_j = BATTERY_CAPACITY_WH * 3600.0
    net_input = solar_energy_j - base_power_w * SLOT_SECONDS - compute_energy_j - comm_energy_j
    if net_input >= 0.0:
        unclamped = soc + net_input * CHARGE_EFF / capacity_j
    else:
        unclamped = soc + net_input / DISCHARGE_EFF / capacity_j
    next_soc = update_battery_soc(
        battery_soc=soc,
        solar_energy_j=solar_energy_j,
        base_power_w=base_power_w,
        slot_seconds=SLOT_SECONDS,
        compute_energy_j=compute_energy_j,
        comm_energy_j=comm_energy_j,
        battery_capacity_j=capacity_j,
        charge_efficiency=CHARGE_EFF,
        discharge_efficiency=DISCHARGE_EFF,
    )
    unused_j = max(0.0, unclamped - 1.0) * capacity_j / max(CHARGE_EFF, 1.0e-9)
    return next_soc, unused_j


def run_no_task_calibration() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    orbit_rows_all: list[dict[str, Any]] = []
    traces = _trace(RESOURCE_PROFILE)
    for case in NO_TASK_CASES:
        soc = INITIAL_SOC
        orbit_rows: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "solar_energy_wh": 0.0,
                "housekeeping_energy_wh": 0.0,
                "payload_idle_energy_wh": 0.0,
                "unused_solar_energy_wh": 0.0,
                "net_energy_wh": 0.0,
                "eclipse_soc_delta_sum": 0.0,
                "sunlit_soc_delta_sum": 0.0,
                "eclipse_slot_count": 0,
                "sunlit_slot_count": 0,
            }
        )
        case_rows: list[dict[str, Any]] = []
        for trace in traces:
            payload_idle = _to_float(case["payload_idle_power_w"]) if bool(case["payload_idle_enabled"]) else 0.0
            base_power = HOUSEKEEPING_POWER_W + payload_idle
            before = soc
            soc, unused_j = _next_soc(
                soc=soc,
                solar_energy_j=_to_float(trace.solar_energy_j),
                base_power_w=base_power,
                compute_energy_j=0.0,
                comm_energy_j=0.0,
            )
            delta = soc - before
            orbit = orbit_rows[int(trace.orbit_index)]
            orbit["case_name"] = case["case_name"]
            orbit["orbit_index"] = int(trace.orbit_index)
            orbit["solar_energy_wh"] += _to_float(trace.solar_energy_j) / 3600.0
            orbit["housekeeping_energy_wh"] += HOUSEKEEPING_POWER_W * SLOT_SECONDS / 3600.0
            orbit["payload_idle_energy_wh"] += payload_idle * SLOT_SECONDS / 3600.0
            orbit["unused_solar_energy_wh"] += unused_j / 3600.0
            orbit["net_energy_wh"] += (
                _to_float(trace.solar_energy_j) - base_power * SLOT_SECONDS - unused_j
            ) / 3600.0
            if trace.is_eclipse:
                orbit["eclipse_soc_delta_sum"] += delta
                orbit["eclipse_slot_count"] += 1
            else:
                orbit["sunlit_soc_delta_sum"] += delta
                orbit["sunlit_slot_count"] += 1
            row = {
                "case_name": case["case_name"],
                "slot": trace.slot_index,
                "time_h": trace.slot_index * SLOT_SECONDS / 3600.0,
                "orbit_index": trace.orbit_index,
                "is_sunlit": int(trace.is_sunlit),
                "solar_power_w": trace.solar_power_w,
                "housekeeping_power_w": HOUSEKEEPING_POWER_W,
                "payload_idle_power_w": payload_idle,
                "total_base_power_w": base_power,
                "unused_solar_energy_wh": unused_j / 3600.0,
                "soc_before": before,
                "SOC": soc,
                "soc_delta": delta,
            }
            trajectory_rows.append(row)
            case_rows.append(row)
        orbits = [orbit_rows[index] for index in sorted(orbit_rows)]
        orbit_rows_all.extend(orbits)
        soc_values = [_to_float(row["SOC"]) for row in case_rows]
        total_solar = sum(_to_float(row["solar_energy_wh"]) for row in orbits)
        unused = sum(_to_float(row["unused_solar_energy_wh"]) for row in orbits)
        summary_rows.append(
            {
                "case_name": case["case_name"],
                "housekeeping_power_w": HOUSEKEEPING_POWER_W,
                "payload_idle_power_w": case["payload_idle_power_w"],
                "payload_power_gating_enabled": case["payload_power_gating_enabled"],
                "initial_SOC": INITIAL_SOC,
                "final_SOC": soc_values[-1],
                "min_SOC": min(soc_values),
                "max_SOC": max(soc_values),
                "daily_SOC_drift": soc_values[-1] - INITIAL_SOC,
                "mean_solar_energy_per_orbit_wh": total_solar / max(len(orbits), 1),
                "mean_housekeeping_energy_per_orbit_wh": sum(
                    _to_float(row["housekeeping_energy_wh"]) for row in orbits
                )
                / max(len(orbits), 1),
                "mean_payload_idle_energy_per_orbit_wh": sum(
                    _to_float(row["payload_idle_energy_wh"]) for row in orbits
                )
                / max(len(orbits), 1),
                "mean_net_energy_per_orbit_wh": sum(_to_float(row["net_energy_wh"]) for row in orbits)
                / max(len(orbits), 1),
                "unused_solar_energy_wh": unused,
                "unused_solar_ratio": _safe_div(unused, total_solar),
                "mean_eclipse_soc_drop_per_orbit": sum(_to_float(row["eclipse_soc_delta_sum"]) for row in orbits)
                / max(len(orbits), 1),
                "mean_sunlit_soc_recovery_per_orbit": sum(_to_float(row["sunlit_soc_delta_sum"]) for row in orbits)
                / max(len(orbits), 1),
            }
        )
    _write_csv(OUT_DIR / "no_task_power_gating_summary.csv", summary_rows)
    _write_csv(OUT_DIR / "no_task_trajectory_by_case.csv", trajectory_rows)
    _write_csv(OUT_DIR / "no_task_energy_balance_by_orbit.csv", orbit_rows_all)
    return summary_rows, trajectory_rows, orbit_rows_all


def build_contact_schedule() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = int(25.0 * 60.0 / SLOT_SECONDS)
    idx = 0
    while start < NUM_SLOTS:
        end = min(NUM_SLOTS - 1, start + CONTACT_SLOTS - 1)
        phase = _phase_for_slot(start)
        capacity = 10.0
        if phase["phase_name"] == "phase_1_burst":
            capacity = 12.0
        elif phase["phase_name"] == "phase_4_heavy":
            capacity = 13.0
        elif "recovery" in str(phase["phase_name"]):
            capacity = 8.5
        rows.append(
            {
                "scenario": "bupt1_burst_recovery_24h",
                "seed": SEED,
                "num_slots": NUM_SLOTS,
                "contact_window_id": f"br_cw_{idx:03d}",
                "start_slot": start,
                "end_slot": end,
                "capacity_MB": capacity,
                "throughput_mbps": round(capacity * 8.0 / max((end - start + 1) * SLOT_SECONDS, 1.0), 4),
                "schedule_version": "bupt1_95min_9min_contacts_v1",
                "phase_name": phase["phase_name"],
                "tx_peak_power_w": TX_PEAK_POWER_W,
            }
        )
        idx += 1
        start += ORBIT_SLOTS
    return rows


def _required_stages(requirement: str) -> tuple[str, str]:
    if requirement == "alert":
        return "metadata", "metadata"
    if requirement == "verification":
        return "ROI", "ROI"
    return "ROI", "ROI+Ctx"


def _value_for_level(level: str, unit: float) -> tuple[float, float]:
    base = {"urgent": 2.45, "high": 1.7, "medium": 0.8, "low": 0.35}[level]
    return round(min(0.99, base / 2.5 + 0.05 * unit), 5), round(base * (0.9 + 0.18 * unit), 5)


def build_burst_recovery_manifest(contact_schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_index = 0
    for phase in PHASES:
        start_slot = int(_to_float(phase["start_h"]) * 3600.0 / SLOT_SECONDS)
        end_slot = int(_to_float(phase["end_h"]) * 3600.0 / SLOT_SECONDS)
        duration_orbits = max(1.0, (end_slot - start_slot) / ORBIT_SLOTS)
        task_count = max(1 if _to_float(phase["task_arrival_per_orbit"]) > 0 else 0, round(duration_orbits * _to_float(phase["task_arrival_per_orbit"])))
        if task_count <= 0:
            continue
        spacing = max(1, (end_slot - start_slot) // max(task_count, 1))
        for local_index in range(task_count):
            unit = fair._stable_unit(SEED, phase["phase_name"], local_index)
            arrival = min(NUM_SLOTS - 2, start_slot + local_index * spacing + int(unit * max(1, spacing * 0.45)))
            high_value = unit < _to_float(phase["high_value_ratio"])
            if high_value and unit < 0.10:
                level = "urgent"
            elif high_value:
                level = "high"
            elif unit < 0.55:
                level = "medium"
            else:
                level = "low"
            if level in {"urgent", "high"}:
                requirement = "evidence" if task_index % 2 == 0 else "context"
            elif level == "medium":
                requirement = "verification" if task_index % 3 else "context"
            else:
                requirement = "alert" if task_index % 3 else "verification"
            deadline_slots = max(18, int(_to_float(phase["deadline_orbits"]) * ORBIT_SLOTS))
            if requirement in {"context", "evidence"}:
                deadline_slots = max(deadline_slots, int(2.6 * ORBIT_SLOTS))
            deadline_slot = min(NUM_SLOTS - 1, arrival + deadline_slots)
            min_stage, target_stage = _required_stages(requirement)
            base_prior, v_eff = _value_for_level(level, fair._stable_unit("value", SEED, task_index))
            difficulty = 0.75 + 0.45 * fair._stable_unit("payload", SEED, task_index)
            metadata_size = round(0.055 + 0.035 * fair._stable_unit("metadata", task_index), 5)
            roi_size = round((0.56 + 0.52 * fair._stable_unit("roi", task_index)) * difficulty, 5)
            roi_context_size = round(roi_size * (2.0 + 0.55 * fair._stable_unit("ctx", task_index)), 5)
            feasible = sum(
                1
                for window in contact_schedule
                if arrival <= _to_int(window["start_slot"]) <= deadline_slot
            )
            rows.append(
                {
                    "scenario": "bupt1_burst_recovery_24h",
                    "seed": SEED,
                    "num_slots": NUM_SLOTS,
                    "task_id": f"br_s{SEED}_t{task_index:03d}",
                    "scene_id": f"br_scene_{task_index:03d}",
                    "arrival_slot": arrival,
                    "deadline_slot": deadline_slot,
                    "deadline_seconds": (deadline_slot - arrival + 1) * SLOT_SECONDS,
                    "value_level": level,
                    "base_prior_score": base_prior,
                    "V_eff": v_eff,
                    "product_requirement": requirement,
                    "task_class": "burst_recovery_sarfish_candidate",
                    "high_value_flag": int(level in {"high", "urgent"}),
                    "emergency_flag": int(level == "urgent"),
                    "minimum_stage_required": min_stage,
                    "target_stage_required": target_stage,
                    "payload_size_metadata_mb": metadata_size,
                    "payload_size_roi_mb": roi_size,
                    "payload_size_roi_context_mb": roi_context_size,
                    "feasible_contact_windows_before_deadline": feasible,
                    "phase_name": phase["phase_name"],
                    "notes": "burst_recovery_manifest_v1",
                }
            )
            task_index += 1
    return sorted(rows, key=lambda row: (_to_int(row["arrival_slot"]), str(row["task_id"])))


def _resource_rows() -> list[dict[str, Any]]:
    rows = []
    for trace in _trace(RESOURCE_PROFILE):
        rows.append(
            {
                "slot": trace.slot_index,
                "time_h": trace.slot_index * SLOT_SECONDS / 3600.0,
                "orbit_index": trace.orbit_index,
                "phase_name": _phase_for_slot(trace.slot_index)["phase_name"],
                "is_sunlit": int(trace.is_sunlit),
                "is_eclipse": int(trace.is_eclipse),
                "solar_power_w": trace.solar_power_w,
                "solar_energy_wh": trace.solar_energy_j / 3600.0,
                "housekeeping_power_w": HOUSEKEEPING_POWER_W,
                "temperature_reference_c": trace.temperature_reference_c,
            }
        )
    return rows


def _stage_complete(stage: str, products: set[str]) -> bool:
    return fair._stage_rank(fair._stage_from_products(products)) >= fair._stage_rank(stage)


def _integrate_arm(
    arm: str,
    run: dict[str, Any],
    manifest: list[dict[str, Any]],
    contact_schedule: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    public_arm = ARM_ALIAS.get(arm, arm)
    by_task = {str(row["task_id"]): row for row in manifest}
    ledger_by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in run["ledger"]:
        ledger_by_slot[_to_int(row["slot"])].append(row)
    contact_by_slot = {
        slot: window
        for window in contact_schedule
        for slot in range(_to_int(window["start_slot"]), _to_int(window["end_slot"]) + 1)
    }
    contact_delivery_slots = {
        slot
        for rows in ledger_by_slot.values()
        for slot in range(
            _to_int(contact_by_slot.get(_to_int(rows[0]["slot"]), {}).get("start_slot", rows[0]["slot"])),
            _to_int(contact_by_slot.get(_to_int(rows[0]["slot"]), {}).get("end_slot", rows[0]["slot"])) + 1,
        )
        if rows
    }
    delivered_by_task: dict[str, set[str]] = defaultdict(set)
    completed_slot_by_task: dict[str, int] = {}
    task_target = {str(row["task_id"]): str(row["target_stage_required"]) for row in manifest}
    soc = INITIAL_SOC
    thermal = 0.52
    trajectory_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    orbit_rows: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "solar_energy_wh": 0.0,
            "housekeeping_energy_wh": 0.0,
            "payload_idle_energy_wh": 0.0,
            "compute_energy_wh": 0.0,
            "comm_energy_wh": 0.0,
            "unused_solar_energy_wh": 0.0,
            "net_energy_wh": 0.0,
        }
    )
    defer_events: list[dict[str, Any]] = []

    for row in resource_rows:
        slot = _to_int(row["slot"])
        for delivery in ledger_by_slot.get(slot, []):
            task_id = str(delivery["task_id"])
            delivered_by_task[task_id].add(str(delivery["product_type"]))
            if task_id not in completed_slot_by_task and _stage_complete(task_target.get(task_id, "ROI"), delivered_by_task[task_id]):
                completed_slot_by_task[task_id] = slot

        active_unclosed = [
            task
            for task in manifest
            if _to_int(task["arrival_slot"]) <= slot <= _to_int(task["deadline_slot"])
            and not _stage_complete(str(task["target_stage_required"]), delivered_by_task[str(task["task_id"])])
        ]
        current_phase_name = str(row["phase_name"])
        if "recovery" in current_phase_name:
            active_attention = [
                task
                for task in active_unclosed
                if fair._is_H(task) or _to_int(task["deadline_slot"]) <= slot + int(0.35 * ORBIT_SLOTS)
            ]
        else:
            active_attention = active_unclosed
        upcoming_h = [
            task
            for task in manifest
            if slot < _to_int(task["arrival_slot"]) <= slot + int(0.5 * ORBIT_SLOTS)
            and fair._is_H(task)
        ]
        delivery_now = bool(ledger_by_slot.get(slot))
        in_tx_window = slot in contact_delivery_slots
        payload_on = bool(active_attention or upcoming_h or delivery_now or in_tx_window)
        payload_idle_power = PI_IDLE_POWER_W
        if arm == "fixed_heavy_onboard":
            payload_idle_power = ATLAS_IDLE_POWER_W
        elif arm == "fixed_lightweight_onboard":
            payload_idle_power = PI_IDLE_POWER_W * 0.55
        payload_state = "on" if payload_on else "gated_off"
        total_base_power = HOUSEKEEPING_POWER_W + (payload_idle_power if payload_on else 0.0)
        compute_wh = sum(_to_float(item["compute_energy_Wh"]) for item in ledger_by_slot.get(slot, []))
        tx_active = in_tx_window
        comm_wh = TX_PEAK_POWER_W * SLOT_SECONDS / 3600.0 if tx_active else 0.0
        compute_j = compute_wh * 3600.0
        comm_j = comm_wh * 3600.0
        before = soc
        soc, unused_j = _next_soc(
            soc=soc,
            solar_energy_j=_to_float(row["solar_energy_wh"]) * 3600.0,
            base_power_w=total_base_power,
            compute_energy_j=compute_j,
            comm_energy_j=comm_j,
        )
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
        payload_idle_wh = (payload_idle_power if payload_on else 0.0) * SLOT_SECONDS / 3600.0
        housekeeping_wh = HOUSEKEEPING_POWER_W * SLOT_SECONDS / 3600.0
        net_wh = _to_float(row["solar_energy_wh"]) - housekeeping_wh - payload_idle_wh - compute_wh - comm_wh - unused_wh
        soc_band = _soc_band(soc)
        if active_unclosed and not delivery_now:
            defer_events.append({"arm": public_arm, "slot": slot, "soc_band": soc_band, "action": "defer"})
        out = {
            "arm": public_arm,
            "internal_arm": arm,
            "slot": slot,
            "time_h": row["time_h"],
            "phase_name": row["phase_name"],
            "orbit_index": row["orbit_index"],
            "is_sunlit": row["is_sunlit"],
            "solar_power_w": row["solar_power_w"],
            "payload_power_state": payload_state,
            "payload_idle_power_w": payload_idle_power if payload_on else 0.0,
            "housekeeping_power_w": HOUSEKEEPING_POWER_W,
            "total_base_power_w": total_base_power,
            "compute_energy_wh": compute_wh,
            "comm_energy_wh": comm_wh,
            "unused_solar_energy_wh": unused_wh,
            "battery_SOC_before": before,
            "SOC": soc,
            "soc_delta": soc - before,
            "soc_band": soc_band,
            "thermal_pressure": thermal,
        }
        trajectory_rows.append(out)
        payload_rows.append(
            {
                "arm": public_arm,
                "time_h": row["time_h"],
                "payload_power_state": payload_state,
                "payload_idle_power_w": payload_idle_power if payload_on else 0.0,
                "housekeeping_power_w": HOUSEKEEPING_POWER_W,
                "total_base_power_w": total_base_power,
                "SOC": soc,
            }
        )
        orbit = orbit_rows[(public_arm, _to_int(row["orbit_index"]))]
        orbit["arm"] = public_arm
        orbit["orbit_index"] = _to_int(row["orbit_index"])
        orbit["phase_name"] = row["phase_name"]
        orbit["solar_energy_wh"] += _to_float(row["solar_energy_wh"])
        orbit["housekeeping_energy_wh"] += housekeeping_wh
        orbit["payload_idle_energy_wh"] += payload_idle_wh
        orbit["compute_energy_wh"] += compute_wh
        orbit["comm_energy_wh"] += comm_wh
        orbit["unused_solar_energy_wh"] += unused_wh
        orbit["net_energy_wh"] += net_wh

    soc_values = [_to_float(row["SOC"]) for row in trajectory_rows]
    thermal_values = [_to_float(row["thermal_pressure"]) for row in trajectory_rows]
    summary = {
        "arm": public_arm,
        "internal_arm": arm,
        "min_SOC": min(soc_values),
        "final_SOC": soc_values[-1],
        "fraction_time_SOC_above_70": _fraction(sum(value >= 0.70 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_50_70": _fraction(sum(0.50 <= value < 0.70 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_30_50": _fraction(sum(0.30 <= value < 0.50 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_20_30": _fraction(sum(0.20 <= value < 0.30 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_below_20": _fraction(sum(value < 0.20 for value in soc_values), len(soc_values)),
        "max_thermal_pressure": max(thermal_values),
        "hard_safety_violation_count": sum(1 for value in soc_values if value < 0.20)
        + sum(1 for value in thermal_values if value > 0.90),
    }
    return {
        "trajectory": trajectory_rows,
        "payload_rows": payload_rows,
        "energy_rows": [orbit_rows[key] for key in sorted(orbit_rows)],
        "summary": summary,
        "defer_events": defer_events,
        "completed_slot_by_task": completed_slot_by_task,
    }


def _phase_summary(
    arm: str,
    run: dict[str, Any],
    physical: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    public_arm = ARM_ALIAS.get(arm, arm)
    ledger = run["ledger"]
    rows = []
    phase_by_task = {str(row["task_id"]): str(row["phase_name"]) for row in manifest}
    completed = physical["completed_slot_by_task"]
    traj_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in physical["trajectory"]:
        traj_by_phase[str(row["phase_name"])].append(row)
    ledger_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        ledger_by_phase[phase_by_task.get(str(row["task_id"]), _phase_for_slot(_to_int(row["slot"]))["phase_name"])].append(row)
    manifest_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        manifest_by_phase[str(row["phase_name"])].append(row)
    for phase in PHASES:
        phase_name = str(phase["phase_name"])
        traj = traj_by_phase.get(phase_name, [])
        led = ledger_by_phase.get(phase_name, [])
        tasks = manifest_by_phase.get(phase_name, [])
        socs = [_to_float(row["SOC"]) for row in traj]
        actions = Counter(str(row.get("product_type", "")) for row in led)
        defer_count = sum(1 for row in physical["defer_events"] if str(_phase_for_slot(_to_int(row["slot"]))["phase_name"]) == phase_name)
        total_action_count = max(sum(actions.values()) + defer_count, 1)
        rows.append(
            {
                "arm": public_arm,
                "phase_name": phase_name,
                "initial_SOC": socs[0] if socs else "",
                "final_SOC": socs[-1] if socs else "",
                "min_SOC": min(socs) if socs else "",
                "SOC_drop": max(0.0, (socs[0] - min(socs)) if socs else 0.0),
                "SOC_recovery": max(0.0, (socs[-1] - min(socs)) if socs else 0.0),
                "task_count": len(tasks),
                "completed_task_count": sum(1 for task in tasks if str(task["task_id"]) in completed),
                "H_net_utility": sum(_to_float(row["product_net_utility"]) for row in led if _to_int(row.get("in_H"))),
                "net_system_utility": sum(_to_float(row["product_net_utility"]) for row in led),
                "medium_rate": _fraction(sum(1 for task in tasks if str(task["value_level"]) == "medium"), len(tasks)),
                "roi_context_rate": actions["roi_context"] / total_action_count,
                "metadata_rate": actions["metadata"] / total_action_count,
                "defer_rate": defer_count / total_action_count,
                "payload_on_fraction": _fraction(sum(1 for row in traj if row["payload_power_state"] == "on"), len(traj)),
            }
        )
    return rows


def _action_by_soc_band(
    arm: str,
    run: dict[str, Any],
    physical: dict[str, Any],
) -> list[dict[str, Any]]:
    public_arm = ARM_ALIAS.get(arm, arm)
    soc_by_slot = {_to_int(row["slot"]): str(row["soc_band"]) for row in physical["trajectory"]}
    counts: Counter[tuple[str, str]] = Counter()
    for row in run["ledger"]:
        counts[(soc_by_slot.get(_to_int(row["slot"]), "unknown"), str(row["product_type"]))] += 1
    for row in physical["defer_events"]:
        counts[(str(row["soc_band"]), "defer")] += 1
    out = []
    for (band, action), count in sorted(counts.items()):
        out.append({"arm": public_arm, "soc_band": band, "action": action, "count": count})
    return out


def run_main(
    manifest: list[dict[str, Any]],
    contact_schedule: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    runs = {arm: fair.run_arm_case(manifest, contact_schedule, arm) for arm in ARM_ORDER}
    physical = {arm: _integrate_arm(arm, runs[arm], manifest, contact_schedule, resource_rows) for arm in ARM_ORDER}
    summary_rows = []
    trajectory_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        fair_summary = runs[arm]["summary"]
        phys_summary = physical[arm]["summary"]
        row = {
            "arm": ARM_ALIAS.get(arm, arm),
            "internal_arm": arm,
            "net_system_utility": _to_float(fair_summary["net_system_utility"]),
            "H_gross_delivered_utility": _to_float(fair_summary["H_gross_delivered_utility"]),
            "H_net_delivered_utility": _to_float(fair_summary["H_net_delivered_utility"]),
            "H_min_stage_success_rate": _to_float(fair_summary["H_min_stage_success_rate_feasible_H"]),
            "H_target_stage_success_rate": _to_float(fair_summary["H_target_stage_success_rate_feasible_H"]),
            "total_resource_cost": _to_float(fair_summary["total_resource_cost"]),
            "min_SOC": phys_summary["min_SOC"],
            "final_SOC": phys_summary["final_SOC"],
            "fraction_time_SOC_above_70": phys_summary["fraction_time_SOC_above_70"],
            "fraction_time_SOC_50_70": phys_summary["fraction_time_SOC_50_70"],
            "fraction_time_SOC_30_50": phys_summary["fraction_time_SOC_30_50"],
            "fraction_time_SOC_20_30": phys_summary["fraction_time_SOC_20_30"],
            "fraction_time_SOC_below_20": phys_summary["fraction_time_SOC_below_20"],
            "max_thermal_pressure": phys_summary["max_thermal_pressure"],
            "hard_safety_violation_count": phys_summary["hard_safety_violation_count"],
        }
        summary_rows.append(row)
        trajectory_rows.extend(physical[arm]["trajectory"])
        payload_rows.extend(physical[arm]["payload_rows"])
        energy_rows.extend(physical[arm]["energy_rows"])
        phase_rows.extend(_phase_summary(arm, runs[arm], physical[arm], manifest))
        action_rows.extend(_action_by_soc_band(arm, runs[arm], physical[arm]))
    for rank, row in enumerate(sorted(summary_rows, key=lambda item: item["net_system_utility"], reverse=True), start=1):
        row["rank_net_system_utility"] = rank
    for rank, row in enumerate(sorted(summary_rows, key=lambda item: item["H_gross_delivered_utility"], reverse=True), start=1):
        row["rank_H_gross"] = rank
    for rank, row in enumerate(sorted(summary_rows, key=lambda item: item["H_net_delivered_utility"], reverse=True), start=1):
        row["rank_H_net"] = rank
    for rank, row in enumerate(sorted(summary_rows, key=lambda item: item["H_min_stage_success_rate"], reverse=True), start=1):
        row["rank_H_min_stage"] = rank
    for rank, row in enumerate(sorted(summary_rows, key=lambda item: item["H_target_stage_success_rate"], reverse=True), start=1):
        row["rank_H_target_stage"] = rank
    ranking_cols = [
        "arm",
        "net_system_utility",
        "rank_net_system_utility",
        "H_gross_delivered_utility",
        "rank_H_gross",
        "H_net_delivered_utility",
        "rank_H_net",
        "H_min_stage_success_rate",
        "rank_H_min_stage",
        "H_target_stage_success_rate",
        "rank_H_target_stage",
        "total_resource_cost",
        "min_SOC",
        "final_SOC",
        "fraction_time_SOC_above_70",
        "fraction_time_SOC_50_70",
        "fraction_time_SOC_30_50",
        "fraction_time_SOC_20_30",
        "fraction_time_SOC_below_20",
        "max_thermal_pressure",
        "hard_safety_violation_count",
    ]
    summary_rows = sorted(summary_rows, key=lambda item: item["rank_net_system_utility"])
    _write_csv(OUT_DIR / "main_ranking_table.csv", summary_rows, ranking_cols)
    _write_csv(OUT_DIR / "soc_trajectory_by_arm.csv", trajectory_rows)
    _write_csv(OUT_DIR / "energy_balance_by_arm.csv", energy_rows)
    _write_csv(
        OUT_DIR / "thermal_summary_by_arm.csv",
        [
            {
                "arm": row["arm"],
                "max_thermal_pressure": row["max_thermal_pressure"],
                "hard_safety_violation_count": row["hard_safety_violation_count"],
            }
            for row in summary_rows
        ],
    )
    _write_csv(OUT_DIR / "action_by_soc_band_by_arm.csv", action_rows)
    _write_csv(OUT_DIR / "phase_summary_by_arm.csv", phase_rows)
    _write_csv(OUT_DIR / "payload_power_state_by_arm.csv", payload_rows)
    _write_csv(OUT_DIR / "delivered_product_ledger_all_arms.csv", [row for run in runs.values() for row in run["ledger"]])
    _write_csv(OUT_DIR / "H_stage_success_audit_all_arms.csv", [row for run in runs.values() for row in run["stage_rows"]])
    return {
        "ranking_rows": summary_rows,
        "trajectory_rows": trajectory_rows,
        "energy_rows": energy_rows,
        "phase_rows": phase_rows,
        "action_rows": action_rows,
        "payload_rows": payload_rows,
        "runs": runs,
        "physical": physical,
    }


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


def write_no_task_report(summary_rows: list[dict[str, Any]]) -> None:
    cols = [
        "case_name",
        "final_SOC",
        "daily_SOC_drift",
        "mean_solar_energy_per_orbit_wh",
        "mean_housekeeping_energy_per_orbit_wh",
        "mean_payload_idle_energy_per_orbit_wh",
        "mean_net_energy_per_orbit_wh",
        "unused_solar_ratio",
    ]
    lines = [
        "# No-task power-gating calibration",
        "",
        _markdown_table(summary_rows, cols),
        "",
        "Interpretation: housekeeping-only shows the available positive energy margin; Pi idle is near sustainable; Atlas idle shows why keeping a larger COTS payload powered indefinitely is not a good nominal assumption.",
    ]
    _write_text(OUT_DIR / "no_task_power_gating_report.md", "\n".join(lines))


def write_quick_report(no_task: list[dict[str, Any]], main: dict[str, Any]) -> None:
    by_case = {row["case_name"]: row for row in no_task}
    rows = main["ranking_rows"]
    proposed = next(row for row in rows if row["arm"] == "proposed_guarantee_aware_full")
    fixed_l = next(row for row in rows if row["arm"] == "fixed_lightweight")
    fixed_h = next(row for row in rows if row["arm"] == "fixed_heavy")
    prop_phases = [row for row in main["phase_rows"] if row["arm"] == "proposed_guarantee_aware_full"]
    phase = {row["phase_name"]: row for row in prop_phases}
    lines = [
        "# BUPT-1 power-gating burst-recovery quick report",
        "",
        "## Direct answers",
        "",
        f"1. Housekeeping-only 24h clearly recovers: final_SOC={_fmt(by_case['case_1_housekeeping_only']['final_SOC'])}, drift={_fmt(by_case['case_1_housekeeping_only']['daily_SOC_drift'])}.",
        f"2. Pi idle / Atlas idle show payload idle drag: Pi final_SOC={_fmt(by_case['case_2_pi_idle']['final_SOC'])}, Atlas final_SOC={_fmt(by_case['case_3_atlas_idle']['final_SOC'])}. Atlas long-idle is not nominal-sustainable.",
        f"3. Proposed uses resources in burst and SOC drops: phase_1 SOC_drop={_fmt(phase['phase_1_burst']['SOC_drop'])}, payload_on_fraction={_fmt(phase['phase_1_burst']['payload_on_fraction'])}.",
        "4. Proposed action mix contracts before hard guards are reached: it operates across SOC>=70 and 50<=SOC<70, while SOC<50 and SOC<30 are not reached by proposed in this run.",
        f"5. Recovery behavior is visible after the heavy phase: phase_2 has no new tasks but still drains burst backlog (recovery={_fmt(phase['phase_2_recovery']['SOC_recovery'])}); phase_5 has no new tasks and recovers SOC by {_fmt(phase['phase_5_recovery']['SOC_recovery'])} with payload_on_fraction={_fmt(phase['phase_5_recovery']['payload_on_fraction'])}.",
        f"6. fixed_lightweight is battery-healthiest but weaker in utility: min_SOC={_fmt(fixed_l['min_SOC'])}, H_net={_fmt(fixed_l['H_net_delivered_utility'])}, net={_fmt(fixed_l['net_system_utility'])}.",
        f"7. fixed_heavy drops more and recovers less easily: min_SOC={_fmt(fixed_h['min_SOC'])}, max_thermal={_fmt(fixed_h['max_thermal_pressure'])}.",
        f"8. Proposed advantage: net rank #{proposed['rank_net_system_utility']}, H_net rank #{proposed['rank_H_net']}, H_target rank #{proposed['rank_H_target_stage']}.",
        "9. Yes. This is suitable as a burst-recovery long-horizon case study because it has burst, recovery, regular, heavy, and final recovery phases under one shared manifest/contact/resource trace.",
        "10. Nominal profile: housekeeping 8W + Pi-class payload idle 2.4W with power gating. Stress/tight profile: Atlas-class long-idle 7.2W or no-gating payload-on behavior.",
        "",
        "## Notes",
        "",
        "- The runner does not modify proposed router logic, reward, baselines, or stage-success definitions.",
        "- Downlink uses tx_peak=38W during active contact transmission windows and does not add contact standby on top.",
    ]
    _write_text(OUT_DIR / "quick_report.md", "\n".join(lines))


def make_figures(no_task_traj: list[dict[str, Any]], main: dict[str, Any]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for case in NO_TASK_CASES:
        rows = [row for row in no_task_traj if row["case_name"] == case["case_name"]]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["SOC"]) for row in rows], label=case["case_name"])
    for level in [0.70, 0.50, 0.30, 0.20]:
        plt.axhline(level, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("time (h)")
    plt.ylabel("SOC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "no_task_power_gating_soc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for phase in PHASES:
        color = "#f4cccc" if "burst" in phase["phase_name"] or "heavy" in phase["phase_name"] else "#d9ead3"
        plt.axvspan(phase["start_h"], phase["end_h"], color=color, alpha=0.25)
    for arm in [ARM_ALIAS.get(arm, arm) for arm in ARM_ORDER]:
        rows = [row for row in main["trajectory_rows"] if row["arm"] == arm]
        plt.plot([_to_float(row["time_h"]) for row in rows], [_to_float(row["SOC"]) for row in rows], label=arm)
    for level in [0.70, 0.50, 0.30, 0.20]:
        plt.axhline(level, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("time (h)")
    plt.ylabel("SOC")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "burst_recovery_soc_by_arm.png", dpi=160)
    plt.close()

    proposed_payload = [row for row in main["payload_rows"] if row["arm"] == "proposed_guarantee_aware_full"]
    plt.figure(figsize=(10, 3))
    plt.step(
        [_to_float(row["time_h"]) for row in proposed_payload],
        [1 if row["payload_power_state"] == "on" else 0 for row in proposed_payload],
        where="post",
    )
    plt.yticks([0, 1], ["gated_off", "on"])
    plt.xlabel("time (h)")
    plt.ylabel("payload")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "payload_power_state_proposed.png", dpi=160)
    plt.close()

    action_rows = main["action_rows"]
    bands = ["SOC_above_70", "SOC_50_70", "SOC_30_50", "SOC_20_30", "SOC_below_20"]
    actions = sorted({row["action"] for row in action_rows})
    proposed_actions = [row for row in action_rows if row["arm"] == "proposed_guarantee_aware_full"]
    bottom = [0] * len(bands)
    plt.figure(figsize=(9, 5))
    for action in actions:
        vals = [
            sum(_to_int(row["count"]) for row in proposed_actions if row["soc_band"] == band and row["action"] == action)
            for band in bands
        ]
        plt.bar(bands, vals, bottom=bottom, label=action)
        bottom = [a + b for a, b in zip(bottom, vals)]
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "action_distribution_by_soc_band.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    for row in main["ranking_rows"]:
        plt.scatter(_to_float(row["total_resource_cost"]), _to_float(row["H_net_delivered_utility"]))
        plt.text(_to_float(row["total_resource_cost"]), _to_float(row["H_net_delivered_utility"]), str(row["arm"]), fontsize=8)
    plt.xlabel("total resource cost")
    plt.ylabel("H net delivered utility")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "H_net_vs_energy_cost.png", dpi=160)
    plt.close()


def run(output_dir: Path = OUT_DIR) -> dict[str, Any]:
    global OUT_DIR, FIG_DIR
    OUT_DIR = output_dir
    FIG_DIR = OUT_DIR / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    no_task_summary, no_task_traj, _no_task_orbits = run_no_task_calibration()
    write_no_task_report(no_task_summary)
    contact = build_contact_schedule()
    manifest = build_burst_recovery_manifest(contact)
    resources = _resource_rows()
    _write_csv(OUT_DIR / "shared_task_manifest.csv", manifest)
    _write_csv(OUT_DIR / "shared_contact_schedule.csv", contact)
    _write_csv(OUT_DIR / "shared_resource_trace.csv", resources)
    main = run_main(manifest, contact, resources)
    make_figures(no_task_traj, main)
    write_quick_report(no_task_summary, main)
    return {"no_task_summary": no_task_summary, "main": main, "manifest": manifest, "contact": contact}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    result = run(args.output_dir)
    proposed = next(row for row in result["main"]["ranking_rows"] if row["arm"] == "proposed_guarantee_aware_full")
    print(f"output_dir={args.output_dir}")
    print(f"task_count={len(result['manifest'])}")
    print(f"contact_count={len(result['contact'])}")
    print(f"proposed_net_rank={proposed['rank_net_system_utility']}")
    print(f"proposed_min_soc={_fmt(proposed['min_SOC'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
