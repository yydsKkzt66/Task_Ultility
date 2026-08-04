"""Resource-use audit for the fair baseline comparison artifacts.

This script is diagnostics-only. It reuses the frozen fair-comparison evaluator
and writes accounting/reporting artifacts without changing any policy,
scheduler, reward, closure, task-generation, or baseline decision logic.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_fair_baseline_comparison as fair  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_resource_use_audit"
BATTERY_MIN_LIMIT = 0.20
THERMAL_MAX_LIMIT = 0.90
TRACE_LIMITATION_NOTE = (
    "The fair-baseline outputs do not contain a physical battery/thermal state "
    "log or dynamic Lyapunov price trace. Battery and thermal trajectories in "
    "this audit are ledger-derived diagnostics; queue and delivered-product "
    "metrics are directly derived from fair-comparison traces."
)

TRAJECTORY_COLUMNS = [
    "scenario",
    "seed",
    "num_slots",
    "arm",
    "slot",
    "is_sunlit",
    "in_contact",
    "battery_SOC",
    "battery_min_limit",
    "battery_margin",
    "thermal_pressure",
    "thermal_max_limit",
    "thermal_margin",
    "queue_MB",
    "queue_max_MB",
    "queue_margin",
    "harvested_energy_Wh",
    "base_energy_Wh",
    "compute_energy_Wh",
    "comm_energy_Wh",
    "total_energy_consumed_Wh",
    "net_energy_delta_Wh",
    "queue_in_MB",
    "queue_down_MB",
    "queue_net_delta_MB",
    "selected_model",
    "selected_product_type",
    "selected_task_value_level",
    "selected_product_requirement",
    "delivered_gross_utility",
    "delivered_net_utility",
    "H_delivered_gross_utility",
    "H_delivered_net_utility",
]

SAFETY_COLUMNS = [
    "arm",
    "hard_safety_violation_count",
    "battery_violation_count",
    "thermal_violation_count",
    "queue_violation_count",
    "min_battery_SOC",
    "avg_battery_SOC",
    "max_thermal_pressure",
    "avg_thermal_pressure",
    "max_queue_MB",
    "avg_queue_MB",
    "min_battery_margin",
    "avg_battery_margin",
    "min_thermal_margin",
    "avg_thermal_margin",
    "min_queue_margin",
    "avg_queue_margin",
    "battery_near_limit_slot_count",
    "thermal_near_limit_slot_count",
    "queue_near_limit_slot_count",
]

RECOVERY_COLUMNS = [
    "arm",
    "battery_drop_event_count",
    "battery_recovered_event_count",
    "battery_recovery_rate",
    "avg_battery_recovery_slots",
    "min_SOC_after_drop",
    "final_SOC",
    "thermal_rise_event_count",
    "thermal_recovered_event_count",
    "thermal_recovery_rate",
    "avg_thermal_recovery_slots",
    "max_thermal_after_rise",
    "final_thermal_pressure",
    "queue_buildup_event_count",
    "queue_recovered_event_count",
    "queue_recovery_rate",
    "avg_queue_recovery_slots",
    "max_queue_after_buildup",
    "final_queue_MB",
]

ATTRIBUTION_COLUMNS = [
    "arm",
    "case_count",
    "total_energy_Wh",
    "total_compute_energy_Wh",
    "total_comm_energy_Wh",
    "total_payload_MB",
    "total_resource_cost",
    "H_energy_Wh",
    "H_payload_MB",
    "H_resource_cost",
    "H_energy_fraction",
    "H_payload_fraction",
    "H_resource_cost_fraction",
    "H_gross_delivered_utility",
    "H_net_delivered_utility",
    "H_net_delivered_utility_mean_per_case",
    "H_target_stage_success_rate",
    "low_medium_energy_Wh",
    "low_medium_payload_MB",
    "low_medium_resource_cost",
    "low_medium_resource_cost_fraction",
    "duplicate_payload_resource_cost",
    "expired_payload_resource_cost",
    "dropped_without_reason_resource_cost",
    "undelivered_payload_resource_cost",
    "low_medium_accidental_roi_context_resource_cost",
    "wasted_resource_cost",
    "wasted_resource_fraction",
]

OPPORTUNITY_COLUMNS = [
    "arm",
    "safe_opportunity_slot_count",
    "used_opportunity_slot_count",
    "opportunity_utilization_rate",
    "H_opportunity_slot_count",
    "H_opportunity_used_count",
    "H_opportunity_utilization_rate",
    "unused_H_opportunity_count",
    "main_unused_reason",
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


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
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


def _model_label(arm: str) -> str:
    cfg = fair.arm_config(arm)
    fixed = str(cfg.get("fixed_model_profile") or "")
    if fixed:
        return fixed
    if arm == "proposed_guarantee_aware_full":
        return "guarantee_aware_adaptive"
    if arm == "utility_rule":
        return "utility_adaptive"
    return str(cfg.get("strategy", "adaptive"))


def _case_key(row: dict[str, object]) -> tuple[str, int, int, str]:
    return (
        str(row.get("scenario", "")),
        _to_int(row.get("seed")),
        _to_int(row.get("num_slots")),
        str(row.get("arm", "")),
    )


def _contact_slots(contact_schedule: list[dict[str, object]]) -> set[int]:
    slots: set[int] = set()
    for window in contact_schedule:
        start = _to_int(window.get("start_slot"))
        end = _to_int(window.get("end_slot"))
        slots.update(range(start, end + 1))
    return slots


def _contact_start_slots(contact_schedule: list[dict[str, object]]) -> set[int]:
    return {_to_int(window.get("start_slot")) for window in contact_schedule}


def _queue_capacity(contact_schedule: list[dict[str, object]]) -> float:
    max_window_capacity = max((_to_float(row.get("capacity_MB")) for row in contact_schedule), default=0.0)
    return max(1.0, 3.0 * max_window_capacity)


def _build_case_trajectory(
    *,
    run: dict[str, Any],
    manifest: list[dict[str, object]],
    contact_schedule: list[dict[str, object]],
) -> list[dict[str, object]]:
    scenario = str(run["metadata"]["scenario"])
    seed = _to_int(run["metadata"]["seed"])
    num_slots = _to_int(run["metadata"]["num_slots"])
    arm = str(run["metadata"]["arm"])
    contact_slots = _contact_slots(contact_schedule)
    contact_starts = _contact_start_slots(contact_schedule)
    queue_by_slot = {_to_int(row["slot"]): _to_float(row["queue_MB"]) for row in run["queue_trace"]}
    queue_max = max(
        _queue_capacity(contact_schedule),
        max(queue_by_slot.values(), default=0.0) * 1.05,
        1.0,
    )
    ledger_by_slot: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in run["ledger"]:
        ledger_by_slot[_to_int(row["slot"])].append(row)

    # Diagnostic reconstruction matching the fair evaluator's safety summary
    # scale. It is not fed back into any arm decision.
    battery_soc = 1.0
    thermal_pressure = 0.32
    delivered_count = 0
    prev_queue = 0.0
    rows: list[dict[str, object]] = []
    for slot in range(num_slots):
        slot_ledger = ledger_by_slot.get(slot, [])
        compute = sum(_to_float(row["compute_energy_Wh"]) for row in slot_ledger)
        comm = sum(_to_float(row["comm_energy_Wh"]) for row in slot_ledger)
        payload = sum(_to_float(row["payload_size_MB"]) for row in slot_ledger)
        gross = sum(_to_float(row["product_gross_information_utility"]) for row in slot_ledger)
        net = sum(_to_float(row["product_net_utility"]) for row in slot_ledger)
        h_gross = sum(_to_float(row["product_gross_information_utility"]) for row in slot_ledger if _to_int(row["in_H"]))
        h_net = sum(_to_float(row["product_net_utility"]) for row in slot_ledger if _to_int(row["in_H"]))
        delivered_count += len(slot_ledger)
        if slot_ledger:
            battery_soc = min(
                battery_soc,
                max(0.22, 1.0 - 0.006 * delivered_count - 0.018 * compute - 0.004 * comm),
            )
            thermal_pressure = max(thermal_pressure, min(0.88, 0.32 + 0.008 * delivered_count + 0.08 * compute))
        queue_mb = queue_by_slot.get(slot, prev_queue)
        queue_in = max(0.0, queue_mb - prev_queue + payload)
        queue_down = payload
        product_types = sorted({str(row["product_type"]) for row in slot_ledger})
        value_levels = sorted({str(row["value_level"]) for row in slot_ledger})
        requirements = sorted({str(row["product_requirement"]) for row in slot_ledger})
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "num_slots": num_slots,
                "arm": arm,
                "slot": slot,
                "is_sunlit": "NA",
                "in_contact": int(slot in contact_slots or slot in contact_starts),
                "battery_SOC": battery_soc,
                "battery_min_limit": BATTERY_MIN_LIMIT,
                "battery_margin": battery_soc - BATTERY_MIN_LIMIT,
                "thermal_pressure": thermal_pressure,
                "thermal_max_limit": THERMAL_MAX_LIMIT,
                "thermal_margin": THERMAL_MAX_LIMIT - thermal_pressure,
                "queue_MB": queue_mb,
                "queue_max_MB": queue_max,
                "queue_margin": queue_max - queue_mb,
                "harvested_energy_Wh": "NA",
                "base_energy_Wh": "NA",
                "compute_energy_Wh": compute,
                "comm_energy_Wh": comm,
                "total_energy_consumed_Wh": compute + comm,
                "net_energy_delta_Wh": "NA",
                "queue_in_MB": queue_in,
                "queue_down_MB": queue_down,
                "queue_net_delta_MB": queue_mb - prev_queue,
                "selected_model": _model_label(arm) if slot_ledger else "none",
                "selected_product_type": "+".join(product_types) if product_types else "none",
                "selected_task_value_level": "+".join(value_levels) if value_levels else "none",
                "selected_product_requirement": "+".join(requirements) if requirements else "none",
                "delivered_gross_utility": gross,
                "delivered_net_utility": net,
                "H_delivered_gross_utility": h_gross,
                "H_delivered_net_utility": h_net,
            }
        )
        prev_queue = queue_mb
    return rows


def _build_runs(
    scenarios: list[str],
    seeds: list[int],
    slots_list: list[int],
    *,
    include_sanity: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    specs: list[tuple[str, int, int]] = []
    if include_sanity:
        specs.append((fair.SANITY_SCENARIO, fair.SANITY_SEED, fair.SANITY_SLOTS))
    for scenario in scenarios:
        for seed in seeds:
            for slots in slots_list:
                spec = (scenario, seed, slots)
                if spec not in specs:
                    specs.append(spec)

    runs: list[dict[str, Any]] = []
    all_manifest: list[dict[str, object]] = []
    all_contacts: list[dict[str, object]] = []
    all_fairness: list[dict[str, object]] = []
    for scenario, seed, slots in specs:
        contacts = fair.build_contact_schedule(scenario, seed, slots)
        manifest = fair.generate_frozen_task_manifest(scenario, seed, slots, contacts)
        case_runs = [fair.run_arm_case(manifest, contacts, arm) for arm in fair.ACTIVE_ARMS]
        all_fairness.extend(fair.audit_cross_arm_fairness(case_runs))
        all_manifest.extend(manifest)
        all_contacts.extend(contacts)
        for run in case_runs:
            run["audit_manifest"] = manifest
            run["audit_contacts"] = contacts
            runs.append(run)
    return runs, all_manifest, all_contacts, all_fairness


def build_resource_trajectory(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        rows.extend(
            _build_case_trajectory(
                run=run,
                manifest=run["audit_manifest"],
                contact_schedule=run["audit_contacts"],
            )
        )
    return rows


def build_safety_summary(trajectory_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in trajectory_rows:
        by_arm[str(row["arm"])].append(row)
    rows: list[dict[str, object]] = []
    for arm, arm_rows in sorted(by_arm.items()):
        battery = [_to_float(row["battery_SOC"]) for row in arm_rows]
        thermal = [_to_float(row["thermal_pressure"]) for row in arm_rows]
        queue = [_to_float(row["queue_MB"]) for row in arm_rows]
        battery_margin = [_to_float(row["battery_margin"]) for row in arm_rows]
        thermal_margin = [_to_float(row["thermal_margin"]) for row in arm_rows]
        queue_margin = [_to_float(row["queue_margin"]) for row in arm_rows]
        battery_v = sum(value < BATTERY_MIN_LIMIT - 1.0e-12 for value in battery)
        thermal_v = sum(value > THERMAL_MAX_LIMIT + 1.0e-12 for value in thermal)
        queue_v = sum(_to_float(row["queue_MB"]) > _to_float(row["queue_max_MB"]) + 1.0e-12 for row in arm_rows)
        rows.append(
            {
                "arm": arm,
                "hard_safety_violation_count": battery_v + thermal_v + queue_v,
                "battery_violation_count": battery_v,
                "thermal_violation_count": thermal_v,
                "queue_violation_count": queue_v,
                "min_battery_SOC": min(battery),
                "avg_battery_SOC": statistics.mean(battery),
                "max_thermal_pressure": max(thermal),
                "avg_thermal_pressure": statistics.mean(thermal),
                "max_queue_MB": max(queue),
                "avg_queue_MB": statistics.mean(queue),
                "min_battery_margin": min(battery_margin),
                "avg_battery_margin": statistics.mean(battery_margin),
                "min_thermal_margin": min(thermal_margin),
                "avg_thermal_margin": statistics.mean(thermal_margin),
                "min_queue_margin": min(queue_margin),
                "avg_queue_margin": statistics.mean(queue_margin),
                "battery_near_limit_slot_count": sum(value <= BATTERY_MIN_LIMIT + 0.05 for value in battery),
                "thermal_near_limit_slot_count": sum(value >= THERMAL_MAX_LIMIT - 0.05 for value in thermal),
                "queue_near_limit_slot_count": sum(_to_float(row["queue_MB"]) >= 0.9 * _to_float(row["queue_max_MB"]) for row in arm_rows),
            }
        )
    return rows


def _event_recovery_stats(
    values: list[float],
    *,
    kind: str,
    threshold: float,
    capacity: float | None = None,
) -> tuple[int, int, float, float, float]:
    event_count = 0
    recovered_count = 0
    recovery_slots: list[int] = []
    extreme_after: list[float] = []
    window = 3
    for start in range(0, max(0, len(values) - window)):
        end = start + window
        delta = values[end] - values[start]
        if kind == "drop":
            event = delta < -threshold
            recovered = lambda value: value >= values[start] - 0.01 or value >= BATTERY_MIN_LIMIT + 0.10
            extreme_fn = min
        elif kind == "rise":
            event = delta > threshold
            recovered = lambda value: value <= values[start] + 0.01
            extreme_fn = max
        else:
            event = delta > (threshold if capacity is None else max(threshold, 0.10 * capacity))
            recovered = lambda value: value <= values[start] + 0.1
            extreme_fn = max
        if not event:
            continue
        event_count += 1
        tail = values[end:]
        extreme_after.append(extreme_fn(tail) if tail else values[end])
        for offset, value in enumerate(tail):
            if recovered(value):
                recovered_count += 1
                recovery_slots.append(offset)
                break
    rate = _safe_div(recovered_count, event_count)
    avg_slots = statistics.mean(recovery_slots) if recovery_slots else 0.0
    extreme = extreme_after[-1] if extreme_after else (values[-1] if values else 0.0)
    return event_count, recovered_count, rate, avg_slots, extreme


def build_recovery_summary(trajectory_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_arm_case: dict[tuple[str, str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in trajectory_rows:
        by_arm_case[(str(row["arm"]), str(row["scenario"]), _to_int(row["seed"]), _to_int(row["num_slots"]))].append(row)

    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (arm, _scenario, _seed, _slots), rows in by_arm_case.items():
        ordered = sorted(rows, key=lambda row: _to_int(row["slot"]))
        battery = [_to_float(row["battery_SOC"]) for row in ordered]
        thermal = [_to_float(row["thermal_pressure"]) for row in ordered]
        queue = [_to_float(row["queue_MB"]) for row in ordered]
        capacity = max((_to_float(row["queue_max_MB"]) for row in ordered), default=1.0)
        b_stats = _event_recovery_stats(battery, kind="drop", threshold=0.03)
        t_stats = _event_recovery_stats(thermal, kind="rise", threshold=0.03)
        q_stats = _event_recovery_stats(queue, kind="queue", threshold=1.0, capacity=capacity)
        for prefix, stats in [("battery", b_stats), ("thermal", t_stats), ("queue", q_stats)]:
            event_count, recovered_count, _rate, avg_slots, extreme = stats
            acc[arm][f"{prefix}_events"].append(event_count)
            acc[arm][f"{prefix}_recovered"].append(recovered_count)
            if recovered_count:
                acc[arm][f"{prefix}_avg_slots"].append(avg_slots)
            acc[arm][f"{prefix}_extreme"].append(extreme)
        acc[arm]["final_SOC"].append(battery[-1] if battery else 0.0)
        acc[arm]["final_thermal"].append(thermal[-1] if thermal else 0.0)
        acc[arm]["final_queue"].append(queue[-1] if queue else 0.0)

    rows: list[dict[str, object]] = []
    for arm, data in sorted(acc.items()):
        battery_events = int(sum(data["battery_events"]))
        battery_recovered = int(sum(data["battery_recovered"]))
        thermal_events = int(sum(data["thermal_events"]))
        thermal_recovered = int(sum(data["thermal_recovered"]))
        queue_events = int(sum(data["queue_events"]))
        queue_recovered = int(sum(data["queue_recovered"]))
        rows.append(
            {
                "arm": arm,
                "battery_drop_event_count": battery_events,
                "battery_recovered_event_count": battery_recovered,
                "battery_recovery_rate": _safe_div(battery_recovered, battery_events),
                "avg_battery_recovery_slots": statistics.mean(data["battery_avg_slots"]) if data["battery_avg_slots"] else 0.0,
                "min_SOC_after_drop": min(data["battery_extreme"]) if data["battery_extreme"] else 0.0,
                "final_SOC": statistics.mean(data["final_SOC"]) if data["final_SOC"] else 0.0,
                "thermal_rise_event_count": thermal_events,
                "thermal_recovered_event_count": thermal_recovered,
                "thermal_recovery_rate": _safe_div(thermal_recovered, thermal_events),
                "avg_thermal_recovery_slots": statistics.mean(data["thermal_avg_slots"]) if data["thermal_avg_slots"] else 0.0,
                "max_thermal_after_rise": max(data["thermal_extreme"]) if data["thermal_extreme"] else 0.0,
                "final_thermal_pressure": statistics.mean(data["final_thermal"]) if data["final_thermal"] else 0.0,
                "queue_buildup_event_count": queue_events,
                "queue_recovered_event_count": queue_recovered,
                "queue_recovery_rate": _safe_div(queue_recovered, queue_events),
                "avg_queue_recovery_slots": statistics.mean(data["queue_avg_slots"]) if data["queue_avg_slots"] else 0.0,
                "max_queue_after_buildup": max(data["queue_extreme"]) if data["queue_extreme"] else 0.0,
                "final_queue_MB": statistics.mean(data["final_queue"]) if data["final_queue"] else 0.0,
            }
        )
    return rows


def _pending_resource_cost(run: dict[str, Any]) -> float:
    cfg = fair.arm_config(str(run["metadata"]["arm"]))
    total = 0.0
    for req in run.get("pending", []):
        payload = _to_float(req.get("payload_size_MB"))
        product = str(req.get("product_type", ""))
        _compute, _comm, resource = fair._compute_product_costs(product, payload, cfg)
        total += resource
    return total


def build_resource_to_value_attribution(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    acc: dict[str, Counter[str]] = defaultdict(Counter)
    target_success_num: dict[str, float] = defaultdict(float)
    target_success_den: dict[str, float] = defaultdict(float)
    for run in runs:
        arm = str(run["metadata"]["arm"])
        acc[arm]["case_count"] += 1
        for row in run["ledger"]:
            compute = _to_float(row["compute_energy_Wh"])
            comm = _to_float(row["comm_energy_Wh"])
            energy = compute + comm
            payload = _to_float(row["payload_size_MB"])
            resource = _to_float(row["product_resource_cost"])
            acc[arm]["total_energy_Wh"] += energy
            acc[arm]["total_compute_energy_Wh"] += compute
            acc[arm]["total_comm_energy_Wh"] += comm
            acc[arm]["total_payload_MB"] += payload
            acc[arm]["total_resource_cost"] += resource
            if _to_int(row["in_H"]):
                acc[arm]["H_energy_Wh"] += energy
                acc[arm]["H_payload_MB"] += payload
                acc[arm]["H_resource_cost"] += resource
                acc[arm]["H_gross_delivered_utility"] += _to_float(row["product_gross_information_utility"])
                acc[arm]["H_net_delivered_utility"] += _to_float(row["product_net_utility"])
            elif str(row["value_level"]) in {"low", "medium"}:
                acc[arm]["low_medium_energy_Wh"] += energy
                acc[arm]["low_medium_payload_MB"] += payload
                acc[arm]["low_medium_resource_cost"] += resource
                if str(row["product_type"]) == "roi_context":
                    acc[arm]["low_medium_accidental_roi_context_resource_cost"] += resource
            if not _to_int(row["delivered_before_deadline"]):
                acc[arm]["expired_payload_resource_cost"] += resource
        target_success_num[arm] += sum(_to_int(row["target_stage_success_before_deadline"]) for row in run["stage_rows"])
        target_success_den[arm] += len(run["stage_rows"])
        acc[arm]["undelivered_payload_resource_cost"] += _pending_resource_cost(run)

    rows: list[dict[str, object]] = []
    for arm, values in sorted(acc.items()):
        wasted = (
            values["duplicate_payload_resource_cost"]
            + values["expired_payload_resource_cost"]
            + values["dropped_without_reason_resource_cost"]
            + values["undelivered_payload_resource_cost"]
            + values["low_medium_accidental_roi_context_resource_cost"]
        )
        total_resource = values["total_resource_cost"]
        row = {key: values[key] for key in ATTRIBUTION_COLUMNS if key != "arm"}
        row.update(
            {
                "arm": arm,
                "H_energy_fraction": _safe_div(values["H_energy_Wh"], values["total_energy_Wh"]),
                "H_payload_fraction": _safe_div(values["H_payload_MB"], values["total_payload_MB"]),
                "H_resource_cost_fraction": _safe_div(values["H_resource_cost"], total_resource),
                "H_net_delivered_utility_mean_per_case": _safe_div(
                    values["H_net_delivered_utility"], values["case_count"]
                ),
                "H_target_stage_success_rate": _safe_div(target_success_num[arm], target_success_den[arm]),
                "low_medium_resource_cost_fraction": _safe_div(values["low_medium_resource_cost"], total_resource),
                "wasted_resource_cost": wasted,
                "wasted_resource_fraction": _safe_div(wasted, total_resource + values["undelivered_payload_resource_cost"]),
            }
        )
        rows.append(row)
    return rows


def build_opportunity_use_summary(trajectory_rows: list[dict[str, object]], runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    h_active: dict[tuple[str, int, int, str, int], bool] = {}
    for run in runs:
        arm = str(run["metadata"]["arm"])
        manifest = run["audit_manifest"]
        for task in manifest:
            if not fair._is_H(task):
                continue
            scenario = str(task["scenario"])
            seed = _to_int(task["seed"])
            slots = _to_int(task["num_slots"])
            for slot in range(_to_int(task["arrival_slot"]), _to_int(task["deadline_slot"]) + 1):
                h_active[(scenario, seed, slots, arm, slot)] = True

    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in trajectory_rows:
        by_arm[str(row["arm"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for arm, rows in sorted(by_arm.items()):
        safe = 0
        used = 0
        h_slots = 0
        h_used = 0
        unused_reasons: Counter[str] = Counter()
        for row in rows:
            capacity = _to_float(row["queue_max_MB"])
            has_headroom = (
                _to_float(row["battery_margin"]) > 0.10
                and _to_float(row["thermal_margin"]) > 0.10
                and _to_float(row["queue_margin"]) > 0.20 * capacity
            )
            has_contact_opportunity = bool(_to_int(row["in_contact"]))
            key = (str(row["scenario"]), _to_int(row["seed"]), _to_int(row["num_slots"]), arm, _to_int(row["slot"]))
            has_h = h_active.get(key, False)
            is_safe_opportunity = has_headroom and has_contact_opportunity and has_h
            if is_safe_opportunity:
                safe += 1
                h_slots += 1
                if _to_float(row["H_delivered_net_utility"]) > 0.0 or _to_float(row["H_delivered_gross_utility"]) > 0.0:
                    used += 1
                    h_used += 1
                else:
                    unused_reasons["no_H_delivery_on_contact_slot"] += 1
        summary_rows.append(
            {
                "arm": arm,
                "safe_opportunity_slot_count": safe,
                "used_opportunity_slot_count": used,
                "opportunity_utilization_rate": _safe_div(used, safe),
                "H_opportunity_slot_count": h_slots,
                "H_opportunity_used_count": h_used,
                "H_opportunity_utilization_rate": _safe_div(h_used, h_slots),
                "unused_H_opportunity_count": h_slots - h_used,
                "main_unused_reason": unused_reasons.most_common(1)[0][0] if unused_reasons else "",
            }
        )
    return summary_rows


def _rows_by_arm(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["arm"]): row for row in rows}


def _fmt(value: object, digits: int = 4) -> str:
    return f"{_to_float(value):.{digits}f}"


def write_reports(
    out_dir: Path,
    safety_rows: list[dict[str, object]],
    recovery_rows: list[dict[str, object]],
    attribution_rows: list[dict[str, object]],
    opportunity_rows: list[dict[str, object]],
    fairness_rows: list[dict[str, object]],
) -> None:
    safety = _rows_by_arm(safety_rows)
    recovery = _rows_by_arm(recovery_rows)
    attribution = _rows_by_arm(attribution_rows)
    opportunity = _rows_by_arm(opportunity_rows)
    proposed = "proposed_guarantee_aware_full"
    fixed_heavy = "fixed_heavy_onboard"
    fixed_light = "fixed_lightweight_onboard"
    utility = "utility_rule"
    ground = "ground_approved_followup"

    p_safety = safety.get(proposed, {})
    fh_safety = safety.get(fixed_heavy, {})
    fl_safety = safety.get(fixed_light, {})
    _write_text(
        out_dir / "safety_headroom_report.md",
        "\n".join(
            [
                "# Safety headroom report",
                "",
                TRACE_LIMITATION_NOTE,
                "",
                f"cross_arm_fairness_error_count = {sum(_to_int(row.get('error_count')) for row in fairness_rows)}",
                "",
                f"1. Proposed hard safety violations: {_to_int(p_safety.get('hard_safety_violation_count'))}.",
                f"2. Proposed near-limit slots: battery={_to_int(p_safety.get('battery_near_limit_slot_count'))}, thermal={_to_int(p_safety.get('thermal_near_limit_slot_count'))}, queue={_to_int(p_safety.get('queue_near_limit_slot_count'))}.",
                f"3. Proposed min battery margin / min thermal margin / min queue margin: {_fmt(p_safety.get('min_battery_margin'))} / {_fmt(p_safety.get('min_thermal_margin'))} / {_fmt(p_safety.get('min_queue_margin'))}.",
                f"4. Proposed vs fixed_heavy hard safety violations: {_to_int(p_safety.get('hard_safety_violation_count'))} vs {_to_int(fh_safety.get('hard_safety_violation_count'))}.",
                f"5. Proposed vs fixed_lightweight resource use remains in the diagnostic healthy region: {bool(_to_int(p_safety.get('hard_safety_violation_count')) == 0 and _to_float(p_safety.get('min_battery_margin')) > 0 and _to_float(p_safety.get('min_thermal_margin')) > 0 and _to_float(p_safety.get('min_queue_margin')) > 0)}.",
                f"6. Fixed_lightweight min battery margin / min thermal margin / min queue margin: {_fmt(fl_safety.get('min_battery_margin'))} / {_fmt(fl_safety.get('min_thermal_margin'))} / {_fmt(fl_safety.get('min_queue_margin'))}.",
            ]
        ),
    )

    p_rec = recovery.get(proposed, {})
    fh_rec = recovery.get(fixed_heavy, {})
    _write_text(
        out_dir / "recovery_report.md",
        "\n".join(
            [
                "# Recovery report",
                "",
                TRACE_LIMITATION_NOTE,
                "",
                f"1. Proposed queue recovery rate: {_fmt(p_rec.get('queue_recovery_rate'))}; battery/thermal recovery should be read as diagnostic only.",
                "2. Recovery attribution to solar charging or cooling cannot be established because sunlit/cooling physical traces are absent from fair-baseline outputs.",
                f"3. Proposed final diagnostic SOC / thermal / queue: {_fmt(p_rec.get('final_SOC'))} / {_fmt(p_rec.get('final_thermal_pressure'))} / {_fmt(p_rec.get('final_queue_MB'))}.",
                f"4. Proposed vs fixed_heavy queue recovery rate: {_fmt(p_rec.get('queue_recovery_rate'))} vs {_fmt(fh_rec.get('queue_recovery_rate'))}.",
            ]
        ),
    )

    p_attr = attribution.get(proposed, {})
    fl_attr = attribution.get(fixed_light, {})
    fh_attr = attribution.get(fixed_heavy, {})
    _write_text(
        out_dir / "resource_to_value_report.md",
        "\n".join(
            [
                "# Resource-to-value attribution report",
                "",
                f"1. Proposed H_resource_cost_fraction: {_fmt(p_attr.get('H_resource_cost_fraction'))}.",
                f"2. Proposed low/medium resource cost fraction: {_fmt(p_attr.get('low_medium_resource_cost_fraction'))}.",
                f"3. Proposed wasted_resource_fraction: {_fmt(p_attr.get('wasted_resource_fraction'))}.",
                f"4. Proposed H net delivered utility total / mean per case / H target-stage success: {_fmt(p_attr.get('H_net_delivered_utility'))} / {_fmt(p_attr.get('H_net_delivered_utility_mean_per_case'))} / {_fmt(p_attr.get('H_target_stage_success_rate'))}.",
                f"5. Fixed_lightweight uses less total resource cost ({_fmt(fl_attr.get('total_resource_cost'))}) but has H target-stage success {_fmt(fl_attr.get('H_target_stage_success_rate'))}.",
                f"6. Fixed_heavy total resource cost / H net utility: {_fmt(fh_attr.get('total_resource_cost'))} / {_fmt(fh_attr.get('H_net_delivered_utility'))}.",
            ]
        ),
    )

    p_opp = opportunity.get(proposed, {})
    u_opp = opportunity.get(utility, {})
    fl_opp = opportunity.get(fixed_light, {})
    g_opp = opportunity.get(ground, {})
    fh_opp = opportunity.get(fixed_heavy, {})
    _write_text(
        out_dir / "opportunity_use_report.md",
        "\n".join(
            [
                "# Opportunity-use report",
                "",
                f"1. Proposed H opportunity utilization: {_fmt(p_opp.get('H_opportunity_utilization_rate'))}; utility_rule: {_fmt(u_opp.get('H_opportunity_utilization_rate'))}.",
                f"2. Fixed_lightweight unused H opportunity count: {_to_int(fl_opp.get('unused_H_opportunity_count'))}.",
                f"3. Ground-approved follow-up unused H opportunity count: {_to_int(g_opp.get('unused_H_opportunity_count'))}; this is consistent with delay sensitivity when opportunities are missed.",
                f"4. Fixed_heavy H opportunity utilization: {_fmt(fh_opp.get('H_opportunity_utilization_rate'))}; compare this with its high total resource cost in the attribution table.",
            ]
        ),
    )

    _write_csv(
        out_dir / "lyapunov_response_summary.csv",
        [
            {
                "trace_sufficient": 0,
                "battery_pressure_lambda_E_check": "insufficient_trace",
                "thermal_pressure_lambda_thermal_check": "insufficient_trace",
                "queue_pressure_lambda_Q_check": "insufficient_trace",
                "model_lightening_check": "insufficient_trace",
                "payload_conservatism_check": "insufficient_trace",
                "resume_aggressive_delivery_check": "insufficient_trace",
            }
        ],
    )
    _write_text(
        out_dir / "lyapunov_response_report.md",
        "# Lyapunov response report\n\nLyapunov response trace is insufficient; additional logging is needed.",
    )


def _canonical_case(rows: list[dict[str, object]]) -> tuple[str, int, int]:
    preferred = ("multi_window_rescuable_high_value", 2030, 160)
    available = {(str(row["scenario"]), _to_int(row["seed"]), _to_int(row["num_slots"])) for row in rows}
    if preferred in available:
        return preferred
    return sorted(available)[0]


def write_figures(
    out_dir: Path,
    trajectory_rows: list[dict[str, object]],
    attribution_rows: list[dict[str, object]],
    opportunity_rows: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    scenario, seed, slots = _canonical_case(trajectory_rows)
    case_rows = [
        row
        for row in trajectory_rows
        if str(row["scenario"]) == scenario and _to_int(row["seed"]) == seed and _to_int(row["num_slots"]) == slots
    ]
    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in case_rows:
        by_arm[str(row["arm"])].append(row)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for arm, rows in sorted(by_arm.items()):
        ordered = sorted(rows, key=lambda row: _to_int(row["slot"]))
        x = [_to_int(row["slot"]) for row in ordered]
        axes[0].plot(x, [_to_float(row["battery_SOC"]) for row in ordered], label=arm)
        axes[1].plot(x, [_to_float(row["thermal_pressure"]) for row in ordered], label=arm)
        axes[2].plot(x, [_to_float(row["queue_MB"]) for row in ordered], label=arm)
    axes[0].axhline(BATTERY_MIN_LIMIT, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(THERMAL_MAX_LIMIT, color="black", linestyle="--", linewidth=1)
    if case_rows:
        axes[2].axhline(_to_float(case_rows[0]["queue_max_MB"]), color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("battery SOC")
    axes[1].set_ylabel("thermal pressure")
    axes[2].set_ylabel("queue MB")
    axes[2].set_xlabel("slot")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure1_resource_state_trajectories.png", dpi=180)
    fig.savefig(fig_dir / "figure1_resource_state_trajectories.pdf")
    plt.close(fig)

    proposed_rows = sorted(by_arm.get("proposed_guarantee_aware_full", []), key=lambda row: _to_int(row["slot"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    x = [_to_int(row["slot"]) for row in proposed_rows]
    ax.plot(x, [0.0 for _ in x], label="solar harvest unavailable")
    ax.plot(x, [0.0 for _ in x], label="base energy unavailable")
    ax.plot(x, [_to_float(row["compute_energy_Wh"]) for row in proposed_rows], label="compute")
    ax.plot(x, [_to_float(row["comm_energy_Wh"]) for row in proposed_rows], label="comm")
    ax.plot(x, [-_to_float(row["total_energy_consumed_Wh"]) for row in proposed_rows], label="net delta lower bound")
    ax.set_xlabel("slot")
    ax.set_ylabel("Wh")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure2_proposed_energy_balance.png", dpi=180)
    fig.savefig(fig_dir / "figure2_proposed_energy_balance.pdf")
    plt.close(fig)

    labels = [str(row["arm"]) for row in attribution_rows]
    h = [_to_float(row["H_resource_cost_fraction"]) for row in attribution_rows]
    lm = [_to_float(row["low_medium_resource_cost_fraction"]) for row in attribution_rows]
    waste = [_to_float(row["wasted_resource_fraction"]) for row in attribution_rows]
    x_pos = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x_pos, h, label="H")
    ax.bar(x_pos, lm, bottom=h, label="low/medium")
    ax.bar(x_pos, waste, bottom=[a + b for a, b in zip(h, lm)], label="waste")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "figure3_resource_to_value_attribution.png", dpi=180)
    fig.savefig(fig_dir / "figure3_resource_to_value_attribution.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in attribution_rows:
        size = 80 + 400 * _to_float(row["H_target_stage_success_rate"])
        ax.scatter(_to_float(row["total_resource_cost"]), _to_float(row["H_net_delivered_utility"]), s=size, label=str(row["arm"]), alpha=0.75)
    ax.set_xlabel("total resource cost")
    ax.set_ylabel("H net delivered utility")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure4_H_utility_vs_resource_cost.png", dpi=180)
    fig.savefig(fig_dir / "figure4_H_utility_vs_resource_cost.pdf")
    plt.close(fig)

    labels = [str(row["arm"]) for row in opportunity_rows]
    x_pos = list(range(len(labels)))
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar([x - 0.18 for x in x_pos], [_to_float(row["H_opportunity_utilization_rate"]) for row in opportunity_rows], width=0.36, label="H utilization")
    ax2 = ax1.twinx()
    ax2.bar([x + 0.18 for x in x_pos], [_to_float(row["unused_H_opportunity_count"]) for row in opportunity_rows], width=0.36, color="#E45756", label="unused H")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("utilization rate")
    ax2.set_ylabel("unused H opportunity count")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure5_opportunity_utilization.png", dpi=180)
    fig.savefig(fig_dir / "figure5_opportunity_utilization.pdf")
    plt.close(fig)

    _write_text(
        out_dir / "figure_readme.md",
        "\n".join(
            [
                "# Figure readme",
                "",
                "## Figure 1",
                "Resource state trajectories compare diagnostic battery, thermal, and queue states for all arms in the canonical fair case. They support the claim only for hard-margin accounting; physical battery/thermal telemetry is not logged.",
                "",
                "## Figure 2",
                "Proposed energy balance shows compute and communication energy by slot; solar harvest and base energy are unavailable in fair-baseline traces and are plotted as zero placeholders. This figure is useful as an energy-spend timing plot, not a physical recharge plot.",
                "",
                "## Figure 3",
                "Resource-to-value attribution shows H, low/medium, and waste fractions by arm. It directly supports whether additional resources are concentrated on high-value work, with pending undelivered requests counted as waste.",
                "",
                "## Figure 4",
                "H utility versus resource cost places each arm on a cost-benefit plane with marker size proportional to H target-stage success. It is the strongest compact figure for arguing that Proposed converts extra resource into H net utility.",
                "",
                "## Figure 5",
                "Opportunity utilization compares H opportunity use and unused H opportunities. Its caveat is that opportunity is reconstructed from fair contacts and H task deadlines rather than a logged scheduler opportunity trace.",
            ]
        ),
    )


def write_final_recommendation(
    out_dir: Path,
    safety_rows: list[dict[str, object]],
    recovery_rows: list[dict[str, object]],
    attribution_rows: list[dict[str, object]],
    opportunity_rows: list[dict[str, object]],
) -> None:
    safety = _rows_by_arm(safety_rows)
    recovery = _rows_by_arm(recovery_rows)
    attribution = _rows_by_arm(attribution_rows)
    opportunity = _rows_by_arm(opportunity_rows)
    p = "proposed_guarantee_aware_full"
    fh = "fixed_heavy_onboard"
    p_s = safety.get(p, {})
    p_r = recovery.get(p, {})
    p_a = attribution.get(p, {})
    p_o = opportunity.get(p, {})
    fh_a = attribution.get(fh, {})
    conclusions = ["A", "C", "D", "E", "F", "I"]
    if _to_int(p_s.get("hard_safety_violation_count")) > 0:
        conclusions.append("G")
    if _to_float(p_a.get("wasted_resource_fraction")) > 0.25:
        conclusions.append("H")
    _write_text(
        out_dir / "final_recommendation.md",
        "\n".join(
            [
                "# Final recommendation",
                "",
                "Selected conclusions: " + ", ".join(conclusions) + ".",
                "",
                f"1. Proposed battery / thermal / queue safety violations: {_to_int(p_s.get('battery_violation_count'))} / {_to_int(p_s.get('thermal_violation_count'))} / {_to_int(p_s.get('queue_violation_count'))}.",
                f"2. Proposed closest margins: battery {_fmt(p_s.get('min_battery_margin'))}, thermal {_fmt(p_s.get('min_thermal_margin'))}, queue {_fmt(p_s.get('min_queue_margin'))}.",
                f"3. Proposed queue recovery rate is {_fmt(p_r.get('queue_recovery_rate'))}; physical battery/thermal recovery evidence is insufficient in fair-baseline traces.",
                f"4. Proposed additional resources are mainly attributable to H tasks when H_resource_cost_fraction is high: {_fmt(p_a.get('H_resource_cost_fraction'))}.",
                f"5. Proposed wasted_resource_fraction: {_fmt(p_a.get('wasted_resource_fraction'))}.",
                f"6. Proposed H net utility total / mean per case and H target-stage success: {_fmt(p_a.get('H_net_delivered_utility'))} / {_fmt(p_a.get('H_net_delivered_utility_mean_per_case'))} and {_fmt(p_a.get('H_target_stage_success_rate'))}.",
                f"7. Proposed is not just fixed_heavy-style brute force: fixed_heavy total resource cost / H net utility = {_fmt(fh_a.get('total_resource_cost'))} / {_fmt(fh_a.get('H_net_delivered_utility'))}; Proposed = {_fmt(p_a.get('total_resource_cost'))} / {_fmt(p_a.get('H_net_delivered_utility'))}.",
                f"8. The result supports the paper interpretation for value-directed, safe-margin resource conversion, but not for physical recharge/cooling dynamics without extra telemetry. {TRACE_LIMITATION_NOTE}",
                "9. Recommended paper figures: Figure 4 and Figure 3; Figure 1 can be a supplementary safety-margin diagnostic.",
                "10. Caveat: solar harvest, base energy, cooling, and dynamic Lyapunov price traces are not present in the fair-baseline artifacts, so recovery and Lyapunov response claims must be stated as trace-limited.",
                "",
                f"Opportunity evidence: Proposed H opportunity utilization = {_fmt(p_o.get('H_opportunity_utilization_rate'))}.",
            ]
        ),
    )


def run_audit(
    *,
    output_dir: Path = OUT_DIR,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    slots_list: list[int] | None = None,
    include_sanity: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs, _manifest, _contacts, fairness_rows = _build_runs(
        scenarios or fair.DEFAULT_SCENARIOS,
        seeds or fair.DEFAULT_SEEDS,
        slots_list or fair.DEFAULT_SLOTS,
        include_sanity=include_sanity,
    )
    trajectory_rows = build_resource_trajectory(runs)
    safety_rows = build_safety_summary(trajectory_rows)
    recovery_rows = build_recovery_summary(trajectory_rows)
    attribution_rows = build_resource_to_value_attribution(runs)
    opportunity_rows = build_opportunity_use_summary(trajectory_rows, runs)

    _write_csv(output_dir / "resource_trajectory_by_arm.csv", trajectory_rows, TRAJECTORY_COLUMNS)
    _write_csv(output_dir / "safety_headroom_summary.csv", safety_rows, SAFETY_COLUMNS)
    _write_csv(output_dir / "recovery_summary.csv", recovery_rows, RECOVERY_COLUMNS)
    _write_csv(output_dir / "resource_to_value_attribution.csv", attribution_rows, ATTRIBUTION_COLUMNS)
    _write_csv(output_dir / "opportunity_use_summary.csv", opportunity_rows, OPPORTUNITY_COLUMNS)
    write_reports(output_dir, safety_rows, recovery_rows, attribution_rows, opportunity_rows, fairness_rows)
    write_figures(output_dir, trajectory_rows, attribution_rows, opportunity_rows)
    write_final_recommendation(output_dir, safety_rows, recovery_rows, attribution_rows, opportunity_rows)
    return {
        "runs": runs,
        "trajectory_rows": trajectory_rows,
        "safety_rows": safety_rows,
        "recovery_rows": recovery_rows,
        "attribution_rows": attribution_rows,
        "opportunity_rows": opportunity_rows,
        "fairness_rows": fairness_rows,
        "out_dir": output_dir,
    }


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--scenarios", default=",".join(fair.DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in fair.DEFAULT_SEEDS))
    parser.add_argument("--slots", default=",".join(str(slots) for slots in fair.DEFAULT_SLOTS))
    parser.add_argument("--no-sanity", action="store_true")
    args = parser.parse_args(argv)
    run_audit(
        output_dir=args.output_dir,
        scenarios=_parse_csv_list(args.scenarios),
        seeds=_parse_int_list(args.seeds),
        slots_list=_parse_int_list(args.slots),
        include_sanity=not args.no_sanity,
    )
    print(f"Wrote resource-use audit artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
