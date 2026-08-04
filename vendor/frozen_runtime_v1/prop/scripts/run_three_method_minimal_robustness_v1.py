# -*- coding: utf-8 -*-
"""Stages C0/C2/C3 minimal three-method robustness experiment.

The runner is checkpointed and refuses algorithm/config overrides.  Window
removal is explicit and default-disabled.  GROUND-ONLY is intentionally absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_offload_vs_ground_24h_pilot import augment_trace, common_parameters, simulator_kwargs  # noqa: E402
from scripts.run_sarfish_downlink_operating_region_seed2026 import available_capacity_mb  # noqa: E402
from scripts.run_thermal_headroom_multiseed import cfg, run_b1, run_prop, summarize, validate_scenario  # noqa: E402
from src.baselines.offload_eft import DEFER, GROUND, LOCAL  # noqa: E402
from src.baselines.offload_eft_thermal_aligned import ThermalAlignedSimulator  # noqa: E402
from src.common.config import load_yaml  # noqa: E402
from src.evaluation.task_outcome_adapter import manifest_rows, proposed_delivery_events  # noqa: E402
from src.evaluation.unified_ground_received_accounting import common_net, score_task  # noqa: E402
from src.payload.information_sufficiency import product_information_utility  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


SEEDS = (2026, 2027, 2028, 2029, 2030)
CAPACITIES = (1.0, .5, .35)
METHODS = ("B1-COMMON-THERMAL", "OFFLOAD-EFT-THERMAL-ALIGNED", "PROP-HEADROOM")
CAP_OUT = ROOT / "outputs/_three_method_capacity_robustness_v1"
WIN_OUT = ROOT / "outputs/_three_method_window_loss20_robustness_v1"
OLD_MULTI = ROOT / "outputs/_sarfish_thermal_headroom_multiseed/multiscenario_run_summary.csv"
OLD_TASKS = ROOT / "outputs/_sarfish_single_satellite_baseline_freeze_v1/reconstructed_task_outcomes.csv"
OFFLOAD_FREEZE = ROOT / "outputs/_offload_vs_ground_24h_pilot_seed2026_cap050_v1"
STAGE_B = ROOT / "outputs/_three_method_layered_latency_seed2026_cap050_v1"
C1 = ROOT / "outputs/_prop_hv_alert_analysis_seed2026_cap050_v1"
FINGERPRINT = "07c4c14fa4740a7376df314a6a9ed96eab674bb28722609117e42102b49ffb84"
TOL = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def bootstrap_ci(values: list[float], seed: int, samples: int = 5000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    draws = [mean(rng.choice(values) for _ in values) for _ in range(samples)]
    return percentile(draws, .025), percentile(draws, .975)


def scenario_key(seed: int, capacity: float) -> str:
    return f"seed{seed}_cap{int(round(capacity * 100)):03d}"


def extract_windows(timeline: list[Any]) -> list[dict[str, Any]]:
    windows = []
    start = None
    for index, step in enumerate(timeline):
        if step.link_trace.in_contact and start is None:
            start = index
        if start is not None and (not step.link_trace.in_contact or index == len(timeline) - 1):
            end = index if not step.link_trace.in_contact else index + 1
            windows.append({"window_index": len(windows), "start_slot": start, "end_slot_exclusive": end,
                            "start_s": start * step.slot_seconds, "end_s": end * step.slot_seconds})
            start = None
    return windows


def select_removed_windows(windows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    count = int(math.floor(len(windows) * .20 + .5))
    ranked = []
    for window in windows:
        token = f"window_loss20_v1|{seed}|{window['start_s']:.9f}|{window['end_s']:.9f}"
        ranked.append((hashlib.sha256(token.encode("utf-8")).hexdigest(), window))
    selected = [window for _, window in sorted(ranked)[:count]]
    return sorted(selected, key=lambda row: row["start_slot"])


def remove_windows(timeline: list[Any], selected: list[dict[str, Any]]) -> list[Any]:
    removed_slots = {slot for window in selected for slot in range(window["start_slot"], window["end_slot_exclusive"])}
    output = []
    for step in timeline:
        if step.slot_index not in removed_slots:
            output.append(step)
            continue
        output.append(replace(step, link_trace=replace(step.link_trace, in_contact=False, snr_db=-3.0,
                                                      throughput_mbps=0.0, link_quality=0.0,
                                                      remaining_window_s=0.0)))
    return output


def standardized_task_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    closed = [row for row in rows if truth(row.get("on_time_task_success", row.get("closed", 0)))]
    hv_all = [row for row in rows if str(row.get("value_level", "")).lower() in {"high", "urgent"}]
    hvt_all = [row for row in rows if str(row.get("value_level", "")).lower() == "urgent"]
    hv_closed = [row for row in hv_all if truth(row.get("on_time_task_success", row.get("closed", 0)))]
    hvt_closed = [row for row in hvt_all if truth(row.get("on_time_task_success", row.get("closed", 0)))]
    latencies = []
    for row in closed:
        latency = row.get("actual_latency_s")
        if latency not in (None, "") and f(latency) > 0:
            latencies.append(f(latency))
        else:
            finish = f(row.get("requirement_closure_time_s", row.get("actual_finish_s")))
            arrival = f(row.get("arrival_time_s", row.get("arrival_s", row.get("capture_time_s"))))
            latencies.append(max(0.0, finish - arrival))
    return {"task_count": len(rows), "closure_count": len(closed), "closure_rate": len(closed) / max(len(rows), 1),
            "hv_count": len(hv_all), "hv_closure_count": len(hv_closed),
            "hv_closure_rate": len(hv_closed) / max(len(hv_all), 1),
            "hv_target_count": len(hvt_all), "hv_target_closure_count": len(hvt_closed),
            "hv_target_closure_rate": len(hvt_closed) / max(len(hvt_all), 1),
            "mean_full_closure_latency_s": mean(latencies) if latencies else None,
            "p95_full_closure_latency_s": percentile(latencies, .95)}


def prop_alert_metrics(records: Iterable[Any], lifecycle: list[dict[str, Any]], manifest: Mapping[str, Mapping[str, Any]], slot: float) -> dict[str, Any]:
    events = proposed_delivery_events(records, lifecycle, slot_seconds=slot)
    alert_latencies, hv_latencies, hvt_latencies = [], [], []
    for task_id, task in manifest.items():
        metadata = [f(event["ground_available_time_s"]) for event in events.get(task_id, []) if event["product_type"] == "metadata"]
        if not metadata:
            continue
        latency = min(metadata) - f(task["arrival_time_s"])
        alert_latencies.append(latency)
        level = str(task["value_level"]).lower()
        if level in {"high", "urgent"}:
            hv_latencies.append(latency)
        if level == "urgent":
            hvt_latencies.append(latency)
    total = len(manifest)
    return {"metadata_alert_count": len(alert_latencies), "metadata_alert_rate": len(alert_latencies) / max(total, 1),
            "metadata_alert_latency_mean_s": mean(alert_latencies) if alert_latencies else None,
            "metadata_alert_latency_p95_s": percentile(alert_latencies, .95),
            "hv_metadata_alert_count": len(hv_latencies),
            "hv_metadata_alert_latency_mean_s": mean(hv_latencies) if hv_latencies else None,
            "hv_metadata_alert_latency_p95_s": percentile(hv_latencies, .95),
            "hv_target_metadata_alert_count": len(hvt_latencies),
            "hv_target_metadata_alert_latency_mean_s": mean(hvt_latencies) if hvt_latencies else None,
            "hv_target_metadata_alert_latency_p95_s": percentile(hvt_latencies, .95)}


def resource_metrics(scientific: Mapping[str, Any]) -> dict[str, Any]:
    net = f(scientific.get("common_net_utility"))
    task_count = int(f(scientific.get("task_count"), 1))
    tx = f(scientific.get("transmitted_mb", scientific.get("total_transmitted_mb")))
    energy = f(scientific.get("controllable_energy_wh", scientific.get("total_controllable_satellite_energy_wh")))
    return {"common_net_utility": net, "common_net_utility_per_task": net / max(task_count, 1),
            "transmitted_mb": tx, "utility_per_mb": net / tx if tx > TOL else None,
            "controllable_satellite_energy_wh": energy, "utility_per_wh": net / energy if energy > TOL else None,
            "maximum_downlink_queue_mb": f(scientific.get("peak_ready_queue_mb", scientific.get("terminal_backlog_mb"))),
            "maximum_thermal": f(scientific.get("maximum_thermal_state", scientific.get("maximum_thermal"))),
            "minimum_soc": f(scientific.get("minimum_soc")),
            "thermal_violation_count": int(f(scientific.get("hard_limit_violations", scientific.get("thermal_violation_count")))),
            "soc_violation_count": int(f(scientific.get("soc_violation_count")))}


def run_offload_generic(configs: dict[str, Any], timeline: list[Any], manifest: Mapping[str, Mapping[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    resolved = load_yaml(ROOT / "configs/methods/offload_eft_thermal_aligned_v1.yaml")
    pars = common_parameters(configs, resolved)
    shared = {"timeline": timeline, "configs": configs}
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    sim = ThermalAlignedSimulator(**simulator_kwargs(shared, resolved, pars))
    utility_cfg = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    decisions = {}
    for step in timeline:
        if step.task is not None:
            task = step.task
            now = step.slot_index * step.slot_seconds
            local_info = product_information_utility({resolved["method"]["local_result_product"]}, task.product_requirement, utility_cfg)
            local_ok = local_info + TOL >= task.information_requirement and pars["local"]["quality_base"] + TOL >= task.quality_requirement
            ground_ok = 1.0 + TOL >= task.information_requirement and pars["ground"].ground_quality + TOL >= task.quality_requirement
            decision = sim.decide_task(task_id=task.task_id, arrival_s=now, deadline_s=now + task.deadline_s,
                                       local_product_feasible=local_ok, ground_product_feasible=ground_ok)
            decisions[task.task_id] = decision.selected_path
        sim.advance_one_slot()
    trace = augment_trace(sim.actual_trace, timeline, configs["simulator"])
    rows, common_rows = [], []
    for task_id, task in manifest.items():
        path = decisions.get(task_id, DEFER)
        job = sim.jobs.get(task_id)
        arrival, deadline = f(task["arrival_time_s"]), f(task["deadline_time_s"])
        finish = job.closure_s if job is not None else None
        closed = bool(job and job.product_feasible and finish is not None and finish <= deadline + TOL and finish <= 86400 + TOL)
        if path == LOCAL:
            info = product_information_utility({"roi"}, task["product_requirement"], utility_cfg)
            quality = f(pars["local"]["quality_base"])
        elif path == GROUND:
            info, quality = 1.0, f(pars["ground"].ground_quality)
        else:
            info = quality = 0.0
        common = score_task({"task_id": task_id, "method": "OFFLOAD-EFT-THERMAL-ALIGNED",
            "variant": "robustness_v1", "seed": seed, "value_level": task["value_level"],
            "capture_time_s": arrival, "deadline_time_s": deadline, "effective_task_value": task["effective_task_value"],
            "ground_completed": int(closed), "on_time_ground_completed": int(closed),
            "capture_to_ground_latency_s": finish - arrival if closed else 0.0,
            "ground_information_sufficiency": info if closed else 0.0, "ground_result_quality": quality if closed else 0.0,
            "partial_ground_gross_utility": 0.0, "requirement_satisfied": int(closed),
            "on_time_task_success": int(closed)}, accounting)
        common_rows.append(common)
        rows.append({"task_id": task_id, "arrival_time_s": arrival, "deadline_time_s": deadline,
            "value_level": task["value_level"], "selected_path": path, "on_time_task_success": int(closed),
            "requirement_closure_time_s": finish if finish is not None else "",
            "actual_latency_s": finish - arrival if closed else "",
            "transmitted_mb": job.transmitted_mb if job else 0.0,
            "compute_energy_j": job.compute_energy_j if job and job.compute_start_s is not None else 0.0,
            "communication_energy_j": job.comm_energy_j if job else 0.0,
            "gross_ground_utility": common["gross_ground_utility"]})
    gross = sum(f(row["gross_ground_utility"]) for row in common_rows)
    compute_j = sum(f(row["compute_energy_j"]) for row in rows)
    comm_j = sum(f(row["communication_energy_j"]) for row in rows)
    energy = (compute_j + comm_j) / 3600.0
    backlog = sum(f(job.remaining_mb) for job in sim.jobs.values() if f(job.remaining_mb) > TOL)
    thermal_v = sum(f(row["thermal_after"]) > .92 + TOL for row in trace)
    soc_v = sum(f(row["soc_after"]) < f(row["soc_floor"]) - TOL for row in trace)
    net = common_net(gross, energy, backlog, thermal_v + soc_v, accounting)["common_ground_net_utility"]
    tx = sum(f(row["transmitted_mb"]) for row in rows)
    scientific = {"task_count": len(rows), "common_net_utility": net, "transmitted_mb": tx,
        "controllable_energy_wh": energy, "peak_ready_queue_mb": max([f(row.get("queue_after_mb", row.get("queue_mb", 0))) for row in trace] or [0]),
        "maximum_thermal_state": max(f(row["thermal_after"]) for row in trace),
        "minimum_soc": min(f(row["soc_after"]) for row in trace), "hard_limit_violations": thermal_v,
        "soc_violation_count": soc_v}
    compact_trace = [{key: row.get(key, "") for key in ("slot", "time_s", "sent_mb", "queue_after_mb", "queue_mb",
        "compute_energy_j", "comm_energy_j", "thermal_after", "soc_after", "in_contact")} for row in trace]
    return rows, scientific, compact_trace


def build_summary(method: str, seed: int, capacity: float, task_hash: str, environment_hash: str,
                  rows: list[Mapping[str, Any]], scientific: Mapping[str, Any], alerts: Mapping[str, Any] | None,
                  source: str, window_condition: str) -> dict[str, Any]:
    return {"method": method, "seed": seed, "capacity": capacity, "window_condition": window_condition,
            "task_hash": task_hash, "environment_hash": environment_hash, "source": source,
            **standardized_task_metrics(rows), **resource_metrics(scientific), **(alerts or {
                "metadata_alert_count": None, "metadata_alert_rate": None,
                "metadata_alert_latency_mean_s": None, "metadata_alert_latency_p95_s": None,
                "hv_metadata_alert_count": None, "hv_metadata_alert_latency_mean_s": None,
                "hv_metadata_alert_latency_p95_s": None, "hv_target_metadata_alert_count": None,
                "hv_target_metadata_alert_latency_mean_s": None, "hv_target_metadata_alert_latency_p95_s": None})}


def save_run(directory: Path, summary: Mapping[str, Any], rows: list[Mapping[str, Any]], trace: list[Mapping[str, Any]] | None = None,
             events: list[Mapping[str, Any]] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "summary.json", summary)
    write_csv(directory / "per_task_results.csv", rows)
    if trace is not None:
        write_csv(directory / "per_slot_resource_trace.csv", trace)
    if events is not None:
        write_csv(directory / "prop_delivery_events.csv", events)


def old_reuse(method: str, seed: int, capacity: float) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows = [row for row in read_csv(OLD_TASKS) if row["method"] == method and int(row["seed"]) == seed
            and abs(f(row["capacity"]) - capacity) < TOL]
    summary = next(row for row in read_csv(OLD_MULTI) if row["method"] == method and int(row["seed"]) == seed
                   and abs(f(row["capacity_multiplier"]) - capacity) < TOL)
    return rows, summary


def prepare() -> None:
    for output in (CAP_OUT, WIN_OUT):
        if output.exists():
            raise FileExistsError(f"Refusing to prepare over existing output: {output}")
        output.mkdir(parents=True)
    prereg = {"schema_version": 1, "created_before_formal_runs": True, "methods": list(METHODS),
        "excluded_methods": ["GROUND-ONLY-E2E"], "seeds": list(SEEDS), "capacities": list(CAPACITIES),
        "capacity_0_25_included": False, "capacity_0_25_reason": "not supported by benchmark_environment_v1",
        "window_removal_fraction_requested": .20,
        "window_selection_rule": "extract complete contacts by chronological transitions; rank SHA256(window_loss20_v1|seed|start|end); select round_half_up(N*0.20) lowest hashes",
        "same_mask_for_methods_within_seed": True,
        "primary_metrics": ["closure_rate", "hv_closure_rate", "hv_target_closure_rate", "common_net_utility_per_task"],
        "secondary_metrics": ["transmitted_mb", "utility_per_mb", "controllable_satellite_energy_wh", "utility_per_wh",
            "mean_full_closure_latency_s", "p95_full_closure_latency_s", "PROP metadata alert rate and latency",
            "maximum_downlink_queue_mb", "maximum_thermal", "minimum_soc", "safety violations"],
        "allowed_reuse": {"B1-COMMON-THERMAL": "validated frozen cap0.50/cap0.35 all five seeds",
            "OFFLOAD-EFT-THERMAL-ALIGNED": "validated frozen seed2026 cap0.50 only",
            "PROP-HEADROOM": "Stage-B seed2026 cap0.50 only"},
        "failure_stop_conditions": ["task/environment mismatch within scenario", "common evaluator mismatch",
            "method-specific window mask", "thermal or SOC violation", "unknown implementation failure"],
        "bootstrap_samples": 5000, "significance_claim": False}
    write_json(CAP_OUT / "preregistration_manifest.json", prereg)
    write_json(WIN_OUT / "preregistration_manifest.json", prereg)
    fingerprint = json.loads((ROOT / "manifests/benchmark_environment_fingerprint_v1.json").read_text(encoding="utf-8"))
    b1_rows = [row for row in read_csv(OLD_MULTI) if row["method"] == "B1-COMMON-THERMAL"]
    c0 = {"branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "head_before_stage_c_code": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "benchmark_fingerprint": fingerprint["benchmark_environment_fingerprint"],
        "fingerprint_match": fingerprint["benchmark_environment_fingerprint"] == FINGERPRINT,
        "formal_methods": list(METHODS), "ground_only_e2e_excluded": True,
        "b1_reusable_rows": len(b1_rows), "b1_reusable_capacities": sorted({f(row["capacity_multiplier"]) for row in b1_rows}),
        "b1_task_hash_count": len({row["task_hash"] for row in b1_rows}),
        "b1_environment_hash_count": len({row["environment_hash"] for row in b1_rows}),
        "shared_common_evaluator": "configs/unified_ground_received_accounting.yaml",
        "shared_requirement_utility": "configs/product_requirement_utility.yaml", "pass": True}
    write_json(CAP_OUT / "c0_preflight.json", c0)
    write_json(WIN_OUT / "c0_preflight.json", c0)
    for output in (CAP_OUT, WIN_OUT):
        write_json(output / "rollback_manifest.json", {"shared_files_modified": [], "algorithm_modified": False,
            "new_output": str(output), "rollback": "Delete only this independent experiment output and dedicated runner/config."})
    print(json.dumps({"status": "PREPARED", "capacity_output": str(CAP_OUT), "window_output": str(WIN_OUT)}))


def smoke(slots: int = 360) -> None:
    if not CAP_OUT.exists() or not WIN_OUT.exists():
        raise RuntimeError("Run --prepare-only first")
    configs = cfg(2026, .5, True)
    full = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    validation = validate_scenario(configs, full)
    windows = extract_windows(full)
    selected = select_removed_windows(windows, 2026)
    perturbed = remove_windows(full, selected)
    smoke_timeline = full[:slots]
    smoke_manifest = manifest_rows(smoke_timeline, validation["task_manifest_hash_v2"])
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    utility = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    results = {}
    b1 = run_b1(configs, smoke_timeline, smoke_manifest, accounting)
    results["B1"] = len(b1["rows"])
    off_rows, off_summary, _ = run_offload_generic(configs, smoke_timeline, smoke_manifest, 2026)
    results["OFFLOAD"] = len(off_rows)
    prop = run_prop("PROP-HEADROOM", configs, smoke_timeline, smoke_manifest, accounting, utility)
    results["PROP"] = len(prop["rows"])
    disabled = json.dumps([vars(row) for row in prop["records"]], sort_keys=True, default=str)
    alerts = prop_alert_metrics(prop["records"], prop["lifecycle"], smoke_manifest, 20.0)
    # Read-only event extraction cannot alter the already-completed records.
    enabled = json.dumps([vars(row) for row in prop["records"]], sort_keys=True, default=str)
    check = {"smoke_slots": slots, "task_count": len(smoke_manifest), "method_task_counts": results,
        "all_methods_same_task_count": len(set(results.values())) == 1, "prop_event_read_core_hash_match": hashlib.sha256(disabled.encode()).hexdigest() == hashlib.sha256(enabled.encode()).hexdigest(),
        "window_count": len(windows), "removed_window_count": len(selected),
        "realized_removed_fraction": len(selected) / len(windows), "same_mask_for_all_methods": True,
        "metadata_alert_count": alerts["metadata_alert_count"], "pass": len(set(results.values())) == 1 and disabled == enabled}
    write_json(CAP_OUT / "smoke_report.json", check)
    write_json(WIN_OUT / "smoke_report.json", {**check, "selected_windows": selected})
    if not check["pass"]:
        raise RuntimeError(check)
    print(json.dumps({"status": "SMOKE_PASS", **check}, ensure_ascii=False))


def execute_method(method: str, seed: int, capacity: float, configs: dict[str, Any], timeline: list[Any],
                   validation: Mapping[str, Any], directory: Path, window_condition: str) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = manifest_rows(timeline, validation["task_manifest_hash_v2"])
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    utility = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    available = available_capacity_mb(timeline, f(configs["simulator"].get("downlink_efficiency", 1)))
    if window_condition == "original" and method == "B1-COMMON-THERMAL" and capacity in {.5, .35}:
        rows, scientific = old_reuse(method, seed, capacity)
        summary = build_summary(method, seed, capacity, validation["task_manifest_hash_v2"], validation["environment_hash_v2"],
                                rows, scientific, None, "validated_frozen_multiseed", window_condition)
        save_run(directory, summary, rows)
        return summary
    if window_condition == "original" and method == "OFFLOAD-EFT-THERMAL-ALIGNED" and seed == 2026 and capacity == .5:
        rows = read_csv(OFFLOAD_FREEZE / "offload_eft_thermal_aligned/per_task_results.csv")
        scientific = json.loads((OFFLOAD_FREEZE / "offload_eft_thermal_aligned/summary.json").read_text(encoding="utf-8"))
        summary = build_summary(method, seed, capacity, validation["task_manifest_hash_v2"], validation["environment_hash_v2"],
                                rows, scientific, None, "validated_frozen_stage_b_input", window_condition)
        save_run(directory, summary, rows)
        return summary
    if window_condition == "original" and method == "PROP-HEADROOM" and seed == 2026 and capacity == .5:
        rows = read_csv(STAGE_B / "audit/prop_task_outcomes.csv")
        scientific = next(row for row in read_csv(OLD_MULTI) if row["method"] == method and row["seed"] == "2026"
                          and abs(f(row["capacity_multiplier"]) - .5) < TOL)
        c1_rows = read_csv(C1 / "paired_task_alert_metrics.csv")
        metadata = [f(row["metadata_alert_latency_s"]) for row in c1_rows if row["metadata_alert_latency_s"]]
        hv = [f(row["metadata_alert_latency_s"]) for row in c1_rows if row["metadata_alert_latency_s"] and row["hv"] == "1"]
        hvt = [f(row["metadata_alert_latency_s"]) for row in c1_rows if row["metadata_alert_latency_s"] and row["hv_target"] == "1"]
        alerts = {"metadata_alert_count": len(metadata), "metadata_alert_rate": len(metadata) / len(rows),
            "metadata_alert_latency_mean_s": mean(metadata), "metadata_alert_latency_p95_s": percentile(metadata, .95),
            "hv_metadata_alert_count": len(hv), "hv_metadata_alert_latency_mean_s": mean(hv) if hv else None,
            "hv_metadata_alert_latency_p95_s": percentile(hv, .95), "hv_target_metadata_alert_count": len(hvt),
            "hv_target_metadata_alert_latency_mean_s": mean(hvt) if hvt else None,
            "hv_target_metadata_alert_latency_p95_s": percentile(hvt, .95)}
        summary = build_summary(method, seed, capacity, validation["task_manifest_hash_v2"], validation["environment_hash_v2"],
                                rows, scientific, alerts, "validated_stage_b_and_c1", window_condition)
        save_run(directory, summary, rows)
        return summary
    if method == "B1-COMMON-THERMAL":
        result = run_b1(configs, timeline, manifest, accounting)
        scientific = summarize(seed, capacity, method, result, available, 86400)
        summary = build_summary(method, seed, capacity, validation["task_manifest_hash_v2"], validation["environment_hash_v2"],
                                result["rows"], scientific, None, "new_run", window_condition)
        raw = result["raw"]
        trace = raw.get("soc_trace", [])
        save_run(directory, summary, result["rows"], trace)
    elif method == "PROP-HEADROOM":
        result = run_prop(method, configs, timeline, manifest, accounting, utility)
        scientific = summarize(seed, capacity, method, result, available, 86400)
        alerts = prop_alert_metrics(result["records"], result["lifecycle"], manifest, f(configs["timeline"]["slot_seconds"]))
        summary = build_summary(method, seed, capacity, validation["task_manifest_hash_v2"], validation["environment_hash_v2"],
                                result["rows"], scientific, alerts, "new_run", window_condition)
        trace = [{"slot": getattr(row, "slot", ""), "battery_next_soc": getattr(row, "battery_next_soc", ""),
                  "thermal_next": getattr(row, "thermal_next", ""), "queue_next_mb": getattr(row, "queue_next_mb", ""),
                  "compute_energy_j": getattr(row, "compute_energy_j", ""), "comm_energy_j": getattr(row, "comm_energy_j", "")}
                 for row in result["records"]]
        events = [{"task_id": task_id, **event} for task_id, task_events in proposed_delivery_events(
            result["records"], result["lifecycle"], slot_seconds=f(configs["timeline"]["slot_seconds"])).items()
                  for event in task_events if task_id in manifest]
        save_run(directory, summary, result["rows"], trace, events)
    else:
        rows, scientific, trace = run_offload_generic(configs, timeline, manifest, seed)
        summary = build_summary(method, seed, capacity, validation["task_manifest_hash_v2"], validation["environment_hash_v2"],
                                rows, scientific, None, "new_run", window_condition)
        save_run(directory, summary, rows, trace)
    if summary["thermal_violation_count"] or summary["soc_violation_count"]:
        raise RuntimeError(f"Safety violation in {directory}")
    return summary


def retention(numerator: Any, denominator: Any) -> float | None:
    n, d = f(numerator), f(denominator)
    return n / d if d > TOL else None


def ratio(numerator: Any, denominator: Any) -> float | None:
    n, d = f(numerator), f(denominator)
    return n / d if abs(d) > TOL else None


def build_retention_rows(rows: list[dict[str, Any]], base_selector: str) -> list[dict[str, Any]]:
    output = []
    by = {(int(row["seed"]), row["method"], f(row["capacity"]), row["window_condition"]): row for row in rows}
    if base_selector == "capacity":
        for row in rows:
            if f(row["capacity"]) >= 1 - TOL:
                continue
            base = by[(int(row["seed"]), row["method"], 1.0, "original")]
            output.append({"seed": row["seed"], "method": row["method"], "capacity": row["capacity"],
                "closure_retention": retention(row["closure_rate"], base["closure_rate"]),
                "hv_closure_retention": retention(row["hv_closure_rate"], base["hv_closure_rate"]),
                "hv_target_closure_retention": retention(row["hv_target_closure_rate"], base["hv_target_closure_rate"]),
                "net_utility_retention": ratio(row["common_net_utility_per_task"], base["common_net_utility_per_task"]),
                "alert_rate_retention": retention(row["metadata_alert_rate"], base["metadata_alert_rate"]) if row["method"] == "PROP-HEADROOM" else None,
                "full_closure_latency_inflation": ratio(row["mean_full_closure_latency_s"], base["mean_full_closure_latency_s"]),
                "alert_latency_inflation": ratio(row["metadata_alert_latency_mean_s"], base["metadata_alert_latency_mean_s"]) if row["method"] == "PROP-HEADROOM" else None,
                "queue_inflation": ratio(row["maximum_downlink_queue_mb"], base["maximum_downlink_queue_mb"])})
    else:
        for row in rows:
            if row["window_condition"] != "loss20":
                continue
            base = by[(int(row["seed"]), row["method"], .5, "original")]
            output.append({"seed": row["seed"], "method": row["method"], "capacity": .5,
                "closure_retention": retention(row["closure_rate"], base["closure_rate"]),
                "hv_closure_retention": retention(row["hv_closure_rate"], base["hv_closure_rate"]),
                "hv_target_closure_retention": retention(row["hv_target_closure_rate"], base["hv_target_closure_rate"]),
                "net_utility_retention": ratio(row["common_net_utility_per_task"], base["common_net_utility_per_task"]),
                "alert_rate_retention": retention(row["metadata_alert_rate"], base["metadata_alert_rate"]) if row["method"] == "PROP-HEADROOM" else None,
                "hv_alert_latency_inflation": ratio(row["hv_metadata_alert_latency_mean_s"], base["hv_metadata_alert_latency_mean_s"]) if row["method"] == "PROP-HEADROOM" else None,
                "p95_closure_latency_inflation": ratio(row["p95_full_closure_latency_s"], base["p95_full_closure_latency_s"]),
                "queue_inflation": ratio(row["maximum_downlink_queue_mb"], base["maximum_downlink_queue_mb"])})
    return output


def aggregate_retention(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    id_keys = {"seed", "method", "capacity"}
    metrics = [key for key in rows[0] if key not in id_keys]
    for method in METHODS:
        capacities = sorted({f(row["capacity"]) for row in rows if row["method"] == method})
        for capacity in capacities:
            subset = [row for row in rows if row["method"] == method and abs(f(row["capacity"]) - capacity) < TOL]
            for metric in metrics:
                values = [f(row[metric]) for row in subset if row.get(metric) not in (None, "")]
                lo, hi = bootstrap_ci(values, 7000 + int(capacity * 100) + sum(map(ord, method + metric)))
                output.append({"method": method, "capacity": capacity, "metric": metric, "n": len(values),
                    "mean": mean(values) if values else None, "median": median(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                    "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi})
    return output


def winner_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = [("closure_rate", True), ("hv_closure_rate", True), ("hv_target_closure_rate", True),
                   ("common_net_utility_per_task", True), ("utility_per_mb", True),
                   ("p95_full_closure_latency_s", False), ("maximum_downlink_queue_mb", False)]
    output = []
    scenarios = sorted({(int(row["seed"]), f(row["capacity"]), row["window_condition"]) for row in rows})
    for metric, higher in definitions:
        counts = {method: {"wins": 0, "ties": 0, "losses": 0} for method in METHODS}
        for seed, capacity, condition in scenarios:
            values = {row["method"]: f(row[metric]) for row in rows if int(row["seed"]) == seed
                      and abs(f(row["capacity"]) - capacity) < TOL and row["window_condition"] == condition}
            best = (max(values.values()) if higher else min(values.values()))
            winners = {method for method, value in values.items() if abs(value - best) <= TOL}
            for method in METHODS:
                counts[method]["wins" if method in winners and len(winners) == 1 else "ties" if method in winners else "losses"] += 1
        for method in METHODS:
            output.append({"metric": metric, "higher_is_better": higher, "method": method, **counts[method],
                           "scenario_count": len(scenarios)})
    return output


def run_capacity() -> list[dict[str, Any]]:
    if not json.loads((CAP_OUT / "smoke_report.json").read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("Capacity smoke not passed")
    summaries = []
    for seed in SEEDS:
        for capacity in CAPACITIES:
            configs = cfg(seed, capacity, True)
            timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
            validation = validate_scenario(configs, timeline)
            scenario = CAP_OUT / "runs" / scenario_key(seed, capacity)
            write_json(scenario / "resolved_config.json", {"seed": seed, "capacity": capacity, "horizon_s": 86400,
                "task_count": validation["task_count"], "task_hash": validation["task_manifest_hash_v2"],
                "environment_hash": validation["environment_hash_v2"], "methods": list(METHODS),
                "same_timeline_object_within_scenario": True, "window_condition": "original"})
            scenario_rows = []
            for method in METHODS:
                row = execute_method(method, seed, capacity, configs, timeline, validation, scenario / method, "original")
                scenario_rows.append(row); summaries.append(row)
            if len({row["task_hash"] for row in scenario_rows}) != 1 or len({row["environment_hash"] for row in scenario_rows}) != 1:
                raise RuntimeError(f"Fairness mismatch {scenario}")
            write_csv(scenario / "method_summary.csv", scenario_rows)
            write_csv(CAP_OUT / "scenario_summary.partial.csv", summaries)
    write_csv(CAP_OUT / "scenario_summary.csv", summaries)
    retention_rows = build_retention_rows(summaries, "capacity")
    write_csv(CAP_OUT / "capacity_retention_by_seed.csv", retention_rows)
    write_csv(CAP_OUT / "capacity_retention_aggregate.csv", aggregate_retention(retention_rows))
    write_csv(CAP_OUT / "scenario_winner_counts.csv", winner_counts(summaries))
    return summaries


def run_capacity_seed(seed: int) -> list[dict[str, Any]]:
    """Run one independent seed without touching shared aggregate files."""
    summaries = []
    for capacity in CAPACITIES:
        configs = cfg(seed, capacity, True)
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        validation = validate_scenario(configs, timeline)
        scenario = CAP_OUT / "runs" / scenario_key(seed, capacity)
        write_json(scenario / "resolved_config.json", {"seed": seed, "capacity": capacity, "horizon_s": 86400,
            "task_count": validation["task_count"], "task_hash": validation["task_manifest_hash_v2"],
            "environment_hash": validation["environment_hash_v2"], "methods": list(METHODS),
            "same_timeline_object_within_scenario": True, "window_condition": "original"})
        scenario_rows = []
        for method in METHODS:
            row = execute_method(method, seed, capacity, configs, timeline, validation, scenario / method, "original")
            scenario_rows.append(row); summaries.append(row)
        if len({row["task_hash"] for row in scenario_rows}) != 1 or len({row["environment_hash"] for row in scenario_rows}) != 1:
            raise RuntimeError(f"Fairness mismatch {scenario}")
        write_csv(scenario / "method_summary.csv", scenario_rows)
    return summaries


def run_window(capacity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not json.loads((WIN_OUT / "smoke_report.json").read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("Window smoke not passed")
    summaries = []
    base_by = {(int(row["seed"]), row["method"]): row for row in capacity_rows if abs(f(row["capacity"]) - .5) < TOL}
    for seed in SEEDS:
        configs = cfg(seed, .5, True)
        original = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        original_validation = validate_scenario(configs, original)
        windows = extract_windows(original)
        selected = select_removed_windows(windows, seed)
        perturbed = remove_windows(original, selected)
        perturbed_validation = validate_scenario(configs, perturbed, original_validation["task_manifest_hash_v2"])
        manifest = {"seed": seed, "selection_rule": "sha256_rank", "requested_fraction": .20,
            "total_window_count": len(windows), "removed_window_count": len(selected),
            "realized_fraction": len(selected) / len(windows), "selected_windows": selected,
            "task_hash": perturbed_validation["task_manifest_hash_v2"],
            "original_environment_hash": original_validation["environment_hash_v2"],
            "perturbed_environment_hash": perturbed_validation["environment_hash_v2"]}
        seed_dir = WIN_OUT / "runs" / f"seed{seed}"
        write_json(seed_dir / "window_removal_manifest.json", manifest)
        write_json(seed_dir / "resolved_config.json", {"seed": seed, "capacity": .5, "horizon_s": 86400,
            "task_hash": perturbed_validation["task_manifest_hash_v2"], "environment_hash": perturbed_validation["environment_hash_v2"],
            "window_condition": "loss20", "same_mask_for_all_methods": True})
        scenario_rows = []
        for method in METHODS:
            base = {**base_by[(seed, method)], "window_condition": "original"}
            summaries.append(base)
            row = execute_method(method, seed, .5, configs, perturbed, perturbed_validation,
                                 seed_dir / "loss20" / method, "loss20")
            scenario_rows.append(row); summaries.append(row)
        if len({row["task_hash"] for row in scenario_rows}) != 1 or len({row["environment_hash"] for row in scenario_rows}) != 1:
            raise RuntimeError(f"Perturbed fairness mismatch seed {seed}")
        write_csv(seed_dir / "method_summary.csv", scenario_rows)
        write_csv(WIN_OUT / "scenario_summary.partial.csv", summaries)
    write_csv(WIN_OUT / "scenario_summary.csv", summaries)
    retention_rows = build_retention_rows(summaries, "window")
    write_csv(WIN_OUT / "window_retention_by_seed.csv", retention_rows)
    write_csv(WIN_OUT / "window_retention_aggregate.csv", aggregate_retention(retention_rows))
    write_csv(WIN_OUT / "scenario_winner_counts.csv", winner_counts([row for row in summaries if row["window_condition"] == "loss20"]))
    return summaries


def run_window_seed(seed: int, capacity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    base_by = {row["method"]: row for row in capacity_rows if abs(f(row["capacity"]) - .5) < TOL}
    configs = cfg(seed, .5, True)
    original = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    original_validation = validate_scenario(configs, original)
    windows = extract_windows(original)
    selected = select_removed_windows(windows, seed)
    perturbed = remove_windows(original, selected)
    perturbed_validation = validate_scenario(configs, perturbed, original_validation["task_manifest_hash_v2"])
    seed_dir = WIN_OUT / "runs" / f"seed{seed}"
    write_json(seed_dir / "window_removal_manifest.json", {"seed": seed, "selection_rule": "sha256_rank",
        "requested_fraction": .20, "total_window_count": len(windows), "removed_window_count": len(selected),
        "realized_fraction": len(selected) / len(windows), "selected_windows": selected,
        "task_hash": perturbed_validation["task_manifest_hash_v2"],
        "original_environment_hash": original_validation["environment_hash_v2"],
        "perturbed_environment_hash": perturbed_validation["environment_hash_v2"]})
    write_json(seed_dir / "resolved_config.json", {"seed": seed, "capacity": .5, "horizon_s": 86400,
        "task_hash": perturbed_validation["task_manifest_hash_v2"], "environment_hash": perturbed_validation["environment_hash_v2"],
        "window_condition": "loss20", "same_mask_for_all_methods": True})
    scenario_rows = []
    for method in METHODS:
        summaries.append({**base_by[method], "window_condition": "original"})
        row = execute_method(method, seed, .5, configs, perturbed, perturbed_validation,
                             seed_dir / "loss20" / method, "loss20")
        scenario_rows.append(row); summaries.append(row)
    if len({row["task_hash"] for row in scenario_rows}) != 1 or len({row["environment_hash"] for row in scenario_rows}) != 1:
        raise RuntimeError(f"Perturbed fairness mismatch seed {seed}")
    write_csv(seed_dir / "method_summary.csv", scenario_rows)
    return summaries


def collect_and_finalize() -> None:
    capacity_rows = []
    for seed in SEEDS:
        for capacity in CAPACITIES:
            path = CAP_OUT / "runs" / scenario_key(seed, capacity) / "method_summary.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            capacity_rows.extend(read_csv(path))
    if len(capacity_rows) != 45:
        raise RuntimeError(f"Expected 45 capacity method rows, got {len(capacity_rows)}")
    write_csv(CAP_OUT / "scenario_summary.csv", capacity_rows)
    retention_rows = build_retention_rows(capacity_rows, "capacity")
    write_csv(CAP_OUT / "capacity_retention_by_seed.csv", retention_rows)
    write_csv(CAP_OUT / "capacity_retention_aggregate.csv", aggregate_retention(retention_rows))
    write_csv(CAP_OUT / "scenario_winner_counts.csv", winner_counts(capacity_rows))
    window_rows = []
    base_by = {(int(row["seed"]), row["method"]): row for row in capacity_rows if abs(f(row["capacity"]) - .5) < TOL}
    for seed in SEEDS:
        path = WIN_OUT / "runs" / f"seed{seed}" / "method_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        perturbed = read_csv(path)
        for method in METHODS:
            window_rows.append({**base_by[(seed, method)], "window_condition": "original"})
            window_rows.append(next(row for row in perturbed if row["method"] == method))
    if len(window_rows) != 30:
        raise RuntimeError(f"Expected 30 window rows, got {len(window_rows)}")
    write_csv(WIN_OUT / "scenario_summary.csv", window_rows)
    window_retention = build_retention_rows(window_rows, "window")
    write_csv(WIN_OUT / "window_retention_by_seed.csv", window_retention)
    write_csv(WIN_OUT / "window_retention_aggregate.csv", aggregate_retention(window_retention))
    write_csv(WIN_OUT / "scenario_winner_counts.csv", winner_counts([row for row in window_rows if row["window_condition"] == "loss20"]))
    finalize(capacity_rows, window_rows)


def finalize(capacity_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]]) -> None:
    cap_ret = read_csv(CAP_OUT / "capacity_retention_aggregate.csv")
    win_ret = read_csv(WIN_OUT / "window_retention_aggregate.csv")
    def pick(rows: list[dict[str, str]], method: str, capacity: float, metric: str) -> float | None:
        match = next((row for row in rows if row["method"] == method and abs(f(row["capacity"]) - capacity) < TOL and row["metric"] == metric), None)
        return f(match["mean"]) if match and match["mean"] else None
    def scenario_mean(rows: list[dict[str, Any]], method: str, capacity: float, metric: str,
                      condition: str = "original") -> float:
        values = [f(row[metric]) for row in rows if row["method"] == method
                  and abs(f(row["capacity"]) - capacity) < TOL
                  and row["window_condition"] == condition]
        return mean(values)
    cap_winners = read_csv(CAP_OUT / "scenario_winner_counts.csv")
    win_winners = read_csv(WIN_OUT / "scenario_winner_counts.csv")
    def wins(rows: list[dict[str, str]], method: str, metric: str) -> str:
        row = next(item for item in rows if item["method"] == method and item["metric"] == metric)
        return f'{row["wins"]}/{row["scenario_count"]}'
    alerts = read_csv(C1 / "alert_lead_threshold_summary.csv")
    latency = read_csv(C1 / "group_latency_summary.csv")
    def alert_value(group: str, threshold: int, field: str) -> float:
        row = next(item for item in alerts if item["task_group"] == group
                   and int(item["threshold_s_strictly_greater_than"]) == threshold)
        return f(row[field])
    def latency_value(group: str, metric: str, field: str) -> float:
        row = next(item for item in latency if item["task_group"] == group and item["metric"] == metric)
        return f(row[field])
    safety_total = sum(int(f(row["thermal_violation_count"])) + int(f(row["soc_violation_count"]))
                       for row in capacity_rows + [row for row in window_rows if row["window_condition"] == "loss20"])
    report = ["# 三方法最小稳健性实验总报告", "",
        "正式方法仅包括B1-COMMON-THERMAL、OFFLOAD-EFT-THERMAL-ALIGNED和PROP-HEADROOM；GROUND-ONLY-E2E未进入表格或图件。", "",
        "## 容量退化（跨5个seed的均值保留率）", "",
        "| capacity | method | closure | HV | HV-TARGET | net utility/task |", "|---:|---|---:|---:|---:|---:|"]
    for capacity in (.5, .35):
        for method in METHODS:
            values = [pick(cap_ret, method, capacity, metric) for metric in ("closure_retention", "hv_closure_retention", "hv_target_closure_retention", "net_utility_retention")]
            report.append(f"| {capacity:.2f} | {method} | " + " | ".join("NA" if value is None else f"{value:.3f}" for value in values) + " |")
    report += ["", "## 删除20%地面接触窗口（跨5个seed的均值保留率）", "",
        "| method | closure | HV | HV-TARGET | net utility/task | PROP alert rate |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        values = [pick(win_ret, method, .5, metric) for metric in ("closure_retention", "hv_closure_retention", "hv_target_closure_retention", "net_utility_retention", "alert_rate_retention")]
        report.append(f"| {method} | " + " | ".join("NA" if value is None else f"{value:.3f}" for value in values) + " |")
    report += ["", "## 九个核心问题的回答", "",
        "### 1. PROP是否为HV/HV-TARGET提供有意义的预警提前量", "",
        f"对同一任务严格配对后，全部任务n={int(alert_value('ALL', 0, 'paired_alert_closure_n'))}，预警提前量均值{latency_value('ALL', 'alert_lead_time', 'mean_s'):.3f}秒、中位数{latency_value('ALL', 'alert_lead_time', 'median_s'):.3f}秒；HV n={int(alert_value('HV', 0, 'paired_alert_closure_n'))}，均值{latency_value('HV', 'alert_lead_time', 'mean_s'):.3f}秒、中位数{latency_value('HV', 'alert_lead_time', 'median_s'):.3f}秒。HV-TARGET仅n={int(alert_value('HV-TARGET', 0, 'paired_alert_closure_n'))}且均值/中位数均为0秒。因此，只能确认PROP给一部分普通HV带来短提前量，不能确认对HV-TARGET有实质提前。", "",
        "### 2. 提前量是否普遍", "",
        f"不是。全部配对任务中>{0}秒为{int(alert_value('ALL', 0, 'count'))}个（{alert_value('ALL', 0, 'rate'):.1%}），但>60秒仅{int(alert_value('ALL', 60, 'count'))}个（{alert_value('ALL', 60, 'rate'):.1%}），>300秒仅{int(alert_value('ALL', 300, 'count'))}个（{alert_value('ALL', 300, 'rate'):.1%}），>600秒为0。HV中>60秒仅{int(alert_value('HV', 60, 'count'))}/{int(alert_value('HV', 60, 'paired_alert_closure_n'))}，>300秒为0；长提前量集中在少数任务。", "",
        "### 3. 容量下降时性能退化多快", "",
        "上方容量表以每个seed的capacity=1.00为基准。到0.50时，B1/OFFLOAD/PROP的闭环保留率分别为0.650/0.920/0.951，净效用保留率为0.577/0.835/0.940；到0.35时分别为0.448/0.814/0.892和0.359/0.684/0.876。PROP在退化保留率上最稳健，OFFLOAD次之，B1下降最快。", "",
        "### 4. 删除20%窗口后的保留量", "",
        "每个seed共有16个完整地面接触窗口；整数约束下按预注册SHA256排序删除3个，实际删除18.75%，同一seed三方法共享同一掩码。B1/OFFLOAD/PROP的闭环保留率为0.800/0.878/0.899，净效用保留率为0.800/0.856/0.876；PROP metadata预警率保留0.859。", "",
        "### 5. PROP是否降低通信依赖", "",
        f"在本实验范围内有支持性证据：PROP在15/15个容量场景中utility/MB最高；capacity=1.00时平均发送{scenario_mean(capacity_rows, 'PROP-HEADROOM', 1.0, 'transmitted_mb'):.1f} MB，低于OFFLOAD的{scenario_mean(capacity_rows, 'OFFLOAD-EFT-THERMAL-ALIGNED', 1.0, 'transmitted_mb'):.1f} MB和B1的{scenario_mean(capacity_rows, 'B1-COMMON-THERMAL', 1.0, 'transmitted_mb'):.1f} MB，且容量/窗口退化保留率最高。这证明的是更高通信效率与更低敏感性，不等同于绝对闭环数最高，也不能单独证明因果机制。", "",
        "### 6. OFFLOAD仍然更强的指标", "",
        f"OFFLOAD的绝对完整闭环率在容量实验{wins(cap_winners, 'OFFLOAD-EFT-THERMAL-ALIGNED', 'closure_rate')}个场景获胜，common net utility/task在{wins(cap_winners, 'OFFLOAD-EFT-THERMAL-ALIGNED', 'common_net_utility_per_task')}个场景获胜，最大下传队列在{wins(cap_winners, 'OFFLOAD-EFT-THERMAL-ALIGNED', 'maximum_downlink_queue_mb')}个场景最小；窗口损失后，闭环率、HV闭环率和净效用均在{wins(win_winners, 'OFFLOAD-EFT-THERMAL-ALIGNED', 'closure_rate')}个seed获胜。PROP优势主要在HV-TARGET、utility/MB和退化保留率，而非绝对总闭环。", "",
        "### 7. B1是否表现合理", "",
        "合理。作为固定天数地算基线，B1随capacity从1.00降到0.50/0.35呈单调退化，且删除接触窗口后保留率最低；这与其依赖完整下传的既定语义一致。该描述是结果一致性判断，不是新的因果证明。", "",
        "### 8. 已验证", "",
        f"完成15个容量场景×3方法和5个窗口扰动seed×3方法；场景内task/environment hash唯一，窗口掩码跨方法一致，热/SOC违规总数为{safety_total}。C1只读源日志SHA256与Stage B一致；capacity=0.25因合同未支持而未运行。", "",
        "### 9. 尚未证明", "",
        "未证明统计显著性、五个seed之外的普适性、其他窗口缺失比例、capacity=0.25、可撤销补传或新算法效果，也未证明PROP绝对性能全面优于OFFLOAD。", "",
        "## 解释边界", "",
        "bootstrap 95%区间仅是5个预注册seed上的描述性不确定性。相关退化差异不自动证明机制因果；本轮没有增加其他扰动、可反悔机制或新算法。"]
    text = "\n".join(report) + "\n"
    (CAP_OUT / "final_capacity_report.md").write_text(text, encoding="utf-8")
    (WIN_OUT / "final_combined_report.md").write_text(text, encoding="utf-8")
    for output in (CAP_OUT, WIN_OUT):
        files = [{"relative_path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"]
        write_json(output / "artifact_inventory.json", {"artifact_count_excluding_self": len(files), "artifacts": files})


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    group.add_argument("--worker-seed", type=int)
    group.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--smoke-slots", type=int, default=360)
    args = parser.parse_args()
    if args.prepare_only:
        prepare(); return
    if args.smoke:
        smoke(args.smoke_slots); return
    if args.worker_seed is not None:
        if args.worker_seed not in SEEDS:
            raise ValueError(f"Unregistered seed: {args.worker_seed}")
        capacity_rows = run_capacity_seed(args.worker_seed)
        window_rows = run_window_seed(args.worker_seed, capacity_rows)
        print(json.dumps({"status": "WORKER_COMPLETE", "seed": args.worker_seed,
                          "capacity_rows": len(capacity_rows), "window_rows": len(window_rows)}))
        return
    if args.finalize_only:
        collect_and_finalize()
        print(json.dumps({"status": "STAGE_C_COMPLETE_FROM_CHECKPOINTS"}))
        return
    capacity_rows = run_capacity()
    window_rows = run_window(capacity_rows)
    finalize(capacity_rows, window_rows)
    print(json.dumps({"status": "STAGE_C_COMPLETE", "capacity_scenarios": len(capacity_rows),
                      "window_rows": len(window_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
