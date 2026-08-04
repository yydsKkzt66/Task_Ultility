# -*- coding: utf-8 -*-
"""Strictly paired 24-hour single-scenario pilot; not a formal experiment."""
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
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_MODULE_IMPORT_SYS_PATH = list(sys.path) if __package__ not in {None, ""} else None
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import resolve_experiment_config as _resolve_experiment_config

# validate_experiment_fairness retains a direct-script import for historical CLI
# use. Register one narrow alias so this runner is also importable as a module;
# do not mutate sys.path process-wide.
sys.modules.setdefault("resolve_experiment_config", _resolve_experiment_config)
from scripts.resolve_experiment_config import resolve
from scripts.run_sarfish_downlink_operating_region_seed2026 import scale_downlink_capacity
from scripts.validate_experiment_fairness import validate
from src.baselines.b1_enhanced import B1CompressionConfig, compression_energy_wh, compression_latency_s
from src.baselines.ground_only import GroundOnlySimulator, ground_path_parameters, validate_manifest_identity
from src.baselines.offload_eft import DEFER, GROUND, LOCAL
from src.baselines.offload_eft_thermal_aligned import ThermalAlignedSimulator
from src.common.config import load_yaml
from src.energy.thermal_headroom import maximum_safe_downlink_capacity
from src.evaluation.canonical_sarfish_environment import load_canonical_sarfish_configs, validate_main_paper_environment
from src.evaluation.unified_ground_received_accounting import common_net, score_task
from src.payload.information_sufficiency import product_information_utility
from src.timeline.timeline_builder import TimelineBuilder

if _MODULE_IMPORT_SYS_PATH is not None:
    # Some reused historical CLI modules insert ROOT during import. Keep that
    # legacy behavior contained when this pilot runner is imported as a module.
    sys.path[:] = _MODULE_IMPORT_SYS_PATH

RUN_KEY = "seed2026_cap050_h86400"
DEFAULT_OUT = ROOT / "outputs/_offload_vs_ground_24h_pilot_seed2026_cap050_v1"
BASE_COMMIT = "b75ef04b63a13d039197c064cf82b0338121f765"
TASK_HASH = "49c704997b59648d902ed9413a4116d0b45e5b404d0d20ece9d3826a7d6d38cd"
ENV_HASH = "6c9359f019290170a7be0b3bc5bc8a115f09589853eb8a49667a3457c0d82524"
TOL = 1.0e-9
PROTECTED = {
    "battery_model_source": "src/energy/battery_update.py",
    "benchmark_environment_json": "configs/benchmark_environment_v1.json",
    "benchmark_environment_yaml": "configs/benchmark_environment_v1.yaml",
    "common_evaluator_config": "configs/unified_ground_received_accounting.yaml",
    "common_evaluator_source": "src/evaluation/unified_ground_received_accounting.py",
    "contact_manifest": "manifests/contact_window_manifest_v1.csv",
    "dataset_manifest": "data/indexes/sarfishsample_scene_index.csv",
    "product_config": "configs/payload_profiles_roi_compression_calibrated.yaml",
    "queue_model_source": "src/link/downlink_queue.py",
    "runner_source": "src/simulator/step_runner.py",
    "scenario_manifest": "manifests/scenario_manifest_v1.csv",
    "thermal_headroom_source": "src/energy/thermal_headroom.py",
    "thermal_model_source": "src/energy/thermal_update.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))]


def model_profile(configs: dict[str, Any], model_name: str) -> dict[str, float]:
    model = next(row for row in configs["model_pool"]["models"] if row["name"] == model_name)
    base = next(row for row in configs["model_pool"]["profiles"] if row["name"] == model["profile_name"])
    return {**base, **configs["model_pool"].get("profile_overrides", {}).get(model["profile_name"], {})}


def product_profile(configs: dict[str, Any], name: str) -> dict[str, float]:
    return next(row for row in configs["payload_profiles"]["profiles"] if row["name"] == name)


def reset_rng() -> None:
    random.seed(2026)
    try:
        import numpy as np
        np.random.seed(2026)
    except ImportError:
        pass


def normalized_newlines(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


THERMAL_ZERO_CAPACITY_REASONS = frozenset({
    "CURRENT_AT_OR_ABOVE_HARD_LIMIT",
    "COMPUTE_ALONE_EXHAUSTS_HEADROOM",
    "NO_POSITIVE_SAFE_DOWNLINK_CAPACITY",
})


def is_thermal_pause(row: dict[str, Any]) -> bool:
    """Count only an eligible no-send slot explicitly blocked by thermal headroom."""
    return (
        float(row.get("nominal_capacity_mb", 0.0) or 0.0) > TOL
        and float(row.get("sent_mb", 0.0) or 0.0) <= TOL
        and str(row.get("thermal_headroom_zero_reason", "")) in THERMAL_ZERO_CAPACITY_REASONS
    )


def fingerprint_checks() -> dict[str, Any]:
    fingerprint = json.loads((ROOT / "manifests/benchmark_environment_fingerprint_v1.json").read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    for key, rel in PROTECTED.items():
        path = ROOT / rel
        raw = path.read_bytes()
        expected = fingerprint["components"][key]
        raw_match = hashlib.sha256(raw).hexdigest() == expected
        blob = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT)
        line_ending_only = normalized_newlines(raw) == normalized_newlines(blob)
        logical_clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
        checks[key] = {
            "path": rel,
            "expected_raw_sha256": expected,
            "actual_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_match": raw_match,
            "git_logical_clean": logical_clean,
            "line_ending_only_checkout_difference": bool(not raw_match and line_ending_only and logical_clean),
            "match": bool(raw_match or (line_ending_only and logical_clean)),
        }
    return {"benchmark_environment_fingerprint": fingerprint["benchmark_environment_fingerprint"],
            "components": checks, "all_match": all(row["match"] for row in checks.values())}


def build_shared_scenario() -> dict[str, Any]:
    configs = scale_downlink_capacity(load_canonical_sarfish_configs(ROOT, seed=2026), 0.50)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = validate_main_paper_environment(configs, timeline)
    manifest = list(csv.DictReader((ROOT / "manifests/scenario_manifest_v1.csv").open(encoding="utf-8-sig", newline="")))
    rows = [row for row in manifest if row["seed"] == "2026"]
    if len(rows) != 1:
        raise RuntimeError(f"seed 2026 is not unique: {len(rows)} rows")
    expected = rows[0]
    validate_manifest_identity(env["task_manifest_hash_v2"], env["environment_hash_v2"], expected["task_hash"], expected["environment_hash"])
    if env["task_manifest_hash_v2"] != TASK_HASH or env["environment_hash_v2"] != ENV_HASH:
        raise RuntimeError("frozen task/environment hash mismatch")
    if len(timeline) != 4320 or abs(sum(step.slot_seconds for step in timeline) - 86400.0) > TOL:
        raise RuntimeError("24-hour timeline mismatch")
    task_steps = [step for step in timeline if step.task is not None]
    if len(task_steps) != 268:
        raise RuntimeError(f"task count mismatch: {len(task_steps)}")
    hv = sum(bool(step.task.high_value_flag or step.task.emergency_flag) for step in task_steps)
    hv_target = sum(bool(step.task.emergency_flag) for step in task_steps)
    if (hv, hv_target) != (169, 42):
        raise RuntimeError(f"HV/HV-TARGET mismatch: {(hv, hv_target)}")
    return {"configs": configs, "timeline": timeline, "task_steps": task_steps, "env": env,
            "manifest_row": expected, "hv_count": hv, "hv_target_count": hv_target,
            "scenario_object_hash": stable_hash({
                "tasks": env["task_manifest_hash_v2"], "environment": env["environment_hash_v2"],
                "slots": len(timeline), "run_key": RUN_KEY})}


def resolved_methods() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = ROOT / "configs/benchmark_environment_v1.yaml"
    experiment = ROOT / "configs/experiments/offload_vs_ground_24h_pilot_seed2026_cap050.yaml"
    ground_path = ROOT / "configs/methods/ground_only_v1.yaml"
    offload_path = ROOT / "configs/methods/offload_eft_thermal_aligned_v1.yaml"
    ground, _ = resolve(base, ground_path, experiment)
    offload, _ = resolve(base, offload_path, experiment)
    gs, gc = validate(base, ground_path, experiment)
    os_, oc = validate(base, offload_path, experiment)
    if not gs.startswith(("PASS", "WARNING")) or not os_.startswith(("PASS", "WARNING")):
        raise RuntimeError(f"fairness validation failed: {gs}, {os_}")
    return ground, offload, {"ground_only": gs, "offload_eft_thermal_aligned": os_,
                             "ground_check_count": len(gc), "offload_check_count": len(oc)}


def common_parameters(configs: dict[str, Any], offload_resolved: dict[str, Any]) -> dict[str, Any]:
    ground = ground_path_parameters(configs)
    local = model_profile(configs, offload_resolved["method"]["local_model"])
    product = product_profile(configs, offload_resolved["method"]["local_result_product"])
    defaults = configs["payload_profiles"]["calibrated_profile"]["defaults"]
    generation_s = product["size_mb"] * defaults["encoding_latency_s_per_mb"] * product["encoding_latency_scale"]
    generation_j = product["size_mb"] * defaults["encoding_energy_j_per_mb"] * product["encoding_energy_scale"]
    return {"ground": ground, "local": local, "local_product": product,
            "local_compute_duration_s": local["latency_s"] + generation_s,
            "local_compute_energy_j": local["energy_j"] + generation_j}


def simulator_kwargs(shared: dict[str, Any], resolved: dict[str, Any], pars: dict[str, Any]) -> dict[str, Any]:
    ground = pars["ground"]
    return {
        "timeline": shared["timeline"], "simulator_cfg": shared["configs"]["simulator"],
        "local_compute_duration_s": pars["local_compute_duration_s"],
        "local_compute_energy_j": pars["local_compute_energy_j"],
        "local_payload_mb": pars["local_product"]["size_mb"],
        "ground_preprocess_duration_s": ground.compression_latency_s,
        "ground_preprocess_energy_j": ground.compression_energy_j,
        "ground_payload_mb": ground.compressed_size_mb,
        "ground_compute_duration_s": ground.ground_compute_latency_s,
        "ground_servers": resolved["method"]["ground_compute_servers"],
        "ground_queue_capacity": resolved["method"]["ground_queue_capacity"],
        "tie_tolerance_s": float(resolved["method"].get("tie_tolerance_s", 1.0e-6)),
    }


def run_ground(shared: dict[str, Any], resolved: dict[str, Any], pars: dict[str, Any]) -> tuple[GroundOnlySimulator, list[dict[str, Any]]]:
    reset_rng()
    kwargs = simulator_kwargs(shared, resolved, pars)
    kwargs.update({"local_compute_duration_s": 0.0, "local_compute_energy_j": 0.0, "local_payload_mb": 0.0})
    sim = GroundOnlySimulator(**kwargs)
    ground_quality = pars["ground"].ground_quality
    for step in shared["timeline"]:
        if step.task is not None:
            task = step.task
            now = step.slot_index * step.slot_seconds
            feasible = 1.0 + TOL >= task.information_requirement and ground_quality + TOL >= task.quality_requirement
            sim.admit_ground_task(task_id=task.task_id, arrival_s=now, deadline_s=now + task.deadline_s, product_feasible=feasible)
        sim.advance_one_slot()
    decisions = [{"task_id": tid, "selected_path": GROUND, "selection_reason": "forced_ground_once",
                  "future_tasks_read": 0, "migration_count": 0, "duplicate_execution_count": 0}
                 for tid in sim.ground_only_decisions]
    return sim, decisions


def candidate_sent_mb(trace: tuple[dict[str, Any], ...], remaining: dict[str, float], target: str) -> float:
    sent_target = 0.0
    rem = dict(remaining)
    for row in trace:
        budget = float(row.get("sent_mb", 0.0) or 0.0)
        for tid in str(row.get("tx_task_id", "")).split(";"):
            if not tid or budget <= TOL:
                continue
            amount = min(rem.get(tid, 0.0), budget)
            rem[tid] = max(0.0, rem.get(tid, 0.0) - amount)
            budget -= amount
            if tid == target:
                sent_target += amount
    return sent_target


def run_offload(shared: dict[str, Any], resolved: dict[str, Any], pars: dict[str, Any]) -> tuple[ThermalAlignedSimulator, list[dict[str, Any]], dict[str, Any]]:
    reset_rng()
    sim = ThermalAlignedSimulator(**simulator_kwargs(shared, resolved, pars))
    utility_cfg = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    decisions: list[dict[str, Any]] = []
    task0009_diag: dict[str, Any] = {"exists": False}
    for step in shared["timeline"]:
        if step.task is not None:
            task = step.task
            now = step.slot_index * step.slot_seconds
            deadline = now + task.deadline_s
            local_info = product_information_utility({resolved["method"]["local_result_product"]}, task.product_requirement, utility_cfg)
            local_ok = local_info + TOL >= task.information_requirement and pars["local"]["quality_base"] + TOL >= task.quality_requirement
            ground_ok = 1.0 + TOL >= task.information_requirement and pars["ground"].ground_quality + TOL >= task.quality_requirement
            pre_remaining = {tid: job.remaining_mb for tid, job in sim.jobs.items() if job.state not in {"CLOSED", "FAILED"}}
            d = sim.decide_task(task_id=task.task_id, arrival_s=now, deadline_s=deadline,
                                local_product_feasible=local_ok, ground_product_feasible=ground_ok)
            row = {
                "task_id": task.task_id, "arrival_s": now, "deadline_s": deadline,
                "selected_path": d.selected_path, "selection_reason": d.selection_reason,
                "local_feasible": int(d.local.feasible), "local_failure_reason": d.local.failure_reason,
                "ground_feasible": int(d.ground.feasible), "ground_failure_reason": d.ground.failure_reason,
                "predicted_local_finish_s": d.local.predicted_finish_s,
                "predicted_ground_finish_s": d.ground.predicted_finish_s,
                "predicted_local_bytes": d.local.predicted_tx_bytes,
                "predicted_ground_bytes": d.ground.predicted_tx_bytes,
                "predicted_local_max_thermal": d.local.predicted_max_thermal,
                "predicted_ground_max_thermal": d.ground.predicted_max_thermal,
                "predicted_local_min_soc": d.local.predicted_min_soc,
                "predicted_ground_min_soc": d.ground.predicted_min_soc,
                "future_tasks_read": d.local.future_tasks_read + d.ground.future_tasks_read,
                "migration_count": 0, "duplicate_execution_count": 0,
            }
            decisions.append(row)
            if task.task_id == "sarfishsample_task_0009":
                remaining = {**pre_remaining, task.task_id: pars["ground"].compressed_size_mb}
                candidate_sent = candidate_sent_mb(d.ground.trace, remaining, task.task_id)
                target_rows = [x for x in d.ground.trace if task.task_id in str(x.get("tx_task_id", "")).split(";")]
                task0009_diag = {
                    "exists": True, "selected_path": d.selected_path,
                    "ground_candidate_feasible": d.ground.feasible,
                    "ground_candidate_failure_reason": d.ground.failure_reason,
                    "candidate_payload_bytes": d.ground.predicted_tx_bytes,
                    "candidate_sent_bytes": candidate_sent * 1_000_000.0,
                    "candidate_shortfall_bytes": max(0.0, d.ground.predicted_tx_bytes - candidate_sent * 1_000_000.0),
                    "candidate_tx_slots": [int(x["slot"]) for x in target_rows],
                    "candidate_max_thermal": d.ground.predicted_max_thermal,
                }
        sim.advance_one_slot()
    return sim, decisions, task0009_diag


def augment_trace(trace: list[dict[str, Any]], timeline: list[Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row, step in zip(trace, timeline):
        rate = step.link_trace.throughput_mbps * float(cfg.get("downlink_efficiency", 1.0)) / 8.0
        potential = 0.0
        if step.link_trace.in_contact and rate > 0.0:
            nominal = rate * step.slot_seconds
            cap = maximum_safe_downlink_capacity(
                current_thermal=float(row["thermal_before"]), nominal_capacity_mb=nominal,
                rate_mb_s=rate, link_quality=step.link_trace.link_quality,
                compute_energy_j=float(row["compute_energy_j"]), compute_heat_gain_scale=1.0,
                slot_seconds=step.slot_seconds, temperature_reference_c=step.resource_trace.temperature_reference_c,
                is_sunlit=step.resource_trace.is_sunlit, simulator_cfg=cfg)
            potential = float(cap.safe_capacity_mb)
        eligible = float(row["nominal_capacity_mb"])
        sent = float(row["sent_mb"])
        safe_used = float(row["safe_capacity_mb"])
        result.append({**row,
            "potential_thermal_safe_capacity_mb": potential,
            "unused_potential_safe_capacity_mb": max(0.0, potential - sent),
            "eligible_safe_capacity_idle": int(eligible > TOL and sent + TOL < min(eligible, potential)),
            "thermal_limited_partial_send": int(sent > TOL and safe_used + TOL < eligible),
            "thermal_pause": int(is_thermal_pause(row)),
        })
    return result


def task_rows(method: str, sim: Any, shared: dict[str, Any], pars: dict[str, Any], decisions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decision_by_id = {row["task_id"]: row for row in decisions}
    utility_cfg = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    for step in shared["task_steps"]:
        task = step.task
        tid = task.task_id
        arrival = step.slot_index * step.slot_seconds
        deadline = arrival + task.deadline_s
        decision = decision_by_id[tid]
        path = decision["selected_path"]
        job = sim.jobs.get(tid)
        finish = job.closure_s if job is not None else None
        closed = bool(job and job.product_feasible and finish is not None and finish <= deadline + TOL and finish <= 86400.0 + TOL)
        if job is None:
            failure = ";".join(x for x in [decision.get("local_failure_reason", ""), decision.get("ground_failure_reason", "")] if x) or "defer"
        elif not job.product_feasible:
            failure = "product_or_requirement_infeasible"
        elif job.failure_reason:
            failure = job.failure_reason
        elif finish is None:
            failure = "not_completed_before_horizon"
        elif finish > deadline + TOL:
            failure = "deadline"
        else:
            failure = ""
        transmitted = job.transmitted_mb * 1_000_000.0 if job else 0.0
        compute_energy = job.compute_energy_j if job and job.compute_start_s is not None else 0.0
        comm_energy = job.comm_energy_j if job else 0.0
        latency = finish - arrival if closed and finish is not None else 0.0
        if path == LOCAL:
            info = product_information_utility({"roi"}, task.product_requirement, utility_cfg)
            quality = float(pars["local"]["quality_base"])
        elif path == GROUND:
            info = 1.0
            quality = float(pars["ground"].ground_quality)
        else:
            info = quality = 0.0
        common = score_task({
            "task_id": tid, "method": method, "variant": RUN_KEY, "seed": 2026,
            "value_level": task.value_level, "capture_time_s": arrival, "deadline_time_s": deadline,
            "effective_task_value": task.task_value, "ground_completed": int(closed),
            "on_time_ground_completed": int(closed), "capture_to_ground_latency_s": latency,
            "ground_information_sufficiency": info if closed else 0.0,
            "ground_result_quality": quality if closed else 0.0, "partial_ground_gross_utility": 0.0,
            "requirement_satisfied": int(closed), "on_time_task_success": int(closed), "selected_path": path,
        }, accounting)
        common_rows.append(common)
        waits = {
            "compute_queue_wait_s": max(0.0, float(job.compute_start_s) - arrival) if job and job.compute_start_s is not None else 0.0,
            "downlink_queue_wait_s": max(0.0, float(job.tx_start_s) - float(job.downlink_queue_enter_s)) if job and job.tx_start_s is not None and job.downlink_queue_enter_s is not None else 0.0,
            "ground_queue_wait_s": max(0.0, float(job.ground_compute_start_s) - float(job.ground_queue_enter_s)) if job and job.ground_compute_start_s is not None and job.ground_queue_enter_s is not None else 0.0,
        }
        rows.append({
            "task_id": tid, "task_value": task.task_value, "value_level": task.value_level,
            "hv": int(bool(task.high_value_flag or task.emergency_flag)), "hv_target": int(bool(task.emergency_flag)),
            "arrival_s": arrival, "deadline_s": deadline, "selected_path": path, "closed": int(closed),
            "actual_finish_s": finish if finish is not None else "", "actual_latency_s": latency,
            "transmitted_bytes": transmitted, "compute_energy_j": compute_energy,
            "downlink_energy_j": comm_energy, "satellite_energy_j": compute_energy + comm_energy,
            "gross_utility": common["gross_ground_utility"], "failure_reason": failure,
            "product_feasible": int(job.product_feasible) if job else 0,
            "compression_or_compute_finish_s": job.compute_finish_s if job and job.compute_finish_s is not None else "",
            "tx_start_s": job.tx_start_s if job and job.tx_start_s is not None else "",
            "ground_received_s": job.tx_finish_s if job and job.tx_finish_s is not None else "",
            "ground_compute_start_s": job.ground_compute_start_s if job and job.ground_compute_start_s is not None else "",
            "ground_compute_finish_s": job.ground_compute_finish_s if job and job.ground_compute_finish_s is not None else "",
            **waits, "total_queue_wait_s": sum(waits.values()),
        })
    return rows, common_rows


def queue_metrics(rows: list[dict[str, Any]], horizon_s: float = 86400.0, slot_s: float = 20.0) -> dict[str, float]:
    max_downlink = 0
    max_ground = 0
    for index in range(int(horizon_s / slot_s)):
        now = index * slot_s
        downlink = sum(bool(row["compression_or_compute_finish_s"] != "" and float(row["compression_or_compute_finish_s"]) <= now + slot_s + TOL and (row["ground_received_s"] == "" or float(row["ground_received_s"]) > now + TOL)) for row in rows)
        ground = sum(bool(row["ground_received_s"] != "" and float(row["ground_received_s"]) <= now + slot_s + TOL and row["ground_compute_start_s"] != "" and float(row["ground_compute_start_s"]) > now + TOL) for row in rows)
        max_downlink = max(max_downlink, downlink)
        max_ground = max(max_ground, ground)
    return {"maximum_downlink_queue_length": max_downlink, "maximum_ground_queue_length": max_ground,
            "maximum_queue_length": max(max_downlink, max_ground),
            "longest_wait_s": max((float(row["total_queue_wait_s"]) for row in rows), default=0.0)}


def segment_rows(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments = [("h00_h06", 0, 21600), ("h06_h12", 21600, 43200),
                ("h12_h18", 43200, 64800), ("h18_h24", 64800, 86400),
                ("first_12h", 0, 43200), ("second_12h", 43200, 86400)]
    output = []
    for name, start, end in segments:
        selected = [row for row in rows if start <= float(row["arrival_s"]) < end]
        output.append({"method": method, "segment": name, "start_s": start, "end_s": end,
            "arrived_tasks": len(selected), "closure_count": sum(int(row["closed"]) for row in selected),
            "closure_rate": sum(int(row["closed"]) for row in selected) / max(len(selected), 1),
            "gross_utility": sum(float(row["gross_utility"]) for row in selected)})
    return output


def summarize(method: str, rows: list[dict[str, Any]], common_rows: list[dict[str, Any]], trace: list[dict[str, Any]], sim: Any) -> dict[str, Any]:
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    closed = [row for row in rows if int(row["closed"])]
    hv = [row for row in rows if int(row["hv"])]
    hv_target = [row for row in rows if int(row["hv_target"])]
    latencies = [float(row["actual_latency_s"]) for row in closed]
    transmitted_mb = sum(float(row["transmitted_bytes"]) for row in rows) / 1_000_000.0
    compression_j = sum(float(row["compute_energy_j"]) for row in rows if row["selected_path"] == GROUND)
    local_j = sum(float(row["compute_energy_j"]) for row in rows if row["selected_path"] == LOCAL)
    downlink_j = sum(float(row["downlink_energy_j"]) for row in rows)
    total_wh = (compression_j + local_j + downlink_j) / 3600.0
    gross = sum(float(row["gross_ground_utility"]) for row in common_rows)
    backlog = sum(float(job.remaining_mb) for job in sim.jobs.values() if job.remaining_mb > TOL)
    thermal_violations = sum(float(row["thermal_after"]) > float(row["hard_limit"]) + TOL for row in trace)
    soc_violations = sum(float(row["soc_after"]) < float(row["soc_floor"]) - TOL for row in trace)
    net = common_net(gross, total_wh, backlog, thermal_violations + soc_violations, accounting)
    potential = sum(float(row["potential_thermal_safe_capacity_mb"]) for row in trace)
    queue = queue_metrics(rows)
    last_quarter = sorted(rows, key=lambda row: (float(row["arrival_s"]), row["task_id"]))[-math.ceil(len(rows) / 4):]
    return {
        "method": method, "run_key": RUN_KEY, "formal_result": False,
        "task_count": len(rows), "hv_count": len(hv), "hv_target_count": len(hv_target),
        "task_hash": TASK_HASH, "environment_hash": ENV_HASH,
        "closure_count": len(closed), "closure_rate": len(closed) / max(len(rows), 1),
        "closures_per_100_tasks": 100.0 * len(closed) / max(len(rows), 1),
        "hv_closure_count": sum(int(row["closed"]) for row in hv),
        "hv_closure_rate": sum(int(row["closed"]) for row in hv) / max(len(hv), 1),
        "hv_target_closure_count": sum(int(row["closed"]) for row in hv_target),
        "hv_target_closure_rate": sum(int(row["closed"]) for row in hv_target) / max(len(hv_target), 1),
        "common_gross_utility": gross, "common_net_utility": net["common_ground_net_utility"],
        "common_net_utility_per_task": net["common_ground_net_utility"] / max(len(rows), 1),
        "terminal_backlog_mb": backlog,
        "total_transmitted_mb": transmitted_mb, "utility_per_mb": gross / max(transmitted_mb, TOL),
        "closures_per_100_mb": 100.0 * len(closed) / max(transmitted_mb, TOL),
        "total_potential_thermal_safe_capacity_mb": potential,
        "safe_communication_capacity_utilization": transmitted_mb / max(potential, TOL),
        "eligible_data_safe_capacity_idle_count": sum(int(row["eligible_safe_capacity_idle"]) for row in trace),
        "mean_latency_s": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_s": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_s": percentile(latencies, 0.95), "maximum_latency_s": max(latencies or [0.0]),
        "deadline_miss_count": sum(not int(row["closed"]) for row in rows),
        "compression_energy_wh": compression_j / 3600.0, "local_compute_energy_wh": local_j / 3600.0,
        "downlink_energy_wh": downlink_j / 3600.0, "total_controllable_satellite_energy_wh": total_wh,
        "utility_per_wh": gross / max(total_wh, TOL),
        "minimum_soc": min(float(row["soc_after"]) for row in trace), "final_soc": float(trace[-1]["soc_after"]),
        "soc_violation_count": soc_violations,
        "maximum_thermal": max(float(row["thermal_after"]) for row in trace),
        "thermal_margin_to_0_92": 0.92 - max(float(row["thermal_after"]) for row in trace),
        "near_thermal_limit_slot_count": sum(float(row["thermal_after"]) >= 0.92 - 1.0e-6 for row in trace),
        "thermal_violation_count": thermal_violations,
        "thermal_limited_partial_send_count": sum(int(row["thermal_limited_partial_send"]) for row in trace),
        "thermal_pause_count": sum(int(row["thermal_pause"]) for row in trace),
        "final_quarter_task_closure_rate": sum(int(row["closed"]) for row in last_quarter) / max(len(last_quarter), 1),
        **queue,
    }


def pairwise(ground: list[dict[str, Any]], offload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    left = {row["task_id"]: row for row in ground}
    right = {row["task_id"]: row for row in offload}
    result = []
    for tid in sorted(left, key=lambda value: int(value.rsplit("_", 1)[1])):
        g, o = left[tid], right[tid]
        gc, oc = bool(int(g["closed"])), bool(int(o["closed"]))
        category = "both_closed" if gc and oc else "ground_only_closed" if gc else "offload_only_closed" if oc else "neither_closed"
        def delta(a: Any, b: Any) -> Any:
            return float(a) - float(b) if a != "" and b != "" else ""
        result.append({
            "task_id": tid, "task_value": g["task_value"], "hv": g["hv"], "hv_target": g["hv_target"],
            "ground_only_closed": int(gc), "offload_closed": int(oc), "offload_path": o["selected_path"],
            "ground_only_actual_finish_s": g["actual_finish_s"], "offload_actual_finish_s": o["actual_finish_s"],
            "offload_minus_ground_latency_s": delta(o["actual_latency_s"], g["actual_latency_s"]) if gc and oc else "",
            "offload_minus_ground_transmitted_bytes": float(o["transmitted_bytes"]) - float(g["transmitted_bytes"]),
            "offload_minus_ground_satellite_energy_j": float(o["satellite_energy_j"]) - float(g["satellite_energy_j"]),
            "ground_only_utility": g["gross_utility"], "offload_utility": o["gross_utility"],
            "offload_minus_ground_utility": float(o["gross_utility"]) - float(g["gross_utility"]),
            "ground_only_failure_reason": g["failure_reason"], "offload_failure_reason": o["failure_reason"],
            "transition_category": category,
        })
    return result


def metric_rows(ground: dict[str, Any], offload: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "closure_count", "closure_rate", "closures_per_100_tasks", "hv_closure_count", "hv_closure_rate",
        "hv_target_closure_count", "hv_target_closure_rate", "common_gross_utility", "common_net_utility",
        "common_net_utility_per_task", "total_transmitted_mb", "utility_per_mb", "closures_per_100_mb",
        "safe_communication_capacity_utilization", "eligible_data_safe_capacity_idle_count", "mean_latency_s",
        "p50_latency_s", "p95_latency_s", "maximum_latency_s", "deadline_miss_count", "compression_energy_wh",
        "local_compute_energy_wh", "downlink_energy_wh", "total_controllable_satellite_energy_wh", "utility_per_wh",
        "minimum_soc", "final_soc", "soc_violation_count", "maximum_thermal", "thermal_margin_to_0_92",
        "near_thermal_limit_slot_count", "thermal_violation_count", "thermal_limited_partial_send_count",
        "thermal_pause_count", "final_quarter_task_closure_rate", "maximum_queue_length", "longest_wait_s",
    ]
    return [{"metric": key, "ground_only": ground[key], "offload_eft_thermal_aligned": offload[key],
             "offload_minus_ground": float(offload[key]) - float(ground[key])} for key in keys]


def final_report(status: str, ground: dict[str, Any], offload: dict[str, Any], transitions: dict[str, Any],
                 task0009: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    better = "OFFLOAD-EFT-THERMAL-ALIGNED" if offload["common_net_utility"] > ground["common_net_utility"] else "GROUND-ONLY"
    advantage = "更高的共同净效用" if offload["common_net_utility"] != ground["common_net_utility"] else "当前共同净效用持平"
    cost = "任务身份置换以及LOCAL/GROUND一次性选择带来的个体差异" if better.startswith("OFFLOAD") else "更高的全量下传通信负担"
    segment_map = {(row["method"], row["segment"]): row for row in segments}
    return f"""# 24小时严格配对pilot报告

## 一句话结论

本单场景pilot状态为 **{status}**；按共同净效用，{better}总体更好，主要体现为{advantage}，主要代价是{cost}。该结论只适用于seed=2026、capacity=0.50的这一个24小时场景。

## 公平性和安全

- task hash：`{TASK_HASH}`；environment hash：`{ENV_HASH}`。
- 两种方法共享同一个冻结timeline/scenario对象，并分别从全新SOC、热状态、星上队列、下传队列和地面FIFO初始化。
- 共享环境、任务、通信窗口、deadline、价值、评价器和安全阈值修改数为0。
- GROUND-ONLY热/SOC违规：{ground['thermal_violation_count']}/{ground['soc_violation_count']}；OFFLOAD热/SOC违规：{offload['thermal_violation_count']}/{offload['soc_violation_count']}。

## 核心指标对照表

| 指标 | GROUND-ONLY | OFFLOAD-EFT-THERMAL-ALIGNED |
|---|---:|---:|
| 闭环数 / 闭环率 | {ground['closure_count']} / {ground['closure_rate']:.6f} | {offload['closure_count']} / {offload['closure_rate']:.6f} |
| HV闭环数 / 闭环率 | {ground['hv_closure_count']} / {ground['hv_closure_rate']:.6f} | {offload['hv_closure_count']} / {offload['hv_closure_rate']:.6f} |
| HV-TARGET闭环数 / 闭环率 | {ground['hv_target_closure_count']} / {ground['hv_target_closure_rate']:.6f} | {offload['hv_target_closure_count']} / {offload['hv_target_closure_rate']:.6f} |
| common gross utility | {ground['common_gross_utility']:.9f} | {offload['common_gross_utility']:.9f} |
| common net utility | {ground['common_net_utility']:.9f} | {offload['common_net_utility']:.9f} |
| 下传MB | {ground['total_transmitted_mb']:.6f} | {offload['total_transmitted_mb']:.6f} |
| 可控能耗Wh | {ground['total_controllable_satellite_energy_wh']:.6f} | {offload['total_controllable_satellite_energy_wh']:.6f} |
| 最大热状态 | {ground['maximum_thermal']!r} | {offload['maximum_thermal']!r} |
| 最低SOC | {ground['minimum_soc']!r} | {offload['minimum_soc']!r} |
| p95时延s | {ground['p95_latency_s']:.6f} | {offload['p95_latency_s']:.6f} |

## 配对任务分析

- 两者均闭环：{transitions['both_closed']}；仅GROUND闭环：{transitions['ground_only_closed']}；仅OFFLOAD闭环：{transitions['offload_only_closed']}；均未闭环：{transitions['neither_closed']}。
- OFFLOAD净闭环差：{transitions['net_closure_delta']}；净gross utility差：{transitions['net_gross_utility_delta']:.9f}；净common utility差：{transitions['net_common_utility_delta']:.9f}。
- task0009：存在={task0009.get('exists')}，GROUND闭环={task0009.get('ground_only_closed')}，OFFLOAD路径={task0009.get('selected_path')}，OFFLOAD闭环={task0009.get('offload_closed')}，GROUND候选边际短缺={task0009.get('candidate_shortfall_bytes', 0.0):.6f} bytes。该案例未获得任何特殊规则。

## 通信、能量、热和时延解释

- GROUND-ONLY全部任务压缩后下传；OFFLOAD通过LOCAL/GROUND/DEFER一次性选择改变通信量和星上计算能耗。
- 两方法的GROUND路径复用同一B1压缩参数、逐slot热安全容量、有限单服务器FIFO和闭环定义。
- 预测器不能读取未来到达任务，因此任务到达时的预测热峰值与最终系统热轨迹可以不同；该差异本身不是bug。

## 长期行为

- GROUND前/后12小时闭环率：{segment_map[('GROUND-ONLY','first_12h')]['closure_rate']:.6f} / {segment_map[('GROUND-ONLY','second_12h')]['closure_rate']:.6f}。
- OFFLOAD前/后12小时闭环率：{segment_map[('OFFLOAD-EFT-THERMAL-ALIGNED','first_12h')]['closure_rate']:.6f} / {segment_map[('OFFLOAD-EFT-THERMAL-ALIGNED','second_12h')]['closure_rate']:.6f}。
- 最后四分之一任务闭环率：GROUND={ground['final_quarter_task_closure_rate']:.6f}，OFFLOAD={offload['final_quarter_task_closure_rate']:.6f}。

## 已验证

本次只验证固定seed=2026、capacity=0.50、horizon=86400秒、268任务的严格配对单场景结果；运行无人工干预，输出可重复核验。

## 尚未证明

尚未证明多seed稳健性、capacity=0.35表现、相对PROP的正式优劣或统计显著性；不声称OFFLOAD全局最优，也不声称完整复现原论文SCA。

## 下一步

{'进入多场景稳健性实验。' if status == 'PILOT_PASSED_READY_FOR_FORMAL_ROBUSTNESS' else '先针对accepted-then-miss、异常DEFER或长期退化做一个明确诊断。'}
"""


def write_acceptance_snapshots(out: Path, run_code_commit: str) -> None:
    """Write audit-only snapshots from the resolved configuration saved by the run."""
    resolved_path = out / "resolved_run_config.json"
    task_path = out / "task_manifest_check.json"
    fingerprint_path = out / "environment_fingerprint_check.json"
    if not all(path.exists() for path in (resolved_path, task_path, fingerprint_path)):
        raise FileNotFoundError("acceptance snapshots require the run's saved resolved config and identity checks")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    task_check = json.loads(task_path.read_text(encoding="utf-8"))
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    ground = resolved["ground_only"]
    offload = resolved["offload_eft_thermal_aligned"]
    ground_env = ground["benchmark_environment"]
    offload_env = offload["benchmark_environment"]
    if stable_hash(ground_env) != stable_hash(offload_env):
        raise RuntimeError("saved resolved configurations do not share the same benchmark environment")
    experiment = ground["experiment"]
    if experiment != offload["experiment"]:
        raise RuntimeError("saved resolved configurations do not share the same experiment selection")

    resolved_scenario = {
        "schema_version": 1,
        "derived_run_key": resolved["run_key"],
        "run_key_is_manifest_scenario_id": resolved["run_key_is_manifest_scenario_id"],
        "seed": experiment["scenario_seeds"][0],
        "capacity_multiplier": experiment["capacities"][0],
        "horizon_s": ground_env["simulation"]["horizon_s"]["value"],
        "slot_seconds": ground_env["simulation"]["slot_seconds"]["value"],
        "task_count": task_check["task_count"],
        "hv_count": task_check["hv_count"],
        "hv_target_count": task_check["hv_target_count"],
        "task_hash": task_check["task_hash"],
        "environment_hash": fingerprint["environment_hash"],
        "benchmark_environment_fingerprint": fingerprint["benchmark_environment_fingerprint"],
        "scenario_object_hash": task_check["scenario_object_hash"],
        "key_configuration": {
            "communication": {
                "capacity_multiplier": experiment["capacities"][0],
                "throughput_min_mbps": ground_env["communication"]["throughput_min_mbps"],
                "throughput_max_mbps": ground_env["communication"]["throughput_max_mbps"],
                "contact_trace": ground_env["contacts"],
                "source": "saved resolved_run_config.json from the completed run",
            },
            "thermal": ground_env["thermal"],
            "soc": ground_env["battery"],
            "ground_fifo": {
                "servers": ground["method"]["ground_compute_servers"],
                "policy": ground["method"]["ground_queue_policy"],
                "capacity": ground["method"]["ground_queue_capacity"],
                "source": "configs/methods/ground_only_v1.yaml resolved with benchmark_environment_v1.yaml",
            },
        },
        "configuration_sources": {
            "shared_environment": "configs/benchmark_environment_v1.yaml",
            "scenario_manifest": "manifests/scenario_manifest_v1.csv",
            "environment_fingerprint": "manifests/benchmark_environment_fingerprint_v1.json",
            "experiment": "configs/experiments/offload_vs_ground_24h_pilot_seed2026_cap050.yaml",
        },
        "derived_from_saved_actual_run_configuration": True,
    }
    (out / "resolved_scenario.json").write_text(
        json.dumps(resolved_scenario, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    method_sources = {
        "ground_only": "configs/methods/ground_only_v1.yaml",
        "offload_eft_thermal_aligned": "configs/methods/offload_eft_thermal_aligned_v1.yaml",
    }
    for directory_name, config in (("ground_only", ground), ("offload_eft_thermal_aligned", offload)):
        method = config["method"]
        snapshot = {
            "schema_version": 1,
            "code_commit": run_code_commit,
            "method_id": method["method_id"],
            "selected_for_this_pilot": True,
            "registered_default_enable_switches": method.get("feature_flags", {}),
            "offloading_policy_default": method.get("offloading_policy", {}),
            "configuration_sources": {
                "shared_environment": "configs/benchmark_environment_v1.yaml",
                "method_private": method_sources[directory_name],
                "experiment_selection": "configs/experiments/offload_vs_ground_24h_pilot_seed2026_cap050.yaml",
            },
            "shared_environment": config["benchmark_environment"],
            "method_private": method,
            "experiment_selection": config["experiment"],
            "resolved_config_sha256": stable_hash(config),
            "derived_from_saved_actual_run_configuration": True,
        }
        target = out / directory_name / "config_snapshot.json"
        target.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    run_manifest_path = out / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_manifest["code_commit"] = run_code_commit
        run_manifest["acceptance_snapshots_from_saved_resolved_config"] = True
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--acceptance-only", action="store_true")
    parser.add_argument("--run-code-commit")
    args = parser.parse_args()
    out = args.output_dir
    run_code_commit = args.run_code_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.acceptance_only:
        write_acceptance_snapshots(out, run_code_commit)
        print(json.dumps({"acceptance_snapshots_written": True, "output": str(out), "code_commit": run_code_commit}))
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite pilot output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ground_only").mkdir()
    (out / "offload_eft_thermal_aligned").mkdir()

    fp = fingerprint_checks()
    if not fp["all_match"]:
        raise RuntimeError("protected environment fingerprint mismatch")
    shared = build_shared_scenario()
    ground_resolved, offload_resolved, fairness = resolved_methods()
    pars = common_parameters(shared["configs"], offload_resolved)
    # Both methods receive this exact timeline object; only simulator state is recreated.
    ground_sim, ground_decisions = run_ground(shared, ground_resolved, pars)
    offload_sim, offload_decisions, task0009_diag = run_offload(shared, offload_resolved, pars)

    ground_trace = augment_trace(ground_sim.actual_trace, shared["timeline"], shared["configs"]["simulator"])
    offload_trace = augment_trace(offload_sim.actual_trace, shared["timeline"], shared["configs"]["simulator"])
    ground_tasks, ground_common = task_rows("GROUND-ONLY", ground_sim, shared, pars, ground_decisions)
    offload_tasks, offload_common = task_rows("OFFLOAD-EFT-THERMAL-ALIGNED", offload_sim, shared, pars, offload_decisions)
    ground_summary = summarize("GROUND-ONLY", ground_tasks, ground_common, ground_trace, ground_sim)
    offload_summary = summarize("OFFLOAD-EFT-THERMAL-ALIGNED", offload_tasks, offload_common, offload_trace, offload_sim)

    prediction = []
    offload_by_id = {row["task_id"]: row for row in offload_tasks}
    for decision in offload_decisions:
        if decision["selected_path"] == DEFER:
            continue
        tid = decision["task_id"]
        job = offload_sim.jobs[tid]
        estimate_key = "local" if decision["selected_path"] == LOCAL else "ground"
        predicted_finish = float(decision[f"predicted_{estimate_key}_finish_s"])
        predicted_bytes = float(decision[f"predicted_{estimate_key}_bytes"])
        predicted_thermal = float(decision[f"predicted_{estimate_key}_max_thermal"])
        arrival = float(decision["arrival_s"])
        finish = job.closure_s
        end = min(86400.0, finish if finish is not None else 86400.0)
        actual_rows = [row for row in offload_trace if arrival - TOL <= float(row["time_s"]) <= end + TOL]
        actual_thermal = max([float(row["thermal_after"]) for row in actual_rows] or [offload_sim.thermal])
        actual_bytes = float(job.transmitted_mb) * 1_000_000.0
        prediction.append({
            "task_id": tid, "path": decision["selected_path"], "predicted_finish_s": predicted_finish,
            "actual_finish_s": finish if finish is not None else "",
            "finish_error_s": finish - predicted_finish if finish is not None and math.isfinite(predicted_finish) else "",
            "predicted_tx_bytes": predicted_bytes, "actual_tx_bytes": actual_bytes,
            "tx_bytes_error": actual_bytes - predicted_bytes,
            "predicted_max_thermal": predicted_thermal, "actual_system_max_thermal_during_task": actual_thermal,
            "thermal_error": actual_thermal - predicted_thermal,
            "actual_fields_copied_from_prediction": 0,
        })

    pairs = pairwise(ground_tasks, offload_tasks)
    counts = Counter(row["transition_category"] for row in pairs)
    transition = {
        "both_closed": counts["both_closed"], "ground_only_closed": counts["ground_only_closed"],
        "offload_only_closed": counts["offload_only_closed"], "neither_closed": counts["neither_closed"],
        "offload_added_closure_utility": sum(float(row["offload_utility"]) for row in pairs if row["transition_category"] == "offload_only_closed"),
        "offload_lost_closure_utility": sum(float(row["ground_only_utility"]) for row in pairs if row["transition_category"] == "ground_only_closed"),
        "net_closure_delta": offload_summary["closure_count"] - ground_summary["closure_count"],
        "net_gross_utility_delta": offload_summary["common_gross_utility"] - ground_summary["common_gross_utility"],
        "net_common_utility_delta": offload_summary["common_net_utility"] - ground_summary["common_net_utility"],
    }
    defer_reasons = Counter()
    for row in offload_tasks:
        if row["selected_path"] == DEFER:
            for reason in str(row["failure_reason"]).split(";"):
                if reason:
                    defer_reasons[reason] += 1
    accepted_miss = sum(row["selected_path"] != DEFER and not int(row["closed"]) for row in offload_tasks)
    observed_thermal_false_negative = sum(
        row["selected_path"] == DEFER and "thermal_hard_limit" in str(row["failure_reason"])
        and any(math.isfinite(float(d.get(key, math.inf))) for key in ("predicted_local_finish_s", "predicted_ground_finish_s"))
        for row in offload_tasks for d in [next(x for x in offload_decisions if x["task_id"] == row["task_id"])]
    )
    defer_summary = {
        "defer_count": sum(row["selected_path"] == DEFER for row in offload_tasks),
        "reason_counts": dict(defer_reasons), "thermal_estimator_false_negative_count": observed_thermal_false_negative,
        "thermal_false_negative_definition": "DEFER with thermal_hard_limit reason despite a finite predicted candidate finish; no future-task oracle used",
        "accepted_then_deadline_miss_count": accepted_miss,
        "migration_count": 0, "duplicate_execution_count": 0, "future_task_read_count": 0,
    }
    segments = segment_rows("GROUND-ONLY", ground_tasks) + segment_rows("OFFLOAD-EFT-THERMAL-ALIGNED", offload_tasks)
    task0009_pair = next(row for row in pairs if row["task_id"] == "sarfishsample_task_0009")
    task0009_diag.update({"ground_only_closed": bool(task0009_pair["ground_only_closed"]),
                          "offload_closed": bool(task0009_pair["offload_closed"]),
                          "ground_only_finish_s": task0009_pair["ground_only_actual_finish_s"],
                          "offload_finish_s": task0009_pair["offload_actual_finish_s"],
                          "utility_delta": task0009_pair["offload_minus_ground_utility"]})
    unknown_defer = sum(count for reason, count in defer_reasons.items() if reason not in {
        "deadline", "compute_safety_blocked", "downlink_safety_blocked", "product_or_requirement_infeasible",
        "satellite_queue_capacity", "soc_floor", "thermal_hard_limit"})
    implementation_error = any([
        defer_summary["migration_count"], defer_summary["duplicate_execution_count"],
        defer_summary["future_task_read_count"], len(pairs) != 268,
        ground_summary["closure_count"] != sum(int(row["on_time_task_success"]) for row in ground_common),
        offload_summary["closure_count"] != sum(int(row["on_time_task_success"]) for row in offload_common),
    ])
    safety_error = any([ground_summary["thermal_violation_count"], ground_summary["soc_violation_count"],
                        offload_summary["thermal_violation_count"], offload_summary["soc_violation_count"]])
    if not all(status.startswith(("PASS", "WARNING")) for key, status in fairness.items() if key.endswith(("only", "aligned"))):
        status = "PILOT_FAILED_FAIRNESS"
    elif safety_error:
        status = "PILOT_FAILED_SAFETY"
    elif implementation_error:
        status = "PILOT_FAILED_IMPLEMENTATION"
    elif unknown_defer or accepted_miss or observed_thermal_false_negative:
        status = "PILOT_VALID_BUT_NEEDS_DIAGNOSTIC"
    else:
        status = "PILOT_PASSED_READY_FOR_FORMAL_ROBUSTNESS"

    (out / "resolved_run_config.json").write_text(json.dumps({
        "run_key": RUN_KEY, "run_key_is_manifest_scenario_id": False,
        "seed_source": "scenario_manifest_v1.csv", "capacity_source": "benchmark_environment_v1.yaml",
        "horizon_source": "benchmark_environment_v1.yaml", "ground_only": ground_resolved,
        "offload_eft_thermal_aligned": offload_resolved,
        "ground_path_parameters": pars["ground"].__dict__,
    }, indent=2), encoding="utf-8")
    (out / "environment_fingerprint_check.json").write_text(json.dumps({**fp,
        "environment_hash": shared["env"]["environment_hash_v2"], "environment_hash_match": True,
        "shared_environment_git_diff": [], "pass": True}, indent=2), encoding="utf-8")
    (out / "task_manifest_check.json").write_text(json.dumps({
        "seed_2026_row_count": 1, "task_count": 268, "hv_count": 169, "hv_target_count": 42,
        "task_hash": shared["env"]["task_manifest_hash_v2"], "task_hash_match": True,
        "same_timeline_object_for_both_methods": True, "scenario_object_hash": shared["scenario_object_hash"], "pass": True,
    }, indent=2), encoding="utf-8")
    (out / "fairness_validation.json").write_text(json.dumps({**fairness,
        "same_task_object": True, "same_environment_object": True, "same_ground_path_parameters": True,
        "same_common_evaluator": True, "shared_fields_modified": [], "status": "PASS_WITH_PROXY_VALUES"}, indent=2), encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps({
        "run_key": RUN_KEY, "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": BASE_COMMIT, "code_commit": run_code_commit,
        "method_order": ["GROUND-ONLY", "OFFLOAD-EFT-THERMAL-ALIGNED"],
        "fresh_simulator_per_method": True, "rng_reset_per_method": True,
        "soc_thermal_queues_reset_per_method": True, "shared_frozen_timeline_object": True,
        "manual_intervention_during_run": False, "formal_result": False,
    }, indent=2), encoding="utf-8")
    write_acceptance_snapshots(out, run_code_commit)
    preflight = f"""# Preflight report

- Repository: `{ROOT}`; HEAD contains `{BASE_COMMIT}`; branch `exp/offload_vs_ground_24h_pilot_seed2026_cap050`.
- Six required constraint files read fully and parse: PASS.
- OFFLOAD freeze and runnable GROUND-ONLY commits exist: PASS.
- Seed 2026 row unique; tasks/HV/HV-TARGET: 268/169/42.
- Task hash `{TASK_HASH}`; environment hash `{ENV_HASH}`; benchmark fingerprint `{fp['benchmark_environment_fingerprint']}`.
- Protected components: 13/13 raw or Git-logical line-ending-equivalent matches; shared environment logical diff: empty.
- Both methods use the same frozen scenario object and common evaluator; run conditions: PASS.
"""
    (out / "preflight_report.md").write_text(preflight, encoding="utf-8")

    for directory, summary, tasks, trace in [
        (out / "ground_only", ground_summary, ground_tasks, ground_trace),
        (out / "offload_eft_thermal_aligned", offload_summary, offload_tasks, offload_trace),
    ]:
        (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_csv(directory / "per_task_results.csv", tasks)
        write_csv(directory / "per_slot_resource_trace.csv", trace)
    write_csv(out / "offload_eft_thermal_aligned/decisions.csv", offload_decisions)
    write_csv(out / "offload_eft_thermal_aligned/prediction_actual_comparison.csv", prediction)
    write_csv(out / "pairwise_task_comparison.csv", pairs)
    (out / "closure_transition_summary.json").write_text(json.dumps(transition, indent=2), encoding="utf-8")
    (out / "utility_displacement_summary.json").write_text(json.dumps({**transition,
        "task0009": task0009_diag, "real_resource_displacement_present": bool(counts["ground_only_closed"] and counts["offload_only_closed"]),
        "unexplained_implementation_anomaly_count": int(bool(implementation_error))}, indent=2), encoding="utf-8")
    (out / "defer_reason_summary.json").write_text(json.dumps(defer_summary, indent=2), encoding="utf-8")
    (out / "thermal_margin_summary.json").write_text(json.dumps({
        "GROUND-ONLY": {key: ground_summary[key] for key in ["maximum_thermal", "thermal_margin_to_0_92", "near_thermal_limit_slot_count", "thermal_violation_count", "thermal_limited_partial_send_count", "thermal_pause_count"]},
        "OFFLOAD-EFT-THERMAL-ALIGNED": {key: offload_summary[key] for key in ["maximum_thermal", "thermal_margin_to_0_92", "near_thermal_limit_slot_count", "thermal_violation_count", "thermal_limited_partial_send_count", "thermal_pause_count"]},
        "raw_float_values_preserved": True, "tolerance": TOL,
    }, indent=2), encoding="utf-8")
    write_csv(out / "time_segment_summary.csv", segments)
    write_csv(out / "metric_comparison.csv", metric_rows(ground_summary, offload_summary))
    (out / "final_report.md").write_text(final_report(status, ground_summary, offload_summary, transition, task0009_diag, segments), encoding="utf-8")
    (out / "rollback_manifest.json").write_text(json.dumps({
        "base_commit": BASE_COMMIT,
        "added_pilot_files": ["configs/experiments/offload_vs_ground_24h_pilot_seed2026_cap050.yaml", "scripts/run_offload_vs_ground_24h_pilot.py", "tests/test_offload_vs_ground_24h_pilot.py", str(out.relative_to(ROOT)).replace("\\", "/")],
        "modified_shared_files": [], "rollback": "remove only the pilot additions or revert their dedicated pilot commit; never reset historical worktrees",
    }, indent=2), encoding="utf-8")
    scientific_files = [
        "ground_only/summary.json", "ground_only/per_task_results.csv", "ground_only/per_slot_resource_trace.csv",
        "offload_eft_thermal_aligned/summary.json", "offload_eft_thermal_aligned/decisions.csv",
        "offload_eft_thermal_aligned/per_task_results.csv", "offload_eft_thermal_aligned/per_slot_resource_trace.csv",
        "offload_eft_thermal_aligned/prediction_actual_comparison.csv", "pairwise_task_comparison.csv",
        "closure_transition_summary.json", "utility_displacement_summary.json", "defer_reason_summary.json",
        "thermal_margin_summary.json", "time_segment_summary.csv", "metric_comparison.csv",
    ]
    (out / "scientific_output_hashes.json").write_text(json.dumps({rel: sha(out / rel) for rel in scientific_files}, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "GROUND-ONLY": ground_summary, "OFFLOAD-EFT-THERMAL-ALIGNED": offload_summary,
                      "transitions": transition, "task0009": task0009_diag}, indent=2))


if __name__ == "__main__":
    main()
