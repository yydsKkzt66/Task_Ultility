# -*- coding: utf-8 -*-
"""Run a deterministic small GROUND-ONLY smoke; never a 24-hour experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.resolve_experiment_config import resolve
from scripts.run_sarfish_downlink_operating_region_seed2026 import scale_downlink_capacity
from scripts.validate_experiment_fairness import validate
from src.baselines.ground_only import GroundOnlySimulator, ground_path_parameters, validate_manifest_identity
from src.common.config import load_yaml
from src.evaluation.canonical_sarfish_environment import load_canonical_sarfish_configs, validate_main_paper_environment
from src.evaluation.unified_ground_received_accounting import score_task
from src.timeline.timeline_builder import TimelineBuilder

DEFAULT_OUT = ROOT / "outputs/_ground_only_smoke_v1"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--task-limit", type=int, default=12)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite GROUND-ONLY smoke: {out}")
    out.mkdir(parents=True, exist_ok=True)

    base = ROOT / "configs/benchmark_environment_v1.yaml"
    method = ROOT / "configs/methods/ground_only_v1.yaml"
    experiment = ROOT / "configs/experiments/ground_only_smoke_v1.yaml"
    resolved, _ = resolve(base, method, experiment)
    fairness_status, fairness_checks = validate(base, method, experiment)
    if not fairness_status.startswith(("PASS", "WARNING")):
        raise RuntimeError(fairness_status)

    configs = scale_downlink_capacity(load_canonical_sarfish_configs(ROOT, seed=2026), 0.50)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = validate_main_paper_environment(configs, timeline)
    manifest = list(csv.DictReader((ROOT / "manifests/scenario_manifest_v1.csv").open(encoding="utf-8-sig", newline="")))
    expected_rows = [row for row in manifest if row["seed"] == "2026"]
    if len(expected_rows) != 1:
        raise RuntimeError("seed 2026 scenario row is not unique")
    expected = expected_rows[0]
    validate_manifest_identity(env["task_manifest_hash_v2"], env["environment_hash_v2"], expected["task_hash"], expected["environment_hash"])

    pars = ground_path_parameters(configs)
    method_cfg = resolved["method"]
    sim = GroundOnlySimulator(
        timeline=timeline,
        simulator_cfg=configs["simulator"],
        local_compute_duration_s=0.0,
        local_compute_energy_j=0.0,
        local_payload_mb=0.0,
        ground_preprocess_duration_s=pars.compression_latency_s,
        ground_preprocess_energy_j=pars.compression_energy_j,
        ground_payload_mb=pars.compressed_size_mb,
        ground_compute_duration_s=pars.ground_compute_latency_s,
        ground_servers=method_cfg["ground_compute_servers"],
        ground_queue_capacity=method_cfg["ground_queue_capacity"],
    )
    task_steps = [step for step in timeline if step.task is not None][: args.task_limit]
    task_ids = {step.task.task_id for step in task_steps}
    last_deadline = max(step.slot_index * step.slot_seconds + step.task.deadline_s for step in task_steps)
    for step in timeline:
        now = step.slot_index * step.slot_seconds
        if step.task is not None and step.task.task_id in task_ids:
            task = step.task
            feasible = 1.0 + 1.0e-9 >= task.information_requirement and pars.ground_quality + 1.0e-9 >= task.quality_requirement
            sim.admit_ground_task(task_id=task.task_id, arrival_s=now, deadline_s=now + task.deadline_s, product_feasible=feasible)
        sim.advance_one_slot()
        if now > last_deadline + pars.ground_compute_latency_s + step.slot_seconds:
            break

    task_by_id = {step.task.task_id: step.task for step in task_steps}
    per_task: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    for tid in sorted(task_ids, key=lambda x: int(x.rsplit("_", 1)[1])):
        task = task_by_id[tid]
        job = sim.jobs[tid]
        closed = sim.is_closed_at(tid, math.inf)
        failure = sim.final_failure_reason(tid)
        latency = (job.closure_s - job.arrival_s) if closed and job.closure_s is not None else 0.0
        row = {
            "task_id": tid,
            "selected_path": "GROUND",
            "arrival_s": job.arrival_s,
            "deadline_s": job.deadline_s,
            "compressed_bytes": pars.compressed_size_mb * 1_000_000.0,
            "compression_latency_s": pars.compression_latency_s,
            "compression_energy_j": pars.compression_energy_j,
            "compression_finish_s": job.compute_finish_s if job.compute_finish_s is not None else "",
            "tx_start_s": job.tx_start_s if job.tx_start_s is not None else "",
            "ground_received_s": job.tx_finish_s if job.tx_finish_s is not None else "",
            "ground_queue_enter_s": job.ground_queue_enter_s if job.ground_queue_enter_s is not None else "",
            "ground_compute_start_s": job.ground_compute_start_s if job.ground_compute_start_s is not None else "",
            "ground_compute_finish_s": job.ground_compute_finish_s if job.ground_compute_finish_s is not None else "",
            "actual_finish_s": job.closure_s if job.closure_s is not None else "",
            "actual_latency_s": latency,
            "transmitted_bytes": job.transmitted_mb * 1_000_000.0,
            "satellite_energy_j": (job.compute_energy_j if job.compute_start_s is not None else 0.0) + job.comm_energy_j,
            "closed": int(closed),
            "failure_reason": failure,
            "migration_count": 0,
            "replication_count": 0,
            "future_tasks_read": 0,
        }
        per_task.append(row)
        common_rows.append(score_task({
            "task_id": tid,
            "method": "GROUND-ONLY",
            "variant": "small_smoke_proxy",
            "capture_time_s": job.arrival_s,
            "deadline_time_s": job.deadline_s,
            "effective_task_value": task.task_value,
            "ground_completed": int(closed),
            "capture_to_ground_latency_s": latency,
            "ground_information_sufficiency": 1.0 if closed else 0.0,
            "ground_result_quality": pars.ground_quality if closed else 0.0,
            "partial_ground_gross_utility": 0.0,
            "requirement_satisfied": int(closed),
            "on_time_task_success": int(closed),
            "selected_path": "GROUND",
        }, accounting))

    max_thermal = max(float(row["thermal_after"]) for row in sim.actual_trace)
    min_soc = min(float(row["soc_after"]) for row in sim.actual_trace)
    summary = {
        "method": "GROUND-ONLY",
        "formal_result": False,
        "run_key": "seed2026_cap050_h86400_smoke12",
        "run_key_is_manifest_scenario_id": False,
        "seed": 2026,
        "capacity_multiplier": 0.50,
        "task_count": len(per_task),
        "ground_decision_count": len(sim.ground_only_decisions),
        "local_decision_count": 0,
        "compression_completed_count": sum(bool(row["compression_finish_s"] != "") for row in per_task),
        "fully_downlinked_count": sum(bool(row["ground_received_s"] != "") for row in per_task),
        "ground_fifo_entry_count": sum(bool(row["ground_queue_enter_s"] != "") for row in per_task),
        "ground_compute_completed_count": sum(bool(row["ground_compute_finish_s"] != "") for row in per_task),
        "closure_count": sum(int(row["closed"]) for row in per_task),
        "deadline_or_infeasible_count": sum(not bool(row["closed"]) for row in per_task),
        "minimum_soc": min_soc,
        "maximum_thermal": max_thermal,
        "soc_violations": sum(float(row["soc_after"]) < configs["simulator"]["battery_min"] - 1.0e-9 for row in sim.actual_trace),
        "thermal_violations": sum(float(row["thermal_after"]) > configs["simulator"]["thermal_max"] + 1.0e-9 for row in sim.actual_trace),
        "duplicate_decisions": len(sim.ground_only_decisions) - len(set(sim.ground_only_decisions)),
        "duplicate_closures": len([r for r in per_task if r["closed"]]) - len({r["task_id"] for r in per_task if r["closed"]}),
        "migration_count": sim.migration_count,
        "replication_count": sim.replication_count,
        "future_tasks_read": sim.future_tasks_read,
        "task_hash": env["task_manifest_hash_v2"],
        "environment_hash": env["environment_hash_v2"],
        "fairness_status": fairness_status,
        "fairness_check_count": len(fairness_checks),
        "common_closure_count": sum(int(row["on_time_task_success"]) for row in common_rows),
    }
    if not (
        summary["compression_completed_count"] >= 1
        and summary["fully_downlinked_count"] >= 1
        and summary["ground_fifo_entry_count"] >= 1
        and summary["closure_count"] >= 1
        and summary["deadline_or_infeasible_count"] >= 1
        and summary["soc_violations"] == summary["thermal_violations"] == 0
        and summary["duplicate_decisions"] == summary["duplicate_closures"] == 0
    ):
        raise RuntimeError(f"GROUND-ONLY smoke acceptance failed: {summary}")
    write_csv(out / "per_task_results.csv", per_task)
    write_csv(out / "per_slot_resource_trace.csv", sim.actual_trace)
    write_csv(out / "common_evaluator_trace.csv", common_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "config_snapshot.json").write_text(json.dumps({"resolved": resolved, "ground_path_parameters": pars.__dict__}, indent=2), encoding="utf-8")
    (out / "key_hashes.json").write_text(json.dumps({
        "summary.json": sha(out / "summary.json"),
        "per_task_results.csv": sha(out / "per_task_results.csv"),
        "per_slot_resource_trace.csv": sha(out / "per_slot_resource_trace.csv"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
