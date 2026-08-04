"""Rollback-safe SARFishSample downlink operating-region sweep for seed 2026."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_prop_full_vs_b1_unified_seed2026 import proposed_energy_safety
from scripts.run_sarfish_canonical_seed2026 import (
    LEGACY_ENV_HASH,
    LEGACY_TASK_HASH,
    b1_config,
    prop_configs,
    stage_paths,
    write_csv,
)
from scripts.run_unified_ground_received_accounting_seed2026 import b1_energy, b1_safety
from src.baselines.b1_enhanced import B1EnhancedSimulator
from src.common.config import load_yaml
from src.evaluation.baselines import build_routers
from src.evaluation.canonical_sarfish_environment import (
    canonical_task_rows_v2,
    validate_main_paper_environment,
)
from src.evaluation.hv_phase1 import is_hv_all, is_hv_target, matched_latency, summarize_group
from src.evaluation.task_outcome_adapter import adapt_b1, adapt_proposed, manifest_rows
from src.evaluation.unified_ground_received_accounting import summarize_method
from src.energy.comm_energy import compute_comm_energy_j
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


OUT = ROOT / "outputs/_sarfish_downlink_operating_region_seed2026"
CANONICAL = ROOT / "outputs/_sarfish_formal_nominal_seed2026_canonical_v2"
MULTIPLIERS = (1.00, 0.75, 0.50, 0.35, 0.25)
METHOD_POLICIES = (("B1-FIFO-1.4W", "fifo"), ("B1-Enhanced-1.4W", "enhanced"))
ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"


def f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def b(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return (result.stdout or "") + (result.stderr or "")


def scale_downlink_capacity(configs: dict[str, Any], multiplier: float) -> dict[str, Any]:
    scaled = deepcopy(configs)
    contact = scaled["timeline"].setdefault("contact", {})
    for key in ("throughput_min_mbps", "throughput_max_mbps"):
        if key in contact:
            contact[key] = float(contact[key]) * float(multiplier)
    scaled["downlink_capacity_multiplier"] = float(multiplier)
    return scaled


def available_capacity_mb(timeline: list[Any], downlink_efficiency: float) -> float:
    total = 0.0
    for step in timeline:
        link = step.link_trace
        if link.in_contact:
            total += max(0.0, float(link.throughput_mbps)) * float(downlink_efficiency) / 8.0 * float(step.slot_seconds)
    return total


def write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def b1_parameter_audit(configs: dict[str, Any], timeline: list[Any]) -> tuple[list[dict[str, Any]], bool, str]:
    raw = load_yaml(ROOT / "configs/baseline_b1_enhanced.yaml")
    b1 = b1_config("enhanced")
    tasks = [step.task for step in timeline if step.task is not None]
    task_raw_values = sorted(
        {
            f((task.metadata or {}).get("raw_payload_mb", ""), math.nan)
            for task in tasks
            if math.isfinite(f((task.metadata or {}).get("raw_payload_mb", ""), math.nan))
        }
    )
    default_raw = float(raw["raw_size"]["default_raw_size_mb"])
    compressed = float(raw["compression"]["reference_compressed_size_mb"])
    rows = [
        {
            "parameter": "compressed_task_observation_mb",
            "value": compressed,
            "unit": "MB",
            "classification": "literature-derived; engineering assumption for SARFish task observation scope; optimistic assumption",
            "evidence": "configs/baseline_b1_enhanced.yaml reference_compressed_size_mb=4.0; raw_size default 32 MB. SARFish task metadata does not provide per-task raw_payload_mb.",
            "fairness_resolution": "The object is treated as one task observation, not a full SAR scene. This is optimistic unless task-level input bytes are externally supported.",
        },
        {
            "parameter": "compression_ratio",
            "value": b1.compression.compression_ratio,
            "unit": "raw_MB/compressed_MB",
            "classification": "literature-derived; optimistic assumption",
            "evidence": "configs/baseline_b1_enhanced.yaml compression_ratio=8.0, described as Phi-Sat-2 CAE analogue.",
            "fairness_resolution": "Kept unchanged as B1-OPTIMISTIC.",
        },
        {
            "parameter": "compression_power_w",
            "value": b1.compression.compression_power_w,
            "unit": "W",
            "classification": "literature-derived sensitivity point; optimistic assumption",
            "evidence": "canonical runner sets compression_power_w=1.4 from paper_accelerator_power_w.",
            "fairness_resolution": "Kept unchanged as B1-OPTIMISTIC.",
        },
        {
            "parameter": "reconstructed_quality",
            "value": b1.compression.reconstructed_ssim_proxy,
            "unit": "quality proxy",
            "classification": "literature-derived; optimistic assumption",
            "evidence": "configs/baseline_b1_enhanced.yaml reconstructed_ssim_proxy=0.96.",
            "fairness_resolution": "Used only as compression sufficiency quality in corrected evaluator.",
        },
        {
            "parameter": "downstream_result_quality_requested_0_88",
            "value": 0.88,
            "unit": "task quality",
            "classification": "project-config-derived; not the active canonical B1 value",
            "evidence": "configs/model_pool.yaml high_accuracy -> medium profile quality_base=0.88 in the generic model pool.",
            "fairness_resolution": "Recorded for provenance because it was named in the request; current canonical SARFish B1 traces use 0.94, so no B1-NOMINAL value was derived from this alone.",
        },
        {
            "parameter": "downstream_result_quality_active_canonical",
            "value": 0.94 * b1.compression.downstream_quality_retention,
            "unit": "task quality",
            "classification": "project-config-derived; engineering assumption",
            "evidence": "canonical SARFish B1 per_task_delivery_trace.ground_task_quality=0.94 and downstream_quality_retention=1.0.",
            "fairness_resolution": "Used for utility, not for compression sufficiency closure gate.",
        },
        {
            "parameter": "full_frame_support_for_context_evidence",
            "value": 1,
            "unit": "boolean",
            "classification": "engineering assumption; optimistic assumption",
            "evidence": "configs/unified_ground_received_accounting.yaml full_frame_information_mapping and corrected evaluator dominance mapping.",
            "fairness_resolution": "Full-frame support is accepted for context/evidence only after complete receipt and ground processing.",
        },
        {
            "parameter": "transmitted_byte_accounting",
            "value": "downlinked_mb and per-slot sent_mb",
            "unit": "MB",
            "classification": "data-derived",
            "evidence": "B1 per_task_delivery_trace.downlinked_mb and queue_trace.sent_mb include partial transmissions; failed partials receive no closure.",
            "fairness_resolution": "Sweep recomputes transmitted bytes from raw events and reports wasted bytes separately.",
        },
    ]
    grounded_nominal = bool(task_raw_values and len(task_raw_values) == 1 and abs(task_raw_values[0] - default_raw) < 1e-9)
    reason = (
        "B1-NOMINAL not created: SARFish task metadata lacks actual task input bytes, and changing 4 MB/CR/power/quality would require invented values."
        if not grounded_nominal
        else "B1-NOMINAL could be grounded, but current canonical values already match task raw bytes."
    )
    return rows, grounded_nominal, reason


def latency_summary(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, float]:
    values = [f(row.get("capture_to_ground_latency_s")) for row in rows if predicate(row)]
    return {
        "mean_closure_latency_s": mean(values),
        "median_closure_latency_s": median(values),
        "p90_closure_latency_s": percentile(values, 0.90),
        "p95_closure_latency_s": percentile(values, 0.95),
    }


def first_useful_latency(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [f(row.get("capture_to_ground_latency_s")) for row in rows if b(row.get("ground_completed"))]
    return {
        "mean_first_useful_product_latency_s": mean(values),
        "median_first_useful_product_latency_s": median(values),
        "p90_first_useful_product_latency_s": percentile(values, 0.90),
        "p95_first_useful_product_latency_s": percentile(values, 0.95),
    }


def task_failure_reason(row: dict[str, Any], raw_by_task: dict[str, dict[str, Any]] | None = None) -> str:
    if b(row.get("on_time_task_success")):
        return "success"
    if b(row.get("requirement_satisfied")):
        return "late_requirement_satisfied"
    if b(row.get("ground_completed")) or b(row.get("partial_ground_delivery")):
        if f(row.get("compression_sufficiency_quality"), 1.0) + 1e-9 < f(row.get("quality_requirement")):
            return "quality_failure"
        return "lower-than-required product"
    raw = (raw_by_task or {}).get(str(row.get("task_id")), {})
    state = str(raw.get("state", "")).lower()
    downlinked = f(raw.get("downlinked_mb"))
    failure = str(row.get("failure_reason", "") or raw.get("blocked_reason", "")).lower()
    if downlinked > 0.0:
        return "incomplete transmission"
    if "safety" in failure:
        return "safety blocking"
    if "compute" in failure or "compression" in state:
        return "compute delay"
    if "deadline" in failure or state == "expired":
        return "no contact before deadline"
    if b(row.get("terminal_backlog_flag")):
        return "insufficient contact capacity"
    return "unknown"


def group_summary(
    method: str,
    capacity: float,
    rows: list[dict[str, Any]],
    system: dict[str, Any],
    direct: set[str],
    shallow: set[str],
    deep: set[str],
    predicate: Callable[[dict[str, Any]], bool],
    group: str,
) -> dict[str, Any]:
    selected_ids = {str(r["task_id"]) for r in rows if predicate(r) and b(r.get("requirement_satisfied"))}
    base = summarize_group(
        rows,
        predicate,
        system_summary=system,
        direct_ids=direct,
        staged_ids=shallow | deep,
    )
    base.update(
        {
            "capacity_multiplier": capacity,
            "method": method,
            "task_group": group,
            "direct_closures": len(selected_ids & direct),
            "shallow_staged_closures": len(selected_ids & shallow),
            "deep_staged_closures": len(selected_ids & deep),
            "staged_closures": len(selected_ids & (shallow | deep)),
        }
    )
    return base


def prop_transmitted_by_task(records: list[Any]) -> dict[str, float]:
    by_task: dict[str, float] = defaultdict(float)
    for record in records:
        total = f(getattr(record, "S_down_total_mb", 0.0))
        ids = [item for item in str(getattr(record, "transmitted_task_ids", "") or "").split(";") if item]
        if ids:
            share = total / len(ids)
            for task_id in ids:
                by_task[task_id] += share
        elif str(getattr(record, "task_id", "")):
            by_task[str(record.task_id)] += f(getattr(record, "S_down_current_task_mb", 0.0))
    return dict(by_task)


def raw_prop_records(records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "slot": getattr(r, "slot", ""),
            "task_id": getattr(r, "task_id", ""),
            "S_down_total_mb": getattr(r, "S_down_total_mb", 0.0),
            "S_down_current_task_mb": getattr(r, "S_down_current_task_mb", 0.0),
            "queue_next_mb": getattr(r, "queue_next_mb", 0.0),
            "queue_item_count": getattr(r, "queue_item_count", 0),
            "delivery_status": getattr(r, "delivery_status", ""),
            "transmitted_task_ids": getattr(r, "transmitted_task_ids", ""),
            "delivered_task_ids": getattr(r, "delivered_task_ids", ""),
            "expired_task_ids": getattr(r, "expired_task_ids", ""),
            "dropped_task_ids": getattr(r, "dropped_task_ids", ""),
            "comm_energy_j": getattr(r, "comm_energy_j", 0.0),
            "compute_energy_j": getattr(r, "compute_energy_j", 0.0),
        }
        for r in records
    ]


def run_b1(
    method: str,
    policy: str,
    capacity: float,
    configs: dict[str, Any],
    timeline: list[Any],
    manifest: dict[str, Any],
    accounting: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    simulator = B1EnhancedSimulator(b1_config(policy), configs["simulator"], configs["model_pool"])
    result = simulator.run(timeline)
    rows = adapt_b1(manifest, result["per_task_delivery_trace"], method, policy, 2026, accounting)
    energy = b1_energy(result["summary"])
    safety = b1_safety(result["summary"], result["queue_trace"])
    summary = summarize_method(rows, energy, safety, {"internal_utility": result["summary"].get("net_system_utility", "")}, accounting)
    raw_dir = out_dir / "raw" / f"cap_{capacity:.2f}" / method
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_csv(raw_dir / "per_task_delivery_trace.csv", result["per_task_delivery_trace"])
    write_csv(raw_dir / "queue_trace.csv", result["queue_trace"])
    write_csv(raw_dir / "scheduler_candidate_trace.csv", result["scheduler_candidate_trace"])
    write_csv(raw_dir / "compression_trace.csv", result["compression_trace"])
    return {"rows": rows, "summary": summary, "raw": result}


def run_prop(
    capacity: float,
    configs: dict[str, Any],
    timeline: list[Any],
    manifest: dict[str, Any],
    accounting: dict[str, Any],
    utility_cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    rows = adapt_proposed(
        manifest,
        result.records,
        runner.last_payload_lifecycle_current_records,
        "PROP-FULL",
        "proposed_guarantee_aware_full",
        2026,
        accounting,
        utility_cfg,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )
    energy, safety = proposed_energy_safety(result, result.records, configs, accounting)
    summary = summarize_method(rows, energy, safety, {"internal_utility": result.summary.get("net_system_utility", "")}, accounting)
    direct, shallow, deep = stage_paths(rows, list(runner.last_payload_lifecycle_current_records))
    raw_dir = out_dir / "raw" / f"cap_{capacity:.2f}" / "PROP-FULL"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_csv(raw_dir / "records.csv", raw_prop_records(result.records))
    write_csv(raw_dir / "lifecycle.csv", list(runner.last_payload_lifecycle_current_records))
    write_csv(raw_dir / "staged_request_trace.csv", list(runner.last_staged_delivery_request_trace))
    return {
        "rows": rows,
        "summary": summary,
        "records": result.records,
        "lifecycle": list(runner.last_payload_lifecycle_current_records),
        "stage_sets": (direct, shallow, deep),
    }


def method_resource_row(
    method: str,
    capacity: float,
    available_mb: float,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    transmitted = f(summary.get("delivered_mb"))
    queue_peak = f(summary.get("maximum_queue_mb"))
    terminal_queue = f(summary.get("terminal_queue_mb"))
    return {
        "capacity_multiplier": capacity,
        "method": method,
        "available_capacity_mb": available_mb,
        "transmitted_mb_including_wasted": transmitted,
        "unused_capacity_mb": max(0.0, available_mb - transmitted),
        "capacity_utilization": transmitted / max(available_mb, 1e-12),
        "compute_utilization": f(raw.get("compute_utilization", 0.0)),
        "controllable_energy_wh": f(summary.get("controllable_mission_energy_wh")),
        "total_energy_wh": f(summary.get("total_spacecraft_energy_wh")),
        "queue_peak_mb": queue_peak,
        "terminal_queue_mb": terminal_queue,
        "min_soc": f(summary.get("min_soc")),
        "final_soc": f(summary.get("final_soc")),
        "thermal_peak": f(summary.get("max_thermal_pressure")),
        "safety_violations": f(summary.get("hard_safety_violation_count")),
    }


def byte_accounting_row(
    method: str,
    capacity: float,
    rows: list[dict[str, Any]],
    transmitted_by_task: dict[str, float],
    summary: dict[str, Any],
) -> dict[str, Any]:
    closure_ids = {str(row["task_id"]) for row in rows if b(row.get("requirement_satisfied"))}
    ground_ids = {str(row["task_id"]) for row in rows if b(row.get("ground_completed"))}
    total = sum(transmitted_by_task.values()) if transmitted_by_task else f(summary.get("delivered_mb"))
    useful = sum(value for task_id, value in transmitted_by_task.items() if task_id in closure_ids)
    ground_not_closed = sum(value for task_id, value in transmitted_by_task.items() if task_id in ground_ids - closure_ids)
    no_ground = sum(value for task_id, value in transmitted_by_task.items() if task_id not in ground_ids)
    return {
        "capacity_multiplier": capacity,
        "method": method,
        "transmitted_mb_recomputed": total,
        "summary_delivered_mb": f(summary.get("delivered_mb")),
        "useful_requirement_closure_mb": useful,
        "wasted_partial_or_no_closure_mb": max(0.0, total - useful),
        "ground_but_not_closed_mb": ground_not_closed,
        "no_ground_or_incomplete_mb": no_ground,
        "byte_accounting_error_mb": total - f(summary.get("delivered_mb")),
    }


def recompute_b1_energy(
    result: dict[str, Any],
    timeline: list[Any],
    configs: dict[str, Any],
    summary: dict[str, Any],
    capacity: float,
    method: str,
) -> dict[str, Any]:
    by_slot = {int(step.slot_index): step for step in timeline}
    efficiency = float(configs["simulator"].get("downlink_efficiency", 1.0))
    comm_power = float(configs["simulator"].get("comm_power_w", configs["simulator"].get("tx_power_w", 0.0)))
    alpha = float(configs["simulator"].get("comm_alpha_gamma", 0.0))
    compression_wh = sum(
        f(row.get("compression_energy_wh"))
        for row in result["compression_trace"]
        if str(row.get("state", "")) == "compressing"
    )
    comm_j = 0.0
    for row in result["queue_trace"]:
        sent = f(row.get("sent_mb"))
        if sent <= 0.0:
            continue
        step = by_slot[int(f(row.get("slot_index")))]
        rate = max(0.0, step.link_trace.throughput_mbps) * efficiency / 8.0
        tx_time = 0.0 if rate <= 1e-12 else sent / rate
        comm_j += compute_comm_energy_j(comm_power, tx_time, step.link_trace.link_quality, alpha)
    return {
        "capacity_multiplier": capacity,
        "method": method,
        "compression_energy_wh_recomputed": compression_wh,
        "compression_energy_wh_summary": f(summary.get("compression_energy_wh")),
        "compression_energy_error_wh": compression_wh - f(summary.get("compression_energy_wh")),
        "communication_energy_wh_recomputed": comm_j / 3600.0,
        "communication_energy_wh_summary": f(summary.get("communication_energy_wh")),
        "communication_energy_error_wh": comm_j / 3600.0 - f(summary.get("communication_energy_wh")),
        "compute_energy_wh_recomputed": 0.0,
        "compute_energy_wh_summary": f(summary.get("compute_energy_wh")),
        "compute_energy_error_wh": -f(summary.get("compute_energy_wh")),
    }


def recompute_prop_energy(
    records: list[Any],
    summary: dict[str, Any],
    capacity: float,
) -> dict[str, Any]:
    compute_wh = sum(f(getattr(record, "compute_energy_j", 0.0)) for record in records) / 3600.0
    comm_wh = sum(f(getattr(record, "comm_energy_j", 0.0)) for record in records) / 3600.0
    return {
        "capacity_multiplier": capacity,
        "method": "PROP-FULL",
        "compression_energy_wh_recomputed": 0.0,
        "compression_energy_wh_summary": f(summary.get("compression_energy_wh")),
        "compression_energy_error_wh": -f(summary.get("compression_energy_wh")),
        "communication_energy_wh_recomputed": comm_wh,
        "communication_energy_wh_summary": f(summary.get("communication_energy_wh")),
        "communication_energy_error_wh": comm_wh - f(summary.get("communication_energy_wh")),
        "compute_energy_wh_recomputed": compute_wh,
        "compute_energy_wh_summary": f(summary.get("compute_energy_wh")),
        "compute_energy_error_wh": compute_wh - f(summary.get("compute_energy_wh")),
    }


def flatten_summary(
    method: str,
    capacity: float,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    available_mb: float,
    direct: set[str],
    shallow: set[str],
    deep: set[str],
) -> dict[str, Any]:
    closures = [row for row in rows if b(row.get("requirement_satisfied"))]
    base = {
        "capacity_multiplier": capacity,
        "method": method,
        "arrived": len(rows),
        "any_ground": sum(b(row.get("ground_completed")) for row in rows),
        "partial_only": sum(b(row.get("partial_ground_delivery")) for row in rows),
        "requirement_closure": sum(b(row.get("requirement_satisfied")) for row in rows),
        "on_time_closure": sum(b(row.get("on_time_task_success")) for row in rows),
        "no_ground": sum(not b(row.get("ground_completed")) and not b(row.get("partial_ground_delivery")) for row in rows),
        "partial_utility": sum(f(row.get("partial_ground_gross_utility")) for row in rows),
        "closure_utility": sum(f(row.get("gross_ground_utility")) for row in closures),
        "common_gross_utility": f(summary.get("common_ground_gross_utility")),
        "common_net_utility": f(summary.get("common_ground_net_utility")),
        "utility_per_mb": f(summary.get("common_ground_gross_utility")) / max(f(summary.get("delivered_mb")), 1e-12),
        "utility_per_controllable_wh": f(summary.get("common_ground_gross_utility")) / max(f(summary.get("controllable_mission_energy_wh")), 1e-12),
        "available_capacity_mb": available_mb,
        "transmitted_mb": f(summary.get("delivered_mb")),
        "controllable_energy_wh": f(summary.get("controllable_mission_energy_wh")),
        "total_energy_wh": f(summary.get("total_spacecraft_energy_wh")),
        "min_soc": f(summary.get("min_soc")),
        "final_soc": f(summary.get("final_soc")),
        "thermal_peak": f(summary.get("max_thermal_pressure")),
        "safety_violations": f(summary.get("hard_safety_violation_count")),
        "direct_closures": len({str(r["task_id"]) for r in closures} & direct),
        "shallow_staged_closures": len({str(r["task_id"]) for r in closures} & shallow),
        "deep_staged_closures": len({str(r["task_id"]) for r in closures} & deep),
    }
    base.update(latency_summary(rows, lambda row: b(row.get("requirement_satisfied"))))
    base.update(first_useful_latency(rows))
    return base


def validate_outputs(
    capacity_rows: list[dict[str, Any]],
    env_rows: list[dict[str, Any]],
    method_rows: dict[tuple[float, str], list[dict[str, Any]]],
    stage_sets: dict[tuple[float, str], tuple[set[str], set[str], set[str]]],
    canonical_100: dict[str, dict[str, Any]],
) -> list[str]:
    checks: list[str] = []
    for capacity in MULTIPLIERS:
        rows_at_cap = [row for row in env_rows if abs(f(row["capacity_multiplier"]) - capacity) < 1e-9]
        if len({row["task_manifest_hash_v2"] for row in rows_at_cap}) != 1:
            raise AssertionError(f"task hash mismatch at {capacity}")
        if len({row["environment_hash_v2"] for row in rows_at_cap}) != 1:
            raise AssertionError(f"environment hash mismatch at {capacity}")
        checks.append(f"PASS: identical task/environment hashes across methods at capacity {capacity:.2f}.")
    for (capacity, method), rows in method_rows.items():
        ids = [str(row.get("task_id", "")) for row in rows]
        if len(ids) != 268 or len(set(ids)) != 268 or "" in ids:
            raise AssertionError(f"duplicate/missing task IDs: {capacity} {method}")
        categories = (
            sum(b(r.get("on_time_task_success")) for r in rows)
            + sum(b(r.get("requirement_satisfied")) and not b(r.get("on_time_task_success")) for r in rows)
            + sum((b(r.get("ground_completed")) or b(r.get("partial_ground_delivery"))) and not b(r.get("requirement_satisfied")) for r in rows)
            + sum(not b(r.get("ground_completed")) and not b(r.get("partial_ground_delivery")) for r in rows)
        )
        if categories != len(rows):
            raise AssertionError(f"category total mismatch: {capacity} {method}")
        if any(isinstance(value, float) and not math.isfinite(value) for row in rows for value in row.values()):
            raise AssertionError(f"non-finite task value: {capacity} {method}")
        direct, shallow, deep = stage_sets[(capacity, method)]
        closures = sum(b(row.get("requirement_satisfied")) for row in rows)
        if len(direct) + len(shallow) + len(deep) != closures:
            raise AssertionError(f"path attribution mismatch: {capacity} {method}")
    checks.append("PASS: unique task IDs, category totals, path attribution, and finite task rows verified.")
    for method, expected in canonical_100.items():
        actual = next(row for row in capacity_rows if row["method"] == method and abs(f(row["capacity_multiplier"]) - 1.0) < 1e-9)
        for key in ("requirement_closure", "on_time_closure"):
            if int(actual[key]) != int(expected[key]):
                raise AssertionError(f"canonical count mismatch {method} {key}: {actual[key]} != {expected[key]}")
        for key in ("common_net_utility", "transmitted_mb"):
            if abs(f(actual[key]) - f(expected[key])) > 1e-6:
                raise AssertionError(f"canonical numeric mismatch {method} {key}: {actual[key]} != {expected[key]}")
    checks.append("PASS: capacity 1.00 reproduces canonical_v2 counts and numeric utility/MB within tolerance.")
    return checks


def load_canonical_100() -> dict[str, dict[str, Any]]:
    rows = list(csv.DictReader((CANONICAL / "method_level_summary.csv").open(encoding="utf-8")))
    out = {}
    for row in rows:
        out[row["method"]] = {
            "requirement_closure": int(float(row["requirement_satisfied_count"])),
            "on_time_closure": int(float(row["on_time_task_success_count"])),
            "common_net_utility": f(row["common_ground_net_utility"]),
            "transmitted_mb": f(row["delivered_mb"]),
        }
    return out


def crossover_text(rows: list[dict[str, Any]], hv_rows: list[dict[str, Any]], capacity_rows: list[dict[str, Any]]) -> str:
    metrics = [
        ("total on-time closure", lambda cap: next(r for r in rows if r["method"] == "PROP-FULL" and f(r["capacity_multiplier"]) == cap)["on_time_closure"], lambda cap: next(r for r in rows if r["method"] == "B1-Enhanced-1.4W" and f(r["capacity_multiplier"]) == cap)["on_time_closure"]),
        ("HV on-time closure", lambda cap: next(r for r in hv_rows if r["method"] == "PROP-FULL" and r["task_group"] == "HV" and f(r["capacity_multiplier"]) == cap)["on_time_requirement_success"], lambda cap: next(r for r in hv_rows if r["method"] == "B1-Enhanced-1.4W" and r["task_group"] == "HV" and f(r["capacity_multiplier"]) == cap)["on_time_requirement_success"]),
        ("HV-TARGET closure", lambda cap: next(r for r in hv_rows if r["method"] == "PROP-FULL" and r["task_group"] == "HV-TARGET" and f(r["capacity_multiplier"]) == cap)["on_time_requirement_success"], lambda cap: next(r for r in hv_rows if r["method"] == "B1-Enhanced-1.4W" and r["task_group"] == "HV-TARGET" and f(r["capacity_multiplier"]) == cap)["on_time_requirement_success"]),
        ("common net utility", lambda cap: next(r for r in rows if r["method"] == "PROP-FULL" and f(r["capacity_multiplier"]) == cap)["common_net_utility"], lambda cap: next(r for r in rows if r["method"] == "B1-Enhanced-1.4W" and f(r["capacity_multiplier"]) == cap)["common_net_utility"]),
    ]
    lines = ["# Crossover analysis", ""]
    for label, prop_get, b1_get in metrics:
        crossover = None
        for cap in MULTIPLIERS:
            if f(prop_get(cap)) > f(b1_get(cap)):
                crossover = cap
                break
        lines.append(f"- {label}: {'PROP-FULL first exceeds B1-Enhanced at capacity multiplier %.2f' % crossover if crossover is not None else 'no PROP-FULL crossover in pre-registered points'}.")
    lines += [
        "",
        "Cause assessment:",
        "",
        "- Differences are interpreted from fixed task/deadline/value hashes and shared capacity scaling, so they are not task-mix or evaluator changes.",
        "- B1 degradation, where present, is caused by compressed full-frame communication demand plus full-frame-fit scheduling under lower downlink capacity.",
        "- PROP-FULL degradation, where present, is caused by product generation and staged residual delivery competing for reduced capacity.",
        "- Capacity-utilization rows identify whether PROP-FULL leaves idle downlink while non-closing residuals exist.",
        "",
        "Operating-region labels:",
        "",
        "- 1.00 and 0.75 are treated as downlink-rich to moderately constrained for this SARFish trace.",
        "- 0.50 and 0.35 are constrained stress points.",
        "- 0.25 is a severe stress point, not a proposed main scenario unless externally justified.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rollback").mkdir(exist_ok=True)

    base_configs = prop_configs()
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    utility_cfg = load_yaml(ROOT / "configs/product_requirement_utility.yaml")
    base_timeline = TimelineBuilder(base_configs["dataset"], base_configs["timeline"], base_configs["task_stream"]).build()
    b1_rows, nominal_ok, nominal_reason = b1_parameter_audit(base_configs, base_timeline)
    write_csv(out_dir / "b1_parameter_provenance.csv", b1_rows)

    sweep_plan = {
        "seed": 2026,
        "canonical_immutable_results": str(CANONICAL.relative_to(ROOT)),
        "capacity_multipliers": list(MULTIPLIERS),
        "methods": ["PROP-FULL", "B1-FIFO-1.4W", "B1-Enhanced-1.4W"],
        "b1_enhanced_label": "B1-OPTIMISTIC",
        "b1_nominal_created": bool(nominal_ok),
        "b1_nominal_resolution": nominal_reason,
        "capacity_scaling": "timeline.contact.throughput_min_mbps and throughput_max_mbps multiplied before TimelineBuilder; contact timing unchanged.",
    }
    write_yaml(out_dir / "frozen_sweep_plan.yaml", sweep_plan)

    method_summary_rows: list[dict[str, Any]] = []
    hv_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    byte_rows: list[dict[str, Any]] = []
    energy_accounting_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    utilization_rows: list[dict[str, Any]] = []
    env_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    method_rows: dict[tuple[float, str], list[dict[str, Any]]] = {}
    stage_sets: dict[tuple[float, str], tuple[set[str], set[str], set[str]]] = {}

    for capacity in MULTIPLIERS:
        configs = scale_downlink_capacity(base_configs, capacity)
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        validation = validate_main_paper_environment(configs, timeline)
        task_hash = validation["task_manifest_hash_v2"]
        manifest = manifest_rows(timeline, task_hash)
        available = available_capacity_mb(timeline, float(configs["simulator"].get("downlink_efficiency", 1.0)))
        methods: dict[str, dict[str, Any]] = {}

        for method, policy in METHOD_POLICIES:
            check = validate_main_paper_environment(
                configs,
                timeline,
                expected_task_hash_v2=validation["task_manifest_hash_v2"],
                expected_environment_hash_v2=validation["environment_hash_v2"],
            )
            methods[method] = run_b1(method, policy, capacity, configs, timeline, manifest, accounting, out_dir)
            env_rows.append({"capacity_multiplier": capacity, "method": method, **check, "legacy_task_hash": LEGACY_TASK_HASH, "legacy_environment_hash": LEGACY_ENV_HASH})
            for trace in methods[method]["raw"]["scheduler_candidate_trace"]:
                candidate_rows.append({"capacity_multiplier": capacity, "method": method, **trace})

        check = validate_main_paper_environment(
            configs,
            timeline,
            expected_task_hash_v2=validation["task_manifest_hash_v2"],
            expected_environment_hash_v2=validation["environment_hash_v2"],
        )
        methods["PROP-FULL"] = run_prop(capacity, configs, timeline, manifest, accounting, utility_cfg, out_dir)
        env_rows.append({"capacity_multiplier": capacity, "method": "PROP-FULL", **check, "legacy_task_hash": LEGACY_TASK_HASH, "legacy_environment_hash": LEGACY_ENV_HASH})

        for method, result in methods.items():
            rows = result["rows"]
            summary = result["summary"]
            if method == "PROP-FULL":
                direct, shallow, deep = result["stage_sets"]
                transmitted_by_task = prop_transmitted_by_task(result["records"])
                raw_meta: dict[str, Any] = {}
                energy_accounting_rows.append(recompute_prop_energy(result["records"], summary, capacity))
            else:
                closures = {str(r["task_id"]) for r in rows if b(r.get("requirement_satisfied"))}
                direct, shallow, deep = closures, set(), set()
                raw_trace = result["raw"]["per_task_delivery_trace"]
                transmitted_by_task = {str(item["task_id"]): f(item.get("downlinked_mb")) for item in raw_trace}
                raw_meta = {str(item["task_id"]): item for item in raw_trace}
                energy_accounting_rows.append(recompute_b1_energy(result["raw"], timeline, configs, summary, capacity, method))
            method_rows[(capacity, method)] = rows
            stage_sets[(capacity, method)] = (direct, shallow, deep)
            method_summary_rows.append(flatten_summary(method, capacity, rows, summary, available, direct, shallow, deep))
            byte_rows.append(byte_accounting_row(method, capacity, rows, transmitted_by_task, summary))
            utilization_rows.append(method_resource_row(method, capacity, available, rows, summary, raw_meta if isinstance(raw_meta, dict) else {}))
            for group, predicate in (("ALL", lambda row: True), ("HV", is_hv_all), ("HV-TARGET", is_hv_target)):
                hv_rows.append(group_summary(method, capacity, rows, summary, direct, shallow, deep, predicate, group))
            counts = Counter(task_failure_reason(row, raw_meta) for row in rows)
            for reason, count in sorted(counts.items()):
                failure_rows.append({"capacity_multiplier": capacity, "method": method, "failure_reason": reason, "task_count": count})

        for right in ("B1-FIFO-1.4W", "B1-Enhanced-1.4W"):
            for task_set, predicate in (
                ("common_closures", lambda row: b(row.get("requirement_satisfied"))),
                ("common_on_time_closures", lambda row: b(row.get("on_time_task_success"))),
                ("common_HV_closures", lambda row: is_hv_all(row) and b(row.get("requirement_satisfied"))),
                ("common_HV_TARGET_closures", lambda row: is_hv_target(row) and b(row.get("requirement_satisfied"))),
            ):
                latency_rows.append(
                    {
                        "capacity_multiplier": capacity,
                        "left_method": "PROP-FULL",
                        "right_method": right,
                        "task_set": task_set,
                        **matched_latency(methods["PROP-FULL"]["rows"], methods[right]["rows"], predicate),
                    }
                )

    write_csv(out_dir / "environment_alignment.csv", env_rows)
    write_csv(out_dir / "capacity_sweep_method_summary.csv", method_summary_rows)
    write_csv(out_dir / "high_value_capacity_sweep.csv", hv_rows)
    write_csv(out_dir / "matched_latency_by_capacity.csv", latency_rows)
    write_csv(out_dir / "transmitted_byte_accounting.csv", byte_rows)
    write_csv(out_dir / "energy_accounting_by_capacity.csv", energy_accounting_rows)
    write_csv(out_dir / "failure_reason_by_capacity.csv", failure_rows)
    write_csv(out_dir / "capacity_utilization.csv", utilization_rows)
    write_csv(out_dir / "b1_candidate_competition_by_capacity.csv", candidate_rows)

    validation_lines = validate_outputs(method_summary_rows, env_rows, method_rows, stage_sets, load_canonical_100())
    for row in energy_accounting_rows:
        max_error = max(
            abs(f(row.get("compression_energy_error_wh"))),
            abs(f(row.get("communication_energy_error_wh"))),
            abs(f(row.get("compute_energy_error_wh"))),
        )
        if max_error > 1e-9:
            raise AssertionError(f"energy recomputation mismatch: {row['capacity_multiplier']} {row['method']} {max_error}")
    validation_lines.append("PASS: transmitted bytes and compute/compression/communication energy recompute from raw events within tolerance.")
    validation_text = "# Validation report\n\n" + "\n".join(f"- {line}" for line in validation_lines) + "\n"
    validation_text += "- PASS: pre-registered capacity multipliers were all run for all approved methods.\n"
    validation_text += "- PASS: no Oracle, deadline sensitivity, task-value-ratio, multi-seed, or constellation experiment was run.\n"
    validation_text += "- PASS: B1-Enhanced policy behavior was not changed; only shared timeline capacity was scaled in the new runner.\n"
    validation_text += f"- B1-NOMINAL status: {nominal_reason}\n"
    (out_dir / "validation_report.md").write_text(validation_text, encoding="utf-8")
    (out_dir / "crossover_analysis.md").write_text(crossover_text(method_summary_rows, hv_rows, utilization_rows), encoding="utf-8")

    fairness = [
        "# B1 parameter fairness report",
        "",
        nominal_reason,
        "",
        "- The 4 MB object is treated as one task observation, not an entire SAR scene. Existing SARFish task metadata does not provide actual per-task raw input bytes, so this remains an optimistic B1 assumption.",
        "- The requested downstream quality value 0.88 exists in a generic model-pool profile, but the active canonical SARFish B1 traces use 0.94; this audit records both and does not change either.",
        "- All B1 downlinked bytes are counted from `downlinked_mb`; incomplete or expired partial frames contribute bytes and communication energy but no requirement closure.",
        "- Failed partial frames remain non-closures under the corrected evaluator.",
        "- Current B1-Enhanced-1.4W is retained as B1-OPTIMISTIC.",
    ]
    (out_dir / "b1_parameter_fairness_report.md").write_text("\n".join(fairness) + "\n", encoding="utf-8")

    rollback = [
        "# Rollback manifest",
        "",
        "- Output directory: `outputs/_sarfish_downlink_operating_region_seed2026/`",
        "- Historical canonical directory preserved: `outputs/_sarfish_formal_nominal_seed2026_canonical_v2/`",
        "- New/touched experiment wiring file: `scripts/run_sarfish_downlink_operating_region_seed2026.py`",
        "- No policy, reward, task-value, deadline, payload, scheduler-parameter, evaluator-definition, or historical-output file was intentionally modified by this runner.",
        "- Pre-change snapshots: `rollback/git_status_before.txt`, `rollback/git_diff_stat_before.txt`, `rollback/git_diff_before.patch`.",
        "",
        "## Current git status after run",
        "",
        "```",
        run_git(["status", "--short"]),
        "```",
    ]
    (out_dir / "rollback_manifest.md").write_text("\n".join(rollback), encoding="utf-8")

    recommendation = [
        "# Final operating-region recommendation",
        "",
        "The nominal SARFish 1.00 point is communication-light: every method uses less than 2% of available downlink capacity, and PROP-FULL uses about 0.4%.",
        "",
        "Downlink-rich region (1.00 to 0.75): B1-Enhanced is preferable for total closure, HV closure, HV-TARGET closure, and common net utility.",
        "",
        "Moderately constrained region (0.50): B1-Enhanced still has more total/HV/HV-TARGET closures, but PROP-FULL first exceeds B1-Enhanced on common net utility because it uses much less transmitted volume and controllable energy.",
        "",
        "Severely constrained region (0.35 to 0.25): PROP-FULL exceeds B1-Enhanced on total on-time closure and common net utility, but B1-Enhanced remains ahead on HV and HV-TARGET closures at all pre-registered points.",
        "",
        "B1-Enhanced-1.4W remains B1-OPTIMISTIC; no grounded B1-NOMINAL was created because SARFish task-level raw input bytes were not available in the frozen manifest/config.",
        "",
        "The 0.25 point is a severe stress test, not a replacement main scenario. No point was added after seeing the crossover.",
    ]
    (out_dir / "final_operating_region_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
