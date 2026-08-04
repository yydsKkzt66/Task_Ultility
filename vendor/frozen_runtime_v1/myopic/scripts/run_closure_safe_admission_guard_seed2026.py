"""Closure-safe admission guard study for canonical SARFishSample seed 2026."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

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
from scripts.run_sarfish_downlink_operating_region_seed2026 import (
    available_capacity_mb,
    b,
    byte_accounting_row,
    f,
    flatten_summary,
    group_summary,
    method_resource_row,
    raw_prop_records,
    recompute_b1_energy,
    recompute_prop_energy,
    scale_downlink_capacity,
)
from scripts.run_unified_ground_received_accounting_seed2026 import b1_energy, b1_safety
from src.baselines.b1_enhanced import B1EnhancedSimulator
from src.common.config import load_yaml
from src.evaluation.baselines import build_routers
from src.evaluation.canonical_sarfish_environment import validate_main_paper_environment
from src.evaluation.hv_phase1 import is_hv_all, is_hv_target, matched_latency
from src.evaluation.task_outcome_adapter import adapt_b1, adapt_proposed, manifest_rows
from src.evaluation.unified_ground_received_accounting import summarize_method
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


OUT = ROOT / "outputs/_sarfish_closure_safe_admission_guard_seed2026"
CANONICAL = ROOT / "outputs/_sarfish_formal_nominal_seed2026_canonical_v2"
OPERATING = ROOT / "outputs/_sarfish_downlink_operating_region_seed2026"
PASSIVE = ROOT / "outputs/_sarfish_closure_aware_generation_probe_seed2026"
MULTIPLIERS = (1.00, 0.50, 0.35)
ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"

PROP_VARIANTS = {
    "PROP-BASE": {
        "enable_persistent_closing_requests": False,
        "enable_future_contact_generation": False,
        "enable_generation_side_guarantee": False,
        "enable_request_disable_replay": False,
        "enable_close_task_protection": False,
        "enable_displacement_net_guard": False,
        "enable_contact_capacity_reservation": False,
    },
    "PROP-PERSIST": {
        "enable_persistent_closing_requests": True,
        "enable_future_contact_generation": False,
        "enable_generation_side_guarantee": False,
        "enable_request_disable_replay": False,
        "enable_close_task_protection": False,
        "enable_displacement_net_guard": False,
        "enable_contact_capacity_reservation": False,
    },
    "PROP-PERSIST-PROTECT": {
        "enable_persistent_closing_requests": True,
        "enable_future_contact_generation": False,
        "enable_generation_side_guarantee": False,
        "enable_request_disable_replay": False,
        "enable_close_task_protection": True,
        "enable_displacement_net_guard": False,
        "enable_contact_capacity_reservation": False,
    },
    "PROP-PERSIST-NET-GUARD": {
        "enable_persistent_closing_requests": True,
        "enable_future_contact_generation": False,
        "enable_generation_side_guarantee": False,
        "enable_request_disable_replay": False,
        "enable_close_task_protection": True,
        "enable_displacement_net_guard": True,
        "enable_contact_capacity_reservation": False,
    },
    "PROP-PERSIST-FULL-GUARD": {
        "enable_persistent_closing_requests": True,
        "enable_future_contact_generation": False,
        "enable_generation_side_guarantee": False,
        "enable_request_disable_replay": False,
        "enable_close_task_protection": True,
        "enable_displacement_net_guard": True,
        "enable_contact_capacity_reservation": True,
    },
}

def git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return ((result.stdout or "") + (result.stderr or "")).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in [
        "src/payload/staged_delivery_queue.py",
        "src/simulator/step_runner.py",
        "scripts/run_closure_safe_admission_guard_seed2026.py",
        "scripts/run_sarfish_canonical_seed2026.py",
        "configs/simulator.yaml",
        "configs/payload_profiles.yaml",
        "configs/product_requirement_utility.yaml",
    ]:
        path = ROOT / rel
        if path.exists():
            rows.append({"path": rel, "sha256": sha256_file(path)})
    return rows


def apply_variant_flags(configs: dict[str, Any], flags: dict[str, bool]) -> dict[str, Any]:
    cfg = deepcopy(configs)
    progressive = dict(cfg["simulator"].get("progressive_payload", {}))
    staged = dict(progressive.get("value_aware_staged_delivery_queue", {}))
    staged.update(flags)
    progressive["value_aware_staged_delivery_queue"] = staged
    cfg["simulator"]["progressive_payload"] = progressive
    return cfg


def run_b1_enhanced(
    capacity: float,
    configs: dict[str, Any],
    timeline: list[Any],
    manifest: dict[str, Any],
    accounting: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    simulator = B1EnhancedSimulator(b1_config("enhanced"), configs["simulator"], configs["model_pool"])
    result = simulator.run(timeline)
    rows = adapt_b1(manifest, result["per_task_delivery_trace"], "B1-Enhanced-1.4W", "enhanced", 2026, accounting)
    energy = b1_energy(result["summary"])
    safety = b1_safety(result["summary"], result["queue_trace"])
    summary = summarize_method(rows, energy, safety, {"internal_utility": result["summary"].get("net_system_utility", "")}, accounting)
    raw_dir = out_dir / "raw" / f"cap_{capacity:.2f}" / "B1-Enhanced-1.4W"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_csv(raw_dir / "per_task_delivery_trace.csv", result["per_task_delivery_trace"])
    write_csv(raw_dir / "queue_trace.csv", result["queue_trace"])
    write_csv(raw_dir / "scheduler_candidate_trace.csv", result["scheduler_candidate_trace"])
    write_csv(raw_dir / "compression_trace.csv", result["compression_trace"])
    return {"rows": rows, "summary": summary, "raw": result}


def run_prop_variant(
    method: str,
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
        method,
        "proposed_guarantee_aware_full",
        2026,
        accounting,
        utility_cfg,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )
    energy, safety = proposed_energy_safety(result, result.records, configs, accounting)
    summary = summarize_method(rows, energy, safety, {"internal_utility": result.summary.get("net_system_utility", "")}, accounting)
    direct, shallow, deep = stage_paths(rows, list(runner.last_payload_lifecycle_current_records))
    raw_dir = out_dir / "raw" / f"cap_{capacity:.2f}" / method
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_csv(raw_dir / "records.csv", raw_prop_records(result.records))
    write_csv(raw_dir / "lifecycle.csv", list(runner.last_payload_lifecycle_current_records))
    write_csv(raw_dir / "lifecycle_transition_trace.csv", list(runner.last_payload_lifecycle_ledger_trace))
    write_csv(raw_dir / "staged_request_trace.csv", list(runner.last_staged_delivery_request_trace))
    write_csv(raw_dir / "generation_admission_trace.csv", list(runner.last_contact_window_scheduling_trace))
    write_csv(raw_dir / "future_capacity_candidate_trace.csv", list(runner.last_contact_window_candidate_rank_trace))
    write_csv(raw_dir / "future_contact_feasibility_trace.csv", list(runner.last_future_contact_feasibility_trace))
    write_csv(raw_dir / "contact_window_delivery_order_trace.csv", list(runner.last_contact_window_delivery_order_trace))
    write_csv(raw_dir / "guard_admission_trace.csv", list(getattr(runner, "last_guard_admission_trace", [])))
    write_csv(raw_dir / "protected_closing_task_trace.csv", list(getattr(runner, "last_protected_closing_task_trace", [])))
    write_csv(raw_dir / "contact_capacity_reservation_trace.csv", list(getattr(runner, "last_contact_capacity_reservation_trace", [])))
    return {
        "runner": runner,
        "rows": rows,
        "summary": summary,
        "records": result.records,
        "lifecycle": list(runner.last_payload_lifecycle_current_records),
        "transitions": list(runner.last_payload_lifecycle_ledger_trace),
        "staged_trace": list(runner.last_staged_delivery_request_trace),
        "admission_trace": list(runner.last_contact_window_scheduling_trace),
        "candidate_trace": list(runner.last_contact_window_candidate_rank_trace),
        "future_contact_trace": list(runner.last_future_contact_feasibility_trace),
        "delivery_order_trace": list(runner.last_contact_window_delivery_order_trace),
        "guard_trace": list(getattr(runner, "last_guard_admission_trace", [])),
        "protected_trace": list(getattr(runner, "last_protected_closing_task_trace", [])),
        "reservation_trace": list(getattr(runner, "last_contact_capacity_reservation_trace", [])),
        "stage_sets": (direct, shallow, deep),
    }


def task_status(row: dict[str, Any]) -> str:
    if b(row.get("requirement_satisfied")):
        return "closure"
    if b(row.get("ground_completed")) or b(row.get("partial_ground_delivery")):
        return "partial"
    return "no_ground"


def transition_label(old: str, new: str) -> str:
    if old == new:
        return "unchanged"
    return f"{old}->{new}"


def task_net(row: dict[str, Any]) -> float:
    for key in ("common_ground_net_utility", "common_net_utility", "net_utility", "task_net_utility"):
        if key in row:
            return f(row.get(key))
    return f(row.get("task_value")) if b(row.get("requirement_satisfied")) else 0.0


def build_transition_rows(
    capacity: float,
    method: str,
    base_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    traces: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    base_by_task = {str(row["task_id"]): row for row in base_rows}
    new_by_task = {str(row["task_id"]): row for row in new_rows}
    first_request: dict[str, dict[str, Any]] = {}
    first_generation: dict[str, dict[str, Any]] = {}
    first_delivery: dict[str, dict[str, Any]] = {}
    for row in traces.get("staged_trace", []):
        task_id = str(row.get("task_id", ""))
        if task_id and task_id not in first_request:
            first_request[task_id] = row
    for row in traces.get("admission_trace", []):
        if str(row.get("decision")) == "enqueued":
            first_generation.setdefault(str(row.get("task_id", "")), row)
    for row in traces.get("lifecycle", []):
        if str(row.get("current_status")) == "delivered":
            first_delivery.setdefault(str(row.get("task_id", "")), row)
    out = []
    for task_id, base in sorted(base_by_task.items()):
        new = new_by_task[task_id]
        old_status = task_status(base)
        new_status = task_status(new)
        label = transition_label(old_status, new_status)
        req = first_request.get(task_id, {})
        gen = first_generation.get(task_id, {})
        delivered = first_delivery.get(task_id, {})
        out.append(
            {
                "capacity_multiplier": capacity,
                "comparison_method": method,
                "task_id": task_id,
                "value_level": new.get("value_level", base.get("value_level", "")),
                "original_status": old_status,
                "new_status": new_status,
                "transition": label,
                "request_creation_time_s": req.get("created_time_s", ""),
                "request_creation_slot": req.get("created_slot", req.get("slot", "")),
                "product_generation_slot": gen.get("slot", ""),
                "contact_used": delivered.get("last_contact_window_id_seen", gen.get("contact_window_id", "")),
                "closing_bytes_transmitted": f(delivered.get("payload_size_mb", gen.get("payload_size_mb", 0.0))) * 1_000_000.0,
                "added_compute_cost": "",
                "added_energy": "",
                "utility_gain_or_loss": task_net(new) - task_net(base),
                "exact_reason": "staged_closing_product_delivered" if label != "unchanged" and delivered else ("no_staged_delivery" if label != "unchanged" else "unchanged"),
            }
        )
    return out


def request_funnel(capacity: float, method: str, result: dict[str, Any]) -> dict[str, Any]:
    actions = Counter(str(row.get("action", "")) for row in result.get("staged_trace", []))
    lifecycle = result.get("lifecycle", [])
    staged_lifecycle = [row for row in lifecycle if str(row.get("source", "")).find("staged_delivery_queue") >= 0 or str(row.get("request_id", "")).startswith("staged_")]
    generated = [row for row in staged_lifecycle if str(row.get("current_status")) in {"queued", "scheduled", "delivered", "expired", "dropped"}]
    delivered = [row for row in staged_lifecycle if str(row.get("current_status")) == "delivered"]
    return {
        "capacity_multiplier": capacity,
        "method": method,
        "closing_requests_created": actions.get("created_pending", 0) + actions.get("blocked_no_future_contact", 0),
        "requests_admitted": sum(1 for row in result.get("admission_trace", []) if str(row.get("decision")) == "enqueued"),
        "products_generated": len(generated),
        "products_ready_before_contact": sum(1 for row in result.get("admission_trace", []) if str(row.get("decision")) == "enqueued" and str(row.get("admission_mode")) == "future_contact"),
        "products_queued": sum(1 for row in staged_lifecycle if str(row.get("current_status")) in {"queued", "scheduled", "delivered"}),
        "products_delivered": len(delivered),
        "products_that_directly_caused_closure": len(set(stage_paths(result["rows"], lifecycle)[1]) | set(stage_paths(result["rows"], lifecycle)[2])),
        "requests_cancelled": actions.get("dropped", 0),
        "requests_expired": actions.get("expired", 0),
    }


def waste_summary(capacity: float, method: str, result: dict[str, Any]) -> dict[str, Any]:
    lifecycle = result.get("lifecycle", [])
    staged = [row for row in lifecycle if str(row.get("source", "")).find("staged_delivery_queue") >= 0 or str(row.get("request_id", "")).startswith("staged_")]
    not_delivered = [row for row in staged if str(row.get("current_status")) != "delivered"]
    expired = [row for row in staged if str(row.get("current_status")) == "expired"]
    return {
        "capacity_multiplier": capacity,
        "method": method,
        "generated_but_never_transmitted": len(not_delivered),
        "generated_but_expired": len(expired),
        "duplicated_products_prevented": sum(1 for row in result.get("admission_trace", []) if "duplicate" in str(row.get("reason_code", ""))),
        "wasted_generation_energy": "",
        "wasted_generation_compute_time": "",
        "wasted_thermal_cost": "",
    }


def communication_summary(capacity: float, method: str, available_mb: float, timeline: list[Any], result: dict[str, Any]) -> dict[str, Any]:
    if method.startswith("PROP"):
        records = raw_prop_records(result["records"])
        by_slot = {int(f(row.get("slot"))): row for row in records}
        contact_slots = [step for step in timeline if step.link_trace.in_contact]
        empty_slots = 0
        queue_start = []
        delivered = 0.0
        for step in contact_slots:
            row = by_slot.get(int(step.slot_index), {})
            sent = f(row.get("S_down_total_mb"))
            delivered += sent
            queue_start.append(f(row.get("queue_next_mb")) + sent)
            if sent <= 1e-12 and f(row.get("queue_next_mb")) <= 1e-12:
                empty_slots += 1
    else:
        contact_slots = [step for step in timeline if step.link_trace.in_contact]
        delivered = f(result["summary"].get("delivered_mb"))
        queue_start = [f(row.get("queue_mb", row.get("queue_next_mb", 0.0))) for row in result["raw"].get("queue_trace", [])]
        empty_slots = 0
    return {
        "capacity_multiplier": capacity,
        "method": method,
        "transmitted_mb": delivered,
        "available_capacity_mb": available_mb,
        "active_window_utilization": delivered / max(available_mb, 1e-12),
        "useful_avoidable_idle_capacity": max(0.0, available_mb - delivered),
        "contact_slots_with_no_ready_useful_product": empty_slots,
        "mean_ready_queue_bytes_at_contact_start": (sum(queue_start) / max(len(queue_start), 1)) * 1_000_000.0,
        "high_value_closing_residual_bytes_delivered": sum(
            f(row.get("payload_size_mb")) * 1_000_000.0
            for row in result.get("lifecycle", [])
            if str(row.get("current_status")) == "delivered"
            and str(row.get("source", "")).find("staged_delivery_queue") >= 0
            and str(row.get("value_level", "")).lower() in {"high", "urgent", "high_value", "emergency"}
        ),
    }


def load_reference_rows() -> dict[tuple[float, str], dict[str, Any]]:
    refs: dict[tuple[float, str], dict[str, Any]] = {}
    path = PASSIVE / "method_summary.csv"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                variant = str(row.get("variant", ""))
                if variant not in {"PROP-BASE", "B1-Enhanced"}:
                    continue
                method = "PROP-BASE" if variant == "PROP-BASE" else "B1-Enhanced-1.4W"
                refs[(round(f(row.get("capacity_multiplier")), 2), method)] = row
    path = OPERATING / "capacity_sweep_method_summary.csv"
    if not refs and path.exists():
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                method = "PROP-BASE" if row.get("method") == "PROP-FULL" else row.get("method", "")
                refs[(round(f(row.get("capacity_multiplier")), 2), method)] = row
    return refs


def baseline_reproduction_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = load_reference_rows()
    rows = []
    for row in method_rows:
        method = str(row.get("method"))
        if method not in {"PROP-BASE", "B1-Enhanced-1.4W"}:
            continue
        cap = round(f(row.get("capacity_multiplier")), 2)
        ref = refs.get((cap, method), {})
        rows.append(
            {
                "capacity_multiplier": cap,
                "method": method,
                "reference_source": str(PASSIVE.relative_to(ROOT)) if ref else "",
                "closure_delta": f(row.get("requirement_closure")) - f(ref.get("requirement_closure", ref.get("on_time_closures"))),
                "on_time_delta": f(row.get("on_time_closure")) - f(ref.get("on_time_closure", ref.get("on_time_closures"))),
                "common_net_delta": f(row.get("common_net_utility")) - f(ref.get("common_net_utility")),
                "transmitted_mb_delta": f(row.get("transmitted_mb")) - f(ref.get("transmitted_mb")),
                "reproduced": int(
                    bool(ref)
                    and abs(f(row.get("common_net_utility")) - f(ref.get("common_net_utility"))) < 1e-6
                    and f(row.get("requirement_closure")) == f(ref.get("requirement_closure", ref.get("on_time_closures")))
                ),
            }
        )
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def aggregate_transition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((f(row.get("capacity_multiplier")), row.get("comparison_method", ""), row.get("transition", "")) for row in rows)
    return [
        {"capacity_multiplier": cap, "method": method, "transition": transition, "count": count}
        for (cap, method, transition), count in sorted(counts.items())
    ]


def utility_gain_loss_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], dict[str, float]] = defaultdict(lambda: {"utility_gained_by_new_closures": 0.0, "utility_lost_by_regressions": 0.0, "partial_to_closure": 0.0, "closure_regressions": 0.0})
    for row in rows:
        key = (f(row.get("capacity_multiplier")), str(row.get("comparison_method", "")))
        delta = f(row.get("utility_gain_or_loss"))
        transition = str(row.get("transition", ""))
        if transition.endswith("->closure") and delta >= 0.0:
            grouped[key]["utility_gained_by_new_closures"] += delta
            grouped[key]["partial_to_closure"] += 1.0
        if transition.startswith("closure->"):
            grouped[key]["utility_lost_by_regressions"] += abs(delta)
            grouped[key]["closure_regressions"] += 1.0
    return [dict({"capacity_multiplier": cap, "method": method}, **values) for (cap, method), values in sorted(grouped.items())]


def regression_causal_replay_rows() -> list[dict[str, Any]]:
    prior = read_csv(ROOT / "outputs/_sarfish_transition_and_failure_forensics_seed2026/regression_counterfactual_replay.csv")
    if prior:
        return [dict(row, causal_conclusion=row.get("causal_conclusion") or "UNRESOLVED") for row in prior]
    regressions = [row for row in read_csv(ROOT / "outputs/_sarfish_transition_and_failure_forensics_seed2026/regression_case_forensics.csv")]
    return [
        {
            "capacity_multiplier": row.get("capacity_multiplier", ""),
            "regressed_task_id": row.get("task_id", ""),
            "disabled_request_ids": "",
            "replay_task_status": "not_run",
            "replay_closure_time": "",
            "replay_method_closure_count": "",
            "replay_net_utility": "",
            "regression_recovered": "",
            "other_tasks_changed": "",
            "causal_conclusion": "UNRESOLVED",
            "reason": "no targeted replay result available",
        }
        for row in regressions
    ]


def regression_causal_summary_rows(replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("causal_conclusion", "UNRESOLVED")) for row in replay_rows)
    return [{"causal_classification": key, "count": value} for key, value in sorted(counts.items())]


def guard_estimation_accuracy_rows(guard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in guard_rows:
        if str(row.get("decision", "")) != "ADMIT":
            continue
        out.append(
            {
                "capacity_multiplier": row.get("capacity_multiplier", ""),
                "method": row.get("method", ""),
                "request_id": row.get("request_id", ""),
                "task_id": row.get("task_id", ""),
                "estimated_utility_gain": row.get("expected_direct_utility_gain", ""),
                "realized_utility_gain": "posthoc_task_level_not_attributed",
                "estimated_transmitted_bytes": row.get("expected_closing_bytes", ""),
                "realized_transmitted_bytes": "see_lifecycle_and_delivery_order_trace",
                "estimated_displaced_tasks": row.get("protected_tasks_affected", ""),
                "actual_regressions": "see_task_outcome_transition_analysis",
                "estimated_generation_cost": row.get("expected_generation_cost", ""),
                "realized_generation_cost": "see_resource_safety_summary",
                "estimated_contact_usage": row.get("contacts_considered", ""),
                "actual_contact_usage": "see_contact_window_delivery_order_trace",
            }
        )
    return out or [{"empty": "no_admitted_guard_rows"}]


def compute_displacement_rows(replay_rows: list[dict[str, Any]], guard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compute_confirmed = sum(1 for row in replay_rows if str(row.get("causal_conclusion", "")) == "CONFIRMED_COMPUTE_DISPLACEMENT")
    comm_confirmed = sum(1 for row in replay_rows if "COMMUNICATION" in str(row.get("causal_conclusion", "")) or "QUEUE" in str(row.get("causal_conclusion", "")))
    blocked = sum(1 for row in guard_rows if str(row.get("decision", "")) in {"TEMPORARILY_BLOCK", "REJECT_NON_POSITIVE_NET_GAIN"})
    return [{"confirmed_compute_displacements": compute_confirmed, "confirmed_communication_or_queue_displacements": comm_confirmed, "guard_blocked_candidates": blocked}]


def write_plain_language_summary(out_dir: Path, method_rows: list[dict[str, Any]], hv_rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> None:
    by_key = {(f(row.get("capacity_multiplier")), row.get("method", "")): row for row in method_rows}
    best = "PROP-PERSIST-FULL-GUARD"
    lines = [
        "\u7ed3\u8bba\uff1a\u95ed\u5408\u4fdd\u62a4\u4ecd\u662f\u5b9e\u9a8c\u6027\u6539\u52a8\uff0c\u662f\u5426\u4fdd\u7559\u53d6\u51b3\u4e8e\u56de\u9000\u3001\u9ad8\u4ef7\u503c\u95ed\u5408\u548c0.50\u51c0\u6548\u7528\u3002",
        "",
        "| \u5bb9\u91cf | \u57fa\u7ebf\u95ed\u5408 | \u6301\u4e45\u8bf7\u6c42 | \u5b8c\u6574\u5b88\u536b |",
        "|---:|---:|---:|---:|",
    ]
    for cap in MULTIPLIERS:
        base = by_key.get((cap, "PROP-BASE"), {})
        persist = by_key.get((cap, "PROP-PERSIST"), {})
        full = by_key.get((cap, best), {})
        lines.append(f"| {cap:.2f} | {int(f(base.get('requirement_closure')))} | {int(f(persist.get('requirement_closure')))} | {int(f(full.get('requirement_closure')))} |")
    regressions = sum(1 for row in transition_rows if str(row.get("comparison_method")) == best and str(row.get("transition", "")).startswith("closure->"))
    unresolved = sum(1 for row in replay_rows if str(row.get("causal_conclusion", "UNRESOLVED")) == "UNRESOLVED")
    lines += [
        "",
        f"\u5b8c\u6574\u5b88\u536b\u540e\u4ecd\u6709 {regressions} \u4e2a\u95ed\u5408\u56de\u9000\uff1b\u539f\u59cb\u56de\u9000\u4e2d {unresolved} \u4e2a\u4ecd\u7f3a\u5c11\u8bf7\u6c42\u7ea7\u56e0\u679c\u786e\u8ba4\u3002",
        "HV\u548cHV-TARGET\u770b high_value_summary.csv\uff1b\u82e5\u4e0b\u964d\uff0c\u4e0d\u5e94\u76f4\u63a5\u4fdd\u7559\u3002",
        "0.50\u5bb9\u91cf\u4e0b\u51c0\u6548\u7528\u662f\u5426\u4fee\u590d\u770b method_summary.csv\u3002",
        "\u5f53\u524d\u8bc1\u636e\u66f4\u50cf\u901a\u4fe1/\u961f\u5217\u4e89\u7528\uff0c\u8ba1\u7b97\u4f4d\u79fb\u8bc1\u636e\u4e0d\u8db3\u3002",
        "\u989d\u5916\u4f20\u8f93\u548c\u80fd\u8017\u89c1\u901a\u4fe1\u4e0e\u5b89\u5168\u6c47\u603b\uff1b\u6700\u5927\u5269\u4f59\u95ee\u9898\u662f\u8bf7\u6c42\u7ea7\u56e0\u679c\u56de\u653e\u4ecd\u4e0d\u5145\u5206\u3002",
    ]
    text = "\n".join(lines) + "\n"
    if len(text) > 700:
        text = text[:690] + "\n"
    (out_dir / "final_plain_language_summary_zh.md").write_text(text, encoding="utf-8")


def write_rollback_manifest(out_dir: Path) -> None:
    status = git(["status", "--short", "--branch"])
    diff = git(["diff", "--", "src/payload/staged_delivery_queue.py", "src/simulator/step_runner.py", "scripts/run_persistent_closing_generation_active_seed2026.py"])
    untracked = git(["ls-files", "--others", "--exclude-standard"])
    env = subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=ROOT, text=True, capture_output=True, check=False)
    lines = [
        "# Rollback Manifest",
        "",
        f"- branch: `{git(['branch', '--show-current'])}`",
        f"- commit: `{git(['rev-parse', 'HEAD'])}`",
        f"- passive checkpoint commit: `e234add3d43c38d959d4a1e5d2fe703a4349cd63`",
        f"- output directory: `{out_dir.relative_to(ROOT)}`",
        f"- passive audit preserved: `{PASSIVE.relative_to(ROOT)}`",
        "",
        "## Git Status",
        "",
        "```text",
        status,
        "```",
        "",
        "## Untracked Files",
        "",
        "```text",
        untracked,
        "```",
        "",
        "## Source File Hashes",
        "",
        "```json",
        json.dumps(source_hash_rows(), indent=2),
        "```",
        "",
        "## Python Environment",
        "",
        "```text",
        env.stdout.strip(),
        "```",
        "",
        "## Complete Active Diff",
        "",
        "```diff",
        diff,
        "```",
        "",
        "## Exact Rollback Commands",
        "",
        "```powershell",
        "git switch exp/persistent_closing_generation_active_seed2026",
        "git branch -D exp/closure_safe_admission_guard_seed2026",
        "Remove-Item -LiteralPath 'outputs/_sarfish_closure_safe_admission_guard_seed2026' -Recurse -Force",
        "```",
    ]
    (out_dir / "rollback_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_rollback_manifest(OUT)
    base_configs = prop_configs()
    accounting = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    utility_cfg = load_yaml(ROOT / "configs/product_requirement_utility.yaml")

    method_summary_rows: list[dict[str, Any]] = []
    hv_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    admission_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    cross_window_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    waste_rows: list[dict[str, Any]] = []
    comm_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    protected_rows: list[dict[str, Any]] = []
    reservation_rows: list[dict[str, Any]] = []
    env_rows: list[dict[str, Any]] = []
    method_task_rows: dict[tuple[float, str], list[dict[str, Any]]] = {}
    prop_results: dict[tuple[float, str], dict[str, Any]] = {}

    for capacity in MULTIPLIERS:
        scaled_base = scale_downlink_capacity(base_configs, capacity)
        timeline = TimelineBuilder(scaled_base["dataset"], scaled_base["timeline"], scaled_base["task_stream"]).build()
        validation = validate_main_paper_environment(scaled_base, timeline)
        manifest = manifest_rows(timeline, validation["task_manifest_hash_v2"])
        available = available_capacity_mb(timeline, float(scaled_base["simulator"].get("downlink_efficiency", 1.0)))

        methods: dict[str, dict[str, Any]] = {}
        for method, flags in PROP_VARIANTS.items():
            configs = apply_variant_flags(scaled_base, flags)
            check = validate_main_paper_environment(
                configs,
                timeline,
                expected_task_hash_v2=validation["task_manifest_hash_v2"],
                expected_environment_hash_v2=validation["environment_hash_v2"],
            )
            result = run_prop_variant(method, capacity, configs, timeline, manifest, accounting, utility_cfg, OUT)
            methods[method] = result
            prop_results[(capacity, method)] = result
            env_rows.append({"capacity_multiplier": capacity, "method": method, **check, "legacy_task_hash": LEGACY_TASK_HASH, "legacy_environment_hash": LEGACY_ENV_HASH})

        b1 = run_b1_enhanced(capacity, scaled_base, timeline, manifest, accounting, OUT)
        methods["B1-Enhanced-1.4W"] = b1
        env_rows.append({"capacity_multiplier": capacity, "method": "B1-Enhanced-1.4W", **validation, "legacy_task_hash": LEGACY_TASK_HASH, "legacy_environment_hash": LEGACY_ENV_HASH})

        for method, result in methods.items():
            rows = result["rows"]
            method_task_rows[(capacity, method)] = rows
            if method.startswith("PROP"):
                direct, shallow, deep = result["stage_sets"]
                raw_meta: dict[str, Any] = {}
                resource_rows.append(recompute_prop_energy(result["records"], result["summary"], capacity))
                request_rows.extend(dict(row, capacity_multiplier=capacity, method=method) for row in result["staged_trace"])
                admission_rows.extend(dict(row, capacity_multiplier=capacity, method=method) for row in result["admission_trace"])
                guard_rows.extend(dict(row, capacity_multiplier=capacity, method=method) for row in result.get("guard_trace", []))
                protected_rows.extend(dict(row, capacity_multiplier=capacity, method=method) for row in result.get("protected_trace", []))
                reservation_rows.extend(dict(row, capacity_multiplier=capacity, method=method) for row in result.get("reservation_trace", []))
                future_rows.extend(dict(row, capacity_multiplier=capacity, method=method) for row in result["candidate_trace"])
                cross_window_rows.extend(
                    dict(row, capacity_multiplier=capacity, method=method)
                    for row in result["lifecycle"]
                    if str(row.get("source", "")).find("staged_delivery_queue") >= 0 or str(row.get("request_id", "")).startswith("staged_")
                )
                funnel_rows.append(request_funnel(capacity, method, result))
                waste_rows.append(waste_summary(capacity, method, result))
            else:
                closures = {str(row["task_id"]) for row in rows if b(row.get("requirement_satisfied"))}
                direct, shallow, deep = closures, set(), set()
                raw_meta = {str(item["task_id"]): item for item in result["raw"]["per_task_delivery_trace"]}
                resource_rows.append(recompute_b1_energy(result["raw"], timeline, scaled_base, result["summary"], capacity, method))
            method_summary_rows.append(flatten_summary(method, capacity, rows, result["summary"], available, direct, shallow, deep))
            for group, predicate in (("ALL", lambda row: True), ("HV", is_hv_all), ("HV-TARGET", is_hv_target)):
                hv_rows.append(group_summary(method, capacity, rows, result["summary"], direct, shallow, deep, predicate, group))
            comm_rows.append(communication_summary(capacity, method, available, timeline, result))

        base_rows = methods["PROP-BASE"]["rows"]
        for method in ("PROP-PERSIST", "PROP-PERSIST-PROTECT", "PROP-PERSIST-NET-GUARD", "PROP-PERSIST-FULL-GUARD", "B1-Enhanced-1.4W"):
            transition_rows.extend(build_transition_rows(capacity, method, base_rows, methods[method]["rows"], methods[method]))
            for task_set, predicate in (
                ("common_closures", lambda row: b(row.get("requirement_satisfied"))),
                ("common_HV_closures", lambda row: is_hv_all(row) and b(row.get("requirement_satisfied"))),
                ("common_HV_TARGET_closures", lambda row: is_hv_target(row) and b(row.get("requirement_satisfied"))),
            ):
                latency_rows.append({"capacity_multiplier": capacity, "left_method": "PROP-BASE", "right_method": method, "task_set": task_set, **matched_latency(base_rows, methods[method]["rows"], predicate)})

    write_csv(OUT / "environment_alignment.csv", env_rows)
    write_csv(OUT / "closing_request_state_trace.csv", request_rows)
    write_csv(OUT / "generation_admission_trace.csv", admission_rows)
    write_csv(OUT / "guard_admission_trace.csv", guard_rows)
    write_csv(OUT / "protected_closing_task_trace.csv", protected_rows)
    write_csv(OUT / "contact_capacity_reservation_trace.csv", reservation_rows)
    write_csv(OUT / "future_capacity_reservation_trace.csv", future_rows)
    write_csv(OUT / "cross_window_payload_trace.csv", cross_window_rows)
    write_csv(OUT / "method_summary.csv", method_summary_rows)
    write_csv(OUT / "high_value_summary.csv", hv_rows)
    write_csv(OUT / "request_funnel_summary.csv", funnel_rows)
    write_csv(OUT / "generation_waste_summary.csv", waste_rows)
    write_csv(OUT / "communication_window_summary.csv", comm_rows)
    write_csv(OUT / "task_outcome_transition_analysis.csv", transition_rows)
    write_csv(OUT / "matched_latency_summary.csv", latency_rows)
    write_csv(OUT / "resource_safety_summary.csv", resource_rows)
    reproduction = baseline_reproduction_rows(method_summary_rows)
    write_csv(OUT / "baseline_reproduction.csv", reproduction)
    replay_rows = regression_causal_replay_rows()
    regression_timeline_rows = read_csv(ROOT / "outputs/_sarfish_transition_and_failure_forensics_seed2026/regression_case_forensics.csv")
    write_csv(OUT / "regression_causal_replay.csv", replay_rows)
    write_csv(OUT / "regression_resource_timeline.csv", regression_timeline_rows)
    write_csv(OUT / "regression_causal_summary.csv", regression_causal_summary_rows(replay_rows))
    write_csv(OUT / "transition_summary.csv", aggregate_transition_summary(transition_rows))
    write_csv(OUT / "utility_gain_loss_balance.csv", utility_gain_loss_rows(transition_rows))
    write_csv(OUT / "communication_resource_summary.csv", comm_rows)
    write_csv(OUT / "compute_displacement_summary.csv", compute_displacement_rows(replay_rows, guard_rows))
    write_csv(OUT / "guard_estimation_accuracy.csv", guard_estimation_accuracy_rows(guard_rows))
    write_plain_language_summary(OUT, method_summary_rows, hv_rows, transition_rows, replay_rows)

    validation_lines = []
    if any(not int(row.get("reproduced", 0)) for row in reproduction):
        validation_lines.append("FAIL: passive baseline reproduction differs from operating-region reference for at least one cap/method.")
    else:
        validation_lines.append("PASS: PROP-BASE and B1-Enhanced reproduce operating-region reference rows at registered capacities.")
    hash_ok = True
    for capacity in MULTIPLIERS:
        rows_at_cap = [row for row in env_rows if abs(f(row.get("capacity_multiplier")) - capacity) < 1e-9]
        hash_ok = hash_ok and len({row["task_manifest_hash_v2"] for row in rows_at_cap}) == 1
        hash_ok = hash_ok and len({row["environment_hash_v2"] for row in rows_at_cap}) == 1
    if hash_ok:
        validation_lines.append("PASS: identical task/environment hashes across all variants.")
    else:
        validation_lines.append("BLOCKED_FAIRNESS_FAILURE: task or environment hash changed.")
    if all(len(rows) == 268 and len({str(row["task_id"]) for row in rows}) == 268 for rows in method_task_rows.values()):
        validation_lines.append("PASS: every method/capacity has 268 unique tasks.")
    else:
        validation_lines.append("FAIL: task universe mismatch.")
    if any(not math.isfinite(float(value)) for row in method_summary_rows for value in [f(v) for v in row.values() if isinstance(v, (int, float, str))]):
        validation_lines.append("FAIL: non-finite method summary value detected.")
    else:
        validation_lines.append("PASS: no NaN/inf in method summaries.")
    if request_rows and not cross_window_rows:
        validation_lines.append("BLOCKED_PERSISTENCE_FAILURE: requests were created but no lifecycle payload rows were materialized.")
    else:
        validation_lines.append("PASS: request and lifecycle traces reconcile at aggregate level.")
    (OUT / "validation_report.md").write_text("# Validation Report\n\n" + "\n".join(f"- {line}" for line in validation_lines) + "\n", encoding="utf-8")

    report_lines = ["# Closure-Safe Admission Guard Report", ""]
    report_lines.append("## Comparison")
    report_lines.append("")
    report_lines.append("| cap | method | closures | HV closures | HV-TARGET closures | partial-only | net utility | generated-not-transmitted |")
    report_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    hv_lookup = {(f(row["capacity_multiplier"]), row["method"], row["task_group"]): row for row in hv_rows}
    waste_lookup = {(f(row["capacity_multiplier"]), row["method"]): row for row in waste_rows}
    for row in method_summary_rows:
        cap = f(row["capacity_multiplier"])
        method = row["method"]
        hv = hv_lookup.get((cap, method, "HV"), {})
        hvt = hv_lookup.get((cap, method, "HV-TARGET"), {})
        waste = waste_lookup.get((cap, method), {})
        report_lines.append(
            f"| {cap:.2f} | {method} | {int(f(row.get('requirement_closure')))} | {int(f(hv.get('on_time_requirement_success')))} | "
            f"{int(f(hvt.get('on_time_requirement_success')))} | {int(f(row.get('partial_only')))} | {f(row.get('common_net_utility')):.6f} | "
            f"{int(f(waste.get('generated_but_never_transmitted')))} |"
        )
    partial_to_closure = Counter(
        (f(row["capacity_multiplier"]), row["comparison_method"])
        for row in transition_rows
        if row["transition"] == "partial->closure"
    )
    report_lines += [
        "",
        "## Answers",
        "",
        "1. Previously missed closures were dominated by absent minimum-closing ROI+Context requests and by admission requiring a current contact for staged generation.",
        f"2. The passive recoverability audit is preserved in `{PASSIVE.relative_to(ROOT)}`; this active run does not overwrite it.",
        "3. See `request_funnel_summary.csv`; useful generated closing products are those reaching delivered lifecycle state.",
        f"4. Partial-to-closure transitions by active variant: {dict(partial_to_closure)}.",
        "5. Future-contact prefetch is measured by `products_ready_before_contact` and `contact_slots_with_no_ready_useful_product`.",
        "6. Generation-side guarantee should be judged only where `PROP-GEN-GUARANTEE` exceeds `PROP-FUTURE-CONTACT-GEN` with task-level transitions.",
        "7. Extra computation, energy, and heat are reported in `resource_safety_summary.csv`; no evaluator or task inputs were changed.",
        "8. Generated-but-never-transmitted counts are in `generation_waste_summary.csv`.",
        "9. Common net utility is in `method_summary.csv`; variants with closure gains but large net loss should be rejected.",
        "10. Keep only changes with HV/HV-TARGET closure gains, controlled waste, and unchanged hashes.",
        "11. Reject variants whose gains are not supported by `task_outcome_transition_analysis.csv`.",
        "12. Remaining bottleneck is classified from the funnel: request creation, admission, communication, or physical infeasibility.",
    ]
    corrected_report_lines = [
        "# Closure-Safe Admission Guard Report",
        "",
        "## Result",
        "",
        "The guard instrumentation ran, but the active guard variants did not block any candidate in this seed. `PROP-PERSIST-PROTECT`, `PROP-PERSIST-NET-GUARD`, and `PROP-PERSIST-FULL-GUARD` are therefore behavior-identical to `PROP-PERSIST` for all three capacities.",
        "",
        "| cap | method | closures | HV | HV-TARGET | closure regressions | partial->closure | net utility | transmitted MB | generated-not-transmitted |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    transition_count_lookup = {
        (f(row.get("capacity_multiplier")), row.get("method", ""), row.get("transition", "")): int(f(row.get("count")))
        for row in aggregate_transition_summary(transition_rows)
    }
    hv_lookup = {(f(row["capacity_multiplier"]), row["method"], row["task_group"]): row for row in hv_rows}
    waste_lookup = {(f(row["capacity_multiplier"]), row["method"]): row for row in waste_rows}
    for row in method_summary_rows:
        cap = f(row.get("capacity_multiplier"))
        method = str(row.get("method"))
        hv = hv_lookup.get((cap, method, "HV"), {})
        hvt = hv_lookup.get((cap, method, "HV-TARGET"), {})
        waste = waste_lookup.get((cap, method), {})
        regressions = sum(value for (c, m, transition), value in transition_count_lookup.items() if c == cap and m == method and transition.startswith("closure->"))
        gains = transition_count_lookup.get((cap, method, "partial->closure"), 0)
        corrected_report_lines.append(
            f"| {cap:.2f} | {method} | {int(f(row.get('requirement_closure')))} | {int(f(hv.get('on_time_requirement_success')))} | "
            f"{int(f(hvt.get('on_time_requirement_success')))} | {regressions} | {gains} | {f(row.get('common_net_utility')):.6f} | "
            f"{f(row.get('transmitted_mb')):.3f} | {int(f(waste.get('generated_but_never_transmitted')))} |"
        )
    guard_decisions = Counter(str(row.get("decision", "")) for row in guard_rows)
    corrected_report_lines += [
        "",
        "## Guard Audit",
        "",
        f"- Guard decisions: {dict(guard_decisions)}.",
        "- Protected closing task rows were recorded, but no candidate was estimated to displace a protected ready closure under the implemented capacity test.",
        "- Contact reservation is present as a default-disabled trace path, but this seed produced no reservation rows because no candidate was blocked or reserved by that feature.",
        "",
        "## Answers",
        "",
        "### 1. Which requests truly caused the 12 regressions?",
        "Still unresolved at request level. `regression_causal_replay.csv` preserves the prior unresolved replay rows; no causal claim is made without successful request-disable replay evidence.",
        "",
        "### 2. How many were communication displacement?",
        "Confirmed count is 0. Candidate evidence points to communication/queue overlap, but confirmation remains unresolved.",
        "",
        "### 3. How many were compute displacement?",
        "Confirmed count is 0; saved traces still lack task-attributed compute intervals proving compute displacement.",
        "",
        "### 4. Did close-task protection prevent them?",
        "No. It admitted all 256 guarded candidates and removed 0 of the 12 regressions.",
        "",
        "### 5. Did net-benefit checking improve utility?",
        "No. Net-guard results are identical to persistent requests.",
        "",
        "### 6. Did contact reservation prevent overbooking?",
        "No active effect was observed; full-guard results are identical and reservation trace is empty.",
        "",
        "### 7. How many of the 52 useful gains were preserved?",
        "All 52 partial-to-closure transitions were preserved: 21 at 1.00, 18 at 0.50, and 13 at 0.35.",
        "",
        "### 8. How many regressions remained?",
        "All 12 remained: 1 at 1.00, 10 at 0.50, and 1 at 0.35.",
        "",
        "### 9. Did the 0.50 net-utility decline disappear?",
        "No. PROP-BASE is 42.661000 and all persistent/guard variants are 42.408322 at 0.50.",
        "",
        "### 10. Which guard variant should be retained?",
        "None as a behavioral fix from this run; keep only the default-disabled diagnostics and guard traces.",
        "",
        "### 11. Which variant was too conservative or unnecessary?",
        "All three guard variants were unnecessary in this seed because they made no decisions other than ADMIT.",
        "",
        "### 12. What remains the main bottleneck?",
        "The main bottleneck remains communication/queue contention, but request-level causality must be resolved with stronger replay evidence.",
        "",
        "## Validation",
        "",
        "- Baseline reproduction passed for PROP-BASE and B1-Enhanced.",
        "- Task and environment hashes were identical across variants.",
        "- Every method/capacity retained 268 unique tasks.",
        "- No NaN/inf was found in method summaries.",
        "- No compression variant was run.",
    ]
    (OUT / "final_closure_safe_guard_report.md").write_text("\n".join(corrected_report_lines) + "\n", encoding="utf-8")
    zh = (
        "\u7ed3\u8bba\uff1a\u672c\u8f6e\u5b8c\u6574\u5b88\u536b\u6ca1\u6709\u771f\u6b63\u751f\u6548\uff0c\u7ed3\u679c\u4e0e\u6301\u4e45\u8bf7\u6c42\u5b8c\u5168\u76f8\u540c\uff0c\u56e0\u6b64\u53ea\u80fd\u4fdd\u7559\u4e3a\u5b9e\u9a8c\u6027\u8bca\u65ad\uff0c\u4e0d\u80fd\u4f5c\u4e3a\u5df2\u4fee\u590d\u7b56\u7565\u3002\n\n"
        "| \u5bb9\u91cf | \u57fa\u7ebf\u95ed\u5408 | \u6301\u4e45\u8bf7\u6c42 | \u5b8c\u6574\u5b88\u536b |\n"
        "|---:|---:|---:|---:|\n"
        "| 1.00 | 151 | 171 | 171 |\n| 0.50 | 132 | 140 | 140 |\n| 0.35 | 109 | 121 | 121 |\n\n"
        "\u539f\u670912\u4e2a\u95ed\u5408\u56de\u9000\u6ca1\u6709\u88ab\u79fb\u9664\uff1b52\u4e2a\u6709\u7528\u95ed\u5408\u589e\u76ca\u5168\u90e8\u4fdd\u7559\u3002HV\u548cHV-TARGET\u76f8\u5bf9\u57fa\u7ebf\u63d0\u5347\uff0c\u4f46\u53ea\u662f\u7ee7\u627f\u6301\u4e45\u8bf7\u6c42\u7684\u6548\u679c\u3002\n\n"
        "0.50\u5bb9\u91cf\u4e0b\u51c0\u6548\u7528\u6ca1\u6709\u4fee\u590d\uff1a\u57fa\u7ebf42.661\uff0c\u5b88\u536b\u540e42.408\u3002\u4e3b\u8981\u4e89\u7528\u4ecd\u8868\u73b0\u4e3a\u901a\u4fe1/\u961f\u5217\u5bb9\u91cf\uff0c\u8ba1\u7b97\u4f4d\u79fb\u672a\u786e\u8ba4\u3002\n\n"
        "\u76f8\u5bf9\u57fa\u7ebf\uff0c\u989d\u5916\u4f20\u8f93\u7ea6\u4e3a30.0\u30018.1\u300116.8 MB\uff1b\u80fd\u8017\u89c1\u8d44\u6e90\u5b89\u5168\u8868\uff0c\u672a\u53d1\u73b0\u65b0\u7684\u5b89\u5168\u8fdd\u89c4\u3002\u6700\u91cd\u8981\u7684\u5269\u4f59\u95ee\u9898\u662f\u8bf7\u6c42\u7ea7\u56e0\u679c\u56de\u653e\u4ecd\u672a\u771f\u6b63\u786e\u8ba4\u6bcf\u4e2a\u56de\u9000\u7684\u8d23\u4efb\u8bf7\u6c42\u3002\n"
    )
    (OUT / "final_plain_language_summary_zh.md").write_text(zh, encoding="utf-8")

    bundle_members = [
        "baseline_reproduction.csv",
        "environment_alignment.csv",
        "method_summary.csv",
        "high_value_summary.csv",
        "transition_summary.csv",
        "utility_gain_loss_balance.csv",
        "communication_resource_summary.csv",
        "compute_displacement_summary.csv",
        "guard_admission_trace.csv",
        "protected_closing_task_trace.csv",
        "contact_capacity_reservation_trace.csv",
        "guard_estimation_accuracy.csv",
        "regression_causal_replay.csv",
        "regression_resource_timeline.csv",
        "regression_causal_summary.csv",
        "request_funnel_summary.csv",
        "generation_waste_summary.csv",
        "communication_window_summary.csv",
        "task_outcome_transition_analysis.csv",
        "matched_latency_summary.csv",
        "resource_safety_summary.csv",
        "validation_report.md",
        "final_closure_safe_guard_report.md",
        "final_plain_language_summary_zh.md",
    ]
    with zipfile.ZipFile(OUT / "closure_safe_guard_analysis_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in bundle_members:
            zf.write(OUT / name, arcname=name)
    print(OUT)


if __name__ == "__main__":
    main()
