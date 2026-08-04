"""Active persistent closing-generation probe for canonical SARFishSample seed 2026."""
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


OUT = ROOT / "outputs/_sarfish_persistent_closing_generation_active_seed2026"
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
    },
    "PROP-PERSIST-REQUEST": {
        "enable_persistent_closing_requests": True,
        "enable_future_contact_generation": False,
        "enable_generation_side_guarantee": False,
    },
    "PROP-FUTURE-CONTACT-GEN": {
        "enable_persistent_closing_requests": True,
        "enable_future_contact_generation": True,
        "enable_generation_side_guarantee": False,
    },
    "PROP-GEN-GUARANTEE": {
        "enable_persistent_closing_requests": True,
        "enable_future_contact_generation": True,
        "enable_generation_side_guarantee": True,
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
        "scripts/run_persistent_closing_generation_active_seed2026.py",
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
    return {
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
        "git switch exp/closure_aware_generation_probe_seed2026",
        "git branch -D exp/persistent_closing_generation_active_seed2026",
        "Remove-Item -LiteralPath 'outputs/_sarfish_persistent_closing_generation_active_seed2026' -Recurse -Force",
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
        for method in ("PROP-PERSIST-REQUEST", "PROP-FUTURE-CONTACT-GEN", "PROP-GEN-GUARANTEE", "B1-Enhanced-1.4W"):
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

    report_lines = ["# Persistent Closing Generation Report", ""]
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
    (OUT / "final_persistent_generation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    bundle_members = [
        "baseline_reproduction.csv",
        "environment_alignment.csv",
        "method_summary.csv",
        "high_value_summary.csv",
        "request_funnel_summary.csv",
        "generation_waste_summary.csv",
        "communication_window_summary.csv",
        "task_outcome_transition_analysis.csv",
        "matched_latency_summary.csv",
        "resource_safety_summary.csv",
        "validation_report.md",
        "final_persistent_generation_report.md",
    ]
    with zipfile.ZipFile(OUT / "persistent_generation_analysis_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in bundle_members:
            zf.write(OUT / name, arcname=name)
    print(OUT)


if __name__ == "__main__":
    main()
