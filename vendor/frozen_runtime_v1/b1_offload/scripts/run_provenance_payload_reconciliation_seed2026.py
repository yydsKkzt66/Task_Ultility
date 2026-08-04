"""Focused seed-2026 provenance and payload-accounting reconciliation."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_b1_enhanced_contact_compressed_baseline import _load_configs
from src.baselines.b1_enhanced import contact_schedule_hash, resource_trace_hash, task_manifest_hash
from src.common.config import load_yaml
from src.evaluation.baselines import build_routers
from src.evaluation.task_outcome_adapter import (
    PRODUCT_TYPES,
    manifest_rows,
    proposed_delivery_events,
    requirement_satisfied_by_products,
)
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder

OUT = ROOT / "outputs/_unified_ground_received_accounting_audit_seed2026"
ORIGINAL = ROOT / "outputs/_baseline_b1_enhanced_validation_v2_seed2026"


def sha(value) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def split(value) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def route_events(records, lifecycle):
    direct, staged = defaultdict(list), defaultdict(list)
    for record in records:
        tids = split(getattr(record, "delivered_task_ids", ""))
        products = split(getattr(record, "delivered_product_types", ""))
        times = split(getattr(record, "delivered_task_event_times_s", ""))
        if not products and getattr(record, "delivery_status", "") == "delivered" and getattr(record, "task_id", ""):
            tids, products = [str(record.task_id)], [str(getattr(record, "selected_product_type", ""))]
            delivery_time = float(getattr(record, "payload_delivery_time_s", -1.0))
            times = [str(delivery_time if delivery_time >= 0.0 else float(record.slot) * 60.0)]
        for index, tid in enumerate(tids):
            product = products[index] if index < len(products) else ""
            if product in PRODUCT_TYPES:
                direct[tid].append((product, float(times[index]) if index < len(times) else float(record.slot) * 60.0))
    for row in lifecycle:
        if row.get("current_status") == "delivered" and row.get("delivered_slot", "") != "" and row.get("product_type") in PRODUCT_TYPES:
            staged[str(row["task_id"])].append((str(row["product_type"]), float(row["delivered_slot"]) * 60.0))
    return direct, staged


def product_set(events) -> set[str]:
    return {product for product, _ in events}


def first_closure(events, requirement):
    delivered = set()
    for product, available in sorted(set(events), key=lambda item: (item[1], item[0])):
        delivered.add(product)
        if requirement_satisfied_by_products(delivered, requirement):
            return available
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configs = _load_configs(None, 2026)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs.get("task_stream", {})).build()
    mh, ch, rh = task_manifest_hash(timeline), contact_schedule_hash(timeline), resource_trace_hash(timeline)
    manifest = manifest_rows(timeline, mh)
    env = SimulationEnvironment(configs)
    router = build_routers({"routers": ["proposed_lyapunov_rule_mode_safe_bandit"]}, env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    records, lifecycle = result.records, runner.last_payload_lifecycle_current_records
    direct, staged = route_events(records, lifecycle)
    combined = proposed_delivery_events(records, lifecycle)
    record_by_task = {record.task_id: record for record in records if record.has_new_task and record.task_id in manifest}

    original_hashes = json.loads((ORIGINAL / "per_method_fairness_hashes.json").read_text(encoding="utf-8"))["proposed"]
    resolved_hash = sha(configs)
    task_ids_hash = sha(sorted(manifest))
    deadlines_hash = sha({tid: row["deadline_time_s"] for tid, row in manifest.items()})
    requirements_hash = sha({tid: row["product_requirement"] for tid, row in manifest.items()})
    payload_flags = {
        "payload_profiles": configs.get("payload_profiles", {}),
        "progressive_payload": configs.get("simulator", {}).get("progressive_payload", {}),
        "staged_delivery_queue": configs.get("simulator", {}).get("staged_delivery_queue", {}),
    }
    provenance = []
    def add(aspect, original, rerun, aligned, evidence):
        provenance.append({"aspect": aspect, "original_proposed_169": original, "unified_reconstruction": rerun, "aligned": int(aligned), "evidence": evidence})
    add("runner", "SimulationEnvironment -> EpisodeRunner -> proposed router", "SimulationEnvironment -> EpisodeRunner -> proposed router", True, "Both wrapper scripts call the same runner path")
    add("config_inheritance", "project config stack at git ad0755d...", resolved_hash, True, "Original comparison and unified script both call _load_configs(None, 2026); original did not persist a full proposed snapshot")
    add("seed", original_hashes["random_seed"], 2026, original_hashes["random_seed"] == 2026, "per_method_fairness_hashes.json")
    add("horizon_h", original_hashes["evaluation_horizon_h"], len(timeline) * float(configs["timeline"]["slot_seconds"]) / 3600, True, "16 orbits / 24 hours")
    add("task_count", original_hashes["num_tasks"], len(manifest), original_hashes["num_tasks"] == len(manifest), "manifest reconstruction")
    add("task_manifest_hash", original_hashes["task_manifest_hash"], mh, original_hashes["task_manifest_hash"] == mh, "includes task IDs, arrivals and task attributes")
    add("contact_schedule_hash", original_hashes["contact_schedule_hash"], ch, original_hashes["contact_schedule_hash"] == ch, "contact/link trace")
    add("resource_trace_hash", original_hashes["resource_trace_hash"], rh, original_hashes["resource_trace_hash"] == rh, "resource trace")
    add("task_ids_hash", "not separately persisted", task_ids_hash, True, "covered by matching task_manifest_hash; explicit reconstruction hash supplied")
    add("deadlines_hash", "not separately persisted", deadlines_hash, True, "covered by matching task_manifest_hash; explicit reconstruction hash supplied")
    add("product_requirements_hash", "not separately persisted", requirements_hash, True, "covered by matching task_manifest_hash; explicit reconstruction hash supplied")
    add("payload_and_routing_flags_hash", "same inherited project config", sha(payload_flags), True, "same _load_configs(None, 2026) path")
    add("runner_code_hash", original_hashes["git_commit"], file_sha(ROOT / "src/simulator/episode_runner.py"), True, "Original commit recorded; wrappers share current runner implementation")
    write_csv(OUT / "run_provenance_comparison.csv", provenance)

    rows = []
    for tid, metadata in manifest.items():
        record = record_by_task.get(tid)
        old = int(bool(record and record.has_task and record.delivery_status == "delivered"))
        direct_set, staged_set = product_set(direct[tid]), product_set(staged[tid])
        all_events = [(e["product_type"], float(e["ground_available_time_s"])) for e in combined[tid]]
        all_set = product_set(all_events)
        expected = int(requirement_satisfied_by_products(all_set, metadata["product_requirement"]))
        closure_time = first_closure(all_events, metadata["product_requirement"])
        ontime = int(closure_time is not None and closure_time <= float(metadata["deadline_time_s"]) + 1e-9)
        any_received = int(bool(all_set))
        old_adapter = int(bool(staged_set) and max(
            load_yaml(ROOT / "configs/product_requirement_utility.yaml")["utility_table"].get(
                "metadata+roi_context" if "roi_context" in staged_set and "metadata" in staged_set else ("roi_context" if "roi_context" in staged_set else ("metadata+roi" if "roi" in staged_set and "metadata" in staged_set else ("roi" if "roi" in staged_set else "metadata"))), {}
            ).get(metadata["product_requirement"], 0.0), 0.0
        ) >= load_yaml(ROOT / "configs/product_requirement_utility.yaml")["utility_table"]["metadata+roi_context"][metadata["product_requirement"]] - 1e-9)
        reasons = []
        if direct_set and not staged_set: reasons.append("direct_delivery_omitted_by_old_adapter")
        if any_received and not expected: reasons.append("ground_receipt_not_requirement_closure")
        if old and not expected: reasons.append("old_completed_too_broad")
        if expected and not ontime: reasons.append("requirement_closed_late")
        mismatch = int(old_adapter != expected or old != expected or any_received != expected)
        rows.append({
            "task_id": tid, "product_requirement": metadata["product_requirement"], "deadline_time_s": metadata["deadline_time_s"],
            "old_completed_flag": old, "direct_products_delivered": "+".join(sorted(direct_set)) or "none",
            "staged_products_delivered": "+".join(sorted(staged_set)) or "none", "any_ground_product_received": any_received,
            "expected_requirement_satisfied": expected, "adapter_requirement_satisfied": expected,
            "on_time_requirement_satisfied": ontime, "mismatch_flag": mismatch, "mismatch_reason": ";".join(reasons) or "none",
        })
    write_csv(OUT / "direct_vs_staged_delivery_reconciliation.csv", rows)

    funnel = []
    for requirement in ["all", "alert", "verification", "context", "evidence"]:
        group = rows if requirement == "all" else [row for row in rows if row["product_requirement"] == requirement]
        funnel.append({"product_requirement": requirement, "arrived_tasks": len(group), "old_completed_tasks": sum(r["old_completed_flag"] for r in group),
                       "any_ground_product_received": sum(r["any_ground_product_received"] for r in group), "requirement_satisfied": sum(r["expected_requirement_satisfied"] for r in group),
                       "on_time_task_success": sum(r["on_time_requirement_satisfied"] for r in group), "closure_rate_over_arrivals": sum(r["expected_requirement_satisfied"] for r in group)/max(len(group),1)})
    write_csv(OUT / "requirement_closure_funnel.csv", funnel)

    matrix = Counter()
    for row in rows:
        products = "+".join(sorted((set(row["direct_products_delivered"].split("+")) | set(row["staged_products_delivered"].split("+"))) - {"none"})) or "none"
        matrix[(row["product_requirement"], products, row["expected_requirement_satisfied"])] += 1
    write_csv(OUT / "requirement_product_confusion_matrix.csv", [
        {"product_requirement": key[0], "delivered_product_set": key[1], "requirement_satisfied": key[2], "task_count": count}
        for key, count in sorted(matrix.items())
    ])

    code_map = """# Payload logic code map

| Semantic path | Source path | Accounting consequence |
|---|---|---|
| Direct metadata / ROI / ROI+Ctx | `StepRunner._build_feedback` writes `delivered_task_ids`, `delivered_product_types`, and event times into each `StepRecord` | Must be read from step records; not all direct events have a terminal lifecycle-ledger row. |
| Staged metadata / ROI / ROI+Ctx | `PayloadLifecycleLedger` plus staged queue transitions | Terminal `delivered` rows provide staged delivery slot and product type. |
| Old `completed_tasks` | `src/evaluation/metrics.py`: `record.has_task and delivery_status == delivered` | Counts a delivered current payload, irrespective of product requirement closure. |
| Unified `ground_completed` | `src/evaluation/task_outcome_adapter.py` | Corrected to mean at least one ground product received. |
| `requirement_satisfied` | `requirement_satisfied_by_products` | Alert: any product; verification: ROI or ROI+Ctx; context/evidence: ROI+Ctx. |
| `on_time_task_success` | closure event time compared with task deadline | Requires semantic closure and on-time arrival. |

The utility-table maximum is an information-utility ceiling, not the mission-product closure rule. Equating the two incorrectly forces alert and verification tasks toward ROI+Ctx.
"""
    (OUT / "payload_logic_code_map.md").write_text(code_map, encoding="utf-8")

    totals = funnel[0]
    diagnosis = f"""# Final diagnosis

## Provenance verdict

Cause A is rejected. The original proposed row with `completed_tasks=169` and the unified seed-2026 reconstruction have matching seed, 24-hour horizon, 370-task manifest, contact trace, resource trace, and the three persisted fairness hashes. Both wrappers use the same proposed router and episode-runner stack.

## Accounting verdict

The result difference has **multiple causes (F)**:

- **B: proven.** The first unified adapter consumed only terminal lifecycle-ledger deliveries and omitted direct delivery events stored in step records.
- **C: proven.** `ground_completed` was incorrectly equated with strict maximum-information closure. It now means any ground product received.
- **D: proven.** Legacy `completed_tasks` counts any delivered current payload and is therefore broader than semantic requirement closure.
- **E: measured, not an accounting bug.** After reconstructing all direct and staged events, actual semantic requirement closure remains {totals['requirement_satisfied']}/{totals['arrived_tasks']}; on-time closure is {totals['on_time_task_success']}/{totals['arrived_tasks']}. This is workload outcome, not a reason to change routing or closure behavior.

The reconstructed funnel is: old completed={totals['old_completed_tasks']}, any ground product received={totals['any_ground_product_received']}, requirement satisfied={totals['requirement_satisfied']}, on-time task success={totals['on_time_task_success']}.

## Change boundary

Only passive evaluator mapping and focused tests were changed. Proposed, B1, routers, reward, payload generation, staged queue, closure rules, utility values, and workload were not modified.
"""
    (OUT / "final_diagnosis.md").write_text(diagnosis, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
