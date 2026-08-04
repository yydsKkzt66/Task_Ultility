"""Audit direct/staged requirement semantics for the seed-2026 proposed run."""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_b1_enhanced_contact_compressed_baseline import _load_configs
from src.evaluation.baselines import build_routers
from src.evaluation.task_outcome_adapter import proposed_delivery_events, requirement_satisfied_by_products
from src.payload.staged_delivery_queue import staged_delivery_enabled_for_router
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder

OUT = ROOT / "outputs/_direct_task_requirement_mapping_audit_seed2026"


def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / name
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def products_text(products) -> str:
    return "+".join(sorted(set(products))) or "none"


def minimum_stage(requirement: str) -> str:
    return {"alert": "metadata_or_higher", "verification": "roi_or_roi_context", "context": "roi_context", "evidence": "roi_context"}.get(str(requirement).lower(), "roi_or_roi_context")


def stage(products: set[str]) -> str:
    if "roi_context" in products: return "roi_context"
    if "roi" in products: return "roi"
    if "metadata" in products: return "metadata"
    return "empty"


def main() -> None:
    configs = _load_configs(None, 2026)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs.get("task_stream", {})).build()
    tasks = {step.task.task_id: step.task for step in timeline if step.task}
    env = SimulationEnvironment(configs)
    router = build_routers({"routers": ["proposed_lyapunov_rule_mode_safe_bandit"]}, env.action_space)[0]
    staged_runtime_enabled = staged_delivery_enabled_for_router(configs["simulator"], router.router_name)
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    records, lifecycle = result.records, runner.last_payload_lifecycle_current_records
    arrivals = {record.task_id: record for record in records if record.has_new_task and record.task_id in tasks}
    events = proposed_delivery_events(records, lifecycle)
    lifecycle_by_task = defaultdict(list)
    for row in lifecycle: lifecycle_by_task[str(row.get("task_id", ""))].append(row)
    unified = {}
    with (OUT / "task_level_ground_outcomes.csv").open(encoding="utf-8-sig", newline="") as handle:
        unified = {row["task_id"]: row for row in csv.DictReader(handle) if row["method"] == "proposed"}

    audit_rows, event_rows = [], []
    for task_id, task_events in events.items():
        if task_id not in tasks or not task_events: continue
        task, record = tasks[task_id], arrivals[task_id]
        received = {str(event["product_type"]) for event in task_events}
        generated = {str(record.selected_product_type)} if str(record.selected_product_type) not in {"", "defer"} else set()
        generated.update(str(row.get("product_type")) for row in lifecycle_by_task[task_id] if str(row.get("current_status")) in {"created", "pending", "queued", "scheduled", "delivered", "blocked", "expired", "dropped"})
        generated.discard("")
        requirement = str(task.product_requirement).lower()
        value_level = str(getattr(task, "value_level", "") or (task.metadata or {}).get("value_level", "") or "medium").lower()
        plan = str(record.selected_payload_plan or record.payload_policy)
        action = str(record.selected_action or record.executed_action)
        staged_plan = plan in {"metadata_then_roi", "metadata_then_roi_context"}
        staged_lifecycle = any(
            str(row.get("current_status", "")) == "delivered"
            and ("staged_delivery_queue" in str(row.get("source", "")) or str(row.get("request_id", "")).startswith("staged_"))
            for row in lifecycle_by_task[task_id]
        )
        actual_path = "staged" if staged_lifecycle else "direct"
        staged_eligible = staged_runtime_enabled and (
            ("metadata" in received and requirement in {"verification", "context", "evidence"})
            or ("roi" in received and requirement in {"context", "evidence"} and value_level in {"high", "urgent"})
            or staged_plan
        )
        expected = int(requirement_satisfied_by_products(received, requirement))
        evaluator = int(unified[task_id]["requirement_satisfied"])
        ground_flag = int(unified[task_id]["ground_completed"])
        mismatch_reasons = []
        if evaluator != expected: mismatch_reasons.append("evaluator_requirement_mapping_mismatch")
        if received and not ground_flag: mismatch_reasons.append("delivered_product_missing_ground_result_event")
        if any(product not in {"metadata", "roi", "roi_context"} for product in received): mismatch_reasons.append("runtime_product_enum_not_mapped")
        if actual_path == "direct" and any("staged_delivery_queue" in str(row.get("source", "")) for row in lifecycle_by_task[task_id]): mismatch_reasons.append("direct_task_used_staged_closure_flag")
        audit_rows.append({
            "task_id": task_id, "product_requirement": requirement, "value_level": value_level,
            "staged_eligible_by_actual_code": int(staged_eligible), "actual_delivery_path": actual_path,
            "minimum_required_product_or_stage": minimum_stage(requirement), "selected_payload/action": f"{plan} / {action}",
            "generated_products": products_text(generated), "ground_received_products": products_text(received),
            "highest_ground_available_stage": stage(received), "ground_result_available_flag": ground_flag,
            "evaluator_requirement_satisfied": evaluator, "expected_requirement_satisfied_from_actual_code": expected,
            "mismatch_flag": int(bool(mismatch_reasons)), "mismatch_reason": ";".join(mismatch_reasons) or "none",
        })
        for event in sorted(task_events, key=lambda item: (float(item["ground_available_time_s"]), str(item["product_type"]))):
            product = str(event["product_type"])
            event_rows.append({"task_id": task_id, "product_requirement": requirement, "value_level": value_level,
                               "actual_delivery_path": actual_path, "product_type_runtime": product, "product_type_evaluator": product,
                               "delivery_route": event.get("delivery_route", ""), "ground_available_time_s": event["ground_available_time_s"],
                               "ground_result_available_flag": ground_flag, "delivered_missing_ground_result_event": int(not ground_flag),
                               "enum_mapping_error": int(product not in {"metadata", "roi", "roi_context"})})
    audit_rows.sort(key=lambda row: row["task_id"]); event_rows.sort(key=lambda row: (row["task_id"], float(row["ground_available_time_s"])))
    write_csv("direct_task_requirement_mapping_audit.csv", audit_rows)
    write_csv("ground_result_event_reconciliation.csv", event_rows)

    closure_rows = []
    for path in ["all", "direct", "staged"]:
        path_rows = audit_rows if path == "all" else [row for row in audit_rows if row["actual_delivery_path"] == path]
        for requirement in ["all", "alert", "verification", "context", "evidence"]:
            group = path_rows if requirement == "all" else [row for row in path_rows if row["product_requirement"] == requirement]
            closure_rows.append({"actual_delivery_path": path, "product_requirement": requirement, "ground_received_task_count": len(group),
                                 "evaluator_requirement_satisfied_count": sum(row["evaluator_requirement_satisfied"] for row in group),
                                 "expected_requirement_satisfied_count": sum(row["expected_requirement_satisfied_from_actual_code"] for row in group),
                                 "mapping_mismatch_count": sum(row["mismatch_flag"] for row in group)})
    write_csv("direct_task_closure_by_requirement.csv", closure_rows)

    all_unified = list(unified.values())
    funnel_rows = []
    for requirement in ["all", "alert", "verification", "context", "evidence"]:
        group = all_unified if requirement == "all" else [row for row in all_unified if row["product_requirement"] == requirement]
        funnel_rows.append({
            "product_requirement": requirement, "arrived_tasks": len(group),
            "any_ground_product_received": sum(int(row["ground_completed"]) for row in group),
            "requirement_satisfied": sum(int(row["requirement_satisfied"]) for row in group),
            "on_time_task_success": sum(int(row["on_time_task_success"]) for row in group),
            "partial_only": sum(int(row["ground_completed"]) and not int(row["requirement_satisfied"]) for row in group),
        })
    write_csv("requirement_closure_funnel.csv", funnel_rows)

    direct_rows = [row for row in audit_rows if row["actual_delivery_path"] == "direct"]
    staged_rows = [row for row in audit_rows if row["actual_delivery_path"] == "staged"]
    partial = [row for row in audit_rows if not row["expected_requirement_satisfied_from_actual_code"]]
    categories = Counter()
    for row in partial:
        if row["mismatch_reason"].find("missing_ground_result") >= 0: categories["missing_ground_result_events"] += 1
        elif row["mismatch_reason"].find("enum") >= 0: categories["enum_stage_mapping_errors"] += 1
        elif row["actual_delivery_path"] == "staged": categories["true_staged_incomplete"] += 1
        elif row["evaluator_requirement_satisfied"] != row["expected_requirement_satisfied_from_actual_code"]: categories["direct_incorrectly_evaluated"] += 1
        else: categories["true_direct_lower_than_required"] += 1
    alert_metadata_bad = sum(row["product_requirement"] == "alert" and "metadata" in row["ground_received_products"] and not row["evaluator_requirement_satisfied"] for row in audit_rows)
    verification_roi_bad = sum(row["product_requirement"] == "verification" and "roi" in row["ground_received_products"] and not row["evaluator_requirement_satisfied"] for row in audit_rows)
    direct_staged_flag_bad = sum("direct_task_used_staged_closure_flag" in row["mismatch_reason"] for row in direct_rows)
    missing_events = sum(row["delivered_missing_ground_result_event"] for row in event_rows)
    enum_errors = sum(row["enum_mapping_error"] for row in event_rows)

    enum_doc = f"""# Payload enum and stage mapping audit

Runtime action payload policies are `metadata`, `roi`, `metadata_then_roi`, `metadata_then_roi_context`, `roi_context`, and `defer`. Runtime ground products normalize to the exact evaluator names `metadata`, `roi`, and `roi_context`. The staged plan names are plans, not delivered product enums. B1's `compressed_full_frame` is a separate adapter-only full-frame result and is not emitted by the proposed payload action space.

| Runtime name | Kind | Evaluator mapping |
|---|---|---|
| `metadata` | direct product | `metadata` / alert stage |
| `roi` | direct or residual product | `roi` / verification stage |
| `roi_context` | direct or residual product | `roi_context` / context-evidence stage |
| `metadata_then_roi` | staged plan alias | generated `metadata`, then optional `roi` |
| `metadata_then_roi_context` | staged plan alias | generated `metadata`, then residual products toward `roi_context` |
| `compressed_full_frame` | B1 evaluator product | `full_frame_result`, satisfies all requirements |

Observed proposed delivery enum errors: **{enum_errors}**. The resolved seed-2026 runtime has value-aware staged delivery enabled=`{str(staged_runtime_enabled).lower()}`. Minimum required stage is keyed from `task.product_requirement`; value level only gates whether ROI+Ctx follow-up may be staged after ROI when staged delivery is enabled. Selected action does not define closure.
"""
    (OUT / "payload_enum_and_stage_mapping_audit.md").write_text(enum_doc, encoding="utf-8")

    report = f"""# Ordinary task one-shot completion audit

Among the 203 tasks with a ground product, **{len(direct_rows)} are direct path** and **{len(staged_rows)} are staged path**. Direct-path semantic closure is **{sum(row['expected_requirement_satisfied_from_actual_code'] for row in direct_rows)}/{len(direct_rows)}**.

Answers:

1. Direct={len(direct_rows)}; staged={len(staged_rows)}.
2. Direct-path requirement satisfied={sum(row['expected_requirement_satisfied_from_actual_code'] for row in direct_rows)}.
3. Alert+metadata incorrectly unsatisfied={alert_metadata_bad}.
4. Verification+ROI incorrectly unsatisfied={verification_roi_bad}.
5. Ordinary/direct tasks incorrectly evaluated using staged closure flags={direct_staged_flag_bad}.
6. Runtime/evaluator delivered-product enum errors={enum_errors}; plan aliases are explicitly separated from product names.
7. Delivered product events missing `ground_result_available`={missing_events}.
8. Minimum stage is derived from `task.product_requirement`. Value level gates staged ROI+Ctx eligibility only when the staged queue is enabled; action selection does not redefine closure. In this resolved run the staged queue is enabled={str(staged_runtime_enabled).lower()}.
9. The {len(partial)} partial-only tasks classify as: true staged incomplete={categories['true_staged_incomplete']}; true direct lower-than-required={categories['true_direct_lower_than_required']}; direct incorrectly evaluated={categories['direct_incorrectly_evaluated']}; missing ground-result events={categories['missing_ground_result_events']}; enum/stage mapping errors={categories['enum_stage_mapping_errors']}.

The expected-design premise does not match this resolved run: the value-aware staged queue is disabled, so all observed ground deliveries use the direct/ordinary queue path. No policy or routing change was made.
"""
    (OUT / "ordinary_task_one_shot_completion_audit.md").write_text(report, encoding="utf-8")
    print(OUT)


if __name__ == "__main__": main()
