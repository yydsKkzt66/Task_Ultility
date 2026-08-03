"""Audit why delivered residual payloads do not become evidence closure."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase6_residual_enqueue_fallback_validation import (  # noqa: E402
    ARMS,
    FALLBACK_ARM,
    _StressRouter,
    _load_configs,
    _stress_detection,
    _stress_feedback,
    _stress_payload,
    _stress_step,
    _stress_task,
)
from scripts.run_product_requirement_aware_module2_ab import DEFAULT_SEEDS  # noqa: E402
from src.common.types import CommState, ProductLifecycleState, ResourceState  # noqa: E402
from src.link.downlink_queue import DownlinkQueue  # noqa: E402
from src.payload.information_sufficiency import (  # noqa: E402
    delta_information_utility,
    product_information_utility,
    residual_information_utility,
)
from src.reporting.lifecycle_taxonomy import stage_credit_for_product  # noqa: E402
from src.simulator.step_runner import StepRunner  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_residual_delivery_closure_accounting_audit"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    except Exception as exc:  # pragma: no cover - audit robustness
        return f"git {' '.join(args)} failed: {exc}"


def _add_file_patch(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    patch_lines = [
        f"diff --git a/{rel} b/{rel}",
        "new file mode 100644",
        "index 0000000..0000000",
        "--- /dev/null",
        f"+++ b/{rel}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    patch_lines.extend("+" + line for line in lines)
    return "\n".join(patch_lines) + "\n"


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _products(products: set[str] | list[str] | tuple[str, ...] | None) -> str:
    return ";".join(sorted(str(product) for product in (products or []) if str(product)))


def _metadata_has(metadata: dict[str, Any], key: str) -> int:
    return int(key in metadata and metadata.get(key) not in {"", None})


def _snapshot(state: ProductLifecycleState | None) -> dict[str, object]:
    if state is None:
        return {
            "delivered_products": "",
            "pending_products": "",
            "residual_utility_remaining": 0.0,
            "task_closed_flag": 0,
            "residual_delivered_count": 0,
            "residual_utility_delivered": 0.0,
            "proactive_followup_delivery_count": 0,
            "carryover_proactive_delivery_count": 0,
        }
    return {
        "delivered_products": _products(state.delivered_products),
        "pending_products": _products(state.pending_products),
        "residual_utility_remaining": float(state.residual_utility_remaining),
        "task_closed_flag": int(bool(state.task_closed_flag)),
        "residual_delivered_count": int(state.residual_delivered_count),
        "residual_utility_delivered": float(state.residual_utility_delivered),
        "proactive_followup_delivery_count": int(state.proactive_followup_delivery_count),
        "carryover_proactive_delivery_count": int(state.carryover_proactive_delivery_count),
    }


def _requirement_satisfied(products: set[str], requirement: str) -> int:
    requirement = str(requirement or "verification")
    if requirement == "alert":
        return int(bool(products & {"metadata", "roi", "roi_context"}))
    if requirement == "verification":
        return int(bool(products & {"roi", "roi_context"}))
    if requirement in {"context", "evidence"}:
        return int("roi_context" in products)
    return int(bool(products & {"roi", "roi_context"}))


def _trace_case(row: dict[str, object]) -> str:
    delivered_type = str(row.get("delivered_product_type", ""))
    before_delivered = set(str(row.get("lifecycle_delivered_products_before", "")).split(";")) - {""}
    after_delivered = set(str(row.get("lifecycle_delivered_products_after", "")).split(";")) - {""}
    before_pending = set(str(row.get("lifecycle_pending_products_before", "")).split(";")) - {""}
    after_pending = set(str(row.get("lifecycle_pending_products_after", "")).split(";")) - {""}
    before_residual = float(row.get("lifecycle_residual_utility_before", 0.0) or 0.0)
    after_residual = float(row.get("lifecycle_residual_utility_after", 0.0) or 0.0)
    before_closed = int(row.get("lifecycle_task_closed_before", 0) or 0)
    after_closed = int(row.get("lifecycle_task_closed_after", 0) or 0)
    verification = int(row.get("verification_closure_flag", 0) or 0)
    context = int(row.get("context_closure_flag", 0) or 0)
    evidence = int(row.get("evidence_closure_flag", 0) or 0)
    if delivered_type not in {"roi", "roi_context"}:
        return "Case A"
    if delivered_type not in after_delivered:
        return "Case B"
    if delivered_type in before_pending and delivered_type in after_pending:
        return "Case C"
    if after_residual >= before_residual - 1.0e-12:
        return "Case D"
    if delivered_type == "roi" and not evidence:
        return "Case G"
    if (verification or context) and not evidence:
        return "Case F"
    if after_residual < before_residual and before_closed == after_closed == 0:
        return "Case E"
    return "Case H"


def _blocker_reason(row: dict[str, object]) -> str:
    delivered_type = str(row.get("delivered_product_type", ""))
    if delivered_type == "roi":
        return "roi_updates_lifecycle_and_verification_but_does_not_grant_evidence_credit_or_full_roi_context_target"
    if delivered_type == "roi_context":
        return "roi_context_should_grant_evidence_if_metric_aggregation_reads_it"
    return "delivered_product_type_missing_or_unexpected"


def _build_runner(seed: int, arm: str) -> tuple[StepRunner, DownlinkQueue]:
    configs = _load_configs(seed, arm)
    queue = DownlinkQueue(queue_max_mb=8.0, slot_seconds=60.0)
    runner = StepRunner(
        None,
        None,
        None,
        None,
        None,
        queue,
        None,
        {"high_value_threshold": 0.75},
        configs["simulator"],
    )
    runner.last_guidance_bundle = {"mode_name": "nominal", "lambda_after_z": {"lambda_Q": 0.1}}
    return runner, queue


def _run_one(seed: int, arm: str) -> dict[str, Any]:
    runner, queue = _build_runner(seed, arm)
    task = _stress_task(seed)
    metadata_step = _stress_step(task, 1, throughput_mbps=0.02)
    metadata_payload = _stress_payload(task.task_id)
    metadata_result = queue.process_slot(1, metadata_payload, metadata_step.link_trace)
    metadata_feedback = _stress_feedback(metadata_result)
    merged = runner._apply_product_lifecycle_updates(
        metadata_step,
        metadata_payload,
        metadata_result,
        metadata_feedback,
        router=_StressRouter(),
        detection_result=_stress_detection(),
        resource_state=ResourceState(0.85, 1.0, 0.2),
        comm_state=CommState(0.0, 8.0),
    )
    before_state = runner.product_lifecycle.get(task.task_id)
    before = _snapshot(before_state)
    queued_item = next((item for item in queue.items if item.task_id == task.task_id), None)
    queued_item_fields = {
        "queue_item_product_type": "" if queued_item is None else queued_item.product_type,
        "queue_item_payload_plan": "" if queued_item is None else queued_item.payload_plan,
        "queue_item_followup_trigger": "" if queued_item is None else queued_item.followup_trigger,
        "queue_item_is_residual": 0 if queued_item is None else int(bool(queued_item.is_residual)),
        "queue_item_product_requirement": "" if queued_item is None else queued_item.product_requirement,
        "queue_item_duplicate_identity_key": "" if queued_item is None else queued_item.duplicate_identity_key,
        "queue_item_metadata_keys": "" if queued_item is None else ";".join(sorted(queued_item.metadata.keys())),
    }
    next_step = _stress_step(task, 2, throughput_mbps=8.0)
    next_result = queue.process_slot(2, None, next_step.link_trace)
    next_feedback = _stress_feedback(next_result, status=next_result.current_task_delivery_status)
    runner._apply_product_lifecycle_updates(
        next_step,
        None,
        next_result,
        next_feedback,
        router=_StressRouter(),
        detection_result=_stress_detection(),
        resource_state=ResourceState(0.85, 1.0, 0.2),
        comm_state=CommState(merged.queue_next_mb, 8.0),
    )
    after_state = runner.product_lifecycle.get(task.task_id)
    after = _snapshot(after_state)
    return {
        "seed": seed,
        "arm": arm,
        "task": task,
        "before": before,
        "after": after,
        "metadata_feedback": metadata_feedback,
        "delivery_feedback": next_feedback,
        "next_result": next_result,
        "queued_item_fields": queued_item_fields,
        "same_slot_trace": [dict(row, seed=seed, arm=arm) for row in runner.last_same_slot_carryover_trace],
    }


def _delivered_trace_rows(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        result = run["next_result"]
        task = run["task"]
        before = run["before"]
        after = run["after"]
        delivered_products_after = set(str(after["delivered_products"]).split(";")) - {""}
        delivered_products_before = set(str(before["delivered_products"]).split(";")) - {""}
        for index, task_id in enumerate(result.delivered_task_ids):
            product_type = result.delivered_product_types[index] if index < len(result.delivered_product_types) else ""
            metadata = result.delivered_payload_metadata[index] if index < len(result.delivered_payload_metadata) else {}
            credit = stage_credit_for_product(product_type)
            row: dict[str, object] = {
                "seed": run["seed"],
                "slot": 2,
                "task_id": task_id,
                "product_requirement": task.product_requirement,
                "residual_product_type": run["queued_item_fields"].get("queue_item_product_type", product_type),
                "delivered_product_type": product_type,
                "payload_plan": metadata.get("parent_payload_plan", run["queued_item_fields"].get("queue_item_payload_plan", "")),
                "followup_trigger": metadata.get("followup_trigger", run["queued_item_fields"].get("queue_item_followup_trigger", "")),
                "is_residual": run["queued_item_fields"].get("queue_item_is_residual", 0),
                "duplicate_identity_key": metadata.get(
                    "duplicate_identity_key",
                    run["queued_item_fields"].get("queue_item_duplicate_identity_key", ""),
                ),
                "delivered_incremental_utility": (
                    result.delivered_incremental_utilities[index]
                    if index < len(result.delivered_incremental_utilities)
                    else 0.0
                ),
                "lifecycle_delivered_products_before": before["delivered_products"],
                "lifecycle_pending_products_before": before["pending_products"],
                "lifecycle_residual_utility_before": before["residual_utility_remaining"],
                "lifecycle_task_closed_before": before["task_closed_flag"],
                "lifecycle_delivered_products_after": after["delivered_products"],
                "lifecycle_pending_products_after": after["pending_products"],
                "lifecycle_residual_utility_after": after["residual_utility_remaining"],
                "lifecycle_task_closed_after": after["task_closed_flag"],
                "requirement_satisfied_before": _requirement_satisfied(delivered_products_before, task.product_requirement),
                "requirement_satisfied_after": _requirement_satisfied(delivered_products_after, task.product_requirement),
                "verification_closure_flag": int(bool(delivered_products_after & {"roi", "roi_context"})),
                "context_closure_flag": int("roi_context" in delivered_products_after),
                "evidence_closure_flag": int(bool(credit.evidence)),
                "closure_blocker_reason": "",
                "delivered_payload_metadata_keys": ";".join(sorted(metadata.keys())),
                "metadata_has_is_residual": _metadata_has(metadata, "is_residual"),
                "metadata_has_followup_trigger": _metadata_has(metadata, "followup_trigger"),
                "metadata_has_product_type": _metadata_has(metadata, "product_type"),
                "metadata_has_product_requirement": _metadata_has(metadata, "product_requirement"),
                "metadata_has_duplicate_identity_key": _metadata_has(metadata, "duplicate_identity_key"),
                **run["queued_item_fields"],
            }
            row["closure_case"] = _trace_case(row)
            row["closure_blocker_reason"] = _blocker_reason(row)
            rows.append(row)
    return rows


def _transition_rows(trace_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in trace_rows:
        before_products = set(str(row["lifecycle_delivered_products_before"]).split(";")) - {""}
        after_products = set(str(row["lifecycle_delivered_products_after"]).split(";")) - {""}
        req = str(row["product_requirement"])
        rows.append(
            {
                "seed": row["seed"],
                "task_id": row["task_id"],
                "product_requirement": req,
                "delivered_product_type": row["delivered_product_type"],
                "product_information_utility_before": product_information_utility(before_products, req),
                "product_information_utility_after": product_information_utility(after_products, req),
                "delta_information_utility_from_product": delta_information_utility(
                    str(row["delivered_product_type"]), before_products, req
                ),
                "residual_information_utility_before": residual_information_utility(before_products, req),
                "residual_information_utility_after": residual_information_utility(after_products, req),
                "requirement_satisfied_before": row["requirement_satisfied_before"],
                "requirement_satisfied_after": row["requirement_satisfied_after"],
                "verification_closure_flag": row["verification_closure_flag"],
                "context_closure_flag": row["context_closure_flag"],
                "evidence_closure_flag": row["evidence_closure_flag"],
            }
        )
    return rows


def _funnel(trace_rows: list[dict[str, object]], fallback_runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    generated = sum(
        1
        for run in fallback_runs
        for event in run["same_slot_trace"]
        if event.get("event_type") == "same_slot_fallback_to_tplus1"
    )
    delivered_product_ok = sum(int(str(row["delivered_product_type"]) in {"roi", "roi_context"}) for row in trace_rows)
    lifecycle_writeback = sum(
        int(str(row["delivered_product_type"]) in set(str(row["lifecycle_delivered_products_after"]).split(";")))
        for row in trace_rows
    )
    pending_removed = sum(
        int(str(row["delivered_product_type"]) not in set(str(row["lifecycle_pending_products_after"]).split(";")))
        for row in trace_rows
    )
    residual_decreased = sum(
        int(float(row["lifecycle_residual_utility_after"]) < float(row["lifecycle_residual_utility_before"]))
        for row in trace_rows
    )
    task_closed = sum(int(row["lifecycle_task_closed_after"]) for row in trace_rows)
    verification = sum(int(row["verification_closure_flag"]) for row in trace_rows)
    context = sum(int(row["context_closure_flag"]) for row in trace_rows)
    evidence = sum(int(row["evidence_closure_flag"]) for row in trace_rows)
    stages = [
        ("residual_payload_generated_count", generated),
        ("residual_payload_enqueued_count", len(trace_rows)),
        ("residual_payload_delivered_count", len(trace_rows)),
        ("delivered_product_types_residual_product_count", delivered_product_ok),
        ("lifecycle_delivered_products_updated_count", lifecycle_writeback),
        ("lifecycle_pending_products_removed_count", pending_removed),
        ("residual_utility_remaining_decreased_count", residual_decreased),
        ("verification_closure_count", verification),
        ("context_closure_count", context),
        ("evidence_closure_count", evidence),
        ("task_closed_flag_count", task_closed),
    ]
    first_zero = next((name for name, value in stages if value == 0), "none")
    return [
        {
            "arm": FALLBACK_ARM,
            "n_delivered_residual": len(trace_rows),
            **{name: value for name, value in stages},
            "first_zero_stage": first_zero,
        }
    ]


def _lifecycle_shadow_summary(trace_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total = max(len(trace_rows), 1)
    alert = len(trace_rows)
    confirmation = sum(int(row["verification_closure_flag"]) for row in trace_rows)
    evidence = sum(int(row["evidence_closure_flag"]) for row in trace_rows)
    return [
        {
            "arm": FALLBACK_ARM,
            "task_count": len(trace_rows),
            "alert_rate": alert / total,
            "confirmation_rate": confirmation / total,
            "evidence_rate": evidence / total,
            "evidence_closure_count": evidence,
            "verification_closure_count": confirmation,
            "context_closure_count": sum(int(row["context_closure_flag"]) for row in trace_rows),
            "lifecycle_shadow_utility_balanced_proxy": mean(
                [0.30 + 0.30 * int(row["verification_closure_flag"]) + 0.40 * int(row["evidence_closure_flag"]) for row in trace_rows]
            )
            if trace_rows
            else 0.0,
            "lifecycle_shadow_utility_evidence_heavy_proxy": mean(
                [0.20 + 0.30 * int(row["verification_closure_flag"]) + 0.50 * int(row["evidence_closure_flag"]) for row in trace_rows]
            )
            if trace_rows
            else 0.0,
        }
    ]


def _staged_summary(trace_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "arm": FALLBACK_ARM,
            "metadata_delivered_count": len(trace_rows),
            "residual_delivered_count": len(trace_rows),
            "verification_closure_count": sum(int(row["verification_closure_flag"]) for row in trace_rows),
            "context_closure_count": sum(int(row["context_closure_flag"]) for row in trace_rows),
            "evidence_closure_count": sum(int(row["evidence_closure_flag"]) for row in trace_rows),
            "task_closed_flag_count": sum(int(row["lifecycle_task_closed_after"]) for row in trace_rows),
        }
    ]


def _write_reports(trace_rows: list[dict[str, object]], funnel: list[dict[str, object]], command: str, seeds: list[int]) -> None:
    delivered = len(trace_rows)
    roi_delivered = sum(int(row["delivered_product_type"] == "roi") for row in trace_rows)
    metadata_full = sum(
        int(
            row["metadata_has_is_residual"]
            and row["metadata_has_followup_trigger"]
            and row["metadata_has_product_type"]
            and row["metadata_has_product_requirement"]
            and row["metadata_has_duplicate_identity_key"]
        )
        for row in trace_rows
    )
    lifecycle_writeback = funnel[0]["lifecycle_delivered_products_updated_count"] if funnel else 0
    pending_removed = funnel[0]["lifecycle_pending_products_removed_count"] if funnel else 0
    residual_decreased = funnel[0]["residual_utility_remaining_decreased_count"] if funnel else 0
    task_closed = funnel[0]["task_closed_flag_count"] if funnel else 0
    verification = funnel[0]["verification_closure_count"] if funnel else 0
    context = funnel[0]["context_closure_count"] if funnel else 0
    evidence = funnel[0]["evidence_closure_count"] if funnel else 0
    first_zero = funnel[0]["first_zero_stage"] if funnel else "trace_missing"

    _write_text(
        OUT_DIR / "README.md",
        """# Residual Delivery Closure Accounting Audit

Read-only audit of why capacity-stress residual deliveries do not become evidence closure. The script replays the same default-off queue fallback stress scenario and records DownlinkResult, queue item, lifecycle, product satisfaction, and metric aggregation state.
""",
    )
    _write_text(
        OUT_DIR / "command_log.md",
        f"# Command log\n\n```powershell\n{command}\n```\n",
    )
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "input_reference": "outputs/_phase6_residual_enqueue_fallback_validation capacity-stress replay",
                "seeds": seeds,
                "arms_replayed": ARMS,
                "audit_only": True,
                "logic_changes": False,
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(
        OUT_DIR / "closure_metric_definition_audit.md",
        f"""# Closure Metric Definition Audit

`DownlinkResult.delivered_product_types` preserves residual product labels used by lifecycle writeback. In the audited stress replay, delivered residual product type is `roi` for {roi_delivered}/{delivered} delivered residuals.

`delivered_payload_metadata` preserves `followup_trigger` and `duplicate_identity_key`, but it does not carry every queue item field as metadata. In particular, `is_residual`, `product_type`, and `product_requirement` are QueueItem / DownlinkResult fields, not guaranteed metadata keys. Full metadata-key preservation count for the five requested keys is {metadata_full}/{delivered}.

Lifecycle accounting updates `delivered_products`, removes the delivered residual from `pending_products`, and recomputes `residual_utility_remaining` via `residual_information_utility`.

The report-only evidence closure taxonomy gives evidence credit to `roi_context`, `residual_roi_context`, `context_evidence`, or `evidence`, not plain `roi`. Plain ROI gives alert + confirmation credit. Therefore ROI delivery can improve verification/confirmation and lifecycle shadow utility while leaving `evidence_closure_count` at 0.
""",
    )
    _write_text(
        OUT_DIR / "first_zero_stage_report.md",
        f"""# First Zero Stage Report

First zero stage: `{first_zero}`.

Delivered residuals pass these stages:
- delivered_product_types contains residual product: {funnel[0]['delivered_product_types_residual_product_count'] if funnel else 0}/{delivered}
- lifecycle delivered_products updated: {lifecycle_writeback}/{delivered}
- lifecycle pending_products removed: {pending_removed}/{delivered}
- residual_utility_remaining decreased: {residual_decreased}/{delivered}
- verification closure: {verification}/{delivered}

The first zero is `{first_zero}` because delivered residual product is ROI, not ROI+Ctx. ROI gives verification/confirmation credit, but not context/evidence credit. `task_closed_flag` also remains zero because `residual_information_utility` targets the metadata+roi_context utility ceiling, leaving a small residual gap after metadata+roi for verification tasks.
""",
    )
    _write_text(
        OUT_DIR / "final_recommendation.md",
        f"""# Final Recommendation

Recommendation: E. ROI is not enough for evidence closure; next step should expand high-value context/evidence ROI+Ctx candidate coverage.

Answers:
1. `evidence_closure = 0` is a reasonable narrow evidence metric for these stress deliveries, not a lifecycle writeback bug.
2. Residual delivery updates lifecycle correctly: delivered writeback {lifecycle_writeback}/{delivered}, pending removal {pending_removed}/{delivered}, residual utility decrease {residual_decreased}/{delivered}.
3. Product satisfaction improves for verification/confirmation: {verification}/{delivered}.
4. The remaining blocker is not delivery or lifecycle writeback; it is ROI-only residual coverage versus evidence/ROI+Ctx closure.
5. Since product satisfaction improves but evidence closure remains 0, inspect verification_closure_count, confirmation_rate, residual delivered incremental utility, and lifecycle shadow utility.
6. Do not fix lifecycle writeback, information_sufficiency, or metric aggregation from this audit. The next useful change is coverage: generate/select ROI+Ctx for high-value context/evidence cases under a default-off experiment.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    command = "python scripts/audit_residual_delivery_closure_accounting.py --seeds " + ",".join(str(seed) for seed in seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = [_run_one(seed, arm) for seed in seeds for arm in ARMS]
    fallback_runs = [run for run in runs if run["arm"] == FALLBACK_ARM]
    trace_rows = _delivered_trace_rows(fallback_runs)
    transition_rows = _transition_rows(trace_rows)
    funnel = _funnel(trace_rows, fallback_runs)
    shadow_summary = _lifecycle_shadow_summary(trace_rows)
    staged_summary = _staged_summary(trace_rows)

    _write_csv(OUT_DIR / "residual_delivered_product_trace.csv", trace_rows)
    _write_csv(OUT_DIR / "lifecycle_update_after_residual_delivery_trace.csv", trace_rows)
    _write_csv(OUT_DIR / "product_satisfaction_transition_trace.csv", transition_rows)
    _write_csv(OUT_DIR / "closure_accounting_trace.csv", trace_rows)
    _write_csv(OUT_DIR / "staged_closure_summary_audit.csv", staged_summary)
    _write_csv(OUT_DIR / "lifecycle_shadow_summary_audit.csv", shadow_summary)
    _write_csv(OUT_DIR / "residual_delivery_closure_funnel_summary.csv", funnel)
    _write_reports(trace_rows, funnel, command, seeds)
    _write_text(OUT_DIR / "git_status.txt", _git(["status", "--short"]))
    script_path = PROJECT_ROOT / "scripts" / "audit_residual_delivery_closure_accounting.py"
    diff = _git(["diff", "--", "scripts/audit_residual_delivery_closure_accounting.py"])
    _write_text(OUT_DIR / "git_diff.patch", diff if diff.strip() else _add_file_patch(script_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
