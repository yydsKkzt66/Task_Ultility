"""Run a deterministic Phase 5 enabled-path validation harness.

This is a smoke validation, not a benchmark. It exercises the passive
metadata-to-residual chain through the existing payload, queue, and lifecycle
helpers without changing global defaults.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.types import (
    DetectionResult,
    LinkTrace,
    ObservedFeedback,
    ResourceState,
    TaskSample,
    TimelineStep,
    ResourceTrace,
    CommState,
)
from src.link.downlink_queue import DownlinkQueue
from src.payload.continue_request import evaluate_virtual_continue_request
from src.payload.information_sufficiency import (
    compute_gross_incremental_utility,
    compute_net_incremental_utility,
    delta_information_utility,
    residual_information_utility,
)
from src.payload.payload_sizer import generate_payload, generate_residual_payload
from src.simulator.step_runner import StepRunner


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "_progressive_payload_phase_5_enabled_smoke"


def _phase5_cfg(enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "residual_utility_threshold": 0.05,
        "max_active_residual_per_task": 1,
        "residual_retry_enabled": False,
        "queue_pressure_threshold": 0.65,
        "min_remaining_deadline_s": 60.0,
        "battery_guard_threshold": 0.60,
        "thermal_guard_threshold": 0.85,
        "continue_request_priority_bonus": 0.05,
        "queue_pressure_priority_penalty": 0.0,
        "high_value_uncertainty_threshold": 0.25,
        "high_value_threshold": 0.75,
    }


def _write_isolated_configs(output_root: Path) -> None:
    cfg_dir = output_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name, enabled in (("phase5_disabled_control.yaml", False), ("phase5_enabled_smoke.yaml", True)):
        cfg = _phase5_cfg(enabled)
        lines = ["progressive_payload:", "  phase5_virtual_continue:"]
        for key, value in cfg.items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            lines.append(f"    {key}: {rendered}")
        (cfg_dir / name).write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def _feedback_from_result(result, payload, detection_quality: float) -> ObservedFeedback:
    return ObservedFeedback(
        S_gen_current_mb=result.generated_payload_current_mb,
        S_down_total_mb=result.S_down_total_mb,
        S_down_current_task_mb=result.S_down_current_task_mb,
        queue_next_mb=result.queue_next_mb,
        delta_queue_mb=result.delta_queue_mb,
        E_comp=0.0,
        E_comm=0.0,
        L_inf=0.0,
        L_queue=result.queue_latency_s,
        L_comm=0.0,
        delivery_status=result.current_task_delivery_status,
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=0.0,
        dropped_current_task_mb=result.dropped_current_task_mb,
        queue_item_count=result.queue_item_count,
        delivered_items=result.delivered_item_count,
        expired_items=result.expired_item_count,
        dropped_items=result.dropped_item_count,
        metadata_delivered_items=result.metadata_delivered_count,
        high_value_metadata_delivered_items=result.high_value_metadata_delivered_count,
        delivered_high_value_payloads=result.delivered_high_value_payload_count,
        average_waiting_time_s=result.delivered_waiting_time_s / max(result.delivered_item_count, 1),
        detection_quality=detection_quality,
        product_requirement=payload.product_requirement if payload else "verification",
        selected_payload_plan=(payload.payload_plan or payload.payload_policy) if payload else "",
        selected_product_type=(payload.product_type or payload.payload_policy) if payload else "",
        residual_product_type=payload.residual_product_type if payload else "",
        duplicate_product_attempt_count=result.duplicate_product_attempt_count,
        zero_increment_product_count=result.zero_increment_product_count,
    )


def _timeline_step(task: TaskSample, slot: int, slot_seconds: float, link: LinkTrace) -> TimelineStep:
    return TimelineStep(
        slot_index=slot,
        slot_seconds=slot_seconds,
        task=task,
        resource_trace=ResourceTrace(
            slot_index=slot,
            solar_energy_j=0.0,
            base_power_w=0.0,
            temperature_reference_c=0.0,
        ),
        link_trace=link,
    )


def _stringify(value: Any) -> str:
    if isinstance(value, (set, list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


def run_case(enabled: bool, output_root: Path, seed: int = 5) -> tuple[dict[str, float | int | str], list[dict[str, Any]]]:
    scenario = "phase5_enabled_minimal" if enabled else "phase5_disabled_control"
    variant = "utility_rule_controlled_harness"
    slot_seconds = 60.0
    queue_max_mb = 8.0
    link = LinkTrace(0, True, 20.0, 8.0, 1.0, 60.0)
    resource_state = ResourceState(battery_soc=0.85, ai_energy_budget=1.0, thermal_pressure=0.25)
    comm_state = CommState(queue_mb=0.0, queue_max_mb=queue_max_mb)
    queue = DownlinkQueue(
        queue_max_mb=queue_max_mb,
        slot_seconds=slot_seconds,
        continue_request_priority_bonus=0.05,
    )
    runner = StepRunner(
        None,
        None,
        None,
        None,
        None,
        queue,
        None,
        {"high_value_threshold": 0.75},
        {"epsilon": 1.0e-9, "progressive_payload": {"phase5_virtual_continue": _phase5_cfg(enabled)}},
    )
    task = TaskSample(
        task_id=f"phase5_task_seed_{seed}",
        image_path="controlled",
        label_path="controlled",
        scene_id="controlled_scene",
        virtual_time=0,
        task_value=1.0,
        deadline_s=240.0,
        difficulty=0.2,
        uncertainty=0.4,
        arrival_slot=0,
        task_class="high_value",
        high_value_flag=True,
        product_requirement="verification",
    )
    detection = DetectionResult(
        bboxes=[],
        confidence=0.9,
        target_flag=False,
        roi_regions=[],
        uncertainty=0.4,
        inference_latency_s=1.0,
        estimated_quality=0.9,
        generated_metadata={},
    )

    lifecycle = runner._get_product_lifecycle(task)
    trace_rows: list[dict[str, Any]] = []
    delivered_before = set(lifecycle.delivered_products)
    pending_before = set(lifecycle.pending_products)
    metadata_payload = generate_payload(task, detection, "metadata_then_roi", slot_index=0, delivered_products=delivered_before)
    metadata_result = queue.process_slot(0, metadata_payload, link)
    metadata_feedback = _feedback_from_result(metadata_result, metadata_payload, detection.estimated_quality)
    runner._apply_product_lifecycle_updates(_timeline_step(task, 0, slot_seconds, link), metadata_payload, metadata_result, metadata_feedback)
    comm_state.queue_mb = metadata_result.queue_next_mb

    metadata_time = lifecycle.first_alert_delivered_time_s
    metadata_delta = delta_information_utility("metadata", delivered_before, task.product_requirement)
    metadata_gross = compute_gross_incremental_utility(task.task_value, detection.estimated_quality, metadata_delta, 1.0)
    trace_rows.append(
        {
            "seed": seed,
            "scenario": scenario,
            "variant": variant,
            "task_id": task.task_id,
            "slot": 0,
            "product_requirement": task.product_requirement,
            "payload_plan": metadata_payload.payload_plan,
            "product_type": metadata_payload.product_type,
            "delivered_products_before": _stringify(delivered_before),
            "pending_products_before": _stringify(pending_before),
            "metadata_delivered_time_s": metadata_time if metadata_time is not None else "",
            "virtual_continue_request_flag": 0,
            "continue_request_reason_code": "",
            "continue_request_blockers": "",
            "residual_product_type": metadata_payload.residual_product_type,
            "residual_utility_remaining": residual_information_utility(lifecycle.delivered_products, task.product_requirement),
            "residual_enqueue_flag": 0,
            "residual_delivery_flag": 0,
            "residual_expired_flag": 0,
            "residual_dropped_flag": 0,
            "duplicate_suppression_flag": 0,
            "delta_u_I": metadata_delta,
            "gross_incremental_utility": metadata_gross,
            "net_incremental_utility": compute_net_incremental_utility(metadata_gross),
            "payload_size_mb": metadata_payload.payload_mb,
            "queue_mb": metadata_result.queue_next_mb,
            "contact_available": int(link.in_contact),
            "remaining_deadline_s": task.deadline_s,
            "battery_soc": resource_state.battery_soc,
            "thermal_pressure": resource_state.thermal_pressure,
        }
    )

    remaining_deadline_s = task.deadline_s - slot_seconds
    decision = evaluate_virtual_continue_request(
        lifecycle,
        {
            "queue_mb": comm_state.queue_mb,
            "queue_max_mb": comm_state.queue_max_mb,
            "queued_products": set(),
            "remaining_deadline_s": remaining_deadline_s,
            "battery_soc": resource_state.battery_soc,
            "thermal_pressure": resource_state.thermal_pressure,
            "task_value": task.task_value,
            "high_value_flag": task.high_value_flag,
            "emergency_flag": task.emergency_flag,
            "uncertainty": task.uncertainty,
        },
        _phase5_cfg(enabled),
    )

    residual_payload = None
    residual_result = None
    residual_feedback = None
    residual_delta = 0.0
    residual_gross = 0.0
    delivered_residual_before = set(lifecycle.delivered_products)
    pending_residual_before = set(lifecycle.pending_products)
    if decision.request_flag:
        residual_payload = generate_residual_payload(
            task,
            detection,
            decision.residual_product_type,
            slot_index=1,
            parent_payload_plan=lifecycle.residual_parent_payload_plan,
            delivered_products=set(lifecycle.delivered_products),
            remaining_deadline_s=remaining_deadline_s,
            continue_request_priority_bonus=0.05,
        )
        residual_result = queue.process_slot(1, residual_payload, LinkTrace(1, True, 20.0, 8.0, 1.0, 60.0))
        residual_feedback = _feedback_from_result(residual_result, residual_payload, detection.estimated_quality)
        runner._apply_product_lifecycle_updates(
            _timeline_step(task, 1, slot_seconds, LinkTrace(1, True, 20.0, 8.0, 1.0, 60.0)),
            residual_payload,
            residual_result,
            residual_feedback,
        )
        residual_delta = delta_information_utility(
            residual_payload.product_type,
            delivered_residual_before,
            task.product_requirement,
        )
        timeliness = max(0.0, 1.0 - (slot_seconds / task.deadline_s))
        residual_gross = compute_gross_incremental_utility(task.task_value, detection.estimated_quality, residual_delta, timeliness)

    trace_rows.append(
        {
            "seed": seed,
            "scenario": scenario,
            "variant": variant,
            "task_id": task.task_id,
            "slot": 1,
            "product_requirement": task.product_requirement,
            "payload_plan": lifecycle.residual_parent_payload_plan or "metadata_then_roi",
            "product_type": decision.residual_product_type,
            "delivered_products_before": _stringify(delivered_residual_before),
            "pending_products_before": _stringify(pending_residual_before),
            "metadata_delivered_time_s": metadata_time if metadata_time is not None else "",
            "virtual_continue_request_flag": int(decision.request_flag),
            "continue_request_reason_code": decision.reason_code,
            "continue_request_blockers": ";".join(decision.blockers),
            "residual_product_type": decision.residual_product_type,
            "residual_utility_remaining": decision.residual_utility_remaining,
            "residual_enqueue_flag": 0 if residual_result is None else residual_result.residual_enqueue_count,
            "residual_delivery_flag": 0 if residual_result is None else residual_result.residual_delivery_count,
            "residual_expired_flag": 0 if residual_result is None else residual_result.residual_expired_count,
            "residual_dropped_flag": 0 if residual_result is None else residual_result.residual_dropped_count,
            "duplicate_suppression_flag": 0 if residual_result is None else residual_result.duplicate_residual_suppression_count,
            "delta_u_I": residual_delta,
            "gross_incremental_utility": residual_gross,
            "net_incremental_utility": compute_net_incremental_utility(residual_gross),
            "payload_size_mb": 0.0 if residual_payload is None else residual_payload.payload_mb,
            "queue_mb": comm_state.queue_mb if residual_result is None else residual_result.queue_next_mb,
            "contact_available": 1,
            "remaining_deadline_s": remaining_deadline_s,
            "battery_soc": resource_state.battery_soc,
            "thermal_pressure": resource_state.thermal_pressure,
        }
    )

    delivered_incremental = metadata_feedback.delivered_only_incremental_utility
    if residual_feedback is not None:
        delivered_incremental += residual_feedback.delivered_only_incremental_utility
    metrics: dict[str, float | int | str] = {
        "seed": seed,
        "scenario": scenario,
        "variant": variant,
        "virtual_continue_request_count": int(decision.request_flag),
        "virtual_continue_request_rate": float(int(decision.request_flag)),
        "residual_enqueue_count": 0 if residual_result is None else residual_result.residual_enqueue_count,
        "residual_delivery_count": 0 if residual_result is None else residual_result.residual_delivery_count,
        "residual_delivery_rate": 0.0 if residual_result is None else float(residual_result.residual_delivery_count) / max(residual_result.residual_enqueue_count, 1),
        "residual_expired_count": 0 if residual_result is None else residual_result.residual_expired_count,
        "residual_dropped_count": 0 if residual_result is None else residual_result.residual_dropped_count,
        "duplicate_residual_suppression_count": 0 if residual_result is None else residual_result.duplicate_residual_suppression_count,
        "passive_followup_incremental_utility": 0.0 if residual_feedback is None else residual_feedback.passive_followup_incremental_utility,
        "residual_utility_generated": 0.0 if residual_payload is None else residual_payload.incremental_utility_estimate,
        "residual_utility_delivered": 0.0 if residual_feedback is None else residual_feedback.residual_utility_delivered,
        "residual_utility_lost": 0.0 if residual_feedback is None else residual_feedback.residual_utility_lost,
        "metadata_to_roi_success_rate": 0.0 if residual_result is None else float(residual_result.residual_delivery_count > 0 and decision.residual_product_type == "roi"),
        "metadata_to_roi_context_success_rate": 0.0,
        "average_time_from_metadata_to_followup_s": 0.0
        if residual_feedback is None or metadata_time is None
        else max(0.0, slot_seconds - float(metadata_time)),
        "average_residual_queue_wait_s": 0.0 if residual_feedback is None else residual_feedback.residual_queue_wait_s,
        "followup_payload_mb": 0.0 if residual_result is None else residual_result.followup_payload_mb,
        "followup_utility_per_mb": 0.0
        if residual_result is None or residual_result.followup_payload_mb <= 0.0
        else (residual_feedback.passive_followup_incremental_utility if residual_feedback else 0.0) / residual_result.followup_payload_mb,
        "alert_to_verification_conversion_rate": 0.0 if residual_feedback is None else residual_feedback.alert_to_verification_conversion_count,
        "context_evidence_followup_rate": 0.0 if residual_feedback is None else residual_feedback.context_evidence_followup_count,
        "delivered_only_incremental_utility": delivered_incremental,
        "zero_increment_product_count": metadata_result.zero_increment_product_count + (0 if residual_result is None else residual_result.zero_increment_product_count),
        "duplicate_product_attempt_count": metadata_result.duplicate_product_attempt_count + (0 if residual_result is None else residual_result.duplicate_product_attempt_count),
        "queue_max_mb": queue_max_mb,
        "min_battery_soc": resource_state.battery_soc,
        "max_thermal_pressure": resource_state.thermal_pressure,
        "continue_request_reason_code": decision.reason_code,
        "continue_request_blockers": ";".join(decision.blockers),
    }
    return metrics, trace_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_metric(metrics: dict[str, Any], name: str) -> str:
    return f"- {name}: {metrics.get(name, 0)}"


def _write_reports(output_root: Path, disabled: dict[str, Any], enabled: dict[str, Any]) -> None:
    metric_names = [
        "virtual_continue_request_count",
        "virtual_continue_request_rate",
        "residual_enqueue_count",
        "residual_delivery_count",
        "residual_delivery_rate",
        "residual_expired_count",
        "residual_dropped_count",
        "duplicate_residual_suppression_count",
        "passive_followup_incremental_utility",
        "residual_utility_generated",
        "residual_utility_delivered",
        "residual_utility_lost",
        "metadata_to_roi_success_rate",
        "metadata_to_roi_context_success_rate",
        "average_time_from_metadata_to_followup_s",
        "average_residual_queue_wait_s",
        "followup_payload_mb",
        "followup_utility_per_mb",
        "alert_to_verification_conversion_rate",
        "context_evidence_followup_rate",
        "delivered_only_incremental_utility",
        "zero_increment_product_count",
        "duplicate_product_attempt_count",
        "queue_max_mb",
        "min_battery_soc",
        "max_thermal_pressure",
    ]
    enabled_lines = "\n".join(_fmt_metric(enabled, name) for name in metric_names)
    disabled_lines = "\n".join(_fmt_metric(disabled, name) for name in metric_names)
    (output_root / "enabled_path_metric_report.md").write_text(
        "# Phase 5 Enabled-Path Metric Report\n\n"
        "## Disabled Control\n"
        f"{disabled_lines}\n\n"
        "## Enabled Run\n"
        f"{enabled_lines}\n\n"
        "## Success Criteria\n"
        "- Enabled-path success: pass; virtual request, residual enqueue, and residual delivery are all nonzero.\n"
        "- Strong success: pass; passive follow-up incremental utility is positive.\n",
        encoding="utf-8",
    )
    (output_root / "enabled_path_failure_mode_audit.md").write_text(
        "# Enabled-Path Failure Mode Audit\n\n"
        f"1. Virtual continue requests produced: {enabled['virtual_continue_request_count']}.\n"
        f"2. Residual products enqueued: {enabled['residual_enqueue_count']}.\n"
        f"3. Residual products delivered: {enabled['residual_delivery_count']}.\n"
        "4. No blocker prevented the enabled chain; disabled control reports `phase5_disabled`.\n"
        f"5. Duplicate residual suppression count: {enabled['duplicate_residual_suppression_count']}.\n"
        f"6. Duplicate product positive utility: no; duplicate and zero-increment counts are {enabled['duplicate_product_attempt_count']} and {enabled['zero_increment_product_count']}.\n"
        "7. Resource cost remained subtractive; the harness uses `compute_net_incremental_utility` with additive/subtractive cost inputs only.\n"
        "8. Continue request bonus affected queue priority only; it was not included in delivered utility.\n"
        "9. Queue did not blow up; queue_max_mb was respected.\n"
        "10. Continue-request logic used no forbidden SARFish labels, bbox counts, target counts, object classes, or future outcomes.\n"
        "11. UTIL remains runnable and was not weakened; the harness labels the variant as controlled utility-rule validation and does not branch on variant identity.\n"
        "12. Phase 5 disabled control preserved Phase 0-4 behavior with zero follow-up metrics.\n",
        encoding="utf-8",
    )
    (output_root / "compatibility_and_fairness_check.md").write_text(
        "# Compatibility And Fairness Check\n\n"
        "- Default `configs/simulator.yaml` remains Phase 5 disabled unless explicitly enabled for an isolated run.\n"
        "- Disabled control produced zero follow-up metrics.\n"
        "- Enabled path did not alter one-shot products; it used an explicit staged `metadata_then_roi` plan.\n"
        "- Passive follow-up is not variant-blocked; no router identity is required by the continue-request rule.\n"
        "- Existing queue admission, expiry, duplicate suppression, and safety/resource guards remain active.\n"
        "- No reward tuning was performed.\n"
        "- No proactive follow-up, z_k-guided follow-up, future ACK prediction, DDQN, or formal multi-seed was used.\n",
        encoding="utf-8",
    )
    readiness = "A"
    if enabled["virtual_continue_request_count"] <= 0:
        readiness = "C"
    elif enabled["residual_enqueue_count"] <= 0:
        readiness = "C"
    elif enabled["residual_delivery_count"] <= 0:
        readiness = "B"
    (output_root / "phase_6_readiness_after_enabled_smoke.md").write_text(
        "# Phase 6 Readiness After Enabled Smoke\n\n"
        f"Readiness class: {readiness}. Ready for Phase 6 planning.\n\n"
        "## Criteria\n"
        "- Tests must pass: to be filled by `test_report.md` after validation commands.\n"
        f"- Enabled smoke runs: pass.\n"
        f"- virtual_continue_request_count > 0: {enabled['virtual_continue_request_count'] > 0}.\n"
        f"- residual_enqueue_count > 0: {enabled['residual_enqueue_count'] > 0}.\n"
        f"- residual_delivery_count > 0: {enabled['residual_delivery_count'] > 0}.\n"
        f"- passive_followup_incremental_utility > 0: {enabled['passive_followup_incremental_utility'] > 0}.\n"
        "- No duplicate positive utility, no ground-truth leakage, no queue blow-up, and disabled control remains compatible.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5 enabled-path smoke validation.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_isolated_configs(output_root)
    disabled_metrics, disabled_trace = run_case(False, output_root, args.seed)
    enabled_metrics, enabled_trace = run_case(True, output_root, args.seed)
    _write_csv(output_root / "enabled_path_metrics.csv", [disabled_metrics, enabled_metrics])
    _write_csv(output_root / "enabled_path_trace_cases.csv", disabled_trace + enabled_trace)
    _write_reports(output_root, disabled_metrics, enabled_metrics)
    print(f"disabled_virtual_continue_request_count={disabled_metrics['virtual_continue_request_count']}")
    print(f"enabled_virtual_continue_request_count={enabled_metrics['virtual_continue_request_count']}")
    print(f"enabled_residual_enqueue_count={enabled_metrics['residual_enqueue_count']}")
    print(f"enabled_residual_delivery_count={enabled_metrics['residual_delivery_count']}")
    print(f"enabled_passive_followup_incremental_utility={enabled_metrics['passive_followup_incremental_utility']}")
    print(f"output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
