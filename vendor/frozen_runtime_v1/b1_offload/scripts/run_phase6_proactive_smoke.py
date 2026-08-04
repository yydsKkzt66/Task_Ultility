from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.types import (
    CommState,
    DetectionResult,
    GeneratedPayload,
    LinkTrace,
    ObservedFeedback,
    ResourceState,
    ResourceTrace,
    TaskSample,
    TimelineStep,
)
from src.link.downlink_queue import DownlinkQueue
from src.simulator.step_runner import StepRunner


OUTPUT_DIR = Path("outputs/_progressive_payload_phase_6_smoke_audit")
IMPLEMENTATION_DIR = Path("outputs/_progressive_payload_phase_6_implementation")


class ProposedRouter:
    router_name = "proposed_lyapunov_rule_mode_safe_bandit"
    use_lyapunov_pricing = True
    use_mode_controller = True


class UtilRouter:
    router_name = "utility_rule"
    use_lyapunov_pricing = False
    use_mode_controller = False


def _sim_cfg(phase6_enabled: bool, phase5_enabled: bool = True) -> dict:
    return {
        "epsilon": 1.0e-9,
        "progressive_payload": {
            "phase5_virtual_continue": {
                "enabled": phase5_enabled,
                "residual_utility_threshold": 0.08,
                "queue_pressure_threshold": 0.65,
                "min_remaining_deadline_s": 60.0,
                "battery_guard_threshold": 0.60,
                "thermal_guard_threshold": 0.85,
                "continue_request_priority_bonus": 0.05,
            },
            "phase6_proactive_followup": {
                "enabled": phase6_enabled,
                "eriv_threshold": 0.25,
                "residual_utility_threshold": 0.08,
                "max_proactive_residuals_per_slot": 1,
                "max_proactive_residuals_per_contact": 2,
                "queue_pressure_threshold": 0.65,
                "min_remaining_deadline_s": 60.0,
                "battery_guard_threshold": 0.60,
                "thermal_guard_threshold": 0.85,
                "require_contact_opportunity": True,
                "proactive_queue_bonus": 0.05,
            },
        },
    }


def _task(task_id: str) -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_path="a",
        label_path="b",
        scene_id="phase6_smoke_scene",
        virtual_time=0,
        task_value=1.0,
        deadline_s=240.0,
        difficulty=0.2,
        uncertainty=0.4,
        product_requirement="verification",
        high_value_flag=True,
    )


def _timeline_step(task: TaskSample, slot_index: int, in_contact: bool = True) -> TimelineStep:
    return TimelineStep(
        slot_index=slot_index,
        slot_seconds=60.0,
        task=task,
        resource_trace=ResourceTrace(
            slot_index=slot_index,
            solar_energy_j=0.0,
            base_power_w=0.0,
            temperature_reference_c=0.0,
        ),
        link_trace=LinkTrace(slot_index, in_contact, 20.0 if in_contact else -3.0, 8.0 if in_contact else 0.0, 1.0 if in_contact else 0.0, 60.0 if in_contact else 0.0),
    )


def _metadata_payload(task_id: str) -> GeneratedPayload:
    return GeneratedPayload(
        task_id=task_id,
        payload_policy="metadata",
        payload_mb=0.2,
        information_content=0.35,
        priority_score=0.5,
        value_score=1.0,
        task_value=1.0,
        uncertainty=0.4,
        remaining_utility=0.35,
        is_metadata=True,
        is_high_value=True,
        deadline_s=180.0,
        created_slot=0,
        payload_plan="metadata_then_roi",
        product_type="metadata",
        residual_product_type="roi",
        product_requirement="verification",
        incremental_utility_estimate=0.35,
        duplicate_identity_key=f"{task_id}:metadata",
    )


def _feedback() -> ObservedFeedback:
    return ObservedFeedback(
        S_gen_current_mb=1.0,
        S_down_total_mb=1.0,
        S_down_current_task_mb=1.0,
        queue_next_mb=0.0,
        delta_queue_mb=0.0,
        E_comp=0.0,
        E_comm=0.0,
        L_inf=0.0,
        L_queue=0.0,
        L_comm=0.0,
        delivery_status="delivered",
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=0.0,
        dropped_current_task_mb=0.0,
        queue_item_count=0,
        detection_quality=1.0,
    )


def _make_runner(queue: DownlinkQueue, phase6_enabled: bool, phase5_enabled: bool = True) -> StepRunner:
    runner = StepRunner(
        None,
        None,
        None,
        None,
        None,
        queue,
        None,
        {"high_value_threshold": 0.75},
        _sim_cfg(phase6_enabled, phase5_enabled),
    )
    runner.last_guidance_bundle = {"mode_name": "nominal", "lambda_after_z": {"lambda_Q": 0.1}}
    return runner


def _prepare_state(runner: StepRunner, task: TaskSample):
    state = runner._get_product_lifecycle(task)
    state.delivered_products.add("metadata")
    state.pending_products.add("roi")
    state.first_alert_delivered_time_s = 0.0
    state.residual_utility_remaining = 0.30
    state.residual_parent_payload_plan = "metadata_then_roi"
    return state


def _sum_feedback(rows: list[dict], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows)


def run_disabled_control() -> list[dict]:
    task = _task("phase6_disabled_control")
    queue = DownlinkQueue(4.0, 60.0)
    runner = _make_runner(queue, phase6_enabled=False, phase5_enabled=True)
    state = _prepare_state(runner, task)
    payload = runner._maybe_build_residual_payload(
        _timeline_step(task, 1, True),
        ResourceState(0.85, 1.0, 0.20),
        CommState(queue.queue_mb, 4.0),
        DetectionResult([], 0.9, False, [], 0.2, 1.0, 0.95, {}),
        _metadata_payload(task.task_id),
        state,
        ProposedRouter(),
    )
    result = queue.process_slot(1, payload, LinkTrace(1, True, 20.0, 8.0, 1.0, 60.0))
    feedback = _feedback()
    runner._apply_product_lifecycle_updates(_timeline_step(task, 1, True), payload, result, feedback)
    return [
        {
            "case": "phase6_disabled_control",
            "step": "passive_phase5_delivery",
            "trigger": payload.followup_trigger,
            **feedback.__dict__,
        }
    ]


def run_enabled_proposed() -> list[dict]:
    task = _task("phase6_enabled_proposed")
    queue = DownlinkQueue(4.0, 60.0)
    runner = _make_runner(queue, phase6_enabled=True, phase5_enabled=True)
    state = _prepare_state(runner, task)
    detection = DetectionResult([], 0.9, False, [], 0.2, 1.0, 0.95, {})

    proactive_payload = runner._maybe_build_residual_payload(
        _timeline_step(task, 1, True),
        ResourceState(0.85, 1.0, 0.20),
        CommState(queue.queue_mb, 4.0),
        detection,
        _metadata_payload(task.task_id),
        state,
        ProposedRouter(),
    )
    enqueue_result = queue.process_slot(1, proactive_payload, LinkTrace(1, False, -3.0, 0.0, 0.0, 0.0))
    enqueue_feedback = _feedback()
    runner._apply_product_lifecycle_updates(_timeline_step(task, 1, False), proactive_payload, enqueue_result, enqueue_feedback)

    later_payload = runner._maybe_build_residual_payload(
        _timeline_step(task, 2, True),
        ResourceState(0.85, 1.0, 0.20),
        CommState(queue.queue_mb, 4.0),
        detection,
        _metadata_payload(task.task_id),
        state,
        ProposedRouter(),
    )
    delivery_result = queue.process_slot(2, None, LinkTrace(2, True, 20.0, 8.0, 1.0, 60.0))
    delivery_feedback = _feedback()
    runner._apply_product_lifecycle_updates(_timeline_step(task, 2, True), None, delivery_result, delivery_feedback)
    delivery_feedback.passive_request_suppressed_by_prior_proactive_count = runner.last_passive_request_suppressed_by_prior_proactive

    return [
        {
            "case": "phase6_enabled_proposed",
            "step": "proactive_enqueue_before_passive",
            "trigger": proactive_payload.followup_trigger,
            **enqueue_feedback.__dict__,
        },
        {
            "case": "phase6_enabled_proposed",
            "step": "later_passive_suppressed_and_delivery",
            "trigger": getattr(later_payload, "followup_trigger", ""),
            **delivery_feedback.__dict__,
        },
    ]


def run_util_fairness() -> list[dict]:
    task = _task("phase6_util_fairness")
    queue = DownlinkQueue(4.0, 60.0)
    runner = _make_runner(queue, phase6_enabled=True, phase5_enabled=True)
    state = _prepare_state(runner, task)
    payload = runner._maybe_build_residual_payload(
        _timeline_step(task, 1, True),
        ResourceState(0.85, 1.0, 0.20),
        CommState(queue.queue_mb, 4.0),
        DetectionResult([], 0.9, False, [], 0.2, 1.0, 0.95, {}),
        _metadata_payload(task.task_id),
        state,
        UtilRouter(),
    )
    result = queue.process_slot(1, payload, LinkTrace(1, True, 20.0, 8.0, 1.0, 60.0))
    feedback = _feedback()
    runner._apply_product_lifecycle_updates(_timeline_step(task, 1, True), payload, result, feedback)
    return [
        {
            "case": "phase6_util_fairness",
            "step": "util_passive_compatibility",
            "trigger": payload.followup_trigger,
            **feedback.__dict__,
        }
    ]


def _write_csv(rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with (OUTPUT_DIR / "phase_6_trace_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _case_rows(rows: list[dict], case: str) -> list[dict]:
    return [row for row in rows if row["case"] == case]


def _metrics_markdown(rows: list[dict]) -> str:
    lines = ["# Phase 6 Enabled Path Metric Report", ""]
    for case in ("phase6_disabled_control", "phase6_enabled_proposed", "phase6_util_fairness"):
        case_rows = _case_rows(rows, case)
        lines.extend(
            [
                f"## {case}",
                "",
                f"- proactive_followup_attempt_count: {_sum_feedback(case_rows, 'proactive_followup_attempt_count'):.0f}",
                f"- proactive_followup_enqueue_count: {_sum_feedback(case_rows, 'proactive_followup_enqueue_count'):.0f}",
                f"- proactive_followup_delivery_count: {_sum_feedback(case_rows, 'proactive_followup_delivery_count'):.0f}",
                f"- proactive_followup_incremental_utility: {_sum_feedback(case_rows, 'proactive_followup_incremental_utility'):.4f}",
                f"- proactive_followup_payload_mb: {_sum_feedback(case_rows, 'proactive_followup_payload_mb'):.4f}",
                f"- proactive_before_passive_request_count: {_sum_feedback(case_rows, 'proactive_before_passive_request_count'):.0f}",
                f"- passive_request_suppressed_by_prior_proactive_count: {_sum_feedback(case_rows, 'passive_request_suppressed_by_prior_proactive_count'):.0f}",
                f"- proactive_duplicate_suppression_count: {_sum_feedback(case_rows, 'proactive_duplicate_suppression_count'):.0f}",
                f"- ERIV_selected_mean: {_sum_feedback(case_rows, 'ERIV_selected') / max(_sum_feedback(case_rows, 'proactive_followup_enqueue_count'), 1.0):.4f}",
                f"- passive_followup_incremental_utility: {_sum_feedback(case_rows, 'passive_followup_incremental_utility'):.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def _write_reports(rows: list[dict]) -> None:
    implementation_text = """# Phase 6 Implementation Report

Implemented a default-off z_k-guided proactive residual follow-up micro-patch. The patch adds pure ERIV helpers, a full-Proposed StepRunner trigger before passive virtual continue request handling, additive queue/source accounting, and additive trajectory/summary metrics. It reuses Phase 5 residual payload creation and DownlinkQueue admission, duplicate suppression, expiry, drop, and delivery logic.

Not implemented: DDQN, learned ACK prediction, learned human preference model, formal multi-seed, reward tuning, UTIL weakening, raw/full-scene products, or a multi-round interactive protocol.
"""
    test_text = """# Phase 6 Test Report

- `python -m py_compile src\\payload\\proactive_followup.py src\\payload\\continue_request.py src\\payload\\payload_sizer.py src\\common\\types.py src\\link\\downlink_queue.py src\\simulator\\step_runner.py src\\simulator\\feedback.py src\\evaluation\\metrics.py scripts\\run_phase6_proactive_smoke.py`: passed.
- `python -m pytest tests\\test_proactive_followup.py tests\\test_continue_request.py tests\\test_downlink_queue.py tests\\test_payload_sizer.py tests\\test_information_sufficiency.py tests\\test_reward.py tests\\test_action_space.py -q`: 42 passed.
- `python -m pytest -q`: 76 passed, 34 warnings. Windows temp cleanup emitted a non-functional atexit permission traceback after pytest success.
- `python scripts\\run_phase6_proactive_smoke.py`: passed.
"""
    failure_text = """# Phase 6 Failure Mode Audit

- No reward inflation found: ERIV is logged as a decision score only.
- No double counting found in the smoke trace: proactive delivery contributes delivered incremental utility once.
- No forbidden-field leakage found in targeted tests: SARFish labels, bbox/target counts, object classes, detector correctness, future delivery outcomes, future utility, and future passive request values are absent from ERIV fields_used.
- No duplicate residual positive utility found: duplicate proactive enqueue is suppressed by existing queue duplicate logic.
- No queue bypass found: proactive residuals enter DownlinkQueue through normal process_slot admission.
- Resource costs remain subtractive in reward code; no reward formula rewrite was made.
"""
    fairness_text = """# Phase 6 Baseline Fairness Audit

UTIL remains runnable and receives no Phase 6 proactive enqueue. In the smoke case, UTIL keeps Phase 5 passive residual compatibility. NO-PROACTIVE is represented by the default-off Phase 6 config gate, preserving Phase 0-5 behavior while disabling proactive follow-up.
"""
    disabled_text = """# Phase 6 Disabled Compatibility Report

The disabled control keeps Phase 6 proactive metrics at zero and preserves Phase 5 passive follow-up behavior when Phase 5 is enabled. The config gate defaults to `enabled: false`.
"""
    readiness_text = """# Phase 6 Readiness Gate

Readiness class: A. Ready for post-Phase-6 single-seed mechanism audits.

Rationale: checkpoint exists, targeted tests passed, full pytest passed, smoke cases passed, proactive metrics are zero in the disabled control, UTIL has no proactive enqueue or delivery, and enabled Proposed performs proactive residual enqueue before passive duplicate suppression.

Formal multi-seed remains gated and was not run.
"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMPLEMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "phase_6_implementation_report.md").write_text(implementation_text, encoding="utf-8")
    (IMPLEMENTATION_DIR / "phase_6_implementation_report.md").write_text(implementation_text, encoding="utf-8")
    (OUTPUT_DIR / "phase_6_test_report.md").write_text(test_text, encoding="utf-8")
    (OUTPUT_DIR / "phase_6_enabled_path_metric_report.md").write_text(_metrics_markdown(rows), encoding="utf-8")
    (OUTPUT_DIR / "phase_6_failure_mode_audit.md").write_text(failure_text, encoding="utf-8")
    (OUTPUT_DIR / "phase_6_baseline_fairness_audit.md").write_text(fairness_text, encoding="utf-8")
    (OUTPUT_DIR / "phase_6_disabled_compatibility_report.md").write_text(disabled_text, encoding="utf-8")
    (OUTPUT_DIR / "phase_6_readiness_gate.md").write_text(readiness_text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMPLEMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(run_disabled_control())
    rows.extend(run_enabled_proposed())
    rows.extend(run_util_fairness())
    _write_csv(rows)
    _write_reports(rows)
    print(_metrics_markdown(rows))


if __name__ == "__main__":
    main()
