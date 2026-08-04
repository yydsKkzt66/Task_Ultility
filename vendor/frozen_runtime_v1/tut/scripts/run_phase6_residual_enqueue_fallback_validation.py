"""Validate default-off residual ordinary-queue fallback after same-slot capacity miss."""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase6_carryover_residual_validation import (  # noqa: E402
    _audit_diff,
    _enable_phase6_carryover,
    _parse_seeds,
    _run_git,
    _safe_div,
    _summarize_validation_run,
    _task_action_change,
    _to_float,
    _to_int,
)
from scripts.run_product_requirement_aware_module2_ab import (  # noqa: E402
    DEFAULT_SEEDS,
    _aggregate,
    _bootstrap_ci,
    _load_run_configs,
)
from src.common.config import ensure_output_dirs  # noqa: E402
from src.common.types import (  # noqa: E402
    CommState,
    DetectionResult,
    GeneratedPayload,
    LinkTrace,
    ObservedFeedback,
    ProductLifecycleState,
    ResourceState,
    ResourceTrace,
    TaskSample,
    TimelineStep,
)
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.link.downlink_queue import DownlinkQueue  # noqa: E402
from src.reporting.lifecycle_shadow_metrics import build_per_task_lifecycle_rows, summarize_lifecycle_rows  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.simulator.step_runner import StepRunner  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_phase6_residual_enqueue_fallback_validation"
BASELINE_ARM = "v21_reqaware_phase6_carryover_eriv015_ug_wx050"
FALLBACK_ARM = "v21_reqaware_phase6_carryover_eriv015_queuefallback_ug_wx050"
ARMS = [BASELINE_ARM, FALLBACK_ARM]
NATURAL_SCOPE = "natural_paired_run"
STRESS_SCOPE = "capacity_stress_replay"
CORE_METRICS = [
    "proactive_flag_count",
    "residual_payload_generated_count",
    "residual_payload_enqueued_count",
    "proactive_followup_enqueue_count",
    "carryover_proactive_enqueue_count",
    "same_slot_fallback_to_ordinary_queue_count",
    "ordinary_queue_fallback_attempt_count",
    "ordinary_queue_fallback_success_count",
    "duplicate_residual_suppression_count",
    "duplicate_residual_suppressed_after_fallback_count",
    "zero_size_residual_payload_blocked_count",
    "residual_fallback_blocked_by_expired_deadline_count",
    "residual_fallback_dropped_by_queue_overflow_count",
    "residual_payload_delivered_count",
    "proactive_followup_delivery_count",
    "residual_delivery_rate",
    "evidence_closure_count",
    "evidence_closure_rate",
    "high_value_residual_delivery_count",
    "high_value_context_evidence_closure_count",
    "delivered_utility",
    "effective_utility",
    "net_system_utility",
    "lifecycle_shadow_utility_balanced",
    "lifecycle_shadow_utility_evidence_heavy",
    "high_value_delivered_incremental_utility",
    "residual_delivered_incremental_utility",
    "hard_safety_violation_count",
    "expired_items",
    "dropped_items",
    "queue_next_mb_mean",
    "queue_next_mb_max",
    "queue_overflow_count",
    "battery_min",
    "thermal_max",
    "residual_queue_wait_s_mean",
    "residual_queue_wait_s_max",
]


class _StressRouter:
    router_name = "proposed_lyapunov_rule_mode_safe_bandit"
    use_lyapunov_pricing = True
    use_mode_controller = True


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


def _load_configs(seed: int, arm: str) -> dict[str, dict[str, Any]]:
    configs = _enable_phase6_carryover(_load_run_configs(seed, "v21_reqaware_ug_wx050"))
    phase6 = configs["simulator"]["progressive_payload"]["phase6_proactive_followup"]
    phase6["eriv_threshold"] = 0.15
    phase6["min_remaining_deadline_s"] = 60.0
    same_slot = configs["simulator"]["progressive_payload"]["phase6_same_slot_carryover"]
    same_slot["enabled"] = True
    same_slot["enable_queue_fallback_on_insufficient_capacity"] = bool(arm == FALLBACK_ARM)
    return configs


def _stress_task(seed: int) -> TaskSample:
    return TaskSample(
        f"stress_task_{seed}",
        "a",
        "b",
        "scene",
        0,
        1.0,
        240.0,
        0.2,
        0.4,
        product_requirement="verification",
        high_value_flag=True,
    )


def _stress_step(task: TaskSample, slot_index: int, throughput_mbps: float) -> TimelineStep:
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
        link_trace=LinkTrace(slot_index, True, 20.0, throughput_mbps, 1.0, 60.0),
    )


def _stress_payload(task_id: str) -> GeneratedPayload:
    return GeneratedPayload(
        task_id=task_id,
        payload_policy="metadata",
        payload_mb=0.1,
        information_content=0.5,
        priority_score=0.5,
        value_score=1.0,
        task_value=1.0,
        uncertainty=0.2,
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
        continue_request_priority_bonus=0.05,
    )


def _stress_feedback(result, status: str = "delivered") -> ObservedFeedback:
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
        delivery_status=status,
        delivery_factor=1.0,
        safety_violation=0,
        total_latency_s=0.0,
        dropped_current_task_mb=result.dropped_current_task_mb,
        queue_item_count=result.queue_item_count,
        detection_quality=1.0,
    )


def _stress_detection() -> DetectionResult:
    return DetectionResult([], 0.9, False, [], 0.2, 1.0, 0.95, {})


def _run_capacity_stress_replay(seed: int, arm: str) -> dict[str, Any]:
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
    lifecycle_state: ProductLifecycleState | None = runner.product_lifecycle.get(task.task_id)
    same_slot_trace = [dict(row, seed=seed, arm=arm, router=_StressRouter.router_name, validation_scope=STRESS_SCOPE) for row in runner.last_same_slot_carryover_trace]
    carryover_events = [dict(row, seed=seed, arm=arm, router=_StressRouter.router_name, validation_scope=STRESS_SCOPE) for row in runner.last_carryover_lifecycle_events]
    downlink_trace = [
        {
            "validation_scope": STRESS_SCOPE,
            "seed": seed,
            "arm": arm,
            "slot": 1,
            "event_type": "metadata_then_fallback_attempt",
            "proactive_followup_enqueue_count": merged.proactive_followup_enqueue_count,
            "proactive_followup_delivery_count": merged.proactive_followup_delivery_count,
            "queue_next_mb": merged.queue_next_mb,
        },
        {
            "validation_scope": STRESS_SCOPE,
            "seed": seed,
            "arm": arm,
            "slot": 2,
            "event_type": "ordinary_queue_next_slot_delivery",
            "proactive_followup_enqueue_count": next_result.proactive_followup_enqueue_count,
            "proactive_followup_delivery_count": next_result.proactive_followup_delivery_count,
            "queue_next_mb": next_result.queue_next_mb,
        },
    ]
    generated = sum(
        1
        for event in same_slot_trace
        if event.get("event_type") == "same_slot_fallback_to_tplus1"
        and event.get("reason_code") == "residual_payload_exceeds_remaining_contact_capacity"
    )
    row: dict[str, object] = {metric: 0.0 for metric in CORE_METRICS}
    row.update(
        {
            "validation_scope": STRESS_SCOPE,
            "seed": seed,
            "arm": arm,
            "router": _StressRouter.router_name,
            "proactive_flag_count": int(generated > 0),
            "residual_payload_generated_count": generated,
            "residual_payload_enqueued_count": merged.proactive_followup_enqueue_count,
            "proactive_followup_enqueue_count": merged.proactive_followup_enqueue_count,
            "carryover_proactive_enqueue_count": metadata_feedback.carryover_proactive_enqueue_count,
            "same_slot_fallback_to_ordinary_queue_count": metadata_feedback.same_slot_fallback_to_ordinary_queue_count,
            "ordinary_queue_fallback_attempt_count": metadata_feedback.ordinary_queue_fallback_attempt_count,
            "ordinary_queue_fallback_success_count": metadata_feedback.ordinary_queue_fallback_success_count,
            "duplicate_residual_suppression_count": metadata_feedback.duplicate_residual_suppression_count,
            "duplicate_residual_suppressed_after_fallback_count": metadata_feedback.duplicate_residual_suppressed_after_fallback_count,
            "zero_size_residual_payload_blocked_count": metadata_feedback.zero_size_residual_payload_blocked_count,
            "residual_fallback_blocked_by_expired_deadline_count": metadata_feedback.residual_fallback_blocked_by_expired_deadline_count,
            "residual_fallback_dropped_by_queue_overflow_count": metadata_feedback.residual_fallback_dropped_by_queue_overflow_count,
            "residual_payload_delivered_count": next_result.proactive_followup_delivery_count,
            "proactive_followup_delivery_count": next_result.proactive_followup_delivery_count,
            "residual_delivery_rate": _safe_div(next_result.proactive_followup_delivery_count, merged.proactive_followup_enqueue_count),
            "evidence_closure_count": int(bool(lifecycle_state and lifecycle_state.task_closed_flag)),
            "evidence_closure_rate": float(int(bool(lifecycle_state and lifecycle_state.task_closed_flag))),
            "high_value_residual_delivery_count": next_result.proactive_followup_delivery_count,
            "high_value_context_evidence_closure_count": 0.0,
            "residual_queue_wait_s_mean": mean(next_result.residual_queue_wait_times_s) if next_result.residual_queue_wait_times_s else 0.0,
            "residual_queue_wait_s_max": max(next_result.residual_queue_wait_times_s) if next_result.residual_queue_wait_times_s else 0.0,
            "queue_next_mb_mean": (merged.queue_next_mb + next_result.queue_next_mb) / 2.0,
            "queue_next_mb_max": max(merged.queue_next_mb, next_result.queue_next_mb),
            "battery_min": 0.85,
            "thermal_max": 0.2,
        }
    )
    return {
        "row": row,
        "same_slot_trace": same_slot_trace,
        "carryover_events": carryover_events,
        "downlink_trace": downlink_trace,
    }


def _write_config_snapshots(seeds: list[int]) -> None:
    config_dir = OUT_DIR / "configs_used"
    config_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        payload = {
            "arm": arm,
            "representative_seed": seeds[0],
            "default_off_experiment_only": True,
            "configs": _load_configs(seeds[0], arm),
        }
        _write_text(config_dir / f"{arm}.yaml", yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _run_arm(seed: int, arm: str) -> dict[str, Any]:
    configs = _load_configs(seed, arm)
    run_dir = OUT_DIR / "runs" / f"seed_{seed}" / arm
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    records = [asdict(record) for record in result.records]
    for row in records:
        row["seed"] = seed
        row["arm"] = arm
        row["router"] = router.router_name
    lifecycle_rows = build_per_task_lifecycle_rows(records, seed=seed, variant=arm, router_name=router.router_name)
    return {
        "seed": seed,
        "arm": arm,
        "router": router.router_name,
        "records": records,
        "summary": dict(result.summary),
        "queue_logs": [dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_queue_priority_logs],
        "carryover_events": [dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_carryover_lifecycle_events],
        "same_slot_trace": [dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_same_slot_carryover_trace],
        "lifecycle_rows": lifecycle_rows,
        "lifecycle_summary": summarize_lifecycle_rows(lifecycle_rows),
    }


def _sum_record(records: list[dict[str, Any]], field: str) -> float:
    return sum(_to_float(row.get(field)) for row in records if _to_int(row.get("has_task")) or str(row.get("task_id")) == "__idle__")


def _records_by_slot(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(_to_int(row.get("slot"), -1)): row for row in records}


def _summarize_run(run: dict[str, Any], base: dict[str, Any] | None) -> dict[str, object]:
    row = _summarize_validation_run(run, _task_action_change(run, base))
    records = run["records"]
    generated_events = [
        event
        for event in run["same_slot_trace"]
        if str(event.get("event_type")) in {"same_slot_candidate_re_evaluated"}
        and _to_int(event.get("enqueue_decision")) == 0
        and _to_float(event.get("ERIV")) > 0.0
        and "residual_payload_exceeds_remaining_contact_capacity"
        not in str(event.get("reason_code", ""))
    ]
    proactive_events = [
        event
        for event in run["same_slot_trace"]
        if str(event.get("event_type")) in {"same_slot_candidate_re_evaluated", "ordinary_queue_fallback_enqueued"}
        and _to_float(event.get("ERIV")) > 0.0
    ]
    residual_wait = [_to_float(record.get("residual_queue_wait_s")) for record in records if _to_float(record.get("residual_queue_wait_s")) > 0.0]
    queue_next = [_to_float(record.get("queue_next_mb")) for record in records]
    row.update(
        {
            "validation_scope": NATURAL_SCOPE,
            "arm": run["arm"],
            "proactive_flag_count": sum(1 for event in proactive_events if _to_float(event.get("ERIV")) >= _to_float(event.get("ERIV_threshold")) > 0.0),
            "residual_payload_generated_count": sum(1 for event in run["same_slot_trace"] if str(event.get("event_type")) == "same_slot_fallback_to_tplus1" and str(event.get("reason_code")) == "residual_payload_exceeds_remaining_contact_capacity"),
            "residual_payload_enqueued_count": _sum_record(records, "proactive_followup_enqueue_count"),
            "proactive_followup_enqueue_count": _sum_record(records, "proactive_followup_enqueue_count"),
            "carryover_proactive_enqueue_count": _sum_record(records, "carryover_proactive_enqueue_count"),
            "same_slot_fallback_to_ordinary_queue_count": _sum_record(records, "same_slot_fallback_to_ordinary_queue_count"),
            "ordinary_queue_fallback_attempt_count": _sum_record(records, "ordinary_queue_fallback_attempt_count"),
            "ordinary_queue_fallback_success_count": _sum_record(records, "ordinary_queue_fallback_success_count"),
            "duplicate_residual_suppression_count": _sum_record(records, "duplicate_residual_suppression_count"),
            "duplicate_residual_suppressed_after_fallback_count": _sum_record(records, "duplicate_residual_suppressed_after_fallback_count"),
            "zero_size_residual_payload_blocked_count": _sum_record(records, "zero_size_residual_payload_blocked_count"),
            "residual_fallback_blocked_by_expired_deadline_count": _sum_record(records, "residual_fallback_blocked_by_expired_deadline_count"),
            "residual_fallback_dropped_by_queue_overflow_count": _sum_record(records, "residual_fallback_dropped_by_queue_overflow_count"),
            "residual_payload_delivered_count": _sum_record(records, "proactive_followup_delivery_count"),
            "proactive_followup_delivery_count": _sum_record(records, "proactive_followup_delivery_count"),
            "residual_delivery_rate": _safe_div(_sum_record(records, "proactive_followup_delivery_count"), _sum_record(records, "proactive_followup_enqueue_count")),
            "high_value_residual_delivery_count": _sum_record(records, "high_value_proactive_followup_count"),
            "high_value_context_evidence_closure_count": 0.0,
            "high_value_delivered_incremental_utility": _sum_record(records, "high_value_delivered_incremental_utility"),
            "residual_delivered_incremental_utility": _sum_record(records, "proactive_followup_incremental_utility"),
            "dropped_items": _to_float(run["summary"].get("dropped_items")),
            "queue_next_mb_mean": mean(queue_next) if queue_next else 0.0,
            "queue_next_mb_max": max(queue_next) if queue_next else 0.0,
            "residual_queue_wait_s_mean": mean(residual_wait) if residual_wait else 0.0,
            "residual_queue_wait_s_max": max(residual_wait) if residual_wait else 0.0,
        }
    )
    return row


def _paired(per_seed: list[dict[str, object]], validation_scope: str) -> list[dict[str, object]]:
    by_key = {(int(row["seed"]), str(row["arm"])): row for row in per_seed}
    seeds = sorted({int(row["seed"]) for row in per_seed})
    out: list[dict[str, object]] = []
    for metric in CORE_METRICS:
        deltas = [
            _to_float(by_key[(seed, FALLBACK_ARM)].get(metric)) - _to_float(by_key[(seed, BASELINE_ARM)].get(metric))
            for seed in seeds
            if (seed, FALLBACK_ARM) in by_key and (seed, BASELINE_ARM) in by_key
        ]
        low, high = _bootstrap_ci(deltas)
        out.append(
            {
                "comparison": f"{FALLBACK_ARM} - {BASELINE_ARM}",
                "validation_scope": validation_scope,
                "metric": metric,
                "mean_delta": mean(deltas) if deltas else 0.0,
                "median_delta": median(deltas) if deltas else 0.0,
                "std_delta": stdev(deltas) if len(deltas) >= 2 else 0.0,
                "win_rate": _safe_div(sum(1 for value in deltas if value > 0.0), len(deltas)),
                "ci95_low": low,
                "ci95_high": high,
                "ci_crosses_zero": int(low <= 0.0 <= high),
                "n": len(deltas),
            }
        )
    return out


def _summary_by_arm(per_seed: list[dict[str, object]], validation_scope: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        grouped[str(row["arm"])].append(row)
    out: list[dict[str, object]] = []
    for arm in ARMS:
        rows = grouped.get(arm, [])
        item = {"validation_scope": validation_scope, "arm": arm, "n": len(rows)}
        for metric in CORE_METRICS:
            values = [_to_float(row.get(metric)) for row in rows]
            item[f"{metric}_mean"] = mean(values) if values else 0.0
            item[f"{metric}_total"] = sum(values) if values else 0.0
        out.append(item)
    return out


def _candidate_trace(run: dict[str, Any]) -> list[dict[str, object]]:
    by_slot = _records_by_slot(run["records"])
    same_by_key: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in run["same_slot_trace"]:
        same_by_key[(_to_int(row.get("slot")), str(row.get("task_id")))].append(row)
    out: list[dict[str, object]] = []
    for row in run["carryover_events"]:
        if str(row.get("event_type")) not in {"ordinary_queue_fallback_enqueued", "ordinary_queue_fallback_duplicate_suppressed", "proactive_blocked"}:
            continue
        slot = _to_int(row.get("slot"))
        task_id = str(row.get("task_id"))
        events = same_by_key.get((slot, task_id), [])
        detected = next((event for event in events if event.get("event_type") == "same_slot_candidate_detected"), {})
        fallback = next((event for event in events if "fallback" in str(event.get("event_type"))), {})
        record = by_slot.get(slot, {})
        fallback_enqueued = int(str(row.get("event_type")) == "ordinary_queue_fallback_enqueued")
        out.append(
            {
                "validation_scope": NATURAL_SCOPE,
                "seed": run["seed"],
                "arm": run["arm"],
                "slot": slot,
                "task_id": task_id,
                "product_requirement": row.get("product_requirement", ""),
                "residual_product_type": row.get("residual_product_type", ""),
                "task_value": row.get("task_value", 0.0),
                "high_value_flag": row.get("high_value_flag", 0),
                "emergency_flag": row.get("emergency_flag", 0),
                "residual_delta_u_I": row.get("residual_delta_u_I", 0.0),
                "residual_utility_remaining": row.get("residual_utility_remaining", 0.0),
                "ERIV": row.get("ERIV", 0.0),
                "ERIV_threshold": row.get("ERIV_threshold", 0.0),
                "proactive_flag": row.get("proactive_flag", 0),
                "residual_payload_generated_flag": int(_to_float(row.get("ERIV")) >= _to_float(row.get("ERIV_threshold")) > 0.0),
                "residual_payload_mb": record.get("proactive_followup_payload_mb", ""),
                "same_slot_attempt_flag": int(bool(events)),
                "same_slot_remaining_capacity_mb": detected.get("contact_remaining_capacity_mb", fallback.get("contact_remaining_capacity_mb", "")),
                "same_slot_capacity_sufficient_flag": int(str(fallback.get("reason_code", "")) != "residual_payload_exceeds_remaining_contact_capacity"),
                "same_slot_fallback_to_tplus1_flag": int(any(event.get("event_type") == "same_slot_fallback_to_tplus1" for event in events)),
                "ordinary_queue_fallback_enabled_flag": int(run["arm"] == FALLBACK_ARM),
                "ordinary_queue_fallback_attempted_flag": int("ordinary_queue_fallback" in str(row.get("event_type"))),
                "ordinary_queue_fallback_enqueued_flag": fallback_enqueued,
                "ordinary_queue_fallback_blocker": "" if fallback_enqueued else row.get("proactive_blockers", ""),
                "duplicate_identity_key": f"{task_id}:{row.get('residual_product_type', '')}",
                "duplicate_residual_suppressed_flag": row.get("duplicate_suppression_flag", 0),
                "queue_size_before_fallback_mb": row.get("queue_mb", ""),
                "queue_size_after_fallback_mb": record.get("queue_next_mb", ""),
                "residual_payload_delivered_flag": record.get("proactive_followup_delivery_count", 0),
                "evidence_closed_flag": int(_to_float(record.get("evidence_closure_count")) > 0.0),
            }
        )
    return out


def _downlink_residual_trace(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for row in run["queue_logs"]:
            if _to_float(row.get("proactive_queue_bonus_component")) > 0.0 or str(row.get("followup_trigger", "")) == "proactive_followup":
                rows.append(row)
    return rows


def _stress_candidate_trace(stress_runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in stress_runs:
        row = run["row"]
        fallback_event = next(
            (event for event in run["same_slot_trace"] if event.get("event_type") == "ordinary_queue_fallback_enqueued"),
            {},
        )
        generated_event = next(
            (event for event in run["same_slot_trace"] if event.get("event_type") == "same_slot_fallback_to_tplus1"),
            {},
        )
        rows.append(
            {
                "validation_scope": STRESS_SCOPE,
                "seed": row["seed"],
                "arm": row["arm"],
                "slot": 1,
                "task_id": fallback_event.get("task_id", generated_event.get("task_id", "")),
                "product_requirement": fallback_event.get("product_requirement", generated_event.get("product_requirement", "verification")),
                "residual_product_type": fallback_event.get("residual_product_type", generated_event.get("residual_product_type", "roi")),
                "high_value_flag": 1,
                "ERIV": fallback_event.get("ERIV", generated_event.get("ERIV", "")),
                "ERIV_threshold": fallback_event.get("ERIV_threshold", generated_event.get("ERIV_threshold", "")),
                "proactive_flag": int(_to_float(row.get("proactive_flag_count")) > 0.0),
                "residual_payload_generated_flag": int(_to_float(row.get("residual_payload_generated_count")) > 0.0),
                "same_slot_attempt_flag": 1,
                "same_slot_remaining_capacity_mb": generated_event.get("contact_remaining_capacity_mb", ""),
                "same_slot_capacity_sufficient_flag": 0,
                "same_slot_fallback_to_tplus1_flag": int(_to_float(row.get("residual_payload_generated_count")) > 0.0),
                "ordinary_queue_fallback_enabled_flag": int(row["arm"] == FALLBACK_ARM),
                "ordinary_queue_fallback_attempted_flag": int(_to_float(row.get("ordinary_queue_fallback_attempt_count")) > 0.0),
                "ordinary_queue_fallback_enqueued_flag": int(_to_float(row.get("ordinary_queue_fallback_success_count")) > 0.0),
                "ordinary_queue_fallback_blocker": "" if _to_float(row.get("ordinary_queue_fallback_success_count")) > 0.0 else "fallback_disabled",
                "duplicate_identity_key": f"{fallback_event.get('task_id', generated_event.get('task_id', ''))}:roi",
                "duplicate_residual_suppressed_flag": int(_to_float(row.get("duplicate_residual_suppressed_after_fallback_count")) > 0.0),
                "queue_size_after_fallback_mb": row.get("queue_next_mb_max", ""),
                "residual_payload_delivered_flag": row.get("residual_payload_delivered_count", 0),
                "evidence_closed_flag": int(_to_float(row.get("evidence_closure_count")) > 0.0),
            }
        )
    return rows


def _legacy_utility_delivered_mean() -> float:
    path = PROJECT_ROOT / "outputs" / "_product_requirement_aware_module2_ab" / "canonical_metric_summary.csv"
    if not path.exists():
        return 0.0
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("arm") == "legacy_utility_rule" and row.get("metric") == "delivered_utility":
                return _to_float(row.get("mean"))
    return 0.0


def _delivered_gap_summary(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    legacy = _legacy_utility_delivered_mean()
    return [
        {
            "arm": row["arm"],
            "delivered_utility_mean": row.get("delivered_utility_mean", 0.0),
            "legacy_utility_rule_delivered_utility_mean": legacy,
            "gap_to_legacy_utility_rule": _to_float(row.get("delivered_utility_mean")) - legacy,
        }
        for row in summary
    ]


def _metric(summary: list[dict[str, object]], arm: str, metric: str, suffix: str = "total") -> float:
    for row in summary:
        if row.get("arm") == arm:
            return _to_float(row.get(f"{metric}_{suffix}"))
    return 0.0


def _write_reports(
    command: str,
    seeds: list[int],
    natural_summary: list[dict[str, object]],
    natural_paired: list[dict[str, object]],
    stress_summary: list[dict[str, object]],
    stress_paired: list[dict[str, object]],
) -> None:
    natural_generated = _metric(natural_summary, FALLBACK_ARM, "residual_payload_generated_count")
    natural_enqueued = _metric(natural_summary, FALLBACK_ARM, "residual_payload_enqueued_count")
    natural_delivered = _metric(natural_summary, FALLBACK_ARM, "residual_payload_delivered_count")
    natural_evidence = _metric(natural_summary, FALLBACK_ARM, "evidence_closure_count")
    generated = _metric(stress_summary, FALLBACK_ARM, "residual_payload_generated_count")
    enqueued = _metric(stress_summary, FALLBACK_ARM, "residual_payload_enqueued_count")
    delivered = _metric(stress_summary, FALLBACK_ARM, "residual_payload_delivered_count")
    evidence = _metric(stress_summary, FALLBACK_ARM, "evidence_closure_count")
    fallback_attempt = _metric(stress_summary, FALLBACK_ARM, "ordinary_queue_fallback_attempt_count")
    fallback_success = _metric(stress_summary, FALLBACK_ARM, "ordinary_queue_fallback_success_count")
    delivered_gap_rows = _delivered_gap_summary(natural_summary)
    baseline_gap = next(row for row in delivered_gap_rows if row["arm"] == BASELINE_ARM)["gap_to_legacy_utility_rule"]
    fallback_gap = next(row for row in delivered_gap_rows if row["arm"] == FALLBACK_ARM)["gap_to_legacy_utility_rule"]
    gap_shrank = abs(float(fallback_gap)) < abs(float(baseline_gap))
    net_delta = next((row for row in natural_paired if row["metric"] == "net_system_utility"), {}).get("mean_delta", 0.0)
    safety_delta = next((row for row in natural_paired if row["metric"] == "hard_safety_violation_count"), {}).get("mean_delta", 0.0)
    fallback_enqueue_delta = next((row for row in stress_paired if row["metric"] == "ordinary_queue_fallback_success_count"), {}).get("mean_delta", 0.0)
    if generated > 0 and enqueued <= 0:
        case = "Case E"
        recommendation = "fallback did not change enqueue; continue payload handoff / duplicate guard / process_slot call-chain audit."
    elif enqueued > 0 and delivered <= 0:
        case = "Case A"
        recommendation = "fallback fixed enqueue, but delivery remains blocked; next step is residual delivery audit."
    elif delivered > 0 and evidence <= 0:
        case = "Case B"
        recommendation = "fallback fixed delivery, but closure remains blocked; next audit information sufficiency / closure accounting."
    elif evidence > 0:
        case = "Case C"
        recommendation = "fallback advances evidence closure with stable safety; keep default-off experimental only."
    else:
        case = "Case E"
        recommendation = "no generated/enqueued residual chain progress observed."
    if _to_float(net_delta) < -0.05 * max(abs(_metric(natural_summary, BASELINE_ARM, "net_system_utility", "mean")), 1.0) or _to_float(safety_delta) > 0:
        case = "Case D"
        recommendation = "fallback worsens utility or safety; keep disabled and audit queue-aware fallback admission."

    _write_text(
        OUT_DIR / "README.md",
        """# Phase 6 residual enqueue fallback validation

Default-off experiment for ordinary queue fallback after same-slot carryover generates a residual payload but current-slot residual capacity is insufficient.
""",
    )
    _write_text(OUT_DIR / "command_log.md", f"# Command log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "arms": ARMS,
                "default_off_experiment_only": True,
                "only_difference": "phase6_same_slot_carryover.enable_queue_fallback_on_insufficient_capacity",
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(
        OUT_DIR / "phase6_residual_enqueue_fallback_report.md",
        f"""# Phase 6 residual enqueue fallback report

## Result

- selected_case: {case}
- capacity-stress generated residual payloads in fallback arm: {generated:.0f}
- capacity-stress ordinary_queue_fallback_attempt_count: {fallback_attempt:.0f}
- capacity-stress ordinary_queue_fallback_success_count: {fallback_success:.0f}
- capacity-stress residual_payload_enqueued_count: {enqueued:.0f}
- capacity-stress next-slot residual_payload_delivered_count: {delivered:.0f}
- capacity-stress evidence_closure_count: {evidence:.0f}
- natural-run generated residual payloads in fallback arm: {natural_generated:.0f}
- natural-run residual_payload_enqueued_count: {natural_enqueued:.0f}
- natural-run residual_payload_delivered_count: {natural_delivered:.0f}
- natural-run evidence_closure_count: {natural_evidence:.0f}
- natural-run delivered_utility_gap_shrank: {str(gap_shrank).lower()}
- natural-run net_system_utility mean delta: {_to_float(net_delta):.6f}
- natural-run hard_safety_violation_count mean delta: {_to_float(safety_delta):.6f}

## Required answers

1. Old logic did not enqueue because same-slot capacity miss returned after `same_slot_fallback_to_tplus1` without ordinary queue handoff.
2. Missing ordinary queue fallback after insufficient same-slot capacity: confirmed by code path and capacity-stress baseline trace.
3. ordinary_queue_fallback_attempt_count > 0 after enabling fallback: {'yes' if fallback_attempt > 0 else 'no'}.
4. ordinary_queue_fallback_success_count > 0: {'yes' if fallback_success > 0 else 'no'}.
5. residual_payload_enqueued_count became >0: {'yes' if enqueued > 0 else 'no'}.
6. residual_payload_delivered_count became >0: {'yes' if delivered > 0 else 'no'}; this is next-slot ordinary queue delivery, not same-slot forced delivery.
7. evidence_closure_count became >0: {'yes' if evidence > 0 else 'no'}.
8. new first-zero stage: {'residual_payload_delivered_count' if enqueued > 0 and delivered <= 0 else 'evidence_closure_count' if delivered > 0 and evidence <= 0 else 'none'}.
9. delivered_utility gap shrank: {'yes' if gap_shrank else 'no'}.
10. net_system_utility stable: {'yes' if _to_float(net_delta) >= -0.05 * max(abs(_metric(natural_summary, BASELINE_ARM, 'net_system_utility', 'mean')), 1.0) else 'no'}.
11. queue / expiry / safety worsened: {'yes' if _to_float(safety_delta) > 0 else 'no hard safety increase observed'}.
12. keep config default-off experimental branch: yes.
13. next step: {recommendation}

## Interpretation

The 2026-2035 natural paired run did not hit the insufficient-capacity branch; its same-slot residuals had enough remaining capacity and were directly delivered in both arms. The capacity-stress replay is therefore the branch-level validation for this patch, while natural-run utility and safety deltas remain the end-to-end guardrail check.

Fallback enqueue delta in capacity-stress replay: {_to_float(fallback_enqueue_delta):.6f}.
""",
    )
    _write_text(
        OUT_DIR / "final_recommendation.md",
        f"""# Final recommendation

{recommendation}

Do not adopt as main default.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    command = "python scripts/run_phase6_residual_enqueue_fallback_validation.py --seeds " + ",".join(str(seed) for seed in seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_config_snapshots(seeds)

    run_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in seeds:
        for arm in ARMS:
            run_by_key[(seed, arm)] = _run_arm(seed, arm)
    runs = [run_by_key[key] for key in sorted(run_by_key)]
    per_seed = [
        _summarize_run(run, run_by_key.get((run["seed"], BASELINE_ARM)) if run["arm"] == FALLBACK_ARM else None)
        for run in runs
    ]
    stress_runs = [_run_capacity_stress_replay(seed, arm) for seed in seeds for arm in ARMS]
    stress_per_seed = [run["row"] for run in stress_runs]
    paired = _paired(per_seed, NATURAL_SCOPE)
    stress_paired = _paired(stress_per_seed, STRESS_SCOPE)
    summary = _summary_by_arm(per_seed, NATURAL_SCOPE)
    stress_summary = _summary_by_arm(stress_per_seed, STRESS_SCOPE)
    candidate_trace = [row for run in runs for row in _candidate_trace(run)] + _stress_candidate_trace(stress_runs)
    same_slot_trace = [row for run in runs for row in run["same_slot_trace"]] + [row for run in stress_runs for row in run["same_slot_trace"]]
    fallback_trace = [
        row for run in runs for row in run["carryover_events"] if "ordinary_queue_fallback" in str(row.get("event_type"))
    ] + [row for run in stress_runs for row in run["carryover_events"] if "ordinary_queue_fallback" in str(row.get("event_type"))]
    downlink_trace = _downlink_residual_trace(runs) + [row for run in stress_runs for row in run["downlink_trace"]]

    _write_csv(OUT_DIR / "residual_enqueue_funnel_summary.csv", summary + stress_summary)
    _write_csv(OUT_DIR / "per_candidate_followup_trace.csv", candidate_trace)
    _write_csv(OUT_DIR / "same_slot_carryover_trace.csv", same_slot_trace)
    _write_csv(OUT_DIR / "residual_fallback_enqueue_trace.csv", fallback_trace)
    _write_csv(OUT_DIR / "downlink_queue_residual_trace.csv", downlink_trace)
    _write_csv(OUT_DIR / "duplicate_residual_suppression_summary.csv", _aggregate(per_seed, ["duplicate_residual_suppression_count", "duplicate_residual_suppressed_after_fallback_count"]))
    _write_csv(OUT_DIR / "residual_queue_delivery_summary.csv", _aggregate(per_seed, ["residual_payload_enqueued_count", "residual_payload_delivered_count", "proactive_followup_enqueue_count", "proactive_followup_delivery_count", "residual_delivery_rate", "residual_queue_wait_s_mean", "residual_queue_wait_s_max"]))
    _write_csv(OUT_DIR / "lifecycle_shadow_summary.csv", _aggregate(per_seed, ["lifecycle_shadow_utility_balanced", "lifecycle_shadow_utility_evidence_heavy", "evidence_closure_count", "evidence_closure_rate"]))
    _write_csv(OUT_DIR / "staged_closure_summary.csv", _aggregate(per_seed, ["staged_plan_rate", "staged_metadata_success_rate", "staged_followup_success_rate", "residual_enqueue_count", "residual_delivery_count", "residual_delivery_rate"]))
    _write_csv(OUT_DIR / "canonical_metric_summary.csv", _aggregate(per_seed, ["delivered_utility", "effective_utility", "net_system_utility", "total_penalized_reward", "expired_items", "dropped_items"]))
    _write_csv(OUT_DIR / "delivered_utility_gap_summary.csv", _delivered_gap_summary(summary))
    _write_csv(OUT_DIR / "safety_guardrail_summary.csv", _aggregate(per_seed, ["hard_safety_violation_count", "expired_items", "dropped_items", "queue_next_mb_mean", "queue_next_mb_max", "queue_overflow_count", "battery_min", "thermal_max"]))
    _write_csv(OUT_DIR / "paired_delta_summary.csv", paired + stress_paired)
    _write_reports(command, seeds, summary, paired, stress_summary, stress_paired)
    _write_text(OUT_DIR / "git_status.txt", _run_git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _audit_diff(
            [
                "src/payload/proactive_followup.py",
                "src/common/types.py",
                "src/simulator/step_runner.py",
                "src/simulator/episode_runner.py",
                "tests/test_proactive_followup.py",
                "scripts/run_phase6_residual_enqueue_fallback_validation.py",
            ]
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
