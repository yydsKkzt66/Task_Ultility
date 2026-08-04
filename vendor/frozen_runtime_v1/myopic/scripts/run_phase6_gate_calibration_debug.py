"""Default-off Phase 6 follow-up gate calibration/debug sweep."""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase6_carryover_residual_validation import (  # noqa: E402
    ARM_B as BASELINE_ARM,
    CORE_METRICS as BASE_CORE_METRICS,
    _audit_diff,
    _enable_phase6_carryover,
    _followup_blocker_summary,
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
    METRICS as BASE_METRICS,
    _aggregate,
    _bootstrap_ci,
    _load_run_configs,
)
from src.common.config import ensure_output_dirs  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.reporting.lifecycle_shadow_metrics import build_per_task_lifecycle_rows, summarize_lifecycle_rows  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_phase6_gate_calibration_debug"
BASELINE_VARIANT = "baseline_gate"
VARIANTS = [
    {"variant": "baseline_gate", "eriv_threshold": 0.25, "min_remaining_deadline_s": 60},
    {"variant": "eriv_020_deadline_60", "eriv_threshold": 0.20, "min_remaining_deadline_s": 60},
    {"variant": "eriv_015_deadline_60", "eriv_threshold": 0.15, "min_remaining_deadline_s": 60},
    {"variant": "eriv_015_deadline_45", "eriv_threshold": 0.15, "min_remaining_deadline_s": 45},
    {"variant": "eriv_015_deadline_30", "eriv_threshold": 0.15, "min_remaining_deadline_s": 30},
    {"variant": "eriv_010_deadline_30", "eriv_threshold": 0.10, "min_remaining_deadline_s": 30},
    {"variant": "eriv_010_deadline_20", "eriv_threshold": 0.10, "min_remaining_deadline_s": 20},
]
CORE_METRICS = [
    *BASE_CORE_METRICS,
    "proactive_flag_count",
    "eriv_below_threshold_count",
    "no_useful_remaining_deadline_count",
    "no_contact_opportunity_count",
    "queue_pressure_block_count",
    "battery_unsafe_block_count",
    "thermal_unsafe_block_count",
    "residual_payload_expired_count",
    "residual_payload_dropped_count",
    "residual_delivery_rate",
    "queue_next_mb_mean",
    "queue_next_mb_max",
    "residual_queue_wait_s_mean",
    "residual_queue_wait_s_max",
]
METRICS = list(dict.fromkeys([*BASE_METRICS, *CORE_METRICS]))


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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def _split_reasons(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.replace(",", ";").split(";") if item.strip()]


def _variant_cfg(variant: str) -> dict[str, object]:
    for cfg in VARIANTS:
        if cfg["variant"] == variant:
            return dict(cfg)
    raise ValueError(f"unknown variant: {variant}")


def _load_gate_configs(seed: int, variant: str) -> dict[str, dict[str, Any]]:
    cfg = _variant_cfg(variant)
    configs = _enable_phase6_carryover(_load_run_configs(seed, "v21_reqaware_ug_wx050"))
    phase6 = configs["simulator"]["progressive_payload"]["phase6_proactive_followup"]
    phase6["eriv_threshold"] = float(cfg["eriv_threshold"])
    phase6["min_remaining_deadline_s"] = float(cfg["min_remaining_deadline_s"])
    return configs


def _write_config_snapshots(seeds: list[int]) -> None:
    config_dir = OUT_DIR / "configs_used"
    config_dir.mkdir(parents=True, exist_ok=True)
    for variant in [str(row["variant"]) for row in VARIANTS]:
        payload = {
            "variant": variant,
            "base_arm": BASELINE_ARM,
            "representative_seed": seeds[0],
            "default_off_experiment_only": True,
            "configs": _load_gate_configs(seeds[0], variant),
        }
        _write_text(config_dir / f"{variant}.yaml", yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _run_variant(seed: int, variant: str) -> dict[str, Any]:
    configs = _load_gate_configs(seed, variant)
    run_dir = OUT_DIR / "runs" / f"seed_{seed}" / variant
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
        row["variant"] = variant
        row["arm"] = variant
        row["router"] = router.router_name
    lifecycle_rows = build_per_task_lifecycle_rows(records, seed=seed, variant=variant, router_name=router.router_name)
    return {
        "seed": seed,
        "variant": variant,
        "arm": variant,
        "router": router.router_name,
        "records": records,
        "summary": dict(result.summary),
        "queue_logs": [dict(row, seed=seed, variant=variant, arm=variant, router=router.router_name) for row in runner.last_queue_priority_logs],
        "carryover_events": [dict(row, seed=seed, variant=variant, arm=variant, router=router.router_name) for row in runner.last_carryover_lifecycle_events],
        "lifecycle_rows": lifecycle_rows,
        "lifecycle_summary": summarize_lifecycle_rows(lifecycle_rows),
    }


def _evaluated_events(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in run["carryover_events"]
        if str(event.get("event_type")) in {"candidate_re_evaluated", "same_slot_candidate_re_evaluated"}
    ]


def _event_counts(events: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for event in events:
        for reason in _split_reasons(event.get("proactive_blockers")):
            counter[reason] += 1
    return counter


def _record_by_slot(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_slot: dict[int, dict[str, Any]] = {}
    for row in records:
        by_slot[int(_to_int(row.get("slot"), -1))] = row
    return by_slot


def _trace_rows(run: dict[str, Any]) -> list[dict[str, object]]:
    by_slot = _record_by_slot(run["records"])
    by_task = {
        str(row.get("task_id")): row
        for row in run["records"]
        if _to_int(row.get("has_task")) and str(row.get("task_id", "")).strip()
    }
    rows: list[dict[str, object]] = []
    for event in _evaluated_events(run):
        slot = _to_int(event.get("slot"), -1)
        rec = by_slot.get(slot, {})
        task_rec = by_task.get(str(event.get("task_id", "")), {})
        deadline_s = _to_float(task_rec.get("deadline_s", rec.get("deadline_s")), 240.0) or 240.0
        remaining = _to_float(event.get("remaining_deadline_s"))
        contact_score = 1.0 if _to_int(event.get("contact_available")) else 0.0
        blockers = str(event.get("proactive_blockers", ""))
        generated = int(_to_int(event.get("proactive_flag")) > 0)
        enqueued = int(_to_int(event.get("proactive_enqueue_flag")) > 0)
        delivered = int(_to_int(event.get("proactive_delivery_flag")) > 0)
        model_quality = _to_float(task_rec.get("detection_quality", rec.get("detection_quality")), 1.0)
        if model_quality <= 0.0:
            model_quality = 1.0
        rows.append(
            {
                "seed": run["seed"],
                "variant": run["variant"],
                "slot": slot,
                "task_id": event.get("task_id", ""),
                "task_value": event.get("task_value", 0.0),
                "uncertainty": task_rec.get("U_t", rec.get("U_t", "")),
                "emergency_flag": event.get("emergency_flag", 0),
                "high_value_flag": event.get("high_value_flag", 0),
                "product_requirement": event.get("product_requirement", ""),
                "residual_product_type": event.get("residual_product_type", ""),
                "delivered_products_before_eval": event.get("delivered_products_before_eval", ""),
                "pending_products_before_eval": event.get("pending_products_before_eval", ""),
                "residual_utility_remaining": event.get("residual_utility_remaining", 0.0),
                "residual_delta_u_I": event.get("residual_delta_u_I", 0.0),
                "payload_size_mb_estimate": _safe_div(
                    _to_float(event.get("task_value"))
                    * _to_float(event.get("residual_delta_u_I"))
                    * _safe_div(remaining, deadline_s)
                    * contact_score
                    * model_quality,
                    _to_float(event.get("ERIV"), 0.0),
                )
                if _to_float(event.get("ERIV")) > 0.0
                else "",
                "remaining_deadline_s": remaining,
                "in_contact": event.get("contact_available", 0),
                "contact_opportunity_score": contact_score,
                "timeliness_factor": _safe_div(remaining, deadline_s),
                "model_quality_factor": model_quality,
                "z_k_followup_gate": 0 if "z_k_guided_followup_permission_blocked" in blockers else 1,
                "ERIV": event.get("ERIV", 0.0),
                "ERIV_threshold": event.get("ERIV_threshold", 0.0),
                "proactive_flag": event.get("proactive_flag", 0),
                "blocker_reason": blockers,
                "residual_payload_generated_flag": generated,
                "residual_payload_enqueued_flag": enqueued,
                "residual_payload_delivered_flag": delivered,
                "evidence_closed_flag": int(delivered and str(event.get("residual_product_type")) == "roi_context"),
            }
        )
    return rows


def _summarize_run(run: dict[str, Any], base: dict[str, Any] | None) -> dict[str, object]:
    row = _summarize_validation_run(run, _task_action_change(run, base))
    events = _evaluated_events(run)
    blockers = _event_counts(events)
    records = run["records"]
    enqueued = sum(_to_int(event.get("proactive_enqueue_flag")) for event in events)
    delivered = sum(_to_int(event.get("proactive_delivery_flag")) for event in events)
    queue_next_values = [_to_float(record.get("queue_next_mb")) for record in records]
    residual_wait = [_to_float(record.get("residual_queue_wait_s")) for record in records if _to_float(record.get("residual_queue_wait_s")) > 0.0]
    row.update(
        {
            "variant": run["variant"],
            "arm": run["variant"],
            "followup_eligibility_evaluated_count": len(events),
            "followup_eligibility_passed_count": sum(_to_int(event.get("proactive_flag")) for event in events),
            "followup_eligibility_blocked_count": sum(int(not _to_int(event.get("proactive_flag"))) for event in events),
            "proactive_flag_count": sum(_to_int(event.get("proactive_flag")) for event in events),
            "eriv_below_threshold_count": blockers["eriv_below_threshold"],
            "no_useful_remaining_deadline_count": blockers["no_useful_remaining_deadline"],
            "no_contact_opportunity_count": blockers["no_contact_opportunity"],
            "queue_pressure_block_count": blockers["queue_pressure_too_high"],
            "battery_unsafe_block_count": blockers["battery_unsafe"],
            "thermal_unsafe_block_count": blockers["thermal_unsafe"],
            "residual_payload_generated_count": sum(_to_int(event.get("proactive_flag")) for event in events),
            "residual_payload_enqueued_count": enqueued,
            "residual_payload_delivered_count": delivered,
            "residual_payload_expired_count": sum(_to_int(record.get("proactive_followup_expired_count")) for record in records),
            "residual_payload_dropped_count": sum(_to_int(record.get("proactive_followup_dropped_count")) for record in records),
            "residual_delivery_rate": _safe_div(delivered, enqueued),
            "queue_next_mb_mean": mean(queue_next_values) if queue_next_values else 0.0,
            "queue_next_mb_max": max(queue_next_values) if queue_next_values else 0.0,
            "residual_queue_wait_s_mean": mean(residual_wait) if residual_wait else 0.0,
            "residual_queue_wait_s_max": max(residual_wait) if residual_wait else 0.0,
        }
    )
    return row


def _paired(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(int(row["seed"]), str(row["variant"])): row for row in per_seed}
    seeds = sorted({int(row["seed"]) for row in per_seed})
    out: list[dict[str, object]] = []
    for variant in [str(row["variant"]) for row in VARIANTS if row["variant"] != BASELINE_VARIANT]:
        for metric in METRICS:
            deltas = [
                _to_float(by_key[(seed, variant)].get(metric)) - _to_float(by_key[(seed, BASELINE_VARIANT)].get(metric))
                for seed in seeds
                if (seed, variant) in by_key and (seed, BASELINE_VARIANT) in by_key
            ]
            low, high = _bootstrap_ci(deltas)
            out.append(
                {
                    "comparison": f"{variant} - {BASELINE_VARIANT}",
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


def _summary_by_variant(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        grouped[str(row["variant"])].append(row)
    for cfg in VARIANTS:
        variant = str(cfg["variant"])
        values = grouped.get(variant, [])
        item = {"variant": variant, "eriv_threshold": cfg["eriv_threshold"], "min_remaining_deadline_s": cfg["min_remaining_deadline_s"], "n": len(values)}
        for metric in CORE_METRICS:
            nums = [_to_float(row.get(metric)) for row in values]
            item[f"{metric}_mean"] = mean(nums) if nums else 0.0
            item[f"{metric}_total"] = sum(nums) if nums else 0.0
        rows.append(item)
    return rows


def _blocker_summary(trace_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in trace_rows:
        variant = str(row["variant"])
        reasons = _split_reasons(row.get("blocker_reason")) or (["eligible"] if _to_int(row.get("proactive_flag")) else ["no_blocker_logged"])
        for reason in reasons:
            grouped[variant][reason] += 1
            key = (variant, reason)
            if len(examples[key]) < 8:
                examples[key].append(str(row.get("task_id", "")))
    out: list[dict[str, object]] = []
    for cfg in VARIANTS:
        variant = str(cfg["variant"])
        total = max(sum(grouped[variant].values()), 1)
        for reason, count in grouped[variant].most_common():
            out.append(
                {
                    "variant": variant,
                    "blocker_reason": reason,
                    "count": count,
                    "percentage": _safe_div(count, total),
                    "example_task_ids": ";".join(examples[(variant, reason)]),
                }
            )
    return out


def _eriv_distribution(trace_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for cfg in VARIANTS:
        variant = str(cfg["variant"])
        values = [_to_float(row.get("ERIV")) for row in trace_rows if row.get("variant") == variant]
        thresholds = [_to_float(row.get("ERIV_threshold")) for row in trace_rows if row.get("variant") == variant]
        out.append(
            {
                "variant": variant,
                "n": len(values),
                "threshold": thresholds[0] if thresholds else cfg["eriv_threshold"],
                "mean": mean(values) if values else 0.0,
                "median": median(values) if values else 0.0,
                "p90": _percentile(values, 0.90),
                "max": max(values) if values else 0.0,
                "above_threshold_count": sum(1 for row in trace_rows if row.get("variant") == variant and _to_int(row.get("proactive_flag"))),
            }
        )
    return out


def _deadline_summary(trace_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for cfg in VARIANTS:
        variant = str(cfg["variant"])
        rows = [row for row in trace_rows if row.get("variant") == variant]
        deadlines = [_to_float(row.get("remaining_deadline_s")) for row in rows]
        out.append(
            {
                "variant": variant,
                "min_remaining_deadline_s": cfg["min_remaining_deadline_s"],
                "n": len(rows),
                "remaining_deadline_mean": mean(deadlines) if deadlines else 0.0,
                "remaining_deadline_min": min(deadlines) if deadlines else 0.0,
                "remaining_deadline_p50": median(deadlines) if deadlines else 0.0,
                "no_useful_remaining_deadline_count": sum(1 for row in rows if "no_useful_remaining_deadline" in str(row.get("blocker_reason"))),
                "no_contact_opportunity_count": sum(1 for row in rows if "no_contact_opportunity" in str(row.get("blocker_reason"))),
                "in_contact_count": sum(_to_int(row.get("in_contact")) for row in rows),
            }
        )
    return out


def _first_variant(rows: list[dict[str, object]], metric: str) -> str:
    for row in rows:
        if _to_float(row.get(f"{metric}_total")) > 0.0:
            return str(row["variant"])
    return "none"


def _variant_metric(rows: list[dict[str, object]], variant: str, metric: str) -> float:
    for row in rows:
        if row.get("variant") == variant:
            return _to_float(row.get(f"{metric}_mean"))
    return 0.0


def _utility_drop_ok(net: float, baseline_net: float) -> bool:
    return (baseline_net - net) <= 0.05 * max(abs(baseline_net), 1.0)


def _write_reports(
    command: str,
    seeds: list[int],
    gate_summary: list[dict[str, object]],
    per_seed: list[dict[str, object]],
    paired: list[dict[str, object]],
    trace_rows: list[dict[str, object]],
) -> None:
    first_proactive = _first_variant(gate_summary, "proactive_flag_count")
    first_generated = _first_variant(gate_summary, "residual_payload_generated_count")
    first_enqueued = _first_variant(gate_summary, "residual_payload_enqueued_count")
    first_delivered = _first_variant(gate_summary, "residual_payload_delivered_count")
    first_evidence = _first_variant(gate_summary, "evidence_closure_count")
    baseline_net = _variant_metric(gate_summary, BASELINE_VARIANT, "net_system_utility")
    candidates: list[dict[str, object]] = []
    for row in gate_summary:
        if row["variant"] == BASELINE_VARIANT:
            continue
        net = _to_float(row.get("net_system_utility_mean"))
        safety_ok = _to_float(row.get("hard_safety_violation_count_mean")) <= _variant_metric(gate_summary, BASELINE_VARIANT, "hard_safety_violation_count")
        utility_ok = _utility_drop_ok(net, baseline_net)
        delivery_gain = _to_float(row.get("residual_payload_delivered_count_total")) > 0.0 or _to_float(row.get("evidence_closure_count_total")) > 0.0
        if utility_ok and safety_ok and delivery_gain:
            candidates.append(row)
    recommendation_variant = str(candidates[0]["variant"]) if candidates else first_generated
    generated_total = max(_to_float(row.get("residual_payload_generated_count_total")) for row in gate_summary)
    enqueued_total = max(_to_float(row.get("residual_payload_enqueued_count_total")) for row in gate_summary)
    delivered_total = max(_to_float(row.get("residual_payload_delivered_count_total")) for row in gate_summary)
    evidence_total = max(_to_float(row.get("evidence_closure_count_total")) for row in gate_summary)
    if first_proactive == "none":
        case = "Case H"
        recommendation = "Gate relaxation did not make proactive_flag nonzero; audit ERIV scale or eligibility logic next."
    elif generated_total > 0.0 and enqueued_total <= 0.0:
        case = "Case C"
        recommendation = "Gate relaxation generates residual payloads, but enqueue remains zero; next audit duplicate/queue/enqueue guards."
    elif enqueued_total > 0.0 and delivered_total <= 0.0:
        case = "Case D"
        recommendation = "Residual enqueue works, delivery remains blocked; next audit downlink residual delivery."
    elif delivered_total > 0.0 and evidence_total <= 0.0:
        case = "Case E"
        recommendation = "Residual delivery works, closure remains zero; audit information sufficiency/closure accounting."
    elif evidence_total > 0.0:
        case = "Case F"
        recommendation = f"Keep `{recommendation_variant}` as the smallest effective default-off gate branch; do not adopt as main default."
    else:
        case = "Case A"
        recommendation = f"ERIV gate was too strict for generation. Keep `{recommendation_variant}` as a default-off gate-calibration result and continue to residual enqueue/delivery audit."
    if first_generated == "none":
        case = "Case B" if first_proactive != "none" else case
        recommendation = "proactive_flag became nonzero but residual generation stayed zero; inspect generate_residual_payload call chain."

    _write_text(
        OUT_DIR / "README.md",
        """# Phase 6 gate calibration debug

Default-off experiment-only sweep over existing Phase 6 proactive follow-up gate parameters. This does not edit reward, Module 2 scoring, SafeFilter, utility_rule, lifecycle closure, queue scheduling, or main default configs.
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
                "seeds": seeds,
                "base_arm": BASELINE_ARM,
                "variants": VARIANTS,
                "output_dir": OUT_DIR.as_posix(),
                "default_off_experiment_only": True,
                "payload_aware_deadline_gate_variant": "not_run_no_existing_config_support_detected",
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(
        OUT_DIR / "phase6_gate_calibration_report.md",
        f"""# Phase 6 gate calibration report

## Main findings

- First variant with proactive_flag_count > 0: {first_proactive}
- First variant with residual_payload_generated_count > 0: {first_generated}
- First variant with residual_payload_enqueued_count > 0: {first_enqueued}
- First variant with residual_payload_delivered_count > 0: {first_delivered}
- First variant with evidence_closure_count > 0: {first_evidence}
- Selected interpretation: {case}
- Recommended default-off gate variant: {recommendation_variant}

## Required answers

1. ERIV threshold as primary blocker: {'yes' if first_proactive != BASELINE_VARIANT and first_proactive != 'none' else 'partly/no'}.
2. min_remaining_deadline_s too hard: {'yes' if first_proactive not in {'none', 'baseline_gate', 'eriv_020_deadline_60', 'eriv_015_deadline_60'} else 'not primary in this sweep'}.
3. First nonzero proactive_flag variant: {first_proactive}.
4. First nonzero residual_payload_generated variant: {first_generated}.
5. First nonzero residual_payload_enqueued/delivered variant: enqueued={first_enqueued}, delivered={first_delivered}.
6. evidence_closure nonzero: {'yes' if first_evidence != 'none' else 'no'}.
7. If evidence closure remains zero, new first-zero stage: {'evidence_closure_count' if delivered_total > 0 and evidence_total <= 0 else 'residual_payload_enqueued_count' if generated_total > 0 and enqueued_total <= 0 else 'residual_payload_generated_count' if generated_total <= 0 else 'residual_payload_delivered_count'}.
8. canonical utility stable: {'yes' if all(_utility_drop_ok(_to_float(row.get('net_system_utility_mean')), baseline_net) for row in gate_summary) else 'no'}.
9. queue/expiry/safety worsened: {'yes' if any(_to_float(row.get('hard_safety_violation_count_total')) > 0 for row in gate_summary) else 'no hard safety increase observed'}.
10. recommended minimal default-off gate config: {recommendation_variant}.
11. Next default-off residual delivery experiment: {'yes' if enqueued_total > 0 and delivered_total <= 0 else 'not yet' if enqueued_total <= 0 else 'delivery already reached; audit closure if needed'}.

## Recommendation

{recommendation}
""",
    )
    _write_text(
        OUT_DIR / "final_recommendation.md",
        f"""# Final recommendation

{recommendation}

Do not recommend main default adoption. Keep any selected gate setting default-off and experiment-only.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    command = "python scripts/run_phase6_gate_calibration_debug.py --seeds " + ",".join(str(seed) for seed in seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_config_snapshots(seeds)

    run_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in seeds:
        for cfg in VARIANTS:
            variant = str(cfg["variant"])
            run_by_key[(seed, variant)] = _run_variant(seed, variant)

    per_seed: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    all_records: list[dict[str, Any]] = []
    all_queue_logs: list[dict[str, object]] = []
    for (seed, variant), run in sorted(run_by_key.items()):
        base = run_by_key.get((seed, BASELINE_VARIANT)) if variant != BASELINE_VARIANT else None
        per_seed.append(_summarize_run(run, base))
        trace_rows.extend(_trace_rows(run))
        all_records.extend(run["records"])
        all_queue_logs.extend(run["queue_logs"])

    paired = _paired(per_seed)
    gate_summary = _summary_by_variant(per_seed)
    blocker_rows = _blocker_summary(trace_rows)
    eriv_rows = _eriv_distribution(trace_rows)
    deadline_rows = _deadline_summary(trace_rows)

    _write_csv(OUT_DIR / "gate_sweep_summary.csv", gate_summary)
    _write_csv(OUT_DIR / "per_seed_gate_metrics.csv", per_seed)
    _write_csv(OUT_DIR / "per_candidate_followup_trace.csv", trace_rows)
    _write_csv(OUT_DIR / "followup_eligibility_blocker_summary.csv", blocker_rows)
    _write_csv(OUT_DIR / "eriv_distribution_summary.csv", eriv_rows)
    _write_csv(OUT_DIR / "deadline_feasibility_summary.csv", deadline_rows)
    _write_csv(OUT_DIR / "residual_payload_generation_summary.csv", _aggregate(per_seed, ["residual_payload_generated_count", "proactive_flag_count", "followup_eligibility_passed_count", "followup_eligibility_blocked_count"]))
    _write_csv(OUT_DIR / "residual_queue_delivery_summary.csv", _aggregate(per_seed, ["residual_payload_enqueued_count", "residual_payload_delivered_count", "residual_payload_expired_count", "residual_payload_dropped_count", "residual_delivery_rate", "queue_next_mb_mean", "queue_next_mb_max", "residual_queue_wait_s_mean", "residual_queue_wait_s_max"]))
    _write_csv(OUT_DIR / "lifecycle_shadow_summary.csv", _aggregate(per_seed, ["alert_rate", "confirmation_rate", "evidence_rate", "first_actionable_latency_p50", "first_actionable_latency_p90", "lifecycle_shadow_utility_balanced", "lifecycle_shadow_utility_alert_heavy", "lifecycle_shadow_utility_evidence_heavy", "evidence_closure_count", "evidence_closure_rate"]))
    _write_csv(OUT_DIR / "staged_closure_summary.csv", _aggregate(per_seed, ["staged_plan_rate", "staged_metadata_success_rate", "staged_followup_success_rate", "metadata_to_roi_success_rate", "metadata_to_roi_context_success_rate", "residual_enqueue_count", "residual_delivery_count", "residual_delivery_rate"]))
    _write_csv(OUT_DIR / "canonical_metric_summary.csv", _aggregate(per_seed, ["net_system_utility", "delivered_utility", "effective_utility", "total_penalized_reward", "mission_utility", "completed_tasks", "delivered_items", "expired_items", "deadline_satisfaction_ratio", "utility_per_joule", "utility_per_downlink_bit"]))
    _write_csv(OUT_DIR / "safety_guardrail_summary.csv", _aggregate(per_seed, ["hard_safety_violation_count", "battery_min", "thermal_max", "queue_overflow_count", "deadline_miss_count", "expired_items", "queue_next_mb_mean", "queue_next_mb_max"]))
    _write_csv(OUT_DIR / "paired_delta_summary.csv", paired)
    _write_reports(command, seeds, gate_summary, per_seed, paired, trace_rows)
    _write_text(OUT_DIR / "git_status.txt", _run_git(["status", "--short"]))
    _write_text(OUT_DIR / "git_diff.patch", _audit_diff(["scripts/run_phase6_gate_calibration_debug.py"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
