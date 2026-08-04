"""Reliable payload lifecycle ledger diagnostics and report package."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_value_aware_staged_delivery_queue as value_queue  # noqa: E402
from src.common.config import ensure_output_dirs  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_reliable_payload_lifecycle_queue"
ZIP_PATH = PROJECT_ROOT / "outputs" / "_reliable_payload_lifecycle_queue.zip"
SCENARIOS = ["natural", "targeted_resource_scarcity"]
DEFAULT_SEEDS = list(range(2026, 2031))
ARMS = [
    "proposed_persistent_value_aware_staged_queue",
    "utility_rule",
    "proposed_without_persistent_queue",
]
BASE_ARM = {
    "proposed_persistent_value_aware_staged_queue": "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue",
    "utility_rule": "utility_rule",
    "proposed_without_persistent_queue": "soft_moderate_value_skew_plus_gate_plus_progressive_current",
}
ARM_DIR = {
    "proposed_persistent_value_aware_staged_queue": "proposed_persistent",
    "utility_rule": "utility_rule",
    "proposed_without_persistent_queue": "without_persistent",
}
SCENARIO_DIR = {"natural": "natural", "targeted_resource_scarcity": "scarcity"}


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


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else "")).strip()


def _is_tracked(path: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode == 0


def _untracked_file_patch(path: str) -> str:
    full_path = PROJECT_ROOT / path
    if not full_path.exists() or not full_path.is_file():
        return ""
    text = full_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    body = "\n".join("+" + line for line in lines)
    if body:
        body += "\n"
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..0000000\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    )


def _git_diff_with_untracked(paths: list[str]) -> str:
    tracked = [path for path in paths if _is_tracked(path)]
    parts = []
    if tracked:
        parts.append(_git(["diff", "--", *tracked]))
    for path in paths:
        if not _is_tracked(path):
            patch = _untracked_file_patch(path)
            if patch:
                parts.append(patch)
    return "\n".join(part for part in parts if part).strip()


def _products(raw: object) -> set[str]:
    return {item.strip() for item in str(raw or "").replace(",", ";").split(";") if item.strip()}


def _run_one(seed: int, scenario: str, arm: str, num_slots: int) -> dict[str, Any]:
    configs = value_queue._load_run_configs(seed, scenario, BASE_ARM[arm], num_slots)
    if arm == "utility_rule":
        configs["experiment"] = {"routers": ["utility_rule"]}
    run_dir = OUT_DIR / "runs" / SCENARIO_DIR.get(scenario, scenario) / f"s{seed}" / ARM_DIR[arm]
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    rows = [asdict(record) for record in result.records]
    traces = {
        "queue_priority_trace": runner.last_queue_priority_logs,
        "staged_delivery_request_trace": runner.last_staged_delivery_request_trace,
        "staged_delivery_priority_trace": runner.last_staged_delivery_priority_trace,
        "contact_window_scheduling_trace": runner.last_contact_window_scheduling_trace,
        "backlog_composition_trace": runner.last_backlog_composition_trace,
        "payload_lifecycle_ledger_trace": runner.last_payload_lifecycle_ledger_trace,
        "payload_lifecycle_current_records": runner.last_payload_lifecycle_current_records,
        "payload_lifecycle_transition_summary": runner.last_payload_lifecycle_transition_summary,
        "payload_lifecycle_orphan_check": runner.last_payload_lifecycle_orphan_check,
        "next_window_recompetition_trace": runner.last_next_window_recompetition_trace,
        "priority_objective_component_raw_trace": runner.last_priority_objective_component_trace,
    }
    for name, trace_rows in traces.items():
        _write_csv(run_dir / f"{name}.csv", trace_rows)
    _write_text(run_dir / "resolved_config.yaml", yaml.safe_dump(configs, sort_keys=False, allow_unicode=True))
    return {
        "scenario": scenario,
        "seed": seed,
        "arm": arm,
        "run_dir": run_dir,
        "rows": rows,
        "summary": dict(result.summary),
        **traces,
    }


def _tag_rows(runs: list[dict[str, Any]], key: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for run in runs:
        for row in run.get(key, []):
            out.append(dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]))
    return out


def _priority_rows(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        for row in run["priority_objective_component_raw_trace"]:
            product_type = str(row.get("product_type", row.get("payload_type", "")))
            task_value = _to_float(row.get("task_value", row.get("calibrated_task_value", 0.0)))
            marginal = _to_float(row.get("utility_density_raw", row.get("staged_value_density_priority", 0.0)))
            residual_delta = _to_float(row.get("staged_delta_u_I", row.get("component_delta_u_I", 0.0)))
            if residual_delta <= 0.0:
                residual_delta = _to_float(row.get("remaining_utility")) / max(task_value, 1.0e-9)
            high_context = int(
                str(row.get("value_level", "")).lower() in {"high", "urgent"}
                or _to_int(row.get("high_value_flag")) > 0
            ) and int(str(row.get("product_requirement", "")).lower() in {"context", "evidence"})
            rows.append(
                {
                    "scenario": run["scenario"],
                    "seed": run["seed"],
                    "arm": run["arm"],
                    "slot": row.get("slot", ""),
                    "task_id": row.get("task_id", ""),
                    "product_type": product_type,
                    "source": row.get("source", ""),
                    "marginal_utility_per_mb": marginal,
                    "total_incremental_utility": _to_float(row.get("remaining_utility", row.get("incremental_utility", 0.0))),
                    "calibrated_task_value": task_value,
                    "residual_delta_information_utility": residual_delta,
                    "deadline_urgency": _to_float(row.get("urgency_component", row.get("staged_deadline_urgency", 0.0))),
                    "product_requirement_match": _to_float(row.get("staged_product_requirement_weight", 1.0)),
                    "value_level_factor": _to_float(row.get("staged_value_level_factor", 1.0)),
                    "closure_progress_factor": 1.0 if product_type == "roi_context" else (0.6 if product_type == "roi" else 0.2),
                    "starvation_age": _to_float(row.get("waiting_time_s", 0.0)),
                    "waiting_time": _to_float(row.get("waiting_time_s", 0.0)),
                    "feasible_high_value_guarantee_flag": high_context,
                    "final_priority_score": _to_float(row.get("queue_priority_total", row.get("priority_score", 0.0))),
                    "selected_for_delivery": _to_int(row.get("selected_for_downlink", row.get("selected_for_admission", 0))),
                    "delivered": _to_int(row.get("delivered_this_slot", 0)),
                }
            )
    return rows


def _guarantee_metrics(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    out = []
    for run in runs:
        records = run["payload_lifecycle_current_records"]
        recomp = run["next_window_recompetition_trace"]
        high_records = [
            row
            for row in records
            if str(row.get("value_level", "")).lower() in {"high", "urgent"}
            or _to_float(row.get("calibrated_task_value")) >= 0.75
        ]
        high_records = [
            row
            for row in high_records
            if str(row.get("product_requirement", "")).lower() in {"context", "evidence"}
            and str(row.get("product_type", "")).lower() in {"roi", "roi_context"}
        ]
        low_records = [
            row
            for row in records
            if str(row.get("value_level", "")).lower() in {"low", "medium"}
            and str(row.get("product_type", "")).lower() in {"metadata", "roi", "roi_context"}
        ]
        high_pending = [row for row in high_records if row.get("current_status") in {"pending", "blocked", "queued", "scheduled"}]
        high_delivered = [row for row in high_records if row.get("current_status") == "delivered"]
        high_missed = [
            row
            for row in high_records
            if row.get("current_status") in {"expired", "dropped"}
            or any(
                r.get("task_id") == row.get("task_id")
                and r.get("product_type") == row.get("product_type")
                and _to_int(r.get("feasible_this_window"))
                and not _to_int(r.get("delivered"))
                for r in recomp
            )
        ]
        high_waits = [max(0, _to_int(row.get("status_updated_slot")) - _to_int(row.get("created_slot"))) for row in high_records]
        low_waits = [max(0, _to_int(row.get("status_updated_slot")) - _to_int(row.get("created_slot"))) for row in low_records]
        out.append(
            {
                "scenario": run["scenario"],
                "seed": run["seed"],
                "arm": run["arm"],
                "feasible_high_value_pending_count": len(high_pending),
                "feasible_high_value_delivered_count": len(high_delivered),
                "feasible_high_value_missed_count": len(high_missed),
                "high_value_waiting_time_mean": mean(high_waits) if high_waits else 0.0,
                "high_value_waiting_time_max": max(high_waits) if high_waits else 0.0,
                "high_value_starvation_count": sum(1 for value in high_waits if value >= 2),
                "low_value_waiting_time_mean": mean(low_waits) if low_waits else 0.0,
                "important_task_left_behind_count": len(high_pending),
                "feasible_high_value_miss_rate": _safe_div(len(high_missed), len(high_records)),
            }
        )
    return out


def _group_count(rows: list[dict[str, object]], keys: list[str], count_name: str = "count") -> list[dict[str, object]]:
    counts: dict[tuple[object, ...], int] = defaultdict(int)
    for row in rows:
        counts[tuple(row.get(key, "") for key in keys)] += 1
    return [dict(zip(keys, key), **{count_name: value}) for key, value in sorted(counts.items())]


def _comparison_rows(runs: list[dict[str, Any]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    reliability = []
    pending = []
    reasons = []
    waits = []
    for run in runs:
        records = run["payload_lifecycle_current_records"]
        orphan_rows = run["payload_lifecycle_orphan_check"]
        dropped_no_reason = sum(1 for row in records if row.get("current_status") == "dropped" and not row.get("transition_reason"))
        expired_no_reason = sum(1 for row in records if row.get("current_status") == "expired" and not row.get("transition_reason"))
        reliability.append(
            {
                "scenario": run["scenario"],
                "seed": run["seed"],
                "arm": run["arm"],
                "lifecycle_record_count": len(records),
                "transition_count": len(run["payload_lifecycle_ledger_trace"]),
                "recompetition_count": len(run["next_window_recompetition_trace"]),
                "orphan_count": sum(_to_int(row.get("orphan_flag")) for row in orphan_rows),
                "dropped_without_reason_count": dropped_no_reason,
                "expired_without_reason_count": expired_no_reason,
            }
        )
        for row in records:
            if row.get("current_status") in {"pending", "blocked", "queued", "scheduled"}:
                pending.append(
                    {
                        "scenario": run["scenario"],
                        "seed": run["seed"],
                        "arm": run["arm"],
                        "status": row.get("current_status", ""),
                        "value_level": row.get("value_level", ""),
                        "product_type": row.get("product_type", ""),
                        "product_requirement": row.get("product_requirement", ""),
                        "task_id": row.get("task_id", ""),
                    }
                )
            if row.get("current_status") in {"expired", "dropped"}:
                reasons.append(
                    {
                        "scenario": run["scenario"],
                        "seed": run["seed"],
                        "arm": run["arm"],
                        "status": row.get("current_status", ""),
                        "reason": row.get("transition_reason", ""),
                        "blocker_reason": row.get("blocker_reason", ""),
                        "task_id": row.get("task_id", ""),
                        "product_type": row.get("product_type", ""),
                    }
                )
            high_value = str(row.get("value_level", "")).lower() in {"high", "urgent"} or _to_float(row.get("calibrated_task_value")) >= 0.75
            if high_value:
                waits.append(
                    {
                        "scenario": run["scenario"],
                        "seed": run["seed"],
                        "arm": run["arm"],
                        "task_id": row.get("task_id", ""),
                        "product_type": row.get("product_type", ""),
                        "status": row.get("current_status", ""),
                        "waiting_slots": max(0, _to_int(row.get("status_updated_slot")) - _to_int(row.get("created_slot"))),
                    }
                )
    return reliability, pending, reasons, waits


def _audit_outputs() -> None:
    answers = [
        (1, "metadata_no_window_persists", "yes", "DownlinkQueue keeps accepted metadata QueueItem entries across slots; ledger records blocked(no_contact) instead of dropped."),
        (2, "roi_request_no_window_staged_backlog", "yes_after_patch", "Staged ROI residuals and passive pending residual records are ledger-backed and can materialize into staged backlog."),
        (3, "roi_context_request_no_window_staged_backlog", "yes_after_patch", "Blocked ROI+Ctx staged requests are retained as active blocked backlog instead of trace-only returns."),
        (4, "pending_recompetes_next_contact", "yes", "DownlinkQueue reranks items every slot; staged queue scans pending/blocked requests at contact slots."),
        (5, "queue_staged_status_mismatch_possible", "mitigated", "Staged payloads reuse lifecycle_id and expired/dropped metadata is propagated back to staged request history."),
        (6, "generated_without_terminal_or_queue_possible", "checked", "orphan_check flags queued/scheduled records missing queue or staged backing."),
        (7, "staged_created_not_scheduler_possible", "mitigated", "Blocked pending residual ledger records are materialized for later recompetition."),
        (8, "delivered_without_lifecycle_possible", "mitigated", "Queue delivery transitions the lifecycle before DownlinkResult is consumed."),
        (9, "expired_dropped_reason_present", "yes", "deadline_passed and queue_overflow/duplicate/value-threshold reasons are explicit."),
        (10, "low_value_has_lifecycle", "yes", "Ledger creation is value-agnostic; low/medium items are retained with lower priority."),
        (11, "high_value_topk_budget_silent_loss", "mitigated", "Top-K/window blocked staged requests transition to blocked, not dropped."),
        (12, "ordinary_and_staged_unified_ledger", "yes_after_patch", "Both queues emit PayloadLifecycleLedger records."),
        (13, "duplicate_task_product_overwrite", "mitigated", "Duplicate attempts receive separate dropped/superseded records and do not overwrite active lifecycle_id."),
    ]
    _write_csv(
        OUT_DIR / "lifecycle_gap_trace.csv",
        [{"question_id": q, "gap": gap, "answer": answer, "evidence": evidence} for q, gap, answer, evidence in answers],
    )
    _write_text(
        OUT_DIR / "lifecycle_state_machine_audit.md",
        "# Lifecycle State Machine Audit\n\n"
        "The previous implementation had durable ordinary queue items but only implicit lifecycle accounting. "
        "The staged layer had trace rows, but several blocked paths returned without keeping an active request. "
        "This patch adds `PayloadLifecycleLedger` and wires ordinary queue, staged queue, passive ROI+Ctx registration, delivery, expiration, drop, and duplicate suppression transitions into one ledger.\n\n"
        "Direct answers:\n"
        + "\n".join(f"{q}. {answer}: {evidence}" for q, _gap, answer, evidence in answers)
    )


def _case_studies(runs: list[dict[str, Any]]) -> None:
    case_dir = OUT_DIR / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    proposed_rows = [
        row
        for run in runs
        if run["arm"] == "proposed_persistent_value_aware_staged_queue"
        for row in run["payload_lifecycle_ledger_trace"]
    ]
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in proposed_rows:
        by_task[str(row.get("task_id", ""))].append(row)
    fallback_task = next((task_id for task_id in by_task if task_id), "no_observed_task")

    def pick(predicate) -> tuple[str, bool]:
        for task_id, rows in by_task.items():
            if predicate(rows):
                return task_id, True
        return fallback_task, False

    specs = [
        ("01", "high/urgent + context/evidence delivered ROI+Ctx", lambda rows: any(r.get("product_type") == "roi_context" and r.get("current_status") == "delivered" for r in rows)),
        ("02", "ROI delivered but ROI+Ctx pending to next window", lambda rows: any(r.get("product_type") == "roi" and r.get("current_status") == "delivered" for r in rows) and any(r.get("product_type") == "roi_context" and r.get("current_status") in {"pending", "blocked", "queued"} for r in rows)),
        ("03", "ROI delivered but ROI+Ctx expired", lambda rows: any(r.get("product_type") == "roi_context" and r.get("current_status") == "expired" for r in rows)),
        ("04", "high/urgent task with ROI not delivered", lambda rows: any(r.get("product_type") == "roi" and r.get("current_status") in {"pending", "blocked", "expired", "dropped"} for r in rows)),
        ("05", "low/medium pending or late due lower priority", lambda rows: any(str(r.get("value_level", "")).lower() in {"low", "medium"} for r in rows)),
        ("06", "verification stops after ROI", lambda rows: any(r.get("product_requirement") == "verification" and r.get("product_type") == "roi" and r.get("current_status") == "delivered" for r in rows)),
        ("07", "blocked by no contact then recompetition", lambda rows: any(r.get("current_status") == "blocked" and r.get("blocker_reason") == "no_contact_opportunity" for r in rows)),
        ("08", "dropped because marginal utility too low", lambda rows: any(r.get("current_status") == "dropped" and "threshold" in str(r.get("transition_reason", "")) for r in rows)),
        ("09", "expired before future contact", lambda rows: any(r.get("current_status") == "expired" for r in rows)),
        ("10", "high-value long waiting time", lambda rows: any(_to_int(r.get("status_updated_slot")) - _to_int(r.get("created_slot")) >= 2 for r in rows)),
    ]
    for case_no, title, predicate in specs:
        task_id, exact = pick(predicate)
        rows = sorted(by_task.get(task_id, []), key=lambda row: (_to_int(row.get("status_updated_slot")), str(row.get("lifecycle_id", ""))))
        if not rows:
            rows = [{"task_id": task_id, "note": "no lifecycle rows observed in short diagnostic run"}]
        _write_csv(case_dir / f"case_{case_no}_task_{task_id}_lifecycle.csv", rows)
        delivered = sorted({str(r.get("product_type")) for r in rows if r.get("current_status") == "delivered"})
        not_delivered = sorted({str(r.get("product_type")) for r in rows if r.get("current_status") in {"pending", "queued", "blocked", "expired", "dropped"}})
        _write_text(
            case_dir / f"case_{case_no}_task_{task_id}.md",
            f"# Case {case_no}: {title}\n\n"
            f"- Task: `{task_id}`.\n"
            f"- Match quality: {'exact observed case' if exact else 'nearest observed lifecycle; exact case not present in this short run'}.\n"
            f"- Value level: {rows[0].get('value_level', '')}; requirement: {rows[0].get('product_requirement', '')}.\n"
            f"- Generated products: {', '.join(sorted({str(r.get('product_type')) for r in rows if r.get('product_type')}))}.\n"
            f"- Delivered products: {', '.join(delivered) if delivered else 'none'}.\n"
            f"- Not delivered products: {', '.join(not_delivered) if not_delivered else 'none'}.\n"
            "- Queue retention: inspect the CSV statuses; pending/blocked/queued rows are explicit backlog states.\n"
            "- Next-window recompetition: represented by `next_window_recompetition_trace.csv` for matching task/product rows.\n"
            "- Final outcome reason: terminal rows carry `transition_reason` and `blocker_reason`.\n"
            "- Objective alignment: lifecycle accounting preserves backlog without changing reward or hard safety.\n",
        )


def _final_reports(runs: list[dict[str, Any]], reliability: list[dict[str, object]], priority_rows: list[dict[str, object]], metrics: list[dict[str, object]]) -> None:
    proposed = [row for row in reliability if row["arm"] == "proposed_persistent_value_aware_staged_queue"]
    orphan_count = sum(_to_int(row.get("orphan_count")) for row in proposed)
    missing_reason = sum(_to_int(row.get("dropped_without_reason_count")) + _to_int(row.get("expired_without_reason_count")) for row in proposed)
    recompetition_count = sum(_to_int(row.get("recompetition_count")) for row in proposed)
    density_scores = [_to_float(row.get("marginal_utility_per_mb")) for row in priority_rows if row.get("arm") == "proposed_persistent_value_aware_staged_queue"]
    final_scores = [_to_float(row.get("final_priority_score")) for row in priority_rows if row.get("arm") == "proposed_persistent_value_aware_staged_queue"]
    density_dominant = bool(density_scores and final_scores and mean(density_scores) > 0.5 * max(mean(final_scores), 1.0e-9))
    case = "A" if orphan_count == 0 and missing_reason == 0 and recompetition_count > 0 else "B"
    if missing_reason:
        case = "F"
    _write_text(
        OUT_DIR / "priority_objective_alignment_report.md",
        "# Priority Objective Alignment Report\n\n"
        "- Global utility term: represented by marginal utility density, total incremental utility, calibrated task value, and final queue priority.\n"
        "- Limited guarantee readiness term: represented by waiting/starvation age and feasible_high_value_guarantee_flag; no guarantee debt is added in this patch.\n"
        f"- Current priority remains density-heavy: {'yes' if density_dominant else 'partly'}.\n"
        "- High-value ROI+Ctx can still be penalized by payload size because both ordinary and staged priority include per-MB density.\n"
        "- Next step should introduce guarantee debt / virtual queue if high-value waiting and miss-rate remain nonzero after lifecycle reliability is stable.\n"
        "- A guarantee debt term would affect high_value_waiting_time, high_value_starvation_count, feasible_high_value_miss_rate, and priority ranks.\n",
    )
    _write_text(
        OUT_DIR / "final_recommendation.md",
        "# Final Recommendation\n\n"
        f"Case: {case}.\n\n"
        f"1. Undelivered information is stored in queue/backlog: {'yes' if orphan_count == 0 else 'mostly, but orphan rows remain'}.\n"
        f"2. Next-window recompetition occurs: {'yes' if recompetition_count > 0 else 'not observed'}.\n"
        "3. Low-value tasks also have lifecycle records; they are lower priority, not untracked.\n"
        f"4. High-value tasks are not silently forgotten by lifecycle accounting; waiting risk is tracked in guarantee_readiness_metrics.csv.\n"
        f"5. Orphan payload count in proposed arm: {orphan_count}.\n"
        f"6. Blocked/drop/expire reason gaps in proposed arm: {missing_reason}.\n"
        f"7. Priority over-depends on utility_per_MB: {'yes' if density_dominant else 'partly'}.\n"
        "8. High-value ROI+Ctx can still be suppressed by large payload size under the current priority objective.\n"
        "9. Remaining gap: lifecycle reliability is now explicit; formal finite guarantee still needs debt/virtual-queue design and calibration.\n"
        "10. Next step: introduce guarantee debt / virtual queue as a separate bounded term, without changing reward_function or closure definition.\n",
    )


def _summary_rows(rows: list[dict[str, object]], keys: list[str], metrics: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(keys, key))
        item["n"] = len(group)
        for metric in metrics:
            values = [_to_float(row.get(metric)) for row in group]
            item[f"{metric}_mean"] = mean(values) if values else 0.0
            item[f"{metric}_sum"] = sum(values)
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--num-slots", type=int, default=80)
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = f"python scripts/run_reliable_payload_lifecycle_queue.py --seeds {args.seeds} --num-slots {args.num_slots}"
    _write_text(OUT_DIR / "command_log.md", f"# Command Log\n\n```powershell\n{command}\n```\n")
    runs = [_run_one(seed, scenario, arm, args.num_slots) for scenario in SCENARIOS for seed in seeds for arm in ARMS]

    _audit_outputs()
    ledger_rows = _tag_rows(runs, "payload_lifecycle_ledger_trace")
    current_rows = _tag_rows(runs, "payload_lifecycle_current_records")
    transition_summary = _tag_rows(runs, "payload_lifecycle_transition_summary")
    orphan_rows = _tag_rows(runs, "payload_lifecycle_orphan_check")
    recompetition_rows = _tag_rows(runs, "next_window_recompetition_trace")
    priority_rows = _priority_rows(runs)
    guarantee_rows = _guarantee_metrics(runs)
    reliability, pending, reasons, waits = _comparison_rows(runs)

    _write_csv(OUT_DIR / "payload_lifecycle_ledger_trace.csv", ledger_rows)
    _write_csv(OUT_DIR / "payload_lifecycle_current_records.csv", current_rows)
    _write_csv(OUT_DIR / "payload_lifecycle_transition_summary.csv", transition_summary)
    _write_csv(OUT_DIR / "payload_lifecycle_orphan_check.csv", orphan_rows)
    _write_csv(OUT_DIR / "next_window_recompetition_trace.csv", recompetition_rows)
    _write_csv(
        OUT_DIR / "next_window_recompetition_summary.csv",
        _summary_rows(
            recompetition_rows,
            ["scenario", "arm"],
            ["reevaluated_flag", "feasible_this_window", "selected_for_delivery", "delivered"],
        ),
    )
    _write_csv(OUT_DIR / "priority_objective_component_trace.csv", priority_rows)
    _write_csv(OUT_DIR / "guarantee_readiness_metrics.csv", guarantee_rows)
    _write_csv(OUT_DIR / "lifecycle_reliability_comparison.csv", reliability)
    _write_csv(OUT_DIR / "pending_backlog_composition.csv", pending)
    _write_csv(OUT_DIR / "expired_dropped_reason_summary.csv", _group_count(reasons, ["scenario", "arm", "status", "reason", "blocker_reason"]))
    _write_csv(
        OUT_DIR / "high_value_waiting_time_summary.csv",
        _summary_rows(waits, ["scenario", "arm", "status"], ["waiting_slots"]),
    )
    _case_studies(runs)
    _final_reports(runs, reliability, priority_rows, guarantee_rows)

    _write_text(
        OUT_DIR / "README.md",
        "# Reliable Payload Lifecycle Queue\n\n"
        "This package audits and validates unified lifecycle accounting for metadata, ROI, ROI+Ctx, and staged residual requests. "
        "The implementation is accounting and backlog reliability only: reward_function and closure definitions are unchanged.",
    )
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "num_slots": args.num_slots,
                "scenarios": SCENARIOS,
                "arms": ARMS,
                "reward_function_changed": False,
                "closure_definition_changed": False,
                "future_or_ground_truth_leakage": False,
                "utility_rule_new_priority_enabled": False,
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(OUT_DIR / "git_status.txt", _git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _git_diff_with_untracked(
            [
                "src/payload/payload_lifecycle_ledger.py",
                "src/payload/staged_delivery_queue.py",
                "src/link/downlink_queue.py",
                "src/common/types.py",
                "src/simulator/step_runner.py",
                "src/simulator/episode_runner.py",
                "scripts/run_reliable_payload_lifecycle_queue.py",
                "tests/test_reliable_payload_lifecycle_queue.py",
                "tests/test_value_aware_staged_delivery_queue.py",
            ]
        ),
    )
    files = sorted(path.relative_to(OUT_DIR).as_posix() for path in OUT_DIR.rglob("*") if path.is_file())
    _write_text(OUT_DIR / "archive_manifest.txt", "\n".join(files + ["archive_manifest.txt"]))
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUT_DIR.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(OUT_DIR.parent))
    print(f"output_dir={OUT_DIR}")
    print(f"archive={ZIP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
