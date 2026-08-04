"""Conservative staged-closure root-cause audit.

This script is diagnostic-only. It reads existing trajectory/candidate logs and
writes a sidecar audit package without changing simulator behavior, reward,
policy scoring, SafeFilter, utility_rule, or default configs.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "_staged_closure_root_cause_audit"
PACKAGE_PATH = PROJECT_ROOT / "outputs" / "_staged_closure_root_cause_audit_package.zip"
LIFECYCLE_DIR = PROJECT_ROOT / "outputs" / "_lifecycle_time_shadow_eval"
MULTISEED_DIR = PROJECT_ROOT / "outputs" / "_utility_guided_exploration_weight_multiseed"
PROPOSED_ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
DEFAULT_SEEDS = list(range(2026, 2036))
FOCUS_VARIANTS = ["utility_rule", "ug_wx100", "ug_wx050", "ug_wx035"]
PAIRWISE_COMPARISONS = [
    ("ug_wx050", "utility_rule"),
    ("ug_wx050", "ug_wx100"),
    ("ug_wx100", "utility_rule"),
]
STATUS_TRUE = "TRUE"
STATUS_FALSE = "FALSE"
STATUS_UNKNOWN = "UNKNOWN"
EPSILON = 1.0e-9

TRACE_PATCH_FIELDS = [
    "task_id",
    "stage_id",
    "payload_id",
    "payload_stage: metadata / roi / roictx / residual",
    "selected_action",
    "generated_time",
    "enqueue_time",
    "delivery_time",
    "deadline_time",
    "delivered_before_deadline",
    "queue_wait_s",
    "contact_window_id",
    "payload_size_mb",
    "drop_or_expire_reason",
    "parent_task_id",
    "residual_followup_of_payload_id",
]


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    text = completed.stdout or ""
    if completed.stderr:
        text += completed.stderr
    return text


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_present(row: dict[str, Any], key: str) -> bool:
    return key in row and str(row.get(key, "")).strip() != ""


def _status_from_bool(value: bool | None) -> str:
    if value is True:
        return STATUS_TRUE
    if value is False:
        return STATUS_FALSE
    return STATUS_UNKNOWN


def _and_status(*values: str) -> str:
    if any(value == STATUS_FALSE for value in values):
        return STATUS_FALSE
    if all(value == STATUS_TRUE for value in values):
        return STATUS_TRUE
    return STATUS_UNKNOWN


def _or_status(*values: str) -> str:
    if any(value == STATUS_TRUE for value in values):
        return STATUS_TRUE
    if values and all(value == STATUS_FALSE for value in values):
        return STATUS_FALSE
    return STATUS_UNKNOWN


def _normalize_product(value: Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "": "",
        "none": "",
        "metadata_only": "metadata",
        "roi_ctx": "roi_context",
        "roictx": "roi_context",
        "context": "roi_context",
        "delay_store": "defer",
    }
    return aliases.get(label, label)


def _split_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.replace(",", ";").split(";") if item.strip()]


def _stage_credit(product: str) -> set[str]:
    product = _normalize_product(product)
    if product == "metadata":
        return {"alert"}
    if product in {"roi", "residual_roi", "verification"}:
        return {"alert", "confirmation"}
    if product in {"roi_context", "residual", "residual_roi_context", "context_evidence", "evidence"}:
        return {"alert", "confirmation", "evidence"}
    return set()


def _selected_stages(row: dict[str, Any]) -> set[str]:
    payload_plan = _normalize_product(row.get("selected_payload_plan", ""))
    product = _normalize_product(row.get("selected_product_type", ""))
    payload = _normalize_product(row.get("selected_payload", row.get("payload_policy", "")))
    candidates = {product, payload}
    if payload_plan == "metadata_then_roi":
        candidates.update({"metadata", "roi"})
    elif payload_plan == "metadata_then_roi_context":
        candidates.update({"metadata", "roi_context"})
    elif payload_plan:
        candidates.add(payload_plan)
    stages: set[str] = set()
    for candidate in candidates:
        stages.update(_stage_credit(candidate))
    return stages


def _selected_status(row: dict[str, Any], stage: str, missing: set[str]) -> str:
    if not any(key in row for key in ("selected_payload_plan", "selected_product_type", "selected_payload", "payload_policy")):
        missing.add("selected_payload_plan/selected_product_type/selected_payload")
        return STATUS_UNKNOWN
    return _status_from_bool(stage in _selected_stages(row))


def _primary_product(row: dict[str, Any]) -> str:
    product = _normalize_product(row.get("selected_product_type", ""))
    if product:
        return product
    return _normalize_product(row.get("selected_payload", row.get("payload_policy", "")))


def _primary_stage_status(row: dict[str, Any], stage: str, missing: set[str]) -> dict[str, str]:
    product = _primary_product(row)
    selected = _selected_status(row, stage, missing)
    product_can_satisfy = stage in _stage_credit(product)
    if selected == STATUS_FALSE or not product_can_satisfy:
        return {
            "generated": STATUS_FALSE if selected != STATUS_UNKNOWN else STATUS_UNKNOWN,
            "enqueued": STATUS_FALSE if selected != STATUS_UNKNOWN else STATUS_UNKNOWN,
            "delivered": STATUS_FALSE if selected != STATUS_UNKNOWN else STATUS_UNKNOWN,
            "before_deadline": STATUS_FALSE if selected != STATUS_UNKNOWN else STATUS_UNKNOWN,
        }

    if "payload_generation_time_s" in row:
        generated = STATUS_TRUE if _to_float(row.get("payload_generation_time_s"), -1.0) >= 0.0 else STATUS_FALSE
    else:
        missing.add("payload_generation_time_s")
        generated = STATUS_UNKNOWN

    delivery_status = str(row.get("delivery_status", "")).strip().lower()
    if "delivery_status" not in row:
        missing.add("delivery_status")
        enqueued = delivered = before = STATUS_UNKNOWN
    else:
        enqueued = STATUS_TRUE if delivery_status in {"queued", "delivered"} else STATUS_FALSE
        delivered = STATUS_TRUE if delivery_status == "delivered" else STATUS_FALSE
        if delivered == STATUS_TRUE:
            if "payload_delivery_time_s" in row and "deadline_time_s" in row:
                delivery_time = _to_float(row.get("payload_delivery_time_s"), -1.0)
                deadline_time = _to_float(row.get("deadline_time_s"), -1.0)
                before = STATUS_TRUE if delivery_time >= 0.0 and deadline_time >= 0.0 and delivery_time <= deadline_time + EPSILON else STATUS_FALSE
            else:
                for key in ("payload_delivery_time_s", "deadline_time_s"):
                    if key not in row:
                        missing.add(key)
                before = STATUS_UNKNOWN
        elif delivery_status in {"expired", "dropped", "deferred"} or _to_int(row.get("expired_flag")):
            before = STATUS_FALSE
        else:
            before = STATUS_UNKNOWN

    return {
        "generated": generated,
        "enqueued": enqueued,
        "delivered": delivered,
        "before_deadline": before,
    }


def _residual_stage_status(row: dict[str, Any], stage: str, missing: set[str]) -> dict[str, str]:
    residual = _normalize_product(row.get("residual_product_type", ""))
    plan = _normalize_product(row.get("selected_payload_plan", ""))
    if stage == "confirmation":
        relevant = residual in {"roi", "roi_context", "residual_roi", "residual_roi_context"} or plan in {
            "metadata_then_roi",
            "metadata_then_roi_context",
        }
    else:
        relevant = residual in {"roi_context", "residual", "residual_roi_context", "context_evidence", "evidence"} or plan == "metadata_then_roi_context"
    if not relevant:
        return {"generated": STATUS_FALSE, "enqueued": STATUS_FALSE, "delivered": STATUS_FALSE, "before_deadline": STATUS_FALSE}

    generated_fields = ["residual_utility_generated", "followup_payload_mb"]
    if not any(key in row for key in generated_fields):
        missing.add("residual_utility_generated/followup_payload_mb")
        generated = STATUS_UNKNOWN
    else:
        generated = STATUS_TRUE if any(_to_float(row.get(key), 0.0) > 0.0 for key in generated_fields if key in row) else STATUS_FALSE

    enqueue_fields = [
        "residual_enqueue_count",
        "proactive_followup_enqueue_count",
        "carryover_proactive_enqueue_count",
        "same_slot_carryover_enqueue_count",
    ]
    if not any(key in row for key in enqueue_fields):
        missing.add("residual_enqueue_count/proactive_followup_enqueue_count")
        enqueued = STATUS_UNKNOWN
    else:
        enqueued = STATUS_TRUE if any(_to_int(row.get(key)) > 0 for key in enqueue_fields if key in row) else STATUS_FALSE

    delivery_fields = [
        "residual_delivery_count",
        "proactive_followup_delivery_count",
        "carryover_proactive_delivery_count",
        "same_slot_carryover_delivery_count",
    ]
    if not any(key in row for key in delivery_fields):
        missing.add("residual_delivery_count/proactive_followup_delivery_count")
        delivered = STATUS_UNKNOWN
    else:
        delivered = STATUS_TRUE if any(_to_int(row.get(key)) > 0 for key in delivery_fields if key in row) else STATUS_FALSE

    if delivered == STATUS_TRUE:
        if "residual_utility_delivered" in row:
            before = STATUS_TRUE if _to_float(row.get("residual_utility_delivered"), 0.0) > 0.0 else STATUS_UNKNOWN
        else:
            missing.add("residual_utility_delivered")
            before = STATUS_UNKNOWN
    elif delivered == STATUS_FALSE:
        expire_or_drop = any(
            _to_int(row.get(key)) > 0
            for key in (
                "residual_expired_count",
                "residual_dropped_count",
                "proactive_followup_expired_count",
                "proactive_followup_dropped_count",
            )
            if key in row
        )
        before = STATUS_FALSE if expire_or_drop or enqueued == STATUS_FALSE else STATUS_UNKNOWN
    else:
        before = STATUS_UNKNOWN

    return {"generated": generated, "enqueued": enqueued, "delivered": delivered, "before_deadline": before}


def _stage_status(row: dict[str, Any], stage: str, missing: set[str]) -> dict[str, str]:
    selected = _selected_status(row, stage, missing)
    primary = _primary_stage_status(row, stage, missing)
    residual = (
        {"generated": STATUS_FALSE, "enqueued": STATUS_FALSE, "delivered": STATUS_FALSE, "before_deadline": STATUS_FALSE}
        if stage == "alert"
        else _residual_stage_status(row, stage, missing)
    )
    return {
        "selected": selected,
        "generated": _or_status(primary["generated"], residual["generated"]),
        "enqueued": _or_status(primary["enqueued"], residual["enqueued"]),
        "delivered": _or_status(primary["delivered"], residual["delivered"]),
        "before_deadline": _or_status(primary["before_deadline"], residual["before_deadline"]),
    }


def _trajectory_path(seed: int, variant: str) -> Path:
    if variant == "utility_rule":
        return LIFECYCLE_DIR / "runs" / f"seed_{seed}" / "utility_rule" / "trajectories" / "utility_rule.csv"
    return MULTISEED_DIR / "runs" / f"seed_{seed}" / variant / "trajectories" / f"{PROPOSED_ROUTER}.csv"


def _candidate_path(seed: int, variant: str) -> Path:
    return MULTISEED_DIR / "candidate_action_logs" / f"seed_{seed}_{variant}.csv"


def _valid_task_id(task_id: Any) -> bool:
    text = str(task_id or "").strip()
    return bool(text) and text.lower() not in {"none", "__idle__", "idle"}


def _event_map(rows: list[dict[str, Any]], ids_key: str, times_key: str) -> dict[str, float]:
    events: dict[str, float] = {}
    for row in rows:
        ids = _split_list(row.get(ids_key))
        times = _split_list(row.get(times_key))
        for index, task_id in enumerate(ids):
            if not _valid_task_id(task_id):
                continue
            event_time = _to_float(times[index], _to_float(row.get("t"), -1.0) * 20.0) if index < len(times) else _to_float(row.get("t"), -1.0) * 20.0
            if task_id not in events or event_time < events[task_id]:
                events[task_id] = event_time
    return events


def _failure_proven_by_events(row: dict[str, Any], expired_events: dict[str, float], dropped_events: dict[str, float]) -> bool:
    task_id = str(row.get("task_id", "")).strip()
    deadline = _to_float(row.get("deadline_time_s"), -1.0)
    if task_id in dropped_events:
        return True
    if task_id in expired_events:
        return deadline < 0.0 or expired_events[task_id] >= deadline - EPSILON
    return False


def _with_event_failures(stage: dict[str, str], row: dict[str, Any], expired_events: dict[str, float], dropped_events: dict[str, float]) -> dict[str, str]:
    if stage["delivered"] == STATUS_FALSE and stage["before_deadline"] == STATUS_UNKNOWN and _failure_proven_by_events(row, expired_events, dropped_events):
        stage = dict(stage)
        stage["before_deadline"] = STATUS_FALSE
    return stage


def _root_cause(row: dict[str, Any], alert: dict[str, str], confirmation: dict[str, str], evidence: dict[str, str], missing: set[str]) -> tuple[str, str]:
    if evidence["before_deadline"] == STATUS_TRUE:
        return "successfully_evidence_closed", "high_confidence"
    if evidence["delivered"] == STATUS_TRUE and evidence["before_deadline"] == STATUS_FALSE:
        return "evidence_delivered_after_deadline", "high_confidence"
    if evidence["enqueued"] == STATUS_TRUE and evidence["delivered"] == STATUS_FALSE:
        blockers = str(row.get("proactive_followup_blockers", "")).lower()
        if "contact" in blockers or _to_int(row.get("proactive_contact_block_count")) or _to_int(row.get("carryover_contact_block_count")):
            return "contact_capacity_insufficient", "medium_confidence"
        if _to_float(row.get("residual_queue_wait_s"), 0.0) > 0.0 or _to_float(row.get("average_waiting_time_s"), 0.0) > 0.0:
            return "queue_wait_too_long", "medium_confidence"
        return "evidence_enqueued_but_not_delivered", "medium_confidence"
    if evidence["generated"] == STATUS_TRUE and evidence["enqueued"] == STATUS_FALSE:
        return "evidence_generated_but_not_enqueued", "medium_confidence"
    if confirmation["before_deadline"] == STATUS_TRUE and evidence["generated"] == STATUS_FALSE:
        return "confirmation_delivered_but_no_evidence_generated", "medium_confidence"
    if confirmation["generated"] == STATUS_TRUE and confirmation["before_deadline"] == STATUS_FALSE:
        return "confirmation_generated_but_not_delivered", "medium_confidence"
    if alert["before_deadline"] == STATUS_TRUE and confirmation["generated"] == STATUS_FALSE:
        return "metadata_delivered_but_no_confirmation_generated", "medium_confidence"
    if alert["generated"] == STATUS_TRUE and alert["before_deadline"] == STATUS_FALSE:
        if _to_int(row.get("deferred_flag")):
            return "deferred_until_expiry", "medium_confidence"
        return "metadata_generated_but_not_delivered", "medium_confidence"
    if alert["selected"] == STATUS_FALSE and confirmation["selected"] == STATUS_FALSE and evidence["selected"] == STATUS_FALSE:
        return "no_staged_action_selected", "high_confidence"
    if any(value == STATUS_UNKNOWN for value in [*alert.values(), *confirmation.values(), *evidence.values()]) or missing:
        return "unknown_due_to_missing_logs", "unknown_due_to_missing_logs"
    return "unknown_due_to_missing_logs", "unknown_due_to_missing_logs"


def _closure_status(evidence: dict[str, str], missing: set[str]) -> tuple[str, str]:
    if evidence["before_deadline"] == STATUS_TRUE:
        return STATUS_TRUE, "high_confidence"
    if evidence["before_deadline"] == STATUS_FALSE and evidence["delivered"] in {STATUS_FALSE, STATUS_TRUE}:
        return STATUS_FALSE, "medium_confidence"
    if missing:
        return STATUS_UNKNOWN, "unknown_due_to_missing_logs"
    return STATUS_UNKNOWN, "low_confidence"


def _task_rows_for_run(seed: int, variant: str, path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _read_csv(path)
    header = list(rows[0].keys()) if rows else []
    expired_events = _event_map(rows, "expired_task_ids", "expired_task_event_times_s")
    dropped_events = _event_map(rows, "dropped_task_ids", "dropped_task_event_times_s")
    task_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not _valid_task_id(task_id) or task_id in seen:
            continue
        seen.add(task_id)
        missing: set[str] = set()
        alert = _with_event_failures(_stage_status(row, "alert", missing), row, expired_events, dropped_events)
        confirmation = _with_event_failures(_stage_status(row, "confirmation", missing), row, expired_events, dropped_events)
        evidence = _with_event_failures(_stage_status(row, "evidence", missing), row, expired_events, dropped_events)
        if (
            str(row.get("delivery_status", "")).strip().lower() == "queued"
            and task_id not in expired_events
            and task_id not in dropped_events
            and str(row.get("delivered_task_ids", "")).find(task_id) < 0
        ):
            missing.add("terminal_delivery_or_expiry_event")
        closure_status, closure_confidence = _closure_status(evidence, missing)
        label, confidence = _root_cause(row, alert, confirmation, evidence, missing)
        delivered_before_deadline = _and_status(evidence["delivered"], evidence["before_deadline"])
        if delivered_before_deadline == STATUS_TRUE:
            closure_status, closure_confidence = STATUS_TRUE, "high_confidence"
        elif evidence["delivered"] == STATUS_TRUE and evidence["before_deadline"] == STATUS_FALSE:
            closure_status, closure_confidence = STATUS_FALSE, "high_confidence"
        elif label == "unknown_due_to_missing_logs" or missing:
            closure_status, closure_confidence = STATUS_UNKNOWN, "unknown_due_to_missing_logs"

        task_rows.append(
            {
                "seed": seed,
                "variant": variant,
                "router_name": "utility_rule" if variant == "utility_rule" else PROPOSED_ROUTER,
                "task_id": task_id,
                "task_class": row.get("task_class", ""),
                "task_value": row.get("V_cal", row.get("V_t", row.get("task_value", ""))),
                "arrival_time_s": row.get("task_arrival_time_s", row.get("t", "")),
                "deadline_s": row.get("deadline_s", ""),
                "deadline_time_s": row.get("deadline_time_s", ""),
                "selected_action": row.get("raw_action", row.get("selected_action", "")),
                "selected_payload_plan": row.get("selected_payload_plan", ""),
                "selected_product_type": row.get("selected_product_type", ""),
                "selected_payload": row.get("selected_payload", row.get("payload_policy", "")),
                "delivery_status": row.get("delivery_status", ""),
                "payload_generation_time_s": row.get("payload_generation_time_s", ""),
                "payload_delivery_time_s": row.get("payload_delivery_time_s", ""),
                "expired_event_time_s": expired_events.get(task_id, ""),
                "dropped_event_time_s": dropped_events.get(task_id, ""),
                "delivered_products": row.get("delivered_products", ""),
                "pending_products": row.get("pending_products", ""),
                "residual_product_type": row.get("residual_product_type", ""),
                "residual_enqueue_count": row.get("residual_enqueue_count", ""),
                "residual_delivery_count": row.get("residual_delivery_count", ""),
                "proactive_followup_enqueue_count": row.get("proactive_followup_enqueue_count", ""),
                "proactive_followup_delivery_count": row.get("proactive_followup_delivery_count", ""),
                "residual_queue_wait_s": row.get("residual_queue_wait_s", ""),
                "queue_item_count": row.get("queue_item_count", ""),
                "average_waiting_time_s": row.get("average_waiting_time_s", ""),
                "alert_selected_status": alert["selected"],
                "alert_generated_status": alert["generated"],
                "alert_enqueued_status": alert["enqueued"],
                "alert_delivered_status": alert["delivered"],
                "alert_before_deadline_status": alert["before_deadline"],
                "confirmation_selected_status": confirmation["selected"],
                "confirmation_generated_status": confirmation["generated"],
                "confirmation_enqueued_status": confirmation["enqueued"],
                "confirmation_delivered_status": confirmation["delivered"],
                "confirmation_before_deadline_status": confirmation["before_deadline"],
                "evidence_selected_status": evidence["selected"],
                "evidence_generated_status": evidence["generated"],
                "evidence_enqueued_status": evidence["enqueued"],
                "evidence_delivered_status": evidence["delivered"],
                "evidence_before_deadline_status": evidence["before_deadline"],
                "evidence_closure_status": closure_status,
                "evidence_closure_confidence": closure_confidence,
                "missing_fields_for_this_task": ";".join(sorted(missing)),
                "root_cause_confidence": confidence,
                "root_cause_label": label,
            }
        )
    return task_rows, header


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summaries(per_task: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_run: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_task:
        by_run[(str(row["variant"]), int(row["seed"]))].append(row)

    stage_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    enqueue_rows: list[dict[str, Any]] = []
    deadline_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []

    for (variant, seed), rows in sorted(by_run.items()):
        total = len(rows)
        alert_true = sum(row["alert_before_deadline_status"] == STATUS_TRUE for row in rows)
        confirmation_true = sum(row["confirmation_before_deadline_status"] == STATUS_TRUE for row in rows)
        evidence_true = sum(row["evidence_before_deadline_status"] == STATUS_TRUE for row in rows)
        evidence_generated = sum(row["evidence_generated_status"] == STATUS_TRUE for row in rows)
        evidence_enqueued = sum(row["evidence_enqueued_status"] == STATUS_TRUE for row in rows)
        evidence_delivered = sum(row["evidence_delivered_status"] == STATUS_TRUE for row in rows)
        metadata_only_alert = sum(
            row["alert_before_deadline_status"] == STATUS_TRUE and row["confirmation_before_deadline_status"] != STATUS_TRUE
            for row in rows
        )
        unknown_stage_chain = sum(
            any(row[f"{stage}_{part}_status"] == STATUS_UNKNOWN for stage in ("alert", "confirmation", "evidence") for part in ("selected", "generated", "enqueued", "delivered", "before_deadline"))
            for row in rows
        )
        confirmed_closed = sum(row["evidence_closure_status"] == STATUS_TRUE for row in rows)
        confirmed_failed = sum(row["evidence_closure_status"] == STATUS_FALSE for row in rows)
        unknown_closed = sum(row["evidence_closure_status"] == STATUS_UNKNOWN for row in rows)
        label_counts = Counter(row["root_cause_label"] for row in rows)
        stage_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "task_count": total,
                "alert_achieved_count": alert_true,
                "alert_achieved_rate": _rate(alert_true, total),
                "confirmation_achieved_count": confirmation_true,
                "confirmation_achieved_rate": _rate(confirmation_true, total),
                "evidence_achieved_count": evidence_true,
                "evidence_achieved_rate": _rate(evidence_true, total),
                "evidence_closure_rate_conditional_on_alert": _rate(evidence_true, alert_true),
                "evidence_closure_rate_conditional_on_confirmation": _rate(evidence_true, confirmation_true),
                "metadata_only_alert_count": metadata_only_alert,
                "metadata_only_alert_rate": _rate(metadata_only_alert, total),
                "no_staged_action_selected_count": label_counts["no_staged_action_selected"],
                "no_staged_action_selected_rate": _rate(label_counts["no_staged_action_selected"], total),
                "evidence_generated_count": evidence_generated,
                "evidence_generated_rate": _rate(evidence_generated, total),
                "evidence_enqueued_count": evidence_enqueued,
                "evidence_enqueued_rate": _rate(evidence_enqueued, total),
                "evidence_delivered_count": evidence_delivered,
                "evidence_delivered_rate": _rate(evidence_delivered, total),
                "expired_after_metadata_alert_count": label_counts["metadata_generated_but_not_delivered"] + label_counts["deferred_until_expiry"],
                "expired_after_metadata_alert_rate": _rate(label_counts["metadata_generated_but_not_delivered"] + label_counts["deferred_until_expiry"], total),
                "queue_blocked_after_metadata_alert_count": label_counts["queue_wait_too_long"],
                "queue_blocked_after_metadata_alert_rate": _rate(label_counts["queue_wait_too_long"], total),
                "capacity_blocked_after_metadata_alert_count": label_counts["contact_capacity_insufficient"],
                "capacity_blocked_after_metadata_alert_rate": _rate(label_counts["contact_capacity_insufficient"], total),
                "confirmed_evidence_closed_count": confirmed_closed,
                "confirmed_evidence_failed_count": confirmed_failed,
                "unknown_evidence_closure_count": unknown_closed,
                "evidence_closure_unknown_rate": _rate(unknown_closed, total),
                "confirmed_metadata_only_alert_count": metadata_only_alert,
                "unknown_stage_chain_count": unknown_stage_chain,
                "unknown_stage_chain_rate": _rate(unknown_stage_chain, total),
                "root_cause_counts": ";".join(f"{key}:{value}" for key, value in sorted(label_counts.items())),
            }
        )
        payload_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "task_count": total,
                "alert_selected_true": sum(row["alert_selected_status"] == STATUS_TRUE for row in rows),
                "alert_generated_true": sum(row["alert_generated_status"] == STATUS_TRUE for row in rows),
                "confirmation_selected_true": sum(row["confirmation_selected_status"] == STATUS_TRUE for row in rows),
                "confirmation_generated_true": sum(row["confirmation_generated_status"] == STATUS_TRUE for row in rows),
                "evidence_selected_true": sum(row["evidence_selected_status"] == STATUS_TRUE for row in rows),
                "evidence_generated_true": evidence_generated,
                "evidence_generated_false": sum(row["evidence_generated_status"] == STATUS_FALSE for row in rows),
                "evidence_generated_unknown": sum(row["evidence_generated_status"] == STATUS_UNKNOWN for row in rows),
            }
        )
        enqueue_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "alert_enqueued_true": sum(row["alert_enqueued_status"] == STATUS_TRUE for row in rows),
                "alert_delivered_true": sum(row["alert_delivered_status"] == STATUS_TRUE for row in rows),
                "confirmation_enqueued_true": sum(row["confirmation_enqueued_status"] == STATUS_TRUE for row in rows),
                "confirmation_delivered_true": sum(row["confirmation_delivered_status"] == STATUS_TRUE for row in rows),
                "evidence_enqueued_true": evidence_enqueued,
                "evidence_delivered_true": evidence_delivered,
                "evidence_enqueued_not_delivered_count": sum(
                    row["evidence_enqueued_status"] == STATUS_TRUE and row["evidence_delivered_status"] == STATUS_FALSE for row in rows
                ),
            }
        )
        deadline_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "alert_before_deadline_true": alert_true,
                "confirmation_before_deadline_true": confirmation_true,
                "evidence_before_deadline_true": evidence_true,
                "evidence_delivered_after_deadline_count": label_counts["evidence_delivered_after_deadline"],
                "deferred_until_expiry_count": label_counts["deferred_until_expiry"],
                "known_expired_event_count": sum(str(row.get("expired_event_time_s", "")) != "" for row in rows),
            }
        )
        queue_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "queue_wait_too_long_count": label_counts["queue_wait_too_long"],
                "contact_capacity_insufficient_count": label_counts["contact_capacity_insufficient"],
                "evidence_generated_but_not_enqueued_count": label_counts["evidence_generated_but_not_enqueued"],
                "mean_queue_item_count_at_decision": mean([_to_float(row.get("queue_item_count"), 0.0) for row in rows]) if rows else 0.0,
                "mean_average_waiting_time_s": mean([_to_float(row.get("average_waiting_time_s"), 0.0) for row in rows]) if rows else 0.0,
                "mean_residual_queue_wait_s": mean([_to_float(row.get("residual_queue_wait_s"), 0.0) for row in rows]) if rows else 0.0,
            }
        )
    return stage_rows, payload_rows, enqueue_rows, deadline_rows, queue_rows


def _variant_means(stage_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = [
        "alert_achieved_rate",
        "confirmation_achieved_rate",
        "evidence_achieved_rate",
        "metadata_only_alert_rate",
        "evidence_closure_rate_conditional_on_alert",
        "expired_after_metadata_alert_rate",
        "queue_blocked_after_metadata_alert_rate",
        "capacity_blocked_after_metadata_alert_rate",
    ]
    out: dict[str, dict[str, float]] = {}
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        by_variant[str(row["variant"])].append(row)
    for variant, rows in by_variant.items():
        out[variant] = {metric: mean([_to_float(row.get(metric), 0.0) for row in rows]) for metric in metrics}
        out[variant]["defer_rate"] = 0.0
    return out


def _paired_delta_rows(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "alert_achieved_rate",
        "confirmation_achieved_rate",
        "evidence_achieved_rate",
        "metadata_only_alert_rate",
        "evidence_closure_rate_conditional_on_alert",
        "expired_after_metadata_alert_rate",
        "queue_blocked_after_metadata_alert_rate",
        "capacity_blocked_after_metadata_alert_rate",
    ]
    by_key = {(int(row["seed"]), str(row["variant"])): row for row in stage_rows}
    seeds = sorted({int(row["seed"]) for row in stage_rows})
    rows: list[dict[str, Any]] = []
    for left, right in PAIRWISE_COMPARISONS:
        paired = [seed for seed in seeds if (seed, left) in by_key and (seed, right) in by_key]
        for metric in metrics:
            deltas = [_to_float(by_key[(seed, left)].get(metric), 0.0) - _to_float(by_key[(seed, right)].get(metric), 0.0) for seed in paired]
            rows.append(
                {
                    "comparison": f"{left}-{right}",
                    "metric": metric,
                    "paired_seed_count": len(paired),
                    "mean_delta": mean(deltas) if deltas else 0.0,
                    "min_delta": min(deltas) if deltas else 0.0,
                    "max_delta": max(deltas) if deltas else 0.0,
                }
            )
    return rows


def _fields_audit(input_files: list[Path], headers: dict[str, list[str]]) -> str:
    requested = {
        "task_id": ["task_id"],
        "seed": ["seed", "path seed"],
        "arm / policy": ["variant", "router_name", "path variant"],
        "selected model": ["selected_model", "model_name"],
        "selected payload strategy": ["selected_payload_plan", "selected_payload", "payload_policy"],
        "generated payload type": ["selected_product_type", "residual_product_type"],
        "metadata generated flag": ["payload_generation_time_s", "selected_product_type"],
        "ROI generated flag": ["payload_generation_time_s", "selected_product_type", "residual_utility_generated"],
        "ROI+Ctx generated flag": ["payload_generation_time_s", "selected_product_type", "residual_utility_generated"],
        "residual generated flag": ["residual_utility_generated", "followup_payload_mb"],
        "payload size": ["S_gen_current_mb", "followup_payload_mb"],
        "enqueue time": [],
        "delivery time": ["payload_delivery_time_s", "delivered_task_event_times_s"],
        "contact window id": [],
        "queue wait time": ["average_waiting_time_s", "residual_queue_wait_s"],
        "deadline": ["deadline_s", "deadline_time_s"],
        "expired flag": ["expired_flag", "expired_task_ids", "expired_task_event_times_s"],
        "delivered flag": ["delivery_status", "delivered_task_ids"],
        "defer flag": ["deferred_flag", "task_deferred"],
        "task value": ["V_cal", "V_t", "task_value"],
        "lifecycle stage credit from sidecar metrics": ["alert_delivered_utility", "verification_delivered_utility", "context_evidence_delivered_utility"],
    }
    all_fields = set().union(*(set(header) for header in headers.values())) if headers else set()
    lines = [
        "# Available Fields Audit",
        "",
        "This audit distinguishes selected, generated, enqueued, and delivered. Missing fields are reported as UNKNOWN, not FALSE.",
        "",
        "## Input files",
        *[f"- {path.relative_to(PROJECT_ROOT)}" for path in input_files[:20]],
        f"- total input trajectory files: {len(input_files)}",
        "",
        "## Field sufficiency",
        "| concept | status | fields used | missing for definitive trace |",
        "| --- | --- | --- | --- |",
    ]
    for concept, fields in requested.items():
        present = [field for field in fields if field in all_fields or field.startswith("path ")]
        missing = [field for field in fields if field not in all_fields and not field.startswith("path ")]
        if fields and present and not missing:
            status = "available"
        elif present:
            status = "partially_available"
        else:
            status = "missing"
        if concept in {"enqueue time", "contact window id"}:
            status = "missing"
            missing = fields or [concept]
        lines.append(
            f"| {concept} | {status} | {', '.join(present) if present else 'none'} | {', '.join(missing) if missing else 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Conservative interpretation",
            "",
            "- Selected means the policy selected a payload/action type.",
            "- Generated means a payload artifact generation field directly indicates generation.",
            "- Enqueued means delivery_status or residual enqueue counters directly indicate queue insertion.",
            "- Delivered means delivery_status/delivery event fields directly indicate delivery, and before-deadline requires delivery_time <= deadline_time.",
            "- Direct ROI+Ctx can close evidence if delivered before deadline.",
            "- Metadata delivery alone does not prove residual generation or evidence closure.",
            "- Missing enqueue_time, contact_window_id, payload_id, and per-residual generated/delivery timestamps limit root-cause confidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _report_text(
    per_task: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> tuple[str, str, str]:
    total = len(per_task)
    closed = sum(row["evidence_closure_status"] == STATUS_TRUE for row in per_task)
    failed = sum(row["evidence_closure_status"] == STATUS_FALSE for row in per_task)
    unknown = sum(row["evidence_closure_status"] == STATUS_UNKNOWN for row in per_task)
    evidence_selected = sum(row["evidence_selected_status"] == STATUS_TRUE for row in per_task)
    evidence_generated = sum(row["evidence_generated_status"] == STATUS_TRUE for row in per_task)
    evidence_enqueued = sum(row["evidence_enqueued_status"] == STATUS_TRUE for row in per_task)
    evidence_delivered = sum(row["evidence_delivered_status"] == STATUS_TRUE for row in per_task)
    alert_closed = sum(row["alert_before_deadline_status"] == STATUS_TRUE for row in per_task)
    confirmation_closed = sum(row["confirmation_before_deadline_status"] == STATUS_TRUE for row in per_task)
    metadata_only = sum(row["alert_before_deadline_status"] == STATUS_TRUE and row["confirmation_before_deadline_status"] != STATUS_TRUE for row in per_task)
    unknown_rate = _rate(unknown, total)
    label_counts = Counter(row["root_cause_label"] for row in per_task)
    confidence_counts = Counter(row["root_cause_confidence"] for row in per_task)
    classification = (
        "D. Evidence-closed staged behavior"
        if closed > 0 and _rate(closed, total) > 0.25 and unknown_rate < 0.25
        else "C. Partial staged behavior"
        if confirmation_closed > 0 and closed <= confirmation_closed
        else "A. Confirmed early-alert-only behavior"
        if closed == 0 and unknown == 0 and alert_closed > 0
        else "B. Apparent early-alert-only behavior with logging uncertainty"
    )

    missing_fields = [
        "payload_id",
        "stage_id",
        "payload_stage",
        "enqueue_time",
        "contact_window_id",
        "per-residual generated_time",
        "per-residual delivery_time",
        "drop_or_expire_reason",
        "parent_task_id",
        "residual_followup_of_payload_id",
    ]
    if classification.startswith("B"):
        why = "Evidence closure is not observed as a confirmed before-deadline event in the audited rows, but payload-level residual traces are insufficient to prove every possible closure failure mode."
    elif classification.startswith("C"):
        why = "Alert and some confirmation stages are observed, while evidence closure is not confirmed and some chains remain UNKNOWN due to logging insufficiency."
    elif classification.startswith("A"):
        why = "Evidence closure is provably zero in the available traces and stage-chain unknowns are absent."
    else:
        why = "Alert, confirmation, and evidence closure are directly observed with enough support."

    report_lines = [
        "# Staged Closure Root-Cause Report",
        "",
        "## Concept Boundary",
        "",
        "1. selected: the policy selected a payload/action type.",
        "2. generated: the simulator actually generated the corresponding payload artifact.",
        "3. enqueued: the payload was inserted into the downlink queue.",
        "4. delivered: the payload was delivered to ground before or after deadline.",
        "",
        "Delivered evidence is never inferred from selected action alone. Residual generation is not inferred from metadata delivery alone. Evidence closure is TRUE only for directly observed or high-confidence before-deadline evidence-grade delivery.",
        "",
        "## Aggregate Counts",
        "",
        f"- total task chains: {total}",
        f"- alert delivered before deadline count: {alert_closed}",
        f"- confirmation delivered before deadline count: {confirmation_closed}",
        f"- evidence selected count: {evidence_selected}",
        f"- evidence generated count: {evidence_generated}",
        f"- evidence enqueued count: {evidence_enqueued}",
        f"- evidence delivered count: {evidence_delivered}",
        f"- confirmed_evidence_closed_count: {closed}",
        f"- confirmed_evidence_failed_count: {failed}",
        f"- unknown_evidence_closure_count: {unknown}",
        f"- evidence_closure_unknown_rate: {unknown_rate:.6f}",
        f"- confirmed_metadata_only_alert_count: {metadata_only}",
        f"- unknown_stage_chain_count: {sum(row.get('unknown_stage_chain_count', 0) for row in stage_rows)}",
        "",
        "## Root-Cause Distribution",
        "",
        *[f"- {label}: {count}" for label, count in sorted(label_counts.items())],
        "",
        "## Root-Cause Confidence Distribution",
        "",
        *[f"- {label}: {count}" for label, count in sorted(confidence_counts.items())],
        "",
        "## Required Answers",
        "",
        f"1. Is evidence closure near zero because evidence-grade payloads are rarely selected? Evidence selected count is {evidence_selected}/{total}; this is an observed selected-stage count, not delivery evidence.",
        f"2. Is evidence closure near zero because residual/ROI+Ctx payloads are not generated? Evidence generated count is {evidence_generated}/{total}; residual generation uses residual generation counters where available and remains limited by missing payload-level timestamps.",
        f"3. Is evidence closure near zero because they are generated but not enqueued? Evidence generated but not enqueued cases: {label_counts['evidence_generated_but_not_enqueued']}.",
        f"4. Is evidence closure near zero because they are enqueued but miss contact capacity? Contact-capacity-labeled cases: {label_counts['contact_capacity_insufficient']}; do not treat this as final fact when confidence is not high.",
        f"5. Is evidence closure near zero because deadline expires before residual delivery? Evidence delivered after deadline cases: {label_counts['evidence_delivered_after_deadline']}; known expiry events are summarized separately.",
        f"6. Is evidence closure near zero because defer is too frequent? Deferred/no staged action cases are {label_counts['deferred_until_expiry'] + label_counts['no_staged_action_selected']}; this is selected/deferred evidence, not closure evidence.",
        "7. Is evidence closure near zero because the current simulator does not model residual follow-up sufficiently? The current logs show aggregate residual counters but lack payload_id-level residual traces, so this remains a behavior-plus-logging concern rather than a final fact.",
        "8. Is utility_rule strong because it delivers direct ROI/ROI+Ctx earlier? Use paired deltas and per-task chains; direct ROI/ROI+Ctx delivery is observed only where delivery_status and delivery_time support it.",
        f"9. Is ug_wx050 mainly producing metadata-only alert rather than completed staged lifecycle? Confirmed metadata-only alert count across all arms is {metadata_only}; see stage_breakpoint_summary.csv for arm-specific rates.",
        "10. Smallest next engineering block: add a default-off trace-only payload lifecycle trace, then rerun the same audit without changing reward.",
        "",
        "## Evidence strength and logging sufficiency",
        "",
        "1. Are logs sufficient to prove that evidence closure is truly zero? No, not globally, because payload_id-level residual generated/enqueued/delivered timestamps are missing.",
        "2. Or do logs only show that evidence closure is not observable? They show confirmed evidence closure where direct ROI+Ctx delivery is observable, confirmed failures where expiry/non-delivery is logged, and UNKNOWN where residual lifecycle fields are insufficient.",
        f"3. Which fields are missing for a definitive conclusion? {', '.join(missing_fields)}.",
        "4. Is the current result a system behavior issue, a logging issue, or both? Both: the observed traces show weak evidence closure, while missing payload-level residual trace fields prevent a definitive zero-closure claim.",
        "5. Minimal trace field patch needed: a default-off trace-only patch with the fields listed in final_recommendation.md. This task does not implement that patch.",
        "",
        "## Final Classification",
        "",
        f"{classification}: {why}",
    ]
    report = "\n".join(report_lines) + "\n"

    failures = [row for row in per_task if row["evidence_closure_status"] != STATUS_TRUE]
    failure_lines = [
        "# Evidence Closure Failure Cases",
        "",
        "Low-confidence or unknown root causes are listed as diagnostics, not final facts.",
        "",
    ]
    for row in failures[:80]:
        failure_lines.append(
            f"- {row['variant']} seed={row['seed']} task={row['task_id']} closure={row['evidence_closure_status']} "
            f"label={row['root_cause_label']} confidence={row['root_cause_confidence']} missing={row['missing_fields_for_this_task'] or 'none'}"
        )
    if len(failures) > 80:
        failure_lines.append(f"- truncated: {len(failures) - 80} additional cases are in per_task_stage_chain.csv")

    recommendation_lines = [
        "# Final Recommendation",
        "",
        f"Final classification: {classification}.",
        "",
        "Do not change reward, enable lifecycle utility, tune lifecycle weights, tune payload sizes, or alter contact/deadline parameters from this audit.",
        "",
        "Recommended next block: F. Logs are insufficient, so add report-only trace fields, default-off.",
        "",
        "Minimal trace field recommendation:",
        *[f"- {field}" for field in TRACE_PATCH_FIELDS],
        "",
        "Patch plan only:",
        "1. Add a disabled-by-default trace collector at the payload lifecycle boundary.",
        "2. Emit one immutable row per payload artifact and lifecycle transition.",
        "3. Keep the trace outside reward/scoring and outside main default configs.",
        "4. Rerun this staged-closure audit and require UNKNOWN closure rate to fall near zero before making behavior claims.",
    ]
    return report, "\n".join(failure_lines) + "\n", "\n".join(recommendation_lines) + "\n"


def _write_docs(
    output_dir: Path,
    input_files: list[Path],
    headers: dict[str, list[str]],
    per_task: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    command: str,
) -> None:
    _write_text(
        output_dir / "README.md",
        "\n".join(
            [
                "# Staged Closure Root-Cause Audit",
                "",
                "Standalone diagnostic-only audit for selected/generated/enqueued/delivered lifecycle stages.",
                "",
                "The audit uses three-valued TRUE/FALSE/UNKNOWN statuses and does not collapse UNKNOWN into FALSE.",
            ]
        )
        + "\n",
    )
    _write_text(
        output_dir / "command_log.md",
        "\n".join(
            [
                "# Command Log",
                "",
                f"- {datetime.now().isoformat(timespec='seconds')}: {command}",
                "- Inputs: existing lifecycle shadow and utility-guided multiseed outputs.",
                "- No reward/scoring/config changes were made by this audit.",
            ]
        )
        + "\n",
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_only": True,
        "input_roots": [str(LIFECYCLE_DIR), str(MULTISEED_DIR)],
        "input_files": [str(path.relative_to(PROJECT_ROOT)) for path in input_files],
        "seeds": sorted({int(row["seed"]) for row in per_task}),
        "variants": sorted({str(row["variant"]) for row in per_task}),
        "status_values": [STATUS_TRUE, STATUS_FALSE, STATUS_UNKNOWN],
    }
    _write_text(output_dir / "input_log_manifest.yaml", yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    _write_text(output_dir / "available_fields_audit.md", _fields_audit(input_files, headers))
    report, failures, recommendation = _report_text(per_task, stage_rows, paired_rows)
    _write_text(output_dir / "staged_closure_root_cause_report.md", report)
    _write_text(output_dir / "evidence_closure_failure_cases.md", failures)
    _write_text(output_dir / "final_recommendation.md", recommendation)


def _write_raw_sample(output_dir: Path, input_files: list[Path]) -> None:
    sample_rows: list[dict[str, Any]] = []
    for path in input_files[:4]:
        for row in _read_csv(path)[:8]:
            item = {"source_file": str(path.relative_to(PROJECT_ROOT))}
            for key in [
                "t",
                "task_id",
                "raw_action",
                "selected_payload_plan",
                "selected_product_type",
                "delivery_status",
                "payload_generation_time_s",
                "payload_delivery_time_s",
                "deadline_time_s",
                "delivered_task_ids",
                "expired_task_ids",
                "delivered_products",
                "pending_products",
                "residual_product_type",
                "residual_enqueue_count",
                "residual_delivery_count",
            ]:
                item[key] = row.get(key, "")
            sample_rows.append(item)
    _write_csv(output_dir / "raw_input_sample.csv", sample_rows)


def _checkpoint(output_dir: Path) -> None:
    _write_text(output_dir / "git_status.txt", _run_text(["git", "status", "--short"]))
    _write_text(output_dir / "git_diff.patch", _run_text(["git", "diff"]))


def _make_package(output_dir: Path) -> None:
    package_root = output_dir / "_package_staging"
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)
    for path in output_dir.iterdir():
        if path.name == "_package_staging":
            continue
        if path.is_file():
            shutil.copy2(path, package_root / path.name)
    scripts_dir = package_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "run_staged_closure_root_cause_audit.py", scripts_dir / "run_staged_closure_root_cause_audit.py")
    if PACKAGE_PATH.exists():
        PACKAGE_PATH.unlink()
    shutil.make_archive(str(PACKAGE_PATH.with_suffix("")), "zip", package_root)
    shutil.rmtree(package_root)


def run_audit(output_dir: Path, seeds: list[int], variants: list[str], command: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files: list[Path] = []
    headers: dict[str, list[str]] = {}
    per_task: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in variants:
            path = _trajectory_path(seed, variant)
            if not path.exists():
                continue
            input_files.append(path)
            task_rows, header = _task_rows_for_run(seed, variant, path)
            headers[str(path.relative_to(PROJECT_ROOT))] = header
            per_task.extend(task_rows)

    if not per_task:
        _write_text(output_dir / "FAILED_missing_required_fields.md", "No reconstructable task rows were found.\n")
        raise SystemExit(1)

    stage_rows, payload_rows, enqueue_rows, deadline_rows, queue_rows = _summaries(per_task)
    paired_rows = _paired_delta_rows(stage_rows)

    _write_csv(output_dir / "per_task_stage_chain.csv", per_task)
    _write_csv(output_dir / "stage_breakpoint_summary.csv", stage_rows)
    _write_csv(output_dir / "payload_generation_summary.csv", payload_rows)
    _write_csv(output_dir / "enqueue_delivery_summary.csv", enqueue_rows)
    _write_csv(output_dir / "deadline_capacity_summary.csv", deadline_rows)
    _write_csv(output_dir / "queue_blocking_summary.csv", queue_rows)
    _write_csv(output_dir / "paired_stage_delta_summary.csv", paired_rows)
    _write_raw_sample(output_dir, input_files)
    _write_docs(output_dir, input_files, headers, per_task, stage_rows, paired_rows, command)
    _checkpoint(output_dir)
    _make_package(output_dir)


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conservative staged-closure root-cause audit.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--variants", default=",".join(FOCUS_VARIANTS))
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    command = (
        "python scripts/run_staged_closure_root_cause_audit.py "
        f"--output-dir {args.output_dir} --seeds {','.join(str(seed) for seed in seeds)} --variants {','.join(variants)}"
    )
    run_audit(args.output_dir, seeds, variants, command)


if __name__ == "__main__":
    main()
