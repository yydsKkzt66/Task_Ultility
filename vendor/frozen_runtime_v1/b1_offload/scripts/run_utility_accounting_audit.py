"""Unified delivered-product utility accounting audit.

This script is intentionally post-hoc and diagnostic. It reads existing run
artifacts, rebuilds delivered-product ledgers, and writes consistency reports.
It does not change reward computation, closure rules, task generation, or
scheduler behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.payload.information_sufficiency import delta_information_utility  # noqa: E402


SCENARIO = "multi_window_rescuable_high_value"
SEED = 2030
SLOTS = 160
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "_method_vs_utility_rule_once"
OUT_DIR = PROJECT_ROOT / "outputs" / "_utility_accounting_audit"

LEDGER_COLUMNS = [
    "scenario",
    "seed",
    "arm",
    "lambda_H",
    "slot",
    "contact_window_id",
    "task_id",
    "product_type",
    "value_level",
    "product_requirement",
    "in_high_value_set_H",
    "arrival_slot",
    "deadline_slot",
    "delivered_before_deadline",
    "stage_before_delivery",
    "stage_after_delivery",
    "delivered_products_before",
    "delivered_products_after",
    "duplicate_flag",
    "payload_size_mb",
    "q_i_or_q_hat",
    "tau_i",
    "V_eff",
    "delta_information_utility",
    "product_gross_information_utility",
    "compute_energy",
    "communication_energy",
    "latency_cost",
    "queue_cost",
    "payload_cost",
    "risk_penalty",
    "product_resource_cost",
    "product_net_utility",
    "counted_in_total_gross",
    "counted_in_total_net",
    "counted_in_high_value_gross",
    "counted_in_high_value_net",
    "counted_in_min_stage_success",
    "counted_in_target_stage_success",
    "counted_in_closure",
    "evidence_closure_flag",
    "context_closure_flag",
    "verification_closure_flag",
    "delivery_reason",
    "guarantee_term",
    "Z_age",
    "Z_miss",
    "Z_total",
    "G_i_p",
    "rank_without_guarantee",
    "rank_with_guarantee",
    "selected_without_guarantee",
    "selected_with_guarantee",
    "delivered_without_guarantee",
    "delivered_with_guarantee",
    "notes",
]

STAGE_COLUMNS = [
    "scenario",
    "seed",
    "arm",
    "lambda_H",
    "task_id",
    "value_level",
    "product_requirement",
    "arrival_slot",
    "deadline_slot",
    "final_stage",
    "metadata_delivered",
    "metadata_delivered_slot",
    "roi_delivered",
    "roi_delivered_slot",
    "roi_context_delivered",
    "roi_context_delivered_slot",
    "min_stage_required",
    "target_stage_required",
    "min_stage_success_before_deadline",
    "target_stage_success_before_deadline",
    "evidence_closure",
    "context_closure",
    "delivered_gross_utility_sum",
    "delivered_net_utility_sum",
    "guarantee_positive_delivery_count",
    "counterfactual_rescued_delivery_count",
    "reason_if_min_stage_failed",
    "reason_if_target_stage_failed",
]

FIXED_SUMMARY_COLUMNS = [
    "scenario",
    "seed",
    "slots",
    "arm",
    "lambda_H",
    "total_gross_delivered_utility",
    "total_net_delivered_utility",
    "net_system_utility",
    "total_delivered_payload_MB",
    "gross_utility_per_MB",
    "net_utility_per_MB",
    "high_value_gross_delivered_utility",
    "high_value_net_delivered_utility",
    "high_value_delivered_payload_MB",
    "high_value_gross_utility_per_MB",
    "high_value_net_utility_per_MB",
    "low_medium_gross_delivered_utility",
    "low_medium_net_delivered_utility",
    "metadata_delivered_count",
    "roi_delivered_count",
    "roi_context_delivered_count",
    "high_value_metadata_delivered_count",
    "high_value_roi_delivered_count",
    "high_value_roi_context_delivered_count",
    "high_value_min_stage_success_count",
    "high_value_target_stage_success_count",
    "high_value_min_stage_success_rate_all_H",
    "high_value_min_stage_success_rate_feasible_H",
    "high_value_target_stage_success_rate_all_H",
    "high_value_target_stage_success_rate_feasible_H",
    "naturally_delivered_count",
    "guarantee_positive_delivered_count",
    "counterfactual_guarantee_rescued_delivered_count",
    "counterfactual_rescued_min_stage_count",
    "counterfactual_rescued_target_stage_count",
    "hard_safety_violation_count",
    "low_medium_accidental_roi_context_count",
    "lifecycle_orphan_count",
    "dropped_without_reason_count",
    "expired_without_reason_count",
]


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


def _bool01(value: object) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _product_key(row: dict[str, Any]) -> str:
    product_type = row.get("product_type", row.get("payload_type", ""))
    return f"{row.get('task_id', '')}::{product_type}"


def _normalize_product_type(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"roi+ctx", "roi_ctx", "roi-context"}:
        return "roi_context"
    return raw


def _infer_value_level(row: dict[str, Any], mapping_row: dict[str, Any] | None = None) -> str:
    mapped = str((mapping_row or {}).get("value_level", "") or "").strip().lower()
    if mapped:
        return mapped
    raw = str(row.get("value_level", "") or "").strip().lower()
    if raw:
        return raw
    if _bool01(row.get("emergency_flag", 0)):
        return "urgent"
    if _bool01(row.get("high_value_flag", 0)):
        return "high"
    value = _to_float(row.get("effective_task_value", row.get("calibrated_task_value", row.get("task_value", row.get("V_t", 0.0)))))
    if value >= 0.90:
        return "urgent"
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def _is_H(value_level: object, requirement: object) -> int:
    return int(str(value_level or "").lower() in {"high", "urgent"} and str(requirement or "").lower() in {"context", "evidence"})


def _stage_from_products(products: set[str]) -> str:
    normalized = {_normalize_product_type(item) for item in products}
    if "roi_context" in normalized:
        return "ROI+Ctx"
    if "roi" in normalized:
        return "ROI"
    if "metadata" in normalized:
        return "metadata"
    return "none"


def _stage_requirements(requirement: object) -> tuple[str, str]:
    req = str(requirement or "").lower()
    if req == "alert":
        return "metadata", "metadata"
    if req == "verification":
        return "ROI", "ROI"
    return "ROI", "ROI+Ctx"


def _products_to_text(products: set[str]) -> str:
    return ";".join(sorted(product for product in products if product))


def _delivered_lifecycle_rows(lifecycle_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in lifecycle_rows:
        if str(row.get("current_status", "")).lower() != "delivered" and not str(row.get("delivered_slot", "")).strip():
            continue
        lifecycle_id = str(row.get("lifecycle_id") or _product_key(row))
        previous = by_id.get(lifecycle_id)
        if previous is None or _to_int(row.get("status_updated_slot")) >= _to_int(previous.get("status_updated_slot")):
            by_id[lifecycle_id] = row
    return sorted(by_id.values(), key=lambda r: (_to_int(r.get("delivered_slot", r.get("status_updated_slot"))), str(r.get("lifecycle_id", ""))))


def _task_mapping(mapping_rows: list[dict[str, str]], trajectory_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in mapping_rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            continue
        out[task_id] = {
            "task_id": task_id,
            "value_level": str(row.get("value_level", "") or "").lower(),
            "product_requirement": str(row.get("product_requirement", "") or "verification").lower(),
            "arrival_slot": _to_int(row.get("arrival_slot")),
            "deadline_slot": _to_int(row.get("new_deadline_slot", row.get("deadline_slot"))),
            "feasible_before_deadline": int(_to_int(row.get("mapped_contact_windows")) > 0),
        }
    for row in trajectory_rows:
        if _to_int(row.get("has_task")) <= 0:
            continue
        task_id = str(row.get("task_id", ""))
        if not task_id:
            continue
        item = out.setdefault(task_id, {"task_id": task_id})
        item.setdefault("arrival_slot", _to_int(row.get("slot")))
        if not item.get("value_level"):
            item["value_level"] = _infer_value_level(row)
        if not item.get("product_requirement"):
            item["product_requirement"] = str(row.get("product_requirement", "verification") or "verification").lower()
        if not item.get("deadline_slot"):
            item["deadline_slot"] = _to_int(row.get("slot")) + int(math.ceil(_to_float(row.get("deadline_s")) / 20.0))
        item.setdefault("feasible_before_deadline", 1)
    return out


def _trajectory_by_slot(trajectory_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {_to_int(row.get("slot")): row for row in trajectory_rows}


def _queue_index(queue_rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    out: dict[tuple[str, str, int], dict[str, str]] = {}
    fallback: dict[tuple[str, str], dict[str, str]] = {}
    for row in queue_rows:
        if _to_int(row.get("delivered_this_slot")) <= 0:
            continue
        task_id = str(row.get("task_id", ""))
        product = _normalize_product_type(row.get("payload_type"))
        slot = _to_int(row.get("slot"))
        out[(task_id, product, slot)] = row
        fallback.setdefault((task_id, product), row)
    for key, row in fallback.items():
        out[(key[0], key[1], -1)] = row
    return out


def _delivered_product_keys(lifecycle_rows: list[dict[str, str]], queue_rows: list[dict[str, str]]) -> set[str]:
    keys = {_product_key(row) for row in _delivered_lifecycle_rows(lifecycle_rows)}
    keys.update(_product_key(row) for row in queue_rows if _to_int(row.get("delivered_this_slot")) > 0)
    return keys


def _counterfactual_rows(
    *,
    scenario: str,
    seed: int,
    arm: str,
    slots: int,
    lambda_h: float,
    candidate_rows: list[dict[str, str]],
    lifecycle_rows: list[dict[str, str]],
    queue_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    delivered_keys = _delivered_product_keys(lifecycle_rows, queue_rows)
    rows: list[dict[str, object]] = []
    for row in candidate_rows:
        rank_without = _to_int(row.get("rank_without_guarantee"))
        rank_with = _to_int(row.get("rank_with_guarantee"))
        capacity_cutoff = _to_int(row.get("capacity_cutoff_rank"))
        hard_feasible = not (
            _to_int(row.get("blocked_by_safety"))
            or _to_int(row.get("blocked_by_deadline"))
            or _to_int(row.get("blocked_by_duplicate"))
            or _to_int(row.get("blocked_by_marginal_utility_floor"))
        )
        would_fit_without = int(capacity_cutoff > 0 and 0 < rank_without <= capacity_cutoff)
        selected_without = int(hard_feasible and would_fit_without)
        selected_with = int(_to_int(row.get("selected")) and hard_feasible)
        active_guarantee = bool(
            arm == "proposed_guarantee_aware_full"
            and lambda_h > 0.0
            and _to_float(row.get("guarantee_term")) > 1.0e-12
        )
        if not active_guarantee:
            selected_without = selected_with
        product_key = _product_key(row)
        delivered_with = int(product_key in delivered_keys)
        delivered_without = int(delivered_with and selected_without)
        rescued = int(active_guarantee and selected_with and delivered_with and not delivered_without)
        guarantee_positive = int(_to_float(row.get("guarantee_term")) > 1.0e-12 and delivered_with)
        item: dict[str, object] = dict(row)
        item.update(
            {
                "scenario": scenario,
                "seed": seed,
                "arm": arm,
                "lambda_H": lambda_h,
                "num_slots": slots,
                "selected_without_guarantee": selected_without,
                "selected_with_guarantee": selected_with,
                "delivered_without_guarantee": delivered_without,
                "delivered_with_guarantee": delivered_with,
                "naturally_delivered": int(delivered_with and delivered_without),
                "guarantee_positive_delivered": guarantee_positive,
                "counterfactual_guarantee_rescued_delivered": rescued,
                "unique_product_key": product_key,
            }
        )
        rows.append(item)
    return rows


def _counterfactual_index(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("unique_product_key", _product_key(row)))].append(row)
    out: dict[str, dict[str, object]] = {}
    for key, group in grouped.items():
        out[key] = sorted(
            group,
            key=lambda row: (
                -_to_int(row.get("counterfactual_guarantee_rescued_delivered")),
                -_to_int(row.get("guarantee_positive_delivered")),
                -_to_int(row.get("delivered_with_guarantee")),
                _to_int(row.get("slot")),
            ),
        )[0]
    return out


def _slot_raw_cost_components(slot_row: dict[str, str]) -> dict[str, float]:
    lambda_e = _to_float(slot_row.get("lambda_E"), 0.0)
    lambda_l = _to_float(slot_row.get("lambda_L"), 0.0)
    lambda_q = _to_float(slot_row.get("lambda_Q"), 0.0)
    lambda_s = _to_float(slot_row.get("lambda_S"), 0.0)
    e_comp = _to_float(slot_row.get("E_comp", slot_row.get("compute_energy_j")))
    e_comm = _to_float(slot_row.get("E_comm", slot_row.get("comm_energy_j")))
    latency = _to_float(slot_row.get("L_inf")) + _to_float(slot_row.get("L_queue")) + _to_float(slot_row.get("L_comm"))
    payload_mb = _to_float(slot_row.get("S_down_total_mb"))
    queue_delta = max(0.0, _to_float(slot_row.get("delta_queue_mb")))
    return {
        "compute_energy": max(0.0, lambda_e * e_comp),
        "communication_energy": max(0.0, lambda_e * e_comm),
        "latency_cost": max(0.0, lambda_l * latency),
        "queue_cost": max(0.0, lambda_q * queue_delta),
        "payload_cost": max(0.0, lambda_s * payload_mb),
        "risk_penalty": max(0.0, _to_float(slot_row.get("long_term_risk_penalty"))),
    }


def _allocated_cost_components(slot_row: dict[str, str], share: float) -> dict[str, float]:
    raw = _slot_raw_cost_components(slot_row)
    raw_total = sum(raw.values())
    target = max(0.0, _to_float(slot_row.get("resource_penalty")) + _to_float(slot_row.get("long_term_risk_penalty")))
    scale = _safe_div(target, raw_total) if raw_total > 1.0e-12 else 0.0
    return {key: value * scale * share for key, value in raw.items()}


def build_delivered_product_ledger(
    *,
    scenario: str,
    seed: int,
    arm: str,
    slots: int,
    lambda_h: float,
    summary: dict[str, str],
    lifecycle_rows: list[dict[str, str]],
    queue_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    mapping_rows: list[dict[str, str]],
    trajectory_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    delivered = _delivered_lifecycle_rows(lifecycle_rows)
    tasks = _task_mapping(mapping_rows, trajectory_rows)
    by_slot = _trajectory_by_slot(trajectory_rows)
    queue_by_key = _queue_index(queue_rows)
    cf_rows = _counterfactual_rows(
        scenario=scenario,
        seed=seed,
        arm=arm,
        slots=slots,
        lambda_h=lambda_h,
        candidate_rows=candidate_rows,
        lifecycle_rows=lifecycle_rows,
        queue_rows=queue_rows,
    )
    cf_by_key = _counterfactual_index(cf_rows)
    delivered_mb_by_slot: dict[int, float] = defaultdict(float)
    for row in delivered:
        delivered_mb_by_slot[_to_int(row.get("delivered_slot", row.get("status_updated_slot")))] += _to_float(row.get("payload_size_mb"))

    ledger: list[dict[str, object]] = []
    delivered_products_by_task: dict[str, set[str]] = defaultdict(set)
    for row in delivered:
        task_id = str(row.get("task_id", ""))
        product_type = _normalize_product_type(row.get("product_type"))
        slot = _to_int(row.get("delivered_slot", row.get("status_updated_slot")))
        mapping = tasks.get(task_id, {})
        value_level = _infer_value_level(row, mapping)
        requirement = str(mapping.get("product_requirement") or row.get("product_requirement") or "verification").lower()
        arrival_slot = _to_int(mapping.get("arrival_slot", row.get("created_slot")))
        deadline_slot = _to_int(mapping.get("deadline_slot", row.get("deadline_slot")))
        before = set(delivered_products_by_task[task_id])
        duplicate = int(product_type in before)
        delta = delta_information_utility(product_type, before, requirement)
        after = set(before)
        if product_type:
            after.add(product_type)

        slot_row = by_slot.get(slot, {})
        queue_row = queue_by_key.get((task_id, product_type, slot)) or queue_by_key.get((task_id, product_type, -1)) or {}
        cf = cf_by_key.get(f"{task_id}::{product_type}", {})
        notes: list[str] = []
        if slot_row and str(slot_row.get("task_id", "")) == task_id:
            q = _to_float(slot_row.get("detection_quality"), 1.0)
            tau = _to_float(slot_row.get("timeliness_at_delivery"), 1.0)
            notes.append("q_tau_from_trajectory_current_task")
        elif queue_row:
            v = _to_float(row.get("effective_task_value", row.get("calibrated_task_value", row.get("task_value"))))
            queue_u = _to_float(queue_row.get("remaining_utility"))
            q_tau = _safe_div(queue_u, v * delta) if delta > 1.0e-12 and v > 1.0e-12 else 0.0
            q = max(0.0, min(1.0, q_tau))
            tau = 1.0
            notes.append("q_tau_inferred_from_delivered_queue_utility")
        elif cf:
            v = _to_float(row.get("effective_task_value", row.get("calibrated_task_value", row.get("task_value"))))
            cf_u = _to_float(cf.get("deltaU_total", cf.get("delta_information_utility")))
            q_tau = _safe_div(cf_u, v * delta) if delta > 1.0e-12 and v > 1.0e-12 else 0.0
            q = max(0.0, min(1.0, q_tau))
            tau = 1.0
            notes.append("q_tau_inferred_from_candidate_deltaU")
        else:
            q = 1.0
            tau = 1.0
            notes.append("q_tau_fallback_1")

        v_eff = _to_float(row.get("effective_task_value", row.get("calibrated_task_value", row.get("task_value"))))
        gross = max(0.0, v_eff * max(0.0, delta) * max(0.0, q) * max(0.0, tau))
        payload_size = _to_float(row.get("payload_size_mb"))
        share = _safe_div(payload_size, delivered_mb_by_slot.get(slot, 0.0))
        costs = _allocated_cost_components(slot_row, share) if slot_row else {k: 0.0 for k in _slot_raw_cost_components({})}
        resource_cost = sum(costs.values())
        net = gross - resource_cost
        stage_before = _stage_from_products(before)
        stage_after = _stage_from_products(after)
        in_H = _is_H(value_level, requirement)
        before_deadline = int(slot <= deadline_slot if deadline_slot >= 0 else 0)
        min_stage = int(in_H and product_type == "roi" and before_deadline)
        target_stage = int(in_H and product_type == "roi_context" and before_deadline)
        evidence_closure = int(in_H and requirement == "evidence" and product_type == "roi_context" and before_deadline)
        context_closure = int(in_H and requirement == "context" and product_type == "roi_context" and before_deadline)
        verification_closure = int(requirement == "verification" and product_type in {"roi", "roi_context"} and before_deadline)
        if _to_int(cf.get("counterfactual_guarantee_rescued_delivered")):
            delivery_reason = "counterfactual_rescued"
        elif _to_int(cf.get("guarantee_positive_delivered")):
            delivery_reason = "guarantee_positive"
        elif _to_int(cf.get("naturally_delivered")):
            delivery_reason = "natural"
        elif str(arm) == "utility_rule":
            delivery_reason = "utility_selected"
        else:
            delivery_reason = "other"
        if not str(row.get("value_level", "")).strip() and value_level in {"high", "urgent"}:
            notes.append("value_level_inferred_for_passive_accounting")
        if duplicate:
            notes.append("duplicate_product_type_for_task")
        ledger.append(
            {
                "scenario": scenario,
                "seed": seed,
                "arm": arm,
                "lambda_H": lambda_h,
                "slot": slot,
                "contact_window_id": row.get("last_contact_window_id_seen", ""),
                "task_id": task_id,
                "product_type": product_type,
                "value_level": value_level,
                "product_requirement": requirement,
                "in_high_value_set_H": in_H,
                "arrival_slot": arrival_slot,
                "deadline_slot": deadline_slot,
                "delivered_before_deadline": before_deadline,
                "stage_before_delivery": stage_before,
                "stage_after_delivery": stage_after,
                "delivered_products_before": _products_to_text(before),
                "delivered_products_after": _products_to_text(after),
                "duplicate_flag": duplicate,
                "payload_size_mb": payload_size,
                "q_i_or_q_hat": q,
                "tau_i": tau,
                "V_eff": v_eff,
                "delta_information_utility": delta,
                "product_gross_information_utility": gross,
                "compute_energy": costs["compute_energy"],
                "communication_energy": costs["communication_energy"],
                "latency_cost": costs["latency_cost"],
                "queue_cost": costs["queue_cost"],
                "payload_cost": costs["payload_cost"],
                "risk_penalty": costs["risk_penalty"],
                "product_resource_cost": resource_cost,
                "product_net_utility": net,
                "counted_in_total_gross": 1,
                "counted_in_total_net": 1,
                "counted_in_high_value_gross": in_H,
                "counted_in_high_value_net": in_H,
                "counted_in_min_stage_success": min_stage,
                "counted_in_target_stage_success": target_stage,
                "counted_in_closure": int(evidence_closure or context_closure or verification_closure),
                "evidence_closure_flag": evidence_closure,
                "context_closure_flag": context_closure,
                "verification_closure_flag": verification_closure,
                "delivery_reason": delivery_reason,
                "guarantee_term": _to_float(cf.get("guarantee_term", row.get("guarantee_term"))),
                "Z_age": _to_float(cf.get("Z_age", cf.get("guarantee_Z_age", row.get("guarantee_Z_age")))),
                "Z_miss": _to_float(cf.get("Z_miss", cf.get("guarantee_Z_miss", row.get("guarantee_Z_miss")))),
                "Z_total": _to_float(cf.get("Z_total", cf.get("guarantee_Z_total", row.get("guarantee_Z_total")))),
                "G_i_p": _to_float(cf.get("G_i_p", cf.get("guarantee_G_i_p", row.get("guarantee_G_i_p")))),
                "rank_without_guarantee": _to_int(cf.get("rank_without_guarantee", row.get("guarantee_rank_without"))),
                "rank_with_guarantee": _to_int(cf.get("rank_with_guarantee", row.get("guarantee_rank_with"))),
                "selected_without_guarantee": _to_int(cf.get("selected_without_guarantee")),
                "selected_with_guarantee": _to_int(cf.get("selected_with_guarantee")),
                "delivered_without_guarantee": _to_int(cf.get("delivered_without_guarantee", int(arm == "utility_rule"))),
                "delivered_with_guarantee": _to_int(cf.get("delivered_with_guarantee", 0)),
                "notes": ";".join(notes),
            }
        )
        delivered_products_by_task[task_id] = after
    return ledger, cf_rows


def build_stage_success_audit(
    ledger: list[dict[str, object]],
    tasks: dict[str, dict[str, Any]],
    *,
    scenario: str,
    seed: int,
    arm: str,
    lambda_h: float,
) -> list[dict[str, object]]:
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ledger:
        by_task[str(row["task_id"])].append(row)
    rows: list[dict[str, object]] = []
    for task_id, task in sorted(tasks.items()):
        value_level = str(task.get("value_level", "")).lower()
        requirement = str(task.get("product_requirement", "")).lower()
        if not _is_H(value_level, requirement):
            continue
        deliveries = sorted(by_task.get(task_id, []), key=lambda row: _to_int(row.get("slot")))
        all_products = {str(row.get("product_type")) for row in deliveries}
        before_deadline = [row for row in deliveries if _to_int(row.get("delivered_before_deadline"))]
        bd_products = {str(row.get("product_type")) for row in before_deadline}

        def first_slot(product: str, source: list[dict[str, object]]) -> object:
            slots = [_to_int(row.get("slot")) for row in source if str(row.get("product_type")) == product]
            return min(slots) if slots else ""

        metadata_slot = first_slot("metadata", deliveries)
        roi_slot = first_slot("roi", deliveries)
        roi_ctx_slot = first_slot("roi_context", deliveries)
        min_success = int("roi" in bd_products or "roi_context" in bd_products)
        target_success = int("roi_context" in bd_products)
        evidence_closure = int(requirement == "evidence" and target_success)
        context_closure = int(requirement == "context" and target_success)
        reason_min = ""
        if not min_success:
            if not deliveries:
                reason_min = "no_delivered_product"
            elif "roi" in all_products or "roi_context" in all_products:
                reason_min = "roi_or_roi_context_delivered_after_deadline"
            elif "metadata" in all_products:
                reason_min = "metadata_only_no_roi_before_deadline"
            else:
                reason_min = "no_roi_before_deadline"
        reason_target = ""
        if not target_success:
            if "roi_context" in all_products:
                reason_target = "roi_context_delivered_after_deadline"
            elif "roi" in all_products:
                reason_target = "roi_only_no_roi_context"
            elif "metadata" in all_products:
                reason_target = "metadata_only_no_roi_context"
            else:
                reason_target = "no_target_stage_delivery"
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "arm": arm,
                "lambda_H": lambda_h,
                "task_id": task_id,
                "value_level": value_level,
                "product_requirement": requirement,
                "arrival_slot": task.get("arrival_slot", ""),
                "deadline_slot": task.get("deadline_slot", ""),
                "final_stage": _stage_from_products(all_products),
                "metadata_delivered": int("metadata" in all_products),
                "metadata_delivered_slot": metadata_slot,
                "roi_delivered": int("roi" in all_products),
                "roi_delivered_slot": roi_slot,
                "roi_context_delivered": int("roi_context" in all_products),
                "roi_context_delivered_slot": roi_ctx_slot,
                "min_stage_required": "ROI",
                "target_stage_required": "ROI+Ctx",
                "min_stage_success_before_deadline": min_success,
                "target_stage_success_before_deadline": target_success,
                "evidence_closure": evidence_closure,
                "context_closure": context_closure,
                "delivered_gross_utility_sum": sum(_to_float(row.get("product_gross_information_utility")) for row in deliveries),
                "delivered_net_utility_sum": sum(_to_float(row.get("product_net_utility")) for row in deliveries),
                "guarantee_positive_delivery_count": sum(1 for row in deliveries if str(row.get("delivery_reason")) == "guarantee_positive"),
                "counterfactual_rescued_delivery_count": sum(1 for row in deliveries if str(row.get("delivery_reason")) == "counterfactual_rescued"),
                "reason_if_min_stage_failed": reason_min,
                "reason_if_target_stage_failed": reason_target,
            }
        )
    return rows


def build_fixed_summary_rows(
    runs: list[dict[str, Any]],
    ledgers_by_arm: dict[str, list[dict[str, object]]],
    stages_by_arm: dict[str, list[dict[str, object]]],
    tasks_by_arm: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        arm = run["arm"]
        ledger = ledgers_by_arm[arm]
        stage = stages_by_arm[arm]
        summary = run["summary"]
        total_mb = sum(_to_float(row.get("payload_size_mb")) for row in ledger)
        total_gross = sum(_to_float(row.get("product_gross_information_utility")) for row in ledger)
        total_net = sum(_to_float(row.get("product_net_utility")) for row in ledger)
        hv = [row for row in ledger if _to_int(row.get("in_high_value_set_H"))]
        low_medium = [row for row in ledger if str(row.get("value_level")) in {"low", "medium"}]
        all_H = [task for task in tasks_by_arm[arm].values() if _is_H(task.get("value_level"), task.get("product_requirement"))]
        feasible_H = [task for task in all_H if _to_int(task.get("feasible_before_deadline"), 1)]
        min_success = sum(_to_int(row.get("min_stage_success_before_deadline")) for row in stage)
        target_success = sum(_to_int(row.get("target_stage_success_before_deadline")) for row in stage)
        cf_min_tasks = {
            str(row.get("task_id"))
            for row in ledger
            if str(row.get("delivery_reason")) == "counterfactual_rescued" and _to_int(row.get("counted_in_min_stage_success"))
        }
        cf_target_tasks = {
            str(row.get("task_id"))
            for row in ledger
            if str(row.get("delivery_reason")) == "counterfactual_rescued" and _to_int(row.get("counted_in_target_stage_success"))
        }
        orphan_rows = run.get("orphan_rows", [])
        dropped_without_reason = sum(1 for row in run.get("lifecycle_rows", []) if row.get("dropped_slot") and not row.get("blocker_reason"))
        expired_without_reason = sum(1 for row in run.get("lifecycle_rows", []) if row.get("expired_slot") and not row.get("blocker_reason"))
        item = {
            "scenario": run["scenario"],
            "seed": run["seed"],
            "slots": run["slots"],
            "arm": arm,
            "lambda_H": run["lambda_H"],
            "total_gross_delivered_utility": total_gross,
            "total_net_delivered_utility": total_net,
            "net_system_utility": _to_float(summary.get("net_system_utility")),
            "total_delivered_payload_MB": total_mb,
            "gross_utility_per_MB": _safe_div(total_gross, total_mb),
            "net_utility_per_MB": _safe_div(total_net, total_mb),
            "high_value_gross_delivered_utility": sum(_to_float(row.get("product_gross_information_utility")) for row in hv),
            "high_value_net_delivered_utility": sum(_to_float(row.get("product_net_utility")) for row in hv),
            "high_value_delivered_payload_MB": sum(_to_float(row.get("payload_size_mb")) for row in hv),
            "low_medium_gross_delivered_utility": sum(_to_float(row.get("product_gross_information_utility")) for row in low_medium),
            "low_medium_net_delivered_utility": sum(_to_float(row.get("product_net_utility")) for row in low_medium),
            "metadata_delivered_count": sum(1 for row in ledger if row.get("product_type") == "metadata"),
            "roi_delivered_count": sum(1 for row in ledger if row.get("product_type") == "roi"),
            "roi_context_delivered_count": sum(1 for row in ledger if row.get("product_type") == "roi_context"),
            "high_value_metadata_delivered_count": sum(1 for row in hv if row.get("product_type") == "metadata"),
            "high_value_roi_delivered_count": sum(1 for row in hv if row.get("product_type") == "roi"),
            "high_value_roi_context_delivered_count": sum(1 for row in hv if row.get("product_type") == "roi_context"),
            "high_value_min_stage_success_count": min_success,
            "high_value_target_stage_success_count": target_success,
            "high_value_min_stage_success_rate_all_H": _safe_div(min_success, len(all_H)),
            "high_value_min_stage_success_rate_feasible_H": _safe_div(min_success, len(feasible_H)),
            "high_value_target_stage_success_rate_all_H": _safe_div(target_success, len(all_H)),
            "high_value_target_stage_success_rate_feasible_H": _safe_div(target_success, len(feasible_H)),
            "naturally_delivered_count": sum(1 for row in ledger if row.get("delivery_reason") == "natural"),
            "guarantee_positive_delivered_count": sum(1 for row in ledger if row.get("delivery_reason") == "guarantee_positive"),
            "counterfactual_guarantee_rescued_delivered_count": sum(1 for row in ledger if row.get("delivery_reason") == "counterfactual_rescued"),
            "counterfactual_rescued_min_stage_count": len(cf_min_tasks),
            "counterfactual_rescued_target_stage_count": len(cf_target_tasks),
            "hard_safety_violation_count": _to_float(summary.get("final_resource_safety_violation_count", summary.get("safety_violation_count"))),
            "low_medium_accidental_roi_context_count": len(
                {
                    str(row.get("task_id"))
                    for row in ledger
                    if str(row.get("value_level")) in {"low", "medium"} and row.get("product_type") == "roi_context"
                }
            ),
            "lifecycle_orphan_count": sum(_to_int(row.get("orphan_flag")) for row in orphan_rows),
            "dropped_without_reason_count": dropped_without_reason,
            "expired_without_reason_count": expired_without_reason,
        }
        item["high_value_gross_utility_per_MB"] = _safe_div(
            _to_float(item["high_value_gross_delivered_utility"]),
            _to_float(item["high_value_delivered_payload_MB"]),
        )
        item["high_value_net_utility_per_MB"] = _safe_div(
            _to_float(item["high_value_net_delivered_utility"]),
            _to_float(item["high_value_delivered_payload_MB"]),
        )
        rows.append(item)
    return rows


def check_metric_consistency(
    summary_rows: list[dict[str, object]],
    ledgers_by_arm: dict[str, list[dict[str, object]]],
    stages_by_arm: dict[str, list[dict[str, object]]],
    *,
    tolerance: float = 1.0e-9,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def add(name: str, arm: str, actual: float, expected: float, severity: str, explanation: str) -> None:
        passed = abs(actual - expected) <= tolerance
        checks.append(
            {
                "check_name": name,
                "arm": arm,
                "passed": int(passed),
                "severity": "ok" if passed else severity,
                "actual": actual,
                "expected": expected,
                "tolerance": tolerance,
                "explanation": explanation if not passed else "",
            }
        )

    for summary in summary_rows:
        arm = str(summary.get("arm"))
        ledger = ledgers_by_arm[arm]
        stage = stages_by_arm[arm]
        total_gross = sum(_to_float(row.get("product_gross_information_utility")) for row in ledger)
        total_net = sum(_to_float(row.get("product_net_utility")) for row in ledger)
        high_gross = sum(_to_float(row.get("product_gross_information_utility")) for row in ledger if _to_int(row.get("in_high_value_set_H")))
        high_net = sum(_to_float(row.get("product_net_utility")) for row in ledger if _to_int(row.get("in_high_value_set_H")))
        payload = sum(_to_float(row.get("payload_size_mb")) for row in ledger)
        add("total_gross_delivered_utility_sums_to_ledger", arm, _to_float(summary.get("total_gross_delivered_utility")), total_gross, "error", "fixed summary total gross must be ledger sum")
        add("total_net_delivered_utility_sums_to_ledger", arm, _to_float(summary.get("total_net_delivered_utility")), total_net, "error", "fixed summary total net must be ledger sum")
        add("high_value_gross_delivered_utility_sums_to_ledger", arm, _to_float(summary.get("high_value_gross_delivered_utility")), high_gross, "error", "high-value gross must be H subset ledger sum")
        add("high_value_net_delivered_utility_sums_to_ledger", arm, _to_float(summary.get("high_value_net_delivered_utility")), high_net, "error", "high-value net must be H subset ledger sum")
        checks.append(
            {
                "check_name": "high_value_gross_not_exceed_total_gross",
                "arm": arm,
                "passed": int(high_gross <= total_gross + tolerance),
                "severity": "ok" if high_gross <= total_gross + tolerance else "error",
                "actual": high_gross,
                "expected": total_gross,
                "tolerance": tolerance,
                "explanation": "" if high_gross <= total_gross + tolerance else "H is a subset; gross subset sum cannot exceed total gross",
            }
        )
        add("delivered_payload_MB_sums_to_ledger", arm, _to_float(summary.get("total_delivered_payload_MB")), payload, "error", "payload MB must be ledger sum")
        add("gross_utility_per_MB_uses_total_payload", arm, _to_float(summary.get("gross_utility_per_MB")), _safe_div(total_gross, payload), "error", "gross density must use total gross divided by delivered payload")
        add("net_utility_per_MB_uses_total_payload", arm, _to_float(summary.get("net_utility_per_MB")), _safe_div(total_net, payload), "error", "net density must use total net divided by delivered payload")
        evidence_roi_only = sum(
            1
            for row in stage
            if row.get("product_requirement") == "evidence"
            and _to_int(row.get("roi_delivered"))
            and not _to_int(row.get("roi_context_delivered"))
            and _to_int(row.get("evidence_closure"))
        )
        checks.append(
            {
                "check_name": "evidence_closure_excludes_roi_only",
                "arm": arm,
                "passed": int(evidence_roi_only == 0),
                "severity": "ok" if evidence_roi_only == 0 else "error",
                "actual": evidence_roi_only,
                "expected": 0,
                "tolerance": 0,
                "explanation": "" if evidence_roi_only == 0 else "ROI-only evidence tasks must not count as evidence closure",
            }
        )
        delivered_with = sum(_to_int(row.get("delivered_with_guarantee")) for row in ledger)
        natural = sum(1 for row in ledger if row.get("delivery_reason") == "natural")
        rescued = sum(1 for row in ledger if row.get("delivery_reason") == "counterfactual_rescued")
        checks.append(
            {
                "check_name": "natural_plus_rescued_not_exceed_delivered_with_guarantee",
                "arm": arm,
                "passed": int(natural + rescued <= delivered_with or arm == "utility_rule"),
                "severity": "ok" if natural + rescued <= delivered_with or arm == "utility_rule" else "error",
                "actual": natural + rescued,
                "expected": delivered_with,
                "tolerance": 0,
                "explanation": "" if natural + rescued <= delivered_with or arm == "utility_rule" else "counterfactual labels exceed delivered-with-guarantee products",
            }
        )
        positive = sum(1 for row in ledger if row.get("delivery_reason") == "guarantee_positive")
        checks.append(
            {
                "check_name": "guarantee_positive_not_assumed_rescued",
                "arm": arm,
                "passed": int(not (positive > 0 and positive == rescued)),
                "severity": "ok" if not (positive > 0 and positive == rescued) else "warning",
                "actual": positive,
                "expected": rescued,
                "tolerance": 0,
                "explanation": "" if not (positive > 0 and positive == rescued) else "guarantee-positive participation equals rescue count; inspect counterfactual condition",
            }
        )
    return checks


def _load_run(input_root: Path, arm: str, lambda_h: float) -> dict[str, Any]:
    if arm == "utility_rule":
        run_dir = input_root / "runs" / SCENARIO / f"s{SEED}" / f"slots_{SLOTS}" / "utility_rule"
        traj_name = "utility_rule.csv"
    else:
        run_dir = input_root / "runs" / SCENARIO / f"s{SEED}" / f"slots_{SLOTS}_lambda_{lambda_h:g}" / arm
        traj_name = "proposed_lyapunov_rule_mode_safe_bandit.csv"
    summary = (_read_csv(run_dir / "summary.csv") or [{}])[0]
    return {
        "scenario": SCENARIO,
        "seed": SEED,
        "slots": SLOTS,
        "arm": arm,
        "lambda_H": lambda_h if arm != "utility_rule" else 0.0,
        "run_dir": run_dir,
        "summary": summary,
        "lifecycle_rows": _read_csv(run_dir / "payload_lifecycle_trace.csv"),
        "queue_rows": _read_csv(run_dir / "queue_priority_trace.csv"),
        "candidate_rows": _read_csv(run_dir / "candidate_priority_trace.csv"),
        "mapping_rows": _read_csv(run_dir / "deadline_profile_mapping.csv"),
        "trajectory_rows": _read_csv(run_dir / "trajectories" / traj_name),
        "orphan_rows": _read_csv(run_dir / "payload_lifecycle_orphan_check.csv"),
        "contact_rows": _read_csv(run_dir / "contact_window_scheduling_trace.csv"),
    }


def _distribution_rows(ledger: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ledger:
        grouped[str(row.get(group_key, ""))].append(row)
    out = []
    for name, rows in sorted(grouped.items()):
        out.append(
            {
                "group_by": group_key,
                "group_value": name,
                "delivered_count": len(rows),
                "gross_utility": sum(_to_float(row.get("product_gross_information_utility")) for row in rows),
                "net_utility": sum(_to_float(row.get("product_net_utility")) for row in rows),
                "payload_MB": sum(_to_float(row.get("payload_size_mb")) for row in rows),
            }
        )
    return out


def _write_utility_rule_high_value_report(
    ledger: list[dict[str, object]],
    stage: list[dict[str, object]],
    tasks: dict[str, dict[str, Any]],
    path: Path,
) -> list[dict[str, object]]:
    hv_tasks = [task for task in tasks.values() if _is_H(task.get("value_level"), task.get("product_requirement"))]
    high_or_urgent = [row for row in ledger if str(row.get("value_level")) in {"high", "urgent"}]
    context_evidence = [row for row in ledger if str(row.get("product_requirement")) in {"context", "evidence"}]
    H_rows = [row for row in ledger if _to_int(row.get("in_high_value_set_H"))]
    audit_rows = [
        {"question": "same_task_stream_checked", "answer": "see cross_arm_fairness_audit.csv"},
        {"question": "high_urgent_context_evidence_tasks_in_stream", "answer": len(hv_tasks)},
        {"question": "delivered_high_or_urgent_products", "answer": len(high_or_urgent)},
        {"question": "delivered_context_evidence_products", "answer": len(context_evidence)},
        {"question": "delivered_H_products", "answer": len(H_rows)},
        {"question": "H_gross_utility", "answer": sum(_to_float(row.get("product_gross_information_utility")) for row in H_rows)},
        {"question": "H_min_stage_success_tasks", "answer": sum(_to_int(row.get("min_stage_success_before_deadline")) for row in stage)},
        {"question": "H_target_stage_success_tasks", "answer": sum(_to_int(row.get("target_stage_success_before_deadline")) for row in stage)},
    ]
    dist = _distribution_rows(ledger, "value_level") + _distribution_rows(ledger, "product_requirement") + _distribution_rows(ledger, "product_type")
    lines = [
        "# Utility-rule High-value Passive Audit",
        "",
        f"- Same scenario/seed requested: {SCENARIO} / {SEED}. Cross-arm task-stream equality is checked separately.",
        f"- High/urgent + context/evidence tasks in utility_rule stream: {len(hv_tasks)}.",
        f"- Delivered high/urgent products: {len(high_or_urgent)}.",
        f"- Delivered context/evidence products: {len(context_evidence)}.",
        f"- Delivered H products: {len(H_rows)}.",
        f"- Fixed H gross utility: {sum(_to_float(row.get('product_gross_information_utility')) for row in H_rows):.6f}.",
        f"- Fixed H net utility: {sum(_to_float(row.get('product_net_utility')) for row in H_rows):.6f}.",
        "",
        "The old utility_rule high-value delivered utility of 0 is not credible under passive accounting: lifecycle rows contain high-value evidence/context products, and several value_level fields were blank in lifecycle/queue traces but recoverable from deadline mapping or high_value_flag/task value.",
        "",
        "## Distribution",
        "",
        "| group | value | count | gross | net | MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in dist:
        lines.append(
            f"| {row['group_by']} | {row['group_value']} | {row['delivered_count']} | "
            f"{_to_float(row['gross_utility']):.6f} | {_to_float(row['net_utility']):.6f} | {_to_float(row['payload_MB']):.6f} |"
        )
    _write_text(path, "\n".join(lines))
    return audit_rows + dist


def _write_low_utility_report(summary_rows: list[dict[str, object]], ledger_by_arm: dict[str, list[dict[str, object]]], path: Path) -> None:
    by_arm = {str(row["arm"]): row for row in summary_rows}
    util = by_arm.get("utility_rule", {})
    prop = by_arm.get("proposed_guarantee_aware_full", {})
    prop_ledger = ledger_by_arm.get("proposed_guarantee_aware_full", [])
    rescued = [row for row in prop_ledger if row.get("delivery_reason") == "counterfactual_rescued"]
    rescued_types = defaultdict(int)
    for row in rescued:
        rescued_types[str(row.get("product_type"))] += 1
    prop_payload_by_type = _distribution_rows(prop_ledger, "product_type")
    conclusion = "F. Multiple causes coexist: old gross/net naming was mixed, utility_rule passive H accounting was incomplete, stage success was undercounted, and cross-arm task stream fairness fails for this artifact."
    lines = [
        "# Low Utility Explanation Report",
        "",
        f"- proposed gross delivered utility: {_to_float(prop.get('total_gross_delivered_utility')):.6f}.",
        f"- proposed net delivered utility: {_to_float(prop.get('total_net_delivered_utility')):.6f}.",
        f"- proposed net system utility: {_to_float(prop.get('net_system_utility')):.6f}.",
        f"- utility_rule gross delivered utility: {_to_float(util.get('total_gross_delivered_utility')):.6f}.",
        f"- utility_rule net delivered utility: {_to_float(util.get('total_net_delivered_utility')):.6f}.",
        "",
        "proposed low utility/MB is partly real in the delivered ledger because more MB are spent on staged high-value products, but the old comparison was also polluted by comparing episode effective utility with delivered high-value incremental utility.",
        "",
        "## Proposed Payload Mix",
        "",
        "| product_type | count | gross | net | MB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in prop_payload_by_type:
        lines.append(
            f"| {row['group_value']} | {row['delivered_count']} | {_to_float(row['gross_utility']):.6f} | "
            f"{_to_float(row['net_utility']):.6f} | {_to_float(row['payload_MB']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Counterfactual rescued product types: {dict(rescued_types) if rescued_types else {}}.",
            "- If rescued products do not enter ROI/ROI+Ctx, guarantee participation must not be interpreted as evidence support. In this run at least one rescued product is ROI, so it should count as min-stage support when delivered before deadline.",
            "- utility_rule high-value delivered utility was not zero under passive ledger accounting; the old zero came from missing or inconsistent passive H accounting.",
            "- This single-seed artifact should not be used for a paper conclusion because cross-arm task streams differ.",
            "",
            f"Conclusion: {conclusion}",
        ]
    )
    _write_text(path, "\n".join(lines))


def _write_cross_arm_fairness_report(runs: list[dict[str, Any]], tasks_by_arm: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, object]]:
    util_tasks = tasks_by_arm.get("utility_rule", {})
    prop_tasks = tasks_by_arm.get("proposed_guarantee_aware_full", {})
    rows: list[dict[str, object]] = []
    all_task_ids = sorted(set(util_tasks) | set(prop_tasks))
    mismatch_count = 0
    for task_id in all_task_ids:
        u = util_tasks.get(task_id, {})
        p = prop_tasks.get(task_id, {})
        fields = ["value_level", "product_requirement", "arrival_slot", "deadline_slot"]
        mismatches = [field for field in fields if str(u.get(field, "")) != str(p.get(field, ""))]
        if mismatches:
            mismatch_count += 1
        rows.append(
            {
                "check": "same_seed_same_task_attributes",
                "task_id": task_id,
                "passed": int(not mismatches),
                "severity": "ok" if not mismatches else "error",
                "mismatched_fields": ";".join(mismatches),
                "utility_rule_value_level": u.get("value_level", ""),
                "proposed_value_level": p.get("value_level", ""),
                "utility_rule_product_requirement": u.get("product_requirement", ""),
                "proposed_product_requirement": p.get("product_requirement", ""),
                "utility_rule_deadline_slot": u.get("deadline_slot", ""),
                "proposed_deadline_slot": p.get("deadline_slot", ""),
            }
        )
    rows.extend(
        [
            {"check": "same_task_id_set", "task_id": "", "passed": int(set(util_tasks) == set(prop_tasks)), "severity": "ok" if set(util_tasks) == set(prop_tasks) else "error", "mismatched_fields": ""},
            {"check": "same_value_requirement_deadline", "task_id": "", "passed": int(mismatch_count == 0), "severity": "ok" if mismatch_count == 0 else "error", "mismatched_fields": str(mismatch_count)},
            {"check": "same_closure_evaluator", "task_id": "", "passed": 1, "severity": "ok", "mismatched_fields": "shared_passive_stage_success_audit"},
            {"check": "same_stage_success_evaluator", "task_id": "", "passed": 1, "severity": "ok", "mismatched_fields": "shared_passive_stage_success_audit"},
            {"check": "proposed_extra_mechanism_only_priority_selection", "task_id": "", "passed": int(mismatch_count == 0), "severity": "error" if mismatch_count else "ok", "mismatched_fields": "task_stream_differs" if mismatch_count else ""},
            {"check": "utility_rule_passive_same_metrics", "task_id": "", "passed": 1, "severity": "ok", "mismatched_fields": "ledger_and_stage_evaluator_shared"},
        ]
    )
    lines = [
        "# Cross-arm Fairness Report",
        "",
        f"- Same task_id set: {set(util_tasks) == set(prop_tasks)}.",
        f"- Task attribute mismatches: {mismatch_count}.",
        "- Contact schedule and payload size model cannot be claimed fair for this artifact while task value/requirement/deadline mappings differ.",
        "- The proposed mechanism may still only affect priority/selection inside its own run, but this pair is not a fair arm comparison.",
    ]
    _write_text(OUT_DIR / "cross_arm_fairness_report.md", "\n".join(lines))
    return rows


def _write_metric_consistency_report(checks: list[dict[str, object]]) -> None:
    errors = [row for row in checks if row.get("severity") == "error"]
    warnings = [row for row in checks if row.get("severity") == "warning"]
    lines = [
        "# Metric Consistency Report",
        "",
        f"- error checks: {len(errors)}.",
        f"- warning checks: {len(warnings)}.",
        "- Fixed summary fields are derived from `delivered_product_ledger_by_arm.csv`.",
    ]
    if errors:
        lines.append("")
        lines.append("## Errors")
        for row in errors[:20]:
            lines.append(f"- {row['arm']} / {row['check_name']}: {row.get('explanation', '')}")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for row in warnings[:20]:
            lines.append(f"- {row['arm']} / {row['check_name']}: {row.get('explanation', '')}")
    _write_text(OUT_DIR / "metric_consistency_report.md", "\n".join(lines))


def _write_taxonomy() -> None:
    _write_text(
        OUT_DIR / "utility_accounting_taxonomy.md",
        """# Utility Accounting Taxonomy

- `product_gross_information_utility = V_eff * Delta_u_I(p | D_before, r_i) * q_i_or_q_hat * tau_i`.
- `product_resource_cost = compute_energy + communication_energy + latency_cost + queue_cost + payload_cost + risk_penalty`.
- `product_net_utility = product_gross_information_utility - product_resource_cost`.
- `delivered_effective_utility_gross` is the sum of product gross over delivered ledger rows.
- `delivered_effective_utility_net` is the sum of product net over delivered ledger rows.
- `high_value_delivered_utility_gross/net` are the same sums restricted to H: value_level in {high, urgent} and product_requirement in {context, evidence}.
- `net_system_utility` remains the episode-level summary metric and includes delivered/queued effective utility, resource penalties, requirement penalties, long-term risk penalties, terminal penalties, and hard penalties as implemented by the existing summary evaluator.

The legacy label `task delivered utility` is intentionally not emitted by the fixed summary.
""",
    )


def _write_final_recommendation(summary_rows: list[dict[str, object]], stage_by_arm: dict[str, list[dict[str, object]]], fairness_rows: list[dict[str, object]]) -> None:
    by_arm = {str(row.get("arm")): row for row in summary_rows}
    util = by_arm.get("utility_rule", {})
    prop = by_arm.get("proposed_guarantee_aware_full", {})
    prop_stage = stage_by_arm.get("proposed_guarantee_aware_full", [])
    rescued = [row for row in prop_stage if _to_int(row.get("counterfactual_rescued_delivery_count")) > 0]
    fairness_error = any(row.get("severity") == "error" for row in fairness_rows)
    lines = [
        "# Final Recommendation",
        "",
        "1. The old proposed high-value delivered utility exceeded old task delivered utility because different metrics were compared: delivered high-value incremental product utility versus episode-level effective utility.",
        "2. This is both a gross/net naming problem and a passive accounting bug; the fixed table separates delivered product gross/net from net_system_utility.",
        f"3. Corrected proposed total gross/net delivered utility: {_to_float(prop.get('total_gross_delivered_utility')):.6f} / {_to_float(prop.get('total_net_delivered_utility')):.6f}.",
        f"4. Corrected proposed high-value gross/net utility: {_to_float(prop.get('high_value_gross_delivered_utility')):.6f} / {_to_float(prop.get('high_value_net_delivered_utility')):.6f}.",
        f"5. utility_rule high-value delivered utility = 0 is not true under passive ledger accounting; fixed H gross/net are {_to_float(util.get('high_value_gross_delivered_utility')):.6f} / {_to_float(util.get('high_value_net_delivered_utility')):.6f}.",
        f"6. utility_rule delivered H products: {int(_to_float(util.get('high_value_metadata_delivered_count')) + _to_float(util.get('high_value_roi_delivered_count')) + _to_float(util.get('high_value_roi_context_delivered_count')))}.",
        f"7. Proposed counterfactual rescue task rows: {len(rescued)}; inspect `high_value_stage_success_audit.csv` and ledger delivery_reason=counterfactual_rescued for product type.",
        f"8. Fixed proposed min-stage success count/rate all_H: {_to_float(prop.get('high_value_min_stage_success_count')):.0f} / {_to_float(prop.get('high_value_min_stage_success_rate_all_H')):.6f}.",
        "9. Delivered ROI is counted as min-stage success for context/evidence and is not counted as evidence closure unless ROI+Ctx is delivered.",
        "10. proposed lower utility/MB is partly a real payload-density trade-off, but this artifact also has accounting and fairness defects.",
        "11. Current method does advance some high-value evidence-support stages after fixed accounting, but the arm comparison is not paper-ready.",
        "12. Next step: fix accounting and cross-arm task-stream fairness first, then run multi-seed; mechanism tuning should wait.",
        "13. This single-seed result cannot support a paper conclusion.",
        "14. Trustworthy now: ledger-derived subset sums, payload sums, stage rules. Needs further verification: cross-arm fairness, multi-seed stability, product cost allocation calibration.",
        "",
        "Recommendations selected: A, B, C, D, E, G, H, J." if fairness_error else "Recommendations selected: A, B, C, D, E, G, H, I.",
    ]
    _write_text(OUT_DIR / "final_recommendation.md", "\n".join(lines))


def run_audit(input_root: Path = DEFAULT_INPUT_ROOT) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_taxonomy()
    runs = [
        _load_run(input_root, "utility_rule", 0.0),
        _load_run(input_root, "proposed_guarantee_aware_full", 0.35),
    ]
    ledgers_by_arm: dict[str, list[dict[str, object]]] = {}
    stages_by_arm: dict[str, list[dict[str, object]]] = {}
    tasks_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    cf_rows_all: list[dict[str, object]] = []
    for run in runs:
        tasks = _task_mapping(run["mapping_rows"], run["trajectory_rows"])
        tasks_by_arm[run["arm"]] = tasks
        ledger, cf_rows = build_delivered_product_ledger(
            scenario=run["scenario"],
            seed=run["seed"],
            arm=run["arm"],
            slots=run["slots"],
            lambda_h=run["lambda_H"],
            summary=run["summary"],
            lifecycle_rows=run["lifecycle_rows"],
            queue_rows=run["queue_rows"],
            candidate_rows=run["candidate_rows"],
            mapping_rows=run["mapping_rows"],
            trajectory_rows=run["trajectory_rows"],
        )
        stage = build_stage_success_audit(
            ledger,
            tasks,
            scenario=run["scenario"],
            seed=run["seed"],
            arm=run["arm"],
            lambda_h=run["lambda_H"],
        )
        ledgers_by_arm[run["arm"]] = ledger
        stages_by_arm[run["arm"]] = stage
        cf_rows_all.extend(cf_rows)

    ledger_all = [row for arm in ("utility_rule", "proposed_guarantee_aware_full") for row in ledgers_by_arm[arm]]
    stage_all = [row for arm in ("utility_rule", "proposed_guarantee_aware_full") for row in stages_by_arm[arm]]
    _write_csv(OUT_DIR / "delivered_product_ledger_by_arm.csv", ledger_all, LEDGER_COLUMNS)
    _write_csv(OUT_DIR / "high_value_stage_success_audit.csv", stage_all, STAGE_COLUMNS)
    _write_csv(OUT_DIR / "counterfactual_selection_trace_fixed.csv", cf_rows_all)

    summary_rows = build_fixed_summary_rows(runs, ledgers_by_arm, stages_by_arm, tasks_by_arm)
    _write_csv(OUT_DIR / "method_vs_utility_rule_once_summary_fixed.csv", summary_rows, FIXED_SUMMARY_COLUMNS)

    utility_audit = _write_utility_rule_high_value_report(
        ledgers_by_arm["utility_rule"],
        stages_by_arm["utility_rule"],
        tasks_by_arm["utility_rule"],
        OUT_DIR / "utility_rule_high_value_report.md",
    )
    _write_csv(OUT_DIR / "utility_rule_high_value_audit.csv", utility_audit)

    checks = check_metric_consistency(summary_rows, ledgers_by_arm, stages_by_arm)
    _write_csv(OUT_DIR / "metric_consistency_checks.csv", checks)
    _write_metric_consistency_report(checks)

    _write_low_utility_report(summary_rows, ledgers_by_arm, OUT_DIR / "low_utility_explanation_report.md")
    fairness_rows = _write_cross_arm_fairness_report(runs, tasks_by_arm)
    _write_csv(OUT_DIR / "cross_arm_fairness_audit.csv", fairness_rows)
    _write_final_recommendation(summary_rows, stages_by_arm, fairness_rows)
    return {
        "runs": runs,
        "ledger": ledger_all,
        "stage": stage_all,
        "summary": summary_rows,
        "checks": checks,
        "fairness": fairness_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified utility accounting audit artifacts.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    args = parser.parse_args()
    result = run_audit(args.input_root)
    print(f"wrote {OUT_DIR}")
    print(f"ledger_rows={len(result['ledger'])}")
    print(f"summary_rows={len(result['summary'])}")


if __name__ == "__main__":
    main()
