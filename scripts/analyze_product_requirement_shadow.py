"""Read-only product-requirement shadow diagnostics.

This script consumes existing paired task traces and candidate-action logs. It
does not run simulations and does not change policy, reward, queue priority,
profiles, deadlines, resource/contact settings, or delivered-only metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "_product_requirement_shadow_audit"
UTIL_AUDIT_DIR = PROJECT_ROOT / "outputs" / "_util_vs_proposed_audit"
READINESS_DIR = PROJECT_ROOT / "outputs" / "_formal_readiness_gate_lite"

PROPOSED_FULL = "proposed_payload_repaired_combined"
PROPOSED_SHORT = "PROP-COMB"
UTIL_SHORT = "UTIL"
SCENARIOS = ("stress", "nominal")
ALLOWED_REQUIREMENTS = ("alert", "verification", "context", "evidence", "ambiguous")
PAYLOAD_TYPES = ("metadata", "roi", "roi_context", "defer/store", "other")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        if not finite(value):
            return "NA"
        return f"{value:.{digits}g}"
    return str(value)


def pct(value: float) -> str:
    if not finite(value):
        return "NA"
    return f"{100.0 * value:.1f}%"


def mean(values: Iterable[float]) -> float:
    clean = [value for value in values if finite(value)]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return math.nan
    return num / den


def bool_field(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def quantiles(values: Iterable[float]) -> tuple[float, float]:
    clean = sorted(value for value in values if finite(value))
    if not clean:
        return math.nan, math.nan
    if len(clean) == 1:
        return clean[0], clean[0]
    q1_idx = int(math.floor((len(clean) - 1) / 3))
    q2_idx = int(math.floor(2 * (len(clean) - 1) / 3))
    return clean[q1_idx], clean[q2_idx]


def tertile(value: float, q1: float, q2: float, labels: tuple[str, str, str]) -> str:
    if not finite(value):
        return "unknown"
    if not finite(q1) or not finite(q2) or q1 == q2:
        return labels[1]
    if value <= q1:
        return labels[0]
    if value <= q2:
        return labels[1]
    return labels[2]


def deadline_bin(seconds: float) -> str:
    if not finite(seconds):
        return "unknown"
    if seconds <= 1800.0:
        return "urgent"
    if seconds <= 7200.0:
        return "medium"
    return "loose"


def payload_type(payload: str, action: str = "", model: str = "") -> str:
    text = f"{payload} {action} {model}".lower()
    if "delay" in text or "defer" in text or "store" in text:
        return "defer/store"
    if "context" in text:
        return "roi_context"
    if "roi" in text:
        return "roi"
    if "metadata" in text:
        return "metadata"
    return "other"


def outcome_is_delivered(row: dict[str, Any], prefix: str = "proposed") -> bool:
    text = f"{row.get(f'lifetime_status_{prefix}', '')} {row.get(f'delivered_status_{prefix}', '')}".lower()
    return "delivered" in text


def outcome_is_expired(row: dict[str, Any], prefix: str = "proposed") -> bool:
    text = f"{row.get(f'lifetime_status_{prefix}', '')} {row.get(f'delivered_status_{prefix}', '')}".lower()
    return "expired" in text


def scenario_root(scenario: str) -> Path:
    return READINESS_DIR / f"logging_smoke_{scenario}"


def trajectory_index(scenario: str) -> dict[str, dict[str, str]]:
    rows = read_csv(scenario_root(scenario) / "trajectories" / f"{PROPOSED_FULL}.csv")
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id and to_int(row.get("has_task")) == 1 and task_id not in indexed:
            indexed[task_id] = row
    return indexed


def candidate_rows_by_task(scenario: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    path = READINESS_DIR / f"candidate_action_score_log_{scenario}.csv"
    for row in read_csv(path):
        if str(row.get("variant", "")) != PROPOSED_FULL:
            continue
        task_id = str(row.get("task_id", ""))
        if task_id:
            grouped[task_id].append(row)
    return grouped


def best_candidate_by_payload(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        kind = payload_type(str(row.get("candidate_payload", "")), str(row.get("candidate_action", "")))
        old = best.get(kind)
        if old is None or to_float(row.get("final_candidate_score")) > to_float(old.get("final_candidate_score")):
            best[kind] = row
    return best


def candidate_payload_available(rows: list[dict[str, str]], kind: str) -> bool:
    for row in rows:
        if payload_type(str(row.get("candidate_payload", "")), str(row.get("candidate_action", ""))) != kind:
            continue
        if to_int(row.get("considered_by_router")) == 1 and str(row.get("safety_status", "")).lower() == "safe":
            return True
    return False


def field_status(row: dict[str, Any], fields: list[str]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for field in fields:
        value = row.get(field)
        if value is None or value == "" or (isinstance(value, float) and not finite(value)):
            missing.append(field)
        else:
            present.append(field)
    return present, missing


def confidence(base: str, missing: list[str], downgrade_if: bool = False) -> str:
    if missing:
        return "low"
    if downgrade_if and base == "high":
        return "medium"
    return base


def label_shadow_requirement(row: dict[str, Any]) -> dict[str, str]:
    value = to_float(row.get("task_value"))
    uncertainty = to_float(row.get("uncertainty"))
    high_value = to_int(row.get("high_value_flag")) == 1
    emergency = to_int(row.get("emergency_flag")) == 1
    remaining_deadline = to_float(row.get("remaining_deadline_s"), to_float(row.get("deadline_s")))
    contact = to_int(row.get("contact_available")) == 1
    queue_bin = str(row.get("queue_pressure_bin_shadow", "unknown"))
    uncertainty_bin = str(row.get("uncertainty_bin_shadow", "unknown"))
    value_bin = str(row.get("task_value_bin_shadow", "unknown"))
    battery = to_float(row.get("battery_soc"))
    thermal = to_float(row.get("thermal_pressure"))
    pressure = remaining_deadline <= 1800.0 or not contact or queue_bin == "high"
    enough_capacity = (
        remaining_deadline >= 1800.0
        and queue_bin != "high"
        and (not finite(battery) or battery >= 0.65)
        and (not finite(thermal) or thermal <= 0.78)
    )
    strong_contact_capacity = contact and enough_capacity

    evidence_fields = [
        "high_value_flag",
        "emergency_flag",
        "uncertainty",
        "remaining_deadline_s",
        "contact_available",
        "queue_mb",
        "battery_soc",
        "thermal_pressure",
    ]
    context_fields = [
        "high_value_flag",
        "uncertainty",
        "remaining_deadline_s",
        "contact_available",
        "queue_mb",
        "battery_soc",
        "thermal_pressure",
    ]
    alert_pressure_fields = [
        "high_value_flag",
        "emergency_flag",
        "remaining_deadline_s",
        "contact_available",
        "queue_mb",
    ]
    alert_sufficient_fields = ["task_value", "uncertainty", "high_value_flag", "emergency_flag"]
    verification_fields = ["task_value", "uncertainty", "high_value_flag", "emergency_flag"]

    if (emergency or high_value) and uncertainty_bin == "high" and strong_contact_capacity:
        _, missing = field_status(row, evidence_fields)
        return {
            "shadow_product_requirement": "evidence",
            "shadow_rule_id": "PR-EVIDENCE-001",
            "shadow_rule_name": "high_value_or_emergency_high_uncertainty_with_contact_capacity",
            "fields_used_for_label": ";".join(evidence_fields),
            "shadow_label_reason": "High-value or emergency task has high online uncertainty and current contact/capacity conditions are not tight.",
            "shadow_label_confidence": confidence("high" if emergency else "medium", missing),
        }
    if high_value and uncertainty_bin == "high" and enough_capacity:
        _, missing = field_status(row, context_fields)
        return {
            "shadow_product_requirement": "context",
            "shadow_rule_id": "PR-CONTEXT-001",
            "shadow_rule_name": "high_value_high_uncertainty_with_capacity",
            "fields_used_for_label": ";".join(context_fields),
            "shadow_label_reason": "High-value task has high online uncertainty and enough deadline/resource slack for contextual interpretation.",
            "shadow_label_confidence": confidence("medium", missing, downgrade_if=not contact),
        }
    if (emergency or high_value) and pressure:
        _, missing = field_status(row, alert_pressure_fields)
        return {
            "shadow_product_requirement": "alert",
            "shadow_rule_id": "PR-ALERT-001",
            "shadow_rule_name": "priority_task_under_deadline_contact_or_queue_pressure",
            "fields_used_for_label": ";".join(alert_pressure_fields),
            "shadow_label_reason": "Priority task is under tight deadline, no current contact, or high queue pressure; early awareness is the cautious diagnostic product.",
            "shadow_label_confidence": confidence("high" if emergency and remaining_deadline <= 1800.0 else "medium", missing),
        }
    if value_bin == "low" and uncertainty_bin == "low" and not emergency and not high_value:
        _, missing = field_status(row, alert_sufficient_fields)
        return {
            "shadow_product_requirement": "alert",
            "shadow_rule_id": "PR-ALERT-002",
            "shadow_rule_name": "low_value_low_uncertainty_metadata_sufficient_proxy",
            "fields_used_for_label": ";".join(alert_sufficient_fields),
            "shadow_label_reason": "Low task value and low online uncertainty suggest metadata-level awareness may be sufficient.",
            "shadow_label_confidence": confidence("medium", missing),
        }
    if high_value or value_bin in {"medium", "high"} or uncertainty_bin in {"medium", "high"}:
        _, missing = field_status(row, verification_fields)
        return {
            "shadow_product_requirement": "verification",
            "shadow_rule_id": "PR-VERIFY-001",
            "shadow_rule_name": "value_or_uncertainty_requires_roi_verification",
            "fields_used_for_label": ";".join(verification_fields),
            "shadow_label_reason": "Task value or online uncertainty is non-low, but evidence/context conditions are not fully supported by available online fields.",
            "shadow_label_confidence": confidence("medium", missing),
        }
    fields = ["task_value", "uncertainty", "high_value_flag", "emergency_flag", "remaining_deadline_s", "contact_available", "queue_mb"]
    _, missing = field_status(row, fields)
    return {
        "shadow_product_requirement": "ambiguous",
        "shadow_rule_id": "PR-AMBIG-001",
        "shadow_rule_name": "insufficient_or_mixed_online_context",
        "fields_used_for_label": ";".join(field for field in fields if field not in missing),
        "shadow_label_reason": "Available online fields are insufficient or mixed; no confident product-requirement label is assigned.",
        "shadow_label_confidence": "low",
    }


def build_labeled_rows(scenario: str) -> list[dict[str, Any]]:
    paired = read_csv(UTIL_AUDIT_DIR / f"paired_task_trace_{scenario}.csv")
    traj = trajectory_index(scenario)
    values = [to_float(row.get("task_value")) for row in paired]
    queues = [to_float(row.get("queue_mb_proposed")) for row in paired]
    uncertainties = [to_float(traj.get(str(row.get("task_id", "")), {}).get("U_t")) for row in paired]
    q1_value, q2_value = quantiles(values)
    q1_queue, q2_queue = quantiles(queues)
    q1_unc, q2_unc = quantiles(uncertainties)

    labeled: list[dict[str, Any]] = []
    for row in paired:
        task_id = str(row.get("task_id", ""))
        trajectory = traj.get(task_id, {})
        uncertainty = to_float(trajectory.get("U_t"))
        contact_available = to_int(row.get("contact_available"), to_int(to_float(trajectory.get("C_t"), 0.0) > 0.0))
        deadline = to_float(row.get("deadline_s"))
        remaining_deadline = to_float(row.get("remaining_deadline_s"), deadline)
        task_value = to_float(row.get("task_value"))
        queue_mb = to_float(row.get("queue_mb_proposed"))
        battery_soc = to_float(row.get("battery_soc_proposed"), to_float(trajectory.get("battery_soc")))
        thermal_pressure = to_float(row.get("thermal_pressure_proposed"), to_float(trajectory.get("Theta_t")))
        base: dict[str, Any] = {
            "scenario": scenario,
            "seed": row.get("seed", ""),
            "slot": row.get("slot", ""),
            "task_id": task_id,
            "scene_id": row.get("scene_id", ""),
            "task_value": task_value,
            "high_value_flag": to_int(row.get("high_value_flag")),
            "emergency_flag": to_int(row.get("emergency_flag")),
            "deadline_s": deadline,
            "remaining_deadline_s": remaining_deadline,
            "uncertainty": uncertainty if finite(uncertainty) else "",
            "uncertainty_available": int(finite(uncertainty)),
            "contact_available": contact_available,
            "queue_mb": queue_mb,
            "battery_soc": battery_soc,
            "thermal_pressure": thermal_pressure,
            "task_class": row.get("task_class", ""),
            "deadline_bin": deadline_bin(remaining_deadline),
            "queue_pressure_bin_shadow": tertile(queue_mb, q1_queue, q2_queue, ("low", "medium", "high")),
            "task_value_bin_shadow": tertile(task_value, q1_value, q2_value, ("low", "medium", "high")),
            "uncertainty_bin_shadow": tertile(uncertainty, q1_unc, q2_unc, ("low", "medium", "high")),
            "current_selected_payload_proposed": row.get("selected_payload_proposed", ""),
            "current_selected_action_proposed": row.get("selected_action_proposed", ""),
            "current_selected_payload_util": row.get("selected_payload_util", ""),
            "current_selected_action_util": row.get("selected_action_util", ""),
            "payload_type_proposed": payload_type(
                str(row.get("selected_payload_proposed", "")),
                str(row.get("selected_action_proposed", "")),
                str(row.get("selected_model_proposed", "")),
            ),
            "payload_type_util": payload_type(
                str(row.get("selected_payload_util", "")),
                str(row.get("selected_action_util", "")),
                str(row.get("selected_model_util", "")),
            ),
            "delivered_status_proposed": row.get("delivered_status_proposed", ""),
            "lifetime_status_proposed": row.get("lifetime_status_proposed", ""),
            "delivered_only_utility_proposed": to_float(row.get("delivered_only_utility_proposed"), 0.0),
            "high_value_delivered_utility_proposed": to_float(row.get("high_value_delivered_utility_proposed"), 0.0),
            "payload_size_mb_proposed": to_float(row.get("payload_size_mb_proposed"), 0.0),
            "information_sufficiency_proposed": to_float(row.get("information_sufficiency_proposed"), 0.0),
            "detection_quality_proposed": to_float(row.get("detection_quality_proposed"), 0.0),
            "timeliness_proposed": to_float(row.get("timeliness_proposed"), 0.0),
            "delivered_status_util": row.get("delivered_status_util", ""),
            "lifetime_status_util": row.get("lifetime_status_util", ""),
            "delivered_only_utility_util": to_float(row.get("delivered_only_utility_util"), 0.0),
            "payload_size_mb_util": to_float(row.get("payload_size_mb_util"), 0.0),
            "information_sufficiency_util": to_float(row.get("information_sufficiency_util"), 0.0),
        }
        label = label_shadow_requirement(base)
        base.update(label)
        labeled.append(base)
    return labeled


def summarize_consistency(rows: list[dict[str, Any]], scenario: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for req in ALLOWED_REQUIREMENTS:
        subset = [row for row in rows if row.get("shadow_product_requirement") == req]
        total_mb = sum(to_float(row.get("payload_size_mb_proposed"), 0.0) for row in subset)
        out.append(
            {
                "scenario": scenario,
                "shadow_product_requirement": req,
                "tasks": len(subset),
                "task_fraction": safe_div(len(subset), len(rows)),
                "metadata_share": safe_div(sum(1 for row in subset if row.get("payload_type_proposed") == "metadata"), len(subset)),
                "roi_share": safe_div(sum(1 for row in subset if row.get("payload_type_proposed") == "roi"), len(subset)),
                "roi_context_share": safe_div(sum(1 for row in subset if row.get("payload_type_proposed") == "roi_context"), len(subset)),
                "defer_store_share": safe_div(sum(1 for row in subset if row.get("payload_type_proposed") == "defer/store"), len(subset)),
                "delivered_only_utility": sum(to_float(row.get("delivered_only_utility_proposed"), 0.0) for row in subset),
                "high_value_delivered_utility": sum(to_float(row.get("high_value_delivered_utility_proposed"), 0.0) for row in subset),
                "delivered_rate": safe_div(sum(1 for row in subset if outcome_is_delivered(row)), len(subset)),
                "expiration_rate": safe_div(sum(1 for row in subset if outcome_is_expired(row)), len(subset)),
                "average_payload_mb": mean([to_float(row.get("payload_size_mb_proposed")) for row in subset]),
                "average_information_sufficiency": mean([to_float(row.get("information_sufficiency_proposed")) for row in subset]),
                "utility_per_mb": safe_div(
                    sum(to_float(row.get("delivered_only_utility_proposed"), 0.0) for row in subset),
                    total_mb,
                ),
            }
        )
    return out


def distribution_by(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    counts = Counter(str(row.get(field, "unknown")) for row in rows)
    total = len(rows)
    return {key: safe_div(value, total) for key, value in sorted(counts.items())}


def requirement_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        req: safe_div(sum(1 for row in rows if row.get("shadow_product_requirement") == req), len(rows))
        for req in ALLOWED_REQUIREMENTS
    }


def distribution_table(rows: list[dict[str, Any]], field: str) -> list[str]:
    lines = ["| Requirement | " + field + " distribution |", "| --- | --- |"]
    for req in ALLOWED_REQUIREMENTS:
        subset = [row for row in rows if row.get("shadow_product_requirement") == req]
        dist = distribution_by(subset, field) if subset else {}
        text = ", ".join(f"{key}: {pct(value)}" for key, value in dist.items()) or "none"
        lines.append(f"| {req} | {text} |")
    return lines


def sanity_check_section(rows: list[dict[str, Any]]) -> tuple[str, bool]:
    dist = requirement_distribution(rows)
    dominant_req, dominant_frac = max(dist.items(), key=lambda item: -1.0 if not finite(item[1]) else item[1])
    biased = finite(dominant_frac) and dominant_frac >= 0.65
    lines = [
        "## Requirement-Distribution Sanity Checks",
        "",
        "| Requirement | Fraction |",
        "| --- | ---: |",
    ]
    for req in ALLOWED_REQUIREMENTS:
        lines.append(f"| {req} | {pct(dist.get(req, math.nan))} |")
    lines.extend(
        [
            "",
            f"- Dominant requirement: `{dominant_req}` at {pct(dominant_frac)}.",
            f"- Potential rule bias flag: {'yes' if biased else 'no'}; threshold is any single requirement at or above 65%.",
            "",
            "### High-Value Distribution By Requirement",
            "",
        ]
    )
    lines.extend(distribution_table(rows, "high_value_flag"))
    lines.extend(["", "### Emergency Distribution By Requirement", ""])
    lines.extend(distribution_table(rows, "emergency_flag"))
    lines.extend(["", "### Deadline-Bin Distribution By Requirement", ""])
    lines.extend(distribution_table(rows, "deadline_bin"))
    lines.extend(["", "### Contact Availability By Requirement", ""])
    lines.extend(distribution_table(rows, "contact_available"))
    return "\n".join(lines), biased


def consistency_report(stress_rows: list[dict[str, Any]], stress_summary: list[dict[str, Any]]) -> str:
    sanity, _ = sanity_check_section(stress_rows)
    lines = [
        "# Payload-Requirement Consistency Report",
        "",
        "This read-only diagnostic uses shadow labels generated from online/task-context fields only. It does not use SARFish labels, bbox count, target count, object class, or detector-derived ground truth.",
        "",
        "## Stress Consistency Summary",
        "",
        "| Requirement | Tasks | Metadata | ROI | ROI+Context | Defer/Store | Delivered Rate | Expiration Rate | Avg MB | Avg Info | Utility/MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stress_summary:
        lines.append(
            "| {req} | {tasks} | {meta} | {roi} | {ctx} | {defer} | {deliv} | {exp} | {mb} | {info} | {upm} |".format(
                req=row["shadow_product_requirement"],
                tasks=row["tasks"],
                meta=pct(to_float(row.get("metadata_share"))),
                roi=pct(to_float(row.get("roi_share"))),
                ctx=pct(to_float(row.get("roi_context_share"))),
                defer=pct(to_float(row.get("defer_store_share"))),
                deliv=pct(to_float(row.get("delivered_rate"))),
                exp=pct(to_float(row.get("expiration_rate"))),
                mb=fmt(to_float(row.get("average_payload_mb"))),
                info=fmt(to_float(row.get("average_information_sufficiency"))),
                upm=fmt(to_float(row.get("utility_per_mb"))),
            )
        )
    lines.extend(
        [
            "",
            "Qualitative interpretation:",
            "",
            "- Alert rows should often use metadata under tight deadline, poor contact, high queue, or low resource margin. High ROI share in alert rows is a diagnostic mismatch, not proof of lower delivered utility.",
            "- Verification rows are expected to use ROI when capacity allows.",
            "- Context and evidence rows should show some ROI+context when feasible. Near-zero ROI+context share suggests the current one-shot payload action may under-serve richer products.",
            "- Ambiguous rows are intentionally retained when the available online fields do not support a confident label.",
            "",
            sanity,
        ]
    )
    return "\n".join(lines) + "\n"


def roi_overuse_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        if row.get("payload_type_proposed") != "roi":
            continue
        reasons: list[str] = []
        if row.get("shadow_product_requirement") == "alert":
            reasons.append("shadow requirement is alert")
        if row.get("task_value_bin_shadow") == "low" and (
            row.get("deadline_bin") == "urgent" or row.get("queue_pressure_bin_shadow") == "high"
        ):
            reasons.append("low task value with tight deadline or high queue")
        if to_int(row.get("contact_available")) == 0 and row.get("deadline_bin") != "loose":
            reasons.append("no current contact with non-loose deadline")
        if row.get("queue_pressure_bin_shadow") == "high":
            reasons.append("high queue pressure")
        if row.get("uncertainty_bin_shadow") == "low":
            reasons.append("low online uncertainty proxy")
        if outcome_is_expired(row) or (not outcome_is_delivered(row) and to_float(row.get("delivered_only_utility_proposed"), 0.0) <= 0.0):
            reasons.append("ROI selected but delivered utility is zero or task expired")
        if not reasons:
            continue
        cases.append(
            {
                "scenario": row.get("scenario"),
                "seed": row.get("seed"),
                "slot": row.get("slot"),
                "task_id": row.get("task_id"),
                "shadow_product_requirement": row.get("shadow_product_requirement"),
                "shadow_rule_id": row.get("shadow_rule_id"),
                "task_value": row.get("task_value"),
                "high_value_flag": row.get("high_value_flag"),
                "emergency_flag": row.get("emergency_flag"),
                "remaining_deadline_s": row.get("remaining_deadline_s"),
                "uncertainty": row.get("uncertainty"),
                "contact_available": row.get("contact_available"),
                "queue_mb": row.get("queue_mb"),
                "battery_soc": row.get("battery_soc"),
                "thermal_pressure": row.get("thermal_pressure"),
                "selected_action": row.get("current_selected_action_proposed"),
                "selected_payload": row.get("current_selected_payload_proposed"),
                "payload_size_mb": row.get("payload_size_mb_proposed"),
                "delivered_status": row.get("delivered_status_proposed"),
                "lifetime_status": row.get("lifetime_status_proposed"),
                "information_sufficiency": row.get("information_sufficiency_proposed"),
                "utility_outcome": row.get("delivered_only_utility_proposed"),
                "potential_roi_overuse": 1,
                "potential_metadata_first_opportunity": 1,
                "roi_overuse_reason": "; ".join(reasons),
                "why_metadata_first_might_have_been_better": "Potential only: metadata is smaller and may provide earlier alert utility under the listed pressure/outcome conditions.",
            }
        )
    return cases


def roi_overuse_report(cases: list[dict[str, Any]], total_tasks: int) -> str:
    rate = safe_div(len(cases), total_tasks)
    by_req = Counter(str(row.get("shadow_product_requirement")) for row in cases)
    lines = [
        "# ROI Overuse Diagnosis",
        "",
        "This is a cautious counterfactual screen. It flags potential ROI overuse; it does not prove metadata-first would improve delivered utility.",
        "",
        f"- Stress tasks inspected: {total_tasks}.",
        f"- Potential ROI overuse cases: {len(cases)} ({pct(rate)}).",
        f"- Cases by shadow requirement: {dict(by_req)}.",
        "",
        "The main diagnostic signal is whether ROI is selected for alert-like or pressure-constrained tasks where a smaller metadata product could plausibly preserve early awareness.",
    ]
    return "\n".join(lines) + "\n"


def feasible_for_context(row: dict[str, Any]) -> bool:
    return (
        to_int(row.get("contact_available")) == 1
        and to_float(row.get("remaining_deadline_s"), 0.0) >= 1800.0
        and row.get("queue_pressure_bin_shadow") != "high"
        and to_float(row.get("battery_soc"), 1.0) >= 0.65
        and to_float(row.get("thermal_pressure"), 0.0) <= 0.78
    )


def missed_context_evidence_cases(
    rows: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        req = str(row.get("shadow_product_requirement"))
        if req not in {"context", "evidence"}:
            continue
        if not (to_int(row.get("high_value_flag")) == 1 or to_int(row.get("emergency_flag")) == 1):
            continue
        if not feasible_for_context(row):
            continue
        if row.get("payload_type_proposed") == "roi_context":
            continue
        cand_rows = candidates.get(str(row.get("task_id")), [])
        best = best_candidate_by_payload(cand_rows)
        ctx = best.get("roi_context", {})
        selected = best.get(str(row.get("payload_type_proposed")), {})
        context_available = candidate_payload_available(cand_rows, "roi_context")
        cases.append(
            {
                "scenario": row.get("scenario"),
                "seed": row.get("seed"),
                "slot": row.get("slot"),
                "task_id": row.get("task_id"),
                "shadow_product_requirement": req,
                "shadow_rule_id": row.get("shadow_rule_id"),
                "task_value": row.get("task_value"),
                "high_value_flag": row.get("high_value_flag"),
                "emergency_flag": row.get("emergency_flag"),
                "remaining_deadline_s": row.get("remaining_deadline_s"),
                "uncertainty": row.get("uncertainty"),
                "contact_available": row.get("contact_available"),
                "queue_mb": row.get("queue_mb"),
                "selected_action": row.get("current_selected_action_proposed"),
                "selected_payload": row.get("current_selected_payload_proposed"),
                "payload_size_mb": row.get("payload_size_mb_proposed"),
                "delivered_status": row.get("delivered_status_proposed"),
                "lifetime_status": row.get("lifetime_status_proposed"),
                "information_sufficiency": row.get("information_sufficiency_proposed"),
                "utility_outcome": row.get("delivered_only_utility_proposed"),
                "roi_context_candidate_available": int(context_available),
                "best_roi_context_candidate_score": ctx.get("final_candidate_score", ""),
                "best_roi_context_predicted_reward": ctx.get("predicted_reward", ""),
                "best_roi_context_safety_status": ctx.get("safety_status", ""),
                "best_roi_context_rejection_reason": ctx.get("safety_rejection_reason", ""),
                "selected_payload_best_candidate_score": selected.get("final_candidate_score", ""),
                "potential_context_under_service": int(req == "context"),
                "potential_evidence_under_service": int(req == "evidence"),
                "missed_context_evidence_reason": "Context/evidence shadow requirement had feasible current contact/deadline/resource conditions, but Proposed selected metadata/ROI rather than ROI+context.",
            }
        )
    return cases


def context_evidence_report(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    context_rows = [row for row in rows if row.get("shadow_product_requirement") == "context"]
    evidence_rows = [row for row in rows if row.get("shadow_product_requirement") == "evidence"]
    feasible_rows = [row for row in rows if row.get("shadow_product_requirement") in {"context", "evidence"} and feasible_for_context(row)]
    served = [row for row in feasible_rows if row.get("payload_type_proposed") == "roi_context"]
    lines = [
        "# Context/Evidence Diagnosis",
        "",
        "This report flags potential context/evidence under-service. It does not claim ROI+context would definitely improve delivered utility.",
        "",
        f"- Context-labeled stress tasks: {len(context_rows)}.",
        f"- Evidence-labeled stress tasks: {len(evidence_rows)}.",
        f"- Feasible current-contact context/evidence tasks: {len(feasible_rows)}.",
        f"- Feasible current-contact context/evidence tasks served by ROI+context: {len(served)} ({pct(safe_div(len(served), len(feasible_rows)))}).",
        f"- Potential missed context/evidence cases: {len(cases)}.",
        "",
        "Candidate-action scores are included in the case CSV when the ROI+context candidate appears in the log.",
    ]
    return "\n".join(lines) + "\n"


PROGRESSIVE_UTILITY = {
    "metadata": {"alert": 0.80, "verification": 0.35, "context": 0.20, "evidence": 0.15},
    "roi": {"alert": 0.90, "verification": 0.90, "context": 0.55, "evidence": 0.65},
    "roi_context": {"alert": 0.95, "verification": 0.96, "context": 0.92, "evidence": 0.95},
    "defer/store": {"alert": 0.00, "verification": 0.00, "context": 0.00, "evidence": 0.00},
}


def progressive_table_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product, values in PROGRESSIVE_UTILITY.items():
        row = {"product": product}
        row.update({f"u_I_{req}": value for req, value in values.items()})
        rows.append(row)
    return rows


def progressive_design_report() -> str:
    return """# Shadow Progressive Utility Design

This is a diagnostic-only product utility model. It is not used by policy, reward, queue priority, or delivered-only metrics.

Let `r_i` be the shadow product requirement and `D_i(t^-)` be the set of products already delivered for task `i` before the current decision. For a candidate product `p`, define:

`Delta_u_I(p | D_i(t^-), r_i) = u_I(D_i(t^-) union {p}, r_i) - u_I(D_i(t^-), r_i)`

The CSV table defines a cautious monotone utility value for each single product and requirement. A future staged metric can lift this to delivered product sets by taking the best available product utility for the set or by defining explicit set utilities.

Interpretation guardrails:

- Metadata can have high partial utility for alert tasks.
- ROI is the expected verification product.
- ROI+context is the expected context/evidence product when contact/deadline/resource conditions allow.
- These values are shadow design values, not tuned reward coefficients and not proof that a staged policy would improve delivered outcomes.
"""


def staged_recommendation(row: dict[str, Any]) -> tuple[str, str]:
    req = str(row.get("shadow_product_requirement"))
    high_or_emergency = to_int(row.get("high_value_flag")) == 1 or to_int(row.get("emergency_flag")) == 1
    poor_resource = to_float(row.get("battery_soc"), 1.0) < 0.60 or to_float(row.get("thermal_pressure"), 0.0) > 0.85
    no_contact = to_int(row.get("contact_available")) == 0
    tight = str(row.get("deadline_bin")) == "urgent"
    high_queue = str(row.get("queue_pressure_bin_shadow")) == "high"
    low_value_low_unc = row.get("task_value_bin_shadow") == "low" and row.get("uncertainty_bin_shadow") == "low"
    good_contact = to_int(row.get("contact_available")) == 1 and not high_queue and not poor_resource

    if poor_resource and not high_or_emergency:
        return "defer/store", "Very poor resource state and non-priority task."
    if req == "alert":
        if high_or_emergency and (no_contact or tight or high_queue):
            return "metadata-now-ROI-later", "Priority alert under pressure; metadata-now is a potential smaller first product while preserving residual verification value."
        return "metadata-first", "Alert requirement favors early metadata-level awareness."
    if req == "verification":
        if good_contact and not tight:
            return "ROI-direct", "Verification requirement with current contact and manageable queue/resource state."
        return "metadata-now-ROI-later", "Verification requirement under contact/deadline/queue pressure; staged metadata-now then ROI-later is a diagnostic alternative."
    if req in {"context", "evidence"}:
        if good_contact and not tight:
            return "ROI+context-direct", "Context/evidence requirement with current feasible contact/capacity."
        if not poor_resource and not tight:
            return "metadata-now-ROI+context-later", "Context/evidence requirement with residual deadline but limited immediate service conditions."
        return "metadata-first", "Context/evidence requirement under tight pressure; metadata-first is a cautious diagnostic fallback."
    if low_value_low_unc:
        return "metadata-first", "Low-value low-uncertainty proxy."
    return "ROI-direct", "Ambiguous/non-low task where ROI is the conservative verification proxy."


def current_recommendation_equivalent(row: dict[str, Any]) -> str:
    kind = str(row.get("payload_type_proposed"))
    if kind == "metadata":
        return "metadata-first"
    if kind == "roi":
        return "ROI-direct"
    if kind == "roi_context":
        return "ROI+context-direct"
    if kind == "defer/store":
        return "defer/store"
    return "unknown"


def staged_recommendation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rec, reason = staged_recommendation(row)
        current = current_recommendation_equivalent(row)
        proposed_roi = row.get("payload_type_proposed") == "roi"
        proposed_metadata = row.get("payload_type_proposed") == "metadata"
        metadata_first_opp = proposed_roi and rec in {"metadata-first", "metadata-now-ROI-later", "metadata-now-ROI+context-later"}
        context_opp = proposed_roi and rec in {"ROI+context-direct", "metadata-now-ROI+context-later"}
        staged_opp = rec in {"metadata-now-ROI-later", "metadata-now-ROI+context-later"}
        out.append(
            {
                "scenario": row.get("scenario"),
                "seed": row.get("seed"),
                "slot": row.get("slot"),
                "task_id": row.get("task_id"),
                "shadow_product_requirement": row.get("shadow_product_requirement"),
                "shadow_rule_id": row.get("shadow_rule_id"),
                "shadow_label_confidence": row.get("shadow_label_confidence"),
                "task_value": row.get("task_value"),
                "high_value_flag": row.get("high_value_flag"),
                "emergency_flag": row.get("emergency_flag"),
                "remaining_deadline_s": row.get("remaining_deadline_s"),
                "uncertainty": row.get("uncertainty"),
                "contact_available": row.get("contact_available"),
                "queue_mb": row.get("queue_mb"),
                "deadline_bin": row.get("deadline_bin"),
                "queue_pressure_bin_shadow": row.get("queue_pressure_bin_shadow"),
                "current_selected_action_proposed": row.get("current_selected_action_proposed"),
                "current_selected_payload_proposed": row.get("current_selected_payload_proposed"),
                "payload_type_proposed": row.get("payload_type_proposed"),
                "shadow_staged_payload_recommendation": rec,
                "shadow_recommendation_reason": reason,
                "current_proposed_equivalent_recommendation": current,
                "proposed_equals_shadow_recommendation": int(current == rec),
                "potential_metadata_first_opportunity": int(metadata_first_opp),
                "potential_roi_overuse": int(proposed_roi and (metadata_first_opp or row.get("shadow_product_requirement") == "alert")),
                "potential_context_under_service": int(row.get("shadow_product_requirement") == "context" and rec in {"ROI+context-direct", "metadata-now-ROI+context-later"} and row.get("payload_type_proposed") != "roi_context"),
                "potential_evidence_under_service": int(row.get("shadow_product_requirement") == "evidence" and rec in {"ROI+context-direct", "metadata-now-ROI+context-later"} and row.get("payload_type_proposed") != "roi_context"),
                "potential_staged_delivery_opportunity": int(staged_opp),
                "proposed_roi_shadow_suggests_metadata_first": int(metadata_first_opp),
                "proposed_roi_shadow_suggests_context_or_staged_context": int(context_opp),
                "proposed_metadata_shadow_suggests_roi": int(proposed_metadata and rec == "ROI-direct"),
                "lifetime_status_proposed": row.get("lifetime_status_proposed"),
                "delivered_only_utility_proposed": row.get("delivered_only_utility_proposed"),
            }
        )
    return out


def fraction(rows: list[dict[str, Any]], field: str) -> float:
    return safe_div(sum(1 for row in rows if to_int(row.get(field)) == 1), len(rows))


def staged_report(recs: list[dict[str, Any]]) -> str:
    high_value = [row for row in recs if to_int(row.get("high_value_flag")) == 1]
    expired_or_low = [
        row
        for row in recs
        if "expired" in str(row.get("lifetime_status_proposed", "")).lower()
        or to_float(row.get("delivered_only_utility_proposed"), 0.0) <= 0.0
    ]
    lines = [
        "# Staged Payload Opportunity Report",
        "",
        "This counterfactual analysis is diagnostic only. It estimates potential staged-delivery opportunities from available context rules and does not rerun the simulator.",
        "",
        f"- Proposed equals shadow recommendation: {pct(fraction(recs, 'proposed_equals_shadow_recommendation'))}.",
        f"- Proposed selected ROI but shadow suggests metadata-first/staged metadata-first: {pct(fraction(recs, 'proposed_roi_shadow_suggests_metadata_first'))}.",
        f"- Proposed selected ROI but shadow suggests ROI+context or staged context: {pct(fraction(recs, 'proposed_roi_shadow_suggests_context_or_staged_context'))}.",
        f"- Proposed selected metadata but shadow suggests ROI: {pct(fraction(recs, 'proposed_metadata_shadow_suggests_roi'))}.",
        f"- High-value subset potential metadata-first opportunity: {pct(fraction(high_value, 'potential_metadata_first_opportunity'))}.",
        f"- High-value subset potential staged-delivery opportunity: {pct(fraction(high_value, 'potential_staged_delivery_opportunity'))}.",
        f"- Expired/zero-utility subset potential metadata-first opportunity: {pct(fraction(expired_or_low, 'potential_metadata_first_opportunity'))}.",
        f"- Expired/zero-utility subset potential staged-delivery opportunity: {pct(fraction(expired_or_low, 'potential_staged_delivery_opportunity'))}.",
        "",
        "Use the cautious `potential_*` fields in the CSV as screening signals only.",
    ]
    return "\n".join(lines) + "\n"


def severity_from_metrics(rows: list[dict[str, Any]], roi_cases: list[dict[str, Any]], missed_cases: list[dict[str, Any]], recs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    roi_overuse_rate = safe_div(len(roi_cases), total)
    metadata_first_rate = fraction(recs, "potential_metadata_first_opportunity")
    context_under_rate = fraction(recs, "potential_context_under_service")
    evidence_under_rate = fraction(recs, "potential_evidence_under_service")
    staged_rate = fraction(recs, "potential_staged_delivery_opportunity")
    alert_rows = [row for row in rows if row.get("shadow_product_requirement") == "alert"]
    alert_roi_share = safe_div(sum(1 for row in alert_rows if row.get("payload_type_proposed") == "roi"), len(alert_rows))
    feasible_ce = [row for row in rows if row.get("shadow_product_requirement") in {"context", "evidence"} and feasible_for_context(row)]
    feasible_ce_served = [row for row in feasible_ce if row.get("payload_type_proposed") == "roi_context"]
    feasible_ce_context_share = safe_div(len(feasible_ce_served), len(feasible_ce))
    max_rate = max(
        value
        for value in [roi_overuse_rate, metadata_first_rate, context_under_rate, evidence_under_rate, staged_rate]
        if finite(value)
    )
    critical_context_starvation = len(feasible_ce) >= 20 and finite(feasible_ce_context_share) and feasible_ce_context_share <= 0.02
    if max_rate > 0.50 or critical_context_starvation:
        severity = "CRITICAL"
        category = "E. Critical issue"
    elif max_rate >= 0.30 or (finite(alert_roi_share) and alert_roi_share >= 0.50):
        severity = "MAJOR"
        category = "D. Major issue"
    elif max_rate >= 0.15:
        severity = "MODERATE"
        category = "C. Moderate issue"
    elif max_rate > 0.0:
        severity = "LOW"
        category = "B. Mild issue"
    else:
        severity = "LOW"
        category = "A. No issue"
    return {
        "severity": severity,
        "category": category,
        "roi_overuse_rate": roi_overuse_rate,
        "metadata_first_opportunity_rate": metadata_first_rate,
        "context_under_service_rate": context_under_rate,
        "evidence_under_service_rate": evidence_under_rate,
        "staged_delivery_opportunity_rate": staged_rate,
        "missed_context_evidence_case_rate": safe_div(len(missed_cases), total),
        "alert_roi_share": alert_roi_share,
        "feasible_context_evidence_tasks": len(feasible_ce),
        "feasible_context_evidence_roi_context_share": feasible_ce_context_share,
    }


def formula_method_report(metrics: dict[str, Any]) -> str:
    severity = str(metrics["severity"])
    lines = [
        "# Formula/Method Implication Report",
        "",
        "The answers below separate observed evidence from cautious diagnostic interpretation.",
        "",
        "1. Is the current one-shot payload action model too rigid?",
        "",
        f"Diagnostic indication: {severity}. The one-shot model appears rigid when potential staged-delivery opportunity is {pct(metrics['staged_delivery_opportunity_rate'])}. This is not proof of utility improvement without a shadow or real delivered-outcome metric.",
        "",
        "2. Is ROI acting as a universal safe harbor?",
        "",
        f"ROI is a potential safe harbor if alert ROI share is high. Stress alert ROI share is {pct(metrics['alert_roi_share'])}.",
        "",
        "3. Is metadata under-valued for alert tasks?",
        "",
        f"Potentially, if metadata-first opportunity rate is treated as a screening metric: {pct(metrics['metadata_first_opportunity_rate'])}. This does not prove metadata would improve delivered utility.",
        "",
        "4. Is ROI+context over-penalized or under-represented?",
        "",
        f"Potentially. Feasible current-contact context/evidence tasks served by ROI+context: {pct(metrics['feasible_context_evidence_roi_context_share'])} over {metrics['feasible_context_evidence_tasks']} tasks.",
        "",
        "5. Does missing product requirement r_i explain the high same-action fraction between Proposed and UTIL?",
        "",
        "It is a plausible explanation, not a definitive one. Without `r_i`, both policies can rank ROI similarly because they see the same broad utility/resource tradeoff rather than different product semantics.",
        "",
        "6. Would adding product requirement r_i likely improve semantic differentiation without changing the main framework?",
        "",
        "Yes as a lightweight utility-model revision or shadow metric: `r_i` can alter information sufficiency by product requirement while leaving the co-design framework intact.",
        "",
        "7. Would progressive delivery likely be more important than immediately adding DDQN?",
        "",
        "For this diagnostic, yes. The missing product semantics are closer to the payload utility definition than to DDQN mode modulation. DDQN should remain future work.",
        "",
        "8. How should the paper describe the current limitation honestly?",
        "",
        "State that current payload actions are one-shot products and that metadata/ROI/ROI+context are not yet conditioned on explicit product-requirement labels. Avoid presenting metadata utility as verification or evidence utility.",
        "",
        "9. What is the minimal next policy change if diagnostics confirm the issue?",
        "",
        "Implement a lightweight product-requirement-aware payload utility or information sufficiency table before formal multi-seed if severity is MAJOR or CRITICAL.",
        "",
        "10. What should remain future work for TWC?",
        "",
        "Full progressive multi-stage delivery, learned residual product scheduling, and DDQN-based dynamic modulation should remain TWC/future work unless the current venue requires that full scope.",
    ]
    return "\n".join(lines) + "\n"


def final_report(metrics: dict[str, Any], biased_rules: bool) -> str:
    severity = str(metrics["severity"])
    category = str(metrics["category"])
    major_or_worse = severity in {"MAJOR", "CRITICAL"}
    critical = severity == "CRITICAL"
    lines = [
        "# Product-Requirement Shadow Final Report",
        "",
        f"Final classification: **{category}**.",
        "",
        "Severity thresholds are diagnostics only: LOW is below 15%, MODERATE is 15-30%, MAJOR is 30-50% or ROI dominance in alert tasks, and CRITICAL is above 50% or near-zero context/evidence service despite feasible current conditions.",
        "",
        "## Evidence",
        "",
        f"- Potential ROI overuse rate: {pct(metrics['roi_overuse_rate'])}.",
        f"- Potential metadata-first opportunity rate: {pct(metrics['metadata_first_opportunity_rate'])}.",
        f"- Potential context under-service rate: {pct(metrics['context_under_service_rate'])}.",
        f"- Potential evidence under-service rate: {pct(metrics['evidence_under_service_rate'])}.",
        f"- Potential staged-delivery opportunity rate: {pct(metrics['staged_delivery_opportunity_rate'])}.",
        f"- Alert-task ROI share: {pct(metrics['alert_roi_share'])}.",
        f"- Requirement-rule potential bias flag: {'yes' if biased_rules else 'no'}.",
        "",
        "## Actionability Guardrail",
        "",
        "- Diagnostic-only issue: all shadow labels and counterfactual flags are screening metrics, not causal proof.",
        "- Paper wording limitation: describe metadata as alert-level utility and ROI as verification-oriented utility; do not imply all delivered utility is semantically equivalent.",
        f"- Lightweight utility-model revision needed: {'yes' if major_or_worse else 'not required before formal multi-seed by this diagnostic'}.",
        f"- Full progressive delivery needed: {'yes before using results as paper evidence' if critical else 'no before formal multi-seed; keep as future/TWC scope unless later diagnostics become critical'}.",
        "- DDQN-related future work: do not implement DDQN in this step; keep it as future dynamic mode-modulation work.",
        "",
        "## Go/No-Go",
        "",
        f"- Should formal multi-seed proceed immediately? {'No' if major_or_worse else 'Yes, with diagnostic wording and shadow metric retained'}.",
        f"- Should product-aware utility be implemented before formal multi-seed? {'Yes' if major_or_worse else 'No, optional shadow metric is enough'}.",
        f"- Should full progressive delivery be implemented before formal multi-seed? {'Yes' if critical else 'No'}.",
        "- Should DDQN be implemented before formal multi-seed? No.",
        "- What is the minimum safe change if the mismatch is MAJOR? Add a lightweight product-requirement-aware payload utility/information-sufficiency table and rerun a smoke check before formal multi-seed.",
        "- What should be left for TWC? "
        + (
            "Learned residual product scheduling, broader multi-orbit staged-product validation, and DDQN dynamic modulation."
            if critical
            else "Full progressive delivery, residual product scheduling, learned staged-product policy, and DDQN dynamic modulation."
        ),
        "",
        "## Recommended Next Action",
        "",
    ]
    if critical:
        lines.append("Implement full progressive payload delivery before using these results as primary paper evidence. Do not implement DDQN in this step.")
    elif major_or_worse:
        lines.append("Implement the minimal product-aware utility revision before formal multi-seed. Keep full progressive delivery and DDQN for TWC/future work.")
    elif severity == "MODERATE":
        lines.append("Add the product-requirement shadow metric to reporting and proceed to formal multi-seed with careful wording.")
    else:
        lines.append("Proceed to formal multi-seed with paper wording that separates alert, verification, context, and evidence utility.")
    return "\n".join(lines) + "\n"


def labeling_rules_report() -> str:
    return """# Shadow Labeling Rules

The shadow product-requirement labels are diagnostic only. Allowed labels are `alert`, `verification`, `context`, `evidence`, and `ambiguous`.

No label rule uses SARFish ground-truth labels, bbox count, target count, object class, or offline detector-derived ground-truth information. If a field is unavailable, the row reports it as unavailable and confidence is downgraded or the label is `ambiguous`.

| Rule ID | Rule Name | Label | Required Online/Context Fields | Cautious Interpretation |
| --- | --- | --- | --- | --- |
| PR-EVIDENCE-001 | high_value_or_emergency_high_uncertainty_with_contact_capacity | evidence | high_value_flag; emergency_flag; uncertainty; remaining_deadline_s; contact_available; queue_mb; battery_soc; thermal_pressure | Richer evidence may be needed only when priority, uncertainty, and current capacity are jointly visible. |
| PR-CONTEXT-001 | high_value_high_uncertainty_with_capacity | context | high_value_flag; uncertainty; remaining_deadline_s; contact_available; queue_mb; battery_soc; thermal_pressure | Context may help interpretation for high-value, high-uncertainty tasks with capacity. |
| PR-ALERT-001 | priority_task_under_deadline_contact_or_queue_pressure | alert | high_value_flag; emergency_flag; remaining_deadline_s; contact_available; queue_mb | Early awareness is the cautious product when priority tasks face deadline/contact/queue pressure. |
| PR-ALERT-002 | low_value_low_uncertainty_metadata_sufficient_proxy | alert | task_value; uncertainty; high_value_flag; emergency_flag | Metadata may be enough for low-value, low-uncertainty tasks. |
| PR-VERIFY-001 | value_or_uncertainty_requires_roi_verification | verification | task_value; uncertainty; high_value_flag; emergency_flag | ROI-level verification is the conservative default for non-low value or uncertainty when context/evidence is not supported. |
| PR-AMBIG-001 | insufficient_or_mixed_online_context | ambiguous | available subset of task_value; uncertainty; flags; deadline; contact; queue | No confident requirement label is assigned. |
"""


def terminal_summary(
    files: list[Path],
    stress_rows: list[dict[str, Any]],
    nominal_rows: list[dict[str, Any]],
    stress_summary: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> str:
    dist = requirement_distribution(stress_rows)
    payload_by_req = {
        row["shadow_product_requirement"]: {
            "metadata": row["metadata_share"],
            "roi": row["roi_share"],
            "roi_context": row["roi_context_share"],
            "defer_store": row["defer_store_share"],
        }
        for row in stress_summary
    }
    file_lines = "\n".join(f"- {path.relative_to(PROJECT_ROOT).as_posix()}" for path in files)
    return "\n".join(
        [
            "Product-requirement shadow audit complete.",
            "",
            "Files created:",
            file_lines,
            "",
            f"Stress and nominal labeling both possible: {'yes' if stress_rows and nominal_rows else 'no'}.",
            f"Number of shadow-labeled tasks: stress={len(stress_rows)}, nominal={len(nominal_rows)}.",
            f"Requirement distribution (stress): { {key: round(value, 4) for key, value in dist.items()} }.",
            f"Proposed payload distribution by requirement (stress): {payload_by_req}.",
            f"ROI overuse rate: {pct(metrics['roi_overuse_rate'])}.",
            f"Missed metadata-first opportunity rate: {pct(metrics['metadata_first_opportunity_rate'])}.",
            f"Missed ROI+context/evidence opportunity rate: {pct(metrics['context_under_service_rate'] + metrics['evidence_under_service_rate'])}.",
            f"Staged-payload opportunity rate: {pct(metrics['staged_delivery_opportunity_rate'])}.",
            f"Final severity classification: {metrics['category']} ({metrics['severity']}).",
            "Recommended next action: "
            + (
                "implement the minimal product-aware utility revision before formal multi-seed"
                if metrics["severity"] == "MAJOR"
                else "implement full progressive delivery before paper-evidence use"
                if metrics["severity"] == "CRITICAL"
                else "proceed with shadow metric and careful wording"
            )
            + ".",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in this audit output directory.")
    args = parser.parse_args()

    stress_rows = build_labeled_rows("stress")
    nominal_rows = build_labeled_rows("nominal")
    stress_candidates = candidate_rows_by_task("stress")
    stress_summary = summarize_consistency(stress_rows, "stress")
    roi_cases = roi_overuse_cases(stress_rows)
    missed_cases = missed_context_evidence_cases(stress_rows, stress_candidates)
    staged_rows = staged_recommendation_rows(stress_rows)
    metrics = severity_from_metrics(stress_rows, roi_cases, missed_cases, staged_rows)
    _, biased_rules = sanity_check_section(stress_rows)

    files: list[tuple[Path, str, list[dict[str, Any]] | None]] = [
        (OUT_DIR / "shadow_product_requirement_labels_stress.csv", "csv", stress_rows),
        (OUT_DIR / "shadow_product_requirement_labels_nominal.csv", "csv", nominal_rows),
        (OUT_DIR / "payload_requirement_consistency_stress.csv", "csv", stress_summary),
        (OUT_DIR / "roi_overuse_cases_stress.csv", "csv", roi_cases),
        (OUT_DIR / "missed_context_evidence_cases_stress.csv", "csv", missed_cases),
        (OUT_DIR / "shadow_progressive_utility_table.csv", "csv", progressive_table_rows()),
        (OUT_DIR / "shadow_staged_payload_recommendations_stress.csv", "csv", staged_rows),
    ]
    text_files = [
        (OUT_DIR / "shadow_labeling_rules.md", labeling_rules_report()),
        (OUT_DIR / "payload_requirement_consistency_report.md", consistency_report(stress_rows, stress_summary)),
        (OUT_DIR / "roi_overuse_diagnosis.md", roi_overuse_report(roi_cases, len(stress_rows))),
        (OUT_DIR / "context_evidence_diagnosis.md", context_evidence_report(missed_cases, stress_rows)),
        (OUT_DIR / "shadow_progressive_utility_design.md", progressive_design_report()),
        (OUT_DIR / "staged_payload_opportunity_report.md", staged_report(staged_rows)),
        (OUT_DIR / "formula_method_implication_report.md", formula_method_report(metrics)),
        (OUT_DIR / "product_requirement_shadow_final_report.md", final_report(metrics, biased_rules)),
    ]

    for path, _, rows in files:
        write_csv(path, rows or [], overwrite=args.overwrite)
    for path, text in text_files:
        write_text(path, text, overwrite=args.overwrite)

    created = [path for path, _, _ in files] + [path for path, _ in text_files]
    print(terminal_summary(created, stress_rows, nominal_rows, stress_summary, metrics))


if __name__ == "__main__":
    main()
