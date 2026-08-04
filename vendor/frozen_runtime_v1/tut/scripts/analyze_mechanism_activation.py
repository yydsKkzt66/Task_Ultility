"""Read-only mechanism activation and ROI sweet-spot audit.

This script consumes existing stress smoke-test logs only. It does not run
simulations and does not modify router, reward, queue, profile, deadline,
resource/contact, or delivered-only metric definitions.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "_mechanism_activation_audit"
READINESS_DIR = PROJECT_ROOT / "outputs" / "_formal_readiness_gate_lite"
UTIL_AUDIT_DIR = PROJECT_ROOT / "outputs" / "_util_vs_proposed_audit"
STRESS_ROOT = READINESS_DIR / "logging_smoke_stress"
PROPOSED = "proposed_payload_repaired_combined"
UTIL = "utility_rule"
SEED = "2026"
SLOT_SECONDS = 20.0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
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


def write_text(path: Path, text: str) -> None:
    if path.exists():
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


def mean(values: Iterable[float]) -> float:
    clean = [value for value in values if finite(value)]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return math.nan
    return num / den


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return math.nan
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for _, y in pairs))
    if den_x == 0 or den_y == 0:
        return math.nan
    return num / (den_x * den_y)


def quantiles(values: list[float]) -> tuple[float, float]:
    clean = sorted(value for value in values if finite(value))
    if not clean:
        return math.nan, math.nan
    if len(clean) == 1:
        return clean[0], clean[0]
    q1_idx = min(len(clean) - 1, max(0, int(math.floor((len(clean) - 1) / 3))))
    q2_idx = min(len(clean) - 1, max(0, int(math.floor(2 * (len(clean) - 1) / 3))))
    return clean[q1_idx], clean[q2_idx]


def tertile_bin(value: float, q1: float, q2: float, labels: tuple[str, str, str]) -> str:
    if not finite(value):
        return "unknown"
    if not finite(q1) or not finite(q2) or q1 == q2:
        return labels[1]
    if value <= q1:
        return labels[0]
    if value <= q2:
        return labels[1]
    return labels[2]


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


def payload_rank(kind: str) -> int:
    return {"defer/store": 0, "metadata": 1, "roi": 2, "roi_context": 3, "other": 1}.get(kind, 1)


def model_rank(model: str, action: str = "") -> int:
    text = f"{model} {action}".lower()
    if "delay" in text or "defer" in text or "store" in text:
        return 0
    if "lightweight" in text:
        return 1
    if "balanced" in text:
        return 2
    if "high_accuracy" in text or "high-accuracy" in text:
        return 3
    return 1


def bool_field(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def action_share(rows: list[dict[str, Any]], field: str, target: str) -> float:
    if not rows:
        return math.nan
    return sum(1 for row in rows if row.get(field) == target) / len(rows)


def load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, dict[str, str]], dict[int, dict[str, Any]]]:
    paired = [dict(row) for row in read_csv(UTIL_AUDIT_DIR / "paired_task_trace_stress.csv")]
    trajectory = read_csv(STRESS_ROOT / "trajectories" / f"{PROPOSED}.csv")
    task_traj: dict[str, dict[str, str]] = {}
    for row in trajectory:
        task_id = str(row.get("task_id", ""))
        if to_int(row.get("has_task")) == 1 and task_id and task_id not in task_traj:
            task_traj[task_id] = row

    slot_state: dict[int, dict[str, Any]] = {}
    for row in trajectory:
        slot = to_int(row.get("slot"), -1)
        if slot < 0:
            continue
        contact = to_float(row.get("C_t"), 0.0) > 0.0
        eclipse = bool_field(row.get("is_eclipse"))
        slot_state[slot] = {"contact": contact, "eclipse": eclipse}

    annotate_temporal_regimes(slot_state)
    return paired, trajectory, task_traj, slot_state


def annotate_temporal_regimes(slot_state: dict[int, dict[str, Any]]) -> None:
    slots = sorted(slot_state)
    for slot in slots:
        state = slot_state[slot]
        contact_run = 0
        probe = slot
        while slot_state.get(probe, {}).get("contact", False):
            contact_run += 1
            probe += 1
        state["remaining_contact_s"] = contact_run * SLOT_SECONDS if state.get("contact") else 0.0
        gap_run = 0
        max_gap = 0
        for probe in range(slot + 1, slot + 91):
            if slot_state.get(probe, {}).get("contact", False):
                gap_run = 0
            else:
                gap_run += 1
                max_gap = max(max_gap, gap_run)
        state["before_long_contact_gap"] = int(max_gap * SLOT_SECONDS >= 600.0)
        in_eclipse = bool(state.get("eclipse"))
        future_eclipse = any(slot_state.get(probe, {}).get("eclipse", False) for probe in range(slot + 1, slot + 61))
        past_eclipse = any(slot_state.get(probe, {}).get("eclipse", False) for probe in range(slot - 60, slot))
        if in_eclipse:
            phase = "in_eclipse"
        elif future_eclipse:
            phase = "before_eclipse"
        elif past_eclipse:
            phase = "after_eclipse"
        else:
            phase = "sunlit_or_unknown"
        state["eclipse_phase"] = phase


def enrich_paired(
    paired: list[dict[str, Any]],
    task_traj: dict[str, dict[str, str]],
    slot_state: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    queue_values = [to_float(row.get("queue_mb_proposed")) for row in paired]
    battery_values = [to_float(row.get("battery_soc_proposed")) for row in paired]
    thermal_values = [to_float(row.get("thermal_pressure_proposed")) for row in paired]
    uncertainty_values = [to_float(task_traj.get(str(row.get("task_id")), {}).get("U_t")) for row in paired]
    value_values = [to_float(row.get("task_value")) for row in paired]
    q1_queue, q2_queue = quantiles(queue_values)
    q1_battery, q2_battery = quantiles(battery_values)
    q1_thermal, q2_thermal = quantiles(thermal_values)
    q1_unc, q2_unc = quantiles(uncertainty_values)
    q1_value, q2_value = quantiles(value_values)

    for row in paired:
        task_id = str(row.get("task_id", ""))
        traj = task_traj.get(task_id, {})
        slot = to_int(row.get("slot"), -1)
        state = slot_state.get(slot, {})
        p_kind = payload_type(str(row.get("selected_payload_proposed", "")), str(row.get("selected_action_proposed", "")), str(row.get("selected_model_proposed", "")))
        u_kind = payload_type(str(row.get("selected_payload_util", "")), str(row.get("selected_action_util", "")), str(row.get("selected_model_util", "")))
        row["payload_type_proposed"] = p_kind
        row["payload_type_util"] = u_kind
        row["payload_rank_proposed"] = payload_rank(p_kind)
        row["payload_rank_util"] = payload_rank(u_kind)
        row["model_rank_proposed"] = model_rank(str(row.get("selected_model_proposed", "")), str(row.get("selected_action_proposed", "")))
        row["model_rank_util"] = model_rank(str(row.get("selected_model_util", "")), str(row.get("selected_action_util", "")))
        row["util_heavier_than_proposed"] = int(row["payload_rank_util"] > row["payload_rank_proposed"])
        row["proposed_heavier_than_util"] = int(row["payload_rank_proposed"] > row["payload_rank_util"])
        row["payload_size_delta_proposed_minus_util"] = to_float(row.get("payload_size_mb_proposed")) - to_float(row.get("payload_size_mb_util"))
        row["queue_mb_delta_proposed_minus_util"] = to_float(row.get("queue_mb_proposed")) - to_float(row.get("queue_mb_util"))
        row["U_t"] = to_float(traj.get("U_t"))
        row["C_t"] = to_float(traj.get("C_t"), to_float(row.get("contact_available")))
        row["mode_name"] = traj.get("mode_name", "")
        row["lambda_total"] = sum(to_float(traj.get(name), 0.0) for name in ["lambda_E", "lambda_L", "lambda_Q", "lambda_S"])
        row["lambda_E"] = to_float(traj.get("lambda_E"))
        row["lambda_L"] = to_float(traj.get("lambda_L"))
        row["lambda_Q"] = to_float(traj.get("lambda_Q"))
        row["lambda_S"] = to_float(traj.get("lambda_S"))
        row["beta_before_z"] = to_float(traj.get("beta_before_z"))
        row["beta_after_z"] = to_float(traj.get("beta_after_z"))
        row["z_guidance_reason"] = traj.get("z_guidance_reason", "")
        row["raw_ucb_bonus"] = to_float(traj.get("raw_ucb_bonus"))
        row["cost_aware_ucb_bonus"] = to_float(traj.get("cost_aware_ucb_bonus"))
        row["is_eclipse"] = int(bool(state.get("eclipse", False)))
        row["eclipse_phase"] = state.get("eclipse_phase", "unknown")
        row["remaining_contact_s_derived"] = state.get("remaining_contact_s", 0.0)
        row["remaining_contact_bin"] = contact_bin(to_float(row["remaining_contact_s_derived"]), to_int(row.get("contact_available")))
        row["before_long_contact_gap"] = int(state.get("before_long_contact_gap", 0))
        row["queue_pressure_tertile"] = tertile_bin(to_float(row.get("queue_mb_proposed")), q1_queue, q2_queue, ("low", "medium", "high"))
        row["battery_soc_tertile"] = tertile_bin(to_float(row.get("battery_soc_proposed")), q1_battery, q2_battery, ("low", "medium", "high"))
        row["thermal_pressure_tertile"] = tertile_bin(to_float(row.get("thermal_pressure_proposed")), q1_thermal, q2_thermal, ("low", "medium", "high"))
        row["uncertainty_bin"] = tertile_bin(row["U_t"], q1_unc, q2_unc, ("low", "medium", "high"))
        row["task_value_bin"] = tertile_bin(to_float(row.get("task_value")), q1_value, q2_value, ("low", "medium", "high"))
        row["deadline_bin_stress"] = deadline_bin(to_float(row.get("remaining_deadline_s"), to_float(row.get("deadline_s"))))
        row["high_value_group"] = "high_value" if to_int(row.get("high_value_flag")) == 1 else "routine_or_lower"
        row["emergency_group"] = "emergency" if to_int(row.get("emergency_flag")) == 1 else "non_emergency"
        row["delivered_outcome_proposed"] = outcome_bin(str(row.get("lifetime_status_proposed", "")))
    return paired


def contact_bin(seconds: float, contact_available: int) -> str:
    if not contact_available:
        return "no_contact"
    if not finite(seconds):
        return "unknown"
    if seconds <= 120:
        return "short_contact"
    if seconds <= 600:
        return "medium_contact"
    return "long_contact"


def deadline_bin(seconds: float) -> str:
    if not finite(seconds):
        return "unknown"
    minutes = seconds / 60.0
    if minutes <= 30:
        return "tight"
    if minutes <= 120:
        return "medium"
    return "loose"


def outcome_bin(status: str) -> str:
    status = status.lower()
    if "delivered" in status:
        return "delivered"
    if "expired" in status:
        return "expired"
    if "deferred" in status:
        return "deferred"
    if "dropped" in status:
        return "dropped"
    return status or "unknown"


REGIME_DIMENSIONS = [
    ("queue_pressure_tertile", "observed queue pressure"),
    ("battery_soc_tertile", "battery SOC"),
    ("thermal_pressure_tertile", "thermal pressure"),
    ("contact_available", "contact availability"),
    ("remaining_contact_bin", "remaining contact duration"),
    ("before_long_contact_gap", "before long contact gap"),
    ("eclipse_phase", "eclipse phase"),
    ("deadline_bin_stress", "deadline pressure"),
    ("high_value_group", "task value class"),
    ("emergency_group", "emergency class"),
    ("uncertainty_bin", "uncertainty"),
    ("task_class", "task class"),
]


LANDSCAPE_DIMENSIONS = [
    ("task_value_bin", "task value"),
    ("high_value_group", "high value flag"),
    ("emergency_group", "emergency flag"),
    ("uncertainty_bin", "uncertainty"),
    ("deadline_bin_stress", "deadline"),
    ("queue_pressure_tertile", "queue pressure"),
    ("contact_available", "contact availability"),
    ("delivered_outcome_proposed", "delivered outcome"),
]


def summarize_segmentation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    total = len(rows)
    for field, label in REGIME_DIMENSIONS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, "unknown"))].append(row)
        for value, subset in sorted(groups.items()):
            output.append(
                {
                    "dimension": field,
                    "dimension_label": label,
                    "regime": value,
                    "tasks": len(subset),
                    "share_of_tasks": safe_div(len(subset), total),
                    "high_value_share": mean([float(to_int(row.get("high_value_flag"))) for row in subset]),
                    "same_action_fraction": mean([float(to_int(row.get("same_action"))) for row in subset]),
                    "proposed_metadata_share": action_share(subset, "payload_type_proposed", "metadata"),
                    "proposed_roi_share": action_share(subset, "payload_type_proposed", "roi"),
                    "proposed_roi_context_share": action_share(subset, "payload_type_proposed", "roi_context"),
                    "proposed_defer_share": action_share(subset, "payload_type_proposed", "defer/store"),
                    "mean_queue_mb": mean([to_float(row.get("queue_mb_proposed")) for row in subset]),
                    "mean_battery_soc": mean([to_float(row.get("battery_soc_proposed")) for row in subset]),
                    "mean_thermal_pressure": mean([to_float(row.get("thermal_pressure_proposed")) for row in subset]),
                    "mean_remaining_deadline_s": mean([to_float(row.get("remaining_deadline_s")) for row in subset]),
                    "mean_uncertainty": mean([to_float(row.get("U_t")) for row in subset]),
                }
            )
    return output


def summarize_divergence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field, label in REGIME_DIMENSIONS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, "unknown"))].append(row)
        for value, subset in sorted(groups.items()):
            output.append(
                {
                    "dimension": field,
                    "dimension_label": label,
                    "regime": value,
                    "tasks": len(subset),
                    "same_action_fraction": mean([float(to_int(row.get("same_action"))) for row in subset]),
                    "divergence_fraction": 1.0 - mean([float(to_int(row.get("same_action"))) for row in subset]),
                    "both_balanced_roi_fraction": mean([
                        1.0
                        if row.get("selected_action_proposed") == "balanced+roi"
                        and row.get("selected_action_util") == "balanced+roi"
                        else 0.0
                        for row in subset
                    ]),
                    "proposed_metadata_util_roi_fraction": mean([float(to_int(row.get("proposed_metadata_util_roi"))) for row in subset]),
                    "proposed_roi_util_metadata_fraction": mean([float(to_int(row.get("proposed_roi_util_metadata"))) for row in subset]),
                    "proposed_defer_util_process_fraction": mean([float(to_int(row.get("proposed_defer_util_process"))) for row in subset]),
                    "util_heavier_than_proposed_fraction": mean([float(to_int(row.get("util_heavier_than_proposed"))) for row in subset]),
                    "proposed_heavier_than_util_fraction": mean([float(to_int(row.get("proposed_heavier_than_util"))) for row in subset]),
                    "mean_utility_delta_proposed_minus_util": mean([to_float(row.get("utility_delta_proposed_minus_util")) for row in subset]),
                    "mean_payload_size_delta_proposed_minus_util": mean([to_float(row.get("payload_size_delta_proposed_minus_util")) for row in subset]),
                    "mean_queue_mb_delta_proposed_minus_util": mean([to_float(row.get("queue_mb_delta_proposed_minus_util")) for row in subset]),
                }
            )
    return output


def selected_candidate_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    path = READINESS_DIR / "candidate_action_score_log_stress.csv"
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("variant") != PROPOSED:
                continue
            if to_int(row.get("selected_flag")) or to_int(row.get("executed_flag")):
                rows.append(row)
    return rows


def candidate_task_diagnostics() -> dict[str, Any]:
    path = READINESS_DIR / "candidate_action_score_log_stress.csv"
    if not path.exists():
        return {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("variant") == PROPOSED:
                grouped[str(row.get("task_id", ""))].append(row)

    tasks = 0
    selected_not_max_predicted = 0
    selected_not_max_guided = 0
    any_safety_rejected = 0
    safety_rejected_candidates = 0
    candidates = 0
    margins: list[float] = []
    for task_id, rows in grouped.items():
        if not task_id or not rows:
            continue
        tasks += 1
        candidates += len(rows)
        rejected = [row for row in rows if str(row.get("safety_status", "")).lower() not in {"safe", "ok", "accepted", ""}]
        safety_rejected_candidates += len(rejected)
        if rejected:
            any_safety_rejected += 1
        selected = [row for row in rows if to_int(row.get("selected_flag")) or to_int(row.get("executed_flag"))]
        if not selected:
            continue
        chosen = selected[0]
        max_pred = max(rows, key=lambda row: to_float(row.get("predicted_reward"), -1e100))
        max_guided = max(rows, key=lambda row: to_float(row.get("final_candidate_score"), -1e100))
        if chosen.get("candidate_action") != max_pred.get("candidate_action"):
            selected_not_max_predicted += 1
        if chosen.get("candidate_action") != max_guided.get("candidate_action"):
            selected_not_max_guided += 1
        margins.append(to_float(chosen.get("final_candidate_score")) - to_float(max_pred.get("final_candidate_score")))
    return {
        "candidate_tasks": tasks,
        "candidate_rows": candidates,
        "selected_not_max_predicted_reward_fraction": safe_div(selected_not_max_predicted, tasks),
        "selected_not_max_final_score_fraction": safe_div(selected_not_max_guided, tasks),
        "tasks_with_any_safety_rejection_fraction": safe_div(any_safety_rejected, tasks),
        "safety_rejected_candidate_fraction": safe_div(safety_rejected_candidates, candidates),
        "mean_selected_vs_max_predicted_final_score_margin": mean(margins),
    }


def summarize_activation(rows: list[dict[str, Any]], selected_candidates: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table: list[dict[str, Any]] = []

    def add_group(metric: str, field: str, subset_rows: list[dict[str, Any]]) -> None:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset_rows:
            groups[str(row.get(field, "unknown"))].append(row)
        for value, subset in sorted(groups.items()):
            table.append(
                {
                    "metric": metric,
                    "group_field": field,
                    "group_value": value,
                    "tasks": len(subset),
                    "metadata_or_defer_share": mean([
                        1.0 if row.get("payload_type_proposed") in {"metadata", "defer/store"} else 0.0 for row in subset
                    ]),
                    "roi_share": action_share(subset, "payload_type_proposed", "roi"),
                    "roi_context_share": action_share(subset, "payload_type_proposed", "roi_context"),
                    "defer_share": action_share(subset, "payload_type_proposed", "defer/store"),
                    "mean_payload_size_mb": mean([to_float(row.get("payload_size_mb_proposed")) for row in subset]),
                    "mean_model_rank": mean([float(row.get("model_rank_proposed", 0)) for row in subset]),
                    "mean_lambda_total": mean([to_float(row.get("lambda_total")) for row in subset]),
                    "mean_lambda_q": mean([to_float(row.get("lambda_Q")) for row in subset]),
                    "mean_ucb_bonus": mean([to_float(row.get("cost_aware_ucb_bonus"), to_float(row.get("raw_ucb_bonus"))) for row in subset]),
                    "mean_delivered_only_utility": mean([to_float(row.get("delivered_only_utility_proposed")) for row in subset]),
                }
            )

    for field in [
        "queue_pressure_tertile",
        "battery_soc_tertile",
        "thermal_pressure_tertile",
        "contact_available",
        "deadline_bin_stress",
        "eclipse_phase",
        "mode_name",
        "z_guidance_reason",
    ]:
        add_group("selected_action_relationship", field, rows)

    correlations = {
        "corr_lambda_total_vs_payload_mb": pearson(
            [to_float(row.get("lambda_total")) for row in rows],
            [to_float(row.get("payload_size_mb_proposed")) for row in rows],
        ),
        "corr_lambda_q_vs_payload_mb": pearson(
            [to_float(row.get("lambda_Q")) for row in rows],
            [to_float(row.get("payload_size_mb_proposed")) for row in rows],
        ),
        "corr_queue_mb_vs_metadata_or_defer": pearson(
            [to_float(row.get("queue_mb_proposed")) for row in rows],
            [1.0 if row.get("payload_type_proposed") in {"metadata", "defer/store"} else 0.0 for row in rows],
        ),
        "corr_battery_soc_vs_high_accuracy": pearson(
            [to_float(row.get("battery_soc_proposed")) for row in rows],
            [1.0 if row.get("model_rank_proposed") == 3 else 0.0 for row in rows],
        ),
        "corr_thermal_pressure_vs_model_rank": pearson(
            [to_float(row.get("thermal_pressure_proposed")) for row in rows],
            [float(row.get("model_rank_proposed", 0)) for row in rows],
        ),
        "corr_contact_available_vs_payload_rank": pearson(
            [float(to_int(row.get("contact_available"))) for row in rows],
            [float(row.get("payload_rank_proposed", 0)) for row in rows],
        ),
        "corr_remaining_deadline_vs_metadata_or_defer": pearson(
            [to_float(row.get("remaining_deadline_s")) for row in rows],
            [1.0 if row.get("payload_type_proposed") in {"metadata", "defer/store"} else 0.0 for row in rows],
        ),
    }

    for name, value in correlations.items():
        table.append({"metric": "correlation", "group_field": name, "group_value": "overall", "tasks": len(rows), "correlation": value})

    if selected_candidates:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in selected_candidates:
            kind = payload_type(str(row.get("candidate_payload", "")), str(row.get("candidate_action", "")), str(row.get("candidate_model", "")))
            groups[kind].append(row)
        for kind, subset in sorted(groups.items()):
            table.append(
                {
                    "metric": "selected_candidate_score_components",
                    "group_field": "selected_payload_type",
                    "group_value": kind,
                    "tasks": len(subset),
                    "mean_predicted_reward": mean([to_float(row.get("predicted_reward")) for row in subset]),
                    "mean_resource_cost": mean([to_float(row.get("predicted_resource_cost")) for row in subset]),
                    "mean_payload_mb": mean([to_float(row.get("predicted_payload_mb")) for row in subset]),
                    "mean_base_lyapunov_price_contribution": mean([to_float(row.get("base_lyapunov_price_contribution")) for row in subset]),
                    "mean_mode_guidance_contribution": mean([to_float(row.get("mode_guidance_contribution")) for row in subset]),
                    "mean_ucb_bonus": mean([to_float(row.get("ucb_bonus")) for row in subset]),
                    "mean_guided_score": mean([to_float(row.get("guided_score")) for row in subset]),
                    "mean_final_candidate_score": mean([to_float(row.get("final_candidate_score")) for row in subset]),
                }
            )

    return table, candidate_task_diagnostics()


def summarize_landscape(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field, label in LANDSCAPE_DIMENSIONS:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(str(row.get(field, "unknown")), str(row.get("payload_type_proposed", "unknown")))].append(row)
        for (value, kind), subset in sorted(groups.items()):
            delivered = [1.0 if row.get("delivered_outcome_proposed") == "delivered" else 0.0 for row in subset]
            expired = [1.0 if row.get("delivered_outcome_proposed") == "expired" else 0.0 for row in subset]
            payload_mb = [to_float(row.get("payload_size_mb_proposed")) for row in subset]
            utility = [to_float(row.get("delivered_only_utility_proposed")) for row in subset]
            output.append(
                {
                    "dimension": field,
                    "dimension_label": label,
                    "regime": value,
                    "payload_type": kind,
                    "tasks": len(subset),
                    "action_share_within_regime": safe_div(
                        len(subset),
                        sum(1 for row in rows if str(row.get(field, "unknown")) == value),
                    ),
                    "average_predicted_reward": mean([to_float(row.get("predicted_reward_proposed")) for row in subset]),
                    "average_delivered_only_utility": mean(utility),
                    "average_information_sufficiency": mean([to_float(row.get("information_sufficiency_proposed")) for row in subset]),
                    "average_payload_size_mb": mean(payload_mb),
                    "delivery_success_rate": mean(delivered),
                    "expiration_risk": mean(expired),
                    "utility_per_mb": safe_div(sum(value for value in utility if finite(value)), sum(value for value in payload_mb if finite(value))),
                }
            )
    return output


def top_rows(rows: list[dict[str, Any]], field: str, n: int = 6) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: to_float(row.get(field), -1e100), reverse=True)[:n]


def report_segmentation(rows: list[dict[str, Any]], seg_rows: list[dict[str, Any]]) -> str:
    contact_available = sum(1 for row in rows if to_int(row.get("contact_available")) == 1)
    high_value = sum(1 for row in rows if to_int(row.get("high_value_flag")) == 1)
    eclipse_counts = Counter(str(row.get("eclipse_phase")) for row in rows)
    uncertainty_missing = sum(1 for row in rows if not finite(to_float(row.get("U_t"))))
    largest = top_rows(seg_rows, "tasks", 8)
    lines = [
        "# Critical-Regime Segmentation Summary",
        "",
        "## Inputs",
        "",
        f"- Source: `outputs/_util_vs_proposed_audit/paired_task_trace_stress.csv` plus PROP-COMB stress trajectory.",
        f"- Aligned stress tasks: {len(rows)}.",
        f"- Contact-available task decisions: {contact_available}.",
        f"- High-value task decisions: {high_value}.",
        f"- Uncertainty field missing for tasks: {uncertainty_missing}.",
        f"- Eclipse phase counts: {dict(eclipse_counts)}.",
        "",
        "## Largest Regime Cells",
        "",
        "| Dimension | Regime | Tasks | Same Action | PROP Metadata | PROP ROI | PROP Defer |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in largest:
        lines.append(
            f"| {row['dimension']} | {row['regime']} | {row['tasks']} | {fmt(to_float(row['same_action_fraction']))} | "
            f"{fmt(to_float(row['proposed_metadata_share']))} | {fmt(to_float(row['proposed_roi_share']))} | {fmt(to_float(row['proposed_defer_share']))} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Remaining-contact duration, before-long-contact-gap, and eclipse phase are derived from trajectory slot fields rather than a dedicated contact-window table.",
        "- Product-requirement class is not present, so semantic validation of alert/verification/context/evidence needs remains incomplete.",
    ]
    return "\n".join(lines) + "\n"


def report_divergence(rows: list[dict[str, Any]], div_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    overall_same = mean([float(to_int(row.get("same_action"))) for row in rows])
    overall_div = 1.0 - overall_same
    critical_fields = {
        ("queue_pressure_tertile", "high"),
        ("battery_soc_tertile", "low"),
        ("thermal_pressure_tertile", "high"),
        ("contact_available", "0"),
        ("deadline_bin_stress", "tight"),
        ("high_value_group", "high_value"),
        ("eclipse_phase", "before_eclipse"),
        ("eclipse_phase", "in_eclipse"),
        ("before_long_contact_gap", "1"),
    }
    critical = [row for row in div_rows if (str(row.get("dimension")), str(row.get("regime"))) in critical_fields]
    weighted_tasks = sum(to_int(row.get("tasks")) for row in critical)
    weighted_div = safe_div(
        sum(to_float(row.get("divergence_fraction")) * to_int(row.get("tasks")) for row in critical),
        weighted_tasks,
    )
    strongest = sorted(critical, key=lambda row: to_float(row.get("divergence_fraction"), -1), reverse=True)[:8]
    lines = [
        "# Proposed-vs-UTIL Action Divergence by Regime",
        "",
        f"- Overall same-action fraction: {fmt(overall_same)}.",
        f"- Overall divergence fraction: {fmt(overall_div)}.",
        f"- Weighted divergence across predefined critical regimes: {fmt(weighted_div)}.",
        "",
        "## Critical Regime Highlights",
        "",
        "| Dimension | Regime | Tasks | Divergence | Both Balanced+ROI | UTIL Heavier | PROP Heavier | Mean Utility Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in strongest:
        lines.append(
            f"| {row['dimension']} | {row['regime']} | {row['tasks']} | {fmt(to_float(row['divergence_fraction']))} | "
            f"{fmt(to_float(row['both_balanced_roi_fraction']))} | {fmt(to_float(row['util_heavier_than_proposed_fraction']))} | "
            f"{fmt(to_float(row['proposed_heavier_than_util_fraction']))} | {fmt(to_float(row['mean_utility_delta_proposed_minus_util']))} |"
        )
    if finite(weighted_div) and weighted_div <= overall_div + 0.03:
        conclusion = "Divergence is not meaningfully amplified in critical regimes; the high same-action fraction is broadly persistent."
    else:
        conclusion = "Divergence increases in at least some critical regimes, suggesting partial mechanism activation under pressure."
    lines += ["", "## Answer", "", conclusion]
    return "\n".join(lines) + "\n", {"overall_same": overall_same, "overall_div": overall_div, "critical_div": weighted_div}


def report_activation(rows: list[dict[str, Any]], table: list[dict[str, Any]], candidate_diag: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    corr = {row["group_field"]: to_float(row.get("correlation")) for row in table if row.get("metric") == "correlation"}
    queue_groups = [row for row in table if row.get("group_field") == "queue_pressure_tertile" and row.get("metric") == "selected_action_relationship"]
    battery_groups = [row for row in table if row.get("group_field") == "battery_soc_tertile" and row.get("metric") == "selected_action_relationship"]
    thermal_groups = [row for row in table if row.get("group_field") == "thermal_pressure_tertile" and row.get("metric") == "selected_action_relationship"]
    mode_values = sorted({str(row.get("mode_name")) for row in rows if str(row.get("mode_name"))})
    z_values = sorted({str(row.get("z_guidance_reason")) for row in rows if str(row.get("z_guidance_reason"))})
    selected_not_pred = to_float(candidate_diag.get("selected_not_max_predicted_reward_fraction"))
    safety_rej = to_float(candidate_diag.get("tasks_with_any_safety_rejection_fraction"))
    signal_strength = "weak"
    if any(abs(value) >= 0.25 for value in corr.values() if finite(value)) or (finite(selected_not_pred) and selected_not_pred >= 0.20):
        signal_strength = "moderate"
    if any(abs(value) >= 0.50 for value in corr.values() if finite(value)):
        signal_strength = "strong"
    lines = [
        "# Long-Term Modulation Activation Report",
        "",
        "## Component Availability",
        "",
        f"- Mode names observed: {mode_values or ['none_logged']}.",
        f"- Z-guidance reasons observed: {z_values or ['none_logged']}.",
        f"- Candidate tasks inspected: {candidate_diag.get('candidate_tasks', 0)}.",
        f"- Selected action differed from max predicted-reward candidate: {fmt(selected_not_pred)} of candidate tasks.",
        f"- Tasks with at least one safety-rejected candidate: {fmt(safety_rej)}.",
        "",
        "## Correlation Checks",
        "",
        "| Relationship | Pearson r | Interpretation |",
        "| --- | ---: | --- |",
    ]
    interpretations = {
        "corr_lambda_total_vs_payload_mb": "negative would indicate higher resource price suppresses payload size",
        "corr_lambda_q_vs_payload_mb": "negative would indicate queue price suppresses payload size",
        "corr_queue_mb_vs_metadata_or_defer": "positive would indicate more conservative choices under queue pressure",
        "corr_battery_soc_vs_high_accuracy": "positive would indicate high-accuracy actions need better SOC",
        "corr_thermal_pressure_vs_model_rank": "negative would indicate thermal pressure suppresses model complexity",
        "corr_contact_available_vs_payload_rank": "positive would indicate richer payloads when contact exists",
        "corr_remaining_deadline_vs_metadata_or_defer": "negative would indicate tighter deadlines favor metadata/defer",
    }
    for name, interp in interpretations.items():
        lines.append(f"| `{name}` | {fmt(corr.get(name))} | {interp} |")
    lines += ["", "## Grouped Relationships", ""]
    for title, group_rows in [
        ("Queue pressure", queue_groups),
        ("Battery SOC", battery_groups),
        ("Thermal pressure", thermal_groups),
    ]:
        lines += [f"### {title}", "", "| Group | Tasks | Metadata/Defer | ROI | ROI+Context | Mean Payload MB |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in group_rows:
            lines.append(
                f"| {row['group_value']} | {row['tasks']} | {fmt(to_float(row.get('metadata_or_defer_share')))} | "
                f"{fmt(to_float(row.get('roi_share')))} | {fmt(to_float(row.get('roi_context_share')))} | {fmt(to_float(row.get('mean_payload_size_mb')))} |"
            )
        lines.append("")
    lines += [
        "## Answers",
        "",
        f"1. Long-term resource pricing shows `{signal_strength}` static/post-processed activation by these coarse relationships.",
        "2. Mode control is logged and varies if multiple mode names appear; the table reports action mix by mode, but this audit cannot prove causality.",
        "3. Proposed is considered more conservative under pressure only if metadata/defer shares rise or payload size falls in pressure bins.",
        "4. The evidence supports a long-term co-design claim only with cautious wording unless critical-regime divergence is also strong.",
        "5. Any remaining gap between mechanism activation and PROP-COMB/UTIL action divergence may reflect current stress coverage, the action-score landscape, and ROI-dominant utility profiles.",
    ]
    return "\n".join(lines) + "\n", {"activation_strength": signal_strength, **corr, **candidate_diag}


def report_roi_collapse(rows: list[dict[str, Any]], landscape: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    overall_roi = action_share(rows, "payload_type_proposed", "roi")
    overall_meta = action_share(rows, "payload_type_proposed", "metadata")
    overall_context = action_share(rows, "payload_type_proposed", "roi_context")
    overall_defer = action_share(rows, "payload_type_proposed", "defer/store")
    regime_roi_majority = []
    regime_count = 0
    for field, _ in LANDSCAPE_DIMENSIONS:
        values = sorted({str(row.get(field, "unknown")) for row in rows})
        for value in values:
            subset = [row for row in rows if str(row.get(field, "unknown")) == value]
            if len(subset) < 10:
                continue
            regime_count += 1
            if action_share(subset, "payload_type_proposed", "roi") >= 0.50:
                regime_roi_majority.append((field, value, len(subset), action_share(subset, "payload_type_proposed", "roi")))
    low_value = [row for row in rows if row.get("task_value_bin") == "low"]
    no_contact = [row for row in rows if str(row.get("contact_available")) == "0"]
    tight = [row for row in rows if row.get("deadline_bin_stress") == "tight"]
    high_value_contact = [
        row
        for row in rows
        if to_int(row.get("high_value_flag")) == 1 and str(row.get("contact_available")) == "1"
    ]
    delivered_roi = [row for row in rows if row.get("payload_type_proposed") == "roi"]
    delivered_meta = [row for row in rows if row.get("payload_type_proposed") == "metadata"]
    roi_majority_fraction = safe_div(len(regime_roi_majority), regime_count)
    collapse = overall_roi >= 0.55 and finite(roi_majority_fraction) and roi_majority_fraction >= 0.50
    severity = "MAJOR" if collapse and overall_context < 0.05 else "MODERATE" if collapse else "MINOR"
    lines = [
        "# ROI Sweet-Spot Collapse Report",
        "",
        "## Overall Proposed Action Mix",
        "",
        f"- Metadata share: {fmt(overall_meta)}.",
        f"- ROI share: {fmt(overall_roi)}.",
        f"- ROI+context share: {fmt(overall_context)}.",
        f"- Defer/store share: {fmt(overall_defer)}.",
        f"- Regime cells with ROI majority: {len(regime_roi_majority)} / {regime_count} cells with at least 10 tasks.",
        "",
        "## Stress Cases",
        "",
        f"- Low task-value ROI share: {fmt(action_share(low_value, 'payload_type_proposed', 'roi'))} over {len(low_value)} tasks.",
        f"- No-contact ROI share: {fmt(action_share(no_contact, 'payload_type_proposed', 'roi'))} over {len(no_contact)} tasks.",
        f"- Tight-deadline ROI share: {fmt(action_share(tight, 'payload_type_proposed', 'roi'))} over {len(tight)} tasks.",
        f"- High-value with contact ROI+context share: {fmt(action_share(high_value_contact, 'payload_type_proposed', 'roi_context'))} over {len(high_value_contact)} tasks.",
        f"- ROI mean information sufficiency: {fmt(mean([to_float(row.get('information_sufficiency_proposed')) for row in delivered_roi]))}.",
        f"- Metadata mean information sufficiency: {fmt(mean([to_float(row.get('information_sufficiency_proposed')) for row in delivered_meta]))}.",
        "",
        "## Answers",
        "",
        f"1. Balanced+ROI dominance assessment: {'yes' if collapse else 'not severe by threshold'}; severity `{severity}`.",
        "2. Low-value/high-certainty interpretation is limited because product-requirement class is missing; low task-value ROI share is reported above.",
        "3. No-contact and tight-deadline ROI shares indicate whether deadline/contact pressure meaningfully shifts away from ROI.",
        "4. ROI+context appears underused if its overall share is near zero, consistent with a high-cost context payload.",
        "5. Metadata is used as alert-level partial value, but the current action landscape can still favor ROI broadly.",
        "6. Static outputs cannot prove the information sufficiency function is wrong, but broad ROI dominance is consistent with ROI being universally attractive.",
        "7. The payload profile may create an ROI safe harbor if ROI combines high information sufficiency with manageable payload size under stress.",
    ]
    return "\n".join(lines) + "\n", {
        "overall_roi_share": overall_roi,
        "overall_metadata_share": overall_meta,
        "overall_roi_context_share": overall_context,
        "overall_defer_share": overall_defer,
        "roi_majority_regime_fraction": roi_majority_fraction,
        "roi_collapse_detected": collapse,
        "roi_collapse_severity": severity,
    }


def report_task_requirement(rows: list[dict[str, Any]]) -> str:
    low_tight_resource = [
        row
        for row in rows
        if row.get("task_value_bin") == "low"
        and (row.get("deadline_bin_stress") == "tight" or row.get("queue_pressure_tertile") == "high" or row.get("battery_soc_tertile") == "low" or row.get("thermal_pressure_tertile") == "high")
    ]
    hv_poor_contact = [
        row for row in rows if to_int(row.get("high_value_flag")) == 1 and str(row.get("contact_available")) == "0"
    ]
    hv_contact = [
        row for row in rows if to_int(row.get("high_value_flag")) == 1 and str(row.get("contact_available")) == "1"
    ]
    low_loose_capacity = [
        row
        for row in rows
        if row.get("task_value_bin") == "low"
        and row.get("deadline_bin_stress") == "loose"
        and str(row.get("contact_available")) == "1"
    ]
    high_pressure = [
        row
        for row in rows
        if row.get("queue_pressure_tertile") == "high" or row.get("battery_soc_tertile") == "low" or row.get("thermal_pressure_tertile") == "high"
    ]
    lines = [
        "# Task-Requirement Consistency Report",
        "",
        "No product-requirement class such as `alert`, `verification`, `contextual`, or `evidence` is present in the inspected logs. The checks below are therefore qualitative proxies, not definitive task-semantic validation.",
        "",
        "| Expected qualitative behavior | Proxy subset | Tasks | Metadata/Defer Share | ROI Share | ROI+Context Share | Assessment |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    checks = [
        (
            "low-value + tight/resources -> metadata or defer",
            low_tight_resource,
            "metadata/defer should be common",
            action_share(low_tight_resource, "payload_type_proposed", "metadata") + action_share(low_tight_resource, "payload_type_proposed", "defer/store") if low_tight_resource else math.nan,
        ),
        (
            "high-value + poor contact -> metadata-first",
            hv_poor_contact,
            "metadata should rise when contact is poor",
            action_share(hv_poor_contact, "payload_type_proposed", "metadata") if hv_poor_contact else math.nan,
        ),
        (
            "high-value + contact -> verification/context",
            hv_contact,
            "ROI/ROI+context should be plausible",
            action_share(hv_contact, "payload_type_proposed", "roi") + action_share(hv_contact, "payload_type_proposed", "roi_context") if hv_contact else math.nan,
        ),
        (
            "low-value + loose + capacity -> ROI acceptable but not universal",
            low_loose_capacity,
            "ROI can appear but should not dominate completely",
            action_share(low_loose_capacity, "payload_type_proposed", "roi") if low_loose_capacity else math.nan,
        ),
        (
            "high queue / low SOC / high thermal -> more conservative than UTIL",
            high_pressure,
            "UTIL-heavier-than-Proposed should rise",
            mean([float(to_int(row.get("util_heavier_than_proposed"))) for row in high_pressure]) if high_pressure else math.nan,
        ),
    ]
    for expected, subset, criterion, signal in checks:
        meta_defer = (
            action_share(subset, "payload_type_proposed", "metadata") + action_share(subset, "payload_type_proposed", "defer/store")
            if subset
            else math.nan
        )
        roi = action_share(subset, "payload_type_proposed", "roi")
        context = action_share(subset, "payload_type_proposed", "roi_context")
        if not subset:
            assessment = "insufficient rows"
        elif finite(signal) and signal >= 0.50:
            assessment = f"partially consistent: {criterion}"
        else:
            assessment = f"weak or mixed: {criterion}"
        lines.append(f"| {expected} | proxy | {len(subset)} | {fmt(meta_defer)} | {fmt(roi)} | {fmt(context)} | {assessment} |")
    lines += [
        "",
        "Definitive validation needs a task information requirement field generated from mission scenario priors, not SARFish labels.",
    ]
    return "\n".join(lines) + "\n"


def report_failure_modes(
    div_stats: dict[str, Any],
    activation_stats: dict[str, Any],
    roi_stats: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]], str]:
    categories: list[dict[str, str]] = []
    overall_div = to_float(div_stats.get("overall_div"))
    critical_div = to_float(div_stats.get("critical_div"))
    activation = str(activation_stats.get("activation_strength", "weak"))
    contact_available = sum(1 for row in rows if to_int(row.get("contact_available")) == 1)
    if finite(critical_div) and finite(overall_div) and critical_div > overall_div + 0.04:
        categories.append(
            {
                "category": "A. No serious issue",
                "severity": "MINOR",
                "evidence": f"Critical-regime divergence {fmt(critical_div)} exceeds overall divergence {fmt(overall_div)}.",
                "recommended_next_action": "Proceed with cautious claims and retain UTIL as strong baseline.",
            }
        )
    if contact_available < 0.10 * len(rows):
        categories.append(
            {
                "category": "B. Stress scenario too weak or too contact-scarce",
                "severity": "MODERATE",
                "evidence": f"Only {contact_available}/{len(rows)} task decisions have contact available, limiting contact-rich co-design contrast.",
                "recommended_next_action": "Keep main setting, but consider one sensitivity-only stress scenario after formal multi-seed.",
            }
        )
    if activation in {"weak", "moderate"} and (not finite(critical_div) or critical_div <= overall_div + 0.04):
        categories.append(
            {
                "category": "C. Long-term modulation weakly activated",
                "severity": "MAJOR" if activation == "weak" else "MODERATE",
                "evidence": f"Activation strength is {activation}; critical divergence {fmt(critical_div)} is not much above overall divergence {fmt(overall_div)}.",
                "recommended_next_action": "Add logging/shadow diagnostics before changing reward or queue behavior.",
            }
        )
    if roi_stats.get("roi_collapse_detected"):
        categories.append(
            {
                "category": "D. ROI sweet-spot collapse",
                "severity": str(roi_stats.get("roi_collapse_severity", "MODERATE")),
                "evidence": f"Proposed ROI share {fmt(to_float(roi_stats.get('overall_roi_share')))} and ROI-majority regime fraction {fmt(to_float(roi_stats.get('roi_majority_regime_fraction')))}.",
                "recommended_next_action": "Use product-requirement shadow diagnostics; do not tune profiles just to beat UTIL.",
            }
        )
    categories.append(
        {
            "category": "E. Task requirement model missing",
            "severity": "MAJOR",
            "evidence": "No alert/verification/contextual/evidence requirement field is present in the inspected logs.",
            "recommended_next_action": "Add product-requirement shadow diagnostics generated from mission priors, not SARFish labels.",
        }
    )
    if roi_stats.get("roi_collapse_detected") and activation in {"weak", "moderate"}:
        categories.append(
            {
                "category": "F. Possible formula/profile mismatch",
                "severity": "MODERATE",
                "evidence": "Broad ROI attractiveness plus weak critical-regime separation may indicate information sufficiency or payload-cost scaling is masking the intended hierarchy.",
                "recommended_next_action": "Perform design review after diagnostics; avoid blind tuning before formal results.",
            }
        )
    categories.append(
        {
            "category": "G. Baseline too strong but fair",
            "severity": "MINOR",
            "evidence": "UTIL is a safety-filtered current-slot utility rule and shares many balanced+ROI choices with Proposed.",
            "recommended_next_action": "Describe UTIL as a strong hand-crafted heuristic baseline.",
        }
    )

    recommendation = "D"
    if any(item["category"].startswith("F") and item["severity"] == "MAJOR" for item in categories):
        recommendation = "E"
    elif any(item["category"].startswith("C") and item["severity"] == "MAJOR" for item in categories) and not roi_stats.get("roi_collapse_detected"):
        recommendation = "F"
    elif not roi_stats.get("roi_collapse_detected") and activation != "weak":
        recommendation = "B"

    lines = [
        "# Mechanism Failure Mode Classification",
        "",
        "| Category | Severity | Evidence | Recommended next action |",
        "| --- | --- | --- | --- |",
    ]
    for item in categories:
        lines.append(f"| {item['category']} | {item['severity']} | {item['evidence']} | {item['recommended_next_action']} |")
    lines += ["", f"Selected recommendation code for final decision: `{recommendation}`."]
    return "\n".join(lines) + "\n", categories, recommendation


def report_final_recommendation(
    recommendation: str,
    div_stats: dict[str, Any],
    activation_stats: dict[str, Any],
    roi_stats: dict[str, Any],
) -> str:
    labels = {
        "A": "Proceed to formal multi-seed now.",
        "B": "Proceed to formal multi-seed, but weaken paper claims about beating UTIL.",
        "C": "Do one sensitivity-only stress scenario after formal multi-seed.",
        "D": "Before formal multi-seed, add only product-requirement shadow diagnostics, no policy change.",
        "E": "Before formal multi-seed, revise the task/product utility model because ROI collapse is severe.",
        "F": "Before formal multi-seed, revise long-term modulation because it does not affect actions in critical regimes.",
    }
    return f"""# Mechanism Activation Final Recommendation

## Recommendation

`{recommendation}. {labels[recommendation]}`

## Rationale

- Overall PROP-COMB/UTIL divergence: {fmt(to_float(div_stats.get("overall_div")))}.
- Critical-regime weighted divergence: {fmt(to_float(div_stats.get("critical_div")))}.
- Long-term activation strength: `{activation_stats.get("activation_strength", "unknown")}`.
- Proposed ROI share: {fmt(to_float(roi_stats.get("overall_roi_share")))}.
- ROI collapse detected: `{roi_stats.get("roi_collapse_detected")}`.

## Safe Guidance

- Do not change reward/profile values just to make Proposed win.
- Do not weaken UTIL unless unfair information use is found.
- If ROI collapse or weak modulation is a concern, add diagnostic/shadow metrics first.
- If an additional stress scenario is later needed, label it as sensitivity analysis rather than replacing the locked main formal setting.
"""


def stream_queue_log_brief() -> dict[str, Any]:
    path = READINESS_DIR / "queue_priority_component_log_stress.csv"
    if not path.exists():
        return {}
    total = 0
    selected = 0
    component_sums: dict[str, float] = defaultdict(float)
    component_counts: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("variant") != PROPOSED:
                continue
            total += 1
            if to_int(row.get("selected_for_downlink")):
                selected += 1
            for name in [
                "queue_priority_total",
                "utility_density_component",
                "urgency_component",
                "age_component",
                "metadata_bonus_component",
                "high_value_bonus_component",
            ]:
                value = to_float(row.get(name))
                if finite(value):
                    component_sums[name] += value
                    component_counts[name] += 1
    return {
        "queue_log_rows_proposed": total,
        "queue_selected_for_downlink_rows": selected,
        **{
            f"mean_{name}": safe_div(component_sums[name], component_counts[name])
            for name in component_sums
        },
    }


def main() -> None:
    if AUDIT_DIR.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {AUDIT_DIR}")
    paired, _, task_traj, slot_state = load_inputs()
    if not paired:
        raise FileNotFoundError("Missing paired stress trace from outputs/_util_vs_proposed_audit/")
    rows = enrich_paired(paired, task_traj, slot_state)
    selected_candidates = selected_candidate_rows()

    segmentation = summarize_segmentation(rows)
    divergence = summarize_divergence(rows)
    activation_table, candidate_diag = summarize_activation(rows, selected_candidates)
    queue_brief = stream_queue_log_brief()
    candidate_diag.update(queue_brief)
    landscape = summarize_landscape(rows)

    write_csv(AUDIT_DIR / "critical_regime_segmentation_summary.csv", segmentation)
    write_text(AUDIT_DIR / "critical_regime_segmentation_summary.md", report_segmentation(rows, segmentation))
    write_csv(AUDIT_DIR / "proposed_util_divergence_by_regime.csv", divergence)
    divergence_report, div_stats = report_divergence(rows, divergence)
    write_text(AUDIT_DIR / "proposed_util_divergence_by_regime.md", divergence_report)
    write_csv(AUDIT_DIR / "long_term_modulation_activation_tables.csv", activation_table)
    activation_report, activation_stats = report_activation(rows, activation_table, candidate_diag)
    write_text(AUDIT_DIR / "long_term_modulation_activation_report.md", activation_report)
    write_csv(AUDIT_DIR / "action_landscape_by_regime.csv", landscape)
    roi_report, roi_stats = report_roi_collapse(rows, landscape)
    write_text(AUDIT_DIR / "roi_sweet_spot_collapse_report.md", roi_report)
    write_text(AUDIT_DIR / "task_requirement_consistency_report.md", report_task_requirement(rows))
    failure_report, categories, recommendation = report_failure_modes(div_stats, activation_stats, roi_stats, rows)
    write_text(AUDIT_DIR / "mechanism_failure_mode_classification.md", failure_report)
    write_text(AUDIT_DIR / "mechanism_activation_final_recommendation.md", report_final_recommendation(recommendation, div_stats, activation_stats, roi_stats))

    print(f"files_created={len(list(AUDIT_DIR.iterdir()))}")
    print("task_regime_segmentation_possible=True")
    print(f"overall_same_action_fraction={fmt(to_float(div_stats.get('overall_same')))}")
    print(f"critical_regime_divergence_fraction={fmt(to_float(div_stats.get('critical_div')))}")
    print(f"proposed_differs_more_under_pressure={to_float(div_stats.get('critical_div')) > to_float(div_stats.get('overall_div')) + 0.03 if finite(to_float(div_stats.get('critical_div'))) else 'unknown'}")
    print(f"roi_collapse_detected={roi_stats.get('roi_collapse_detected')}")
    print(f"long_term_modulation_activation={activation_stats.get('activation_strength')}")
    print("failure_modes=" + "; ".join(item["category"].split(".")[0] for item in categories))
    print(f"recommendation={recommendation}")


if __name__ == "__main__":
    main()
