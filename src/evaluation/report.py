"""CSV 鎶ュ憡鍐欏嚭銆?"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from pathlib import Path

from src.common.types import EpisodeResult

LOGGER = logging.getLogger(__name__)
TRAJECTORY_ACCOUNTING_FIELDS = (
    "resource_safety_violation",
    "task_deferred",
    "deadline_missed",
    "service_failure_reason",
    "safety_violation_reason_before",
    "safety_violation_reason_after",
    "raw_safety_violation",
    "final_resource_safety_violation",
    "downlink_throttled",
    "downlink_throttle_reason",
    "planned_downlink_mb_before_throttle",
    "actual_downlink_mb_after_throttle",
    "planned_comm_energy_j_before_throttle",
    "actual_comm_energy_j_after_throttle",
    "thermal_guard_exceeded",
    "battery_guard_exceeded",
    "thermal_guard_penalty",
    "battery_reserve_penalty",
    "downlink_burden_penalty",
    "queue_pressure_risk_penalty",
    "long_term_risk_penalty",
    "net_system_reward_slot",
)
SUMMARY_ACCOUNTING_FIELDS = (
    "raw_safety_violation_count",
    "final_resource_safety_violation_count",
    "task_deferred_count",
    "deadline_missed_count",
    "service_failure_count",
    "resource_feasible_flag",
    "downlink_throttle_count",
    "total_planned_downlink_mb_before_throttle",
    "total_actual_downlink_mb_after_throttle",
    "downlink_throttle_thermal_count",
    "gross_task_utility",
    "net_system_utility",
    "total_long_term_risk_penalty",
    "total_thermal_guard_penalty",
    "total_battery_reserve_penalty",
    "total_downlink_burden_penalty",
    "total_queue_pressure_risk_penalty",
    "terminal_state_penalty",
    "thermal_guard_exceed_count",
    "thermal_090_exceed_count",
    "thermal_091_exceed_count",
    "battery_below_080_count",
    "battery_below_077_count",
    "battery_below_075_count",
    "rank_by_gross_task_utility",
    "rank_by_total_penalized_reward",
    "rank_by_net_system_utility",
)
SERVICE_FAILURE_PRIORITY = (
    "deferred",
    "deadline_missed",
    "quality_gap",
    "information_gap",
    "battery_violation",
    "thermal_violation",
    "queue_violation",
)
RESOURCE_REASON_LABELS = {
    "predicted_battery_too_low": "battery_violation",
    "battery_reserve_violation": "battery_violation",
    "predicted_thermal_too_high": "thermal_violation",
    "thermal_pressure_violation": "thermal_violation",
    "predicted_queue_overflow": "queue_violation",
    "queue_overflow_violation": "queue_violation",
    "ai_energy_budget_violation": "battery_violation",
}
CORE_COMPARISON_KEYS = (
    "total_mission_utility",
    "total_effective_utility",
    "total_penalized_reward",
    "deadline_satisfaction_ratio",
    "average_queue_mb",
    "total_energy",
    "total_generated_payload_mb",
)
COMPARISON_TOLERANCE = 1.0e-12


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_reason_label(reason: object) -> str:
    label = str(reason).strip().lower().replace("/", "_").replace(" ", "_")
    if not label:
        return ""
    return RESOURCE_REASON_LABELS.get(label, label)


def _normalize_reason_list(value: object) -> list[str]:
    if isinstance(value, list):
        raw_reasons = value
    else:
        raw_reasons = str(value or "").split(";")
    labels: list[str] = []
    for reason in raw_reasons:
        label = _normalize_reason_label(reason)
        if label and label != "none" and label not in labels:
            labels.append(label)
    return labels


def _format_reasons(value: object, default: str = "none") -> str:
    labels = _normalize_reason_list(value)
    return ";".join(labels) if labels else default


def _derive_task_deferred(row: dict[str, object]) -> int:
    if "has_task" in row and _to_int(row.get("has_task")) == 0:
        return 0
    executed_action = str(row.get("executed_action", row.get("selected_action", ""))).strip().lower()
    selected_model = str(row.get("selected_model", row.get("model_name", ""))).strip().lower()
    payload_policy = str(row.get("selected_payload", row.get("payload_policy", ""))).strip().lower()
    delivery_status = str(row.get("delivery_status", "")).strip().lower()
    deferred_actions = {"delay/store", "delay_store", "defer", "no-op", "noop", "no_op"}
    return int(
        executed_action in deferred_actions
        or selected_model in {"delay_store", "no-op", "noop", "no_op"}
        or payload_policy == "defer"
        or delivery_status == "deferred"
    )


def _derive_deadline_missed(row: dict[str, object]) -> int:
    if "has_task" in row and _to_int(row.get("has_task")) == 0:
        return 0
    deadline_s = _to_float(row.get("deadline_s"))
    if deadline_s <= 0.0:
        return 0
    latency_s = _to_float(row.get("L_inf")) + _to_float(row.get("L_queue")) + _to_float(row.get("L_comm"))
    return int(latency_s > deadline_s + COMPARISON_TOLERANCE)


def _select_service_failure_reason(row: dict[str, object], after_reasons: list[str]) -> str:
    existing_reasons = _normalize_reason_list(row.get("service_failure_reason"))
    candidates = list(existing_reasons)
    if _to_int(row.get("task_deferred")):
        candidates.append("deferred")
    if _to_int(row.get("deadline_missed")):
        candidates.append("deadline_missed")
    if _to_float(row.get("quality_gap")) > 0.0:
        candidates.append("quality_gap")
    if _to_float(row.get("information_gap")) > 0.0:
        candidates.append("information_gap")
    if _to_int(row.get("final_resource_safety_violation")):
        candidates.extend(after_reasons)

    normalized_candidates = {_normalize_reason_label(reason) for reason in candidates}
    for reason in SERVICE_FAILURE_PRIORITY:
        if reason in normalized_candidates:
            return reason
    return "none"


def _ensure_trajectory_accounting_fields(row: dict[str, object]) -> dict[str, object]:
    row = dict(row)
    raw_safety_violation = _to_int(
        row.get("raw_safety_violation", row.get("safety_violation_before_correction", 0))
    )
    final_resource_safety_violation = _to_int(
        row.get(
            "final_resource_safety_violation",
            row.get(
                "resource_safety_violation",
                row.get("safety_violation_after_correction", row.get("safety_violation", 0)),
            ),
        )
    )

    row["raw_safety_violation"] = raw_safety_violation
    row["final_resource_safety_violation"] = final_resource_safety_violation
    row["resource_safety_violation"] = final_resource_safety_violation
    row["task_deferred"] = _to_int(row.get("task_deferred", _derive_task_deferred(row)))
    row["deadline_missed"] = _to_int(row.get("deadline_missed", _derive_deadline_missed(row)))
    row["safety_violation_reason_before"] = _format_reasons(row.get("safety_violation_reason_before"))

    after_reasons = _normalize_reason_list(row.get("safety_violation_reason_after"))
    if final_resource_safety_violation == 0:
        after_reasons = []
    row["safety_violation_reason_after"] = ";".join(after_reasons) if after_reasons else "none"
    row["service_failure_reason"] = _select_service_failure_reason(row, after_reasons)
    row["downlink_throttled"] = _to_int(row.get("downlink_throttled", 0))
    row["downlink_throttle_reason"] = str(row.get("downlink_throttle_reason", "none") or "none")
    row["planned_downlink_mb_before_throttle"] = _to_float(row.get("planned_downlink_mb_before_throttle", 0.0))
    row["actual_downlink_mb_after_throttle"] = _to_float(row.get("actual_downlink_mb_after_throttle", 0.0))
    row["planned_comm_energy_j_before_throttle"] = _to_float(row.get("planned_comm_energy_j_before_throttle", 0.0))
    row["actual_comm_energy_j_after_throttle"] = _to_float(row.get("actual_comm_energy_j_after_throttle", 0.0))
    row["thermal_guard_exceeded"] = _to_int(row.get("thermal_guard_exceeded", 0))
    row["battery_guard_exceeded"] = _to_int(row.get("battery_guard_exceeded", 0))
    row["thermal_guard_penalty"] = _to_float(row.get("thermal_guard_penalty", 0.0))
    row["battery_reserve_penalty"] = _to_float(row.get("battery_reserve_penalty", 0.0))
    row["downlink_burden_penalty"] = _to_float(row.get("downlink_burden_penalty", 0.0))
    row["queue_pressure_risk_penalty"] = _to_float(row.get("queue_pressure_risk_penalty", 0.0))
    row["long_term_risk_penalty"] = _to_float(row.get("long_term_risk_penalty", 0.0))
    row["net_system_reward_slot"] = _to_float(row.get("net_system_reward_slot", 0.0))
    return row


def _ensure_summary_accounting_fields(
    summary: dict[str, float | int | str],
    rows: list[dict[str, object]],
) -> dict[str, float | int | str]:
    summary = dict(summary)
    summary.setdefault("raw_safety_violation_count", sum(_to_int(row.get("raw_safety_violation")) for row in rows))
    summary.setdefault(
        "final_resource_safety_violation_count",
        sum(_to_int(row.get("final_resource_safety_violation")) for row in rows),
    )
    summary.setdefault("task_deferred_count", sum(_to_int(row.get("task_deferred")) for row in rows))
    summary.setdefault("deadline_missed_count", sum(_to_int(row.get("deadline_missed")) for row in rows))
    summary.setdefault(
        "service_failure_count",
        sum(1 for row in rows if str(row.get("service_failure_reason", "none")) != "none"),
    )
    summary.setdefault("resource_feasible_flag", int(summary["final_resource_safety_violation_count"] == 0))
    summary.setdefault("downlink_throttle_count", sum(_to_int(row.get("downlink_throttled")) for row in rows))
    summary.setdefault(
        "total_planned_downlink_mb_before_throttle",
        sum(_to_float(row.get("planned_downlink_mb_before_throttle")) for row in rows),
    )
    summary.setdefault(
        "total_actual_downlink_mb_after_throttle",
        sum(_to_float(row.get("actual_downlink_mb_after_throttle")) for row in rows),
    )
    summary.setdefault(
        "downlink_throttle_thermal_count",
        sum(
            1
            for row in rows
            if _to_int(row.get("downlink_throttled"))
            and str(row.get("downlink_throttle_reason", "none")) == "thermal_guard"
        ),
    )
    summary.setdefault("gross_task_utility", summary.get("total_effective_utility", 0.0))
    summary.setdefault(
        "total_thermal_guard_penalty",
        sum(_to_float(row.get("thermal_guard_penalty")) for row in rows),
    )
    summary.setdefault(
        "total_battery_reserve_penalty",
        sum(_to_float(row.get("battery_reserve_penalty")) for row in rows),
    )
    summary.setdefault(
        "total_downlink_burden_penalty",
        sum(_to_float(row.get("downlink_burden_penalty")) for row in rows),
    )
    summary.setdefault(
        "total_queue_pressure_risk_penalty",
        sum(_to_float(row.get("queue_pressure_risk_penalty")) for row in rows),
    )
    summary.setdefault("total_long_term_risk_penalty", sum(_to_float(row.get("long_term_risk_penalty")) for row in rows))
    summary.setdefault("terminal_state_penalty", 0.0)
    summary.setdefault(
        "net_system_utility",
        _to_float(summary.get("gross_task_utility"))
        - _to_float(summary.get("total_resource_penalty"))
        - _to_float(summary.get("total_requirement_penalty"))
        - _to_float(summary.get("total_safety_penalty"))
        - _to_float(summary.get("total_long_term_risk_penalty"))
        - _to_float(summary.get("terminal_state_penalty")),
    )
    summary.setdefault("thermal_guard_exceed_count", sum(_to_int(row.get("thermal_guard_exceeded")) for row in rows))
    summary.setdefault("thermal_090_exceed_count", sum(1 for row in rows if _to_float(row.get("thermal_next")) > 0.90))
    summary.setdefault("thermal_091_exceed_count", sum(1 for row in rows if _to_float(row.get("thermal_next")) > 0.91))
    summary.setdefault("battery_below_080_count", sum(1 for row in rows if _to_float(row.get("battery_next_soc")) < 0.80))
    summary.setdefault("battery_below_077_count", sum(1 for row in rows if _to_float(row.get("battery_next_soc")) < 0.77))
    summary.setdefault("battery_below_075_count", sum(1 for row in rows if _to_float(row.get("battery_next_soc")) < 0.75))
    return summary


def _ordered_fieldnames(
    rows: list[dict[str, object]],
    required_fields: tuple[str, ...],
) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for key in required_fields:
        if key not in fieldnames:
            fieldnames.append(key)
    return fieldnames


def _warn_if_proposed_matches_no_mode(results: list[EpisodeResult]) -> None:
    by_router = {result.router_name: result for result in results}
    proposed = by_router.get("proposed_lyapunov_rule_mode_safe_bandit")
    no_mode = by_router.get("no_mode")
    if proposed is None or no_mode is None:
        return

    if [record.selected_action for record in proposed.records] == [record.selected_action for record in no_mode.records]:
        LOGGER.warning(
            "proposed_lyapunov_rule_mode_safe_bandit and no_mode selected identical actions; "
            "z_k/mode guidance may not be taking effect."
        )

    comparable_keys = [key for key in CORE_COMPARISON_KEYS if key in proposed.summary and key in no_mode.summary]
    matching_keys = [
        key
        for key in comparable_keys
        if abs(float(proposed.summary[key]) - float(no_mode.summary[key])) <= COMPARISON_TOLERANCE
    ]
    if comparable_keys and len(matching_keys) == len(comparable_keys):
        LOGGER.warning(
            "proposed_lyapunov_rule_mode_safe_bandit and no_mode have identical core summary metrics (%s); "
            "z_k/mode guidance may not be taking effect.",
            ", ".join(matching_keys),
        )


def _sort_summary_rows(summary_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    for rank, row in enumerate(
        sorted(summary_rows, key=lambda item: float(item["gross_task_utility"]), reverse=True),
        start=1,
    ):
        row["rank_by_gross_task_utility"] = rank
    for rank, row in enumerate(
        sorted(summary_rows, key=lambda item: float(item["total_penalized_reward"]), reverse=True),
        start=1,
    ):
        row["rank_by_total_penalized_reward"] = rank
    for rank, row in enumerate(
        sorted(summary_rows, key=lambda item: float(item["net_system_utility"]), reverse=True),
        start=1,
    ):
        row["rank_by_net_system_utility"] = rank
    for rank, row in enumerate(
        sorted(summary_rows, key=lambda item: float(item["total_penalized_reward"]), reverse=True),
        start=1,
    ):
        row["rank_by_penalized_reward"] = rank
    for rank, row in enumerate(
        sorted(summary_rows, key=lambda item: float(item["total_effective_utility"]), reverse=True),
        start=1,
    ):
        row["rank_by_effective_utility"] = rank
    for rank, row in enumerate(
        sorted(summary_rows, key=lambda item: float(item["high_value_task_utility"]), reverse=True),
        start=1,
    ):
        row["rank_by_high_value_utility"] = rank

    ranked = sorted(
        summary_rows,
        key=lambda row: (
            -int(row["resource_feasible_flag"]),
            -float(row["total_penalized_reward"]),
            -float(row["total_effective_utility"]),
            -float(row["high_value_task_utility"]),
            -float(row["deadline_satisfaction_ratio"]),
            float(row["queue_overflow_ratio"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _round_for_display(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
    rounded: dict[str, float | int | str] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def write_episode_results(results: list[EpisodeResult], output_dirs: dict[str, Path]) -> tuple[Path, list[Path]]:
    """鍐欏嚭 summary.csv 涓?trajectory CSV銆?"""

    _warn_if_proposed_matches_no_mode(results)

    summary_path = output_dirs["root"] / "summary.csv"
    summary_raw_path = output_dirs["root"] / "summary_raw.csv"
    summary_display_path = output_dirs["root"] / "summary_display.csv"
    trajectory_paths: list[Path] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for result in results:
        rows = [_ensure_trajectory_accounting_fields(asdict(record)) for record in result.records]
        summary_rows.append(_ensure_summary_accounting_fields(result.summary, rows))
        trajectory_path = output_dirs["trajectories"] / f"{result.router_name}.csv"
        with trajectory_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=_ordered_fieldnames(rows, TRAJECTORY_ACCOUNTING_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        trajectory_paths.append(trajectory_path)

    ranked_rows = _sort_summary_rows(summary_rows)
    display_rows = [_round_for_display(row) for row in ranked_rows]

    with summary_raw_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_ordered_fieldnames(summary_rows, SUMMARY_ACCOUNTING_FIELDS))
        writer.writeheader()
        writer.writerows(summary_rows)

    with summary_display_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_ordered_fieldnames(display_rows, SUMMARY_ACCOUNTING_FIELDS))
        writer.writeheader()
        writer.writerows(display_rows)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_ordered_fieldnames(ranked_rows, SUMMARY_ACCOUNTING_FIELDS))
        writer.writeheader()
        writer.writerows(ranked_rows)
    return summary_path, trajectory_paths
