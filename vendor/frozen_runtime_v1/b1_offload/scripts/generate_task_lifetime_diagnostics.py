"""Generate task lifetime diagnostics from trajectory CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


ROUTERS_OF_INTEREST = (
    "proposed_lyapunov_rule_mode_safe_bandit",
    "utility_rule",
    "fixed_lightweight",
    "fixed_high_accuracy",
    "no_mode",
    "no_lyapunov",
)


LIFETIME_FIELDS = [
    "router_name",
    "task_id",
    "scene_id",
    "arrival_slot",
    "arrival_time_s",
    "task_class",
    "task_value",
    "deadline_s",
    "selected_action",
    "model_name",
    "payload_policy",
    "generated_payload_mb",
    "current_step_status",
    "first_queued_slot",
    "delivered_slot",
    "expired_slot",
    "dropped_slot",
    "payload_generation_time_s",
    "payload_delivery_time_s",
    "payload_expiration_time_s",
    "payload_drop_time_s",
    "deadline_time_s",
    "final_lifetime_status",
    "lifetime_latency_s",
    "met_deadline_lifetime",
    "exact_lifetime_timing_available",
    "detection_quality",
    "information_sufficiency",
    "timeliness_at_delivery",
    "delivered_only_utility",
    "delivered_before_deadline_utility",
    "high_value_flag",
    "effective_utility",
    "requirement_penalty",
    "lifecycle_event_ids_available",
    "missing_lifecycle_fields",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _split_ids(value: str | None) -> list[str]:
    if not value:
        return []
    value = str(value).strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]").replace("'", "").replace('"', "")
        return [item.strip() for item in value.split(",") if item.strip()]
    return [item.strip() for item in value.split(";") if item.strip()]


def _first_event_slots(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    slots: dict[str, int] = {}
    for row in rows:
        slot = _to_int(row.get("slot"))
        for task_id in _split_ids(row.get(field)):
            slots.setdefault(task_id, slot)
    return slots


def _first_event_times(rows: list[dict[str, str]], ids_field: str, times_field: str) -> dict[str, float]:
    times_by_task: dict[str, float] = {}
    for row in rows:
        ids = _split_ids(row.get(ids_field))
        times = _split_ids(row.get(times_field))
        for index, task_id in enumerate(ids):
            if task_id in times_by_task or index >= len(times):
                continue
            try:
                times_by_task[task_id] = float(times[index])
            except ValueError:
                continue
    return times_by_task


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _timeliness(latency_s: float | None, deadline_s: float) -> float | None:
    if latency_s is None or deadline_s <= 1.0e-9:
        return None
    return max(0.0, min(1.0, 1.0 - latency_s / deadline_s))


def _delivered_only_utility(row: dict[str, str], timeliness_at_delivery: float | None) -> float | None:
    if timeliness_at_delivery is None:
        return None
    value = _to_float(row.get("V_t"))
    quality = _to_float(row.get("detection_quality"))
    info = _to_float(row.get("information_factor"))
    return max(0.0, value) * max(0.0, quality) * max(0.0, info) * max(0.0, timeliness_at_delivery)


def build_lifetime_rows(output_dir: Path, slot_seconds: float) -> list[dict[str, object]]:
    trajectory_dir = output_dir / "trajectories"
    lifetime_rows: list[dict[str, object]] = []

    for trajectory_path in sorted(trajectory_dir.glob("*.csv")):
        rows = _read_csv(trajectory_path)
        if not rows:
            continue
        router_name = rows[0].get("router_name") or rows[0].get("router") or trajectory_path.stem
        event_id_fields = ("delivered_task_ids", "expired_task_ids", "dropped_task_ids")
        has_event_fields = all(field in rows[0] for field in event_id_fields)
        has_event_time_fields = all(
            field in rows[0]
            for field in ("delivered_task_event_times_s", "expired_task_event_times_s", "dropped_task_event_times_s")
        )
        missing = [field for field in event_id_fields if field not in rows[0]]
        if not has_event_time_fields:
            missing.extend(
                field
                for field in ("delivered_task_event_times_s", "expired_task_event_times_s", "dropped_task_event_times_s")
                if field not in rows[0]
            )
        missing_fields = ";".join(missing)
        delivered_slots = _first_event_slots(rows, "delivered_task_ids") if has_event_fields else {}
        expired_slots = _first_event_slots(rows, "expired_task_ids") if has_event_fields else {}
        dropped_slots = _first_event_slots(rows, "dropped_task_ids") if has_event_fields else {}
        delivered_times = _first_event_times(rows, "delivered_task_ids", "delivered_task_event_times_s") if has_event_time_fields else {}
        expired_times = _first_event_times(rows, "expired_task_ids", "expired_task_event_times_s") if has_event_time_fields else {}
        dropped_times = _first_event_times(rows, "dropped_task_ids", "dropped_task_event_times_s") if has_event_time_fields else {}

        task_rows = [row for row in rows if _to_int(row.get("has_task")) == 1]
        for row in task_rows:
            task_id = str(row.get("task_id", ""))
            arrival_slot = _to_int(row.get("slot"))
            current_status = str(row.get("delivery_status", "unknown") or "unknown")
            generated_payload_mb = _to_float(row.get("S_gen_current_mb"))
            first_queued_slot: int | None = None
            if generated_payload_mb > 0.0 and current_status in {"queued", "delivered", "dropped", "expired"}:
                first_queued_slot = arrival_slot

            delivered_slot = delivered_slots.get(task_id)
            expired_slot = expired_slots.get(task_id)
            dropped_slot = dropped_slots.get(task_id)
            delivered_time_s = delivered_times.get(task_id)
            expired_time_s = expired_times.get(task_id)
            dropped_time_s = dropped_times.get(task_id)

            if current_status == "delivered" and delivered_slot is None:
                delivered_slot = arrival_slot
                delivered_time_s = _to_float(row.get("payload_delivery_time_s"), None)
            if current_status == "expired" and expired_slot is None:
                expired_slot = arrival_slot
                expired_time_s = _to_float(row.get("expired_task_event_times_s"), None)
            if current_status == "dropped" and dropped_slot is None:
                dropped_slot = arrival_slot
                dropped_time_s = _to_float(row.get("dropped_task_event_times_s"), None)

            if current_status == "deferred":
                final_status = "deferred"
            elif delivered_slot is not None:
                final_status = "delivered"
            elif expired_slot is not None:
                final_status = "expired"
            elif dropped_slot is not None:
                final_status = "dropped"
            elif current_status == "queued":
                final_status = "undelivered_by_end"
            else:
                final_status = "unknown"

            lifetime_latency_s: float | None = None
            met_deadline: str = "unknown"
            arrival_time_s = arrival_slot * slot_seconds
            payload_generation_time_s = _to_float(row.get("payload_generation_time_s"), arrival_time_s)
            if payload_generation_time_s < 0.0:
                payload_generation_time_s = arrival_time_s
            deadline_time_s = arrival_time_s + _to_float(row.get("deadline_s"))
            exact_lifetime_timing_available = False
            if final_status == "delivered" and delivered_slot is not None:
                if delivered_time_s is not None and delivered_time_s >= 0.0:
                    lifetime_latency_s = _to_float(row.get("L_inf")) + max(0.0, delivered_time_s - payload_generation_time_s)
                    exact_lifetime_timing_available = True
                elif delivered_slot == arrival_slot:
                    lifetime_latency_s = _to_float(row.get("L_inf")) + _to_float(row.get("L_queue")) + _to_float(row.get("L_comm"))
                else:
                    lifetime_latency_s = max(0.0, (delivered_slot - arrival_slot) * slot_seconds)
                met_deadline = str(lifetime_latency_s <= _to_float(row.get("deadline_s"))).lower()
            elif final_status in {"expired", "deferred", "dropped"}:
                exact_lifetime_timing_available = (
                    final_status == "expired" and expired_time_s is not None
                ) or (final_status == "dropped" and dropped_time_s is not None)
                met_deadline = "false"
            timeliness_at_delivery = _timeliness(lifetime_latency_s, _to_float(row.get("deadline_s")))
            delivered_only_utility = (
                _delivered_only_utility(row, timeliness_at_delivery)
                if final_status == "delivered" and met_deadline == "true"
                else 0.0
            )

            lifetime_rows.append(
                {
                    "router_name": router_name,
                    "task_id": task_id,
                    "scene_id": row.get("scene_id", ""),
                    "arrival_slot": arrival_slot,
                    "arrival_time_s": _format_optional_float(arrival_time_s),
                    "task_class": row.get("task_class", ""),
                    "task_value": row.get("V_t", ""),
                    "deadline_s": row.get("deadline_s", ""),
                    "selected_action": row.get("selected_action", row.get("executed_action", "")),
                    "model_name": row.get("model_name", ""),
                    "payload_policy": row.get("payload_policy", ""),
                    "generated_payload_mb": row.get("S_gen_current_mb", ""),
                    "current_step_status": current_status,
                    "first_queued_slot": _format_optional_int(first_queued_slot),
                    "delivered_slot": _format_optional_int(delivered_slot),
                    "expired_slot": _format_optional_int(expired_slot),
                    "dropped_slot": _format_optional_int(dropped_slot),
                    "payload_generation_time_s": _format_optional_float(payload_generation_time_s),
                    "payload_delivery_time_s": _format_optional_float(delivered_time_s),
                    "payload_expiration_time_s": _format_optional_float(expired_time_s),
                    "payload_drop_time_s": _format_optional_float(dropped_time_s),
                    "deadline_time_s": _format_optional_float(deadline_time_s),
                    "final_lifetime_status": final_status,
                    "lifetime_latency_s": _format_optional_float(lifetime_latency_s),
                    "met_deadline_lifetime": met_deadline,
                    "exact_lifetime_timing_available": str(exact_lifetime_timing_available).lower(),
                    "detection_quality": row.get("detection_quality", ""),
                    "information_sufficiency": row.get("information_factor", ""),
                    "timeliness_at_delivery": _format_optional_float(timeliness_at_delivery),
                    "delivered_only_utility": _format_optional_float(delivered_only_utility),
                    "delivered_before_deadline_utility": _format_optional_float(delivered_only_utility),
                    "high_value_flag": row.get("high_value_flag", ""),
                    "effective_utility": row.get("effective_utility", ""),
                    "requirement_penalty": row.get("requirement_penalty", ""),
                    "lifecycle_event_ids_available": str(has_event_fields).lower(),
                    "missing_lifecycle_fields": missing_fields,
                }
            )
    return lifetime_rows


def _rows_by_router(rows: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["router_name"])].append(row)
    return grouped


def _counter(rows: list[dict[str, object]], field: str) -> Counter:
    return Counter(str(row.get(field, "")) for row in rows)


def _queued_then(rows: list[dict[str, object]], final_status: str) -> int:
    return sum(
        1
        for row in rows
        if str(row.get("current_step_status")) == "queued" and str(row.get("final_lifetime_status")) == final_status
    )


def write_lifetime_diagnosis(output_dir: Path, lifetime_rows: list[dict[str, object]]) -> Path:
    summary_rows = _read_csv(output_dir / "summary.csv")
    summary_by_router = {row["router"]: row for row in summary_rows}
    grouped = _rows_by_router(lifetime_rows)
    proposed_rows = grouped.get("proposed_lyapunov_rule_mode_safe_bandit", [])
    utility_rows = grouped.get("utility_rule", [])

    def summary_value(router: str, field: str) -> float:
        return _to_float(summary_by_router.get(router, {}).get(field))

    lines: list[str] = [
        "# Task Lifetime Diagnosis",
        "",
        "## Accounting definitions",
        "",
        "- `deadline_missed_count`: current task row latency check (`L_inf + L_queue + L_comm > deadline_s`). It does not identify queued payloads that expire later.",
        "- `completed_tasks`: current task rows whose payload is delivered in the same step.",
        "- `delivered_items`: queue items completed in any step, including items generated by earlier tasks.",
        "- `expired_items`: queue items removed because their queue lifetime exceeded their deadline.",
        "",
        "The lifecycle CSV uses diagnostic task-id lists exported by the queue (`delivered_task_ids`, `expired_task_ids`, `dropped_task_ids`) when available. Rows with missing lifecycle event ids are marked in `lifecycle_event_ids_available` and `missing_lifecycle_fields`.",
        "",
        "## Lifetime outcomes by router",
        "",
        "| router | tasks | current queued -> delivered | current queued -> expired | current queued -> undelivered_by_end | deferred | final delivered | final expired | final undelivered_by_end | final dropped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for router in ROUTERS_OF_INTEREST:
        rows = grouped.get(router, [])
        final_counts = _counter(rows, "final_lifetime_status")
        current_counts = _counter(rows, "current_step_status")
        lines.append(
            f"| {router} | {len(rows)} | {_queued_then(rows, 'delivered')} | {_queued_then(rows, 'expired')} | "
            f"{_queued_then(rows, 'undelivered_by_end')} | {current_counts.get('deferred', 0)} | "
            f"{final_counts.get('delivered', 0)} | {final_counts.get('expired', 0)} | "
            f"{final_counts.get('undelivered_by_end', 0)} | {final_counts.get('dropped', 0)} |"
        )

    prop_action_counts = _counter(proposed_rows, "selected_action")
    util_action_counts = _counter(utility_rows, "selected_action")
    prop_payload_mb = sum(_to_float(row.get("generated_payload_mb")) for row in proposed_rows)
    util_payload_mb = sum(_to_float(row.get("generated_payload_mb")) for row in utility_rows)
    prop_req = summary_value("proposed_lyapunov_rule_mode_safe_bandit", "total_requirement_penalty")
    util_req = summary_value("utility_rule", "total_requirement_penalty")
    prop_expired = _counter(proposed_rows, "final_lifetime_status").get("expired", 0)
    util_expired = _counter(utility_rows, "final_lifetime_status").get("expired", 0)

    lines.extend(
        [
            "",
            "## Proposed vs utility_rule",
            "",
            f"- proposed action distribution: {dict(prop_action_counts)}",
            f"- utility_rule action distribution: {dict(util_action_counts)}",
            f"- proposed generated payload MB: {prop_payload_mb:.3f}; utility_rule generated payload MB: {util_payload_mb:.3f}.",
            f"- proposed final expired tasks: {prop_expired}; utility_rule final expired tasks: {util_expired}.",
            f"- proposed total requirement penalty: {prop_req:.3f}; utility_rule total requirement penalty: {util_req:.3f}.",
            "",
        ]
    )
    if prop_req > util_req and prop_payload_mb < util_payload_mb:
        lines.append(
            "Diagnosis: proposed is lower mainly because it chooses more conservative metadata/defer actions, generates less useful payload, and pays higher requirement penalty. Queue expiration still matters, but it is not uniquely worse than utility_rule in this run."
        )
    elif prop_expired > util_expired:
        lines.append(
            "Diagnosis: proposed is lower mainly because queued payloads expire more often than utility_rule, with action and requirement differences as secondary factors."
        )
    else:
        lines.append(
            "Diagnosis: proposed/utility_rule gap is mixed; inspect action distribution, delivered MB, requirement penalty, and final lifetime status together rather than reading any one summary metric alone."
        )

    proposed_summary = summary_by_router.get("proposed_lyapunov_rule_mode_safe_bandit", {})
    deadline_missed = _to_int(proposed_summary.get("deadline_missed_count"))
    final_expired = _counter(proposed_rows, "final_lifetime_status").get("expired", 0)
    lines.extend(
        [
            "",
            "## Summary interpretation risk",
            "",
            f"For proposed, `deadline_missed_count={deadline_missed}` while lifetime final expired tasks={final_expired}.",
        ]
    )
    if deadline_missed == 0 and final_expired > 0:
        lines.append(
            "This is a summary interpretation risk: current-step deadline misses can be zero while queued payloads expire later."
        )
    else:
        lines.append(
            "No severe deadline/expiration contradiction is visible, but current-step and lifetime metrics should still be reported separately."
        )

    path = output_dir / "task_lifetime_diagnosis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_lifetime_outputs(output_dir: Path, slot_seconds: float) -> tuple[Path, Path, list[dict[str, object]]]:
    rows = build_lifetime_rows(output_dir, slot_seconds)
    csv_path = output_dir / "task_lifetime_outcomes.csv"
    _write_csv(csv_path, rows, LIFETIME_FIELDS)
    report_path = write_lifetime_diagnosis(output_dir, rows)
    return csv_path, report_path, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task lifetime diagnostics from trajectory outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slot-seconds", type=float, required=True)
    args = parser.parse_args()
    generate_lifetime_outputs(args.output_dir, args.slot_seconds)


if __name__ == "__main__":
    main()
