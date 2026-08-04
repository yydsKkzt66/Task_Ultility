"""Frozen progressive marginal-utility ledger and corrected latency metrics."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Sequence


ZERO = Decimal("0")
ONE = Decimal("1")


def _d(value: Any) -> Decimal:
    return Decimal(str(value))


def _cumulative(stage: str, requirement: str, contract: Mapping[str, Any]) -> Decimal:
    table = contract["utility_table"]
    key = {"metadata": "metadata", "roi": "roi", "evidence": "roi_context"}.get(stage)
    return ONE if stage == "full" else (_d(table[key][requirement]) if key else ZERO)


def account_task(task: Mapping[str, Any], events: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    arrival, deadline = _d(task["arrival_time"]), _d(task["deadline"])
    lifetime = max(deadline - arrival, Decimal("0.000000000000000001"))
    value = _d(task["effective_task_value"])
    requirement = str(task.get("product_requirement", "verification")).lower()
    ordered = sorted(enumerate(events), key=lambda pair: (_d(pair[1]["ground_available_time"]), pair[0]))
    highest = ZERO
    rows = []
    for input_index, event in ordered:
        stage = str(event["stage"]).lower()
        cumulative = _cumulative(stage, requirement, contract)
        marginal = max(ZERO, cumulative - highest)
        duplicate = bool(int(event.get("duplicate_flag", 0))) or marginal == ZERO
        if marginal > ZERO:
            highest = max(highest, cumulative)
        ground = _d(event["ground_available_time"])
        on_time = int(ground <= deadline)
        timeliness = max(ZERO, min(ONE, ONE - (ground - arrival) / lifetime)) if on_time else ZERO
        positive = value * marginal * timeliness if on_time and not duplicate else ZERO
        weight = value * marginal if on_time and not duplicate else ZERO
        rows.append({"task_id": str(task["task_id"]), "stage": stage,
            "ground_available_time": str(ground), "arrival_time": str(arrival), "deadline": str(deadline),
            "latency_s": str(ground - arrival), "on_time": on_time,
            "cumulative_information_utility": str(cumulative),
            "marginal_information_utility": str(marginal), "timeliness_factor": str(timeliness),
            "effective_task_value": str(value), "utility_weight": str(weight),
            "positive_gross_utility": str(positive),
            "dedup_status": "DUPLICATE_OR_DOMINATED" if duplicate else "FIRST_POSITIVE_INCREMENT",
            "event_source": str(event.get("event_source", "")), "input_index": input_index})
    return rows


def _percentile(values: list[Decimal], q: Decimal) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int((q * len(ordered)).to_integral_value(rounding="ROUND_CEILING")) - 1))
    return ordered[rank]


def summarize_progressive(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if int(row.get("on_time", 0)) and _d(row.get("utility_weight", 0)) > ZERO]
    denominator = sum((_d(row["utility_weight"]) for row in valid), ZERO)
    numerator = sum((_d(row["utility_weight"]) * _d(row["latency_s"]) for row in valid), ZERO)
    gross = sum((_d(row.get("positive_gross_utility", 0)) for row in rows), ZERO)
    return {"j_prog_gross": str(gross),
        "effective_marginal_utility_denominator": str(denominator),
        "updl_s": str(numerator / denominator) if denominator > ZERO else None,
        "tasks_with_on_time_positive_layer": len({str(row["task_id"]) for row in valid}),
        "metadata_layer_count": sum(row["stage"] == "metadata" for row in valid),
        "roi_layer_count": sum(row["stage"] == "roi" for row in valid),
        "evidence_layer_count": sum(row["stage"] in {"evidence", "full"} for row in valid)}


def corrected_latency(task_rows: Sequence[Mapping[str, Any]], ledger_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[Mapping[str, Any]]] = {}
    for row in ledger_rows:
        if int(row.get("on_time", 0)):
            by.setdefault(str(row["task_id"]), []).append(row)
    rank = {"metadata": 1, "roi": 2, "evidence": 3, "full": 4}
    required_latency: list[Decimal] = []
    evidence_latency: list[Decimal] = []
    for task in task_rows:
        events = sorted(by.get(str(task["task_id"]), []), key=lambda row: _d(row["ground_available_time"]))
        required = {"alert": 1, "verification": 2, "context": 3, "evidence": 3}.get(str(task.get("product_requirement", "verification")).lower(), 2)
        closure = next((row for row in events if rank.get(str(row["stage"]), 0) >= required), None)
        full_evidence = next((row for row in events if rank.get(str(row["stage"]), 0) >= 3), None)
        arrival = _d(task.get("arrival_time", 0))
        if closure is not None:
            required_latency.append(_d(closure.get("latency_s", _d(closure["ground_available_time"]) - arrival)))
        if full_evidence is not None:
            evidence_latency.append(_d(full_evidence.get("latency_s", _d(full_evidence["ground_available_time"]) - arrival)))
    def mean(values: list[Decimal]) -> str | None:
        return str(sum(values, ZERO) / len(values)) if values else None
    return {"required_layer_success_count": len(required_latency),
        "required_layer_mean_latency_s": mean(required_latency),
        "required_layer_p95_latency_s": str(_percentile(required_latency, Decimal("0.95"))) if required_latency else None,
        "full_evidence_success_count": len(evidence_latency),
        "full_evidence_mean_latency_s": mean(evidence_latency),
        "full_evidence_p95_latency_s": str(_percentile(evidence_latency, Decimal("0.95"))) if evidence_latency else None}
