"""Strict MYOPIC-only immutable decision-value domain adapter for final-v6.

The canonical scientific value is never overwritten.  The adapter supplies a
unit-interval value only through the native task's calibrated decision field,
which is the field read by the frozen MYOPIC decision context.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import is_dataclass, replace
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from typing import Any, Iterable, Mapping


MYOPIC_VALUE_REFERENCE = Decimal("2.4787343282595278")
DECIMAL_CONTEXT = Context(prec=50, rounding=ROUND_HALF_EVEN)
NORMALIZATION_FORMULA = "myopic_decision_value = effective_task_value / 2.4787343282595278"
EXPECTED_MAPPING_SEQUENCE_SHA256 = "ff5259635e151a5d06196b5d5d96e7df4b6f51e5dbb6ee82c7c6f5be031f8615"
PAIR_RATIO_TOLERANCE = Decimal("1e-48")


def canonical_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _raw_decimal(token: Any) -> Decimal:
    try:
        value = Decimal(str(token))
    except Exception as error:
        raise ValueError(f"invalid effective_task_value: {token!r}") from error
    if not value.is_finite():
        raise ValueError(f"nonfinite effective_task_value: {token!r}")
    if value < 0 or value > MYOPIC_VALUE_REFERENCE:
        raise ValueError(f"effective_task_value outside registered MYOPIC adapter domain: {token!r}")
    return value


def myopic_decision_value(effective_task_value: Any) -> Decimal:
    raw = _raw_decimal(effective_task_value)
    with localcontext(DECIMAL_CONTEXT):
        decision = raw / MYOPIC_VALUE_REFERENCE
    if not decision.is_finite() or decision < 0 or decision > 1:
        raise ValueError(f"derived MYOPIC decision value outside [0,1]: {decision}")
    return decision


def _mapping_rows(tasks: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate canonical task_id: {task_id}")
        seen.add(task_id)
        raw_token = str(task["effective_task_value"])
        raw = _raw_decimal(raw_token)
        decision = myopic_decision_value(raw_token)
        rows.append({
            "task_id": task_id,
            "canonical_effective_task_value": raw_token,
            "myopic_decision_value": canonical_decimal(decision),
            "V_ref": canonical_decimal(MYOPIC_VALUE_REFERENCE),
            "normalization_formula": NORMALIZATION_FORMULA,
            "value_multiplier": str(task.get("value_multiplier", "1.0")),
            "_raw": canonical_decimal(raw),
        })
    return rows


def audit_canonical_value_mapping(tasks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    mappings = _mapping_rows(tasks)
    if not mappings:
        raise ValueError("canonical task mapping must not be empty")
    raw = [Decimal(row["_raw"]) for row in mappings]
    decision = [Decimal(row["myopic_decision_value"]) for row in mappings]
    pair_count = 0
    ordering_preserved = True
    all_pair_ratios = True
    maximum_cross_product_error = Decimal(0)
    for left in range(len(mappings)):
        for right in range(left + 1, len(mappings)):
            pair_count += 1
            raw_cmp = (raw[left] > raw[right]) - (raw[left] < raw[right])
            decision_cmp = (decision[left] > decision[right]) - (decision[left] < decision[right])
            if raw_cmp != decision_cmp:
                ordering_preserved = False
            with localcontext(DECIMAL_CONTEXT):
                error = abs(decision[left] * raw[right] - decision[right] * raw[left])
            maximum_cross_product_error = max(maximum_cross_product_error, error)
            if error > PAIR_RATIO_TOLERANCE:
                all_pair_ratios = False
            # The registered formula uses one exact common rational divisor.
            if ((Fraction(raw[left]) / Fraction(MYOPIC_VALUE_REFERENCE)) /
                    (Fraction(raw[right]) / Fraction(MYOPIC_VALUE_REFERENCE)) !=
                    Fraction(raw[left]) / Fraction(raw[right])):
                all_pair_ratios = False
    sequence = "".join(
        f"{row['task_id']}\t{row['canonical_effective_task_value']}\t"
        f"{row['myopic_decision_value']}\t{row['V_ref']}\n"
        for row in mappings
    ).encode("utf-8")
    sequence_sha = hashlib.sha256(sequence).hexdigest()
    value_multiplier_mismatches = sum(Decimal(row["value_multiplier"]) != Decimal(1) for row in mappings)
    finite = all(value.is_finite() and math.isfinite(float(value)) for value in decision)
    nonmaximum_saturation = sum(value == 1 and source != MYOPIC_VALUE_REFERENCE
                                for value, source in zip(decision, raw))
    passed = all((
        len(mappings) == 268,
        min(decision) >= 0,
        max(decision) == 1,
        ordering_preserved,
        all_pair_ratios,
        finite,
        nonmaximum_saturation == 0,
        value_multiplier_mismatches == 0,
        sequence_sha == EXPECTED_MAPPING_SEQUENCE_SHA256,
    ))
    report = {
        "status": "MYOPIC_DECISION_VALUE_CONTRACT_PASSED" if passed else "MYOPIC_DECISION_VALUE_CONTRACT_FAILED",
        "change_type": "MYOPIC_DECISION_VALUE_DOMAIN_ADAPTER",
        "task_count": len(mappings),
        "V_ref": canonical_decimal(MYOPIC_VALUE_REFERENCE),
        "normalization_formula": NORMALIZATION_FORMULA,
        "decimal_precision": DECIMAL_CONTEXT.prec,
        "pair_ratio_tolerance": str(PAIR_RATIO_TOLERANCE),
        "raw_minimum": canonical_decimal(min(raw)),
        "raw_maximum": canonical_decimal(max(raw)),
        "raw_greater_than_one_count": sum(value > 1 for value in raw),
        "minimum_myopic_decision_value": canonical_decimal(min(decision)),
        "maximum_myopic_decision_value": canonical_decimal(max(decision)),
        "all_values_in_unit_interval": all(0 <= value <= 1 for value in decision),
        "ordering_preserved": ordering_preserved,
        "all_pair_ratio_preserved": all_pair_ratios,
        "pair_count_checked": pair_count,
        "maximum_pair_cross_product_error": canonical_decimal(maximum_cross_product_error),
        "clipped_or_saturated_nonmaximum_count": nonmaximum_saturation,
        "nan_or_inf_count": sum(not value.is_finite() or not math.isfinite(float(value)) for value in decision),
        "value_multiplier_mismatch_count": value_multiplier_mismatches,
        "mapping_sequence_sha256": sequence_sha,
        "expected_mapping_sequence_sha256": EXPECTED_MAPPING_SEQUENCE_SHA256,
        "mappings": [{key: value for key, value in row.items() if key != "_raw"} for row in mappings],
    }
    if not passed:
        raise ValueError(f"MYOPIC decision-value contract gate failed: {report}")
    return report


def _adapt_frozen_task(task: Any, raw_token: str) -> Any:
    if not is_dataclass(task):
        raise TypeError(f"MYOPIC runtime task is not a dataclass: {type(task)!r}")
    raw_float = float(Decimal(raw_token))
    if Decimal.from_float(float(task.task_value)) != Decimal.from_float(raw_float):
        raise ValueError(f"canonical task value changed before MYOPIC adaptation: {task.task_id}")
    if Decimal(str(task.value_multiplier)) != Decimal(1):
        raise ValueError(f"runtime value_multiplier changed before MYOPIC adaptation: {task.task_id}")
    decision = myopic_decision_value(raw_token)
    return replace(task, calibrated_task_value=float(decision))


def adapt_myopic_scenario(scenario: Any) -> tuple[Any, dict[str, Any]]:
    audit = audit_canonical_value_mapping(scenario.canonical_tasks)
    canonical = {row["task_id"]: row for row in scenario.canonical_tasks}
    seen: set[str] = set()
    adapted_timeline: list[Any] = []
    for step in scenario.timeline:
        task = getattr(step, "task", None)
        if task is None:
            adapted_timeline.append(step)
            continue
        task_id = str(task.task_id)
        if task_id not in canonical:
            raise ValueError(f"MYOPIC adapter encountered noncanonical runtime task: {task_id}")
        if task_id in seen:
            raise ValueError(f"MYOPIC adapter encountered duplicate runtime task: {task_id}")
        seen.add(task_id)
        if not is_dataclass(step):
            raise TypeError(f"MYOPIC runtime timeline step is not a dataclass: {type(step)!r}")
        adapted_task = _adapt_frozen_task(task, str(canonical[task_id]["effective_task_value"]))
        adapted_timeline.append(replace(step, task=adapted_task))
    if seen != set(canonical):
        raise ValueError("MYOPIC adapter runtime task coverage mismatch")
    if not is_dataclass(scenario):
        raise TypeError(f"MYOPIC native scenario is not a dataclass: {type(scenario)!r}")
    adapted_scenario = replace(scenario, timeline=adapted_timeline)
    return adapted_scenario, audit


def validate_myopic_runtime_values(scenario: Any, bundle: Any) -> dict[str, Any]:
    audit = audit_canonical_value_mapping(bundle.tasks)
    expected = {row["task_id"]: row for row in audit["mappings"]}
    runtime: dict[str, Any] = {}
    for step in scenario.timeline:
        task = getattr(step, "task", None)
        if task is not None:
            runtime[str(task.task_id)] = task
    if set(runtime) != set(expected):
        raise ValueError("runtime MYOPIC task coverage differs from preregistered mapping")
    for task_id, mapping in expected.items():
        task = runtime[task_id]
        raw = float(Decimal(mapping["canonical_effective_task_value"]))
        decision = float(Decimal(mapping["myopic_decision_value"]))
        if Decimal.from_float(float(task.task_value)) != Decimal.from_float(raw):
            raise ValueError(f"runtime MYOPIC canonical value mismatch: {task_id}")
        if Decimal.from_float(float(task.calibrated_task_value)) != Decimal.from_float(decision):
            raise ValueError(f"runtime MYOPIC decision value mismatch: {task_id}")
        if Decimal(str(task.value_multiplier)) != Decimal(1):
            raise ValueError(f"runtime MYOPIC value_multiplier mismatch: {task_id}")
    return {key: value for key, value in audit.items() if key != "mappings"}
