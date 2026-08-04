"""Method-neutral validation and atomic persistence of native run tables."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    from .atomic_io import atomic_write_csv, atomic_write_json
    from .contracts import EVENT_FIELDS, PER_TASK_FIELDS, SLOT_RESOURCE_FIELDS
except ImportError:
    from atomic_io import atomic_write_csv, atomic_write_json
    from contracts import EVENT_FIELDS, PER_TASK_FIELDS, SLOT_RESOURCE_FIELDS


def _exact(rows: list[Mapping[str, Any]], fields: tuple[str, ...], name: str) -> None:
    expected = set(fields)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(f"{name} row {index} schema mismatch")


def validate_unified_outputs(outputs: Mapping[str, Any], *, expected_tasks: int, expected_slots: int) -> None:
    tasks = list(outputs.get("per_task", []))
    events = list(outputs.get("events", []))
    slots = list(outputs.get("slots", []))
    if len(tasks) != expected_tasks or len(slots) != expected_slots:
        raise ValueError("unified row count mismatch")
    _exact(tasks, PER_TASK_FIELDS, "per-task")
    _exact(slots, SLOT_RESOURCE_FIELDS, "slot")
    if events:
        _exact(events, EVENT_FIELDS, "event")
    allowed = {"metadata", "roi", "evidence", "full"}
    seen: set[tuple[str, str, str]] = set()
    for row in events:
        if row.get("stage") not in allowed:
            raise ValueError("invalid delivery stage")
        physical = (str(row.get("task_id")), str(row.get("event_source")), str(row.get("ground_available_time")))
        if physical in seen and not int(row.get("duplicate_flag", 0)):
            raise ValueError("one physical event cannot receive duplicate positive credit")
        seen.add(physical)


def write_unified_outputs(output: Path, outputs: Mapping[str, Any]) -> None:
    output = Path(output)
    atomic_write_csv(output / "per_task_outcomes.csv", outputs["per_task"], PER_TASK_FIELDS)
    atomic_write_csv(output / "ground_delivery_events.csv", outputs["events"], EVENT_FIELDS)
    atomic_write_csv(output / "per_slot_resource_trace.csv", outputs["slots"], SLOT_RESOURCE_FIELDS)
    atomic_write_json(output / "resource_summary.json", outputs["resource_summary"])

