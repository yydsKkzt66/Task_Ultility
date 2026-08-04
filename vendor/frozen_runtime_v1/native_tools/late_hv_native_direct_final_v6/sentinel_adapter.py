"""Strict post-simulation classification of native control sentinel events."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping


CONTROL_TASK_IDS = frozenset({"__idle__"})


class UnknownCanonicalTaskId(KeyError):
    """Raised when a non-control native event cannot be mapped to a task."""


@dataclass
class ControlEventAudit:
    count: int = 0
    event_type_counts: Counter[str] = field(default_factory=Counter)
    task_id_counts: Counter[str] = field(default_factory=Counter)
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None

    def record(self, task_id: str, row: Mapping[str, Any], native_event_index: int) -> None:
        snapshot = {"native_event_index": native_event_index, "task_id": task_id, **dict(row)}
        event_type = str(row.get("product_type", row.get("stage", "unknown")))
        self.count += 1
        self.event_type_counts[event_type] += 1
        self.task_id_counts[task_id] += 1
        if self.first is None:
            self.first = snapshot
        self.last = snapshot

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_event_count": self.count,
            "control_event_type_counts": dict(sorted(self.event_type_counts.items())),
            "control_task_id_counts": dict(sorted(self.task_id_counts.items())),
            "first_control_event": self.first,
            "last_control_event": self.last,
        }


def classify_native_event(
    task_id: Any,
    row: Mapping[str, Any],
    native_event_index: int,
    canonical_task_by_id: Mapping[str, Mapping[str, Any]],
    audit: ControlEventAudit,
) -> Mapping[str, Any] | None:
    """Return the exact canonical task, or classify the exact control sentinel.

    No fallback is allowed: every non-whitelisted ID must be canonical.
    """
    if task_id in CONTROL_TASK_IDS:
        audit.record(task_id, row, native_event_index)
        return None
    try:
        return canonical_task_by_id[task_id]
    except (KeyError, TypeError) as error:
        raise UnknownCanonicalTaskId(f"UNKNOWN_CANONICAL_TASK_ID: {task_id!r}") from error
