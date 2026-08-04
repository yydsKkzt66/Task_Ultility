"""Unified lifecycle ledger for onboard information products.

The ledger is accounting-only. It records queue and staged-request transitions
without changing reward, closure, safety, or router decision rules.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


LIFECYCLE_STATUSES = {
    "created",
    "pending",
    "queued",
    "scheduled",
    "delivered",
    "expired",
    "dropped",
    "blocked",
    "superseded",
}
TERMINAL_LIFECYCLE_STATUSES = {"delivered", "expired", "dropped", "superseded"}
ACTIVE_LIFECYCLE_STATUSES = LIFECYCLE_STATUSES - TERMINAL_LIFECYCLE_STATUSES


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _metadata(source: Any) -> Mapping[str, Any]:
    metadata = getattr(source, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _identity_key(task_id: str, product_type: str) -> str:
    return f"{task_id}:{product_type}"


def _format_components(components: Mapping[str, Any] | None) -> str:
    if not components:
        return ""
    pairs = []
    for key, value in sorted(components.items()):
        if isinstance(value, (int, float)):
            pairs.append(f"{key}={float(value):.6g}")
        else:
            pairs.append(f"{key}={value}")
    return ";".join(pairs)


@dataclass
class PayloadLifecycleRecord:
    lifecycle_id: str
    task_id: str
    product_type: str
    parent_product_type: str = ""
    request_id: str = ""
    value_level: str = ""
    product_requirement: str = ""
    calibrated_task_value: float = 0.0
    effective_task_value: float = 0.0
    residual_delta_information_utility: float = 0.0
    payload_size_mb: float = 0.0
    created_slot: int = -1
    deadline_slot: int = -1
    current_status: str = "created"
    previous_status: str = ""
    status_updated_slot: int = -1
    transition_reason: str = ""
    blocker_reason: str = ""
    priority_score: float = 0.0
    priority_components: dict[str, Any] = field(default_factory=dict)
    queue_id: str = ""
    staged_queue_id: str = ""
    delivered_slot: int = -1
    expired_slot: int = -1
    dropped_slot: int = -1
    retry_count: int = 0
    reevaluation_count: int = 0
    last_contact_window_id_seen: str = ""
    next_contact_window_id_candidate: str = ""
    leakage_safe_flag: bool = True
    resource_status: str = ""
    source: str = ""


class PayloadLifecycleLedger:
    """Transition log and current-state index for payload lifecycle accounting."""

    def __init__(self, slot_seconds: float = 1.0) -> None:
        self.slot_seconds = max(1.0e-9, float(slot_seconds))
        self.records: dict[str, PayloadLifecycleRecord] = {}
        self._seq = 0
        self._active_by_identity: dict[str, str] = {}
        self._by_request_id: dict[str, str] = {}
        self.transition_trace: list[dict[str, Any]] = []
        self.recompetition_trace: list[dict[str, Any]] = []
        self.priority_component_trace: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "records": deepcopy(self.records),
            "seq": self._seq,
            "active_by_identity": deepcopy(self._active_by_identity),
            "by_request_id": deepcopy(self._by_request_id),
            "transition_trace": deepcopy(self.transition_trace),
            "recompetition_trace": deepcopy(self.recompetition_trace),
            "priority_component_trace": deepcopy(self.priority_component_trace),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.records = deepcopy(snapshot["records"])
        self._seq = int(snapshot["seq"])
        self._active_by_identity = deepcopy(snapshot["active_by_identity"])
        self._by_request_id = deepcopy(snapshot["by_request_id"])
        self.transition_trace = deepcopy(snapshot["transition_trace"])
        self.recompetition_trace = deepcopy(snapshot["recompetition_trace"])
        self.priority_component_trace = deepcopy(snapshot["priority_component_trace"])

    def _new_id(self) -> str:
        self._seq += 1
        return f"pl_{self._seq:08d}"

    def active_lifecycle_id(self, task_id: str, product_type: str) -> str:
        lifecycle_id = self._active_by_identity.get(_identity_key(str(task_id), str(product_type)))
        if not lifecycle_id:
            return ""
        record = self.records.get(lifecycle_id)
        if record is None or record.current_status in TERMINAL_LIFECYCLE_STATUSES:
            return ""
        return lifecycle_id

    def get(self, lifecycle_id: str) -> PayloadLifecycleRecord | None:
        return self.records.get(str(lifecycle_id or ""))

    def create_record(
        self,
        *,
        task_id: str,
        product_type: str,
        slot: int,
        parent_product_type: str = "",
        request_id: str = "",
        value_level: str = "",
        product_requirement: str = "",
        calibrated_task_value: float = 0.0,
        effective_task_value: float = 0.0,
        residual_delta_information_utility: float = 0.0,
        payload_size_mb: float = 0.0,
        created_slot: int | None = None,
        deadline_slot: int = -1,
        priority_score: float = 0.0,
        priority_components: Mapping[str, Any] | None = None,
        queue_id: str = "",
        staged_queue_id: str = "",
        leakage_safe_flag: bool = True,
        source: str = "",
        set_active: bool = True,
        transition_reason: str = "payload_created",
    ) -> PayloadLifecycleRecord:
        lifecycle_id = self._new_id()
        record = PayloadLifecycleRecord(
            lifecycle_id=lifecycle_id,
            task_id=str(task_id),
            product_type=str(product_type),
            parent_product_type=str(parent_product_type or ""),
            request_id=str(request_id or ""),
            value_level=str(value_level or ""),
            product_requirement=str(product_requirement or ""),
            calibrated_task_value=float(calibrated_task_value),
            effective_task_value=float(effective_task_value or calibrated_task_value),
            residual_delta_information_utility=float(residual_delta_information_utility),
            payload_size_mb=float(payload_size_mb),
            created_slot=int(slot if created_slot is None else created_slot),
            deadline_slot=int(deadline_slot),
            current_status="created",
            status_updated_slot=int(slot),
            transition_reason=str(transition_reason or "payload_created"),
            priority_score=float(priority_score),
            priority_components=dict(priority_components or {}),
            queue_id=str(queue_id or ""),
            staged_queue_id=str(staged_queue_id or ""),
            leakage_safe_flag=bool(leakage_safe_flag),
            source=str(source or ""),
        )
        self.records[lifecycle_id] = record
        if record.request_id:
            self._by_request_id[record.request_id] = lifecycle_id
        if set_active:
            self._active_by_identity[_identity_key(record.task_id, record.product_type)] = lifecycle_id
        self._append_transition(record)
        return record

    def create_for_payload(self, payload: Any, *, slot: int, set_active: bool = True) -> PayloadLifecycleRecord:
        metadata = _metadata(payload)
        product_type = str(getattr(payload, "product_type", "") or getattr(payload, "payload_policy", "") or "")
        existing_id = str(metadata.get("lifecycle_id", ""))
        if set_active and existing_id and existing_id in self.records:
            return self.records[existing_id]
        deadline_s = _as_float(getattr(payload, "deadline_s", 0.0), 0.0)
        created_slot = _as_int(getattr(payload, "created_slot", slot), slot)
        deadline_slot = created_slot + max(0, int(deadline_s // self.slot_seconds))
        record = self.create_record(
            task_id=str(getattr(payload, "task_id", "")),
            product_type=product_type,
            slot=int(slot),
            parent_product_type=str(metadata.get("parent_product_plan", metadata.get("parent_payload_plan", ""))),
            request_id=str(metadata.get("staged_request_id", "")),
            value_level=str(metadata.get("value_level", "")),
            product_requirement=str(getattr(payload, "product_requirement", "") or metadata.get("product_requirement", "")),
            calibrated_task_value=_as_float(metadata.get("calibrated_task_value", getattr(payload, "task_value", 0.0))),
            effective_task_value=_as_float(getattr(payload, "task_value", metadata.get("task_value", 0.0))),
            residual_delta_information_utility=_as_float(
                getattr(payload, "incremental_information_utility", metadata.get("residual_delta_information_utility", 0.0))
            ),
            payload_size_mb=_as_float(getattr(payload, "payload_mb", 0.0)),
            deadline_slot=deadline_slot,
            priority_score=_as_float(getattr(payload, "priority_score", metadata.get("expected_residual_information_value", 0.0))),
            priority_components={},
            leakage_safe_flag=bool(metadata.get("leakage_safe_flag", True)),
            source=str(metadata.get("followup_source", metadata.get("followup_trigger", "ordinary_downlink_queue"))),
            set_active=set_active,
            transition_reason="payload_generated",
        )
        try:
            payload.metadata = {**dict(metadata), "lifecycle_id": record.lifecycle_id}
        except Exception:
            pass
        return record

    def create_for_staged_request(self, request: Any, *, slot: int, set_active: bool = True) -> PayloadLifecycleRecord:
        existing_id = str(getattr(request, "lifecycle_id", "") or "")
        if existing_id and existing_id in self.records:
            return self.records[existing_id]
        request_id = str(getattr(request, "request_id", ""))
        if request_id and request_id in self._by_request_id:
            record = self.records[self._by_request_id[request_id]]
            try:
                request.lifecycle_id = record.lifecycle_id
            except Exception:
                pass
            return record
        record = self.create_record(
            task_id=str(getattr(request, "task_id", "")),
            product_type=str(getattr(request, "product_type", "")),
            parent_product_type=str(getattr(request, "parent_product_type", "")),
            request_id=request_id,
            value_level=str(getattr(request, "value_level", "")),
            product_requirement=str(getattr(request, "product_requirement", "")),
            calibrated_task_value=_as_float(getattr(request, "calibrated_task_value", 0.0)),
            effective_task_value=_as_float(getattr(request, "calibrated_task_value", 0.0)),
            residual_delta_information_utility=_as_float(getattr(request, "residual_delta_information_utility", 0.0)),
            payload_size_mb=_as_float(getattr(request, "payload_size_mb", 0.0)),
            created_slot=_as_int(getattr(request, "created_slot", slot), slot),
            slot=slot,
            deadline_slot=_as_int(getattr(request, "deadline_slot", -1), -1),
            priority_score=_as_float(getattr(request, "priority_score", 0.0)),
            priority_components=dict(getattr(request, "priority_components", {}) or {}),
            staged_queue_id=str(getattr(request, "duplicate_key", "")),
            leakage_safe_flag=bool(getattr(request, "leakage_safe_flag", True)),
            source="staged_delivery_queue",
            set_active=set_active,
            transition_reason="staged_request_created",
        )
        try:
            request.lifecycle_id = record.lifecycle_id
        except Exception:
            pass
        return record

    def transition(
        self,
        lifecycle_id: str,
        status: str,
        *,
        slot: int,
        reason: str,
        blocker_reason: str = "",
        priority_score: float | None = None,
        priority_components: Mapping[str, Any] | None = None,
        queue_id: str = "",
        staged_queue_id: str = "",
        contact_window_id: str = "",
        next_contact_window_id_candidate: str = "",
        resource_status: str = "",
    ) -> PayloadLifecycleRecord | None:
        status = str(status or "")
        if status not in LIFECYCLE_STATUSES:
            raise ValueError(f"Unknown payload lifecycle status: {status}")
        record = self.records.get(str(lifecycle_id or ""))
        if record is None:
            return None
        previous = record.current_status
        record.previous_status = previous
        record.current_status = status
        record.status_updated_slot = int(slot)
        record.transition_reason = str(reason or status)
        record.blocker_reason = str(blocker_reason or "")
        if priority_score is not None:
            record.priority_score = float(priority_score)
        if priority_components is not None:
            record.priority_components = dict(priority_components)
        if queue_id:
            record.queue_id = str(queue_id)
        if staged_queue_id:
            record.staged_queue_id = str(staged_queue_id)
        if contact_window_id:
            record.last_contact_window_id_seen = str(contact_window_id)
        if next_contact_window_id_candidate:
            record.next_contact_window_id_candidate = str(next_contact_window_id_candidate)
        if resource_status:
            record.resource_status = str(resource_status)
        if status == "delivered":
            record.delivered_slot = int(slot)
            self._active_by_identity.pop(_identity_key(record.task_id, record.product_type), None)
        elif status == "expired":
            record.expired_slot = int(slot)
            self._active_by_identity.pop(_identity_key(record.task_id, record.product_type), None)
        elif status == "dropped":
            record.dropped_slot = int(slot)
            self._active_by_identity.pop(_identity_key(record.task_id, record.product_type), None)
        elif status == "superseded":
            self._active_by_identity.pop(_identity_key(record.task_id, record.product_type), None)
        elif status == "blocked":
            record.retry_count += 1
        elif status in {"pending", "queued", "scheduled"}:
            self._active_by_identity[_identity_key(record.task_id, record.product_type)] = record.lifecycle_id
        self._append_transition(record)
        return record

    def record_recompetition(
        self,
        *,
        slot: int,
        contact_window_id: str,
        task_id: str,
        product_type: str,
        previous_status: str,
        reevaluated_flag: bool,
        feasible_this_window: bool,
        priority_score: float,
        priority_rank: int,
        selected_for_delivery: bool,
        delivered: bool,
        blocker_reason: str,
        status_after_window: str,
        scenario: str = "",
        seed: int | str = "",
    ) -> None:
        lifecycle_id = self.active_lifecycle_id(task_id, product_type)
        if lifecycle_id and lifecycle_id in self.records:
            self.records[lifecycle_id].reevaluation_count += int(bool(reevaluated_flag))
        self.recompetition_trace.append(
            {
                "scenario": scenario,
                "seed": seed,
                "slot": int(slot),
                "contact_window_id": contact_window_id,
                "task_id": task_id,
                "product_type": product_type,
                "previous_status": previous_status,
                "reevaluated_flag": int(bool(reevaluated_flag)),
                "feasible_this_window": int(bool(feasible_this_window)),
                "priority_score": float(priority_score),
                "priority_rank": int(priority_rank),
                "selected_for_delivery": int(bool(selected_for_delivery)),
                "delivered": int(bool(delivered)),
                "blocker_reason": blocker_reason,
                "status_after_window": status_after_window,
                "lifecycle_id": lifecycle_id,
            }
        )

    def record_priority_components(self, row: Mapping[str, Any], *, source: str = "downlink_queue") -> None:
        product_type = str(row.get("product_type", row.get("payload_type", "")))
        task_id = str(row.get("task_id", ""))
        lifecycle_id = str(row.get("lifecycle_id", "")) or self.active_lifecycle_id(task_id, product_type)
        item = dict(row)
        item.update({"source": source, "lifecycle_id": lifecycle_id})
        self.priority_component_trace.append(item)

    def _append_transition(self, record: PayloadLifecycleRecord) -> None:
        self.transition_trace.append(self.record_to_row(record))

    @staticmethod
    def record_to_row(record: PayloadLifecycleRecord) -> dict[str, Any]:
        return {
            "lifecycle_id": record.lifecycle_id,
            "task_id": record.task_id,
            "product_type": record.product_type,
            "parent_product_type": record.parent_product_type,
            "request_id": record.request_id,
            "value_level": record.value_level,
            "product_requirement": record.product_requirement,
            "calibrated_task_value": record.calibrated_task_value,
            "effective_task_value": record.effective_task_value,
            "residual_delta_information_utility": record.residual_delta_information_utility,
            "payload_size_mb": record.payload_size_mb,
            "created_slot": record.created_slot,
            "deadline_slot": record.deadline_slot,
            "current_status": record.current_status,
            "previous_status": record.previous_status,
            "status_updated_slot": record.status_updated_slot,
            "transition_reason": record.transition_reason,
            "blocker_reason": record.blocker_reason,
            "priority_score": record.priority_score,
            "priority_components": _format_components(record.priority_components),
            "queue_id": record.queue_id,
            "staged_queue_id": record.staged_queue_id,
            "delivered_slot": "" if record.delivered_slot < 0 else record.delivered_slot,
            "expired_slot": "" if record.expired_slot < 0 else record.expired_slot,
            "dropped_slot": "" if record.dropped_slot < 0 else record.dropped_slot,
            "retry_count": record.retry_count,
            "reevaluation_count": record.reevaluation_count,
            "last_contact_window_id_seen": record.last_contact_window_id_seen,
            "next_contact_window_id_candidate": record.next_contact_window_id_candidate,
            "leakage_safe_flag": int(bool(record.leakage_safe_flag)),
            "resource_status": record.resource_status,
            "source": record.source,
        }

    def current_rows(self) -> list[dict[str, Any]]:
        return [self.record_to_row(record) for record in self.records.values()]

    def transition_summary_rows(self) -> list[dict[str, Any]]:
        counts: dict[tuple[str, str, str], int] = {}
        for row in self.transition_trace:
            key = (
                str(row.get("previous_status", "")),
                str(row.get("current_status", "")),
                str(row.get("transition_reason", "")),
            )
            counts[key] = counts.get(key, 0) + 1
        return [
            {
                "previous_status": previous,
                "current_status": current,
                "transition_reason": reason,
                "count": count,
            }
            for (previous, current, reason), count in sorted(counts.items())
        ]

    def orphan_check_rows(
        self,
        *,
        queue_items: Iterable[Any] = (),
        staged_requests: Iterable[Any] = (),
    ) -> list[dict[str, Any]]:
        queue_lifecycle_ids = {
            str(_metadata(item).get("lifecycle_id", ""))
            for item in queue_items
            if str(_metadata(item).get("lifecycle_id", ""))
        }
        staged_lifecycle_ids = {
            str(getattr(request, "lifecycle_id", ""))
            for request in staged_requests
            if str(getattr(request, "lifecycle_id", ""))
        }
        rows = []
        for record in self.records.values():
            in_queue = record.lifecycle_id in queue_lifecycle_ids
            in_staged = record.lifecycle_id in staged_lifecycle_ids
            terminal = record.current_status in TERMINAL_LIFECYCLE_STATUSES
            missing_reason = ""
            orphan = 0
            if record.current_status in {"queued", "scheduled"} and not in_queue and not in_staged and not terminal:
                orphan = 1
                missing_reason = "queued_or_scheduled_lifecycle_missing_queue_item"
            elif record.current_status in {"pending", "blocked"} and record.product_type in {"roi", "roi_context"} and not in_queue and not in_staged:
                orphan = 1
                missing_reason = "pending_or_blocked_residual_missing_staged_backlog"
            elif record.current_status == "delivered" and record.delivered_slot < 0:
                orphan = 1
                missing_reason = "delivered_missing_delivered_slot"
            elif record.current_status == "expired" and (record.expired_slot < 0 or not record.transition_reason):
                orphan = 1
                missing_reason = "expired_missing_slot_or_reason"
            elif record.current_status == "dropped" and (record.dropped_slot < 0 or not record.transition_reason):
                orphan = 1
                missing_reason = "dropped_missing_slot_or_reason"
            rows.append(
                {
                    "lifecycle_id": record.lifecycle_id,
                    "task_id": record.task_id,
                    "product_type": record.product_type,
                    "current_status": record.current_status,
                    "in_downlink_queue": int(in_queue),
                    "in_staged_queue": int(in_staged),
                    "orphan_flag": orphan,
                    "orphan_reason": missing_reason,
                    "transition_reason": record.transition_reason,
                    "blocker_reason": record.blocker_reason,
                }
            )
        return rows
