"""contact-aware closed-loop downlink queue銆?"""

from __future__ import annotations

from copy import deepcopy
import math

from src.common.constants import (
    DELIVERY_STATUS_DELIVERED,
    DELIVERY_STATUS_DEFERRED,
    DELIVERY_STATUS_DROPPED,
    DELIVERY_STATUS_EMPTY,
    DELIVERY_STATUS_EXPIRED,
    DELIVERY_STATUS_QUEUED,
    EPSILON,
)
from src.common.types import DownlinkResult, GeneratedPayload, LinkTrace, QueueItem
from src.link.contact_window import compute_available_capacity_mb
from src.payload.payload_lifecycle_ledger import PayloadLifecycleLedger


class DownlinkQueue:
    """utility-priority contact-aware downlink queue銆?"""

    def __init__(
        self,
        queue_max_mb: float,
        slot_seconds: float,
        downlink_efficiency: float = 1.0,
        queue_priority_tau_d: float = 60.0,
        queue_age_bonus: float = 0.01,
        queue_meta_first_bonus: float = 0.25,
        queue_high_value_bonus: float = 0.20,
        continue_request_priority_bonus: float = 0.0,
        queue_pressure_priority_penalty: float = 0.0,
        lifecycle_ledger: PayloadLifecycleLedger | None = None,
    ) -> None:
        self.queue_max_mb = queue_max_mb
        self.slot_seconds = slot_seconds
        self.downlink_efficiency = downlink_efficiency
        self.queue_priority_tau_d = queue_priority_tau_d
        self.queue_age_bonus = queue_age_bonus
        self.queue_meta_first_bonus = queue_meta_first_bonus
        self.queue_high_value_bonus = queue_high_value_bonus
        self.continue_request_priority_bonus = continue_request_priority_bonus
        self.queue_pressure_priority_penalty = queue_pressure_priority_penalty
        self.lifecycle_ledger = lifecycle_ledger
        self.items: list[QueueItem] = []
        self.last_priority_component_log: list[dict[str, object]] = []

    @property
    def queue_mb(self) -> float:
        return sum(item.remaining_mb for item in self.items)

    def _current_time_s(self, slot_index: int) -> float:
        return float(slot_index) * self.slot_seconds

    def _update_waiting_time(self, item: QueueItem, current_time_s: float) -> None:
        item.waiting_time_s = max(0.0, current_time_s - item.generated_time_s)

    def _priority_components(self, item: QueueItem, current_time_s: float) -> dict[str, float]:
        self._update_waiting_time(item, current_time_s)
        utility_density = item.remaining_utility / max(item.remaining_mb, EPSILON)
        slack = max(0.0, item.deadline_time_s - current_time_s)
        urgency = math.exp(-slack / max(self.queue_priority_tau_d, EPSILON))
        age_bonus = self.queue_age_bonus * item.waiting_time_s
        metadata_bonus = self.queue_meta_first_bonus if item.is_metadata else 0.0
        high_value_bonus = self.queue_high_value_bonus if item.is_high_value else 0.0
        continue_bonus = float(item.continue_request_priority_bonus or self.continue_request_priority_bonus) if item.is_residual else 0.0
        proactive_bonus = continue_bonus if item.followup_trigger == "proactive_followup" else 0.0
        queue_pressure_penalty = self.queue_pressure_priority_penalty * (self.queue_mb / max(self.queue_max_mb, EPSILON))
        utility_density_component = utility_density * urgency
        total = utility_density_component + age_bonus + metadata_bonus + high_value_bonus + continue_bonus - queue_pressure_penalty
        components = {
            "queue_priority_total": total,
            "utility_density_component": utility_density_component,
            "utility_density_raw": utility_density,
            "urgency_component": urgency,
            "age_component": age_bonus,
            "metadata_bonus_component": metadata_bonus,
            "high_value_bonus_component": high_value_bonus,
            "continue_request_bonus_component": continue_bonus,
            "proactive_queue_bonus_component": proactive_bonus,
            "queue_pressure_penalty_component": queue_pressure_penalty,
            "remaining_deadline_s": slack,
        }
        if bool(item.metadata.get("staged_delivery_queue_enabled", False)):
            deadline_span_s = max(
                EPSILON,
                float(item.metadata.get("staged_deadline_s", item.deadline_time_s - item.generated_time_s) or 0.0),
            )
            linear_deadline_urgency = max(0.0, min(1.0, 1.0 - slack / deadline_span_s))
            staged_density = (
                float(item.metadata.get("staged_V_eff", item.task_value) or 0.0)
                * float(item.metadata.get("staged_delta_u_I", item.incremental_utility) or 0.0)
                * float(item.metadata.get("staged_detection_quality", 1.0) or 1.0)
                * float(item.metadata.get("staged_timeliness_factor", 1.0) or 1.0)
                * float(item.metadata.get("staged_product_requirement_weight", 1.0) or 1.0)
                * float(item.metadata.get("staged_value_level_factor", 1.0) or 1.0)
                / max(float(item.metadata.get("staged_payload_size_mb", item.payload_size_mb) or item.payload_size_mb), EPSILON)
            )
            deadline_bonus = float(item.metadata.get("staged_deadline_urgency_lambda", 0.0) or 0.0) * linear_deadline_urgency
            fixed_bonus = float(item.metadata.get("staged_fixed_bonus_debug", 0.0) or 0.0)
            staged_total = max(0.0, staged_density) + max(0.0, deadline_bonus) + fixed_bonus
            if bool(item.metadata.get("guarantee_priority_enabled", False)):
                staged_total = max(0.0, float(item.metadata.get("guarantee_priority_score", staged_total) or staged_total))
            components.update(
                {
                    "queue_priority_total": staged_total,
                    "staged_priority_total": staged_total,
                    "staged_value_density_priority": max(0.0, staged_density),
                    "staged_deadline_urgency": linear_deadline_urgency,
                    "staged_deadline_bonus": max(0.0, deadline_bonus),
                    "staged_product_requirement_weight": float(
                        item.metadata.get("staged_product_requirement_weight", 1.0) or 1.0
                    ),
                    "staged_value_level_factor": float(item.metadata.get("staged_value_level_factor", 1.0) or 1.0),
                    "staged_fixed_bonus_debug": fixed_bonus,
                    "guarantee_priority_enabled": int(bool(item.metadata.get("guarantee_priority_enabled", False))),
                    "guarantee_priority_score": float(item.metadata.get("guarantee_priority_score", 0.0) or 0.0),
                    "guarantee_debt_z": float(item.metadata.get("guarantee_debt_z", 0.0) or 0.0),
                    "guarantee_Z_age": float(item.metadata.get("guarantee_Z_age", 0.0) or 0.0),
                    "guarantee_Z_miss": float(item.metadata.get("guarantee_Z_miss", 0.0) or 0.0),
                    "guarantee_Z_total": float(item.metadata.get("guarantee_Z_total", item.metadata.get("guarantee_debt_z", 0.0)) or 0.0),
                    "guarantee_progress_gate": float(item.metadata.get("guarantee_progress_gate", 0.0) or 0.0),
                    "guarantee_G_i_p": float(item.metadata.get("guarantee_G_i_p", item.metadata.get("guarantee_progress_gate", 0.0)) or 0.0),
                    "guarantee_term": float(item.metadata.get("guarantee_term", 0.0) or 0.0),
                    "guarantee_rank_without": int(float(item.metadata.get("guarantee_rank_without", 0) or 0)),
                    "guarantee_rank_with": int(float(item.metadata.get("guarantee_rank_with", 0) or 0)),
                    "guarantee_assisted_candidate": int(bool(item.metadata.get("guarantee_assisted_candidate", False))),
                    "guarantee_priority_without_guarantee": float(
                        item.metadata.get("guarantee_priority_without_guarantee", 0.0) or 0.0
                    ),
                }
            )
            if bool(item.metadata.get("staged_residual_preempts_ordinary", False)):
                preemption_floor = float(item.metadata.get("staged_preemption_priority_floor", 1000.0) or 0.0)
                components["queue_priority_total"] = preemption_floor + staged_total
                components["staged_preemption_floor_component"] = preemption_floor
            else:
                components["staged_preemption_floor_component"] = 0.0
        return components

    def _compute_priority(self, item: QueueItem, current_time_s: float) -> float:
        return self._priority_components(item, current_time_s)["queue_priority_total"]

    def _make_queue_item(self, payload: GeneratedPayload) -> QueueItem:
        generated_time_s = payload.created_slot * self.slot_seconds
        deadline_time_s = generated_time_s + payload.deadline_s
        product_type = payload.product_type or payload.payload_policy
        return QueueItem(
            task_id=payload.task_id,
            generated_time_s=generated_time_s,
            deadline_time_s=deadline_time_s,
            payload_size_mb=payload.payload_mb,
            remaining_mb=payload.payload_mb,
            remaining_utility=payload.remaining_utility,
            task_value=payload.task_value,
            uncertainty=payload.uncertainty,
            payload_type=product_type,
            waiting_time_s=0.0,
            is_metadata=payload.is_metadata,
            is_high_value=payload.is_high_value,
            metadata=dict(payload.metadata),
            payload_plan=payload.payload_plan or payload.payload_policy,
            product_type=product_type,
            product_requirement=payload.product_requirement,
            incremental_utility=payload.incremental_utility_estimate,
            first_alert_product=(product_type == "metadata"),
            duplicate_product_attempt=payload.duplicate_product_attempt,
            zero_increment_product=payload.zero_increment_product,
            is_residual=payload.is_residual,
            duplicate_identity_key=payload.duplicate_identity_key or f"{payload.task_id}:{product_type}",
            continue_request_priority_bonus=payload.continue_request_priority_bonus,
            followup_trigger=payload.followup_trigger or str(payload.metadata.get("followup_trigger", "")),
        )

    @staticmethod
    def _payload_id(item: QueueItem) -> str:
        return f"{item.task_id}:{item.product_type or item.payload_type}:{item.generated_time_s:.6f}"

    def _payload_queue_id(self, payload: GeneratedPayload) -> str:
        product_type = payload.product_type or payload.payload_policy
        generated_time_s = payload.created_slot * self.slot_seconds
        return f"{payload.task_id}:{product_type}:{generated_time_s:.6f}"

    def _ensure_payload_lifecycle_record(self, payload: GeneratedPayload, slot_index: int, *, duplicate: bool) -> str:
        if self.lifecycle_ledger is None:
            return ""
        record = self.lifecycle_ledger.create_for_payload(payload, slot=slot_index, set_active=not duplicate)
        payload.metadata = {**dict(payload.metadata), "lifecycle_id": record.lifecycle_id}
        if record.current_status == "created":
            self.lifecycle_ledger.transition(
                record.lifecycle_id,
                "pending",
                slot=slot_index,
                reason="payload_generated_pending_queue_admission",
                queue_id=self._payload_queue_id(payload),
            )
        return record.lifecycle_id

    def _transition_item(
        self,
        item: QueueItem,
        status: str,
        *,
        slot_index: int,
        reason: str,
        blocker_reason: str = "",
        contact_window_id: str = "",
        priority_components: dict[str, float] | None = None,
    ) -> None:
        if self.lifecycle_ledger is None:
            return
        lifecycle_id = str(item.metadata.get("lifecycle_id", ""))
        if not lifecycle_id:
            return
        self.lifecycle_ledger.transition(
            lifecycle_id,
            status,
            slot=slot_index,
            reason=reason,
            blocker_reason=blocker_reason,
            priority_score=None if priority_components is None else float(priority_components.get("queue_priority_total", 0.0)),
            priority_components=priority_components,
            queue_id=self._payload_id(item),
            contact_window_id=contact_window_id,
        )

    def enqueue(self, payload: GeneratedPayload) -> bool:
        if payload.payload_mb <= 0.0:
            return False
        product_type = payload.product_type or payload.payload_policy
        if any(item.task_id == payload.task_id and (item.product_type or item.payload_type) == product_type for item in self.items):
            return False
        self.items.append(self._make_queue_item(payload))
        return True

    def _drop_expired(
        self, current_time_s: float, current_task_id: str
    ) -> tuple[float, int, list[str], list[str], list[str], list[dict[str, object]], int, int]:
        expired_current = 0.0
        expired_count = 0
        residual_expired_count = 0
        proactive_expired_count = 0
        expired_task_ids: list[str] = []
        expired_payload_ids: list[str] = []
        expired_product_types: list[str] = []
        expired_payload_metadata: list[dict[str, object]] = []
        kept_items: list[QueueItem] = []
        for item in self.items:
            self._update_waiting_time(item, current_time_s)
            if item.deadline_time_s < current_time_s - EPSILON:
                self._transition_item(
                    item,
                    "expired",
                    slot_index=int(current_time_s // self.slot_seconds),
                    reason="deadline_passed",
                    blocker_reason="deadline_passed",
                )
                expired_count += 1
                expired_task_ids.append(item.task_id)
                expired_payload_ids.append(self._payload_id(item))
                expired_product_types.append(item.product_type or item.payload_type)
                expired_payload_metadata.append(dict(item.metadata))
                residual_expired_count += int(item.is_residual)
                proactive_expired_count += int(item.is_residual and item.followup_trigger == "proactive_followup")
                if item.task_id == current_task_id:
                    expired_current += item.remaining_mb
                continue
            kept_items.append(item)
        self.items = kept_items
        return (
            expired_current,
            expired_count,
            expired_task_ids,
            expired_payload_ids,
            expired_product_types,
            expired_payload_metadata,
            residual_expired_count,
            proactive_expired_count,
        )

    def _sort_items(self, current_time_s: float) -> None:
        self.items.sort(
            key=lambda item: (
                -self._compute_priority(item, current_time_s),
                item.deadline_time_s,
                item.remaining_mb,
            )
        )

    def _drop_overflow_items(
        self,
        current_task_id: str,
        *,
        slot_index: int = -1,
        contact_window_id: str = "",
    ) -> tuple[float, int, list[str], list[str], list[str], list[dict[str, object]], int, int]:
        dropped_current = 0.0
        dropped_count = 0
        residual_dropped_count = 0
        proactive_dropped_count = 0
        dropped_task_ids: list[str] = []
        dropped_payload_ids: list[str] = []
        dropped_product_types: list[str] = []
        dropped_payload_metadata: list[dict[str, object]] = []
        while self.queue_mb > self.queue_max_mb + EPSILON and self.items:
            item = self.items.pop()
            self._transition_item(
                item,
                "dropped",
                slot_index=slot_index,
                reason="queue_overflow_lowest_priority_evicted",
                blocker_reason="queue_overflow",
                contact_window_id=contact_window_id,
            )
            dropped_count += 1
            dropped_task_ids.append(item.task_id)
            dropped_payload_ids.append(self._payload_id(item))
            dropped_product_types.append(item.product_type or item.payload_type)
            dropped_payload_metadata.append(dict(item.metadata))
            residual_dropped_count += int(item.is_residual)
            proactive_dropped_count += int(item.is_residual and item.followup_trigger == "proactive_followup")
            if item.task_id == current_task_id:
                dropped_current += item.remaining_mb
        return (
            dropped_current,
            dropped_count,
            dropped_task_ids,
            dropped_payload_ids,
            dropped_product_types,
            dropped_payload_metadata,
            residual_dropped_count,
            proactive_dropped_count,
        )

    def process_slot(
        self,
        slot_index: int,
        payload: GeneratedPayload | None,
        link_trace: LinkTrace,
        capacity_override_mb: float | None = None,
        contact_window_id: str = "",
    ) -> DownlinkResult:
        current_time_s = self._current_time_s(slot_index)
        self.last_priority_component_log = []
        queue_before = self.queue_mb
        current_task_id = payload.task_id if payload is not None else "__idle__"
        (
            expired_current,
            expired_count,
            expired_task_ids,
            expired_payload_ids,
            expired_product_types,
            expired_payload_metadata,
            residual_expired_count,
            proactive_expired_count,
        ) = self._drop_expired(current_time_s, current_task_id)
        duplicate_product_attempt_count = 0
        zero_increment_product_count = 0
        residual_enqueue_count = 0
        duplicate_residual_suppression_count = 0
        proactive_followup_enqueue_count = 0
        proactive_duplicate_suppression_count = 0

        payload_lifecycle_id = ""
        if payload is not None and payload.payload_policy != "defer" and payload.payload_mb > 0.0:
            product_type = payload.product_type or payload.payload_policy
            duplicate_before_enqueue = any(
                item.task_id == payload.task_id and (item.product_type or item.payload_type) == product_type
                for item in self.items
            )
            payload_lifecycle_id = self._ensure_payload_lifecycle_record(
                payload,
                slot_index,
                duplicate=duplicate_before_enqueue,
            )

        if payload is None:
            status = DELIVERY_STATUS_EMPTY
        elif payload.payload_policy == "defer":
            status = DELIVERY_STATUS_DEFERRED
        elif payload.payload_mb <= 0.0:
            status = DELIVERY_STATUS_EMPTY
        else:
            duplicate_product_attempt_count += int(payload.duplicate_product_attempt)
            zero_increment_product_count += int(payload.zero_increment_product)
            enqueued = self.enqueue(payload)
            duplicate_product_attempt_count += int(not enqueued)
            duplicate_residual_suppression_count += int(payload.is_residual and not enqueued)
            residual_enqueue_count += int(payload.is_residual and enqueued)
            is_proactive = payload.is_residual and (
                payload.followup_trigger == "proactive_followup"
                or str(payload.metadata.get("followup_trigger", "")) == "proactive_followup"
            )
            proactive_followup_enqueue_count += int(is_proactive and enqueued)
            proactive_duplicate_suppression_count += int(is_proactive and not enqueued)
            status = DELIVERY_STATUS_QUEUED if enqueued else DELIVERY_STATUS_DROPPED
            if self.lifecycle_ledger is not None and payload_lifecycle_id:
                if enqueued:
                    self.lifecycle_ledger.transition(
                        payload_lifecycle_id,
                        "queued",
                        slot=slot_index,
                        reason="admitted_to_downlink_queue",
                        queue_id=self._payload_queue_id(payload),
                        contact_window_id=contact_window_id,
                    )
                else:
                    self.lifecycle_ledger.transition(
                        payload_lifecycle_id,
                        "dropped",
                        slot=slot_index,
                        reason="duplicate_pending_queue_item",
                        blocker_reason="duplicate_product_already_queued",
                        queue_id=self._payload_queue_id(payload),
                        contact_window_id=contact_window_id,
                    )

        self._sort_items(current_time_s)
        (
            dropped_current,
            dropped_count,
            dropped_task_ids,
            dropped_payload_ids,
            dropped_product_types,
            dropped_payload_metadata,
            residual_dropped_count,
            proactive_dropped_count,
        ) = self._drop_overflow_items(current_task_id, slot_index=slot_index, contact_window_id=contact_window_id)
        self._sort_items(current_time_s)
        priority_log: list[dict[str, object]] = []
        for order, item in enumerate(self.items):
            components = self._priority_components(item, current_time_s)
            lifecycle_id = str(item.metadata.get("lifecycle_id", ""))
            status_before = "queued"
            if self.lifecycle_ledger is not None and lifecycle_id:
                record = self.lifecycle_ledger.get(lifecycle_id)
                if record is not None:
                    status_before = record.current_status
            priority_log.append(
                {
                    "slot": slot_index,
                    "time_s": current_time_s,
                    "task_id": item.task_id,
                    "payload_id": self._payload_id(item),
                    "lifecycle_id": str(item.metadata.get("lifecycle_id", "")),
                    "status_before": status_before,
                    "payload_type": item.product_type or item.payload_type,
                    "payload_plan": item.payload_plan or item.payload_type,
                    "product_requirement": item.product_requirement,
                    "payload_size_mb": item.payload_size_mb,
                    "remaining_mb": item.remaining_mb,
                    "task_value": item.task_value,
                    "high_value_flag": int(item.is_high_value),
                    "value_level": str(item.metadata.get("value_level", "")),
                    "is_residual": int(bool(item.is_residual)),
                    "followup_trigger": item.followup_trigger,
                    "is_staged_delivery_queue": int(bool(item.metadata.get("staged_delivery_queue_enabled", False))),
                    "staged_request_id": str(item.metadata.get("staged_request_id", "")),
                    "staged_duplicate_key": str(item.metadata.get("staged_duplicate_key", "")),
                    "staged_residual_preempts_ordinary": int(
                        bool(item.metadata.get("staged_residual_preempts_ordinary", False))
                    ),
                    "staged_priority_score": float(item.metadata.get("staged_priority_score", 0.0) or 0.0),
                    "deadline_s": item.deadline_time_s - item.generated_time_s,
                    "remaining_deadline_s": components["remaining_deadline_s"],
                    "remaining_utility": item.remaining_utility,
                    "waiting_time_s": item.waiting_time_s,
                    "queue_order_before_downlink": order,
                    "contact_available": int(link_trace.in_contact),
                    "link_rate_bps": max(0.0, link_trace.throughput_mbps * 1_000_000.0),
                    "selected_for_downlink": 0,
                    "delivered_this_slot": 0,
                    "expired_this_slot": 0,
                    **components,
                }
            )

        capacity_mb = (
            compute_available_capacity_mb(link_trace, self.slot_seconds, self.downlink_efficiency)
            if capacity_override_mb is None
            else max(0.0, float(capacity_override_mb))
        )
        initial_capacity_mb = capacity_mb
        transmitted_total = 0.0
        transmitted_current = 0.0
        transmitted_task_ids: list[str] = []
        transmitted_payload_ids: list[str] = []
        delivered_task_ids: list[str] = []
        delivered_payload_ids: list[str] = []
        delivered_product_types: list[str] = []
        delivered_incremental_utilities: list[float] = []
        residual_delivered_incremental_utilities: list[float] = []
        proactive_delivered_incremental_utilities: list[float] = []
        residual_queue_wait_times_s: list[float] = []
        delivered_followup_triggers: list[str] = []
        delivered_payload_metadata: list[dict[str, object]] = []
        delivered_task_event_times_s: list[float] = []
        delivered_item_count = 0
        residual_delivery_count = 0
        proactive_followup_delivery_count = 0
        metadata_delivered_count = 0
        high_value_metadata_delivered_count = 0
        delivered_high_value_payload_count = 0
        delivered_waiting_time_s = 0.0

        if capacity_mb > 0.0:
            new_items: list[QueueItem] = []
            rate_mb_s = (
                max(0.0, link_trace.throughput_mbps * self.downlink_efficiency / 8.0)
                if link_trace.in_contact
                else 0.0
            )
            elapsed_tx_s = 0.0
            for item in self.items:
                if capacity_mb <= 0.0:
                    new_items.append(item)
                    continue

                self._update_waiting_time(item, current_time_s)
                before_mb = item.remaining_mb
                if before_mb <= EPSILON:
                    continue

                item_payload_id = self._payload_id(item)
                components = self._priority_components(item, current_time_s)
                self._transition_item(
                    item,
                    "scheduled",
                    slot_index=slot_index,
                    reason="selected_by_contact_window_scheduler",
                    contact_window_id=contact_window_id,
                    priority_components=components,
                )
                tx_mb = min(before_mb, capacity_mb)
                utility_fraction = tx_mb / max(before_mb, EPSILON)
                item.remaining_mb -= tx_mb
                item.remaining_utility = max(0.0, item.remaining_utility * (1.0 - utility_fraction))
                capacity_mb -= tx_mb
                transmitted_total += tx_mb
                transmitted_task_ids.append(item.task_id)
                transmitted_payload_ids.append(item_payload_id)
                if item.task_id == current_task_id:
                    transmitted_current += tx_mb
                if rate_mb_s > EPSILON:
                    elapsed_tx_s += tx_mb / rate_mb_s

                if item.remaining_mb > EPSILON:
                    self._transition_item(
                        item,
                        "queued",
                        slot_index=slot_index,
                        reason="partial_downlink_remaining",
                        contact_window_id=contact_window_id,
                        priority_components=components,
                    )
                    new_items.append(item)
                else:
                    delivered_item_count += 1
                    delivered_task_ids.append(item.task_id)
                    delivered_payload_ids.append(self._payload_id(item))
                    delivered_product_types.append(item.product_type or item.payload_type)
                    delivered_incremental_utilities.append(item.incremental_utility)
                    delivered_payload_metadata.append(dict(item.metadata))
                    delivered_task_event_times_s.append(current_time_s + elapsed_tx_s)
                    delivered_waiting_time_s += item.waiting_time_s
                    if item.is_residual:
                        residual_delivery_count += 1
                        residual_delivered_incremental_utilities.append(item.incremental_utility)
                        residual_queue_wait_times_s.append(item.waiting_time_s)
                        delivered_followup_triggers.append(item.followup_trigger)
                        if item.followup_trigger == "proactive_followup":
                            proactive_followup_delivery_count += 1
                            proactive_delivered_incremental_utilities.append(item.incremental_utility)
                    if item.is_metadata:
                        metadata_delivered_count += 1
                    if item.is_high_value:
                        delivered_high_value_payload_count += 1
                        if item.is_metadata:
                            high_value_metadata_delivered_count += 1
                    self._transition_item(
                        item,
                        "delivered",
                        slot_index=slot_index,
                        reason="full_payload_downlinked",
                        contact_window_id=contact_window_id,
                        priority_components=components,
                    )
            self.items = new_items

        transmitted_ids = set(transmitted_payload_ids)
        delivered_ids = set(delivered_payload_ids)
        expired_ids = set(expired_payload_ids)
        for row in priority_log:
            payload_id = str(row["payload_id"])
            task_id = str(row["task_id"])
            product_type = str(row["payload_type"])
            selected = payload_id in transmitted_ids
            delivered = payload_id in delivered_ids
            expired = payload_id in expired_ids
            row["selected_for_downlink"] = int(selected)
            row["delivered_this_slot"] = int(delivered)
            row["expired_this_slot"] = int(expired)
            if self.lifecycle_ledger is not None:
                status_after = "delivered" if delivered else ("queued" if selected else str(row.get("status_before", "queued") or "queued"))
                blocker = ""
                feasible = bool(link_trace.in_contact and initial_capacity_mb > EPSILON)
                if not selected and not delivered and not expired:
                    matching_item = next((item for item in self.items if self._payload_id(item) == payload_id), None)
                    if matching_item is not None:
                        components = {
                            key: value
                            for key, value in row.items()
                            if key.endswith("_component") or key in {"queue_priority_total", "staged_priority_total"}
                        }
                        blocker = "no_contact_opportunity" if not link_trace.in_contact else "insufficient_contact_capacity"
                        status_after = "blocked"
                        self._transition_item(
                            matching_item,
                            "blocked",
                            slot_index=slot_index,
                            reason=blocker,
                            blocker_reason=blocker,
                            contact_window_id=contact_window_id,
                            priority_components=components,
                        )
                self.lifecycle_ledger.record_recompetition(
                    slot=slot_index,
                    contact_window_id=contact_window_id,
                    task_id=task_id,
                    product_type=product_type,
                    previous_status=str(row.get("status_before", "queued") or "queued"),
                    reevaluated_flag=True,
                    feasible_this_window=feasible,
                    priority_score=float(row.get("queue_priority_total", 0.0) or 0.0),
                    priority_rank=int(row.get("queue_order_before_downlink", 0) or 0),
                    selected_for_delivery=selected,
                    delivered=delivered,
                    blocker_reason=blocker,
                    status_after_window=status_after,
                )
                self.lifecycle_ledger.record_priority_components(row, source="downlink_queue")
        self.last_priority_component_log = priority_log

        if payload is not None and payload.payload_mb > 0.0:
            if transmitted_current >= payload.payload_mb - EPSILON:
                status = DELIVERY_STATUS_DELIVERED
            elif any(item.task_id == current_task_id for item in self.items):
                status = DELIVERY_STATUS_QUEUED
            elif expired_current > 0.0:
                status = DELIVERY_STATUS_EXPIRED
            elif dropped_current > 0.0:
                status = DELIVERY_STATUS_DROPPED

        queue_after = self.queue_mb
        queue_reference_mb = max(queue_before, queue_after)
        queue_latency_s = (
            0.0
            if queue_reference_mb <= 0.0
            else self.slot_seconds * min(1.0, queue_reference_mb / max(self.queue_max_mb, EPSILON))
        )

        return DownlinkResult(
            S_down_total_mb=transmitted_total,
            S_down_current_task_mb=transmitted_current,
            generated_payload_current_mb=0.0 if payload is None else payload.payload_mb,
            dropped_current_task_mb=dropped_current + expired_current,
            queue_next_mb=queue_after,
            delta_queue_mb=queue_after - queue_before,
            queue_latency_s=queue_latency_s,
            queue_item_count=len(self.items),
            current_task_delivery_status=status,
            delivered_item_count=delivered_item_count,
            expired_item_count=expired_count,
            dropped_item_count=dropped_count,
            metadata_delivered_count=metadata_delivered_count,
            high_value_metadata_delivered_count=high_value_metadata_delivered_count,
            delivered_high_value_payload_count=delivered_high_value_payload_count,
            delivered_waiting_time_s=delivered_waiting_time_s,
            transmitted_task_ids=transmitted_task_ids,
            delivered_task_ids=delivered_task_ids,
            expired_task_ids=expired_task_ids,
            dropped_task_ids=dropped_task_ids,
            delivered_task_event_times_s=delivered_task_event_times_s,
            expired_task_event_times_s=[current_time_s for _ in expired_task_ids],
            dropped_task_event_times_s=[current_time_s for _ in dropped_task_ids],
            delivered_payload_ids=delivered_payload_ids,
            delivered_product_types=delivered_product_types,
            expired_payload_ids=expired_payload_ids,
            expired_product_types=expired_product_types,
            dropped_payload_ids=dropped_payload_ids,
            dropped_product_types=dropped_product_types,
            delivered_incremental_utilities=delivered_incremental_utilities,
            duplicate_product_attempt_count=duplicate_product_attempt_count,
            zero_increment_product_count=zero_increment_product_count,
            residual_enqueue_count=residual_enqueue_count,
            residual_delivery_count=residual_delivery_count,
            residual_expired_count=residual_expired_count,
            residual_dropped_count=residual_dropped_count,
            duplicate_residual_suppression_count=duplicate_residual_suppression_count,
            followup_payload_mb=payload.payload_mb if payload is not None and payload.is_residual else 0.0,
            residual_delivered_incremental_utilities=residual_delivered_incremental_utilities,
            residual_queue_wait_times_s=residual_queue_wait_times_s,
            delivered_followup_triggers=delivered_followup_triggers,
            proactive_followup_enqueue_count=proactive_followup_enqueue_count,
            proactive_followup_delivery_count=proactive_followup_delivery_count,
            proactive_followup_expired_count=proactive_expired_count,
            proactive_followup_dropped_count=proactive_dropped_count,
            proactive_duplicate_suppression_count=proactive_duplicate_suppression_count,
            proactive_followup_payload_mb=(
                payload.payload_mb
                if payload is not None
                and payload.is_residual
                and (
                    payload.followup_trigger == "proactive_followup"
                    or str(payload.metadata.get("followup_trigger", "")) == "proactive_followup"
                )
                else 0.0
            ),
            proactive_delivered_incremental_utilities=proactive_delivered_incremental_utilities,
            delivered_payload_metadata=delivered_payload_metadata,
            expired_payload_metadata=expired_payload_metadata,
            dropped_payload_metadata=dropped_payload_metadata,
        )

    def preview_slot(
        self,
        slot_index: int,
        payload: GeneratedPayload | None,
        link_trace: LinkTrace,
        capacity_override_mb: float | None = None,
    ) -> DownlinkResult:
        """Return a process_slot result without changing the stored queue."""

        items_snapshot = deepcopy(self.items)
        ledger = self.lifecycle_ledger
        try:
            self.lifecycle_ledger = None
            return self.process_slot(
                slot_index=slot_index,
                payload=payload,
                link_trace=link_trace,
                capacity_override_mb=capacity_override_mb,
            )
        finally:
            self.items = items_snapshot
            self.lifecycle_ledger = ledger
