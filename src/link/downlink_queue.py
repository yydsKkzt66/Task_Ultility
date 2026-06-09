"""contact-aware closed-loop downlink queue。"""

from __future__ import annotations

from src.common.constants import (
    DELIVERY_STATUS_DELIVERED,
    DELIVERY_STATUS_DEFERRED,
    DELIVERY_STATUS_DROPPED,
    DELIVERY_STATUS_EMPTY,
    DELIVERY_STATUS_QUEUED,
)
from src.common.types import DownlinkResult, GeneratedPayload, LinkTrace, QueueItem
from src.link.contact_window import compute_available_capacity_mb


class DownlinkQueue:
    """优先级驱动的下传队列。

    它不只是一个缓存容器，而是算通协同闭环中的关键通信模块：
    1. 感知 contact window；
    2. 感知链路容量；
    3. 维护历史 payload 队列；
    4. 对当前任务和历史任务做统一调度。
    """

    def __init__(self, queue_max_mb: float, slot_seconds: float) -> None:
        self.queue_max_mb = queue_max_mb
        self.slot_seconds = slot_seconds
        self.items: list[QueueItem] = []

    @property
    def queue_mb(self) -> float:
        """返回当前物理队列大小。"""

        return sum(item.remaining_mb for item in self.items)

    def _drop_expired(self, slot_index: int, current_task_id: str) -> float:
        dropped_current = 0.0
        kept_items: list[QueueItem] = []
        for item in self.items:
            if item.deadline_slot < slot_index:
                if item.task_id == current_task_id:
                    dropped_current += item.remaining_mb
                continue
            kept_items.append(item)
        self.items = kept_items
        return dropped_current

    def _make_queue_item(self, payload: GeneratedPayload) -> QueueItem:
        deadline_slots = max(1, int(round(payload.deadline_s / self.slot_seconds)))
        return QueueItem(
            task_id=payload.task_id,
            payload_policy=payload.payload_policy,
            remaining_mb=payload.payload_mb,
            original_payload_mb=payload.payload_mb,
            priority_score=payload.priority_score,
            value_score=payload.value_score,
            created_slot=payload.created_slot,
            deadline_slot=payload.created_slot + deadline_slots,
            metadata=dict(payload.metadata),
        )

    def enqueue(self, payload: GeneratedPayload) -> None:
        """把当前 payload 放入下传队列。"""

        if payload.payload_mb <= 0.0:
            return
        self.items.append(self._make_queue_item(payload))

    def _sort_items(self) -> None:
        """按价值、紧迫性与剩余负载联合排序。"""

        self.items.sort(
            key=lambda item: (
                -item.priority_score,
                item.deadline_slot,
                item.remaining_mb,
            )
        )

    def process_slot(
        self,
        slot_index: int,
        payload: GeneratedPayload,
        link_trace: LinkTrace,
    ) -> DownlinkResult:
        """在一个时隙内执行过期清理、入队与下传调度。"""

        queue_before = self.queue_mb
        dropped_current = self._drop_expired(slot_index, payload.task_id)

        if payload.payload_policy == "defer":
            status = DELIVERY_STATUS_DEFERRED
        elif payload.payload_mb <= 0.0:
            status = DELIVERY_STATUS_EMPTY
        else:
            self.enqueue(payload)
            status = DELIVERY_STATUS_QUEUED

        self._sort_items()
        capacity_mb = compute_available_capacity_mb(link_trace, self.slot_seconds)
        transmitted_total = 0.0
        transmitted_current = 0.0
        transmitted_task_ids: list[str] = []

        if capacity_mb > 0.0:
            new_items: list[QueueItem] = []
            for item in self.items:
                if capacity_mb <= 0.0:
                    new_items.append(item)
                    continue
                tx_mb = min(item.remaining_mb, capacity_mb)
                item.remaining_mb -= tx_mb
                capacity_mb -= tx_mb
                transmitted_total += tx_mb
                transmitted_task_ids.append(item.task_id)
                if item.task_id == payload.task_id:
                    transmitted_current += tx_mb
                if item.remaining_mb > 1.0e-9:
                    new_items.append(item)
            self.items = new_items

        if payload.payload_mb > 0.0:
            if transmitted_current >= payload.payload_mb - 1.0e-9:
                status = DELIVERY_STATUS_DELIVERED
            elif any(item.task_id == payload.task_id for item in self.items):
                status = DELIVERY_STATUS_QUEUED
            elif dropped_current > 0.0:
                status = DELIVERY_STATUS_DROPPED

        queue_after = self.queue_mb
        queue_reference_mb = max(queue_before, queue_after)
        queue_latency_s = (
            0.0
            if queue_reference_mb <= 0.0
            else self.slot_seconds * min(1.0, queue_reference_mb / max(self.queue_max_mb, 1.0e-9))
        )
        return DownlinkResult(
            S_down_total_mb=transmitted_total,
            S_down_current_task_mb=transmitted_current,
            generated_payload_current_mb=payload.payload_mb,
            dropped_current_task_mb=dropped_current,
            queue_next_mb=queue_after,
            delta_queue_mb=queue_after - queue_before,
            queue_latency_s=queue_latency_s,
            queue_item_count=len(self.items),
            current_task_delivery_status=status,
            transmitted_task_ids=transmitted_task_ids,
        )
