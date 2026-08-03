from __future__ import annotations

from src.common.types import GeneratedPayload, LinkTrace
from src.link.downlink_queue import DownlinkQueue


def _payload(
    task_id: str,
    policy: str,
    payload_mb: float,
    information: float,
    priority: float,
    task_value: float,
    created_slot: int,
    deadline_s: float = 10.0,
    uncertainty: float = 0.2,
    remaining_utility: float | None = None,
    is_metadata: bool | None = None,
    is_high_value: bool | None = None,
):
    return GeneratedPayload(
        task_id=task_id,
        payload_policy=policy,
        payload_mb=payload_mb,
        information_content=information,
        priority_score=priority,
        value_score=task_value,
        task_value=task_value,
        uncertainty=uncertainty,
        remaining_utility=task_value * information if remaining_utility is None else remaining_utility,
        is_metadata=(policy == "metadata") if is_metadata is None else is_metadata,
        is_high_value=(task_value >= 0.75) if is_high_value is None else is_high_value,
        deadline_s=deadline_s,
        created_slot=created_slot,
        metadata={},
        payload_plan=policy,
        product_type=policy,
        product_requirement="verification",
        incremental_utility_estimate=task_value * information if remaining_utility is None else remaining_utility,
    )


def test_downlink_queue_is_not_fifo_and_prefers_metadata_high_utility_items():
    queue = DownlinkQueue(
        queue_max_mb=4.0,
        slot_seconds=10.0,
        queue_priority_tau_d=60.0,
        queue_age_bonus=0.01,
        queue_meta_first_bonus=0.5,
        queue_high_value_bonus=0.3,
    )
    no_contact = LinkTrace(0, False, -3.0, 0.0, 0.0, 0.0)
    weak_contact = LinkTrace(1, True, 12.0, 1.6, 0.8, 10.0)  # 2MB capacity

    low_priority_old = _payload("task_low", "roi", 2.0, 0.4, 0.4, 0.4, 0)
    high_priority_new = _payload("task_high_meta", "metadata", 0.5, 0.8, 0.9, 0.9, 1)

    queue.process_slot(0, low_priority_old, no_contact)
    delivered = queue.process_slot(1, high_priority_new, weak_contact)

    assert delivered.transmitted_task_ids
    assert delivered.transmitted_task_ids[0] == "task_high_meta"
    assert delivered.metadata_delivered_count == 1
    assert delivered.delivered_high_value_payload_count == 1


def test_downlink_queue_supports_expire_and_drop_overflow():
    queue = DownlinkQueue(queue_max_mb=2.0, slot_seconds=10.0)
    no_contact = LinkTrace(0, False, -3.0, 0.0, 0.0, 0.0)

    stale_payload = _payload("task_stale", "roi", 1.5, 0.7, 0.7, 0.7, 0, deadline_s=10.0)
    queue.process_slot(0, stale_payload, no_contact)
    expired = queue.process_slot(2, _payload("task_defer", "defer", 0.0, 0.05, 0.0, 0.0, 2), no_contact)
    assert expired.expired_item_count == 1

    overflow_queue = DownlinkQueue(queue_max_mb=1.0, slot_seconds=10.0)
    overflow_queue.process_slot(0, _payload("task_keep", "metadata", 0.4, 0.8, 0.9, 0.9, 0), no_contact)
    dropped = overflow_queue.process_slot(1, _payload("task_drop", "roi_context", 1.4, 1.0, 0.1, 0.2, 1), no_contact)
    assert dropped.dropped_item_count >= 1


def test_downlink_queue_suppresses_duplicate_pending_product():
    queue = DownlinkQueue(queue_max_mb=4.0, slot_seconds=10.0)
    no_contact = LinkTrace(0, False, -3.0, 0.0, 0.0, 0.0)

    first = queue.process_slot(0, _payload("task_dup", "metadata", 0.4, 0.8, 0.9, 0.9, 0), no_contact)
    second = queue.process_slot(1, _payload("task_dup", "metadata", 0.4, 0.8, 0.9, 0.9, 1), no_contact)

    assert first.queue_item_count == 1
    assert second.duplicate_product_attempt_count >= 1
    assert second.queue_item_count == 1
