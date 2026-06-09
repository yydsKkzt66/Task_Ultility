from src.common.types import GeneratedPayload, LinkTrace
from src.link.downlink_queue import DownlinkQueue


def test_downlink_queue_supports_enqueue_transmit_and_drop_expired():
    queue = DownlinkQueue(queue_max_mb=4.0, slot_seconds=10.0)
    payload = GeneratedPayload("task_a", "roi", 2.0, 0.8, 0.9, 0.8, 10.0, 0, {})
    no_contact = LinkTrace(0, False, -3.0, 0.0, 0.0, 0.0)
    queued = queue.process_slot(0, payload, no_contact)
    assert queued.queue_next_mb == 2.0
    assert queued.current_task_delivery_status == "queued"
    assert queued.queue_latency_s > 0.0

    contact = LinkTrace(1, True, 12.0, 2.0, 0.8, 10.0)
    delivered = queue.process_slot(1, GeneratedPayload("task_b", "defer", 0.0, 0.0, 0.0, 0.0, 10.0, 1, {}), contact)
    assert delivered.S_down_total_mb > 0.0

    stale_payload = GeneratedPayload("task_c", "roi", 1.5, 0.8, 0.7, 0.7, 10.0, 1, {})
    queue.process_slot(1, stale_payload, no_contact)
    dropped = queue.process_slot(3, GeneratedPayload("task_d", "defer", 0.0, 0.0, 0.0, 0.0, 10.0, 3, {}), no_contact)
    assert dropped.queue_item_count == 0
