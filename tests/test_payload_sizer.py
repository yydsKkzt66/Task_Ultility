from src.common.types import DetectionResult, TaskSample
from src.payload.payload_sizer import generate_payload


def test_delay_store_produces_zero_payload():
    task = TaskSample("task_0001", "a", "b", "scene", 1, 0.5, 30.0, 0.3, 0.3)
    detection = DetectionResult([], 0.0, False, [], 0.3, 0.0, 0.0, {}, skipped=True, skip_reason="skip")
    payload = generate_payload(task, detection, "defer", slot_index=1)
    assert payload.payload_mb == 0.0
    assert payload.information_content <= 0.05
