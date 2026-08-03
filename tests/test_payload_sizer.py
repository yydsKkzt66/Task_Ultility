from src.common.types import DetectionResult, TaskSample
from src.payload.payload_sizer import generate_payload


def test_delay_store_produces_zero_payload():
    task = TaskSample("task_0001", "a", "b", "scene", 1, 0.5, 30.0, 0.3, 0.3)
    detection = DetectionResult([], 0.0, False, [], 0.3, 0.0, 0.0, {}, skipped=True, skip_reason="skip")
    payload = generate_payload(task, detection, "defer", slot_index=1)
    assert payload.payload_mb == 0.0
    assert payload.information_content <= 0.05


def test_staged_plan_generates_metadata_first_and_records_residual():
    task = TaskSample(
        "task_0002",
        "a",
        "b",
        "scene",
        1,
        0.8,
        30.0,
        0.3,
        0.3,
        product_requirement="verification",
    )
    detection = DetectionResult([], 0.8, False, [], 0.3, 1.0, 0.8, {})
    payload = generate_payload(task, detection, "metadata_then_roi_context", slot_index=1)
    assert payload.payload_plan == "metadata_then_roi_context"
    assert payload.product_type == "metadata"
    assert payload.residual_product_type == "roi_context"
    assert payload.is_metadata
    assert payload.incremental_information_utility > 0.0


def test_old_one_shot_payload_name_still_works():
    task = TaskSample("task_0003", "a", "b", "scene", 1, 0.8, 30.0, 0.3, 0.3)
    detection = DetectionResult([], 0.8, False, [], 0.3, 1.0, 0.8, {})
    payload = generate_payload(task, detection, "roi", slot_index=1)
    assert payload.payload_policy == "roi"
    assert payload.payload_plan == "roi"
    assert payload.product_type == "roi"
