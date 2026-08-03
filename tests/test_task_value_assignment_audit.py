from __future__ import annotations

from types import SimpleNamespace

from scripts.run_value_aware_staged_delivery_queue import _task_value_rows
from src.common.types import TaskSample


def _task(task_id: str, value_level: str = "high") -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_path="mock://image",
        label_path="mock://label",
        scene_id=f"scene_{task_id}",
        virtual_time=0,
        task_value=0.5,
        deadline_s=120.0,
        difficulty=0.1,
        uncertainty=0.2,
        metadata={
            "old_task_value": 0.5,
            "area_prior_bucket": "high",
            "source_priority": 0.9,
            "value_assignment_reason": "unit_test_rule",
        },
        arrival_slot=2,
        task_class="high_value",
        high_value_flag=True,
        product_requirement="context",
        product_requirement_source="mission_prior_proxy",
        base_prior_score=0.8,
        value_level=value_level,
        value_multiplier=5.0,
        normalization_alpha=0.2,
        calibrated_task_value=0.9,
        value_assignment_reason="unit_test_rule",
    )


def test_task_value_assignment_trace_has_no_leakage_fields():
    run = {
        "scenario": "natural",
        "seed": 2026,
        "arm": "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue",
        "timeline": [SimpleNamespace(task=_task("a"))],
    }
    rows = _task_value_rows([run])
    assert len(rows) == 1
    online_fields = rows[0]["fields_used_for_value_assignment"]
    forbidden = ["label", "bbox", "delivery", "delivered", "closure", "reward", "future", "target_count"]
    assert not [token for token in forbidden if token in online_fields]
    assert rows[0]["leakage_safe_flag"] == 1


def test_value_level_distribution_counts_unique_tasks_not_evaluations():
    task = _task("repeat")
    run = {
        "scenario": "natural",
        "seed": 2026,
        "arm": "soft_moderate_value_skew_plus_gate_plus_persistent_priority_queue",
        "timeline": [SimpleNamespace(task=task), SimpleNamespace(task=task), SimpleNamespace(task=None)],
    }
    rows = _task_value_rows([run])
    assert len(rows) == 1
    assert rows[0]["task_id"] == "repeat"
