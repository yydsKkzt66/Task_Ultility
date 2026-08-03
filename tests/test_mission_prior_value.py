from src.data.mission_prior_value import (
    compute_mission_prior_value_v2,
    mission_prior_value_v2_enabled,
)
from src.data.mock_dataset import MockDatasetAdapter


def _enabled_cfg() -> dict[str, object]:
    return {
        "task_value_model": {
            "enable_mission_prior_value_v2": True,
            "clip_min": 0.05,
            "clip_max": 0.95,
            "weights": {
                "task_class": 0.50,
                "evidence_need": 0.25,
                "area_prior": 0.15,
                "source_priority": 0.10,
            },
            "source_priority_default": 0.0,
        }
    }


def test_mission_prior_value_v2_is_disabled_by_default():
    assert not mission_prior_value_v2_enabled({})


def test_mission_prior_value_v2_is_deterministic_and_uses_scene_id_only_for_area_prior():
    cfg = _enabled_cfg()
    first = compute_mission_prior_value_v2(
        task_id="task_1",
        scene_id="scene_a",
        task_class="high_value",
        product_requirement="evidence",
        deadline_s=600.0,
        container_cfg=cfg,
    )
    second = compute_mission_prior_value_v2(
        task_id="task_1",
        scene_id="scene_a",
        task_class="high_value",
        product_requirement="evidence",
        deadline_s=600.0,
        container_cfg=cfg,
    )

    assert first == second
    assert 0.05 <= first.task_value <= 0.95
    assert first.no_label_leakage_flag is True
    assert first.area_prior_bucket in {"low", "medium", "high"}


def test_evidence_requirement_raises_value_all_else_equal():
    cfg = {
        "task_value_model": {
            "enable_mission_prior_value_v2": True,
            "area_prior_mode": "fixed_scene_prior_table",
            "scene_area_prior_table": {"scene_a": "medium"},
        }
    }
    alert = compute_mission_prior_value_v2(
        task_id="task_1",
        scene_id="scene_a",
        task_class="high_value",
        product_requirement="alert",
        deadline_s=600.0,
        container_cfg=cfg,
    )
    evidence = compute_mission_prior_value_v2(
        task_id="task_1",
        scene_id="scene_a",
        task_class="high_value",
        product_requirement="evidence",
        deadline_s=600.0,
        container_cfg=cfg,
    )

    assert evidence.task_value > alert.task_value


def test_mock_dataset_disabled_task_value_model_preserves_stream_values():
    base_cfg = {"mode": "observation_window", "background_task_probability": 0.10}
    disabled_cfg = {
        **base_cfg,
        "task_value_model": {
            "enable_mission_prior_value_v2": False,
        },
    }
    adapter = MockDatasetAdapter({"dataset_name": "mock", "seed": 2026})
    base_tasks = adapter.build_task_stream(160, 60.0, base_cfg)
    disabled_tasks = adapter.build_task_stream(160, 60.0, disabled_cfg)

    base_samples = [task for task in base_tasks if task is not None]
    disabled_samples = [task for task in disabled_tasks if task is not None]
    assert [(task.task_id, task.task_value, task.product_requirement) for task in base_samples] == [
        (task.task_id, task.task_value, task.product_requirement) for task in disabled_samples
    ]


def test_mock_dataset_enabled_adds_component_metadata_without_label_fields():
    cfg = {
        "mode": "observation_window",
        "background_task_probability": 0.10,
        "task_value_model": {"enable_mission_prior_value_v2": True},
    }
    adapter = MockDatasetAdapter({"dataset_name": "mock", "seed": 2026})
    samples = [task for task in adapter.build_task_stream(160, 60.0, cfg) if task is not None]

    assert samples
    for task in samples:
        assert task.metadata["value_model_version"] == "mission_prior_value_v2"
        assert task.metadata["no_label_leakage_flag"] is True
        assert "old_task_value" in task.metadata
        assert "new_task_value" in task.metadata
        assert "target_count" not in task.metadata
        assert "bbox_area_mean" not in task.metadata
