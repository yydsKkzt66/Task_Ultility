from pathlib import Path

from src.data.mock_dataset import MockDatasetAdapter
from src.data.sarfishsample_dataset import SARFishSampleDatasetAdapter
from src.data.task_stream import TaskStreamBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dataset_cfg() -> dict[str, object]:
    return {
        "dataset_name": "sarfishsample",
        "root": "datasets/SARFishSample",
        "product_type": "GRD",
        "split": "validation",
        "scene_index_path": "data/indexes/sarfishsample_scene_index.csv",
        "task_state_profile_path": "data/indexes/sarfishsample_task_state_profile.json",
        "seed": 2026,
    }


def _task_stream_cfg() -> dict[str, object]:
    return {
        "mode": "observation_window",
        "orbit_period_min": 90,
        "observation_windows_per_orbit": [2, 3],
        "window_duration_s_range": [240, 420],
        "intra_window_task_interval_slots": [1, 2],
        "background_task_probability": 0.0,
        "emergency_window_probability": 0.20,
        "high_value_window_probability": 0.30,
    }


def test_loads_scene_index_and_task_state_profile():
    adapter = SARFishSampleDatasetAdapter(_dataset_cfg())

    assert adapter.scene_index
    assert adapter.task_state_profile["target_count_distribution"]["count"] > 0
    assert all(scene["product_type"] == "GRD" for scene in adapter.scene_index)
    assert all(scene["logical_image_path"].startswith("GRD/") for scene in adapter.scene_index)
    assert all(not Path(scene["image_path"]).is_absolute() for scene in adapter.scene_index)


def test_build_task_stream_returns_requested_length_and_observation_window_tasks():
    adapter = SARFishSampleDatasetAdapter(_dataset_cfg())
    tasks = adapter.build_task_stream(num_slots=120, slot_seconds=60.0, task_stream_cfg=_task_stream_cfg())

    assert len(tasks) == 120
    samples = [task for task in tasks if task is not None]
    assert samples
    assert any(task.observation_window_id for task in samples)


def test_sarfishsample_tasks_have_portable_paths_and_valid_online_fields():
    adapter = SARFishSampleDatasetAdapter(_dataset_cfg())
    tasks = adapter.build_task_stream(num_slots=180, slot_seconds=60.0, task_stream_cfg=_task_stream_cfg())
    samples = [task for task in tasks if task is not None]

    assert samples
    for task in samples:
        assert not task.image_path.startswith("mock://")
        assert task.metadata["logical_image_path"].startswith("GRD/")
        assert task.task_class in {"routine", "high_value", "emergency"}
        assert 0.50 <= task.task_value <= 0.98
        assert 30.0 <= task.deadline_s <= 120.0
        assert 0.05 <= task.difficulty <= 1.0
        assert 0.05 <= task.uncertainty <= 1.0
        assert task.high_value_flag is (task.task_class in {"high_value", "emergency"})
        assert task.emergency_flag is (task.task_class == "emergency")
        assert task.value_level == {"routine": "medium", "high_value": "high", "emergency": "urgent"}[task.task_class]
        assert task.metadata["offline_label_calibrated"] is True
        assert "offline_label_summary" in task.metadata
        assert task.metadata["ground_truth_usage"] == "offline_metadata_only_not_used_for_online_priority"


def test_builder_dispatches_sarfishsample_without_affecting_mock_dataset():
    sarfish_tasks = TaskStreamBuilder(_dataset_cfg(), _task_stream_cfg()).build(num_slots=60, slot_seconds=60.0)
    assert len(sarfish_tasks) == 60
    assert any(task is not None and not task.image_path.startswith("mock://") for task in sarfish_tasks)

    mock_tasks = MockDatasetAdapter({"dataset_name": "mock", "seed": 2026}).build_task_stream(
        num_slots=12,
        slot_seconds=60.0,
        task_stream_cfg={"mode": "one_task_per_slot"},
    )
    assert len(mock_tasks) == 12
    assert all(task.image_path.startswith("mock://") for task in mock_tasks)


def test_sarfishsample_v2_value_requirement_uses_label_free_proxy():
    cfg = {
        **_task_stream_cfg(),
        "task_value_model": {"enable_mission_prior_value_v2": True},
    }
    adapter = SARFishSampleDatasetAdapter(_dataset_cfg())
    samples = [
        task
        for task in adapter.build_task_stream(num_slots=120, slot_seconds=60.0, task_stream_cfg=cfg)
        if task is not None
    ]

    assert samples
    for task in samples:
        assert task.metadata["value_model_version"] == "mission_prior_value_v2"
        assert task.metadata["no_label_leakage_flag"] is True
        assert task.product_requirement_source == "mission_prior_proxy_label_free_for_v2"
        assert task.metadata["value_model_product_requirement_source"] == "mission_prior_proxy_label_free_for_v2"
        assert task.metadata["value_model_product_requirement_uncertainty_used"] == 0.0
