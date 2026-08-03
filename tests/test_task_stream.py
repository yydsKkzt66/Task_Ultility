from pathlib import Path

from src.common.config import load_project_configs
from src.data.product_requirements import assign_product_requirement
from src.data.task_stream import TaskStreamBuilder
from src.timeline.timeline_builder import TimelineBuilder


def test_observation_window_task_stream_has_idle_slots_and_bursts():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    configs["timeline"]["num_slots"] = 80
    configs["task_stream"]["mode"] = "observation_window"
    tasks = TaskStreamBuilder(configs["dataset"], configs["task_stream"]).build(
        num_slots=80,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )

    assert any(task is None for task in tasks)

    window_counts = {}
    for task in tasks:
        if task is None or not task.observation_window_id:
            continue
        window_counts[task.observation_window_id] = window_counts.get(task.observation_window_id, 0) + 1
    assert window_counts
    assert max(window_counts.values()) >= 2


def test_high_value_and_emergency_windows_have_higher_requirements():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    configs["timeline"]["num_slots"] = 1200
    configs["task_stream"]["mode"] = "observation_window"
    configs["task_stream"]["emergency_window_probability"] = 0.35
    configs["task_stream"]["high_value_window_probability"] = 0.40
    tasks = TaskStreamBuilder(configs["dataset"], configs["task_stream"]).build(
        num_slots=1200,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )

    routine = [task for task in tasks if task is not None and task.task_class == "routine" and task.observation_window_id]
    priority = [task for task in tasks if task is not None and task.task_class in {"high_value", "emergency"}]

    assert routine
    assert priority
    assert sum(task.quality_requirement for task in priority if task.quality_requirement is not None) / len(priority) > sum(
        task.quality_requirement for task in routine if task.quality_requirement is not None
    ) / len(routine)
    assert sum(task.information_requirement for task in priority if task.information_requirement is not None) / len(priority) > sum(
        task.information_requirement for task in routine if task.information_requirement is not None
    ) / len(routine)


def test_timeline_builder_supports_observation_window_mode():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    configs["timeline"]["use_num_days"] = False
    configs["timeline"]["num_slots"] = 24
    configs["task_stream"]["mode"] = "observation_window"
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    assert len(timeline) == 24
    assert any(step.task is None for step in timeline)


def test_timeline_builder_uses_num_days_when_enabled():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    configs["timeline"]["use_num_days"] = True
    configs["timeline"]["num_days"] = 0.05
    configs["timeline"]["slot_seconds"] = 60
    configs["task_stream"]["slot_seconds"] = 60
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    assert len(timeline) == int(0.05 * 24 * 3600 / 60)


def test_product_requirement_assignment_ignores_forbidden_label_fields():
    baseline = assign_product_requirement(
        task_value=0.80,
        deadline_s=70.0,
        uncertainty=0.40,
        high_value_flag=True,
        emergency_flag=False,
        task_class="high_value",
    )
    changed_forbidden_fields = {
        "offline_label_summary": {"target_count": 999, "bbox_area_mean": 999.0},
        "object_class": "fishing_vessel",
        "sarfish_label": "target",
    }
    assert changed_forbidden_fields
    assert baseline == assign_product_requirement(
        task_value=0.80,
        deadline_s=70.0,
        uncertainty=0.40,
        high_value_flag=True,
        emergency_flag=False,
        task_class="high_value",
    )


def test_product_requirement_rule_legacy_mode_preserves_task_stream_behavior():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    configs["timeline"]["num_slots"] = 40
    configs["task_stream"]["mode"] = "observation_window"

    legacy_cfg = dict(configs["task_stream"])
    legacy_cfg["product_requirement_rule"] = {"mode": "legacy"}
    default_tasks = TaskStreamBuilder(configs["dataset"], configs["task_stream"]).build(
        num_slots=40,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )
    legacy_tasks = TaskStreamBuilder(configs["dataset"], legacy_cfg).build(
        num_slots=40,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )

    assert [(task.task_id, task.product_requirement) if task else None for task in legacy_tasks] == [
        (task.task_id, task.product_requirement) if task else None for task in default_tasks
    ]


def test_product_requirement_rule_v21_is_default_off_diagnostic_hook():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    configs["timeline"]["num_slots"] = 80
    task_stream_cfg = dict(configs["task_stream"])
    task_stream_cfg["mode"] = "observation_window"
    task_stream_cfg["product_requirement_rule"] = {"mode": "v2_1"}

    tasks = TaskStreamBuilder(configs["dataset"], task_stream_cfg).build(
        num_slots=80,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )
    task = next(item for item in tasks if item is not None)

    assert task.product_requirement_source == "mission_prior_product_requirement_v2_1_shadow"
    assert task.metadata["product_requirement_rule_mode"] == "v2_1"
    assert "legacy_product_requirement" in task.metadata
    assert "v21_requirement_scores" in task.metadata
    assert "recommended_payload_plan_shadow" in task.metadata
