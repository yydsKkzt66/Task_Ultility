from src.actions.action_space import ActionSpace
from src.common.types import PredictedMetrics, TaskSample
from src.routers.safe_filter import SafeFilter


def test_safe_filter_removes_violating_actions():
    actions = ActionSpace().list_actions()
    predicted_map = {}
    for action in actions:
        predicted_map[action.action_name] = PredictedMetrics(
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            2.0 if action.action_name == "high_accuracy+roi_context" else 0.1,
            0.1 if action.action_name == "high_accuracy+roi_context" else 0.5,
            0.5,
            0.95 if action.action_name == "high_accuracy+roi_context" else 0.2,
            0.0,
            0.0,
            "queued",
            [],
        )
    safe_filter = SafeFilter({"battery_min": 0.18, "thermal_max": 0.92, "queue_max_mb": 1.0})
    task = TaskSample("task", "a", "b", "scene", 0, 0.5, 20.0, 0.5, 0.5)
    safe_actions, _ = safe_filter.filter_actions(task, actions, predicted_map)
    assert all(action.action_name != "high_accuracy+roi_context" for action in safe_actions)
