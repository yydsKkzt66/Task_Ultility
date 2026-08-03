from src.actions.action_space import ActionSpace
from src.common.types import PredictedMetrics, TaskSample
from src.routers.safe_filter import SafeFilter


def _build_predicted_metrics_by_action():
    actions = ActionSpace().list_actions()
    predicted_map = {}
    for action in actions:
        predicted_map[action.action_name] = PredictedMetrics(
            0.2,
            0.1,
            0.2 if action.action_name == "high_accuracy+roi_context" else 0.05,
            0.05,
            2.5 if action.action_name == "high_accuracy+roi_context" else 1.0,
            0.0,
            0.0,
            2.0 if action.action_name == "high_accuracy+roi_context" else 0.1,
            0.1 if action.action_name == "high_accuracy+roi_context" else 0.5,
            0.5,
            0.95 if action.action_name == "high_accuracy+roi_context" else 0.2,
            0.5,
            1.0 if action.action_name == "high_accuracy+roi_context" else 0.0,
            1.5 if action.action_name == "high_accuracy+roi_context" else 0.8,
            "queued",
            [],
            0.8,
        )
    return actions, predicted_map


def test_safe_filter_removes_violating_actions():
    actions, predicted_map = _build_predicted_metrics_by_action()
    safe_filter = SafeFilter({"battery_min": 0.18, "thermal_max": 0.92, "queue_max_mb": 1.0})
    task = TaskSample("task", "a", "b", "scene", 0, 0.5, 20.0, 0.5, 0.5)
    safe_actions, _ = safe_filter.filter_actions(task, actions, predicted_map)
    assert all(action.action_name != "high_accuracy+roi_context" for action in safe_actions)


def test_shield_corrects_unsafe_raw_action_to_safe_fallback():
    actions, predicted_map = _build_predicted_metrics_by_action()
    safe_filter = SafeFilter(
        {
            "battery_min": 0.18,
            "thermal_max": 0.92,
            "queue_max_mb": 1.0,
            "safety_terminal_fallback_candidates": ["delay/store", "lightweight+metadata"],
        }
    )
    task = TaskSample("task", "a", "b", "scene", 0, 0.5, 20.0, 0.5, 0.5)
    raw_action = next(action for action in actions if action.action_name == "high_accuracy+roi_context")

    decision = safe_filter.shield_or_fallback(task, raw_action, actions, predicted_map)

    assert decision.raw_action.action_name == "high_accuracy+roi_context"
    assert decision.executed_action.action_name != decision.raw_action.action_name
    assert decision.safety_corrected == 1
    assert decision.safety_violation_before_correction == 1
    assert decision.safety_violation_after_correction == 0


def test_deadline_risk_is_task_service_risk_not_resource_safety():
    actions, predicted_map = _build_predicted_metrics_by_action()
    safe_filter = SafeFilter({"battery_min": 0.18, "thermal_max": 0.92, "queue_max_mb": 10.0})
    task = TaskSample("task", "a", "b", "scene", 0, 0.5, 0.5, 0.5, 0.5)
    action = next(action for action in actions if action.action_name == "lightweight+metadata")

    assert safe_filter.evaluate_action(task, action, predicted_map) == []
    assert safe_filter.evaluate_task_service_risk(task, action, predicted_map) == ["deadline_risk"]
