"""安全动作筛选。"""

from __future__ import annotations

from src.common.constants import ACTION_FALLBACK_NAME
from src.common.types import JointAction, PredictedMetrics, TaskSample


class SafeFilter:
    """在 router 选择之前，先使用预测器结果过滤不安全动作。"""

    def __init__(self, simulator_cfg: dict[str, float]) -> None:
        self.simulator_cfg = simulator_cfg

    def filter_actions(
        self,
        task: TaskSample,
        actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
    ) -> tuple[list[JointAction], dict[str, list[str]]]:
        """返回安全动作集合与不安全原因字典。"""

        safe_actions: list[JointAction] = []
        reasons_by_action: dict[str, list[str]] = {}
        battery_min = float(self.simulator_cfg["battery_min"])
        thermal_max = float(self.simulator_cfg["thermal_max"])
        queue_max = float(self.simulator_cfg["queue_max_mb"])

        for action in actions:
            predicted = predicted_map[action.action_name]
            reasons: list[str] = list(predicted.safety_reasons)
            if predicted.predicted_B_next < battery_min:
                reasons.append("battery reserve violation")
            if predicted.predicted_E_AI_next < 0.0:
                reasons.append("AI energy budget violation")
            if predicted.predicted_Theta_next > thermal_max:
                reasons.append("thermal pressure violation")
            if predicted.predicted_L_inf + predicted.predicted_L_queue + predicted.predicted_L_comm > task.deadline_s:
                reasons.append("deadline risk violation")
            if predicted.predicted_Q_next_mb > queue_max:
                reasons.append("queue overflow violation")

            if reasons:
                reasons_by_action[action.action_name] = reasons
            else:
                safe_actions.append(action)

        if safe_actions:
            return safe_actions, reasons_by_action

        for action in actions:
            if action.action_name == ACTION_FALLBACK_NAME:
                return [action], reasons_by_action
        return [actions[0]], reasons_by_action
