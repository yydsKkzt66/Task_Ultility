"""瀹夊叏鍔ㄤ綔绛涢€夈€?"""

from __future__ import annotations

from src.common.constants import ACTION_FALLBACK_NAME
from src.common.types import JointAction, PredictedMetrics, ShieldDecision, TaskSample


class SafeFilter:
    """鍦?router 閫夋嫨鍓嶅悗缁熶竴鎻愪緵瀹夊叏璇勪及銆佺粺涓€ shield 涓?fallback閫夋嫨銆?"""

    def __init__(self, simulator_cfg: dict[str, float | str | list[str]]) -> None:
        self.simulator_cfg = simulator_cfg

    @staticmethod
    def _unique_reasons(reasons: list[str]) -> list[str]:
        unique: list[str] = []
        for reason in reasons:
            if reason not in unique:
                unique.append(reason)
        return unique

    def evaluate_action(
        self,
        task: TaskSample,
        action: JointAction,
        predicted_map: dict[str, PredictedMetrics],
    ) -> list[str]:
        """杩斿洖鍗曚釜鍔ㄤ綔鐨勫畨鍏ㄨ繚瑙勫師鍥犮€?"""

        predicted = predicted_map[action.action_name]
        reasons: list[str] = list(predicted.safety_reasons)
        battery_min = float(self.simulator_cfg["battery_min"])
        thermal_max = float(self.simulator_cfg["thermal_max"])
        queue_max = float(self.simulator_cfg["queue_max_mb"])

        if predicted.predicted_B_next < battery_min:
            reasons.append("battery reserve violation")
        if predicted.predicted_E_AI_next < 0.0:
            reasons.append("AI energy budget violation")
        if predicted.predicted_Theta_next > thermal_max:
            reasons.append("thermal pressure violation")
        if predicted.predicted_Q_next_mb > queue_max:
            reasons.append("queue overflow violation")
        return self._unique_reasons(reasons)

    def evaluate_task_service_risk(
        self,
        task: TaskSample,
        action: JointAction,
        predicted_map: dict[str, PredictedMetrics],
    ) -> list[str]:
        predicted = predicted_map[action.action_name]
        reasons: list[str] = []
        if action.model_name == "delay_store" or action.payload_policy == "defer":
            reasons.append("task_deferred")
        if predicted.predicted_L_inf + predicted.predicted_L_queue + predicted.predicted_L_comm > task.deadline_s:
            reasons.append("deadline_risk")
        return reasons

    def filter_actions(
        self,
        task: TaskSample,
        actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
    ) -> tuple[list[JointAction], dict[str, list[str]]]:
        """杩斿洖鐪熷疄瀹夊叏鍔ㄤ綔闆嗗悎涓庝笉瀹夊叏鍘熷洜銆?"""

        safe_actions: list[JointAction] = []
        reasons_by_action: dict[str, list[str]] = {}

        for action in actions:
            reasons = self.evaluate_action(task, action, predicted_map)
            if reasons:
                reasons_by_action[action.action_name] = reasons
            else:
                safe_actions.append(action)
        return safe_actions, reasons_by_action

    def _pick_highest_predicted_utility(
        self,
        actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
    ) -> JointAction:
        return max(
            actions,
            key=lambda action: (
                float(predicted_map[action.action_name].predicted_reward),
                -float(predicted_map[action.action_name].predicted_E_comp + predicted_map[action.action_name].predicted_E_comm),
            ),
        )

    def _pick_lowest_energy(
        self,
        actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
    ) -> JointAction:
        return min(
            actions,
            key=lambda action: (
                float(predicted_map[action.action_name].predicted_E_comp + predicted_map[action.action_name].predicted_E_comm),
                float(predicted_map[action.action_name].predicted_L_inf + predicted_map[action.action_name].predicted_L_queue + predicted_map[action.action_name].predicted_L_comm),
            ),
        )

    def choose_fallback_action(
        self,
        task: TaskSample,
        actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        safe_actions: list[JointAction] | None = None,
    ) -> JointAction:
        """鎸夌粺涓€ fallback 瑙勫垯閫夋嫨鎵ц鍔ㄤ綔銆?"""

        safe_actions = safe_actions if safe_actions is not None else self.filter_actions(task, actions, predicted_map)[0]
        if safe_actions:
            strategy = str(self.simulator_cfg.get("safety_fallback_strategy", "predicted_effective_utility"))
            try:
                if strategy == "lowest_energy_safe_action":
                    return self._pick_lowest_energy(safe_actions, predicted_map)
                return self._pick_highest_predicted_utility(safe_actions, predicted_map)
            except Exception:
                return self._pick_lowest_energy(safe_actions, predicted_map)

        candidate_names = self.simulator_cfg.get("safety_terminal_fallback_candidates", [ACTION_FALLBACK_NAME, "lightweight+metadata"])
        if isinstance(candidate_names, str):
            candidate_names = [candidate_names]

        for action_name in candidate_names:
            for action in actions:
                if action.action_name == action_name:
                    return action

        return self._pick_lowest_energy(actions, predicted_map)

    def shield_or_fallback(
        self,
        task: TaskSample,
        raw_action: JointAction,
        actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
    ) -> ShieldDecision:
        """瀵?router 鐨?raw_action 杩涜缁熶竴 safety shield銆?"""

        reasons_before = self.evaluate_action(task, raw_action, predicted_map)
        if not reasons_before:
            return ShieldDecision(
                raw_action=raw_action,
                executed_action=raw_action,
                safety_corrected=0,
                safety_violation_before_correction=0,
                safety_violation_after_correction=0,
                reasons_before_correction=[],
                reasons_after_correction=[],
            )

        safe_actions, _ = self.filter_actions(task, actions, predicted_map)
        executed_action = self.choose_fallback_action(task, actions, predicted_map, safe_actions=safe_actions)
        reasons_after = self.evaluate_action(task, executed_action, predicted_map)
        return ShieldDecision(
            raw_action=raw_action,
            executed_action=executed_action,
            safety_corrected=int(executed_action.action_name != raw_action.action_name),
            safety_violation_before_correction=1,
            safety_violation_after_correction=int(bool(reasons_after)),
            reasons_before_correction=reasons_before,
            reasons_after_correction=reasons_after,
        )
