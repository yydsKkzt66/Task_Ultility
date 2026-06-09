"""规则驱动路由器。"""

from __future__ import annotations

from src.common.constants import ACTION_FALLBACK_NAME
from src.common.types import ContextVector, JointAction, PredictedMetrics
from src.routers.base_router import BaseRouter


class RuleBasedRouter(BaseRouter):
    """提供两个轻量规则基线。

    1. `comm_conservative` 更关注链路机会和队列压力。
    2. `utility_rule` 更关注任务价值、不确定性和预测 reward。
    """

    def __init__(self, router_name: str, mode: str) -> None:
        super().__init__(router_name)
        self.mode = mode

    def _find_action(self, safe_actions: list[JointAction], action_name: str) -> JointAction | None:
        for action in safe_actions:
            if action.action_name == action_name:
                return action
        return None

    def select_action(
        self,
        context: ContextVector,
        safe_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
    ) -> JointAction:
        if self.mode == "comm_conservative":
            if context.C_t < 0.20 or context.Q_t > 0.75:
                return self._find_action(safe_actions, ACTION_FALLBACK_NAME) or safe_actions[0]
            if context.C_t < 0.45:
                return self._find_action(safe_actions, "balanced+metadata") or safe_actions[0]
            return self._find_action(safe_actions, "balanced+roi") or safe_actions[0]

        if self.mode == "utility_rule":
            if context.V_t > 0.78 and context.U_t > 0.50 and context.B_t > 0.35 and context.C_t > 0.45:
                preferred = self._find_action(safe_actions, "high_accuracy+roi_context")
                if preferred is not None:
                    return preferred
            if context.V_t > 0.55 or context.U_t > 0.55:
                preferred = self._find_action(safe_actions, "balanced+roi")
                if preferred is not None:
                    return preferred
            return max(safe_actions, key=lambda action: predicted_map[action.action_name].predicted_reward)

        return safe_actions[0]
