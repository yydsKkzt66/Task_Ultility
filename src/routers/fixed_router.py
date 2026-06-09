"""固定动作基线。"""

from __future__ import annotations

from src.common.constants import ACTION_FALLBACK_NAME
from src.common.types import ContextVector, JointAction, PredictedMetrics
from src.routers.base_router import BaseRouter


class FixedRouter(BaseRouter):
    """始终优先选择同一个动作的 baseline。"""

    def __init__(self, router_name: str, preferred_action_name: str) -> None:
        super().__init__(router_name)
        self.preferred_action_name = preferred_action_name

    def select_action(
        self,
        context: ContextVector,
        safe_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
    ) -> JointAction:
        _ = (context, predicted_map)
        for action in safe_actions:
            if action.action_name == self.preferred_action_name:
                return action
        for action in safe_actions:
            if action.action_name == ACTION_FALLBACK_NAME:
                return action
        return safe_actions[0]
