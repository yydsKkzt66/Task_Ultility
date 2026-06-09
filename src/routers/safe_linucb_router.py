"""Safe LinUCB router。"""

from __future__ import annotations

import numpy as np

from src.common.types import ContextVector, JointAction, PredictedMetrics
from src.routers.base_router import BaseRouter


class SafeLinUCBRouter(BaseRouter):
    """仅在安全动作集合内做 LinUCB 选择。"""

    def __init__(self, action_names: list[str], alpha: float = 0.70, ridge: float = 1.0) -> None:
        super().__init__("safe_linucb")
        self.alpha = alpha
        self.ridge = ridge
        self.action_names = action_names
        self.dimension = 9
        self.A = {name: np.eye(self.dimension) * ridge for name in action_names}
        self.b = {name: np.zeros(self.dimension) for name in action_names}

    def select_action(
        self,
        context: ContextVector,
        safe_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
    ) -> JointAction:
        x = context.as_array()
        best_action = safe_actions[0]
        best_score = -float("inf")

        for action in safe_actions:
            A_inv = np.linalg.inv(self.A[action.action_name])
            theta = A_inv @ self.b[action.action_name]
            exploit = float(theta @ x)
            explore = float(np.sqrt(x @ A_inv @ x))
            prior = 0.20 * predicted_map[action.action_name].predicted_reward
            score = exploit + self.alpha * explore + prior
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def update(self, context: ContextVector, action: JointAction, observed_reward: float) -> None:
        x = context.as_array()
        self.A[action.action_name] = self.A[action.action_name] + np.outer(x, x)
        self.b[action.action_name] = self.b[action.action_name] + observed_reward * x
