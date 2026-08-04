"""瑙勫垯椹卞姩璺敱鍣ㄣ€?"""

from __future__ import annotations

from src.common.constants import ACTION_FALLBACK_NAME
from src.common.types import ContextVector, JointAction, PredictedMetrics
from src.routers.base_router import BaseRouter


class RuleBasedRouter(BaseRouter):
    """鎻愪緵涓や釜杞婚噺瑙勫垯鍩虹嚎銆?
    1. `comm_conservative` 鏇村叧娉ㄩ摼璺満浼氬拰闃熷垪鍘嬪姏銆?
    2. `utility_rule` 鏇村叧娉ㄤ换鍔′环鍊笺€佷笉纭畾鎬у拰棰勬祴 reward銆?"""

    def __init__(self, router_name: str, mode: str) -> None:
        super().__init__(router_name)
        self.mode = mode

    def _find_action(self, candidate_actions: list[JointAction], action_name: str) -> JointAction | None:
        for action in candidate_actions:
            if action.action_name == action_name:
                return action
        return None

    def select_action(
        self,
        context: ContextVector,
        candidate_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        guidance_bundle: dict[str, object] | None = None,
    ) -> JointAction:
        _ = guidance_bundle
        if self.mode == "comm_conservative":
            if context.C_t < 0.20 or context.Q_t > 0.75:
                return self._find_action(candidate_actions, ACTION_FALLBACK_NAME) or candidate_actions[0]
            if context.C_t < 0.45:
                return self._find_action(candidate_actions, "balanced+metadata") or candidate_actions[0]
            return self._find_action(candidate_actions, "balanced+roi") or candidate_actions[0]

        if self.mode == "utility_rule":
            if context.V_t > 0.78 and context.U_t > 0.50 and context.B_t > 0.35 and context.C_t > 0.45:
                preferred = self._find_action(candidate_actions, "high_accuracy+roi_context")
                if preferred is not None:
                    return preferred
            if context.V_t > 0.55 or context.U_t > 0.55:
                preferred = self._find_action(candidate_actions, "balanced+roi")
                if preferred is not None:
                    return preferred
            return max(candidate_actions, key=lambda action: predicted_map[action.action_name].predicted_reward)

        return candidate_actions[0]
