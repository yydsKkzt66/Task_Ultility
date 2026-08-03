"""鍥哄畾鍔ㄤ綔鍩虹嚎銆?"""

from __future__ import annotations

from src.common.constants import ACTION_FALLBACK_NAME
from src.common.types import ContextVector, JointAction, PredictedMetrics
from src.routers.base_router import BaseRouter


class FixedRouter(BaseRouter):
    """濮嬬粓浼樺厛閫夋嫨鍚屼竴涓姩浣滅殑 baseline銆?"""

    def __init__(self, router_name: str, preferred_action_name: str) -> None:
        super().__init__(router_name)
        self.preferred_action_name = preferred_action_name

    def select_action(
        self,
        context: ContextVector,
        candidate_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        guidance_bundle: dict[str, object] | None = None,
    ) -> JointAction:
        _ = (context, predicted_map, guidance_bundle)
        for action in candidate_actions:
            if action.action_name == self.preferred_action_name:
                return action
        for action in candidate_actions:
            if action.action_name == ACTION_FALLBACK_NAME:
                return action
        return candidate_actions[0]
