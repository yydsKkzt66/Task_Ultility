"""Router placeholder for the B1-Enhanced architecture-equivalent baseline."""

from __future__ import annotations

from src.common.types import ContextVector, JointAction, PredictedMetrics
from src.routers.base_router import BaseRouter


class B1EnhancedRouter(BaseRouter):
    """Registration-only router for unified benchmark naming.

    The B1-Enhanced method has an independent compression, scheduling, and
    ground-processing lifecycle, so it is executed by its dedicated runner.
    This router exists only so config-driven benchmark registries can recognize
    the method name without enabling proposed-mode controllers or bandits.
    """

    def __init__(self, router_name: str = "b1_enhanced_contact_compressed") -> None:
        super().__init__(router_name)
        self.use_lyapunov_pricing = False
        self.use_mode_controller = False
        self.use_guided_utility = False

    def select_action(
        self,
        context: ContextVector,
        candidate_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        guidance_bundle: dict[str, object] | None = None,
    ) -> JointAction:
        _ = (context, predicted_map, guidance_bundle)
        for action in candidate_actions:
            if action.payload_policy == "defer" or action.model_name == "delay_store":
                return action
        return candidate_actions[0]
