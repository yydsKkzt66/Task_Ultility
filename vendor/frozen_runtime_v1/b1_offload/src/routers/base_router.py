"""Router 鎶借薄鍩虹被銆?"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.types import ContextVector, JointAction, PredictedMetrics


class BaseRouter(ABC):
    """缁熶竴 router 鎺ュ彛銆?"""

    def __init__(self, router_name: str) -> None:
        self.router_name = router_name
        self.use_lyapunov_pricing = False
        self.use_mode_controller = False
        self.use_guided_utility = False
        self.last_selected_action_score: float | None = None
        self.last_score_components: dict[str, object] = {}
        self.last_all_score_components: dict[str, dict[str, object]] = {}

    @abstractmethod
    def select_action(
        self,
        context: ContextVector,
        candidate_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        guidance_bundle: dict[str, object] | None = None,
    ) -> JointAction:
        """鍦ㄥ€欓€夊姩浣滈泦鍚堜腑杩斿洖 raw action銆?"""

    def update(self, context: ContextVector, action: JointAction, observed_reward: float) -> None:
        """浣跨敤 observed reward 鏇存柊 router銆?
        鍥哄畾绛栫暐鍜岃鍒欑瓥鐣ラ粯璁や笉闇€瑕佹洿鏂帮紱bandit 璺敱鍣ㄤ細瑕嗙洊姝ゆ柟娉曘€?"""

        _ = (context, action, observed_reward)
