"""Router 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.types import ContextVector, JointAction, PredictedMetrics


class BaseRouter(ABC):
    """统一 router 接口。"""

    def __init__(self, router_name: str) -> None:
        self.router_name = router_name

    @abstractmethod
    def select_action(
        self,
        context: ContextVector,
        safe_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
    ) -> JointAction:
        """在安全动作集合中选择动作。"""

    def update(self, context: ContextVector, action: JointAction, observed_reward: float) -> None:
        """使用 observed reward 更新 router。

        固定策略和规则策略默认不需要更新；bandit 路由器会覆盖此方法。
        """

        _ = (context, action, observed_reward)
