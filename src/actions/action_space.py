"""联合动作空间。

第一版固定提供 5 个 joint actions，便于在 mock 场景中稳定运行，
同时也为后续替换真实模型池与通信策略保留清晰接口。
"""

from __future__ import annotations

from src.common.constants import DEFAULT_ACTION_SPECS
from src.common.types import JointAction


class ActionSpace:
    """提供固定候选动作集合。"""

    def __init__(self) -> None:
        self._actions = [
            JointAction(
                action_name=action_name,
                model_name=model_name,
                payload_policy=payload_policy,
                transmission_mode=transmission_mode,
                expected_payload_level=payload_level,
                expected_compute_level=compute_level,
            )
            for action_name, model_name, payload_policy, transmission_mode, payload_level, compute_level in DEFAULT_ACTION_SPECS
        ]

    def list_actions(self) -> list[JointAction]:
        """返回当前版本支持的全部联合动作。"""

        return list(self._actions)

    def get_action(self, action_name: str) -> JointAction:
        """按名字查动作。"""

        for action in self._actions:
            if action.action_name == action_name:
                return action
        raise KeyError(f"未知动作: {action_name}")
