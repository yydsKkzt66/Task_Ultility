"""鑱斿悎鍔ㄤ綔绌洪棿銆?"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.common.config import load_yaml
from src.common.types import JointAction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION_CONFIG_PATH = PROJECT_ROOT / "configs" / "action_space.yaml"


@lru_cache(maxsize=1)
def _default_action_space_cfg() -> dict[str, object]:
    return load_yaml(DEFAULT_ACTION_CONFIG_PATH)


class ActionSpace:
    """鎻愪緵閰嶇疆鍖栫殑 joint action 闆嗗悎銆?"""

    def __init__(self, action_space_cfg: dict[str, object] | None = None) -> None:
        action_space_cfg = action_space_cfg or _default_action_space_cfg()
        self._actions = []
        for item in action_space_cfg.get("actions", []):
            if not bool(item.get("enabled", True)):
                continue
            self._actions.append(
                JointAction(
                    action_name=str(item["action_name"]),
                    model_name=str(item["model_name"]),
                    payload_policy=str(item["payload_policy"]),
                    transmission_mode=str(item.get("transmission_option", item.get("transmission_mode", "contact_aware"))),
                    expected_payload_level=float(item["expected_payload_level"]),
                    expected_compute_level=float(item["expected_compute_level"]),
                    payload_plan=str(item.get("payload_plan", item["payload_policy"])),
                )
            )

        if not self._actions:
            raise ValueError("action_space.actions must not be empty")

    def list_actions(self) -> list[JointAction]:
        """杩斿洖褰撳墠鐗堟湰鏀寔鐨勫叏閮ㄨ仈鍚堝姩浣溿€?"""

        return list(self._actions)

    def get_action(self, action_name: str) -> JointAction:
        """鎸夊悕瀛楁煡鍔ㄤ綔銆?"""

        for action in self._actions:
            if action.action_name == action_name:
                return action
        raise KeyError(f"Unknown action: {action_name}")
