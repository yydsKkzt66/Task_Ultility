"""payload policy 閰嶇疆銆?"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.common.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAYLOAD_CONFIG_PATH = PROJECT_ROOT / "configs" / "payload_profiles.yaml"


@dataclass(frozen=True)
class PayloadPolicyProfile:
    """鎶借薄 payload policy 鎻忚堪銆?"""

    name: str
    base_size_mb: float
    information_level: float
    priority_weight: float
    enabled: bool = True
    parameters: dict[str, object] = field(default_factory=dict)

    @property
    def size_mb(self) -> float:
        return self.base_size_mb

    @property
    def information_factor(self) -> float:
        return self.information_level


class PayloadPolicyRegistry:
    def __init__(self, payload_profiles_cfg: dict[str, object]) -> None:
        profiles = payload_profiles_cfg.get("profiles", [])
        if not profiles:
            raise ValueError("payload_profiles.profiles must not be empty")

        self._profiles = {}
        for item in profiles:
            if not bool(item.get("enabled", True)):
                continue
            name = str(item["name"])
            information_level = float(item.get("information_factor", item.get("raw_info_gain", 0.0)))
            base_size_mb = float(item.get("size_mb", float(item.get("base_size_kb", 0.0)) / 1000.0))
            self._profiles[name] = PayloadPolicyProfile(
                name=name,
                base_size_mb=base_size_mb,
                information_level=information_level,
                priority_weight=float(item.get("priority_weight", 1.0)),
                enabled=bool(item.get("enabled", True)),
                parameters=dict(item),
            )

    def get(self, policy_name: str) -> PayloadPolicyProfile:
        return self._profiles[policy_name]


@lru_cache(maxsize=1)
def _default_registry() -> PayloadPolicyRegistry:
    return PayloadPolicyRegistry(load_yaml(DEFAULT_PAYLOAD_CONFIG_PATH))


def get_payload_policy_profile(
    policy_name: str,
    payload_profiles_cfg: dict[str, object] | None = None,
) -> PayloadPolicyProfile:
    """鑾峰彇 payload policy 鐨勫熀纭€鎻忚堪銆?"""

    registry = _default_registry() if payload_profiles_cfg is None else PayloadPolicyRegistry(payload_profiles_cfg)
    return registry.get(policy_name)
