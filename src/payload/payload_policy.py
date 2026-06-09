"""payload policy 配置。"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.constants import PAYLOAD_BASE_SIZE_MB, PAYLOAD_INFORMATION_LEVEL


@dataclass(frozen=True)
class PayloadPolicyProfile:
    """抽象 payload policy 描述。"""

    name: str
    base_size_mb: float
    information_level: float


def get_payload_policy_profile(policy_name: str) -> PayloadPolicyProfile:
    """获取 payload policy 的基础描述。"""

    return PayloadPolicyProfile(
        name=policy_name,
        base_size_mb=PAYLOAD_BASE_SIZE_MB[policy_name],
        information_level=PAYLOAD_INFORMATION_LEVEL[policy_name],
    )
