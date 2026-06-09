"""信息充分度建模。"""

from __future__ import annotations

from src.common.types import ContextVector


def compute_information_requirement(context: ContextVector, reward_cfg: dict[str, float]) -> float:
    """计算当前任务所需信息量 I_req(x_t)。"""

    return (
        float(reward_cfg["info_base"])
        + float(reward_cfg["info_value_weight"]) * context.V_t
        + float(reward_cfg["info_difficulty_weight"]) * context.H_t
        + float(reward_cfg["info_uncertainty_weight"]) * context.U_t
        + float(reward_cfg["info_urgency_weight"]) * context.D_t
    )


def compute_information_sufficiency(
    context: ContextVector,
    payload_information: float,
    reward_cfg: dict[str, float],
) -> float:
    """计算信息充分度 S_info。

    这里强调“刚好够用”，而不是 payload 越大越好。
    因此使用 `min(1, I / I_req)` 的饱和形式。
    """

    requirement = max(compute_information_requirement(context, reward_cfg), 1.0e-9)
    return min(1.0, payload_information / requirement)
