"""任务效用相关函数。"""

from __future__ import annotations

from src.common.types import ContextVector
from src.payload.information_sufficiency import compute_information_sufficiency as _payload_information_sufficiency


def compute_information_sufficiency(
    context: ContextVector,
    payload_information: float,
    reward_cfg: dict[str, float],
) -> float:
    """对 payload 层的信息充分度函数做 reward 侧封装。"""

    return _payload_information_sufficiency(context, payload_information, reward_cfg)


def compute_dynamic_energy_weight(
    context: ContextVector,
    reward_cfg: dict[str, float],
) -> float:
    """计算动态能耗惩罚权重。

    lambda_E(t) = lambda_E_base * (1 + eta_B * (1 - B_t) + eta_Theta * Theta_t)
    """

    lambda_E_base = float(reward_cfg["lambda_E_base"])
    eta_B = float(reward_cfg["eta_B"])
    eta_Theta = float(reward_cfg["eta_Theta"])
    return lambda_E_base * (1.0 + eta_B * (1.0 - context.B_t) + eta_Theta * context.Theta_t)
