"""任务效用相关函数。"""

from __future__ import annotations

from src.common.types import ContextVector
from src.payload.information_sufficiency import compute_information_requirement
from src.payload.information_sufficiency import compute_information_sufficiency as _payload_information_sufficiency


def compute_information_sufficiency(
    context: ContextVector,
    payload_information: float,
    reward_cfg: dict[str, float],
) -> float:
    """对 payload 层的信息充分度函数做 reward 侧封装。"""

    return _payload_information_sufficiency(context, payload_information, reward_cfg)


def compute_information_requirement_from_context(
    context: ContextVector,
    reward_cfg: dict[str, float],
) -> float:
    """瀵?payload 灞傜殑 I_req 鍋氫竴灞傚皝瑁呫€?"""

    return compute_information_requirement(context, reward_cfg)


def compute_quality_requirement(
    context: ContextVector,
    reward_cfg: dict[str, float],
) -> float:
    """璁＄畻褰撳墠浠诲姟瀵规娴嬭川閲忕殑闇€姹?Q_req銆?"""

    quality_req = (
        float(reward_cfg.get("quality_base_requirement", 0.40))
        + float(reward_cfg.get("quality_value_weight", 0.18)) * context.V_t
        + float(reward_cfg.get("quality_difficulty_weight", 0.22)) * context.H_t
        + float(reward_cfg.get("quality_uncertainty_weight", 0.16)) * context.U_t
        + float(reward_cfg.get("quality_urgency_weight", 0.12)) * context.D_t
    )
    return max(0.0, min(1.0, quality_req))


def compute_dynamic_energy_weight(
    context: ContextVector,
    reward_cfg: dict[str, float],
) -> float:
    """计算动态能耗惩罚权重。

    lambda_E(t) = lambda_E_base * (1 + eta_B * (1 - B_t) + eta_Theta * Theta_t)
    """

    lambda_E_base = float(reward_cfg.get("lambda_E_base", reward_cfg.get("energy_penalty", 0.0)))
    eta_B = float(reward_cfg["eta_B"])
    eta_Theta = float(reward_cfg["eta_Theta"])
    return lambda_E_base * (1.0 + eta_B * (1.0 - context.B_t) + eta_Theta * context.Theta_t)
