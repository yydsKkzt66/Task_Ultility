"""安全惩罚。"""

from __future__ import annotations


def compute_safety_penalty(safety_violation: int, reward_cfg: dict[str, float]) -> float:
    """根据安全约束违背情况计算惩罚。"""

    return float(reward_cfg["lambda_Omega"]) * float(safety_violation)
