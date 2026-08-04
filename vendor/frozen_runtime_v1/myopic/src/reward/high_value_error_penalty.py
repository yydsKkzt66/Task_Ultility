"""高价值任务错判重罚项。"""

from __future__ import annotations

from src.common.types import ContextVector


def compute_high_value_error_penalty(
    context: ContextVector,
    reward_cfg: dict[str, float],
    miss_signal: float,
    false_alarm_signal: float,
) -> float:
    """计算高价值任务错判惩罚。

    这里故意把惩罚写成独立项，并通过 `high_value_error_penalty_enabled`
    开关控制。这样默认主仿真不受影响；只有独立实验支线会启用它。

    `miss_signal / false_alarm_signal` 可以是：
    - observed 阶段的 0/1 标记；
    - predicted 阶段的概率值。
    """

    if not reward_cfg.get("high_value_error_penalty_enabled", False):
        return 0.0

    high_value_threshold = float(reward_cfg.get("high_value_threshold", 0.75))
    if context.V_t < high_value_threshold:
        return 0.0

    severity = miss_signal + float(reward_cfg.get("false_alarm_penalty_scale", 0.50)) * false_alarm_signal
    if severity <= 0.0:
        return 0.0

    amplification = (
        1.0
        + float(reward_cfg.get("error_penalty_urgency_weight", 0.60)) * context.D_t
        + float(reward_cfg.get("error_penalty_difficulty_weight", 0.70)) * context.H_t
        + float(reward_cfg.get("error_penalty_uncertainty_weight", 0.70)) * context.U_t
    )
    base = float(reward_cfg.get("high_value_error_penalty_base", 1.0))
    multiplier = float(reward_cfg.get("high_value_error_penalty_multiplier", 5.0))
    return base * multiplier * context.V_t * severity * amplification
