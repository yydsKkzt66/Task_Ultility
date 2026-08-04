"""模型复杂度动作定义。"""

from __future__ import annotations


def get_model_action_levels() -> dict[str, float]:
    """返回模型复杂度的代表性等级。

    这些等级并不是具体模型名字，而是 case study 中的抽象复杂度层级。
    """

    return {
        "lightweight": 0.20,
        "balanced": 0.55,
        "high_accuracy": 0.95,
        "delay_store": 0.00,
    }
