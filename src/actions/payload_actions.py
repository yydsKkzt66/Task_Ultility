"""下传负载动作定义。"""

from __future__ import annotations


def get_payload_action_levels() -> dict[str, float]:
    """返回不同 payload policy 的代表性负载等级。"""

    return {
        "metadata": 0.20,
        "roi": 0.55,
        "roi_context": 0.95,
        "defer": 0.00,
    }
