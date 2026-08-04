"""任务难度与不确定性生成逻辑。"""

from __future__ import annotations

import math
import random


def sample_task_value(slot_index: int, rng: random.Random) -> float:
    """生成任务价值，混合周期趋势与局部随机性。"""

    base = 0.45 + 0.30 * math.sin(slot_index / 9.0)
    return max(0.10, min(1.0, base + rng.uniform(-0.15, 0.15)))


def sample_deadline_s(slot_index: int, rng: random.Random) -> float:
    """生成任务截止期限。

    数值越小表示越紧急。
    """

    deadline = 35.0 + 65.0 * (0.5 + 0.5 * math.cos(slot_index / 11.0))
    return max(10.0, deadline + rng.uniform(-8.0, 8.0))


def sample_difficulty(slot_index: int, rng: random.Random) -> float:
    """生成输入难度。"""

    difficulty = 0.40 + 0.30 * math.sin(slot_index / 7.0 + 0.8)
    return max(0.05, min(1.0, difficulty + rng.uniform(-0.12, 0.12)))


def sample_uncertainty(slot_index: int, rng: random.Random) -> float:
    """生成模型先验不确定性。"""

    uncertainty = 0.35 + 0.25 * math.cos(slot_index / 13.0 + 0.2)
    return max(0.05, min(1.0, uncertainty + rng.uniform(-0.12, 0.12)))
