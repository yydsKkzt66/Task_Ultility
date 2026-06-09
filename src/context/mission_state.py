"""任务状态辅助函数。"""

from __future__ import annotations

from src.common.types import MissionState


def build_initial_mission_state() -> MissionState:
    """创建初始 mission state。"""

    return MissionState()
