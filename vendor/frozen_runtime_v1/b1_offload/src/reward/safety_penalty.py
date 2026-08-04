"""瀹夊叏鎯╃綒銆?"""

from __future__ import annotations


def compute_safety_penalty(safety_violation: int, reward_cfg: dict[str, float]) -> float:
    """鏍规嵁瀹夊叏绾︽潫杩濊儗鎯呭喌璁＄畻鎯╃綒銆?"""

    return float(reward_cfg.get("lambda_Omega", reward_cfg.get("violation_penalty", 0.0))) * float(safety_violation)
