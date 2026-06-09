"""Mock detector 的可选错判风险模型。

这个模块只在独立实验支线中启用：
默认主仿真不会打开它，因此不会改变当前已有结果。
"""

from __future__ import annotations

import random

from src.common.types import ProfileMetrics, TaskSample


def is_error_model_enabled(error_model_cfg: dict[str, float] | None) -> bool:
    """判断是否启用可选错判模型。"""

    return bool(error_model_cfg and error_model_cfg.get("enabled", False))


def estimate_miss_probability(
    task: TaskSample,
    profile: ProfileMetrics,
    estimated_quality: float,
    error_model_cfg: dict[str, float] | None,
) -> float:
    """估计当前模型在当前任务上的漏检风险。

    这里的目标不是复刻真实检测器，而是提供一个可解释、可控的 synthetic 风险项，
    让“高价值任务做错需要重罚”的实验支线有稳定的错判事件来源。
    """

    if not is_error_model_enabled(error_model_cfg):
        return 0.0

    threshold = float(error_model_cfg.get("high_value_threshold", 0.75))
    high_value_bonus = float(error_model_cfg.get("high_value_bonus", 0.0)) if task.task_value >= threshold else 0.0
    risk = (
        float(error_model_cfg.get("base_miss_probability", 0.03))
        + float(error_model_cfg.get("difficulty_weight", 0.30)) * task.difficulty
        + float(error_model_cfg.get("uncertainty_weight", 0.22)) * task.uncertainty
        + float(error_model_cfg.get("low_quality_weight", 0.55)) * (1.0 - estimated_quality)
        + float(error_model_cfg.get("low_robustness_weight", 0.18)) * (1.0 - profile.robustness)
        + high_value_bonus * task.task_value
    )
    min_risk = float(error_model_cfg.get("min_error_probability", 0.02))
    max_risk = float(error_model_cfg.get("max_error_probability", 0.95))
    return max(min_risk, min(max_risk, risk))


def estimate_false_alarm_probability(
    task: TaskSample,
    estimated_quality: float,
    miss_probability: float,
    error_model_cfg: dict[str, float] | None,
) -> float:
    """估计误警风险。"""

    if not is_error_model_enabled(error_model_cfg):
        return 0.0

    false_alarm_scale = float(error_model_cfg.get("false_alarm_scale", 0.35))
    quality_term = 0.70 + 0.30 * (1.0 - estimated_quality)
    value_term = 0.80 + 0.20 * task.task_value
    false_alarm_probability = miss_probability * false_alarm_scale * quality_term * value_term
    return max(0.0, min(0.95, false_alarm_probability))


def sample_error_flags(
    seed: str,
    miss_probability: float,
    false_alarm_probability: float,
) -> tuple[bool, bool]:
    """基于固定 seed 采样漏检/误警标记。

    使用字符串 seed 保证同一个 task-profile 组合在不同运行中复现相同结果。
    """

    rng = random.Random(seed)
    miss_flag = rng.random() < miss_probability
    false_alarm_flag = False if miss_flag else (rng.random() < false_alarm_probability)
    return miss_flag, false_alarm_flag
