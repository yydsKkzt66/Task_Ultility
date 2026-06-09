"""模型 profiling 存储。"""

from __future__ import annotations

from src.common.types import ProfileMetrics


class ProfilingStore:
    """管理 lightweight / balanced / high_accuracy 的代表性 profile。"""

    DEFAULT_PROFILE_DEFINITIONS = {
        "tiny": {"latency_s": 0.8, "quality_base": 0.58, "estimated_power_w": 8.0, "robustness": 0.52},
        "small": {"latency_s": 1.6, "quality_base": 0.74, "estimated_power_w": 13.0, "robustness": 0.68},
        "medium": {"latency_s": 2.8, "quality_base": 0.88, "estimated_power_w": 19.0, "robustness": 0.82},
        "defer": {"latency_s": 0.0, "quality_base": 0.0, "estimated_power_w": 0.0, "robustness": 0.0},
    }

    def __init__(self, profile_overrides: dict[str, dict[str, float]] | None = None) -> None:
        """创建 profiling store。

        `profile_overrides` 用于参数敏感性实验：
        它允许脚本在不改动主代码路径的情况下，给 `tiny/small/medium`
        注入新的 latency / quality / power / robustness 组合。
        """

        profile_definitions = {
            profile_name: dict(values)
            for profile_name, values in self.DEFAULT_PROFILE_DEFINITIONS.items()
        }
        for profile_name, overrides in (profile_overrides or {}).items():
            if profile_name not in profile_definitions:
                raise KeyError(f"未知 profile: {profile_name}")
            profile_definitions[profile_name].update(overrides)

        self._profiles = {
            profile_name: ProfileMetrics(profile_name=profile_name, **values)
            for profile_name, values in profile_definitions.items()
        }

    def get(self, profile_name: str) -> ProfileMetrics:
        """按名字获取 profile。"""

        return self._profiles[profile_name]

    def export_table(self) -> list[dict[str, float | str]]:
        """导出 profile 表，便于脚本查看。"""

        return [
            {
                "profile_name": profile.profile_name,
                "latency_s": profile.latency_s,
                "quality_base": profile.quality_base,
                "estimated_power_w": profile.estimated_power_w,
                "robustness": profile.robustness,
            }
            for profile in self._profiles.values()
        ]
