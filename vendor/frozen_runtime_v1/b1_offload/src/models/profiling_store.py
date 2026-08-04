"""妯″瀷 profiling 瀛樺偍銆?"""

from __future__ import annotations

from src.common.types import ProfileMetrics


class ProfilingStore:
    """绠＄悊妯″瀷 profile 琛ㄣ€?"""

    def __init__(
        self,
        profile_definitions: list[dict[str, object]] | None = None,
        profile_overrides: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """鍒涘缓 profiling store銆?
        `profile_overrides` 鐢ㄤ簬鍙傛暟鏁忔劅鎬у疄楠岋紝鍦ㄤ笉鏀瑰姩鍩虹 config 鐨勫墠鎻愪笅娉ㄥ叆鏂扮殑 E/L/Q/robustness 缁勫悎銆?"""

        if not profile_definitions:
            raise ValueError("model_pool.profiles must not be empty")

        definitions_by_name: dict[str, dict[str, object]] = {}
        for item in profile_definitions:
            if not bool(item.get("enabled", True)):
                continue
            definitions_by_name[str(item["name"])] = dict(item)

        for profile_name, overrides in (profile_overrides or {}).items():
            if profile_name not in definitions_by_name:
                raise KeyError(f"Unknown profile override target: {profile_name}")
            normalized = self._normalize_profile_fields(profile_name, overrides)
            definitions_by_name[profile_name].update(normalized)

        self._profiles = {
            profile_name: ProfileMetrics(
                profile_name=profile_name,
                latency_s=float(values["latency_s"]),
                quality_base=float(values["quality_base"]),
                energy_j=float(values["energy_j"]),
                robustness_to_difficulty=float(values["robustness_to_difficulty"]),
                thermal_gain_scale=float(values.get("thermal_gain_scale", 1.0)),
                enabled=bool(values.get("enabled", True)),
            )
            for profile_name, values in definitions_by_name.items()
        }

    @staticmethod
    def _normalize_profile_fields(profile_name: str, values: dict[str, object]) -> dict[str, object]:
        normalized = dict(values)
        if "robustness" in normalized and "robustness_to_difficulty" not in normalized:
            normalized["robustness_to_difficulty"] = normalized["robustness"]
        if "estimated_power_w" in normalized and "energy_j" not in normalized:
            latency = float(normalized.get("latency_s", 0.0))
            normalized["energy_j"] = float(normalized["estimated_power_w"]) * latency

        required = {"latency_s", "quality_base", "energy_j", "robustness_to_difficulty"}
        missing = [key for key in required if key not in normalized]
        if missing:
            raise KeyError(f"Profile {profile_name} missing required fields: {missing}")
        return normalized

    def get(self, profile_name: str) -> ProfileMetrics:
        """鎸夊悕瀛楄幏鍙?profile銆?"""

        return self._profiles[profile_name]

    def export_table(self) -> list[dict[str, float | str | bool]]:
        """瀵煎嚭 profile 琛紝渚夸簬鑴氭湰鏌ョ湅銆?"""

        return [
            {
                "profile_name": profile.profile_name,
                "latency_s": profile.latency_s,
                "quality_base": profile.quality_base,
                "energy_j": profile.energy_j,
                "estimated_power_w": profile.estimated_power_w,
                "robustness_to_difficulty": profile.robustness_to_difficulty,
                "robustness": profile.robustness,
                "thermal_gain_scale": profile.thermal_gain_scale,
                "enabled": profile.enabled,
            }
            for profile in self._profiles.values()
        ]
