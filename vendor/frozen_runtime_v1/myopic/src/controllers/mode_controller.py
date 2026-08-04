"""Mode controller interfaces and rule-based implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.types import LongTermState


class BaseModeController(ABC):
    """Unified interface for long-horizon mode selection."""

    def __init__(self, controller_cfg: dict[str, object]) -> None:
        self.controller_cfg = controller_cfg
        self.update_interval_slots = int(controller_cfg.get("update_interval_slots", 10))

    @abstractmethod
    def select_mode(self, long_state: LongTermState) -> str:
        """Select c_k / mode name from long-horizon state."""

    @abstractmethod
    def build_guidance(self, lambda_base: dict[str, float], mode_name: str) -> dict[str, object]:
        """Build z_k from lambda_base^k plus mode-dependent c_k."""


class RuleBasedModeController(BaseModeController):
    """Lightweight, interpretable mode controller with no training."""

    def __init__(self, controller_cfg: dict[str, object]) -> None:
        super().__init__(controller_cfg)
        self.thresholds = dict(controller_cfg.get("thresholds", {}))
        self.mode_cfgs = dict(controller_cfg.get("modes", {}))

    def select_mode(self, long_state: LongTermState) -> str:
        battery_low = float(self.thresholds.get("battery_low", 0.30))
        thermal_high = float(self.thresholds.get("thermal_high", 0.80))
        queue_high = float(self.thresholds.get("queue_high", 0.65))
        deadline_pressure_high = float(self.thresholds.get("deadline_pressure_high", 0.65))
        high_value_ratio_high = float(self.thresholds.get("high_value_ratio_high", 0.55))
        next_contact_long_s = float(self.thresholds.get("next_contact_long_s", 120.0))

        queue_pressure = float(long_state.queue_mb) / max(float(long_state.queue_max_mb), 1.0e-9)

        if float(long_state.battery_soc) <= battery_low or float(long_state.thermal_pressure) >= thermal_high:
            return "energy_saving"
        if queue_pressure >= queue_high:
            return "queue_draining"
        if float(long_state.recent_deadline_urgency) >= deadline_pressure_high:
            return "latency_critical"
        if float(long_state.recent_high_value_ratio) >= high_value_ratio_high:
            return "mission_priority"
        if float(long_state.next_contact_s) >= next_contact_long_s and float(long_state.queue_growth_rate) > 0.0:
            return "egress_conservative"
        return "balanced"

    def build_guidance(self, lambda_base: dict[str, float], mode_name: str) -> dict[str, object]:
        base_mode_cfg = dict(self.mode_cfgs.get("balanced", {}))
        mode_cfg = dict(self.mode_cfgs.get(mode_name, base_mode_cfg))
        multipliers = dict(mode_cfg.get("price_multipliers", {}))
        z_k = {
            key: max(0.0, float(lambda_base[key]) * float(multipliers.get(key, 1.0)))
            for key in ("lambda_E", "lambda_L", "lambda_Q", "lambda_S")
        }
        return {
            "mode_name": mode_name,
            "lambda_base": dict(lambda_base),
            "lambda_E": z_k["lambda_E"],
            "lambda_L": z_k["lambda_L"],
            "lambda_Q": z_k["lambda_Q"],
            "lambda_S": z_k["lambda_S"],
            "price_multipliers": {
                key: float(multipliers.get(key, 1.0))
                for key in ("lambda_E", "lambda_L", "lambda_Q", "lambda_S")
            },
            "exploration_scale": float(mode_cfg.get("exploration_scale", 1.0)),
            "model_complexity_bias": float(mode_cfg.get("model_complexity_bias", 0.0)),
            "payload_size_bias": float(mode_cfg.get("payload_size_bias", 0.0)),
            "high_value_bonus_scale": float(mode_cfg.get("high_value_bonus_scale", 1.0)),
            "safety_margin_scale": float(mode_cfg.get("safety_margin_scale", 1.0)),
            "c_k": {"mode_parameters": mode_cfg},
            "z_k": z_k,
        }
