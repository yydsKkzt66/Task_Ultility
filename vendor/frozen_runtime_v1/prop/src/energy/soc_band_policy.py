"""Optional SOC-band guidance for long-horizon energy-health audits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SocBandPolicyEffect:
    soc_band: str
    lambda_E_multiplier: float
    model_complexity_bias_delta: float
    payload_size_bias_delta: float
    high_value_bonus_scale: float
    safety_margin_scale: float
    hard_safety_zone: bool


def classify_soc_band(
    battery_soc: float,
    soft_target: float = 0.70,
    strong_guard: float = 0.50,
    hard_min: float = 0.20,
) -> str:
    """Classify SOC into nominal health bands without redefining the hard bound."""

    soc = float(battery_soc)
    if soc < float(hard_min):
        return "hard_safety"
    if soc < float(strong_guard):
        return "strong_conservation"
    if soc < float(soft_target):
        return "soft_pressure"
    return "normal"


def soc_band_policy_effect(battery_soc: float, cfg: dict[str, object] | None) -> SocBandPolicyEffect:
    cfg = cfg or {}
    soft_target = float(cfg.get("soft_target", 0.70))
    strong_guard = float(cfg.get("strong_guard", 0.50))
    hard_min = float(cfg.get("hard_min", 0.20))
    band = classify_soc_band(battery_soc, soft_target, strong_guard, hard_min)

    if band == "hard_safety":
        prefix = "hard"
        hard_zone = True
    elif band == "strong_conservation":
        prefix = "strong"
        hard_zone = False
    elif band == "soft_pressure":
        prefix = "soft"
        hard_zone = False
    else:
        return SocBandPolicyEffect("normal", 1.0, 0.0, 0.0, 1.0, 1.0, False)

    return SocBandPolicyEffect(
        soc_band=band,
        lambda_E_multiplier=float(cfg.get(f"{prefix}_lambda_E_multiplier", 1.0)),
        model_complexity_bias_delta=float(cfg.get(f"{prefix}_model_complexity_bias_delta", 0.0)),
        payload_size_bias_delta=float(cfg.get(f"{prefix}_payload_size_bias_delta", 0.0)),
        high_value_bonus_scale=float(cfg.get(f"{prefix}_high_value_bonus_scale", 1.0)),
        safety_margin_scale=float(cfg.get(f"{prefix}_safety_margin_scale", 1.0)),
        hard_safety_zone=hard_zone,
    )


def apply_soc_band_policy(
    guidance_bundle: dict[str, object],
    battery_soc: float,
    cfg: dict[str, object] | None,
) -> dict[str, object]:
    """Apply optional SOC-band multipliers to a guidance bundle."""

    if not cfg or not bool(cfg.get("enabled", False)):
        return guidance_bundle

    guidance = dict(guidance_bundle)
    effect = soc_band_policy_effect(battery_soc, cfg)

    guidance["soc_band"] = effect.soc_band
    guidance["soc_band_lambda_E_multiplier"] = effect.lambda_E_multiplier
    guidance["soc_band_hard_safety_zone"] = int(effect.hard_safety_zone)
    guidance["lambda_E"] = float(guidance.get("lambda_E", 0.0)) * effect.lambda_E_multiplier
    guidance["model_complexity_bias"] = (
        float(guidance.get("model_complexity_bias", 0.0)) + effect.model_complexity_bias_delta
    )
    guidance["payload_size_bias"] = float(guidance.get("payload_size_bias", 0.0)) + effect.payload_size_bias_delta
    guidance["high_value_bonus_scale"] = (
        float(guidance.get("high_value_bonus_scale", 1.0)) * effect.high_value_bonus_scale
    )
    guidance["safety_margin_scale"] = float(guidance.get("safety_margin_scale", 1.0)) * effect.safety_margin_scale
    if "lambda_after_z" in guidance and isinstance(guidance["lambda_after_z"], dict):
        lambda_after_z = dict(guidance["lambda_after_z"])
        lambda_after_z["lambda_E"] = float(guidance.get("lambda_E", 0.0))
        guidance["lambda_after_z"] = lambda_after_z
    return guidance
