"""Synthetic resource trace loader."""

from __future__ import annotations

import math

from src.common.types import ResourceTrace


def _merged_resource_cfg(timeline_cfg: dict[str, object]) -> dict[str, object]:
    resource_cfg = timeline_cfg.get("resource", {})
    if isinstance(resource_cfg, dict):
        return {**timeline_cfg, **resource_cfg}
    return dict(timeline_cfg)


def load_synthetic_resource_trace(
    num_slots: int,
    timeline_cfg: dict[str, object],
) -> list[ResourceTrace]:
    """Generate a synthetic orbit-aware resource trace."""

    cfg = _merged_resource_cfg(timeline_cfg)
    slot_seconds = float(timeline_cfg.get("slot_seconds", cfg.get("slot_seconds", 10.0)))

    orbit_period_s = float(
        cfg.get(
            "orbit_period_s",
            float(cfg.get("orbit_period_slots", 0.0)) * slot_seconds
            if "orbit_period_slots" in cfg
            else float(cfg.get("orbit_period_min", 90.0)) * 60.0,
        )
    )
    orbit_period_s = max(slot_seconds, orbit_period_s)

    eclipse_duration_s = cfg.get("eclipse_duration_s")
    sunlit_duration_s = cfg.get("sunlit_duration_s")
    if eclipse_duration_s is not None:
        eclipse_duration_s = max(0.0, min(float(eclipse_duration_s), orbit_period_s))
        sunlit_duration_s = max(0.0, orbit_period_s - eclipse_duration_s)
    elif sunlit_duration_s is not None:
        sunlit_duration_s = max(0.0, min(float(sunlit_duration_s), orbit_period_s))
        eclipse_duration_s = max(0.0, orbit_period_s - sunlit_duration_s)
    else:
        sunlight_fraction = max(0.05, min(0.95, float(cfg.get("sunlight_fraction", 0.62))))
        sunlit_duration_s = orbit_period_s * sunlight_fraction
        eclipse_duration_s = orbit_period_s - sunlit_duration_s

    day_cycle_orbits = max(1, int(cfg.get("day_cycle_orbits", 16)))
    solar_peak_w = float(cfg.get("solar_peak_w", cfg.get("solar_charge_power_w", cfg.get("solar_peak_j", 38.0))))
    solar_shape = str(cfg.get("solar_shape", "sinusoidal")).strip().lower()
    solar_shape_exponent = float(cfg.get("solar_shape_exponent", 1.0))
    base_power_w = float(cfg.get("idle_power_w", cfg.get("resource_base_power_w", 2.8)))
    power_variation_w = float(cfg.get("resource_power_variation_w", 0.6))
    sunlit_temperature_c = float(cfg.get("sunlit_temperature_c", 18.0))
    eclipse_temperature_c = float(cfg.get("eclipse_temperature_c", 4.0))
    temperature_cycle_variation_c = float(cfg.get("temperature_cycle_variation_c", 2.0))
    phase_offset_s = float(cfg.get("orbit_phase_offset_s", 0.0))

    traces: list[ResourceTrace] = []
    for slot_index in range(num_slots):
        current_time_s = slot_index * slot_seconds + phase_offset_s
        orbit_index = int(current_time_s // orbit_period_s)
        orbit_time_s = current_time_s % orbit_period_s
        orbit_phase = orbit_time_s / orbit_period_s
        is_sunlit = orbit_time_s < float(sunlit_duration_s)
        is_eclipse = not is_sunlit
        day_index = orbit_index // day_cycle_orbits

        if is_sunlit:
            normalized_sun_phase = orbit_time_s / max(float(sunlit_duration_s), 1.0e-9)
            sunlight_shape = math.sin(math.pi * normalized_sun_phase)
            if solar_shape in {"sine_power", "sin_power"}:
                sunlight_shape = max(0.0, sunlight_shape) ** max(solar_shape_exponent, 0.0)
        else:
            sunlight_shape = 0.0

        multi_day_scale = 0.96 + 0.08 * math.sin(2.0 * math.pi * orbit_index / day_cycle_orbits + 0.4)
        solar_power_w = solar_peak_w * max(0.0, sunlight_shape) * multi_day_scale if is_sunlit else 0.0
        solar_energy_j = solar_power_w * slot_seconds

        orbital_power_variation = power_variation_w * (0.5 + 0.5 * math.cos(2.0 * math.pi * orbit_phase + 0.3))
        eclipse_housekeeping_penalty = 0.18 if is_eclipse else 0.0
        task_day_scale = 1.0 + 0.03 * math.sin(2.0 * math.pi * day_index / 5.0 + 0.2)
        current_base_power_w = (base_power_w + orbital_power_variation + eclipse_housekeeping_penalty) * task_day_scale

        thermal_anchor = sunlit_temperature_c if is_sunlit else eclipse_temperature_c
        temperature_reference_c = thermal_anchor + temperature_cycle_variation_c * math.sin(
            2.0 * math.pi * orbit_phase + (0.8 if is_sunlit else -0.6)
        )

        traces.append(
            ResourceTrace(
                slot_index=slot_index,
                solar_energy_j=solar_energy_j,
                base_power_w=current_base_power_w,
                temperature_reference_c=temperature_reference_c,
                is_sunlit=is_sunlit,
                is_eclipse=is_eclipse,
                orbit_index=orbit_index,
                day_index=day_index,
                orbit_phase=orbit_phase,
                solar_power_w=solar_power_w,
            )
        )
    return traces
