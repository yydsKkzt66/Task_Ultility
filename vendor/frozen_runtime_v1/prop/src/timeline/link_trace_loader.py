"""Synthetic link/contact trace loader."""

from __future__ import annotations

import math
from typing import Any

from src.common.types import LinkTrace


def _as_window_tuple(item: Any) -> tuple[float, float]:
    if isinstance(item, dict):
        return float(item.get("start_s", 0.0)), float(item.get("end_s", item.get("duration_s", 0.0)))
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return float(item[0]), float(item[1])
    raise ValueError(f"Invalid contact window specification: {item!r}")


def _window_list(cfg: dict[str, object], slot_seconds: float) -> tuple[float, list[tuple[float, float]]]:
    if "contact_windows_s" in cfg:
        period_s = max(slot_seconds, float(cfg.get("contact_period_s", 18.0 * slot_seconds)))
        windows = [_as_window_tuple(item) for item in cfg.get("contact_windows_s", [])]  # type: ignore[arg-type]
        return period_s, [(max(0.0, start), max(0.0, end)) for start, end in windows if end > start]

    if "contact_period_s" in cfg or "contact_duration_s" in cfg or "contact_phase_offset_s" in cfg:
        period_s = max(slot_seconds, float(cfg.get("contact_period_s", 18.0 * slot_seconds)))
        duration_s = max(0.0, min(float(cfg.get("contact_duration_s", 5.0 * slot_seconds)), period_s))
        phase_offset_s = float(cfg.get("contact_phase_offset_s", 4.0 * slot_seconds)) % period_s
        return period_s, [(phase_offset_s, min(period_s, phase_offset_s + duration_s))]

    period_s = 18.0 * slot_seconds
    return period_s, [(4.0 * slot_seconds, 9.0 * slot_seconds), (13.0 * slot_seconds, 16.0 * slot_seconds)]


def _in_absolute_outage(current_time_s: float, outage_windows_s: list[tuple[float, float]]) -> bool:
    return any(start <= current_time_s < end for start, end in outage_windows_s)


def _contact_remaining_s(period_phase_s: float, windows: list[tuple[float, float]]) -> tuple[bool, float]:
    for start_s, end_s in windows:
        if start_s <= period_phase_s < end_s:
            return True, max(0.0, end_s - period_phase_s)
    return False, 0.0


def load_synthetic_link_trace(
    num_slots: int,
    slot_seconds: float,
    contact_cfg: dict[str, object] | None = None,
) -> list[LinkTrace]:
    """Generate a synthetic link trace using physical seconds for contact cadence."""

    cfg = contact_cfg or {}
    period_s, windows = _window_list(cfg, slot_seconds)
    outage_windows_s = [_as_window_tuple(item) for item in cfg.get("outage_windows_s", [])]  # type: ignore[arg-type]
    throughput_min_mbps = float(cfg.get("throughput_min_mbps", 1.5))
    throughput_max_mbps = float(cfg.get("throughput_max_mbps", 6.0))
    snr_min_db = float(cfg.get("snr_min_db", 5.0))
    snr_span_db = float(cfg.get("snr_span_db", 12.0))
    quality_period_s = float(cfg.get("link_quality_period_s", 8.0 * slot_seconds))
    quality_phase = float(cfg.get("link_quality_phase", 0.5))

    traces: list[LinkTrace] = []
    for slot_index in range(num_slots):
        current_time_s = slot_index * slot_seconds
        period_phase_s = current_time_s % period_s
        in_contact, remaining_window_s = _contact_remaining_s(period_phase_s, windows)
        if _in_absolute_outage(current_time_s, outage_windows_s):
            in_contact = False
            remaining_window_s = 0.0

        link_quality = max(0.0, min(1.0, 0.45 + 0.35 * math.sin(current_time_s / max(quality_period_s, 1.0e-9) + quality_phase)))
        throughput_mbps = (throughput_min_mbps + (throughput_max_mbps - throughput_min_mbps) * link_quality) if in_contact else 0.0
        snr_db = (snr_min_db + snr_span_db * link_quality) if in_contact else -3.0
        traces.append(
            LinkTrace(
                slot_index=slot_index,
                in_contact=in_contact,
                snr_db=snr_db,
                throughput_mbps=throughput_mbps,
                link_quality=link_quality if in_contact else 0.0,
                remaining_window_s=remaining_window_s,
            )
        )
    return traces
