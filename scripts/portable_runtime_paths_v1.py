"""Resolve the frozen runtime snapshots used by the final V25 experiments.

The portable, Git-tracked snapshots are preferred.  Environment variables can
override individual roots for an audited external installation.  The legacy
``D:/wt`` locations remain a last-resort fallback so existing workstations keep
working without configuration changes.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTABLE_ROOT = ROOT / "vendor" / "frozen_runtime_v1"


def _resolve(env_name: str, portable_relative: str, legacy: str) -> Path:
    override = os.environ.get(env_name)
    candidates = [
        Path(override).expanduser() if override else None,
        PORTABLE_ROOT / portable_relative,
        Path(legacy),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    # Return the portable location so a later preflight reports a useful,
    # repository-relative missing path instead of a workstation-specific one.
    return (PORTABLE_ROOT / portable_relative).resolve()


SOURCE_BUNDLE = _resolve(
    "LEO_V25_SOURCE_BUNDLE",
    "inputs/V25-MAIN-v1",
    "D:/wt/pilot/outputs/_updated_prop_ground_ack_v1/bundles/V25-MAIN-v1",
)
TOOL_DIR = _resolve(
    "LEO_NATIVE_TOOL_DIR",
    "native_tools/late_hv_native_direct_final_v6",
    "D:/wt/pilot_late_hv_native_direct_v1/tools/late_hv_native_direct_final_v6",
)
SOURCE_PROP = _resolve(
    "LEO_PROP_RUNTIME_ROOT",
    "prop",
    "D:/wt/pilot_updated_prop_ground_ack_v1",
)
SOURCE_B1_OFFLOAD = _resolve(
    "LEO_B1_OFFLOAD_RUNTIME_ROOT",
    "b1_offload",
    "D:/wt/pilot_frozen_runtime_native_direct_v1",
)
SOURCE_TUT = _resolve(
    "LEO_TUT_RUNTIME_ROOT",
    "tut",
    "D:/wt/tut_uoi_rate_lite",
)
SOURCE_MYOPIC = _resolve(
    "LEO_MYOPIC_RUNTIME_ROOT",
    "myopic",
    "D:/wt/pilot_myopic_frozen_calibrated_v1",
)
MYOPIC_PARAM = _resolve(
    "LEO_MYOPIC_PARAMETER_FILE",
    "inputs/myopic_frozen_calibration_v1.json",
    "D:/wt/pilot/outputs/_myopic_frozen_calibrated_v1/myopic_frozen_calibration_v1.json",
)
SOLAR_TRACE = _resolve(
    "LEO_SOLAR_TRACE_FILE",
    "inputs/solar_trace_one_orbit.csv",
    str(ROOT / "outputs" / "_v25_bupt1_power_calibrated_seed2026_v1" / "calibration" / "solar_trace_one_orbit.csv"),
)
SELECTED_POWER_CALIBRATION = _resolve(
    "LEO_POWER_CALIBRATION_FILE",
    "inputs/selected_power_calibration.json",
    str(ROOT / "outputs" / "_v25_bupt1_power_calibrated_seed2026_v1" / "config" / "selected_power_calibration.json"),
)


REQUIRED_PATHS = {
    "source_bundle": SOURCE_BUNDLE,
    "native_tools": TOOL_DIR,
    "prop_runtime": SOURCE_PROP,
    "b1_offload_runtime": SOURCE_B1_OFFLOAD,
    "tut_runtime": SOURCE_TUT,
    "myopic_runtime": SOURCE_MYOPIC,
    "myopic_parameters": MYOPIC_PARAM,
    "solar_trace": SOLAR_TRACE,
    "power_calibration": SELECTED_POWER_CALIBRATION,
}


def assert_portable_runtime_complete() -> None:
    missing = {name: str(path) for name, path in REQUIRED_PATHS.items() if not path.exists()}
    if missing:
        raise FileNotFoundError({"PORTABLE_RUNTIME_INCOMPLETE": missing})
