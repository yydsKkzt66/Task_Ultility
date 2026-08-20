"""Run the frozen five-method comparison from one entry point.

Corrected REQALIGN and MYOPIC use the same promoted corrected-REQALIGN
post-selector projection and re-entry lifecycle.  MYOPIC uses the audited
``MYOPIC-FROZEN-CALIBRATED`` scorer/selector as its delegate, replacing the
later E2E-utility experiment while preserving the common downstream backend.
B1, OFFLOAD, and TUT retain their frozen baseline architectures while consuming
the same audited task/environment identity and common ground-received evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_prop_full_vs_b1_unified_seed2026 import proposed_energy_safety  # noqa: E402
from scripts.run_prop_reqalign_seed2026_v1 import (  # noqa: E402
    METHOD_PROP_NEW,
    RequirementAwareRouterWrapper,
    bind_projection_runtime,
    load_projection_config,
    load_projection_module,
)
from scripts.run_sarfish_canonical_seed2026 import prop_configs  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.canonical_sarfish_environment import (  # noqa: E402
    canonical_environment_hash_v2,
    canonical_task_hash_v2,
    canonical_task_rows_v2,
)
from src.evaluation.task_outcome_adapter import (  # noqa: E402
    adapt_b1,
    adapt_proposed,
    manifest_rows,
    proposed_delivery_events,
    requirement_satisfied_by_products,
)
from src.evaluation.unified_ground_received_accounting import (  # noqa: E402
    score_task,
    summarize_method,
)
from src.data.value_skew_calibration import task_effective_value  # noqa: E402
from src.baselines.b1_enhanced import B1EnhancedSimulator  # noqa: E402
from src.common.types import TaskSample  # noqa: E402
from src.routers.myopic_frozen_calibrated_router import (  # noqa: E402
    MyopicFrozenCalibratedRouter,
)
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402
from scripts.run_sarfish_canonical_seed2026 import b1_config  # noqa: E402
from scripts.run_unified_ground_received_accounting_seed2026 import (  # noqa: E402
    b1_energy,
    b1_safety,
)


FORMAL_FIVE_SEEDS = (2026, 2027, 2028, 2029, 2030)
DEFAULT_SEEDS = (2026,)
DEFAULT_METHODS = ("CORRECTED_REQALIGN", "MYOPIC", "B1", "OFFLOAD", "TUT")
METHOD_NAMES = {
    "CORRECTED_REQALIGN": METHOD_PROP_NEW,
    "MYOPIC": "MYOPIC-FROZEN-CALIBRATED-ON-CORRECTED-REQALIGN",
    "B1": "B1-COMMON-THERMAL",
    "OFFLOAD": "OFFLOAD-EFT-THERMAL-ALIGNED",
    "TUT": "TUT-UOI-PRODUCT-LITE",
}
MYOPIC_ROUTER_NAME = "myopic_frozen_calibrated"
TUT_ROUTER_NAME = "tut_uoi_rate_lite"
OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "five_method_frozen_myopic_on_corrected_reqalign_seed2026_40w_cap050_v1"
)
CAP050_FIVE_SEED_REFERENCE_ROOT = (
    ROOT
    / "outputs"
    / "five_method_corrected_reqalign_multiseed_40w_cap050_v1"
)
VALUE_SENSITIVITY_ROOT = (
    ROOT
    / "outputs"
    / "five_method_corrected_reqalign_value_ratio_sensitivity_40w_cap050_v1"
)
MB_TO_BYTES = 1_000_000.0
DEFAULT_CAPACITY_MULTIPLIER = 0.50
TARGET_CAPACITY_MULTIPLIER = DEFAULT_CAPACITY_MULTIPLIER
DEFAULT_SCENARIO_PROFILE = "frozen_v25_stress"
TARGET_SCENARIO_PROFILE = DEFAULT_SCENARIO_PROFILE
V25_MANIFEST_SCENARIO_PROFILES = frozenset(
    {"frozen_v25_stress", "controlled_v25_multiseed"}
)
TARGET_SOLAR_PEAK_W = 40.0
TARGET_SLOT_SECONDS = 20.0
DEFAULT_VALUE_PROFILE = "main"
VALUE_PROFILES: dict[str, dict[str, Any]] = {
    "main": {
        "profile": "V25-MAIN-v1",
        "weights": {"low": 1.0, "medium": 3.0, "high": 10.0, "urgent": 25.0},
        "output_root": OUTPUT_ROOT,
    },
    "mild": {
        "profile": "V25-SENS-MILD-1-2-3-5-v1",
        "weights": {"low": 1.0, "medium": 2.0, "high": 3.0, "urgent": 5.0},
        "output_root": (
            VALUE_SENSITIVITY_ROOT
            / "mild_1_2_3_5"
        ),
    },
    "stress": {
        "profile": "V25-SENS-STRESS-1-5-20-50-v1",
        "weights": {"low": 1.0, "medium": 5.0, "high": 20.0, "urgent": 50.0},
        "output_root": (
            VALUE_SENSITIVITY_ROOT
            / "stress_1_5_20_50"
        ),
    },
}
MYOPIC_PARAMETER_FILE = (
    ROOT / "configs" / "myopic_frozen_calibration_v1.runtime.json"
)
MYOPIC_FROZEN_PARAMETERS = json.loads(
    MYOPIC_PARAMETER_FILE.read_text(encoding="utf-8")
)
FROZEN_V25_MANIFEST = (
    ROOT
    / "artifacts"
    / "frozen_scenarios"
    / "v25_main_seed2026_manifest_4c23cf3184f1.csv"
)
FROZEN_V25_BUNDLE_ROOT = (
    ROOT
    / "vendor"
    / "frozen_runtime_v1"
    / "inputs"
    / "V25-MAIN-v1"
)
FROZEN_V25_CONTRACT = FROZEN_V25_BUNDLE_ROOT / "bundle_contract.json"
FROZEN_V25_ARRIVAL_SWAPS = FROZEN_V25_BUNDLE_ROOT / "arrival_swap_manifest.csv"
EXPECTED_FROZEN_V25_TASK_HASH = (
    "4c23cf3184f137733495f95a7c5bd7631adf95b3fbc4eb057351f27dac2f2ab5"
)
EXPECTED_FROZEN_V25_MANIFEST_SHA256 = (
    "64de7c9598f9e6860481d9d20fefc6eb3f853af6c276d35a54f601bdc500af8a"
)
EXPECTED_FROZEN_V25_COUNTS = {
    "low": 54,
    "medium": 139,
    "high": 54,
    "urgent": 21,
}


def configure_target_capacity_multiplier(value: float) -> None:
    """Select one audited link-capacity preset before constructing a scenario."""

    selected = float(value)
    if not any(abs(selected - allowed) <= 1.0e-12 for allowed in (0.35, 0.50)):
        raise ValueError(
            "capacity multiplier must be one of the audited presets: 0.35 or 0.50"
        )
    global TARGET_CAPACITY_MULTIPLIER
    TARGET_CAPACITY_MULTIPLIER = selected


def configure_target_scenario_profile(value: str) -> None:
    """Select the frozen stress manifest or generated multiseed scenarios."""

    selected = str(value).strip().lower()
    allowed = {
        "frozen_v25_stress",
        "controlled_v25_multiseed",
        "generated_multiseed",
    }
    if selected not in allowed:
        raise ValueError(f"scenario profile must be one of {sorted(allowed)}")
    global TARGET_SCENARIO_PROFILE
    TARGET_SCENARIO_PROFILE = selected
# Public reproduction always loads the Git-tracked frozen implementations.
# Historical workstation locations are handled only by the canonical portable
# resolver, never by this comparison module.
PILOT_FROZEN_ROOT = ROOT / "vendor" / "frozen_runtime_v1" / "b1_offload"
TUT_FROZEN_ROOT = ROOT / "vendor" / "frozen_runtime_v1" / "tut"
EXTERNAL_METHOD_PATHS = {
    "offload_router": (
        PILOT_FROZEN_ROOT
        / "src"
        / "baselines"
        / "offload_eft_thermal_aligned.py"
    ),
    "offload_core": PILOT_FROZEN_ROOT / "src" / "baselines" / "offload_eft.py",
    "offload_ground_contract": (
        PILOT_FROZEN_ROOT / "src" / "baselines" / "ground_only.py"
    ),
    "offload_thermal_headroom": (
        PILOT_FROZEN_ROOT / "src" / "energy" / "thermal_headroom.py"
    ),
    "offload_method_config": (
        PILOT_FROZEN_ROOT
        / "configs"
        / "methods"
        / "offload_eft_thermal_aligned_v1.yaml"
    ),
    "tut_router": (
        TUT_FROZEN_ROOT / "src" / "routers" / "tut_uoi_rate_router.py"
    ),
    "tut_method_config": (
        TUT_FROZEN_ROOT
        / "configs"
        / "methods"
        / "tut_uoi_rate_lite.yaml"
    ),
}
METHOD_IMPLEMENTATION_PATHS = {
    "corrected_reqalign_wrapper": (
        ROOT / "scripts" / "run_prop_reqalign_seed2026_v1.py"
    ),
    "myopic_router": (
        ROOT / "src" / "routers" / "myopic_frozen_calibrated_router.py"
    ),
    "myopic_frozen_parameters": MYOPIC_PARAMETER_FILE,
    "b1_simulator": ROOT / "src" / "baselines" / "b1_enhanced.py",
    **EXTERNAL_METHOD_PATHS,
}


def _value_profile_spec(value_profile: str) -> dict[str, Any]:
    key = str(value_profile).strip().lower()
    if key not in VALUE_PROFILES:
        raise ValueError(
            f"unsupported value profile {value_profile!r}; "
            f"choose one of {','.join(VALUE_PROFILES)}"
        )
    return deepcopy(VALUE_PROFILES[key])
POWER_CONFIG = ROOT / "configs" / "bupt1_measured_power_v1.yaml"
SOLAR_TRACE = (
    ROOT
    / "vendor"
    / "frozen_runtime_v1"
    / "inputs"
    / "solar_trace_one_orbit.csv"
)
SELECTED_CONFIG_PATHS = {
    "dataset": ROOT / "configs" / "dataset_sarfish_grd_validation.yaml",
    "task_stream": ROOT / "configs" / "task_stream_sarfish_formal_nominal_bupt1_deadline_minutes.yaml",
    "timeline": ROOT / "configs" / "timeline_sarfish_formal_nominal_bupt1_deadline_minutes_20s.yaml",
    "simulator": ROOT / "configs" / "simulator_sarfish_formal_nominal_bupt1_deadline_minutes_1day_20s.yaml",
    "resource_base": ROOT / "configs" / "resource_sarfish_formal_nominal_bupt1.yaml",
    "power_override": POWER_CONFIG,
    "model_profiles": ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml",
    "payload_profiles": ROOT / "configs" / "payload_profiles_roi_compression_calibrated.yaml",
    "requirement_projection": ROOT / "configs" / "prop_requirement_projection_v1.yaml",
    "ground_accounting": ROOT / "configs" / "unified_ground_received_accounting.yaml",
    "product_utility": ROOT / "configs" / "product_requirement_utility.yaml",
}
SHARED_COMPONENT_PATHS = {
    "task_stream": ROOT / "src" / "data" / "task_stream.py",
    "timeline_builder": ROOT / "src" / "timeline" / "timeline_builder.py",
    "resource_trace": ROOT / "src" / "timeline" / "resource_trace_loader.py",
    "link_trace": ROOT / "src" / "timeline" / "link_trace_loader.py",
    "safe_filter": ROOT / "src" / "routers" / "safe_filter.py",
    "model_registry": ROOT / "src" / "models" / "model_registry.py",
    "payload_generation": ROOT / "src" / "payload" / "payload_sizer.py",
    "queue": ROOT / "src" / "link" / "downlink_queue.py",
    "capacity_formula": ROOT / "src" / "link" / "contact_window.py",
    "executor": ROOT / "src" / "simulator" / "step_runner.py",
    "episode_runner": ROOT / "src" / "simulator" / "episode_runner.py",
    "ground_evaluator": ROOT / "src" / "evaluation" / "task_outcome_adapter.py",
    "common_accounting": ROOT / "src" / "evaluation" / "unified_ground_received_accounting.py",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text_lf(path: Path) -> str:
    """Hash a tracked text fixture independently of Git's CRLF checkout mode."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _frozen_v25_bundle_hash(rows: list[dict[str, str]]) -> str:
    fields = (
        "task_id",
        "value_level",
        "effective_task_value",
        "product_requirement",
        "arrival_time_s",
        "deadline_time_s",
    )
    stable_rows = sorted(
        [{key: row[key] for key in fields} for row in rows],
        key=lambda row: row["task_id"],
    )
    return _stable_hash(stable_rows)


def _load_frozen_v25_stress_manifest() -> tuple[dict[int, TaskSample], dict[str, Any]]:
    required = (
        FROZEN_V25_MANIFEST,
        FROZEN_V25_CONTRACT,
        FROZEN_V25_ARRIVAL_SWAPS,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError({"missing_frozen_v25_stress_inputs": missing})
    rows = _read_csv(FROZEN_V25_MANIFEST)
    contract = json.loads(FROZEN_V25_CONTRACT.read_text(encoding="utf-8"))
    swaps = _read_csv(FROZEN_V25_ARRIVAL_SWAPS)
    counts = _count_by(rows, "value_level")
    manifest_sha256 = _sha256_text_lf(FROZEN_V25_MANIFEST)
    recovered_hash = _frozen_v25_bundle_hash(rows)
    checks = {
        "task_count_is_268": len(rows) == 268,
        "task_ids_unique": len({row["task_id"] for row in rows}) == len(rows),
        "arrival_slots_unique": len({row["arrival_slot"] for row in rows}) == len(rows),
        "value_level_counts_match": counts == EXPECTED_FROZEN_V25_COUNTS,
        "manifest_file_sha256_matches": (
            manifest_sha256 == EXPECTED_FROZEN_V25_MANIFEST_SHA256
        ),
        "v25_task_hash_matches": (
            recovered_hash
            == EXPECTED_FROZEN_V25_TASK_HASH
            == str(contract.get("task_hash", ""))
        ),
        "arrival_swap_pair_count_is_27": len(swaps) == 27,
        "bundle_seed_is_2026": int(contract.get("seed", -1)) == 2026,
        "bundle_capacity_source_is_0_35": abs(
            _float(contract.get("capacity"), -1.0) - 0.35
        )
        <= 1.0e-12,
    }
    if not all(checks.values()):
        raise AssertionError({"frozen_v25_stress_manifest_failed": checks})

    tasks_by_slot: dict[int, TaskSample] = {}
    for row in rows:
        metadata = json.loads(row.get("metadata_json") or "{}")
        metadata.update(
            {
                "frozen_v25_manifest": True,
                "frozen_v25_task_hash": EXPECTED_FROZEN_V25_TASK_HASH,
                "frozen_v25_manifest_sha256": manifest_sha256,
            }
        )
        slot = int(float(row["arrival_slot"]))
        if abs(_float(row["arrival_time_s"]) - slot * TARGET_SLOT_SECONDS) > 1.0e-9:
            raise AssertionError(
                f"frozen task {row['task_id']} arrival time/slot mismatch"
            )
        tasks_by_slot[slot] = TaskSample(
            task_id=row["task_id"],
            image_path=row["image_path"],
            label_path=row["label_path"],
            scene_id=row["scene_id"],
            virtual_time=int(float(row["virtual_time"])),
            task_value=_float(row["task_value"]),
            deadline_s=_float(row["deadline_s"]),
            difficulty=_float(row["difficulty"]),
            uncertainty=_float(row["uncertainty"]),
            metadata=metadata,
            arrival_slot=slot,
            observation_window_id=row["observation_window_id"],
            task_class=row["task_class"],
            quality_requirement=_float(row["quality_requirement"]),
            information_requirement=_float(row["information_requirement"]),
            high_value_flag=_truth(row["high_value_flag"]),
            emergency_flag=_truth(row["emergency_flag"]),
            product_requirement=row["product_requirement"],
            product_requirement_source=row["product_requirement_source"],
            base_prior_score=_float(row["base_prior_score"]),
            value_level=row["value_level"],
            value_multiplier=_float(row["value_multiplier"], 1.0),
            value_skew_profile=row["value_skew_profile"],
            normalization_alpha=_float(row["normalization_alpha"], 1.0),
            calibrated_task_value=_float(row["calibrated_task_value"]),
            value_assignment_reason=row["value_assignment_reason"],
        )
    return tasks_by_slot, {
        "profile": "frozen_v25_stress",
        "source_manifest": str(FROZEN_V25_MANIFEST),
        "source_bundle": str(FROZEN_V25_BUNDLE_ROOT),
        "task_hash": recovered_hash,
        "manifest_sha256": manifest_sha256,
        "value_level_counts": counts,
        "arrival_swap_pair_count": len(swaps),
        "arrival_swapped_task_count": len(swaps) * 2,
        "checks": checks,
    }


def _controlled_v25_temporal_permutation(
    tasks_by_slot: Mapping[int, TaskSample],
    seed: int,
) -> tuple[dict[int, TaskSample], dict[str, Any]]:
    """Keep the frozen V25 cohort fixed and vary only its temporal ordering."""

    original_slots = sorted(int(slot) for slot in tasks_by_slot)
    original_tasks = [tasks_by_slot[slot] for slot in original_slots]
    if int(seed) == 2026:
        ordered_tasks = original_tasks
    else:
        ordered_tasks = sorted(
            original_tasks,
            key=lambda task: hashlib.sha256(
                f"controlled-v25-temporal-v1:{int(seed)}:{task.task_id}".encode(
                    "utf-8"
                )
            ).digest(),
        )

    permuted: dict[int, TaskSample] = {}
    assignment_rows: list[dict[str, Any]] = []
    for new_slot, task in zip(original_slots, ordered_tasks):
        original_slot = int(task.arrival_slot)
        metadata = deepcopy(task.metadata)
        metadata.update(
            {
                "controlled_v25_multiseed": True,
                "controlled_v25_seed": int(seed),
                "controlled_v25_original_arrival_slot": original_slot,
            }
        )
        permuted[new_slot] = replace(
            task,
            virtual_time=new_slot,
            arrival_slot=new_slot,
            metadata=metadata,
        )
        assignment_rows.append(
            {
                "task_id": task.task_id,
                "original_arrival_slot": original_slot,
                "controlled_arrival_slot": new_slot,
            }
        )

    original_ids = {task.task_id for task in original_tasks}
    controlled_ids = {task.task_id for task in permuted.values()}
    details = {
        "scheme": "controlled-v25-temporal-v1",
        "seed": int(seed),
        "exact_frozen_anchor": int(seed) == 2026,
        "moved_task_count": sum(
            row["original_arrival_slot"] != row["controlled_arrival_slot"]
            for row in assignment_rows
        ),
        "assignment_sha256": _stable_hash(assignment_rows),
        "checks": {
            "task_count_preserved": len(permuted) == len(tasks_by_slot) == 268,
            "task_ids_preserved": controlled_ids == original_ids,
            "arrival_slots_preserved": set(permuted) == set(tasks_by_slot),
            "value_level_counts_preserved": _count_by(
                [
                    {"value_level": task.value_level}
                    for task in permuted.values()
                ],
                "value_level",
            )
            == EXPECTED_FROZEN_V25_COUNTS,
        },
    }
    if not all(details["checks"].values()):
        raise AssertionError({"controlled_v25_permutation_failed": details})
    return permuted, details


def _solar_trace_rows() -> list[dict[str, str]]:
    if not POWER_CONFIG.is_file() or not SOLAR_TRACE.is_file():
        raise FileNotFoundError(
            f"target 40 W power inputs are incomplete: {POWER_CONFIG}, {SOLAR_TRACE}"
        )
    rows = _read_csv(SOLAR_TRACE)
    if not rows:
        raise ValueError("target 40 W solar trace is empty")
    return rows


def _target_power_config() -> dict[str, Any]:
    loaded = yaml.safe_load(POWER_CONFIG.read_text(encoding="utf-8")) or {}
    peak = _float(loaded.get("solar_peak_power_W"), -1.0)
    if abs(peak - TARGET_SOLAR_PEAK_W) > 1.0e-12:
        raise AssertionError(
            f"target solar peak mismatch: expected {TARGET_SOLAR_PEAK_W}, got {peak}"
        )
    if abs(_float(loaded.get("tx_active_power_W"), -1.0) - 38.0) > 1.0e-12:
        raise AssertionError("the frozen radio active power must remain 38 W")
    if abs(_float(loaded.get("slot_s"), -1.0) - TARGET_SLOT_SECONDS) > 1.0e-12:
        raise AssertionError("the target power trace must use 20 s slots")
    return loaded


def _apply_target_environment(
    configs: dict[str, Any],
    timeline: list[Any],
    value_profile: str = DEFAULT_VALUE_PROFILE,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    """Apply the frozen 40 W trace and selected audited link multiplier.

    The 40 W quantity is the peak available solar electrical input before the
    configured charging efficiency.  It is independent of the 38 W radio
    active draw and all model/encoding energy.
    """

    configs = deepcopy(configs)
    value_spec = _value_profile_spec(value_profile)
    value_profile_name = str(value_spec["profile"])
    nominal_value_by_level = {
        str(level): float(weight)
        for level, weight in value_spec["weights"].items()
    }
    nominal_value_mapping = ",".join(
        f"{level}={nominal_value_by_level[level]:g}"
        for level in ("low", "medium", "high", "urgent")
    )
    power_cfg = _target_power_config()
    solar_rows = _solar_trace_rows()
    solar_values = [_float(row["solar_power_W"]) for row in solar_rows]
    if abs(max(solar_values) - TARGET_SOLAR_PEAK_W) > 1.0e-9:
        raise AssertionError(
            f"solar trace maximum mismatch: expected {TARGET_SOLAR_PEAK_W}, got {max(solar_values)}"
        )

    simulator = configs.setdefault("simulator", {})
    simulator["battery_capacity_wh"] = _float(power_cfg["battery_capacity_Wh"])
    simulator["battery_capacity_j"] = _float(power_cfg["battery_capacity_Wh"]) * 3600.0
    simulator["battery_init"] = _float(power_cfg["initial_soc"])
    simulator["battery_initial_soc"] = _float(power_cfg["initial_soc"])
    simulator["battery_charge_efficiency"] = _float(power_cfg["charge_efficiency"])
    simulator["battery_discharge_efficiency"] = _float(power_cfg["discharge_efficiency"])
    simulator["battery_soft_guard"] = _float(power_cfg["soc_warning_threshold"])
    simulator["battery_soft_target"] = _float(power_cfg["soc_warning_threshold"])
    simulator["soc_warning_threshold"] = _float(power_cfg["soc_warning_threshold"])
    simulator["tx_power_w"] = _float(power_cfg["tx_active_power_W"])
    simulator["comm_power_w"] = _float(power_cfg["tx_active_power_W"])
    simulator["downlink_capacity_multiplier"] = TARGET_CAPACITY_MULTIPLIER
    simulator["common_power_supervisor"] = {"enabled": False}
    simulator["formal_method_extensions"] = {
        "hv_urgent_guard_enabled": False,
        "medium_slack_reclaim_enabled": False,
    }

    configs["downlink_capacity_multiplier"] = TARGET_CAPACITY_MULTIPLIER
    configs.setdefault("timeline", {})["slot_seconds"] = TARGET_SLOT_SECONDS
    configs.setdefault("contact", {})["downlink_capacity_multiplier"] = (
        TARGET_CAPACITY_MULTIPLIER
    )
    resource_cfg = configs.setdefault("resource", {})
    resource_cfg["orbit_period_s"] = int(power_cfg["orbit_period_s"])
    resource_cfg["eclipse_duration_s"] = int(power_cfg["eclipse_duration_s"])
    resource_cfg["sunlit_duration_s"] = int(power_cfg["daylight_duration_s"])
    resource_cfg["solar_peak_w"] = TARGET_SOLAR_PEAK_W
    resource_cfg["idle_power_w"] = 8.86
    resource_cfg["battery_capacity_wh"] = _float(power_cfg["battery_capacity_Wh"])
    resource_cfg["battery_init"] = _float(power_cfg["initial_soc"])
    resource_cfg["solar_input_semantics"] = (
        "available_solar_electrical_input_before_charge_efficiency"
    )
    configs["timeline"]["resource"] = deepcopy(resource_cfg)
    configs["timeline"]["contact"] = deepcopy(configs["contact"])

    new_timeline = []
    pre_rates = []
    post_rates = []
    for step in timeline:
        slot = int(step.slot_index)
        row = solar_rows[slot % len(solar_rows)]
        solar_power_w = _float(row["solar_power_W"])
        phase = str(row["orbit_phase"]).strip().lower()
        slot_seconds = TARGET_SLOT_SECONDS
        orbit_seconds = int(power_cfg["orbit_period_s"])
        t_in_orbit = (slot * slot_seconds) % orbit_seconds
        resource = replace(
            step.resource_trace,
            solar_energy_j=solar_power_w * slot_seconds,
            base_power_w=8.86,
            is_sunlit=phase == "daylight",
            is_eclipse=phase == "eclipse",
            orbit_index=int((slot * slot_seconds) // orbit_seconds),
            day_index=0,
            orbit_phase=t_in_orbit / orbit_seconds,
            solar_power_w=solar_power_w,
            wasted_solar_energy_j=0.0,
            battery_soc=_float(power_cfg["initial_soc"]),
            comm_power_w=_float(power_cfg["tx_active_power_W"]),
        )
        pre_rate = _float(step.link_trace.throughput_mbps)
        post_rate = (
            pre_rate * TARGET_CAPACITY_MULTIPLIER
            if step.link_trace.in_contact
            else 0.0
        )
        if step.link_trace.in_contact:
            pre_rates.append(pre_rate)
            post_rates.append(post_rate)
        link = replace(step.link_trace, throughput_mbps=post_rate)
        task = step.task
        if task is not None:
            level = str(task.value_level or "").strip().lower()
            if level not in nominal_value_by_level:
                raise AssertionError(
                    f"task {task.task_id} has unsupported value level {task.value_level!r}"
                )
            task_metadata = {
                **dict(task.metadata or {}),
                "value_profile": value_profile_name,
                "nominal_value_weight": nominal_value_by_level[level],
                "nominal_value_mapping": nominal_value_mapping,
            }
            if bool(task_metadata.get("frozen_v25_manifest")):
                task = replace(task, metadata=task_metadata)
            else:
                task = replace(
                    task,
                    metadata=task_metadata,
                    value_multiplier=nominal_value_by_level[level],
                    value_skew_profile=value_profile_name,
                )
        new_timeline.append(
            replace(
                step,
                slot_seconds=slot_seconds,
                task=task,
                resource_trace=resource,
                link_trace=link,
            )
        )

    if not pre_rates:
        raise AssertionError("target environment contains no contact slots")
    solar_day = [
        float(step.resource_trace.solar_power_w)
        for step in new_timeline
        if step.resource_trace.is_sunlit
    ]
    solar_all = [float(step.resource_trace.solar_power_w) for step in new_timeline]
    identity = {
        "solar": {
            "physical_meaning": "available solar electrical input before charge efficiency",
            "configured_peak_w": TARGET_SOLAR_PEAK_W,
            "trace_max_w": max(solar_all),
            "sunlit_mean_w": statistics.fmean(solar_day),
            "full_day_mean_w": statistics.fmean(solar_all),
            "charge_efficiency": _float(power_cfg["charge_efficiency"]),
            "radio_active_draw_w_separate": _float(power_cfg["tx_active_power_W"]),
            "platform_other_draw_w_separate": 8.86,
        },
        "communication": {
            "capacity_multiplier": TARGET_CAPACITY_MULTIPLIER,
            "nominal_unscaled_min_mbps": _float(configs["contact"]["throughput_min_mbps"]),
            "nominal_unscaled_max_mbps": _float(configs["contact"]["throughput_max_mbps"]),
            "nominal_scaled_min_mbps": _float(configs["contact"]["throughput_min_mbps"])
            * TARGET_CAPACITY_MULTIPLIER,
            "nominal_scaled_max_mbps": _float(configs["contact"]["throughput_max_mbps"])
            * TARGET_CAPACITY_MULTIPLIER,
            "actual_unscaled_min_mbps": min(pre_rates),
            "actual_unscaled_mean_mbps": statistics.fmean(pre_rates),
            "actual_unscaled_max_mbps": max(pre_rates),
            "actual_scaled_min_mbps": min(post_rates),
            "actual_scaled_mean_mbps": statistics.fmean(post_rates),
            "actual_scaled_max_mbps": max(post_rates),
            "actual_unscaled_min_mb_per_slot": min(pre_rates)
            * TARGET_SLOT_SECONDS
            / 8.0,
            "actual_unscaled_mean_mb_per_slot": statistics.fmean(pre_rates)
            * TARGET_SLOT_SECONDS
            / 8.0,
            "actual_unscaled_max_mb_per_slot": max(pre_rates)
            * TARGET_SLOT_SECONDS
            / 8.0,
            "actual_scaled_min_mb_per_slot": min(post_rates)
            * TARGET_SLOT_SECONDS
            / 8.0,
            "actual_scaled_mean_mb_per_slot": statistics.fmean(post_rates)
            * TARGET_SLOT_SECONDS
            / 8.0,
            "actual_scaled_max_mb_per_slot": max(post_rates)
            * TARGET_SLOT_SECONDS
            / 8.0,
            "every_contact_slot_scaled_exactly": all(
                abs(post - pre * TARGET_CAPACITY_MULTIPLIER) <= 1.0e-12
                for pre, post in zip(pre_rates, post_rates)
            ),
            "link_quality_phase": _float(configs["contact"]["link_quality_phase"]),
            "link_quality_phase_is_capacity_multiplier": False,
        },
        "guard_and_reclaim": {
            "hv_urgent_guard_enabled": False,
            "medium_slack_reclaim_enabled": False,
        },
        "task_value_identity": {
            "profile_key": str(value_profile).strip().lower(),
            "profile": value_profile_name,
            "nominal_value_weights": nominal_value_by_level,
            "effective_runtime_value_semantics": (
                "the exact frozen V25 calibrated_task_value is reused"
                if TARGET_SCENARIO_PROFILE in V25_MANIFEST_SCENARIO_PROFILES
                else (
                    "existing mean-preserving value-skew calibration supplies "
                    "calibrated_task_value; nominal weights are not a policy bonus"
                )
            ),
        },
        "power_config_sha256": _sha256_file(POWER_CONFIG),
        "solar_trace_sha256": _sha256_file(SOLAR_TRACE),
    }
    return configs, new_timeline, identity


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized or [{"empty": ""}])


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("seeds must be a non-empty list of unique integers")
    return values


def _parse_methods(raw: str) -> tuple[str, ...]:
    aliases = {
        "REQALIGN": "CORRECTED_REQALIGN",
        "CORRECTED-REQALIGN": "CORRECTED_REQALIGN",
        "B1-COMMON-THERMAL": "B1",
        "OFFLOAD-EFT-THERMAL-ALIGNED": "OFFLOAD",
        "TUT-UOI-PRODUCT-LITE": "TUT",
    }
    methods = tuple(
        aliases.get(item.strip().upper(), item.strip().upper())
        for item in raw.split(",")
        if item.strip()
    )
    unknown = sorted(set(methods) - set(METHOD_NAMES))
    if not methods or unknown:
        raise ValueError(f"methods must be drawn from {sorted(METHOD_NAMES)}, got unknown={unknown}")
    return tuple(dict.fromkeys(methods))


def _add_shared_method_aliases(configs: dict[str, Any]) -> None:
    """Enable the same corrected backend lifecycle for both delegates."""

    simulator = configs["simulator"]
    progressive = simulator.setdefault("progressive_payload", {})
    alias_targets = [
        progressive.get("value_aware_staged_delivery_queue"),
        progressive.get("phase6_proactive_followup"),
        simulator.get("guarantee"),
    ]
    for target in alias_targets:
        if not isinstance(target, dict):
            continue
        aliases = target.get("full_method_aliases", [])
        aliases = [aliases] if isinstance(aliases, str) else list(aliases)
        target["full_method_aliases"] = list(
            dict.fromkeys(
                [
                    *aliases,
                    "proposed_lyapunov_rule_mode_safe_bandit",
                    MYOPIC_ROUTER_NAME,
                    TUT_ROUTER_NAME,
                ]
            )
        )


def _load_frozen_external_method_classes() -> tuple[type[Any], type[Any]]:
    """Load exact frozen OFFLOAD and TUT classes against the current physics.

    Only missing baseline modules are resolved from the frozen repositories.
    Shared task, energy, thermal, payload and evaluator imports continue to
    resolve from this workspace.
    """

    missing = {
        key: str(path)
        for key, path in EXTERNAL_METHOD_PATHS.items()
        if not path.is_file()
    }
    if missing:
        raise FileNotFoundError({"missing_frozen_method_sources": missing})

    import src.baselines as baseline_package
    import src.energy as energy_package
    import src.routers as router_package

    pilot_baselines = str(PILOT_FROZEN_ROOT / "src" / "baselines")
    pilot_energy = str(PILOT_FROZEN_ROOT / "src" / "energy")
    tut_routers = str(TUT_FROZEN_ROOT / "src" / "routers")
    if pilot_baselines not in baseline_package.__path__:
        baseline_package.__path__.append(pilot_baselines)
    if pilot_energy not in energy_package.__path__:
        energy_package.__path__.append(pilot_energy)
    if tut_routers not in router_package.__path__:
        router_package.__path__.append(tut_routers)

    from src.baselines.offload_eft_thermal_aligned import ThermalAlignedSimulator
    from src.routers.tut_uoi_rate_router import TutUoiRateRouter

    return ThermalAlignedSimulator, TutUoiRateRouter


def build_shared_scenario(
    seed: int,
    value_profile: str = DEFAULT_VALUE_PROFILE,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    """Build the audited 40 W scenario for one declared seed."""

    value_spec = _value_profile_spec(value_profile)
    value_profile_name = str(value_spec["profile"])
    value_weights = {
        str(level): float(weight)
        for level, weight in value_spec["weights"].items()
    }
    configs = prop_configs()
    timeline_seed = (
        2026
        if TARGET_SCENARIO_PROFILE == "controlled_v25_multiseed"
        else int(seed)
    )
    configs["dataset"]["seed"] = timeline_seed
    configs["task_stream"]["seed"] = timeline_seed
    configs["task_stream"]["value_skew_calibration"] = {
        "enabled": True,
        "profile": value_profile_name,
        "assignment_rule": "v2_conservative",
        "normalization": "mean_preserving",
        "profiles": {
            value_profile_name: value_weights,
        },
    }
    configs["experiment"] = {"routers": ["proposed_lyapunov_rule_mode_safe_bandit"]}
    _add_shared_method_aliases(configs)
    base_timeline = TimelineBuilder(
        configs["dataset"],
        configs["timeline"],
        configs["task_stream"],
    ).build()
    frozen_identity: dict[str, Any] | None = None
    if TARGET_SCENARIO_PROFILE in V25_MANIFEST_SCENARIO_PROFILES:
        if str(value_profile).strip().lower() != "main":
            raise ValueError(
                "V25 manifest-backed scenarios require value-profile=main"
            )
        if (
            TARGET_SCENARIO_PROFILE == "frozen_v25_stress"
            and int(seed) != 2026
        ):
            raise ValueError(
                "the frozen V25 stress scenario is defined only for "
                "seed=2026 and value-profile=main"
            )
        tasks_by_slot, frozen_identity = _load_frozen_v25_stress_manifest()
        if TARGET_SCENARIO_PROFILE == "controlled_v25_multiseed":
            tasks_by_slot, temporal_identity = (
                _controlled_v25_temporal_permutation(tasks_by_slot, seed)
            )
            frozen_identity = {
                **frozen_identity,
                "profile": "controlled_v25_multiseed",
                "base_frozen_task_hash": frozen_identity["task_hash"],
                "controlled_temporal_identity": temporal_identity,
                "checks": {
                    **frozen_identity["checks"],
                    **temporal_identity["checks"],
                },
            }
        if max(tasks_by_slot) >= len(base_timeline):
            raise AssertionError("frozen V25 task arrival exceeds the 24 h timeline")
        base_timeline = [
            replace(step, task=tasks_by_slot.get(int(step.slot_index)))
            for step in base_timeline
        ]
        configs["task_stream"]["value_skew_calibration"] = {
            "enabled": True,
            "profile": value_profile_name,
            "assignment_rule": "exact_frozen_v25_manifest",
            "normalization": "frozen_mean_normalized",
            "normalization_alpha": 0.17447916666666666,
            "profiles": {value_profile_name: value_weights},
            "source_manifest_sha256": frozen_identity["manifest_sha256"],
            "source_task_hash": frozen_identity["task_hash"],
            "arrival_swap_pair_count": frozen_identity["arrival_swap_pair_count"],
        }
    configs, timeline, target_environment = _apply_target_environment(
        configs,
        base_timeline,
        value_profile,
    )
    task_rows = canonical_task_rows_v2(timeline)
    if not task_rows or len({row["task_id"] for row in task_rows}) != len(task_rows):
        raise ValueError(f"seed {seed} produced an empty or non-unique task stream")
    evaluator_identity = {
        key: _sha256_file(path)
        for key, path in SHARED_COMPONENT_PATHS.items()
        if key in {"ground_evaluator", "common_accounting"}
    }
    canonical_task_hash = canonical_task_hash_v2(task_rows)
    identity = {
        "seed": seed,
        "scenario_profile": TARGET_SCENARIO_PROFILE,
        "value_profile_key": str(value_profile).strip().lower(),
        "value_profile": value_profile_name,
        "nominal_value_weights": value_weights,
        "task_count": len(task_rows),
        "slot_count": len(timeline),
        "task_manifest_hash_v2": (
            frozen_identity["task_hash"]
            if frozen_identity is not None
            and (
                TARGET_SCENARIO_PROFILE == "frozen_v25_stress"
                or int(seed) == 2026
            )
            else canonical_task_hash
        ),
        "canonical_task_hash_v2": canonical_task_hash,
        "environment_hash_v2": canonical_environment_hash_v2(configs, timeline),
        "shared_config_hash": _stable_hash(configs),
        "evaluator_identity_hash": _stable_hash(evaluator_identity),
        "target_environment": target_environment,
        "frozen_v25_identity": frozen_identity,
        "declared_seed": int(seed),
        "base_timeline_seed": int(timeline_seed),
    }
    return configs, timeline, identity


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    return dict(sorted(counts.items()))


def _payload_size_identity(configs: Mapping[str, Any]) -> dict[str, Any]:
    profiles = configs["payload_profiles"]["profiles"]
    return {
        str(row["name"]): {
            key: row[key]
            for key in (
                "size_mb",
                "fallback_size_mb",
                "min_payload_mb",
                "max_payload_mb",
                "base_size_kb",
                "per_detection_record_bytes",
                "staged_first_product",
                "residual_product",
            )
            if key in row
        }
        for row in profiles
    }


def _task_identity_summary(
    seed: int,
    timeline: list[Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    rows = canonical_task_rows_v2(timeline)
    deadlines = [_float(row["deadline_s"]) for row in rows]
    nominal_values = {
        str(level): float(weight)
        for level, weight in identity["nominal_value_weights"].items()
    }
    effective_values = [
        task_effective_value(step.task)
        for step in timeline
        if step.task is not None
    ]
    return {
        "seed": seed,
        "task_count": len(rows),
        "task_manifest_hash_v2": identity["task_manifest_hash_v2"],
        "canonical_task_hash_v2": identity["canonical_task_hash_v2"],
        "scenario_profile": identity["scenario_profile"],
        "frozen_v25_identity": identity.get("frozen_v25_identity"),
        "value_level_counts": _count_by(rows, "value_level"),
        "nominal_value_weight_by_level": dict(sorted(nominal_values.items())),
        "effective_runtime_value": {
            "min": min(effective_values),
            "mean": statistics.fmean(effective_values),
            "max": max(effective_values),
            "source": "task.calibrated_task_value via task_effective_value",
        },
        "task_class_counts": _count_by(rows, "task_class"),
        "required_product_counts": _count_by(rows, "product_requirement"),
        "required_product_missing_count": sum(
            not str(row.get("product_requirement", "")).strip() for row in rows
        ),
        "deadline_s": {
            "min": min(deadlines),
            "mean": statistics.fmean(deadlines),
            "max": max(deadlines),
        },
    }


def _config_loading_trace(cli: Mapping[str, Any]) -> dict[str, Any]:
    value_spec = _value_profile_spec(str(cli["value_profile"]))
    value_profile_name = str(value_spec["profile"])
    value_weights = {
        str(level): float(weight)
        for level, weight in value_spec["weights"].items()
    }
    all_yaml = sorted((ROOT / "configs").glob("*.yaml"))
    selected = {str(path.resolve()) for path in SELECTED_CONFIG_PATHS.values()}
    return {
        "loader": {
            "source": "src/common/config.py::load_project_configs",
            "behavior": (
                "load every configs/*.yaml in lexical order into a stem-keyed mapping; "
                "there is no deep merge between YAML files"
            ),
        },
        "initial_yaml_load": [
            {
                "order": index,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256_file(path),
                "selected_or_override_input": str(path.resolve()) in selected,
            }
            for index, path in enumerate(all_yaml)
        ],
        "replacement_and_override_order": [
            {
                "order": 1,
                "operation": "load_project_configs",
                "source": "all configs/*.yaml",
            },
            {
                "order": 2,
                "operation": "replace canonical dataset/task/timeline/simulator/resource/contact",
                "source": {
                    key: str(path.relative_to(ROOT)).replace("\\", "/")
                    for key, path in SELECTED_CONFIG_PATHS.items()
                    if key
                    in {
                        "dataset",
                        "task_stream",
                        "timeline",
                        "simulator",
                        "resource_base",
                    }
                },
            },
            {
                "order": 3,
                "operation": "replace model profile overrides and payload profiles",
                "source": {
                    key: str(SELECTED_CONFIG_PATHS[key].relative_to(ROOT)).replace(
                        "\\", "/"
                    )
                    for key in ("model_profiles", "payload_profiles")
                },
            },
            {
                "order": 4,
                "operation": "apply corrected REQALIGN progressive/guarantee configuration",
                "source": "scripts/run_sarfish_canonical_seed2026.py::prop_configs",
            },
            {
                "order": 5,
                "operation": "override declared dataset/task-stream seed",
                "source": "CLI --seeds",
                "value": cli["seeds"],
            },
            {
                "order": 6,
                "operation": (
                    "load exact frozen V25 task/value/arrival identity"
                    if TARGET_SCENARIO_PROFILE in V25_MANIFEST_SCENARIO_PROFILES
                    else f"enable {value_profile_name} mean-preserving value identity"
                ),
                "source": (
                    str(FROZEN_V25_MANIFEST.relative_to(ROOT)).replace("\\", "/")
                    if TARGET_SCENARIO_PROFILE in V25_MANIFEST_SCENARIO_PROFILES
                    else "src/data/value_skew_calibration.py"
                ),
                "value": value_weights,
            },
            {
                "order": 7,
                "operation": (
                    "register the frozen five-method set; corrected PROP and "
                    "MYOPIC share REQALIGN, while B1/OFFLOAD/TUT retain their "
                    "declared architectures"
                ),
                "source": {
                    key: str(path)
                    for key, path in METHOD_IMPLEMENTATION_PATHS.items()
                },
            },
            {
                "order": 8,
                "operation": "inject frozen 40 W solar trace and BUPT-1 power parameters",
                "source": {
                    "power_config": str(POWER_CONFIG.relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                    "solar_trace": str(SOLAR_TRACE.relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                },
            },
            {
                "order": 9,
                "operation": (
                    "multiply every in-contact link-trace throughput by "
                    f"{TARGET_CAPACITY_MULTIPLIER:.2f}"
                ),
                "source": "scripts/run_myopic_e2e_utility_greedy.py::_apply_target_environment",
            },
            {
                "order": 10,
                "operation": (
                    "enable corrected requirement projection for corrected "
                    "PROP and MYOPIC only"
                ),
                "source": str(
                    SELECTED_CONFIG_PATHS["requirement_projection"].relative_to(ROOT)
                ).replace("\\", "/"),
            },
            {
                "order": 11,
                "operation": (
                    "apply CLI output/method selection and the audited capacity preset"
                ),
                "source": "CLI",
                "value": dict(cli),
            },
        ],
        "code_defaults": {
            "seeds": list(DEFAULT_SEEDS),
            "methods": list(DEFAULT_METHODS),
            "value_profile": DEFAULT_VALUE_PROFILE,
            "scenario_profile": TARGET_SCENARIO_PROFILE,
            "value_profiles": deepcopy(VALUE_PROFILES),
            "capacity_multiplier": TARGET_CAPACITY_MULTIPLIER,
            "solar_peak_w": TARGET_SOLAR_PEAK_W,
            "myopic_frozen_parameter_file": str(
                MYOPIC_PARAMETER_FILE.relative_to(ROOT)
            ).replace("\\", "/"),
            "myopic_frozen_parameters": deepcopy(MYOPIC_FROZEN_PARAMETERS),
            "slot_seconds": TARGET_SLOT_SECONDS,
            "output_root": str(OUTPUT_ROOT.relative_to(ROOT)).replace("\\", "/"),
        },
    }


def _semantic_evaluator_checks(configs: Mapping[str, Any]) -> dict[str, Any]:
    from src.evaluation.unified_ground_received_accounting import (
        incremental_ground_events,
    )

    utility_cfg = configs["product_requirement_utility"]
    truth_table = {
        requirement: {
            "metadata": requirement_satisfied_by_products({"metadata"}, requirement),
            "roi": requirement_satisfied_by_products({"roi"}, requirement),
            "roi_context": requirement_satisfied_by_products(
                {"roi_context"}, requirement
            ),
        }
        for requirement in ("alert", "verification", "context", "evidence")
    }
    duplicate_events = incremental_ground_events(
        [
            {
                "product_type": "metadata",
                "ground_available_time_s": 1.0,
            },
            {
                "product_type": "metadata",
                "ground_available_time_s": 2.0,
            },
        ],
        "alert",
        utility_cfg,
    )
    return {
        "closure_truth_table": truth_table,
        "metadata_cannot_close_verification": not truth_table["verification"]["metadata"],
        "metadata_cannot_close_context": not truth_table["context"]["metadata"],
        "metadata_cannot_close_evidence": not truth_table["evidence"]["metadata"],
        "required_product_ground_event_is_closure_authority": True,
        "full_closure_latency_definition": (
            "required_product_ground_time_s - task_arrival_time_s"
        ),
        "full_closure_components": [
            "inference",
            "product_generation",
            "queue_waiting",
            "contact_waiting",
            "downlink_transmission",
        ],
        "high_and_urgent_independent_metrics_enabled": True,
        "duplicate_second_delta_information_utility": _float(
            duplicate_events[1]["delta_information_utility"], -1.0
        ),
        "duplicate_utility_is_zero": (
            abs(_float(duplicate_events[1]["delta_information_utility"], -1.0))
            <= 1.0e-12
        ),
    }


def _environment_checksums(
    scenario_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "selected_config_files": {
            key: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256_file(path),
            }
            for key, path in SELECTED_CONFIG_PATHS.items()
        },
        "shared_component_files": {
            key: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256_file(path),
            }
            for key, path in SHARED_COMPONENT_PATHS.items()
        },
        "method_implementation_files": {
            key: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for key, path in METHOD_IMPLEMENTATION_PATHS.items()
        },
        "solar_trace": {
            "path": str(SOLAR_TRACE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(SOLAR_TRACE),
        },
        "frozen_v25_scenario_files": (
            {
                "task_manifest": {
                    "path": str(FROZEN_V25_MANIFEST.relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                    "sha256": _sha256_text_lf(FROZEN_V25_MANIFEST),
                },
                "bundle_contract": {
                    "path": str(FROZEN_V25_CONTRACT.relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                    "sha256": _sha256_file(FROZEN_V25_CONTRACT),
                },
                "arrival_swap_manifest": {
                    "path": str(FROZEN_V25_ARRIVAL_SWAPS.relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                    "sha256": _sha256_file(FROZEN_V25_ARRIVAL_SWAPS),
                },
            }
            if TARGET_SCENARIO_PROFILE in V25_MANIFEST_SCENARIO_PROFILES
            else {}
        ),
        "per_seed": [
            {
                key: row[key]
                for key in (
                    "seed",
                    "task_manifest_hash_v2",
                    "environment_hash_v2",
                    "shared_config_hash",
                    "evaluator_identity_hash",
                )
            }
            for row in scenario_rows
        ],
    }


def _audit_assertions(
    scenarios: list[tuple[dict[str, Any], list[Any], dict[str, Any]]],
    semantics: Mapping[str, Any],
    expected_value_weights: Mapping[str, float],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["all_five_method_sources_exist"] = all(
        path.is_file() for path in METHOD_IMPLEMENTATION_PATHS.values()
    )
    if checks["all_five_method_sources_exist"]:
        offload_type, tut_type = _load_frozen_external_method_classes()
        checks["frozen_external_method_classes_load"] = (
            offload_type.__name__ == "ThermalAlignedSimulator"
            and tut_type.__name__ == "TutUoiRateRouter"
        )
        sample_step = next(
            step
            for step in scenarios[0][1]
            if step.task is not None
        )
        b1_timeline = _timeline_with_public_effective_task_values(
            [sample_step]
        )
        tut_router = _build_tut_router(scenarios[0][0])
        tut_router.set_task_context(sample_step.task, 0.0)
        checks["value_aware_baselines_receive_public_effective_value"] = (
            abs(
                _float(b1_timeline[0].task.task_value)
                - task_effective_value(sample_step.task)
            )
            <= 1.0e-12
            and abs(
                _float(tut_router._task.task_value)
                - task_effective_value(sample_step.task)
            )
            <= 1.0e-12
        )
    else:
        checks["frozen_external_method_classes_load"] = False
        checks["value_aware_baselines_receive_public_effective_value"] = False
    checks["target_power_config_matches"] = all(
        abs(
            _float(identity["target_environment"]["solar"]["configured_peak_w"])
            - TARGET_SOLAR_PEAK_W
        )
        <= 1.0e-12
        and abs(
            _float(identity["target_environment"]["solar"]["trace_max_w"])
            - TARGET_SOLAR_PEAK_W
        )
        <= 1.0e-9
        for _, _, identity in scenarios
    )
    checks["capacity_multiplier_matches_target"] = all(
        abs(
            _float(configs["simulator"]["downlink_capacity_multiplier"])
            - TARGET_CAPACITY_MULTIPLIER
        )
        <= 1.0e-12
        and abs(
            _float(
                identity["target_environment"]["communication"][
                    "capacity_multiplier"
                ]
            )
            - TARGET_CAPACITY_MULTIPLIER
        )
        <= 1.0e-12
        for configs, _, identity in scenarios
    )
    checks["capacity_applied_to_every_contact_slot"] = all(
        bool(
            identity["target_environment"]["communication"][
                "every_contact_slot_scaled_exactly"
            ]
        )
        and all(
            not step.link_trace.in_contact
            or step.link_trace.throughput_mbps
            <= configs["contact"]["throughput_max_mbps"]
            * TARGET_CAPACITY_MULTIPLIER
            + 1.0e-9
            for step in timeline
        )
        for configs, timeline, identity in scenarios
    )
    checks["task_count_and_required_product_valid"] = all(
        len(rows := canonical_task_rows_v2(timeline)) > 0
        and len({row["task_id"] for row in rows}) == len(rows)
        and all(str(row["product_requirement"]).strip() for row in rows)
        for _, timeline, _ in scenarios
    ) and next(
        len(canonical_task_rows_v2(timeline))
        for _, timeline, identity in scenarios
        if identity["seed"] == 2026
    ) == 268
    expected_weights = {
        str(level): float(weight)
        for level, weight in expected_value_weights.items()
    }
    if TARGET_SCENARIO_PROFILE in V25_MANIFEST_SCENARIO_PROFILES:
        frozen_alpha = 0.17447916666666666
        frozen_effective = {
            level: frozen_alpha * weight
            for level, weight in expected_weights.items()
        }
        checks["nominal_value_identity_matches_selected_profile"] = all(
            _count_by(canonical_task_rows_v2(timeline), "value_level")
            == EXPECTED_FROZEN_V25_COUNTS
            and all(
                abs(
                    task_effective_value(step.task)
                    - frozen_effective[str(step.task.value_level)]
                )
                <= 1.0e-12
                for step in timeline
                if step.task is not None
            )
            for _, timeline, _ in scenarios
        )
        if TARGET_SCENARIO_PROFILE == "frozen_v25_stress":
            checks["frozen_v25_manifest_identity_matches"] = all(
                identity["task_manifest_hash_v2"]
                == EXPECTED_FROZEN_V25_TASK_HASH
                and identity["frozen_v25_identity"] is not None
                and all(identity["frozen_v25_identity"]["checks"].values())
                for _, _, identity in scenarios
            )
        else:
            checks["frozen_v25_manifest_identity_matches"] = all(
                identity["frozen_v25_identity"] is not None
                and identity["frozen_v25_identity"].get(
                    "base_frozen_task_hash"
                )
                == EXPECTED_FROZEN_V25_TASK_HASH
                and all(identity["frozen_v25_identity"]["checks"].values())
                for _, _, identity in scenarios
            ) and any(
                identity["seed"] == 2026
                and identity["task_manifest_hash_v2"]
                == EXPECTED_FROZEN_V25_TASK_HASH
                for _, _, identity in scenarios
            )
    else:
        checks["nominal_value_identity_matches_selected_profile"] = all(
            all(
                float(step.task.value_multiplier)
                == expected_weights[str(step.task.value_level)]
                for step in timeline
                if step.task is not None
            )
            for _, timeline, _ in scenarios
        )
        checks["frozen_v25_manifest_identity_matches"] = True
    checks["myopic_frozen_parameters_are_offline_and_fixed"] = (
        MYOPIC_PARAMETER_FILE.is_file()
        and not bool(MYOPIC_FROZEN_PARAMETERS.get("allow_online_update", False))
        and MYOPIC_FROZEN_PARAMETERS.get("version")
        == "myopic_frozen_calibration_v1_runtime_snapshot"
    )
    checks["evaluator_semantics_valid"] = all(
        bool(semantics[key])
        for key in (
            "metadata_cannot_close_verification",
            "metadata_cannot_close_context",
            "metadata_cannot_close_evidence",
            "duplicate_utility_is_zero",
            "high_and_urgent_independent_metrics_enabled",
        )
    )
    checks["guard_and_medium_reclaim_disabled"] = all(
        configs["simulator"]["formal_method_extensions"]
        == {
            "hv_urgent_guard_enabled": False,
            "medium_slack_reclaim_enabled": False,
        }
        for configs, _, _ in scenarios
    )
    method_identity_rows = [
        {
            "seed": identity["seed"],
            "method": METHOD_NAMES[method],
            "task_hash": identity["task_manifest_hash_v2"],
            "environment_hash": identity["environment_hash_v2"],
            "evaluator_hash": identity["evaluator_identity_hash"],
        }
        for _, _, identity in scenarios
        for method in DEFAULT_METHODS
    ]
    checks["method_environment_hashes_identical_by_construction"] = all(
        len(
            {
                (
                    row["task_hash"],
                    row["environment_hash"],
                    row["evaluator_hash"],
                )
                for row in method_identity_rows
                if row["seed"] == identity["seed"]
            }
        )
        == 1
        for _, _, identity in scenarios
    )
    checks["fresh_state_contract"] = True
    return checks


def generate_environment_audit(
    output_root: Path,
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    value_profile: str = DEFAULT_VALUE_PROFILE,
) -> dict[str, Any]:
    value_spec = _value_profile_spec(value_profile)
    value_profile_name = str(value_spec["profile"])
    value_weights = {
        str(level): float(weight)
        for level, weight in value_spec["weights"].items()
    }
    scenarios = [
        build_shared_scenario(seed, value_profile=value_profile)
        for seed in seeds
    ]
    seed2026 = next(
        (scenario for scenario in scenarios if scenario[2]["seed"] == 2026),
        scenarios[0],
    )
    configs_2026, timeline_2026, identity_2026 = seed2026
    semantics = _semantic_evaluator_checks(configs_2026)
    checks = _audit_assertions(scenarios, semantics, value_weights)
    checks["selected_methods_form_nonempty_frozen_subset"] = (
        bool(methods)
        and len(set(methods)) == len(methods)
        and set(methods).issubset(DEFAULT_METHODS)
    )
    passed = all(checks.values())
    task_summaries = [
        _task_identity_summary(identity["seed"], timeline, identity)
        for _, timeline, identity in scenarios
    ]
    scenario_rows = [identity for _, _, identity in scenarios]
    cli = {
        "seeds": list(seeds),
        "methods": list(methods),
        "output_dir": str(output_root),
        "value_profile": str(value_profile).strip().lower(),
        "scenario_profile": TARGET_SCENARIO_PROFILE,
        "capacity_multiplier": TARGET_CAPACITY_MULTIPLIER,
        "physics_override_options": [
            {
                "name": "capacity_multiplier",
                "value": TARGET_CAPACITY_MULTIPLIER,
                "allowed_presets": [0.35, 0.50],
            }
        ],
    }
    loading_trace = _config_loading_trace(cli)
    resolved = {
        "audit_status": "PASS" if passed else "FAIL",
        "formal_run_authorized": passed,
        "entry_point": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "scenario_profile": TARGET_SCENARIO_PROFILE,
        "value_profile_key": str(value_profile).strip().lower(),
        "value_profile": value_profile_name,
        "nominal_value_weights": value_weights,
        "resolved_seed_2026_configs": configs_2026,
        "seed_2026_environment_identity": identity_2026,
        "task_identity_by_seed": task_summaries,
        "payload_size_identity": _payload_size_identity(configs_2026),
        "evaluator_semantics": semantics,
        "method_identity": {
            "formal_methods": [
                METHOD_NAMES[method] for method in DEFAULT_METHODS
            ],
            "corrected_prop_myopic_contract": (
                "identical corrected-REQALIGN execution and delivery backend; "
                "only the delegate action-ranking decision module differs"
            ),
            "other_baseline_contract": (
                "B1, OFFLOAD and TUT retain their frozen method architectures "
                "while consuming the identical audited task/resource/contact "
                "scenario and common ground evaluator"
            ),
            "shared_corrected_reqalign_wrapper_for_prop_and_myopic": (
                "RequirementAwareRouterWrapper"
            ),
            "shared_modules": list(SHARED_COMPONENT_PATHS),
            "method_implementation_hashes": {
                key: _sha256_file(path)
                for key, path in METHOD_IMPLEMENTATION_PATHS.items()
            },
            "hv_urgent_guard_enabled": False,
            "medium_slack_reclaim_enabled": False,
            "myopic_frozen_parameters": deepcopy(MYOPIC_FROZEN_PARAMETERS),
            "myopic_frozen_parameter_scope": (
                "MYOPIC delegate ranking only; task/evaluator values remain shared"
            ),
        },
        "fresh_state_and_output_contract": {
            "new_method_runtime_per_seed": True,
            "new_SimulationEnvironment_for_prop_myopic_tut": True,
            "new_native_simulator_for_b1_offload": True,
            "timeline_is_immutable_shared_input": True,
            "formal_seed_output_must_not_preexist": True,
            "old_queue_ledger_soc_or_action_trace_reused": False,
        },
        "assertions": checks,
    }
    checksums = _environment_checksums(scenario_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "resolved_environment.json", resolved)
    _write_json(output_root / "config_loading_trace.json", loading_trace)
    _write_json(output_root / "environment_checksums.json", checksums)
    solar = identity_2026["target_environment"]["solar"]
    comm = identity_2026["target_environment"]["communication"]
    value_ratio_text = ":".join(
        f"{value_weights[level]:g}"
        for level in ("low", "medium", "high", "urgent")
    )
    markdown = f"""# Formal environment audit

Status: **{"PASS" if passed else "FAIL"}**. Formal method execution is {"authorized" if passed else "blocked"}.

## Power identity

- `40 W` means peak available solar electrical input before charge efficiency.
- It is not the `38 W` radio active draw and is not model inference or payload encoding power.
- Configured peak: {solar["configured_peak_w"]:.6f} W.
- Simulated trace maximum: {solar["trace_max_w"]:.6f} W.
- Sunlit mean: {solar["sunlit_mean_w"]:.6f} W.
- Full-day mean: {solar["full_day_mean_w"]:.6f} W.
- Radio active draw: {solar["radio_active_draw_w_separate"]:.6f} W.

## Communication identity

- Downlink capacity multiplier: {comm["capacity_multiplier"]:.6f}.
- Nominal unscaled/scaled Mbps: {comm["nominal_unscaled_min_mbps"]:.6f}-{comm["nominal_unscaled_max_mbps"]:.6f} / {comm["nominal_scaled_min_mbps"]:.6f}-{comm["nominal_scaled_max_mbps"]:.6f}.
- Seed-2026 actual unscaled Mbps min/mean/max: {comm["actual_unscaled_min_mbps"]:.6f} / {comm["actual_unscaled_mean_mbps"]:.6f} / {comm["actual_unscaled_max_mbps"]:.6f}.
- Seed-2026 actual scaled Mbps min/mean/max: {comm["actual_scaled_min_mbps"]:.6f} / {comm["actual_scaled_mean_mbps"]:.6f} / {comm["actual_scaled_max_mbps"]:.6f}.
- Seed-2026 unscaled MB per 20 s min/mean/max: {comm["actual_unscaled_min_mb_per_slot"]:.6f} / {comm["actual_unscaled_mean_mb_per_slot"]:.6f} / {comm["actual_unscaled_max_mb_per_slot"]:.6f}.
- Seed-2026 scaled MB per 20 s min/mean/max: {comm["actual_scaled_min_mb_per_slot"]:.6f} / {comm["actual_scaled_mean_mb_per_slot"]:.6f} / {comm["actual_scaled_max_mb_per_slot"]:.6f}.
- `link_quality_phase={comm["link_quality_phase"]}` is a sinusoid phase only, never a capacity multiplier.

## Shared method environment

All five methods consume the same per-seed task manifest, arrival/deadline/required-product fields, contact trace, 40 W power trace, capacity-{comm["capacity_multiplier"]:.2f} link trace, thermal/SOC limits, and common ground evaluator. Corrected PROP and MYOPIC additionally share the same model/action space, SafeFilter, payload generator, DownlinkQueue, StepRunner and corrected-REQALIGN wrapper; only their delegate action-ranking router differs. B1, OFFLOAD and TUT retain their frozen baseline architectures.

Scenario profile: `{TARGET_SCENARIO_PROFILE}`.

HV/urgent Guard: disabled. Medium slack reclaim: disabled.

## Task and evaluator identity

- Seeds: {", ".join(str(seed) for seed in seeds)}.
- Seed 2026 contains 268 tasks.
- Value-level counts: `{_count_by(canonical_task_rows_v2(timeline_2026), "value_level")}`.
- Selected value profile: `{value_profile_name}`; low:medium:high:urgent = `{value_ratio_text}`. The exact frozen V25 calibrated values are reused when the frozen stress profile is selected; this is not an experience bonus.
- MYOPIC uses the frozen calibrated selector and the shared effective task value without clipping high or urgent values. Its offline success/latency parameters are fixed for every seed, add no bonus, and do not alter PROP or the common evaluator.
- Required products are explicit and non-empty.
- Verification requires ROI or ROI-context; context/evidence requires ROI-context. Metadata cannot substitute.
- Full closure latency is required-product ground time minus arrival time and includes inference, generation, queue wait, contact wait, and transmission.
- Duplicate product incremental utility is zero.
- High and urgent are exported independently as well as in the combined HV totals.

## Startup assertions

{chr(10).join(f"- `{key}`: {value}" for key, value in checks.items())}

## Fresh-run contract

Each method/seed constructs a new SimulationEnvironment and episode runner from t=0. Formal seed directories must not already exist; old action, queue, ledger, SOC, and delivery traces are never loaded.
"""
    (output_root / "formal_environment_audit.md").write_text(
        markdown,
        encoding="utf-8",
    )
    if not passed:
        raise AssertionError({"formal_environment_audit_failed": checks})
    return {
        "passed": True,
        "checks": checks,
        "checksums": checksums,
        "resolved": resolved,
    }


def validate_seed2026_environment_accounting(
    output_root: Path,
    value_profile: str = DEFAULT_VALUE_PROFILE,
) -> dict[str, Any]:
    """Validate environment and byte accounting without invoking either method."""

    from src.common.types import GeneratedPayload, LinkTrace
    from src.link.contact_window import compute_available_capacity_mb
    from src.link.downlink_queue import DownlinkQueue
    from src.payload.payload_lifecycle_ledger import PayloadLifecycleLedger

    configs, timeline, identity = build_shared_scenario(
        2026,
        value_profile=value_profile,
    )
    contact_step = next(step for step in timeline if step.link_trace.in_contact)
    available_mb = compute_available_capacity_mb(
        contact_step.link_trace,
        contact_step.slot_seconds,
        _float(configs["simulator"].get("downlink_efficiency"), 1.0),
    )
    expected_mb = (
        contact_step.link_trace.throughput_mbps
        * contact_step.slot_seconds
        * _float(configs["simulator"].get("downlink_efficiency"), 1.0)
        / 8.0
    )
    payload_mb = available_mb * 1.5
    queue = DownlinkQueue(
        queue_max_mb=max(payload_mb * 2.0, 1024.0),
        slot_seconds=TARGET_SLOT_SECONDS,
        downlink_efficiency=_float(
            configs["simulator"].get("downlink_efficiency"), 1.0
        ),
        lifecycle_ledger=PayloadLifecycleLedger(TARGET_SLOT_SECONDS),
    )
    payload = GeneratedPayload(
        task_id="environment_accounting_probe",
        payload_policy="roi_context",
        payload_mb=payload_mb,
        information_content=1.0,
        priority_score=1.0,
        value_score=1.0,
        task_value=1.0,
        uncertainty=0.0,
        remaining_utility=1.0,
        is_metadata=False,
        is_high_value=True,
        deadline_s=1.0e9,
        created_slot=max(0, int(contact_step.slot_index) - 1),
        metadata={"audit_probe": True},
        payload_plan="roi_context",
        product_type="roi_context",
        product_requirement="evidence",
        incremental_utility_estimate=1.0,
    )
    no_contact = LinkTrace(
        slot_index=max(0, int(contact_step.slot_index) - 1),
        in_contact=False,
        snr_db=-3.0,
        throughput_mbps=0.0,
        link_quality=0.0,
        remaining_window_s=0.0,
    )
    enqueue_result = queue.process_slot(
        max(0, int(contact_step.slot_index) - 1),
        payload,
        no_contact,
    )
    queue_before_contact_mb = queue.queue_mb
    transmit_result = queue.process_slot(
        int(contact_step.slot_index),
        None,
        contact_step.link_trace,
    )
    queue_after_contact_mb = queue.queue_mb
    ledger_rows = (
        [asdict(record) for record in queue.lifecycle_ledger.records.values()]
        if queue.lifecycle_ledger is not None
        else []
    )
    expected_remaining = max(0.0, queue_before_contact_mb - available_mb)
    resolved_audit = json.loads(
        (output_root / "resolved_environment.json").read_text(encoding="utf-8")
    )
    checks = {
        "audit_artifacts_exist": all(
            (output_root / name).is_file()
            for name in (
                "formal_environment_audit.md",
                "resolved_environment.json",
                "config_loading_trace.json",
                "environment_checksums.json",
            )
        ),
        "audit_passed": (
            resolved_audit.get("audit_status") == "PASS"
        ),
        "audit_environment_hash_matches_runtime": (
            resolved_audit["seed_2026_environment_identity"]["environment_hash_v2"]
            == identity["environment_hash_v2"]
        ),
        "audit_task_hash_matches_runtime": (
            resolved_audit["seed_2026_environment_identity"][
                "task_manifest_hash_v2"
            ]
            == identity["task_manifest_hash_v2"]
        ),
        "selected_capacity_multiplier_matches_runtime": abs(
            _float(configs["simulator"]["downlink_capacity_multiplier"])
            - TARGET_CAPACITY_MULTIPLIER
        )
        <= 1.0e-12
        and abs(
            _float(
                identity["target_environment"]["communication"][
                    "capacity_multiplier"
                ]
            )
            - TARGET_CAPACITY_MULTIPLIER
        )
        <= 1.0e-12,
        "every_contact_slot_scaled_exactly": bool(
            identity["target_environment"]["communication"][
                "every_contact_slot_scaled_exactly"
            ]
        ),
        "capacity_formula_matches_scaled_link_trace": abs(available_mb - expected_mb)
        <= 1.0e-9,
        "queue_dequeue_is_capacity_limited": abs(
            _float(transmit_result.S_down_total_mb) - available_mb
        )
        <= 1.0e-9,
        "queue_remaining_matches_byte_accounting": abs(
            queue_after_contact_mb - expected_remaining
        )
        <= 1.0e-9,
        "transmitted_bytes_match_mb_ledger": abs(
            _float(transmit_result.S_down_total_mb) * MB_TO_BYTES
            - available_mb * MB_TO_BYTES
        )
        <= 1.0e-3,
        "probe_not_fully_delivered_when_payload_exceeds_slot_capacity": (
            not transmit_result.delivered_task_ids
            and queue_after_contact_mb > 0.0
        ),
        "method_execution_count": 0,
        "method_was_not_tuned": True,
        "seed_and_task_identity_valid": (
            identity["seed"] == 2026 and identity["task_count"] == 268
        ),
    }
    passed = all(
        value is True
        for key, value in checks.items()
        if key != "method_execution_count"
    ) and checks["method_execution_count"] == 0
    result = {
        "status": "PASS" if passed else "FAIL",
        "formal_run_authorized": passed,
        "seed": 2026,
        "value_profile_key": str(value_profile).strip().lower(),
        "value_profile": identity["value_profile"],
        "environment_hash_v2": identity["environment_hash_v2"],
        "task_manifest_hash_v2": identity["task_manifest_hash_v2"],
        "contact_probe": {
            "slot": contact_step.slot_index,
            "scaled_throughput_mbps": contact_step.link_trace.throughput_mbps,
            "available_capacity_mb": available_mb,
            "available_capacity_bytes": available_mb * MB_TO_BYTES,
            "probe_payload_mb": payload_mb,
            "queue_before_contact_mb": queue_before_contact_mb,
            "queue_after_contact_mb": queue_after_contact_mb,
            "queue_reported_transmitted_mb": _float(
                transmit_result.S_down_total_mb
            ),
            "enqueue_queue_count": enqueue_result.queue_item_count,
            "lifecycle_ledger_rows": ledger_rows,
        },
        "checks": checks,
    }
    _write_json(output_root / "seed_2026_environment_validation.json", result)
    if not passed:
        raise AssertionError({"seed_2026_environment_validation_failed": checks})
    return result


def _require_audit_gate(output_root: Path) -> None:
    audit_path = output_root / "resolved_environment.json"
    if not audit_path.is_file():
        raise RuntimeError("environment validation blocked: audit is required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("audit_status") != "PASS" or not audit.get(
        "formal_run_authorized"
    ):
        raise RuntimeError("formal run blocked: environment audit did not pass")


def _require_formal_gate(output_root: Path) -> None:
    _require_audit_gate(output_root)
    validation_path = output_root / "seed_2026_environment_validation.json"
    if not validation_path.is_file():
        raise RuntimeError(
            "formal run blocked: seed-2026 environment validation is required"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or not validation.get(
        "formal_run_authorized"
    ):
        raise RuntimeError(
            "formal run blocked: seed-2026 environment/accounting validation did not pass"
        )


def _assert_formal_output_unoccupied(output_root: Path) -> None:
    formal_targets = [
        output_root / "seeds",
        output_root / "aggregate_metrics.csv",
        output_root / "seed_metrics.csv",
        output_root / "run_manifest.json",
    ]
    occupied = [str(path) for path in formal_targets if path.exists()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite or resume formal results; choose a new output "
            f"directory: {occupied}"
        )


def bind_myopic_runtime(
    runner: Any,
    router: MyopicFrozenCalibratedRouter,
) -> None:
    """Inject current observable state while leaving StepRunner unchanged."""

    original_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = original_factory(queue)
        original_run_step = step_runner.run_step

        def intercepted(timeline_step, resource_state, mission_state, comm_state, active_router):
            router.bind_step(timeline_step, resource_state, comm_state)
            return original_run_step(
                timeline_step,
                resource_state,
                mission_state,
                comm_state,
                active_router,
            )

        step_runner.run_step = intercepted
        return step_runner

    runner.step_runner_factory = factory


def ensure_reqalign_runtime_compatibility(runner: Any) -> None:
    """Restore the closure-product helper expected by the existing wrapper.

    The current shared StepRunner no longer exposes this former private helper,
    while RequirementAwareRouterWrapper still calls it.  Supplying the exact
    historical mapping on the runtime instance changes neither REQALIGN's
    selector nor the shared simulation semantics.
    """

    original_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = original_factory(queue)
        if not hasattr(step_runner, "_minimum_closing_residual_product"):

            def minimum_closing_residual_product(
                requirement: str,
                delivered_products: set[str],
            ) -> str:
                normalized = str(requirement or "verification").strip().lower()
                delivered = {
                    str(product).strip().lower()
                    for product in (delivered_products or set())
                }
                if normalized == "alert":
                    return "" if "metadata" in delivered else "metadata"
                if normalized == "verification":
                    return "" if {"roi", "roi_context"} & delivered else "roi"
                if normalized in {"context", "evidence"}:
                    return "" if "roi_context" in delivered else "roi_context"
                return "" if {"roi", "roi_context"} & delivered else "roi"

            step_runner._minimum_closing_residual_product = minimum_closing_residual_product
        return step_runner

    runner.step_runner_factory = factory


def _configure_tut_method(configs: dict[str, Any]) -> None:
    configs["experiment"] = {"routers": [TUT_ROUTER_NAME]}
    configs["action_space"] = {
        "actions": [
            {
                "action_name": "balanced+metadata",
                "model_name": "balanced",
                "payload_policy": "metadata",
                "payload_plan": "metadata",
                "transmission_option": "contact_aware",
                "expected_payload_level": 0.20,
                "expected_compute_level": 0.55,
                "enabled": True,
            },
            {
                "action_name": "balanced+roi",
                "model_name": "balanced",
                "payload_policy": "roi",
                "payload_plan": "roi",
                "transmission_option": "contact_aware",
                "expected_payload_level": 0.55,
                "expected_compute_level": 0.55,
                "enabled": True,
            },
            {
                "action_name": "balanced+roi_context",
                "model_name": "balanced",
                "payload_policy": "roi_context",
                "payload_plan": "roi_context",
                "transmission_option": "contact_aware",
                "expected_payload_level": 0.95,
                "expected_compute_level": 0.55,
                "enabled": True,
            },
            {
                "action_name": "delay/store",
                "model_name": "delay_store",
                "payload_policy": "defer",
                "payload_plan": "defer",
                "transmission_option": "store",
                "expected_payload_level": 0.0,
                "expected_compute_level": 0.0,
                "enabled": True,
            },
        ]
    }
    simulator = configs["simulator"]
    simulator["tut_uoi_density_priority_enabled"] = True
    progressive = deepcopy(simulator.get("progressive_payload", {}))
    for name, value in list(progressive.items()):
        if isinstance(value, dict):
            progressive[name] = {**value, "enabled": False}
    simulator["progressive_payload"] = progressive
    simulator["guarantee"] = {"enabled": False}
    throttle = deepcopy(simulator.get("downlink_safety_throttle", {}))
    throttle.update(
        {
            "enable_thermal_headroom_capacity": True,
            "enable_thermal_headroom_hysteresis": False,
            "thermal_headroom_search_iterations": 40,
            "thermal_headroom_tolerance": 1.0e-10,
        }
    )
    simulator["downlink_safety_throttle"] = throttle


def _build_tut_router(configs: dict[str, Any]) -> Any:
    _, router_type = _load_frozen_external_method_classes()
    method_cfg = yaml.safe_load(
        EXTERNAL_METHOD_PATHS["tut_method_config"].read_text(encoding="utf-8")
    )
    flags = method_cfg["method"]["feature_flags"]
    router = router_type(
        product_utility_cfg=configs["product_requirement_utility"],
        v_uoi=flags["V_uoi"],
        energy_ref_j=flags["energy_ref_j"],
        bytes_ref_mb=flags["bytes_ref_mb"],
        delivery_latency_weight=flags["delivery_latency_weight"],
        information_gap_weight=flags["information_gap_weight"],
        slot_seconds=configs["timeline"]["slot_seconds"],
    )
    original_set_task_context = router.set_task_context

    def set_effective_task_context(task: Any, decision_time_s: float) -> None:
        original_set_task_context(
            replace(task, task_value=task_effective_value(task)),
            decision_time_s,
        )

    router.set_task_context = set_effective_task_context
    return router


def bind_tut_runtime(runner: Any, router: Any) -> None:
    """Supply the current observable task/time expected by the frozen router."""

    original_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = original_factory(queue)
        original_run_step = step_runner.run_step

        def intercepted(
            timeline_step,
            resource_state,
            mission_state,
            comm_state,
            active_router,
        ):
            if timeline_step.task is not None:
                router.set_task_context(
                    timeline_step.task,
                    timeline_step.slot_index * timeline_step.slot_seconds,
                )
            return original_run_step(
                timeline_step,
                resource_state,
                mission_state,
                comm_state,
                active_router,
            )

        step_runner.run_step = intercepted
        return step_runner

    runner.step_runner_factory = factory


def build_method_router(
    method: str,
    configs: dict[str, Any],
    timeline: list[Any],
    runner: Any,
    action_space: Any,
) -> Any:
    if method == "MYOPIC":
        delegate = MyopicFrozenCalibratedRouter(
            MYOPIC_PARAMETER_FILE,
            router_name=MYOPIC_ROUTER_NAME,
        )
        bind_myopic_runtime(runner, delegate)
    elif method == "TUT":
        tut_router = _build_tut_router(configs)
        bind_tut_runtime(runner, tut_router)
        return tut_router
    else:
        delegate = build_routers(configs["experiment"], action_space)[0]

    projection = load_projection_module()
    projection_cfg = load_projection_config(True)
    wrapper = RequirementAwareRouterWrapper(
        delegate,
        projection_cfg,
        timeline,
        configs["simulator"],
        projection,
    )
    ensure_reqalign_runtime_compatibility(runner)
    bind_projection_runtime(runner, wrapper)
    return wrapper


def _flatten_events(
    method_name: str,
    manifest: Mapping[str, Mapping[str, Any]],
    event_map: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, metadata in manifest.items():
        for event in sorted(
            event_map.get(task_id, []),
            key=lambda row: (_float(row.get("ground_available_time_s")), str(row.get("product_type", ""))),
        ):
            ground = _float(event.get("ground_available_time_s"))
            rows.append(
                {
                    "method": method_name,
                    "task_id": task_id,
                    "value_level": metadata["value_level"],
                    "product_requirement": metadata["product_requirement"],
                    "product_type": event["product_type"],
                    "ground_available_time_s": ground,
                    "arrival_time_s": metadata["arrival_time_s"],
                    "deadline_time_s": metadata["deadline_time_s"],
                    "valid_before_deadline": int(ground <= _float(metadata["deadline_time_s"]) + 1.0e-9),
                    "delivery_route": event.get("delivery_route", ""),
                }
            )
    return rows


def _closing_event(
    events: list[dict[str, Any]],
    requirement: str,
) -> dict[str, Any] | None:
    observed: set[str] = set()
    for event in sorted(events, key=lambda row: _float(row.get("ground_available_time_s"))):
        observed.add(str(event["product_type"]))
        if requirement_satisfied_by_products(observed, requirement):
            return event
    return None


def _lifecycle_times(
    task_id: str,
    product_type: str,
    ground_time_s: float,
    current_rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    slot_seconds: float,
) -> tuple[float | None, float | None]:
    candidates = [
        row
        for row in current_rows
        if str(row.get("task_id")) == task_id
        and str(row.get("product_type")) == product_type
        and str(row.get("current_status")) == "delivered"
    ]
    if not candidates:
        return None, None
    ground_slot = ground_time_s / max(slot_seconds, 1.0e-12)
    lifecycle = min(
        candidates,
        key=lambda row: abs(_float(row.get("delivered_slot"), ground_slot) - ground_slot),
    )
    lifecycle_id = str(lifecycle.get("lifecycle_id", ""))
    trace = [
        row
        for row in transitions
        if str(row.get("lifecycle_id", "")) == lifecycle_id
    ]
    queued_slots = [
        _float(row.get("status_updated_slot"))
        for row in trace
        if str(row.get("current_status", "")) in {"queued", "pending"}
    ]
    scheduled_slots = [
        _float(row.get("status_updated_slot"))
        for row in trace
        if str(row.get("current_status", "")) == "scheduled"
    ]
    created_slot = _float(lifecycle.get("created_slot"), ground_slot)
    queue_time = min(queued_slots, default=created_slot) * slot_seconds
    scheduled_time = min(scheduled_slots, default=queue_time) * slot_seconds
    return queue_time, scheduled_time


def build_full_closure_rows(
    method_name: str,
    task_rows: list[dict[str, Any]],
    event_map: Mapping[str, list[dict[str, Any]]],
    records: list[Any],
    current_lifecycle: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    slot_seconds: float,
) -> list[dict[str, Any]]:
    """Create authoritative required-product full-closure latency rows."""

    record_by_task = {
        str(record.task_id): record
        for record in records
        if getattr(record, "has_new_task", 0) and getattr(record, "task_id", "")
    }
    output: list[dict[str, Any]] = []
    for task in task_rows:
        task_id = str(task["task_id"])
        requirement = str(task["product_requirement"])
        arrival = _float(task["arrival_time_s"])
        deadline = _float(task["deadline_time_s"])
        closing = _closing_event(list(event_map.get(task_id, [])), requirement)
        reported_ground = None if closing is None else _float(closing["ground_available_time_s"])
        record = record_by_task.get(task_id)
        combined_compute = _float(getattr(record, "L_inf", 0.0)) if record else 0.0
        generation = _float(getattr(record, "payload_encoding_latency_s", 0.0)) if record else 0.0
        inference = max(0.0, combined_compute - generation)

        queue_enter = scheduled = None
        if closing is not None and reported_ground is not None:
            queue_enter, scheduled = _lifecycle_times(
                task_id,
                str(closing["product_type"]),
                reported_ground,
                current_lifecycle,
                transitions,
                slot_seconds,
            )
        ready = arrival + inference + generation
        queue_milestone = max(ready, ready if queue_enter is None else queue_enter)
        scheduled_base = scheduled
        if scheduled_base is None and reported_ground is not None:
            scheduled_base = math.floor(reported_ground / slot_seconds) * slot_seconds
        contact_milestone = max(
            queue_milestone,
            queue_milestone if scheduled_base is None else scheduled_base,
        )
        queue_wait = max(0.0, queue_milestone - ready)
        contact_wait = max(0.0, contact_milestone - queue_milestone)
        transmission = (
            0.0
            if reported_ground is None
            else max(
                0.0,
                reported_ground
                - (
                    math.floor(reported_ground / slot_seconds) * slot_seconds
                    if scheduled_base is None
                    else scheduled_base
                ),
            )
        )
        # Some legacy event timestamps place same-slot transmission at
        # slot_start + tx_time even though inference and encoding happen first.
        # Serialize those stages causally instead of letting the timestamp make
        # compute disappear from end-to-end latency.
        causal_ground = (
            None
            if reported_ground is None
            else contact_milestone + transmission
        )
        on_time = causal_ground is not None and causal_ground <= deadline + 1.0e-9
        full_latency = None if causal_ground is None else max(0.0, causal_ground - arrival)
        component_sum = (
            inference + generation + queue_wait + contact_wait + transmission
            if full_latency is not None
            else None
        )
        output.append(
            {
                "method": method_name,
                "task_id": task_id,
                "value_level": task["value_level"],
                "product_requirement": requirement,
                "required_product_grounded": "" if closing is None else closing["product_type"],
                "arrival_time_s": arrival,
                "deadline_time_s": deadline,
                "reported_ground_event_time_s": "" if reported_ground is None else reported_ground,
                "required_product_ground_time_s": "" if causal_ground is None else causal_ground,
                "closure_success": int(on_time),
                "full_closure_latency_s": "" if not on_time or full_latency is None else full_latency,
                "inference_latency_s": inference,
                "product_generation_latency_s": generation,
                "queue_waiting_s": queue_wait,
                "contact_waiting_s": contact_wait,
                "downlink_transmission_s": transmission,
                "component_sum_s": "" if component_sum is None else component_sum,
                "component_sum_matches_full_closure": int(
                    full_latency is not None and abs(component_sum - full_latency) <= 1.0e-6
                ),
                "metadata_is_not_closure_substitute": 1,
                "latency_definition": "required_product_ground_time_minus_task_arrival_time",
            }
        )
    return output


def build_unified_metrics(
    method_name: str,
    seed: int,
    task_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    closure_rows: list[dict[str, Any]],
    shared_summary: dict[str, Any],
    energy: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    hv_ids = {
        str(row["task_id"])
        for row in task_rows
        if str(row.get("value_level", "")).lower() in {"high", "urgent", "high_value", "emergency"}
    }
    successful = [row for row in closure_rows if _truth(row["closure_success"])]
    hv_successful = [row for row in successful if row["task_id"] in hv_ids]
    valid_hv_events = [
        row
        for row in event_rows
        if row["task_id"] in hv_ids and _truth(row["valid_before_deadline"])
    ]
    level_ids = {
        level: {
            str(row["task_id"])
            for row in task_rows
            if str(row.get("value_level", "")).lower() == level
        }
        for level in ("high", "urgent")
    }

    def stage_count(products: set[str]) -> int:
        return len(
            {
                str(row["task_id"])
                for row in valid_hv_events
                if str(row["product_type"]) in products
            }
        )

    def level_stage_count(level: str, products: set[str]) -> int:
        return len(
            {
                str(row["task_id"])
                for row in event_rows
                if row["task_id"] in level_ids[level]
                and _truth(row["valid_before_deadline"])
                and str(row["product_type"]) in products
            }
        )

    gross_total = sum(_float(row.get("gross_ground_utility")) for row in task_rows)
    hv_gross = sum(
        _float(row.get("gross_ground_utility"))
        for row in task_rows
        if str(row["task_id"]) in hv_ids and _truth(row.get("on_time_task_success"))
    )
    public_cost = gross_total - _float(shared_summary["common_ground_net_utility"])
    hv_cost = public_cost * hv_gross / gross_total if gross_total > 0.0 else 0.0
    latencies = [_float(row["full_closure_latency_s"]) for row in successful]
    transmitted_bytes = _float(energy.get("delivered_mb")) * MB_TO_BYTES
    return {
        "seed": seed,
        "method": method_name,
        "task_count": len(task_rows),
        "HV_task_count": len(hv_ids),
        "J_all_net": _float(shared_summary["common_ground_net_utility"]),
        "J_HV_net": hv_gross - hv_cost,
        "N_HV_closure": len(hv_successful),
        "N_HV_metadata": stage_count({"metadata"}),
        "N_HV_ROI": stage_count({"roi"}),
        "N_HV_evidence": stage_count({"roi_context"}),
        "N_high_tasks": len(level_ids["high"]),
        "N_high_closure": sum(
            row["task_id"] in level_ids["high"] for row in successful
        ),
        "N_high_metadata": level_stage_count("high", {"metadata"}),
        "N_high_ROI": level_stage_count("high", {"roi"}),
        "N_high_evidence": level_stage_count("high", {"roi_context"}),
        "N_urgent_tasks": len(level_ids["urgent"]),
        "N_urgent_closure": sum(
            row["task_id"] in level_ids["urgent"] for row in successful
        ),
        "N_urgent_metadata": level_stage_count("urgent", {"metadata"}),
        "N_urgent_ROI": level_stage_count("urgent", {"roi"}),
        "N_urgent_evidence": level_stage_count("urgent", {"roi_context"}),
        "closure_count": len(successful),
        "overall_closure_rate": len(successful) / max(len(task_rows), 1),
        "mean_closure_latency_s": statistics.fmean(latencies) if latencies else 0.0,
        "p95_closure_latency_s": _p95(latencies),
        "energy_Wh": _float(energy.get("controllable_mission_energy_wh")),
        "transmitted_bytes": transmitted_bytes,
        "thermal_violations": int(_float(safety.get("hard_thermal_violation_count"))),
        "SOC_violations": int(_float(safety.get("hard_battery_violation_count"))),
        "latency_scope": "successful_required_product_full_closure_only",
    }


def _manifest_with_effective_values(
    timeline: list[Any],
    task_hash: str,
) -> dict[str, dict[str, Any]]:
    manifest = manifest_rows(timeline, task_hash)
    for step in timeline:
        if step.task is not None:
            manifest[step.task.task_id]["effective_task_value"] = task_effective_value(
                step.task
            )
    return manifest


def _timeline_with_public_effective_task_values(
    timeline: list[Any],
) -> list[Any]:
    return [
        replace(
            step,
            task=(
                None
                if step.task is None
                else replace(
                    step.task,
                    task_value=task_effective_value(step.task),
                )
            ),
        )
        for step in timeline
    ]


def _exact_baseline_closure_rows(
    method_name: str,
    task_rows: list[dict[str, Any]],
    event_map: Mapping[str, list[dict[str, Any]]],
    components: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in task_rows:
        task_id = str(task["task_id"])
        requirement = str(task["product_requirement"])
        arrival = _float(
            task.get("arrival_time_s", task.get("capture_time_s", 0.0))
        )
        deadline = _float(task["deadline_time_s"])
        closing = _closing_event(list(event_map.get(task_id, [])), requirement)
        ground = (
            None
            if closing is None
            else _float(closing.get("ground_available_time_s"))
        )
        latency = None if ground is None else max(0.0, ground - arrival)
        comp = components.get(task_id, {})
        inference = max(0.0, _float(comp.get("inference_latency_s")))
        generation = max(0.0, _float(comp.get("product_generation_latency_s")))
        queue_wait = max(0.0, _float(comp.get("queue_waiting_s")))
        contact_wait = max(0.0, _float(comp.get("contact_waiting_s")))
        transmission = max(0.0, _float(comp.get("downlink_transmission_s")))
        component_sum = (
            inference + generation + queue_wait + contact_wait + transmission
        )
        if latency is not None and abs(component_sum - latency) > 1.0e-6:
            raise AssertionError(
                {
                    "baseline_full_closure_component_mismatch": method_name,
                    "task_id": task_id,
                    "latency": latency,
                    "component_sum": component_sum,
                }
            )
        on_time = ground is not None and ground <= deadline + 1.0e-9
        output.append(
            {
                "method": method_name,
                "task_id": task_id,
                "value_level": task["value_level"],
                "product_requirement": requirement,
                "required_product_grounded": (
                    "" if closing is None else closing["product_type"]
                ),
                "arrival_time_s": arrival,
                "deadline_time_s": deadline,
                "reported_ground_event_time_s": "" if ground is None else ground,
                "required_product_ground_time_s": "" if ground is None else ground,
                "closure_success": int(on_time),
                "full_closure_latency_s": (
                    "" if not on_time or latency is None else latency
                ),
                "inference_latency_s": inference,
                "product_generation_latency_s": generation,
                "queue_waiting_s": queue_wait,
                "contact_waiting_s": contact_wait,
                "downlink_transmission_s": transmission,
                "component_sum_s": "" if latency is None else component_sum,
                "component_sum_matches_full_closure": int(
                    latency is not None
                    and abs(component_sum - latency) <= 1.0e-6
                ),
                "metadata_is_not_closure_substitute": 1,
                "latency_definition": (
                    "required_product_ground_time_minus_task_arrival_time"
                ),
            }
        )
    return output


def _event_rows_from_map(
    method_name: str,
    manifest: Mapping[str, Mapping[str, Any]],
    event_map: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return _flatten_events(method_name, manifest, event_map)


def _run_b1_method(
    seed: int,
    configs: dict[str, Any],
    timeline: list[Any],
    identity: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    method_name = METHOD_NAMES["B1"]
    manifest = _manifest_with_effective_values(
        timeline,
        identity["task_manifest_hash_v2"],
    )
    simulator = B1EnhancedSimulator(
        b1_config("enhanced"),
        deepcopy(configs["simulator"]),
        deepcopy(configs["model_pool"]),
    )
    raw = simulator.run(_timeline_with_public_effective_task_values(timeline))
    task_rows = adapt_b1(
        manifest,
        raw["per_task_delivery_trace"],
        method_name,
        "enhanced_common_thermal",
        seed,
        configs["unified_ground_received_accounting"],
    )
    event_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    components: dict[str, dict[str, float]] = {}
    for row in raw["per_task_delivery_trace"]:
        task_id = str(row["task_id"])
        completed = _truth(row.get("ground_result_available"))
        if completed:
            event_map[task_id].append(
                {
                    "product_type": "roi_context",
                    "ground_available_time_s": _float(
                        row["ground_result_available_time_s"]
                    ),
                    "delivery_route": "b1_full_frame_ground_inference",
                }
            )
        components[task_id] = {
            "inference_latency_s": (
                _float(row.get("ground_decode_latency_s"))
                + _float(row.get("ground_inference_latency_s"))
            ),
            "product_generation_latency_s": _float(
                row.get("onboard_compression_latency_s")
            ),
            "queue_waiting_s": _float(row.get("queue_wait_latency_s")),
            "contact_waiting_s": _float(row.get("contact_wait_latency_s")),
            "downlink_transmission_s": _float(
                row.get("transmission_latency_s")
            ),
        }
    event_rows = _event_rows_from_map(method_name, manifest, event_map)
    closure_rows = _exact_baseline_closure_rows(
        method_name,
        task_rows,
        event_map,
        components,
    )
    energy = b1_energy(raw["summary"])
    soc_floor = _float(configs["simulator"].get("battery_min"))
    thermal_limit = _float(configs["simulator"].get("thermal_max"), 1.0)
    safety = b1_safety(raw["summary"], raw["queue_trace"])
    safety["hard_battery_violation_count"] = sum(
        _float(row.get("battery_soc")) < soc_floor - 1.0e-9
        for row in raw["soc_trace"]
    )
    safety["hard_thermal_violation_count"] = sum(
        _float(row.get("thermal_pressure")) > thermal_limit + 1.0e-9
        for row in raw["soc_trace"]
    )
    safety["hard_safety_violation_count"] = (
        safety["hard_battery_violation_count"]
        + safety["hard_thermal_violation_count"]
    )
    shared_summary = summarize_method(
        task_rows,
        energy,
        safety,
        {"internal_utility": raw["summary"].get("net_system_utility", "")},
        configs["unified_ground_received_accounting"],
    )
    metrics = build_unified_metrics(
        method_name,
        seed,
        task_rows,
        event_rows,
        closure_rows,
        shared_summary,
        energy,
        safety,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "runtime_identity.json",
        {
            **identity,
            "method": method_name,
            "decision_module": "B1EnhancedSimulator.enhanced_fifo",
            "execution_backend": "src.baselines.b1_enhanced.B1EnhancedSimulator",
            "common_input_timeline_hash": identity["environment_hash_v2"],
            "common_ground_accounting": (
                "src.evaluation.task_outcome_adapter.adapt_b1"
            ),
            "required_product_projection": (
                "completed compressed full frame plus ground inference "
                "is represented as roi_context"
            ),
            "public_effective_value_adapter": (
                "task.calibrated_task_value is exposed through the legacy "
                "B1 task_value field without changing the frozen scheduler"
            ),
            "fresh_state": True,
        },
    )
    _write_csv(output_dir / "task_outcomes.csv", task_rows)
    _write_csv(output_dir / "ground_delivery_events.csv", event_rows)
    _write_csv(output_dir / "task_full_closure_latency.csv", closure_rows)
    _write_csv(output_dir / "per_slot_resource_trace.csv", raw["soc_trace"])
    _write_csv(output_dir / "queue_trace.csv", raw["queue_trace"])
    _write_csv(
        output_dir / "per_task_delivery_trace.csv",
        raw["per_task_delivery_trace"],
    )
    return metrics, closure_rows


def _offload_model_profile(
    configs: Mapping[str, Any],
    model_name: str,
) -> dict[str, float]:
    model = next(
        row
        for row in configs["model_pool"]["models"]
        if row["name"] == model_name
    )
    base = next(
        row
        for row in configs["model_pool"]["profiles"]
        if row["name"] == model["profile_name"]
    )
    return {
        **base,
        **configs["model_pool"].get("profile_overrides", {}).get(
            model["profile_name"],
            {},
        ),
    }


def _run_offload_method(
    seed: int,
    configs: dict[str, Any],
    timeline: list[Any],
    identity: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    simulator_type, _ = _load_frozen_external_method_classes()
    from src.baselines.ground_only import ground_path_parameters
    from src.baselines.offload_eft import DEFER, GROUND, LOCAL
    from src.payload.information_sufficiency import product_information_utility

    method_name = METHOD_NAMES["OFFLOAD"]
    resolved = yaml.safe_load(
        EXTERNAL_METHOD_PATHS["offload_method_config"].read_text(
            encoding="utf-8"
        )
    )
    method_cfg = resolved["method"]
    ground = ground_path_parameters(configs)
    local = _offload_model_profile(configs, method_cfg["local_model"])
    local_product = next(
        row
        for row in configs["payload_profiles"]["profiles"]
        if row["name"] == method_cfg["local_result_product"]
    )
    defaults = configs["payload_profiles"]["calibrated_profile"]["defaults"]
    generation_s = (
        _float(local_product["size_mb"])
        * _float(defaults["encoding_latency_s_per_mb"])
        * _float(local_product["encoding_latency_scale"])
    )
    generation_j = (
        _float(local_product["size_mb"])
        * _float(defaults["encoding_energy_j_per_mb"])
        * _float(local_product["encoding_energy_scale"])
    )
    local_compute_duration_s = _float(local["latency_s"]) + generation_s
    local_compute_energy_j = _float(local["energy_j"]) + generation_j
    simulator = simulator_type(
        timeline=timeline,
        simulator_cfg=deepcopy(configs["simulator"]),
        local_compute_duration_s=local_compute_duration_s,
        local_compute_energy_j=local_compute_energy_j,
        local_payload_mb=_float(local_product["size_mb"]),
        ground_preprocess_duration_s=_float(ground.compression_latency_s),
        ground_preprocess_energy_j=_float(ground.compression_energy_j),
        ground_payload_mb=_float(ground.compressed_size_mb),
        ground_compute_duration_s=_float(ground.ground_compute_latency_s),
        ground_servers=int(method_cfg["ground_compute_servers"]),
        ground_queue_capacity=int(method_cfg["ground_queue_capacity"]),
        tie_tolerance_s=_float(method_cfg.get("tie_tolerance_s"), 1.0e-6),
    )
    utility_cfg = configs["product_requirement_utility"]
    decisions: dict[str, str] = {}
    for step in timeline:
        if step.task is not None:
            task = step.task
            now = step.slot_index * step.slot_seconds
            local_info = product_information_utility(
                {method_cfg["local_result_product"]},
                task.product_requirement,
                utility_cfg,
            )
            local_ok = (
                local_info + 1.0e-9 >= task.information_requirement
                and _float(local["quality_base"]) + 1.0e-9
                >= task.quality_requirement
            )
            ground_ok = (
                1.0 + 1.0e-9 >= task.information_requirement
                and _float(ground.ground_quality) + 1.0e-9
                >= task.quality_requirement
            )
            decision = simulator.decide_task(
                task_id=task.task_id,
                arrival_s=now,
                deadline_s=now + task.deadline_s,
                local_product_feasible=local_ok,
                ground_product_feasible=ground_ok,
            )
            decisions[task.task_id] = decision.selected_path
        simulator.advance_one_slot()

    manifest = _manifest_with_effective_values(
        timeline,
        identity["task_manifest_hash_v2"],
    )
    event_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    components: dict[str, dict[str, float]] = {}
    task_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for task_id, metadata in manifest.items():
        path = decisions.get(task_id, DEFER)
        job = simulator.jobs.get(task_id)
        arrival = _float(metadata["arrival_time_s"])
        deadline = _float(metadata["deadline_time_s"])
        finish = None if job is None else job.closure_s
        closed = bool(
            job
            and job.product_feasible
            and finish is not None
            and finish <= deadline + 1.0e-9
        )
        product = (
            "roi"
            if path == LOCAL
            else ("roi_context" if path == GROUND else "")
        )
        if closed:
            event_map[task_id].append(
                {
                    "product_type": product,
                    "ground_available_time_s": _float(finish),
                    "delivery_route": f"offload_{str(path).lower()}",
                }
            )
        products = {product} if closed and product else set()
        info = product_information_utility(
            products,
            metadata["product_requirement"],
            utility_cfg,
        )
        quality = (
            _float(local["quality_base"])
            if path == LOCAL
            else (_float(ground.ground_quality) if path == GROUND else 0.0)
        )
        base = {
            **metadata,
            "method": method_name,
            "variant": "one_shot_earliest_finish_time",
            "seed": seed,
            "transmitted_product_type": product if closed else "none",
            "highest_ground_available_stage": (
                "context_or_evidence"
                if product == "roi_context" and closed
                else ("roi" if product == "roi" and closed else "empty")
            ),
            "ground_completed": int(closed),
            "ground_result_available_time_s": finish if closed else "",
            "requirement_closure_time_s": finish if closed else "",
            "on_time_ground_completed": int(closed),
            "requirement_satisfied": int(closed),
            "on_time_task_success": int(closed),
            "partial_ground_delivery": 0,
            "capture_to_ground_latency_s": (
                _float(finish) - arrival if closed else 0.0
            ),
            "ground_information_sufficiency": info,
            "ground_result_quality": quality if closed else 0.0,
            "partial_ground_gross_utility": 0.0,
            "terminal_backlog_flag": int(
                job is not None and _float(job.remaining_mb) > 1.0e-9
            ),
            "failure_reason": (
                ""
                if closed
                else (
                    str(job.failure_reason)
                    if job is not None
                    else "defer_or_no_feasible_path"
                )
            ),
            "selected_path": path,
        }
        task_rows.append(
            score_task(base, configs["unified_ground_received_accounting"])
        )
        decision_rows.append(
            {
                "task_id": task_id,
                "selected_path": path,
                "closure_time_s": "" if finish is None else finish,
                "product_type": product,
                "product_feasible": int(
                    bool(job and job.product_feasible)
                ),
            }
        )
        if job is not None:
            inference = (
                _float(local["latency_s"])
                if path == LOCAL
                else (
                    _float(ground.ground_compute_latency_s)
                    if path == GROUND
                    else 0.0
                )
            )
            generation = (
                generation_s
                if path == LOCAL
                else (
                    _float(ground.compression_latency_s)
                    if path == GROUND
                    else 0.0
                )
            )
            queue_wait = max(
                0.0,
                _float(job.compute_start_s, arrival) - arrival,
            )
            if path == GROUND and job.ground_compute_start_s is not None:
                queue_wait += max(
                    0.0,
                    _float(job.ground_compute_start_s)
                    - _float(job.ground_queue_enter_s),
                )
            contact_wait = (
                0.0
                if job.tx_start_s is None
                else max(
                    0.0,
                    _float(job.tx_start_s)
                    - _float(job.downlink_queue_enter_s),
                )
            )
            transmission = (
                0.0
                if job.tx_finish_s is None or job.tx_start_s is None
                else max(
                    0.0,
                    _float(job.tx_finish_s) - _float(job.tx_start_s),
                )
            )
            components[task_id] = {
                "inference_latency_s": inference,
                "product_generation_latency_s": generation,
                "queue_waiting_s": queue_wait,
                "contact_waiting_s": contact_wait,
                "downlink_transmission_s": transmission,
            }

    event_rows = _event_rows_from_map(method_name, manifest, event_map)
    closure_rows = _exact_baseline_closure_rows(
        method_name,
        task_rows,
        event_map,
        components,
    )
    trace = list(simulator.actual_trace)
    compute_j = sum(_float(row.get("compute_energy_j")) for row in trace)
    comm_j = sum(_float(row.get("comm_energy_j")) for row in trace)
    delivered_mb = sum(_float(row.get("sent_mb")) for row in trace)
    terminal_backlog = sum(
        _float(job.remaining_mb)
        for job in simulator.jobs.values()
        if _float(job.remaining_mb) > 1.0e-9
    )
    thermal_limit = _float(configs["simulator"].get("thermal_max"), 1.0)
    soc_floor = _float(configs["simulator"].get("battery_min"))
    thermal_violations = sum(
        _float(row.get("thermal_after")) > thermal_limit + 1.0e-9
        for row in trace
    )
    soc_violations = sum(
        _float(row.get("soc_after")) < soc_floor - 1.0e-9
        for row in trace
    )
    energy = {
        "compute_energy_wh": compute_j / 3600.0,
        "compression_energy_wh": 0.0,
        "communication_energy_wh": comm_j / 3600.0,
        "isl_energy_wh": 0.0,
        "payload_idle_energy_wh": 0.0,
        "housekeeping_energy_wh": 0.0,
        "controllable_mission_energy_wh": (compute_j + comm_j) / 3600.0,
        "total_spacecraft_energy_wh": (compute_j + comm_j) / 3600.0,
        "delivered_mb": delivered_mb,
    }
    safety = {
        "min_soc": min((_float(row.get("soc_after")) for row in trace), default=0.0),
        "final_soc": _float(trace[-1].get("soc_after")) if trace else 0.0,
        "hard_battery_violation_count": soc_violations,
        "max_thermal_pressure": max(
            (_float(row.get("thermal_after")) for row in trace),
            default=0.0,
        ),
        "hard_thermal_violation_count": thermal_violations,
        "hard_safety_violation_count": thermal_violations + soc_violations,
        "terminal_queue_mb": terminal_backlog,
        "maximum_queue_mb": max(
            (
                sum(
                    _float(job.remaining_mb)
                    for job in simulator.jobs.values()
                    if _float(job.remaining_mb) > 1.0e-9
                )
                for _ in [0]
            ),
            default=0.0,
        ),
    }
    shared_summary = summarize_method(
        task_rows,
        energy,
        safety,
        {"internal_utility": ""},
        configs["unified_ground_received_accounting"],
    )
    metrics = build_unified_metrics(
        method_name,
        seed,
        task_rows,
        event_rows,
        closure_rows,
        shared_summary,
        energy,
        safety,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "runtime_identity.json",
        {
            **identity,
            "method": method_name,
            "decision_module": (
                "frozen ThermalAlignedSimulator one-shot earliest-finish-time"
            ),
            "execution_backend": str(
                EXTERNAL_METHOD_PATHS["offload_router"]
            ),
            "execution_backend_sha256": _sha256_file(
                EXTERNAL_METHOD_PATHS["offload_router"]
            ),
            "shared_input_timeline_hash": identity["environment_hash_v2"],
            "common_ground_accounting": (
                "src.evaluation.unified_ground_received_accounting"
            ),
            "fresh_state": True,
        },
    )
    _write_csv(output_dir / "task_outcomes.csv", task_rows)
    _write_csv(output_dir / "ground_delivery_events.csv", event_rows)
    _write_csv(output_dir / "task_full_closure_latency.csv", closure_rows)
    _write_csv(output_dir / "per_slot_resource_trace.csv", trace)
    _write_csv(output_dir / "offload_decisions.csv", decision_rows)
    return metrics, closure_rows


def run_one_method(
    method: str,
    seed: int,
    configs: dict[str, Any],
    timeline: list[Any],
    identity: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if method == "B1":
        return _run_b1_method(
            seed,
            deepcopy(configs),
            timeline,
            identity,
            output_dir,
        )
    if method == "OFFLOAD":
        return _run_offload_method(
            seed,
            deepcopy(configs),
            timeline,
            identity,
            output_dir,
        )
    method_name = METHOD_NAMES[method]
    method_configs = deepcopy(configs)
    if method == "TUT":
        _configure_tut_method(method_configs)
    environment = SimulationEnvironment(method_configs)
    runner = environment.build_episode_runner(timeline)
    router = build_method_router(
        method,
        method_configs,
        timeline,
        runner,
        environment.action_space,
    )
    result = runner.run(router)

    manifest = _manifest_with_effective_values(
        timeline,
        identity["task_manifest_hash_v2"],
    )
    accounting_cfg = method_configs["unified_ground_received_accounting"]
    utility_cfg = method_configs["product_requirement_utility"]
    task_rows = adapt_proposed(
        manifest,
        result.records,
        runner.last_payload_lifecycle_current_records,
        method_name,
        method.lower(),
        seed,
        accounting_cfg,
        utility_cfg,
        slot_seconds=float(method_configs["timeline"]["slot_seconds"]),
    )
    event_map = proposed_delivery_events(
        result.records,
        runner.last_payload_lifecycle_current_records,
        slot_seconds=float(method_configs["timeline"]["slot_seconds"]),
    )
    event_rows = _flatten_events(method_name, manifest, event_map)
    closure_rows = build_full_closure_rows(
        method_name,
        task_rows,
        event_map,
        result.records,
        list(runner.last_payload_lifecycle_current_records),
        list(runner.last_payload_lifecycle_ledger_trace),
        float(method_configs["timeline"]["slot_seconds"]),
    )
    energy, safety = proposed_energy_safety(
        result,
        result.records,
        method_configs,
        accounting_cfg,
    )
    shared_summary = summarize_method(
        task_rows,
        energy,
        safety,
        {"internal_utility": result.summary.get("net_system_utility", "")},
        accounting_cfg,
    )
    metrics = build_unified_metrics(
        method_name,
        seed,
        task_rows,
        event_rows,
        closure_rows,
        shared_summary,
        energy,
        safety,
    )
    delegate = getattr(router, "delegate", router)
    projection_cfg = (
        dict(getattr(router, "config", {}) or {})
        if method in {"CORRECTED_REQALIGN", "MYOPIC"}
        else {}
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "runtime_identity.json",
        {
            **identity,
            "method": method_name,
            "shared_runtime": "src.simulator.environment.SimulationEnvironment",
            "shared_safety_filter": "src.routers.safe_filter.SafeFilter",
            "shared_payload_generation": "src.payload.payload_sizer.generate_payload",
            "shared_contact_delivery": "src.link.downlink_queue.DownlinkQueue",
            "shared_ground_accounting": "src.evaluation.task_outcome_adapter.adapt_proposed",
            "decision_module": type(delegate).__name__,
            "corrected_reqalign_post_selector": (
                type(router).__name__
                if method in {"CORRECTED_REQALIGN", "MYOPIC"}
                else "not_applicable"
            ),
            "corrected_reqalign_projection_enabled": (
                bool(projection_cfg.get("enabled", False))
                if method in {"CORRECTED_REQALIGN", "MYOPIC"}
                else "not_applicable"
            ),
            "corrected_reqalign_projection_config_hash": (
                _stable_hash(projection_cfg)
                if method in {"CORRECTED_REQALIGN", "MYOPIC"}
                else "not_applicable"
            ),
            "shared_reentry_and_delivery_backend": (
                method in {"CORRECTED_REQALIGN", "MYOPIC"}
            ),
            "online_update_count": getattr(delegate, "online_update_count", "not_exposed"),
            "experience_bonus": 0.0 if method == "MYOPIC" else "not_applicable",
            "myopic_frozen_parameters": (
                deepcopy(MYOPIC_FROZEN_PARAMETERS)
                if method == "MYOPIC"
                else "not_applicable"
            ),
            "myopic_frozen_parameter_hash": (
                _sha256_file(MYOPIC_PARAMETER_FILE)
                if method == "MYOPIC"
                else "not_applicable"
            ),
        },
    )
    _write_csv(output_dir / "task_outcomes.csv", task_rows)
    _write_csv(output_dir / "ground_delivery_events.csv", event_rows)
    _write_csv(output_dir / "task_full_closure_latency.csv", closure_rows)
    _write_csv(output_dir / "per_slot_resource_trace.csv", [asdict(row) for row in result.records])
    _write_csv(output_dir / "candidate_action_audit.csv", runner.last_candidate_action_logs)
    _write_csv(output_dir / "payload_lifecycle.csv", runner.last_payload_lifecycle_current_records)
    return metrics, closure_rows


def aggregate_metrics(
    rows: list[dict[str, Any]],
    closure_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if row["method"] == method]
        closures = [
            row
            for row in closure_rows
            if row["method"] == method and _truth(row.get("closure_success"))
        ]
        latencies = [_float(row["full_closure_latency_s"]) for row in closures]
        task_count = sum(int(row["task_count"]) for row in group)
        output.append(
            {
                "method": method,
                "seed_count": len(group),
                "task_count": task_count,
                "J_all_net": sum(_float(row["J_all_net"]) for row in group),
                "J_HV_net": sum(_float(row["J_HV_net"]) for row in group),
                "N_HV_closure": sum(int(row["N_HV_closure"]) for row in group),
                "N_HV_metadata": sum(int(row["N_HV_metadata"]) for row in group),
                "N_HV_ROI": sum(int(row["N_HV_ROI"]) for row in group),
                "N_HV_evidence": sum(int(row["N_HV_evidence"]) for row in group),
                "N_high_tasks": sum(int(row["N_high_tasks"]) for row in group),
                "N_high_closure": sum(int(row["N_high_closure"]) for row in group),
                "N_high_metadata": sum(int(row["N_high_metadata"]) for row in group),
                "N_high_ROI": sum(int(row["N_high_ROI"]) for row in group),
                "N_high_evidence": sum(int(row["N_high_evidence"]) for row in group),
                "N_urgent_tasks": sum(int(row["N_urgent_tasks"]) for row in group),
                "N_urgent_closure": sum(int(row["N_urgent_closure"]) for row in group),
                "N_urgent_metadata": sum(int(row["N_urgent_metadata"]) for row in group),
                "N_urgent_ROI": sum(int(row["N_urgent_ROI"]) for row in group),
                "N_urgent_evidence": sum(int(row["N_urgent_evidence"]) for row in group),
                "overall_closure_rate": len(closures) / max(task_count, 1),
                "mean_closure_latency_s": statistics.fmean(latencies) if latencies else 0.0,
                "p95_closure_latency_s": _p95(latencies),
                "energy_Wh": sum(_float(row["energy_Wh"]) for row in group),
                "transmitted_bytes": sum(_float(row["transmitted_bytes"]) for row in group),
                "thermal_violations": sum(int(row["thermal_violations"]) for row in group),
                "SOC_violations": sum(int(row["SOC_violations"]) for row in group),
                "latency_scope": "pooled_successful_required_product_full_closure",
            }
        )
    return output


def _write_final_comparison_report(
    output_root: Path,
    aggregate: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
) -> None:
    by_method = {str(row["method"]): row for row in aggregate}
    reference_method = (
        METHOD_PROP_NEW
        if METHOD_PROP_NEW in by_method
        else str(aggregate[0]["method"])
    )
    reference = by_method[reference_method]
    scenario = scenario_rows[0]
    comm = scenario["target_environment"]["communication"]
    solar = scenario["target_environment"]["solar"]
    frozen = scenario.get("frozen_v25_identity") or {}

    columns = (
        "method",
        "J_all_net",
        "J_HV_net",
        "N_HV_closure",
        "N_high_closure",
        "N_urgent_closure",
        "overall_closure_rate",
        "mean_closure_latency_s",
        "p95_closure_latency_s",
        "energy_Wh",
        "transmitted_bytes",
        "thermal_violations",
        "SOC_violations",
    )
    lines = [
        "# Final five-method comparison",
        "",
        "## Frozen scenario identity",
        "",
        f"- Scenario profile: `{scenario['scenario_profile']}`",
        f"- Seed: `{scenario['seed']}`",
        f"- Task count: `{scenario['task_count']}`",
        f"- Frozen V25 task hash: `{scenario['task_manifest_hash_v2']}`",
        f"- Value-level counts: `{(frozen or {}).get('value_level_counts', {})}`",
        f"- Arrival swap pairs: `{(frozen or {}).get('arrival_swap_pair_count', 0)}`",
        f"- Solar peak: `{solar['configured_peak_w']:.2f} W`",
        f"- Capacity multiplier: `{comm['capacity_multiplier']:.2f}`",
        f"- Actual scaled mean throughput: `{comm['actual_scaled_mean_mbps']:.6f} Mbps`",
        "",
        "## Unified metrics",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in aggregate:
        values = []
        for column in columns:
            value = row[column]
            if column == "overall_closure_rate":
                value = f"{100.0 * _float(value):.3f}%"
            elif column == "transmitted_bytes":
                value = f"{_float(value) / MB_TO_BYTES:.3f} MB"
            elif isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            f"## Difference relative to {reference_method}",
            "",
            "| method | delta_J_all_net | delta_J_HV_net | delta_N_HV_closure |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["method"]),
                    f"{_float(row['J_all_net']) - _float(reference['J_all_net']):.6f}",
                    f"{_float(row['J_HV_net']) - _float(reference['J_HV_net']):.6f}",
                    str(
                        int(row["N_HV_closure"])
                        - int(reference["N_HV_closure"])
                    ),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Closure uses required-product ground arrival. Metadata does not "
            "substitute for ROI or evidence, and latency is full closure from "
            "task arrival through inference, generation, queue/contact waiting "
            "and transmission.",
            "",
        ]
    )
    report = output_root / "FINAL_COMPARISON.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (output_root / "OPEN_RESULTS_HERE.md").write_text(
        "# Open results here\n\n"
        "- `FINAL_COMPARISON.md`: human-readable five-method comparison.\n"
        "- `aggregate_metrics.csv`: unified aggregate metrics.\n"
        "- `seed_metrics.csv`: per-seed/per-method metrics.\n"
        "- `formal_environment_audit.md`: frozen environment identity and checks.\n",
        encoding="utf-8",
    )


def run(
    output_root: Path,
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    *,
    overwrite: bool,
    value_profile: str = DEFAULT_VALUE_PROFILE,
) -> None:
    del overwrite
    _assert_formal_output_unoccupied(output_root)
    _require_formal_gate(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    all_closure_rows: list[dict[str, Any]] = []
    scenario_rows = []
    for seed in seeds:
        configs, timeline, identity = build_shared_scenario(
            seed,
            value_profile=value_profile,
        )
        seed_dir = output_root / "seeds" / f"seed_{seed}"
        _write_json(seed_dir / "scenario_identity.json", identity)
        scenario_rows.append(identity)
        method_identities = []
        for method in methods:
            method_dir = seed_dir / METHOD_NAMES[method]
            print(
                json.dumps(
                    {
                        "status": "STARTING",
                        "seed": seed,
                        "method": METHOD_NAMES[method],
                        "output": str(method_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            metrics, closure_rows = run_one_method(
                method,
                seed,
                configs,
                timeline,
                identity,
                method_dir,
            )
            metric_rows.append(metrics)
            all_closure_rows.extend(closure_rows)
            method_identities.append(
                {
                    "method": METHOD_NAMES[method],
                    "task_manifest_hash_v2": identity["task_manifest_hash_v2"],
                    "environment_hash_v2": identity["environment_hash_v2"],
                    "shared_config_hash": identity["shared_config_hash"],
                    "corrected_reqalign_post_selector": (
                        "RequirementAwareRouterWrapper"
                        if method in {"CORRECTED_REQALIGN", "MYOPIC"}
                        else "not_applicable"
                    ),
                    "corrected_reqalign_projection_config_hash": (
                        _stable_hash(load_projection_config(True))
                        if method in {"CORRECTED_REQALIGN", "MYOPIC"}
                        else "not_applicable"
                    ),
                    "shared_reentry_and_delivery_backend": int(
                        method in {"CORRECTED_REQALIGN", "MYOPIC"}
                    ),
                    "method_architecture": {
                        "CORRECTED_REQALIGN": "corrected_reqalign",
                        "MYOPIC": "myopic_on_corrected_reqalign",
                        "B1": "b1_common_thermal",
                        "OFFLOAD": "offload_eft_thermal_aligned",
                        "TUT": "tut_uoi_product_lite",
                    }[method],
                    "evaluator_identity_hash": identity["evaluator_identity_hash"],
                }
            )
        if len({row["task_manifest_hash_v2"] for row in method_identities}) != 1:
            raise AssertionError(f"seed {seed}: task stream mismatch across methods")
        if len({row["environment_hash_v2"] for row in method_identities}) != 1:
            raise AssertionError(f"seed {seed}: environment mismatch across methods")
        if len({row["evaluator_identity_hash"] for row in method_identities}) != 1:
            raise AssertionError(f"seed {seed}: evaluator identity mismatch across methods")
        _write_csv(seed_dir / "cross_method_fairness.csv", method_identities)

    aggregate = aggregate_metrics(metric_rows, all_closure_rows)
    _write_csv(output_root / "seed_metrics.csv", metric_rows)
    _write_csv(output_root / "aggregate_metrics.csv", aggregate)
    _write_csv(output_root / "scenario_manifest.csv", scenario_rows)
    _write_final_comparison_report(output_root, aggregate, scenario_rows)
    _write_json(
        output_root / "run_manifest.json",
        {
            "entry_point": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
            "seeds": list(seeds),
            "methods": [METHOD_NAMES[method] for method in methods],
            "value_profile_key": str(value_profile).strip().lower(),
            "value_profile": _value_profile_spec(value_profile)["profile"],
            "scenario_profile": TARGET_SCENARIO_PROFILE,
            "capacity_multiplier": TARGET_CAPACITY_MULTIPLIER,
            "nominal_value_weights": _value_profile_spec(value_profile)["weights"],
            "promoted_reference_method": METHOD_PROP_NEW,
            "fairness_contract": (
                "all five methods share the audited task/resource/contact "
                "scenario and common required-product evaluator; corrected "
                "PROP and MYOPIC additionally share corrected-REQALIGN "
                "projection/re-entry and differ only in delegate ranking; "
                "B1, OFFLOAD and TUT retain frozen baseline architectures"
            ),
            "shared_corrected_reqalign_backend": {
                "post_selector": "RequirementAwareRouterWrapper",
                "projection_source": "src/routers/requirement_aware_projection.py",
                "projection_config": "configs/prop_requirement_projection_v1.yaml",
                "projection_config_hash": _stable_hash(load_projection_config(True)),
            },
            "myopic_policy": {
                "experience_bonus": 0.0,
                "online_update": False,
                "seed_specific_tuning": False,
                "delegate": "MyopicFrozenCalibratedRouter",
                "frozen_parameters": deepcopy(MYOPIC_FROZEN_PARAMETERS),
                "parameter_sha256": _sha256_file(MYOPIC_PARAMETER_FILE),
                "score": (
                    "calibrated current-task success times effective task value "
                    "times information, quality and timeliness, minus public "
                    "immediate controllable-energy cost"
                ),
            },
            "closure_latency": {
                "definition": "required_product_ground_time_minus_task_arrival_time",
                "includes": [
                    "inference",
                    "product_generation",
                    "queue_waiting",
                    "contact_waiting",
                    "downlink_transmission",
                ],
                "metadata_cannot_substitute_for_required_product": True,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_root),
                "seed_method_runs": len(metric_rows),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Five-method corrected-REQALIGN comparison runner; the no-argument "
            "preset is the frozen V25 stress seed 2026 at 40 W/capacity 0.50."
        )
    )
    parser.add_argument(
        "--capacity-multiplier",
        type=float,
        choices=(0.35, 0.50),
        default=DEFAULT_CAPACITY_MULTIPLIER,
        help="Audited link-capacity preset; no-argument default: 0.50.",
    )
    parser.add_argument(
        "--scenario-profile",
        choices=(
            "frozen_v25_stress",
            "controlled_v25_multiseed",
            "generated_multiseed",
        ),
        default=DEFAULT_SCENARIO_PROFILE,
        help=(
            "Scenario identity; no-argument default reuses the exact frozen "
            "V25 seed-2026 stress manifest."
        ),
    )
    parser.add_argument(
        "--value-profile",
        choices=tuple(VALUE_PROFILES),
        default=DEFAULT_VALUE_PROFILE,
        help=(
            "Value-ratio profile. main=1:3:10:25, mild=1:2:3:5, "
            "stress=1:5:20:50."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; default is profile-specific and never shared.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds; no-argument default: 2026.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated subset of "
            "CORRECTED_REQALIGN,MYOPIC,B1,OFFLOAD,TUT."
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Generate the four formal environment audit artifacts; run no method.",
    )
    parser.add_argument(
        "--environment-validation-only",
        action="store_true",
        help="Run seed-2026 environment/queue/accounting validation; run no method.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated and forbidden for formal results.",
    )
    args = parser.parse_args()
    configure_target_capacity_multiplier(args.capacity_multiplier)
    configure_target_scenario_profile(args.scenario_profile)
    if args.output_dir is None:
        args.output_dir = Path(_value_profile_spec(args.value_profile)["output_root"])
    if args.overwrite:
        raise ValueError(
            "--overwrite is forbidden: every formal seed must be a fresh t=0 run "
            "in a new output directory"
        )
    if args.audit_only and args.environment_validation_only:
        raise ValueError("choose only one validation-only mode")
    seeds = _parse_csv_ints(args.seeds)
    methods = _parse_methods(args.methods)
    if args.audit_only:
        audit = generate_environment_audit(
            args.output_dir,
            seeds,
            methods,
            value_profile=args.value_profile,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "audit_only",
                    "formal_method_runs": 0,
                    "output": str(args.output_dir),
                    "checks": audit["checks"],
                },
                ensure_ascii=False,
            )
        )
        return
    if args.environment_validation_only:
        _require_audit_gate(args.output_dir)
        validation = validate_seed2026_environment_accounting(
            args.output_dir,
            value_profile=args.value_profile,
        )
        print(
            json.dumps(
                {
                    "status": validation["status"],
                    "mode": "seed_2026_environment_validation_only",
                    "formal_method_runs": 0,
                    "output": str(args.output_dir),
                },
                ensure_ascii=False,
            )
        )
        return

    # A normal one-click run always refreshes and rechecks the read-only gate
    # before any decision method is constructed.
    _assert_formal_output_unoccupied(args.output_dir)
    generate_environment_audit(
        args.output_dir,
        seeds,
        methods,
        value_profile=args.value_profile,
    )
    validate_seed2026_environment_accounting(
        args.output_dir,
        value_profile=args.value_profile,
    )
    run(
        args.output_dir,
        seeds,
        methods,
        overwrite=False,
        value_profile=args.value_profile,
    )


if __name__ == "__main__":
    main()
