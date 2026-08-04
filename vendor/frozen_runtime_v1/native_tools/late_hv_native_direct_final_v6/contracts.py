"""Immutable native-direct identities and output contracts."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any


CONTRACT_SHA256 = "5f6b6f1be7ecc28fc428b248823f60b5dc20bf9f118bb74e6dd3dabea6c42c32"
BASELINE_COMMIT = "96a90f5a95cf6bddea312b31a5c2986f3059608c"
METHODS = (
    "PROP-HEADROOM", "MYOPIC-E2E-UTILITY-GREEDY", "TUT-UOI-PRODUCT-LITE-v2",
    "B1-COMMON-THERMAL", "OFFLOAD-EFT-THERMAL-ALIGNED",
)
PER_TASK_FIELDS = (
    "method", "task_id", "arrival_time", "deadline", "value_level", "effective_task_value",
    "product_requirement", "selected_action", "executed_action", "selected_model",
    "delivered_product", "closure_success", "closure_ground_time", "failure_reason",
    "controllable_energy_j", "transmitted_bytes",
)
EVENT_FIELDS = (
    "method", "task_id", "stage", "ground_available_time", "deadline",
    "valid_before_deadline", "event_source", "duplicate_flag",
)
RESOURCE_FIELDS = (
    "method", "slot_count", "minimum_soc", "final_soc", "maximum_thermal",
    "soc_violation_count", "thermal_violation_count", "controllable_energy_wh", "transmitted_mb",
)
SLOT_RESOURCE_FIELDS = (
    "method", "slot", "time_s", "battery_soc", "thermal", "queue_mb",
    "compute_energy_j", "comm_energy_j", "idle_energy_j", "transmitted_mb",
    "in_contact", "soc_violation", "thermal_violation",
)
PRODUCT_STAGE_MAP = {
    "metadata": "metadata", "roi": "roi", "roi_context": "evidence",
    "context": "evidence", "evidence": "evidence", "full": "full",
    "compressed_full_frame": "full", "full_frame_result": "full",
}
PRODUCT_LAYER_ELIGIBILITY = {
    "metadata": ("metadata",),
    "roi": ("metadata", "roi"),
    "roi_context": ("metadata", "roi", "evidence"),
    "context": ("metadata", "roi", "evidence"),
    "evidence": ("metadata", "roi", "evidence"),
    "full": ("metadata", "roi", "evidence", "full"),
    "compressed_full_frame": ("metadata", "roi", "evidence", "full"),
    "full_frame_result": ("metadata", "roi", "evidence", "full"),
}

_PORTABLE_ROOT = Path(__file__).resolve().parents[2]


def _runtime_root(env_name: str, portable_name: str, legacy: str) -> str:
    override = os.environ.get(env_name)
    if override:
        return str(Path(override).expanduser().resolve())
    portable = _PORTABLE_ROOT / portable_name
    if portable.exists():
        return str(portable.resolve())
    return str(Path(legacy).resolve())


_CLEAN = _runtime_root(
    "LEO_B1_OFFLOAD_RUNTIME_ROOT",
    "b1_offload",
    "D:/wt/pilot_frozen_runtime_native_direct_v1",
)
_METHODS: dict[str, dict[str, Any]] = {
    "PROP-HEADROOM": {
        "source_root": _CLEAN, "wrapper": "run_prop_native.py",
        "entry_point": "src.evaluation.baselines.build_routers", "identifier": "proposed_lyapunov_rule_mode_safe_bandit",
        "router_class": "src.routers.safe_linucb_router.SafeLinUCBRouter",
        "hashes": {"src/evaluation/baselines.py": "a675b4a320383379d838542af2b04e5f2ed39371cb3b3305dc45d89f6e05367c", "configs/experiment.yaml": "a65f616e65dc482b17b80952905b94c9a62818652dc1f17aed811fbc9b213b32", "scripts/run_thermal_headroom_multiseed.py": "7c636b66f64a6b41a225b2eccb658669503f2f5c9d0e72c2df52110515f030a1"},
    },
    "MYOPIC-E2E-UTILITY-GREEDY": {
        "source_root": _runtime_root("LEO_MYOPIC_RUNTIME_ROOT", "myopic", "D:/wt/myopic_formal_nominal_v1"), "wrapper": "run_myopic_native.py",
        "entry_point": "src.evaluation.baselines.build_routers", "identifier": "myopic_e2e_utility_greedy",
        "router_class": "src.routers.myopic_e2e_utility_greedy.MyopicE2EUtilityGreedyRouter",
        "hashes": {"src/evaluation/baselines.py": "540415e16eb0007fd40e071897832b16dccbd7a6ca9cdf6ec5519f4bc8b3b721", "src/routers/myopic_e2e_utility_greedy.py": "7f02e53ba4bee061fea64d648011e8430811d2abf6e118797c64aa6276d11ae2", "configs/myopic_e2e_utility_greedy.yaml": "dce32c56ed9c19c1048c0f48a2466da8acc8950bcb4653f8ea6f3d444676f314"},
    },
    "TUT-UOI-PRODUCT-LITE-v2": {
        "source_root": _runtime_root("LEO_TUT_RUNTIME_ROOT", "tut", "D:/wt/tut_uoi_rate_lite"), "wrapper": "run_tut_native.py",
        "entry_point": "scripts.run_tut_uoi_rate_lite.build_router", "identifier": "TUT-v2-energy-fixed",
        "router_class": "src.routers.tut_uoi_rate_router.TutUoiRateRouter",
        "hashes": {"configs/experiments/tut_uoi_rate_lite_v2_formal_v1.yaml": "e49c37a1d099109de4b4183dec97e40d620f7f8c3083ebaeaf60b8dc9c475b9b", "scripts/run_tut_uoi_rate_lite_v2.py": "847a0f98e794d3be1e53e7e858d67371b18928a3b0120a60dc76f99b991ad448", "src/routers/safe_filter.py": "8cd989061309141db816e7ea98e3658e79531971d2104ca585b809847020b4e1", "src/simulator/step_runner.py": "1774494f0217912659005d15ff605468401fdba84659c55f5a80abe7f4adf8f0"},
    },
    "B1-COMMON-THERMAL": {
        "source_root": _CLEAN, "wrapper": "run_b1_native.py",
        "entry_point": "src.baselines.b1_enhanced.B1EnhancedSimulator", "identifier": "enhanced",
        "router_class": "src.baselines.b1_enhanced.B1EnhancedSimulator",
        "hashes": {"src/baselines/b1_enhanced.py": "7014ba3639e911a23a33a44cbe61de9ddbc604d927003b55d8e0af2cabae541c", "configs/baseline_b1_enhanced.yaml": "93914391e002a02de38fe223c3d35ebe0951ec1d63033cf5c5b5b6521106f33b"},
    },
    "OFFLOAD-EFT-THERMAL-ALIGNED": {
        "source_root": _CLEAN, "wrapper": "run_offload_native.py",
        "entry_point": "scripts.run_offload_vs_ground_24h_pilot.common_parameters", "identifier": "offload_eft_thermal_aligned",
        "router_class": "src.baselines.ground_only.GroundPathParameters",
        "hashes": {"scripts/run_three_method_minimal_robustness_v1.py": "5fbf97d9957a857634731fab94085dd694a9214fcbac8a88f384c535ea05ff40", "configs/methods/offload_eft_thermal_aligned_v1.yaml": "7a9d964b60e0eec28bebe955f0d9648cf6b7e689058ab70a4acf706682b58660"},
    },
}


def build_native_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_version": "late-hv-value-skew-native-direct-final-v6",
        "change_type": "MYOPIC_DECISION_VALUE_DOMAIN_ADAPTER",
        "baseline_commit": BASELINE_COMMIT,
        "canonical_contract_sha256": CONTRACT_SHA256, "method_order": list(METHODS),
        "methods": deepcopy(_METHODS),
        "scenario": {"seed": 2026, "capacity": 0.35, "horizon_s": 86400, "slot_seconds": 20, "task_count": 268},
        "decision_value_adapter": {
            "method": "MYOPIC-E2E-UTILITY-GREEDY",
            "canonical_field": "effective_task_value",
            "decision_field": "calibrated_task_value",
            "V_ref": "2.4787343282595278",
            "formula": "myopic_decision_value = effective_task_value / 2.4787343282595278",
            "mapping_sequence_sha256": "ff5259635e151a5d06196b5d5d96e7df4b6f51e5dbb6ee82c7c6f5be031f8615",
            "clip": False,
            "immutability_strategy": "dataclasses.replace",
        },
    }
