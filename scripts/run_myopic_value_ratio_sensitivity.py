"""One-click five-seed value-ratio sensitivity runs.

This entry runs the frozen five-method comparison for the mild 1:2:3:5 and
stress 1:5:20:50 profiles.  Before any method is invoked, it proves that the
profiles differ from the frozen 1:3:10:25 main scenario only in the declared
value-skew calibration and the resulting calibrated task-value fields.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_myopic_e2e_utility_greedy import (
    CAP050_FIVE_SEED_REFERENCE_ROOT,
    DEFAULT_METHODS,
    FORMAL_FIVE_SEEDS,
    VALUE_PROFILES,
    VALUE_SENSITIVITY_ROOT,
    _stable_hash,
    _write_json,
    build_shared_scenario,
    configure_target_capacity_multiplier,
    configure_target_scenario_profile,
    generate_environment_audit,
    run,
    validate_seed2026_environment_accounting,
)
from src.data.value_skew_calibration import task_effective_value
from src.evaluation.canonical_sarfish_environment import canonical_task_rows_v2


SENSITIVITY_PROFILES = ("mild", "stress")


def _configs_without_value_ratio(configs: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(configs)
    result.get("task_stream", {}).pop("value_skew_calibration", None)
    return result


def _target_environment_without_value_ratio(identity: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(identity["target_environment"])
    result.pop("task_value_identity", None)
    return result


def _task_values(timeline: list[Any]) -> list[float]:
    return [
        float(task_effective_value(step.task))
        for step in timeline
        if step.task is not None
    ]


def build_isolation_preflight(
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "method_set_is_frozen_five_method_set": methods
        == (
            "CORRECTED_REQALIGN",
            "MYOPIC",
            "B1",
            "OFFLOAD",
            "TUT",
        ),
        "sensitivity_profiles_are_exactly_mild_and_stress": (
            SENSITIVITY_PROFILES == ("mild", "stress")
        ),
        "mild_ratio_is_1_2_3_5": VALUE_PROFILES["mild"]["weights"]
        == {"low": 1.0, "medium": 2.0, "high": 3.0, "urgent": 5.0},
        "stress_ratio_is_1_5_20_50": VALUE_PROFILES["stress"]["weights"]
        == {"low": 1.0, "medium": 5.0, "high": 20.0, "urgent": 50.0},
    }
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        scenarios = {
            profile: build_shared_scenario(seed, value_profile=profile)
            for profile in ("main", *SENSITIVITY_PROFILES)
        }
        main_configs, main_timeline, main_identity = scenarios["main"]
        main_tasks = canonical_task_rows_v2(main_timeline)
        main_config_invariant_hash = _stable_hash(
            _configs_without_value_ratio(main_configs)
        )
        main_target_invariant_hash = _stable_hash(
            _target_environment_without_value_ratio(main_identity)
        )
        main_values = _task_values(main_timeline)
        profile_rows: dict[str, Any] = {}
        for profile in SENSITIVITY_PROFILES:
            configs, timeline, identity = scenarios[profile]
            tasks = canonical_task_rows_v2(timeline)
            profile_values = _task_values(timeline)
            profile_checks = {
                "canonical_task_manifest_identical": tasks == main_tasks,
                "task_manifest_hash_identical": (
                    identity["task_manifest_hash_v2"]
                    == main_identity["task_manifest_hash_v2"]
                ),
                "environment_hash_identical": (
                    identity["environment_hash_v2"]
                    == main_identity["environment_hash_v2"]
                ),
                "evaluator_identity_identical": (
                    identity["evaluator_identity_hash"]
                    == main_identity["evaluator_identity_hash"]
                ),
                "all_non_value_configs_identical": (
                    _stable_hash(_configs_without_value_ratio(configs))
                    == main_config_invariant_hash
                ),
                "power_contact_guard_environment_identical": (
                    _stable_hash(_target_environment_without_value_ratio(identity))
                    == main_target_invariant_hash
                ),
                "value_level_assignment_identical": (
                    [row["value_level"] for row in tasks]
                    == [row["value_level"] for row in main_tasks]
                ),
                "declared_weights_applied": (
                    identity["nominal_value_weights"]
                    == VALUE_PROFILES[profile]["weights"]
                ),
                "effective_values_recomputed": profile_values != main_values,
            }
            checks[f"seed_{seed}_{profile}_only_value_ratio_changes"] = all(
                profile_checks.values()
            )
            profile_rows[profile] = {
                "profile": identity["value_profile"],
                "weights": identity["nominal_value_weights"],
                "task_manifest_hash_v2": identity["task_manifest_hash_v2"],
                "environment_hash_v2": identity["environment_hash_v2"],
                "shared_config_hash": identity["shared_config_hash"],
                "effective_value_min": min(profile_values),
                "effective_value_max": max(profile_values),
                "checks": profile_checks,
            }
        per_seed.append(
            {
                "seed": seed,
                "main": {
                    "profile": main_identity["value_profile"],
                    "weights": main_identity["nominal_value_weights"],
                    "task_manifest_hash_v2": main_identity["task_manifest_hash_v2"],
                    "environment_hash_v2": main_identity["environment_hash_v2"],
                    "shared_config_hash": main_identity["shared_config_hash"],
                    "effective_value_min": min(main_values),
                    "effective_value_max": max(main_values),
                },
                "sensitivity": profile_rows,
            }
        )
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "formal_runs_authorized": passed,
        "main_reference_output": str(CAP050_FIVE_SEED_REFERENCE_ROOT),
        "main_reference_complete": (
            CAP050_FIVE_SEED_REFERENCE_ROOT / "run_manifest.json"
        ).is_file(),
        "seeds": list(seeds),
        "methods": list(methods),
        "run_order": list(SENSITIVITY_PROFILES),
        "only_allowed_change": (
            "declared low/medium/high/urgent value weights and the "
            "mean-preserving calibrated task-value fields derived from them"
        ),
        "checks": checks,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run both five-seed value-ratio sensitivities for the frozen "
            "five-method comparison in the 40 W/capacity-0.50 environment."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VALUE_SENSITIVITY_ROOT,
        help="Fresh parent directory for both sensitivity profiles.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write the cross-profile isolation audit without running a method.",
    )
    args = parser.parse_args()
    configure_target_capacity_multiplier(0.50)
    configure_target_scenario_profile("generated_multiseed")
    output_root = args.output_dir
    if output_root.exists():
        raise FileExistsError(
            f"sensitivity output already exists and will not be overwritten: {output_root}"
        )

    preflight = build_isolation_preflight(FORMAL_FIVE_SEEDS, DEFAULT_METHODS)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "value_ratio_isolation_preflight.json", preflight)
    if preflight["status"] != "PASS":
        raise AssertionError(
            {"value_ratio_isolation_preflight_failed": preflight["checks"]}
        )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "preflight_only",
                    "formal_method_runs": 0,
                    "output": str(output_root),
                },
                ensure_ascii=False,
            )
        )
        return

    for profile in SENSITIVITY_PROFILES:
        profile_dir = output_root / (
            "mild_1_2_3_5" if profile == "mild" else "stress_1_5_20_50"
        )
        print(
            json.dumps(
                {
                    "status": "STARTING",
                    "profile": profile,
                    "weights": VALUE_PROFILES[profile]["weights"],
                    "output": str(profile_dir),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        generate_environment_audit(
            profile_dir,
            FORMAL_FIVE_SEEDS,
            DEFAULT_METHODS,
            value_profile=profile,
        )
        validate_seed2026_environment_accounting(
            profile_dir,
            value_profile=profile,
        )
        run(
            profile_dir,
            FORMAL_FIVE_SEEDS,
            DEFAULT_METHODS,
            overwrite=False,
            value_profile=profile,
        )

    manifest = {
        "status": "PASS",
        "entry_point": "scripts/run_myopic_value_ratio_sensitivity.py",
        "seeds": list(FORMAL_FIVE_SEEDS),
        "methods": list(DEFAULT_METHODS),
        "profiles": {
            profile: {
                "profile": VALUE_PROFILES[profile]["profile"],
                "weights": VALUE_PROFILES[profile]["weights"],
                "output": str(
                    output_root
                    / (
                        "mild_1_2_3_5"
                        if profile == "mild"
                        else "stress_1_5_20_50"
                    )
                ),
            }
            for profile in SENSITIVITY_PROFILES
        },
    }
    _write_json(output_root / "sensitivity_run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
