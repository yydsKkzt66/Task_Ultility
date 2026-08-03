"""Run the frozen legacy B1 baseline on the restored ordinary seed-2026 scene.

This is an isolated diagnostic entry point.  It does not modify the B1 source,
the trusted scenario bundle, or any prior output directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_normal_scenario_tut_proxy_ground_recovery_v1 as normal


ROOT = Path(__file__).resolve().parents[1]
METHOD = "B1-COMMON-THERMAL"
BUNDLE_ROOT = (
    ROOT
    / "outputs"
    / "_normal_scenario_single_seed_tut_proxy_ground_recovery_v1"
    / "attempt_006"
    / "scenario"
    / "ordinary_random_restored_seed2026"
)
OUTPUT_ROOT = ROOT / "outputs" / "_normal_scenario_seed2026_b1_common_thermal_rerun_v1"
GROUND_METRICS = (
    ROOT
    / "outputs"
    / "_normal_scenario_single_seed_tut_proxy_ground_recovery_v1"
    / "attempt_006"
    / "runs"
    / "GROUND_ONLY_FIFO_CORRECTED"
    / "frozen_public_metrics.json"
)


def next_attempt() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    index = 1
    while (OUTPUT_ROOT / f"attempt_{index:03d}").exists():
        index += 1
    return OUTPUT_ROOT / f"attempt_{index:03d}"


def delta_rows(b1: dict[str, Any], ground: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "J_all_net",
        "J_HV_net",
        "all_closure_count",
        "HV_closure_count",
        "high_closure_count",
        "urgent_closure_count",
        "transmitted_MB",
        "terminal_queue_MB",
        "peak_queue_MB",
        "mean_queue_MB",
        "mean_full_closure_latency_s",
        "p95_full_closure_latency_s",
        "controllable_energy_Wh",
        "minimum_soc",
        "maximum_thermal",
        "soc_violation_count",
        "thermal_violation_count",
    )
    rows = []
    for key in keys:
        old = normal.f(ground.get(key))
        new = normal.f(b1.get(key))
        rows.append({"metric": key, "GROUND_ONLY_FIFO_CORRECTED": old, METHOD: new, "B1_minus_GROUND": new - old})
    return rows


def main() -> int:
    attempt = next_attempt()
    run_dir = attempt / "runs" / METHOD
    audit_dir = attempt / "audit"
    table_dir = attempt / "tables"
    audit_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)

    before = {
        "bundle_contract_sha256": normal.sha256_file(BUNDLE_ROOT / "bundle_contract.json"),
        "task_manifest_sha256": normal.sha256_file(BUNDLE_ROOT / "task_manifest.csv"),
        "b1_source_sha256": normal.sha256_file(normal.v25.SOURCE_B1_OFFLOAD / "src" / "baselines" / "b1_enhanced.py"),
    }
    bundle = normal.load_neutral_bundle(BUNDLE_ROOT)
    if bundle.contract["task_hash"] != normal.EXPECTED_TASK_HASH:
        raise RuntimeError("ORDINARY_TASK_HASH_MISMATCH")

    normal.v25.child_setup_path(normal.v25.SOURCE_B1_OFFLOAD)
    import native_b1

    binding = native_b1.make_binding(normal.v25.SOURCE_B1_OFFLOAD.resolve())
    source_identity = dict(binding.identity())
    if not source_identity.get("source_identity_passed", False):
        raise RuntimeError("B1_SOURCE_IDENTITY_MISMATCH")
    scenario = binding.build_scenario(bundle)
    scenario, power = normal.v25.inject_power(scenario, normal.v25.SOURCE_B1_OFFLOAD)
    scenario, capacity_identity = normal.guardrun.apply_capacity(scenario, normal.CAPACITY)
    identity = normal.run_identity(METHOD, scenario, bundle, power, capacity_identity, source_identity)

    native = binding.simulate(scenario, normal.SLOTS)
    normalized = dict(binding.normalize_outputs(native, bundle))
    run_dir.mkdir(parents=True)
    normal.v25.write_unified_method_outputs(run_dir, normalized)
    normal.write_json(run_dir / "runtime_identity.json", identity)
    normal.write_json(run_dir / "resolved_config.json", scenario.configs)

    metrics = normal.evaluate(METHOD, run_dir, BUNDLE_ROOT)
    ground = normal.read_json(GROUND_METRICS)
    rows = delta_rows(metrics, ground)
    normal.write_csv(table_dir / "b1_vs_ground_metrics.csv", rows)
    normal.write_md(table_dir / "b1_vs_ground_metrics.md", rows)

    after = {
        "bundle_contract_sha256": normal.sha256_file(BUNDLE_ROOT / "bundle_contract.json"),
        "task_manifest_sha256": normal.sha256_file(BUNDLE_ROOT / "task_manifest.csv"),
        "b1_source_sha256": normal.sha256_file(normal.v25.SOURCE_B1_OFFLOAD / "src" / "baselines" / "b1_enhanced.py"),
    }
    audit = {
        "status": "PASS" if before == after else "READ_ONLY_INPUT_MUTATION_DETECTED",
        "method": METHOD,
        "seed": normal.SEED,
        "capacity": normal.CAPACITY,
        "task_hash": bundle.contract["task_hash"],
        "ordinary_random_restored": True,
        "fresh_state": True,
        "legacy_b1_source_unchanged": before == after,
        "hashes_before": before,
        "hashes_after": after,
    }
    normal.write_json(audit_dir / "identity_and_read_only_audit.json", audit)
    completion = {
        "status": "NORMAL_SCENARIO_B1_COMMON_THERMAL_SEED2026_COMPLETE" if audit["status"] == "PASS" else audit["status"],
        "output": str(attempt.resolve()),
        "metrics": metrics,
    }
    normal.write_json(attempt / "completion_status.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
