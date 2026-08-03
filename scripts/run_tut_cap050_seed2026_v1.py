from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25
import scripts.run_hv_urgent_residual_protection_v1 as guardrun
import scripts.run_prop_reqalign_seed2026_v1 as reqrun


OUT = ROOT / "outputs" / "_v25_bupt1_tut_seed2026_cap050_v1"
CAP050_PROP = (
    ROOT
    / "outputs"
    / "_v25_bupt1_hv_urgent_residual_guard_seed2026_cap050_v1"
)
OFFICIAL_CAP035 = (
    ROOT / "outputs" / "_v25_bupt1_power_true_rerun_seed2026_v1"
)
METHOD = "TUT-UOI-PRODUCT-LITE-v2"
CAPACITY = 0.50
SLOTS = 4320


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_tut(bundle_root: Path, output: Path) -> dict[str, Any]:
    v25.child_setup_path(v25.SOURCE_TUT)
    bundle = v25.load_bundle(bundle_root)
    module = __import__("native_tut")
    binding = module.make_binding(v25.SOURCE_TUT.resolve())
    binding_identity = dict(binding.identity())
    if not binding_identity.get("source_identity_passed", False):
        raise RuntimeError(
            {"native_source_identity_failed": METHOD, "identity": binding_identity}
        )
    scenario = binding.build_scenario(bundle)
    scenario, power_identity = v25.inject_power(scenario, v25.SOURCE_TUT)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, CAPACITY)
    scenario_identity = v25.relaxed_scenario_identity(
        METHOD, scenario, bundle, power_identity
    )
    if (
        not scenario_identity["task_hash_match"]
        or scenario_identity["runtime_value_mismatch_count"]
    ):
        raise RuntimeError({"scenario_identity_failed": scenario_identity})

    native = binding.simulate(scenario, SLOTS)
    normalized = dict(binding.normalize_outputs(native, bundle))
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    identity = {
        **binding_identity,
        "capacity_identity": capacity_identity,
        "scenario_identity": scenario_identity,
        "frozen_power_identity": power_identity,
        "slots": SLOTS,
    }
    status = {
        "status": "PASS",
        "phase": "fresh_tut_cap050",
        "method": METHOD,
        "scenario_reconstruction_count": 1,
        "simulation_started": True,
        "fresh_state": True,
        "event_count": int(
            native.get("event_count", 0) if isinstance(native, dict) else 0
        ),
        "reused_action_trace": False,
        "reused_decision_trace": False,
        "reused_queue_trace": False,
        "reused_delivery_ledger": False,
        "reused_soc_trace": False,
        "reused_safety_filter_result": False,
        "identity": identity,
    }
    guardrun.write_json(output / "native_identity.json", identity)
    guardrun.write_json(
        output / "reconstructed_scenario_identity.json", scenario_identity
    )
    guardrun.write_json(output / "method_run_status.json", status)
    guardrun.write_json(
        output / "method_completed.json",
        {
            "phase": "fresh_tut_cap050",
            "method": METHOD,
            "status": "PASS",
            "scenario_reconstruction_count": 1,
        },
    )
    return identity


def method_paths(out: Path) -> dict[str, Path]:
    return {
        reqrun.METHOD_PROP_OLD: CAP050_PROP / "runs" / "OLD_PROP",
        reqrun.METHOD_PROP_NEW: CAP050_PROP / "runs" / "REQALIGN_REFERENCE",
        guardrun.METHOD_GUARD: CAP050_PROP / "runs" / "HV_GUARD",
        METHOD: out / "runs" / "TUT",
    }


def evaluate(
    out: Path, bundle_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_mod = v25.load_eval_module(out)
    config = reqrun.read_json(bundle_root / "resolved_config.json")
    rows = []
    audits = []
    for method, path in method_paths(out).items():
        source = (
            "FRESH_TUT_CAP050"
            if method == METHOD
            else "AUDITED_EXISTING_FRESH_CAP050_PROP_FAMILY"
        )
        metric, accounting = eval_mod.summarize_method(
            method,
            source,
            path,
            config["product_requirement_utility"],
            config["unified_ground_received_accounting"],
        )
        metric["J_HV_net"] = reqrun.hv_net(
            eval_mod,
            path,
            metric,
            config["product_requirement_utility"],
        )
        metric.update(
            reqrun.subgroup_metrics(
                eval_mod,
                path,
                metric,
                config["product_requirement_utility"],
            )
        )
        energy = reqrun.energy_metrics(method, path)
        metric.update(energy)
        metric["total_transmitted_bytes"] = (
            reqrun.dec(energy["transmitted_MB"]) * Decimal("1000000")
        )
        metric["public_accounting_passed"] = bool(
            accounting.get("passed", True)
        )
        metric["run_path"] = str(path)
        rows.append(metric)
        audits.append(
            {
                "method": method,
                "path": str(path),
                "passed": bool(accounting.get("passed", True)),
                "accounting_error": accounting.get("accounting_error", 0),
            }
        )
    return rows, audits


def select_tables(
    metrics: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    primary = [
        {
            "method": row["method"],
            "result_source": row["result_source"],
            "J_all_net": row["J_all_net"],
            "J_HV_net": row["J_HV_net"],
            "HV_closure_count": row["HV_closure_count"],
            "high_closure_count": row["high_closure_count"],
            "urgent_closure_count": row["urgent_closure_count"],
        }
        for row in metrics
    ]
    auxiliary = [
        {
            "method": row["method"],
            "HV_metadata_delivered_count": row[
                "HV_metadata_delivered_count"
            ],
            "HV_ROI_delivered_count": row["HV_ROI_delivered_count"],
            "HV_evidence_delivered_count": row[
                "HV_evidence_delivered_count"
            ],
            "all_task_closure_count": row["closure_count"],
            "all_task_closure_rate": row["closure_rate"],
            "successful_closure_mean_latency_s": row[
                "successful_closure_mean_latency_s"
            ],
            "successful_closure_p95_latency_s": row[
                "successful_closure_p95_latency_s"
            ],
            "HV_successful_closure_mean_latency_s": row[
                "HV_successful_closure_mean_latency_s"
            ],
            "HV_successful_closure_p95_latency_s": row[
                "HV_successful_closure_p95_latency_s"
            ],
            "controllable_compute_communication_energy_Wh": row[
                "controllable_compute_communication_energy_Wh"
            ],
            "transmitted_MB": row["transmitted_MB"],
            "total_transmitted_bytes": row["total_transmitted_bytes"],
            "peak_downlink_queue_MB": row["peak_downlink_queue_MB"],
            "mean_downlink_queue_MB": row["mean_downlink_queue_MB"],
        }
        for row in metrics
    ]
    safety = [
        {
            "method": row["method"],
            "soc_violation_count": row["soc_violation_count"],
            "thermal_violation_count": row["thermal_violation_count"],
            "minimum_soc": row["minimum_soc"],
            "maximum_thermal_state": row["maximum_thermal_state"],
            "maximum_DoD": row["maximum_DoD"],
            "max_energy_balance_residual_Wh": row[
                "max_energy_balance_residual_Wh"
            ],
        }
        for row in metrics
    ]
    return primary, auxiliary, safety


def write_table(out: Path, name: str, rows: list[dict[str, Any]]) -> None:
    guardrun.write_csv(out / "tables" / f"{name}.csv", rows)
    (out / "tables" / f"{name}.md").write_text(
        guardrun.markdown_table(rows) + "\n", encoding="utf-8"
    )


def input_audit(
    bundle_root: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    official_bundle = OFFICIAL_CAP035 / "config" / "frozen_runtime_bundle"
    file_rows = []
    for name in (
        "task_manifest.csv",
        "slot_trace.csv",
        "resolved_config.json",
        "bundle_contract.json",
    ):
        current = bundle_root / name
        official = official_bundle / name
        current_hash = reqrun.sha256_file(current)
        official_hash = reqrun.sha256_file(official)
        file_rows.append(
            {
                "file": name,
                "current_sha256": current_hash,
                "official_sha256": official_hash,
                "exact_match": current_hash == official_hash,
            }
        )
    capacity = identity["capacity_identity"]
    scenario = identity["scenario_identity"]
    passed = (
        all(row["exact_match"] for row in file_rows)
        and identity["source_identity_passed"]
        and capacity["runtime_capacity"] == CAPACITY
        and capacity["source_bundle_capacity"] == 0.35
        and capacity["contact_slots_unchanged"]
        and scenario["task_hash_match"]
        and scenario["runtime_value_mismatch_count"] == 0
        and scenario["frozen_power_identity_match"]
    )
    return {
        "passed": passed,
        "seed": 2026,
        "capacity": CAPACITY,
        "duration_s": 86400,
        "slot_s": 20,
        "task_hash": v25.EXPECTED_TASK_HASH,
        "method": METHOD,
        "source_identity_passed": identity["source_identity_passed"],
        "capacity_identity": capacity,
        "scenario_identity_summary": {
            "task_hash_match": scenario["task_hash_match"],
            "runtime_value_mismatch_count": scenario[
                "runtime_value_mismatch_count"
            ],
            "frozen_power_identity_match": scenario[
                "frozen_power_identity_match"
            ],
        },
        "bundle_files": file_rows,
    }


def comparison_deltas(
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in metrics}
    tut = by_method[METHOD]
    rows = []
    for reference in (
        reqrun.METHOD_PROP_OLD,
        reqrun.METHOD_PROP_NEW,
        guardrun.METHOD_GUARD,
    ):
        other = by_method[reference]
        rows.append(
            {
                "candidate": METHOD,
                "reference": reference,
                "delta_J_all_net": (
                    reqrun.dec(tut["J_all_net"])
                    - reqrun.dec(other["J_all_net"])
                ),
                "delta_J_HV_net": (
                    reqrun.dec(tut["J_HV_net"])
                    - reqrun.dec(other["J_HV_net"])
                ),
                "delta_HV_closure_count": (
                    reqrun.integer(tut["HV_closure_count"])
                    - reqrun.integer(other["HV_closure_count"])
                ),
                "delta_high_closure_count": (
                    reqrun.integer(tut["high_closure_count"])
                    - reqrun.integer(other["high_closure_count"])
                ),
                "delta_urgent_closure_count": (
                    reqrun.integer(tut["urgent_closure_count"])
                    - reqrun.integer(other["urgent_closure_count"])
                ),
            }
        )
    return rows


def final_report(
    out: Path,
    primary: list[dict[str, Any]],
    auxiliary: list[dict[str, Any]],
    safety: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    identity_audit: dict[str, Any],
) -> None:
    tut = next(row for row in primary if row["method"] == METHOD)
    tut_aux = next(row for row in auxiliary if row["method"] == METHOD)
    tut_safe = next(row for row in safety if row["method"] == METHOD)
    lines = [
        "# TUT capacity=0.50 fresh rerun",
        "",
        "Scope: frozen seed=2026 scenario, capacity=0.50, duration=86400 s, slot=20 s.",
        "",
        "This run started from t=0 with new state, queue, SOC, thermal state, and delivery ledger. It did not reuse the previously incomplete capacity=0.50 grid.",
        "",
        "## Primary metrics and same-environment PROP comparison",
        "",
        guardrun.markdown_table(primary),
        "",
        "## Auxiliary metrics",
        "",
        guardrun.markdown_table(auxiliary),
        "",
        "## Safety",
        "",
        guardrun.markdown_table(safety),
        "",
        "## TUT deltas",
        "",
        guardrun.markdown_table(deltas),
        "",
        "## TUT result",
        "",
        f"- J_all_net: {tut['J_all_net']}",
        f"- J_HV_net: {tut['J_HV_net']}",
        f"- HV_closure_count: {tut['HV_closure_count']}",
        f"- high / urgent closure: {tut['high_closure_count']} / {tut['urgent_closure_count']}",
        f"- all-task closure rate: {tut_aux['all_task_closure_rate']}",
        f"- controllable energy: {tut_aux['controllable_compute_communication_energy_Wh']} Wh",
        f"- transmitted: {tut_aux['transmitted_MB']} MB",
        f"- SOC / thermal violations: {tut_safe['soc_violation_count']} / {tut_safe['thermal_violation_count']}",
        f"- input and identity audit passed: {identity_audit['passed']}",
        "",
        "This is a single-seed diagnostic, not a statistical significance claim.",
    ]
    (out / "final_assessment.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parent_main(overwrite: bool) -> int:
    if OUT.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite without --overwrite: {OUT}"
            )
        resolved = OUT.resolve()
        if (
            resolved.parent != (ROOT / "outputs").resolve()
            or resolved.name != "_v25_bupt1_tut_seed2026_cap050_v1"
        ):
            raise RuntimeError(f"Unsafe output deletion target: {resolved}")
        shutil.rmtree(OUT)
    for name in ("audit", "config", "logs", "runs", "scenario", "tables"):
        (OUT / name).mkdir(parents=True, exist_ok=True)

    input_preflight, frozen = v25.preflight(OUT)
    reqrun.write_preflight_audits(
        OUT, v25, input_preflight, frozen
    )
    bundle_root = v25.prepare_runtime_bundle(OUT)

    started = utc_now()
    try:
        identity = run_tut(bundle_root, OUT / "runs" / "TUT")
        run_status = {
            "status": "PASS",
            "started_utc": started,
            "finished_utc": utc_now(),
            "method": METHOD,
            "fresh_state": True,
        }
    except Exception as exc:
        run_status = {
            "status": "FAIL",
            "started_utc": started,
            "finished_utc": utc_now(),
            "method": METHOD,
            "error": repr(exc),
        }
        guardrun.write_json(OUT / "audit" / "run_status.json", run_status)
        guardrun.write_json(
            OUT / "completion_status.json",
            {"status": "TUT_RUNTIME_FAILED"},
        )
        raise
    guardrun.write_json(OUT / "audit" / "run_status.json", run_status)

    identity_audit = input_audit(bundle_root, identity)
    guardrun.write_json(
        OUT / "audit" / "scenario_method_identity_audit.json",
        identity_audit,
    )
    if not identity_audit["passed"]:
        guardrun.write_json(
            OUT / "completion_status.json",
            {"status": "SCENARIO_OR_METHOD_IDENTITY_MISMATCH"},
        )
        return 2

    metrics, accounting = evaluate(OUT, bundle_root)
    primary, auxiliary, safety = select_tables(metrics)
    deltas = comparison_deltas(metrics)
    write_table(OUT, "primary_metrics", primary)
    write_table(OUT, "auxiliary_metrics", auxiliary)
    write_table(OUT, "safety_metrics", safety)
    write_table(OUT, "tut_vs_cap050_prop_deltas", deltas)
    guardrun.write_csv(OUT / "tables" / "all_metrics.csv", metrics)
    accounting_passed = all(row["passed"] for row in accounting)
    guardrun.write_json(
        OUT / "audit" / "public_accounting_audit.json",
        {"passed": accounting_passed, "rows": accounting},
    )
    safety_passed = all(
        reqrun.integer(row["soc_violation_count"]) == 0
        and reqrun.integer(row["thermal_violation_count"]) == 0
        for row in safety
    )
    guardrun.write_json(
        OUT / "audit" / "safety_audit.json",
        {"passed": safety_passed, "rows": safety},
    )
    final_report(
        OUT,
        primary,
        auxiliary,
        safety,
        deltas,
        identity_audit,
    )
    passed = identity_audit["passed"] and accounting_passed and safety_passed
    final_status = (
        "TUT_CAP050_FRESH_RERUN_COMPLETE"
        if passed
        else "TUT_CAP050_AUDIT_FAILED"
    )
    guardrun.write_json(
        OUT / "completion_status.json",
        {
            "status": final_status,
            "method": METHOD,
            "capacity": CAPACITY,
            "seed": 2026,
            "output": str(OUT),
        },
    )
    print(guardrun.markdown_table(primary))
    print(f"OUTPUT_DIR={OUT}")
    print(f"FINAL_STATUS={final_status}")
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return parent_main(args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
