from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_hv_urgent_residual_protection_v1 as guardrun
import scripts.run_prop_reqalign_seed2026_v1 as reqrun


OUT = (
    ROOT
    / "outputs"
    / "_v25_bupt1_corrected_reqalign_guard_seed2026_cap035_v1"
)
OFFICIAL = ROOT / "outputs" / "_v25_bupt1_power_true_rerun_seed2026_v1"
CAPACITY = 0.35
SLOTS = 4320

METHOD_OLD = reqrun.METHOD_PROP_OLD
METHOD_CORRECTED = reqrun.METHOD_PROP_NEW
METHOD_GUARD = guardrun.METHOD_GUARD
BASELINES = (
    "B1-COMMON-THERMAL",
    "OFFLOAD-EFT-THERMAL-ALIGNED",
    "TUT-UOI-PRODUCT-LITE-v2",
    "MYOPIC-FROZEN-CALIBRATED",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def launch_child(
    label: str,
    bundle_root: Path,
    output: Path,
    log_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--label",
        label,
        "--bundle-root",
        str(bundle_root),
        "--output",
        str(output),
    ]
    started = utc_now()
    process = subprocess.run(
        command,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}_stdout.txt").write_text(
        process.stdout, encoding="utf-8"
    )
    (log_dir / f"{label}_stderr.txt").write_text(
        process.stderr, encoding="utf-8"
    )
    return {
        "label": label,
        "status": (
            "PASS"
            if process.returncode == 0
            and (output / "method_completed.json").exists()
            else "FAIL"
        ),
        "returncode": process.returncode,
        "started_utc": started,
        "finished_utc": utc_now(),
        "output": str(output),
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-6000:],
    }


def child_main(args: argparse.Namespace) -> int:
    method_map = {
        "CORRECTED_REQALIGN": (True, False, METHOD_CORRECTED),
        "HV_GUARD": (True, True, METHOD_GUARD),
    }
    if args.label not in method_map:
        raise ValueError(args.label)
    reqalign_enabled, guard_enabled, method = method_map[args.label]
    guardrun.CAPACITY = CAPACITY
    guardrun.run_prop_variant(
        reqrun.load_v25(),
        Path(args.bundle_root),
        Path(args.output),
        reqalign_enabled=reqalign_enabled,
        guard_enabled=guard_enabled,
        method=method,
        slots=SLOTS,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "method": method,
                "capacity": CAPACITY,
                "fresh_state": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


def method_paths(out: Path) -> dict[str, Path]:
    return {
        METHOD_OLD: OFFICIAL / "formal" / METHOD_OLD,
        METHOD_CORRECTED: out / "runs" / "CORRECTED_REQALIGN",
        METHOD_GUARD: out / "runs" / "HV_GUARD",
        **{
            method: OFFICIAL / "formal" / method
            for method in BASELINES
        },
    }


def evaluate_all(
    out: Path,
    bundle_root: Path,
    v25: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_mod = v25.load_eval_module(out)
    config = reqrun.read_json(bundle_root / "resolved_config.json")
    metrics: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for method, path in method_paths(out).items():
        result_source = (
            "FRESH_CAP035_CURRENT_CODE"
            if method in {METHOD_CORRECTED, METHOD_GUARD}
            else "AUDITED_OFFICIAL_CAP035_ANCHOR"
        )
        metric, accounting = eval_mod.summarize_method(
            method,
            result_source,
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
        metrics.append(metric)
        audits.append(
            {
                "method": method,
                "path": str(path),
                "result_source": result_source,
                "passed": bool(accounting.get("passed", True)),
                "accounting_error": accounting.get("accounting_error", 0),
            }
        )
    return metrics, audits


def selected_tables(
    metrics: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
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
    high_urgent = [
        {
            "method": row["method"],
            "high_closure_count": row["high_closure_count"],
            "urgent_closure_count": row["urgent_closure_count"],
            "high_gross_utility": row["high_gross_utility"],
            "urgent_gross_utility": row["urgent_gross_utility"],
            "high_net_utility": row["high_net_utility"],
            "urgent_net_utility": row["urgent_net_utility"],
            "urgent_metadata_delivered_count": row[
                "urgent_metadata_delivered_count"
            ],
            "urgent_ROI_delivered_count": row[
                "urgent_ROI_delivered_count"
            ],
            "urgent_evidence_delivered_count": row[
                "urgent_evidence_delivered_count"
            ],
        }
        for row in metrics
    ]
    return primary, auxiliary, safety, high_urgent


def delta_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in metrics}
    rows: list[dict[str, Any]] = []
    for candidate in (METHOD_CORRECTED, METHOD_GUARD):
        for reference in (METHOD_OLD, *BASELINES):
            current = by_method[candidate]
            other = by_method[reference]
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "delta_J_all_net": (
                        reqrun.dec(current["J_all_net"])
                        - reqrun.dec(other["J_all_net"])
                    ),
                    "delta_J_HV_net": (
                        reqrun.dec(current["J_HV_net"])
                        - reqrun.dec(other["J_HV_net"])
                    ),
                    "delta_HV_closure_count": (
                        reqrun.integer(current["HV_closure_count"])
                        - reqrun.integer(other["HV_closure_count"])
                    ),
                    "delta_high_closure_count": (
                        reqrun.integer(current["high_closure_count"])
                        - reqrun.integer(other["high_closure_count"])
                    ),
                    "delta_urgent_closure_count": (
                        reqrun.integer(current["urgent_closure_count"])
                        - reqrun.integer(other["urgent_closure_count"])
                    ),
                    "delta_all_task_closure_count": (
                        reqrun.integer(current["closure_count"])
                        - reqrun.integer(other["closure_count"])
                    ),
                    "delta_energy_Wh": (
                        reqrun.dec(
                            current[
                                "controllable_compute_communication_energy_Wh"
                            ]
                        )
                        - reqrun.dec(
                            other[
                                "controllable_compute_communication_energy_Wh"
                            ]
                        )
                    ),
                    "delta_transmitted_MB": (
                        reqrun.dec(current["transmitted_MB"])
                        - reqrun.dec(other["transmitted_MB"])
                    ),
                }
            )
    return rows


def rank_rows(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rankings: dict[str, dict[str, int]] = {}
    for field in ("J_all_net", "J_HV_net", "HV_closure_count"):
        ordered = sorted(
            primary,
            key=lambda row: (
                reqrun.dec(row[field]),
                row["method"],
            ),
            reverse=True,
        )
        rankings[field] = {
            row["method"]: index + 1 for index, row in enumerate(ordered)
        }
    return [
        {
            "method": row["method"],
            "J_all_net_rank": rankings["J_all_net"][row["method"]],
            "J_HV_net_rank": rankings["J_HV_net"][row["method"]],
            "HV_closure_count_rank": rankings["HV_closure_count"][
                row["method"]
            ],
        }
        for row in primary
    ]


def write_table(out: Path, name: str, rows: list[dict[str, Any]]) -> None:
    guardrun.write_csv(out / "tables" / f"{name}.csv", rows)
    (out / "tables" / f"{name}.md").write_text(
        guardrun.markdown_table(rows) + "\n",
        encoding="utf-8",
    )


def input_identity_audit(
    out: Path,
    bundle_root: Path,
) -> dict[str, Any]:
    official_bundle = OFFICIAL / "config" / "frozen_runtime_bundle"
    files = (
        "task_manifest.csv",
        "slot_trace.csv",
        "resolved_config.json",
        "bundle_contract.json",
    )
    file_rows = []
    for name in files:
        current = bundle_root / name
        official = official_bundle / name
        file_rows.append(
            {
                "file": name,
                "current_sha256": reqrun.sha256_file(current),
                "official_sha256": reqrun.sha256_file(official),
                "exact_match": (
                    reqrun.sha256_file(current)
                    == reqrun.sha256_file(official)
                ),
            }
        )
    identity_rows = []
    for label in ("CORRECTED_REQALIGN", "HV_GUARD"):
        identity = reqrun.read_json(
            out / "runs" / label / "native_identity.json"
        )
        status = reqrun.read_json(
            out / "runs" / label / "method_run_status.json"
        )
        identity_rows.append(
            {
                "label": label,
                "method": identity["method"],
                "runtime_capacity": identity["capacity_identity"][
                    "runtime_capacity"
                ],
                "task_hash_match": identity["scenario_identity"][
                    "task_hash_match"
                ],
                "runtime_value_mismatch_count": identity[
                    "scenario_identity"
                ]["runtime_value_mismatch_count"],
                "fresh_state": status["fresh_state"],
                "reused_action_trace": status["reused_action_trace"],
                "reused_queue_trace": status["reused_queue_trace"],
                "reused_delivery_ledger": status[
                    "reused_delivery_ledger"
                ],
            }
        )
    official_status = reqrun.read_json(OFFICIAL / "final_status.json")
    passed = (
        all(row["exact_match"] for row in file_rows)
        and all(
            row["runtime_capacity"] == CAPACITY
            and row["task_hash_match"]
            and row["runtime_value_mismatch_count"] == 0
            and row["fresh_state"]
            and not row["reused_action_trace"]
            and not row["reused_queue_trace"]
            and not row["reused_delivery_ledger"]
            for row in identity_rows
        )
        and official_status["status"]
        == "BUPT1_FROZEN_POWER_TRUE_RERUN_FIVE_METHOD_COMPLETE"
    )
    return {
        "passed": passed,
        "capacity": CAPACITY,
        "seed": 2026,
        "duration_s": 86400,
        "slot_s": 20,
        "task_hash": reqrun.load_v25().EXPECTED_TASK_HASH,
        "bundle_files": file_rows,
        "new_run_identities": identity_rows,
        "official_anchor_status": official_status["status"],
    }


def final_report(
    out: Path,
    primary: list[dict[str, Any]],
    auxiliary: list[dict[str, Any]],
    safety: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    by_method = {row["method"]: row for row in primary}
    best_all = max(primary, key=lambda row: reqrun.dec(row["J_all_net"]))
    best_hv = max(primary, key=lambda row: reqrun.dec(row["J_HV_net"]))
    best_count = max(
        primary, key=lambda row: reqrun.dec(row["HV_closure_count"])
    )
    lines = [
        "# Capacity 0.35 corrected REQALIGN / Guard comparison",
        "",
        "Scope: seed=2026, capacity=0.35, duration=86400 s, slot=20 s.",
        "",
        "## Which methods were missing",
        "",
        "- Old PROP already had an audited formal capacity=0.35 run.",
        "- The historical pre-correction REQALIGN had run at 0.35, but it is not the corrected implementation.",
        "- Corrected REQALIGN enabled and HV Guard enabled had not run formally at 0.35; both were freshly run here.",
        "",
        "The old PROP and four baselines are read-only audited anchors from the same frozen bundle. The two current methods were initialized from t=0 in separate processes and did not reuse actions, queues, or delivery ledgers.",
        "",
        "## Primary metrics",
        "",
        guardrun.markdown_table(primary),
        "",
        "## Requested auxiliary metrics",
        "",
        guardrun.markdown_table(auxiliary),
        "",
        "## Safety",
        "",
        guardrun.markdown_table(safety),
        "",
        "## Current methods versus anchors",
        "",
        guardrun.markdown_table(deltas),
        "",
        "## Direct findings",
        "",
        f"- Best J_all_net: {best_all['method']} = {best_all['J_all_net']}.",
        f"- Best J_HV_net: {best_hv['method']} = {best_hv['J_HV_net']}.",
        f"- Best HV_closure_count: {best_count['method']} = {best_count['HV_closure_count']}.",
        f"- Corrected REQALIGN primary: J_all={by_method[METHOD_CORRECTED]['J_all_net']}, J_HV={by_method[METHOD_CORRECTED]['J_HV_net']}, HV closures={by_method[METHOD_CORRECTED]['HV_closure_count']}.",
        f"- Guard primary: J_all={by_method[METHOD_GUARD]['J_all_net']}, J_HV={by_method[METHOD_GUARD]['J_HV_net']}, HV closures={by_method[METHOD_GUARD]['HV_closure_count']}.",
        f"- Input/method identity audit passed: {identity['passed']}.",
        "",
        "These are single-seed mechanism diagnostics, not evidence of statistical significance.",
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
        expected_parent = (ROOT / "outputs").resolve()
        if (
            resolved.parent != expected_parent
            or resolved.name
            != "_v25_bupt1_corrected_reqalign_guard_seed2026_cap035_v1"
        ):
            raise RuntimeError(f"Unsafe output deletion target: {resolved}")
        shutil.rmtree(OUT)
    for name in ("audit", "config", "logs", "runs", "scenario", "tables"):
        (OUT / name).mkdir(parents=True, exist_ok=True)

    if not guardrun.run_tests(OUT):
        guardrun.write_json(
            OUT / "completion_status.json",
            {"status": "UNIT_TEST_FAILED"},
        )
        return 2

    v25 = reqrun.load_v25()
    input_audit, frozen = v25.preflight(OUT)
    reqrun.write_preflight_audits(OUT, v25, input_audit, frozen)
    bundle_root = v25.prepare_runtime_bundle(OUT)

    run_status = [
        launch_child(
            "CORRECTED_REQALIGN",
            bundle_root,
            OUT / "runs" / "CORRECTED_REQALIGN",
            OUT / "logs",
        ),
        launch_child(
            "HV_GUARD",
            bundle_root,
            OUT / "runs" / "HV_GUARD",
            OUT / "logs",
        ),
    ]
    guardrun.write_json(OUT / "audit" / "run_status.json", run_status)
    if not all(row["status"] == "PASS" for row in run_status):
        guardrun.write_json(
            OUT / "completion_status.json",
            {"status": "CURRENT_METHOD_RUNTIME_FAILED"},
        )
        return 2

    identity = input_identity_audit(OUT, bundle_root)
    guardrun.write_json(
        OUT / "audit" / "scenario_and_fresh_run_identity_audit.json",
        identity,
    )
    if not identity["passed"]:
        guardrun.write_json(
            OUT / "completion_status.json",
            {"status": "SCENARIO_PAIRING_MISMATCH"},
        )
        return 2

    metrics, accounting = evaluate_all(OUT, bundle_root, v25)
    primary, auxiliary, safety, high_urgent = selected_tables(metrics)
    deltas = delta_rows(metrics)
    rankings = rank_rows(primary)
    write_table(OUT, "primary_metrics", primary)
    write_table(OUT, "requested_auxiliary_metrics", auxiliary)
    write_table(OUT, "safety_metrics", safety)
    write_table(OUT, "high_urgent_breakdown", high_urgent)
    write_table(OUT, "current_methods_vs_anchors_deltas", deltas)
    write_table(OUT, "primary_metric_ranks", rankings)
    guardrun.write_csv(OUT / "tables" / "all_metrics.csv", metrics)
    guardrun.write_json(
        OUT / "audit" / "public_accounting_audit.json",
        {
            "passed": all(row["passed"] for row in accounting),
            "rows": accounting,
        },
    )
    no_specific = guardrun.no_task_specific_logic_audit()
    guardrun.write_json(
        OUT / "audit" / "no_task_specific_logic_audit.json",
        no_specific,
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
    final_report(OUT, primary, auxiliary, safety, deltas, identity)
    passed = (
        identity["passed"]
        and all(row["passed"] for row in accounting)
        and no_specific["passed"]
        and safety_passed
    )
    guardrun.write_json(
        OUT / "completion_status.json",
        {
            "status": (
                "CAP035_CORRECTED_REQALIGN_GUARD_COMPARISON_COMPLETE"
                if passed
                else "COMPARISON_AUDIT_FAILED"
            ),
            "capacity": CAPACITY,
            "seed": 2026,
            "fresh_methods": [METHOD_CORRECTED, METHOD_GUARD],
            "audited_anchor_methods": [METHOD_OLD, *BASELINES],
            "output": str(OUT),
        },
    )
    print(guardrun.markdown_table(primary))
    print(f"OUTPUT_DIR={OUT}")
    print(
        "FINAL_STATUS="
        + (
            "CAP035_CORRECTED_REQALIGN_GUARD_COMPARISON_COMPLETE"
            if passed
            else "COMPARISON_AUDIT_FAILED"
        )
    )
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--bundle-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.child_run:
        return child_main(args)
    return parent_main(args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
