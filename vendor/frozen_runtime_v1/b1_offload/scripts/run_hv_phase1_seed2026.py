"""Authoritative seed=2026 Phase-1 reconciliation; no oracle or deadline runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_yaml
from src.evaluation.hv_phase1 import (
    classify_outcome,
    evaluate_b1_frame,
    is_hv_all,
    is_hv_target,
    matched_latency,
    summarize_group,
    validate_method_rows,
)
from src.evaluation.unified_ground_received_accounting import score_task


OUT = ROOT / "outputs/_hv_oracle_deadline_study_seed2026"
PROP_SOURCE = ROOT / "outputs/_prop_full_vs_b1_unified_seed2026"
CORRECTED_SOURCE = ROOT / "outputs/_direct_task_requirement_mapping_audit_seed2026"
B1_SOURCE = ROOT / "outputs/_baseline_b1_enhanced_validation_v2_seed2026"

PHASE1_REQUIRED_FILES = (
    "authoritative_run_manifest.yaml",
    "authoritative_task_level_inputs.csv",
    "environment_alignment.csv",
    "accounting_validation_report.md",
    "b1_ground_processing_semantics_audit.md",
    "b1_task_level_ground_outcomes.csv",
    "high_value_method_comparison.csv",
    "guarantee_target_comparison.csv",
    "high_value_failure_reasons.csv",
    "high_value_matched_latency.csv",
    "closure_set_decomposition.csv",
    "phase1_checkpoint_report.md",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _i(value: Any) -> int:
    return int(round(_f(value)))


def _canonical_inputs(prop_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    rows = [row for row in prop_rows if row["method"] == "PROP-FULL"]
    if len(rows) != 370:
        raise ValueError(f"PROP-FULL source must contain 370 tasks, got {len(rows)}")
    return {
        row["task_id"]: {
            "task_id": row["task_id"],
            "arrival_time_s": _f(row["arrival_time_s"]),
            "capture_time_s": _f(row["capture_time_s"]),
            "deadline_time_s": _f(row["deadline_time_s"]),
            "value_level": row["value_level"],
            "effective_task_value": _f(row["effective_task_value"]),
            "product_requirement": row["product_requirement"],
            # The mission model defines all four downstream products from a captured full observation.
            # Keeping this explicit prevents receipt alone from becoming the closure predicate.
            "full_frame_information_capabilities": "alert;verification;context;evidence",
            "task_manifest_hash": row["task_manifest_hash"],
        }
        for row in rows
    }


def _rename_rows(rows: list[dict[str, str]], source_method: str, method: str, variant: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("method") != source_method:
            continue
        selected.append({**row, "method": method, "variant": variant})
    return selected


def _b1_rows(
    tasks: dict[str, dict[str, Any]], trace: list[dict[str, str]], method: str, variant: str, accounting_cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row["task_id"]: row for row in trace}
    if set(by_id) != set(tasks):
        raise ValueError(f"{method}: B1 trace task IDs do not match authoritative task table")
    outcomes: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        record = by_id[task_id]
        evaluation = evaluate_b1_frame(record, task)
        completed = int(evaluation.full_frame_received and evaluation.ground_processing_success)
        completion = _f(record.get("ground_result_available_time_s")) if completed else 0.0
        quality = _f(record.get("ground_task_quality")) if completed else 0.0
        bytes_delivered = _f(record.get("compressed_size_mb")) if evaluation.full_frame_received else 0.0
        base = {
            **task,
            "method": method,
            "variant": variant,
            "seed": 2026,
            "transmitted_product_type": "compressed_full_frame",
            "highest_ground_available_stage": "full_frame_result" if completed else "empty",
            "ground_completed": completed,
            "ground_result_available_time_s": completion if completed else "",
            "requirement_closure_time_s": completion if evaluation.requirement_satisfied else "",
            "on_time_ground_completed": int(completed and completion <= _f(task["deadline_time_s"]) + 1.0e-9),
            "requirement_satisfied": int(evaluation.requirement_satisfied),
            "on_time_task_success": int(evaluation.on_time_success),
            "partial_ground_delivery": int(completed and not evaluation.requirement_satisfied),
            "capture_to_ground_latency_s": _f(record.get("capture_to_ground_latency_s")) if completed else 0.0,
            "ground_model_quality": quality,
            "compression_quality_retention": evaluation.retained_compression_quality,
            "retained_compression_quality": evaluation.retained_compression_quality,
            "quality_threshold": evaluation.quality_threshold,
            "quality_pass": int(evaluation.quality_pass),
            "full_frame_generated": int(evaluation.full_frame_generated),
            "full_frame_received": int(evaluation.full_frame_received),
            "ground_processing_success": int(evaluation.ground_processing_success),
            "frame_contains_required_information": int(evaluation.frame_contains_required_information),
            "ground_information_sufficiency": 1.0 if evaluation.frame_contains_required_information and completed else 0.0,
            "ground_result_quality": quality,
            "partial_ground_gross_utility": 0.0,
            "bytes_delivered_mb": bytes_delivered,
            "failure_reason": evaluation.failure_reason,
            "expired": int(record.get("state") == "expired"),
            "dropped": 0,
            "terminal_backlog_flag": int(record.get("state") in {"compressed_queued", "transmitting"}),
        }
        outcomes.append(score_task(base, accounting_cfg))
        audit.append({
            "method": method,
            "task_id": task_id,
            "required_product": task["product_requirement"],
            "full_frame_compressed_generated": int(evaluation.full_frame_generated),
            "compressed_frame_size_mb": _f(record.get("compressed_size_mb")),
            "bytes_delivered_mb": bytes_delivered,
            "full_frame_received": int(evaluation.full_frame_received),
            "ground_processing_finished": int(evaluation.ground_processing_success),
            "completion_time_s": completion if completed else "",
            "retained_compression_quality": evaluation.retained_compression_quality,
            "quality_threshold": evaluation.quality_threshold,
            "quality_pass": int(evaluation.quality_pass),
            "frame_contains_required_information": int(evaluation.frame_contains_required_information),
            "requirement_satisfied": int(evaluation.requirement_satisfied),
            "on_time_success": int(evaluation.on_time_success),
            "final_failure_reason": evaluation.failure_reason,
        })
    return outcomes, audit


def _system_summaries() -> dict[str, dict[str, Any]]:
    current = read_csv(PROP_SOURCE / "method_level_unified_summary.csv")
    corrected = read_csv(CORRECTED_SOURCE / "method_level_unified_summary.csv")
    return {
        "PROP-FULL": next(row for row in current if row["method"] == "PROP-FULL"),
        "PROP-NO-STAGE": next(row for row in corrected if row["method"] == "proposed"),
        "B1-FIFO-1.4W": next(row for row in corrected if row["method"] == "b1_fifo"),
        "B1-Enhanced-1.4W": next(row for row in corrected if row["method"] == "b1_enhanced_1p4w"),
    }


def _failure_reason(row: dict[str, Any]) -> str:
    reason = str(row.get("failure_reason", "")).strip()
    if reason and reason not in {"none", "success"}:
        aliases = {
            "expired": "zero_feasible_contact_before_deadline",
            "blocked": "safety_blocked",
            "ground_receipt_not_requirement_closure": "lower_than_required_product",
        }
        return aliases.get(reason, reason)
    if _i(row.get("on_time_task_success")):
        return "success"
    if _i(row.get("requirement_satisfied")):
        return "ground_processing_late"
    if _i(row.get("ground_completed")):
        return "lower_than_required_product"
    if _i(row.get("expired")):
        return "zero_feasible_contact_before_deadline"
    if _i(row.get("terminal_backlog_flag")):
        return "deferred"
    return "no_ground_trace_evidence"


def _staged_task_ids() -> set[str]:
    ledger = read_csv(PROP_SOURCE / "prop_full_lifecycle_ledger_trace.csv")
    return {
        row["task_id"]
        for row in ledger
        if row.get("current_status") == "delivered" and ("staged" in row.get("source", "") or row.get("request_id", "").startswith("staged_"))
    }


def _matched_rows(method_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    comparisons = [("PROP-FULL", "PROP-NO-STAGE"), ("PROP-FULL", "B1-Enhanced-1.4W")]
    sets: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("matched_ground_received_tasks", lambda row: bool(_i(row.get("ground_completed")))),
        ("matched_requirement_closures", lambda row: bool(_i(row.get("requirement_satisfied")))),
        ("matched_HV_requirement_closures", lambda row: is_hv_all(row) and bool(_i(row.get("requirement_satisfied")))),
        ("matched_HV_TARGET_requirement_closures", lambda row: is_hv_target(row) and bool(_i(row.get("requirement_satisfied")))),
    ]
    output = []
    for left, right in comparisons:
        for set_name, predicate in sets:
            output.append({"left_method": left, "right_method": right, "paired_set": set_name, **matched_latency(method_rows[left], method_rows[right], predicate)})
        left_ids = {row["task_id"] for row in method_rows[left] if _i(row.get("requirement_satisfied"))}
        right_ids = {row["task_id"] for row in method_rows[right] if _i(row.get("requirement_satisfied"))}
        for label, ids, source in (("completed_only_by_left", left_ids - right_ids, method_rows[left]), ("completed_only_by_right", right_ids - left_ids, method_rows[right])):
            by_id = {row["task_id"]: row for row in source}
            lat = [_f(by_id[task_id].get("capture_to_ground_latency_s")) for task_id in ids]
            output.append({"left_method": left, "right_method": right, "paired_set": label, "matched_task_count": len(ids), "matched_task_ids": ";".join(sorted(ids)), "mean_paired_latency_difference_s": statistics.fmean(lat) if lat else 0.0, "median_paired_latency_difference_s": statistics.median(lat) if lat else 0.0})
    return output


def _finite_rows(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def verify_phase1_output(output_dir: Path) -> dict[str, Any]:
    missing = [name for name in PHASE1_REQUIRED_FILES if not (output_dir / name).exists()]
    result: dict[str, Any] = {"passed": not missing, "missing_files": missing}
    if missing:
        return result
    alignment = read_csv(output_dir / "environment_alignment.csv")
    b1 = read_csv(output_dir / "b1_task_level_ground_outcomes.csv")
    result["alignment_rows_passed"] = all(_i(row["aligned"]) == 1 for row in alignment)
    result["b1_rows"] = len(b1)
    result["passed"] = result["passed"] and result["alignment_rows_passed"] and len(b1) == 740
    return result


def run_phase1(output_dir: Path = OUT) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Phase-1 output exists and will not be overwritten: {output_dir}")
    temp = output_dir.with_name(output_dir.name + ".phase1_tmp")
    if temp.exists():
        raise FileExistsError(f"Temporary Phase-1 output exists: {temp}")
    temp.mkdir(parents=True)

    prop_all = read_csv(PROP_SOURCE / "task_level_ground_outcomes.csv")
    corrected_all = read_csv(CORRECTED_SOURCE / "task_level_ground_outcomes.csv")
    tasks = _canonical_inputs(prop_all)
    prop_full = _rename_rows(prop_all, "PROP-FULL", "PROP-FULL", "proposed_guarantee_aware_full")
    no_stage = _rename_rows(corrected_all, "proposed", "PROP-NO-STAGE", "corrected_historical_no_stage")
    accounting_cfg = load_yaml(ROOT / "configs/unified_ground_received_accounting.yaml")
    b1_fifo, fifo_audit = _b1_rows(tasks, read_csv(B1_SOURCE / "b1_fifo_paper_power_retention_1_00/per_task_delivery_trace.csv"), "B1-FIFO-1.4W", "b1_fifo_paper_power_retention_1_00", accounting_cfg)
    b1_enhanced, enhanced_audit = _b1_rows(tasks, read_csv(B1_SOURCE / "b1_enhanced_paper_power_retention_1_00/per_task_delivery_trace.csv"), "B1-Enhanced-1.4W", "b1_enhanced_paper_power_retention_1_00", accounting_cfg)
    methods = {"PROP-FULL": prop_full, "PROP-NO-STAGE": no_stage, "B1-FIFO-1.4W": b1_fifo, "B1-Enhanced-1.4W": b1_enhanced}
    for method, rows in methods.items():
        validate_method_rows(method, rows)

    canonical_ids = set(tasks)
    alignment = []
    for method, rows in methods.items():
        ids = {row["task_id"] for row in rows}
        fields_match = all(
            math.isclose(_f(row["arrival_time_s"]), _f(tasks[row["task_id"]]["arrival_time_s"]), abs_tol=1e-9)
            and math.isclose(_f(row["capture_time_s"]), _f(tasks[row["task_id"]]["capture_time_s"]), abs_tol=1e-9)
            and math.isclose(_f(row["deadline_time_s"]), _f(tasks[row["task_id"]]["deadline_time_s"]), abs_tol=1e-9)
            and row["product_requirement"] == tasks[row["task_id"]]["product_requirement"]
            and row["value_level"] == tasks[row["task_id"]]["value_level"]
            for row in rows
        )
        alignment.append({"method": method, "task_count": len(rows), "unique_task_count": len(ids), "task_ids_match": int(ids == canonical_ids), "task_fields_match": int(fields_match), "aligned": int(ids == canonical_ids and fields_match)})

    system = _system_summaries()
    staged_ids = _staged_task_ids()
    prop_closure_ids = {row["task_id"] for row in prop_full if _i(row["requirement_satisfied"])}
    staged_closure_ids = prop_closure_ids & staged_ids
    direct_closure_ids = prop_closure_ids - staged_closure_ids
    hv_rows = []
    target_rows = []
    for method, rows in methods.items():
        path_args = {"direct_ids": direct_closure_ids, "staged_ids": staged_closure_ids} if method == "PROP-FULL" else {}
        hv_rows.append({"method": method, "group": "HV-ALL", **summarize_group(rows, is_hv_all, system_summary=system[method], **path_args)})
        target_rows.append({"method": method, "group": "HV-TARGET", **summarize_group(rows, is_hv_target, system_summary=system[method], **path_args)})

    failure_rows = []
    for method, rows in methods.items():
        counts = Counter(_failure_reason(row) for row in rows if is_hv_all(row) and not _i(row.get("on_time_task_success")))
        for reason, count in sorted(counts.items()):
            failure_rows.append({"method": method, "group": "HV-ALL_not_on_time", "failure_reason": reason, "task_count": count})

    closure_rows = []
    for method, rows in methods.items():
        closure_ids = {row["task_id"] for row in rows if _i(row.get("requirement_satisfied"))}
        if method == "PROP-FULL":
            closure_rows.extend([
                {"method": method, "closure_path": "direct", "task_count": len(closure_ids & direct_closure_ids), "task_ids": ";".join(sorted(closure_ids & direct_closure_ids))},
                {"method": method, "closure_path": "staged", "task_count": len(closure_ids & staged_closure_ids), "task_ids": ";".join(sorted(closure_ids & staged_closure_ids))},
                {"method": method, "closure_path": "total", "task_count": len(closure_ids), "task_ids": ";".join(sorted(closure_ids))},
            ])
        else:
            closure_rows.append({"method": method, "closure_path": "architecture_native", "task_count": len(closure_ids), "task_ids": ";".join(sorted(closure_ids))})

    checks: list[tuple[str, bool, str]] = []
    checks.append(("identical_task_ids", all(row["aligned"] == 1 for row in alignment), "all four methods use the same 370 task IDs and immutable task fields"))
    checks.append(("no_duplicate_task_outcomes", all(len({row["task_id"] for row in rows}) == len(rows) for rows in methods.values()), "one outcome per method/task"))
    for method, rows in methods.items():
        categories = Counter(classify_outcome(row) for row in rows)
        checks.append((f"partition_equals_arrivals:{method}", sum(categories.values()) == 370, json.dumps(categories, sort_keys=True)))
        checks.append((f"on_time_implies_requirement:{method}", all(not _i(row.get("on_time_task_success")) or _i(row.get("requirement_satisfied")) for row in rows), "subset check"))
    checks.append(("b1_processed_frames_use_dominance_mapping", all("quality_threshold" in row and "frame_contains_required_information" in row for row in fifo_audit + enhanced_audit), "four predicates recorded"))
    checks.append(("b1_incomplete_frames_never_close", all(not _i(row["requirement_satisfied"]) for row in b1_fifo + b1_enhanced if not _i(row["ground_completed"])), "incomplete frames have zero closure"))
    checks.append(("prop_direct_plus_staged_equals_total", len(direct_closure_ids) + len(staged_closure_ids) == len(prop_closure_ids) == 132, f"{len(direct_closure_ids)}+{len(staged_closure_ids)}=132"))
    checks.append(("no_stale_17_closure_input", sum(_i(row["requirement_satisfied"]) for row in no_stage) == 72, "corrected source closure=72"))
    checks.append(("no_nan_or_infinity", all(_finite_rows(rows) for rows in methods.values()), "all in-memory numeric fields finite"))
    if not all(passed for _, passed, _ in checks):
        failed = [name for name, passed, _ in checks if not passed]
        raise AssertionError(f"Phase-1 mandatory checks failed: {failed}")

    input_paths = [
        PROP_SOURCE / "task_level_ground_outcomes.csv",
        PROP_SOURCE / "method_level_unified_summary.csv",
        PROP_SOURCE / "prop_full_lifecycle_ledger_trace.csv",
        CORRECTED_SOURCE / "task_level_ground_outcomes.csv",
        CORRECTED_SOURCE / "method_level_unified_summary.csv",
        B1_SOURCE / "b1_fifo_paper_power_retention_1_00/per_task_delivery_trace.csv",
        B1_SOURCE / "b1_enhanced_paper_power_retention_1_00/per_task_delivery_trace.csv",
        ROOT / "configs/unified_ground_received_accounting.yaml",
        ROOT / "configs/product_requirement_utility.yaml",
    ]
    manifest = {
        "phase": 1,
        "seed": 2026,
        "policy_behavior_modified": False,
        "historical_outputs_overwritten": False,
        "prop_no_stage_authoritative_closure": 72,
        "oracle_status": "not_started_phase_2",
        "deadline_sensitivity_status": "not_started_phase_3",
        "inputs": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)} for path in input_paths],
        "code_locations": [
            "src/evaluation/hv_phase1.py",
            "src/evaluation/task_outcome_adapter.py",
            "src/evaluation/unified_ground_received_accounting.py",
            "src/baselines/b1_enhanced.py",
            "scripts/run_hv_phase1_seed2026.py",
        ],
    }
    (temp / "authoritative_run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    write_csv(temp / "authoritative_task_level_inputs.csv", list(tasks.values()))
    write_csv(temp / "environment_alignment.csv", alignment)
    write_csv(temp / "b1_task_level_ground_outcomes.csv", b1_fifo + b1_enhanced)
    write_csv(temp / "high_value_method_comparison.csv", hv_rows)
    write_csv(temp / "guarantee_target_comparison.csv", target_rows)
    write_csv(temp / "high_value_failure_reasons.csv", failure_rows)
    write_csv(temp / "high_value_matched_latency.csv", _matched_rows(methods))
    write_csv(temp / "closure_set_decomposition.csv", closure_rows)

    check_lines = ["# Phase-1 Accounting Validation", "", *[f"- {'PASS' if passed else 'FAIL'} `{name}`: {detail}" for name, passed, detail in checks]]
    (temp / "accounting_validation_report.md").write_text("\n".join(check_lines) + "\n", encoding="utf-8")
    b1_closure = sum(_i(row["requirement_satisfied"]) for row in b1_enhanced)
    b1_audit = f"""# B1 Ground-Processing Semantics Audit

The corrected passive evaluator does **not** equate receipt alone with requirement closure. A B1 task closes only when the complete compressed frame is received, ground decode/inference finishes, the source-frame capability mapping includes the required result, and reconstructed compression quality meets the requirement threshold.

## Full-frame dominance mapping

| Requirement | Required ground result | Reconstructed-quality threshold | Source |
|---|---|---:|---|
| alert | ground alert from the captured full observation | 0.95 | requirement target in `configs/product_requirement_utility.yaml` |
| verification | ground verification from the captured full observation | 0.96 | requirement target in `configs/product_requirement_utility.yaml` |
| context | ground context result from the captured full observation | 0.92 | requirement target in `configs/product_requirement_utility.yaml` |
| evidence | ground evidence result from the captured full observation | 0.95 | requirement target in `configs/product_requirement_utility.yaml` |

The current mission task manifest does not persist a per-task “information absent from source frame” flag. Its declared full-observation semantics support all four downstream result classes; this capability is therefore recorded explicitly for every task rather than inferred from receipt. If future datasets add task-level source-content flags, they replace this mission-level mapping without changing the other gates.

Both FIFO and Enhanced use the identical evaluator. Incomplete frames receive no closure or utility. Ground inference latency is included in completion time. Main 1.4 W traces use reconstructed quality 0.96 and downstream ground quality 0.88. Enhanced requirement closures after applying all gates: **{b1_closure}**.

See `b1_task_level_ground_outcomes.csv` for all four predicates and final reasons.
"""
    (temp / "b1_ground_processing_semantics_audit.md").write_text(b1_audit, encoding="utf-8")

    hv_counts = {row["method"]: row["on_time_requirement_success"] for row in hv_rows}
    target_counts = {row["method"]: row["on_time_requirement_success"] for row in target_rows}
    checkpoint = f"""# Phase-1 Checkpoint, seed=2026

Phase 1 is limited to authoritative reconciliation, B1 ground semantics, HV/HV-TARGET comparison, matched latency, and consistency checks. No Oracle model was built or solved and no deadline sensitivity run was executed.

## Authoritative counts

- PROP-NO-STAGE requirement closures: 72 (corrected historical evaluator; stale 17-row result rejected).
- PROP-FULL requirement closures: 132.
- HV on-time closures: {json.dumps(hv_counts, sort_keys=True)}.
- HV-TARGET on-time closures: {json.dumps(target_counts, sort_keys=True)}.

## Exact Phase-2 resume point: offline Oracle

Inputs are frozen in `authoritative_run_manifest.yaml` with SHA-256 hashes. Start from `authoritative_task_level_inputs.csv`, the canonical task outcomes listed in the manifest, `configs/model_pool.yaml`, `configs/payload_profiles.yaml`, the contact/resource/solar trace configs resolved by `scripts/run_prop_full_vs_b1_unified_seed2026.py`, and B1 physical parameters in `configs/baseline_b1_enhanced.yaml`. Implement legal plan enumeration and solver code under `src/evaluation/hv_oracle_study/plans.py`, `oracle.py`, and `replay.py`, following `docs/superpowers/specs/2026-07-11-hv-oracle-deadline-study-design.md`. Solve total closures and HV-TARGET closures first; distinguish integer incumbent, proven optimum, and LP upper bound; replay independently. Do not use runtime policy feasibility denominators before this phase finishes.

## Exact Phase-3 resume point: deadline sensitivity

Reuse the authoritative task table and manifest hashes. Add deadline replacement only through `arrival + k * (original_deadline - arrival)` for k=1.0, 1.25, 1.5, 2.0. Run the four canonical methods without tuning. Add passive pre/post-filter B1 candidate logging at `ContactAwareCompressedGroundScheduler.rank/select` in `src/baselines/b1_enhanced.py`; preserve selection behavior with `tests/test_b1_enhanced_baseline.py`. Write sensitivity orchestration separately from `scripts/run_hv_phase1_seed2026.py`. If Enhanced never activates, report `none` and stop at 2.0.

## Phase-1 code and tests

- Evaluator and metrics: `src/evaluation/hv_phase1.py`
- Runner and verification contract: `scripts/run_hv_phase1_seed2026.py`
- Focused tests: `tests/test_hv_phase1.py`, `tests/test_hv_phase1_runner.py`
"""
    (temp / "phase1_checkpoint_report.md").write_text(checkpoint, encoding="utf-8")
    temp.replace(output_dir)
    verification = verify_phase1_output(output_dir)
    if not verification["passed"]:
        raise AssertionError(f"Phase-1 output verification failed: {verification}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        result = verify_phase1_output(args.verify_only)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result["passed"] else 1)
    run_phase1(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
