"""Focused Phase-1.5 passive correction audit from existing seed=2026 outputs."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_hv_phase1_seed2026 import (
    B1_SOURCE,
    CORRECTED_SOURCE,
    OUT,
    PROP_SOURCE,
    _system_summaries,
    read_csv,
    write_csv,
)
from src.evaluation.hv_phase1 import is_hv_all, is_hv_target, matched_latency, summarize_group, validate_method_rows


OUTPUT_FILES = (
    "high_value_method_comparison_corrected.csv",
    "guarantee_target_comparison_corrected.csv",
    "staged_closure_task_audit.csv",
    "b1_quality_semantics_resolution.md",
    "b1_hv_requirement_breakdown.csv",
    "phase1_final_validation.md",
)


def _i(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _f(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def staged_closure_audit(outcomes: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join delivered staged lifecycle IDs to final task outcomes without changing runtime behavior."""

    by_task = {str(row["task_id"]): row for row in outcomes}
    delivered_staged = {
        str(row["task_id"]): row
        for row in ledger
        if row.get("current_status") == "delivered"
        and ("staged" in str(row.get("source", "")) or str(row.get("request_id", "")).startswith("staged_"))
    }
    rows: list[dict[str, Any]] = []
    for task_id in sorted(delivered_staged):
        outcome = by_task[task_id]
        if not _i(outcome.get("requirement_satisfied")):
            raise ValueError(f"delivered staged task is not a final requirement closure: {task_id}")
        events = [
            row for row in ledger
            if str(row.get("task_id")) == task_id and row.get("current_status") == "delivered"
        ]
        events.sort(key=lambda row: (_i(row.get("delivered_slot")), str(row.get("product_type"))))
        stages: list[str] = []
        for event in events:
            product = str(event.get("product_type", ""))
            if product and product not in stages:
                stages.append(product)
        staged_event = delivered_staged[task_id]
        value_level = str(outcome.get("value_level", ""))
        requirement = str(outcome.get("product_requirement", ""))
        in_hv = is_hv_all(outcome)
        in_target = is_hv_target(outcome)
        selected_action = "metadata_then_roi" if stages == ["metadata", "roi"] else "then".join(stages)
        rows.append({
            "task_id": task_id,
            "value_level": value_level,
            "runtime_staged_value_level": staged_event.get("value_level", ""),
            "product_requirement": requirement,
            "selected_action": selected_action,
            "selected_action_source": "passive reconstruction from delivered ordinary plus staged lifecycle products; raw joint-action ID not persisted",
            "delivered_stages": ";".join(stages),
            "closure_time_s": _f(outcome.get("requirement_closure_time_s")),
            "final_path": "staged_requirement_closure",
            "staged_request_id": staged_event.get("request_id", ""),
            "staged_product_type": staged_event.get("product_type", ""),
            "in_HV_ALL": int(in_hv),
            "in_HV_TARGET": int(in_target),
        })
    return rows


def b1_hv_breakdown(method_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, rows in method_rows.items():
        closures = [row for row in rows if is_hv_all(row) and _i(row.get("requirement_satisfied"))]
        for category in ("alert", "verification", "context", "evidence"):
            selected = [row for row in closures if str(row.get("product_requirement")) == category]
            output.append({"method": method, "dimension": "requirement", "category": category,
                           "closure_count": len(selected), "unique_task_count": len({row["task_id"] for row in selected}),
                           "on_time_closure_count": sum(_i(row.get("on_time_task_success")) for row in selected)})
        for category, repository_value in (("high", "high_value"), ("urgent", "emergency")):
            selected = [row for row in closures if str(row.get("value_level")) == repository_value]
            output.append({"method": method, "dimension": "value_level", "category": category,
                           "closure_count": len(selected), "unique_task_count": len({row["task_id"] for row in selected}),
                           "on_time_closure_count": sum(_i(row.get("on_time_task_success")) for row in selected)})
    return output


def _load_methods() -> dict[str, list[dict[str, Any]]]:
    prop_rows = read_csv(PROP_SOURCE / "task_level_ground_outcomes.csv")
    corrected = read_csv(CORRECTED_SOURCE / "task_level_ground_outcomes.csv")
    b1 = read_csv(OUT / "b1_task_level_ground_outcomes.csv")
    methods = {
        "PROP-FULL": [{**row, "method": "PROP-FULL"} for row in prop_rows if row["method"] == "PROP-FULL"],
        "PROP-NO-STAGE": [{**row, "method": "PROP-NO-STAGE"} for row in corrected if row["method"] == "proposed"],
        "B1-FIFO-1.4W": [row for row in b1 if row["method"] == "B1-FIFO-1.4W"],
        "B1-Enhanced-1.4W": [row for row in b1 if row["method"] == "B1-Enhanced-1.4W"],
    }
    for method, rows in methods.items():
        validate_method_rows(method, rows)
    return methods


def _group_rows(
    methods: dict[str, list[dict[str, Any]]],
    system: dict[str, dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    group: str,
    prop_staged: set[str],
) -> list[dict[str, Any]]:
    output = []
    for method, rows in methods.items():
        closure_ids = {row["task_id"] for row in rows if _i(row.get("requirement_satisfied"))}
        staged = closure_ids & prop_staged if method == "PROP-FULL" else set()
        direct = closure_ids - staged
        summary = summarize_group(rows, predicate, system_summary=system[method], direct_ids=direct, staged_ids=staged)
        if _i(summary["direct_closures"]) + _i(summary["staged_closures"]) != _i(summary["requirement_satisfied"]):
            raise AssertionError(f"{method}/{group}: direct + staged != total closure")
        output.append({"method": method, "group": group, **summary})
    return output


def _timing_rows(methods: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    definitions: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("matched_ground_received_tasks", lambda row: bool(_i(row.get("ground_completed")))),
        ("matched_requirement_closures", lambda row: bool(_i(row.get("requirement_satisfied")))),
        ("matched_HV_requirement_closures", lambda row: is_hv_all(row) and bool(_i(row.get("requirement_satisfied")))),
        ("matched_HV_TARGET_requirement_closures", lambda row: is_hv_target(row) and bool(_i(row.get("requirement_satisfied")))),
    ]
    output = []
    for left, right in (("PROP-FULL", "PROP-NO-STAGE"), ("PROP-FULL", "B1-Enhanced-1.4W")):
        for name, predicate in definitions:
            output.append({"left_method": left, "right_method": right, "task_set": name,
                           **matched_latency(methods[left], methods[right], predicate)})
    return output


def run(output_dir: Path = OUT) -> None:
    if not output_dir.exists():
        raise FileNotFoundError(f"Phase-1 checkpoint directory is missing: {output_dir}")
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Phase-1.5 outputs already exist and will not be overwritten: {existing}")

    methods = _load_methods()
    ledger = read_csv(PROP_SOURCE / "prop_full_lifecycle_ledger_trace.csv")
    staged_audit = staged_closure_audit(methods["PROP-FULL"], ledger)
    prop_staged = {row["task_id"] for row in staged_audit}
    if len(staged_audit) != 25:
        raise AssertionError(f"expected 25 staged closure task IDs, got {len(staged_audit)}")
    system = _system_summaries()
    hv = _group_rows(methods, system, is_hv_all, "HV-ALL", prop_staged)
    target = _group_rows(methods, system, is_hv_target, "HV-TARGET", prop_staged)
    breakdown = b1_hv_breakdown({key: value for key, value in methods.items() if key.startswith("B1-")})
    timing = _timing_rows(methods)

    checks: list[tuple[str, bool, str]] = []
    checks.append(("unique_task_ids_only", all(len(rows) == len({row["task_id"] for row in rows}) == 370 for rows in methods.values()), "370 unique IDs per method"))
    for method, rows in methods.items():
        categories = {
            "on_time": sum(_i(row.get("on_time_task_success")) for row in rows),
            "late": sum(_i(row.get("requirement_satisfied")) and not _i(row.get("on_time_task_success")) for row in rows),
            "partial": sum(not _i(row.get("requirement_satisfied")) and _i(row.get("ground_completed")) for row in rows),
            "no_ground": sum(not _i(row.get("requirement_satisfied")) and not _i(row.get("ground_completed")) for row in rows),
        }
        checks.append((f"category_total:{method}", sum(categories.values()) == 370, json.dumps(categories, sort_keys=True)))
    checks.append(("no_stale_17", sum(_i(row["requirement_satisfied"]) for row in methods["PROP-NO-STAGE"]) == 72, "PROP-NO-STAGE=72"))
    checks.append(("all_25_staged_accounted", len(prop_staged) == len(staged_audit) == 25, "25 unique delivered staged closure IDs"))
    checks.append(("staged_outside_HV", sum(_i(row["in_HV_ALL"]) for row in staged_audit) == 0, "all 25 are routine/verification"))
    checks.append(("correct_group_denominators", all(_i(row["efficiency_denominator_G_on_time_success"]) == _i(row["on_time_requirement_success"]) for row in hv + target), "G-specific success count"))
    checks.append(("direct_plus_staged_all_groups", all(_i(row["direct_closures"]) + _i(row["staged_closures"]) == _i(row["requirement_satisfied"]) for row in hv + target), "numeric attribution for every method/group"))
    checks.append(("b1_41_HV_all_target", all(sum(_i(row["requirement_satisfied"]) for row in methods[m] if is_hv_all(row)) == 41 and sum(_i(row["requirement_satisfied"]) for row in methods[m] if is_hv_target(row)) == 41 for m in ("B1-FIFO-1.4W", "B1-Enhanced-1.4W")), "41 unique HV closures; all context/evidence"))
    checks.append(("timing_counts", sum(_i(row["requirement_satisfied"]) for row in methods["PROP-FULL"]) == 132 and sum(_i(row["on_time_task_success"]) for row in methods["PROP-FULL"]) == 132 and sum(_i(row["requirement_satisfied"]) for row in methods["B1-Enhanced-1.4W"]) == 107 and sum(_i(row["on_time_task_success"]) for row in methods["B1-Enhanced-1.4W"]) == 82, "PROP=132/132; B1=107/82"))
    finite = all(math.isfinite(float(value)) for collection in (hv, target, staged_audit, breakdown, timing) for row in collection for value in row.values() if isinstance(value, (int, float)))
    checks.append(("no_nan_inf", finite, "all generated numeric values finite"))

    phase1_manifest = (OUT / "authoritative_run_manifest.yaml").read_text(encoding="utf-8")
    raw_paths = [
        PROP_SOURCE / "task_level_ground_outcomes.csv",
        PROP_SOURCE / "prop_full_lifecycle_ledger_trace.csv",
        CORRECTED_SOURCE / "task_level_ground_outcomes.csv",
        OUT / "b1_task_level_ground_outcomes.csv",
    ]
    before_hashes = {str(path): _sha256(path) for path in raw_paths}
    checks.append(("no_algorithm_behavior_changed", all(path.exists() for path in raw_paths) and "policy_behavior_modified: false" in phase1_manifest.lower(), "passive readers only; no simulator or policy invoked"))
    if not all(passed for _, passed, _ in checks):
        raise AssertionError([name for name, passed, _ in checks if not passed])

    write_csv(output_dir / OUTPUT_FILES[0], hv)
    write_csv(output_dir / OUTPUT_FILES[1], target)
    write_csv(output_dir / OUTPUT_FILES[2], staged_audit)
    write_csv(output_dir / OUTPUT_FILES[4], breakdown)

    quality_md = """# B1 Quality Semantics Resolution

Three variables must remain separate:

- **Reconstructed frame quality = 0.96.** This is `reconstructed_image_quality_proxy`, an SSIM-like proxy for the decoded compressed image.
- **Downstream ground-result quality = 0.88.** This is `ground_task_quality`, inherited from the ground `high_accuracy` model profile and used in common gross utility.
- **Requirement thresholds = 0.92–0.96.** Phase 1 compared reconstructed frame quality against values copied from the requirement-information utility target table: alert 0.95, verification 0.96, context 0.92, evidence 0.95.

The variable currently controlling B1 requirement closure is **reconstructed frame quality (0.96)**, not downstream ground-result quality (0.88). This preserves the authoritative B1 closure count of 107 because all completed main-run frames pass the adopted thresholds.

## Semantic finding

The 0.92–0.96 values are defined in `configs/product_requirement_utility.yaml` as information-sufficiency utility targets for delivered product types. They are not documented or calibrated as compression-quality acceptance thresholds. Therefore the current comparison cannot honestly confirm that they represent physical compression sufficiency. Reusing them as image-quality thresholds is a **semantic inconsistency**, although it does not change the current 107/82 B1 counts.

No threshold or result was changed in Phase 1.5. Before publication, a separately justified compression-quality threshold (or a task-level calibrated mapping) is required. Final inference quality 0.88 should continue to control utility unless the mission specification explicitly defines it as a closure gate.
"""
    (output_dir / OUTPUT_FILES[3]).write_text(quality_md, encoding="utf-8")

    timing_lines = [
        "| Left | Right | Set | n | Mean Δs | Median Δs | P90 Δs | P95 Δs | Faster | Equal | Slower |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in timing:
        timing_lines.append(
            f"| {row['left_method']} | {row['right_method']} | {row['task_set']} | {row['matched_task_count']} | "
            f"{_f(row['mean_paired_latency_difference_s']):.6f} | {_f(row['median_paired_latency_difference_s']):.6f} | "
            f"{_f(row['p90_paired_latency_difference_s']):.6f} | {_f(row['p95_paired_latency_difference_s']):.6f} | "
            f"{row['left_faster_count']} | {row['equal_count']} | {row['left_slower_count']} |"
        )
    validation = [
        "# Phase-1 Final Validation",
        "",
        "## Status",
        "",
        "All mechanical Phase-1.5 reconciliation checks pass. Phase 1 is **not semantically final for publication** because the B1 compression-threshold interpretation remains unresolved; see `b1_quality_semantics_resolution.md`.",
        "",
        "## Attribution resolution",
        "",
        "The HV and HV-TARGET staged count of zero is genuine: all 25 staged closure IDs are `routine/verification`, outside both groups. There is no missing join key or wrong direct/staged label. The proven evaluator defect was the reused HV-ALL denominator in the HV-TARGET efficiency columns; corrected files use each group G's own on-time-success count. Non-staged architectures now use numeric `direct=total, staged=0` attribution.",
        "",
        "## Timing definitions",
        "",
        "PROP-FULL total/on-time closures are 132/132. B1 total/on-time closures are 107/82. A matched set is the exact intersection of unique task IDs satisfying the named predicate in both methods. Paired difference is left latency minus right latency; negative means the left method is faster.",
        "",
        *timing_lines,
        "",
        "## Automatic checks",
        "",
        *[f"- {'PASS' if passed else 'FAIL'} `{name}`: {detail}" for name, passed, detail in checks],
        "",
        "## Input immutability",
        "",
        *[f"- `{Path(path).relative_to(ROOT)}`: `{digest}`" for path, digest in before_hashes.items()],
    ]
    (output_dir / OUTPUT_FILES[5]).write_text("\n".join(validation) + "\n", encoding="utf-8")
    after_hashes = {str(path): _sha256(path) for path in raw_paths}
    if after_hashes != before_hashes:
        raise AssertionError("historical/raw input hash changed during passive Phase-1.5 audit")


if __name__ == "__main__":
    run()
    print(OUT)
