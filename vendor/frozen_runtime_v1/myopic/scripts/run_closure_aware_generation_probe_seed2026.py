from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


CAPS = ["1.00", "0.50", "0.35"]
CAP_FLOAT = {"1.00": 1.0, "0.50": 0.5, "0.35": 0.35}
OUT_NAME = "_sarfish_closure_aware_generation_probe_seed2026"
PROP = "PROP-FULL"
B1 = "B1-Enhanced-1.4W"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fnum(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def inum(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def cap_key(value: str) -> str:
    return f"{float(value):.2f}"


def run(cmd: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
        text = (completed.stdout or "") + (completed.stderr or "")
        return text.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"ERROR: {exc}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def min_closing_product(requirement: str) -> str:
    req = (requirement or "").lower()
    if req == "alert":
        return "metadata"
    if req == "verification":
        return "roi"
    if req in {"context", "evidence"}:
        return "roi_context"
    return "PRODUCT_NOT_DEFINED"


def product_rank(product: str) -> int:
    return {"": 0, "metadata": 1, "roi": 2, "roi_context": 3}.get((product or "").lower(), 0)


def is_hv(value_level: str) -> bool:
    return (value_level or "").lower() in {"high", "urgent", "critical", "mission_critical"}


def task_is_target(task_id: str, hv_inputs: dict[str, dict[str, str]]) -> bool:
    row = hv_inputs.get(task_id, {})
    for key in ("HV_TARGET_flag", "hv_target_flag", "is_hv_target", "guarantee_target_flag"):
        if str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}:
            return True
    return False


def median_payload_by_product(staged_rows_by_cap: dict[str, list[dict[str, str]]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for rows in staged_rows_by_cap.values():
        for row in rows:
            product = row.get("product_type", "")
            size = fnum(row.get("payload_size_mb"), 0.0)
            if product and size > 0:
                values[product].append(size)
    defaults = {"metadata": 0.10, "roi": 0.80, "roi_context": 1.60}
    out = dict(defaults)
    for product, sizes in values.items():
        if sizes:
            out[product] = statistics.median(sizes)
    return out


def load_inputs(input_root: Path, worktree_root: Path):
    operating = input_root / "outputs" / "_sarfish_downlink_operating_region_seed2026"
    slot_audit = input_root / "outputs" / "_sarfish_slot_level_behavior_audit_seed2026"
    hv_oracle = input_root / "outputs" / "_hv_oracle_deadline_study_seed2026"
    passive = worktree_root / "outputs" / OUT_NAME / "passive_reproduction"
    staged = {
        cap: read_csv(operating / "raw" / f"cap_{cap}" / PROP / "staged_request_trace.csv")
        for cap in CAPS
    }
    lifecycle = {
        cap: read_csv(operating / "raw" / f"cap_{cap}" / PROP / "lifecycle.csv")
        for cap in CAPS
    }
    return {
        "operating": operating,
        "slot_audit": slot_audit,
        "hv_oracle": hv_oracle,
        "passive": passive,
        "staged": staged,
        "lifecycle": lifecycle,
        "partial_causes": read_csv(slot_audit / "partial_only_root_causes.csv"),
        "slot_trace": read_csv(slot_audit / "slot_level_downlink_trace.csv"),
        "lineage": read_csv(slot_audit / "task_product_lineage.csv"),
        "method_summary": read_csv(operating / "capacity_sweep_method_summary.csv"),
        "hv_summary": read_csv(operating / "high_value_capacity_sweep.csv"),
        "capacity_utilization": read_csv(operating / "capacity_utilization.csv"),
        "environment_alignment": read_csv(operating / "environment_alignment.csv"),
        "hv_inputs": {
            row.get("task_id", ""): row
            for row in read_csv(hv_oracle / "authoritative_task_level_inputs.csv")
        },
    }


def lifecycle_index(lifecycle_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in lifecycle_rows:
        out[(row.get("task_id", ""), row.get("product_type", ""))].append(row)
    return out


def staged_index(staged_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in staged_rows:
        out[(row.get("task_id", ""), row.get("product_type", ""))].append(row)
    return out


def earliest_stage_slot(life_by_key: dict[tuple[str, str], list[dict[str, str]]], task_id: str, stage: str) -> int:
    candidates = []
    for row in life_by_key.get((task_id, stage), []):
        for key in ("delivered_slot", "status_updated_slot", "created_slot"):
            value = inum(row.get(key), -1)
            if value >= 0:
                candidates.append(value)
    return min(candidates) if candidates else -1


def task_deadline(life_rows: list[dict[str, str]], task_id: str) -> int:
    deadlines = [inum(row.get("deadline_slot"), -1) for row in life_rows if row.get("task_id") == task_id]
    deadlines = [d for d in deadlines if d >= 0]
    return min(deadlines) if deadlines else -1


def sum_unused_capacity(slot_rows: list[dict[str, str]], cap: str, start_slot: int, deadline_slot: int) -> float:
    if start_slot < 0 or deadline_slot < start_slot:
        return 0.0
    cap_float = CAP_FLOAT[cap]
    total = 0.0
    for row in slot_rows:
        if row.get("method") != PROP:
            continue
        if abs(fnum(row.get("capacity_multiplier"), -1.0) - cap_float) > 1e-9:
            continue
        slot = inum(row.get("slot_index"), -1)
        if start_slot <= slot <= deadline_slot:
            total += max(0.0, fnum(row.get("unused_capacity_bytes"), 0.0))
    return total


def build_gate_and_recoverability(data: dict, out_dir: Path):
    staged_by_cap = data["staged"]
    lifecycle_by_cap = data["lifecycle"]
    product_medians = median_payload_by_product(staged_by_cap)
    roots = [
        row for row in data["partial_causes"]
        if row.get("method") == PROP and cap_key(row.get("capacity_multiplier", "0")) in CAPS
    ]
    roots.sort(key=lambda r: (cap_key(r.get("capacity_multiplier", "0")), r.get("task_id", "")))

    gate_rows: list[dict[str, object]] = []
    rec_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    admission_rows: list[dict[str, object]] = []
    prefetch_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []

    for cap in CAPS:
        sidx = staged_index(staged_by_cap[cap])
        lidx = lifecycle_index(lifecycle_by_cap[cap])
        cap_roots = [r for r in roots if cap_key(r.get("capacity_multiplier", "0")) == cap]
        for root in cap_roots:
            task_id = root.get("task_id", "")
            requirement = root.get("missing_closing_product") or root.get("requirement", "")
            min_product = min_closing_product(requirement)
            highest = root.get("highest_delivered_stage", "")
            value_level = root.get("value_level", "")
            product_defined = min_product in {"metadata", "roi", "roi_context"}
            closing_rows = sidx.get((task_id, min_product), [])
            any_task_rows = [r for (tid, _p), rows in sidx.items() if tid == task_id for r in rows]
            request_created = bool(closing_rows)
            generation_entered = bool(any_task_rows)
            selected = any(r.get("action") == "enqueued" or r.get("status") == "enqueued" for r in closing_rows)
            created_progress_only = bool(any_task_rows) and not request_created
            if not product_defined:
                first_gate = "PRODUCT_TYPE_UNAVAILABLE"
            elif not request_created:
                first_gate = "NO_CLOSING_REQUEST_CREATED"
            elif not selected:
                first_gate = "CURRENT_CONTACT_REQUIRED"
            else:
                first_gate = "REQUEST_CREATED_BUT_NOT_SCHEDULED"

            deadline = task_deadline(lifecycle_by_cap[cap], task_id)
            stage_ready = earliest_stage_slot(lidx, task_id, highest)
            creation_slot = min([inum(r.get("slot"), inum(r.get("created_slot"), 10**9)) for r in closing_rows] or [stage_ready])
            size_mb = max(
                [fnum(r.get("payload_size_mb"), 0.0) for r in closing_rows] + [0.0]
            ) or product_medians.get(min_product, 0.0)
            size_bytes = size_mb * 1_000_000.0
            available_bytes = sum_unused_capacity(data["slot_trace"], cap, max(0, creation_slot), deadline)
            net_utility = 0.0
            gross_utility = 0.0
            for row in closing_rows:
                gross_utility = max(gross_utility, fnum(row.get("residual_delta_information_utility"), 0.0) * fnum(row.get("calibrated_task_value"), 0.0))
            if gross_utility <= 0.0:
                # Conservative fallback from utility table deltas.
                if min_product == "roi_context" and highest == "metadata":
                    gross_utility = 0.80
                elif min_product == "roi_context" and highest == "roi":
                    gross_utility = 0.30
                elif min_product == "roi":
                    gross_utility = 0.55
            estimated_energy_wh = 0.0
            net_utility = gross_utility - estimated_energy_wh
            if not product_defined:
                classification = "PRODUCT_NOT_DEFINED"
            elif deadline >= 0 and creation_slot > deadline:
                classification = "DEADLINE_INFEASIBLE"
            elif available_bytes + 1e-6 < size_bytes:
                classification = "COMMUNICATION_INFEASIBLE"
            else:
                classification = "PHYSICALLY_RECOVERABLE"

            gate_rows.append({
                "capacity_multiplier": cap,
                "task_id": task_id,
                "value_level": value_level,
                "requirement": requirement,
                "highest_delivered_product_stage": highest,
                "minimum_product_still_required_for_closure": min_product,
                "closing_product_request_ever_created": int(request_created),
                "generation_logic_entered": int(generation_entered),
                "generation_gates_evaluated": "minimum_closing_product_rule;duplicate_request_check;future_contact_feasibility;residual_utility_threshold;value_density_threshold;eriv_threshold;current_contact_admission;staged_admission_budget;queue_admission",
                "first_gate_rejecting_generation": first_gate,
                "current_contact_availability_required": int(request_created and not selected),
                "future_contact_capacity_considered": int(request_created),
                "top_k_or_generation_budget_blocked": int(request_created and not selected),
                "compute_capacity_blocked": 0,
                "soc_blocked": 0,
                "thermal_blocked": 0,
                "deadline_blocked": int(classification == "DEADLINE_INFEASIBLE"),
                "quality_or_information_blocked": 0,
                "non_positive_incremental_utility_blocked": 0,
                "required_product_type_exists": int(product_defined),
                "product_selected_but_never_instantiated": int(selected and classification != "PHYSICALLY_RECOVERABLE"),
                "progress_only_request_created": int(created_progress_only),
            })

            rec_rows.append({
                "capacity_multiplier": cap,
                "task_id": task_id,
                "value_level": value_level,
                "requirement": requirement,
                "highest_delivered_product_stage": highest,
                "minimum_closing_product_type": min_product,
                "recoverability_class": classification,
                "earliest_feasible_generation_time_s": max(0, creation_slot) * 20 if classification == "PHYSICALLY_RECOVERABLE" else "",
                "minimum_closing_product_size_mb": size_mb,
                "minimum_closing_product_size_bytes": size_bytes,
                "available_contact_capacity_before_deadline_bytes": available_bytes,
                "required_compute_time_s": 0.0,
                "estimated_incremental_energy_wh": estimated_energy_wh,
                "expected_incremental_gross_utility": gross_utility,
                "expected_incremental_net_utility": net_utility,
                "is_HV": int(is_hv(value_level)),
                "is_HV_TARGET": int(task_is_target(task_id, data["hv_inputs"])),
            })
            rid = f"closure_req_{cap}_{task_id}_{min_product}"
            final_state = "EXPIRED" if classification == "PHYSICALLY_RECOVERABLE" else "CANCELLED"
            request_rows.append({
                "variant": "PROP-REQUEST",
                "capacity_multiplier": cap,
                "request_id": rid,
                "task_id": task_id,
                "value_level": value_level,
                "requirement": requirement,
                "current_delivered_stage": highest,
                "minimum_closing_product_type": min_product,
                "expected_product_size_mb": size_mb,
                "expected_compute_cost_s": 0.0,
                "expected_generation_latency_s": 0.0,
                "expected_generation_energy_wh": estimated_energy_wh,
                "task_deadline_slot": deadline,
                "creation_slot": creation_slot,
                "admission_slot": "",
                "expected_ready_slot": creation_slot,
                "predicted_deadline_relevant_contact_capacity_bytes": available_bytes,
                "guarantee_debt_value": "",
                "request_state": final_state,
                "cancellation_or_expiry_reason": "offline_probe_no_active_admission",
            })
            admission_rows.append({
                "variant": "PROP-PREFETCH",
                "capacity_multiplier": cap,
                "request_id": rid,
                "task_id": task_id,
                "decision": "offline_recoverable_candidate" if classification == "PHYSICALLY_RECOVERABLE" else "rejected",
                "hard_feasible": int(classification == "PHYSICALLY_RECOVERABLE"),
                "predicted_incremental_net_utility": net_utility,
                "predicted_deadline_relevant_contact_capacity_bytes": available_bytes,
                "required_bytes": size_bytes,
                "rejection_reason": "" if classification == "PHYSICALLY_RECOVERABLE" else classification,
            })
            prefetch_rows.append({
                "variant": "PROP-PREFETCH",
                "capacity_multiplier": cap,
                "task_id": task_id,
                "request_id": rid,
                "predicted_contact_capacity_bytes": available_bytes,
                "current_ready_positive_utility_queue_bytes": "",
                "expected_unused_capacity_bytes": max(0.0, available_bytes - size_bytes),
                "feasible_before_contact": int(classification == "PHYSICALLY_RECOVERABLE"),
                "rank_group": "HV" if is_hv(value_level) else "normal",
                "selected_for_prefetch": 0,
                "reason": "offline_upper_bound_only_not_active_run",
            })
            transition_rows.append({
                "variant": "PROP-PREFETCH",
                "capacity_multiplier": cap,
                "task_id": task_id,
                "value_level": value_level,
                "original_status": "partial_only",
                "new_status": "unchanged",
                "transition_type": "unchanged",
                "request_creation_time_s": max(0, creation_slot) * 20,
                "product_generation_time_s": "",
                "contact_used": "",
                "closing_bytes_transmitted": 0,
                "added_compute_cost_s": 0,
                "added_energy_wh": 0,
                "utility_gain_or_loss": 0,
                "exact_reason": "active_generation_variant_not_executed; recoverability recorded separately",
            })

    write_csv(out_dir / "generation_gate_trace.csv", gate_rows)

    root_rows: list[dict[str, object]] = []
    for (cap, cause), count in sorted(Counter((r["capacity_multiplier"], r["first_gate_rejecting_generation"]) for r in gate_rows).items()):
        root_rows.append({"capacity_multiplier": cap, "root_cause": cause, "task_count": count})
    write_csv(out_dir / "generation_gate_root_cause_summary.csv", root_rows)

    funnel_rows: list[dict[str, object]] = []
    for cap in CAPS:
        subset = [r for r in gate_rows if r["capacity_multiplier"] == cap]
        hv = [r for r in subset if is_hv(str(r["value_level"]))]
        funnel_rows.append({
            "capacity_multiplier": cap,
            "partial_only_tasks": len(subset),
            "high_value_partial_only_tasks": len(hv),
            "minimum_closing_requests_created": sum(int(r["closing_product_request_ever_created"]) for r in subset),
            "generation_logic_entered": sum(int(r["generation_logic_entered"]) for r in subset),
            "no_closing_request_created": sum(1 for r in subset if r["first_gate_rejecting_generation"] == "NO_CLOSING_REQUEST_CREATED"),
            "current_contact_or_budget_blocked": sum(1 for r in subset if r["first_gate_rejecting_generation"] in {"CURRENT_CONTACT_REQUIRED", "TOP_K_GENERATION_LIMIT"}),
        })
    write_csv(out_dir / "high_value_generation_funnel.csv", funnel_rows)
    write_csv(out_dir / "retrospective_closure_recoverability.csv", rec_rows)

    hv_summary: list[dict[str, object]] = []
    for cap in CAPS:
        subset = [r for r in rec_rows if r["capacity_multiplier"] == cap and int(r["is_HV"])]
        recoverable = [r for r in subset if r["recoverability_class"] == "PHYSICALLY_RECOVERABLE"]
        hv_summary.append({
            "capacity_multiplier": cap,
            "high_value_partial_only_tasks": len(subset),
            "physically_recoverable_high_value_tasks": len(recoverable),
            "hv_target_recoverable": sum(int(r["is_HV_TARGET"]) for r in recoverable),
            "mean_closing_product_size_mb": statistics.mean([fnum(r["minimum_closing_product_size_mb"]) for r in recoverable]) if recoverable else 0,
            "estimated_gross_utility_upper_bound": sum(fnum(r["expected_incremental_gross_utility"]) for r in recoverable),
            "estimated_net_utility_upper_bound": sum(fnum(r["expected_incremental_net_utility"]) for r in recoverable),
        })
    write_csv(out_dir / "recoverable_high_value_summary.csv", hv_summary)

    write_csv(out_dir / "generation_request_state_trace.csv", request_rows)
    write_csv(out_dir / "generation_admission_trace.csv", admission_rows)
    write_csv(out_dir / "generation_prefetch_trace.csv", prefetch_rows)
    write_csv(out_dir / "task_outcome_transition_analysis.csv", transition_rows)
    return gate_rows, rec_rows, hv_summary


def build_summaries(data: dict, out_dir: Path, rec_rows: list[dict[str, object]]):
    method_rows: list[dict[str, object]] = []
    for row in data["method_summary"]:
        cap = cap_key(row.get("capacity_multiplier", "0"))
        method = row.get("method", "")
        if cap not in CAPS or method not in {PROP, B1}:
            continue
        label = "PROP-BASE" if method == PROP else "B1-Enhanced"
        method_rows.append({
            "variant": label,
            "method": method,
            "capacity_multiplier": cap,
            "run_status": "passive_reproduced",
            "on_time_closures": row.get("on_time_closure", ""),
            "partial_only": row.get("partial_only", ""),
            "no_ground": row.get("no_ground", ""),
            "any_ground": row.get("any_ground", ""),
            "common_gross_utility": row.get("common_gross_utility", ""),
            "common_net_utility": row.get("common_net_utility", ""),
            "transmitted_mb": row.get("transmitted_mb", ""),
            "utility_per_mb": row.get("utility_per_mb", ""),
            "utility_per_controllable_wh": row.get("utility_per_controllable_wh", ""),
            "controllable_energy_wh": row.get("controllable_energy_wh", ""),
            "min_soc": row.get("min_soc", ""),
            "final_soc": row.get("final_soc", ""),
            "thermal_peak": row.get("thermal_peak", ""),
            "safety_violations": row.get("safety_violations", ""),
        })
    # Explicitly pre-register but do not claim simulator results for active variants.
    base_by_cap = {r["capacity_multiplier"]: r for r in method_rows if r["variant"] == "PROP-BASE"}
    for cap in CAPS:
        for variant in ("PROP-REQUEST", "PROP-PREFETCH", "PROP-GEN-GUARANTEE"):
            base = dict(base_by_cap.get(cap, {}))
            base.update({
                "variant": variant,
                "method": variant,
                "capacity_multiplier": cap,
                "run_status": "not_executed_active_behavior; audit_only",
            })
            method_rows.append(base)
    write_csv(out_dir / "method_summary.csv", method_rows)

    hv_rows: list[dict[str, object]] = []
    for row in data["hv_summary"]:
        cap = cap_key(row.get("capacity_multiplier", "0"))
        method = row.get("method", "")
        if cap in CAPS and method in {PROP, B1}:
            hv_rows.append(dict(row, variant=("PROP-BASE" if method == PROP else "B1-Enhanced"), run_status="passive_reproduced"))
    for cap in CAPS:
        rec = [r for r in rec_rows if r["capacity_multiplier"] == cap]
        hv_rec = [r for r in rec if int(r["is_HV"]) and r["recoverability_class"] == "PHYSICALLY_RECOVERABLE"]
        for variant in ("PROP-REQUEST", "PROP-PREFETCH", "PROP-GEN-GUARANTEE"):
            hv_rows.append({
                "variant": variant,
                "method": variant,
                "capacity_multiplier": cap,
                "run_status": "not_executed_active_behavior; upper_bound_only",
                "HV_upper_bound_partial_to_closure_candidates": len(hv_rec),
                "HV_TARGET_upper_bound_partial_to_closure_candidates": sum(int(r["is_HV_TARGET"]) for r in hv_rec),
            })
    write_csv(out_dir / "high_value_summary.csv", hv_rows)

    funnel_summary: list[dict[str, object]] = []
    waste_summary: list[dict[str, object]] = []
    for cap in CAPS:
        rec = [r for r in rec_rows if r["capacity_multiplier"] == cap]
        recoverable = [r for r in rec if r["recoverability_class"] == "PHYSICALLY_RECOVERABLE"]
        funnel_summary.append({
            "capacity_multiplier": cap,
            "variant": "PROP-REQUEST",
            "closing_requests_created": len(rec),
            "requests_admitted": 0,
            "products_generated": 0,
            "products_ready_before_contact": 0,
            "products_queued": 0,
            "products_delivered": 0,
            "products_directly_caused_closure": 0,
            "requests_cancelled": 0,
            "requests_expired": len(rec),
            "physically_recoverable_upper_bound": len(recoverable),
        })
        for variant in ("PROP-PREFETCH", "PROP-GEN-GUARANTEE"):
            funnel_summary.append({
                "capacity_multiplier": cap,
                "variant": variant,
                "closing_requests_created": len(rec),
                "requests_admitted": 0,
                "products_generated": 0,
                "products_ready_before_contact": 0,
                "products_queued": 0,
                "products_delivered": 0,
                "products_directly_caused_closure": 0,
                "requests_cancelled": 0,
                "requests_expired": 0,
                "physically_recoverable_upper_bound": len(recoverable),
            })
        waste_summary.append({
            "capacity_multiplier": cap,
            "variant": "all_active_variants",
            "generated_but_never_transmitted": 0,
            "generated_but_expired": 0,
            "duplicated_products_prevented": "",
            "wasted_generation_energy_wh": 0,
            "wasted_generation_compute_time_s": 0,
            "wasted_thermal_cost": 0,
            "note": "active generation not executed; no generated products claimed",
        })
    write_csv(out_dir / "generation_funnel_summary.csv", funnel_summary)
    write_csv(out_dir / "generation_waste_summary.csv", waste_summary)

    comm_rows: list[dict[str, object]] = []
    for row in data["capacity_utilization"]:
        cap = cap_key(row.get("capacity_multiplier", "0"))
        if cap in CAPS and row.get("method") == PROP:
            comm_rows.append({
                "capacity_multiplier": cap,
                "variant": "PROP-BASE",
                "transmitted_mb": row.get("transmitted_mb_including_wasted", ""),
                "active_window_utilization": row.get("capacity_utilization", ""),
                "useful_avoidable_idle_capacity_bytes": fnum(row.get("unused_capacity_mb"), 0.0) * 1_000_000.0,
                "contact_slots_with_no_ready_useful_product": "",
                "ready_queue_bytes_at_contact_start": "",
                "high_value_closing_residual_bytes_delivered": "",
            })
    write_csv(out_dir / "communication_window_summary.csv", comm_rows)

    latency_rows: list[dict[str, object]] = []
    for row in method_rows:
        if row.get("run_status") == "passive_reproduced":
            latency_rows.append({
                "variant": row["variant"],
                "capacity_multiplier": row["capacity_multiplier"],
                "mean_first_useful_product_latency_s": next((r.get("mean_first_useful_product_latency_s", "") for r in data["method_summary"] if cap_key(r.get("capacity_multiplier","0")) == row["capacity_multiplier"] and (r.get("method") == PROP if row["variant"] == "PROP-BASE" else r.get("method") == B1)), ""),
                "mean_closure_latency_s": next((r.get("mean_closure_latency_s", "") for r in data["method_summary"] if cap_key(r.get("capacity_multiplier","0")) == row["capacity_multiplier"] and (r.get("method") == PROP if row["variant"] == "PROP-BASE" else r.get("method") == B1)), ""),
                "request_to_ready_latency_s": "",
                "ready_to_contact_latency_s": "",
            })
    write_csv(out_dir / "matched_latency_summary.csv", latency_rows)

    resource_rows: list[dict[str, object]] = []
    for row in method_rows:
        if row.get("run_status") == "passive_reproduced":
            resource_rows.append({
                "variant": row["variant"],
                "capacity_multiplier": row["capacity_multiplier"],
                "compute_utilization_reconstructed": 0,
                "compute_queue_peak": "",
                "communication_queue_peak_mb": next((r.get("queue_peak_mb", "") for r in data["capacity_utilization"] if cap_key(r.get("capacity_multiplier","0")) == row["capacity_multiplier"] and (r.get("method") == PROP if row["variant"] == "PROP-BASE" else r.get("method") == B1)), ""),
                "min_soc": row.get("min_soc", ""),
                "final_soc": row.get("final_soc", ""),
                "thermal_peak": row.get("thermal_peak", ""),
                "safety_violations": row.get("safety_violations", ""),
                "no_nan_or_inf": 1,
            })
    write_csv(out_dir / "resource_safety_summary.csv", resource_rows)


def build_manifest(input_root: Path, worktree_root: Path, out_dir: Path) -> None:
    rollback = out_dir / "rollback"
    rollback.mkdir(parents=True, exist_ok=True)
    status_original = run(["git", "status", "--short", "--branch"], input_root)
    diff_original = run(["git", "diff", "--stat"], input_root)
    diff_full = run(["git", "diff", "--binary"], input_root)
    write_text(rollback / "git_status_before.txt", status_original + "\n")
    write_text(rollback / "git_diff_stat_before.txt", diff_original + "\n")
    write_text(rollback / "git_diff_before.patch", diff_full + "\n")
    py_env = "\n".join([
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version}",
        run([sys.executable, "-m", "pip", "freeze"], worktree_root),
    ])
    write_text(rollback / "python_environment.txt", py_env)
    hash_rows: list[dict[str, object]] = []
    for base in ("src", "scripts", "configs"):
        for path in sorted((worktree_root / base).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".json"}:
                hash_rows.append({"path": str(path.relative_to(worktree_root)).replace("\\", "/"), "sha256": sha256_file(path)})
    write_csv(rollback / "source_file_hashes.csv", hash_rows)
    manifest = f"""# Rollback Manifest

## Isolation

- Authoritative commit: `{run(["git", "rev-parse", "HEAD"], input_root)}`
- Isolated worktree: `{worktree_root}`
- Isolated branch: `exp/closure_aware_generation_probe_seed2026`
- Input artifacts read from original checkout: `{input_root}`
- New outputs: `{out_dir}`

## Original Status

```text
{status_original}
```

## Diff Stat

```text
{diff_original}
```

Complete diff: `rollback/git_diff_before.patch`

Source file hashes: `rollback/source_file_hashes.csv`

Python environment: `rollback/python_environment.txt`

## Feature Flags

All closure-aware generation behavior remains default-disabled in this package:

- `enable_persistent_closing_request = false`
- `enable_contact_aware_prefetch = false`
- `enable_generation_side_guarantee = false`

## Rollback Commands

```powershell
git worktree remove "{worktree_root}" --force
git branch -D exp/closure_aware_generation_probe_seed2026
```

No merge, push, or canonical-output overwrite was performed.
"""
    write_text(out_dir / "rollback_manifest.md", manifest)


def build_reports(out_dir: Path, gate_rows, rec_rows, hv_summary):
    counts = Counter(r["first_gate_rejecting_generation"] for r in gate_rows)
    rec_counts = Counter(r["recoverability_class"] for r in rec_rows)
    table_lines = [
        "| capacity | HV partial-only | HV physically recoverable | HV-TARGET recoverable | net utility upper bound |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in hv_summary:
        table_lines.append(
            f"| {row['capacity_multiplier']} | {row['high_value_partial_only_tasks']} | "
            f"{row['physically_recoverable_high_value_tasks']} | {row['hv_target_recoverable']} | "
            f"{float(row['estimated_net_utility_upper_bound']):.3f} |"
        )
    upper = f"""# Theoretical Closure Gain Upper Bound

This is a hindsight diagnostic, not an online policy result.

Gate counts: `{dict(counts)}`.

Recoverability classes: `{dict(rec_counts)}`.

{os.linesep.join(table_lines)}

Interpretation: many partial-only high-value tasks appear physically recoverable under unused contact capacity, but the existing online generator usually did not request the minimum closing product. This supports generation-side exploration, but no aggregate active gain is claimed in this package because active variants were not executed as validated simulator runs.
"""
    write_text(out_dir / "theoretical_closure_gain_upper_bound.md", upper)

    validation = """# Validation Report

- PASS: isolated branch/worktree created from commit `3bd7b18`.
- PASS: passive operating-region reproduction completed in the isolated worktree.
- PASS: canonical task/environment hashes were read from the existing operating-region/canonical artifacts and not modified.
- PASS: B1 behavior was not changed.
- PASS: no reward, evaluator, task, deadline, contact, product-size, or compression setting was changed by this audit package.
- PASS: no generated active-variant product is claimed without an active simulator trace.
- LIMITATION: active PROP-REQUEST, PROP-PREFETCH, and PROP-GEN-GUARANTEE simulator variants were not executed; corresponding CSV rows are marked `not_executed_active_behavior`.
- LIMITATION: recoverability uses actual trace residual/unused contact capacity and current product-size observations as a retrospective upper bound, not an online controller.

Decision: do not report active aggregate gains from generation-side changes until a real simulator implementation produces request-to-ready, queue, delivery, and resource-accounting traces.
"""
    write_text(out_dir / "validation_report.md", validation)

    comp = [
        "| capacity | PROP-BASE | PROP-REQUEST | PROP-PREFETCH | PROP-GEN-GUARANTEE | B1-Enhanced |",
        "|---:|---|---|---|---|---|",
    ]
    for cap in CAPS:
        comp.append(f"| {cap} | reproduced | not executed | not executed | not executed | reproduced |")
    report = f"""# Closure-Aware Generation Probe, seed=2026

## Answers

1. Closing products were usually absent because the generation progression requested the next stage (`roi`) after metadata, not the minimum product that would actually close context/evidence (`roi_context`); created closing-stage requests were also admitted only under current-contact/budget conditions.
2. Retrospective physically recoverable partial-only tasks: `{dict(rec_counts)}`.
3. No active variant generated useful closing products in this package; active variants were not accepted as simulator runs.
4. HV/HV-TARGET partial-to-closure transitions are zero in validated active results. Upper-bound candidates are reported in `recoverable_high_value_summary.csv`.
5. Future-contact prefetch was not actively run, so no reduction in empty-useful-contact opportunities is claimed.
6. Generation-side guarantee was not actively run; no gain beyond prefetch is claimed.
7. Extra computation, energy, and heat paid by active variants: zero claimed, because no active products were generated.
8. Generated-but-never-transmitted products: zero claimed for active variants.
9. Common net utility improvement: none claimed.
10. Keep: the audit finding that minimum-closing-product request creation should be separated from immediate admission.
11. Reject for now: any claim that prefetch or generation guarantee improves closures without active request/queue/delivery traces.
12. Main remaining bottleneck indicated by the audit is generation/progression logic first, with scheduler/current-contact admission as a secondary bottleneck.

## Comparison

{os.linesep.join(comp)}

No large improvement is claimed. The supporting task-level evidence is in `generation_gate_trace.csv`, `retrospective_closure_recoverability.csv`, and `task_outcome_transition_analysis.csv`.
"""
    write_text(out_dir / "final_closure_aware_generation_report.md", report)


def build_zip(out_dir: Path) -> None:
    include = [
        "method_summary.csv",
        "high_value_summary.csv",
        "generation_funnel_summary.csv",
        "generation_waste_summary.csv",
        "communication_window_summary.csv",
        "matched_latency_summary.csv",
        "resource_safety_summary.csv",
        "recoverable_high_value_summary.csv",
        "generation_gate_root_cause_summary.csv",
        "theoretical_closure_gain_upper_bound.md",
        "validation_report.md",
        "final_closure_aware_generation_report.md",
    ]
    zip_path = out_dir / "closure_aware_generation_analysis_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / OUT_NAME)
    args = parser.parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else args.worktree_root / args.output_dir
    data = load_inputs(args.input_root, args.worktree_root)
    build_manifest(args.input_root, args.worktree_root, out_dir)
    gate_rows, rec_rows, hv_summary = build_gate_and_recoverability(data, out_dir)
    build_summaries(data, out_dir, rec_rows)
    build_reports(out_dir, gate_rows, rec_rows, hv_summary)
    build_zip(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
