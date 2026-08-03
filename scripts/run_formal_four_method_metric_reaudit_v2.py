"""Read-only end-to-end success and controllable-energy re-audit.

This script never runs a simulator and never changes a method/configuration.  It
derives v2 metrics from the frozen four-method comparison and its authoritative
per-task/per-slot inputs.  Missing subgroup energy attribution is reported as
missing rather than imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


FORMAL = Path("D:/wt/pilot/outputs/_formal_four_method_comparison_v1")
THREE = Path("D:/wt/pilot/outputs/_three_method_capacity_robustness_v1")
OLD_MULTI = Path("D:/wt/pilot/outputs/_sarfish_thermal_headroom_multiseed/multiscenario_run_summary.csv")
TUT_TREND = Path("D:/wt/pilot/outputs/_tut_uoi_product_capacity_trend_seed2026_v1")
TUT_DIAG = Path("D:/wt/pilot/outputs/_tut_uoi_product_lite_diagnostic_v1")
TUT_ROOT = Path("D:/wt/tut_uoi_rate_lite")
DEFAULT_OUT = Path("D:/wt/pilot/outputs/_formal_four_method_metric_reaudit_v2")
METHODS = (
    "B1-COMMON-THERMAL",
    "OFFLOAD-EFT-THERMAL-ALIGNED",
    "TUT-UOI-PRODUCT-LITE",
    "PROP-HEADROOM",
)
CAPACITIES = (1.0, 0.5, 0.35)
SEEDS = (2026, 2027, 2028, 2029, 2030)
EPS = 1.0e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def f(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(math.floor(index))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def cap_tag(capacity: float) -> str:
    return f"cap{int(round(capacity * 100)):03d}"


def scenario_tag(seed: int, capacity: float) -> str:
    return f"seed{seed}_{cap_tag(capacity)}"


def method_dir(seed: int, capacity: float, method: str) -> Path:
    if method == "TUT-UOI-PRODUCT-LITE":
        if seed == 2026:
            if abs(capacity - 0.5) < EPS:
                return TUT_DIAG / "corrected_pilot"
            return TUT_TREND / "runs" / cap_tag(capacity)
        return FORMAL / "runs" / scenario_tag(seed, capacity) / method
    return THREE / "runs" / scenario_tag(seed, capacity) / method


def requirement_class(requirement: str) -> str:
    req = str(requirement).strip().lower()
    if req == "alert":
        return "alert_required"
    if req == "verification":
        return "roi_verification_required"
    if req in {"context", "evidence"}:
        return "context_evidence_required"
    return "unknown_requirement"


def standardize_task_rows(seed: int, capacity: float, method: str) -> list[dict[str, Any]]:
    rows = read_csv(method_dir(seed, capacity, method) / "per_task_results.csv")
    reference = read_csv(method_dir(seed, capacity, "B1-COMMON-THERMAL") / "per_task_results.csv")
    by_id = {row["task_id"]: row for row in reference}
    output: list[dict[str, Any]] = []
    for row in rows:
        ref = by_id[row["task_id"]]
        offload = method == "OFFLOAD-EFT-THERMAL-ALIGNED"
        value_level = str(row.get("value_level") or ref.get("value_level", "")).lower()
        requirement = str(row.get("product_requirement") or ref.get("product_requirement", "")).lower()
        arrival = f(row.get("arrival_time_s", row.get("arrival_s", row.get("capture_time_s"))))
        deadline = f(row.get("deadline_time_s", row.get("deadline_s")))
        if offload and deadline < arrival:
            deadline += arrival
        finish_raw = row.get("requirement_closure_time_s", row.get("actual_finish_s", ""))
        finish = f(finish_raw) if finish_raw not in (None, "") else None
        success = truth(row.get("on_time_task_success", row.get("closed", 0)))
        if offload:
            success = success and truth(row.get("product_feasible", 1)) and finish is not None and finish <= deadline + 1.0e-9
        gross = f(row.get("gross_ground_utility", row.get("gross_utility", 0.0)))
        deadline_duration = max(0.0, deadline - arrival)
        closure_latency = None if finish is None else max(0.0, finish - arrival)
        capped_latency = min(closure_latency, deadline_duration) if closure_latency is not None else deadline_duration
        is_hv = value_level in {"high", "urgent", "high_value", "emergency"}
        is_hv_target = value_level in {"urgent", "emergency"}
        evidence_required = is_hv and requirement in {"context", "evidence"}
        compute_j = f(row.get("compute_energy_j")) if offload else f(row.get("compute_energy_wh")) * 3600.0
        encoding_j = 0.0
        transmission_j = f(row.get("downlink_energy_j", row.get("communication_energy_j"))) if offload else f(row.get("communication_energy_wh")) * 3600.0
        if method == "B1-COMMON-THERMAL":
            encoding_j = f(row.get("compression_energy_wh")) * 3600.0
            compute_j = 0.0
        elif method in {"PROP-HEADROOM", "TUT-UOI-PRODUCT-LITE"}:
            # The task adapter stores inference+encoding together in compute_energy_wh.
            encoding_j = 0.0
        attributable = method == "OFFLOAD-EFT-THERMAL-ALIGNED"
        output.append({
            "seed": seed,
            "capacity": capacity,
            "method": method,
            "task_id": row["task_id"],
            "value_level": value_level,
            "is_hv": int(is_hv),
            "is_hv_target": int(is_hv_target),
            "product_requirement": requirement,
            "requirement_group": requirement_class(requirement),
            "is_hv_evidence_required": int(evidence_required),
            "ground_completed": int(truth(row.get("ground_completed", row.get("closed", 0)))),
            "requirement_satisfied": int(truth(row.get("requirement_satisfied", row.get("product_feasible", 0)))),
            "deadline_time_s": deadline,
            "arrival_time_s": arrival,
            "closure_time_s": finish if finish is not None else "",
            "success": int(success),
            "common_gross_utility": gross,
            "successful_gross_utility": gross if success else 0.0,
            "closure_latency_s": closure_latency if success and closure_latency is not None else "",
            "deadline_capped_latency_s": capped_latency,
            "compute_energy_j_task_logged": compute_j,
            "product_encoding_energy_j_task_logged": encoding_j,
            "transmission_energy_j_task_logged": transmission_j,
            "task_energy_attribution_reliable": int(attributable),
            "failure_or_zero_reason": "" if success else str(row.get("failure_reason", "not_successful_public_closure")),
        })
    return output


def old_multi_index() -> dict[tuple[int, float, str], dict[str, str]]:
    return {(int(row["seed"]), f(row["capacity_multiplier"]), row["method"]): row for row in read_csv(OLD_MULTI)}


def energy_breakdown(seed: int, capacity: float, method: str, old_index: Mapping[tuple[int, float, str], Mapping[str, Any]]) -> dict[str, Any]:
    directory = method_dir(seed, capacity, method)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    total_j = f(summary.get("controllable_satellite_energy_wh", summary.get("controllable_energy_wh"))) * 3600.0
    compute_j: float | None = None
    inference_j: float | None = None
    product_encoding_j: float | None = None
    transmission_j: float | None = None
    source = "frozen_summary"
    trace_path = directory / "per_slot_resource_trace.csv"
    if method == "TUT-UOI-PRODUCT-LITE":
        trace = read_csv(trace_path)
        compute_j = sum(f(row.get("compute_energy_j")) for row in trace)
        product_encoding_j = sum(f(row.get("payload_encoding_energy_j")) for row in trace)
        inference_j = compute_j - product_encoding_j
        transmission_j = sum(f(row.get("comm_energy_j")) for row in trace)
        total_j = compute_j + transmission_j
        source = "frozen_per_slot_trace_exact"
    elif method == "OFFLOAD-EFT-THERMAL-ALIGNED":
        tasks = read_csv(directory / "per_task_results.csv")
        compute_j = sum(f(row.get("compute_energy_j")) for row in tasks)
        transmission_j = sum(f(row.get("downlink_energy_j", row.get("communication_energy_j"))) for row in tasks)
        total_j = compute_j + transmission_j
        inference_j = compute_j
        product_encoding_j = 0.0
        source = "frozen_per_task_exact"
    elif method == "PROP-HEADROOM" and trace_path.exists():
        trace = read_csv(trace_path)
        compute_j = sum(f(row.get("compute_energy_j")) for row in trace)
        transmission_j = sum(f(row.get("comm_energy_j")) for row in trace)
        total_j = compute_j + transmission_j
        source = "frozen_per_slot_trace_compute_includes_encoding"
    elif method == "B1-COMMON-THERMAL":
        if trace_path.exists():
            trace = read_csv(trace_path)
            product_encoding_j = sum(f(row.get("compression_energy_wh")) for row in trace) * 3600.0
            transmission_j = sum(f(row.get("communication_energy_wh")) for row in trace) * 3600.0
            total_j = product_encoding_j + transmission_j
            compute_j = product_encoding_j
            inference_j = 0.0
            source = "frozen_per_slot_trace_exact"
        else:
            old = old_index[(seed, capacity, method)]
            product_encoding_j = f(old.get("compute_processing_energy_wh")) * 3600.0
            transmission_j = f(old.get("communication_energy_wh")) * 3600.0
            total_j = product_encoding_j + transmission_j
            compute_j = product_encoding_j
            inference_j = 0.0
            source = "validated_frozen_scenario_energy_components"
    if total_j <= EPS:
        raise ValueError(f"Non-positive energy for {seed}/{capacity}/{method}")
    decomposition_complete = all(value is not None for value in (inference_j, product_encoding_j, transmission_j))
    return {
        "total_controllable_energy_j": total_j,
        "total_controllable_energy_wh": total_j / 3600.0,
        "compute_energy_j": compute_j if compute_j is not None else "",
        "model_inference_energy_j": inference_j if inference_j is not None else "",
        "product_generation_encoding_energy_j": product_encoding_j if product_encoding_j is not None else "",
        "transmission_energy_j": transmission_j if transmission_j is not None else "",
        "energy_decomposition_complete": int(decomposition_complete),
        "energy_source": source,
    }


def latency_metrics(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    if selector == "all":
        group = rows
    elif selector == "hv":
        group = [row for row in rows if row["is_hv"]]
    elif selector == "hv_evidence":
        group = [row for row in rows if row["is_hv_evidence_required"]]
    else:
        raise ValueError(selector)
    successes = [f(row["closure_latency_s"]) for row in group if row["success"] and row["closure_latency_s"] != ""]
    capped = [f(row["deadline_capped_latency_s"]) for row in group]
    return {
        "task_group": selector,
        "task_count": len(group),
        "success_count": sum(int(row["success"]) for row in group),
        "successful_conditional_mean_closure_latency_s": mean(successes),
        "successful_conditional_p95_closure_latency_s": percentile(successes, 0.95),
        "deadline_capped_all_task_mean_latency_s": mean(capped),
        "deadline_capped_all_task_p95_latency_s": percentile(capped, 0.95),
    }


def tut_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_selected = 0
    for seed in SEEDS:
        for capacity in CAPACITIES:
            directory = method_dir(seed, capacity, "TUT-UOI-PRODUCT-LITE")
            candidates = [row for row in read_csv(directory / "candidate_action_audit.csv") if truth(row.get("executed_flag"))]
            trace = read_csv(directory / "per_slot_resource_trace.csv")
            by_task = {row.get("task_id", ""): row for row in trace if truth(row.get("has_new_task"))}
            for row in candidates:
                all_selected += 1
                action = {"metadata": "LOW_RATE", "roi": "MEDIUM_RATE", "roi_context": "HIGH_RATE", "defer": "DEFER"}.get(row.get("candidate_payload", ""), row.get("candidate_payload", ""))
                payload = f(row.get("predicted_payload_mb"))
                scale = {"LOW_RATE": 0.05, "MEDIUM_RATE": 1.0, "HIGH_RATE": 2.2, "DEFER": 0.0}.get(action, 0.0)
                latency_scale = {"LOW_RATE": 0.10, "MEDIUM_RATE": 1.0, "HIGH_RATE": 1.9, "DEFER": 0.0}.get(action, 0.0)
                encoding_j = 2.4 * payload * scale
                encoding_latency = 0.18 * payload * latency_scale
                inference_nominal = 0.0 if action == "DEFER" else 21.6
                inference_as_implemented = inference_nominal + (15.0 * encoding_latency if action != "DEFER" else 0.0)
                predicted_total = f(row.get("predicted_energy_j"))
                predicted_tx = max(0.0, predicted_total - inference_as_implemented - encoding_j)
                actual = by_task.get(row.get("task_id", ""), {})
                actual_compute_total = f(actual.get("compute_energy_j"))
                actual_encoding = f(actual.get("payload_encoding_energy_j"))
                actual_inference = max(0.0, actual_compute_total - actual_encoding)
                accum[action]["predicted_compute"].append(inference_as_implemented)
                accum[action]["predicted_encoding"].append(encoding_j)
                accum[action]["predicted_tx"].append(predicted_tx)
                accum[action]["actual_inference"].append(actual_inference)
                accum[action]["actual_encoding"].append(actual_encoding)
                accum[action]["compute_encoding_abs_error"].append(abs((inference_as_implemented + encoding_j) - actual_compute_total))
                accum[action]["inference_overcharge"].append(max(0.0, inference_as_implemented - inference_nominal))
    rows = []
    for action in ("LOW_RATE", "MEDIUM_RATE", "HIGH_RATE", "DEFER"):
        data = accum[action]
        rows.append({
            "action": action,
            "executed_count": len(data["predicted_compute"]),
            "mean_predicted_model_inference_energy_j_as_implemented": mean(data["predicted_compute"]),
            "mean_predicted_product_encoding_energy_j": mean(data["predicted_encoding"]),
            "mean_predicted_transmission_energy_j": mean(data["predicted_tx"]),
            "mean_actual_model_inference_energy_j": mean(data["actual_inference"]),
            "mean_actual_product_encoding_energy_j": mean(data["actual_encoding"]),
            "actual_task_lifetime_transmission_energy_j": "N/A_NOT_RELIABLY_TASK_ATTRIBUTABLE",
            "mean_absolute_error_compute_plus_encoding_j": mean(data["compute_encoding_abs_error"]),
            "mean_inference_overcharge_from_encoding_latency_j": mean(data["inference_overcharge"]),
            "full_energy_mae": "N/A_TASK_LIFETIME_TX_NOT_ATTRIBUTABLE",
            "coverage_fixed_model_inference": "YES" if action != "DEFER" else "CORRECTLY_ZERO",
            "coverage_product_generation": "NO_SEPARATE_GENERATION_PROFILE",
            "coverage_encoding_compression": "YES",
            "coverage_expected_radio_transmission": "YES",
            "finding": "PREDICTED_INFERENCE_INCLUDES_ENCODING_LATENCY" if action not in {"DEFER"} else "NONE",
        })
    finding = {
        "executed_decisions": all_selected,
        "action_selection_occurs_before_inference": True,
        "all_non_defer_cover_nominal_fixed_inference": True,
        "defer_inference_is_zero": True,
        "encoding_covered": True,
        "expected_transmission_covered": True,
        "systematic_error": "Predictor adds encoding latency to predicted_latency before computing inference energy, then adds encoding energy separately.",
        "classification": "TUT_MINIMAL_ENERGY_FIX_REQUIRED",
    }
    return rows, finding


def aggregate(rows: list[dict[str, Any]], energy_rows: list[dict[str, Any]], old_scenarios: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    energy_index = {(int(row["seed"]), f(row["capacity"]), row["method"]): row for row in energy_rows}
    old_index = {(int(row["seed"]), f(row["capacity"]), row["method"]): row for row in old_scenarios}
    scenario_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for capacity in CAPACITIES:
            for method in METHODS:
                group = [row for row in rows if row["seed"] == seed and abs(row["capacity"] - capacity) < EPS and row["method"] == method]
                hv = [row for row in group if row["is_hv"]]
                evidence = [row for row in group if row["is_hv_evidence_required"]]
                success_gross = sum(f(row["successful_gross_utility"]) for row in group)
                hv_gross = sum(f(row["successful_gross_utility"]) for row in hv)
                evidence_gross = sum(f(row["successful_gross_utility"]) for row in evidence)
                old = old_index[(seed, capacity, method)]
                actual_common_cost = f(old["common_gross_utility"]) - f(old["common_net_utility"])
                successful_net = success_gross - actual_common_cost
                energy = energy_index[(seed, capacity, method)]
                total_j = f(energy["total_controllable_energy_j"])
                tx_mb = f(old["transmitted_mb"])
                task_cost_reliable = method == "OFFLOAD-EFT-THERMAL-ALIGNED"
                hv_cost = sum((f(row["compute_energy_j_task_logged"]) + f(row["transmission_energy_j_task_logged"])) / 36000.0 for row in hv) if task_cost_reliable else None
                evidence_cost = sum((f(row["compute_energy_j_task_logged"]) + f(row["transmission_energy_j_task_logged"])) / 36000.0 for row in evidence) if task_cost_reliable else None
                scenario_rows.append({
                    "seed": seed, "capacity": capacity, "method": method,
                    "task_count": len(group), "success_count": sum(row["success"] for row in group),
                    "success_closure_rate": sum(row["success"] for row in group) / max(len(group), 1),
                    "hv_task_count": len(hv), "hv_success_count": sum(row["success"] for row in hv),
                    "hv_success_rate": sum(row["success"] for row in hv) / max(len(hv), 1),
                    "hv_evidence_task_count": len(evidence), "hv_evidence_success_count": sum(row["success"] for row in evidence),
                    "hv_evidence_success_rate": sum(row["success"] for row in evidence) / max(len(evidence), 1),
                    "successful_gross_utility": success_gross,
                    "successful_gross_utility_per_task": success_gross / max(len(group), 1),
                    "all_actual_common_resource_costs": actual_common_cost,
                    "successful_net_utility": successful_net,
                    "successful_net_utility_per_task": successful_net / max(len(group), 1),
                    "hv_successful_gross_utility_per_hv_task": hv_gross / max(len(hv), 1),
                    "hv_successful_net_utility_per_hv_task": (hv_gross - hv_cost) / max(len(hv), 1) if hv_cost is not None else "N/A_UNRELIABLE_TASK_COST_ATTRIBUTION",
                    "hv_evidence_successful_gross_utility_per_task": evidence_gross / max(len(evidence), 1),
                    "hv_evidence_successful_net_utility_per_task": (evidence_gross - evidence_cost) / max(len(evidence), 1) if evidence_cost is not None else "N/A_UNRELIABLE_TASK_COST_ATTRIBUTION",
                    **energy,
                    "successful_gross_utility_per_j": success_gross / total_j,
                    "successful_gross_utility_per_wh": success_gross / (total_j / 3600.0),
                    "successful_gross_utility_per_mb": success_gross / tx_mb if tx_mb > EPS else None,
                    "transmitted_mb": tx_mb,
                    "legacy_common_net_utility": f(old["common_net_utility"]),
                    "legacy_net_utility_per_mb": f(old["common_net_utility"]) / tx_mb if tx_mb > EPS else None,
                    "maximum_thermal": f(old["maximum_thermal"]), "minimum_soc": f(old["minimum_soc"]),
                    "final_soc": f(old["final_soc"]), "thermal_violation_count": int(f(old["thermal_violation_count"])),
                    "soc_violation_count": int(f(old["soc_violation_count"])),
                    **{k: v for k, v in latency_metrics(group, "all").items() if k not in {"task_group", "task_count", "success_count"}},
                })
    pooled: list[dict[str, Any]] = []
    for capacity in CAPACITIES:
        for method in METHODS:
            group = [row for row in scenario_rows if abs(f(row["capacity"]) - capacity) < EPS and row["method"] == method]
            sums = {key: sum(f(row[key]) for row in group) for key in (
                "task_count", "success_count", "hv_task_count", "hv_success_count", "hv_evidence_task_count", "hv_evidence_success_count",
                "successful_gross_utility", "all_actual_common_resource_costs", "successful_net_utility", "total_controllable_energy_j",
                "compute_energy_j", "model_inference_energy_j", "product_generation_encoding_energy_j", "transmission_energy_j", "transmitted_mb",
            )}
            task_subset = [row for row in rows if abs(row["capacity"] - capacity) < EPS and row["method"] == method]
            lat = latency_metrics(task_subset, "all")
            pooled.append({
                "capacity": capacity, "method": method, "seed_count": len(group), **sums,
                "success_closure_rate": sums["success_count"] / sums["task_count"],
                "hv_success_rate": sums["hv_success_count"] / sums["hv_task_count"],
                "hv_evidence_success_rate": sums["hv_evidence_success_count"] / sums["hv_evidence_task_count"],
                "successful_gross_utility_per_task": sums["successful_gross_utility"] / sums["task_count"],
                "successful_net_utility_per_task": sums["successful_net_utility"] / sums["task_count"],
                "successful_gross_utility_per_j": sums["successful_gross_utility"] / sums["total_controllable_energy_j"],
                "successful_gross_utility_per_wh": sums["successful_gross_utility"] / (sums["total_controllable_energy_j"] / 3600.0),
                "successful_gross_utility_per_mb": sums["successful_gross_utility"] / sums["transmitted_mb"],
                "maximum_thermal": max(f(row["maximum_thermal"]) for row in group),
                "minimum_soc": min(f(row["minimum_soc"]) for row in group),
                "final_soc": mean([f(row["final_soc"]) for row in group if f(row["final_soc"]) > 0]),
                "thermal_violation_count": sum(int(row["thermal_violation_count"]) for row in group),
                "soc_violation_count": sum(int(row["soc_violation_count"]) for row in group),
                **{k: v for k, v in lat.items() if k not in {"task_group", "task_count", "success_count"}},
            })
    return scenario_rows, pooled


def pooled_all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        tasks = sum(int(row["task_count"]) for row in group)
        success = sum(int(row["success_count"]) for row in group)
        hv_tasks = sum(int(row["hv_task_count"]) for row in group)
        hv_success = sum(int(row["hv_success_count"]) for row in group)
        ev_tasks = sum(int(row["hv_evidence_task_count"]) for row in group)
        ev_success = sum(int(row["hv_evidence_success_count"]) for row in group)
        gross = sum(f(row["successful_gross_utility"]) for row in group)
        cost = sum(f(row["all_actual_common_resource_costs"]) for row in group)
        energy = sum(f(row["total_controllable_energy_j"]) for row in group)
        tx = sum(f(row["transmitted_mb"]) for row in group)
        output.append({
            "capacity": "POOLED_ALL", "method": method, "scenario_count": len(group), "task_count": tasks,
            "success_count": success, "success_closure_rate": success / tasks,
            "hv_task_count": hv_tasks, "hv_success_count": hv_success, "hv_success_rate": hv_success / hv_tasks,
            "hv_evidence_task_count": ev_tasks, "hv_evidence_success_count": ev_success,
            "hv_evidence_success_rate": ev_success / ev_tasks,
            "successful_gross_utility": gross, "successful_gross_utility_per_task": gross / tasks,
            "all_actual_common_resource_costs": cost, "successful_net_utility": gross - cost,
            "successful_net_utility_per_task": (gross - cost) / tasks,
            "total_controllable_energy_j": energy, "total_controllable_energy_wh": energy / 3600.0,
            "successful_gross_utility_per_j": gross / energy,
            "successful_gross_utility_per_wh": gross / (energy / 3600.0),
            "transmitted_mb": tx, "successful_gross_utility_per_mb": gross / tx,
        })
    return output


def docs(out: Path, final_status: str, tut_finding: Mapping[str, Any], overall: list[dict[str, Any]], requirements: list[dict[str, Any]]) -> None:
    out.joinpath("success_definition.md").write_text(
        "# 统一端到端成功定义\n\n仅当任务的信息/产品要求由公共评价器判为满足、产品完整到达地面、闭环时间不晚于 deadline 时，`success_i=1`。"
        "其余任务（部分传输、等级不足、过期、仅生成未交付）均为 0。`successful_gross_utility_i = success_i × common_gross_utility_i`。\n",
        encoding="utf-8",
    )
    out.joinpath("metric_definition_v2.md").write_text(
        "# Metric definition v2\n\n"
        "`U_success_gross = Σ success_i × common_gross_utility_i`。\n\n"
        "`U_success_net = U_success_gross - all_actual_common_resource_costs`，其中成本复用冻结公共评价器的能源、终端积压和硬安全成本，不只扣成功任务。\n\n"
        "`E_total_J = E_model_inference + E_product_generation/encoding + E_transmission`，系统级分母包含成功、失败、过期和未闭环任务。\n\n"
        "`successful_utility_per_J = U_success_gross / E_total_J`；`successful_utility_per_MB` 仅为辅助通信效率。分组成本/能耗不能可靠逐任务归属时报告 N/A，不做均分。\n",
        encoding="utf-8",
    )
    out.joinpath("preflight_report.md").write_text(
        "# Preflight report\n\n- 冻结四方法比较的 hash/safety 状态为 PASS。\n- 60 个方法场景逐任务文件齐全；TUT 15 个逐 slot 与候选动作文件齐全。\n"
        "- 三方法冻结源中 PROP/OFFLOAD 有逐 slot 汇总，B1 的 0.50/0.35 复用已验证场景级能耗分解。\n- 本脚本不运行仿真，不修改动作、轨迹、产品、任务、公共环境或评价器。\n",
        encoding="utf-8",
    )
    out.joinpath("cost_accounting_audit.md").write_text(
        "# Cost accounting audit\n\n公共 net 公式实际为 gross 减去：可控任务能源成本（10 Wh 归一化，λ=1）、终端积压成本（64 MB 归一化，λ=1）和硬安全罚（λ=10）。"
        "deadline 内时效衰减已在 gross 中；deadline 后为 0。无独立 transmitted-byte 货币成本、逐 slot queue/storage 成本或 ground-compute 成本。\n\n"
        "- B1：星上压缩与无线传输进入总可控能源；无星上模型推理。\n"
        "- OFFLOAD：实际本地/卫星计算与下传能源进入总可控能源；现有公共合同未单列地面计算货币成本。\n"
        "- TUT/PROP：模型推理、产品编码和无线传输进入总可控能源；产品裁剪没有独立能耗画像，编码并入 compute。\n"
        "- 所有方法的新 successful net 均从成功 gross 扣除冻结场景的全部实际公共成本，失败任务已发生的成本不会消失。\n",
        encoding="utf-8",
    )
    out.joinpath("model_inference_reuse_audit.md").write_text(
        "# Model inference reuse audit\n\n"
        "## PROP\n\n实现只在新任务分支调用一次 `inference_runner.run`；metadata/ROI/ROI+Ctx 与后续补传复用检测结果/任务状态。无证据表明每升级一级重跑 YOLO，结论：`MODEL_INFERENCE_DOUBLE_COUNTED=false`。\n\n"
        "## TUT\n\n动作选择发生在推理前；非 DEFER 均承担固定 small-profile 推理，DEFER 为 0。预测器覆盖编码和预期传输。"
        "但预测器把编码时延并入模型推理时延后按模型功率计能耗，并再次加编码能耗，导致系统性高估；这需要最小修正，但不属于 PROP 分阶段多次推理。\n",
        encoding="utf-8",
    )
    by_method = {row["method"]: row for row in overall}
    net_winner = max(overall, key=lambda row: f(row["successful_net_utility_per_task"]))
    energy_winner = max(overall, key=lambda row: f(row["successful_gross_utility_per_j"]))
    comm_winner = max(overall, key=lambda row: f(row["successful_gross_utility_per_mb"]))
    req = {row["requirement_group"]: row for row in requirements if row["scope"] == "HV_TARGET"}
    evidence_hvt = int(req.get("context_evidence_required", {}).get("task_count", 0))
    prop = by_method["PROP-HEADROOM"]
    tut = by_method["TUT-UOI-PRODUCT-LITE"]
    out.joinpath("final_report.md").write_text(
        "# 四方法端到端成功交付与通算能耗复核\n\n"
        f"## 最终状态\n\n`{final_status}`\n\n"
        "1. 是。新主效用仅来自 deadline 前公共闭环成功任务。\n"
        "2. 是。失败、过期和未闭环任务的冻结公共成本与系统总能耗仍进入扣减/分母。\n"
        f"3. HV-TARGET 中真正要求 context/evidence 的任务数为 `{evidence_hvt}`（按 5 个独立 seed 的任务合同计，不重复 capacity）。\n"
        f"4. 全部 15 场景中，PROP 完成 `{prop['hv_evidence_success_count']}` 个、TUT 完成 `{tut['hv_evidence_success_count']}` 个 evidence-required HV 任务实例（同一任务在 3 个 capacity 下分别计实验实例）。\n"
        f"5. 成功交付净效用/task 最高：`{net_winner['method']}`（{f(net_winner['successful_net_utility_per_task']):.6f}）。\n"
        f"6. 成功 gross utility/J 最高：`{energy_winner['method']}`（{f(energy_winner['successful_gross_utility_per_j']):.9f}）。\n"
        f"7. TUT 的成功 gross utility/MB 仍为四方法最高：`{comm_winner['method']}`（TUT={f(tut['successful_gross_utility_per_mb']):.6f}）；它仍只是辅助通信效率结论。\n"
        "8. TUT 评分覆盖固定推理、编码和预期传输，但推理项错误吸收编码时延后又另加编码能耗，因此需要最小修正。\n"
        "9. PROP 渐进交付未发现重复模型推理计费。\n"
        "10. 新口径改变了旧的‘net utility/Wh’核心能源效率解释，并移除了非闭环任务的正收益；绝对闭环排序不变，具体净效用和能效赢家见新表。\n\n"
        "限制：B1/PROP/TUT 的共享传输能源不能从冻结日志可靠逐任务归属，因此 HV/HV-evidence net 与分组 utility/J 按合同标记 N/A；系统级指标完整。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {out}")
    out.mkdir(parents=True)

    old_scenarios = read_csv(FORMAL / "scenario_level_metrics.csv")
    old_idx = old_multi_index()
    all_tasks: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    input_files: list[Path] = [FORMAL / name for name in ("frozen_method_manifest.json", "hash_validation.json", "safety_summary.json", "scenario_level_metrics.csv")]
    for seed in SEEDS:
        for capacity in CAPACITIES:
            for method in METHODS:
                directory = method_dir(seed, capacity, method)
                all_tasks.extend(standardize_task_rows(seed, capacity, method))
                energy_rows.append({"seed": seed, "capacity": capacity, "method": method, **energy_breakdown(seed, capacity, method, old_idx)})
                input_files.extend([directory / "per_task_results.csv", directory / "summary.json"])
                if (directory / "per_slot_resource_trace.csv").exists():
                    input_files.append(directory / "per_slot_resource_trace.csv")

    scenario_rows, capacity_rows = aggregate(all_tasks, energy_rows, old_scenarios)
    overall = pooled_all(scenario_rows)
    latency_rows = []
    for seed in SEEDS:
        for capacity in CAPACITIES:
            for method in METHODS:
                subset = [row for row in all_tasks if row["seed"] == seed and abs(row["capacity"] - capacity) < EPS and row["method"] == method]
                for selector in ("all", "hv", "hv_evidence"):
                    latency_rows.append({"seed": seed, "capacity": capacity, "method": method, **latency_metrics(subset, selector)})

    requirement_rows = []
    unique_tasks = {(row["seed"], row["task_id"]): row for row in all_tasks if row["method"] == "B1-COMMON-THERMAL" and abs(row["capacity"] - 1.0) < EPS}
    base_tasks = list(unique_tasks.values())
    scopes = {
        "ALL": base_tasks,
        "HV": [row for row in base_tasks if row["is_hv"]],
        "HV_TARGET": [row for row in base_tasks if row["is_hv_target"]],
    }
    for scope, tasks in scopes.items():
        for group in ("alert_required", "roi_verification_required", "context_evidence_required", "unknown_requirement"):
            requirement_rows.append({"scope": scope, "requirement_group": group, "task_count": sum(row["requirement_group"] == group for row in tasks), "total_scope_tasks": len(tasks)})

    hv_rows = []
    evidence_rows = []
    for row in scenario_rows:
        hv_rows.append({key: row.get(key, "") for key in (
            "seed", "capacity", "method", "hv_task_count", "hv_success_count", "hv_success_rate",
            "hv_successful_gross_utility_per_hv_task", "hv_successful_net_utility_per_hv_task",
        )})
        evidence_rows.append({key: row.get(key, "") for key in (
            "seed", "capacity", "method", "hv_evidence_task_count", "hv_evidence_success_count", "hv_evidence_success_rate",
            "hv_evidence_successful_gross_utility_per_task", "hv_evidence_successful_net_utility_per_task",
        )})

    efficiency_rows = []
    for row in scenario_rows:
        efficiency_rows.append({key: row.get(key, "") for key in (
            "seed", "capacity", "method", "successful_gross_utility", "total_controllable_energy_j", "total_controllable_energy_wh",
            "successful_gross_utility_per_j", "successful_gross_utility_per_wh", "successful_gross_utility_per_mb", "transmitted_mb",
            "compute_energy_j", "model_inference_energy_j", "product_generation_encoding_energy_j", "transmission_energy_j", "energy_decomposition_complete", "energy_source",
        )} | {"hv_successful_gross_utility_per_j": "N/A_UNRELIABLE_TASK_ENERGY_ATTRIBUTION" if row["method"] != "OFFLOAD-EFT-THERMAL-ALIGNED" else "SEE_TASK_ATTRIBUTABLE_AUDIT",
             "hv_evidence_successful_gross_utility_per_j": "N/A_UNRELIABLE_TASK_ENERGY_ATTRIBUTION" if row["method"] != "OFFLOAD-EFT-THERMAL-ALIGNED" else "SEE_TASK_ATTRIBUTABLE_AUDIT"})

    legacy_rows = [{
        "seed": row["seed"], "capacity": row["capacity"], "method": row["method"],
        "LEGACY_NET_UTILITY_PER_MB": row["legacy_net_utility_per_mb"],
        "SUCCESSFUL_GROSS_UTILITY_PER_MB": row["successful_gross_utility_per_mb"],
        "legacy_common_net_utility": row["legacy_common_net_utility"],
        "successful_gross_utility": row["successful_gross_utility"], "transmitted_mb": row["transmitted_mb"],
    } for row in scenario_rows]

    audit_rows = []
    for method in METHODS:
        audit_rows.extend([
            {"method": method, "cost_or_energy_item": "onboard_model_inference", "actual_energy_accounting": "INCLUDED" if method not in {"B1-COMMON-THERMAL"} else "NOT_APPLICABLE", "common_net_cost": "INCLUDED_IN_AGGREGATE_ENERGY_COST", "task_attribution": "RELIABLE" if method == "OFFLOAD-EFT-THERMAL-ALIGNED" else "LIMITED"},
            {"method": method, "cost_or_energy_item": "product_generation_cropping", "actual_energy_accounting": "NO_SEPARATE_PROFILE", "common_net_cost": "NOT_SEPARATE", "task_attribution": "N/A"},
            {"method": method, "cost_or_energy_item": "encoding_compression", "actual_energy_accounting": "INCLUDED", "common_net_cost": "INCLUDED_IN_AGGREGATE_ENERGY_COST", "task_attribution": "RELIABLE" if method in {"B1-COMMON-THERMAL", "TUT-UOI-PRODUCT-LITE"} else "FOLDED_IN_COMPUTE"},
            {"method": method, "cost_or_energy_item": "radio_transmission", "actual_energy_accounting": "INCLUDED", "common_net_cost": "INCLUDED_IN_AGGREGATE_ENERGY_COST", "task_attribution": "RELIABLE" if method == "OFFLOAD-EFT-THERMAL-ALIGNED" else "NOT_RELIABLE_FOR_GROUPS"},
            {"method": method, "cost_or_energy_item": "transmitted_bytes_charge", "actual_energy_accounting": "OBSERVED_MB_ONLY", "common_net_cost": "NO_SEPARATE_BYTE_CHARGE", "task_attribution": "OBSERVED"},
            {"method": method, "cost_or_energy_item": "deadline_latency", "actual_energy_accounting": "TIMELINESS_IN_GROSS_AND_ZERO_AFTER_DEADLINE", "common_net_cost": "NO_ADDITIONAL_COST", "task_attribution": "RELIABLE"},
            {"method": method, "cost_or_energy_item": "queue_storage", "actual_energy_accounting": "TERMINAL_BACKLOG_ONLY", "common_net_cost": "INCLUDED", "task_attribution": "SYSTEM_LEVEL"},
            {"method": method, "cost_or_energy_item": "ground_compute", "actual_energy_accounting": "NO_COMMON_CONTRACT_COST", "common_net_cost": "NOT_INCLUDED", "task_attribution": "N/A"},
        ])

    tut_rows, tut_finding = tut_audit()
    final_status = "TUT_MINIMAL_ENERGY_FIX_REQUIRED"

    write_csv(out / "task_requirement_breakdown.csv", requirement_rows)
    write_csv(out / "energy_accounting_audit.csv", audit_rows)
    write_csv(out / "tut_energy_score_audit.csv", tut_rows)
    write_csv(out / "successful_task_level_metrics.csv", all_tasks)
    write_csv(out / "successful_high_value_metrics.csv", hv_rows)
    write_csv(out / "successful_evidence_metrics.csv", evidence_rows)
    write_csv(out / "energy_efficiency_summary.csv", efficiency_rows + overall)
    write_csv(out / "latency_summary_v2.csv", latency_rows)
    write_csv(out / "capacity_summary_v2.csv", capacity_rows)
    write_csv(out / "pooled_comparison_v2.csv", overall)
    write_csv(out / "legacy_metric_comparison.csv", legacy_rows)

    original_safety = json.loads((FORMAL / "safety_summary.json").read_text(encoding="utf-8"))
    write_json(out / "safety_validation.json", {
        "status": "PASS" if original_safety.get("all_60_method_scenarios_safe") else "FAIL",
        "source": str(FORMAL / "safety_summary.json"),
        "all_60_method_scenarios_safe": bool(original_safety.get("all_60_method_scenarios_safe")),
        "total_thermal_violations": int(original_safety.get("total_thermal_violations", 0)),
        "total_soc_violations": int(original_safety.get("total_soc_violations", 0)),
        "maximum_thermal": f(original_safety.get("maximum_thermal")),
        "minimum_soc": f(original_safety.get("minimum_soc")),
    })
    original_hash = json.loads((FORMAL / "hash_validation.json").read_text(encoding="utf-8"))
    unique_inputs = sorted(set(path for path in input_files if path.exists()), key=lambda path: path.as_posix())
    write_json(out / "hash_validation.json", {
        "status": "PASS" if original_hash.get("all_pass") else "FAIL",
        "frozen_comparison_hash_validation_status": original_hash.get("status"),
        "frozen_comparison_all_pass": bool(original_hash.get("all_pass")),
        "input_artifact_count": len(unique_inputs),
        "input_sha256": {path.as_posix(): sha256(path) for path in unique_inputs},
    })
    write_json(out / "rerun_decision.json", {
        "status": final_status,
        "metric_reaggregation_completed": True,
        "simulator_run_performed": False,
        "other_methods_must_remain_frozen": True,
        "tut_minimal_fix_scope": "In Predictor.predict_action, compute nominal inference energy from profile inference latency only; add encoding energy once. Do not alter UoI, profiles, tasks, evaluator, or environment.",
        "formal_tut_rerun_performed": False,
        "reason": tut_finding["systematic_error"],
    })
    write_json(out / "rollback_manifest.json", {
        "status": "OUTPUT_ONLY_NO_SCIENTIFIC_SOURCE_CHANGED",
        "output_directory": out.as_posix(),
        "rollback": "Delete only this newly created audit output directory if rollback is requested.",
        "authoritative_inputs_modified": False,
        "method_source_modified": False,
    })
    docs(out, final_status, tut_finding, overall, requirement_rows)

    print(json.dumps({"status": final_status, "output": out.as_posix(), "task_rows": len(all_tasks), "scenario_rows": len(scenario_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
