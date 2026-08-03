"""Pure audit/accounting helpers for the frozen V2 final gate.

This module is additive.  It does not mutate the frozen V2 implementation or
any historical result.  The public accounting intentionally mirrors the
frozen conditional-calibration adapter, including its proportional HV cost
allocation and closed-HV gross convention.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PRODUCT_ALIASES = {
    "": "",
    "defer": "defer",
    "metadata": "metadata",
    "alert": "metadata",
    "roi": "roi",
    "verification": "roi",
    "evidence": "roi_context",
    "context": "roi_context",
    "roi_context": "roi_context",
    "roi+context": "roi_context",
    "full": "roi_context",
    "compressed_full_frame": "roi_context",
}
PRODUCT_RANK = {"": -1, "defer": -1, "metadata": 0, "roi": 1, "roi_context": 2}
MODEL_ALIASES = {
    "lightweight": "lightweight",
    "balanced": "balanced",
    "high_accuracy": "high_accuracy",
    "ground_high_accuracy": "high_accuracy",
    "local": "high_accuracy",  # OFFLOAD frozen config local_model
}
STAGE_PRODUCT = {"metadata": "metadata", "roi": "roi", "evidence": "roi_context", "full": "roi_context"}


def f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def i(value: Any) -> int:
    return int(float(value or 0))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def product_normalize(product: Any) -> str:
    key = str(product or "").strip().lower().replace("+", "+").replace(" ", "_")
    return PRODUCT_ALIASES.get(key, key)


def product_satisfies(product: Any, required: Any) -> bool:
    return PRODUCT_RANK.get(product_normalize(product), -1) >= PRODUCT_RANK.get(product_normalize(required), 99)


def delivered_satisfies(delivered: Any, required: Any) -> bool:
    """Evaluate cumulative ledgers such as ``metadata+roi_context``."""

    tokens = re.split(r"[+;,|]", str(delivered or ""))
    highest = max((PRODUCT_RANK.get(product_normalize(token), -1) for token in tokens), default=-1)
    return highest >= PRODUCT_RANK.get(product_normalize(required), 99)


def model_normalize(model: Any) -> str:
    return MODEL_ALIASES.get(str(model or "").strip().lower(), str(model or "").strip().lower())


def scan_used_seeds(root: Path, scan_roots: Iterable[str] = ("configs", "outputs")) -> tuple[set[int], dict[str, Any]]:
    """Scan path names plus compact provenance/config files, never event CSVs."""

    used: set[int] = set()
    path_pattern = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)
    seed_line_pattern = re.compile(r"(?:seed|seeds|random_seed)", re.IGNORECASE)
    integer_pattern = re.compile(r"(?<![\w.])(\d{1,9})(?![\w.])")
    inspected = 0
    skipped_large = 0
    for name in scan_roots:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            for match in path_pattern.finditer(path.name):
                used.add(int(match.group(1)))
            if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".json", ".md", ".txt", ".py"}:
                continue
            lower = path.name.lower()
            if name == "outputs" and not any(token in lower for token in ("config", "audit", "identity", "status", "complete", "seed", "spec", "manifest", "report")):
                continue
            if path.stat().st_size > 2_000_000:
                skipped_large += 1
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            inspected += 1
            for match in path_pattern.finditer(text):
                used.add(int(match.group(1)))
            for line in text.splitlines():
                if seed_line_pattern.search(line):
                    for match in integer_pattern.finditer(line):
                        used.add(int(match.group(1)))
    return used, {"content_files_inspected": inspected, "large_files_skipped": skipped_large, "used_seed_count": len(used)}


def choose_unused_seeds(used: set[int], count: int, start: int = 10000) -> list[int]:
    selected: list[int] = []
    candidate = int(start)
    while len(selected) < count:
        if candidate not in used:
            selected.append(candidate)
        candidate += 1
    return selected


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = int(math.floor(index)), int(math.ceil(index))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def _decision_assignment(decisions: list[dict[str, str]], q_map: Mapping[str, float]) -> dict[tuple[str, str], tuple[str, float]]:
    mapping: dict[tuple[str, str], tuple[str, float]] = {}
    for row in decisions:
        product = product_normalize(row.get("selected_product"))
        model = model_normalize(row.get("selected_model"))
        if product not in {"", "defer"} and model in q_map:
            mapping[(row["task_id"], product)] = (model, f(q_map[model]))
    return mapping


def evaluate_public_run(
    method: str,
    path: Path,
    resolved_config: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the frozen q-once public evaluator to one unified run directory."""

    from src.payload.information_sufficiency import delta_information_utility

    tasks = {row["task_id"]: row for row in read_csv(path / "per_task_outcomes.csv")}
    events = sorted(read_csv(path / "ground_delivery_events.csv"), key=lambda row: (f(row["ground_available_time"]), row["task_id"], row["stage"]))
    decisions = read_csv(path / "v2_decision_trace.csv") if (path / "v2_decision_trace.csv").is_file() else []
    q_map = {key: f(value) for key, value in frozen["q_eff"].items()}
    assignment = _decision_assignment(decisions, q_map)
    delivered: dict[str, set[str]] = defaultdict(set)
    ledger: list[dict[str, Any]] = []
    gross = hv_gross = 0.0
    missing_model = 0
    for event in events:
        if not i(event.get("valid_before_deadline")):
            continue
        product = STAGE_PRODUCT.get(str(event.get("stage", "")).lower(), "")
        if not product or event["task_id"] not in tasks:
            continue
        task = tasks[event["task_id"]]
        model_q = assignment.get((event["task_id"], product))
        model_source = "v2_executed_product"
        if model_q is None and decisions:
            prior = [value for (task_id, prior_product), value in assignment.items() if task_id == event["task_id"] and prior_product == "metadata"]
            prior = prior or [value for (task_id, _), value in assignment.items() if task_id == event["task_id"]]
            if prior:
                model_q = prior[-1]
                model_source = "cached_executed_inference_for_residual"
        if model_q is None:
            model = model_normalize(task.get("selected_model"))
            if model in q_map:
                model_q = (model, q_map[model])
                model_source = "unified_task_actual_model"
        if model_q is None:
            missing_model += 1
            continue
        model, q_eff = model_q
        delta_i = delta_information_utility(product, delivered[event["task_id"]], task["product_requirement"], resolved_config["product_requirement_utility"])
        arrival = f(task["arrival_time"])
        lifetime = max(f(task["deadline"]) - arrival, 1.0e-9)
        timely = max(0.0, min(1.0, 1.0 - (f(event["ground_available_time"]) - arrival) / lifetime))
        positive = f(task["effective_task_value"]) * delta_i * q_eff * timely
        delivered[event["task_id"]].add(product)
        gross += positive
        if task["value_level"] in {"high", "urgent"} and i(task["closure_success"]):
            hv_gross += positive
        ledger.append({
            "method": method, "task_id": event["task_id"], "value_level": task["value_level"],
            "required_product": product_normalize(task["product_requirement"]), "stage": event["stage"],
            "product": product, "model": model, "model_source": model_source,
            "V": task["effective_task_value"], "delta_I": delta_i, "q_eff": q_eff, "T": timely,
            "gross": positive, "gross_recomputed": f(task["effective_task_value"]) * delta_i * q_eff * timely,
            "q_count": 1, "ground_available_time_s": event["ground_available_time"],
        })
    trace = read_csv(path / "per_slot_resource_trace.csv")
    resource_json = path / "resource_summary.json"
    resource = json.loads(resource_json.read_text(encoding="utf-8")) if resource_json.is_file() else read_csv(path / "resource_summary.csv")[0]
    energy_wh = f(resource.get("controllable_energy_wh"))
    tx_mb = sum(f(row.get("transmitted_mb")) for row in trace)
    terminal_q = f(trace[-1].get("queue_mb")) if trace else 0.0
    peak_q = max([f(row.get("queue_mb")) for row in trace] or [0.0])
    soc_v = sum(i(row.get("soc_violation")) for row in trace)
    thermal_v = sum(i(row.get("thermal_violation")) for row in trace)
    prices = frozen["public_prices"]
    energy_cost = f(prices["energy_price_per_Wh"]) * energy_wh
    tx_cost = f(prices["downlink_price_per_MB"]) * tx_mb
    terminal_cost = f(prices["terminal_backlog_price_per_MB"]) * terminal_q
    total_cost = energy_cost + tx_cost + terminal_cost
    hv_cost = total_cost * hv_gross / gross if gross > 0 else 0.0
    closed = [row for row in tasks.values() if i(row["closure_success"])]
    latencies = [f(row["closure_ground_time"]) - f(row["arrival_time"]) for row in closed if row.get("closure_ground_time") not in {None, ""}]
    model_counts = Counter(model_normalize(row.get("selected_model")) for row in tasks.values() if model_normalize(row.get("selected_model")) in q_map)
    event_products = Counter(row["product"] for row in ledger)
    metric = {
        "method": method, "J_all_net": gross - total_cost, "J_HV_net": hv_gross - hv_cost,
        "gross_utility": gross, "HV_gross_utility": hv_gross, "energy_cost": energy_cost,
        "downlink_cost": tx_cost, "terminal_backlog_cost": terminal_cost, "total_cost": total_cost,
        "controllable_energy_Wh": energy_wh, "transmitted_MB": tx_mb, "terminal_queue_MB": terminal_q,
        "peak_queue_MB": peak_q, "minimum_soc": min([f(row.get("battery_soc")) for row in trace] or [0.0]),
        "maximum_thermal": max([f(row.get("thermal")) for row in trace] or [0.0]),
        "soc_violation_count": soc_v, "thermal_violation_count": thermal_v,
        "feasibility_status": "Infeasible" if soc_v or thermal_v else "Feasible",
        "all_closure_count": len(closed),
        "HV_closure_count": sum(row["value_level"] in {"high", "urgent"} for row in closed),
        "high_closure_count": sum(row["value_level"] == "high" for row in closed),
        "urgent_closure_count": sum(row["value_level"] == "urgent" for row in closed),
        "all_closure_rate": len(closed) / max(len(tasks), 1),
        "mean_full_closure_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "p95_full_closure_latency_s": percentile(latencies, 0.95),
        "metadata_delivery_count": event_products["metadata"], "ROI_delivery_count": event_products["roi"],
        "Evidence_delivery_count": event_products["roi_context"],
        "missing_event_model_count": missing_model,
        "q_once_max_recompute_error": max([abs(f(row["gross"]) - f(row["gross_recomputed"])) for row in ledger] or [0.0]),
        "model_lightweight_count": model_counts["lightweight"], "model_balanced_count": model_counts["balanced"],
        "model_high_accuracy_count": model_counts["high_accuracy"],
    }
    return metric, ledger


def ordinary_behavior_audit(decisions: list[dict[str, str]], candidates: list[dict[str, str]]) -> dict[str, Any]:
    ordinary = [row for row in decisions if row.get("value_level") in {"low", "medium"}]
    by_task = Counter(row["task_id"] for row in ordinary if row.get("selected_product") != "defer")
    residual = sum(i(row.get("residual_registered")) for row in ordinary)
    matrix = Counter((product_normalize(row.get("required_product")), product_normalize(row.get("selected_product"))) for row in ordinary)
    candidate_grid = {(row.get("model"), product_normalize(row.get("product"))) for row in candidates if row.get("model") in {"lightweight", "balanced", "high_accuracy"}}
    return {
        "ordinary_decision_count": len(ordinary),
        "ordinary_non_defer_task_max_decisions": max(by_task.values(), default=0),
        "ordinary_residual_registered_count": residual,
        "ordinary_one_shot_pass": max(by_task.values(), default=0) <= 1 and residual == 0,
        "under_requirement_selection_count": sum(not product_satisfies(row.get("selected_product"), row.get("required_product")) and row.get("selected_product") != "defer" for row in ordinary),
        "over_requirement_selection_count": sum(PRODUCT_RANK.get(product_normalize(row.get("selected_product")), -1) > PRODUCT_RANK.get(product_normalize(row.get("required_product")), 99) for row in ordinary),
        "required_selected_matrix": [{"required_product": key[0], "selected_product": key[1], "count": value} for key, value in sorted(matrix.items())],
        "candidate_grid": sorted([list(item) for item in candidate_grid]),
        "candidate_grid_complete": candidate_grid == {(model, product) for model in ("lightweight", "balanced", "high_accuracy") for product in ("metadata", "roi", "roi_context")},
    }
