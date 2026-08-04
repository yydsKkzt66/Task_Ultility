"""Pure-table completion gates, progressive metrics, and fairness reporting."""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    from .atomic_io import atomic_write_csv, atomic_write_json
    from .contracts import EVENT_FIELDS, METHODS, PER_TASK_FIELDS, SLOT_RESOURCE_FIELDS
    from .ledger import account_task, corrected_latency, summarize_progressive
except ImportError:
    from atomic_io import atomic_write_csv, atomic_write_json
    from contracts import EVENT_FIELDS, METHODS, PER_TASK_FIELDS, SLOT_RESOURCE_FIELDS
    from ledger import account_task, corrected_latency, summarize_progressive


@dataclass(frozen=True)
class MethodTables:
    method: str
    tasks: list[dict[str, str]]
    events: list[dict[str, str]]
    slots: list[dict[str, str]]
    resource: dict[str, Any]
    identity: dict[str, Any]
    scenario: dict[str, Any]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), tuple(reader.fieldnames or ())


def validate_completed_method(path: Path, phase: str, method: str, expected_slots: int, expected_tasks: int = 268) -> MethodTables:
    path = Path(path)
    for name in ("method_completed.json", "method_run_status.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    completed, status = _json(path / "method_completed.json"), _json(path / "method_run_status.json")
    if completed.get("status") != "PASS" or status.get("status") != "PASS":
        raise ValueError("method completion/status is not PASS")
    for name in ("native_identity.json", "reconstructed_scenario_identity.json", "per_task_outcomes.csv",
                 "ground_delivery_events.csv", "per_slot_resource_trace.csv", "resource_summary.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    if completed.get("phase") != phase or completed.get("method") != method:
        raise ValueError("phase-method completion mismatch")
    tasks, task_fields = _csv(path / "per_task_outcomes.csv")
    events, event_fields = _csv(path / "ground_delivery_events.csv")
    slots, slot_fields = _csv(path / "per_slot_resource_trace.csv")
    if task_fields != PER_TASK_FIELDS or event_fields != EVENT_FIELDS or slot_fields != SLOT_RESOURCE_FIELDS:
        raise ValueError("unified CSV schema mismatch")
    if len(tasks) != expected_tasks or len(slots) != expected_slots:
        raise ValueError("unified CSV row count mismatch")
    if len({row["task_id"] for row in tasks}) != expected_tasks:
        raise ValueError("per-task ID coverage mismatch")
    identity, scenario = _json(path / "native_identity.json"), _json(path / "reconstructed_scenario_identity.json")
    if not identity.get("source_identity_passed") or not identity.get("all_src_modules_within_source_root"):
        raise ValueError("source identity mismatch")
    if not scenario.get("task_hash_match") or not scenario.get("environment_hash_match") or not scenario.get("benchmark_fingerprint_match"):
        raise ValueError("scenario identity mismatch")
    return MethodTables(method, tasks, events, slots, _json(path / "resource_summary.json"), identity, scenario)


def _d(value: Any) -> Decimal:
    return Decimal(str(value or 0))


def _p95(values: list[Decimal]) -> str | None:
    if not values: return None
    ordered=sorted(values); return str(ordered[max(0, (95 * len(values) + 99) // 100 - 1)])


def _method_metrics(table: MethodTables, utility: dict[str, Any], accounting: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_events: dict[str, list[dict[str, str]]] = {}
    for row in table.events: by_events.setdefault(row["task_id"], []).append(row)
    ledger = []
    for task in table.tasks: ledger.extend(account_task(task, by_events.get(task["task_id"], []), utility))
    progressive = summarize_progressive(ledger); latency = corrected_latency(table.tasks, ledger)
    energy = _d(table.resource.get("controllable_energy_wh", 0)); tx = _d(table.resource.get("transmitted_mb", 0))
    backlog = _d(table.slots[-1]["queue_mb"] if table.slots else 0)
    safe = _d(table.resource.get("soc_violation_count", 0)) + _d(table.resource.get("thermal_violation_count", 0))
    cost = (_d(accounting["lambda_E_common"]) * energy / _d(accounting["energy_normalization"]["controllable_mission_energy_wh"])
            + _d(accounting["lambda_Q_common"]) * backlog / _d(accounting["backlog_normalization"]["terminal_backlog_mb"])
            + _d(accounting["lambda_safe_common"]) * safe)
    gross = _d(progressive["j_prog_gross"]); net = gross - cost
    closed=[r for r in table.tasks if int(r["closure_success"] or 0)]; hv=[r for r in table.tasks if r["value_level"] in {"high","urgent"}]
    hv_closed=[r for r in hv if int(r["closure_success"] or 0)]
    action=Counter(r["executed_action"] or "NONE" for r in table.tasks)
    product=Counter(r["delivered_product"] or "NONE" for r in table.tasks)
    return ({"method":table.method,"J_prog_gross":str(gross),"J_prog_net":str(net),"common_actual_cost":str(cost),
        "UPDL_s":progressive["updl_s"],"progressive_utility_denominator":progressive["effective_marginal_utility_denominator"],
        "tasks_with_progressive_delivery":progressive["tasks_with_on_time_positive_layer"],
        "metadata_layer_count":progressive["metadata_layer_count"],"roi_layer_count":progressive["roi_layer_count"],
        "evidence_layer_count":progressive["evidence_layer_count"],"closure_count":len(closed),
        "closure_rate":str(Decimal(len(closed))/Decimal(max(len(table.tasks),1))),"hv_closure_count":len(hv_closed),
        "hv_closure_rate":str(Decimal(len(hv_closed))/Decimal(max(len(hv),1))),
        **latency,"controllable_energy_wh":str(energy),"transmitted_mb":str(tx),
        "utility_per_wh":str(net/energy) if energy>0 else None,"utility_per_mb":str(net/tx) if tx>0 else None,
        "minimum_soc":table.resource.get("minimum_soc"),"final_soc":table.resource.get("final_soc"),
        "maximum_thermal":table.resource.get("maximum_thermal"),"soc_violation_count":table.resource.get("soc_violation_count"),
        "thermal_violation_count":table.resource.get("thermal_violation_count"),
        "defer_count":sum(1 for r in table.tasks if (r["executed_action"] or "").lower() in {"delay/store","defer"}),
        "action_distribution":json.dumps(action,sort_keys=True),"product_distribution":json.dumps(product,sort_keys=True)}, ledger)


def aggregate_phase(root: Path, phase: str, *, expected_slots: int, expected_tasks: int = 268, write_outputs: bool = False) -> dict[str, Any]:
    root=Path(root); phase_root=root/phase
    missing=[method for method in METHODS if not (phase_root/method/"method_completed.json").is_file()]
    if missing:
        raise RuntimeError(f"incomplete five-method phase: {missing}")
    tables=[validate_completed_method(phase_root/method,phase,method,expected_slots,expected_tasks) for method in METHODS]
    config=_json(root/"canonical_scenario_bundle/resolved_config.json")
    metrics=[]; ledgers={}
    for table in tables:
        metric,ledger=_method_metrics(table,config["product_requirement_utility"],config["unified_ground_received_accounting"])
        metrics.append(metric); ledgers[table.method]=ledger
    fairness={"status":"PASS","phase":phase,"methods":list(METHODS),
        "task_hashes":sorted({t.scenario["recomputed_task_hash"] for t in tables}),
        "environment_hashes":sorted({t.scenario["recomputed_environment_hash"] for t in tables}),
        "fingerprints":sorted({t.scenario["recomputed_benchmark_fingerprint"] for t in tables}),
        "single_reconstruction":all(_json(phase_root/t.method/"method_run_status.json").get("scenario_reconstruction_count")==1 for t in tables),
        "parameter_classification":[
            {"parameter":"capacity","classification":"ACTIVE","evidence":"canonical_scenario_bundle/slot_trace.csv link throughput"},
            {"parameter":"ground_compute_energy","classification":"INACTIVE","evidence":"OFFLOAD resource_summary ground_compute_energy_counted=false"},
            {"parameter":"B1 compression quality","classification":"PROXY","evidence":"resolved_config.baseline_b1_enhanced.compression"}]}
    if len(fairness["task_hashes"])!=1 or len(fairness["environment_hashes"])!=1 or len(fairness["fingerprints"])!=1 or not fairness["single_reconstruction"]:
        raise RuntimeError("cross-method fairness identity mismatch")
    result={"status":"PASS","phase":phase,"method_metrics":metrics,"fairness":fairness,"ledger_rows":ledgers}
    if write_outputs:
        prefix="" if phase=="formal" else f"{phase}_"
        fields=tuple(metrics[0])
        atomic_write_csv(root/f"{prefix}method_metrics.csv",metrics,fields)
        report_prefix="formal_" if phase=="formal" else prefix
        atomic_write_json(root/f"{report_prefix}fairness_report.json",fairness)
        atomic_write_json(root/f"{report_prefix}ledger_report.json",{"phase":phase,"methods":ledgers})
    return result
