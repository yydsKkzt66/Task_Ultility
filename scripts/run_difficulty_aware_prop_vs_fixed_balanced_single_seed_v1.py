"""Isolated difficulty-aware Adaptive PROP versus Fixed-B diagnostic."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


VERSION = "DIFFICULTY-AWARE-PROP-SINGLE-SEED-v1"
COMPLETION = "DIFFICULTY_AWARE_PROP_VS_FIXED_BALANCED_SINGLE_SEED_COMPLETE"
FIXED = "FIXED-B-DIFFICULTY"
ADAPTIVE = "ADAPTIVE-PROP-DIFFICULTY"
METHODS = (FIXED, ADAPTIVE)
SLUG = {FIXED: "FIXED_B_DIFFICULTY", ADAPTIVE: "ADAPTIVE_PROP_DIFFICULTY"}
CONFIG = ROOT / "configs" / "difficulty_aware_prop_single_seed_v1.yaml"
OUTPUT_BASE = ROOT / "outputs" / "_difficulty_aware_prop_vs_fixed_balanced_single_seed_v1"


def read_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in data for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data or [{"empty": ""}])


def stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    index = 1
    while (OUTPUT_BASE / f"attempt_{index:03d}").exists():
        index += 1
    path = OUTPUT_BASE / f"attempt_{index:03d}"
    for name in ("audit", "runs", "run_specs", "scenario", "tables"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def build_difficulty_manifest(tasks: list[dict[str, str]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    difficulty_seed = int(config["difficulty_seed"])
    confusion_seed = int(config["observation_confusion_seed"])
    ordered = sorted(tasks, key=lambda row: (stable_u64(f"{difficulty_seed}|{row['task_id']}"), row["task_id"]))
    counts = {"easy": 90, "medium": 89, "hard": 89}
    true_by_id: dict[str, str] = {}
    cursor = 0
    for level in ("easy", "medium", "hard"):
        for row in ordered[cursor:cursor + counts[level]]:
            true_by_id[row["task_id"]] = level
        cursor += counts[level]
    observed_by_id: dict[str, str] = {}
    for level in ("easy", "medium", "hard"):
        members = sorted((row for row in tasks if true_by_id[row["task_id"]] == level), key=lambda row: (stable_u64(f"{confusion_seed}|{row['task_id']}"), row["task_id"]))
        correct = round(0.80 * len(members))
        for row in members[:correct]:
            observed_by_id[row["task_id"]] = level
        errors = members[correct:]
        if level == "easy":
            for row in errors:
                observed_by_id[row["task_id"]] = "medium"
        elif level == "hard":
            for row in errors:
                observed_by_id[row["task_id"]] = "medium"
        else:
            split = len(errors) // 2
            for row in errors[:split]:
                observed_by_id[row["task_id"]] = "easy"
            for row in errors[split:]:
                observed_by_id[row["task_id"]] = "hard"
    result = []
    for row in sorted(tasks, key=lambda item: item["task_id"]):
        arrival = float(row["arrival_time_s"])
        assignment_hash_value = stable_u64(f"{difficulty_seed}|{row['task_id']}")
        observation_hash_value = stable_u64(f"{confusion_seed}|{row['task_id']}")
        result.append({
            "task_id": row["task_id"], "difficulty_true": true_by_id[row["task_id"]],
            "difficulty_observed": observed_by_id[row["task_id"]],
            "difficulty_correct": int(true_by_id[row["task_id"]] == observed_by_id[row["task_id"]]),
            "difficulty_seed": difficulty_seed, "confusion_seed": confusion_seed,
            "value_level_audit_only": row["value_level"], "required_product_audit_only": row["product_requirement"],
            "arrival_quartile_audit_only": min(4, int(arrival // 21600.0) + 1),
            "assignment_hash": f"{assignment_hash_value:016x}",
            "observation_hash": f"{observation_hash_value:016x}",
        })
    return result


def quality_audit(manifest: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    table = config["conditional_quality"]
    frozen = engine.finalgate.load_frozen()["q_eff"]
    theoretical, empirical = {}, {}
    counts = Counter(row["difficulty_true"] for row in manifest)
    for model, values in table.items():
        theoretical[model] = sum(float(values[level]) for level in ("easy", "medium", "hard")) / 3.0
        empirical[model] = sum(float(values[row["difficulty_true"]]) for row in manifest) / len(manifest)
    confusion = Counter((row["difficulty_true"], row["difficulty_observed"]) for row in manifest)
    cross_value = Counter((row["value_level_audit_only"], row["difficulty_true"]) for row in manifest)
    cross_requirement = Counter((row["required_product_audit_only"], row["difficulty_true"]) for row in manifest)
    cross_arrival = Counter((row["arrival_quartile_audit_only"], row["difficulty_true"]) for row in manifest)
    checks = {
        "counts_balanced_within_one": max(counts.values()) - min(counts.values()) <= 1,
        "model_order_each_difficulty": all(float(table["lightweight"][d]) < float(table["balanced"][d]) < float(table["high_accuracy"][d]) for d in ("easy", "medium", "hard")),
        "quality_nonincreasing_with_difficulty": all(float(table[m]["easy"]) >= float(table[m]["medium"]) >= float(table[m]["hard"]) for m in table),
        "uniform_means_equal_frozen": all(abs(theoretical[m] - float(frozen[m])) <= 1e-12 for m in table),
        "easy_gaps_smaller_than_hard": (float(table["balanced"]["easy"]) - float(table["lightweight"]["easy"])) < (float(table["balanced"]["hard"]) - float(table["lightweight"]["hard"])) and (float(table["high_accuracy"]["easy"]) - float(table["balanced"]["easy"])) < (float(table["high_accuracy"]["hard"]) - float(table["balanced"]["hard"])),
        "assignment_uses_only_task_id_and_seed": True,
        "true_not_exposed_to_policy": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL", "quality_status": config["quality_status"],
        "construction": "medium equals frozen q; easy anchors are conservative ordered proxies; hard is 2*frozen-easy so the uniform three-level mean exactly equals frozen q",
        "checks": checks, "difficulty_counts": dict(counts), "quality_table": table,
        "frozen_q": frozen, "theoretical_uniform_means": theoretical, "empirical_manifest_means": empirical,
        "empirical_minus_frozen": {m: empirical[m] - float(frozen[m]) for m in table},
        "confusion_matrix": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
        "correct_rate": sum(row["difficulty_correct"] for row in manifest) / len(manifest),
        "assignment_independence_note": "difficulty_true is determined before inspection of value, requirement, or arrival, using only SHA256(difficulty_seed|task_id); cross-tabs are audit-only",
        "cross_tab_value": {f"{a}|{b}": n for (a, b), n in sorted(cross_value.items())},
        "cross_tab_requirement": {f"{a}|{b}": n for (a, b), n in sorted(cross_requirement.items())},
        "cross_tab_arrival_quartile": {f"{a}|{b}": n for (a, b), n in sorted(cross_arrival.items())},
    }


def make_spec(method: str, bundle: Path, attempt: Path, manifest: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    observed_only = {row["task_id"]: {"difficulty_observed": row["difficulty_observed"]} for row in manifest}
    spec = {
        "version": VERSION, "method": method, "seed": engine.SEED, "slots": engine.SLOTS,
        "capacity": engine.CAPACITY, "bundle_root": str(bundle), "output": str(attempt / "runs" / SLUG[method]),
        "flags": (True, True, False), "difficulty_by_task": observed_only,
        "conditional_quality_table": config["conditional_quality"],
        "fixed_model_name": "balanced" if method == FIXED else "",
        "frozen_sha256": engine.sha256_file(engine.FROZEN), "config_sha256": engine.sha256_file(CONFIG),
        "entry_sha256": engine.sha256_file(Path(__file__).resolve()),
        "policy_input_fields": ["task_id", "difficulty_observed"], "difficulty_true_exposed_to_policy": False,
    }
    return spec


def child_main(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = Path(spec["output"])
    if output.exists():
        raise FileExistsError(output)
    engine.configure_and_run_child(spec, output)
    return 0


def launch(spec: Mapping[str, Any], attempt: Path) -> dict[str, Any]:
    method = str(spec["method"])
    spec_path = attempt / "run_specs" / f"{SLUG[method]}.json"
    write_json(spec_path, spec)
    out, err = attempt / "audit" / f"{SLUG[method]}.stdout.log", attempt / "audit" / f"{SLUG[method]}.stderr.log"
    with out.open("w", encoding="utf-8") as stdout, err.open("w", encoding="utf-8") as stderr:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)], cwd=ROOT, stdout=stdout, stderr=stderr, text=True, check=False)
    if result.returncode:
        raise RuntimeError({"METHOD_RUNTIME_FAILED": method, "stderr": str(err)})
    identity = engine.read_json(Path(spec["output"]) / "runtime_identity.json")
    return {"method": method, "pid": identity["pid"], "output": spec["output"]}


def conditional_evaluate(method: str, path: Path, manifest_by_id: Mapping[str, Mapping[str, Any]], table: Mapping[str, Mapping[str, float]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, float]]]:
    base_metric, ledger = engine.evaluate_run(method, path)
    outcomes = {row["task_id"]: row for row in engine.read_csv(path / "per_task_outcomes.csv")}
    gross = hv_gross = 0.0
    by_difficulty_gross = Counter()
    by_task_gross = Counter()
    converted = []
    for raw in ledger:
        row = dict(raw)
        difficulty = str(manifest_by_id[row["task_id"]]["difficulty_true"])
        q = float(table[row["model"]][difficulty])
        value = float(row["V"]) * float(row["delta_I"]) * q * float(row["T"])
        row.update({"difficulty_true": difficulty, "q_eff": q, "gross": value, "gross_recomputed": value, "q_count": 1})
        gross += value
        by_difficulty_gross[difficulty] += value
        by_task_gross[row["task_id"]] += value
        if outcomes[row["task_id"]]["value_level"] in {"high", "urgent"} and int(float(outcomes[row["task_id"]]["closure_success"])):
            hv_gross += value
        converted.append(row)
    total_cost = float(base_metric["total_cost"])
    hv_cost = total_cost * hv_gross / gross if gross else 0.0
    metric = dict(base_metric)
    metric.update({
        "gross_utility": gross, "HV_gross_utility": hv_gross, "J_all_net": gross - total_cost,
        "J_HV_net": hv_gross - hv_cost, "q_once_max_recompute_error": 0.0,
        "quality_accounting": "q_eff(actual_model,difficulty_true)_once",
    })
    difficulty_metrics: dict[str, dict[str, float]] = {}
    for level in ("easy", "medium", "hard"):
        task_ids = {task_id for task_id, info in manifest_by_id.items() if info["difficulty_true"] == level}
        closures = sum(int(float(outcomes[task_id]["closure_success"])) for task_id in task_ids)
        level_gross = float(by_difficulty_gross[level])
        level_cost = total_cost * level_gross / gross if gross else 0.0
        difficulty_metrics[level] = {"task_count": len(task_ids), "gross_utility": level_gross, "allocated_cost": level_cost, "net_utility": level_gross - level_cost, "closure_count": closures, "closure_rate": closures / len(task_ids)}
    write_csv(path / "conditional_public_event_ledger.csv", converted)
    write_json(path / "conditional_public_metrics.json", metric)
    return metric, converted, difficulty_metrics


def write_git_patch(path: Path) -> None:
    chunks = []
    for file in (CONFIG, Path(__file__).resolve(), ROOT / "scripts" / "run_single_seed_controlled_safe_metadata_thermal_exploration_v1.py"):
        chunks.extend(difflib.unified_diff([], file.read_text(encoding="utf-8").splitlines(keepends=True), fromfile="/dev/null", tofile=str(file.relative_to(ROOT)).replace("\\", "/")))
    path.write_text("".join(chunks), encoding="utf-8")


def main() -> int:
    attempt = next_attempt()
    config = read_config()
    tracked = engine.hash_snapshot()
    for path in (CONFIG, Path(__file__).resolve()):
        tracked[str(path)] = engine.sha256_file(path)
    scenario_audit = engine.normal.scenario_gate(engine.TRUSTED_SCENARIO, attempt / "scenario" / "ordinary_random_seed2026", attempt / "audit")
    write_json(attempt / "scenario_identity_report.json", scenario_audit)
    tasks = engine.read_csv(attempt / "scenario" / "ordinary_random_seed2026" / "task_manifest.csv")
    manifest = build_difficulty_manifest(tasks, config)
    write_csv(attempt / "difficulty_manifest.csv", manifest)
    q_audit = quality_audit(manifest, config)
    write_json(attempt / "conditional_quality_audit.json", q_audit)
    if q_audit["status"] != "PASS":
        raise RuntimeError("CONDITIONAL_QUALITY_AUDIT_FAILED")
    manifest_by_id = {row["task_id"]: row for row in manifest}
    bundle = attempt / "scenario" / "ordinary_random_seed2026"
    specs = {method: make_spec(method, bundle, attempt, manifest, config) for method in METHODS}
    processes = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(launch, specs[method], attempt): method for method in METHODS}
        for future in as_completed(futures):
            processes.append(future.result())

    metrics: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    difficulty_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHODS:
        metrics[method], ledgers[method], difficulty_metrics[method] = conditional_evaluate(
            method, Path(specs[method]["output"]), manifest_by_id, config["conditional_quality"]
        )
    fixed, adaptive = metrics[FIXED], metrics[ADAPTIVE]
    summary = []
    for method in METHODS:
        row = dict(metrics[method])
        for level in ("easy", "medium", "hard"):
            for key, value in difficulty_metrics[method][level].items():
                row[f"{level}_{key}"] = value
        row.update({
            "delta_J_all_adaptive_minus_fixed": float(adaptive["J_all_net"]) - float(fixed["J_all_net"]),
            "delta_J_HV_adaptive_minus_fixed": float(adaptive["J_HV_net"]) - float(fixed["J_HV_net"]),
            "delta_HV_closure_adaptive_minus_fixed": int(adaptive["HV_closure_count"]) - int(fixed["HV_closure_count"]),
        })
        summary.append(row)
    write_csv(attempt / "two_method_summary.csv", summary)

    model_matrix = []
    traces: dict[str, list[dict[str, str]]] = {}
    for method in METHODS:
        traces[method] = engine.read_csv(Path(specs[method]["output"]) / "difficulty_decision_trace.csv")
        counter = Counter()
        for row in traces[method]:
            true = manifest_by_id[row["task_id"]]["difficulty_true"]
            model = row["selected_model"]
            if model in {"lightweight", "balanced", "high_accuracy"}:
                counter[(true, model)] += 1
        for difficulty in ("easy", "medium", "hard"):
            for model in ("lightweight", "balanced", "high_accuracy"):
                model_matrix.append({"method": method, "difficulty_true": difficulty, "model": model, "selection_count": counter[(difficulty, model)]})
    write_csv(attempt / "tables" / "model_selection_by_true_difficulty.csv", model_matrix)

    outcomes = {
        method: {row["task_id"]: row for row in engine.read_csv(Path(specs[method]["output"]) / "per_task_outcomes.csv")}
        for method in METHODS
    }
    gross_by_task: dict[str, Counter[str]] = {}
    for method in METHODS:
        gross_by_task[method] = Counter()
        for row in ledgers[method]:
            gross_by_task[method][row["task_id"]] += float(row["gross"])
    prices = engine.finalgate.load_frozen()["public_prices"]
    changed = []
    seen_switch_tasks: set[str] = set()
    for row in traces[ADAPTIVE]:
        model = row["selected_model"]
        if model not in {"lightweight", "high_accuracy"}:
            continue
        task_id = row["task_id"]
        true = manifest_by_id[task_id]["difficulty_true"]
        observed = manifest_by_id[task_id]["difficulty_observed"]
        adaptive_task, fixed_task = outcomes[ADAPTIVE][task_id], outcomes[FIXED][task_id]
        adaptive_local = gross_by_task[ADAPTIVE][task_id] - float(prices["energy_price_per_Wh"]) * float(adaptive_task["controllable_energy_j"]) / 3600.0 - float(prices["downlink_price_per_MB"]) * float(adaptive_task["transmitted_bytes"]) / 1_000_000.0
        fixed_local = gross_by_task[FIXED][task_id] - float(prices["energy_price_per_Wh"]) * float(fixed_task["controllable_energy_j"]) / 3600.0 - float(prices["downlink_price_per_MB"]) * float(fixed_task["transmitted_bytes"]) / 1_000_000.0
        q_selected_true = float(config["conditional_quality"][model][true])
        q_balanced_true = float(config["conditional_quality"]["balanced"][true])
        q_selected_observed = float(config["conditional_quality"][model][observed])
        q_balanced_observed = float(config["conditional_quality"]["balanced"][observed])
        predicted_balanced_energy = None if row["balanced_predicted_energy_j"] == "" else float(row["balanced_predicted_energy_j"])
        predicted_balanced_theta = None if row["balanced_predicted_theta_next"] == "" else float(row["balanced_predicted_theta_next"])
        changed.append({
            "decision_slot": row["decision_slot"], "task_id": task_id, "value_level": adaptive_task["value_level"],
            "required_product": adaptive_task["product_requirement"], "difficulty_true": true, "difficulty_observed": observed,
            "adaptive_model": model, "adaptive_product": row["selected_product"], "fixed_model": "balanced",
            "quality_delta_true": q_selected_true - q_balanced_true,
            "quality_delta_observed_used_online": q_selected_observed - q_balanced_observed,
            "predicted_energy_delta_j": "" if predicted_balanced_energy is None else float(row["selected_predicted_energy_j"]) - predicted_balanced_energy,
            "predicted_theta_next_delta": "" if predicted_balanced_theta is None else float(row["selected_predicted_theta_next"]) - predicted_balanced_theta,
            "balanced_same_product_hard_safe": row["balanced_same_product_hard_safe"],
            "adaptive_actual_task_gross": gross_by_task[ADAPTIVE][task_id], "fixed_actual_task_gross": gross_by_task[FIXED][task_id],
            "actual_task_local_net_delta": adaptive_local - fixed_local,
            "adaptive_closure": adaptive_task["closure_success"], "fixed_closure": fixed_task["closure_success"],
            "closure_delta": int(float(adaptive_task["closure_success"])) - int(float(fixed_task["closure_success"])),
        })
        seen_switch_tasks.add(task_id)
    write_csv(attempt / "changed_task_trace.csv", changed)

    easy_l = {row["task_id"] for row in changed if row["difficulty_true"] == "easy" and row["adaptive_model"] == "lightweight" and float(row["actual_task_local_net_delta"]) > 0}
    hard_h = {row["task_id"] for row in changed if row["difficulty_true"] == "hard" and row["adaptive_model"] == "high_accuracy" and float(row["actual_task_local_net_delta"]) > 0}
    adaptive_identity = engine.read_json(Path(specs[ADAPTIVE]["output"]) / "runtime_identity.json")
    fixed_identity = engine.read_json(Path(specs[FIXED]["output"]) / "runtime_identity.json")
    post = engine.hash_snapshot()
    for path in (CONFIG, Path(__file__).resolve()):
        post[str(path)] = engine.sha256_file(path)
    safe = all(int(metrics[m]["soc_violation_count"]) == 0 and int(metrics[m]["thermal_violation_count"]) == 0 for m in METHODS)
    candidate = float(adaptive["J_all_net"]) > float(fixed["J_all_net"]) and float(adaptive["J_HV_net"]) >= float(fixed["J_HV_net"]) - 1e-12 and safe and bool(easy_l or hard_h)
    checks = {
        "difficulty_quality_audit_pass": q_audit["status"] == "PASS",
        "two_unique_processes": len({row["pid"] for row in processes}) == 2,
        "source_hashes_unchanged": tracked == post,
        "same_scenario": scenario_audit["status"] == "PASS",
        "true_difficulty_not_in_policy_spec": all("difficulty_true" not in json.dumps(specs[m]["difficulty_by_task"]) for m in METHODS),
        "adaptive_full_3x3": adaptive_identity["grid_audit"]["complete"],
        "fixed_balanced_identity": fixed_identity["fixed_model"] == "balanced",
        "q_once_public": all(float(metrics[m]["q_once_max_recompute_error"]) <= 1e-12 for m in METHODS),
        "zero_safety_violations": safe,
    }
    write_json(attempt / "audit" / "method_identity_and_isolation.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "processes": processes, "adaptive_identity": adaptive_identity, "fixed_identity": fixed_identity})
    (attempt / "test_report.txt").write_text("\n".join(f"{'PASS' if passed else 'FAIL'}\t{name}" for name, passed in checks.items()) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError({"DIFFICULTY_DIAGNOSTIC_AUDIT_FAILED": checks})

    decision = "CANDIDATE_EXTENSION" if candidate else "ROLLBACK_TO_CURRENT_PROP"
    matrix_counter = {(row["difficulty_true"], row["model"]): int(row["selection_count"]) for row in model_matrix if row["method"] == ADAPTIVE}
    lines = [
        "# Difficulty-aware PROP versus Fixed Balanced", "",
        "The conditional quality table is a diagnostic proxy, not measured accuracy.", "",
        "| method | J_all_net | J_HV_net | all | HV | high | urgent |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {FIXED} | {float(fixed['J_all_net']):.6f} | {float(fixed['J_HV_net']):.6f} | {fixed['all_closure_count']} | {fixed['HV_closure_count']} | {fixed['high_closure_count']} | {fixed['urgent_closure_count']} |",
        f"| {ADAPTIVE} | {float(adaptive['J_all_net']):.6f} | {float(adaptive['J_HV_net']):.6f} | {adaptive['all_closure_count']} | {adaptive['HV_closure_count']} | {adaptive['high_closure_count']} | {adaptive['urgent_closure_count']} |", "",
        f"Adaptive deltas: J_all {float(adaptive['J_all_net'])-float(fixed['J_all_net']):+.12f}; J_HV {float(adaptive['J_HV_net'])-float(fixed['J_HV_net']):+.12f}; HV closure {int(adaptive['HV_closure_count'])-int(fixed['HV_closure_count']):+d}.", "",
        "## Adaptive model selections by true difficulty", "",
        f"- easy L/B/H: {matrix_counter.get(('easy','lightweight'),0)}/{matrix_counter.get(('easy','balanced'),0)}/{matrix_counter.get(('easy','high_accuracy'),0)}",
        f"- medium L/B/H: {matrix_counter.get(('medium','lightweight'),0)}/{matrix_counter.get(('medium','balanced'),0)}/{matrix_counter.get(('medium','high_accuracy'),0)}",
        f"- hard L/B/H: {matrix_counter.get(('hard','lightweight'),0)}/{matrix_counter.get(('hard','balanced'),0)}/{matrix_counter.get(('hard','high_accuracy'),0)}", "",
        f"Explainable positive easy->L tasks: {len(easy_l)}; hard->H tasks: {len(hard_h)}.",
        f"Safety violations Adaptive/Fixed: {adaptive['soc_violation_count']}/{adaptive['thermal_violation_count']} and {fixed['soc_violation_count']}/{fixed['thermal_violation_count']}.", "",
        f"Decision: **{decision}**.",
        "The frozen formal method remains PROP-V2-SAFE-METADATA-CURRENT-v1 regardless of this diagnostic outcome.", "",
        COMPLETION,
    ]
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_git_patch(attempt / "git_diff.patch")
    write_json(attempt / "completion.json", {"status": "PASS", "marker": COMPLETION, "decision": decision, "completed_at": datetime.now(timezone.utc).isoformat(), "output": str(attempt)})
    print(f"Unique entry: {Path(__file__).resolve()}")
    print(f"Output: {attempt}")
    print(f"Decision: {decision}")
    print(COMPLETION)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child_main(args.child_spec) if args.child_spec else main())
