"""Read-only Oracle difficulty upper-bound diagnostic, seed 2026."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_difficulty_aware_prop_vs_fixed_balanced_single_seed_v1 as prior  # noqa: E402
import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


VERSION = "ORACLE-DIFFICULTY-MODEL-ADAPTATION-DIAGNOSTIC-v1"
COMPLETION = "ORACLE_DIFFICULTY_MODEL_ADAPTATION_DIAGNOSTIC_COMPLETE"
FIXED = "FIXED-B-DIFFICULTY"
NOISY = "ADAPTIVE-NOISY-DIFFICULTY"
ORACLE = "ADAPTIVE-ORACLE-DIFFICULTY"
METHODS = (FIXED, NOISY, ORACLE)
SLUG = {FIXED: "FIXED_B_DIFFICULTY", NOISY: "ADAPTIVE_NOISY_DIFFICULTY", ORACLE: "ADAPTIVE_ORACLE_DIFFICULTY"}
OUTPUT_BASE = ROOT / "outputs" / "_oracle_difficulty_model_adaptation_diagnostic_v1"
PRIOR_OUTPUT = ROOT / "outputs" / "_difficulty_aware_prop_vs_fixed_balanced_single_seed_v1" / "attempt_002"
PRIOR_MANIFEST = PRIOR_OUTPUT / "difficulty_manifest.csv"
PRIOR_QUALITY_AUDIT = PRIOR_OUTPUT / "conditional_quality_audit.json"
PRIOR_SUMMARY = PRIOR_OUTPUT / "two_method_summary.csv"


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


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    index = 1
    while (OUTPUT_BASE / f"attempt_{index:03d}").exists():
        index += 1
    path = OUTPUT_BASE / f"attempt_{index:03d}"
    for name in ("audit", "runs", "run_specs", "scenario", "tables", "frozen_inputs"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def make_spec(method: str, bundle: Path, attempt: Path, manifest: list[dict[str, str]], quality_table: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    if method == ORACLE:
        policy_difficulty = {row["task_id"]: {"difficulty_observed": row["difficulty_true"]} for row in manifest}
        input_semantics = "oracle_true_difficulty_explicitly_authorized"
    else:
        policy_difficulty = {row["task_id"]: {"difficulty_observed": row["difficulty_observed"]} for row in manifest}
        input_semantics = "noisy_observed_difficulty_only"
    return {
        "version": VERSION, "method": method, "seed": engine.SEED, "slots": engine.SLOTS,
        "capacity": engine.CAPACITY, "bundle_root": str(bundle), "output": str(attempt / "runs" / SLUG[method]),
        "flags": (True, True, False), "fixed_model_name": "balanced" if method == FIXED else "",
        "difficulty_by_task": policy_difficulty, "conditional_quality_table": quality_table,
        "difficulty_policy_input_semantics": input_semantics,
        "prior_manifest_sha256": engine.sha256_file(PRIOR_MANIFEST),
        "prior_quality_audit_sha256": engine.sha256_file(PRIOR_QUALITY_AUDIT),
        "entry_sha256": engine.sha256_file(Path(__file__).resolve()),
    }


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


def task_local_net(method: str, outcomes: Mapping[str, Mapping[str, str]], ledger: Iterable[Mapping[str, Any]], prices: Mapping[str, float]) -> dict[str, float]:
    gross = Counter()
    for row in ledger:
        gross[row["task_id"]] += float(row["gross"])
    result = {}
    for task_id, row in outcomes.items():
        result[task_id] = gross[task_id] - float(prices["energy_price_per_Wh"]) * float(row["controllable_energy_j"]) / 3600.0 - float(prices["downlink_price_per_MB"]) * float(row["transmitted_bytes"]) / 1_000_000.0
    return result


def indexed_trace(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    counts = Counter()
    output = {}
    for row in rows:
        task_id = row["task_id"]
        ordinal = counts[task_id]
        counts[task_id] += 1
        output[(task_id, ordinal)] = row
    return output


def main() -> int:
    if not all(path.is_file() for path in (PRIOR_MANIFEST, PRIOR_QUALITY_AUDIT, PRIOR_SUMMARY)):
        raise RuntimeError("PRIOR_DIFFICULTY_ASSET_MISSING")
    attempt = next_attempt()
    pre_hashes = engine.hash_snapshot()
    for path in (Path(__file__).resolve(), PRIOR_MANIFEST, PRIOR_QUALITY_AUDIT, prior.CONFIG):
        pre_hashes[str(path)] = engine.sha256_file(path)
    manifest = engine.read_csv(PRIOR_MANIFEST)
    quality_audit = engine.read_json(PRIOR_QUALITY_AUDIT)
    if quality_audit["status"] != "PASS" or len(manifest) != 268:
        raise RuntimeError("PRIOR_DIFFICULTY_ASSET_INVALID")
    write_csv(attempt / "frozen_inputs" / "difficulty_manifest.csv", manifest)
    write_json(attempt / "frozen_inputs" / "conditional_quality_audit.json", quality_audit)
    write_json(attempt / "audit" / "frozen_input_identity.json", {
        "status": "PASS", "source_manifest": str(PRIOR_MANIFEST), "source_quality_audit": str(PRIOR_QUALITY_AUDIT),
        "manifest_sha256": engine.sha256_file(PRIOR_MANIFEST), "quality_audit_sha256": engine.sha256_file(PRIOR_QUALITY_AUDIT),
        "manifest_semantic_hash": engine.stable_hash(manifest), "quality_table": quality_audit["quality_table"],
    })
    manifest_by_id = {row["task_id"]: row for row in manifest}
    scenario_audit = engine.normal.scenario_gate(engine.TRUSTED_SCENARIO, attempt / "scenario" / "ordinary_random_seed2026", attempt / "audit")
    write_json(attempt / "scenario_identity_report.json", scenario_audit)
    bundle = attempt / "scenario" / "ordinary_random_seed2026"
    specs = {method: make_spec(method, bundle, attempt, manifest, quality_audit["quality_table"]) for method in METHODS}
    processes = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(launch, specs[method], attempt): method for method in METHODS}
        for future in as_completed(futures):
            processes.append(future.result())

    metrics: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    difficulty_metrics: dict[str, dict[str, dict[str, float]]] = {}
    traces: dict[str, list[dict[str, str]]] = {}
    outcomes: dict[str, dict[str, dict[str, str]]] = {}
    for method in METHODS:
        run = Path(specs[method]["output"])
        metrics[method], ledgers[method], difficulty_metrics[method] = prior.conditional_evaluate(method, run, manifest_by_id, quality_audit["quality_table"])
        traces[method] = engine.read_csv(run / "difficulty_decision_trace.csv")
        outcomes[method] = {row["task_id"]: row for row in engine.read_csv(run / "per_task_outcomes.csv")}

    fixed, noisy, oracle = metrics[FIXED], metrics[NOISY], metrics[ORACLE]
    summary = []
    for method in METHODS:
        row = dict(metrics[method])
        for level in ("easy", "medium", "hard"):
            for key, value in difficulty_metrics[method][level].items():
                row[f"{level}_{key}"] = value
        row.update({
            "delta_J_all_vs_fixed": float(row["J_all_net"]) - float(fixed["J_all_net"]),
            "delta_J_HV_vs_fixed": float(row["J_HV_net"]) - float(fixed["J_HV_net"]),
            "delta_HV_closure_vs_fixed": int(row["HV_closure_count"]) - int(fixed["HV_closure_count"]),
            "delta_J_all_vs_noisy": float(row["J_all_net"]) - float(noisy["J_all_net"]),
            "delta_J_HV_vs_noisy": float(row["J_HV_net"]) - float(noisy["J_HV_net"]),
        })
        summary.append(row)
    write_csv(attempt / "three_method_summary.csv", summary)

    prices = engine.finalgate.load_frozen()["public_prices"]
    local_net = {method: task_local_net(method, outcomes[method], ledgers[method], prices) for method in METHODS}
    noisy_index, oracle_index = indexed_trace(traces[NOISY]), indexed_trace(traces[ORACLE])
    corrected = []
    for key in sorted(set(noisy_index) | set(oracle_index)):
        n, o = noisy_index.get(key), oracle_index.get(key)
        task_id, ordinal = key
        if n is not None and o is not None and n["selected_model"] == o["selected_model"] and n["selected_product"] == o["selected_product"]:
            continue
        corrected.append({
            "task_id": task_id, "decision_ordinal": ordinal, "difficulty_true": manifest_by_id[task_id]["difficulty_true"],
            "difficulty_observed": manifest_by_id[task_id]["difficulty_observed"],
            "noisy_model": "" if n is None else n["selected_model"], "oracle_model": "" if o is None else o["selected_model"],
            "noisy_product": "" if n is None else n["selected_product"], "oracle_product": "" if o is None else o["selected_product"],
            "oracle_corrected_model_choice": int(n is not None and o is not None and n["selected_model"] != o["selected_model"]),
            "noisy_task_local_net": local_net[NOISY][task_id], "oracle_task_local_net": local_net[ORACLE][task_id],
            "oracle_minus_noisy_task_local_net": local_net[ORACLE][task_id] - local_net[NOISY][task_id],
            "noisy_closure": outcomes[NOISY][task_id]["closure_success"], "oracle_closure": outcomes[ORACLE][task_id]["closure_success"],
        })
    write_csv(attempt / "oracle_corrected_choices.csv", corrected)

    oracle_vs_fixed = []
    for row in traces[ORACLE]:
        if row["selected_model"] not in {"lightweight", "high_accuracy"}:
            continue
        task_id = row["task_id"]
        true = manifest_by_id[task_id]["difficulty_true"]
        model = row["selected_model"]
        balanced_energy = None if row["balanced_predicted_energy_j"] == "" else float(row["balanced_predicted_energy_j"])
        balanced_theta = None if row["balanced_predicted_theta_next"] == "" else float(row["balanced_predicted_theta_next"])
        oracle_vs_fixed.append({
            "decision_slot": row["decision_slot"], "task_id": task_id, "value_level": outcomes[ORACLE][task_id]["value_level"],
            "required_product": outcomes[ORACLE][task_id]["product_requirement"], "difficulty_true": true,
            "oracle_model": model, "oracle_product": row["selected_product"], "fixed_model": "balanced",
            "quality_delta_true": float(quality_audit["quality_table"][model][true]) - float(quality_audit["quality_table"]["balanced"][true]),
            "predicted_energy_delta_j": "" if balanced_energy is None else float(row["selected_predicted_energy_j"]) - balanced_energy,
            "predicted_theta_next_delta": "" if balanced_theta is None else float(row["selected_predicted_theta_next"]) - balanced_theta,
            "balanced_same_product_hard_safe": row["balanced_same_product_hard_safe"],
            "oracle_task_local_net": local_net[ORACLE][task_id], "fixed_task_local_net": local_net[FIXED][task_id],
            "task_local_final_net_delta": local_net[ORACLE][task_id] - local_net[FIXED][task_id],
            "system_final_J_all_delta_oracle_minus_fixed": float(oracle["J_all_net"]) - float(fixed["J_all_net"]),
            "oracle_closure": outcomes[ORACLE][task_id]["closure_success"], "fixed_closure": outcomes[FIXED][task_id]["closure_success"],
        })
    write_csv(attempt / "oracle_vs_fixed_b_task_trace.csv", oracle_vs_fixed)

    model_matrix = Counter()
    for method in METHODS:
        for row in traces[method]:
            if row["selected_model"] in {"lightweight", "balanced", "high_accuracy"}:
                model_matrix[(method, manifest_by_id[row["task_id"]]["difficulty_true"], row["selected_model"])] += 1
    decision_hashes = {method: engine.decision_semantic_hash(Path(specs[method]["output"])) for method in METHODS}
    audit_lines = [
        "# Difficulty model-selection audit", "",
        f"Manifest SHA256: `{engine.sha256_file(PRIOR_MANIFEST)}`", "",
        "| method | difficulty | L | B | H |",
        "|---|---|---:|---:|---:|",
    ]
    for method in METHODS:
        for level in ("easy", "medium", "hard"):
            audit_lines.append(f"| {method} | {level} | {model_matrix[(method,level,'lightweight')]} | {model_matrix[(method,level,'balanced')]} | {model_matrix[(method,level,'high_accuracy')]} |")
    audit_lines.extend(["", "## Decision hashes", ""])
    audit_lines.extend(f"- {method}: `{decision_hashes[method]}`" for method in METHODS)
    audit_lines.extend(["", "`task_local_final_net_delta` is a task-local accounting diagnostic. Because actions alter later shared state, it is not claimed as an additive causal decomposition of system J.", ""])
    (attempt / "difficulty_model_selection_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")

    fixed_pass = lambda metric: float(metric["J_all_net"]) > float(fixed["J_all_net"]) + 1e-12 and float(metric["J_HV_net"]) >= float(fixed["J_HV_net"]) - 1e-12
    if not fixed_pass(oracle):
        conclusion = "MODEL_ADAPTATION_STRUCTURALLY_UNSUPPORTED"
    elif not fixed_pass(noisy):
        conclusion = "DIFFICULTY_ESTIMATION_IS_THE_BOTTLENECK"
    else:
        conclusion = "DIFFICULTY_AWARE_MODEL_ADAPTATION_SUPPORTED"

    prior_rows = {row["method"]: row for row in engine.read_csv(PRIOR_SUMMARY)}
    reproducible = all(abs(float(noisy[key]) - float(prior_rows[prior.ADAPTIVE][key])) <= 1e-10 for key in ("J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "controllable_energy_Wh", "transmitted_MB")) and all(abs(float(fixed[key]) - float(prior_rows[prior.FIXED][key])) <= 1e-10 for key in ("J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "controllable_energy_Wh", "transmitted_MB"))
    post_hashes = engine.hash_snapshot()
    for path in (Path(__file__).resolve(), PRIOR_MANIFEST, PRIOR_QUALITY_AUDIT, prior.CONFIG):
        post_hashes[str(path)] = engine.sha256_file(path)
    identities = {method: engine.read_json(Path(specs[method]["output"]) / "runtime_identity.json") for method in METHODS}
    checks = {
        "prior_fixed_and_noisy_exactly_reproduced": reproducible,
        "three_unique_processes": len({row["pid"] for row in processes}) == 3,
        "source_and_frozen_inputs_unchanged": pre_hashes == post_hashes,
        "same_scenario": scenario_audit["status"] == "PASS",
        "fixed_balanced_identity": identities[FIXED]["fixed_model"] == "balanced",
        "noisy_uses_observed_only": specs[NOISY]["difficulty_policy_input_semantics"] == "noisy_observed_difficulty_only",
        "oracle_uses_true_as_only_difficulty_input": specs[ORACLE]["difficulty_policy_input_semantics"] == "oracle_true_difficulty_explicitly_authorized",
        "q_once_public": all(float(metrics[m]["q_once_max_recompute_error"]) <= 1e-12 for m in METHODS),
        "zero_safety_violations": all(int(metrics[m]["soc_violation_count"]) == 0 and int(metrics[m]["thermal_violation_count"]) == 0 for m in METHODS),
        "decision_hashes_recorded": all(
            isinstance(value, str) and len(value) == 64
            for value in decision_hashes.values()
        ),
    }
    write_json(attempt / "audit" / "method_identity_safety_hash_audit.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "processes": processes, "decision_hashes": decision_hashes, "identities": identities})
    (attempt / "test_report.txt").write_text("\n".join(f"{'PASS' if passed else 'FAIL'}\t{name}" for name, passed in checks.items()) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError({"ORACLE_DIAGNOSTIC_AUDIT_FAILED": checks})

    lines = [
        "# Oracle difficulty upper-bound diagnostic", "",
        "The prior manifest and conditional quality table were reused byte-for-byte; no parameter was changed.", "",
        "| method | J_all_net | J_HV_net | all | HV | high | urgent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metric = metrics[method]
        lines.append(f"| {method} | {float(metric['J_all_net']):.6f} | {float(metric['J_HV_net']):.6f} | {metric['all_closure_count']} | {metric['HV_closure_count']} | {metric['high_closure_count']} | {metric['urgent_closure_count']} |")
    lines.extend([
        "", f"Oracle minus Noisy: J_all {float(oracle['J_all_net'])-float(noisy['J_all_net']):+.12f}; J_HV {float(oracle['J_HV_net'])-float(noisy['J_HV_net']):+.12f}.",
        f"Oracle minus Fixed-B: J_all {float(oracle['J_all_net'])-float(fixed['J_all_net']):+.12f}; J_HV {float(oracle['J_HV_net'])-float(fixed['J_HV_net']):+.12f}.", "",
        f"Conclusion: **{conclusion}**.",
        "The formal method remains PROP-V2-SAFE-METADATA-CURRENT-v1; this directory can be deleted without changing it.", "",
        COMPLETION,
    ])
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(attempt / "completion.json", {"status": "PASS", "marker": COMPLETION, "conclusion": conclusion, "completed_at": datetime.now(timezone.utc).isoformat(), "output": str(attempt)})
    print(f"Unique entry: {Path(__file__).resolve()}")
    print(f"Output: {attempt}")
    print(f"Conclusion: {conclusion}")
    print(COMPLETION)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child_main(args.child_spec) if args.child_spec else main())
