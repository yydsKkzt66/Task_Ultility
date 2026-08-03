"""One-click V1 + V2 controlled single-seed experiment.

Run with the IDE button or ``python scripts/run_v1_v2_combined_single_seed_v1.py``.
Private child arguments are used only for process isolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as base  # noqa: E402


VERSION = "V1-V2-COMBINED-SINGLE-SEED-v1"
COMPLETION = "V1_V2_COMBINED_SINGLE_SEED_COMPLETE"
CONFIG = ROOT / "configs" / "v1_v2_combined_single_seed_v1.yaml"
OUTPUT_BASE = ROOT / "outputs" / "_v1_v2_combined_single_seed_v1"
V0 = base.V0
V1 = base.V1
V2 = base.V2
V12 = "V12-SAFE-CONSISTENT-HV-METADATA-REPAIR"
METHODS = (V0, V1, V2, V12)
FLAGS = {V0: (False, False, False), V1: (True, False, False), V2: (False, True, False), V12: (True, True, False)}
SLUG = {method: method.replace("-", "_").replace("+", "_") for method in METHODS}


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
    number = 1
    while (OUTPUT_BASE / f"attempt_{number:03d}").exists():
        number += 1
    path = OUTPUT_BASE / f"attempt_{number:03d}"
    for name in ("audit", "runs", "run_specs", "scenario", "tables"):
        (path / name).mkdir(parents=True, exist_ok=name != "audit")
    return path


def make_spec(method: str, bundle: Path, attempt: Path) -> dict[str, Any]:
    return {
        "version": VERSION, "method": method, "seed": base.SEED, "slots": base.SLOTS,
        "capacity": base.CAPACITY, "bundle_root": str(bundle), "output": str(attempt / "runs" / SLUG[method]),
        "flags": FLAGS[method], "frozen_sha256": base.sha256_file(base.FROZEN),
        "config_sha256": base.sha256_file(CONFIG), "entry_sha256": base.sha256_file(Path(__file__).resolve()),
    }


def child_main(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = Path(spec["output"])
    if output.exists():
        raise FileExistsError(output)
    base.configure_and_run_child(spec, output)
    return 0


def launch(spec: Mapping[str, Any], attempt: Path) -> dict[str, Any]:
    spec_path = attempt / "run_specs" / f"{SLUG[str(spec['method'])]}.json"
    write_json(spec_path, spec)
    stdout_path = attempt / "audit" / f"{SLUG[str(spec['method'])]}.stdout.log"
    stderr_path = attempt / "audit" / f"{SLUG[str(spec['method'])]}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)],
            cwd=ROOT, stdout=stdout, stderr=stderr, text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError({"METHOD_RUNTIME_FAILED": spec["method"], "stderr": str(stderr_path)})
    identity = base.read_json(Path(spec["output"]) / "runtime_identity.json")
    return {"method": spec["method"], "pid": identity["pid"], "output": spec["output"]}


def paired_changes(attempt: Path) -> list[dict[str, Any]]:
    def outcomes(method: str) -> dict[str, dict[str, str]]:
        return {row["task_id"]: row for row in base.read_csv(attempt / "runs" / SLUG[method] / "per_task_outcomes.csv")}

    def sequences(method: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for row in base.read_csv(attempt / "runs" / SLUG[method] / "v2_decision_trace.csv"):
            result[row["task_id"]].append(row["selected_action"])
        return result

    old, new = outcomes(V0), outcomes(V12)
    old_seq, new_seq = sequences(V0), sequences(V12)
    rows = []
    for task_id in sorted(old):
        if old_seq.get(task_id, []) == new_seq.get(task_id, []) and old[task_id]["closure_success"] == new[task_id]["closure_success"]:
            continue
        rows.append({
            "task_id": task_id, "value_level": old[task_id]["value_level"],
            "required_product": old[task_id]["product_requirement"],
            "V0_action_sequence": ";".join(old_seq.get(task_id, [])),
            "V12_action_sequence": ";".join(new_seq.get(task_id, [])),
            "V0_closure": old[task_id]["closure_success"], "V12_closure": new[task_id]["closure_success"],
            "closure_change": int(float(new[task_id]["closure_success"])) - int(float(old[task_id]["closure_success"])),
            "V0_transmitted_bytes": old[task_id]["transmitted_bytes"], "V12_transmitted_bytes": new[task_id]["transmitted_bytes"],
            "V0_energy_j": old[task_id]["controllable_energy_j"], "V12_energy_j": new[task_id]["controllable_energy_j"],
        })
    return rows


def main() -> int:
    attempt = next_attempt()
    pre_hash = base.hash_snapshot()
    pre_hash[str(CONFIG)] = base.sha256_file(CONFIG)
    pre_hash[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__).resolve())
    write_json(attempt / "audit" / "pre_run_hashes.json", pre_hash)
    scenario_audit = base.normal.scenario_gate(
        base.TRUSTED_SCENARIO, attempt / "scenario" / "ordinary_random_seed2026", attempt / "audit"
    )
    write_json(attempt / "scenario_identity_report.json", scenario_audit)
    bundle = attempt / "scenario" / "ordinary_random_seed2026"
    specs = {method: make_spec(method, bundle, attempt) for method in METHODS}

    processes = [launch(specs[V0], attempt)]
    metrics: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    metrics[V0], ledgers[V0] = base.evaluate_run(V0, Path(specs[V0]["output"]))
    regression = base.v0_regression(metrics[V0], Path(specs[V0]["output"]))
    write_json(attempt / "audit" / "V0_regression.json", regression)
    if regression["status"] != "PASS":
        raise RuntimeError("V0_BASELINE_REGRESSION_MISMATCH")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(launch, specs[method], attempt): method for method in (V1, V2, V12)}
        for future in as_completed(futures):
            processes.append(future.result())
    for method in (V1, V2, V12):
        metrics[method], ledgers[method] = base.evaluate_run(method, Path(specs[method]["output"]))

    summary = []
    reference = metrics[V0]
    for method in METHODS:
        row = dict(metrics[method])
        row.update({
            "ordinary_closure_count": int(row["all_closure_count"]) - int(row["HV_closure_count"]),
            "delta_J_all_vs_V0": float(row["J_all_net"]) - float(reference["J_all_net"]),
            "delta_J_HV_vs_V0": float(row["J_HV_net"]) - float(reference["J_HV_net"]),
            "delta_all_closure_vs_V0": int(row["all_closure_count"]) - int(reference["all_closure_count"]),
            "delta_HV_closure_vs_V0": int(row["HV_closure_count"]) - int(reference["HV_closure_count"]),
            "delta_high_closure_vs_V0": int(row["high_closure_count"]) - int(reference["high_closure_count"]),
            "delta_urgent_closure_vs_V0": int(row["urgent_closure_count"]) - int(reference["urgent_closure_count"]),
        })
        summary.append(row)
    write_csv(attempt / "four_method_summary.csv", summary)
    write_csv(attempt / "tables" / "four_method_summary.csv", summary)
    write_csv(attempt / "paired_V0_V12_task_changes.csv", paired_changes(attempt))

    v12 = next(row for row in summary if row["method"] == V12)
    v1 = next(row for row in summary if row["method"] == V1)
    v2 = next(row for row in summary if row["method"] == V2)
    interaction_j = float(v12["delta_J_all_vs_V0"]) - float(v1["delta_J_all_vs_V0"]) - float(v2["delta_J_all_vs_V0"])
    interaction_hv = float(v12["delta_J_HV_vs_V0"]) - float(v1["delta_J_HV_vs_V0"]) - float(v2["delta_J_HV_vs_V0"])
    interaction_closure = int(v12["delta_HV_closure_vs_V0"]) - int(v1["delta_HV_closure_vs_V0"]) - int(v2["delta_HV_closure_vs_V0"])
    assessment = [
        "# V1 + V2 single-seed assessment", "",
        f"V0 exact regression: **{regression['status']}**, decision trace exact: **{regression['decision_trace_exact']}**.", "",
        f"V1+V2 J_all_net = {float(v12['J_all_net']):.12f} (delta {float(v12['delta_J_all_vs_V0']):+.12f}).",
        f"V1+V2 J_HV_net = {float(v12['J_HV_net']):.12f} (delta {float(v12['delta_J_HV_vs_V0']):+.12f}).",
        f"V1+V2 all/HV/high/urgent closures = {v12['all_closure_count']}/{v12['HV_closure_count']}/{v12['high_closure_count']}/{v12['urgent_closure_count']}.",
        f"V1+V2 ordinary closure count = {v12['ordinary_closure_count']}.",
        f"V1+V2 selector-final mismatch count = {v12['selector_final_guard_mismatch_count']}; SOC/thermal violations = {v12['soc_violation_count']}/{v12['thermal_violation_count']}.", "",
        "## Interaction beyond additive single-factor deltas", "",
        f"- J_all interaction: {interaction_j:+.12f}",
        f"- J_HV interaction: {interaction_hv:+.12f}",
        f"- HV-closure interaction: {interaction_closure:+d}", "",
        "The interaction is descriptive for this one deterministic seed only. No parameters were changed from the prior controlled experiment.", "",
        COMPLETION,
    ]
    (attempt / "final_assessment.md").write_text("\n".join(assessment) + "\n", encoding="utf-8")

    post_hash = base.hash_snapshot()
    post_hash[str(CONFIG)] = base.sha256_file(CONFIG)
    post_hash[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__).resolve())
    hash_pass = pre_hash == post_hash
    isolation_pass = len({row["pid"] for row in processes}) == 4
    write_json(attempt / "audit" / "identity_and_isolation.json", {
        "status": "PASS" if hash_pass and isolation_pass else "FAIL", "source_hashes_unchanged": hash_pass,
        "V0_gate_then_three_process_parallel": True, "unique_pid_count": len({row["pid"] for row in processes}), "processes": processes,
    })
    checks = {
        "V0_exact_regression": regression["status"] == "PASS", "source_hashes_unchanged": hash_pass,
        "four_unique_processes": isolation_pass,
        "zero_safety_violations": all(int(metrics[m]["soc_violation_count"]) == 0 and int(metrics[m]["thermal_violation_count"]) == 0 for m in METHODS),
        "V12_both_mechanisms_enabled": FLAGS[V12] == (True, True, False),
    }
    (attempt / "test_report.txt").write_text("\n".join(f"{'PASS' if passed else 'FAIL'}\t{name}" for name, passed in checks.items()) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError({"V1_V2_COMBINED_AUDIT_FAILED": checks})
    write_json(attempt / "completion.json", {"status": "PASS", "marker": COMPLETION, "completed_at": datetime.now(timezone.utc).isoformat(), "output": str(attempt)})
    print(f"Unique entry: {Path(__file__).resolve()}")
    print(f"Output: {attempt}")
    print("Execution: V0 gate, then V1/V2/V1+V2 in three independent parallel processes")
    print(COMPLETION)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child_main(args.child_spec) if args.child_spec else main())
