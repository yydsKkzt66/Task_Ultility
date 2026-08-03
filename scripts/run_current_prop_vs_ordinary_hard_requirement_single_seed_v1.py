"""One-click current PROP versus ordinary hard-requirement granularity."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


VERSION = "CURRENT-PROP-ORDINARY-HARD-REQUIREMENT-SINGLE-SEED-v1"
COMPLETION = "CURRENT_PROP_VS_ORDINARY_HARD_REQUIREMENT_SINGLE_SEED_COMPLETE"
CURRENT = "PROP-V2-SAFE-METADATA-CURRENT-v1"
HARD = "ORDINARY-HARD-REQUIREMENT-GRANULARITY"
METHODS = (CURRENT, HARD)
SLUG = {CURRENT: "CURRENT_PROP", HARD: "ORDINARY_HARD_REQUIREMENT"}
CONFIG = ROOT / "configs" / "current_prop_ordinary_hard_requirement_single_seed_v1.yaml"
OUTPUT_BASE = ROOT / "outputs" / "_current_prop_vs_ordinary_hard_requirement_single_seed_v1"
REFERENCE_CURRENT = ROOT / "outputs" / "_current_prop_v1_v2_ablation_single_seed_v1" / "attempt_001" / "runs" / "PROP_V2_SAFE_METADATA_CURRENT_v1"


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
    for name in ("audit", "runs", "run_specs", "scenario"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def make_spec(method: str, bundle: Path, attempt: Path) -> dict[str, Any]:
    return {
        "version": VERSION, "method": method, "seed": engine.SEED, "slots": engine.SLOTS,
        "capacity": engine.CAPACITY, "bundle_root": str(bundle), "output": str(attempt / "runs" / SLUG[method]),
        "flags": (True, True, False), "ordinary_hard_requirement": method == HARD,
        "frozen_sha256": engine.sha256_file(engine.FROZEN), "config_sha256": engine.sha256_file(CONFIG),
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
    out = attempt / "audit" / f"{SLUG[method]}.stdout.log"
    err = attempt / "audit" / f"{SLUG[method]}.stderr.log"
    with out.open("w", encoding="utf-8") as stdout, err.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)], cwd=ROOT, stdout=stdout, stderr=stderr, text=True, check=False)
    if completed.returncode:
        raise RuntimeError({"METHOD_RUNTIME_FAILED": method, "stderr": str(err)})
    identity = engine.read_json(Path(spec["output"]) / "runtime_identity.json")
    return {"method": method, "pid": identity["pid"], "output": spec["output"]}


def main() -> int:
    attempt = next_attempt()
    tracked = engine.hash_snapshot()
    for path in (CONFIG, Path(__file__).resolve()):
        tracked[str(path)] = engine.sha256_file(path)
    scenario = engine.normal.scenario_gate(engine.TRUSTED_SCENARIO, attempt / "scenario" / "ordinary_random_seed2026", attempt / "audit")
    write_json(attempt / "scenario_identity_report.json", scenario)
    bundle = attempt / "scenario" / "ordinary_random_seed2026"
    specs = {method: make_spec(method, bundle, attempt) for method in METHODS}
    processes = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(launch, specs[method], attempt): method for method in METHODS}
        for future in as_completed(futures):
            processes.append(future.result())
    metrics, ledgers = {}, {}
    for method in METHODS:
        metrics[method], ledgers[method] = engine.evaluate_run(method, Path(specs[method]["output"]))
    current, hard = metrics[CURRENT], metrics[HARD]
    summary = []
    for method in METHODS:
        row = dict(metrics[method])
        row.update({
            "ordinary_closure_count": int(row["all_closure_count"]) - int(row["HV_closure_count"]),
            "delta_J_all_vs_current": float(row["J_all_net"]) - float(current["J_all_net"]),
            "delta_J_HV_vs_current": float(row["J_HV_net"]) - float(current["J_HV_net"]),
            "delta_all_closure_vs_current": int(row["all_closure_count"]) - int(current["all_closure_count"]),
            "delta_HV_closure_vs_current": int(row["HV_closure_count"]) - int(current["HV_closure_count"]),
            "delta_high_closure_vs_current": int(row["high_closure_count"]) - int(current["high_closure_count"]),
            "delta_urgent_closure_vs_current": int(row["urgent_closure_count"]) - int(current["urgent_closure_count"]),
        })
        summary.append(row)
    write_csv(attempt / "current_vs_ordinary_hard_requirement_summary.csv", summary)

    current_tasks = {row["task_id"]: row for row in engine.read_csv(Path(specs[CURRENT]["output"]) / "per_task_outcomes.csv")}
    hard_tasks = {row["task_id"]: row for row in engine.read_csv(Path(specs[HARD]["output"]) / "per_task_outcomes.csv")}
    paired = []
    for task_id in sorted(current_tasks):
        c, h = current_tasks[task_id], hard_tasks[task_id]
        if c["selected_action"] == h["selected_action"] and c["delivered_product"] == h["delivered_product"] and c["closure_success"] == h["closure_success"]:
            continue
        paired.append({
            "task_id": task_id, "value_level": c["value_level"], "required_product": c["product_requirement"],
            "current_action": c["selected_action"], "hard_requirement_action": h["selected_action"],
            "current_delivered_product": c["delivered_product"], "hard_requirement_delivered_product": h["delivered_product"],
            "current_closure": c["closure_success"], "hard_requirement_closure": h["closure_success"],
            "closure_delta": int(float(h["closure_success"])) - int(float(c["closure_success"])),
        })
    write_csv(attempt / "paired_task_changes.csv", paired)

    reference = engine.read_json(REFERENCE_CURRENT / "frozen_public_metrics.json")
    fields = ("J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count", "controllable_energy_Wh", "transmitted_MB")
    current_exact = all(abs(float(current[field]) - float(reference[field])) <= 1.0e-10 for field in fields) and engine.decision_semantic_hash(Path(specs[CURRENT]["output"])) == engine.decision_semantic_hash(REFERENCE_CURRENT)
    hard_identity = engine.read_json(Path(specs[HARD]["output"]) / "runtime_identity.json")
    post = engine.hash_snapshot()
    for path in (CONFIG, Path(__file__).resolve()):
        post[str(path)] = engine.sha256_file(path)
    checks = {
        "current_exact_reproduction": current_exact, "ordinary_hard_requirement_enabled": hard_identity["ordinary_hard_requirement"],
        "same_scenario": scenario["status"] == "PASS", "two_unique_processes": len({row["pid"] for row in processes}) == 2,
        "source_hashes_unchanged": tracked == post,
        "zero_safety_violations": all(int(metrics[m]["soc_violation_count"]) == 0 and int(metrics[m]["thermal_violation_count"]) == 0 for m in METHODS),
    }
    write_json(attempt / "audit" / "identity_and_regression.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "processes": processes, "hard_identity": hard_identity})
    (attempt / "test_report.txt").write_text("\n".join(f"{'PASS' if passed else 'FAIL'}\t{name}" for name, passed in checks.items()) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError({"ORDINARY_HARD_REQUIREMENT_AUDIT_FAILED": checks})
    lines = [
        "# Current PROP versus ordinary hard-requirement granularity", "",
        f"Current exact reproduction: **{current_exact}**.", "",
        "| method | J_all_net | J_HV_net | all | ordinary | HV | high | urgent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {CURRENT} | {float(current['J_all_net']):.6f} | {float(current['J_HV_net']):.6f} | {current['all_closure_count']} | {int(current['all_closure_count'])-int(current['HV_closure_count'])} | {current['HV_closure_count']} | {current['high_closure_count']} | {current['urgent_closure_count']} |",
        f"| {HARD} | {float(hard['J_all_net']):.6f} | {float(hard['J_HV_net']):.6f} | {hard['all_closure_count']} | {int(hard['all_closure_count'])-int(hard['HV_closure_count'])} | {hard['HV_closure_count']} | {hard['high_closure_count']} | {hard['urgent_closure_count']} |", "",
        f"Hard-requirement deltas: J_all {float(hard['J_all_net'])-float(current['J_all_net']):+.12f}; J_HV {float(hard['J_HV_net'])-float(current['J_HV_net']):+.12f}; all/HV/high/urgent closure {int(hard['all_closure_count'])-int(current['all_closure_count']):+d}/{int(hard['HV_closure_count'])-int(current['HV_closure_count']):+d}/{int(hard['high_closure_count'])-int(current['high_closure_count']):+d}/{int(hard['urgent_closure_count'])-int(current['urgent_closure_count']):+d}.", "",
        COMPLETION,
    ]
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(attempt / "completion.json", {"status": "PASS", "marker": COMPLETION, "completed_at": datetime.now(timezone.utc).isoformat(), "output": str(attempt)})
    print(f"Unique entry: {Path(__file__).resolve()}")
    print(f"Output: {attempt}")
    print(COMPLETION)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child_main(args.child_spec) if args.child_spec else main())
