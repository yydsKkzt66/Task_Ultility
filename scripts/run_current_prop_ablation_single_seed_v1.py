"""One-click six-method ablation of the current V1+V2 PROP candidate."""

from __future__ import annotations

import argparse
import csv
import json
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

import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


VERSION = "CURRENT-PROP-ABLATION-SINGLE-SEED-v1"
COMPLETION = "CURRENT_PROP_V1_V2_ABLATION_SINGLE_SEED_COMPLETE"
CONFIG = ROOT / "configs" / "current_prop_ablation_single_seed_v1.yaml"
CURRENT_CONFIG = ROOT / "configs" / "prop_v2_safe_metadata_current_v1.yaml"
OUTPUT_BASE = ROOT / "outputs" / "_current_prop_v1_v2_ablation_single_seed_v1"

LEGACY = "LEGACY-PROP-V2"
CURRENT = "PROP-V2-SAFE-METADATA-CURRENT-v1"
A1 = "A1-NO-PROGRESSIVE"
A2 = "A2-FIXED-BALANCED"
A3 = "A3-NO-SLOW-LYAPUNOV"
A4 = "A4-SAFE-GREEDY-NO-ONLINE-BANDIT"
METHODS = (LEGACY, CURRENT, A1, A2, A3, A4)
SLUG = {method: method.replace("-", "_") for method in METHODS}
OPTIONS = {
    LEGACY: {"flags": (False, False, False)},
    CURRENT: {"flags": (True, True, False)},
    A1: {"flags": (True, True, False), "direct_only": True},
    A2: {"flags": (True, True, False), "fixed_model": True},
    A3: {"flags": (True, True, False), "single_timescale": True},
    A4: {"flags": (True, True, False), "greedy_no_bandit": True},
}


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
    for name in ("audit", "runs", "run_specs", "scenario", "tables"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def make_spec(method: str, bundle: Path, attempt: Path) -> dict[str, Any]:
    spec = {
        "version": VERSION, "method": method, "seed": engine.SEED, "slots": engine.SLOTS,
        "capacity": engine.CAPACITY, "bundle_root": str(bundle), "output": str(attempt / "runs" / SLUG[method]),
        "frozen_sha256": engine.sha256_file(engine.FROZEN), "config_sha256": engine.sha256_file(CONFIG),
        "current_method_config_sha256": engine.sha256_file(CURRENT_CONFIG), "entry_sha256": engine.sha256_file(Path(__file__).resolve()),
    }
    spec.update(OPTIONS[method])
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
    stdout_path = attempt / "audit" / f"{SLUG[method]}.stdout.log"
    stderr_path = attempt / "audit" / f"{SLUG[method]}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)],
            cwd=ROOT, stdout=stdout, stderr=stderr, text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError({"METHOD_RUNTIME_FAILED": method, "stderr": str(stderr_path), "returncode": completed.returncode})
    identity = engine.read_json(Path(spec["output"]) / "runtime_identity.json")
    return {"method": method, "pid": identity["pid"], "output": spec["output"]}


def task_pairing(attempt: Path) -> list[dict[str, Any]]:
    def outcomes(method: str) -> dict[str, dict[str, str]]:
        return {row["task_id"]: row for row in engine.read_csv(attempt / "runs" / SLUG[method] / "per_task_outcomes.csv")}

    current = outcomes(CURRENT)
    result = []
    for method in (LEGACY, A1, A2, A3, A4):
        other = outcomes(method)
        for task_id in sorted(current):
            c, o = current[task_id], other[task_id]
            if c["closure_success"] == o["closure_success"] and c["selected_action"] == o["selected_action"] and c["delivered_product"] == o["delivered_product"]:
                continue
            result.append({
                "comparison_method": method, "task_id": task_id, "value_level": c["value_level"],
                "required_product": c["product_requirement"], "current_selected_action": c["selected_action"],
                "ablation_selected_action": o["selected_action"], "current_delivered_product": c["delivered_product"],
                "ablation_delivered_product": o["delivered_product"], "current_closure": c["closure_success"],
                "ablation_closure": o["closure_success"],
                "closure_delta_ablation_minus_current": int(float(o["closure_success"])) - int(float(c["closure_success"])),
            })
    return result


def main() -> int:
    attempt = next_attempt()
    tracked = engine.hash_snapshot()
    for path in (CONFIG, CURRENT_CONFIG, Path(__file__).resolve()):
        tracked[str(path)] = engine.sha256_file(path)
    write_json(attempt / "audit" / "pre_run_hashes.json", tracked)
    scenario = engine.normal.scenario_gate(
        engine.TRUSTED_SCENARIO, attempt / "scenario" / "ordinary_random_seed2026", attempt / "audit"
    )
    write_json(attempt / "scenario_identity_report.json", scenario)
    bundle = attempt / "scenario" / "ordinary_random_seed2026"
    specs = {method: make_spec(method, bundle, attempt) for method in METHODS}

    processes = [launch(specs[LEGACY], attempt)]
    metrics: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    metrics[LEGACY], ledgers[LEGACY] = engine.evaluate_run(LEGACY, Path(specs[LEGACY]["output"]))
    regression = engine.v0_regression(metrics[LEGACY], Path(specs[LEGACY]["output"]))
    write_json(attempt / "audit" / "legacy_exact_regression.json", regression)
    if regression["status"] != "PASS":
        raise RuntimeError("LEGACY_PROP_REGRESSION_MISMATCH")

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(launch, specs[method], attempt): method for method in (CURRENT, A1, A2, A3, A4)}
        for future in as_completed(futures):
            processes.append(future.result())
    for method in (CURRENT, A1, A2, A3, A4):
        metrics[method], ledgers[method] = engine.evaluate_run(method, Path(specs[method]["output"]))

    current = metrics[CURRENT]
    legacy = metrics[LEGACY]
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
            "delta_J_all_vs_legacy": float(row["J_all_net"]) - float(legacy["J_all_net"]),
            "delta_J_HV_vs_legacy": float(row["J_HV_net"]) - float(legacy["J_HV_net"]),
            "delta_HV_closure_vs_legacy": int(row["HV_closure_count"]) - int(legacy["HV_closure_count"]),
        })
        summary.append(row)
    write_csv(attempt / "six_method_summary.csv", summary)
    write_csv(attempt / "tables" / "six_method_summary.csv", summary)
    write_csv(attempt / "paired_task_changes_vs_current.csv", task_pairing(attempt))

    identities = {method: engine.read_json(Path(specs[method]["output"]) / "runtime_identity.json") for method in METHODS}
    post = engine.hash_snapshot()
    for path in (CONFIG, CURRENT_CONFIG, Path(__file__).resolve()):
        post[str(path)] = engine.sha256_file(path)
    checks = {
        "legacy_exact_regression": regression["status"] == "PASS" and regression["decision_trace_exact"],
        "six_unique_processes": len({row["pid"] for row in processes}) == 6,
        "source_hashes_unchanged": tracked == post,
        "all_methods_zero_safety_violations": all(int(metrics[m]["soc_violation_count"]) == 0 and int(metrics[m]["thermal_violation_count"]) == 0 for m in METHODS),
        "current_V1_V2_only": identities[CURRENT]["safe_set_consistent_fallback"] and identities[CURRENT]["hv_metadata_residual_repair"] and not identities[CURRENT]["thermal_price_aware"],
        "A1_direct_only": identities[A1]["direct_only"],
        "A2_balanced_only": identities[A2]["fixed_model"] == "balanced" and int(metrics[A2]["model_lightweight_count"]) == 0 and int(metrics[A2]["model_high_accuracy_count"]) == 0,
        "A3_single_timescale": identities[A3]["single_timescale"] and int(identities[A3]["slow_runtime_audit"]["static_guidance_call_count"]) == engine.SLOTS,
        "A4_no_online_bandit": identities[A4]["greedy_no_online_bandit"] and int(identities[A4]["greedy_audit"]["blocked_online_update_count"]) > 0,
    }
    write_json(attempt / "audit" / "method_identity_and_isolation.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "processes": processes, "identities": identities})
    (attempt / "test_report.txt").write_text("\n".join(f"{'PASS' if passed else 'FAIL'}\t{name}" for name, passed in checks.items()) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError({"ABLATION_AUDIT_FAILED": checks})

    by_method = {row["method"]: row for row in summary}
    lines = [
        "# Current PROP V1+V2 ablation assessment", "",
        f"Legacy exact regression: **{regression['status']}**, decision trace exact: **{regression['decision_trace_exact']}**.", "",
        "| method | J_all_net | J_HV_net | all | HV | high | urgent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = by_method[method]
        lines.append(f"| {method} | {float(row['J_all_net']):.6f} | {float(row['J_HV_net']):.6f} | {row['all_closure_count']} | {row['HV_closure_count']} | {row['high_closure_count']} | {row['urgent_closure_count']} |")
    lines.extend(["", "## Paired ablation deltas versus current PROP", ""])
    for method in (A1, A2, A3, A4):
        row = by_method[method]
        lines.append(f"- {method}: delta J_all={float(row['delta_J_all_vs_current']):+.12f}, delta J_HV={float(row['delta_J_HV_vs_current']):+.12f}, delta all/HV/high/urgent closure={int(row['delta_all_closure_vs_current']):+d}/{int(row['delta_HV_closure_vs_current']):+d}/{int(row['delta_high_closure_vs_current']):+d}/{int(row['delta_urgent_closure_vs_current']):+d}.")
    lines.extend(["", "All runs use the same seed, task/window hashes, capacity, frozen q/prices and public evaluator. No parameter was tuned after observing results.", "", COMPLETION])
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(attempt / "completion.json", {"status": "PASS", "marker": COMPLETION, "completed_at": datetime.now(timezone.utc).isoformat(), "output": str(attempt)})
    print(f"Unique entry: {Path(__file__).resolve()}")
    print(f"Output: {attempt}")
    print("Execution: legacy gate, then current+A1-A4 in five independent parallel processes")
    print(COMPLETION)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child_main(args.child_spec) if args.child_spec else main())
