"""One-seed controlled comparison for the default-off MYOPIC HV closure gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402
import scripts.run_prop_reqalign_seed2026_v1 as reqrun  # noqa: E402


SEED = 2026
SLOTS = 4320
DEFAULT_CAPACITY = 0.35
METHOD_OLD = exact.METHOD_MYOPIC
METHOD_GATE = "MYOPIC-FROZEN-CALIBRATED-REQALIGN-HV-CLOSURE-NET-GATE-v1"
METHODS = (METHOD_OLD, METHOD_GATE)
GATE_SOURCE = ROOT / "src" / "routers" / "hv_closure_net_gate.py"
GATE_CONFIG = ROOT / "configs" / "myopic_hv_closure_net_gate_v1.yaml"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "_v25_myopic_hv_closure_net_gate_seed2026_cap035_v1"
)
OLD_ZERO_W_REFERENCE = (
    ROOT
    / "outputs"
    / "_v25_sar_soc_pressure_seed2026_cap035_v1"
    / "runs"
    / "0W"
    / METHOD_OLD
)
OLD_REGRESSION_FILES = (
    "per_task_outcomes.csv",
    "ground_delivery_events.csv",
    "per_slot_resource_trace.csv",
    "candidate_action_log.csv",
    "resource_summary.json",
    "energy_components.json",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized or [{"empty": ""}])


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_gate_config(enabled: bool) -> dict[str, Any]:
    loaded = yaml.safe_load(GATE_CONFIG.read_text(encoding="utf-8"))
    config = dict(loaded["hv_closure_net_gate"])
    config["enabled"] = bool(enabled)
    return config


def run_gated(bundle_root: Path, output: Path, capacity: float) -> None:
    """Fresh native run; only the projection candidate gate is adapted."""

    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.simulator.environment import SimulationEnvironment

    prop_mod = v25.import_prop_script()
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(
        bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep
    )
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configs = prop_mod._configure(scenario.configs, calibration, {"enabled": False})
    scenario = replace(scenario, configs=configs)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, capacity)
    identity = exact.scenario_identity(
        METHOD_GATE, SEED, scenario, bundle, power, capacity
    )
    identity["capacity_identity"] = capacity_identity

    environment = SimulationEnvironment(scenario.configs)
    runner = environment.build_episode_runner(scenario.timeline[:SLOTS])
    delegate = exact.load_myopic_router_class()(
        v25.MYOPIC_PARAM, router_name="MYOPIC-FROZEN-CALIBRATED"
    )
    exact.bind_myopic_runtime(runner, delegate)

    base_projection = reqrun.load_projection_module()
    gate_module = reqrun.load_python_module(
        "hv_closure_net_gate_runtime_v1", GATE_SOURCE
    )
    facade = gate_module.HVClosureNetGateProjectionFacade(
        base_projection,
        delegate,
        load_gate_config(True),
    )

    class GatedRequirementAwareWrapper(reqrun.RequirementAwareRouterWrapper):
        def bind_step(self, timeline_step, resource_state, comm_state):
            super().bind_step(timeline_step, resource_state, comm_state)
            facade.bind_step(
                timeline_step.task,
                slot_seconds=float(timeline_step.slot_seconds),
            )

    wrapper = GatedRequirementAwareWrapper(
        delegate,
        reqrun.load_projection_config(True),
        scenario.timeline[:SLOTS],
        scenario.configs["simulator"],
        facade,
    )
    reqrun.bind_projection_runtime(runner, wrapper)
    result = runner.run(wrapper)
    normalized = normalize_episode(METHOD_GATE, scenario, result, runner, SLOTS)

    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(
        output / "candidate_action_log.csv",
        list(getattr(runner, "last_candidate_action_logs", [])),
    )
    projection_rows = reqrun.enrich_projection_rows(wrapper, output)
    write_csv(output / "requirement_projection_log.csv", projection_rows)
    write_csv(output / "hv_closure_net_gate_candidate_audit.csv", facade.candidate_audit_rows)
    reqrun.write_energy_components(output, result)
    write_json(
        output / "native_identity.json",
        {
            "method": METHOD_GATE,
            "base_method": METHOD_OLD,
            "fresh_state": True,
            "corrected_reqalign_enabled": True,
            "hv_closure_net_gate_enabled": True,
            "ordinary_tasks_bypass_gate": True,
            "original_action_ranking_preserved": True,
            "gate_source": str(GATE_SOURCE),
            "gate_source_sha256": reqrun.sha256_file(GATE_SOURCE),
            "gate_config": str(GATE_CONFIG),
            "gate_config_sha256": reqrun.sha256_file(GATE_CONFIG),
            "scenario_identity": identity,
        },
    )
    write_json(
        output / "method_run_status.json",
        {
            "status": "PASS",
            "method": METHOD_GATE,
            "seed": SEED,
            "capacity": capacity,
            "simulation_started": True,
            "fresh_state": True,
        },
    )
    write_json(
        output / "method_completed.json",
        {"status": "PASS", "method": METHOD_GATE, "seed": SEED},
    )


def launch_gate(bundle: Path, output: Path, log_dir: Path, capacity: float) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--bundle-root",
        str(bundle),
        "--run-output",
        str(output),
        "--capacity",
        str(capacity),
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.run(
        command,
        cwd=str(ROOT),
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "gate_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (log_dir / "gate_stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode or not (output / "method_completed.json").is_file():
        raise RuntimeError(
            {
                "gate_runtime_failed": process.returncode,
                "stdout": process.stdout[-1000:],
                "stderr": process.stderr[-4000:],
            }
        )


def paired_rows(old_path: Path, new_path: Path) -> list[dict[str, Any]]:
    old = {row["task_id"]: row for row in read_csv(old_path / "per_task_outcomes.csv")}
    new = {row["task_id"]: row for row in read_csv(new_path / "per_task_outcomes.csv")}
    rows = []
    for task_id in sorted(old):
        before = old[task_id]
        after = new[task_id]
        old_closed = int(Decimal(before.get("closure_success") or "0"))
        new_closed = int(Decimal(after.get("closure_success") or "0"))
        rows.append(
            {
                "task_id": task_id,
                "value_level": before.get("value_level", ""),
                "required_product": before.get("product_requirement", ""),
                "old_closure": old_closed,
                "new_closure": new_closed,
                "closure_delta": new_closed - old_closed,
                "old_selected_action": before.get("selected_action", ""),
                "new_selected_action": after.get("selected_action", ""),
                "old_executed_action": before.get("executed_action", ""),
                "new_executed_action": after.get("executed_action", ""),
                "old_delivered_product": before.get("delivered_product", ""),
                "new_delivered_product": after.get("delivered_product", ""),
                "old_ground_time_s": before.get("closure_ground_time", ""),
                "new_ground_time_s": after.get("closure_ground_time", ""),
                "old_energy_j": before.get("controllable_energy_j", ""),
                "new_energy_j": after.get("controllable_energy_j", ""),
                "old_transmitted_bytes": before.get("transmitted_bytes", ""),
                "new_transmitted_bytes": after.get("transmitted_bytes", ""),
            }
        )
    return rows


def parent_main(output: Path, capacity: float) -> int:
    if output.exists():
        raise FileExistsError(f"fresh output required: {output}")
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "config" / "controlled_bundle_seed2026"
    bundle_identity = exact.prepare_controlled_bundle(SEED, bundle)
    write_json(output / "audit" / "controlled_bundle_audit.json", bundle_identity)

    old_path = output / "runs" / METHOD_OLD
    new_path = output / "runs" / METHOD_GATE
    statuses = []
    statuses.append(
        exact.launch(METHOD_OLD, SEED, bundle, old_path, output / "logs", capacity)
    )
    regression_rows = []
    if abs(capacity - 0.35) <= 1.0e-12 and OLD_ZERO_W_REFERENCE.is_dir():
        for name in OLD_REGRESSION_FILES:
            fresh = old_path / name
            reference = OLD_ZERO_W_REFERENCE / name
            fresh_hash = reqrun.sha256_file(fresh)
            reference_hash = reqrun.sha256_file(reference)
            regression_rows.append(
                {
                    "file": name,
                    "fresh_sha256": fresh_hash,
                    "reference_sha256": reference_hash,
                    "exact_match": fresh_hash == reference_hash,
                }
            )
        if not all(row["exact_match"] for row in regression_rows):
            raise RuntimeError({"OLD_MYOPIC_DISABLED_REGRESSION_MISMATCH": regression_rows})
    write_json(
        output / "audit" / "old_myopic_exact_regression.json",
        {
            "reference": str(OLD_ZERO_W_REFERENCE),
            "reference_available": OLD_ZERO_W_REFERENCE.is_dir(),
            "capacity": capacity,
            "exact_match": bool(regression_rows)
            and all(row["exact_match"] for row in regression_rows),
            "files": regression_rows,
        },
    )
    launch_gate(bundle, new_path, output / "logs", capacity)

    eval_mod = v25.load_eval_module(output)
    metrics = []
    for method, path in ((METHOD_OLD, old_path), (METHOD_GATE, new_path)):
        raw = exact.evaluate_method(method, path, bundle, eval_mod, capacity)
        selected = exact.selected_metrics(SEED, raw)
        metrics.append(selected)
        if not bool(selected["public_accounting_passed"]):
            raise RuntimeError({"PUBLIC_ACCOUNTING_MISMATCH": method})
        if int(selected["SOC_violations"]) or int(selected["thermal_violations"]):
            raise RuntimeError({"HARD_SAFETY_VIOLATION": method})

    old_metric, new_metric = metrics
    delta_fields = (
        "J_all_net",
        "J_HV_net",
        "N_HV_closure",
        "N_high_closure",
        "N_urgent_closure",
        "all_task_closure_count",
        "all_task_closure_rate",
        "N_HV_metadata",
        "N_HV_ROI",
        "N_HV_evidence",
        "successful_closure_mean_latency_s",
        "HV_successful_closure_mean_latency_s",
        "urgent_successful_closure_mean_latency_s",
        "controllable_energy_Wh",
        "transmitted_MB",
        "peak_queue_MB",
        "mean_queue_MB",
        "minimum_soc",
        "maximum_thermal",
        "SOC_violations",
        "thermal_violations",
    )
    deltas = [
        {
            "metric": field,
            "old_myopic_reqalign": old_metric[field],
            "hv_closure_net_gate": new_metric[field],
            "delta_new_minus_old": float(new_metric[field]) - float(old_metric[field]),
        }
        for field in delta_fields
    ]
    pairs = paired_rows(old_path, new_path)
    pair_summary = []
    for level in ("low", "medium", "high", "urgent", "all"):
        group = pairs if level == "all" else [row for row in pairs if row["value_level"] == level]
        pair_summary.append(
            {
                "value_level": level,
                "new_only_success": sum(row["old_closure"] == 0 and row["new_closure"] == 1 for row in group),
                "old_only_success": sum(row["old_closure"] == 1 and row["new_closure"] == 0 for row in group),
                "both_success": sum(row["old_closure"] == 1 and row["new_closure"] == 1 for row in group),
                "neither_success": sum(row["old_closure"] == 0 and row["new_closure"] == 0 for row in group),
            }
        )

    gate_audit = read_csv(new_path / "hv_closure_net_gate_candidate_audit.csv")
    gate_summary = {
        "candidate_count": len(gate_audit),
        "rescued_candidate_count": sum(int(row["rescued_negative_original_score"] == "True") for row in gate_audit),
        "rescued_task_count": len(
            {
                row["task_id"]
                for row in gate_audit
                if row["rescued_negative_original_score"] == "True"
            }
        ),
        "ordinary_rescued_candidate_count": sum(
            row["value_level"] in {"low", "medium"}
            and row["rescued_negative_original_score"] == "True"
            for row in gate_audit
        ),
    }
    write_csv(output / "tables" / "metrics.csv", metrics)
    write_md(output / "tables" / "metrics.md", metrics)
    write_csv(output / "tables" / "old_new_deltas.csv", deltas)
    write_md(output / "tables" / "old_new_deltas.md", deltas)
    write_csv(output / "tables" / "paired_tasks.csv", pairs)
    write_csv(output / "tables" / "paired_summary.csv", pair_summary)
    write_md(output / "tables" / "paired_summary.md", pair_summary)
    write_json(output / "audit" / "gate_activation_audit.json", gate_summary)
    write_json(output / "audit" / "old_launch_status.json", statuses)

    assessment = [
        "# MYOPIC high-value closure-net gate, seed 2026",
        "",
        f"- capacity: {capacity}",
        "- Frozen V25 task stream and all hard constraints are unchanged.",
        "- Low and medium tasks bypass the gate exactly.",
        "- This is a one-seed mechanism diagnostic, not evidence of statistical superiority.",
        "",
        "## Primary result",
        "",
        *[
            f"- {row['metric']}: old={row['old_myopic_reqalign']}, new={row['hv_closure_net_gate']}, delta={row['delta_new_minus_old']}"
            for row in deltas[:5]
        ],
        "",
        "## Activation",
        "",
        f"- Rescued candidates: {gate_summary['rescued_candidate_count']} across {gate_summary['rescued_task_count']} tasks.",
        f"- Ordinary rescued candidates: {gate_summary['ordinary_rescued_candidate_count']} (must be zero).",
    ]
    (output / "final_assessment.md").write_text(
        "\n".join(assessment) + "\n", encoding="utf-8"
    )
    completion = {
        "status": "MYOPIC_HV_CLOSURE_NET_GATE_SEED2026_COMPLETE",
        "capacity": capacity,
        "seed": SEED,
        "methods": list(METHODS),
        "gate_activation": gate_summary,
        "output": str(output),
    }
    write_json(output / "completion_status.json", completion)
    print(json.dumps(completion, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capacity", type=float, default=DEFAULT_CAPACITY)
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--run-output", type=Path)
    args = parser.parse_args()
    if args.child_run:
        run_gated(args.bundle_root, args.run_output, float(args.capacity))
        print(json.dumps({"status": "PASS", "method": METHOD_GATE}, ensure_ascii=False))
        return 0
    return parent_main(args.output_dir, float(args.capacity))


if __name__ == "__main__":
    raise SystemExit(main())
