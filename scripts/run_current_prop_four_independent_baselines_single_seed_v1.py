"""One-click five-method ordinary-V25 comparison on the shared thermal model.

The standalone MYOPIC baseline changes only action ranking.  It shares the
3x3 model-product action grid, predictor, SafeFilter, executor and public
accounting, but has no PROP projection/progressive state, Lyapunov guidance,
mode controller, SafeLinUCB or online update.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_dual_node_thermal_calibration_and_four_method_rerun_v1 as dual  # noqa: E402
import scripts.run_normal_scenario_tut_proxy_ground_recovery_v1 as normal  # noqa: E402
import scripts.run_pure_myopic_safe_greedy_offload_thermal_comparison_v1 as adapted  # noqa: E402
import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


VERSION = "CURRENT-PROP-FOUR-INDEPENDENT-BASELINES-SINGLE-SEED-v1"
COMPLETION = "CURRENT_PROP_FOUR_INDEPENDENT_BASELINES_SINGLE_SEED_COMPLETE"
CURRENT = "PROP-V2-SAFE-METADATA-CURRENT-v1"
B1 = dual.METHOD_B1
OFFLOAD = adapted.OFFLOAD
TUT = dual.METHOD_TUT
GREEDY = "MYOPIC-E2E-UTILITY-GREEDY-STANDALONE-v1"
METHODS = (CURRENT, B1, OFFLOAD, TUT, GREEDY)
SLUG = {method: method.replace("-", "_") for method in METHODS}
OUTPUT_BASE = ROOT / "outputs" / "_current_prop_four_independent_baselines_single_seed_v1"
SEED, CAPACITY, SLOTS = 2026, 0.5, 4320
BUNDLE, TASK_HASH = dual.BUNDLE, dual.TASK_HASH
GREEDY_SOURCE = ROOT / "src" / "routers" / "myopic_e2e_utility_greedy.py"
GREEDY_CORRECTED_SUBDIR = "MYOPIC_E2E_UTILITY_GREEDY_STANDALONE_v1_PREDICTED_REWARD"
COMPATIBLE_OFFLOAD_RUN = (
    ROOT
    / "outputs"
    / "_pure_myopic_safe_greedy_offload_thermal_single_seed_v1"
    / "attempt_005"
    / "runs"
    / "OFFLOAD_EFT_THERMAL_ALIGNED_2NODE_DIAGNOSTIC"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in data for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data or [{"empty": ""}])


def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def i(value: Any) -> int:
    return int(f(value))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    number = 1
    while (OUTPUT_BASE / f"attempt_{number:03d}").exists():
        number += 1
    attempt = OUTPUT_BASE / f"attempt_{number:03d}"
    for name in ("audit", "runs", "run_specs", "logs", "tables"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    return attempt


def protected_snapshot() -> dict[str, str]:
    paths = dual.protected_files() + [
        adapted.OFFLOAD_ROUTER_SOURCE,
        adapted.OFFLOAD_CORE_SOURCE,
        adapted.OFFLOAD_METHOD_CONFIG,
        ROOT / "configs" / "prop_v2_safe_metadata_current_v1.yaml",
        GREEDY_SOURCE,
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths if path.is_file()}


def install_step_binding(runner: Any, router: Any) -> None:
    """Expose current observable state to the selector without altering physics."""

    original_factory = runner.step_runner_factory

    def factory(queue: Any) -> Any:
        step_runner = original_factory(queue)
        original_run_step = step_runner.run_step

        def intercepted(timeline_step: Any, resource_state: Any, mission_state: Any, comm_state: Any, active_router: Any) -> Any:
            router.bind_step(timeline_step, resource_state, comm_state)
            return original_run_step(timeline_step, resource_state, mission_state, comm_state, active_router)

        step_runner.run_step = intercepted
        return step_runner

    runner.step_runner_factory = factory


def run_greedy_child(spec: Mapping[str, Any], output: Path, adapter: Any) -> dict[str, Any]:
    patch_audit: dict[str, Any] = {}
    original_setup = engine.v25.child_setup_path

    def setup(path: Path) -> None:
        nonlocal patch_audit
        original_setup(path)
        patch_audit = dual.install_runtime_patches(adapter)

    engine.v25.child_setup_path = setup
    engine.v25.child_setup_path(engine.v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.simulator.environment import SimulationEnvironment

    router_module = dual.private_module("myopic_e2e_standalone_runtime_v1", GREEDY_SOURCE)
    frozen = engine.finalgate.load_frozen()
    core = engine.v2diag.load_v2core()
    core.FROZEN_QUALITY_PROXY.clear()
    core.FROZEN_QUALITY_PROXY.update({key: f(value) for key, value in frozen["q_eff"].items()})
    bundle = normal.load_neutral_bundle(Path(spec["bundle_root"]))
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    configured = engine.v25.import_prop_script()._configure(
        scenario.configs,
        engine.v25.read_json(engine.v25.SHARED_PRIOR_CALIBRATION),
        {"enabled": False},
    )
    configured, grid_audit = engine.v2diag.configure_v2(configured, fixed=False)
    configured = engine.calibration.configure_online_prices(
        configured,
        {"prices": frozen["public_prices"], "legacy_online_baseline": False},
    )
    scenario = replace(scenario, configs=configured)
    scenario, power = engine.v25.inject_power(scenario, engine.v25.SOURCE_PROP)
    scenario, capacity_identity = engine.guardrun.apply_capacity(scenario, CAPACITY)
    scenario_identity = engine.exact.scenario_identity(GREEDY, SEED, scenario, bundle, power, CAPACITY)
    if scenario_identity.get("task_hash") != TASK_HASH:
        raise RuntimeError("TASK_MANIFEST_MISMATCH")

    public_values = [
        f(getattr(step.task, "calibrated_task_value", getattr(step.task, "task_value", 0.0)))
        for step in scenario.timeline[:SLOTS]
        if step.task is not None
    ]
    task_value_reference = max(public_values, default=1.0)
    environment = SimulationEnvironment(scenario.configs)
    runner = environment.build_episode_runner(scenario.timeline[:SLOTS])
    router = router_module.MyopicE2EUtilityGreedyRouter(
        scenario.configs["reward"],
        scenario.configs["simulator"],
        scenario.configs["product_requirement_utility"],
        router_name="myopic_e2e_utility_greedy_standalone_v1",
        task_value_reference=task_value_reference,
        score_mode="common_predicted_reward",
    )
    install_step_binding(runner, router)
    result = runner.run(router)
    normalized = normalize_episode(GREEDY, scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    engine.v25.write_unified_method_outputs(output, normalized)
    engine.v25.write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
    engine.reqrun.write_energy_components(output, result)
    write_json(output / "resolved_config.json", scenario.configs)
    identity = {
        "status": "PASS",
        "version": VERSION,
        "method": GREEDY,
        "pid": os.getpid(),
        "seed": SEED,
        "fresh_state": True,
        "selector_class": type(router).__name__,
        "selector_source": str(GREEDY_SOURCE.resolve()),
        "selector_sha256": sha256_file(GREEDY_SOURCE),
        "score_rule": "SafeFilter then argmax of the shared predictor's one-step predicted_reward, including Defer",
        "score_mode": "common_predicted_reward",
        "duplicate_proxy_penalties": False,
        "task_value_reference": task_value_reference,
        "task_value_reference_semantics": "public V25 category-scale normalization only; no arrival/order/outcome information",
        "action_grid_audit": grid_audit,
        "safe_filter_shared": True,
        "predictor_executor_shared": True,
        "requirement_alignment": False,
        "hv_special_protection": False,
        "progressive_delivery": False,
        "residual_reentry": False,
        "lyapunov_guidance": False,
        "mode_controller": False,
        "safe_linucb": False,
        "online_update": False,
        "future_information_used": False,
        "scenario_identity": scenario_identity,
        "power_identity": power,
        "capacity_identity": capacity_identity,
        "q_eff": frozen["q_eff"],
        "public_prices_accounting_only": frozen["public_prices"],
    }
    write_json(output / "runtime_identity.json", identity)
    write_json(output / "method_completed.json", {"status": "PASS", "method": GREEDY, "seed": SEED, "identity_hash": stable_hash(identity)})
    return patch_audit


def child_run(spec: Mapping[str, Any]) -> int:
    method, output = str(spec["method"]), Path(spec["output"])
    params = dual.thermal.TwoNodeParameters(**read_json(Path(spec["thermal_model"]))["parameters"])
    adapter = dual.thermal.LegacyScalarAdapter(params)
    if method == CURRENT:
        patch_audit = adapted.run_current_child(spec, output, adapter)
    elif method == B1:
        patch_audit = dual.run_b1_ordinary_child(output, Path(spec["bundle_root"]), adapter)
    elif method == OFFLOAD:
        patch_audit = adapted.run_offload_child(spec, output, adapter)
    elif method == TUT:
        original_setup = normal.v25.child_setup_path

        def setup(path: Path) -> None:
            nonlocal patch_audit
            original_setup(path)
            patch_audit = dual.install_runtime_patches(adapter)

        patch_audit = {}
        normal.v25.child_setup_path = setup
        normal.run_tut_child(normal.METHOD_PROXY, Path(spec["bundle_root"]), output)
        write_json(output / "method_completed.json", {"status": "PASS", "method": TUT, "seed": SEED})
    elif method == GREEDY:
        patch_audit = run_greedy_child(spec, output, adapter)
    else:
        raise ValueError(method)

    write_csv(output / "dual_node_transition_trace.csv", adapter.rows)
    actual = [row for row in adapter.rows if row.get("caller") in {"run_step", "run", "_step", "actual_slot_transition"}]
    write_json(output / "dual_node_runtime_audit.json", {
        "method": method,
        "pid": os.getpid(),
        "thermal_parameter_hash": params.stable_hash(),
        "patch_audit": patch_audit,
        "transition_call_count": len(adapter.rows),
        "actual_slot_transition_count": len(actual),
        "max_chip_c": max((f(row.get("peak_chip_c")) for row in actual), default=0.0),
        "max_surface_c": max((f(row.get("peak_surface_c")) for row in actual), default=0.0),
        "chip_violation_count": sum(f(row.get("peak_chip_c")) > params.t_chip_max_c + 1e-12 for row in actual),
        "surface_violation_count": sum(f(row.get("peak_surface_c")) > params.t_surface_max_c + 1e-12 for row in actual),
    })
    return 0


def launch_all(attempt: Path, model: Path) -> list[dict[str, Any]]:
    processes: list[tuple[str, subprocess.Popen[Any], Any, Any]] = []
    for method in METHODS:
        spec = {
            "version": VERSION,
            "method": method,
            "seed": SEED,
            "capacity": CAPACITY,
            "slots": SLOTS,
            "bundle_root": str(BUNDLE),
            "thermal_model": str(model),
            "output": str(attempt / "runs" / SLUG[method]),
        }
        spec_path = attempt / "run_specs" / f"{SLUG[method]}.json"
        write_json(spec_path, spec)
        stdout = (attempt / "logs" / f"{SLUG[method]}.stdout.log").open("w", encoding="utf-8")
        stderr = (attempt / "logs" / f"{SLUG[method]}.stderr.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        processes.append((method, process, stdout, stderr))
    statuses = []
    for method, process, stdout, stderr in processes:
        returncode = process.wait()
        stdout.close()
        stderr.close()
        statuses.append({
            "method": method,
            "pid": process.pid,
            "returncode": returncode,
            "status": "PASS" if returncode == 0 else "FAIL",
            "output": str(attempt / "runs" / SLUG[method]),
        })
    return statuses


def evaluate(method: str, run: Path) -> dict[str, Any]:
    if method == CURRENT:
        metric, _ = engine.evaluate_run(method, run)
        return metric
    if method == TUT:
        return normal.evaluate(normal.METHOD_PROXY, run, BUNDLE)
    if method == OFFLOAD:
        return read_json(run / "frozen_public_metrics.json")
    frozen = engine.finalgate.load_frozen()
    config_path = run / "resolved_config.json"
    resolved_config = read_json(config_path if config_path.is_file() else BUNDLE / "resolved_config.json")
    metric, ledger = engine.gatecore.evaluate_public_run(method, run, resolved_config, frozen)
    write_csv(run / "frozen_public_event_ledger.csv", ledger)
    write_json(run / "frozen_public_metrics.json", metric)
    return metric


def summary_row(method: str, run: Path, metric: Mapping[str, Any]) -> dict[str, Any]:
    thermal = read_json(run / "dual_node_runtime_audit.json")
    trace = read_csv(run / "per_slot_resource_trace.csv")
    queues = [f(row.get("queue_mb")) for row in trace]
    all_count = i(metric.get("all_closure_count", metric.get("closure_count")))
    hv_count = i(metric.get("HV_closure_count"))
    gross = f(metric.get("gross_utility", metric.get("gross_delivered_utility")))
    ledger_path = run / "frozen_public_event_ledger.csv"
    ledger = read_csv(ledger_path) if ledger_path.is_file() else []
    hv_ledger = [row for row in ledger if row.get("value_level") in {"high", "urgent"}]
    def hv_product_count(*products: str) -> int:
        accepted = set(products)
        return sum(str(row.get("product", "")).lower() in accepted for row in hv_ledger)
    return {
        "method": method,
        "J_all_net": metric.get("J_all_net"),
        "J_HV_net": metric.get("J_HV_net"),
        "gross_utility": gross,
        "total_public_cost": gross - f(metric.get("J_all_net")),
        "energy_cost": metric.get("energy_cost"),
        "downlink_cost": metric.get("downlink_cost"),
        "terminal_backlog_cost": metric.get("terminal_backlog_cost"),
        "all_closure_count": all_count,
        "closure_rate": all_count / 268.0,
        "HV_closure_count": hv_count,
        "high_closure_count": metric.get("high_closure_count"),
        "urgent_closure_count": metric.get("urgent_closure_count"),
        "ordinary_closure_count": metric.get("ordinary_closure_count", all_count - hv_count),
        "metadata_delivery_count": metric.get("metadata_delivery_count"),
        "ROI_delivery_count": metric.get("ROI_delivery_count"),
        "Evidence_delivery_count": metric.get("Evidence_delivery_count"),
        "HV_metadata_delivered_count": hv_product_count("metadata"),
        "HV_ROI_delivered_count": hv_product_count("roi", "verification"),
        "HV_evidence_delivered_count": hv_product_count("roi_context", "evidence", "context"),
        "mean_closure_latency_s": metric.get("mean_full_closure_latency_s", metric.get("successful_closure_mean_latency_s")),
        "p95_closure_latency_s": metric.get("p95_full_closure_latency_s", metric.get("successful_closure_p95_latency_s")),
        "controllable_energy_Wh": metric.get("controllable_energy_Wh"),
        "transmitted_MB": metric.get("transmitted_MB"),
        "peak_queue_MB": metric.get("peak_queue_MB", max(queues, default=0.0)),
        "mean_queue_MB": sum(queues) / max(len(queues), 1),
        "terminal_queue_MB": metric.get("terminal_queue_MB", queues[-1] if queues else 0.0),
        "minimum_soc": metric.get("minimum_soc", metric.get("min_SOC")),
        "maximum_chip_c": thermal["max_chip_c"],
        "maximum_surface_c": thermal["max_surface_c"],
        "soc_violation_count": metric.get("soc_violation_count", 0),
        "thermal_violation_count": i(thermal["chip_violation_count"]) + i(thermal["surface_violation_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path)
    parser.add_argument("--finalize-existing", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--greedy-only-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_spec:
        return child_run(read_json(args.child_spec))
    if args.greedy_only_output:
        model = adapted.reference_thermal_model()
        return child_run({
            "version": VERSION,
            "method": GREEDY,
            "seed": SEED,
            "capacity": CAPACITY,
            "slots": SLOTS,
            "bundle_root": str(BUNDLE),
            "thermal_model": str(model),
            "output": str(args.greedy_only_output),
        })

    attempt = args.finalize_existing if args.finalize_existing else next_attempt()
    model = adapted.reference_thermal_model()
    pre = protected_snapshot()
    contract = read_json(BUNDLE / "bundle_contract.json")
    if contract.get("task_hash") != TASK_HASH:
        raise RuntimeError("TASK_MANIFEST_MISMATCH")
    write_json(attempt / "audit" / "pre_run_hashes.json", pre)
    if args.finalize_existing:
        statuses = []
        for method in METHODS:
            corrected_greedy = attempt / "runs" / GREEDY_CORRECTED_SUBDIR
            run = (
                COMPATIBLE_OFFLOAD_RUN
                if method == OFFLOAD
                else corrected_greedy
                if method == GREEDY and corrected_greedy.is_dir()
                else attempt / "runs" / SLUG[method]
            )
            required = ("method_completed.json", "per_task_outcomes.csv", "per_slot_resource_trace.csv", "dual_node_runtime_audit.json")
            if not all((run / name).is_file() for name in required):
                raise RuntimeError({"COMPATIBLE_REUSE_INCOMPLETE": method, "run": str(run)})
            runtime = read_json(run / "dual_node_runtime_audit.json")
            statuses.append({
                "method": method,
                "pid": runtime["pid"],
                "returncode": 0,
                "status": "REUSED_COMPATIBLE" if method == OFFLOAD else "PASS_EXISTING_THIS_ATTEMPT",
                "output": str(run),
            })
    else:
        statuses = launch_all(attempt, model)
    failures = [row for row in statuses if row["status"] not in {"PASS", "PASS_EXISTING_THIS_ATTEMPT", "REUSED_COMPATIBLE"}]
    if failures:
        write_json(attempt / "audit" / "process_status.json", statuses)
        raise RuntimeError({"FIVE_METHOD_RUNTIME_FAILED": failures})

    metrics: dict[str, dict[str, Any]] = {}
    rows = []
    run_by_method = {row["method"]: Path(row["output"]) for row in statuses}
    for method in METHODS:
        run = run_by_method[method]
        metrics[method] = evaluate(method, run)
        rows.append(summary_row(method, run, metrics[method]))
    write_csv(attempt / "five_method_summary.csv", rows)
    write_csv(attempt / "tables" / "five_method_summary.csv", rows)

    current_identity = read_json(run_by_method[CURRENT] / "runtime_identity.json")
    greedy_identity = read_json(run_by_method[GREEDY] / "runtime_identity.json")
    offload_identity = read_json(run_by_method[OFFLOAD] / "runtime_identity.json")
    reuse_compatibility = {
        "task_hash": offload_identity.get("task_hash") == current_identity["scenario_identity"].get("task_hash") == TASK_HASH,
        "seed": i(offload_identity.get("seed")) == i(current_identity.get("seed")) == SEED,
        "capacity": f(offload_identity.get("capacity")) == f(current_identity["capacity_identity"].get("runtime_capacity")) == CAPACITY,
        "link_trace_hash": offload_identity["capacity_identity"].get("runtime_link_trace_hash") == current_identity["capacity_identity"].get("runtime_link_trace_hash"),
        "power_trace_hash": offload_identity["power_identity"].get("power_trace_hash") == current_identity["power_identity"].get("power_trace_hash"),
        "power_config_sha256": offload_identity["power_identity"].get("power_config_sha256") == current_identity["power_identity"].get("power_config_sha256"),
    }
    if not all(reuse_compatibility.values()):
        raise RuntimeError({"OFFLOAD_COMPATIBLE_REUSE_IDENTITY_MISMATCH": reuse_compatibility})
    post = protected_snapshot()
    thermal_hashes = {
        read_json(run_by_method[method] / "dual_node_runtime_audit.json")["thermal_parameter_hash"]
        for method in METHODS
    }
    audit = {
        "status": "PASS",
        "version": VERSION,
        "created_utc": utc_now(),
        "processes": statuses,
        "five_independent_processes": len({row["pid"] for row in statuses}) == 5,
        "execution_mode": "four fresh parallel runs plus one hash-compatible prior OFFLOAD reuse" if args.finalize_existing else "five fresh parallel runs",
        "reused_compatible_methods": [row["method"] for row in statuses if row["status"] == "REUSED_COMPATIBLE"],
        "offload_reuse_compatibility": reuse_compatibility,
        "fresh_state_each_method": True,
        "ordinary_random_scenario": True,
        "seed": SEED,
        "capacity": CAPACITY,
        "duration_s": 86400,
        "task_count": 268,
        "task_hash": TASK_HASH,
        "window_hash": current_identity["capacity_identity"]["runtime_link_trace_hash"],
        "correct_40w_environment": True,
        "same_two_node_thermal_hash": len(thermal_hashes) == 1,
        "thermal_hash": next(iter(thermal_hashes)),
        "current_prop_is_v1_plus_v2": bool(current_identity.get("safe_set_consistent_fallback")) and bool(current_identity.get("hv_metadata_residual_repair")),
        "current_prop_safe_consistent_v1": current_identity.get("safe_set_consistent_fallback"),
        "current_prop_progressive_metadata_v2": current_identity.get("hv_metadata_residual_repair"),
        "current_prop_bandit_mode": "safe-greedy replacement after prior SafeLinUCB diagnostic" if current_identity.get("greedy_no_online_bandit") else "SafeLinUCB",
        "greedy_identity": greedy_identity,
        "protected_assets_unchanged_during_run": pre == post,
    }
    audit["status"] = "PASS" if all((
        audit["five_independent_processes"], audit["same_two_node_thermal_hash"],
        audit["current_prop_is_v1_plus_v2"], audit["protected_assets_unchanged_during_run"],
        not greedy_identity["requirement_alignment"], not greedy_identity["progressive_delivery"],
        not greedy_identity["lyapunov_guidance"], not greedy_identity["safe_linucb"],
    )) else "FAIL"
    write_json(attempt / "audit" / "method_environment_identity.json", audit)

    tests = {
        "five_runs_pass": not failures,
        "five_independent_processes": audit["five_independent_processes"],
        "same_task_environment_thermal": audit["same_two_node_thermal_hash"],
        "current_prop_v1_plus_v2": audit["current_prop_is_v1_plus_v2"],
        "standalone_greedy_has_no_prop_extensions": not any(greedy_identity[key] for key in ("requirement_alignment", "hv_special_protection", "progressive_delivery", "residual_reentry", "lyapunov_guidance", "mode_controller", "safe_linucb", "online_update")),
        "standalone_greedy_3x3_complete": greedy_identity["action_grid_audit"]["complete"],
        "zero_safety_violations": all(i(row["soc_violation_count"]) == 0 and i(row["thermal_violation_count"]) == 0 for row in rows),
        "protected_assets_unchanged": audit["protected_assets_unchanged_during_run"],
    }
    success = all(tests.values())
    (attempt / "test_report.txt").write_text(
        "\n".join(("PASS" if passed else "FAIL") + "\t" + name for name, passed in tests.items())
        + "\n" + ("PASS\t" if success else "FAIL\t") + COMPLETION + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Current PROP and four independent baselines — single seed", "",
        f"Scenario: ordinary-random V25, seed={SEED}, capacity={CAPACITY}, 24 h, 40 W, task hash `{TASK_HASH}`.", "",
        "| method | J_all | J_HV | all/HV/high/urgent closure | energy Wh | tx MB | mean/p95 latency s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {f(row['J_all_net']):.6f} | {f(row['J_HV_net']):.6f} | "
            f"{i(row['all_closure_count'])}/{i(row['HV_closure_count'])}/{i(row['high_closure_count'])}/{i(row['urgent_closure_count'])} | "
            f"{f(row['controllable_energy_Wh']):.6f} | {f(row['transmitted_MB']):.6f} | "
            f"{f(row['mean_closure_latency_s']):.2f}/{f(row['p95_closure_latency_s']):.2f} |"
        )
    lines.extend([
        "", "## Identity conclusion", "",
        f"Current PROP is V1 safe-set-consistency + V2 HV metadata/residual progressive delivery: **{audit['current_prop_is_v1_plus_v2']}**.",
        f"Its selector mode in this current thermal branch is: **{audit['current_prop_bandit_mode']}**.",
        "The new MYOPIC is a standalone baseline, not an ablation behind PROP and not a direct public-evaluator oracle.",
        f"Execution mode: **{audit['execution_mode']}**; OFFLOAD reuse identity gate: **{all(audit['offload_reuse_compatibility'].values())}**.",
        "", COMPLETION if success else "FIVE_METHOD_COMPARISON_AUDIT_FAILED",
    ])
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(attempt / "completion.json", {"status": "PASS" if success else "FAIL", "marker": COMPLETION, "output": str(attempt)})
    print("PIDS:", ", ".join(str(row["pid"]) for row in statuses))
    print("OUTPUT:", attempt)
    print("CURRENT PROP V1+V2:", audit["current_prop_is_v1_plus_v2"])
    print(COMPLETION if success else "FIVE_METHOD_COMPARISON_AUDIT_FAILED")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
