"""Four-way slow-control ablation for the current PROP candidate.

Public use: run this file without arguments.  The private ``--child-spec``
mode exists only to give every method an isolated interpreter and simulator.
Existing code, configurations, and outputs are read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


VERSION = "CURRENT-PROP-SLOW-CONTROL-FACTORIAL-SINGLE-SEED-v1"
COMPLETION = "CURRENT_PROP_SLOW_CONTROL_FACTORIAL_SINGLE_SEED_COMPLETE"
OUTPUT_BASE = ROOT / "outputs" / "_current_prop_slow_control_factorial_single_seed_v1"
REFERENCE_RUN = (
    ROOT
    / "outputs"
    / "_current_prop_v1_v2_ablation_single_seed_v1"
    / "attempt_001"
    / "runs"
    / "PROP_V2_SAFE_METADATA_CURRENT_v1"
)

FULL = "PROP-V2-SAFE-METADATA-CURRENT-v1"
NO_LYAPUNOV = "A3a-NO-LYAPUNOV-KEEP-MODE"
NO_MODE = "A3b-NO-MODE-KEEP-LYAPUNOV"
NO_BOTH = "A3c-NO-LYAPUNOV-AND-MODE"
METHODS = (FULL, NO_LYAPUNOV, NO_MODE, NO_BOTH)
SLUG = {method: method.replace("-", "_") for method in METHODS}
VARIANT = {
    FULL: "full",
    NO_LYAPUNOV: "no_lyapunov",
    NO_MODE: "no_mode",
    NO_BOTH: "no_lyapunov_and_mode",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
    attempt = OUTPUT_BASE / f"attempt_{index:03d}"
    for name in ("audit", "runs", "run_specs", "scenario", "tables"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    return attempt


def make_spec(method: str, bundle: Path, attempt: Path) -> dict[str, Any]:
    return {
        "version": VERSION,
        "method": method,
        "variant": VARIANT[method],
        "seed": engine.SEED,
        "slots": engine.SLOTS,
        "capacity": engine.CAPACITY,
        "bundle_root": str(bundle),
        "output": str(attempt / "runs" / SLUG[method]),
        "flags": (True, True, False),
        "entry_sha256": engine.sha256_file(Path(__file__).resolve()),
        "frozen_sha256": engine.sha256_file(engine.FROZEN),
    }


def _new_guidance_audit(variant: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "call_count": 0,
        "none_bundle_count": 0,
        "zero_lambda_call_count": 0,
        "nonzero_lambda_call_count": 0,
        "mode_counts": Counter(),
        "lambda_min": {key: float("inf") for key in ("lambda_E", "lambda_L", "lambda_Q", "lambda_S")},
        "lambda_max": {key: float("-inf") for key in ("lambda_E", "lambda_L", "lambda_Q", "lambda_S")},
    }


class _GuidanceRouterProxy:
    """Expose alternate capabilities only to the guidance builder."""

    def __init__(self, router: Any, *, use_lyapunov: bool, use_mode: bool) -> None:
        self._router = router
        self.use_lyapunov_pricing = bool(use_lyapunov)
        self.use_mode_controller = bool(use_mode)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)


def _neutral_guidance() -> dict[str, Any]:
    zeros = {name: 0.0 for name in ("lambda_E", "lambda_L", "lambda_Q", "lambda_S")}
    return {
        "mode_name": "lyapunov_and_mode_disabled_neutral",
        "lambda_base": dict(zeros),
        "lambda_E_base": 0.0,
        "lambda_L_base": 0.0,
        "lambda_Q_base": 0.0,
        "lambda_S_base": 0.0,
        **zeros,
        "price_multipliers": {name: 1.0 for name in zeros},
        "exploration_scale": 1.0,
        "model_complexity_bias": 0.0,
        "payload_size_bias": 0.0,
        "high_value_bonus_scale": 1.0,
        "safety_margin_scale": 1.0,
        "c_k": {"mode_parameters": "disabled_by_factorial_ablation"},
        "z_k": dict(zeros),
    }


def _bind_guidance_intervention_and_audit(runner: Any, variant: str, audit: dict[str, Any]) -> None:
    previous_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = previous_factory(queue)
        original = step_runner._build_guidance_bundle

        def observed(self, slot_index, long_term_state, router):
            if variant == "full":
                bundle = original(slot_index, long_term_state, router)
            elif variant == "no_lyapunov":
                proxy = _GuidanceRouterProxy(router, use_lyapunov=False, use_mode=True)
                bundle = original(slot_index, long_term_state, proxy)
            elif variant == "no_mode":
                proxy = _GuidanceRouterProxy(router, use_lyapunov=True, use_mode=False)
                bundle = original(slot_index, long_term_state, proxy)
            elif variant == "no_lyapunov_and_mode":
                bundle = _neutral_guidance()
            else:
                raise ValueError(variant)
            audit["call_count"] += 1
            if bundle is None:
                audit["none_bundle_count"] += 1
                audit["mode_counts"]["none"] += 1
                values = {key: 0.0 for key in audit["lambda_min"]}
            else:
                audit["mode_counts"][str(bundle.get("mode_name", "missing"))] += 1
                values = {key: float(bundle.get(key, 0.0)) for key in audit["lambda_min"]}
            if any(abs(value) > 1.0e-12 for value in values.values()):
                audit["nonzero_lambda_call_count"] += 1
            else:
                audit["zero_lambda_call_count"] += 1
            for key, value in values.items():
                audit["lambda_min"][key] = min(audit["lambda_min"][key], value)
                audit["lambda_max"][key] = max(audit["lambda_max"][key], value)
            return bundle

        step_runner._build_guidance_bundle = types.MethodType(observed, step_runner)
        return step_runner

    runner.step_runner_factory = factory


def _finish_guidance_audit(audit: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(audit)
    result["mode_counts"] = dict(sorted(result["mode_counts"].items()))
    for group in ("lambda_min", "lambda_max"):
        result[group] = {
            key: (0.0 if value in {float("inf"), float("-inf")} else value)
            for key, value in result[group].items()
        }
    return result


def configure_and_run_child(spec: Mapping[str, Any], output: Path) -> None:
    method = str(spec["method"])
    variant = str(spec["variant"])
    engine.v25.child_setup_path(engine.v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    frozen = engine.finalgate.load_frozen()
    core = engine.v2diag.load_v2core()
    core.FROZEN_QUALITY_PROXY.clear()
    core.FROZEN_QUALITY_PROXY.update({key: engine.f(value) for key, value in frozen["q_eff"].items()})
    bundle = engine.normal.load_neutral_bundle(Path(spec["bundle_root"]))
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

    intervention: dict[str, Any] = {
        "variant": variant,
        "lyapunov_pricing": "enabled",
        "resource_mode_control": "enabled",
        "clean_factorial_intervention": True,
        "router_capability_flags_preserved": True,
        "progressive_and_residual_gates_preserved": True,
    }
    scenario = replace(scenario, configs=configured)
    scenario, power = engine.v25.inject_power(scenario, engine.v25.SOURCE_PROP)
    scenario, capacity_identity = engine.guardrun.apply_capacity(scenario, engine.CAPACITY)
    identity = engine.exact.scenario_identity(method, engine.SEED, scenario, bundle, power, engine.CAPACITY)
    if identity.get("task_hash") != engine.EXPECTED_TASK_HASH:
        raise RuntimeError({"TASK_HASH_MISMATCH": identity.get("task_hash")})

    environment = SimulationEnvironment(scenario.configs)
    runner = environment.build_episode_runner(scenario.timeline[: engine.SLOTS])
    delegate = build_routers(scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"]))[0]
    original_flags = {
        "use_lyapunov_pricing": bool(delegate.use_lyapunov_pricing),
        "use_mode_controller": bool(delegate.use_mode_controller),
        "use_guided_utility": bool(delegate.use_guided_utility),
    }
    if variant == "no_lyapunov":
        intervention["lyapunov_pricing"] = "zeroed_only_inside_guidance_builder"
    elif variant == "no_mode":
        intervention["resource_mode_control"] = "neutralized_only_inside_guidance_builder"
    elif variant == "no_lyapunov_and_mode":
        intervention["lyapunov_pricing"] = "zeroed_only_inside_guidance_builder"
        intervention["resource_mode_control"] = "neutralized_only_inside_guidance_builder"
    elif variant != "full":
        raise ValueError(variant)

    wrapper = engine.ControlledExplorationWrapper(
        delegate,
        engine.reqrun.load_projection_config(True),
        scenario.timeline[: engine.SLOTS],
        scenario.configs["simulator"],
        engine.reqrun.load_projection_module(),
        product_utility_cfg=scenario.configs["product_requirement_utility"],
        reward_cfg=scenario.configs["reward"],
        safe_consistent=True,
        metadata_repair=True,
        thermal_price=False,
        thermal_alpha=engine.f(scenario.configs["lyapunov_pricing"]["alpha_thermal"]),
        mode_update_slots=engine.i(scenario.configs["mode_controller"]["update_interval_slots"]),
    )
    engine.reqrun.bind_projection_runtime(runner, wrapper)
    guidance_audit = _new_guidance_audit(variant)
    _bind_guidance_intervention_and_audit(runner, variant, guidance_audit)

    result = runner.run(wrapper)
    normalized = normalize_episode(method, scenario, result, runner, engine.SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    engine.v25.write_unified_method_outputs(output, normalized)
    engine.v25.write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
    engine.reqrun.write_energy_components(output, result)
    write_csv(output / "v2_candidate_score_audit.csv", wrapper.candidate_rows)
    write_csv(output / "v2_decision_trace.csv", wrapper.v2_decision_rows)
    write_csv(output / "mechanism_intervention_trace.csv", wrapper.intervention_rows)
    write_csv(output / "unified_guard_trace.csv", wrapper.guard_rows)

    router_flags = {
        "use_lyapunov_pricing": bool(wrapper.use_lyapunov_pricing),
        "use_mode_controller": bool(wrapper.use_mode_controller),
        "use_guided_utility": bool(wrapper.use_guided_utility),
    }
    run_identity = {
        "status": "PASS",
        "version": VERSION,
        "method": method,
        "variant": variant,
        "seed": engine.SEED,
        "pid": os.getpid(),
        "fresh_state": True,
        "independent_rng_queue_soc_thermal_bandit_residual": True,
        "scenario_identity": identity,
        "power_identity": power,
        "capacity_identity": capacity_identity,
        "q_eff": frozen["q_eff"],
        "public_prices": frozen["public_prices"],
        "grid_audit": grid_audit,
        "original_delegate_flags": original_flags,
        "runtime_router_flags": router_flags,
        "intervention": intervention,
        "guidance_runtime_audit": _finish_guidance_audit(guidance_audit),
        "safe_set_consistent_fallback": True,
        "hv_metadata_residual_repair": True,
        "thermal_price_aware": False,
        "hard_safe_filter_unchanged": True,
        "public_evaluator_unchanged": True,
        "future_information_used": False,
        "spec_sha256": engine.stable_hash(spec),
    }
    write_json(output / "runtime_identity.json", run_identity)
    write_json(output / "resolved_config.json", scenario.configs)
    write_json(
        output / "method_completed.json",
        {"status": "PASS", "method": method, "seed": engine.SEED, "spec_sha256": engine.stable_hash(spec)},
    )


def child_main(spec_path: Path) -> int:
    spec = engine.read_json(spec_path)
    output = Path(spec["output"])
    if output.exists():
        raise FileExistsError(output)
    configure_and_run_child(spec, output)
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
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            {"METHOD_RUNTIME_FAILED": method, "returncode": completed.returncode, "stderr": str(stderr_path)}
        )
    identity = engine.read_json(Path(spec["output"]) / "runtime_identity.json")
    return {"method": method, "pid": identity["pid"], "output": spec["output"]}


def _paired_changes(attempt: Path) -> list[dict[str, Any]]:
    def outcomes(method: str) -> dict[str, dict[str, str]]:
        path = attempt / "runs" / SLUG[method] / "per_task_outcomes.csv"
        return {row["task_id"]: row for row in engine.read_csv(path)}

    full = outcomes(FULL)
    rows: list[dict[str, Any]] = []
    for method in METHODS[1:]:
        other = outcomes(method)
        for task_id in sorted(full):
            a, b = full[task_id], other[task_id]
            if (
                a["closure_success"] == b["closure_success"]
                and a["selected_action"] == b["selected_action"]
                and a["delivered_product"] == b["delivered_product"]
            ):
                continue
            rows.append(
                {
                    "comparison_method": method,
                    "task_id": task_id,
                    "value_level": a["value_level"],
                    "required_product": a["product_requirement"],
                    "full_selected_action": a["selected_action"],
                    "ablation_selected_action": b["selected_action"],
                    "full_delivered_product": a["delivered_product"],
                    "ablation_delivered_product": b["delivered_product"],
                    "full_closure": a["closure_success"],
                    "ablation_closure": b["closure_success"],
                    "closure_delta": int(float(b["closure_success"])) - int(float(a["closure_success"])),
                }
            )
    return rows


def _full_regression(attempt: Path, metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not REFERENCE_RUN.exists():
        return {"status": "REFERENCE_MISSING", "reference": str(REFERENCE_RUN)}
    reference_metrics, _ = engine.evaluate_run(FULL, REFERENCE_RUN)
    fields = (
        "J_all_net",
        "J_HV_net",
        "all_closure_count",
        "HV_closure_count",
        "high_closure_count",
        "urgent_closure_count",
        "controllable_energy_Wh",
        "transmitted_MB",
        "minimum_soc",
        "maximum_thermal",
    )
    deltas = {key: float(metrics[key]) - float(reference_metrics[key]) for key in fields}
    current_run = attempt / "runs" / SLUG[FULL]
    hashes = {}
    for name in ("candidate_action_log.csv", "v2_decision_trace.csv", "per_task_outcomes.csv"):
        hashes[name] = {
            "reference": engine.sha256_file(REFERENCE_RUN / name),
            "current": engine.sha256_file(current_run / name),
        }
        hashes[name]["exact"] = hashes[name]["reference"] == hashes[name]["current"]
    passed = all(abs(value) <= 1.0e-12 for value in deltas.values()) and all(
        value["exact"] for value in hashes.values()
    )
    return {"status": "PASS" if passed else "FAIL", "metric_deltas": deltas, "trace_hashes": hashes}


def main() -> int:
    attempt = next_attempt()
    tracked = engine.hash_snapshot()
    tracked[str(Path(__file__).resolve())] = engine.sha256_file(Path(__file__).resolve())
    write_json(attempt / "audit" / "pre_run_hashes.json", tracked)
    scenario = engine.normal.scenario_gate(
        engine.TRUSTED_SCENARIO,
        attempt / "scenario" / "ordinary_random_seed2026",
        attempt / "audit",
    )
    write_json(attempt / "scenario_identity_report.json", scenario)
    bundle = attempt / "scenario" / "ordinary_random_seed2026"
    specs = {method: make_spec(method, bundle, attempt) for method in METHODS}

    processes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(launch, specs[method], attempt): method for method in METHODS}
        for future in as_completed(futures):
            processes.append(future.result())

    metrics: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        metrics[method], _ = engine.evaluate_run(method, Path(specs[method]["output"]))
    full = metrics[FULL]
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        row = dict(metrics[method])
        row.update(
            {
                "variant": VARIANT[method],
                "delta_J_all_vs_full": float(row["J_all_net"]) - float(full["J_all_net"]),
                "delta_J_HV_vs_full": float(row["J_HV_net"]) - float(full["J_HV_net"]),
                "delta_all_closure_vs_full": int(row["all_closure_count"]) - int(full["all_closure_count"]),
                "delta_HV_closure_vs_full": int(row["HV_closure_count"]) - int(full["HV_closure_count"]),
                "delta_high_closure_vs_full": int(row["high_closure_count"]) - int(full["high_closure_count"]),
                "delta_urgent_closure_vs_full": int(row["urgent_closure_count"]) - int(full["urgent_closure_count"]),
            }
        )
        summary.append(row)
    write_csv(attempt / "four_method_summary.csv", summary)
    write_csv(attempt / "tables" / "four_method_summary.csv", summary)
    write_csv(attempt / "tables" / "paired_task_changes_vs_full.csv", _paired_changes(attempt))

    identities = {
        method: engine.read_json(Path(specs[method]["output"]) / "runtime_identity.json") for method in METHODS
    }
    regression = _full_regression(attempt, metrics[FULL])
    write_json(attempt / "audit" / "full_prop_regression.json", regression)
    post = engine.hash_snapshot()
    post[str(Path(__file__).resolve())] = engine.sha256_file(Path(__file__).resolve())

    guidance = {method: identities[method]["guidance_runtime_audit"] for method in METHODS}
    checks = {
        "four_unique_processes": len({row["pid"] for row in processes}) == 4,
        "source_hashes_unchanged": tracked == post,
        "full_prop_regression_pass": regression["status"] == "PASS",
        "same_task_hash": len({identities[m]["scenario_identity"]["task_hash"] for m in METHODS}) == 1,
        "same_capacity": all(float(identities[m]["capacity_identity"]["runtime_capacity"]) == 0.5 for m in METHODS),
        "all_zero_safety_violations": all(
            int(metrics[m]["soc_violation_count"]) == 0 and int(metrics[m]["thermal_violation_count"]) == 0
            for m in METHODS
        ),
        "full_has_dynamic_prices": guidance[FULL]["nonzero_lambda_call_count"] > 0,
        "full_has_dynamic_modes": len(guidance[FULL]["mode_counts"]) > 1,
        "no_lyapunov_all_prices_zero": guidance[NO_LYAPUNOV]["nonzero_lambda_call_count"] == 0,
        "no_lyapunov_mode_retained": (
            identities[NO_LYAPUNOV]["runtime_router_flags"]["use_mode_controller"]
            and len(guidance[NO_LYAPUNOV]["mode_counts"]) > 1
        ),
        "no_mode_lyapunov_retained": (
            identities[NO_MODE]["runtime_router_flags"]["use_lyapunov_pricing"]
            and guidance[NO_MODE]["nonzero_lambda_call_count"] > 0
        ),
        "no_mode_is_neutral": set(guidance[NO_MODE]["mode_counts"]) == {"no_mode"},
        "all_router_capability_flags_preserved": all(
            identities[m]["runtime_router_flags"]["use_lyapunov_pricing"]
            and identities[m]["runtime_router_flags"]["use_mode_controller"]
            for m in METHODS
        ),
        "no_both_all_prices_zero": guidance[NO_BOTH]["nonzero_lambda_call_count"] == 0,
        "no_both_is_neutral": set(guidance[NO_BOTH]["mode_counts"])
        == {"lyapunov_and_mode_disabled_neutral"},
        "all_guidance_nonnull": all(guidance[m]["none_bundle_count"] == 0 for m in METHODS),
        "hard_safety_and_public_evaluator_unchanged": all(
            identities[m]["hard_safe_filter_unchanged"] and identities[m]["public_evaluator_unchanged"]
            for m in METHODS
        ),
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "processes": sorted(processes, key=lambda row: METHODS.index(row["method"])),
        "guidance": guidance,
        "identities": identities,
    }
    write_json(attempt / "audit" / "method_identity_and_isolation.json", audit)
    (attempt / "test_report.txt").write_text(
        "\n".join(f"{'PASS' if passed else 'FAIL'}\t{name}" for name, passed in checks.items()) + "\n",
        encoding="utf-8",
    )

    by_method = {row["method"]: row for row in summary}
    lines = [
        "# Current PROP slow-control factorial single-seed result",
        "",
        "Ordinary-random V25, seed=2026, capacity=0.5, 24 h, 20 s slots, frozen 40 W environment.",
        "",
        "| method | J_all_net | J_HV_net | all | HV | high | urgent | energy Wh | transmitted MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = by_method[method]
        lines.append(
            f"| {method} | {float(row['J_all_net']):.6f} | {float(row['J_HV_net']):.6f} | "
            f"{row['all_closure_count']} | {row['HV_closure_count']} | {row['high_closure_count']} | "
            f"{row['urgent_closure_count']} | {float(row['controllable_energy_Wh']):.6f} | "
            f"{float(row['transmitted_MB']):.6f} |"
        )
    lines.extend(
        [
            "",
            "All arms keep the router capability flags and progressive/residual gates enabled. "
            "Only the per-slot guidance bundle is changed: zero Lyapunov prices, neutral mode control, or both. "
            "Unlike the earlier broad A3, this clean factorial does not disable z coupling, SOC-band policy, or long-term soft risk configuration.",
            "",
            f"Identity audit: **{audit['status']}**; full regression: **{regression['status']}**.",
            "",
            COMPLETION,
        ]
    )
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        attempt / "completion.json",
        {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "marker": COMPLETION,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output": str(attempt),
        },
    )
    print(f"Unique entry: {Path(__file__).resolve()}")
    print("Execution: four isolated child processes in parallel")
    print(f"Output: {attempt}")
    print(f"Audit: {audit['status']}")
    print(COMPLETION)
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child_main(args.child_spec) if args.child_spec else main())
