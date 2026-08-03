"""Run current PROP, pure myopic greedy, and frozen OFFLOAD on one thermal model.

All three runs use independent Python processes and fresh state.  The OFFLOAD
policy remains the frozen one-shot LOCAL/GROUND earliest-finish-time method;
only its scalar thermal transition is connected to the shared diagnostic
two-node adapter used by PROP.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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

import scripts.run_current_prop_slow_control_factorial_single_seed_v1 as slow  # noqa: E402
import scripts.run_dual_node_thermal_calibration_and_four_method_rerun_v1 as dual  # noqa: E402
import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402

# Load the local selector without importing the workspace ``src`` package.
# Child setup must remain free to bind ``src`` to the frozen runtime tree.
pure_selector = dual.private_module(
    "pure_myopic_safe_greedy_isolated_v2",
    ROOT / "src" / "routers" / "pure_myopic_safe_greedy.py",
)


VERSION = "PURE-MYOPIC-SAFE-GREEDY-OFFLOAD-THERMAL-COMPARISON-v1"
COMPLETION = "PURE_MYOPIC_SAFE_GREEDY_OFFLOAD_THERMAL_SINGLE_SEED_COMPLETE"
CURRENT = "PROP-V2-SAFE-METADATA-CURRENT-v1"
PURE = "PURE-MYOPIC-SAFE-GREEDY-v2"
OFFLOAD = "OFFLOAD-EFT-THERMAL-ALIGNED-2NODE-DIAGNOSTIC"
METHODS = (CURRENT, PURE, OFFLOAD)
SLUG = {method: method.replace("-", "_") for method in METHODS}
OUTPUT_BASE = ROOT / "outputs" / "_pure_myopic_safe_greedy_offload_thermal_single_seed_v1"
SEED, CAPACITY, SLOTS = 2026, 0.5, 4320
TASK_HASH, BUNDLE = dual.TASK_HASH, dual.BUNDLE
PILOT_FROZEN_ROOT = Path("D:/wt/pilot")
OFFLOAD_ROUTER_SOURCE = PILOT_FROZEN_ROOT / "src" / "baselines" / "offload_eft_thermal_aligned.py"
OFFLOAD_CORE_SOURCE = PILOT_FROZEN_ROOT / "src" / "baselines" / "offload_eft.py"
OFFLOAD_METHOD_CONFIG = PILOT_FROZEN_ROOT / "configs" / "methods" / "offload_eft_thermal_aligned_v1.yaml"


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


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    number = 1
    while (OUTPUT_BASE / f"attempt_{number:03d}").exists():
        number += 1
    attempt = OUTPUT_BASE / f"attempt_{number:03d}"
    for name in ("audit", "runs", "run_specs", "logs", "tables"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    return attempt


def reference_thermal_model() -> Path:
    base = ROOT / "outputs" / "_dual_node_thermal_calibration_four_method_rerun_v1"
    for attempt in sorted(base.glob("attempt_*"), reverse=True):
        model, report = attempt / "thermal_model_v1.json", attempt / "test_report.txt"
        if model.is_file() and report.is_file() and "PASS\t" + dual.COMPLETION in report.read_text(encoding="utf-8"):
            if read_json(model).get("accepted") is True:
                return model
    raise RuntimeError("NO_ACCEPTED_REFERENCE_THERMAL_MODEL")


def protected_snapshot() -> dict[str, str]:
    paths = dual.protected_files() + [
        OFFLOAD_ROUTER_SOURCE,
        OFFLOAD_CORE_SOURCE,
        OFFLOAD_METHOD_CONFIG,
        ROOT / "configs" / "prop_v2_safe_metadata_current_v1.yaml",
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths if path.is_file()}


def install_common_thermal(adapter: Any) -> dict[str, Any]:
    return dual.install_runtime_patches(adapter)


def run_current_child(spec: Mapping[str, Any], output: Path, adapter: Any) -> dict[str, Any]:
    patch_audit: dict[str, Any] = {}
    original_setup = engine.v25.child_setup_path

    def setup(path: Path) -> None:
        nonlocal patch_audit
        original_setup(path)
        patch_audit = install_common_thermal(adapter)

    engine.v25.child_setup_path = setup
    local = dict(spec)
    local.update({"flags": [True, True, False], "greedy_no_bandit": True, "fixed_model": False, "fixed_model_name": ""})
    engine.configure_and_run_child(local, output)
    return patch_audit


def run_pure_child(spec: Mapping[str, Any], output: Path, adapter: Any) -> dict[str, Any]:
    patch_audit: dict[str, Any] = {}
    selector_audit: dict[str, Any] = {}
    static_price_audit: dict[str, Any] = {
        "score_call_count": 0,
        "non_defer_score_count": 0,
        "original_dynamic_cost_nonzero_count": 0,
        "energy_price_per_Wh": 0.0,
        "downlink_price_per_MB": 0.0,
        "terminal_backlog_price_used_online": False,
        "future_queue_or_task_information_used": False,
    }
    original_setup = engine.v25.child_setup_path

    def setup(path: Path) -> None:
        nonlocal patch_audit
        original_setup(path)
        patch_audit = install_common_thermal(adapter)
        import src.evaluation.baselines as baselines

        # With Lyapunov/mode guidance disabled, the stock V2 dynamic resource
        # term becomes zero and the score degenerates to gross utility.  A
        # *pure myopic net* baseline must still charge the already-frozen public
        # marginal cost of this action.  These are fixed prices, not stateful
        # long-horizon guidance, and use no future task/queue information.
        frozen = engine.finalgate.load_frozen()
        prices = frozen["public_prices"]
        static_price_audit["energy_price_per_Wh"] = f(prices["energy_price_per_Wh"])
        static_price_audit["downlink_price_per_MB"] = f(prices["downlink_price_per_MB"])
        v2core = engine.v2diag.load_v2core()
        original_score = v2core.score_action_q_once

        def score_with_static_marginal_cost(*args: Any, **kwargs: Any):
            score = original_score(*args, **kwargs)
            predicted = kwargs["predicted"]
            action = kwargs["action"]
            static_price_audit["score_call_count"] += 1
            static_price_audit["original_dynamic_cost_nonzero_count"] += int(abs(f(score.dynamic_resource_cost)) > 1.0e-12)
            if str(getattr(action, "payload_policy", "")) == "defer":
                return replace(score, dynamic_resource_cost=0.0, net_score=0.0)
            static_price_audit["non_defer_score_count"] += 1
            energy_wh = (
                f(getattr(predicted, "predicted_E_comp", 0.0))
                + f(getattr(predicted, "predicted_E_comm", 0.0))
            ) / 3600.0
            payload_mb = f(getattr(predicted, "predicted_S_gen_mb", 0.0))
            current_action_cost = (
                static_price_audit["energy_price_per_Wh"] * energy_wh
                + static_price_audit["downlink_price_per_MB"] * payload_mb
            )
            return replace(
                score,
                dynamic_resource_cost=current_action_cost,
                net_score=f(score.positive_benefit) - current_action_cost + f(score.bias_term) - f(score.safety_margin_penalty),
            )

        v2core.score_action_q_once = score_with_static_marginal_cost

        original_build = baselines.build_routers

        def build_routers(*args: Any, **kwargs: Any):
            routers = original_build(*args, **kwargs)
            pure_selector.install_pure_myopic_safe_greedy(routers[0], selector_audit)
            return routers

        baselines.build_routers = build_routers

    engine.v25.child_setup_path = setup
    local = dict(spec)
    local["variant"] = "no_lyapunov_and_mode"
    slow.configure_and_run_child(local, output)
    identity_path = output / "runtime_identity.json"
    identity = read_json(identity_path)
    identity["pure_myopic_selector_audit"] = selector_audit
    identity["pure_myopic_static_price_audit"] = static_price_audit
    identity["selector_module"] = "src.routers.pure_myopic_safe_greedy"
    identity["selector_module_sha256"] = sha256_file(ROOT / "src" / "routers" / "pure_myopic_safe_greedy.py")
    identity["online_bandit_disabled"] = True
    identity["online_objective"] = "current V*dI*q*p*T - frozen_energy_price*predicted_Wh - frozen_downlink_price*predicted_MB"
    identity["long_term_resource_guidance"] = False
    write_json(identity_path, identity)
    return patch_audit


def reconstruct_common_scenario(method: str, five_runtime: Any) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    tool_dir = str(engine.v25.TOOL_DIR.resolve())
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)
    from native_scenario import reconstruct_native_scenario
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep

    frozen = engine.finalgate.load_frozen()
    core = engine.v2diag.load_v2core()
    core.FROZEN_QUALITY_PROXY.clear()
    core.FROZEN_QUALITY_PROXY.update({key: f(value) for key, value in frozen["q_eff"].items()})
    bundle = engine.normal.load_neutral_bundle(BUNDLE)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    # OFFLOAD does not consume PROP's action grid.  Preserve the trusted bundle
    # and add only the already-frozen public price fields used by accounting.
    configured = engine.calibration.configure_online_prices(
        scenario.configs, {"prices": frozen["public_prices"], "legacy_online_baseline": False}
    )
    scenario = replace(scenario, configs=configured)
    scenario, power = engine.v25.inject_power(scenario, engine.v25.SOURCE_PROP)
    scenario, capacity_identity = engine.guardrun.apply_capacity(scenario, CAPACITY)
    common_identity = engine.exact.scenario_identity(method, SEED, scenario, bundle, power, CAPACITY)
    if common_identity.get("task_hash") != TASK_HASH:
        raise RuntimeError("TASK_MANIFEST_MISMATCH")
    offload_identity = {
        **common_identity,
        "task_manifest_hash_v2": TASK_HASH,
        "environment_hash_v2": five_runtime.canonical_environment_hash_v2(scenario.configs, scenario.timeline[:SLOTS]),
        "capacity_identity": capacity_identity,
        "power_identity": power,
    }
    return scenario, bundle, frozen, offload_identity, capacity_identity


def run_offload_child(spec: Mapping[str, Any], output: Path, adapter: Any) -> dict[str, Any]:
    # The five-method loader is specifically designed to load the frozen
    # OFFLOAD implementation against this workspace's shared physics/evaluator.
    import scripts.run_myopic_e2e_utility_greedy as five_runtime
    import src.energy as energy_package

    external_energy = str((engine.v25.SOURCE_PROP / "src" / "energy").resolve())
    if external_energy not in energy_package.__path__:
        energy_package.__path__.append(external_energy)
    import src.energy.thermal_headroom as thermal_headroom
    import src.energy.thermal_update as thermal_update

    thermal_update.update_thermal_pressure = adapter.update
    thermal_headroom.update_thermal_pressure = adapter.update
    patch_audit = {
        "offload_shared_thermal_update_patched": True,
        "offload_headroom_thermal_update_patched": True,
        "prop_predictor_or_executor_modified": False,
    }

    scenario, _bundle, frozen, identity, _capacity = reconstruct_common_scenario(OFFLOAD, five_runtime)
    simulator_type, _ = five_runtime._load_frozen_external_method_classes()
    module = sys.modules[simulator_type.__module__]
    # OFFLOAD performs a full cloned rollout for both paths of every task.
    # Keep those calculations exact, but do not retain millions of diagnostic
    # rows from the clones.  Real executor steps (record=True) remain logged.
    adapter._offload_record_enabled = False

    class ActualOnlyRows(list):
        def append(self, value: Any) -> None:
            if adapter._offload_record_enabled:
                super().append(value)

    adapter.rows = ActualOnlyRows()
    original_step = simulator_type._step

    def step_with_audit_scope(self: Any, *args: Any, **kwargs: Any):
        record = bool(kwargs.get("record", args[0] if args else False))
        previous = adapter._offload_record_enabled
        adapter._offload_record_enabled = record
        row_start = len(adapter.rows)
        try:
            result = original_step(self, *args, **kwargs)
            if record:
                # The frozen simulator calls the same transition for probes and
                # for the committed state update.  The final call in an actual
                # step is the committed transition; tag only that row as actual.
                new_rows = adapter.rows[row_start:]
                transition_rows = [row for row in new_rows if row.get("caller") == "_transition"]
                if transition_rows:
                    transition_rows[-1]["caller"] = "actual_slot_transition"
                # Export the real committed satellite FIFO occupancy without
                # changing admission, ordering, or transmission behavior.
                if getattr(self, "actual_trace", None):
                    self.actual_trace[-1]["queue_mb"] = sum(
                        f(job.remaining_mb)
                        for job in self.jobs.values()
                        if job.state not in {"CLOSED", "FAILED"}
                    )
            return result
        finally:
            adapter._offload_record_enabled = previous

    simulator_type._step = step_with_audit_scope
    original_dry_run = simulator_type.dry_run

    def dry_run_without_copying_committed_trace(self: Any, *args: Any, **kwargs: Any):
        # ``actual_trace`` is diagnostic history only and is absent from the
        # simulator state fingerprint/decision rule.  Avoid deep-copying an
        # ever-growing 4320-row history into every LOCAL/GROUND candidate clone.
        # The clone still records every future predicted step exactly.
        committed_trace = self.actual_trace
        self.actual_trace = []
        try:
            return original_dry_run(self, *args, **kwargs)
        finally:
            self.actual_trace = committed_trace

    simulator_type.dry_run = dry_run_without_copying_committed_trace
    module.update_thermal_pressure = adapter.update
    observed: list[dict[str, Any]] = []
    original_decide = simulator_type.decide_task

    def decide(self: Any, *args: Any, **kwargs: Any):
        result = original_decide(self, *args, **kwargs)
        observed.append(
            {
                "task_id": result.task_id,
                "selected_path": result.selected_path,
                "selection_reason": result.selection_reason,
                "local_feasible": int(result.local.feasible),
                "ground_feasible": int(result.ground.feasible),
                "local_failure_reason": result.local.failure_reason,
                "ground_failure_reason": result.ground.failure_reason,
                "predicted_local_finish_s": result.local.predicted_finish_s,
                "predicted_ground_finish_s": result.ground.predicted_finish_s,
            }
        )
        return result

    simulator_type.decide_task = decide
    metrics, _ = five_runtime._run_offload_method(SEED, scenario.configs, scenario.timeline[:SLOTS], identity, output)
    normalize_offload_for_public_evaluator(output, scenario.configs, frozen, metrics)
    write_csv(output / "offload_policy_audit.csv", observed)
    runtime_path = output / "runtime_identity.json"
    runtime = read_json(runtime_path)
    runtime.update(
        {
            "method": OFFLOAD,
            "reported_policy_identity": "OFFLOAD-EFT-THERMAL-ALIGNED",
            "policy_source_unchanged": True,
            "policy_source_runtime": str(Path(module.__file__).resolve()),
            "policy_source_sha256": sha256_file(Path(module.__file__).resolve()),
            "frozen_reference_source": str(OFFLOAD_ROUTER_SOURCE.resolve()),
            "frozen_reference_sha256": sha256_file(OFFLOAD_ROUTER_SOURCE),
            "method_config_sha256": sha256_file(OFFLOAD_METHOD_CONFIG),
            "decision_rule": "one-shot LOCAL/GROUND earliest finish; both infeasible -> DEFER",
            "local_model": "high_accuracy",
            "local_product": "roi",
            "ground_servers": 1,
            "ground_queue_capacity": 32,
            "added_net_utility_gate": False,
            "added_requirement_projection": False,
            "thermal_only_adapter": True,
            "thermal_adapter_callable": "LegacyScalarAdapter.update",
            "candidate_dry_run_thermal_rows_retained": False,
            "candidate_dry_run_thermal_calculation_preserved": True,
            "candidate_dry_run_committed_trace_copy_elided": True,
            "committed_trace_not_in_state_fingerprint_or_decision_rule": True,
            "actual_executor_thermal_rows_retained": True,
            "native_outputs_preserved": True,
            "frozen_public_evaluator_applied": True,
            "actual_model_provenance": {"LOCAL": "high_accuracy", "GROUND": "ground_high_accuracy"},
            "public_prices": frozen["public_prices"],
            "q_eff": frozen["q_eff"],
            "pid": os.getpid(),
        }
    )
    write_json(runtime_path, runtime)
    public_metric = read_json(output / "frozen_public_metrics.json")
    write_json(output / "method_completed.json", {"status": "PASS", "method": OFFLOAD, "seed": SEED, "native_metrics_hash": stable_hash(metrics), "public_metrics_hash": stable_hash(public_metric)})
    bound = module.update_thermal_pressure
    same_adapter = getattr(bound, "__self__", None) is adapter and getattr(bound, "__func__", None) is getattr(adapter.update, "__func__", None)
    return {**patch_audit, "offload_module_update_is_shared_adapter": same_adapter}


def normalize_offload_for_public_evaluator(
    output: Path,
    resolved_config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    native_metrics: Mapping[str, Any],
) -> None:
    """Adapt frozen OFFLOAD outputs to the common q-once schema.

    This function is accounting-only.  It runs after the native simulator has
    finished and preserves every native table before writing unified aliases.
    Both frozen OFFLOAD paths execute the configured high-accuracy model:
    LOCAL onboard and GROUND on the shared FIFO ground server.
    """

    native_tasks = read_csv(output / "task_outcomes.csv")
    native_events = read_csv(output / "ground_delivery_events.csv")
    native_trace = read_csv(output / "per_slot_resource_trace.csv")
    write_csv(output / "native_task_outcomes.csv", native_tasks)
    write_csv(output / "native_ground_delivery_events.csv", native_events)
    write_csv(output / "native_per_slot_resource_trace.csv", native_trace)
    write_json(output / "native_metrics.json", dict(native_metrics))

    tasks: list[dict[str, Any]] = []
    for row in native_tasks:
        path = str(row.get("selected_path", "")).upper()
        delivered = engine.gatecore.product_normalize(row.get("transmitted_product_type"))
        closed = i(row.get("on_time_task_success", row.get("requirement_satisfied")))
        tasks.append(
            {
                "method": OFFLOAD,
                "task_id": row["task_id"],
                "arrival_time": row.get("arrival_time_s"),
                "deadline": row.get("deadline_time_s"),
                "value_level": row.get("value_level"),
                "effective_task_value": row.get("effective_task_value"),
                "product_requirement": row.get("product_requirement"),
                "selected_action": path or "DEFER",
                "executed_action": path or "DEFER",
                "selected_model": "high_accuracy" if path == "LOCAL" else ("ground_high_accuracy" if path == "GROUND" else "delay_store"),
                "selected_product": delivered if path in {"LOCAL", "GROUND"} else "defer",
                "delivered_product": delivered if i(row.get("ground_completed")) else "",
                "closure_success": closed,
                "closure_ground_time": row.get("requirement_closure_time_s") if closed else "",
                "failure_reason": row.get("failure_reason", ""),
                "transmitted_bytes": f(row.get("transmitted_bytes")),
            }
        )

    product_stage = {"metadata": "metadata", "roi": "roi", "roi_context": "evidence"}
    events: list[dict[str, Any]] = []
    for row in native_events:
        product = engine.gatecore.product_normalize(row.get("product_type"))
        if product not in product_stage:
            continue
        events.append(
            {
                "method": OFFLOAD,
                "task_id": row["task_id"],
                "stage": product_stage[product],
                "ground_available_time": row.get("ground_available_time_s"),
                "deadline": row.get("deadline_time_s"),
                "valid_before_deadline": row.get("valid_before_deadline"),
                "event_source": "frozen_offload_native:" + str(row.get("delivery_route", "")),
                "duplicate_flag": 0,
            }
        )

    slots: list[dict[str, Any]] = []
    battery_min = f(resolved_config["simulator"].get("battery_min"))
    thermal_max = f(resolved_config["simulator"].get("thermal_max"), 1.0)
    for row in native_trace:
        soc, thermal = f(row.get("soc_after")), f(row.get("thermal_after"))
        slots.append(
            {
                "method": OFFLOAD,
                "slot": row.get("slot"),
                "time_s": row.get("time_s"),
                "battery_soc": soc,
                "thermal": thermal,
                "queue_mb": f(row.get("queue_mb")),
                "compute_energy_j": f(row.get("compute_energy_j")),
                "comm_energy_j": f(row.get("comm_energy_j")),
                "transmitted_mb": f(row.get("sent_mb")),
                "in_contact": i(row.get("in_contact")),
                "soc_violation": int(soc < battery_min - 1.0e-12),
                "thermal_violation": int(thermal > thermal_max + 1.0e-12),
            }
        )
    resource = {
        "method": OFFLOAD,
        "slot_count": len(slots),
        "minimum_soc": min((row["battery_soc"] for row in slots), default=0.0),
        "final_soc": slots[-1]["battery_soc"] if slots else 0.0,
        "maximum_thermal": max((row["thermal"] for row in slots), default=0.0),
        "soc_violation_count": sum(row["soc_violation"] for row in slots),
        "thermal_violation_count": sum(row["thermal_violation"] for row in slots),
        "controllable_energy_wh": f(native_metrics.get("energy_Wh")),
        "transmitted_mb": f(native_metrics.get("transmitted_bytes")) / 1_000_000.0,
        "native_summary": dict(native_metrics),
    }
    write_csv(output / "per_task_outcomes.csv", tasks)
    write_csv(output / "ground_delivery_events.csv", events)
    write_csv(output / "per_slot_resource_trace.csv", slots)
    write_json(output / "resource_summary.json", resource)
    write_json(output / "resolved_config.json", dict(resolved_config))
    metric, ledger = engine.gatecore.evaluate_public_run(OFFLOAD, output, resolved_config, frozen)
    metric.update(
        {
            "seed": SEED,
            "native_policy_J_all_net": native_metrics.get("J_all_net"),
            "native_policy_J_HV_net": native_metrics.get("J_HV_net"),
            "accounting_adapter_only": True,
        }
    )
    write_csv(output / "frozen_public_event_ledger.csv", ledger)
    write_json(output / "frozen_public_metrics.json", metric)


def child_run(spec: Mapping[str, Any]) -> int:
    method, output = str(spec["method"]), Path(spec["output"])
    params = dual.thermal.TwoNodeParameters(**read_json(Path(spec["thermal_model"]))["parameters"])
    adapter = dual.thermal.LegacyScalarAdapter(params)
    if method == CURRENT:
        patch_audit = run_current_child(spec, output, adapter)
    elif method == PURE:
        patch_audit = run_pure_child(spec, output, adapter)
    elif method == OFFLOAD:
        patch_audit = run_offload_child(spec, output, adapter)
    else:
        raise ValueError(method)
    write_csv(output / "dual_node_transition_trace.csv", adapter.rows)
    actual = [row for row in adapter.rows if row.get("caller") in {"run_step", "run", "_step", "actual_slot_transition"}]
    write_json(
        output / "dual_node_runtime_audit.json",
        {
            "method": method, "pid": os.getpid(), "thermal_parameter_hash": params.stable_hash(),
            "patch_audit": patch_audit, "transition_call_count": len(adapter.rows),
            "actual_slot_transition_count": len(actual),
            "max_chip_c": max((f(row.get("peak_chip_c")) for row in actual), default=0.0),
            "max_surface_c": max((f(row.get("peak_surface_c")) for row in actual), default=0.0),
            "chip_violation_count": sum(f(row.get("peak_chip_c")) > params.t_chip_max_c + 1e-12 for row in actual),
            "surface_violation_count": sum(f(row.get("peak_surface_c")) > params.t_surface_max_c + 1e-12 for row in actual),
        },
    )
    return 0


def launch(attempt: Path, model: Path) -> list[dict[str, Any]]:
    processes = []
    for method in METHODS:
        spec = {"version": VERSION, "method": method, "seed": SEED, "capacity": CAPACITY, "slots": SLOTS, "bundle_root": str(BUNDLE), "thermal_model": str(model), "output": str(attempt / "runs" / SLUG[method])}
        spec_path = attempt / "run_specs" / f"{SLUG[method]}.json"
        write_json(spec_path, spec)
        stdout = (attempt / "logs" / f"{SLUG[method]}.stdout.log").open("w", encoding="utf-8")
        stderr = (attempt / "logs" / f"{SLUG[method]}.stderr.log").open("w", encoding="utf-8")
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)], cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
        processes.append((method, process, stdout, stderr))
    statuses = []
    for method, process, stdout, stderr in processes:
        returncode = process.wait(); stdout.close(); stderr.close()
        statuses.append({"method": method, "pid": process.pid, "returncode": returncode, "status": "PASS" if returncode == 0 else "FAIL", "output": str(attempt / "runs" / SLUG[method])})
    return statuses


def evaluate(method: str, run: Path) -> dict[str, Any]:
    if method == OFFLOAD:
        return read_json(run / "frozen_public_metrics.json")
    metric, _ = engine.evaluate_run(method, run)
    return metric


def summary_row(method: str, run: Path, metric: Mapping[str, Any]) -> dict[str, Any]:
    thermal = read_json(run / "dual_node_runtime_audit.json")
    trace = read_csv(run / "per_slot_resource_trace.csv")
    queues = [f(row.get("queue_mb")) for row in trace]
    all_count = i(metric.get("all_closure_count", metric.get("closure_count")))
    hv_count = i(metric.get("HV_closure_count"))
    return {
        "method": method, "J_all_net": metric.get("J_all_net"), "J_HV_net": metric.get("J_HV_net"),
        "gross_utility": metric.get("gross_utility", metric.get("gross_delivered_utility")),
        "all_closure_count": all_count, "HV_closure_count": hv_count,
        "high_closure_count": metric.get("high_closure_count"), "urgent_closure_count": metric.get("urgent_closure_count"),
        "ordinary_closure_count": metric.get("ordinary_closure_count", all_count - hv_count),
        "mean_closure_latency_s": metric.get("mean_full_closure_latency_s", metric.get("successful_closure_mean_latency_s")),
        "p95_closure_latency_s": metric.get("p95_full_closure_latency_s", metric.get("successful_closure_p95_latency_s")),
        "controllable_energy_Wh": metric.get("controllable_energy_Wh"), "transmitted_MB": metric.get("transmitted_MB"),
        "peak_queue_MB": metric.get("peak_queue_MB", max(queues, default=0.0)),
        "mean_queue_MB": sum(queues) / max(len(queues), 1), "terminal_queue_MB": metric.get("terminal_queue_MB"),
        "minimum_soc": metric.get("minimum_soc", metric.get("min_SOC")),
        "maximum_chip_c": thermal["max_chip_c"], "maximum_surface_c": thermal["max_surface_c"],
        "soc_violation_count": metric.get("soc_violation_count", 0),
        "thermal_violation_count": thermal["chip_violation_count"] + thermal["surface_violation_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--child-spec", type=Path); args = parser.parse_args()
    if args.child_spec:
        return child_run(read_json(args.child_spec))
    attempt, model = next_attempt(), reference_thermal_model()
    pre = protected_snapshot()
    if read_json(BUNDLE / "bundle_contract.json").get("task_hash") != TASK_HASH:
        raise RuntimeError("TASK_MANIFEST_MISMATCH")
    statuses = launch(attempt, model)
    summaries, metrics = [], {}
    for status in statuses:
        if status["status"] == "PASS":
            method = status["method"]; run = attempt / "runs" / SLUG[method]
            metric = evaluate(method, run); metrics[method] = metric; summaries.append(summary_row(method, run, metric))
    write_csv(attempt / "three_method_summary.csv", summaries)
    write_csv(attempt / "tables" / "three_method_summary.csv", summaries)

    pure_identity = read_json(attempt / "runs" / SLUG[PURE] / "runtime_identity.json") if PURE in metrics else {}
    offload_identity = read_json(attempt / "runs" / SLUG[OFFLOAD] / "runtime_identity.json") if OFFLOAD in metrics else {}
    decisions = read_csv(attempt / "runs" / SLUG[OFFLOAD] / "offload_policy_audit.csv") if OFFLOAD in metrics else []
    path_counts = {path: sum(row.get("selected_path") == path for row in decisions) for path in ("LOCAL", "GROUND", "DEFER")}
    pure_guidance = pure_identity.get("guidance_runtime_audit", {})
    pure_selector = pure_identity.get("pure_myopic_selector_audit", {})
    pure_static = pure_identity.get("pure_myopic_static_price_audit", {})
    post = protected_snapshot()
    audit = {
        "version": VERSION, "statuses": statuses, "parallel_independent_processes": len({row["pid"] for row in statuses}) == 3,
        "task_hash": TASK_HASH, "seed": SEED, "capacity": CAPACITY, "ordinary_random_scenario": True, "correct_40w_environment": True,
        "thermal_model_source": str(model.resolve()), "thermal_model_sha256": sha256_file(model),
        "same_thermal_hash_all_methods": len({read_json(attempt / "runs" / SLUG[m] / "dual_node_runtime_audit.json")["thermal_parameter_hash"] for m in metrics}) == 1,
        "pure_zero_lambda_calls": pure_guidance.get("zero_lambda_call_count", 0),
        "pure_nonzero_lambda_calls": pure_guidance.get("nonzero_lambda_call_count", 0),
        "pure_mode_counts": pure_guidance.get("mode_counts", {}),
        "pure_selector_nonzero_guidance_violations": pure_selector.get("nonzero_guidance_violation_count", -1),
        "pure_bandit_updates_blocked": pure_selector.get("blocked_online_update_count", 0),
        "pure_static_price_audit": pure_static,
        "offload_path_counts": path_counts,
        "offload_policy_source_unchanged": offload_identity.get("policy_source_unchanged", False),
        "offload_added_net_gate": offload_identity.get("added_net_utility_gate", True),
        "offload_added_reqalign": offload_identity.get("added_requirement_projection", True),
        "offload_frozen_public_evaluator_applied": offload_identity.get("frozen_public_evaluator_applied", False),
        "protected_assets_unchanged_during_run": pre == post,
    }
    write_json(attempt / "method_environment_audit.json", audit)

    rows = {row["method"]: row for row in summaries}
    deltas = []
    if CURRENT in rows:
        for method in (PURE, OFFLOAD):
            if method in rows:
                deltas.append({"method": method, **{f"delta_{key}_vs_current": f(rows[method].get(key)) - f(rows[CURRENT].get(key)) for key in ("J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count")}})
    write_csv(attempt / "paired_metric_deltas.csv", deltas)
    tests = {
        "three_runs_pass": len(metrics) == 3 and all(row["status"] == "PASS" for row in statuses),
        "three_independent_processes": audit["parallel_independent_processes"],
        "same_thermal_model": audit["same_thermal_hash_all_methods"],
        "pure_guidance_zero": i(audit["pure_nonzero_lambda_calls"]) == 0 and i(audit["pure_zero_lambda_calls"]) > 0,
        "pure_selector_saw_no_guidance": i(audit["pure_selector_nonzero_guidance_violations"]) == 0,
        "pure_bandit_disabled": i(audit["pure_bandit_updates_blocked"]) > 0,
        # ``original_dynamic_cost_nonzero_count`` is explicitly a shadow value
        # observed *before* the fixed marginal-cost replacement.  It must not
        # be required to be zero: every returned score is overwritten below,
        # and the independent guidance audit is the authority for lambda=0.
        "pure_fixed_current_action_prices_active": i(pure_static.get("score_call_count")) > 0 and f(pure_static.get("energy_price_per_Wh")) > 0 and f(pure_static.get("downlink_price_per_MB")) > 0 and not pure_static.get("terminal_backlog_price_used_online", True) and not pure_static.get("future_queue_or_task_information_used", True),
        "offload_original_policy_unchanged": bool(audit["offload_policy_source_unchanged"]) and not audit["offload_added_net_gate"] and not audit["offload_added_reqalign"],
        "offload_public_accounting_common": bool(audit["offload_frozen_public_evaluator_applied"]) and i(metrics.get(OFFLOAD, {}).get("missing_event_model_count")) == 0 and f(metrics.get(OFFLOAD, {}).get("q_once_max_recompute_error")) <= 1.0e-12,
        "offload_kept_local_and_ground_paths": path_counts["LOCAL"] > 0 and path_counts["GROUND"] > 0,
        "zero_safety_violations": all(i(row["soc_violation_count"]) == 0 and i(row["thermal_violation_count"]) == 0 for row in summaries),
        "protected_assets_unchanged": audit["protected_assets_unchanged_during_run"],
    }
    (attempt / "test_report.txt").write_text("\n".join(("PASS" if ok else "FAIL") + "\t" + key for key, ok in tests.items()) + "\n" + ("PASS\t" if all(tests.values()) else "FAIL\t") + COMPLETION + "\n", encoding="utf-8")
    lines = ["# PURE-MYOPIC与OFFLOAD双节点热环境单seed对比", "", "| method | J_all | J_HV | all/HV/high/urgent | energy Wh | transmitted MB |", "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['method']} | {f(row['J_all_net']):.9f} | {f(row['J_HV_net']):.9f} | {i(row['all_closure_count'])}/{i(row['HV_closure_count'])}/{i(row['high_closure_count'])}/{i(row['urgent_closure_count'])} | {f(row['controllable_energy_Wh']):.6f} | {f(row['transmitted_MB']):.6f} |")
    lines.extend(["", f"PURE guidance: zero/nonzero calls = {audit['pure_zero_lambda_calls']}/{audit['pure_nonzero_lambda_calls']}; bandit updates blocked = {audit['pure_bandit_updates_blocked']}.", f"OFFLOAD paths LOCAL/GROUND/DEFER = {path_counts['LOCAL']}/{path_counts['GROUND']}/{path_counts['DEFER']}; no new net gate or REQALIGN = {not audit['offload_added_net_gate'] and not audit['offload_added_reqalign']}.", "", COMPLETION])
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    success = all(tests.values())
    print("PIDS:", ", ".join(str(row["pid"]) for row in statuses)); print("OUTPUT:", attempt); print("METHOD STATUS:", json.dumps(statuses, ensure_ascii=False)); print(COMPLETION if success else "PURE_MYOPIC_OFFLOAD_COMPARISON_FAILED")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
