"""Five-method frozen V25 SAR payload SOC pressure diagnostic.

The SAR debit is a common environment hook applied before the online decision
at each physical acquisition event.  It changes only battery SOC via the
existing battery update function; task flow, method parameters, solar input,
base power, compute/communication energy and thermal dynamics are untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402
import scripts.run_prop_reqalign_seed2026_v1 as reqrun  # noqa: E402


SEED = 2026
CAPACITY = 0.35
SLOTS = 4320
SMOKE_SLOTS = 240
METHODS = (
    exact.METHOD_B1,
    exact.METHOD_OFFLOAD,
    exact.METHOD_TUT,
    exact.METHOD_MYOPIC,
    exact.METHOD_CORRECTED,
)
PROFILES = {
    "0W": {"enabled": False, "power_w": 0.0, "duration_s": 10.0},
    "800W_x_10s": {"enabled": True, "power_w": 800.0, "duration_s": 10.0},
    "1600W_x_10s": {"enabled": True, "power_w": 1600.0, "duration_s": 10.0},
    "3870W_x_10s": {"enabled": True, "power_w": 3870.0, "duration_s": 10.0},
}
SOURCE_BUNDLE = exact.SOURCE_BUNDLE
SAR_MODULE_PATH = ROOT / "src" / "energy" / "sar_payload_energy.py"
SAR_CONFIG_PATH = ROOT / "configs" / "sar_payload_energy_v1.yaml"
REFERENCE_ROOT = ROOT / "outputs" / "_v25_exact_native_five_method_controlled_fiveseed_cap035_v1" / "runs" / "seed_2026"
DEFAULT_OUTPUT = ROOT / "outputs" / "_v25_sar_soc_pressure_seed2026_cap035_v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    exact.write_csv(path, rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sar_module() -> Any:
    name = "common_sar_payload_energy_v1"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SAR_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SAR_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def profile_config(profile: str) -> Any:
    values = PROFILES[profile]
    module = load_sar_module()
    return module.SarPayloadConfig(
        sar_payload_enabled=bool(values["enabled"]),
        sar_active_power_w=float(values["power_w"]),
        sar_active_duration_s=float(values["duration_s"]),
        charge_once_per_acquisition=True,
    )


def task_rows(bundle_root: Path) -> list[dict[str, str]]:
    return read_csv(bundle_root / "task_manifest.csv")


def schedule(bundle_root: Path, profile: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    config = profile_config(profile)
    mapping, events = load_sar_module().build_acquisition_schedule(task_rows(bundle_root), config)
    return mapping, events, config


def make_controller(method: str, profile: str, events: list[dict[str, Any]], scenario: Any) -> Any:
    from src.energy.battery_update import update_battery_soc

    simulator = scenario.configs["simulator"]
    return load_sar_module().SarSocDebitController(
        events,
        battery_capacity_j=float(simulator["battery_capacity_j"]),
        discharge_efficiency=float(simulator.get("battery_discharge_efficiency", 1.0)),
        battery_update=update_battery_soc,
        method=method,
        profile=profile,
    )


def bind_episode_runner(runner: Any, controller: Any) -> None:
    original_factory = runner.step_runner_factory

    def factory(queue: Any) -> Any:
        step_runner = original_factory(queue)
        original_run_step = step_runner.run_step

        def intercepted(timeline_step: Any, resource_state: Any, mission_state: Any, comm_state: Any, router: Any):
            controller.debit_resource_state(resource_state, int(timeline_step.slot_index))
            return original_run_step(timeline_step, resource_state, mission_state, comm_state, router)

        step_runner.run_step = intercepted
        return step_runner

    runner.step_runner_factory = factory


def run_prop_family(
    method: str,
    profile: str,
    bundle_root: Path,
    output: Path,
    capacity: float,
    slots: int,
    smoke: bool,
) -> None:
    sar = load_sar_module()
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    prop_mod = v25.import_prop_script()
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configs = prop_mod._configure(scenario.configs, calibration, {"enabled": False})
    scenario = replace(scenario, configs=configs)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, capacity)
    identity = exact.scenario_identity(method, SEED, scenario, bundle, power, capacity)
    identity["capacity_identity"] = capacity_identity
    _mapping, events, config = schedule(bundle_root, profile)
    controller = make_controller(method, profile, events, scenario)

    environment = SimulationEnvironment(scenario.configs)
    runner = environment.build_episode_runner(scenario.timeline[:slots])
    if method == exact.METHOD_CORRECTED:
        delegate = build_routers(scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"]))[0]
    else:
        delegate = exact.load_myopic_router_class()(v25.MYOPIC_PARAM, router_name="MYOPIC-FROZEN-CALIBRATED")
        exact.bind_myopic_runtime(runner, delegate)
    wrapper = reqrun.RequirementAwareRouterWrapper(
        delegate,
        reqrun.load_projection_config(True),
        scenario.timeline[:slots],
        scenario.configs["simulator"],
        reqrun.load_projection_module(),
    )
    reqrun.bind_projection_runtime(runner, wrapper)
    if config.sar_payload_enabled:
        bind_episode_runner(runner, controller)
    result = runner.run(wrapper)
    normalized = normalize_episode(method, scenario, result, runner, slots)
    output.mkdir(parents=True, exist_ok=False)
    if not smoke:
        v25.write_unified_method_outputs(output, normalized)
        v25.write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
        reqrun.enrich_projection_rows(wrapper, output)
        reqrun.write_energy_components(output, result)
    else:
        write_json(output / "smoke_normalized_counts.json", {
            "tasks": len(normalized["per_task"]), "events": len(normalized["events"]), "slots": len(normalized["slots"])
        })
    write_csv(output / "sar_soc_debit_trace.csv", controller.trace)
    write_json(output / "sar_run_identity.json", {
        "method": method, "profile": profile, "slots": slots,
        "sar_enabled": config.sar_payload_enabled,
        "physical_acquisition_count": len(events),
        "actual_debit_count": len(controller.trace),
        "scheduled_sar_energy_wh": sum(float(row["sar_energy_wh"]) for row in events),
        "scenario_identity": identity,
    })
    if not smoke:
        write_json(output / "native_identity.json", {"method": method, "source_root": str(v25.SOURCE_PROP), "scenario_identity": identity})
        write_json(output / "reconstructed_scenario_identity.json", identity)
        write_json(output / "method_run_status.json", {"status": "PASS", "method": method, "seed": SEED})
        write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": SEED})


def run_baseline(
    method: str,
    profile: str,
    bundle_root: Path,
    output: Path,
    capacity: float,
    slots: int,
    smoke: bool,
) -> None:
    load_sar_module()
    source_root = v25.SOURCE_TUT if method == exact.METHOD_TUT else v25.SOURCE_B1_OFFLOAD
    v25.child_setup_path(source_root)
    bundle = v25.load_bundle(bundle_root)
    module = __import__({exact.METHOD_B1: "native_b1", exact.METHOD_OFFLOAD: "native_offload", exact.METHOD_TUT: "native_tut"}[method])
    binding = module.make_binding(source_root.resolve())
    source_identity = dict(binding.identity())
    if not source_identity.get("source_identity_passed", False):
        raise RuntimeError({"native_source_identity_failed": source_identity})
    scenario = binding.build_scenario(bundle)
    scenario, power = v25.inject_power(scenario, source_root)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, capacity)
    identity = exact.scenario_identity(method, SEED, scenario, bundle, power, capacity)
    identity["capacity_identity"] = capacity_identity
    _mapping, events, config = schedule(bundle_root, profile)
    controller = make_controller(method, profile, events, scenario)

    restore: tuple[Any, str, Any] | None = None
    if config.sar_payload_enabled and method == exact.METHOD_B1:
        from src.baselines.b1_enhanced import B1EnhancedSimulator
        original = B1EnhancedSimulator._create_task

        def wrapped(simulator: Any, step: Any) -> None:
            controller.debit_attribute(simulator, "_battery_soc", int(step.slot_index))
            return original(simulator, step)

        restore = (B1EnhancedSimulator, "_create_task", original)
        B1EnhancedSimulator._create_task = wrapped
    elif config.sar_payload_enabled and method == exact.METHOD_OFFLOAD:
        from src.baselines.offload_eft_thermal_aligned import ThermalAlignedSimulator
        original = ThermalAlignedSimulator.decide_task

        def wrapped(simulator: Any, *args: Any, **kwargs: Any):
            controller.debit_attribute(simulator, "soc", int(simulator.timeline[simulator.current_index].slot_index))
            return original(simulator, *args, **kwargs)

        restore = (ThermalAlignedSimulator, "decide_task", original)
        ThermalAlignedSimulator.decide_task = wrapped
    elif config.sar_payload_enabled and method == exact.METHOD_TUT:
        from src.simulator.environment import SimulationEnvironment
        original = SimulationEnvironment.build_episode_runner

        def wrapped(environment: Any, *args: Any, **kwargs: Any):
            runner = original(environment, *args, **kwargs)
            bind_episode_runner(runner, controller)
            return runner

        restore = (SimulationEnvironment, "build_episode_runner", original)
        SimulationEnvironment.build_episode_runner = wrapped
    try:
        native = binding.simulate(scenario, slots)
    finally:
        if restore is not None:
            setattr(*restore)
    normalized = dict(binding.normalize_outputs(native, bundle))
    output.mkdir(parents=True, exist_ok=False)
    if not smoke:
        v25.write_unified_method_outputs(output, normalized)
    else:
        write_json(output / "smoke_normalized_counts.json", {
            "tasks": len(normalized["per_task"]), "events": len(normalized["events"]), "slots": len(normalized["slots"])
        })
    write_csv(output / "sar_soc_debit_trace.csv", controller.trace)
    write_json(output / "sar_run_identity.json", {
        "method": method, "profile": profile, "slots": slots,
        "sar_enabled": config.sar_payload_enabled,
        "physical_acquisition_count": len(events),
        "actual_debit_count": len(controller.trace),
        "scheduled_sar_energy_wh": sum(float(row["sar_energy_wh"]) for row in events),
        "scenario_identity": identity,
    })
    if not smoke:
        write_json(output / "native_identity.json", {**source_identity, "method": method, "scenario_identity": identity})
        write_json(output / "reconstructed_scenario_identity.json", identity)
        write_json(output / "method_run_status.json", {"status": "PASS", "method": method, "seed": SEED})
        write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": SEED})


def child_main(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    method = str(args.method)
    if method in {exact.METHOD_CORRECTED, exact.METHOD_MYOPIC}:
        run_prop_family(method, args.profile, Path(args.bundle_root), output, args.capacity, args.slots, args.smoke)
    else:
        run_baseline(method, args.profile, Path(args.bundle_root), output, args.capacity, args.slots, args.smoke)
    print(json.dumps({"status": "PASS", "method": method, "profile": args.profile, "smoke": args.smoke}, ensure_ascii=False))
    return 0


def launch(method: str, profile: str, output: Path, root: Path, *, slots: int, smoke: bool) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--child-run", "--method", method,
               "--profile", profile, "--bundle-root", str(SOURCE_BUNDLE), "--output", str(output),
               "--capacity", str(CAPACITY), "--slots", str(slots)]
    if smoke:
        command.append("--smoke")
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    logs = root / "logs" / ("smoke" if smoke else "formal")
    logs.mkdir(parents=True, exist_ok=True)
    safe = f"{profile}__{method}".replace("/", "_")
    (logs / f"{safe}.stdout.log").write_text(process.stdout, encoding="utf-8")
    (logs / f"{safe}.stderr.log").write_text(process.stderr, encoding="utf-8")
    status = {"method": method, "profile": profile, "smoke": smoke, "returncode": process.returncode,
              "stdout_tail": process.stdout[-1000:], "stderr_tail": process.stderr[-4000:]}
    if process.returncode != 0:
        raise RuntimeError({"SAR_RUNTIME_FAILED": status})
    return status


def soc_metrics(run_dir: Path) -> dict[str, Any]:
    slots = read_csv(run_dir / "per_slot_resource_trace.csv")
    sar_trace = read_csv(run_dir / "sar_soc_debit_trace.csv")
    values = [(float(row["battery_soc"]), float(row["time_s"])) for row in slots]
    values.extend((float(row["soc_after_sar"]), float(row["event_time_s"])) for row in sar_trace if row.get("soc_after_sar") not in (None, ""))
    minimum_soc, minimum_time_s = min(values, key=lambda pair: (pair[0], pair[1]))
    return {
        "minimum_soc_including_instantaneous_sar": minimum_soc,
        "minimum_soc_time_s": minimum_time_s,
        "final_soc": float(slots[-1]["battery_soc"]),
        "slots_soc_below_0_80": sum(float(row["battery_soc"]) < 0.80 for row in slots),
        "duration_soc_below_0_80_s": 20 * sum(float(row["battery_soc"]) < 0.80 for row in slots),
        "slots_soc_below_0_70": sum(float(row["battery_soc"]) < 0.70 for row in slots),
        "duration_soc_below_0_70_s": 20 * sum(float(row["battery_soc"]) < 0.70 for row in slots),
    }


def recovery_samples(profile: str, method: str, run_dir: Path) -> list[dict[str, Any]]:
    slots = read_csv(run_dir / "per_slot_resource_trace.csv")
    tasks = {int(float(row["slot_index"])): row.get("task_id", "") for row in read_csv(SOURCE_BUNDLE / "slot_trace.csv")}
    solar = read_csv(v25.SOLAR_TRACE)
    samples = []
    for index in range(1, len(slots)):
        row = slots[index]
        previous = slots[index - 1]
        slot = int(float(row["slot"]))
        solar_row = solar[slot % len(solar)]
        idle = (
            not tasks.get(slot)
            and abs(float(row.get("compute_energy_j", 0) or 0)) < 1e-12
            and abs(float(row.get("comm_energy_j", 0) or 0)) < 1e-12
        )
        delta = float(row["battery_soc"]) - float(previous["battery_soc"])
        if idle and solar_row["orbit_phase"] == "daylight" and delta > 1e-8:
            samples.append({"profile": profile, "method": method, "slot": slot, "time_s": row["time_s"],
                            "solar_power_w": solar_row["solar_power_W"], "soc_before": previous["battery_soc"],
                            "soc_after": row["battery_soc"], "soc_recovery": delta})
            if len(samples) >= 5:
                break
    return samples


def make_figure(output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for axis, profile in zip(axes.flat, PROFILES):
        for method in METHODS:
            rows = read_csv(output / "runs" / profile / method / "per_slot_resource_trace.csv")
            axis.plot([float(row["time_s"]) / 3600.0 for row in rows], [float(row["battery_soc"]) for row in rows],
                      linewidth=1.0, label=method)
        axis.axhline(0.80, color="gray", linestyle="--", linewidth=0.8)
        axis.axhline(0.70, color="red", linestyle=":", linewidth=0.8)
        axis.set_title(profile)
        axis.set_xlabel("Time (h)")
        axis.set_ylabel("SOC")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    (output / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "figures" / "soc_pressure_trajectories.png", dpi=300)
    fig.savefig(output / "figures" / "soc_pressure_trajectories.pdf")
    plt.close(fig)


def main(output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SAR_CONFIG_PATH, output / "config" / SAR_CONFIG_PATH.name)
    mapping, base_events, _config = schedule(SOURCE_BUNDLE, "0W")
    if len(mapping) != 268 or len(base_events) != 50:
        raise RuntimeError({"ACQUISITION_MAPPING_MISMATCH": {"tasks": len(mapping), "acquisitions": len(base_events)}})
    write_csv(output / "audit" / "acquisition_task_mapping.csv", mapping)
    write_csv(output / "audit" / "physical_acquisition_events_0W.csv", base_events)
    write_json(output / "audit" / "acquisition_mapping_summary.json", {
        "task_count": len(mapping), "physical_acquisition_count": len(base_events),
        "identity_sources": sorted({row["identity_source"] for row in mapping}),
        "max_derived_tasks_per_acquisition": max(row["derived_task_count"] for row in mapping),
        "charge_once_per_acquisition": True,
    })

    test = subprocess.run([sys.executable, "-m", "pytest", "tests/test_sar_payload_energy.py", "-q"], cwd=ROOT, text=True, capture_output=True)
    (output / "smoke_test_report.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    if test.returncode != 0:
        raise RuntimeError("SAR_UNIT_TEST_FAILED")

    statuses = []
    for method in METHODS:
        statuses.append(launch(method, "1600W_x_10s", output / "smoke" / method, output, slots=SMOKE_SLOTS, smoke=True))
    write_json(output / "audit" / "smoke_run_status.json", statuses)

    jobs = []
    for profile in PROFILES:
        _mapping, events, config = schedule(SOURCE_BUNDLE, profile)
        write_csv(output / "audit" / f"physical_acquisition_events_{profile}.csv", events)
        for method in METHODS:
            jobs.append((profile, method))
    # OFFLOAD is much slower than the other native methods.  Start its four
    # fully isolated configurations first so wall-clock time stays suitable for
    # this diagnostic; no simulator state or output is shared across workers.
    jobs.sort(key=lambda item: (item[1] != exact.METHOD_OFFLOAD, tuple(PROFILES).index(item[0]), METHODS.index(item[1])))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                launch,
                method,
                profile,
                output / "runs" / profile / method,
                output,
                slots=SLOTS,
                smoke=False,
            ): (profile, method)
            for profile, method in jobs
        }
        for future in as_completed(futures):
            statuses.append(future.result())
    write_json(output / "audit" / "all_run_status.json", statuses)

    eval_mod = v25.load_eval_module(output)
    metrics = []
    by_key = {}
    recovery = []
    for profile in PROFILES:
        _mapping, events, config = schedule(SOURCE_BUNDLE, profile)
        total_sar_wh = sum(float(row["sar_energy_wh"]) for row in events)
        for method in METHODS:
            run_dir = output / "runs" / profile / method
            metric = exact.evaluate_method(method, run_dir, SOURCE_BUNDLE, eval_mod, CAPACITY)
            row = {
                "profile": profile, "sar_payload_enabled": config.sar_payload_enabled,
                "sar_active_power_w": config.sar_active_power_w, "sar_active_duration_s": config.sar_active_duration_s,
                "physical_acquisition_count": len(events), "sar_total_energy_Wh": total_sar_wh,
                "method": method, "J_all_net": metric["J_all_net"], "J_HV_net": metric["J_HV_net"],
                "HV_closure_count": metric["HV_closure_count"], "high_closure_count": metric["high_closure_count"],
                "urgent_closure_count": metric["urgent_closure_count"],
                "soc_violation_count": metric["soc_violation_count"], "thermal_violation_count": metric["thermal_violation_count"],
                **soc_metrics(run_dir),
            }
            metrics.append(row)
            by_key[(profile, method)] = row
            recovery.extend(recovery_samples(profile, method, run_dir))

    for row in metrics:
        baseline = by_key[("0W", row["method"])]
        for key in ("J_all_net", "J_HV_net", "HV_closure_count", "high_closure_count", "urgent_closure_count"):
            row[f"delta_{key}_vs_0W"] = Decimal(str(row[key])) - Decimal(str(baseline[key]))
    write_csv(output / "tables" / "sar_soc_pressure_metrics.csv", metrics)
    exact.write_md(output / "tables" / "sar_soc_pressure_metrics.md", metrics)
    write_csv(output / "tables" / "daylight_idle_soc_recovery_samples.csv", recovery)

    core_files = ("per_task_outcomes.csv", "ground_delivery_events.csv", "per_slot_resource_trace.csv", "resource_summary.json")
    regression = []
    for method in METHODS:
        for name in core_files:
            observed = sha256_file(output / "runs" / "0W" / method / name)
            expected = sha256_file(REFERENCE_ROOT / method / name)
            regression.append({"method": method, "file": name, "disabled_sha256": observed,
                               "reference_sha256": expected, "exact_match": observed == expected})
    write_csv(output / "audit" / "sar_disabled_exact_regression.csv", regression)
    if not all(row["exact_match"] for row in regression):
        raise RuntimeError("SAR_DISABLED_REGRESSION_MISMATCH")

    energy_audit = []
    for profile in PROFILES:
        expected_wh = float(by_key[(profile, METHODS[0])]["sar_total_energy_Wh"])
        schedule_hashes = []
        for method in METHODS:
            identity = json.loads((output / "runs" / profile / method / "sar_run_identity.json").read_text(encoding="utf-8"))
            schedule_hashes.append((method, identity["scheduled_sar_energy_wh"]))
        same = all(abs(float(value) - expected_wh) < 1e-9 for _, value in schedule_hashes)
        for method, value in schedule_hashes:
            energy_audit.append({"profile": profile, "method": method, "sar_total_energy_Wh": value,
                                 "expected_common_energy_Wh": expected_wh, "same_energy_across_methods": same})
    write_csv(output / "audit" / "common_sar_energy_identity.csv", energy_audit)
    if not all(row["same_energy_across_methods"] for row in energy_audit):
        raise RuntimeError("SAR_ENERGY_IDENTITY_MISMATCH")

    make_figure(output)
    rows_1600 = [row for row in metrics if row["profile"] == "1600W_x_10s"]
    safe_1600 = all(int(row["soc_violation_count"]) == 0 and int(row["thermal_violation_count"]) == 0 for row in rows_1600)
    drops = [float(by_key[("0W", row["method"])]["minimum_soc_including_instantaneous_sar"]) - float(row["minimum_soc_including_instantaneous_sar"]) for row in rows_1600]
    clear_pressure = min(drops) >= 0.05 or any(int(row["slots_soc_below_0_80"]) > int(by_key[("0W", row["method"])]["slots_soc_below_0_80"]) for row in rows_1600)
    assessment = {
        "status": "SAR_SOC_PRESSURE_DIAGNOSTIC_COMPLETE",
        "seed": SEED, "capacity": CAPACITY, "physical_acquisition_count": 50,
        "sar_1600w_clear_pressure": clear_pressure, "sar_1600w_safe_all_methods": safe_1600,
        "recommend_5seeds_x_3capacities": bool(clear_pressure and safe_1600),
        "recommendation": (
            "1600 W forms visible and safe SOC pressure; a preregistered 5 seeds x 3 capacities run is reasonable."
            if clear_pressure and safe_1600 else
            "Do not expand yet: 1600 W did not satisfy both visible-pressure and zero-violation checks."
        ),
    }
    write_json(output / "assessment.json", assessment)
    (output / "final_assessment.md").write_text(
        "# SAR SOC pressure diagnostic\n\n"
        f"Physical acquisitions: 50 (268 tasks mapped by product identifier).\n\n"
        f"1600 W clear pressure: {clear_pressure}; safe across all five methods: {safe_1600}.\n\n"
        f"Recommendation: {assessment['recommendation']}\n",
        encoding="utf-8",
    )
    write_json(output / "completion.json", assessment)
    finalizer = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "finalize_sar_soc_pressure_v1.py"), "--output", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (output / "logs" / "finalizer.stdout.log").write_text(finalizer.stdout, encoding="utf-8")
    (output / "logs" / "finalizer.stderr.log").write_text(finalizer.stderr, encoding="utf-8")
    if finalizer.returncode != 0:
        raise RuntimeError({"SAR_FINALIZER_FAILED": finalizer.stderr[-4000:]})
    print(json.dumps(assessment, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--method")
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--bundle-root", type=Path, default=SOURCE_BUNDLE)
    parser.add_argument("--capacity", type=float, default=CAPACITY)
    parser.add_argument("--slots", type=int, default=SLOTS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.child_run:
        raise SystemExit(child_main(args))
    raise SystemExit(main(args.output))
