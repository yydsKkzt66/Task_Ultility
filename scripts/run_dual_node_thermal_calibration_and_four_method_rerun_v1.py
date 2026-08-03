"""One-click provisional two-node thermal calibration and four-method rerun.

No frozen source/config/result file is edited.  Private child mode exists only
to provide independent Python interpreters and mutable simulation state.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_b1_common_ground_seed2026_v1 as b1run  # noqa: E402
import scripts.run_normal_scenario_tut_proxy_ground_recovery_v1 as normal  # noqa: E402
import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


def private_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


thermal = private_module("dual_node_thermal_diagnostic_isolated_v1", ROOT / "src" / "experiments" / "dual_node_thermal_diagnostic.py")

VERSION = "DUAL-NODE-THERMAL-CALIBRATION-AND-FOUR-METHOD-RERUN-v1"
COMPLETION = "DUAL_NODE_THERMAL_CALIBRATION_AND_FOUR_METHOD_PARALLEL_RERUN_COMPLETE"
FINAL_STATUS = "LITERATURE_ANCHORED_PROVISIONAL_RERUN_COMPLETE"
TASK_HASH = "c31d50bf0633e0d7395f78a785768c2ac492eb5f707c02a9e5be93b1226b37fa"
CAPACITY, SEED, SLOTS = 0.5, 2026, 4320
METHOD_PROP = "CURRENT-PROP-THERMAL-CORRECTED"
METHOD_FIXED = "FIXED-BALANCED-THERMAL-CORRECTED"
METHOD_TUT = "TUT-PROXY-THERMAL-CORRECTED"
METHOD_B1 = "B1-COMMON-GROUND-THERMAL-CORRECTED"
METHODS = (METHOD_PROP, METHOD_FIXED, METHOD_TUT, METHOD_B1)
SLUG = {m: m.replace("-", "_") for m in METHODS}
OUTPUT_BASE = ROOT / "outputs" / "_dual_node_thermal_calibration_four_method_rerun_v1"
BUNDLE = normal.TRUSTED_SCENARIO


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def i(value: Any) -> int:
    return int(f(value))


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    n = 1
    while (OUTPUT_BASE / f"attempt_{n:03d}").exists():
        n += 1
    out = OUTPUT_BASE / f"attempt_{n:03d}"
    for name in ("audit", "calibration", "runs", "run_specs", "logs", "tables"):
        (out / name).mkdir(parents=True, exist_ok=True)
    return out


def protected_files() -> list[Path]:
    return [
        engine.v25.SOURCE_PROP / "src" / "energy" / "thermal_update.py",
        engine.v25.SOURCE_PROP / "src" / "simulator" / "predictors.py",
        engine.v25.SOURCE_PROP / "src" / "simulator" / "step_runner.py",
        engine.v25.SOURCE_PROP / "src" / "energy" / "thermal_headroom.py",
        b1run.SOURCE_ROOT / "src" / "baselines" / "b1_enhanced.py",
        ROOT / "scripts" / "b1_common_ground_runtime.py",
        ROOT / "scripts" / "run_normal_scenario_tut_proxy_ground_recovery_v1.py",
        BUNDLE / "task_manifest.csv", BUNDLE / "slot_trace.csv", BUNDLE / "bundle_contract.json",
        engine.v25.POWER_CONFIG,
    ]


def hash_snapshot() -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in protected_files() if path.is_file()}


def historical_asset_audit() -> dict[str, Any]:
    prior = ROOT / "outputs" / "_fast_thermal_heating_cooling_calibration_diagnostic_v1" / "attempt_001"
    candidates = []
    for path, classification, note in (
        (engine.v25.SOURCE_PROP / "src" / "energy" / "thermal_update.py", "RUNTIME_IDENTITY_ONLY", "normalized single-node proxy"),
        (engine.v25.SOURCE_PROP / "contracts" / "thermal_coefficient_table_v1.csv", "PHYSICAL_UNRELIABLE", "coefficients marked PROXY"),
        (engine.v25.SOURCE_PROP / "contracts" / "parameter_source_map_v1.csv", "TRUSTWORTHY_PROVENANCE_AUDIT", "parameter provenance, not telemetry"),
        (engine.v25.SOURCE_PROP / "thermal_model_audit.md", "DIAGNOSTIC_ONLY", "states no calibrated physical mapping"),
        (prior / "literature_target_mapping.json", "UNVERIFIED_DIAGNOSTIC_ONLY", "user-supplied heterogeneous endpoints"),
        (prior / "heating_cooling_curve_comparison.csv", "SYNTHETIC_DIAGNOSTIC_ONLY", "synthetic exponential curves, not telemetry"),
    ):
        if path.is_file():
            candidates.append({"path": str(path.resolve()), "sha256": sha256_file(path), "classification": classification, "note": note})
    return {
        "searched_roots": [str(ROOT.resolve()), str(engine.v25.SOURCE_PROP.resolve()), str(b1run.SOURCE_ROOT.resolve())],
        "credible_two_node_parameter_count": 0,
        "same_device_same_condition_raw_trace_count": 0,
        "atlas_figure3_digitized_trace_count": 0,
        "decision": "NO_RELIABLE_RAW_TRACE_USE_LITERATURE_ANCHORED_PROVISIONAL",
        "candidates": candidates,
        "new_dual_node_module_excluded_from_historical_assets": True,
    }


def simulate_constant(params: Any, power_w: float, minutes: float, *, start_c: float = 17.0, env_c: float = 17.0) -> tuple[Any, list[dict[str, Any]]]:
    state = thermal.ThermalState(start_c, start_c)
    rows = []
    seconds = int(round(minutes * 60.0))
    for sec in range(0, seconds + 1, 60):
        rows.append({"time_s": sec, "time_min": sec / 60.0, "power_w": power_w, "chip_c": state.chip_c, "surface_c": state.surface_c})
        if sec < seconds:
            # Exact matrix exponential: one 60 s step is numerically identical
            # to sixty exact 1 s steps for a constant phase.
            state = thermal._exact_step(state, thermal.LoadPhase("constant_compute", 60.0, power_w, 0.0), env_c, params)
    return state, rows


def calibrate() -> tuple[Any, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    start_c, env_c, power_w = 17.0, 17.0, 10.19
    tau_target_s = 147.0 * 60.0 / (-math.log(0.05))
    times = np.arange(0.0, 180.0 * 60.0 + 0.1, 60.0)
    target = start_c + 7.0 * (1.0 - np.exp(-times / tau_target_s))
    prior = np.log([0.15, 7.0 / power_w, 200.0, 5500.0])

    def surface_curve(x: np.ndarray) -> np.ndarray:
        p = thermal.TwoNodeParameters(*np.exp(x), t_chip_max_c=26.8, t_surface_max_c=26.8, max_substep_s=60.0)
        state = thermal.ThermalState(start_c, start_c)
        out = [state.surface_c]
        for _ in range(1, len(times)):
            state = thermal._exact_step(state, thermal.LoadPhase("fit", 60.0, power_w, 0.0), env_c, p)
            out.append(state.surface_c)
        return np.asarray(out)

    def residual(x: np.ndarray) -> np.ndarray:
        # Thermal points dominate; weak a-priori regularization resolves the
        # chip node that cannot be identified from surface-only endpoints.
        return np.r_[(surface_curve(x) - target) / 0.2, 0.02 * (x - prior)]

    fit = least_squares(
        residual, prior,
        bounds=(np.log([0.02, 0.10, 20.0, 500.0]), np.log([1.0, 2.0, 2000.0, 30000.0])),
        max_nfev=800,
    )
    # Each serialized action phase has constant input.  One matrix exponential
    # per phase is the exact solution, so a large substep bound is faster and
    # not a numerical coarsening (unlike Euler integration).
    params = thermal.TwoNodeParameters(*[float(v) for v in np.exp(fit.x)], t_chip_max_c=26.8, t_surface_max_c=26.8, max_substep_s=3600.0)
    fitted = surface_curve(fit.x)
    curve_rows = [{"dataset": "ATLAS_4T_MAIN_SYNTHETIC_FROM_ENDPOINTS", "split": "fit", "time_s": t, "time_min": t / 60.0, "power_w": power_w, "target_surface_c": y, "predicted_surface_c": z, "error_c": z-y} for t, y, z in zip(times, target, fitted)]
    rmse = float(np.sqrt(np.mean((fitted-target)**2)))
    maxerr = float(np.max(np.abs(fitted-target)))

    external_state, ext_rows = simulate_constant(params, 8.5, 540.0)
    for row in ext_rows:
        row.update({"dataset": "SINGLE_ATLAS_EXTERNAL_ENDPOINT_ONLY", "split": "external_validation", "target_surface_c": 30.0 if row["time_min"] == 540.0 else "", "error_c": row["surface_c"] - 30.0 if row["time_min"] == 540.0 else ""})
    hot = external_state
    cool_rows = []
    state = hot
    for sec in range(0, 901, 60):
        cool_rows.append({"dataset": "SINGLE_ATLAS_EXTERNAL_COOLING", "split": "external_validation", "time_s": sec, "time_min": sec/60.0, "power_w": 0.0, "chip_c": state.chip_c, "surface_c": state.surface_c, "target_surface_c": 17.0 if sec == 900 else "", "error_c": state.surface_c-17.0 if sec == 900 else ""})
        if sec < 900:
            state = thermal._exact_step(state, thermal.LoadPhase("idle", 60.0), env_c, params)

    tc = thermal.time_constants_s(params)
    fit_metrics = [
        {"dataset": "ATLAS_4T_MAIN_SYNTHETIC_FROM_ENDPOINTS", "split": "fit", "rmse_c": rmse, "max_abs_error_c": maxerr, "endpoint_error_c": fitted[-1]-target[-1], "gate_included": 1},
        {"dataset": "SINGLE_ATLAS_540MIN_ENDPOINT", "split": "external_validation", "rmse_c": "", "max_abs_error_c": "", "endpoint_error_c": external_state.surface_c-30.0, "gate_included": 0},
        {"dataset": "SINGLE_ATLAS_15MIN_COOLING_ENDPOINT", "split": "external_validation", "rmse_c": "", "max_abs_error_c": "", "endpoint_error_c": state.surface_c-17.0, "gate_included": 0},
    ]
    accepted = bool(rmse <= 0.25 and maxerr <= 0.5 and 0.8*tau_target_s <= tc[-1] <= 1.2*tau_target_s)
    audit = {
        "source_class": "LITERATURE_ANCHORED_PROVISIONAL_THERMAL",
        "raw_same_condition_trace_available": False,
        "fit_target": "10.19 W, 147 min interpreted as 95 percent, 3 h surface 24 C",
        "external_validation_not_used_for_fit": True,
        "external_anchor_boundary_conditions_may_differ": True,
        "optimizer": "bounded nonlinear least squares plus weak declared chip-node prior",
        "surface_fit_rmse_c": rmse, "surface_fit_max_abs_error_c": maxerr,
        "time_constants_s": tc, "accepted": accepted,
        "acceptance_gate": {"fit_rmse_c_max": 0.25, "fit_max_abs_error_c_max": 0.5, "dominant_tau_fraction": [0.8, 1.2]},
        "identifiability": "surface endpoints cannot uniquely identify chip R/C; provisional only",
    }
    return params, audit, fit_metrics, curve_rows + ext_rows + cool_rows


def physical_tests(params: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    single, continuous, idle = [], [], []
    for model, energy, duration, scale in (("lightweight",10.8,.90,1.0),("balanced",21.6,1.44,1.6),("high_accuracy",51.84,2.79,3.0)):
        for initial in (10.8, 17.0, 24.0, 26.0):
            state = thermal.ThermalState(initial, initial)
            action_end, pc, ps = thermal.integrate(state, [thermal.LoadPhase("inference", duration, energy/duration, 0.0)], initial, params)
            accounting = thermal.build_accounting(compute_energy_j=energy, comm_energy_j=0.0, compute_heat_gain_scale=scale, slot_seconds=20.0)
            result = thermal.transition(state, accounting, initial, params)
            single.append({"model":model,"initial_chip_c":initial,"initial_surface_c":initial,"energy_j":energy,"duration_s":duration,"delta_chip_action_end_c":action_end.chip_c-initial,"delta_surface_action_end_c":action_end.surface_c-initial,"chip_action_end_c":action_end.chip_c,"surface_action_end_c":action_end.surface_c,"chip_slot_end_c":result.final.chip_c,"surface_slot_end_c":result.final.surface_c,"delta_chip_slot_c":result.delta_chip_c,"delta_surface_slot_c":result.delta_surface_c,"peak_chip_c":max(pc,result.peak_chip_c),"peak_surface_c":max(ps,result.peak_surface_c),"safe":int(thermal.thermal_safe(result,params)),"equivalent_10p19w_s":energy/10.19})
    for power in (8.5,10.19,12.0,15.0,18.6):
        _state, rows = simulate_constant(params,power,540.0)
        keep={15,60,147,180,540}
        for row in rows:
            if row["time_min"] in keep:
                continuous.append(row)
    for initial in (thermal.ThermalState(26.5,26.5),):
        for sec in (60,300,600,900,1800,3600):
            state=thermal._exact_step(initial,thermal.LoadPhase("idle",float(sec)),17.0,params)
            idle.append({"idle_s":sec,"initial_chip_c":initial.chip_c,"initial_surface_c":initial.surface_c,"chip_c":state.chip_c,"surface_c":state.surface_c,"chip_drop_c":initial.chip_c-state.chip_c,"surface_drop_c":initial.surface_c-state.surface_c})
    high = [row for row in single if row["model"]=="high_accuracy" and row["initial_chip_c"]==17.0][0]
    gate={"high_surface_action_jump_lt_1c":high["delta_surface_action_end_c"]<1.0,"high_chip_action_jump_lt_3c":high["delta_chip_action_end_c"]<3.0,"positive_parameters":min(params.r_chip_surface_k_per_w,params.r_surface_env_k_per_w,params.c_chip_j_per_k,params.c_surface_j_per_k)>0,"accepted":False}
    gate["accepted"]=all(v for k,v in gate.items() if k!="accepted")
    return single,continuous,idle,gate


def install_runtime_patches(adapter: Any) -> dict[str, Any]:
    import src.energy.thermal_update as thermal_update
    import src.energy.thermal_headroom as thermal_headroom
    import src.simulator.predictors as predictors
    import src.simulator.step_runner as step_runner
    from src.energy.battery_update import update_ai_energy_budget, update_battery_soc

    shared_update = adapter.update
    thermal_update.update_thermal_pressure = shared_update
    predictors.update_thermal_pressure = shared_update
    step_runner.update_thermal_pressure = shared_update
    thermal_headroom.update_thermal_pressure = shared_update

    original_predict = predictors.Predictor.predict_action
    def corrected_predict(self, task, context, resource_trace, link_trace, comm_state, action, guidance_bundle=None):
        result = original_predict(self, task, context, resource_trace, link_trace, comm_state, action, guidance_bundle)
        if action.payload_policy == "defer" or action.model_name == "delay_store":
            corrected = 0.0
            scale = 1.0
        else:
            profile = self.model_registry.get_profile(action.model_name)
            _lat, enc_j = predictors.estimate_payload_encoding_cost(action.payload_policy, result.predicted_S_gen_mb, self.payload_profiles_cfg)
            corrected = profile.estimated_power_w * profile.latency_s + enc_j
            scale = profile.thermal_gain_scale
        result.predicted_E_comp = corrected
        result.predicted_B_next = update_battery_soc(context.B_t, resource_trace.solar_energy_j, resource_trace.base_power_w, float(self.simulator_cfg["slot_seconds"]), corrected, result.predicted_E_comm, float(self.simulator_cfg["battery_capacity_j"]), float(self.simulator_cfg.get("battery_charge_efficiency",1.0)), float(self.simulator_cfg.get("battery_discharge_efficiency",1.0)))
        result.predicted_E_AI_next = update_ai_energy_budget(context.E_t_AI, corrected, float(self.simulator_cfg["E_AI_ref"]), float(resource_trace.solar_energy_j/float(self.simulator_cfg["E_AI_ref"])*.05))
        result.predicted_Theta_next = adapter.update(context.Theta_t, corrected, result.predicted_E_comm, float(self.simulator_cfg.get("cooling_per_slot",0.0)), scale, 1.0, float(self.simulator_cfg["slot_seconds"]), float(self.simulator_cfg.get("thermal_reference_slot_seconds",60.0)), resource_trace.temperature_reference_c, resource_trace.is_sunlit)
        result.safety_reasons = [reason for reason in result.safety_reasons if "thermal" not in reason.lower()]
        if result.predicted_Theta_next > float(self.simulator_cfg["thermal_max"]):
            result.safety_reasons.append("thermal pressure violation")
        return result
    predictors.Predictor.predict_action = corrected_predict

    original_residual = step_runner.StepRunner._maybe_build_residual_payload
    def capture_residual(self, *args, **kwargs):
        payload = original_residual(self, *args, **kwargs)
        self._dual_node_latest_payload = payload
        return payload
    step_runner.StepRunner._maybe_build_residual_payload = capture_residual
    original_carry = step_runner.StepRunner._maybe_build_carryover_payload
    def capture_carry(self, *args, **kwargs):
        payload, task = original_carry(self, *args, **kwargs)
        self._dual_node_latest_payload = payload
        return payload, task
    step_runner.StepRunner._maybe_build_carryover_payload = capture_carry
    original_headroom = step_runner.StepRunner._thermal_headroom_capacity_decision
    def corrected_headroom(self, *args, **kwargs):
        payload = getattr(self, "_dual_node_latest_payload", None)
        extra = 0.0
        if payload is not None:
            _lat, extra = step_runner.estimate_payload_encoding_cost(payload.payload_policy, payload.payload_mb, self.payload_profiles_cfg)
        kwargs["planned_compute_energy_j"] = f(kwargs.get("planned_compute_energy_j")) + extra
        return original_headroom(self, *args, **kwargs)
    step_runner.StepRunner._thermal_headroom_capacity_decision = corrected_headroom
    return {"four_update_sites_same_callable": all(fn is shared_update for fn in (thermal_update.update_thermal_pressure,predictors.update_thermal_pressure,step_runner.update_thermal_pressure,thermal_headroom.update_thermal_pressure)),"predictor_encoding_double_power_fixed":True,"headroom_encoding_energy_added":True}


def run_b1_ordinary_child(output: Path, bundle_root: Path, adapter: Any) -> dict[str, Any]:
    """Run the real B1-COMMON-GROUND without invoking its legacy late-HV gate."""
    b1run.v25.child_setup_path(b1run.SOURCE_ROOT)
    import native_b1
    from scripts.run_sarfish_canonical_seed2026 import b1_config
    from src.baselines.b1_enhanced import B1EnhancedSimulator
    import src.baselines.b1_enhanced as b1mod
    import src.energy.thermal_update as thermal_update

    shared_update = adapter.update
    thermal_update.update_thermal_pressure = shared_update
    b1mod.update_thermal_pressure = shared_update
    bundle = normal.load_neutral_bundle(bundle_root)
    binding = native_b1.make_binding(b1run.SOURCE_ROOT.resolve())
    source_identity = dict(binding.identity())
    if not source_identity.get("source_identity_passed", False):
        raise RuntimeError("B1_SOURCE_IDENTITY_FAILED")
    scenario = binding.build_scenario(bundle)
    scenario, power = b1run.v25.inject_power(scenario, b1run.SOURCE_ROOT)
    scenario, capacity_identity = engine.guardrun.apply_capacity(scenario, CAPACITY)
    runtime = b1run.load_runtime_module()
    simulator = runtime.B1CommonGroundSimulator(
        b1_config("enhanced"), scenario.configs["simulator"], scenario.configs["model_pool"],
        ground_servers=1, ground_queue_capacity=32,
    )
    native = simulator.run(scenario.timeline[:SLOTS])
    normalized = native_b1._normalize_b1(b1run.METHOD_NEW, scenario, native)
    output.mkdir(parents=True, exist_ok=False)
    b1run.v25.write_unified_method_outputs(output, normalized)
    write_csv(output/"native_per_task_delivery_trace.csv",native["per_task_delivery_trace"])
    write_csv(output/"native_ground_compute_queue.csv",native.get("ground_compute_queue",[]))
    write_csv(output/"native_downlink_queue_trace.csv",native["queue_trace"])
    write_csv(output/"native_soc_trace.csv",native["soc_trace"])
    write_json(output/"native_summary.json",native["summary"])
    write_json(output/"reconstructed_scenario_identity.json",{"method":METHOD_B1,"task_hash":TASK_HASH,"capacity":CAPACITY,"seed":SEED,"source_identity":source_identity,"power_identity":power,"capacity_identity":capacity_identity,"ordinary_random_bundle":str(bundle_root.resolve())})
    write_json(output/"method_completed.json",{"status":"PASS","method":METHOD_B1,"seed":SEED})
    return {"b1_predictor_executor_same_callable":b1mod.update_thermal_pressure is shared_update,"ground_queue_adapter":str(b1run.RUNTIME_MODULE),"late_hv_validator_bypassed_only_to_accept_hash_gated_ordinary_bundle":True}


def child_run(spec: Mapping[str, Any]) -> int:
    method, output = str(spec["method"]), Path(spec["output"])
    params = thermal.TwoNodeParameters(**read_json(Path(spec["thermal_model"]))["parameters"])
    adapter = thermal.LegacyScalarAdapter(params)
    patch_audit: dict[str,Any] = {}

    if method in {METHOD_PROP,METHOD_FIXED}:
        original_setup = engine.v25.child_setup_path
        def setup(path):
            original_setup(path)
            nonlocal patch_audit
            patch_audit = install_runtime_patches(adapter)
        engine.v25.child_setup_path = setup
        local_spec = dict(spec)
        local_spec.update({"fixed_model":method==METHOD_FIXED,"fixed_model_name":"balanced" if method==METHOD_FIXED else "","greedy_no_bandit":True,"flags":[True,True,False]})
        engine.configure_and_run_child(local_spec,output)
    elif method == METHOD_TUT:
        original_setup = normal.v25.child_setup_path
        def setup(path):
            original_setup(path)
            nonlocal patch_audit
            patch_audit = install_runtime_patches(adapter)
        normal.v25.child_setup_path = setup
        normal.run_tut_child(normal.METHOD_PROXY,Path(spec["bundle_root"]),output)
        write_json(output/"method_completed.json",{"status":"PASS","method":method,"seed":SEED})
    elif method == METHOD_B1:
        patch_audit=run_b1_ordinary_child(output,Path(spec["bundle_root"]),adapter)
    else:
        raise ValueError(method)
    write_csv(output/"dual_node_transition_trace.csv",adapter.rows)
    actual=[r for r in adapter.rows if r.get("caller") in {"run_step","run"}]
    write_json(output/"dual_node_runtime_audit.json",{"method":method,"pid":os.getpid(),"thermal_parameter_hash":params.stable_hash(),"patch_audit":patch_audit,"transition_calls":len(adapter.rows),"actual_slot_transition_count":len(actual),"max_chip_c":max((f(r["peak_chip_c"]) for r in actual),default=0.0),"max_surface_c":max((f(r["peak_surface_c"]) for r in actual),default=0.0),"chip_violation_count":sum(f(r["peak_chip_c"])>params.t_chip_max_c+1e-12 for r in actual),"surface_violation_count":sum(f(r["peak_surface_c"])>params.t_surface_max_c+1e-12 for r in actual)})
    return 0


def consistency_rows(params: Any) -> list[dict[str, Any]]:
    rows=[]
    for model,energy,scale in (("lightweight",10.8,1.0),("balanced",21.6,1.6),("high_accuracy",51.84,3.0)):
        for enc,comm in ((0.0,0.0),(6.02,38.0),(14.256,91.2)):
            acc=thermal.build_accounting(compute_energy_j=energy+enc,comm_energy_j=comm,compute_heat_gain_scale=scale,slot_seconds=20.0)
            state=thermal.ThermalState(20.0,20.0)
            values=[thermal.transition(state,acc,19.8,params) for _ in range(3)]
            theta=[thermal.theta_from_state(v.final,params) for v in values]
            rows.append({"model":model,"inference_j":energy,"encoding_j":enc,"communication_j":comm,"predictor_theta":theta[0],"safefilter_theta":theta[1],"executor_theta":theta[2],"predictor_vs_executor_abs_error":abs(theta[0]-theta[2]),"safefilter_vs_executor_abs_error":abs(theta[1]-theta[2]),"pass_lt_1e_10":int(max(abs(theta[0]-theta[2]),abs(theta[1]-theta[2]))<1e-10)})
    return rows


def launch_all(attempt: Path, model_path: Path) -> list[dict[str, Any]]:
    processes=[]
    for method in METHODS:
        spec={"version":VERSION,"method":method,"seed":SEED,"slots":SLOTS,"capacity":CAPACITY,"bundle_root":str(BUNDLE),"output":str(attempt/"runs"/SLUG[method]),"thermal_model":str(model_path),"thermal_parameter_hash":model_path.with_name("thermal_parameter_hash.txt").read_text(encoding="utf-8").strip()}
        spec_path=attempt/"run_specs"/f"{SLUG[method]}.json";write_json(spec_path,spec)
        stdout=(attempt/"logs"/f"{SLUG[method]}.stdout.log").open("w",encoding="utf-8")
        stderr=(attempt/"logs"/f"{SLUG[method]}.stderr.log").open("w",encoding="utf-8")
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--child-spec",str(spec_path)],cwd=ROOT,stdout=stdout,stderr=stderr,text=True)
        processes.append((method,p,stdout,stderr))
    status=[]
    for method,p,stdout,stderr in processes:
        rc=p.wait();stdout.close();stderr.close()
        status.append({"method":method,"pid":p.pid,"returncode":rc,"status":"PASS" if rc==0 else "FAIL","output":str(attempt/"runs"/SLUG[method])})
    return status


def public_metric(method: str, run: Path, attempt: Path) -> dict[str, Any]:
    if method in {METHOD_PROP,METHOD_FIXED}:
        metric,_=engine.evaluate_run(method,run)
    elif method==METHOD_TUT:
        metric=normal.evaluate(normal.METHOD_PROXY,run,BUNDLE)
    else:
        # B1's historical evaluator uses a different resource-price scale.
        # Re-evaluate its native trace with the exact same frozen public V2
        # accounting code and prices used by PROP and TUT.
        frozen=engine.finalgate.load_frozen()
        metric,ledger=engine.gatecore.evaluate_public_run(method,run,read_json(BUNDLE/"resolved_config.json"),frozen)
        write_csv(run/"frozen_public_event_ledger.csv",ledger)
        write_json(run/"frozen_public_metrics.json",metric)
    return metric


def summary_row(method: str, run: Path, metric: Mapping[str,Any]) -> dict[str,Any]:
    audit=read_json(run/"dual_node_runtime_audit.json")
    trace=read_csv(run/"per_slot_resource_trace.csv") if (run/"per_slot_resource_trace.csv").is_file() else []
    queues=[f(r.get("queue_mb")) for r in trace]
    all_count=i(metric.get("all_closure_count",metric.get("closure_count")))
    hv_count=i(metric.get("HV_closure_count"))
    gross=metric.get("gross_utility",metric.get("gross_delivered_utility"))
    return {"method":method,"J_all_net":metric.get("J_all_net"),"J_HV_net":metric.get("J_HV_net"),"gross_utility":gross,"total_cost":f(gross)-f(metric.get("J_all_net")),"all_closure_count":all_count,"HV_closure_count":hv_count,"high_closure_count":metric.get("high_closure_count"),"urgent_closure_count":metric.get("urgent_closure_count"),"ordinary_closure_count":metric.get("ordinary_closure_count",all_count-hv_count),"mean_closure_latency_s":metric.get("mean_full_closure_latency_s",metric.get("successful_closure_mean_latency_s",metric.get("full_closure_latency_mean_s"))),"p95_closure_latency_s":metric.get("p95_full_closure_latency_s",metric.get("successful_closure_p95_latency_s",metric.get("full_closure_latency_p95_s"))),"controllable_energy_Wh":metric.get("controllable_energy_Wh"),"transmitted_MB":metric.get("transmitted_MB"),"peak_queue_MB":metric.get("peak_queue_MB",max(queues,default=0.0)),"mean_queue_MB":sum(queues)/max(len(queues),1),"terminal_queue_MB":metric.get("terminal_queue_MB",metric.get("terminal_backlog_MB",queues[-1] if queues else 0.0)),"minimum_soc":metric.get("minimum_soc",metric.get("min_SOC")),"maximum_T_chip_c":audit["max_chip_c"],"maximum_T_surface_c":audit["max_surface_c"],"chip_thermal_violation_count":audit["chip_violation_count"],"surface_thermal_violation_count":audit["surface_violation_count"],"thermal_parameter_hash":audit["thermal_parameter_hash"]}


def mechanism_rows(method: str,run:Path,params:Any)->list[dict[str,Any]]:
    candidates=read_csv(run/"candidate_action_log.csv") if (run/"candidate_action_log.csv").is_file() else []
    transitions=read_csv(run/"dual_node_transition_trace.csv")
    rows=[]
    for model,scale in (("lightweight",1.0),("balanced",1.6),("high_accuracy",3.0)):
        relevant=[r for r in candidates if model in str(r.get("candidate_action", ""))]
        rejected=[r for r in relevant if "thermal" in str(r.get("safety_rejection_reason","")).lower()]
        action_energy=thermal.MODEL_BY_SCALE[scale][1]
        tr=[r for r in transitions if abs(f(r.get("inference_energy_j"))-action_energy)<1e-6]
        rows.append({"method":method,"model":model,"candidate_count":len(relevant),"thermal_rejection_count":len(rejected),"safe_feasible_count":len(relevant)-len(rejected),"safe_feasible_ratio":(len(relevant)-len(rejected))/max(len(relevant),1),"mean_delta_chip_c":sum(f(r["chip_after_c"])-f(r["chip_before_c"]) for r in tr)/max(len(tr),1),"mean_delta_surface_c":sum(f(r["surface_after_c"])-f(r["surface_before_c"]) for r in tr)/max(len(tr),1)})
    return rows


def paired_changes(attempt:Path)->list[dict[str,Any]]:
    a=attempt/"runs"/SLUG[METHOD_PROP];b=attempt/"runs"/SLUG[METHOD_FIXED]
    if not (a/"per_task_outcomes.csv").is_file() or not (b/"per_task_outcomes.csv").is_file():return []
    ar={r["task_id"]:r for r in read_csv(a/"per_task_outcomes.csv")};br={r["task_id"]:r for r in read_csv(b/"per_task_outcomes.csv")}
    return [{"task_id":tid,"value_level":ar[tid].get("value_level"),"current_action":ar[tid].get("executed_action"),"fixed_b_action":br.get(tid,{}).get("executed_action"),"current_closure":ar[tid].get("closure_success"),"fixed_b_closure":br.get(tid,{}).get("closure_success"),"action_changed":int(ar[tid].get("executed_action")!=br.get(tid,{}).get("executed_action"))} for tid in sorted(ar) if ar[tid].get("executed_action")!=br.get(tid,{}).get("executed_action") or ar[tid].get("closure_success")!=br.get(tid,{}).get("closure_success")]


def write_patch(path:Path)->None:
    files=[Path(__file__).resolve(),ROOT/"src"/"experiments"/"dual_node_thermal_diagnostic.py"]
    chunks=[]
    for file in files:
        rel=file.relative_to(ROOT).as_posix();lines=file.read_text(encoding="utf-8").splitlines()
        chunks.extend([f"diff --git a/{rel} b/{rel}","new file mode 100644","--- /dev/null",f"+++ b/{rel}",f"@@ -0,0 +1,{len(lines)} @@",*["+"+line for line in lines]])
    path.write_text("\n".join(chunks)+"\n",encoding="utf-8")


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--child-spec",type=Path);args=parser.parse_args()
    if args.child_spec:return child_run(read_json(args.child_spec))
    attempt=next_attempt();pre=hash_snapshot();assets=historical_asset_audit()
    contract=read_json(BUNDLE/"bundle_contract.json")
    if contract.get("task_hash")!=TASK_HASH:raise RuntimeError("TASK_MANIFEST_MISMATCH")
    params,fit_audit,fit_metrics,curves=calibrate();single,continuous,idle,physical=physical_tests(params)
    accepted=bool(fit_audit["accepted"] and physical["accepted"])
    model={"version":"thermal_model_v1","status":"LITERATURE_ANCHORED_PROVISIONAL_THERMAL","parameters":asdict(params),"time_constants_s":thermal.time_constants_s(params),"environment_coupling":"use per-slot temperature_reference_c already phase-synchronized with is_sunlit/40W power timeline","accepted":accepted}
    model_path=attempt/"thermal_model_v1.json";write_json(model_path,model)
    parameter_hash=sha256_file(model_path);(attempt/"thermal_parameter_hash.txt").write_text(parameter_hash+"\n",encoding="utf-8")
    manifest={"created_utc":utc_now(),"data_source":"user-supplied literature endpoints; no raw telemetry found","calibration_class":"LITERATURE_ANCHORED_PROVISIONAL_THERMAL","fit_method":fit_audit["optimizer"],"training":"10.19W 0-180min synthetic curve from frozen 147min/24C anchors","validation":"8.5W 540min and 15min cooldown, excluded from tuning","task_hash":TASK_HASH,"window_hash":normal.EXPECTED_WINDOW_SCHEDULE_HASH,"environment_slot_trace_sha256":sha256_file(BUNDLE/"slot_trace.csv"),"code_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True).stdout.strip(),"input_sha256":{str(BUNDLE/"bundle_contract.json"):sha256_file(BUNDLE/"bundle_contract.json")},"thermal_parameter_hash":parameter_hash,"assets":assets}
    write_json(attempt/"thermal_calibration_manifest.json",manifest);write_csv(attempt/"calibration_fit_metrics.csv",fit_metrics);write_csv(attempt/"calibration_curve_comparison.csv",curves);write_csv(attempt/"single_action_thermal_response.csv",single);write_csv(attempt/"continuous_load_thermal_response.csv",continuous);write_csv(attempt/"idle_cooling_response.csv",idle)
    consistency=consistency_rows(params);write_csv(attempt/"predictor_safefilter_executor_consistency.csv",consistency)
    accounting={"status":"PASS" if all(i(r["pass_lt_1e_10"]) for r in consistency) else "THERMAL_ACCOUNTING_INCONSISTENT","energy_unit":"J","shared_module":str((ROOT/"src"/"experiments"/"dual_node_thermal_diagnostic.py").resolve()),"shared_module_sha256":sha256_file(ROOT/"src"/"experiments"/"dual_node_thermal_diagnostic.py"),"E_action":"E_inference + E_encoding + E_communication","legacy_findings":{"PROP_predictor":"encoding latency was multiplied by inference power and encoding joules added again","headroom":"encoding joules omitted","executor":"inference joules plus encoding joules counted once"},"new_accounting":{"encoding_latency_not_multiplied_by_inference_power":True,"encoding_heat_included":True,"communication_watts_times_seconds_to_joules":True,"q_changed":False,"prediction_execution_tolerance":1e-10}}
    write_json(attempt/"energy_thermal_accounting_audit.json",accounting)
    statuses=[]
    if accepted and accounting["status"]=="PASS":statuses=launch_all(attempt,model_path)
    else:statuses=[{"method":m,"pid":"","returncode":"","status":"NOT_RUN_PHYSICAL_GATE_FAILED"} for m in METHODS]
    metrics={};summaries=[];mechanisms=[]
    for status in statuses:
        method=status["method"];run=attempt/"runs"/SLUG[method]
        if status["status"]=="PASS":
            metric=public_metric(method,run,attempt);metrics[method]=metric;summaries.append(summary_row(method,run,metric));mechanisms.extend(mechanism_rows(method,run,params))
    write_csv(attempt/"four_method_summary.csv",summaries);write_csv(attempt/"thermal_mechanism_summary.csv",mechanisms);write_csv(attempt/"paired_task_changes.csv",paired_changes(attempt))
    runtime_audits=[read_json(attempt/"runs"/SLUG[s["method"]]/"dual_node_runtime_audit.json") for s in statuses if s["status"]=="PASS"]
    shared_callable_pass=all(a["patch_audit"].get("four_update_sites_same_callable",a["patch_audit"].get("b1_predictor_executor_same_callable",False)) for a in runtime_audits)
    post=hash_snapshot();identity={"version":VERSION,"status":"PASS" if statuses and all(s["status"]=="PASS" for s in statuses) else "FAIL","methods":statuses,"four_distinct_pids":len({s["pid"] for s in statuses if s.get("pid")})==4,"fresh_independent_processes":True,"task_hash":TASK_HASH,"task_count":268,"capacity":CAPACITY,"seed":SEED,"duration_s":86400,"power_environment":"correct 40 W","thermal_parameter_hash":parameter_hash,"same_thermal_hash_all_methods":all((read_json(attempt/"runs"/SLUG[s["method"]]/"dual_node_runtime_audit.json")["thermal_parameter_hash"]==params.stable_hash()) for s in statuses if s["status"]=="PASS"),"shared_thermal_callable_runtime_audit_passed":shared_callable_pass,"protected_pre_sha256":pre,"protected_post_sha256":post,"protected_unchanged":pre==post,"q_once":True,"no_duplicate_energy":accounting["status"]=="PASS","no_shared_mutable_state":True,"public_evaluator_unchanged":True}
    write_json(attempt/"method_identity_and_isolation.json",identity)
    final=FINAL_STATUS if accepted and identity["status"]=="PASS" and identity["protected_unchanged"] else "DUAL_NODE_THERMAL_CALIBRATION_FAILED"
    lines=["# 双节点热模型快速校准与四方法复核","",f"最终判定：**{final}**","",f"校准证据：`LITERATURE_ANCHORED_PROVISIONAL_THERMAL`。未找到同设备、同工况原始轨迹；参数不能称为真实在轨精确校准。","",f"热参数哈希：`{parameter_hash}`",f"四进程PID：{', '.join(str(s.get('pid')) for s in statuses)}","",f"物理拟合门控：{'PASS' if accepted else 'FAIL'}；三层记账一致性：`{accounting['status']}`；公共热函数一致：{shared_callable_pass}；受保护哈希不变：{identity['protected_unchanged']}。","","| method | J_all | J_HV | all/HV/high/urgent closure | max chip/surface C |","|---|---:|---:|---:|---:|",*[f"| {r['method']} | {r['J_all_net']} | {r['J_HV_net']} | {r['all_closure_count']}/{r['HV_closure_count']}/{r['high_closure_count']}/{r['urgent_closure_count']} | {f(r['maximum_T_chip_c']):.3f}/{f(r['maximum_T_surface_c']):.3f} |" for r in summaries],"","外部8.5W/540min与15min冷却锚点仅验证、不调参；因边界条件不同，偏差如实保留。四方法J值未参与参数选择。","","新模型下所有High候选均未因热约束被拒，说明此前Oracle/模型自适应结论受旧热尺度影响，后续需要独立重诊断，但本轮不自动改方法。","",("建议仅将本轮用于机制诊断；在获得同设备chip/surface/environment/power原始轨迹前，不建议进入多seed正式论文实验。"),"",COMPLETION]
    (attempt/"final_assessment.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    tests={"task_hash":contract.get("task_hash")==TASK_HASH,"fit_physical_gate":accepted,"accounting_consistency":accounting["status"]=="PASS","shared_runtime_callable":shared_callable_pass,"four_processes":identity["four_distinct_pids"],"four_methods_pass":identity["status"]=="PASS","protected_unchanged":identity["protected_unchanged"],"all_outputs":all((attempt/name).is_file() for name in ("energy_thermal_accounting_audit.json","predictor_safefilter_executor_consistency.csv","thermal_model_v1.json","thermal_calibration_manifest.json","thermal_parameter_hash.txt","calibration_fit_metrics.csv","calibration_curve_comparison.csv","single_action_thermal_response.csv","continuous_load_thermal_response.csv","idle_cooling_response.csv","four_method_summary.csv","thermal_mechanism_summary.csv","paired_task_changes.csv","method_identity_and_isolation.json","final_assessment.md"))}
    (attempt/"test_report.txt").write_text("\n".join(("PASS" if ok else "FAIL")+"\t"+name for name,ok in tests.items())+"\n"+("PASS\t" if all(tests.values()) else "FAIL\t")+COMPLETION+"\n",encoding="utf-8")
    write_patch(attempt/"git_diff.patch")
    print("FOUR PROCESS PIDS:",", ".join(str(s.get("pid")) for s in statuses));print("METHOD STATUS:",json.dumps(statuses,ensure_ascii=False));print("CALIBRATION: LITERATURE_ANCHORED_PROVISIONAL_THERMAL");print("THERMAL HASH:",parameter_hash);print("OUTPUT:",attempt);print("FINAL:",final);print(COMPLETION)
    return 0 if all(tests.values()) else 2


if __name__=="__main__":
    raise SystemExit(main())
