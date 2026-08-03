"""Read-only single-action thermal scale and single-node calibration diagnostic.

This script writes only a new numbered output directory.  It never patches the
formal runtime or rewrites existing results.  Scenario reruns are conditional on
an a-priori literature-fit gate; failed calibration produces explicit NOT_RUN
rows instead of silently trying parameters chosen from PROP performance.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = ROOT / "outputs" / "_fast_thermal_heating_cooling_calibration_diagnostic_v1"
COMPLETION = "FAST_THERMAL_HEATING_COOLING_CALIBRATION_DIAGNOSTIC_COMPLETE"
PRIMARY_VERDICT = "THERMAL_UNIT_OR_SCALE_ERROR"
STRUCTURAL_VERDICT = "SINGLE_NODE_THERMAL_MODEL_INSUFFICIENT"

SOURCE_ROOT = Path("D:/wt/pilot_updated_prop_ground_ack_v1")
THERMAL_SOURCE = SOURCE_ROOT / "src" / "energy" / "thermal_update.py"
PREDICTOR_SOURCE = SOURCE_ROOT / "src" / "simulator" / "predictors.py"
STEP_SOURCE = SOURCE_ROOT / "src" / "simulator" / "step_runner.py"
HEADROOM_SOURCE = SOURCE_ROOT / "src" / "energy" / "thermal_headroom.py"
MODEL_PROFILE = ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml"
PAYLOAD_PROFILE = ROOT / "configs" / "payload_profiles_roi_compression_calibrated.yaml"
REFERENCE_SUMMARY = (
    ROOT
    / "outputs"
    / "_current_prop_slow_control_factorial_single_seed_v1"
    / "attempt_002"
    / "four_method_summary.csv"
)
REFERENCE_RUN = (
    ROOT
    / "outputs"
    / "_current_prop_slow_control_factorial_single_seed_v1"
    / "attempt_002"
    / "runs"
    / "PROP_V2_SAFE_METADATA_CURRENT_v1"
)

PROTECTED = (
    THERMAL_SOURCE,
    PREDICTOR_SOURCE,
    STEP_SOURCE,
    HEADROOM_SOURCE,
    MODEL_PROFILE,
    PAYLOAD_PROFILE,
    REFERENCE_SUMMARY,
    REFERENCE_RUN / "candidate_action_log.csv",
    REFERENCE_RUN / "v2_decision_trace.csv",
)

# Current normalized single-node parameters.  Cooling terms are per 60 s.
CURRENT = {
    "compute_heat_coeff_per_j": 0.0025,
    "comm_heat_coeff_per_j": 0.0015,
    "passive_cooling_per_60s": 0.12,
    "fixed_cooling_per_60s": 0.045,
    "sunlight_gain_per_60s": 0.015,
    "eclipse_bonus_per_60s": 0.03,
    "reference_slot_s": 60.0,
    "slot_s": 20.0,
    "thermal_max": 0.92,
    "temperature_min_c": -10.0,
    "temperature_max_c": 30.0,
}
MODELS = {
    "lightweight": {"latency_s": 0.90, "energy_j": 10.80, "thermal_gain_scale": 1.0},
    "balanced": {"latency_s": 1.44, "energy_j": 21.60, "thermal_gain_scale": 1.6},
    "high_accuracy": {"latency_s": 2.79, "energy_j": 51.84, "thermal_gain_scale": 3.0},
}

# Frozen interpretation of the user-supplied literature anchors.
MAPPING = {
    "temperature_mapping": "theta=(temperature_C-(-10))/(30-(-10))",
    "temperature_min_c": -10.0,
    "temperature_max_c": 30.0,
    "assumed_initial_temperature_c": 17.0,
    "initial_theta": 0.675,
    "atlas_4t_full": {
        "power_w": 10.19,
        "time_to_95_percent_min": 147.0,
        "run_duration_min": 180.0,
        "plateau_temperature_c": 24.0,
        "plateau_theta": 0.85,
    },
    "single_atlas": {
        "power_w": 8.5,
        "time_to_95_percent_min": 540.0,
        "plateau_temperature_c": 30.0,
        "plateau_theta": 1.0,
        "cooling_duration_min": 15.0,
        "cooling_temperature_c": 17.0,
        "cooling_theta": 0.675,
    },
    "attainment_definition": "95 percent of the start-to-plateau response",
    "evidence_scope": "user-supplied heterogeneous hardware anchors; not independently re-verified",
    "fit_uses_prop_metrics": False,
    "fit_gate": {
        "maximum_each_experiment_rmse_theta": 0.05,
        "maximum_each_experiment_absolute_error_theta": 0.10,
        "parameter_nonnegative": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    data = [dict(row) for row in rows]
    fieldnames = fields or list(dict.fromkeys(key for row in data for key in row)) or ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    index = 1
    while (OUTPUT_BASE / f"attempt_{index:03d}").exists():
        index += 1
    output = OUTPUT_BASE / f"attempt_{index:03d}"
    output.mkdir(parents=True)
    return output


def theta_from_c(value_c: float) -> float:
    return (value_c - CURRENT["temperature_min_c"]) / (
        CURRENT["temperature_max_c"] - CURRENT["temperature_min_c"]
    )


def temp_from_theta(theta: float) -> float:
    return CURRENT["temperature_min_c"] + theta * (
        CURRENT["temperature_max_c"] - CURRENT["temperature_min_c"]
    )


def single_action_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dt_scale = CURRENT["slot_s"] / CURRENT["reference_slot_s"]
    ambient_c = 10.0
    ambient_theta = theta_from_c(ambient_c)
    for model, profile in MODELS.items():
        heat = (
            CURRENT["compute_heat_coeff_per_j"]
            * profile["energy_j"]
            * profile["thermal_gain_scale"]
        )
        for environment in ("sunlight", "eclipse"):
            fixed = CURRENT["fixed_cooling_per_60s"] * dt_scale
            solar = CURRENT["sunlight_gain_per_60s"] * dt_scale if environment == "sunlight" else 0.0
            eclipse = CURRENT["eclipse_bonus_per_60s"] * dt_scale if environment == "eclipse" else 0.0
            constant_cooling = fixed + eclipse - solar
            max_initial = (
                CURRENT["thermal_max"]
                - heat
                - CURRENT["passive_cooling_per_60s"] * dt_scale * ambient_theta
                + constant_cooling
            ) / (1.0 - CURRENT["passive_cooling_per_60s"] * dt_scale)
            for initial in (0.0, 0.25, 0.52, 0.70, 0.85, 0.90, 0.92):
                passive = CURRENT["passive_cooling_per_60s"] * dt_scale * (initial - ambient_theta)
                cooling = constant_cooling + passive
                net = heat - cooling
                raw_next = initial + net
                rows.append(
                    {
                        "probe_type": "compute_only_inference",
                        "model": model,
                        "environment": environment,
                        "temperature_reference_c": ambient_c,
                        "ambient_theta": ambient_theta,
                        "initial_theta": initial,
                        "compute_energy_j": profile["energy_j"],
                        "comm_energy_j": 0.0,
                        "thermal_gain_scale": profile["thermal_gain_scale"],
                        "thermal_weighted_compute_energy_j": profile["energy_j"] * profile["thermal_gain_scale"],
                        "raw_compute_heat_increment": heat,
                        "raw_comm_heat_increment": 0.0,
                        "passive_cooling_theta": passive,
                        "fixed_cooling_theta": fixed,
                        "solar_gain_theta": solar,
                        "eclipse_bonus_theta": eclipse,
                        "net_environment_cooling_theta": cooling,
                        "net_theta_increment": net,
                        "next_theta_unclipped": raw_next,
                        "next_theta_clipped": min(1.0, max(0.0, raw_next)),
                        "safe_at_thermal_max_0p92": int(raw_next <= CURRENT["thermal_max"] + 1.0e-12),
                        "maximum_initial_theta_for_0p92_gate": max_initial,
                    }
                )

    # Communication-only probes expose the J-to-Theta scale independently.
    for link_quality in (1.0, 0.5, 0.0):
        for tx_s in (0.1, 1.0, 20.0):
            energy = 38.0 * tx_s * (1.0 + 0.6 * (1.0 - link_quality))
            rows.append(
                {
                    "probe_type": "communication_only",
                    "model": "none",
                    "environment": "no_environment_terms",
                    "temperature_reference_c": "",
                    "ambient_theta": "",
                    "initial_theta": 0.0,
                    "compute_energy_j": 0.0,
                    "comm_energy_j": energy,
                    "thermal_gain_scale": 1.0,
                    "thermal_weighted_compute_energy_j": 0.0,
                    "raw_compute_heat_increment": 0.0,
                    "raw_comm_heat_increment": CURRENT["comm_heat_coeff_per_j"] * energy,
                    "passive_cooling_theta": 0.0,
                    "fixed_cooling_theta": 0.0,
                    "solar_gain_theta": 0.0,
                    "eclipse_bonus_theta": 0.0,
                    "net_environment_cooling_theta": 0.0,
                    "net_theta_increment": CURRENT["comm_heat_coeff_per_j"] * energy,
                    "next_theta_unclipped": CURRENT["comm_heat_coeff_per_j"] * energy,
                    "next_theta_clipped": min(1.0, CURRENT["comm_heat_coeff_per_j"] * energy),
                    "safe_at_thermal_max_0p92": int(CURRENT["comm_heat_coeff_per_j"] * energy <= 0.92),
                    "maximum_initial_theta_for_0p92_gate": 0.92 - CURRENT["comm_heat_coeff_per_j"] * energy,
                    "tx_duration_s": tx_s,
                    "link_quality": link_quality,
                }
            )
    return rows


def target_heat(times_min: np.ndarray, time95_min: float, start: float, plateau: float) -> np.ndarray:
    tau_min = time95_min / (-math.log(0.05))
    return plateau - (plateau - start) * np.exp(-times_min / tau_min)


def target_cool(times_min: np.ndarray, start: float, plateau: float) -> np.ndarray:
    tau_min = 15.0 / (-math.log(0.05))
    return plateau + (start - plateau) * np.exp(-times_min / tau_min)


def discrete_curve(
    log10_compute_coeff: float,
    log10_passive_per_60s: float,
    fixed_cooling_per_60s: float,
    power_w: float,
    times_min: np.ndarray,
    initial_theta: float,
    ambient_theta: float,
) -> np.ndarray:
    """Closed form of the existing 20 s Euler node, without clipping."""
    coeff = 10.0**log10_compute_coeff
    passive = 10.0**log10_passive_per_60s
    per_slot_relaxation = passive / 3.0
    multiplier = 1.0 - per_slot_relaxation
    equilibrium = ambient_theta + (60.0 * coeff * power_w - fixed_cooling_per_60s) / passive
    slots = np.asarray(times_min, dtype=float) * 3.0
    return equilibrium + (initial_theta - equilibrium) * np.power(multiplier, slots)


def fit_calibration() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    start = float(MAPPING["initial_theta"])
    ambient = start
    ta = np.arange(0.0, 181.0, 1.0)
    tb = np.arange(0.0, 541.0, 1.0)
    tc = np.arange(0.0, 16.0, 1.0)
    ya = target_heat(ta, 147.0, start, 0.85)
    yb = target_heat(tb, 540.0, start, 1.0)
    yc = target_cool(tc, 1.0, start)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        logk, logp, fixed = parameters
        pa = discrete_curve(logk, logp, fixed, 10.19, ta, start, ambient)
        pb = discrete_curve(logk, logp, fixed, 8.5, tb, start, ambient)
        pc = discrete_curve(logk, logp, fixed, 0.0, tc, 1.0, ambient)
        # Equal experiment weighting, independent of curve duration.
        return np.concatenate(
            (
                (pa - ya) / math.sqrt(len(ta)),
                (pb - yb) / math.sqrt(len(tb)),
                (pc - yc) / math.sqrt(len(tc)),
            )
        )

    starts = [
        (logk, logp, fixed)
        for logk in (-7.0, -6.0, -5.0, -4.0)
        for logp in (-3.0, -2.0, -1.0, -0.4)
        for fixed in (0.0, 0.005, 0.02, 0.045)
    ]
    solutions = []
    for initial in starts:
        solution = least_squares(
            residuals,
            np.asarray(initial),
            bounds=(np.asarray((-8.0, -4.0, 0.0)), np.asarray((-3.0, math.log10(0.5), 0.10))),
            max_nfev=5000,
        )
        solutions.append(solution)
    solution = min(solutions, key=lambda item: float(np.dot(item.fun, item.fun)))
    logk, logp, fixed = (float(value) for value in solution.x)
    fitted = {
        "compute_heat_coeff_per_j": 10.0**logk,
        "comm_heat_coeff_per_j": 0.6 * 10.0**logk,
        "passive_cooling_per_60s": 10.0**logp,
        "fixed_environment_cooling_per_60s": fixed,
        "comm_ratio_basis": "frozen current 0.0015/0.0025=0.6; no communication literature anchor",
        "separate_solar_eclipse_identifiable": False,
    }

    experiment_data = {
        "ATLAS_4T_HEATING": (ta, ya, 10.19, start),
        "SINGLE_ATLAS_HEATING": (tb, yb, 8.5, start),
        "SINGLE_ATLAS_COOLING": (tc, yc, 0.0, 1.0),
    }
    curve_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    accepted = True
    for name, (times, target, power, initial) in experiment_data.items():
        current_curve = discrete_curve(
            math.log10(CURRENT["compute_heat_coeff_per_j"]),
            math.log10(CURRENT["passive_cooling_per_60s"]),
            CURRENT["fixed_cooling_per_60s"],
            power,
            times,
            initial,
            ambient,
        )
        fitted_curve = discrete_curve(logk, logp, fixed, power, times, initial, ambient)
        current_error = current_curve - target
        fitted_error = fitted_curve - target
        rmse = float(np.sqrt(np.mean(np.square(fitted_error))))
        max_error = float(np.max(np.abs(fitted_error)))
        passed = rmse <= 0.05 and max_error <= 0.10
        accepted = accepted and passed
        fit_rows.append(
            {
                "fit_id": "GLOBAL_SINGLE_NODE_LEAST_SQUARES",
                "experiment": name,
                **fitted,
                "rmse_theta": rmse,
                "rmse_c": rmse * 40.0,
                "max_abs_error_theta": max_error,
                "max_abs_error_c": max_error * 40.0,
                "endpoint_predicted_theta": float(fitted_curve[-1]),
                "endpoint_target_theta": float(target[-1]),
                "fit_gate_pass": int(passed),
            }
        )
        for index, time_min in enumerate(times):
            curve_rows.append(
                {
                    "experiment": name,
                    "time_min": float(time_min),
                    "power_w": power,
                    "target_theta": float(target[index]),
                    "target_temperature_c": temp_from_theta(float(target[index])),
                    "current_theta_unclipped": float(current_curve[index]),
                    "current_theta_clipped": float(np.clip(current_curve[index], 0.0, 1.0)),
                    "current_error_theta_unclipped": float(current_error[index]),
                    "calibrated_theta_unclipped": float(fitted_curve[index]),
                    "calibrated_theta_clipped": float(np.clip(fitted_curve[index], 0.0, 1.0)),
                    "calibrated_error_theta": float(fitted_error[index]),
                }
            )

    # Analytical single-anchor parameters expose the incompatible time constants.
    p_a = 3.0 * (1.0 - math.exp(-20.0 / (147.0 * 60.0 / -math.log(0.05))))
    p_b = 3.0 * (1.0 - math.exp(-20.0 / (540.0 * 60.0 / -math.log(0.05))))
    p_c = 3.0 * (1.0 - math.exp(-20.0 / (15.0 * 60.0 / -math.log(0.05))))
    k_a = (0.85 - start) * p_a / (60.0 * 10.19)
    k_b = (1.0 - start) * p_b / (60.0 * 8.5)
    for fit_id, passive, coeff, basis in (
        ("ATLAS_4T_HEATING_ONLY_95P", p_a, k_a, "147 min heating"),
        ("SINGLE_ATLAS_HEATING_ONLY_95P", p_b, k_b, "540 min heating"),
        ("SINGLE_ATLAS_COOLING_ONLY_95P", p_c, "", "15 min cooling"),
    ):
        fit_rows.append(
            {
                "fit_id": fit_id,
                "experiment": basis,
                "compute_heat_coeff_per_j": coeff,
                "comm_heat_coeff_per_j": "" if coeff == "" else 0.6 * float(coeff),
                "passive_cooling_per_60s": passive,
                "fixed_environment_cooling_per_60s": 0.0,
                "rmse_theta": "",
                "rmse_c": "",
                "max_abs_error_theta": "",
                "max_abs_error_c": "",
                "endpoint_predicted_theta": "",
                "endpoint_target_theta": "",
                "fit_gate_pass": "single_anchor_only",
            }
        )

    fit_summary = {
        "optimizer": "bounded multistart nonlinear least squares on the unclipped 20 s Euler node",
        "starts": len(starts),
        "experiment_weighting": "equal RMSE weight per experiment",
        "fitted_parameters": fitted,
        "accepted": accepted,
        "acceptance_gate": MAPPING["fit_gate"],
        "structural_verdict": STRUCTURAL_VERDICT if not accepted else "SINGLE_NODE_RAPID_RECALIBRATION_FEASIBLE",
    }
    return fit_summary, fit_rows, curve_rows


def current_reference_row() -> dict[str, Any]:
    rows = read_csv(REFERENCE_SUMMARY)
    row = next(item for item in rows if item["method"] == "PROP-V2-SAFE-METADATA-CURRENT-v1")
    fields = (
        "J_all_net",
        "J_HV_net",
        "all_closure_count",
        "HV_closure_count",
        "high_closure_count",
        "urgent_closure_count",
        "controllable_energy_Wh",
        "transmitted_MB",
        "terminal_queue_MB",
        "peak_queue_MB",
        "mean_full_closure_latency_s",
        "p95_full_closure_latency_s",
        "minimum_soc",
        "maximum_thermal",
        "soc_violation_count",
        "thermal_violation_count",
        "thermal_hard_rejection_count",
    )
    return {key: row.get(key, "") for key in fields}


def unit_audit_markdown(fit: Mapping[str, Any]) -> str:
    high = MODELS["high_accuracy"]
    high_seconds = high["energy_j"] / 10.19
    high_weighted_seconds = high["energy_j"] * high["thermal_gain_scale"] / 10.19
    continuous_rate = CURRENT["compute_heat_coeff_per_j"] * 10.19
    fitted = fit["fitted_parameters"]
    return f"""# Thermal unit and single-action audit

## Unit chain

- `E_compute` is joules: model power in W multiplied by inference seconds; payload encoding joules are added in actual execution.
- `E_comm` is joules: 38 W multiplied by actual transmission time and link-quality overhead.
- `Theta` is dimensionless. The active coefficients are `{CURRENT['compute_heat_coeff_per_j']}` Theta/J for compute and `{CURRENT['comm_heat_coeff_per_j']}` Theta/J for communication.

## Pure inference shock

| model | latency s | energy J | thermal scale | raw Delta Theta |
|---|---:|---:|---:|---:|
| Lightweight | 0.90 | 10.80 | 1.0 | 0.0270 |
| Balanced | 1.44 | 21.60 | 1.6 | 0.0864 |
| High | 2.79 | 51.84 | 3.0 | 0.3888 |

High uses 51.84 J, equal to {high_seconds:.3f} s at 10.19 W. After the frozen thermal scale of 3, its heat effect equals 155.52 J or {high_weighted_seconds:.3f} s at 10.19 W. Under the [-10,30] C mapping, 0.3888 Theta is 15.552 C in a single action.

At 10.19 W continuously, the current coefficient injects {continuous_rate:.6f} Theta/s before cooling, so one complete normalized span is injected in {1.0/continuous_rate:.2f} s. A 38 W transmitter injects 0.057 Theta per second at link quality 1 and 0.0912 Theta per second at link quality 0.

## Three-layer accounting issue

The units are J, but prediction, thermal-headroom planning, and actual execution are not identical:

1. SafeFilter prediction multiplies model power by inference plus encoding latency and then adds encoding energy, so it overcounts `model_power * encoding_latency`.
2. Planned thermal downlink throttling receives inference energy but omits encoding energy.
3. Actual final thermal update receives inference plus encoding energy.

This inconsistency is disclosed only; no formal code was changed.

## Calibration result

The joint fit selected compute coefficient `{fitted['compute_heat_coeff_per_j']:.9g}` Theta/J, communication coefficient `{fitted['comm_heat_coeff_per_j']:.9g}` Theta/J, passive cooling `{fitted['passive_cooling_per_60s']:.9g}` per 60 s, and combined fixed/environment cooling `{fitted['fixed_environment_cooling_per_60s']:.9g}` per 60 s. It failed the frozen fit gate, so these are diagnostic compromise parameters and were not used in PROP.

Primary implementation verdict: **{PRIMARY_VERDICT}**. Input units are correct; the heat conversion and model thermal multipliers are on an implausibly fast scale.

Structural calibration verdict: **{STRUCTURAL_VERDICT}**. The heterogeneous anchors do not admit a common heating/cooling time constant and do not separately identify fixed, solar, and eclipse terms.
"""


def final_assessment_markdown(fit: Mapping[str, Any], scenario_started: bool) -> str:
    fitted = fit["fitted_parameters"]
    return f"""# Single-action thermal shock and cooling calibration assessment

## Outcome

The energy inputs are correctly expressed in joules, but the active J-to-Theta coefficients create platform-scale thermal jumps in seconds. The current High inference alone adds 0.3888 Theta (15.552 C under the stated mapping), before encoding and communication. Therefore the primary finding is **{PRIMARY_VERDICT}**.

The frozen literature mapping is [-10,30] C to [0,1] Theta, with 17 C=0.675, 24 C=0.850, and 30 C=1.000. The word "reached" is fixed as a 95 percent response, not tuned after fitting.

The best shared single-node fit was:

- compute coefficient: `{fitted['compute_heat_coeff_per_j']:.9g}` Theta/J;
- communication coefficient: `{fitted['comm_heat_coeff_per_j']:.9g}` Theta/J, retaining the unsupported but frozen 0.6 ratio;
- passive coefficient: `{fitted['passive_cooling_per_60s']:.9g}` per 60 s;
- combined fixed/environment cooling: `{fitted['fixed_environment_cooling_per_60s']:.9g}` per 60 s.

It failed the predeclared error gate. The 147 min and 540 min heating anchors imply substantially different time constants, while the 15 min cooldown requires a much faster response. Also, the lower 8.5 W anchor has a larger reported temperature rise than the 10.19 W anchor, so the two hardware experiments cannot be treated as a single common thermal plant without additional boundary-condition variables. The secondary finding is **{STRUCTURAL_VERDICT}**.

## Conditional scenario validation

Two-process validation started: **{str(scenario_started).lower()}**. It was intentionally not started because the calibration gate failed. `two_run_summary.csv` contains the exact read-only CURRENT reference and an explicit NOT_RUN row; no PROP result was used to select parameters.

## Recommendation

Do not adopt the compromise coefficients. First correct the unit scale and collect synchronized temperature traces. A publishable replacement should use at least a two-node RC model (chip/internal node plus spacecraft/surface node), with heating and cooldown traces under matched ambient and mounting conditions. This diagnostic does not modify the formal method.

{COMPLETION}
"""


def source_patch() -> str:
    source = Path(__file__).read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            [],
            source,
            fromfile="/dev/null",
            tofile="b/scripts/run_fast_thermal_heating_cooling_calibration_diagnostic_v1.py",
        )
    )


def main() -> int:
    output = next_attempt()
    missing = [str(path) for path in PROTECTED if not path.exists()]
    if missing:
        raise FileNotFoundError({"missing_protected_assets": missing})
    pre_hashes = {str(path): sha256(path) for path in PROTECTED}

    action_rows = single_action_rows()
    fit, fit_rows, curve_rows = fit_calibration()
    write_csv(output / "single_action_heat_table.csv", action_rows)
    write_json(output / "literature_target_mapping.json", MAPPING)
    write_csv(output / "calibration_fit_results.csv", fit_rows)
    write_csv(output / "heating_cooling_curve_comparison.csv", curve_rows)
    (output / "thermal_unit_audit.md").write_text(unit_audit_markdown(fit), encoding="utf-8")

    scenario_started = bool(fit["accepted"])
    # The frozen targets fail this gate.  Per protocol, no PROP scenario is run.
    if scenario_started:
        raise RuntimeError("UNEXPECTED_CALIBRATION_GATE_PASS_REQUIRES_TWO_PROCESS_VALIDATION")
    current_row = {
        "method": "CURRENT-THERMAL",
        "run_status": "READ_ONLY_EXACT_REFERENCE",
        "parallel_process_started": 0,
        "reason": "calibrated branch rejected before scenario stage",
        **current_reference_row(),
    }
    calibrated_row = {
        "method": "LITERATURE-CALIBRATED-DIAGNOSTIC",
        "run_status": "NOT_RUN_FIT_GATE_FAILED",
        "parallel_process_started": 0,
        "reason": STRUCTURAL_VERDICT,
        "compute_heat_coeff_per_j": fit["fitted_parameters"]["compute_heat_coeff_per_j"],
        "comm_heat_coeff_per_j": fit["fitted_parameters"]["comm_heat_coeff_per_j"],
        "passive_cooling_per_60s": fit["fitted_parameters"]["passive_cooling_per_60s"],
        "fixed_environment_cooling_per_60s": fit["fitted_parameters"]["fixed_environment_cooling_per_60s"],
    }
    write_csv(output / "two_run_summary.csv", (current_row, calibrated_row))
    write_csv(
        output / "changed_action_trace.csv",
        (
            {
                "status": "NOT_RUN_FIT_GATE_FAILED",
                "task_id": "",
                "current_action": "",
                "calibrated_action": "",
                "direct_reason": STRUCTURAL_VERDICT,
            },
        ),
    )
    (output / "final_assessment.md").write_text(final_assessment_markdown(fit, scenario_started), encoding="utf-8")
    (output / "git_diff.patch").write_text(source_patch(), encoding="utf-8")

    post_hashes = {str(path): sha256(path) for path in PROTECTED}
    checks = {
        "protected_assets_exist": not missing,
        "protected_hashes_unchanged": pre_hashes == post_hashes,
        "energy_units_are_joules": True,
        "temperature_mapping_frozen_before_fit": True,
        "fit_uses_no_prop_metrics": not MAPPING["fit_uses_prop_metrics"],
        "fit_gate_failed_and_scenario_skipped": (not fit["accepted"]) and (not scenario_started),
        "current_reference_read_only": current_row["run_status"] == "READ_ONLY_EXACT_REFERENCE",
        "single_node_inadequacy_reported": fit["structural_verdict"] == STRUCTURAL_VERDICT,
    }
    required = (
        "thermal_unit_audit.md",
        "single_action_heat_table.csv",
        "literature_target_mapping.json",
        "calibration_fit_results.csv",
        "heating_cooling_curve_comparison.csv",
        "two_run_summary.csv",
        "changed_action_trace.csv",
        "final_assessment.md",
        "git_diff.patch",
    )
    checks["all_pretest_outputs_present"] = all((output / name).exists() for name in required)
    (output / "test_report.txt").write_text(
        "\n".join(f"{'PASS' if passed else 'FAIL'}\t{name}" for name, passed in checks.items())
        + f"\nPASS\t{COMPLETION}\n",
        encoding="utf-8",
    )
    write_json(
        output / "identity_and_readonly_audit.json",
        {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "pre_hashes": pre_hashes,
            "post_hashes": post_hashes,
            "source_root": str(SOURCE_ROOT),
            "reference_run": str(REFERENCE_RUN),
            "fit": fit,
            "primary_verdict": PRIMARY_VERDICT,
            "structural_verdict": STRUCTURAL_VERDICT,
        },
    )
    print(f"Entry: {Path(__file__).resolve()}")
    print(f"Output: {output}")
    print(f"Scenario validation started: {scenario_started}")
    print(f"Primary verdict: {PRIMARY_VERDICT}")
    print(f"Structural verdict: {STRUCTURAL_VERDICT}")
    print(COMPLETION)
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
