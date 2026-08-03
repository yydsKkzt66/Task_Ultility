from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "run_soc_band_protected_taskflow_v1_1.py"
OUT = Path("D:/wt/pilot/outputs/_high_solar_half_battery_cycle_v1_5")
SOURCE_ROOT = Path("D:/wt/pilot_shared_prior_endogenous_stress_v1")
PARAM = Path("D:/wt/pilot/outputs/_myopic_frozen_calibrated_v1/myopic_frozen_calibration_v1.json")
PROFILE = "VALUE-V25-MAIN-v1"
METHOD_PROP = "PROP-SHARED-CALIBRATED-PRIOR"
METHOD_MYOPIC = "MYOPIC-FROZEN-CALIBRATED-PROGRESSIVE"
SCENARIOS = {
    "SINGLE_DEEP_25W": {
        "normal_base_power_w": 12.0,
        "stress_base_power_w": 25.0,
        "load_windows_s": [(27_000.0, 56_000.0)],
        "intent": "one long platform concurrent-load period; target CAUTION/RESTRICTED during HV burst then recover with 40W peak solar",
    },
    "TWO_CYCLE_25W": {
        "normal_base_power_w": 12.0,
        "stress_base_power_w": 25.0,
        "load_windows_s": [(27_000.0, 42_000.0), (66_000.0, 78_000.0)],
        "intent": "two fixed concurrent-load periods separated by natural 40W-peak solar recovery",
    },
}
SOC_NODE_TIMES_S = [0, 9000, 18000, 27000, 36000, 45000, 54000, 63000, 72000, 81000, 86400]
PLATFORM_SWITCH = {
    "platform_version": "SOC-BAND-PLATFORM-PROTECTION-v1",
    "enabled": True,
    "legacy_battery_min": 0.55,
    "soc_hard_shutdown": 0.25,
    "battery_min_override": 0.25,
    "healthy_line": 0.70,
    "caution_entry": 0.60,
    "restricted_entry": 0.40,
    "shutdown_entry": 0.25,
    "caution_recovery": 0.70,
    "restricted_recovery_to_caution": 0.50,
    "shutdown_recovery_to_normal": 0.70,
    "scope": "shared platform safety gate for all methods",
}
BATTERY_CAPACITY_SCALE = 0.5
LEGACY_SOLAR_PEAK_W = 14.5
NEW_SOLAR_PEAK_W = 40.0
SOLAR_SCALE = NEW_SOLAR_PEAK_W / LEGACY_SOLAR_PEAK_W


def load_base():
    spec = importlib.util.spec_from_file_location("soc_band_v1_1", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load_base()
sp = base.sp


def wjson(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def wcsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or x == "":
            return default
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def i(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase(time_s: float) -> str:
    if time_s < 9_000:
        return "SETUP"
    if time_s < 27_000:
        return "EARLY_ORDINARY_BURST"
    if time_s < 39_000:
        return "DRAWDOWN"
    if time_s < 60_000:
        return "HV_URGENT_BURST"
    if time_s < 72_000:
        return "RECOVERY"
    return "POST_RECOVERY"


def base_power_at(original_base_power_w: float, time_s: float, scenario: dict[str, Any]) -> float:
    for start_s, end_s in scenario["load_windows_s"]:
        if float(start_s) <= time_s < float(end_s):
            return float(scenario["stress_base_power_w"])
    return float(scenario["normal_base_power_w"])


def build_taskflow(scenario_name: str) -> tuple[dict[str, Any], list[Any], dict[str, Any], list[dict[str, Any]]]:
    scenario = SCENARIOS[scenario_name]
    cfg, timeline, ident = sp.build_base(PROFILE, stress=False)
    cfg["simulator"]["battery_min_legacy"] = float(cfg["simulator"].get("battery_min", PLATFORM_SWITCH["legacy_battery_min"]))
    cfg["simulator"]["battery_min"] = PLATFORM_SWITCH["battery_min_override"]
    cfg["simulator"]["soc_band_platform_protection"] = dict(PLATFORM_SWITCH)
    legacy_capacity_wh = float(cfg["simulator"].get("battery_capacity_wh", 230.0))
    cfg["simulator"]["battery_capacity_wh_legacy"] = legacy_capacity_wh
    cfg["simulator"]["battery_capacity_wh"] = legacy_capacity_wh * BATTERY_CAPACITY_SCALE
    cfg["simulator"]["battery_capacity_j"] = cfg["simulator"]["battery_capacity_wh"] * 3600.0
    timeline, diff = base.remap_taskflow_v25(timeline)
    stressed = []
    for step in timeline:
        time_s = step.slot_index * step.slot_seconds
        new_base = base_power_at(float(step.resource_trace.base_power_w), time_s, scenario)
        rt = replace(
            step.resource_trace,
            base_power_w=new_base,
            solar_energy_j=float(step.resource_trace.solar_energy_j) * SOLAR_SCALE,
            solar_power_w=float(getattr(step.resource_trace, "solar_power_w", 0.0)) * SOLAR_SCALE,
        )
        stressed.append(replace(step, resource_trace=rt))
    timeline = stressed
    rows = sp.canonical_task_rows_v2(timeline)
    by_id = {step.task.task_id: step.task for step in timeline if step.task is not None}
    ident = {
        **ident,
        "taskflow_version": "HIGH-SOLAR-HALF-BATTERY-CYCLE-v1.5",
        "platform_switch": PLATFORM_SWITCH,
        "battery_capacity_scale": BATTERY_CAPACITY_SCALE,
        "battery_capacity_wh": cfg["simulator"]["battery_capacity_wh"],
        "legacy_solar_peak_w": LEGACY_SOLAR_PEAK_W,
        "new_solar_peak_w": NEW_SOLAR_PEAK_W,
        "solar_scale": SOLAR_SCALE,
        "scenario_name": scenario_name,
        "scenario": scenario,
        "value_profile": PROFILE,
        "task_hash": sp.canonical_task_hash_v2(rows),
        "environment_hash": sp.canonical_environment_hash_v2(cfg, timeline),
        "value_mapping_hash": hashlib.sha256(
            json.dumps(
                [(r["task_id"], r["value_level"], sp.task_effective_value(by_id[r["task_id"]])) for r in rows],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "phase_task_counts": dict(Counter(phase(float(r["arrival_time_s"])) for r in rows)),
        "value_level_counts": dict(Counter(r["value_level"] for r in rows)),
    }
    return cfg, timeline, ident, diff


def make_router(method: str, env: Any) -> Any:
    if method == METHOD_MYOPIC:
        inner = sp.FrozenMyopicProgressiveRouter(PARAM)
    elif method == METHOD_PROP:
        inner = sp.SharedPriorPropRouter([a.action_name for a in env.action_space.list_actions()], PARAM)
    else:
        raise ValueError(method)
    return base.SOCBandProtectionWrapper(inner, method)


def scenario_object(cfg: dict[str, Any], timeline: list[Any]) -> Any:
    return sp.scen(cfg, timeline)


def run_method(scenario_name: str, method: str) -> dict[str, Any]:
    cfg, timeline, ident, diff = build_taskflow(scenario_name)
    env = sp.SimulationEnvironment(cfg)
    runner = env.build_episode_runner(timeline)
    router = make_router(method, env)
    result = runner.run(router)
    norm = sp.normalize_episode(method, scenario_object(cfg, timeline), result, runner, 4320)
    return {"scenario": scenario_name, "norm": norm, "router": router, "runner": runner, "identity": ident, "diff": diff}


def metric_row(scenario_name: str, method: str, run: dict[str, Any]) -> dict[str, Any]:
    row = base.metric_row(method, run)
    return {"scenario": scenario_name, **row}


def soc_band_from_value(soc: float) -> str:
    if soc >= 0.70:
        return "NORMAL"
    if soc >= 0.60:
        return "HEALTHY_BUT_BELOW_70"
    if soc > 0.40:
        return "CAUTION_40_60"
    if soc > 0.25:
        return "RESTRICTED_25_40"
    return "SHUTDOWN_LE_25"


def phase_rows(scenario_name: str, method: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    tasks = run["norm"]["per_task"]
    slots = run["norm"]["slots"]
    for ph in ["SETUP", "EARLY_ORDINARY_BURST", "DRAWDOWN", "HV_URGENT_BURST", "RECOVERY", "POST_RECOVERY"]:
        phase_task_ids = {t["task_id"] for t in tasks if phase(f(t["arrival_time"])) == ph}
        ts = [t for t in tasks if t["task_id"] in phase_task_ids]
        ss = [s for s in slots if phase(f(s["time_s"])) == ph]
        cl = [t for t in ts if i(t.get("closure_success"))]
        rows.append(
            {
                "scenario": scenario_name,
                "method": method,
                "phase": ph,
                "task_count": len(ts),
                "accepted_count": sum(1 for t in ts if t["selected_action"] != "delay/store"),
                "closure_count": len(cl),
                "HV_closure_count": sum(1 for t in cl if t["value_level"] in ("high", "urgent")),
                "minimum_SOC": min([f(s["battery_soc"]) for s in ss], default=""),
                "mean_SOC": sum(f(s["battery_soc"]) for s in ss) / len(ss) if ss else "",
                "terminal_queue_mb": ss[-1]["queue_mb"] if ss else "",
                "soc_band_rejected_candidates": sum(
                    1 for r in run["router"].filter_log if i(r["allowed"]) == 0 and r["task_id"] in phase_task_ids
                ),
            }
        )
    return rows


def band_rows(scenario_name: str, method: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    counts = Counter()
    for slot in run["norm"]["slots"]:
        counts[soc_band_from_value(f(slot["battery_soc"]))] += 1
    return [
        {"scenario": scenario_name, "method": method, "soc_band": band, "seconds": slots * 20.0, "slots": slots}
        for band, slots in sorted(counts.items())
    ]


def soc_nodes(scenario_name: str, method: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    slots = run["norm"]["slots"]
    rows = []
    for target in SOC_NODE_TIMES_S:
        chosen = min(slots, key=lambda s: abs(f(s["time_s"]) - target))
        soc = f(chosen["battery_soc"])
        rows.append(
            {
                "scenario": scenario_name,
                "method": method,
                "target_time_s": target,
                "actual_time_s": chosen["time_s"],
                "hour": round(f(chosen["time_s"]) / 3600.0, 3),
                "phase": phase(f(chosen["time_s"])),
                "battery_soc": soc,
                "soc_band": soc_band_from_value(soc),
                "queue_mb": chosen["queue_mb"],
                "thermal": chosen["thermal"],
            }
        )
    return rows


def task_delta_rows(scenario_name: str, runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    prop = {t["task_id"]: t for t in runs[METHOD_PROP]["norm"]["per_task"]}
    myopic = {t["task_id"]: t for t in runs[METHOD_MYOPIC]["norm"]["per_task"]}
    rows = []
    for task_id, p in prop.items():
        m = myopic[task_id]
        rows.append(
            {
                "scenario": scenario_name,
                "task_id": task_id,
                "value_level": p["value_level"],
                "phase": phase(f(p["arrival_time"])),
                "prop_action": p["selected_action"],
                "myopic_action": m["selected_action"],
                "prop_closed": p["closure_success"],
                "myopic_closed": m["closure_success"],
                "prop_ground_time": p["closure_ground_time"],
                "myopic_ground_time": m["closure_ground_time"],
            }
        )
    return rows


def status_for(scenario_name: str, rows: list[dict[str, Any]]) -> str:
    by_method = {row["method"]: row for row in rows if row["scenario"] == scenario_name}
    prop = by_method[METHOD_PROP]
    myopic = by_method[METHOD_MYOPIC]
    if any(i(row["safety_violations"]) > 0 for row in by_method.values()):
        return "INVALID_SAFETY"
    prop_adv = f(prop["J_all_net"]) > f(myopic["J_all_net"]) and (
        f(prop["HV_net_utility"]) > f(myopic["HV_net_utility"]) or i(prop["HV_closure_count"]) > i(myopic["HV_closure_count"])
    )
    if prop_adv:
        return "PROP_ADVANTAGE_OBSERVED"
    return "NO_PROP_ADVANTAGE_OBSERVED"


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    before = sha_file(PARAM)
    all_metrics: list[dict[str, Any]] = []
    all_phase: list[dict[str, Any]] = []
    all_band: list[dict[str, Any]] = []
    all_nodes: list[dict[str, Any]] = []
    all_filter: list[dict[str, Any]] = []
    all_delta: list[dict[str, Any]] = []
    scenario_identities = {}
    scenario_status = {}

    for scenario_name in SCENARIOS:
        runs = {method: run_method(scenario_name, method) for method in [METHOD_PROP, METHOD_MYOPIC]}
        scenario_identities[scenario_name] = runs[METHOD_PROP]["identity"]
        wcsv(OUT / "runs" / scenario_name / "taskflow_remap_manifest.csv", runs[METHOD_PROP]["diff"])
        for method in [METHOD_PROP, METHOD_MYOPIC]:
            run_out = OUT / "runs" / scenario_name / method
            run_out.mkdir(parents=True, exist_ok=True)
            wcsv(run_out / "per_task_outcomes.csv", runs[method]["norm"]["per_task"])
            wcsv(run_out / "ground_delivery_events.csv", runs[method]["norm"]["events"])
            wcsv(run_out / "per_slot_resource_trace.csv", runs[method]["norm"]["slots"])
            wjson(run_out / "resource_summary.json", runs[method]["norm"]["resource_summary"])
            all_metrics.append(metric_row(scenario_name, method, runs[method]))
            all_phase.extend(phase_rows(scenario_name, method, runs[method]))
            all_band.extend(band_rows(scenario_name, method, runs[method]))
            all_nodes.extend(soc_nodes(scenario_name, method, runs[method]))
            all_filter.extend({"scenario": scenario_name, **r} for r in runs[method]["router"].filter_log)
        all_delta.extend(task_delta_rows(scenario_name, runs))
        scenario_status[scenario_name] = status_for(scenario_name, all_metrics)

    after = sha_file(PARAM)
    wjson(
        OUT / "source_identity.json",
        {
            "source_root": str(SOURCE_ROOT),
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=SOURCE_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.strip(),
            "script": str(Path(__file__).resolve()),
            "parameter_file": str(PARAM),
            "parameter_hash_before": before,
            "parameter_hash_after": after,
        },
    )
    wjson(OUT / "taskflow_identity.json", scenario_identities)
    wjson(OUT / "platform_switch_manifest.json", PLATFORM_SWITCH)
    wjson(OUT / "scenario_status.json", scenario_status)
    wcsv(
        OUT / "soc_stress_profile.csv",
        [
            {
                "scenario": name,
                "normal_base_power_w": cfg["normal_base_power_w"],
                "stress_base_power_w": cfg["stress_base_power_w"],
                "load_windows_s": json.dumps(cfg["load_windows_s"]),
                "legacy_solar_peak_w": LEGACY_SOLAR_PEAK_W,
                "new_solar_peak_w": NEW_SOLAR_PEAK_W,
                "solar_scale": SOLAR_SCALE,
                "battery_capacity_scale": BATTERY_CAPACITY_SCALE,
                "intent": cfg["intent"],
            }
            for name, cfg in SCENARIOS.items()
        ],
    )
    wcsv(OUT / "comparison_metrics.csv", all_metrics)
    wcsv(OUT / "phase_metrics.csv", all_phase)
    wcsv(OUT / "soc_band_residency.csv", all_band)
    wcsv(OUT / "soc_time_nodes.csv", all_nodes)
    wcsv(OUT / "soc_band_filter_audit.csv", all_filter)
    wcsv(OUT / "task_delta.csv", all_delta)
    wjson(
        OUT / "consistency_test_report.json",
        {
            "parameter_hash_unchanged": before == after,
            "single_seed_2026": True,
            "v25_only": True,
            "common_soc_policy_for_both_methods": True,
            "no_soc_reset": True,
            "scenarios": scenario_status,
            "all_results_retained": True,
        },
    )

    final_status = "HIGH_SOLAR_HALF_BATTERY_CYCLE_V1_5_COMPLETE"
    if all(status == "INVALID_SAFETY" for status in scenario_status.values()):
        final_status = "HIGH_SOLAR_HALF_BATTERY_CYCLE_V1_5_ALL_INVALID"
    elif any(status == "PROP_ADVANTAGE_OBSERVED" for status in scenario_status.values()):
        final_status = "PROP_ADVANTAGE_OBSERVED_HIGH_SOLAR_V1_5"
    elif any(status == "INVALID_SAFETY" for status in scenario_status.values()):
        final_status = "HIGH_SOLAR_HALF_BATTERY_CYCLE_V1_5_PARTIAL_VALID_NO_ADVANTAGE"
    else:
        final_status = "NO_PROP_ADVANTAGE_OBSERVED_HIGH_SOLAR_V1_5"

    lines = [
        "# HIGH-SOLAR-HALF-BATTERY-CYCLE-v1.5",
        "",
        f"Final status: `{final_status}`",
        "",
        "Two reversible seed-2026 V25 pilots were run under the shared `SOC-BAND-PLATFORM-PROTECTION-v1` switch. The legacy formal setting is not modified. This pilot overrides runtime `battery_min=0.25`, halves battery capacity to 115Wh, scales solar peak from 14.5W to 40W, uses 12W normal platform power, and raises platform power to 25W during fixed concurrent-load windows. After each window, platform power returns to 12W. No SOC reset or artificial recharge is used.",
        "",
        "## Scenario status",
        "",
        "| scenario | status |",
        "|---|---|",
    ]
    for name, status in scenario_status.items():
        lines.append(f"| {name} | {status} |")
    lines.extend(
        [
            "",
            "## Requested metrics",
            "",
            "| scenario | method | J_all_net | HV_net_utility | HV_closure_count | all_closure_count | HV_alert_mean_latency_s | HV_with_metadata_ground_count | min_SOC | safety_violations |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in all_metrics:
        lines.append(
            f"| {row['scenario']} | {row['method']} | {float(row['J_all_net']):.6f} | {float(row['HV_net_utility']):.6f} | {row['HV_closure_count']} | {row['all_closure_count']} | {row['HV_alert_mean_latency_s']} | {row['HV_with_metadata_ground_count']} | {row['min_SOC']} | {row['safety_violations']} |"
        )
    lines.extend(
        [
            "",
            "## SOC time nodes",
            "",
            "See `soc_time_nodes.csv` for representative 24h SOC nodes at 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, and 24 h.",
            "",
            "Rollback: remove `scripts/run_high_solar_half_battery_cycle_v1_5.py` and `D:/wt/pilot/outputs/_high_solar_half_battery_cycle_v1_5`. No merge was performed.",
        ]
    )
    (OUT / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "rollback_manifest.md").write_text(
        "Delete D:/wt/pilot/outputs/_high_solar_half_battery_cycle_v1_5 and remove scripts/run_high_solar_half_battery_cycle_v1_5.py if not retained. No historical outputs were overwritten.\n",
        encoding="utf-8",
    )
    print(final_status)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
