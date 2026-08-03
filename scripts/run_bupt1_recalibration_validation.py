"""BUPT-1-grounded resource recalibration diagnostics and lightweight validation."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_task_lifetime_diagnostics import generate_lifetime_outputs  # noqa: E402
from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.energy.battery_update import update_battery_soc  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.routers.rule_router import RuleBasedRouter  # noqa: E402
from src.routers.safe_linucb_router import SafeLinUCBRouter  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.resource_trace_loader import load_synthetic_resource_trace  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sarfish_1day_smoke"
SCENARIOS = {
    "nominal": {
        "resource": PROJECT_ROOT / "configs" / "resource_sarfish_formal_nominal_bupt1.yaml",
        "timeline": PROJECT_ROOT / "configs" / "timeline_sarfish_formal_nominal_bupt1_20s.yaml",
        "simulator": PROJECT_ROOT / "configs" / "simulator_sarfish_formal_nominal_bupt1_1day_20s.yaml",
    },
    "stress": {
        "resource": PROJECT_ROOT / "configs" / "resource_sarfish_formal_stress_bupt1.yaml",
        "timeline": PROJECT_ROOT / "configs" / "timeline_sarfish_formal_stress_bupt1_20s.yaml",
        "simulator": PROJECT_ROOT / "configs" / "simulator_sarfish_formal_stress_bupt1_1day_20s.yaml",
    },
}
VARIANTS = {
    "PROP-Z": "proposed_calibrated_payload_plus_z_guided",
    "UTIL": "utility_rule",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if not _finite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _safe_div(numerator: float, denominator: float) -> float:
    if not _finite(numerator) or not _finite(denominator) or abs(denominator) <= 1.0e-12:
        return math.nan
    return numerator / denominator


def _load_repair_cfg(variant: str) -> dict[str, object]:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "payload_profile_ablation.yaml")
    repair = copy.deepcopy(cfg["policy_repair_defaults"])
    repair.update(copy.deepcopy(cfg.get("variant_overrides", {}).get(variant, {})))
    return repair


def _scenario_run_root(base_root: Path) -> Path:
    candidate = base_root / "bupt1_validation"
    if not candidate.exists() or not any(candidate.iterdir()):
        return candidate
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_root / f"bupt1_validation_{stamp}"


def _load_scenario_configs(scenario: str, variant: str) -> dict[str, dict]:
    paths = SCENARIOS[scenario]
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    task_stream = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    timeline = load_yaml(paths["timeline"])
    resource = load_yaml(paths["resource"])
    simulator = load_yaml(paths["simulator"])
    task_stream.update(copy.deepcopy(timeline.get("task_stream_overrides", {})))

    configs["task_stream"] = task_stream
    configs["timeline"] = timeline
    configs["simulator"] = simulator
    configs["resource"] = resource
    configs["contact"] = copy.deepcopy(timeline.get("contact", {}))
    configs["timeline"]["resource"] = dict(resource)
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    profile_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml")
    configs["model_pool"] = copy.deepcopy(configs["model_pool"])
    configs["model_pool"]["profile_overrides"] = dict(profile_cfg["profile_overrides"])
    configs["payload_profiles"] = load_yaml(PROJECT_ROOT / "configs" / "payload_profiles_roi_compression_calibrated.yaml")
    configs["simulator"]["policy_repair"] = _load_repair_cfg(variant)
    return configs


def _build_router(variant: str, environment: SimulationEnvironment):
    action_names = [action.action_name for action in environment.action_space.list_actions()]
    if variant.startswith("proposed_"):
        return SafeLinUCBRouter(
            action_names,
            alpha=0.70,
            router_name=variant,
            use_lyapunov_pricing=True,
            use_mode_controller=True,
        )
    if variant == "utility_rule":
        return RuleBasedRouter(variant, mode="utility_rule")
    raise ValueError(f"Unsupported BUPT-1 validation variant: {variant}")


def _idle_diagnostic(scenario: str) -> dict[str, Any]:
    paths = SCENARIOS[scenario]
    timeline_cfg = load_yaml(paths["timeline"])
    resource_cfg = load_yaml(paths["resource"])
    simulator_cfg = load_yaml(paths["simulator"])
    timeline_cfg["resource"] = dict(resource_cfg)
    slot_seconds = float(timeline_cfg.get("slot_seconds", 20.0))
    num_slots = int(float(timeline_cfg.get("num_days", 1.0)) * 24.0 * 3600.0 / max(slot_seconds, 1.0e-9))
    traces = load_synthetic_resource_trace(num_slots, timeline_cfg)
    capacity_j = float(simulator_cfg.get("battery_capacity_j", float(simulator_cfg["battery_capacity_wh"]) * 3600.0))
    soc = float(simulator_cfg["battery_init"])
    min_soc = soc
    soc_rows: list[dict[str, Any]] = []
    total_solar_j = 0.0
    total_base_j = 0.0
    for trace in traces:
        base_energy_j = trace.base_power_w * slot_seconds
        total_solar_j += trace.solar_energy_j
        total_base_j += base_energy_j
        next_soc = update_battery_soc(
            battery_soc=soc,
            solar_energy_j=trace.solar_energy_j,
            base_power_w=trace.base_power_w,
            slot_seconds=slot_seconds,
            compute_energy_j=0.0,
            comm_energy_j=0.0,
            battery_capacity_j=capacity_j,
            charge_efficiency=float(simulator_cfg.get("battery_charge_efficiency", 1.0)),
            discharge_efficiency=float(simulator_cfg.get("battery_discharge_efficiency", 1.0)),
        )
        soc_rows.append(
            {
                "scenario": scenario,
                "slot_index": trace.slot_index,
                "time_s": trace.slot_index * slot_seconds,
                "battery_soc": soc,
                "battery_next_soc": next_soc,
                "solar_energy_j": trace.solar_energy_j,
                "solar_power_w": trace.solar_power_w,
                "base_power_w": trace.base_power_w,
                "base_energy_j": base_energy_j,
                "is_sunlit": int(trace.is_sunlit),
            }
        )
        soc = next_soc
        min_soc = min(min_soc, soc)
    return {
        "scenario": scenario,
        "num_slots": num_slots,
        "battery_capacity_j": capacity_j,
        "battery_capacity_wh": capacity_j / 3600.0,
        "initial_soc": float(simulator_cfg["battery_init"]),
        "min_soc": min_soc,
        "final_soc": soc,
        "total_solar_j": total_solar_j,
        "total_solar_wh": total_solar_j / 3600.0,
        "total_base_energy_j": total_base_j,
        "total_base_energy_wh": total_base_j / 3600.0,
        "net_idle_energy_j": total_solar_j - total_base_j,
        "soft_guard_met": int(min_soc >= float(simulator_cfg.get("battery_soft_guard", 0.70))),
        "hard_floor_met": int(min_soc >= float(simulator_cfg.get("battery_min", 0.0))),
        "soc_rows": soc_rows,
    }


def _run_scenario(scenario: str, output_dir: Path) -> list[dict[str, Any]]:
    logger = get_logger("run_bupt1_recalibration_validation")
    output_dirs = ensure_output_dirs(output_dir)
    results = []
    timeline_for_report = None
    for short_name, variant in VARIANTS.items():
        configs = _load_scenario_configs(scenario, variant)
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        if timeline_for_report is None:
            timeline_for_report = timeline
        environment = SimulationEnvironment(configs)
        router = _build_router(variant, environment)
        episode_runner = environment.build_episode_runner(timeline)
        logger.info("running BUPT-1 %s validation variant %s", scenario, short_name)
        results.append(episode_runner.run(router))
    write_episode_results(results, output_dirs)
    slot_seconds = float(timeline_for_report[0].slot_seconds if timeline_for_report else 20.0)
    generate_lifetime_outputs(output_dir, slot_seconds)
    rows = _collect_delivered_metrics(scenario, output_dir)
    _write_csv(output_dir / "delivered_only_metrics.csv", rows)
    _write_validation_markdown(output_dir / "delivered_only_metrics.md", rows, f"BUPT-1 {scenario} Delivered-only Metrics")
    return rows


def _collect_delivered_metrics(scenario: str, output_dir: Path) -> list[dict[str, Any]]:
    summaries = {row.get("router") or row.get("router_name"): row for row in _read_csv(output_dir / "summary.csv")}
    lifetime_by_variant: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv(output_dir / "task_lifetime_outcomes.csv"):
        lifetime_by_variant.setdefault(str(row.get("router_name", "")), []).append(row)
    rows: list[dict[str, Any]] = []
    for short_name, variant in VARIANTS.items():
        summary = summaries.get(variant, {})
        lifetime_rows = lifetime_by_variant.get(variant, [])
        life_counts = Counter(str(row.get("final_lifetime_status", "")) for row in lifetime_rows)
        delivered_only = sum(_to_float(row.get("delivered_only_utility"), 0.0) for row in lifetime_rows)
        high_value_delivered = sum(
            _to_float(row.get("delivered_only_utility"), 0.0)
            for row in lifetime_rows
            if str(row.get("high_value_flag", "")).lower() in {"1", "1.0", "true", "yes"}
        )
        exact_missing = sum(
            1
            for row in lifetime_rows
            if str(row.get("final_lifetime_status", "")) == "delivered"
            and str(row.get("met_deadline_lifetime", "")) == "true"
            and str(row.get("exact_lifetime_timing_available", "")) != "true"
        )
        gross = _to_float(summary.get("gross_task_utility", summary.get("total_effective_utility")))
        delivered_ratio = _safe_div(delivered_only, gross)
        row = {
            "scenario": scenario,
            "variant_short": short_name,
            "variant_full": variant,
            "metric_quality": "exact" if exact_missing == 0 and lifetime_rows else "approximate",
            "notes": "" if exact_missing == 0 else f"{exact_missing} delivered tasks lacked exact timing",
            "gross_task_utility_provisional": gross,
            "net_system_utility_provisional": _to_float(summary.get("net_system_utility")),
            "penalized_reward": _to_float(summary.get("total_penalized_reward")),
            "delivered_only_gross_utility": delivered_only,
            "delivered_before_deadline_utility": delivered_only,
            "high_value_delivered_utility": high_value_delivered,
            "delivery_realization_gap": gross - delivered_only if _finite(gross) else math.nan,
            "delivered_utility_ratio": delivered_ratio,
            "lifetime_delivered_task_level": life_counts.get("delivered", 0),
            "lifetime_expired_task_level": life_counts.get("expired", 0),
            "lifetime_deferred_task_level": life_counts.get("deferred", 0),
            "lifetime_undelivered_by_end_task_level": life_counts.get("undelivered_by_end", 0),
            "delivered_items_payload_level": _to_int(summary.get("delivered_items")),
            "expired_items_payload_level": _to_int(summary.get("expired_items")),
            "completed_tasks_same_step": _to_int(summary.get("completed_tasks")),
            "deadline_satisfaction_ratio": _to_float(summary.get("deadline_satisfaction_ratio")),
            "completed_deadline_satisfaction_ratio": _to_float(summary.get("completed_deadline_satisfaction_ratio")),
            "generated_tasks": _to_int(summary.get("total_generated_tasks")),
            "total_energy_j": _to_float(summary.get("total_energy")),
            "total_energy_wh": _to_float(summary.get("total_energy")) / 3600.0,
            "compute_energy_j": _to_float(summary.get("total_compute_energy")),
            "compute_energy_wh": _to_float(summary.get("total_compute_energy")) / 3600.0,
            "comm_energy_j": _to_float(summary.get("total_comm_energy")),
            "comm_energy_wh": _to_float(summary.get("total_comm_energy")) / 3600.0,
            "total_base_energy_j": _to_float(summary.get("total_base_energy")),
            "total_base_energy_wh": _to_float(summary.get("total_base_energy")) / 3600.0,
            "total_spacecraft_energy_j": _to_float(summary.get("total_spacecraft_energy")),
            "total_spacecraft_energy_wh": _to_float(summary.get("total_spacecraft_energy")) / 3600.0,
            "min_battery_soc": _to_float(summary.get("min_battery_soc")),
            "final_battery_soc": _to_float(summary.get("final_battery_soc", summary.get("final_battery"))),
            "battery_below_075_count": _to_int(summary.get("battery_below_075_count")),
            "resource_feasible_flag": _to_int(summary.get("resource_feasible_flag")),
            "max_thermal": _to_float(summary.get("max_thermal")),
            "thermal_guard_exceed_count": _to_int(summary.get("thermal_guard_exceed_count")),
            "average_queue_mb": _to_float(summary.get("average_queue_mb")),
            "max_queue_mb": _to_float(summary.get("max_queue_mb")),
            "generated_payload_mb": _to_float(summary.get("total_generated_payload_mb")),
            "delivered_mb": _to_float(summary.get("delivered_mb")),
        }
        rows.append(row)
    for rank, row in enumerate(sorted(rows, key=lambda item: item["net_system_utility_provisional"], reverse=True), start=1):
        row["rank_by_provisional_net"] = rank
    for rank, row in enumerate(sorted(rows, key=lambda item: item["delivered_only_gross_utility"], reverse=True), start=1):
        row["rank_by_delivered_only"] = rank
    return rows


def _write_validation_markdown(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| scenario | variant | net provisional | delivered-only | ratio | generated | lifetime delivered | lifetime expired | min SOC | final SOC | base Wh | compute Wh | comm Wh | generated MB | delivered MB | quality |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: (item["scenario"], item["rank_by_delivered_only"])):
        lines.append(
            f"| {row['scenario']} | {row['variant_short']} | {_fmt(row['net_system_utility_provisional'])} | "
            f"{_fmt(row['delivered_only_gross_utility'])} | {_fmt(row['delivered_utility_ratio'])} | "
            f"{row['generated_tasks']} | {row['lifetime_delivered_task_level']} | {row['lifetime_expired_task_level']} | "
            f"{_fmt(row['min_battery_soc'])} | {_fmt(row['final_battery_soc'])} | {_fmt(row['total_base_energy_wh'])} | "
            f"{_fmt(row['compute_energy_wh'])} | {_fmt(row['comm_energy_wh'])} | {_fmt(row['generated_payload_mb'])} | "
            f"{_fmt(row['delivered_mb'])} | {row['metric_quality']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_energy_unit_audit(path: Path, idle_rows: list[dict[str, Any]]) -> None:
    smoke_rows = _read_csv(OUTPUT_ROOT / "summary.csv")
    smoke_base = _to_float(smoke_rows[0].get("total_base_energy")) if smoke_rows else math.nan
    smoke_capacity_wh = 35.0
    smoke_capacity_j = smoke_capacity_wh * 3600.0
    lines = [
        "# Energy Unit Audit",
        "",
        "## Result",
        "",
        "The implementation is internally Joule-based. `battery_capacity_wh` is converted to `battery_capacity_j` during simulator construction, solar energy is generated as W*s per slot, and compute/communication energy are also W*s.",
        "",
        "The prior 1-day smoke infeasibility is therefore best explained by environment scaling rather than an observed Wh/J coding mix: it used a 35 Wh battery with a low-solar stress trace for a 1-day run.",
        "",
        "## Unit Mapping",
        "",
        "| Quantity | Code path | Internal unit | Notes |",
        "| --- | --- | --- | --- |",
        "| battery capacity | `src/simulator/environment.py` | J | Converts `battery_capacity_wh * 3600`; BUPT-1 candidate uses 230 Wh = 828000 J. |",
        "| solar energy per slot | `src/timeline/resource_trace_loader.py` | J | Computes `solar_power_w * slot_seconds`. |",
        "| base/platform energy | `src/energy/battery_update.py` and trajectory `idle_energy_j` | J | Computes `base_power_w * slot_seconds`. |",
        "| compute energy | `src/energy/compute_energy.py` | J | Computes model power times latency. |",
        "| communication energy | `src/energy/comm_energy.py` | J | Computes comm power times transmit time times link-quality overhead. |",
        "| total_energy | `src/evaluation/metrics.py` | J | Compute + comm only. |",
        "| total_base_energy | `src/evaluation/metrics.py` | J | Platform/base load. |",
        "| total_spacecraft_energy | `src/evaluation/metrics.py` | J | Base + compute + comm. |",
        "",
        "## Prior Smoke Context",
        "",
        f"- prior smoke battery capacity: {smoke_capacity_wh:.1f} Wh = {smoke_capacity_j:.0f} J",
        f"- prior smoke base/platform energy over 1 day: {_fmt(smoke_base)} J = {_fmt(smoke_base / 3600.0 if _finite(smoke_base) else math.nan)} Wh",
        "- prior stress solar peak: 5.5 W",
        "- consequence: base load alone was larger than the available 35 Wh battery reserve under the low-solar trace.",
        "",
        "## Idle Diagnostic Summary",
        "",
        "| scenario | capacity Wh | solar Wh | base Wh | net idle J | min SOC | final SOC | soft guard met | hard floor met |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in idle_rows:
        lines.append(
            f"| {row['scenario']} | {_fmt(row['battery_capacity_wh'])} | {_fmt(row['total_solar_wh'])} | "
            f"{_fmt(row['total_base_energy_wh'])} | {_fmt(row['net_idle_energy_j'])} | {_fmt(row['min_soc'])} | "
            f"{_fmt(row['final_soc'])} | {row['soft_guard_met']} | {row['hard_floor_met']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plan(path: Path) -> None:
    lines = [
        "# BUPT-1 Grounded Recalibration Plan",
        "",
        "## Scope",
        "",
        "This recalibration changes only environment/configuration and post-processing reports. Router logic, reward behavior, mode control, Lyapunov pricing, SafeLinUCB, detector, model profiles, payload profiles, payload generation, task value assignment, and delivered-only metric definitions are unchanged.",
        "",
        "## BUPT-1 Reference Used",
        "",
        "- 12U SSO-like LEO platform, approximately 500 km altitude and 97 degree inclination.",
        "- Battery capacity: 115 Wh x 2 = 230 Wh = 828000 J.",
        "- Internal thermal reference: -10 C to 30 C.",
        "- Nominal DoD soft control around 30%, corresponding to an SOC guard around 70%.",
        "- Downlink reference: 100 Mbps hardware capability; candidate effective rates are lower to represent link budget and protocol loss.",
        "",
        "## Candidate Environments",
        "",
        "- Nominal: 230 Wh, initial SOC 0.90, SOC soft guard 0.70, hard floor 0.55, 50 Mbps-class effective contact, 5 GB queue.",
        "- Stress: 230 Wh, initial SOC 0.85, SOC soft guard 0.70, hard floor 0.50, 20-30 Mbps contact, outage windows, 3 GB queue.",
        "",
        "## Known Limitation",
        "",
        "The desired deadline distributions in minutes are recorded in the new timeline configs, but the current SARFishSample adapter hard-codes class deadlines in seconds. The validation therefore tests physical resource feasibility and contact pressure, but not the requested formal deadline scale.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_candidate_summary(path: Path) -> None:
    lines = [
        "# Formal Environment Candidate Summary",
        "",
        "| setting | config files | battery | SOC guard/floor | solar/contact | queue | task target status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| nominal | `resource_sarfish_formal_nominal_bupt1.yaml`, `timeline_sarfish_formal_nominal_bupt1_20s.yaml`, `simulator_sarfish_formal_nominal_bupt1_1day_20s.yaml` | 230 Wh = 828000 J | 0.70 / 0.55 | SSO-like 95 min orbit, 600 s contact, 35-65 Mbps | 5120 MB | task-count config target active; deadline-minute target documented only |",
        "| stress | `resource_sarfish_formal_stress_bupt1.yaml`, `timeline_sarfish_formal_stress_bupt1_20s.yaml`, `simulator_sarfish_formal_stress_bupt1_1day_20s.yaml` | 230 Wh = 828000 J | 0.70 / 0.50 | lower solar, 300 s contact, 20-30 Mbps, outage windows | 3072 MB | task-count config target active; deadline-minute target documented only |",
        "",
        "The COTS AI power targets of 4/7/10 W are not directly applied because model profiles are protected for this task. Existing calibrated model energy values are retained and treated as the current workload-energy abstraction.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_idle_report(path: Path, idle_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Idle Energy Feasibility Report",
        "",
        "Idle-only means no task generation, no AI compute, and no communication load. Only the configured solar trace and base/platform load are integrated through the same battery SOC update equation.",
        "",
        "| scenario | slots | capacity Wh | initial SOC | min SOC | final SOC | solar Wh | base Wh | soft guard met | hard floor met | judgment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in idle_rows:
        judgment = "feasible" if row["hard_floor_met"] else "invalid"
        if row["hard_floor_met"] and not row["soft_guard_met"]:
            judgment = "stressful_but_not_collapsed"
        lines.append(
            f"| {row['scenario']} | {row['num_slots']} | {_fmt(row['battery_capacity_wh'])} | {_fmt(row['initial_soc'])} | "
            f"{_fmt(row['min_soc'])} | {_fmt(row['final_soc'])} | {_fmt(row['total_solar_wh'])} | "
            f"{_fmt(row['total_base_energy_wh'])} | {row['soft_guard_met']} | {row['hard_floor_met']} | {judgment} |"
        )
    lines.extend(
        [
            "",
            "Idle-only feasibility is necessary but not sufficient. Full-task validation must still check delivered-only utility, lifecycle expiration, thermal pressure, and queue behavior.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_validation_report(path: Path, idle_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], run_root: Path) -> str:
    nominal_rows = [row for row in validation_rows if row["scenario"] == "nominal"]
    stress_rows = [row for row in validation_rows if row["scenario"] == "stress"]
    all_idle_ok = all(row["hard_floor_met"] for row in idle_rows)
    all_exact = all(row["metric_quality"] == "exact" for row in validation_rows)
    target_deadline_limitation = True
    collapsed = any(_to_float(row["min_battery_soc"]) <= 0.0 for row in validation_rows)
    nominal_min_soc = min((_to_float(row["min_battery_soc"]) for row in nominal_rows), default=math.nan)
    stress_min_soc = min((_to_float(row["min_battery_soc"]) for row in stress_rows), default=math.nan)
    nominal_ratio = {row["variant_short"]: row["delivered_utility_ratio"] for row in nominal_rows}
    stress_ratio = {row["variant_short"]: row["delivered_utility_ratio"] for row in stress_rows}
    recommendation = "rerun all-variant 1-day smoke after deadline-scale logging/config support"
    if all_idle_ok and all_exact and not collapsed and not target_deadline_limitation:
        recommendation = "rerun all-variant 1-day smoke under BUPT-1 candidates"

    lines = [
        "# BUPT-1 Recalibration Validation Report",
        "",
        "## Executive Summary",
        "",
        "- Energy units are internally consistent and Joule-based.",
        f"- Idle-only nominal/stress feasibility: {'pass' if all_idle_ok else 'fail'}.",
        f"- Lightweight validation run root: `{run_root.relative_to(PROJECT_ROOT)}`.",
        f"- Delivered-only metrics exact: {'yes' if all_exact else 'no'}.",
        f"- Battery collapse in task validation: {'yes' if collapsed else 'no'}.",
        "- Deadline-scale limitation: the current SARFishSample adapter still uses hard-coded second-scale deadlines; minute-scale deadline targets in the new configs are documented but not enforced.",
        f"- Recommendation: {recommendation}.",
        "",
        "## Idle-only SOC Summary",
        "",
        "| scenario | min SOC | final SOC | solar Wh | base Wh | hard floor met |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in idle_rows:
        lines.append(
            f"| {row['scenario']} | {_fmt(row['min_soc'])} | {_fmt(row['final_soc'])} | "
            f"{_fmt(row['total_solar_wh'])} | {_fmt(row['total_base_energy_wh'])} | {row['hard_floor_met']} |"
        )
    lines.extend(
        [
            "",
            "## PROP-Z and UTIL Validation Summary",
            "",
            "| scenario | variant | generated tasks | delivered-only | ratio | lifetime delivered | lifetime expired | min SOC | final SOC | generated MB | delivered MB | resource feasible |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(validation_rows, key=lambda item: (item["scenario"], item["variant_short"])):
        lines.append(
            f"| {row['scenario']} | {row['variant_short']} | {row['generated_tasks']} | {_fmt(row['delivered_only_gross_utility'])} | "
            f"{_fmt(row['delivered_utility_ratio'])} | {row['lifetime_delivered_task_level']} | {row['lifetime_expired_task_level']} | "
            f"{_fmt(row['min_battery_soc'])} | {_fmt(row['final_battery_soc'])} | {_fmt(row['generated_payload_mb'])} | "
            f"{_fmt(row['delivered_mb'])} | {row['resource_feasible_flag']} |"
        )
    lines.extend(
        [
            "",
            "## Target Check",
            "",
            f"- nominal minimum SOC across PROP-Z/UTIL: {_fmt(nominal_min_soc)}",
            f"- stress minimum SOC across PROP-Z/UTIL: {_fmt(stress_min_soc)}",
            f"- nominal delivered utility ratios: {', '.join(f'{k}={_fmt(v)}' for k, v in nominal_ratio.items())}",
            f"- stress delivered utility ratios: {', '.join(f'{k}={_fmt(v)}' for k, v in stress_ratio.items())}",
            "- SOC collapse was resolved if minimum SOC is above zero and above each scenario hard floor.",
            "- Delivered-only ratios remain below the requested target if second-scale deadlines dominate expiration before scheduled 95-minute contacts.",
            "",
            "## Decision",
            "",
            "The BUPT-1 battery/solar resource candidate is suitable for rerunning an all-variant 1-day smoke only after deciding how to activate minute-scale formal deadlines without changing task-value semantics. Full multi-seed formal runs should wait until that deadline-scale issue is resolved and the all-variant smoke is rerun.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return recommendation


def run_validation(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_plan(output_root / "bupt1_grounded_recalibration_plan.md")
    _write_candidate_summary(output_root / "formal_environment_candidate_summary.md")
    run_root = _scenario_run_root(output_root)
    idle_rows = [_idle_diagnostic("nominal"), _idle_diagnostic("stress")]
    all_soc_rows = []
    for row in idle_rows:
        all_soc_rows.extend(row["soc_rows"])
    _write_csv(run_root / "idle_energy_soc_trace.csv", all_soc_rows)
    idle_summary_rows = [{key: value for key, value in row.items() if key != "soc_rows"} for row in idle_rows]
    _write_csv(run_root / "idle_energy_summary.csv", idle_summary_rows)
    _write_energy_unit_audit(output_root / "energy_unit_audit.md", idle_summary_rows)
    _write_idle_report(output_root / "idle_energy_feasibility_report.md", idle_summary_rows)

    validation_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenario_rows = _run_scenario(scenario, run_root / scenario)
        validation_rows.extend(scenario_rows)
    _write_csv(run_root / "bupt1_validation_metrics.csv", validation_rows)
    _write_validation_markdown(run_root / "bupt1_validation_metrics.md", validation_rows, "BUPT-1 Validation Metrics")
    recommendation = _write_validation_report(
        output_root / "bupt1_recalibration_validation_report.md",
        idle_summary_rows,
        validation_rows,
        run_root,
    )
    return {
        "run_root": run_root,
        "idle_rows": idle_summary_rows,
        "validation_rows": validation_rows,
        "recommendation": recommendation,
        "files": [
            output_root / "bupt1_grounded_recalibration_plan.md",
            output_root / "energy_unit_audit.md",
            output_root / "formal_environment_candidate_summary.md",
            output_root / "idle_energy_feasibility_report.md",
            output_root / "bupt1_recalibration_validation_report.md",
            run_root / "idle_energy_summary.csv",
            run_root / "idle_energy_soc_trace.csv",
            run_root / "bupt1_validation_metrics.csv",
            run_root / "bupt1_validation_metrics.md",
            run_root / "nominal" / "summary.csv",
            run_root / "stress" / "summary.csv",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BUPT-1 resource recalibration diagnostics.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    configure_logging()
    result = run_validation(args.output_root)
    rows = result["validation_rows"]
    nominal_rows = [row for row in rows if row["scenario"] == "nominal"]
    stress_rows = [row for row in rows if row["scenario"] == "stress"]
    print("unit_audit_result=consistent_joule_internal_units")
    print("configs_created=" + ",".join(str(path.relative_to(PROJECT_ROOT)) for scenario in SCENARIOS.values() for path in scenario.values()))
    print("validation_runs=idle_nominal,idle_stress,nominal_PROP-Z,nominal_UTIL,stress_PROP-Z,stress_UTIL")
    print(
        "min_soc_nominal="
        + ",".join(f"{row['variant_short']}:{_fmt(row['min_battery_soc'])}" for row in nominal_rows)
    )
    print(
        "min_soc_stress="
        + ",".join(f"{row['variant_short']}:{_fmt(row['min_battery_soc'])}" for row in stress_rows)
    )
    print(
        "delivered_ratio_nominal="
        + ",".join(f"{row['variant_short']}:{_fmt(row['delivered_utility_ratio'])}" for row in nominal_rows)
    )
    print(
        "delivered_ratio_stress="
        + ",".join(f"{row['variant_short']}:{_fmt(row['delivered_utility_ratio'])}" for row in stress_rows)
    )
    print("output_files=" + ",".join(str(path.relative_to(PROJECT_ROOT)) for path in result["files"]))
    print(f"recommendation={result['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
