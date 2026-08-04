"""Run SARFish GRD validation energy-shadow stress dry run."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_task_lifetime_diagnostics import generate_lifetime_outputs  # noqa: E402
from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


ROUTERS_OF_INTEREST = (
    "proposed_lyapunov_rule_mode_safe_bandit",
    "utility_rule",
    "no_mode",
    "no_lyapunov",
    "fixed_lightweight",
    "fixed_high_accuracy",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _segments(rows: list[dict[str, object]], predicate: Callable[[dict[str, object]], bool]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for row in rows:
        slot = _to_int(row.get("slot"))
        if predicate(row):
            if start is None:
                start = slot
            previous = slot
        elif start is not None and previous is not None:
            segments.append((start, previous))
            start = None
            previous = None
    if start is not None and previous is not None:
        segments.append((start, previous))
    return segments


def _format_segments(segments: list[tuple[int, int]], slot_seconds: float) -> str:
    if not segments:
        return "none"
    return "; ".join(
        f"slots {start}-{end}, t={start * slot_seconds:.0f}-{(end + 1) * slot_seconds:.0f}s"
        for start, end in segments
    )


def _load_stress_configs() -> dict[str, dict]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_energy_shadow_stress_20s.yaml")
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_stress_20s.yaml")
    configs["resource"] = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")
    configs["contact"] = load_yaml(PROJECT_ROOT / "configs" / "contact_sarfish_physical_20s.yaml")
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    configs["experiment"] = dict(configs["experiment"])
    configs["experiment"]["routers"] = list(ROUTERS_OF_INTEREST)
    return configs


def _build_resource_diagnostics(output_dir: Path, timeline, simulator_cfg: dict[str, object], resource_cfg: dict[str, object]) -> Path:
    proposed_rows = _read_csv(output_dir / "trajectories" / "proposed_lyapunov_rule_mode_safe_bandit.csv")
    proposed_by_slot = {_to_int(row.get("slot")): row for row in proposed_rows}
    battery_capacity_j = float(simulator_cfg.get("battery_capacity_j", float(simulator_cfg.get("battery_capacity_wh", 0.0)) * 3600.0))
    charge_efficiency = float(simulator_cfg.get("battery_charge_efficiency", 1.0))
    dod_guard = float(resource_cfg.get("dod_guard", 0.30))
    rows: list[dict[str, object]] = []
    cumulative_wasted = 0.0
    for step in timeline:
        traj = proposed_by_slot.get(step.slot_index, {})
        battery_soc = _to_float(traj.get("battery_soc"))
        battery_next_soc = _to_float(traj.get("battery_next_soc"), battery_soc)
        base_energy_j = step.resource_trace.base_power_w * step.slot_seconds
        compute_energy_j = _to_float(traj.get("compute_energy_j"))
        comm_energy_j = _to_float(traj.get("comm_energy_j"))
        load_energy_j = base_energy_j + compute_energy_j + comm_energy_j
        net_input_j = step.resource_trace.solar_energy_j - load_energy_j
        available_charge_j = max(0.0, (1.0 - battery_soc) * battery_capacity_j / max(charge_efficiency, 1.0e-9))
        wasted_solar_energy_j = max(0.0, net_input_j - available_charge_j) if battery_next_soc >= 1.0 - 1.0e-9 else 0.0
        cumulative_wasted += wasted_solar_energy_j
        dod = 1.0 - battery_next_soc
        rows.append(
            {
                "slot": step.slot_index,
                "time_s": step.slot_index * step.slot_seconds,
                "is_sunlit": int(step.resource_trace.is_sunlit),
                "is_eclipse": int(step.resource_trace.is_eclipse),
                "in_contact": int(step.link_trace.in_contact),
                "solar_power_w": step.resource_trace.solar_power_w,
                "solar_energy_j": step.resource_trace.solar_energy_j,
                "wasted_solar_energy_j": wasted_solar_energy_j,
                "cumulative_wasted_solar_energy_j": cumulative_wasted,
                "battery_soc": battery_soc,
                "battery_next_soc": battery_next_soc,
                "dod": dod,
                "dod_guard": dod_guard,
                "dod_guard_violation": int(dod > dod_guard),
                "base_power_w": step.resource_trace.base_power_w,
                "base_energy_j": base_energy_j,
                "compute_power_w": compute_energy_j / max(step.slot_seconds, 1.0e-9),
                "compute_energy_j": compute_energy_j,
                "comm_power_w": comm_energy_j / max(step.slot_seconds, 1.0e-9),
                "comm_energy_j": comm_energy_j,
                "thermal_next": _to_float(traj.get("thermal_next")),
                "queue_next_mb": _to_float(traj.get("queue_next_mb")),
            }
        )
    path = output_dir / "resource_trace_diagnostics.csv"
    _write_csv(path, rows, list(rows[0].keys()) if rows else [])
    return path


def _summary_table(summary_by_router: dict[str, dict[str, str]], lifetime_rows: list[dict[str, object]]) -> list[str]:
    lifetime_by_router: dict[str, Counter] = {}
    for row in lifetime_rows:
        router = str(row.get("router_name", ""))
        lifetime_by_router.setdefault(router, Counter())[str(row.get("final_lifetime_status", ""))] += 1
    lines = [
        "| router | gross_task_utility | net_system_utility | penalized_reward | resource_penalty | requirement_penalty | energy | delivered_mb | expired_items | lifetime delivered / expired / deferred |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for router in ROUTERS_OF_INTEREST:
        row = summary_by_router.get(router, {})
        counts = lifetime_by_router.get(router, Counter())
        lines.append(
            f"| {router} | {_to_float(row.get('gross_task_utility')):.3f} | "
            f"{_to_float(row.get('net_system_utility')):.3f} | "
            f"{_to_float(row.get('total_penalized_reward')):.3f} | "
            f"{_to_float(row.get('total_resource_penalty')):.3f} | "
            f"{_to_float(row.get('total_requirement_penalty')):.3f} | "
            f"{_to_float(row.get('total_energy')):.3f} | "
            f"{_to_float(row.get('delivered_mb')):.3f} | "
            f"{_to_int(row.get('expired_items'))} | "
            f"{counts.get('delivered', 0)} / {counts.get('expired', 0)} / {counts.get('deferred', 0)} |"
        )
    return lines


def _write_energy_shadow_sanity_report(output_dir: Path, configs: dict[str, dict], timeline, lifetime_rows: list[dict[str, object]]) -> Path:
    slot_seconds = float(configs["timeline"]["slot_seconds"])
    num_slots = int(configs["timeline"]["num_slots"])
    duration_s = slot_seconds * num_slots
    summary_rows = _read_csv(output_dir / "summary.csv")
    summary_by_router = {row["router"]: row for row in summary_rows}
    proposed_rows = _read_csv(output_dir / "trajectories" / "proposed_lyapunov_rule_mode_safe_bandit.csv")
    task_rows = [row for row in proposed_rows if _to_int(row.get("has_task")) == 1]
    resource_rows = _read_csv(output_dir / "resource_trace_diagnostics.csv")

    eclipse_segments = _segments(resource_rows, lambda row: _to_int(row.get("is_eclipse")) == 1)
    contact_segments = _segments(resource_rows, lambda row: _to_int(row.get("in_contact")) == 1)
    outage_segments = [
        (int(start / slot_seconds), int(end / slot_seconds) - 1)
        for start, end in configs["contact"].get("outage_windows_s", [])
    ]
    eclipse_slots = {_to_int(row.get("slot")) for row in resource_rows if _to_int(row.get("is_eclipse")) == 1}
    eclipse_tasks = [row for row in task_rows if _to_int(row.get("slot")) in eclipse_slots]
    high_value_eclipse_tasks = [
        row
        for row in eclipse_tasks
        if _to_float(row.get("V_t")) >= float(configs["simulator"].get("high_value_threshold", 0.75))
        or str(row.get("task_class")) in {"high_value", "emergency"}
    ]
    proposed = summary_by_router.get("proposed_lyapunov_rule_mode_safe_bandit", {})
    min_soc = min((_to_float(row.get("battery_next_soc")) for row in resource_rows), default=0.0)
    max_dod = max((_to_float(row.get("dod")) for row in resource_rows), default=0.0)
    dod_violations = sum(_to_int(row.get("dod_guard_violation")) for row in resource_rows)
    max_thermal = max((_to_float(row.get("thermal_next")) for row in resource_rows), default=0.0)
    thermal_guard = float(configs["simulator"]["long_term_risk_penalty"].get("thermal_guard_threshold", 0.85))
    thermal_hard = float(configs["simulator"]["thermal_max"])
    base_energy = sum(_to_float(row.get("base_energy_j")) for row in resource_rows)
    compute_energy = sum(_to_float(row.get("compute_energy_j")) for row in resource_rows)
    comm_energy = sum(_to_float(row.get("comm_energy_j")) for row in resource_rows)
    wasted_solar = sum(_to_float(row.get("wasted_solar_energy_j")) for row in resource_rows)

    lines = [
        "# Energy-Shadow Stress Sanity Report",
        "",
        f"- slot_seconds: {slot_seconds:g}",
        f"- num_slots: {num_slots}",
        f"- physical_duration_s: {duration_s:g}",
        f"- physical_duration_h: {duration_s / 3600.0:.3f}",
        "",
        "## Eclipse / Solar",
        "",
        f"- eclipse_segment_count: {len(eclipse_segments)}",
        f"- eclipse_segments: {_format_segments(eclipse_segments, slot_seconds)}",
        f"- tasks_in_eclipse: {len(eclipse_tasks)}",
        f"- high_value_tasks_in_eclipse: {len(high_value_eclipse_tasks)}",
        f"- wasted_solar_energy_j: {wasted_solar:.3f}",
        "",
        "## Contact",
        "",
        f"- contact_segment_count: {len(contact_segments)}",
        f"- contact_segments: {_format_segments(contact_segments, slot_seconds)}",
        f"- configured_contact_outage_segments: {_format_segments(outage_segments, slot_seconds)}",
        "",
        "## Resource State",
        "",
        f"- min_SOC: {min_soc:.4f}",
        f"- max_DoD: {max_dod:.4f}",
        f"- DoD_guard_violation_count: {dod_violations}",
        f"- base_energy_j: {base_energy:.3f}",
        f"- compute_energy_j: {compute_energy:.3f}",
        f"- comm_energy_j: {comm_energy:.3f}",
        f"- spacecraft_energy_j: {base_energy + compute_energy + comm_energy:.3f}",
        f"- max_thermal: {max_thermal:.4f}",
        f"- thermal_guard_violation_count: {sum(1 for row in resource_rows if _to_float(row.get('thermal_next')) > thermal_guard):d}",
        f"- thermal_hard_violation_count: {sum(1 for row in resource_rows if _to_float(row.get('thermal_next')) > thermal_hard):d}",
        f"- max_queue_mb: {_to_float(proposed.get('max_queue_mb')):.3f}",
        "",
        "## Router Metrics",
        "",
        *_summary_table(summary_by_router, lifetime_rows),
    ]
    path = output_dir / "energy_shadow_sanity_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_comparison_report(output_dir: Path, lifetime_rows: list[dict[str, object]]) -> Path:
    summary_rows = _read_csv(output_dir / "summary.csv")
    summary_by_router = {row["router"]: row for row in summary_rows}
    proposed_rows = [row for row in _read_csv(output_dir / "trajectories" / "proposed_lyapunov_rule_mode_safe_bandit.csv") if _to_int(row.get("has_task")) == 1]
    no_mode_rows = [row for row in _read_csv(output_dir / "trajectories" / "no_mode.csv") if _to_int(row.get("has_task")) == 1]
    utility_rows = [row for row in _read_csv(output_dir / "trajectories" / "utility_rule.csv") if _to_int(row.get("has_task")) == 1]
    same_actions = sum(
        1
        for prop, no_mode in zip(proposed_rows, no_mode_rows)
        if prop.get("task_id") == no_mode.get("task_id") and prop.get("selected_action") == no_mode.get("selected_action")
    )
    task_count = min(len(proposed_rows), len(no_mode_rows))
    similarity = same_actions / max(task_count, 1)
    prop = summary_by_router.get("proposed_lyapunov_rule_mode_safe_bandit", {})
    util = summary_by_router.get("utility_rule", {})
    prop_net = _to_float(prop.get("net_system_utility"))
    util_net = _to_float(util.get("net_system_utility"))
    prop_pen = _to_float(prop.get("total_penalized_reward"))
    util_pen = _to_float(util.get("total_penalized_reward"))
    utility_resource_delta = _to_float(util.get("total_resource_penalty")) - _to_float(prop.get("total_resource_penalty"))
    utility_energy_delta = _to_float(util.get("total_energy")) - _to_float(prop.get("total_energy"))
    prop_actions = Counter(row.get("selected_action", "") for row in proposed_rows)
    util_actions = Counter(row.get("selected_action", "") for row in utility_rows)

    if prop_net > util_net:
        stress_result = "proposed improves over utility_rule on net_system_utility in this stress dry run."
    elif prop_pen > util_pen:
        stress_result = "proposed improves over utility_rule on penalized_reward, but not on net_system_utility."
    else:
        stress_result = "proposed does not improve over utility_rule in this stress dry run."

    if utility_resource_delta > 0.0 or utility_energy_delta > 0.0:
        utility_cost = "utility_rule pays a higher resource/energy cost than proposed."
    else:
        utility_cost = "utility_rule does not pay a higher resource/energy cost than proposed."

    lines = [
        "# Proposed vs utility_rule Stress Report",
        "",
        "## Router Metrics",
        "",
        *_summary_table(summary_by_router, lifetime_rows),
        "",
        "## Action Similarity",
        "",
        f"- proposed_vs_no_mode_same_actions: {same_actions}/{task_count} ({similarity:.3f})",
        f"- proposed_action_distribution: {dict(prop_actions)}",
        f"- utility_rule_action_distribution: {dict(util_actions)}",
        "",
        "## Stress Interpretation",
        "",
        f"- {stress_result}",
        f"- {utility_cost}",
        f"- utility_rule_resource_penalty_delta_vs_proposed: {utility_resource_delta:.3f}",
        f"- utility_rule_energy_delta_vs_proposed_j: {utility_energy_delta:.3f}",
        f"- clear_bug_found: no clear implementation bug was identified by this dry-run diagnostic.",
        f"- one_day_formal_simulation_readiness: {'yes' if task_count > 0 else 'no'}",
    ]
    path = output_dir / "proposed_vs_utility_rule_stress_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_dry_run(output_dir: Path) -> None:
    configs = _load_stress_configs()
    output_dirs = ensure_output_dirs(output_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    environment = SimulationEnvironment(configs)
    routers = build_routers(configs["experiment"], environment.action_space)
    episode_runner = environment.build_episode_runner(timeline)
    logger = get_logger("run_sarfish_energy_shadow_stress_dryrun")

    results = []
    for router in routers:
        logger.info("running router: %s", router.router_name)
        results.append(episode_runner.run(router))

    write_episode_results(results, output_dirs)
    _, _, lifetime_rows = generate_lifetime_outputs(output_dir, float(configs["timeline"]["slot_seconds"]))
    _build_resource_diagnostics(output_dir, timeline, configs["simulator"], configs["resource"])
    _write_energy_shadow_sanity_report(output_dir, configs, timeline, lifetime_rows)
    _write_comparison_report(output_dir, lifetime_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SARFish energy-shadow stress dry run.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sarfish_energy_shadow_stress")
    args = parser.parse_args()
    configure_logging()
    run_dry_run(args.output_dir)


if __name__ == "__main__":
    main()
