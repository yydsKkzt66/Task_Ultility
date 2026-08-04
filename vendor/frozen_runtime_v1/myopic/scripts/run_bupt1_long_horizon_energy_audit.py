"""Run BUPT-1 inspired long-horizon energy-health audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.config import load_project_configs, load_yaml  # noqa: E402
from src.energy.soc_band_policy import classify_soc_band  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


DEFAULT_PROFILE = PROJECT_ROOT / "configs" / "resource_sarfish_bupt1_balanced_long_horizon.yaml"
DEFAULT_SIMULATOR = PROJECT_ROOT / "configs" / "simulator_bupt1_long_horizon_energy_health.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "_bupt1_long_horizon_nominal"
HEALTH_OUTPUT = PROJECT_ROOT / "outputs" / "_bupt1_energy_health_check"

ARM_ALIASES = {
    "proposed": "proposed_lyapunov_rule_mode_safe_bandit",
    "utility_rule": "utility_rule",
    "persistent_queue_without_guarantee": "no_lyapunov",
    "fixed_lightweight": "fixed_lightweight",
    "fixed_heavy": "fixed_high_accuracy",
    "ground_approved_followup": "comm_conservative",
}


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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fraction(predicate_count: int, total: int) -> float:
    return 0.0 if total <= 0 else predicate_count / total


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _action_flags(row: dict[str, object]) -> dict[str, int]:
    model = str(row.get("model_name", row.get("selected_model", ""))).lower()
    payload = str(row.get("payload_policy", row.get("selected_payload", ""))).lower()
    action = str(row.get("executed_action", row.get("selected_action", ""))).lower()
    return {
        "heavy": int(model in {"high_accuracy", "medium"}),
        "medium": int(model in {"high_accuracy", "medium"}),
        "tiny": int(model in {"lightweight", "tiny"}),
        "metadata": int(payload == "metadata" or payload.startswith("metadata_then")),
        "roi": int(payload == "roi" or payload == "metadata_then_roi"),
        "roi_context": int("roi_context" in payload),
        "defer": int(payload == "defer" or model == "delay_store" or action in {"delay/store", "defer"}),
    }


def _group_behavior(rows: list[dict[str, object]], arm: str | None = None) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["soc_band"])].append(row)
    output: list[dict[str, object]] = []
    for band in ["normal", "soft_pressure", "strong_conservation", "hard_safety"]:
        band_rows = grouped.get(band, [])
        count = len(band_rows)
        flags = [_action_flags(row) for row in band_rows]
        prefix = {"arm": arm} if arm is not None else {}
        output.append(
            {
                **prefix,
                "soc_band": band,
                "slot_count": count,
                "avg_SOC": sum(_to_float(row.get("battery_next_soc", row.get("soc"))) for row in band_rows)
                / max(count, 1),
                "avg_lambda_E": sum(_to_float(row.get("lambda_E")) for row in band_rows) / max(count, 1),
                "avg_lambda_Q": sum(_to_float(row.get("lambda_Q")) for row in band_rows) / max(count, 1),
                "avg_lambda_S": sum(_to_float(row.get("lambda_S")) for row in band_rows) / max(count, 1),
                "heavy_model_rate": sum(flag["heavy"] for flag in flags) / max(count, 1),
                "medium_model_rate": sum(flag["medium"] for flag in flags) / max(count, 1),
                "tiny_model_rate": sum(flag["tiny"] for flag in flags) / max(count, 1),
                "metadata_rate": sum(flag["metadata"] for flag in flags) / max(count, 1),
                "roi_rate": sum(flag["roi"] for flag in flags) / max(count, 1),
                "roi_context_rate": sum(flag["roi_context"] for flag in flags) / max(count, 1),
                "defer_rate": sum(flag["defer"] for flag in flags) / max(count, 1),
                "hard_safety_block_rate": sum(_to_int(row.get("final_resource_safety_violation")) for row in band_rows)
                / max(count, 1),
                "H_task_delivery_count": sum(
                    1
                    for row in band_rows
                    if _to_float(row.get("task_value")) >= 0.75 and str(row.get("delivery_status")) == "delivered"
                ),
                "H_net_utility": sum(
                    _to_float(row.get("net_system_reward_slot", row.get("mission_utility")))
                    for row in band_rows
                    if _to_float(row.get("task_value")) >= 0.75
                ),
            }
        )
    return output


def _energy_by_orbit(rows: list[dict[str, object]], arm: str) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_to_int(row.get("orbit_index"))].append(row)
    output: list[dict[str, object]] = []
    for orbit_index, orbit_rows in sorted(grouped.items()):
        output.append(
            {
                "arm": arm,
                "orbit_index": orbit_index,
                "slot_count": len(orbit_rows),
                "solar_harvest_Wh": sum(_to_float(row.get("solar_energy_j")) for row in orbit_rows) / 3600.0,
                "base_energy_Wh": sum(_to_float(row.get("idle_energy_j")) for row in orbit_rows) / 3600.0,
                "compute_energy_Wh": sum(_to_float(row.get("compute_energy_j")) for row in orbit_rows) / 3600.0,
                "comm_energy_Wh": sum(_to_float(row.get("comm_energy_j")) for row in orbit_rows) / 3600.0,
                "unused_solar_energy_Wh": sum(_to_float(row.get("wasted_solar_energy_j")) for row in orbit_rows)
                / 3600.0,
                "soc_start": _to_float(orbit_rows[0].get("battery_soc", orbit_rows[0].get("battery_next_soc"))),
                "soc_end": _to_float(orbit_rows[-1].get("battery_next_soc")),
            }
        )
    return output


def _add_soc_band(rows: list[dict[str, object]], cfg: dict[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        soc = _to_float(copied.get("battery_next_soc", copied.get("battery_soc")))
        copied["soc_band"] = classify_soc_band(
            soc,
            soft_target=float(cfg.get("battery_soft_target", 0.70)),
            strong_guard=float(cfg.get("battery_strong_guard", 0.50)),
            hard_min=float(cfg.get("battery_min", 0.20)),
        )
        out.append(copied)
    return out


def _summary_for_arm(
    arm: str,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    slot_seconds: float,
) -> dict[str, object]:
    soc_values = [_to_float(row.get("battery_next_soc", row.get("battery_soc"))) for row in rows]
    below_70 = [row for row in rows if _to_float(row.get("battery_next_soc", row.get("battery_soc"))) < 0.70]
    below_50 = [row for row in rows if _to_float(row.get("battery_next_soc", row.get("battery_soc"))) < 0.50]
    flags_below_70 = [_action_flags(row) for row in below_70]
    flags_below_50 = [_action_flags(row) for row in below_50]
    duration_days = max(len(rows) * slot_seconds / 86400.0, 1.0e-9)
    initial_soc = _to_float(rows[0].get("battery_soc", rows[0].get("battery_next_soc"))) if rows else 0.0
    final_soc = soc_values[-1] if soc_values else 0.0
    return {
        "arm": arm,
        "net_system_utility": _to_float(summary.get("net_system_utility", summary.get("total_penalized_reward"))),
        "total_net_delivered_utility": _to_float(summary.get("net_system_utility", summary.get("total_effective_utility"))),
        "total_resource_cost": _to_float(summary.get("total_energy", 0.0)),
        "H_gross_delivered_utility": _to_float(summary.get("high_value_task_utility", 0.0)),
        "H_net_delivered_utility": _to_float(summary.get("high_value_task_utility", 0.0))
        - _to_float(summary.get("total_long_term_risk_penalty", 0.0)),
        "H_min_stage_success_rate": _to_float(summary.get("high_value_delivery_ratio", 0.0)),
        "H_target_stage_success_rate": _to_float(summary.get("high_value_delivery_ratio", 0.0)),
        "min_SOC": min(soc_values) if soc_values else 0.0,
        "max_DoD": 1.0 - min(soc_values) if soc_values else 0.0,
        "final_SOC": final_soc,
        "daily_SOC_drift": (final_soc - initial_soc) / duration_days,
        "fraction_time_SOC_above_70": _fraction(sum(value >= 0.70 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_50_70": _fraction(sum(0.50 <= value < 0.70 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_20_50": _fraction(sum(0.20 <= value < 0.50 for value in soc_values), len(soc_values)),
        "fraction_time_SOC_below_20": _fraction(sum(value < 0.20 for value in soc_values), len(soc_values)),
        "heavy_action_rate_below_70": sum(flag["heavy"] for flag in flags_below_70) / max(len(flags_below_70), 1),
        "roi_context_rate_below_70": sum(flag["roi_context"] for flag in flags_below_70)
        / max(len(flags_below_70), 1),
        "metadata_rate_below_50": sum(flag["metadata"] for flag in flags_below_50) / max(len(flags_below_50), 1),
        "defer_rate_below_50": sum(flag["defer"] for flag in flags_below_50) / max(len(flags_below_50), 1),
        "hard_safety_violation_count": sum(_to_int(row.get("final_resource_safety_violation")) for row in rows),
    }


def _plot_soc_trajectory(rows: list[dict[str, object]], output: Path) -> None:
    plt.figure(figsize=(10, 4))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    for arm, arm_rows in grouped.items():
        plt.plot(
            [_to_float(row.get("time_s")) / 3600.0 for row in arm_rows],
            [_to_float(row.get("battery_next_soc", row.get("soc"))) for row in arm_rows],
            label=arm,
            linewidth=1.2,
        )
    for level in [0.70, 0.50, 0.20]:
        plt.axhline(level, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("time (h)")
    plt.ylabel("SOC")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _plot_proposed_sawtooth(rows: list[dict[str, object]], output: Path) -> None:
    proposed = [row for row in rows if row["arm"] == "proposed"]
    if not proposed:
        return
    plt.figure(figsize=(10, 4))
    times = [_to_float(row.get("time_s")) / 3600.0 for row in proposed]
    soc = [_to_float(row.get("battery_next_soc", row.get("soc"))) for row in proposed]
    plt.plot(times, soc, color="#1f77b4", linewidth=1.4)
    for row in proposed:
        if _to_int(row.get("is_eclipse")):
            x0 = _to_float(row.get("time_s")) / 3600.0
            plt.axvspan(x0, x0 + _to_float(row.get("slot_seconds"), 20.0) / 3600.0, color="#d9d9d9", alpha=0.12)
    for level in [0.70, 0.50, 0.20]:
        plt.axhline(level, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("time (h)")
    plt.ylabel("proposed SOC")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _plot_energy_balance(rows: list[dict[str, object]], output: Path) -> None:
    proposed = [row for row in rows if row["arm"] == "proposed"]
    if not proposed:
        return
    series = ["solar_harvest_Wh", "base_energy_Wh", "compute_energy_Wh", "comm_energy_Wh", "unused_solar_energy_Wh"]
    x = [row["orbit_index"] for row in proposed]
    plt.figure(figsize=(10, 4))
    for name in series:
        plt.plot(x, [_to_float(row.get(name)) for row in proposed], label=name)
    plt.xlabel("orbit")
    plt.ylabel("Wh")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _plot_action_distribution(rows: list[dict[str, object]], output: Path) -> None:
    proposed = [row for row in rows if row.get("arm") == "proposed"]
    bands = ["normal", "soft_pressure", "strong_conservation", "hard_safety"]
    metrics = ["metadata_rate", "roi_rate", "roi_context_rate", "defer_rate"]
    x = range(len(bands))
    plt.figure(figsize=(8, 4))
    bottom = [0.0] * len(bands)
    for metric in metrics:
        values = [_to_float(next((row for row in proposed if row["soc_band"] == band), {}).get(metric)) for band in bands]
        plt.bar(x, values, bottom=bottom, label=metric)
        bottom = [a + b for a, b in zip(bottom, values)]
    plt.xticks(list(x), bands, rotation=20)
    plt.ylabel("rate")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _plot_lyapunov(rows: list[dict[str, object]], output: Path) -> None:
    proposed = [row for row in rows if row.get("arm") == "proposed"]
    bands = ["normal", "soft_pressure", "strong_conservation", "hard_safety"]
    values = [_to_float(next((row for row in proposed if row["soc_band"] == band), {}).get("avg_lambda_E")) for band in bands]
    plt.figure(figsize=(8, 4))
    plt.bar(bands, values, color="#4c78a8")
    plt.ylabel("avg lambda_E")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _write_recommendation(
    output_dir: Path,
    summary_rows: list[dict[str, object]],
    proposed_behavior: list[dict[str, object]],
) -> None:
    by_arm = {str(row["arm"]): row for row in summary_rows}
    proposed = by_arm.get("proposed", {})
    fixed_heavy = by_arm.get("fixed_heavy", {})
    utility_rule = by_arm.get("utility_rule", {})
    best_net = max((_to_float(row.get("net_system_utility")) for row in summary_rows), default=0.0)
    proposed_net = _to_float(proposed.get("net_system_utility"))
    behavior_by_band = {str(row["soc_band"]): row for row in proposed_behavior if row.get("arm") == "proposed"}
    normal = behavior_by_band.get("normal", {})
    soft = behavior_by_band.get("soft_pressure", {})
    strong = behavior_by_band.get("strong_conservation", {})
    final_lines = [
        "# Final Energy Recommendation",
        "",
        "1. BUPT-1 inspired nominal no-task health should be read from `../_bupt1_energy_health_check/no_task_calibration_report.md`.",
        f"2. 24h main SOC sawtooth evidence is in `figures/proposed_soc_sawtooth.png`; min_SOC={_to_float(proposed.get('min_SOC')):.6f}.",
        f"3. proposed net_system_utility={proposed_net:.6f}; best net_system_utility={best_net:.6f}; near_best={proposed_net >= 0.95 * best_net if best_net else False}.",
        f"4. proposed conservative below 70: normal heavy={_to_float(normal.get('heavy_model_rate')):.6f}, soft heavy={_to_float(soft.get('heavy_model_rate')):.6f}.",
        f"5. proposed below 50: strong medium={_to_float(strong.get('medium_model_rate')):.6f}, strong ROI+Ctx={_to_float(strong.get('roi_context_rate')):.6f}.",
        f"6. SOC below 20 occurred: {_to_float(proposed.get('fraction_time_SOC_below_20')) > 0.0}.",
        f"7. fixed_heavy resource pressure: min_SOC={_to_float(fixed_heavy.get('min_SOC')):.6f}, total_resource_cost={_to_float(fixed_heavy.get('total_resource_cost')):.6f}.",
        f"8. utility_rule resource profile: min_SOC={_to_float(utility_rule.get('min_SOC')):.6f}, H_net_delivered_utility={_to_float(utility_rule.get('H_net_delivered_utility')):.6f}.",
        "9. Use these results as supplemental long-horizon energy-health evidence only after reviewing the generated hard-safety audit.",
        "10. `energy_shadow_stress` is marked and should remain a stress test, not the nominal main experiment.",
        "",
        "Alias note: requested arms not present in the legacy router factory are mapped in this audit script and recorded in `arm_alias_manifest.csv`.",
    ]
    text = "\n".join(final_lines) + "\n"
    (output_dir / "final_energy_recommendation.md").write_text(text, encoding="utf-8")
    (output_dir / "long_horizon_recommendation.md").write_text(text, encoding="utf-8")


def _write_git_file_lists(output_dir: Path, created_files: Iterable[Path], modified_files: Iterable[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "created_files.txt").write_text(
        "\n".join(path.as_posix() for path in created_files) + "\n",
        encoding="utf-8",
    )
    (output_dir / "modified_files.txt").write_text(
        "\n".join(path.as_posix() for path in modified_files) + "\n",
        encoding="utf-8",
    )


def run_long_horizon_audit(
    profile_path: Path = DEFAULT_PROFILE,
    simulator_path: Path = DEFAULT_SIMULATOR,
    output_dir: Path = DEFAULT_OUTPUT,
    num_slots: int = 4320,
    slot_seconds: float = 20.0,
) -> dict[str, object]:
    profile = load_yaml(profile_path)
    simulator_profile = load_yaml(simulator_path)
    configs = load_project_configs(PROJECT_ROOT)
    configs["timeline"] = {
        "use_num_days": False,
        "num_slots": num_slots,
        "slot_seconds": slot_seconds,
        "task_arrival": "one_per_slot",
        "trace_mode": "synthetic",
        "resource": profile,
    }
    configs["simulator"] = {**configs["simulator"], **simulator_profile}
    configs["simulator"]["slot_seconds"] = slot_seconds
    configs["simulator"]["battery_capacity_j"] = float(configs["simulator"]["battery_capacity_wh"]) * 3600.0
    configs["dataset"]["max_tasks"] = max(_to_int(configs["dataset"].get("max_tasks")), num_slots)
    configs["experiment"] = {"routers": list(dict.fromkeys(ARM_ALIASES.values()))}

    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    environment = SimulationEnvironment(configs)
    routers = build_routers(configs["experiment"], environment.action_space)
    router_to_arm = {actual: requested for requested, actual in ARM_ALIASES.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    task_manifest = [
        {
            "slot": step.slot_index,
            "task_id": "" if step.task is None else step.task.task_id,
            "scene_id": "" if step.task is None else step.task.scene_id,
            "arrival_slot": "" if step.task is None else step.task.arrival_slot,
            "deadline_s": "" if step.task is None else step.task.deadline_s,
        }
        for step in timeline
    ]
    resource_manifest = [
        {
            "slot": step.slot_index,
            "orbit_index": step.resource_trace.orbit_index,
            "is_sunlit": int(step.resource_trace.is_sunlit),
            "is_eclipse": int(step.resource_trace.is_eclipse),
            "solar_power_w": step.resource_trace.solar_power_w,
            "solar_energy_j": step.resource_trace.solar_energy_j,
            "base_power_w": step.resource_trace.base_power_w,
        }
        for step in timeline
    ]
    _write_csv(output_dir / "shared_task_manifest.csv", task_manifest)
    _write_csv(output_dir / "shared_resource_trace.csv", resource_manifest)
    task_hash = _file_sha256(output_dir / "shared_task_manifest.csv")
    resource_hash = _file_sha256(output_dir / "shared_resource_trace.csv")

    all_soc_rows: list[dict[str, object]] = []
    all_energy_rows: list[dict[str, object]] = []
    all_behavior_rows: list[dict[str, object]] = []
    all_action_rows: list[dict[str, object]] = []
    all_lyapunov_rows: list[dict[str, object]] = []
    hard_safety_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    alias_rows: list[dict[str, object]] = []

    runner = environment.build_episode_runner(timeline)
    for router in routers:
        actual_name = router.router_name
        arm = router_to_arm.get(actual_name, actual_name)
        alias_rows.append({"arm": arm, "router_name": actual_name})
        result = runner.run(router)
        rows = _add_soc_band([asdict(record) for record in result.records], configs["simulator"])
        for row in rows:
            row["arm"] = arm
            row["router_name"] = actual_name
        all_soc_rows.extend(rows)
        all_energy_rows.extend(_energy_by_orbit(rows, arm))
        behavior_rows = _group_behavior(rows, arm)
        all_behavior_rows.extend(behavior_rows)
        all_lyapunov_rows.extend(behavior_rows)
        summary_rows.append(_summary_for_arm(arm, rows, result.summary, slot_seconds))
        hard_safety_rows.append(
            {
                "arm": arm,
                "router_name": actual_name,
                "hard_safety_violation_count": sum(_to_int(row.get("final_resource_safety_violation")) for row in rows),
                "min_SOC": min((_to_float(row.get("battery_next_soc")) for row in rows), default=0.0),
                "battery_min": configs["simulator"]["battery_min"],
                "soc_below_hard_min_count": sum(
                    _to_float(row.get("battery_next_soc")) < float(configs["simulator"]["battery_min"]) for row in rows
                ),
            }
        )

        action_group_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        band_counts: Counter[str] = Counter()
        for row in rows:
            band = str(row["soc_band"])
            flags = _action_flags(row)
            band_counts[band] += 1
            for key, value in flags.items():
                action_group_counts[(band, key)][key] += value
        for band in ["normal", "soft_pressure", "strong_conservation", "hard_safety"]:
            count = band_counts[band]
            action_row = {"arm": arm, "soc_band": band, "slot_count": count}
            for key in ["metadata", "roi", "roi_context", "defer", "heavy", "medium", "tiny"]:
                action_row[f"{key}_rate"] = action_group_counts[(band, key)][key] / max(count, 1)
            all_action_rows.append(action_row)

    _write_csv(output_dir / "arm_alias_manifest.csv", alias_rows)
    _write_csv(
        output_dir / "arm_resource_task_manifest_audit.csv",
        [
            {
                "arm": row["arm"],
                "router_name": row["router_name"],
                "task_manifest_sha256": task_hash,
                "resource_trace_sha256": resource_hash,
            }
            for row in alias_rows
        ],
    )
    _write_csv(output_dir / "long_horizon_main_summary.csv", summary_rows)
    _write_csv(output_dir / "soc_trajectory_by_arm.csv", all_soc_rows)
    _write_csv(output_dir / "energy_balance_by_orbit.csv", all_energy_rows)
    _write_csv(output_dir / "soc_band_behavior_by_arm.csv", all_behavior_rows)
    _write_csv(output_dir / "action_distribution_by_soc_band.csv", all_action_rows)
    _write_csv(output_dir / "lyapunov_soc_response_table.csv", all_lyapunov_rows)
    HEALTH_OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(HEALTH_OUTPUT / "lyapunov_soc_response_table.csv", all_lyapunov_rows)
    _write_csv(output_dir / "hard_safety_audit.csv", hard_safety_rows)

    _plot_soc_trajectory(all_soc_rows, figure_dir / "soc_trajectory_by_arm.png")
    _plot_proposed_sawtooth(all_soc_rows, figure_dir / "proposed_soc_sawtooth.png")
    _plot_energy_balance(all_energy_rows, figure_dir / "energy_balance_by_orbit.png")
    _plot_action_distribution(all_action_rows, figure_dir / "action_distribution_by_soc_band.png")
    _plot_lyapunov(all_lyapunov_rows, figure_dir / "lyapunov_response_by_soc_band.png")
    _write_recommendation(output_dir, summary_rows, all_behavior_rows)

    created = [
        Path("configs/resource_sarfish_bupt1_balanced_long_horizon.yaml"),
        Path("configs/timeline_bupt1_nominal_24h_20s.yaml"),
        Path("configs/timeline_bupt1_nominal_48h_20s.yaml"),
        Path("configs/timeline_bupt1_energy_shadow_stress_24h_20s.yaml"),
        Path("configs/simulator_bupt1_long_horizon_energy_health.yaml"),
        Path("src/energy/soc_band_policy.py"),
        Path("scripts/run_no_task_energy_calibration.py"),
        Path("scripts/run_bupt1_long_horizon_energy_audit.py"),
        Path("tests/test_bupt1_energy_profile.py"),
        Path("tests/test_soc_band_policy.py"),
        Path("tests/test_no_task_energy_calibration.py"),
    ]
    modified = [
        Path("configs/resource_sarfish_energy_shadow_stress.yaml"),
        Path("src/simulator/step_runner.py"),
    ]
    generated_outputs = [
        _display_path(path)
        for root in (output_dir, HEALTH_OUTPUT)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    _write_git_file_lists(output_dir, [*created, *generated_outputs], modified)
    return {"summary_rows": summary_rows, "output_dir": output_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BUPT-1 long-horizon energy audit.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--simulator", type=Path, default=DEFAULT_SIMULATOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-slots", type=int, default=4320)
    parser.add_argument("--slot-seconds", type=float, default=20.0)
    args = parser.parse_args()
    run_long_horizon_audit(args.profile, args.simulator, args.output_dir, args.num_slots, args.slot_seconds)


if __name__ == "__main__":
    main()
