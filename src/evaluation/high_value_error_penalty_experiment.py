"""高价值任务错判重罚实验。"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt

from src.common.config import ensure_output_dirs, load_project_configs
from src.context.comm_state import build_initial_comm_state
from src.context.mission_state import build_initial_mission_state
from src.context.resource_state import build_initial_resource_state
from src.evaluation.baselines import build_routers
from src.evaluation.metrics import summarize_episode
from src.link.downlink_queue import DownlinkQueue
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


@dataclass(frozen=True)
class HighValuePenaltyScenario:
    """单个高价值错判惩罚场景。"""

    name: str
    description: str
    multiplier: float
    enabled: bool
    reward_overrides: dict[str, float] = field(default_factory=dict)


DEFAULT_ERROR_MODEL_CFG = {
    "enabled": True,
    "base_miss_probability": 0.03,
    "difficulty_weight": 0.30,
    "uncertainty_weight": 0.22,
    "low_quality_weight": 0.55,
    "low_robustness_weight": 0.18,
    "high_value_bonus": 0.10,
    "high_value_threshold": 0.75,
    "false_alarm_scale": 0.35,
    "min_error_probability": 0.02,
    "max_error_probability": 0.95,
}


def build_penalty_scenarios() -> list[HighValuePenaltyScenario]:
    """构建 0x / 5x / 10x 三组独立实验场景。"""

    return [
        HighValuePenaltyScenario(
            name="error_model_only_0x",
            description="启用错判模型，但不额外对高价值任务错判做乘法重罚。",
            multiplier=0.0,
            enabled=False,
        ),
        HighValuePenaltyScenario(
            name="high_value_error_penalty_5x",
            description="对高价值任务错判启用 5x 惩罚。",
            multiplier=5.0,
            enabled=True,
        ),
        HighValuePenaltyScenario(
            name="high_value_error_penalty_10x",
            description="对高价值任务错判启用 10x 惩罚。",
            multiplier=10.0,
            enabled=True,
        ),
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _apply_scenario(
    base_configs: dict[str, dict],
    scenario: HighValuePenaltyScenario,
    num_slots: int | None,
) -> dict[str, dict]:
    """把实验场景覆盖到默认配置副本中。"""

    configs = copy.deepcopy(base_configs)
    configs["model_pool"]["error_model"] = copy.deepcopy(DEFAULT_ERROR_MODEL_CFG)
    reward_cfg = configs["reward"]
    reward_cfg.update(scenario.reward_overrides)
    reward_cfg["high_value_error_penalty_enabled"] = scenario.enabled
    reward_cfg["high_value_threshold"] = 0.75
    reward_cfg["high_value_error_penalty_multiplier"] = scenario.multiplier
    reward_cfg["high_value_error_penalty_base"] = 1.0
    reward_cfg["false_alarm_penalty_scale"] = 0.45
    reward_cfg["error_penalty_urgency_weight"] = 0.60
    reward_cfg["error_penalty_difficulty_weight"] = 0.70
    reward_cfg["error_penalty_uncertainty_weight"] = 0.70

    if num_slots is not None:
        configs["timeline"]["num_slots"] = int(num_slots)
        configs["dataset"]["max_tasks"] = max(int(configs["dataset"]["max_tasks"]), int(num_slots))
    return configs


def _run_single_router(environment: SimulationEnvironment, timeline, router) -> dict[str, object]:
    """运行单个 router，并采集默认 summary 之外的错判重罚统计。"""

    resource_state = build_initial_resource_state(environment.simulator_cfg)
    mission_state = build_initial_mission_state()
    comm_state = build_initial_comm_state(environment.simulator_cfg)
    queue = DownlinkQueue(
        queue_max_mb=float(environment.simulator_cfg["queue_max_mb"]),
        slot_seconds=float(environment.simulator_cfg["slot_seconds"]),
    )
    step_runner = environment.build_step_runner(queue)
    records = []
    miss_count = 0
    false_alarm_count = 0
    high_value_error_count = 0
    total_mission_error_penalty = 0.0
    high_value_threshold = float(environment.reward_cfg.get("high_value_threshold", 0.75))

    for timeline_step in timeline:
        record, _reward = step_runner.run_step(timeline_step, resource_state, mission_state, comm_state, router)
        records.append(record)
        feedback = step_runner.last_feedback
        breakdown = step_runner.last_reward_breakdown
        miss_count += int(feedback.miss_signal >= 1.0)
        false_alarm_count += int(feedback.false_alarm_signal >= 1.0)
        if timeline_step.task.task_value >= high_value_threshold and (feedback.miss_signal >= 1.0 or feedback.false_alarm_signal >= 1.0):
            high_value_error_count += 1
        total_mission_error_penalty += breakdown.mission_error_penalty

    summary = summarize_episode(
        router_name=router.router_name,
        records=records,
        queue_max_mb=float(environment.simulator_cfg["queue_max_mb"]),
        final_battery=resource_state.battery_soc,
        max_thermal=max([record.Theta_t for record in records] + [resource_state.thermal_pressure]),
    )
    summary.update(
        {
            "miss_count": miss_count,
            "false_alarm_count": false_alarm_count,
            "high_value_error_count": high_value_error_count,
            "total_mission_error_penalty": total_mission_error_penalty,
        }
    )
    return summary


def _write_report(path: Path, rows: list[dict[str, object]]) -> None:
    """写出易读文本摘要。"""

    scenario_names = []
    lines = ["High-Value Error Penalty Experiment Report", ""]
    for scenario_name in sorted({str(row["scenario_name"]) for row in rows}):
        scenario_names.append(scenario_name)
        lines.append(f"Scenario: {scenario_name}")
        ranked = sorted(
            [row for row in rows if str(row["scenario_name"]) == scenario_name],
            key=lambda item: float(item["total_mission_utility"]),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            lines.append(
                f"- rank={rank}, router={row['router']}, total_mission_utility={float(row['total_mission_utility']):.3f}, "
                f"high_value_error_count={row['high_value_error_count']}, total_mission_error_penalty={float(row['total_mission_error_penalty']):.3f}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figure(output_dirs: dict[str, Path], rows: list[dict[str, object]]) -> None:
    """生成一个简单的排名对比图。"""

    scenario_names = sorted({str(row["scenario_name"]) for row in rows})
    router_names = sorted({str(row["router"]) for row in rows})
    plt.figure(figsize=(10, 4.8))
    for router_name in router_names:
        y_values = []
        for scenario_name in scenario_names:
            matched = next(
                row for row in rows
                if str(row["scenario_name"]) == scenario_name and str(row["router"]) == router_name
            )
            y_values.append(float(matched["total_mission_utility"]))
        plt.plot(scenario_names, y_values, marker="o", label=router_name)
    plt.title("Router Ranking Under High-Value Error Penalty")
    plt.ylabel("total_mission_utility")
    plt.xticks(rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dirs["figures"] / "high_value_error_penalty_ranking.png", dpi=180)
    plt.close()


def run_high_value_error_penalty_experiment(
    project_root: Path,
    output_dir: Path,
    num_slots: int | None = None,
    scenarios: list[HighValuePenaltyScenario] | None = None,
) -> dict[str, Path]:
    """运行独立的高价值错判重罚实验。"""

    base_configs = load_project_configs(project_root)
    scenarios = scenarios or build_penalty_scenarios()

    timeline_configs = copy.deepcopy(base_configs)
    if num_slots is not None:
        timeline_configs["timeline"]["num_slots"] = int(num_slots)
        timeline_configs["dataset"]["max_tasks"] = max(int(timeline_configs["dataset"]["max_tasks"]), int(num_slots))
    timeline = TimelineBuilder(timeline_configs["dataset"], timeline_configs["timeline"]).build()

    output_dirs = ensure_output_dirs(output_dir)
    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for scenario in scenarios:
        scenario_configs = _apply_scenario(base_configs, scenario, num_slots)
        environment = SimulationEnvironment(scenario_configs)
        routers = build_routers(scenario_configs["experiment"], environment.action_space)
        manifest_rows.append(
            {
                "scenario_name": scenario.name,
                "description": scenario.description,
                "high_value_threshold": scenario_configs["reward"]["high_value_threshold"],
                "multiplier": scenario.multiplier,
                "penalty_enabled": scenario.enabled,
                "error_model_cfg": DEFAULT_ERROR_MODEL_CFG,
            }
        )
        for router in routers:
            row = _run_single_router(environment, timeline, router)
            row.update(
                {
                    "scenario_name": scenario.name,
                    "scenario_description": scenario.description,
                    "high_value_multiplier": scenario.multiplier,
                    "penalty_enabled": scenario.enabled,
                }
            )
            summary_rows.append(row)

    ranking_rows = []
    for scenario_name in sorted({str(row["scenario_name"]) for row in summary_rows}):
        ranked = sorted(
            [row for row in summary_rows if str(row["scenario_name"]) == scenario_name],
            key=lambda item: float(item["total_mission_utility"]),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            ranking_rows.append(
                {
                    "scenario_name": scenario_name,
                    "rank": rank,
                    "router": row["router"],
                    "total_mission_utility": row["total_mission_utility"],
                    "high_value_error_count": row["high_value_error_count"],
                    "total_mission_error_penalty": row["total_mission_error_penalty"],
                    "deadline_satisfaction_rate": row["deadline_satisfaction_rate"],
                    "safety_violations": row["safety_violations"],
                }
            )

    manifest_path = output_dirs["root"] / "scenario_manifest.csv"
    summary_path = output_dirs["root"] / "router_summary.csv"
    ranking_path = output_dirs["root"] / "router_rankings.csv"
    report_path = output_dirs["root"] / "experiment_report.txt"
    _write_csv(manifest_path, manifest_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(ranking_path, ranking_rows)
    _write_report(report_path, summary_rows)
    _make_figure(output_dirs, summary_rows)
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "rankings": ranking_path,
        "report": report_path,
        "figure": output_dirs["figures"] / "high_value_error_penalty_ranking.png",
    }
