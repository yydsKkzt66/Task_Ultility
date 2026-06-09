"""小型参数敏感性实验工具。

这个模块的目标不是做大规模超参搜索，而是给 case study 提供一个
“小而清晰”的稳定性检查：

1. 扫几组 reward lambda 组合；
2. 扫几组 model profile 组合；
3. 看不同 router 的相对排序是否稳定；
4. 重点观察 `fixed_lightweight` 和 `safe_linucb` 的胜负关系是否依赖单一参数集。
"""

from __future__ import annotations

import copy
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt

from src.common.config import ensure_output_dirs, load_project_configs
from src.evaluation.baselines import build_routers
from src.models.profiling_store import ProfilingStore
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


@dataclass(frozen=True)
class RewardScenario:
    """一组 reward 权重覆盖。"""

    name: str
    description: str
    overrides: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileScenario:
    """一组模型 profile 覆盖。"""

    name: str
    description: str
    overrides: dict[str, dict[str, float]] = field(default_factory=dict)


def build_default_reward_scenarios(base_reward_cfg: dict[str, float]) -> list[RewardScenario]:
    """构建默认 reward 敏感性场景。

    这三组场景分别代表：
    1. 原始配置；
    2. 更保守、更看重资源与安全；
    3. 更偏任务收益、更愿意为高价值任务支付代价。
    """

    return [
        RewardScenario(
            name="baseline_reward",
            description="直接使用 reward.yaml 中的默认权重。",
            overrides={},
        ),
        RewardScenario(
            name="resource_guarded_reward",
            description="提高能耗、时延、队列和安全惩罚，模拟资源更紧张的卫星策略。",
            overrides={
                "lambda_E_base": round(float(base_reward_cfg["lambda_E_base"]) * 1.40, 4),
                "lambda_L": round(float(base_reward_cfg["lambda_L"]) + 0.05, 4),
                "lambda_Q": round(float(base_reward_cfg["lambda_Q"]) + 0.03, 4),
                "lambda_Omega": round(float(base_reward_cfg["lambda_Omega"]) + 0.40, 4),
                "eta_B": round(float(base_reward_cfg["eta_B"]) + 0.30, 4),
                "eta_Theta": round(float(base_reward_cfg["eta_Theta"]) + 0.20, 4),
            },
        ),
        RewardScenario(
            name="mission_priority_reward",
            description="降低能耗与时延惩罚，模拟更重视高价值任务收益的策略。",
            overrides={
                "lambda_E_base": round(float(base_reward_cfg["lambda_E_base"]) * 0.70, 4),
                "lambda_L": round(max(0.05, float(base_reward_cfg["lambda_L"]) - 0.05), 4),
                "lambda_Q": round(max(0.03, float(base_reward_cfg["lambda_Q"]) - 0.03), 4),
                "lambda_Omega": round(max(1.20, float(base_reward_cfg["lambda_Omega"]) - 0.30), 4),
                "eta_B": round(float(base_reward_cfg["eta_B"]), 4),
                "eta_Theta": round(float(base_reward_cfg["eta_Theta"]), 4),
            },
        ),
    ]


def build_default_profile_scenarios() -> list[ProfileScenario]:
    """构建默认模型 profile 敏感性场景。"""

    return [
        ProfileScenario(
            name="baseline_profiles",
            description="使用当前 scaffold 中的 tiny/small/medium mock profile。",
            overrides={},
        ),
        ProfileScenario(
            name="heavy_friendly_profiles",
            description="提高重模型质量和鲁棒性，并适度降低其代价，模拟重模型更值得调用的条件。",
            overrides={
                "tiny": {"quality_base": 0.60, "robustness": 0.44},
                "small": {"latency_s": 1.45, "quality_base": 0.78, "estimated_power_w": 12.0, "robustness": 0.74},
                "medium": {"latency_s": 2.20, "quality_base": 0.92, "estimated_power_w": 15.5, "robustness": 0.94},
            },
        ),
        ProfileScenario(
            name="heavy_expensive_profiles",
            description="保留重模型质量优势，但进一步增加其时延和功耗，模拟不利硬件条件。",
            overrides={
                "tiny": {"quality_base": 0.58, "robustness": 0.50},
                "small": {"latency_s": 1.85, "quality_base": 0.75, "estimated_power_w": 14.5, "robustness": 0.69},
                "medium": {"latency_s": 3.40, "quality_base": 0.90, "estimated_power_w": 23.5, "robustness": 0.84},
            },
        ),
    ]


def _scenario_name(reward_scenario: RewardScenario, profile_scenario: ProfileScenario) -> str:
    return f"{reward_scenario.name}__{profile_scenario.name}"


def _merge_configs(
    base_configs: dict[str, dict],
    reward_scenario: RewardScenario,
    profile_scenario: ProfileScenario,
    num_slots: int | None,
) -> dict[str, dict]:
    """基于默认配置生成单个敏感性场景的配置副本。"""

    configs = copy.deepcopy(base_configs)
    configs["reward"].update(reward_scenario.overrides)
    configs["model_pool"]["profile_overrides"] = copy.deepcopy(profile_scenario.overrides)
    if num_slots is not None:
        configs["timeline"]["num_slots"] = int(num_slots)
        configs["dataset"]["max_tasks"] = max(int(configs["dataset"]["max_tasks"]), int(num_slots))
    return configs


def _build_manifest_row(
    reward_scenario: RewardScenario,
    profile_scenario: ProfileScenario,
    scenario_configs: dict[str, dict],
) -> dict[str, object]:
    """生成单个场景的参数清单。"""

    profile_rows = ProfilingStore(profile_scenario.overrides).export_table()
    profile_by_name = {row["profile_name"]: row for row in profile_rows}
    reward_cfg = scenario_configs["reward"]
    row: dict[str, object] = {
        "scenario_name": _scenario_name(reward_scenario, profile_scenario),
        "reward_scenario": reward_scenario.name,
        "profile_scenario": profile_scenario.name,
        "reward_description": reward_scenario.description,
        "profile_description": profile_scenario.description,
        "reward_overrides_json": json.dumps(reward_scenario.overrides, ensure_ascii=False, sort_keys=True),
        "profile_overrides_json": json.dumps(profile_scenario.overrides, ensure_ascii=False, sort_keys=True),
        "lambda_E_base": reward_cfg["lambda_E_base"],
        "lambda_L": reward_cfg["lambda_L"],
        "lambda_Q": reward_cfg["lambda_Q"],
        "lambda_Omega": reward_cfg["lambda_Omega"],
        "eta_B": reward_cfg["eta_B"],
        "eta_Theta": reward_cfg["eta_Theta"],
    }
    for profile_name in ("tiny", "small", "medium"):
        profile = profile_by_name[profile_name]
        row[f"{profile_name}_latency_s"] = profile["latency_s"]
        row[f"{profile_name}_quality_base"] = profile["quality_base"]
        row[f"{profile_name}_estimated_power_w"] = profile["estimated_power_w"]
        row[f"{profile_name}_robustness"] = profile["robustness"]
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """写 CSV；若没有行，仍输出一个空文件头占位。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_rankings(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """按场景构造 router 排名表。"""

    rows_by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_scenario[str(row["scenario_name"])].append(row)

    ranking_rows: list[dict[str, object]] = []
    for scenario_name, rows in rows_by_scenario.items():
        ranked = sorted(rows, key=lambda item: float(item["total_mission_utility"]), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            ranking_rows.append(
                {
                    "scenario_name": scenario_name,
                    "reward_scenario": row["reward_scenario"],
                    "profile_scenario": row["profile_scenario"],
                    "rank": rank,
                    "router": row["router"],
                    "total_mission_utility": row["total_mission_utility"],
                    "deadline_satisfaction_rate": row["deadline_satisfaction_rate"],
                    "safety_violations": row["safety_violations"],
                }
            )
    return ranking_rows


def _write_stability_report(path: Path, summary_rows: list[dict[str, object]], ranking_rows: list[dict[str, object]]) -> None:
    """写出人类可读的稳定性摘要。"""

    scenario_count = len({row["scenario_name"] for row in summary_rows})
    winner_counts: dict[str, int] = defaultdict(int)
    rank_values: dict[str, list[int]] = defaultdict(list)
    utility_values: dict[str, list[float]] = defaultdict(list)
    rows_by_scenario_router: dict[tuple[str, str], dict[str, object]] = {}

    for row in summary_rows:
        rows_by_scenario_router[(str(row["scenario_name"]), str(row["router"]))] = row
        utility_values[str(row["router"])].append(float(row["total_mission_utility"]))

    for row in ranking_rows:
        router_name = str(row["router"])
        rank_values[router_name].append(int(row["rank"]))
        if int(row["rank"]) == 1:
            winner_counts[router_name] += 1

    safe_vs_lightweight_deltas: list[tuple[str, float]] = []
    for scenario_name in {row["scenario_name"] for row in summary_rows}:
        light_row = rows_by_scenario_router.get((str(scenario_name), "fixed_lightweight"))
        safe_row = rows_by_scenario_router.get((str(scenario_name), "safe_linucb"))
        if light_row is None or safe_row is None:
            continue
        delta = float(safe_row["total_mission_utility"]) - float(light_row["total_mission_utility"])
        safe_vs_lightweight_deltas.append((str(scenario_name), delta))

    lines = [
        "Sensitivity Experiment Stability Report",
        f"total_scenarios: {scenario_count}",
        "",
        "1. Winner frequency by router",
    ]
    for router_name, wins in sorted(winner_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {router_name}: {wins}/{scenario_count}")

    lines.append("")
    lines.append("2. Average rank and average mission utility")
    for router_name in sorted(rank_values):
        avg_rank = sum(rank_values[router_name]) / max(len(rank_values[router_name]), 1)
        avg_utility = sum(utility_values[router_name]) / max(len(utility_values[router_name]), 1)
        lines.append(f"- {router_name}: avg_rank={avg_rank:.2f}, avg_total_utility={avg_utility:.3f}")

    lines.append("")
    lines.append("3. SafeLinUCB minus fixed_lightweight mission-utility delta")
    if safe_vs_lightweight_deltas:
        positive_count = sum(1 for _name, delta in safe_vs_lightweight_deltas if delta > 0.0)
        mean_delta = sum(delta for _name, delta in safe_vs_lightweight_deltas) / len(safe_vs_lightweight_deltas)
        min_name, min_delta = min(safe_vs_lightweight_deltas, key=lambda item: item[1])
        max_name, max_delta = max(safe_vs_lightweight_deltas, key=lambda item: item[1])
        lines.append(f"- positive_scenarios: {positive_count}/{len(safe_vs_lightweight_deltas)}")
        lines.append(f"- mean_delta: {mean_delta:.3f}")
        lines.append(f"- worst_delta: {min_delta:.3f} ({min_name})")
        lines.append(f"- best_delta: {max_delta:.3f} ({max_name})")
    else:
        lines.append("- 对应 router 不完整，未计算该对比。")

    path.write_text("\n".join(lines), encoding="utf-8")


def _make_sensitivity_figures(output_dirs: dict[str, Path], summary_rows: list[dict[str, object]], ranking_rows: list[dict[str, object]]) -> None:
    """生成敏感性实验的简要图表。"""

    figure_dir = output_dirs["figures"]
    figure_dir.mkdir(parents=True, exist_ok=True)

    scenario_names = sorted({str(row["scenario_name"]) for row in summary_rows})
    router_names = sorted({str(row["router"]) for row in summary_rows})

    plt.figure(figsize=(12, 5))
    for router_name in router_names:
        y_values = []
        for scenario_name in scenario_names:
            matched = next(
                row for row in summary_rows
                if str(row["scenario_name"]) == scenario_name and str(row["router"]) == router_name
            )
            y_values.append(float(matched["total_mission_utility"]))
        plt.plot(scenario_names, y_values, marker="o", label=router_name)
    plt.title("Total Mission Utility Across Sensitivity Scenarios")
    plt.ylabel("total_mission_utility")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "router_utility_by_scenario.png", dpi=180)
    plt.close()

    winner_counts: dict[str, int] = defaultdict(int)
    for row in ranking_rows:
        if int(row["rank"]) == 1:
            winner_counts[str(row["router"])] += 1
    plt.figure(figsize=(8, 4.5))
    ordered_names = sorted(router_names)
    plt.bar(ordered_names, [winner_counts.get(name, 0) for name in ordered_names])
    plt.title("Winner Frequency Across Scenarios")
    plt.ylabel("scenario wins")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "winner_frequency.png", dpi=180)
    plt.close()


def run_sensitivity_experiment(
    project_root: Path,
    output_dir: Path,
    num_slots: int | None = None,
    reward_scenarios: list[RewardScenario] | None = None,
    profile_scenarios: list[ProfileScenario] | None = None,
) -> dict[str, Path]:
    """执行敏感性实验并导出结果。"""

    base_configs = load_project_configs(project_root)
    reward_scenarios = reward_scenarios or build_default_reward_scenarios(base_configs["reward"])
    profile_scenarios = profile_scenarios or build_default_profile_scenarios()

    timeline_configs = copy.deepcopy(base_configs)
    if num_slots is not None:
        timeline_configs["timeline"]["num_slots"] = int(num_slots)
        timeline_configs["dataset"]["max_tasks"] = max(int(timeline_configs["dataset"]["max_tasks"]), int(num_slots))
    timeline = TimelineBuilder(timeline_configs["dataset"], timeline_configs["timeline"]).build()

    output_dirs = ensure_output_dirs(output_dir)
    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for reward_scenario in reward_scenarios:
        for profile_scenario in profile_scenarios:
            scenario_configs = _merge_configs(base_configs, reward_scenario, profile_scenario, num_slots)
            scenario_name = _scenario_name(reward_scenario, profile_scenario)
            manifest_rows.append(_build_manifest_row(reward_scenario, profile_scenario, scenario_configs))

            environment = SimulationEnvironment(scenario_configs)
            routers = build_routers(scenario_configs["experiment"], environment.action_space)
            episode_runner = environment.build_episode_runner(timeline)

            for router in routers:
                result = episode_runner.run(router)
                row = dict(result.summary)
                row.update(
                    {
                        "scenario_name": scenario_name,
                        "reward_scenario": reward_scenario.name,
                        "profile_scenario": profile_scenario.name,
                    }
                )
                summary_rows.append(row)

    ranking_rows = _build_rankings(summary_rows)
    manifest_path = output_dirs["root"] / "scenario_manifest.csv"
    summary_path = output_dirs["root"] / "router_summary.csv"
    ranking_path = output_dirs["root"] / "router_rankings.csv"
    report_path = output_dirs["root"] / "stability_report.txt"

    _write_csv(manifest_path, manifest_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(ranking_path, ranking_rows)
    _write_stability_report(report_path, summary_rows, ranking_rows)
    _make_sensitivity_figures(output_dirs, summary_rows, ranking_rows)

    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "rankings": ranking_path,
        "report": report_path,
        "figures_dir": output_dirs["figures"],
    }
