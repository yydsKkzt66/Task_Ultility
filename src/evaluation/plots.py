"""绘图模块。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _to_float_series(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def _save_line_plot(df_dict: dict[str, list[dict[str, str]]], y_column: str, title: str, save_path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    for router_name, rows in df_dict.items():
        plt.plot(_to_float_series(rows, "t"), _to_float_series(rows, y_column), label=router_name)
    plt.title(title)
    plt.xlabel("virtual slot")
    plt.ylabel(y_column)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def make_figures(summary_path: Path, trajectory_paths: list[Path], figure_dir: Path) -> None:
    """生成要求的图表。"""

    summary_rows = _read_csv_rows(summary_path)
    trajectory_rows = {path.stem: _read_csv_rows(path) for path in trajectory_paths}

    _save_line_plot(trajectory_rows, "observed_reward", "Mission Utility", figure_dir / "mission_utility.png")
    _save_line_plot(trajectory_rows, "queue_next_mb", "Queue Pressure", figure_dir / "queue_pressure.png")
    _save_line_plot(trajectory_rows, "B_t", "Battery State", figure_dir / "battery_state.png")

    plt.figure(figsize=(10, 4.5))
    for router_name, rows in trajectory_rows.items():
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["selected_action"]] = counts.get(row["selected_action"], 0) + 1
        actions = sorted(counts)
        values = [counts[action] for action in actions]
        plt.plot(actions, values, marker="o", label=router_name)
    plt.title("Action Distribution")
    plt.xticks(rotation=15)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "action_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(
        [row["router"] for row in summary_rows],
        [float(row["utility_per_joule"]) for row in summary_rows],
    )
    plt.title("Utility Per Joule")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figure_dir / "utility_per_joule.png", dpi=180)
    plt.close()
