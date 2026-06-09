"""CSV 报告写出。"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from src.common.types import EpisodeResult


def write_episode_results(results: list[EpisodeResult], output_dirs: dict[str, Path]) -> tuple[Path, list[Path]]:
    """写出 summary.csv 与 trajectory CSV。"""

    summary_path = output_dirs["root"] / "summary.csv"
    trajectory_paths: list[Path] = []
    summary_rows: list[dict] = []

    for result in results:
        summary_rows.append(result.summary)
        trajectory_path = output_dirs["trajectories"] / f"{result.router_name}.csv"
        rows = [asdict(record) for record in result.records]
        with trajectory_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        trajectory_paths.append(trajectory_path)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path, trajectory_paths
