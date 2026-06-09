"""导出 mock 模型 profiling 表。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_project_configs  # noqa: E402
from src.models.model_registry import ModelRegistry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 mock profiling 表。")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "profiles.csv")
    args = parser.parse_args()

    configs = load_project_configs(PROJECT_ROOT)
    registry = ModelRegistry(configs["model_pool"])
    rows = registry.profiling_store.export_table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
