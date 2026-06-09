"""配置加载工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """读取单个 YAML 配置文件。"""

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def load_project_configs(project_root: Path) -> dict[str, dict[str, Any]]:
    """加载整个项目配置集合。

    返回值以文件名 stem 为 key，例如 `configs/reward.yaml` 会映射为 `reward`。
    """

    config_dir = project_root / "configs"
    return {path.stem: load_yaml(path) for path in sorted(config_dir.glob("*.yaml"))}


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    """创建脚本运行所需的输出目录。"""

    trajectories = output_root / "trajectories"
    figures = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    trajectories.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {"root": output_root, "trajectories": trajectories, "figures": figures}
