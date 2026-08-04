"""Fail-closed environment construction for isolated native children."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping


ENV_ALLOWLIST = ("SystemRoot", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP")


def build_child_env(parent: Mapping[str, str], source_root: Path) -> dict[str, str]:
    result = {key: str(parent[key]) for key in ENV_ALLOWLIST if key in parent and parent[key]}
    result["PYTHONPATH"] = Path(source_root).resolve().as_posix()
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result
