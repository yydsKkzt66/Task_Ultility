"""Byte-exact inventory checks for the frozen runtime source snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_BANNED_PARTS = {".git", "outputs", "__pycache__", "recovery_01", "recovery_02"}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inventory_snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files: list[dict[str, str]] = []
    banned: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        parts = set(relative.parts)
        if parts & _BANNED_PARTS:
            banned.add(relative.parts[0] if relative.parts else relative.as_posix())
        if path.is_file():
            if path.name == "scenario.pkl" or "late_hv_native_direct" in relative.as_posix():
                banned.add(relative.as_posix())
            files.append({
                "relative_path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    return {
        "root": root.as_posix(),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": _canonical_sha256(files),
        "banned_entries": sorted(banned),
    }
