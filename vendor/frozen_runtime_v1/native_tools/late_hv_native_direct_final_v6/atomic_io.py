"""Exclusive atomic JSON/CSV output helpers."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _commit(path: Path, temporary: str, exclusive: bool) -> None:
    if exclusive and path.exists():
        Path(temporary).unlink(missing_ok=True)
        raise FileExistsError(path)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".tmp", dir=path.parent, delete=False)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _commit(path, handle.name, exclusive)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def atomic_write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str], *, exclusive: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", suffix=".tmp", dir=path.parent, delete=False)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        _commit(path, handle.name, exclusive)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, *, exclusive: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".tmp", dir=path.parent, delete=False)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _commit(path, handle.name, exclusive)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
