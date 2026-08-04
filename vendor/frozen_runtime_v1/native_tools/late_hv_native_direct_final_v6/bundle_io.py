"""Contract-first canonical neutral bundle verification and copying."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NeutralBundle:
    root: Path
    contract: dict[str, Any]
    tasks: tuple[dict[str, str], ...]
    slots: tuple[dict[str, str], ...]
    swaps: tuple[dict[str, str], ...]
    values: tuple[dict[str, str], ...]
    config: dict[str, Any]
    task_values: dict[str, Decimal]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return tuple(dict(row) for row in csv.DictReader(stream))


def _verify_source(source: Path, expected_contract_sha256: str) -> tuple[dict[str, Any], dict[str, str]]:
    contract_path = source / "bundle_contract.json"
    if _sha256_file(contract_path) != expected_contract_sha256:
        raise ValueError("canonical bundle contract SHA-256 mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    file_hashes = dict(contract.get("file_sha256", {}))
    if not file_hashes or any(Path(name).suffix.lower() in {".pkl", ".pickle"} for name in file_hashes):
        raise ValueError("contract payload list missing or contains pickle")
    for name, expected in file_hashes.items():
        path = source / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"canonical bundle payload SHA-256 mismatch: {name}")
    return contract, file_hashes


def verify_and_copy_bundle(source: Path, destination: Path, expected_contract_sha256: str) -> dict[str, Any]:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    contract, file_hashes = _verify_source(source, expected_contract_sha256)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copyfile(source / "bundle_contract.json", destination / "bundle_contract.json")
        for name in file_hashes:
            shutil.copyfile(source / name, destination / name)
        if _sha256_file(destination / "bundle_contract.json") != expected_contract_sha256:
            raise ValueError("copied contract SHA-256 mismatch")
        copied = {name: _sha256_file(destination / name) for name in file_hashes}
        if copied != file_hashes:
            raise ValueError("copied payload SHA-256 mismatch")
    except BaseException:
        # The caller treats a partial destination as a failed immutable attempt.
        raise
    return {"contract_sha256": expected_contract_sha256, "files": copied,
            "task_hash": contract["task_hash"], "environment_hash": contract["environment_hash"]}


def load_bundle(root: Path, expected_contract_sha256: str) -> NeutralBundle:
    root = Path(root).resolve()
    contract, _ = _verify_source(root, expected_contract_sha256)
    tasks = _read_csv(root / "task_manifest.csv")
    slots = _read_csv(root / "slot_trace.csv")
    swaps = _read_csv(root / "arrival_swap_manifest.csv")
    values = _read_csv(root / "task_value_assignment_trace.csv")
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    if len(tasks) != 268 or len(slots) != 4320 or len(swaps) != 27:
        raise ValueError("canonical bundle row counts mismatch")
    if len({row["task_id"] for row in tasks}) != 268:
        raise ValueError("duplicate canonical task IDs")
    levels = Counter(row["value_level"] for row in tasks)
    if levels != Counter({"low": 54, "medium": 139, "high": 54, "urgent": 21}):
        raise ValueError("canonical value-level counts mismatch")
    late = Counter(row["value_level"] for row in tasks if Decimal("43200") <= Decimal(row["arrival_time_s"]) < Decimal("64800"))
    if late["high"] != 30 or late["urgent"] != 15:
        raise ValueError("canonical late-window counts mismatch")
    task_values: dict[str, Decimal] = {}
    try:
        for row in tasks:
            value = Decimal(row["effective_task_value"])
            if not value.is_finite() or value <= 0:
                raise ValueError("nonpositive/nonfinite task value")
            task_values[row["task_id"]] = value
    except InvalidOperation as error:
        raise ValueError("invalid canonical Decimal token") from error
    return NeutralBundle(root, contract, tasks, slots, swaps, values, config, task_values)
