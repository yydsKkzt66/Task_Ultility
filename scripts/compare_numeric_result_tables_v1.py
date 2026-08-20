"""Compare two result CSVs with explicit floating-point tolerances."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(
    path: Path,
    keys: list[str],
) -> tuple[list[str], dict[tuple[str, ...], dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        missing_keys = [key for key in keys if key not in fields]
        if missing_keys:
            raise ValueError(f"Missing key columns in {path}: {missing_keys}")
        rows: dict[tuple[str, ...], dict[str, str]] = {}
        for row in reader:
            identity = tuple(row[key] for key in keys)
            if identity in rows:
                raise ValueError(f"Duplicate row key in {path}: {identity}")
            rows[identity] = row
    return fields, rows


def as_finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--key", action="append", required=True)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    args = parser.parse_args()

    ref_fields, ref_rows = read_rows(args.reference, args.key)
    new_fields, new_rows = read_rows(args.candidate, args.key)
    failures: list[dict[str, object]] = []
    max_absolute_difference = 0.0

    if ref_fields != new_fields:
        failures.append({"kind": "header_mismatch"})
    if set(ref_rows) != set(new_rows):
        failures.append({
            "kind": "row_key_mismatch",
            "missing": sorted(set(ref_rows) - set(new_rows)),
            "unexpected": sorted(set(new_rows) - set(ref_rows)),
        })

    for key in sorted(set(ref_rows) & set(new_rows)):
        for field in ref_fields:
            reference = ref_rows[key][field]
            candidate = new_rows[key][field]
            ref_number = as_finite_float(reference)
            new_number = as_finite_float(candidate)
            if ref_number is not None and new_number is not None:
                difference = abs(ref_number - new_number)
                max_absolute_difference = max(max_absolute_difference, difference)
                tolerance = args.atol + args.rtol * abs(ref_number)
                if difference > tolerance:
                    failures.append({
                        "kind": "numeric_mismatch",
                        "key": key,
                        "field": field,
                        "reference": ref_number,
                        "candidate": new_number,
                        "absolute_difference": difference,
                        "tolerance": tolerance,
                    })
            elif reference != candidate:
                failures.append({
                    "kind": "text_mismatch",
                    "key": key,
                    "field": field,
                    "reference": reference,
                    "candidate": candidate,
                })

    result = {
        "status": "PASS" if not failures else "FAIL",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "keys": args.key,
        "row_count": len(ref_rows),
        "atol": args.atol,
        "rtol": args.rtol,
        "max_absolute_difference": max_absolute_difference,
        "failure_count": len(failures),
        "failures": failures[:20],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
