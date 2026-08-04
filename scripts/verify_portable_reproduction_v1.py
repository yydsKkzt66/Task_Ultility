"""Verify the clean-clone reproduction bundle without running an experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "frozen_runtime_v1"
MANIFEST = ROOT / "artifacts" / "reproducibility" / "frozen_runtime_v1_sha256.json"
REFERENCE_PYTHON = "3.9.13"
REQUIRED_PACKAGES = {
    "numpy": "2.0.2",
    "matplotlib": "3.8.4",
    "contourpy": "1.3.0",
    "cycler": "0.12.1",
    "fonttools": "4.60.2",
    "kiwisolver": "1.4.7",
    "packaging": "26.1",
    "Pillow": "11.3.0",
    "pyparsing": "3.3.2",
    "python-dateutil": "2.9.0.post0",
    "six": "1.17.0",
    "PyYAML": "6.0",
    "pytest": "7.1.2",
    "scipy": "1.13.1",
}
CRITICAL_RELATIVE_PATHS = (
    "prop/scripts/run_updated_prop_ground_ack_v1.py",
    "prop/src/simulator/step_runner.py",
    "b1_offload/src/baselines/b1_enhanced.py",
    "b1_offload/src/baselines/offload_eft_thermal_aligned.py",
    "tut/scripts/run_tut_uoi_rate_lite_v2.py",
    "tut/src/routers/tut_uoi_rate_router.py",
    "myopic/src/routers/myopic_frozen_calibrated_router.py",
    "native_tools/late_hv_native_direct_final_v6/native_b1.py",
    "native_tools/late_hv_native_direct_final_v6/native_offload.py",
    "native_tools/late_hv_native_direct_final_v6/native_tut.py",
    "inputs/V25-MAIN-v1/bundle_contract.json",
    "inputs/myopic_frozen_calibration_v1.json",
    "inputs/solar_trace_one_orbit.csv",
    "inputs/selected_power_calibration.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_files() -> list[Path]:
    return sorted(
        path for path in VENDOR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    )


def build_manifest() -> dict[str, Any]:
    files = {
        path.relative_to(VENDOR).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in runtime_files()
    }
    tree_payload = "".join(f"{name}\0{entry['sha256']}\n" for name, entry in files.items())
    return {
        "schema_version": 1,
        "bundle": "frozen_runtime_v1",
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files.values()),
        "tree_sha256": hashlib.sha256(tree_payload.encode("utf-8")).hexdigest(),
        "files": files,
    }


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in REQUIRED_PACKAGES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "MISSING"
    return versions


def verify(strict_versions: bool) -> dict[str, Any]:
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = build_manifest()
    expected_files = expected["files"]
    actual_files = actual["files"]
    missing = sorted(set(expected_files) - set(actual_files))
    unexpected = sorted(set(actual_files) - set(expected_files))
    mismatched = sorted(
        name for name in set(expected_files) & set(actual_files)
        if expected_files[name]["sha256"] != actual_files[name]["sha256"]
    )
    versions = installed_versions()
    package_mismatches = {
        name: {"expected": expected_version, "actual": versions[name]}
        for name, expected_version in REQUIRED_PACKAGES.items()
        if versions[name] != expected_version
    }
    critical_missing = [name for name in CRITICAL_RELATIVE_PATHS if not (VENDOR / name).is_file()]
    checks = {
        "manifest_exists": MANIFEST.is_file(),
        "runtime_files_match": not missing and not unexpected and not mismatched,
        "tree_hash_match": expected["tree_sha256"] == actual["tree_sha256"],
        "critical_files_present": not critical_missing,
        "python_major_minor_match": sys.version_info[:2] == (3, 9),
        "package_versions_match": not package_mismatches,
        "final_main_entry_present": (ROOT / "scripts" / "run_current_prop_four_baselines_fiveseed_dual_thermal_v1.py").is_file(),
        "main_figure_entry_present": (ROOT / "scripts" / "main metrics.py").is_file(),
        "sensitivity_figure_entry_present": (ROOT / "scripts" / "sensitive.py").is_file(),
    }
    required_checks = dict(checks)
    if not strict_versions:
        required_checks.pop("python_major_minor_match")
        required_checks.pop("package_versions_match")
    return {
        "status": "PASS" if all(required_checks.values()) else "FAIL",
        "strict_versions": strict_versions,
        "checks": checks,
        "reference_python": REFERENCE_PYTHON,
        "actual_python": platform.python_version(),
        "package_versions": versions,
        "package_mismatches": package_mismatches,
        "missing_files": missing,
        "unexpected_files": unexpected,
        "mismatched_files": mismatched,
        "critical_missing": critical_missing,
        "tree_sha256": actual["tree_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--strict-versions", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        payload = build_manifest()
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"WROTE {MANIFEST} ({payload['file_count']} files, {payload['tree_sha256']})")
        return 0
    if not MANIFEST.is_file():
        print(f"FAIL: missing manifest {MANIFEST}", file=sys.stderr)
        return 2
    result = verify(args.strict_versions)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
