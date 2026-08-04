"""Dry-run cleanup planner for experiment artifacts.

Default behavior only reports. With --apply, archive candidates are moved under
outputs/_archive/; files are never deleted.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
INDEX_DIR = OUTPUT_ROOT / "_index"
ARCHIVE_ROOT = OUTPUT_ROOT / "_archive"

PROTECTED_OUTPUT_DIRS = {
    "_index",
    "_archive",
    "sarfish_20s_dry_run",
    "sarfish_energy_shadow_stress",
    "sarfish_yolo_profile_ablation",
    "sarfish_policy_repair",
    "sarfish_payload_profile_calibration",
    "sarfish_grd_validation_audit",
    "sarfish_grd_validation_short_sim",
}

ARCHIVE_PREFIXES = (
    "debug_",
    "phase",
    "demo",
)

CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _archive_candidates() -> list[Path]:
    if not OUTPUT_ROOT.exists():
        return []
    candidates: list[Path] = []
    for path in sorted(item for item in OUTPUT_ROOT.iterdir() if item.is_dir()):
        if path.name in PROTECTED_OUTPUT_DIRS:
            continue
        if path.name.startswith(ARCHIVE_PREFIXES) or path.name in {"figures", "trajectories", "doc_comparison", "high_value_error_penalty", "sarfishsample_short_sim", "sarfishsample_audit", "sarfish_profile_calibration"}:
            candidates.append(path)
    return candidates


def _cache_candidates() -> list[Path]:
    candidates: list[Path] = []
    for root in [PROJECT_ROOT, OUTPUT_ROOT]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() and path.name in CACHE_DIR_NAMES:
                candidates.append(path)
    return sorted(set(candidates))


def _summary_groups() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in OUTPUT_ROOT.rglob("summary*.csv"):
        if any(part in {"_archive"} for part in path.parts):
            continue
        groups[path.name].append(path)
    return groups


def _large_trajectories(threshold_mb: float) -> list[Path]:
    threshold = threshold_mb * 1024 * 1024
    rows = []
    for path in OUTPUT_ROOT.rglob("trajectories/*.csv"):
        try:
            if path.stat().st_size >= threshold:
                rows.append(path)
        except OSError:
            pass
    return sorted(rows)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_plan(threshold_mb: float) -> str:
    archive_dirs = _archive_candidates()
    cache_dirs = _cache_candidates()
    summary_groups = _summary_groups()
    large_traj = _large_trajectories(threshold_mb)

    lines = [
        "# Cleanup Plan",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Default mode is dry-run. No files are deleted. With `--apply`, archive candidates are moved to `outputs/_archive/` only.",
        "",
        "## Protected Output Directories",
        "",
        *[f"- outputs/{name}" for name in sorted(PROTECTED_OUTPUT_DIRS)],
        "",
        "## Archive Candidates",
        "",
    ]
    if archive_dirs:
        for path in archive_dirs:
            lines.append(f"- {_relative(path)} ({_format_mb(_dir_size_bytes(path))})")
    else:
        lines.append("- none")

    lines.extend(["", "## Cache Candidates", ""])
    if cache_dirs:
        for path in cache_dirs:
            lines.append(f"- {_relative(path)}")
    else:
        lines.append("- none")

    lines.extend(["", "## Duplicate Summary Families", ""])
    for name, paths in sorted(summary_groups.items()):
        if len(paths) <= 1:
            continue
        lines.append(f"- {name}: {len(paths)} copies")
        for path in paths:
            lines.append(f"  - {_relative(path)}")

    lines.extend(["", f"## Large Trajectory Files (>= {threshold_mb:g} MB)", ""])
    if large_traj:
        for path in large_traj:
            lines.append(f"- {_relative(path)} ({_format_mb(path.stat().st_size)})")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Apply Semantics",
            "",
            "- `--apply` moves only archive candidate directories.",
            "- `--apply` does not move caches, configs, src, tests, docs, current stage outputs, or individual files.",
            "- Nothing is directly deleted.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_plan(threshold_mb: float) -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEX_DIR / "cleanup_plan.md"
    path.write_text(build_plan(threshold_mb), encoding="utf-8")
    return path


def apply_archive_moves() -> list[tuple[Path, Path]]:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    moves: list[tuple[Path, Path]] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_root = ARCHIVE_ROOT / stamp
    target_root.mkdir(parents=True, exist_ok=True)
    for source in _archive_candidates():
        destination = target_root / source.name
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")
        shutil.move(str(source), str(destination))
        moves.append((source, destination))
    return moves


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or apply experiment artifact cleanup.")
    parser.add_argument("--apply", action="store_true", help="Move archive candidates to outputs/_archive/ instead of dry-run only.")
    parser.add_argument("--large-threshold-mb", type=float, default=1.0, help="Large trajectory threshold in MB.")
    args = parser.parse_args()

    plan_path = write_plan(args.large_threshold_mb)
    print(f"Wrote cleanup plan: {_relative(plan_path)}")
    if not args.apply:
        print("Dry-run only; no files moved.")
        return 0

    moves = apply_archive_moves()
    print(f"Moved {len(moves)} directories to {_relative(ARCHIVE_ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
