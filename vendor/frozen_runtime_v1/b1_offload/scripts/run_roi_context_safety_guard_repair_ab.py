"""Small A/B wrapper for the ROI+Ctx hard-safety guard repair."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_value_skew_progressive_followup_ab as base_ab  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_roi_context_safety_guard_repair_ab"
BEFORE_DIR = PROJECT_ROOT / "outputs" / "_value_skew_progressive_followup_ab"
AFTER_RUN_DIR = OUT_DIR / "after_fix_runs"
SCENARIOS = ["natural", "targeted_resource_scarcity"]
SEED_DEFAULT = "2026,2027,2028,2029,2030,2031,2032,2033,2034,2035"
AFTER_ARMS = [
    "legacy_proposed",
    "soft_moderate_value_skew_plus_gate",
    "soft_moderate_value_skew_plus_gate_plus_progressive_followup",
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else "")).strip()


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _before_summary_rows(seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        scenario_dir = base_ab.SCENARIO_DIR.get(scenario, scenario)
        for seed in seeds:
            run_dir = BEFORE_DIR / "runs" / scenario_dir / f"s{seed}" / "skew_gate_prog"
            summary_rows = _read_csv(run_dir / "summary.csv")
            if not summary_rows:
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "arm": "soft_moderate_value_skew_plus_gate_plus_progressive_followup_before_fix",
                        "source": "missing_existing_before_fix_result",
                    }
                )
                continue
            summary = summary_rows[0]
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "arm": "soft_moderate_value_skew_plus_gate_plus_progressive_followup_before_fix",
                    "source": str(run_dir.relative_to(PROJECT_ROOT)),
                    "hard_safety_violation_count": _to_float(
                        summary.get("final_resource_safety_violation_count", summary.get("safety_violation_count"))
                    ),
                    "net_system_utility": _to_float(summary.get("net_system_utility")),
                    "expired_items": _to_float(summary.get("expired_items")),
                    "queue_next_mb_max": _to_float(summary.get("max_queue_mb", summary.get("queue_next_mb_max"))),
                    "metadata_to_roi_context_success_rate": _to_float(summary.get("metadata_to_roi_context_success_rate")),
                    "context_evidence_closure_proxy": _to_float(summary.get("context_evidence_followup_count")),
                }
            )
    return rows


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "hard_safety_violation_count",
        "net_system_utility",
        "expired_items",
        "queue_next_mb_max",
        "metadata_to_roi_context_success_rate",
        "high_value_roi_context_delivered_count",
        "context_closure_count",
        "evidence_closure_count",
        "context_evidence_closure_proxy",
    ]
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("scenario", "")), str(row.get("arm", "")))].append(row)
    out: list[dict[str, object]] = []
    for (scenario, arm), group_rows in sorted(groups.items()):
        item: dict[str, object] = {"scenario": scenario, "arm": arm, "n": len(group_rows)}
        for metric in metrics:
            values = [_to_float(row.get(metric)) for row in group_rows if str(row.get("source", "")) != "missing_existing_before_fix_result"]
            item[f"{metric}_mean"] = mean(values) if values else 0.0
        out.append(item)
    return out


def _write_configs_used() -> None:
    cfg_dir = OUT_DIR / "configs_used"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(AFTER_RUN_DIR.rglob("resolved_config.yaml"))[:6]:
        target = cfg_dir / f"resolved_config_{copied}.yaml"
        shutil.copyfile(path, target)
        copied += 1
    _write_text(cfg_dir / "README.md", f"Copied {copied} representative resolved after-fix configs.")


def _final_case(aggregate_rows: list[dict[str, object]]) -> str:
    after = [
        row
        for row in aggregate_rows
        if str(row.get("arm")) == "soft_moderate_value_skew_plus_gate_plus_progressive_followup_after_fix"
    ]
    if any(_to_float(row.get("hard_safety_violation_count_mean")) > 0.0 for row in after):
        return "D"
    delivered = sum(_to_float(row.get("metadata_to_roi_context_success_rate_mean")) for row in after)
    closure = sum(
        _to_float(row.get("context_closure_count_mean")) + _to_float(row.get("evidence_closure_count_mean"))
        for row in after
    )
    if delivered <= 0.0:
        return "C"
    if closure <= 0.0:
        return "B"
    return "A"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=SEED_DEFAULT)
    parser.add_argument("--num-slots", type=int, default=80)
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = f"python scripts/run_roi_context_safety_guard_repair_ab.py --seeds {args.seeds} --num-slots {args.num_slots}"

    base_ab.OUT_DIR = AFTER_RUN_DIR
    base_ab.ZIP_PATH = OUT_DIR / "after_fix_runs.zip"
    after_runs: list[dict[str, Any]] = []
    after_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for seed in seeds:
            for arm in AFTER_ARMS:
                run = base_ab._run_one(seed, scenario, arm, args.num_slots)
                after_runs.append(run)
                row = base_ab._summarize_run(run)
                if arm == "soft_moderate_value_skew_plus_gate_plus_progressive_followup":
                    row["arm"] = "soft_moderate_value_skew_plus_gate_plus_progressive_followup_after_fix"
                after_rows.append(row)

    before_rows = _before_summary_rows(seeds)
    all_rows = before_rows + after_rows
    aggregate_rows = _aggregate(all_rows)
    case = _final_case(aggregate_rows)

    _write_text(
        OUT_DIR / "README.md",
        "# ROI+Ctx Safety Guard Repair A/B\n\nSmall A/B using the existing value-skew progressive runner. Before-fix rows are read from existing `_value_skew_progressive_followup_ab` outputs when present; after-fix rows are rerun with the repaired guard.",
    )
    _write_text(OUT_DIR / "command_log.md", f"# Command Log\n\n```powershell\n{command}\n```\n")
    _write_text(OUT_DIR / "git_status.txt", _git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _git(
            [
                "diff",
                "--",
                "src/simulator/step_runner.py",
                "src/simulator/episode_runner.py",
                "src/payload/continue_request.py",
                "scripts/run_value_skew_progressive_followup_ab.py",
                "scripts/run_roi_context_safety_guard_repair_ab.py",
                "tests/test_followup_hard_safety_guard.py",
                "tests/test_roi_roi_context_utility.py",
            ]
        ),
    )
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "num_slots": args.num_slots,
                "scenarios": SCENARIOS,
                "after_fix_arms": AFTER_ARMS,
                "before_fix_source": str(BEFORE_DIR.relative_to(PROJECT_ROOT)),
                "final_recommendation_case": case,
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_configs_used()
    _write_csv(OUT_DIR / "before_after_safety_summary.csv", aggregate_rows)
    _write_csv(OUT_DIR / "before_after_progressive_summary.csv", all_rows)
    _write_csv(
        OUT_DIR / "hard_safety_violation_reason_summary.csv",
        [
            {
                "reason": "battery_or_thermal_or_queue_hard_violation",
                "after_fix_total_mean": sum(
                    _to_float(row.get("hard_safety_violation_count_mean"))
                    for row in aggregate_rows
                    if str(row.get("arm")).endswith("after_fix")
                ),
            }
        ],
    )
    _write_csv(
        OUT_DIR / "roi_context_delivery_summary.csv",
        [
            row
            for row in aggregate_rows
            if "roi_context" in ";".join(row.keys()) or "progressive_followup" in str(row.get("arm"))
        ],
    )
    _write_csv(
        OUT_DIR / "context_evidence_closure_summary.csv",
        [
            {
                "scenario": row.get("scenario"),
                "arm": row.get("arm"),
                "context_closure_count_mean": row.get("context_closure_count_mean", 0.0),
                "evidence_closure_count_mean": row.get("evidence_closure_count_mean", 0.0),
                "context_evidence_closure_proxy_mean": row.get("context_evidence_closure_proxy_mean", 0.0),
            }
            for row in aggregate_rows
        ],
    )
    _write_csv(
        OUT_DIR / "queue_expired_safety_guardrail_summary.csv",
        [
            {
                "scenario": row.get("scenario"),
                "arm": row.get("arm"),
                "queue_next_mb_max_mean": row.get("queue_next_mb_max_mean", 0.0),
                "expired_items_mean": row.get("expired_items_mean", 0.0),
                "hard_safety_violation_count_mean": row.get("hard_safety_violation_count_mean", 0.0),
            }
            for row in aggregate_rows
        ],
    )
    _write_csv(
        OUT_DIR / "utility_delta_summary.csv",
        [
            {
                "scenario": row.get("scenario"),
                "arm": row.get("arm"),
                "net_system_utility_mean": row.get("net_system_utility_mean", 0.0),
            }
            for row in aggregate_rows
        ],
    )
    _write_text(
        OUT_DIR / "final_recommendation.md",
        f"# Final Recommendation\n\nRecommendation case: {case}.\n\n"
        "The after-fix progressive arm was rerun with the same scenarios and requested seeds. Case A/B/C/D is assigned from after-fix hard-safety violations, ROI+Ctx delivery, and context/evidence closure proxies.\n",
    )
    files = sorted(path.relative_to(OUT_DIR).as_posix() for path in OUT_DIR.rglob("*") if path.is_file())
    _write_text(OUT_DIR / "archive_manifest.txt", "\n".join(files + ["archive_manifest.txt"]))
    print(f"repair A/B output directory: {OUT_DIR}")
    print(f"final recommendation case: {case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
