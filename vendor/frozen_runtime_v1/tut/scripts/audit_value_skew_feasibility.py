"""Audit feasibility of value-skew calibration without changing simulator defaults."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_product_requirement_aware_module2_ab import DEFAULT_SEEDS, _load_run_configs  # noqa: E402
from src.data.mission_prior_value import compute_mission_prior_value_v2  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_value_skew_feasibility_audit"
ZIP_PATH = PROJECT_ROOT / "outputs" / "_value_skew_feasibility_audit.zip"
OLD_THRESHOLD = 0.15

SCALES: dict[str, dict[str, float]] = {
    "mild": {"low": 1.0, "medium": 2.0, "high": 3.0, "urgent": 5.0},
    "moderate": {"low": 1.0, "medium": 3.0, "high": 8.0, "urgent": 15.0},
    "stress": {"low": 1.0, "medium": 5.0, "high": 20.0, "urgent": 50.0},
}
ALL_SCALES = ["old", "mild", "moderate", "stress"]
BASELINES = {
    "proposed_requirement_aware": {
        "source": "product_requirement_ab",
        "arm": "v21_reqaware_ug_wx050",
        "router": "proposed_lyapunov_rule_mode_safe_bandit",
    },
    "utility_rule": {
        "source": "product_requirement_ab",
        "arm": "legacy_utility_rule",
        "router": "utility_rule",
    },
    "no_mode": {"source": "root", "path": "outputs/trajectories/no_mode.csv"},
    "no_lyapunov": {"source": "root", "path": "outputs/trajectories/no_lyapunov.csv"},
    "fixed_lightweight": {"source": "root", "path": "outputs/trajectories/fixed_lightweight.csv"},
    "fixed_high_accuracy": {"source": "root", "path": "outputs/trajectories/fixed_high_accuracy.csv"},
    "safe_linucb": {"source": "root", "path": "outputs/trajectories/safe_linucb.csv"},
}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


def _git(args: list[str]) -> str:
    try:
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
    except Exception as exc:  # pragma: no cover - audit robustness
        return f"git {' '.join(args)} failed: {exc}"


def _add_file_patch(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [
        f"diff --git a/{rel} b/{rel}",
        "new file mode 100644",
        "index 0000000..0000000",
        "--- /dev/null",
        f"+++ b/{rel}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    return "\n".join(header + ["+" + line for line in lines]) + "\n"


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _gini(values: list[float]) -> float:
    clean = sorted(max(0.0, value) for value in values)
    if not clean or sum(clean) <= 0.0:
        return 0.0
    n = len(clean)
    weighted = sum((idx + 1) * value for idx, value in enumerate(clean))
    return (2.0 * weighted) / (n * sum(clean)) - (n + 1.0) / n


def _top_share(values: list[float], fraction: float = 0.10) -> float:
    clean = sorted((max(0.0, value) for value in values), reverse=True)
    if not clean or sum(clean) <= 0.0:
        return 0.0
    k = max(1, int(math.ceil(len(clean) * fraction)))
    return sum(clean[:k]) / sum(clean)


def _stable_key(seed: int, task_id: str) -> str:
    return f"{seed}:{task_id}"


def _base_f(base_prior: float) -> float:
    return 0.8 + 0.4 * max(0.0, min(1.0, base_prior))


def _load_tasks(seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        configs = _load_run_configs(seed, "v21_reqaware_ug_wx050")
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        for step in timeline:
            task = step.task
            if task is None:
                continue
            metadata = getattr(task, "metadata", {}) or {}
            prior_result = compute_mission_prior_value_v2(
                task_id=task.task_id,
                scene_id=task.scene_id,
                task_class=getattr(task, "task_class", "routine"),
                product_requirement=getattr(task, "product_requirement", "verification"),
                deadline_s=float(task.deadline_s),
                container_cfg=configs["task_stream"],
                source_priority=None,
            )
            base_prior = _to_float(metadata.get("new_task_value"), prior_result.task_value)
            source_priority = _to_float(metadata.get("h_source_priority"), _to_float(metadata.get("source_priority"), 0.0))
            region_prior = metadata.get("area_prior_bucket", prior_result.area_prior_bucket)
            rows.append(
                {
                    "seed": seed,
                    "task_id": task.task_id,
                    "scene_id": task.scene_id,
                    "slot": int(step.slot_index),
                    "task_class": getattr(task, "task_class", "routine"),
                    "product_requirement": getattr(task, "product_requirement", "verification"),
                    "high_value_flag": int(bool(getattr(task, "high_value_flag", False))),
                    "emergency_flag": int(bool(getattr(task, "emergency_flag", False))),
                    "task_source_priority": source_priority,
                    "region_prior": region_prior,
                    "base_prior_score": base_prior,
                    "old_task_value": float(task.task_value),
                    "uncertainty": float(task.uncertainty),
                    "deadline_s": float(task.deadline_s),
                    "v21_missing_fields": ";".join(str(x) for x in metadata.get("v21_missing_fields", [])),
                }
            )
    return rows


def _assign_levels(task_rows: list[dict[str, object]]) -> None:
    base_values = [_to_float(row["base_prior_score"]) for row in task_rows]
    q33 = _percentile(base_values, 0.33)
    q50 = _percentile(base_values, 0.50)
    q75 = _percentile(base_values, 0.75)
    for row in task_rows:
        task_class = str(row.get("task_class", "")).lower()
        requirement = str(row.get("product_requirement", "")).lower()
        base = _to_float(row.get("base_prior_score"))
        high = bool(_to_int(row.get("high_value_flag")))
        emergency = bool(_to_int(row.get("emergency_flag"))) or task_class == "emergency"
        if emergency:
            level = "urgent"
            reason = "emergency_flag_or_class"
        elif high or requirement in {"context", "evidence"} or base >= q75:
            level = "high"
            reason = "high_value_or_context_evidence_or_top_quartile_base_prior"
        elif task_class in {"routine", "background"} and base <= q33:
            level = "low"
            reason = "routine_low_base_prior"
        elif requirement == "verification" or base >= q50:
            level = "medium"
            reason = "verification_or_mid_base_prior"
        else:
            level = "medium"
            reason = "conservative_default_medium_due_to_limited_fields"
        row["assigned_value_level"] = level
        row["assignment_reason"] = reason


def _calibrate_values(task_rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    old_mean = mean([_to_float(row["old_task_value"]) for row in task_rows]) if task_rows else 1.0
    raw_by_scale: dict[str, dict[str, float]] = {}
    for scale, multipliers in SCALES.items():
        raw_by_scale[scale] = {
            _stable_key(_to_int(row["seed"]), str(row["task_id"])): multipliers[str(row["assigned_value_level"])] * _base_f(_to_float(row["base_prior_score"]))
            for row in task_rows
        }
    values_by_scale: dict[str, dict[str, float]] = {
        "old": {
            _stable_key(_to_int(row["seed"]), str(row["task_id"])): _to_float(row["old_task_value"]) for row in task_rows
        }
    }
    for scale, raw_map in raw_by_scale.items():
        raw_mean = mean(raw_map.values()) if raw_map else 1.0
        alpha = old_mean / max(raw_mean, 1.0e-12)
        values_by_scale[scale] = {key: alpha * value for key, value in raw_map.items()}
        max_raw = max(raw_map.values()) if raw_map else 1.0
        values_by_scale[f"{scale}_maxnorm"] = {key: value / max(max_raw, 1.0e-12) for key, value in raw_map.items()}
    for row in task_rows:
        key = _stable_key(_to_int(row["seed"]), str(row["task_id"]))
        for scale in SCALES:
            row[f"{scale}_multiplier"] = SCALES[scale][str(row["assigned_value_level"])]
            row[f"{scale}_calibrated_value"] = values_by_scale[scale][key]
            row[f"{scale}_maxnorm_value"] = values_by_scale[f"{scale}_maxnorm"][key]
    return values_by_scale


def _base_prior_distribution(task_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_name, group_rows in [("all", task_rows)] + sorted(
        ((level, [row for row in task_rows if row["assigned_value_level"] == level]) for level in {"low", "medium", "high", "urgent"}),
        key=lambda item: item[0],
    ):
        values = [_to_float(row["base_prior_score"]) for row in group_rows]
        old_values = [_to_float(row["old_task_value"]) for row in group_rows]
        rows.append(
            {
                "group": group_name,
                "n": len(group_rows),
                "base_prior_mean": mean(values) if values else 0.0,
                "base_prior_min": min(values) if values else 0.0,
                "base_prior_max": max(values) if values else 0.0,
                "old_task_value_mean": mean(old_values) if old_values else 0.0,
                "old_task_value_min": min(old_values) if old_values else 0.0,
                "old_task_value_max": max(old_values) if old_values else 0.0,
            }
        )
    return rows


def _distribution_summary(task_rows: list[dict[str, object]], values_by_scale: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    level_by_key = {_stable_key(_to_int(row["seed"]), str(row["task_id"])): str(row["assigned_value_level"]) for row in task_rows}
    for scale in ALL_SCALES:
        value_map = values_by_scale[scale]
        values = list(value_map.values())
        low_values = [value for key, value in value_map.items() if level_by_key[key] == "low"]
        high_values = [value for key, value in value_map.items() if level_by_key[key] == "high"]
        urgent_values = [value for key, value in value_map.items() if level_by_key[key] == "urgent"]
        high_urgent_values = [value for key, value in value_map.items() if level_by_key[key] in {"high", "urgent"}]
        low_count = sum(1 for level in level_by_key.values() if level == "low")
        rows.append(
            {
                "value_scale": scale,
                "normalization": "current" if scale == "old" else "mean_preserving",
                "n": len(values),
                "value_min": min(values) if values else 0.0,
                "value_p25": _percentile(values, 0.25),
                "value_mean": mean(values) if values else 0.0,
                "value_median": median(values) if values else 0.0,
                "value_p75": _percentile(values, 0.75),
                "value_p90": _percentile(values, 0.90),
                "value_p95": _percentile(values, 0.95),
                "value_max": max(values) if values else 0.0,
                "high_low_mean_ratio": (mean(high_values) / max(mean(low_values), 1.0e-12)) if high_values and low_values else 0.0,
                "urgent_low_mean_ratio": (mean(urgent_values) / max(mean(low_values), 1.0e-12)) if urgent_values and low_values else 0.0,
                "gini_coefficient": _gini(values),
                "top_10pct_tasks_share_total_value": _top_share(values, 0.10),
                "low_value_tasks_share_total_count": low_count / max(len(values), 1),
                "high_urgent_tasks_share_total_value": sum(high_urgent_values) / max(sum(values), 1.0e-12),
            }
        )
    for scale in ["mild", "moderate", "stress"]:
        value_map = values_by_scale[f"{scale}_maxnorm"]
        values = list(value_map.values())
        rows.append(
            {
                "value_scale": scale,
                "normalization": "max",
                "n": len(values),
                "value_min": min(values) if values else 0.0,
                "value_p25": _percentile(values, 0.25),
                "value_mean": mean(values) if values else 0.0,
                "value_median": median(values) if values else 0.0,
                "value_p75": _percentile(values, 0.75),
                "value_p90": _percentile(values, 0.90),
                "value_p95": _percentile(values, 0.95),
                "value_max": max(values) if values else 0.0,
                "high_low_mean_ratio": 0.0,
                "urgent_low_mean_ratio": 0.0,
                "gini_coefficient": _gini(values),
                "top_10pct_tasks_share_total_value": _top_share(values, 0.10),
                "low_value_tasks_share_total_count": 0.0,
                "high_urgent_tasks_share_total_value": 0.0,
            }
        )
    return rows


def _value_trace(task_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "seed",
        "task_id",
        "scene_id",
        "slot",
        "task_class",
        "product_requirement",
        "high_value_flag",
        "emergency_flag",
        "task_source_priority",
        "region_prior",
        "base_prior_score",
        "old_task_value",
        "assigned_value_level",
        "assignment_reason",
        "mild_multiplier",
        "moderate_multiplier",
        "stress_multiplier",
        "mild_calibrated_value",
        "moderate_calibrated_value",
        "stress_calibrated_value",
    ]
    return [{field: row.get(field, "") for field in fields} for row in task_rows]


def _candidate_rows(task_rows: list[dict[str, object]], values_by_scale: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    task_by_key = {_stable_key(_to_int(row["seed"]), str(row["task_id"])): row for row in task_rows}
    rows = []
    seen: set[tuple[str, str, str, str]] = set()
    paths = [
        PROJECT_ROOT / "outputs" / "_phase6_residual_enqueue_fallback_validation" / "same_slot_carryover_trace.csv",
        PROJECT_ROOT / "outputs" / "_phase6_residual_enqueue_fallback_validation" / "per_candidate_followup_trace.csv",
    ]
    for path in paths:
        for row in _read_csv(path):
            event_type = str(row.get("event_type", ""))
            if event_type and event_type not in {"same_slot_candidate_re_evaluated", "ordinary_queue_fallback_enqueued"}:
                continue
            eriv = _to_float(row.get("ERIV"))
            if eriv <= 0.0:
                continue
            seed = _to_int(row.get("seed"))
            task_id = str(row.get("task_id", ""))
            key = _stable_key(seed, task_id)
            task = task_by_key.get(key)
            if task is None:
                continue
            identity = (str(path.name), str(row.get("arm", "")), str(row.get("slot", "")), key)
            if identity in seen:
                continue
            seen.add(identity)
            old_value = max(_to_float(task.get("old_task_value")), 1.0e-12)
            item: dict[str, object] = {
                "seed": seed,
                "task_id": task_id,
                "slot": row.get("slot", ""),
                "arm": row.get("arm", ""),
                "product_requirement": task.get("product_requirement", ""),
                "assigned_value_level": task.get("assigned_value_level", ""),
                "high_value_flag": task.get("high_value_flag", 0),
                "residual_product_type": row.get("residual_product_type", ""),
                "old_eriv": eriv,
                "payload_size_proxy_mb": row.get("residual_payload_mb", row.get("queue_size_after_fallback_mb", "")),
            }
            for scale in ALL_SCALES:
                calibrated = values_by_scale[scale].get(key, old_value)
                item[f"{scale}_eriv"] = eriv * calibrated / old_value
            rows.append(item)
    return rows


def _eriv_summary(candidate_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    distribution: list[dict[str, object]] = []
    suggestions: list[dict[str, object]] = []
    for scale in ALL_SCALES:
        values = [_to_float(row.get(f"{scale}_eriv")) for row in candidate_rows]
        old_values = [_to_float(row.get("old_eriv")) for row in candidate_rows]
        old_count = sum(1 for value in old_values if value >= OLD_THRESHOLD)
        above = sum(1 for value in values if value >= OLD_THRESHOLD)
        sorted_values = sorted(values, reverse=True)
        match_threshold = sorted_values[old_count - 1] if old_count and len(sorted_values) >= old_count else 0.0
        high_values = [
            _to_float(row.get(f"{scale}_eriv"))
            for row in candidate_rows
            if _to_int(row.get("high_value_flag")) or str(row.get("assigned_value_level")) in {"high", "urgent"}
        ]
        context_values = [
            _to_float(row.get(f"{scale}_eriv"))
            for row in candidate_rows
            if str(row.get("product_requirement")) in {"context", "evidence"}
        ]
        distribution.append(
            {
                "value_scale": scale,
                "candidate_count": len(values),
                "eriv_min": min(values) if values else 0.0,
                "eriv_mean": mean(values) if values else 0.0,
                "eriv_median": median(values) if values else 0.0,
                "eriv_p75": _percentile(values, 0.75),
                "eriv_p90": _percentile(values, 0.90),
                "eriv_p95": _percentile(values, 0.95),
                "eriv_max": max(values) if values else 0.0,
                "candidates_above_old_threshold_0_15": above,
                "suggested_threshold_to_match_old_candidate_count": match_threshold,
                "suggested_threshold_to_allow_top_k_high_value_candidates": _percentile(high_values, 0.50) if high_values else match_threshold,
                "suggested_threshold_for_selective_roi_context": _percentile(context_values, 0.75) if context_values else match_threshold,
                "risk_of_over_triggering": "high" if above > max(old_count * 2, old_count + 3) else "medium" if above > old_count else "low",
                "risk_of_under_triggering": "high" if above < max(1, old_count // 2) else "low",
            }
        )
        suggestions.append(
            {
                "value_scale": scale,
                "old_threshold": OLD_THRESHOLD,
                "old_candidate_count": old_count,
                "candidate_count_at_old_threshold": above,
                "threshold_match_old_count": match_threshold,
                "threshold_selective_high_value": _percentile(high_values, 0.50) if high_values else match_threshold,
                "threshold_selective_roi_context": _percentile(context_values, 0.75) if context_values else match_threshold,
                "recommended_recalibration_direction": (
                    "raise_threshold" if above > old_count else "keep_or_slightly_lower" if above < old_count else "keep_near_old"
                ),
            }
        )
    return distribution, suggestions


def _roi_context_pressure(
    task_rows: list[dict[str, object]], candidate_rows: list[dict[str, object]], eriv_suggestions: list[dict[str, object]]
) -> list[dict[str, object]]:
    suggestions = {row["value_scale"]: row for row in eriv_suggestions}
    rows: list[dict[str, object]] = []
    context_tasks = [row for row in task_rows if str(row["product_requirement"]) in {"context", "evidence"}]
    hv_context_tasks = [row for row in context_tasks if _to_int(row["high_value_flag"]) or row["assigned_value_level"] in {"high", "urgent"}]
    urgent_context_tasks = [row for row in context_tasks if row["assigned_value_level"] == "urgent"]
    roi_only = [row for row in candidate_rows if str(row.get("residual_product_type")) == "roi"]
    for scale in ALL_SCALES:
        threshold = _to_float(suggestions.get(scale, {}).get("threshold_match_old_count"), OLD_THRESHOLD)
        eligible = [
            row
            for row in candidate_rows
            if str(row.get("product_requirement")) in {"context", "evidence"}
            and str(row.get("assigned_value_level")) in {"high", "urgent"}
        ]
        at_old = [row for row in eligible if _to_float(row.get(f"{scale}_eriv")) >= OLD_THRESHOLD]
        at_recal = [row for row in eligible if _to_float(row.get(f"{scale}_eriv")) >= threshold]
        low_accidental = [
            row
            for row in candidate_rows
            if str(row.get("assigned_value_level")) == "low"
            and str(row.get("product_requirement")) in {"context", "evidence"}
            and _to_float(row.get(f"{scale}_eriv")) >= threshold
        ]
        rows.append(
            {
                "value_scale": scale,
                "context_evidence_task_count": len(context_tasks),
                "high_value_context_evidence_task_count": len(hv_context_tasks),
                "urgent_context_evidence_task_count": len(urgent_context_tasks),
                "roi_only_residual_candidate_count": len(roi_only),
                "roi_context_eligible_candidate_count": len(eligible),
                "expected_roi_context_upgrade_candidate_count_under_old_threshold": len(at_old),
                "expected_roi_context_upgrade_candidate_count_under_recalibrated_threshold": len(at_recal),
                "high_value_roi_context_candidate_count": len([row for row in at_recal if _to_int(row.get("high_value_flag"))]),
                "low_value_roi_context_accidental_trigger_count": len(low_accidental),
                "accidental_low_value_upgrade_risk": len(low_accidental) / max(len(candidate_rows), 1),
                "queue_pressure_proxy_if_all_candidates_upgraded": len(eligible) * 1.5,
                "queue_pressure_proxy_if_only_selective_candidates_upgraded": len(at_recal) * 1.5,
                "semantic_gate_needed": int(True),
                "value_density_gate_needed": int(scale in {"moderate", "stress"}),
            }
        )
    return rows


def _residual_followup_summary(candidate_rows: list[dict[str, object]], suggestions: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_scale_threshold = {row["value_scale"]: _to_float(row["threshold_match_old_count"], OLD_THRESHOLD) for row in suggestions}
    for scale in ALL_SCALES:
        threshold = by_scale_threshold.get(scale, OLD_THRESHOLD)
        selected = [row for row in candidate_rows if _to_float(row.get(f"{scale}_eriv")) >= threshold]
        out.append(
            {
                "value_scale": scale,
                "candidate_count": len(candidate_rows),
                "candidate_count_above_recalibrated_threshold": len(selected),
                "high_or_urgent_candidate_count": sum(int(str(row.get("assigned_value_level")) in {"high", "urgent"}) for row in selected),
                "context_evidence_candidate_count": sum(int(str(row.get("product_requirement")) in {"context", "evidence"}) for row in selected),
                "roi_only_candidate_count": sum(int(str(row.get("residual_product_type")) == "roi") for row in selected),
                "mean_eriv_selected": mean([_to_float(row.get(f"{scale}_eriv")) for row in selected]) if selected else 0.0,
            }
        )
    return out


def _trajectory_paths_for_baseline(name: str) -> list[Path]:
    spec = BASELINES[name]
    if spec["source"] == "root":
        path = PROJECT_ROOT / str(spec["path"])
        return [path] if path.exists() else []
    paths: list[Path] = []
    base = PROJECT_ROOT / "outputs" / "_product_requirement_aware_module2_ab" / "runs"
    arm = str(spec["arm"])
    router = str(spec["router"])
    for seed_dir in sorted(base.glob("seed_*")):
        path = seed_dir / arm / "trajectories" / f"{router}.csv"
        if path.exists():
            paths.append(path)
    return paths


def _row_level_from_traj(row: dict[str, str]) -> str:
    if _to_int(row.get("emergency_flag")):
        return "urgent"
    if _to_int(row.get("high_value_flag")) or str(row.get("product_requirement")) in {"context", "evidence"}:
        return "high"
    value = _to_float(row.get("V_t"), _to_float(row.get("task_value")))
    if value <= 0.45:
        return "low"
    return "medium"


def _row_scale_value(row: dict[str, str], scale: str, old_mean: float) -> float:
    old_value = _to_float(row.get("V_t"), _to_float(row.get("task_value"), 0.0))
    if scale == "old":
        return old_value
    level = _row_level_from_traj(row)
    raw = SCALES[scale][level] * _base_f(old_value)
    return raw * old_mean / max(SCALES[scale]["medium"] * _base_f(old_mean), 1.0e-12)


def _baseline_fairness() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for baseline in BASELINES:
        paths = _trajectory_paths_for_baseline(baseline)
        traj_rows = [row for path in paths for row in _read_csv(path) if _to_int(row.get("has_task"), 1)]
        old_mean = mean([_to_float(row.get("V_t"), _to_float(row.get("task_value"), 0.5)) for row in traj_rows]) if traj_rows else 0.5
        for scale in ALL_SCALES:
            delivered_rescaled = 0.0
            old_delivered = 0.0
            high_delivered = 0.0
            low_delivered = 0.0
            completion = 0
            low_bad = 0
            queue_max = 0.0
            for row in traj_rows:
                old_v = max(_to_float(row.get("V_t"), _to_float(row.get("task_value"), old_mean)), 1.0e-12)
                new_v = _row_scale_value(row, scale, old_mean)
                ratio = new_v / old_v
                delivered = _to_float(row.get("delivered_only_incremental_utility"), _to_float(row.get("delivered_before_deadline_utility")))
                old_delivered += delivered
                value = delivered * ratio
                delivered_rescaled += value
                level = _row_level_from_traj(row)
                if level in {"high", "urgent"}:
                    high_delivered += value
                    completion += int(str(row.get("delivery_status", "")).lower() == "delivered")
                elif level == "low":
                    low_delivered += value
                    low_bad += int(str(row.get("delivery_status", "")).lower() in {"deferred", "expired"} or _to_int(row.get("expired_flag")))
                queue_max = max(queue_max, _to_float(row.get("queue_next_mb"), _to_float(row.get("queue_mb"))))
            resource_cost = sum(_to_float(row.get("resource_cost"), _to_float(row.get("resource_penalty"))) for row in traj_rows)
            penalty = sum(
                _to_float(row.get("hard_penalty"), 0.0)
                + _to_float(row.get("safety_penalty"), 0.0)
                + _to_float(row.get("requirement_penalty"), 0.0)
                for row in traj_rows
            )
            rows.append(
                {
                    "value_scale": scale,
                    "baseline_name": baseline,
                    "delivered_utility_rescaled": delivered_rescaled,
                    "net_system_utility_rescaled": delivered_rescaled - resource_cost - penalty,
                    "high_value_delivered_utility": high_delivered,
                    "low_value_delivered_utility": low_delivered,
                    "high_value_task_completion_proxy": completion,
                    "low_value_deferred_or_expired_proxy": low_bad,
                    "queue_pressure_proxy": queue_max,
                    "expected_baseline_advantage_or_risk": (
                        "utility_rule_also_benefits_from_same_scale"
                        if baseline == "utility_rule"
                        else "proposed_advantage_requires_resource_constrained_high_value_closure"
                        if baseline == "proposed_requirement_aware"
                        else "fixed_or_ablated_baseline_may_not_reallocate_toward_high_value"
                    ),
                    "fairness_note": "counterfactual rescaling; all baselines use same value scale but behavior was not rerun",
                }
            )
    return rows


def _metric_sensitivity(roi_pressure: list[dict[str, object]], baseline_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pressure_by_scale = {row["value_scale"]: row for row in roi_pressure}
    natural_by_scale = {
        row["value_scale"]: row
        for row in baseline_rows
        if row["baseline_name"] == "proposed_requirement_aware"
    }
    scenario_dirs = {
        "natural": None,
        "resource_constrained": PROJECT_ROOT / "outputs" / "_targeted_resource_scarcity_stress_test" / "case_runs" / "regime2_shadow_resource_scarce" / "proposed_full_current" / "summary.csv",
        "contact_shortage": PROJECT_ROOT / "outputs" / "_targeted_resource_scarcity_stress_test" / "case_runs" / "regime1_contact_scarce" / "proposed_full_current" / "summary.csv",
        "queue_pressure": PROJECT_ROOT / "outputs" / "_targeted_resource_scarcity_stress_test" / "case_runs" / "regime3_critical_burst_after_greedy_window" / "proposed_full_current" / "summary.csv",
    }
    for scale in ALL_SCALES:
        for scenario, path in scenario_dirs.items():
            if path is None:
                base = natural_by_scale.get(scale, {})
                rows.append(
                    {
                        "value_scale": scale,
                        "scenario": scenario,
                        "high_value_completion_proxy": base.get("high_value_task_completion_proxy", 0),
                        "high_value_delivered_utility": base.get("high_value_delivered_utility", 0.0),
                        "low_value_sacrifice_ratio": _to_float(base.get("low_value_deferred_or_expired_proxy")) / 100.0,
                        "delivered_utility": base.get("delivered_utility_rescaled", 0.0),
                        "net_system_utility": base.get("net_system_utility_rescaled", 0.0),
                        "queue_next_mb_max": base.get("queue_pressure_proxy", 0.0),
                        "expired_items": "",
                        "hard_safety_violation_count": "",
                        "residual_followup_count": "",
                        "roi_context_pressure_proxy": pressure_by_scale.get(scale, {}).get("queue_pressure_proxy_if_only_selective_candidates_upgraded", 0.0),
                    }
                )
                continue
            summary_rows = _read_csv(path)
            summary = summary_rows[0] if summary_rows else {}
            rows.append(
                {
                    "value_scale": scale,
                    "scenario": scenario,
                    "high_value_completion_proxy": summary.get("completed_tasks", ""),
                    "high_value_delivered_utility": summary.get("high_value_task_utility", ""),
                    "low_value_sacrifice_ratio": summary.get("deadline_missed_count", ""),
                    "delivered_utility": summary.get("total_effective_utility", ""),
                    "net_system_utility": summary.get("total_penalized_reward", ""),
                    "queue_next_mb_max": summary.get("max_queue_mb", ""),
                    "expired_items": summary.get("expired_items", ""),
                    "hard_safety_violation_count": summary.get("safety_violation_count", ""),
                    "residual_followup_count": summary.get("residual_delivery_count", ""),
                    "roi_context_pressure_proxy": pressure_by_scale.get(scale, {}).get("queue_pressure_proxy_if_only_selective_candidates_upgraded", 0.0),
                }
            )
    return rows


def _high_value_funnel(task_rows: list[dict[str, object]], baseline_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for scale in ALL_SCALES:
        b = next((row for row in baseline_rows if row["value_scale"] == scale and row["baseline_name"] == "proposed_requirement_aware"), {})
        high_count = sum(int(row["assigned_value_level"] in {"high", "urgent"}) for row in task_rows)
        rows.append(
            {
                "value_scale": scale,
                "high_or_urgent_task_count": high_count,
                "total_task_count": len(task_rows),
                "high_or_urgent_task_share": high_count / max(len(task_rows), 1),
                "high_value_delivered_utility_proxy": b.get("high_value_delivered_utility", 0.0),
                "high_value_completion_proxy": b.get("high_value_task_completion_proxy", 0),
                "low_value_sacrifice_proxy": b.get("low_value_deferred_or_expired_proxy", 0),
            }
        )
    return rows


def _write_config_candidates() -> None:
    payload = {
        "audit_only": True,
        "base_prior_score": "mission_prior_value_v2 output B_i; current audit falls back to computed v2 score when trace metadata lacks new_task_value",
        "smooth_base_function": "f(B_i)=0.8+0.4*B_i",
        "normalization_candidates": {
            "mean_preserving": "alpha = old_mean_task_value / mean(M_i*f(B_i)); G_i=alpha*M_i*f(B_i)",
            "max": "G_i = M_i*f(B_i) / max(M_i*f(B_i))",
        },
        "value_skew_candidates": {
            "old": "current task_value distribution",
            **SCALES,
        },
        "recommended_initial_candidate": "moderate with mean-preserving normalization",
    }
    _write_text(OUT_DIR / "value_skew_config_candidates.yaml", yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def _report_text(
    distribution: list[dict[str, object]],
    eriv_distribution: list[dict[str, object]],
    roi_pressure: list[dict[str, object]],
) -> tuple[str, str, str]:
    dist = {row["value_scale"]: row for row in distribution if row.get("normalization") in {"current", "mean_preserving"}}
    eriv = {row["value_scale"]: row for row in eriv_distribution}
    roi = {row["value_scale"]: row for row in roi_pressure}
    moderate_ratio = _to_float(dist.get("moderate", {}).get("high_low_mean_ratio"))
    stress_risk = roi.get("stress", {}).get("accidental_low_value_upgrade_risk", 0.0)
    rec_case = "A"
    if moderate_ratio > 10.0:
        rec_case = "B"
    if _to_float(stress_risk) > 0.25:
        rec_case = "D"
    recommended = f"""# Recommended Value-Skew Design

## Executive Summary

The old value scale is too conservative for a resource-constrained task-utility claim. It preserves only a narrow high/low spread, so cost, deadline, and queue effects can dominate task criticality. A mean-preserving value-skew layer keeps the reward scale stable while widening mission-value heterogeneity.

## Recommendation

- Main experiment: `moderate = 1:3:8:15` with mean-preserving normalization.
- Sensitivity: `mild = 1:2:3:5` and `stress = 1:5:20:50`.
- Legacy reference: keep current old scale as an ablation/reference.
- ERIV threshold: recalibrate from candidate density, not by direct multiplier. Moderate suggested match-count threshold is `{_to_float(eriv.get('moderate', {}).get('suggested_threshold_to_match_old_candidate_count')):.4f}`.
- ROI+Ctx gate: add a semantic gate so ROI+Ctx upgrade is limited to context/evidence/high/urgent tasks, then add a value-density gate.
- Scenario design: run natural plus resource-constrained/contact-shortage cases. Natural runs alone are unlikely to expose the proposed method's advantage.

## Why Old 1-2x May Be Too Conservative

Old high/low mean ratio: `{_to_float(dist.get('old', {}).get('high_low_mean_ratio')):.3f}`. Moderate high/low mean ratio: `{moderate_ratio:.3f}`. Stress high/low mean ratio: `{_to_float(dist.get('stress', {}).get('high_low_mean_ratio')):.3f}`.

## Base Prior vs Calibrated Mission Value

Treat mission_prior_value_v2 as `B_i`, a leakage-free base prior score. The skew layer should sit after it:

`G_i = alpha * M_i * (0.8 + 0.4 B_i)`.

This preserves the prior's no-leakage properties and makes scenario-level value heterogeneity explicit.

## No-Leakage and No-Double-Counting Check

The skew level uses task class, requirement class, high-value/emergency flags, and conservative prior quantiles. It does not use delivery outcomes, detector labels, residual success, baseline wins, or queue outcomes. Resource cost remains in the action-dependent cost term, so value-skew does not duplicate energy/queue costs.

## Normalization

Mean-preserving normalization is preferred. Max normalization compresses most low and medium tasks after the largest urgent/high multiplier and changes the global reward scale, making comparisons harder.

## ROI+Ctx Upgrade Pressure

Moderate increases pressure toward context/evidence candidates while keeping a semantic gate necessary. Stress is useful as sensitivity only; it can over-amplify high/urgent candidates and should not be the main configuration.

## Baseline Fairness

All baselines must use the same calibrated task value. Utility rule will also benefit from a stronger value scale. The proposed advantage should be judged on resource-constrained high-value closure, not raw delivered count.

## Resource-Constrained Scenario Implication

The value-skew layer should be evaluated jointly with resource scarcity, contact shortage, and queue pressure. These scenarios are where a task-utility-driven co-design should dominate a utility-only or fixed baseline.

## Paper Wording

We do not assign absolute monetary values to EO tasks. Instead, we use scenario-dependent value-skew factors to model mission-value heterogeneity, while keeping resource costs in the action-dependent cost term. The original leakage-free prior is retained as a base prior score and calibrated by the value-skew multiplier. We evaluate mild, moderate, and stress value-skew settings to avoid overfitting to a single assumed multiplier.
"""
    implementation = """# Implementation Plan For Next Step

Do not implement in this audit. Minimal next-step implementation:

1. Add `src/data/value_skew_calibration.py`.
2. Keep mission_prior_value_v2 as `base_prior_score`.
3. Add `calibrated_task_value`, `value_level`, `value_multiplier`, and `value_skew_normalization`.
4. Add copied default-off configs: `value_skew_legacy.yaml`, `value_skew_mild.yaml`, `value_skew_moderate.yaml`, `value_skew_stress.yaml`.
5. Ensure every baseline reads the same calibrated value.
6. Route ERIV and selective ROI+Ctx gates through calibrated value in copied experiment configs only.
7. Keep old `task_value` or add a backward-compatible alias to avoid breaking existing scripts.
8. Add tests for no-leakage, monotonicity, mean-preserving normalization, baseline fairness, and unchanged old config.
9. Add `scripts/run_value_skew_ab.py`.
10. Add `scripts/run_value_skew_gate_recalibration.py`.
"""
    final = f"""# Final Recommendation

Recommendation case: {rec_case}.

Adopt moderate `1:3:8:15` as the next main experiment candidate, with mild and stress as sensitivity settings, under mean-preserving normalization. This audit does not change main defaults and does not tune reward logic.

Required answers:
1. mission_prior_value_v2 is suitable as `base_prior_score`.
2. The value-skew calibration layer should be external after the base prior.
3. It does not break no-leakage if classification uses only task prior fields.
4. It does not double-count resource cost because resource cost remains action-dependent.
5. Mean-preserving normalization keeps reward scale stable.
6. Max normalization can compress low-value tasks and alter global scale.
7. Mean-preserving normalization is better for formal integration.
8. Old 1-2x is too conservative for resource-constrained high-value prioritization.
9. Moderate widens heterogeneity without using the stress extreme.
10. Stress should remain sensitivity.
11. Mild is a conservative comparison.
12. All baselines can and should share the same value scale.
13. Utility rule will also become stronger, so proposed should be evaluated on high-value closure under scarcity.
14. The most revealing metrics are high-value completion, context/evidence closure, lifecycle shadow utility, residual follow-up utility, and low-value sacrifice ratio.
15. Next code step: implement a default-off value-skew calibration layer plus ERIV/ROI+Ctx gate recalibration.
"""
    return recommended, implementation, final


def _archive_manifest() -> None:
    files = sorted(path.relative_to(OUT_DIR).as_posix() for path in OUT_DIR.rglob("*") if path.is_file())
    _write_text(OUT_DIR / "archive_manifest.txt", "\n".join(files))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--scenarios", default="natural,targeted_resource_scarcity")
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    command = f"python scripts/audit_value_skew_feasibility.py --seeds {','.join(str(seed) for seed in seeds)}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    task_rows = _load_tasks(seeds)
    _assign_levels(task_rows)
    values_by_scale = _calibrate_values(task_rows)
    base_summary = _base_prior_distribution(task_rows)
    value_trace = _value_trace(task_rows)
    distribution = _distribution_summary(task_rows, values_by_scale)
    candidate_rows = _candidate_rows(task_rows, values_by_scale)
    eriv_distribution, eriv_suggestions = _eriv_summary(candidate_rows)
    roi_pressure = _roi_context_pressure(task_rows, candidate_rows, eriv_suggestions)
    residual_summary = _residual_followup_summary(candidate_rows, eriv_suggestions)
    baseline_fairness = _baseline_fairness()
    sensitivity = _metric_sensitivity(roi_pressure, baseline_fairness)
    high_value_funnel = _high_value_funnel(task_rows, baseline_fairness)
    recommended, implementation, final = _report_text(distribution, eriv_distribution, roi_pressure)

    _write_text(
        OUT_DIR / "README.md",
        "# Value-Skew Feasibility Audit\n\nAudit-only counterfactual value-skew calibration study. No main defaults, reward logic, Module 2 scoring, SafeFilter, utility_rule, queue policy, lifecycle closure, or information_sufficiency rules are changed.",
    )
    _write_text(OUT_DIR / "command_log.md", f"# Command log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "scenarios": args.scenarios,
                "audit_only": True,
                "main_defaults_changed": False,
                "normalization_compared": ["mean_preserving", "max"],
                "value_scales": {"old": "current", **SCALES},
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_config_candidates()
    _write_csv(OUT_DIR / "base_prior_distribution_summary.csv", base_summary)
    _write_csv(OUT_DIR / "value_level_assignment_trace.csv", value_trace)
    _write_csv(OUT_DIR / "calibrated_value_distribution_summary.csv", distribution)
    _write_csv(OUT_DIR / "old_vs_mild_moderate_stress_value_trace.csv", value_trace)
    _write_csv(OUT_DIR / "eriv_rescaled_distribution_summary.csv", eriv_distribution)
    _write_csv(OUT_DIR / "eriv_threshold_recalibration_suggestion.csv", eriv_suggestions)
    _write_csv(OUT_DIR / "roi_context_upgrade_pressure_summary.csv", roi_pressure)
    _write_csv(OUT_DIR / "residual_followup_candidate_rescale_summary.csv", residual_summary)
    _write_csv(OUT_DIR / "high_value_task_priority_funnel.csv", high_value_funnel)
    _write_csv(OUT_DIR / "baseline_fairness_impact_summary.csv", baseline_fairness)
    _write_csv(OUT_DIR / "value_skew_metric_sensitivity_summary.csv", sensitivity)
    _write_text(OUT_DIR / "recommended_value_skew_design.md", recommended)
    _write_text(OUT_DIR / "implementation_plan_for_next_step.md", implementation)
    _write_text(OUT_DIR / "final_recommendation.md", final)
    _write_text(OUT_DIR / "git_status.txt", _git(["status", "--short"]))
    script_path = PROJECT_ROOT / "scripts" / "audit_value_skew_feasibility.py"
    diff = _git(["diff", "--", "scripts/audit_value_skew_feasibility.py"])
    _write_text(OUT_DIR / "git_diff.patch", diff if diff.strip() else _add_file_patch(script_path))
    _archive_manifest()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", OUT_DIR)

    old_ratio = next(row for row in distribution if row["value_scale"] == "old" and row["normalization"] == "current")["high_low_mean_ratio"]
    mild_ratio = next(row for row in distribution if row["value_scale"] == "mild" and row["normalization"] == "mean_preserving")["high_low_mean_ratio"]
    moderate_ratio = next(row for row in distribution if row["value_scale"] == "moderate" and row["normalization"] == "mean_preserving")["high_low_mean_ratio"]
    stress_ratio = next(row for row in distribution if row["value_scale"] == "stress" and row["normalization"] == "mean_preserving")["high_low_mean_ratio"]
    moderate_threshold = next(row for row in eriv_suggestions if row["value_scale"] == "moderate")["threshold_match_old_count"]
    print(f"audit output directory: {OUT_DIR}")
    print(f"zip path: {ZIP_PATH}")
    print(f"old high/low value ratio: {_to_float(old_ratio):.4f}")
    print(f"mild high/low value ratio: {_to_float(mild_ratio):.4f}")
    print(f"moderate high/low value ratio: {_to_float(moderate_ratio):.4f}")
    print(f"stress high/low value ratio: {_to_float(stress_ratio):.4f}")
    print("suggested main value scale: moderate=1:3:8:15 with mean-preserving normalization")
    print(f"suggested ERIV recalibration range: around {_to_float(moderate_threshold):.4f} for match-count, then sweep around it")
    print("suggested ROI+Ctx gate recalibration direction: semantic gate + value-density gate")
    print("final recommendation case: A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
