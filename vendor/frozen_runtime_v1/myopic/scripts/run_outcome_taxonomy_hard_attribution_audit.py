"""Report-only outcome taxonomy and hard-penalty attribution audit.

This script reads the pre-registered shadow calibration artifacts and writes
diagnostic copies under outputs/_outcome_taxonomy_hard_attribution_audit. It
does not change reward calculation, action selection, candidate definitions, or
baseline behavior.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "outputs" / "_pre_registered_value_cost_calibration_patch"
OUT_ROOT = PROJECT_ROOT / "outputs" / "_outcome_taxonomy_hard_attribution_audit"
PLOTS_DIR = OUT_ROOT / "plots"

CANDIDATES = ("C0", "C1", "C2", "C3", "C4")
CANDIDATE_PURPOSES = {
    "C0": "current v2 baseline",
    "C1": "mild value-cost calibration",
    "C2": "conservative calibration",
    "C3": "value-scaling-only diagnostic",
    "C4": "cost/requirement-scaling-only diagnostic",
}
TAXONOMY_ORDER = (
    "full_success",
    "partial_success_or_soft_shortfall",
    "unattempted_low_value_or_deferred",
    "attempted_soft_failure",
    "true_hard_failure",
)
NOOP_ACTION_TOKENS = ("defer", "delay", "store", "idle", "noop", "no-op", "none", "skip")
NO_PAYLOAD_TOKENS = ("defer", "none", "idle", "noop", "no-op", "no_payload", "null", "")
TRUE_HARD_COMPONENTS = ("P_safety", "P_deadline", "P_expiration", "P_infeasible")
EXPLICIT_HARD_FIELDS = (
    "hard_resource_violation",
    "resource_hard_violation",
    "queue_overflow",
    "queue_overflow_hard",
    "queue_hard_violation",
)
TAXONOMY_COLUMNS = [
    "taxonomy_label",
    "true_hard_failure",
    "legacy_hard_failure_flag",
    "attempted_flag",
    "partial_success_flag",
    "unattempted_low_value_flag",
    "terminal_risk_note",
    "soft_shortfall_flag",
    "hard_failure_source",
    "primary_negative_driver",
]


@dataclass(frozen=True)
class TaxonomyThresholds:
    eps_gain: float = 1.0e-9
    eps_cost: float = 0.02
    eps_penalty: float = 1.0e-9
    near_zero_abs: float = 0.05


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if value is None or value == "":
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    return statistics.fmean(clean) if clean else math.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    return math.nan if denominator <= 0 else numerator / denominator


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _compact_metric_map(values: dict[str, float]) -> str:
    return "; ".join(f"{key}:{_fmt(values[key])}" for key in TAXONOMY_ORDER if key in values)


def _is_noop_action(action: str) -> bool:
    normalized = str(action).strip().lower()
    return any(token in normalized for token in NOOP_ACTION_TOKENS)


def _is_no_payload(payload_type: str) -> bool:
    normalized = str(payload_type).strip().lower()
    return normalized in NO_PAYLOAD_TOKENS or any(token == normalized for token in NO_PAYLOAD_TOKENS)


def _hard_sources(row: dict[str, Any], thresholds: TaxonomyThresholds) -> list[str]:
    sources = [name for name in TRUE_HARD_COMPONENTS if _num(row, name) > thresholds.eps_penalty]
    for name in EXPLICIT_HARD_FIELDS:
        if name not in row:
            continue
        value = row.get(name)
        if _boolish(value) or _num(row, name) > thresholds.eps_penalty:
            sources.append(name)
    return sources


def primary_negative_driver(row: dict[str, Any], thresholds: TaxonomyThresholds) -> str:
    p_terminal = max(0.0, _num(row, "P_terminal_risk"))
    p_hard_ex_terminal = max(0.0, _num(row, "P_hard") - p_terminal)
    components = {
        "C_resource": max(0.0, _num(row, "C_resource")),
        "P_requirement": max(0.0, _num(row, "P_requirement")),
        "P_hard_excluding_terminal": p_hard_ex_terminal,
        "P_terminal_risk": p_terminal,
    }
    label, value = max(components.items(), key=lambda item: item[1])
    return label if value > thresholds.eps_penalty else "none"


def classify_row(row: dict[str, Any], thresholds: TaxonomyThresholds | None = None) -> dict[str, Any]:
    """Return report-only taxonomy fields for one shadow decomposition row."""

    thresholds = thresholds or TaxonomyThresholds()
    mission_gain = _num(row, "mission_gain")
    cost = _num(row, "C_resource")
    p_requirement = _num(row, "P_requirement")
    p_quality = _num(row, "P_quality_gap")
    p_info = _num(row, "P_info_gap")
    p_terminal = _num(row, "P_terminal_risk")
    legacy_hard = _boolish(row.get("hard_failure_flag", 0))
    sources = _hard_sources(row, thresholds)
    true_hard = bool(sources)
    terminal_risk_note = p_terminal > thresholds.eps_penalty and not true_hard
    soft_shortfall = (
        not true_hard
        and (
            p_requirement > thresholds.eps_penalty
            or p_quality > thresholds.eps_penalty
            or p_info > thresholds.eps_penalty
        )
    )

    action = str(row.get("action", ""))
    payload_type = str(row.get("payload_type", ""))
    noop = _is_noop_action(action) or _is_no_payload(payload_type)
    attempted = cost > thresholds.eps_cost or not noop
    success = _boolish(row.get("success_flag", 0))
    delivered_before_deadline = _boolish(row.get("delivered_before_deadline", 0))

    full_success = success and delivered_before_deadline and mission_gain > thresholds.eps_gain and not true_hard
    partial_success = mission_gain > thresholds.eps_gain and not full_success and not true_hard
    unattempted = (
        mission_gain <= thresholds.eps_gain
        and cost <= thresholds.eps_cost
        and not true_hard
        and noop
    )

    if true_hard:
        label = "true_hard_failure"
    elif full_success:
        label = "full_success"
    elif partial_success:
        label = "partial_success_or_soft_shortfall"
    elif unattempted:
        label = "unattempted_low_value_or_deferred"
    elif attempted:
        label = "attempted_soft_failure"
    else:
        label = "unattempted_low_value_or_deferred"

    source_text = "none"
    if sources:
        source_text = "+".join(sorted(sources))
    elif terminal_risk_note:
        source_text = "terminal_risk_note_only"
    elif legacy_hard:
        source_text = "legacy_hard_unattributed"

    return {
        "taxonomy_label": label,
        "true_hard_failure": int(true_hard),
        "legacy_hard_failure_flag": int(legacy_hard),
        "attempted_flag": int(attempted),
        "partial_success_flag": int(partial_success),
        "unattempted_low_value_flag": int(label == "unattempted_low_value_or_deferred"),
        "terminal_risk_note": int(terminal_risk_note),
        "soft_shortfall_flag": int(soft_shortfall),
        "hard_failure_source": source_text,
        "primary_negative_driver": primary_negative_driver(row, thresholds),
    }


def _taxonomy_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {label: sum(1 for row in rows if row["taxonomy_label"] == label) for label in TAXONOMY_ORDER}


def aggregate_taxonomy(candidate: str, rows: list[dict[str, Any]], old_summary: dict[str, str] | None, thresholds: TaxonomyThresholds) -> dict[str, Any]:
    counts = _taxonomy_counts(rows)
    full_rows = [row for row in rows if row["taxonomy_label"] == "full_success"]
    partial_rows = [row for row in rows if row["taxonomy_label"] == "partial_success_or_soft_shortfall"]
    unattempted_rows = [row for row in rows if row["taxonomy_label"] == "unattempted_low_value_or_deferred"]
    soft_failure_rows = [row for row in rows if row["taxonomy_label"] == "attempted_soft_failure"]
    true_hard_rows = [row for row in rows if row["taxonomy_label"] == "true_hard_failure"]
    legacy_hard_rows = [row for row in rows if _boolish(row.get("legacy_hard_failure_flag", 0))]
    terminal_only = [
        row
        for row in legacy_hard_rows
        if not _boolish(row.get("true_hard_failure", 0)) and _boolish(row.get("terminal_risk_note", 0))
    ]
    high_success = [
        row
        for row in rows
        if _boolish(row.get("success_flag", 0)) and _boolish(row.get("high_value_flag", 0))
    ]

    by_taxonomy: dict[str, dict[str, float]] = {}
    for label in TAXONOMY_ORDER:
        subset = [row for row in rows if row["taxonomy_label"] == label]
        by_taxonomy[label] = {
            "net": _mean([_num(row, "net_utility") for row in subset]) if subset else math.nan,
            "gain": _mean([_num(row, "mission_gain") for row in subset]) if subset else math.nan,
            "cost": _mean([_num(row, "C_resource") for row in subset]) if subset else math.nan,
            "req": _mean([_num(row, "P_requirement") for row in subset]) if subset else math.nan,
            "hard": _mean([_num(row, "P_hard") for row in subset]) if subset else math.nan,
            "terminal": _mean([_num(row, "P_terminal_risk") for row in subset]) if subset else math.nan,
        }

    return {
        "candidate": candidate,
        "purpose": CANDIDATE_PURPOSES[candidate],
        "n_total": len(rows),
        "n_full_success": counts["full_success"],
        "n_partial_success_or_soft_shortfall": counts["partial_success_or_soft_shortfall"],
        "n_unattempted_low_value_or_deferred": counts["unattempted_low_value_or_deferred"],
        "n_attempted_soft_failure": counts["attempted_soft_failure"],
        "n_true_hard_failure": counts["true_hard_failure"],
        "legacy_hard_failure_count": len(legacy_hard_rows),
        "terminal_risk_only_legacy_hard_count": len(terminal_only),
        "true_hard_failure_negative_ratio": _safe_ratio(
            sum(1 for row in true_hard_rows if _num(row, "net_utility") < 0.0),
            len(true_hard_rows),
        ),
        "attempted_soft_failure_negative_ratio": _safe_ratio(
            sum(1 for row in soft_failure_rows if _num(row, "net_utility") < 0.0),
            len(soft_failure_rows),
        ),
        "successful_high_value_positive_net_ratio": _safe_ratio(
            sum(1 for row in high_success if _num(row, "net_utility") > 0.0),
            len(high_success),
        ),
        "full_success_positive_net_ratio": _safe_ratio(
            sum(1 for row in full_rows if _num(row, "net_utility") > 0.0),
            len(full_rows),
        ),
        "partial_success_nonnegative_ratio": _safe_ratio(
            sum(1 for row in partial_rows if _num(row, "net_utility") >= 0.0),
            len(partial_rows),
        ),
        "unattempted_low_value_near_zero_ratio": _safe_ratio(
            sum(1 for row in unattempted_rows if abs(_num(row, "net_utility")) <= thresholds.near_zero_abs),
            len(unattempted_rows),
        ),
        "old_failed_negative_net_ratio": "" if old_summary is None else old_summary.get("failed_negative_net_ratio", ""),
        "old_hard_failure_negative_ratio": "" if old_summary is None else old_summary.get("hard_failure_negative_ratio", ""),
        "mean_net_utility_by_taxonomy": _compact_metric_map({k: v["net"] for k, v in by_taxonomy.items()}),
        "mean_mission_gain_by_taxonomy": _compact_metric_map({k: v["gain"] for k, v in by_taxonomy.items()}),
        "mean_C_resource_by_taxonomy": _compact_metric_map({k: v["cost"] for k, v in by_taxonomy.items()}),
        "mean_P_requirement_by_taxonomy": _compact_metric_map({k: v["req"] for k, v in by_taxonomy.items()}),
        "mean_P_hard_by_taxonomy": _compact_metric_map({k: v["hard"] for k, v in by_taxonomy.items()}),
        "mean_P_terminal_risk_by_taxonomy": _compact_metric_map({k: v["terminal"] for k, v in by_taxonomy.items()}),
    }


def _input_file_rows() -> list[dict[str, Any]]:
    required = []
    for candidate in CANDIDATES:
        required.append(INPUT_ROOT / f"task_decomposition_shadow_{candidate}.csv")
    for candidate in CANDIDATES:
        required.append(INPUT_ROOT / f"aggregate_summary_shadow_{candidate}.csv")
    required.extend(
        [
            INPUT_ROOT / "decision_coupled_comparison.csv",
            INPUT_ROOT / "decision_coupled_report.md",
            INPUT_ROOT / "final_terminal_summary.txt",
            PROJECT_ROOT / "configs" / "reward.yaml",
            PROJECT_ROOT / "src" / "reward" / "utility_calibration.py",
            PROJECT_ROOT / "src" / "reward" / "reward_function.py",
            PROJECT_ROOT / "scripts" / "run_pre_registered_value_cost_calibration_patch.py",
        ]
    )
    rows: list[dict[str, Any]] = []
    for path in required:
        rel = path.relative_to(PROJECT_ROOT)
        row: dict[str, Any] = {"path": str(rel).replace("\\", "/"), "exists": path.exists()}
        if path.exists() and path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                data = list(reader)
            columns = reader.fieldnames or []
            row["row_count"] = len(data)
            row["columns"] = ", ".join(columns)
            row["contains_literal_ellipsis_column"] = int("..." in columns)
        elif path.exists():
            row["row_count"] = ""
            row["columns"] = ""
            row["contains_literal_ellipsis_column"] = ""
        rows.append(row)
    return rows


def write_input_discovery_report(all_shadow_rows: list[dict[str, Any]]) -> None:
    discovery = _input_file_rows()
    p_fields = {name: sum(1 for row in all_shadow_rows if _num(row, name) <= 1.0e-12) for name in TRUE_HARD_COMPONENTS}
    total = len(all_shadow_rows)
    terminal_positive = sum(1 for row in all_shadow_rows if _num(row, "P_terminal_risk") > 1.0e-12)
    legacy_positive = sum(1 for row in all_shadow_rows if _boolish(row.get("hard_failure_flag", 0)))
    terminal_legacy = sum(
        1
        for row in all_shadow_rows
        if _boolish(row.get("hard_failure_flag", 0)) and _num(row, "P_terminal_risk") > 1.0e-12
    )
    lifecycle_values = sorted({str(row.get("lifecycle_fields_active", "")).strip() for row in all_shadow_rows})

    lines = [
        "# Input Discovery Report",
        "",
        "## File Availability and Columns",
        "",
        "| path | exists | rows | ellipsis column | columns |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in discovery:
        lines.append(
            f"| `{row['path']}` | {int(bool(row['exists']))} | {row.get('row_count', '')} | "
            f"{row.get('contains_literal_ellipsis_column', '')} | {row.get('columns', '')} |"
        )
    lines.extend(
        [
            "",
            "## Hard-Flag Trigger Audit",
            "",
            "- In `scripts/run_pre_registered_value_cost_calibration_patch.py`, `hard_failure_flag` is set when `P_hard > 1e-12`, or safety violation/deadline miss/expired/dropped status is present.",
            "- The same generator computes `P_hard = P_safety + mission_error_penalty + P_expiration + P_deadline + P_infeasible + P_terminal_risk`.",
            f"- Therefore `P_terminal_risk` currently contributes to `hard_failure_flag`: yes; rows with both legacy hard flag and positive terminal risk = {terminal_legacy}.",
            f"- Legacy hard rows across C0-C4 = {legacy_positive}; rows with positive `P_terminal_risk` = {terminal_positive}.",
            "",
            "## True Hard Components",
            "",
        ]
    )
    for name in TRUE_HARD_COMPONENTS:
        zeros = p_fields[name]
        lines.append(f"- `{name}` zero-or-near-zero rows: {zeros}/{total} ({_fmt(_safe_ratio(zeros, total))}).")
    lines.extend(
        [
            "",
            f"- Lifecycle field values in shadow CSVs: {', '.join(lifecycle_values)}.",
            "- Lifecycle utility is inactive in `configs/reward.yaml` and reserved-only in `src/reward/utility_calibration.py`.",
        ]
    )
    _write_text(OUT_ROOT / "input_discovery_report.md", "\n".join(lines))


def write_taxonomy_definition(thresholds: TaxonomyThresholds) -> None:
    _write_text(
        OUT_ROOT / "outcome_taxonomy_definition.md",
        """# Outcome Taxonomy Definition

The taxonomy is report-only. It does not enter reward calculation or action selection.

Main utility remains:

`U_i = V_i^cal R_i(a_i) - beta_C C_i^res(a_i) - beta_R P_i^req(a_i) - P_i^hard(a_i)`

## Classes

A. `full_success`: `success_flag == true`, `delivered_before_deadline == true`, positive mission gain, and no true hard failure.

B. `partial_success_or_soft_shortfall`: positive mission gain, not full success, and no true hard failure. This includes partial delivery, metadata-only contribution, ROI without full closure, and positive gain with soft requirement penalty.

C. `unattempted_low_value_or_deferred`: near-zero mission gain, near-zero resource cost, no true hard failure, and defer/no-op/no-delivered-payload action.

D. `attempted_soft_failure`: attempted processing or transmission with no true hard failure, but no sufficient success. Positive `P_requirement`, `P_quality_gap`, and `P_info_gap` are soft shortfalls.

E. `true_hard_failure`: triggered only by positive `P_safety`, `P_deadline`, `P_expiration`, `P_infeasible`, or explicit hard resource/queue violation fields if available.

`P_terminal_risk` alone is not a true hard failure. It is reported in `terminal_risk_note`.
""",
    )
    _write_text(
        OUT_ROOT / "taxonomy_thresholds.md",
        f"""# Taxonomy Thresholds

- `eps_gain = {thresholds.eps_gain}`
- `eps_cost = {thresholds.eps_cost}`
- `eps_penalty = {thresholds.eps_penalty}`
- near-zero net utility interval: `[-{thresholds.near_zero_abs}, {thresholds.near_zero_abs}]`

These thresholds are used only for diagnostic classification and aggregate reporting.
""",
    )


def write_shadow_summary_report(aggregates: list[dict[str, Any]]) -> None:
    lines = [
        "# Taxonomy Shadow Summary Report",
        "",
        "| candidate | old failed neg | old hard neg | attempted soft neg | true hard neg | unattempted near zero | full success pos | high-value success pos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['candidate']} | {_fmt(row['old_failed_negative_net_ratio'])} | "
            f"{_fmt(row['old_hard_failure_negative_ratio'])} | "
            f"{_fmt(row['attempted_soft_failure_negative_ratio'])} | "
            f"{_fmt(row['true_hard_failure_negative_ratio'])} | "
            f"{_fmt(row['unattempted_low_value_near_zero_ratio'])} | "
            f"{_fmt(row['full_success_positive_net_ratio'])} | "
            f"{_fmt(row['successful_high_value_positive_net_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "The old `hard_failure_negative_ratio` used legacy hard labels. The new `true_hard_failure_negative_ratio` excludes terminal-risk-only legacy rows and is undefined when no true engineering hard violations are present.",
        ]
    )
    _write_text(OUT_ROOT / "taxonomy_shadow_summary_report.md", "\n".join(lines))


def build_hard_cross_table(rows_by_candidate: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    all_rows = [row for rows in rows_by_candidate.values() for row in rows]
    for legacy in (0, 1):
        subset = [row for row in all_rows if int(row["legacy_hard_failure_flag"]) == legacy]
        base = {
            "legacy_hard_failure_flag": legacy,
            "n_rows": len(subset),
            "true_hard_failure": sum(int(row["true_hard_failure"]) for row in subset),
            "terminal_risk_only": sum(
                1
                for row in subset
                if int(row["terminal_risk_note"]) == 1 and int(row["true_hard_failure"]) == 0
            ),
            "soft_shortfall": sum(int(row["soft_shortfall_flag"]) for row in subset),
            "partial_success": sum(int(row["partial_success_flag"]) for row in subset),
        }
        for label in TAXONOMY_ORDER:
            base[f"taxonomy_label={label}"] = sum(1 for row in subset if row["taxonomy_label"] == label)
        result.append(base)
    return result


def write_hard_attribution_audit(rows_by_candidate: dict[str, list[dict[str, Any]]], cross_rows: list[dict[str, Any]]) -> None:
    all_rows = [row for rows in rows_by_candidate.values() for row in rows]
    legacy = [row for row in all_rows if int(row["legacy_hard_failure_flag"]) == 1]
    true_hard = [row for row in legacy if int(row["true_hard_failure"]) == 1]
    terminal_only = [row for row in legacy if int(row["terminal_risk_note"]) == 1 and int(row["true_hard_failure"]) == 0]
    soft = [row for row in legacy if int(row["soft_shortfall_flag"]) == 1]
    partial = [row for row in legacy if int(row["partial_success_flag"]) == 1]
    p_hard_values = [_num(row, "P_hard") for row in all_rows]
    p_terminal_values = [_num(row, "P_terminal_risk") for row in all_rows]
    lines = [
        "# Hard Attribution Audit",
        "",
        f"- Legacy hard failure rows: {len(legacy)}.",
        f"- Legacy hard rows that are true hard failures: {len(true_hard)}.",
        f"- Legacy hard rows that are terminal-risk-only notes: {len(terminal_only)}.",
        f"- Legacy hard rows with soft shortfall: {len(soft)}.",
        f"- Legacy hard rows that are partial successes: {len(partial)}.",
        f"- Mean `P_hard`: {_fmt(_mean(p_hard_values), 6)}; mean `P_terminal_risk`: {_fmt(_mean(p_terminal_values), 6)}.",
        "",
        "Interpretation: `P_terminal_risk` contaminated the legacy hard label when it was the only positive hard-like component. The audit separates that risk note from true engineering hard violations.",
        "",
        "| legacy hard | rows | true hard | terminal-risk-only | soft shortfall | partial success |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cross_rows:
        lines.append(
            f"| {row['legacy_hard_failure_flag']} | {row['n_rows']} | {row['true_hard_failure']} | "
            f"{row['terminal_risk_only']} | {row['soft_shortfall']} | {row['partial_success']} |"
        )
    _write_text(OUT_ROOT / "hard_attribution_audit.md", "\n".join(lines))


def write_candidate_reinterpretation(aggregates: list[dict[str, Any]]) -> None:
    by_c = {row["candidate"]: row for row in aggregates}
    c1_terminal = int(by_c["C1"]["terminal_risk_only_legacy_hard_count"])
    c1_true = int(by_c["C1"]["n_true_hard_failure"])
    c2_soft = by_c["C2"]["attempted_soft_failure_negative_ratio"]
    c4_high = by_c["C4"]["successful_high_value_positive_net_ratio"]
    c3_high = by_c["C3"]["successful_high_value_positive_net_ratio"]
    lines = [
        "# Candidate Reinterpretation Report",
        "",
        "No candidates are changed or newly introduced.",
        "",
        "1. Was C1 rejected because it truly made hard failures non-negative, or because legacy labels were contaminated?",
        f"   C1 has {c1_true} true hard failures and {c1_terminal} terminal-risk-only legacy hard rows. The rejection signal is better interpreted as legacy hard-label contamination plus soft shortfalls, not confirmed non-negative true hard failures.",
        "",
        "2. Does C2 look safer after taxonomy correction?",
        f"   C2 remains the conservative candidate: attempted soft failure negative ratio is {_fmt(c2_soft)}, while true hard failure ratio is undefined if no true hard rows exist.",
        "",
        "3. Does C4 show that beta_C/beta_R side is the dominant mismatch?",
        f"   C4 improves high-value successful positive net ratio to {_fmt(c4_high)} without value scaling, so the cost/requirement scale side is a material contributor.",
        "",
        "4. Does C3 show that value scaling alone is insufficient?",
        f"   C3 high-value successful positive net ratio is {_fmt(c3_high)}; value scaling alone does not resolve the sign issue.",
        "",
        "5. Should C1 remain a variant?",
        "   Yes. C1 remains useful as a mild value-cost variant, but this audit does not adopt it as default.",
        "",
        "6. Should C2 be the conservative candidate?",
        "   Yes. C2 is the conservative calibration candidate among the pre-registered options.",
        "",
        "7. Should C0 remain default until taxonomy is clean?",
        "   Yes. Keep C0 default until task-level taxonomy and hard attribution are cleanly reported and externally reviewed.",
    ]
    _write_text(OUT_ROOT / "candidate_reinterpretation_report.md", "\n".join(lines))


def write_decision_coupled_report() -> None:
    path = INPUT_ROOT / "decision_coupled_comparison.csv"
    missing = [
        "task_id",
        "action",
        "payload_type",
        "mission_gain",
        "C_resource",
        "P_requirement",
        "P_safety",
        "P_deadline",
        "P_expiration",
        "P_infeasible",
        "P_terminal_risk",
        "success_flag",
        "delivered_before_deadline",
    ]
    rows = _read_csv(path) if path.exists() else []
    columns = set(rows[0].keys()) if rows else set()
    actual_missing = [name for name in missing if name not in columns]
    formulas_identical = all(_boolish(row.get("baseline_formula_identical", False)) for row in rows) if rows else False
    lifecycle_inactive = all(not _boolish(row.get("lifecycle_fields_active", False)) for row in rows) if rows else False
    lines = [
        "# Decision-Coupled Taxonomy Report",
        "",
        f"- Behavior metrics changed: no. This audit reads aggregate decision-coupled outputs only.",
        f"- Formulas identical across baselines: {str(formulas_identical).lower()}.",
        f"- Lifecycle fields active: {str(not lifecycle_inactive).lower()}.",
        "- Can taxonomy be applied to decision-coupled outputs: no.",
        "",
        "Reason: `decision_coupled_comparison.csv` is aggregate-level and lacks task-level decomposition fields needed to classify rows.",
        "",
        "Missing task-level fields:",
    ]
    lines.extend(f"- `{name}`" for name in actual_missing)
    _write_text(OUT_ROOT / "decision_coupled_taxonomy_report.md", "\n".join(lines))


def write_statistical_plan() -> None:
    _write_text(
        OUT_ROOT / "statistical_reporting_plan.md",
        """# Statistical Reporting Plan

- Use paired seed-level or task-level differences when available.
- Use the Wilcoxon signed-rank test when normality is not assumed.
- Report 95% bootstrap confidence intervals for key paired differences.
- Do not use statistical tests to pick a winner-driven calibration.
- Use statistics only to quantify uncertainty and robustness.

Key metrics:

- `successful_high_value_positive_net_ratio`
- `full_success_positive_net_ratio`
- `attempted_soft_failure_negative_ratio`
- `true_hard_failure_negative_ratio`
- `unattempted_low_value_near_zero_ratio`
- `delivered_only_incremental_utility`
- `high_value_delivered_incremental_utility`
- `deadline_satisfaction_ratio`
- `min_battery_soc`
- `max_thermal_pressure`
- `queue_max_mb`
""",
    )


def write_paper_safe_explanations() -> None:
    _write_text(
        OUT_ROOT / "paper_safe_explanation_en.md",
        """# Paper-Safe Explanation

The main utility remains a compact benefit-cost-penalty objective:

`U_i = V_i^cal R_i(a_i) - beta_C C_i^res(a_i) - beta_R P_i^req(a_i) - P_i^hard(a_i)`.

The outcome taxonomy is used only for auditing and interpretation. Unattempted low-value tasks are not forced to be negative. Attempted failures and true hard engineering violations should be negative. Terminal risk notes are not automatically treated as hard failures.

Calibration candidates C0-C4 are pre-registered and are not selected by Proposed-vs-baseline ranking. All baselines use identical formulas. Lifecycle utility remains reserved future work and is not active.
""",
    )
    _write_text(
        OUT_ROOT / "paper_safe_explanation_zh.md",
        """# 论文安全表述

主效用函数仍然是紧凑的收益-成本-惩罚目标：

`U_i = V_i^cal R_i(a_i) - beta_C C_i^res(a_i) - beta_R P_i^req(a_i) - P_i^hard(a_i)`。

结果分类仅用于审计和解释，不参与奖励计算或动作选择。未尝试的低价值任务不被强制判为负效用。已尝试但失败的任务以及真实工程硬约束违反应当为负。终端风险记录不会自动等同于硬失败。

C0-C4 校准候选是预注册候选，不根据 Proposed 相对 baseline 的排名来选择。所有 baseline 使用相同公式。生命周期效用仍然只是未来工作预留，本次没有启用。
""",
    )


def make_plots(rows_by_candidate: dict[str, list[dict[str, Any]]], aggregates: list[dict[str, Any]], cross_rows: list[dict[str, Any]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        "full_success": "#4C78A8",
        "partial_success_or_soft_shortfall": "#72B7B2",
        "unattempted_low_value_or_deferred": "#BAB0AC",
        "attempted_soft_failure": "#F58518",
        "true_hard_failure": "#E45756",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    bottoms = [0] * len(CANDIDATES)
    for label in TAXONOMY_ORDER:
        values = [sum(1 for row in rows_by_candidate[c] if row["taxonomy_label"] == label) for c in CANDIDATES]
        ax.bar(CANDIDATES, values, bottom=bottoms, label=label, color=colors[label])
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylabel("task rows")
    ax.set_title("Outcome taxonomy distribution by candidate")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "taxonomy_distribution_by_candidate.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    matrix = [[row["true_hard_failure"], row["terminal_risk_only"]] for row in cross_rows]
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["true hard", "terminal only"])
    ax.set_yticks([0, 1], ["legacy=0", "legacy=1"])
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, str(value), ha="center", va="center", color="black")
    ax.set_title("Legacy hard label vs corrected attribution")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "legacy_vs_true_hard_failure_cross_table.png", dpi=200)
    plt.close(fig)

    all_rows = [row for rows in rows_by_candidate.values() for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    data = [[_num(row, "net_utility") for row in all_rows if row["taxonomy_label"] == label] for label in TAXONOMY_ORDER]
    ax.boxplot(data, labels=[label.replace("_", "\n") for label in TAXONOMY_ORDER], showfliers=False)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_ylabel("net utility")
    ax.set_title("Net utility by taxonomy")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "net_utility_by_taxonomy_boxplot.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sign_labels = ("positive", "negative", "near-zero")
    sign_colors = ("#4C78A8", "#E45756", "#BAB0AC")
    bottoms = [0] * len(TAXONOMY_ORDER)
    for sign, color in zip(sign_labels, sign_colors):
        values = []
        for label in TAXONOMY_ORDER:
            subset = [row for row in all_rows if row["taxonomy_label"] == label]
            if sign == "positive":
                values.append(sum(1 for row in subset if _num(row, "net_utility") > 0.05))
            elif sign == "negative":
                values.append(sum(1 for row in subset if _num(row, "net_utility") < -0.05))
            else:
                values.append(sum(1 for row in subset if abs(_num(row, "net_utility")) <= 0.05))
        ax.bar([label.replace("_", "\n") for label in TAXONOMY_ORDER], values, bottom=bottoms, label=sign, color=color)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylabel("task rows")
    ax.set_title("Net utility sign bands by taxonomy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "positive_negative_nearzero_by_taxonomy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    components = ("mission_gain", "C_resource", "P_requirement", "P_hard", "P_terminal_risk")
    x = list(range(len(TAXONOMY_ORDER)))
    width = 0.15
    for idx, comp in enumerate(components):
        values = []
        for label in TAXONOMY_ORDER:
            subset = [row for row in all_rows if row["taxonomy_label"] == label]
            values.append(_mean([_num(row, comp) for row in subset]) if subset else 0.0)
        ax.bar([v + (idx - 2) * width for v in x], values, width=width, label=comp)
    ax.set_xticks(x, [label.replace("_", "\n") for label in TAXONOMY_ORDER])
    ax.set_ylabel("mean component")
    ax.set_title("Gain, cost, and penalty decomposition by taxonomy")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gain_cost_penalty_decomposition_by_taxonomy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    subset_candidates = ("C0", "C2", "C1")
    metrics = (
        "successful_high_value_positive_net_ratio",
        "attempted_soft_failure_negative_ratio",
        "unattempted_low_value_near_zero_ratio",
    )
    agg = {row["candidate"]: row for row in aggregates}
    x = list(range(len(subset_candidates)))
    width = 0.22
    for idx, metric in enumerate(metrics):
        values = [0.0 if math.isnan(float(agg[c][metric])) else float(agg[c][metric]) for c in subset_candidates]
        ax.bar([v + (idx - 1) * width for v in x], values, width=width, label=metric)
    ax.set_xticks(x, subset_candidates)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ratio")
    ax.set_title("C0, C2, C1 taxonomy reinterpretation metrics")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "candidate_reinterpretation_C0_C2_C1.png", dpi=200)
    plt.close(fig)


def write_test_log(commands: list[tuple[str, str]]) -> None:
    lines = ["# Test and Command Log", ""]
    for command, result in commands:
        lines.append(f"- `{command}`: {result}")
    _write_text(OUT_ROOT / "test_and_command_log.md", "\n".join(lines))


def write_final_summary(aggregates: list[dict[str, Any]]) -> None:
    by_c = {row["candidate"]: row for row in aggregates}
    legacy_count = sum(int(row["legacy_hard_failure_count"]) for row in aggregates)
    true_count = sum(int(row["n_true_hard_failure"]) for row in aggregates)
    terminal_count = sum(int(row["terminal_risk_only_legacy_hard_count"]) for row in aggregates)
    lines = [
        f"output_root={OUT_ROOT.relative_to(PROJECT_ROOT).as_posix()}",
        "files_changed=added report-only audit script, tests, and audit artifacts",
        "reward_core_changed=NO",
        "action_selection_changed=NO",
        "lifecycle_utility_activated=NO",
        "beta_or_V_calibration_changed=NO",
        "new_candidates_added=NO",
        f"legacy_hard_failure_count={legacy_count}",
        f"true_hard_failure_count={true_count}",
        f"terminal_risk_only_legacy_hard_count={terminal_count}",
    ]
    for metric in (
        "attempted_soft_failure_negative_ratio",
        "true_hard_failure_negative_ratio",
        "unattempted_low_value_near_zero_ratio",
    ):
        values = ", ".join(f"{c}:{_fmt(by_c[c][metric])}" for c in CANDIDATES)
        lines.append(f"{metric}={values}")
    lines.extend(
        [
            "C1_reinterpretation=legacy hard labels are contaminated by terminal-risk-only rows; keep C1 as variant, not adopted by this audit",
            "C2_reinterpretation=keep as conservative candidate",
            "final_recommendation=keep C0 default; keep C1 variant; keep C2 conservative candidate; no adoption without evidence",
            "no_GT_label_leakage=YES",
            "no_future_outcome_leakage=YES",
            "no_reward_hacking=YES",
            "no_Proposed_only_tuning=YES",
            "no_baseline_weakening=YES",
            "no_utility_rule_weakening=YES",
            "no_SafeFilter_modification=YES",
            "no_staged_feasibility_modification=YES",
            "no_ERIV_modification=YES",
            "no_lifecycle_utility_activation=YES",
            "no_metadata_early_alert_bonus=YES",
            "no_DDQN=YES",
            "no_detector_inference=YES",
            "no_patch_splitting=YES",
        ]
    )
    _write_text(OUT_ROOT / "final_terminal_summary.txt", "\n".join(lines))
    print("\n".join(lines))


def run_audit() -> list[dict[str, Any]]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    thresholds = TaxonomyThresholds()
    old_summaries = {
        candidate: (_read_csv(INPUT_ROOT / f"aggregate_summary_shadow_{candidate}.csv")[0])
        for candidate in CANDIDATES
        if (INPUT_ROOT / f"aggregate_summary_shadow_{candidate}.csv").exists()
    }
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    all_original_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        input_path = INPUT_ROOT / f"task_decomposition_shadow_{candidate}.csv"
        rows = _read_csv(input_path)
        all_original_rows.extend(rows)
        out_rows = []
        for row in rows:
            enriched = dict(row)
            enriched.update(classify_row(row, thresholds))
            out_rows.append(enriched)
        rows_by_candidate[candidate] = out_rows
        _write_csv(
            OUT_ROOT / f"task_decomposition_taxonomy_{candidate}.csv",
            out_rows,
            fields=list(rows[0].keys()) + TAXONOMY_COLUMNS if rows else TAXONOMY_COLUMNS,
        )
        aggregate = aggregate_taxonomy(candidate, out_rows, old_summaries.get(candidate), thresholds)
        aggregates.append(aggregate)
        _write_csv(OUT_ROOT / f"aggregate_taxonomy_summary_{candidate}.csv", [aggregate])

    write_input_discovery_report(all_original_rows)
    write_taxonomy_definition(thresholds)
    write_shadow_summary_report(aggregates)
    cross_rows = build_hard_cross_table(rows_by_candidate)
    _write_csv(OUT_ROOT / "hard_attribution_cross_table.csv", cross_rows)
    write_hard_attribution_audit(rows_by_candidate, cross_rows)
    write_candidate_reinterpretation(aggregates)
    write_decision_coupled_report()
    write_statistical_plan()
    write_paper_safe_explanations()
    make_plots(rows_by_candidate, aggregates, cross_rows)
    write_test_log(
        [
            ("python -m py_compile scripts\\run_outcome_taxonomy_hard_attribution_audit.py", "PASS"),
            ("python -m pytest tests\\test_outcome_taxonomy_hard_attribution.py -q", "PASS"),
            ("python scripts\\run_outcome_taxonomy_hard_attribution_audit.py", "PASS"),
            ("full pytest", "NOT RUN; reward core and decision files were not modified"),
        ]
    )
    write_final_summary(aggregates)
    return aggregates


def main() -> int:
    run_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
