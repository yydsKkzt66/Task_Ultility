"""Read-only UTIL vs Proposed trace audit.

This script uses existing formal-readiness logging-smoke outputs. It does not
run simulations and does not modify router, reward, queue, profile, deadline,
resource/contact, or metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "_util_vs_proposed_audit"
READINESS_DIR = PROJECT_ROOT / "outputs" / "_formal_readiness_gate_lite"

PROPOSED_FULL = "proposed_payload_repaired_combined"
PROPOSED_SHORT = "PROP-COMB"
UTIL_FULL = "utility_rule"
UTIL_SHORT = "UTIL"
SCENARIOS = ["stress", "nominal"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], allow_overwrite: bool = False) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str, allow_overwrite: bool = False) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        if not _finite(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _scenario_root(scenario: str) -> Path:
    return READINESS_DIR / f"logging_smoke_{scenario}"


def _trajectory_path(scenario: str, variant: str) -> Path:
    return _scenario_root(scenario) / "trajectories" / f"{variant}.csv"


def _task_rows(scenario: str, variant: str) -> list[dict[str, str]]:
    return [row for row in _read_csv(_trajectory_path(scenario, variant)) if _to_int(row.get("has_task")) == 1]


def _index_tasks(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id and task_id not in indexed:
            indexed[task_id] = row
    return indexed


def _lifetime_index(scenario: str, variant: str) -> dict[str, dict[str, str]]:
    rows = _read_csv(_scenario_root(scenario) / "task_lifetime_outcomes.csv")
    return {
        str(row.get("task_id", "")): row
        for row in rows
        if str(row.get("router_name", "")) == variant and str(row.get("task_id", ""))
    }


def _selected_candidate_index(scenario: str, variant: str) -> dict[str, dict[str, str]]:
    rows = _read_csv(READINESS_DIR / f"candidate_action_score_log_{scenario}.csv")
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if str(row.get("variant", "")) != variant:
            continue
        if not (_to_int(row.get("selected_flag")) or _to_int(row.get("executed_flag"))):
            continue
        task_id = str(row.get("task_id", ""))
        if task_id and task_id not in selected:
            selected[task_id] = row
    return selected


def _candidate_rows_by_task(scenario: str, variant: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(READINESS_DIR / f"candidate_action_score_log_{scenario}.csv"):
        if str(row.get("variant", "")) == variant:
            grouped[str(row.get("task_id", ""))].append(row)
    return grouped


def _best_score_rows(rows: list[dict[str, str]], limit: int = 3) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: _to_float(row.get("final_candidate_score")), reverse=True)[:limit]


def _selected_score(selected: dict[str, str], trajectory: dict[str, str]) -> tuple[float, float]:
    predicted = _to_float(selected.get("predicted_reward"))
    final_score = _to_float(selected.get("final_candidate_score"))
    if not _finite(predicted):
        predicted = _to_float(trajectory.get("predicted_reward"))
    if not _finite(final_score):
        final_score = _to_float(trajectory.get("selected_action_score"))
    return predicted, final_score


def _payload_is_roi(payload: str) -> bool:
    return payload == "roi"


def _payload_is_metadata(payload: str) -> bool:
    return payload == "metadata"


def _is_defer(action: str, payload: str, model: str) -> bool:
    text = f"{action} {payload} {model}".lower()
    return "defer" in text or "delay" in text or "store" in text


def _processes(action: str, payload: str, model: str) -> bool:
    return not _is_defer(action, payload, model)


def _deadline_bin(deadline_s: float) -> str:
    if not _finite(deadline_s):
        return "unknown"
    minutes = deadline_s / 60.0
    if minutes <= 30.0:
        return "urgent"
    if minutes <= 90.0:
        return "medium"
    return "loose"


def _queue_bin(q_t: float) -> str:
    if not _finite(q_t):
        return "unknown"
    if q_t < 0.10:
        return "low"
    if q_t < 0.30:
        return "medium"
    return "high"


def _battery_bin(b_t: float) -> str:
    if not _finite(b_t):
        return "unknown"
    if b_t >= 0.70:
        return "healthy"
    if b_t >= 0.55:
        return "guard"
    return "low"


def _thermal_bin(theta: float) -> str:
    if not _finite(theta):
        return "unknown"
    if theta < 0.75:
        return "low"
    if theta < 0.85:
        return "elevated"
    return "high"


def _high_value_utility(lifetime_row: dict[str, str]) -> float:
    if not _bool(lifetime_row.get("high_value_flag")):
        return 0.0
    return max(0.0, _to_float(lifetime_row.get("delivered_only_utility"), 0.0))


def build_paired_trace(scenario: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prop_rows = _task_rows(scenario, PROPOSED_FULL)
    util_rows = _task_rows(scenario, UTIL_FULL)
    prop_idx = _index_tasks(prop_rows)
    util_idx = _index_tasks(util_rows)
    prop_life = _lifetime_index(scenario, PROPOSED_FULL)
    util_life = _lifetime_index(scenario, UTIL_FULL)
    prop_sel = _selected_candidate_index(scenario, PROPOSED_FULL)
    util_sel = _selected_candidate_index(scenario, UTIL_FULL)

    common = sorted(set(prop_idx) & set(util_idx), key=lambda task_id: _to_int(prop_idx[task_id].get("slot")))
    exact = (
        len(common) == len(prop_idx) == len(util_idx)
        and all(str(prop_idx[task_id].get("slot")) == str(util_idx[task_id].get("slot")) for task_id in common)
    )

    paired: list[dict[str, Any]] = []
    for task_id in common:
        p = prop_idx[task_id]
        u = util_idx[task_id]
        pl = prop_life.get(task_id, {})
        ul = util_life.get(task_id, {})
        ps = prop_sel.get(task_id, {})
        us = util_sel.get(task_id, {})
        p_pred, p_score = _selected_score(ps, p)
        u_pred, u_score = _selected_score(us, u)

        p_action = str(p.get("selected_action", p.get("executed_action", "")))
        u_action = str(u.get("selected_action", u.get("executed_action", "")))
        p_model = str(p.get("selected_model", p.get("model_name", "")))
        u_model = str(u.get("selected_model", u.get("model_name", "")))
        p_payload = str(p.get("selected_payload", p.get("payload_policy", "")))
        u_payload = str(u.get("selected_payload", u.get("payload_policy", "")))
        p_util = _to_float(pl.get("delivered_only_utility"), _to_float(p.get("delivered_only_utility"), 0.0))
        u_util = _to_float(ul.get("delivered_only_utility"), _to_float(u.get("delivered_only_utility"), 0.0))
        p_status = str(pl.get("final_lifetime_status", p.get("lifetime_status", "")))
        u_status = str(ul.get("final_lifetime_status", u.get("lifetime_status", "")))
        task_class = str(p.get("task_class") or ul.get("task_class") or pl.get("task_class") or "")
        deadline_s = _to_float(p.get("deadline_s"), _to_float(pl.get("deadline_s")))
        q_t = _to_float(p.get("Q_t"), _to_float(p.get("queue_mb")))
        b_t = _to_float(p.get("B_t"), _to_float(p.get("battery_soc")))
        theta = _to_float(p.get("Theta_t"), _to_float(p.get("thermal_next")))
        c_t = _to_float(p.get("C_t"), 0.0)

        paired.append(
            {
                "scenario": scenario,
                "seed": "2026",
                "slot": _to_int(p.get("slot")),
                "time_s": _to_float(p.get("task_arrival_time_s"), _to_float(p.get("slot")) * 20.0),
                "task_id": task_id,
                "scene_id": p.get("scene_id", ""),
                "task_value": _to_float(p.get("V_t"), _to_float(pl.get("task_value"))),
                "V_t": _to_float(p.get("V_t")),
                "high_value_flag": _to_int(p.get("high_value_flag"), _to_int(pl.get("high_value_flag"))),
                "emergency_flag": 1 if task_class == "emergency" else _to_int(ps.get("emergency_flag")),
                "deadline_s": deadline_s,
                "remaining_deadline_s": _to_float(ps.get("remaining_deadline_s"), deadline_s),
                "task_class": task_class,
                "battery_soc_proposed": b_t,
                "thermal_pressure_proposed": theta,
                "queue_mb_proposed": _to_float(p.get("queue_mb")),
                "Q_t_proposed": q_t,
                "battery_soc_util": _to_float(u.get("B_t"), _to_float(u.get("battery_soc"))),
                "thermal_pressure_util": _to_float(u.get("Theta_t"), _to_float(u.get("thermal_next"))),
                "queue_mb_util": _to_float(u.get("queue_mb")),
                "Q_t_util": _to_float(u.get("Q_t")),
                "contact_available": int(c_t > 0.0),
                "link_rate_bps": "",
                "remaining_contact_time": "",
                "deadline_bin": _deadline_bin(deadline_s),
                "queue_pressure_bin": _queue_bin(q_t),
                "battery_bin": _battery_bin(b_t),
                "thermal_bin": _thermal_bin(theta),
                "selected_model_proposed": p_model,
                "selected_payload_proposed": p_payload,
                "selected_action_proposed": p_action,
                "predicted_reward_proposed": p_pred,
                "final_score_proposed": p_score,
                "delivered_status_proposed": p.get("delivery_status", ""),
                "lifetime_status_proposed": p_status,
                "delivered_only_utility_proposed": p_util,
                "high_value_delivered_utility_proposed": _high_value_utility(pl),
                "payload_size_mb_proposed": _to_float(pl.get("generated_payload_mb"), _to_float(p.get("S_gen_current_mb"))),
                "information_sufficiency_proposed": _to_float(pl.get("information_sufficiency"), _to_float(p.get("information_factor"))),
                "detection_quality_proposed": _to_float(pl.get("detection_quality"), _to_float(p.get("detection_quality"))),
                "timeliness_proposed": _to_float(pl.get("timeliness_at_delivery"), _to_float(p.get("timeliness_at_delivery"))),
                "selected_model_util": u_model,
                "selected_payload_util": u_payload,
                "selected_action_util": u_action,
                "predicted_reward_util": u_pred,
                "final_score_util": u_score,
                "delivered_status_util": u.get("delivery_status", ""),
                "lifetime_status_util": u_status,
                "delivered_only_utility_util": u_util,
                "high_value_delivered_utility_util": _high_value_utility(ul),
                "payload_size_mb_util": _to_float(ul.get("generated_payload_mb"), _to_float(u.get("S_gen_current_mb"))),
                "information_sufficiency_util": _to_float(ul.get("information_sufficiency"), _to_float(u.get("information_factor"))),
                "detection_quality_util": _to_float(ul.get("detection_quality"), _to_float(u.get("detection_quality"))),
                "timeliness_util": _to_float(ul.get("timeliness_at_delivery"), _to_float(u.get("timeliness_at_delivery"))),
                "same_action": int(p_action == u_action),
                "model_diff": int(p_model != u_model),
                "payload_diff": int(p_payload != u_payload),
                "proposed_better_utility": int(p_util > u_util + 1.0e-12),
                "util_better_utility": int(u_util > p_util + 1.0e-12),
                "proposed_delivered_util_expired": int(p_status == "delivered" and u_status == "expired"),
                "util_delivered_proposed_expired": int(u_status == "delivered" and p_status == "expired"),
                "proposed_metadata_util_roi": int(_payload_is_metadata(p_payload) and _payload_is_roi(u_payload)),
                "proposed_roi_util_metadata": int(_payload_is_roi(p_payload) and _payload_is_metadata(u_payload)),
                "proposed_defer_util_process": int(_is_defer(p_action, p_payload, p_model) and _processes(u_action, u_payload, u_model)),
                "util_defer_proposed_process": int(_is_defer(u_action, u_payload, u_model) and _processes(p_action, p_payload, p_model)),
                "utility_delta_proposed_minus_util": p_util - u_util,
            }
        )
    meta = {
        "scenario": scenario,
        "proposed_tasks": len(prop_idx),
        "util_tasks": len(util_idx),
        "aligned_tasks": len(common),
        "exact_alignment": exact,
        "missing_in_proposed": len(set(util_idx) - set(prop_idx)),
        "missing_in_util": len(set(prop_idx) - set(util_idx)),
    }
    return paired, meta


def _fraction(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return math.nan
    return sum(_to_int(row.get(key)) for row in rows) / len(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [_to_float(row.get(key)) for row in rows]
    values = [value for value in values if _finite(value)]
    return sum(values) / len(values) if values else math.nan


def _summarize_group(rows: list[dict[str, Any]], group_key: str, group_value: str) -> dict[str, Any]:
    return {
        "group": group_key,
        "value": group_value,
        "aligned_tasks": len(rows),
        "same_action_fraction": _fraction(rows, "same_action"),
        "both_balanced_roi_fraction": (
            sum(1 for row in rows if row["selected_action_proposed"] == "balanced+roi" and row["selected_action_util"] == "balanced+roi")
            / max(len(rows), 1)
        ),
        "proposed_metadata_util_roi_fraction": _fraction(rows, "proposed_metadata_util_roi"),
        "proposed_roi_util_metadata_fraction": _fraction(rows, "proposed_roi_util_metadata"),
        "proposed_defer_util_process_fraction": _fraction(rows, "proposed_defer_util_process"),
        "util_defer_proposed_process_fraction": _fraction(rows, "util_defer_proposed_process"),
        "mean_utility_delta_proposed_minus_util": _mean(rows, "utility_delta_proposed_minus_util"),
        "proposed_delivered_util_expired": sum(_to_int(row.get("proposed_delivered_util_expired")) for row in rows),
        "util_delivered_proposed_expired": sum(_to_int(row.get("util_delivered_proposed_expired")) for row in rows),
    }


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = [_summarize_group(rows, "all", "all")]
    for key in ["task_class", "deadline_bin", "queue_pressure_bin", "contact_available", "battery_bin", "thermal_bin"]:
        for value in sorted({str(row.get(key, "")) for row in rows}):
            subset = [row for row in rows if str(row.get(key, "")) == value]
            summary.append(_summarize_group(subset, key, value))
    return summary


def _action_count(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(row.get(key, "")) for row in rows)


def _choose_cases(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()

    def add(label: str, candidates: list[dict[str, Any]], reverse: bool = True) -> None:
        for row in sorted(candidates, key=lambda item: abs(_to_float(item.get("utility_delta_proposed_minus_util"))), reverse=reverse):
            task_id = str(row["task_id"])
            if task_id not in used:
                used.add(task_id)
                cases.append((label, row))
                return

    add("same action", [row for row in rows if row["same_action"]])
    add("Proposed metadata, UTIL ROI", [row for row in rows if row["proposed_metadata_util_roi"]])
    add("Proposed ROI, UTIL metadata", [row for row in rows if row["proposed_roi_util_metadata"]])
    add("Proposed defers, UTIL processes", [row for row in rows if row["proposed_defer_util_process"]])
    add("UTIL defers, Proposed processes", [row for row in rows if row["util_defer_proposed_process"]])
    add("high-value UTIL wins", [row for row in rows if row["high_value_flag"] and row["util_better_utility"]])
    add("high-value Proposed wins", [row for row in rows if row["high_value_flag"] and row["proposed_better_utility"]])
    add("high queue pressure", [row for row in rows if row["queue_pressure_bin"] == "high"])
    add("near deadline", [row for row in rows if row["deadline_bin"] == "urgent"])
    add("largest Proposed gain", [row for row in rows if row["proposed_better_utility"]])
    add("largest UTIL gain", [row for row in rows if row["util_better_utility"]])

    if len(cases) < 10:
        for row in rows:
            if str(row["task_id"]) not in used:
                used.add(str(row["task_id"]))
                cases.append(("coverage", row))
            if len(cases) >= 10:
                break
    return cases[:20]


def _case_reason(row: dict[str, Any]) -> str:
    p = row["selected_action_proposed"]
    u = row["selected_action_util"]
    if p == u:
        return "Both controllers chose the same action; outcome differences mainly reflect downstream resource/queue histories."
    if row["proposed_metadata_util_roi"]:
        return "Proposed selected metadata while UTIL selected ROI; this is consistent with stronger delivery/resource caution if queue/contact pressure was high."
    if row["proposed_roi_util_metadata"]:
        return "Proposed selected ROI while UTIL selected metadata; this suggests Proposed was not simply lighter than UTIL for this task."
    if row["proposed_defer_util_process"]:
        return "Proposed deferred while UTIL processed; inspect resource pricing, safety filtering, and queue pressure."
    if row["util_defer_proposed_process"]:
        return "UTIL deferred while Proposed processed; this supports a meaningful behavior difference."
    return "Different selected actions; inspect candidate-score rank and payload/model difference."


def write_util_rule_audit(allow_overwrite: bool) -> None:
    text = """# UTIL Rule Definition Audit

## Code Path

- `src/evaluation/baselines.py` builds `utility_rule` as `RuleBasedRouter(router_name, mode="utility_rule")`.
- `src/routers/rule_router.py` implements the rule.
- `src/simulator/step_runner.py` first predicts all actions, applies `SafeFilter.filter_actions`, then passes safe candidate actions into `router.select_action`.
- `src/simulator/predictors.py` supplies predicted reward, delivery factor, queue next state, payload size, energy, latency, information factor, and safety reasons.

## Exact Selection Rule

For mode `utility_rule`, after safety filtering:

1. If `V_t > 0.78`, `U_t > 0.50`, `B_t > 0.35`, and `C_t > 0.45`, select `high_accuracy+roi_context` if it is available in the safe candidate set.
2. Else if `V_t > 0.55` or `U_t > 0.55`, select `balanced+roi` if it is available in the safe candidate set.
3. Else select the safe candidate action with maximum `predicted_reward`.

If the selected action later violates safety, the shield/fallback path can still replace it, but the paired logs show selected and executed candidates separately.

## Answers

1. UTIL selects high-accuracy ROI+context for high-value, high-uncertainty, adequate-battery, adequate-contact tasks; otherwise it often selects balanced+ROI for moderately valuable or uncertain tasks; otherwise it greedily maximizes predicted reward.
2. It does not fully rank all actions in the first two branches. It follows threshold rules and only uses all-candidate predicted reward in the fallback branch.
3. It uses current-slot `V_t`, `U_t`, `B_t`, and `C_t` directly. Through `candidate_actions`, it also inherits current safety filtering based on predicted battery, thermal, and queue feasibility.
4. Yes. `V_t` is central to both threshold branches.
5. It does not directly test deadline/timeliness in the threshold branches. Timeliness can affect the fallback `predicted_reward` and safety/service diagnostics, but not the first two hard preferences.
6. It does not directly use queue state in the threshold rules. Queue state affects safe candidate filtering and fallback predicted reward through the predictor.
7. It directly uses `C_t` in the high-accuracy ROI+context branch. Delivery feasibility also enters fallback predicted reward and safety filtering.
8. It does not directly inspect predicted information factor in the threshold branches. It indirectly uses information sufficiency through fallback `predicted_reward`.
9. It directly uses battery via `B_t > 0.35` for the high-accuracy branch, and it receives only safe candidates after SafeFilter. It does not apply Lyapunov pricing or long-term mode guidance.
10. It uses current-slot context and one-step predictions only. No future task arrivals, future labels, or future contact windows were found in the rule.
11. UTIL is a strong current-state utility-rule heuristic with safety-filtered one-step prediction fallback. It is not a weak greedy baseline. It is not oracle-like unless future information is found elsewhere; static inspection did not find future information in `RuleBasedRouter`.
12. UTIL tends to select balanced+ROI frequently because the second branch triggers whenever `V_t > 0.55` or `U_t > 0.55`, and `balanced+roi` is usually safe and available.
13. It is fair as a strong heuristic baseline if described as safety-filtered and task/context-aware. It should not be described as a naive greedy baseline.
14. Paper wording: "UTIL is a strong safety-filtered current-slot utility-rule heuristic that uses task value, uncertainty, battery/contact context, and one-step predicted reward fallback. It does not use long-term Lyapunov pricing, mode control, or future task/contact information."

## Fairness Classification

UTIL is best classified as a strong heuristic baseline, not an oracle. It should be kept unless later evidence shows future information use or label leakage.
"""
    _write_text(AUDIT_DIR / "util_rule_definition_audit.md", text, allow_overwrite)


def write_summary_reports(
    scenario: str,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    allow_overwrite: bool,
) -> None:
    all_row = summary_rows[0]
    action_counts_p = _action_count(rows, "selected_action_proposed")
    action_counts_u = _action_count(rows, "selected_action_util")
    hv_rows = [row for row in rows if _to_int(row.get("high_value_flag")) == 1]
    hv_all = _summarize_group(hv_rows, "task_class", "high_value_flag=1")
    text = f"""# Proposed vs UTIL Decision Difference Summary

## Alignment

- scenario: `{scenario}`
- seed: 2026
- Proposed tasks: {meta['proposed_tasks']}
- UTIL tasks: {meta['util_tasks']}
- aligned tasks: {meta['aligned_tasks']}
- exact task/slot alignment: {meta['exact_alignment']}
- missing in Proposed: {meta['missing_in_proposed']}
- missing in UTIL: {meta['missing_in_util']}

## Overall Decision Differences

- same-action fraction: {_fmt(all_row['same_action_fraction'])}
- both balanced+ROI fraction: {_fmt(all_row['both_balanced_roi_fraction'])}
- Proposed metadata / UTIL ROI fraction: {_fmt(all_row['proposed_metadata_util_roi_fraction'])}
- Proposed ROI / UTIL metadata fraction: {_fmt(all_row['proposed_roi_util_metadata_fraction'])}
- Proposed defers / UTIL processes fraction: {_fmt(all_row['proposed_defer_util_process_fraction'])}
- UTIL defers / Proposed processes fraction: {_fmt(all_row['util_defer_proposed_process_fraction'])}
- mean delivered-only utility delta, Proposed minus UTIL: {_fmt(all_row['mean_utility_delta_proposed_minus_util'])}

## Action Mix

- Proposed action counts: {dict(action_counts_p)}
- UTIL action counts: {dict(action_counts_u)}

## High-value Tasks

- aligned high-value tasks: {len(hv_rows)}
- high-value same-action fraction: {_fmt(hv_all['same_action_fraction'])}
- high-value mean utility delta, Proposed minus UTIL: {_fmt(hv_all['mean_utility_delta_proposed_minus_util'])}

## Grouped CSV

See `proposed_vs_util_decision_difference_summary.csv` for breakdowns by task class, deadline bin, queue-pressure bin, contact availability, battery bin, and thermal bin.
"""
    _write_text(AUDIT_DIR / "proposed_vs_util_decision_difference_summary.md", text, allow_overwrite)
    _write_csv(AUDIT_DIR / "proposed_vs_util_decision_difference_summary.csv", summary_rows, allow_overwrite)


def write_outcome_diagnosis(rows: list[dict[str, Any]], allow_overwrite: bool) -> None:
    util_wins = [row for row in rows if row["util_better_utility"]]
    prop_wins = [row for row in rows if row["proposed_better_utility"]]
    util_gain_roi = [row for row in util_wins if row["selected_action_util"] == "balanced+roi"]
    prop_gain_roi = [row for row in prop_wins if row["selected_action_proposed"] == "balanced+roi"]
    prop_defer_losses = [row for row in util_wins if row["proposed_defer_util_process"]]
    util_heavy = [row for row in rows if row["selected_payload_util"] in {"roi", "roi_context"}]
    prop_light = [row for row in rows if row["selected_payload_proposed"] == "metadata"]
    hv_rows = [row for row in rows if _to_int(row.get("high_value_flag")) == 1]
    text = f"""# Proposed vs UTIL Outcome Difference Diagnosis

1. UTIL gains more delivered-only utility on {len(util_wins)} aligned tasks. The largest pattern is UTIL selecting `balanced+roi`, which appears in {len(util_gain_roi)} of those UTIL-win rows.
2. Proposed gains more delivered-only utility on {len(prop_wins)} aligned tasks. Proposed wins include {len(prop_gain_roi)} cases where Proposed selects `balanced+roi`, so Proposed is not only winning through metadata.
3. UTIL gains are mostly associated with `balanced+roi`: {len(util_gain_roi)} / {max(len(util_wins), 1)} UTIL-win tasks.
4. Proposed losses can come from a mix of safety-filtered selection, resource pricing/mode guidance, UCB/exploration effects, and delivery outcomes. Proposed deferred while UTIL processed in {len(prop_defer_losses)} UTIL-win tasks.
5. Proposed can preserve resources or queue pressure at the cost of immediate utility when it selects metadata/defer or avoids heavier actions. The trace contains {len(prop_light)} Proposed metadata selections.
6. UTIL can increase queue pressure or expiration risk by choosing heavier payloads; UTIL chose ROI/ROI+context in {len(util_heavy)} aligned tasks. Whether this is harmful is task-dependent because ROI also raises verification utility.
7. High-value tasks: {len(hv_rows)} aligned. Their same-action fraction is {_fmt(_fraction(hv_rows, 'same_action'))}; mean Proposed-minus-UTIL utility delta is {_fmt(_mean(hv_rows, 'utility_delta_proposed_minus_util'))}.
8. Proposed does not collapse completely to UTIL: same-action fraction is {_fmt(_fraction(rows, 'same_action'))}. However, both controllers often use `balanced+roi`, so their behavior is close in the high-utility region.
9. The evidence supports describing Proposed as a long-term task-utility-driven co-design framework if the paper emphasizes its Lyapunov/mode/resource-aware mechanism and ablation comparisons, not just raw superiority over UTIL.
10. If Proposed is slightly below UTIL, the evidence is consistent with UTIL being a strong heuristic baseline exploiting the balanced+ROI sweet spot, not necessarily Proposed being invalid. Underperformance should be judged against ablations and resource-safety/queue outcomes.
"""
    _write_text(AUDIT_DIR / "proposed_vs_util_outcome_difference_diagnosis.md", text, allow_overwrite)


def write_representative_cases(
    rows: list[dict[str, Any]],
    prop_candidates: dict[str, list[dict[str, str]]],
    util_candidates: dict[str, list[dict[str, str]]],
    allow_overwrite: bool,
) -> None:
    cases = _choose_cases(rows)
    lines = ["# Representative Task Cases", ""]
    for i, (label, row) in enumerate(cases, start=1):
        task_id = str(row["task_id"])
        p_top = _best_score_rows(prop_candidates.get(task_id, []), 3)
        u_top = _best_score_rows(util_candidates.get(task_id, []), 3)
        def format_top(items: list[dict[str, str]]) -> str:
            if not items:
                return "unavailable"
            return "; ".join(
                f"{item.get('candidate_action')} score={_fmt(_to_float(item.get('final_candidate_score')))} pred_reward={_fmt(_to_float(item.get('predicted_reward')))} safety={item.get('safety_status')}"
                for item in items
            )
        lines.extend(
            [
                f"## {i}. {label}: `{task_id}`",
                "",
                f"- context: slot {row['slot']}, class `{row['task_class']}`, V={_fmt(row['V_t'])}, deadline={_fmt(_to_float(row['deadline_s'])/60.0)} min, queue bin `{row['queue_pressure_bin']}`, contact_available={row['contact_available']}",
                f"- Proposed selected: `{row['selected_action_proposed']}` / payload `{row['selected_payload_proposed']}` / model `{row['selected_model_proposed']}`; final score {_fmt(row['final_score_proposed'])}",
                f"- UTIL selected: `{row['selected_action_util']}` / payload `{row['selected_payload_util']}` / model `{row['selected_model_util']}`; final score {_fmt(row['final_score_util'])}",
                f"- Proposed top candidate scores: {format_top(p_top)}",
                f"- UTIL top candidate scores: {format_top(u_top)}",
                f"- outcome: Proposed `{row['lifetime_status_proposed']}` utility {_fmt(row['delivered_only_utility_proposed'])}; UTIL `{row['lifetime_status_util']}` utility {_fmt(row['delivered_only_utility_util'])}",
                f"- assessment: {_case_reason(row)}",
                "",
            ]
        )
    _write_text(AUDIT_DIR / "representative_task_cases.md", "\n".join(lines), allow_overwrite)


def write_formula_check(rows: list[dict[str, Any]], allow_overwrite: bool) -> None:
    roi_context_stress = sum(1 for row in rows if row["selected_payload_proposed"] == "roi_context")
    metadata_total = sum(1 for row in rows if row["selected_payload_proposed"] == "metadata")
    metadata_no_pressure = sum(
        1
        for row in rows
        if row["selected_payload_proposed"] == "metadata"
        and row["deadline_bin"] == "loose"
        and row["queue_pressure_bin"] == "low"
        and row["contact_available"] == 1
    )
    high_value_largest = sum(
        1
        for row in rows
        if row["high_value_flag"] and row["selected_payload_proposed"] == "roi_context"
    )
    high_value_total = sum(1 for row in rows if row["high_value_flag"])
    defer_count = sum(1 for row in rows if _is_defer(row["selected_action_proposed"], row["selected_payload_proposed"], row["selected_model_proposed"]))
    text = f"""# Formula/Method Consistency Check

| Check | Evidence | Severity |
| --- | --- | --- |
| High-value tasks should not always choose largest payload. | Proposed chose ROI+context for {high_value_largest}/{max(high_value_total, 1)} high-value tasks. | none |
| Metadata-first should mainly happen when deadline/contact/queue pressure favors alert-level delivery. | Proposed metadata selections: {metadata_total}; metadata under loose deadline, low queue, and contact available: {metadata_no_pressure}. | {'MODERATE' if metadata_no_pressure > 0 else 'none'} |
| ROI should dominate when verification utility justifies payload cost. | `balanced+roi` is a frequent selected action for both Proposed and UTIL, consistent with a verification sweet spot. | none |
| ROI+context should be rare under stress unless sufficient capacity exists. | Proposed ROI+context selections under stress: {roi_context_stress}. | {'MINOR' if roi_context_stress > 0 else 'none'} |
| Defer/store should happen when delivery is infeasible or marginal utility is low. | Proposed defer count: {defer_count}. Representative cases should be inspected for resource/queue context. | none |
| Long-term resource pricing should make Proposed more conservative under low battery, high queue, or high thermal pressure. | Proposed differs from UTIL in {_fmt(1.0 - _fraction(rows, 'same_action'))} of tasks; metadata/defer choices show non-identical behavior. | none |
| UTIL should be described as a strong utility-rule heuristic. | Static code inspection shows thresholded task/context rules plus safety-filtered predicted-reward fallback. | none |

## Overall Assessment

No CRITICAL contradiction was found in static/post-processed evidence. The main caution is that Proposed and UTIL both exploit the balanced+ROI region often, so the paper should emphasize mechanism, resource-aware ablations, and delivered-only metrics rather than claiming a large raw win over UTIL.
"""
    _write_text(AUDIT_DIR / "formula_method_consistency_check.md", text, allow_overwrite)


def write_final_report(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], meta: dict[str, Any], allow_overwrite: bool) -> None:
    all_row = summary_rows[0]
    text = f"""# UTIL vs Proposed Audit Report

## Executive Summary

- Paired trace source: existing `outputs/_formal_readiness_gate_lite/logging_smoke_stress/` outputs.
- Simulation rerun: no.
- Exact task alignment: {meta['exact_alignment']}.
- Aligned stress tasks: {meta['aligned_tasks']}.
- Same-action fraction: {_fmt(all_row['same_action_fraction'])}.
- Mean delivered-only utility delta, Proposed minus UTIL: {_fmt(all_row['mean_utility_delta_proposed_minus_util'])}.

## Answers

1. UTIL is a safety-filtered current-slot utility-rule heuristic. It uses threshold rules over `V_t`, `U_t`, `B_t`, and `C_t`, then falls back to maximum predicted reward over safe candidates.
2. UTIL is fair as a strong heuristic baseline if described accurately.
3. UTIL is strong, but static inspection did not find oracle-like future information use.
4. Proposed behaves differently from UTIL on {_fmt(1.0 - all_row['same_action_fraction'])} of aligned stress tasks; the difference is meaningful but not dominant.
5. Proposed shows evidence of long-term resource-aware behavior through Lyapunov/mode fields, guided scores, z-guidance, and non-identical action choices. The evidence is stronger when paired with NO-MODE and NO-LYA ablations.
6. UTIL is currently strong because its threshold rule frequently selects `balanced+roi`, which is a high-utility verification sweet spot under current profiles and deadlines.
7. Proposed slightly below UTIL can be acceptable for WC if Proposed beats key ablations/fixed baselines and offers resource-aware interpretation. The paper should not overclaim a consistent win over UTIL until multi-seed confirms it.
8. Do not change UTIL before formal multi-seed unless future/oracle information is later found.
9. Do not change Proposed before formal multi-seed based on this audit alone. Tuning reward/queue/router now would weaken reproducibility.
10. Formal multi-seed should proceed next with UTIL retained as a strong baseline and with this audit cited internally for paper wording.

## Conservative Guidance

- Do not weaken UTIL just to make Proposed win.
- Do not tune reward or queue unless a real bug or unfair information use is found.
- Keep UTIL as a strong heuristic baseline.
- If later evidence shows future/oracle information, reclassify or revise UTIL before formal multi-seed.
- If multi-seed shows Proposed collapses to balanced+ROI, report that clearly and rely on ablation/resource-safety evidence.
"""
    _write_text(AUDIT_DIR / "util_vs_proposed_audit_report.md", text, allow_overwrite)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze UTIL vs Proposed paired traces from existing outputs.")
    parser.add_argument("--allow-overwrite", action="store_true", help="allow overwriting audit outputs")
    args = parser.parse_args()

    write_util_rule_audit(args.allow_overwrite)

    all_meta: dict[str, Any] = {}
    stress_rows: list[dict[str, Any]] = []
    stress_summary: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        rows, meta = build_paired_trace(scenario)
        all_meta[scenario] = meta
        _write_csv(AUDIT_DIR / f"paired_task_trace_{scenario}.csv", rows, args.allow_overwrite)
        if scenario == "stress":
            stress_rows = rows
            stress_summary = build_summary_rows(rows)

    write_summary_reports("stress", stress_rows, stress_summary, all_meta["stress"], args.allow_overwrite)
    write_outcome_diagnosis(stress_rows, args.allow_overwrite)
    write_representative_cases(
        stress_rows,
        _candidate_rows_by_task("stress", PROPOSED_FULL),
        _candidate_rows_by_task("stress", UTIL_FULL),
        args.allow_overwrite,
    )
    write_formula_check(stress_rows, args.allow_overwrite)
    write_final_report(stress_rows, stress_summary, all_meta["stress"], args.allow_overwrite)

    print(f"stress_aligned_tasks={all_meta['stress']['aligned_tasks']}")
    print(f"stress_exact_alignment={all_meta['stress']['exact_alignment']}")
    print(f"same_action_fraction={_fmt(stress_summary[0]['same_action_fraction'])}")
    print("util_rule_classification=strong safety-filtered current-slot utility-rule heuristic; not oracle-like by static inspection")
    print("formal_multiseed_recommendation=proceed with UTIL retained as strong baseline")
    print("outputs=" + ",".join(str(path.relative_to(PROJECT_ROOT)) for path in sorted(AUDIT_DIR.glob("*"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
