"""Audit the v2_1 product requirement to Module 2 candidate scoring path."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "_product_requirement_aware_module2_ab"
REPORT_PATH = OUT_DIR / "causal_path_audit.md"


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


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _module2_consumes_shadow() -> bool:
    token = "recommended_payload_plan_shadow"
    module2_paths = [
        "src/routers/safe_linucb_router.py",
        "src/routers/rule_router.py",
        "src/routers/safe_filter.py",
        "src/simulator/step_runner.py",
        "src/actions/action_space.py",
    ]
    for relative in module2_paths:
        path = PROJECT_ROOT / relative
        if path.exists() and token in path.read_text(encoding="utf-8", errors="replace"):
            return True
    return False


def _action_change_rate(trace_rows: list[dict[str, str]]) -> float:
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for row in trace_rows:
        key = (row.get("seed", ""), row.get("task_id", ""))
        arm = row.get("arm", "")
        if arm in {"v21_no_scoring_ug_wx050", "v21_reqaware_ug_wx050"}:
            grouped.setdefault(key, {})[arm] = row.get("selected_action", "")
    total = 0
    changed = 0
    for values in grouped.values():
        if "v21_no_scoring_ug_wx050" not in values or "v21_reqaware_ug_wx050" not in values:
            continue
        total += 1
        changed += int(values["v21_no_scoring_ug_wx050"] != values["v21_reqaware_ug_wx050"])
    return _safe_div(changed, total)


def _candidate_examples(rows: list[dict[str, str]], limit: int = 12) -> str:
    chosen = [
        row
        for row in rows
        if row.get("arm") == "v21_reqaware_ug_wx050"
        and row.get("product_requirement_aware_enabled") in {"1", "1.0", "True", "true"}
    ][:limit]
    if not chosen:
        return "- No enabled candidate rows available yet."
    lines = [
        "| seed | task_id | requirement | candidate | payload | req_fit | deadline_fit | staged_fit | final_score |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in chosen:
        lines.append(
            "| {seed} | {task} | {req} | {action} | {payload} | {req_fit:.3f} | {deadline:.3f} | {staged:.3f} | {score:.3f} |".format(
                seed=row.get("seed", ""),
                task=row.get("task_id", ""),
                req=row.get("ideal_product_requirement", ""),
                action=row.get("candidate_action", row.get("action_name", "")),
                payload=row.get("candidate_payload", ""),
                req_fit=_to_float(row.get("requirement_fit_score")),
                deadline=_to_float(row.get("deadline_feasibility_score")),
                staged=_to_float(row.get("staged_delivery_feasibility_score")),
                score=_to_float(row.get("final_candidate_score", row.get("final_action_score"))),
            )
        )
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_rows = _read_csv(OUT_DIR / "per_candidate_score_trace.csv")
    task_rows = _read_csv(OUT_DIR / "per_task_action_trace.csv")
    enabled_rows = [
        row
        for row in candidate_rows
        if row.get("product_requirement_aware_enabled") in {"1", "1.0", "True", "true"}
    ]
    requirement_rows = [row for row in candidate_rows if row.get("ideal_product_requirement", "")]
    bonus_rows = [row for row in enabled_rows if abs(_to_float(row.get("product_requirement_aware_bonus"))) > 1.0e-12]
    payload_counts = Counter(row.get("candidate_payload", "") for row in enabled_rows)
    action_change = _action_change_rate(task_rows)
    score_delta_mean = mean(
        [
            _to_float(row.get("product_requirement_aware_score_shadow"))
            - _to_float(row.get("utility_guided_score_shadow"))
            for row in enabled_rows
        ]
    ) if enabled_rows else 0.0
    shadow_consumed = _module2_consumes_shadow()
    report = f"""# Causal path audit

## Static path

- `product_requirement_rule.mode=v2_1` writes `task.product_requirement` as the ideal product requirement.
- `StepRunner` copies `task.product_requirement` into `guidance_bundle.product_requirement_context.ideal_product_requirement`.
- `SafeLinUCBRouter` computes `requirement_fit_score`, `model_fit_score`, `deadline_feasibility_score`, and `staged_delivery_feasibility_score` for each safe candidate action.
- `final_action_score` is the score used for selection; product-aware terms are added only when `module2_product_requirement_aware_scoring.enabled=true` and at least one weight is nonzero.
- SafeFilter remains upstream of scoring, so the new branch only ranks already-safe candidates.

## Shadow guardrail

- Module 2 consumes `recommended_payload_plan_shadow`: {str(shadow_consumed).upper()}

## Runtime evidence from A/B traces

- candidate rows: {len(candidate_rows)}
- candidate rows with ideal_product_requirement: {len(requirement_rows)}
- product-aware enabled candidate rows: {len(enabled_rows)}
- enabled candidate rows with nonzero product-aware bonus: {len(bonus_rows)}
- mean product-aware score delta over utility-guided shadow: {score_delta_mean:.6f}
- selected_action change rate, C vs B reconstructed from per_task_action_trace: {action_change:.6f}
- enabled candidate payload distribution: {dict(sorted(payload_counts.items()))}

## Candidate examples

{_candidate_examples(candidate_rows)}

## Interpretation

If enabled candidate rows and nonzero bonuses are present, v2_1 reached Module 2 candidate scoring. If selected_action change rate is above zero, the requirement-aware term had causal influence on final action selection. If the change rate is zero despite nonzero bonuses, the branch is connected but too weak relative to utility-guided score scale or candidate constraints.
"""
    REPORT_PATH.write_text(report.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
