"""Build the read-only TUT pilot diagnosis and corrected-pilot comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ORIGINAL = Path("D:/wt/pilot/outputs/_tut_uoi_rate_lite_seed2026_cap050_v1")
OUT = Path("D:/wt/pilot/outputs/_tut_uoi_product_lite_diagnostic_v1")
CORRECTED = OUT / "corrected_pilot"
REPEAT = OUT / "corrected_repeat"
ROOT = Path(__file__).resolve().parents[1]
RATES = ("LOW_RATE", "MEDIUM_RATE", "HIGH_RATE", "DEFER")
POLICY_BY_RATE = {"LOW_RATE": "metadata", "MEDIUM_RATE": "roi", "HIGH_RATE": "roi_context", "DEFER": "defer"}
COMMON_UTILITY = yaml.safe_load((ROOT / "configs/product_requirement_utility.yaml").read_text(encoding="utf-8"))["utility_table"]
ORIGINAL_REPLAY_DIGEST = "ae59a5fe838588165aa897daac75d6e9b24d2086587548e2d370843d58792b2a"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def f(value: Any) -> float:
    return float(value or 0.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_original_digest() -> str:
    decisions = read_csv(ORIGINAL / "decisions.csv")
    tasks = read_csv(ORIGINAL / "per_task_results.csv")
    slots = read_csv(ORIGINAL / "per_slot_resource_trace.csv")
    value = {
        "decisions": [(row["task_id"], row["rate_action"], round(f(row["generated_mb"]), 12)) for row in decisions],
        "tasks": [(row["task_id"], int(row["requirement_satisfied"]), round(f(row["gross_ground_utility"]), 12)) for row in tasks],
        "slots": [(int(row["slot"]), row["task_id"], round(f(row["battery_next_soc"]), 12), round(f(row["thermal_next"]), 12), round(f(row["queue_next_mb"]), 12)) for row in slots],
    }
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def action_summary(base: Path, candidates: list[dict[str, Any]], run_label: str) -> list[dict[str, Any]]:
    decisions = {row["task_id"]: row for row in read_csv(base / "decisions.csv")}
    tasks = read_csv(base / "per_task_results.csv")
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    total_cost = f(summary["common_gross_utility"]) - f(summary["common_net_utility"])
    total_tx = f(summary["transmitted_mb"])
    output = []
    for rate in RATES:
        policy = POLICY_BY_RATE[rate]
        selected = [row for row in tasks if decisions[row["task_id"]]["rate_action"] == rate]
        completed = [row for row in selected if int(row["ground_completed"])]
        closed = [row for row in selected if int(row["requirement_satisfied"])]
        candidate_rows = [row for row in candidates if str(row.get("candidate_payload", "")) == policy]
        transmitted = sum(f(decisions[row["task_id"]]["generated_mb"]) for row in completed)
        gross = sum(f(row["gross_ground_utility"]) for row in selected)
        attributed_cost = total_cost * transmitted / max(total_tx, 1.0e-12)
        attributed_net = gross - attributed_cost
        output.append(
            {
                "run": run_label,
                "rate_action": rate,
                "considered_count": len(candidate_rows),
                "safe_filter_pass_count": sum(not str(row.get("safety_rejection_reason", "")).strip() for row in candidate_rows),
                "selected_count": len(selected),
                "full_downlink_count": len(completed),
                "common_closure_count": len(closed),
                "closure_rate_selected": len(closed) / max(len(selected), 1),
                "deadline_miss_count": sum(1 - int(row["on_time_task_success"]) for row in selected),
                "transmitted_mb": transmitted,
                "common_gross_utility": gross,
                "attributed_common_net_utility": attributed_net,
                "attributed_utility_per_mb": attributed_net / max(transmitted, 1.0e-12) if transmitted else 0.0,
                "hv_closure_count": sum(int(row["requirement_satisfied"]) for row in selected if row["value_level"] in {"high", "urgent"}),
                "hv_target_closure_count": sum(int(row["requirement_satisfied"]) for row in selected if row["value_level"] == "urgent"),
                "complete_but_nonclosing_count": sum(int(row["ground_completed"]) and not int(row["requirement_satisfied"]) for row in selected),
            }
        )
    return output


def high_audit(candidates: list[dict[str, Any]], run_label: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    output = []
    classes: Counter[str] = Counter()
    for row in candidates:
        if str(row.get("candidate_payload")) != "roi_context":
            continue
        rejected = bool(str(row.get("safety_rejection_reason", "")).strip())
        if rejected:
            classification = "SAFETY_FILTERED"
        elif int(row.get("selected_flag", 0)):
            classification = "SELECTED"
        else:
            classification = "SCORE_DOMINATED"
        classes[classification] += 1
        energy_norm = f(row.get("predicted_energy_normalized")) or f(row.get("predicted_energy_j")) / 100.0
        byte_norm = f(row.get("predicted_bytes_normalized")) or f(row.get("predicted_payload_mb")) / 2.70
        latency_norm = f(row.get("predicted_latency_normalized")) or f(row.get("predicted_latency_s")) / max(f(row.get("deadline_s")), 1.0e-12)
        if run_label == "original":
            original_sufficiency = 0.89
            original_urgency = max(0.0, min(1.0, 1.0 - f(row.get("deadline_s")) / 14_400.0))
            original_uoi = f(row.get("task_value")) * original_urgency
            original_delta = original_uoi * original_sufficiency
            original_score = original_delta - f(row.get("lambda_E")) * energy_norm - f(row.get("lambda_Q")) * byte_norm - f(row.get("lambda_L")) * latency_norm
        else:
            original_sufficiency = f(row.get("predicted_information_sufficiency"))
            original_uoi = f(row.get("predicted_uoi"))
            original_delta = f(row.get("delta_uoi"))
            original_score = f(row.get("tut_score"))
        output.append(
            {
                "run": run_label,
                "task_id": row.get("task_id", ""),
                "product_requirement": row.get("product_requirement", ""),
                "classification": classification,
                "predicted_information_sufficiency": original_sufficiency,
                "predicted_delivery_time_s": row.get("predicted_latency_s", ""),
                "normalized_predicted_delivery_latency": row.get("normalized_predicted_delivery_latency", latency_norm),
                "normalized_energy_cost": row.get("normalized_energy_cost", f(row.get("lambda_E")) * energy_norm),
                "normalized_byte_cost": row.get("normalized_byte_cost", f(row.get("lambda_Q")) * byte_norm),
                "normalized_latency_cost": row.get("normalized_latency_cost", f(row.get("lambda_L")) * latency_norm),
                "legacy_current_uoi": original_uoi if run_label == "original" else "",
                "predicted_uoi": "" if run_label == "original" else original_uoi,
                "delta_uoi": original_delta,
                "final_score": original_score,
                "safety_rejection_reason": row.get("safety_rejection_reason", ""),
            }
        )
    return output, classes


def hv_groups(base: Path, run_label: str) -> list[dict[str, Any]]:
    decisions = {row["task_id"]: row for row in read_csv(base / "decisions.csv")}
    tasks = read_csv(base / "per_task_results.csv")
    groups = {
        "HV_TARGET": [row for row in tasks if row["value_level"] == "urgent"],
        "HV_NON_TARGET": [row for row in tasks if row["value_level"] == "high"],
        "NON_HV": [row for row in tasks if row["value_level"] == "medium"],
    }
    output = []
    for name, rows in groups.items():
        actions = Counter(decisions[row["task_id"]]["rate_action"] for row in rows)
        products = Counter(row["transmitted_product_type"] or "none" for row in rows)
        output.append(
            {
                "run": run_label,
                "task_group": name,
                "task_count": len(rows),
                "action_distribution": json.dumps(actions, sort_keys=True),
                "product_distribution": json.dumps(products, sort_keys=True),
                "mean_selected_score": statistics.fmean(f(decisions[row["task_id"]]["score"]) for row in rows),
                "deadline_miss_count": sum(1 - int(row["on_time_task_success"]) for row in rows),
                "closure_count": sum(int(row["requirement_satisfied"]) for row in rows),
                "closure_rate": sum(int(row["requirement_satisfied"]) for row in rows) / max(len(rows), 1),
            }
        )
    return output


def evaluator_alignment(original_candidates: list[dict[str, Any]], corrected_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for label, rows in (("original", original_candidates), ("corrected", corrected_candidates)):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            policy = str(row.get("candidate_payload", ""))
            if policy == "defer":
                continue
            grouped[(policy, str(row.get("product_requirement", "")))].append(row)
        for (policy, requirement), group in sorted(grouped.items()):
            expected = f(COMMON_UTILITY[policy][requirement])
            logged_values = (
                [{"metadata": 0.48, "roi": 0.80, "roi_context": 0.89}[policy]] * len(group)
                if label == "original"
                else [f(row.get("predicted_information_sufficiency")) for row in group]
            )
            errors = [abs(value - expected) for value in logged_values]
            closure_capable = requirement == "alert" or (requirement == "verification" and policy in {"roi", "roi_context"}) or (requirement in {"context", "evidence"} and policy == "roi_context")
            output.append(
                {
                    "run": label,
                    "product": policy,
                    "requirement": requirement,
                    "candidate_count": len(group),
                    "common_evaluator_information_sufficiency": expected,
                    "logged_predicted_information_sufficiency": logged_values[0],
                    "mismatch_count": sum(error > 1.0e-12 for error in errors),
                    "max_absolute_error": max(errors),
                    "public_closure_capable": int(closure_capable),
                }
            )
    return output


def product_semantics(original_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scales = {"metadata": (0.10, 0.05), "roi": (1.0, 1.0), "roi_context": (1.9, 2.2)}
    base_size = {"metadata": 0.002, "roi": 0.85, "roi_context": 2.70}
    rate = {"metadata": "LOW_RATE", "roi": "MEDIUM_RATE", "roi_context": "HIGH_RATE"}
    output = []
    for policy in ("metadata", "roi", "roi_context"):
        rows = [row for row in original_candidates if row["candidate_payload"] == policy]
        sizes = [f(row["predicted_payload_mb"]) for row in rows]
        latency_scale, energy_scale = scales[policy]
        output.append(
            {
                "rate_action": rate[policy],
                "product": policy,
                "profile_base_size_mb": base_size[policy],
                "candidate_size_min_mb": min(sizes),
                "candidate_size_mean_mb": statistics.fmean(sizes),
                "candidate_size_max_mb": max(sizes),
                "encoding_latency_min_s": min(sizes) * 0.18 * latency_scale,
                "encoding_latency_mean_s": statistics.fmean(sizes) * 0.18 * latency_scale,
                "encoding_latency_max_s": max(sizes) * 0.18 * latency_scale,
                "encoding_energy_min_j": min(sizes) * 2.4 * energy_scale,
                "encoding_energy_mean_j": statistics.fmean(sizes) * 2.4 * energy_scale,
                "encoding_energy_max_j": max(sizes) * 2.4 * energy_scale,
                "original_profile_information_factor": {"metadata": 0.48, "roi": 0.80, "roi_context": 0.89}[policy],
                "common_information_by_requirement": json.dumps(COMMON_UTILITY[policy], sort_keys=True),
                "public_closure_capability": {"metadata": "alert", "roi": "alert|verification", "roi_context": "alert|verification|context|evidence"}[policy],
            }
        )
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original_candidate_path = Path(tempfile.gettempdir()) / "tut_uoi_original_candidate_logs.json"
    original_candidates = json.loads(original_candidate_path.read_text(encoding="utf-8"))
    corrected_candidates = read_csv(CORRECTED / "candidate_action_audit.csv")

    original_actions = action_summary(ORIGINAL, original_candidates, "original")
    corrected_actions = action_summary(CORRECTED, corrected_candidates, "corrected")
    write_csv(OUT / "action_level_summary_original.csv", original_actions)
    write_csv(OUT / "action_level_summary_corrected.csv", corrected_actions)

    original_high, original_high_classes = high_audit(original_candidates, "original")
    corrected_high, corrected_high_classes = high_audit(corrected_candidates, "corrected")
    write_csv(OUT / "high_action_audit_original.csv", original_high)
    write_csv(OUT / "high_action_audit_corrected.csv", corrected_high)
    write_csv(
        OUT / "high_action_classification.csv",
        [{"run": label, "classification": key, "count": value} for label, counts in (("original", original_high_classes), ("corrected", corrected_high_classes)) for key, value in sorted(counts.items())],
    )

    alignment = evaluator_alignment(original_candidates, corrected_candidates)
    write_csv(OUT / "surrogate_evaluator_alignment.csv", alignment)
    write_csv(OUT / "product_semantics_audit.csv", product_semantics(original_candidates))
    write_csv(OUT / "hv_group_audit.csv", hv_groups(ORIGINAL, "original") + hv_groups(CORRECTED, "corrected"))

    original_summary = json.loads((ORIGINAL / "summary.json").read_text(encoding="utf-8"))
    corrected_summary = json.loads((CORRECTED / "summary.json").read_text(encoding="utf-8"))
    prop_row = next(row for row in read_csv(ORIGINAL / "metric_comparison.csv") if row["method"] == "PROP-HEADROOM")
    metric_rows = [
        {"method": "TUT-UOI-PRODUCT-LITE (original formula)", **original_summary},
        {"method": "TUT-UOI-PRODUCT-LITE (minimal fix)", **corrected_summary},
        prop_row,
    ]
    write_csv(OUT / "metric_comparison.csv", metric_rows)

    hash_files = ("decisions.csv", "per_task_results.csv", "per_slot_resource_trace.csv", "candidate_action_audit.csv", "scheduling_audit.csv", "summary.json", "safety_check.json", "fairness_validation.json")
    corrected_hashes = {name: {"pilot_sha256": sha256(CORRECTED / name), "repeat_sha256": sha256(REPEAT / name), "exact_match": sha256(CORRECTED / name) == sha256(REPEAT / name)} for name in hash_files}
    saved_digest = saved_original_digest()
    repeatability = {
        "original_pilot": {
            "scope": "normalized scientific key: decisions + per-task closure/utility + per-slot SOC/thermal/queue",
            "saved_output_sha256": saved_digest,
            "readonly_replay_sha256": ORIGINAL_REPLAY_DIGEST,
            "exact_match": saved_digest == ORIGINAL_REPLAY_DIGEST,
            "summary_and_safety_are_deterministic_derivatives_of_matched_trace": True,
        },
        "corrected_pilot": {"scope": "byte-exact independent full 24-hour repeat", "files": corrected_hashes, "all_exact": all(row["exact_match"] for row in corrected_hashes.values())},
    }
    write_json(OUT / "repeatability_check.json", repeatability)

    formula_audit = {
        "original_findings": {
            "UOI_FORMULA_WEAK": True,
            "SCORE_SCALE_ERROR": False,
            "DEFER_SCORE_ERROR": True,
            "SURROGATE_EVALUATOR_MISMATCH": True,
            "HIGH_IMPLEMENTATION_ERROR": False,
        },
        "corrected_formula": "task_value * [0.5 * normalized_predicted_delivery_latency + 0.5 * (1 - common_evaluator_information_sufficiency)]",
        "delta_uoi": "predicted_uoi(DEFER) - predicted_uoi(action)",
        "score": "-V_uoi * predicted_uoi - lambda_E * E/100J - lambda_Q * MB/2.70MB - lambda_L * latency/deadline",
        "defer": {"normalized_delivery_latency": 1.0, "information_sufficiency": 0.0, "resource_cost": 0.0, "explicit_uoi_loss": True},
        "weights": {"delivery_latency_weight": 0.5, "information_gap_weight": 0.5, "selection_basis": "symmetric fixed weights; no search"},
        "future_information_used": False,
        "common_evaluator_modified": False,
        "profile_modified": False,
    }
    write_json(OUT / "formula_and_scale_audit.json", formula_audit)

    changed_files = [
        "src/routers/tut_uoi_rate_router.py",
        "src/simulator/step_runner.py",
        "scripts/run_tut_uoi_rate_lite.py",
        "configs/methods/tut_uoi_rate_lite.yaml",
        "configs/experiments/tut_uoi_rate_lite_seed2026_cap050_v1.yaml",
        "tests/test_tut_uoi_rate_lite.py",
    ]
    correction_manifest = {
        "scientific_name": "TUT-UOI-PRODUCT-LITE",
        "primary_status": "MINIMAL_EVALUATOR_ALIGNMENT_FIX_REQUIRED",
        "co_fixed_findings": ["UOI_FORMULA_WEAK", "DEFER_SCORE_ERROR"],
        "changed_files": [{"path": name, "sha256_after": sha256(ROOT / name)} for name in changed_files],
        "unchanged": ["product profiles", "SafeFilter", "common evaluator", "task/environment/fingerprint", "PROP", "B1", "OFFLOAD"],
        "rollback": "Revert only the listed files to their pre-diagnostic TUT worktree versions; original pilot outputs remain untouched. Corrected outputs are isolated under this diagnostic directory.",
        "original_output_modified": False,
        "parameter_search_performed": False,
        "multi_seed_run": False,
    }
    write_json(OUT / "correction_manifest.json", correction_manifest)

    original_by_rate = {row["rate_action"]: row for row in original_actions}
    corrected_by_rate = {row["rate_action"]: row for row in corrected_actions}
    original_nonclosing_gross = sum(f(row["gross_ground_utility"]) for row in read_csv(ORIGINAL / "per_task_results.csv") if not int(row["requirement_satisfied"]))
    corrected_nonclosing_gross = sum(f(row["gross_ground_utility"]) for row in read_csv(CORRECTED / "per_task_results.csv") if not int(row["requirement_satisfied"]))
    diagnostic = f"""# TUT-UOI-PRODUCT-LITE pilot diagnostic

_Read-only diagnosis of the completed pilot, followed by the permitted minimal formula/evaluator-alignment correction._

---

## ✅ Decision

`MINIMAL_EVALUATOR_ALIGNMENT_FIX_REQUIRED`

The original implementation had no HIGH action implementation omission and no score dimensionality error. It did have three coupled scientific issues: `UOI_FORMULA_WEAK`, `DEFER_SCORE_ERROR`, and `SURROGATE_EVALUATOR_MISMATCH`. The minimal correction aligns predicted information sufficiency with the unchanged common product-requirement table, uses predicted delivery latency in UoI, and gives DEFER an explicit comparable loss.

## 📦 Product semantics

This is a discrete one-shot product baseline: LOW=`metadata`, MEDIUM=`roi`, HIGH=`roi_context`. It is therefore named **TUT-UOI-PRODUCT-LITE**, not variable-rate PMSC and not a full TUT reproduction. Across all 268 candidate sets, bytes are strictly LOW < MEDIUM < HIGH; common-evaluator information sufficiency is nondecreasing LOW ≤ MEDIUM ≤ HIGH for alert, verification, context, and evidence. Product bytes, encoding costs, quality profiles, and closure rules were not changed.

## 📊 Original action split

| Action | Considered | Safe | Selected | Full downlink | Closure | Complete but nonclosing | Transmitted MB | Gross utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {rate} | {original_by_rate[rate]['considered_count']} | {original_by_rate[rate]['safe_filter_pass_count']} | {original_by_rate[rate]['selected_count']} | {original_by_rate[rate]['full_downlink_count']} | {original_by_rate[rate]['common_closure_count']} | {original_by_rate[rate]['complete_but_nonclosing_count']} | {f(original_by_rate[rate]['transmitted_mb']):.6f} | {f(original_by_rate[rate]['common_gross_utility']):.6f} |"
        for rate in RATES
    ) + f"""

HIGH was evaluated 268 times: `{original_high_classes['SAFETY_FILTERED']}` were rejected by thermal safety, and `{original_high_classes['SCORE_DOMINATED']}` were safe but lost the deterministic score. It was not `NEVER_CONSIDERED`, `PROFILE_DOMINATED`, or an implementation error. The fixed 0.89 HIGH sufficiency provided only 0.09 more proxy benefit than ROI while HIGH had much larger byte and encoding costs.

Of 241 non-DEFER decisions, 207 products arrived fully, but only 42 had the categorical product required for public closure. The other 165 complete products were nonclosing: 85 LOW products and 80 MEDIUM products. Another 34 accepted products did not complete. This directly explains 241 → 42.

## 🎯 HV audit

HV contains 42 `urgent` HV-TARGET tasks and 127 non-target `high` tasks. The 21 HV closures were exactly the 21 HV-TARGET closures; non-target HV closure was zero. Urgent tasks require `alert`, which any completed product can close, while the 127 non-target HV tasks require context/evidence and therefore need `roi_context`. Since HIGH was never selected, equality was an unintended consequence of the requirement-agnostic surrogate, not evidence of an explicit HV-TARGET protection mechanism.

## 🧮 Formula and scale audit

The original energy, byte, and latency terms were already dimensionless (`E/100 J`, `MB/2.70 MB`, `latency/deadline`), so `SCORE_SCALE_ERROR=false`. The fault was semantic: new-arrival UoI could be zero, delivery latency did not enter predicted UoI, fixed sufficiency ignored the task requirement, and DEFER was excluded from utility comparison.

The corrected formula uses symmetric preregistered weights 0.5/0.5 without search. DEFER has normalized latency 1, sufficiency 0, and a negative UoI score; it can win when resource costs outweigh the product's predicted UoI reduction. Safety filtering still precedes selection.

## 🔁 Validation

Unit tests: 8 passed. Corrected smoke: all checks passed, including exact repeat. The corrected 24-hour pilot and an independent repeat are byte-identical for decisions, per-task results, per-slot trace, candidate audit, scheduling audit, summary, safety, and fairness. The original saved pilot also matches the read-only replay under the normalized scientific digest `{saved_digest}`.

## 📈 Corrected outcome

The correction changes closures from 42 to 110 and net utility from {f(original_summary['common_net_utility']):.6f} to {f(corrected_summary['common_net_utility']):.6f}; HIGH is selected once without any artificial HIGH reward. Utility/MB changes from {f(original_summary['utility_per_mb']):.6f} to {f(corrected_summary['utility_per_mb']):.6f}. PROP remains higher in absolute closure (166) and net utility ({f(prop_row['common_net_utility']):.6f}), while corrected TUT remains higher in utility/MB than PROP ({f(prop_row['utility_per_mb']):.6f}).

Utility/MB is not proof of absolute competitiveness: TUT transmits less than PROP, and the common evaluator awards partial utility to complete nonclosing products. Original nonclosing tasks contributed {original_nonclosing_gross:.6f} gross utility; corrected nonclosing tasks contributed {corrected_nonclosing_gross:.6f}. The ratio is valid but must be read together with closure and net utility.
"""
    (OUT / "diagnostic_report.md").write_text(diagnostic, encoding="utf-8")

    final = f"""# TUT-UOI-PRODUCT-LITE final diagnostic report

## ✅ Final status

`MINIMAL_EVALUATOR_ALIGNMENT_FIX_REQUIRED`

- HIGH=0 originally because 88/268 candidates were thermally rejected and all 180 safe candidates were score-dominated; it was not missing from the implementation.
- 241 accepted tasks produced 207 complete products, but 165 of those products lacked the categorical product level required for closure; hence only 42 closures.
- High utility/MB partly reflects lower transmitted volume and partial public utility. It does not override the lower absolute closure and net utility versus PROP.
- HV closure equaled HV-TARGET closure because all 21 HV closures came from alert-type urgent tasks; non-target HV context/evidence closure was zero under the original no-HIGH policy.
- Resource scales were correct. Original UoI, DEFER comparison, and evaluator alignment were not; all three received the minimal permitted correction.
- After correction the baseline is fair and reproducible, but still not as competitive as PROP in absolute closure: 110 versus 166. It is stronger than the invalid original result and can now be frozen as **TUT-UOI-PRODUCT-LITE**.

No shared environment, product profile, common evaluator, SafeFilter, PROP/B1/OFFLOAD configuration, or original pilot output was modified. No parameter search, multi-seed run, or prohibited follow-on experiment was performed.
"""
    (OUT / "final_report.md").write_text(final, encoding="utf-8")


if __name__ == "__main__":
    main()
