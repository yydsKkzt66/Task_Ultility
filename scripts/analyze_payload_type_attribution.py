"""Payload-type attribution and shadow progressive information diagnostics.

This script is read-only with respect to simulator outputs. It post-processes
existing nominal/stress BUPT-1 smoke-test artifacts and writes audit files under
outputs/_audit. It does not run simulations and does not affect policy,
reward, queue priority, or delivered-only metric definitions.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "_audit"
SHADOW_CONFIG = PROJECT_ROOT / "configs" / "shadow_progressive_information_metric.yaml"

SCENARIOS = {
    "nominal": PROJECT_ROOT / "outputs" / "sarfish_1day_smoke_bupt1_deadline_minutes_nominal",
    "stress": PROJECT_ROOT / "outputs" / "sarfish_1day_smoke_bupt1_deadline_minutes_stress",
}

KEY_VARIANTS = ["FIX-L", "UTIL", "PROP-COMB", "PROP-Z", "PROP-PAY", "PROP-UCB"]
PAYLOAD_TYPES = ["metadata", "roi", "roi_context", "defer_store", "other", "uncertain"]
CORE_PAYLOAD_TYPES = ["metadata", "roi", "roi_context"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
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


def _safe_div(numerator: float, denominator: float) -> float:
    if not _finite(numerator) or not _finite(denominator) or abs(denominator) <= 1.0e-12:
        return math.nan
    return numerator / denominator


def _fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        if not _finite(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def _boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    values = [value for value in values if _finite(value)]
    return statistics.fmean(values) if values else math.nan


def _median(values: list[float]) -> float:
    values = [value for value in values if _finite(value)]
    return statistics.median(values) if values else math.nan


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _variant_alias_map(output_dir: Path) -> dict[str, str]:
    rows = _read_csv(output_dir / "delivered_only_metrics.csv")
    aliases: dict[str, str] = {}
    for row in rows:
        full = str(row.get("variant_full", ""))
        short = str(row.get("variant_short", full))
        if full:
            aliases[full] = short
    return aliases


def _summary_map(output_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(output_dir / "delivered_only_metrics.csv")
    return {str(row.get("variant_full", "")): row for row in rows if row.get("variant_full")}


def _trajectory_index(output_dir: Path, variant: str) -> dict[str, dict[str, str]]:
    path = output_dir / "trajectories" / f"{variant}.csv"
    rows = _read_csv(path)
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if _to_int(row.get("has_task")) == 1 and task_id and task_id not in index:
            index[task_id] = row
    return index


def _normalize_payload_type(value: str) -> str:
    text = value.strip().lower().replace("-", "_").replace("+", "_")
    if text in {"metadata", "meta"}:
        return "metadata"
    if text in {"roi"}:
        return "roi"
    if text in {"roi_context", "roicontext", "context_roi"}:
        return "roi_context"
    if text in {"defer", "delay_store", "delay/store", "store", "none", "no_task"}:
        return "defer_store"
    if text in {"", "nan"}:
        return "uncertain"
    return "other"


def infer_payload_type(row: dict[str, str], trajectory_row: dict[str, str] | None = None) -> tuple[str, str]:
    for key in ("payload_policy", "selected_payload"):
        payload = _normalize_payload_type(str(row.get(key, "")))
        if payload != "uncertain":
            return payload, f"exact_{key}"
    if trajectory_row is not None:
        for key in ("payload_policy", "selected_payload"):
            payload = _normalize_payload_type(str(trajectory_row.get(key, "")))
            if payload != "uncertain":
                return payload, f"trajectory_{key}"
    action = str(row.get("selected_action", "") or (trajectory_row or {}).get("selected_action", "")).lower()
    if "metadata" in action:
        return "metadata", "inferred_selected_action"
    if "roi_context" in action:
        return "roi_context", "inferred_selected_action"
    if "roi" in action:
        return "roi", "inferred_selected_action"
    if "defer" in action or "delay" in action or "store" in action:
        return "defer_store", "inferred_selected_action"
    return "uncertain", "uncertain"


def _is_delivered_before_deadline(row: dict[str, str]) -> bool:
    return (
        str(row.get("final_lifetime_status", "")).strip().lower() == "delivered"
        and _boolish(row.get("met_deadline_lifetime"))
    )


def _is_high_value(row: dict[str, str], threshold: float = 0.75) -> bool:
    if _boolish(row.get("high_value_flag")):
        return True
    task_value = _to_float(row.get("task_value"))
    return _finite(task_value) and task_value >= threshold


def _utility(row: dict[str, str]) -> float:
    value = _to_float(row.get("delivered_before_deadline_utility"))
    if _finite(value):
        return value
    value = _to_float(row.get("delivered_only_utility"))
    return value if _finite(value) else 0.0


def _delivered_mb(row: dict[str, str], delivered: bool) -> float:
    if not delivered:
        return 0.0
    value = _to_float(row.get("generated_payload_mb"), 0.0)
    return max(0.0, value) if _finite(value) else 0.0


def _event_time(row: dict[str, str]) -> float:
    for key in ("payload_delivery_time_s", "delivered_slot", "arrival_slot"):
        value = _to_float(row.get(key))
        if _finite(value):
            return value
    return math.inf


def build_attribution_rows(scenario: str, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    aliases = _variant_alias_map(output_dir)
    summaries = _summary_map(output_dir)
    lifetime_rows = _read_csv(output_dir / "task_lifetime_outcomes.csv")
    trajectory_indexes: dict[str, dict[str, dict[str, str]]] = {}
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in lifetime_rows:
        variant = str(row.get("router_name", ""))
        if not variant:
            continue
        if variant not in trajectory_indexes:
            trajectory_indexes[variant] = _trajectory_index(output_dir, variant)
        trajectory_row = trajectory_indexes[variant].get(str(row.get("task_id", "")))
        payload_type, inference = infer_payload_type(row, trajectory_row)
        delivered = _is_delivered_before_deadline(row)
        high_value = _is_high_value(row)
        generated_mb = max(0.0, _to_float(row.get("generated_payload_mb"), 0.0))
        enriched = {
            **row,
            "_variant": variant,
            "_variant_short": aliases.get(variant, variant),
            "_payload_type": payload_type,
            "_payload_inference": inference,
            "_delivered_before_deadline": delivered,
            "_high_value": high_value,
            "_utility": _utility(row) if delivered else 0.0,
            "_generated_mb": generated_mb if _finite(generated_mb) else 0.0,
            "_delivered_mb": _delivered_mb(row, delivered),
            "_expired": str(row.get("final_lifetime_status", "")).strip().lower() == "expired",
        }
        by_variant[variant].append(enriched)

    rows_out: list[dict[str, Any]] = []
    for variant, rows in sorted(by_variant.items(), key=lambda item: aliases.get(item[0], item[0])):
        summary = summaries.get(variant, {})
        delivered_total = sum(float(row["_utility"]) for row in rows if row["_delivered_before_deadline"])
        high_value_total = sum(
            float(row["_utility"]) for row in rows if row["_delivered_before_deadline"] and row["_high_value"]
        )
        if summary:
            delivered_total = _to_float(summary.get("delivered_only_gross_utility"), delivered_total)
            high_value_total = _to_float(summary.get("high_value_delivered_utility"), high_value_total)
        for payload_type in PAYLOAD_TYPES:
            group = [row for row in rows if row["_payload_type"] == payload_type]
            delivered_group = [row for row in group if row["_delivered_before_deadline"]]
            hv_delivered_group = [row for row in delivered_group if row["_high_value"]]
            generated_mb = sum(float(row["_generated_mb"]) for row in group)
            delivered_mb = sum(float(row["_delivered_mb"]) for row in delivered_group)
            delivered_utility = sum(float(row["_utility"]) for row in delivered_group)
            hv_utility = sum(float(row["_utility"]) for row in hv_delivered_group)
            inference_counts = Counter(str(row["_payload_inference"]) for row in group)
            rows_out.append(
                {
                    "scenario": scenario,
                    "variant_short": aliases.get(variant, variant),
                    "variant_full": variant,
                    "payload_type": payload_type,
                    "payload_type_inference": ";".join(f"{k}:{v}" for k, v in sorted(inference_counts.items())) if group else "",
                    "generated_task_count": len(group),
                    "delivered_only_utility_total": delivered_total,
                    "high_value_delivered_utility_total": high_value_total,
                    "delivered_only_utility_by_payload_type": delivered_utility,
                    "high_value_delivered_utility_by_payload_type": hv_utility,
                    "delivered_task_count_by_payload_type": len(delivered_group),
                    "high_value_delivered_task_count_by_payload_type": len(hv_delivered_group),
                    "expired_task_count_by_payload_type": sum(1 for row in group if row["_expired"]),
                    "average_information_sufficiency_by_payload_type": _mean(
                        [_to_float(row.get("information_sufficiency")) for row in delivered_group]
                    ),
                    "median_information_sufficiency_by_payload_type": _median(
                        [_to_float(row.get("information_sufficiency")) for row in delivered_group]
                    ),
                    "average_detection_quality_by_payload_type": _mean(
                        [_to_float(row.get("detection_quality")) for row in delivered_group]
                    ),
                    "average_timeliness_at_delivery_by_payload_type": _mean(
                        [_to_float(row.get("timeliness_at_delivery")) for row in delivered_group]
                    ),
                    "generated_payload_mb_by_payload_type": generated_mb,
                    "delivered_payload_mb_by_payload_type": delivered_mb,
                    "utility_per_delivered_mb_by_payload_type": _safe_div(delivered_utility, delivered_mb),
                    "high_value_utility_per_delivered_mb_by_payload_type": _safe_div(hv_utility, delivered_mb),
                    "fraction_delivered_only_utility_by_payload_type": _safe_div(delivered_utility, delivered_total),
                    "fraction_high_value_delivered_utility_by_payload_type": _safe_div(hv_utility, high_value_total),
                }
            )
    return rows_out, by_variant


def write_attribution_markdown(path: Path, scenario: str, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Payload-Type Attribution: {scenario}",
        "",
        "Payload type is exact when sourced from `task_lifetime_outcomes.csv::payload_policy`; trajectory/action inference is marked if needed. A delivered task means lifetime-delivered before deadline.",
        "",
        "| variant | payload | inference | generated tasks | delivered tasks | HV delivered tasks | delivered utility | HV utility | generated MB | delivered MB | avg info | avg quality | avg timeliness | HV utility / delivered MB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["generated_task_count"] == 0 and row["delivered_task_count_by_payload_type"] == 0:
            continue
        lines.append(
            "| {variant} | {payload} | {infer} | {generated} | {delivered} | {hv_count} | {utility} | {hv_utility} | {generated_mb} | {delivered_mb} | {info} | {quality} | {time} | {hv_per_mb} |".format(
                variant=row["variant_short"],
                payload=row["payload_type"],
                infer=str(row["payload_type_inference"]).replace("|", "/"),
                generated=row["generated_task_count"],
                delivered=row["delivered_task_count_by_payload_type"],
                hv_count=row["high_value_delivered_task_count_by_payload_type"],
                utility=_fmt(row["delivered_only_utility_by_payload_type"]),
                hv_utility=_fmt(row["high_value_delivered_utility_by_payload_type"]),
                generated_mb=_fmt(row["generated_payload_mb_by_payload_type"]),
                delivered_mb=_fmt(row["delivered_payload_mb_by_payload_type"]),
                info=_fmt(row["average_information_sufficiency_by_payload_type"]),
                quality=_fmt(row["average_detection_quality_by_payload_type"]),
                time=_fmt(row["average_timeliness_at_delivery_by_payload_type"]),
                hv_per_mb=_fmt(row["high_value_utility_per_delivered_mb_by_payload_type"]),
            )
        )
    _write_text(path, "\n".join(lines) + "\n")


def build_dominance_rows(
    all_by_scenario: dict[str, dict[str, list[dict[str, Any]]]],
    attribution_by_scenario: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    dominance_rows: list[dict[str, Any]] = []
    for scenario, by_variant in all_by_scenario.items():
        attribution = {
            (row["variant_full"], row["payload_type"]): row
            for row in attribution_by_scenario[scenario]
        }
        for variant, rows in sorted(by_variant.items()):
            short = str(rows[0]["_variant_short"]) if rows else variant
            hv_rows = [row for row in rows if row["_delivered_before_deadline"] and row["_high_value"]]
            hv_total = sum(float(row["_utility"]) for row in hv_rows)
            payload_sets: dict[str, set[str]] = defaultdict(set)
            for row in hv_rows:
                payload_sets[str(row.get("task_id", ""))].add(str(row["_payload_type"]))
            metadata_only_count = sum(1 for payloads in payload_sets.values() if payloads == {"metadata"})
            out: dict[str, Any] = {
                "scenario": scenario,
                "variant_short": short,
                "variant_full": variant,
                "high_value_delivered_utility_total": hv_total,
                "high_value_delivered_task_count": len(payload_sets),
                "high_value_delivered_tasks_metadata_only": metadata_only_count,
                "fraction_high_value_delivered_tasks_metadata_only": _safe_div(metadata_only_count, len(payload_sets)),
            }
            for payload_type in CORE_PAYLOAD_TYPES:
                attr = attribution.get((variant, payload_type), {})
                utility = _to_float(attr.get("high_value_delivered_utility_by_payload_type"), 0.0)
                count = _to_int(attr.get("high_value_delivered_task_count_by_payload_type"), 0)
                delivered_mb = _to_float(attr.get("delivered_payload_mb_by_payload_type"), 0.0)
                hv_payload_rows = [
                    row for row in hv_rows if row["_payload_type"] == payload_type
                ]
                out[f"high_value_utility_{payload_type}"] = utility
                out[f"fraction_high_value_utility_{payload_type}"] = _safe_div(utility, hv_total)
                out[f"high_value_delivered_count_{payload_type}"] = count
                out[f"avg_information_sufficiency_high_value_{payload_type}"] = _mean(
                    [_to_float(row.get("information_sufficiency")) for row in hv_payload_rows]
                )
                out[f"high_value_utility_per_delivered_mb_{payload_type}"] = _safe_div(utility, delivered_mb)
            top_payload = "none"
            top_value = -1.0
            for payload_type in CORE_PAYLOAD_TYPES:
                value = _to_float(out.get(f"high_value_utility_{payload_type}"), 0.0)
                if value > top_value:
                    top_payload = payload_type
                    top_value = value
            out["top_high_value_payload_type"] = top_payload if top_value > 0.0 else "none"
            dominance_rows.append(out)
    return dominance_rows


def write_dominance_report(path: Path, dominance_rows: list[dict[str, Any]]) -> None:
    stress_rows = [row for row in dominance_rows if row["scenario"] == "stress"]
    key_rows = [row for row in stress_rows if row["variant_short"] in KEY_VARIANTS]
    lines = [
        "# High-Value Payload Dominance Diagnosis",
        "",
        "This report uses existing lifetime-delivered-before-deadline rows only. It does not change delivered-only metrics.",
        "",
        "## Stress Key Variants",
        "",
        "| variant | HV utility | top payload | metadata frac | ROI frac | ROI+ctx frac | metadata-only HV task frac | avg info metadata | avg info ROI | avg info ROI+ctx | HV utility/MB metadata | HV utility/MB ROI | HV utility/MB ROI+ctx |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in key_rows:
        lines.append(
            "| {variant} | {total} | {top} | {meta_frac} | {roi_frac} | {ctx_frac} | {meta_only} | {meta_info} | {roi_info} | {ctx_info} | {meta_mb} | {roi_mb} | {ctx_mb} |".format(
                variant=row["variant_short"],
                total=_fmt(row["high_value_delivered_utility_total"]),
                top=row["top_high_value_payload_type"],
                meta_frac=_fmt(row["fraction_high_value_utility_metadata"]),
                roi_frac=_fmt(row["fraction_high_value_utility_roi"]),
                ctx_frac=_fmt(row["fraction_high_value_utility_roi_context"]),
                meta_only=_fmt(row["fraction_high_value_delivered_tasks_metadata_only"]),
                meta_info=_fmt(row["avg_information_sufficiency_high_value_metadata"]),
                roi_info=_fmt(row["avg_information_sufficiency_high_value_roi"]),
                ctx_info=_fmt(row["avg_information_sufficiency_high_value_roi_context"]),
                meta_mb=_fmt(row["high_value_utility_per_delivered_mb_metadata"]),
                roi_mb=_fmt(row["high_value_utility_per_delivered_mb_roi"]),
                ctx_mb=_fmt(row["high_value_utility_per_delivered_mb_roi_context"]),
            )
        )

    fix_l = next((row for row in key_rows if row["variant_short"] == "FIX-L"), {})
    util = next((row for row in key_rows if row["variant_short"] == "UTIL"), {})
    prop = next((row for row in key_rows if row["variant_short"] == "PROP-COMB"), {})
    fix_meta_frac = _to_float(fix_l.get("fraction_high_value_utility_metadata"))
    util_roi_frac = _to_float(util.get("fraction_high_value_utility_roi"))
    prop_meta_frac = _to_float(prop.get("fraction_high_value_utility_metadata"))
    prop_roi_frac = _to_float(prop.get("fraction_high_value_utility_roi"))

    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- FIX-L high-value utility is mostly metadata alert utility: {'yes' if fix_meta_frac > 0.8 else 'not exclusively'} (metadata fraction {_fmt(fix_meta_frac)}).",
            f"- Metadata information sufficiency for high-value FIX-L deliveries averages {_fmt(_to_float(fix_l.get('avg_information_sufficiency_high_value_metadata')))}. This supports an alert-level interpretation, not necessarily a verification-level product.",
            "- ROI and ROI+context improve verification/context evidence but carry much larger payload volume, so utility per delivered MB is typically lower than metadata.",
            f"- UTIL's high-value utility is dominated by ROI if its ROI fraction is high; stress UTIL ROI fraction is {_fmt(util_roi_frac)}.",
            f"- PROP-COMB balance: metadata fraction {_fmt(prop_meta_frac)}, ROI fraction {_fmt(prop_roi_frac)}, ROI+context fraction {_fmt(_to_float(prop.get('fraction_high_value_utility_roi_context')))}. It is more balanced than FIX-L when ROI/ROI+context fractions are non-zero, but stress output still shows metadata/ROI tradeoffs rather than a clean dominance win.",
            "",
        ]
    )
    _write_text(path, "\n".join(lines))


def load_shadow_config() -> dict[str, Any]:
    data = _load_yaml(SHADOW_CONFIG)
    if data:
        return data
    return {
        "payload_set_sufficiency": {
            "empty": 0.0,
            "metadata": 0.45,
            "roi": 0.65,
            "roi_context": 0.75,
            "metadata_roi": 0.80,
            "metadata_roi_context": 0.90,
            "metadata_roi_roi_context": 1.00,
        },
        "level_mapping": {
            "alert": ["metadata"],
            "verification": ["roi"],
            "context": ["roi_context"],
        },
    }


def _set_key(payloads: set[str]) -> str:
    order = ["metadata", "roi", "roi_context"]
    kept = [payload for payload in order if payload in payloads]
    return "_".join(kept) if kept else "empty"


def _shadow_sufficiency(payloads: set[str], sufficiency_map: dict[str, float]) -> float:
    key = _set_key(payloads)
    if key in sufficiency_map:
        return float(sufficiency_map[key])
    values = [float(sufficiency_map.get(payload, 0.0)) for payload in payloads]
    return max(values) if values else float(sufficiency_map.get("empty", 0.0))


def build_shadow_rows(
    scenario: str,
    by_variant: dict[str, list[dict[str, Any]]],
    shadow_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    sufficiency_map = {str(k): float(v) for k, v in dict(shadow_cfg.get("payload_set_sufficiency", {})).items()}
    level_mapping = {
        str(level): {str(payload) for payload in payloads}
        for level, payloads in dict(shadow_cfg.get("level_mapping", {})).items()
        if isinstance(payloads, list)
    }
    rows_out: list[dict[str, Any]] = []

    for variant, rows in sorted(by_variant.items(), key=lambda item: item[0]):
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["_delivered_before_deadline"] and row["_payload_type"] in CORE_PAYLOAD_TYPES:
                by_task[str(row.get("task_id", ""))].append(row)

        totals = Counter()
        utility_total = 0.0
        hv_utility_total = 0.0
        current_total = 0.0
        current_hv_total = 0.0
        delivered_task_count = 0
        high_value_task_count = 0

        for task_rows in by_task.values():
            delivered_task_count += 1
            task_rows.sort(key=_event_time)
            delivered_set: set[str] = set()
            task_is_high_value = any(bool(row["_high_value"]) for row in task_rows)
            if task_is_high_value:
                high_value_task_count += 1
            for row in task_rows:
                payload_type = str(row["_payload_type"])
                before = set(delivered_set)
                after = set(delivered_set)
                after.add(payload_type)
                delta_info = max(
                    0.0,
                    _shadow_sufficiency(after, sufficiency_map) - _shadow_sufficiency(before, sufficiency_map),
                )
                task_value = max(0.0, _to_float(row.get("task_value"), 0.0))
                quality = max(0.0, _to_float(row.get("detection_quality"), 0.0))
                timeliness = max(0.0, _to_float(row.get("timeliness_at_delivery"), 0.0))
                shadow_utility = task_value * quality * delta_info * timeliness
                utility_total += shadow_utility
                totals[f"shadow_increment_{payload_type}"] += shadow_utility
                if payload_type in level_mapping.get("alert", set()):
                    totals["shadow_alert_level_utility"] += shadow_utility
                if payload_type in level_mapping.get("verification", set()):
                    totals["shadow_verification_level_utility"] += shadow_utility
                if payload_type in level_mapping.get("context", set()):
                    totals["shadow_context_level_utility"] += shadow_utility
                if bool(row["_high_value"]):
                    hv_utility_total += shadow_utility
                current_total += float(row["_utility"])
                if bool(row["_high_value"]):
                    current_hv_total += float(row["_utility"])
                delivered_set = after

        short = str(rows[0]["_variant_short"]) if rows else variant
        row_out = {
            "scenario": scenario,
            "variant_short": short,
            "variant_full": variant,
            "shadow_progressive_delivered_utility": utility_total,
            "shadow_progressive_high_value_delivered_utility": hv_utility_total,
            "shadow_alert_level_utility": totals["shadow_alert_level_utility"],
            "shadow_verification_level_utility": totals["shadow_verification_level_utility"],
            "shadow_context_level_utility": totals["shadow_context_level_utility"],
            "shadow_increment_metadata": totals["shadow_increment_metadata"],
            "shadow_increment_roi": totals["shadow_increment_roi"],
            "shadow_increment_roi_context": totals["shadow_increment_roi_context"],
            "current_delivered_only_utility_same_rows": current_total,
            "current_high_value_delivered_utility_same_rows": current_hv_total,
            "shadow_to_current_delivered_ratio": _safe_div(utility_total, current_total),
            "shadow_to_current_high_value_ratio": _safe_div(hv_utility_total, current_hv_total),
            "delivered_task_count": delivered_task_count,
            "high_value_delivered_task_count": high_value_task_count,
        }
        rows_out.append(row_out)
    return rows_out


def write_shadow_report(path: Path, shadow_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Shadow Progressive Information Metrics",
        "",
        "The shadow metric is diagnostic only. It uses `configs/shadow_progressive_information_metric.yaml` and is not used by policy, reward, queue priority, or delivered-only metrics.",
        "",
        "Definition: each task accumulates a delivered payload set. The incremental credit for a delivered payload is the increase in configured monotone sufficiency for that set. Shadow utility is `V_i * detection_quality * Delta u_I * timeliness_at_delivery`.",
        "",
    ]
    for scenario in ("nominal", "stress"):
        scenario_rows = [row for row in shadow_rows if row["scenario"] == scenario]
        current_rank = sorted(scenario_rows, key=lambda row: _to_float(row["current_high_value_delivered_utility_same_rows"]), reverse=True)
        shadow_rank = sorted(scenario_rows, key=lambda row: _to_float(row["shadow_progressive_high_value_delivered_utility"]), reverse=True)
        lines.extend(
            [
                f"## {scenario.title()}",
                "",
                "| rank shadow HV | variant | shadow HV utility | current HV utility | alert | verification | context | shadow/current HV ratio |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for rank, row in enumerate(shadow_rank, start=1):
            lines.append(
                "| {rank} | {variant} | {shadow_hv} | {current_hv} | {alert} | {verify} | {context} | {ratio} |".format(
                    rank=rank,
                    variant=row["variant_short"],
                    shadow_hv=_fmt(row["shadow_progressive_high_value_delivered_utility"]),
                    current_hv=_fmt(row["current_high_value_delivered_utility_same_rows"]),
                    alert=_fmt(row["shadow_alert_level_utility"]),
                    verify=_fmt(row["shadow_verification_level_utility"]),
                    context=_fmt(row["shadow_context_level_utility"]),
                    ratio=_fmt(row["shadow_to_current_high_value_ratio"]),
                )
            )
        lines.extend(
            [
                "",
                f"Current high-value rank leader: {current_rank[0]['variant_short'] if current_rank else 'none'}.",
                f"Shadow high-value rank leader: {shadow_rank[0]['variant_short'] if shadow_rank else 'none'}.",
                "",
            ]
        )
    _write_text(path, "\n".join(lines))


def write_product_requirement_note(path: Path) -> None:
    lines = [
        "# Product Requirement Class Design Note",
        "",
        "This is a documentation proposal only. It does not modify task generation, reward, queue priority, router behavior, or current metrics.",
        "",
        "## Motivation",
        "",
        "The current task prior separates task value from SARFish label content, which is the right no-leakage baseline. A future extension can add a mission-scenario information requirement class `r_i` to explain whether a task primarily needs an alert, verification, context, or evidence product.",
        "",
        "## Proposed Classes",
        "",
        "| class | interpretation | expected marginal information pattern |",
        "| --- | --- | --- |",
        "| alert | Fast early warning is the primary product. | Metadata carries high initial utility; ROI is useful but secondary. |",
        "| verification | The mission needs confirmation of the alert. | ROI has high marginal utility; metadata alone is incomplete. |",
        "| contextual | Interpretation depends on local surroundings. | ROI+context has high utility for scene understanding. |",
        "| evidence | Post-event analysis or analyst handoff needs richer evidence. | Full/context-rich payloads have high utility; metadata is only a cue. |",
        "",
        "## No-Label-Leakage Generation Rule",
        "",
        "`r_i` should be generated from mission scenario priors such as observation window type, operational campaign, emergency bulletin, contact plan, and analyst product requirement. It should not be generated from SARFish target count, bbox count, vessel/fishing class labels, or target presence.",
        "",
        "Examples:",
        "",
        "- A scheduled wide-area monitoring pass can sample mostly `alert` with some `verification` follow-up products.",
        "- A mission-commanded suspicious vessel revisit can sample `verification` or `contextual` based on the planned product, not based on labels found in the image.",
        "- A post-event evidence collection campaign can sample `evidence` before image content is known.",
        "",
        "## Future Integration",
        "",
        "A later design review could use `r_i` to define progressive information sufficiency curves per class. That should be introduced first as a shadow metric, then validated against mission product requirements before any reward or queue changes are considered.",
        "",
    ]
    _write_text(path, "\n".join(lines))


def write_queue_priority_audit(path: Path, attribution_rows: list[dict[str, Any]]) -> None:
    stress = [row for row in attribution_rows if row["scenario"] == "stress"]
    key = [row for row in stress if row["variant_short"] in {"FIX-L", "UTIL", "PROP-COMB", "PROP-Z"}]
    lines = [
        "# Queue Priority Component Audit",
        "",
        "Static code inspection shows `src/link/downlink_queue.py` prioritizes queued items by utility density, deadline urgency, age bonus, metadata bonus, and high-value bonus. Existing trajectory/lifetime outputs do not log those component values per queue item.",
        "",
        "Formula inspected:",
        "",
        "`priority = remaining_utility / remaining_mb * urgency + age_bonus + metadata_bonus + high_value_bonus`",
        "",
        "where urgency is an exponential function of deadline slack, metadata gets a fixed bonus, and high-value items get a fixed bonus.",
        "",
        "## Observable Stress Evidence",
        "",
        "| variant | payload | generated MB | delivered MB | HV utility | HV utility / delivered MB |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in key:
        if row["payload_type"] not in CORE_PAYLOAD_TYPES:
            continue
        lines.append(
            "| {variant} | {payload} | {generated} | {delivered} | {hv} | {hv_per_mb} |".format(
                variant=row["variant_short"],
                payload=row["payload_type"],
                generated=_fmt(row["generated_payload_mb_by_payload_type"]),
                delivered=_fmt(row["delivered_payload_mb_by_payload_type"]),
                hv=_fmt(row["high_value_delivered_utility_by_payload_type"]),
                hv_per_mb=_fmt(row["high_value_utility_per_delivered_mb_by_payload_type"]),
            )
        )
    lines.extend(
        [
            "",
            "## What Can Be Answered Now",
            "",
            "- Small payload size: partially answerable from generated/delivered MB. FIX-L's metadata volume is much smaller than ROI/ROI+context-heavy variants.",
            "- Metadata bonus: not attributable quantitatively from current outputs because queue priority components are not logged.",
            "- High-value bonus: not attributable quantitatively from current outputs because queue priority components are not logged.",
            "- Utility density: partially inferable from remaining utility and payload size in code, but not logged per queue decision.",
            "- Deadline urgency: not attributable quantitatively from current outputs because per-item slack and urgency components are not logged.",
            "",
            "## Future Logging Recommendation",
            "",
            "Add logging-only fields for queue decisions, without changing queue logic:",
            "",
            "- `queue_priority_total`",
            "- `utility_density_component`",
            "- `urgency_component`",
            "- `age_component`",
            "- `metadata_bonus_component`",
            "- `high_value_bonus_component`",
            "- `remaining_mb`",
            "- `remaining_utility`",
            "- `deadline_slack_s`",
            "- `queue_item_payload_type`",
            "",
        ]
    )
    _write_text(path, "\n".join(lines))


def write_candidate_score_plan(path: Path) -> None:
    lines = [
        "# Candidate Action Score Logging Plan",
        "",
        "Documentation only. Do not modify router behavior for this audit.",
        "",
        "For every task and every candidate action, future logging should record:",
        "",
        "- `scenario`",
        "- `variant`",
        "- `slot`",
        "- `task_id`",
        "- `task_value` / `V_t`",
        "- `high_value_flag`",
        "- `candidate_action`",
        "- `candidate_model`",
        "- `candidate_payload`",
        "- `predicted_quality`",
        "- `predicted_information_factor`",
        "- `predicted_delivery_factor`",
        "- `predicted_reward`",
        "- `predicted_effective_utility`",
        "- `resource_cost`",
        "- `predicted_energy`",
        "- `predicted_latency`",
        "- `predicted_payload_mb`",
        "- `predicted_queue_next_mb`",
        "- `UCB_bonus`",
        "- `risk_discount`",
        "- `cost_aware_ucb_bonus`",
        "- `safety_status`",
        "- `safety_reasons`",
        "- `final_score`",
        "- `selected_flag`",
        "",
        "This will allow later diagnosis of why PROP-Z or PROP-COMB selects metadata, ROI, ROI+context, or defer under specific resource/contact states. The log should be generated after prediction and safety filtering but before action execution, and it should remain a diagnostic artifact only.",
        "",
    ]
    _write_text(path, "\n".join(lines))


def write_synthesis_report(
    path: Path,
    dominance_rows: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
) -> None:
    stress_dom = [row for row in dominance_rows if row["scenario"] == "stress"]
    stress_shadow = [row for row in shadow_rows if row["scenario"] == "stress"]
    key = {row["variant_short"]: row for row in stress_dom}
    shadow_key = {row["variant_short"]: row for row in stress_shadow}
    current_hv_rank = sorted(stress_dom, key=lambda row: _to_float(row["high_value_delivered_utility_total"]), reverse=True)
    shadow_hv_rank = sorted(stress_shadow, key=lambda row: _to_float(row["shadow_progressive_high_value_delivered_utility"]), reverse=True)
    fix_l = key.get("FIX-L", {})
    prop_comb = key.get("PROP-COMB", {})
    util = key.get("UTIL", {})
    metadata_dominated_variants = [
        row["variant_short"]
        for row in stress_dom
        if _to_float(row.get("fraction_high_value_utility_metadata")) > 0.5
    ]
    prop_attr = [
        row for row in attribution_rows
        if row["scenario"] == "stress" and row["variant_short"] == "PROP-COMB" and row["payload_type"] in CORE_PAYLOAD_TYPES
    ]
    prop_generated = sum(_to_float(row["generated_payload_mb_by_payload_type"], 0.0) for row in prop_attr)
    fix_attr = [
        row for row in attribution_rows
        if row["scenario"] == "stress" and row["variant_short"] == "FIX-L" and row["payload_type"] in CORE_PAYLOAD_TYPES
    ]
    fix_generated = sum(_to_float(row["generated_payload_mb_by_payload_type"], 0.0) for row in fix_attr)

    lines = [
        "# Progressive Delivery Shadow Audit Report",
        "",
        "This is a read-only/post-processing report. It does not change reward, delivered-only metric definitions, queue priority, profiles, deadlines, or policy decisions.",
        "",
        "## Key Stress Findings",
        "",
        f"- Current high-value delivered utility leader: {current_hv_rank[0]['variant_short'] if current_hv_rank else 'none'}.",
        f"- Shadow progressive high-value utility leader: {shadow_hv_rank[0]['variant_short'] if shadow_hv_rank else 'none'}.",
        f"- FIX-L metadata fraction of high-value utility: {_fmt(_to_float(fix_l.get('fraction_high_value_utility_metadata')))}.",
        f"- UTIL ROI fraction of high-value utility: {_fmt(_to_float(util.get('fraction_high_value_utility_roi')))}.",
        f"- PROP-COMB metadata/ROI/ROI+context fractions: {_fmt(_to_float(prop_comb.get('fraction_high_value_utility_metadata')))} / {_fmt(_to_float(prop_comb.get('fraction_high_value_utility_roi')))} / {_fmt(_to_float(prop_comb.get('fraction_high_value_utility_roi_context')))}.",
        f"- Stress variants with metadata contributing more than half of high-value delivered utility: {', '.join(metadata_dominated_variants) if metadata_dominated_variants else 'none'}.",
        "",
        "## Answers",
        "",
        "1. Does the current result support a metadata-first interpretation?",
        "",
        f"Yes for FIX-L and any variant with high metadata fraction. FIX-L's high-value utility is metadata-dominated ({_fmt(_to_float(fix_l.get('fraction_high_value_utility_metadata')))}), and its generated payload volume is far smaller than PROP-COMB ({_fmt(fix_generated)} MB vs {_fmt(prop_generated)} MB for core payload types).",
        "",
        "2. Is FIX-L strongest only at alert-level metadata delivery?",
        "",
        "The evidence supports that interpretation. FIX-L uses metadata only among core payload types, so its high-value advantage should be described as alert-level delivery, not verification/context evidence.",
        "",
        "3. Does PROP-COMB show a better balance between delivered-only utility, high-value utility, payload volume, and resource safety?",
        "",
        "PROP-COMB is more balanced than FIX-L because it can deliver metadata and ROI products, and it is close to UTIL on delivered-only utility in stress. It does not beat FIX-L on high-value delivered utility because heavier ROI/context products consume more communication capacity.",
        "",
        "4. Is the current high-value delivered utility metric too metadata-dominated?",
        "",
        "It is metadata-dominated for FIX-L. That is not automatically a bug if metadata is a valid alert product, but the paper should avoid presenting this as verification-level performance. The attribution tables make the distinction explicit.",
        "",
        "5. Should progressive information delivery be introduced as a paper explanation only, a shadow metric, or a future core implementation?",
        "",
        "Use it now as a shadow metric and paper explanation. A future core implementation should wait for product-requirement class design and validation.",
        "",
        "6. Is it safe to proceed to formal multi-seed after this audit, or is one more logging-only smoke test needed?",
        "",
        "For metric-table readiness, the attribution and shadow metrics are enough to proceed cautiously. For causal claims about why proposed variants choose metadata/ROI/ROI+context, one logging-only smoke test with candidate action scores and queue priority components is recommended before formal multi-seed.",
        "",
        "7. What changes are safe and what changes should be avoided?",
        "",
        "- Safe A: paper explanation only, clearly naming metadata as alert-level utility.",
        "- Safe B: keep these shadow metrics and attribution tables as post-processing diagnostics.",
        "- Safe C: add logging-only candidate action and queue priority fields, then rerun one smoke test if causal attribution is needed.",
        "- Avoid D for now: reward, queue, profile, or task-prior modifications. Current evidence is not strong enough to justify behavior changes.",
        "",
        "## Recommendation",
        "",
        "Recommended next step: B, with C if the paper needs causal action-selection attribution before formal multi-seed.",
        "",
    ]
    _write_text(path, "\n".join(lines))


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    shadow_cfg = load_shadow_config()
    attribution_by_scenario: dict[str, list[dict[str, Any]]] = {}
    by_scenario: dict[str, dict[str, list[dict[str, Any]]]] = {}
    shadow_rows_all: list[dict[str, Any]] = []

    for scenario, output_dir in SCENARIOS.items():
        attribution_rows, by_variant = build_attribution_rows(scenario, output_dir)
        attribution_by_scenario[scenario] = attribution_rows
        by_scenario[scenario] = by_variant
        _write_csv(AUDIT_DIR / f"payload_type_attribution_{scenario}.csv", attribution_rows)
        write_attribution_markdown(AUDIT_DIR / f"payload_type_attribution_{scenario}.md", scenario, attribution_rows)
        shadow_rows = build_shadow_rows(scenario, by_variant, shadow_cfg)
        shadow_rows_all.extend(shadow_rows)
        _write_csv(AUDIT_DIR / f"shadow_progressive_information_metrics_{scenario}.csv", shadow_rows)

    dominance_rows = build_dominance_rows(by_scenario, attribution_by_scenario)
    _write_csv(AUDIT_DIR / "high_value_payload_dominance_diagnosis.csv", dominance_rows)
    write_dominance_report(AUDIT_DIR / "high_value_payload_dominance_diagnosis.md", dominance_rows)
    write_shadow_report(AUDIT_DIR / "shadow_progressive_information_metrics.md", shadow_rows_all)
    write_product_requirement_note(AUDIT_DIR / "product_requirement_class_design_note.md")
    write_queue_priority_audit(
        AUDIT_DIR / "queue_priority_component_audit.md",
        [row for rows in attribution_by_scenario.values() for row in rows],
    )
    write_candidate_score_plan(AUDIT_DIR / "candidate_action_score_logging_plan.md")
    write_synthesis_report(
        AUDIT_DIR / "progressive_delivery_shadow_audit_report.md",
        dominance_rows,
        shadow_rows_all,
        [row for rows in attribution_by_scenario.values() for row in rows],
    )

    stress_key = [
        row for row in dominance_rows
        if row["scenario"] == "stress" and row["variant_short"] in {"FIX-L", "UTIL", "PROP-COMB", "PROP-Z"}
    ]
    print("created_files=" + ",".join(
        [
            "outputs/_audit/payload_type_attribution_nominal.csv",
            "outputs/_audit/payload_type_attribution_stress.csv",
            "outputs/_audit/payload_type_attribution_nominal.md",
            "outputs/_audit/payload_type_attribution_stress.md",
            "outputs/_audit/high_value_payload_dominance_diagnosis.csv",
            "outputs/_audit/high_value_payload_dominance_diagnosis.md",
            "outputs/_audit/shadow_progressive_information_metrics_nominal.csv",
            "outputs/_audit/shadow_progressive_information_metrics_stress.csv",
            "outputs/_audit/shadow_progressive_information_metrics.md",
            "outputs/_audit/product_requirement_class_design_note.md",
            "outputs/_audit/queue_priority_component_audit.md",
            "outputs/_audit/candidate_action_score_logging_plan.md",
            "outputs/_audit/progressive_delivery_shadow_audit_report.md",
        ]
    ))
    print("payload_type_inference=exact_payload_policy_for_lifetime_rows_when_payload_policy_present")
    for row in sorted(stress_key, key=lambda item: KEY_VARIANTS.index(item["variant_short"]) if item["variant_short"] in KEY_VARIANTS else 99):
        print(
            "stress_top_high_value_payload={variant}:{payload}:metadata_fraction={fraction}".format(
                variant=row["variant_short"],
                payload=row["top_high_value_payload_type"],
                fraction=_fmt(row["fraction_high_value_utility_metadata"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
