"""Post-process delivered-only utility metrics from existing experiment outputs.

This script does not run simulations and does not modify experiment directories.
It reads summary, trajectory, lifetime, and leaderboard artifacts and writes
canonical evaluation-layer tables under outputs/_index.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
INDEX_DIR = OUTPUT_ROOT / "_index"

STAGES = [
    {
        "stage": "nominal_sarfish_20s",
        "output_dir": OUTPUT_ROOT / "sarfish_20s_dry_run",
        "config": PROJECT_ROOT / "configs" / "simulator_sarfish_20s_dryrun.yaml",
    },
    {
        "stage": "energy_shadow_stress",
        "output_dir": OUTPUT_ROOT / "sarfish_energy_shadow_stress",
        "config": PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_stress_20s.yaml",
    },
    {
        "stage": "yolo_profile_ablation",
        "output_dir": OUTPUT_ROOT / "sarfish_yolo_profile_ablation",
        "config": PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_yolo_profile_ablation_20s.yaml",
    },
    {
        "stage": "policy_repair_ablation",
        "output_dir": OUTPUT_ROOT / "sarfish_policy_repair",
        "config": PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_policy_repair_ablation_20s.yaml",
    },
    {
        "stage": "payload_profile_calibration",
        "output_dir": OUTPUT_ROOT / "sarfish_payload_profile_calibration",
        "config": PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_payload_profile_ablation_20s.yaml",
    },
]

METRIC_FIELDS = [
    "experiment_stage",
    "output_dir",
    "variant_short",
    "variant_full",
    "metric_quality",
    "notes",
    "gross_task_utility_provisional",
    "net_system_utility_provisional",
    "penalized_reward",
    "delivered_only_gross_utility",
    "delivered_only_incremental_utility",
    "delivered_before_deadline_utility",
    "high_value_delivered_utility",
    "queued_provisional_utility",
    "queued_provisional_ratio",
    "delivered_existing_utility",
    "delivery_realization_gap",
    "delivered_utility_ratio",
    "expiration_adjusted_net_utility",
    "expiration_adjusted_source",
    "eta_exp",
    "explicit_size_cost_net_utility",
    "lambda_S_eval",
    "generated_payload_mb_normalized",
    "total_generated_payload_mb",
    "delivered_mb",
    "completed_tasks_same_step",
    "delivered_items_payload_level",
    "expired_items_payload_level",
    "lifetime_delivered_task_level",
    "lifetime_expired_task_level",
    "lifetime_deferred_task_level",
    "lifetime_undelivered_by_end_task_level",
    "delivered_late_task_count",
    "delivered_before_deadline_task_count",
    "high_value_threshold",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


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


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if not _finite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _safe_div(numerator: float, denominator: float) -> float:
    if not _finite(numerator) or not _finite(denominator) or abs(denominator) <= 1.0e-12:
        return math.nan
    return numerator / denominator


def _timeliness(latency_s: float, deadline_s: float) -> float:
    if not _finite(latency_s) or not _finite(deadline_s) or deadline_s <= 1.0e-12:
        return math.nan
    return max(0.0, min(1.0, 1.0 - latency_s / deadline_s))


def _variant_name(row: dict[str, str]) -> str:
    return str(row.get("router") or row.get("router_name") or row.get("variant") or "")


def _summary_by_variant(output_dir: Path) -> dict[str, dict[str, str]]:
    return {_variant_name(row): row for row in _read_csv(output_dir / "summary.csv") if _variant_name(row)}


def _lifetime_by_variant(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(output_dir / "task_lifetime_outcomes.csv"):
        router = str(row.get("router_name") or row.get("router") or "")
        if router:
            grouped[router].append(row)
    return grouped


def _trajectory_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return _read_csv(output_dir / "trajectories" / f"{variant}.csv")


def _task_trajectory_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        if _to_int(row.get("has_task")) != 1:
            continue
        task_id = str(row.get("task_id", ""))
        if task_id and task_id not in index:
            index[task_id] = row
    return index


def _row_utility_without_delivery(row: dict[str, str], lifetime_latency_s: float | None = None) -> float:
    value = _to_float(row.get("V_t", row.get("task_value")))
    quality = _to_float(row.get("detection_quality"))
    info_factor = _to_float(row.get("information_factor"))
    deadline = _to_float(row.get("deadline_s"))
    if lifetime_latency_s is None or not _finite(lifetime_latency_s):
        lifetime_latency_s = (
            _to_float(row.get("L_inf"), 0.0)
            + _to_float(row.get("L_queue"), 0.0)
            + _to_float(row.get("L_comm"), 0.0)
        )
    time_factor = _timeliness(lifetime_latency_s, deadline)
    parts = (value, quality, info_factor, time_factor)
    if not all(_finite(part) for part in parts):
        return math.nan
    return max(0.0, value) * max(0.0, quality) * max(0.0, info_factor) * max(0.0, time_factor)


def _same_step_delivered_utility(rows: list[dict[str, str]]) -> float:
    total = 0.0
    for row in rows:
        if _to_int(row.get("has_task")) != 1:
            continue
        if str(row.get("delivery_status", "")).lower() != "delivered":
            continue
        total += _to_float(row.get("effective_utility"), 0.0)
    return total


def _queued_provisional_utility(rows: list[dict[str, str]]) -> float:
    total = 0.0
    for row in rows:
        if _to_int(row.get("has_task")) != 1:
            continue
        if str(row.get("delivery_status", "")).lower() == "queued":
            total += _to_float(row.get("effective_utility"), 0.0)
    return total


def _delivered_only_incremental_utility(rows: list[dict[str, str]]) -> float:
    return sum(_to_float(row.get("delivered_only_incremental_utility"), 0.0) for row in rows)


def _average_lambda_s(rows: list[dict[str, str]]) -> float:
    values = [
        _to_float(row.get("lambda_S"))
        for row in rows
        if _to_int(row.get("has_task")) == 1 and _finite(_to_float(row.get("lambda_S")))
    ]
    if not values:
        return math.nan
    return sum(values) / len(values)


def _alias_maps() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(INDEX_DIR / "leaderboard.csv")
    if rows:
        return {
            (str(row.get("experiment_stage", "")), str(row.get("variant_full", ""))): row
            for row in rows
            if row.get("experiment_stage") and row.get("variant_full")
        }
    data = _load_yaml(PROJECT_ROOT / "configs" / "variant_aliases.yaml")
    aliases: dict[tuple[str, str], dict[str, str]] = {}
    for item in data.get("variants", []):
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", ""))
        full = str(item.get("full_name", ""))
        if stage and full:
            aliases[(stage, full)] = {str(k): str(v) for k, v in item.items()}
            if stage == "all":
                aliases[("*", full)] = {str(k): str(v) for k, v in item.items()}
    return aliases


def _alias_for(stage: str, variant: str, aliases: dict[tuple[str, str], dict[str, str]]) -> tuple[str, str]:
    row = aliases.get((stage, variant)) or aliases.get(("*", variant)) or {}
    return str(row.get("variant_short") or row.get("short_name") or variant), str(row.get("enabled_features", ""))


def _compute_from_lifetime(
    lifetime_rows: list[dict[str, str]],
    trajectory_index: dict[str, dict[str, str]],
    high_value_threshold: float,
) -> tuple[float, float, int, int, int, list[str]]:
    delivered_utility = 0.0
    high_value_utility = 0.0
    delivered_before_deadline_count = 0
    delivered_late_count = 0
    missing_count = 0
    notes: list[str] = []

    for lifetime in lifetime_rows:
        if str(lifetime.get("final_lifetime_status", "")).lower() != "delivered":
            continue
        if str(lifetime.get("met_deadline_lifetime", "")).lower() != "true":
            delivered_late_count += 1
            continue

        task_id = str(lifetime.get("task_id", ""))
        row = trajectory_index.get(task_id)
        value = _to_float(lifetime.get("delivered_only_utility"))
        if not _finite(value):
            if row is None:
                missing_count += 1
                continue
            latency = _to_float(lifetime.get("lifetime_latency_s"))
            if not _finite(latency):
                latency = None
            value = _row_utility_without_delivery(row, latency)
        if not _finite(value):
            missing_count += 1
            continue
        delivered_utility += value
        delivered_before_deadline_count += 1

        high_value_flag = str(lifetime.get("high_value_flag", "")).lower() in {"1", "1.0", "true", "yes"}
        task_value = _to_float(lifetime.get("task_value"))
        if not high_value_flag and row is not None:
            task_value = _to_float(row.get("V_t", task_value))
        if high_value_flag or (_finite(task_value) and task_value >= high_value_threshold):
            high_value_utility += value

    if missing_count:
        notes.append(f"{missing_count} lifetime-delivered tasks lacked reconstructable factors")
    return (
        delivered_utility,
        high_value_utility,
        delivered_before_deadline_count,
        delivered_late_count,
        missing_count,
        notes,
    )


def _lifetime_counts(rows: list[dict[str, str]]) -> Counter:
    return Counter(str(row.get("final_lifetime_status", "")) for row in rows)


def _metric_quality(lifetime_rows: list[dict[str, str]], missing_count: int) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not lifetime_rows:
        return "approximate", ["task_lifetime_outcomes.csv missing; used same-step delivered trajectory rows only"]
    if missing_count:
        notes.append("some delivered tasks could not be reconstructed")
        return "approximate", notes
    if lifetime_rows and "exact_lifetime_timing_available" in lifetime_rows[0]:
        inexact_delivered = 0
        for row in lifetime_rows:
            if str(row.get("final_lifetime_status", "")).lower() != "delivered":
                continue
            if str(row.get("met_deadline_lifetime", "")).lower() != "true":
                continue
            if str(row.get("exact_lifetime_timing_available", "")).lower() != "true":
                inexact_delivered += 1
        if inexact_delivered:
            notes.append(f"{inexact_delivered} delivered-before-deadline tasks lack exact event timing")
            return "approximate", notes
        return "exact", []
    delayed_deliveries = 0
    for row in lifetime_rows:
        if str(row.get("final_lifetime_status", "")).lower() != "delivered":
            continue
        arrival = _to_int(row.get("arrival_slot"), -1)
        delivered = _to_int(row.get("delivered_slot"), -1)
        if arrival >= 0 and delivered >= 0 and delivered != arrival:
            delayed_deliveries += 1
    if delayed_deliveries:
        notes.append(f"{delayed_deliveries} delivered tasks used slot-level lifetime latency")
        return "approximate", notes
    return "exact", []


def compute_metrics(eta_exp: float = 1.0) -> list[dict[str, Any]]:
    aliases = _alias_maps()
    rows: list[dict[str, Any]] = []
    reward_cfg = _load_yaml(PROJECT_ROOT / "configs" / "reward.yaml")
    high_value_threshold = float(reward_cfg.get("high_value_threshold", 0.75))

    for stage_info in STAGES:
        stage = str(stage_info["stage"])
        output_dir = Path(stage_info["output_dir"])
        if not output_dir.exists():
            continue
        summaries = _summary_by_variant(output_dir)
        lifetimes = _lifetime_by_variant(output_dir)
        simulator_cfg = _load_yaml(Path(stage_info["config"]))
        queue_max_mb = float(simulator_cfg.get("queue_max_mb", 64.0))

        variants = sorted(set(summaries) | {path.stem for path in (output_dir / "trajectories").glob("*.csv")})
        for variant in variants:
            summary = summaries.get(variant, {})
            trajectory = _trajectory_rows(output_dir, variant)
            if not trajectory and not summary:
                continue
            task_index = _task_trajectory_index(trajectory)
            lifetime_rows = lifetimes.get(variant, [])
            lifetime_counter = _lifetime_counts(lifetime_rows)
            alias, _features = _alias_for(stage, variant, aliases)

            gross = _to_float(summary.get("gross_task_utility", summary.get("total_effective_utility")))
            if not _finite(gross):
                gross = sum(_to_float(row.get("effective_utility"), 0.0) for row in trajectory)
            net = _to_float(summary.get("net_system_utility"))
            penalized = _to_float(summary.get("total_penalized_reward", summary.get("penalized_reward")))

            if lifetime_rows:
                (
                    delivered_only,
                    high_value_delivered,
                    delivered_before_deadline_count,
                    delivered_late_count,
                    missing_count,
                    reconstruction_notes,
                ) = _compute_from_lifetime(lifetime_rows, task_index, high_value_threshold)
            else:
                delivered_only = 0.0
                high_value_delivered = 0.0
                delivered_before_deadline_count = 0
                delivered_late_count = 0
                missing_count = 0
                reconstruction_notes = []
                for row in trajectory:
                    if _to_int(row.get("has_task")) != 1:
                        continue
                    if str(row.get("delivery_status", "")).lower() != "delivered":
                        continue
                    if _to_int(row.get("deadline_missed")):
                        continue
                    value = _row_utility_without_delivery(row)
                    if _finite(value):
                        delivered_only += value
                        delivered_before_deadline_count += 1
                        task_value = _to_float(row.get("V_t"))
                        if _finite(task_value) and task_value >= high_value_threshold:
                            high_value_delivered += value

            quality, quality_notes = _metric_quality(lifetime_rows, missing_count)
            queued = _queued_provisional_utility(trajectory)
            delivered_existing = _same_step_delivered_utility(trajectory)
            delivered_incremental = _delivered_only_incremental_utility(trajectory)
            delivery_gap = gross - delivered_only if _finite(gross) else math.nan
            delivered_ratio = _safe_div(delivered_only, gross)
            queued_ratio = _safe_div(queued, gross)

            lifetime_expired = lifetime_counter.get("expired", 0)
            expired_items = _to_int(summary.get("expired_items"))
            expiration_source = "lifetime_expired" if lifetime_rows else "expired_items"
            expiration_count = lifetime_expired if lifetime_rows else expired_items
            expiration_adjusted = net - eta_exp * expiration_count if _finite(net) else math.nan

            generated_payload = _to_float(summary.get("total_generated_payload_mb", summary.get("generated_payload_mb")))
            generated_norm = generated_payload / max(queue_max_mb, 1.0e-12) if _finite(generated_payload) else math.nan
            lambda_s_eval = _average_lambda_s(trajectory)
            explicit_size_net = (
                net - lambda_s_eval * generated_norm
                if _finite(net) and _finite(lambda_s_eval) and _finite(generated_norm)
                else math.nan
            )

            notes = []
            notes.extend(reconstruction_notes)
            notes.extend(quality_notes)
            if _finite(explicit_size_net):
                notes.append("explicit size-cost metric is diagnostic only")
            else:
                notes.append("explicit size-cost metric unavailable")

            rows.append(
                {
                    "experiment_stage": stage,
                    "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                    "variant_short": alias,
                    "variant_full": variant,
                    "metric_quality": quality,
                    "notes": "; ".join(dict.fromkeys(note for note in notes if note)),
                    "gross_task_utility_provisional": gross,
                    "net_system_utility_provisional": net,
                    "penalized_reward": penalized,
                    "delivered_only_gross_utility": delivered_only,
                    "delivered_only_incremental_utility": delivered_incremental,
                    "delivered_before_deadline_utility": delivered_only,
                    "high_value_delivered_utility": high_value_delivered,
                    "queued_provisional_utility": queued,
                    "queued_provisional_ratio": queued_ratio,
                    "delivered_existing_utility": delivered_existing,
                    "delivery_realization_gap": delivery_gap,
                    "delivered_utility_ratio": delivered_ratio,
                    "expiration_adjusted_net_utility": expiration_adjusted,
                    "expiration_adjusted_source": expiration_source,
                    "eta_exp": eta_exp,
                    "explicit_size_cost_net_utility": explicit_size_net,
                    "lambda_S_eval": lambda_s_eval,
                    "generated_payload_mb_normalized": generated_norm,
                    "total_generated_payload_mb": generated_payload,
                    "delivered_mb": _to_float(summary.get("delivered_mb", summary.get("total_downlinked_payload_mb"))),
                    "completed_tasks_same_step": _to_int(summary.get("completed_tasks")),
                    "delivered_items_payload_level": _to_int(summary.get("delivered_items")),
                    "expired_items_payload_level": expired_items,
                    "lifetime_delivered_task_level": lifetime_counter.get("delivered", ""),
                    "lifetime_expired_task_level": lifetime_counter.get("expired", ""),
                    "lifetime_deferred_task_level": lifetime_counter.get("deferred", ""),
                    "lifetime_undelivered_by_end_task_level": lifetime_counter.get("undelivered_by_end", ""),
                    "delivered_late_task_count": delivered_late_count,
                    "delivered_before_deadline_task_count": delivered_before_deadline_count,
                    "high_value_threshold": high_value_threshold,
                }
            )
    return rows


def write_definitions(path: Path) -> None:
    lines = [
        "# Delivered-only Metric Definitions",
        "",
        "These metrics are post-processing diagnostics computed from existing artifacts. They do not change simulator reward, router behavior, action selection, profiles, traces, or generated results.",
        "",
        "## Existing Metrics Reinterpreted",
        "",
        "- `gross_task_utility`: expected/provisional effective utility. It may include queued credit through `delivery_beta`.",
        "- `net_system_utility`: expected/provisional system utility. It subtracts resource, requirement, safety, long-term risk, and terminal penalties, but does not directly subtract expired item count.",
        "- `penalized_reward`: slot-level reward aggregate. It is useful for debugging/training consistency, not the primary system metric.",
        "",
        "## New Metrics",
        "",
        "- `delivered_only_gross_utility`: strict utility for tasks delivered before deadline. It reconstructs `V_t * detection_quality * information_sufficiency * timeliness` from trajectory and lifetime files when available.",
        "- `delivered_only_incremental_utility`: product-level delivered-only incremental utility from the progressive payload lifecycle when present.",
        "- `delivered_before_deadline_utility`: equal to `delivered_only_gross_utility` in this post-processor because deadline satisfaction is enforced.",
        "- `high_value_delivered_utility`: delivered-before-deadline utility restricted to tasks with `V_t >= high_value_threshold`. The threshold defaults to `configs/reward.yaml` or 0.75.",
        "- `queued_provisional_utility`: sum of existing `effective_utility` for rows where `delivery_status == queued`.",
        "- `delivered_existing_utility`: sum of existing `effective_utility` for rows where `delivery_status == delivered`; this only captures same-step delivery credit and can differ from lifetime delivered utility.",
        "- `delivery_realization_gap`: `gross_task_utility - delivered_only_gross_utility`.",
        "- `delivered_utility_ratio`: `delivered_only_gross_utility / gross_task_utility`, with blank output for zero or unavailable denominators.",
        "- `expiration_adjusted_net_utility`: diagnostic sensitivity metric `net_system_utility - eta_exp * lifetime_expired` when lifetime data exists, otherwise `net_system_utility - eta_exp * expired_items`. Default `eta_exp` is 1.0.",
        "- `explicit_size_cost_net_utility`: diagnostic sensitivity metric `net_system_utility - mean(lambda_S) * generated_payload_mb / queue_max_mb` when trajectory lambda and config queue capacity are available.",
        "",
        "## Metric Quality",
        "",
        "- `exact`: lifetime files exist, all delivered-before-deadline tasks were delivered in their arrival slot, and all needed utility factors were present.",
        "- `approximate`: lifetime files are missing, some delivered tasks were delivered later using slot-level lifetime latency, or a reconstruction fallback was required.",
        "- `unavailable`: no usable summary/trajectory data was found. Current generated rows skip unavailable variants rather than emitting empty rows.",
        "",
        "## Lifecycle Distinctions",
        "",
        "- `completed_tasks` is same-step task delivery.",
        "- `delivered_items` is queue item-level delivery.",
        "- `lifetime_delivered` is task-level lifetime delivery.",
        "- `expired_items` is queue item-level expiration.",
        "- `lifetime_expired` is task-level lifetime expiration.",
        "- `delivered_mb` and `generated_payload_mb` both use MB, but `delivered_mb` is downlinked payload while `generated_payload_mb` is generated current-task payload.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metrics_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Delivered-only Metrics",
        "",
        "Rows are stage-local. Existing gross/net utility columns are provisional expected utilities; delivered-only columns enforce lifetime delivery before deadline when data allows.",
        "",
        "| stage | short | full | quality | gross provisional | delivered-only | ratio | queued provisional | gap | lifetime delivered | lifetime expired | exp-adjusted net | notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {short} | {full} | {quality} | {gross} | {delivered} | {ratio} | {queued} | {gap} | {life_del} | {life_exp} | {expnet} | {notes} |".format(
                stage=row["experiment_stage"],
                short=row["variant_short"],
                full=row["variant_full"],
                quality=row["metric_quality"],
                gross=_fmt(row["gross_task_utility_provisional"]),
                delivered=_fmt(row["delivered_only_gross_utility"]),
                ratio=_fmt(row["delivered_utility_ratio"]),
                queued=_fmt(row["queued_provisional_utility"]),
                gap=_fmt(row["delivery_realization_gap"]),
                life_del=row["lifetime_delivered_task_level"],
                life_exp=row["lifetime_expired_task_level"],
                expnet=_fmt(row["expiration_adjusted_net_utility"]),
                notes=str(row["notes"]).replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _top_by(rows: list[dict[str, Any]], field: str, stage: str, n: int = 3, reverse: bool = True) -> list[dict[str, Any]]:
    stage_rows = [row for row in rows if row["experiment_stage"] == stage and _finite(_to_float(row.get(field)))]
    return sorted(stage_rows, key=lambda row: float(row[field]), reverse=reverse)[:n]


def _format_variant_list(rows: list[dict[str, Any]], field: str) -> str:
    if not rows:
        return "none"
    return ", ".join(f"{row['variant_short']}={_fmt(row[field])}" for row in rows)


def write_readiness_report(path: Path, rows: list[dict[str, Any]]) -> None:
    stages = sorted({str(row["experiment_stage"]) for row in rows})
    exact_count = sum(1 for row in rows if row.get("metric_quality") == "exact")
    approximate_count = sum(1 for row in rows if row.get("metric_quality") == "approximate")
    unavailable_count = sum(1 for row in rows if row.get("metric_quality") == "unavailable")

    lines = [
        "# Formal Metric Readiness Update",
        "",
        "## 1. Executive Summary",
        "",
        f"Delivered-only metrics were computed for {len(rows)} variants across {len(stages)} available stages. Exact rows: {exact_count}; approximate rows: {approximate_count}; unavailable rows: {unavailable_count}.",
        "",
        "The current artifacts are sufficient for a conservative delivered-only post-processing layer. Most delivered-only rows are marked approximate because lifetime-delivered tasks can be delivered after their arrival slot and the lifetime diagnostic stores slot-level latency for those cases.",
        "",
        "Recommendation: prepare a 1-day single-seed smoke test using both provisional net utility and delivered-only/lifecycle metrics. Do not freeze full formal multi-seed results until paper tables adopt the delivered-only metrics and lifecycle definitions below.",
        "",
        "## 2. New Metric Definitions",
        "",
        "- `gross_task_utility_provisional` and `net_system_utility_provisional`: existing utility metrics interpreted as expected/provisional because queued rows can receive delivery-beta credit.",
        "- `delivered_only_gross_utility`: reconstructed delivered-before-deadline utility, without queued credit.",
        "- `delivered_only_incremental_utility`: product-level delivered-only incremental utility from progressive payload lifecycle columns, if present.",
        "- `delivered_before_deadline_utility`: identical to delivered-only utility in this implementation.",
        "- `high_value_delivered_utility`: delivered-before-deadline utility for tasks with `V_t >= high_value_threshold`.",
        "- `queued_provisional_utility`: existing effective utility credited to queued rows.",
        "- `delivered_existing_utility`: existing effective utility for same-step delivered rows only.",
        "- `delivery_realization_gap`: provisional gross utility minus delivered-only gross utility.",
        "- `delivered_utility_ratio`: delivered-only gross utility divided by provisional gross utility.",
        "- `expiration_adjusted_net_utility`: diagnostic sensitivity subtracting `eta_exp=1.0` per lifetime-expired task or expired item.",
        "- `explicit_size_cost_net_utility`: diagnostic size-cost sensitivity using mean trajectory `lambda_S` and generated payload normalized by queue capacity.",
        "",
        "## 3. Data Availability and Approximation Notes",
        "",
    ]
    for stage in stages:
        stage_rows = [row for row in rows if row["experiment_stage"] == stage]
        stage_exact = sum(1 for row in stage_rows if row["metric_quality"] == "exact")
        stage_approx = sum(1 for row in stage_rows if row["metric_quality"] == "approximate")
        lines.append(f"- `{stage}`: {len(stage_rows)} variants, {stage_exact} exact, {stage_approx} approximate. Delivered-only utility used trajectory utility factors plus lifetime delivery status when available.")
    lines.extend(["", "## 4. Results Overview", ""])
    for stage in stages:
        lines.append(f"### {stage}")
        lines.append(f"- strongest delivered-only variants: {_format_variant_list(_top_by(rows, 'delivered_only_gross_utility', stage), 'delivered_only_gross_utility')}.")
        lines.append(f"- highest queued provisional utility: {_format_variant_list(_top_by(rows, 'queued_provisional_utility', stage), 'queued_provisional_utility')}.")
        lines.append(f"- largest delivery realization gap: {_format_variant_list(_top_by(rows, 'delivery_realization_gap', stage), 'delivery_realization_gap')}.")
        lines.append(f"- highest lifetime expiration: {_format_variant_list(_top_by(rows, 'lifetime_expired_task_level', stage), 'lifetime_expired_task_level')}.")
        low_ratio_high_net = [
            row for row in _top_by(rows, "net_system_utility_provisional", stage, n=5)
            if _finite(_to_float(row.get("delivered_utility_ratio"))) and float(row["delivered_utility_ratio"]) < 0.35
        ]
        lines.append(f"- high provisional net but low delivered ratio: {_format_variant_list(low_ratio_high_net, 'delivered_utility_ratio')}.")
        lines.append("")

    lines.extend(
        [
            "## 5. Paper Usage Recommendation",
            "",
            "| Metric | Recommendation |",
            "| --- | --- |",
            "| `delivered_only_gross_utility` / `delivered_before_deadline_utility` | primary paper metric |",
            "| `high_value_delivered_utility` | primary high-value task metric when high-value threshold is disclosed |",
            "| `lifetime_delivered_task_level` / `lifetime_expired_task_level` | primary lifecycle metrics |",
            "| `net_system_utility_provisional` | secondary/system expected utility |",
            "| `gross_task_utility_provisional` | secondary expected utility; avoid calling it delivered utility |",
            "| `queued_provisional_utility` / `delivery_realization_gap` | diagnostic only |",
            "| `expiration_adjusted_net_utility` | diagnostic sensitivity only |",
            "| `explicit_size_cost_net_utility` | diagnostic sensitivity only |",
            "| `penalized_reward` | debug/training diagnostic only |",
            "| `action_similarity_to_no_mode` | diagnostic only |",
            "",
            "## 6. Next Step Recommendation",
            "",
            "A 1-day single-seed smoke test can be prepared now, provided the formal table template includes delivered-only utility, high-value delivered utility, and lifetime delivered/expired counts. Full formal multi-seed runs should wait until the paper metric table is updated and the current provisional-net interpretation is explicitly accepted.",
            "",
            "No simulator reward definitions need to be revisited immediately. Revisit reward definitions only if the paper requires realized reward itself to be delivered-only or to include a direct `lambda_S * S_gen` term.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_merged_leaderboard(rows: list[dict[str, Any]]) -> tuple[Path | None, Path | None]:
    leaderboard = _read_csv(INDEX_DIR / "leaderboard.csv")
    if not leaderboard:
        return None, None
    metric_map = {
        (str(row["experiment_stage"]), str(row["variant_full"])): row
        for row in rows
    }
    metric_subset = [
        "metric_quality",
        "delivered_only_gross_utility",
        "delivered_before_deadline_utility",
        "high_value_delivered_utility",
        "queued_provisional_utility",
        "delivery_realization_gap",
        "delivered_utility_ratio",
        "expiration_adjusted_net_utility",
        "explicit_size_cost_net_utility",
        "lifetime_delivered_task_level",
        "lifetime_expired_task_level",
    ]
    merged: list[dict[str, Any]] = []
    fields = list(leaderboard[0].keys())
    for field in metric_subset:
        if field not in fields:
            fields.append(field)
    for row in leaderboard:
        out = dict(row)
        metric = metric_map.get((str(row.get("experiment_stage")), str(row.get("variant_full"))), {})
        for field in metric_subset:
            out[field] = metric.get(field, "")
        merged.append(out)

    csv_path = INDEX_DIR / "leaderboard_with_delivered_only.csv"
    _write_csv(csv_path, merged, fields)

    md_path = INDEX_DIR / "leaderboard_with_delivered_only.md"
    lines = [
        "# Leaderboard With Delivered-only Metrics",
        "",
        "This is a non-invasive merge of the existing leaderboard with delivered-only post-processing metrics. Original leaderboard files are unchanged.",
        "",
        "| stage | rank | short | full | net provisional | delivered-only | ratio | queued provisional | lifetime delivered | lifetime expired | quality |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in merged:
        lines.append(
            "| {stage} | {rank} | {short} | {full} | {net} | {delivered} | {ratio} | {queued} | {life_del} | {life_exp} | {quality} |".format(
                stage=row.get("experiment_stage", ""),
                rank=row.get("stage_rank_by_net", ""),
                short=row.get("variant_short", ""),
                full=row.get("variant_full", ""),
                net=_fmt(_to_float(row.get("net_system_utility"))),
                delivered=_fmt(_to_float(row.get("delivered_only_gross_utility"))),
                ratio=_fmt(_to_float(row.get("delivered_utility_ratio"))),
                queued=_fmt(_to_float(row.get("queued_provisional_utility"))),
                life_del=row.get("lifetime_delivered_task_level", ""),
                life_exp=row.get("lifetime_expired_task_level", ""),
                quality=row.get("metric_quality", ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute delivered-only metrics from existing outputs.")
    parser.add_argument("--eta-exp", type=float, default=1.0, help="diagnostic expiration penalty weight")
    args = parser.parse_args()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows = compute_metrics(eta_exp=args.eta_exp)
    if not rows:
        print("No delivered-only metrics could be computed.")
        return 1

    metrics_csv = INDEX_DIR / "delivered_only_metrics.csv"
    metrics_md = INDEX_DIR / "delivered_only_metrics.md"
    definitions_md = INDEX_DIR / "delivered_only_metric_definitions.md"
    readiness_md = INDEX_DIR / "formal_metric_readiness_update.md"

    _write_csv(metrics_csv, rows, METRIC_FIELDS)
    write_metrics_markdown(metrics_md, rows)
    write_definitions(definitions_md)
    write_readiness_report(readiness_md, rows)
    merged_csv, merged_md = write_merged_leaderboard(rows)

    stages = {row["experiment_stage"] for row in rows}
    exact_count = sum(1 for row in rows if row["metric_quality"] == "exact")
    approximate_count = sum(1 for row in rows if row["metric_quality"] == "approximate")
    unavailable_count = sum(1 for row in rows if row["metric_quality"] == "unavailable")
    output_files = [metrics_csv, metrics_md, definitions_md, readiness_md]
    if merged_csv:
        output_files.append(merged_csv)
    if merged_md:
        output_files.append(merged_md)

    print(f"stages_processed={len(stages)}")
    print(f"variants_processed={len(rows)}")
    print(f"exact_delivered_only_metrics={exact_count}")
    print(f"approximate_metrics={approximate_count}")
    print(f"unavailable_metrics={unavailable_count}")
    print("output_files=" + ",".join(str(path.relative_to(PROJECT_ROOT)) for path in output_files))
    print("recommendation=prepare 1-day single-seed smoke test; wait to freeze full multi-seed until formal tables use delivered-only and lifecycle metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
