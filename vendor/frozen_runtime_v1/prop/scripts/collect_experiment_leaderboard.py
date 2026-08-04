"""Collect stage-local experiment summaries into a canonical leaderboard."""

from __future__ import annotations

import csv
import math
import sys
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
ALIAS_PATH = PROJECT_ROOT / "configs" / "variant_aliases.yaml"

STAGES = [
    {
        "stage": "nominal_sarfish_20s",
        "output_dir": OUTPUT_ROOT / "sarfish_20s_dry_run",
        "summary": "summary.csv",
    },
    {
        "stage": "energy_shadow_stress",
        "output_dir": OUTPUT_ROOT / "sarfish_energy_shadow_stress",
        "summary": "summary.csv",
    },
    {
        "stage": "yolo_profile_ablation",
        "output_dir": OUTPUT_ROOT / "sarfish_yolo_profile_ablation",
        "summary": "summary.csv",
    },
    {
        "stage": "policy_repair_ablation",
        "output_dir": OUTPUT_ROOT / "sarfish_policy_repair",
        "summary": "summary.csv",
    },
    {
        "stage": "payload_profile_calibration",
        "output_dir": OUTPUT_ROOT / "sarfish_payload_profile_calibration",
        "summary": "summary.csv",
    },
]

LEADERBOARD_FIELDS = [
    "experiment_stage",
    "output_dir",
    "variant_short",
    "variant_full",
    "enabled_features",
    "gross_task_utility",
    "net_system_utility",
    "penalized_reward",
    "resource_penalty",
    "requirement_penalty",
    "total_energy",
    "compute_energy",
    "comm_energy",
    "delivered_mb",
    "generated_payload_mb",
    "expired_items",
    "lifetime_delivered",
    "lifetime_expired",
    "high_accuracy_roi_context_count",
    "balanced_roi_count",
    "metadata_count",
    "roi_count",
    "roi_context_count",
    "defer_count",
    "action_similarity_to_no_mode",
    "stage_rank_by_net",
    "stage_best",
    "beats_utility_rule",
    "beats_no_mode",
    "notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _to_float(value: object) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _to_int(value: object) -> int | str:
    try:
        if value is None or value == "":
            return ""
        return int(float(value))
    except (TypeError, ValueError):
        return ""


def _format_number(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _alias_index() -> dict[tuple[str, str], dict[str, str]]:
    data = _load_yaml(ALIAS_PATH)
    index: dict[tuple[str, str], dict[str, str]] = {}
    for item in data.get("variants", []):
        if not isinstance(item, dict):
            continue
        full = str(item.get("full_name", ""))
        stage = str(item.get("stage", ""))
        if not full:
            continue
        entry = {str(key): str(value) for key, value in item.items()}
        index[(stage, full)] = entry
        if stage == "all":
            index[("*", full)] = entry
    return index


def _alias_for(stage: str, variant: str, aliases: dict[tuple[str, str], dict[str, str]]) -> dict[str, str]:
    return (
        aliases.get((stage, variant))
        or aliases.get(("*", variant))
        or aliases.get(("nominal_or_legacy", variant))
        or {
            "short_name": variant,
            "full_name": variant,
            "enabled_features": "",
            "one_sentence_description": "No alias entry yet.",
        }
    )


def _variant_name(row: dict[str, str]) -> str:
    return str(row.get("router") or row.get("router_name") or row.get("variant") or "")


def _summary_value(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        if key in row:
            value = _to_float(row.get(key))
            if not math.isnan(value):
                return value
    return math.nan


def _trajectory_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return _read_csv(output_dir / "trajectories" / f"{variant}.csv")


def _count_selected_actions(rows: list[dict[str, str]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        if str(row.get("has_task", "1")) in {"0", "0.0"}:
            continue
        action = str(row.get("selected_action") or row.get("executed_action") or "")
        if action:
            counts[action] += 1
    return counts


def _count_payloads(rows: list[dict[str, str]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        if str(row.get("has_task", "1")) in {"0", "0.0"}:
            continue
        payload = str(row.get("payload_policy") or row.get("selected_payload") or "")
        if payload:
            counts[payload] += 1
    return counts


def _lifetime_counts(output_dir: Path) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in _read_csv(output_dir / "task_lifetime_outcomes.csv"):
        router = str(row.get("router_name") or row.get("router") or "")
        status = str(row.get("final_lifetime_status") or "")
        if router and status:
            counts[router][status] += 1
    return counts


def _action_similarity_to_no_mode(output_dir: Path, variant: str) -> str:
    if variant == "no_mode":
        return "1.000"
    rows = _trajectory_rows(output_dir, variant)
    no_mode_rows = _trajectory_rows(output_dir, "no_mode")
    if not rows or not no_mode_rows:
        return ""
    pairs = [(a, b) for a, b in zip(rows, no_mode_rows) if a.get("task_id") == b.get("task_id")]
    if not pairs:
        return ""
    same = sum(1 for a, b in pairs if a.get("selected_action") == b.get("selected_action"))
    return f"{same / len(pairs):.3f}"


def _stage_rows(stage_info: dict[str, Any], aliases: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    stage = str(stage_info["stage"])
    output_dir = Path(stage_info["output_dir"])
    summary_rows = _read_csv(output_dir / str(stage_info["summary"]))
    if not summary_rows:
        return []

    lifetime = _lifetime_counts(output_dir)
    utility_net = math.nan
    no_mode_net = math.nan
    for row in summary_rows:
        name = _variant_name(row)
        if name == "utility_rule":
            utility_net = _summary_value(row, "net_system_utility")
        if name == "no_mode":
            no_mode_net = _summary_value(row, "net_system_utility")

    rows: list[dict[str, Any]] = []
    for summary in summary_rows:
        variant = _variant_name(summary)
        alias = _alias_for(stage, variant, aliases)
        trajectories = _trajectory_rows(output_dir, variant)
        actions = _count_selected_actions(trajectories)
        payloads = _count_payloads(trajectories)
        net = _summary_value(summary, "net_system_utility")
        notes: list[str] = []
        if not trajectories:
            notes.append("no trajectory csv")
        if math.isnan(utility_net):
            notes.append("utility_rule missing")
        if math.isnan(no_mode_net):
            notes.append("no_mode missing")
        rows.append(
            {
                "experiment_stage": stage,
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "variant_short": alias.get("short_name", variant),
                "variant_full": variant,
                "enabled_features": alias.get("enabled_features", ""),
                "gross_task_utility": _summary_value(summary, "gross_task_utility", "total_effective_utility"),
                "net_system_utility": net,
                "penalized_reward": _summary_value(summary, "total_penalized_reward", "penalized_reward"),
                "resource_penalty": _summary_value(summary, "total_resource_penalty", "resource_penalty"),
                "requirement_penalty": _summary_value(summary, "total_requirement_penalty", "requirement_penalty"),
                "total_energy": _summary_value(summary, "total_energy"),
                "compute_energy": _summary_value(summary, "total_compute_energy", "compute_energy"),
                "comm_energy": _summary_value(summary, "total_comm_energy", "comm_energy"),
                "delivered_mb": _summary_value(summary, "delivered_mb", "total_downlinked_payload_mb"),
                "generated_payload_mb": _summary_value(summary, "total_generated_payload_mb"),
                "expired_items": _to_int(summary.get("expired_items")),
                "lifetime_delivered": lifetime.get(variant, Counter()).get("delivered", ""),
                "lifetime_expired": lifetime.get(variant, Counter()).get("expired", ""),
                "high_accuracy_roi_context_count": actions.get("high_accuracy+roi_context", 0),
                "balanced_roi_count": actions.get("balanced+roi", 0),
                "metadata_count": payloads.get("metadata", 0),
                "roi_count": payloads.get("roi", 0),
                "roi_context_count": payloads.get("roi_context", 0),
                "defer_count": payloads.get("defer", 0),
                "action_similarity_to_no_mode": _action_similarity_to_no_mode(output_dir, variant),
                "stage_rank_by_net": "",
                "stage_best": "",
                "beats_utility_rule": "" if math.isnan(utility_net) or math.isnan(net) else int(net > utility_net),
                "beats_no_mode": "" if math.isnan(no_mode_net) or math.isnan(net) else int(net > no_mode_net),
                "notes": "; ".join(notes),
            }
        )
    rows.sort(key=lambda item: (math.isnan(float(item["net_system_utility"])), -float(item["net_system_utility"]) if not math.isnan(float(item["net_system_utility"])) else 0.0))
    for rank, row in enumerate(rows, start=1):
        row["stage_rank_by_net"] = rank
        row["stage_best"] = int(rank == 1)
    return rows


def collect_leaderboard() -> list[dict[str, Any]]:
    aliases = _alias_index()
    all_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        all_rows.extend(_stage_rows(stage, aliases))
    return all_rows


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Canonical Experiment Leaderboard",
        "",
        "Rows are sorted by `net_system_utility` within each stage. Cross-stage comparisons are diagnostic only because stages use different configs/profiles/stressors.",
        "",
        "| stage | rank | best | short | full | net | gross | penalized | beats UTIL | beats NO-MODE | notes |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {experiment_stage} | {stage_rank_by_net} | {stage_best} | {variant_short} | {variant_full} | {net} | {gross} | {penalized} | {beats_util} | {beats_nomode} | {notes} |".format(
                experiment_stage=row["experiment_stage"],
                stage_rank_by_net=row["stage_rank_by_net"],
                stage_best=row["stage_best"],
                variant_short=row["variant_short"],
                variant_full=row["variant_full"],
                net=_format_number(row["net_system_utility"]),
                gross=_format_number(row["gross_task_utility"]),
                penalized=_format_number(row["penalized_reward"]),
                beats_util=row["beats_utility_rule"],
                beats_nomode=row["beats_no_mode"],
                notes=row["notes"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_consistency_check(path: Path, rows: list[dict[str, Any]]) -> None:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stage[str(row["experiment_stage"])].append(row)
    lines = [
        "# Metric Consistency Quick Check",
        "",
        "## Main Findings",
        "",
        "1. A variant can show high `expired_items` while retaining high `net_system_utility` when delivered high-value utility and lower resource/requirement penalties dominate the expired-item count. `expired_items` is a service outcome count, not a direct additive term in the net utility formula.",
        "2. Different stages should not be compared as one global ranking: nominal dry run, energy-shadow stress, profile ablation, policy repair, and payload calibration use different configs, profiles, repair switches, and sometimes different variant sets.",
        "3. The same full variant name can mean a different baseline context across stages. For example `no_mode`, `no_lyapunov`, `utility_rule`, `fixed_lightweight`, and `fixed_high_accuracy` are stage-local baselines and inherit that stage's model/payload/profile configuration.",
        "4. `PROP-BASE` is intentionally stage-local: in nominal/stress it means `proposed_lyapunov_rule_mode_safe_bandit`, in policy repair it means `proposed_yolo_calibrated_baseline`, and in payload calibration it means `proposed_yolo_profile_policy_repair_baseline`.",
        "5. Valid direct comparisons are within the same stage directory. Cross-stage reads are useful for narrative trend checks only, not for claiming one numeric winner over all experiments.",
        "",
        "## Stage Checks",
        "",
    ]
    for stage, stage_rows in by_stage.items():
        best = next((row for row in stage_rows if str(row.get("stage_best")) == "1"), None)
        utility = next((row for row in stage_rows if row.get("variant_full") == "utility_rule"), None)
        no_mode = next((row for row in stage_rows if row.get("variant_full") == "no_mode"), None)
        lines.append(f"### {stage}")
        if best:
            lines.append(f"- best by net_system_utility: {best['variant_short']} / {best['variant_full']} = {_format_number(best['net_system_utility'])}.")
        if utility:
            lines.append(f"- utility_rule present: net {_format_number(utility['net_system_utility'])}.")
        else:
            lines.append("- utility_rule missing: `beats_utility_rule` is blank.")
        if no_mode:
            lines.append(f"- no_mode present: net {_format_number(no_mode['net_system_utility'])}.")
        else:
            lines.append("- no_mode missing: `beats_no_mode` is blank.")
        expired_with_positive_rank = [
            row for row in stage_rows[:3] if str(row.get("expired_items", "")) not in {"", "0"}
        ]
        if expired_with_positive_rank:
            labels = ", ".join(f"{row['variant_short']} expired={row['expired_items']}" for row in expired_with_positive_rank)
            lines.append(f"- top-ranked variants with expired items: {labels}; inspect lifetime reports before interpreting delivery reliability.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_leaderboard()
    if not rows:
        print("No summary.csv files found in configured stages.", file=sys.stderr)
        return 1
    _write_csv(INDEX_DIR / "leaderboard.csv", rows)
    write_markdown(INDEX_DIR / "leaderboard.md", rows)
    write_metric_consistency_check(INDEX_DIR / "metric_consistency_check.md", rows)
    print(f"Wrote {len(rows)} leaderboard rows to {INDEX_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
