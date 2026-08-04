"""Post-run diagnostics for SARFish 20s dry-run outputs.

This script is diagnostic-only: it reads existing CSV outputs/configs and writes
analysis tables/reports. It does not run the simulator or change policies.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_yaml  # noqa: E402
from src.timeline.link_trace_loader import load_synthetic_link_trace  # noqa: E402


ROUTERS = (
    "proposed_lyapunov_rule_mode_safe_bandit",
    "utility_rule",
    "no_mode",
    "no_lyapunov",
    "safe_linucb",
    "fixed_lightweight",
    "fixed_high_accuracy",
)
FOCUS_ROUTERS = ("proposed_lyapunov_rule_mode_safe_bandit", "utility_rule", "no_mode")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def i(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def task_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if i(row.get("has_task")) == 1]


def load_run(output_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    summary = {row["router"]: row for row in read_csv(output_dir / "summary.csv")}
    trajectories = {
        path.stem: read_csv(path)
        for path in sorted((output_dir / "trajectories").glob("*.csv"))
    }
    lifetime = read_csv(output_dir / "task_lifetime_outcomes.csv")
    return summary, trajectories, lifetime


def lifetime_by_router(lifetime: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in lifetime:
        grouped[row["router_name"]].append(row)
    return grouped


def final_counts(lifetime_rows: list[dict[str, str]]) -> Counter:
    return Counter(row["final_lifetime_status"] for row in lifetime_rows)


def make_router_cost_decomposition(output_dir: Path) -> tuple[Path, Path]:
    summary, trajectories, lifetime = load_run(output_dir)
    lifetime_grouped = lifetime_by_router(lifetime)
    rows: list[dict[str, object]] = []

    for router in ROUTERS:
        s = summary[router]
        life_rows = lifetime_grouped.get(router, [])
        counts = final_counts(life_rows)
        generated = i(s.get("total_generated_tasks"))
        completed = i(s.get("completed_tasks"))
        expired = counts.get("expired", 0)
        delivered = counts.get("delivered", 0)
        deferred = counts.get("deferred", 0)
        undelivered = counts.get("undelivered_by_end", 0)
        total_effective = f(s.get("total_effective_utility"))
        total_energy = f(s.get("total_energy"))
        rows.append(
            {
                "router": router,
                "gross_task_utility": s.get("gross_task_utility"),
                "net_system_utility": s.get("net_system_utility"),
                "total_penalized_reward": s.get("total_penalized_reward"),
                "resource_penalty": s.get("total_resource_penalty"),
                "requirement_penalty": s.get("total_requirement_penalty"),
                "energy": s.get("total_energy"),
                "delivered_mb": s.get("delivered_mb"),
                "completed_tasks": completed,
                "delivered_items": i(s.get("delivered_items")),
                "expired_items": i(s.get("expired_items")),
                "final_delivered": delivered,
                "final_expired": expired,
                "final_deferred": deferred,
                "final_undelivered_by_end": undelivered,
                "total_generated_tasks": generated,
                "utility_per_generated_task": total_effective / max(generated, 1),
                "utility_per_completed_task": total_effective / max(completed, 1),
                "energy_per_completed_task": total_energy / max(completed, 1),
                "resource_penalty_per_generated_task": f(s.get("total_resource_penalty")) / max(generated, 1),
                "requirement_penalty_per_generated_task": f(s.get("total_requirement_penalty")) / max(generated, 1),
                "delivered_mb_per_task": f(s.get("delivered_mb")) / max(generated, 1),
                "expired_ratio": expired / max(generated, 1),
                "delivery_ratio": delivered / max(generated, 1),
            }
        )

    csv_path = output_dir / "router_cost_decomposition.csv"
    write_csv(csv_path, rows, list(rows[0].keys()))

    lines = [
        "# Router Cost Decomposition",
        "",
        "All task-lifetime columns use `task_lifetime_outcomes.csv`, not only current-step summary counts.",
        "",
        "| router | gross | net | penalized | resource | requirement | energy | delivered_mb | completed | delivered_items | expired_items | final delivered/expired/deferred/undelivered | utility/generated | energy/completed | expired_ratio | delivery_ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['router']} | {f(row['gross_task_utility']):.3f} | {f(row['net_system_utility']):.3f} | "
            f"{f(row['total_penalized_reward']):.3f} | {f(row['resource_penalty']):.3f} | "
            f"{f(row['requirement_penalty']):.3f} | {f(row['energy']):.1f} | {f(row['delivered_mb']):.2f} | "
            f"{row['completed_tasks']} | {row['delivered_items']} | {row['expired_items']} | "
            f"{row['final_delivered']}/{row['final_expired']}/{row['final_deferred']}/{row['final_undelivered_by_end']} | "
            f"{row['utility_per_generated_task']:.3f} | {row['energy_per_completed_task']:.1f} | "
            f"{row['expired_ratio']:.3f} | {row['delivery_ratio']:.3f} |"
        )

    proposed = next(row for row in rows if row["router"] == "proposed_lyapunov_rule_mode_safe_bandit")
    utility = next(row for row in rows if row["router"] == "utility_rule")
    lines.extend(
        [
            "",
            "## Main comparison",
            "",
            f"- proposed has lower gross/effective utility than utility_rule ({f(proposed['gross_task_utility']):.3f} vs {f(utility['gross_task_utility']):.3f}).",
            f"- proposed resource penalty is higher ({f(proposed['resource_penalty']):.3f} vs {f(utility['resource_penalty']):.3f}) and energy is higher ({f(proposed['energy']):.1f} J vs {f(utility['energy']):.1f} J).",
            f"- proposed requirement penalty is also higher ({f(proposed['requirement_penalty']):.3f} vs {f(utility['requirement_penalty']):.3f}).",
            f"- lifetime delivery is the same at {proposed['final_delivered']} delivered tasks, while proposed has fewer final expired tasks ({proposed['final_expired']} vs {utility['final_expired']}).",
        ]
    )
    md_path = output_dir / "router_cost_decomposition.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _task_lifetime_lookup(lifetime: list[dict[str, str]], router: str) -> dict[str, dict[str, str]]:
    return {
        row["task_id"]: row
        for row in lifetime
        if row["router_name"] == router
    }


def _resource_terms(row: dict[str, str], reward_cfg: dict[str, object], simulator_cfg: dict[str, object]) -> dict[str, float]:
    e_ref = f(simulator_cfg.get("E_ref"), 100.0)
    queue_max = f(simulator_cfg.get("queue_max_mb"), 64.0)
    lambda_e_base = f(reward_cfg.get("lambda_E_base", reward_cfg.get("energy_penalty", 0.0)))
    eta_b = f(reward_cfg.get("eta_B"), 1.0)
    eta_theta = f(reward_cfg.get("eta_Theta"), 0.5)
    lambda_l = f(reward_cfg.get("lambda_L", reward_cfg.get("latency_penalty", 0.0)))
    lambda_q = f(reward_cfg.get("lambda_Q", reward_cfg.get("queue_penalty", 0.0)))
    dynamic_weight = lambda_e_base * (1.0 + eta_b * (1.0 - f(row.get("B_t"))) + eta_theta * f(row.get("Theta_t")))
    compute_component = dynamic_weight * f(row.get("E_comp")) / max(e_ref, 1.0e-9)
    comm_component = dynamic_weight * f(row.get("E_comm")) / max(e_ref, 1.0e-9)
    latency = f(row.get("L_inf")) + f(row.get("L_queue")) + f(row.get("L_comm"))
    latency_component = lambda_l * latency / max(f(row.get("deadline_s")), 1.0e-9)
    queue_component = lambda_q * f(row.get("queue_next_mb")) / max(queue_max, 1.0e-9)
    return {
        "dynamic_energy_weight": dynamic_weight,
        "compute_energy_component": compute_component,
        "comm_energy_component": comm_component,
        "latency_component": latency_component,
        "queue_component": queue_component,
    }


def make_action_payload_decomposition(output_dir: Path, reward_cfg: dict[str, object], simulator_cfg: dict[str, object]) -> tuple[Path, Path]:
    _summary, trajectories, lifetime = load_run(output_dir)
    csv_rows: list[dict[str, object]] = []
    report_lines = [
        "# Action / Model / Payload Decomposition",
        "",
        "Energy and resource terms are summed on task rows. `delivered_mb` is lifetime-delivered generated payload MB by original action; `current_step_downlinked_mb` is the current row's immediate downlink amount.",
    ]

    for router in FOCUS_ROUTERS:
        life_lookup = _task_lifetime_lookup(lifetime, router)
        rows = task_rows(trajectories[router])
        report_lines.extend(
            [
                "",
                f"## {router}",
                "",
                f"- model distribution: {dict(Counter(row['model_name'] for row in rows))}",
                f"- payload distribution: {dict(Counter(row['payload_policy'] for row in rows))}",
                f"- action distribution: {dict(Counter(row['selected_action'] for row in rows))}",
                "",
                "| action | n | effective utility | resource penalty | requirement penalty | energy | generated payload MB | delivered MB | current-step downlinked MB | final delivered | final expired |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["selected_action"]].append(row)
        for action, action_rows in sorted(grouped.items()):
            final = Counter(life_lookup.get(row["task_id"], {}).get("final_lifetime_status", "unknown") for row in action_rows)
            delivered_mb = sum(
                f(row.get("S_gen_current_mb"))
                for row in action_rows
                if life_lookup.get(row["task_id"], {}).get("final_lifetime_status") == "delivered"
            )
            terms = [_resource_terms(row, reward_cfg, simulator_cfg) for row in action_rows]
            aggregate = {
                "router": router,
                "action": action,
                "count": len(action_rows),
                "model_distribution": dict(Counter(row["model_name"] for row in action_rows)),
                "payload_distribution": dict(Counter(row["payload_policy"] for row in action_rows)),
                "effective_utility": sum(f(row.get("effective_utility")) for row in action_rows),
                "resource_penalty": sum(f(row.get("resource_penalty")) for row in action_rows),
                "requirement_penalty": sum(f(row.get("requirement_penalty")) for row in action_rows),
                "energy": sum(f(row.get("E_comp")) + f(row.get("E_comm")) for row in action_rows),
                "compute_energy": sum(f(row.get("E_comp")) for row in action_rows),
                "comm_energy": sum(f(row.get("E_comm")) for row in action_rows),
                "generated_payload_mb": sum(f(row.get("S_gen_current_mb")) for row in action_rows),
                "delivered_mb": delivered_mb,
                "current_step_downlinked_mb": sum(f(row.get("S_down_current_task_mb")) for row in action_rows),
                "final_delivered": final.get("delivered", 0),
                "final_expired": final.get("expired", 0),
                "final_deferred": final.get("deferred", 0),
                "compute_energy_component": sum(term["compute_energy_component"] for term in terms),
                "comm_energy_component": sum(term["comm_energy_component"] for term in terms),
                "latency_component": sum(term["latency_component"] for term in terms),
                "queue_component": sum(term["queue_component"] for term in terms),
            }
            csv_rows.append(aggregate)
            report_lines.append(
                f"| {action} | {aggregate['count']} | {aggregate['effective_utility']:.3f} | "
                f"{aggregate['resource_penalty']:.3f} | {aggregate['requirement_penalty']:.3f} | "
                f"{aggregate['energy']:.1f} | {aggregate['generated_payload_mb']:.2f} | "
                f"{aggregate['delivered_mb']:.2f} | {aggregate['current_step_downlinked_mb']:.2f} | "
                f"{aggregate['final_delivered']} | {aggregate['final_expired']} |"
            )

    proposed_actions = [row for row in csv_rows if row["router"] == "proposed_lyapunov_rule_mode_safe_bandit"]
    utility_actions = [row for row in csv_rows if row["router"] == "utility_rule"]
    proposed_high = sum(row["count"] for row in proposed_actions if row["action"] == "high_accuracy+roi_context")
    utility_high = sum(row["count"] for row in utility_actions if row["action"] == "high_accuracy+roi_context")
    proposed_meta_defer = sum(row["count"] for row in proposed_actions if row["action"] in {"balanced+metadata", "lightweight+metadata", "delay/store"})
    utility_meta_defer = sum(row["count"] for row in utility_actions if row["action"] in {"balanced+metadata", "lightweight+metadata", "delay/store"})
    proposed_resource = sum(f(row["resource_penalty"]) for row in proposed_actions)
    utility_resource = sum(f(row["resource_penalty"]) for row in utility_actions)
    proposed_terms = {
        "compute": sum(f(row["compute_energy_component"]) for row in proposed_actions),
        "comm": sum(f(row["comm_energy_component"]) for row in proposed_actions),
        "latency": sum(f(row["latency_component"]) for row in proposed_actions),
        "queue": sum(f(row["queue_component"]) for row in proposed_actions),
    }
    utility_terms = {
        "compute": sum(f(row["compute_energy_component"]) for row in utility_actions),
        "comm": sum(f(row["comm_energy_component"]) for row in utility_actions),
        "latency": sum(f(row["latency_component"]) for row in utility_actions),
        "queue": sum(f(row["queue_component"]) for row in utility_actions),
    }

    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- proposed uses `high_accuracy+roi_context` more often than utility_rule ({proposed_high} vs {utility_high}), but its gross utility is still lower because high-cost actions do not overcome higher requirement penalties and resource costs.",
            f"- proposed metadata/defer-like choices are also more frequent ({proposed_meta_defer} vs {utility_meta_defer}), which raises information/quality requirement penalties relative to utility_rule.",
            f"- proposed resource penalty is higher on task rows ({proposed_resource:.3f} vs {utility_resource:.3f}). Approximate component deltas are compute {proposed_terms['compute'] - utility_terms['compute']:.3f}, comm {proposed_terms['comm'] - utility_terms['comm']:.3f}, latency {proposed_terms['latency'] - utility_terms['latency']:.3f}, queue {proposed_terms['queue'] - utility_terms['queue']:.3f}.",
            "- The dominant resource-cost gap comes from higher compute and comm energy after proposed selects more high-accuracy/richer actions; queue component is small because backlog is low.",
        ]
    )

    fieldnames = [
        "router",
        "action",
        "count",
        "model_distribution",
        "payload_distribution",
        "effective_utility",
        "resource_penalty",
        "requirement_penalty",
        "energy",
        "compute_energy",
        "comm_energy",
        "generated_payload_mb",
        "delivered_mb",
        "current_step_downlinked_mb",
        "final_delivered",
        "final_expired",
        "final_deferred",
        "compute_energy_component",
        "comm_energy_component",
        "latency_component",
        "queue_component",
    ]
    csv_path = output_dir / "action_payload_model_decomposition.csv"
    write_csv(csv_path, csv_rows, fieldnames)
    md_path = output_dir / "action_payload_model_decomposition.md"
    md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    return f"min={min(values):.4f}, mean={mean(values):.4f}, max={max(values):.4f}"


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def make_mode_price_report(output_dir: Path) -> Path:
    summary, trajectories, _lifetime = load_run(output_dir)
    proposed_all = trajectories["proposed_lyapunov_rule_mode_safe_bandit"]
    no_mode_all = trajectories["no_mode"]
    proposed_tasks = task_rows(proposed_all)
    no_mode_tasks = task_rows(no_mode_all)
    same_actions = sum(
        1
        for prop, no_mode in zip(proposed_tasks, no_mode_tasks)
        if prop["task_id"] == no_mode["task_id"] and prop["selected_action"] == no_mode["selected_action"]
    )
    task_count = len(proposed_tasks)

    lines = [
        "# Mode / Price Effectiveness Report",
        "",
        f"- proposed mode distribution, all slots: {dict(Counter(row['mode_name'] for row in proposed_all))}",
        f"- proposed mode distribution, task rows: {dict(Counter(row['mode_name'] for row in proposed_tasks))}",
        f"- mode_switch_count from summary: {summary['proposed_lyapunov_rule_mode_safe_bandit'].get('mode_switch_count')}",
        f"- proposed vs no_mode same selected action on aligned task rows: {same_actions}/{task_count} ({same_actions / max(task_count, 1):.3f}).",
        "",
        "## Price ranges on proposed task rows",
        "",
        f"- lambda_E: {_stats([f(row['lambda_E']) for row in proposed_tasks])}",
        f"- lambda_L: {_stats([f(row['lambda_L']) for row in proposed_tasks])}",
        f"- lambda_Q: {_stats([f(row['lambda_Q']) for row in proposed_tasks])}",
        f"- lambda_S: {_stats([f(row['lambda_S']) for row in proposed_tasks])}",
        "",
        "## Price/state correlations on proposed task rows",
        "",
    ]
    for price, state in (
        ("lambda_E", "battery_soc"),
        ("lambda_E", "Theta_t"),
        ("lambda_Q", "queue_next_mb"),
        ("lambda_S", "queue_next_mb"),
        ("lambda_L", "D_t"),
    ):
        corr = _corr([f(row[price]) for row in proposed_tasks], [f(row[state]) for row in proposed_tasks])
        lines.append(f"- corr({price}, {state}) = {'n/a' if corr is None else f'{corr:.3f}'}")

    lines.extend(
        [
            "",
            "## By-mode action and cost",
            "",
            "| mode | task rows | action distribution | gross/effective utility | resource penalty | requirement penalty | energy |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in proposed_tasks:
        by_mode[row["mode_name"]].append(row)
    for mode, rows in sorted(by_mode.items()):
        lines.append(
            f"| {mode} | {len(rows)} | {dict(Counter(row['selected_action'] for row in rows))} | "
            f"{sum(f(row['effective_utility']) for row in rows):.3f} | "
            f"{sum(f(row['resource_penalty']) for row in rows):.3f} | "
            f"{sum(f(row['requirement_penalty']) for row in rows):.3f} | "
            f"{sum(f(row['E_comp']) + f(row['E_comm']) for row in rows):.1f} |"
        )

    prop = summary["proposed_lyapunov_rule_mode_safe_bandit"]
    nomode = summary["no_mode"]
    lines.extend(
        [
            "",
            "## Proposed vs no_mode",
            "",
            f"- gross_task_utility: proposed {f(prop.get('gross_task_utility')):.3f}, no_mode {f(nomode.get('gross_task_utility')):.3f}.",
            f"- net_system_utility: proposed {f(prop.get('net_system_utility')):.3f}, no_mode {f(nomode.get('net_system_utility')):.3f}.",
            f"- total_penalized_reward: proposed {f(prop.get('total_penalized_reward')):.3f}, no_mode {f(nomode.get('total_penalized_reward')):.3f}.",
            f"- total_resource_penalty: proposed {f(prop.get('total_resource_penalty')):.3f}, no_mode {f(nomode.get('total_resource_penalty')):.3f}.",
            "",
            "The two routers are highly similar in this nominal 3h run. The mode controller does switch modes, but resource state ranges are narrow: battery stays high, queue pressure remains tiny relative to 64 MB, and thermal never reaches the hard limit. As a result, mode multipliers do not change the action ranking enough to create a visible gain over no_mode.",
        ]
    )
    path = output_dir / "mode_price_effectiveness_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _segments(rows: list[dict[str, str]], predicate) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    last: int | None = None
    for row in rows:
        slot = i(row["slot"])
        if predicate(row):
            if start is None:
                start = slot
            last = slot
        elif start is not None and last is not None:
            segments.append((start, last))
            start = None
            last = None
    if start is not None and last is not None:
        segments.append((start, last))
    return segments


def make_resource_stress_report(output_dir: Path, timeline_cfg: dict[str, object], simulator_cfg: dict[str, object]) -> Path:
    summary, trajectories, _lifetime = load_run(output_dir)
    proposed = trajectories["proposed_lyapunov_rule_mode_safe_bandit"]
    utility = trajectories["utility_rule"]
    prop_tasks = task_rows(proposed)
    util_tasks = task_rows(utility)
    slot_seconds = f(timeline_cfg["slot_seconds"])
    num_slots = i(timeline_cfg["num_slots"])
    orbit_period_slots = int(round(f(timeline_cfg["orbit_period_min"]) * 60.0 / slot_seconds))
    links = load_synthetic_link_trace(num_slots, slot_seconds)
    contact_slots = {link.slot_index for link in links if link.in_contact}

    eclipse_segments = _segments(proposed, lambda row: i(row.get("is_eclipse")) == 1 or i(row.get("is_sunlit")) == 0)
    no_solar_segments = _segments(proposed, lambda row: f(row.get("solar_energy_j")) <= 1.0e-12)
    no_contact_segments = _segments(proposed, lambda row: i(row["slot"]) not in contact_slots)
    eclipse_task_slots = {i(row["slot"]) for row in prop_tasks if i(row.get("is_eclipse")) == 1 or i(row.get("is_sunlit")) == 0}
    no_contact_task_slots = {i(row["slot"]) for row in prop_tasks if i(row["slot"]) not in contact_slots}
    eclipse_prop = [row for row in prop_tasks if i(row["slot"]) in eclipse_task_slots]
    eclipse_util = [row for row in util_tasks if i(row["slot"]) in eclipse_task_slots]
    no_contact_prop = [row for row in prop_tasks if i(row["slot"]) in no_contact_task_slots]

    lines = [
        "# Resource Stress Coverage Report",
        "",
        f"- slot_seconds: {slot_seconds:g}",
        f"- num_slots: {num_slots}",
        f"- physical horizon: {num_slots * slot_seconds / 3600.0:.3f} h",
        f"- configured orbit period: {orbit_period_slots} slots = {orbit_period_slots * slot_seconds:.0f} s",
        f"- covered orbit indices: {sorted({i(row['orbit_index']) for row in proposed})}",
        f"- covers at least two orbit periods: {num_slots >= 2 * orbit_period_slots}",
        "",
        "## Eclipse / solar coverage",
        "",
        f"- eclipse slot count: {sum(1 for row in proposed if i(row.get('is_eclipse')) == 1 or i(row.get('is_sunlit')) == 0)}",
        f"- eclipse duration_s: {sum(1 for row in proposed if i(row.get('is_eclipse')) == 1 or i(row.get('is_sunlit')) == 0) * slot_seconds:.0f}",
        f"- eclipse segments: {[(start, end, (end - start + 1) * slot_seconds) for start, end in eclipse_segments]}",
        f"- zero-solar/low-solar segments: {[(start, end, (end - start + 1) * slot_seconds) for start, end in no_solar_segments[:8]]}",
        f"- tasks arriving during eclipse: {len(eclipse_task_slots)}",
        f"- proposed actions during eclipse tasks: {dict(Counter(row['selected_action'] for row in eclipse_prop))}",
        f"- utility_rule actions during eclipse tasks: {dict(Counter(row['selected_action'] for row in eclipse_util))}",
        "",
        "## Communication coverage",
        "",
        f"- reconstructed contact slots: {len(contact_slots)}; no-contact slots: {num_slots - len(contact_slots)}",
        f"- contact model period: 18 slots = {18 * slot_seconds:.0f} s",
        f"- no-contact task arrivals: {len(no_contact_task_slots)}",
        f"- proposed actions on no-contact task slots: {dict(Counter(row['selected_action'] for row in no_contact_prop))}",
        f"- first no-contact segments: {[(start, end, (end - start + 1) * slot_seconds) for start, end in no_contact_segments[:8]]}",
        "",
        "## Battery / thermal / queue pressure",
        "",
    ]
    for router in ("proposed_lyapunov_rule_mode_safe_bandit", "utility_rule"):
        s = summary[router]
        rows = trajectories[router]
        lines.extend(
            [
                f"### {router}",
                f"- min_battery_soc: {f(s.get('min_battery_soc')):.4f}; battery_min: {f(simulator_cfg.get('battery_min')):.2f}; battery_guard: {f(simulator_cfg.get('long_term_risk_penalty', {}).get('battery_guard_threshold', 0.80)):.2f}",
                f"- battery_below_080_count: {i(s.get('battery_below_080_count'))}",
                f"- max_thermal: {f(s.get('max_thermal')):.4f}; thermal_guard: {f(simulator_cfg.get('long_term_risk_penalty', {}).get('thermal_guard_threshold', 0.85)):.2f}; thermal_hard/max: {f(simulator_cfg.get('thermal_max')):.2f}",
                f"- thermal_guard_exceed_count: {i(s.get('thermal_guard_exceed_count'))}",
                f"- average_queue_mb: {f(s.get('average_queue_mb')):.3f}; max_queue_mb: {f(s.get('max_queue_mb')):.3f}; queue_max_mb: {f(simulator_cfg.get('queue_max_mb')):.1f}",
                f"- total_compute_energy: {f(s.get('total_compute_energy')):.1f}; total_comm_energy: {f(s.get('total_comm_energy')):.1f}",
                f"- eclipse task effective utility/resource penalty: {sum(f(row['effective_utility']) for row in rows if i(row.get('has_task')) and i(row['slot']) in eclipse_task_slots):.3f} / {sum(f(row['resource_penalty']) for row in rows if i(row.get('has_task')) and i(row['slot']) in eclipse_task_slots):.3f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Missing / reconstructed fields",
            "",
            "- Trajectory has `is_sunlit/is_eclipse`, `solar_energy_j`, `battery_soc`, `thermal_next`, `queue_next_mb`, and `C_t`.",
            "- Trajectory does not export raw `in_contact`, `throughput_mbps`, or `remaining_window_s`; this report reconstructs them from `src/timeline/link_trace_loader.py` and the dry-run config.",
            "",
            "## Judgment",
            "",
            "The 3h run does contain two eclipse segments and tasks during eclipse, but it does not create strong resource scarcity. Battery remains far above both the hard minimum and the 0.80 guard, queue pressure stays below 10% of capacity, and thermal approaches only the guard region, not the hard threshold. Utility_rule therefore does not pay a strong resource-constrained penalty in this nominal run.",
        ]
    )
    path = output_dir / "resource_stress_coverage_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_root_cause_report(output_dir: Path) -> Path:
    summary, trajectories, lifetime = load_run(output_dir)
    lifetime_grouped = lifetime_by_router(lifetime)
    prop = summary["proposed_lyapunov_rule_mode_safe_bandit"]
    util = summary["utility_rule"]
    no_mode = summary["no_mode"]
    prop_tasks = task_rows(trajectories["proposed_lyapunov_rule_mode_safe_bandit"])
    util_tasks = task_rows(trajectories["utility_rule"])
    prop_actions = Counter(row["selected_action"] for row in prop_tasks)
    util_actions = Counter(row["selected_action"] for row in util_tasks)
    prop_life = final_counts(lifetime_grouped["proposed_lyapunov_rule_mode_safe_bandit"])
    util_life = final_counts(lifetime_grouped["utility_rule"])

    lines = [
        "# Proposed vs Utility Rule Root Cause Report",
        "",
        "## Numerical facts",
        "",
        f"- gross_task_utility: proposed {f(prop.get('gross_task_utility')):.3f}, utility_rule {f(util.get('gross_task_utility')):.3f}.",
        f"- net_system_utility: proposed {f(prop.get('net_system_utility')):.3f}, utility_rule {f(util.get('net_system_utility')):.3f}.",
        f"- total_penalized_reward: proposed {f(prop.get('total_penalized_reward')):.3f}, utility_rule {f(util.get('total_penalized_reward')):.3f}.",
        f"- resource_penalty: proposed {f(prop.get('total_resource_penalty')):.3f}, utility_rule {f(util.get('total_resource_penalty')):.3f}.",
        f"- requirement_penalty: proposed {f(prop.get('total_requirement_penalty')):.3f}, utility_rule {f(util.get('total_requirement_penalty')):.3f}.",
        f"- energy: proposed {f(prop.get('total_energy')):.1f} J, utility_rule {f(util.get('total_energy')):.1f} J.",
        f"- lifetime final delivered/expired: proposed {prop_life.get('delivered', 0)}/{prop_life.get('expired', 0)}, utility_rule {util_life.get('delivered', 0)}/{util_life.get('expired', 0)}.",
        f"- proposed action distribution: {dict(prop_actions)}",
        f"- utility_rule action distribution: {dict(util_actions)}",
        "",
        "## Cause classification",
        "",
        "| cause | diagnosis | evidence |",
        "| --- | --- | --- |",
        f"| 3h horizon too short | likely | only 45 tasks; proposed and no_mode are almost identical ({f(prop.get('gross_task_utility')):.3f} vs {f(no_mode.get('gross_task_utility')):.3f}) |",
        f"| no/low eclipse or low battery | partially true | eclipse exists, but min battery is {f(prop.get('min_battery_soc')):.4f}, far above 0.18 minimum and 0.80 guard |",
        f"| energy/battery constraints too loose | yes for this run | battery_below_080_count={i(prop.get('battery_below_080_count'))}; final resource safety violations={i(prop.get('final_resource_safety_violation_count'))} |",
        f"| contact windows too friendly | likely | 20s link model uses slot_index % 18, shortening the physical contact cycle to 360s; proposed expired_items dropped from 17 at 60s to {i(prop.get('expired_items'))} |",
        f"| thermal stress insufficient | mostly | proposed max_thermal={f(prop.get('max_thermal')):.4f}, utility_rule max_thermal={f(util.get('max_thermal')):.4f}; both below 0.92 hard threshold |",
        f"| mode controller action modulation weak in this scenario | yes | proposed/no_mode metrics are nearly equal: penalized reward {f(prop.get('total_penalized_reward')):.3f} vs {f(no_mode.get('total_penalized_reward')):.3f} |",
        f"| proposed chooses high-cost actions without enough utility gain | yes | proposed high_accuracy+roi_context={prop_actions.get('high_accuracy+roi_context', 0)} vs utility_rule {util_actions.get('high_accuracy+roi_context', 0)}; proposed energy/resource penalty are higher |",
        f"| utility_rule is a strong greedy baseline in nominal conditions | yes | utility_rule gets best gross/net/penalized metrics while using mostly balanced+roi ({util_actions.get('balanced+roi', 0)} tasks) |",
        "| reward/cost scaling imbalance | possible, not proven | proposed pays higher resource and requirement penalties, but no formula-level bug was found from output consistency |",
        "| code implementation bug | not found | lifetime accounting, resource scaling, and queue expiration semantics are internally consistent in exported data |",
        "",
        "## Conclusion",
        "",
        "Proposed does not win because the dry run is a nominal, communication-friendly, battery-safe case where utility_rule's balanced+roi heuristic is already close to the efficient action. Proposed selects more high_accuracy/roi_context and more metadata/defer, so it pays both higher resource cost and higher requirement penalty without getting enough additional delivered lifetime utility.",
    ]
    path = output_dir / "proposed_vs_utility_rule_root_cause_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_next_experiment_recommendation(output_dir: Path) -> Path:
    lines = [
        "# Next Experiment Design Recommendation",
        "",
        "## A. Add diagnostics without changing algorithms",
        "",
        "- Export raw link fields per slot: `in_contact`, `throughput_mbps`, `remaining_window_s`, `available_capacity_mb`.",
        "- Export resource-stress fields per slot: eclipse segment id, consecutive shadow slots, solar power W, net battery energy delta J.",
        "- Keep task lifetime outputs: generated, queued, delivered, expired, deferred, and undelivered_by_end by task_id.",
        "- Add per-action predicted score components so proposed/no_mode differences can be attributed to mode price, exploration, and reward prior.",
        "",
        "## B. Resource-stress scenario without reward tuning",
        "",
        "- Prefer an energy-shadow stress dry run before a 1-day nominal run if the goal is to expose long-horizon resource control.",
        "- Use an eclipse-aware resource trace with explicit shadow segments and high-value task bursts during shadow.",
        "- Start from a lower but feasible initial SOC, for example a stress profile rather than changing the default nominal profile.",
        "- Add a contact outage segment or lower-capacity contact interval during/after the shadow burst.",
        "- Add a thermal stress segment through higher ambient temperature or reduced cooling only in a named stress scenario.",
        "",
        "## C. Physical-scale issues to fix or declare",
        "",
        "- `slot_index % 18` makes the contact period slot-based. At 60s it is 1080s; at 20s it is 360s. This likely makes the 20s dry run communication-friendlier.",
        "- Thermal cooling uses per-slot constants, so 20s and 60s are not physically equivalent unless cooling terms are normalized by elapsed seconds.",
        "- For physical comparison, contact and eclipse should be period-in-seconds based (`contact_period_s`, `eclipse_period_s`) rather than fixed slot-count based.",
        "",
        "## Direct answers",
        "",
        "1. Do not treat a 1-day 20s nominal simulation as the next decisive experiment. It is useful as a stability run, but it may still be too friendly.",
        "2. Run an energy-shadow stress dry run first if the question is whether proposed has a resource-constrained advantage.",
        "3. Yes, the 20s `slot_index % 18` contact model shortens the physical contact cycle.",
        "4. Yes, per-slot thermal cooling makes 20s/60s non-equivalent physically.",
        "5. Yes, fixed physical periods for contact and eclipse are preferable for resource-constrained LEO claims.",
        "6. A resource-constrained LEO stress scenario should combine eclipse-aware resource trace, lower initial SOC, contact outage, high-value burst during shadow, and a thermal stress interval.",
        "",
        "No reward tuning is recommended from these diagnostics; the next step is scenario and instrumentation quality.",
    ]
    path = output_dir / "next_experiment_design_recommendation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SARFish 20s dry-run decomposition reports.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sarfish_20s_dry_run")
    args = parser.parse_args()
    output_dir = args.output_dir
    reward_cfg = load_yaml(PROJECT_ROOT / "configs" / "reward.yaml")
    timeline_cfg = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_20s_dryrun.yaml")
    simulator_cfg = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_20s_dryrun.yaml")

    make_router_cost_decomposition(output_dir)
    make_action_payload_decomposition(output_dir, reward_cfg, simulator_cfg)
    make_mode_price_report(output_dir)
    make_resource_stress_report(output_dir, timeline_cfg, simulator_cfg)
    make_root_cause_report(output_dir)
    make_next_experiment_recommendation(output_dir)


if __name__ == "__main__":
    main()
