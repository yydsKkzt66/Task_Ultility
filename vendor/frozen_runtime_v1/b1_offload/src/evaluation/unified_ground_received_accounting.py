"""Offline, policy-independent accounting for ground-received task outcomes."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable

from src.payload.information_sufficiency import product_information_utility
from src.reward.timeliness import compute_timeliness


def requirement_target(requirement: str, utility_cfg: dict[str, Any] | None = None) -> float:
    return product_information_utility({"metadata", "roi_context"}, requirement, utility_cfg)


def incremental_ground_events(events: Iterable[dict[str, Any]], requirement: str, utility_cfg=None) -> list[dict[str, Any]]:
    delivered: set[str] = set(); output = []
    for event in sorted(events, key=lambda row: (float(row["ground_available_time_s"]), str(row["product_type"]))):
        product = str(event["product_type"])
        before = product_information_utility(delivered, requirement, utility_cfg)
        after = product_information_utility(delivered | {product}, requirement, utility_cfg)
        delta = max(0.0, after - before) if product not in delivered else 0.0
        delivered.add(product)
        output.append({**event, "delta_information_utility": delta, "information_sufficiency": after})
    return output


def score_task(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    completed = bool(row.get("ground_completed", 0))
    latency = float(row.get("capture_to_ground_latency_s", 0.0) or 0.0)
    deadline_s = max(0.0, float(row["deadline_time_s"]) - float(row["capture_time_s"]))
    timeliness = compute_timeliness(latency, deadline_s) if completed else 0.0
    info = float(row.get("ground_information_sufficiency", 0.0) or 0.0)
    quality = float(row.get("ground_result_quality", 0.0) or 0.0)
    gross = float(row.get("effective_task_value", 0.0)) * info * quality * timeliness if completed else float(row.get("partial_ground_gross_utility", 0.0) or 0.0)
    return {**row, "timeliness_factor": timeliness, "gross_ground_utility": gross}


def common_net(gross: float, controllable_wh: float, terminal_backlog_mb: float, violations: float, cfg: dict[str, Any]) -> dict[str, float]:
    e_norm = controllable_wh / max(float(cfg["energy_normalization"]["controllable_mission_energy_wh"]), 1e-12)
    q_norm = terminal_backlog_mb / max(float(cfg["backlog_normalization"]["terminal_backlog_mb"]), 1e-12)
    e_cost = float(cfg["lambda_E_common"]) * e_norm
    q_cost = float(cfg["lambda_Q_common"]) * q_norm
    safe_cost = float(cfg["lambda_safe_common"]) * violations
    return {"common_ground_gross_utility": gross, "common_controllable_energy_cost": e_cost, "common_terminal_backlog_cost": q_cost, "common_hard_safety_cost": safe_cost, "common_ground_net_utility": gross-e_cost-q_cost-safe_cost}


def summarize_method(rows: list[dict[str, Any]], energy: dict[str, float], safety: dict[str, float], internal: dict[str, float], cfg: dict[str, Any]) -> dict[str, Any]:
    arrived=len(rows); completed=[r for r in rows if int(r["ground_completed"])]; success=[r for r in rows if int(r["on_time_task_success"])]; high=[r for r in rows if str(r["value_level"]) in {"high_value","emergency"}]; high_c=[r for r in high if int(r["ground_completed"])]; high_s=[r for r in high if int(r["on_time_task_success"])]
    lat=[float(r["capture_to_ground_latency_s"]) for r in completed]; slat=[float(r["capture_to_ground_latency_s"]) for r in success]
    gross=sum(float(r["gross_ground_utility"]) for r in rows); net=common_net(gross,energy["controllable_mission_energy_wh"],safety["terminal_queue_mb"],safety["hard_safety_violation_count"],cfg)
    def mean(vals): return statistics.fmean(vals) if vals else 0.0
    return {"method":rows[0]["method"],"variant":rows[0]["variant"],"seed":rows[0]["seed"],"arrived_task_count":arrived,"ground_completed_count":len(completed),"on_time_ground_completed_count":sum(int(r["on_time_ground_completed"]) for r in rows),"requirement_satisfied_count":sum(int(r["requirement_satisfied"]) for r in rows),"on_time_task_success_count":len(success),"ground_completed_ratio_all_arrivals":len(completed)/max(arrived,1),"on_time_task_success_ratio_all_arrivals":len(success)/max(arrived,1),"deadline_satisfaction_ratio_among_ground_completed":sum(int(r["on_time_ground_completed"]) for r in completed)/max(len(completed),1),"average_latency_all_completed":mean(lat),"average_latency_on_time_success":mean(slat),"median_latency_all_completed":statistics.median(lat) if lat else 0.0,"p95_latency_all_completed":sorted(lat)[min(len(lat)-1,math.ceil(.95*len(lat))-1)] if lat else 0.0,"conditional_ground_received_quality":mean([float(r["ground_result_quality"]) for r in completed]),"arrival_normalized_ground_quality":sum(float(r["ground_result_quality"]) if int(r["ground_completed"]) else 0 for r in rows)/max(arrived,1),"average_on_time_success_quality":mean([float(r["ground_result_quality"]) for r in success]),**net,**internal,**energy,"delivered_mb":energy.get("delivered_mb",0.0),"energy_per_ground_completed_task":energy["controllable_mission_energy_wh"]/max(len(completed),1),"energy_per_on_time_task_success":energy["controllable_mission_energy_wh"]/max(len(success),1),"gross_utility_per_wh_controllable":gross/max(energy["controllable_mission_energy_wh"],1e-12),"gross_utility_per_delivered_mb":gross/max(energy.get("delivered_mb",0.0),1e-12),"high_value_arrived_count":len(high),"high_value_ground_completed_count":len(high_c),"high_value_on_time_success_count":len(high_s),"high_value_ground_completed_ratio":len(high_c)/max(len(high),1),"high_value_on_time_success_ratio":len(high_s)/max(len(high),1),"high_value_average_latency":mean([float(r["capture_to_ground_latency_s"]) for r in high_c]),"high_value_average_quality":mean([float(r["ground_result_quality"]) for r in high_c]),"high_value_gross_ground_utility":sum(float(r["gross_ground_utility"]) for r in high),"high_value_common_net_utility":sum(float(r["gross_ground_utility"]) for r in high)-net["common_controllable_energy_cost"]*(len(high)/max(arrived,1)),**safety}

