"""Adapters from B1/proposed traces to the unified task schema."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.evaluation.unified_ground_received_accounting import incremental_ground_events, score_task
from src.payload.information_sufficiency import product_information_utility

PRODUCT_TYPES = {"metadata", "roi", "roi_context"}


def _split(value: Any) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def requirement_satisfied_by_products(products: set[str], requirement: str) -> bool:
    requirement = str(requirement or "verification").strip().lower()
    if requirement == "alert":
        return bool(products & PRODUCT_TYPES)
    if requirement == "verification":
        return bool(products & {"roi", "roi_context"})
    if requirement in {"context", "evidence"}:
        return "roi_context" in products
    return bool(products & {"roi", "roi_context"})


def proposed_delivery_events(records, lifecycle, *, slot_seconds: float = 60.0):
    """Combine direct step-record events and staged lifecycle events without duplicates."""
    events = defaultdict(list)
    seen = set()
    for record in records:
        task_ids = _split(getattr(record, "delivered_task_ids", ""))
        products = _split(getattr(record, "delivered_product_types", ""))
        times = _split(getattr(record, "delivered_task_event_times_s", ""))
        if not products and getattr(record, "delivery_status", "") == "delivered" and getattr(record, "task_id", ""):
            task_ids = [str(record.task_id)]
            products = [str(getattr(record, "selected_product_type", ""))]
            delivery_time = float(getattr(record, "payload_delivery_time_s", -1.0))
            times = [str(delivery_time if delivery_time >= 0.0 else float(getattr(record, "slot", 0)) * slot_seconds)]
        for index, task_id in enumerate(task_ids):
            product = products[index] if index < len(products) else ""
            if product not in PRODUCT_TYPES:
                continue
            available = float(times[index]) if index < len(times) else float(getattr(record, "slot", 0)) * slot_seconds
            key = (task_id, product)
            if key in seen:
                continue
            seen.add(key)
            events[task_id].append({"product_type": product, "ground_available_time_s": available, "delivery_route": "direct_record"})
    for life in lifecycle:
        if str(life.get("current_status")) != "delivered" or life.get("delivered_slot", "") == "":
            continue
        task_id, product = str(life["task_id"]), str(life["product_type"])
        if product not in PRODUCT_TYPES:
            continue
        available = float(life["delivered_slot"]) * slot_seconds
        key = (task_id, product)
        if key in seen:
            continue
        seen.add(key)
        source = str(life.get("source", ""))
        route = "staged_lifecycle" if "staged_delivery_queue" in source or str(life.get("request_id", "")).startswith("staged_") else "direct_lifecycle"
        events[task_id].append({"product_type": product, "ground_available_time_s": available, "delivery_route": route})
    return events


def manifest_rows(timeline, manifest_hash: str):
    out = {}
    for step in timeline:
        if not step.task:
            continue
        task = step.task
        capture = step.slot_index * step.slot_seconds
        out[task.task_id] = {
            "task_id": task.task_id,
            "task_manifest_hash": manifest_hash,
            "capture_time_s": capture,
            "arrival_time_s": capture,
            "deadline_time_s": capture + task.deadline_s,
            "value_level": task.value_level,
            "effective_task_value": task.task_value,
            "product_requirement": task.product_requirement,
            "quality_requirement": task.quality_requirement,
            "information_requirement": task.information_requirement,
            "scene_id": task.scene_id,
            "logical_image_path": str((task.metadata or {}).get("logical_image_path", "")),
        }
    return out


def adapt_b1(manifest, trace, method, variant, seed, cfg):
    by = {row["task_id"]: row for row in trace}
    rows = []
    for task_id, metadata in manifest.items():
        record = by[task_id]
        completed = int(float(record.get("ground_result_available", 0)))
        available = float(record["ground_result_available_time_s"]) if completed else 0.0
        latency = float(record.get("capture_to_ground_latency_s") or 0)
        quality = float(record.get("ground_task_quality") or 0)
        reconstruction_quality = float(record.get("reconstructed_image_quality_proxy") or 0)
        information = float(cfg["full_frame_information_mapping"]) if completed else 0.0
        quality_pass = int(completed and reconstruction_quality + 1e-9 >= float(metadata.get("quality_requirement", 0.0)))
        information_pass = int(completed and information + 1e-9 >= float(metadata.get("information_requirement", 0.0)))
        requirement_satisfied = int(completed and quality_pass and information_pass)
        ontime_ground = int(completed and available <= float(metadata["deadline_time_s"]) + 1e-9)
        ontime = int(requirement_satisfied and available <= float(metadata["deadline_time_s"]) + 1e-9)
        base = {
            **metadata, "method": method, "variant": variant, "seed": seed,
            "transmitted_product_type": "compressed_full_frame", "highest_ground_available_stage": "full_frame_result" if completed else "empty",
            "ground_completed": completed, "ground_result_available_time_s": available if completed else "", "requirement_closure_time_s": available if requirement_satisfied else "",
            "on_time_ground_completed": ontime_ground, "requirement_satisfied": requirement_satisfied, "on_time_task_success": ontime, "partial_ground_delivery": int(completed and not requirement_satisfied),
            "capture_to_ground_latency_s": latency, "ground_model_quality": quality, "compression_quality_retention": reconstruction_quality,
            "compression_sufficiency_quality": reconstruction_quality, "compression_quality_pass": quality_pass,
            "full_frame_information_pass": information_pass, "ground_processing_success": completed,
            "ground_information_sufficiency": information, "ground_result_quality": quality if completed else 0.0,
            "partial_ground_gross_utility": 0.0, "compute_energy_wh": 0.0, "compression_energy_wh": float(record.get("compression_energy_wh") or 0),
            "communication_energy_wh": 0.0, "payload_idle_energy_wh": 0.0, "housekeeping_energy_wh": 0.0,
            "controllable_mission_energy_wh": float(record.get("compression_energy_wh") or 0), "total_spacecraft_energy_wh": float(record.get("compression_energy_wh") or 0),
            "expired": int(record.get("state") == "expired"), "dropped": 0, "terminal_backlog_flag": int(record.get("state") in {"compressed_queued", "transmitting"}),
            "failure_reason": record.get("blocked_reason") or ("expired" if record.get("state") == "expired" else ""),
        }
        rows.append(score_task(base, cfg))
    return rows


def adapt_proposed(manifest, records, lifecycle, method, variant, seed, cfg, utility_cfg=None, *, slot_seconds: float = 60.0):
    record_by_task = {record.task_id: record for record in records if record.has_new_task and record.task_id in manifest}
    events = proposed_delivery_events(records, lifecycle, slot_seconds=slot_seconds)
    rows = []
    for task_id, metadata in manifest.items():
        record = record_by_task.get(task_id)
        incremental = incremental_ground_events(events[task_id], metadata["product_requirement"], utility_cfg)
        products = {event["product_type"] for event in incremental}
        quality = float(getattr(record, "detection_quality", 0.0)) if record else 0.0
        partial = sum(
            float(metadata["effective_task_value"]) * float(event["delta_information_utility"]) * quality
            * max(0.0, min(1.0, 1.0 - (float(event["ground_available_time_s"]) - float(metadata["capture_time_s"])) / max(float(metadata["deadline_time_s"]) - float(metadata["capture_time_s"]), 1e-9)))
            for event in incremental
        )
        first = incremental[0] if incremental else None
        satisfied_event = None
        observed = set()
        for event in incremental:
            observed.add(event["product_type"])
            if requirement_satisfied_by_products(observed, metadata["product_requirement"]):
                satisfied_event = event
                break
        completed = int(first is not None)
        requirement_satisfied = int(satisfied_event is not None)
        available = float(first["ground_available_time_s"]) if first else 0.0
        closure_available = float(satisfied_event["ground_available_time_s"]) if satisfied_event else 0.0
        ontime_ground = int(completed and available <= float(metadata["deadline_time_s"]) + 1e-9)
        ontime_success = int(requirement_satisfied and closure_available <= float(metadata["deadline_time_s"]) + 1e-9)
        stage = "context_or_evidence" if "roi_context" in products else ("roi" if "roi" in products else ("metadata" if "metadata" in products else "empty"))
        base = {
            **metadata, "method": method, "variant": variant, "seed": seed,
            "transmitted_product_type": "+".join(sorted(products)) if products else "none", "highest_ground_available_stage": stage,
            "ground_completed": completed, "ground_result_available_time_s": available if completed else "", "requirement_closure_time_s": closure_available if requirement_satisfied else "",
            "on_time_ground_completed": ontime_ground, "requirement_satisfied": requirement_satisfied, "on_time_task_success": ontime_success,
            "partial_ground_delivery": int(completed and not requirement_satisfied), "capture_to_ground_latency_s": available - float(metadata["capture_time_s"]) if completed else 0.0,
            "ground_model_quality": quality, "compression_quality_retention": 1.0,
            "ground_information_sufficiency": product_information_utility(products, metadata["product_requirement"], utility_cfg),
            "ground_result_quality": quality if completed else 0.0, "partial_ground_gross_utility": partial if completed and not requirement_satisfied else 0.0,
            "compute_energy_wh": float(getattr(record, "compute_energy_j", 0.0)) / 3600 if record else 0.0, "compression_energy_wh": 0.0,
            "communication_energy_wh": float(getattr(record, "comm_energy_j", 0.0)) / 3600 if record else 0.0, "payload_idle_energy_wh": 0.0,
            "housekeeping_energy_wh": float(getattr(record, "idle_energy_j", 0.0)) / 3600 if record else 0.0,
            "controllable_mission_energy_wh": (float(getattr(record, "compute_energy_j", 0.0)) + float(getattr(record, "comm_energy_j", 0.0))) / 3600 if record else 0.0,
            "total_spacecraft_energy_wh": (float(getattr(record, "compute_energy_j", 0.0)) + float(getattr(record, "comm_energy_j", 0.0)) + float(getattr(record, "idle_energy_j", 0.0))) / 3600 if record else 0.0,
            "expired": int(any(str(row.get("current_status")) == "expired" for row in lifecycle if row.get("task_id") == task_id)),
            "dropped": int(any(str(row.get("current_status")) == "dropped" for row in lifecycle if row.get("task_id") == task_id)),
            "terminal_backlog_flag": int(any(str(row.get("current_status")) in {"queued", "generated"} for row in lifecycle if row.get("task_id") == task_id)),
            "failure_reason": getattr(record, "service_failure_reason", "") if record else "missing_record",
        }
        rows.append(score_task(base, cfg))
    return rows
