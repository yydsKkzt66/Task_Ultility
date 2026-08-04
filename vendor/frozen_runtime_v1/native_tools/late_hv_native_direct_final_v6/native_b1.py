from copy import deepcopy
from pathlib import Path
try:
    from .native_support import IdentityBinding
except ImportError:
    from native_support import IdentityBinding


class B1Binding(IdentityBinding):
    def __init__(self, source_root: Path): super().__init__("B1-COMMON-THERMAL", source_root)
    def _instantiate(self):
        from scripts.run_sarfish_canonical_seed2026 import b1_config
        from src.baselines.b1_enhanced import B1EnhancedSimulator
        from src.common.config import load_project_configs
        configs = load_project_configs(self.source_root)
        return B1EnhancedSimulator(b1_config("enhanced"), configs["simulator"], configs["model_pool"])
    def simulate(self, simulation_input, slots: int):
        from scripts.run_sarfish_canonical_seed2026 import b1_config
        from src.baselines.b1_enhanced import B1EnhancedSimulator
        configs = deepcopy(simulation_input.configs)
        native = B1EnhancedSimulator(b1_config("enhanced"), configs["simulator"], configs["model_pool"]).run(
            simulation_input.timeline[:slots])
        normalized = _normalize_b1(self.method, simulation_input, native)
        return {"normalized": normalized, "event_count": len(normalized["events"])}


def make_binding(source_root: Path): return B1Binding(source_root)


def _normalize_b1(method, scenario, native):
    by = {str(row["task_id"]): row for row in native["per_task_delivery_trace"]}
    tasks, events = [], []
    for row in scenario.canonical_tasks:
        item = by.get(row["task_id"])
        available = item.get("ground_result_available_time_s", "") if item else ""
        completed = available not in (None, "", 0, 0.0) and str(item.get("state", "")) == "ground_result_available"
        if completed:
            events.append({"method": method, "task_id": row["task_id"], "stage": "full",
                "ground_available_time": float(available), "deadline": float(row["deadline_time_s"]),
                "valid_before_deadline": int(float(available) <= float(row["deadline_time_s"])),
                "event_source": "b1_native:complete_result_ground_available", "duplicate_flag": 0})
        ontime = bool(completed and float(available) <= float(row["deadline_time_s"]))
        tasks.append({"method": method, "task_id": row["task_id"], "arrival_time": row["arrival_time_s"],
            "deadline": row["deadline_time_s"], "value_level": row["value_level"],
            "effective_task_value": row["effective_task_value"], "product_requirement": row["product_requirement"],
            "selected_action": "B1_ENHANCED" if item else "", "executed_action": "B1_ENHANCED" if item else "",
            "selected_model": "ground_high_accuracy" if item else "", "delivered_product": "compressed_full_frame" if completed else "",
            "closure_success": int(ontime), "closure_ground_time": float(available) if completed else "",
            "failure_reason": "" if ontime else (str(item.get("blocked_reason") or item.get("state") or "not_arrived_in_phase") if item else "not_arrived_in_phase"),
            "controllable_energy_j": float(item.get("compression_energy_wh", 0.0)) * 3600.0 if item else 0.0,
            "transmitted_bytes": float(item.get("downlinked_mb", 0.0)) * 1_000_000.0 if item else 0.0})
    queues = {int(row["slot_index"]): row for row in native["queue_trace"]}
    slot_rows = []
    battery_min = float(scenario.configs["simulator"].get("battery_min", 0.0))
    thermal_max = float(scenario.configs["simulator"].get("thermal_max", 1.0))
    for row in native["soc_trace"]:
        slot = int(row["slot_index"]); queue = queues.get(slot, {})
        soc = float(row["battery_soc"]); thermal = float(row["thermal_pressure"])
        slot_rows.append({"method": method, "slot": slot, "time_s": float(row["time_s"]),
            "battery_soc": soc, "thermal": thermal, "queue_mb": float(queue.get("queue_mb", 0.0)),
            "compute_energy_j": float(row.get("compression_energy_wh", 0.0)) * 3600.0,
            "comm_energy_j": float(row.get("communication_energy_wh", 0.0)) * 3600.0,
            "idle_energy_j": float(row.get("base_power_w", 0.0)) * float(scenario.configs["timeline"]["slot_seconds"]),
            "transmitted_mb": float(queue.get("sent_mb", 0.0)), "in_contact": int(queue.get("in_contact", 0)),
            "soc_violation": int(soc < battery_min), "thermal_violation": int(thermal > thermal_max)})
    resource = {"method": method, "slot_count": len(slot_rows),
        "minimum_soc": min((r["battery_soc"] for r in slot_rows), default=None),
        "final_soc": slot_rows[-1]["battery_soc"] if slot_rows else None,
        "maximum_thermal": max((r["thermal"] for r in slot_rows), default=None),
        "soc_violation_count": sum(r["soc_violation"] for r in slot_rows),
        "thermal_violation_count": sum(r["thermal_violation"] for r in slot_rows),
        "controllable_energy_wh": sum(r["compute_energy_j"] + r["comm_energy_j"] for r in slot_rows) / 3600.0,
        "transmitted_mb": sum(r["transmitted_mb"] for r in slot_rows), "native_summary": native["summary"]}
    return {"per_task": tasks, "events": events, "slots": slot_rows, "resource_summary": resource}
