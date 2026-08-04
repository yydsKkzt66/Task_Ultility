from copy import deepcopy
from pathlib import Path
try:
    from .native_support import IdentityBinding
except ImportError:
    from native_support import IdentityBinding


class OffloadBinding(IdentityBinding):
    def __init__(self, source_root: Path): super().__init__("OFFLOAD-EFT-THERMAL-ALIGNED", source_root)
    def _instantiate(self):
        from scripts.run_offload_vs_ground_24h_pilot import common_parameters
        from src.common.config import load_yaml
        from src.evaluation.canonical_sarfish_environment import load_canonical_sarfish_configs
        config = load_canonical_sarfish_configs(self.source_root, seed=2026)
        method = load_yaml(self.source_root / "configs/methods/offload_eft_thermal_aligned_v1.yaml")
        return common_parameters(config, method)["ground"]
    def simulate(self, simulation_input, slots: int):
        from scripts.run_three_method_minimal_robustness_v1 import run_offload_generic
        configs = deepcopy(simulation_input.configs)
        manifest = {row["task_id"]: {"task_id": row["task_id"], "arrival_time_s": float(row["arrival_time_s"]),
            "deadline_time_s": float(row["deadline_time_s"]), "value_level": row["value_level"],
            "effective_task_value": float(row["effective_task_value"]), "product_requirement": row["product_requirement"],
            "quality_requirement": float(row["quality_requirement"]), "information_requirement": float(row["information_requirement"])}
            for row in simulation_input.canonical_tasks}
        rows, scientific, trace = run_offload_generic(configs, simulation_input.timeline[:slots], manifest, 2026)
        normalized = _normalize_offload(self.method, simulation_input, rows, scientific, trace)
        return {"normalized": normalized, "event_count": len(normalized["events"])}


def make_binding(source_root: Path): return OffloadBinding(source_root)


def _normalize_offload(method, scenario, native_rows, scientific, trace):
    by = {str(row["task_id"]): row for row in native_rows}; tasks=[]; events=[]
    for row in scenario.canonical_tasks:
        item = by[row["task_id"]]; finish = item.get("requirement_closure_time_s", "")
        path = str(item.get("selected_path", "DEFER")); product = "full" if path == "GROUND" else ("roi" if path == "LOCAL" else "")
        if finish not in (None, ""):
            events.append({"method": method, "task_id": row["task_id"], "stage": "full" if path == "GROUND" else "roi",
                "ground_available_time": float(finish), "deadline": float(row["deadline_time_s"]),
                "valid_before_deadline": int(float(finish) <= float(row["deadline_time_s"])),
                "event_source": "offload_native:ground_fifo_compute_complete" if path == "GROUND" else "offload_native:local_result_ground_available",
                "duplicate_flag": 0})
        success = int(item.get("on_time_task_success", 0))
        tasks.append({"method": method, "task_id": row["task_id"], "arrival_time": row["arrival_time_s"],
            "deadline": row["deadline_time_s"], "value_level": row["value_level"],
            "effective_task_value": row["effective_task_value"], "product_requirement": row["product_requirement"],
            "selected_action": path, "executed_action": path, "selected_model": "ground_high_accuracy" if path == "GROUND" else ("local" if path == "LOCAL" else ""),
            "delivered_product": product if finish not in (None, "") else "", "closure_success": success,
            "closure_ground_time": finish, "failure_reason": "" if success else ("late_or_incomplete" if path != "DEFER" else "deferred"),
            "controllable_energy_j": float(item.get("compute_energy_j", 0.0)) + float(item.get("communication_energy_j", 0.0)),
            "transmitted_bytes": float(item.get("transmitted_mb", 0.0)) * 1_000_000.0})
    slot_rows=[]
    battery_min=float(scenario.configs["simulator"].get("battery_min",0)); thermal_max=float(scenario.configs["simulator"].get("thermal_max",1))
    for row in trace:
        soc=float(row.get("soc_after",0)); thermal=float(row.get("thermal_after",0)); slot=int(row.get("slot",len(slot_rows)))
        slot_rows.append({"method":method,"slot":slot,"time_s":float(row.get("time_s",slot*20)),"battery_soc":soc,
            "thermal":thermal,"queue_mb":float(row.get("queue_after_mb",row.get("queue_mb",0)) or 0),
            "compute_energy_j":float(row.get("compute_energy_j",0) or 0),"comm_energy_j":float(row.get("comm_energy_j",0) or 0),
            "idle_energy_j":float(scenario.timeline[slot].resource_trace.base_power_w)*float(scenario.timeline[slot].slot_seconds),
            "transmitted_mb":float(row.get("sent_mb",0) or 0),"in_contact":int(row.get("in_contact",0) or 0),
            "soc_violation":int(soc<battery_min),"thermal_violation":int(thermal>thermal_max)})
    resource={"method":method,"slot_count":len(slot_rows),"minimum_soc":min((r["battery_soc"] for r in slot_rows),default=None),
        "final_soc":slot_rows[-1]["battery_soc"] if slot_rows else None,"maximum_thermal":max((r["thermal"] for r in slot_rows),default=None),
        "soc_violation_count":sum(r["soc_violation"] for r in slot_rows),"thermal_violation_count":sum(r["thermal_violation"] for r in slot_rows),
        "controllable_energy_wh":float(scientific.get("controllable_energy_wh",0)),"transmitted_mb":float(scientific.get("transmitted_mb",0)),
        "ground_compute_energy_counted":False,"native_summary":scientific}
    return {"per_task":tasks,"events":events,"slots":slot_rows,"resource_summary":resource}
