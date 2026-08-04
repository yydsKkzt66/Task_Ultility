"""Pre-registered cross-capacity validation for the selected headroom controller."""
from __future__ import annotations
import csv,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_closure_safe_admission_guard_seed2026 import apply_variant_flags,run_prop_variant,run_b1_enhanced
from scripts.run_sarfish_canonical_seed2026 import prop_configs,write_csv
from scripts.run_sarfish_downlink_operating_region_seed2026 import available_capacity_mb,b,f,flatten_summary,group_summary,scale_downlink_capacity
from src.common.config import load_yaml
from src.evaluation.canonical_sarfish_environment import validate_main_paper_environment
from src.evaluation.hv_phase1 import is_hv_all,is_hv_target
from src.evaluation.task_outcome_adapter import manifest_rows
from src.timeline.timeline_builder import TimelineBuilder

OUT=ROOT/"outputs/_sarfish_thermal_safe_throttling_seed2026";GUARD=ROOT/"outputs/_sarfish_closure_safe_admission_guard_seed2026"
FLAGS={"enable_persistent_closing_requests":True,"enable_future_contact_generation":False,"enable_generation_side_guarantee":False,"enable_request_disable_replay":False,
 "enable_close_task_protection":False,"enable_displacement_net_guard":False,"enable_contact_capacity_reservation":False}
def read(p:Path)->list[dict[str,str]]:
    with p.open(encoding="utf-8-sig",newline="") as s:return list(csv.DictReader(s))
def cfg(base:dict[str,Any])->dict[str,Any]:
    c=apply_variant_flags(base,FLAGS);t=dict(c["simulator"].get("downlink_safety_throttle",{}));t.update({"enable_thermal_headroom_capacity":True,
      "enable_thermal_headroom_hysteresis":False,"thermal_headroom_search_iterations":40,"thermal_headroom_tolerance":1e-10});c["simulator"]["downlink_safety_throttle"]=t;return c
def rec(r:Any,k:str,d:float=0)->float:return f(getattr(r,k,d))
def main()->None:
    methods=read(OUT/"method_summary.csv");hv=read(OUT/"high_value_summary.csv");thermal=read(OUT/"thermal_safety_summary.csv");comm=read(OUT/"communication_summary.csv")
    refs=read(GUARD/"method_summary.csv");hrefs=read(GUARD/"high_value_summary.csv")
    accounting=load_yaml(ROOT/"configs/unified_ground_received_accounting.yaml");utility=load_yaml(ROOT/"configs/product_requirement_utility.yaml")
    for cap in (1.00,.35):
        base=scale_downlink_capacity(prop_configs(),cap);timeline=TimelineBuilder(base["dataset"],base["timeline"],base["task_stream"]).build();env=validate_main_paper_environment(base,timeline)
        manifest=manifest_rows(timeline,env["task_manifest_hash_v2"]);available=available_capacity_mb(timeline,float(base["simulator"].get("downlink_efficiency",1)))
        # Reuse validated pause/B1 references.
        for source,name in (("PROP-PERSIST","PROP-PAUSE"),("B1-Enhanced-1.4W","B1-CANONICAL-SHADOW")):
            row=next(r for r in refs if abs(f(r.get("capacity_multiplier"))-cap)<1e-9 and r.get("method")==source);methods.append(dict(row,method=name))
            hv += [dict(r,method=name) for r in hrefs if abs(f(r.get("capacity_multiplier"))-cap)<1e-9 and r.get("method")==source and r.get("task_group") in {"HV","HV-TARGET"}]
        c=cfg(base);pr=run_prop_variant("PROP-HEADROOM",cap,c,timeline,manifest,accounting,utility,OUT/"cross_capacity_raw")
        flat=flatten_summary("PROP-HEADROOM",cap,pr["rows"],pr["summary"],available,*pr["stage_sets"]);methods.append(flat)
        for group,pred in (("HV",is_hv_all),("HV-TARGET",is_hv_target)):hv.append(group_summary("PROP-HEADROOM",cap,pr["rows"],pr["summary"],*pr["stage_sets"],pred,group))
        ts=[rec(r,"thermal_next") for r in pr["records"]];trace=pr["runner"].last_thermal_safe_capacity_trace
        thermal.append({"capacity_multiplier":cap,"method":"PROP-HEADROOM","maximum_thermal_state":max(ts),"mean_thermal_state":sum(ts)/len(ts),"slots_above_0_85":sum(x>.85 for x in ts),
          "slots_at_or_above_0_92":sum(x>=.92 for x in ts),"hard_limit_violations":sum(x>.92 for x in ts),"zero_capacity_thermal_slots":sum(f(x.get("safe_capacity_bytes"))<=1e-9 and f(x.get("nominal_capacity_bytes"))>1e-9 for x in trace),
          "partially_throttled_slots":sum(1e-9<f(x.get("safe_capacity_bytes"))<f(x.get("nominal_capacity_bytes"))-1e-9 for x in trace)})
        br=run_b1_enhanced(cap,c,timeline,manifest,accounting,OUT/"cross_capacity_raw");closures={str(r["task_id"]) for r in br["rows"] if b(r.get("requirement_satisfied"))}
        flat=flatten_summary("B1-COMMON-HEADROOM",cap,br["rows"],br["summary"],available,closures,set(),set());flat["min_soc"]=f(br["raw"]["summary"].get("min_soc"));flat["final_soc"]=f(br["raw"]["summary"].get("final_soc"));methods.append(flat)
        for group,pred in (("HV",is_hv_all),("HV-TARGET",is_hv_target)):hv.append(group_summary("B1-COMMON-HEADROOM",cap,br["rows"],br["summary"],closures,set(),set(),pred,group))
        ts=[f(r.get("thermal_pressure")) for r in br["raw"]["soc_trace"]];trace=br["raw"].get("thermal_safe_capacity_trace",[])
        thermal.append({"capacity_multiplier":cap,"method":"B1-COMMON-HEADROOM","maximum_thermal_state":max(ts),"mean_thermal_state":sum(ts)/len(ts),"slots_above_0_85":sum(x>.85 for x in ts),
          "slots_at_or_above_0_92":sum(x>=.92 for x in ts),"hard_limit_violations":sum(x>.92 for x in ts),"zero_capacity_thermal_slots":sum(f(x.get("safe_capacity_bytes"))<=1e-9 and f(x.get("nominal_capacity_bytes"))>1e-9 for x in trace),
          "partially_throttled_slots":sum(1e-9<f(x.get("safe_capacity_bytes"))<f(x.get("nominal_capacity_bytes"))-1e-9 for x in trace)})
    write_csv(OUT/"method_summary.csv",methods);write_csv(OUT/"high_value_summary.csv",hv);write_csv(OUT/"thermal_safety_summary.csv",thermal);print(OUT)
if __name__=="__main__":main()
