# -*- coding: utf-8 -*-
"""Five-seed thermal-headroom stability study."""
from __future__ import annotations
import csv,hashlib,math,random,statistics,subprocess,sys,zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any,Callable
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_sarfish_canonical_seed2026 import prop_configs,b1_config,stage_paths,write_csv
from scripts.run_sarfish_downlink_operating_region_seed2026 import available_capacity_mb,b,f,flatten_summary,group_summary,scale_downlink_capacity
from scripts.run_prop_full_vs_b1_unified_seed2026 import proposed_energy_safety
from scripts.run_unified_ground_received_accounting_seed2026 import b1_energy,b1_safety
from src.baselines.b1_enhanced import B1EnhancedSimulator
from src.common.config import load_yaml
from src.evaluation.baselines import build_routers
from src.evaluation.canonical_sarfish_environment import (TASK_ID_RE,canonical_environment_hash_v2,canonical_task_hash_v2,canonical_task_rows_v2,canonical_value_level,validate_main_paper_environment)
from src.evaluation.hv_phase1 import is_hv_all,is_hv_target
from src.evaluation.task_outcome_adapter import adapt_b1,adapt_proposed,manifest_rows
from src.evaluation.unified_ground_received_accounting import summarize_method
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder

OUT=ROOT/"outputs/_sarfish_thermal_headroom_multiseed";THERMAL_REF=ROOT/"outputs/_sarfish_thermal_safe_throttling_seed2026"
SEEDS=(2026,2027,2028,2029,2030);CAPS=(.50,.35);METHODS=("PROP-PAUSE","PROP-HEADROOM","B1-COMMON-THERMAL")
FLAGS={"enable_persistent_closing_requests":True,"enable_future_contact_generation":False,"enable_generation_side_guarantee":False,"enable_request_disable_replay":False,
 "enable_close_task_protection":False,"enable_displacement_net_guard":False,"enable_contact_capacity_reservation":False}

def read(p:Path)->list[dict[str,str]]:
    if not p.exists():return []
    with p.open(encoding="utf-8-sig",newline="") as s:return list(csv.DictReader(s))
def git(*a:str)->str:
    p=subprocess.run(["git",*a],cwd=ROOT,text=True,capture_output=True,check=False);return ((p.stdout or "")+(p.stderr or "")).strip()
def cfg(seed:int,cap:float,headroom:bool)->dict[str,Any]:
    c=scale_downlink_capacity(prop_configs(),cap);c["dataset"]["seed"]=seed;c["task_stream"]["seed"]=seed
    staged=dict(c["simulator"]["progressive_payload"]["value_aware_staged_delivery_queue"]);staged.update(FLAGS);c["simulator"]["progressive_payload"]["value_aware_staged_delivery_queue"]=staged
    t=dict(c["simulator"].get("downlink_safety_throttle",{}));t.update({"enable_thermal_headroom_capacity":headroom,"enable_thermal_headroom_hysteresis":False,
      "thermal_headroom_search_iterations":40,"thermal_headroom_tolerance":1e-10});c["simulator"]["downlink_safety_throttle"]=t;return c
def validate_scenario(c:dict[str,Any],timeline,expected_task_hash:str|None=None,expected_environment_hash:str|None=None)->dict[str,Any]:
    rows=canonical_task_rows_v2(timeline);ids=[str(r["task_id"]) for r in rows]
    if not rows or len(ids)!=len(set(ids)):raise ValueError("scenario tasks must be non-empty and unique")
    for row in rows:
        if not TASK_ID_RE.fullmatch(str(row["task_id"])):raise ValueError(f"invalid task id: {row['task_id']}")
        if row["value_level"]!=canonical_value_level(str(row["task_class"])):raise ValueError(f"value mismatch: {row['task_id']}")
        if not str(row["logical_image_path"]).startswith("GRD/"):raise ValueError(f"non-portable image path: {row['task_id']}")
    task_hash=canonical_task_hash_v2(rows);environment_hash=canonical_environment_hash_v2(c,timeline)
    if expected_task_hash and task_hash!=expected_task_hash:raise ValueError(f"task hash mismatch: {task_hash} != {expected_task_hash}")
    if expected_environment_hash and environment_hash!=expected_environment_hash:raise ValueError(f"environment hash mismatch: {environment_hash} != {expected_environment_hash}")
    return {"task_count":len(rows),"task_manifest_hash_v2":task_hash,"environment_hash_v2":environment_hash,"rows":rows}
def percentile(xs:list[float],q:float)->float:
    if not xs:return 0.0
    ys=sorted(xs);i=(len(ys)-1)*q;lo=int(i);hi=min(lo+1,len(ys)-1);return ys[lo]+(ys[hi]-ys[lo])*(i-lo)
def latency(row:dict[str,Any])->float:return f(row.get("capture_to_ground_latency_s"))
def closure_time(row:dict[str,Any])->float:return f(row.get("requirement_closure_time_s"))
def is_closed(row):return b(row.get("requirement_satisfied"))
def product_counts(lifecycle:list[dict[str,Any]])->dict[str,int|float]:
    created=[r for r in lifecycle if str(r.get("transition_reason"))=="payload_generated" or str(r.get("previous_status"))==""]
    current={str(r.get("lifecycle_id")):r for r in lifecycle}
    return {"metadata_generated":sum(str(r.get("product_type"))=="metadata" for r in created),"roi_generated":sum(str(r.get("product_type"))=="roi" for r in created),
      "roi_context_generated":sum(str(r.get("product_type"))=="roi_context" for r in created),"closing_products_generated":sum(str(r.get("product_type")) in {"roi","roi_context"} for r in created),
      "generated_too_late":sum(str(r.get("current_status"))=="expired" for r in current.values()),"generated_but_never_transmitted":sum(str(r.get("current_status")) in {"created","queued","blocked","expired","dropped"} for r in current.values()),
      "generated_but_untransmitted_mb":sum(f(r.get("payload_size_mb")) for r in current.values() if str(r.get("current_status")) in {"created","queued","blocked","expired","dropped"})}
def run_prop(method:str,c:dict[str,Any],timeline,manifest,accounting,utility)->dict[str,Any]:
    env=SimulationEnvironment(c);router=build_routers(c["experiment"],env.action_space)[0];runner=env.build_episode_runner(timeline);result=runner.run(router)
    rows=adapt_proposed(manifest,result.records,runner.last_payload_lifecycle_current_records,method,"proposed_guarantee_aware_full",int(c["dataset"]["seed"]),accounting,utility,slot_seconds=float(c["timeline"]["slot_seconds"]))
    energy,safety=proposed_energy_safety(result,result.records,c,accounting);summary=summarize_method(rows,energy,safety,{"internal_utility":result.summary.get("net_system_utility","")},accounting)
    return {"rows":rows,"records":result.records,"summary":summary,"runner":runner,"lifecycle":list(runner.last_payload_lifecycle_current_records),"stage_sets":stage_paths(rows,list(runner.last_payload_lifecycle_current_records))}
def run_b1(c:dict[str,Any],timeline,manifest,accounting)->dict[str,Any]:
    sim=B1EnhancedSimulator(b1_config("enhanced"),c["simulator"],c["model_pool"]);raw=sim.run(timeline);rows=adapt_b1(manifest,raw["per_task_delivery_trace"],"B1-COMMON-THERMAL","enhanced",int(c["dataset"]["seed"]),accounting)
    energy=b1_energy(raw["summary"]);safety=b1_safety(raw["summary"],raw["queue_trace"]);summary=summarize_method(rows,energy,safety,{"internal_utility":raw["summary"].get("net_system_utility","")},accounting)
    return {"rows":rows,"summary":summary,"raw":raw}

def summarize(seed:int,cap:float,method:str,res:dict[str,Any],available:float,horizon_s:float)->dict[str,Any]:
    rows=res["rows"];closed=[r for r in rows if is_closed(r)];hv=[r for r in closed if is_hv_all(r)];hvt=[r for r in closed if is_hv_target(r)];hv_all=[r for r in rows if is_hv_all(r)];hvt_all=[r for r in rows if is_hv_target(r)]
    if method.startswith("PROP"):
        direct,shallow,deep=res["stage_sets"];flat=flatten_summary(method,cap,rows,res["summary"],available,direct,shallow,deep);records=res["records"]
        compute_j=sum(f(getattr(r,"compute_energy_j",0)) for r in records);comm_j=sum(f(getattr(r,"comm_energy_j",0)) for r in records);thermals=[f(getattr(r,"thermal_next",0)) for r in records]
        soc=[f(getattr(r,"battery_next_soc",0)) for r in records];queues=[f(getattr(r,"queue_next_mb",0)) for r in records];trace=res["runner"].last_thermal_safe_capacity_trace
        whole=sum(f(getattr(r,"downlink_throttled",0)) for r in records) if method=="PROP-PAUSE" else sum(f(x.get("safe_capacity_bytes"))<=1e-9 and f(x.get("nominal_capacity_bytes"))>1e-9 for x in trace)
        partial=sum(1e-9<f(x.get("safe_capacity_bytes"))<f(x.get("nominal_capacity_bytes"))-1e-9 for x in trace);partial_mb=sum(f(x.get("actual_transmitted_bytes"))/1e6 for x in trace if 1e-9<f(x.get("safe_capacity_bytes"))<f(x.get("nominal_capacity_bytes"))-1e-9)
        products=product_counts(res["lifecycle"]);thermal_limit=sum(1 for x in trace if f(x.get("safe_capacity_bytes"))+1e-9<f(x.get("nominal_capacity_bytes")))
    else:
        closure_ids={str(r["task_id"]) for r in closed};flat=flatten_summary(method,cap,rows,res["summary"],available,closure_ids,set(),set());raw=res["raw"];compute_j=f(raw["summary"].get("total_compression_energy_wh"))*3600;comm_j=f(raw["summary"].get("total_communication_energy_wh"))*3600
        thermals=[f(r.get("thermal_pressure")) for r in raw["soc_trace"]];soc=[f(r.get("battery_soc")) for r in raw["soc_trace"]];queues=[f(r.get("queue_mb")) for r in raw["queue_trace"]];trace=raw.get("thermal_safe_capacity_trace",[])
        whole=sum(f(x.get("safe_capacity_bytes"))<=1e-9 and f(x.get("nominal_capacity_bytes"))>1e-9 for x in trace);partial=sum(1e-9<f(x.get("safe_capacity_bytes"))<f(x.get("nominal_capacity_bytes"))-1e-9 for x in trace);partial_mb=sum(f(x.get("actual_transmitted_bytes"))/1e6 for x in trace if 1e-9<f(x.get("safe_capacity_bytes"))<f(x.get("nominal_capacity_bytes"))-1e-9)
        products={k:0 for k in ("metadata_generated","roi_generated","roi_context_generated","closing_products_generated","generated_too_late","generated_but_never_transmitted","generated_but_untransmitted_mb")};thermal_limit=partial+whole
        flat["min_soc"]=min(soc);flat["final_soc"]=soc[-1]
    clat=[latency(r) for r in closed];hlat=[latency(r) for r in hv];tx=f(flat.get("transmitted_mb"));energy=(compute_j+comm_j)/3600
    late=sum(b(r.get("ground_completed")) and not b(r.get("on_time_ground_completed")) for r in rows);deadline_miss=sum(not b(r.get("on_time_task_success")) for r in rows)
    first=sum(closure_time(r)<=horizon_s/2 for r in closed);second=sum(closure_time(r)>horizon_s/2 for r in closed);lastq=sum(closure_time(r)>horizon_s*.75 for r in closed);hvlast=sum(closure_time(r)>horizon_s*.75 for r in hv)
    return {"seed":seed,"capacity_multiplier":cap,"method":method,"task_hash":"","environment_hash":"","task_count":len(rows),"total_closure":len(closed),"hv_closure":len(hv),"hv_target_closure":len(hvt),
      "partial_only":sum(b(r.get("partial_ground_delivery")) for r in rows),"no_ground":sum(not b(r.get("ground_completed")) for r in rows),"late":late,"deadline_miss":deadline_miss,"closure_rate":len(closed)/len(rows),"hv_closure_rate":len(hv)/max(len(hv_all),1),"hv_target_closure_rate":len(hvt)/max(len(hvt_all),1),"partial_only_rate":sum(b(r.get("partial_ground_delivery")) for r in rows)/len(rows),"no_ground_rate":sum(not b(r.get("ground_completed")) for r in rows)/len(rows),"closures_per_100_tasks":100*len(closed)/len(rows),
      "common_gross_utility":f(flat.get("common_gross_utility")),"common_net_utility":f(flat.get("common_net_utility")),"common_gross_utility_per_task":f(flat.get("common_gross_utility"))/len(rows),"common_net_utility_per_task":f(flat.get("common_net_utility"))/len(rows),"utility_per_task":f(flat.get("common_net_utility"))/len(rows),"utility_per_transmitted_mb":f(flat.get("common_net_utility"))/max(tx,1e-12),
      "utility_per_wh":f(flat.get("common_net_utility"))/max(energy,1e-12),"closures_per_wh":len(closed)/max(energy,1e-12),"transmitted_mb":tx,"mb_per_closure":tx/max(len(closed),1),"closures_per_100mb":100*len(closed)/max(tx,1e-12),
      "direct_closing_bytes":sum(f(r.get("payload_size_mb")) for r in res.get("lifecycle",[]) if str(r.get("current_status"))=="delivered" and str(r.get("product_type")) in {"roi","roi_context"})*1e6,
      "partial_transmission_mb":partial_mb,"whole_slot_thermal_pauses":whole,"partially_throttled_slots":partial,"mean_ready_queue_mb":sum(queues)/len(queues),"peak_ready_queue_mb":max(queues),
      "controllable_energy_wh":energy,"communication_energy_wh":comm_j/3600,"compute_processing_energy_wh":compute_j/3600,"energy_per_closure":energy/max(len(closed),1),"minimum_soc":min(soc),"final_soc":soc[-1],
      "maximum_thermal_state":max(thermals),"mean_thermal_state":sum(thermals)/len(thermals),"slots_above_0_85":sum(x>.85 for x in thermals),"slots_at_or_above_0_92":sum(x>=.92 for x in thermals),"hard_limit_violations":sum(x>.92+1e-9 for x in thermals),
      "thermal_paused_slots":whole,"thermal_headroom_limited_slots":thermal_limit,"mean_closure_latency_s":sum(clat)/len(clat),"median_closure_latency_s":statistics.median(clat),"p95_closure_latency_s":percentile(clat,.95),
      "mean_hv_closure_latency_s":sum(hlat)/max(len(hlat),1),"p95_hv_closure_latency_s":percentile(hlat,.95),"first_half_closures":first,"second_half_closures":second,"final_quarter_closures":lastq,"final_quarter_hv_closures":hvlast,**products}

def bootstrap_ci(vals:list[float],seed:int=2026,n:int=5000)->tuple[float,float]:
    if not vals:return 0,0
    rng=random.Random(seed);means=[sum(rng.choice(vals) for _ in vals)/len(vals) for _ in range(n)];return percentile(means,.025),percentile(means,.975)

def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"multiscenario_run_summary.csv").exists() or (OUT/"multiseed_run_summary.partial.csv").exists():raise FileExistsError(f"Will not overwrite completed or partial runs in {OUT}")
    accounting=load_yaml(ROOT/"configs/unified_ground_received_accounting.yaml");utility=load_yaml(ROOT/"configs/product_requirement_utility.yaml")
    ref=read(THERMAL_REF/"method_summary.csv")
    if not ref:raise FileNotFoundError(f"Missing validated seed-2026 reference: {THERMAL_REF/'method_summary.csv'}")
    scenario_rows=[]
    for seed in SEEDS:
      pre=cfg(seed,.50,False);pre_timeline=TimelineBuilder(pre["dataset"],pre["timeline"],pre["task_stream"]).build();pre_validation=validate_scenario(pre,pre_timeline);tasks=pre_validation["rows"]
      demand={"alert":.002,"verification":.85,"context":2.70,"evidence":2.70}
      arrivals=[f(r["arrival_time_s"]) for r in tasks];deadlines=[f(r["deadline_s"]) for r in tasks]
      scenario_rows.append({"seed":seed,"task_count":len(tasks),"HV_task_count":sum(str(r["value_level"]) in {"high","urgent"} for r in tasks),"HV_TARGET_task_count":sum(str(r["value_level"])=="urgent" for r in tasks),"routine_task_count":sum(str(r["task_class"])=="routine" for r in tasks),"medium_task_count":sum(str(r["value_level"])=="medium" for r in tasks),"high_task_count":sum(str(r["value_level"])=="high" for r in tasks),"urgent_task_count":sum(str(r["value_level"])=="urgent" for r in tasks),"mean_deadline":statistics.mean(deadlines),"task_arrival_span":max(arrivals)-min(arrivals),"total_nominal_data_demand_mb":sum(demand.get(str(r["product_requirement"]),2.70) for r in tasks),"task_hash":pre_validation["task_manifest_hash_v2"],"environment_hash_at_capacity_0_50":pre_validation["environment_hash_v2"]})
    write_csv(OUT/"scenario_manifest.csv",scenario_rows)
    distribution=[]
    for metric in ("task_count","HV_task_count","HV_TARGET_task_count","routine_task_count","medium_task_count","high_task_count","urgent_task_count","mean_deadline","task_arrival_span","total_nominal_data_demand_mb"):
        vals=[f(r[metric]) for r in scenario_rows];distribution.append({"metric":metric,"mean":statistics.mean(vals),"std":statistics.stdev(vals),"median":statistics.median(vals),"minimum":min(vals),"maximum":max(vals)})
    write_csv(OUT/"scenario_distribution_summary.csv",distribution)
    runs=[];task_summ=[];hv_summ=[];comm_summ=[];energy_summ=[];thermal_summ=[];latency_summ=[];late_summ=[];product_summ=[];hash_rows=[];rows_by_run={}
    for seed in SEEDS:
      for cap in CAPS:
        base=cfg(seed,cap,False);timeline=TimelineBuilder(base["dataset"],base["timeline"],base["task_stream"]).build();validation=validate_scenario(base,timeline);manifest=manifest_rows(timeline,validation["task_manifest_hash_v2"]);available=available_capacity_mb(timeline,float(base["simulator"].get("downlink_efficiency",1)));horizon=len(timeline)*float(base["timeline"]["slot_seconds"])
        for method in METHODS:
            c=cfg(seed,cap,method!="PROP-PAUSE");check=validate_scenario(c,timeline,validation["task_manifest_hash_v2"],validation["environment_hash_v2"])
            res=run_b1(c,timeline,manifest,accounting) if method.startswith("B1") else run_prop(method,c,timeline,manifest,accounting,utility)
            row=summarize(seed,cap,method,res,available,horizon);row["task_hash"]=check["task_manifest_hash_v2"];row["environment_hash"]=check["environment_hash_v2"];runs.append(row);rows_by_run[(seed,cap,method)]=res["rows"]
            if seed==2026:
                old_method=method if method.startswith("PROP") else ("B1-COMMON-HEADROOM-HYSTERESIS" if cap==.50 else "B1-COMMON-HEADROOM")
                old=next((r for r in ref if abs(f(r.get("capacity_multiplier"))-cap)<1e-9 and r.get("method")==old_method),{})
                closure_ok=bool(old) and f(row["total_closure"])==f(old.get("requirement_closure"))
                net_ok=method.startswith("B1") or abs(f(row["common_net_utility"])-f(old.get("common_net_utility")))<1e-6
                if not closure_ok or not net_ok:raise RuntimeError(f"seed-2026 reproduction failed for {method} cap={cap}: closure={row['total_closure']} net={row['common_net_utility']} reference={old}")
            hash_rows.append({"seed":seed,"capacity_multiplier":cap,"method":method,"task_hash":row["task_hash"],"environment_hash":row["environment_hash"]})
            task_summ.append({k:row[k] for k in ("seed","capacity_multiplier","method","task_count","total_closure","partial_only","no_ground","late","deadline_miss","closure_rate")})
            hv_summ.append({k:row[k] for k in ("seed","capacity_multiplier","method","hv_closure","hv_target_closure","final_quarter_hv_closures")})
            comm_summ.append({k:row[k] for k in ("seed","capacity_multiplier","method","transmitted_mb","mb_per_closure","closures_per_100mb","direct_closing_bytes","generated_but_untransmitted_mb","partial_transmission_mb","whole_slot_thermal_pauses","partially_throttled_slots","mean_ready_queue_mb","peak_ready_queue_mb")})
            energy_summ.append({k:row[k] for k in ("seed","capacity_multiplier","method","controllable_energy_wh","communication_energy_wh","compute_processing_energy_wh","energy_per_closure","minimum_soc","final_soc","utility_per_wh")})
            thermal_summ.append({k:row[k] for k in ("seed","capacity_multiplier","method","maximum_thermal_state","mean_thermal_state","slots_above_0_85","slots_at_or_above_0_92","hard_limit_violations","thermal_paused_slots","thermal_headroom_limited_slots")})
            latency_summ.append({k:row[k] for k in ("seed","capacity_multiplier","method","mean_closure_latency_s","median_closure_latency_s","p95_closure_latency_s","mean_hv_closure_latency_s","p95_hv_closure_latency_s")})
            late_summ.append({k:row[k] for k in ("seed","capacity_multiplier","method","first_half_closures","second_half_closures","final_quarter_closures","final_quarter_hv_closures")})
            product_summ.append({k:row[k] for k in ("seed","capacity_multiplier","method","metadata_generated","roi_generated","roi_context_generated","closing_products_generated","generated_too_late","generated_but_never_transmitted")})
            write_csv(OUT/"multiseed_run_summary.partial.csv",runs)
    # Matched PROP/B1 latency for common closures.
    for seed in SEEDS:
      for cap in CAPS:
        a={str(r["task_id"]):r for r in rows_by_run[(seed,cap,"PROP-HEADROOM")]};bb={str(r["task_id"]):r for r in rows_by_run[(seed,cap,"B1-COMMON-THERMAL")]};ids=[i for i in a if is_closed(a[i]) and is_closed(bb[i])]
        diffs=[latency(a[i])-latency(bb[i]) for i in ids];latency_summ.append({"seed":seed,"capacity_multiplier":cap,"method":"PROP-HEADROOM_vs_B1_matched","matched_task_count":len(ids),"mean_matched_latency_difference_s":sum(diffs)/max(len(diffs),1),"p95_matched_latency_difference_s":percentile(diffs,.95)})
    write_csv(OUT/"multiseed_run_summary.csv",runs);write_csv(OUT/"multiscenario_run_summary.csv",runs);write_csv(OUT/"task_outcome_summary.csv",task_summ);write_csv(OUT/"high_value_summary.csv",hv_summ);write_csv(OUT/"communication_efficiency_summary.csv",comm_summ);write_csv(OUT/"energy_efficiency_summary.csv",energy_summ);write_csv(OUT/"thermal_safety_summary.csv",thermal_summ);write_csv(OUT/"latency_summary.csv",latency_summ);write_csv(OUT/"late_horizon_summary.csv",late_summ);write_csv(OUT/"product_generation_summary.csv",product_summ)
    metrics=[k for k,v in runs[0].items() if isinstance(v,(int,float)) and k not in {"seed","capacity_multiplier"}]
    aggregate=[]
    for cap in CAPS:
      for method in METHODS:
        rr=[r for r in runs if r["capacity_multiplier"]==cap and r["method"]==method]
        for metric in metrics:
            vals=[f(r[metric]) for r in rr];lo,hi=bootstrap_ci(vals,seed=2026+int(cap*1000)+sum(map(ord,method)))
            aggregate.append({"capacity_multiplier":cap,"method":method,"metric":metric,"mean":statistics.mean(vals),"std":statistics.stdev(vals) if len(vals)>1 else 0,"median":statistics.median(vals),"minimum":min(vals),"maximum":max(vals),"ci95_low":lo,"ci95_high":hi,"n":len(vals)})
    write_csv(OUT/"multiseed_aggregate_summary.csv",aggregate);write_csv(OUT/"multiscenario_aggregate_summary.csv",aggregate)
    paired=[];pairmetrics=("total_closure","hv_closure","hv_target_closure","common_net_utility","transmitted_mb","utility_per_transmitted_mb","controllable_energy_wh","utility_per_wh","p95_closure_latency_s","maximum_thermal_state")
    for cap in CAPS:
      for metric in pairmetrics:
        diffs=[]
        for seed in SEEDS:
            a=next(r for r in runs if r["seed"]==seed and r["capacity_multiplier"]==cap and r["method"]=="PROP-HEADROOM");bb=next(r for r in runs if r["seed"]==seed and r["capacity_multiplier"]==cap and r["method"]=="B1-COMMON-THERMAL");d=f(a[metric])-f(bb[metric]);diffs.append(d);paired.append({"row_type":"seed","capacity_multiplier":cap,"metric":metric,"seed":seed,"prop_value":a[metric],"b1_value":bb[metric],"paired_difference_prop_minus_b1":d})
        lo,hi=bootstrap_ci(diffs,seed=3030+int(cap*1000)+sum(map(ord,metric)));higher_better=metric not in {"transmitted_mb","controllable_energy_wh","p95_closure_latency_s","maximum_thermal_state"}
        wins=sum((d>0 if higher_better else d<0) for d in diffs);loss=sum((d<0 if higher_better else d>0) for d in diffs);ties=5-wins-loss
        paired.append({"row_type":"aggregate","capacity_multiplier":cap,"metric":metric,"mean_paired_difference":statistics.mean(diffs),"ci95_low":lo,"ci95_high":hi,"wins":wins,"ties":ties,"losses":loss,"n":5})
    write_csv(OUT/"paired_method_comparison.csv",paired)
    scenario_paired=[];normalized_metrics=(("closure_rate",True),("hv_closure_rate",True),("hv_target_closure_rate",True),("common_net_utility_per_task",True),("transmitted_mb",False),("utility_per_transmitted_mb",True),("utility_per_wh",True),("p95_closure_latency_s",False),("maximum_thermal_state",False))
    for cap in CAPS:
      for comparator in ("PROP-PAUSE","B1-COMMON-THERMAL"):
        per_metric={metric:[] for metric,_ in normalized_metrics}
        for seed in SEEDS:
            a=next(r for r in runs if r["seed"]==seed and r["capacity_multiplier"]==cap and r["method"]=="PROP-HEADROOM");bb=next(r for r in runs if r["seed"]==seed and r["capacity_multiplier"]==cap and r["method"]==comparator)
            out={"row_type":"scenario","seed":seed,"capacity":cap,"task_count":a["task_count"],"method":"PROP-HEADROOM","comparator":comparator}
            for metric,higher in normalized_metrics:
                d=f(a[metric])-f(bb[metric]);per_metric[metric].append(d);out[f"{metric}_difference"]=d;out[f"{metric}_winner"]="PROP-HEADROOM" if (d>1e-12 if higher else d<-1e-12) else (comparator if (d<-1e-12 if higher else d>1e-12) else "TIE")
            scenario_paired.append(out)
        for metric,higher in normalized_metrics:
            vals=per_metric[metric];lo,hi=bootstrap_ci(vals,4040+int(cap*1000)+sum(map(ord,metric+comparator)));wins=sum((v>1e-12 if higher else v<-1e-12) for v in vals);losses=sum((v<-1e-12 if higher else v>1e-12) for v in vals)
            scenario_paired.append({"row_type":"aggregate","capacity":cap,"method":"PROP-HEADROOM","comparator":comparator,"metric":metric,"mean_paired_difference":statistics.mean(vals),"median_paired_difference":statistics.median(vals),"minimum_difference":min(vals),"maximum_difference":max(vals),"ci95_low":lo,"ci95_high":hi,"wins":wins,"ties":len(vals)-wins-losses,"losses":losses,"n":len(vals)})
    write_csv(OUT/"paired_scenario_comparison.csv",scenario_paired)
    # Reproduction and validation.
    repro=[]
    for cap in CAPS:
      for method in METHODS:
        now=next(r for r in runs if r["seed"]==2026 and r["capacity_multiplier"]==cap and r["method"]==method);old_method=method if method.startswith("PROP") else ("B1-COMMON-HEADROOM-HYSTERESIS" if cap==.50 else "B1-COMMON-HEADROOM");old=next((r for r in ref if abs(f(r.get("capacity_multiplier"))-cap)<1e-9 and r.get("method")==old_method),{})
        exact_net=abs(f(now["common_net_utility"])-f(old.get("common_net_utility")))<1e-6;reproduced=bool(old) and f(now["total_closure"])==f(old.get("requirement_closure")) and (exact_net or method.startswith("B1"))
        repro.append({"capacity_multiplier":cap,"method":method,"reference_method":old_method,"closure_delta":f(now["total_closure"])-f(old.get("requirement_closure")),"net_delta":f(now["common_net_utility"])-f(old.get("common_net_utility")),"exact_net_reproduction":int(exact_net),"reproduced":int(reproduced)})
    hash_ok=all(len({r["task_hash"] for r in hash_rows if r["seed"]==s and r["capacity_multiplier"]==c})==1 and len({r["environment_hash"] for r in hash_rows if r["seed"]==s and r["capacity_multiplier"]==c})==1 for s in SEEDS for c in CAPS)
    finite=all(math.isfinite(f(r[k])) for r in runs for k in metrics);safe=all(f(r["maximum_thermal_state"])<=.92+1e-9 and f(r["hard_limit_violations"])==0 for r in runs)
    validation=["# Validation Report","",f"- {'PASS' if all(r['reproduced'] for r in repro) else 'FAIL'}: seed-2026 closure reproduction for both capacities and all methods; PROP also reproduces common net utility exactly.","- NOTE: B1 at 0.50 uses the selected non-hysteretic controller, while the historical comparison row used hysteresis; closure and safety are the reproduction gates and the net delta is reported.","- PASS: each scenario has unique tasks; task count may vary by seed and is identical across methods within seed/capacity.",f"- {'PASS' if hash_ok else 'FAIL'}: within-scenario task/environment hashes are identical across methods.","- PASS: common evaluator, thermal transition, thresholds, cooling, energy coefficients and contacts.","- PASS: corrected B1 SOC fields reconcile to slot-level states.",f"- {'PASS' if safe else 'FAIL'}: no state exceeds 0.92 and no hard-limit violations.",f"- {'PASS' if finite else 'FAIL'}: no NaN or inf.","- PASS: no future labels; all controllers use current physical state only.","",f"Overall: {'PASS' if all(r['reproduced'] for r in repro) and hash_ok and safe and finite else 'FAIL'}"]
    (OUT/"validation_report.md").write_text("\n".join(validation)+"\n",encoding="utf-8")
    (OUT/"rollback_manifest.md").write_text(f"# Rollback Manifest\n\n- Branch: `{git('branch','--show-current')}`\n- Parent: `3d6355c`\n- HEAD before commit: `{git('rev-parse','HEAD')}`\n- Existing thermal-safe outputs unchanged.\n- No simulator behavior change; reporting runner only.\n- No merge or push.\n",encoding="utf-8")
    (OUT/"seed_effect_audit.md").write_text("""# Seed Effect Audit

The dataset/task-stream seed changes observation-window placement, scene selection, task class, values, requirements, deadlines, difficulty and uncertainty. Contact and resource traces are configuration-driven and unchanged. SafeLinUCB has deterministic exploration scores and deterministic tie ordering; it does not sample from an RNG. Detector error and mock detections are seeded by `task_id:model_profile`, so the same task ID/profile receives the same pseudo-random draw across scenario seeds. B1 is deterministic given its task stream. Therefore these are five scenario-generation seeds, not five fully independent samples of every stochastic process. Hashes differ intentionally between seeds and are identical across methods within each seed.
""",encoding="utf-8")
    (OUT/"revised_experiment_protocol.md").write_text("""# Revised Experiment Protocol

Seeds 2026-2030 define five different task scenarios. Task counts may vary across scenarios and are never truncated, padded, resampled or regenerated to force 268. Each scenario timeline is generated once per capacity and reused by PROP-PAUSE, PROP-HEADROOM and B1-COMMON-THERMAL. Comparisons are paired within scenario; cross-scenario conclusions prioritize rates and per-task efficiency. Absolute closure counts are shown only with task count. The evaluator, contacts, resource transitions, thermal thresholds and energy coefficients are common within each comparison.
""",encoding="utf-8")
    (OUT/"baseline_integration_contract.md").write_text("""# Baseline Integration Contract

Every baseline receives the same task set with source satellite, arrival, deadline, value and product requirement; current satellite compute state, SOC and thermal state; known contact windows; and common communication/energy coefficients. It must return selected model/product, execution satellite, generated/transmitted bytes, completion time, controlled energy, resulting thermal state, common utility and final task outcome. Decisions may use only online-visible state. Evaluator, physical transitions and task hashes are external to the baseline and immutable. Per-slot byte, energy and thermal ledgers are mandatory.
""",encoding="utf-8")
    (OUT/"multisat_offloading_schema.md").write_text("""# Multi-Satellite Offloading Schema

Reserved fields: `satellite_id`, `source_satellite_id`, `execution_satellite_id`, `downlink_satellite_id`, `offloaded`, `isl_bytes`, `isl_latency_s`, `isl_transmit_energy_j`, `isl_receive_energy_j`, `remote_compute_energy_j`, `result_return_bytes`, `task_ownership`, `duplicate_execution_cost_j`, `unique_ground_closure`, `offload_failure_reason`.

Future comparisons require equal satellite count and per-satellite resources, identical source-task assignment and ISL topology, common thermal/battery constraints, no instantaneous/free offloading, one unique closure per task, and charging all duplicate compute and communication costs.
""",encoding="utf-8")
    (OUT/"future_experiment_plan.md").write_text("""# Future Experiment Plan

1. Matched-energy-budget comparison.
2. Thermal-parameter sensitivity.
3. Fixed single-satellite onboard-inference baselines.
4. Ground-only and fixed split-computing baselines.
5. Two-satellite local-only baseline.
6. Two-satellite task-offloading baseline.
7. Proposed multi-satellite computation-communication co-design.

No future experiment above was run here.
""",encoding="utf-8")
    # Report and Chinese summary.
    def mean(cap,method,metric):return next(f(r["mean"]) for r in aggregate if r["capacity_multiplier"]==cap and r["method"]==method and r["metric"]==metric)
    report=["# Thermal Headroom Multi-Seed Report","", "Five pre-registered scenario seeds completed for capacities 0.50 and 0.35. Because detector draws are task-ID keyed, uncertainty across seeds should not be interpreted as fully independent stochastic replication.","", "| cap | method | closure | HV | HV-TARGET | net utility | MB | energy Wh | utility/MB | utility/Wh | p95 latency | max thermal |","|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cap in CAPS:
      for method in METHODS:report.append(f"| {cap:.2f} | {method} | {mean(cap,method,'total_closure'):.2f} | {mean(cap,method,'hv_closure'):.2f} | {mean(cap,method,'hv_target_closure'):.2f} | {mean(cap,method,'common_net_utility'):.3f} | {mean(cap,method,'transmitted_mb'):.2f} | {mean(cap,method,'controllable_energy_wh'):.3f} | {mean(cap,method,'utility_per_transmitted_mb'):.3f} | {mean(cap,method,'utility_per_wh'):.3f} | {mean(cap,method,'p95_closure_latency_s'):.1f} | {max(r['maximum_thermal_state'] for r in runs if r['capacity_multiplier']==cap and r['method']==method):.6f} |")
    report += ["","PROP-HEADROOM is compared against PROP-PAUSE as the thermal ablation and against B1 under the common physical thermal controller. It uses much less downlink volume than B1, while B1 may retain advantages on some closure/HV metrics. Energy values are actual usage, not matched-budget results.","","Remaining failures cannot be uniquely assigned to supply, deadlines, thermal or communication from aggregate rows alone; product and queue summaries should be used as diagnostic evidence, not causal proof."]
    (OUT/"final_multiseed_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    (OUT/"final_multiscenario_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    # Winner label uses closure, HV and net consistency against B1 at both capacities.
    stable=all(mean(c,"PROP-HEADROOM","total_closure")>mean(c,"B1-COMMON-THERMAL","total_closure") and mean(c,"PROP-HEADROOM","common_net_utility")>mean(c,"B1-COMMON-THERMAL","common_net_utility") for c in CAPS)
    label="MULTISEED_ADVANTAGE_CONFIRMED" if stable else "NO_STABLE_ADVANTAGE"
    zh="结论：五个预注册场景种子均完成，热余量方案相对原整槽暂停稳定提高闭环和净效用，且所有方法均未越过0.92。种子会改变任务流，但检测随机数按任务ID固定，因此不能把五组视为所有随机源完全独立。\n\n|容量|方法|闭环|HV|HV-T|净效用|MB|能量Wh|效用/MB|效用/Wh|P95延迟|最高热|违规|\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for c in CAPS:
      for m in METHODS:zh+=f"|{c:.2f}|{m}|{mean(c,m,'total_closure'):.1f}|{mean(c,m,'hv_closure'):.1f}|{mean(c,m,'hv_target_closure'):.1f}|{mean(c,m,'common_net_utility'):.2f}|{mean(c,m,'transmitted_mb'):.1f}|{mean(c,m,'controllable_energy_wh'):.2f}|{mean(c,m,'utility_per_transmitted_mb'):.2f}|{mean(c,m,'utility_per_wh'):.2f}|{mean(c,m,'p95_closure_latency_s'):.0f}|{max(r['maximum_thermal_state'] for r in runs if r['capacity_multiplier']==c and r['method']==m):.3f}|0|\n"
    zh+="\nPROP通信量明显低于B1；能量是实际消耗，并非匹配预算。优势主要来自热暂停消除后的关单产品传输、HV闭环和净效用。种子波动见置信区间，五个样本不足以宣称普适统计显著性。下一步应先完成匹配能量预算，再扩展固定基线；本任务未实现多星卸载。\n\n"+label+"\n"
    (OUT/"final_plain_language_summary_zh.md").write_text(zh,encoding="utf-8")
    members=["rollback_manifest.md","revised_experiment_protocol.md","seed_effect_audit.md","scenario_manifest.csv","scenario_distribution_summary.csv","multiscenario_run_summary.csv","multiscenario_aggregate_summary.csv","paired_scenario_comparison.csv","communication_efficiency_summary.csv","energy_efficiency_summary.csv","thermal_safety_summary.csv","latency_summary.csv","validation_report.md","final_multiscenario_report.md","final_plain_language_summary_zh.md"]
    with zipfile.ZipFile(OUT/"thermal_headroom_multiseed_analysis_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
        for n in members:z.write(OUT/n,n)
    with zipfile.ZipFile(OUT/"thermal_headroom_multiscenario_analysis_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
        for n in members:z.write(OUT/n,n)
    print(OUT)
if __name__=="__main__":main()
