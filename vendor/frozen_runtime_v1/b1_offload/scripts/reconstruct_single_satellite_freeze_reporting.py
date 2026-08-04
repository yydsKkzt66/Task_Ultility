"""Reporting-only reconstruction for the frozen five-scenario comparison."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_thermal_headroom_multiseed import CAPS,METHODS,SEEDS,available_capacity_mb,cfg,f,manifest_rows,run_b1,run_prop,summarize,validate_scenario
from src.common.config import load_yaml
from src.timeline.timeline_builder import TimelineBuilder

OUT=ROOT/"outputs/_sarfish_single_satellite_baseline_freeze_v1"
REFERENCE=ROOT/"outputs/_sarfish_thermal_headroom_multiseed/multiscenario_run_summary.csv"

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(path,rows):
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def product_summary(seed,cap,method,res):
    if method.startswith("B1"):
        trace=res["raw"]["per_task_delivery_trace"]
        return {"scenario_seed":seed,"capacity":cap,"method":method,"metadata_generated":0,"roi_generated":0,"roi_context_generated":0,"minimum_closing_products_generated":0,"generated_before_deadline":sum(f(r.get("compression_latency_s"))<=f(r.get("deadline_s")) for r in trace),"generated_too_late":sum(f(r.get("compression_latency_s"))>f(r.get("deadline_s")) for r in trace),"generated_but_never_transmitted":sum(f(r.get("downlinked_mb"))<=0 for r in trace),"partially_transmitted":sum(0<f(r.get("downlinked_mb"))<f(r.get("compressed_size_mb"))-1e-9 for r in trace),"fully_transmitted":sum(f(r.get("downlinked_mb"))>=f(r.get("compressed_size_mb"))-1e-9 for r in trace),"reconstruction_source":"rerun B1 per-task compression/downlink trace; staged product fields not applicable"}
    current={str(r["lifecycle_id"]):r for r in res["runner"].last_payload_lifecycle_current_records}
    transitions=list(res["runner"].last_payload_lifecycle_ledger_trace)
    generated={str(r["lifecycle_id"]):r for r in transitions if str(r.get("transition_reason"))=="payload_generated"}
    partial_ids={str(r["lifecycle_id"]) for r in transitions if str(r.get("transition_reason"))=="partial_downlink_remaining"}
    minimum={"alert":"metadata","verification":"roi","context":"roi_context","evidence":"roi_context"}
    return {"scenario_seed":seed,"capacity":cap,"method":method,"metadata_generated":sum(str(r.get("product_type"))=="metadata" for r in generated.values()),"roi_generated":sum(str(r.get("product_type"))=="roi" for r in generated.values()),"roi_context_generated":sum(str(r.get("product_type"))=="roi_context" for r in generated.values()),"minimum_closing_products_generated":sum(str(r.get("product_type"))==minimum.get(str(r.get("product_requirement"))) for r in generated.values()),"generated_before_deadline":sum(f(r.get("created_slot"))<=f(r.get("deadline_slot")) for r in generated.values()),"generated_too_late":sum(f(r.get("created_slot"))>f(r.get("deadline_slot")) for r in generated.values()),"generated_but_never_transmitted":sum(i not in partial_ids and str(current.get(i,{}).get("current_status"))!="delivered" for i in generated),"partially_transmitted":sum(i in partial_ids and str(current.get(i,{}).get("current_status"))!="delivered" for i in generated),"fully_transmitted":sum(str(current.get(i,{}).get("current_status"))=="delivered" for i in generated),"reconstruction_source":"reporting-only rerun; unique payload_generated lifecycle IDs reconciled to final lifecycle state"}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seeds",default=",".join(map(str,SEEDS)));parser.add_argument("--suffix",default="");args=parser.parse_args();selected=tuple(int(x) for x in args.seeds.split(",") if x);suffix=f".{args.suffix}" if args.suffix else ""
    OUT.mkdir(parents=True,exist_ok=True)
    def target(stem,partial=False):return OUT/f"{stem}{suffix}{'.partial' if partial else ''}.csv"
    for name in (target("reconstructed_task_outcomes"),target("product_generation_summary_corrected"),target("reconstruction_method_summary")):
        if name.exists():raise FileExistsError(name)
    accounting=load_yaml(ROOT/"configs/unified_ground_received_accounting.yaml");utility=load_yaml(ROOT/"configs/product_requirement_utility.yaml");reference=read(REFERENCE)
    tasks=[];products=[];summaries=[]
    for seed in selected:
      for cap in CAPS:
        base=cfg(seed,cap,False);timeline=TimelineBuilder(base["dataset"],base["timeline"],base["task_stream"]).build();check=validate_scenario(base,timeline);manifest=manifest_rows(timeline,check["task_manifest_hash_v2"]);classes={step.task.task_id:step.task.task_class for step in timeline if step.task};available=available_capacity_mb(timeline,float(base["simulator"].get("downlink_efficiency",1)));horizon=len(timeline)*float(base["timeline"]["slot_seconds"])
        for method in METHODS:
            c=cfg(seed,cap,method!="PROP-PAUSE");res=run_b1(c,timeline,manifest,accounting) if method.startswith("B1") else run_prop(method,c,timeline,manifest,accounting,utility);summary=summarize(seed,cap,method,res,available,horizon);summary["task_hash"]=check["task_manifest_hash_v2"];summary["environment_hash"]=check["environment_hash_v2"]
            old=next(r for r in reference if int(r["seed"])==seed and f(r["capacity_multiplier"])==cap and r["method"]==method)
            for key in ("task_count","total_closure","hv_closure","hv_target_closure","transmitted_mb","common_net_utility","maximum_thermal_state"):
                if abs(f(summary[key])-f(old[key]))>1e-8:raise RuntimeError(f"frozen result changed: {seed} {cap} {method} {key}")
            summaries.append(summary);products.append(product_summary(seed,cap,method,res))
            life_by_task={}
            for life in res.get("lifecycle",[]):
                life_by_task.setdefault(str(life.get("task_id")),[]).append(life)
            b1_by_task={str(r["task_id"]):r for r in res.get("raw",{}).get("per_task_delivery_trace",[])}
            for row in res["rows"]:
                task_id=str(row["task_id"]);lives=life_by_task.get(task_id,[]);b1=b1_by_task.get(task_id,{})
                transmitted=sum(f(x.get("payload_size_mb")) for x in lives if str(x.get("current_status"))=="delivered") if lives else f(b1.get("downlinked_mb"))
                tasks.append({**row,"scenario_seed":seed,"capacity":cap,"method":method,"task_class":classes[task_id],"attributed_transmitted_mb":transmitted,"failure_reason_frozen":row.get("failure_reason") or b1.get("blocked_reason") or ("partial_only" if row.get("partial_ground_delivery") else "no_ground" if not row.get("ground_completed") else "")})
            write(target("reconstructed_task_outcomes",True),tasks);write(target("product_generation_summary_corrected",True),products)
    write(target("reconstructed_task_outcomes"),tasks);write(target("product_generation_summary_corrected"),products);write(target("reconstruction_method_summary"),summaries)
    print(OUT)
if __name__=="__main__":main()
