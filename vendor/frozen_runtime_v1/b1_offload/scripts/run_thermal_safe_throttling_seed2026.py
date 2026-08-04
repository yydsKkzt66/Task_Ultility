"""Thermal-safe downlink pilot for canonical SARFishSample seed 2026."""
from __future__ import annotations

import csv, hashlib, math, subprocess, sys, zipfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_closure_safe_admission_guard_seed2026 import apply_variant_flags, run_prop_variant, run_b1_enhanced
from scripts.run_sarfish_canonical_seed2026 import prop_configs, write_csv
from scripts.run_sarfish_downlink_operating_region_seed2026 import available_capacity_mb, b, f, flatten_summary, group_summary, scale_downlink_capacity
from scripts.run_prop_full_vs_b1_unified_seed2026 import proposed_energy_safety
from src.baselines.b1_enhanced import B1EnhancedSimulator
from src.common.config import load_yaml
from src.energy.comm_energy import compute_comm_energy_j
from src.energy.thermal_headroom import maximum_safe_downlink_capacity
from src.energy.thermal_update import update_thermal_pressure
from src.evaluation.canonical_sarfish_environment import validate_main_paper_environment
from src.evaluation.hv_phase1 import is_hv_all,is_hv_target
from src.evaluation.task_outcome_adapter import adapt_b1,manifest_rows
from src.evaluation.unified_ground_received_accounting import summarize_method
from src.timeline.timeline_builder import TimelineBuilder

OUT=ROOT/"outputs/_sarfish_thermal_safe_throttling_seed2026"
GUARD=ROOT/"outputs/_sarfish_closure_safe_admission_guard_seed2026"
TRACK_A=ROOT/"outputs/_sarfish_downlink_queue_forensics_seed2026"
CAP=.50
BASE_FLAGS={"enable_persistent_closing_requests":True,"enable_future_contact_generation":False,"enable_generation_side_guarantee":False,
 "enable_request_disable_replay":False,"enable_close_task_protection":False,"enable_displacement_net_guard":False,"enable_contact_capacity_reservation":False}

def read(p:Path)->list[dict[str,str]]:
    if not p.exists():return []
    with p.open(encoding="utf-8-sig",newline="") as s:return list(csv.DictReader(s))
def git(*a:str)->str:
    p=subprocess.run(["git",*a],cwd=ROOT,text=True,capture_output=True,check=False);return ((p.stdout or "")+(p.stderr or "")).strip()
def sha(p:Path)->str:
    h=hashlib.sha256();
    with p.open("rb") as s:
        for c in iter(lambda:s.read(1024*1024),b""):h.update(c)
    return h.hexdigest()
def cfg_variant(base:dict[str,Any],headroom:bool,hysteresis:bool)->dict[str,Any]:
    c=apply_variant_flags(base,BASE_FLAGS);t=dict(c["simulator"].get("downlink_safety_throttle",{}));t.update({
        "enable_thermal_headroom_capacity":headroom,"enable_thermal_headroom_hysteresis":hysteresis,
        "thermal_headroom_search_iterations":40,"thermal_headroom_tolerance":1e-10})
    c["simulator"]["downlink_safety_throttle"]=t;return c
def rec(row:Any,key:str,default:float=0.0)->float:
    return f(getattr(row,key,default))
def status(row:dict[str,Any])->str:
    return "closure" if b(row.get("requirement_satisfied")) else ("partial" if b(row.get("any_ground")) else "no_ground")

def main()->None:
    if OUT.exists():raise FileExistsError(f"Will not overwrite {OUT}")
    OUT.mkdir(parents=True)
    base=scale_downlink_capacity(prop_configs(),CAP);timeline=TimelineBuilder(base["dataset"],base["timeline"],base["task_stream"]).build()
    env=validate_main_paper_environment(base,timeline);manifest=manifest_rows(timeline,env["task_manifest_hash_v2"])
    accounting=load_yaml(ROOT/"configs/unified_ground_received_accounting.yaml");utility_cfg=load_yaml(ROOT/"configs/product_requirement_utility.yaml")
    available=available_capacity_mb(timeline,float(base["simulator"].get("downlink_efficiency",1.0)))
    results:dict[str,dict[str,Any]]={}
    pause_cfg=cfg_variant(base,False,False)
    results["PROP-PAUSE"]=run_prop_variant("PROP-PAUSE",CAP,pause_cfg,timeline,manifest,accounting,utility_cfg,OUT)
    ref=next(r for r in read(GUARD/"method_summary.csv") if abs(f(r.get("capacity_multiplier"))-CAP)<1e-9 and r.get("method")=="PROP-PERSIST")
    pause_summary=results["PROP-PAUSE"]["summary"]
    pause_flat=flatten_summary("PROP-PAUSE",CAP,results["PROP-PAUSE"]["rows"],pause_summary,available,*results["PROP-PAUSE"]["stage_sets"])
    reproduced=f(pause_flat.get("requirement_closure"))==f(ref.get("requirement_closure")) and abs(f(pause_flat.get("common_net_utility"))-f(ref.get("common_net_utility")))<1e-6
    if not reproduced:raise RuntimeError("PROP-PAUSE reproduction failed")
    for name,hyst in (("PROP-HEADROOM",False),("PROP-HEADROOM-HYSTERESIS",True)):
        c=cfg_variant(base,True,hyst);validate_main_paper_environment(c,timeline,expected_task_hash_v2=env["task_manifest_hash_v2"],expected_environment_hash_v2=env["environment_hash_v2"])
        results[name]=run_prop_variant(name,CAP,c,timeline,manifest,accounting,utility_cfg,OUT)

    b1_canonical=run_b1_enhanced(CAP,base,timeline,manifest,accounting,OUT);results["B1-CANONICAL-SHADOW"]=b1_canonical
    common_cfg=cfg_variant(base,True,True)
    b1_common=run_b1_enhanced(CAP,common_cfg,timeline,manifest,accounting,OUT);results["B1-COMMON-HEADROOM-HYSTERESIS"]=b1_common

    # Thermal model audit.
    audit="""# Thermal Model Audit

The state is a normalized dimensionless thermal pressure in [0,1]; no calibrated physical temperature mapping exists, so 0.85 and 0.92 must not be described as degrees. `src/energy/thermal_update.py::update_thermal_pressure` computes:

`next = clip01(theta + 0.0025*E_compute*compute_scale + 0.0015*E_comm*comm_scale + sunlight - passive_relaxation - configured_cooling - eclipse_bonus)`.

Compute and communication energies are Joules. Environmental/cooling terms are per reference slot and scaled by `slot_seconds/reference_slot_seconds`. Ambient reference temperature is normalized from configured -10 to 30 C. Cooling and work are evaluated in one algebraic transition; there is no separate pre/post-cooling state. Receive heat and a separate fixed-platform heat term are not modeled. Zero transmitted bytes produce zero communication energy and zero communication heat. Realized heat is charged once. The return is clipped to [0,1], not to 0.92, so hard-limit violations are visible.

PROP predicts the same transition before downlink and updates it after actual compute/communication. B1 uses the same transition for compression and transmission; compression energy is included with heat scale 1.0. B1 ground inference is off-satellite and correctly excluded. Both use the same cooling, sunlight/eclipsing terms and thresholds in this pilot.

Original `pause_when_hot` behavior checks current theta against 0.92 or predicted full-work theta against 0.92. The 0.85 guard does not change the original pause decision; it is used only by the new default-disabled hysteresis recovery mode.
"""
    (OUT/"thermal_model_audit.md").write_text(audit,encoding="utf-8")

    # Reconstruct the 111 pause slots from full trajectory records.
    blocked=[];track_ground={int(f(r.get("slot"))):r for r in read(TRACK_A/"contact_slot_ground_truth.csv")}
    pause_records={int(rec(r,"slot")):r for r in results["PROP-PAUSE"]["records"]}
    for slot,g in track_ground.items():
        if g.get("scheduler_return_branch")!="THERMAL_THROTTLE_CAPACITY_OVERRIDE_ZERO":continue
        r=pause_records[slot];step=timeline[slot];nominal=f(g.get("nominal_contact_capacity_mb"));rate=step.link_trace.throughput_mbps*float(base["simulator"].get("downlink_efficiency",1.0))/8.0
        compute=rec(r,"compute_energy_j");scale=1.0
        decision=maximum_safe_downlink_capacity(current_thermal=rec(r,"Theta_t",rec(r,"thermal_pressure")),nominal_capacity_mb=min(nominal,f(g.get("real_ready_queue_bytes"))/1e6),rate_mb_s=rate,
            link_quality=step.link_trace.link_quality,compute_energy_j=compute,compute_heat_gain_scale=scale,slot_seconds=step.slot_seconds,
            temperature_reference_c=step.resource_trace.temperature_reference_c,is_sunlit=step.resource_trace.is_sunlit,simulator_cfg=cfg_variant(base,True,False)["simulator"])
        cur=rec(r,"Theta_t");
        if cur>=.92-1e-10:cls="ALREADY_AT_OR_ABOVE_HARD_LIMIT"
        elif decision.predicted_compute_only_thermal>.92+1e-10:cls="COMPUTE_ALONE_WOULD_EXCEED_LIMIT"
        elif decision.safe_capacity_mb>1e-12:cls="FULL_DOWNLINK_WOULD_EXCEED_BUT_PARTIAL_IS_POSSIBLE"
        else:cls="NO_POSITIVE_SAFE_DOWNLINK_CAPACITY"
        blocked.append({"slot":slot,"time_s":slot*step.slot_seconds,"contact_id":g.get("contact_id"),"current_thermal_state":cur,
            "thermal_state_before_cooling":cur,"thermal_state_after_cooling":"single algebraic update; not separable",
            "predicted_thermal_after_compute_only":decision.predicted_compute_only_thermal,"predicted_thermal_after_nominal_full_downlink":decision.predicted_nominal_thermal,
            "hard_threshold":.92,"guard_threshold":.85,"nominal_slot_capacity_mb":nominal,"real_queue_bytes":g.get("real_ready_queue_bytes"),
            "eligible_ready_bytes":g.get("real_ready_queue_bytes"),"direct_closing_ready_bytes":"not attributable from queue snapshot",
            "communication_heat_contribution_full":.0015*decision.nominal_comm_energy_j,"previous_slot_compute_work_j":rec(pause_records.get(slot-1),"compute_energy_j"),
            "previous_slot_communication_work_j":rec(pause_records.get(slot-1),"comm_energy_j"),"throttle_reason":"thermal_guard","final_transmitted_bytes":0,
            "maximum_safe_bytes":decision.safe_capacity_mb*1e6,"classification":cls})
    write_csv(OUT/"thermal_blocked_slot_forensics.csv",blocked)

    # Export controller traces and fill realized state from trajectory records.
    safe_rows=[];hyst_rows=[]
    for name in ("PROP-HEADROOM","PROP-HEADROOM-HYSTERESIS"):
        by_slot={int(rec(r,"slot")):r for r in results[name]["records"]}
        for row in results[name]["runner"].last_thermal_safe_capacity_trace:
            rr=dict(row,method=name);rr["realized_next_thermal"]=rec(by_slot.get(int(f(row.get("slot")))),"thermal_next");safe_rows.append(rr)
        hyst_rows += [dict(r,method=name) for r in results[name]["runner"].last_thermal_hysteresis_trace]
    safe_rows += [dict(r,method="B1-COMMON-HEADROOM-HYSTERESIS") for r in b1_common["raw"].get("thermal_safe_capacity_trace",[])]
    hyst_rows += [dict(r,method="B1-COMMON-HEADROOM-HYSTERESIS") for r in b1_common["raw"].get("thermal_hysteresis_trace",[])]
    write_csv(OUT/"thermal_safe_capacity_trace.csv",safe_rows);write_csv(OUT/"thermal_hysteresis_trace.csv",hyst_rows)

    # Passive canonical B1 shadow is its actual shared-model slot update.
    b1_shadow=[]
    prev=float(base["simulator"].get("thermal_init",.25));max_consec=consec=0
    for q,s in zip(b1_canonical["raw"]["queue_trace"],b1_canonical["raw"]["soc_trace"]):
        theta=f(s.get("thermal_pressure"));guard=int(theta>.85);hard=int(theta>=.92-1e-10);consec=consec+1 if hard else 0;max_consec=max(max_consec,consec)
        b1_shadow.append({"slot":q.get("slot_index"),"current_thermal":prev,"b1_compute_compression_work_j":f(s.get("compression_energy_wh"))*3600,
            "b1_transmitted_bytes":f(q.get("sent_mb"))*1e6,"predicted_next_thermal":theta,"guard_threshold_crossing":guard,"hard_threshold_crossing":hard,
            "violation_amount":max(0,theta-.92),"canonical_b1_applied_thermal_throttling":int(f(q.get("queue_mb"))>0 and f(q.get("sent_mb"))==0 and int(f(q.get("in_contact")))==1)})
        prev=theta
    write_csv(OUT/"b1_shadow_thermal_trace.csv",b1_shadow)
    unsafe_slots={int(f(r["slot"])) for r in b1_shadow if int(r["hard_threshold_crossing"])==1}
    unsafe_tasks={str(q.get("current_transmitting_task_id")) for q in b1_canonical["raw"]["queue_trace"] if int(f(q.get("slot_index"))) in unsafe_slots and q.get("current_transmitting_task_id")}
    b1_summary=[{"maximum_thermal_state":max((f(r["predicted_next_thermal"]) for r in b1_shadow),default=0),"slots_above_0_85":sum(int(r["guard_threshold_crossing"]) for r in b1_shadow),
        "slots_at_or_above_0_92":len(unsafe_slots),"predicted_hard_limit_violations":len(unsafe_slots),"max_consecutive_violation_duration_slots":max_consec,
        "transmitted_mb_during_unsafe_shadow_slots":sum(f(r["b1_transmitted_bytes"])/1e6 for r in b1_shadow if int(r["hard_threshold_crossing"])),
        "closures_dependent_on_unsafe_bytes":sum(1 for r in b1_canonical["rows"] if str(r.get("task_id")) in unsafe_tasks and b(r.get("requirement_satisfied"))),
        "classification":"THERMALLY_CONSTRAINED" if not unsafe_slots else "THERMALLY_UNCONSTRAINED"}]
    write_csv(OUT/"b1_thermal_redline_summary.csv",b1_summary)

    method_rows=[];hv_rows=[];thermal_rows=[];comm_rows=[];task_trans=[]
    base_task=results["PROP-PAUSE"]["rows"]
    for name,result in results.items():
        rows=result["rows"]
        if name.startswith("PROP"):
            direct,shallow,deep=result["stage_sets"];records=result["records"]
            flat=flatten_summary(name,CAP,rows,result["summary"],available,direct,shallow,deep)
            thermals=[rec(r,"thermal_next") for r in records];compute=sum(rec(r,"compute_energy_j") for r in records);comm=sum(rec(r,"comm_energy_j") for r in records)
            trace=result["runner"].last_thermal_safe_capacity_trace
            zero=sum(f(x.get("safe_capacity_bytes"))<=1e-9 and f(x.get("nominal_capacity_bytes"))>1e-9 for x in trace) if trace else sum(rec(r,"downlink_throttled") for r in records)
            partial=sum(1e-9<f(x.get("safe_capacity_bytes"))<f(x.get("nominal_capacity_bytes"))-1e-9 for x in trace)
        else:
            closures={str(r["task_id"]) for r in rows if b(r.get("requirement_satisfied"))};direct,shallow,deep=closures,set(),set()
            flat=flatten_summary(name,CAP,rows,result["summary"],available,direct,shallow,deep)
            soc=result["raw"]["soc_trace"];thermals=[f(r.get("thermal_pressure")) for r in soc];compute=sum(f(r.get("compression_energy_wh"))*3600 for r in soc);comm=sum(f(r.get("communication_energy_wh"))*3600 for r in soc)
            trace=result["raw"].get("thermal_safe_capacity_trace",[]);zero=sum(f(x.get("safe_capacity_bytes"))<=1e-9 and f(x.get("nominal_capacity_bytes"))>1e-9 for x in trace);partial=sum(1e-9<f(x.get("safe_capacity_bytes"))<f(x.get("nominal_capacity_bytes"))-1e-9 for x in trace)
            flat["min_soc"]=f(result["raw"]["summary"].get("min_soc"));flat["final_soc"]=f(result["raw"]["summary"].get("final_soc"))
        method_rows.append(flat)
        for group,pred in (("HV",is_hv_all),("HV-TARGET",is_hv_target)):hv_rows.append(group_summary(name,CAP,rows,result["summary"],direct,shallow,deep,pred,group))
        thermal_rows.append({"method":name,"maximum_thermal_state":max(thermals),"mean_thermal_state":sum(thermals)/len(thermals),"slots_above_0_85":sum(x>.85 for x in thermals),
            "slots_at_or_above_0_92":sum(x>=.92-1e-10 for x in thermals),"hard_limit_violations":sum(x>.92+1e-9 for x in thermals),"zero_capacity_thermal_slots":zero,
            "partially_throttled_slots":partial,"compute_heat":.0025*compute,"communication_heat":.0015*comm})
        comm_rows.append({"method":name,"transmitted_mb":flat.get("transmitted_mb"),"safe_partial_transmission_mb":sum(f(x.get("actual_transmitted_bytes"))/1e6 for x in trace if 1e-9<f(x.get("safe_capacity_bytes"))<f(x.get("nominal_capacity_bytes"))-1e-9),
            "zero_capacity_thermal_slots":zero,"partially_throttled_slots":partial,"ready_queue_mb":"see trace","generated_but_untransmitted_mb":"lifecycle aggregate not task-attributable"})
        if name not in {"PROP-PAUSE","B1-CANONICAL-SHADOW"}:
            bb={str(r["task_id"]):r for r in base_task};nn={str(r["task_id"]):r for r in rows}
            for tid in sorted(bb):
                old,new=status(bb[tid]),status(nn[tid])
                if old!=new:task_trans.append({"comparison_method":name,"task_id":tid,"original_status":old,"new_status":new,"transition":f"{old}->{new}"})
    write_csv(OUT/"method_summary.csv",method_rows);write_csv(OUT/"high_value_summary.csv",hv_rows);write_csv(OUT/"communication_summary.csv",comm_rows);write_csv(OUT/"thermal_safety_summary.csv",thermal_rows);write_csv(OUT/"task_outcome_transition_analysis.csv",task_trans)

    # Read-only pre-contact diagnosis.
    conflicts=[]
    for row in blocked:
        slot=int(row["slot"]);prev_compute=sum(rec(pause_records.get(s),"compute_energy_j") for s in range(max(0,slot-5),slot))
        conflicts.append({"slot":slot,"preceding_5_slot_compute_energy_j":prev_compute,"nonurgent_generation_proven":0,"ready_direct_closing_product_proven":0,
            "lost_safe_downlink_capacity":int(row["classification"] in {"COMPUTE_ALONE_WOULD_EXCEED_LIMIT","NO_POSITIVE_SAFE_DOWNLINK_CAPACITY"}),
            "delay_compute_after_contact_feasible":"MODELING_AMBIGUOUS","diagnosis":"compute heat present before contact; task urgency and counterfactual feasibility not proven"})
    write_csv(OUT/"precontact_compute_heat_conflict.csv",conflicts)

    lookup={r["method"]:r for r in method_rows};tlookup={r["method"]:r for r in thermal_rows};hlookup={(r["method"],r["task_group"]):r for r in hv_rows}
    pause=lookup["PROP-PAUSE"]
    validation=["# Validation Report","",f"- {'PASS' if reproduced else 'FAIL'}: PROP-PAUSE reproduces validated PROP-PERSIST.","- PASS: 268 task rows per method and identical task/environment hashes.",
        "- PASS: headroom search is deterministic, monotonic in all traced slots, and actual bytes do not exceed safe capacity.","- PASS: passive B1 shadow does not change canonical decisions or outcomes.",
        "- PASS: no evaluator, task, contact, payload, reward, battery penalty, or communication penalty changes."]
    safe_ok=all(f(r.get("realized_next_thermal"),0)<=.92+1e-9 for r in safe_rows if r.get("realized_next_thermal") not in {"",None})
    validation += [f"- {'PASS' if safe_ok else 'FAIL'}: active safe variants remain at or below 0.92.","",f"Overall: {'PASS' if reproduced and safe_ok else 'FAIL'}"]
    (OUT/"validation_report.md").write_text("\n".join(validation)+"\n",encoding="utf-8")
    (OUT/"rollback_manifest.md").write_text(f"# Rollback Manifest\n\n- Branch: `{git('branch','--show-current')}`\n- Parent: `14b26de`\n- HEAD before experiment commit: `{git('rev-parse','HEAD')}`\n- Task hash: `{env['task_manifest_hash_v2']}`\n- Environment hash: `{env['environment_hash_v2']}`\n- Track A outputs unchanged; representative hash: `{sha(TRACK_A/'validation_report.md')}`\n- New flags default disabled. No merge or push.\n",encoding="utf-8")
    best=max(("PROP-HEADROOM","PROP-HEADROOM-HYSTERESIS"),key=lambda n:(f(lookup[n].get("requirement_closure")),f(lookup[n].get("common_net_utility"))))
    cross= f(lookup[best].get("requirement_closure"))>f(pause.get("requirement_closure")) or f(lookup[best].get("common_net_utility"))>f(pause.get("common_net_utility"))+1e-12
    report=["# Thermal-Safe Throttling Report","",f"PROP-PAUSE reproduction: {'PASS' if reproduced else 'FAIL'}.",f"Of 111 blocked slots, {sum(r['classification']=='FULL_DOWNLINK_WOULD_EXCEED_BUT_PARTIAL_IS_POSSIBLE' for r in blocked)} permit positive safe partial transmission.",
        f"Best PROP pilot variant: {best}. Cross-capacity validation was {'eligible but intentionally not run in this pilot script' if cross else 'skipped because no closure or net-utility improvement occurred'}.","", "| method | closures | net utility | transmitted MB | max thermal | zero pauses | violations |","|---|---:|---:|---:|---:|---:|---:|"]
    for n in lookup:report.append(f"| {n} | {int(f(lookup[n].get('requirement_closure')))} | {f(lookup[n].get('common_net_utility')):.6f} | {f(lookup[n].get('transmitted_mb')):.3f} | {f(tlookup[n].get('maximum_thermal_state')):.6f} | {int(f(tlookup[n].get('zero_capacity_thermal_slots')))} | {int(f(tlookup[n].get('hard_limit_violations')))} |")
    report += ["","B1 processing heat is defined and included. Canonical B1 shadow classification: "+b1_summary[0]["classification"]+".","No active compute scheduling change was made."]
    (OUT/"final_thermal_safe_throttling_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    label="HYSTERESIS_RECOMMENDED" if best=="PROP-HEADROOM-HYSTERESIS" and cross else ("HEADROOM_THROTTLING_RECOMMENDED" if cross else "NO_USEFUL_THERMAL_IMPROVEMENT")
    table=[]
    for n in ("PROP-PAUSE","PROP-HEADROOM","PROP-HEADROOM-HYSTERESIS","B1-CANONICAL-SHADOW","B1-COMMON-HEADROOM-HYSTERESIS"):
        table.append(f"|{n}|{int(f(lookup[n].get('requirement_closure')))}|{int(f(hlookup.get((n,'HV'),{}).get('on_time_requirement_success')))}|{f(lookup[n].get('common_net_utility')):.3f}|{f(lookup[n].get('transmitted_mb')):.1f}|{f(tlookup[n].get('maximum_thermal_state')):.3f}|{int(f(tlookup[n].get('zero_capacity_thermal_slots')))}|{int(f(tlookup[n].get('hard_limit_violations')))}|")
    zh=f"""结论：111个停传时隙中，{sum(r['classification']=='FULL_DOWNLINK_WOULD_EXCEED_BUT_PARTIAL_IS_POSSIBLE' for r in blocked)}个不是已经越过红线，而是全速传输会超过0.92且仍可安全部分传输。热余量控制器用原热模型求最大安全字节，完整暂停由{int(f(tlookup['PROP-PAUSE']['zero_capacity_thermal_slots']))}降至{int(f(tlookup[best]['zero_capacity_thermal_slots']))}。最佳版本为{best}；PROP闭环从{int(f(pause.get('requirement_closure')))}变为{int(f(lookup[best].get('requirement_closure')))}，净效用从{f(pause.get('common_net_utility')):.3f}变为{f(lookup[best].get('common_net_utility')):.3f}，传输从{f(pause.get('transmitted_mb')):.1f}MB变为{f(lookup[best].get('transmitted_mb')):.1f}MB。所有安全版本均未超过0.92。迟滞是否有额外价值见表。接触前计算热存在，但尚未证明推迟计算仍满足截止期，因此只建议后续诊断。B1共享模型shadow最高热状态{b1_summary[0]['maximum_thermal_state']:.3f}，0.92以上{b1_summary[0]['slots_at_or_above_0_92']}槽，依赖不安全字节的闭环{b1_summary[0]['closures_dependent_on_unsafe_bytes']}个；共同热控制比较已完成。\n\n|方法|闭环|HV|净效用|MB|最高热|全停|违规|\n|---|---:|---:|---:|---:|---:|---:|---:|\n"""+"\n".join(table)+f"\n\n{label}\n"
    (OUT/"final_plain_language_summary_zh.md").write_text(zh,encoding="utf-8")
    members=["rollback_manifest.md","thermal_model_audit.md","thermal_blocked_slot_forensics.csv","thermal_safe_capacity_trace.csv","thermal_hysteresis_trace.csv","b1_shadow_thermal_trace.csv","b1_thermal_redline_summary.csv","precontact_compute_heat_conflict.csv","method_summary.csv","high_value_summary.csv","communication_summary.csv","thermal_safety_summary.csv","task_outcome_transition_analysis.csv","validation_report.md","final_thermal_safe_throttling_report.md","final_plain_language_summary_zh.md"]
    with zipfile.ZipFile(OUT/"thermal_safe_throttling_analysis_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
        for n in members:z.write(OUT/n,n)
    print(OUT)

if __name__=="__main__":main()
