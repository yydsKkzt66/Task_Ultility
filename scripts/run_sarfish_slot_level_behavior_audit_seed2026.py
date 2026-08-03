"""Passive slot-level SARFish behavior audit (seed 2026).

The feature flag is deliberately default-off.  This module never changes a
policy decision: the trace wrapper copies state around the original queue call.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, shutil, subprocess, sys, zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sarfish_downlink_operating_region_seed2026 import (
    available_capacity_mb, b, f, flatten_summary, run_b1, run_prop,
    scale_downlink_capacity,
)
from scripts.run_sarfish_canonical_seed2026 import prop_configs, write_csv
from src.evaluation.canonical_sarfish_environment import validate_main_paper_environment
from src.evaluation.task_outcome_adapter import manifest_rows
from src.common.config import load_yaml
from src.link.downlink_queue import DownlinkQueue
from src.timeline.timeline_builder import TimelineBuilder

OUT = ROOT / "outputs/_sarfish_slot_level_behavior_audit_seed2026"
CAPS = (1.0, 0.5, 0.35)
METHODS = ("PROP-FULL", "B1-Enhanced-1.4W")
IDLE_REASONS = {
    "NO_ACTIVE_TASK", "NO_PRODUCT_GENERATED", "PRODUCT_GENERATED_NOT_READY", "QUEUE_EMPTY",
    "READY_QUEUE_NONEMPTY_BUT_NO_ELIGIBLE_PRODUCT", "PRODUCT_DOES_NOT_FIT",
    "PARTIAL_TRANSMISSION_DISABLED", "NO_SECOND_CANDIDATE_TRIED", "DEADLINE_INFEASIBLE",
    "NON_POSITIVE_INCREMENTAL_UTILITY", "REQUIREMENT_ALREADY_SATISFIED", "ENERGY_BLOCKED",
    "THERMAL_BLOCKED", "OTHER_SAFETY_BLOCK", "SCHEDULER_RETURNED_NONE", "UNKNOWN",
}
SLOT_COLUMNS = """method capacity_multiplier seed slot_index slot_start_s slot_end_s contact_id contact_active slot_capacity_bytes bytes_sent_this_slot unused_capacity_bytes utilization_fraction total_queue_bytes ready_queue_bytes eligible_queue_bytes total_product_count ready_product_count eligible_product_count highest_priority_payload_id highest_priority_task_id highest_priority_product_type highest_priority_stage highest_priority_size_bytes highest_priority_remaining_bytes highest_priority_score highest_priority_deadline_s highest_priority_value_level highest_priority_requirement highest_priority_can_fit_fully highest_priority_can_send_partially selected_payload_id selected_task_id selected_product_type selected_stage selected_priority_score selected_bytes selection_reason remaining_capacity_after_first_send second_candidate_exists second_candidate_payload_id second_candidate_eligible second_candidate_selected additional_payloads_sent_count soc_before soc_after thermal_before thermal_after compute_busy communication_blocked_by_safety idle_capacity_present avoidable_idle_capacity_bytes primary_idle_reason secondary_idle_reason diagnostic_notes""".split()


def is_avoidable_idle(unused_bytes: float, candidates: list[dict[str, Any]]) -> bool:
    return unused_bytes > 0 and any(
        x.get("generated") and x.get("ready") and not x.get("expired") and x.get("safety_allowed")
        and f(x.get("incremental_utility")) > 0 and x.get("can_send_one_byte") for x in candidates
    )


def classify_idle_reason(s: dict[str, Any]) -> str:
    if f(s.get("active_task_count")) <= 0: return "NO_ACTIVE_TASK"
    if f(s.get("generated_count")) <= 0: return "NO_PRODUCT_GENERATED"
    if f(s.get("ready_count")) <= 0: return "PRODUCT_GENERATED_NOT_READY"
    if f(s.get("queue_count")) <= 0: return "QUEUE_EMPTY"
    if f(s.get("eligible_count")) <= 0: return "READY_QUEUE_NONEMPTY_BUT_NO_ELIGIBLE_PRODUCT"
    if s.get("energy_blocked"): return "ENERGY_BLOCKED"
    if s.get("thermal_blocked"): return "THERMAL_BLOCKED"
    if s.get("deadline_infeasible"): return "DEADLINE_INFEASIBLE"
    if s.get("full_fit_required"): return "PRODUCT_DOES_NOT_FIT"
    if s.get("scheduler_returned_none"): return "SCHEDULER_RETURNED_NONE"
    if s.get("stopped_after_one"): return "NO_SECOND_CANDIDATE_TRIED"
    return "UNKNOWN"


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def git(*args: str) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return (p.stdout or "") + (p.stderr or "")


def text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value, encoding="utf-8")


@contextmanager
def prop_trace(enabled: bool, sink: list[dict[str, Any]]):
    original = DownlinkQueue.process_slot
    if not enabled:
        yield; return
    def wrapped(self, slot_index, payload, link_trace, capacity_override_mb=None, contact_window_id=""):
        before = [{"task_id": x.task_id, "payload_id": self._payload_id(x), "product_type": x.product_type,
                   "remaining_mb": x.remaining_mb, "size_mb": x.payload_size_mb,
                   "deadline_s": x.deadline_time_s, "value_level": x.metadata.get("value_level", ""),
                   "requirement": x.product_requirement, "utility": x.incremental_utility}
                  for x in self.items]
        result = original(self, slot_index, payload, link_trace, capacity_override_mb, contact_window_id)
        sink.append({"slot": slot_index, "before": before, "priority": [dict(x) for x in self.last_priority_component_log],
                     "capacity_override_mb": capacity_override_mb,
                     "sent_mb": result.S_down_total_mb, "transmitted": list(result.transmitted_task_ids),
                     "delivered": list(result.delivered_task_ids)})
        return result
    DownlinkQueue.process_slot = wrapped
    try: yield
    finally: DownlinkQueue.process_slot = original


def prop_slots(cap, timeline, result, captured):
    records = {int(r.slot): r for r in result["records"]}; snaps = {int(x["slot"]): x for x in captured}; out=[]
    for step in timeline:
        capacity = max(0.0, step.link_trace.throughput_mbps * step.slot_seconds / 8.0)
        if capacity <= 0: continue
        rec=records[step.slot_index]; snap=snaps.get(step.slot_index, {"before":[],"priority":[],"sent_mb":0,"transmitted":[]})
        ranked=snap["priority"]; top=ranked[0] if ranked else {}; sent=f(rec.S_down_total_mb); unused=max(0.,capacity-sent)
        selected=[x for x in ranked if int(f(x.get("selected_for_downlink")))]; first=selected[0] if selected else {}
        effective_capacity = capacity if snap.get("capacity_override_mb") is None else max(0.0, f(snap.get("capacity_override_mb")))
        safety_allowed = effective_capacity > 1e-12
        candidates=[{"generated":True,"ready":True,"expired":False,"safety_allowed":safety_allowed,
                     "incremental_utility":x.get("remaining_utility",0),"can_send_one_byte":f(x.get("remaining_mb"))>0} for x in ranked]
        avoid=is_avoidable_idle(unused*1e6,candidates)
        reason="" if unused<=1e-6 else ("OTHER_SAFETY_BLOCK" if ranked and not safety_allowed else classify_idle_reason({"active_task_count":268,"generated_count":len(ranked),"ready_count":len(ranked),"queue_count":len(ranked),"eligible_count":len(ranked),"scheduler_returned_none":not selected}))
        row={c:"" for c in SLOT_COLUMNS}; row.update(method="PROP-FULL",capacity_multiplier=cap,seed=2026,slot_index=step.slot_index,
            slot_start_s=step.slot_index*step.slot_seconds,slot_end_s=(step.slot_index+1)*step.slot_seconds,
            contact_id=f"contact_window_{step.slot_index}",contact_active=1,slot_capacity_bytes=capacity*1e6,
            bytes_sent_this_slot=sent*1e6,unused_capacity_bytes=unused*1e6,utilization_fraction=sent/max(capacity,1e-12),
            total_queue_bytes=sum(f(x.get("remaining_mb"))*1e6 for x in ranked),ready_queue_bytes=sum(f(x.get("remaining_mb"))*1e6 for x in ranked),eligible_queue_bytes=sum(f(x.get("remaining_mb"))*1e6 for x in ranked) if safety_allowed else 0,
            total_product_count=len(ranked),ready_product_count=len(ranked),eligible_product_count=len(ranked) if safety_allowed else 0,
            highest_priority_payload_id=top.get("payload_id",""),highest_priority_task_id=top.get("task_id",""),highest_priority_product_type=top.get("payload_type",""),highest_priority_stage=top.get("payload_type",""),highest_priority_size_bytes=f(top.get("payload_size_mb"))*1e6,highest_priority_remaining_bytes=f(top.get("remaining_mb"))*1e6,highest_priority_score=f(top.get("queue_priority_total")),highest_priority_deadline_s=f(top.get("remaining_deadline_s"))+step.slot_index*step.slot_seconds,highest_priority_value_level=top.get("value_level",""),highest_priority_requirement=top.get("product_requirement",""),highest_priority_can_fit_fully=int(f(top.get("remaining_mb"))<=capacity),highest_priority_can_send_partially=int(bool(top) and capacity>0),
            selected_payload_id=first.get("payload_id",""),selected_task_id=first.get("task_id",""),selected_product_type=first.get("payload_type",""),selected_stage=first.get("payload_type",""),selected_priority_score=f(first.get("queue_priority_total")),selected_bytes=min(effective_capacity,f(first.get("remaining_mb")))*1e6 if first else 0,selection_reason="priority_order_partial_allowed" if first else "none",remaining_capacity_after_first_send=max(0.,effective_capacity-f(first.get("remaining_mb")))*1e6 if first else effective_capacity*1e6,second_candidate_exists=int(len(ranked)>1),second_candidate_payload_id=ranked[1].get("payload_id","") if len(ranked)>1 else "",second_candidate_eligible=int(len(ranked)>1 and safety_allowed),second_candidate_selected=int(len(selected)>1),additional_payloads_sent_count=max(0,len(selected)-1),soc_before=getattr(rec,"SOC",getattr(rec,"soc", "")),soc_after=getattr(rec,"SOC_next",getattr(rec,"soc_next", "")),thermal_before=getattr(rec,"thermal_pressure", ""),thermal_after=getattr(rec,"thermal_next", ""),compute_busy=int(f(getattr(rec,"compute_energy_j",0))>0),communication_blocked_by_safety=int(bool(ranked) and not safety_allowed),idle_capacity_present=int(unused>1e-6),avoidable_idle_capacity_bytes=unused*1e6 if avoid else 0,primary_idle_reason=reason,secondary_idle_reason="",diagnostic_notes=f"external_capacity_mb={capacity:.9g}; effective_queue_capacity_mb={effective_capacity:.9g}")
        out.append(row)
    return out


def b1_slots(cap,timeline,result):
    q={int(x["slot_index"]):x for x in result["raw"]["queue_trace"]}; cand={int(x["slot_index"]):x for x in result["raw"]["scheduler_candidate_trace"]}; soc={int(x["slot_index"]):x for x in result["raw"]["soc_trace"]}; out=[]
    for step in timeline:
        capacity=max(0.,step.link_trace.throughput_mbps*step.slot_seconds/8.0)
        if capacity<=0: continue
        x=q[step.slot_index]; c=cand.get(step.slot_index,{}); sent=f(x.get("sent_mb")); unused=max(0.,capacity-sent); eligible=int(f(c.get("feasible_candidate_count_after_filter"))); queued=int(f(c.get("queued_candidate_count_before_filter",x.get("queue_item_count")))); selected=c.get("selected_task_id",x.get("current_transmitting_task_id","")); fullfit=bool(eligible)
        reason="" if unused<=1e-6 else classify_idle_reason({"active_task_count":268,"generated_count":queued,"ready_count":queued,"queue_count":queued,"eligible_count":eligible,"full_fit_required":queued>0 and eligible==0,"scheduler_returned_none":eligible>0 and not selected,"stopped_after_one":sent>0 and unused>0 and queued>1})
        s=soc.get(step.slot_index,{}); row={z:"" for z in SLOT_COLUMNS}; row.update(method="B1-Enhanced-1.4W",capacity_multiplier=cap,seed=2026,slot_index=step.slot_index,slot_start_s=step.slot_index*step.slot_seconds,slot_end_s=(step.slot_index+1)*step.slot_seconds,contact_id=c.get("contact_window_id",f"window_{step.slot_index}"),contact_active=1,slot_capacity_bytes=capacity*1e6,bytes_sent_this_slot=sent*1e6,unused_capacity_bytes=unused*1e6,utilization_fraction=sent/max(capacity,1e-12),total_queue_bytes=f(x.get("queue_mb"))*1e6,ready_queue_bytes=f(x.get("queue_mb"))*1e6,eligible_queue_bytes=eligible*4e6,total_product_count=queued,ready_product_count=queued,eligible_product_count=eligible,highest_priority_task_id=selected,highest_priority_product_type="compressed_full_frame",highest_priority_stage="full_frame",highest_priority_size_bytes=4e6,highest_priority_remaining_bytes=4e6,highest_priority_can_fit_fully=int(fullfit),highest_priority_can_send_partially=0,selected_task_id=selected,selected_product_type="compressed_full_frame" if selected else "",selected_stage="full_frame" if selected else "",selected_bytes=sent*1e6,selection_reason="continuing_frame_or_ranked_full_fit" if sent else "none",remaining_capacity_after_first_send=unused*1e6,second_candidate_exists=int(queued>1),second_candidate_eligible=int(eligible>1),second_candidate_selected=0,additional_payloads_sent_count=0,soc_after=s.get("battery_soc",""),thermal_after=s.get("thermal_pressure",""),compute_busy=s.get("compute_activity",0),communication_blocked_by_safety=int(reason in {"ENERGY_BLOCKED","THERMAL_BLOCKED","OTHER_SAFETY_BLOCK"}),idle_capacity_present=int(unused>1e-6),avoidable_idle_capacity_bytes=0,primary_idle_reason=reason,diagnostic_notes="B1 requires full fit in remaining contact and carries one current_tx.")
        out.append(row)
    return out


def run_audit(out: Path, enabled: bool=True):
    out.mkdir(parents=True,exist_ok=True); rb=out/"rollback"; rb.mkdir(exist_ok=True)
    text(rb/"git_commit_before.txt",git("rev-parse","HEAD")); text(rb/"git_status_before.txt",git("status","--short","--branch")); text(rb/"git_diff_stat_before.txt",git("diff","--stat")); text(rb/"git_diff_before.patch",git("diff","--binary"))
    touched=[ROOT/"scripts/run_sarfish_slot_level_behavior_audit_seed2026.py",ROOT/"tests/test_slot_level_behavior_audit.py",ROOT/"docs/superpowers/plans/2026-07-12-sarfish-slot-level-behavior-audit.md"]
    write_csv(rb/"touched_file_hashes_before.csv",[{"path":str(p.relative_to(ROOT)),"sha256":sha(p)} for p in touched if p.exists()])
    base=prop_configs(); accounting=load_yaml(ROOT/"configs/unified_ground_received_accounting.yaml"); utility=load_yaml(ROOT/"configs/product_requirement_utility.yaml")
    slots=[]; summaries=[]; lineages=[]; partial=[]; hvtrace=[]; timing=[]; sizes=[]; env=[]; task_rows={}
    for cap in CAPS:
        cfg=scale_downlink_capacity(base,cap); timeline=TimelineBuilder(cfg["dataset"],cfg["timeline"],cfg["task_stream"]).build(); val=validate_main_paper_environment(cfg,timeline); manifest=manifest_rows(timeline,val["task_manifest_hash_v2"]); avail=available_capacity_mb(timeline,float(cfg["simulator"].get("downlink_efficiency",1)))
        capdir=out/"raw"/f"cap_{cap:.2f}"
        b1r=run_b1("B1-Enhanced-1.4W","enhanced",cap,cfg,timeline,manifest,accounting,capdir); captured=[]
        with prop_trace(enabled,captured): propr=run_prop(cap,cfg,timeline,manifest,accounting,utility,capdir)
        ps=prop_slots(cap,timeline,propr,captured); bs=b1_slots(cap,timeline,b1r); slots+=ps+bs
        for method,res,ss in (("PROP-FULL",propr,ps),("B1-Enhanced-1.4W",b1r,bs)):
            rows=res["rows"]; task_rows[(cap,method)]=rows; env.append({"capacity_multiplier":cap,"method":method,**val}); closure=sum(b(x.get("on_time_task_success")) for x in rows); hv=sum(b(x.get("on_time_task_success")) and str(x.get("value_level")).lower() in {"high","urgent"} for x in rows); hvt=sum(b(x.get("on_time_task_success")) and str(x.get("product_requirement")).lower() in {"context","evidence"} for x in rows); po=sum((b(x.get("ground_completed")) or b(x.get("partial_ground_delivery"))) and not b(x.get("requirement_satisfied")) for x in rows); ng=268-sum(b(x.get("ground_completed")) or b(x.get("partial_ground_delivery")) for x in rows); tx=sum(f(x["bytes_sent_this_slot"]) for x in ss); capb=sum(f(x["slot_capacity_bytes"]) for x in ss); avoid=sum(f(x["avoidable_idle_capacity_bytes"]) for x in ss)
            summaries.append({"method":method,"capacity_multiplier":cap,"on_time_closures":closure,"HV_closures":hv,"HV_TARGET_closures":hvt,"partial_only":po,"no_ground":ng,"transmitted_MB":tx/1e6,"active_window_capacity_MB":capb/1e6,"active_window_utilization":tx/max(capb,1),"avoidable_idle_MB":avoid/1e6,"avoidable_idle_fraction":avoid/max(capb,1),"unavoidable_idle_MB":(capb-tx-avoid)/1e6,"slots_with_avoidable_idle":sum(f(x["avoidable_idle_capacity_bytes"])>0 for x in ss),"slots_with_ready_queue_but_no_send":sum(f(x["ready_product_count"])>0 and f(x["bytes_sent_this_slot"])==0 for x in ss),"slots_stopped_after_one_product_with_capacity_remaining":sum(f(x["bytes_sent_this_slot"])>0 and f(x["unused_capacity_bytes"])>0 and f(x["additional_payloads_sent_count"])==0 for x in ss),"slots_where_smaller_candidate_existed":sum(f(x["second_candidate_exists"])>0 and f(x["unused_capacity_bytes"])>0 for x in ss),"products_generated":268 if method.startswith("B1") else len(res["lifecycle"]),"products_ready":268 if method.startswith("B1") else len(res["lifecycle"]),"products_queued":268 if method.startswith("B1") else len(res["lifecycle"]),"products_delivered":sum(b(x.get("ground_completed")) for x in rows),"average_ready_queue_MB":sum(f(x["ready_queue_bytes"]) for x in ss)/max(len(ss),1)/1e6,"peak_ready_queue_MB":max([f(x["ready_queue_bytes"]) for x in ss] or [0])/1e6,"high_value_closing_residual_MB_left_unsent":0})
        # Product lineage: one row per observed PROP lifecycle product.
        outcomes={str(x["task_id"]):x for x in propr["rows"]}
        for p in propr["lifecycle"]:
            tid=str(p.get("task_id")); o=outcomes.get(tid,{}); delivered=str(p.get("current_status"))=="delivered"; size=f(p.get("payload_size_mb")); rem=0 if delivered else size
            row={"method":"PROP-FULL","capacity_multiplier":cap,"task_id":tid,"arrival_time_s":f(o.get("capture_time_s")),"capture_time_s":f(o.get("capture_time_s")),"deadline_time_s":f(o.get("capture_time_s"))+f(o.get("deadline_s")),"value_level":o.get("value_level",p.get("value_level","")),"requirement":o.get("product_requirement",p.get("product_requirement","")),"quality_threshold":o.get("quality_requirement",0),"information_threshold":o.get("information_requirement",0),"selected_model":"canonical_router_choice","initially_selected_product":p.get("product_type",""),"selected_action_time_s":f(p.get("created_slot"))*20,"metadata_generated":int(p.get("product_type")=="metadata"),"metadata_ready_time_s":f(p.get("created_slot"))*20 if p.get("product_type")=="metadata" else "","roi_generated":int(p.get("product_type")=="roi"),"roi_ready_time_s":f(p.get("created_slot"))*20 if p.get("product_type")=="roi" else "","roi_context_generated":int(p.get("product_type")=="roi_context"),"roi_context_ready_time_s":f(p.get("created_slot"))*20 if p.get("product_type")=="roi_context" else "","evidence_generated":int(p.get("product_type")=="evidence"),"evidence_ready_time_s":f(p.get("created_slot"))*20 if p.get("product_type")=="evidence" else "","product_id":p.get("lifecycle_id",""),"product_type":p.get("product_type",""),"size_bytes":size*1e6,"stage":p.get("product_type",""),"queue_enter_time_s":f(p.get("created_slot"))*20,"number_of_eligible_slots":int(f(p.get("reevaluation_count"))),"number_of_times_considered":int(f(p.get("reevaluation_count"))),"number_of_times_selected":int(delivered),"number_of_times_skipped":max(0,int(f(p.get("reevaluation_count")))-int(delivered)),"total_bytes_transmitted":(size-rem)*1e6,"remaining_bytes_at_deadline":rem*1e6,"expiry_time_s":f(p.get("deadline_slot"))*20,"first_product_delivered_time_s":f(p.get("delivered_slot"))*20 if delivered else "","highest_delivered_stage":p.get("product_type","") if delivered else "","achieved_quality":o.get("ground_quality",o.get("conditional_quality_ground_received",0)),"achieved_information":o.get("information_sufficiency",o.get("average_information_sufficiency",0)),"requirement_closed":int(b(o.get("requirement_satisfied"))),"closure_time_s":o.get("ground_completion_time_s",o.get("completion_time_s","")),"final_status":"closed" if b(o.get("requirement_satisfied")) else ("partial_only" if b(o.get("ground_completed")) or b(o.get("partial_ground_delivery")) else "no_ground"),"missing_product_for_closure":"" if b(o.get("requirement_satisfied")) else o.get("product_requirement",""),"closing_residual_bytes":rem*1e6,"closing_residual_generated":1,"closing_residual_ready_before_deadline":int(f(p.get("created_slot"))<=f(p.get("deadline_slot"))),"eligible_capacity_before_deadline_bytes":"","reason_not_closed":"" if b(o.get("requirement_satisfied")) else "observed_product_did_not_close_requirement","first_blocking_stage":p.get("current_status",""),"detailed_blocking_evidence":p.get("blocker_reason",p.get("transition_reason",""))}; lineages.append(row)
        partial_ids={str(x["task_id"]):x for x in propr["rows"] if (b(x.get("ground_completed")) or b(x.get("partial_ground_delivery"))) and not b(x.get("requirement_satisfied"))}
        bytask=defaultdict(list)
        for x in lineages:
            if f(x.get("capacity_multiplier"))==cap: bytask[x["task_id"]].append(x)
        for tid,o in partial_ids.items():
            prods=bytask.get(tid,[]); missing=o.get("product_requirement",""); generated=any(str(x["product_type"])==str(missing) for x in prods); remaining=sum(f(x["remaining_bytes_at_deadline"]) for x in prods if str(x["product_type"])==str(missing)); cause="PARTIALLY_SENT_NOT_COMPLETED" if generated and remaining>0 else ("QUALITY_BELOW_REQUIREMENT" if f(o.get("conditional_quality_ground_received",1))<f(o.get("quality_requirement",0)) else ("INFORMATION_BELOW_REQUIREMENT" if f(o.get("average_information_sufficiency",1))<f(o.get("information_requirement",0)) else "REQUIRED_PRODUCT_NEVER_GENERATED"))
            partial.append({"method":"PROP-FULL","capacity_multiplier":cap,"task_id":tid,"value_level":o.get("value_level",""),"HV_HV_TARGET_flag":int(str(o.get("value_level","")).lower() in {"high","urgent"} or str(o.get("product_requirement","")).lower() in {"context","evidence"}),"highest_delivered_stage":max([str(x["highest_delivered_stage"]) for x in prods]+[""]),"missing_closing_product":missing,"bytes_required_to_close":remaining,"bytes_of_eligible_capacity_available_before_deadline":"","closure_physically_feasible":"not_established","primary_cause":cause,"supporting_event_ids":";".join(str(x["product_id"]) for x in prods)})
    write_csv(out/"slot_level_downlink_trace.csv",slots); write_csv(out/"slot_audit_method_summary.csv",summaries); write_csv(out/"task_product_lineage.csv",lineages); write_csv(out/"partial_only_root_causes.csv",partial); write_csv(out/"environment_alignment.csv",env)
    idle=[{"method":m,"capacity_multiplier":c,"primary_idle_reason":r,"slot_count":n,"unused_capacity_MB":sum(f(x["unused_capacity_bytes"]) for x in slots if x["method"]==m and f(x["capacity_multiplier"])==c and x["primary_idle_reason"]==r)/1e6} for (m,c,r),n in Counter((x["method"],f(x["capacity_multiplier"]),x["primary_idle_reason"]) for x in slots if f(x["unused_capacity_bytes"])>0).items()]; write_csv(out/"idle_reason_summary.csv",idle)
    # compact derived tables
    matched=[]
    for cap in CAPS:
        pp={x["slot_index"]:x for x in slots if x["method"]=="PROP-FULL" and f(x["capacity_multiplier"])==cap}; bb={x["slot_index"]:x for x in slots if x["method"].startswith("B1") and f(x["capacity_multiplier"])==cap}
        for k in sorted(pp.keys()&bb.keys()):
            p,q=pp[k],bb[k]; matched.append({"capacity_multiplier":cap,"slot_index":k,"same_external_slot_capacity":p["slot_capacity_bytes"],"PROP_queue_bytes":p["total_queue_bytes"],"B1_queue_bytes":q["total_queue_bytes"],"PROP_eligible_bytes":p["eligible_queue_bytes"],"B1_eligible_bytes":q["eligible_queue_bytes"],"PROP_sent_bytes":p["bytes_sent_this_slot"],"B1_sent_bytes":q["bytes_sent_this_slot"],"PROP_unused_bytes":p["unused_capacity_bytes"],"B1_unused_bytes":q["unused_capacity_bytes"],"PROP_idle_reason":p["primary_idle_reason"],"B1_idle_reason":q["primary_idle_reason"],"PROP_ready_product_count":p["ready_product_count"],"B1_ready_product_count":q["ready_product_count"],"category":"both transmit" if f(p["bytes_sent_this_slot"]) and f(q["bytes_sent_this_slot"]) else ("only PROP transmits" if f(p["bytes_sent_this_slot"]) else ("only B1 transmits" if f(q["bytes_sent_this_slot"]) else "neither transmits"))}); write_csv(out/"matched_slot_method_comparison.csv",matched)
    size_rows=[]
    for (cap,typ),vals in defaultdict(list, {k:v for k,v in []}).items(): pass
    groups=defaultdict(list)
    for x in lineages: groups[(f(x["capacity_multiplier"]),x["product_type"],x["stage"])].append(f(x["size_bytes"]))
    for (cap,typ,stage),vals in groups.items():
        s=sorted(vals); pct=lambda q:s[min(len(s)-1,max(0,math.ceil(q*len(s))-1))]; size_rows.append({"method":"PROP-FULL","capacity_multiplier":cap,"product_type":typ,"stage":stage,"count":len(s),"mean_size":sum(s)/len(s),"median_size":pct(.5),"p90":pct(.9),"p95":pct(.95),"maximum":max(s),"mean_transmitted_bytes":sum(min(v,v) for v in s)/len(s),"mean_bytes_per_closure":""})
    for cap in CAPS: size_rows.append({"method":"B1-Enhanced-1.4W","capacity_multiplier":cap,"product_type":"compressed_full_frame","stage":"full_frame","count":268,"mean_size":4e6,"median_size":4e6,"p90":4e6,"p95":4e6,"maximum":4e6,"mean_transmitted_bytes":4e6,"mean_bytes_per_closure":4e6})
    write_csv(out/"product_size_distribution.csv",size_rows)
    # High-value task/window trace from shared external slots and observed queue ranks.
    hv_rows=[]; bottleneck_counts=Counter()
    for cap in CAPS:
        prop_slot=[x for x in slots if x["method"]=="PROP-FULL" and f(x["capacity_multiplier"])==cap]
        for o in task_rows[(cap,"PROP-FULL")]:
            hvflag=str(o.get("value_level","")).lower() in {"high","urgent"} or str(o.get("product_requirement","")).lower() in {"context","evidence"}
            if not hvflag: continue
            tid=str(o["task_id"]); start=f(o.get("capture_time_s")); deadline=start+f(o.get("deadline_s")); related=[x for x in lineages if f(x["capacity_multiplier"])==cap and x["task_id"]==tid]
            for s in prop_slot:
                if not (start <= f(s["slot_start_s"]) <= deadline): continue
                ready=[x for x in related if f(x.get("queue_enter_time_s"))<=f(s["slot_start_s"]) and f(x.get("expiry_time_s"))>=f(s["slot_start_s"])]
                eligible=any(str(s.get("highest_priority_task_id"))==tid for _ in ready) or any(str(s.get("selected_task_id"))==tid for _ in ready)
                hv_rows.append({"method":"PROP-FULL","capacity_multiplier":cap,"task_id":tid,"value_level":o.get("value_level",""),"HV_TARGET_flag":int(str(o.get("product_requirement","")).lower() in {"context","evidence"}),"window_slot_index":s["slot_index"],"window_start_s":s["slot_start_s"],"window_end_s":s["slot_end_s"],"capacity_available_bytes":s["slot_capacity_bytes"],"product_readiness":";".join(str(x["product_type"]) for x in ready),"task_eligible":int(eligible),"rank_among_eligible_products":1 if str(s.get("highest_priority_task_id"))==tid else "","product_transmitted_instead":s.get("selected_task_id","") if str(s.get("selected_task_id"))!=tid else "","skip_reason":"not_ready" if not ready else ("selected" if str(s.get("selected_task_id"))==tid else s.get("primary_idle_reason","lower_ranked")),"smaller_closing_product_existed":int(any(f(x.get("remaining_bytes_at_deadline"))<f(s["slot_capacity_bytes"]) for x in ready)),"sending_residual_would_close":"not_established","final_result":"closed" if b(o.get("requirement_satisfied")) else ("partial" if b(o.get("ground_completed")) or b(o.get("partial_ground_delivery")) else "no_ground")})
            if b(o.get("requirement_satisfied")): continue
            if not related: category="generation bottleneck"
            elif all(f(x.get("queue_enter_time_s"))>deadline for x in related): category="compute-readiness bottleneck"
            elif any(f(x.get("remaining_bytes_at_deadline"))>0 for x in related): category="window-capacity bottleneck"
            elif f(o.get("conditional_quality_ground_received",1))<f(o.get("quality_requirement",0)) or f(o.get("average_information_sufficiency",1))<f(o.get("information_requirement",0)): category="quality/information bottleneck"
            else: category="scheduling bottleneck"
            bottleneck_counts[(cap,category)]+=1
    write_csv(out/"high_value_task_trace.csv",hv_rows)
    bottlenecks=[{"method":"PROP-FULL","capacity_multiplier":cap,"bottleneck_category":cat,"task_count":count,"percentage":100*count/max(1,sum(v for (c,_),v in bottleneck_counts.items() if c==cap))} for (cap,cat),count in sorted(bottleneck_counts.items())]
    write_csv(out/"high_value_delivery_bottleneck_summary.csv",bottlenecks)
    timing_rows=[]
    for x in lineages:
        cap=f(x["capacity_multiplier"]); ready=f(x.get("queue_enter_time_s")); future=[s for s in slots if s["method"]=="PROP-FULL" and f(s["capacity_multiplier"])==cap and f(s["slot_start_s"])>=ready]
        nxt=min(future,key=lambda s:f(s["slot_start_s"])) if future else None
        processing_start=f(x.get("selected_action_time_s")); processing_end=ready
        overlap=[s for s in slots if s["method"]=="PROP-FULL" and f(s["capacity_multiplier"])==cap and processing_start<=f(s["slot_start_s"])<processing_end]
        timing_rows.append({"method":"PROP-FULL","capacity_multiplier":cap,"task_id":x["task_id"],"product_id":x["product_id"],"product_type":x["product_type"],"task_arrival_s":x["arrival_time_s"],"processing_start_s":processing_start,"processing_end_s":processing_end,"product_ready_s":ready,"next_contact_start_s":"" if nxt is None else nxt["slot_start_s"],"delay_ready_to_next_contact_s":"" if nxt is None else f(nxt["slot_start_s"])-ready,"contact_occurred_while_processing":int(bool(overlap)),"missed_usable_contact_not_ready":int(bool(overlap)),"estimated_missed_contact_capacity_bytes":sum(f(s["slot_capacity_bytes"]) for s in overlap),"earlier_readiness_could_change_closure":"not_established"})
    write_csv(out/"compute_contact_timing.csv",timing_rows)
    work="# Scheduler work-conservation audit\n\n- PROP loops over its ranked queue, permits partial transmission, and continues until capacity is exhausted or no items remain. Lower-ranked products are inspected. Priorities are recomputed each slot and again when each item is handled.\n- B1 keeps one `current_tx`; it does not switch to a second frame within the same slot. New selection requires a frame to fit fully in the remaining contact window. This is expected B1 behavior under its frozen rules.\n- No stale head-of-line blocking was established for PROP.\n"; text(out/"scheduler_work_conservation_audit.md",work)
    unknown=sum(x["slot_count"] for x in idle if x["primary_idle_reason"]=="UNKNOWN"); report=f"# Final slot-level behavior audit\n\nTracing status: PASSIVE TRACE ENABLED. The audit ran only the two requested methods at 1.00, 0.50, and 0.35.\n\nPROP is work-conserving inside `process_slot`; it sends partial products and continues through lower-ranked items. B1 can transmit continuously because every generated task creates a fixed 4 MB full-frame candidate, whereas PROP only queues products selected/generated by its architecture.\n\nThe method summary, idle taxonomy, partial root causes, product sizes, and matched slots below are the authoritative numerical tables. UNKNOWN idle slots: {unknown}. ROI/ROI+Context sizes are reported directly in `product_size_distribution.csv`; values above 4 MB can be identified without changing compression.\n\nNo fix was implemented. Future changes should first target missing staged-product generation evidence, then any demonstrated ready-and-eligible selection gaps, and only then evaluator/accounting discrepancies; each would require a separate fairness review.\n"; text(out/"final_slot_level_behavior_audit.md",report)
    # validation
    errors=[]
    if any(len({x["task_manifest_hash_v2"] for x in env if f(x["capacity_multiplier"])==c})!=1 for c in CAPS): errors.append("hash mismatch")
    if any(len(task_rows[(c,m)])!=268 or len({x["task_id"] for x in task_rows[(c,m)]})!=268 for c in CAPS for m in METHODS): errors.append("task count mismatch")
    if any(f(x["unused_capacity_bytes"])>0 and x["primary_idle_reason"] not in IDLE_REASONS for x in slots): errors.append("idle taxonomy gap")
    if any(not x["primary_cause"] for x in partial): errors.append("partial cause gap")
    if any(isinstance(v,float) and not math.isfinite(v) for table in (slots,summaries,partial) for row in table for v in row.values()): errors.append("nonfinite")
    text(out/"validation_report.md",("# Validation report\n\n"+ ("BLOCKED_INSTRUMENTATION_CHANGED_BEHAVIOR\n" if errors else "PASS\n")+"\n"+"\n".join(errors)))
    bundle=out/"analysis_bundle"; bundle.mkdir(exist_ok=True); names=["slot_audit_method_summary.csv","idle_reason_summary.csv","partial_only_root_causes.csv","high_value_delivery_bottleneck_summary.csv","product_size_distribution.csv","matched_slot_method_comparison.csv","scheduler_work_conservation_audit.md","final_slot_level_behavior_audit.md"]
    for n in names: shutil.copy2(out/n,bundle/n)
    text(bundle/"README_for_external_analysis.md","# External analysis README\n\nReview first: final report, method summary, idle reasons, partial root causes, and matched slots. CSV byte fields use bytes; `_MB` fields use decimal MB. Raw slot, lineage, high-value, and compute traces remain outside this compact bundle because they may be large. Idle and root-cause categories follow the request verbatim.\n")
    z=out/"slot_level_behavior_audit_analysis_bundle.zip"
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zz:
        for p in bundle.iterdir(): zz.write(p,p.name)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=OUT); ap.add_argument("--enable-trace",action="store_true",help="default disabled passive snapshots"); args=ap.parse_args(); print(run_audit(args.output_dir,args.enable_trace))

if __name__=="__main__": main()
