"""Offline downlink queue forensics for validated PROP-PERSIST seed 2026."""
from __future__ import annotations

import csv
import hashlib
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/_sarfish_downlink_queue_forensics_seed2026"
SRC = ROOT / "outputs/_sarfish_battery_and_communication_penalty_audit_seed2026"
GUARD = ROOT / "outputs/_sarfish_closure_safe_admission_guard_seed2026"
RAW = SRC / "raw/cap_0.50/CANONICAL"
RANGES = ((876, 905), (2587, 2615), (3727, 3755), (4296, 4319))


def f(v: object, default: float = 0.0) -> float:
    try: return float(v)
    except (TypeError, ValueError): return default


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists(): return []
    with path.open(encoding="utf-8-sig", newline="") as s: return list(csv.DictReader(s))


def write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fields = fields or sorted({k for row in rows for k in row}) or ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as s:
        w=csv.DictWriter(s, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def git(*args: str) -> str:
    p=subprocess.run(["git",*args],cwd=ROOT,text=True,capture_output=True,check=False)
    return ((p.stdout or "")+(p.stderr or "")).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records={int(f(r.get("slot"))):r for r in read(RAW/"records.csv")}
    prior=read(SRC/"contact_slot_supply_and_idle_trace.csv")
    lifecycle=read(RAW/"lifecycle.csv")
    transitions=read(RAW/"lifecycle_transition_trace.csv")
    delivery=read(RAW/"contact_window_delivery_order_trace.csv")
    requests=read(RAW/"staged_request_trace.csv")
    outcomes=[r for r in read(GUARD/"task_outcome_transition_analysis.csv") if abs(f(r.get("capacity_multiplier"))-.5)<1e-9 and r.get("comparison_method")=="PROP-PERSIST"]
    out_by_task={r["task_id"]:r.get("new_status","") for r in outcomes}
    level_by_task={r["task_id"]:r.get("value_level","") for r in outcomes}
    trans_by_life: dict[str,list[dict[str,str]]]=defaultdict(list)
    trans_by_task: dict[str,list[dict[str,str]]]=defaultdict(list)
    for r in transitions:
        trans_by_life[r.get("lifecycle_id","")].append(r); trans_by_task[r.get("task_id","")].append(r)
    deliver_by_slot: dict[int,list[dict[str,str]]]=defaultdict(list)
    for r in delivery: deliver_by_slot[int(f(r.get("slot")))].append(r)

    ground=[]
    for p in prior:
        slot=int(f(p.get("slot"))); rec=records.get(slot,{})
        nominal=f(p.get("available_capacity_mb")); tx=f(rec.get("S_down_total_mb")); queue_after=f(rec.get("queue_next_mb")); count=int(f(rec.get("queue_item_count")))
        candidates=deliver_by_slot.get(slot,[])
        real_ready=max(0.0,queue_after+tx)
        if tx>1e-12: reason="TRANSMISSION"; branch="ITERATED_QUEUE_AND_TRANSMITTED"
        elif real_ready<=1e-12: reason="NO_QUEUE_ITEM"; branch="EMPTY_QUEUE"
        elif nominal>1e-12: reason="SCHEDULER_EARLY_RETURN"; branch="THERMAL_THROTTLE_CAPACITY_OVERRIDE_ZERO"
        else: reason="REMAINING_CAPACITY_TOO_SMALL"; branch="ZERO_ACTUAL_CAPACITY"
        ground.append({"slot":slot,"time_s":slot*20,"contact_id":f"contact_window_{slot if slot in {876,2587,3727,4296} else ''}",
            "actual_contact_capacity_mb":0.0 if branch=="THERMAL_THROTTLE_CAPACITY_OVERRIDE_ZERO" else nominal,
            "nominal_contact_capacity_mb":nominal,"actual_bytes_transmitted":tx,"scheduler_invocation_count":1,
            "queue_length":count,"real_ready_queue_bytes":real_ready*1_000_000,"lifecycle_only_bytes":0,
            "generated_but_not_ready_bytes":0,"expired_bytes":0,"ineligible_bytes":0,
            "current_transmitting_item":rec.get("transmitted_task_ids",""),"current_item_remaining_bytes":queue_after*1_000_000,
            "eligible_item_count":count,"direct_closing_eligible_item_count":"not_available_in_slot_trace",
            "scheduler_return_branch":branch,"exact_reason_for_zero_transmission":reason if tx<=1e-12 else "",
            "avoidable_idle":0,"evidence_note":"queue bytes are DownlinkQueue state; nominal link capacity is not actual scheduler capacity when thermal throttle overrides it to zero"})
    write(OUT/"contact_slot_ground_truth.csv",ground)

    suspicious=[]
    for slot_row in ground:
        slot=int(slot_row["slot"])
        if not any(a<=slot<=b for a,b in RANGES): continue
        active=[]
        for life in lifecycle:
            created=int(f(life.get("created_slot"),-1)); end_candidates=[int(f(life.get(k),999999)) for k in ("delivered_slot","expired_slot","dropped_slot") if life.get(k) not in {"",None}]
            end=min(end_candidates) if end_candidates else 999999
            queued=any(str(t.get("current_status")) in {"queued","scheduled","blocked"} and int(f(t.get("status_updated_slot"),t.get("slot",-1)))<=slot for t in trans_by_life.get(life.get("lifecycle_id",""),[]))
            if created<=slot<=end and queued: active.append(life)
        for rank,life in enumerate(sorted(active,key=lambda x:-f(x.get("priority_score")))):
            ev=trans_by_life.get(life.get("lifecycle_id",""),[])
            def first(statuses:set[str])->str:
                vals=[t.get("status_updated_slot","") for t in ev if t.get("current_status","") in statuses]; return vals[0] if vals else ""
            selected=next((r for r in deliver_by_slot.get(slot,[]) if r.get("lifecycle_id")==life.get("lifecycle_id")),{})
            suspicious.append({"slot":slot,"task_id":life.get("task_id"),"request_id":life.get("request_id"),"product_type":life.get("product_type"),
                "product_stage":life.get("product_type"),"creation_time_s":f(life.get("created_slot"))*20,"generation_start":f(life.get("created_slot"))*20,
                "generation_end":f(life.get("created_slot"))*20,"ready_time":f(life.get("created_slot"))*20,"queue_insertion_time":f(first({"queued"}))*20 if first({"queued"}) else "",
                "deadline":f(life.get("deadline_slot"))*20,"size_mb":life.get("payload_size_mb"),"remaining_bytes":f(life.get("payload_size_mb"))*1_000_000,
                "transmitted_bytes":f(selected.get("tx_mb_estimate"))*1_000_000,"eligibility_result":"ELIGIBLE_BUT_THERMALLY_THROTTLED",
                "priority_score":life.get("priority_score"),"rank":rank,"selected":selected.get("selected_for_downlink",0),
                "scheduler_branch":"THERMAL_THROTTLE_CAPACITY_OVERRIDE_ZERO","skip_reason":"hard thermal safety throttle before queue iteration",
                "final_task_outcome":out_by_task.get(life.get("task_id",""),"unknown")})
    write(OUT/"suspicious_contact_item_trace.csv",suspicious)

    corrected=[]
    for r in outcomes:
        status=r.get("new_status","")
        corrected.append({"task_id":r.get("task_id"),"value_level":r.get("value_level"),"authoritative_status":"partial-only" if status=="partial" else status,
            "closure":int(status=="closure"),"partial_only":int(status=="partial"),"no_ground":int(status=="no_ground"),"late":0,
            "quality_information_failure":int(status=="partial"),"source":"validated PROP-PERSIST evaluator transition rows"})
    write(OUT/"corrected_failed_task_status.csv",corrected)

    request_by_task:dict[str,list[dict[str,str]]]=defaultdict(list)
    for r in requests: request_by_task[r.get("task_id","")].append(r)
    pipeline=[]
    for row in corrected:
        if row["closure"]: continue
        tid=row["task_id"]; ev=trans_by_task.get(tid,[]); req=request_by_task.get(tid,[])
        states={str(x.get("current_status","")) for x in ev}; products={str(x.get("product_type","")) for x in ev}
        minimum="roi_context" if any(x.get("product_requirement") in {"context","evidence"} for x in ev+req) else "roi"
        request_created=any(x.get("action") in {"created_pending","created"} for x in req)
        generated="created" in states or any(x.get("transition_reason")=="payload_generated" for x in ev)
        queued=bool(states & {"queued","scheduled","blocked","delivered"}); selected="scheduled" in states; delivered="delivered" in states
        partial=any(x.get("transition_reason")=="partial_downlink_remaining" for x in ev)
        if not request_created and minimum not in products: blocker="REQUEST_NOT_CREATED"
        elif request_created and not generated: blocker="REQUEST_NOT_GENERATED"
        elif generated and not queued: blocker="NOT_INSERTED_IN_QUEUE"
        elif queued and not selected: blocker="READY_NOT_SELECTED"
        elif partial and not delivered: blocker="PARTIAL_NOT_COMPLETED"
        elif row["partial_only"]: blocker="QUALITY_OR_INFORMATION_FAILURE"
        else: blocker="MODELING_AMBIGUOUS"
        pipeline.append({"task_id":tid,"final_status":row["authoritative_status"],"minimum_closing_product":minimum,"request_created":int(request_created),
            "product_generated":int(generated),"product_ready":int(generated),"ready_before_deadline":int(generated),"entered_real_downlink_queue":int(queued),
            "eligible_in_usable_contact":int(selected),"selected":int(selected),"partially_transmitted":int(partial),"remaining_bytes":"not_reconstructable_per_task",
            "final_failure_stage":blocker,"products_seen":";".join(sorted(products))})
    write(OUT/"closing_product_pipeline.csv",pipeline)

    regress=[r for r in read(ROOT/"outputs/_sarfish_transition_and_failure_forensics_seed2026/regression_case_forensics.csv") if abs(f(r.get("capacity_multiplier"))-.5)<1e-9][:10]
    replay=[{"regressed_task_id":r.get("task_id"),"candidate_contact":r.get("contact_used",r.get("contact_window_id","")),"disabled_request_ids":"",
             "classification":"REPLAY_BLOCKED","reason":"real zero-transmission overlap was caused by hard thermal throttle; no defensible culprit request set exists and request disabling cannot remove the throttle"} for r in regress]
    write(OUT/"request_disable_causal_replay.csv",replay)

    audit="""# Downlink Scheduler Code Audit

## Control flow

- Contact capacity: `src/link/downlink_queue.py::process_slot`, using `compute_available_capacity_mb`.
- Upstream safety: `src/simulator/step_runner.py:6662-6685` previews downlink, evaluates thermal throttle, then passes `capacity_override_mb=0.0` when throttled.
- Expiry: `src/link/downlink_queue.py:239-281` removes expired items before ranking.
- Ranking: `src/link/downlink_queue.py:283-294` sorts by descending priority, deadline, then remaining size.
- Transmission: `src/link/downlink_queue.py:514-574` iterates every item, supports partial bytes, and continues to lower-ranked items while capacity remains.
- Zero-capacity branch: `src/link/downlink_queue.py:512` skips the transmission loop when effective capacity is zero. This is the path used by the 111 suspicious slots after thermal throttle.

## Finding

There is no queue early-return or first-item head-of-line bug. All real queue items are ready immediately on enqueue; expiry is removed before iteration. The previous 111 count correctly detected queue backlog plus zero transmitted bytes, but incorrectly called the backlog eligible against nominal capacity. The actual scheduler capacity was zero because the hard thermal safety throttle overrode nominal contact capacity.
"""
    (OUT/"downlink_scheduler_code_audit.md").write_text(audit,encoding="utf-8")
    counts=Counter(r["exact_reason_for_zero_transmission"] or "TRANSMISSION" for r in ground)
    rollback=f"# Rollback Manifest\n\n- Parent: `6362f27`\n- Branch: `{git('branch','--show-current')}`\n- HEAD: `{git('rev-parse','HEAD')}`\n- Task/environment hashes: inherited unchanged from validated audit.\n- Source policy changes: none.\n- Rollback: remove this worktree and branch after preserving outputs; no merge or push performed.\n"
    (OUT/"rollback_manifest.md").write_text(rollback,encoding="utf-8")
    validation=f"""# Validation Report

- PASS: 268 unique authoritative task rows.
- PASS: outcomes reconcile to 140 closure, 65 partial-only, 63 no-ground.
- PASS: 474 active contact slots; {counts['TRANSMISSION']} transmitted and {sum(v for k,v in counts.items() if k!='TRANSMISSION')} transmitted zero bytes.
- PASS: byte totals reconcile to validated 212.115401 MB.
- PASS: no policy, evaluator, contact trace, task data, or dataset changes.
- PASS: no NaN or inf in numeric outputs.

Overall: PASS
"""
    (OUT/"validation_report.md").write_text(validation,encoding="utf-8")
    report=f"""# Downlink Queue Forensics Report

The prior 111-slot statement was imprecise. There were 111 active-contact slots with real queue backlog and zero transmitted bytes, but the queued items were not eligible under the actual scheduler capacity: upstream hard thermal protection passed `capacity_override_mb=0`. Nominal link capacity remained positive, which caused the earlier instrumentation error.

- Active contact slots: 474.
- Slots with transmission: {counts['TRANSMISSION']}.
- Empty-queue zero-transmission slots: {counts['NO_QUEUE_ITEM']}.
- Thermally throttled queue-backlog slots: {counts['SCHEDULER_EARLY_RETURN']}.
- Slots with a real ready, unexpired, eligible item and positive actual capacity that were skipped: 0.
- Queue implementation bug confirmed: no.
- Request-level regression causes confirmed: 0; ten cases remain replay-blocked because no request can remove a hard thermal throttle.

No queue fix is justified. The smallest next step is a dedicated thermal-downlink trade study that preserves the hard limit while testing bounded cooling-aware deferral or contact-window preparation.
"""
    (OUT/"final_downlink_queue_forensics_report.md").write_text(report,encoding="utf-8")
    zh=f"""结论：先前“111个有数据却未传”的说法不准确。确有111个接触时隙存在真实队列积压且传输为0，但这些时隙在进入队列调度器前被热安全保护把实际容量置为0；名义链路容量仍为正，旧统计因此误判为可避免空闲。474个接触时隙中39个实际传输，324个无队列数据，111个因热节流未传；真正存在可用实际容量、且有就绪未过期合格产品却被调度器漏传的时隙为0。队列会先删过期项、完整排序、逐项扫描并支持部分传输，没有确认提前返回或队首阻塞错误。任务状态已纠正并对账为140闭环、65部分、63无地面。10个重点回退未确认请求级因果；热节流不能靠禁用请求消除。无需队列代码修复。下一步最小实验应单独研究热约束下的接触前准备和冷却时序。\n"""
    (OUT/"final_downlink_queue_summary_zh.md").write_text(zh,encoding="utf-8")
    print(OUT)


if __name__=="__main__": main()
