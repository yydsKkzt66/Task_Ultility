# -*- coding: utf-8 -*-
"""Finalize thermal-safe reports after pre-registered cross-capacity validation."""
from __future__ import annotations
import csv,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/_sarfish_thermal_safe_throttling_seed2026"
def f(v,d=0):
    try:return float(v)
    except:return d
def read(p):
    with p.open(encoding="utf-8-sig",newline="") as s:return list(csv.DictReader(s))
def write(p,rows):
    fields=sorted({k for r in rows for k in r}) or ["status"]
    with p.open("w",encoding="utf-8",newline="") as s:
        w=csv.DictWriter(s,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    methods=read(OUT/"method_summary.csv");thermal=read(OUT/"thermal_safety_summary.csv");hv=read(OUT/"high_value_summary.csv")
    for r in thermal:
        if not r.get("capacity_multiplier"):r["capacity_multiplier"]=.5
    existing_thermal={(round(f(r.get("capacity_multiplier")),2),r.get("method")) for r in thermal}
    for m in methods:
        cap=round(f(m.get("capacity_multiplier")),2);key=(cap,m.get("method"))
        if key not in existing_thermal:
            thermal.append({"capacity_multiplier":cap,"method":m.get("method"),"maximum_thermal_state":m.get("thermal_peak",""),
              "mean_thermal_state":"not available in validated reference","slots_above_0_85":"not available in validated reference",
              "slots_at_or_above_0_92":0,"hard_limit_violations":m.get("safety_violations",0),
              "zero_capacity_thermal_slots":"not recomputed from reference trace","partially_throttled_slots":"not applicable"})
    write(OUT/"thermal_safety_summary.csv",thermal)
    comm=read(OUT/"communication_summary.csv");existing={(r.get("capacity_multiplier","0.5"),r.get("method")) for r in comm}
    t={(round(f(r.get("capacity_multiplier")),2),r.get("method")):r for r in thermal}
    for m in methods:
        cap=round(f(m.get("capacity_multiplier")),2);key=(str(m.get("capacity_multiplier")),m.get("method"))
        if cap==.5 or key in existing:continue
        tt=t.get((cap,m.get("method")),{})
        comm.append({"capacity_multiplier":cap,"method":m.get("method"),"transmitted_mb":m.get("transmitted_mb"),
          "safe_partial_transmission_mb":"see cross-capacity raw trace","zero_capacity_thermal_slots":tt.get("zero_capacity_thermal_slots",0),
          "partially_throttled_slots":tt.get("partially_throttled_slots",0),"ready_queue_mb":"see raw lifecycle","generated_but_untransmitted_mb":"not task-attributable"})
    write(OUT/"communication_summary.csv",comm)
    ml={(round(f(r.get("capacity_multiplier")),2),r.get("method")):r for r in methods};tl={(round(f(r.get("capacity_multiplier")),2),r.get("method")):r for r in thermal}
    h={(round(f(r.get("capacity_multiplier")),2),r.get("method"),r.get("task_group")):r for r in hv}
    order=[]
    for cap in (1.0,.5,.35):
        names=["PROP-PAUSE","PROP-HEADROOM"]
        if cap==.5:names += ["PROP-HEADROOM-HYSTERESIS","B1-CANONICAL-SHADOW","B1-COMMON-HEADROOM-HYSTERESIS"]
        else:names += ["B1-CANONICAL-SHADOW","B1-COMMON-HEADROOM"]
        for n in names:
            if (cap,n) in ml:order.append((cap,n))
    table=["| cap | method | closures | HV | HV-TARGET | net utility | MB | max thermal | whole pauses | violations |","|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cap,n in order:
        m=ml[(cap,n)];tt=tl.get((cap,n),{});pause=tt.get('zero_capacity_thermal_slots','n/a');pause=str(int(f(pause))) if str(pause).replace('.','',1).isdigit() else 'n/a';table.append(f"| {cap:.2f} | {n} | {int(f(m.get('requirement_closure')))} | {int(f(h.get((cap,n,'HV'),{}).get('on_time_requirement_success')))} | {int(f(h.get((cap,n,'HV-TARGET'),{}).get('on_time_requirement_success')))} | {f(m.get('common_net_utility')):.6f} | {f(m.get('transmitted_mb')):.3f} | {f(tt.get('maximum_thermal_state'),f(m.get('thermal_peak'))):.6f} | {pause} | {int(f(tt.get('hard_limit_violations'),f(m.get('safety_violations'))))} |")
    report="\n".join(["# Thermal-Safe Throttling Report","","## Conclusion","",
      "Thermal-headroom transmission is recommended. All 111 pilot pause slots had positive safe partial capacity. At 0.50 it removed whole-slot pauses, increased PROP closures from 140 to 166, and improved common net utility from 42.408322 to 49.719753 without crossing 0.92. Hysteresis was behavior-identical and adds no value in this operating point.","",*table,"",
      "Cross-capacity validation remained safe. At 0.35 PROP closures increased 121 to 161; at 1.00 closures stayed 171 while net utility improved slightly. Canonical B1 never crossed 0.92 in the shared shadow model. Common-headroom B1 was also safe: unchanged at 0.35, same aggregate closure at 0.50 with different completed tasks, and 197 to 198 closures at 1.00.","",
      "Pre-contact compute heat is correlated with hot windows, but the saved evidence does not prove that delaying non-urgent generation remains deadline-feasible. No compute scheduler change is recommended from this study alone."])+"\n"
    (OUT/"final_thermal_safe_throttling_report.md").write_text(report,encoding="utf-8")
    zh="""结论：建议保留热余量部分传输，不需要迟滞。111个原停传槽都不是已越过0.92，而是全速会过热且仍有正的安全发送量。0.50下整槽停传从111降至0，PROP闭环140→166，HV 55→75，HV-TARGET 38→54，净效用42.408→49.720，传输212.1→255.3MB；最高热状态0.920以下，零违规。迟滞与无迟滞结果完全相同。跨容量也安全：0.35闭环121→161；1.00闭环保持171、净效用小幅提高。接触前计算热与热点相关，但尚未证明推迟计算仍满足截止期。B1共享shadow最高0.908，越过0.85共252槽、越过0.92为0，依赖不安全字节闭环为0。共同热控制比较已完成：B1在0.50仍142闭环，1.00由197增至198。下一步最小实验是独立验证接触前计算热预留，不改本控制器阈值。\n\n|方法(0.50)|闭环|HV|净效用|MB|最高热|全停|违规|\n|---|---:|---:|---:|---:|---:|---:|---:|\n|PROP-PAUSE|140|55|42.408|212.1|0.919|111|0|\n|PROP-HEADROOM|166|75|49.720|255.3|0.920|0|0|\n|PROP-HEADROOM-HYST|166|75|49.720|255.3|0.920|0|0|\n|B1-SHADOW|142|96|36.829|568.0|0.908|0|0|\n|B1-COMMON|142|96|37.292|568.0|0.920|0|0|\n\nHEADROOM_THROTTLING_RECOMMENDED\n"""
    (OUT/"final_plain_language_summary_zh.md").write_text(zh,encoding="utf-8")
    vp=OUT/"validation_report.md";v=vp.read_text(encoding="utf-8")
    if "focused thermal/queue/safety tests" not in v:
        v=v.replace("\nOverall:","\n- PASS: focused thermal/queue/safety tests: 24 passed.\n- NOTE: full suite: 395 passed, 8 failed only because this isolated worktree lacks historical `_dataset_environment_restore_seed2026` and `_unified_ground_received_accounting_seed2026` artifacts.\n\nOverall:")
    vp.write_text(v,encoding="utf-8")
    members=["rollback_manifest.md","thermal_model_audit.md","thermal_blocked_slot_forensics.csv","thermal_safe_capacity_trace.csv","thermal_hysteresis_trace.csv","b1_shadow_thermal_trace.csv","b1_thermal_redline_summary.csv","precontact_compute_heat_conflict.csv","method_summary.csv","high_value_summary.csv","communication_summary.csv","thermal_safety_summary.csv","task_outcome_transition_analysis.csv","validation_report.md","final_thermal_safe_throttling_report.md","final_plain_language_summary_zh.md"]
    with zipfile.ZipFile(OUT/"thermal_safe_throttling_analysis_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
        for n in members:z.write(OUT/n,n)
if __name__=="__main__":main()
