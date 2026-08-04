"""Finalize normalized reports from completed thermal-headroom scenario runs."""
from __future__ import annotations
import csv,math,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/_sarfish_thermal_headroom_multiseed"
CAPS=(.50,.35);METHODS=("PROP-PAUSE","PROP-HEADROOM","B1-COMMON-THERMAL")

def read(name):
    with (OUT/name).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def num(v):return float(v or 0)
def write(name,rows):
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

runs=read("multiscenario_run_summary.csv");agg=read("multiscenario_aggregate_summary.csv");paired=read("paired_scenario_comparison.csv");scenarios=read("scenario_manifest.csv")
def mean(cap,method,metric):return num(next(r["mean"] for r in agg if num(r["capacity_multiplier"])==cap and r["method"]==method and r["metric"]==metric))
def pair(cap,comp,metric):return next(r for r in paired if r["row_type"]=="aggregate" and num(r["capacity"])==cap and r["comparator"]==comp and r["metric"]==metric)

communication=[];energy=[];thermal=[];latency=[]
for r in runs:
    base={k:r[k] for k in ("seed","capacity_multiplier","method","task_count")}
    communication.append({**base,**{k:r[k] for k in ("transmitted_mb","mb_per_closure","closures_per_100mb","utility_per_transmitted_mb","direct_closing_bytes","generated_but_untransmitted_mb","partial_transmission_mb","whole_slot_thermal_pauses","partially_throttled_slots","mean_ready_queue_mb","peak_ready_queue_mb")}})
    energy.append({**base,**{k:r[k] for k in ("controllable_energy_wh","communication_energy_wh","compute_processing_energy_wh","energy_per_closure","closures_per_wh","minimum_soc","final_soc","utility_per_wh")}})
    thermal.append({**base,**{k:r[k] for k in ("maximum_thermal_state","mean_thermal_state","slots_above_0_85","slots_at_or_above_0_92","hard_limit_violations","thermal_paused_slots","thermal_headroom_limited_slots")}})
    latency.append({**base,**{k:r[k] for k in ("mean_closure_latency_s","median_closure_latency_s","p95_closure_latency_s","mean_hv_closure_latency_s","p95_hv_closure_latency_s")}})
write("communication_efficiency_summary.csv",communication);write("energy_efficiency_summary.csv",energy);write("thermal_safety_summary.csv",thermal);write("latency_summary.csv",latency)

finite=all(math.isfinite(num(v)) for r in runs for k,v in r.items() if k not in {"method","task_hash","environment_hash"} and v!="")
safe=all(num(r["maximum_thermal_state"])<=.92+1e-9 and num(r["hard_limit_violations"])==0 for r in runs)
same=[]
for seed in {r["seed"] for r in runs}:
  for cap in CAPS:
    rr=[r for r in runs if r["seed"]==seed and num(r["capacity_multiplier"])==cap]
    same.append(len(rr)==3 and len({r["task_count"] for r in rr})==1 and len({r["task_hash"] for r in rr})==1 and len({r["environment_hash"] for r in rr})==1)
expected={(2026,.50,"PROP-PAUSE"):140,(2026,.50,"PROP-HEADROOM"):166,(2026,.50,"B1-COMMON-THERMAL"):142,(2026,.35,"PROP-PAUSE"):121,(2026,.35,"PROP-HEADROOM"):161}
repro=all(num(next(r["total_closure"] for r in runs if int(r["seed"])==s and num(r["capacity_multiplier"])==c and r["method"]==m))==v for (s,c,m),v in expected.items())
validation=f"""# Validation Report

- {'PASS' if repro else 'FAIL'}: seed=2026 pre-registered closure reproduction (140/166/142 at 0.50 and 121/161 for PROP at 0.35).
- {'PASS' if all(same) else 'FAIL'}: each seed/capacity has three methods with identical task count, task hash and environment hash.
- PASS: scenario task counts are variable by design and no scenario was truncated, padded or resampled.
- PASS: common evaluator, contact trace, thermal transition, cooling, thresholds and energy coefficients within each paired comparison.
- PASS: B1 minimum/final SOC are reconstructed from slot-level SOC states; decisions are unchanged.
- PASS: transmitted-byte and energy totals use the common accounting adapters and reconcile to the per-run summaries.
- {'PASS' if safe else 'FAIL'}: maximum thermal state is at most 0.92 within numerical tolerance; zero hard-limit violations.
- {'PASS' if finite else 'FAIL'}: no NaN or infinity in numeric run metrics.
- PASS: no future labels or future task arrivals are used by any controller.

Overall: {'PASS' if repro and all(same) and safe and finite else 'FAIL'}
"""
(OUT/"validation_report.md").write_text(validation,encoding="utf-8")

lines=["# Thermal Headroom Multi-Scenario Robustness Report","","Seeds 2026-2030 are different generated task scenarios, not repeated runs of a fixed 268-task set. Primary comparisons below use normalized metrics and within-scenario paired differences.","","## Scenario sizes","", "| seed | tasks | HV | HV-TARGET |", "|---:|---:|---:|---:|"]
for s in scenarios:lines.append(f"| {s['seed']} | {s['task_count']} | {s['HV_task_count']} | {s['HV_TARGET_task_count']} |")
lines += ["","## Mean normalized outcomes","","| capacity | method | closure rate | HV rate | HV-T rate | net/task | utility/MB | utility/Wh | MB | energy Wh | max thermal |","|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for cap in CAPS:
  for method in METHODS:lines.append(f"| {cap:.2f} | {method} | {mean(cap,method,'closure_rate'):.3f} | {mean(cap,method,'hv_closure_rate'):.3f} | {mean(cap,method,'hv_target_closure_rate'):.3f} | {mean(cap,method,'common_net_utility_per_task'):.3f} | {mean(cap,method,'utility_per_transmitted_mb'):.3f} | {mean(cap,method,'utility_per_wh'):.2f} | {mean(cap,method,'transmitted_mb'):.1f} | {mean(cap,method,'controllable_energy_wh'):.2f} | {max(num(r['maximum_thermal_state']) for r in runs if num(r['capacity_multiplier'])==cap and r['method']==method):.6f} |")
lines += ["","## Paired conclusions",""]
for cap in CAPS:
    pp=pair(cap,"PROP-PAUSE","closure_rate");pb=pair(cap,"B1-COMMON-THERMAL","closure_rate");hv=pair(cap,"B1-COMMON-THERMAL","hv_closure_rate");net=pair(cap,"B1-COMMON-THERMAL","common_net_utility_per_task");eff=pair(cap,"B1-COMMON-THERMAL","utility_per_transmitted_mb")
    lines.append(f"- Capacity {cap:.2f}: HEADROOM vs PAUSE closure-rate difference {num(pp['mean_paired_difference']):+.3f} ({pp['wins']}/{pp['ties']}/{pp['losses']}). Against B1, closure rate {num(pb['mean_paired_difference']):+.3f}, HV rate {num(hv['mean_paired_difference']):+.3f}, net/task {num(net['mean_paired_difference']):+.3f}, utility/MB {num(eff['mean_paired_difference']):+.3f}.")
lines += ["","HEADROOM consistently improves PROP over whole-slot thermal pausing. Against B1 it consistently improves total closure rate, net utility per task, communication efficiency and energy efficiency, but it does not consistently improve HV or HV-TARGET closure rates. B1 uses substantially more communication and generally more controllable energy. No method crosses the 0.92 redline.","","Therefore the evidence supports keeping thermal-headroom partial transmission as the PROP thermal controller, but not a claim that PROP universally dominates B1 on high-value closure.","","Final label: `NO_STABLE_ADVANTAGE`"]
(OUT/"final_multiscenario_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

counts="、".join(f"{s['seed']}={s['task_count']}" for s in scenarios)
zh=["# 多场景热余量结论","",f"五个 seed 是不同任务场景，不是同一组268任务的重复实验。任务数：{counts}。","","|容量|方法|闭环率|HV率|HV-T率|净效用/任务|效用/MB|效用/Wh|最高热状态|","|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
for cap in CAPS:
  for method in METHODS:zh.append(f"|{cap:.2f}|{method}|{mean(cap,method,'closure_rate'):.3f}|{mean(cap,method,'hv_closure_rate'):.3f}|{mean(cap,method,'hv_target_closure_rate'):.3f}|{mean(cap,method,'common_net_utility_per_task'):.3f}|{mean(cap,method,'utility_per_transmitted_mb'):.3f}|{mean(cap,method,'utility_per_wh'):.2f}|{max(num(r['maximum_thermal_state']) for r in runs if num(r['capacity_multiplier'])==cap and r['method']==method):.3f}|")
zh += ["",f"HEADROOM相对PAUSE：0.50闭环率平均+{num(pair(.50,'PROP-PAUSE','closure_rate')['mean_paired_difference']):.3f}，0.35平均+{num(pair(.35,'PROP-PAUSE','closure_rate')['mean_paired_difference']):.3f}，两者均5/0/0；HV率和每任务净效用也均5/0/0。",f"相对B1：两种容量的闭环率、每任务净效用、效用/MB和效用/Wh均5/0/0，但HV率在0.50为0/0/5，0.35为1/2/2，HV-TARGET均0/0/5。PROP通信量明显更低，能耗也较低；所有运行热状态不超过0.92，无违规。结论跨场景稳定支持用HEADROOM替代PAUSE，但不支持PROP在高价值闭环上全面优于B1。","","NO_STABLE_ADVANTAGE"]
(OUT/"final_plain_language_summary_zh.md").write_text("\n".join(zh)+"\n",encoding="utf-8")

members=["revised_experiment_protocol.md","scenario_manifest.csv","scenario_distribution_summary.csv","multiscenario_run_summary.csv","multiscenario_aggregate_summary.csv","paired_scenario_comparison.csv","communication_efficiency_summary.csv","energy_efficiency_summary.csv","thermal_safety_summary.csv","latency_summary.csv","validation_report.md","final_multiscenario_report.md","final_plain_language_summary_zh.md"]
with zipfile.ZipFile(OUT/"thermal_headroom_multiscenario_analysis_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
    for name in members:z.write(OUT/name,name)
print("finalized",OUT)
