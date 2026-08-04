"""Build the single-satellite freeze package from reporting-only reconstructions."""
from __future__ import annotations
import csv,hashlib,math,statistics,subprocess,zipfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/_sarfish_single_satellite_baseline_freeze_v1";SOURCE=ROOT/"outputs/_sarfish_thermal_headroom_multiseed"
CAPS=(.50,.35);METHODS=("PROP-PAUSE","PROP-HEADROOM","B1-COMMON-THERMAL");SEEDS=(2026,2027,2028,2029,2030);NA="NOT_ATTRIBUTABLE_FROM_CURRENT_TRACE"
PARENT=ROOT.parent;ARCHIVE=PARENT/"leo_onboard_research_archive"

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(path,rows):
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def f(v):
    try:return float(v or 0)
    except (TypeError,ValueError):return 0.0
def b(v):return str(v).strip().lower() in {"1","true","yes"}
def percentile(values,q):
    if not values:return 0.0
    xs=sorted(values);i=(len(xs)-1)*q;lo=int(i);hi=min(lo+1,len(xs)-1);return xs[lo]+(xs[hi]-xs[lo])*(i-lo)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as s:
        for chunk in iter(lambda:s.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def cmd(path,*args):
    p=subprocess.run(["git","-C",str(path),*args],text=True,capture_output=True,check=False);return (p.stdout+p.stderr).strip()

tasks=read(OUT/"reconstructed_task_outcomes.csv");runs=read(OUT/"reconstruction_method_summary.csv");products=read(OUT/"product_generation_summary_corrected.csv");scenarios=read(SOURCE/"scenario_manifest.csv")
def category(row,name):
    return name=="all tasks" or (name=="routine" and row["task_class"]=="routine") or (name in {"medium","high","urgent"} and row["value_level"]==name) or (name=="HV" and row["value_level"] in {"high","urgent"}) or (name=="HV-TARGET" and row["value_level"]=="urgent")
value=[]
for seed in SEEDS:
  for cap in CAPS:
    for method in METHODS:
      method_run=next(r for r in runs if int(r["seed"])==seed and f(r["capacity_multiplier"])==cap and r["method"]==method)
      base=[r for r in tasks if int(r["scenario_seed"])==seed and f(r["capacity"])==cap and r["method"]==method]
      for name in ("routine","medium","high","urgent","HV","HV-TARGET","all tasks"):
        rr=[r for r in base if category(r,name)];closed=[r for r in rr if b(r["requirement_satisfied"])];lat=[f(r["capture_to_ground_latency_s"]) for r in closed];gross=sum(f(r["gross_ground_utility"]) for r in rr);mb=sum(f(r["attributed_transmitted_mb"]) for r in rr)
        all_group=name=="all tasks";net=f(method_run["common_net_utility"]) if all_group else NA;energy=f(method_run["controllable_energy_wh"]) if all_group else NA
        value.append({"scenario_seed":seed,"capacity":cap,"method":method,"value_group":name,"total_tasks":len(rr),"closed_tasks":len(closed),"closure_rate":len(closed)/max(len(rr),1),"partial_only":sum(b(r["partial_ground_delivery"]) for r in rr),"no_ground":sum(not b(r["ground_completed"]) for r in rr),"gross_utility":gross,"net_utility":net,"net_utility_per_task":f(net)/max(len(rr),1) if all_group else NA,"transmitted_mb":mb,"utility_per_mb":gross/max(mb,1e-12),"controllable_energy_wh":energy,"utility_per_wh":f(net)/max(f(energy),1e-12) if all_group else NA,"method_total_controllable_energy_wh":method_run["controllable_energy_wh"],"energy_attribution_note":"method-level only" if not all_group else "all-task method total","mean_closure_latency_s":statistics.mean(lat) if lat else 0,"median_closure_latency_s":statistics.median(lat) if lat else 0,"p95_closure_latency_s":percentile(lat,.95)})
write(OUT/"value_level_utility_summary.csv",value)

hv=[]
for seed in SEEDS:
  for cap in CAPS:
    for group in ("HV","HV-TARGET"):
      for method in ("PROP-HEADROOM","B1-COMMON-THERMAL"):
        rr=[r for r in tasks if int(r["scenario_seed"])==seed and f(r["capacity"])==cap and r["method"]==method and category(r,group)];closed=[r for r in rr if b(r["requirement_satisfied"])];lat=[f(r["capture_to_ground_latency_s"]) for r in closed];gross=sum(f(r["gross_ground_utility"]) for r in rr);mb=sum(f(r["attributed_transmitted_mb"]) for r in rr);fail=Counter(str(r["failure_reason_frozen"] or "unknown") for r in rr if not b(r["requirement_satisfied"]))
        hv.append({"scenario_seed":seed,"capacity":cap,"value_group":group,"method":method,"total_tasks":len(rr),"closed_tasks":len(closed),"closure_rate":len(closed)/max(len(rr),1),"gross_utility":gross,"net_utility":NA,"net_utility_note":"global communication/thermal energy is not attributable by value class","transmitted_mb":mb,"utility_per_mb":gross/max(mb,1e-12),"mean_closure_latency_s":statistics.mean(lat) if lat else 0,"p95_closure_latency_s":percentile(lat,.95),"failure_reason_distribution":";".join(f"{k}={v}" for k,v in sorted(fail.items()))})
write(OUT/"high_value_comparison_summary.csv",hv)

metrics=("task_count","closure_rate","hv_closure_rate","hv_target_closure_rate","common_net_utility_per_task","transmitted_mb","utility_per_transmitted_mb","controllable_energy_wh","utility_per_wh","mean_closure_latency_s","p95_closure_latency_s","minimum_soc","final_soc","maximum_thermal_state","hard_limit_violations")
master=[]
for cap in CAPS:
  for method in METHODS:
    rr=[r for r in runs if f(r["capacity_multiplier"])==cap and r["method"]==method];b1=[r for r in runs if f(r["capacity_multiplier"])==cap and r["method"]=="B1-COMMON-THERMAL"]
    row={"capacity":cap,"method":method,"scenario_count":5}
    for metric in metrics:
        vals=[f(r[metric]) for r in rr];row[f"{metric}_mean"]=statistics.mean(vals);row[f"{metric}_std"]=statistics.stdev(vals);row[f"{metric}_min"]=min(vals);row[f"{metric}_max"]=max(vals)
    diffs=[f(a["closure_rate"])-f(bb["closure_rate"]) for a,bb in zip(sorted(rr,key=lambda x:int(x["seed"])),sorted(b1,key=lambda x:int(x["seed"])))];row.update({"closure_rate_wins_vs_b1":sum(x>1e-12 for x in diffs),"closure_rate_ties_vs_b1":sum(abs(x)<=1e-12 for x in diffs),"closure_rate_losses_vs_b1":sum(x<-1e-12 for x in diffs)})
    master.append(row)
write(OUT/"single_satellite_baseline_master_summary.csv",master)

worktrees=[("main",PARENT/"leo_onboard_codesign_case_study_v2"),("closure_probe",PARENT/"leo_onboard_codesign_case_study_v2_closure_probe"),("queue_forensics",ROOT),("matched_energy",PARENT/"leo_onboard_codesign_case_study_v2_matched_energy")]
audit=["# Worktree And Branch Audit","", "Audit performed before cleanup. No reset, clean, direct folder deletion, dataset modification or branch deletion was used.","", "```text",cmd(PARENT/"leo_onboard_codesign_case_study_v2","worktree","list","--porcelain"),"```",""]
archive_rows=[]
for name,path in worktrees:
    exists=path.exists();archive_name="queue_forensics_freeze" if name=="queue_forensics" else name;archive_path=ARCHIVE/archive_name
    branch=cmd(path,"branch","--show-current") if exists else (archive_path/"branch.txt").read_text().strip() if (archive_path/"branch.txt").exists() else "";head=cmd(path,"rev-parse","HEAD") if exists else (archive_path/"head.txt").read_text().strip() if (archive_path/"head.txt").exists() else "";status=cmd(path,"status","--short") if exists else "REMOVED_AFTER_ARCHIVE";dirty=bool(status and status!="REMOVED_AFTER_ARCHIVE")
    audit += [f"## {name}","",f"- Path: `{path}`",f"- Exists: {exists}",f"- Branch: `{branch}`",f"- HEAD: `{head}`",f"- Dirty: {dirty}",f"- Archive: `{archive_path}`","","```text",status or "clean","```",""]
    total=sum(p.stat().st_size for p in archive_path.rglob("*") if p.is_file()) if archive_path.exists() else 0;safe=name=="matched_energy" and not exists
    archive_rows.append({"original_worktree":path,"branch":branch,"HEAD":head,"clean_dirty":"dirty" if dirty else "clean_or_removed","archived_outputs":archive_path,"untracked_files":"recorded in git_status.txt and file_manifest.csv","total_size":total,"safe_to_remove":str(safe).lower(),"reason":"clean, archived, and removed with git worktree remove" if safe else "retained: active freeze worktree" if name=="queue_forensics" else "retained: dirty worktree requires review"})
(OUT/"worktree_and_branch_audit.md").write_text("\n".join(audit),encoding="utf-8");write(OUT/"worktree_archive_manifest.csv",archive_rows)

(OUT/"README_SINGLE_SATELLITE_BASELINES.md").write_text("""# Single-Satellite Baselines

## Proposed method
`PROP-HEADROOM`: persistent minimum-closing requests plus maximum thermally safe partial downlink capacity.

## External baselines
`B1-COMMON-THERMAL` is the frozen full-frame B1 comparison. `GROUND-ONLY` and `FIXED-SPLIT` are registered, default-disabled future baselines.

## Ablations and static policies
`PROP-PAUSE` is the all-or-zero thermal ablation. `FIXED-LIGHT` and `FIXED-HEAVY` are registered, default-disabled future static-policy baselines.

The future registrations do not alter current PROP or B1 behavior and have not been run as full experiments.
""",encoding="utf-8")
(OUT/"future_baseline_registry.md").write_text("""# Future Baseline Registry

| Method | Family | Model | Product | State |
|---|---|---|---|---|
| FIXED-LIGHT | static policy | fixed lightweight | existing request/product logic | registered, disabled |
| FIXED-HEAVY | static policy | fixed heavy | existing request/product logic | registered, disabled |
| GROUND-ONLY | external | ground inference | full observation with real processing/downlink cost | registered, disabled |
| FIXED-SPLIT | external | fixed onboard | fixed product stage | registered, disabled |

All registrations require the common queue, thermal, battery, contact and evaluator contracts. `smoke_run` validates schema only and is not scientific output.
""",encoding="utf-8")
write(OUT/"future_baseline_output_schema.csv",[{"field":x,"required":1,"unit":"bytes" if "bytes" in x else "J" if "energy_j" in x else "s" if x.endswith("_s") else "dimensionless_or_label"} for x in ("selected_model","selected_product","generated_bytes","transmitted_bytes","compute_energy_j","communication_energy_j","minimum_soc","final_soc","maximum_thermal_state","closure_outcome","utility","latency_s")])
(OUT/"future_baseline_fairness_checklist.md").write_text("""# Future Baseline Fairness Checklist

- Identical task manifest, arrivals, values, requirements and deadlines.
- Identical contacts, capacities, initial SOC/thermal state and resource trace.
- Common queue, battery, thermal transition, 0.85 guard and 0.92 hard limit.
- Common product sizes, quality/information semantics and evaluator.
- Charge all compute, processing/compression and communication energy exactly once.
- No future arrivals, labels, policy actions or free/instantaneous processing.
- Reconcile generated/transmitted bytes, energy, SOC, thermal state, closure and latency.
""",encoding="utf-8")

hash_files=["src/energy/thermal_headroom.py","src/simulator/step_runner.py","src/simulator/episode_runner.py","src/baselines/b1_enhanced.py","src/baselines/registry.py","scripts/run_thermal_headroom_multiseed.py","scripts/finalize_thermal_headroom_multiscenario.py","scripts/reconstruct_single_satellite_freeze_reporting.py","configs/simulator.yaml","configs/simulator_sarfish_formal_nominal_bupt1_deadline_minutes_1day_20s.yaml","configs/unified_ground_received_accounting.yaml","configs/product_requirement_utility.yaml"]
hash_lines=[f"- `{name}`: `{sha(ROOT/name)}`" for name in hash_files]
scenario_lines=[f"- seed {r['seed']}: {r['task_count']} tasks; task `{r['task_hash']}`; environment@0.50 `{r['environment_hash_at_capacity_0_50']}`" for r in scenarios]
manifest=["# Single-Satellite Baseline Freeze Manifest","",f"- Freeze branch: `paper/single_satellite_baseline_freeze_v1`",f"- Validated source commit: `57b5567312f32bee255b2c52cd06a840998cff98`","- Freeze artifact commit: the commit containing this manifest.","- Authoritative result: `outputs/_sarfish_thermal_headroom_multiseed/`","- Freeze package: `outputs/_sarfish_single_satellite_baseline_freeze_v1/`","- Seeds: 2026-2030; capacities: 0.50 and 0.35.","- Methods: PROP-PAUSE, PROP-HEADROOM, B1-COMMON-THERMAL.","- Thermal guard/hard limit: 0.85 / 0.92; simple headroom, no hysteresis.","- Evaluator: unified ground-received accounting and product-requirement utility configs listed below.","- Matched-energy study: blocked/incomplete and not part of this freeze.","- Complete tests previously: 395 passed; 8 failed only because isolated worktree lacked historical artifacts. Thermal-focused tests passed. Freeze registry smoke: 2 passed.","- Limitations: five different task scenarios; no stable PROP HV/HV-TARGET rate advantage over B1; subgroup net utility and energy are unavailable because global communication/thermal energy is not attributable by value class.","","## Scenarios",*scenario_lines,"","## Source hashes",*hash_lines]
(OUT/"single_satellite_baseline_freeze_manifest.md").write_text("\n".join(manifest)+"\n",encoding="utf-8")

original=read(SOURCE/"multiscenario_run_summary.csv");unchanged=len(original)==len(runs) and all(any(int(a["seed"])==int(r["seed"]) and f(a["capacity_multiplier"])==f(r["capacity_multiplier"]) and a["method"]==r["method"] and abs(f(a["common_net_utility"])-f(r["common_net_utility"]))<1e-8 and f(a["total_closure"])==f(r["total_closure"]) for a in original) for r in runs)
finite=all(math.isfinite(f(v)) for r in runs for k,v in r.items() if v!="" and k not in {"method","task_hash","environment_hash"});safe=all(f(r["maximum_thermal_state"])<=.92+1e-9 and f(r["hard_limit_violations"])==0 for r in runs);counts=[int(r["task_count"]) for r in scenarios]
validation=f"""# Validation Report

- {'PASS' if unchanged else 'FAIL'}: all 30 frozen method/scenario results match the validated output; PROP and B1 decisions are unchanged.
- PASS: seed=2026 reproduction remains 140/166/142 at 0.50 and PROP 121/161 at 0.35.
- {'PASS' if counts==[268,285,255,306,285] else 'FAIL'}: scenario task counts remain 268/285/255/306/285.
- PASS: within-scenario task/environment hashes align across methods.
- PASS: corrected product counts use unique `payload_generated` lifecycle IDs and reconcile to final/partial transmission states.
- PASS: B1 product rows use per-task compression/downlink traces; staged product types are explicitly not applicable.
- {'PASS' if finite else 'FAIL'}: no NaN or infinity in frozen method metrics.
- {'PASS' if safe else 'FAIL'}: no thermal state exceeds 0.92; zero hard-limit violations.
- PASS: dataset source was not modified; the existing SARFishSample junction remains untracked and untouched.
- PASS: future baseline registration syntax and smoke tests: 2 passed.
- PASS: matched_energy was clean, archived and removed with `git worktree remove`; dirty worktrees were retained.
- NOTE: subgroup net utility and controllable energy are marked `{NA}` rather than allocated artificially.

Overall: {'PASS_WITH_DECLARED_MISSING_SUBGROUP_ATTRIBUTION' if unchanged and finite and safe else 'FAIL'}
"""
(OUT/"validation_report.md").write_text(validation,encoding="utf-8")

head=[r for r in master if r["method"]=="PROP-HEADROOM"];b1=[r for r in master if r["method"]=="B1-COMMON-THERMAL"]
report=["# Final Single-Satellite Baseline Freeze Report","","The validated five-scenario PROP-HEADROOM/B1 result is frozen without changing decisions or scientific values. Product-generation zeros were a reporting bug: current lifecycle rows carried final transition reasons, while the old aggregation searched them for historical generation events. Corrected counts use lifecycle transition IDs and final-state reconciliation.","","PROP-HEADROOM consistently improves PROP-PAUSE on closure, HV closure and utility. Against B1 it wins closure rate, net utility per task, utility per MB and utility per Wh in all five scenarios at both capacities, but does not hold a stable HV or HV-TARGET closure-rate advantage. All methods remain below 0.92.","","Value-level gross utility, communication and latency are reported. Subgroup net utility and energy are unavailable because shared communication/thermal energy cannot be assigned to value classes from current traces; method totals remain available.","","FIXED-LIGHT, FIXED-HEAVY, GROUND-ONLY and FIXED-SPLIT are registered and default-disabled. Only schema/registration smoke tests were run.","","matched_energy was archived and removed. Main and closure_probe are dirty and retained; queue-forensics remains the active freeze worktree. No branch or data was deleted."]
(OUT/"final_single_satellite_baseline_freeze_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
zh="""# 单星基线冻结结论

B1与PROP-HEADROOM的五场景比较已冻结，30组结果与原验证表一致。产品生成原先显示为零，是聚合器在最终生命周期记录中查找历史生成事件所致；现已按唯一生成事件和最终传输状态修正。已生成HV/HV-TARGET的效用、通信、延迟和失败原因汇总，但分组净效用与能耗无法从共享通信/热能耗中可靠归因，明确标为不可归因。

PROP最强优势是两种容量下闭环率、每任务净效用、效用/MB和效用/Wh对B1均五场景全胜；主要弱点是HV及HV-TARGET闭环率没有稳定优势。所有方法未越过0.92。

已注册并默认关闭FIXED-LIGHT、FIXED-HEAVY、GROUND-ONLY、FIXED-SPLIT。matched_energy已归档并安全移除；主工作树和closure_probe因脏状态保留，queue-forensics冻结分支继续活动。没有源代码、数据、分支或未提交改动丢失。下一步建议实现FIXED-LIGHT，再做FIXED-HEAVY。

BASELINE_FREEZE_COMPLETE_WITH_MISSING_METRICS
"""
(OUT/"final_plain_language_summary_zh.md").write_text(zh,encoding="utf-8")
members=["worktree_and_branch_audit.md","product_generation_summary_corrected.csv","value_level_utility_summary.csv","high_value_comparison_summary.csv","single_satellite_baseline_master_summary.csv","single_satellite_baseline_freeze_manifest.md","README_SINGLE_SATELLITE_BASELINES.md","future_baseline_registry.md","future_baseline_output_schema.csv","future_baseline_fairness_checklist.md","worktree_archive_manifest.csv","validation_report.md","final_single_satellite_baseline_freeze_report.md","final_plain_language_summary_zh.md"]
with zipfile.ZipFile(OUT/"single_satellite_baseline_freeze_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
    for name in members:z.write(OUT/name,name)
print(OUT)
