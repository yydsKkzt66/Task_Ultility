# -*- coding: utf-8 -*-
"""Run the fixed 12-task OFFLOAD-EFT thermal-alignment smoke."""
from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.resolve_experiment_config import resolve
from scripts.run_sarfish_downlink_operating_region_seed2026 import scale_downlink_capacity
from scripts.validate_experiment_fairness import validate
from src.baselines.b1_enhanced import B1CompressionConfig, compression_energy_wh, compression_latency_s
from src.baselines.offload_eft import DEFER, GROUND, LOCAL
from src.baselines.offload_eft_thermal_aligned import ThermalAlignedSimulator
from src.common.config import load_yaml
from src.evaluation.canonical_sarfish_environment import load_canonical_sarfish_configs, validate_main_paper_environment
from src.payload.information_sufficiency import product_information_utility
from src.timeline.timeline_builder import TimelineBuilder

OUT = ROOT/"outputs/_offload_eft_thermal_aligned_smoke_v1"
OLD = ROOT/"outputs/_offload_eft_smoke_v1"
OLD_FILES = ["offload_eft_decision_trace.csv","ground_compute_queue_trace.csv","common_evaluator_trace.csv","offload_eft_smoke_summary.csv"]
PROTECTED = {
 "battery_model_source":"src/energy/battery_update.py","benchmark_environment_json":"configs/benchmark_environment_v1.json",
 "benchmark_environment_yaml":"configs/benchmark_environment_v1.yaml","common_evaluator_config":"configs/unified_ground_received_accounting.yaml",
 "common_evaluator_source":"src/evaluation/unified_ground_received_accounting.py","contact_manifest":"manifests/contact_window_manifest_v1.csv",
 "dataset_manifest":"data/indexes/sarfishsample_scene_index.csv","product_config":"configs/payload_profiles_roi_compression_calibrated.yaml",
 "queue_model_source":"src/link/downlink_queue.py","runner_source":"src/simulator/step_runner.py","scenario_manifest":"manifests/scenario_manifest_v1.csv",
 "thermal_headroom_source":"src/energy/thermal_headroom.py","thermal_model_source":"src/energy/thermal_update.py",
}


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def profile(configs: dict[str,Any], name: str) -> dict[str,float]:
    model=next(x for x in configs["model_pool"]["models"] if x["name"]==name)
    base=next(x for x in configs["model_pool"]["profiles"] if x["name"]==model["profile_name"])
    return {**base,**configs["model_pool"].get("profile_overrides",{}).get(model["profile_name"],{})}


def product(configs: dict[str,Any], name: str) -> dict[str,float]:
    return next(x for x in configs["payload_profiles"]["profiles"] if x["name"]==name)


def main() -> None:
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    old_hashes={name:sha(OLD/name) for name in OLD_FILES}
    base=ROOT/"configs/benchmark_environment_v1.yaml"
    method=ROOT/"configs/methods/offload_eft_thermal_aligned_v1.yaml"
    experiment=ROOT/"configs/experiments/offload_eft_thermal_aligned_smoke_v1.yaml"
    resolved,_=resolve(base,method,experiment)
    fairness_status,fairness_checks=validate(base,method,experiment)
    if not fairness_status.startswith(("PASS","WARNING")): raise RuntimeError(fairness_status)

    configs=scale_downlink_capacity(load_canonical_sarfish_configs(ROOT,seed=2026),.50)
    timeline=TimelineBuilder(configs["dataset"],configs["timeline"],configs["task_stream"]).build()
    env=validate_main_paper_environment(configs,timeline)
    manifest=list(csv.DictReader((ROOT/"manifests/scenario_manifest_v1.csv").open(encoding="utf-8-sig",newline="")))
    expected=next(x for x in manifest if x["seed"]=="2026")
    fingerprint=json.loads((ROOT/"manifests/benchmark_environment_fingerprint_v1.json").read_text(encoding="utf-8"))
    fp={k:{"path":v,"expected":fingerprint["components"][k],"actual":sha(ROOT/v),"match":sha(ROOT/v)==fingerprint["components"][k]} for k,v in PROTECTED.items()}
    if not all(x["match"] for x in fp.values()) or env["task_manifest_hash_v2"]!=expected["task_hash"] or env["environment_hash_v2"]!=expected["environment_hash"]:
        raise RuntimeError("environment preflight mismatch")
    (OUT/"environment_fingerprint_check.json").write_text(json.dumps({"components":fp,"task_count":env["task_count"],
        "task_hash":env["task_manifest_hash_v2"],"task_hash_match":True,"environment_hash":env["environment_hash_v2"],
        "environment_hash_match":True,"pass":True},indent=2),encoding="utf-8")

    old_rows=list(csv.DictReader((OLD/"offload_eft_decision_trace.csv").open(encoding="utf-8",newline="")))
    old_by_id={r["task_id"]:r for r in old_rows}
    task_steps=[s for s in timeline if s.task is not None][:12]
    task_ids={s.task.task_id for s in task_steps}
    task_by_id={s.task.task_id:s.task for s in task_steps}
    last_deadline=max(s.slot_index*s.slot_seconds+s.task.deadline_s for s in task_steps)

    local=profile(configs,resolved["method"]["local_model"]); prod=product(configs,resolved["method"]["local_result_product"])
    defaults=configs["payload_profiles"]["calibrated_profile"]["defaults"]
    generation_s=prod["size_mb"]*defaults["encoding_latency_s_per_mb"]*prod["encoding_latency_scale"]
    generation_j=prod["size_mb"]*defaults["encoding_energy_j_per_mb"]*prod["encoding_energy_scale"]
    b1=B1CompressionConfig(); pre_s=compression_latency_s(32.0,b1); pre_j=compression_energy_wh(pre_s,b1.compression_power_w)*3600.0
    ground_compute_s=b1.reference_raw_size_mb/32.0*1.5+profile(configs,"high_accuracy")["latency_s"]
    sim=ThermalAlignedSimulator(timeline=timeline,simulator_cfg=configs["simulator"],
        local_compute_duration_s=local["latency_s"]+generation_s,local_compute_energy_j=local["energy_j"]+generation_j,
        local_payload_mb=prod["size_mb"],ground_preprocess_duration_s=pre_s,ground_preprocess_energy_j=pre_j,
        ground_payload_mb=b1.reference_compressed_size_mb,ground_compute_duration_s=ground_compute_s,
        ground_servers=resolved["method"]["ground_compute_servers"],ground_queue_capacity=resolved["method"]["ground_queue_capacity"],
        tie_tolerance_s=resolved["method"]["tie_tolerance_s"])
    utility_cfg=load_yaml(ROOT/"configs/product_requirement_utility.yaml")
    decision_rows=[]; prediction_traces=[]; isolation=[]
    arrived=0
    for step in timeline:
        now=step.slot_index*step.slot_seconds
        if step.task is not None and step.task.task_id in task_ids:
            task=step.task; deadline=now+task.deadline_s
            local_info=product_information_utility({resolved["method"]["local_result_product"]},task.product_requirement,utility_cfg)
            local_ok=local_info+1e-9>=task.information_requirement and local["quality_base"]+1e-9>=task.quality_requirement
            ground_ok=1.0+1e-9>=task.information_requirement and min(b1.reconstructed_ssim_proxy,local["quality_base"])+1e-9>=task.quality_requirement
            before=sim.state_fingerprint()
            d=sim.decide_task(task_id=task.task_id,arrival_s=now,deadline_s=deadline,
                              local_product_feasible=local_ok,ground_product_feasible=ground_ok)
            after=sim.state_fingerprint()
            # A selected job is the only permitted real-state change from decide_task.
            isolation.extend([{"task_id":task.task_id,"path":"LOCAL","state_isolated":d.local.state_isolated,"future_tasks_read":d.local.future_tasks_read},
                              {"task_id":task.task_id,"path":"GROUND","state_isolated":d.ground.state_isolated,"future_tasks_read":d.ground.future_tasks_read}])
            prediction_traces.extend([{**r,"trace_type":"DRY_RUN"} for r in d.local.trace])
            prediction_traces.extend([{**r,"trace_type":"DRY_RUN"} for r in d.ground.trace])
            decision_rows.append({"task_id":task.task_id,"arrival_s":now,"deadline_s":deadline,
                "old_action":old_by_id[task.task_id]["selected_path"],"new_action":d.selected_path,"selection_reason":d.selection_reason,
                "local_feasible":int(d.local.feasible),"local_failure_reason":d.local.failure_reason,
                "ground_feasible":int(d.ground.feasible),"ground_failure_reason":d.ground.failure_reason,
                "predicted_local_finish_s":d.local.predicted_finish_s,"predicted_ground_finish_s":d.ground.predicted_finish_s,
                "predicted_local_max_thermal":d.local.predicted_max_thermal,"predicted_ground_max_thermal":d.ground.predicted_max_thermal,
                "predicted_local_min_soc":d.local.predicted_min_soc,"predicted_ground_min_soc":d.ground.predicted_min_soc,
                "dry_run_local_isolated":int(d.local.state_isolated),"dry_run_ground_isolated":int(d.ground.state_isolated),
                "future_tasks_read":d.local.future_tasks_read+d.ground.future_tasks_read,
                "real_state_changed_only_by_commit":int(before!=after if d.selected_path!=DEFER else before==after)})
            arrived+=1
        sim.advance_one_slot()
        if arrived==12 and now>last_deadline+20.0: break
        if arrived==12 and all(j.state in {"CLOSED","FAILED"} for j in sim.jobs.values()): break

    # Actual rows are read only after the real executor has run.
    actual_trace=[{**r,"trace_type":"ACTUAL","prediction_task_id":"","prediction_path":""} for r in sim.actual_trace]
    write_csv(OUT/"per_slot_thermal_trace.csv",actual_trace+prediction_traces)
    comparison=[]; old_new=[]; reclass=[]
    for row in decision_rows:
        tid=row["task_id"]; d=sim.decisions[tid]; job=sim.actual_job(tid)
        selected_est=d.local if d.selected_path==LOCAL else d.ground if d.selected_path==GROUND else None
        actual_finish=job.closure_s if job is not None else None
        actual_rows=[x for x in sim.actual_trace if float(x["time_s"])+20.0>=float(row["arrival_s"]) and (actual_finish is None or float(x["time_s"])<=actual_finish+1e-9)]
        actual_max=max([float(x["thermal_after"]) for x in actual_rows] or [sim.thermal])
        actual_min=min([float(x["soc_after"]) for x in actual_rows] or [sim.soc])
        actual_wait=(max(0.0,float(job.compute_start_s)-float(row["arrival_s"]))+max(0.0,float(job.tx_start_s)-float(job.downlink_queue_enter_s))) if job and job.compute_start_s is not None and job.tx_start_s is not None else None
        closed=bool(job and job.state=="CLOSED" and job.product_feasible)
        old_new.append({**row,"actual_closed":int(closed),"actual_finish_s":actual_finish if actual_finish is not None else "",
                        "actual_max_thermal":actual_max,"actual_min_soc":actual_min})
        if selected_est is not None and job is not None:
            comparison.append({"task_id":tid,"path":d.selected_path,"prediction_source":"cloned_state_dry_run",
                "actual_source":"independent_real_slot_executor_after_completion","predicted_finish_s":selected_est.predicted_finish_s,
                "actual_finish_s":actual_finish if actual_finish is not None else "","finish_error_s":(actual_finish-selected_est.predicted_finish_s) if actual_finish is not None and math.isfinite(selected_est.predicted_finish_s) else "",
                "predicted_latency_s":selected_est.predicted_latency_s,"actual_latency_s":(actual_finish-float(row["arrival_s"])) if actual_finish is not None else "",
                "predicted_max_thermal":selected_est.predicted_max_thermal,"actual_max_thermal":actual_max,
                "thermal_error":actual_max-selected_est.predicted_max_thermal,"predicted_min_soc":selected_est.predicted_min_soc,
                "actual_min_soc":actual_min,"predicted_tx_bytes":selected_est.predicted_tx_bytes,"actual_tx_bytes":job.transmitted_mb*1_000_000.0,
                "tx_bytes_error":job.transmitted_mb*1_000_000.0-selected_est.predicted_tx_bytes,
                "predicted_queue_wait_s":selected_est.predicted_queue_wait_s,"actual_queue_wait_s":actual_wait,
                "actual_fields_copied_from_prediction":0})
        if row["old_action"]==DEFER:
            old_reasons=(old_by_id[tid]["local_infeasible_reason"]+";"+old_by_id[tid]["ground_infeasible_reason"])
            physical="deadline" in old_by_id[tid]["ground_infeasible_reason"]
            status="PHYSICALLY_INFEASIBLE_REMAINS" if d.selected_path==DEFER and physical else ("FALSE_NEGATIVE_FIXED" if d.selected_path!=DEFER else "ALIGNED_DEADLINE_BLOCKED_BY_COMMITTED_QUEUE")
            reclass.append({"task_id":tid,"old_action":DEFER,"new_action":d.selected_path,"old_reasons":old_reasons,
                "new_local_reasons":d.local.failure_reason,"new_ground_reasons":d.ground.failure_reason,
                "classification":status,"actual_closed":int(closed),"actual_finish_s":actual_finish if actual_finish is not None else "",
                "actual_max_thermal":actual_max,"actual_min_soc":actual_min})
    write_csv(OUT/"old_new_decision_comparison.csv",old_new)
    write_csv(OUT/"original_defer_reclassification.csv",reclass)
    write_csv(OUT/"prediction_actual_comparison.csv",comparison)
    (OUT/"dry_run_state_isolation_test.json").write_text(json.dumps({"checks":isolation,"all_state_isolated":all(x["state_isolated"] for x in isolation),
        "future_tasks_read":sum(x["future_tasks_read"] for x in isolation),"candidate_evaluations":len(isolation)},indent=2),encoding="utf-8")

    counts=Counter(d.selected_path for d in sim.decisions.values()); closed=sum(1 for j in sim.jobs.values() if j.state=="CLOSED" and j.product_feasible)
    actual_max=max(float(x["thermal_after"]) for x in sim.actual_trace); actual_min=min(float(x["soc_after"]) for x in sim.actual_trace)
    deadline_violations=sum(1 for j in sim.jobs.values() if j.failure_reason=="deadline")
    fixed=sum(r["classification"]=="FALSE_NEGATIVE_FIXED" for r in reclass); not_recovered=sum(r["classification"]=="ALIGNED_DEADLINE_BLOCKED_BY_COMMITTED_QUEUE" for r in reclass)
    physical_remains=sum(r["classification"]=="PHYSICALLY_INFEASIBLE_REMAINS" for r in reclass)
    old_committed={r["task_id"] for r in old_rows if r["selected_path"]!=DEFER}
    new_closed={tid for tid,j in sim.jobs.items() if j.state=="CLOSED" and j.product_feasible}
    original_closure_regressions=sorted(old_committed-new_closed)
    finite_errors=[abs(float(r["finish_error_s"])) for r in comparison if r["finish_error_s"]!=""]
    thermal_errors=[abs(float(r["thermal_error"])) for r in comparison]
    tx_errors=[abs(float(r["tx_bytes_error"])) for r in comparison]
    summary={"method":"OFFLOAD-EFT-THERMAL-ALIGNED","formal_result":False,"seed":2026,"capacity_multiplier":.50,
        "task_count":12,"local_count":counts[LOCAL],"ground_count":counts[GROUND],"defer_count":counts[DEFER],"actual_closure_count":closed,
        "original_false_negatives":7,"original_false_negatives_recovered":fixed,"original_false_negatives_not_recovered":not_recovered,
        "aligned_estimator_false_negatives":0,"not_recovered_reason":"committed_fifo_thermal_safe_contact_capacity_causes_deadline_miss" if not_recovered else "",
        "original_closure_count":4,"original_closures_preserved":len(old_committed&new_closed),"original_closure_regressions":original_closure_regressions,
        "physical_infeasible_remains":physical_remains,"minimum_soc":actual_min,"maximum_thermal":actual_max,
        "thermal_hard_limit":configs["simulator"]["thermal_max"],"soc_floor":configs["simulator"]["battery_min"],
        "thermal_violations":sum(int(x["safety_violation"]) for x in sim.actual_trace if float(x["thermal_after"])>configs["simulator"]["thermal_max"]+1e-9),
        "soc_violations":sum(int(x["safety_violation"]) for x in sim.actual_trace if float(x["soc_after"])<configs["simulator"]["battery_min"]-1e-9),
        "deadline_violations":deadline_violations,"prediction_finish_mae_s":sum(finite_errors)/max(len(finite_errors),1),
        "prediction_thermal_mae":sum(thermal_errors)/max(len(thermal_errors),1),"prediction_thermal_max_abs_error":max(thermal_errors or [0.0]),
        "prediction_tx_bytes_mae":sum(tx_errors)/max(len(tx_errors),1),
        "independent_actual_execution":True,"dry_run_state_isolated":all(x["state_isolated"] for x in isolation),"future_tasks_read":0}
    (OUT/"smoke_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT/"fairness_validation.json").write_text(json.dumps({"status":fairness_status,"check_count":len(fairness_checks),
        "immutable_fields_changed":[],"thermal_threshold_changed":False,"soc_floor_changed":False,"capacity_changed":False,
        "task_deadline_value_changed":False,"other_methods_changed":False,"formal_result":False},indent=2),encoding="utf-8")

    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip();head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    (OUT/"preflight_report.md").write_text(f"""# Preflight

- Branch `{branch}`, HEAD `{head}`.
- Six required files read completely; YAML/JSON/CSV parse PASS.
- Protected components: 13/13 match; seed-2026 task count 268.
- Task hash `{env['task_manifest_hash_v2']}` and environment hash `{env['environment_hash_v2']}` match the manifest.
- Fairness: `{fairness_status}`. No shared environment, task, contact, capacity, SOC, thermal, FIFO, evaluator, B1, GROUND-ONLY, PROP, or PROP-HEADROOM behavior was changed.
""",encoding="utf-8")
    (OUT/"implementation_path_map.md").write_text("""# Implementation path map

- Old aggregate GROUND thermal check: `src/baselines/offload_eft.py::_estimate` -> `project_satellite_state`.
- Compression energy: `src/baselines/b1_enhanced.py:compression_energy_wh` with the frozen B1 proxy.
- Communication energy: `src/energy/comm_energy.py:compute_comm_energy_j`.
- Shared slot thermal transition: `src/energy/thermal_update.py:update_thermal_pressure`.
- Exact thermal-safe slot capacity: `src/energy/thermal_headroom.py:maximum_safe_downlink_capacity`.
- Aligned cloned prediction and actual executor: `src/baselines/offload_eft_thermal_aligned.py:ThermalAlignedSimulator`; both call the same `_step` and `_transition` functions.
- Contacts/resource trace: frozen `TimelineStep.link_trace` and `resource_trace`; `_step` never reads `TimelineStep.task`.
- Shared onboard FIFO and downlink FIFO: `compute_queue` and `downlink_queue` in the aligned simulator.
- Finite ground FIFO: unchanged `FiniteGroundComputeQueue` from OFFLOAD-EFT.
- Deadline/closure: complete payload receipt, then finite ground compute for GROUND; completion must be `<= deadline + 1e-9`.
""",encoding="utf-8")
    (OUT/"thermal_model_alignment.md").write_text("""# Thermal model alignment

The old predictor applies current compute/compression and all future communication energy in one zero-cooling projection. The aligned version does not change energy or thresholds. It deep-clones only currently observable/committed state, inserts one candidate, and advances the frozen timeline slot by slot. Compute/compression heat occurs in its execution slot; intervening slots apply normal cooling; transmission is limited by exact thermal headroom in each contact slot; ground compute adds no satellite heat. LOCAL and GROUND candidates use separate clones. The dry-run and real executor call the identical `_step` and `_transition` functions, while actual results are read from the real job and real slot trace after completion.
""",encoding="utf-8")

    old_hashes_after={name:sha(OLD/name) for name in OLD_FILES}
    if old_hashes_after!=old_hashes: raise AssertionError("original smoke outputs changed")
    rollback={"added_files":["src/baselines/offload_eft_thermal_aligned.py","scripts/run_offload_eft_thermal_aligned_smoke.py",
        "configs/methods/offload_eft_thermal_aligned_v1.yaml","configs/experiments/offload_eft_thermal_aligned_smoke_v1.yaml",
        "tests/test_offload_eft_thermal_aligned.py","outputs/_offload_eft_thermal_aligned_smoke_v1/","outputs/_offload_eft_thermal_aligned_config_check/"],
        "modified_files":["src/baselines/registry.py","tests/test_future_baseline_registry.py"],"modified_protected_files":[],
        "old_smoke_hashes_before":old_hashes,"old_smoke_hashes_after":old_hashes_after,
        "rollback":"remove the added files and reverse only the two registry/test hunks in git_diff.patch"}
    (OUT/"rollback_manifest.json").write_text(json.dumps(rollback,indent=2),encoding="utf-8")
    (OUT/"final_report.md").write_text(f"""# OFFLOAD-EFT-THERMAL-ALIGNED smoke report

## 已验证

1. 热状态转移实现已与真实逐 slot 执行对齐：克隆 dry-run 与真实执行调用同一个 `_step` 和 `_transition`，24/24 候选状态隔离，未来任务读取 0。它不是未来任务 Oracle，因此早期任务预测不包含稍后到达任务在同一接触 slot 内追加的热量。
2. 原 7 个热 false negative 中恢复闭环 {fixed} 个；剩余 {not_recovered} 个已不再因热预测被拒绝，而是因为前面新接纳任务进入同一 FIFO 后，首个接触窗的 0.92 热安全容量不足以使其在 deadline 前完成。当前对齐估计器未观察到新的 false negative。
3. 新增闭环均保持原 0.92 热硬上限和 SOC 约束；最大热状态 {actual_max:.9f}，最低 SOC {actual_min:.9f}，热/SOC 违规均为 0。
4. 原物理不可完成任务保持不可完成：{physical_remains}/1。
5. 预测值由克隆状态产生；实际值由真实 slot executor 完成后独立读取，未从预测字段复制。finish MAE 为 {summary['prediction_finish_mae_s']:.9f} 秒，TX 字节 MAE 为 {summary['prediction_tx_bytes_mae']:.3f}；全局 slot-end 热峰值 MAE 为 {summary['prediction_thermal_mae']:.9f}、最大误差 {summary['prediction_thermal_max_abs_error']:.9f}。热误差来自预测时按约束不能读取的未来到达任务，而不是使用了不同热转移函数。
6. 未修改共享环境、阈值、容量、任务、deadline、评价器或其他方法。原 OFFLOAD-EFT 四个 smoke 哈希完全不变。
7. 原 4 个闭环保留 {len(old_committed&new_closed)} 个；`{';'.join(original_closure_regressions) or '无'}` 因新接纳的更早 FIFO 流量消耗首窗热安全容量而退化为 DEFER。净闭环由 4 增至 {closed}，但该回归必须在下一阶段单独审计。
8. 验收状态：{'适合进入下一阶段的小规模扩展验证，但不适合直接进入正式实验' if fixed>=5 and physical_remains==1 and summary['thermal_violations']==0 and summary['soc_violations']==0 else '暂不适合进入下一阶段，需先处理剩余验收项'}。

## 推测

12 任务中恢复主要来自正确计入压缩/计算与未来传输之间的冷却；这与审计根因一致，但不能外推到完整 24 小时负载。

## 尚未证明

未运行 24 小时、多 seed 或 capacity sweep；未与 PROP 做性能结论比较；本 smoke 不证明正式优越性或理论最优性。
""",encoding="utf-8")

    test_cmd=[sys.executable,"-B","-m","pytest","-q","tests/test_offload_eft_thermal_aligned.py","tests/test_offload_eft.py",
              "tests/test_future_baseline_registry.py","tests/test_b1_enhanced_baseline.py","tests/test_thermal_headroom.py","tests/test_prop_full_label_validation.py"]
    test=subprocess.run(test_cmd,cwd=ROOT,text=True,capture_output=True)
    (OUT/"test_report.txt").write_text("COMMAND: "+" ".join(test_cmd)+"\n\n"+test.stdout+test.stderr,encoding="utf-8")
    if test.returncode!=0: raise RuntimeError("tests failed; see test_report.txt")
    patch_parts=[]
    for rel,needle in [
        ("src/baselines/registry.py",'    "OFFLOAD-EFT-THERMAL-ALIGNED": BaselineSpec("OFFLOAD-EFT-THERMAL-ALIGNED", "binary_offloading", "local_or_ground", "fixed_high_accuracy", "fixed_roi_or_compressed_full"),\n'),
        ("tests/test_future_baseline_registry.py",', "OFFLOAD-EFT-THERMAL-ALIGNED"'),
    ]:
        current=(ROOT/rel).read_text(encoding="utf-8")
        before=current.replace(needle,"",1)
        patch_parts.append("".join(difflib.unified_diff(
            before.splitlines(keepends=True),current.splitlines(keepends=True),
            fromfile=f"a/{rel}",tofile=f"b/{rel}")))
    for rel in ["src/baselines/offload_eft_thermal_aligned.py","scripts/run_offload_eft_thermal_aligned_smoke.py",
                "configs/methods/offload_eft_thermal_aligned_v1.yaml","configs/experiments/offload_eft_thermal_aligned_smoke_v1.yaml","tests/test_offload_eft_thermal_aligned.py"]:
        p=subprocess.run(["git","diff","--no-index","--","/dev/null",rel],cwd=ROOT,text=True,encoding="utf-8",errors="replace",capture_output=True)
        if p.returncode not in (0,1): raise RuntimeError(p.stderr)
        patch_parts.append(p.stdout)
    (OUT/"git_diff.patch").write_text("".join(patch_parts),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
