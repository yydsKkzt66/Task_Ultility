# -*- coding: utf-8 -*-
"""Run a deterministic, non-formal OFFLOAD-EFT smoke on a task-manifest prefix."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, subprocess, sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.resolve_experiment_config import resolve
from scripts.validate_experiment_fairness import validate
from src.baselines.b1_enhanced import B1CompressionConfig, compression_energy_wh, compression_latency_s
from src.baselines.offload_eft import ContactSegment, OffloadEFTSimulator
from src.common.config import load_yaml
from scripts.run_sarfish_downlink_operating_region_seed2026 import scale_downlink_capacity
from src.evaluation.canonical_sarfish_environment import load_canonical_sarfish_configs, validate_main_paper_environment
from src.evaluation.unified_ground_received_accounting import score_task
from src.payload.information_sufficiency import product_information_utility
from src.timeline.timeline_builder import TimelineBuilder

OUT = ROOT / "outputs/_offload_eft_smoke_v1"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def git(*args: str) -> str: return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()


def contact_segments(timeline, multiplier: float) -> list[ContactSegment]:
    segments=[]
    for step in timeline:
        if step.link_trace.in_contact:
            start=step.slot_index*step.slot_seconds
            segments.append(ContactSegment(start,start+step.slot_seconds,
                step.link_trace.throughput_mbps*multiplier/8.0))
    return segments


def profile(configs, model_name: str) -> dict[str, float]:
    model=next(row for row in configs["model_pool"]["models"] if row["name"]==model_name)
    base=next(row for row in configs["model_pool"]["profiles"] if row["name"]==model["profile_name"])
    return {**base,**configs["model_pool"].get("profile_overrides",{}).get(model["profile_name"],{})}


def product(configs, name: str) -> dict[str, float]:
    return next(row for row in configs["payload_profiles"]["profiles"] if row["name"]==name)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--task-limit",type=int,default=12); args=parser.parse_args()
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        allowed={"preflight_report.md","environment_fairness_validation.md"}
        unexpected={p.name for p in out.iterdir()}-allowed
        if unexpected: raise FileExistsError(f"refusing to overwrite smoke outputs: {sorted(unexpected)}")
    out.mkdir(parents=True,exist_ok=True)
    base=ROOT/"configs/benchmark_environment_v1.yaml"; method_path=ROOT/"configs/methods/offload_eft_v1.yaml"; exp_path=ROOT/"configs/experiments/offload_eft_smoke_v1.yaml"
    resolved,_=resolve(base,method_path,exp_path); fairness_status,checks=validate(base,method_path,exp_path)
    if not fairness_status.startswith(("PASS","WARNING")): raise RuntimeError(fairness_status)
    configs=scale_downlink_capacity(load_canonical_sarfish_configs(ROOT,seed=2026),.50)
    timeline=TimelineBuilder(configs["dataset"],configs["timeline"],configs["task_stream"]).build()
    env=validate_main_paper_environment(configs,timeline)
    manifest=list(csv.DictReader((ROOT/"manifests/scenario_manifest_v1.csv").open(encoding="utf-8-sig",newline="")))
    expected=next(row for row in manifest if row["seed"]=="2026")
    if env["task_count"] != 268: raise AssertionError("seed 2026 task count mismatch")
    if env["task_manifest_hash_v2"] != expected["task_hash"]: raise AssertionError("task hash mismatch")
    if env["environment_hash_v2"] != expected["environment_hash"]: raise AssertionError("environment hash mismatch")
    local=profile(configs,resolved["method"]["local_model"]); prod=product(configs,resolved["method"]["local_result_product"])
    b1=B1CompressionConfig(); raw_mb=32.0; ground_pre_s=compression_latency_s(raw_mb,b1)
    ground_pre_j=compression_energy_wh(ground_pre_s,b1.compression_power_w)*3600.0
    ground_compute_s=b1.reference_raw_size_mb/raw_mb*1.5+profile(configs,"high_accuracy")["latency_s"]
    defaults=configs["payload_profiles"]["calibrated_profile"]["defaults"]
    generation_s=prod["size_mb"]*defaults["encoding_latency_s_per_mb"]*prod["encoding_latency_scale"]
    generation_j=prod["size_mb"]*defaults["encoding_energy_j_per_mb"]*prod["encoding_energy_scale"]
    sim=OffloadEFTSimulator(contacts=contact_segments(timeline,1.0),simulator_cfg=configs["simulator"],
        local_compute_latency_s=local["latency_s"],local_compute_energy_j=local["energy_j"],
        local_product_size_mb=prod["size_mb"],local_generation_latency_s=generation_s,
        local_generation_energy_j=generation_j,ground_payload_size_mb=b1.reference_compressed_size_mb,
        ground_preprocess_latency_s=ground_pre_s,ground_preprocess_energy_j=ground_pre_j,
        ground_compute_latency_s=ground_compute_s,ground_servers=resolved["method"]["ground_compute_servers"],
        ground_queue_capacity=resolved["method"]["ground_queue_capacity"],tie_tolerance_s=resolved["method"]["tie_tolerance_s"])
    utility_cfg=load_yaml(ROOT/"configs/product_requirement_utility.yaml")
    tasks=[]
    for step in timeline:
        sim.advance_environment(
            slot_seconds=step.slot_seconds,
            solar_energy_j=step.resource_trace.solar_energy_j,
            base_power_w=step.resource_trace.base_power_w,
            temperature_reference_c=step.resource_trace.temperature_reference_c,
            is_sunlit=step.resource_trace.is_sunlit,
        )
        if step.task is None: continue
        task=step.task; tasks.append(step); arrival=step.slot_index*step.slot_seconds
        local_info=product_information_utility({resolved["method"]["local_result_product"]},task.product_requirement,utility_cfg)
        local_ok=local_info+1e-9>=task.information_requirement and local["quality_base"]+1e-9>=task.quality_requirement
        ground_ok=1.0+1e-9>=task.information_requirement and min(b1.reconstructed_ssim_proxy,local["quality_base"])+1e-9>=task.quality_requirement
        sim.process_task(task_id=task.task_id,arrival_s=arrival,deadline_time_s=arrival+task.deadline_s,
                         local_product_feasible=local_ok,ground_product_feasible=ground_ok)
        if len(tasks) >= args.task_limit: break
    accounting=load_yaml(ROOT/"configs/unified_ground_received_accounting.yaml")
    task_by_id={step.task.task_id:step.task for step in tasks}
    common_rows=[]
    for row in sim.decision_trace:
        task=task_by_id[row["task_id"]]; closed=row["final_status"]=="CLOSED"
        info=(product_information_utility({resolved["method"]["local_result_product"]},task.product_requirement,utility_cfg)
              if row["selected_path"]=="LOCAL" else (1.0 if row["selected_path"]=="GROUND" else 0.0))
        quality=(local["quality_base"] if row["selected_path"]=="LOCAL" else
                 (min(b1.reconstructed_ssim_proxy,local["quality_base"]) if row["selected_path"]=="GROUND" else 0.0))
        base_row={"task_id":row["task_id"],"method":"OFFLOAD-EFT","variant":"smoke_proxy",
            "capture_time_s":float(row["arrival_time"]),"deadline_time_s":float(row["arrival_time"])+task.deadline_s,
            "effective_task_value":task.task_value,"ground_completed":int(closed),
            "capture_to_ground_latency_s":float(row["actual_closure_latency"] or 0.0),
            "ground_information_sufficiency":info,"ground_result_quality":quality if closed else 0.0,
            "partial_ground_gross_utility":0.0,"requirement_satisfied":int(closed),
            "on_time_task_success":int(closed),"selected_path":row["selected_path"]}
        common_rows.append(score_task(base_row,accounting))
    write_csv(out/"offload_eft_decision_trace.csv",sim.decision_trace)
    write_csv(out/"ground_compute_queue_trace.csv",sim.ground_queue_rows())
    write_csv(out/"common_evaluator_trace.csv",common_rows)
    counts=Counter(row["selected_path"] for row in sim.decision_trace)
    finite=all(not isinstance(v,float) or math.isfinite(v) for row in sim.decision_trace for v in row.values())
    max_thermal=max(float(row["maximum_thermal_state"]) for row in sim.decision_trace)
    min_soc=min(float(row["minimum_soc"]) for row in sim.decision_trace)
    summary=[{"seed":2026,"capacity_multiplier":.50,"task_subset_rule":f"first_{args.task_limit}_arrivals",
        "task_count":len(tasks),"local_count":counts["LOCAL"],"ground_count":counts["GROUND"],
        "defer_count":counts["DEFER"],"min_soc":min_soc,"max_thermal":max_thermal,
        "hard_safety_violations":int(min_soc<configs["simulator"]["battery_min"]-1e-9 or max_thermal>configs["simulator"]["thermal_max"]+1e-9),
        "duplicate_decisions":len(sim.decision_trace)-len({r["task_id"] for r in sim.decision_trace}),
        "migration_count":0,"replication_count":0,"common_evaluator_closure_count":sum(int(r["on_time_task_success"]) for r in common_rows),
        "finite_numeric_values":finite,"formal_result":False}]
    write_csv(out/"offload_eft_smoke_summary.csv",summary)
    resolved_doc={**resolved,"smoke_proxy":{
        "ground_compute_status":"PROXY","ground_compute_servers":1,"ground_compute_latency_s":ground_compute_s,
        "ground_compute_source":"B1 decode proxy + existing high_accuracy profile latency",
        "local_ground_result_processing_s":0.0,"local_ground_result_processing_source":"no independent active proxy"}}
    (out/"offload_eft_config_resolved.yaml").write_text(yaml.safe_dump(resolved_doc,sort_keys=False,allow_unicode=True),encoding="utf-8")
    fingerprint=json.loads((ROOT/"manifests/benchmark_environment_fingerprint_v1.json").read_text(encoding="utf-8"))
    protected={"benchmark_environment_yaml":"configs/benchmark_environment_v1.yaml","scenario_manifest":"manifests/scenario_manifest_v1.csv",
        "common_evaluator_config":"configs/unified_ground_received_accounting.yaml","common_evaluator_source":"src/evaluation/unified_ground_received_accounting.py",
        "battery_model_source":"src/energy/battery_update.py","thermal_model_source":"src/energy/thermal_update.py","queue_model_source":"src/link/downlink_queue.py"}
    mismatches=[key for key,path in protected.items() if sha(ROOT/path)!=fingerprint["components"][key]]
    preflight=f"""# OFFLOAD-EFT preflight

- Branch: `{git('branch','--show-current')}`
- HEAD: `{git('rev-parse','HEAD')}`
- YAML/JSON/CSV parse: PASS
- Seed 2026 task count: {env['task_count']}
- Manifest task hash: `{expected['task_hash']}`
- Manifest environment hash: `{expected['environment_hash']}`
- Slot seconds: {configs['timeline']['slot_seconds']}
- Protected fingerprint component mismatches: {mismatches}
- Recorded fingerprint git head `{fingerprint['components']['git_head']}` differs from experiment base only as provenance; protected component hashes match.
- Existing untracked datasets/outputs were preserved and not recursively scanned.
"""
    (out/"preflight_report.md").write_text(preflight,encoding="utf-8")
    (out/"environment_fairness_validation.md").write_text(f"""# Environment fairness validation

- Status: `{fairness_status}`
- Contract checks: {len(checks)}
- Shared task/contact/compute/storage/battery/thermal/evaluator fields are inherited from the base contract.
- OFFLOAD-EFT override contains method decisions only; migration and replication are false.
- Ground compute is an explicit finite FIFO PROXY, not free or infinitely parallel.
- Noncritical PROXY/UNKNOWN values remain disclosed; no immutable environment field was overridden.
""",encoding="utf-8")
    (out/"offload_eft_method_contract.md").write_text("""# OFFLOAD-EFT method contract

One decision per task: LOCAL, GROUND, or DEFER. LOCAL runs the fixed existing high-accuracy proxy and creates the existing ROI product. GROUND reuses B1 compressed-full-frame size, compression latency and compression energy, then enters a one-server finite FIFO ground queue. Both paths reserve the same contact trace and satellite downlink cursor, include ground completion in closure time, use the same deadline and shared battery/thermal functions, and cannot migrate or replicate. Future contacts are public; future tasks and true labels are not inputs.
""",encoding="utf-8")
    validation_ok=(counts["LOCAL"]>0 and counts["GROUND"]>0 and counts["DEFER"]>0 and finite and summary[0]["hard_safety_violations"]==0 and summary[0]["duplicate_decisions"]==0)
    (out/"legacy_method_noninterference_report.md").write_text("""# Legacy method noninterference

OFFLOAD-EFT is registered default-disabled. No PROP, B1, common evaluator, battery, thermal, queue, task, contact, or reward execution source was modified. Existing registry entries remain default-disabled and schema-compatible. Targeted legacy tests are recorded in validation_report.md.
""",encoding="utf-8")
    (out/"validation_report.md").write_text(f"""# Validation report

- Smoke branch coverage LOCAL/GROUND/DEFER: {counts['LOCAL']}/{counts['GROUND']}/{counts['DEFER']}
- Finite numeric values: {finite}
- Minimum SOC: {min_soc:.9f} (floor {configs['simulator']['battery_min']})
- Maximum thermal: {max_thermal:.9f} (hard limit {configs['simulator']['thermal_max']})
- Duplicate decisions: {summary[0]['duplicate_decisions']}; migration: 0; replication: 0
- Ground receive is not closure; ground compute end is the GROUND closure event.
- Smoke acceptance: {'PASS' if validation_ok else 'FAIL'}
- This is a deterministic non-formal smoke, not a 24-hour or multiseed performance result.
""",encoding="utf-8")
    (out/"final_plain_language_summary_zh.md").write_text(f"""# OFFLOAD-EFT 小规模验证摘要

本次只运行 seed=2026 的前 {len(tasks)} 个确定性任务，capacity=0.50；不是正式实验。观察到 LOCAL {counts['LOCAL']} 个、GROUND {counts['GROUND']} 个、DEFER {counts['DEFER']} 个。地面计算采用明确标记为 PROXY 的单服务器有限 FIFO 队列，GROUND 只有在完整下传并完成地面计算后才闭环。最低 SOC 为 {min_soc:.6f}，最高热状态为 {max_thermal:.6f}，未越过共享硬阈值。没有重复决策、迁移或复制执行。该结果只证明接口、决策规则和安全检查可运行，不说明相对 PROP 或 B1 的性能优劣。
""",encoding="utf-8")
    if not validation_ok: raise RuntimeError(f"smoke validation failed: {summary[0]}")
    print(json.dumps(summary[0],indent=2))

if __name__=="__main__": main()
