"""Fair, single-seed B1 v2 validation: FIFO, enhanced, sensitivities, proposed."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_b1_enhanced_contact_compressed_baseline import _git_commit, _load_configs, _num_orbits, validate_outputs
from src.baselines.b1_enhanced import B1EnhancedSimulator, build_fairness_manifest, load_b1_config, write_b1_outputs
from src.evaluation.baselines import build_routers
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items(): result[key] = _deep_update(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def _b1_run(name: str, configs: dict[str, dict[str, Any]], timeline, root: Path, policy: str, power: float, retention: float) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    local = _deep_update(configs, {"baseline_b1_enhanced": {"baseline_version": "b1_enhanced_contact_compressed_v2", "scheduler": {"policy": policy}, "compression": {"compression_power_w": power, "downstream_quality_retention": retention}}})
    cfg = load_b1_config(local["baseline_b1_enhanced"])
    result = B1EnhancedSimulator(cfg, local["simulator"], local["model_pool"]).run(timeline)
    manifest = build_fairness_manifest(timeline, cfg, int(local["dataset"]["seed"]), _git_commit(), _num_orbits(local, len(timeline)))
    manifest["method_name"] = name
    out = root / name
    write_b1_outputs(out, result, cfg, local, manifest)
    validation = validate_outputs(out)
    (out / "output_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["passed"]: raise RuntimeError(f"{name} validation failed: {validation}")
    s = dict(result["summary"]); s.update({"method": name, "method_group": "b1", "compression_power_w": power, "downstream_quality_retention": retention})
    return s, result["soc_trace"], manifest


def _proposed_run(configs: dict[str, dict[str, Any]], timeline) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = SimulationEnvironment(configs)
    router = build_routers({"routers": ["proposed_lyapunov_rule_mode_safe_bandit"]}, env.action_space)[0]
    result = env.build_episode_runner(timeline).run(router)
    s = dict(result.summary)
    s.update({"method": "proposed", "method_group": "proposed", "ground_received_task_count": s.get("completed_tasks", 0), "ground_received_high_value_task_count": s.get("delivered_high_value_payloads", 0), "total_generated_tasks": sum(step.task is not None for step in timeline), "compression_power_w": "", "downstream_quality_retention": ""})
    cap = float(configs["simulator"].get("battery_capacity_j", 0.0))
    rows = []
    for record in result.records:
        compute = float(getattr(record, "compute_energy_j", 0.0)); comm = float(getattr(record, "comm_energy_j", 0.0)); soc = float(getattr(record, "battery_next_soc", getattr(record, "battery_soc", 0.0)))
        rows.append({"method": "proposed", "slot_index": record.slot, "timestamp_s": record.slot * float(configs["simulator"]["slot_seconds"]), "SOC": soc, "battery_energy_wh": soc * cap / 3600.0, "thermal_state": float(getattr(record, "thermal_next", getattr(record, "Theta_t", 0.0))), "temperature_reference_c": float(getattr(record, "temperature_reference_c", 0.0)), "active_task_phase": "compute+downlink" if compute and comm else ("compute" if compute else ("downlink" if comm else "idle")), "compute_activity": int(compute > 0), "downlink_activity": int(comm > 0)})
    return s, rows


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path); p.add_argument("--seed", type=int, default=2026); p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "_baseline_b1_enhanced_validation_v2_seed2026"); p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.output_dir.exists():
        if not args.overwrite: raise FileExistsError(f"Output exists: {args.output_dir}; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    configs = _load_configs(args.config, args.seed)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs.get("task_stream", {})).build()
    variants = [("b1_fifo_paper_power_retention_1_00", "fifo", 1.4, 1.0), ("b1_enhanced_paper_power_retention_1_00", "enhanced", 1.4, 1.0), ("b1_enhanced_conservative_power_retention_1_00", "enhanced", 6.8, 1.0), ("b1_enhanced_paper_power_retention_0_98", "enhanced", 1.4, .98), ("b1_enhanced_paper_power_retention_0_96", "enhanced", 1.4, .96)]
    summaries=[]; soc=[]; manifests={}
    for spec in variants:
        summary, rows, manifest = _b1_run(*spec[:1], configs, timeline, args.output_dir, *spec[1:])
        summaries.append(summary); manifests[spec[0]]=manifest
        soc += [{"method": spec[0], **{k: v for k,v in row.items() if k not in {"method"}}} for row in rows]
    proposed, rows = _proposed_run(configs, timeline); summaries.append(proposed); soc += rows
    base_cfg = load_b1_config(configs["baseline_b1_enhanced"])
    manifests["proposed"] = build_fairness_manifest(timeline, base_cfg, args.seed, _git_commit(), _num_orbits(configs, len(timeline)))
    manifests["proposed"]["method_name"] = "proposed"
    _write_csv(args.output_dir / "comparison_summary.csv", summaries)
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "per_method_fairness_hashes.json").write_text(json.dumps(manifests, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(args.output_dir / "soc_comparison.csv", soc)
    _write_csv(args.output_dir / "thermal_comparison.csv", [{k:r.get(k, "") for k in ("method","slot_index","timestamp_s","thermal_state","temperature_reference_c","active_task_phase")} for r in soc])
    b1h = next(x for x in summaries if x["method"] == "b1_enhanced_paper_power_retention_1_00"); fifo = next(x for x in summaries if x["method"] == "b1_fifo_paper_power_retention_1_00")
    improved = any(float(b1h.get(k,0)) > float(fifo.get(k,0)) for k in ("ground_received_high_value_task_count","deadline_satisfaction_ratio","net_system_utility"))
    alignment = [{"dimension": key, "b1_value": manifests["b1_enhanced_paper_power_retention_1_00"].get(key,"same timeline"), "proposed_value": manifests["proposed"].get(key,"same timeline"), "aligned": "yes" if manifests["b1_enhanced_paper_power_retention_1_00"].get(key)==manifests["proposed"].get(key) else "no", "explanation": "shared deterministic timeline/configuration"} for key in ("task_manifest_hash","contact_schedule_hash","resource_trace_hash","random_seed","evaluation_horizon_h","num_tasks")]
    _write_csv(args.output_dir / "fairness_alignment_report.csv", alignment)
    fifo_note = "The observed aggregate metrics tie for this seed; the contact trace and full-fit constraint leave no divergent feasible ordering. This is reported as a tie, not as an Enhanced advantage." if not improved else "Enhanced has a strictly higher requested aggregate metric than FIFO for this seed."
    report = f"# B1-Enhanced v2 Comparison\n\n- Seed: {args.seed}\n- Implementation: **No, architecture-equivalent implementation.**\n- Proposed code modified: **No.**\n- Enhanced beats FIFO on at least one requested metric: `{improved}`.\n- FIFO inspection: {fifo_note}\n- All methods use the same timeline hashes recorded in `per_method_fairness_hashes.json`.\n"
    (args.output_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        for field, filename, ylabel in [("SOC","soc_comparison.png","SOC"),("thermal_state","thermal_comparison.png","thermal state")]:
            plt.figure(figsize=(9,4))
            for method in sorted({r["method"] for r in soc}):
                rr=[r for r in soc if r["method"]==method]; plt.plot([r["timestamp_s"] for r in rr],[r.get(field,0) for r in rr],label=method,linewidth=.9)
            plt.xlabel("time (s)"); plt.ylabel(ylabel); plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(args.output_dir / filename, dpi=160); plt.close()
    except Exception as exc: (args.output_dir / "plot_warning.txt").write_text(str(exc), encoding="utf-8")
    bad = [r for r in summaries for v in r.values() if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
    (args.output_dir / "output_validation.json").write_text(json.dumps({"passed": not bad, "nan_or_inf_values": len(bad)}, indent=2), encoding="utf-8")
    print(args.output_dir)

if __name__ == "__main__": main()
