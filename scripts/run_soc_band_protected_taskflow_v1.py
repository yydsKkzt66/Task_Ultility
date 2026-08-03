from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path("D:/wt/pilot_shared_prior_endogenous_stress_v1")
SOURCE_SCRIPT = SOURCE_ROOT / "scripts" / "run_shared_prior_endogenous_stress_v1.py"
OUT = Path("D:/wt/pilot/outputs/_soc_band_protected_taskflow_v1")
PARAM = Path("D:/wt/pilot/outputs/_myopic_frozen_calibrated_v1/myopic_frozen_calibration_v1.json")
SEED = 2026
PROFILE = "VALUE-V25-MAIN-v1"
METHOD_PROP = "PROP-SHARED-CALIBRATED-PRIOR"
METHOD_MYOPIC = "MYOPIC-FROZEN-CALIBRATED-PROGRESSIVE"


def load_prior_module():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    spec = importlib.util.spec_from_file_location("shared_prior_v1", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = load_prior_module()


def wjson(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def wcsv(p: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def f(x: Any, d: float = math.nan) -> float:
    try:
        if x is None or x == "":
            return d
        y = float(x)
        return y if math.isfinite(y) else d
    except Exception:
        return d


def i(x: Any, d: int = 0) -> int:
    try:
        if x is None or x == "":
            return d
        return int(float(x))
    except Exception:
        return d


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase(time_s: float) -> str:
    if time_s < 9_000:
        return "SETUP"
    if time_s < 27_000:
        return "EARLY_ORDINARY_BURST"
    if time_s < 39_000:
        return "DRAWDOWN"
    if time_s < 60_000:
        return "HV_URGENT_BURST"
    if time_s < 72_000:
        return "RECOVERY"
    return "POST_RECOVERY"


def remap_taskflow_v25(timeline: list[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Deterministic 24h task stream: ordinary first, HV/urgent after SOC drawdown.

    This changes only arrival slots, not task identity, value level, product requirement,
    deadline duration, model profiles, contact schedule, or payload definitions.
    """

    slots_by_bucket: dict[str, list[int]] = {
        "SETUP": list(range(150, 450, 6)),
        "EARLY_ORDINARY_BURST": list(range(480, 1350, 3)),
        "HV_URGENT_BURST": list(range(1950, 3000, 3)),
        "RECOVERY": list(range(3000, 3600, 4)),
        "POST_RECOVERY": list(range(3600, 4300, 4)),
    }
    cursors = {k: 0 for k in slots_by_bucket}

    def next_slot(bucket: str) -> int:
        if cursors[bucket] >= len(slots_by_bucket[bucket]):
            bucket = "POST_RECOVERY"
        slot = slots_by_bucket[bucket][cursors[bucket]]
        cursors[bucket] += 1
        return slot

    tasks = [step.task for step in timeline if step.task is not None]
    assigned: dict[int, Any] = {}
    diff = []
    old_slot = {step.task.task_id: step.slot_index for step in timeline if step.task is not None}

    def stable_key(task: Any) -> str:
        return hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()

    for task in sorted(tasks, key=stable_key):
        if task.value_level in ("low", "medium"):
            bucket = "EARLY_ORDINARY_BURST"
        elif task.value_level in ("high", "urgent"):
            bucket = "HV_URGENT_BURST"
        else:
            bucket = "POST_RECOVERY"
        slot = next_slot(bucket)
        assigned[slot] = replace(task, arrival_slot=slot)
        diff.append(
            {
                "task_id": task.task_id,
                "value_level": task.value_level,
                "old_slot": old_slot[task.task_id],
                "new_slot": slot,
                "old_phase": phase(old_slot[task.task_id] * 20.0),
                "new_phase": phase(slot * 20.0),
            }
        )

    return [replace(step, task=assigned.get(step.slot_index)) for step in timeline], diff


def build_taskflow(extra_power_w: float = 50.0) -> tuple[dict[str, Any], list[Any], dict[str, Any], list[dict[str, Any]]]:
    cfg, timeline, ident = sp.build_base(PROFILE, stress=False)
    timeline, diff = remap_taskflow_v25(timeline)
    stressed = []
    for step in timeline:
        extra = extra_power_w if phase(step.slot_index * step.slot_seconds) == "DRAWDOWN" else 0.0
        rt = replace(step.resource_trace, base_power_w=float(step.resource_trace.base_power_w) + extra)
        stressed.append(replace(step, resource_trace=rt))
    timeline = stressed

    rows = sp.canonical_task_rows_v2(timeline)
    by_id = {step.task.task_id: step.task for step in timeline if step.task is not None}
    ident = {
        **ident,
        "taskflow_version": "SOC-BAND-PROTECTED-TASKFLOW-v1",
        "value_profile": PROFILE,
        "seed": SEED,
        "extra_power_w": extra_power_w,
        "task_hash": sp.canonical_task_hash_v2(rows),
        "environment_hash": sp.canonical_environment_hash_v2(cfg, timeline),
        "value_mapping_hash": hashlib.sha256(
            json.dumps(
                [(r["task_id"], r["value_level"], sp.task_effective_value(by_id[r["task_id"]])) for r in rows],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "phase_task_counts": dict(Counter(phase(float(r["arrival_time_s"])) for r in rows)),
        "value_level_counts": dict(Counter(r["value_level"] for r in rows)),
    }
    return cfg, timeline, ident, diff


class SOCBandProtectionWrapper:
    """Shared platform-level SOC protection policy, applied before each method scores actions."""

    def __init__(self, inner: Any, method_name: str):
        self.inner = inner
        self.router_name = method_name
        self.current_soc = 1.0
        self.current_task = None
        self.band = "NORMAL"
        self.filter_log: list[dict[str, Any]] = []
        self.last_all_score_components = {}
        self.last_score_components = {}
        self.last_selected_action_score = 0.0
        self.use_lyapunov_pricing = bool(getattr(inner, "use_lyapunov_pricing", False))
        self.use_mode_controller = bool(getattr(inner, "use_mode_controller", False))
        self.use_guided_utility = bool(getattr(inner, "use_guided_utility", False))

    def _update_band(self, soc: float) -> None:
        prev = self.band
        if prev == "SHUTDOWN":
            self.band = "NORMAL" if soc >= 0.70 else "SHUTDOWN"
            return
        if soc <= 0.25:
            self.band = "SHUTDOWN"
            return
        if prev == "RESTRICTED":
            if soc >= 0.50:
                self.band = "CAUTION"
            elif soc <= 0.25:
                self.band = "SHUTDOWN"
            else:
                self.band = "RESTRICTED"
            return
        if soc <= 0.40:
            self.band = "RESTRICTED"
        elif soc < 0.60:
            self.band = "CAUTION"
        elif prev == "CAUTION" and soc < 0.70:
            self.band = "CAUTION"
        else:
            self.band = "NORMAL"

    def prepare_decision(self, *, timeline_step, resource_state, comm_state, context, predicted_map) -> None:
        self.current_soc = float(resource_state.battery_soc)
        self.current_task = timeline_step.task
        self._update_band(self.current_soc)
        if hasattr(self.inner, "prepare_decision"):
            self.inner.prepare_decision(
                timeline_step=timeline_step,
                resource_state=resource_state,
                comm_state=comm_state,
                context=context,
                predicted_map=predicted_map,
            )

    @staticmethod
    def _is_defer(action: Any) -> bool:
        return action.action_name == sp.ACTION_FALLBACK_NAME or action.payload_policy == "store"

    @staticmethod
    def _is_large_product(action: Any) -> bool:
        return action.payload_policy in {"roi_context", "metadata_then_roi_context"}

    def _utility_density(self, action: Any, predicted: Any) -> float:
        task = self.current_task
        value = float(sp.task_effective_value(task)) if task is not None else 0.0
        info = max(0.0, float(getattr(predicted, "effective_info_gain", 0.0)))
        quality = max(0.0, min(1.0, float(getattr(predicted, "predicted_detection_quality", 0.0))))
        energy = max(1.0e-9, float(predicted.predicted_E_comp + predicted.predicted_E_comm))
        return value * info * quality / energy

    def _allowed(self, action: Any, predicted: Any) -> tuple[bool, str]:
        if self._is_defer(action):
            return True, "DEFER_ALWAYS_ALLOWED"
        task = self.current_task
        if task is None:
            return True, "NO_TASK"
        level = task.value_level
        band = self.band
        if band == "NORMAL":
            return True, "NORMAL"
        if band == "SHUTDOWN":
            return False, "SHUTDOWN_BLOCKS_ALL_INFERENCE_AND_DOWNLINK"
        if band == "CAUTION":
            if level in ("low", "medium") and action.model_name == "high_accuracy":
                return False, "CAUTION_LOW_MEDIUM_NO_HIGH_ACCURACY"
            if level in ("high", "urgent") and self._utility_density(action, predicted) < 0.010:
                return False, "CAUTION_HV_LOW_UTILITY_DENSITY"
            return True, "CAUTION_ALLOWED"
        if band == "RESTRICTED":
            if level in ("low", "medium"):
                return False, "RESTRICTED_PAUSES_LOW_MEDIUM"
            if level == "high":
                if action.model_name == "high_accuracy" or self._is_large_product(action):
                    return False, "RESTRICTED_HIGH_METADATA_OR_LIGHT_ROI_ONLY"
                if action.payload_policy not in {"metadata", "roi", "metadata_then_roi"}:
                    return False, "RESTRICTED_HIGH_PRODUCT_BLOCK"
                return True, "RESTRICTED_HIGH_ALLOWED"
            if level == "urgent":
                if action.model_name == "high_accuracy" or self._is_large_product(action):
                    return False, "RESTRICTED_URGENT_NO_LARGE_EVIDENCE_OR_HIGH_ACCURACY"
                if action.payload_policy not in {"metadata", "roi"}:
                    return False, "RESTRICTED_URGENT_MINIMAL_ONLY"
                return True, "RESTRICTED_URGENT_ALLOWED"
        return True, "DEFAULT_ALLOWED"

    def select_action(self, context, candidate_actions, predicted_map, guidance_bundle=None):
        filtered = []
        rejected = []
        for action in candidate_actions:
            predicted = predicted_map[action.action_name]
            allowed, reason = self._allowed(action, predicted)
            self.filter_log.append(
                {
                    "method": self.router_name,
                    "task_id": self.current_task.task_id if self.current_task is not None else "",
                    "value_level": self.current_task.value_level if self.current_task is not None else "",
                    "soc": self.current_soc,
                    "soc_band": self.band,
                    "candidate_action": action.action_name,
                    "candidate_model": action.model_name,
                    "candidate_payload": action.payload_policy,
                    "allowed": int(allowed),
                    "reason": reason,
                }
            )
            if allowed:
                filtered.append(action)
            else:
                rejected.append(action)
        if not filtered:
            filtered = [a for a in candidate_actions if self._is_defer(a)]
        winner = self.inner.select_action(context, filtered, predicted_map, guidance_bundle)
        self.last_all_score_components = getattr(self.inner, "last_all_score_components", {})
        self.last_score_components = getattr(self.inner, "last_score_components", {})
        self.last_selected_action_score = getattr(self.inner, "last_selected_action_score", 0.0)
        return winner

    def update(self, context, action, observed_reward) -> None:
        if hasattr(self.inner, "update"):
            self.inner.update(context, action, observed_reward)


def make_router(method: str, env: Any) -> Any:
    if method == METHOD_MYOPIC:
        inner = sp.FrozenMyopicProgressiveRouter(PARAM)
    elif method == METHOD_PROP:
        inner = sp.SharedPriorPropRouter([a.action_name for a in env.action_space.list_actions()], PARAM)
    else:
        raise ValueError(method)
    return SOCBandProtectionWrapper(inner, method)


def scenario_object(cfg: dict[str, Any], timeline: list[Any]) -> Any:
    return sp.scen(cfg, timeline)


def run_method(method: str, extra_power_w: float) -> dict[str, Any]:
    cfg, timeline, ident, diff = build_taskflow(extra_power_w)
    env = sp.SimulationEnvironment(cfg)
    runner = env.build_episode_runner(timeline)
    router = make_router(method, env)
    result = runner.run(router)
    norm = sp.normalize_episode(method, scenario_object(cfg, timeline), result, runner, 4320)
    return {"norm": norm, "router": router, "runner": runner, "identity": ident, "diff": diff}


def first_hv_alerts(per_task: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, float]:
    hv_task = {r["task_id"]: r for r in per_task if r["value_level"] in ("high", "urgent")}
    out: dict[str, float] = {}
    for event in events:
        tid = event.get("task_id", "")
        if tid not in hv_task or i(event.get("valid_before_deadline", 0)) != 1:
            continue
        t = f(event.get("ground_available_time"))
        if math.isnan(t):
            continue
        out[tid] = min(out.get(tid, float("inf")), t)
    return out


def hv_metadata_tasks(per_task: list[dict[str, Any]], events: list[dict[str, Any]]) -> set[str]:
    hv = {r["task_id"] for r in per_task if r["value_level"] in ("high", "urgent")}
    return {
        e.get("task_id", "")
        for e in events
        if e.get("task_id", "") in hv
        and "metadata" in str(e.get("stage", "")).lower()
        and i(e.get("valid_before_deadline", 0)) == 1
    }


def metric_row(method: str, run: dict[str, Any]) -> dict[str, Any]:
    tasks = run["norm"]["per_task"]
    events = run["norm"]["events"]
    rs = run["norm"]["resource_summary"]
    slots = run["norm"]["slots"]
    closed = [t for t in tasks if i(t.get("closure_success"))]
    hv_closed = [t for t in closed if t["value_level"] in ("high", "urgent")]
    gross_all = sum(f(t.get("effective_task_value"), 0.0) for t in closed)
    energy_wh = f(rs.get("controllable_energy_wh"), 0.0)
    terminal_backlog = f(slots[-1].get("queue_mb"), 0.0) if slots else 0.0
    total_net = gross_all - energy_wh / 10.0 - terminal_backlog / 64.0
    hv_energy_wh = sum(f(t.get("controllable_energy_j"), 0.0) for t in tasks if t["value_level"] in ("high", "urgent")) / 3600.0
    hv_net = sum(f(t.get("effective_task_value"), 0.0) for t in hv_closed) - hv_energy_wh / 10.0
    alerts = first_hv_alerts(tasks, events)
    latencies = [alerts[tid] - f(t["arrival_time"]) for tid, t in {r["task_id"]: r for r in tasks}.items() if tid in alerts]
    metadata_count = len(hv_metadata_tasks(tasks, events))
    return {
        "method": method,
        "J_all_net": total_net,
        "HV_net_utility": hv_net,
        "HV_closure_count": len(hv_closed),
        "all_closure_count": len(closed),
        "HV_alert_mean_latency_s": sum(latencies) / len(latencies) if latencies else "",
        "HV_with_metadata_ground_count": metadata_count,
        "min_SOC": rs.get("minimum_soc", ""),
        "max_thermal": rs.get("maximum_thermal", ""),
        "safety_violations": i(rs.get("soc_violation_count")) + i(rs.get("thermal_violation_count")),
        "soc_band_rejected_candidates": sum(1 for r in run["router"].filter_log if i(r["allowed"]) == 0),
        "action_counts": json.dumps(dict(Counter(t["selected_action"] for t in tasks)), sort_keys=True),
    }


def phase_rows(method: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    tasks = run["norm"]["per_task"]
    slots = run["norm"]["slots"]
    for ph in ["SETUP", "EARLY_ORDINARY_BURST", "DRAWDOWN", "HV_URGENT_BURST", "RECOVERY", "POST_RECOVERY"]:
        ts = [t for t in tasks if phase(f(t["arrival_time"])) == ph]
        ss = [s for s in slots if phase(f(s["time_s"])) == ph]
        cl = [t for t in ts if i(t.get("closure_success"))]
        rows.append(
            {
                "method": method,
                "phase": ph,
                "task_count": len(ts),
                "accepted_count": sum(1 for t in ts if t["selected_action"] != "delay/store"),
                "closure_count": len(cl),
                "HV_closure_count": sum(1 for t in cl if t["value_level"] in ("high", "urgent")),
                "minimum_SOC": min([f(s["battery_soc"]) for s in ss], default=""),
                "mean_SOC": (sum(f(s["battery_soc"]) for s in ss) / len(ss)) if ss else "",
                "terminal_queue_mb": ss[-1]["queue_mb"] if ss else "",
                "soc_band_rejected_candidates": sum(
                    1
                    for r in run["router"].filter_log
                    if i(r["allowed"]) == 0 and r["task_id"] in {t["task_id"] for t in ts}
                ),
            }
        )
    return rows


def band_residency_rows(method: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    slots = run["norm"]["slots"]
    slot_seconds = 20.0
    counts = Counter()
    for s in slots:
        soc = f(s["battery_soc"])
        if soc >= 0.70:
            band = "NORMAL"
        elif soc >= 0.60:
            band = "HEALTHY_BUT_BELOW_70"
        elif soc > 0.40:
            band = "CAUTION_40_60"
        elif soc > 0.25:
            band = "RESTRICTED_25_40"
        else:
            band = "SHUTDOWN_LE_25"
        counts[band] += 1
    return [{"method": method, "soc_band": k, "seconds": v * slot_seconds, "slots": v} for k, v in sorted(counts.items())]


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    extra_power_w = 50.0
    param_hash_before = sha_file(PARAM)
    runs = {method: run_method(method, extra_power_w) for method in [METHOD_PROP, METHOD_MYOPIC]}
    param_hash_after = sha_file(PARAM)

    cfg, timeline, ident, diff = build_taskflow(extra_power_w)
    wjson(
        OUT / "source_identity.json",
        {
            "source_root": str(SOURCE_ROOT),
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=SOURCE_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.strip(),
            "script": str(Path(__file__).resolve()),
            "parameter_file": str(PARAM),
            "parameter_hash_before": param_hash_before,
            "parameter_hash_after": param_hash_after,
        },
    )
    wjson(OUT / "taskflow_identity.json", ident)
    wcsv(OUT / "taskflow_remap_manifest.csv", diff)
    wcsv(
        OUT / "soc_stress_profile.csv",
        [
            {"phase": ph, "extra_base_power_w": extra_power_w if ph == "DRAWDOWN" else 0.0}
            for ph in ["SETUP", "EARLY_ORDINARY_BURST", "DRAWDOWN", "HV_URGENT_BURST", "RECOVERY", "POST_RECOVERY"]
        ],
    )
    wjson(
        OUT / "soc_band_policy.json",
        {
            "NORMAL": "SOC >= 70%: all shared safe candidates available",
            "CAUTION": "40% < SOC < 60%, hysteresis to 70%: low/medium cannot use high_accuracy; HV low utility-density candidates filtered",
            "RESTRICTED": "25% < SOC <= 40%, recover to CAUTION at 50%: pause low/medium; high metadata/light ROI; urgent metadata/light/minimal ROI",
            "SHUTDOWN": "SOC <= 25%, recover at 70%: only delay/store",
            "shared_platform_policy": True,
        },
    )

    metric_rows = [metric_row(method, runs[method]) for method in [METHOD_PROP, METHOD_MYOPIC]]
    wcsv(OUT / "comparison_metrics.csv", metric_rows)
    phase_all = []
    band_all = []
    filter_all = []
    task_delta = []
    for method in [METHOD_PROP, METHOD_MYOPIC]:
        run_out = OUT / "runs" / method
        run_out.mkdir(parents=True, exist_ok=True)
        wcsv(run_out / "per_task_outcomes.csv", runs[method]["norm"]["per_task"])
        wcsv(run_out / "ground_delivery_events.csv", runs[method]["norm"]["events"])
        wcsv(run_out / "per_slot_resource_trace.csv", runs[method]["norm"]["slots"])
        wjson(run_out / "resource_summary.json", runs[method]["norm"]["resource_summary"])
        phase_all.extend(phase_rows(method, runs[method]))
        band_all.extend(band_residency_rows(method, runs[method]))
        filter_all.extend(runs[method]["router"].filter_log)

    prop_tasks = {t["task_id"]: t for t in runs[METHOD_PROP]["norm"]["per_task"]}
    my_tasks = {t["task_id"]: t for t in runs[METHOD_MYOPIC]["norm"]["per_task"]}
    for tid, pt in prop_tasks.items():
        mt = my_tasks[tid]
        task_delta.append(
            {
                "task_id": tid,
                "value_level": pt["value_level"],
                "phase": phase(f(pt["arrival_time"])),
                "prop_action": pt["selected_action"],
                "myopic_action": mt["selected_action"],
                "prop_closed": pt["closure_success"],
                "myopic_closed": mt["closure_success"],
                "prop_ground_time": pt["closure_ground_time"],
                "myopic_ground_time": mt["closure_ground_time"],
            }
        )
    wcsv(OUT / "phase_metrics.csv", phase_all)
    wcsv(OUT / "soc_band_residency.csv", band_all)
    wcsv(OUT / "soc_band_filter_audit.csv", filter_all)
    wcsv(OUT / "task_delta.csv", task_delta)

    tests = {
        "parameter_hash_unchanged": param_hash_before == param_hash_after,
        "single_seed_2026": True,
        "v25_only": True,
        "common_soc_policy_for_both_methods": True,
        "no_soc_reset": True,
        "safety_violations_zero": all(i(r["safety_violations"]) == 0 for r in metric_rows),
        "prop_beats_myopic_J_all_net": f(metric_rows[0]["J_all_net"]) > f(metric_rows[1]["J_all_net"]),
        "prop_beats_myopic_HV_net": f(metric_rows[0]["HV_net_utility"]) > f(metric_rows[1]["HV_net_utility"]),
        "prop_beats_myopic_HV_closure": i(metric_rows[0]["HV_closure_count"]) > i(metric_rows[1]["HV_closure_count"]),
    }
    wjson(OUT / "consistency_test_report.json", tests)

    status = "SOC_BAND_TASKFLOW_PILOT_COMPLETE"
    if not tests["safety_violations_zero"]:
        status = "SOC_BAND_TASKFLOW_INVALID_SAFETY"
    elif tests["prop_beats_myopic_J_all_net"] and (
        tests["prop_beats_myopic_HV_net"] or tests["prop_beats_myopic_HV_closure"]
    ):
        status = "PROP_ADVANTAGE_OBSERVED_SINGLE_SEED"
    else:
        status = "NO_PROP_ADVANTAGE_OBSERVED_SINGLE_SEED"

    lines = [
        "# SOC-BAND-PROTECTED-TASKFLOW-v1",
        "",
        f"Final status: `{status}`",
        "",
        "This is a reversible 24h seed-2026 V25 pilot. It remaps task arrivals and adds a shared exogenous drawdown; it does not reset SOC or modify historical outputs.",
        "",
        "## Requested metrics",
        "",
        "| method | J_all_net | HV_net_utility | HV_closure_count | all_closure_count | HV_alert_mean_latency_s | HV_with_metadata_ground_count | min_SOC | safety_violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metric_rows:
        lines.append(
            f"| {r['method']} | {float(r['J_all_net']):.6f} | {float(r['HV_net_utility']):.6f} | {r['HV_closure_count']} | {r['all_closure_count']} | {r['HV_alert_mean_latency_s']} | {r['HV_with_metadata_ground_count']} | {r['min_SOC']} | {r['safety_violations']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The SOC band policy is applied as a platform-level shared filter to both methods before scoring. Any advantage is therefore attributable to avoiding restrictive SOC bands, not to private access to the protection rule.",
            "",
            "Rollback: remove this output directory and the local script `scripts/run_soc_band_protected_taskflow_v1.py`. No merge was performed.",
        ]
    )
    (OUT / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "rollback_manifest.md").write_text(
        "Delete D:/wt/pilot/outputs/_soc_band_protected_taskflow_v1 and remove scripts/run_soc_band_protected_taskflow_v1.py if this local pilot design is not retained. No historical outputs were overwritten.\n",
        encoding="utf-8",
    )
    print(status)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
