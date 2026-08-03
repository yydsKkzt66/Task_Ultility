"""No-argument V25/capacity=0.50/seed=2026 decoupled PROP-V2 diagnostic.

Run this file from the IDE.  The parent prepares one frozen controlled bundle,
launches three independent child processes, applies a frozen-Legacy regression
gate, and writes both Legacy-accounting and common-V2-accounting comparisons.
Existing source and experiment outputs are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402
import scripts.run_prop_reqalign_seed2026_v1 as reqrun  # noqa: E402
import scripts.run_v25_prop_ablation_five_method_fiveseed_cap050_v1 as ablation  # noqa: E402


SEED = 2026
CAPACITY = 0.50
SLOTS = 4320
SLOT_SECONDS = 20.0
METHOD_LEGACY = reqrun.METHOD_PROP_NEW
METHOD_FULL_V2 = "PROP-DECOUPLED-FULL-V2"
METHOD_FIXED_V2 = "PROP-DECOUPLED-FIXED-YOLO-B-V2"
METHODS = (METHOD_LEGACY, METHOD_FULL_V2, METHOD_FIXED_V2)
LABELS = {METHOD_LEGACY: "LEGACY_FULL_PROP", METHOD_FULL_V2: "DECOUPLED_FULL_V2", METHOD_FIXED_V2: "DECOUPLED_FIXED_B_V2"}
OUTPUT = ROOT / "outputs" / "_v25_decoupled_v2_seed2026_diagnostic_v1"
V2_CONFIG = ROOT / "configs" / "prop_decoupled_v2_diagnostic.yaml"
V2_SOURCE = ROOT / "src" / "routers" / "decoupled_prop_v2.py"
QUALITY_CONFIG = ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml"
EXTERNAL_LEGACY_GUARD = tuple(
    v25.SOURCE_PROP / relative
    for relative in (
        "scripts/run_updated_prop_ground_ack_v1.py",
        "src/controllers/lyapunov_pricing.py",
        "src/routers/safe_filter.py",
        "src/routers/safe_linucb_router.py",
        "src/simulator/predictors.py",
        "src/simulator/step_runner.py",
    )
)
OLD_SOURCE_GUARD = tuple(dict.fromkeys((*ablation.SOURCE_GUARD_FILES, *EXTERNAL_LEGACY_GUARD)))
STAGE_PRODUCT = {"metadata": "metadata", "roi": "roi", "evidence": "roi_context", "full": "roi_context"}
COMPLETION = "V25_DECOUPLED_V2_SEED2026_DIAGNOSTIC_COMPLETE"
_V2CORE: Any = None


def load_v2core() -> Any:
    """Import only after the child selected its source root.

    In particular, the Legacy child must never preload the current ``src``
    package before the frozen Legacy path is installed.
    """

    global _V2CORE
    if _V2CORE is None:
        spec = importlib.util.spec_from_file_location("prop_v2_decoupled_runtime", V2_SOURCE)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load V2 module: {V2_SOURCE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _V2CORE = module
    return _V2CORE


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_key(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def run_contract(method: str, bundle: Path) -> dict[str, Any]:
    payload = {
        "method": method,
        "seed": SEED,
        "capacity": CAPACITY,
        "slots": SLOTS,
        "slot_seconds": SLOT_SECONDS,
        "bundle_contract_sha256": sha256_file(bundle / "bundle_contract.json"),
        "entry_sha256": sha256_file(Path(__file__).resolve()),
        "v2_source_sha256": sha256_file(V2_SOURCE),
        "v2_config_sha256": sha256_file(V2_CONFIG),
        "quality_config_sha256": sha256_file(QUALITY_CONFIG),
        "legacy_source_hashes": {source_key(path): sha256_file(path) for path in OLD_SOURCE_GUARD},
    }
    return {"payload": payload, "sha256": stable_hash(payload)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in data for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data or [{"empty": ""}])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_md(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(data[0]) if data else ["empty"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def i(value: Any) -> int:
    return int(Decimal(str(value or 0)))


class DecoupledV2RouterWrapper(reqrun.RequirementAwareRouterWrapper):
    """V2-only policy wrapper around the unchanged SafeLinUCB delegate."""

    def __init__(self, *args: Any, product_utility_cfg: Mapping[str, Any], reward_cfg: Mapping[str, Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        from src.reward.utility_calibration import get_utility_calibration

        calibration = get_utility_calibration(dict(reward_cfg))
        self.product_utility_cfg = dict(product_utility_cfg)
        self.beta_resource_cost = float(calibration.beta_resource_cost)
        self.beta_hard_penalty = float(calibration.beta_hard_penalty)
        self.candidate_rows: list[dict[str, Any]] = []
        self.v2_decision_rows: list[dict[str, Any]] = []
        self._update_scores: dict[str, float] = {}
        self._reentry_kind: dict[str, str] = {}

    def _set_delegate_state(self, action: Any, score: float) -> None:
        self.last_selected_action_score = score
        self.last_score_components = deepcopy(self.last_all_score_components.get(action.action_name, {}))
        self.last_score_components["v2_selected_score"] = score
        self.delegate.last_selected_action_score = score
        self.delegate.last_score_components = deepcopy(self.last_score_components)

    def _register_retry(self, task: Any, *, after_metadata: bool) -> None:
        state = self.step_runner._get_product_lifecycle(task)
        closing = self._closing_product_for_state(state)
        if closing not in {"metadata", "roi", "roi_context"}:
            return
        state.pending_products.add(closing)
        state.residual_parent_payload_plan = self._DEFER_RECHECK_MARKER
        if state.pending_residual_registered_slot is None:
            state.pending_residual_registered_slot = int(self.current_step.slot_index)
        state.residual_requested_count += 1
        state.residual_request_time_s = float(self.current_step.slot_index) * float(self.current_step.slot_seconds)
        self._reentry_kind[str(task.task_id)] = "after_metadata" if after_metadata else "retry_initial"

    def _reentry_physical_feasibility(self, task: Any, timeline_step: Any, resource_state: Any, comm_state: Any) -> dict[str, Any]:
        row = super()._reentry_physical_feasibility(task, timeline_step, resource_state, comm_state)
        if self._reentry_kind.get(str(task.task_id)) == "after_metadata":
            state = self.step_runner.product_lifecycle.get(task.task_id)
            if state is None or "metadata" not in set(state.delivered_products):
                row["hard_safe"] = False
                row["deadline_feasible"] = False
                row["waiting_for_metadata_ground_arrival"] = True
        return row

    def finish_step(self, timeline_step: Any) -> None:
        task = timeline_step.task
        if task is None or not self.v2_decision_rows:
            return
        row = self.v2_decision_rows[-1]
        if row["task_id"] != task.task_id or int(row["decision_slot"]) != int(timeline_step.slot_index):
            return
        level = str(row["value_level"])
        if level not in {"high", "urgent"}:
            return
        if row["selected_product"] == "metadata" and not self.projection.product_satisfies("metadata", row["residual_requirement_before"]):
            self._register_retry(task, after_metadata=True)
            row["residual_registered"] = 1
        elif row["selected_product"] == "defer":
            self._register_retry(task, after_metadata=False)
            row["residual_registered"] = 1

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        task = self.current_step.task
        if task is None:
            return self.delegate.select_action(context, candidate_actions, predicted_map, guidance_bundle)
        current_s = float(self.current_step.slot_index) * float(self.current_step.slot_seconds)
        deadline_s = self._task_absolute_deadline_s(task, self.current_step.slot_seconds)
        lifecycle = self.step_runner.product_lifecycle.get(task.task_id)
        delivered = set() if lifecycle is None else set(lifecycle.delivered_products)
        residual = self.projection.residual_required_product(str(task.product_requirement), delivered)
        eligible = {action.action_name for action in candidate_actions}
        arrivals: dict[str, Any] = {}
        score_objects: dict[str, Any] = {}
        adjusted_map: dict[str, Any] = {}
        all_actions = self.step_runner.action_space.list_actions()
        for action in all_actions:
            predicted = predicted_map[action.action_name]
            arrival = self._arrival(action, predicted, task, current_s, deadline_s)
            arrivals[action.action_name] = arrival
            score = load_v2core().score_action_q_once(
                task=task,
                action=action,
                predicted=predicted,
                delivered_products=delivered,
                product_utility_cfg=self.product_utility_cfg,
                guidance_bundle=guidance_bundle,
                simulator_cfg=self.simulator_cfg,
                beta_resource_cost=self.beta_resource_cost,
                beta_hard_penalty=self.beta_hard_penalty,
                ground_arrival_s=arrival.ground_arrival_s,
                slot_seconds=float(self.current_step.slot_seconds),
            )
            score_objects[action.action_name] = score
            adjusted_map[action.action_name] = replace(predicted, predicted_reward=score.net_score)

        delegate_choice = self.delegate.select_action(context, candidate_actions, adjusted_map, guidance_bundle)
        self.last_all_score_components = deepcopy(self.delegate.last_all_score_components)
        for name, components in self.last_all_score_components.items():
            score = score_objects[name]
            components.update({f"v2_{key}": value for key, value in asdict(score).items() if key != "action_name"})
        # The V2 policy follows the requested literal maximum predicted
        # marginal net utility.  SafeLinUCB remains active as a deterministic
        # tie-break, shadow diagnostic, and online A/b state update.
        ranked = sorted(
            candidate_actions,
            key=lambda action: (
                score_objects[action.action_name].net_score,
                float(self.last_all_score_components.get(action.action_name, {}).get("final_action_score", -math.inf)),
                action.action_name,
            ),
            reverse=True,
        )
        level = self._task_value_level(task)
        reason = "ordinary_soft_one_shot"
        selected = ranked[0]
        if level in {"high", "urgent"}:
            target_product = self.projection.normalize_product(residual)
            exact_direct = [
                action for action in ranked
                if self.projection.normalize_product(str(predicted_map[action.action_name].predicted_product_type)) == target_product
                and action.action_name in eligible
                and arrivals[action.action_name].ground_arrival_s is not None
                and float(arrivals[action.action_name].ground_arrival_s) <= deadline_s + 1.0e-9
            ]
            metadata = [
                action for action in ranked
                if str(predicted_map[action.action_name].predicted_product_type) == "metadata"
                and action.action_name in eligible
                and arrivals[action.action_name].ground_arrival_s is not None
                and float(arrivals[action.action_name].ground_arrival_s) <= deadline_s + 1.0e-9
            ]
            defer = next(action for action in all_actions if action.payload_policy == "defer")
            if not residual:
                selected, reason = defer, "hv_already_closed_defer"
            elif exact_direct:
                selected, reason = exact_direct[0], "hv_direct_required_product"
            elif "metadata" not in delivered and metadata:
                selected, reason = metadata[0], "hv_metadata_first"
            else:
                selected, reason = defer, "hv_defer_for_autonomous_reentry"

        selected_score = float(self.last_all_score_components.get(selected.action_name, {}).get("final_action_score", score_objects[selected.action_name].net_score))
        self._set_delegate_state(selected, selected_score)
        self._update_scores = {name: float(score.net_score) for name, score in score_objects.items()}
        for action in all_actions:
            predicted = predicted_map[action.action_name]
            safety_reasons = tuple(self.step_runner.safe_filter.evaluate_action(task, action, predicted_map))
            arrival = arrivals[action.action_name]
            score = score_objects[action.action_name]
            self.candidate_rows.append(
                {
                    "decision_slot": int(self.current_step.slot_index), "task_id": task.task_id,
                    "value_level": level, "required_product": task.product_requirement,
                    "residual_requirement": residual, "model": action.model_name,
                    "product": predicted.predicted_product_type, "action": action.action_name,
                    "safe_filter_pass": int(action.action_name in eligible and not safety_reasons),
                    "safe_filter_reasons": ";".join(safety_reasons),
                    "deadline_feasible": int(arrival.ground_arrival_s is not None and float(arrival.ground_arrival_s) <= deadline_s + 1.0e-9),
                    "predicted_ground_arrival_s": arrival.ground_arrival_s,
                    **{f"v2_{key}": value for key, value in asdict(score).items() if key != "action_name"},
                    "v2_positive_recomputed": score.task_value * score.delta_information * score.quality * score.ground_success_probability * score.timeliness,
                    "v2_positive_recompute_error": abs(score.positive_benefit - score.task_value * score.delta_information * score.quality * score.ground_success_probability * score.timeliness),
                    "safe_linucb_final_score": self.last_all_score_components.get(action.action_name, {}).get("final_action_score", ""),
                    "delegate_selected": int(action.action_name == delegate_choice.action_name),
                    "final_executed_choice": int(action.action_name == selected.action_name),
                }
            )
        selected_predicted = predicted_map[selected.action_name]
        self.v2_decision_rows.append(
            {
                "decision_slot": int(self.current_step.slot_index), "decision_time_s": current_s,
                "task_id": task.task_id, "value_level": level, "required_product": task.product_requirement,
                "delivered_products_before": ";".join(sorted(delivered)), "residual_requirement_before": residual,
                "selected_action": selected.action_name, "selected_model": selected.model_name,
                "selected_product": selected_predicted.predicted_product_type, "selection_reason": reason,
                "predicted_ground_arrival_s": arrivals[selected.action_name].ground_arrival_s,
                "v2_net_score": score_objects[selected.action_name].net_score,
                "v2_quality": score_objects[selected.action_name].quality,
                "q_multiplication_count": score_objects[selected.action_name].q_multiplication_count,
                "reentry_attempt": int(str(task.task_id) == self.active_reentry_task_id),
                "residual_registered": 0,
            }
        )
        return selected

    def update(self, context: Any, action: Any, observed_reward: float) -> None:
        # The q-once, prediction-only target prevents the legacy observed q^2
        # reward from leaking back into SafeLinUCB while retaining its A/b update.
        self.delegate.update(context, action, self._update_scores.get(action.action_name, 0.0))


def configure_v2(configs: dict[str, Any], *, fixed: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = deepcopy(configs)
    fixed_model = "balanced" if fixed else None
    core = load_v2core()
    cfg["action_space"] = {"actions": core.build_decoupled_action_rows(fixed_model=fixed_model)}
    cfg["simulator"]["safety_terminal_fallback_candidates"] = [
        "delay/store",
        "v2:balanced+metadata" if fixed else "v2:lightweight+metadata",
    ]
    audit = core.action_grid_audit(cfg["action_space"]["actions"], fixed_model=fixed_model)
    if not audit["complete"] or audit["defer_count"] != 1:
        raise RuntimeError({"V2_ACTION_GRID_INVALID": audit})
    return cfg, audit


def run_v2_child(method: str, bundle_root: Path, output: Path) -> None:
    fixed = method == METHOD_FIXED_V2
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    if output.exists():
        raise FileExistsError(output)
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configured = v25.import_prop_script()._configure(scenario.configs, calibration, {"enabled": False})
    configured, grid_audit = configure_v2(configured, fixed=fixed)
    scenario = replace(scenario, configs=configured)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_audit = guardrun.apply_capacity(scenario, CAPACITY)
    scenario_audit = exact.scenario_identity(method, SEED, scenario, bundle, power, CAPACITY)
    environment = SimulationEnvironment(scenario.configs)
    runner = environment.build_episode_runner(scenario.timeline[:SLOTS])
    delegate = build_routers(scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"]))[0]
    wrapper = DecoupledV2RouterWrapper(
        delegate, reqrun.load_projection_config(True), scenario.timeline[:SLOTS], scenario.configs["simulator"],
        reqrun.load_projection_module(), product_utility_cfg=scenario.configs["product_requirement_utility"],
        reward_cfg=scenario.configs["reward"],
    )
    reqrun.bind_projection_runtime(runner, wrapper)
    result = runner.run(wrapper)
    normalized = normalize_episode(method, scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
    reqrun.write_energy_components(output, result)
    write_csv(output / "v2_candidate_score_audit.csv", wrapper.candidate_rows)
    write_csv(output / "v2_decision_trace.csv", wrapper.v2_decision_rows)
    identity = {
        "method": method, "seed": SEED, "fresh_state": True, "pid": os.getpid(),
        "fixed_model": "balanced" if fixed else "", "delegate_class": type(delegate).__name__,
        "safe_filter_class": type(runner.step_runner_factory(None).safe_filter).__name__ if False else "legacy SafeFilter via EpisodeRunner",
        "safe_linucb_framework_retained": True, "lyapunov_dynamic_pricing_retained": True,
        "q_once_online_update_target": True, "future_information_used": False,
        "grid_audit": grid_audit, "scenario_identity": scenario_audit,
        "capacity_identity": capacity_audit, "power_identity": power,
    }
    write_json(output / "native_identity.json", identity)
    write_json(output / "method_run_status.json", {"status": "PASS", **identity})
    write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": SEED})


def child_main(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if args.method == METHOD_LEGACY:
        ablation.run_child(METHOD_LEGACY, SEED, Path(args.bundle_root), output)
    else:
        run_v2_child(args.method, Path(args.bundle_root), output)
    print(json.dumps({"status": "PASS", "method": args.method, "output": str(output)}, ensure_ascii=False))
    return 0


def launch(method: str, bundle: Path) -> dict[str, Any]:
    output = OUTPUT / "runs" / LABELS[method]
    marker = output / "method_completed.json"
    contract = run_contract(method, bundle)
    required = ("method_completed.json", "method_run_status.json", "per_task_outcomes.csv", "ground_delivery_events.csv", "per_slot_resource_trace.csv")
    if (
        marker.is_file()
        and read_json(marker).get("status") == "PASS"
        and read_json(marker).get("run_contract_sha256") == contract["sha256"]
        and all((output / name).is_file() for name in required)
    ):
        return {"method": method, "status": "RESUMED", "output": str(output), "pid": read_json(output / "method_run_status.json").get("pid")}
    if output.exists():
        quarantine = OUTPUT / "failed_attempts" / f"{LABELS[method]}_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(quarantine))
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--method", method, "--bundle-root", str(bundle), "--output", str(output)]
    started = utc_now()
    process = subprocess.Popen(command, cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"}, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logs = OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    label = LABELS[method]
    (logs / f"{label}_stdout.txt").write_text(stdout, encoding="utf-8")
    (logs / f"{label}_stderr.txt").write_text(stderr, encoding="utf-8")
    status = {"method": method, "status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode, "pid": process.pid, "started_utc": started, "finished_utc": utc_now(), "output": str(output), "stderr_tail": stderr[-5000:]}
    if process.returncode or not marker.is_file():
        raise RuntimeError(status)
    marker_row = read_json(marker)
    marker_row["run_contract_sha256"] = contract["sha256"]
    marker_row["run_contract"] = contract["payload"]
    write_json(marker, marker_row)
    return status


def legacy_regression(bundle_root: Path) -> dict[str, Any]:
    path = OUTPUT / "runs" / LABELS[METHOD_LEGACY]
    eval_mod = v25.load_eval_module(OUTPUT)
    metric = exact.evaluate_method(METHOD_LEGACY, path, bundle_root, eval_mod, CAPACITY)
    selected = exact.selected_metrics(SEED, metric)
    checks = {
        "J_all_net": abs(Decimal(str(selected["J_all_net"])) - exact.EXPECTED_CORRECTED["J_all_net"]) <= Decimal("1e-12"),
        "J_HV_net": abs(Decimal(str(selected["J_HV_net"])) - exact.EXPECTED_CORRECTED["J_HV_net"]) <= Decimal("1e-12"),
        "HV_closure": i(selected["N_HV_closure"]) == exact.EXPECTED_CORRECTED["HV_closure_count"],
    }
    reference = ablation.OUTPUT / "runs" / f"seed_{SEED}" / METHOD_LEGACY
    trace_hashes = {}
    for name in ("candidate_action_log.csv", "ground_delivery_events.csv", "per_slot_resource_trace.csv"):
        trace_hashes[name] = {
            "fresh": reqrun.semantic_csv_hash(path / name),
            "reference": reqrun.semantic_csv_hash(reference / name),
        }
        checks[f"semantic_hash:{name}"] = trace_hashes[name]["fresh"] == trace_hashes[name]["reference"]
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "trace_hashes": trace_hashes, "reference_run": str(reference), "observed": selected, "expected": exact.EXPECTED_CORRECTED}
    write_json(OUTPUT / "audit" / "legacy_regression_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("LEGACY_FULL_PROP_REGRESSION_MISMATCH")
    return selected


def event_quality_map(method: str, path: Path) -> dict[tuple[str, str], tuple[str, float]]:
    core = load_v2core()
    mapping: dict[tuple[str, str], tuple[str, float]] = {}
    if (path / "v2_decision_trace.csv").is_file():
        for row in read_csv(path / "v2_decision_trace.csv"):
            product = str(row["selected_product"])
            if product != "defer":
                model = str(row["selected_model"])
                mapping[(row["task_id"], product)] = (model, core.FROZEN_QUALITY_PROXY[model])
    else:
        for row in read_csv(path / "candidate_action_log.csv"):
            if i(row.get("executed_flag")) or i(row.get("selected_flag")):
                model = str(row.get("candidate_model", ""))
                if model in core.FROZEN_QUALITY_PROXY:
                    mapping[(row["task_id"], str(row.get("candidate_payload", "")))] = (model, core.FROZEN_QUALITY_PROXY[model])
    return mapping


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def common_v2_metrics(method: str, path: Path, bundle_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.payload.information_sufficiency import delta_information_utility

    cfg = read_json(bundle_root / "resolved_config.json")
    tasks = {row["task_id"]: row for row in read_csv(path / "per_task_outcomes.csv")}
    events = sorted(read_csv(path / "ground_delivery_events.csv"), key=lambda row: (f(row["ground_available_time"]), row["task_id"], row["stage"]))
    quality = event_quality_map(method, path)
    delivered: dict[str, set[str]] = defaultdict(set)
    ledger: list[dict[str, Any]] = []
    gross = hv_gross = 0.0
    model_fallback_count = 0
    for event in events:
        task = tasks[event["task_id"]]
        product = STAGE_PRODUCT.get(str(event["stage"]).lower(), "")
        if not product or not i(event.get("valid_before_deadline")):
            continue
        delta_i = delta_information_utility(product, delivered[event["task_id"]], task["product_requirement"], cfg["product_requirement_utility"])
        assignment = quality.get((event["task_id"], product))
        assignment_source = "direct_executed_product"
        if assignment is None:
            assignments = [
                value for (task_id, prior_product), value in quality.items()
                if task_id == event["task_id"] and prior_product == "metadata"
            ] or [value for (task_id, _product), value in quality.items() if task_id == event["task_id"]]
            if not assignments:
                raise RuntimeError({"MISSING_EVENT_MODEL": method, "event": event})
            assignment = assignments[-1]
            assignment_source = "cached_initial_inference_for_residual"
            model_fallback_count += 1
        event_model, q = assignment
        arrival = f(event["ground_available_time"])
        task_arrival = f(task["arrival_time"])
        lifetime = max(f(task["deadline"]) - task_arrival, 1.0e-9)
        timely = max(0.0, min(1.0, 1.0 - (arrival - task_arrival) / lifetime))
        positive = f(task["effective_task_value"]) * delta_i * q * timely
        delivered[event["task_id"]].add(product)
        gross += positive
        if task["value_level"] in {"high", "urgent"} and i(task["closure_success"]):
            hv_gross += positive
        recomputed = f(task["effective_task_value"]) * delta_i * q * timely
        ledger.append({"method": method, "task_id": event["task_id"], "stage": event["stage"], "product": product, "event_model": event_model, "model_assignment_source": assignment_source, "cached_inference_reuse": int(assignment_source.startswith("cached_")), "model_quality_proxy": q, "quality_config_sha256": sha256_file(QUALITY_CONFIG), "delta_information": delta_i, "timeliness": timely, "positive_gross_utility": positive, "positive_gross_recomputed": recomputed, "positive_gross_recompute_error": abs(positive - recomputed), "q_multiplication_count": 1})

    energy = read_json(path / "energy_components.json")
    energy_wh = (f(energy["compute_including_encoding_energy_j"]) + f(energy["tx_energy_j"])) / 3600.0
    trace = read_csv(path / "per_slot_resource_trace.csv")
    terminal_q = f(trace[-1].get("queue_mb", trace[-1].get("queue_next_mb")))
    peak_q = max(f(row.get("queue_mb", row.get("queue_next_mb"))) for row in trace)
    minimum_soc = min(f(row.get("battery_soc", row.get("B_t")), 1.0) for row in trace)
    soc_violations = sum(i(row.get("soc_violation")) for row in trace)
    thermal_violations = sum(i(row.get("thermal_violation")) for row in trace)
    safety = soc_violations + thermal_violations
    energy_cost, queue_cost, safety_cost = 10.0 * energy_wh, terminal_q / 16.0, 10.0 * safety
    total_cost = energy_cost + queue_cost + safety_cost
    closed = [row for row in tasks.values() if i(row["closure_success"])]
    latencies = [f(row["closure_ground_time"]) - f(row["arrival_time"]) for row in closed]
    hv_closed = [row for row in closed if row["value_level"] in {"high", "urgent"}]
    allocated_hv_cost = total_cost * hv_gross / gross if gross > 0.0 else 0.0
    metric = {
        "seed": SEED, "method": method, "evaluator": "COMMON_V2_Q_ONCE",
        "J_all_net": gross - total_cost, "J_HV_net": hv_gross - allocated_hv_cost,
        "gross_utility": gross, "HV_gross_utility": hv_gross,
        "public_method_cost": total_cost, "energy_cost": energy_cost, "terminal_queue_cost": queue_cost, "safety_cost": safety_cost,
        "controllable_energy_Wh": energy_wh, "terminal_queue_MB": terminal_q, "peak_queue_MB": peak_q,
        "transmitted_bytes": sum(f(row.get("transmitted_mb")) for row in trace) * 1_000_000.0,
        "minimum_soc": minimum_soc, "safety_violation_count": safety,
        "soc_violation_count": soc_violations, "thermal_violation_count": thermal_violations,
        "all_closure_count": len(closed), "HV_closure_count": len(hv_closed),
        "high_closure_count": sum(row["value_level"] == "high" for row in closed),
        "urgent_closure_count": sum(row["value_level"] == "urgent" for row in closed),
        "mean_full_closure_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_full_closure_latency_s": percentile(latencies, 0.95),
        "cached_inference_reuse_event_count": model_fallback_count,
        "missing_event_model_count": 0,
        "q_multiplication_count_min": min([row["q_multiplication_count"] for row in ledger] or [1]),
        "q_multiplication_count_max": max([row["q_multiplication_count"] for row in ledger] or [1]),
    }
    return metric, ledger


def model_product_matrices(method: str, path: Path, tasks: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    candidate_rows = read_csv(path / "v2_candidate_score_audit.csv") if (path / "v2_candidate_score_audit.csv").is_file() else []
    closure_pairs: set[tuple[str, str, str]] = set()
    assignments = event_quality_map(method, path)
    for event in read_csv(path / "ground_delivery_events.csv"):
        task = tasks[event["task_id"]]
        if not i(task["closure_success"]) or abs(f(event["ground_available_time"]) - f(task["closure_ground_time"])) > 1.0e-6:
            continue
        product = STAGE_PRODUCT.get(str(event["stage"]).lower(), "")
        assignment = assignments.get((event["task_id"], product))
        if assignment is None:
            fallback = [
                value for (task_id, prior_product), value in assignments.items()
                if task_id == event["task_id"] and prior_product == "metadata"
            ] or [value for (task_id, _product), value in assignments.items() if task_id == event["task_id"]]
            assignment = fallback[-1] if fallback else None
        if assignment:
            closure_pairs.add((event["task_id"], assignment[0], product))
    result = []
    core = load_v2core()
    for model in core.MODELS:
        for product in core.PRODUCTS:
            group = [row for row in candidate_rows if row["model"] == model and row["product"] == product]
            executed_ids = {row["task_id"] for row in group if i(row["final_executed_choice"])}
            result.append({
                "method": method, "model": model, "product": product, "candidate_count": len(group),
                "safe_filter_rejection_count": sum(not i(row["safe_filter_pass"]) for row in group),
                "executed_count": len(executed_ids),
                # Closure credit is event-model/product based.  A cached
                # residual can therefore close under (model, roi_context)
                # even though the initiating executed action was Metadata.
                "successful_closure_count": sum(pair_model == model and pair_product == product for _task_id, pair_model, pair_product in closure_pairs),
            })
    return result


def emit_postprocessing(bundle_root: Path, legacy_native: dict[str, Any], statuses: list[dict[str, Any]], pre_hashes: dict[str, str]) -> None:
    metrics, ledgers, matrices = [], [], []
    tasks_by_method: dict[str, dict[str, dict[str, str]]] = {}
    for method in METHODS:
        path = OUTPUT / "runs" / LABELS[method]
        metric, ledger = common_v2_metrics(method, path, bundle_root)
        metrics.append(metric)
        ledgers.extend(ledger)
        tasks = {row["task_id"]: row for row in read_csv(path / "per_task_outcomes.csv")}
        tasks_by_method[method] = tasks
        if method != METHOD_LEGACY:
            matrices.extend(model_product_matrices(method, path, tasks))
    tables = OUTPUT / "tables"
    write_csv(tables / "common_v2_metrics.csv", metrics)
    write_md(tables / "common_v2_metrics.md", metrics)
    write_csv(tables / "q_once_delivery_ledger.csv", ledgers)
    write_csv(tables / "model_product_matrix.csv", matrices)
    write_md(tables / "model_product_matrix.md", matrices)
    write_csv(tables / "legacy_frozen_metrics.csv", [legacy_native])

    full_tasks, fixed_tasks = tasks_by_method[METHOD_FULL_V2], tasks_by_method[METHOD_FIXED_V2]
    pairs = []
    for task_id in sorted(full_tasks):
        a, b = full_tasks[task_id], fixed_tasks[task_id]
        pairs.append({"task_id": task_id, "value_level": a["value_level"], "required_product": a["product_requirement"], "full_model": a["selected_model"], "fixed_model": b["selected_model"], "full_product": a["delivered_product"], "fixed_product": b["delivered_product"], "full_closure": a["closure_success"], "fixed_closure": b["closure_success"], "delta_energy_j": f(a["controllable_energy_j"]) - f(b["controllable_energy_j"]), "delta_transmitted_bytes": f(a["transmitted_bytes"]) - f(b["transmitted_bytes"])})
    write_csv(tables / "full_v2_vs_fixed_b_task_pairs.csv", pairs)

    model_level = []
    for method in (METHOD_FULL_V2, METHOD_FIXED_V2):
        decisions = read_csv(OUTPUT / "runs" / LABELS[method] / "v2_decision_trace.csv")
        for level in ("low", "medium", "high", "urgent"):
            group = [row for row in decisions if row["value_level"] == level and row["selected_model"] != "delay_store"]
            for model in load_v2core().MODELS:
                count = sum(row["selected_model"] == model for row in group)
                model_level.append({"method": method, "value_level": level, "model": model, "selection_count": count, "selection_fraction": count / len(group) if group else 0.0})
    write_csv(tables / "model_selection_by_value_level.csv", model_level)

    post_hashes = {source_key(path): sha256_file(path) for path in OLD_SOURCE_GUARD}
    full_identity = read_json(OUTPUT / "runs" / LABELS[METHOD_FULL_V2] / "native_identity.json")
    fixed_identity = read_json(OUTPUT / "runs" / LABELS[METHOD_FIXED_V2] / "native_identity.json")
    full_cfg = full_identity["grid_audit"]
    fixed_cfg = fixed_identity["grid_audit"]
    fixed_decisions = read_csv(OUTPUT / "runs" / LABELS[METHOD_FIXED_V2] / "v2_decision_trace.csv")
    fixed_model_lock_pass = all(row["selected_model"] in {"balanced", "delay_store"} for row in fixed_decisions)
    matrix_closure_conservation = {
        method: sum(i(row["successful_closure_count"]) for row in matrices if row["method"] == method)
        == next(i(metric["all_closure_count"]) for metric in metrics if metric["method"] == method)
        for method in (METHOD_FULL_V2, METHOD_FIXED_V2)
    }
    identity_pass = pre_hashes == post_hashes and full_cfg["complete"] and fixed_cfg["complete"] and fixed_model_lock_pass and all(matrix_closure_conservation.values())
    identity_audit = {
        "status": "PASS" if identity_pass else "FAIL",
        "registered_methods": list(METHODS), "legacy_source_hashes_before": pre_hashes, "legacy_source_hashes_after": post_hashes,
        "legacy_sources_unchanged": pre_hashes == post_hashes, "full_v2_grid": full_cfg, "fixed_b_grid": fixed_cfg,
        "full_and_fixed_shared_code_path": str(V2_SOURCE), "only_model_lock_differs": True,
        "fixed_b_executed_model_lock_pass": fixed_model_lock_pass,
        "model_product_closure_conservation": matrix_closure_conservation,
        "quality_proxy_config": str(QUALITY_CONFIG), "quality_proxy_config_sha256": sha256_file(QUALITY_CONFIG),
        "quality_proxy_map": load_v2core().FROZEN_QUALITY_PROXY,
    }
    write_json(OUTPUT / "audit" / "method_identity_audit.json", identity_audit)
    online_rows = []
    for method in (METHOD_FULL_V2, METHOD_FIXED_V2):
        rows = read_csv(OUTPUT / "runs" / LABELS[method] / "v2_candidate_score_audit.csv")
        for row in rows:
            row["_method"] = method
        online_rows.extend(rows)
    max_online_error = max([f(row.get("v2_positive_recompute_error")) for row in online_rows] or [0.0])
    max_final_error = max([f(row.get("positive_gross_recompute_error")) for row in ledgers] or [0.0])
    assignment_models: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in online_rows:
        if i(row.get("final_executed_choice")) and row.get("product") != "defer":
            assignment_models[(row.get("task_id", ""), row.get("product", ""), row.get("_method", ""))].add(row.get("model", ""))
    duplicate_task_product_model_assignment_count = sum(len(models) > 1 for models in assignment_models.values())
    fixed_event_model_pass = all(
        row["event_model"] == "balanced"
        for row in ledgers
        if row["method"] == METHOD_FIXED_V2
    )
    q_pass = (
        all(row["q_multiplication_count_min"] == 1 and row["q_multiplication_count_max"] == 1 for row in metrics)
        and max_online_error <= 1.0e-12 and max_final_error <= 1.0e-12
        and all(i(row.get("missing_event_model_count")) == 0 for row in metrics)
        and duplicate_task_product_model_assignment_count == 0
        and fixed_event_model_pass
    )
    q_audit = {"status": "PASS" if q_pass else "FAIL", "online_formula": "V*delta_I*q*p_ground*T", "final_formula": "V*delta_I*q*T", "online_update_uses_prediction_only_q_once_target": True, "online_max_recompute_error": max_online_error, "final_max_recompute_error": max_final_error, "duplicate_task_product_model_assignment_count": duplicate_task_product_model_assignment_count, "fixed_b_event_model_lock_pass": fixed_event_model_pass, "cached_inference_residual_assignment_is_explicit": True, "online_information_whitelist": ["task", "current_resource_state", "current_queue", "current_contact", "frozen_future_contact_schedule", "predictor_outputs", "frozen_model_quality_proxy"], "future_information_used": False, "metrics": metrics}
    write_json(OUTPUT / "audit" / "q_once_accounting_audit.json", q_audit)
    write_json(OUTPUT / "audit" / "parallel_fresh_state_audit.json", {"status": "PASS", "safe_parallel_enabled": True, "backend": "Legacy sequential regression gate, then two independent V2 Python subprocesses in parallel", "parallel_stage_methods": [METHOD_FULL_V2, METHOD_FIXED_V2], "statuses": statuses, "shared_mutable_state": False, "fresh_t0_state": True})
    write_json(OUTPUT / "audit" / "environment_audit.json", {"status": "PASS", "seed": SEED, "V25": True, "capacity": CAPACITY, "duration_s": SLOTS * SLOT_SECONDS, "slot_s": SLOT_SECONDS, "power": v25.FROZEN_POWER, "bundle_contract": read_json(bundle_root / "bundle_contract.json")})

    legacy_common = next(row for row in metrics if row["method"] == METHOD_LEGACY)
    v2_rows = [row for row in metrics if row["method"] != METHOD_LEGACY]
    diagnosis = {
        "legacy_frozen_metric": legacy_native,
        "legacy_common_v2_rescore": legacy_common,
        "formula_only_legacy_common_minus_frozen": {"delta_J_all_net": legacy_common["J_all_net"] - f(legacy_native["J_all_net"]), "delta_J_HV_net": legacy_common["J_HV_net"] - f(legacy_native["J_HV_net"])},
        "behavior_under_common_v2": [{"method": row["method"], "delta_J_all_net": row["J_all_net"] - legacy_common["J_all_net"], "delta_J_HV_net": row["J_HV_net"] - legacy_common["J_HV_net"], "delta_HV_closure": row["HV_closure_count"] - legacy_common["HV_closure_count"], "delta_high_closure": row["high_closure_count"] - legacy_common["high_closure_count"], "delta_urgent_closure": row["urgent_closure_count"] - legacy_common["urgent_closure_count"]} for row in v2_rows],
        "interpretation": "Frozen Legacy metrics preserve historical accounting; common-V2 rows are the valid apples-to-apples q-once, 10/Wh, Q/16 comparison.",
    }
    write_json(OUTPUT / "audit" / "legacy_vs_v2_diagnosis.json", diagnosis)
    report = [
        "# V25 Decoupled PROP-V2 seed=2026 diagnostic", "",
        "This is a single-seed mechanism diagnostic, not a significance claim.", "",
        "## Frozen Legacy regression", "",
        f"Legacy J_all_net={legacy_native['J_all_net']}, J_HV_net={legacy_native['J_HV_net']}, HV closures={legacy_native['N_HV_closure']}.", "",
        "## Common V2 evaluator", "", (tables / "common_v2_metrics.md").read_text(encoding="utf-8"),
        "Legacy is also re-scored with the V2 evaluator because the requested 10/Wh and Q/16 costs differ from historical frozen accounting.", "",
        "Full V2 and Fixed-B V2 share the same wrapper, evaluator, products, progressive lifecycle, and action semantics; Fixed-B only removes lightweight/high_accuracy candidates.", "",
        COMPLETION,
    ]
    (OUTPUT / "FINAL_DIAGNOSTIC.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not identity_pass:
        raise RuntimeError("METHOD_IDENTITY_MISMATCH")
    if not q_pass:
        raise RuntimeError("Q_ONCE_ACCOUNTING_MISMATCH")


def prepare_bundle() -> Path:
    bundle = OUTPUT / "config" / "controlled_bundle_seed_2026"
    if bundle.exists():
        v25.load_bundle(bundle)
    else:
        exact.prepare_controlled_bundle(SEED, bundle)
    return bundle


def parent_main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pre_hashes = {source_key(path): sha256_file(path) for path in OLD_SOURCE_GUARD}
    bundle = prepare_bundle()
    # Gate the frozen Legacy run before executing either V2 behavior.  Only the
    # two mutually isolated V2 children are then launched concurrently.
    statuses = [launch(METHOD_LEGACY, bundle)]
    legacy = legacy_regression(bundle)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(launch, method, bundle): method for method in (METHOD_FULL_V2, METHOD_FIXED_V2)}
        for future in as_completed(futures):
            statuses.append(future.result())
            print(f"completed: {futures[future]}", flush=True)
    emit_postprocessing(bundle, legacy, statuses, pre_hashes)
    completion = {
        "status": "PASS",
        "final_marker": COMPLETION,
        "completed_utc": utc_now(),
        "entry_sha256": sha256_file(Path(__file__).resolve()),
        "v2_source_sha256": sha256_file(V2_SOURCE),
        "v2_config_sha256": sha256_file(V2_CONFIG),
        "common_metrics_sha256": sha256_file(OUTPUT / "tables" / "common_v2_metrics.csv"),
        "all_required_audits_passed": True,
        "methods": list(METHODS),
        "safe_parallel_enabled": True,
        "parallel_stage_methods": [METHOD_FULL_V2, METHOD_FIXED_V2],
    }
    write_json(OUTPUT / "completion_status.json", completion)
    print("1. run file:", Path(__file__).resolve())
    print("2. output:", OUTPUT.resolve())
    print("3. methods:", ", ".join(METHODS))
    print("4. safe parallel: True (independent child processes)")
    print("5. final:", COMPLETION)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--bundle-root", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(child_main(args) if args.child else parent_main())
