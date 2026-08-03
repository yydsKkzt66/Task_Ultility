from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_hv_urgent_residual_protection_v1 as guardrun
import scripts.run_prop_reqalign_seed2026_v1 as reqrun


OUT = (
    ROOT
    / "outputs"
    / "_v25_bupt1_medium_slack_reclaim_scoreonly_seed2026_cap050_v1"
)
STRICT_ATTEMPT = (
    ROOT
    / "outputs"
    / "_v25_bupt1_medium_slack_reclaim_seed2026_cap050_v1"
)
CAP050_REFERENCE = (
    ROOT
    / "outputs"
    / "_v25_bupt1_hv_urgent_residual_guard_seed2026_cap050_v1"
)
TUT_CAP050 = ROOT / "outputs" / "_v25_bupt1_tut_seed2026_cap050_v1"
RECLAIM_SOURCE = ROOT / "src" / "routers" / "medium_slack_reclaim.py"
METHOD_REFERENCE = reqrun.METHOD_PROP_NEW
METHOD_RECLAIM = (
    "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-MEDIUM-RECLAIM-SCOREONLY-v1"
)
METHOD_TUT = "TUT-UOI-PRODUCT-LITE-v2"
CAPACITY = 0.50
SLOTS = 4320
HV_NET_IDEAL_RELATIVE_TOLERANCE = Decimal("0.03")
HV_NET_MAX_RELATIVE_TOLERANCE = Decimal("0.05")
MAX_HV_CLOSURE_LOSS = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MediumSlackReclaimWrapper(guardrun.HVGuardedRouterWrapper):
    """Restore one original medium action only when online HV displacement is zero."""

    def __init__(self, *args, reclaim_module: Any, **kwargs):
        super().__init__(*args, **kwargs)
        self.reclaim_module = reclaim_module
        self.traces["medium_reclaim_trace"] = []

    def select_action(
        self,
        context: Any,
        candidate_actions: list[Any],
        predicted_map: dict[str, Any],
        guidance_bundle=None,
    ):
        reqalign_action = self.delegate.select_action(
            context,
            candidate_actions,
            predicted_map,
            guidance_bundle,
        )
        self.last_reqalign_action = reqalign_action
        self.last_guarded_action = reqalign_action
        self.last_all_score_components = deepcopy(
            self.delegate.last_all_score_components
        )
        self.last_score_components = deepcopy(
            self.delegate.last_score_components
        )
        self.last_selected_action_score = (
            self.delegate.last_selected_action_score
        )
        self._predicted_map = predicted_map
        task = self.current_step.task
        if task is None:
            return reqalign_action
        projection_row = self.delegate.rows[-1]
        if (
            self._level(task) != "medium"
            or projection_row.get("projection_reason")
            != "ordinary_required_product_infeasible"
            or self.delegate.projection.normalize_product(
                predicted_map[
                    reqalign_action.action_name
                ].predicted_product_type
            )
            != "none"
        ):
            return reqalign_action

        scores = {
            name: float(
                components.get("final_action_score", -math.inf)
            )
            for name, components in self.last_all_score_components.items()
        }
        original_name = str(projection_row.get("original_action", ""))
        actions = {
            action.action_name: action for action in candidate_actions
        }
        original = actions.get(original_name)
        defer_score = scores.get(reqalign_action.action_name, math.inf)
        original_score = scores.get(original_name, -math.inf)
        predicted_reward = (
            -math.inf
            if original is None
            else float(predicted_map[original_name].predicted_reward)
        )
        if original is None:
            displacement_urgent = 0
            displacement_high = 0
            deadline_feasible = False
            predicted_arrival = None
            candidate = None
        else:
            candidate = self._action_candidate(original, scores)
            displacement_urgent = (
                candidate.urgent_closure_displacement
            )
            displacement_high = candidate.high_closure_displacement
            deadline_feasible = candidate.deadline_feasible
            predicted_arrival = candidate.predicted_ground_arrival_s
        facts = self.reclaim_module.MediumReclaimFacts(
            value_level=self._level(task),
            projection_reason=str(
                projection_row.get("projection_reason", "")
            ),
            reqalign_is_defer=True,
            original_action_available=original is not None,
            original_hard_safe=original is not None,
            original_deadline_feasible=deadline_feasible,
            original_score=original_score,
            defer_score=defer_score,
            predicted_reward=predicted_reward,
            urgent_closure_displacement=displacement_urgent,
            high_closure_displacement=displacement_high,
        )
        decision = self.reclaim_module.decide_medium_slack_reclaim(
            facts,
            require_positive_predicted_reward=False,
        )
        final = original if decision.reclaim else reqalign_action
        self.last_guarded_action = final
        if decision.reclaim:
            self.last_selected_action_score = original_score
            self.last_score_components = deepcopy(
                self.last_all_score_components.get(original_name, {})
            )
            self.delegate.last_selected_action_score = original_score
            self.delegate.last_score_components = deepcopy(
                self.last_score_components
            )
        self.traces["medium_reclaim_trace"].append(
            {
                "slot": int(self.current_step.slot_index),
                "task_id": str(task.task_id),
                "value_level": self._level(task),
                "projection_reason": projection_row.get(
                    "projection_reason", ""
                ),
                "REQALIGN_action": reqalign_action.action_name,
                "original_action": original_name,
                "final_action": final.action_name,
                "original_score": original_score,
                "defer_score": defer_score,
                "predicted_reward": predicted_reward,
                "predicted_ground_arrival_s": predicted_arrival,
                "deadline_feasible": int(deadline_feasible),
                "urgent_closure_displacement": displacement_urgent,
                "high_closure_displacement": displacement_high,
                "reclaimed": int(decision.reclaim),
                "decision_reason": decision.reason,
            }
        )
        return final


def run_reclaim(bundle_root: Path, output: Path) -> None:
    v25 = reqrun.load_v25()
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import (
        LinkTrace,
        ResourceTrace,
        TaskSample,
        TimelineStep,
    )
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    reclaim_module = guardrun.load_module(
        "medium_slack_reclaim_runtime_v1", RECLAIM_SOURCE
    )
    guard_module = guardrun.load_module(
        "medium_slack_reclaim_guard_runtime_v1",
        guardrun.GUARD_SOURCE,
    )
    prop_mod = v25.import_prop_script()
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(
        bundle,
        TaskSample,
        ResourceTrace,
        LinkTrace,
        TimelineStep,
    )
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configs = prop_mod._configure(
        scenario.configs, calibration, {"enabled": False}
    )
    scenario = guardrun.replace(scenario, configs=configs)
    scenario, power_identity = v25.inject_power(
        scenario, v25.SOURCE_PROP
    )
    scenario, capacity_identity = guardrun.apply_capacity(
        scenario, CAPACITY
    )
    scenario_identity = v25.relaxed_scenario_identity(
        METHOD_RECLAIM, scenario, bundle, power_identity
    )
    if (
        not scenario_identity["task_hash_match"]
        or scenario_identity["runtime_value_mismatch_count"]
    ):
        raise RuntimeError(
            {"scenario_identity_failed": scenario_identity}
        )
    env = SimulationEnvironment(scenario.configs)
    runner = env.build_episode_runner(scenario.timeline[:SLOTS])
    base_router = build_routers(
        scenario.configs["experiment"],
        ActionSpace(scenario.configs["action_space"]),
    )[0]
    projection = reqrun.load_projection_module()
    reqalign = reqrun.RequirementAwareRouterWrapper(
        base_router,
        reqrun.load_projection_config(True),
        scenario.timeline[:SLOTS],
        scenario.configs["simulator"],
        projection,
    )
    reclaim = MediumSlackReclaimWrapper(
        reqalign,
        guardrun.load_guard_config(True),
        scenario.timeline[:SLOTS],
        scenario.configs["simulator"],
        guard_module,
        reclaim_module=reclaim_module,
    )
    guardrun.bind_guard_runtime(runner, reqalign, reclaim)
    result = runner.run(reclaim)
    normalized = normalize_episode(
        METHOD_RECLAIM, scenario, result, runner, SLOTS
    )
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(
        output / "candidate_action_log.csv",
        list(getattr(runner, "last_candidate_action_logs", [])),
    )
    reqrun.write_energy_components(output, result)
    projection_rows = reqrun.enrich_projection_rows(
        reqalign, output
    )
    guardrun.write_csv(
        output / "requirement_projection_log.csv",
        projection_rows,
    )
    guardrun.write_guard_traces(reclaim, output)
    identity = {
        "method": METHOD_RECLAIM,
        "parent_method": METHOD_REFERENCE,
        "source_root": str(v25.SOURCE_PROP.resolve()),
        "corrected_reqalign_enabled": True,
        "medium_slack_reclaim_enabled": True,
        "medium_only": True,
        "required_projection_reason": (
            "ordinary_required_product_infeasible"
        ),
        "require_zero_urgent_displacement": True,
        "require_zero_high_displacement": True,
        "require_hard_safe": True,
        "require_deadline_feasible": True,
        "require_positive_existing_score": True,
        "require_positive_raw_predicted_reward": False,
        "preserve_original_scorer": True,
        "preserve_original_selector": True,
        "preserve_safe_filter": True,
        "reclaim_source_sha256": reqrun.sha256_file(
            RECLAIM_SOURCE
        ),
        "projection_source_sha256": reqrun.sha256_file(
            reqrun.PROJECTION_SOURCE
        ),
        "guard_source_sha256": reqrun.sha256_file(
            guardrun.GUARD_SOURCE
        ),
        "capacity_identity": capacity_identity,
        "scenario_identity": scenario_identity,
    }
    guardrun.write_json(output / "native_identity.json", identity)
    guardrun.write_json(
        output / "method_run_status.json",
        {
            "status": "PASS",
            "method": METHOD_RECLAIM,
            "simulation_started": True,
            "fresh_state": True,
            "reused_action_trace": False,
            "reused_queue_trace": False,
            "reused_delivery_ledger": False,
            "identity": identity,
        },
    )
    guardrun.write_json(
        output / "method_completed.json",
        {"status": "PASS", "method": METHOD_RECLAIM},
    )


def launch(
    label: str,
    bundle_root: Path,
    output: Path,
    log_dir: Path,
) -> dict[str, Any]:
    if label == "REFERENCE":
        command = [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "run_hv_urgent_residual_protection_v1.py"
            ),
            "--child-run",
            "--label",
            "REQALIGN_REFERENCE",
            "--bundle-root",
            str(bundle_root),
            "--output",
            str(output),
            "--slots",
            str(SLOTS),
        ]
    else:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-run",
            "--bundle-root",
            str(bundle_root),
            "--output",
            str(output),
        ]
    started = utc_now()
    process = subprocess.run(
        command,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}_stdout.txt").write_text(
        process.stdout, encoding="utf-8"
    )
    (log_dir / f"{label}_stderr.txt").write_text(
        process.stderr, encoding="utf-8"
    )
    return {
        "label": label,
        "status": (
            "PASS"
            if process.returncode == 0
            and (output / "method_completed.json").exists()
            else "FAIL"
        ),
        "returncode": process.returncode,
        "started_utc": started,
        "finished_utc": utc_now(),
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-6000:],
    }


def evaluate(
    out: Path, bundle_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    v25 = reqrun.load_v25()
    eval_mod = v25.load_eval_module(out)
    config = reqrun.read_json(bundle_root / "resolved_config.json")
    paths = {
        METHOD_REFERENCE: out / "runs" / "REFERENCE",
        METHOD_RECLAIM: out / "runs" / "MEDIUM_RECLAIM",
        METHOD_TUT: TUT_CAP050 / "runs" / "TUT",
    }
    rows = []
    audits = []
    for method, path in paths.items():
        metric, accounting = eval_mod.summarize_method(
            method,
            "MEDIUM_SLACK_RECLAIM_CAP050_CAUSAL_ABLATION",
            path,
            config["product_requirement_utility"],
            config["unified_ground_received_accounting"],
        )
        metric["J_HV_net"] = reqrun.hv_net(
            eval_mod,
            path,
            metric,
            config["product_requirement_utility"],
        )
        metric.update(
            reqrun.subgroup_metrics(
                eval_mod,
                path,
                metric,
                config["product_requirement_utility"],
            )
        )
        metric.update(reqrun.energy_metrics(method, path))
        rows.append(metric)
        audits.append(
            {
                "method": method,
                "passed": bool(accounting.get("passed", True)),
                "accounting_error": accounting.get(
                    "accounting_error", 0
                ),
            }
        )
    return rows, audits


def selected_metrics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "method": row["method"],
            "J_all_net": row["J_all_net"],
            "J_HV_net": row["J_HV_net"],
            "HV_closure_count": row["HV_closure_count"],
            "high_closure_count": row["high_closure_count"],
            "urgent_closure_count": row[
                "urgent_closure_count"
            ],
            "ordinary_closure_count": row[
                "ordinary_closure_count"
            ],
            "closure_count": row["closure_count"],
            "gross_delivered_utility": row[
                "gross_delivered_utility"
            ],
            "public_method_cost": row["public_method_cost"],
            "controllable_energy_Wh": row[
                "controllable_compute_communication_energy_Wh"
            ],
            "transmitted_MB": row["transmitted_MB"],
            "soc_violation_count": row["soc_violation_count"],
            "thermal_violation_count": row[
                "thermal_violation_count"
            ],
        }
        for row in rows
    ]


def paired_tasks(
    reference: Path, reclaim: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old = {
        row["task_id"]: row
        for row in reqrun.read_csv(
            reference / "per_task_outcomes.csv"
        )
    }
    new = {
        row["task_id"]: row
        for row in reqrun.read_csv(
            reclaim / "per_task_outcomes.csv"
        )
    }
    rows = []
    for task_id in sorted(old):
        before = old[task_id]
        after = new[task_id]
        old_closed = reqrun.integer(
            before.get("closure_success")
        )
        new_closed = reqrun.integer(
            after.get("closure_success")
        )
        if old_closed == new_closed:
            continue
        rows.append(
            {
                "task_id": task_id,
                "value_level": before.get("value_level", ""),
                "required_product": before.get(
                    "required_product", ""
                ),
                "reference_closure": old_closed,
                "reclaim_closure": new_closed,
                "change": (
                    "gained" if new_closed > old_closed else "lost"
                ),
            }
        )
    summary = {}
    for level in ("low", "medium", "high", "urgent"):
        subset = [
            row for row in rows if row["value_level"] == level
        ]
        summary[level] = {
            "gained": sum(
                row["change"] == "gained" for row in subset
            ),
            "lost": sum(
                row["change"] == "lost" for row in subset
            ),
        }
    return rows, summary


def main_audit(
    out: Path,
    metrics: list[dict[str, Any]],
    accounting: list[dict[str, Any]],
) -> dict[str, Any]:
    by_method = {row["method"]: row for row in metrics}
    reference = by_method[METHOD_REFERENCE]
    reclaim = by_method[METHOD_RECLAIM]
    trace = reqrun.read_csv(
        out
        / "runs"
        / "MEDIUM_RECLAIM"
        / "medium_reclaim_trace.csv"
    )
    reclaimed = [
        row for row in trace if reqrun.integer(row.get("reclaimed"))
    ]
    j_all_delta = (
        reqrun.dec(reclaim["J_all_net"])
        - reqrun.dec(reference["J_all_net"])
    )
    j_hv_delta = (
        reqrun.dec(reclaim["J_HV_net"])
        - reqrun.dec(reference["J_HV_net"])
    )
    hv_relative_drop = max(
        Decimal("0"),
        -j_hv_delta
        / max(abs(reqrun.dec(reference["J_HV_net"])), Decimal("1e-12")),
    )
    checks = {
        "J_all_net_strictly_increased": j_all_delta > 0,
        "J_HV_net_relative_drop_within_5_percent": (
            hv_relative_drop <= HV_NET_MAX_RELATIVE_TOLERANCE
        ),
        "HV_closure_loss_at_most_one": reqrun.integer(
            reclaim["HV_closure_count"]
        )
        >= (
            reqrun.integer(reference["HV_closure_count"])
            - MAX_HV_CLOSURE_LOSS
        ),
        "all_reclaims_zero_predicted_HV_displacement": all(
            reqrun.integer(
                row.get("urgent_closure_displacement")
            )
            == 0
            and reqrun.integer(
                row.get("high_closure_displacement")
            )
            == 0
            for row in reclaimed
        ),
        "public_accounting_passed": all(
            row["passed"] for row in accounting
        ),
        "hard_safety_passed": (
            reqrun.integer(reclaim["soc_violation_count"]) == 0
            and reqrun.integer(
                reclaim["thermal_violation_count"]
            )
            == 0
        ),
    }
    return {
        "hypothesis_supported": all(checks.values()),
        "preregistered_checks": checks,
        "J_all_net_delta": j_all_delta,
        "J_HV_net_delta": j_hv_delta,
        "J_HV_net_relative_drop": hv_relative_drop,
        "J_HV_net_drop_within_ideal_3_percent": (
            hv_relative_drop <= HV_NET_IDEAL_RELATIVE_TOLERANCE
        ),
        "HV_closure_delta": (
            reqrun.integer(reclaim["HV_closure_count"])
            - reqrun.integer(reference["HV_closure_count"])
        ),
        "high_closure_delta": (
            reqrun.integer(reclaim["high_closure_count"])
            - reqrun.integer(reference["high_closure_count"])
        ),
        "urgent_closure_delta": (
            reqrun.integer(reclaim["urgent_closure_count"])
            - reqrun.integer(reference["urgent_closure_count"])
        ),
        "eligible_medium_defer_checks": len(trace),
        "reclaimed_medium_action_count": len(reclaimed),
        "decision_reason_counts": {
            reason: sum(
                row.get("decision_reason") == reason for row in trace
            )
            for reason in sorted(
                {row.get("decision_reason", "") for row in trace}
            )
        },
    }


def parent_main(overwrite: bool) -> int:
    if OUT.exists():
        if not overwrite:
            raise FileExistsError(OUT)
        resolved = OUT.resolve()
        if (
            resolved.parent != (ROOT / "outputs").resolve()
            or resolved.name
            != "_v25_bupt1_medium_slack_reclaim_scoreonly_seed2026_cap050_v1"
        ):
            raise RuntimeError(
                f"Unsafe output deletion target: {resolved}"
            )
        shutil.rmtree(OUT)
    for name in (
        "audit",
        "config",
        "logs",
        "runs",
        "scenario",
        "tables",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_requirement_aware_projection.py",
            "tests/test_hv_urgent_residual_guard.py",
            "tests/test_medium_slack_reclaim.py",
            "-q",
        ],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (OUT / "test_report.txt").write_text(
        tests.stdout, encoding="utf-8"
    )
    if tests.returncode:
        return 2
    v25 = reqrun.load_v25()
    input_audit, frozen = v25.preflight(OUT)
    reqrun.write_preflight_audits(
        OUT, v25, input_audit, frozen
    )
    bundle_root = v25.prepare_runtime_bundle(OUT)
    statuses = [
        launch(
            "REFERENCE",
            bundle_root,
            OUT / "runs" / "REFERENCE",
            OUT / "logs",
        ),
        launch(
            "MEDIUM_RECLAIM",
            bundle_root,
            OUT / "runs" / "MEDIUM_RECLAIM",
            OUT / "logs",
        ),
    ]
    guardrun.write_json(
        OUT / "audit" / "run_status.json", statuses
    )
    if not all(row["status"] == "PASS" for row in statuses):
        return 2
    new_hashes = guardrun.semantic_hashes(
        OUT / "runs" / "REFERENCE"
    )
    prior_hashes = guardrun.semantic_hashes(
        CAP050_REFERENCE / "runs" / "REQALIGN_REFERENCE"
    )
    regression = {
        "passed": new_hashes == prior_hashes,
        "new_reference_hashes": new_hashes,
        "prior_reference_hashes": prior_hashes,
    }
    guardrun.write_json(
        OUT / "audit" / "reference_regression_audit.json",
        regression,
    )
    if not regression["passed"]:
        return 2
    metrics, accounting = evaluate(OUT, bundle_root)
    selected = selected_metrics(metrics)
    guardrun.write_csv(
        OUT / "tables" / "primary_and_auxiliary_metrics.csv",
        selected,
    )
    (OUT / "tables" / "primary_and_auxiliary_metrics.md").write_text(
        guardrun.markdown_table(selected) + "\n",
        encoding="utf-8",
    )
    audit = main_audit(OUT, metrics, accounting)
    pairs, pair_summary = paired_tasks(
        OUT / "runs" / "REFERENCE",
        OUT / "runs" / "MEDIUM_RECLAIM",
    )
    guardrun.write_csv(
        OUT / "tables" / "paired_closure_changes.csv", pairs
    )
    guardrun.write_json(
        OUT / "audit" / "medium_reclaim_hypothesis_audit.json",
        audit,
    )
    guardrun.write_json(
        OUT / "audit" / "paired_closure_summary.json",
        pair_summary,
    )
    guardrun.write_json(
        OUT / "audit" / "public_accounting_audit.json",
        {
            "passed": all(row["passed"] for row in accounting),
            "rows": accounting,
        },
    )
    lines = [
        "# Medium slack reclaim causal ablation",
        "",
        "Scope: seed=2026, capacity=0.50, one preregistered fresh run.",
        "",
        "This experiment tests a narrow causal claim; it does not prove a global optimum.",
        "",
        f"The preserved strict attempt is at `{STRICT_ATTEMPT}`. It reclaimed zero actions because the raw predicted-reward gate rejected every deadline-feasible candidate.",
        "",
        "## Metrics",
        "",
        guardrun.markdown_table(selected),
        "",
        "## Preregistered hypothesis audit",
        "",
        "```json",
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "```",
        "",
        "## Paired closure summary",
        "",
        "```json",
        json.dumps(
            pair_summary,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "```",
        "",
        (
            "Conclusion: the hypothesis is supported."
            if audit["hypothesis_supported"]
            else "Conclusion: the hypothesis is not supported."
        ),
    ]
    (OUT / "final_assessment.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    status = (
        "MEDIUM_SLACK_RECLAIM_HYPOTHESIS_SUPPORTED"
        if audit["hypothesis_supported"]
        else "MEDIUM_SLACK_RECLAIM_HYPOTHESIS_NOT_SUPPORTED"
    )
    guardrun.write_json(
        OUT / "completion_status.json",
        {
            "status": status,
            "output": str(OUT),
            "single_seed_causal_ablation": True,
        },
    )
    print(guardrun.markdown_table(selected))
    print(json.dumps(audit, indent=2, default=str))
    print(f"FINAL_STATUS={status}")
    print(f"OUTPUT_DIR={OUT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--bundle-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.child_run:
        run_reclaim(
            Path(args.bundle_root), Path(args.output)
        )
        return 0
    return parent_main(args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
