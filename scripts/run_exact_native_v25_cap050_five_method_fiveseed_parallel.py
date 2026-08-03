"""Parallel, resumable exact-native V25/capacity-0.50 five-method rerun.

This module changes orchestration only.  Controlled-bundle construction,
method child processes, the public evaluator, regression anchors and report
tables are delegated to ``run_exact_native_v25_cap050_five_method_fiveseed``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "_v25_exact_native_five_method_controlled_fiveseed_cap050_parallel_rerun_v1"
)
CAPACITY = 0.50


def _status_path(output: Path, seed: int, method: str) -> Path:
    return output / "runs" / f"seed_{seed}" / method / "method_run_status.json"


def _run_path(output: Path, seed: int, method: str) -> Path:
    return output / "runs" / f"seed_{seed}" / method


def _is_complete(output: Path, seed: int, method: str) -> bool:
    return (
        (_run_path(output, seed, method) / "method_completed.json").is_file()
        and _status_path(output, seed, method).is_file()
    )


def _run_one(
    output: Path,
    bundles: dict[int, Path],
    seed: int,
    method: str,
) -> dict[str, Any]:
    status = exact.launch(
        method,
        seed,
        bundles[seed],
        _run_path(output, seed, method),
        output / "logs",
        CAPACITY,
    )
    exact.write_json(_status_path(output, seed, method), status)
    print(
        json.dumps(
            {"status": "RUN_COMPLETE", "seed": seed, "method": method},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return status


def _run_jobs(
    output: Path,
    bundles: dict[int, Path],
    jobs: list[tuple[int, str]],
    workers: int,
) -> None:
    pending = [(seed, method) for seed, method in jobs if not _is_complete(output, seed, method)]
    print(
        json.dumps(
            {
                "status": "PARALLEL_BATCH",
                "job_count": len(jobs),
                "completed_before_batch": len(jobs) - len(pending),
                "remaining": len(pending),
                "workers": min(max(1, workers), max(1, len(pending))),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_run_one, output, bundles, seed, method): (seed, method)
            for seed, method in pending
        }
        for future in as_completed(futures):
            seed, method = futures[future]
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(
                    {"V25_PARALLEL_RUNTIME_FAILED": {"seed": seed, "method": method, "error": str(error)}}
                ) from error


def _load_bundles(output: Path, resume: bool) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    bundles: dict[int, Path] = {}
    identities: list[dict[str, Any]] = []
    audit_path = output / "audit" / "controlled_bundle_audit.json"
    if resume:
        if not audit_path.is_file():
            raise FileNotFoundError(f"resume requires {audit_path}")
        for seed in exact.SEEDS:
            bundle = output / "config" / "controlled_bundles" / f"seed_{seed}"
            if not bundle.is_dir():
                raise FileNotFoundError(bundle)
            bundles[seed] = bundle
        identities = json.loads(audit_path.read_text(encoding="utf-8"))
        return bundles, identities

    if output.exists():
        raise FileExistsError(f"fresh output required: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for seed in exact.SEEDS:
        bundle = output / "config" / "controlled_bundles" / f"seed_{seed}"
        bundles[seed] = bundle
        identities.append(exact.prepare_controlled_bundle(seed, bundle))
    exact.write_json(audit_path, identities)
    return bundles, identities


def _evaluate_and_finish(output: Path, bundles: dict[int, Path]) -> None:
    eval_mod = exact.v25.load_eval_module(output)
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    anchors = exact.expected_seed2026(CAPACITY)

    for seed in exact.SEEDS:
        for method in exact.METHODS:
            if not _is_complete(output, seed, method):
                raise RuntimeError({"INCOMPLETE_V25_PARALLEL_GRID": {"seed": seed, "method": method}})
            status = json.loads(_status_path(output, seed, method).read_text(encoding="utf-8"))
            statuses.append(status)
            metrics = exact.evaluate_method(
                method,
                _run_path(output, seed, method),
                bundles[seed],
                eval_mod,
                CAPACITY,
            )
            if not bool(metrics["public_accounting_passed"]):
                raise RuntimeError({"PUBLIC_ACCOUNTING_MISMATCH": {"seed": seed, "method": method}})
            if exact.integer(metrics["soc_violation_count"]) or exact.integer(metrics["thermal_violation_count"]):
                raise RuntimeError({"HARD_SAFETY_VIOLATION": {"seed": seed, "method": method}})
            rows.append(exact.selected_metrics(seed, metrics))

            if seed == 2026 and method in anchors:
                gate = exact.regression_gate(metrics, anchors[method])
                exact.write_json(output / "audit" / f"seed2026_{method}_regression_gate.json", gate)
                if not gate["passed"]:
                    raise RuntimeError({"EXACT_NATIVE_SEED2026_REGRESSION_MISMATCH": method, "gate": gate})

    if len(rows) != len(exact.SEEDS) * len(exact.METHODS):
        raise RuntimeError({"INCOMPLETE_V25_PARALLEL_GRID": len(rows)})
    exact.emit_report_tables(output, rows)
    exact.write_json(output / "audit" / "method_run_status.json", statuses)
    completion = {
        "status": "EXACT_NATIVE_V25_CAP050_FIVE_METHOD_FIVESEED_PARALLEL_RERUN_COMPLETE",
        "seeds": list(exact.SEEDS),
        "methods": list(exact.METHODS),
        "run_count": len(rows),
        "capacity": CAPACITY,
        "workers": 5,
        "seed2026_regression_passed": True,
        "public_accounting_passed": True,
        "hard_safety_passed": True,
        "output": str(output),
    }
    exact.write_json(output / "completion_status.json", completion)
    print(json.dumps(completion, ensure_ascii=False), flush=True)


def parent_main(output: Path, workers: int, resume: bool, preflight_only: bool) -> int:
    bundles, identities = _load_bundles(output, resume)
    if not identities or not all(all(item["checks"].values()) for item in identities):
        raise RuntimeError({"CONTROLLED_BUNDLE_AUDIT_FAILED": identities})
    if preflight_only:
        print(json.dumps({"status": "PREFLIGHT_COMPLETE", "output": str(output)}, ensure_ascii=False))
        return 0

    # Preserve the exact-native contract: validate the two seed-2026 anchors
    # before releasing the remaining grid.  The two anchors may run together.
    anchor_jobs = [(2026, exact.METHOD_CORRECTED), (2026, exact.METHOD_TUT)]
    _run_jobs(output, bundles, anchor_jobs, min(2, workers))
    eval_mod = exact.v25.load_eval_module(output)
    for seed, method in anchor_jobs:
        metrics = exact.evaluate_method(
            method,
            _run_path(output, seed, method),
            bundles[seed],
            eval_mod,
            CAPACITY,
        )
        gate = exact.regression_gate(metrics, exact.expected_seed2026(CAPACITY)[method])
        exact.write_json(output / "audit" / f"seed2026_{method}_regression_gate.json", gate)
        if not gate["passed"]:
            raise RuntimeError({"EXACT_NATIVE_SEED2026_REGRESSION_MISMATCH": method, "gate": gate})

    all_jobs = [(seed, method) for seed in exact.SEEDS for method in exact.METHODS]
    remaining_jobs = [job for job in all_jobs if job not in anchor_jobs]
    _run_jobs(output, bundles, remaining_jobs, workers)
    _evaluate_and_finish(output, bundles)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    return parent_main(args.output_dir, max(1, int(args.workers)), args.resume, args.preflight_only)


if __name__ == "__main__":
    raise SystemExit(main())
