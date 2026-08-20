"""One entry point for clean-clone verification, experiments, and figures.

No arguments performs a fast preflight and regenerates the two final figure
sets.  Use ``--scope single``, ``main``, ``ablation``, ``sensitivity``, or
``all`` for simulations.  ``main`` and ``all`` execute seed 2026 fresh too.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("figures", "single", "main", "ablation", "sensitivity", "all"),
        default="figures",
        help="Default 'figures' is fast; 'all' performs every final simulation and redraws the figures.",
    )
    parser.add_argument("--strict-versions", action="store_true")
    args = parser.parse_args()

    verify = [PYTHON, str(ROOT / "scripts" / "verify_portable_reproduction_v1.py")]
    if args.strict_versions:
        verify.append("--strict-versions")
    run("portable reproduction preflight", verify)
    run(
        "competition result consistency",
        [PYTHON, str(ROOT / "scripts" / "verify_competition_results_v1.py")],
    )

    if args.scope in {"single", "all"}:
        run("single-seed five-method experiment", [PYTHON, str(ROOT / "scripts" / "run_current_prop_four_independent_baselines_single_seed_v1.py")])
    if args.scope in {"main", "all"}:
        run("fresh five-seed main experiment", [PYTHON, str(ROOT / "scripts" / "run_current_prop_four_baselines_fiveseed_dual_thermal_v1.py"), "--fresh-all"])
    if args.scope in {"ablation", "all"}:
        run("five-seed ablation", [PYTHON, str(ROOT / "scripts" / "run_current_prop_five_ablation_multiseed_dual_thermal_v1.py")])
    if args.scope in {"sensitivity", "all"}:
        run("five-seed sensitivity", [PYTHON, str(ROOT / "scripts" / "run_current_prop_value_capacity_sensitivity_multiseed_dual_thermal_v1.py")])
    if args.scope in {"figures", "all"}:
        run("main publication figures", [PYTHON, str(ROOT / "scripts" / "main metrics.py")])
        run("sensitivity and ablation figures", [PYTHON, str(ROOT / "scripts" / "sensitive.py")])

    print("\nPORTABLE_REPRODUCTION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
