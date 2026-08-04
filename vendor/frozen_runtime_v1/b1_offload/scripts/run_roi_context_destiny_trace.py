"""Generate read-only ROI+context staged-delivery destiny diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.roi_context_destiny_trace import (  # noqa: E402
    PERSISTENT_ARM_DIR,
    analyze_runs,
    infer_run_dirs,
    load_run,
    write_analysis_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "_value_aware_staged_delivery_queue",
        help="Directory containing the prior staged-delivery run tree.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "_roi_context_destiny_trace",
        help="Output directory for the diagnostic package.",
    )
    parser.add_argument(
        "--arm-dir",
        default=PERSISTENT_ARM_DIR,
        help="Run directory name under each scenario/seed to audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = infer_run_dirs(args.source_dir, args.arm_dir)
    if not run_dirs:
        raise SystemExit(f"No run directories found under {args.source_dir / 'runs'} for arm {args.arm_dir!r}")
    runs = [load_run(run_dir) for run_dir in run_dirs]
    analysis = analyze_runs(runs)
    command = (
        "python scripts/run_roi_context_destiny_trace.py "
        f"--source-dir {args.source_dir.as_posix()} --out-dir {args.out_dir.as_posix()} --arm-dir {args.arm_dir}"
    )
    write_analysis_outputs(
        analysis,
        args.out_dir,
        source_dir=args.source_dir,
        command=command,
        project_root=PROJECT_ROOT,
    )
    print(f"Wrote ROI+Ctx destiny trace package to {args.out_dir}")
    print(f"Runs audited: {len(runs)}")
    print(f"High/urgent context/evidence tasks: {len(analysis.destiny_rows)}")


if __name__ == "__main__":
    main()
