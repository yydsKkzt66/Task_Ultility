"""Shared CLI mechanics; method behavior remains in explicit native modules."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from atomic_io import atomic_write_json
from bundle_io import load_bundle
from contracts import CONTRACT_SHA256
from runtime import RunRequest, execute_identity, execute_run
from diagnostics import DiagnosticProbe, persist_failure_bundle


class _Tee(io.TextIOBase):
    def __init__(self, original, capture: io.StringIO) -> None:
        self.original = original
        self.capture = capture

    def write(self, value):
        self.capture.write(value)
        return self.original.write(value)

    def flush(self):
        self.capture.flush()
        return self.original.flush()


def main(make_binding) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("identity", "run"), required=True)
    parser.add_argument("--phase", choices=("identity", "smoke", "formal", "diagnostic"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--slots", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "method_started.json", {"phase": args.phase, "mode": args.mode,
                                                        "slots": args.slots, "simulation_started": False})
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    binding = None
    with contextlib.redirect_stdout(_Tee(sys.stdout, stdout_capture)), contextlib.redirect_stderr(_Tee(sys.stderr, stderr_capture)):
        try:
            binding = make_binding(args.source_root.resolve())
            if args.mode == "identity":
                result = execute_identity(binding)
                atomic_write_json(output / "native_identity.json", result)
                atomic_write_json(output / "method_run_status.json", {"status": "PASS", "phase": args.phase, "mode": args.mode})
                atomic_write_json(output / "method_completed.json", {"status": "PASS", "phase": args.phase, "mode": args.mode})
            else:
                if args.bundle is None or args.slots is None:
                    raise ValueError("run mode requires --bundle and --slots")
                bundle = load_bundle(args.bundle.resolve(), CONTRACT_SHA256)
                result = execute_run(binding, RunRequest(bundle=bundle, output=output, slots=args.slots, phase=args.phase))
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        except BaseException as error:
            failure = {"status": "FAIL", "phase": args.phase, "mode": args.mode,
                       "error_type": type(error).__name__, "error": str(error), "error_repr": repr(error),
                       "scenario_reconstruction_count": 0, "simulation_started": False, "event_count": 0}
            if not (output / "method_run_status.json").exists():
                atomic_write_json(output / "method_run_status.json", failure)
            probe = getattr(binding, "diagnostic_probe", None) if binding is not None else None
            if probe is None:
                probe = DiagnosticProbe(output)
            persist_failure_bundle(output, error, probe, stdout_capture.getvalue(), stderr_capture.getvalue())
            raise
