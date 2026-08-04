"""Standard-library-only one-shot orchestration for native-direct phases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .aggregate_results import aggregate_phase
    from .atomic_io import atomic_write_csv, atomic_write_json, atomic_write_text
    from .bundle_io import load_bundle, verify_and_copy_bundle
    from .contracts import CONTRACT_SHA256, METHODS, build_native_manifest
    from .process_env import build_child_env
    from .value_adapter import audit_canonical_value_mapping
except ImportError:
    from aggregate_results import aggregate_phase
    from atomic_io import atomic_write_csv, atomic_write_json, atomic_write_text
    from bundle_io import load_bundle, verify_and_copy_bundle
    from contracts import CONTRACT_SHA256, METHODS, build_native_manifest
    from process_env import build_child_env
    from value_adapter import audit_canonical_value_mapping


PHASES = ("identity", "smoke", "formal")
TOOL_ROOT = Path(__file__).resolve().parent
WORKTREE = TOOL_ROOT.parents[1]
SOURCE_BUNDLE = Path(r"D:/wt/pilot/outputs/_late_hv_value_skew_native_direct_v1/canonical_scenario_bundle")


def phase_method_dir(output_root: Path, phase: str, method: str) -> Path:
    if phase not in PHASES:
        raise ValueError(f"unsupported phase: {phase}")
    if not method or any(separator in method for separator in ("/", "\\", "..")):
        raise ValueError("invalid method directory name")
    return Path(output_root) / phase / method


def launch_fresh(command: Sequence[str], cwd: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=Path(cwd), env=dict(env), text=True, capture_output=True, check=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_commands(output_root: Path, phase: str, slots: int) -> list[dict[str, Any]]:
    manifest = build_native_manifest(); bundle = Path(output_root) / "canonical_scenario_bundle"
    rows=[]
    for method in METHODS:
        spec=manifest["methods"][method]; source=Path(spec["source_root"])
        command=[sys.executable,str(TOOL_ROOT/spec["wrapper"]),"--mode","run","--phase",phase,
                 "--source-root",str(source),"--bundle",str(bundle),"--output",str(phase_method_dir(output_root,phase,method)),
                 "--slots",str(slots)]
        rows.append({"phase":phase,"method":method,"source_root":str(source),"cwd":str(source),"command":command})
    return rows


def identity_commands(output_root: Path) -> list[dict[str, Any]]:
    manifest = build_native_manifest(); rows = []
    for method in METHODS:
        spec = manifest["methods"][method]; source = Path(spec["source_root"])
        command = [sys.executable, str(TOOL_ROOT / spec["wrapper"]), "--mode", "identity",
                   "--phase", "identity", "--source-root", str(source),
                   "--output", str(phase_method_dir(output_root, "identity", method))]
        rows.append({"phase": "identity", "method": method, "source_root": str(source),
                     "cwd": str(source), "command": command})
    return rows


def verify_phase_unstarted(root: Path, phase: str) -> None:
    for method in METHODS:
        directory=phase_method_dir(root,phase,method)
        if directory.exists() or (directory/"method_started.json").exists():
            raise FileExistsError(directory)


def prepare(root: Path) -> dict[str, Any]:
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/"native_runner_manifest.json"; expected=build_native_manifest()
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("existing native runner manifest mismatch")
    else:
        atomic_write_json(manifest_path,expected)
    destination=root/"canonical_scenario_bundle"
    if destination.exists():
        raise FileExistsError(destination)
    copied=verify_and_copy_bundle(SOURCE_BUNDLE,destination,CONTRACT_SHA256)
    atomic_write_json(root/"canonical_bundle_hashes.json",copied)
    return copied


def run_myopic_value_contract_gate(root: Path, bundle: Any) -> dict[str, Any]:
    """Evaluate all canonical tasks before smoke; never reconstruct or simulate."""
    report = audit_canonical_value_mapping(bundle.tasks)
    atomic_write_json(Path(root) / "myopic_decision_value_contract_gate.json", report)
    return report


def _identity_gate(root: Path) -> dict[str, Any]:
    rows=[]
    for method in METHODS:
        directory=phase_method_dir(root,"identity",method)
        identity=json.loads((directory/"native_identity.json").read_text(encoding="utf-8"))
        completed=json.loads((directory/"method_completed.json").read_text(encoding="utf-8"))
        if completed.get("status")!="PASS" or not identity.get("source_identity_passed"):
            raise RuntimeError(f"identity gate failed: {method}")
        rows.append(identity)
    return {"methods":rows}


def run_identity(root: Path) -> dict[str, Any]:
    root = Path(root); verify_phase_unstarted(root, "identity")
    executions = []
    for row in identity_commands(root):
        source = Path(row["source_root"])
        cp = launch_fresh(row["command"], source, build_child_env(os.environ, source))
        executions.append({"method": row["method"], "returncode": cp.returncode,
                           "command": row["command"], "stdout_tail": cp.stdout[-4000:],
                           "stderr_tail": cp.stderr[-4000:]})
        if cp.returncode:
            report = {"status": "FINAL_V6_SOURCE_IDENTITY_FAILED", "executions": executions}
            atomic_write_json(root / "source_identity_matrix.json", report)
            return report
    identities = _identity_gate(root)
    report = {"status": "FINAL_V6_SOURCE_IDENTITY_PASSED", "executions": executions, **identities}
    atomic_write_json(root / "source_identity_matrix.json", report)
    return report


def _input_hashes(root: Path) -> dict[str, Any]:
    manifest=build_native_manifest(); bundle=json.loads((root/"canonical_scenario_bundle/bundle_contract.json").read_text(encoding="utf-8"))
    return {"native_runner_manifest_sha256":_sha(root/"native_runner_manifest.json"),
        "bundle_contract_sha256":_sha(root/"canonical_scenario_bundle/bundle_contract.json"),
        "bundle_files":bundle["file_sha256"],
        "source_config_hashes":{method:manifest["methods"][method]["hashes"] for method in METHODS},
        "frozen_runtime_manifest_sha256":_sha(root/"frozen_runtime_source_manifest.json"),
        "reconciliation_matrix_sha256":_sha(root/"frozen_runtime_reconciliation_identity_matrix.json"),
        "micro_reference_sha256":_sha(root/"frozen_runtime_micro_reference_test.json"),
        "native_direct_code_hashes":{path.relative_to(WORKTREE).as_posix():_sha(path)
            for base in (WORKTREE/"tools/late_hv_native_direct_final_v6",WORKTREE/"tests/late_hv_native_direct",
                         WORKTREE/"tests/late_hv_native_direct_fixed_v3",WORKTREE/"tests/late_hv_native_direct_final_v6")
            for path in sorted(base.rglob("*.py"))}}


def authorize_smoke(root: Path) -> dict[str, Any]:
    root=Path(root); _identity_gate(root)
    if _sha(root/"canonical_scenario_bundle/bundle_contract.json") != CONTRACT_SHA256:
        raise ValueError("copied bundle contract mismatch")
    if json.loads((root/"frozen_runtime_reconciliation_status.json").read_text(encoding="utf-8"))["status"] != "FROZEN_RUNTIME_SOURCE_RECONCILED":
        raise RuntimeError("frozen runtime reconciliation missing")
    bundle = load_bundle(root/"canonical_scenario_bundle", CONTRACT_SHA256)
    value_gate = run_myopic_value_contract_gate(root, bundle)
    commands=[
        [sys.executable,"-m","unittest","discover","-s","tests/late_hv_native_direct","-v"],
        [sys.executable,"-m","unittest","discover","-s","tests/late_hv_native_direct_fixed_v3","-v"],
        [sys.executable,"-m","unittest","discover","-s","tests/late_hv_native_direct_final_v6","-v"],
        [sys.executable,"-m","compileall","-q","tools/late_hv_native_direct_final_v6","tests/late_hv_native_direct_final_v6"],
        ["git","diff","--check"],
    ]
    runs=[]
    for command in commands:
        cp=subprocess.run(command,cwd=WORKTREE,text=True,capture_output=True,check=False)
        runs.append({"command":command,"returncode":cp.returncode,"stdout":cp.stdout,"stderr":cp.stderr})
        if cp.returncode:
            raise RuntimeError(f"nonperformance gate failed: {command}")
    report="\n\n".join([f"COMMAND: {' '.join(r['command'])}\nEXIT: {r['returncode']}\n{r['stdout']}\n{r['stderr']}" for r in runs])
    report_path=root/"wrapper_unit_test_report_reconciled.txt"; atomic_write_text(report_path,report)
    payload={"status":"SMOKE_AUTHORIZED","created_before_smoke":True,"nonperformance_runs":runs,
        "input_hashes":_input_hashes(root),"smoke_commands":phase_commands(root,"smoke",360),
        "test_report_sha256":_sha(report_path),
        "myopic_value_contract_gate_sha256":_sha(root/"myopic_decision_value_contract_gate.json"),
        "myopic_value_contract_status":value_gate["status"],
        "myopic_mapping_sequence_sha256":value_gate["mapping_sequence_sha256"]}
    atomic_write_json(root/"smoke_authorized.json",payload)
    return payload


def _run_phase(root: Path, phase: str, slots: int, authorization: str) -> dict[str, Any]:
    root=Path(root)
    if not (root/authorization).is_file(): raise FileNotFoundError(root/authorization)
    verify_phase_unstarted(root,phase)
    commands=phase_commands(root,phase,slots)
    atomic_write_json(root/f"{phase}_started.json",{"phase":phase,"slots":slots,"commands":commands,"input_hashes":_input_hashes(root)})
    executions=[]
    for row in commands:
        source=Path(row["source_root"]); cp=launch_fresh(row["command"],source,build_child_env(os.environ,source))
        execution={"method":row["method"],"returncode":cp.returncode,"command":row["command"],"stdout_tail":cp.stdout[-4000:],"stderr_tail":cp.stderr[-4000:]}
        executions.append(execution)
        if cp.returncode:
            report={"status":f"NATIVE_DIRECT_FINAL_V6_{phase.upper()}_FAILED","phase":phase,"executions":executions}
            atomic_write_json(root/f"{phase}_report.json",report)
            return report
    aggregate=aggregate_phase(root,phase,expected_slots=slots,write_outputs=True)
    status="NATIVE_DIRECT_FINAL_V6_SMOKE_PASSED_READY_FOR_SINGLE_FORMAL" if phase=="smoke" else "FINAL_V6_FORMAL_FIVE_METHOD_RUN_COMPLETE_READY_TO_FINALIZE"
    report={"status":status,"phase":phase,"executions":executions,"fairness":aggregate["fairness"],"method_metrics":aggregate["method_metrics"]}
    atomic_write_json(root/f"{phase}_report.json",report)
    return report


def authorize_formal(root: Path) -> dict[str, Any]:
    root=Path(root); smoke=json.loads((root/"smoke_report.json").read_text(encoding="utf-8"))
    if smoke.get("status")!="NATIVE_DIRECT_FINAL_V6_SMOKE_PASSED_READY_FOR_SINGLE_FORMAL": raise RuntimeError("smoke did not pass")
    for name in ("smoke_fairness_report.json","smoke_ledger_report.json","smoke_method_metrics.csv"):
        if not (root/name).is_file(): raise FileNotFoundError(root/name)
    verify_phase_unstarted(root,"formal")
    payload={"status":"FORMAL_AUTHORIZED","input_hashes":_input_hashes(root),"smoke_report_sha256":_sha(root/"smoke_report.json"),
        "smoke_fairness_sha256":_sha(root/"smoke_fairness_report.json"),"smoke_ledger_sha256":_sha(root/"smoke_ledger_report.json"),
        "formal_commands":phase_commands(root,"formal",4320)}
    atomic_write_json(root/"formal_authorized.json",payload); return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig",newline="") as stream: return list(csv.DictReader(stream))


def finalize(root: Path) -> dict[str, Any]:
    root=Path(root); formal=json.loads((root/"formal_report.json").read_text(encoding="utf-8"))
    if formal.get("status")!="FINAL_V6_FORMAL_FIVE_METHOD_RUN_COMPLETE_READY_TO_FINALIZE":
        raise RuntimeError("five-method formal run is incomplete")
    metrics=list(formal["method_metrics"])
    layered=[{key:row.get(key) for key in ("method","UPDL_s","progressive_utility_denominator","metadata_layer_count","roi_layer_count",
        "evidence_layer_count","required_layer_success_count","required_layer_mean_latency_s","required_layer_p95_latency_s",
        "full_evidence_success_count","full_evidence_mean_latency_s","full_evidence_p95_latency_s")} for row in metrics]
    atomic_write_csv(root/"layered_latency_metrics.csv",layered,tuple(layered[0]))
    value_rows=[]; late_rows=[]
    for method in METHODS:
        tasks=_read_csv(root/"formal"/method/"per_task_outcomes.csv")
        for level in ("low","medium","high","urgent"):
            group=[row for row in tasks if row["value_level"]==level]; closed=[row for row in group if int(row["closure_success"] or 0)]
            value_rows.append({"method":method,"value_level":level,"task_count":len(group),"closure_count":len(closed),
                "closure_rate":len(closed)/max(len(group),1),"offered_effective_value":sum(float(r["effective_task_value"]) for r in group),
                "closed_effective_value":sum(float(r["effective_task_value"]) for r in closed)})
        late=[row for row in tasks if 43200<=float(row["arrival_time"])<64800]; closed=[row for row in late if int(row["closure_success"] or 0)]
        hv=[row for row in late if row["value_level"] in {"high","urgent"}]; hv_closed=[row for row in hv if int(row["closure_success"] or 0)]
        late_rows.append({"method":method,"late_window_task_count":len(late),"late_window_closure_count":len(closed),
            "late_window_closure_rate":len(closed)/max(len(late),1),"late_hv_task_count":len(hv),
            "late_hv_closure_count":len(hv_closed),"late_hv_closure_rate":len(hv_closed)/max(len(hv),1)})
    atomic_write_csv(root/"value_level_metrics.csv",value_rows,tuple(value_rows[0]))
    atomic_write_csv(root/"late_burst_metrics.csv",late_rows,tuple(late_rows[0]))
    ranked=sorted(metrics,key=lambda row:Decimal(str(row["J_prog_net"])),reverse=True)
    ranking=[{"rank":index+1,"method":row["method"],"J_prog_net":row["J_prog_net"],"J_prog_gross":row["J_prog_gross"],
              "UPDL_s":row["UPDL_s"],"closure_rate":row["closure_rate"]} for index,row in enumerate(ranked)]
    atomic_write_csv(root/"method_rankings.csv",ranking,tuple(ranking[0]))
    lines=["# Late-HV Value-Skew Native-Direct Final V6", "", "Terminal status: `LATE_HV_VALUE_SKEW_NATIVE_DIRECT_FINAL_V6_FORMAL_COMPLETE`.",
        "", "Change type: `MYOPIC_DECISION_VALUE_DOMAIN_ADAPTER` (MYOPIC decision boundary only).",
        "", "MYOPIC uses effective_task_value / 2.4787343282595278 for its frozen unit-interval input; the common ledger uses the original effective_task_value once.",
        "", "Single-seed system-level proxy; no multi-seed, sensitivity, ablation, or result-driven tuning was performed.", "",
        "## J_prog_net ranking", ""]
    lines += [f"{row['rank']}. {row['method']}: J_prog_net={row['J_prog_net']}, J_prog_gross={row['J_prog_gross']}, UPDL={row['UPDL_s']} s, closure={row['closure_rate']}" for row in ranking]
    lines += ["", "Required-layer closure and full-evidence closure are reported separately; metadata never substitutes for evidence.",
              "OFFLOAD ground-compute energy is excluded, while satellite-side compute/encoding, communication, waiting, FIFO and ground-completion latency remain represented."]
    atomic_write_text(root/"formal_final_report.md","\n".join(lines)+"\n")
    implementation=[path for base in (WORKTREE/"tools/late_hv_native_direct_final_v6",WORKTREE/"tests/late_hv_native_direct_final_v6") for path in base.rglob("*.py")]
    implementation += [WORKTREE/"docs/superpowers/specs/2026-07-18-late-hv-value-skew-native-direct-design.md",
                       WORKTREE/"docs/superpowers/plans/2026-07-18-late-hv-value-skew-native-direct.md"]
    patch_parts=[]
    for path in sorted(implementation):
        cp=subprocess.run(["git","diff","--no-index","--","/dev/null",str(path)],cwd=WORKTREE,text=True,capture_output=True,check=False)
        if cp.returncode not in (0,1): raise RuntimeError(f"cannot generate patch for {path}")
        patch_parts.append(cp.stdout)
    atomic_write_text(root/"formal_git_diff.patch","".join(patch_parts))
    rollback={"status":"MANUAL_ONLY","authorized_deletions":[],"new_worktree":str(WORKTREE),"new_output_root":str(root),
        "frozen_snapshot":"D:/wt/pilot_frozen_runtime_native_direct_v1","historical_outputs_modified":False,"git_commit_created":False}
    atomic_write_json(root/"formal_rollback_manifest.json",rollback)
    terminal={"status":"LATE_HV_VALUE_SKEW_NATIVE_DIRECT_FINAL_V6_FORMAL_COMPLETE","method_count":5,
        "formal_slots_per_method":4320,"ranking_file":"method_rankings.csv","report":"formal_final_report.md"}
    atomic_write_json(root/"formal_completion_status.json",terminal)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("prepare","identity","authorize-smoke","smoke","authorize-formal","formal","finalize")); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=="prepare": result=prepare(args.output)
    elif args.command=="identity": result=run_identity(args.output)
    elif args.command=="authorize-smoke": result=authorize_smoke(args.output)
    elif args.command=="smoke": result=_run_phase(args.output,"smoke",360,"smoke_authorized.json")
    elif args.command=="authorize-formal": result=authorize_formal(args.output)
    elif args.command=="formal": result=_run_phase(args.output,"formal",4320,"formal_authorized.json")
    else: result=finalize(args.output)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,default=str))
    return 0 if "FAILED" not in str(result.get("status","")) else 1


if __name__=="__main__": raise SystemExit(main())
