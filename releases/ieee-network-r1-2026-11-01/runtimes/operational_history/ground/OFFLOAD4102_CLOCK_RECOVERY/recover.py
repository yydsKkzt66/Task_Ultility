"""Launch one versioned OFFLOAD recovery; retain the failed attempt verbatim."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
ROOT = HERE.parents[2] / '.network_ground_parallel_work'
CONTROL = PACK / 'RUN_CONTROL_5'
OLD = PACK / 'A_RUNS/4102_OFFLOAD/attempt_02'
NEW = PACK / 'A_RUNS/4102_OFFLOAD/attempt_03'
sys.path.insert(0, str(ROOT))
import run
sys.path.insert(0, str(CONTROL))
import formal_common as fc
from runtime_io import write, overlay_identity
from exact_arrival import install, identity, VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true')
    args = ap.parse_args()
    evidence = identity()
    valid = fc.read(HERE / 'VALIDATION.json')
    assert valid['pass_'] and valid['execution_overlay'] == evidence
    base_ids = fc.verify()
    ids = dict(base_ids, **{VERSION: evidence['sha256']})
    prefix = HERE / 'PREFIX_MIGRATION/T04200.pkl'
    if not args.worker:
        assert not NEW.exists(), 'Recovery attempt already exists'
        original = OLD / 'CHECKPOINTS/T04200.pkl'
        meta = fc.read(original.with_suffix('.json'))
        assert fc.sha(original) == meta['pickle_sha256'] and meta['identity']['manifests'] == base_ids
        prefix.parent.mkdir(exist_ok=False)
        shutil.copyfile(original, prefix)
        migrated = dict(meta)
        migrated['identity'] = dict(meta['identity'], manifests=ids)
        migrated['prefix_migration'] = dict(parent_checkpoint=str(original), parent_sha256=fc.sha(original),
            parent_metadata_sha256=fc.sha(original.with_suffix('.json')),
            reason='All342 prefix arrivals exact; new arrival-only branch would not fire; state bytes unchanged',
            validation_sha256=fc.sha(HERE / 'VALIDATION.json'))
        write(prefix.with_suffix('.json'), migrated)
        write(HERE / 'PREFIX_MIGRATION.json', migrated['prefix_migration'])
        write(HERE / 'FROZEN_RECOVERY_MANIFEST.json', dict(base_manifests=base_ids, effective_manifests=ids,
            files={name:fc.sha(HERE/name) for name in ('exact_arrival.py','recover.py','validate.py','VALIDATION.json')},
            failed_audit_sha256=fc.sha(OLD/'AUDIT.json'), failed_arrivals_sha256=fc.sha(OLD/'RESULT/arrival_events.json'),
            failed_source_identity_sha256=fc.sha(OLD/'IDENTITY.json'), prefix_sha256=fc.sha(prefix)))
        command = [sys.executable, str(HERE / 'recover.py'), '--worker']
        log = (NEW.parent / 'attempt_03.log').open('ab')
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_BREAKAWAY_FROM_JOB if os.name == 'nt' else 0
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, creationflags=flags)
        log.close()
        launch = dict(pid=process.pid, command=command, created_epoch=time.time(), resume=str(prefix),
                      output=str(NEW), source_version=evidence, prior_attempt_unchanged=str(OLD))
        write(HERE / 'LAUNCH.json', launch)
        print(json.dumps(launch, ensure_ascii=False, indent=2))
        return

    manifest = fc.read(HERE / 'FROZEN_RECOVERY_MANIFEST.json')
    assert manifest['effective_manifests'] == ids
    for name, value in manifest['files'].items():
        assert fc.sha(HERE/name) == value
    install(ROOT)

    def output(path, value):
        if Path(path).name == 'IDENTITY.json':
            value = dict(value, operational_overlay=overlay_identity(), execution_overlay=evidence,
                         prefix_migration=fc.read(HERE / 'PREFIX_MIGRATION.json'))
        elif Path(path).name == 'AUDIT.json':
            value = dict(value, execution_overlay=evidence, strict_arrival_identity_unchanged=True)
        return write(path, value)
    fc.write = output
    from formal_worker import execute
    from src.network_revision.runtime import NetworkExperiment
    from window_metrics import record
    original_run = NetworkExperiment.run

    def observed_run(e, *pos, **kw):
        old_hook = e.slot_complete_hook
        ends = {4200., 9900., 15600., 21300.}
        count = len(getattr(e, 'exact_arrival_tail_records', []))
        def hook(exp, slot):
            nonlocal count
            old_hook(exp, slot)
            if exp.physics.now in ends:
                record(exp, NEW)
            tails = getattr(exp, 'exact_arrival_tail_records', [])
            if len(tails) != count:
                write(NEW / 'ARRIVAL_TAIL_EVENTS.json', dict(version=VERSION, records=tails))
                count = len(tails)
        e.slot_complete_hook = hook
        record(e, NEW)
        return original_run(e, *pos, **kw)
    NetworkExperiment.run = observed_run
    result = execute(fc.job_for(4102, 'OFFLOAD'), NEW, resume=prefix, identities=ids)
    old_summary = fc.read(OLD / 'RESULT/AUDITED_SUMMARY.json')
    new_summary = fc.read(NEW / 'RESULT/AUDITED_SUMMARY.json')
    costs = lambda s: dict(gross=s['gross'], net=s['net'], **s['costs'], closed=s['closed'])
    before, after = costs(old_summary), costs(new_summary)
    comparison = dict(status='COMPLETED_AND_AUDIT_PASS', new_attempt=str(NEW),
        old_attempt_retained_failed=True, execution_overlay=evidence,
        tail_records=getattr(result, 'exact_arrival_tail_records', []),
        original_failed_metrics=before, recovered_metrics=after,
        metric_deltas={k:after[k]-before[k] for k in before},
        audit_sha256=fc.sha(NEW/'AUDIT.json'), source_manifests_verified=fc.verify()==base_ids,
        actual_arrivals=len(result.arrival_events), exact_arrivals=sum(
            r['time_s']==result.arrived[r['task_id']].arrival_s for r in result.arrival_events))
    write(HERE/'RECOVERY_RESULT.json', comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
