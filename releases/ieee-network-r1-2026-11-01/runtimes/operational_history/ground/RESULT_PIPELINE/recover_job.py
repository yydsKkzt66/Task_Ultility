"""Recover only a proven tiny positive arrival-clock defect, from an intact prefix."""
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
ROOT = HERE.parents[2] / '.network_ground_parallel_work'
sys.path.insert(0, str(ROOT))
import run
sys.path.insert(0, str(PACK / 'RUN_CONTROL_5'))
import formal_common as fc
from runtime_io import write, overlay_identity
from exact_arrival import identity, install, VERSION


def diagnose(previous, seed):
    audit = fc.read(previous/'AUDIT.json')
    assert audit['failures'] == ['arrival_identity'], 'NOT_A_CLOCK_ONLY_FAILURE'
    assert all(v for k, v in audit['checks'].items() if k != 'arrival_identity')
    with (ROOT/f'network_inputs/seed_{seed}/tasks.csv').open(encoding='utf-8-sig') as f:
        expected = {r['task_id']:float(r['arrival_time_s']) for r in csv.DictReader(f)}
    arrivals = fc.read(previous/'RESULT/arrival_events.json')
    assert len(arrivals) == len({r['task_id'] for r in arrivals}) == 1600
    assert set(expected) == {r['task_id'] for r in arrivals}
    bad = [dict(**r, expected_s=expected[r['task_id']], gap_s=expected[r['task_id']]-r['time_s'])
           for r in arrivals if r['time_s'] != expected[r['task_id']]]
    assert bad and all(0. < r['gap_s'] <= 1e-8 for r in bad), 'NOT_A_TINY_POSITIVE_GAP'
    first = min(r['expected_s'] for r in bad)
    checkpoints = [p for p in (previous/'CHECKPOINTS').glob('*.json') if fc.read(p)['time_s'] < first]
    assert checkpoints, 'NO_UNAFFECTED_SAVED_CHECKPOINT'
    meta = max(checkpoints, key=lambda p:fc.read(p)['time_s'])
    return dict(bad_arrivals=bad, first_bad_s=first, checkpoint=str(meta.with_suffix('.pkl')),
                old_audit_sha256=fc.sha(previous/'AUDIT.json'), old_identity_sha256=fc.sha(previous/'IDENTITY.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--arm', choices=['GROUND','OFFLOAD'], required=True)
    ap.add_argument('--previous', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--worker', action='store_true')
    a = ap.parse_args()
    previous, out = a.previous.resolve(), a.out.resolve()
    assert out.parent == previous.parent and out != previous
    assert out.parent.parent == (PACK/'A_RUNS').resolve(), 'LOCAL_A_ONLY'
    evidence = identity()
    valid = fc.read(HERE/'VALIDATION.json')
    assert valid['pass_'] and valid['execution_overlay'] == evidence
    base_ids = fc.verify()
    ids = dict(base_ids, **{VERSION:evidence['sha256']})
    work = HERE/'RECOVERIES'/f'{a.seed}_{a.arm}_{out.name}'
    prefix = work/'PREFIX.pkl'
    if not a.worker:
        assert not out.exists()
        record = diagnose(previous, a.seed)
        old_ck = Path(record['checkpoint'])
        old_meta = fc.read(old_ck.with_suffix('.json'))
        assert old_meta['identity']['manifests'] == base_ids, 'DO_NOT_REMIGRATE_UNKNOWN_VERSION'
        assert old_meta['identity']['job'] == fc.job_for(a.seed,a.arm)
        assert fc.sha(old_ck) == old_meta['pickle_sha256']
        exp = fc.load_checkpoint(old_ck, 'B1' if a.arm == 'GROUND' else 'OFFLOAD')
        assert all(r['time_s'] == exp.arrived[r['task_id']].arrival_s for r in exp.arrival_events)
        record.update(prefix_time_s=exp.physics.now, prefix_arrivals=len(exp.arrival_events),
                      parent_checkpoint_sha256=fc.sha(old_ck), parent_metadata_sha256=fc.sha(old_ck.with_suffix('.json')),
                      exact_prefix_no_trigger=True, execution_overlay=evidence,
                      policy='Original scientific parameters and strict final audit; actual positive tail integration')
        del exp
        work.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(old_ck, prefix)
        meta = dict(old_meta, identity=dict(old_meta['identity'], manifests=ids), prefix_migration=record)
        write(prefix.with_suffix('.json'), meta)
        write(work/'MIGRATION.json', record)
        write(work/'SOURCE_FREEZE.json', dict(base_manifests=base_ids, effective_manifests=ids,
              files={n:fc.sha(HERE/n) for n in ('exact_arrival.py','recover_job.py','validate.py','VALIDATION.json')}))
        cmd = [sys.executable,str(Path(__file__)), '--seed',str(a.seed),'--arm',a.arm,
               '--previous',str(previous),'--out',str(out),'--worker']
        log = out.parent/(out.name+'.log')
        with log.open('ab') as fp:
            proc = subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=fp,stderr=subprocess.STDOUT,
                 creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_BREAKAWAY_FROM_JOB if os.name=='nt' else 0)
        launch = dict(pid=proc.pid,command=cmd,epoch=time.time(),out=str(out),previous=str(previous),
                      resume_s=record['prefix_time_s'],execution_overlay=evidence)
        write(work/'LAUNCH.json',launch)
        print(json.dumps(launch,ensure_ascii=False),flush=True)
        return

    freeze = fc.read(work/'SOURCE_FREEZE.json')
    assert freeze['effective_manifests'] == ids
    assert all(fc.sha(HERE/n)==h for n,h in freeze['files'].items())
    install(ROOT)
    def output(path,value):
        if Path(path).name == 'IDENTITY.json':
            value = dict(value,execution_overlay=evidence,prefix_migration=fc.read(work/'MIGRATION.json'),operational_overlay=overlay_identity())
        elif Path(path).name == 'AUDIT.json':
            value = dict(value,execution_overlay=evidence,strict_arrival_identity_unchanged=True)
        write(path,value)
    fc.write = output
    from formal_worker import execute
    from src.network_revision.runtime import NetworkExperiment
    from window_metrics import record
    original_run = NetworkExperiment.run
    def observed_run(e,*args,**kw):
        original_hook = e.slot_complete_hook
        count = len(getattr(e,'exact_arrival_tail_records',[]))
        def hook(exp,slot):
            nonlocal count
            original_hook(exp,slot)
            if exp.physics.now in {4200.,9900.,15600.,21300.}:record(exp,out)
            tails = getattr(exp,'exact_arrival_tail_records',[])
            if len(tails)!=count:
                write(out/'ARRIVAL_TAIL_EVENTS.json',dict(version=VERSION,records=tails));count=len(tails)
        e.slot_complete_hook=hook
        record(e,out)
        return original_run(e,*args,**kw)
    NetworkExperiment.run=observed_run
    e=execute(fc.job_for(a.seed,a.arm),out,resume=prefix,identities=ids)
    old=fc.read(previous/'RESULT/AUDITED_SUMMARY.json');new=fc.read(out/'RESULT/AUDITED_SUMMARY.json')
    flat=lambda s:dict(gross=s['gross'],net=s['net'],HV_net=s['groups']['high_value']['net'],closed=s['closed'],**s['costs'])
    x,y=flat(old),flat(new)
    write(work/'RECOVERY_RESULT.json',dict(status='COMPLETED_AUDIT_PASS',old=x,new=y,delta={k:y[k]-x[k] for k in x},
          old_failed_result_retained=True,execution_overlay=evidence,tail_records=getattr(e,'exact_arrival_tail_records',[]),
          exact_arrivals=sum(r['time_s']==e.arrived[r['task_id']].arrival_s for r in e.arrival_events),audit_sha256=fc.sha(out/'AUDIT.json')))


if __name__=='__main__':main()
