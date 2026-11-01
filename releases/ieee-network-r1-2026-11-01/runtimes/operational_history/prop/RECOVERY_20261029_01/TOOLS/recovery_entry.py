from pathlib import Path
import argparse,sys,os,runpy,json,hashlib,time,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('mode',choices=['worker','supervise','selfcheck']);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int);ap.add_argument('--arm');ap.add_argument('--resume');ap.add_argument('--stop',type=float,default=21600.)
a=ap.parse_args();source=Path(a.source).resolve();out=Path(a.out).resolve()
_bootstrap=runpy.run_path(str(source/'run.py'),run_name='_recovery_environment')
sys.path.insert(0,str(HERE))
from io_observer import install,atomic_write,VERSION
from formal_common import verify,job_for,sha
identities=verify()
if a.mode=='selfcheck':
    import pytest
    sys.path.insert(0,str(source/'tests'));out.mkdir(parents=True,exist_ok=True)
    code=pytest.main(['-q','-p','no:cacheprovider',str(HERE/'test_io_recovery.py'),'--junitxml='+str(out/'pytest.xml')])
    atomic_write(out/'TEST_STATUS.json',dict(exitcode=code,manifests=identities));raise SystemExit(code)
install()
if a.mode=='worker':
    if out.exists():raise FileExistsError(out)
    if a.resume:
        from formal_common import read,load_checkpoint
        checkpoint=Path(a.resume);meta=read(checkpoint.with_suffix('.json'));job=job_for(a.seed,a.arm)
        assert meta['identity']==dict(job=job,manifests=identities,python=sys.version),'RESTORE_IDENTITY'
        assert sha(checkpoint)==meta['pickle_sha256'],'CHECKPOINT_SHA'
        e=load_checkpoint(checkpoint,job['method'])
        assert e.physics.now==meta['time_s'] and e.method==job['method'] and e.variant==job['variant'],'RESTORE_STATE'
        expected=sum(sum(d['candidate_categories'].values()) for d in e.decisions)
        assert expected==meta['candidate_count'],'RESTORE_CANDIDATE_COUNT'
        log=Path(meta['candidate_file'])
        if not log.exists():log=checkpoint.parent.parent/'candidates.jsonl'
        size=meta['candidate_bytes'];digest=hashlib.sha256();count=0;last=None
        with log.open('rb') as stream:
            while size:
                line=stream.readline();assert line and len(line)<=size,'RESTORE_PREFIX_TRUNCATION'
                digest.update(line);size-=len(line);count+=1;last=json.loads(line)
        assert count==expected,'RESTORE_PREFIX_COUNT'
        assert last is None or last['decision_s']<=e.physics.now,'RESTORE_FUTURE_CANDIDATE'
        atomic_write(out.parent/(out.name+'_RESTORE_CHECK.json'),dict(pass_=True,checkpoint_sha256=meta['pickle_sha256'],candidate_prefix_sha256=digest.hexdigest(),candidate_count=count,time_s=e.physics.now,arrivals=len(e.arrived),closed=len(e.closed),source_manifests=identities))
        del e
    # Observer provenance resides outside the frozen identity compared by the checkpoint.
    atomic_write(out.parent/(out.name+'_OBSERVER.json'),dict(version=VERSION,source_manifests=identities,source=str(source),
        observer_files={p.name:sha(p) for p in HERE.glob('*.py')},resume=a.resume,commands=sys.argv))
    from formal_worker import execute
    execute(job_for(a.seed,a.arm),out,stop=a.stop,resume=a.resume,identities=identities)
elif a.mode=='supervise':
    from recovery_supervisor import main
    main(source,out,HERE)
