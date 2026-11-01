from pathlib import Path
from types import SimpleNamespace as NS
from threading import Thread
import json,os,time,copy
import pytest
from io_observer import atomic_write,arrival_check,install,RETRIES

def test_real_windows_locked_reader_then_retry(tmp_path):
    import formal_common
    path=tmp_path/'STATUS.json';formal_common.write(path,{'old':1})
    held=path.open('rb')
    try:
        with pytest.raises(PermissionError):formal_common.write(path,{'new':2})
        def release():time.sleep(.15);held.close()
        thread=Thread(target=release);thread.start();atomic_write(path,{'new':2});thread.join()
        assert json.loads(path.read_text())=={'new':2} and RETRIES[-1]['attempts']>0
    finally:held.close()

def test_persistent_lock_not_silently_accepted(tmp_path):
    p=tmp_path/'critical.json';atomic_write(p,{'old':1})
    with p.open('rb'):
        with pytest.raises(PermissionError):atomic_write(p,{'new':2},timeout_s=.05)
    assert json.loads(p.read_text())=={'old':1}

def test_declared_time_tolerance_and_negative_controls():
    tasks={'a':NS(arrival_s=9377.797976806285)}
    events=[dict(task_id='a',time_s=9377.797976806243)]
    assert arrival_check(events,tasks,1e-7)['passed']
    assert not arrival_check(events,tasks,1e-7)['literal_timestamp_equality']
    assert not arrival_check(events+events,tasks,1e-7)['passed']
    bad=copy.deepcopy(events);bad[0]['time_s']+=1e-5
    assert not arrival_check(bad,tasks,1e-7)['passed']
    assert not arrival_check([],tasks,1e-7)['passed']

def test_observer_does_not_change_decisions_physics_or_evaluation(tmp_path):
    from test_network_formal6h import scene,SEED
    from formal_common import job_for
    from formal_worker import execute
    a=scene();job=job_for(SEED,'PROP')
    execute(job,tmp_path/'before',120.,experiment=a,identities={'test':'equivalent'})
    install();b=scene()
    execute(job,tmp_path/'prefix',60.,experiment=b,identities={'test':'equivalent'})
    c=execute(job,tmp_path/'after',120.,resume=tmp_path/'prefix/CHECKPOINTS/T00060.pkl',identities={'test':'equivalent'})
    assert a.physics.identity()==c.physics.identity()
    for key in ['decisions','arrival_events','ground_ledger','closed','pending']:assert getattr(a,key)==getattr(c,key)
    assert a.evaluate()==c.evaluate()

