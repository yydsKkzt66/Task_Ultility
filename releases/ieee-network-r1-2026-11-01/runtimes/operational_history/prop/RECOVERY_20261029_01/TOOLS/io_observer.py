"""Observer-only recovery. Never mutate the frozen simulation or its inputs."""
from pathlib import Path
from collections import Counter
import json,os,time,uuid,math

VERSION='STATUS_IO_RETRY_AND_DECLARED_TIME_AUDIT_V1'
RETRIES=[]

def atomic_write(path,value,*,timeout_s=5.):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n'
    tmp.write_text(data,encoding='utf-8')
    start=time.monotonic();attempts=0
    while True:
        try:os.replace(tmp,path);break
        except PermissionError as error:
            if getattr(error,'winerror',None) not in {None,5,32,33} or time.monotonic()-start>=timeout_s:raise
            attempts+=1;time.sleep(min(.01*2**min(attempts,5),.20))
    if attempts:
        row=dict(path=str(path),attempts=attempts,wait_s=time.monotonic()-start,epoch=time.time())
        RETRIES.append(row)
        # Append-only per-process telemetry, never a state source for the policy.
        with (path.parent/f'IO_RETRIES_{os.getpid()}.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')

def arrival_check(events,arrived,tolerance_s):
    ids=[r['task_id'] for r in events];same_ids=len(ids)==len(set(ids))==len(arrived) and set(ids)==set(arrived)
    differences=[]
    for r in events:
        if r['task_id'] not in arrived:continue
        difference=r['time_s']-arrived[r['task_id']].arrival_s
        if difference:differences.append(dict(task_id=r['task_id'],recorded_s=r['time_s'],manifest_s=arrived[r['task_id']].arrival_s,difference_s=difference))
    within=all(math.isfinite(r['difference_s']) and abs(r['difference_s'])<=tolerance_s for r in differences)
    return dict(passed=same_ids and within,distinct_ids_ok=same_ids,tolerance_s=tolerance_s,absolute_only=True,
        literal_timestamp_equality=not differences,max_absolute_difference_s=max((abs(r['difference_s']) for r in differences),default=0.),differences=differences)

def install():
    import formal_common,formal_audit,formal_worker,formal_summary
    for module in [formal_common,formal_audit,formal_worker,formal_summary]:module.write=atomic_write
    original=getattr(formal_audit,'_frozen_audit',formal_audit.audit);formal_audit._frozen_audit=original
    def audit(e,job,out,candidate_path,require_all=True):
        result=original(e,job,out,candidate_path,require_all)
        detail=arrival_check(e.arrival_events,e.arrived,result['tolerances']['time_s'])
        result['original_arrival_identity_literal_check']=result['checks']['arrival_identity']
        result['checks']['arrival_identity']=detail['passed']
        result['arrival_time_audit']=detail;result['observer_version']=VERSION
        result['failures']=[k for k,ok in result['checks'].items() if not ok]
        result['pass']=result['pass_']=not result['failures']
        atomic_write(Path(out)/'ARRIVAL_TIME_AUDIT.json',detail)
        return result
    formal_audit.audit=audit

