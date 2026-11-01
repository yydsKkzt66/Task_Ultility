from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import time
import psutil
from memory_guard import memory_status

HERE=Path(__file__).resolve().parent
MAIN=HERE.parent/'MAIN_RUNS'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
deadline=time.monotonic()+180
last=None
while True:
    rows=[]
    for seed in (4101,4102,4105):
        folder=MAIN/f'{seed}_PROP'
        attempts=sorted(p for p in folder.glob('attempt_*') if p.is_dir())
        a=attempts[-1]
        s=read(a/'STATUS.json') if (a/'STATUS.json').exists() else {}
        live=False;cpu=None
        if 'pid' in s:
            try:
                p=psutil.Process(s['pid'])
                live=abs(p.create_time()-s['process_created'])<.1
                if live:cpu=sum(p.cpu_times()[:2])
            except psutil.Error:pass
        restore=a.parent/(a.name+'_RESTORE_CHECK.json')
        row=dict(seed=seed,attempt=str(a),pid=s.get('pid'),alive=live,time_s=s.get('time_s'),
                 arrivals=s.get('arrivals'),closed=s.get('closed'),decisions=s.get('decisions'),
                 cpu_seconds=cpu,phase=s.get('phase'),failed=(a/'FAILED.json').exists(),
                 restore_check=read(restore) if restore.exists() else None)
        rows.append(row)
    marker=[(r['seed'],r['pid'],r['alive'],r['phase']) for r in rows]
    if marker!=last:
        print('RECOVERY_STATE',marker,flush=True);last=marker
    if all(r['alive'] and r['restore_check'] and r['restore_check']['pass_'] for r in rows):
        time.sleep(10)
        for r in rows:
            p=psutil.Process(r['pid']);new=read(Path(r['attempt'])/'STATUS.json')
            r['cpu_gain_10s']=sum(p.cpu_times()[:2])-r['cpu_seconds']
            r['decision_gain_10s']=new['decisions']-r['decisions']
            r['simulation_gain_10s']=new['time_s']-r['time_s']
            r['time_s']=new['time_s'];r['arrivals']=new['arrivals'];r['closed']=new['closed']
            r['failed']=(Path(r['attempt'])/'FAILED.json').exists()
        break
    if time.monotonic()>=deadline:break
    time.sleep(10)
sup=read(MAIN/'SUPERVISOR.json');ground=psutil.Process(15288)
report=dict(time_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
            all_three_prop_restored=all(r['alive'] and r['restore_check'] and r['restore_check']['pass_'] and not r['failed'] and r.get('cpu_gain_10s',0)>.1 for r in rows),
            rows=rows,supervisor=sup,memory=memory_status(),ground_4102_alive=ground.is_running(),
            ground_4102_pid=ground.pid)
(HERE/'RECOVERY_HEALTH.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
