"""Replace only the monitoring process; adopt live simulation workers unchanged."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,os,time,json,subprocess,re
import psutil
from formal_common import read,verify,alive
from io_observer import atomic_write
from memory_guard import VERSION,memory_status,can_launch,approved_memory_resume

def main(source,out,here):
    verify();out.mkdir(exist_ok=True);jobs=[j for j in read(source/'RUN_MATRIX.json')['jobs'] if j['group']=='main']
    lock=out/'SUPERVISOR.json'
    if lock.exists() and alive(read(lock)):raise RuntimeError('ANOTHER_SUPERVISOR_ACTIVE')
    atomic_write(lock,dict(pid=os.getpid(),process_created=psutil.Process().create_time(),version=VERSION,command=sys.argv))
    running={};logs={};saved_exit=set();started_at=time.time()
    def attempts(j):return sorted(p for p in (out/f"{j['seed']}_{j['arm']}").glob('attempt_*') if p.is_dir())
    def tail(p):
        if not p.exists():return ''
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));return f.read().decode('utf-8',errors='replace')
    def state(j):
        aa=attempts(j);key=(j['seed'],j['arm']);child=running.get(key)
        if not aa:return ('STARTING' if child and child.poll() is None else 'PENDING'),None,{}
        a=aa[-1];s=read(a/'STATUS.json') if (a/'STATUS.json').exists() else {}
        if (a/'COMPLETED.json').exists() and read(a/'AUDIT.json')['pass']:return 'COMPLETED',a,s
        if alive(s):return 'RUNNING',a,s
        if child and child.poll() is None:return 'STARTING',a,s
        if child and child.pid not in saved_exit:
            atomic_write(a/'PROCESS_EXIT.json',dict(pid=child.pid,returncode=child.poll(),epoch=time.time()));saved_exit.add(child.pid)
        text=tail(a.parent/(a.name+'.log'))
        failure=read(a/'FAILED.json') if (a/'FAILED.json').exists() else {}
        known_io=('PermissionError' in text and 'STATUS.json' in text) or ('PermissionError' in failure.get('traceback','') and 'STATUS.json' in failure.get('traceback',''))
        completed_audit_only='RESULT_AUDIT_FAILED' in failure.get('error','') and read(a/'AUDIT.json')['failures']==['arrival_identity']
        known_memory=approved_memory_resume(a)
        patched_attempts=sum((x.parent/(x.name+'_OBSERVER.json')).exists() for x in aa)
        if (a/'LATEST_CHECKPOINT.json').exists() and (known_io or completed_audit_only or known_memory) and patched_attempts<4:return 'RESUME_QUEUED',a,s
        return 'FAILED_REVIEW',a,s
    while True:
        states=[(j,*state(j)) for j in jobs];active=sum(st in {'RUNNING','STARTING'} for _,st,_,_ in states)
        queued=[x for x in states if x[1] in {'RESUME_QUEUED','PENDING'}]
        queued.sort(key=lambda x:(x[0]['arm']!='PROP',-(x[3].get('time_s') or 0),x[0]['seed']))
        for j,st,a,s in queued:
            if active>=5 or not can_launch(memory_status()):break
            parent=out/f"{j['seed']}_{j['arm']}";parent.mkdir(exist_ok=True)
            n=max([int(x.name.split('_')[1]) for x in attempts(j) if re.fullmatch('attempt_[0-9]+',x.name)]+[0])+1
            target=parent/f'attempt_{n:02d}'
            command=[sys.executable,'-I',str(here/'recovery_entry.py'),'--source',str(source),'worker','--out',str(target),'--seed',str(j['seed']),'--arm',j['arm']]
            if a:command+=['--resume',str(a/read(a/'LATEST_CHECKPOINT.json')['path'])]
            log=(parent/(target.name+'.log')).open('ab')
            child=subprocess.Popen(command,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            running[(j['seed'],j['arm'])]=child;logs[child.pid]=log;active+=1
            atomic_write(parent/(target.name+'_LAUNCH.json'),dict(command=command,pid=child.pid,version=VERSION,epoch=time.time()))
            print('RECOVER',j['seed'],j['arm'],child.pid,'from',s.get('time_s'),flush=True)
            break  # One launch per 60s cycle; observe real host commit before the next.
        rows=[]
        for j in jobs:
            st,a,s=state(j)
            if st in {'RESUME_QUEUED','PENDING'} and not can_launch(memory_status()):st='WAITING_HOST_MEMORY'
            row=dict(seed=j['seed'],arm=j['arm'],status=st,attempt=str(a) if a else None,**{k:s.get(k) for k in ['time_s','arrivals','closed','active_wall_s','total_wall_s','cpu_s','rss_bytes','peak_rss_bytes','pid','process_created']})
            if st=='RUNNING':
                try:
                    p=psutil.Process(row['pid']);row['rss_bytes']=p.memory_info().rss;row['live_cpu_s']=sum(p.cpu_times()[:2])
                except psutil.Error:pass
            t=row.get('time_s') or 0;wall=row.get('active_wall_s') or 0
            row['ETA_remaining_s_linear_observed']=wall/t*(21600-t) if t>=3600 and st=='RUNNING' else None
            rows.append(row)
        atomic_write(out/'PROGRESS.json',dict(updated_epoch=time.time(),rows=rows,supervisor_version=VERSION,memory_available_bytes=psutil.virtual_memory().available,host_memory=memory_status()))
        now=datetime.now(timezone(timedelta(hours=8)))
        lines=['# 正式运行进度','',f'更新时间：{now:%Y-%m-%d %H:%M:%S}（北京时间）','',
            '已接管日志恢复：原 INTERRUPTED/FAILED 记录保留，新attempt使用同一冻结决策、物理与输入。Windows状态文件占用采用短暂重试。',
            '', '|方法|种子|状态|仿真秒|到达|闭环|有效墙钟秒|含中断墙钟秒|CPU秒|内存MB|PID|粗略剩余h|',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            fmt=lambda x:'—' if x is None else f'{x:.1f}' if isinstance(x,float) else str(x)
            values=[r[k] for k in ['arm','seed','status','time_s','arrivals','closed','active_wall_s','total_wall_s','cpu_s']]+[r['rss_bytes']/1e6 if r['rss_bytes'] else None,r['pid'],r['ETA_remaining_s_linear_observed']/3600 if r['ETA_remaining_s_linear_observed'] is not None else None]
            lines.append('|'+ '|'.join(map(fmt,values))+'|')
        lines+=['','每60s更新；最大同时5个仿真作业，保留正在运行的作业。ETA按已完成进度平均速度估计，窗口续评会使实际时间变化；排队等待另计。',
            'FAILED_REVIEW不会自动略过审计；未完成不计0。TUT4103原审计失败保留，按预先声明1e-7s时间比较容差重审，原始事件与所有科学数值不改。']
        mem=memory_status()
        lines += ['', f"Host commit headroom: {mem['commit_available_bytes']/1024**3:.2f} GiB; physical available: {mem['physical_available_bytes']/1024**3:.2f} GiB. New launch requires commit >= 4 GiB and physical >= 3 GiB. WAITING_HOST_MEMORY means saved checkpoint queued, not simulating. At most one new worker per 60s."]
        # Open/write, no rename, for a human-facing non-authoritative view.
        (out/'00_PROGRESS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        from formal_summary import summarize
        summarize(out,plots=all(r['status']=='COMPLETED' for r in rows))
        if all(r['status'] in {'COMPLETED','FAILED_REVIEW'} for r in rows):
            atomic_write(out/'RECOVERY_SUPERVISOR_FINISHED.json',dict(epoch=time.time(),rows=rows,all_completed=all(r['status']=='COMPLETED' for r in rows)));break
        time.sleep(60.)

