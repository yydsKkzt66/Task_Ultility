"""Persistent A-run recovery and audited-result-to-figure pipeline; no B launches."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import psutil

HERE=Path(__file__).resolve().parent
PACK=HERE.parent
ROOT=HERE.parents[2]/'.network_ground_parallel_work'
OUT=PACK/'A_RUNS'
sys.path.insert(0,str(ROOT))
import run
sys.path.insert(0,str(PACK/'RUN_CONTROL_5'))
import formal_common as fc
import ground_queue as q
from runtime_io import write,status_read
from window_metrics import rows as window_rows
from collect_results import collect


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--detach',action='store_true');ap.add_argument('--takeover',action='store_true')
    a=ap.parse_args();ids=fc.verify();read=fc.read
    supervisor=OUT/'SUPERVISOR.json'
    if supervisor.exists() and fc.alive(read(supervisor)):
        old=read(supervisor)
        if not a.takeover:raise SystemExit('ACTIVE_SUPERVISOR_NO_DUPLICATE')
        proc=psutil.Process(old['pid']);cmd=proc.cmdline()
        assert any(('RUN_CONTROL_5' in part and part.endswith('control.py')) or
                   (part.endswith('auto_finish.py') and Path(part).resolve()==Path(__file__).resolve()) for part in cmd),cmd
        write(HERE/f"PREVIOUS_SUPERVISOR_{old['pid']}.json",dict(identity=old,command=cmd))
        proc.terminate();proc.wait(timeout=10)
    if a.detach:
        cmd=[sys.executable,str(Path(__file__))]
        with (HERE/'auto_finish.log').open('ab') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                 creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_BREAKAWAY_FROM_JOB if os.name=='nt' else 0)
        write(HERE/'LAUNCH.json',dict(pid=proc.pid,command=cmd,created=q.stamp(),scope='A recovery plus audited result collection; B data import only'))
        print('SUPERVISOR_PID',proc.pid);return
    write(supervisor,dict(pid=os.getpid(),process_created=psutil.Process().create_time(),package='A',max_workers=5,
                         controller_sha256=fc.sha(__file__),automatic_clock_recovery=True,automatic_figure_update=True))
    q.read=lambda p:status_read(p,read);q.write=write;fc.write=write
    jobs=[j for j in read(ROOT/'RUN_MATRIX.json')['jobs'] if j['package']=='A']
    jobs.sort(key=lambda j:(j['arm']!='GROUND',j['seed']))
    cached=read(OUT/'WINDOW_NET_SUMMARY.json').get('by_job',{})
    blocked=set();launches={};last_memory=0;previous_signature=None;plot_process=None;pending_signature=None
    while True:
        try:
            states=[(j,*q.scan(OUT/f"{j['seed']}_{j['arm']}",{})) for j in jobs]
            active=sum(st=='RUNNING' for _,st,_,_ in states)
            for j,st,p,s in states:
                if st in {'RESUMABLE','INTERRUPTED'} and p is not None and (p/'LATEST_CHECKPOINT.json').exists() and active<5 and psutil.virtual_memory().available>=2*1024**3:
                    if str(p) in launches:continue
                    new=p.parent/f"attempt_{1+max(int(x.name.split('_')[1]) for x in p.parent.glob('attempt_*') if x.is_dir()):02d}"
                    cmd=[sys.executable,str(HERE/'resume_saved.py'),'--previous',str(p),'--out',str(new)]
                    with (p.parent/(new.name+'.log')).open('ab') as log:
                        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    write(HERE/f"RESUME_{j['seed']}_{j['arm']}_{new.name}.json",dict(pid=proc.pid,command=cmd,time=q.stamp()))
                    active+=1;launches[str(p)]=str(new);continue
                if st!='FAILED' or str(p) in blocked or active>=5 or psutil.virtual_memory().available<2*1024**3:continue
                if not (p/'AUDIT.json').exists():continue
                audit=read(p/'AUDIT.json')
                if audit.get('failures')!=['arrival_identity']:continue
                ident=read(p/'IDENTITY.json')
                if ident.get('execution_overlay',{}).get('version')=='GROUND_OFFLOAD_EXACT_ARRIVAL_TAIL_V2':
                    blocked.add(str(p));continue
                new=p.parent/f"attempt_{1+max(int(x.name.split('_')[1]) for x in p.parent.glob('attempt_*') if x.is_dir()):02d}"
                cmd=[sys.executable,str(HERE/'recover_job.py'),'--seed',str(j['seed']),'--arm',j['arm'],'--previous',str(p),'--out',str(new)]
                completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
                record=dict(time=q.stamp(),command=cmd,returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
                write(HERE/f"AUTO_RECOVERY_{j['seed']}_{j['arm']}_{new.name}.json",record)
                if completed.returncode:blocked.add(str(p))
                else:active+=1;launches[str(p)]=str(new)
            report=q.summarize(OUT,jobs,{},'AUTOMATIC_RECOVERY_AND_AUDITED_FIGURE_UPDATE')
            for r in report:
                p=Path(r['result_path']) if r['result_path'] else None
                if p is None:continue
                key=f"{r['seed']}_{r['arm']}"
                if (p/'WINDOW_NET.json').exists():cached[key]=read(p/'WINDOW_NET.json')['rows'];continue
                needed=sum((r['time_s'] or 0)>=t for t in [4200,9900,15600,21300]);known=len(cached.get(key,[]))
                if needed<=known or not (p/'LATEST_CHECKPOINT.json').exists():continue
                ck=q.checkpoint(p);meta=read(ck.with_suffix('.json'))
                if sum(meta['time_s']>=t for t in [4200,9900,15600,21300])<=known:continue
                assert meta['identity']['manifests']==ids and fc.sha(ck)==meta['pickle_sha256']
                e=fc.load_checkpoint(ck,'B1' if r['arm']=='GROUND' else r['arm']);before=e.physics.identity()
                cached[key]=window_rows(e,source='VERIFIED_CHECKPOINT_READ_ONLY');assert before==e.physics.identity();del e
            flat=[x for rr in cached.values() for x in rr]
            write(OUT/'WINDOW_NET_SUMMARY.json',dict(by_job=cached,note='Actual cumulative gross-E-B; finalQ at21600'))
            if flat:q.csvout(OUT/'WINDOW_NET_BY_SEED.csv',flat)
            path=OUT/'00_PROGRESS_CN.md';text=path.read_text(encoding='utf-8').replace('最多4并行','最多5并行').replace('60秒更新进度','30秒更新进度')
            text+='\n已启用：仅对已定位的正向浮点到达尾差自动从故障前检查点恢复；原失败记录保留。通过原最终审计后自动计入统计和图，其他审计失败不强制改为成功。\n'
            text+='\n## 各窗口结束时的累计净效用\n\n单位沿用公共评价；每格为 **总体 / HV**。\n\n|方法|种子|第1窗4200s|第2窗9900s|第3窗15600s|第4窗21300s|\n|---|---:|---:|---:|---:|---:|\n'
            for j in jobs:
                rr={int(x['window']):x for x in cached.get(f"{j['seed']}_{j['arm']}",[])}
                cells=[f"{rr[i]['all_net']:.6f} / {rr[i]['HV_net']:.6f}" if i in rr else '—' for i in range(1,5)]
                text+='|'+'|'.join([j['arm'],str(j['seed'])]+cells)+'|\n'
            text+='\n窗口值=实际到地毛效用−已发生固定E/B费用，不提前扣终态Q；正式21600秒结果结算Q。未完成不填0。\n'
            data,state=collect()
            text+=f"\n## 自动统计与图\n\n已验收 {state['accepted']}/20 组主实验（含复用的因果TUT和正式PROP）。\n\n"
            text+='[逐种子合格结果](../MAIN_FIGURE_UPDATED/ACCEPTED_PER_SEED.csv) · [新版图与版本说明](../MAIN_FIGURE_UPDATED/00_README_CN.md)\n\n'
            text+='五种子不齐时只生成明确标注样本数的进度图；全20组齐全后自动生成正式5种子均值±样本标准差图。B包只导入回传结果，不在本机启动。\n'
            path.write_text(text,encoding='utf-8')
            if plot_process is not None and plot_process.poll() is not None:
                if plot_process.returncode==0:previous_signature=pending_signature
                else:write(HERE/'PLOT_LAST_ERROR.json',dict(time=q.stamp(),returncode=plot_process.returncode))
                plot_process=None
            if state['signature']!=previous_signature and plot_process is None:
                with (HERE/'plot_update.log').open('ab') as log:
                    plot_process=subprocess.Popen([sys.executable,str(HERE/'plot_results.py')],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                pending_signature=state['signature']
            progress=read(OUT/'PROGRESS.json');progress.update(max_parallel=5,window_rows=flat,plot_state=state)
            write(OUT/'PROGRESS.json',progress)
            if time.time()-last_memory>=1200:
                with (OUT/'MEMORY_20MIN.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(time=q.stamp(),max_workers=5,**psutil.virtual_memory()._asdict()))+'\n')
                last_memory=time.time()
            if all(r['status']=='COMPLETED' for r in report):
                write(OUT/'FINISHED.json',dict(time=q.stamp(),all_completed=True,rows=report))
            write(HERE/'HEARTBEAT.json',dict(time=q.stamp(),pid=os.getpid(),accepted=state['accepted'],rows=report,blocked_recoveries=sorted(blocked)))
            if state['ready_for_formal_figure'] and previous_signature==state['signature']:break
        except Exception as error:
            import traceback
            write(HERE/'PIPELINE_ERROR.json',dict(time=q.stamp(),error=repr(error),traceback=traceback.format_exc()))
        time.sleep(30)
    write(HERE/'PIPELINE_COMPLETED.json',dict(time=q.stamp(),all20_complete=True))


if __name__=='__main__':main()
