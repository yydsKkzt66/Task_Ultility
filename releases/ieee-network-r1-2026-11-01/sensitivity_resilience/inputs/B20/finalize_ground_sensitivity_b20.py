"""Collect only verified B20 metrics; automatically deliver a small final ZIP.

No simulation runs or result modifications. Pending/failed attempts are never
counted as scientific results. Polling only examines completion markers.
"""
from pathlib import Path
import argparse,csv,hashlib,html,json,math,os,shutil,statistics,subprocess,sys,time,traceback,zipfile
from datetime import datetime

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from summarize_ground_parallel_b5 import workbook
from io_retry_runner import make_retry_replace

FIELDS={
 'overall_net':('总体净效用','utility','全部到达任务实际到地收益减其可控资源费用'),
 'hv_net':('HV净效用','utility','high+urgent任务实际收益减其实际可控资源费用'),
 'ordinary_net':('普通任务净效用','utility','low+medium任务实际收益减其实际可控资源费用'),
 'overall_gross':('总体毛效用','utility','观察期内实际有效信息收益'),
 'energy_cost':('能耗扣分E','utility','实际可控J/3600乘统一固定每Wh价格'),
 'downlink_cost':('下传扣分B','utility','实际十进制MB乘统一固定每MB价格'),
 'queue_cost':('期末ready积压扣分Q','utility','21600s ready积压按固定价格结算'),
 'fixed_cost':('资源总扣分','utility','E+B+Q'),
 'hv_closed':('HV闭环数','tasks','high+urgent，必须有效满足需求'),
 'ordinary_closed':('普通闭环数','tasks','low+medium，必须有效满足需求'),
 'total_closed':('总体闭环数','tasks','HV与普通闭环数之和'),
 'hv_closure_rate_percent':('HV闭环率','%','HV闭环数/448*100'),
 'ordinary_closure_rate_percent':('普通闭环率','%','普通闭环数/1152*100'),
 'closure_rate_percent':('总体闭环率','%','总体闭环数/1600*100'),
 'controllable_energy_kj':('可控能耗','kJ','推理+编码+实际RF发送；背景单列不重复计费'),
 'downlink_mb':('实际下传量','MB','十进制MB；全部实际传输成本保留'),
 'latency_mean_s':('闭环时延均值','s','只统计本方法已闭环任务，真实到达到首次有效满足需求'),
 'latency_p95_s':('闭环时延95分位','s','仅闭环样本，(n-1)*0.95线性插值'),
 'hv_latency_mean_s':('HV闭环时延均值','s','仅实际闭环的high+urgent任务'),
 'hv_latency_p95_s':('HV闭环时延95分位','s','仅实际闭环的high+urgent任务'),
 'effective_delivered':('有有效到地产品任务数','tasks','最高有效产品按任务去重；每任务计一次'),
 'partial_only':('仅部分交付任务数','tasks','有有效产品但未满足最终需求'),
 'no_valid_product':('无有效到地产品任务数','tasks','最高有效产品为none'),
 'right_censored_tasks':('期末右删失','tasks','21600s未闭环但尚未过DDL'),
 'deadline_failed_tasks':('已过DDL未闭环','tasks','与已闭环、右删失互斥'),
 'actual_safety_violations':('实际安全违例','events','沿用冻结审计阈值与公差')}
for product in ['metadata','roi','roi_context']:
    for suffix,label in [('satisfied','满足需求'),('partial','仅部分交付'),('total','总数')]:
        FIELDS[product+'_'+suffix]=(product+' '+label,'tasks','每个任务仅计最高有效到地产品，满足需求与部分交付互斥')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(p,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def near(a,b):assert math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-9),(a,b)


def snapshot(root,out):
    pf=read(out/'LOCAL_CHECKS/PREFLIGHT.json')
    jobs=pf['jobs'];rows=[];groups=[];provenance=[];prices=None
    for job in jobs:
        attempts=sorted(p for p in (out/job['job_id']).glob('attempt_*') if p.is_dir())
        verified=[p for p in attempts if (p/'VERIFIED_COMPLETE.json').is_file()]
        if not verified:continue
        p=verified[-1];v=read(p/'VERIFIED_COMPLETE.json')
        assert v['job']==job and v['source_audit_pass'] and v['sensitivity_audit_pass'] and v['metrics_pass']
        assert v['manifests']['SOURCE_MANIFEST.json']==pf['frozen']['source_sha256']
        assert v['manifests']['INPUT_MANIFEST.json']==pf['frozen']['input_sha256']
        assert read(p/'AUDIT.json')['pass'] and read(p/'SENSITIVITY_AUDIT.json')['pass_']
        assert read(p/'METRICS/METRIC_RECONCILIATION.json')['metric_reconciliation_pass']
        cfg=read(p/'RESULT/CONFIG_EFFECTIVE.json')
        if prices is None:prices=cfg['public_prices']
        assert prices==cfg['public_prices']
        rr=csvread(p/'METRICS/metrics.csv');assert len(rr)==1
        raw=rr[0];assert set(FIELDS).issubset(raw)
        r={k:(float(value) if value else None) if k not in ['method','seed','condition'] else int(value) if k=='seed' else value for k,value in raw.items()}
        assert r['method']==job['arm'] and r['seed']==job['seed'] and r['condition']==job['condition']
        assert r['arrivals']==1600 and r['hv_arrivals']==448 and r['ordinary_arrivals']==1152
        assert r['actual_safety_violations']==0
        near(r['overall_gross']-r['fixed_cost'],r['overall_net'])
        near(sum(r[k] for k in ['energy_cost','downlink_cost','queue_cost']),r['fixed_cost'])
        near(r['overall_net'],r['hv_net']+r['ordinary_net'])
        near(r['controllable_energy_kj']/3.6*prices['energy_price_per_Wh'],r['energy_cost'])
        near(r['downlink_mb']*prices['downlink_price_per_MB'],r['downlink_cost'])
        near(r['hv_closed']/448*100,r['hv_closure_rate_percent'])
        near(r['ordinary_closed']/1152*100,r['ordinary_closure_rate_percent'])
        near(r['hv_closed']+r['ordinary_closed'],r['total_closed'])
        near(r['total_closed']+r['partial_only'],r['effective_delivered'])
        near(r['total_closed']+r['right_censored_tasks']+r['deadline_failed_tasks'],1600)
        for product in ['metadata','roi','roi_context']:
            near(r[product+'_satisfied']+r[product+'_partial'],r[product+'_total'])
        near(sum(r[p+'_total'] for p in ['metadata','roi','roi_context']),r['effective_delivered'])
        r.update(job_id=job['job_id'],source_attempt=str(p),audit_pass=True,
            source_sha256=pf['frozen']['source_sha256'],input_sha256=pf['frozen']['input_sha256'])
        rows.append(r)
        groups.extend(dict(job_id=job['job_id'],condition=job['condition'],method=job['arm'],seed=job['seed'],**g)
                      for g in csvread(p/'METRICS/group_metrics.csv'))
        for rel in ['VERIFIED_COMPLETE.json','AUDIT.json','SENSITIVITY_AUDIT.json','IDENTITY.json',
                    'METRICS/metrics.csv','METRICS/METRIC_RECONCILIATION.json','RESULT/CONFIG_EFFECTIVE.json']:
            provenance.append(dict(job_id=job['job_id'],path=str(p/rel),sha256=sha(p/rel)))
    return rows,groups,provenance,prices


def render(out,rows,groups,provenance,prices):
    dest=out/'METRICS_DELIVERY';dest.mkdir(exist_ok=True)
    fields=['job_id','condition','method','seed',*FIELDS,'source_attempt','audit_pass']
    fields=list(dict.fromkeys(fields+list(dict.fromkeys(k for r in rows for k in r))))
    csvsave(dest/'per_seed_metrics.csv',rows,fields)
    if groups:csvsave(dest/'group_metrics.csv',groups)
    if provenance:csvsave(dest/'provenance_sha256.csv',provenance)
    dictionary=[dict(metric=k,label_CN=v[0],unit=v[1],definition_CN=v[2]) for k,v in FIELDS.items()]
    csvsave(dest/'metric_dictionary.csv',dictionary)
    stats=[]
    for condition,method in sorted({(r['condition'],r['method']) for r in rows}):
        subset=[r for r in rows if r['condition']==condition and r['method']==method]
        seeds={r['seed'] for r in subset};n=len(subset)
        for metric in FIELDS:
            values=[r[metric] for r in subset]
            stats.append(dict(condition=condition,method=method,metric=metric,n=n,seeds=';'.join(map(str,sorted(seeds))),
                mean=statistics.mean(values) if all(v is not None for v in values) else None,
                sample_SD=statistics.stdev(values) if n>1 and all(v is not None for v in values) else None,
                complete_three_seeds=seeds=={4101,4102,4103},scope='AVAILABLE_SUBSET_ONLY_UNLESS_ALL_THREE_SEEDS_PRESENT'))
    if stats:csvsave(dest/'available_subset_statistics.csv',stats)
    instructions=[
        'B20地面敏感性指标汇总',
        f'完成且原审计、敏感性审计、指标对账通过：{len(rows)}/20；更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}。',
        '本机范围11组GROUND、9组OFFLOAD；A包另有22组，不用旧地面结果补空缺。完整同条件三种子统计需合并A包；当前统计文件明确给出n、种子及完整性标记。',
        '总体净效用汇总所有到达任务，HV净效用汇总high和urgent。每组使用相同固定E/B/Q价格，净效用=实际毛效用−能耗费用−实际下传费用−21600s期末ready积压费用。',
        '分别报告HV/普通任务闭环数与闭环率；最高有效Metadata、ROI、ROI+Ctx每任务只计一次，区分满足需求及仅部分交付。',
        '可控能耗单位kJ，实际下传单位十进制MB。平均闭环时延、P95闭环时延、HV平均闭环时延均只覆盖本方法已闭环任务；时延从真实到达到首次有效满足需求，P95按(n−1)×0.95线性插值。',
        '原实验、账本、候选证据和检查点保留；本交付只收集轻量指标及来源哈希。地面推理配置为完整接收后独立4.29s；20s宏槽、真实物理与事件、DDL、1600任务/21600s均保留。',
        '原始逐任务产品与闭环数据位于各成功attempt/METRICS/task_product_and_closure.csv；未重复打包完整原始日志。',
    ]
    (dest/'00_REPORT_CN.txt').write_text('\n\n'.join(instructions)+'\n',encoding='utf-8-sig')
    columns=[('condition','条件'),('method','方法'),('seed','种子'),('overall_net','总体净效用'),('hv_net','HV净效用'),
        ('overall_gross','毛效用'),('fixed_cost','资源总扣分'),('hv_closed','HV闭环'),('ordinary_closed','普通闭环'),
        ('controllable_energy_kj','可控能耗 kJ'),('downlink_mb','实际下传 MB'),('latency_mean_s','平均时延 s'),
        ('latency_p95_s','P95时延 s'),('hv_latency_mean_s','HV平均时延 s')]
    def fmt(v):return '—' if v is None else f'{v:.3f}' if isinstance(v,float) else html.escape(str(v))
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B20完整指标汇总</title><style>body{font:16px/1.8 "Microsoft YaHei",sans-serif;margin:30px;color:#263c48}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap}td,th{padding:7px 10px;border:1px solid #cdd8de;text-align:right}th{background:#244b61;color:white}.note{background:#edf4f7;padding:14px}a{color:#126a9c}</style><h1>B20 地面敏感性指标汇总</h1>']
    page+=['<p class="note">'+html.escape(t)+'</p>' for t in instructions[1:3]]
    page+=['<p><a href="per_seed_metrics.csv">逐种子完整指标CSV</a> · <a href="metric_dictionary.csv">指标定义</a> · <a href="../00_PROGRESS_CN.md">实时推进表</a></p>']
    if rows:
        page+=['<p><a href="B20_METRICS.xlsx">Excel工作簿</a> · <a href="group_metrics.csv">分组费用及闭环</a> · <a href="available_subset_statistics.csv">明确样本数的统计表</a></p>']
    page+=['<div class="scroll"><table><tr>'+''.join('<th>'+label+'</th>' for _,label in columns)+'</tr>']
    page+=['<tr>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k,_ in columns)+'</tr>' for r in rows]
    page+=['</table></div><h2>统计口径</h2>']+['<p>'+html.escape(t)+'</p>' for t in instructions[3:]]
    page+=['</html>']
    (dest/'00_REPORT_CN.html').write_text('\n'.join(page),encoding='utf-8')
    if rows:
        workbook(dest/'B20_METRICS.xlsx',[('统计口径',[dict(item=i,text=t) for i,t in enumerate(instructions,1)]),
            ('逐种子完整指标',rows),('HV普通分组',groups),('当前子集统计',stats),('指标字典',dictionary),('证据来源',provenance)])
    write(dest/'DELIVERY_STATE.json',dict(updated=datetime.now().astimezone().isoformat(),verified_complete=len(rows),
        expected=20,all_completed=len(rows)==20,prices=prices,source_result_modified=False,
        per_method_per_condition_three_seeds_not_assumed=True))
    return dest


def finalize(out,dest,rows):
    assert len(rows)==20
    for name in ['CURRENT_FAILURES.csv','FAILURE_HISTORY.csv','LAUNCH_PLAN.json','FINAL_SOURCE_VERIFICATION.json']:
        if (out/name).exists():shutil.copyfile(out/name,dest/name)
    cleanup=out/'SAFE_CACHE_CLEANUP/REMOVED_FILES.csv'
    if cleanup.exists():
        removed=csvread(cleanup)
        summary={drive:dict(files=sum(r['path'].startswith(drive) for r in removed),
            bytes=sum(int(r['bytes']) for r in removed if r['path'].startswith(drive))) for drive in ['C:','E:']}
        write(dest/'CACHE_CLEANUP_SUMMARY.json',dict(removed=summary,experiment_data_and_checkpoints_preserved=True))
    shutil.copyfile(__file__,dest/Path(__file__).name)
    for helper in ['summarize_ground_parallel_b5.py','io_retry_runner.py']:
        shutil.copyfile(BASE/helper,dest/helper)
    zip_path=out/'B20_METRICS_RESULTS.zip';temp=out/'B20_METRICS_RESULTS.zip.tmp'
    files=sorted(p for p in dest.iterdir() if p.is_file())
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,(Path('B20_METRICS_RESULTS')/p.name).as_posix())
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None and len(z.namelist())==len(files)
        for p in files:assert hashlib.sha256(z.read('B20_METRICS_RESULTS/'+p.name)).hexdigest()==sha(p)
    os.replace(temp,zip_path)
    write(out/'FINAL_METRICS_READY.json',dict(at=datetime.now().astimezone().isoformat(),verified_jobs=20,
        report=str(dest/'00_REPORT_CN.html'),excel=str(dest/'B20_METRICS.xlsx'),archive=str(zip_path),
        archive_sha256=sha(zip_path),archive_bytes=zip_path.stat().st_size))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--detach',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args()
    import psutil
    root,out=Path(a.root).resolve(),Path(a.out).resolve()
    marker=out/'METRICS_COLLECTOR.json'
    if a.detach:
        if marker.exists():
            m=read(marker)
            try:
                proc=psutil.Process(m['pid'])
                if proc.is_running() and abs(proc.create_time()-m['created'])<.1:
                    print('Collector already running',proc.pid);return
            except psutil.Error:pass
        command=[str(Path(sys.executable).with_name('pythonw.exe')),'-I','-B',str(Path(__file__).resolve()),
            '--root',str(root),'--out',str(out)]
        with (out/'metrics_collector.log').open('ab') as f:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_BREAKAWAY_FROM_JOB)
        write(out/'METRICS_COLLECTOR_LAUNCH.json',dict(pid=proc.pid,command=command))
        print('METRICS_COLLECTOR_PID',proc.pid);return
    os.replace=make_retry_replace(os.replace,out,max_wait_s=30.)
    write(marker,dict(pid=os.getpid(),created=psutil.Process().create_time()))
    previous=None
    while True:
        try:
            rows,groups,provenance,prices=snapshot(root,out)
            identity=tuple((r['job_id'],r['source_attempt']) for r in rows)
            if identity!=previous:
                dest=render(out,rows,groups,provenance,prices);previous=identity
                print('VERIFIED_METRICS',len(rows),flush=True)
            else:dest=out/'METRICS_DELIVERY'
            if len(rows)==20:
                finalize(out,dest,rows);return
            if a.once:return
        except Exception as error:
            with (out/'METRICS_COLLECTOR_ERRORS.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()),ensure_ascii=False)+'\n')
            if a.once:raise
        time.sleep(60)


if __name__=='__main__':main()
