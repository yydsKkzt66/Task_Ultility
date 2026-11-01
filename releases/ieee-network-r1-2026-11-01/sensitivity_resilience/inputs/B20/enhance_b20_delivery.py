"""Read-only final B20 reconciliation, condition definitions and data figures."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse,csv,hashlib,html,json,math,os,shutil,statistics,sys,zipfile

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from finalize_ground_sensitivity_b20 import snapshot,FIELDS,csvread,csvsave,read,write,sha,near
from summarize_ground_parallel_b5 import workbook
from io_retry_runner import make_retry_replace

CONDITIONS=['A1','A2','A3','A4','A5','B1','B2']
NAMES={'A1':'Capacity 0.15','A2':'Capacity 0.30','A3':'Capacity 0.90',
       'A4':'Value weights 1:2:5:10','A5':'Value weights 1:5:20:50',
       'B1':'Outage [9300, 9900) s','B2':'Outage [9300, 15600) s'}
CN={'A1':'信道容量0.15','A2':'信道容量0.30','A3':'信道容量0.90',
    'A4':'任务价值权重1:2:5:10','A5':'任务价值权重1:5:20:50',
    'B1':'服务中断[9300,9900)s','B2':'服务中断[9300,15600)s'}
METHODS=['GROUND','OFFLOAD']
RANK=dict(none=0,metadata=1,roi=2,roi_context=3)


def independent(out,root,rows):
    import numpy as np
    verified=[]
    for r in rows:
        p=Path(r['source_attempt']);result=p/'RESULT'
        audit=read(p/'AUDIT.json');extra=read(p/'SENSITIVITY_AUDIT.json')
        assert audit['pass'] and not audit['failures'] and all(audit['checks'].values())
        assert extra['pass_'] and all(extra['checks'].values())
        assert read(p/'COMPLETED.json')['time_s']==21600
        tasks={t['task_id']:t for t in csvread(result/'task_accounting.csv')}
        assert len(tasks)==1600
        assert all((t['HV']=='True')==(t['value_level'] in ['high','urgent']) for t in tasks.values())
        events=read(result/'ground_ledger.json');first={};highest={}
        for event in sorted(events,key=lambda e:e['ground_available_s']):
            if not event['valid']:continue
            tid=event['task_id'];assert event['ground_available_s']<=min(float(tasks[tid]['deadline_s']),21600)+1e-7
            level=event['product']
            if RANK[level]>RANK[highest.get(tid,'none')]:highest[tid]=level
            if event['closes']:first.setdefault(tid,event['ground_available_s'])
        assert set(first)==set(read(result/'terminal_state.json')['closed'])
        for t in tasks.values():
            assert t['highest_effective_ground_product']==highest.get(t['task_id'],'none')
            assert (t['satisfies_requirement']=='True')==(t['task_id'] in first)
        for native,label in [('gross','overall_gross'),('E_cost','energy_cost'),('B_cost','downlink_cost'),('Q_cost','queue_cost')]:
            near(math.fsum(float(t[native]) for t in tasks.values()),r[label])
        for group,label in [('all',''),('HV','hv_'),('ordinary','ordinary_')]:
            ids=[tid for tid,t in tasks.items() if group=='all' or (t['HV']=='True')==(group=='HV')]
            closed=[tid for tid in ids if tid in first]
            values=[first[tid]-float(tasks[tid]['arrival_s']) for tid in closed]
            near(len(closed),r['total_closed' if group=='all' else label+'closed'])
            near(statistics.mean(values),r[label+'latency_mean_s'])
            near(float(np.quantile(values,.95,method='linear')),r[label+'latency_p95_s'])
            net=math.fsum(float(tasks[tid]['gross'])-sum(float(tasks[tid][k]) for k in ['E_cost','B_cost','Q_cost']) for tid in ids)
            near(net,r['overall_net' if group=='all' else label+'net'])
        counts=Counter(highest.values())
        for level in ['metadata','roi','roi_context']:near(counts[level],r[level+'_total'])
        assert int(r['actual_safety_violations'])==0
        verified.append(dict(job_id=r['job_id'],original_audit_pass=True,original_checks=len(audit['checks']),
            sensitivity_audit_pass=True,metrics_reconciled=True,task_costs_recomputed=True,
            highest_product_deduplicated=True,closed_latency_mean_and_p95_recomputed=True,
            actual_safety_violations=0,source_attempt=str(p)))
    package=root.parent
    frozen=csvread(package/'B_FILE_SHA256.csv')
    for f in frozen:assert sha(package/f['file'])==f['sha256']
    return verified,len(frozen)


def figures(dest,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':10,'axes.labelsize':10,'xtick.labelsize':8.2,'ytick.labelsize':9,
        'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.8})
    blue,teal,coral='#496f83','#79b4c4','#e28376'
    def style(ax,rr,ylabel,caption):
        ax.set_xticks(range(len(rr)),[r['condition']+'\n'+str(r['seed']) for r in rr])
        ax.tick_params(direction='in',top=True,right=True,length=2.3,width=.7)
        ax.grid(True,axis='both',color='#e2e5e8',linewidth=.55);ax.set_axisbelow(True)
        ax.set_ylabel(ylabel,labelpad=4)
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#555c61')
        ax.text(.5,-.235,caption,transform=ax.transAxes,ha='center',va='top',fontsize=11,fontweight='bold')
        ax.margins(x=.025)
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(dest/(name+'.'+ext),dpi=600,facecolor='white')
        fig.savefig(dest/(name+'_PREVIEW.png'),dpi=135,facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7.3))
    fig.subplots_adjust(left=.065,right=.987,top=.97,bottom=.12,wspace=.19,hspace=.65)
    for k,method in enumerate(METHODS):
        rr=[r for r in rows if r['method']==method];x=np.arange(len(rr));ax=axes[k,0]
        for shift,metric,color,label in [(-.18,'overall_net',coral,'Overall'),(.18,'hv_net',blue,'HV')]:
            ax.bar(x+shift,[r[metric] for r in rr],.34,color=color,edgecolor='#555c61',linewidth=.5,label=label)
        ax.axhline(0,color='#555c61',linewidth=.7)
        ax.set_ylim(min(0,min(r['overall_net'] for r in rr))*1.3-5,max(r['hv_net'] for r in rr)*1.3)
        style(ax,rr,'Net utility',f"({'ac'[k]}) {method}: realized net utility")
        ax.legend(loc='upper right',framealpha=1,edgecolor='#777',handlelength=1.2)
        ax=axes[k,1];hv=[r['hv_closed'] for r in rr]
        ax.bar(x,hv,.62,color=blue,edgecolor='#555c61',linewidth=.5,label='HV')
        ax.bar(x,[r['ordinary_closed'] for r in rr],.62,bottom=hv,color=teal,edgecolor='#555c61',linewidth=.5,label='Ordinary')
        ax.set_ylim(0,1800)
        for xi,r in zip(x,rr):ax.text(xi,r['total_closed']+25,str(int(r['total_closed'])),ha='center',fontsize=8)
        style(ax,rr,'Closed tasks',f"({'bd'[k]}) {method}: task closure")
        ax.legend(loc='upper right',framealpha=1,edgecolor='#777',ncols=2,handlelength=1.2,columnspacing=.9)
    save(fig,'B20_UTILITY_AND_CLOSURE')
    fig,axes=plt.subplots(2,3,figsize=(14,7.3))
    fig.subplots_adjust(left=.06,right=.988,top=.97,bottom=.12,wspace=.25,hspace=.66)
    for k,method in enumerate(METHODS):
        rr=[r for r in rows if r['method']==method];x=np.arange(len(rr))
        for col,metric,ylabel,cap in [(0,'controllable_energy_kj','Controllable energy (kJ)','Energy'),(1,'downlink_mb','Actual downlink (MB)','Downlink')]:
            ax=axes[k,col];ax.bar(x,[r[metric] for r in rr],.64,color=teal if method=='GROUND' else coral,edgecolor='#555c61',linewidth=.5)
            ax.set_ylim(0,max(r[metric] for r in rr)*1.2)
            style(ax,rr,ylabel,f"({chr(97+3*k+col)}) {method}: {cap.lower()}")
        ax=axes[k,2]
        ax.bar(x-.18,[r['latency_mean_s'] for r in rr],.34,color=teal,edgecolor='#555c61',linewidth=.5,label='Mean')
        ax.bar(x+.18,[r['latency_p95_s'] for r in rr],.34,color=blue,edgecolor='#555c61',linewidth=.5,label='P95')
        ax.plot(x,[r['hv_latency_mean_s'] for r in rr],color=coral,linestyle='--',marker='o',markersize=3,linewidth=1.2,label='HV mean')
        ax.set_ylim(0,max(r['latency_p95_s'] for r in rr)*1.34)
        style(ax,rr,'Closure latency (s)',f"({'cf'[k]}) {method}: closed-task latency")
        ax.legend(loc='upper right',framealpha=1,edgecolor='#777',ncols=3,handlelength=1.,columnspacing=.7,fontsize=8.2)
    save(fig,'B20_RESOURCES_AND_LATENCY')


def htable(rows,columns,digits=3):
    def fmt(v):return '—' if v is None else f'{v:,.{digits}f}' if isinstance(v,float) else html.escape(str(v))
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(l)+'</th>' for _,l in columns)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k,_ in columns)+'</tr>' for r in rows)+'</table></div>'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();root=a.root.resolve();dest=out/'METRICS_DELIVERY'
    os.replace=make_retry_replace(os.replace,out,max_wait_s=30.)
    finished=read(out/'FINISHED.json');assert finished['all_completed']
    rows,groups,provenance,prices=snapshot(root,out);assert len(rows)==20 and len(groups)==60
    rows.sort(key=lambda r:(METHODS.index(r['method']),CONDITIONS.index(r['condition']),r['seed']))
    checks,nfrozen=independent(out,root,rows)
    definitions=[];coverage=[];condition_summary=[]
    for cond in CONDITIONS:
        example=next(r for r in rows if r['condition']==cond)
        cfg=read(Path(example['source_attempt'])/'RESULT/CONFIG_EFFECTIVE.json')['sensitivity']
        definitions.append(dict(condition=cond,label_CN=CN[cond],label_EN=NAMES[cond],
            capacity=cfg['capacity_multiplier'],value_weights=':'.join(map(str,cfg['value_weights'])),
            value_normalization_factor=cfg['value_normalization_factor'],
            outage_start_s=cfg['outage'][0] if cfg['outage'] else None,
            outage_end_s=cfg['outage'][1] if cfg['outage'] else None,
            notification_s=cfg['notification_s']))
        for method in METHODS:
            rr=[r for r in rows if r['condition']==cond and r['method']==method]
            seeds=sorted(r['seed'] for r in rr);missing=sorted({4101,4102,4103}-set(seeds))
            cov=dict(condition=cond,method=method,n=len(rr),seeds=';'.join(map(str,seeds)),
                     missing_seeds=';'.join(map(str,missing)),complete_three_seeds=not missing)
            coverage.append(cov)
            summ=dict(cov)
            for key in ['overall_net','hv_net','overall_gross','fixed_cost','total_closed','hv_closed',
                        'ordinary_closed','controllable_energy_kj','downlink_mb','latency_mean_s','latency_p95_s','hv_latency_mean_s']:
                summ[key+'_mean']=statistics.mean(r[key] for r in rr)
                summ[key+'_sample_SD']=statistics.stdev(r[key] for r in rr) if len(rr)>1 else None
            condition_summary.append(summ)
    csvsave(dest/'condition_definitions.csv',definitions);csvsave(dest/'seed_coverage.csv',coverage)
    csvsave(dest/'condition_summary_available_subset.csv',condition_summary);csvsave(dest/'independent_validation.csv',checks)
    products=[{k:r[k] for k in ['job_id','condition','method','seed','metadata_satisfied','metadata_partial','metadata_total',
        'roi_satisfied','roi_partial','roi_total','roi_context_satisfied','roi_context_partial','roi_context_total',
        'partial_only','effective_delivered','no_valid_product']} for r in rows]
    csvsave(dest/'highest_ground_product_coverage.csv',products)
    latencies=[{k:r[k] for k in ['job_id','condition','method','seed','total_closed','hv_closed','ordinary_closed',
        'latency_mean_s','latency_p95_s','hv_latency_mean_s','hv_latency_p95_s','ordinary_latency_mean_s','ordinary_latency_p95_s']} for r in rows]
    csvsave(dest/'closure_latency_summary.csv',latencies)
    figures(dest,rows)
    start=read(out/'START.json')['epoch'];end=datetime.fromisoformat(finished['updated']).timestamp();elapsed=end-start
    negatives=[r['job_id'] for r in rows if r['overall_net']<0]
    notes=[
        f'20/20组于2026-11-01 02:57:55（香港时间）完成；历时{int(elapsed//3600)}小时{int(elapsed%3600//60)}分钟。11组GROUND、9组OFFLOAD；原审计、敏感性审计、指标对账均通过；实际安全违例和实验失败为0。',
        '本报告只汇总B包20组。每个条件/方法在本包含1–2个种子，不能视作完整三种子结果；A包另22组到齐后可按4101–4103同条件合并，使用样本标准差ddof=1。图中每组柱对应一个真实种子，不画未完成数据或推测值。',
        '总体净效用=实际毛效用−可控能耗扣分E−实际下传扣分B−21600s期末ready积压扣分Q；HV=high+urgent，普通=low+medium。三项固定价格所有方法和条件相同。',
        '每任务仅按最高有效到地产品计一次，区分满足需求及仅部分交付。可控能耗以kJ计，下传以十进制MB计。闭环均值/P95及HV平均时延只统计本方法闭环任务；终点是首次有效满足需求的到地时间，起点是真实到达。P95按(n−1)×0.95线性插值。',
        'A4/A5的价值权重经过归一化，使总价值保持一致；不能把权重表直接解释成全部任务收益被同比放大。B1/B2均在9300s通知中断，区间分别为[9300,9900)和[9300,15600)，未提前泄漏。',
        f'独立复算20组task_accounting和ground_ledger：收益、E/B/Q、HV/普通分组净效用、闭环计数、最高有效产品去重及时延均一致；再次验证{nfrozen}个冻结文件哈希未变。每组1600任务、21600s、20s宏槽及独立4.29s地面处理均保留。',
        '本包20组的HV净效用均为正；总体净效用为负的5组是：'+ '、'.join(negatives)+'。这说明这些轨迹的实际总费用超过总体实际收益；完整跨条件或方法比较仍需合并同种子的A包结果。',
        '阶段窗口净效用尚未扣终态Q，不能替代21600s的最终净效用。当前正文、Excel及CSV均取已结算终态；本次只整理和绘图，没有重跑或修改实验。',
    ]
    if len(negatives)!=5:raise AssertionError('Negative-count text requires update')
    (dest/'00_REPORT_CN.txt').write_text('\n\n'.join(notes)+'\n\n逐种子数据：\n'+'\n'.join(
        f"{r['job_id']}：总体净效用{r['overall_net']:.6f}；HV净效用{r['hv_net']:.6f}；闭环{int(r['total_closed'])}（HV{int(r['hv_closed'])}、普通{int(r['ordinary_closed'])}）；可控能耗{r['controllable_energy_kj']:.3f}kJ；实际下传{r['downlink_mb']:.3f}MB；平均/P95/HV平均闭环时延{r['latency_mean_s']:.3f}/{r['latency_p95_s']:.3f}/{r['hv_latency_mean_s']:.3f}s。" for r in rows)+'\n',encoding='utf-8-sig')
    css='body{font:16px/1.8 "Microsoft YaHei",sans-serif;background:#f4f7fa;color:#263b48;margin:0}main{max-width:1500px;margin:auto;padding:30px;background:white}h1{font-size:27px}h2{font-size:21px;margin:30px 0 12px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:14px}td,th{border:1px solid #d1dce3;padding:7px 10px;text-align:right}th{background:#244b61;color:white}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f2f6f8}.note{padding:12px 16px;background:#eaf3f7;border-left:4px solid #79a8bc}img{width:100%;display:block}a{color:#1a6381}'
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B20最终指标与图表</title><style>'+css+'</style><main><h1>B20 地面敏感性：最终数据与图表</h1>']
    page+=['<p class="note">'+html.escape(t)+'</p>' for t in notes[:2]]
    page+=['<p><a href="B20_METRICS.xlsx">Excel完整工作簿</a> · <a href="per_seed_metrics.csv">逐种子57项数据CSV</a> · <a href="00_REPORT_CN.txt">纯文本报告</a> · <a href="DELIVERY_FILES.csv">文件清单</a></p>',
        '<h2>实验条件与种子范围</h2>',htable(definitions,[('condition','条件'),('label_CN','设置'),('capacity','容量系数'),('value_weights','价值权重'),('value_normalization_factor','价值归一化因子'),('notification_s','通知 s')]),
        htable(coverage,[('condition','条件'),('method','方法'),('n','本包种子数'),('seeds','已完成种子'),('missing_seeds','尚需A包种子')]),
        '<h2>总体/HV净效用与收益费用</h2>',htable(rows,[('condition','条件'),('method','方法'),('seed','种子'),('overall_gross','毛效用'),('energy_cost','能耗E'),('downlink_cost','下传B'),('queue_cost','期末Q'),('fixed_cost','资源总扣分'),('overall_net','总体净效用'),('hv_net','HV净效用'),('ordinary_net','普通净效用')]),
        '<h2>闭环数与闭环率</h2>',htable(rows,[('condition','条件'),('method','方法'),('seed','种子'),('total_closed','总闭环'),('closure_rate_percent','总闭环率 %'),('hv_closed','HV闭环'),('hv_closure_rate_percent','HV闭环率 %'),('ordinary_closed','普通闭环'),('ordinary_closure_rate_percent','普通闭环率 %'),('right_censored_tasks','右删失'),('deadline_failed_tasks','DDL已过未闭环')]),
        '<h2>资源与闭环时延</h2>',htable(rows,[('condition','条件'),('method','方法'),('seed','种子'),('controllable_energy_kj','可控能耗 kJ'),('downlink_mb','实际下传 MB'),('latency_mean_s','平均时延 s'),('latency_p95_s','P95时延 s'),('hv_latency_mean_s','HV平均时延 s')]),
        '<h2>每任务最高有效到地产品</h2>',htable(rows,[('condition','条件'),('method','方法'),('seed','种子'),('metadata_total','Metadata'),('roi_total','ROI'),('roi_context_total','ROI+Ctx'),('partial_only','仅部分交付'),('effective_delivered','有效交付总数'),('no_valid_product','无有效产品')]),
        '<p><a href="highest_ground_product_coverage.csv">各产品分别按满足需求/部分交付细分CSV</a> · <a href="closure_latency_summary.csv">时延完整CSV</a> · <a href="group_metrics.csv">HV/普通分组费用CSV</a></p>',
        '<h2>逐种子数据图</h2><p>每一横轴位置代表一个真实运行；方法分开展示，保留全部负净效用。不同条件种子不齐，图形用于展示本包结果，不替代完整三种子配对分析。</p>',
        '<img src="B20_UTILITY_AND_CLOSURE.png" alt="Per-seed net utility and task closure"><p><a href="B20_UTILITY_AND_CLOSURE.pdf">矢量PDF</a> · <a href="B20_UTILITY_AND_CLOSURE.svg">SVG</a> · <a href="B20_UTILITY_AND_CLOSURE.png">600dpi PNG</a></p>',
        '<img src="B20_RESOURCES_AND_LATENCY.png" alt="Per-seed energy downlink and closed-task latency"><p><a href="B20_RESOURCES_AND_LATENCY.pdf">矢量PDF</a> · <a href="B20_RESOURCES_AND_LATENCY.svg">SVG</a> · <a href="B20_RESOURCES_AND_LATENCY.png">600dpi PNG</a></p>',
        '<h2>统计口径与可确认的观察</h2>']
    page+=['<p>'+html.escape(t)+'</p>' for t in notes[2:]]
    page+=['<p>固定价格：能量 '+str(prices['energy_price_per_Wh'])+' utility/Wh；下传 '+str(prices['downlink_price_per_MB'])+' utility/MB；期末ready积压 '+str(prices['terminal_backlog_price_per_MB'])+' utility/MB。</p>',
        '<p><a href="condition_summary_available_subset.csv">按条件汇总的当前子集均值/样本SD</a>（n=1时SD留空） · <a href="independent_validation.csv">20组独立复算记录</a> · <a href="metric_dictionary.csv">指标字典</a> · <a href="provenance_sha256.csv">原始证据SHA</a></p></main></html>']
    (dest/'00_REPORT_CN.html').write_text('\n'.join(page),encoding='utf-8')
    workbook(dest/'B20_METRICS.xlsx',[('说明',[dict(item=i,text=t) for i,t in enumerate(notes,1)]),('条件定义',definitions),
        ('种子覆盖',coverage),('逐种子完整指标',rows),('HV普通分组',groups),('当前子集均值SD',condition_summary),
        ('最高有效产品',products),('闭环时延',latencies),('独立复算',checks),('指标字典',csvread(dest/'metric_dictionary.csv')),
        ('证据来源',provenance)])
    shutil.copyfile(__file__,dest/Path(__file__).name)
    check=dict(passed=True,at=datetime.now().astimezone().isoformat(),jobs=20,arrivals_each=1600,
        horizon_s=21600,elapsed_s=elapsed,original_audit_checks=sum(c['original_checks'] for c in checks),
        sensitivity_and_metrics_audits_passed=True,independent_raw_accounting_and_ground_events_recomputed=True,
        frozen_files_verified=nfrozen,safety_violations=0,unresolved_or_historical_experiment_failures=0,
        negative_overall_net_jobs=negatives,available_seeds_per_condition_method=coverage,
        simulations_rerun=0,original_code_and_results_unchanged=True)
    write(dest/'FINAL_DELIVERY_VALIDATION.json',check)
    listing=[dict(filename=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='DELIVERY_FILES.csv']
    csvsave(dest/'DELIVERY_FILES.csv',listing)
    archive=out/'B20_METRICS_RESULTS.zip';temp=out/'B20_METRICS_RESULTS.zip.tmp'
    files=sorted(p for p in dest.iterdir() if p.is_file())
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,(Path('B20_METRICS_RESULTS')/f.name).as_posix())
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for f in files:assert hashlib.sha256(z.read('B20_METRICS_RESULTS/'+f.name)).hexdigest()==sha(f)
    os.replace(temp,archive)
    write(out/'FINAL_METRICS_READY.json',dict(at=datetime.now().astimezone().isoformat(),verified_jobs=20,
        report=str(dest/'00_REPORT_CN.html'),excel=str(dest/'B20_METRICS.xlsx'),archive=str(archive),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,enhanced_with_conditions_and_figures=True))
    print(json.dumps(dict(passed=True,jobs=20,original_audit_checks=check['original_audit_checks'],
        frozen_files_verified=nfrozen,elapsed_s=elapsed,delivery_files=len(files),archive_bytes=archive.stat().st_size,
        condition_summary=[{k:r[k] for k in ['condition','method','n','seeds','overall_net_mean','hv_net_mean']} for r in condition_summary]),
        ensure_ascii=False,indent=2))


if __name__=='__main__':main()
