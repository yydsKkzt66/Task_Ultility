"""Strict read-only import and portable 96-row sensitivity aggregation.

Initial: python merge_data.py --collect --b-root PATH
Portable: python merge_data.py --output fresh_tables
No simulator is run. Original files and older summaries are immutable.
"""
from pathlib import Path, PureWindowsPath
from collections import Counter, defaultdict
import argparse, csv, hashlib, importlib.util, itertools, json, math, shutil, statistics

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
CONDITIONS=['NORMAL','A1','A2','A3','A4','A5','B1','B2']
METHODS=['GROUND','OFFLOAD','TUT-Adapt','PROP']
SEEDS=[4101,4102,4103]
LABELS={'NORMAL':'正常0.50 / 1:3:10:25','A1':'容量0.15','A2':'容量0.30','A3':'容量0.90',
        'A4':'价值1:2:5:10','A5':'价值1:5:20:50','B1':'失去第2窗','B2':'连续失去第2、3窗'}
CORE=['overall_net','hv_net','ordinary_net','overall_gross','hv_gross','ordinary_gross',
      'energy_cost','downlink_cost','queue_cost','fixed_cost','hv_closed','ordinary_closed','total_closed',
      'hv_closure_rate_percent','ordinary_closure_rate_percent','closure_rate_percent',
      'metadata_satisfied','metadata_partial','metadata_total','roi_satisfied','roi_partial','roi_total',
      'roi_context_satisfied','roi_context_partial','roi_context_total','effective_delivered','partial_only','no_valid_product',
      'controllable_energy_kj','downlink_mb','latency_mean_s','latency_p95_s','hv_latency_mean_s',
      'hv_latency_p95_s','ordinary_latency_mean_s','ordinary_latency_p95_s','right_censored_tasks','deadline_failed_tasks',
      'actual_safety_violations']
EXTRA=['thermal_wait_s','min_soc','chip_peak_c','rf_peak_c','active_wall_s','cpu_s','peak_rss_bytes',
       'hv_energy_cost','hv_downlink_cost','hv_queue_cost','hv_fixed_cost',
       'ordinary_energy_cost','ordinary_downlink_cost','ordinary_queue_cost','ordinary_fixed_cost']


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writecsv(p,rows,fields=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def numberize(r):
    result={}
    for k,v in r.items():
        if v=='':result[k]=None;continue
        try:result[k]=float(v)
        except (TypeError,ValueError):result[k]=v
    if 'seed' in result:result['seed']=int(result['seed'])
    return result
def near(x,y):
    assert math.isfinite(float(x)) and math.isfinite(float(y))
    assert math.isclose(float(x),float(y),rel_tol=1e-9,abs_tol=1e-7),(x,y)
def key(r):return r['condition'],r['method'],int(r['seed'])
def unique(rows):
    result={key(r):r for r in rows}
    assert len(result)==len(rows),'DUPLICATE_CONDITION_METHOD_SEED'
    return result
def fill(row,extra):
    for k,v in extra.items():
        if v is None:continue
        if row.get(k) is None:row[k]=v
        elif isinstance(v,(float,int)) and isinstance(row[k],(float,int)):near(row[k],v)


def collect(broot):
    inputs=HERE/'inputs';assert not inputs.exists(),'Use a new output for a new collection.'
    previous=BASE/'RESULT_SUMMARY_20261101_01'
    for f in readcsv(previous/'FILE_SHA256.csv'):
        assert sha(previous/f['file'])==f['sha256'],f['file']
    while not (broot/'per_seed_metrics.csv').exists():
        children=[p for p in broot.iterdir() if p.is_dir()]
        assert len(children)==1,'Ambiguous B20 directory'
        broot=children[0]
    delivery=readcsv(broot/'DELIVERY_FILES.csv')
    for r in delivery:
        assert (broot/r['filename']).stat().st_size==int(r['bytes'])
        assert sha(broot/r['filename'])==r['sha256'],r['filename']
    sources=[]
    def copy(src,name):
        dst=inputs/name;dst.parent.mkdir(parents=True,exist_ok=True)
        before=sha(src);shutil.copy2(src,dst);assert before==sha(src)==sha(dst)
        sources.append(dict(file=name,original_path=str(src.resolve()),bytes=dst.stat().st_size,sha256=before))
    for name in ['ALL_AVAILABLE_PER_SEED_METRICS.csv','A22_GROUP_COSTS.csv','A22_WINDOW_DELIVERY_BACKLOG.csv','A22_WINDOW_RECOVERY.csv']:
        copy(previous/'tables'/name,'prior/'+name)
    for name in ['SENSITIVITY_MATRIX.json','SOURCE_MANIFEST.json','INPUT_MANIFEST.json','MAIN_SOURCE_INDEX.csv',
                 'REUSE_PROP_TUT_48.csv','NORMAL_NEW_GROUND_OFFLOAD_6.csv','SCIENTIFIC_SOURCE_PARITY.json',
                 'FROZEN_IDENTITY.json','experiment_scale_snapshot.json','COLLECTION.json']:
        copy(previous/'inputs'/name,'prior/'+name)
    # Small A evidence stays self-contained; omit raw candidate/physical/checkpoint files.
    for f in (previous/'inputs/A22').rglob('*'):
        if f.is_file():copy(f,'prior/A22/'+f.relative_to(previous/'inputs/A22').as_posix())
    for f in sorted(broot.iterdir()):
        if f.is_file() and f.suffix.lower() in {'.csv','.json','.txt','.py'}:
            copy(f,'B20/'+f.name)
    # Fill only missing normal-reference fields using the already accepted source.
    spec=importlib.util.spec_from_file_location('readonly_metrics',BASE/'runtime/export_run_metrics.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    normal_rows=[];normal_groups=[]
    for idx in readcsv(BASE/'MAIN_SOURCE_INDEX.csv'):
        if int(idx['seed']) not in SEEDS:continue
        p=Path(idx['source'])
        if p.is_dir():
            m,g,t,e=mod.summarize(p)
            identity=load(p/'IDENTITY.json')
            m.update(source_manifest_sha256=identity['manifests']['SOURCE_MANIFEST.json'],
                     input_manifest_sha256=identity['manifests']['INPUT_MANIFEST.json'],
                     config_sha256=sha(p/'RESULT/CONFIG_EFFECTIVE.json'),
                     task_input_sha256=identity['job'].get('task_sha256'))
            normal_rows.append(m)
            normal_groups.extend(dict(method=m['method'],seed=m['seed'],condition='NORMAL',**x) for x in g)
            save(inputs/f'normal/{m["method"]}_{m["seed"]}_RECONCILIATION.json',e)
            for f in ['AUDIT.json','IDENTITY.json','STATUS.json','RESULT/CONFIG_EFFECTIVE.json']:
                copy(p/f,f'normal/{m["method"]}_{m["seed"]}/'+f)
        else:
            assert idx['method']=='OFFLOAD' and idx['seed']=='4103'
            copy(p,'normal/RETURNED_MAIN_B_PER_SEED.csv')
            copy(p.parent/'group_costs_and_closure.csv','normal/RETURNED_MAIN_B_GROUPS.csv')
    writecsv(inputs/'normal/LOCAL5_ENRICHMENT.csv',normal_rows)
    writecsv(inputs/'normal/LOCAL5_GROUPS.csv',normal_groups)
    save(inputs/'COLLECTION.json',dict(B_received_root=str(broot),B_delivery_files_checked=len(delivery),
        source_files=sources,simulations_run=0,original_source_modifications=0,
        raw_B_audits_and_trajectories_in_package=False,previous_76_table_sha256=sha(previous/'tables/ALL_AVAILABLE_PER_SEED_METRICS.csv')))
    # Cover original copies and independently derived enrichment inputs.
    files=[p for p in inputs.rglob('*') if p.is_file()]
    writecsv(inputs/'FILE_SHA256.csv',[dict(file=p.relative_to(inputs).as_posix(),sha256=sha(p)) for p in sorted(files)])


def check_b(inputs):
    jobs={j['job_id']:j for j in load(inputs/'prior/SENSITIVITY_MATRIX.json')['jobs'] if j['package']=='B'}
    plan=load(inputs/'B20/LAUNCH_PLAN.json')
    assert plan['package']=='B' and not plan['scientific_code_modified']
    assert plan['horizon_s']==21600 and plan['tasks_each']==1600 and plan['macro_slot_s']==20
    assert {j['job_id'] for j in plan['jobs']}==set(jobs)
    for j in plan['jobs']:
        assert {k:v for k,v in j.items() if k not in ['queue_priority','estimated_cpu_s']}=={
            k:v for k,v in jobs[j['job_id']].items() if k not in ['queue_priority','estimated_cpu_s']}
    source_sha=sha(inputs/'prior/SOURCE_MANIFEST.json');input_sha=sha(inputs/'prior/INPUT_MANIFEST.json')
    assert plan['source_sha256']==source_sha and plan['input_sha256']==input_sha
    final=load(inputs/'B20/FINAL_DELIVERY_VALIDATION.json')
    assert final['passed'] and final['jobs']==20 and final['horizon_s']==21600 and final['arrivals_each']==1600
    assert final['original_audit_checks']==400 and final['safety_violations']==0
    assert final['sensitivity_and_metrics_audits_passed'] and final['independent_raw_accounting_and_ground_events_recomputed']
    validation={r['job_id']:r for r in readcsv(inputs/'B20/independent_validation.csv')}
    provenance=readcsv(inputs/'B20/provenance_sha256.csv')
    rows=[numberize(r) for r in readcsv(inputs/'B20/per_seed_metrics.csv')]
    assert len(rows)==20 and set(r['job_id'] for r in rows)==set(jobs)
    assertions=[]
    for r in rows:
        job=jobs[r['job_id']];assert key(r)==(job['condition'],job['arm'],job['seed'])
        assert r['audit_pass']=='True' and r['source_sha256']==source_sha and r['input_sha256']==input_sha
        v=validation[r['job_id']]
        assert all(v[k]=='True' for k in ['original_audit_pass','sensitivity_audit_pass','metrics_reconciled',
                       'task_costs_recomputed','highest_product_deduplicated','closed_latency_mean_and_p95_recomputed'])
        assert int(v['original_checks'])==20 and int(v['actual_safety_violations'])==0
        assert r['source_attempt']==v['source_attempt']
        pp={PureWindowsPath(p['path']).name:p for p in provenance if p['job_id']==r['job_id']}
        assert len(pp)==7 and pp['CONFIG_EFFECTIVE.json']['sha256']==job['config_sha256']
        assert all(str(PureWindowsPath(p['path'])).startswith(r['source_attempt']+'\\') for p in pp.values())
        r.update(source_kind='B_RETURN_VERIFIED_SUMMARY',source_path=r.pop('source_attempt'),
                 source_manifest_sha256=r.pop('source_sha256'),input_manifest_sha256=r.pop('input_sha256'),
                 config_sha256=job['config_sha256'],task_input_sha256=job['task_sha256'],
                 acceptance='RETURNED_AUDITS_PASS_AND_LOCAL_TABLE_RECONCILIATION',
                 original_audit_sha256=pp['AUDIT.json']['sha256'],original_completion_sha256=pp['VERIFIED_COMPLETE.json']['sha256'])
        assertions.append(dict(job_id=r['job_id'],source_manifest_match=True,input_manifest_match=True,config_match=True,
              task_input_match_in_launch_plan=True,original_audit_checks_reported=20,local_numeric_reconciliation=True,
              raw_B_audit_reperformed_here=False,source_path=r['source_path'],config_sha256=r['config_sha256'],
              original_audit_sha256=r['original_audit_sha256'],original_completion_sha256=r['original_completion_sha256']))
    unique(rows)
    return rows,assertions


def validate_row(r,prices):
    assert r['arrivals']==1600 and r['hv_arrivals']==448 and r['ordinary_arrivals']==1152
    for k in CORE:assert isinstance(r.get(k),(int,float)) and math.isfinite(r[k]),(key(r),k,r.get(k))
    near(r['overall_gross']-r['energy_cost']-r['downlink_cost']-r['queue_cost'],r['overall_net'])
    near(r['fixed_cost'],r['energy_cost']+r['downlink_cost']+r['queue_cost'])
    near(r['overall_net'],r['hv_net']+r['ordinary_net']);near(r['overall_gross'],r['hv_gross']+r['ordinary_gross'])
    near(r['energy_cost'],r['controllable_energy_kj']/3.6*prices['energy_price_per_Wh'])
    near(r['downlink_cost'],r['downlink_mb']*prices['downlink_price_per_MB'])
    near(r['total_closed'],r['hv_closed']+r['ordinary_closed'])
    near(r['hv_closure_rate_percent'],100*r['hv_closed']/448)
    near(r['ordinary_closure_rate_percent'],100*r['ordinary_closed']/1152)
    near(r['closure_rate_percent'],100*r['total_closed']/1600)
    near(r['arrivals'],r['total_closed']+r['right_censored_tasks']+r['deadline_failed_tasks'])
    for suffix,total in [('total','effective_delivered'),('satisfied','total_closed'),('partial','partial_only')]:
        near(sum(r[p+'_'+suffix] for p in ['metadata','roi','roi_context']),r[total])
    near(r['arrivals'],r['effective_delivered']+r['no_valid_product'])
    for p in ['metadata','roi','roi_context']:near(r[p+'_total'],r[p+'_satisfied']+r[p+'_partial'])
    assert r['latency_mean_s']>=0 and r['latency_p95_s']>=0 and r['hv_latency_mean_s']>=0


def build(inputs,output):
    assert not output.exists() or not any(output.iterdir()),'Use a fresh tables output.'
    for r in readcsv(inputs/'FILE_SHA256.csv'):assert sha(inputs/r['file'])==r['sha256'],r['file']
    prior=[numberize(r) for r in readcsv(inputs/'prior/ALL_AVAILABLE_PER_SEED_METRICS.csv')]
    original={key(r):dict(r) for r in prior};assert len(original)==76
    b,import_checks=check_b(inputs)
    assert not set(unique(prior))&set(unique(b))
    rows=prior+b;bykey=unique(rows)
    assert set(bykey)==set(itertools.product(CONDITIONS,METHODS,SEEDS))
    groups=[numberize(r) for r in readcsv(inputs/'prior/A22_GROUP_COSTS.csv')]
    groups += [numberize(r) for r in readcsv(inputs/'B20/group_metrics.csv')]
    groups += [numberize(r) for r in readcsv(inputs/'normal/LOCAL5_GROUPS.csv')]
    for r in readcsv(inputs/'normal/LOCAL5_ENRICHMENT.csv'):
        r=numberize(r);fill(bykey[('NORMAL',r['method'],r['seed'])],r)
    # Returned NORMAL OFFLOAD4103: supplement missing fields only, preserving previous numbers.
    raw=next(numberize(r) for r in readcsv(inputs/'normal/RETURNED_MAIN_B_PER_SEED.csv') if r['method']=='OFFLOAD' and r['seed']=='4103')
    supplemental={'hv_gross':raw['HV_gross_utility'],'ordinary_gross':raw['gross_utility']-raw['HV_gross_utility']}
    mapping={'HV_closure_latency_p95_s':'hv_latency_p95_s','ordinary_closure_latency_mean_s':'ordinary_latency_mean_s',
             'ordinary_closure_latency_p95_s':'ordinary_latency_p95_s','right_censored_tasks':'right_censored_tasks',
             'deadline_failed_tasks':'deadline_failed_tasks','safety_violations':'actual_safety_violations',
             'chip_peak_C':'chip_peak_c','RF_peak_C':'rf_peak_c','SOC_min':'min_soc','thermal_wait_s':'thermal_wait_s',
             'active_wall_s':'active_wall_s','CPU_s':'cpu_s','source_manifest_sha256':'source_manifest_sha256',
             'input_manifest_sha256':'input_manifest_sha256','config_sha256':'config_sha256'}
    for old,new in mapping.items():supplemental[new]=raw[old]
    supplemental['peak_rss_bytes']=raw['peak_RSS_MB']*1e6
    fill(bykey[('NORMAL','OFFLOAD',4103)],supplemental)
    for r in readcsv(inputs/'normal/RETURNED_MAIN_B_GROUPS.csv'):
        if r['method']!='OFFLOAD' or r['seed']!='4103':continue
        r=numberize(r)
        groups.append(dict(condition='NORMAL',method='OFFLOAD',seed=4103,group=r['group'],
            arrivals=r['tasks'],closed=r['closed'],gross=r['gross_utility'],net=r['net_utility'],
            energy_cost=r['energy_penalty'],downlink_cost=r['downlink_penalty'],queue_cost=r['terminal_backlog_penalty'],
            latency_mean_s=r['closure_latency_mean_s'],latency_p95_s=r['closure_latency_p95_s']))
    assert len(groups)==144
    for g in groups:
        r=bykey[key(g)];prefix={'all':'','HV':'hv_','ordinary':'ordinary_'}[g['group']]
        near(g['gross']-g['energy_cost']-g['downlink_cost']-g['queue_cost'],g['net'])
        near(g['net'],r['overall_net'] if not prefix else r[prefix+'net'])
        if prefix:
            fill(r,{prefix+k:g[k] for k in ['energy_cost','downlink_cost','queue_cost']})
    prices=load(inputs/'B20/DELIVERY_STATE.json')['prices']
    for config in (inputs/'prior/A22').glob('*/RESULT/CONFIG_EFFECTIVE.json'):
        assert load(config)['public_prices']==prices
    for r in rows:
        r['horizon_s']=21600
        r['hv_fixed_cost']=r['hv_gross']-r['hv_net'];r['ordinary_fixed_cost']=r['ordinary_gross']-r['ordinary_net']
        validate_row(r,prices)
        if key(r) in original:
            for k,v in original[key(r)].items():
                if v is not None:assert r[k]==v,('PRIOR_VALUE_CHANGED',key(r),k,v,r[k])
    rows.sort(key=lambda r:(CONDITIONS.index(r['condition']),METHODS.index(r['method']),r['seed']))
    output.mkdir(parents=True,exist_ok=True)
    writecsv(output/'ALL_96_PER_SEED_METRICS.csv',rows)
    writecsv(output/'GROUND_OFFLOAD_GROUP_COSTS_144.csv',groups)
    writecsv(output/'B20_IMPORT_VERIFICATION.csv',import_checks)
    agg=[];wide=[];coverage=[]
    grouped=defaultdict(list)
    for r in rows:grouped[(r['condition'],r['method'])].append(r)
    for (c,m),rr in grouped.items():
        assert {r['seed'] for r in rr}==set(SEEDS) and len(rr)==3
        record=dict(condition=c,condition_CN=LABELS[c],method=m,n=3,ddof=1)
        for metric in CORE+EXTRA:
            values=[r.get(metric) for r in rr];complete=all(isinstance(x,(int,float)) for x in values)
            coverage.append(dict(condition=c,method=m,metric=metric,available_seeds=sum(x is not None for x in values),
                complete_three_seeds=complete,missing_seeds=';'.join(str(r['seed']) for r in rr if r.get(metric) is None)))
            if complete:
                mean=statistics.mean(values);sd=statistics.stdev(values)
                agg.append(dict(condition=c,method=m,metric=metric,n=3,mean=mean,sample_sd=sd,ddof=1))
                record[metric+'_mean']=mean;record[metric+'_sample_sd']=sd
        wide.append(record)
    writecsv(output/'MEAN_SD_32_GROUPS_LONG.csv',agg)
    writecsv(output/'MEAN_SD_32_GROUPS_WIDE.csv',wide)
    writecsv(output/'FIELD_COMPLETENESS.csv',coverage)
    paired=[]
    for r in rows:
        if r['condition']=='NORMAL':continue
        n=bykey[('NORMAL',r['method'],r['seed'])]
        p=dict(condition=r['condition'],method=r['method'],seed=r['seed'])
        for metric in CORE:
            p['normal_'+metric]=n[metric];p['delta_'+metric]=r[metric]-n[metric]
        for metric in ['overall_net','hv_net']:
            p[metric+'_retention_percent']=100*r[metric]/n[metric] if n[metric]>1e-6 else None
        p['overall_normal_net_to_gross_percent']=100*n['overall_net']/n['overall_gross']
        p['ratio_note']='GROUND_SMALL_NET_BASELINE_USE_ABSOLUTE_DELTA_TOO' if r['method']=='GROUND' else 'paired_seed_ratio'
        paired.append(p)
    paired_groups=[]
    for c in CONDITIONS[1:]:
        for m in METHODS:
            pp=[p for p in paired if p['condition']==c and p['method']==m]
            assert {p['seed'] for p in pp}==set(SEEDS)
            q=dict(condition=c,method=m,n=3,ddof=1,aggregation='arithmetic_mean_of_paired_seed_ratios')
            for metric in ['overall_net','hv_net']:
                delta=[p['delta_'+metric] for p in pp];ratio=[p[metric+'_retention_percent'] for p in pp]
                q[metric+'_delta_mean']=statistics.mean(delta);q[metric+'_delta_sample_sd']=statistics.stdev(delta)
                q[metric+'_retention_mean_percent']=statistics.mean(ratio) if all(x is not None for x in ratio) else None
                q[metric+'_retention_sample_sd_percent']=statistics.stdev(ratio) if all(x is not None for x in ratio) else None
            paired_groups.append(q)
    writecsv(output/'MATCHED_84_SEEDS_CHANGES_RETENTION.csv',paired)
    writecsv(output/'MATCHED_28_GROUPS_CHANGES_RETENTION.csv',paired_groups)
    for filename,conditions in [('CAPACITY',['A1','A2','NORMAL','A3']),('VALUE',['A4','NORMAL','A5']),('OUTAGE',['NORMAL','B1','B2'])]:
        figure=[]
        for c in conditions:
            for m in METHODS:
                r=next(x for x in wide if x['condition']==c and x['method']==m)
                figure.append({k:r[k] for k in ['condition','condition_CN','method','n','ddof','overall_net_mean',
                    'overall_net_sample_sd','hv_net_mean','hv_net_sample_sd']})
        writecsv(output/('FIGURE_'+filename+'.csv'),figure)
    writecsv(output/'SOURCE_INDEX_96.csv',[{k:r.get(k) for k in ['condition','method','seed','source_kind','source_path',
        'source_manifest_sha256','input_manifest_sha256','config_sha256','task_input_sha256','acceptance','version_family',
        'original_audit_sha256','original_completion_sha256']} for r in rows])
    checks=dict(pass_=True,rows=96,groups=32,seeds=SEEDS,each_group_n=3,ddof=1,
        A_new=22,B_new=20,new_ground_offload_total=42,normal_ground_offload=6,PROP_TUT_reused=48,
        sensitivity_rows=84,normal_rows=12,duplicate_keys=0,missing_keys=0,prior76_existing_values_changed=0,
        B_job_config_source_input_matches=20,B_delivery_file_SHA_pass=33,
        B_original_raw_audits_reperformed_locally=False,B_validation_source='Returned independent_validation and final validation; hashes retained',
        raw_PROP_TUT_dirs_missing_locally=24,core_metric_cells_missing=0,
        actual_safety_violations=sum(r['actual_safety_violations'] for r in rows),
        original_results_modified=False,simulations_run=0,baseline_exception='NORMAL GROUND4102 author-approved numerical arrival tolerance remains explicit')
    save(output/'MERGE_VALIDATION.json',checks)
    scale=load(inputs/'prior/experiment_scale_snapshot.json')
    scale.update(methods=METHODS,new_runs=0,merged_rows=96,condition_groups=8,method_condition_groups=32,
                 total_formal_new_runs_represented=84,normal_reference_rows=12,actual_metrics='ALL_96_PER_SEED_METRICS.csv',
                 energy_shares='Not recomputed; controllable task energy remains separate from background')
    save(output/'experiment_scale_snapshot.json',scale)
    report(output,wide,paired_groups,checks)
    print(json.dumps(checks,ensure_ascii=False))


def report(output,wide,paired,checks):
    lines=['# 完整三种子敏感性数据汇总','',
        '**已合并96条唯一逐种子记录，32个条件×方法组全部具备4101、4102、4103；无重复、无缺失。**',
        '构成：A包22 + B包20 + 新版GROUND/OFFLOAD正常参照6 + 冻结PROP及共同因果TUT-Adapt参考48。即84条扰动实验和12条正常参照，不使用五种子正常均值。','',
        '## 来源核验','',
        'B包33个交付文件SHA全部通过；20个job与原B清单完全对应，源码/输入清单SHA与A包相同，20个实际配置的回传SHA与本机冻结配置逐项一致。A/B键互不交叠。',
        'B包回传的每组20项原审计、敏感性审计及指标对账均为PASS；本次独立核对逐种子及分组收益、E/B/Q、能量/字节费用换算、闭环分母、产品去重和右删失恒等式。B机原始审计文件/大轨迹没有随轻量包回传，不把回传证明说成本机重新执行了物理审计。',
        '原76条已有数值逐字段保持不变；新版正常参照原表缺少的HV毛效用、删失和补充时延等，分别从原接受来源补齐并对照已有指标。NORMAL GROUND4102保留作者已批准到达浮点容差接受状态，没有把其原始审计改写为严格PASS。',
        'PROP/TUT使用原冻结48条及各自原版本标签，其中24条原始目录不在本机，本次沿用既有接受证据，不声称重新审核缺失原始轨迹。GROUND/OFFLOAD均使用GROUND_PARALLEL_V1独立4.29s地面处理；没有加入旧地面串行/缓存配置或旧TUT结果。','',
        '## 统计口径','',
        '全部32组从三个种子原值重新计算均值±样本标准差（ddof=1），没有平均A/B局部均值。所有核心指标在96条记录中均有值；未记录的附加字段保持空值，详见FIELD_COMPLETENESS.csv。',
        '时延只覆盖各方法实际闭环任务；每种子先计算平均、P95及HV平均，再跨种子求均值/标准差。产品每任务只计最高有效到地等级，分满足需求/部分交付。未到DDL且未闭环任务保持右删失。',
        '保留率=扰动净效用/同方法同种子正常净效用×100%；组保留率是三个种子比值的算术均值±样本标准差。另存逐种子基准和绝对变化。保留率可能为负或超过100%，不是成功概率；非正或数值零基准不计算。',
        'GROUND正常总体净效用约8.74–30.40，接近毛效用与费用相抵后的余量，大百分比容易受小基准放大；比较时应先看净效用绝对值和绝对变化，不按该比值宣称优势。','',
        '容量倍率按名义速率绝对取0.15/0.30/0.50/0.90；A4/A5相对价值1:2:5:10、1:5:20:50，均沿用冻结总先验价值归一化。B1=[9300,9900)，B2=[9300,15600)，通知在9300s发生。未修改参数或重新运行。','',
        '## 总体/HV净效用（均值±样本标准差）','',
        '|条件|方法|总体净效用|HV净效用|种子数|','|---|---|---:|---:|---:|']
    for r in wide:
        lines.append(f'|{LABELS[r["condition"]]}|{r["method"]}|{r["overall_net_mean"]:.6f} ± {r["overall_net_sample_sd"]:.6f}|{r["hv_net_mean"]:.6f} ± {r["hv_net_sample_sd"]:.6f}|3|')
    lines += ['', '## 文件入口','',
        '- ALL_96_PER_SEED_METRICS.csv：完整精度逐种子核心及补充指标。',
        '- MEAN_SD_32_GROUPS_WIDE.csv / LONG.csv：全指标32组均值和样本标准差。',
        '- MATCHED_84_SEEDS_CHANGES_RETENTION.csv / MATCHED_28_GROUPS_CHANGES_RETENTION.csv：配对正常差值和保留率。',
        '- FIGURE_CAPACITY.csv、FIGURE_VALUE.csv、FIGURE_OUTAGE.csv：三类子图的总体/HV净效用及误差。正常点在不同图重复引用，但主数据只保存一次。',
        '- SOURCE_INDEX_96.csv、B20_IMPORT_VERIFICATION.csv、MERGE_VALIDATION.json：来源与核验。',
        '- GROUND_OFFLOAD_GROUP_COSTS_144.csv：两地面方法48组的总体/HV/普通费用及闭环明细。',
        '- FIELD_COMPLETENESS.csv：附加字段缺项；空值不代表0。','',
        '现有96条来源均报告实际安全违规0。原代码、配置、结果、旧汇总均保留。本包提供轻量证据和可移植复算脚本，不打包完整日志/轨迹。B中断恢复贡献及逐窗积压未随本次B20指标包返回，不能把A包已有附表冒充全42组窗口明细。']
    (output/'00_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collect',action='store_true')
    p.add_argument('--b-root',type=Path);p.add_argument('--output',type=Path,default=HERE/'tables')
    a=p.parse_args()
    if a.collect:
        assert a.b_root is not None;collect(a.b_root.resolve())
    build(HERE/'inputs',a.output.resolve())
