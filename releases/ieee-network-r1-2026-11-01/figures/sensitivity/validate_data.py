"""Independent full-precision CSV checks. No simulator or project runtime imports."""
from pathlib import Path
import csv, hashlib, json, math, statistics

HERE=Path(__file__).resolve().parent
METHODS=['GROUND','OFFLOAD','TUT-Adapt','PROP']
SEEDS=[4101,4102,4103]
CONDITIONS=['NORMAL','A1','A2','A3','A4','A5','B1','B2']
PANELS={'capacity':(['A1','A2','NORMAL','A3'],[.15,.30,.50,.90]),'value':(['A4','NORMAL','A5'],[0,1,2]),'service':(['NORMAL','B1','B2'],[0,1,2])}
METRICS={'Overall':'overall_net','HV':'hv_net'}
Y_LIMITS=(-80,450)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(p,rr):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def close(a,b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-7),(a,b)
def key(r):return r['condition'],r['method'],int(r['seed'])
def verify_keys(rr):
    expected={(c,m,s) for c in CONDITIONS for m in METHODS for s in SEEDS}
    assert len(rr)==96 and {key(r) for r in rr}==expected,'MISSING_OR_DUPLICATE_RUN'
def check_row(r,prices):
    n=lambda k:float(r[k])
    assert n('arrivals')==1600 and n('hv_arrivals')==448 and n('ordinary_arrivals')==1152 and n('horizon_s')==21600
    for name in ['overall_net','hv_net','ordinary_net','overall_gross','fixed_cost','controllable_energy_kj','downlink_mb']:assert math.isfinite(n(name))
    close(n('overall_net'),n('hv_net')+n('ordinary_net'))
    close(n('overall_gross'),n('hv_gross')+n('ordinary_gross'))
    costs=sum(n(k) for k in ['energy_cost','downlink_cost','queue_cost'])
    close(costs,n('fixed_cost'));close(n('overall_gross')-costs,n('overall_net'))
    close(n('energy_cost'),n('controllable_energy_kj')/3.6*prices['energy_price_per_Wh'])
    close(n('downlink_cost'),n('downlink_mb')*prices['downlink_price_per_MB'])
    assert n('total_closed')==n('hv_closed')+n('ordinary_closed')
    for p in ['metadata','roi','roi_context']:assert n(p+'_satisfied')+n(p+'_partial')==n(p+'_total')
    assert sum(n(p+'_satisfied') for p in ['metadata','roi','roi_context'])==n('total_closed')
    assert sum(n(p+'_partial') for p in ['metadata','roi','roi_context'])==n('partial_only')
    assert sum(n(p+'_total') for p in ['metadata','roi','roi_context'])==n('effective_delivered')
    assert n('total_closed')+n('partial_only')==n('effective_delivered')
    assert n('effective_delivered')+n('no_valid_product')==1600
    assert n('total_closed')+n('right_censored_tasks')+n('deadline_failed_tasks')==1600
    assert n('actual_safety_violations')==0

def validate(plot_csv,per_seed_csv,out):
    data=rows(plot_csv);per=rows(per_seed_csv);verify_keys(per)
    prices=read(HERE/'inputs/PUBLIC_PRICES.json')
    versions=read(HERE/'DATA_VERSION_AUDIT.json')
    assert versions['passed'] and versions['all_points_eligible'] and versions['GROUND_low_capacity_and_normal_same_ground_model']
    vk={key(r):r for r in versions['records']};assert set(vk)=={key(r) for r in per}
    groups={}
    for r in per:
        check_row(r,prices);groups.setdefault((r['condition'],r['method']),[]).append(r)
        assert r['config_sha256']==vk[key(r)]['config_sha256']
        assert r['source_manifest_sha256']==vk[key(r)]['source_manifest_sha256']
        assert r['acceptance']==vk[key(r)]['acceptance']
    assert len(groups)==32 and all(sorted(int(r['seed']) for r in rr)==SEEDS for rr in groups.values())
    expected={(p,c,m,metric) for p,(conditions,_) in PANELS.items() for c in conditions for m in METHODS for metric in METRICS}
    assert len(data)==80 and {(r['panel'],r['condition'],r['method'],r['metric']) for r in data}==expected
    used=[];recomputed=[]
    for r in data:
        rr=groups[r['condition'],r['method']];values=[float(x[METRICS[r['metric']]]) for x in rr]
        mean=statistics.mean(values);sd=statistics.stdev(values)
        close(mean,r['mean']);close(sd,r['sample_sd']);assert int(r['n'])==3 and r['include_in_main_plot']=='True'
        conditions,x=PANELS[r['panel']];assert float(r['x'])==x[conditions.index(r['condition'])]
        assert Y_LIMITS[0]<mean-sd and mean+sd<Y_LIMITS[1]
        # Draw the recomputed full-precision statistics, not rounded report values.
        row=dict(r,mean=mean,sample_sd=sd,ddof=1,seeds='4101;4102;4103',draw_in_main=True,
                 version_family=';'.join(sorted({vk[key(t)]['plot_version'] for t in rr})))
        used.append(row)
        recomputed.append(dict(panel=r['panel'],condition=r['condition'],method=r['method'],metric=r['metric'],mean=mean,sample_sd=sd,n=3,ddof=1,maximum_source_difference=max(abs(mean-float(r['mean'])),abs(sd-float(r['sample_sd'])))))
    summary=rows(HERE/'inputs/MEAN_SD_32_GROUPS_LONG.csv')
    for r in summary:
        vals=[float(x[r['metric']]) for x in groups[r['condition'],r['method']]]
        assert len(vals)==int(r['n'])==3 and int(r['ddof'])==1
        mean=math.fsum(vals)/3;sd=math.sqrt(math.fsum((v-mean)**2 for v in vals)/2)
        assert math.isclose(mean,float(r['mean']),rel_tol=1e-12,abs_tol=1e-10)
        assert math.isclose(sd,float(r['sample_sd']),rel_tol=1e-12,abs_tol=1e-10)
    normal=rows(HERE/'inputs/NORMAL_REFERENCE_4101_4103.csv');assert len(normal)==12
    for r in normal:
        m='TUT-Adapt' if r['method']=='TUT_ADAPTED' else r['method']
        actual=next(t for t in groups['NORMAL',m] if int(t['seed'])==int(r['seed']))
        for metric in ['overall_net','hv_net','total_closed','effective_delivered','controllable_energy_kj','downlink_mb']:close(r[metric],actual[metric])
    write_csv(out/'plot_data_used.csv',used);write_csv(out/'recomputed_plot_statistics.csv',recomputed)
    write_csv(out/'normal_reference_4101_4103.csv',[r for r in per if r['condition']=='NORMAL'])
    minimum=min(r['mean']-r['sample_sd'] for r in used);maximum=max(r['mean']+r['sample_sd'] for r in used)
    report=dict(status='PASS',unique_runs=96,sensitivity_runs=84,normal_runs=12,method_condition_groups=32,seeds=SEEDS,ddof=1,plot_rows_checked=80,main_points=80,excluded_points=0,ground_low_capacity_points_restored=4,summary_mean_sd_pairs_recomputed=len(summary),maximum_plot_summary_difference=max(r['maximum_source_difference'] for r in recomputed),normal_matches_existing_three_seed_main_reference=True,five_seed_means_used=False,all_per_seed_accounting_and_product_identities_pass=True,capacity_not_rescaled=True,mean_minus_sd_min=minimum,mean_plus_sd_max=maximum,shared_y_limits=list(Y_LIMITS),source_plot_sha256=sha(plot_csv),source_per_seed_sha256=sha(per_seed_csv),raw_trajectories_reaudited=False,simulations_run=0)
    save(out/'VALIDATION.json',report)
    snapshot=read(HERE/'inputs/experiment_scale_snapshot.json')
    snapshot.update(plotting_run_count=96,sd_ddof=1,plotting_simulations_started=0,shared_y_limits=list(Y_LIMITS),data_backend='Frozen engineering proxy profiles; no image pixels read by this postprocessor',unavailable_evidence='B raw trajectories and 24 inherited PROP/TUT raw result directories absent; prior acceptance retained',normal_tolerance='GROUND/4102 previous author-approved numerical tolerance retained',source_per_seed_csv_sha256=sha(per_seed_csv))
    save(out/'experiment_scale_snapshot.json',snapshot)
    return used,report
