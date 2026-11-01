"""Read-only admission of complete, audited results into the updated main figure."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import statistics

HERE=Path(__file__).resolve().parent
PACK=HERE.parent
REPO=HERE.parents[2]
SOURCE=REPO/'.network_ground_parallel_work'
OUT=PACK/'MAIN_FIGURE_UPDATED'
METHODS=['GROUND','OFFLOAD','TUT_ADAPTED','PROP']
SEEDS=range(4101,4106)
_SHA_CACHE={}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    p=Path(p);key=(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns)
    if key not in _SHA_CACHE:_SHA_CACHE[key]=hashlib.sha256(p.read_bytes()).hexdigest()
    return _SHA_CACHE[key]
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def csvout(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    tmp.replace(p)
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)


def completed(base):
    return [a for a in sorted(base.glob('attempt_*'),reverse=True) if a.is_dir() and
            (a/'COMPLETED.json').is_file() and (a/'AUDIT.json').is_file() and read(a/'AUDIT.json').get('pass')]


def references():
    records={}
    causal=read(REPO/'handoff_out/NETWORK_TUT_CAUSAL5SEED_REPORT_20261030_01/SOURCES.json')
    assert causal['original_TUT_included'] is False
    for source in causal['rows']:
        p=Path(source['original_path']);assert sha(p/'RESULT/AUDITED_SUMMARY.json')==source['summary_sha256']
        records['TUT_ADAPTED',source['seed']]=p
    for seed in SEEDS:
        found=completed(REPO/f'handoff_out/NETWORK_FORMAL6H_20261028_01/MAIN_RUNS/{seed}_PROP')
        assert len(found)==1
        records['PROP',seed]=found[0]
    return records


def admit(method,seed,path,job):
    ident=read(path/'IDENTITY.json');done=read(path/'COMPLETED.json');audit=read(path/'AUDIT.json')
    s=read(path/'RESULT/AUDITED_SUMMARY.json');cfg=read(path/'RESULT/CONFIG_EFFECTIVE.json')
    assert audit['pass'] and done['phase']=='COMPLETED' and done['time_s']==s['stop_s']==21600
    assert done['arrivals']==s['arrivals']==1600 and s['terminal_settled'] and s['seed']==seed
    for name in [f'seed_{seed}/tasks.csv',f'seed_{seed}/calendar.json',f'seed_{seed}/exogenous.json']:
        assert ident['job']['input_hashes'][name]==job['input_hashes'][name]
        assert sha(SOURCE/'network_inputs'/name)==job['input_hashes'][name]
    basecfg=read(SOURCE/f"configs/network_revision/{job['variant']}.json")
    for name in ['public_prices','q_eff','capacity','horizon_s']:
        assert cfg[name]==basecfg[name], (method,seed,name)
    assert s['actual_safety_violations']==0
    overlay=ident.get('execution_overlay')
    if method in ['GROUND','OFFLOAD']:
        assert ident['job']['arm']==method and ident['job']['seed']==seed
        assert ident['job']['config_sha256']==job['config_sha256']
        assert ident['manifests']['SOURCE_MANIFEST.json']==sha(SOURCE/'SOURCE_MANIFEST.json')
        assert ident['manifests']['INPUT_MANIFEST.json']==sha(SOURCE/'INPUT_MANIFEST.json')
        assert cfg['ground_processing']==basecfg['ground_processing']
        expected={'SOURCE_MANIFEST.json':sha(SOURCE/'SOURCE_MANIFEST.json'),'INPUT_MANIFEST.json':sha(SOURCE/'INPUT_MANIFEST.json')}
        if overlay:
            files={'OFFLOAD_EXACT_ARRIVAL_TAIL_V1':PACK/'OFFLOAD4102_CLOCK_RECOVERY/exact_arrival.py',
                   'GROUND_OFFLOAD_EXACT_ARRIVAL_TAIL_V2':HERE/'exact_arrival.py'}
            assert overlay['version'] in files and sha(files[overlay['version']])==overlay['sha256']
            expected[overlay['version']]=overlay['sha256']
        assert ident['manifests']==expected, 'UNKNOWN_EXECUTION_VERSION'
        with (SOURCE/f'network_inputs/seed_{seed}/tasks.csv').open(encoding='utf-8-sig') as f:
            arrival={r['task_id']:float(r['arrival_time_s']) for r in csv.DictReader(f)}
        observed=read(path/'RESULT/arrival_events.json')
        assert len(observed)==len({r['task_id'] for r in observed})==1600
        assert all(r['time_s']==arrival[r['task_id']] for r in observed)
    group=s['groups'];g=group['high_value'];o=group['ordinary'];cost=s['costs']
    assert close(s['net'],s['gross']-sum(cost.values())) and close(s['net'],g['net']+o['net'])
    with (path/'RESULT/task_accounting.csv').open(encoding='utf-8-sig',newline='') as f:tasks=list(csv.DictReader(f))
    assert len(tasks)==len({t['task_id'] for t in tasks})==1600
    true=lambda x:str(x).lower()=='true'
    coverage=Counter()
    for t in tasks:
        product=t['highest_effective_ground_product']
        if product!='none':
            suffix='satisfied' if true(t['satisfies_requirement']) else 'partial'
            assert suffix=='satisfied' or true(t['partial_only'])
            coverage[product+'_'+suffix]+=1
    row=dict(method=method,seed=seed,overall_net=s['net'],hv_net=g['net'],ordinary_net=o['net'],
             gross=s['gross'],hv_gross=g['gross'],E_cost=cost['energy'],B_cost=cost['downlink'],Q_cost=cost['terminal_backlog'],
             hv_closed=g['closed'],ordinary_closed=o['closed'],total_closed=s['closed'],
             hv_closure_rate=g['closed']/448,ordinary_closure_rate=o['closed']/1152,
             effective_delivered=sum(coverage.values()),partial_only=sum(v for k,v in coverage.items() if k.endswith('_partial')),
             controllable_energy_kj=s['ledger']['controlled_j']/1000,downlink_mb=s['ledger']['downlink_mb'])
    for p in ['metadata','roi','roi_context']:
        for suffix in ['satisfied','partial']:row[p+'_'+suffix]=coverage[p+'_'+suffix]
    assert sum(row[p+'_satisfied'] for p in ['metadata','roi','roi_context'])==s['closed']==g['closed']+o['closed']
    assert row['effective_delivered']==s['closed']+row['partial_only']
    arrivals={t['task_id']:float(t['arrival_s']) for t in tasks};hv={t['task_id'] for t in tasks if true(t['HV'])}
    closed={}
    for event in read(path/'RESULT/ground_ledger.json'):
        if event['closes']:closed.setdefault(event['task_id'],event['ground_available_s'])
    assert len(closed)==s['closed']
    latency=sorted(t-arrivals[k] for k,t in closed.items());hl=[t-arrivals[k] for k,t in closed.items() if k in hv]
    x=(len(latency)-1)*.95;lo=int(x);p95=latency[lo]+(latency[min(lo+1,len(latency)-1)]-latency[lo])*(x-lo)
    row.update(closure_mean_s=statistics.mean(latency),closure_p95_s=p95,HV_closure_mean_s=statistics.mean(hl),
               safety_violations=s['actual_safety_violations'],thermal_wait_s=s['thermal_wait_s'])
    provenance=dict(method=method,seed=seed,path=str(path),version=overlay['version'] if overlay else 'ORIGINAL_ACCEPTED_SOURCE',
                    source_manifest_sha256=ident['manifests']['SOURCE_MANIFEST.json'],config_sha256=ident['job']['config_sha256'],
                    input_sha256=job['input_hashes'][f'seed_{seed}/tasks.csv'],execution_overlay=overlay,
                    no_arrival_tail_needed_on_original_trace=not overlay if method in ['GROUND','OFFLOAD'] else None,
                    files={n:sha(path/n) for n in ['IDENTITY.json','AUDIT.json','COMPLETED.json','RESULT/AUDITED_SUMMARY.json',
                        'RESULT/CONFIG_EFFECTIVE.json','RESULT/task_accounting.csv','RESULT/ground_ledger.json']})
    return row,provenance


def collect():
    OUT.mkdir(exist_ok=True)
    matrix=read(SOURCE/'RUN_MATRIX.json')['jobs'];jobs={(j['arm'],j['seed']):j for j in matrix}
    chosen=references();status=[];rejected=[]
    for method in ['GROUND','OFFLOAD']:
        for seed in SEEDS:
            job=jobs[method,seed];folder=f'{seed}_{method}'
            locations=[PACK/'A_RUNS'/folder] if job['package']=='A' else [PACK/'B_RETURN'/folder]
            external=PACK/'B_RETURN'
            if job['package']=='B' and external.exists():
                locations+=list(external.glob('*/'+folder))+list(external.glob('*/*/'+folder))
            options=list(dict.fromkeys(a for base in locations for a in completed(base)))
            assert len(options)<=1, 'DUPLICATE_COMPLETED_RESULTS_ACROSS_RETURNS'
            if options:chosen[method,seed]=options[0]
            status.append(dict(method=method,seed=seed,package=job['package'],status='COMPLETED' if options else 'PENDING'))
    data=[];provenance=[]
    for (method,seed),path in chosen.items():
        try:
            row,prov=admit(method,seed,path,jobs.get((method,seed),jobs['GROUND',seed]))
            data.append(row);provenance.append(prov)
        except Exception as error:
            rejected.append(dict(method=method,seed=seed,path=str(path),error=repr(error)))
            for r in status:
                if (r['method'],r['seed'])==(method,seed):r['status']='REJECTED_VALIDATION'
    data.sort(key=lambda r:(METHODS.index(r['method']),r['seed']))
    provenance.sort(key=lambda r:(METHODS.index(r['method']),r['seed']))
    means=[]
    for method in METHODS:
        rows=[r for r in data if r['method']==method]
        if {r['seed'] for r in rows}!=set(SEEDS):continue
        for metric in rows[0]:
            if metric in ['method','seed']:continue
            vals=[r[metric] for r in rows]
            means.append(dict(method=method,metric=metric,n=5,mean=statistics.mean(vals),sd=statistics.stdev(vals),ddof=1))
    digest=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    state=dict(signature=digest,accepted=len(data),required=20,ready_for_formal_figure=len(data)==20,
               counts=dict(Counter(r['method'] for r in data)),rejected=rejected,
               rule='Only actual21600s/1600-task audited final runs; no failed or partial endpoint substitution',
               version_note='Exact-tail overlay identities retained. Original exact-arrival traces have zero new branch activations; no whole-source byte-equivalence claim.')
    csvout(OUT/'ACCEPTED_PER_SEED.csv',data);csvout(OUT/'FIVE_SEED_MEAN_SD.csv',means)
    csvout(OUT/'BASELINE_COMPLETION.csv',status)
    write(OUT/'SOURCES.json',provenance);write(OUT/'STATE.json',state)
    return data,state


if __name__=='__main__':
    _,state=collect();print(json.dumps(state,ensure_ascii=False,indent=2))
