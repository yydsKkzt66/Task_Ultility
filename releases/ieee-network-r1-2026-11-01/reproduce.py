"""Verify/recompute this release without simulation. Python 3.11+, stdlib for tables.

Examples:
  python reproduce.py verify
  python reproduce.py tables --out reproduced_tables
  python reproduce.py figures --out reproduced_figures
  python reproduce.py extract-runtime --name prop_formal --out runtime_prop
"""
from pathlib import Path, PurePosixPath
import argparse, csv, hashlib, json, math, statistics, subprocess, sys, zipfile

HERE=Path(__file__).resolve().parent
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-7),(a,b)
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def verify():
    manifest=rows(HERE/'FILE_SHA256.csv')
    for r in manifest:
        p=HERE/r['file'];assert p.is_relative_to(HERE) and p.is_file(),r['file']
        assert p.stat().st_size==int(r['bytes']) and sha(p)==r['sha256'],r['file']
    count=0
    for bundle in read(HERE/'runtimes/INDEX.json'):
        p=HERE/bundle['archive'];assert sha(p)==bundle['archive_sha256']
        with zipfile.ZipFile(p) as z:
            assert len(z.namelist())==len(set(z.namelist()))==bundle['files']
            for n,h in bundle['manifests'].items():
                assert hashlib.sha256(z.read(n)).hexdigest()==h
                for fn,digest in json.loads(z.read(n))['files'].items():
                    assert hashlib.sha256(z.read(fn)).hexdigest()==digest,(bundle['runtime'],fn)
                    count+=1
    main=rows(HERE/'main/analysis_input_per_seed.csv')
    expected={(m,str(s)) for m in ['GROUND','OFFLOAD','TUT_ADAPTED','PROP'] for s in range(4101,4106)}
    assert len(main)==20 and {(r['method'],r['seed']) for r in main}==expected
    main_index={(r['method'].replace('TUT_ADAPTED','TUT-Adapt'),int(r['seed'])):r for r in main}
    for r in main:
        close(r['overall_net'],float(r['hv_net'])+float(r['ordinary_net']))
        close(r['overall_net'],float(r['overall_gross'])-sum(float(r[x]) for x in ['energy_cost','downlink_cost','queue_cost']))
        close(r['total_closed'],float(r['hv_closed'])+float(r['ordinary_closed']))
        close(r['effective_delivered'],float(r['total_closed'])+float(r['partial_only']))
        close(r['arrivals'],1600)
    sens=rows(HERE/'sensitivity_resilience/tables/ALL_96_PER_SEED_METRICS.csv')
    conditions=['NORMAL','A1','A2','A3','A4','A5','B1','B2'];methods=['GROUND','OFFLOAD','TUT-Adapt','PROP']
    assert len(sens)==96 and {(r['condition'],r['method'],int(r['seed'])) for r in sens}=={(c,m,s) for c in conditions for m in methods for s in range(4101,4104)}
    for r in sens:
        close(r['overall_net'],float(r['overall_gross'])-sum(float(r[x]) for x in ['energy_cost','downlink_cost','queue_cost']))
        close(r['overall_net'],float(r['hv_net'])+float(r['ordinary_net']))
        if r['condition']=='NORMAL':
            ref=main_index[r['method'],int(r['seed'])]
            for k in ['overall_net','hv_net','total_closed','controllable_energy_kj','downlink_mb']:close(r[k],ref[k])
    result=dict(status='PASS',release_files=len(manifest),runtime_manifest_entries_verified=count,main_records=20,sensitivity_records=96,paired_reference_matches=12,simulations_run=0,ablation=read(HERE/'RELEASE_STATUS.json')['ablation'])
    print(json.dumps(result));return result
def fresh(path):
    p=Path(path).resolve()
    if p.exists() and any(p.iterdir()):raise ValueError('Output must be new or empty; original artifacts are immutable.')
    if p==HERE or HERE in p.parents and p.parts[len(HERE.parts)] in {'main','runtimes','figures','sensitivity_resilience','ablation'}:raise ValueError('Cannot overwrite a release directory.')
    p.mkdir(parents=True,exist_ok=True);return p
def tables(out):
    main=rows(HERE/'main/analysis_input_per_seed.csv');aggregate=[]
    for m in ['GROUND','OFFLOAD','TUT_ADAPTED','PROP']:
        group=[r for r in main if r['method']==m]
        for k in main[0]:
            if k in {'method','seed'}:continue
            values=[float(r[k]) for r in group]
            aggregate.append(dict(method=m,metric=k,n=5,mean=statistics.mean(values),sample_sd=statistics.stdev(values),ddof=1))
    with (out/'MAIN_MEAN_SD.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    subprocess.run([sys.executable,str(HERE/'sensitivity_resilience/merge_data.py'),'--output',str(out/'sensitivity')],cwd=out,check=True)
    expected=HERE/'sensitivity_resilience/tables'
    for p in expected.iterdir():
        if p.is_file():assert sha(p)==sha(out/'sensitivity'/p.name),p.name
    result=dict(pass_=True,main_metric_groups=len(aggregate),sensitivity_files_byte_identical=len(list(expected.iterdir())),simulations_run=0)
    save(out/'TABLE_VALIDATION.json',result);print(json.dumps(result))
def extract(name,out):
    bundle=next(b for b in read(HERE/'runtimes/INDEX.json') if b['runtime']==name)
    with zipfile.ZipFile(HERE/bundle['archive']) as z:
        for info in z.infolist():
            rel=PurePosixPath(info.filename)
            if rel.is_absolute() or '..' in rel.parts or ':' in info.filename or '\\' in info.filename:raise ValueError('Unsafe archive member')
            p=out.joinpath(*rel.parts);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(info))
    print(json.dumps(dict(runtime=name,files=bundle['files'],out=str(out),simulation_started=False)))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['verify','tables','figures','extract-runtime']);ap.add_argument('--out');ap.add_argument('--name')
    a=ap.parse_args();verify()
    if a.command=='verify':return
    if not a.out:ap.error('--out is required')
    out=fresh(a.out)
    if a.command=='tables':tables(out)
    elif a.command=='extract-runtime':
        if not a.name:ap.error('--name is required')
        extract(a.name,out)
    else:
        for src,dest in [('main/plot_main_exp.py','main'),('figures/sensitivity/plot_sensitivity.py','sensitivity')]:
            subprocess.run([sys.executable,str(HERE/'plot_environment.py'),str(HERE/src),'--output-dir',str(out/dest)],cwd=out,check=True)
if __name__=='__main__':main()
