"""Independent arithmetic checks and corruption controls; no original writes."""
from pathlib import Path
from unittest.mock import patch
import copy, hashlib, json, math, shutil, subprocess, sys, tempfile
import merge_data as m


def rejected(fn):
    try: fn()
    except AssertionError:return True
    return False


def main():
    here=Path(__file__).resolve().parent;inputs=here/'inputs';tables=here/'tables'
    rows=[m.numberize(r) for r in m.readcsv(tables/'ALL_96_PER_SEED_METRICS.csv')]
    prices=m.load(inputs/'B20/DELIVERY_STATE.json')['prices']
    checks={}
    checks['duplicate_key_rejected']=rejected(lambda:m.unique(rows+[rows[0]]))
    corrupt=dict(rows[0],overall_net=rows[0]['overall_net']+1)
    checks['utility_corruption_rejected']=rejected(lambda:m.validate_row(corrupt,prices))
    corrupt=dict(rows[0],roi_total=rows[0]['roi_total']+1)
    checks['product_double_count_rejected']=rejected(lambda:m.validate_row(corrupt,prices))
    corrupt=dict(rows[0],controllable_energy_kj=rows[0]['controllable_energy_kj']*3.6)
    checks['energy_unit_error_rejected']=rejected(lambda:m.validate_row(corrupt,prices))
    reader=m.readcsv
    for label,filename,field,value in [
        ('wrong_seed_rejected','per_seed_metrics.csv','seed','4101'),
        ('wrong_version_rejected','per_seed_metrics.csv','source_sha256','0'*64),
        ('wrong_config_hash_rejected','provenance_sha256.csv','sha256','0'*64)]:
        def changed(p,filename=filename,field=field,value=value):
            rr=copy.deepcopy(reader(p))
            if Path(p).name==filename:
                if filename=='provenance_sha256.csv':
                    target=next(r for r in rr if r['path'].endswith('CONFIG_EFFECTIVE.json'))
                else:target=rr[0]
                target[field]=value
            return rr
        with patch.object(m,'readcsv',changed):checks[label]=rejected(lambda:m.check_b(inputs))
    agg=reader(tables/'MEAN_SD_32_GROUPS_LONG.csv')
    for r in agg:
        vals=[x[r['metric']] for x in rows if x['condition']==r['condition'] and x['method']==r['method']]
        avg=math.fsum(vals)/3;sd=math.sqrt(math.fsum((x-avg)**2 for x in vals)/2)
        assert math.isclose(float(r['mean']),avg,rel_tol=1e-12,abs_tol=1e-10)
        assert math.isclose(float(r['sample_sd']),sd,rel_tol=1e-12,abs_tol=1e-10)
    checks['all_group_means_and_ddof1_independently_checked']=True
    keyed=m.unique(rows)
    pairs=reader(tables/'MATCHED_84_SEEDS_CHANGES_RETENTION.csv')
    assert len(pairs)==84
    for r in pairs:
        run=keyed[(r['condition'],r['method'],int(r['seed']))]
        normal=keyed[('NORMAL',r['method'],int(r['seed']))]
        for field in ['overall_net','hv_net']:
            m.near(r['delta_'+field],run[field]-normal[field])
            m.near(r[field+'_retention_percent'],100*run[field]/normal[field])
    checks['all_84_pairs_use_same_seed_normal']=True
    grouped=reader(tables/'MATCHED_28_GROUPS_CHANGES_RETENTION.csv')
    assert len(grouped)==28
    for r in grouped:
        pp=[p for p in pairs if p['condition']==r['condition'] and p['method']==r['method']]
        assert len(pp)==3
        for field in ['overall_net','hv_net']:
            m.near(r[field+'_retention_mean_percent'],math.fsum(float(p[field+'_retention_percent']) for p in pp)/3)
    checks['retention_is_mean_of_three_seed_ratios']=True
    with tempfile.TemporaryDirectory(prefix='complete_merge_portable_') as td:
        q=Path(td);shutil.copy2(here/'merge_data.py',q/'merge_data.py');shutil.copytree(inputs,q/'inputs')
        result=subprocess.run([sys.executable,str(q/'merge_data.py'),'--output',str(q/'recomputed')],cwd=q,
                              capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.iterdir() if p.is_file()}
        actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (q/'recomputed').iterdir() if p.is_file()}
        assert actual==expected
    checks['standalone_recompute_byte_identical']=True
    result=dict(pass_=all(checks.values()),checks=checks,aggregate_metrics_verified=len(agg),
                paired_seed_rows_verified=len(pairs),independent_derived_files=len(expected),
                source_edits=0,simulations_run=0)
    m.save(here/'INDEPENDENT_VALIDATION.json',result)
    print(json.dumps(result))
    assert result['pass_']


if __name__=='__main__':main()
