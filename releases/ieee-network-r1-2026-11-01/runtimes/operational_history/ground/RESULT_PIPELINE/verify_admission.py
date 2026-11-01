"""Negative controls for admitting result records, no simulations or source edits."""
from copy import deepcopy
from unittest.mock import patch
import json
import collect_results as c

path=c.PACK/'A_RUNS/4103_GROUND/attempt_01'
job=next(j for j in c.read(c.SOURCE/'RUN_MATRIX.json')['jobs'] if j['seed']==4103 and j['arm']=='GROUND')
checks={}
row,_=c.admit('GROUND',4103,path,job)
checks['accepted_final_value_from_audit']=row['overall_net']==c.read(path/'RESULT/AUDITED_SUMMARY.json')['net']
original_read=c.read

def rejects(name,filename,mutate):
    def alternate(p):
        value=original_read(p)
        if c.Path(p)==path/filename:
            value=deepcopy(value);mutate(value)
        return value
    try:
        with patch.object(c,'read',alternate):c.admit('GROUND',4103,path,job)
    except (AssertionError,KeyError):checks[name]=True
    else:checks[name]=False

rejects('failed_audit_excluded','AUDIT.json',lambda d:d.update({'pass':False}))
rejects('partial_endpoint_excluded','COMPLETED.json',lambda d:d.update(time_s=4200))
rejects('wrong_population_excluded','RESULT/AUDITED_SUMMARY.json',lambda d:d.update(arrivals=268))
rejects('unsettled_Q_excluded','RESULT/AUDITED_SUMMARY.json',lambda d:d.update(terminal_settled=False))
rejects('incorrect_accounting_excluded','RESULT/AUDITED_SUMMARY.json',lambda d:d.update(net=d['net']+1))
rejects('unsafe_result_excluded','RESULT/AUDITED_SUMMARY.json',lambda d:d.update(actual_safety_violations=1))
rejects('changed_input_excluded','IDENTITY.json',lambda d:d['job']['input_hashes'].update({'seed_4103/tasks.csv':'wrong'}))
rejects('unknown_source_excluded','IDENTITY.json',lambda d:d['manifests'].update({'SOURCE_MANIFEST.json':'wrong'}))
rejects('unknown_overlay_excluded','IDENTITY.json',lambda d:d.update(execution_overlay={'version':'unknown','sha256':'wrong'}))
rejects('changed_price_excluded','RESULT/CONFIG_EFFECTIVE.json',lambda d:d['public_prices'].update(downlink_price_per_MB=0))
data,state=c.collect()
checks['failed_GROUND4101_not_in_graph']=not any(r['method']=='GROUND' and r['seed']==4101 and
    c.Path(r['path']).name=='attempt_01' for r in c.read(c.OUT/'SOURCES.json'))
checks['causal_TUT_has_five_seeds']=sum(r['method']=='TUT_ADAPTED' for r in data)==5
incomplete={m for m in c.METHODS if sum(r['method']==m for r in data)<5}
checks['incomplete_baselines_have_no_five_seed_statistics']=all(r['method'] not in incomplete for r in
    list(c.csv.DictReader((c.OUT/'FIVE_SEED_MEAN_SD.csv').open(encoding='utf-8-sig'))))
result=dict(pass_=all(checks.values()),checks=checks,accepted_at_validation=state['accepted'])
c.write(c.HERE/'ADMISSION_VALIDATION.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
assert result['pass_']
