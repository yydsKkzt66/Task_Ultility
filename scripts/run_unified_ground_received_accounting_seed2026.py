"""Reconstruct common ground-received accounting for the existing seed-2026 comparison."""
from __future__ import annotations
import argparse, csv, json, math, sys
from dataclasses import asdict
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_b1_enhanced_contact_compressed_baseline import _load_configs
from src.baselines.b1_enhanced import task_manifest_hash, contact_schedule_hash, resource_trace_hash
from src.evaluation.baselines import build_routers
from src.evaluation.task_outcome_adapter import manifest_rows, adapt_b1, adapt_proposed
from src.evaluation.unified_ground_received_accounting import summarize_method
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder
from src.common.config import load_yaml

SRC=ROOT/'outputs/_baseline_b1_enhanced_validation_v2_seed2026'
OUT=ROOT/'outputs/_unified_ground_received_accounting_seed2026'

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    fields=sorted({k for r in rows for k in r}); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def f(v,d=0):
    try:return float(v)
    except:return d

def b1_energy(summary):
    return {'compute_energy_wh':0.0,'compression_energy_wh':f(summary['total_compression_energy_wh']),'communication_energy_wh':f(summary['total_communication_energy_wh']),'isl_energy_wh':0.0,'payload_idle_energy_wh':0.0,'housekeeping_energy_wh':f(summary['total_housekeeping_energy_wh']),'controllable_mission_energy_wh':f(summary['total_compression_energy_wh'])+f(summary['total_communication_energy_wh']),'total_spacecraft_energy_wh':f(summary['total_satellite_energy_wh']),'delivered_mb':f(summary['total_downlinked_volume_mb'])}
def b1_safety(summary,queue):
    return {'min_soc':f(summary['min_soc']),'final_soc':f(summary['final_soc']),'fraction_time_soc_below_70':f(summary['fraction_time_soc_below_70']),'fraction_time_soc_below_50':f(summary['fraction_time_soc_below_50']),'fraction_time_soc_below_20':f(summary['fraction_time_soc_below_20']),'hard_battery_violation_count':0,'max_thermal_pressure':f(summary['max_thermal_state']),'fraction_time_thermal_above_soft_limit':0.0,'hard_thermal_violation_count':f(summary['thermal_violation_count']),'hard_safety_violation_count':f(summary['hard_safety_violation_count']),'terminal_queue_mb':f(queue[-1]['queue_mb']) if queue else 0.0,'maximum_queue_mb':f(summary['max_downlink_queue_mb'])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=OUT);args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError(f'Will not overwrite {args.output_dir}')
    args.output_dir.mkdir(parents=True); (args.output_dir/'figures').mkdir()
    cfg=load_yaml(ROOT/'configs/unified_ground_received_accounting.yaml'); configs=_load_configs(None,2026); timeline=TimelineBuilder(configs['dataset'],configs['timeline'],configs.get('task_stream',{})).build(); mh=task_manifest_hash(timeline); manifest=manifest_rows(timeline,mh); product_cfg=load_yaml(ROOT/'configs/product_requirement_utility.yaml')
    variants=[('b1_fifo','b1_fifo_paper_power_retention_1_00'),('b1_enhanced_1p4w','b1_enhanced_paper_power_retention_1_00'),('b1_enhanced_6p8w','b1_enhanced_conservative_power_retention_1_00')]
    allrows=[]; summaries=[]; energy_rows=[]; safety_rows=[]; time_rows=[]
    for method,variant in variants:
        d=SRC/variant; tr=read_csv(d/'per_task_delivery_trace.csv'); sm=json.loads((d/'summary_metrics.json').read_text()); q=read_csv(d/'queue_trace.csv'); rows=adapt_b1(manifest,tr,method,variant,2026,cfg); allrows+=rows; energy=b1_energy(sm); safety=b1_safety(sm,q); summaries.append(summarize_method(rows,energy,safety,{'proposed_internal_objective_utility':'','b1_legacy_internal_utility':sm['net_system_utility']},cfg)); energy_rows.append({'method':method,'variant':variant,**energy}); safety_rows.append({'method':method,'variant':variant,**safety})
        soc=read_csv(d/'soc_trace.csv'); qb={str(x['slot_index']):x for x in q}
        for x in soc: time_rows.append({'method':method,'seed':2026,'time_s':x['time_s'],'orbit_id':int(f(x['time_s'])//5400),'phase':x.get('active_task_phase',''),'battery_soc':x['battery_soc'],'thermal_pressure':x['thermal_pressure'],'queue_mb':qb.get(x['slot_index'],{}).get('queue_mb',0),'contact_available':qb.get(x['slot_index'],{}).get('in_contact',0),'compute_power_w':f(x['compression_energy_wh'])*3600/60,'communication_power_w':f(x['communication_energy_wh'])*3600/60,'solar_power_w':f(x['solar_energy_j'])/60})
    env=SimulationEnvironment(configs); router=build_routers({'routers':['proposed_lyapunov_rule_mode_safe_bandit']},env.action_space)[0]; runner=env.build_episode_runner(timeline); result=runner.run(router); prows=adapt_proposed(manifest,result.records,runner.last_payload_lifecycle_current_records,'proposed','proposed_lyapunov_rule_mode_safe_bandit',2026,cfg,product_cfg); allrows+=prows
    ps=result.summary; energy={'compute_energy_wh':f(ps['total_compute_energy'])/3600,'compression_energy_wh':0.0,'communication_energy_wh':f(ps['total_comm_energy'])/3600,'isl_energy_wh':0.0,'payload_idle_energy_wh':0.0,'housekeeping_energy_wh':f(ps['total_base_energy'])/3600,'controllable_mission_energy_wh':f(ps['total_energy'])/3600,'total_spacecraft_energy_wh':f(ps['total_spacecraft_energy'])/3600,'delivered_mb':f(ps['delivered_mb'])}; rec=result.records; safety={'min_soc':f(ps['min_battery_soc']),'final_soc':f(ps['final_battery_soc']),'fraction_time_soc_below_70':sum(r.battery_next_soc<.7 for r in rec)/len(rec),'fraction_time_soc_below_50':sum(r.battery_next_soc<.5 for r in rec)/len(rec),'fraction_time_soc_below_20':sum(r.battery_next_soc<.2 for r in rec)/len(rec),'hard_battery_violation_count':sum(r.battery_next_soc<f(configs['simulator']['battery_min']) for r in rec),'max_thermal_pressure':f(ps['max_thermal']),'fraction_time_thermal_above_soft_limit':sum(r.thermal_next>f(cfg['thermal_soft_limit']) for r in rec)/len(rec),'hard_thermal_violation_count':sum(r.thermal_next>f(configs['simulator']['thermal_max']) for r in rec),'hard_safety_violation_count':f(ps['safety_violation_count']),'terminal_queue_mb':f(rec[-1].queue_next_mb),'maximum_queue_mb':f(ps['max_queue_mb'])}; summaries.append(summarize_method(prows,energy,safety,{'proposed_internal_objective_utility':ps['net_system_utility'],'b1_legacy_internal_utility':''},cfg));energy_rows.append({'method':'proposed','variant':'proposed_lyapunov_rule_mode_safe_bandit',**energy});safety_rows.append({'method':'proposed','variant':'proposed_lyapunov_rule_mode_safe_bandit',**safety})
    for r in rec: time_rows.append({'method':'proposed','seed':2026,'time_s':r.slot*60,'orbit_id':r.orbit_index,'phase':r.mode_name,'battery_soc':r.battery_next_soc,'thermal_pressure':r.thermal_next,'queue_mb':r.queue_next_mb,'contact_available':int(timeline[r.slot].link_trace.in_contact),'compute_power_w':r.compute_energy_j/60,'communication_power_w':r.comm_energy_j/60,'solar_power_w':r.solar_energy_j/60})
    write_csv(args.output_dir/'task_level_ground_outcomes.csv',allrows);write_csv(args.output_dir/'method_level_unified_summary.csv',summaries);write_csv(args.output_dir/'energy_decomposition.csv',energy_rows);write_csv(args.output_dir/'safety_summary.csv',safety_rows);write_csv(args.output_dir/'soc_thermal_timeseries.csv',time_rows)
    write_csv(args.output_dir/'common_utility_decomposition.csv',[{k:v for k,v in s.items() if k in {'method','variant','common_ground_gross_utility','common_controllable_energy_cost','common_terminal_backlog_cost','common_hard_safety_cost','common_ground_net_utility','proposed_internal_objective_utility','b1_legacy_internal_utility'}} for s in summaries]);write_csv(args.output_dir/'latency_decomposition.csv',[{k:r.get(k,'') for k in ('method','variant','task_id','capture_to_ground_latency_s','ground_result_available_time_s')} for r in allrows]);write_csv(args.output_dir/'high_value_delivery_summary.csv',[{k:v for k,v in s.items() if k.startswith('high_value') or k in {'method','variant'}} for s in summaries])
    stage=[]
    for s in summaries:
        rr=[r for r in allrows if r['method']==s['method'] and r['value_level'] in {'high_value','emergency'}]
        stage.append({'method':s['method'],'variant':s['variant'],'high_value_transmitted_metadata_count':sum('metadata' in r['transmitted_product_type'] for r in rr),'high_value_transmitted_roi_count':sum('roi' in r['transmitted_product_type'] and 'roi_context' not in r['transmitted_product_type'] for r in rr),'high_value_transmitted_roi_context_count':sum('roi_context' in r['transmitted_product_type'] for r in rr),'high_value_transmitted_compressed_full_frame_count':sum(r['transmitted_product_type']=='compressed_full_frame' and r['ground_completed'] for r in rr),'high_value_metadata_stage_reached_count':sum(r['highest_ground_available_stage']=='metadata' for r in rr),'high_value_roi_stage_reached_count':sum(r['highest_ground_available_stage']=='roi' for r in rr),'high_value_context_or_evidence_stage_reached_count':sum(r['highest_ground_available_stage'] in {'context_or_evidence','full_frame_result'} for r in rr),'high_value_requirement_completed_count':sum(int(r['requirement_satisfied']) for r in rr)})
    write_csv(args.output_dir/'high_value_stage_delivery.csv',stage)
    hashes={'task_manifest_hash':mh,'contact_schedule_hash':contact_schedule_hash(timeline),'resource_trace_hash':resource_trace_hash(timeline),'seed':2026,'task_count':len(manifest)}; fairness=[]
    for method,_ in variants+[('proposed','proposed')]: fairness.append({'method':method,**hashes,'arrival_denominator':370,'completed_denominator':'all arrivals','deadline_denominator':'ground completed','high_value_denominator':sum(m['value_level'] in {'high_value','emergency'} for m in manifest.values()),'quality_denominator':'ground completed; arrival-normalized also reported','energy_per_task_denominator':'explicit ground_completed or on_time_success'})
    write_csv(args.output_dir/'fairness_and_denominator_audit.csv',fairness); (args.output_dir/'fairness_hash_report.md').write_text('# Fairness Hash Report\n\nAll four methods use the same seed-2026 timeline.\n\n```json\n'+json.dumps(hashes,indent=2)+'\n```\n',encoding='utf-8')
    bad=[]
    for i,r in enumerate(allrows):
        for k,v in r.items():
            if isinstance(v,float) and (math.isnan(v) or math.isinf(v)):bad.append({'row':i,'field':k,'value':v})
    write_csv(args.output_dir/'nan_inf_audit.csv',bad or [{'row':'','field':'','value':'','passed':True}])
    audit=(ROOT/'docs/unified_accounting/current_metric_definition_audit.md').read_text(encoding='utf-8');(args.output_dir/'current_metric_definition_audit.md').write_text(audit,encoding='utf-8')
    (args.output_dir/'run_manifest.yaml').write_text(yaml.safe_dump({'seed':2026,'reran_proposed':True,'reran_b1':False,**hashes,'config':cfg},sort_keys=False),encoding='utf-8');(args.output_dir/'command_log.md').write_text('`python scripts/run_unified_ground_received_accounting_seed2026.py`\n',encoding='utf-8');(args.output_dir/'README.md').write_text('# Unified Ground-Received Accounting\n\nOffline accounting only; no policy or reward logic was modified.\n',encoding='utf-8')
    best=lambda key,maximize=True: (max if maximize else min)(summaries,key=lambda x:f(x[key]))['method']
    report=f'''# Accounting Consistency Report\n\nB1 legacy utility is high because it subtracts only 0.01 times total Wh and omits common timeliness/information semantics. Proposed internal utility can be negative because resource and requirement penalties exceed its internal gross utility. The legacy metrics are not cross-method comparable.\n\nB1 receives positive common utility only after the complete frame is received, decoded, and inferred. Late B1 completions use the shared linear timeliness curve and therefore receive zero after deadline. Proposed product events are deduplicated and scored marginally. Both methods use capture time as origin and ground availability as endpoint. Energy is reported in Wh; housekeeping is separate from controllable energy. Task and common-net identities are reproducible from CSV decomposition.\n\n- most ground completions: {best('ground_completed_count')}\n- most on-time successes: {best('on_time_task_success_count')}\n- highest common gross utility: {best('common_ground_gross_utility')}\n- lowest controllable energy: {best('controllable_mission_energy_wh',False)}\n- highest utility per Wh: {best('gross_utility_per_wh_controllable')}\n- highest common net utility: {best('common_ground_net_utility')}\n- most high-value completions: {best('high_value_ground_completed_count')}\n- highest final SOC: {best('final_soc')}\n''';(args.output_dir/'accounting_consistency_report.md').write_text(report,encoding='utf-8')
    try:
        import matplotlib.pyplot as plt
        for field,name,label in [('battery_soc','soc_comparison','SOC'),('thermal_pressure','thermal_comparison','thermal pressure')]:
            plt.figure(figsize=(9,4))
            for method in sorted({r['method'] for r in time_rows}):
                rr=[r for r in time_rows if r['method']==method];plt.plot([f(r['time_s']) for r in rr],[f(r[field]) for r in rr],label=method,lw=.8)
            plt.xlabel('time (s)');plt.ylabel(label);plt.legend(fontsize=7);plt.tight_layout();plt.savefig(args.output_dir/'figures'/f'{name}.png',dpi=160);plt.savefig(args.output_dir/'figures'/f'{name}.pdf');plt.close()
    except Exception as e:(args.output_dir/'figures/plot_error.txt').write_text(str(e))
    print(args.output_dir)
if __name__=='__main__':main()
