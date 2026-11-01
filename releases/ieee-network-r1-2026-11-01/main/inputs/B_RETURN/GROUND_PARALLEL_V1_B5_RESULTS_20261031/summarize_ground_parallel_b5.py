"""Read audited B5 results; write compact tables/figures, never run a simulation.

Reproduce with Python 3.11 and matplotlib. Inputs are immutable; --out must be
a separate directory. Baseline metrics are imported references, not re-audited
original runs. Per-method summaries describe the available n=2 / n=3 subsets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import statistics as st
import zipfile
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

PACKAGE = Path(r'E:\Utility-Driven星上智能\NETWORK_GROUND_PARALLEL_V1_B(1)')
RUNS = Path(r'C:\Users\78689\NetworkExperimentDelivery\GROUND_PARALLEL_V1_B5_20261031')
DEFAULT_OUT = Path(r'C:\Users\78689\NetworkExperimentDelivery\GROUND_PARALLEL_V1_B5_RESULTS_20261031')
METHODS = ('GROUND', 'OFFLOAD')
LEVEL = {'none': 0, 'metadata': 1, 'roi': 2, 'roi_context': 3}
PRODUCTS = (('metadata', 'Metadata'), ('roi', 'ROI'), ('roi_context', 'ROI_Ctx'))
GROUND_RULE = dict(policy='independent_fixed', service_s=4.29,
                   receive_capacity='nonbinding', version='GROUND_PARALLEL_V1')
BASE_MAP = {'gross_utility': 'gross', 'overall_net_utility': 'net',
    'HV_net_utility': 'HV_net', 'HV_gross_utility': 'HV_gross',
    'energy_penalty': 'E_cost', 'downlink_penalty': 'B_cost',
    'terminal_backlog_penalty': 'Q_cost', 'closed_HV': 'HV_closed',
    'closed_ordinary': 'ordinary_closed', 'actual_downlink_MB': 'downlink_MB',
    'terminal_backlog_MB': 'Q_end_MB', 'chip_peak_C': 'chip_peak_C',
    'RF_peak_C': 'RF_peak_C', 'SOC_min': 'SOC_min',
    'thermal_wait_s': 'thermal_wait_s', 'right_censored_tasks': 'right_censored',
    'deadline_failed_tasks': 'DDL_failed'}
DESCRIPTIONS = {
    'arrivals': ('到达任务数', 'tasks', '每组1600个任务'),
    'overall_net_utility': ('总体净效用', 'utility', '毛效用减E/B/Q三项费用'),
    'HV_net_utility': ('高价值净效用', 'utility', 'high和urgent任务收益减其实际E/B/Q费用'),
    'ordinary_net_utility': ('普通任务净效用', 'utility', 'low和medium任务收益减其实际E/B/Q费用'),
    'gross_utility': ('总体毛效用', 'utility', '观察期内有效到地信息收益之和'),
    'HV_gross_utility': ('高价值毛效用', 'utility', 'high和urgent任务有效到地收益之和'),
    'resource_total_penalty': ('资源总扣分', 'utility', 'E+B+Q'),
    'energy_penalty': ('能耗扣分E', 'utility', '可控能耗J/3600乘固定每Wh价格'),
    'downlink_penalty': ('下传扣分B', 'utility', '实际十进制MB乘固定每MB价格'),
    'terminal_backlog_penalty': ('期末积压扣分Q', 'utility', '21600s时ready积压MB乘固定价格'),
    'closed_all': ('总体闭环数', 'tasks', '首个有效满足最终需求的到地产品；部分交付不算闭环'),
    'closed_HV': ('高价值闭环数', 'tasks', 'high和urgent闭环任务数；分母448'),
    'closed_ordinary': ('普通闭环数', 'tasks', 'low和medium闭环任务数；分母1152'),
    'closure_rate_all_percent': ('总体闭环率', '%', 'closed_all/1600*100'),
    'closure_rate_HV_percent': ('高价值闭环率', '%', 'closed_HV/448*100'),
    'closure_rate_ordinary_percent': ('普通闭环率', '%', 'closed_ordinary/1152*100'),
    'controlled_energy_kJ': ('可控能耗', 'kJ', '推理+编码+实际RF发送；不含背景'),
    'background_energy_kJ': ('背景能耗', 'kJ', '单独列出，不重复纳入可控能耗费用'),
    'actual_downlink_MB': ('实际下传量', 'MB', '十进制MB=8e6bit，按实际传输记账'),
    'terminal_backlog_MB': ('期末ready积压量', 'MB', '已ready但21600s时尚未发完的数据'),
    'closure_latency_mean_s': ('闭环时延均值', 's', '真实到达到首个有效满足需求的产品到地；只统计闭环任务'),
    'closure_latency_p95_s': ('闭环时延P95', 's', '闭环样本排序后按(n-1)*0.95线性插值'),
    'HV_closure_latency_mean_s': ('高价值闭环时延均值', 's', '只统计实际闭环的high和urgent任务'),
    'HV_closure_latency_p95_s': ('高价值闭环时延P95', 's', '只统计实际闭环的high和urgent任务'),
    'ordinary_closure_latency_mean_s': ('普通闭环时延均值', 's', '只统计实际闭环的普通任务'),
    'ordinary_closure_latency_p95_s': ('普通闭环时延P95', 's', '只统计实际闭环的普通任务'),
    'effective_ground_product_tasks': ('具有有效到地产品的任务数', 'tasks', '按每任务最高有效产品去重，含闭环及仅部分交付'),
    'partial_only_tasks': ('仅部分交付任务数', 'tasks', '有有效到地产品但不满足最终需求'),
    'no_effective_ground_product_tasks': ('无有效到地产品任务数', 'tasks', '最高有效产品none'),
    'sent_without_effective_product_tasks': ('已发送但无有效到地产品任务数', 'tasks', '实际TX>0且最高有效产品none；不推断原因'),
    'sent_without_effective_product_MB': ('已发送但无有效产品的下传量', 'MB', '上述任务实际TX字节之和'),
    'right_censored_tasks': ('期末右删失数', 'tasks', '21600s尚未过DDL且未闭环；不强判失败'),
    'deadline_failed_tasks': ('DDL已过未闭环数', 'tasks', 'deadline_elapsed_unclosed'),
    'chip_peak_C': ('芯片温度峰值', 'degC', '审计摘要的积分区间真实峰值'),
    'RF_peak_C': ('RF温度峰值', 'degC', '审计摘要峰值；约1e-8浮点误差按原1e-7公差判定'),
    'SOC_min': ('最小荷电状态', 'fraction', '审计摘要最小SOC'),
    'safety_violations': ('实际安全违例数', 'events', '原审计口径'),
    'thermal_wait_s': ('实际热保护等待', 's', '物理时间，不是桌面耗时或地面推理等待'),
    'ground_path_events': ('地面推理路径事件数', 'events', '非LOCAL到地事件；逐事件校验接收后独立4.29s'),
    'ground_queue_wait_max_s': ('地面推理最大排队时延', 's', '非LOCAL地面开始时间减完整接收时间'),
    'ground_service_max_error_s': ('地面4.29s处理最大误差', 's', '非LOCAL处理时长减4.29s的绝对值最大值'),
    'active_wall_s': ('有效路径桌面墙钟时间', 's', '原摘要active_wall_s，恢复作废尾段不计其中，不作算法速度结论'),
    'elapsed_since_original_start_s': ('从初始启动到完成的总历时', 's', '原COMPLETED总wall，含中断及恢复等待'),
    'CPU_s': ('有效路径桌面CPU时间', 's', '非星载实测；恢复作废尾段不计其中'),
    'peak_RSS_MB': ('进程峰值工作集', 'MB', '原结果peak_rss_bytes/1e6'),
    'candidate_count': ('候选总次数', 'candidates', '候选次数不等于独立任务数'),
    'feasible_candidates': ('可行候选次数', 'candidates', '同一任务可有多个候选'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def near(a, b, tol=1e-8):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=tol), (a, b)


def truth(v):
    assert v in ('True', 'False', True, False), v
    return v is True or v == 'True'


def columns(rows):
    return list(dict.fromkeys(k for r in rows for k in r))


def write_csv(path, rows):
    assert rows, path
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns(rows))
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def latency(values):
    if not values:
        return None, None
    data = sorted(values)
    p = (len(data) - 1) * .95
    lo, hi = math.floor(p), math.ceil(p)
    return st.mean(data), data[lo] + (data[hi] - data[lo]) * (p - lo)


def workbook(path, tables):
    """Small, dependency-free XLSX with typed cells, filters and frozen headers."""
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    def col(n):
        out = ''
        while n:
            n, r = divmod(n - 1, 26)
            out = chr(65 + r) + out
        return out
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        types = ['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                 '<Default Extension="xml" ContentType="application/xml"/>',
                 '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
        types += [f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                  for i in range(1, len(tables) + 1)]
        z.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + ''.join(types) + '</Types>')
        z.writestr('_rels/.rels', f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>' + ''.join(
            f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, (name, _) in enumerate(tables, 1)) + '</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(
            f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(tables) + 1)) + '</Relationships>')
        for i, (_, rows) in enumerate(tables, 1):
            keys = columns(rows)
            chunks = [f'<worksheet xmlns="{ns}"><dimension ref="A1:{col(len(keys))}{len(rows)+1}"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultColWidth="21"/><sheetData>']
            for rn, row in enumerate([dict(zip(keys, keys)), *rows], 1):
                cells = []
                for cn, key in enumerate(keys, 1):
                    value = row.get(key)
                    ref = f'{col(cn)}{rn}'
                    if value is None:
                        continue
                    if isinstance(value, bool):
                        cells.append(f'<c r="{ref}" t="b"><v>{int(value)}</v></c>')
                    elif isinstance(value, (int, float)) and math.isfinite(value):
                        cells.append(f'<c r="{ref}"><v>{value}</v></c>')
                    else:
                        value = escape(str(value).replace('\x00', '')[:32767])
                        cells.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{value}</t></is></c>')
                chunks.append(f'<row r="{rn}">' + ''.join(cells) + '</row>')
            chunks.append(f'</sheetData><autoFilter ref="A1:{col(len(keys))}{len(rows)+1}"/></worksheet>')
            z.writestr(f'xl/worksheets/sheet{i}.xml', ''.join(chunks))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for i, (_, rows) in enumerate(tables, 1):
            assert len(ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml')).find(f'{{{ns}}}sheetData')) == len(rows) + 1


class Extractor:
    def __init__(self):
        self.provenance = []
        self.seen = set()
        self.groups = []
        self.delays = []
        self.windows = []
        self.audit_checks = []
        self.prices = None
        self.manifests = {name: digest(PACKAGE / name) for name in ('SOURCE_MANIFEST.json', 'INPUT_MANIFEST.json')}

    def source(self, p, role):
        p = Path(p)
        if p not in self.seen:
            self.seen.add(p)
            self.provenance.append(dict(role=role, path=str(p), bytes=p.stat().st_size, sha256=digest(p)))

    def extract(self, completed_row, job):
        arm, seed = job['arm'], job['seed']
        attempt = Path(completed_row['result_path'])
        result = attempt / 'RESULT'
        for name in ('AUDIT.json', 'IDENTITY.json', 'COMPLETED.json',
                     'RESULT/AUDITED_SUMMARY.json', 'RESULT/CONFIG_EFFECTIVE.json',
                     'RESULT/task_accounting.csv', 'RESULT/ground_ledger.json'):
            self.source(attempt / name, f'{arm}_{seed}')
        audit, ident, done = [read(attempt / n) for n in ('AUDIT.json', 'IDENTITY.json', 'COMPLETED.json')]
        s, cfg = [read(result / n) for n in ('AUDITED_SUMMARY.json', 'CONFIG_EFFECTIVE.json')]
        assert ident['job'] == job and ident['manifests'] == self.manifests
        assert cfg['input_hashes'] == job['input_hashes']
        assert s['arm'] == arm and s['method'] == job['method'] and s['seed'] == seed
        assert done['phase'] == 'COMPLETED' and completed_row['status'] == 'COMPLETED'
        assert done['time_s'] == s['stop_s'] == 21600 and s['arrivals'] == 1600
        assert s['terminal_settled'] is True and cfg['ground_processing'] == GROUND_RULE
        assert audit['pass'] is True and not audit['failures'] and all(audit['checks'].values())
        assert len(audit['checks']) == 20 and s['actual_safety_violations'] == 0
        assert digest(PACKAGE / 'configs/network_revision' / f"{job['variant']}.json") == job['config_sha256']
        if self.prices is None:
            self.prices = cfg['public_prices']
        assert self.prices == cfg['public_prices']
        tasks = {t['task_id']: t for t in csv_read(result / 'task_accounting.csv')}
        assert len(tasks) == 1600
        assert sum(truth(t['HV']) for t in tasks.values()) == 448
        assert all(truth(t['HV']) == (t['value_level'] in ('high', 'urgent')) for t in tasks.values())
        events = read(result / 'ground_ledger.json')
        first_close, highest = {}, {}
        for e in sorted(events, key=lambda e: (e['ground_available_s'], e['product_id'])):
            assert e['task_id'] in tasks and e['product'] in LEVEL
            if not e['valid']:
                continue
            assert e['ground_available_s'] <= min(21600, float(tasks[e['task_id']]['deadline_s'])) + 1e-7
            tid = e['task_id']
            if LEVEL[e['product']] > LEVEL[highest.get(tid, 'none')]:
                highest[tid] = e['product']
            if e['closes']:
                first_close.setdefault(tid, e['ground_available_s'])
        coverage, groups_delays = Counter(), defaultdict(list)
        for tid, t in tasks.items():
            product = highest.get(tid, 'none')
            closed = tid in first_close
            assert product == t['highest_effective_ground_product']
            assert closed == truth(t['satisfies_requirement'])
            assert truth(t['partial_only']) == (product != 'none' and not closed)
            coverage[product + ('_fulfilled' if closed else '_partial')] += 1
            if closed:
                d = first_close[tid] - float(t['arrival_s'])
                assert d >= -1e-7
                label = 'HV' if truth(t['HV']) else 'ordinary'
                groups_delays['all'].append(d)
                groups_delays[label].append(d)
                self.delays.append(dict(method=arm, seed=seed, task_id=tid,
                    group=label, arrival_s=float(t['arrival_s']),
                    first_valid_closure_s=first_close[tid], deadline_s=float(t['deadline_s']),
                    closure_latency_s=d))
        assert len(first_close) == s['closed']
        costs = {key: math.fsum(float(t[key]) for t in tasks.values()) for key in ('E_cost', 'B_cost', 'Q_cost')}
        gross = math.fsum(float(t['gross']) for t in tasks.values())
        near(gross, s['gross'])
        near(math.fsum(e['gross'] for e in events), gross)
        for key, native in [('E_cost', 'energy'), ('B_cost', 'downlink'), ('Q_cost', 'terminal_backlog')]:
            near(costs[key], s['costs'][native])
        near(gross - math.fsum(costs.values()), s['net'])
        ledger = s['ledger']
        energy = math.fsum(math.fsum(a.get(k, 0) for k in ('inference_j', 'encoding_j', 'tx_j'))
                           for a in ledger['by_task'].values())
        tx = math.fsum(a.get('downlink_mb', 0) for a in ledger['by_task'].values())
        backlog = math.fsum(float(t['ready_backlog_MB']) for t in tasks.values())
        near(energy, ledger['controlled_j'], 1e-5)
        near(tx, ledger['downlink_mb'])
        near(backlog, ledger['backlog_mb'])
        near(energy / 3600 * self.prices['energy_price_per_Wh'], costs['E_cost'])
        near(tx * self.prices['downlink_price_per_MB'], costs['B_cost'])
        near(backlog * self.prices['terminal_backlog_price_per_MB'], costs['Q_cost'])
        near(ledger['background_j'], 191376, 1e-5)
        row = dict(method=arm, seed=seed, configuration='GROUND_PARALLEL_V1', status='COMPLETED',
            audit_pass=True, audit_checks_passed=len(audit['checks']), horizon_s=21600,
            arrivals=1600, HV_arrivals=448, ordinary_arrivals=1152,
            gross_utility=gross, overall_net_utility=s['net'], HV_gross_utility=s['groups']['high_value']['gross'],
            HV_net_utility=s['groups']['high_value']['net'], ordinary_net_utility=s['groups']['ordinary']['net'],
            energy_penalty=costs['E_cost'], downlink_penalty=costs['B_cost'],
            terminal_backlog_penalty=costs['Q_cost'], resource_total_penalty=math.fsum(costs.values()),
            closed_all=len(first_close), closed_HV=len(groups_delays['HV']), closed_ordinary=len(groups_delays['ordinary']),
            closure_rate_all_percent=100 * len(first_close) / 1600,
            closure_rate_HV_percent=100 * len(groups_delays['HV']) / 448,
            closure_rate_ordinary_percent=100 * len(groups_delays['ordinary']) / 1152,
            controlled_energy_kJ=energy / 1000, actual_downlink_MB=tx,
            terminal_backlog_MB=backlog, background_energy_kJ=ledger['background_j'] / 1000,
            chip_peak_C=s['peak_temperatures_c']['chip'], RF_peak_C=s['peak_temperatures_c']['rf'],
            SOC_min=s['min_soc'], safety_violations=s['actual_safety_violations'], thermal_wait_s=s['thermal_wait_s'],
            active_wall_s=done['active_wall_s'], elapsed_since_original_start_s=done['total_wall_s'],
            CPU_s=done['cpu_s'], peak_RSS_MB=done['peak_rss_bytes']/1e6,
            candidate_count=s['candidate_denominator'], feasible_candidates=s['feasible_candidates'])
        for label, native in [('all', 'all'), ('HV', 'high_value'), ('ordinary', 'ordinary')]:
            subset = [t for t in tasks.values() if label == 'all' or truth(t['HV']) == (label == 'HV')]
            g = s['groups'][native]
            grouped_gross = math.fsum(float(t['gross']) for t in subset)
            gc = {k: math.fsum(float(t[k]) for t in subset) for k in costs}
            grouped_net = grouped_gross - math.fsum(gc.values())
            near(grouped_gross, g['gross'])
            near(grouped_net, g['net'])
            for key, native_cost in [('E_cost', 'energy_cost'), ('B_cost', 'downlink_cost'), ('Q_cost', 'terminal_cost')]:
                near(gc[key], g[native_cost])
            assert g['closed'] == len(groups_delays[label]) and g['tasks'] == len(subset)
            dmean, dp95 = latency(groups_delays[label])
            prefix = '' if label == 'all' else label + '_'
            row[prefix + 'closure_latency_mean_s'] = dmean
            row[prefix + 'closure_latency_p95_s'] = dp95
            self.groups.append(dict(method=arm, seed=seed, group=label, tasks=len(subset),
                gross_utility=grouped_gross, net_utility=grouped_net,
                energy_penalty=gc['E_cost'], downlink_penalty=gc['B_cost'], terminal_backlog_penalty=gc['Q_cost'],
                resource_total_penalty=math.fsum(gc.values()), closed=g['closed'],
                closure_rate_percent=100*g['closed']/len(subset),
                closure_latency_mean_s=dmean, closure_latency_p95_s=dp95))
        for level, label in PRODUCTS:
            for suffix in ('fulfilled', 'partial'):
                row[label + '_' + suffix] = coverage[level + '_' + suffix]
                native = level + ('_satisfied' if suffix == 'fulfilled' else '_partial')
                assert row[label + '_' + suffix] == s['ground_state_counts'].get(native, 0)
            row[label + '_total'] = row[label + '_fulfilled'] + row[label + '_partial']
        row['effective_ground_product_tasks'] = sum(row[label + '_total'] for _, label in PRODUCTS)
        row['partial_only_tasks'] = sum(row[label + '_partial'] for _, label in PRODUCTS)
        row['no_effective_ground_product_tasks'] = 1600-row['effective_ground_product_tasks']
        assert row['effective_ground_product_tasks'] == row['closed_all'] + row['partial_only_tasks']
        assert row['no_effective_ground_product_tasks'] == s['ground_state_counts'].get('none', 0)
        sent_none = [tid for tid in tasks if highest.get(tid, 'none') == 'none'
                     and ledger['by_task'].get(tid, {}).get('downlink_mb', 0) > 0]
        row['sent_without_effective_product_tasks'] = len(sent_none)
        row['sent_without_effective_product_MB'] = math.fsum(ledger['by_task'][tid]['downlink_mb'] for tid in sent_none)
        statuses = s['groups']['all']['status_counts']
        row['right_censored_tasks'] = sum(v for k, v in statuses.items() if k.startswith('right_censored'))
        row['deadline_failed_tasks'] = statuses.get('deadline_elapsed_unclosed', 0)
        assert row['closed_all'] + row['right_censored_tasks'] + row['deadline_failed_tasks'] == 1600
        ground = [e for e in events if e['route'] != 'LOCAL']
        waits = [e['ground_start_s']-e['tx_end_s'] for e in ground]
        errors = [abs(e['ground_available_s']-e['ground_start_s']-4.29) for e in ground]
        assert max(map(abs, waits), default=0) < 1e-7 and max(errors, default=0) < 1e-7
        row.update(ground_path_events=len(ground), ground_queue_wait_max_s=max(waits, default=0),
                   ground_service_max_error_s=max(errors, default=0), result_path=str(result),
                   attempt=attempt.name, source_manifest_sha256=self.manifests['SOURCE_MANIFEST.json'],
                   input_manifest_sha256=self.manifests['INPUT_MANIFEST.json'],
                   config_sha256=job['config_sha256'])
        for w in s['windows']:
            self.windows.append(dict(method=arm, seed=seed, **w))
        near(math.fsum(w['downlink_MB'] for w in s['windows']), tx)
        for k, v in audit['checks'].items():
            self.audit_checks.append(dict(method=arm, seed=seed, check=k, pass_=v, source=str(attempt/'AUDIT.json')))
        return row


def compare(rows, prices):
    baseline = {(r['method'], int(r['seed'])): r for r in csv_read(PACKAGE/'BASELINE_METRICS.csv')}
    sources = {(r['arm'], int(r['seed'])): r for r in csv_read(PACKAGE/'BASELINE_SOURCES.csv')}
    long, paired = [], []
    for r in rows:
        key = r['method'], r['seed']
        b, source = baseline[key], sources[key]
        near(float(b['controlled_J'])/3600*prices['energy_price_per_Wh'],float(b['E_cost']))
        near(float(b['downlink_MB'])*prices['downlink_price_per_MB'],float(b['B_cost']))
        near(float(b['Q_end_MB'])*prices['terminal_backlog_price_per_MB'],float(b['Q_cost']))
        near(float(b['gross'])-sum(float(b[k]) for k in ('E_cost','B_cost','Q_cost')),float(b['net']))
        old = {new: float(b[native]) for new, native in BASE_MAP.items()}
        old.update(controlled_energy_kJ=float(b['controlled_J'])/1000,
                   closed_all=int(b['HV_closed'])+int(b['ordinary_closed']),
                   resource_total_penalty=sum(float(b[k]) for k in ('E_cost', 'B_cost', 'Q_cost')),
                   ordinary_net_utility=float(b['net'])-float(b['HV_net']))
        for metric, old_value in old.items():
            delta = r[metric]-old_value
            long.append(dict(method=r['method'], seed=r['seed'], metric=metric,
                old=old_value, new=r[metric], delta=delta,
                relative_change_percent=100*delta/old_value if old_value else None,
                scope='CONFIGURATION_AND_VERSION_COMPARISON_NOT_CAPACITY_ONLY',
                baseline_evidence='PACKAGED_METRICS_AND_SOURCE_HASHES_ONLY'))
        near(r['overall_net_utility']-old['overall_net_utility'],
             (r['gross_utility']-old['gross_utility'])-(r['resource_total_penalty']-old['resource_total_penalty']))
        p = dict(method=r['method'], seed=r['seed'])
        for metric in ('gross_utility','overall_net_utility','HV_net_utility','ordinary_net_utility',
                       'resource_total_penalty','energy_penalty','downlink_penalty','terminal_backlog_penalty',
                       'closed_all','closed_HV','closed_ordinary','controlled_energy_kJ','actual_downlink_MB'):
            p.update({f'old_{metric}': old[metric], f'new_{metric}': r[metric], f'delta_{metric}': r[metric]-old[metric]})
        p.update(baseline_original_path=source['result'], baseline_summary_sha256=source['summary_sha256'],
                 baseline_audit_sha256=source['audit_sha256'], new_result_path=r['result_path'],
                 baseline_reaudit_performed=False)
        paired.append(p)
    return long, paired


def subset_stats(rows):
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
               and k not in ('seed', 'horizon_s', 'audit_checks_passed')]
    out = []
    for method in METHODS:
        rr = [r for r in rows if r['method'] == method]
        for metric in numeric:
            values = [r[metric] for r in rr]
            out.append(dict(method=method, seeds=';'.join(str(r['seed']) for r in rr), n=len(rr),
                metric=metric, mean=st.mean(values), sample_SD=st.stdev(values),
                minimum=min(values), maximum=max(values),
                scope='DESCRIPTIVE_AVAILABLE_SUBSET_NOT_FIVE_SEEDS'))
    return out


def figures(out, rows, paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':10, 'axes.labelsize':10.5, 'xtick.labelsize':9.2, 'ytick.labelsize':9.2,
        'legend.fontsize':9, 'axes.linewidth':.8, 'pdf.fonttype':42, 'ps.fonttype':42,
        'svg.fonttype':'none', 'savefig.facecolor':'white'})
    x = np.arange(len(rows))
    labels = [str(r['seed']) for r in rows]
    teal, blue, coral, gray = '#79b4c4', '#496f83', '#e28376', '#dce7ec'
    def style(ax, label):
        ax.set_xticks(x, labels)
        for method in METHODS:
            positions=[i for i,r in enumerate(rows) if r['method']==method]
            ax.text(st.mean(positions),-.105,method,transform=ax.get_xaxis_transform(),
                    ha='center',va='top',fontsize=9.5)
        ax.tick_params(direction='in', top=True, right=True, length=2.4, width=.7)
        ax.grid(True, axis='both', color='#e2e5e8', linewidth=.55)
        ax.set_axisbelow(True)
        ax.set_ylabel(label, labelpad=4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#51555a')
        ax.margins(x=.07)
    def caption(ax, text):
        ax.text(.5, -.26, text, transform=ax.transAxes, ha='center', va='top', fontsize=11, fontweight='bold')
    def save(fig, stem):
        for extension in ('pdf','svg','png'):
            fig.savefig(out/f'{stem}.{extension}', dpi=600)
        fig.savefig(out/f'{stem}_PREVIEW.png', dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.8))
    fig.subplots_adjust(left=.068, right=.985, bottom=.12, top=.965, wspace=.37, hspace=.64)
    ax = axes[0,0]
    for shift, metric, color, name in [(-.19,'overall_net_utility',coral,'Overall'),(.19,'HV_net_utility',blue,'HV')]:
        ax.bar(x+shift, [r[metric] for r in rows], .35, color=color, edgecolor='#555b60', linewidth=.55, label=name)
    style(ax,'Net utility')
    ax.set_ylim(0, 335)
    ax.legend(loc='upper right', framealpha=1, edgecolor='#777', handlelength=1.3)
    caption(ax,'(a) Realized net utility')
    ax = axes[0,1]
    hv = [r['closed_HV'] for r in rows]
    ax.bar(x, hv, .62, color=blue, edgecolor='#555b60', linewidth=.55, label='HV')
    bars=ax.bar(x, [r['closed_ordinary'] for r in rows], .62, bottom=hv, color=teal, edgecolor='#555b60', linewidth=.55, label='Ordinary')
    for xi, r in zip(x,rows):
        ax.text(xi, r['closed_all']+24, str(r['closed_all']), ha='center', fontsize=9)
    style(ax,'Closed tasks')
    ax.set_ylim(0,1600)
    ax.legend(loc='upper right', framealpha=1, edgecolor='#777', handlelength=1.3)
    caption(ax,'(b) Requirement closure')
    ax = axes[0,2]
    bottom=np.zeros(len(rows))
    for product, color in [('Metadata',gray),('ROI',teal),('ROI_Ctx',blue)]:
        for suffix in ('fulfilled','partial'):
            values=[r[product+'_'+suffix] for r in rows]
            ax.bar(x, values, .62, bottom=bottom, color=color, edgecolor='#555b60',
                   linewidth=.55, hatch='//' if suffix=='partial' else '',
                   label=product.replace('_','+') if suffix=='fulfilled' and any(values) else None)
            bottom += np.array(values)
    for xi, r in zip(x,rows):
        ax.text(xi,r['effective_ground_product_tasks']+24,str(r['effective_ground_product_tasks']),ha='center',fontsize=9)
    style(ax,'Tasks with a valid ground product')
    ax.set_ylim(0,1600)
    ax.legend(loc='upper right', framealpha=1, edgecolor='#777', handlelength=1.3)
    caption(ax,'(c) Highest valid product')
    for ax, metric, label, title in [
        (axes[1,0],'controlled_energy_kJ','Controllable energy (kJ)','(d) Energy consumption'),
        (axes[1,1],'actual_downlink_MB','Actual downlink (MB)','(e) Downlink volume')]:
        ax.bar(x,[r[metric] for r in rows],.62,color=[teal if r['method']=='GROUND' else coral for r in rows],edgecolor='#555b60',linewidth=.55)
        style(ax,label)
        ax.set_ylim(0,max(r[metric] for r in rows)*1.18)
        for xi,r in zip(x,rows):
            ax.annotate(f"{r[metric]:.1f}",(xi,r[metric]),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
        caption(ax,title)
    ax = axes[1,2]
    for shift, metric, color, name in [(-.19,'closure_latency_mean_s',teal,'Mean'),(.19,'closure_latency_p95_s',blue,'95th percentile')]:
        ax.bar(x+shift,[r[metric] for r in rows],.35,color=color,edgecolor='#555b60',linewidth=.55,label=name)
    style(ax,'Closure latency (s)')
    ax.set_ylim(0,max(r['closure_latency_p95_s'] for r in rows)*1.35)
    ax.legend(loc='upper right',framealpha=1,edgecolor='#777',handlelength=1.3)
    caption(ax,'(f) Latency of closed tasks')
    save(fig,'B5_MAIN_METRICS')
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.8))
    fig.subplots_adjust(left=.085,right=.98,bottom=.12,top=.965,wspace=.24,hspace=.66)
    panels=[('overall_net_utility','Overall net utility','(a) Overall net utility'),
            ('HV_net_utility','HV net utility','(b) High-value net utility'),
            ('closed_all','Closed tasks','(c) Requirement closure'),
            ('resource_total_penalty','Resource penalty (utility)','(d) Total resource penalty')]
    for ax,(metric,ylabel,title) in zip(axes.flat,panels):
        old=[r['old_'+metric] for r in paired]
        new=[r['new_'+metric] for r in paired]
        ax.bar(x-.18,old,.33,color='#c5ccd1',edgecolor='#555b60',linewidth=.55,label='Original reference')
        ax.bar(x+.18,new,.33,color=teal,edgecolor='#555b60',linewidth=.55,label='Independent 4.29 s')
        style(ax,ylabel)
        ax.set_ylim(0,max(old+new)*1.38)
        ax.legend(loc='upper right',framealpha=1,edgecolor='#777',handlelength=1.3)
        caption(ax,title)
    save(fig,'B5_OLD_NEW_COMPARISON')


def table(rows, fields, digits=3):
    def cell(v):
        if v is None:
            return '—'
        if isinstance(v,float):
            return f'{v:,.{digits}f}'
        return html.escape(str(v))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in fields)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+cell(r.get(key))+'</td>' for key,_ in fields)+'</tr>' for r in rows)+'</tbody></table></div>'


def report(out, rows, paired, stats, ex, history):
    stat_index={(r['method'],r['metric']):r for r in stats}
    top_fields=[('method','方法'),('seed','种子'),('gross_utility','毛效用'),('overall_net_utility','总体净效用'),
        ('HV_net_utility','HV净效用'),('resource_total_penalty','总扣分'),('closed_all','闭环总数'),('closed_HV','HV闭环'),
        ('controlled_energy_kJ','可控能耗 kJ'),('actual_downlink_MB','实际下传 MB')]
    texts=[
        'GROUND_PARALLEL_V1 B包数据汇总',
        '5/5组已完成21600s并通过原20项审计；每组1600任务（HV448，普通1152），实际安全违例均为0。完成时间：2026-10-31 21:42:52（香港时间）。',
        '范围：GROUND种子4104、4105（n=2）；OFFLOAD种子4103、4104、4105（n=3）。缺少A包的GROUND4101–4103及OFFLOAD4101–4102；本报告不构成每方法五种子结果。两方法的n=2/n=3统计仅描述当前子集，不能当作配对五种子方法比较。',
        '配置：GROUND/OFFLOAD的地面推理路径完整接收后各自独立处理4.29s，接收容量不构成约束；OFFLOAD的LOCAL路径不额外加4.29s。逐事件复核地面推理等待为0、处理时长为4.29s（数值公差1e-7s）。实际能耗、实际发送成本、DDL及21600s观察终点均保留。',
        '本次是该冻结配置完整在线运行后的结果，不是固定发送轨迹下的离线估算。独立并行处理与此前“扩大缓存但保留单服务台”的诊断属于不同配置。新旧表是配置及实现版本对比，不把差值归因于单一容量参数；不能仅凭审计PASS认定跨版本等价。',
        '统计：总体/HV净效用均以统一固定价格计费；HV=high+urgent。闭环时延从真实到达到首次有效满足需求的到地产品，只统计本方法实际闭环任务。P95按(n−1)×0.95线性插值。均值和样本SD先逐种子计算再跨种子汇总，不合并不同种子任务冒充五种子。',
        '产品：每任务仅按最高有效到地产品计一次；Metadata、ROI、ROI+Ctx分别区分满足需求与仅部分交付。当前5组仅部分交付均为0。闭环+右删失+DDL已过未闭环=1600，三者互斥；期末未到DDL不自动判失败。',
    ]
    for r,p in zip(rows,paired):
        texts.append(f"{r['method']} {r['seed']}：毛效用{r['gross_utility']:.6f}−E{r['energy_penalty']:.6f}−B{r['downlink_penalty']:.6f}−Q{r['terminal_backlog_penalty']:.6f}＝总体净效用{r['overall_net_utility']:.6f}；HV净效用{r['HV_net_utility']:.6f}；闭环{r['closed_all']}（HV{r['closed_HV']}，普通{r['closed_ordinary']}），可控能耗{r['controlled_energy_kJ']:.3f}kJ，实际下传{r['actual_downlink_MB']:.3f}MB；相比同种子旧参考净效用增加{p['delta_overall_net_utility']:.6f}。")
    for method in METHODS:
        rr=[r for r in rows if r['method']==method]
        texts.append(f"{method}当前子集n={len(rr)}：总体净效用{st.mean(r['overall_net_utility'] for r in rr):.3f}±{st.stdev(r['overall_net_utility'] for r in rr):.3f}，HV净效用{st.mean(r['HV_net_utility'] for r in rr):.3f}±{st.stdev(r['HV_net_utility'] for r in rr):.3f}（均值±样本SD）。")
    texts.extend([
        '主要观察：GROUND4104/4105闭环数由旧参考414/393增至1223/1165，HV闭环由240/224增至368/370；毛效用增量较大，但新增下传、能耗及期末ready积压费用抵消了大部分收益。两组期末ready积压308/568MB，对应Q扣分28.343/52.270。',
        'HV净效用高于总体净效用不矛盾：普通任务净效用为负。两组GROUND普通任务净效用分别为'+ '、'.join(f"{r['ordinary_net_utility']:.3f}" for r in rows if r['method']=='GROUND')+'；成本归属表给出了HV/普通收益和三项费用。',
        'OFFLOAD三组相对同种子旧参考总体净效用均增加，同时HV闭环增加、普通闭环减少；具体机制需分析在线事件，不将汇总表视为单因果证明。不同方法闭环样本不同，时延均值/P95变化包含闭环样本构成变化。',
        '恢复记录：OFFLOAD4104在18272.317683s因STATUS文件原子替换被拒绝访问失败，保留原attempt_01，从13380s检查点恢复至attempt_02并完成。失败前2818条候选记录重放字节完全一致；原物理、计分与审计源码不变。OFFLOAD4105旧尝试中断后恢复至attempt_02；原中断原因未记录，不推断。当前未解决失败为0，历史记录单列。',
        '证据边界：新5组从task_accounting、ground_ledger及实际资源账独立复算；原始20项审计另保留。331个冻结分发文件哈希再次核对。旧参考仅使用包内BASELINE_METRICS.csv及BASELINE_SOURCES.csv，旧原始轨迹位于另一台电脑D盘，本轮未重审旧结果，也未虚构旧时延。',
        '运行开销：active_wall_s及CPU_s沿用成功路径统计，恢复作废尾段不包含其中；elapsed_since_original_start_s保留初始启动到完成的总历时。地面配置及计算实现版本均变化，不由桌面耗时差宣称算法加速。RF峰值约65+1e-8℃为原审计1e-7公差内数值误差，未改物理阈值。',
        '未改动PROP、消融、原代码及原始结果；未启动任何实验重跑。输出仅含轻量汇总、图和可复现脚本，不打包原始日志/检查点。',
    ])
    (out/'00_RESULTS_CN.txt').write_text('\n\n'.join(texts)+'\n',encoding='utf-8-sig')
    summary=[]
    for metric in ('gross_utility','overall_net_utility','HV_net_utility','ordinary_net_utility','resource_total_penalty',
        'energy_penalty','downlink_penalty','terminal_backlog_penalty','closed_all','closed_HV','closed_ordinary',
        'closure_rate_all_percent','closure_rate_HV_percent','closure_rate_ordinary_percent','controlled_energy_kJ',
        'actual_downlink_MB','closure_latency_mean_s','closure_latency_p95_s','HV_closure_latency_mean_s'):
        d=DESCRIPTIONS[metric]
        r=dict(metric=d[0],unit=d[1])
        for method in METHODS:
            a=stat_index[method,metric]
            r[method]=f"{a['mean']:.3f} ± {a['sample_SD']:.3f}"
        summary.append(r)
    utility_fields=[('method','方法'),('seed','种子'),('gross_utility','毛效用'),('energy_penalty','E扣分'),
        ('downlink_penalty','B扣分'),('terminal_backlog_penalty','Q扣分'),('overall_net_utility','总体净效用'),
        ('HV_net_utility','HV净效用'),('ordinary_net_utility','普通净效用')]
    closure_fields=[('method','方法'),('seed','种子'),('closed_all','闭环'),('closure_rate_all_percent','闭环率 %'),
        ('closed_HV','HV闭环'),('closure_rate_HV_percent','HV闭环率 %'),('closed_ordinary','普通闭环'),
        ('closure_rate_ordinary_percent','普通闭环率 %'),('right_censored_tasks','期末右删失'),('deadline_failed_tasks','DDL已过未闭环')]
    product_fields=[('method','方法'),('seed','种子'),('Metadata_total','Metadata'),('ROI_total','ROI'),
        ('ROI_Ctx_total','ROI+Ctx'),('partial_only_tasks','仅部分交付'),('effective_ground_product_tasks','有效交付总数'),
        ('no_effective_ground_product_tasks','无有效产品'),('sent_without_effective_product_tasks','已发但无有效产品'),
        ('sent_without_effective_product_MB','对应下传 MB')]
    latency_fields=[('method','方法'),('seed','种子'),('closed_all','时延样本数'),('closure_latency_mean_s','总体均值 s'),
        ('closure_latency_p95_s','总体P95 s'),('HV_closure_latency_mean_s','HV均值 s'),
        ('HV_closure_latency_p95_s','HV P95 s'),('ordinary_closure_latency_mean_s','普通均值 s')]
    safety_fields=[('method','方法'),('seed','种子'),('controlled_energy_kJ','可控能耗 kJ'),('actual_downlink_MB','实际下传 MB'),
        ('terminal_backlog_MB','期末ready MB'),('chip_peak_C','芯片峰温 ℃'),('RF_peak_C','RF峰温 ℃'),('SOC_min','最小SOC'),
        ('safety_violations','违例'),('thermal_wait_s','热保护等待 s')]
    change_fields=[('method','方法'),('seed','种子'),('old_overall_net_utility','旧净效用'),('new_overall_net_utility','新净效用'),
        ('delta_overall_net_utility','净效用变化'),('delta_gross_utility','毛效用变化'),('delta_resource_total_penalty','总扣分变化'),
        ('old_closed_all','旧闭环'),('new_closed_all','新闭环'),('delta_HV_net_utility','HV净效用变化')]
    h=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
       '<title>GROUND/OFFLOAD B包完成数据</title><style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#243542;background:#f4f7fa;margin:0}main{max-width:1450px;margin:auto;padding:30px 28px 60px;background:white}h1{font-size:27px;margin:0 0 12px}h2{font-size:21px;margin:30px 0 10px;color:#244b61}p{margin:10px 0}.notice{padding:12px 18px;background:#e9f3f6;border-left:5px solid #629cad}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:14px;white-space:nowrap;width:100%;margin:12px 0}th{background:#244b61;color:white;text-align:right}th,td{padding:7px 11px;border:1px solid #dbe3ea;text-align:right}th:first-child,td:first-child{text-align:left}tr:nth-child(even){background:#f3f7fa}.plot{width:100%;display:block;margin:16px 0}a{color:#21617b}.note{color:#526371;font-size:14px}</style><main>',
       '<h1>'+texts[0]+'</h1><p class="notice">'+texts[1]+'</p>',
       '<p><a href="GROUND_PARALLEL_B5_RESULTS.xlsx">下载Excel工作簿</a>　<a href="per_seed_metrics.csv">逐种子CSV</a>　<a href="00_RESULTS_CN.txt">纯文本报告</a>　<a href="DELIVERY_FILES.csv">全部文件清单</a></p>']
    h += ['<p>'+html.escape(t)+'</p>' for t in texts[2:7]]
    h += ['<h2>主要结果：逐种子</h2>',table(rows,top_fields),
          '<h2>收益与费用守恒</h2>',table(rows,utility_fields),
          '<p>每行：毛效用−能耗扣分−实际下传扣分−期末积压扣分＝总体净效用；总体净效用＝HV净效用＋普通净效用。完整分组费用在 <a href="group_costs_and_closure.csv">分组明细CSV</a>。</p>',
          '<h2>闭环、有效产品与观察终点</h2>',table(rows,closure_fields),table(rows,product_fields),
          '<h2>闭环时延</h2>',table(rows,latency_fields),
          '<p>时延只覆盖已闭环任务；<a href="closed_task_latencies.csv">4476个闭环事件的轻量时延表</a>保留任务ID、到达、有效闭环和DDL，便于复算。跨方法样本不同，时延差不能单独解释为同一任务加速。</p>',
          '<h2>资源与安全</h2>',table(rows,safety_fields),
          '<h2>当前子集均值±样本标准差</h2>',
          '<p class="notice">GROUND n=2（4104、4105）；OFFLOAD n=3（4103、4104、4105）。样本数及种子不同，本表用于描述当前进度，不作为完整五种子配对统计。</p>',
          table(summary,[('metric','指标'),('unit','单位'),('GROUND','GROUND n=2'),('OFFLOAD','OFFLOAD n=3')]),
          '<h2>新旧配置：同方法、同种子对照</h2>',table(paired,change_fields),
          '<p>旧参考来自包内冻结的指标CSV及来源哈希；新结果为GROUND_PARALLEL_V1。该表为配置及实现版本对比，不声称跨版本等价或单一容量因果。旧原始日志未在本轮重审。</p>',
          '<h2>数据图</h2><p>每根柱对应一个独立种子结果，未伪装成五种子均值。图中所有产品均已满足需求，因而没有部分交付斜线段。</p>',
          '<img class="plot" src="B5_MAIN_METRICS.png" alt="Per-seed utilities closure products energy downlink and latencies">',
          '<p><a href="B5_MAIN_METRICS.pdf">矢量PDF</a> · <a href="B5_MAIN_METRICS.svg">SVG</a> · <a href="B5_MAIN_METRICS.png">PNG</a></p>',
          '<img class="plot" src="B5_OLD_NEW_COMPARISON.png" alt="Original reference versus independent ground processing">',
          '<p><a href="B5_OLD_NEW_COMPARISON.pdf">新旧对比PDF</a> · <a href="B5_OLD_NEW_COMPARISON.svg">SVG</a></p>',
          '<h2>可得出的结论与证据边界</h2>']
    h += ['<p>'+html.escape(t)+'</p>' for t in texts[-7:-3]]
    h += ['<h2>恢复及失败记录</h2>',table(history,[('arm','方法'),('seed','种子'),('status','历史状态'),('error','历史原因'),('time_s','中断时间 s'),('current_status','当前状态')]),
          '<p>'+html.escape(texts[-3])+'</p><p>'+html.escape(texts[-2])+'</p>',
          '<h2>完整轻量文件</h2><ul>']
    for name,label in [('GROUND_PARALLEL_B5_RESULTS.xlsx','Excel：逐种子、分组、统计、新旧比较、窗口及审计'),
        ('per_seed_metrics.csv','逐种子全部指标'),('subset_mean_sample_sd.csv','当前可用子集均值与样本标准差'),
        ('old_new_paired.csv','同方法同种子新旧差值宽表'),('old_new_long.csv','新旧差值长表'),
        ('group_costs_and_closure.csv','总体/HV/普通费用及闭环'),('contact_window_metrics.csv','四个接触窗口资源明细'),
        ('closed_task_latencies.csv','闭环时延逐任务轻量数据'),('audit_checks.csv','100项原审计检查结果'),
        ('failure_history.csv','历史失败及恢复记录'),('source_provenance.csv','输入证据路径及SHA-256'),
        ('metric_dictionary.csv','指标定义与单位'),('VALIDATION.json','独立复算与交付校验'),
        ('summarize_ground_parallel_b5.py','只读整理及绘图脚本')]:
        h.append('<li><a href="'+name+'">'+label+'</a></li>')
    h.append('</ul><p class="note">'+html.escape(texts[-1])+'</p></main></html>')
    (out/'00_RESULTS_CN.html').write_text('\n'.join(h),encoding='utf-8')
    return texts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    out=args.out.resolve()
    assert not out.is_relative_to(RUNS.resolve()) and not out.is_relative_to(PACKAGE.resolve())
    out.mkdir(parents=True,exist_ok=True)
    finished=read(RUNS/'FINISHED.json')
    assert finished['all_completed'] and len(finished['rows'])==5
    assert read(RUNS/'FINAL_SOURCE_VERIFICATION.json')['source_and_input_manifests_match']
    frozen=csv_read(PACKAGE/'FILES_SHA256.csv')
    for r in frozen:
        p=PACKAGE/r['path']
        assert p.resolve().is_relative_to(PACKAGE.resolve())
        assert p.stat().st_size==int(r['bytes']) and digest(p)==r['sha256'],str(p)
    ex=Extractor()
    for name in ('SOURCE_MANIFEST.json','INPUT_MANIFEST.json','RUN_MATRIX.json','PACKAGE_B_JOBS.json',
                 'FILES_SHA256.csv','BASELINE_METRICS.csv','BASELINE_SOURCES.csv'):
        ex.source(PACKAGE/name,'frozen_package')
    for name in ('FINISHED.json','FINAL_SOURCE_VERIFICATION.json','FAILURE_HISTORY.csv','COMPLETED_NEW_METRICS.csv'):
        ex.source(RUNS/name,'completed_registry')
    jobs={(j['arm'],j['seed']):j for j in read(PACKAGE/'PACKAGE_B_JOBS.json')['jobs']}
    assert set(jobs)=={('GROUND',4104),('GROUND',4105),('OFFLOAD',4103),('OFFLOAD',4104),('OFFLOAD',4105)}
    rows=[]
    for done in sorted(finished['rows'],key=lambda r:(METHODS.index(r['arm']),r['seed'])):
        rows.append(ex.extract(done,jobs[done['arm'],done['seed']]))
    # Cross-check the independent extraction against the existing completed CSV.
    registered={(r['method'],int(r['seed'])):r for r in csv_read(RUNS/'COMPLETED_NEW_METRICS.csv')}
    for r in rows:
        b=registered[r['method'],r['seed']]
        near(r['overall_net_utility'],float(b['net']))
        near(r['HV_net_utility'],float(b['HV_net']))
        near(r['actual_downlink_MB'],float(b['downlink_MB']))
        assert r['closed_all']==int(b['HV_closed'])+int(b['ordinary_closed'])
    long,paired=compare(rows,ex.prices)
    stats=subset_stats(rows)
    history=csv_read(RUNS/'FAILURE_HISTORY.csv')
    assert len(csv_read(RUNS/'FAILURES.csv'))==0
    assert len(ex.delays)==sum(r['closed_all'] for r in rows)==4476
    assert len(ex.windows)==20 and len(ex.audit_checks)==100
    dictionary=[]
    meta={'method':('方法','text','GROUND或OFFLOAD；GROUND内部method=B1'),
          'seed':('种子','integer','当前可用GROUND4104/4105，OFFLOAD4103/4104/4105'),
          'configuration':('地面配置','text','GROUND_PARALLEL_V1：独立4.29s'),
          'status':('最终状态','text','只纳入COMPLETED'), 'audit_pass':('审计结果','boolean','原20项审计均通过'),
          'audit_checks_passed':('原审计通过项数','checks','每组20项'), 'horizon_s':('观察终点','s','21600s'),
          'HV_arrivals':('高价值到达数','tasks','448'), 'ordinary_arrivals':('普通到达数','tasks','1152'),
          'result_path':('正式成功结果目录','path','由FINISHED清单定位，不混入失败尝试'),
          'attempt':('正式成功尝试','text','OFFLOAD4104/4105为attempt_02'),
          'source_manifest_sha256':('代码清单哈希','SHA-256','核对冻结源身份'),
          'input_manifest_sha256':('输入清单哈希','SHA-256','核对冻结输入身份'),
          'config_sha256':('种子配置哈希','SHA-256','与JOB及冻结配置一致')}
    for key in columns(rows):
        if key in DESCRIPTIONS:
            label,unit,note=DESCRIPTIONS[key]
        elif key in meta:
            label,unit,note=meta[key]
        elif any(key.startswith(label+'_') for _,label in PRODUCTS):
            label=key.replace('_fulfilled','满足需求').replace('_partial','仅部分交付').replace('_total','总数')
            unit,note='tasks','每任务最高有效到地产品去重；fulfilled与partial互斥'
        else:
            raise AssertionError(('Missing metric definition',key))
        dictionary.append(dict(metric=key,label_CN=label,unit=unit,definition_CN=note))
    tables={'per_seed_metrics.csv':rows,'group_costs_and_closure.csv':ex.groups,
        'subset_mean_sample_sd.csv':stats,'old_new_paired.csv':paired,'old_new_long.csv':long,
        'contact_window_metrics.csv':ex.windows,'closed_task_latencies.csv':ex.delays,
        'audit_checks.csv':ex.audit_checks,'failure_history.csv':history,
        'source_provenance.csv':ex.provenance,'metric_dictionary.csv':dictionary}
    for name, data in tables.items():
        write_csv(out/name,data)
    figures(out,rows,paired)
    texts=report(out,rows,paired,stats,ex,history)
    readme=[dict(item='范围',value=texts[2]),dict(item='地面配置',value=texts[3]),
            dict(item='统计口径',value=texts[5]),dict(item='固定公共价格',value=json.dumps(ex.prices,ensure_ascii=False)),
            dict(item='旧结果证据',value='包内旧指标CSV和来源哈希，未在本轮重新审计旧原始轨迹'),
            dict(item='保持原结果',value='只读提取；未重跑或修改代码、原结果、PROP或消融')]
    workbook(out/'GROUND_PARALLEL_B5_RESULTS.xlsx',[('说明',readme),('逐种子指标',rows),('分组收益费用',ex.groups),
        ('可用子集均值SD',stats),('新旧配对',paired),('新旧长表',long),('窗口明细',ex.windows),
        ('闭环时延',ex.delays),('审计检查',ex.audit_checks),('失败恢复历史',history),
        ('证据来源',ex.provenance),('指标字典',dictionary)])
    if Path(__file__).resolve() != (out/Path(__file__).name).resolve():
        shutil.copyfile(__file__,out/Path(__file__).name)
    # Source evidence must still be unchanged after plotting and packaging tables.
    for r in ex.provenance:
        assert digest(r['path'])==r['sha256'],r['path']
    for name,data in tables.items():
        assert len(csv_read(out/name))==len(data)
    validation=dict(passed=True,generated_at=datetime.now().astimezone().isoformat(),
        completed_jobs=5,available_seeds={'GROUND':[4104,4105],'OFFLOAD':[4103,4104,4105]},
        complete_five_seeds_per_method=False,per_job_original_checks=20,total_original_checks=100,
        frozen_files_verified=len(frozen),fixed_public_prices=ex.prices,
        independent_checks=dict(source_and_input_identity=True,complete_21600s=True,task_counts_1600=True,
            task_costs_and_net_recomputed=True,actual_energy_and_tx_reconciled=True,group_costs_recomputed=True,
            highest_product_deduplicated=True,closure_and_latency_recomputed=True,
            independent_ground_exact_4_29s=True,no_ground_queue_wait=True,
            old_new_delta_conservation=True,available_subset_mean_sample_SD=True,
            CSV_row_counts=True,XLSX_zip_and_XML=True,source_files_unchanged=True),
        closed_latency_rows=len(ex.delays),source_evidence_files=len(ex.provenance),
        unresolved_failures=0,historical_failed_or_interrupted_attempts=len(history),
        no_simulation_rerun=True,baseline_reaudit_performed=False)
    write_json(out/'VALIDATION.json',validation)
    listing=[dict(filename=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.iterdir())
             if p.is_file() and p.name!='DELIVERY_FILES.csv']
    write_csv(out/'DELIVERY_FILES.csv',listing)
    print(json.dumps(dict(out=str(out),passed=True,jobs=5,metrics_per_seed=len(rows[0]),
        closed_tasks=len(ex.delays),files=len(listing)+1,total_bytes=sum(x['bytes'] for x in listing),
        subset_net=[dict(method=m,n=len([r for r in rows if r['method']==m]),
                        mean=st.mean(r['overall_net_utility'] for r in rows if r['method']==m)) for m in METHODS]),
        ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
