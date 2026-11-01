"""Read-only completion watcher; never changes simulation or audit outcomes."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
ATTEMPT = PACK / 'A_RUNS/4102_OFFLOAD/attempt_03'
REPO = HERE.parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package():
    members = [p for p in HERE.iterdir() if p.is_file() and p.suffix in {'.py', '.md', '.json', '.diff'}]
    result = {}
    for p in members:
        result[p.name] = p
    for name in ['IDENTITY.json', 'AUDIT.json', 'COMPLETED.json', 'ARRIVAL_TAIL_EVENTS.json']:
        if (ATTEMPT/name).exists():
            result['RECOVERED/' + name] = ATTEMPT/name
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['path', 'sha256', 'bytes'])
    for name, path in sorted(result.items()):
        writer.writerow([name, sha(path), path.stat().st_size])
    (HERE/'FILES_SHA256.csv').write_text(stream.getvalue(), encoding='utf-8-sig')
    target = PACK / 'OFFLOAD4102_CLOCK_RECOVERY_EVIDENCE.zip'
    tmp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, path in result.items():
            z.write(path, 'OFFLOAD4102_CLOCK_RECOVERY/' + name)
        z.write(HERE/'FILES_SHA256.csv', 'OFFLOAD4102_CLOCK_RECOVERY/FILES_SHA256.csv')
    os.replace(tmp, target)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name, path in result.items():
            assert hashlib.sha256(z.read('OFFLOAD4102_CLOCK_RECOVERY/'+name)).hexdigest() == sha(path)
    target.with_suffix('.zip.sha256').write_text(sha(target) + '  ' + target.name + '\n', encoding='ascii')


def finish():
    result = read(HERE/'RECOVERY_RESULT.json')
    audit = read(ATTEMPT/'AUDIT.json')
    assert audit['pass'] and (ATTEMPT/'COMPLETED.json').exists()
    assert result['exact_arrivals'] == result['actual_arrivals'] == 1600
    inventory = read(HERE/'FAILED_ATTEMPT_FILES_SHA256.json')
    bad = [r['path'] for r in inventory if sha(REPO/r['path']) != r['sha256']]
    summary = dict(original_failed_files_unchanged=not bad, modified_old_files=bad,
                   final_audit_pass=audit['pass'], final_passed_checks=sum(audit['checks'].values()),
                   exact_arrivals=result['exact_arrivals'], result_sha256=sha(HERE/'RECOVERY_RESULT.json'))
    (HERE/'FINAL_VERIFICATION.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    assert not bad
    note = '\n## 恢复完成\n\n'
    note += f"attempt_03已完成21600秒，{len(audit['checks'])}项原审计全部PASS；1600条到达均与冻结输入精确相等。原失败文件SHA逐一复核未变。\n\n"
    note += '|指标|原失败attempt_02（诊断值）|恢复后正式attempt_03|差值|\n|---|---:|---:|---:|\n'
    for name, old in result['original_failed_metrics'].items():
        note += f"|{name}|{old:.12g}|{result['recovered_metrics'][name]:.12g}|{result['metric_deltas'][name]:.12g}|\n"
    (HERE/'00_REPORT_CN.md').write_text((HERE/'00_REPORT_CN.md').read_text(encoding='utf-8').split('\n## 恢复完成')[0]+note, encoding='utf-8')
    package()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--detach', action='store_true')
    args = ap.parse_args()
    if args.detach:
        log = (HERE/'completion_watch.log').open('ab')
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_BREAKAWAY_FROM_JOB if os.name == 'nt' else 0
        p = subprocess.Popen([sys.executable, str(Path(__file__)), '--watch'], cwd=REPO,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        log.close()
        (HERE/'WATCHER.json').write_text(json.dumps(dict(pid=p.pid, scope='read-only final verification/package'), indent=2))
        package()
        print('WATCHER_PID', p.pid)
        return
    if args.watch:
        while not (HERE/'RECOVERY_RESULT.json').exists():
            if (ATTEMPT/'FAILED.json').exists():
                (HERE/'RECOVERY_STILL_FAILED.json').write_text(json.dumps(read(ATTEMPT/'FAILED.json'), ensure_ascii=False, indent=2), encoding='utf-8')
                package()
                raise SystemExit('REQUIRES_REVIEW_NO_AUDIT_BYPASS')
            time.sleep(20)
        finish()
    else:
        package()


if __name__ == '__main__':
    main()
