"""Validate frozen files and committed prefixes without loading large states."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / '.network_formal6h_work'
MAIN = HERE.parent / 'MAIN_RUNS'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

manifests = {}
for name in ('SOURCE_MANIFEST.json', 'INPUT_MANIFEST.json'):
    for filename, expected in read(SOURCE / name)['files'].items():
        assert sha(SOURCE / filename) == expected, filename
    manifests[name] = sha(SOURCE / name)
jobs = read(SOURCE / 'RUN_MATRIX.json')['jobs']
approved = {}
for seed, number in ((4101, 5), (4102, 2), (4105, 2)):
    attempt = MAIN / f'{seed}_PROP' / f'attempt_{number:02d}'
    failed = read(attempt / 'FAILED.json')
    assert failed['error'] == 'MemoryError()'
    assert 'physics.py' in failed['traceback'] and 'fingerprint' in failed['traceback']
    latest = read(attempt / 'LATEST_CHECKPOINT.json')
    cp = attempt / latest['path']
    meta = read(cp.with_suffix('.json'))
    job = next(j for j in jobs if j['seed'] == seed and j['arm'] == 'PROP')
    assert meta['identity'] == dict(job=job, manifests=manifests, python=sys.version)
    assert sha(cp) == meta['pickle_sha256']
    assert meta['time_s'] <= failed['status']['time_s']
    digest = hashlib.sha256()
    count = 0
    remaining = meta['candidate_bytes']
    with Path(meta['candidate_file']).open('rb') as stream:
        while remaining:
            line = stream.readline(remaining)
            assert line and line.endswith(b'\n'), 'INCOMPLETE_CANDIDATE_PREFIX'
            remaining -= len(line)
            digest.update(line)
            candidate = json.loads(line)
            assert candidate['decision_s'] <= meta['time_s']
            count += 1
    assert count == meta['candidate_count']
    target = HERE / 'BEFORE' / f'{seed}_PROP'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('FAILED.json', 'STATUS.json', 'PROCESS_EXIT.json', 'LATEST_CHECKPOINT.json'):
        shutil.copy2(attempt / name, target / name)
    shutil.copy2(cp.with_suffix('.json'), target / cp.with_suffix('.json').name)
    shutil.copy2(attempt.parent / (attempt.name + '.log'), target / 'worker.log')
    approved[str(attempt.resolve())] = dict(
        seed=seed, arm='PROP', failed_sha256=sha(attempt / 'FAILED.json'),
        latest_pointer_sha256=sha(attempt / 'LATEST_CHECKPOINT.json'),
        checkpoint=str(cp), checkpoint_sha256=meta['pickle_sha256'],
        resume_time_s=meta['time_s'], failure_time_s=failed['status']['time_s'],
        resume_arrivals=meta['status']['arrivals'], resume_closed=meta['status']['closed'],
        candidate_count=count, candidate_prefix_sha256=digest.hexdigest(),
        candidate_prefix_bytes=meta['candidate_bytes'], frozen_identity=meta['identity'],
        deserialization='DEFERRED_TO_ORIGINAL_WORKER_RESTORE_CHECK_UNTIL_HOST_MEMORY_READY')
shutil.copy2(MAIN / 'SUPERVISOR.json', HERE / 'BEFORE_SUPERVISOR.json')
(HERE / 'APPROVED_RESUMES.json').write_text(json.dumps(dict(
    prepared_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    attempts=approved, rule='Only these exact MemoryError attempts; original worker identity and restore gates remain active'),
    ensure_ascii=False, indent=2), encoding='utf-8')
for row in approved.values():
    print('VALIDATED', row['seed'], row['resume_time_s'], row['candidate_count'], row['resume_arrivals'], row['resume_closed'])
