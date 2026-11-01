"""Run the frozen entry point with bounded retries for Windows output-file locks.

Only os.replace for files within this invocation's output directory is guarded.
Scientific sources, checkpoint payloads, identities and audit rules stay intact.
"""
from pathlib import Path
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time


RETRYABLE_WINERRORS = {5, 32, 33}


def make_retry_replace(original, output_root, on_event=None, max_wait_s=10.0):
    output_root = Path(output_root).resolve()

    def guarded_replace(source, destination, *args, **kwargs):
        started = time.monotonic()
        deadline = started + max_wait_s
        delay = 0.02
        retries = 0
        while True:
            try:
                result = original(source, destination, *args, **kwargs)
            except OSError as error:
                remaining = deadline - time.monotonic()
                if (getattr(error, 'winerror', None) not in RETRYABLE_WINERRORS
                        or not Path(destination).resolve().is_relative_to(output_root)
                        or remaining <= 0):
                    raise
                retries += 1
                if on_event and retries == 1:
                    on_event({'event': 'retry_started', 'destination': str(destination),
                              'winerror': error.winerror})
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, 0.5)
            else:
                if on_event and retries:
                    on_event({'event': 'retry_recovered', 'destination': str(destination),
                              'retries': retries, 'wait_s': time.monotonic() - started})
                return result

    return guarded_replace


def main():
    root = Path(__file__).resolve().parent.parent
    runner = Path(__file__).resolve()
    arguments = sys.argv[1:]
    if not arguments:
        raise SystemExit('Use: io_retry_runner.py {selfcheck|worker|launch|summarize} ...')
    if '--out' in arguments:
        output = Path(arguments[arguments.index('--out') + 1]).resolve()
    else:
        output = Path('SELF_CHECK' if arguments[0] == 'selfcheck' else 'outputs').resolve()

    def event(record):
        print('IO_RETRY ' + json.dumps(dict(pid=os.getpid(), epoch=time.time(), **record)),
              file=sys.stderr, flush=True)

    os.replace = make_retry_replace(os.replace, output, event)
    original_popen = subprocess.Popen

    class GuardedPopen(original_popen):
        def __init__(self, command, *args, **kwargs):
            if (isinstance(command, (list, tuple)) and len(command) > 1
                    and Path(command[1]).resolve() == root / 'run.py'):
                command = [command[0], '-I', str(runner), *command[2:]]
            super().__init__(command, *args, **kwargs)

    subprocess.Popen = GuardedPopen
    records = root / '.runtime' / 'io_guard_records'
    records.mkdir(parents=True, exist_ok=True)
    record = {'pid': os.getpid(), 'epoch': time.time(), 'runner': str(runner),
              'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
              'arguments': arguments, 'output_root': str(output),
              'max_retry_wait_s': 10.0, 'retryable_winerrors': sorted(RETRYABLE_WINERRORS),
              'scope': 'Atomic output replacement only; frozen run.py executes unchanged.'}
    (records / f'pid_{os.getpid()}.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    sys.argv = [str(root / 'run.py'), *arguments]
    runpy.run_path(str(root / 'run.py'), run_name='__main__')


if __name__ == '__main__':
    main()
