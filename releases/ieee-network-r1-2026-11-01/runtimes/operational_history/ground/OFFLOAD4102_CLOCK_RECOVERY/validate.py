"""Bounded controls for a positive physical arrival tail, no audit relaxation."""
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / '.network_ground_parallel_work'
sys.path.insert(0, str(ROOT))
import run
sys.path.insert(0, str(ROOT / 'tests'))
from formal_common import load_checkpoint, read, sha, verify
from src.network_revision.runtime import NetworkExperiment
from test_network_formal6h import scene
from src.link.event_queue_runtime_v3 import ComputeIntervalV3
from exact_arrival import install, identity

started = time.perf_counter()
checks = {}
original = NetworkExperiment.run
plain = scene('OFFLOAD')
plain.run(120., full_scope='formal')
install(ROOT)
patched = scene('OFFLOAD')
patched.run(120., full_scope='formal')
checks['no_trigger_physics_parity'] = plain.physics.identity() == patched.physics.identity()
checks['no_trigger_policy_parity'] = plain.decisions == patched.decisions and plain.candidates == patched.candidates
checks['no_trigger_ground_parity'] = plain.ground_ledger == patched.ground_ledger

target = 1.
e = scene('OFFLOAD')
e.physics.execute(target - 4.547473508864641e-11)
before = e.physics.now
old = e.physics.project(end_s=target)
checks['original_kernel_skips_reproduced_tail'] = old['state'].time_s == before
e._finish_exact_arrival(target)
checks['tail_reaches_exact_clock'] = e.physics.now == target
row = e.physics.trajectory[-1]
checks['actual_physics_interval_retained'] = row['start_s'] == before and row['end_s'] == target and row['thermal']['safe']
checks['no_fictitious_TX'] = not e.physics.flows
checks['background_charged'] = e.physics.ledger()['background_j'] > 0
checks['zero_tail_noop'] = e._finish_exact_arrival(target) is False and len(e.exact_arrival_tail_records) == 1
for name, dest in [('backward_rejected', .999), ('large_gap_rejected', 1.01)]:
    try:
        e._finish_exact_arrival(dest)
        checks[name] = False
    except ValueError:
        checks[name] = True

busy = scene('OFFLOAD')
busy.physics.execute(before)
busy.physics.work.append(ComputeIntervalV3('tail-test', 'synthetic_0', 'inference', before, 2., 10.))
busy._finish_exact_arrival(target)
checks['concurrent_compute_preserved'] = busy.physics.trajectory[-1]['board_incremental_w'] == 10.

unsafe = scene('OFFLOAD')
unsafe.physics.state = replace(unsafe.physics.state, time_s=before,
                               temperatures_c=tuple(100. for _ in unsafe.physics.state.temperatures_c))
try:
    unsafe._finish_exact_arrival(target)
    checks['unsafe_tail_rejected'] = False
except RuntimeError as error:
    checks['unsafe_tail_rejected'] = 'EXECUTION_SAFETY_STOP' in str(error)
checks['tail_override_removed_after_failure'] = 'project' not in unsafe.physics.kernel.__dict__

# Exact source/input/physical state for the real, pre-defect resume point.
oldrun = HERE.parent / 'A_RUNS/4102_OFFLOAD/attempt_02'
ck = oldrun / 'CHECKPOINTS/T04200.pkl'
meta = read(ck.with_suffix('.json'))
checks['prefix_SHA'] = sha(ck) == meta['pickle_sha256']
checks['prefix_manifests'] = meta['identity']['manifests'] == verify()
prefix = load_checkpoint(ck, 'OFFLOAD')
checks['all_prefix_arrivals_exact'] = all(r['time_s'] == prefix.arrived[r['task_id']].arrival_s for r in prefix.arrival_events)
checks['prefix_before_first_defect'] = prefix.physics.now == 4200. < 9406.642356677172
left, right = deepcopy(prefix), deepcopy(prefix)
original(left, 4220.)
right.run(4220.)
checks['real_checkpoint_no_trigger_physics_parity'] = left.physics.identity() == right.physics.identity()
checks['real_checkpoint_no_trigger_policy_parity'] = left.decisions == right.decisions and left.ground_ledger == right.ground_ledger
result = dict(pass_=all(checks.values()), checks=checks, seconds=time.perf_counter()-started,
              execution_overlay=identity(), prefix=str(ck), prefix_sha256=sha(ck))
(HERE / 'VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result['pass_']
