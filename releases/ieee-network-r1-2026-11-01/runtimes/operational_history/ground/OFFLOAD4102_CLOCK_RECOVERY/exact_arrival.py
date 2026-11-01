"""Explicit arrival-tail execution overlay; frozen source/audit stay unchanged."""
from pathlib import Path
from types import MethodType
import hashlib
import inspect
import textwrap

VERSION = 'OFFLOAD_EXACT_ARRIVAL_TAIL_V1'
EXPECTED = {
    'src/network_revision/runtime.py': '28aafda6b6b35efd4b5cb2a3630f462c706fc23be9c4d70832493ee9fa996fa6',
    'src/link/event_queue_runtime_v3.py': '260d6dfc5f83b06371bf6e4edca88234c6f402c92bfa62b50e7be6444a3cd504',
    'formal_audit.py': '2efca189ac1c4cc84bba50704ab949ac1a8529f5221dfedefd9de76affb37d89',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(root):
    import src.network_revision.runtime as runtime
    import src.link.event_queue_runtime_v3 as kernel_module
    for name, expected in EXPECTED.items():
        if sha(Path(root) / name) != expected:
            raise ValueError('UNREVIEWED_SOURCE:' + name)
    cls = runtime.NetworkExperiment
    if getattr(cls.run, '_exact_arrival_version', None) == VERSION:
        return
    original_run = cls.run
    original_kernel = kernel_module.EventQueueKernelV3._project
    source = textwrap.dedent(inspect.getsource(original_kernel))
    old = 'while state.time_s < end_s-1e-10:'
    assert source.count(old) == 1
    # Only the separate, explicitly requested positive tail uses this entry.
    # All original power integration, whole-bit accounting and safety checks
    # are retained. Ordinary execution and every forecast use frozen code.
    source = source.replace(old, 'while state.time_s < end_s:')
    namespace = {}
    exec(compile(source, str(Path(__file__)) + ':tail_kernel', 'exec'),
         kernel_module.__dict__, namespace)
    tail_kernel = namespace['_project']

    def finish_arrival(e, target):
        p = e.physics
        if p.now == target:
            return False
        before = p.now
        if e.method != 'OFFLOAD' or not 0. < target - before <= 1e-8:
            raise ValueError('UNEXPECTED_ARRIVAL_CLOCK_GAP:' + repr((before, target)))
        if 'project' in p.kernel.__dict__:
            raise ValueError('UNEXPECTED_INSTANCE_PROJECT_OVERRIDE')
        ntraj, nflows, nground = len(p.trajectory), len(p.flows), len(e.ground_ledger)
        # Bypass the old epsilon-cutoff memo for this one execution call.
        # NetworkPhysics.execute remains the sole state/ledger owner.
        p.kernel.project = MethodType(tail_kernel, p.kernel)
        try:
            p.execute(target)
        finally:
            del p.kernel.project
        if p.now != target:
            raise ValueError('ARRIVAL_TAIL_DID_NOT_REACH_EXACT_TARGET')
        e.observe_ground()
        records = e.__dict__.setdefault('exact_arrival_tail_records', [])
        records.append(dict(start_s=before, end_s=target, duration_s=target-before,
                            physical_intervals=len(p.trajectory)-ntraj,
                            TX_intervals=len(p.flows)-nflows,
                            new_ground_events=len(e.ground_ledger)-nground,
                            version=VERSION))
        return len(e.ground_ledger) > nground

    source = textwrap.dedent(inspect.getsource(original_run))
    old = '            register(batch)'
    assert source.count(old) == 1
    source = source.replace(old, '            boundary_ground = self._finish_exact_arrival(when) or boundary_ground\n' + old)
    namespace = {}
    exec(compile(source, str(Path(__file__)) + ':arrival_run', 'exec'),
         runtime.__dict__, namespace)
    patched = namespace['run']
    patched._exact_arrival_version = VERSION
    patched._original_run = original_run
    cls._finish_exact_arrival = finish_arrival
    cls.run = patched


def identity():
    return dict(version=VERSION, sha256=sha(__file__),
                scope='OFFLOAD arrival registration only; original integrator/safety/fees; strict audit unchanged',
                frozen_dependencies=EXPECTED)
