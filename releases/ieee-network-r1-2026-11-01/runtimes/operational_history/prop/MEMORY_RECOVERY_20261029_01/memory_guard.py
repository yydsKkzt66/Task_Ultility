"""Host scheduling guard only; never touches simulator resource thresholds."""
import ctypes
import hashlib
import json
from pathlib import Path

VERSION = 'HOST_COMMIT_GATED_RESUME_20261029_V1'
GIB = 1024 ** 3
MIN_COMMIT_HEADROOM = 4 * GIB
MIN_PHYSICAL_HEADROOM = 3 * GIB

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + [
        (key, ctypes.c_ulonglong) for key in (
            'ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile',
            'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]

def memory_status():
    value = MEMORYSTATUSEX()
    value.dwLength = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('Cannot inspect Windows commit headroom')
    return dict(physical_available_bytes=value.ullAvailPhys,
                commit_available_bytes=value.ullAvailPageFile,
                commit_limit_bytes=value.ullTotalPageFile,
                min_commit_headroom_bytes=MIN_COMMIT_HEADROOM,
                min_physical_headroom_bytes=MIN_PHYSICAL_HEADROOM)

def can_launch(status):
    return (status['commit_available_bytes'] >= MIN_COMMIT_HEADROOM and
            status['physical_available_bytes'] >= MIN_PHYSICAL_HEADROOM)

def approved_memory_resume(attempt):
    approval = json.loads((Path(__file__).parent / 'APPROVED_RESUMES.json').read_text(encoding='utf-8'))
    row = approval['attempts'].get(str(Path(attempt).resolve()))
    if not row:
        return False
    failure = Path(attempt) / 'FAILED.json'
    latest = Path(attempt) / 'LATEST_CHECKPOINT.json'
    return (hashlib.sha256(failure.read_bytes()).hexdigest() == row['failed_sha256'] and
            hashlib.sha256(latest.read_bytes()).hexdigest() == row['latest_pointer_sha256'] and
            json.loads(failure.read_text(encoding='utf-8'))['error'] == 'MemoryError()')
