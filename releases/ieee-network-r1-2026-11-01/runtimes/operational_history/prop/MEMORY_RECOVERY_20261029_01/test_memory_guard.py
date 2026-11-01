import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import memory_guard as guard

class HostMemoryGuardTests(unittest.TestCase):
    def test_free_ram_does_not_override_low_commit(self):
        self.assertFalse(guard.can_launch(dict(physical_available_bytes=12*guard.GIB, commit_available_bytes=guard.GIB)))

    def test_physical_floor(self):
        self.assertFalse(guard.can_launch(dict(physical_available_bytes=2*guard.GIB, commit_available_bytes=10*guard.GIB)))

    def test_declared_launch_boundary(self):
        self.assertTrue(guard.can_launch(dict(physical_available_bytes=3*guard.GIB, commit_available_bytes=4*guard.GIB)))
        self.assertFalse(guard.can_launch(dict(physical_available_bytes=3*guard.GIB, commit_available_bytes=4*guard.GIB-1)))

    def test_live_windows_counter(self):
        s=guard.memory_status()
        self.assertGreater(s['commit_limit_bytes'],0)
        self.assertGreaterEqual(s['commit_available_bytes'],0)
        self.assertLessEqual(s['commit_available_bytes'],s['commit_limit_bytes'])

    def test_only_exact_reviewed_failure_is_resumable(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp);a=folder/'attempt_01';a.mkdir()
            failure=a/'FAILED.json';pointer=a/'LATEST_CHECKPOINT.json'
            failure.write_text(json.dumps({'error':'MemoryError()'}),encoding='utf-8')
            pointer.write_text(json.dumps({'path':'CHECKPOINTS/T01000.pkl'}),encoding='utf-8')
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            approved={'attempts':{str(a.resolve()):{'failed_sha256':digest(failure),'latest_pointer_sha256':digest(pointer)}}}
            (folder/'APPROVED_RESUMES.json').write_text(json.dumps(approved),encoding='utf-8')
            with patch.object(guard,'__file__',str(folder/'memory_guard.py')):
                self.assertTrue(guard.approved_memory_resume(a))
                self.assertFalse(guard.approved_memory_resume(folder/'unreviewed_attempt'))
                failure.write_text(json.dumps({'error':'ValueError()'}),encoding='utf-8')
                self.assertFalse(guard.approved_memory_resume(a))
                failure.write_text(json.dumps({'error':'MemoryError()'}),encoding='utf-8')
                pointer.write_text(json.dumps({'path':'CHECKPOINTS/other.pkl'}),encoding='utf-8')
                self.assertFalse(guard.approved_memory_resume(a))

if __name__ == '__main__':
    unittest.main(verbosity=2)
