"""Fault injection kills only process identities created by this test."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from workbench.resources import ExecutionLease, ResourcePolicy, group_running, run_owned
from workbench.contracts import PipelineError


@unittest.skipUnless(os.name == "posix", "POSIX ownership")
class ProcessFaultTests(unittest.TestCase):
    def test_guardian_records_rss_trigger_before_outer_sampling_can_see_it(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with __import__('unittest.mock', fromlist=['patch']).patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": d}):
                with self.assertRaises(PipelineError):
                    run_owned([sys.executable, "-c", "import time; a=bytearray(100*1024**2);time.sleep(30)"],
                        log_path=root / "model.log", policy=ResourcePolicy(1, 96*1024**2, 10, reserve_bytes=0),
                        interval=2, sampler=lambda: {"available_bytes": 10**12, "pressure": "normal", "swap_used_bytes": 0})
            evidence = json.loads((root / "model.guard.json").read_text())
            self.assertEqual(evidence["trigger"], "group_rss")
            self.assertGreater(evidence["group_rss_bytes"], evidence["max_rss_bytes"])
            self.assertTrue(evidence["owner_alive"])

    def test_caller_worker_and_guardian_death_cleanup_owned_descendants(self):
        for target in ("caller", "worker", "guardian"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                worker = root / "fake_worker.py"
                worker.write_text('''import os,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]); (root/'worker.pid').write_text(str(os.getpid()))
child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'],stdout=subprocess.PIPE,text=True)
child.stdout.readline(); (root/'descendant.pid').write_text(str(child.pid)); time.sleep(60)
''')
                caller_code = '''import json,sys
from pathlib import Path
from workbench.resources import ResourcePolicy,run_owned
root=Path(sys.argv[1])
try:
 run_owned([sys.executable,str(root/'fake_worker.py'),str(root)],log_path=root/'model.log',
  policy=ResourcePolicy(1,10**10,20,reserve_bytes=0),interval=.03,
  sampler=lambda:{'available_bytes':10**12,'pressure':'normal','swap_used_bytes':0},
  on_spawn=lambda pid,birth:(root/'guardian.pid').write_text(str(pid)))
except Exception as e:(root/'error.json').write_text(json.dumps({'code':getattr(e,'code',type(e).__name__)}))
'''
                caller = subprocess.Popen([sys.executable, "-c", caller_code, d],
                    env={**os.environ, "NEXTECHO_RUNTIME_DIR": d})
                try:
                    deadline = time.monotonic() + 8
                    while not (root / "descendant.pid").is_file() and time.monotonic() < deadline:
                        time.sleep(.03)
                    self.assertTrue((root / "descendant.pid").exists())
                    pid = caller.pid if target == "caller" else int((root / f"{target}.pid").read_text())
                    os.kill(pid, signal.SIGKILL)
                    caller.wait(timeout=10)
                    pgid = int((root / "guardian.pid").read_text())
                    deadline = time.monotonic() + 6
                    while group_running(pgid) and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertFalse(group_running(pgid), target)
                    with ExecutionLease(root, timeout=1):
                        pass
                    if target != "caller":
                        self.assertIn(json.loads((root / "error.json").read_text())["code"], {"WORKER_FAILED", "RESOURCE_LIMIT"})
                finally:
                    if caller.poll() is None:
                        caller.kill()
                        caller.wait(timeout=3)
