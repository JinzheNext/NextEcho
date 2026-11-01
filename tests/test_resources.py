import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.contracts import PipelineError
from workbench.resources import ExecutionLease, ResourcePolicy, process_alive, process_birth, run_owned


@unittest.skipUnless(os.name == "posix", "POSIX resource control")
class ResourceTests(unittest.TestCase):
    def test_three_processes_are_serialized(self):
        with tempfile.TemporaryDirectory() as d:
            code = '''import sys,time
from pathlib import Path
from workbench.resources import ExecutionLease
with ExecutionLease(Path(sys.argv[1])):
 with open(sys.argv[2],"a") as f: f.write("start\\n")
 time.sleep(.08)
 with open(sys.argv[2],"a") as f: f.write("end\\n")
'''
            path = Path(d) / "events"
            children = [subprocess.Popen([sys.executable, "-c", code, d, str(path)]) for _ in range(3)]
            for p in children:
                self.assertEqual(p.wait(timeout=5), 0)
            self.assertEqual(path.read_text().splitlines(), ["start", "end"] * 3)

    def test_inherited_lock_survives_supervisor_exit(self):
        with tempfile.TemporaryDirectory() as d:
            code = '''import subprocess,sys
from pathlib import Path
from workbench.resources import ExecutionLease
with ExecutionLease(Path(sys.argv[1])) as lock:
 p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(.8)"],pass_fds=(lock.fd,))
 print(p.pid,flush=True)
 import time; time.sleep(10)
'''
            parent = subprocess.Popen([sys.executable, "-c", code, d], stdout=subprocess.PIPE, text=True)
            parent.stdout.readline()
            parent.kill()
            self.assertEqual(parent.wait(timeout=3), -9)
            with self.assertRaises(TimeoutError):
                with ExecutionLease(Path(d), timeout=.1):
                    pass
            time.sleep(.8)
            with ExecutionLease(Path(d), timeout=1):
                pass
            parent.stdout.close()

    def test_resource_preflight_blocks_unknown_and_insufficient(self):
        policy = ResourcePolicy(100, 1000, 1, reserve_bytes=10)
        for sample in ({"available_bytes": None, "pressure": "normal"}, {"available_bytes": 109, "pressure": "normal"}):
            with self.assertRaises(PipelineError):
                policy.preflight(sample)

    def test_worker_timeout_reaps_owned_group(self):
        with tempfile.TemporaryDirectory() as d:
            def sample():
                return {"available_bytes": 10**12, "pressure": "normal", "swap_used_bytes": 0}
            with self.assertRaises(PipelineError) as caught:
                run_owned([sys.executable, "-c", "import time; time.sleep(10)"],
                          log_path=Path(d) / "log", policy=ResourcePolicy(1, 10**10, .15, reserve_bytes=0),
                          sampler=sample, interval=.05)
            self.assertEqual(caught.exception.code, "STAGE_TIMEOUT")
            self.assertTrue((Path(d) / "log.resources.json").exists())

    def test_cancel_reaps_owned_descendant_and_preserves_unrelated_process(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            child_id = root / "child.pid"
            code = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(20)"
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
            try:
                with self.assertRaises(PipelineError) as caught:
                    run_owned([sys.executable, "-c", code, str(child_id)], log_path=root / "asr.log",
                        policy=ResourcePolicy(1, 10**10, 10, reserve_bytes=0), interval=.05,
                        sampler=lambda: {"available_bytes": 10**12, "pressure": "normal", "swap_used_bytes": 0},
                        cancel=child_id.exists)
                self.assertEqual(caught.exception.code, "CANCELLED")
                self.assertIsNone(unrelated.poll())
                pid = child_id.read_text()
                result = subprocess.run(["ps", "-p", pid, "-o", "stat="], capture_output=True, text=True)
                self.assertTrue(not result.stdout.strip() or result.stdout.strip().startswith("Z"))
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=3)

    def test_pid_identity_is_checked(self):
        self.assertTrue(process_alive(os.getpid(), process_birth(os.getpid())))
        self.assertFalse(process_alive(os.getpid(), "different-start"))

    def test_host_pressure_swap_unknown_and_recovery_are_debounced(self):
        good = {"available_bytes": 10**12, "pressure": "normal", "swap_used_bytes": 0}
        faults = [{**good, "pressure": "critical"}, {**good, "swap_used_bytes": 101},
                  {**good, "available_bytes": None}, {**good, "swap_used_bytes": None}]
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as d:
                values = iter([good, fault, good, fault, fault])
                samples = []
                def sample():
                    row = dict(next(values))
                    samples.append(row)
                    return row
                with self.assertRaises(PipelineError) as caught:
                    run_owned([sys.executable, "-c", "import time; time.sleep(10)"], log_path=Path(d) / "model.log",
                        policy=ResourcePolicy(1, 10**10, 5, reserve_bytes=0, max_swap_delta_bytes=100),
                        sampler=sample, interval=.03)
                self.assertEqual(caught.exception.code, "RESOURCE_LIMIT")
                self.assertEqual(len(samples), 5)

    def test_queue_time_does_not_consume_stage_deadline_and_preflight_is_after_wait(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": d}):
            held = subprocess.Popen([sys.executable, "-c",
                "from workbench.resources import ExecutionLease; import time; "
                "lock=ExecutionLease(); lock.__enter__(); print('held',flush=True); time.sleep(.7)"], stdout=subprocess.PIPE, text=True)
            self.assertEqual(held.stdout.readline().strip(), "held")
            started = time.monotonic()
            sampled = []
            def sample():
                sampled.append(time.monotonic() - started)
                return {"available_bytes": 10**12, "pressure": "normal", "swap_used_bytes": 0}
            result = run_owned([sys.executable, "-c", "pass"], log_path=Path(d) / "model.log",
                policy=ResourcePolicy(1, 10**10, .6, reserve_bytes=0), sampler=sample, interval=.03)
            self.assertGreater(sampled[0], .6)
            self.assertLess(result["elapsed_seconds"], .6)
            held.wait(timeout=3)
            held.stdout.close()
