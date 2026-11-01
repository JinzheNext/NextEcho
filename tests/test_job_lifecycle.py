"""Real detached processes, ffmpeg and fake model: cancellation and recovery."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path


@unittest.skipUnless(os.name == "posix" and shutil.which("ffmpeg"), "POSIX + ffmpeg")
class JobLifecycleTests(unittest.TestCase):
    def test_wait_cancel_resume_and_corrupted_stage_in_new_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = root / "model"
            model.mkdir()
            (model / "model.int8.onnx").write_bytes(b"FAKE")
            (model / "tokens.txt").write_text("FAKE")
            (root / "vad.onnx").write_bytes(b"FAKE")
            (root / "mode").write_text("wait")
            worker = root / "fake-python"
            worker.write_text(f"#!{sys.executable}\n" + '''import json,sys,time,subprocess
from pathlib import Path
r=json.loads(Path(sys.argv[2]).read_text()); root=Path(__file__).parent
with (root/'calls').open('a') as f: f.write('called\\n')
if (root/'mode').read_text()=='wait':
 child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
 (root/'child.pid').write_text(str(child.pid)); time.sleep(60)
Path(r['output']).write_text(json.dumps({'schema_version':1,'text':'FAKE BODY',
 'segments':[{'text':'FAKE BODY','start_ms':0,'end_ms':1000}],
 'coverage':{'covered_seconds':1,'partial':False}}))
''')
            worker.chmod(0o755)
            wav = root / "audio.wav"
            with wave.open(str(wav), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(b"\0\0" * 16000)
            config = root / "config.toml"
            config.write_text(f'''[hardware]
reserve_memory_gb=0.0
[asr]
vad_model={json.dumps(str(root/'vad.onnx'))}
[models.sensevoice-int8]
path={json.dumps(str(model))}
worker_python={json.dumps(str(worker))}
estimated_bytes=1
max_rss_bytes=524288000
timeout_seconds=30
resource_sample_seconds=0.05
''')
            env = {**os.environ, "NEXTECHO_RUNTIME_DIR": str(root / "runtime"), "NEXTECHO_CONFIG": str(config)}
            def cli(*args):
                result = subprocess.run([sys.executable, "-m", "workbench.cli", *args, "--output", "json"],
                    capture_output=True, text=True, env=env, timeout=15)
                self.assertNotIn("FAKE BODY", result.stdout)
                self.assertLessEqual(len(result.stdout.encode()), 2048)
                return result.returncode, json.loads(result.stdout)
            _, accepted = cli("run", str(wav), "--profile", "fast", "--postprocess", "off", "--detach")
            job_id = accepted["job_id"]
            try:
                deadline = time.monotonic() + 10
                while not (root / "child.pid").exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue((root / "child.pid").exists())
                self.assertEqual(cli("jobs", "wait", job_id, "--timeout", "0")[0], 7)
                self.assertEqual(cli("jobs", "cancel", job_id)[0], 6)
                self.assertEqual(cli("jobs", "cancel", job_id)[1]["status"], "cancelled")
                child = subprocess.run(["ps", "-p", (root / "child.pid").read_text(), "-o", "stat="], capture_output=True, text=True)
                self.assertTrue(not child.stdout.strip() or child.stdout.strip().startswith("Z"))
                (root / "mode").write_text("complete")
                _, resumed = cli("jobs", "resume", job_id)
                self.assertNotEqual(resumed["job_id"], job_id)
                code, final = cli("jobs", "wait", resumed["job_id"], "--timeout", "10")
                self.assertEqual((code, final["status"]), (0, "ready"), final)
                self.assertEqual(cli("jobs", "status", job_id)[1]["status"], "cancelled")
                run_dir = Path(final["receipt_path"]).parent
                transcript = next(run_dir.glob("items/*/transcript.txt"))
                transcript.write_text("corrupted")
                _, retry = cli("jobs", "resume", final["job_id"])
                code, corrected = cli("jobs", "wait", retry["job_id"], "--timeout", "10")
                self.assertEqual((code, corrected["status"]), (0, "ready"), corrected)
                self.assertEqual(transcript.read_text(), "corrupted")
                self.assertIn("FAKE BODY", Path(corrected["primary_file"]).read_text())
                self.assertEqual(len((root / "calls").read_text().splitlines()), 3)
                from workbench.job_store import JobStore
                record = JobStore(root / "runtime").get(retry["job_id"])
                self.assertEqual(record["parent_job_id"], final["job_id"])
                self.assertGreater(record["attempt"], 1)
            finally:
                # On a failed assertion, keep test workers out of later tests.
                subprocess.run([sys.executable, "-m", "workbench.cli", "jobs", "cancel", job_id, "--output", "json"],
                    capture_output=True, env=env, timeout=10)
