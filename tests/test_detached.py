"""Protocol integration with real processes/ffmpeg and a FAKE ASR model."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


@unittest.skipUnless(os.name == "posix" and shutil.which("ffmpeg"), "POSIX + ffmpeg integration")
class DetachedIntegrationTests(unittest.TestCase):
    def test_detached_job_and_compact_wait_without_web(self):
        with tempfile.TemporaryDirectory(prefix="NextEcho 中文 ") as d:
            root = Path(d)
            model = root / "model"
            model.mkdir()
            (model / "model.int8.onnx").write_bytes(b"FAKE MODEL: protocol test only")
            (model / "tokens.txt").write_text("test")
            vad = root / "vad.onnx"
            vad.write_bytes(b"FAKE VAD")
            fake = root / "fake-python"
            fake.write_text(f"#!{sys.executable}\n" + '''import sys,json
from pathlib import Path
r=json.loads(Path(sys.argv[2]).read_text())
Path(r['output']).write_text(json.dumps({'schema_version':1,'text':'FAKE ASR TEST TRANSCRIPT',
 'segments':[{'text':'FAKE ASR TEST TRANSCRIPT','start_ms':0,'end_ms':1000}],
 'runtime_versions':{'fake':'1'},'coverage':{'covered_seconds':1,'partial':False}}))
''')
            fake.chmod(0o755)
            wav = root / "输入.wav"
            with wave.open(str(wav), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(b"\0\0" * 16000)
            config = root / "config.toml"
            config.write_text(f'''[hardware]
reserve_memory_gb = 0.0
[asr]
vad_model = {json.dumps(str(vad))}
[models.sensevoice-int8]
path = {json.dumps(str(model))}
worker_python = {json.dumps(str(fake))}
estimated_bytes = 1024
max_rss_bytes = 524288000
timeout_seconds = 10
''')
            env = {**os.environ, "NEXTECHO_RUNTIME_DIR": str(root / "runtime"), "NEXTECHO_CONFIG": str(config)}
            submitted = subprocess.run([sys.executable, "-m", "workbench.cli", "run", str(wav),
                "--profile", "fast", "--postprocess", "off", "--detach", "--output", "json"],
                capture_output=True, text=True, env=env, timeout=10)
            self.assertEqual(submitted.returncode, 0, submitted.stderr + submitted.stdout)
            receipt = json.loads(submitted.stdout)
            self.assertTrue(receipt["job_id"])
            waited = subprocess.run([sys.executable, "-m", "workbench.cli", "jobs", "wait", receipt["job_id"],
                "--timeout", "10", "--output", "json"], capture_output=True, text=True, env=env, timeout=15)
            self.assertEqual(waited.returncode, 0, waited.stderr + waited.stdout)
            final = json.loads(waited.stdout)
            self.assertEqual(final["status"], "ready")
            self.assertIn("FAKE ASR TEST TRANSCRIPT", Path(final["primary_file"]).read_text())
            self.assertNotIn("FAKE ASR TEST TRANSCRIPT", waited.stdout)
            self.assertLessEqual(len(waited.stdout.encode()), 2048)
