"""Three actual entry processes, using fake model data and real normalization."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


@unittest.skipUnless(os.name == "posix" and shutil.which("ffmpeg"), "POSIX and ffmpeg")
class EntryConcurrencyTests(unittest.TestCase):
    def test_web_and_two_cli_entries_share_the_model_lease(self):
        with tempfile.TemporaryDirectory(prefix="three entries 中文 ") as d:
            root = Path(d)
            model = root / "model"
            model.mkdir()
            (model / "model.int8.onnx").write_bytes(b"FAKE")
            (model / "tokens.txt").write_text("fake")
            vad = root / "vad.onnx"
            vad.write_bytes(b"FAKE")
            events = root / "events.txt"
            fake = root / "fake-python"
            fake.write_text(f"#!{sys.executable}\n" + f'''import sys,json,time
from pathlib import Path
r=json.loads(Path(sys.argv[2]).read_text())
with open({str(events)!r},'a') as f:f.write('start\\n')
time.sleep(.15)
Path(r['output']).write_text(json.dumps({{'schema_version':1,'text':'FAKE TEST','segments':[{{'text':'FAKE TEST','start_ms':0,'end_ms':1000}}],'coverage':{{'covered_seconds':1,'partial':False}}}}))
with open({str(events)!r},'a') as f:f.write('end\\n')
''')
            fake.chmod(0o755)
            audio = root / "audio.wav"
            with wave.open(str(audio), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(b"\0\0" * 16000)
            config = root / "config.toml"
            config.write_text(f'''[hardware]
reserve_memory_gb=0
[asr]
engine="sensevoice"
vad_model={json.dumps(str(vad))}
[postprocess]
profile="off"
[models.sensevoice-int8]
path={json.dumps(str(model))}
worker_python={json.dumps(str(fake))}
estimated_bytes=1024
max_rss_bytes=524288000
timeout_seconds=10
''')
            env = {**os.environ, "NEXTECHO_RUNTIME_DIR": str(root / "runtime"), "NEXTECHO_CONFIG": str(config)}
            web_code = '''import sys
from pathlib import Path
import app
app.JOBS['test']=app._new_job('test','transcription')
app.run_transcription_job('test',[sys.argv[1]],Path(sys.argv[2]),'base','zh',0,'',False,'',asr_engine='sensevoice',speaker_mode='off')
job=app.JOBS['test']
if job['status']!='ready': raise RuntimeError(str(job.get('error') or job.get('message')))
'''
            commands = [[sys.executable, "-c", web_code, str(audio), str(root / "web")]]
            commands += [[sys.executable, "-m", "workbench.cli", "transcribe", str(audio), "--asr-engine", "sensevoice",
                          "--speaker-mode", "off", "--postprocess-profile", "off", "--output-dir", str(root / f"cli{i}"), "--json"] for i in range(2)]
            children = [subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for cmd in commands]
            try:
                for child in children:
                    stdout, stderr = child.communicate(timeout=30)
                    self.assertEqual(child.returncode, 0, stderr + stdout[-1000:])
                self.assertEqual(events.read_text().splitlines(), ["start", "end"] * 3)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=5)
