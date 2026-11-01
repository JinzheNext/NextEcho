"""Actual copied Skill processes, response loss, frozen cwd/binding and retry."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix", "POSIX script runtime")
class SkillIntentTests(unittest.TestCase):
    def fixture(self, root):
        skill = root / "Copied Skill"
        shutil.copytree(Path(__file__).resolve().parents[1] / "skills/nextecho", skill)
        result = root / "readable.md"
        result.write_text("PRIVATE FULL BODY MUST NOT ENTER STDOUT")
        runtime = root / "runtime-python"
        runtime.write_text(f"#!{sys.executable}\n" + '''import json,sys,os
from pathlib import Path
root=Path(__file__).parent
if '--version' in sys.argv:
 print('nextecho 0.2.5 protocol=1');sys.exit(0)
args=sys.argv[1:]
jobs_path=root/'fake-server-jobs.json'
jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
def emit(data,code=0):
 print(json.dumps(data));sys.exit(code)
if 'lookup' in args:
 key=args[args.index('lookup')+1]
 if key not in jobs:emit({'schema_version':1,'status':'failed','job_id':'','primary_file':'','receipt_path':'','error_code':'REQUEST_NOT_FOUND'},3)
 emit(jobs[key])
key=args[args.index('--idempotency-key')+1]
with (root/'dispatches.jsonl').open('a') as f:f.write(json.dumps({'args':args,'cwd':os.getcwd()})+'\\n')
data={'schema_version':1,'status':'ready','job_id':'job-'+str(len(jobs)+1),'primary_file':str(root/'readable.md'),'receipt_path':str(root/'receipt.json')}
jobs[key]=data;jobs_path.write_text(json.dumps(jobs))
if (root/'lose-next-response').exists():
 (root/'lose-next-response').unlink();print('LOST OR INCOMPLETE RESPONSE');sys.exit(0)
emit(data)
''')
        runtime.chmod(0o755)
        binding = root / "binding.json"
        binding.write_text(json.dumps({"schema_version": 1, "python": str(runtime), "config": "", "runtime_dir": ""}))
        return skill / "scripts/run.py", binding

    def invoke(self, script, binding, cwd, *args):
        result = subprocess.run([sys.executable, str(script), "--binding", str(binding), *args],
                                cwd=cwd, text=True, capture_output=True, timeout=10)
        self.assertLessEqual(len(result.stdout.encode()), 2048)
        self.assertNotIn("PRIVATE FULL BODY", result.stdout)
        return result

    def prepare(self, script, binding, cwd, *args):
        result = self.invoke(script, binding, cwd, "--create-request", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "request_prepared")
        self.assertEqual(Path(data["request_file"]).stat().st_mode & 0o777, 0o600)
        return data

    def test_lost_response_uses_same_intent_and_lookup_never_resubmits(self):
        with tempfile.TemporaryDirectory(prefix="Intent 中文 ") as d:
            root = Path(d).resolve()
            script, binding = self.fixture(root)
            prepared = self.prepare(script, binding, root, "https://example.org/feed.xml", "--profile", "fast")
            request = Path(prepared["request_file"])
            original = request.read_bytes()
            self.assertFalse((root / "dispatches.jsonl").exists())
            (root / "lose-next-response").touch()
            first = self.invoke(script, binding, root, "--request-file", str(request))
            self.assertEqual(first.returncode, 3)
            self.assertEqual(first.stdout, "")
            self.assertIn(str(request), first.stderr)
            # A changed binding/config must not redirect an accepted intent.
            binding.write_text(json.dumps({"schema_version": 1, "python": "/must-not-be-used", "config": "/changed-config"}))
            other = root / "another cwd"
            other.mkdir()
            retry = self.invoke(script, binding, other, "--request-file", str(request))
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(retry.stdout.strip(), str(root / "readable.md"))
            self.assertEqual(len((root / "dispatches.jsonl").read_text().splitlines()), 1)
            self.assertEqual(request.read_bytes(), original)

    def test_new_user_intent_same_url_gets_distinct_uuid_and_new_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            script, binding = self.fixture(root)
            first = self.prepare(script, binding, root, "https://example.org/feed.xml")
            second = self.prepare(script, binding, root, "https://example.org/feed.xml")
            self.assertNotEqual(first["request_id"], second["request_id"])
            for data in (first, second):
                actual = self.invoke(script, binding, root, "--request-file", data["request_file"])
                self.assertEqual(actual.returncode, 0, actual.stderr)
            self.assertEqual(len((root / "dispatches.jsonl").read_text().splitlines()), 2)

    def test_frozen_cwd_preserves_relative_media_and_lookup_has_no_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            script, binding = self.fixture(root)
            here, there = root / "caller", root / "later"
            here.mkdir()
            there.mkdir()
            (here / "clip.wav").write_bytes(b"fixture")
            data = self.prepare(script, binding, here, "clip.wav", "--postprocess", "off")
            actual = self.invoke(script, binding, there, "--request-file", data["request_file"])
            self.assertEqual(actual.returncode, 0, actual.stderr)
            call = json.loads((root / "dispatches.jsonl").read_text())
            self.assertEqual(call["cwd"], str(here))
            self.assertIn("clip.wav", call["args"])
            self.assertIn("intent-" + data["request_id"], call["args"])
            read = self.invoke(script, binding, there, "jobs", "lookup", "intent-" + data["request_id"], "--output", "json")
            self.assertEqual(read.returncode, 0, read.stderr)
            self.assertEqual(json.loads(read.stdout)["status"], "ready")
            self.assertEqual(len((root / "dispatches.jsonl").read_text().splitlines()), 1)

    def test_tampered_or_amended_intent_cannot_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            script, binding = self.fixture(root)
            data = self.prepare(script, binding, root, "https://example.org/media.mp3")
            amended = self.invoke(script, binding, root, "--request-file", data["request_file"], "--speaker-mode", "off")
            self.assertEqual(amended.returncode, 3)
            path = Path(data["request_file"])
            saved = json.loads(path.read_text())
            saved["arguments"].append("--allow-fallback")
            path.write_text(json.dumps(saved))
            tampered = self.invoke(script, binding, root, "--request-file", str(path))
            self.assertEqual(tampered.returncode, 3)
            self.assertEqual(tampered.stdout, "")
            self.assertFalse((root / "dispatches.jsonl").exists())

    def test_explicit_intent_file_never_overwrites_previous_intent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            script, binding = self.fixture(root)
            path = root / "single-request.json"
            created = self.invoke(script, binding, root, "--create-request", "--request-file", str(path), "https://example.org/media.mp3")
            self.assertEqual(created.returncode, 0, created.stderr)
            original = path.read_bytes()
            duplicate = self.invoke(script, binding, root, "--create-request", "--request-file", str(path), "https://example.org/different.mp3")
            self.assertEqual(duplicate.returncode, 3)
            self.assertEqual(path.read_bytes(), original)

    def test_resume_is_also_a_single_persistent_mutating_intent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            script, binding = self.fixture(root)
            data = self.prepare(script, binding, root, "jobs", "resume", "parent", "--mode", "refresh")
            for _ in range(2):
                actual = self.invoke(script, binding, root, "--request-file", data["request_file"])
                self.assertEqual(actual.returncode, 0, actual.stderr)
            self.assertEqual(len((root / "dispatches.jsonl").read_text().splitlines()), 1)
            call = json.loads((root / "dispatches.jsonl").read_text())
            self.assertIn("resume", call["args"])
            self.assertIn("refresh", call["args"])

    def test_old_runtime_cannot_bind_new_skill(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            script, binding = self.fixture(root)
            old = root / "old-python"
            old.write_text(f"#!{sys.executable}\nprint('nextecho 0.2.4 protocol=1')\n")
            old.chmod(0o755)
            result = subprocess.run([sys.executable, str(script.with_name("bind_runtime.py")), "--python", str(old),
                "--binding", str(root / "must-not-exist.json")], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((root / "must-not-exist.json").exists())


if __name__ == "__main__":
    unittest.main()
