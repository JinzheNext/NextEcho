import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix", "POSIX script execution")
class SkillScriptTests(unittest.TestCase):
    def test_binding_and_only_path_output_from_another_directory(self):
        self.check_runtime(False)

    def test_invalid_runtime_json_is_not_success(self):
        self.check_runtime(True)

    def test_unknown_runtime_status_is_not_success(self):
        self.check_runtime("unknown")

    def test_published_speaker_refinement_uses_its_own_cli_route_and_returns_review_paths(self):
        with tempfile.TemporaryDirectory(prefix="Speaker refinement ") as d:
            root = Path(d)
            skill = root / "Copied Skill"
            shutil.copytree(Path(__file__).resolve().parents[1] / "skills/nextecho", skill)
            transcript = root / "speaker.md"
            review = root / "review.html"
            primary = root / "result.md"
            receipt = root / "receipt.json"
            for path in (transcript, review, primary, receipt):
                path.write_text("artifact")
            payload = {
                "schema_version": 1, "artifact_kind": "speaker-refinement", "job_id": "speaker-job",
                "status": "partial", "primary_file": str(primary), "receipt_path": str(receipt),
                "speaker_markdown_paths": [str(transcript)], "speaker_identity_review_paths": [str(review)],
            }
            fake = root / "fake-runtime"
            fake.write_text(
                f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
                "if '--version' in sys.argv: print('nextecho 0.2.5 protocol=1')\n"
                f"elif 'lookup' in sys.argv: print({json.dumps(json.dumps({'schema_version': 1, 'status': 'failed', 'job_id': '', 'primary_file': '', 'receipt_path': '', 'error_code': 'REQUEST_NOT_FOUND'}))}); sys.exit(3)\n"
                "else:\n"
                f"  Path({str(root / 'invocation.json')!r}).write_text(json.dumps(sys.argv))\n"
                f"  print({json.dumps(json.dumps(payload))})\n  sys.exit(4)\n"
            )
            fake.chmod(0o755)
            binding = root / "binding.json"
            bound = subprocess.run([sys.executable, str(skill / "scripts/bind_runtime.py"), "--python", str(fake),
                                    "--binding", str(binding)], cwd=root, capture_output=True, text=True, timeout=10)
            self.assertEqual(bound.returncode, 0, bound.stderr)
            request = subprocess.run([sys.executable, str(skill / "scripts/run.py"), "--binding", str(binding),
                                      "--create-request", "refine-speakers", str(root / "published-run"),
                                      "--backend", "nemotron-metal-candidate"],
                                     cwd=root, capture_output=True, text=True, timeout=10)
            self.assertEqual(request.returncode, 0, request.stderr)
            request_data = json.loads(request.stdout)
            completed = subprocess.run([sys.executable, str(skill / "scripts/run.py"), "--binding", str(binding),
                                         "--request-file", request_data["request_file"]], cwd=root,
                                        capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 4, completed.stderr)
            self.assertEqual(completed.stdout.splitlines(), [str(transcript), str(review)])
            invocation = json.loads((root / "invocation.json").read_text())
            self.assertEqual(invocation[invocation.index("refine-speakers") + 1], str(root / "published-run"))
            self.assertNotIn("run", invocation)
            self.assertNotIn("--speaker-mode", invocation)
            self.assertTrue(any(arg.startswith("intent-") for arg in invocation))

    def check_runtime(self, invalid):
        with tempfile.TemporaryDirectory(prefix="Skill 中文 ") as d:
            root = Path(d)
            source = Path(__file__).resolve().parents[1] / "skills/nextecho"
            skill = root / "Copied Skill"
            shutil.copytree(source, skill)
            result = root / "整理稿.md"
            result.write_text("This content must not enter stdout")
            receipt = root / "receipt.json"
            receipt.write_text("{}")
            payload = {"schema_version": 1, "job_id": "test", "status": "ready", "primary_file": str(result), "receipt_path": str(receipt)}
            fake = root / "fake runtime"
            if invalid == "unknown":
                payload["status"] = "invented-success"
            output = "NOT JSON" if invalid is True else json.dumps(payload)
            fake.write_text(f"#!{sys.executable}\nimport sys,json\nfrom pathlib import Path\nif '--version' in sys.argv: print('nextecho 0.2.5 protocol=1')\nelse:\n Path({str(root / 'invocation.json')!r}).write_text(json.dumps(sys.argv))\n print({output!r})\n")
            fake.chmod(0o755)
            binding = root / "binding.json"
            bound = subprocess.run([sys.executable, str(skill / "scripts/bind_runtime.py"), "--python", str(fake), "--binding", str(binding)], cwd=root, capture_output=True, text=True, timeout=10)
            self.assertEqual(bound.returncode, 0, bound.stderr)
            invoked = subprocess.run([sys.executable, str(skill / "scripts/run.py"), "--binding", str(binding), "https://example.org/media.mp3"], cwd=root, capture_output=True, text=True, timeout=10)
            if invalid:
                self.assertEqual(invoked.returncode, 3)
                self.assertEqual(invoked.stdout, "")
            else:
                self.assertEqual(invoked.returncode, 0, invoked.stderr)
                self.assertEqual(invoked.stdout, str(result) + "\n")
            self.assertEqual(binding.stat().st_mode & 0o777, 0o600)
            recorded = json.loads((root / "invocation.json").read_text())
            self.assertEqual(recorded[recorded.index("--speaker-mode") + 1], "auto")
            overridden = subprocess.run([sys.executable, str(skill / "scripts/run.py"), "--binding", str(binding),
                                        "https://example.org/media.mp3", "--speaker-mode=off"],
                                       cwd=root, capture_output=True, text=True, timeout=10)
            recorded = json.loads((root / "invocation.json").read_text())
            self.assertNotIn("--speaker-mode", recorded)
            self.assertIn("--speaker-mode=off", recorded)
            self.assertEqual(overridden.returncode, invoked.returncode)
