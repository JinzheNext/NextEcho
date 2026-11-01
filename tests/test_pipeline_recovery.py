import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.cli import main
from workbench.config import load_config
from workbench.contracts import PipelineRequest, plan_request
from workbench.delivery import assemble, write_receipt
from workbench.io_utils import atomic_write, redact_structure, write_json
from workbench.job_store import JobStore
from workbench.transcription import transcribe_media_sources


class PipelineRecoveryTests(unittest.TestCase):
    def test_resume_preserves_raw_json_and_spoken_paths_while_relocating_artifact_links(self):
        from workbench.pipeline import execute_plan
        from workbench.contracts import PipelineError
        from workbench.io_utils import file_hash
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            source = root / "source.wav"
            source.write_bytes(b"FAKE audio")
            plan = plan_request(PipelineRequest((str(source),), postprocess="off"), load_config())
            old, new = root / "old", root / "new"
            item = old / "items/001_source"
            item.mkdir(parents=True)
            new.mkdir()
            spoken = str(item / "spoken-file.txt")
            raw = {"params": {"file": str(item / "audio.wav")}, "transcription": [{"text": spoken}]}
            write_json(item / "transcript.json", raw)
            write_json(item / "metadata.json", {"output_dir": str(item), "json_path": str(item / "transcript.json"), "note": spoken,
                                               "candidate_origin_path": str(item / "postprocess/polish/candidates/original.json")})
            checkpoint = item / "postprocess/polish/chunk.json"
            write_json(checkpoint, {"segments": [{"text": spoken}]})
            digest = file_hash(item / "transcript.json")
            def inspect(*args, **kwargs):
                copied = new / "items/001_source"
                self.assertEqual(file_hash(copied / "transcript.json"), digest)
                self.assertEqual(json.loads((copied / "postprocess/polish/chunk.json").read_text())["segments"][0]["text"], spoken)
                metadata = json.loads((copied / "metadata.json").read_text())
                self.assertEqual(metadata["output_dir"], str(copied))
                self.assertEqual(metadata["json_path"], str(copied / "transcript.json"))
                self.assertEqual(metadata["note"], spoken)
                self.assertEqual(metadata["candidate_origin_path"], str(item / "postprocess/polish/candidates/original.json"))
                raise PipelineError("WORKER_FAILED", "stop after cache inspection")
            with patch("workbench.pipeline.transcribe_media_sources", side_effect=inspect):
                with self.assertRaises(PipelineError):
                    execute_plan(plan, new, job_id="test", resume_from=str(old))

    def plan(self, source="https://example.org/audio?token=private"):
        return plan_request(PipelineRequest((source,), postprocess="off"), load_config()).to_dict()

    def test_first_speaker_output_is_available_to_polish_in_legacy_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            item = root / "items/one"
            item.mkdir(parents=True)
            write_json(item / "transcript.json", {"transcription": []})
            payload = {"manifest_path": root / "manifest.json", "results": [{"status": "ready", "output_dir": str(item)}]}
            write_json(payload["manifest_path"], {"results": payload["results"]})
            order = []
            def speaker(_):
                order.append("speaker")
                write_json(item / "transcript.speakers.json", {"speakers": []})
                return {"status": "ready"}
            def polish(*args, **kwargs):
                self.assertTrue((item / "transcript.speakers.json").is_file())
                order.append("polish")
                return {"status": "ready"}
            with patch("workbench.transcription._transcribe_media_sources", return_value=payload), patch("workbench.postprocess.process_transcript_artifacts", side_effect=polish):
                transcribe_media_sources([], output_dir=root, postprocess_profile="quality", speaker_stage=speaker)
            self.assertEqual(order, ["speaker", "polish"])

    def test_all_failed_job_receipt_preserves_root_error_and_action(self):
        import os
        from workbench.job_worker import run_job
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}):
            store = JobStore()
            job, _ = store.submit(self.plan("https://example.org/test.wav"))
            def failed(*args, output_dir, **kwargs):
                results = [{"status": "failed", "error_code": "RESOURCE_LIMIT", "error": "test resource boundary"}]
                write_json(output_dir / "manifest.json", {"results": results})
                return {"results": results}
            with patch("workbench.pipeline.transcribe_media_sources", side_effect=failed):
                run_job(store, job["job_id"])
            final = store.get(job["job_id"])
            self.assertEqual(final["status"], "failed")
            self.assertEqual(final["error_code"], "RESOURCE_LIMIT")
            self.assertIn("资源曲线", final["next_action"])

    def test_speaker_off_auto_and_required_share_capability_policy(self):
        from workbench.pipeline import run_post_asr_stages
        from workbench.contracts import PipelineError
        for mode in ("off", "auto", "required"):
            payload = {"results": [{"status": "ready", "char_count": 8, "asr": {"actual_engine": "sensevoice", "timing_precision": "vad-segment"}}]}
            with patch("builtins.print") as speaker:
                if mode == "required":
                    with self.assertRaises(PipelineError):
                        run_post_asr_stages(payload, speaker=speaker, speaker_mode=mode)
                else:
                    result = run_post_asr_stages(payload, speaker=speaker, speaker_mode=mode)
                    self.assertEqual(result["status"] if result else None, "partial" if mode == "auto" else None)
                speaker.assert_not_called()

    def test_stale_cas_cannot_publish_an_interrupt_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            store = JobStore(Path(d))
            job, _ = store.submit(self.plan())
            newest = store.update(job["job_id"], stage="asr")
            actual = store.update(job["job_id"], expected_revision=job["revision"], receipt_factory=lambda: self.fail("stale CAS published a receipt"))
            self.assertEqual(actual["revision"], newest["revision"])
            self.assertEqual(actual["stage"], "asr")

    def test_live_owner_is_never_replaced_by_heartbeat_age(self):
        import os
        from workbench.resources import process_birth
        with tempfile.TemporaryDirectory() as d:
            store = JobStore(Path(d))
            with patch("workbench.job_store.time.time", return_value=1):
                job, _ = store.submit(self.plan())
                store.update(job["job_id"], status="running", owner_pid=os.getpid(), owner_birth=process_birth(os.getpid()))
            self.assertEqual(store.get(job["job_id"])["status"], "running")

    def test_long_paths_fail_before_filesystem_access_or_inference(self):
        from workbench.contracts import PipelineError
        with tempfile.TemporaryDirectory() as d:
            store = JobStore(Path(d) / "runtime")
            with self.assertRaises(PipelineError) as caught:
                store.submit(self.plan(), output_dir=Path(d).joinpath(*(["中文" * 30] * 20)))
            self.assertEqual(caught.exception.code, "OUTPUT_PATH_TOO_LONG")

    def test_receipt_reconciliation_checks_all_files_and_preserves_private_plan(self):
        for damage in (False, True):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as d:
                store = JobStore(Path(d) / "runtime")
                with patch("workbench.job_store.time.time", return_value=1):
                    job, _ = store.submit(self.plan())
                run = Path(job["run_dir"])
                text = run / "transcript.txt"
                text.write_text("全文")
                results = [{"status": "ready", "text_path": str(text)}]
                write_json(run / "manifest.json", {"results": results})
                assemble(run, results)
                write_receipt(run, job_id=job["job_id"], status="ready", results=results, plan=job["plan"])
                if damage:
                    text.unlink()
                final = store.get(job["job_id"])
                self.assertEqual(final["status"], "failed" if damage else "ready")
                self.assertIn("token=private", final["plan"]["request"]["sources"][0])

    def test_atomic_write_interruption_keeps_last_complete_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "result.md"
            path.write_text("complete")
            with patch("workbench.io_utils.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    atomic_write(path, "partial")
            self.assertEqual(path.read_text(), "complete")
            self.assertEqual(list(path.parent.glob(".pending-*")), [])

    def test_error_redaction_and_bounded_argument_protocol(self):
        secret = {"authorization": "Bearer first-secret", "message": "Authorization: Bearer second-secret\nCookie: session=third-secret; user=private\nhttps://host/audio?sig=fourth-secret", "input": "https://name:password@host/file?token=fifth-secret"}
        serialized = json.dumps(redact_structure(secret))
        for value in ("first-secret", "second-secret", "third-secret", "fourth-secret", "fifth-secret", "password@"):
            self.assertNotIn(value, serialized)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = main(["run", "audio.wav", "--profile", "invalid", "--output", "json"])
        self.assertEqual(status, 2)
        result = json.loads(out.getvalue())
        self.assertEqual(result["error_code"], "INVALID_INPUT")
        self.assertLessEqual(len(out.getvalue().encode()), 2048)
