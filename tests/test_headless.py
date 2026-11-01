from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.asr import TranscriptResult
from workbench.audio_segments import merge_recognized_segments, split_window
from workbench.cli import build_parser
from workbench.config import load_config
from workbench.contracts import PipelineError, PipelineRequest, plan_request
from workbench.delivery import assemble, compact, write_receipt
from workbench.headless_cli import emit, wait_for
from workbench.io_utils import write_json
from workbench.job_store import JobStore
from workbench.job_worker import run_job
from workbench.local_asr import validate_worker_result
from workbench.sources import resolve_sources
from workbench.transcription import transcribe_single_source, write_qwen_transcript_result


class HeadlessTests(unittest.TestCase):
    def test_new_command_parser_preserves_intent(self):
        a = build_parser().parse_args(["run", "example.wav", "--profile=quality", "--speaker-mode=off", "--postprocess=off"])
        self.assertEqual((a.profile, a.speaker_mode, a.postprocess), ("quality", "off", "off"))
        self.assertFalse(a.detach)
        self.assertEqual(a.output, "paths")

    def test_asr_without_measured_times_cannot_create_fake_subtitles(self):
        with tempfile.TemporaryDirectory() as d:
            r = TranscriptResult("没有时间戳", "zh", [], "nano", "model")
            with self.assertRaises(ValueError):
                write_qwen_transcript_result(r, output_dir=Path(d), output_base="transcript", output_formats=["txt", "json", "srt"])
            self.assertFalse((Path(d) / "transcript.srt").exists())

    def test_measured_segments_and_silence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            r = TranscriptResult("句子", "zh", [], "sensevoice", "model", segments=[{"text": "句子", "start_ms": 1200, "end_ms": 2500}], timing_precision="vad-segment")
            write_qwen_transcript_result(r, output_dir=root, output_base="transcript", output_formats=["txt", "json", "srt"])
            self.assertIn("00:00:01,200 --> 00:00:02,500", (root / "transcript.srt").read_text())
            empty = TranscriptResult("", "zh", [], "sensevoice", "model", coverage={"content_state": "no_speech"})
            write_qwen_transcript_result(empty, output_dir=root, output_base="silent", output_formats=["txt", "json", "srt"])
            self.assertFalse((root / "silent.srt").exists())

    def test_windows_keep_tail_and_do_not_delete_nonoverlapping_repetition(self):
        self.assertEqual([(w.start_sample, w.end_sample) for w in split_window(0, 61 * 16000)], [(0, 480000), (480000, 960000), (960000, 976000)])
        rows, review = merge_recognized_segments([{"text": "非常重要", "start_ms": 0, "end_ms": 1000}, {"text": "非常重要", "start_ms": 1000, "end_ms": 2000}])
        self.assertEqual(len(rows), 2)
        self.assertEqual(review, [])

    def test_changed_model_cannot_reuse_old_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.wav"
            source.write_bytes(b"source")
            descriptor = resolve_sources([str(source)])[0]
            target = root / "items" / descriptor.source_label
            target.mkdir(parents=True)
            (target / "transcript.txt").write_text("旧稿")
            write_json(target / "metadata.json", {"asr": {"actual_model": "old-model"}})
            with self.assertRaises(PipelineError) as c:
                transcribe_single_source(descriptor=descriptor, items_dir=root / "items", model_path=None,
                    whisper_bin=None, ffmpeg_bin="ffmpeg", language="zh", max_seconds=0, output_formats=["txt"],
                    glossary_text="", enable_enhancement=False, ai_polish=False, ai_style="", index=1,
                    asr_engine="qwen3-asr", asr_model="new-model")
            self.assertEqual(c.exception.code, "CACHE_MISMATCH")
            self.assertEqual((target / "transcript.txt").read_text(), "旧稿")

    def test_idempotency_and_terminal_state(self):
        with tempfile.TemporaryDirectory() as d:
            store = JobStore(Path(d) / "runtime")
            plan = plan_request(PipelineRequest(("audio.wav",)), load_config()).to_dict()
            first, made = store.submit(plan, key="retry")
            second, made_again = store.submit(plan, key="retry")
            self.assertTrue(made)
            self.assertFalse(made_again)
            self.assertEqual(first["job_id"], second["job_id"])
            with self.assertRaises(PipelineError):
                store.submit({"different": True}, key="retry")
            store.update(first["job_id"], status="ready")
            self.assertEqual(store.update(first["job_id"], status="running")["status"], "ready")

    def test_fake_pipeline_delivers_full_file_and_only_path(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}):
            store = JobStore()
            source = Path(d) / "recording.wav"
            source.write_bytes(b"fake audio")
            plan = plan_request(PipelineRequest((str(source),), postprocess="off"), load_config())
            job, _ = store.submit(plan.to_dict())
            def fake(sources, *, output_dir, **kwargs):
                item = output_dir / "items/001_audio"
                item.mkdir(parents=True)
                path = item / "transcript.txt"
                path.write_text("完整正文" * 10000)
                results = [{"status": "ready", "text_path": str(path), "output_dir": str(item), "char_count": 40000}]
                write_json(output_dir / "manifest.json", {"results": results})
                return {"results": results}
            with patch("workbench.pipeline.transcribe_media_sources", side_effect=fake):
                run_job(store, job["job_id"])
            result = store.get(job["job_id"])
            self.assertEqual(result["status"], "ready")
            self.assertIn("完整正文", Path(result["primary_file"]).read_text())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(emit(result, "paths"), 0)
            self.assertEqual(out.getvalue(), result["primary_file"] + "\n")
            self.assertLessEqual(len(json.dumps(compact(result), ensure_ascii=False).encode()), 2048)

    def test_partial_exit_and_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "transcript.txt"
            text.write_text("可用正文")
            items = [{"status": "ready", "text_path": str(text)}, {"status": "failed", "error": "source failed"}]
            assemble(root, items)
            result = write_receipt(root, job_id="j", status="partial", results=items)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(emit(result, "paths"), 4)
            self.assertIn("partial", err.getvalue())

    def test_unknown_external_request_is_not_cleared_by_owner_exit(self):
        with tempfile.TemporaryDirectory() as d:
            store = JobStore(Path(d))
            store.provider_start("http://127.0.0.1:8081", "one")
            with self.assertRaises(PipelineError):
                store.provider_start("http://127.0.0.1:8081", "two")
            from workbench.resources import ExecutionLease
            with self.assertRaises(PipelineError):
                with ExecutionLease(Path(d)):
                    self.fail("unknown provider admitted a new model stage")
            store.provider_end("http://127.0.0.1:8081", completed=True)
            store.provider_start("http://127.0.0.1:8081", "two")

    def test_registry_migration_backup_and_newer_version_refusal(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / "jobs.sqlite3"
            with sqlite3.connect(db) as c:
                c.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE, request_hash TEXT, status TEXT NOT NULL, data TEXT NOT NULL)")
            JobStore(root)
            backups = list(root.glob("*.backup.sqlite3"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].stat().st_mode & 0o777, 0o600)
            with sqlite3.connect(backups[0]) as c:
                self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], 0)
            with sqlite3.connect(db) as c:
                self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], 1)
                c.execute("PRAGMA user_version=2")
            with self.assertRaises(PipelineError):
                JobStore(root)
            with sqlite3.connect(db) as c:
                self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], 2)

    def test_parallel_idempotent_submissions_create_one_job(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            JobStore(root)
            plan = plan_request(PipelineRequest(("https://example.org/audio",)), load_config()).to_dict()
            def submit(_): return JobStore(root).submit(plan, key="parallel")[0]["job_id"]
            with ThreadPoolExecutor(max_workers=3) as pool:
                self.assertEqual(len(set(pool.map(submit, range(3)))), 1)
            first = JobStore(root).submit(plan)[0]
            second = JobStore(root).submit(plan)[0]
            self.assertNotEqual(first["job_id"], second["job_id"])

    def test_spawn_failure_has_durable_diagnostic_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            store = JobStore(Path(d) / "runtime")
            plan = plan_request(PipelineRequest(("audio.wav",)), load_config()).to_dict()
            job, _ = store.submit(plan)
            with patch("workbench.job_store.subprocess.Popen", side_effect=OSError("cannot execute")):
                with self.assertRaises(PipelineError):
                    store.spawn(job["job_id"])
            final = store.get(job["job_id"])
            self.assertEqual(final["error_code"], "SPAWN_FAILED")
            self.assertEqual(final["status"], "failed")
            self.assertEqual(json.loads(Path(final["receipt_path"]).read_text())["job_id"], job["job_id"])

    def test_worker_protocol_rejects_unmeasured_or_out_of_range_results(self):
        for segments in ([{"text": "text", "start_ms": 0, "end_ms": 0}],
                         [{"text": "text", "start_ms": 0, "end_ms": 2001}], []):
            with self.assertRaises(ValueError):
                validate_worker_result({"schema_version": 1, "text": "text", "segments": segments,
                                        "coverage": {"covered_seconds": 1}})

    def test_link_wrapper_forwards_same_pipeline(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "scripts/transcribe_link.py"
        spec = importlib.util.spec_from_file_location("link_wrapper", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch("workbench.cli.main", return_value=4) as main:
            self.assertEqual(module.main(["https://example.org/episode", "--profile", "fast", "--json"]), 4)
            self.assertEqual(main.call_args.args[0], ["run", "https://example.org/episode", "--profile", "fast", "--output", "json"])

    def test_multi_item_delivery_silence_preview_and_raw_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            item = root / "items/one"
            item.mkdir(parents=True)
            text = item / "transcript.txt"
            text.write_text("原始全文保留")
            results = [{"status": "ready", "text_path": str(text), "asr": {"timing_precision": "vad-segment", "coverage": {"partial": True, "covered_seconds": 300}}},
                       {"status": "ready", "asr": {"coverage": {"content_state": "no_speech"}}}]
            path = assemble(root, results)
            body = path.read_text()
            self.assertIn("原始全文保留", body)
            self.assertIn("仅覆盖部分音频", body)
            self.assertIn("没有检测到可识别语音", body)
            self.assertNotIn("items/one/transcript.txt", body)
            receipt = write_receipt(root, job_id="reading", status="partial", results=results)
            self.assertEqual(receipt["items"][0]["text_path"], str(text))
            self.assertIn(str(text.resolve()), receipt["artifact_hashes"])
            self.assertNotIn("transcript.srt", body)
            self.assertIn("powered by github.com/JinzheNext/NextEcho", body)

    def test_legacy_non_posix_whisper_does_not_require_posix_guard(self):
        from types import SimpleNamespace
        from workbench.transcription import _bounded_whisper
        with patch("workbench.transcription.os", SimpleNamespace(name="nt")), patch("workbench.transcription.validate_whisper_model", return_value={"bytes": 100}), patch("workbench.transcription._write_json"), patch("workbench.transcription.subprocess.run") as run:
            _bounded_whisper(["whisper-cli"], Path("unused-model"), Path("unused-output"))
            self.assertEqual(run.call_args.kwargs["timeout"], 7200)

    def test_wait_timeout_does_not_cancel(self):
        with tempfile.TemporaryDirectory() as d:
            store = JobStore(Path(d) / "runtime")
            plan = plan_request(PipelineRequest(("audio.wav",)), load_config()).to_dict()
            job, _ = store.submit(plan)
            self.assertEqual(wait_for(store, job["job_id"], 0)["status"], "queued")
            self.assertFalse(store.cancelled(job["job_id"]))
