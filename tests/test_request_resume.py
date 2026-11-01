"""Persistent intent/resume contracts; no model or external network."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.cli import main
from workbench.config import NextEchoConfig, load_config
from workbench.contracts import PipelineError, PipelineRequest, plan_request
from workbench.io_utils import file_hash, value_hash, write_json
from workbench.job_store import JobStore
from workbench.resume import capture_resume_inputs, resume_plan


class ResumePlanTests(unittest.TestCase):
    def test_segment_repair_is_explicit_refresh_and_does_not_change_original_request_model(self):
        with tempfile.TemporaryDirectory() as d:
            _, parent = self.parent(Path(d))
            old = json.loads(json.dumps(parent['plan']))
            with self.assertRaises(PipelineError):
                resume_plan(parent, repair_polish=True)
            plan, audit = resume_plan(parent, mode='refresh', config=load_config(), repair_polish=True)
            self.assertTrue(plan['effective']['postprocess']['repair_candidates'])
            self.assertEqual(plan['request'], old['request'])
            self.assertEqual(parent['plan'], old)
            self.assertIn('polish', audit['configuration_affected_stage_rechecks'])

    def parent(self, root):
        store = JobStore(root / "runtime")
        plan = plan_request(PipelineRequest(("https://example.org/feed.xml",), profile="fast", postprocess="off"), load_config()).to_dict()
        job, _ = store.submit(plan)
        return store, store.update(job["job_id"], status="cancelled")

    def invoke(self, *args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main([*args, "--output", "json"])
        self.assertLessEqual(len(stdout.getvalue().encode()), 2048)
        return code, json.loads(stdout.getvalue())

    def test_exact_uses_original_plan_and_date_without_loading_changed_config(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}):
            root = Path(d)
            store, parent = self.parent(root)
            source = Path(parent["run_dir"])
            before = (source / "request.json").read_bytes()
            with patch("workbench.headless_cli.bound_config", side_effect=AssertionError("must not reload config")), patch("workbench.job_store.JobStore.spawn") as spawn:
                code, result = self.invoke("jobs", "resume", parent["job_id"], "--mode", "exact", "--idempotency-key", "same-intent")
                again_code, again = self.invoke("jobs", "resume", parent["job_id"], "--mode", "exact", "--idempotency-key", "same-intent")
            self.assertEqual((code, again_code), (0, 0))
            self.assertEqual(result["job_id"], again["job_id"])
            spawn.assert_called_once()
            child = store.get(result["job_id"], reconcile=False)
            self.assertEqual(value_hash(child["plan"]), value_hash(parent["plan"]))
            self.assertEqual(child["transcribed_on"], parent["transcribed_on"])
            self.assertEqual(child["transcription_timezone"], parent["transcription_timezone"])
            self.assertEqual(child["attempt"], parent["attempt"] + 1)
            audit = json.loads((Path(child["run_dir"]) / "resume.plan.json").read_text())
            self.assertEqual(audit["differences"], [])
            self.assertEqual(audit["mode"], "exact")
            self.assertEqual((source / "request.json").read_bytes(), before)
            self.assertEqual(store.get(parent["job_id"])["status"], "cancelled")

    def test_refresh_has_new_plan_redacted_diff_and_same_frozen_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store, parent = self.parent(root)
            config = load_config().to_dict()
            config.setdefault("profiles", {}).setdefault("fast", {})["polish"] = "qwen35-9b"
            config.setdefault("providers", {}).setdefault("ollama", {})["api_key"] = "DO-NOT-PRINT"
            old = json.loads(json.dumps(parent["plan"]))
            new, audit = resume_plan(parent, mode="refresh", config=NextEchoConfig(config))
            self.assertEqual(new["polish_profile"], "qwen35-9b")
            self.assertNotEqual(audit["parent_plan_sha256"], audit["effective_plan_sha256"])
            self.assertTrue(audit["configuration_changed"])
            self.assertTrue(any(row["path"] == ["polish_profile"] for row in audit["differences"]))
            self.assertNotIn("DO-NOT-PRINT", json.dumps(audit))
            self.assertEqual(audit["source_policy"], "frozen")
            self.assertEqual(audit["date_policy"], "inherit-original")
            self.assertNotIn("asr", audit["configuration_affected_stage_rechecks"])
            self.assertIn("polish", audit["configuration_affected_stage_rechecks"])
            self.assertEqual(parent["plan"], old)
            child, _ = store.submit(new, parent=parent["job_id"], resume_from=parent["run_dir"], key="refresh", resume_policy=audit)
            self.assertEqual(child["plan"]["polish_profile"], "qwen35-9b")
            self.assertEqual(child["transcribed_on"], parent["transcribed_on"])

    def test_same_resume_key_different_mode_conflicts_and_lookup_recovers_child(self):
        with tempfile.TemporaryDirectory() as d:
            store, parent = self.parent(Path(d))
            plan, exact = resume_plan(parent)
            child, _ = store.submit(plan, parent=parent["job_id"], resume_from=parent["run_dir"], key="uuid-1", resume_policy=exact)
            self.assertEqual(store.lookup_key("uuid-1")["job_id"], child["job_id"])
            with self.assertRaises(PipelineError):
                store.submit(plan, parent=parent["job_id"], resume_from=parent["run_dir"], key="uuid-1", resume_policy={**exact, "mode": "refresh"})
            with self.assertRaises(PipelineError) as error:
                store.lookup_key("never-submitted")
            self.assertEqual(error.exception.code, "REQUEST_NOT_FOUND")

    def test_post_submission_context_change_blocks_before_asr_or_recrawl(self):
        from workbench.pipeline import execute_plan
        from workbench.contracts import PipelinePlan
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, parent = self.parent(root)
            old = Path(parent["run_dir"])
            context = old / "items/001_old/source.context.json"
            write_json(context, {"shownotes": {"text": "frozen"}})
            _, audit = resume_plan(parent)
            write_json(context, {"shownotes": {"text": "changed after submission"}})
            new = root / "new"
            new.mkdir()
            plan = parent["plan"]
            with patch("workbench.pipeline.resolve_media_inputs") as resolve, patch("workbench.pipeline.transcribe_media_sources") as transcribe:
                with self.assertRaises(PipelineError) as error:
                    execute_plan(PipelinePlan(PipelineRequest(**plan["request"]), plan["asr_profile"], plan["polish_profile"], plan["effective"]),
                        new, job_id="test", resume_from=str(old), resume_policy=audit)
                self.assertEqual(error.exception.code, "CACHE_MISMATCH")
                resolve.assert_not_called()
                transcribe.assert_not_called()

    def test_known_original_context_hash_is_not_redefined_at_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            path = root / "items/001_old/source.context.json"
            write_json(path, {"source": "original"})
            parent = {"run_dir": str(root), "artifact_hashes": {str(path): file_hash(path)}}
            write_json(path, {"source": "tampered"})
            with self.assertRaises(PipelineError):
                capture_resume_inputs(parent)

    def test_lookup_revalidates_delivery_and_never_rewrites_terminal_record(self):
        from workbench.publication import publish_pipeline
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = JobStore(root / "runtime")
            plan = plan_request(PipelineRequest(("https://example.org/media.wav",), postprocess="off"), load_config()).to_dict()
            job, _ = store.submit(plan, key="uuid")
            folder = Path(job["run_dir"])
            text = folder / "text.txt"
            text.write_text("真实写出器fixture")
            receipt = publish_pipeline(folder, job_id=job["job_id"], plan=plan, results=[{"status": "ready", "text_path": str(text)}])
            store.update(job["job_id"], **{k: v for k, v in receipt.items() if k not in {"job_id", "plan"}})
            old = (folder / "receipt.json").read_bytes()
            text.write_text("corrupt")
            self.assertEqual(store.lookup_key("uuid")["status"], "blocked")
            self.assertEqual(store.get(job["job_id"], reconcile=False)["status"], "ready")
            self.assertEqual((folder / "receipt.json").read_bytes(), old)


class FrozenContextTests(unittest.TestCase):
    def test_resume_uses_original_media_after_local_source_changes_or_disappears(self):
        import shutil
        from workbench.sources import resolve_source
        from workbench.transcription import transcribe_single_source
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            source = root / "local.wav"
            source.write_bytes(b"original input bytes")
            descriptor = resolve_source(str(source))
            descriptor.source_label = "001_local"
            request = dict(descriptor=descriptor, items_dir=root / "old/items", model_path=None, whisper_bin="fixture",
                ffmpeg_bin="fixture", language="zh", max_seconds=0, output_formats=["txt", "json"], glossary_text="",
                enable_enhancement=False, ai_polish=False, ai_style="", index=1, asr_engine="whisper-cpp", asr_model="")
            def extract(**kw):
                kw["target_path"].write_bytes(b"normalized fixture")
            def recognize(**kw):
                (kw["output_dir"] / "transcript.txt").write_text("旧媒体文字")
                write_json(kw["output_dir"] / "transcript.json", {"transcription": [{"text": "旧媒体文字"}]})
                return {"actual_engine": "fixture"}
            with patch("workbench.transcription.extract_audio", side_effect=extract) as normalize, patch("workbench.transcription.transcribe_audio_with_fallback", side_effect=recognize) as asr:
                first = transcribe_single_source(**request)
                snapshot = capture_resume_inputs({"run_dir": str(root / "old")})
                shutil.copytree(root / "old", root / "new")
                request.update(items_dir=root / "new/items", config=NextEchoConfig({"runtime": {"resume_verified": True, "frozen_media": snapshot["media"]}}))
                source.write_bytes(b"new content at same original path")
                self.assertEqual(transcribe_single_source(**request)["status"], "cached")
                source.unlink()
                self.assertEqual(transcribe_single_source(**request)["status"], "cached")
                self.assertEqual(asr.call_count, 1)
                self.assertEqual(normalize.call_count, 1)
                self.assertEqual(Path(first["media_path"]).read_bytes(), b"original input bytes")
                frozen = root / "new" / snapshot["media"]["items/001_local"]["relative_path"]
                frozen.write_bytes(b"corrupt frozen media")
                with self.assertRaises(PipelineError):
                    transcribe_single_source(**request)
                self.assertEqual(asr.call_count, 1)

    def test_rerun_asr_keeps_frozen_context_and_missing_context_is_not_fetched(self):
        from workbench.sources import SourceDescriptor
        from workbench.transcription import transcribe_single_source
        for captured in (True, False):
            with self.subTest(captured=captured), tempfile.TemporaryDirectory() as d:
                root = Path(d).resolve()
                item = root / "items/001_old"
                item.mkdir(parents=True)
                context = item / "source.context.json"
                snapshots = {}
                if captured:
                    write_json(context, {"schema_version": 1, "shownotes": {"text": "OLD frozen notes"}})
                    snapshots["items/001_old/source.context.json"] = file_hash(context)
                media = root / "source.mp3"
                media.write_bytes(b"synthetic media")
                descriptor = SourceDescriptor("page", "xiaoyuzhou", "https://www.xiaoyuzhoufm.com/episode/old",
                    "https://www.xiaoyuzhoufm.com/episode/old", "https://media.test/old.mp3", "old", "show", 1, "", "001_old", "fixture")
                config = NextEchoConfig({"runtime": {"resume_verified": True, "freeze_source_context": True,
                                            "frozen_source_contexts": snapshots}})
                def extract(**kw):
                    kw["target_path"].write_bytes(b"synthetic PCM")
                def recognize(**kw):
                    (kw["output_dir"] / "transcript.txt").write_text("新的识别文字")
                    write_json(kw["output_dir"] / "transcript.json", {"transcription": [{"text": "新的识别文字"}]})
                    return {"actual_engine": "fixture"}
                with patch("workbench.transcription.crawl_episode_context", side_effect=AssertionError("exact must not recrawl")) as crawl, patch("workbench.transcription.prepare_media_source", return_value=media), patch("workbench.transcription.extract_audio", side_effect=extract), patch("workbench.transcription.transcribe_audio_with_fallback", side_effect=recognize) as asr:
                    result = transcribe_single_source(descriptor=descriptor, items_dir=root / "items", model_path=None,
                        whisper_bin="fixture", ffmpeg_bin="fixture", language="zh", max_seconds=0, output_formats=["txt", "json"],
                        glossary_text="", enable_enhancement=False, ai_polish=False, ai_style="", postprocess_profile="off",
                        progress_callback=None, base_progress=0, item_count=1, index=1, asr_engine="whisper-cpp", asr_model="", config=config)
                self.assertEqual(result["status"], "ready", result)
                self.assertEqual(asr.call_count, 1)
                crawl.assert_not_called()
                if captured:
                    self.assertEqual(result["source_context"]["shownotes"]["text"], "OLD frozen notes")
                    self.assertEqual(file_hash(context), snapshots["items/001_old/source.context.json"])
                    self.assertEqual(result["source_context_json_path"], str(context))
                else:
                    self.assertEqual(result["source_context"]["status"], "not_captured")
                    self.assertFalse(context.exists())


if __name__ == "__main__":
    unittest.main()
