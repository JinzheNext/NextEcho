"""Speaker-only reruns stay isolated from frozen ASR and adopted text."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.contracts import PipelineError
from workbench.config import NextEchoConfig
from workbench.io_utils import file_hash, write_json
from workbench.publication import publish_pipeline, verify_publication
from workbench.speaker_revision import refine_published_speakers
from tests.test_publication import factory_item


class SpeakerRevisionTests(unittest.TestCase):
    def make_published_run(self, root: Path) -> tuple[Path, dict]:
        run = root / "base-run"
        item = factory_item(run)
        item_dir = Path(item["output_dir"])
        (item_dir / "transcript.enhanced.json").write_text(
            json.dumps({"transcription": [{"text": "采用文字"}]}, ensure_ascii=False), encoding="utf-8"
        )
        (item_dir / "transcript.enhanced.txt").write_text("采用文字", encoding="utf-8")
        (item_dir / "postprocess.audit.json").write_text("{}", encoding="utf-8")
        write_json(run / "manifest.json", {"results": [item], "speaker_transcript": {"status": "not_started"}})
        receipt = publish_pipeline(run, job_id="base-job", results=[item], stages={"asr": "completed", "polish": "completed"})
        return run, receipt

    @staticmethod
    def fake_speaker_builder(run_dir: Path, **kwargs) -> dict:
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        item = manifest["results"][0]
        folder = Path(item["output_dir"])
        outputs = {
            "speaker_json_path": folder / "transcript.speakers.json",
            "speaker_text_path": folder / "transcript.speakers.txt",
            "speaker_markdown_path": folder / "transcript.speakers.md",
            "speaker_map_path": folder / "speaker_map.json",
            "dialogue_json_path": folder / "transcript.dialogue.json",
        }
        write_json(outputs["speaker_json_path"], {"schema_version": 3, "status": "ready", "speakers": [], "speaker_turns": []})
        outputs["speaker_text_path"].write_text("说话人 1：你好。", encoding="utf-8")
        outputs["speaker_markdown_path"].write_text("说话人 1 [00:00:00 - 00:00:01]：你好。", encoding="utf-8")
        write_json(outputs["speaker_map_path"], {})
        write_json(outputs["dialogue_json_path"], {
            "transcription": [], "speakers": {"speakers": []}, "speaker_labels": {},
            "alignment_audit": {}, "audio_sha256": "", "text_layer": "enhanced",
        })
        item.update(speaker_status="ready", **{key: str(value.resolve()) for key, value in outputs.items()})
        manifest["speaker_transcript"] = {"status": "ready", "items": []}
        write_json(manifest_path, manifest)
        return manifest["speaker_transcript"]

    @staticmethod
    def fake_review_writer(path: Path, *_args, **_kwargs) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("review pending", encoding="utf-8")
        return {"fingerprint": "f" * 64}

    def test_publishes_new_speaker_revision_without_asr_or_polish_and_freezes_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base, original_receipt = self.make_published_run(root)
            frozen_names = (
                "transcript.json", "transcript.txt", "transcript.enhanced.json",
                "transcript.enhanced.txt", "postprocess.audit.json",
            )
            before = {
                name: file_hash(base / "items" / "one" / name)
                for name in frozen_names if (base / "items" / "one" / name).is_file()
            }
            original_config = NextEchoConfig({})
            with patch("workbench.speaker_revision.build_run_speakers", side_effect=self.fake_speaker_builder) as speaker_builder, \
                 patch("workbench.identity_review.write_identity_review", side_effect=self.fake_review_writer), \
                 patch("workbench.transcription.transcribe_media_sources", side_effect=AssertionError("ASR must not run")), \
                 patch("workbench.postprocess.process_transcript_artifacts", side_effect=AssertionError("polish must not run")):
                target, manifest, receipt = refine_published_speakers(
                    base, backend="test-local", config=original_config,
                    idempotency_key="intent-speaker-review-1",
                )
                repeated_target, repeated_manifest, repeated_receipt = refine_published_speakers(
                    base, backend="test-local", config=original_config,
                    idempotency_key="intent-speaker-review-1",
                )

            self.assertNotEqual(target, base)
            self.assertTrue(target.is_dir())
            self.assertEqual(repeated_target, target)
            self.assertEqual(repeated_receipt, receipt)
            self.assertEqual(speaker_builder.call_count, 1)
            runtime = speaker_builder.call_args.kwargs["config"].get_path("runtime")
            self.assertIs(runtime["resume_verified"], True)
            self.assertEqual(Path(runtime["resume_parent_dir"]), base.resolve())
            self.assertEqual(Path(runtime["resume_run_dir"]), target.resolve())
            self.assertIsNone(original_config.get_path("runtime", "resume_verified"))
            self.assertEqual(receipt["status"], "ready")
            self.assertEqual(verify_publication(base, job_id=original_receipt["job_id"]), original_receipt)
            self.assertEqual(verify_publication(target, job_id=receipt["job_id"]), receipt)
            after = {
                name: file_hash(target / "items" / "one" / name)
                for name in frozen_names if (target / "items" / "one" / name).is_file()
            }
            self.assertEqual(before, after)
            refinement = manifest["speaker_refinement"]
            self.assertEqual(refinement["models_called"], {"asr": 0, "polish": 0, "summary": 0})
            self.assertIn("asr", refinement["stage_blocklist"])
            self.assertEqual(manifest["results"][0]["speaker_identity_review_status"], "human-review-pending")
            self.assertTrue(Path(manifest["results"][0]["speaker_identity_review_path"]).is_file())

    def test_rejects_any_change_to_frozen_transcript_and_keeps_source_published(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base, original_receipt = self.make_published_run(root)

            def tampering_builder(run_dir: Path, **kwargs):
                result = self.fake_speaker_builder(run_dir, **kwargs)
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                Path(manifest["results"][0]["output_dir"], "transcript.enhanced.txt").write_text("tampered", encoding="utf-8")
                return result

            with patch("workbench.speaker_revision.build_run_speakers", side_effect=tampering_builder), \
                 patch("workbench.identity_review.write_identity_review", side_effect=self.fake_review_writer):
                with self.assertRaises(PipelineError) as caught:
                    refine_published_speakers(base, backend="test-local", config=NextEchoConfig({}),
                                              idempotency_key="intent-speaker-tamper")
            self.assertEqual(caught.exception.code, "CACHE_MISMATCH")
            self.assertEqual(verify_publication(base, job_id=original_receipt["job_id"]), original_receipt)
            self.assertEqual(list(root.glob("base-run-speakers-*")), [])

    def test_candidate_speaker_result_is_published_as_partial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base, _ = self.make_published_run(root)

            def candidate_builder(run_dir: Path, **kwargs):
                view = self.fake_speaker_builder(run_dir, **kwargs)
                manifest_path = run_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["speaker_transcript"]["release_status"] = "candidate"
                write_json(manifest_path, manifest)
                return {**view, "release_status": "candidate"}

            with patch("workbench.speaker_revision.build_run_speakers", side_effect=candidate_builder):
                target, manifest, receipt = refine_published_speakers(
                    base, backend="candidate-test", config=NextEchoConfig({}),
                    idempotency_key="candidate-speaker-test",
                )

            self.assertEqual(receipt["status"], "partial")
            self.assertEqual(manifest["speaker_refinement"]["release_status"], "candidate")
            self.assertEqual(manifest["results"][0]["speaker_status"], "partial")
            self.assertEqual(
                manifest["results"][0]["speaker_identity_review_status"],
                "unavailable-audio-integrity",
            )
            self.assertEqual(verify_publication(target, job_id=receipt["job_id"]), receipt)

    def test_requires_a_published_source_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(PipelineError) as caught:
                refine_published_speakers(Path(temp_dir) / "missing")
            self.assertEqual(caught.exception.code, "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()
