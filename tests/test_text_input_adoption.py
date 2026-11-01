"""Actual layer writers/guards/cache replay; model calls are deterministic fixtures."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.acoustic_dialogue import build_dialogue_payload, write_dialogue_outputs
from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError
from workbench.enhancement import enhance_transcript_artifacts
from workbench.io_utils import file_hash, write_json
from workbench.postprocess import process_transcript_artifacts, _value_signature
from workbench.text_inputs import select_text_input


class TextInputAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.environment = patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(self.root / "runtime")})
        self.environment.start()
        self.config = NextEchoConfig({"postprocess": {"preferred_model_id": "qwen3:4b", "fallback_model": "qwen3:4b"}})
        self.text = "金额36万元，不能超过限额。"
        write_json(self.root / "transcript.json", {"transcription": [{"id": "raw-1", "text": self.text, "offsets": {"from": 0, "to": 2000}}]})
        (self.root / "transcript.txt").write_text(self.text)
        self.raw_hash = file_hash(self.root / "transcript.json")
        self.requests = []

    def tearDown(self):
        self.environment.stop()
        self.tmp.cleanup()

    def model(self, **kwargs):
        data = json.loads(kwargs["user"].split("\n", 1)[1])
        self.requests.append(data)
        return {"segments": [{"id": r["id"], "polished_text": r["text"], "uncertain_terms": [], "change_notes": []} for r in data["segments"]]}, "qwen3:4b", []

    def process(self, artifacts):
        with patch("workbench.postprocess.ollama_status", return_value={"ready": True, "models": ["qwen3:4b"], "tags_digest": "fixture"}), patch("workbench.postprocess.call_json", side_effect=self.model):
            return process_transcript_artifacts(self.root, profile="fast", config=self.config, summarize=False, input_artifacts=artifacts)

    def dialogue(self):
        raw = [{"id": "raw-1", "text": self.text, "start_ms": 0, "end_ms": 2000}]
        tokens = [{"text": c, "start_ms": i * 100, "end_ms": None} for i, c in enumerate(self.text)]
        payload = build_dialogue_payload(raw, raw, tokens, [{"speaker": "A", "start": 0, "end": 2}], {})
        payload["raw_sha256"] = self.raw_hash
        return write_dialogue_outputs(self.root, payload)

    def test_explicit_raw_never_decodes_stale_dialogue_or_speaker_files(self):
        for name in ("transcript.dialogue.json", "transcript.speakers.json"):
            (self.root / name).write_text("NOT JSON 陈旧人物")
        original = {p.name: file_hash(p) for p in self.root.glob("transcript.*")}
        with patch("workbench.postprocess.load_speaker_turns", side_effect=AssertionError("must not read old speaker")), patch("workbench.acoustic_dialogue.load_dialogue_segments", side_effect=AssertionError("must not read old dialogue")):
            structured = self.process({})
            deterministic = enhance_transcript_artifacts(self.root, input_artifacts={})
        self.assertEqual(structured["input_selection"]["layer"], "raw-asr")
        self.assertIsNone(json.loads(Path(deterministic["enhanced_json_path"]).read_text())["enhancement"]["speaker_source"])
        body = (self.root / "transcript.enhanced.txt").read_text()
        self.assertIn("36万元", body)
        self.assertNotIn("陈旧人物", body)
        self.assertEqual({p: file_hash(self.root / p) for p in original}, original)

    def test_current_dialogue_then_off_changes_input_and_does_not_adopt_old_cache(self):
        paths = self.dialogue()
        selected = {k: paths[k] for k in ("dialogue_json_path", "speaker_json_path")}
        first = self.process(selected)
        self.assertEqual(first["input_selection"]["layer"], "acoustic-dialogue")
        self.assertIn("说话人", (self.root / "transcript.enhanced.txt").read_text())
        speaker_hashes = {p.name: file_hash(p) for p in self.root.glob("transcript.*.json") if "enhanced" not in p.name}
        count = len(self.requests)
        second = self.process({})
        self.assertFalse(second.get("cached", False))
        self.assertEqual(second["input_selection"]["layer"], "raw-asr")
        self.assertGreater(len(self.requests), count)
        self.assertNotIn("说话人", (self.root / "transcript.enhanced.txt").read_text())
        self.assertTrue(self.process({})["cached"])
        self.assertEqual({p: file_hash(self.root / p) for p in speaker_hashes}, speaker_hashes)

    def test_explicit_current_legacy_export_does_not_discover_old_canonical(self):
        write_json(self.root / "transcript.speakers.json", {"schema_version": 3, "speakers": [],
            "speaker_turns": [{"id": "r1", "speaker": "本轮说话人", "text": self.text, "start": 0, "end": 2}]})
        (self.root / "transcript.dialogue.json").write_text("stale invalid canonical")
        result = self.process({"speaker_json_path": str(self.root / "transcript.speakers.json")})
        self.assertEqual(result["input_selection"]["layer"], "speaker-artifacts")
        self.assertIn("本轮说话人", (self.root / "transcript.enhanced.txt").read_text())

    def test_missing_external_or_malformed_current_inputs_block_instead_of_raw_fallback(self):
        with self.assertRaises(PipelineError):
            self.process({"speaker_json_path": str(self.root / "missing.json")})
        (self.root / "transcript.speakers.json").write_text("bad current JSON")
        with self.assertRaises((PipelineError, ValueError)):
            self.process({"speaker_json_path": str(self.root / "transcript.speakers.json")})
        with self.assertRaises(PipelineError):
            select_text_input(self.root, {"speaker_json_path": str(self.root / "transcript.json")})

    def test_checkpoint_text_must_match_saved_candidate_even_if_signature_rewritten(self):
        self.process({})
        checkpoint = self.root / "postprocess/polish/chunk_000.json"
        data = json.loads(checkpoint.read_text())
        data["segments"][0]["text"] = data["segments"][0]["text"].replace("36", "360")
        data["output_signature"] = _value_signature(data["segments"])
        write_json(checkpoint, data)
        (self.root / "transcript.enhanced.txt").write_text("force full artifact repair")
        repaired = self.process({})
        self.assertEqual(len(self.requests), 2)
        self.assertIn(0, repaired["polish"]["stale_checkpoints"])
        self.assertNotIn("360万元", (self.root / "transcript.enhanced.txt").read_text())

    def test_full_adoption_rechecks_current_protected_fields_not_only_own_hashes(self):
        self.process({})
        path = self.root / "transcript.enhanced.json"
        data = json.loads(path.read_text())
        data["enhanced_transcription"][0]["text"] = data["enhanced_transcription"][0]["text"].replace("36", "360")
        write_json(path, data)
        audit_path = self.root / "postprocess.audit.json"
        audit = json.loads(audit_path.read_text())
        audit["artifact_hashes"][path.name] = file_hash(path)
        write_json(audit_path, audit)
        repaired = self.process({})
        self.assertFalse(repaired.get("cached", False))
        self.assertEqual(len(self.requests), 1)  # valid original candidate replay
        self.assertNotIn("360万元", (self.root / "transcript.enhanced.json").read_text())

    def test_speaker_guard_code_change_invalidates_adoption_without_touching_raw(self):
        self.process({})
        original = file_hash
        def changed(path):
            return "guard-revision-changed" if Path(path).name == "speaker_artifacts.py" else original(path)
        with patch("workbench.postprocess.file_hash", side_effect=changed):
            new = self.process({})
        self.assertFalse(new.get("cached", False))
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(file_hash(self.root / "transcript.json"), self.raw_hash)

    def test_compat_pipeline_uses_only_current_speaker_stage_even_with_old_files(self):
        from workbench.transcription import transcribe_media_sources
        for mode in ("off", "auto"):
            with self.subTest(mode=mode):
                item = {"status": "ready", "source_label": "001_test", "char_count": len(self.text),
                    "output_dir": str(self.root), "text_path": str(self.root / "transcript.txt"),
                    "json_path": str(self.root / "transcript.json")}
                payload = {"results": [item], "manifest_path": self.root / "manifest.json", "elapsed_seconds": 0}
                write_json(self.root / "manifest.json", {"results": [item]})
                (self.root / "transcript.dialogue.json").write_text("stale invalid canonical")
                (self.root / "transcript.speakers.json").write_text("stale invalid export")
                stages = []
                def speaker(current):
                    stages.append("speaker")
                    paths = self.dialogue()
                    current["results"][0].update(paths)
                    write_json(self.root / "manifest.json", {"results": current["results"]})
                    return {"status": "ready"}
                with patch("workbench.transcription._transcribe_media_sources", return_value=payload), patch("workbench.postprocess.ollama_status", return_value={"ready": True, "models": ["qwen3:4b"]}), patch("workbench.postprocess.call_json", side_effect=self.model):
                    result = transcribe_media_sources([], output_dir=self.root, speaker_stage=speaker, speaker_mode=mode,
                        postprocess_profile="fast", enable_enhancement=True, config=self.config)
                selection = result["results"][0]["enhancement"]["input_selection"]
                self.assertEqual(selection["layer"], "raw-asr" if mode == "off" else "acoustic-dialogue")
                self.assertEqual(stages, [] if mode == "off" else ["speaker"])
                self.assertIn("receipt", result)


if __name__ == "__main__":
    unittest.main()
