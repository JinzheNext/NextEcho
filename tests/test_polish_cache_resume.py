"""Real cache/adoption writers with synthetic local model boundaries."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import tests.test_native_speaker as native_fixtures
from workbench import postprocess as P
from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError
from workbench.enhancement import default_glossary_rules
from workbench.io_utils import file_hash, write_json
from workbench.llm_providers import CONTEXT, GenerationContext
from workbench.polish_cache import cache_input_identity, local_candidate_records
from workbench.speaker_service import build_item_speakers
from workbench.text_inputs import current_speaker_inputs, select_text_input


class PolishCheckpointCopyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.parent = self.root / "parent"
        self.parent.mkdir()
        self.calls = []

    def run_polish(self, root, segments, model):
        token = CONTEXT.set(GenerationContext({"polish_max_segments": 1}))
        try:
            with patch.object(P, "call_json", side_effect=model):
                return P.polish_segments(segments, rules=default_glossary_rules(), models=["fixture"],
                    profile="balanced", style="", endpoint="http://127.0.0.1:1", checkpoint_dir=root)
        finally:
            CONTEXT.reset(token)

    def copy_without_original_path(self):
        child = self.root / "child"
        shutil.copytree(self.parent, child)
        self.parent.rename(self.root / "historical-parent")
        return child

    def test_relative_candidate_replays_from_copy_with_original_path_unavailable(self):
        source = [P.Segment("a", "金额36万元，不能增加。", 0, 2000, "A")]
        def model(**kwargs):
            self.calls.append(kwargs)
            return {"segments": [{"id": "a", "polished_text": source[0].text}]}, "fixture", []
        self.run_polish(self.parent, source, model)
        original_checkpoint = (self.parent / "chunk_000.json").read_bytes()
        child = self.copy_without_original_path()
        output, audit = self.run_polish(child, source, model)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(output, source)
        self.assertEqual(audit["cached_chunks"], [0])
        self.assertEqual((child / "chunk_000.json").read_bytes(), original_checkpoint)
        record = audit["candidate_attempts"][0]
        self.assertTrue(Path(record["candidate_path"]).is_relative_to(child))
        self.assertTrue(Path(record["candidate_origin_path"]).is_relative_to(self.parent))

    def test_only_failed_chunk_reruns_and_does_not_erase_original_fallback(self):
        source = [P.Segment("a", "金额36万元，不能增加。", 0, 2000, "A"),
                  P.Segment("b", "金额78万元，不能增加。", 2000, 4000, "B")]
        def model(**kwargs):
            rows = json.loads(kwargs["user"].split("\n", 1)[1])["segments"]
            self.calls.extend(r["id"] for r in rows)
            return {"segments": [{"id": r["id"], "polished_text": r["text"].replace("78", "780")} for r in rows]}, "fixture", []
        first, _ = self.run_polish(self.parent, source, model)
        self.assertEqual(first, source)
        child = self.copy_without_original_path()
        second, audit = self.run_polish(child, source, model)
        self.assertEqual(self.calls, ["a", "b", "b", "b", "b"])
        self.assertEqual(second, source)
        self.assertEqual(audit["cached_chunks"], [0])
        self.assertEqual(audit["original_fallback_chunks"], [1])
        self.assertEqual(audit["status"], "partial")

    def test_cached_semantic_review_stays_partial_after_copy(self):
        source = [P.Segment("a", "我可能讲的是李纪刚。", 0, 2000, "A")]
        def model(**kwargs):
            self.calls.append(kwargs)
            return {"segments": [{"id": "a", "polished_text": "我可能讲的是李继刚。"}]}, "fixture", []
        _, first = self.run_polish(self.parent, source, model)
        self.assertTrue(first["review_required"])
        child = self.copy_without_original_path()
        _, second = self.run_polish(child, source, model)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(second["status"], "partial")
        self.assertTrue(second["review_required"])
        self.assertEqual(first["review_segments"], second["review_segments"])

    def test_relative_pointer_cannot_escape_or_fall_back_to_external_origin(self):
        source = [P.Segment("a", "金额36万元，不能增加。", 0, 2000, "A")]
        def model(**kwargs):
            return {"segments": [{"id": "a", "polished_text": source[0].text}]}, "fixture", []
        _, audit = self.run_polish(self.parent, source, model)
        record = audit["candidate_attempts"][0]
        for edits in [{"candidate_relative_path": "../outside.json"}, {"path_protocol": "unknown"},
                      {"candidate_relative_path": "candidates/another.json"}, {"sha256": "0" * 64}]:
            with self.subTest(edits=edits):
                self.assertIsNone(local_candidate_records([{**record, **edits}], self.parent))
        child = self.copy_without_original_path()
        (child / record["candidate_relative_path"]).unlink()
        self.assertIsNone(local_candidate_records([record], child))

    def test_candidate_and_checkpoint_forged_hashes_do_not_bypass_current_numeric_guard(self):
        source = [P.Segment("a", "金额36万元，不能增加。", 0, 2000, "A")]
        def model(**kwargs):
            self.calls.append(kwargs)
            return {"segments": [{"id": "a", "polished_text": source[0].text}]}, "fixture", []
        self.run_polish(self.parent, source, model)
        checkpoint = self.parent / "chunk_000.json"
        saved = json.loads(checkpoint.read_text())
        record = saved["candidate_attempts"][0]
        candidate = Path(record["candidate_path"])
        data = json.loads(candidate.read_text())
        data["candidate"]["segments"][0]["polished_text"] = source[0].text.replace("36", "360")
        write_json(candidate, data)
        record["sha256"] = file_hash(candidate)
        saved["segments"][0]["text"] = data["candidate"]["segments"][0]["polished_text"]
        saved["output_signature"] = P._value_signature(saved["segments"])
        write_json(checkpoint, saved)
        output, _ = self.run_polish(self.parent, source, model)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(output, source)


class NativePolishCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = native_fixtures.NativeSpeakerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.calls = []

    def process(self, root, item, config):
        effective = config.to_dict()
        effective.setdefault("postprocess", {}).update(preferred_model_id="qwen3:4b", fallback_model="qwen3:4b",
                                                       preserve_standalone_acknowledgements=False)
        def model(**kwargs):
            data = json.loads(kwargs["user"].split("\n", 1)[1])
            self.calls.append(data)
            return {"segments": [{"id": r["id"], "polished_text": r["text"]} for r in data["segments"]]}, "qwen3:4b", []
        with patch.object(P, "call_json", side_effect=model), patch.object(P, "ollama_status", return_value={"ready": True, "models": ["qwen3:4b"], "tags_digest": "fixture"}):
            return P.process_transcript_artifacts(root, profile="fast", config=NextEchoConfig(effective),
                summarize=False, input_artifacts=current_speaker_inputs(item))

    def test_real_native_relocation_reuses_polish_but_refreshes_current_adoption_provenance(self):
        f = self.fixture
        with f.synthetic_models():
            f.build()
            cfg = NextEchoConfig({"diarization": f.settings})
            first = self.process(f.folder, f.item, cfg)
            first_count = len(self.calls)
            self.assertGreater(first_count, 0)
            original = {p: file_hash(p) for p in f.folder.rglob("*") if p.is_file()}
            child, item, config = f.resume_copy()
            build_item_speakers(child, item, config=config)
            second = self.process(child, item, config)
            self.assertEqual(len(self.calls), first_count)
            self.assertEqual(second["polish"]["cached_chunks"], list(range(second["polish"]["chunks"])))
            self.assertNotEqual(first["input_selection"], second["input_selection"])
            enhanced = json.loads((child / "transcript.enhanced.json").read_text())
            self.assertEqual(enhanced["enhancement"]["dialogue_provenance"]["sha256"], file_hash(child / "transcript.dialogue.json"))
            self.assertEqual(enhanced["enhancement"]["input_selection"], second["input_selection"])
            self.assertTrue(all(file_hash(p) == digest for p, digest in original.items()))
            self.assertEqual(len(f.native_calls), 1)
            self.assertEqual(len(f.ctc_calls), 1)

    def test_changed_attribution_profile_invalidates_cache_even_when_the_model_is_reused(self):
        f = self.fixture
        with f.synthetic_models():
            f.build()
            config = NextEchoConfig({"diarization": f.settings})
            self.process(f.folder, f.item, config)
            count = len(self.calls)
            f.settings["nemotron"]["attribution_profile"] = "posterior-priority"
            f.build()
            self.process(f.folder, f.item, NextEchoConfig({"diarization": f.settings}))
            self.assertGreater(len(self.calls), count)
            self.assertEqual(len(f.native_calls), 1)

    def test_evidence_tamper_blocks_before_cache_or_model_and_unknown_semantic_fields_are_retained(self):
        f = self.fixture
        with f.synthetic_models():
            f.build()
            config = NextEchoConfig({"diarization": f.settings})
            self.process(f.folder, f.item, config)
            count = len(self.calls)
            from workbench.acoustic_dialogue import load_dialogue_segments
            _, payload = load_dialogue_segments(f.folder, file_hash(f.folder / "transcript.json"))
            selection = select_text_input(f.folder, current_speaker_inputs(f.item))
            first = cache_input_identity(f.folder, selection, payload)
            extra = copy.deepcopy(payload)
            extra["future_semantic_constraint"] = "preserve this decision"
            self.assertNotEqual(first, cache_input_identity(f.folder, selection, extra))
            probability = Path(f.item["native_probability_path"])
            probability.write_bytes(b"\0" * probability.stat().st_size)
            with self.assertRaises(PipelineError) as caught:
                self.process(f.folder, f.item, config)
            self.assertEqual(caught.exception.code, "CACHE_MISMATCH")
            self.assertEqual(len(self.calls), count)


if __name__ == "__main__":
    unittest.main()
