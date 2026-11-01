from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.acoustic_dialogue import build_dialogue_payload, load_dialogue_segments, write_dialogue_outputs
from workbench.contracts import PipelineError
from workbench.dialogue_alignment import align_dialogue_segments
from workbench.pipeline import run_post_asr_stages
from workbench.io_utils import file_hash, write_json


class AcousticDialogueTests(unittest.TestCase):
    def test_unverified_name_does_not_erase_measured_anonymous_voices(self):
        text = "欢迎。大家好，我是李继刚。"
        raw = [{"id":"s", "text":text, "start_ms":0, "end_ms":3000}]
        tokens = [{"text":c,"start_ms":i*200,"end_ms":i*200+100} for i,c in enumerate(text)]
        boundary = text.index("继")*.2
        data = build_dialogue_payload(raw,raw,tokens,
            [{"speaker":"A","start":0,"end":boundary},{"speaker":"B","start":boundary,"end":3}],
            {"speaker_candidates":[{"name":"李继刚"}]})
        self.assertEqual("".join(r["text"] for r in data["transcription"]),text)
        self.assertEqual([r["speaker"] for r in data["transcription"]], ["说话人 1", "说话人 2"])
        self.assertTrue(all(r["speaker_id"] and r["identity_status"] == "unknown" for r in data["transcription"]))
        self.assertEqual(data["unassigned_text_chars"], 0)
        self.assertEqual(data["status"], "ready")
        self.assertNotIn("李继刚",data["speaker_labels"].values())

    def test_voice_without_sufficient_acoustic_prototype_is_not_a_new_person(self):
        raw = [{"id": "one", "text": "对。", "start_ms": 0, "end_ms": 1000}]
        data = build_dialogue_payload(raw, raw, [{"text": "对。", "start_ms": 100, "end_ms": 800}],
            [{"speaker": "weak", "start": 0, "end": 1}], {}, uncertain_labels={"weak"})
        self.assertEqual(data["speaker_labels"], {})
        self.assertEqual(data["transcription"][0]["speaker"], "归属待核对")
        self.assertEqual(data["status"], "partial")

    def test_native_onset_is_not_shifted_to_next_token_midpoint(self):
        rows, _ = align_dialogue_segments(
            [{"id": "s", "text": "甲乙", "start_ms": 0, "end_ms": 2000}],
            [{"text": "甲", "start_ms": 100, "end_ms": None},
             {"text": "乙", "start_ms": 1500, "end_ms": None}],
            [{"speaker": "A", "start": 0, "end": .5},
             {"speaker": "B", "start": .5, "end": 2}], prefix="test")
        self.assertEqual([(r["text"], r["speaker"]) for r in rows], [("甲", "A"), ("乙", "B")])

    def fixture(self):
        text = "欢迎来聊。大家好，我是李继刚。继续提问。"
        raw = [{"id": "raw-1", "text": text, "start_ms": 0, "end_ms": len(text) * 200}]
        left = text.index("大家好") * 200
        right = text.index("继续提问") * 200
        tokens = [{"text": c, "start_ms": i * 200, "end_ms": i * 200 + 150} for i, c in enumerate(text)]
        turns = [{"start": 0, "end": left / 1000, "speaker": "A"},
                 {"start": left / 1000, "end": right / 1000, "speaker": "B"},
                 {"start": right / 1000, "end": len(text) * .2, "speaker": "A"}]
        return raw, tokens, turns

    def test_native_anchors_map_guest_and_keep_unknown_host_stable(self):
        raw, tokens, turns = self.fixture()
        data = build_dialogue_payload(raw, raw, tokens, turns, {"speaker_candidates": [
            {"name": "孟岩", "role": "host"}, {"name": "李继刚", "role": "guest"}]})
        rows = data["transcription"]
        self.assertEqual("".join(r["text"] for r in rows), raw[0]["text"])
        self.assertEqual(rows[0]["speaker"], "说话人 1")
        self.assertEqual(rows[-1]["speaker"], "说话人 1")
        self.assertTrue(any(r["speaker"] == "李继刚" for r in rows))
        self.assertTrue(all(r["segment_ids"] == ["raw-1"] for r in rows))
        self.assertFalse(data["alignment_audit"]["word_timestamps_generated"])

    def test_unmatched_multivoice_text_is_not_a_third_named_speaker(self):
        raw, _, turns = self.fixture()
        data = build_dialogue_payload(raw, raw, [], turns, {})
        self.assertEqual(data["status"], "partial")
        self.assertEqual(data["transcription"][0]["speaker"], "归属待核对")
        self.assertEqual(len(data["speaker_labels"]), 2)
        self.assertFalse(data["transcription"][0]["speaker_id"])

    def test_explicit_overlap_flag_cannot_become_a_confident_single_voice(self):
        raw, tokens, _ = self.fixture()
        turns = [{"start": 0, "end": 20, "speaker": "A", "overlap": True}]
        data = build_dialogue_payload(raw, raw, tokens, turns, {"speaker_candidates": [{"name": "李继刚"}]})
        self.assertEqual(data["status"], "partial")
        self.assertTrue(all(r["speaker"].startswith("交叠发言") for r in data["transcription"]))
        self.assertTrue(all(r["identity_status"] != "resolved" for r in data["transcription"]))

    def test_derived_and_actual_enhanced_text_keep_layer_and_parent_references(self):
        raw, tokens, turns = self.fixture()
        adopted = [{**raw[0], "text": raw[0]["text"].replace("欢迎来聊", "欢迎来聊聊")}]
        data = build_dialogue_payload(raw, adopted, tokens, turns, {})
        data["raw_sha256"] = "rawhash"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = write_dialogue_outputs(root, data)
            rows, saved = load_dialogue_segments(root, "rawhash")
            self.assertEqual("".join(r["text"] for r in rows), adopted[0]["text"])
            self.assertIn("欢迎来聊聊", Path(paths["dialogue_text_path"]).read_text())
            self.assertEqual(json.loads(Path(paths["speaker_json_path"]).read_text())["speaker_labels"], data["speaker_labels"])
            with self.assertRaises(PipelineError):
                load_dialogue_segments(root, "different raw hash")
            with patch("workbench.acoustic_dialogue._implementation_hashes", return_value={"changed": "code"}):
                with self.assertRaises(PipelineError):
                    load_dialogue_segments(root, "rawhash")
            derived = Path(paths["dialogue_json_path"])
            changed = json.loads(derived.read_text())
            changed["transcription"][0]["speaker"] = "未经确认的姓名"
            derived.write_text(json.dumps(changed))
            with self.assertRaises(PipelineError):
                load_dialogue_segments(root, "rawhash")

    def test_native_alignment_allows_speaker_before_polish_without_precise_claim(self):
        order = []
        payload = {"results": [{"status": "ready", "char_count": 10,
                    "asr": {"actual_engine": "fun-asr-nano", "timing_precision": "vad-segment"}}]}
        run_post_asr_stages(payload, speaker=lambda _: order.append("speaker"),
                            polish=lambda: order.append("polish"), speaker_mode="required", native_speaker_alignment=True)
        self.assertEqual(order, ["speaker", "polish"])
        self.assertEqual(payload["results"][0]["asr"]["timing_precision"], "vad-segment")

    def test_polish_reads_measured_turns_and_keeps_raw_and_parent_provenance(self):
        from workbench.postprocess import process_transcript_artifacts, load_segments
        raw, tokens, turns = self.fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "transcript.json"
            original = {"transcription": [{"id": "raw-1", "text": raw[0]["text"],
                         "offsets": {"from": 0, "to": raw[0]["end_ms"]}}]}
            write_json(source, original)
            digest = file_hash(source)
            data = build_dialogue_payload(raw, raw, tokens, turns, {"speaker_candidates": [{"name": "李继刚"}]})
            data["raw_sha256"] = digest
            write_dialogue_outputs(root, data)

            def polish(segments, **kwargs):
                self.assertTrue(any(s.speaker == "李继刚" for s in segments))
                self.assertTrue(all(s.id.startswith("dialogue-") for s in segments))
                return segments, {"status": "ready", "actual_models": ["fixture"]}

            with patch("workbench.postprocess.ollama_status", return_value={"models": []}), \
                    patch("workbench.postprocess.resolve_postprocess_models", return_value={
                        "polish_models": ["fixture"], "summary_models": []}), \
                    patch("workbench.postprocess.attach_speaker_labels", side_effect=AssertionError("must not remap")), \
                    patch("workbench.postprocess.polish_segments", side_effect=polish), \
                    patch("workbench.postprocess.summarize_segments", side_effect=AssertionError("summary disabled")):
                process_transcript_artifacts(root, summarize=False)
            enhanced = json.loads((root / "transcript.enhanced.json").read_text())
            self.assertEqual(file_hash(source), digest)
            self.assertEqual(enhanced["transcription"], original["transcription"])
            self.assertEqual([s.id for s in load_segments(enhanced, layer="enhanced")], [r["id"] for r in data["transcription"]])
            self.assertEqual(enhanced["enhancement"]["dialogue_provenance"]["sha256"], file_hash(root / "transcript.dialogue.json"))


if __name__ == "__main__":
    unittest.main()
