"""Behavioral counterexamples from Pro R04/R08/R09, across both entry paths."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.acoustic_dialogue import build_dialogue_payload, write_dialogue_outputs
from workbench.diarization import DiarizationTurn
from workbench.dialogue_alignment import align_dialogue_segments
from workbench.speaker_identity import DefaultSpeakerIdentityResolver
from workbench.speaker_transcript import (
    TranscriptSegment, TranscriptWord, align_segments_to_speakers,
    apply_identity_map_to_turns, detect_speaker_backend_status, parse_transcript_words,
    run_timeline_fallback_diarization, speaker_turn_to_dict, write_speaker_outputs,
    speaker_for_interval, nearest_speaker_for_interval,
)


class SpeakerEvidenceContractTests(unittest.TestCase):
    def test_legacy_clock_strings_are_seconds_even_in_from_to_fields(self):
        words = [{"text": "我是", "from": "00:00:16,000", "to": "00:00:17,000"},
                 {"text": "李继刚", "from": "00:00:17,000", "to": "00:00:18,000"}]
        result = DefaultSpeakerIdentityResolver().resolve(
            [DiarizationTurn("A", 0, 15), DiarizationTurn("B", 15, 30)],
            [{"text": "我是李继刚。", "start": 0, "end": 30, "words": words}],
            {"speaker_candidates": ["李继刚"]})
        self.assertEqual(result.get("B").display_name, "李继刚")
        self.assertEqual(result.get("A").display_name, "unknown")
        parsed = parse_transcript_words(words, segment_start=0, segment_end=30)
        self.assertEqual([(w.start, w.end) for w in parsed], [(16, 17), (17, 18)])

    def test_compatibility_helpers_cannot_borrow_or_majority_assign_a_voice(self):
        turns = [DiarizationTurn("A", 0, 1), DiarizationTurn("B", 2, 4)]
        self.assertEqual(speaker_for_interval(.5, 3.5, turns)[:2], ("unknown", ""))
        self.assertEqual(nearest_speaker_for_interval(1.2, 1.5, turns, max_distance=2)[:2], ("unknown", ""))
        self.assertEqual(speaker_for_interval(2, 3, turns)[:2], ("B", "B"))

    def test_native_boundary_onsets_use_half_open_point_not_fake_overlap(self):
        source = [{"id": "s", "text": "甲乙丙", "start_ms": 0, "end_ms": 3000}]
        tokens = [{"text": c, "start_ms": t, "end_ms": None} for c, t in zip("甲乙丙", [500, 1000, 2000])]
        turns = [{"speaker": "A", "start": 0, "end": 1}, {"speaker": "B", "start": 1, "end": 3}]
        rows, audit = align_dialogue_segments(source, tokens, turns, prefix="t")
        self.assertEqual([(r["text"], r["speaker"]) for r in rows], [("甲", "A"), ("乙丙", "B")])
        evidence = [e for r in rows for e in r["alignment_spans"]]
        self.assertEqual([e["native_start_ms"] for e in evidence], [500, 1000, 2000])
        self.assertTrue(all(e["native_end_ms"] is None for e in evidence))
        self.assertEqual(evidence[0]["review_end_ms"], 1000)
        self.assertTrue(all(r["annotation_kind"] == "text-review-envelope" for r in rows))
        self.assertFalse(audit["scoring_eligible"])
        self.assertTrue(all(t["end_ms"] is None for t in tokens))

    def test_sequential_voices_are_not_overlap_but_real_overlap_is(self):
        source = [{"text": "甲", "start_ms": 0, "end_ms": 2000}]
        token = [{"text": "甲", "start_ms": 100, "end_ms": 1900}]
        for turns, expected in [
            ([{"speaker": "A", "start": 0, "end": 1}, {"speaker": "B", "start": 1, "end": 2}], "归属待核对"),
            ([{"speaker": "A", "start": 0, "end": 1.2}, {"speaker": "B", "start": 1, "end": 2}], "交叠发言（A＋B）"),
        ]:
            rows, _ = align_dialogue_segments(source, token, turns, prefix="t")
            self.assertEqual(rows[0]["speaker"], expected)

    def test_legacy_missing_words_never_drop_or_inject_characters(self):
        text = " 你好，世界。再见！ "
        for words in [[TranscriptWord("你好", 0, 1)],
                      [TranscriptWord("你好幻觉", 0, 1), TranscriptWord("再见", 3, 4)],
                      [TranscriptWord("世界", 2, 2)], []]:
            rows, _ = align_segments_to_speakers([TranscriptSegment(0, 5, text, "s", words)],
                [DiarizationTurn("A", 0, 2), DiarizationTurn("B", 2, 5)], 0,
                context={"speaker_candidates": ["孟岩", "李继刚"]})
            self.assertEqual("".join(r.text for r in rows), text)
            self.assertTrue(all(r.speaker in {"A", "B", "unknown", "归属待核对"} for r in rows))
            if words != [TranscriptWord("世界", 2, 2)]:
                self.assertTrue(any(r.speaker in {"unknown", "归属待核对"} for r in rows))
        self.assertTrue(all(e["native_end_ms"] is None for r in rows for e in r.alignment_spans))

    def test_native_words_missing_or_invalid_times_do_not_gain_durations(self):
        words = parse_transcript_words([
            {"text": "甲", "start_ms": 100, "end_ms": None},
            {"text": "乙", "start_ms": 200, "end_ms": 200},
            {"text": "丙", "start_ms": 300, "end_ms": 250},
            {"text": "丁"}, {"text": "戊", "start": "garbage"},
            {"text": "己", "start": float("nan")},
        ], segment_start=0, segment_end=2)
        self.assertEqual([(w.text, w.start, w.end) for w in words], [("甲", .1, None), ("乙", .2, .2)])
        aligned, _ = align_segments_to_speakers(
            [TranscriptSegment(0, 1, "乙", "zero", [words[1]])],
            [DiarizationTurn("A", 0, 1)], 0,
        )
        self.assertEqual(aligned[0].speaker, "归属待核对")
        self.assertEqual(aligned[0].alignment_spans[0]["assignment_reason"], "missing_duration")

    def test_no_acoustic_backend_cannot_create_person_or_name_and_text_is_written(self):
        source = TranscriptSegment(0, 3, "大家好，我是李继刚。", "s")
        turns = run_timeline_fallback_diarization([source])
        self.assertEqual(turns, [])
        identities = DefaultSpeakerIdentityResolver().resolve(turns, [source], {"speaker_candidates": ["李继刚"]})
        self.assertEqual(identities.speakers, [])
        aligned, pending = align_segments_to_speakers([source], turns, 0)
        aligned, mapping = apply_identity_map_to_turns(aligned, identities)
        self.assertEqual(mapping, {})
        self.assertEqual(pending, [source])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_speaker_outputs(root, {"speaker_turns": [speaker_turn_to_dict(r) for r in aligned], "speakers": []})
            self.assertIn(source.text, (root / "transcript.speakers.txt").read_text())
            self.assertEqual(json.loads((root / "transcript.speakers.json").read_text())["speakers"], [])
        with patch("workbench.speaker_transcript.package_available", return_value=False), patch("workbench.speaker_transcript.find_hf_token", return_value=""):
            status = detect_speaker_backend_status()
        self.assertFalse(status["fallback_backend_available"])
        self.assertFalse(status["ready"])

    def test_full_introduction_must_be_same_voice_and_words_must_cover_prefix(self):
        base = {"id": "s", "text": "我是李继刚。", "start": 0, "end": 4}
        turns = [DiarizationTurn("A", 0, 1), DiarizationTurn("B", 1, 4)]
        words = [{"text": "我是", "start": 0, "end": .8}, {"text": "李继刚", "start": 1, "end": 2}]
        resolver = DefaultSpeakerIdentityResolver()
        result = resolver.resolve(turns, [{**base, "words": words}], {"speaker_candidates": ["李继刚"]})
        self.assertTrue(all(s.identity_status == "unknown" for s in result.speakers))
        # Even an otherwise uniform segment cannot supply a missing native prefix.
        for partial in [words[1:], [{"text": "幻觉", "start": 1, "end": 2}],
                        [{"text": "我是李继刚", "start": "garbage", "end": 2}]]:
            result = resolver.resolve([DiarizationTurn("B", 0, 4)], [{**base, "words": partial}], {"speaker_candidates": ["李继刚"]})
            self.assertEqual(result.get("B").identity_status, "unknown")
        # Full declaration in one voice is valid; no other candidate is inferred.
        words[0] = {"text": "我是", "start": 1, "end": 1.2}
        words[1] = {"text": "李继刚", "start": 1.2, "end": 2}
        result = resolver.resolve(turns, [{**base, "words": words}], {"speaker_candidates": ["孟岩", "李继刚"]})
        self.assertEqual(result.get("B").display_name, "李继刚")
        self.assertEqual(result.get("A").display_name, "unknown")
        self.assertEqual(result.get("B").identity_evidence[0].details["declaration_text_span"], (0, 5))

    def test_identity_unknown_retains_voice_in_both_paths_and_serialized_evidence(self):
        text = "我是李继刚。"
        tokens = [{"text": c, "start_ms": i * 200, "end_ms": None} for i, c in enumerate(text)]
        source = [{"id": "s", "text": text, "start_ms": 0, "end_ms": 2000}]
        measured = [{"speaker": "A", "start": 0, "end": .4}, {"speaker": "B", "start": .4, "end": 2}]
        data = build_dialogue_payload(source, source, tokens, measured, {"speaker_candidates": ["李继刚"]})
        self.assertEqual([r["speaker"] for r in data["transcription"]], ["说话人 1", "说话人 2"])
        self.assertTrue(all(r["identity_status"] == "unknown" for r in data["transcription"]))
        self.assertEqual(data["acoustic_activity"]["turns"], measured)
        legacy_source = TranscriptSegment(0, 2, text, "s", [TranscriptWord(t["text"], t["start_ms"] / 1000, None) for t in tokens])
        turns = [DiarizationTurn("A", 0, .4), DiarizationTurn("B", .4, 2)]
        rows, _ = align_segments_to_speakers([legacy_source], turns, 0)
        identities = DefaultSpeakerIdentityResolver().resolve(turns, [legacy_source], {"speaker_candidates": ["李继刚"]})
        rows, mapping = apply_identity_map_to_turns(rows, identities)
        self.assertFalse(mapping)
        self.assertEqual([r.speaker for r in rows], ["说话人 1", "说话人 2"])
        self.assertEqual("".join(r.text for r in rows), text)
        self.assertTrue(all(e["native_end_ms"] is None for r in rows for e in r.alignment_spans))
        with tempfile.TemporaryDirectory() as d:
            paths = write_dialogue_outputs(Path(d), data)
            saved = json.loads(Path(paths["dialogue_json_path"]).read_text())
            self.assertTrue(all(e["native_end_ms"] is None for r in saved["transcription"] for e in r["alignment_spans"]))
            self.assertEqual(saved["annotation_kind"], "text-review-envelope")


if __name__ == "__main__":
    unittest.main()
