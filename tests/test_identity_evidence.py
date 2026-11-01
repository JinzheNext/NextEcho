from __future__ import annotations

import unittest

from workbench.diarization import DiarizationTurn
from workbench.speaker_identity import DefaultSpeakerIdentityResolver, VerifiedVoiceMatch, detect_speaker_self_introduction
from workbench.speaker_transcript import render_speaker_text


class IdentityEvidenceTests(unittest.TestCase):
    def resolve(self, rows, turns=None, names=None):
        return DefaultSpeakerIdentityResolver().resolve(
            turns or [DiarizationTurn("A", 0, 15), DiarizationTurn("B", 15, 30)],
            rows, context={"speaker_candidates": names or ["孟岩", "李继刚"]},
        )

    def test_mentions_quotes_negations_and_name_prefixes_are_not_introductions(self):
        for text in ["他是李继刚。", "嘉宾叫李继刚。", "他说：我是李继刚。",
                     "张三说：我是李继刚。", "你说，我是李继刚。", "演员喊：我是李继刚。",
                     "她说，我是李继刚。", "比如，我是李继刚。", "“我是李继刚。”",
                     "我是李继刚的朋友。", "我是李继刚强。", "我不是李继刚。"]:
            with self.subTest(text=text):
                self.assertEqual(detect_speaker_self_introduction(text, ["李继刚"]), "")
        for text in ["大家好，我是李继刚。", "大家好我是李继刚", "我是李继刚。", "本人叫李继刚。"]:
            self.assertEqual(detect_speaker_self_introduction(text, ["李继刚"]), "李继刚")

    def test_moving_text_in_a_mixed_window_never_changes_cluster_identity(self):
        for text in ["我是李继刚。" + "讨论。" * 20, "讨论。" * 20 + "我是李继刚。"]:
            result = self.resolve([{"text": text, "start": 0, "end": 30}])
            self.assertTrue(all(s.identity_status == "unknown" for s in result.speakers))

    def test_whole_unambiguous_acoustic_segment_can_anchor_name(self):
        result = self.resolve([{"text": "大家好，我是李继刚。", "start": 15, "end": 30}])
        self.assertEqual(result.get("B").display_name, "李继刚")
        self.assertEqual(result.get("A").display_name, "unknown")
        self.assertEqual(result.to_dict()["confidence_kind"], "rule_evidence_score_not_probability")

    def test_native_word_anchor_uses_introduction_instead_of_earlier_mention(self):
        row = {"text": "他是李继刚。大家好，我是李继刚。", "start": 0, "end": 30,
               "words": [{"text": "他是李继刚。", "start": 1, "end": 2},
                         {"text": "大家好，我是", "start": 16, "end": 17},
                         {"text": "李", "start": 17, "end": 18},
                         {"text": "继刚。", "start": 18, "end": 19}]}
        result = self.resolve([row])
        self.assertEqual(result.get("A").display_name, "unknown")
        self.assertEqual(result.get("B").display_name, "李继刚")

    def test_overlapping_or_poorly_covered_voice_is_not_an_anchor(self):
        for turns in [[DiarizationTurn("A", 0, 30), DiarizationTurn("B", 10, 11)],
                      [DiarizationTurn("A", 0, 30, overlap=True)],
                      [DiarizationTurn("A", 0, 1)]]:
            result = self.resolve([{"text": "我是李继刚。", "start": 0, "end": 30}], turns)
            self.assertTrue(all(s.display_name == "unknown" for s in result.speakers))

    def test_native_onsets_without_durations_are_points_not_invented_word_ends(self):
        words = [{"text": "大家好我是", "start_ms": 16000, "end_ms": None},
                 {"text": "李", "start_ms": 17000, "end_ms": None},
                 {"text": "继刚", "start_ms": 18000, "end_ms": None}]
        row = {"text": "大家好我是李继刚", "start_ms": 0, "end_ms": 30000, "words": words}
        result = self.resolve([row])
        self.assertEqual(result.get("B").display_name, "李继刚")
        self.assertEqual(result.get("A").display_name, "unknown")
        self.assertIsNone(words[-1]["end_ms"])

    def test_zero_width_explicit_native_word_times_cannot_authorize_identity(self):
        words = [
            {"text": "大家好", "start": 16.0, "end": 16.0},
            {"text": "我是", "start": 17.0, "end": 17.0},
            {"text": "李继刚", "start": 18.0, "end": 18.0},
        ]
        result = self.resolve([{
            "text": "大家好我是李继刚。", "start": 0, "end": 30, "words": words,
        }], [DiarizationTurn("A", 0, 30)])
        self.assertEqual(result.get("A").identity_status, "unknown")
        self.assertEqual(result.get("A").display_name, "unknown")

    def test_conflicting_direct_introductions_remain_conflict(self):
        result = self.resolve([{"text": "我是孟岩。我是李继刚。", "start": 0, "end": 10}],
                              [DiarizationTurn("A", 0, 10)])
        self.assertEqual(result.get("A").identity_status, "conflict")
        self.assertEqual(result.get("A").display_name, "unknown")

    def test_arbitrary_profile_dict_and_unmeasured_threshold_never_resolve_identity(self):
        result = DefaultSpeakerIdentityResolver().resolve(
            [DiarizationTurn("A", 0, 10)], [], profiles=[{
                "raw_speaker_label": "A", "display_name": "孟岩", "confidence": 0.99,
                "threshold": 0.78, "verified": True,
            }],
        )
        self.assertEqual(result.get("A").identity_status, "unknown")
        with self.assertRaises(ValueError):
            VerifiedVoiceMatch("A", "孟岩", "p1", float("nan"), 0.78,
                "a" * 64, "b" * 64, "resemblyzer/v1", "c" * 64, "recording-1")

    def test_typed_content_bound_match_records_actual_score_and_threshold(self):
        match = VerifiedVoiceMatch(
            "A", "孟岩", "p1", 0.91, 0.83, "a" * 64, "b" * 64,
            "encoder/v1", "c" * 64, "recording-1",
        )
        result = DefaultSpeakerIdentityResolver().resolve(
            [DiarizationTurn("A", 0, 10)], [], profiles=[match]
        )
        identity = result.get("A")
        self.assertEqual(identity.identity_status, "resolved")
        self.assertEqual(identity.identity_evidence[0].details["score"], 0.91)
        self.assertEqual(identity.identity_evidence[0].details["threshold"], 0.83)

    def test_rendering_preserves_financial_words_and_acknowledgement(self):
        text = "金额36万元，额外费用不能超过限额。哈佛也有这个例子。"
        rows = [{"speaker": "说话人 1", "start": 0, "end": 5, "text": text},
                {"speaker": "说话人 2", "start": 5, "end": 6, "text": "嗯。"}]
        for timestamps in [False, True]:
            output = render_speaker_text(rows, with_timestamps=timestamps)
            self.assertIn(text, output)
            self.assertIn("嗯。", output)
        self.assertEqual(rows[0]["text"], text)


if __name__ == "__main__":
    unittest.main()
