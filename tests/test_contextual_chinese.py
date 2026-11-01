"""Preserve speech, raw bytes and provenance while adding useful Chinese hints."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.chinese_text import simplified, write_simplified_artifacts
from workbench.enhancement import default_glossary_rules
from workbench.episode_context import _top_comments, split_shownotes_sections
from workbench.llm_providers import CONTEXT, GenerationContext
from workbench.polish_context import (
    asr_hotwords,
    retrieve_context,
    validate_spoken_forms,
)
from workbench.postprocess import (
    Segment,
    artifact_fingerprint,
    ensure_preserved,
    polish_segments,
    merge_dialogue_segments,
    polish_schema,
    render_dialogue_line,
    _polish_text,
    infer_transcript_coverage,
)


class ContextualChineseTests(unittest.TestCase):
    def test_whole_audio_coverage_includes_silent_edges_without_fake_speech(self):
        with tempfile.TemporaryDirectory() as folder:
            segments = [Segment("a", "最后一句。", 5000, 940000)]
            raw = {"coverage": {"covered_seconds": 1000.125, "partial": False}}
            source = {"episode": {"duration_seconds": 1000}}
            whole = infer_transcript_coverage(
                Path(folder), raw, segments, source_context=source
            )
            self.assertFalse(whole.partial)
            self.assertEqual(whole.start_ms, 0)
            self.assertEqual(whole.effective_end_ms, 1000125)
            self.assertEqual(
                whole.end_ms, 940000
            )  # measured speech end remains separate
            limited = infer_transcript_coverage(
                Path(folder), raw, segments, source_context=source, max_seconds=300
            )
            self.assertTrue(limited.partial)
            self.assertEqual(limited.effective_end_ms, 300000)

    def test_hesitation_cleanup_cannot_corrupt_financial_words_or_names(self):
        token = CONTEXT.set(
            GenerationContext(
                {
                    "contextual_polish": True,
                    "editing_mode": "readable-transcript",
                    "preserve_standalone_acknowledgements": True,
                }
            )
        )
        text = "金额36万元，额外费用不能超过限额。哈佛的案例。嗯，继续。"
        try:
            normalized = _polish_text(text, default_glossary_rules())
            rendered = render_dialogue_line(Segment("a", normalized, 0, 30000))
            for term in ["金额", "额外", "限额", "哈佛", "36万元", "不能"]:
                self.assertIn(term, normalized)
                self.assertIn(term, rendered)
            self.assertIn("嗯，继续。", rendered)
            self.assertEqual(render_dialogue_line(Segment("ack", "嗯", 0, 1000)), "嗯")
        finally:
            CONTEXT.reset(token)

    def test_compact_schema_and_nonspeech_presentation_match_the_prompt(self):
        token = CONTEXT.set(GenerationContext({"contextual_polish": True}))
        try:
            item = polish_schema()["properties"]["segments"]["items"]
            self.assertEqual(set(item["properties"]), {"id", "polished_text"})
            self.assertEqual(item["required"], ["id", "polished_text"])
            self.assertEqual(
                render_dialogue_line(Segment("noise", "[noise]", 0, 3000)), ""
            )
            self.assertEqual(
                render_dialogue_line(Segment("speech", "noise指噪声。", 0, 3000)),
                "noise指噪声。",
            )
        finally:
            CONTEXT.reset(token)

    def test_editorial_target_is_hidden_and_cannot_replace_spoken_form(self):
        context = self.context()
        source = [Segment("x", "你说的失控玩家，我也想到了。", 20000, 25000)]
        supplied = retrieve_context(context, source)
        editorial = next(
            h
            for h in supplied["hints"]
            if h["kind"] == "publisher_editorial_correction"
        )
        self.assertEqual(editorial["spoken_form"], "失控玩家")
        self.assertNotIn("头号玩家", editorial["text"])
        with self.assertRaisesRegex(ValueError, "编者勘误"):
            validate_spoken_forms(
                source,
                [Segment("x", "你说的头号玩家，我也想到了。", 20000, 25000)],
                context,
                supplied,
            )
        validate_spoken_forms(source, source, context, supplied)

    def test_local_glossary_candidates_are_available_even_when_asr_misspells_them(self):
        supplied = retrieve_context(
            self.context(), [Segment("x", "贝叶丝这个定理。", 10000, 18000)]
        )
        self.assertIn("贝叶斯定理", [t["term"] for t in supplied["terminology"]])

    def test_editorial_retry_receives_feedback_without_more_than_two_attempts(self):
        token = CONTEXT.set(
            GenerationContext(
                {"contextual_polish": True, "editing_mode": "readable-transcript"}
            )
        )
        source = [Segment("x", "你说的失控玩家，我也想到了。", 20000, 25000)]
        answers = [
            ({"segments": [{"id": "x", "polished_text": text}]}, "4B", [])
            for text in ["你说的头号玩家，我也想到了。", source[0].text]
        ]
        try:
            with (
                tempfile.TemporaryDirectory() as folder,
                patch("workbench.postprocess.call_json", side_effect=answers) as call,
            ):
                result, audit = polish_segments(
                    source,
                    rules=default_glossary_rules(),
                    models=["4B"],
                    profile="balanced",
                    style="",
                    endpoint="http://127.0.0.1:11435",
                    checkpoint_dir=Path(folder),
                    source_context=self.context(),
                )
            self.assertEqual(call.call_count, 2)
            retry = json.loads(call.call_args_list[1].kwargs["user"].split("\n", 1)[1])
            self.assertIn("编者勘误", retry["validation_feedback"])
            self.assertEqual(result, source)
            self.assertEqual(audit["status"], "ready")
        finally:
            CONTEXT.reset(token)

    def test_unassigned_voices_do_not_become_one_episode_length_paragraph(self):
        token = CONTEXT.set(GenerationContext({"contextual_polish": True}))
        try:
            rows = [
                Segment("a", "第一段。", 0, 20000),
                Segment("b", "下一段。", 20000, 40000),
            ]
            self.assertEqual(merge_dialogue_segments(rows), rows)
        finally:
            CONTEXT.reset(token)

    def context(self):
        return {
            "speaker_candidates": [{"name": "李继刚", "source": "shownotes"}],
            "shownotes": {
                "text": "李继刚",
                "sections": split_shownotes_sections(
                    "00:10 贝叶斯定理（Bayes）：概率更新。\n00:20 此处提及的「失控玩家」实为电影《头号玩家》。"
                ),
            },
            "comments": [
                {
                    "context_ref": "comment:1",
                    "text": "贝叶斯定理的概率更新很有意思。",
                    "replies": [
                        {"context_ref": "comment:2", "text": "我也听到了贝叶斯定理。"}
                    ],
                }
            ],
        }

    def test_t2s_preserves_literals_and_distinguishes_raw(self):
        text = "學習繁體，網址https://example.org/學習 和 `學習 = 1`。"
        self.assertEqual(
            simplified(text), "学习繁体，网址https://example.org/學習 和 `學習 = 1`。"
        )
        payload = {
            "transcription": [
                {
                    "id": "a",
                    "text": "學習需要兩萬元。",
                    "offsets": {"from": 0, "to": 3000},
                    "speaker": "甲",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = json.dumps(payload, ensure_ascii=False).encode()
            (root / "transcript.json").write_bytes(raw)
            paths = write_simplified_artifacts(root, payload)
            self.assertEqual((root / "transcript.json").read_bytes(), raw)
            self.assertTrue(all(Path(p).is_file() for p in paths.values()))
            norm = json.loads(Path(paths["normalized_json_path"]).read_text())[
                "transcription"
            ][0]
            self.assertEqual(norm["text"], "学习需要两万元。")
            self.assertEqual(norm["offsets"], payload["transcription"][0]["offsets"])
            self.assertEqual(norm["speaker"], "甲")

    def test_restarting_and_duplicate_timestamps_have_positive_spans(self):
        sections = split_shownotes_sections(
            "10:00 时间线末尾\n00:05 术语1\n00:05 术语2\n00:08 下一项"
        )
        self.assertTrue(
            all(s["end_ms"] is None or s["end_ms"] > s["start_ms"] for s in sections)
        )
        self.assertEqual(
            [s["context_ref"] for s in sections],
            ["shownotes:0", "shownotes:1", "shownotes:2", "shownotes:3"],
        )
        self.assertEqual(sections[1]["end_ms"], 8000)

    def test_capture_retains_reply_preview_with_context_only_marker(self):
        rows = _top_comments(
            [
                {
                    "id": "a",
                    "text": "主评论",
                    "replyCount": 15,
                    "replies": [{"id": "b", "text": "回复"}],
                }
            ],
            limit=20,
        )
        self.assertEqual(rows[0]["reply_count"], 15)
        self.assertEqual(rows[0]["replies"][0]["context_ref"], "comment:b")
        self.assertFalse(rows[0]["replies"][0]["evidence_eligible"])

    def test_context_retrieval_supplies_editorial_labels_and_comments(self):
        hints = retrieve_context(
            self.context(),
            [Segment("a", "李记刚讲贝叶斯定理的概率更新。", 10000, 25000, "甲")],
            max_chars=500,
        )
        self.assertIn("李继刚", [t["term"] for t in hints["terminology"]])
        self.assertTrue(any(h["kind"] == "comment_hint" for h in hints["hints"]))
        self.assertTrue(
            any(h["kind"] == "publisher_editorial_correction" for h in hints["hints"])
        )
        self.assertLessEqual(sum(len(h["text"]) for h in hints["hints"]), 500)
        self.assertTrue(
            all(
                not h["transcript_evidence"] and h["context_ref"]
                for h in hints["hints"]
            )
        )
        hotwords = asr_hotwords(self.context())
        self.assertIn("李继刚", hotwords)
        self.assertIn("贝叶斯定理", hotwords)
        self.assertNotIn("头号玩家", hotwords)

    def test_actual_polish_request_contains_context_and_preserves_raw(self):
        token = CONTEXT.set(
            GenerationContext(
                {
                    "contextual_polish": True,
                    "output_script": "simplified",
                    "editing_mode": "readable-transcript",
                    "context_tokens": 16384,
                    "max_output_tokens": 2048,
                }
            )
        )
        try:
            source = [Segment("a", "李记刚讲贝叶斯定理。", 10000, 18000, "甲")]
            with (
                tempfile.TemporaryDirectory() as folder,
                patch(
                    "workbench.postprocess.call_json",
                    return_value=(
                        {
                            "segments": [
                                {"id": "a", "polished_text": "李继刚讲贝叶斯定理。"}
                            ]
                        },
                        "4B",
                        [],
                    ),
                ) as call,
            ):
                output, audit = polish_segments(
                    source,
                    rules=default_glossary_rules(),
                    models=["4B"],
                    profile="balanced",
                    style="",
                    endpoint="http://127.0.0.1:11435",
                    checkpoint_dir=Path(folder),
                    source_context=self.context(),
                )
            request = json.loads(call.call_args.kwargs["user"].split("\n", 1)[1])
            self.assertTrue(request["source_context"]["hints"])
            self.assertEqual(output[0].text, "李继刚讲贝叶斯定理。")
            self.assertEqual(source[0].text, "李记刚讲贝叶斯定理。")
            self.assertTrue(audit["context_requests"])
            ensure_preserved(
                "預算兩萬元，不能超支。",
                "预算两万元，不能超支。",
                default_glossary_rules(),
            )
            with self.assertRaises(ValueError):
                ensure_preserved("不能超支。", "可以超支。", default_glossary_rules())
        finally:
            CONTEXT.reset(token)

    def test_reply_context_change_invalidates_final_artifact_cache(self):
        token = CONTEXT.set(GenerationContext({"contextual_polish": True}))
        try:
            c = self.context()
            rules = default_glossary_rules()
            a = artifact_fingerprint({"transcription": []}, rules, "balanced", "", c)
            c["comments"][0]["replies"][0]["text"] = "修正术语拼写"
            self.assertNotEqual(
                a, artifact_fingerprint({"transcription": []}, rules, "balanced", "", c)
            )
        finally:
            CONTEXT.reset(token)


if __name__ == "__main__":
    unittest.main()
