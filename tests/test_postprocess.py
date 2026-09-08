from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.enhancement import default_glossary_rules, strip_filler_words
from workbench.postprocess import Segment, context_keywords, merge_dialogue_segments, process_transcript_artifacts, render_dialogue_line, validate_polish_response


def final_summary() -> dict:
    return {
        "title": "测试节目",
        "brief_summary": "讨论 NextEcho 的本地转录。",
        "detailed_summary": "节目说明了本地优先的整理流程。",
        "key_points": [{"text": "保留原始稿", "segment_ids": ["seg_00000"]}],
        "chapters": [{"text": "开始", "segment_ids": ["seg_00000"]}],
        "people_organizations_tools": [{"name": "NextEcho", "segment_ids": ["seg_00000"]}],
        "action_items": [],
        "open_questions": [],
        "keywords": ["NextEcho"],
        "notable_quotes": [{"quote": "NextEcho 在 2026 年发布。", "segment_ids": ["seg_00000"]}],
    }


class PostprocessTests(unittest.TestCase):
    def test_removes_fillers_and_keeps_dialogue_speaker(self) -> None:
        self.assertEqual(strip_filler_words("嗯嗯大家好呃，哈"), "大家好，")
        self.assertEqual(render_dialogue_line(Segment("seg", "嗯大家好。", 0, 1000, "孟岩")), "孟岩：大家好。")

    def test_merges_same_speaker_segments_for_readable_dialogue(self) -> None:
        merged = merge_dialogue_segments(
            [
                Segment("seg_00000", "跟", 0, 800, "孟岩"),
                Segment("seg_00001", "纪刚录博客。", 850, 1800, "孟岩"),
                Segment("seg_00002", "大家好。", 2000, 2600, "李继刚"),
            ]
        )
        self.assertEqual([(item.speaker, item.text) for item in merged], [("孟岩", "跟纪刚录博客。"), ("李继刚", "大家好。")])

    def test_context_keywords_exclude_fillers(self) -> None:
        keywords = context_keywords(
            {
                "shownotes": {"text": "AI 时代的人机协作与信息输入。"},
                "comments": [{"text": "嗯，AI 时代更需要心力。"}, {"text": "人机协作值得长期讨论。"}],
            }
        )
        self.assertIn("人机", keywords)
        self.assertNotIn("嗯", keywords)

    def test_polish_validator_preserves_numbers_and_protected_terms(self) -> None:
        source = [Segment("seg_00000", "NextEcho 在 2026 年发布。", 0, 1000)]
        rules = default_glossary_rules("protect: NextEcho")
        with self.assertRaises(ValueError):
            validate_polish_response(
                source,
                {"segments": [{"id": "seg_00000", "polished_text": "NextEcho 已发布。"}]},
                rules,
            )

    def test_creates_parallel_polished_and_knowledge_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "transcription": [
                            {
                                "id": "seg_00000",
                                "text": "NextEcho 在 2026 年发布。",
                                "offsets": {"from": 0, "to": 2500},
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            def fake_call_json(**kwargs):
                schema = kwargs["schema"]
                if "segments" in schema["properties"]:
                    return ({"segments": [{"id": "seg_00000", "polished_text": "NextEcho 在 2026 年发布。", "uncertain_terms": [], "change_notes": []}]}, "qwen3:4b", [])
                if "summary" in schema["properties"]:
                    return ({"summary": "本地转录。", "key_points": [{"text": "本地处理", "segment_ids": ["seg_00000"]}], "segment_ids": ["seg_00000"]}, "qwen3:4b", [])
                return (final_summary(), "qwen3.5:9b-q4_K_M", [])

            with patch("workbench.postprocess.call_json", side_effect=fake_call_json), patch("workbench.postprocess.unload_model"):
                audit = process_transcript_artifacts(root, profile="balanced", user_glossary_text="protect: NextEcho")

            self.assertEqual(audit["status"], "ready")
            self.assertTrue((root / "transcript.enhanced.txt").exists())
            self.assertTrue((root / "transcript.enhanced.json").exists())
            self.assertTrue((root / "transcript.enhanced.srt").exists())
            self.assertTrue((root / "transcript.enhanced.vtt").exists())
            self.assertTrue((root / "transcript.summary.json").exists())
            self.assertTrue((root / "transcript.knowledge.md").exists())
            self.assertIn("NextEcho 在 2026 年发布。", (root / "transcript.enhanced.txt").read_text(encoding="utf-8"))
            self.assertIn("测试节目", (root / "transcript.knowledge.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
