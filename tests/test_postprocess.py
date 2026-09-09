from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from workbench.cli import cmd_enhance
from workbench.enhancement import default_glossary_rules, strip_filler_words
from workbench.postprocess import (
    POLISH_MAX_CHARS,
    POLISH_MAX_SEGMENTS,
    POLISH_TARGET_CHARS,
    Segment,
    TranscriptCoverage,
    chunk_segments,
    context_keywords,
    context_reference_inventory,
    merge_dialogue_segments,
    polish_segments,
    process_transcript_artifacts,
    render_dialogue_line,
    render_knowledge_markdown,
    resolve_postprocess_models,
    scope_source_context,
    sanitize_summary_references,
    validate_polish_response,
    validate_references,
)


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

    def test_polish_does_not_send_empty_filler_segments_to_model(self) -> None:
        response = {
            "segments": [
                {
                    "id": "seg_text",
                    "polished_text": "大家好。",
                    "uncertain_terms": [],
                    "change_notes": [],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "workbench.postprocess.call_json", return_value=(response, "qwen3:4b", [])
        ) as call:
            polished, audit = polish_segments(
                [Segment("seg_filler", "嗯嗯嗯", 0, 1000, "unknown"), Segment("seg_text", "大家好。", 1000, 2000, "Speaker 1")],
                rules=default_glossary_rules(""),
                models=["qwen3:4b"],
                profile="fast",
                style="",
                endpoint="http://127.0.0.1:11434",
                checkpoint_dir=Path(temp_dir),
            )
        sent = json.loads(call.call_args.kwargs["user"].split("\n", 1)[1])
        self.assertEqual([item["id"] for item in sent["segments"]], ["seg_text"])
        self.assertEqual([item.id for item in polished], ["seg_filler", "seg_text"])
        self.assertEqual(polished[0].text, "")
        self.assertEqual(audit["status"], "ready")

    def test_merges_same_speaker_segments_for_readable_dialogue(self) -> None:
        merged = merge_dialogue_segments(
            [
                Segment("seg_00000", "跟", 0, 800, "孟岩"),
                Segment("seg_00001", "纪刚录博客。", 850, 1800, "孟岩"),
                Segment("seg_00002", "大家好。", 2000, 2600, "李继刚"),
            ]
        )
        self.assertEqual([(item.speaker, item.text) for item in merged], [("孟岩", "跟纪刚录博客。"), ("李继刚", "大家好。")])

    def test_knowledge_index_uses_dialogue_blocks_without_internal_ids(self) -> None:
        markdown = render_knowledge_markdown(
            final_summary(),
            [
                Segment("seg_00000", "嗯", 0, 500, "孟岩"),
                Segment("seg_00001", "大家好。", 500, 1500, "孟岩"),
                Segment("seg_00002", "我是李继刚。", 1600, 2500, "李继刚"),
            ],
        )
        self.assertIn("## 对话索引", markdown)
        self.assertNotIn("seg_00001", markdown)
        self.assertNotIn("：\n", markdown)

    def test_context_keywords_exclude_fillers(self) -> None:
        keywords = context_keywords(
            {
                "shownotes": {"text": "AI 时代的人机协作与信息输入。"},
                "comments": [{"text": "嗯，AI 时代更需要心力。"}, {"text": "人机协作值得长期讨论。"}],
            }
        )
        self.assertIn("人机", keywords)
        self.assertNotIn("嗯", keywords)

    def test_partial_context_excludes_later_shownotes_and_marks_comments(self) -> None:
        context = {
            "episode": {"duration_seconds": 3600},
            "shownotes": {"text": "00:00 前五分钟。\n05:00 后半段事实。\n20:00 更后段事实。"},
            "comments": [{"id": "c1", "text": "评论里的事实说法"}],
        }
        scoped = scope_source_context(
            context,
            TranscriptCoverage(end_ms=299000, limit_end_ms=300000, max_seconds=300, episode_duration_seconds=3600, partial=True),
        )
        self.assertIn("前五分钟", scoped["shownotes"]["text"])
        self.assertNotIn("后半段事实", scoped["shownotes"]["text"])
        self.assertNotIn("更后段事实", scoped["shownotes"]["text"])
        self.assertFalse(scoped["comments"][0]["evidence_eligible"])
        self.assertTrue(scoped["comments"][0]["context_only"])
        refs, types = context_reference_inventory(scoped)
        with self.assertRaises(ValueError):
            validate_references(
                {"key_points": [{"text": "评论事实", "context_refs": ["comment:c1"]}]},
                {"seg_00000"},
                known_context_refs=refs,
                context_ref_types=types,
            )

    def test_invalid_summary_claim_is_dropped_without_losing_valid_claims(self) -> None:
        cleaned, dropped = sanitize_summary_references(
            {
                "title": "测试",
                "key_points": [
                    {"text": "合法", "segment_ids": ["seg_0"]},
                    {"text": "越界", "segment_ids": ["seg_999"]},
                ],
                "notable_quotes": [],
            },
            {"seg_0"},
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(cleaned["title"], "测试")
        self.assertEqual(cleaned["key_points"], [{"text": "合法", "segment_ids": ["seg_0"]}])

    def test_chinese_keyword_fallback_avoids_garbage_bigrams(self) -> None:
        keywords = context_keywords(
            {
                "shownotes": {"text": "知有 入议 李继 经元 人机协作与生成式人工智能。"},
                "comments": [],
            }
        )
        self.assertIn("人机", keywords)
        self.assertIn("生成式人工智能", keywords)
        for garbage in ("知有", "入议", "李继", "经元"):
            self.assertNotIn(garbage, keywords)

    def test_polish_chunks_are_large_and_keep_speaker_boundaries(self) -> None:
        segments = [
            Segment("a0", "甲" * 700 + "。", 0, 1000, "A"),
            Segment("a1", "乙" * 700 + "。", 1000, 2000, "A"),
            Segment("b0", "丙" * 700 + "。", 2000, 3000, "B"),
            Segment("b1", "丁" * 700 + "。", 3000, 4000, "B"),
        ]
        chunks = chunk_segments(segments, max_chars=POLISH_MAX_CHARS, target_chars=POLISH_TARGET_CHARS)
        self.assertEqual([[item.id for item in chunk] for chunk in chunks], [["a0", "a1"], ["b0", "b1"]])
        self.assertTrue(all(sum(len(item.text) for item in chunk) <= POLISH_MAX_CHARS for chunk in chunks))
        self.assertEqual([item.id for chunk in chunks for item in chunk], [item.id for item in segments])

    def test_polish_chunks_cap_structured_segment_count_without_splitting_segments(self) -> None:
        segments = [Segment(f"seg_{index:02d}", "完整短句。", index * 1000, (index + 1) * 1000, "A") for index in range(26)]
        chunks = chunk_segments(
            segments,
            max_chars=POLISH_MAX_CHARS,
            target_chars=POLISH_TARGET_CHARS,
            max_segments=POLISH_MAX_SEGMENTS,
        )
        self.assertEqual([len(chunk) for chunk in chunks], [12, 12, 2])
        self.assertEqual([item.id for chunk in chunks for item in chunk], [item.id for item in segments])

    def test_polish_validator_rejects_immutable_speaker_and_time_changes(self) -> None:
        source = [Segment("seg", "原始内容。", 100, 200, "spk_0")]
        with self.assertRaises(ValueError):
            validate_polish_response(
                source,
                {"segments": [{"id": "seg", "speaker": "spk_1", "start_ms": 100, "end_ms": 200, "polished_text": "原始内容。", "uncertain_terms": [], "change_notes": []}]},
                default_glossary_rules(),
            )

    def test_polish_validator_rejects_large_meaningful_deletion(self) -> None:
        source = [Segment("seg", "她说你有没有意识到你很喜欢纪刚，我说是吗，这你都看出来了。", 100, 200, "孟岩")]
        with self.assertRaisesRegex(ValueError, "retention too low"):
            validate_polish_response(
                source,
                {"segments": [{"id": "seg", "polished_text": "了。", "uncertain_terms": [], "change_notes": []}]},
                default_glossary_rules(),
            )
        with self.assertRaises(ValueError):
            validate_polish_response(
                source,
                {"segments": [{"id": "seg", "speaker": "spk_0", "start_ms": 101, "end_ms": 200, "polished_text": "原始内容。", "uncertain_terms": [], "change_notes": []}]},
                default_glossary_rules(),
            )

    def test_model_selection_falls_back_when_preferred_tag_is_missing(self) -> None:
        config = {
            "postprocess": {
                "provider": "selection",
                "preferred_model_id": "gemma-e4b-iq4xs",
                "fallback_model": "qwen3:4b",
                "quality_model": "qwen3.5:9b-q4_K_M",
            }
        }
        tags = {"ready": True, "models": [{"name": "qwen3:4b", "digest": "sha256:qwen"}]}
        fast = resolve_postprocess_models("fast", config=config, ollama_info=tags)
        balanced = resolve_postprocess_models("balanced", config=config, ollama_info=tags)
        quality = resolve_postprocess_models("quality", config=config, ollama_info=tags)
        self.assertEqual(fast["selected_4b"], "qwen3:4b")
        self.assertEqual(fast["summary_models"], ["qwen3:4b"])
        self.assertEqual(balanced["summary_models"], ["qwen3:4b"])
        self.assertEqual(quality["summary_models"], ["qwen3:4b"])

    def test_model_selection_uses_configured_4b_and_quality_only_when_explicit(self) -> None:
        config = {
            "postprocess": {
                "provider": "selection",
                "preferred_model_id": "gemma-e4b-iq4xs",
                "fallback_model": "qwen3:4b",
                "quality_model": "qwen3.5:9b-q4_K_M",
            }
        }
        tags = {
            "ready": True,
            "models": [
                {"name": "gemma-e4b-iq4xs", "digest": "sha256:gemma"},
                {"name": "qwen3.5:9b-q4_K_M", "digest": "sha256:9b"},
            ],
        }
        fast = resolve_postprocess_models("fast", config=config, ollama_info=tags)
        quality = resolve_postprocess_models("quality", config=config, ollama_info=tags)
        self.assertEqual(fast["polish_models"], ["gemma-e4b-iq4xs"])
        self.assertEqual(fast["summary_models"], ["gemma-e4b-iq4xs"])
        self.assertEqual(quality["summary_models"], ["qwen3.5:9b-q4_K_M", "gemma-e4b-iq4xs"])

    def test_llama_cpp_provider_is_explicitly_unsupported(self) -> None:
        with self.assertRaises(NotImplementedError):
            resolve_postprocess_models("fast", config={"postprocess": {"provider": "llama.cpp"}})

    def test_cli_enhance_passes_config_to_postprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "transcript.json").write_text("{}", encoding="utf-8")
            config = object()
            args = Namespace(
                run_dir=root,
                profile="fast",
                glossary=None,
                ai_style="",
                json=True,
                config_path="/tmp/nextecho-test.toml",
            )
            with patch("workbench.cli.config_for_args", return_value=config) as config_for_args, patch(
                "workbench.cli.process_transcript_artifacts", return_value={"status": "ready", "paths": {}}
            ) as process:
                cmd_enhance(args)

            config_for_args.assert_called_once_with(args)
            self.assertIs(process.call_args.kwargs["config"], config)

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
