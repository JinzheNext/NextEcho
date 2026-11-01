"""Publisher corrections, low-trust comments and explicit crawler choice."""

import hashlib
import io
import json
from email.message import Message
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workbench.episode_context import (
    _fetch_page,
    crawl_episode_context,
    split_shownotes_sections,
)
from workbench.enhancement import GlossaryRules, default_glossary_rules
from workbench.io_utils import file_hash, write_json
from workbench.llm_providers import CONTEXT, GenerationContext
from workbench.polish_context import (
    editorial_notes,
    parse_editorial_text,
    retrieve_context,
    asr_hotwords,
    validate_spoken_forms,
)
from workbench import postprocess as P


class Response(io.BytesIO):
    def __init__(self, text):
        super().__init__(text.encode())
        self.headers = Message()
        self.headers["Content-Type"] = "text/html; charset=utf-8"


def context(marker="实为", quote=("「", "」")):
    text = f"00:20 此处提及的{quote[0]}失控玩家{quote[1]}{marker}电影《头号玩家》。"
    return {"shownotes": {"text": text, "sections": split_shownotes_sections(text)}}


class SourceContextPolicyTests(unittest.TestCase):
    def setUp(self):
        self.source = [P.Segment("x", "我想到了失控玩家这部电影。", 20000, 25000)]
        self.wrong = [P.Segment("x", "我想到了头号玩家这部电影。", 20000, 25000)]

    def test_all_reported_markers_and_quote_styles_share_one_guard(self):
        for marker in ("实为", "应为", "应是", "实际为"):
            for quotes in [
                ("「", "」"),
                ("《", "》"),
                ("“", "”"),
                ("‘", "’"),
                ('"', '"'),
                ("'", "'"),
            ]:
                with self.subTest(marker=marker, quotes=quotes):
                    c = context(marker, quotes)
                    supplied = retrieve_context(c, self.source)
                    note = editorial_notes(c)[0]
                    self.assertEqual(
                        (note["marker"], note["spoken_form"], note["editorial_form"]),
                        (marker, "失控玩家", "头号玩家"),
                    )
                    self.assertEqual(note["parse_status"], "parsed")
                    self.assertFalse(note["transcript_evidence"])
                    self.assertNotIn(
                        "头号玩家", json.dumps(supplied, ensure_ascii=False)
                    )
                    self.assertNotIn("头号玩家", asr_hotwords(c))
                    with self.assertRaisesRegex(ValueError, "编者勘误"):
                        validate_spoken_forms(self.source, self.wrong, c, supplied)
                    validate_spoken_forms(self.source, self.source, c, supplied)

    def test_budget_removal_or_forged_supplied_hint_cannot_disable_guard(self):
        c = context("应是")
        empty = retrieve_context(c, self.source, max_chars=0)
        self.assertEqual(empty["hints"], [])
        for supplied in (
            empty,
            {"hints": [{"context_ref": "shownotes:0", "spoken_form": "other"}]},
        ):
            with self.assertRaises(ValueError):
                validate_spoken_forms(self.source, self.wrong, c, supplied)

    def test_original_target_is_allowed_and_one_character_misspelling_is_protected(
        self,
    ):
        c = context()
        legitimate = [P.Segment("x", "失控玩家与头号玩家都提到了。", 20000, 25000)]
        validate_spoken_forms(legitimate, legitimate, c, {})
        misspelling = [P.Segment("x", "我想到了失空玩家这部电影。", 20000, 25000)]
        with self.assertRaises(ValueError):
            validate_spoken_forms(misspelling, self.wrong, c, {})
        validate_spoken_forms(misspelling, self.source, c, {})

    def test_unresolved_editorial_is_withheld_and_untimed_note_is_not_lost(self):
        text = "00:20 此处有勘误，实际为电影《头号玩家》。"
        c = {"shownotes": {"text": text, "sections": split_shownotes_sections(text)}}
        notes = editorial_notes(c)
        self.assertEqual(notes[0]["parse_status"], "unresolved")
        supplied = retrieve_context(c, self.source)
        self.assertNotIn("头号玩家", json.dumps(supplied, ensure_ascii=False))
        c = {
            "shownotes": {
                "text": "此处提及的「失控玩家」应为电影《头号玩家》。",
                "sections": [],
            }
        }
        with self.assertRaises(ValueError):
            validate_spoken_forms(self.source, self.wrong, c, {})

    def test_multiple_corrections_preserve_separate_typed_evidence_and_text_hash(self):
        text = "此处提及的「甲书」应为《乙书》，说的「丙书」实际为《丁书》。"
        rows = parse_editorial_text(text, "shownotes:0")
        self.assertEqual(
            [(r.spoken_form, r.editorial_form) for r in rows],
            [("甲书", "乙书"), ("丙书", "丁书")],
        )
        self.assertTrue(
            all(
                r.source_text_sha256 == hashlib.sha256(text.encode()).hexdigest()
                for r in rows
            )
        )

    def test_comment_flags_cannot_make_comment_an_asr_or_identity_authority(self):
        c = context()
        c["comments"] = [
            {
                "context_ref": "comment:x",
                "text": "失控玩家这部电影与作者说的观点有关，有人认为应改成头号玩家。",
                "evidence_eligible": True,
                "transcript_evidence": True,
                "speaker_identity_evidence": True,
            }
        ]
        supplied = retrieve_context(c, self.source)
        hints = [h for h in supplied["hints"] if h["kind"] == "comment_hint"]
        self.assertTrue(hints)
        self.assertTrue(
            all(
                h["trust"] == "unverified-comment"
                and not h["transcript_evidence"]
                and not h["speaker_identity_evidence"]
                for h in hints
            )
        )
        self.assertNotIn("作者", asr_hotwords(c))
        self.assertNotIn("头号玩家", json.dumps(supplied, ensure_ascii=False))

    def test_polish_modes_reject_target_then_adopt_speech_and_keep_separate_notes(self):
        for contextual in (False, True):
            token = CONTEXT.set(
                GenerationContext(
                    {"contextual_polish": contextual, "editing_mode": "minimal"}
                )
            )
            try:
                answers = [
                    ({"segments": [{"id": "x", "polished_text": s[0].text}]}, "4B", [])
                    for s in [self.wrong, self.source]
                ]
                with (
                    tempfile.TemporaryDirectory() as d,
                    patch(
                        "workbench.postprocess.call_json", side_effect=answers
                    ) as call,
                ):
                    result, audit = P.polish_segments(
                        self.source,
                        rules=default_glossary_rules(),
                        models=["4B"],
                        profile="balanced",
                        style="",
                        endpoint="http://127.0.0.1:1",
                        checkpoint_dir=Path(d),
                        source_context=context("应是"),
                    )
                self.assertEqual(call.call_count, 2)
                self.assertEqual(result, self.source)
                self.assertEqual(
                    audit["publisher_editorial_notes"][0]["editorial_form"], "头号玩家"
                )
                sent = json.loads(
                    call.call_args_list[0].kwargs["user"].split("\n", 1)[1]
                )
                self.assertNotIn("头号玩家", json.dumps(sent, ensure_ascii=False))
            finally:
                CONTEXT.reset(token)

    def test_deterministic_glossary_cannot_turn_editorial_note_into_speech(self):
        token = CONTEXT.set(GenerationContext({"contextual_polish": False}))
        try:
            with (
                tempfile.TemporaryDirectory() as d,
                patch(
                    "workbench.postprocess.call_json",
                    return_value=(
                        {
                            "segments": [
                                {"id": "x", "polished_text": self.source[0].text}
                            ]
                        },
                        "4B",
                        [],
                    ),
                ),
            ):
                result, audit = P.polish_segments(
                    self.source,
                    rules=GlossaryRules(
                        replacements=[("失控玩家", "头号玩家")], protected=[]
                    ),
                    models=["4B"],
                    profile="balanced",
                    style="",
                    endpoint="http://127.0.0.1:1",
                    checkpoint_dir=Path(d),
                    source_context=context(),
                )
            self.assertEqual(result, self.source)
            self.assertTrue(audit["normalization_errors"])
        finally:
            CONTEXT.reset(token)

    def test_candidate_and_adopted_caches_recheck_same_typed_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            candidate = root / "candidate.json"
            write_json(
                candidate,
                {
                    "input_segments": [P.segment_to_dict(s) for s in self.source],
                    "candidate": {
                        "segments": [{"id": "x", "polished_text": self.wrong[0].text}]
                    },
                },
            )
            rows = [P.segment_to_dict(s) for s in self.wrong]
            cache = {
                "segments": rows,
                "output_signature": P._value_signature(rows),
                "chunk_ids": ["x"],
                "model": "4B",
                "candidate_attempts": [
                    {
                        "model": "4B",
                        "validation": "accepted-protected-fields",
                        "candidate_path": str(candidate),
                    }
                ],
            }
            self.assertTrue(
                P._checkpoint_segments_match(
                    cache, self.source, rules=default_glossary_rules()
                )
            )
            self.assertFalse(
                P._checkpoint_segments_match(
                    cache,
                    self.source,
                    rules=default_glossary_rules(),
                    source_context=context("实际为"),
                )
            )
            write_json(
                root / "transcript.enhanced.json", {"enhanced_transcription": rows}
            )
            self.assertFalse(
                P._cached_enhanced_valid(
                    root,
                    self.source,
                    default_glossary_rules(),
                    source_context=context(),
                )
            )

    def test_native_and_auto_ignore_existing_cloud_key_and_do_not_fallback(self):
        for choice in ("native", "auto"):
            with (
                patch.dict(
                    "os.environ", {"FIRECRAWL_API_KEY": "synthetic-not-a-secret"}
                ),
                patch("workbench.episode_context._fetch_with_firecrawl") as cloud,
                patch(
                    "workbench.episode_context.urlopen",
                    return_value=Response("<html>native</html>"),
                ),
            ):
                html, provider, error = _fetch_page("https://example.test/", choice)
            self.assertIn("native", html)
            self.assertEqual(provider, "xiaoyuzhou-next-data")
            self.assertFalse(error)
            cloud.assert_not_called()
        with (
            patch("workbench.episode_context.urlopen", side_effect=TimeoutError()),
            patch("workbench.episode_context._fetch_with_firecrawl") as cloud,
        ):
            self.assertEqual(_fetch_page("https://example.test/", "auto")[0], "")
            cloud.assert_not_called()

    def test_explicit_cloud_failure_unknown_mode_and_size_limit_are_honest(self):
        for result in ("", RuntimeError("token=synthetic-must-not-be-stored")):
            with (
                patch(
                    "workbench.episode_context._fetch_with_firecrawl",
                    **(
                        {"side_effect": result}
                        if isinstance(result, Exception)
                        else {"return_value": result}
                    ),
                ),
                patch("workbench.episode_context.urlopen") as native,
            ):
                html, provider, error = _fetch_page(
                    "https://example.test/", "firecrawl"
                )
            self.assertEqual((html, provider), ("", "firecrawl"))
            self.assertNotIn("synthetic", error)
            native.assert_not_called()
        with (
            patch("workbench.episode_context.urlopen") as native,
            patch("workbench.episode_context._fetch_with_firecrawl") as cloud,
        ):
            self.assertEqual(
                _fetch_page("https://example.test/", "unknown")[1], "unselected"
            )
            native.assert_not_called()
            cloud.assert_not_called()
        with (
            patch("workbench.episode_context.MAX_CONTEXT_PAGE_BYTES", 8),
            patch("workbench.episode_context.urlopen", return_value=Response("x" * 9)),
        ):
            self.assertIn(
                "size limit", _fetch_page("https://example.test/", "native")[2]
            )

    def test_default_and_environment_choices_are_recorded_with_snapshot_hashes(self):
        next_data = {
            "props": {
                "pageProps": {
                    "episode": {
                        "eid": "x",
                        "title": "节目",
                        "shownotes": "<p>00:00 开场</p>",
                    },
                    "comments": [],
                }
            }
        }
        html = '<script id="__NEXT_DATA__">' + json.dumps(next_data) + "</script>"
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(
                "os.environ",
                {"NEXTECHO_CRAWLER": "auto", "FIRECRAWL_API_KEY": "synthetic"},
            ),
            patch("workbench.episode_context.urlopen", return_value=Response(html)),
            patch("workbench.episode_context._fetch_with_firecrawl") as cloud,
        ):
            root = Path(d)
            data = crawl_episode_context("https://example.test/", root)
            self.assertFalse(data["fetch_policy"]["cloud_selected_explicitly"])
            self.assertEqual(data["fetch_policy"]["effective"], "native")
            for name, sha in data["source_snapshot"]["files"].items():
                self.assertEqual(file_hash(root / name), sha)
            cloud.assert_not_called()
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict("os.environ", {"NEXTECHO_CRAWLER": "firecrawl"}),
            patch(
                "workbench.episode_context._fetch_with_firecrawl", return_value=html
            ) as cloud,
            patch("workbench.episode_context.urlopen") as native,
        ):
            data = crawl_episode_context("https://example.test/", Path(d))
            self.assertTrue(data["fetch_policy"]["cloud_selected_explicitly"])
            cloud.assert_called_once()
            native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
