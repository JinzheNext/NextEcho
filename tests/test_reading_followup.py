from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote

from workbench.delivery import assemble
from workbench.io_utils import file_hash, write_json
from workbench.publication import begin_pipeline, publish_pipeline, verify_publication
from workbench.reading_revision import publish_reading_revision
from workbench.reading_view import clean_boundaries, render_reading
from workbench.contracts import PipelineError
from workbench.speaker_service import speaker_artifact_paths


def row(id, text, voice, start, duration=120):
    return {"id": id, "text": text, "speaker": voice, "speaker_id": voice,
            "start_ms": start, "end_ms": start + duration}


class ReadingFollowupTests(unittest.TestCase):
    def test_real_reported_splits_do_not_create_fake_speaker_changes(self):
        source = [row("a", "前面的讨论。", "说话人 2", 0),
                  row("b", "嗯，那", "说话人 1", 120, 720),
                  row("c", "里边是各种信息的罗列堆叠。", "说话人 2", 840),
                  row("d", "然", "说话人 1", 960),
                  row("e", "后我我自己现在找来找去，", "说话人 2", 1080)]
        original = copy.deepcopy(source)
        body, audit = render_reading(source, {})
        self.assertEqual(source, original)
        self.assertIn("嗯，那里边是各种信息的罗列堆叠。然后我自己现在找来找去，", body)
        self.assertEqual(body.count("**说话人 2：**"), 1)
        self.assertNotIn("**说话人 1：**", body)
        self.assertEqual(audit["split_word_joins"], 2)
        self.assertEqual(audit["accounted_segment_ids"], [r["id"] for r in source])

    def test_short_answers_quantities_negations_and_long_pauses_remain(self):
        source = [row("a", "不能。", "A", 0), row("b", "36万元。", "B", 120),
                  row("c", "对。", "A", 240), row("d", "然。", "A", 360),
                  row("e", "后面的意见", "B", 480), row("f", "因", "A", 600),
                  row("g", "为别人考虑。", "B", 3000)]
        body, audit = render_reading(source, {})
        self.assertIn("不能。", body)
        self.assertIn("36万元。", body)
        self.assertIn("对。", body)
        self.assertEqual(audit["split_word_joins"], 0)

    def test_nearest_unknown_group_is_overridden_by_clear_lexical_continuation(self):
        source = [row("a", "嗯，", "说话人 1", 0), row("b", "那", "归属待核对", 120),
                  row("c", "里边有一个问题。", "说话人 2", 240)]
        rows, _ = clean_boundaries(source)
        self.assertEqual(next(r for r in rows if r["id"] == "b")["reading_identity"], "说话人 2")

    def test_brief_hesitation_is_hidden_but_real_short_response_stays(self):
        source = [row("a", "嗯，", "说话人 1", 0), row("b", "对。", "说话人 1", 120),
                  row("c", "嗯？", "说话人 1", 240), row("d", "不能。", "说话人 2", 360)]
        body, audit = render_reading(source, {})
        self.assertNotIn("嗯，", body)
        for text in ["对。", "嗯？", "不能。"]:
            self.assertIn(text, body)
        self.assertEqual(audit["omitted_backchannels"], 1)

    def fixture(self, root):
        folder = root / "items" / "one"
        folder.mkdir(parents=True)
        source = [row("a", "然", "说话人 1", 0), row("b", "后我我自己现在看。", "说话人 2", 120)]
        write_json(folder / "transcript.enhanced.json", {"transcription": source, "enhanced_transcription": source})
        (folder / "transcript.txt").write_text("原稿保持。")
        (folder / "transcript.srt").write_text("原字幕保持。")
        replies = [{"author": "回复者", "text": f"回复 {i}", "like_count": i} for i in range(35)]
        comments = [{"author": "听众", "text": f"评论 {i}", "replies": replies if i == 0 else []} for i in range(20)]
        return {"title": "测试节目", "status": "ready", "text_path": str(folder / "transcript.txt"),
                "enhanced_json_path": str(folder / "transcript.enhanced.json"),
                "asr": {"coverage": {"partial": False}}, "speaker_status": "partial",
                "enhancement": {"status": "partial", "polish": {"processing_complete": True}},
                "source_context": {"comments": comments, "comment_selection": {"complete": False},
                    "shownotes": {"sections": [{"start_ms": 0, "text": "00:00 话题 A？（一）"}]}}}

    def test_comments_visible_and_heading_links_resolve_after_single_file_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = self.fixture(root)
            body = assemble(root, [item]).read_text()
            headings = {line.lstrip("# ") for line in body.splitlines() if line.startswith("#")}
            for target in re.findall(r"\]\(#([^)]*)\)", body):
                self.assertIn(unquote(target), headings)
            self.assertEqual(body.count("#### 评论 "), 20)
            self.assertEqual(body.count("回复 · **回复者**"), 35)
            for removed in ["音频覆盖：", "全文润色处理已完成", "说话人部分完成：", "逐段原始/润色记录", "覆盖："]:
                self.assertNotIn(removed, body)
            self.assertTrue(Path(item["comments_markdown_path"]).is_file())

    def test_revision_is_immutable_model_free_and_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "original"
            base.mkdir()
            item = self.fixture(base)
            begin_pipeline(base, job_id="original")
            publish_pipeline(base, job_id="original", results=[item])
            before = {str(p.relative_to(base)): file_hash(p) for p in base.rglob("*") if p.is_file()}
            with patch("workbench.postprocess.call_json", side_effect=AssertionError("must not run a model")):
                receipt = publish_reading_revision(base)
            target = Path(receipt["receipt_path"]).parent
            verify_publication(target)
            self.assertEqual(before, {str(p.relative_to(base)): file_hash(p) for p in base.rglob("*") if p.is_file()})
            for name in ["transcript.txt", "transcript.srt", "transcript.enhanced.json"]:
                self.assertEqual(file_hash(base / "items/one" / name), file_hash(target / "items/one" / name))
            self.assertEqual(json.loads((target / "reading.source.json").read_text())["polish_calls"], 0)

    def test_copied_native_evidence_uses_exact_local_hash_not_external_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "copy/items/one"
            local = folder / "speaker-stages/native/result.json"
            write_json(local, {"measured": "fixed source bytes"})
            external = root / "old/items/one/speaker-stages/native/result.json"
            payload = {"acoustic_evidence": {"paths": {"native_report_json_path": str(external)},
                       "hashes": {"native_report_json_path": file_hash(local)}}}
            self.assertEqual(speaker_artifact_paths(folder, payload=payload)["native_report_json_path"], str(local.resolve()))
            local.write_text("tampered")
            with self.assertRaises(PipelineError):
                speaker_artifact_paths(folder, payload=payload)

    def test_evidence_relocation_cannot_follow_symlink_or_guess_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / "copy/items/one"
            local = folder / "speaker-stages/result.json"
            other = root / "unrelated.json"
            write_json(other, {"secret": "synthetic"})
            local.parent.mkdir(parents=True)
            local.symlink_to(other)
            payload = {"acoustic_evidence": {"paths": {"native_report_json_path": str(root / "old/items/one/speaker-stages/result.json")},
                       "hashes": {"native_report_json_path": file_hash(other)}}}
            with self.assertRaises(PipelineError):
                speaker_artifact_paths(folder, payload=payload)
            local.unlink(); local.write_bytes(other.read_bytes())
            payload["acoustic_evidence"]["paths"]["native_report_json_path"] = str(root / "unrelated/result.json")
            with self.assertRaises(PipelineError):
                speaker_artifact_paths(folder, payload=payload)

    def test_copied_reading_can_accept_names_and_be_read_again_without_models(self):
        from app import load_speaker_transcript, _apply_manual_speaker_map
        from workbench.identity_revision import publish_identity_revision
        from workbench.speaker_service import _speaker_item_payload, _aggregate_speaker_payload
        from tests.test_speaker_artifacts import make_dialogue
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "original"; base.mkdir()
            item = self.fixture(base); folder = base / "items/one"
            _, paths = make_dialogue(folder); item.update(paths, source_label="one", output_dir=str(folder))
            _aggregate_speaker_payload(base, [_speaker_item_payload(folder, "one")])
            begin_pipeline(base, job_id="original"); publish_pipeline(base, job_id="original", results=[item])
            clone = Path(publish_reading_revision(base)["receipt_path"]).parent
            before = {str(p.relative_to(clone)): file_hash(p) for p in clone.rglob("*") if p.is_file()}
            view = load_speaker_transcript(clone)
            mapping = {person["speaker_id"]: name for person, name in zip(view["speakers"], ["孟岩", "李继刚"])}
            with patch("workbench.postprocess.call_json", side_effect=AssertionError("no inference")):
                target, _, receipt = publish_identity_revision(clone, _apply_manual_speaker_map(view, mapping), mapping)
            result = load_speaker_transcript(target)
            self.assertEqual({p["display_name"] for p in result["speakers"]}, {"孟岩", "李继刚"})
            self.assertEqual(before, {str(p.relative_to(clone)): file_hash(p) for p in clone.rglob("*") if p.is_file()})
            verify_publication(target)
            body = Path(receipt["primary_file"]).read_text()
            self.assertIn("对话人物：孟岩、李继刚", body)
            self.assertIn("**孟岩：**", body)
            self.assertIn("**李继刚：**", body)
            self.assertEqual(body.count("#### 评论 "), 20)
            self.assertIn("话题 A？（一）", body)


if __name__ == "__main__":
    unittest.main()
