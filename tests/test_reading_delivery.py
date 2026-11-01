from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.contracts import PipelineError
from workbench.delivery import assemble, compact, write_receipt
from workbench.io_utils import write_json
from workbench.reading_artifacts import freeze_delivery_context, primary_path, reading_filename


class ReadingDeliveryTests(unittest.TestCase):
    def item(self, root, *, title="E45 孟岩对话李继刚：人何以自处"):
        folder = root / "items" / "one"
        folder.mkdir(parents=True, exist_ok=True)
        text = folder / "transcript.txt"
        text.write_text("说话人 1：金额36万元，额外费用不能超过限额。\n李继刚：我是李继刚。", encoding="utf-8")
        return {"status": "ready", "title": title, "text_path": str(text),
                "source_context": {"url": "https://example.org/episode/1", "episode": {
                    "id": "e1", "title": title, "podcast": "无人知晓", "published_at": "2026-03-03"},
                    "speaker_candidates": [{"name": "孟岩"}, {"name": "李继刚"}],
                    "shownotes": {"text": "这是官方简介。", "sections": [{"text": "04:15 讨论主题", "start_ms": 255000}]},
                    "comment_selection": {"returned": 20, "returned_replies": 35, "reported_total": 3138, "complete": False}}}

    def test_descriptive_primary_header_receipt_and_legacy_alias(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            stamp = freeze_delivery_context(root, created_at=1792368000)
            item = self.item(root)
            path = assemble(root, [item])
            self.assertEqual(path.name, f"无人知晓_E45 孟岩对话李继刚：人何以自处_{stamp['transcribed_on']}.md")
            body = path.read_text()
            self.assertEqual(body, (root / "result.md").read_text())
            self.assertLess(body.index("这是官方简介"), body.index("金额36万元"))
            self.assertIn("不代表声音已确认", body)
            self.assertIn("完整性未确认", body)
            receipt = write_receipt(root, job_id="one", status="ready", results=[item])
            self.assertEqual(receipt["primary_file"], str(path.resolve()))
            self.assertIn(str((root / "result.md").resolve()), receipt["artifact_hashes"])
            self.assertLessEqual(len(json.dumps(compact(receipt), ensure_ascii=False).encode()), 2048)

    def test_partial_keeps_failed_diagnostics_without_requiring_failed_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            good=self.item(root)
            existing=root/'failed-audit.json'
            existing.write_text('{"synthetic":"failed stage"}')
            missing=root/'failed'/'transcript.txt'
            failed={'status':'failed','title':'未完成项','text_path':str(missing),
                    'error':'synthetic failure',
                    'metadata':{'audit_path':str(existing),'pending_path':str(root/'absent.json')}}
            assemble(root,[good,failed])
            receipt=write_receipt(root,job_id='mixed',status='partial',results=[good,failed])
            self.assertEqual(receipt['status'],'partial')
            self.assertTrue(Path(receipt['primary_file']).is_file())
            self.assertIn(str(existing.resolve()),receipt['artifact_hashes'])
            self.assertNotIn(str(missing.resolve()),receipt['artifact_hashes'])
            errors=json.loads((root/'errors.json').read_text())
            self.assertEqual(errors['items'][0]['text_path'],str(missing))

    def test_successful_item_still_requires_files_in_partial_delivery(self):
        for status in ['ready','partial']:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                good=self.item(root)
                assemble(root,[good])
                Path(good['text_path']).unlink()
                with self.assertRaises(PipelineError):
                    write_receipt(root,job_id='missing',status=status,results=[good])

    def test_special_and_long_unicode_titles_are_safe_and_collision_resistant(self):
        for title in ["../CON:秘密?\\路径\x00", "播客🙂" * 200, "A/B", "A?B"]:
            name = reading_filename([{"title": title, "show_title": "CON"}], "2026-10-19")
            self.assertEqual(Path(name).name, name)
            self.assertNotRegex(name, r'[<>:"/\\|?*\x00-\x1f]')
            self.assertLessEqual(len(name.encode()), 220)
            self.assertTrue(name.endswith("_2026-10-19.md"))
        self.assertNotEqual(reading_filename([{"title": "A/B"}], "2026-10-19"),
                            reading_filename([{"title": "A?B"}], "2026-10-19"))

    def test_resume_inherits_original_day_and_is_stable_on_reassembly(self):
        with tempfile.TemporaryDirectory() as d:
            parent, child = Path(d) / "old", Path(d) / "new"
            parent.mkdir()
            child.mkdir()
            first = freeze_delivery_context(parent, created_at=1792368000)
            resumed = freeze_delivery_context(child, created_at=1892368000, parent_dir=parent)
            self.assertEqual(first, resumed)
            self.assertEqual(freeze_delivery_context(child, created_at=1992368000), first)
            path = assemble(child, [self.item(child)])
            self.assertIn(first["transcribed_on"], path.name)

    def test_old_request_without_delivery_context_inherits_creation_time(self):
        with tempfile.TemporaryDirectory() as d:
            parent, child = Path(d) / "old", Path(d) / "new"
            parent.mkdir()
            child.mkdir()
            write_json(parent / "request.json", {"created_at": 0})
            data = freeze_delivery_context(child, parent_dir=parent)
            self.assertEqual(data["transcribed_on"], "1970-01-01")

    def test_job_attempt_keeps_original_transcription_day(self):
        from workbench.job_store import JobStore
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = JobStore(root / "runtime")
            with patch("workbench.job_store.time.time", return_value=1792368000):
                first, _ = store.submit({"request": {}}, output_dir=root / "first")
            original_request = (root / "first/request.json").read_bytes()
            with patch("workbench.job_store.time.time", return_value=1892368000):
                second, _ = store.submit({"request": {}}, output_dir=root / "second",
                    parent=first["job_id"], resume_from=first["run_dir"])
            self.assertNotEqual(first["created_at"], second["created_at"])
            self.assertEqual(first["transcribed_on"], second["transcribed_on"])
            self.assertEqual((root / "first/request.json").read_bytes(), original_request)

    def test_index_cannot_escape_or_hide_missing_or_modified_primary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = assemble(root, [self.item(root)])
            path.write_text("changed")
            with self.assertRaises(PipelineError):
                write_receipt(root, job_id="one", status="ready")
            write_json(root / "delivery.index.json", {"primary_name": "../escape.md", "primary_sha256": "x"})
            with self.assertRaises(PipelineError):
                primary_path(root)

    def test_multiple_items_silence_and_failed_item_keep_real_content(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            items = [self.item(root), {"status": "ready", "title": "静音", "asr": {"coverage": {"content_state": "no_speech"}}},
                     {"status": "failed", "title": "失败项", "error": "source failed"}]
            path = assemble(root, items)
            self.assertIn("播客合集-3", path.name)
            body = path.read_text()
            self.assertIn("没有检测到可识别语音", body)
            self.assertIn("source failed", body)
            self.assertIn("金额36万元", body)
            receipt = write_receipt(root, job_id="many", status="partial", results=items)
            self.assertEqual(receipt["primary_file"], str(path.resolve()))

    def test_source_markup_is_inert_and_historical_result_still_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            item = self.item(root, title='<script>alert(1)</script> ![x](https://example.org/x)')
            item["source_context"]["shownotes"]["text"] = '<img src="x">\n![x](https://example.org/x)'
            body = assemble(root, [item]).read_text()
            self.assertNotIn("<script>", body)
            self.assertNotIn("<img", body)
            self.assertNotIn("![x]", body)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "result.md").write_text("legacy")
            receipt = write_receipt(root, job_id="old", status="ready")
            self.assertEqual(receipt["primary_file"], str((root / "result.md").resolve()))


if __name__ == "__main__":
    unittest.main()
