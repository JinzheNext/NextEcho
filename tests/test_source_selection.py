import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.config import load_config
from workbench.contracts import PipelineError, PipelineRequest, plan_request
from workbench.pipeline import execute_plan
from workbench.sources import resolve_media_inputs


class SourceSelectionTests(unittest.TestCase):
    def test_headless_feed_selects_latest_even_when_publisher_order_is_oldest_first(self):
        xml = '''<rss><channel><title>test</title>
        <item><title>old</title><pubDate>Fri, 02 Jan 2026 00:00:00 GMT</pubDate><enclosure url="https://media/old.mp3"/></item>
        <item><title>new</title><pubDate>Sun, 18 Oct 2026 01:00:00 GMT</pubDate><enclosure url="https://media/new.mp3"/></item>
        </channel></rss>'''
        with patch("workbench.sources._fetch_text", return_value=(xml, "")):
            selected = resolve_media_inputs(["https://example.org/feed.xml"])
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].title, "new")

    def test_show_page_uses_latest_and_does_not_skip_private_episode(self):
        episodes = [{"eid": "old", "title": "old", "pubDate": "2026-01-01", "enclosure": {"url": "https://media/old.mp3"}},
                    {"eid": "new", "title": "new", "pubDate": "2026-10-18", "isPrivateMedia": True,
                     "enclosure": {"url": "https://media/private.mp3"}}]
        html = '<script id="__NEXT_DATA__" type="application/json">' + json.dumps({"props": {"pageProps": {"podcast": {"episodes": episodes}}}}) + '</script>'
        with patch("workbench.sources._fetch_text", return_value=(html, "")):
            selected = resolve_media_inputs(["https://www.xiaoyuzhoufm.com/podcast/test"])
            self.assertEqual(len(selected), 1)
            self.assertTrue(selected[0].canonical_url.endswith("/new"))
            self.assertTrue(selected[0].requires_auth)
            self.assertEqual(selected[0].resolved_media_url, "")
            self.assertEqual(len(resolve_media_inputs(["https://www.xiaoyuzhoufm.com/podcast/test"], limit=2)), 2)

    def test_resume_freezes_resolved_episode_and_rejects_modified_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan = plan_request(PipelineRequest(("https://example.org/feed.xml",), postprocess="off"), load_config())
            from workbench.sources import SourceDescriptor
            descriptor = SourceDescriptor("feed", "rss", plan.request.sources[0], "https://example.org/old", "https://media/old.mp3", "old", "", 1, "", "001_old", "rss")
            def stop_after_selection(*args, **kwargs):
                raise PipelineError("WORKER_FAILED", "stop after recording source")
            first = root / "first"
            first.mkdir()
            with patch("workbench.pipeline.resolve_media_inputs", return_value=[descriptor]), patch("workbench.pipeline.transcribe_media_sources", side_effect=stop_after_selection):
                with self.assertRaises(PipelineError):
                    execute_plan(plan, first, job_id="test")
            second = root / "second"
            second.mkdir()
            with patch("workbench.pipeline.resolve_media_inputs", side_effect=AssertionError("must not resolve latest episode again")), patch("workbench.pipeline.transcribe_media_sources", side_effect=stop_after_selection) as transcribe:
                with self.assertRaises(PipelineError):
                    execute_plan(plan, second, job_id="test2", resume_from=str(first))
                self.assertEqual(transcribe.call_args.args[0][0].canonical_url, "https://example.org/old")
            data = json.loads((first / "resolved_sources.json").read_text())
            data["descriptors"][0]["canonical_url"] = "https://example.org/new"
            (first / "resolved_sources.json").write_text(json.dumps(data))
            third = root / "third"
            third.mkdir()
            with self.assertRaises(PipelineError) as caught:
                execute_plan(plan, third, job_id="test3", resume_from=str(first))
            self.assertEqual(caught.exception.code, "CACHE_MISMATCH")
