"""来源选择反例及真实CLI/持久请求边界，无模型或网络调用。"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.cli import main
from workbench.config import NextEchoConfig, load_config
from workbench.contracts import PipelineError, PipelineRequest, plan_request
from workbench.io_utils import file_hash, write_json
from workbench.job_store import JobStore
from workbench.sources import (
    _resolve_audio_from_rss, _resolve_audio_from_rss_candidates, canonicalize_url,
    resolve_feed, resolve_source, slugify,
)
from workbench.transcription import prepare_media_source, transcribe_single_source


def feed(items):
    return "<rss><channel>" + "".join(
        f"<item><title>{title}</title><guid>{guid}</guid><link>{link}</link>"
        f'<enclosure url="{media}"/></item>' for title, guid, link, media in items
    ) + "</channel></rss>"


class SourceIdentityTests(unittest.TestCase):
    def test_specific_episode_missing_prefix_collision_and_conflict_do_not_select_first(self):
        target = "https://www.xiaoyuzhoufm.com/episode/wanted"
        unrelated = ("wanted title", "wanted-other", target + "-other", "https://media/other.mp3")
        with patch("workbench.sources._fetch_text", return_value=(feed([unrelated]), "")):
            self.assertEqual(_resolve_audio_from_rss("https://feed/rss.xml", target), "")
        exact = ("title", "wanted", target, "https://media/exact.mp3")
        with patch("workbench.sources._fetch_text", return_value=(feed([unrelated, exact]), "")):
            self.assertEqual(_resolve_audio_from_rss("https://feed/rss.xml", target), exact[-1])
        conflict = (*exact[:-1], "https://media/conflicting.mp3")
        with patch("workbench.sources._fetch_text", return_value=(feed([exact, conflict]), "")):
            self.assertEqual(_resolve_audio_from_rss("https://feed/rss.xml", target), "")

    def test_specific_page_rss_mismatch_exposes_error_and_no_other_media(self):
        url = "https://www.xiaoyuzhoufm.com/episode/wanted"
        html = '<meta property="og:title" content="wanted"><link type="application/rss+xml" href="https://feed/rss.xml">'
        xml = feed([("other", "other", "https://www.xiaoyuzhoufm.com/episode/other", "https://media/other.mp3")])
        with patch("workbench.sources._fetch_text", side_effect=[(html, ""), (xml, "")]):
            descriptor = resolve_source(url)
        self.assertTrue(descriptor.error)
        self.assertEqual(descriptor.resolved_media_url, "")
        self.assertEqual(descriptor.canonical_url, url)

    def test_rss_candidates_require_exact_evidence_and_check_ambiguity_across_feeds(self):
        settings = dict(episode_title="A requested episode", duration_seconds=100,
                        episode_id="wanted", episode_url="https://publisher/episode/wanted")
        fuzzy = feed([("A requested episodex", "other", "https://publisher/other", "https://media/other.mp3")])
        with patch("workbench.sources._fetch_text", return_value=(fuzzy, "")):
            self.assertEqual(_resolve_audio_from_rss_candidates(["https://feed/rss.xml"], **settings), ("", ""))
        exact = feed([("A requested episode", "wanted", settings["episode_url"], "https://media/one.mp3")])
        conflicting = exact.replace("one.mp3", "two.mp3")
        with patch("workbench.sources._fetch_text", side_effect=[(exact, ""), (conflicting, "")]):
            self.assertEqual(_resolve_audio_from_rss_candidates(["https://feed/a.xml", "https://feed/b.xml"], **settings), ("", ""))
        with patch("workbench.sources._fetch_text", return_value=(exact, "")):
            self.assertEqual(_resolve_audio_from_rss_candidates(["https://feed/a.xml", "https://feed/b.xml"], **settings)[0], "https://media/one.mp3")

    def test_bilibili_part_is_retained_through_metadata_and_mismatch_is_rejected(self):
        base = "https://www.bilibili.com/video/BVexample"
        self.assertEqual(canonicalize_url(base + "/?p=2&share_source=test#marker", "bilibili"), base + "?p=2")
        payload = {"id": "BVexample_p2", "title": "part2", "webpage_url": base,
                   "url": "https://media/part2.mp4", "duration": 100}
        with patch("workbench.sources._dump_ytdlp_metadata", return_value=(payload, "")) as inspect:
            d = resolve_source(base + "?p=2&tracking=ignored")
        inspect.assert_called_once_with(base + "?p=2")
        self.assertEqual(d.canonical_url, base + "?p=2")
        self.assertEqual(d.selection_evidence["part"], 2)
        with tempfile.TemporaryDirectory() as temp, patch("workbench.transcription._download_media_with_ytdlp", return_value=Path(temp) / "media.mp4") as download:
            prepare_media_source(d, Path(temp))
            self.assertEqual(download.call_args.args[0], base + "?p=2")
        with patch("workbench.sources._dump_ytdlp_metadata", return_value=({**payload, "id": "BVexample_p1"}, "")):
            wrong = resolve_source(base + "?p=2")
        self.assertEqual(wrong.resolved_media_url, "")
        self.assertTrue(wrong.error)
        with tempfile.TemporaryDirectory() as temp, patch("workbench.transcription._download_media_with_ytdlp") as download, self.assertRaises(PipelineError):
            prepare_media_source(wrong, Path(temp))
        download.assert_not_called()
        for query in ("p=0", "p=-1", "p=", "p=2&p=3"):
            with self.assertRaises(PipelineError):
                canonicalize_url(base + "?" + query, "bilibili")

    def test_bilibili_short_redirect_keeps_observed_part_and_part_ids_are_distinct(self):
        base = "https://www.bilibili.com/video/BVexample"
        payload = {"id": "BVexample_p2", "webpage_url": base + "?p=2", "url": "https://media/part2.mp4"}
        with patch("workbench.sources._dump_ytdlp_metadata", return_value=(payload, "")):
            second = resolve_source("https://b23.tv/short")
        self.assertEqual(second.canonical_url, base + "?p=2")
        with patch("workbench.sources._dump_ytdlp_metadata", return_value=({**payload, "id": "BVexample_p1", "webpage_url": base}, "")):
            first = resolve_source(base)
        self.assertNotEqual(first.source_id, second.source_id)

    def test_duration_keeps_declared_units_and_item_slug_is_bounded_not_title(self):
        with patch("workbench.sources._fetch_text", return_value=('<meta property="og:title" content="标题"><script>{"duration":100,"audioUrl":"https://media/one.mp3"}</script>', "")):
            self.assertEqual(resolve_source("https://www.xiaoyuzhoufm.com/episode/one").duration_seconds, 100)
        with patch("workbench.sources._fetch_text", return_value=('<meta property="og:title" content="标题">87分钟<script>{"audioUrl":"https://media/one.mp3"}</script>', "")):
            self.assertEqual(resolve_source("https://www.xiaoyuzhoufm.com/episode/one").duration_seconds, 5220)
        apple = '<meta property="og:title" content="episode">37 min<audio src="https://media/one.mp3"></audio>'
        with patch("workbench.sources._fetch_text", return_value=(apple, "")):
            self.assertEqual(resolve_source("https://podcasts.apple.com/us/podcast/example/id1?i=2").duration_seconds, 2220)
        apple += '<script type="application/ld+json">{"@type":"PodcastEpisode","duration":"PT1M40S"}</script>'
        with patch("workbench.sources._fetch_text", return_value=(apple, "")):
            self.assertEqual(resolve_source("https://podcasts.apple.com/us/podcast/example/id1?i=2").duration_seconds, 100)
        long_title = "中文" * 90
        xml = feed([(long_title, "episode", "https://publisher/episode", "https://media/one.mp3")])
        with patch("workbench.sources._fetch_text", return_value=(xml, "")):
            d = resolve_feed("https://feed/rss.xml")[0]
        self.assertEqual(d.title, long_title)
        self.assertLess(len(d.source_label.encode()), 90)
        self.assertNotEqual(slugify(long_title), slugify(long_title + "另"))
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / d.source_label).mkdir()

    def test_feed_guid_identity_is_stable_under_order_changes_and_distinguishes_no_link_episodes(self):
        first = ("same title", "guid-one", "", "https://media/one.mp3")
        second = ("same title", "guid-two", "", "https://media/two.mp3")
        with patch("workbench.sources._fetch_text", return_value=(feed([first, second]), "")):
            a = resolve_feed("https://feed/rss.xml", limit=2)
        with patch("workbench.sources._fetch_text", return_value=(feed([second, first]), "")):
            b = resolve_feed("https://feed/rss.xml", limit=2)
        self.assertEqual(a[0].source_id, b[1].source_id)
        self.assertNotEqual(a[0].source_id, a[1].source_id)
        self.assertEqual(a[0].to_dict()["selection_evidence"]["guid"], "guid-one")

    def test_cli_freezes_caller_path_and_preserves_original_input_before_detaching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            caller, worker = root / "caller", root / "worker"
            caller.mkdir()
            worker.mkdir()
            (caller / "input.wav").write_bytes(b"requested audio")
            (worker / "input.wav").write_bytes(b"wrong audio")
            with patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(root / "runtime")}):
                out = io.StringIO()
                with contextlib.chdir(caller), contextlib.redirect_stdout(out), patch("workbench.job_store.JobStore.spawn"):
                    code = main(["run", "input.wav", "--profile", "fast", "--postprocess", "off", "--detach", "--output", "json"])
                self.assertEqual(code, 0)
                job_id = json.loads(out.getvalue())["job_id"]
                job = JobStore().get(job_id, reconcile=False)
                self.assertEqual(job["submission_context"], {"caller_cwd": str(caller), "original_inputs": ["input.wav"]})
                self.assertEqual(job["plan"]["request"]["sources"], [str(caller / "input.wav")])
                with contextlib.chdir(worker):
                    source = resolve_source(job["plan"]["request"]["sources"][0])
                self.assertEqual(Path(source.input).read_bytes(), b"requested audio")

    def test_equivalent_relative_output_directory_keeps_same_idempotency_hash(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.chdir(temp):
            store = JobStore(Path(temp) / "runtime")
            plan = plan_request(PipelineRequest(("https://media/file.wav",), postprocess="off"), load_config()).to_dict()
            first, created = store.submit(plan, output_dir=Path("result"), key="one-intent")
            same, again = store.submit(plan, output_dir=Path(temp) / "result", key="one-intent")
            self.assertTrue(created)
            self.assertFalse(again)
            self.assertEqual(first["job_id"], same["job_id"])

    def test_blank_or_directory_input_fails_before_persisting_or_spawning_a_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for source in ("", "   ", str(root)):
                with self.subTest(source=source), patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(root / "runtime")}), patch("workbench.job_store.JobStore.spawn") as spawn:
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        code = main(["run", source, "--profile", "fast", "--postprocess", "off", "--detach", "--output", "json"])
                    self.assertEqual(code, 2)
                    self.assertEqual(json.loads(out.getvalue())["error_code"], "INVALID_INPUT")
                    spawn.assert_not_called()
                    self.assertFalse((root / "runtime/jobs.sqlite3").exists())
            self.assertEqual(resolve_source(str(root)).source_type, "unknown")

    def test_changed_feed_guid_invalidates_copied_media_and_asr_without_altering_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            descriptors = []
            for guid in ("first", "second"):
                with patch("workbench.sources._fetch_text", return_value=(feed([("same title", guid, "", f"https://media/{guid}.wav")]), "")):
                    descriptors.append(resolve_feed("https://feed/rss.xml")[0])
            media = root / "remote.wav"
            media.write_bytes(b"media")
            config = NextEchoConfig({"runtime": {"resume_verified": True}})
            request = dict(model_path=None, whisper_bin=None, ffmpeg_bin="fake", language="zh", max_seconds=0,
                           output_formats=["txt", "json"], glossary_text="", enable_enhancement=False,
                           ai_polish=False, ai_style="", index=1, asr_engine="sensevoice", config=config)
            def normalize(**kwargs):
                kwargs["target_path"].write_bytes(b"normalized")
            def recognize(**kwargs):
                (kwargs["output_dir"] / "transcript.txt").write_text("synthetic recognized text")
                write_json(kwargs["output_dir"] / "transcript.json", {"transcription": []})
                return {"actual_engine": "sensevoice"}
            with patch("workbench.transcription.prepare_media_source", return_value=media) as download, patch("workbench.transcription.extract_audio", side_effect=normalize), patch("workbench.transcription.transcribe_audio_with_fallback", side_effect=recognize) as asr, patch("workbench.capabilities.probe_environment", return_value={}):
                first = transcribe_single_source(descriptor=descriptors[0], items_dir=root / "old/items", **request)
                original = Path(first["json_path"])
                sha = file_hash(original)
                shutil.copytree(root / "old/items", root / "new/items")
                second = transcribe_single_source(descriptor=descriptors[1], items_dir=root / "new/items", **request)
                self.assertEqual(second["status"], "ready")
                self.assertEqual(download.call_count, 2)
                self.assertEqual(asr.call_count, 2)
                self.assertEqual(file_hash(original), sha)
                self.assertNotEqual(first["source_id"], second["source_id"])

    def test_asr_cache_does_not_adopt_old_derived_delivery_when_later_stages_are_off(self):
        from workbench.publication import publish_pipeline
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.wav"
            source.write_bytes(b"source fixture")
            descriptor = resolve_source(str(source))
            run = root / "run"
            request = dict(descriptor=descriptor, items_dir=run / "items", model_path=None,
                whisper_bin=None, ffmpeg_bin="fake", language="zh", max_seconds=0,
                output_formats=["txt", "json"], glossary_text="", enable_enhancement=False,
                ai_polish=False, ai_style="", index=1, asr_engine="sensevoice", defer_enhancement=True,
                config=NextEchoConfig({"runtime": {"resume_verified": True}}))
            def normalize(**kwargs):
                kwargs["target_path"].write_bytes(b"normalized")
            def recognize(**kwargs):
                (kwargs["output_dir"] / "transcript.txt").write_text("本轮只请求原始识别全文")
                write_json(kwargs["output_dir"] / "transcript.json", {"transcription": []})
                return {"actual_engine": "sensevoice"}
            with patch("workbench.transcription.extract_audio", side_effect=normalize), patch("workbench.transcription.transcribe_audio_with_fallback", side_effect=recognize) as asr, patch("workbench.capabilities.probe_environment", return_value={}):
                first = transcribe_single_source(**request)
                item = Path(first["output_dir"])
                raw_sha = file_hash(item / "transcript.json")
                stale = item / "old-reading.txt"
                stale.write_text("OLD_DERIVATIVE_SHOULD_NOT_BE_SELECTED")
                old = {**first, "identity_text_path": str(stale), "enhanced_text_path": str(stale),
                    "dialogue_text_path": str(stale), "speaker_status": "partial",
                    "summary_json_path": str(stale), "speaker_map_path": str(stale),
                    "enhancement": {"status": "partial", "enabled": True, "enhanced_text_path": str(stale)}}
                write_json(item / "metadata.json", old)
                cached = transcribe_single_source(**request)
                self.assertEqual(cached["status"], "cached")
                self.assertEqual(asr.call_count, 1)
                self.assertFalse(cached["enhancement"]["enabled"])
                self.assertFalse(cached.get("identity_text_path"))
                self.assertFalse(cached.get("dialogue_text_path"))
                self.assertEqual(cached["cache_artifact_hashes"], first["cache_artifact_hashes"])
                receipt = publish_pipeline(run, job_id="cache-delivery", results=[cached],
                    plan=plan_request(PipelineRequest((str(source),), profile="fast", postprocess="off", speaker_mode="off"), load_config()).to_dict())
                self.assertEqual(receipt["status"], "ready")
                rendered = Path(receipt["primary_file"]).read_text()
                self.assertIn("本轮只请求原始识别全文", rendered)
                self.assertNotIn("OLD_DERIVATIVE_SHOULD_NOT_BE_SELECTED", rendered)
                self.assertNotIn(str(stale.resolve()), receipt["artifact_hashes"])
                self.assertTrue(stale.is_file())
                self.assertEqual(file_hash(item / "transcript.json"), raw_sha)


if __name__ == "__main__":
    unittest.main()
