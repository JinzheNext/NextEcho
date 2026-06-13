from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from workbench.sources import SourceDescriptor, resolve_feed, resolve_source
from workbench.transcription import (
    LONG_AUDIO_CPU_FALLBACK_MS,
    _append_shifted_srt,
    _append_shifted_vtt,
    _which_binary,
    format_transcript_attribution,
    transcribe_audio_file,
    transcribe_media_sources,
)


class SourceResolverTests(unittest.TestCase):
    def test_which_binary_uses_stable_local_lookup(self) -> None:
        with patch("workbench.transcription.shutil.which", return_value="/usr/bin/curl") as mock_which:
            self.assertEqual(_which_binary("curl"), "/usr/bin/curl")
        mock_which.assert_called_once_with("curl")

    def test_resolve_youtube_with_ytdlp_metadata(self) -> None:
        payload = {
            "title": "Andrej Karpathy: From Vibe Coding to Agentic Engineering",
            "uploader": "Y Combinator",
            "duration": 1234,
            "upload_date": "20260601",
            "webpage_url": "https://www.youtube.com/watch?v=96jN2OCOfLs",
            "url": "https://media.example/video.mp4",
        }
        completed = MagicMock(stdout=json.dumps(payload))
        with patch("workbench.sources.subprocess.run", return_value=completed):
            descriptor = resolve_source("https://www.youtube.com/watch?v=96jN2OCOfLs")
        self.assertEqual(descriptor.platform, "youtube")
        self.assertEqual(descriptor.resolver, "yt_dlp")
        self.assertEqual(descriptor.title, payload["title"])
        self.assertEqual(descriptor.author, payload["uploader"])
        self.assertEqual(descriptor.duration_seconds, 1234)
        self.assertEqual(descriptor.published_at, "2026-06-01")
        self.assertEqual(descriptor.resolved_media_url, payload["url"])
        self.assertFalse(descriptor.error)

    def test_resolve_xiaoyuzhou_from_page_metadata(self) -> None:
        html = """
        <html>
          <head>
            <meta property="og:title" content="E237 出发吧，扔掉旧地图，去 AI 时代的新世界尽情探索">
            <meta property="og:site_name" content="知行小酒馆">
          </head>
          <body>
            <script>
              window.__DATA__ = {"audioUrl":"https:\\/\\/media.example\\/episode.m4a","pubDate":"2026-06-01"};
            </script>
            <div>87分钟</div>
          </body>
        </html>
        """
        response = MagicMock()
        response.read.return_value = html.encode("utf-8")
        response.headers.get_content_charset.return_value = "utf-8"
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        with patch("workbench.sources.urlopen", return_value=response):
            descriptor = resolve_source("https://www.xiaoyuzhoufm.com/episode/6a19390a7460cabdeb57c0e5")
        self.assertEqual(descriptor.platform, "xiaoyuzhou")
        self.assertEqual(descriptor.resolver, "xiaoyuzhou_page")
        self.assertEqual(descriptor.title, "E237 出发吧，扔掉旧地图，去 AI 时代的新世界尽情探索")
        self.assertEqual(descriptor.author, "知行小酒馆")
        self.assertEqual(descriptor.duration_seconds, 87 * 60)
        self.assertEqual(descriptor.resolved_media_url, "https://media.example/episode.m4a")
        self.assertFalse(descriptor.error)

    def test_resolve_feed_returns_enclosure_items(self) -> None:
        xml = """
        <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
          <channel>
            <title>知行小酒馆</title>
            <item>
              <title>E237 出发吧</title>
              <link>https://www.xiaoyuzhoufm.com/episode/6a19390a7460cabdeb57c0e5</link>
              <author>知行小酒馆</author>
              <pubDate>Sat, 01 Jun 2026 10:00:00 GMT</pubDate>
              <itunes:duration>01:27:00</itunes:duration>
              <enclosure url="https://media.example/feed-episode.m4a" type="audio/mpeg" />
            </item>
          </channel>
        </rss>
        """
        response = MagicMock()
        response.read.return_value = xml.encode("utf-8")
        response.headers.get_content_charset.return_value = "utf-8"
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        with patch("workbench.sources.urlopen", return_value=response):
            descriptors = resolve_feed("https://example.com/feed.xml", limit=2)
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0].platform, "rss")
        self.assertEqual(descriptors[0].duration_seconds, 5220)
        self.assertEqual(descriptors[0].resolved_media_url, "https://media.example/feed-episode.m4a")
        self.assertFalse(descriptors[0].error)

    def test_transcription_manifest_keeps_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_media = root / "sample.mp4"
            source_media.write_bytes(b"media")
            descriptor = SourceDescriptor(
                source_type="page",
                platform="bilibili",
                input="https://www.bilibili.com/video/BV1g6okBLEtL/",
                canonical_url="https://www.bilibili.com/video/BV1g6okBLEtL/",
                resolved_media_url="",
                title="Bilibili sample",
                author="Uploader",
                duration_seconds=42,
                published_at="2026-06-06",
                source_label="001_bilibili-sample",
                resolver="yt_dlp",
            )
            output_dir = root / "run_test"

            def fake_prepare_media_source(_, __):
                return source_media

            def fake_extract_audio(**kwargs):
                Path(kwargs["target_path"]).write_bytes(b"audio")
                return Path(kwargs["target_path"])

            def fake_run_whisper_cli(**kwargs):
                base = Path(kwargs["output_dir"]) / kwargs["output_base"]
                (base.with_suffix(".txt")).write_text("hello world", encoding="utf-8")
                (base.with_suffix(".json")).write_text("{}", encoding="utf-8")
                (base.with_suffix(".srt")).write_text("1", encoding="utf-8")
                (base.with_suffix(".vtt")).write_text("WEBVTT", encoding="utf-8")

            with patch("workbench.transcription.resolve_whisper_model", return_value=root / "model.bin"):
                (root / "model.bin").write_bytes(b"model")
                with patch("workbench.transcription._which_binary", side_effect=["/usr/bin/whisper-cli", "/usr/bin/ffmpeg"]):
                    with patch("workbench.transcription.prepare_media_source", side_effect=fake_prepare_media_source):
                        with patch("workbench.transcription.extract_audio", side_effect=fake_extract_audio):
                            with patch("workbench.transcription.run_whisper_cli", side_effect=fake_run_whisper_cli):
                                with patch("workbench.transcription.probe_audio_duration_ms", return_value=60_000):
                                    payload = transcribe_media_sources([descriptor], output_dir=output_dir)
            item = payload["results"][0]
            self.assertEqual(item["platform"], "bilibili")
            self.assertEqual(item["title"], "Bilibili sample")
            self.assertEqual(item["resolver"], "yt_dlp")
            transcript_text = Path(item["text_path"]).read_text(encoding="utf-8")
            self.assertEqual(transcript_text, format_transcript_attribution("hello world"))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["results"][0]["platform"], "bilibili")

    def test_transcription_writes_errors_json_when_item_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            descriptor = SourceDescriptor(
                source_type="page",
                platform="youtube",
                input="https://www.youtube.com/watch?v=96jN2OCOfLs",
                canonical_url="https://www.youtube.com/watch?v=96jN2OCOfLs",
                resolved_media_url="",
                title="Broken sample",
                author="Y Combinator",
                duration_seconds=10,
                published_at="2026-06-06",
                source_label="001_broken-sample",
                resolver="yt_dlp",
            )
            output_dir = root / "run_failed"
            with patch("workbench.transcription.resolve_whisper_model", return_value=root / "model.bin"):
                (root / "model.bin").write_bytes(b"model")
                with patch("workbench.transcription._which_binary", side_effect=["/usr/bin/whisper-cli", "/usr/bin/ffmpeg"]):
                    with patch("workbench.transcription.prepare_media_source", side_effect=RuntimeError("download failed")):
                        payload = transcribe_media_sources([descriptor], output_dir=output_dir)
            self.assertTrue(payload["errors_path"])
            errors = json.loads(Path(payload["errors_path"]).read_text(encoding="utf-8"))
            self.assertEqual(errors["items"][0]["status"], "failed")
            self.assertIn("download failed", errors["items"][0]["error"])

    def test_shifted_subtitles_keep_global_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            srt_path = root / "chunk.srt"
            vtt_path = root / "chunk.vtt"
            srt_path.write_text("1\n00:00:05,000 --> 00:00:07,500\n你好\n", encoding="utf-8")
            vtt_path.write_text("WEBVTT\n\n00:00:05.000 --> 00:00:07.500\n你好\n", encoding="utf-8")
            srt_lines: list[str] = []
            vtt_lines: list[str] = ["WEBVTT", ""]
            next_index = _append_shifted_srt(srt_path, srt_lines, 60_000, 1)
            _append_shifted_vtt(vtt_path, vtt_lines, 60_000)
            self.assertEqual(next_index, 2)
            self.assertIn("00:01:05,000 --> 00:01:07,500", "\n".join(srt_lines))
            self.assertIn("00:01:05.000 --> 00:01:07.500", "\n".join(vtt_lines))

    def test_long_audio_uses_chunked_cpu_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"audio")
            with patch("workbench.transcription.probe_audio_duration_ms", return_value=LONG_AUDIO_CPU_FALLBACK_MS):
                with patch("workbench.transcription.transcribe_audio_file_in_chunks") as chunked:
                    transcribe_audio_file(
                        audio_path=audio_path,
                        output_dir=root,
                        output_base="transcript",
                        whisper_bin="/usr/bin/whisper-cli",
                        model_path=root / "model.bin",
                        language="auto",
                        output_formats=["txt"],
                    )
            chunked.assert_called_once()

    def test_format_transcript_attribution_wraps_plain_text(self) -> None:
        rendered = format_transcript_attribution("你好，世界。")
        self.assertTrue(rendered.startswith("本文转录由 GitHub 项目：NextEcho 提供支持"))
        self.assertIn("你好，世界。", rendered)
        self.assertIn("powered by GitHub repo NextEcho", rendered)


if __name__ == "__main__":
    unittest.main()
