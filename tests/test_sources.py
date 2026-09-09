from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from workbench.sources import SourceDescriptor, _cookies_browser_candidates, build_ytdlp_command, resolve_feed, resolve_source
from workbench.transcription import (
    LONG_AUDIO_CPU_FALLBACK_MS,
    _append_shifted_srt,
    _append_shifted_vtt,
    _infer_source_label,
    _which_binary,
    format_transcript_attribution,
    list_supported_whisper_models,
    recommend_whisper_model,
    resolve_requested_model,
    transcribe_audio_file,
    transcribe_media_sources,
)


class SourceResolverTests(unittest.TestCase):
    def test_transcribe_media_sources_records_elapsed_after_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            descriptor = SourceDescriptor(
                input=str(root / "sample.mp3"),
                source_type="local_file",
                platform="local",
                canonical_url="",
                resolved_media_url="",
                title="sample",
                author="",
                duration_seconds=0,
                published_at="",
                resolver="direct",
                source_label="001_sample",
            )
            transcript_dir = root / "out"
            fake_time = iter([10.0, 16.4])

            def fake_transcribe_single_source(**_: object) -> dict:
                item_dir = transcript_dir / "items" / "001_sample"
                item_dir.mkdir(parents=True, exist_ok=True)
                transcript_path = item_dir / "transcript.txt"
                transcript_path.write_text("hello", encoding="utf-8")
                return {
                    "source": descriptor.input,
                    "source_label": "001_sample",
                    "status": "ready",
                    "created_at": "2026-06-13T00:00:00Z",
                    "updated_at": "2026-06-13T00:00:00Z",
                    "char_count": 5,
                    "text_path": str(transcript_path),
                    "json_path": "",
                    "srt_path": "",
                    "vtt_path": "",
                    "media_path": "",
                    "audio_path": "",
                    "asr": {
                        "actual_engine": "qwen3-asr",
                        "actual_model": "owner/qwen-5bit",
                        "provenance": {"revision": "abc123", "sha256": "deadbeef"},
                    },
                    "output_dir": str(item_dir),
                    "excerpt": "hello",
                    "error": "",
                }

            with patch("workbench.transcription.time.monotonic", side_effect=lambda: next(fake_time)):
                with patch("workbench.transcription.resolve_whisper_model", return_value=Path("/tmp/model.bin")):
                    with patch("workbench.transcription._which_binary", side_effect=lambda name: f"/usr/bin/{name}"):
                        with patch("workbench.transcription.transcribe_single_source", side_effect=fake_transcribe_single_source):
                            payload = transcribe_media_sources([descriptor], output_dir=transcript_dir)

            self.assertEqual(payload["elapsed_seconds"], 6)
            manifest = json.loads((transcript_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["elapsed_seconds"], 6)
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(manifest["asr"]["actual_engine"], "qwen3-asr")
            self.assertEqual(manifest["asr"]["provenance"][0]["revision"], "abc123")
            self.assertTrue(manifest["completed_at"])
            run_config = json.loads((transcript_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["elapsed_seconds"], 6)
            self.assertTrue(run_config["completed_at"])
            self.assertEqual(run_config["schema_version"], 3)

    def test_which_binary_uses_stable_local_lookup(self) -> None:
        with patch("workbench.binaries.shutil.which", return_value="/usr/bin/curl") as mock_which:
            self.assertEqual(_which_binary("curl"), "/usr/bin/curl")
        self.assertGreaterEqual(mock_which.call_count, 1)
        self.assertEqual(mock_which.call_args_list[0].args[0], "curl")

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

    def test_resolve_apple_podcasts_page_metadata(self) -> None:
        html = """
        <html>
          <head>
            <title>The Daily - Podcast - Apple Podcasts</title>
            <meta property="og:title" content="The Daily">
          </head>
          <body>
            <div>Creator</div>
            <div>The New York Times</div>
            <a href="https://www.nytimes.com/the-daily">Show Website</a>
            <div>37 min</div>
          </body>
        </html>
        """
        publisher_html = """
        <html>
          <head>
            <link rel="alternate" type="application/rss+xml" href="https://example.com/feed.xml">
          </head>
        </html>
        """
        rss_xml = """
        <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
          <channel>
            <item>
              <title>The Daily</title>
              <link>https://www.nytimes.com/the-daily</link>
              <itunes:duration>37:00</itunes:duration>
              <enclosure url="https://cdn.example.com/daily-episode.mp3" />
            </item>
          </channel>
        </rss>
        """
        response = MagicMock()
        response.read.return_value = html.encode("utf-8")
        response.headers.get_content_charset.return_value = "utf-8"
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        publisher_response = MagicMock()
        publisher_response.read.return_value = publisher_html.encode("utf-8")
        publisher_response.headers.get_content_charset.return_value = "utf-8"
        publisher_response.__enter__.return_value = publisher_response
        publisher_response.__exit__.return_value = None
        rss_response = MagicMock()
        rss_response.read.return_value = rss_xml.encode("utf-8")
        rss_response.headers.get_content_charset.return_value = "utf-8"
        rss_response.__enter__.return_value = rss_response
        rss_response.__exit__.return_value = None
        with patch("workbench.sources.urlopen", side_effect=[response, publisher_response, rss_response]):
            descriptor = resolve_source("https://podcasts.apple.com/us/podcast/the-daily/id1200361736")
        self.assertEqual(descriptor.platform, "apple_podcasts")
        self.assertEqual(descriptor.resolver, "apple_podcasts_page")
        self.assertEqual(descriptor.title, "The Daily")
        self.assertEqual(descriptor.resolved_media_url, "https://cdn.example.com/daily-episode.mp3")
        self.assertFalse(descriptor.error)

    def test_resolve_apple_podcasts_fallback_contains_rss_candidates(self) -> None:
        html = """
        <html>
          <head>
            <title>岩中花述 - Apple Podcasts</title>
          </head>
          <body>
            <a href="https://podcast.example.com">Show Website</a>
          </body>
        </html>
        """
        response = MagicMock()
        response.read.return_value = html.encode("utf-8")
        response.headers.get_content_charset.return_value = "utf-8"
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        publisher_response = MagicMock()
        publisher_response.read.return_value = "<html></html>".encode("utf-8")
        publisher_response.headers.get_content_charset.return_value = "utf-8"
        publisher_response.__enter__.return_value = publisher_response
        publisher_response.__exit__.return_value = None
        with patch("workbench.sources.urlopen", side_effect=[response, publisher_response]):
            descriptor = resolve_source("https://podcasts.apple.com/cn/podcast/example/id1?i=100")
        self.assertIn("已识别 Apple Podcasts 页面", descriptor.error)
        self.assertEqual(descriptor.show_website, "https://podcast.example.com")

    def test_build_ytdlp_command_for_youtube_enables_ejs_and_js_runtimes(self) -> None:
        with patch("workbench.sources.find_binary", side_effect=lambda name: {
            "yt-dlp": "/usr/local/bin/yt-dlp",
            "deno": "/usr/local/bin/deno",
            "node": "/usr/local/bin/node",
        }.get(name)):
            command = build_ytdlp_command(
                "https://www.youtube.com/watch?v=pyqUiHyz_-c",
                platform="youtube",
                command_kind="metadata",
            )
        self.assertIn("--remote-components", command)
        self.assertIn("ejs:github", command)
        self.assertIn("deno:/usr/local/bin/deno", command)
        self.assertIn("node:/usr/local/bin/node", command)

    def test_build_ytdlp_command_prefers_cookies_file_when_configured(self) -> None:
        with patch.dict("workbench.sources.os.environ", {"TRANSCRIBE_YTDLP_COOKIES_FILE": "/tmp/cookies.txt"}, clear=False):
            with patch("workbench.sources.find_binary", side_effect=lambda name: "/usr/local/bin/yt-dlp" if name == "yt-dlp" else None):
                command = build_ytdlp_command(
                    "https://www.youtube.com/watch?v=pyqUiHyz_-c",
                    platform="youtube",
                    command_kind="metadata",
                    browser="chrome",
                )
        self.assertIn("--cookies", command)
        self.assertIn("/tmp/cookies.txt", command)
        self.assertNotIn("--cookies-from-browser", command)

    def test_resolve_bilibili_retries_with_browser_cookies_on_412(self) -> None:
        failure = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: [BiliBili] BV1soizB5Eog: Unable to download webpage: HTTP Error 412: Precondition Failed",
        )
        payload = {
            "title": "Bilibili sample",
            "uploader": "Uploader",
            "duration": 120,
            "upload_date": "20260601",
            "webpage_url": "https://www.bilibili.com/video/BV1soizB5Eog",
            "url": "https://media.example/video.mp4",
        }
        completed = MagicMock(stdout=json.dumps(payload))
        with patch("workbench.sources.os_platform.system", return_value="Darwin"):
            with patch("workbench.sources.subprocess.run", side_effect=[failure, completed]) as mock_run:
                descriptor = resolve_source("https://www.bilibili.com/video/BV1soizB5Eog")
        self.assertFalse(descriptor.error)
        retry_command = mock_run.call_args_list[1].args[0]
        self.assertIn("--cookies-from-browser", retry_command)
        self.assertIn("chrome", retry_command)

    def test_resolve_bilibili_formats_antibot_error(self) -> None:
        failure = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: [BiliBili] BV1soizB5Eog: Unable to download webpage: HTTP Error 412: Precondition Failed",
        )
        with patch("workbench.sources.subprocess.run", side_effect=failure):
            descriptor = resolve_source("https://www.bilibili.com/video/BV1soizB5Eog")
        self.assertIn("Bilibili blocked the request with anti-bot protection", descriptor.error)
        self.assertIn("update yt-dlp", descriptor.error)

    def test_resolve_youtube_retries_with_browser_cookies_on_bot_check(self) -> None:
        failure = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: [youtube] pyqUiHyz_-c: Sign in to confirm you're not a bot.",
        )
        payload = {
            "title": "YouTube sample",
            "uploader": "Uploader",
            "duration": 180,
            "upload_date": "20260601",
            "webpage_url": "https://www.youtube.com/watch?v=pyqUiHyz_-c",
            "url": "https://media.example/video.mp4",
        }
        completed = MagicMock(stdout=json.dumps(payload))
        with patch("workbench.sources.os_platform.system", return_value="Darwin"):
            with patch("workbench.sources._browser_profile_exists", side_effect=lambda browser: browser == "firefox"):
                with patch("workbench.sources.subprocess.run", side_effect=[failure, completed]) as mock_run:
                    descriptor = resolve_source("https://www.youtube.com/watch?v=pyqUiHyz_-c")
        self.assertFalse(descriptor.error)
        retry_command = mock_run.call_args_list[1].args[0]
        self.assertIn("--cookies-from-browser", retry_command)
        self.assertIn("firefox", retry_command)

    def test_resolve_youtube_formats_bot_check_error(self) -> None:
        failure = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: [youtube] pyqUiHyz_-c: Sign in to confirm you're not a bot.",
        )
        with patch("workbench.sources.subprocess.run", side_effect=failure):
            descriptor = resolve_source("https://www.youtube.com/watch?v=pyqUiHyz_-c")
        self.assertIn("YouTube asked for a bot check", descriptor.error)
        self.assertIn("browser cookies", descriptor.error)

    def test_resolve_youtube_formats_reload_cookie_error(self) -> None:
        failure = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: [youtube] pyqUiHyz_-c: The page needs to be reloaded.",
        )
        with patch("workbench.sources.subprocess.run", side_effect=failure):
            descriptor = resolve_source("https://www.youtube.com/watch?v=pyqUiHyz_-c")
        self.assertIn("cookies stale", descriptor.error)
        self.assertIn("TRANSCRIBE_YTDLP_COOKIES_FILE", descriptor.error)

    def test_resolve_permission_error_formats_macos_cookie_guidance(self) -> None:
        failure = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: [Errno 1] Operation not permitted: '/Users/test/Library/Containers/com.apple.Safari/Data/Library/Cookies/Cookies.binarycookies'",
        )
        with patch("workbench.sources.subprocess.run", side_effect=failure):
            descriptor = resolve_source("https://www.youtube.com/watch?v=pyqUiHyz_-c")
        self.assertIn("macOS blocked access", descriptor.error)
        self.assertIn("firefox or chrome", descriptor.error)

    def test_cookie_browser_candidates_skip_missing_profiles_on_macos(self) -> None:
        def fake_exists(self: Path) -> bool:
            return str(self).endswith("/Google/Chrome/Default/Cookies")

        with patch("workbench.sources.os_platform.system", return_value="Darwin"):
            with patch("pathlib.Path.exists", fake_exists):
                candidates = _cookies_browser_candidates("youtube")
        self.assertIn("chrome", candidates)
        self.assertNotIn("edge", candidates)

    def test_resolve_configured_missing_browser_falls_back_to_next_candidate(self) -> None:
        initial_antibot = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: [youtube] pyqUiHyz_-c: Sign in to confirm you're not a bot.",
        )
        missing_edge = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr='ERROR: could not find edge cookies database in "/Users/test/Library/Application Support/Microsoft Edge"',
        )
        payload = {
            "title": "YouTube sample",
            "uploader": "Uploader",
            "duration": 180,
            "upload_date": "20260601",
            "webpage_url": "https://www.youtube.com/watch?v=pyqUiHyz_-c",
            "url": "https://media.example/video.mp4",
        }
        completed = MagicMock(stdout=json.dumps(payload))
        with patch.dict("workbench.sources.os.environ", {"TRANSCRIBE_YTDLP_BROWSER": "edge"}, clear=False):
            with patch("workbench.sources.os_platform.system", return_value="Darwin"):
                with patch("workbench.sources._browser_profile_exists", side_effect=lambda browser: browser == "chrome"):
                    with patch("workbench.sources.subprocess.run", side_effect=[initial_antibot, missing_edge, completed]) as mock_run:
                        descriptor = resolve_source("https://www.youtube.com/watch?v=pyqUiHyz_-c")
        self.assertFalse(descriptor.error)
        retry_command = mock_run.call_args_list[2].args[0]
        self.assertIn("--cookies-from-browser", retry_command)
        self.assertIn("chrome", retry_command)

    def test_resolve_error_includes_attempt_history(self) -> None:
        initial_antibot = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: [youtube] pyqUiHyz_-c: Sign in to confirm you're not a bot.",
        )
        missing_chrome = subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr='ERROR: could not find chrome cookies database in "/Users/test/Library/Application Support/Google/Chrome"',
        )
        with patch("workbench.sources._browser_profile_exists", side_effect=lambda browser: browser == "chrome"):
            with patch("workbench.sources.subprocess.run", side_effect=[initial_antibot, missing_chrome]):
                descriptor = resolve_source("https://www.youtube.com/watch?v=pyqUiHyz_-c")
        self.assertIn("Tried:", descriptor.error)
        self.assertIn("direct:", descriptor.error)
        self.assertIn("chrome:", descriptor.error)

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
                                    payload = transcribe_media_sources(
                                        [descriptor],
                                        output_dir=output_dir,
                                        asr_engine="whisper-cpp",
                                    )
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
            self.assertEqual(chunked.call_args.kwargs["model_path"], root / "model.bin")

    def test_format_transcript_attribution_wraps_plain_text(self) -> None:
        rendered = format_transcript_attribution("你好，世界。")
        self.assertTrue(rendered.startswith("本文转录由 GitHub 项目：NextEcho 提供支持"))
        self.assertIn("你好，世界。", rendered)
        self.assertIn("powered by github.com/JinzheNext/NextEcho - 让播客自动转录进知识库", rendered)

    def test_generic_local_media_label_is_human_readable(self) -> None:
        self.assertEqual(_infer_source_label("/tmp/source.m4a", 1), "001_录音原声")

    def test_recommend_whisper_model_respects_memory_tiers(self) -> None:
        self.assertEqual(recommend_whisper_model(None)[0], "base")
        self.assertEqual(recommend_whisper_model(4.0)[0], "tiny")
        self.assertEqual(recommend_whisper_model(7.0)[0], "base")
        self.assertEqual(recommend_whisper_model(12.0)[0], "small")
        self.assertEqual(recommend_whisper_model(20.0)[0], "medium")

    def test_resolve_requested_model_prefers_explicit_value(self) -> None:
        self.assertEqual(resolve_requested_model("small", "accurate", memory_gb=32.0), "small")
        self.assertEqual(resolve_requested_model(None, "fast", memory_gb=32.0), "base")
        self.assertEqual(resolve_requested_model(None, "accurate", memory_gb=12.0), "small")

    def test_list_supported_whisper_models_has_expected_entries(self) -> None:
        names = {item["name"] for item in list_supported_whisper_models()}
        self.assertIn("tiny", names)
        self.assertIn("base", names)
        self.assertIn("small", names)
        self.assertIn("medium", names)
        self.assertIn("large-v3-turbo-q5_0", names)


if __name__ == "__main__":
    unittest.main()
