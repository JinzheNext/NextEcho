from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import urlopen
from xml.etree import ElementTree


YTDLP_PLATFORMS = {"youtube", "bilibili", "generic"}
DIRECT_MEDIA_TOKENS = [".mp3", ".mp4", ".m4a", ".wav", ".flac", ".aac", "/stream/"]
RSS_HINTS = [".xml", "/feed", "/rss"]


@dataclass
class SourceDescriptor:
    source_type: str
    platform: str
    input: str
    canonical_url: str
    resolved_media_url: str
    title: str
    author: str
    duration_seconds: int
    published_at: str
    source_label: str
    resolver: str
    requires_auth: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_sources(inputs: list[str]) -> list[SourceDescriptor]:
    return [resolve_source(item, index=index) for index, item in enumerate(inputs, start=1)]


def resolve_feed(feed_url: str, *, limit: int = 5) -> list[SourceDescriptor]:
    canonical_url = canonicalize_url(feed_url, "generic")
    xml_text, fetch_error = _fetch_text(canonical_url)
    if not xml_text:
        return [
            SourceDescriptor(
                source_type="feed",
                platform="rss",
                input=feed_url,
                canonical_url=canonical_url,
                resolved_media_url="",
                title="",
                author="",
                duration_seconds=0,
                published_at="",
                source_label="001_rss-feed",
                resolver="rss",
                error=fetch_error or "Failed to fetch RSS feed.",
            )
        ]
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        return [
            SourceDescriptor(
                source_type="feed",
                platform="rss",
                input=feed_url,
                canonical_url=canonical_url,
                resolved_media_url="",
                title="",
                author="",
                duration_seconds=0,
                published_at="",
                source_label="001_rss-feed",
                resolver="rss",
                error=f"Unreadable RSS XML: {exc}",
            )
        ]
    channel_title = (root.findtext(".//channel/title") or "").strip()
    feed_items: list[SourceDescriptor] = []
    for index, item in enumerate(root.findall(".//item")[:limit], start=1):
        enclosure = item.find("enclosure")
        audio_url = (enclosure.attrib.get("url") or "").strip() if enclosure is not None else ""
        title = (item.findtext("title") or "").strip()
        author = (item.findtext("author") or item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}author") or channel_title).strip()
        published_at = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()
        descriptor = SourceDescriptor(
            source_type="feed_item",
            platform="rss",
            input=feed_url,
            canonical_url=link or canonical_url,
            resolved_media_url=audio_url,
            title=title or f"Episode {index}",
            author=author,
            duration_seconds=_parse_duration_seconds(
                item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration") or ""
            ),
            published_at=published_at,
            source_label=f"{index:03d}_{slugify(title or 'episode')}",
            resolver="rss",
            error="" if audio_url else "RSS item is missing an enclosure URL.",
        )
        feed_items.append(descriptor)
    if not feed_items:
        feed_items.append(
            SourceDescriptor(
                source_type="feed",
                platform="rss",
                input=feed_url,
                canonical_url=canonical_url,
                resolved_media_url="",
                title=channel_title,
                author="",
                duration_seconds=0,
                published_at="",
                source_label="001_rss-feed",
                resolver="rss",
                error="RSS feed did not contain any <item> entries.",
            )
        )
    return feed_items


def resolve_source(source: str, *, index: int = 1) -> SourceDescriptor:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        platform = detect_platform(source)
        canonical_url = canonicalize_url(source, platform)
        if looks_like_feed_url(source):
            feed_items = resolve_feed(source, limit=1)
            return feed_items[0]
        if looks_like_direct_media_url(source):
            return SourceDescriptor(
                source_type="direct_media",
                platform=platform,
                input=source,
                canonical_url=canonical_url,
                resolved_media_url=source,
                title=Path(parsed.path).name or canonical_url,
                author="",
                duration_seconds=0,
                published_at="",
                source_label=f"{index:03d}_{slugify(Path(parsed.path).stem or platform)}",
                resolver="direct",
            )
        if platform == "xiaoyuzhou":
            return _resolve_xiaoyuzhou(source, canonical_url, index=index)
        return _resolve_with_ytdlp(source, canonical_url, platform, index=index)

    local_path = Path(source).expanduser().resolve()
    if local_path.exists():
        return SourceDescriptor(
            source_type="local_file",
            platform="local",
            input=str(local_path),
            canonical_url=str(local_path),
            resolved_media_url="",
            title=local_path.name,
            author="",
            duration_seconds=0,
            published_at="",
            source_label=f"{index:03d}_{slugify(local_path.stem)}",
            resolver="local",
        )

    return SourceDescriptor(
        source_type="unknown",
        platform="generic",
        input=source,
        canonical_url=source,
        resolved_media_url="",
        title="",
        author="",
        duration_seconds=0,
        published_at="",
        source_label=f"{index:03d}_{slugify(source)}",
        resolver="unknown",
        error="Input is neither a readable local file nor a supported URL.",
    )


def detect_platform(source: str) -> str:
    hostname = (urlparse(source).hostname or "").lower()
    if "youtube.com" in hostname or "youtu.be" in hostname:
        return "youtube"
    if "bilibili.com" in hostname or "b23.tv" in hostname:
        return "bilibili"
    if "xiaoyuzhoufm.com" in hostname:
        return "xiaoyuzhou"
    return "generic"


def canonicalize_url(source: str, platform: str) -> str:
    parsed = urlparse(source)
    if platform == "youtube":
        if "youtu.be" in (parsed.hostname or ""):
            video_id = parsed.path.strip("/")
            query = urlencode({"v": video_id}) if video_id else ""
            return urlunparse(("https", "www.youtube.com", "/watch", "", query, ""))
        query_items = parse_qs(parsed.query)
        params: dict[str, str] = {}
        if query_items.get("v"):
            params["v"] = query_items["v"][0]
        if query_items.get("list"):
            params["list"] = query_items["list"][0]
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urlencode(params), ""))
    if platform in {"bilibili", "xiaoyuzhou"}:
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, ""))


def looks_like_direct_media_url(source: str) -> bool:
    lowered = source.lower()
    return any(token in lowered for token in DIRECT_MEDIA_TOKENS)


def looks_like_feed_url(source: str) -> bool:
    lowered = source.lower()
    return any(token in lowered for token in RSS_HINTS)


def _resolve_with_ytdlp(source: str, canonical_url: str, platform: str, *, index: int) -> SourceDescriptor:
    payload, error = _dump_ytdlp_metadata(canonical_url)
    if payload is None:
        return SourceDescriptor(
            source_type="page",
            platform=platform,
            input=source,
            canonical_url=canonical_url,
            resolved_media_url="",
            title="",
            author="",
            duration_seconds=0,
            published_at="",
            source_label=f"{index:03d}_{slugify(platform)}",
            resolver="yt_dlp",
            error=error or "yt-dlp failed to resolve this page. Try updating yt-dlp or supplying browser cookies later.",
        )
    title = str(payload.get("title") or canonical_url)
    uploader = str(payload.get("uploader") or payload.get("channel") or payload.get("channel_id") or "")
    duration = int(payload.get("duration") or 0)
    published_at = _coerce_published_at(str(payload.get("upload_date") or ""))
    direct_url = str(payload.get("url") or "")
    source_label = f"{index:03d}_{slugify(title or platform)}"
    return SourceDescriptor(
        source_type="page",
        platform=platform,
        input=source,
        canonical_url=str(payload.get("webpage_url") or canonical_url),
        resolved_media_url=direct_url,
        title=title,
        author=uploader,
        duration_seconds=duration,
        published_at=published_at,
        source_label=source_label,
        resolver="yt_dlp",
    )


def _resolve_xiaoyuzhou(source: str, canonical_url: str, *, index: int) -> SourceDescriptor:
    html, fetch_error = _fetch_text(canonical_url)
    if not html:
        return SourceDescriptor(
            source_type="page",
            platform="xiaoyuzhou",
            input=source,
            canonical_url=canonical_url,
            resolved_media_url="",
            title="",
            author="",
            duration_seconds=0,
            published_at="",
            source_label=f"{index:03d}_xiaoyuzhou",
            resolver="xiaoyuzhou_page",
            error=fetch_error or "Failed to fetch Xiaoyuzhou page.",
        )

    title = _first_match(
        html,
        [
            r'<meta property="og:title" content="([^"]+)"',
            r'"title"\s*:\s*"([^"]+)"',
        ],
    )
    author = _first_match(
        html,
        [
            r'<meta property="og:site_name" content="([^"]+)"',
            r'"podcast"\s*:\s*\{[^}]*"title"\s*:\s*"([^"]+)"',
            r'"author"\s*:\s*"([^"]+)"',
        ],
    )
    duration_text = _first_match(
        html,
        [
            r'(\d+)\s*分钟',
            r'"duration"\s*:\s*(\d+)',
        ],
    )
    published_at = _first_match(
        html,
        [
            r'"pubDate"\s*:\s*"([^"]+)"',
            r'"datePublished"\s*:\s*"([^"]+)"',
        ],
    )
    audio_url = _first_match(
        html,
        [
            r'"audioUrl"\s*:\s*"([^"]+)"',
            r'"audio"\s*:\s*"([^"]+)"',
            r'<meta property="og:audio" content="([^"]+)"',
            r'"enclosureUrl"\s*:\s*"([^"]+)"',
        ],
    )
    source_label = f"{index:03d}_{slugify(title or 'xiaoyuzhou')}"
    if not audio_url:
        rss_url = _first_match(
            html,
            [
                r'<link[^>]+type="application/rss\+xml"[^>]+href="([^"]+)"',
                r'"rssUrl"\s*:\s*"([^"]+)"',
            ],
        )
        if rss_url:
            audio_url = _resolve_audio_from_rss(rss_url, canonical_url)
    error = ""
    if not audio_url:
        error = "Xiaoyuzhou page parsed, but no audio URL was found. Tried page metadata first, then RSS fallback."
    duration_seconds = int(duration_text) * 60 if duration_text.isdigit() else 0
    return SourceDescriptor(
        source_type="page",
        platform="xiaoyuzhou",
        input=source,
        canonical_url=canonical_url,
        resolved_media_url=html_unescape(audio_url),
        title=html_unescape(title or ""),
        author=html_unescape(author or ""),
        duration_seconds=duration_seconds,
        published_at=published_at,
        source_label=source_label,
        resolver="xiaoyuzhou_page",
        error=error,
    )


def _resolve_audio_from_rss(rss_url: str, episode_url: str) -> str:
    xml_text, _ = _fetch_text(rss_url)
    if not xml_text:
        return ""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return ""
    path_tail = episode_url.rstrip("/").split("/")[-1]
    for item in root.findall(".//item"):
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or "").strip()
        enclosure = item.find("enclosure")
        if path_tail and path_tail not in link and path_tail not in guid:
            continue
        if enclosure is not None:
            return (enclosure.attrib.get("url") or "").strip()
    first = root.find(".//item/enclosure")
    return (first.attrib.get("url") or "").strip() if first is not None else ""


def _fetch_text(url: str) -> tuple[str, str]:
    try:
        with urlopen(url, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace"), ""
    except Exception as exc:
        return "", str(exc)


def _dump_ytdlp_metadata(source: str) -> tuple[dict[str, Any] | None, str]:
    try:
        completed = subprocess.run(
            ["yt-dlp", "--dump-single-json", "--no-playlist", source],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return None, "yt-dlp is not installed."
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        return None, stderr or "yt-dlp failed to inspect this page."
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None, "yt-dlp returned unreadable JSON metadata."
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list) and payload["entries"]:
        first_entry = payload["entries"][0]
        if isinstance(first_entry, dict):
            return first_entry, ""
    return payload if isinstance(payload, dict) else None, ""


def _coerce_published_at(raw: str) -> str:
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def _parse_duration_seconds(raw: str) -> int:
    value = raw.strip()
    if not value:
        return 0
    if value.isdigit():
        return int(value)
    parts = value.split(":")
    if not all(part.isdigit() for part in parts):
        return 0
    total = 0
    for part in parts:
        total = total * 60 + int(part)
    return total


def _first_match(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return value or "item"


def html_unescape(value: str) -> str:
    return unescape(value).replace("\\u002F", "/").replace("\\/", "/")
