from __future__ import annotations

import json
import os
import platform as os_platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .binaries import find_binary
from .media_commands import run_media_command
from .contracts import PipelineError
from .io_utils import value_hash


YTDLP_PLATFORMS = {"youtube", "bilibili", "generic"}
DIRECT_MEDIA_TOKENS = [".mp3", ".mp4", ".m4a", ".wav", ".flac", ".aac", "/stream/"]
RSS_HINTS = [".xml", "/feed", "/rss"]
DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


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
    show_website: str = ""
    rss_candidates: list[str] = field(default_factory=list)
    resolution_notes: list[str] = field(default_factory=list)
    podcast_title: str = ""
    source_id: str = ""
    selection_evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # 旧显式构造的描述符保持原形；新解析选择另带版本化身份。
        for key in ("source_id", "selection_evidence"):
            if not data[key]:
                data.pop(key)
        return data


def absolute_media_inputs(inputs: list[str], *, cwd: Path | None = None) -> list[str]:
    """提交前固定调用者文件路径；worker工作目录不能改变输入含义。"""
    caller = cwd or Path.cwd()
    result = []
    for source in inputs:
        if not isinstance(source, str) or not source.strip():
            raise PipelineError("INVALID_INPUT", "媒体输入不能为空")
        if urlparse(source).scheme in {"http", "https"}:
            result.append(source)
        else:
            path = (caller / Path(source).expanduser()).resolve()
            if path.is_dir():
                raise PipelineError("INVALID_INPUT", "请输入本地媒体文件，不能提交目录")
            result.append(str(path))
    return result


def _freeze_descriptor(descriptor: SourceDescriptor) -> SourceDescriptor:
    evidence = descriptor.selection_evidence
    identity = {"schema_version": 1, "platform": descriptor.platform}
    if evidence.get("kind") == "rss_item":
        identity.update(feed=evidence["feed_url"], episode=evidence.get("guid")
                        or evidence.get("item_link") or descriptor.resolved_media_url)
    else:
        identity["canonical_url"] = descriptor.canonical_url or descriptor.input
    descriptor.source_id = "src_" + value_hash(identity)[:32]
    return descriptor


def resolve_sources(inputs: list[str]) -> list[SourceDescriptor]:
    return [resolve_source(item, index=index) for index, item in enumerate(inputs, start=1)]


def resolve_media_inputs(inputs: list[str], *, limit: int = 1) -> list[SourceDescriptor]:
    """One headless route, including latest episode selection from a show page."""
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    result = []
    for source in inputs:
        parsed = urlparse(source)
        if parsed.hostname in {"www.xiaoyuzhoufm.com", "xiaoyuzhoufm.com"} and parsed.path.startswith("/podcast/"):
            result.extend(resolve_podcast_page(source, limit=limit))
        elif looks_like_feed_url(source):
            result.extend(resolve_feed(source, limit=limit, latest_first=True))
        else:
            result.extend(resolve_sources([source]))
    if len(result) > 20:
        raise ValueError("A run supports at most 20 episodes")
    return [_freeze_descriptor(d) for d in result]


def resolve_podcast_page(source: str, *, limit: int = 1) -> list[SourceDescriptor]:
    from .episode_context import _extract_next_data
    html, error = _fetch_text(source)
    props = (_extract_next_data(html) or {}).get("props", {}).get("pageProps", {})
    podcast = props.get("podcast") or {}
    episodes = sorted((e for e in podcast.get("episodes", []) if isinstance(e, dict) and e.get("eid")),
                      key=lambda e: _publication_timestamp(str(e.get("pubDate") or "")), reverse=True)
    if not episodes:
        return [SourceDescriptor("feed", "xiaoyuzhou", source, source, "", "", "", 0, "", "001_podcast",
                "xiaoyuzhou_podcast", error=error or "节目主页中没有可解析的单集列表")]
    result = []
    for index, episode in enumerate(episodes[:limit], 1):
        enclosure = episode.get("enclosure") or {}
        media = episode.get("media") or {}
        url = enclosure.get("url") or (media.get("source") or {}).get("url") or ""
        private = bool(episode.get("isPrivateMedia"))
        title = str(episode.get("title") or episode["eid"])
        result.append(SourceDescriptor("page", "xiaoyuzhou", source,
            "https://www.xiaoyuzhoufm.com/episode/" + str(episode["eid"]), url if not private else "",
            title, str(podcast.get("author") or podcast.get("title") or ""), int(episode.get("duration") or 0),
            str(episode.get("pubDate") or ""), f"{index:03d}_{slugify(title)}", "xiaoyuzhou_page", requires_auth=private,
            error="节目需要用户授权，未选择其他单集替代" if private else "" if url else "未解析到单集媒体",
            show_website=source, podcast_title=str(podcast.get("title") or ""),
            resolution_notes=["show page selection: latest published episode", f"limit={limit}"]))
    return [_freeze_descriptor(d) for d in result]


def _publication_timestamp(value: str) -> float:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError):
            return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def resolve_feed(feed_url: str, *, limit: int = 5, latest_first: bool = False) -> list[SourceDescriptor]:
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
    items = root.findall(".//item")
    if latest_first:
        items.sort(key=lambda item: _publication_timestamp(item.findtext("pubDate") or ""), reverse=True)
    for index, item in enumerate(items[:limit], start=1):
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
            podcast_title=channel_title,
            selection_evidence={"schema_version": 1, "kind": "rss_item", "feed_url": canonical_url,
                "guid": (item.findtext("guid") or "").strip(), "item_link": link,
                "selection": "latest_by_pubDate" if latest_first else "publisher_order",
                "rank": index, "limit": limit},
        )
        feed_items.append(descriptor)
        if latest_first:
            descriptor.resolution_notes.append("latest published episode by pubDate; missing dates retain publisher order")
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
    return [_freeze_descriptor(d) for d in feed_items]


def resolve_source(source: str, *, index: int = 1) -> SourceDescriptor:
    return _freeze_descriptor(_resolve_source(source, index=index))


def _resolve_source(source: str, *, index: int = 1) -> SourceDescriptor:
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
        if platform == "apple_podcasts":
            return _resolve_apple_podcasts(source, canonical_url, index=index)
        return _resolve_with_ytdlp(source, canonical_url, platform, index=index)

    local_path = Path(source).expanduser().resolve()
    if local_path.is_file():
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
    if "podcasts.apple.com" in hostname:
        return "apple_podcasts"
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
    if platform == "apple_podcasts":
        query_items = parse_qs(parsed.query)
        params: dict[str, str] = {}
        if query_items.get("i"):
            params["i"] = query_items["i"][0]
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", urlencode(params), ""))
    if platform == "bilibili":
        parts = parse_qs(parsed.query, keep_blank_values=True).get("p", [])
        if parts and (len(parts) != 1 or not re.fullmatch(r"[0-9]+", parts[0]) or len(parts[0]) > 9 or int(parts[0]) < 1):
            raise PipelineError("INVALID_INPUT", "B站分P必须是单个正整数")
        query = urlencode({"p": int(parts[0])}) if parts else ""
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", query, ""))
    if platform == "xiaoyuzhou":
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
    resolved_url = str(payload.get("webpage_url") or canonical_url)
    selection = {}
    selection_error = ""
    if platform == "bilibili":
        requested = parse_qs(urlparse(canonical_url).query).get("p")
        returned = parse_qs(urlparse(resolved_url).query).get("p")
        id_part = re.search(r"_p(\d+)$", str(payload.get("id") or ""))
        observed_part = int(id_part.group(1)) if id_part else int(returned[0]) if returned and returned[0].isdigit() else None
        requested_part = int(requested[0]) if requested else observed_part or 1
        resolved_url = canonicalize_url(resolved_url, platform)
        parsed = urlparse(resolved_url)
        resolved_url = urlunparse(parsed._replace(query=urlencode({"p": requested_part})))
        selection = {"schema_version": 1, "kind": "bilibili_part", "part": requested_part,
                     "metadata_id": str(payload.get("id") or ""),
                     "requested_url": canonical_url, "observed_part": observed_part}
        if observed_part is not None and observed_part != requested_part:
            direct_url = ""
            selection_error = "B站返回的分P与明确请求不一致，未下载其他分P替代"
    return SourceDescriptor(
        source_type="page",
        platform=platform,
        input=source,
        canonical_url=resolved_url,
        resolved_media_url=direct_url,
        title=title,
        author=uploader,
        duration_seconds=duration,
        published_at=published_at,
        source_label=source_label,
        resolver="yt_dlp",
        selection_evidence=selection,
        error=selection_error,
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
    duration_json = _first_match(html, [r'"duration"\s*:\s*(\d+)'])
    duration_minutes = _first_match(html, [r'(\d+)\s*分钟'])
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
    duration_seconds = int(duration_json) if duration_json.isdigit() else int(duration_minutes) * 60 if duration_minutes.isdigit() else 0
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
        resolution_notes=["duration: structured seconds" if duration_json else "duration: displayed minutes" if duration_minutes else "duration: unavailable"],
    )


def _resolve_apple_podcasts(source: str, canonical_url: str, *, index: int) -> SourceDescriptor:
    html, fetch_error = _fetch_text(canonical_url)
    if not html:
        return SourceDescriptor(
            source_type="page",
            platform="apple_podcasts",
            input=source,
            canonical_url=canonical_url,
            resolved_media_url="",
            title="",
            author="",
            duration_seconds=0,
            published_at="",
            source_label=f"{index:03d}_apple-podcasts",
            resolver="apple_podcasts_page",
            error=fetch_error or "Failed to fetch Apple Podcasts page.",
        )

    json_ld_objects = _extract_json_ld_objects(html)
    title = _first_nonempty(
        _json_ld_value(json_ld_objects, "name", expected_types={"PodcastEpisode", "Episode"}),
        _first_match(
            html,
            [
                r"<title>([^<]+?)\s*-\s*Apple Podcasts</title>",
                r'<meta property="og:title" content="([^"]+)"',
                r"<h1[^>]*>([^<]+)</h1>",
            ],
        ),
    )
    title = re.sub(r"\s*-\s*Podcast\s*$", "", title).strip()
    author = _first_nonempty(
        _json_ld_author(json_ld_objects),
        _first_match(
            html,
            [
                r"<meta name=\"apple:title\" content=\"([^\"]+)\"",
                r"Creator\s*</[^>]+>\s*([^<]+)",
                r"Channel\s*</[^>]+>\s*([^<]+)",
            ],
        ),
    )
    duration_seconds = _first_int(
        _json_ld_duration_seconds(json_ld_objects),
        60 * _parse_duration_seconds(
            _first_match(
                html,
                [
                    r"(\d+)\s*min",
                    r"(\d+)\s*分钟",
                ],
            )
        ),
    )
    show_website = html_unescape(
        _first_nonempty(
            _json_ld_show_website(json_ld_objects),
            _first_match(
                html,
                [
                    r"Show Website\s*</[^>]+>\s*<[^>]+href=\"([^\"]+)\"",
                    r"节目网站\s*</[^>]+>\s*<[^>]+href=\"([^\"]+)\"",
                    r'<a[^>]+href="([^"]+)"[^>]*>\s*Show Website\s*</a>',
                    r'<a[^>]+href="([^"]+)"[^>]*>\s*节目网站\s*</a>',
                ],
            ),
        )
    )
    direct_media_url = _pick_media_url(_extract_media_urls_from_html(html, canonical_url) + _json_ld_media_urls(json_ld_objects))
    resolution_notes: list[str] = []
    if direct_media_url:
        resolution_notes.append("Apple page exposed a direct media URL.")
    publisher_episode_url = _first_nonempty(
        _json_ld_external_episode_url(json_ld_objects),
        _first_match(
            html,
            [
                r'"episodeUrl"\s*:\s*"([^"]+)"',
                r'"webUrl"\s*:\s*"([^"]+)"',
                r'<meta property="og:url" content="(https?://(?!podcasts\.apple\.com)[^"]+)"',
            ],
        ),
    )
    if publisher_episode_url:
        publisher_episode_url = html_unescape(publisher_episode_url)
        resolution_notes.append(f"Found publisher episode page: {publisher_episode_url}")

    rss_candidates = dedupe_strings(_discover_rss_candidates(canonical_url, html))
    if show_website and not direct_media_url:
        publisher_html, publisher_error = _fetch_text(show_website)
        if publisher_html:
            direct_media_url = _pick_media_url(_extract_media_urls_from_html(publisher_html, show_website))
            rss_candidates = dedupe_strings([*rss_candidates, *_discover_rss_candidates(show_website, publisher_html)])
            if direct_media_url:
                resolution_notes.append("Found direct media on the publisher site.")
        elif publisher_error:
            resolution_notes.append(f"Could not fetch show website: {publisher_error}")

    if publisher_episode_url and not direct_media_url:
        episode_html, episode_error = _fetch_text(publisher_episode_url)
        if episode_html:
            direct_media_url = _pick_media_url(_extract_media_urls_from_html(episode_html, publisher_episode_url))
            rss_candidates = dedupe_strings([*rss_candidates, *_discover_rss_candidates(publisher_episode_url, episode_html)])
            if direct_media_url:
                resolution_notes.append("Found direct media on the publisher episode page.")
        elif episode_error:
            resolution_notes.append(f"Could not fetch publisher episode page: {episode_error}")

    episode_id = _apple_episode_id(source)
    if not direct_media_url and rss_candidates:
        direct_media_url, matched_rss = _resolve_audio_from_rss_candidates(
            rss_candidates,
            episode_title=title,
            duration_seconds=duration_seconds,
            episode_id=episode_id,
            episode_url=publisher_episode_url or canonical_url,
        )
        if matched_rss:
            resolution_notes.append(f"Resolved audio from RSS: {matched_rss}")

    error = ""
    if not direct_media_url:
        error = "已识别 Apple Podcasts 页面，但还没有找到可转录的音频直链。"
        if show_website:
            error += f" 节目官网：{show_website}。"
        if rss_candidates:
            error += f" 已发现 RSS 候选：{', '.join(rss_candidates[:3])}。"
    return SourceDescriptor(
        source_type="page",
        platform="apple_podcasts",
        input=source,
        canonical_url=canonical_url,
        resolved_media_url=direct_media_url,
        title=html_unescape(title or ""),
        author=html_unescape(author or ""),
        duration_seconds=duration_seconds,
        published_at="",
        source_label=f"{index:03d}_{slugify(title or 'apple-podcasts')}",
        resolver="apple_podcasts_page",
        error=error,
        show_website=show_website,
        rss_candidates=rss_candidates,
        resolution_notes=resolution_notes,
    )


def _resolve_audio_from_rss(rss_url: str, episode_url: str) -> str:
    xml_text, _ = _fetch_text(rss_url)
    if not xml_text:
        return ""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return ""
    matches = []
    for item in root.findall(".//item"):
        if not _rss_episode_identity_matches(item, episode_url=episode_url):
            continue
        enclosure = item.find("enclosure")
        matches.append((enclosure.attrib.get("url") or "").strip() if enclosure is not None else "")
    # 找不到、缺媒体或冲突均不能改选第一集。相同条目的重复声明可合并。
    return urljoin(rss_url, matches[0]) if matches and all(matches) and len(set(matches)) == 1 else ""


def _rss_episode_identity_matches(item, *, episode_url: str, episode_id: str = "") -> bool:
    values = [(item.findtext(key) or "").strip() for key in ("link", "guid")]
    expected = canonicalize_url(episode_url, detect_platform(episode_url)) if episode_url else ""
    for value in values:
        if value and expected and urlparse(value).scheme in {"https", "http"}:
            if canonicalize_url(value, detect_platform(value)) == expected:
                return True
    # 不作子串匹配：missing不能命中missing-other；Apple查询ID独立保留。
    identity = episode_id or (urlparse(episode_url).path.rstrip("/").split("/")[-1] if episode_url else "")
    if not identity:
        return False
    return any(value == identity or (urlparse(value).scheme in {"https", "http"}
        and (urlparse(value).path.rstrip("/").split("/")[-1] == identity
             or parse_qs(urlparse(value).query).get("i") == [identity])) for value in values)


def _resolve_audio_from_rss_candidates(
    rss_candidates: list[str],
    *,
    episode_title: str,
    duration_seconds: int,
    episode_id: str,
    episode_url: str,
) -> tuple[str, str]:
    candidates = []
    for rss_url in rss_candidates:
        xml_text, _ = _fetch_text(rss_url)
        if not xml_text:
            continue
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            continue
        items = root.findall(".//item")
        if not items:
            continue
        for item in items:
            exact_title = bool(_normalized_match_text(episode_title)) and _normalized_match_text(item.findtext("title") or "") == _normalized_match_text(episode_title)
            exact_identity = _rss_episode_identity_matches(item, episode_url=episode_url, episode_id=episode_id)
            if not exact_title and not exact_identity:
                continue
            score = _score_rss_item(item, episode_title=episode_title, duration_seconds=duration_seconds,
                                    episode_id=episode_id, episode_url=episode_url)
            if exact_identity:
                score += 1000
            enclosure = item.find("enclosure")
            candidate_url = (enclosure.attrib.get("url") or "").strip() if enclosure is not None else ""
            candidates.append((score, urljoin(rss_url, candidate_url) if candidate_url else "", rss_url))
    candidates.sort(key=lambda row: row[0], reverse=True)
    if not candidates or not candidates[0][1]:
        return "", ""
    best_score, best_url, best_rss = candidates[0]
    # 跨feed也核对歧义；同一媒体在多个feed重复声明不算不同单集。
    if any(url != best_url and best_score - score < 5 for score, url, _ in candidates[1:]):
        return "", ""
    return best_url, best_rss


def _fetch_text(url: str) -> tuple[str, str]:
    last_error = ""
    for attempt in range(2):
        try:
            request = Request(url, headers=DEFAULT_HTTP_HEADERS)
            with urlopen(request, timeout=20) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace"), ""
        except Exception as exc:
            last_error = str(exc)
            if attempt == 0 and any(token in last_error for token in ["503", "Service Unavailable", "429", "Forbidden"]):
                continue
            break
    return "", last_error


def _discover_rss_candidates(base_url: str, html: str) -> list[str]:
    candidates = [
        *re.findall(r'<link[^>]+type=["\']application/rss\+xml["\'][^>]+href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE),
        *re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(?:RSS|Feed)</a>', html, flags=re.IGNORECASE),
        *re.findall(r'https?://[^"\']+(?:/feed|/rss|\.xml)', html, flags=re.IGNORECASE),
    ]
    normalized: list[str] = []
    for item in candidates:
        candidate = html_unescape(item.strip())
        if not candidate:
            continue
        normalized.append(urljoin(base_url, candidate))
    return dedupe_strings(normalized)


def _extract_media_urls_from_html(html: str, base_url: str) -> list[str]:
    candidates = re.findall(r'https?://[^"\']+\.(?:mp3|m4a|mp4|aac|wav|flac)(?:\?[^"\']*)?', html, flags=re.IGNORECASE)
    candidates.extend(
        urljoin(base_url, item)
        for item in re.findall(r'["\']([^"\']+\.(?:mp3|m4a|mp4|aac|wav|flac)(?:\?[^"\']*)?)["\']', html, flags=re.IGNORECASE)
    )
    return dedupe_strings([html_unescape(item.strip()) for item in candidates if item.strip()])


def _pick_media_url(media_urls: list[str]) -> str:
    for media_url in media_urls:
        lowered = media_url.lower()
        if any(token in lowered for token in [".mp3", ".m4a", ".mp4", ".aac", ".wav", ".flac", "/stream/"]):
            return media_url
    return media_urls[0] if media_urls else ""


def _extract_json_ld_objects(html: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for raw in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.IGNORECASE | re.DOTALL):
        try:
            payload = json.loads(html_unescape(raw.strip()))
        except json.JSONDecodeError:
            continue
        objects.extend(_flatten_json_ld_nodes(payload))
    return objects


def _flatten_json_ld_nodes(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        rows = [node]
        graph = node.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                rows.extend(_flatten_json_ld_nodes(item))
        return rows
    if isinstance(node, list):
        rows: list[dict[str, Any]] = []
        for item in node:
            rows.extend(_flatten_json_ld_nodes(item))
        return rows
    return []


def _json_ld_value(objects: list[dict[str, Any]], key: str, *, expected_types: set[str] | None = None) -> str:
    for item in objects:
        if expected_types and not _json_ld_has_type(item, expected_types):
            continue
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _json_ld_author(objects: list[dict[str, Any]]) -> str:
    for item in objects:
        author = item.get("author") or item.get("publisher")
        if isinstance(author, dict):
            value = author.get("name")
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(author, list):
            for row in author:
                if isinstance(row, dict) and isinstance(row.get("name"), str) and row.get("name", "").strip():
                    return row["name"].strip()
    return ""


def _json_ld_duration_seconds(objects: list[dict[str, Any]]) -> int:
    for item in objects:
        value = item.get("duration")
        if isinstance(value, str) and value.strip():
            parsed = _parse_duration_seconds(value)
            if parsed:
                return parsed
    return 0


def _json_ld_show_website(objects: list[dict[str, Any]]) -> str:
    for item in objects:
        for key in ("url", "sameAs"):
            value = item.get(key)
            if isinstance(value, str) and value.startswith("http") and "podcasts.apple.com" not in value:
                return value
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, str) and row.startswith("http") and "podcasts.apple.com" not in row:
                        return row
        part = item.get("partOfSeries") or item.get("isPartOf")
        if isinstance(part, dict):
            value = part.get("url")
            if isinstance(value, str) and value.startswith("http") and "podcasts.apple.com" not in value:
                return value
    return ""


def _json_ld_external_episode_url(objects: list[dict[str, Any]]) -> str:
    for item in objects:
        value = item.get("url")
        if isinstance(value, str) and value.startswith("http") and "podcasts.apple.com" not in value:
            return value
    return ""


def _json_ld_media_urls(objects: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for item in objects:
        for key in ("contentUrl", "embedUrl"):
            value = item.get(key)
            if isinstance(value, str) and value.startswith("http"):
                urls.append(value)
        associated = item.get("associatedMedia")
        if isinstance(associated, dict):
            value = associated.get("contentUrl")
            if isinstance(value, str) and value.startswith("http"):
                urls.append(value)
    return dedupe_strings(urls)


def _json_ld_has_type(item: dict[str, Any], expected_types: set[str]) -> bool:
    value = item.get("@type")
    if isinstance(value, str):
        return value in expected_types
    if isinstance(value, list):
        return any(isinstance(row, str) and row in expected_types for row in value)
    return False


def _score_rss_item(
    item: ElementTree.Element,
    *,
    episode_title: str,
    duration_seconds: int,
    episode_id: str,
    episode_url: str,
) -> float:
    title = (item.findtext("title") or "").strip()
    score = 0.0
    if episode_id and _rss_episode_identity_matches(item, episode_url=episode_url, episode_id=episode_id):
        score += 200
    normalized_title = _normalized_match_text(title)
    normalized_episode_title = _normalized_match_text(episode_title)
    if normalized_title and normalized_episode_title:
        if normalized_title == normalized_episode_title:
            score += 120
        score += SequenceMatcher(None, normalized_title, normalized_episode_title).ratio() * 80
    item_duration = _parse_duration_seconds(
        item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration") or ""
    )
    if duration_seconds and item_duration:
        diff = abs(duration_seconds - item_duration)
        if diff <= 30:
            score += 40
        elif diff <= 120:
            score += 20
    if episode_url and _rss_episode_identity_matches(item, episode_url=episode_url, episode_id=episode_id):
        score += 40
    return score


def _normalized_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def _apple_episode_id(source: str) -> str:
    parsed = urlparse(source)
    query = parse_qs(parsed.query)
    return str(query.get("i", [""])[0]).strip()


def _first_nonempty(*values: str) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_int(*values: int) -> int:
    for value in values:
        if isinstance(value, int) and value > 0:
            return value
    return 0


def dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _dump_ytdlp_metadata(source: str) -> tuple[dict[str, Any] | None, str]:
    completed, error = _run_ytdlp(
        source,
        platform=detect_platform(source),
        command_kind="metadata",
    )
    if completed is None:
        return None, error or "yt-dlp failed to inspect this page."
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None, "yt-dlp returned unreadable JSON metadata."
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list) and payload["entries"]:
        first_entry = payload["entries"][0]
        if isinstance(first_entry, dict):
            return first_entry, ""
    return payload if isinstance(payload, dict) else None, ""


def build_ytdlp_command(
    source: str,
    *,
    platform: str,
    command_kind: str,
    browser: str | None = None,
    output_template: str | None = None,
) -> list[str]:
    yt_dlp_bin = find_binary("yt-dlp")
    if not yt_dlp_bin:
        raise FileNotFoundError("yt-dlp")
    command = [yt_dlp_bin, "--no-update", "--no-playlist"]
    cookies_file = os.environ.get("TRANSCRIBE_YTDLP_COOKIES_FILE", "").strip()
    browser_profile = os.environ.get("TRANSCRIBE_YTDLP_BROWSER_PROFILE", "").strip()
    user_agent = os.environ.get("TRANSCRIBE_YTDLP_USER_AGENT", "").strip()
    if platform == "youtube":
        remote_components = [item.strip() for item in os.environ.get("TRANSCRIBE_YTDLP_REMOTE_COMPONENTS", "ejs:github").split(",") if item.strip()]
        for component in remote_components:
            command.extend(["--remote-components", component])
        deno_bin = find_binary("deno")
        node_bin = find_binary("node")
        if deno_bin:
            command.extend(["--js-runtimes", f"deno:{deno_bin}"])
        if node_bin:
            command.extend(["--js-runtimes", f"node:{node_bin}"])
    if user_agent:
        command.extend(["--user-agent", user_agent])
    if cookies_file:
        command.extend(["--cookies", cookies_file])
    elif browser:
        browser_value = browser if not browser_profile else f"{browser}:{browser_profile}"
        command.extend(["--cookies-from-browser", browser_value])
    if command_kind == "metadata":
        command.append("--dump-single-json")
    if output_template:
        command.extend(["-o", output_template])
    command.append(source)
    return command


def _cookies_browser_candidates(platform: str) -> list[str]:
    configured = os.environ.get("TRANSCRIBE_YTDLP_BROWSER", "").strip()
    system = os_platform.system().lower()
    defaults: list[str]
    if platform == "youtube":
        if system == "darwin":
            defaults = ["firefox", "chrome", "brave", "edge"]
        elif system == "windows":
            defaults = ["firefox", "chrome", "brave", "edge"]
        else:
            defaults = ["firefox", "chrome", "chromium", "brave", "edge"]
    elif platform == "bilibili":
        if system == "darwin":
            defaults = ["chrome", "edge", "firefox", "brave"]
        elif system == "windows":
            defaults = ["edge", "chrome", "firefox", "brave"]
        else:
            defaults = ["firefox", "chrome", "chromium", "edge", "brave"]
    else:
        defaults = []

    ordered = [configured] + defaults if configured else defaults
    seen: set[str] = set()
    unique: list[str] = []
    for item in ordered:
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    if not unique:
        return unique
    configured = os.environ.get("TRANSCRIBE_YTDLP_BROWSER", "").strip()
    filtered: list[str] = []
    for item in unique:
        if item == configured or _browser_profile_exists(item):
            filtered.append(item)
    return filtered


def _browser_profile_exists(browser: str) -> bool:
    system = os_platform.system().lower()
    home = Path.home()
    browser = browser.lower()
    if system == "darwin":
        candidates = {
            "chrome": [home / "Library/Application Support/Google/Chrome/Default/Cookies"],
            "brave": [home / "Library/Application Support/BraveSoftware/Brave-Browser/Default/Cookies"],
            "edge": [home / "Library/Application Support/Microsoft Edge/Default/Cookies"],
            "firefox": [home / "Library/Application Support/Firefox/Profiles"],
            "safari": [home / "Library/Containers/com.apple.Safari/Data/Library/Cookies/Cookies.binarycookies"],
        }
        return any(path.exists() for path in candidates.get(browser, []))
    if system == "windows":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        roaming = Path(os.environ.get("APPDATA", ""))
        candidates = {
            "chrome": [local / "Google/Chrome/User Data/Default/Network/Cookies", local / "Google/Chrome/User Data/Default/Cookies"],
            "brave": [local / "BraveSoftware/Brave-Browser/User Data/Default/Network/Cookies", local / "BraveSoftware/Brave-Browser/User Data/Default/Cookies"],
            "edge": [local / "Microsoft/Edge/User Data/Default/Network/Cookies", local / "Microsoft/Edge/User Data/Default/Cookies"],
            "firefox": [roaming / "Mozilla/Firefox/Profiles"],
        }
        return any(path.exists() for path in candidates.get(browser, []))
    candidates = {
        "chrome": [home / ".config/google-chrome/Default/Cookies", home / ".config/chrome/Default/Cookies"],
        "chromium": [home / ".config/chromium/Default/Cookies"],
        "brave": [home / ".config/BraveSoftware/Brave-Browser/Default/Cookies"],
        "edge": [home / ".config/microsoft-edge/Default/Cookies"],
        "firefox": [home / ".mozilla/firefox"],
    }
    return any(path.exists() for path in candidates.get(browser, []))


def _looks_like_cookie_permission_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return "operation not permitted" in lowered and "cookies.binarycookies" in lowered


def _looks_like_missing_browser_cookie_db_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return "could not find" in lowered and "cookies database" in lowered


def _looks_like_antibot_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        token in lowered
        for token in [
            "http error 412",
            "precondition failed",
            "http error 403",
            "cloudflare",
            "captcha",
            "sign in to confirm you’re not a bot",
            "sign in to confirm you're not a bot",
            "confirm you are not a bot",
            "confirm you're not a bot",
        ]
    )


def _looks_like_reload_cookie_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return "page needs to be reloaded" in lowered


def _format_ytdlp_error(stderr: str, *, platform: str, tried_browser_cookies: bool) -> str:
    cleaned = stderr.strip()
    if platform == "youtube" and _looks_like_reload_cookie_error(cleaned):
        guidance = (
            "YouTube accepted the browser session but considers the current cookies stale for this request. "
            "Open the same video in Chrome, reload it there, confirm playback works, and retry immediately."
        )
        guidance += " If this keeps happening, export Chrome cookies to a Netscape cookies.txt file and set TRANSCRIBE_YTDLP_COOKIES_FILE to that path before launching NextEcho."
        return f"{guidance}\nOriginal yt-dlp error:\n{cleaned}" if cleaned else guidance
    if _looks_like_missing_browser_cookie_db_error(cleaned):
        guidance = (
            "NextEcho tried to read cookies from a browser profile that does not exist on this machine. "
            "Please switch to a browser you actually use, such as Firefox or Chrome."
        )
        guidance += " You can set TRANSCRIBE_YTDLP_BROWSER=firefox or chrome before launching NextEcho."
        return f"{guidance}\nOriginal yt-dlp error:\n{cleaned}" if cleaned else guidance
    if _looks_like_cookie_permission_error(cleaned):
        guidance = (
            "NextEcho could not read browser cookies because macOS blocked access to the browser cookie store. "
            "On macOS, prefer Firefox or Chrome cookies instead of Safari, or grant the launching app Full Disk Access and retry."
        )
        guidance += " You can set TRANSCRIBE_YTDLP_BROWSER=firefox or chrome before launching NextEcho."
        return f"{guidance}\nOriginal yt-dlp error:\n{cleaned}" if cleaned else guidance
    if platform == "youtube" and _looks_like_antibot_error(cleaned):
        guidance = (
            "YouTube asked for a bot check or authenticated browser session. "
            "Please open the same video in your browser, confirm it plays normally, and retry."
        )
        if tried_browser_cookies:
            guidance += " NextEcho already retried with browser cookies."
        else:
            guidance += " You can also set TRANSCRIBE_YTDLP_BROWSER=firefox or chrome before retrying."
        guidance += " Update yt-dlp to the latest release or nightly build. Some YouTube requests may also need a fresh browser cookie session."
        return f"{guidance}\nOriginal yt-dlp error:\n{cleaned}" if cleaned else guidance
    if platform == "bilibili" and _looks_like_antibot_error(cleaned):
        guidance = (
            "Bilibili blocked the request with anti-bot protection. "
            "Please open the same video in your browser, confirm it plays normally, and retry."
        )
        if tried_browser_cookies:
            guidance += " NextEcho already retried with browser cookies."
        else:
            guidance += " You can also set TRANSCRIBE_YTDLP_BROWSER=chrome (or safari/firefox/edge) before retrying."
        guidance += " Also update yt-dlp to the latest release or nightly build."
        return f"{guidance}\nOriginal yt-dlp error:\n{cleaned}" if cleaned else guidance
    return cleaned


def _run_ytdlp(source: str, *, platform: str, command_kind: str, output_template: str | None = None) -> tuple[subprocess.CompletedProcess[str] | None, str]:
    try:
        direct_command = build_ytdlp_command(
            source,
            platform=platform,
            command_kind=command_kind,
            output_template=output_template,
        )
    except FileNotFoundError:
        return None, "yt-dlp is not installed."

    attempts: list[tuple[str | None, list[str]]] = [(None, direct_command)]
    browser_candidates = _cookies_browser_candidates(platform)
    for browser in browser_candidates:
        attempts.append(
            (
                browser,
                build_ytdlp_command(
                    source,
                    platform=platform,
                    command_kind=command_kind,
                    browser=browser,
                    output_template=output_template,
                ),
            )
        )

    last_stderr = ""
    tried_browser_cookies = False
    attempt_notes: list[str] = []
    for attempt_index, (browser, command) in enumerate(attempts):
        try:
            return (
                run_media_command(
                    command, timeout=120, stage="source-probe",
                    capture_output=True,
                    text=True,
                    check=True,
                ),
                "",
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            last_stderr = stderr or "yt-dlp failed to inspect this page."
            if browser:
                attempt_notes.append(f"{browser}: {last_stderr.splitlines()[-1] if last_stderr else 'failed'}")
            else:
                attempt_notes.append(f"direct: {last_stderr.splitlines()[-1] if last_stderr else 'failed'}")
            if browser and _looks_like_missing_browser_cookie_db_error(last_stderr):
                tried_browser_cookies = True
                continue
            if browser:
                tried_browser_cookies = True
                continue
            should_retry_with_cookies = platform in {"bilibili", "youtube"} and _looks_like_antibot_error(last_stderr) and len(attempts) > 1
            if not should_retry_with_cookies or attempt_index == len(attempts) - 1:
                break
    formatted = _format_ytdlp_error(last_stderr, platform=platform, tried_browser_cookies=tried_browser_cookies)
    if attempt_notes:
        formatted = f"{formatted}\nTried:\n- " + "\n- ".join(attempt_notes)
    return None, formatted


def _coerce_published_at(raw: str) -> str:
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def _parse_duration_seconds(raw: str) -> int:
    value = raw.strip()
    if not value:
        return 0
    iso_match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value, flags=re.IGNORECASE)
    if iso_match:
        hours = int(iso_match.group(1) or 0)
        minutes = int(iso_match.group(2) or 0)
        seconds = int(iso_match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
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
    value = value or "item"
    if len(value.encode("utf-8")) > 80:
        value = value.encode("utf-8")[:64].decode("utf-8", errors="ignore").rstrip("-") + "-" + value_hash(value)[:12]
    return value


def html_unescape(value: str) -> str:
    return unescape(value).replace("\\u002F", "/").replace("\\/", "/")
