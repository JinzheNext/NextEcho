"""Local, provenance-preserving context collection for podcast episode pages.

The Xiaoyuzhou episode page already contains a server-rendered ``__NEXT_DATA__``
payload with the complete shownotes and the first page of comments.  We parse
that payload locally so a normal run does not require a cloud crawler or an API
key.  When a Firecrawl Python client and ``FIRECRAWL_API_KEY`` are available,
``NEXTECHO_CRAWLER=firecrawl`` can be used as an explicit alternative.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


CRAWL_SCHEMA_VERSION = 1
DEFAULT_COMMENT_LIMIT = 20
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def crawl_episode_context(
    url: str,
    output_dir: Path,
    *,
    comment_limit: int = DEFAULT_COMMENT_LIMIT,
    crawler: str | None = None,
) -> dict[str, Any]:
    """Fetch and persist shownotes/comments without making ASR depend on them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_html, provider, provider_error = _fetch_page(url, crawler or os.environ.get("NEXTECHO_CRAWLER", "auto"))
    if not raw_html:
        payload = {
            "schema_version": CRAWL_SCHEMA_VERSION,
            "status": "failed",
            "url": url,
            "provider": provider,
            "error": provider_error or "empty page response",
            "speaker_candidates": [],
            "comments": [],
        }
        context_path = output_dir / "source.context.json"
        context_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return _path_payload(payload, context_path=context_path)

    raw_path = output_dir / "source.raw.html"
    raw_path.write_text(raw_html, encoding="utf-8")
    next_data = _extract_next_data(raw_html)
    page_props = ((next_data or {}).get("props") or {}).get("pageProps") or {}
    episode = page_props.get("episode") if isinstance(page_props.get("episode"), dict) else {}
    shownotes_html = str(episode.get("shownotes") or "")
    shownotes_text = html_to_text(shownotes_html)
    comments = _top_comments(page_props.get("comments"), limit=comment_limit)
    speaker_candidates = infer_speaker_candidates(episode, shownotes_text)
    title = str(episode.get("title") or page_props.get("title") or "")
    payload = {
        "schema_version": CRAWL_SCHEMA_VERSION,
        "status": "ready" if next_data else "partial",
        "url": url,
        "provider": provider,
        "fetched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "episode": {
            "id": str(episode.get("eid") or page_props.get("id") or ""),
            "title": title,
            "podcast": str(((episode.get("podcast") or {}).get("title") or "")),
            "host": str(((episode.get("podcast") or {}).get("author") or "")),
            "published_at": str(episode.get("pubDate") or ""),
            "duration_seconds": int(episode.get("duration") or 0),
            "play_count": int(episode.get("playCount") or 0),
            "comment_count": int(episode.get("commentCount") or 0),
        },
        "speaker_candidates": speaker_candidates,
        "shownotes": {
            "html": shownotes_html,
            "text": shownotes_text,
            "character_count": len(shownotes_text),
        },
        "comments": comments,
        "comment_selection": {
            "requested": comment_limit,
            "returned": len(comments),
            "order": "likeCount descending",
            "scope": "top-level comments; reply counts retained",
        },
    }
    shownotes_path = output_dir / "source.shownotes.md"
    shownotes_path.write_text(
        f"# {title or '节目 Show Notes'}\n\n来源：{url}\n\n{shownotes_text.strip()}\n",
        encoding="utf-8",
    )
    comments_path = output_dir / "source.comments.top20.json"
    comments_path.write_text(json.dumps(comments, ensure_ascii=False, indent=2), encoding="utf-8")
    context_path = output_dir / "source.context.json"
    context_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["paths"] = {
        "raw_html": str(raw_path),
        "shownotes": str(shownotes_path),
        "comments": str(comments_path),
        "context": str(context_path),
    }
    context_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def infer_speaker_candidates(episode: dict[str, Any], shownotes_text: str) -> list[dict[str, str]]:
    """Extract host/guest names as hints; never treat them as verified diarization."""
    candidates: list[dict[str, str]] = []
    host = str(((episode.get("podcast") or {}).get("author") or "")).strip()
    if host:
        candidates.append({"name": host, "role": "host", "source": "podcast.author"})
    lines = [line.strip(" \t:：") for line in shownotes_text.splitlines() if line.strip()]
    active = False
    for line in lines:
        if re.search(r"(?:本期嘉宾|嘉宾|guest|guests)", line, flags=re.IGNORECASE):
            active = True
            continue
        if active and (re.search(r"(?:时间戳|shownotes|show notes|官网|链接)", line, flags=re.IGNORECASE) or line.startswith("#")):
            break
        if active:
            for name in re.findall(r"[\u4e00-\u9fff]{2,8}", line):
                if not any(item["name"] == name for item in candidates):
                    candidates.append({"name": name, "role": "guest", "source": "shownotes"})
    return candidates


def html_to_text(value: str) -> str:
    # Keep the target of show-note links in the local text artifact; the raw
    # HTML is also retained for exact provenance.
    value = re.sub(
        r'<a\b[^>]*?href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        lambda match: f"{match.group(2)} ({match.group(1)})",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    parser = _TextParser()
    parser.feed(unescape(value))
    parser.close()
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", "".join(parser.parts))).strip()


def _extract_next_data(html: str) -> dict[str, Any] | None:
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _top_comments(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        author = item.get("author") if isinstance(item.get("author"), dict) else {}
        rows.append(
            {
                "id": str(item.get("id") or ""),
                "text": str(item.get("text") or "").strip(),
                "author": str(author.get("nickname") or ""),
                "like_count": int(item.get("likeCount") or 0),
                "created_at": str(item.get("createdAt") or ""),
                "pinned": bool(item.get("pinned")),
                "reply_count": int(item.get("replyCount") or item.get("threadReplyCount") or 0),
                "ip_location": str(item.get("ipLoc") or ""),
            }
        )
    rows.sort(key=lambda row: (row["like_count"], row["pinned"]), reverse=True)
    for index, row in enumerate(rows[: max(0, limit)], start=1):
        row["rank"] = index
    return rows[: max(0, limit)]


def _fetch_page(url: str, crawler: str) -> tuple[str, str, str]:
    if crawler == "firecrawl" or (crawler == "auto" and os.environ.get("FIRECRAWL_API_KEY")):
        try:
            html = _fetch_with_firecrawl(url)
            if html:
                return html, "firecrawl", ""
        except Exception as exc:
            if crawler == "firecrawl":
                return "", "firecrawl", str(exc)
            firecrawl_error = str(exc)
        else:
            firecrawl_error = "Firecrawl returned an empty page"
    else:
        firecrawl_error = ""
    try:
        request = Request(url, headers=DEFAULT_HEADERS)
        with urlopen(request, timeout=30) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace"), "xiaoyuzhou-next-data", firecrawl_error
    except Exception as exc:
        return "", "xiaoyuzhou-next-data", str(exc) or firecrawl_error


def _fetch_with_firecrawl(url: str) -> str:
    token = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if not token:
        raise RuntimeError("FIRECRAWL_API_KEY is not set")
    try:
        from firecrawl import FirecrawlApp  # type: ignore

        client = FirecrawlApp(api_key=token)
    except ImportError:
        from firecrawl import Firecrawl  # type: ignore

        client = Firecrawl(api_key=token)
    if hasattr(client, "scrape_url"):
        result = client.scrape_url(url, params={"formats": ["html"]})
    else:
        result = client.scrape(url, formats=["html"])
    if isinstance(result, dict):
        return str(result.get("html") or result.get("data", {}).get("html") or "")
    return str(getattr(result, "html", "") or "")


def _path_payload(payload: dict[str, Any], *, context_path: Path) -> dict[str, Any]:
    payload["paths"] = {"context": str(context_path)}
    return payload


class _TextParser(HTMLParser):
    BLOCK_TAGS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "br", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
