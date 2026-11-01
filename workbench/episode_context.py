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
from .io_utils import atomic_write, file_hash


CRAWL_SCHEMA_VERSION = 5
MAX_CONTEXT_PAGE_BYTES = 8 * 1024**2
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
    selected = str(crawler if crawler is not None else os.environ.get("NEXTECHO_CRAWLER", "native")).strip().lower()
    raw_html, provider, provider_error = _fetch_page(url, selected)
    fetch_policy = {"requested": selected, "effective": "native" if selected == "auto" else selected,
                    "cloud_selected_explicitly": selected == "firecrawl", "implicit_fallback": False}
    if not raw_html:
        payload = {
            "schema_version": CRAWL_SCHEMA_VERSION,
            "status": "failed",
            "url": url,
            "provider": provider,
            "fetch_policy": fetch_policy,
            "error": provider_error or "empty page response",
            "speaker_candidates": [],
            "comments": [],
            "context_policy": {
                "comments_are_evidence": False,
                "shownotes_require_timestamp_for_partial": True,
            },
        }
        context_path = output_dir / "source.context.json"
        context_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return _path_payload(payload, context_path=context_path)

    raw_path = output_dir / "source.raw.html"
    atomic_write(raw_path, raw_html)
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
        "fetch_policy": fetch_policy,
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
        "speaker_candidate_policy": {
            "parser": "explicit-credits-v2", "limit": 32,
            "exhaustive": False, "acoustic_identity_evidence": False,
            "publisher_author_is_not_automatically_a_host": True,
        },
        "shownotes": {
            "html": shownotes_html,
            "text": shownotes_text,
            "character_count": len(shownotes_text),
            "sections": split_shownotes_sections(shownotes_text),
        },
        "comments": comments,
        "context_policy": {
            "comments_are_evidence": False,
            "shownotes_require_timestamp_for_partial": True,
            "reference_format": "shownotes:<index> / comment:<id-or-rank>",
        },
        "comment_selection": {
            "requested": comment_limit,
            "returned": len(comments),
            "order": "likeCount descending within captured public page",
            "scope": "public first-page top-level comments and embedded reply previews",
            "available_top_level": len(page_props.get("comments") or []),
            "returned_replies": sum(len(c.get("replies", [])) for c in comments),
            "reported_total": int(episode.get("commentCount") or 0),
            "complete": False,
            "pagination": "public embedded page only; no pagination cursor returned; remaining comments not captured",
        },
    }
    shownotes_path = output_dir / "source.shownotes.md"
    shownotes_path.write_text(
        f"# {title or '节目 Show Notes'}\n\n来源：{url}\n\n{shownotes_text.strip()}\n",
        encoding="utf-8",
    )
    comments_path = output_dir / "source.comments.json"
    comments_path.write_text(json.dumps(comments, ensure_ascii=False, indent=2), encoding="utf-8")
    if comment_limit == DEFAULT_COMMENT_LIMIT:
        (output_dir / "source.comments.top20.json").write_text(comments_path.read_text(), encoding="utf-8")
    context_path = output_dir / "source.context.json"
    context_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["paths"] = {
        "raw_html": str(raw_path),
        "shownotes": str(shownotes_path),
        "comments": str(comments_path),
        "context": str(context_path),
    }
    payload["source_snapshot"] = {"parser_schema": CRAWL_SCHEMA_VERSION,
        "files": {p.name: file_hash(p) for p in (raw_path, shownotes_path, comments_path)},
        "hash_scope": "saved UTF-8 decoded HTML and derived files, not original wire bytes",
        "freshness": "captured-at-fetch; remote freshness not subsequently checked",
        "reuse_policy": "run-scoped snapshot; no TTL-driven refresh; resume input ledger governs reuse"}
    atomic_write(context_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def infer_speaker_candidates(episode: dict[str, Any], shownotes_text: str) -> list[dict[str, str]]:
    """Read explicit credits, not every word following a guest heading.

    A publisher account can be a company or programme, so podcast.author is
    not a personal host credit. These bounded names remain unverified hints.
    """
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_names(value: str, role: str, source: str) -> bool:
        value = re.split(r"[:：\(（]", value, maxsplit=1)[0].strip()
        pieces = re.split(r"[、,，/|｜]", value)
        if len(pieces) == 1 and re.fullmatch(r"[\u3400-\u9fff]{2,8}(?:\s+[\u3400-\u9fff]{2,8})+", value):
            pieces = value.split()
        added = False
        for piece in pieces:
            name = piece.strip(" \t《》「」“”\"'")
            if not re.fullmatch(r"[\u3400-\u9fff]{2,8}(?:·[\u3400-\u9fff]{1,8})*|[A-Za-z][A-Za-z .'-]{0,61}[A-Za-z]", name):
                continue
            if name in {"主持", "主持人", "主播", "嘉宾", "嘉宾团队", "财经作家", "相关链接"} or name.casefold() in {"http", "https", "www"}:
                continue
            added = True
            key = name.casefold()
            if key not in seen and len(candidates) < 32:
                seen.add(key)
                candidates.append({"name": name, "role": role, "source": source})
        return added

    lines = [line.strip(" \t:：") for line in shownotes_text.splitlines() if line.strip()]
    active = False
    for line in lines:
        heading = re.sub(r"^[^\u3400-\u9fffA-Za-z0-9]+", "", line).strip()
        for credit in re.split(r"[|｜]", heading):
            match = re.match(r"\s*(?:本期|本集)?(主持人?|主播|嘉宾|hosts?|guests?)(?:\s*[:：]\s*|\s+)(.+)$", credit, re.I)
            if match:
                role = "guest" if re.fullmatch(r"嘉宾|guests?", match[1], re.I) else "host"
                add_names(match[2], role, "shownotes:credits")
        if re.fullmatch(r"(?:本期|本集)?(?:嘉宾|guests?)\s*[:：]?", heading, re.I):
            active = True
            continue
        if active and (re.match(r"(?:时间[轴戳线]|本期内容|本期提及|延伸阅读|相关链接|官网|链接|show\s*notes|https?://|www\.|\d{1,3}:[0-5]\d)", heading, re.I) or line.startswith("#")):
            active = False
            continue
        if active:
            # A profile's name precedes its colon; biographies and chapter
            # paragraphs are never split into arbitrary 2–8-character names.
            if not add_names(heading, "guest", "shownotes:guest-section"):
                active = False
    title = str(episode.get("title") or "")
    explicit = re.search(r"(?:对话|访谈|专访)\s*([^:：|｜!?！？\n]+)", title)
    if explicit:
        add_names(explicit[1], "guest", "episode.title")
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


_TIMESTAMP_RE = re.compile(
    r"(?<![\d-])(?:\[\s*)?(?P<h>\d{1,3}):(?P<m>[0-5]\d)(?::(?P<s>[0-5]\d))?(?:[.,](?P<f>\d{1,3}))?(?:\s*\])?"
)


def parse_timestamp_ms(value: str) -> int | None:
    """Parse a show-note timestamp (``MM:SS`` or ``HH:MM:SS``) to ms."""

    match = _TIMESTAMP_RE.search(str(value))
    if not match:
        return None
    hours_or_minutes = int(match.group("h"))
    minute_or_seconds = int(match.group("m"))
    fraction = str(match.group("f") or "")
    if match.group("s") is None:
        # The common short form is MM:SS (including 00:00).  Long episodes
        # may exceed 59 minutes, so normalize that first field into hours.
        hours, minutes = divmod(hours_or_minutes, 60)
        seconds = minute_or_seconds
    else:
        hours = hours_or_minutes
        minutes = minute_or_seconds
        seconds = int(match.group("s"))
    millis = int((fraction + "00")[:3]) if fraction else 0
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def split_shownotes_sections(text: str) -> list[dict[str, Any]]:
    """Split shownotes into timestamp-addressable context sections.

    A section starts at a timestamp and runs until the next timestamp.  Text
    before the first timestamp is deliberately not emitted: it cannot be
    safely associated with a partial audio window.  The returned refs are
    deterministic and can be used in evidence records without copying source
    text into a transcript segment.
    """

    source = str(text or "")
    matches = list(_TIMESTAMP_RE.finditer(source))
    if not matches:
        return []
    sections: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start_ms = parse_timestamp_ms(match.group(0))
        if start_ms is None:
            continue
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        # Episode timelines and glossary lists often restart at 00:00 or have
        # several entries at the same instant. Never emit negative/zero spans.
        later = [parse_timestamp_ms(m.group(0)) for m in matches]
        end_ms = min((t for t in later if t is not None and t > start_ms), default=None)
        section_text = source[match.start() : next_start].strip()
        if not section_text:
            continue
        sections.append(
            {
                "context_ref": f"shownotes:{len(sections)}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": section_text,
            }
        )
    return sections


# Friendly alias for callers that describe these as timestamped show-note
# entries rather than sections.
extract_shownotes_sections = split_shownotes_sections


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
                "thread_reply_count": int(item.get("threadReplyCount") or 0),
                "replies": _top_comments(item.get("replies"), limit=20),
                "ip_location": str(item.get("ipLoc") or ""),
                "context_ref": "",
                "evidence_eligible": False,
                "context_only": True,
            }
        )
    rows.sort(key=lambda row: (row["like_count"], row["pinned"]), reverse=True)
    for index, row in enumerate(rows[: max(0, limit)], start=1):
        row["rank"] = index
        row["context_ref"] = f"comment:{row['id'] or index}"
    return rows[: max(0, limit)]


def _fetch_page(url: str, crawler: str) -> tuple[str, str, str]:
    crawler = str(crawler).strip().lower()
    if crawler not in {"native", "auto", "firecrawl"}:
        return "", "unselected", "unknown crawler; select native or firecrawl explicitly"
    if crawler == "firecrawl":
        try:
            html = _fetch_with_firecrawl(url)
            if html:
                if len(html.encode("utf-8")) <= MAX_CONTEXT_PAGE_BYTES:
                    return html, "firecrawl", ""
                return "", "firecrawl", "page exceeds context size limit"
        except Exception as exc:
            return "", "firecrawl", type(exc).__name__ + ": explicit cloud fetch failed"
        return "", "firecrawl", "Firecrawl returned an empty page"
    try:
        request = Request(url, headers=DEFAULT_HEADERS)
        with urlopen(request, timeout=30) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read(MAX_CONTEXT_PAGE_BYTES + 1)
            if len(raw) > MAX_CONTEXT_PAGE_BYTES:
                return "", "xiaoyuzhou-next-data", "page exceeds context size limit"
            return raw.decode(charset, errors="replace"), "xiaoyuzhou-next-data", ""
    except Exception as exc:
        return "", "xiaoyuzhou-next-data", type(exc).__name__ + ": native fetch failed"


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
