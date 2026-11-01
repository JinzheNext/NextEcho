"""Source-derived reading names and metadata; no model or network access."""
from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

from .contracts import PipelineError
from .io_utils import file_hash, write_json

NAMING_VERSION = "source-reading-v1"


def freeze_delivery_context(run_dir: Path, *, created_at: float | None = None,
                            parent_dir: Path | None = None) -> dict:
    """Freeze the original transcription day, including across resumed jobs."""
    path = run_dir / "delivery.context.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        datetime.strptime(data["transcribed_on"], "%Y-%m-%d")
        return data
    source = parent_dir or run_dir
    inherited = source / "delivery.context.json"
    if parent_dir and inherited.is_file():
        data = json.loads(inherited.read_text(encoding="utf-8"))
        datetime.strptime(data["transcribed_on"], "%Y-%m-%d")
    else:
        request = source / "request.json"
        if request.is_file():
            saved = json.loads(request.read_text(encoding="utf-8"))
            created_at = saved.get("created_at", created_at)
        stamp = datetime.fromtimestamp(created_at).astimezone() if created_at is not None else datetime.now().astimezone()
        data = {"schema_version": 1, "naming_version": NAMING_VERSION,
                "transcribed_on": stamp.date().isoformat(), "timezone": str(stamp.tzinfo),
                "created_at": stamp.isoformat()}
    write_json(path, data)
    return data


def source_info(result: dict) -> dict:
    context = result.get("source_context") or {}
    episode = context.get("episode") or {}
    podcast = episode.get("podcast") or result.get("podcast_title") or result.get("show_title") or ""
    if isinstance(podcast, dict):
        podcast = podcast.get("title") or ""
    return {"show": str(podcast), "title": str(episode.get("title") or result.get("title") or result.get("source_label") or "媒体转录"),
            "url": str(context.get("url") or result.get("canonical_url") or ""),
            "published_at": str(episode.get("published_at") or result.get("published_at") or ""),
            "id": str(episode.get("id") or result.get("canonical_url") or result.get("source") or ""),
            "context": context}


def _component(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "-", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", text, flags=re.I):
        text = "_" + text
    return text or "未命名"


def reading_filename(results: list[dict], transcribed_on: str) -> str:
    datetime.strptime(transcribed_on, "%Y-%m-%d")
    infos = [source_info(row) for row in results]
    identity = json.dumps([{k: v for k, v in info.items() if k != "context"} for info in infos], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
    if len(infos) == 1:
        show, title = infos[0]["show"] or "未分类栏目", infos[0]["title"]
    else:
        shows = {i["show"] for i in infos if i["show"]}
        show = next(iter(shows)) if len(shows) == 1 else "多栏目"
        title = f"播客合集-{len(infos)}-{digest}"
    raw = show + "_" + title
    base = _component(show) + "_" + _component(title)
    suffix = "_" + transcribed_on + ".md"
    if base != raw or len((base + suffix).encode()) > 220:
        suffix = "_" + digest + suffix
        budget = 220 - len(suffix.encode())
        base = base.encode()[:budget].decode("utf-8", errors="ignore").rstrip(" .")
    return base + suffix


def primary_path(run_dir: Path) -> Path:
    index = run_dir / "delivery.index.json"
    if not index.is_file():
        return run_dir / "result.md"  # Historical runs remain readable.
    try:
        record = json.loads(index.read_text(encoding="utf-8"))
        name = record["primary_name"]
        if not isinstance(name, str) or "/" in name or "\\" in name or not name.endswith(".md"):
            raise ValueError("invalid name")
        path = run_dir / name
        if path.resolve().parent != run_dir.resolve() or not path.is_file():
            raise ValueError("missing or external primary")
        if file_hash(path) != record["primary_sha256"]:
            raise ValueError("primary hash mismatch")
        return path
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PipelineError("PROTOCOL_ERROR", "主文件索引无效或内容已改变，请检查delivery.index.json") from exc


def markdown_text(value: object) -> str:
    return re.sub(r"([\\`*_{}\[\]()#!|])", r"\\\1", html.escape(str(value), quote=False))


def metadata_header(result: dict, context: dict, *, item_index: int = 1) -> list[str]:
    info = source_info(result)
    rows = [f"栏目：{markdown_text(info['show'] or '未知')}  ",
            f"单集：{markdown_text(info['title'])}  ",
            f"转录日期：{context['transcribed_on']}（{markdown_text(context['timezone'])}）  "]
    url = info["url"]
    try:
        parts = urlsplit(url)
        valid_source = parts.scheme in {"http", "https"} and parts.hostname and not parts.username
    except ValueError:
        valid_source = False
    if valid_source:
        rows.append(f"来源：[原节目页面]({quote(url, safe=':/?&=%#')})  ")
    if info["published_at"]:
        rows.append(f"发布时间：{markdown_text(info['published_at'])}  ")
    source = info["context"]
    names = [str(c.get("name")) for c in source.get("speaker_candidates", []) if isinstance(c, dict) and c.get("name")]
    confirmed = [p.get("display_name") for p in (result.get("speaker_identity") or {}).get("speakers", [])
                 if p.get("identity_status") == "resolved" and p.get("display_name") not in {None, "", "unknown"}]
    if confirmed:
        rows.append("对话人物：" + markdown_text("、".join(dict.fromkeys(confirmed))) + "  ")
    elif names:
        rows.append("官方人物候选（不代表声音已确认）：" + markdown_text("、".join(dict.fromkeys(names))) + "  ")
    notes = source.get("shownotes") or {}
    intro = str(notes.get("text") or "").strip()
    if intro:
        rows += ["", "### 节目资料摘录", "", "以下来自官方 shownotes，与音频实际发言分别呈现。", ""]
        excerpt = intro[:900]
        rows.extend("> " + markdown_text(line) for line in excerpt.splitlines())
        if len(intro) > len(excerpt):
            rows += ["> ……（摘录；完整资料见来源文件）"]
    from .reading_view import chapters_from_context, chapter_heading
    sections = chapters_from_context(source)
    if sections:
        rows += ["", f"### 节目目录（完整 {len(sections)} 章）", ""]
        for index, section in enumerate(sections):
            stamp = section["start_ms"] // 1000
            title = f"{stamp // 60:02d}:{stamp % 60:02d} {markdown_text(section['title'])}"
            if result.get("identity_json_path") or (result.get("enhanced_json_path") and not result.get("identity_text_path")):
                target = quote(chapter_heading(section, index, item_index), safe='')
                rows.append(f"- [{title}](#{target})")
            else:
                rows.append("- " + title)
    selection = source.get("comment_selection") or {}
    if selection:
        comments = source.get("comments") or []
        if comments:
            heading = comment_heading(item_index)
            rows += ["", f"听众评论：[{len(comments)} 条主评论、{count_replies(comments)} 条回复预览](#{quote(heading, safe='')})。"]
        else:
            rows += ["", "评论资料完整性未确认；本次未捕获可展示的评论正文。"]
    rows += ["", "### 转录正文", ""]
    return rows


def comment_heading(item_index=1):
    return f"{item_index}. 听众评论"


def count_replies(comments):
    return sum(len(c.get("replies") or []) + count_replies(c.get("replies") or []) for c in comments)


def render_comments(source, *, item_index=1):
    comments = source.get("comments") or []
    if not comments:
        return ""
    selection = source.get("comment_selection") or {}
    rows = [f"### {comment_heading(item_index)}", "",
            f"本次捕获 {len(comments)} 条公开主评论、{count_replies(comments)} 条回复预览。"
            + ("仅为公开页面快照，完整性未确认。" if selection.get("complete") is not True else ""),
            "听众观点与节目发言分别呈现；以下保留评论原文。", ""]
    def append(comment, depth=0):
        name = markdown_text(comment.get("author") or "匿名听众")
        likes = markdown_text(comment.get("like_count", 0))
        rows.extend([f"{'回复 · ' if depth else ''}**{name}** · {likes} 赞", ""])
        rows.extend("> " + markdown_text(line) for line in str(comment.get("text") or "").splitlines())
        rows.append("")
        for reply in comment.get("replies") or []:
            append(reply, depth + 1)
    for index, comment in enumerate(comments, 1):
        rows += [f"#### 评论 {index}", ""]
        append(comment)
    return "\n".join(rows)
