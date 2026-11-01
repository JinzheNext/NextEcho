"""Audited presentation cleanup. Acoustic identities and source text stay intact."""
from __future__ import annotations

import copy
import re

from .io_utils import value_hash
from .reading_artifacts import markdown_text

VERSION = "reading-boundaries-v2"
_ACK = re.compile(r"(?:(?:嗯|恩|呃|额|啊|哦|噢|唔|哎|诶|对|是的|好的|好吧|好|对啊|对呀|对的))+")
_HESITATION = re.compile(r"[嗯呃额]+")
# Only lexical continuations, never a general instruction to move short replies.
_SPLITS = {"然": ("后",), "因": ("为",), "所": ("以",), "而": ("且",),
           "互联": ("网",), "思": ("维",), "明": ("白",), "最": ("终",),
           "O": ("K",), "o": ("k",), "那": ("里",), "这": ("里",)}


def chapter_heading(chapter, index, item_index=1):
    stamp = chapter["start_ms"] // 1000
    # One canonical visible title is shared by the TOC and body. Obsidian
    # resolves heading text, rather than arbitrary HTML id attributes.
    title = re.sub(r"\s+", " ", str(chapter["title"])).strip()
    return f"{item_index}.{index + 1:02d} {stamp // 60:02d}:{stamp % 60:02d} {title}"


def uncertain(row):
    label = str(row.get("speaker") or "")
    return not label or label in {"unknown", "归属待核对"} or label.startswith("交叠发言")


def plain(text):
    return re.sub(r"[\W_]", "", text, flags=re.UNICODE)


def chapters_from_context(context):
    """The first chronological list is the publisher's chapter list.

    A later time reset starts timestamped glossary notes, not more chapters.
    Only the first line is a chapter title; section bodies can include ads/links.
    """
    result, previous = [], -1
    for section in (context.get("shownotes") or {}).get("sections", []):
        stamp = section.get("start_ms")
        if not isinstance(stamp, (int, float)) or stamp < 0:
            continue
        if stamp < previous:
            break
        text = str(section.get("text") or "").splitlines()[0].strip()
        title = re.sub(r"^\d+:\d{2}(?::\d{2})?\s*", "", text).strip()
        if title:
            result.append({"start_ms": int(stamp), "title": title})
        previous = stamp
    return result


def clean_boundaries(segments, *, max_join_chars=12, max_gap_ms=2500):
    rows = [copy.deepcopy(row) for row in segments]
    actions, omitted = [], set()
    for index, row in enumerate(rows):
        text = str(row.get("text") or "")
        token = plain(text)
        unknown_ack = uncertain(row) and token and _ACK.fullmatch(token)
        brief_hesitation = (token and _HESITATION.fullmatch(token)
                            and row["end_ms"] - row["start_ms"] <= 1200
                            and not re.search(r"[？?！!]", text))
        if unknown_ack or brief_hesitation:
            omitted.add(index)
            actions.append({"segment_id": row["id"], "action": "omit-backchannel",
                            "source_text": text, "source_sha256": value_hash(text),
                            "reason": "unknown-ack" if unknown_ack else "brief-hesitation"})
    for index, row in enumerate(rows):
        if index in omitted or not uncertain(row) or len(plain(row.get("text", ""))) > max_join_chars:
            continue
        neighbors = []
        for direction in (-1, 1):
            target = index + direction
            while 0 <= target < len(rows) and (target in omitted or uncertain(rows[target])):
                target += direction
            if not 0 <= target < len(rows):
                continue
            other = rows[target]
            gap = (row["start_ms"] - other["end_ms"] if direction == -1
                   else other["start_ms"] - row["end_ms"])
            if gap <= max_gap_ms:
                neighbors.append((max(0, gap), 0 if direction == -1 else 1, target))
        if neighbors:
            # This is a reading-only grouping, explicitly not a new voice label.
            _, _, target = min(neighbors)
            row["reading_speaker"] = rows[target].get("speaker", "")
            row["reading_identity"] = rows[target].get("speaker_uuid") or rows[target].get("speaker_id") or rows[target].get("speaker", "")
            row["reading_join"] = True
            actions.append({"segment_id": row["id"], "action": "join-neighbor-context",
                            "neighbor_segment_id": rows[target]["id"], "source_text": row.get("text", ""),
                            "acoustic_identity_changed": False})
    kept = []
    # Override nearest-neighbor grouping when a word was split across voice
    # boundaries. A 120ms '然' is not a separate conversational answer.
    for index in range(len(rows) - 2, -1, -1):
        row, other = rows[index], rows[index + 1]
        if index in omitted or index + 1 in omitted:
            continue
        raw = str(row.get("text") or "")
        token = plain(raw)
        token = re.sub(r"^(?:嗯|呃|啊)+", "", token)
        tails = _SPLITS.get(token)
        if (not tails or re.search(r"[。！？!?；;]", raw)
                or row["end_ms"] - row["start_ms"] > 1000
                or not -100 <= other["start_ms"] - row["end_ms"] <= 350
                or not plain(other.get("text", "")).startswith(tails)):
            continue
        identity = other.get("reading_identity") or other.get("speaker_uuid") or other.get("speaker_id") or other.get("speaker", "")
        label = other.get("reading_speaker") or other.get("speaker", "")
        if label in {"", "unknown", "归属待核对"} or label.startswith("交叠发言"):
            continue
        row.update(reading_identity=identity, reading_speaker=label, reading_join=True)
        # Remove only punctuation at the broken word, not the word or filler.
        row["text"] = re.sub(r"[，,、：:\s]+$", "", raw)
        actions.append({"segment_id": row["id"], "action": "join-split-word",
                        "neighbor_segment_id": other["id"], "source_text": raw,
                        "reading_text": row["text"], "word": token + tails[0],
                        "source_speaker_id": row.get("speaker_id"),
                        "acoustic_identity_changed": False})
    for index, row in enumerate(rows):
        if index in omitted:
            continue
        row.setdefault("reading_speaker", row.get("speaker") or "发言人1")
        row.setdefault("reading_identity", row.get("speaker_uuid") or row.get("speaker_id") or row["reading_speaker"])
        if row["reading_speaker"] in {"unknown", "发言者未确认"}:
            row["reading_speaker"] = "发言人1"
        row["source_ids"] = [row["id"]]
        readable = re.sub(r"我我(?=自己|现在|觉得|想|发现|认为|在|就|已经|把)", "我", row["text"])
        if readable != row["text"]:
            actions.append({"segment_id": row["id"], "action": "remove-mechanical-repeat",
                            "source_text": row["text"], "reading_text": readable})
            row["text"] = readable
        kept.append(row)
    return kept, {"schema_version": 1, "version": VERSION, "source_segments": len(rows),
                  "retained_segments": len(kept), "omitted_backchannels": len(omitted),
                  "neighbor_joins": sum(a["action"] == "join-neighbor-context" for a in actions),
                  "split_word_joins": sum(a["action"] == "join-split-word" for a in actions),
                  "actions": actions, "source_sha256": value_hash(segments),
                  "identity_policy": "presentation-only; source speaker IDs/UUIDs unchanged"}


def render_reading(segments, context, *, item_index=1):
    rows, audit = clean_boundaries(segments)
    chapters = chapters_from_context(context)
    output, pending, chapter_index = [], None, 0

    def flush():
        nonlocal pending
        if pending:
            output.extend([f"**{markdown_text(pending['reading_speaker'])}：**{markdown_text(pending['text'])}", ""])
            pending = None

    for row in rows:
        while chapter_index < len(chapters) and row["start_ms"] >= chapters[chapter_index]["start_ms"]:
            flush()
            chapter = chapters[chapter_index]
            stamp = chapter["start_ms"] // 1000
            output.extend([f'<a id="item-{item_index}-chapter-{chapter_index}"></a>',
                           f"#### {markdown_text(chapter_heading(chapter, chapter_index, item_index))}", ""])
            chapter_index += 1
        if (pending and pending["reading_identity"] == row["reading_identity"]
                and row["start_ms"] - pending["end_ms"] <= 2500
                and len(pending["text"]) + len(row["text"]) <= 1200):
            pending["text"] += row["text"]
            pending["end_ms"] = max(pending["end_ms"], row["end_ms"])
            pending["source_ids"].extend(row["source_ids"])
        else:
            flush()
            pending = copy.deepcopy(row)
    flush()
    audit.update(chapters=chapters, rendered_chapters=chapter_index,
                 accounted_segment_ids=[r["id"] for r in segments],
                 retained_text_sha256=value_hash("".join(r["text"] for r in rows)))
    return "\n".join(output), audit
