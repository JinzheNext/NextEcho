"""Small, cited terminology hints retrieved from the episode's own sources.

Retrieval never edits transcript text. Comment opinions and publisher corrections
of an actual spoken mistake cannot become purported speech.
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

CONTEXT_VERSION = "episode-terminology-v5-typed-editorial"
_MARKER = re.compile(r"实际为|实为|应为|应是")
_QUOTED = re.compile(r'''《([^》\n]{1,80})》|「([^」\n]{1,80})」|“([^”\n]{1,80})”|‘([^’\n]{1,80})’|"([^"\n]{1,80})"|'([^'\n]{1,80})' ''', re.X)


@dataclass(frozen=True)
class EditorialEvidence:
    context_ref: str
    source_kind: str
    marker: str
    spoken_form: str | None
    editorial_form: str | None
    parse_status: str
    source_text_sha256: str


def parse_editorial_text(text: str, context_ref: str, *, source_kind="publisher") -> list[EditorialEvidence]:
    """One parser for retrieval, hotwords, validation and separate audit notes.

    A publisher's quotation is still a claim about speech, not acoustic truth.
    Ambiguous corrections are withheld rather than supplied as ordinary terms.
    """
    rows = []
    for clause in re.split(r"[。！？；\n]", text):
        for marker in _MARKER.finditer(clause):
            left, right = clause[:marker.start()], clause[marker.end():].lstrip(" ：:")
            quotes = list(_QUOTED.finditer(left))
            intent = bool(re.search(r"此处|提及|说的|口误|勘误|更正", left))
            if not quotes and not intent:
                continue
            spoken = next((g for g in quotes[-1].groups() if g is not None), None) if quotes else None
            if not spoken:
                plain = re.search(r"(?:提及的|说的)\s*([^，,:：]{2,32})$", left)
                spoken = plain[1].strip() if plain else None
            target = re.sub(r"^(?:电影|书籍|书名|播客)\s*", "", right)
            quoted = _QUOTED.match(target)
            editorial = next((g for g in quoted.groups() if g is not None), None) if quoted else None
            if not editorial:
                plain = re.match(r"([\u3400-\u9fffA-Za-z0-9·.-]{2,32})(?=[，,:：\s]|$)", target)
                editorial = plain[1] if plain else None
            rows.append(EditorialEvidence(context_ref, source_kind, marker[0], spoken, editorial,
                "parsed" if intent and spoken and editorial else "unresolved", hashlib.sha256(text.encode()).hexdigest()))
    return rows


def editorial_evidence(payload: dict) -> list[EditorialEvidence]:
    rows = []
    notes = payload.get("shownotes") or {}
    sections = notes.get("sections", [])
    for section in sections:
        rows.extend(parse_editorial_text(str(section.get("text", "")), str(section.get("context_ref", ""))))
    untimed = str(notes.get("text", ""))
    if sections:
        untimed = re.split(r"(?<!\d)\d{1,3}:[0-5]\d", untimed, maxsplit=1)[0]
    rows.extend(parse_editorial_text(untimed, "shownotes:untimed"))
    return rows


def editorial_notes(payload: dict) -> list[dict]:
    return [{**asdict(row), "transcript_evidence": False, "speaker_identity_evidence": False,
             "policy": "publisher claim; separate note only, not a transcript replacement"}
            for row in editorial_evidence(payload)]


def terminology(payload: dict) -> list[dict]:
    terms = []
    corrections = editorial_evidence(payload)
    editorial_targets = {row.editorial_form for row in corrections if row.editorial_form}
    for speaker in payload.get("speaker_candidates", []):
        name = str(speaker.get("name", "")).strip()
        if name:
            terms.append(
                {
                    "term": name,
                    "context_ref": "metadata:" + str(speaker.get("source", "speaker")),
                }
            )
    shownotes = payload.get("shownotes") or {}
    # Official programme/product names can occur only in the introduction or
    # credits. A partial coverage view has already removed untimed text.
    for match in re.finditer(
        r"(?:欢迎来到|播客|产品|公司|公众号)[^。\n]{0,12}?[「“]([^」”]{2,12})[」”]",
        str(shownotes.get("text", "")),
    ):
        terms.append(
            {"term": match.group(1), "context_ref": "metadata:shownotes-names"}
        )
    for section in (payload.get("shownotes") or {}).get("sections", []):
        text = str(section.get("text", ""))
        if parse_editorial_text(text, str(section.get("context_ref", ""))):
            continue
        # Glossary entries: timestamp + term + colon/parenthesized English.
        head = re.sub(r"^\s*\[?\d{1,3}:[0-5]\d(?::[0-5]\d)?\]?\s*", "", text)
        match = re.match(r"([^：:\n（(]{2,32})\s*(?:[：:（(])", head)
        if match:
            term = match.group(1).strip(" 《》「」“”")
            book_title = head.startswith("《")
            if (
                len(term) <= 24
                and not re.search(
                    r"[，。；！？]|(?:此处|嘉宾|点击|作者|下文|以下)", term
                )
                and (
                    book_title
                    or len(term) <= 10
                    and not re.search(
                        r"[、的]|^(?:我|你|他|她|所有|回归)|两篇文章|终点", term
                    )
                )
            ):
                terms.append(
                    {"term": term, "context_ref": str(section.get("context_ref", ""))}
                )
    unique = {}
    for item in terms:
        if item["term"] not in editorial_targets:
            unique.setdefault(item["term"], item)
    return list(unique.values())[:48]


def asr_hotwords(payload: dict, explicit: list | None = None) -> list[str]:
    # Only publisher metadata/glossary, never viewer opinions, enter ASR prompts.
    values = [str(v).strip() for v in (explicit or [])] + [
        r["term"] for r in terminology(payload)
    ]
    return list(
        dict.fromkeys(
            v for v in values if 1 < len(v) <= 32 and not re.search(r"[\n\r\[\]]", v)
        )
    )[:48]


def _grams(text: str) -> set:
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9.-]{2,}", text.lower())) | {
        run[i : i + 2]
        for run in re.findall(r"[\u3400-\u9fff]+", text)
        for i in range(len(run) - 1)
    }
    return tokens - set(
        "然后 就是 这个 那个 我们 你们 他们 自己 时候 觉得 知道 可以 不是 没有 一个 什么 那么 这样 所以 因为 还是 现在 可能 其实 比较 所有 但是 如果 已经 事情 大家 这个 个人 东西 这些 那些".split()
    )


def retrieve_context(payload: dict, segments: list, *, max_chars: int = 850) -> dict:
    query = " ".join(s.text for s in segments)
    grams = _grams(query)
    left = min((s.start_ms for s in segments), default=0)
    right = max((s.end_ms for s in segments), default=0)
    candidates = []
    publisher_targets = sorted({e.editorial_form for e in editorial_evidence(payload)
        if e.editorial_form and e.editorial_form != e.spoken_form}, key=len, reverse=True)
    for section in (payload.get("shownotes") or {}).get("sections", []):
        text = str(section.get("text", "")).split("\n\n", 1)[0]
        start = section.get("start_ms")
        distance = (
            max(left - start, start - right, 0) if isinstance(start, int) else 10**10
        )
        other = _grams(text)
        overlap = len(grams & other)
        density = overlap / max(1, min(len(grams), len(other)))
        if distance <= 120000 or overlap >= 4 and density >= 0.15:
            evidence = parse_editorial_text(str(section.get("text", "")), str(section.get("context_ref", "")))
            spoken = list(dict.fromkeys(e.spoken_form for e in evidence if e.parse_status == "parsed"))
            if evidence:
                text = ("编者引述的原声用词：" + "、".join("「" + word + "」" for word in spoken)
                        + "。仅辅助音近误识别核对，未获声音确认，不修正说话人的知识或观点。") if spoken else "本处有未解析的编者勘误，不能当作原声或拼写事实；待人工审核。"
            candidates.append(
                (
                    overlap + (12 if distance <= 120000 else 0),
                    {
                        "context_ref": section.get("context_ref"),
                        "text": text,
                        "kind": "publisher_editorial_correction"
                        if evidence
                        else "shownote",
                        "transcript_evidence": False,
                        "speaker_identity_evidence": False,
                        "trust": "publisher-claim-not-acoustic-truth",
                        **(
                            {"spoken_form": spoken[0], "spoken_forms": spoken}
                            if spoken
                            else {}
                        ),
                    },
                )
            )
    for comment in payload.get("comments", []):
        for row in [comment, *comment.get("replies", [])]:
            text = str(row.get("text", ""))
            # Rank sentences rather than truncate a long comment's first page.
            for sentence in re.split(r"(?<=[。！？\n])", text):
                other = _grams(sentence)
                score = len(grams & other)
                if score >= 3 and score / max(1, min(len(grams), len(other))) >= 0.15:
                    excerpt = sentence
                    for target in publisher_targets:
                        excerpt = excerpt.replace(target, "[编者修正词另见审计]")
                    candidates.append(
                        (
                            min(10, score),
                            {
                                "context_ref": row.get("context_ref"),
                                "text": excerpt,
                                "kind": "comment_hint",
                                "transcript_evidence": False,
                                "speaker_identity_evidence": False,
                                "trust": "unverified-comment",
                                "allowed_use": "spelling-hint-only",
                                "editorial_target_redacted": excerpt != sentence,
                            },
                        )
                    )
    selected, used, comment_refs = [], 0, set()
    for _, row in sorted(candidates, key=lambda v: v[0], reverse=True):
        if used >= max_chars:
            break
        if row["kind"] == "comment_hint":
            if len(comment_refs) >= 2 or row["context_ref"] in comment_refs:
                continue
            comment_refs.add(row["context_ref"])
        text = re.sub(r"https?://\S+", "[链接省略]", row["text"]).strip()
        text = text[: min(300, max_chars - used)]
        if text:
            selected.append({**row, "text": text})
            used += len(text)
    selected_refs = {h["context_ref"] for h in selected}
    names = [
        r
        for r in terminology(payload)
        if r["context_ref"].startswith("metadata:")
        or r["term"] in query
        or r["context_ref"] in selected_refs
    ]
    names.extend(
        {
            "term": form,
            "context_ref": h["context_ref"],
            "kind": "spoken_form",
        }
        for h in selected
        for form in h.get("spoken_forms", [])
    )
    return {
        "version": CONTEXT_VERSION,
        "terminology": names[:12],
        "hints": selected,
        "policy": "仅辅助音近错字、专名拼写与断句；评论不是事实证据；编辑勘误不能替换说话人实际口误。",
    }


def validate_spoken_forms(source, edited, context: dict, supplied: dict) -> None:
    """Reject copying an editor's correction over an actual spoken mistake."""
    by_id = {s.id: s for s in edited}
    from .chinese_text import simplified
    # Validate the full scoped source, even when budget fitting removes hints.
    # Never trust a mutable supplied spoken_form as the parser's authority.
    for evidence in editorial_evidence(context):
        spoken, editorial = evidence.spoken_form, evidence.editorial_form
        if evidence.parse_status != "parsed":
            continue
        spoken, editorial = simplified(spoken), simplified(editorial)
        for segment in source:
            before, after = simplified(segment.text), simplified(by_id[segment.id].text)
            similar = (
                spoken in before
                or len(spoken) >= 3
                and spoken[-2:] in before
                and any(
                    SequenceMatcher(
                        None, spoken, before[i : i + len(spoken)], autojunk=False
                    ).ratio()
                    >= 0.75
                    for i in range(max(0, len(before) - len(spoken) + 1))
                )
            )
            if similar and editorial not in before and editorial in after:
                raise ValueError(
                    "不得把编者勘误写成原声；本段保留说话人用词：" + spoken
                )
