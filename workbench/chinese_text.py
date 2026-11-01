"""A derived script layer: raw ASR bytes remain immutable."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .io_utils import atomic_write, file_hash, write_json

SCRIPT_VERSION = "opencc-python-reimplemented-0.1.7:t2s:protected-literals-v1"
_LITERAL = re.compile(
    r"```[\s\S]*?```|`[^`\n]*`|https?://[^\s<>]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
)


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC

    return OpenCC("t2s")


def simplified(text: str) -> str:
    parts, start = [], 0
    for match in _LITERAL.finditer(text):
        parts.extend((_converter().convert(text[start : match.start()]), match.group()))
        start = match.end()
    parts.append(_converter().convert(text[start:]))
    return "".join(parts)


def write_simplified_artifacts(item_dir: Path, payload: dict) -> dict:
    from .postprocess import load_segments, segment_to_transcription
    from .transcription import (
        format_transcript_attribution,
        write_subtitles_from_segments,
    )

    raw = item_dir / "transcript.json"
    rows, edits = [], []
    for segment in load_segments(payload):
        converted = simplified(segment.text)
        if converted != segment.text:
            edits.append({"id": segment.id, "before": segment.text, "after": converted})
        row = segment_to_transcription(segment)
        row["text"] = converted
        rows.append(row)
    paths = {
        "normalized_" + kind + "_path": str(item_dir / ("transcript.simplified." + ext))
        for kind, ext in (
            ("text", "txt"),
            ("json", "json"),
            ("srt", "srt"),
            ("vtt", "vtt"),
        )
    }
    audit = {"version": SCRIPT_VERSION, "raw_sha256": file_hash(raw), "edits": edits}
    write_json(
        Path(paths["normalized_json_path"]),
        {**payload, "transcription": rows, "script_normalization": audit},
    )
    atomic_write(
        Path(paths["normalized_text_path"]),
        format_transcript_attribution("\n".join(r["text"] for r in rows)),
    )
    write_subtitles_from_segments(
        rows,
        srt_path=Path(paths["normalized_srt_path"]),
        vtt_path=Path(paths["normalized_vtt_path"]),
    )
    write_json(item_dir / "script-normalization.audit.json", audit)
    return paths
