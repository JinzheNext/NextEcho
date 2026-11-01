"""Whisper原生分段与字幕cue；跨界完整保留，不推测字符时间。"""
from __future__ import annotations

import re

from .contracts import PipelineError


def keep_interval(start_ms: int, end_ms: int, trim_before_ms: int) -> bool:
    # end恰好等于边界的旧上下文归前片；跨界整段保留，不把start裁成边界。
    return not (start_ms < trim_before_ms and end_ms <= trim_before_ms)


def shifted_segments(payload: dict, offset_ms: int, trim_before_ms: int = 0) -> list[dict]:
    if any(type(t) is not int or t < 0 for t in (offset_ms, trim_before_ms)):
        raise PipelineError("INVALID_INPUT", "Whisper分片偏移无效", stage="asr")
    if not isinstance(payload, dict):
        raise PipelineError("PROTOCOL_ERROR", "Whisper分片不是JSON对象", stage="asr")
    rows = payload.get("transcription")
    if not isinstance(rows, list):
        raise PipelineError("PROTOCOL_ERROR", "Whisper分片缺少原生分段列表", stage="asr")
    result = []
    previous = -1
    for index, item in enumerate(rows):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise PipelineError("PROTOCOL_ERROR", "Whisper原生分段格式无效", stage="asr")
        if not item["text"].strip():
            continue
        offsets = item.get("offsets") or {}
        left, right = offsets.get("from"), offsets.get("to")
        if (any(type(t) is not int for t in (left, right)) or left < 0 or right <= left or left < previous):
            raise PipelineError("PROTOCOL_ERROR", "Whisper原生分段时间缺失、逆序或无效", stage="asr")
        previous = left
        start, end = left + offset_ms, right + offset_ms
        if not keep_interval(start, end, trim_before_ms):
            continue
        row = {"offsets": {"from": start, "to": end}, "text": item["text"],
               "source_segment_index": index, "source_offsets": {"from": left, "to": right},
               "source_chunk_offset_ms": offset_ms}
        if item.get("id") is not None:
            row["source_segment_id"] = item["id"]
        if start < trim_before_ms < end:
            row["seam_review"] = {"reason": "cross_boundary_without_word_alignment",
                "boundary_ms": trim_before_ms, "policy": "retain_whole_native_segment",
                "possible_duplicate_prefix": True}
        result.append(row)
    return result


def parse_timestamp(value: str, *, vtt: bool) -> int:
    separator = "." if vtt else ","
    if not re.fullmatch(r"(?:\d{2,}:)?\d{2}:\d{2}" + re.escape(separator) + r"\d{3}", value):
        raise PipelineError("PROTOCOL_ERROR", "字幕时间格式无效", stage="asr")
    seconds, ms = value.rsplit(separator, 1)
    parts = list(map(int, seconds.split(":")))
    if len(parts) == 2:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    if minutes >= 60 or seconds >= 60:
        raise PipelineError("PROTOCOL_ERROR", "字幕时间超出范围", stage="asr")
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + int(ms)


def subtitle_cues(text: str, *, vtt: bool) -> list[dict]:
    """保留cue ID、多行正文和VTT设置；header/NOTE/STYLE/REGION不是正文。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    result = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = block.strip().splitlines()
        if not lines:
            continue
        if vtt and (lines[0].startswith("WEBVTT") or re.match(r"^(NOTE|STYLE|REGION)(?:\s|$)", lines[0])):
            continue
        at = next((i for i in range(min(2, len(lines))) if "-->" in lines[i]), None)
        if at is None or len(lines) <= at + 1:
            raise PipelineError("PROTOCOL_ERROR", "字幕cue缺少完整时间和正文", stage="asr")
        timing = re.fullmatch(r"(\S+)\s+-->\s+(\S+)(.*)", lines[at])
        if not timing:
            raise PipelineError("PROTOCOL_ERROR", "字幕cue时间行无效", stage="asr")
        start = parse_timestamp(timing[1], vtt=vtt)
        end = parse_timestamp(timing[2], vtt=vtt)
        if end <= start:
            raise PipelineError("PROTOCOL_ERROR", "字幕cue时间范围无效", stage="asr")
        result.append({"start_ms": start, "end_ms": end, "settings": timing[3] if vtt else "",
                       "identifier": lines[0] if at else "", "lines": lines[at + 1:]})
    return result
