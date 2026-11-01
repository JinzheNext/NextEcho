"""Independent default binarization and long-stream artifact consistency.

This validates native output retention, not speaker accuracy or identities.
Constants and order match the pinned NeMo-Speech.cpp DiarSegmentationCfg.
"""
from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path

from .io_utils import file_hash


def segment_retained_probabilities(values, *, first_frame, slots=8, cadence=.01):
    if len(values) % slots:
        raise ValueError("incomplete probability frame")
    frames = len(values) // slots
    total = frames * cadence
    onset, offset = ctypes.c_float(.641).value, ctypes.c_float(.561).value
    turns = []
    for speaker in range(slots):
        active, start, intervals = False, 0, []
        for frame in range(frames):
            value = values[frame * slots + speaker]
            if not active and value > onset:
                active, start = True, frame
            elif active and value < offset:
                active = False
                intervals.append((max(0., start * cadence - .229), min(total, frame * cadence + .079)))
        if active:
            intervals.append((max(0., start * cadence - .229), total))
        merged = []
        for a, b in intervals:
            if merged and a - merged[-1][1] < .296:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            if b - a >= .511:
                base = first_frame * cadence
                turns.append({"start": a + base, "end": b + base, "speaker": speaker + 1})
    return sorted(turns, key=lambda r: (r["start"], r["speaker"]))


def validate_segments(rows, duration, slots=8):
    if not isinstance(rows, list) or len(rows) > 100000:
        raise ValueError("too many or missing native segments")
    previous = -1
    for row in rows:
        a, b, speaker = row["start"], row["end"], row["speaker"]
        if (type(a) not in (int, float) or type(b) not in (int, float)
                or not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b <= duration + .16
                or a < previous or type(speaker) is not int or not 1 <= speaker <= slots):
            raise ValueError("invalid native interval/channel")
        previous = a


def verify_long_timeline(data, folder, probabilities):
    count, first, duration = data["frame_count"], data["retained_frame_start"], data["duration_seconds"]
    cadence = data["seconds_per_frame"]
    if (data.get("timeline_protocol") != "native-retained-tail-v2"
            or data.get("probability_scope") != "retained-tail"
            or data.get("segments_scope") != "native-whole-timeline"
            or data.get("completed") is not True
            or type(data.get("input_samples_consumed")) is not int
            or data["input_samples_consumed"] != round(duration * 16000)
            or data.get("resume_granularity") != "whole-stage"
            or count < math.floor(duration * 100) - 16):
        raise ValueError("incomplete or ambiguous long-stream coverage")
    validate_segments(data["segments"], duration)
    base = first * cadence
    if any(r["start"] < base - 1e-6 and r["end"] > base + 1e-6 for r in data["segments"]):
        raise ValueError("native segment straddles the compaction cut")
    actual_tail = sorted((r["speaker"], r["start"], r["end"]) for r in data["segments"] if r["start"] >= base - 1e-6)
    expected = segment_retained_probabilities(probabilities, first_frame=first, cadence=cadence)
    expected_tail = sorted((r["speaker"], r["start"], r["end"]) for r in expected)
    if len(actual_tail) != len(expected_tail) or any(
        a[0] != b[0] or abs(a[1] - b[1]) > 1e-6 or abs(a[2] - b[2]) > 1e-6
        for a, b in zip(actual_tail, expected_tail)
    ):
        raise ValueError("native tail differs from independent default probability binarization/global clock")
    checkpoints = data.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints or len(checkpoints) > 1000:
        raise ValueError("bounded long-stream checkpoints required")
    last_count = last_first = last_samples = -1
    final_turns = {(r["speaker"], r["start"], r["end"]) for r in data["segments"]}
    frozen_checked = 0
    for item in checkpoints:
        path = Path(item["path"])
        if (not path.is_absolute() or not path.resolve().is_relative_to(folder.resolve())
                or not path.is_file() or path.stat().st_size > 32 * 1024**2 or file_hash(path) != item["sha256"]):
            raise ValueError("checkpoint changed or outside probe")
        checkpoint = json.loads(path.read_text())
        n, f, samples = checkpoint["frame_count"], checkpoint["retained_frame_start"], checkpoint["input_samples_consumed"]
        if (any(type(x) is not int for x in (n, f, samples))
                or not max(0, last_count) <= n <= count or not 0 <= f <= n or f < last_first
                or not max(0, last_samples) <= samples <= data["input_samples_consumed"]
                or n * cadence > samples / 16000 + .16):
            raise ValueError("checkpoint clock went backwards/out of final scope")
        validate_segments(checkpoint["segments"], duration)
        frozen = {(r["speaker"], r["start"], r["end"]) for r in checkpoint["segments"] if r["end"] <= f * cadence}
        if not frozen.issubset(final_turns):
            raise ValueError("final output lost or rewrote frozen prefix segments")
        frozen_checked += len(frozen)
        last_count, last_first, last_samples = n, f, samples
    if (last_count, last_first, last_samples) != (count, first, data["input_samples_consumed"]):
        raise ValueError("missing exact final checkpoint")
    return {"probability_tail_segments_compared": len(expected_tail), "checkpoint_count": len(checkpoints),
            "frozen_prefix_segment_observations_checked": frozen_checked,
            "retained_probability_frames": count - first, "compaction_observed": first > 0,
            "scope": "artifact retention and native default binarization only; no speaker quality or identity proof"}
