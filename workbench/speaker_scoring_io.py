"""Recording-scoped RTTM/UEM input and dscore's default 10ms JER semantics.

No model, NumPy or reference scorer is required. The independent reference
is used by a separate verification script, never vendored into this module.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import math
from pathlib import Path
import wave

from .diarization_metrics import evaluate_continuous_regions
from .speaker_benchmark import _maximum_weight_pairs


@dataclass(frozen=True)
class _Region:
    start_ms: float
    end_ms: float
    speaker: str
    start_seconds: float
    end_seconds: float


class _UemRange(list):
    def __init__(self, start, end):
        super().__init__([start * 1000, end * 1000])
        self.start_seconds, self.end_seconds = start, end


def _read(path):
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("scoring input must be a regular file <=32MiB")
    content = path.read_bytes()
    return content.decode("utf-8-sig"), hashlib.sha256(content).hexdigest()


def _seconds(value):
    try:
        number = Decimal(value)
        result = float(number)
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError("RTTM/UEM time must be a finite decimal") from None
    if not math.isfinite(result) or not math.isfinite(result * 1000) or result < 0:
        raise ValueError("RTTM/UEM time must be finite and nonnegative")
    return result


def read_rttm(path):
    """Return exact file/channel IDs, with decimal seconds kept at float ms.

    SPEAKER rows may have the standard nine or ten fields. Blank/comment
    lines and SPKR-INFO metadata are allowed; no other row kind is ignored.
    """
    text, digest = _read(Path(path))
    recordings = defaultdict(list)
    for index, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith(("#", ";")):
            continue
        if fields[0] == "SPKR-INFO":
            if len(fields) not in {9, 10}:
                raise ValueError(f"invalid SPKR-INFO row {index}")
            continue
        if len(fields) not in {9, 10} or fields[0] != "SPEAKER":
            raise ValueError(f"invalid RTTM SPEAKER row {index}")
        start, duration = _seconds(fields[3]), _seconds(fields[4])
        end = start + duration
        if (
            duration <= 0
            or not math.isfinite(end)
            or end <= start
            or fields[7] == "<NA>"
        ):
            raise ValueError(f"invalid RTTM interval/label at row {index}")
        recordings[(fields[1], fields[2])].append(
            _Region(start * 1000, end * 1000, fields[7], start, end)
        )
    return dict(recordings), digest


def read_uem(path):
    text, digest = _read(Path(path))
    recordings = defaultdict(list)
    for index, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith(("#", ";")):
            continue
        if len(fields) != 4:
            raise ValueError(f"invalid UEM row {index}")
        start, end = _seconds(fields[2]), _seconds(fields[3])
        if not start < end:
            raise ValueError(f"invalid UEM range at row {index}")
        recordings[(fields[0], fields[1])].append(_UemRange(start, end))
    return dict(recordings), digest


def _merge(ranges):
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def _intersection(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


class _Grid:
    def __init__(self, duration_seconds):
        # Match dscore: floor(float_seconds / .010), then times=.010*arange.
        self.count = int(duration_seconds / 0.010)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return 0.010 * index


def dscore_10ms_jer(reference, hypothesis, *, uem_ms):
    """JER on full UEM: no DER collar or overlap exclusion is applied.

    Trim before enumerating labels, like dscore CLI. Keep labels with zero
    sampled duration; the official implementation can be undefined for them.
    Sparse frame-index ranges avoid a recording-duration-sized allocation.
    """
    if not uem_ms:
        raise ValueError("JER requires explicit nonempty UEM")
    if any(
        type(x) not in (int, float) or not math.isfinite(x)
        for pair in uem_ms
        for x in pair
    ) or any(not 0 <= a < b for a, b in uem_ms):
        raise ValueError("invalid JER UEM")
    uem = _merge(
        [
            (
                getattr(pair, "start_seconds", pair[0] / 1000),
                getattr(pair, "end_seconds", pair[1] / 1000),
            )
            for pair in uem_ms
        ]
    )
    grid = _Grid(max(end for _, end in uem))

    def ranges(regions):
        grouped = defaultdict(list)
        for region in regions:
            if (
                type(region.start_ms) not in (int, float)
                or type(region.end_ms) not in (int, float)
                or not math.isfinite(region.start_ms)
                or not math.isfinite(region.end_ms)
                or not 0 <= region.start_ms < region.end_ms
                or not isinstance(region.speaker, str)
                or not region.speaker.strip()
            ):
                raise ValueError("invalid JER speaker region")
            for a, b in uem:
                start = max(a, getattr(region, "start_seconds", region.start_ms / 1000))
                end = min(b, getattr(region, "end_seconds", region.end_ms / 1000))
                if start < end:
                    grouped[region.speaker].append(
                        (bisect_left(grid, start), bisect_left(grid, end))
                    )
        if len(grouped) > 256:
            raise ValueError("scoring supports at most 256 labels per recording")
        return {label: _merge(values) for label, values in sorted(grouped.items())}

    refs, hyps = ranges(reference), ranges(hypothesis)
    rd = {label: sum(b - a for a, b in values) for label, values in refs.items()}
    hd = {label: sum(b - a for a, b in values) for label, values in hyps.items()}
    result = {
        "metric_protocol": "dscore-default-10ms-v1",
        "reference_commit": "e02f949ac6592279300a2c33d03daf9e0c12fd27",
        "frame_step_seconds": 0.010,
        "collar_applied": False,
        "overlap_included": True,
        "mapping_scope": "recording",
        "mapping_objective": "intersection_over_union",
        "reference_speaker_frames": rd,
        "hypothesis_speaker_frames": hd,
        "reference_speaker_count": len(rd),
        "hypothesis_speaker_count": len(hd),
        "reference_zero_frame_labels": [x for x in rd if rd[x] == 0],
        "hypothesis_zero_frame_labels": [x for x in hd if hd[x] == 0],
        "defined": True,
    }
    if any(x == 0 for x in rd.values()) and any(x == 0 for x in hd.values()):
        return {
            **result,
            "defined": False,
            "jer": None,
            "speaker_error_sum": None,
            "reason": "dscore_assignment_undefined_zero_union; no fabricated score",
            "speaker_mapping": {},
        }
    labels_r, labels_h = list(rd), list(hd)
    ious = []
    for h in labels_h:
        row = []
        for r in labels_r:
            intersection = _intersection(refs[r], hyps[h])
            row.append(intersection / (rd[r] + hd[h] - intersection))
        ious.append(row)
    pairs = _maximum_weight_pairs(ious)
    errors = len(rd) - sum(ious[h][r] for h, r in pairs)
    return {
        **result,
        "speaker_error_sum": errors,
        "jer": errors / len(rd) if rd else float(bool(hd)),
        "speaker_mapping": {labels_h[h]: labels_r[r] for h, r in pairs},
    }


def _canonical_regions(regions, uem_ms):
    grouped = defaultdict(list)
    for region in regions:
        for pair in uem_ms:
            a = getattr(pair, "start_seconds", pair[0] / 1000)
            b = getattr(pair, "end_seconds", pair[1] / 1000)
            start, end = max(a, region.start_seconds), min(b, region.end_seconds)
            if start < end:
                grouped[region.speaker].append((start, end))
    if len(grouped) > 256:
        raise ValueError("scoring supports at most 256 labels per recording")
    return [
        _Region(a * 1000, b * 1000, label, a, b)
        for label, values in sorted(grouped.items())
        for a, b in _merge(values)
    ]


def evaluate_recording_manifest(manifest_path, payload):
    """Version 3 explicitly binds conventional RTTM/UEM to actual PCM audio."""
    root = manifest_path.parent
    entries = payload.get("items")
    if (
        payload.get("schema_version") != 3
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError(
            "recording-eval-v3 requires schema_version 3 and nonempty items"
        )
    collar, skip = payload.get("collar_ms", 0), payload.get("skip_overlap", False)
    synthetic = payload.get("synthetic_only", False)
    if type(synthetic) is not bool:
        raise ValueError("synthetic_only must be an explicit boolean")
    text_requested = any(isinstance(e, dict) and "text_reference" in e for e in entries)
    text_task = payload.get("text_task", "raw-asr")
    if text_requested:
        from .transcript_metrics import aggregate_text_scores, evaluate_text_files
        if not isinstance(text_task, str) or text_task not in {"raw-asr", "postprocessed"}:
            raise ValueError("text_task must be raw-asr or postprocessed")
        if not all(isinstance(e, dict) and "text_reference" in e for e in entries):
            raise ValueError("text reference is required for every recording, never a scored subset")
    identities, audio_hashes, engine_set, items, totals = set(), set(), None, [], {}
    if any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("recording_id"), str)
        or not entry["recording_id"].strip()
        for entry in entries
    ):
        raise ValueError("every scoring entry requires a nonempty recording_id")
    allowed = {
        (entry["recording_id"], str(entry.get("channel", "1"))) for entry in entries
    }

    def path(value):
        if not isinstance(value, str) or not value:
            raise ValueError("scoring file path is required")
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    def annotation(value, kind, key, duration, audio_hash):
        if (
            not isinstance(value, dict)
            or value.get("annotation_kind") != kind
            or value.get("format") != "rttm"
        ):
            raise ValueError(f"explicit RTTM {kind} declaration required")
        if kind == "human-diarization" and value.get("annotation_status") != "complete":
            raise ValueError("human reference must be complete; pending is not silence")
        if (
            value.get("recording_id") != key[0]
            or value.get("source_audio_sha256") != audio_hash
            or str(value.get("channel", "1")) != key[1]
        ):
            raise ValueError(
                "annotation must bind recording/channel and actual audio hash"
            )
        p = path(value.get("path"))
        data, digest = read_rttm(p)
        if set(data) - allowed:
            raise ValueError("RTTM contains undeclared recording/channel IDs")
        regions = data.get(key, [])
        if (not regions) != (value.get("content_state") == "no_speech"):
            raise ValueError(
                "empty RTTM selection requires explicit no_speech, never pending"
            )
        if any(r.end_ms > duration + 1e-6 for r in regions):
            raise ValueError("RTTM extends beyond actual source audio")
        return p, digest, regions

    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("recording_id"), str)
            or not entry["recording_id"]
            or entry["recording_id"] in identities
        ):
            raise ValueError(
                "each recording_id must occur once; combine sparse UEM windows"
            )
        rid = entry["recording_id"]
        identities.add(rid)
        key = (rid, str(entry.get("channel", "1")))
        source = entry.get("source_audio") or {}
        if not isinstance(source, dict):
            raise ValueError("source_audio must be a file/hash declaration")
        audio_path = path(source.get("path"))
        if not audio_path.is_file():
            raise ValueError("source audio must be a regular file")
        before_audio = audio_path.stat()
        h = hashlib.sha256()
        with audio_path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        audio_hash = h.hexdigest()
        if source.get("sha256") != audio_hash or audio_hash in audio_hashes:
            raise ValueError("source audio hash mismatch or duplicate source recording")
        audio_hashes.add(audio_hash)
        with wave.open(str(audio_path), "rb") as wav:
            if (
                wav.getcomptype() != "NONE"
                or not key[1].isdigit()
                or not 1 <= int(key[1]) <= wav.getnchannels()
            ):
                raise ValueError("scoring requires PCM WAV and matching channel")
            frames, rate, width, channels = (
                wav.getnframes(),
                wav.getframerate(),
                wav.getsampwidth(),
                wav.getnchannels(),
            )
            actual = 0
            while data := wav.readframes(65536):
                actual += len(data)
            if not frames or actual != frames * width * channels:
                raise ValueError("source audio is empty or truncated")
            duration = frames * 1000 / rate
        after_audio = audio_path.stat()
        if (before_audio.st_ino, before_audio.st_size, before_audio.st_mtime_ns) != (
            after_audio.st_ino,
            after_audio.st_size,
            after_audio.st_mtime_ns,
        ):
            raise ValueError("source audio changed during validation")
        uem_path = path(entry.get("uem"))
        uems, uem_hash = read_uem(uem_path)
        if (
            set(uems) - allowed
            or key not in uems
            or any(end > duration + 1e-6 for _, end in uems[key])
        ):
            raise ValueError("UEM must match recording/channel and actual audio range")
        ref_path, ref_hash, reference = annotation(
            entry.get("reference"), "human-diarization", key, duration, audio_hash
        )
        hypotheses = entry.get("hypotheses")
        if (
            not isinstance(hypotheses, dict)
            or not hypotheses
            or any(not isinstance(x, str) or not x.strip() for x in hypotheses)
        ):
            raise ValueError("named hypotheses are required")
        if engine_set is not None and set(hypotheses) != engine_set:
            raise ValueError("all recordings must evaluate the same engine set")
        engine_set = set(hypotheses)
        for engine, value in hypotheses.items():
            hyp_path, hyp_hash, hypothesis = annotation(
                value, "acoustic-diarization", key, duration, audio_hash
            )
            continuous = evaluate_continuous_regions(
                _canonical_regions(reference, uems[key]),
                _canonical_regions(hypothesis, uems[key]),
                uem_ms=uems[key],
                collar_ms=collar,
                skip_overlap=skip,
                allow_fractional_ms=True,
            )
            sampled = dscore_10ms_jer(reference, hypothesis, uem_ms=uems[key])
            if not text_requested and ("text" in value or "fixed_speaker_mapping" in value):
                raise ValueError("hypothesis text requires a common text_reference")
            text_score = evaluate_text_files(
                entry["text_reference"], value.get("text"), root=root,
                recording_id=rid, source_audio_sha256=audio_hash, uem_ms=uems[key],
                duration_ms=duration, task=text_task, fixed_mapping=value.get("fixed_speaker_mapping"),
                channel=key[1],
                reference_speakers={r.speaker for r in reference},
                hypothesis_speakers={r.speaker for r in hypothesis},
            ) if text_requested else None
            items.append(
                {
                    "recording_id": rid,
                    "engine": engine,
                    "source_audio_sha256": audio_hash,
                    "source_audio_duration_ms": duration,
                    "channel": key[1],
                    "reference_sha256": ref_hash,
                    "hypothesis_sha256": hyp_hash,
                    "uem_sha256": uem_hash,
                    "reference": str(ref_path),
                    "hypothesis": str(hyp_path),
                    "uem": str(uem_path),
                    "continuous": continuous,
                    "dscore_10ms": sampled,
                    "text_metrics": text_score,
                }
            )
            total = totals.setdefault(
                engine,
                {
                    "reference_speaker_ms": 0,
                    "miss_ms": 0,
                    "false_alarm_ms": 0,
                    "confusion_ms": 0,
                    "continuous_jer_error_sum": 0,
                    "continuous_jer_speakers": 0,
                    "continuous_hypothesis_speakers": 0,
                    "sampled_jer_error_sum": 0,
                    "sampled_reference_speakers": 0,
                    "sampled_hypothesis_speakers": 0,
                    "sampled_undefined_recordings": [],
                    "recordings": 0,
                },
            )
            for k in (
                "reference_speaker_ms",
                "miss_ms",
                "false_alarm_ms",
                "confusion_ms",
            ):
                total[k] += continuous[k]
            total["continuous_jer_error_sum"] += continuous["jer_speaker_error_sum"]
            total["continuous_jer_speakers"] += continuous["jer_speaker_count"]
            total["continuous_hypothesis_speakers"] += len(
                continuous["hypothesis_speakers"]
            )
            total["sampled_reference_speakers"] += sampled["reference_speaker_count"]
            total["sampled_hypothesis_speakers"] += sampled["hypothesis_speaker_count"]
            total["recordings"] += 1
            if sampled["defined"]:
                total["sampled_jer_error_sum"] += sampled["speaker_error_sum"]
            else:
                total["sampled_undefined_recordings"].append(rid)
    aggregate = {}
    for engine, total in totals.items():
        denominator, speakers = (
            total["reference_speaker_ms"],
            total["sampled_reference_speakers"],
        )
        aggregate[engine] = {
            **total,
            "der": sum(total[k] for k in ("miss_ms", "false_alarm_ms", "confusion_ms"))
            / denominator
            if denominator
            else None,
            "jer_continuous": total["continuous_jer_error_sum"]
            / total["continuous_jer_speakers"]
            if total["continuous_jer_speakers"]
            else float(total["continuous_hypothesis_speakers"] > 0),
            "jer_dscore_10ms": None
            if total["sampled_undefined_recordings"]
            else total["sampled_jer_error_sum"] / speakers
            if speakers
            else float(total["sampled_hypothesis_speakers"] > 0),
            "text_metrics": aggregate_text_scores([
                item["text_metrics"] for item in items if item["engine"] == engine
            ]) if text_requested else None,
        }
    return {
        "schema_version": 3,
        "metric_protocol": "recording-eval-v3",
        "synthetic_only": synthetic,
        "model_release_approved": False,
        "mapping_scope": "recording",
        "text_status": "scored" if text_requested else "not_requested",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "aggregate": aggregate,
        "items": items,
        "continuous_input_policy": "RTTM trimmed and same-speaker merged to UEM before DER collar, matching dscore CLI",
        "scope": "synthetic software test only"
        if synthetic
        else "declared completed human UEM only; not automatic whole-episode or identity verification",
    }
