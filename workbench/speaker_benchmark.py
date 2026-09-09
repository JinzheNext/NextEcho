from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpeakerRegion:
    start_ms: int
    end_ms: int
    speaker: str


def load_regions(path: Path) -> list[SpeakerRegion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("segments") or payload.get("speaker_turns") or payload.get("turns") or []
    else:
        rows = []
    regions: list[SpeakerRegion] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start_ms = _time_ms(row, "start_ms", "start")
        end_ms = _time_ms(row, "end_ms", "end")
        speaker = str(
            row.get("speaker_id")
            or row.get("display_name")
            or row.get("speaker")
            or row.get("raw_speaker_label")
            or ""
        ).strip()
        if speaker and end_ms > start_ms >= 0:
            regions.append(SpeakerRegion(start_ms, end_ms, speaker))
    return sorted(regions, key=lambda item: (item.start_ms, item.end_ms, item.speaker))


def evaluate_regions(
    reference: list[SpeakerRegion],
    hypothesis: list[SpeakerRegion],
    *,
    frame_ms: int = 20,
    switch_tolerance_ms: int = 500,
) -> dict[str, Any]:
    if not reference:
        raise ValueError("reference contains no valid speaker regions")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    start_ms = min(item.start_ms for item in reference)
    end_ms = max(item.end_ms for item in reference)
    ref_frames = _frame_sets(reference, start_ms, end_ms, frame_ms)
    hyp_frames = _frame_sets(hypothesis, start_ms, end_ms, frame_ms)
    mapping = optimal_speaker_mapping(ref_frames, hyp_frames)

    missed = 0
    false_alarm = 0
    confusion = 0
    correct = 0
    reference_total = 0
    for ref_set, hyp_set in zip(ref_frames, hyp_frames):
        mapped_hyp = {mapping.get(label, f"__unmapped__:{label}") for label in hyp_set}
        frame_correct = len(ref_set & mapped_hyp)
        reference_total += len(ref_set)
        correct += frame_correct
        missed += max(len(ref_set) - len(mapped_hyp), 0)
        false_alarm += max(len(mapped_hyp) - len(ref_set), 0)
        confusion += max(min(len(ref_set), len(mapped_hyp)) - frame_correct, 0)

    denominator_ms = max(reference_total * frame_ms, 1)
    miss_ms = missed * frame_ms
    false_alarm_ms = false_alarm * frame_ms
    confusion_ms = confusion * frame_ms
    der = (miss_ms + false_alarm_ms + confusion_ms) / denominator_ms
    jer = _jer(ref_frames, hyp_frames, mapping)
    false_switches = count_false_switches(reference, hypothesis, mapping, tolerance_ms=switch_tolerance_ms)
    return {
        "frame_ms": frame_ms,
        "scored_start_ms": start_ms,
        "scored_end_ms": end_ms,
        "reference_speaker_ms": reference_total * frame_ms,
        "correct_speaker_ms": correct * frame_ms,
        "miss_ms": miss_ms,
        "false_alarm_ms": false_alarm_ms,
        "confusion_ms": confusion_ms,
        "der": round(der, 6),
        "jer": round(jer, 6),
        "false_switches": false_switches,
        "reference_speakers": sorted({item.speaker for item in reference}),
        "hypothesis_speakers": sorted({item.speaker for item in hypothesis}),
        "speaker_mapping": mapping,
    }


def evaluate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("speaker benchmark manifest must be a list or contain an items list")
    frame_ms = int(payload.get("frame_ms", 20)) if isinstance(payload, dict) else 20
    tolerance_ms = int(payload.get("switch_tolerance_ms", 500)) if isinstance(payload, dict) else 500
    per_item: list[dict[str, Any]] = []
    totals: dict[str, dict[str, float]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item_id = str(entry.get("id") or f"item_{len(per_item) + 1:03d}")
        reference_path = _resolve_path(manifest_path.parent, entry.get("reference"))
        reference = load_regions(reference_path)
        hypotheses = entry.get("hypotheses") or {}
        if not isinstance(hypotheses, dict):
            continue
        for engine, value in hypotheses.items():
            hypothesis_path = _resolve_path(manifest_path.parent, value)
            metrics = evaluate_regions(reference, load_regions(hypothesis_path), frame_ms=frame_ms, switch_tolerance_ms=tolerance_ms)
            per_item.append({"id": item_id, "engine": str(engine), "reference": str(reference_path), "hypothesis": str(hypothesis_path), **metrics})
            total = totals.setdefault(str(engine), {"reference_speaker_ms": 0, "miss_ms": 0, "false_alarm_ms": 0, "confusion_ms": 0, "jer_sum": 0, "items": 0, "false_switches": 0})
            for key in ("reference_speaker_ms", "miss_ms", "false_alarm_ms", "confusion_ms", "false_switches"):
                total[key] += float(metrics[key])
            total["jer_sum"] += float(metrics["jer"])
            total["items"] += 1

    aggregate: dict[str, Any] = {}
    for engine, total in totals.items():
        denominator = max(total["reference_speaker_ms"], 1)
        aggregate[engine] = {
            "items": int(total["items"]),
            "reference_speaker_ms": int(total["reference_speaker_ms"]),
            "miss_ms": int(total["miss_ms"]),
            "false_alarm_ms": int(total["false_alarm_ms"]),
            "confusion_ms": int(total["confusion_ms"]),
            "der": round((total["miss_ms"] + total["false_alarm_ms"] + total["confusion_ms"]) / denominator, 6),
            "jer": round(total["jer_sum"] / max(total["items"], 1), 6),
            "false_switches": int(total["false_switches"]),
        }
    return {"schema_version": 1, "frame_ms": frame_ms, "switch_tolerance_ms": tolerance_ms, "aggregate": aggregate, "items": per_item}


def optimal_speaker_mapping(reference_frames: list[set[str]], hypothesis_frames: list[set[str]]) -> dict[str, str]:
    reference_labels = sorted(set().union(*reference_frames) if reference_frames else set())
    hypothesis_labels = sorted(set().union(*hypothesis_frames) if hypothesis_frames else set())
    if not reference_labels or not hypothesis_labels:
        return {}
    overlap: dict[tuple[str, str], int] = {}
    for ref_set, hyp_set in zip(reference_frames, hypothesis_frames):
        for hypothesis_label in hyp_set:
            for reference_label in ref_set:
                overlap[(hypothesis_label, reference_label)] = overlap.get((hypothesis_label, reference_label), 0) + 1

    # Podcast and meeting jobs normally have fewer than eight speakers.  Use
    # exact assignment there; retain a deterministic greedy fallback for an
    # unexpectedly large panel rather than introducing a SciPy dependency.
    if max(len(reference_labels), len(hypothesis_labels)) <= 8:
        padded_refs: list[str | None] = reference_labels + [None] * max(0, len(hypothesis_labels) - len(reference_labels))
        best_score = -1
        best: tuple[str | None, ...] = tuple()
        for permutation in itertools.permutations(padded_refs, len(hypothesis_labels)):
            score = sum(overlap.get((hypothesis_label, reference_label), 0) for hypothesis_label, reference_label in zip(hypothesis_labels, permutation) if reference_label)
            if score > best_score:
                best_score = score
                best = permutation
        return {hypothesis_label: reference_label for hypothesis_label, reference_label in zip(hypothesis_labels, best) if reference_label}

    mapping: dict[str, str] = {}
    available = set(reference_labels)
    for hypothesis_label in hypothesis_labels:
        chosen = max(available, key=lambda ref: overlap.get((hypothesis_label, ref), 0), default="")
        if chosen:
            mapping[hypothesis_label] = chosen
            available.remove(chosen)
    return mapping


def count_false_switches(
    reference: list[SpeakerRegion],
    hypothesis: list[SpeakerRegion],
    mapping: dict[str, str],
    *,
    tolerance_ms: int,
) -> int:
    reference_switches = _switch_times(reference, mapping=None)
    hypothesis_switches = _switch_times(hypothesis, mapping=mapping)
    return sum(1 for value in hypothesis_switches if not any(abs(value - expected) <= tolerance_ms for expected in reference_switches))


def _switch_times(regions: list[SpeakerRegion], mapping: dict[str, str] | None) -> list[int]:
    switches: list[int] = []
    previous = ""
    for region in sorted(regions, key=lambda item: (item.start_ms, item.end_ms)):
        speaker = mapping.get(region.speaker, region.speaker) if mapping is not None else region.speaker
        if previous and speaker != previous:
            switches.append(region.start_ms)
        previous = speaker
    return switches


def _jer(reference_frames: list[set[str]], hypothesis_frames: list[set[str]], mapping: dict[str, str]) -> float:
    reference_labels = sorted(set().union(*reference_frames) if reference_frames else set())
    if not reference_labels:
        return 0.0
    errors: list[float] = []
    for label in reference_labels:
        intersection = 0
        union = 0
        for ref_set, hyp_set in zip(reference_frames, hypothesis_frames):
            mapped_hyp = {mapping.get(item, f"__unmapped__:{item}") for item in hyp_set}
            in_ref = label in ref_set
            in_hyp = label in mapped_hyp
            intersection += int(in_ref and in_hyp)
            union += int(in_ref or in_hyp)
        errors.append(1.0 - intersection / max(union, 1))
    return sum(errors) / len(errors)


def _frame_sets(regions: list[SpeakerRegion], start_ms: int, end_ms: int, frame_ms: int) -> list[set[str]]:
    frames: list[set[str]] = []
    for frame_start in range(start_ms, end_ms, frame_ms):
        frame_end = min(frame_start + frame_ms, end_ms)
        frames.append({region.speaker for region in regions if region.start_ms < frame_end and region.end_ms > frame_start})
    return frames


def _time_ms(row: dict[str, Any], milliseconds_key: str, seconds_key: str) -> int:
    if row.get(milliseconds_key) is not None:
        return int(round(float(row[milliseconds_key])))
    return int(round(float(row.get(seconds_key) or 0) * 1000))


def _resolve_path(base: Path, value: Any) -> Path:
    if not value:
        raise ValueError("benchmark path is missing")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
