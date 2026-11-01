from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpeakerRegion:
    start_ms: int
    end_ms: int
    speaker: str


def load_regions(path: Path, *, strict: bool = False) -> list[SpeakerRegion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if strict and not any(key in payload for key in ("segments", "speaker_turns", "turns")):
            raise ValueError("speaker annotation has no rows field (use an explicit empty list for silence)")
        rows = next((payload[key] for key in ("segments", "speaker_turns", "turns") if key in payload), [])
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("speaker annotation rows must be a list")
    regions: list[SpeakerRegion] = []
    for row in rows:
        if not isinstance(row, dict):
            if strict:
                raise ValueError("speaker annotation contains a non-object row")
            continue
        if strict and (not any(key in row for key in ("start_ms", "start")) or not any(key in row for key in ("end_ms", "end"))):
            raise ValueError("speaker annotation is missing a start or end boundary")
        if strict:
            for milliseconds, seconds in (("start_ms", "start"), ("end_ms", "end")):
                raw = row.get(milliseconds) if milliseconds in row else row.get(seconds)
                if type(raw) not in (int, float) or not math.isfinite(raw) or (milliseconds in row and raw != int(raw)):
                    raise ValueError("speaker annotation boundaries must be numeric, with integer milliseconds")
        start_ms = _time_ms(row, "start_ms", "start")
        end_ms = _time_ms(row, "end_ms", "end")
        label = (
            row.get("speaker_id")
            or row.get("display_name")
            or row.get("speaker")
            or row.get("raw_speaker_label")
            or ""
        )
        if strict and not isinstance(label, str):
            raise ValueError("speaker annotation labels must be strings")
        speaker = str(label).strip()
        if speaker and end_ms > start_ms >= 0:
            regions.append(SpeakerRegion(start_ms, end_ms, speaker))
        elif strict:
            raise ValueError("speaker annotation contains an invalid interval or empty label")
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
        "metric_protocol": "legacy-frames-v1",
        "comparison_eligible": False,
        "warnings": ["Legacy frame rasterization/reference-extent/JER mapping; use continuous-ms-v2 for model comparisons."],
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
    if isinstance(payload, dict) and payload.get("metric_protocol") == "recording-eval-v3":
        from .speaker_scoring_io import evaluate_recording_manifest
        return evaluate_recording_manifest(manifest_path, payload)
    if isinstance(payload, dict) and payload.get("metric_protocol") == "continuous-ms-v2":
        return _evaluate_continuous_manifest(manifest_path, payload)
    if isinstance(payload, dict) and (payload.get("schema_version", 1) != 1 or payload.get("metric_protocol", "legacy-frames-v1") != "legacy-frames-v1"):
        raise ValueError("unsupported scoring schema/metric_protocol")
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
    return {"schema_version": 1, "metric_protocol": "legacy-frames-v1", "comparison_eligible": False, "warnings": ["Legacy scores are diagnostic only; provide continuous-ms-v2 with explicit UEM for model comparisons."], "frame_ms": frame_ms, "switch_tolerance_ms": tolerance_ms, "aggregate": aggregate, "items": per_item}


def run_speaker_benchmark(*, gold_dir: Path, config=None, backend="auto", json_output=False) -> dict[str, Any]:
    """CLI adapter for frozen-output evaluation, never model execution."""
    source = Path(gold_dir).expanduser().resolve()
    manifest = source / "manifest.json" if source.is_dir() else source
    return {**evaluate_manifest(manifest), "execution": "frozen-output-scoring-only",
        "backend_option": {"requested": backend, "applied": False,
            "reason": "All named hypotheses in the frozen manifest are scored; this command never runs a backend."}}


def _evaluate_continuous_manifest(manifest_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    import hashlib

    from .diarization_metrics import evaluate_continuous_regions

    if payload.get("schema_version") != 2 or "frame_ms" in payload:
        raise ValueError("continuous-ms-v2 requires schema_version 2 and no frame_ms")
    entries = payload.get("items")
    if not isinstance(entries, list) or not entries:
        raise ValueError("continuous manifest requires a nonempty items list")
    collar = payload.get("collar_ms", 0)
    skip_overlap = payload.get("skip_overlap", False)
    items: list[dict[str, Any]] = []
    ids: set[str] = set()
    recordings: set[str] = set()
    audio_hashes: set[str] = set()
    engine_set: set[str] | None = None
    totals: dict[str, dict[str, float]] = {}

    def annotation(value: Any, kind: str, recording_id: str) -> tuple[Path, str, str, list[SpeakerRegion]]:
        if not isinstance(value, dict) or value.get("annotation_kind") != kind:
            raise ValueError(f"annotation must declare annotation_kind={kind}")
        path = _resolve_path(manifest_path.parent, value.get("path"))
        content = path.read_bytes()
        data = json.loads(content)
        if not isinstance(data, dict) or data.get("annotation_kind") != kind:
            raise ValueError(f"annotation file must declare annotation_kind={kind}: {path.name}")
        audio_hash = data.get("source_audio_sha256")
        if data.get("recording_id") != recording_id or not isinstance(audio_hash, str) or len(audio_hash) != 64 or any(char not in "0123456789abcdef" for char in audio_hash):
            raise ValueError("annotation requires matching recording_id and full source_audio_sha256")
        return path, hashlib.sha256(content).hexdigest(), audio_hash, load_regions(path, strict=True)

    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"].strip() or entry["id"] in ids:
            raise ValueError("each scoring item requires a unique nonempty string id")
        ids.add(entry["id"])
        recording_id = entry.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id.strip() or recording_id in recordings:
            raise ValueError("each recording_id must occur once: combine its windows in one UEM")
        recordings.add(recording_id)
        hypotheses = entry.get("hypotheses")
        if not isinstance(hypotheses, dict) or not hypotheses or any(not isinstance(name, str) or not name.strip() for name in hypotheses):
            raise ValueError("each scoring item requires named hypotheses")
        if engine_set is not None and set(hypotheses) != engine_set:
            raise ValueError("all items must evaluate the same engine set")
        engine_set = set(hypotheses)
        ref_path, ref_hash, audio_hash, reference = annotation(entry.get("reference"), "human-diarization", recording_id)
        if audio_hash in audio_hashes:
            raise ValueError("the same source audio must be scored once, with one recording-wide mapping")
        audio_hashes.add(audio_hash)
        for engine, value in hypotheses.items():
            hyp_path, hyp_hash, hyp_audio_hash, hypothesis = annotation(value, "acoustic-diarization", recording_id)
            if hyp_audio_hash != audio_hash:
                raise ValueError("reference and hypothesis must describe the same source audio")
            metrics = evaluate_continuous_regions(reference, hypothesis, uem_ms=entry.get("uem_ms"), collar_ms=collar, skip_overlap=skip_overlap)
            items.append({"id": entry["id"], "recording_id": recording_id, "source_audio_sha256": audio_hash, "engine": engine, "reference": str(ref_path), "reference_sha256": ref_hash, "hypothesis": str(hyp_path), "hypothesis_sha256": hyp_hash, **metrics})
            total = totals.setdefault(engine, {key: 0.0 for key in ("reference_speaker_ms", "correct_speaker_ms", "miss_ms", "false_alarm_ms", "confusion_ms", "jer_speaker_error_sum", "jer_speaker_count", "hypothesis_speaker_count", "items")})
            for key in total.keys() - {"items", "hypothesis_speaker_count"}:
                total[key] += metrics[key]
            total["hypothesis_speaker_count"] += len(metrics["hypothesis_speakers"])
            total["items"] += 1
    aggregate: dict[str, Any] = {}
    for engine, total in totals.items():
        denominator = total["reference_speaker_ms"]
        errors = total["miss_ms"] + total["false_alarm_ms"] + total["confusion_ms"]
        speakers = total["jer_speaker_count"]
        aggregate[engine] = {
            **total,
            "der": errors / denominator if denominator else None,
            "der_defined": denominator > 0,
            "jer": total["jer_speaker_error_sum"] / speakers if speakers else float(total["hypothesis_speaker_count"] > 0),
        }
    return {"schema_version": 2, "metric_protocol": "continuous-ms-v2", "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "collar_ms": collar, "skip_overlap": skip_overlap, "mapping_scope": "recording", "aggregate": aggregate, "items": items}


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

    weights = [[overlap.get((hyp, ref), 0) for ref in reference_labels] for hyp in hypothesis_labels]
    return {
        hypothesis_labels[hyp]: reference_labels[ref]
        for hyp, ref in _maximum_weight_pairs(weights)
        if weights[hyp][ref] > 0
    }


def _maximum_weight_pairs(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Exact rectangular Hungarian assignment, without a model/SciPy dependency.

    Keep the smaller dimension as rows: O(min(n, m)^2 * max(n, m)).
    Sorted speaker labels and ascending scans make tied optima deterministic.
    Zero-weight pairs may be returned; callers decide whether to retain them.
    """
    if not weights or not weights[0]:
        return []
    transposed = len(weights) > len(weights[0])
    costs = list(map(list, zip(*weights))) if transposed else weights
    rows, columns = len(costs), len(costs[0])
    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    matched_row = [0] * (columns + 1)
    predecessor = [0] * (columns + 1)
    for row in range(1, rows + 1):
        matched_row[0] = row
        column = 0
        distance = [float("inf")] * (columns + 1)
        visited = [False] * (columns + 1)
        while True:
            visited[column] = True
            active_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, columns + 1):
                if visited[candidate]:
                    continue
                reduced = -costs[active_row - 1][candidate - 1] - row_potential[active_row] - column_potential[candidate]
                if reduced < distance[candidate]:
                    distance[candidate] = reduced
                    predecessor[candidate] = column
                if distance[candidate] < delta:
                    delta = distance[candidate]
                    next_column = candidate
            for candidate in range(columns + 1):
                if visited[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while column:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
    pairs = [(matched_row[column] - 1, column - 1) for column in range(1, columns + 1) if matched_row[column]]
    return sorted((column, row) if transposed else (row, column) for row, column in pairs)


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
