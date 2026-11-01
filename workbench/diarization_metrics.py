"""Continuous-time diarization scoring, independent of model runtimes.

Protocol continuous-ms-v2 uses explicit UEM, recording-wide optimal DER
assignment and a separate maximum-IoU assignment for JER (dscore convention).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

from .speaker_benchmark import SpeakerRegion, _maximum_weight_pairs


def evaluate_continuous_regions(
    reference: list[SpeakerRegion],
    hypothesis: list[SpeakerRegion],
    *,
    uem_ms: list[list[int]] | list[tuple[int, int]],
    collar_ms: int = 0,
    skip_overlap: bool = False,
    allow_fractional_ms: bool = False,
) -> dict[str, Any]:
    """Score half-open millisecond intervals; collar is total centered width.

    Duplicate/overlapping tracks of the same label count as one active speaker.
    All labels share one mapping over the complete supplied UEM, including gaps.
    Zero-reference DER is null, flagged as undefined for ratio comparisons;
    false-alarm durations remain available for aggregation.
    """
    if type(collar_ms) is not int or collar_ms < 0 or type(skip_overlap) is not bool or type(allow_fractional_ms) is not bool:
        raise ValueError("collar_ms must be a nonnegative integer and skip_overlap a boolean")
    def valid_time(value):
        return ((type(value) in (int, float) and math.isfinite(value)) if allow_fractional_ms else type(value) is int)
    if not isinstance(uem_ms, (list, tuple)) or not uem_ms:
        raise ValueError("continuous scoring requires a nonempty explicit uem_ms")
    uem: list[tuple[int, int]] = []
    for interval in uem_ms:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("each UEM interval must be [start_ms, end_ms]")
        start, end = interval
        if not valid_time(start) or not valid_time(end) or not 0 <= start < end:
            raise ValueError("UEM boundaries must be nonnegative integer milliseconds with start < end")
        uem.append((start, end))
    merged_uem: list[list[int]] = []
    for start, end in sorted(uem):
        if merged_uem and start <= merged_uem[-1][1]:
            merged_uem[-1][1] = max(end, merged_uem[-1][1])
        else:
            merged_uem.append([start, end])

    events: dict[float, list[tuple[str, str, int]]] = defaultdict(list)

    def interval_event(start: float, end: float, kind: str, label: str = "") -> None:
        events[start].append((kind, label, 1))
        events[end].append((kind, label, -1))

    for kind, regions in (("ref", reference), ("hyp", hypothesis)):
        for region in regions:
            if (not valid_time(region.start_ms) or not valid_time(region.end_ms)
                    or not 0 <= region.start_ms < region.end_ms
                    or not isinstance(region.speaker, str) or not region.speaker.strip()):
                raise ValueError("speaker regions require valid integer millisecond intervals and nonempty labels")
            interval_event(region.start_ms, region.end_ms, kind, region.speaker)
            if kind == "ref" and collar_ms:
                half = collar_ms / 2
                for boundary in (region.start_ms, region.end_ms):
                    interval_event(boundary - half, boundary + half, "collar")
    for start, end in merged_uem:
        interval_event(start, end, "uem")

    active: dict[str, Counter[str]] = {kind: Counter() for kind in ("ref", "hyp", "uem", "collar")}
    cells: list[tuple[float, set[str], set[str]]] = []
    mapping_intersection: Counter[tuple[str, str]] = Counter()
    mapping_refs: set[str] = set()
    mapping_hyps: set[str] = set()
    scored_uem: list[list[float]] = []
    boundaries = sorted(events)
    for start, end in zip(boundaries, boundaries[1:]):
        for kind, label, delta in events[start]:
            active[kind][label] += delta
            if active[kind][label] == 0:
                del active[kind][label]
        ref_set, hyp_set = set(active["ref"]), set(active["hyp"])
        if not active["uem"]:
            continue
        # md-eval-22 assigns on the original UEM BEFORE removing collars or
        # overlap. pyannote DER instead assigns after exclusions; specify ours.
        mapping_refs.update(ref_set)
        mapping_hyps.update(hyp_set)
        for hyp in hyp_set:
            for ref in ref_set:
                mapping_intersection[hyp, ref] += end - start
        if active["collar"] or (skip_overlap and len(ref_set) > 1):
            continue
        cells.append((end - start, ref_set, hyp_set))
        if scored_uem and scored_uem[-1][1] == start:
            scored_uem[-1][1] = end
        else:
            scored_uem.append([start, end])
    if not cells:
        raise ValueError("no scored time remains after UEM/collar/overlap exclusions")

    ref_durations: Counter[str] = Counter()
    hyp_durations: Counter[str] = Counter()
    intersection: Counter[tuple[str, str]] = Counter()
    for duration, ref_set, hyp_set in cells:
        for ref in ref_set:
            ref_durations[ref] += duration
        for hyp in hyp_set:
            hyp_durations[hyp] += duration
            for ref in ref_set:
                intersection[hyp, ref] += duration
    refs, hyps = sorted(ref_durations), sorted(hyp_durations)
    der_refs, der_hyps = sorted(mapping_refs), sorted(mapping_hyps)
    cooccurrence = [[mapping_intersection[hyp, ref] for ref in der_refs] for hyp in der_hyps]
    mapping = {
        der_hyps[hyp]: der_refs[ref] for hyp, ref in _maximum_weight_pairs(cooccurrence)
        if cooccurrence[hyp][ref] > 0
    }
    iou = [
        [intersection[hyp, ref] / (hyp_durations[hyp] + ref_durations[ref] - intersection[hyp, ref]) for ref in refs]
        for hyp in hyps
    ]
    jer_pairs = _maximum_weight_pairs(iou)
    jer_mapping = {hyps[hyp]: refs[ref] for hyp, ref in jer_pairs if iou[hyp][ref] > 0}
    jer_error_sum = max(0.0, len(refs) - sum(iou[hyp][ref] for hyp, ref in jer_pairs))
    missed = false_alarm = confusion = correct = 0.0
    for duration, ref_set, hyp_set in cells:
        matches = sum(mapping.get(hyp) in ref_set for hyp in hyp_set)
        correct += matches * duration
        missed += max(len(ref_set) - len(hyp_set), 0) * duration
        false_alarm += max(len(hyp_set) - len(ref_set), 0) * duration
        confusion += (min(len(ref_set), len(hyp_set)) - matches) * duration
    total = sum(ref_durations.values())
    errors = missed + false_alarm + confusion
    return {
        "metric_protocol": "continuous-float-ms-v3" if allow_fractional_ms else "continuous-ms-v2",
        "time_unit": "ms",
        "uem_ms": merged_uem,
        "scored_uem_ms": scored_uem,
        "scored_duration_ms": sum(end - start for start, end in scored_uem),
        "collar_ms": collar_ms,
        "collar_semantics": "total_width_centered_on_reference_boundaries",
        "skip_overlap": skip_overlap,
        "mapping_scope": "recording",
        "der_mapping_domain": "uem_before_collar_and_overlap_exclusions",
        "jer_mapping_domain": "scored_uem_after_exclusions",
        "jer_mapping_objective": "intersection_over_union",
        "reference_speaker_ms": total,
        "correct_speaker_ms": correct,
        "miss_ms": missed,
        "false_alarm_ms": false_alarm,
        "confusion_ms": confusion,
        "der": errors / total if total else None,
        "der_defined": total > 0,
        "empty_reference_policy": "null_der_report_false_alarm_duration",
        "jer": jer_error_sum / len(refs) if refs else float(bool(hyps)),
        "jer_speaker_error_sum": jer_error_sum,
        "jer_speaker_count": len(refs),
        "reference_speakers": refs,
        "hypothesis_speakers": hyps,
        "mapping_reference_speakers": der_refs,
        "mapping_hypothesis_speakers": der_hyps,
        "speaker_count_error": abs(len(refs) - len(hyps)),
        "speaker_mapping": mapping,
        "jer_speaker_mapping": jer_mapping,
    }
