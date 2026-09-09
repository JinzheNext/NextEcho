"""Cluster-to-person identity mapping for speaker transcripts.

Diarization only knows anonymous acoustic labels.  This module is the narrow
identity layer that may attach a display name using explicit evidence.  In
particular, a self-introduction is attached to the cluster active at that
timestamp; it is never a mutable global anchor that changes later turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .diarization import DiarizationResult, DiarizationTurn, coerce_turns


UNKNOWN_DISPLAY_NAME = "unknown"
DEFAULT_MINIMUM_CONFIDENCE = 0.90


@dataclass(frozen=True)
class IdentityEvidence:
    type: str
    source: str = ""
    segment_ids: tuple[str, ...] = ()
    raw_speaker_label: str = ""
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "source": self.source,
            "segment_ids": list(self.segment_ids),
            "confidence": round(max(0.0, min(1.0, self.confidence)), 4),
        }
        if self.raw_speaker_label:
            payload["raw_speaker_label"] = self.raw_speaker_label
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class SpeakerIdentity:
    speaker_id: str
    raw_labels: list[str]
    display_name: str = UNKNOWN_DISPLAY_NAME
    confidence: float = 0.0
    identity_status: str = "unknown"
    identity_evidence: list[IdentityEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "raw_labels": list(self.raw_labels),
            "display_name": self.display_name or UNKNOWN_DISPLAY_NAME,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 4),
            "identity_status": self.identity_status,
            "identity_evidence": [item.to_dict() for item in self.identity_evidence],
        }


@dataclass
class SpeakerIdentityMap:
    speakers: list[SpeakerIdentity]
    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE

    def by_raw_label(self) -> dict[str, SpeakerIdentity]:
        return {
            raw_label: speaker
            for speaker in self.speakers
            for raw_label in speaker.raw_labels
        }

    def get(self, raw_label: str) -> SpeakerIdentity | None:
        return self.by_raw_label().get(raw_label)

    @property
    def raw_to_speaker_id(self) -> dict[str, str]:
        return {
            raw_label: speaker.speaker_id
            for speaker in self.speakers
            for raw_label in speaker.raw_labels
        }

    @property
    def raw_to_display_name(self) -> dict[str, str]:
        return {
            raw_label: speaker.display_name
            for speaker in self.speakers
            for raw_label in speaker.raw_labels
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "minimum_confidence": self.minimum_confidence,
            "speakers": [item.to_dict() for item in self.speakers],
        }


class SpeakerIdentityResolver(Protocol):
    """Resolve anonymous diarization clusters without changing raw turns."""

    def resolve(
        self,
        diarization: DiarizationResult | Sequence[Any],
        transcript: Sequence[Any],
        context: dict[str, Any] | None = None,
        profiles: Sequence[dict[str, Any]] | None = None,
    ) -> SpeakerIdentityMap:
        ...


class DefaultSpeakerIdentityResolver:
    """Evidence-first resolver used by the local speaker transcript path."""

    def __init__(self, *, minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE) -> None:
        self.minimum_confidence = max(0.0, min(1.0, float(minimum_confidence)))

    def resolve(
        self,
        diarization: DiarizationResult | Sequence[Any],
        transcript: Sequence[Any],
        context: dict[str, Any] | None = None,
        profiles: Sequence[dict[str, Any]] | None = None,
    ) -> SpeakerIdentityMap:
        turns = coerce_turns(diarization.turns if isinstance(diarization, DiarizationResult) else diarization)
        segments = list(transcript or [])
        labels = _ordered_labels(turns)
        evidence_by_label: dict[str, list[IdentityEvidence]] = {label: [] for label in labels}
        candidate_names = _candidate_names(context or {})
        candidate_items = _candidate_items(context or {})

        for index, segment in enumerate(segments):
            text = _segment_text(segment)
            if not text:
                continue
            segment_id = _segment_id(segment, index)
            for name in candidate_names:
                match = detect_speaker_self_introduction(text, [name])
                if not match:
                    continue
                label = _anchor_label(segment, name, turns)
                if not label:
                    continue
                evidence_by_label.setdefault(label, []).append(
                    IdentityEvidence(
                        type="self_introduction",
                        source="transcript",
                        segment_ids=(segment_id,),
                        raw_speaker_label=label,
                        confidence=0.98,
                        details={"name": name, "text": text},
                    )
                )
                candidate_item = next(
                    (item for item in candidate_items if item[0] == name and item[1]),
                    (name, ""),
                )
                if candidate_item[1]:
                    evidence_by_label[label].append(
                        IdentityEvidence(
                            type="metadata",
                            source=candidate_item[1],
                            segment_ids=(segment_id,),
                            raw_speaker_label=label,
                            confidence=0.65,
                            details={"name": name},
                        )
                    )
                # A segment should contribute at most one identity name.  A
                # second matching candidate is treated as a separate conflict
                # only if the text genuinely contains both names.
                break

        # Explicit local enrollment mappings can be supplied by a future
        # embedding backend.  We only consume an already verified profile
        # match; this resolver never invents a match from a file name.
        for profile in profiles or []:
            if not isinstance(profile, dict):
                continue
            label = str(profile.get("raw_speaker_label") or profile.get("raw_label") or "").strip()
            name = str(profile.get("display_name") or profile.get("name") or "").strip()
            if not label or not name or label not in evidence_by_label:
                continue
            confidence = _float(profile.get("confidence"), 0.0)
            if confidence <= 0:
                confidence = _float(profile.get("threshold"), 0.0)
            if confidence <= 0:
                continue
            evidence_by_label[label].append(
                IdentityEvidence(
                    type="voice_profile",
                    source=str(profile.get("person_id") or profile.get("source") or "local"),
                    raw_speaker_label=label,
                    confidence=confidence,
                    details={"name": name},
                )
            )
            # Keep the profile's name in a private detail for the reducer.

        identities: list[SpeakerIdentity] = []
        for index, label in enumerate(labels):
            evidences = evidence_by_label.get(label, [])
            chosen_name, confidence, status = _reduce_evidence(evidences, self.minimum_confidence)
            identities.append(
                SpeakerIdentity(
                    speaker_id=f"spk_{index:02d}",
                    raw_labels=[label],
                    display_name=chosen_name or UNKNOWN_DISPLAY_NAME,
                    confidence=confidence,
                    identity_status=status,
                    identity_evidence=evidences,
                )
            )
        return SpeakerIdentityMap(identities, minimum_confidence=self.minimum_confidence)


def resolve_speaker_identities(
    diarization: DiarizationResult | Sequence[Any],
    transcript: Sequence[Any],
    *,
    context: dict[str, Any] | None = None,
    profiles: Sequence[dict[str, Any]] | None = None,
    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
) -> SpeakerIdentityMap:
    return DefaultSpeakerIdentityResolver(minimum_confidence=minimum_confidence).resolve(
        diarization,
        transcript,
        context=context,
        profiles=profiles,
    )


def detect_speaker_self_introduction(text: str, candidates: Sequence[str]) -> str:
    """Return a candidate explicitly named by the current text only."""

    normalized_text = str(text or "")
    for name in candidates:
        clean_name = str(name or "").strip()
        if not clean_name:
            continue
        if re.search(rf"(?:大家好[，, ]*)?(?:我|本人)?\s*(?:是|叫)\s*{re.escape(clean_name)}", normalized_text):
            return clean_name
    return ""


def _reduce_evidence(evidences: list[IdentityEvidence], minimum: float) -> tuple[str, float, str]:
    if not evidences:
        return UNKNOWN_DISPLAY_NAME, 0.0, "unknown"
    names: dict[str, list[IdentityEvidence]] = {}
    for evidence in evidences:
        name = str(evidence.details.get("name") or "").strip()
        if name:
            names.setdefault(name, []).append(evidence)
    if not names:
        return UNKNOWN_DISPLAY_NAME, 0.0, "unknown"
    ranked = sorted(
        names.items(),
        key=lambda item: (
            max(evidence.confidence for evidence in item[1]),
            sum(evidence.confidence for evidence in item[1]),
        ),
        reverse=True,
    )
    name, selected = ranked[0]
    confidence = max(evidence.confidence for evidence in selected)
    conflicting = any(other_name != name for other_name, _ in ranked[1:])
    if conflicting:
        return UNKNOWN_DISPLAY_NAME, min(confidence, 0.49), "conflict"
    if confidence < minimum:
        return UNKNOWN_DISPLAY_NAME, confidence, "hinted"
    return name, confidence, "resolved"


def _ordered_labels(turns: Sequence[DiarizationTurn]) -> list[str]:
    labels: list[str] = []
    for turn in sorted(turns, key=lambda item: (item.start, item.end)):
        label = turn.raw_label.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _candidate_names(context: dict[str, Any]) -> list[str]:
    return [name for name, _ in _candidate_items(context)]


def _candidate_items(context: dict[str, Any]) -> list[tuple[str, str]]:
    names: list[str] = []
    items: list[tuple[str, str]] = []
    candidates = context.get("speaker_candidates") or []
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    for item in candidates:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            source = str(item.get("source") or item.get("role") or "context").strip()
        else:
            name = str(item or "").strip()
            source = "context"
        if name and name not in names:
            names.append(name)
            items.append((name, source))
    return items


def _segment_text(segment: Any) -> str:
    if isinstance(segment, dict):
        return str(segment.get("text") or "").strip()
    return str(getattr(segment, "text", "") or "").strip()


def _segment_id(segment: Any, index: int) -> str:
    if isinstance(segment, dict):
        return str(segment.get("id") or segment.get("segment_id") or f"seg_{index:05d}")
    return str(getattr(segment, "id", "") or getattr(segment, "segment_id", "") or f"seg_{index:05d}")


def _segment_range(segment: Any) -> tuple[float, float]:
    if isinstance(segment, dict):
        start = segment.get("start")
        end = segment.get("end")
        if start is None:
            start = float(segment.get("start_ms") or 0) / 1000
        if end is None:
            end = float(segment.get("end_ms") or 0) / 1000
        return float(start or 0), float(end or 0)
    return float(getattr(segment, "start", 0.0) or 0.0), float(getattr(segment, "end", 0.0) or 0.0)


def _segment_words(segment: Any) -> list[Any]:
    if isinstance(segment, dict):
        value = segment.get("words")
    else:
        value = getattr(segment, "words", None)
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _word_range(word: Any) -> tuple[float, float]:
    if isinstance(word, dict):
        start_key = "start" if "start" in word else ("from" if "from" in word else "start_ms")
        end_key = "end" if "end" in word else ("to" if "to" in word else "end_ms")
        start = word.get(start_key)
        end = word.get(end_key)
        if start is None:
            start = float(word.get("start_ms") or 0) / 1000
        if end is None:
            end = float(word.get("end_ms") or 0) / 1000
        return _time_value(start, assume_milliseconds=start_key in {"from", "start_ms"}), _time_value(
            end,
            assume_milliseconds=end_key in {"to", "end_ms"},
        )
    return float(getattr(word, "start", 0.0) or 0.0), float(getattr(word, "end", 0.0) or 0.0)


def _word_text(word: Any) -> str:
    if isinstance(word, dict):
        return str(word.get("text") or word.get("word") or "")
    return str(getattr(word, "text", "") or getattr(word, "word", "") or "")


def _anchor_label(segment: Any, name: str, turns: Sequence[DiarizationTurn]) -> str:
    start, end = _segment_range(segment)
    if end <= start:
        return ""
    words = _segment_words(segment)
    if words:
        for word in words:
            if name in _word_text(word):
                word_start, word_end = _word_range(word)
                label = _label_for_interval(word_start, word_end, turns)
                if label:
                    return label
    text = _segment_text(segment)
    position = text.find(name)
    ratio = (position + max(1, len(name)) / 2) / max(1, len(text)) if position >= 0 else 0.5
    point = start + (end - start) * ratio
    containing = [turn for turn in turns if turn.start <= point < turn.end]
    if containing:
        containing.sort(key=lambda turn: (turn.overlap, turn.confidence or 0.0), reverse=True)
        return containing[0].raw_label
    return _label_for_interval(start, end, turns)


def _label_for_interval(start: float, end: float, turns: Sequence[DiarizationTurn]) -> str:
    overlap_rows: list[tuple[float, float, str]] = []
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > 0:
            overlap_rows.append((overlap, turn.confidence or 0.0, turn.raw_label))
    if not overlap_rows:
        return ""
    overlap_rows.sort(reverse=True)
    return overlap_rows[0][2]


def _time_value(value: Any, *, assume_milliseconds: bool = False) -> float:
    if isinstance(value, (int, float)):
        return float(value) / 1000 if assume_milliseconds else float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        result = float(text.replace(",", "."))
        return result / 1000 if assume_milliseconds else result
    except ValueError:
        parts = text.replace(",", ".").split(":")
        if len(parts) == 3:
            result = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return result / 1000 if assume_milliseconds else result
        return 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "DEFAULT_MINIMUM_CONFIDENCE",
    "DefaultSpeakerIdentityResolver",
    "IdentityEvidence",
    "SpeakerIdentity",
    "SpeakerIdentityMap",
    "SpeakerIdentityResolver",
    "UNKNOWN_DISPLAY_NAME",
    "detect_speaker_self_introduction",
    "resolve_speaker_identities",
]
