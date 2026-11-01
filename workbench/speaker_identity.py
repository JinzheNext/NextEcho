"""Cluster-to-person identity mapping for speaker transcripts.

Diarization only knows anonymous acoustic labels.  This module is the narrow
identity layer that may attach a display name using explicit evidence.  In
particular, a self-introduction is attached to the cluster active at that
timestamp; it is never a mutable global anchor that changes later turns.
"""

from __future__ import annotations

import re
import math
import difflib
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
            "confidence": round(_finite_unit(self.confidence), 4),
        }
        if self.raw_speaker_label:
            payload["raw_speaker_label"] = self.raw_speaker_label
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True)
class VerifiedVoiceMatch:
    """A locally measured, content-bound voice match.

    Plain profile dictionaries are intentionally not accepted as identity
    evidence. A match must carry the actual finite similarity score, its
    calibrated threshold, and hashes tying it to the recording, profile
    embedding, model and calibration report.
    """

    raw_speaker_label: str
    display_name: str
    person_id: str
    score: float
    threshold: float
    source_audio_sha256: str
    profile_embedding_sha256: str
    model_id: str
    calibration_report_sha256: str
    recording_scope_id: str

    def __post_init__(self) -> None:
        score, threshold = float(self.score), float(self.threshold)
        if not (math.isfinite(score) and -1.0 <= score <= 1.0):
            raise ValueError("voice match score must be a finite cosine in [-1, 1]")
        if not (math.isfinite(threshold) and -1.0 <= threshold <= 1.0):
            raise ValueError("voice match threshold must be finite and in [-1, 1]")
        if score < threshold:
            raise ValueError("voice match score is below its calibrated threshold")
        required = (
            self.raw_speaker_label, self.display_name, self.person_id,
            self.model_id, self.recording_scope_id,
        )
        if any(not str(value).strip() for value in required):
            raise ValueError("voice match is missing its label, name, model or recording scope")
        for name, value in (
            ("source audio", self.source_audio_sha256),
            ("profile embedding", self.profile_embedding_sha256),
            ("calibration report", self.calibration_report_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(value or "")):
                raise ValueError(f"voice match {name} SHA-256 is invalid")


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
            "confidence": round(_finite_unit(self.confidence), 4),
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
            "resolver_version": "acoustic-identity-v3-full-declaration",
            "confidence_kind": "rule_evidence_score_not_probability",
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
        self.minimum_confidence = _finite_unit(minimum_confidence)

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
                        details={"name": name, "text": text,
                                 "declaration_text_span": _self_introduction_span(text, name, full=True),
                                 "timing_evidence": "native_token_intervals_or_onsets" if _segment_words(segment) else "single_voice_segment"},
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
                # Keep conflicting direct introductions as conflict evidence.

        # Explicit local enrollment mappings can be supplied by a future
        # embedding backend.  We only consume an already verified profile
        # match; this resolver never invents a match from a file name.
        for profile in profiles or []:
            if not isinstance(profile, VerifiedVoiceMatch):
                continue
            label = profile.raw_speaker_label
            name = profile.display_name
            if not label or not name or label not in evidence_by_label:
                continue
            evidence_by_label[label].append(
                IdentityEvidence(
                    type="verified_voice_match",
                    source=profile.person_id,
                    raw_speaker_label=label,
                    confidence=profile.score,
                    details={
                        "name": name,
                        "score": profile.score,
                        "threshold": profile.threshold,
                        "source_audio_sha256": profile.source_audio_sha256,
                        "profile_embedding_sha256": profile.profile_embedding_sha256,
                        "model_id": profile.model_id,
                        "calibration_report_sha256": profile.calibration_report_sha256,
                        "recording_scope_id": profile.recording_scope_id,
                    },
                )
            )
            # Keep the profile's name in a private detail for the reducer.

        # A two-name title and two acoustic clusters do not identify the host:
        # quoted audio, additional voices and clustering errors remain possible.

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
    """Recognize direct first-person evidence, never mentions or quotations."""
    for name in candidates:
        clean_name = str(name or "").strip()
        if clean_name and _self_introduction_span(str(text or ""), clean_name):
            return clean_name
    return ""


def _self_introduction_span(text: str, name: str, *, full: bool = False) -> tuple[int, int] | None:
    # Preserve positions while removing quoted examples from consideration.
    unquoted = re.sub(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"',
                       lambda m: " " * len(m.group()), text)
    pattern = (r"(?:^|[，,。！？!?；;：:\n]|大家好[，,\s]*|各位好[，,\s]*)"
               r"\s*(?P<declaration>(?:我(?:的名字)?|本人)\s*(?:是|叫)\s*"
               rf"(?P<name>{re.escape(name)}))(?=$|[\s，,。！？!?；;：:、])")
    for match in re.finditer(pattern, unquoted):
        prefix = re.split(r"[。！？!?；;\n]", text[:match.start("name")])[-1]
        if re.search(r"他说|她说|你说|说过|引用|比如|例如|假设|模仿|复述|介绍说|讲过|(?:说|讲|喊|问)\s*[：:]|[“「『\"]", prefix):
            continue
        return match.span("declaration" if full else "name")
    return None


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


def _candidate_records(context: dict[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    candidates = context.get("speaker_candidates") or []
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        records.append(
            {
                "name": name,
                "role": str(item.get("role") or "").strip().lower(),
                "source": str(item.get("source") or item.get("role") or "context").strip(),
            }
        )
    return records


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
        start_key = next((k for k in ("start", "from", "start_ms") if word.get(k) is not None), "start")
        end_key = next((k for k in ("end", "to", "end_ms") if word.get(k) is not None), "end")
        start = word.get(start_key)
        end = word.get(end_key)
        return _time_value(start, assume_milliseconds=start_key in {"from", "start_ms"}), (_time_value(
            end,
            assume_milliseconds=end_key in {"to", "end_ms"},
        ) if end is not None else 0.0)
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
        text = _segment_text(segment)
        span = _self_introduction_span(text, name, full=True)
        if span is None:
            return ""
        source_positions = [i for i, c in enumerate(text) if not c.isspace()]
        reference, word_indices = [], []
        for index, word in enumerate(words):
            for char in _word_text(word):
                if not char.isspace():
                    reference.append(char.casefold())
                    word_indices.append(index)
        matcher = difflib.SequenceMatcher(None, [text[i].casefold() for i in source_positions], reference, autojunk=False)
        anchors = {}
        for a, b, size in matcher.get_matching_blocks():
            for offset in range(size):
                anchors[source_positions[a + offset]] = word_indices[b + offset]
        required = [i for i in range(*span) if not text[i].isspace()]
        # Partial words cannot authorize a declaration whose "I am" is absent.
        if not required or any(i not in anchors for i in required):
            return ""
        matched, previous = [], -math.inf
        for index in dict.fromkeys(anchors[i] for i in required):
            word = words[index]
            if isinstance(word, dict) and not any(word.get(k) is not None for k in ("start", "from", "start_ms")):
                return ""
            try:
                a, b = _word_range(word)
            except (TypeError, ValueError):
                return ""
            # Missing end is a native point. An explicit interval must have
            # positive measured duration; a quantized zero-width interval is
            # not enough evidence to bind a person name.
            explicit_end = (any(word.get(k) is not None for k in ("end", "to", "end_ms"))
                            if isinstance(word, dict) else getattr(word, "end", None) is not None)
            if (not math.isfinite(a) or not math.isfinite(b) or not start <= a < end or a < previous or
                    (explicit_end and (b <= a or b > end))):
                return ""
            matched.append((a, b if explicit_end else a))
            previous = a
        point_labels = {_label_at_point(a, turns) for a, _ in matched}
        if "" in point_labels or len(point_labels) != 1:
            return ""
        left = min(a for a, _ in matched)
        right = max(b for _, b in matched)
        return _label_for_interval(left, right, turns) if right > left else next(iter(point_labels))
    # A coarse segment can only identify a voice when the entire interval is
    # acoustically unambiguous. Text position contains no timing evidence.
    return _label_for_interval(start, end, turns)


def _label_at_point(point: float, turns: Sequence[DiarizationTurn]) -> str:
    if not math.isfinite(point):
        return ""
    active = [t for t in turns if t.start <= point < t.end]
    labels = {t.raw_label for t in active}
    if len(labels) != 1 or any(t.overlap for t in active):
        return ""
    return next(iter(labels))


def _label_for_interval(start: float, end: float, turns: Sequence[DiarizationTurn]) -> str:
    if not all(math.isfinite(v) for v in (start, end)) or end <= start:
        return ""
    spans, labels = [], set()
    for turn in turns:
        left, right = max(start, turn.start), min(end, turn.end)
        if right > left:
            if turn.overlap:
                return ""
            spans.append((left, right))
            labels.add(turn.raw_label)
    if len(labels) != 1:
        return ""
    covered, last = 0.0, start
    for left, right in sorted(spans):
        covered += max(0.0, right - max(last, left))
        last = max(last, right)
    return next(iter(labels)) if covered >= end - start - 1e-7 else ""


def _time_value(value: Any, *, assume_milliseconds: bool = False) -> float:
    if isinstance(value, (int, float)):
        return float(value) / 1000 if assume_milliseconds else float(value)
    text = str(value or "").strip()
    if not text:
        raise ValueError("missing native word time")
    try:
        result = float(text.replace(",", "."))
        return result / 1000 if assume_milliseconds else result
    except ValueError:
        parts = text.replace(",", ".").split(":")
        if len(parts) == 3:
            result = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            # HH:MM:SS is already seconds, independently of the field name.
            return result
        raise ValueError("invalid native word time")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


__all__ = [
    "DEFAULT_MINIMUM_CONFIDENCE",
    "DefaultSpeakerIdentityResolver",
    "IdentityEvidence",
    "SpeakerIdentity",
    "SpeakerIdentityMap",
    "SpeakerIdentityResolver",
    "VerifiedVoiceMatch",
    "UNKNOWN_DISPLAY_NAME",
    "detect_speaker_self_introduction",
    "resolve_speaker_identities",
]
