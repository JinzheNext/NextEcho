"""Speaker diarization contracts and provenance-preserving raw artifacts.

The speaker transcript command historically returned a list of ad-hoc
``DiarizationTurn`` objects and immediately applied display names to them.
This module keeps the acoustic result separate from identity resolution.  The
objects here deliberately have no dependency on pyannote, Resemblyzer, numpy,
or scikit-learn; optional backends are imported only when their ``diarize``
method is called.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


@dataclass
class DiarizationTurn:
    """One anonymous acoustic turn.

    ``speaker`` is retained as the positional/legacy field used by the
    original implementation.  ``raw_speaker_label`` is the canonical name
    written to the immutable raw artifact.  Optional confidence/overlap
    fields let a backend expose uncertainty without making callers depend on a
    particular backend's result type.
    """

    speaker: str = ""
    start: float = 0.0
    end: float = 0.0
    confidence: float | None = None
    overlap: bool = False
    turn_id: str = ""
    raw_speaker_label: str = ""

    def __post_init__(self) -> None:
        if not self.raw_speaker_label:
            self.raw_speaker_label = self.speaker
        if not self.speaker:
            self.speaker = self.raw_speaker_label
        self.start = float(self.start)
        self.end = float(self.end)
        if self.confidence is not None:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def raw_label(self) -> str:
        """Short alias used by identity mapping code and third-party adapters."""

        return self.raw_speaker_label or self.speaker

    def to_dict(self, index: int = 0) -> dict[str, Any]:
        turn_id = self.turn_id or f"turn_{index:05d}"
        payload: dict[str, Any] = {
            "turn_id": turn_id,
            "raw_speaker_label": self.raw_label,
            # ``speaker`` is retained for consumers of the old artifact shape.
            "speaker": self.speaker,
            "start_ms": max(0, round(self.start * 1000)),
            "end_ms": max(0, round(self.end * 1000)),
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "overlap": bool(self.overlap),
            # A missing backend confidence is explicitly represented as 0.0;
            # consumers must not mistake a legacy turn for a verified match.
            "confidence": round(float(self.confidence) if self.confidence is not None else 0.0, 4),
        }
        return payload


@dataclass
class DiarizationResult:
    """Uniform result returned by every diarization backend."""

    backend: str = ""
    turns: list[DiarizationTurn] = field(default_factory=list)
    model: str = ""
    model_revision: str = ""
    model_digest: str = ""
    audio_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": int(self.schema_version or 1),
            "backend": self.backend,
            "model": self.model,
            "model_revision": self.model_revision,
            "model_digest": self.model_digest,
            "audio_sha256": self.audio_sha256,
            "turns": [turn.to_dict(index) for index, turn in enumerate(self.turns)],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class DiarizationBackend(Protocol):
    """Backend contract shared by pyannote, Resemblyzer and fallback."""

    name: str

    def diarize(
        self,
        audio_path: Path,
        options: Any | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> DiarizationResult:
        ...


@dataclass
class DiarizationOptions:
    """Portable options object accepted by every backend adapter."""

    min_speakers: int = 2
    max_speakers: int = 6
    intro_end: float = 0.0
    transcript_segments: list[Any] = field(default_factory=list)
    use_exclusive_timeline: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "intro_end": self.intro_end,
            "transcript_segments": self.transcript_segments,
            "use_exclusive_timeline": self.use_exclusive_timeline,
        }


def _option(options: Any, name: str, default: Any) -> Any:
    if options is None:
        return default
    if isinstance(options, dict):
        return options.get(name, default)
    return getattr(options, name, default)


class _LegacyFunctionBackend:
    """Adapter for the existing optional implementations.

    Keeping the imports in the method means importing NextEcho never imports a
    heavyweight acoustic runtime.  It also makes this adapter straightforward
    to monkeypatch in no-dependency tests.
    """

    name = ""
    model = ""
    runner_name = ""

    def diarize(
        self,
        audio_path: Path,
        options: Any | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> DiarizationResult:
        from . import speaker_transcript

        runner = getattr(speaker_transcript, self.runner_name)
        turns = runner(
            Path(audio_path),
            transcript_segments=list(_option(options, "transcript_segments", []) or []),
            intro_end=float(_option(options, "intro_end", 0.0) or 0.0),
            min_speakers=int(_option(options, "min_speakers", 2) or 2),
            max_speakers=int(_option(options, "max_speakers", 6) or 6),
        )
        return DiarizationResult(
            backend=self.name,
            model=self.model,
            turns=coerce_turns(turns),
        )


class PyannoteCommunityBackend(_LegacyFunctionBackend):
    name = "pyannote-community-1"
    model = "pyannote/speaker-diarization-community-1"
    runner_name = "run_pyannote_diarization"


class ResemblyzerFallbackBackend(_LegacyFunctionBackend):
    name = "resemblyzer"
    model = "resemblyzer/VoiceEncoder"
    runner_name = "run_resemblyzer_diarization"


class ResemblyzerBackend(ResemblyzerFallbackBackend):
    """Compatibility name for callers that treat Resemblyzer as a backend."""


class SegmentClusteringFallbackBackend(_LegacyFunctionBackend):
    name = "segment-clustering"
    model = "next-echo/acoustic-segment-features"
    runner_name = "run_segment_clustering_diarization"


class SegmentClusteringBackend(SegmentClusteringFallbackBackend):
    """Public name for the local acoustic/timeline fallback."""


class SherpaOnnxBackend(SegmentClusteringFallbackBackend):
    """Optional sherpa-onnx slot with a safe local fallback implementation."""

    name = "sherpa-onnx"
    model = "sherpa-onnx"


class FallbackDiarizationBackend(SegmentClusteringFallbackBackend):
    """Compatibility alias for the dependency-free fallback backend."""


def create_diarization_backend(name: str) -> DiarizationBackend:
    """Create a backend without importing its optional runtime."""

    normalized = (name or "").strip().lower().replace("_", "-")
    if normalized in {"pyannote", "pyannote-community-1", "community-1"}:
        return PyannoteCommunityBackend()
    if normalized in {"resemblyzer", "speaker-embedding", "embedding"}:
        return ResemblyzerFallbackBackend()
    if normalized == "sherpa-onnx":
        return SherpaOnnxBackend()
    if normalized in {"segment-clustering", "fallback", "local", "simple"}:
        return SegmentClusteringBackend()
    raise ValueError(f"Unsupported diarization backend: {name}")


def coerce_turns(turns: Sequence[Any] | None) -> list[DiarizationTurn]:
    """Normalize legacy dataclasses and backend dictionaries to our shape."""

    output: list[DiarizationTurn] = []
    for item in turns or []:
        if isinstance(item, DiarizationTurn):
            output.append(item)
            continue
        if isinstance(item, dict):
            raw = str(item.get("raw_speaker_label") or item.get("raw_label") or item.get("speaker") or "")
            start = item.get("start")
            end = item.get("end")
            if start is None:
                start = float(item.get("start_ms") or 0) / 1000
            if end is None:
                end = float(item.get("end_ms") or 0) / 1000
            output.append(
                DiarizationTurn(
                    speaker=raw,
                    start=float(start or 0),
                    end=float(end or 0),
                    confidence=_as_optional_float(item.get("confidence")),
                    overlap=bool(item.get("overlap")),
                    turn_id=str(item.get("turn_id") or ""),
                    raw_speaker_label=raw,
                )
            )
            continue
        raw = str(getattr(item, "raw_speaker_label", "") or getattr(item, "speaker", "") or "")
        output.append(
            DiarizationTurn(
                speaker=raw,
                start=float(getattr(item, "start", 0.0) or 0.0),
                end=float(getattr(item, "end", 0.0) or 0.0),
                confidence=_as_optional_float(getattr(item, "confidence", None)),
                overlap=bool(getattr(item, "overlap", False)),
                turn_id=str(getattr(item, "turn_id", "") or ""),
                raw_speaker_label=raw,
            )
        )
    return output


def coerce_result(
    result: DiarizationResult | Sequence[Any],
    *,
    backend: str = "",
    audio_path: Path | None = None,
    model: str = "",
    metadata: dict[str, Any] | None = None,
) -> DiarizationResult:
    """Convert old ``list[DiarizationTurn]`` results to ``DiarizationResult``."""

    if isinstance(result, DiarizationResult):
        if audio_path and not result.audio_sha256:
            result.audio_sha256 = sha256_file(audio_path)
        return result
    return DiarizationResult(
        backend=backend,
        model=model,
        turns=coerce_turns(result),
        audio_sha256=sha256_file(audio_path) if audio_path else "",
        metadata=metadata or {},
    )


def sha256_file(path: Path | str) -> str:
    """Hash an existing local artifact; return empty for test/missing paths."""

    candidate = Path(path)
    if not candidate.is_file():
        return ""
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_rttm(result: DiarizationResult, *, file_id: str = "audio") -> str:
    """Render standard RTTM without identity/display names."""

    lines: list[str] = []
    for turn in result.turns:
        duration = max(0.0, float(turn.end) - float(turn.start))
        if duration <= 0:
            continue
        lines.append(
            "SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>".format(
                file_id=file_id,
                start=max(0.0, float(turn.start)),
                duration=duration,
                label=turn.raw_label,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _write_immutable(path: Path, content: str) -> None:
    """Write once and refuse to silently replace an existing raw artifact."""

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise FileExistsError(f"Immutable diarization artifact already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_raw_diarization_outputs(
    run_dir: Path,
    result: DiarizationResult | Sequence[Any],
    *,
    backend: str = "",
    audio_path: Path | None = None,
    model: str = "",
    file_id: str = "audio",
) -> tuple[Path, Path]:
    """Persist immutable ``diarization.raw.json`` and ``diarization.raw.rttm``."""

    normalized = coerce_result(result, backend=backend, audio_path=audio_path, model=model)
    json_path = Path(run_dir) / "diarization.raw.json"
    rttm_path = Path(run_dir) / "diarization.raw.rttm"
    json_text = json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2) + "\n"
    _write_immutable(json_path, json_text)
    _write_immutable(rttm_path, render_rttm(normalized, file_id=file_id))
    return json_path, rttm_path


# Clear aliases make the intended artifact API discoverable to integrations
# while retaining one implementation.
write_diarization_artifacts = write_raw_diarization_outputs
write_raw_diarization = write_raw_diarization_outputs


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
