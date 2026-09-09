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
import importlib
import json
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


DEFAULT_SHERPA_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "speaker-diarization"


class DiarizationBackendUnavailable(RuntimeError):
    """Requested diarization backend cannot be initialized on this machine."""


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
    backend_config: dict[str, Any] = field(default_factory=dict)
    strict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "intro_end": self.intro_end,
            "transcript_segments": self.transcript_segments,
            "use_exclusive_timeline": self.use_exclusive_timeline,
            "backend_config": self.backend_config,
            "strict": self.strict,
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
    """Optional sherpa-onnx speaker diarization adapter.

    Unlike ``auto`` selection, an explicit sherpa request must never silently
    switch to an acoustic or timeline fallback.  The adapter validates the
    Python module, configured model paths, and the diarization API before
    returning a result; any missing piece raises
    :class:`DiarizationBackendUnavailable` with an actionable message.
    """

    name = "sherpa-onnx"
    model = "sherpa-onnx"

    def diarize(
        self,
        audio_path: Path,
        options: Any | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> DiarizationResult:
        config = dict(_option(options, "backend_config", {}) or {})
        injected_runner = config.get("runner")
        if callable(injected_runner):
            result = injected_runner(Path(audio_path), options)
            return DiarizationResult(self.name, turns=coerce_turns(result), model=self.model)

        try:
            module = importlib.import_module("sherpa_onnx")
        except (ImportError, ModuleNotFoundError) as exc:
            raise DiarizationBackendUnavailable(
                "sherpa-onnx backend requested but the `sherpa_onnx` Python package is not installed. "
                "Install the local sherpa-onnx runtime or choose backend=auto."
            ) from exc

        model_paths = _sherpa_model_paths(config)
        missing = [str(path) for path in model_paths if not path.is_file()]
        if not model_paths:
            raise DiarizationBackendUnavailable(
                "sherpa-onnx backend requested but no diarization models were configured. "
                "Expected models/speaker-diarization/segmentation.onnx and embedding.onnx, "
                "or set diarization.sherpa.model_paths in config."
            )
        if missing:
            raise DiarizationBackendUnavailable(
                "sherpa-onnx diarization model file(s) are missing: " + ", ".join(missing)
            )

        diarizer_factory = config.get("factory")
        if callable(diarizer_factory):
            diarizer = diarizer_factory(module=module, config=config)
        else:
            diarizer = _build_sherpa_diarizer(module, config, model_paths, _option(options, "min_speakers", 2), _option(options, "max_speakers", 6))
        if diarizer is None:
            raise DiarizationBackendUnavailable(
                "sherpa-onnx is installed but its OfflineSpeakerDiarization API is unavailable or "
                "not configured for the supplied models."
            )
        turns = _run_sherpa_diarizer(diarizer, Path(audio_path))
        return DiarizationResult(self.name, turns=coerce_turns(turns), model=str(config.get("model") or self.model))


class FallbackDiarizationBackend(SegmentClusteringFallbackBackend):
    """Compatibility alias for the dependency-free fallback backend."""


def _sherpa_model_paths(config: dict[str, Any]) -> list[Path]:
    nested = config.get("sherpa") if isinstance(config.get("sherpa"), dict) else config
    configured = nested.get("model_paths") if isinstance(nested, dict) else None
    values: list[Any] = []
    if isinstance(configured, dict):
        # Keep the constructor's segmentation/embedding order deterministic;
        # JSON/TOML object order is not a model-role contract.
        for key in ("segmentation", "segmentation_model", "embedding", "embedding_model"):
            if configured.get(key):
                values.append(configured[key])
        if not values:
            values.extend(configured.values())
    elif isinstance(configured, (list, tuple)):
        values.extend(configured)
    elif configured:
        values.append(configured)
    if not values and isinstance(nested, dict):
        segmentation = nested.get("segmentation_model") or nested.get("segmentation")
        embedding = nested.get("embedding_model") or nested.get("embedding")
        if isinstance(segmentation, dict):
            pyannote = segmentation.get("pyannote")
            segmentation = (pyannote or {}).get("model") if isinstance(pyannote, dict) else segmentation.get("model")
        if isinstance(embedding, dict):
            embedding = embedding.get("model")
        if segmentation:
            values.append(segmentation)
        if embedding:
            values.append(embedding)
        if not values:
            for key in ("model_path", "model"):
                value = nested.get(key)
                if value:
                    values.append(value)
    if not values:
        values = [
            DEFAULT_SHERPA_MODEL_DIR / "segmentation.onnx",
            DEFAULT_SHERPA_MODEL_DIR / "embedding.onnx",
        ]
    paths: list[Path] = []
    for value in values:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent.parent / path).resolve()
        if path not in paths:
            paths.append(path)
    return paths


def _build_sherpa_diarizer(
    module: Any,
    config: dict[str, Any],
    model_paths: list[Path],
    min_speakers: Any,
    max_speakers: Any,
) -> Any | None:
    diarizer_cls = getattr(module, "OfflineSpeakerDiarization", None)
    if diarizer_cls is None:
        return None
    supplied_config = config.get("config")
    if supplied_config is not None:
        try:
            return diarizer_cls(supplied_config)
        except TypeError:
            return diarizer_cls(config=supplied_config)

    constructor_kwargs = config.get("constructor_kwargs")
    if isinstance(constructor_kwargs, dict):
        try:
            return diarizer_cls(**constructor_kwargs)
        except TypeError as exc:
            raise DiarizationBackendUnavailable(
                f"sherpa-onnx OfflineSpeakerDiarization rejected constructor_kwargs: {exc}"
            ) from exc

    # sherpa-onnx 1.13.x requires the official nested config graph.  Passing
    # flat ``segmentation_model=...`` kwargs happens to work with a few old
    # test doubles but is rejected by the real binding, so prefer the nested
    # constructors whenever they are available.
    config_cls = getattr(module, "OfflineSpeakerDiarizationConfig", None)
    segmentation_cls = getattr(module, "OfflineSpeakerSegmentationModelConfig", None)
    pyannote_cls = getattr(module, "OfflineSpeakerSegmentationPyannoteModelConfig", None)
    embedding_cls = getattr(module, "SpeakerEmbeddingExtractorConfig", None)
    clustering_cls = getattr(module, "FastClusteringConfig", None)
    if all(item is not None for item in (config_cls, segmentation_cls, pyannote_cls, embedding_cls, clustering_cls)):
        segmentation_model = str(model_paths[0])
        embedding_model = str(model_paths[1] if len(model_paths) > 1 else model_paths[0])
        num_clusters = max(2, _sherpa_int(config.get("num_clusters"), _sherpa_int(min_speakers, 2)))
        threshold = _sherpa_float(config.get("cluster_threshold"), 0.5)
        try:
            segmentation = segmentation_cls(pyannote=pyannote_cls(model=segmentation_model))
            embedding = embedding_cls(model=embedding_model)
            clustering = clustering_cls(num_clusters=num_clusters, threshold=threshold)
            cfg = config_cls(
                segmentation=segmentation,
                embedding=embedding,
                clustering=clustering,
                min_duration_on=_sherpa_float(config.get("min_duration_on"), 0.3),
                min_duration_off=_sherpa_float(config.get("min_duration_off"), 0.5),
            )
            validate = getattr(cfg, "validate", None)
            if callable(validate) and not validate():
                raise ValueError("OfflineSpeakerDiarizationConfig.validate() returned false")
            return diarizer_cls(cfg)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise DiarizationBackendUnavailable(
                f"sherpa-onnx nested model/API configuration failed: {exc}"
            ) from exc

    # Older package builds may expose only the direct constructor.  Keep this
    # compatibility branch explicit; never silently fall back to another
    # diarization backend when the user selected sherpa.
    kwargs = {
        "segmentation_model": str(model_paths[0]),
        "embedding_model": str(model_paths[1] if len(model_paths) > 1 else model_paths[0]),
        "min_speakers": int(min_speakers or 1),
        "max_speakers": int(max_speakers or 6),
    }
    try:
        return diarizer_cls(**kwargs)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise DiarizationBackendUnavailable(
            f"sherpa-onnx OfflineSpeakerDiarization has no supported constructor: {exc}"
        ) from exc


def _sherpa_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _sherpa_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _resample_sherpa_samples(samples: Sequence[float], source_rate: int, target_rate: int) -> list[float]:
    if not samples or source_rate <= 0 or target_rate <= 0 or source_rate == target_rate:
        return list(samples)
    target_size = max(1, int(round(len(samples) * target_rate / source_rate)))
    if len(samples) == 1:
        return [float(samples[0])] * target_size
    output: list[float] = []
    scale = (len(samples) - 1) / max(1, target_size - 1)
    for index in range(target_size):
        source_index = index * scale
        left = min(len(samples) - 1, int(source_index))
        right = min(len(samples) - 1, left + 1)
        fraction = source_index - left
        output.append(float(samples[left]) * (1.0 - fraction) + float(samples[right]) * fraction)
    return output


def _run_sherpa_diarizer(diarizer: Any, audio_path: Path) -> list[DiarizationTurn]:
    try:
        with wave.open(str(audio_path), "rb") as handle:
            sample_width = handle.getsampwidth()
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise DiarizationBackendUnavailable(f"Unable to read sherpa-onnx audio input: {audio_path}") from exc
    if sample_width != 2:
        raise DiarizationBackendUnavailable("sherpa-onnx adapter requires 16-bit PCM WAV audio.")
    pcm = array("h")
    pcm.frombytes(raw)
    if channels > 1:
        samples = [
            sum(pcm[index : index + channels]) / channels
            for index in range(0, len(pcm), channels)
        ]
    else:
        samples = list(pcm)
    scale = 32768.0
    normalized = [float(sample) / scale for sample in samples]
    target_sample_rate = _sherpa_int(getattr(diarizer, "sample_rate", sample_rate), sample_rate)
    if target_sample_rate > 0 and sample_rate != target_sample_rate:
        normalized = _resample_sherpa_samples(normalized, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    try:
        import numpy as np

        process_samples: Any = np.asarray(normalized, dtype=np.float32)
    except ImportError:
        # The official wheel depends on numpy, but this keeps the adapter
        # importable/testable with a light fake backend in a clean install.
        process_samples = array("f", normalized)

    process = getattr(diarizer, "process", None) or getattr(diarizer, "diarize", None)
    if not callable(process):
        raise DiarizationBackendUnavailable(
            "sherpa-onnx diarizer has no process()/diarize() method."
        )
    try:
        output = process(process_samples)
    except TypeError:
        try:
            output = process(process_samples, sample_rate)
        except TypeError as exc:
            raise DiarizationBackendUnavailable(
                f"sherpa-onnx process() signature is unsupported: {exc}"
            ) from exc
    if output is None:
        output = diarizer
    sorter = getattr(output, "sort_by_start_time", None)
    if callable(sorter):
        output = sorter()
    return _extract_sherpa_turns(output, sample_rate)


def _extract_sherpa_turns(value: Any, sample_rate: int) -> list[DiarizationTurn]:
    if isinstance(value, DiarizationResult):
        return coerce_turns(value.turns)
    if isinstance(value, (list, tuple)):
        output: list[DiarizationTurn] = []
        for index, item in enumerate(value):
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                output.append(DiarizationTurn(_sherpa_speaker_label(item[2]), float(item[0]), float(item[1]), confidence=0.5))
                continue
            if isinstance(item, dict):
                speaker_value = item.get("speaker")
                if speaker_value is None or speaker_value == "":
                    speaker_value = item.get("label")
                if speaker_value is None or speaker_value == "":
                    speaker_value = item.get("raw_speaker_label")
                raw = _sherpa_speaker_label(speaker_value)
                start = item.get("start", item.get("start_ms", 0))
                end = item.get("end", item.get("end_ms", 0))
                start = float(start) / 1000 if "start_ms" in item and "start" not in item else float(start)
                end = float(end) / 1000 if "end_ms" in item and "end" not in item else float(end)
                if raw and end > start:
                    output.append(DiarizationTurn(raw, start, end, confidence=float(item.get("confidence") or 0.5)))
                continue
            speaker_value = getattr(item, "speaker", None)
            if speaker_value is None or speaker_value == "":
                speaker_value = getattr(item, "label", None)
            if speaker_value is None or speaker_value == "":
                speaker_value = f"SPEAKER_{index:02d}"
            raw = _sherpa_speaker_label(speaker_value)
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            if raw and start is not None and end is not None and float(end) > float(start):
                output.append(DiarizationTurn(raw, float(start), float(end), confidence=0.5))
        return output
    itertracks = getattr(value, "itertracks", None)
    if callable(itertracks):
        output = []
        try:
            rows = itertracks(yield_label=True)
        except TypeError:
            rows = itertracks()
        for row in rows:
            if isinstance(row, tuple) and len(row) >= 3:
                turn, _, label = row[:3]
                output.append(DiarizationTurn(str(label), float(turn.start), float(turn.end), confidence=0.5))
        return output
    count = getattr(value, "num_segments", None)
    getter = getattr(value, "get_segment", None)
    if count is not None and callable(getter):
        output = []
        for index in range(int(count)):
            item = getter(index)
            speaker_value = getattr(item, "speaker", None)
            if speaker_value is None or speaker_value == "":
                speaker_value = getattr(item, "label", None)
            if speaker_value is None or speaker_value == "":
                speaker_value = f"SPEAKER_{index:02d}"
            raw = _sherpa_speaker_label(speaker_value)
            start = float(getattr(item, "start", 0.0))
            end = float(getattr(item, "end", 0.0))
            if end > start:
                output.append(DiarizationTurn(raw, start, end, confidence=0.5))
        return output
    # OfflineSpeakerDiarizationResult in sherpa-onnx is iterable but is not a
    # list/tuple and exposes each segment as ``start/end/speaker`` attributes.
    try:
        iterator = iter(value)
    except TypeError:
        iterator = None
    if iterator is not None and not isinstance(value, (str, bytes, dict)):
        output = []
        for index, item in enumerate(iterator):
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            if start is None or end is None or float(end) <= float(start):
                continue
            speaker_value = getattr(item, "speaker", None)
            if speaker_value is None or speaker_value == "":
                speaker_value = getattr(item, "label", None)
            if speaker_value is None or speaker_value == "":
                speaker_value = f"SPEAKER_{index:02d}"
            raw = _sherpa_speaker_label(speaker_value)
            output.append(DiarizationTurn(raw, float(start), float(end), confidence=0.5))
        if output:
            return output
    raise DiarizationBackendUnavailable(
        "sherpa-onnx diarizer returned an unsupported segment result shape."
    )


def _sherpa_speaker_label(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"SPEAKER_{int(value):02d}"
    return str(value or "")


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
