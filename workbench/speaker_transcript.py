from __future__ import annotations

import contextlib
import json
import hashlib
import math
import importlib.util
import os
import re
import sys
import wave
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from .enhancement import strip_filler_words
from typing import Any, Callable
from uuid import uuid4

from .diarization import (
    DiarizationBackendUnavailable as DiarizationBackendUnavailable,
    DiarizationResult,
    DiarizationTurn,
    DEFAULT_SHERPA_MODEL_DIR,
    coerce_result,
    coerce_turns,
    create_diarization_backend,
    sha256_file,
    write_raw_diarization_outputs,
)
from .resources import serialized_task
from .media_commands import run_media_command
from .speaker_identity import (
    DefaultSpeakerIdentityResolver,
    SpeakerIdentityMap,
    detect_speaker_self_introduction as detect_identity_self_introduction,
)
from .transcription import DEFAULT_MODEL_NAME, format_transcript_attribution, transcribe_media_sources

QUALITY_TO_MODEL = {
    "accurate": DEFAULT_MODEL_NAME,
    "fast": "base",
}
INTRO_PHRASES = [
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
]
PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"
PYANNOTE_COMMUNITY_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
HF_TOKEN_ENV_VARS = [
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_ACCESS_TOKEN",
]


@contextlib.contextmanager
def _local_model_execution():
    """Keep gated-model loading local and suppress optional usage telemetry."""
    names = {
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYANNOTE_METRICS_ENABLED": "0",
    }
    original_environment = {name: os.environ.get(name) for name in names}
    existing_modules = set(sys.modules)
    module_flags = []
    for name, module in tuple(sys.modules.items()):
        if name.startswith("huggingface_hub") and hasattr(module, "HF_HUB_OFFLINE"):
            module_flags.append((module, "HF_HUB_OFFLINE", module.HF_HUB_OFFLINE))
            module.HF_HUB_OFFLINE = True
    try:
        os.environ.update(names)
        yield
    finally:
        original_offline = str(original_environment["HF_HUB_OFFLINE"] or "").lower() in {"1", "on", "yes", "true"}
        for name, module in tuple(sys.modules.items()):
            if name.startswith("huggingface_hub") and name not in existing_modules and hasattr(module, "HF_HUB_OFFLINE"):
                module.HF_HUB_OFFLINE = original_offline
        for module, attribute, value in module_flags:
            setattr(module, attribute, value)
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    segment_id: str = ""
    words: list[TranscriptWord] = field(default_factory=list)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.segment_id and self.id:
            self.segment_id = self.id
        if not self.id and self.segment_id:
            self.id = self.segment_id


@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float | None
    word_id: str = ""


@dataclass
class SpeakerTextTurn:
    speaker: str
    start: float
    end: float
    text: str
    raw_speaker: str = ""
    speaker_id: str = ""
    display_name: str = ""
    confidence: float | None = None
    identity_status: str = "unknown"
    identity_evidence: list[dict[str, Any]] = field(default_factory=list)
    segment_ids: list[str] = field(default_factory=list)
    timing_precision: str = "review-envelope"
    alignment_spans: list[dict[str, Any]] = field(default_factory=list)

    @property
    def raw_speaker_label(self) -> str:
        return self.raw_speaker


@dataclass
class AudioWindow:
    start: float
    end: float


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def resolve_speaker_backend(
    requested: str | None = "auto",
    *,
    config: Any | None = None,
    backend_status: dict[str, Any] | None = None,
) -> tuple[str, str, bool, dict[str, Any]]:
    """Resolve one requested/configured backend and its strictness.

    ``auto`` may use the detected best available backend and fall back when an
    optional runtime fails.  Any explicit backend (including sherpa-onnx) is
    strict: initialization errors are surfaced to the caller rather than being
    hidden behind a different backend.
    """

    status = backend_status or detect_speaker_backend_status()
    configured = _config_section(config, "diarization")
    requested_name = str(requested or "auto").strip().lower() or "auto"
    if requested_name == "auto":
        configured_name = str(configured.get("backend") or "auto").strip().lower()
        requested_name = configured_name or "auto"
    strict_backend = requested_name not in {"", "auto"}
    if requested_name == "auto":
        selected = str(status.get("selected_backend") or "segment-clustering")
    else:
        selected = _normalize_backend_name(requested_name)
        # Validate aliases up front so a typo does not get interpreted as the
        # dependency-free fallback by a future backend selector.
        create_diarization_backend(selected)
    backend_config: dict[str, Any] = {}
    configured_backend_options = configured.get(selected) or configured.get(requested_name) or {}
    if isinstance(configured_backend_options, dict):
        backend_config.update(configured_backend_options)
    # The repository config may keep sherpa settings under a nested section;
    # copy it into the adapter's private options without leaking unrelated
    # diarization controls.
    if selected == "sherpa-onnx" and isinstance(configured.get("sherpa"), dict):
        backend_config = {**configured.get("sherpa", {}), **backend_config}
    if selected == "sherpa-onnx":
        for key in ("model_paths", "segmentation_model", "segmentation", "embedding_model", "embedding", "model_path", "model", "constructor_kwargs", "factory"):
            if key in configured and key not in backend_config:
                backend_config[key] = configured[key]
    return requested_name, selected, strict_backend, backend_config


def _config_section(config: Any | None, name: str) -> dict[str, Any]:
    if config is None:
        return {}
    try:
        section = config.get(name, {})
    except AttributeError:
        try:
            section = config[name]
        except (KeyError, TypeError):
            section = {}
    return dict(section) if isinstance(section, dict) else {}


def _config_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _config_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(value)


def _normalize_backend_name(value: str) -> str:
    normalized = (value or "").strip().lower().replace("_", "-")
    return {
        "community-1": "pyannote",
        "pyannote-community-1": "pyannote",
        "speaker-embedding": "resemblyzer",
        "embedding": "resemblyzer",
        "fallback": "segment-clustering",
        "local": "segment-clustering",
        "simple": "segment-clustering",
    }.get(normalized, normalized)


@serialized_task
def build_speaker_transcript(
    input_path: str | Path,
    *,
    quality: str = "accurate",
    model_name: str | None = None,
    output_dir: Path | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    backend: str = "auto",
    config: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    backend_status = ensure_speaker_dependencies(config=config) if config is not None else ensure_speaker_dependencies()
    requested_backend, selected_backend, strict_backend, backend_config = resolve_speaker_backend(
        backend,
        config=config,
        backend_status=backend_status,
    )
    backend_status = dict(backend_status)
    backend_status["requested_backend"] = requested_backend
    backend_status["selected_backend"] = selected_backend
    backend_status["strict_backend"] = strict_backend
    backend_status["backend_config"] = backend_config
    workspace = resolve_speaker_workspace(input_path, quality=quality, output_dir=output_dir, model_name=model_name)
    _emit_progress(progress_callback, stage="preprocess", message="准备访谈音频", progress=10)
    derived_dir = workspace["run_dir"] / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    # Speaker embeddings must see the normalized original, not the aggressively
    # denoised/compressed transcription path used by older versions.  The
    # compatibility function name is retained, but its implementation now only
    # performs mono/16 kHz conversion and light loudness normalization.
    voice_audio = derived_dir / "audio.voice_reference.wav"
    enhance_voice_audio(workspace["audio_path"], voice_audio)

    transcript_segments = load_transcript_segments(workspace["transcript_json_path"])
    speaker_context = load_speaker_context(workspace.get("context_path"))
    intro_removed, intro_range = detect_intro_range(transcript_segments)

    _emit_progress(progress_callback, stage="diarization", message="正在区分说话人", progress=40)
    diarization_kwargs: dict[str, Any] = {
        "transcript_segments": transcript_segments,
        "intro_end": intro_range[1] if intro_removed else 0.0,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "backend_status": backend_status,
        "progress_callback": progress_callback,
    }
    # Avoid breaking test doubles/third-party callers that still implement the
    # pre-config run_diarization signature when no new options are requested.
    if backend_config or strict_backend:
        diarization_kwargs.update(
            {
                "backend_config": backend_config,
                "strict_backend": strict_backend,
            }
        )
    diarization_output = run_diarization(voice_audio, **diarization_kwargs)
    diarization_result = _unpack_diarization_output(diarization_output)
    backend_name = _legacy_backend_name(diarization_result.backend)
    diarization_result = coerce_result(
        diarization_result,
        backend=backend_name,
        audio_path=voice_audio,
        model=_backend_model_name(backend_name),
        metadata={
            **diarization_result.metadata,
            "audio_path": str(voice_audio),
            "audio_role": "speaker_embedding_reference",
            "transcript_coverage": {
                "start_ms": round(min((item.start for item in transcript_segments), default=0.0) * 1000),
                "end_ms": round(max((item.end for item in transcript_segments), default=0.0) * 1000),
            },
        },
    )
    diarization_turns = coerce_turns(diarization_result.turns)
    # Preserve the complete overlapping track for DER/JER and raw artifacts.
    # The reading layer consumes a separate exclusive projection so text can
    # be attributed once without erasing acoustic overlap evidence.
    if diarization_result.exclusive_turns:
        reading_turns = coerce_turns(diarization_result.exclusive_turns)
        reading_track_source = "backend_exclusive"
    else:
        reading_turns = exclusive_diarization_turns(diarization_turns)
        reading_track_source = "derived_exclusive_midpoint_for_reading_only"
    raw_json_path, raw_rttm_path = write_raw_diarization_outputs(
        workspace["run_dir"],
        diarization_result,
        file_id=Path(workspace["source_audio_path"]).stem or "audio",
    )

    _emit_progress(progress_callback, stage="align", message="正在对齐转写与说话人", progress=75)
    speaker_turns, unassigned = align_segments_to_speakers(
        transcript_segments,
        reading_turns,
        intro_range[1] if intro_removed else 0.0,
        context=speaker_context,
    )
    voice_enrollment_evidence, voice_enrollment_status = collect_voice_enrollment_evidence(
        voice_audio,
        diarization_turns,
        config=config,
        backend_name=backend_name,
    )
    identity_config = _config_section(config, "speaker_identity")
    minimum_identity_confidence = _config_float(
        identity_config.get("minimum_confidence"),
        0.90,
    )
    identity_map = DefaultSpeakerIdentityResolver(minimum_confidence=minimum_identity_confidence).resolve(
        diarization_result,
        transcript_segments,
        context=speaker_context,
        profiles=voice_enrollment_evidence,
    )
    speaker_turns, _resolved_speaker_map = apply_identity_map_to_turns(speaker_turns, identity_map)
    speaker_map = {
        raw_label: (identity.display_name if identity.identity_status == "resolved" else "")
        for identity in identity_map.speakers
        for raw_label in identity.raw_labels
    }
    speaker_name_inference = identity_inference_payload(identity_map, speaker_context)
    speaker_name_inference["voice_enrollment"] = voice_enrollment_status
    preview_text = build_preview_text(speaker_turns, 200)
    speaker_count_detected = len(identity_map.speakers)
    fallback_used = backend_name not in {"pyannote", "pyannote-community-1"}
    quality_note = {
        "pyannote": "高精度说话人分离（pyannote）",
        "pyannote-community-1": "高精度说话人分离（pyannote）",
        "resemblyzer": "本地 speaker embedding 分离（请人工复核）",
    }.get(backend_name, "本地简化分离（准确度较低）")

    payload = {
        "schema_version": 3,
        "source_audio": str(workspace["source_audio_path"]),
        "source_audio_sha256": sha256_file(workspace["source_audio_path"]),
        "recording_id": str(workspace.get("recording_id") or Path(workspace["source_audio_path"]).stem),
        "transcript_source_path": str(workspace["transcript_json_path"]),
        "transcript_source_sha256": sha256_file(workspace["transcript_json_path"]),
        "run_dir": str(workspace["run_dir"]),
        "backend": backend_name,
        "backend_requested": requested_backend,
        "backend_selected": selected_backend,
        "backend_configured": bool(backend_config),
        "speaker_count_detected": speaker_count_detected,
        "fallback_used": fallback_used,
        "quality_note": quality_note,
        "speaker_map": speaker_map,
        "speaker_name_inference": speaker_name_inference,
        "voice_enrollment": voice_enrollment_status,
        "speakers": [item.to_dict() for item in identity_map.speakers],
        "diarization": diarization_result.to_dict(),
        "reading_track": {
            "source": reading_track_source,
            "turn_count": len(reading_turns),
            "regular_overlap_track_preserved": True,
            "scoring_eligible": False,
        },
        "diarization_raw": {
            "json_path": str(raw_json_path),
            "rttm_path": str(raw_rttm_path),
            "sha256": sha256_file(raw_json_path),
        },
        "source_context_path": str(workspace.get("context_path") or ""),
        "intro_removed": intro_removed,
        "intro_range": intro_range,
        "preprocess": {
            "normalized": True,
            "voice_enhanced": False,
            "voice_reference_audio": str(voice_audio),
            "voice_reference_preserves_normalized_original": True,
            "denoise_applied": False,
        },
        "speaker_turns": [speaker_turn_to_dict(turn) for turn in speaker_turns],
        "status": "partial" if any(not t.raw_speaker for t in speaker_turns) else "ready",
        "annotation_kind": "text-review-envelope",
        "segments_unassigned": [
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text,
            }
            for segment in unassigned
        ],
        "preview_text": preview_text,
    }

    _emit_progress(progress_callback, stage="write", message="正在写出访谈逐字稿", progress=90)
    write_speaker_outputs(workspace["run_dir"], payload)
    _emit_progress(progress_callback, stage="complete", message="访谈逐字稿已完成", progress=100)
    return payload


def resolve_speaker_workspace(
    input_path: str | Path,
    *,
    quality: str,
    output_dir: Path | None,
    model_name: str | None = None,
) -> dict[str, Path]:
    path = Path(input_path).expanduser().resolve()
    requested_output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if path.is_file():
        run_dir = requested_output_dir or (Path(__file__).resolve().parent.parent / "outputs" / "transcriptions" / f"run_{now_slug()}_{uuid4().hex[:8]}")
        chosen_model = model_name or QUALITY_TO_MODEL.get(quality, DEFAULT_MODEL_NAME)
        payload = transcribe_media_sources(
            [str(path)],
            output_dir=run_dir,
            model_name=chosen_model,
            language="zh",
            max_seconds=0,
        )
        manifest = json.loads(payload["manifest_path"].read_text(encoding="utf-8"))
        result = manifest["results"][0]
        item_dir = _resolve_run_path(run_dir, result.get("output_dir"), fallback=run_dir)
        audio_path = _resolve_run_path(item_dir, result.get("audio_path"), fallback=item_dir / "转录音频.wav")
        transcript_path = _resolve_run_path(item_dir, result.get("json_path"), fallback=item_dir / "transcript.json")
        return {
            "run_dir": run_dir,
            "audio_path": audio_path,
            "transcript_json_path": transcript_path,
            "source_audio_path": path,
            "context_path": item_dir / "source.context.json",
        }

    if not path.is_dir():
        raise FileNotFoundError(path)

    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results = manifest.get("results", [])
        if len(results) != 1:
            raise ValueError("speaker-transcript currently supports single-source run directories.")
        result = results[0]
        item_dir = _resolve_run_path(path, result.get("output_dir"), fallback=path)
        audio_path = _resolve_run_path(item_dir, result.get("audio_path"), fallback=item_dir / "转录音频.wav")
        transcript_path = _resolve_run_path(item_dir, result.get("json_path"), fallback=item_dir / "transcript.json")
        return {
            # A run directory is an input artifact when supplied here.  Keep
            # all newly derived speaker outputs in the requested destination;
            # the source audio/transcript/context below remain read-only.
            "run_dir": requested_output_dir or path,
            "audio_path": audio_path,
            "transcript_json_path": transcript_path,
            "source_audio_path": _resolve_run_path(item_dir, result.get("media_path"), fallback=audio_path),
            "context_path": item_dir / "source.context.json",
        }

    transcript_candidates = [path / "transcript.json"] if (path / "transcript.json").is_file() else [
        candidate
        for candidate in sorted(path.glob("transcript*.json"))
        if not candidate.name.startswith("transcript.speakers.")
    ]
    audio_candidates = [path / "转录音频.wav", *sorted(path.glob("audio*.wav"))]
    audio_candidates = [candidate for candidate in audio_candidates if candidate.exists()]
    if not transcript_candidates or not audio_candidates:
        raise ValueError("Could not locate transcript JSON and audio WAV in the provided directory.")
    return {
        "run_dir": requested_output_dir or path,
        "audio_path": audio_candidates[0],
        "transcript_json_path": transcript_candidates[0],
        "source_audio_path": audio_candidates[0],
        "context_path": path / "source.context.json",
    }


def _resolve_run_path(base_dir: Path, value: Any, *, fallback: Path) -> Path:
    if not value:
        return fallback
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate
    # Manifest paths are normally absolute, while older fixtures used paths
    # relative to the run directory or the current working directory.
    run_candidate = base_dir / candidate
    return run_candidate if run_candidate.exists() else candidate.resolve()


def enhance_voice_audio(source_audio: Path, output_audio: Path) -> Path:
    """Create a speaker-reference WAV without changing spectral identity.

    Older code applied high/low-pass filters, ``afftdn`` and a compressor here.
    Those operations are useful for intelligibility but can move speaker
    embeddings away from the distribution seen during enrollment.  Keep this
    path as normalized original audio: only channel/sample-rate conversion and
    a conservative loudness normalization are allowed.
    """

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_audio),
        "-af",
        "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio),
    ]
    run_media_command(command, timeout=7200, stage="voice-normalize")
    return output_audio


# Explicit name for new callers; ``enhance_voice_audio`` remains the public
# compatibility entry point used by older scripts and tests.
normalize_voice_reference_audio = enhance_voice_audio


def collect_voice_enrollment_evidence(
    audio_path: Path,
    diarization_turns: list[DiarizationTurn],
    *,
    config: Any | None = None,
    backend_name: str = "",
) -> tuple[list[Any], dict[str, Any]]:
    """Keep cross-recording profile matches out of automatic naming.

    A manually chosen cosine threshold is not calibrated identity evidence.
    The recording-only validation scope therefore does not read or match
    enrolled profile embeddings in this pipeline.
    """

    identity_config = _config_section(config, "speaker_identity")
    enabled = _config_bool(identity_config.get("enable_voice_enrollment"), False)
    status: dict[str, Any] = {
        "enabled": enabled,
        "backend": "resemblyzer",
        "profile_count": 0,
        "cluster_count": len({turn.raw_label for turn in diarization_turns if turn.raw_label}),
        "embedded_cluster_count": 0,
        "match_count": 0,
        "matches": [],
        "errors": [],
    }
    status["diarization_backend"] = backend_name or ""
    status["reason"] = "disabled_by_recording_only_validation_scope" if enabled else "disabled"
    return [], status


def extract_cluster_speaker_embeddings(
    samples: Any,
    sample_rate: int,
    diarization_turns: list[DiarizationTurn],
    encoder: Any,
    *,
    np_module: Any | None = None,
) -> dict[str, Any]:
    """Extract one normalized embedding per raw diarization cluster."""

    np = np_module
    if np is None:
        import numpy as np  # type: ignore[no-redef]
    by_label: dict[str, list[tuple[float, float]]] = {}
    for turn in coerce_turns(diarization_turns):
        label = turn.raw_label.strip()
        if label and turn.end > turn.start:
            by_label.setdefault(label, []).append((turn.start, turn.end))
    output: dict[str, Any] = {}
    for label, spans in by_label.items():
        vectors: list[Any] = []
        for start, end in _embedding_spans(spans):
            start_index = max(0, int(start * sample_rate))
            end_index = min(len(samples), int(end * sample_rate))
            chunk = samples[start_index:end_index]
            if len(chunk) < sample_rate:
                continue
            try:
                vector = encoder.embed_utterance(chunk, return_partials=False)
            except TypeError:
                vector = encoder.embed_utterance(chunk)
            values = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(values)) if values.size else 0.0
            if norm > 1e-12 and np.isfinite(values).all():
                vectors.append(values / norm)
        if not vectors:
            continue
        average = np.mean(np.vstack(vectors), axis=0)
        norm = float(np.linalg.norm(average))
        if norm > 1e-12 and np.isfinite(average).all():
            output[label] = (average / norm).astype(np.float32)
    return output


def _embedding_spans(spans: list[tuple[float, float]], *, max_chunks: int = 8, chunk_seconds: float = 3.2) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for start, end in spans:
        duration = end - start
        if duration <= 0:
            continue
        if duration <= chunk_seconds:
            output.append((start, end))
            continue
        step = max(1.6, (duration - chunk_seconds) / max(1, max_chunks - 1))
        cursor = start
        count = 0
        while cursor < end and count < max_chunks:
            chunk_end = min(end, cursor + chunk_seconds)
            if chunk_end - cursor >= 1.0:
                output.append((cursor, chunk_end))
                count += 1
            if chunk_end >= end:
                break
            cursor += step
    return output


def load_transcript_segments(transcript_json_path: Path) -> list[TranscriptSegment]:
    payload = json.loads(transcript_json_path.read_text(encoding="utf-8"))
    transcription = payload.get("transcription", [])
    segments: list[TranscriptSegment] = []
    for index, item in enumerate(transcription):
        if not isinstance(item, dict):
            continue
        timestamps = item.get("timestamps", {}) if isinstance(item.get("timestamps", {}), dict) else {}
        offsets = item.get("offsets", {}) if isinstance(item.get("offsets", {}), dict) else {}
        start_value, start_is_ms = _pick_transcript_time(item, timestamps, offsets, "from", "start", "start_ms")
        end_value, end_is_ms = _pick_transcript_time(item, timestamps, offsets, "to", "end", "end_ms")
        start = _read_time_value(
            start_value,
            assume_milliseconds=start_is_ms,
        )
        end = _read_time_value(
            end_value,
            assume_milliseconds=end_is_ms,
        )
        text = str(item.get("text", "")).strip()
        if text:
            segment_id = str(item.get("id") or item.get("segment_id") or f"seg_{index:05d}")
            words = parse_transcript_words(item.get("words"), segment_start=start, segment_end=end)
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    segment_id=segment_id,
                    words=words,
                )
            )
    return segments


def _pick_transcript_time(
    item: dict[str, Any],
    timestamps: dict[str, Any],
    offsets: dict[str, Any],
    timestamp_key: str,
    seconds_key: str,
    milliseconds_key: str,
) -> tuple[Any, bool]:
    if timestamp_key in timestamps:
        return timestamps[timestamp_key], False
    if timestamp_key in offsets:
        return offsets[timestamp_key], True
    if seconds_key in item:
        return item[seconds_key], False
    if milliseconds_key in item:
        return item[milliseconds_key], True
    return 0.0, False


def parse_transcript_words(
    value: Any, *, segment_start: float = 0.0, segment_end: float = 0.0,
) -> list[TranscriptWord]:
    """Keep measured intervals/onsets; never manufacture or clip word times."""
    if not isinstance(value, (list, tuple)):
        return []
    words = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("word") or item.get("token") or "")
        start_key = next((k for k in ("start", "from", "start_ms") if item.get(k) is not None), None)
        end_key = next((k for k in ("end", "to", "end_ms") if item.get(k) is not None), None)
        if not text or start_key is None:
            continue
        try:
            for key in (start_key, end_key):
                if key and isinstance(item[key], str) and not re.fullmatch(r"\s*(?:[+-]?\d+(?:[.,]\d+)?|\d+:\d+:\d+(?:[.,]\d+)?)\s*", item[key]):
                    raise ValueError("invalid native word time")
            start = _read_time_value(item[start_key], assume_milliseconds=start_key in {"from", "start_ms"})
            end = (_read_time_value(item[end_key], assume_milliseconds=end_key in {"to", "end_ms"})
                   if end_key else None)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end < start)):
            continue
        if segment_end > segment_start and (start < segment_start or start >= segment_end or
                                             (end is not None and end > segment_end)):
            continue
        words.append(TranscriptWord(text, start, end,
                                    str(item.get("id") or f"word_{index:05d}")))
    return words

def _read_time_value(value: Any, *, assume_milliseconds: bool = False) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        result = float(value)
        return result / 1000.0 if assume_milliseconds else result
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        try:
            return parse_timestamp(text.replace(".", ",") if "." in text and "," not in text else text)
        except (TypeError, ValueError):
            return 0.0
    try:
        result = float(text.replace(",", "."))
    except ValueError:
        return 0.0
    return result / 1000.0 if assume_milliseconds else result


def detect_intro_range(segments: list[TranscriptSegment]) -> tuple[bool, list[float]]:
    promo_hits: list[TranscriptSegment] = []
    for segment in segments[:12]:
        normalized = normalize_text(segment.text)
        if any(normalize_text(phrase) in normalized for phrase in INTRO_PHRASES):
            promo_hits.append(segment)
    if len(promo_hits) >= 2:
        return True, [0.0, promo_hits[-1].end]
    return False, [0.0, 0.0]


def run_diarization(
    audio_path: Path,
    *,
    transcript_segments: list[TranscriptSegment],
    intro_end: float,
    min_speakers: int | None,
    max_speakers: int | None,
    backend_status: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    backend_config: dict[str, Any] | None = None,
    strict_backend: bool | None = None,
) -> DiarizationResult:
    selected_backend = str(backend_status.get("selected_backend") or "").strip().lower()
    # Keep the old short names accepted by doctor/CLI while routing execution
    # through one backend contract.  ``auto`` is resolved by
    # detect_speaker_backend_status before this function is called.
    if selected_backend in {"pyannote-community-1", "community-1"}:
        selected_backend = "pyannote"
    if selected_backend in {"fallback", "local", "simple"}:
        selected_backend = "segment-clustering"
    if not selected_backend:
        selected_backend = "segment-clustering"

    options = {
        "transcript_segments": transcript_segments,
        "intro_end": intro_end,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "backend_config": dict(backend_config or backend_status.get("backend_config") or {}),
        "strict": bool(backend_status.get("strict_backend")) if strict_backend is None else bool(strict_backend),
    }
    strict = bool(options["strict"])
    try:
        backend = create_diarization_backend(selected_backend)
        result = backend.diarize(audio_path, options=options, progress_callback=progress_callback)
        normalized = coerce_result(result, backend=selected_backend, audio_path=audio_path)
        normalized.backend = _legacy_backend_name(selected_backend)
        return normalized
    except Exception as exc:
        if strict:
            # Explicit backend selection is a user-visible contract.  Keep
            # the original exception type/message so callers can distinguish a
            # missing sherpa model/API from a normal auto fallback.
            raise
        if selected_backend in {"pyannote", "resemblyzer"} and backend_status.get("fallback_backend_available", True):
            message = (
                "pyannote 不可用，回退到本地聚类分离"
                if selected_backend == "pyannote"
                else "speaker embedding 不可用，回退到本地聚类分离"
            )
            _emit_progress(
                progress_callback,
                stage="diarization",
                message=f"{message}（{exc.__class__.__name__}）",
                progress=48,
            )
            fallback = create_diarization_backend("segment-clustering")
            fallback_result = fallback.diarize(audio_path, options=options, progress_callback=progress_callback)
            fallback_normalized = coerce_result(
                fallback_result, backend="segment-clustering", audio_path=audio_path
            )
            fallback_normalized.backend = "segment-clustering"
            fallback_normalized.metadata.setdefault("fallback_from", selected_backend)
            fallback_normalized.metadata.setdefault("fallback_reason", exc.__class__.__name__)
            return fallback_normalized
        raise


def _legacy_backend_name(name: str) -> str:
    normalized = (name or "").lower()
    if normalized.startswith("pyannote") or normalized in {"community-1", "pyannote"}:
        return "pyannote"
    if normalized in {"resemblyzer", "speaker-embedding", "embedding"}:
        return "resemblyzer"
    if normalized == "sherpa-onnx":
        return "sherpa-onnx"
    return "segment-clustering"


def run_resemblyzer_diarization(
    audio_path: Path,
    *,
    transcript_segments: list[TranscriptSegment],
    intro_end: float,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[DiarizationTurn]:
    """Cluster local speaker embeddings when the optional CPU backend is installed."""
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
    from resemblyzer import VoiceEncoder
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    sample_rate, samples = load_wave_samples(audio_path)
    # Filler/music-only ASR segments are not useful speaker evidence.  In
    # particular, an intro full of repeated "嗯/啊" can form a spurious
    # cluster before the actual dialogue starts.  The source transcript stays
    # untouched; this filter only controls the embedding windows.
    windows = build_audio_windows(
        transcript_segments,
        intro_end=intro_end,
        window_seconds=3.2,
        step_seconds=1.6,
        lexical_only=True,
    )
    encoder = VoiceEncoder("cpu")
    embeddings: list[Any] = []
    kept_windows: list[AudioWindow] = []
    for window in windows:
        start = max(0, int(window.start * sample_rate))
        end = min(len(samples), int(window.end * sample_rate))
        chunk = samples[start:end]
        if len(chunk) < sample_rate:
            continue
        embedding = encoder.embed_utterance(chunk, return_partials=False)
        if embedding is not None and len(embedding):
            embeddings.append(embedding)
            kept_windows.append(window)
    if not kept_windows:
        return []
    if len(kept_windows) == 1:
        return [DiarizationTurn("Speaker 1", kept_windows[0].start, kept_windows[0].end, confidence=0.45)]

    # VoiceEncoder returns L2-like speaker vectors.  StandardScaler + Ward
    # changes that geometry and can collapse distinct voices; retain cosine
    # geometry with L2-normalized vectors and average linkage instead.
    matrix = normalize_speaker_embeddings(np.vstack(embeddings), np_module=np)
    min_clusters = min(max(int(min_speakers or 2), 2), len(kept_windows))
    max_clusters = min(max(int(max_speakers or 6), min_clusters), len(kept_windows) - 1 if len(kept_windows) > 2 else len(kept_windows))
    # The local podcast path is intentionally two-speaker by default.  This
    # avoids silhouette-selected extra clusters being driven by short pauses
    # or residual music; callers asking for at least three speakers retain the
    # old bounded model-selection behavior.
    cluster_counts = [min_clusters] if min_clusters <= 2 else list(range(min_clusters, max_clusters + 1))
    best_labels: list[int] | None = None
    best_score = float("-inf")
    for speaker_count in cluster_counts:
        labels = cluster_speaker_embeddings(
            matrix,
            speaker_count,
            AgglomerativeClustering,
            np_module=np,
        ).tolist()
        if len(set(labels)) < 2 or len(kept_windows) <= speaker_count:
            current_score = -1.0
        else:
            try:
                current_score = float(silhouette_score(matrix, labels, metric="cosine"))
            except TypeError:
                # Older sklearn releases may not expose the metric keyword.
                current_score = float(silhouette_score(matrix, labels))
        if current_score > best_score or best_labels is None:
            best_score = current_score
            best_labels = labels
    labels = smooth_window_labels(best_labels or [0] * len(kept_windows), kept_windows)
    speaker_map: dict[int, str] = {}
    turns: list[DiarizationTurn] = []
    for label, window in zip(labels, kept_windows):
        if label not in speaker_map:
            speaker_map[label] = f"Speaker {len(speaker_map) + 1}"
        turns.append(DiarizationTurn(speaker_map[label], window.start, window.end, confidence=0.45))
    return exclusive_diarization_turns(turns)


def run_pyannote_diarization(
    audio_path: Path,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    backend_config: dict[str, Any] | None = None,
) -> DiarizationResult:
    config = dict(backend_config or {})
    snapshot = Path(str(config.get("snapshot_path") or "")).expanduser()
    revision = str(config.get("revision") or "").strip()
    model_id = str(config.get("model_id") or PYANNOTE_MODEL_ID).strip()
    if (not snapshot.is_dir() or revision != PYANNOTE_COMMUNITY_REVISION or
            snapshot.resolve().name != revision or model_id != PYANNOTE_MODEL_ID or
            not (snapshot / "config.yaml").is_file()):
        raise DiarizationBackendUnavailable(
            "Community-1 requires the fixed model ID, pinned local snapshot directory, and revision; "
            "runtime model downloads are disabled."
        )
    configured_digest = str(config.get("model_digest") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", configured_digest):
        raise DiarizationBackendUnavailable(
            "Community-1 requires a pre-recorded SHA-256 for the complete local snapshot."
        )
    model_digest = _directory_sha256(snapshot)
    if configured_digest and configured_digest != model_digest:
        raise DiarizationBackendUnavailable(
            "Community-1 local snapshot digest does not match the configured model_digest."
        )
    constraints = {}
    if min_speakers is not None:
        constraints["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        constraints["max_speakers"] = int(max_speakers)
    with _local_model_execution():
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(str(snapshot))
        diarization = pipeline(str(audio_path), **constraints)

    # Community-1 returns both regular (overlap-preserving) and exclusive
    # annotations. Keep both tracks. The regular track is the acoustic source
    # of truth; exclusive is only a deterministic reading projection.
    regular = getattr(diarization, "speaker_diarization", None)
    if regular is None:
        regular = diarization
    elif callable(regular):
        regular = regular()
    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if callable(exclusive):
        exclusive = exclusive()

    speaker_labels: dict[str, str] = {}

    def convert_track(annotation) -> list[DiarizationTurn]:
        output: list[DiarizationTurn] = []
        if annotation is None:
            return output
        for turn, _, speaker_label in annotation.itertracks(yield_label=True):
            key = str(speaker_label)
            if key not in speaker_labels:
                speaker_labels[key] = f"Speaker {len(speaker_labels) + 1}"
            output.append(DiarizationTurn(
                speaker=speaker_labels[key],
                start=float(turn.start),
                end=float(turn.end),
                # The community model does not expose a calibrated turn
                # confidence in this adapter.
                confidence=None,
                raw_speaker_label=key,
            ))
        for index, current in enumerate(output):
            current.overlap = any(
                other.raw_label != current.raw_label
                and min(current.end, other.end) > max(current.start, other.start)
                for other_index, other in enumerate(output) if other_index != index
            )
        return output

    return DiarizationResult(
        backend="pyannote-community-1",
        model=model_id,
        model_revision=revision,
        model_digest=model_digest,
        turns=convert_track(regular),
        exclusive_turns=convert_track(exclusive),
        metadata={
            "regular_track": "speaker_diarization",
            "exclusive_track": "exclusive_speaker_diarization" if exclusive is not None else None,
            "speaker_count_constraint_applied": bool(constraints),
            "speaker_count_constraints": constraints,
            "runtime_model_downloads": False,
        },
    )


def run_segment_clustering_diarization(
    audio_path: Path,
    *,
    transcript_segments: list[TranscriptSegment],
    intro_end: float,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[DiarizationTurn]:
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except (ImportError, ModuleNotFoundError):
        return run_timeline_fallback_diarization(transcript_segments, intro_end=intro_end)

    try:
        sr, samples = load_wave_samples(audio_path)
    except (OSError, RuntimeError, ValueError):
        return run_timeline_fallback_diarization(transcript_segments, intro_end=intro_end)
    windows = build_audio_windows(transcript_segments, intro_end=intro_end)

    features: list[Any] = []
    kept_windows: list[AudioWindow] = []
    for window in windows:
        vector = extract_segment_feature(samples, sr, window.start, window.end)
        if vector is None:
            continue
        features.append(vector)
        kept_windows.append(window)

    if not kept_windows:
        return run_timeline_fallback_diarization(transcript_segments, intro_end=intro_end)

    if len(kept_windows) == 1:
        return [DiarizationTurn("Speaker 1", kept_windows[0].start, kept_windows[0].end, confidence=0.35)]

    matrix = StandardScaler().fit_transform(np.vstack(features))
    min_clusters = min(max(int(min_speakers or 2), 2), len(kept_windows))
    max_clusters = min(max(int(max_speakers or 6), min_clusters), len(kept_windows) - 1 if len(kept_windows) > 2 else len(kept_windows))
    best_labels: list[int] | None = None
    best_score = float("-inf")
    for speaker_count in range(min_clusters, max_clusters + 1):
        labels = AgglomerativeClustering(n_clusters=speaker_count, linkage="ward").fit_predict(matrix).tolist()
        if len(set(labels)) < 2 or len(kept_windows) <= speaker_count:
            current_score = -1.0
        else:
            current_score = float(silhouette_score(matrix, labels))
        if current_score > best_score or best_labels is None:
            best_score = current_score
            best_labels = labels
    labels = smooth_window_labels(best_labels or [0] * len(kept_windows), kept_windows)

    speaker_map: dict[int, str] = {}
    turns: list[DiarizationTurn] = []
    for label, window in zip(labels, kept_windows):
        if label not in speaker_map:
            speaker_map[label] = f"Speaker {len(speaker_map) + 1}"
        turns.append(
            DiarizationTurn(
                speaker=speaker_map[label],
                start=window.start,
                end=window.end,
                confidence=0.35,
            )
        )
    return exclusive_diarization_turns(turns)


def run_timeline_fallback_diarization(
    transcript_segments: list[TranscriptSegment], *, intro_end: float = 0.0,
) -> list[DiarizationTurn]:
    """No acoustic measurement means no voice track or person-count evidence."""
    return []

def load_wave_samples(audio_path: Path) -> tuple[int, Any]:
    import numpy as np

    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)

    if sample_width != 2:
        raise RuntimeError(f"Only 16-bit PCM WAV is supported for fallback diarization: {audio_path}")

    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return sample_rate, samples.astype(np.float32) / 32768.0


def extract_segment_feature(samples: Any, sample_rate: int, start: float, end: float) -> Any | None:
    import numpy as np

    start_index = max(0, int((start - 0.12) * sample_rate))
    end_index = min(len(samples), int((end + 0.12) * sample_rate))
    if end_index - start_index < int(sample_rate * 0.8):
        return None

    chunk = samples[start_index:end_index].astype(np.float32)
    chunk = chunk - float(chunk.mean())
    rms = float(np.sqrt(np.mean(chunk**2) + 1e-8))
    if rms < 1e-4:
        return None
    chunk = chunk / (rms + 1e-8)

    frame_length = 400
    hop_length = 160
    if len(chunk) < frame_length:
        chunk = np.pad(chunk, (0, frame_length - len(chunk)))

    rows: list[list[float]] = []
    frequencies = np.fft.rfftfreq(frame_length, d=1 / sample_rate)
    hamming = np.hamming(frame_length).astype(np.float32)
    for offset in range(0, len(chunk) - frame_length + 1, hop_length):
        frame = chunk[offset : offset + frame_length] * hamming
        spectrum = np.abs(np.fft.rfft(frame)) + 1e-8
        power = spectrum**2
        power_sum = float(power.sum())
        centroid = float((power * frequencies).sum() / power_sum)
        bandwidth = float(np.sqrt(np.maximum(((frequencies - centroid) ** 2 * power).sum() / power_sum, 0.0)))
        rolloff_index = int(np.searchsorted(np.cumsum(power), 0.85 * power_sum))
        rolloff = float(frequencies[min(rolloff_index, len(frequencies) - 1)])
        flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(frame)))))
        log_energy = float(np.log(np.mean(frame**2) + 1e-8))
        low_ratio = float(power[(frequencies >= 85) & (frequencies < 255)].sum() / power_sum)
        mid_ratio = float(power[(frequencies >= 255) & (frequencies < 2000)].sum() / power_sum)
        high_ratio = float(power[frequencies >= 2000].sum() / power_sum)
        rows.append([centroid, bandwidth, rolloff, flatness, zcr, log_energy, low_ratio, mid_ratio, high_ratio])

    if not rows:
        return None

    matrix = np.asarray(rows, dtype=np.float32)
    return np.concatenate([matrix.mean(axis=0), matrix.std(axis=0), np.asarray([len(chunk) / sample_rate], dtype=np.float32)])


def smooth_window_labels(labels: list[int], windows: list[AudioWindow]) -> list[int]:
    if len(labels) < 3:
        return labels
    smoothed = labels[:]
    for index in range(1, len(labels) - 1):
        duration = windows[index].end - windows[index].start
        if duration <= 6.0 and labels[index - 1] == labels[index + 1] != labels[index]:
            smoothed[index] = labels[index - 1]
    return smoothed


_NONLEXICAL_MARKERS = (
    "backgroundmusic",
    "instrumentalmusic",
    "music",
    "soundeffect",
    "soundeffects",
    "applause",
    "laughter",
    "laugh",
    "silence",
    "静音",
    "背景音乐",
    "片头音乐",
    "音乐",
    "音效",
    "掌声",
    "笑声",
    "笑",
)


def has_lexical_content(text: str) -> bool:
    """Return whether an ASR label contains speech lexical evidence.

    Diarization windows are driven by ASR timing, so a segment can be long
    enough to embed while containing only hesitation runs or an editorial
    ``[音乐]`` marker.  Keep this intentionally conservative: real words,
    names, numbers and non-Chinese scripts are retained, while punctuation,
    filler-only runs and common music/noise markers are rejected.
    """

    cleaned = strip_filler_words(str(text or ""))
    if not cleaned:
        return False
    # Remove bracketed/parenthesized annotations before comparing markers.
    cleaned = re.sub(r"[\[\](){}<>【】（）「」『』]", "", cleaned)
    compact = re.sub(r"[\s\W_]+", "", cleaned, flags=re.UNICODE).lower()
    if not compact:
        return False
    if compact in _NONLEXICAL_MARKERS:
        return False
    if re.fullmatch(r"[嗯呃额啊哈唉诶]+", compact):
        return False
    # Some ASR engines repeat a marker with separators, e.g. "音乐音乐".
    marker_residue = compact
    for marker in sorted(_NONLEXICAL_MARKERS, key=len, reverse=True):
        marker_residue = marker_residue.replace(marker, "")
    if not marker_residue:
        return False
    # Enrollment eligibility is separate from transcript presentation. Ignore
    # long leading hesitations here without deleting them from any transcript.
    marker_residue = re.sub(r"^[嗯呃额啊哈唉诶]{3,}", "", marker_residue)
    # One-character remnants such as the trailing "跟" in a long ASR
    # hallucination ("啊啊……跟") are too weak to anchor a voice cluster.
    # Genuine short backchannels are likewise unsuitable enrollment evidence.
    if len(marker_residue) < 2:
        return False
    # ``compact`` may consist solely of symbols that survived Unicode word
    # normalization; require an actual letter/number/CJK code point.
    return bool(re.search(r"[0-9A-Za-z\u00c0-\u024f\u3400-\u9fff]", marker_residue))


def is_filler_only_text(text: str) -> bool:
    """Compatibility/readability alias for callers filtering ASR segments."""

    return not has_lexical_content(text)


def normalize_speaker_embeddings(matrix: Any, *, np_module: Any | None = None) -> Any:
    """L2-normalize one embedding per row without changing its direction."""

    np = np_module
    if np is None:
        import numpy as np  # type: ignore[no-redef]

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("speaker embeddings must be a 2-D matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return values / norms


def cluster_speaker_embeddings(
    matrix: Any,
    n_clusters: int,
    clustering_cls: Any,
    *,
    np_module: Any | None = None,
) -> Any:
    """Cluster normalized speaker vectors with cosine-average linkage.

    sklearn renamed ``affinity`` to ``metric``.  Probe the modern API first
    so current installations use the intended cosine geometry, then retain a
    narrow compatibility fallback for older installations.
    """

    normalized = normalize_speaker_embeddings(matrix, np_module=np_module)
    try:
        estimator = clustering_cls(
            n_clusters=int(n_clusters),
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        estimator = clustering_cls(
            n_clusters=int(n_clusters),
            affinity="cosine",
            linkage="average",
        )
    return estimator.fit_predict(normalized)


def exclusive_diarization_turns(turns: list[DiarizationTurn]) -> list[DiarizationTurn]:
    """Convert overlapping acoustic windows to an exclusive timeline.

    For each pair of overlapping windows, add the midpoint of their overlap
    as a candidate boundary.  Intervals with no active window are omitted,
    preserving genuine pauses/gaps instead of filling the whole transcript.
    Within an overlap, the active window whose center is nearest owns the
    interval; this makes a two-window overlap split at its temporal midpoint.
    """

    normalized = [
        turn
        for turn in coerce_turns(turns)
        if turn.end > turn.start
    ]
    if not normalized:
        return []
    normalized.sort(key=lambda item: (item.start, item.end, item.raw_label))
    points = {float(turn.start) for turn in normalized} | {float(turn.end) for turn in normalized}
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            overlap_start = max(left.start, right.start)
            overlap_end = min(left.end, right.end)
            if overlap_end > overlap_start:
                points.add((overlap_start + overlap_end) / 2.0)
    ordered_points = sorted(points)
    pieces: list[DiarizationTurn] = []
    epsilon = 1e-7
    for start, end in zip(ordered_points, ordered_points[1:]):
        if end - start <= epsilon:
            continue
        midpoint = (start + end) / 2.0
        active = [turn for turn in normalized if turn.start < midpoint < turn.end]
        if not active:
            continue
        chosen = min(
            enumerate(active),
            key=lambda item: (
                abs(((item[1].start + item[1].end) / 2.0) - midpoint),
                item[0],
            ),
        )[1]
        pieces.append(
            DiarizationTurn(
                speaker=chosen.raw_label,
                raw_speaker_label=chosen.raw_label,
                start=start,
                end=end,
                confidence=chosen.confidence,
                overlap=False,
                turn_id=chosen.turn_id,
            )
        )
    if not pieces:
        return []

    merged: list[DiarizationTurn] = [pieces[0]]
    for piece in pieces[1:]:
        current = merged[-1]
        if piece.raw_label == current.raw_label and abs(piece.start - current.end) <= epsilon:
            current.end = piece.end
            if current.confidence is None:
                current.confidence = piece.confidence
            elif piece.confidence is not None:
                current.confidence = min(current.confidence, piece.confidence)
            continue
        merged.append(piece)
    return merged


def merge_adjacent_diarization_turns(turns: list[DiarizationTurn]) -> list[DiarizationTurn]:
    normalized = coerce_turns(turns)
    if not normalized:
        return []
    merged = [normalized[0]]
    for turn in normalized[1:]:
        current = merged[-1]
        if turn.raw_label == current.raw_label and turn.start - current.end <= 2.0:
            current.end = turn.end
            if current.confidence is None:
                current.confidence = turn.confidence
            elif turn.confidence is not None:
                current.confidence = min(current.confidence, turn.confidence)
            current.overlap = current.overlap or turn.overlap
            continue
        merged.append(turn)
    return merged


def align_segments_to_speakers(
    transcript_segments: list[TranscriptSegment], diarization_turns: list[DiarizationTurn],
    intro_end: float, context: dict[str, Any] | None = None, *,
    word_timestamps: list[TranscriptWord] | None = None,
) -> tuple[list[SpeakerTextTurn], list[TranscriptSegment]]:
    """Align exact source text to anonymous acoustic evidence, never names."""
    assigned, unassigned = [], []
    turns = coerce_turns(diarization_turns)
    external = _coerce_transcript_words(word_timestamps or [])
    for segment in transcript_segments:
        if segment.end <= intro_end:
            continue
        words = _coerce_transcript_words(list(segment.words or []))
        if not words:
            words = [w for w in external if segment.start <= w.start < segment.end]
        pieces = _align_words_for_segment(segment, words, turns)
        if not turns:
            unassigned.append(segment)
        assigned.extend(pieces)
    return merge_adjacent_turns(assigned), unassigned


def _align_words_for_segment(
    segment: TranscriptSegment, words: list[TranscriptWord], diarization_turns: list[DiarizationTurn],
) -> list[SpeakerTextTurn]:
    from .dialogue_alignment import align_dialogue_segments
    rows, _ = align_dialogue_segments(
        [{"id": segment.segment_id, "text": segment.text,
          "start_ms": segment.start * 1000, "end_ms": segment.end * 1000}],
        [{"text": w.text, "start_ms": w.start * 1000,
          "end_ms": w.end * 1000 if w.end is not None else None} for w in words],
        [{"speaker": t.raw_label, "start": t.start, "end": t.end,
          "overlap": t.overlap} for t in diarization_turns], prefix="legacy")
    labels = {t.raw_label for t in diarization_turns}
    return [SpeakerTextTurn(
        speaker=r["speaker"],
        raw_speaker=r["speaker"] if r["speaker"] in labels else "",
        start=r["start_ms"] / 1000, end=r["end_ms"] / 1000, text=r["text"],
        segment_ids=r["segment_ids"], timing_precision=r["timing_precision"],
        alignment_spans=r["alignment_spans"],
    ) for r in rows]


def _coerce_transcript_words(value: list[Any]) -> list[TranscriptWord]:
    output = []
    for item in value:
        if isinstance(item, TranscriptWord):
            if (math.isfinite(item.start) and item.start >= 0 and
                    (item.end is None or (math.isfinite(item.end) and item.end >= item.start))):
                output.append(item)
        elif isinstance(item, dict):
            output.extend(parse_transcript_words([item]))
    return output


def _split_segment_at_diarization_boundaries(
    segment: TranscriptSegment, diarization_turns: list[DiarizationTurn],
) -> list[SpeakerTextTurn]:
    return _align_words_for_segment(segment, [], diarization_turns)

def speaker_for_interval(
    start: float, end: float, diarization_turns: list[DiarizationTurn],
) -> tuple[str, str, float | None]:
    """Compatibility helper using the shared full-interval evidence rule."""
    from .dialogue_alignment import _voice_between
    label, _ = _voice_between(start * 1000, end * 1000, [
        {"speaker": t.raw_label, "start": t.start, "end": t.end, "overlap": t.overlap}
        for t in diarization_turns])
    active = [t for t in diarization_turns if calculate_overlap(start, end, t.start, t.end) > 0]
    confidence = min((t.confidence for t in active if t.confidence is not None), default=None)
    if label not in {t.raw_label for t in active}:
        return "unknown", "", confidence
    return label, label, confidence


def nearest_speaker_for_interval(
    start: float, end: float, diarization_turns: list[DiarizationTurn], *, max_distance: float,
) -> tuple[str, str, float | None]:
    """Legacy signature retained; distance alone is never acoustic evidence."""
    return speaker_for_interval(start, end, diarization_turns)


def detect_speaker_self_introduction(text: str, candidates: list[str]) -> str:
    return detect_identity_self_introduction(text, candidates)


def merge_adjacent_turns(turns: list[SpeakerTextTurn]) -> list[SpeakerTextTurn]:
    if not turns:
        return []
    merged = [turns[0]]
    for turn in turns[1:]:
        current = merged[-1]
        if (
            turn.speaker == current.speaker
            and (turn.raw_speaker or turn.speaker) == (current.raw_speaker or current.speaker)
            and turn.start - current.end <= 1.0
        ):
            current.end = max(current.end, turn.end)
            current.text = f"{current.text}{turn.text}"
            current.alignment_spans.extend(turn.alignment_spans)
            current.segment_ids = list(dict.fromkeys([*current.segment_ids, *turn.segment_ids]))
            if current.confidence is None:
                current.confidence = turn.confidence
            elif turn.confidence is not None:
                current.confidence = min(current.confidence, turn.confidence)
            continue
        merged.append(turn)
    return merged


def apply_identity_map_to_turns(
    turns: list[SpeakerTextTurn], identity_map: SpeakerIdentityMap,
) -> tuple[list[SpeakerTextTurn], dict[str, str]]:
    """Names never change voice assignment, source text, or timing evidence."""
    by_label, mapping, output = identity_map.by_raw_label(), {}, []
    for turn in turns:
        raw_label = turn.raw_speaker or turn.speaker
        identity = by_label.get(raw_label)
        if identity is None:
            output.append(replace(turn, raw_speaker="", identity_status="unassigned"))
            continue
        display = (identity.display_name if identity.identity_status == "resolved" else
                   f"说话人 {int(identity.speaker_id.split('_')[-1]) + 1}")
        if identity.identity_status == "resolved":
            mapping[raw_label] = identity.display_name
        output.append(replace(turn, speaker=display, raw_speaker=raw_label,
            speaker_id=identity.speaker_id, display_name=identity.display_name,
            confidence=identity.confidence, identity_status=identity.identity_status,
            identity_evidence=[item.to_dict() for item in identity.identity_evidence]))
    return merge_adjacent_turns(output), mapping

def identity_inference_payload(identity_map: SpeakerIdentityMap, context: dict[str, Any]) -> dict[str, Any]:
    anchors = [
        evidence
        for identity in identity_map.speakers
        for evidence in identity.identity_evidence
        if evidence.type == "self_introduction"
    ]
    return {
        "status": "resolved" if any(item.identity_status == "resolved" for item in identity_map.speakers) else "unresolved",
        "matched": sum(1 for item in identity_map.speakers if item.identity_status == "resolved"),
        "label_count": len(identity_map.speakers),
        "candidate_count": len(_context_candidate_names(context)),
        "minimum_confidence": identity_map.minimum_confidence,
        "text_anchor_count": len(anchors),
        "text_anchor_names": list(dict.fromkeys(str(item.details.get("name") or "") for item in anchors if item.details.get("name"))),
        "identity_evidence": [item.to_dict() for identity in identity_map.speakers for item in identity.identity_evidence],
        "note": "姓名只通过当前匿名 cluster 的身份证据映射；未达到置信度阈值时保留 unknown。",
    }


def speaker_turn_to_dict(turn: SpeakerTextTurn) -> dict[str, Any]:
    raw_label = turn.raw_speaker or (turn.speaker if turn.speaker not in {"unknown", "归属待核对"} and not turn.speaker.startswith("交叠发言") else "")
    payload: dict[str, Any] = {
        # Legacy field retained for postprocess and older UI consumers.
        "speaker": turn.speaker,
        "raw_speaker": raw_label,
        "speaker_id": turn.speaker_id,
        "display_name": turn.display_name or (turn.speaker if turn.speaker != "unknown" else "unknown"),
        "identity_status": turn.identity_status,
        "start": round(turn.start, 3),
        "end": round(turn.end, 3),
        "text": turn.text,
        "segment_ids": list(turn.segment_ids),
        "identity_evidence": list(turn.identity_evidence),
        "timing_precision": turn.timing_precision,
        "annotation_kind": "text-review-envelope",
        "review_start_ms": turn.start * 1000,
        "review_end_ms": turn.end * 1000,
        "alignment_spans": list(turn.alignment_spans),
    }
    if turn.confidence is not None:
        payload["confidence"] = round(float(turn.confidence), 4)
    return payload


def _context_candidate_names(context: dict[str, Any]) -> list[str]:
    candidates = context.get("speaker_candidates") or []
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    names: list[str] = []
    for item in candidates:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _unpack_diarization_output(output: Any) -> DiarizationResult:
    if isinstance(output, DiarizationResult):
        output.backend = _legacy_backend_name(output.backend)
        return output
    if isinstance(output, tuple) and len(output) == 2:
        backend, turns = output
        if isinstance(turns, DiarizationResult):
            turns.backend = _legacy_backend_name(str(backend or turns.backend))
            return turns
        return DiarizationResult(
            backend=_legacy_backend_name(str(backend or "segment-clustering")),
            turns=coerce_turns(turns),
        )
    return DiarizationResult(
        backend="segment-clustering",
        turns=coerce_turns(output if isinstance(output, list) else []),
    )


def _backend_model_name(backend_name: str) -> str:
    return {
        "pyannote": PYANNOTE_MODEL_ID,
        "pyannote-community-1": PYANNOTE_MODEL_ID,
        "resemblyzer": "resemblyzer/VoiceEncoder",
        "segment-clustering": "next-echo/acoustic-segment-features",
    }.get(backend_name, backend_name)


def build_preview_text(turns: list[SpeakerTextTurn], limit: int) -> str:
    return "".join(turn.text for turn in turns)[:limit]


def load_speaker_context(context_path: Path | None) -> dict[str, Any]:
    if not context_path or not context_path.exists():
        return {}
    try:
        payload = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def apply_context_speaker_names(
    turns: list[SpeakerTextTurn], *, context: dict[str, Any],
) -> tuple[list[SpeakerTextTurn], dict[str, str], dict[str, Any]]:
    """Deprecated text-only helper: candidates are context, not voice evidence."""
    labels = list(dict.fromkeys(turn.speaker for turn in turns))
    return turns, {label: "" for label in labels}, {
        "status": "unresolved" if _context_candidate_names(context) else "no_context",
        "matched": 0, "label_count": len(labels),
        "candidate_count": len(_context_candidate_names(context)),
        "note": "此入口没有独立声学证据；姓名仅由SpeakerIdentityResolver确认。",
    }

def build_audio_windows(
    transcript_segments: list[TranscriptSegment],
    *,
    intro_end: float,
    window_seconds: float = 3.2,
    step_seconds: float = 1.6,
    lexical_only: bool = False,
) -> list[AudioWindow]:
    windows: list[AudioWindow] = []
    for segment in transcript_segments:
        if segment.end <= intro_end:
            continue
        if lexical_only and not has_lexical_content(segment.text):
            continue
        start = max(segment.start, intro_end)
        end = segment.end
        duration = end - start
        if duration <= 0.8:
            continue
        if duration <= window_seconds:
            windows.append(AudioWindow(start=start, end=end))
            continue
        cursor = start
        while cursor < end:
            window_end = min(end, cursor + window_seconds)
            if window_end - cursor >= 1.2:
                windows.append(AudioWindow(start=cursor, end=window_end))
            if window_end >= end:
                break
            cursor += step_seconds
    return windows


def write_speaker_outputs(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "transcript.speakers.json"
    txt_path = run_dir / "transcript.speakers.txt"
    md_path = run_dir / "transcript.speakers.md"
    map_path = run_dir / "speaker_map.json"

    payload.setdefault("schema_version", 3)
    turns = payload.get("speaker_turns") or []
    normalized_turns: list[dict[str, Any]] = []
    for item in turns:
        if isinstance(item, SpeakerTextTurn):
            normalized_turns.append(speaker_turn_to_dict(item))
            continue
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized.setdefault("raw_speaker", str(normalized.get("speaker") or ""))
        normalized.setdefault("speaker_id", "")
        normalized.setdefault("display_name", str(normalized.get("speaker") or "unknown"))
        normalized.setdefault("identity_status", "unknown")
        normalized.setdefault("identity_evidence", [])
        normalized.setdefault("segment_ids", [])
        normalized_turns.append(normalized)
    payload["speaker_turns"] = normalized_turns
    if not payload.get("speakers"):
        payload["speakers"] = _legacy_speaker_entries(normalized_turns)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(render_speaker_text(normalized_turns, with_timestamps=False), encoding="utf-8")
    md_path.write_text(render_speaker_text(normalized_turns, with_timestamps=True), encoding="utf-8")
    speaker_map = payload.get("speaker_map") or {turn.get("raw_speaker") or turn.get("speaker", ""): "" for turn in normalized_turns}
    map_path.write_text(json.dumps(speaker_map, ensure_ascii=False, indent=2), encoding="utf-8")


def _legacy_speaker_entries(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    labels: list[str] = []
    for turn in turns:
        if turn.get("identity_status") == "unassigned" or turn.get("speaker") in {"unknown", "归属待核对"} or str(turn.get("speaker", "")).startswith("交叠发言"):
            continue
        label = str(turn.get("raw_speaker") or turn.get("speaker") or "").strip()
        if not label or label in labels:
            continue
        labels.append(label)
        entries.append(
            {
                "speaker_id": str(turn.get("speaker_id") or f"spk_{len(entries):02d}"),
                "raw_labels": [label],
                "display_name": str(turn.get("display_name") or "unknown"),
                "confidence": float(turn.get("confidence") or 0.0),
                "identity_status": str(turn.get("identity_status") or "unknown"),
                "identity_evidence": list(turn.get("identity_evidence") or []),
            }
        )
    return entries


def render_speaker_text(speaker_turns: list[dict[str, Any]], *, with_timestamps: bool) -> str:
    blocks: list[str] = []
    for turn in speaker_turns:
        if with_timestamps:
            label = f"{turn['speaker']} [{format_seconds(turn['start'])} - {format_seconds(turn['end'])}]"
        else:
            label = str(turn["speaker"])
        # Rendering is lossless; any editorial cleanup belongs in an audited
        # derived layer, including initial acknowledgements and hesitations.
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        blocks.append(f"{label}：{text}")
    return format_transcript_attribution("\n\n".join(blocks))


def parse_timestamp(value: str) -> float:
    hours, minutes, seconds_ms = value.split(":")
    seconds, millis = seconds_ms.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def format_seconds(value: float) -> str:
    total = int(value)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calculate_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def find_hf_token() -> str:
    for name in HF_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _directory_sha256(directory: Path) -> str:
    """Hash a local model snapshot by relative path and file contents."""
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def package_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _configured_sherpa_model_paths(configured: dict[str, Any] | None) -> list[Path]:
    values: list[Any] = []
    config = configured if isinstance(configured, dict) else {}
    model_paths = config.get("model_paths")
    if isinstance(model_paths, dict):
        values.extend(model_paths.values())
    elif isinstance(model_paths, (list, tuple)):
        values.extend(model_paths)
    elif model_paths:
        values.append(model_paths)
    if not values:
        for key in ("segmentation_model", "segmentation", "embedding_model", "embedding", "model_path", "model"):
            if config.get(key):
                values.append(config[key])
    if not values:
        values = [DEFAULT_SHERPA_MODEL_DIR / "segmentation.onnx", DEFAULT_SHERPA_MODEL_DIR / "embedding.onnx"]
    paths: list[Path] = []
    for value in values:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent.parent / path).resolve()
        if path not in paths:
            paths.append(path)
    return paths


def detect_speaker_backend_status(config: Any | None = None) -> dict[str, Any]:
    if _config_section(config, "diarization").get("backend") == "nemotron-metal-candidate":
        from .capabilities import acoustic_speaker_status
        from .config import NextEchoConfig
        checked = acoustic_speaker_status(config if hasattr(config, "get_path") else NextEchoConfig(config or {}))
        return {**checked, "selected_backend": "nemotron-metal-candidate", "ready": checked["status"] == "ready",
                "backends": {"nemotron-metal-candidate": checked["status"] == "ready"}}
    pyannote_installed = package_available("pyannote.audio")
    resemblyzer_installed = package_available("resemblyzer")
    sherpa_onnx_installed = package_available("sherpa_onnx")
    # Empty timeline fallback carries no acoustic evidence and is not ready.
    numpy_available = package_available("numpy")
    sklearn_available = package_available("sklearn")
    fallback_backend_available = numpy_available and sklearn_available
    configured = _config_section(config, "diarization")
    pyannote_config = configured.get("pyannote") or configured.get("pyannote-community-1") or {}
    pyannote_config = pyannote_config if isinstance(pyannote_config, dict) else {}
    pyannote_snapshot_value = str(pyannote_config.get("snapshot_path") or "").strip()
    pyannote_snapshot = Path(pyannote_snapshot_value).expanduser() if pyannote_snapshot_value else None
    if pyannote_snapshot is not None and not pyannote_snapshot.is_absolute():
        pyannote_snapshot = (Path(__file__).resolve().parent.parent / pyannote_snapshot).resolve()
    pyannote_revision = str(pyannote_config.get("revision") or "").strip()
    pyannote_model_id = str(pyannote_config.get("model_id") or PYANNOTE_MODEL_ID).strip()
    pyannote_digest = str(pyannote_config.get("model_digest") or "").strip().lower()
    pyannote_snapshot_ready = bool(
        pyannote_snapshot is not None
        and pyannote_snapshot.is_dir()
        and pyannote_snapshot.resolve().name == PYANNOTE_COMMUNITY_REVISION
        and (pyannote_snapshot / "config.yaml").is_file()
        and pyannote_revision == PYANNOTE_COMMUNITY_REVISION
        and pyannote_model_id == PYANNOTE_MODEL_ID
        and re.fullmatch(r"[0-9a-f]{64}", pyannote_digest)
    )
    sherpa_config = configured.get("sherpa") if isinstance(configured.get("sherpa"), dict) else configured
    sherpa_model_paths = _configured_sherpa_model_paths(sherpa_config)
    sherpa_models_ready = bool(sherpa_onnx_installed and sherpa_model_paths and all(path.is_file() for path in sherpa_model_paths))
    availability = {
        "pyannote": pyannote_installed and pyannote_snapshot_ready,
        "sherpa-onnx": sherpa_models_ready,
        "resemblyzer": resemblyzer_installed,
        "segment-clustering": fallback_backend_available,
    }
    configured_backend = _normalize_backend_name(str(configured.get("backend") or "auto"))
    if configured_backend not in {"", "auto"}:
        selected_backend = configured_backend
    elif availability["pyannote"]:
        selected_backend = "pyannote"
    elif availability["sherpa-onnx"]:
        selected_backend = "sherpa-onnx"
    elif availability["resemblyzer"]:
        selected_backend = "resemblyzer"
    else:
        selected_backend = "segment-clustering"
    return {
        "pyannote_installed": pyannote_installed,
        "pyannote_snapshot_path": str(pyannote_snapshot) if pyannote_snapshot is not None else "",
        "pyannote_revision": pyannote_revision,
        "pyannote_snapshot_ready": pyannote_snapshot_ready,
        "pyannote_digest_configured": bool(re.fullmatch(r"[0-9a-f]{64}", pyannote_digest)),
        "resemblyzer_installed": resemblyzer_installed,
        "sherpa_onnx_installed": sherpa_onnx_installed,
        "sherpa_model_paths": [str(path) for path in sherpa_model_paths],
        "sherpa_models_ready": sherpa_models_ready,
        "numpy_installed": numpy_available,
        "sklearn_installed": sklearn_available,
        "hf_token_env": next((name for name in HF_TOKEN_ENV_VARS if os.environ.get(name)), ""),
        "fallback_backend_available": fallback_backend_available,
        "backends": {
            "pyannote-community-1": pyannote_installed and pyannote_snapshot_ready,
            "sherpa-onnx": sherpa_models_ready,
            "resemblyzer": resemblyzer_installed,
            "segment-clustering": fallback_backend_available,
        },
        "selected_backend": selected_backend,
        "ready": bool(availability.get(selected_backend, False)),
    }


def ensure_speaker_dependencies(config: Any | None = None) -> dict[str, Any]:
    status = detect_speaker_backend_status(config=config)
    if status["ready"]:
        return status
    if not status.get("pyannote_installed"):
        raise RuntimeError(
            "speaker transcript 依赖未安装：请先执行 `pip install -r requirements-speakers-lite.txt` 或 `requirements-speakers-embedding.txt`，"
            "若你要启用 pyannote 主方案，再安装 `requirements-speakers.txt`。"
        )
    raise RuntimeError(
        "未检测到可用的说话人分离后端。请设置 HF_TOKEN 以启用 pyannote，或安装 speaker embedding 依赖并确保 numpy 和 scikit-learn 可用。"
    )


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    stage: str,
    message: str,
    progress: int,
) -> None:
    if callback:
        callback({"stage": stage, "message": message, "progress": progress})
