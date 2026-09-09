from __future__ import annotations

import json
import importlib.util
import os
import re
import subprocess
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .diarization import (
    DiarizationBackendUnavailable,
    DiarizationResult,
    DiarizationTurn,
    DEFAULT_SHERPA_MODEL_DIR,
    coerce_result,
    coerce_turns,
    create_diarization_backend,
    sha256_file,
    write_raw_diarization_outputs,
)
from .enhancement import strip_filler_words
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
HF_TOKEN_ENV_VARS = [
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_ACCESS_TOKEN",
]


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
    end: float
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


def build_speaker_transcript(
    input_path: str | Path,
    *,
    quality: str = "accurate",
    model_name: str | None = None,
    output_dir: Path | None = None,
    min_speakers: int = 2,
    max_speakers: int = 6,
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
    backend_name, diarization_turns = _unpack_diarization_output(diarization_output)
    diarization_result = coerce_result(
        diarization_turns,
        backend=backend_name,
        audio_path=voice_audio,
        model=_backend_model_name(backend_name),
        metadata={
            "audio_path": str(voice_audio),
            "audio_role": "speaker_embedding_reference",
            "transcript_coverage": {
                "start_ms": round(min((item.start for item in transcript_segments), default=0.0) * 1000),
                "end_ms": round(max((item.end for item in transcript_segments), default=0.0) * 1000),
            },
        },
    )
    raw_json_path, raw_rttm_path = write_raw_diarization_outputs(
        workspace["run_dir"],
        diarization_result,
        file_id=Path(workspace["source_audio_path"]).stem or "audio",
    )

    _emit_progress(progress_callback, stage="align", message="正在对齐转写与说话人", progress=75)
    speaker_turns, unassigned = align_segments_to_speakers(
        transcript_segments,
        diarization_turns,
        intro_range[1] if intro_removed else 0.0,
        context=speaker_context,
    )
    identity_map = DefaultSpeakerIdentityResolver().resolve(
        diarization_result,
        transcript_segments,
        context=speaker_context,
    )
    speaker_turns, _resolved_speaker_map = apply_identity_map_to_turns(speaker_turns, identity_map)
    speaker_map = {
        raw_label: (identity.display_name if identity.identity_status == "resolved" else "")
        for identity in identity_map.speakers
        for raw_label in identity.raw_labels
    }
    speaker_name_inference = identity_inference_payload(identity_map, speaker_context)
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
        "speakers": [item.to_dict() for item in identity_map.speakers],
        "diarization": diarization_result.to_dict(),
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
    if path.is_file():
        run_dir = output_dir or (Path(__file__).resolve().parent.parent / "outputs" / "transcriptions" / f"run_{now_slug()}_{uuid4().hex[:8]}")
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
            "run_dir": path,
            "audio_path": audio_path,
            "transcript_json_path": transcript_path,
            "source_audio_path": _resolve_run_path(item_dir, result.get("media_path"), fallback=audio_path),
            "context_path": item_dir / "source.context.json",
        }

    transcript_candidates = sorted(path.glob("transcript*.json"))
    audio_candidates = [path / "转录音频.wav", *sorted(path.glob("audio*.wav"))]
    audio_candidates = [candidate for candidate in audio_candidates if candidate.exists()]
    if not transcript_candidates or not audio_candidates:
        raise ValueError("Could not locate transcript JSON and audio WAV in the provided directory.")
    return {
        "run_dir": path,
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
    subprocess.run(command, capture_output=True, text=True, check=True)
    return output_audio


# Explicit name for new callers; ``enhance_voice_audio`` remains the public
# compatibility entry point used by older scripts and tests.
normalize_voice_reference_audio = enhance_voice_audio


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
    value: Any,
    *,
    segment_start: float = 0.0,
    segment_end: float = 0.0,
) -> list[TranscriptWord]:
    """Parse common Whisper/Qwen word timestamp shapes.

    ASR engines and historical fixtures use ``start/end``, ``from/to`` or
    millisecond offsets.  Invalid words are ignored so coarse segment
    alignment remains available as a deterministic fallback.
    """

    if not isinstance(value, (list, tuple)):
        return []
    words: list[TranscriptWord] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("word") or item.get("token") or "")
        if not text:
            continue
        start_key = "start" if "start" in item else ("from" if "from" in item else "start_ms")
        end_key = "end" if "end" in item else ("to" if "to" in item else "end_ms")
        if not start_key or not end_key:
            continue
        start = _read_time_value(item.get(start_key), assume_milliseconds=start_key in {"from", "start_ms"})
        end = _read_time_value(item.get(end_key), assume_milliseconds=end_key in {"to", "end_ms"})
        if end <= start:
            continue
        if segment_end > segment_start:
            start = max(segment_start, start)
            end = min(segment_end, end)
        if end <= start:
            continue
        words.append(TranscriptWord(text=text, start=start, end=end, word_id=str(item.get("id") or f"word_{index:05d}")))
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
    min_speakers: int,
    max_speakers: int,
    backend_status: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    backend_config: dict[str, Any] | None = None,
    strict_backend: bool | None = None,
) -> tuple[str, list[DiarizationTurn]]:
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
        turns = coerce_turns(result.turns if isinstance(result, DiarizationResult) else result)
        return _legacy_backend_name(selected_backend), turns
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
            fallback_turns = coerce_turns(
                fallback_result.turns if isinstance(fallback_result, DiarizationResult) else fallback_result
            )
            return "segment-clustering", fallback_turns
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
    min_speakers: int,
    max_speakers: int,
) -> list[DiarizationTurn]:
    """Cluster local speaker embeddings when the optional CPU backend is installed."""
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
    from resemblyzer import VoiceEncoder
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    sample_rate, samples = load_wave_samples(audio_path)
    windows = build_audio_windows(transcript_segments, intro_end=intro_end, window_seconds=3.2, step_seconds=1.6)
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

    matrix = StandardScaler().fit_transform(np.vstack(embeddings))
    min_clusters = min(max(min_speakers, 2), len(kept_windows))
    max_clusters = min(max(max_speakers, min_clusters), len(kept_windows) - 1 if len(kept_windows) > 2 else len(kept_windows))
    best_labels: list[int] | None = None
    best_score = float("-inf")
    for speaker_count in range(min_clusters, max_clusters + 1):
        labels = AgglomerativeClustering(n_clusters=speaker_count, linkage="ward").fit_predict(matrix).tolist()
        current_score = -1.0 if len(set(labels)) < 2 or len(kept_windows) <= speaker_count else float(silhouette_score(matrix, labels))
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
    return merge_adjacent_diarization_turns(turns)


def run_pyannote_diarization(
    audio_path: Path,
    *,
    min_speakers: int,
    max_speakers: int,
) -> list[DiarizationTurn]:
    token = find_hf_token()
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL_ID, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL_ID, use_auth_token=token)

    diarization = pipeline(str(audio_path), min_speakers=min_speakers, max_speakers=max_speakers)
    # Community-1 exposes an exclusive timeline in recent pyannote versions.
    # Prefer it when present so word alignment has one deterministic label per
    # instant; older versions fall back to the regular annotation.
    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        diarization = exclusive() if callable(exclusive) else exclusive
    speaker_labels: dict[str, str] = {}
    turns: list[DiarizationTurn] = []
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        if speaker_label not in speaker_labels:
            speaker_labels[speaker_label] = f"Speaker {len(speaker_labels) + 1}"
        turns.append(
            DiarizationTurn(
                speaker=speaker_labels[speaker_label],
                start=float(turn.start),
                end=float(turn.end),
                confidence=0.75,
                raw_speaker_label=str(speaker_label),
            )
        )
    return turns


def run_segment_clustering_diarization(
    audio_path: Path,
    *,
    transcript_segments: list[TranscriptSegment],
    intro_end: float,
    min_speakers: int,
    max_speakers: int,
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
    min_clusters = min(max(min_speakers, 2), len(kept_windows))
    max_clusters = min(max(max_speakers, min_clusters), len(kept_windows) - 1 if len(kept_windows) > 2 else len(kept_windows))
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
    return merge_adjacent_diarization_turns(turns)


def run_timeline_fallback_diarization(
    transcript_segments: list[TranscriptSegment],
    *,
    intro_end: float = 0.0,
) -> list[DiarizationTurn]:
    """Dependency-free safety net for architecture/tests/offline runs.

    It intentionally emits one low-confidence anonymous stream rather than
    fabricating speaker switches from text.  A real acoustic backend can be
    plugged in later without changing the raw-result or identity contracts.
    """

    segments = [segment for segment in transcript_segments if segment.end > intro_end and segment.end > segment.start]
    if not segments:
        return []
    start = max(intro_end, min(segment.start for segment in segments))
    end = max(segment.end for segment in segments)
    return [DiarizationTurn("Speaker 1", start, end, confidence=0.2)]


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
    transcript_segments: list[TranscriptSegment],
    diarization_turns: list[DiarizationTurn],
    intro_end: float,
    context: dict[str, Any] | None = None,
    *,
    word_timestamps: list[TranscriptWord] | None = None,
) -> tuple[list[SpeakerTextTurn], list[TranscriptSegment]]:
    """Align transcript text to anonymous diarization turns.

    Word timings are preferred.  If an engine did not emit words, a segment is
    split at every diarization boundary and text is distributed proportionally
    across the resulting intervals.  Self-introduction text is intentionally a
    *segment-local* compatibility hint here.  The production build resolves it
    again in :mod:`speaker_identity`, keyed by the raw cluster, so no mutable
    ``active_anchor`` can leak into later turns.
    """

    assigned: list[SpeakerTextTurn] = []
    unassigned: list[TranscriptSegment] = []
    turns = coerce_turns(diarization_turns)
    candidates = _context_candidate_names(context or {})
    first_anchor_index: int | None = None
    first_anchor_name = ""
    for index, segment in enumerate(transcript_segments):
        anchor = detect_speaker_self_introduction(segment.text, candidates)
        if anchor and first_anchor_index is None:
            first_anchor_index = index
            first_anchor_name = anchor

    external_words = word_timestamps or []
    for index, segment in enumerate(transcript_segments):
        if segment.end <= intro_end:
            continue
        words = _coerce_transcript_words(list(segment.words or []))
        if not words and external_words:
            words = [
                word
                for word in _coerce_transcript_words(external_words)
                if calculate_overlap(segment.start, segment.end, word.start, word.end) > 0
            ]
        if words:
            pieces = _align_words_for_segment(segment, words, turns)
        else:
            pieces = _split_segment_at_diarization_boundaries(segment, turns)
        if not pieces:
            # Preserve the old contract: callers can explicitly review text
            # that did not overlap any acoustic turn.
            unassigned.append(segment)
            continue

        anchor = detect_speaker_self_introduction(segment.text, candidates)
        for piece in pieces:
            speaker = piece.speaker
            raw_speaker = piece.raw_speaker or speaker
            # Before the first explicit introduction, the two-name case is a
            # useful human-review hint.  Crucially this is calculated per
            # segment; it is not retained as an active anchor.
            if not anchor and first_anchor_index is not None and index < first_anchor_index and len(candidates) == 2:
                speaker = next((name for name in candidates if name != first_anchor_name), speaker)
            elif anchor and _piece_contains_anchor(piece.text, anchor):
                # Apply an introduction only to the current piece carrying the
                # words.  Other pieces of the same segment keep their raw
                # diarization label.
                speaker = anchor
            piece.speaker = speaker
            piece.raw_speaker = raw_speaker
            assigned.append(piece)
    return merge_adjacent_turns(assigned), unassigned


def _align_words_for_segment(
    segment: TranscriptSegment,
    words: list[TranscriptWord],
    diarization_turns: list[DiarizationTurn],
) -> list[SpeakerTextTurn]:
    pieces: list[SpeakerTextTurn] = []
    for word in sorted(words, key=lambda item: (item.start, item.end)):
        if not word.text:
            continue
        speaker, raw_speaker, confidence = speaker_for_interval(word.start, word.end, diarization_turns)
        if not raw_speaker and not speaker:
            continue
        piece = SpeakerTextTurn(
            speaker=speaker or "unknown",
            start=max(segment.start, word.start),
            end=min(segment.end, word.end),
            text=word.text,
            raw_speaker=raw_speaker,
            confidence=confidence,
            segment_ids=[segment.segment_id] if segment.segment_id else [],
        )
        if piece.end <= piece.start:
            continue
        if pieces and _can_merge_word_piece(pieces[-1], piece):
            pieces[-1].end = piece.end
            pieces[-1].text += piece.text
            if piece.confidence is not None:
                pieces[-1].confidence = min(
                    value for value in [pieces[-1].confidence, piece.confidence] if value is not None
                )
            continue
        pieces.append(piece)
    return pieces


def _coerce_transcript_words(value: list[Any]) -> list[TranscriptWord]:
    output: list[TranscriptWord] = []
    for index, item in enumerate(value):
        if isinstance(item, TranscriptWord):
            output.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("word") or item.get("token") or "")
        if not text:
            continue
        start_key = "start" if "start" in item else ("from" if "from" in item else "start_ms")
        end_key = "end" if "end" in item else ("to" if "to" in item else "end_ms")
        start = _read_time_value(item.get(start_key), assume_milliseconds=start_key in {"from", "start_ms"})
        end = _read_time_value(item.get(end_key), assume_milliseconds=end_key in {"to", "end_ms"})
        if end > start:
            output.append(TranscriptWord(text, start, end, str(item.get("id") or f"word_{index:05d}")))
    return output


def _split_segment_at_diarization_boundaries(
    segment: TranscriptSegment,
    diarization_turns: list[DiarizationTurn],
) -> list[SpeakerTextTurn]:
    if segment.end <= segment.start:
        return []
    intersections: list[tuple[float, float, str, str, float | None]] = []
    boundaries = {segment.start, segment.end}
    relevant = [turn for turn in diarization_turns if calculate_overlap(segment.start, segment.end, turn.start, turn.end) > 0]
    for turn in relevant:
        boundaries.add(max(segment.start, turn.start))
        boundaries.add(min(segment.end, turn.end))
    ordered = sorted(value for value in boundaries if segment.start <= value <= segment.end)
    for start, end in zip(ordered, ordered[1:]):
        if end - start <= 1e-6:
            continue
        speaker, raw_speaker, confidence = speaker_for_interval(start, end, diarization_turns)
        if not raw_speaker and not speaker:
            continue
        intersections.append((start, end, speaker or "unknown", raw_speaker, confidence))
    if not intersections:
        return []

    # Assign text by interval duration.  This is only a fallback for ASR
    # engines without word timestamps; boundaries still guarantee that a
    # speaker switch cannot be swallowed by one 15-second segment.
    text = segment.text
    total_duration = max(segment.end - segment.start, 1e-6)
    pieces: list[SpeakerTextTurn] = []
    cursor = 0
    for index, (start, end, speaker, raw_speaker, confidence) in enumerate(intersections):
        if index == len(intersections) - 1:
            end_cursor = len(text)
        else:
            target = round(len(text) * (end - segment.start) / total_duration)
            end_cursor = max(cursor, min(len(text), target))
        part = text[cursor:end_cursor]
        cursor = end_cursor
        if not part:
            continue
        pieces.append(
            SpeakerTextTurn(
                speaker=speaker,
                start=start,
                end=end,
                text=part,
                raw_speaker=raw_speaker,
                confidence=confidence,
                segment_ids=[segment.segment_id] if segment.segment_id else [],
            )
        )
    if cursor < len(text) and pieces:
        pieces[-1].text += text[cursor:]
    return pieces


def speaker_for_interval(
    start: float,
    end: float,
    diarization_turns: list[DiarizationTurn],
) -> tuple[str, str, float | None]:
    """Return display-compatible and raw labels for one time interval.

    A word overlapping multiple active raw turns is marked ``unknown`` rather
    than silently choosing one voice.  For non-overlap intervals the acoustic
    label remains anonymous until identity mapping runs.
    """

    overlaps: list[tuple[float, DiarizationTurn]] = []
    for turn in diarization_turns:
        amount = calculate_overlap(start, end, turn.start, turn.end)
        if amount > 0:
            overlaps.append((amount, turn))
    if not overlaps:
        return "", "", None
    overlaps.sort(key=lambda item: (item[0], item[1].confidence or 0.0), reverse=True)
    labels = {turn.raw_label for _, turn in overlaps}
    if len(labels) > 1:
        return "unknown", "", min((turn.confidence for _, turn in overlaps if turn.confidence is not None), default=0.0)
    amount, selected = overlaps[0]
    return selected.raw_label, selected.raw_label, selected.confidence


def _can_merge_word_piece(previous: SpeakerTextTurn, current: SpeakerTextTurn) -> bool:
    return (
        previous.speaker == current.speaker
        and (previous.raw_speaker or previous.speaker) == (current.raw_speaker or current.speaker)
        and current.start - previous.end <= 0.8
    )


def _piece_contains_anchor(text: str, anchor: str) -> bool:
    return anchor in text or bool(re.search(rf"(?:是|叫)\s*{re.escape(anchor)}", text))


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
            current.end = turn.end
            current.text = f"{current.text}{turn.text}"
            current.segment_ids = list(dict.fromkeys([*current.segment_ids, *turn.segment_ids]))
            if current.confidence is None:
                current.confidence = turn.confidence
            elif turn.confidence is not None:
                current.confidence = min(current.confidence, turn.confidence)
            continue
        merged.append(turn)
    return merged


def apply_identity_map_to_turns(
    turns: list[SpeakerTextTurn],
    identity_map: SpeakerIdentityMap,
) -> tuple[list[SpeakerTextTurn], dict[str, str]]:
    """Attach resolved display names while preserving each raw cluster label."""

    by_label = identity_map.by_raw_label()
    mapping: dict[str, str] = {}
    output: list[SpeakerTextTurn] = []
    for turn in turns:
        raw_label = turn.raw_speaker or turn.speaker
        identity = by_label.get(raw_label)
        if identity is None:
            # A word in an overlap region may intentionally have no unique raw
            # label.  Preserve it as unknown instead of borrowing a nearby
            # identity.
            output.append(
                SpeakerTextTurn(
                    speaker="unknown",
                    start=turn.start,
                    end=turn.end,
                    text=turn.text,
                    raw_speaker="",
                    confidence=turn.confidence,
                    identity_status="unknown",
                    segment_ids=list(turn.segment_ids),
                )
            )
            continue
        display_name = identity.display_name if identity.identity_status == "resolved" else raw_label
        if identity.identity_status == "resolved":
            mapping[raw_label] = identity.display_name
        output.append(
            SpeakerTextTurn(
                speaker=display_name,
                start=turn.start,
                end=turn.end,
                text=turn.text,
                raw_speaker=raw_label,
                speaker_id=identity.speaker_id,
                display_name=identity.display_name,
                confidence=identity.confidence,
                identity_status=identity.identity_status,
                identity_evidence=[item.to_dict() for item in identity.identity_evidence],
                segment_ids=list(turn.segment_ids),
            )
        )
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
    raw_label = turn.raw_speaker or (turn.speaker if turn.speaker != "unknown" else "")
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


def _unpack_diarization_output(output: Any) -> tuple[str, list[DiarizationTurn]]:
    if isinstance(output, DiarizationResult):
        return _legacy_backend_name(output.backend), coerce_turns(output.turns)
    if isinstance(output, tuple) and len(output) == 2:
        backend, turns = output
        if isinstance(turns, DiarizationResult):
            return str(backend or turns.backend), coerce_turns(turns.turns)
        return str(backend or "segment-clustering"), coerce_turns(turns)
    return "segment-clustering", coerce_turns(output if isinstance(output, list) else [])


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
    turns: list[SpeakerTextTurn],
    *,
    context: dict[str, Any],
) -> tuple[list[SpeakerTextTurn], dict[str, str], dict[str, Any]]:
    """Use page-provided names as hints without pretending diarization knows identity."""
    labels = list(dict.fromkeys(turn.speaker for turn in turns))
    candidates = [
        item for item in (context.get("speaker_candidates") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    if not labels or not candidates:
        return turns, {label: "" for label in labels}, {"status": "no_context"}

    scores: dict[tuple[str, str], int] = {}
    for label in labels:
        text = "".join(turn.text for turn in turns if turn.speaker == label)
        for candidate in candidates:
            name = str(candidate["name"]).strip()
            score = 0
            if re.search(rf"(?:我|在下|本人)\s*(?:是|叫)\s*{re.escape(name)}", text):
                score += 8
            if re.search(rf"(?:大家好[，, ]*)?(?:我)?\s*是\s*{re.escape(name)}", text):
                score += 8
            score += min(3, text.count(name))
            scores[(label, name)] = score

    mapping: dict[str, str] = {label: "" for label in labels}
    used: set[str] = set()
    for label, name in sorted(scores, key=lambda key: scores[key], reverse=True):
        if scores[(label, name)] <= 0 or mapping[label] or name in used:
            continue
        mapping[label] = name
        used.add(name)
    # When the page gives exactly as many candidates as diarization labels and
    # one name was identified explicitly, the remaining label is still a safe
    # deterministic hint for the operator to verify.
    if len(labels) == len(candidates) and len(used) == len(labels) - 1:
        remaining_name = next((str(item["name"]).strip() for item in candidates if str(item["name"]).strip() not in used), "")
        remaining_label = next((label for label in labels if not mapping[label]), "")
        if remaining_name and remaining_label:
            mapping[remaining_label] = remaining_name
            used.add(remaining_name)

    renamed: list[SpeakerTextTurn] = []
    for turn in turns:
        renamed.append(
            SpeakerTextTurn(
                speaker=mapping.get(turn.speaker) or turn.speaker,
                start=turn.start,
                end=turn.end,
                text=turn.text,
                raw_speaker=turn.raw_speaker,
                speaker_id=turn.speaker_id,
                display_name=mapping.get(turn.speaker) or turn.display_name,
                confidence=turn.confidence,
                identity_status="hinted" if mapping.get(turn.speaker) else turn.identity_status,
                identity_evidence=list(turn.identity_evidence),
                segment_ids=list(turn.segment_ids),
            )
        )
    return renamed, mapping, {
        "status": "hinted" if any(mapping.values()) else "unresolved",
        "matched": sum(1 for value in mapping.values() if value),
        "label_count": len(labels),
        "candidate_count": len(candidates),
        "note": "姓名来自节目资料与自我介绍线索，仍需人工核对声纹对应关系。",
    }


def build_audio_windows(
    transcript_segments: list[TranscriptSegment],
    *,
    intro_end: float,
    window_seconds: float = 3.2,
    step_seconds: float = 1.6,
) -> list[AudioWindow]:
    windows: list[AudioWindow] = []
    for segment in transcript_segments:
        if segment.end <= intro_end:
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
        text = strip_filler_words(str(turn.get("text") or ""))
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
    pyannote_installed = package_available("pyannote.audio")
    resemblyzer_installed = package_available("resemblyzer")
    sherpa_onnx_installed = package_available("sherpa_onnx")
    hf_token = find_hf_token()
    # The fallback has a dependency-free timeline mode; numpy/sklearn only
    # raise its acoustic feature quality.  Reporting it as available keeps the
    # backend contract testable on a clean install.
    numpy_available = package_available("numpy")
    sklearn_available = package_available("sklearn")
    fallback_backend_available = True
    configured = _config_section(config, "diarization")
    sherpa_config = configured.get("sherpa") if isinstance(configured.get("sherpa"), dict) else configured
    sherpa_model_paths = _configured_sherpa_model_paths(sherpa_config)
    sherpa_models_ready = bool(sherpa_onnx_installed and sherpa_model_paths and all(path.is_file() for path in sherpa_model_paths))
    selected_backend = ""
    if pyannote_installed and hf_token:
        selected_backend = "pyannote"
    elif sherpa_models_ready:
        selected_backend = "sherpa-onnx"
    elif resemblyzer_installed:
        selected_backend = "resemblyzer"
    else:
        selected_backend = "segment-clustering"
    return {
        "pyannote_installed": pyannote_installed,
        "resemblyzer_installed": resemblyzer_installed,
        "sherpa_onnx_installed": sherpa_onnx_installed,
        "sherpa_model_paths": [str(path) for path in sherpa_model_paths],
        "sherpa_models_ready": sherpa_models_ready,
        "numpy_installed": numpy_available,
        "sklearn_installed": sklearn_available,
        "hf_token_env": next((name for name in HF_TOKEN_ENV_VARS if os.environ.get(name)), ""),
        "fallback_backend_available": fallback_backend_available,
        "backends": {
            "pyannote-community-1": pyannote_installed and bool(hf_token),
            "sherpa-onnx": sherpa_models_ready,
            "resemblyzer": resemblyzer_installed,
            "segment-clustering": True,
        },
        "selected_backend": selected_backend,
        "ready": bool(selected_backend),
    }


def ensure_speaker_dependencies(config: Any | None = None) -> dict[str, Any]:
    status = detect_speaker_backend_status(config=config)
    if status["ready"]:
        return status
    # ``segment-clustering`` is normally always available through its
    # dependency-free timeline fallback.  Keep the historical diagnostics for
    # callers that inject a status provider during health checks.
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
