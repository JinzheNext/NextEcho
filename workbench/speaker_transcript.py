from __future__ import annotations

import json
import importlib.util
import os
import re
import subprocess
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

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


@dataclass
class DiarizationTurn:
    speaker: str
    start: float
    end: float


@dataclass
class SpeakerTextTurn:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class AudioWindow:
    start: float
    end: float


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def build_speaker_transcript(
    input_path: str | Path,
    *,
    quality: str = "accurate",
    model_name: str | None = None,
    output_dir: Path | None = None,
    min_speakers: int = 2,
    max_speakers: int = 6,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    backend_status = ensure_speaker_dependencies()
    workspace = resolve_speaker_workspace(input_path, quality=quality, output_dir=output_dir, model_name=model_name)
    _emit_progress(progress_callback, stage="preprocess", message="准备访谈音频", progress=10)
    derived_dir = workspace["run_dir"] / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    enhanced_audio = derived_dir / "audio.voice_enhanced.wav"
    enhance_voice_audio(workspace["audio_path"], enhanced_audio)

    transcript_segments = load_transcript_segments(workspace["transcript_json_path"])
    intro_removed, intro_range = detect_intro_range(transcript_segments)

    _emit_progress(progress_callback, stage="diarization", message="正在区分说话人", progress=40)
    backend_name, diarization_turns = run_diarization(
        enhanced_audio,
        transcript_segments=transcript_segments,
        intro_end=intro_range[1] if intro_removed else 0.0,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        backend_status=backend_status,
        progress_callback=progress_callback,
    )

    _emit_progress(progress_callback, stage="align", message="正在对齐转写与说话人", progress=75)
    speaker_turns, unassigned = align_segments_to_speakers(
        transcript_segments,
        diarization_turns,
        intro_range[1] if intro_removed else 0.0,
    )
    speaker_turns, speaker_map, speaker_name_inference = apply_context_speaker_names(
        speaker_turns,
        context=load_speaker_context(workspace.get("context_path")),
    )
    preview_text = build_preview_text(speaker_turns, 200)
    speaker_count_detected = len({turn.speaker for turn in diarization_turns})
    fallback_used = backend_name != "pyannote"
    quality_note = {
        "pyannote": "高精度说话人分离（pyannote）",
        "resemblyzer": "本地 speaker embedding 分离（请人工复核）",
    }.get(backend_name, "本地简化分离（准确度较低）")

    payload = {
        "source_audio": str(workspace["source_audio_path"]),
        "run_dir": str(workspace["run_dir"]),
        "backend": backend_name,
        "speaker_count_detected": speaker_count_detected,
        "fallback_used": fallback_used,
        "quality_note": quality_note,
        "speaker_map": speaker_map,
        "speaker_name_inference": speaker_name_inference,
        "source_context_path": str(workspace.get("context_path") or ""),
        "intro_removed": intro_removed,
        "intro_range": intro_range,
        "preprocess": {
            "normalized": True,
            "voice_enhanced": True,
        },
        "speaker_turns": [
            {
                "speaker": turn.speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "text": turn.text,
            }
            for turn in speaker_turns
        ],
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
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_audio),
        "-af",
        "highpass=f=80,lowpass=f=7600,afftdn=nf=-20,acompressor=threshold=-18dB:ratio=2:attack=20:release=250:makeup=1",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    return output_audio


def load_transcript_segments(transcript_json_path: Path) -> list[TranscriptSegment]:
    payload = json.loads(transcript_json_path.read_text(encoding="utf-8"))
    transcription = payload.get("transcription", [])
    segments: list[TranscriptSegment] = []
    for item in transcription:
        if not isinstance(item, dict):
            continue
        timestamps = item.get("timestamps", {})
        start = parse_timestamp(str(timestamps.get("from", "00:00:00,000")))
        end = parse_timestamp(str(timestamps.get("to", "00:00:00,000")))
        text = str(item.get("text", "")).strip()
        if text:
            segments.append(TranscriptSegment(start=start, end=end, text=text))
    return segments


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
) -> tuple[str, list[DiarizationTurn]]:
    selected_backend = str(backend_status.get("selected_backend") or "")
    if selected_backend == "pyannote":
        try:
            turns = run_pyannote_diarization(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            return "pyannote", turns
        except Exception as exc:
            if backend_status.get("fallback_backend_available"):
                _emit_progress(
                    progress_callback,
                    stage="diarization",
                    message=f"pyannote 不可用，回退到本地聚类分离（{exc.__class__.__name__}）",
                    progress=48,
                )
            else:
                raise
    if selected_backend == "resemblyzer":
        try:
            turns = run_resemblyzer_diarization(
                audio_path,
                transcript_segments=transcript_segments,
                intro_end=intro_end,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            return "resemblyzer", turns
        except Exception as exc:
            if backend_status.get("fallback_backend_available"):
                _emit_progress(
                    progress_callback,
                    stage="diarization",
                    message=f"speaker embedding 不可用，回退到本地聚类分离（{exc.__class__.__name__}）",
                    progress=48,
                )
            else:
                raise
    turns = run_segment_clustering_diarization(
        audio_path,
        transcript_segments=transcript_segments,
        intro_end=intro_end,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return "segment-clustering", turns


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
        return [DiarizationTurn("Speaker 1", kept_windows[0].start, kept_windows[0].end)]

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
        turns.append(DiarizationTurn(speaker_map[label], window.start, window.end))
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
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    sr, samples = load_wave_samples(audio_path)
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
        return []

    if len(kept_windows) == 1:
        return [DiarizationTurn("Speaker 1", kept_windows[0].start, kept_windows[0].end)]

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
            )
        )
    return merge_adjacent_diarization_turns(turns)


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
    if not turns:
        return []
    merged = [turns[0]]
    for turn in turns[1:]:
        current = merged[-1]
        if turn.speaker == current.speaker and turn.start - current.end <= 2.0:
            current.end = turn.end
            continue
        merged.append(turn)
    return merged


def align_segments_to_speakers(
    transcript_segments: list[TranscriptSegment],
    diarization_turns: list[DiarizationTurn],
    intro_end: float,
) -> tuple[list[SpeakerTextTurn], list[TranscriptSegment]]:
    assigned: list[SpeakerTextTurn] = []
    unassigned: list[TranscriptSegment] = []
    for segment in transcript_segments:
        if segment.end <= intro_end:
            continue
        overlaps = [
            (calculate_overlap(segment.start, segment.end, turn.start, turn.end), turn)
            for turn in diarization_turns
        ]
        best_overlap, best_turn = max(overlaps, key=lambda item: item[0], default=(0.0, None))
        if best_turn is None or best_overlap <= 0:
            unassigned.append(segment)
            continue
        assigned.append(
            SpeakerTextTurn(
                speaker=best_turn.speaker,
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
        )
    return merge_adjacent_turns(assigned), unassigned


def merge_adjacent_turns(turns: list[SpeakerTextTurn]) -> list[SpeakerTextTurn]:
    if not turns:
        return []
    merged = [turns[0]]
    for turn in turns[1:]:
        current = merged[-1]
        if turn.speaker == current.speaker and turn.start - current.end <= 1.0:
            current.end = turn.end
            current.text = f"{current.text}{turn.text}"
            continue
        merged.append(turn)
    return merged


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
    json_path = run_dir / "transcript.speakers.json"
    txt_path = run_dir / "transcript.speakers.txt"
    md_path = run_dir / "transcript.speakers.md"
    map_path = run_dir / "speaker_map.json"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(render_speaker_text(payload["speaker_turns"], with_timestamps=False), encoding="utf-8")
    md_path.write_text(render_speaker_text(payload["speaker_turns"], with_timestamps=True), encoding="utf-8")
    speaker_map = payload.get("speaker_map") or {turn["speaker"]: "" for turn in payload["speaker_turns"]}
    map_path.write_text(json.dumps(speaker_map, ensure_ascii=False, indent=2), encoding="utf-8")


def render_speaker_text(speaker_turns: list[dict[str, Any]], *, with_timestamps: bool) -> str:
    blocks: list[str] = []
    for turn in speaker_turns:
        if with_timestamps:
            label = f"{turn['speaker']} [{format_seconds(turn['start'])} - {format_seconds(turn['end'])}]"
        else:
            label = str(turn["speaker"])
        blocks.append(f"{label}\n{turn['text']}")
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


def detect_speaker_backend_status() -> dict[str, Any]:
    pyannote_installed = package_available("pyannote.audio")
    resemblyzer_installed = package_available("resemblyzer")
    hf_token = find_hf_token()
    fallback_backend_available = package_available("numpy") and package_available("sklearn")
    selected_backend = ""
    if pyannote_installed and hf_token:
        selected_backend = "pyannote"
    elif resemblyzer_installed:
        selected_backend = "resemblyzer"
    elif fallback_backend_available:
        selected_backend = "segment-clustering"
    return {
        "pyannote_installed": pyannote_installed,
        "resemblyzer_installed": resemblyzer_installed,
        "hf_token_env": next((name for name in HF_TOKEN_ENV_VARS if os.environ.get(name)), ""),
        "fallback_backend_available": fallback_backend_available,
        "selected_backend": selected_backend,
        "ready": bool(selected_backend),
    }


def ensure_speaker_dependencies() -> dict[str, Any]:
    status = detect_speaker_backend_status()
    if status["ready"]:
        return status

    if not status["pyannote_installed"]:
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
