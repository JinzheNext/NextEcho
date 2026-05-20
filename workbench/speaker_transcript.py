from __future__ import annotations

import json
import importlib.util
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .transcription import DEFAULT_MODEL_NAME, transcribe_media_sources

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


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def build_speaker_transcript(
    input_path: str | Path,
    *,
    quality: str = "accurate",
    output_dir: Path | None = None,
    min_speakers: int = 2,
    max_speakers: int = 2,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ensure_speaker_dependencies()
    workspace = resolve_speaker_workspace(input_path, quality=quality, output_dir=output_dir)
    _emit_progress(progress_callback, stage="preprocess", message="准备访谈音频", progress=10)
    derived_dir = workspace["run_dir"] / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    enhanced_audio = derived_dir / "audio.voice_enhanced.wav"
    enhance_voice_audio(workspace["audio_path"], enhanced_audio)

    transcript_segments = load_transcript_segments(workspace["transcript_json_path"])
    intro_removed, intro_range = detect_intro_range(transcript_segments)

    _emit_progress(progress_callback, stage="diarization", message="正在区分说话人", progress=40)
    diarization_turns = run_pyannote_diarization(
        enhanced_audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    _emit_progress(progress_callback, stage="align", message="正在对齐转写与说话人", progress=75)
    speaker_turns, unassigned = align_segments_to_speakers(
        transcript_segments,
        diarization_turns,
        intro_range[1] if intro_removed else 0.0,
    )
    preview_text = build_preview_text(speaker_turns, 200)

    payload = {
        "source_audio": str(workspace["source_audio_path"]),
        "run_dir": str(workspace["run_dir"]),
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
) -> dict[str, Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        run_dir = output_dir or (Path(__file__).resolve().parent.parent / "outputs" / "transcriptions" / f"run_{now_slug()}_{uuid4().hex[:8]}")
        model_name = QUALITY_TO_MODEL.get(quality, DEFAULT_MODEL_NAME)
        payload = transcribe_media_sources(
            [str(path)],
            output_dir=run_dir,
            model_name=model_name,
            language="zh",
            max_seconds=0,
        )
        manifest = json.loads(payload["manifest_path"].read_text(encoding="utf-8"))
        result = manifest["results"][0]
        return {
            "run_dir": run_dir,
            "audio_path": Path(result["audio_path"]),
            "transcript_json_path": Path(result["json_path"]),
            "source_audio_path": path,
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
        return {
            "run_dir": path,
            "audio_path": Path(result["audio_path"]),
            "transcript_json_path": Path(result["json_path"]),
            "source_audio_path": Path(result.get("media_path") or result["audio_path"]),
        }

    transcript_candidates = sorted(path.glob("transcript*.json"))
    audio_candidates = sorted(path.glob("audio*.wav"))
    if not transcript_candidates or not audio_candidates:
        raise ValueError("Could not locate transcript JSON and audio WAV in the provided directory.")
    return {
        "run_dir": path,
        "audio_path": audio_candidates[0],
        "transcript_json_path": transcript_candidates[0],
        "source_audio_path": audio_candidates[0],
    }


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


def write_speaker_outputs(run_dir: Path, payload: dict[str, Any]) -> None:
    json_path = run_dir / "transcript.speakers.json"
    txt_path = run_dir / "transcript.speakers.txt"
    md_path = run_dir / "transcript.speakers.md"
    map_path = run_dir / "speaker_map.json"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(render_speaker_text(payload["speaker_turns"], with_timestamps=False), encoding="utf-8")
    md_path.write_text(render_speaker_text(payload["speaker_turns"], with_timestamps=True), encoding="utf-8")
    map_path.write_text(json.dumps({"Speaker 1": "", "Speaker 2": ""}, ensure_ascii=False, indent=2), encoding="utf-8")


def render_speaker_text(speaker_turns: list[dict[str, Any]], *, with_timestamps: bool) -> str:
    blocks: list[str] = []
    for turn in speaker_turns:
        if with_timestamps:
            label = f"{turn['speaker']} [{format_seconds(turn['start'])} - {format_seconds(turn['end'])}]"
        else:
            label = str(turn["speaker"])
        blocks.append(f"{label}\n{turn['text']}")
    return "\n\n".join(blocks).strip() + "\n"


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


def ensure_speaker_dependencies() -> None:
    try:
        pyannote_available = importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:
        pyannote_available = False

    if not pyannote_available:
        raise RuntimeError("pyannote.audio is not installed. Install requirements-speakers.txt first.")
    if not find_hf_token():
        raise RuntimeError(
            "Speaker diarization requires a Hugging Face token. Set HF_TOKEN or HUGGINGFACE_TOKEN before running speaker-transcript."
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
