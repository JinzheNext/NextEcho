from __future__ import annotations

import json
import os
import math
import shutil
import subprocess
import tempfile
import time
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from .sources import SourceDescriptor, resolve_sources


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRANSCRIPTION_ROOT = ROOT / "outputs" / "transcriptions"
DEFAULT_MODEL_ROOT = ROOT / "models" / "whisper.cpp"
KNOWN_MODEL_ROOTS = [
    Path.home() / "Documents" / "jerry_projects" / "AI-hot-topic" / "models" / "whisper.cpp",
]
DEFAULT_MODEL_NAME = "large-v3-turbo-q5_0"
DEFAULT_MAX_SECONDS = 0
DEFAULT_OUTPUT_FORMATS = ["txt", "json", "srt", "vtt"]
WHISPER_MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
LONG_AUDIO_CHUNK_MS = 10 * 60 * 1000
LONG_AUDIO_CPU_FALLBACK_MS = 15 * 60 * 1000
PROJECT_NAME = "NextEcho"
PROJECT_TAGLINE = "让播客自动转录进知识库"
TRANSCRIPT_ATTRIBUTION_HEADER = "本文转录由 GitHub 项目：NextEcho 提供支持，作者 @金哲Next（小红书、公众号同名）"
TRANSCRIPT_ATTRIBUTION_FOOTER = "powered by GitHub repo NextEcho - 让播客自动转录进知识库"


def transcribe_media_sources(
    sources: list[str | SourceDescriptor],
    *,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    language: str = "zh",
    max_seconds: int = DEFAULT_MAX_SECONDS,
    output_formats: list[str] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    descriptors = _coerce_descriptors(sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, stage="model", message="准备模型", progress=5)
    model_path = resolve_whisper_model(model_name)
    _emit_progress(progress_callback, stage="model_ready", message="模型已就绪", progress=15)
    whisper_bin = _which_binary("whisper-cli")
    ffmpeg_bin = _which_binary("ffmpeg")
    if not whisper_bin or not ffmpeg_bin:
        missing = [name for name, value in [("whisper-cli", whisper_bin), ("ffmpeg", ffmpeg_bin)] if not value]
        raise FileNotFoundError(f"Missing local dependencies: {', '.join(missing)}")

    formats = dedupe_strings(output_formats or DEFAULT_OUTPUT_FORMATS)
    generated_at = now_iso()
    run_config = {
        "generated_at": generated_at,
        "model_name": model_name,
        "model_path": str(model_path),
        "language": language,
        "max_seconds": max_seconds,
        "output_formats": formats,
        "source_count": len(descriptors),
    }
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    results: list[dict[str, Any]] = []
    for index, descriptor in enumerate(descriptors, start=1):
        base_progress = 15 + int(((index - 1) / max(len(descriptors), 1)) * 80)
        result = transcribe_single_source(
            descriptor=descriptor,
            items_dir=items_dir,
            model_path=model_path,
            whisper_bin=whisper_bin,
            ffmpeg_bin=ffmpeg_bin,
            language=language,
            max_seconds=max_seconds,
            output_formats=formats,
            index=index,
            progress_callback=progress_callback,
            base_progress=base_progress,
            item_count=len(descriptors),
        )
        results.append(result)
    failed_results = [item for item in results if item.get("status") == "failed"]

    manifest = {
        "generated_at": generated_at,
        "model_name": model_name,
        "model_path": str(model_path),
        "language": language,
        "max_seconds": max_seconds,
        "output_formats": formats,
        "run_config_path": str(run_config_path),
        "items_dir": str(items_dir),
        "results": results,
        "error_count": len(failed_results),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_path = None
    if failed_results:
        errors_path = output_dir / "errors.json"
        errors_path.write_text(json.dumps({"generated_at": generated_at, "items": failed_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit_progress(progress_callback, stage="complete", message="转写完成", progress=100)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "run_config_path": run_config_path,
        "results": results,
        "errors_path": errors_path,
    }


def transcribe_single_source(
    *,
    descriptor: SourceDescriptor,
    items_dir: Path,
    model_path: Path,
    whisper_bin: str,
    ffmpeg_bin: str,
    language: str,
    max_seconds: int,
    output_formats: list[str],
    index: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    base_progress: int = 15,
    item_count: int = 1,
) -> dict[str, Any]:
    source_label = descriptor.source_label or _infer_source_label(descriptor.input, index)
    item_dir = items_dir / source_label
    item_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = item_dir / "transcript.txt"
    if transcript_path.exists():
        cached_metadata = _read_cached_metadata(item_dir)
        if cached_metadata:
            cached_metadata["status"] = "cached"
            cached_metadata["updated_at"] = now_iso()
            _write_json(item_dir / "metadata.json", cached_metadata)
            _emit_progress(progress_callback, stage="cached", message=f"{source_label} 命中缓存", progress=_item_progress(base_progress, 1.0, item_count))
            return cached_metadata
        return _build_metadata(
            descriptor=descriptor,
            source_label=source_label,
            status="cached",
            transcript_text=_safe_read_text(transcript_path),
            item_dir=item_dir,
            media_path=_find_persisted_media(item_dir),
            audio_path=item_dir / "audio.wav",
            created_at=now_iso(),
        )
    if descriptor.error and descriptor.source_type != "local_file" and not descriptor.resolved_media_url:
        metadata = _build_metadata(
            descriptor=descriptor,
            source_label=source_label,
            status="failed",
            transcript_text="",
            item_dir=item_dir,
            media_path=_find_persisted_media(item_dir),
            audio_path=item_dir / "audio.wav",
            created_at=now_iso(),
            error=descriptor.error,
        )
        _write_json(item_dir / "metadata.json", metadata)
        _emit_progress(progress_callback, stage="item_failed", message=f"{source_label} 失败：{descriptor.error}", progress=_item_progress(base_progress, 1.0, item_count))
        return metadata

    try:
        with tempfile.TemporaryDirectory(dir=item_dir) as temp_dir:
            temp_path = Path(temp_dir)
            _emit_progress(progress_callback, stage="media", message=f"{source_label} 获取媒体", progress=_item_progress(base_progress, 0.1, item_count))
            media_path = prepare_media_source(descriptor, temp_path)
            persisted_media_path = _persist_media(media_path, item_dir)
            audio_path = item_dir / "audio.wav"
            _emit_progress(progress_callback, stage="audio", message=f"{source_label} 抽取音频", progress=_item_progress(base_progress, 0.35, item_count))
            extract_audio(
                source_path=persisted_media_path,
                target_path=audio_path,
                ffmpeg_bin=ffmpeg_bin,
                max_seconds=max_seconds,
            )
            _emit_progress(progress_callback, stage="asr", message=f"{source_label} 正在识别", progress=_item_progress(base_progress, 0.55, item_count))
            transcribe_audio_file(
                audio_path=audio_path,
                output_dir=item_dir,
                output_base="transcript",
                whisper_bin=whisper_bin,
                model_path=model_path,
                language=language,
                output_formats=output_formats,
            )
        transcript_text = transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else ""
        metadata = _build_metadata(
            descriptor=descriptor,
            source_label=source_label,
            status="ready" if transcript_text.strip() else "empty",
            transcript_text=transcript_text,
            item_dir=item_dir,
            media_path=persisted_media_path,
            audio_path=audio_path,
            created_at=now_iso(),
        )
        _emit_progress(progress_callback, stage="item_complete", message=f"{source_label} 已完成", progress=_item_progress(base_progress, 1.0, item_count))
    except Exception as exc:
        metadata = _build_metadata(
            descriptor=descriptor,
            source_label=source_label,
            status="failed",
            transcript_text="",
            item_dir=item_dir,
            media_path=_find_persisted_media(item_dir),
            audio_path=item_dir / "audio.wav",
            created_at=now_iso(),
            error=str(exc),
        )
        _emit_progress(progress_callback, stage="item_failed", message=f"{source_label} 失败：{exc}", progress=_item_progress(base_progress, 1.0, item_count))
    _write_json(item_dir / "metadata.json", metadata)
    return metadata



def compare_transcription_runs(
    left_dir: Path,
    right_dir: Path,
    *,
    output_path: Path,
    left_label: str = "left",
    right_label: str = "right",
) -> Path:
    left_manifest = json.loads((left_dir / "manifest.json").read_text(encoding="utf-8"))
    right_manifest = json.loads((right_dir / "manifest.json").read_text(encoding="utf-8"))
    left_results = {item["source_label"]: item for item in left_manifest.get("results", [])}
    right_results = {item["source_label"]: item for item in right_manifest.get("results", [])}
    rows: list[dict[str, Any]] = []
    for source_label in sorted(set(left_results) | set(right_results)):
        left_item = left_results.get(source_label, {})
        right_item = right_results.get(source_label, {})
        left_text = _safe_read_text(Path(left_item["text_path"])) if left_item.get("text_path") else ""
        right_text = _safe_read_text(Path(right_item["text_path"])) if right_item.get("text_path") else ""
        rows.append(
            {
                "source_label": source_label,
                f"{left_label}_status": left_item.get("status", "missing"),
                f"{right_label}_status": right_item.get("status", "missing"),
                f"{left_label}_char_count": len(left_text),
                f"{right_label}_char_count": len(right_text),
                f"{left_label}_excerpt": left_text[:240],
                f"{right_label}_excerpt": right_text[:240],
            }
        )
    payload = {
        "generated_at": now_iso(),
        "left_label": left_label,
        "right_label": right_label,
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "comparisons": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def ensure_whisper_model(model_name: str, model_root: Path = DEFAULT_MODEL_ROOT) -> Path:
    existing = _find_existing_model(model_name, model_root)
    if existing is not None:
        return existing
    model_root.mkdir(parents=True, exist_ok=True)
    candidate = model_root / f"ggml-{model_name}.bin"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate

    remote_url = f"{WHISPER_MODEL_BASE_URL}/ggml-{model_name}.bin"
    curl_bin = _which_binary("curl")
    if not curl_bin:
        raise FileNotFoundError("curl")
    lock_path = model_root / f".ggml-{model_name}.lock"
    part_path = model_root / f".ggml-{model_name}.{os.getpid()}.part"
    lock_handle = None
    try:
        while True:
            try:
                lock_handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if candidate.exists() and candidate.stat().st_size > 0:
                    return candidate
                time.sleep(1)
        command = [
            curl_bin,
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "-o",
            str(part_path),
            remote_url,
        ]
        subprocess.run(command, capture_output=True, text=True, check=True)
        part_path.replace(candidate)
        return candidate
    finally:
        if part_path.exists():
            part_path.unlink(missing_ok=True)
        if lock_handle is not None:
            os.close(lock_handle)
            lock_path.unlink(missing_ok=True)


def resolve_whisper_model(model_name_or_path: str) -> Path:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return ensure_whisper_model(model_name_or_path)


def prepare_media_source(source: str | SourceDescriptor, work_dir: Path) -> Path:
    descriptor = source if isinstance(source, SourceDescriptor) else _coerce_descriptors([source])[0]
    parsed = urlparse(descriptor.input)
    if descriptor.source_type == "local_file":
        local_path = Path(descriptor.input).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        return local_path
    if parsed.scheme in {"http", "https"}:
        if descriptor.resolver in {"direct", "xiaoyuzhou_page"} and descriptor.resolved_media_url:
            return _download_direct_media(descriptor.resolved_media_url, work_dir)
        download_source = descriptor.canonical_url or descriptor.input
        if _looks_like_direct_media_url(download_source):
            return _download_direct_media(download_source, work_dir)
        return _download_media_with_ytdlp(download_source, work_dir)
    local_path = Path(descriptor.input).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    return local_path


def extract_audio(
    *,
    source_path: Path,
    target_path: Path,
    ffmpeg_bin: str,
    max_seconds: int,
) -> Path:
    command = [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target_path),
    ]
    if max_seconds > 0:
        command[6:6] = ["-t", str(max_seconds)]
    subprocess.run(command, capture_output=True, text=True, check=True)
    return target_path


def _persist_media(source_path: Path, item_dir: Path) -> Path:
    suffix = source_path.suffix or ".mp4"
    target_path = item_dir / f"source{suffix}"
    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def run_whisper_cli(
    *,
    audio_path: Path,
    output_dir: Path,
    output_base: str,
    whisper_bin: str,
    model_path: Path,
    language: str,
    output_formats: list[str],
) -> None:
    format_flags = {
        "txt": "-otxt",
        "json": "-oj",
        "srt": "-osrt",
        "vtt": "-ovtt",
        "csv": "-ocsv",
    }
    command = [
        whisper_bin,
        "-m",
        str(model_path),
        "-l",
        language,
        "-f",
        str(audio_path),
        "-of",
        str(output_dir / output_base),
        "-nt",
        "-np",
    ]
    for item in dedupe_strings(output_formats):
        flag = format_flags.get(item)
        if flag:
            command.append(flag)
    subprocess.run(command, capture_output=True, text=True, check=True)


def transcribe_audio_file(
    *,
    audio_path: Path,
    output_dir: Path,
    output_base: str,
    whisper_bin: str,
    model_path: Path,
    language: str,
    output_formats: list[str],
) -> None:
    duration_ms = probe_audio_duration_ms(audio_path)
    should_chunk = duration_ms >= LONG_AUDIO_CPU_FALLBACK_MS
    chunk_model_path = model_path
    if should_chunk and model_path.name != "ggml-base.bin":
        chunk_model_path = resolve_whisper_model("base")
    try:
        if should_chunk:
            raise RuntimeError("long_audio_chunk_fallback")
        run_whisper_cli(
            audio_path=audio_path,
            output_dir=output_dir,
            output_base=output_base,
            whisper_bin=whisper_bin,
            model_path=model_path,
            language=language,
            output_formats=output_formats,
        )
        _decorate_transcript_text(output_dir / f"{output_base}.txt")
        return
    except RuntimeError as exc:
        if str(exc) != "long_audio_chunk_fallback":
            raise
    except subprocess.CalledProcessError as exc:
        if not _should_retry_with_chunked_cpu(exc, duration_ms):
            raise
    transcribe_audio_file_in_chunks(
        audio_path=audio_path,
        output_dir=output_dir,
        output_base=output_base,
        whisper_bin=whisper_bin,
        model_path=chunk_model_path,
        language="zh" if language == "auto" else language,
        output_formats=output_formats,
        duration_ms=duration_ms,
    )


def transcribe_audio_file_in_chunks(
    *,
    audio_path: Path,
    output_dir: Path,
    output_base: str,
    whisper_bin: str,
    model_path: Path,
    language: str,
    output_formats: list[str],
    duration_ms: int,
) -> None:
    chunk_count = max(1, math.ceil(duration_ms / LONG_AUDIO_CHUNK_MS))
    aggregated_texts: list[str] = []
    aggregated_segments: list[dict[str, Any]] = []
    aggregated_srt: list[str] = []
    aggregated_vtt: list[str] = ["WEBVTT", ""]
    subtitle_index = 1

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        temp_root = Path(temp_dir)
        for chunk_index in range(chunk_count):
            offset_ms = chunk_index * LONG_AUDIO_CHUNK_MS
            chunk_duration_ms = min(LONG_AUDIO_CHUNK_MS, max(duration_ms - offset_ms, 0))
            if chunk_duration_ms <= 0:
                break
            chunk_audio = temp_root / f"chunk_{chunk_index:03d}.wav"
            chunk_output_base = temp_root / f"chunk_{chunk_index:03d}" / "transcript"
            chunk_output_base.parent.mkdir(parents=True, exist_ok=True)
            extract_audio_segment(
                source_path=audio_path,
                target_path=chunk_audio,
                offset_ms=offset_ms,
                duration_ms=chunk_duration_ms,
            )
            run_whisper_cli_cpu(
                audio_path=chunk_audio,
                output_dir=chunk_output_base.parent,
                output_base=chunk_output_base.name,
                whisper_bin=whisper_bin,
                model_path=model_path,
                language=language,
                output_formats=output_formats,
            )
            chunk_txt = chunk_output_base.with_suffix(".txt")
            chunk_json = chunk_output_base.with_suffix(".json")
            chunk_srt = chunk_output_base.with_suffix(".srt")
            chunk_vtt = chunk_output_base.with_suffix(".vtt")
            if chunk_txt.exists():
                text = chunk_txt.read_text(encoding="utf-8").strip()
                if text:
                    aggregated_texts.append(text)
            if chunk_json.exists():
                aggregated_segments.extend(_load_shifted_transcription_segments(chunk_json, offset_ms))
            if chunk_srt.exists():
                subtitle_index = _append_shifted_srt(chunk_srt, aggregated_srt, offset_ms, subtitle_index)
            if chunk_vtt.exists():
                _append_shifted_vtt(chunk_vtt, aggregated_vtt, offset_ms)

    txt_path = output_dir / f"{output_base}.txt"
    txt_path.write_text("\n".join(aggregated_texts).strip() + "\n", encoding="utf-8")
    _decorate_transcript_text(txt_path)
    if "json" in output_formats:
        json_path = output_dir / f"{output_base}.json"
        json_payload = {
            "systeminfo": "chunked-cpu-fallback",
            "model": {"path": str(model_path)},
            "params": {"language": language, "chunk_ms": LONG_AUDIO_CHUNK_MS, "cpu_fallback": True},
            "result": {"language": language},
            "transcription": aggregated_segments,
        }
        json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if "srt" in output_formats:
        (output_dir / f"{output_base}.srt").write_text("\n".join(aggregated_srt).strip() + "\n", encoding="utf-8")
    if "vtt" in output_formats:
        (output_dir / f"{output_base}.vtt").write_text("\n".join(aggregated_vtt).strip() + "\n", encoding="utf-8")


def run_whisper_cli_cpu(
    *,
    audio_path: Path,
    output_dir: Path,
    output_base: str,
    whisper_bin: str,
    model_path: Path,
    language: str,
    output_formats: list[str],
) -> None:
    format_flags = {
        "txt": "-otxt",
        "json": "-oj",
        "srt": "-osrt",
        "vtt": "-ovtt",
        "csv": "-ocsv",
    }
    command = [
        whisper_bin,
        "-m",
        str(model_path),
        "-l",
        language,
        "-f",
        str(audio_path),
        "-of",
        str(output_dir / output_base),
        "-nt",
        "-np",
        "-ng",
    ]
    for item in dedupe_strings(output_formats):
        flag = format_flags.get(item)
        if flag:
            command.append(flag)
    subprocess.run(command, capture_output=True, text=True, check=True)


def extract_audio_segment(
    *,
    source_path: Path,
    target_path: Path,
    offset_ms: int,
    duration_ms: int,
) -> Path:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{offset_ms / 1000:.3f}",
        "-t",
        f"{duration_ms / 1000:.3f}",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target_path),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    return target_path


def probe_audio_duration_ms(audio_path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return round(float(completed.stdout.strip()) * 1000)


def _should_retry_with_chunked_cpu(exc: subprocess.CalledProcessError, duration_ms: int) -> bool:
    if duration_ms < LONG_AUDIO_CPU_FALLBACK_MS:
        return exc.returncode in {-11, 139}
    return exc.returncode in {-11, 139} or duration_ms >= LONG_AUDIO_CPU_FALLBACK_MS


def _load_shifted_transcription_segments(path: Path, offset_ms: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = []
    for item in payload.get("transcription", []):
        if not isinstance(item, dict):
            continue
        timestamps = item.get("timestamps", {})
        offsets = item.get("offsets", {})
        start_ms = int(offsets.get("from", 0)) + offset_ms
        end_ms = int(offsets.get("to", 0)) + offset_ms
        segments.append(
            {
                "timestamps": {
                    "from": _format_srt_timestamp(start_ms),
                    "to": _format_srt_timestamp(end_ms),
                },
                "offsets": {
                    "from": start_ms,
                    "to": end_ms,
                },
                "text": item.get("text", ""),
            }
        )
    return segments


def _append_shifted_srt(path: Path, output: list[str], offset_ms: int, next_index: int) -> int:
    blocks = [block.strip() for block in path.read_text(encoding="utf-8").split("\n\n") if block.strip()]
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        start_raw, end_raw = lines[1].split(" --> ")
        start_ms = _parse_srt_timestamp(start_raw) + offset_ms
        end_ms = _parse_srt_timestamp(end_raw) + offset_ms
        output.extend(
            [
                str(next_index),
                f"{_format_srt_timestamp(start_ms)} --> {_format_srt_timestamp(end_ms)}",
                *lines[2:],
                "",
            ]
        )
        next_index += 1
    return next_index


def _append_shifted_vtt(path: Path, output: list[str], offset_ms: int) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT":
            continue
        if " --> " in stripped:
            start_raw, end_raw = stripped.split(" --> ")
            start_ms = _parse_vtt_timestamp(start_raw) + offset_ms
            end_ms = _parse_vtt_timestamp(end_raw) + offset_ms
            output.append(f"{_format_vtt_timestamp(start_ms)} --> {_format_vtt_timestamp(end_ms)}")
        else:
            output.append(line)
    output.append("")


def _parse_srt_timestamp(value: str) -> int:
    hh, mm, rest = value.split(":")
    ss, ms = rest.split(",")
    return ((int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000) + int(ms)


def _parse_vtt_timestamp(value: str) -> int:
    hh, mm, rest = value.split(":")
    ss, ms = rest.split(".")
    return ((int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000) + int(ms)


def _format_srt_timestamp(value_ms: int) -> str:
    total_seconds, ms = divmod(value_ms, 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def _format_vtt_timestamp(value_ms: int) -> str:
    total_seconds, ms = divmod(value_ms, 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def _download_direct_media(source: str, work_dir: Path) -> Path:
    curl_bin = _which_binary("curl")
    if not curl_bin:
        raise FileNotFoundError("curl")
    target = work_dir / f"source{_infer_extension(source)}"
    command = [
        curl_bin,
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--max-time",
        os.environ.get("TRANSCRIBE_CURL_MAX_TIME", "180"),
        "-o",
        str(target),
        source,
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    return target


def _download_media_with_ytdlp(source: str, work_dir: Path) -> Path:
    yt_dlp_bin = _which_binary("yt-dlp")
    if not yt_dlp_bin:
        raise FileNotFoundError("yt-dlp")
    output_template = str(work_dir / "source.%(ext)s")
    command = [
        yt_dlp_bin,
        "--no-playlist",
        "-o",
        output_template,
        source,
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(stderr or "yt-dlp failed to download this media page.") from exc
    matches = sorted(work_dir.glob("source.*"))
    if not matches:
        raise FileNotFoundError("yt-dlp output not found")
    return matches[0]


def _which_binary(name: str) -> str | None:
    return shutil.which(name)


def _infer_source_label(source: str, index: int) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        tail = Path(parsed.path).stem or f"remote-{index:02d}"
        return f"{index:03d}_{slugify(tail)}"
    return f"{index:03d}_{slugify(Path(source).stem)}"


def _infer_extension(source: str) -> str:
    parsed = urlparse(source)
    suffix = Path(parsed.path).suffix
    return suffix if suffix else ".bin"


def _looks_like_direct_media_url(source: str) -> bool:
    lowered = source.lower()
    return any(token in lowered for token in [".mp3", ".mp4", ".m4a", ".wav", ".flac", ".aac", "/stream/"])


def _pick_media_url(media_urls: list[str]) -> str:
    for media_url in media_urls:
        lowered = media_url.lower()
        if ".mp4" in lowered or ".mp3" in lowered or ".m4a" in lowered or "/stream/" in lowered:
            return media_url
    return media_urls[0] if media_urls else ""


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def format_transcript_attribution(text: str) -> str:
    body = text.strip()
    parts = [TRANSCRIPT_ATTRIBUTION_HEADER]
    if body:
        parts.append(body)
    parts.append(TRANSCRIPT_ATTRIBUTION_FOOTER)
    return "\n\n".join(parts).strip() + "\n"


def _decorate_transcript_text(path: Path) -> None:
    if not path.exists():
        return
    path.write_text(format_transcript_attribution(path.read_text(encoding="utf-8")), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return value or "item"


def dedupe_strings(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_persisted_media(item_dir: Path) -> Path:
    matches = sorted(path for path in item_dir.glob("source.*") if path.is_file())
    return matches[0] if matches else item_dir / "source"


def _read_cached_metadata(item_dir: Path) -> dict[str, Any] | None:
    metadata_path = item_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _build_metadata(
    *,
    descriptor: SourceDescriptor,
    source_label: str,
    status: str,
    transcript_text: str,
    item_dir: Path,
    media_path: Path,
    audio_path: Path,
    created_at: str,
    error: str = "",
) -> dict[str, Any]:
    payload = {
        "source": descriptor.input,
        "source_label": source_label,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "char_count": len(transcript_text),
        "text_path": str(item_dir / "transcript.txt"),
        "json_path": str(item_dir / "transcript.json"),
        "srt_path": str(item_dir / "transcript.srt"),
        "vtt_path": str(item_dir / "transcript.vtt"),
        "media_path": str(media_path),
        "audio_path": str(audio_path),
        "output_dir": str(item_dir),
        "excerpt": transcript_text[:400],
        "error": error,
    }
    payload.update(descriptor.to_dict())
    payload["source_label"] = source_label
    payload["error"] = error or descriptor.error
    return payload


def _item_progress(base_progress: int, fraction: float, item_count: int) -> int:
    span = 80 / max(item_count, 1)
    return min(99, round(base_progress + span * fraction))


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    stage: str,
    message: str,
    progress: int,
) -> None:
    if callback:
        callback({"stage": stage, "message": message, "progress": progress})


def _find_existing_model(model_name: str, model_root: Path) -> Path | None:
    filename = f"ggml-{model_name}.bin"
    candidate_roots: list[Path] = []
    for env_name in ("TRANSCRIBE_MODEL_DIR", "WHISPER_MODEL_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidate_roots.append(Path(env_value).expanduser())
    candidate_roots.extend([model_root, *KNOWN_MODEL_ROOTS])
    seen: set[Path] = set()
    for root in candidate_roots:
        resolved_root = root.expanduser()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        candidate = resolved_root / filename
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate.resolve()
    return None


def _coerce_descriptors(sources: list[str | SourceDescriptor]) -> list[SourceDescriptor]:
    raw_inputs = [item for item in sources if isinstance(item, str)]
    resolved_inputs = iter(resolve_sources(raw_inputs)) if raw_inputs else iter(())
    descriptors: list[SourceDescriptor] = []
    for item in sources:
        if isinstance(item, SourceDescriptor):
            descriptors.append(item)
        else:
            descriptors.append(next(resolved_inputs))
    return descriptors
