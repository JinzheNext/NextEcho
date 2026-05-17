from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse


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


def transcribe_media_sources(
    sources: list[str],
    *,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    language: str = "zh",
    max_seconds: int = DEFAULT_MAX_SECONDS,
    output_formats: list[str] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
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
        "source_count": len(sources),
    }
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    results: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        base_progress = 15 + int(((index - 1) / max(len(sources), 1)) * 80)
        result = transcribe_single_source(
            source=source,
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
            item_count=len(sources),
        )
        results.append(result)

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
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit_progress(progress_callback, stage="complete", message="转写完成", progress=100)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "run_config_path": run_config_path,
        "results": results,
    }


def transcribe_single_source(
    *,
    source: str,
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
    source_label = _infer_source_label(source, index)
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
            source=source,
            source_label=source_label,
            status="cached",
            transcript_text=_safe_read_text(transcript_path),
            item_dir=item_dir,
            media_path=_find_persisted_media(item_dir),
            audio_path=item_dir / "audio.wav",
            created_at=now_iso(),
        )

    with tempfile.TemporaryDirectory(dir=item_dir) as temp_dir:
        temp_path = Path(temp_dir)
        _emit_progress(progress_callback, stage="media", message=f"{source_label} 获取媒体", progress=_item_progress(base_progress, 0.1, item_count))
        media_path = prepare_media_source(source, temp_path)
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
        run_whisper_cli(
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
        source=source,
        source_label=source_label,
        status="ready" if transcript_text.strip() else "empty",
        transcript_text=transcript_text,
        item_dir=item_dir,
        media_path=persisted_media_path,
        audio_path=audio_path,
        created_at=now_iso(),
    )
    _write_json(item_dir / "metadata.json", metadata)
    _emit_progress(progress_callback, stage="item_complete", message=f"{source_label} 已完成", progress=_item_progress(base_progress, 1.0, item_count))
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


def prepare_media_source(source: str, work_dir: Path) -> Path:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        if _looks_like_direct_media_url(source):
            return _download_direct_media(source, work_dir)
        return _download_media_with_ytdlp(source, work_dir)
    local_path = Path(source).expanduser().resolve()
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
    subprocess.run(command, capture_output=True, text=True, check=True)
    matches = sorted(work_dir.glob("source.*"))
    if not matches:
        raise FileNotFoundError("yt-dlp output not found")
    return matches[0]


def _which_binary(name: str) -> str | None:
    completed = subprocess.run(
        f"zsh -ic {shlex.quote(f'command -v {name}')}",
        shell=True,
        capture_output=True,
        text=True,
    )
    result = completed.stdout.strip()
    return result or None


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
    source: str,
    source_label: str,
    status: str,
    transcript_text: str,
    item_dir: Path,
    media_path: Path,
    audio_path: Path,
    created_at: str,
) -> dict[str, Any]:
    return {
        "source": source,
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
        "error": "",
    }


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
