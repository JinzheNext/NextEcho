from __future__ import annotations

import json
import os
import math
import shutil
import subprocess
import tempfile
import time
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from .asr import TranscriptResult, choose_asr_engine, qwen_model_for_quality, transcribe_qwen_mlx
from .config import NextEchoConfig, load_config
from .episode_context import crawl_episode_context
from .enhancement import enhance_transcript_artifacts
from .sources import SourceDescriptor, build_ytdlp_command, detect_platform, resolve_sources
from .binaries import find_binary


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
# Keep chunks large enough to reduce boundary artefacts.  The overlap is
# removed deterministically when the chunk outputs are merged, so it is only
# a temporary ASR context window and never duplicated in final artifacts.
LONG_AUDIO_CHUNK_OVERLAP_MS = 30 * 1000
LONG_AUDIO_CPU_FALLBACK_MS = 15 * 60 * 1000
PERSISTED_MEDIA_BASENAME = "录音原声"
PERSISTED_AUDIO_NAME = "转录音频.wav"
PROJECT_NAME = "NextEcho"
PROJECT_TAGLINE = "让播客自动转录进知识库"
TRANSCRIPT_ATTRIBUTION_HEADER = "本文转录由 GitHub 项目：NextEcho 提供支持，作者 @金哲Next（小红书、公众号同名）"
TRANSCRIPT_ATTRIBUTION_FOOTER = "powered by github.com/JinzheNext/NextEcho - 让播客自动转录进知识库"
WHISPER_MODEL_SPECS = {
    "tiny": {
        "tier": "smallest",
        "summary": "最快、最省资源，适合老机器或快速预览。",
        "min_memory_gb": 4.0,
    },
    "base": {
        "tier": "fast",
        "summary": "轻量稳定，适合 8GB 左右内存和日常批量转写。",
        "min_memory_gb": 6.0,
    },
    "small": {
        "tier": "balanced",
        "summary": "精度和速度更平衡，适合 8GB 到 16GB 机器。",
        "min_memory_gb": 8.0,
    },
    "medium": {
        "tier": "quality",
        "summary": "更高精度，适合 16GB 以上机器。",
        "min_memory_gb": 16.0,
    },
    "large-v3-turbo-q5_0": {
        "tier": "best",
        "summary": "默认高精度方案，适合追求质量的现代机器。",
        "min_memory_gb": 16.0,
    },
}


def transcribe_media_sources(
    sources: list[str | SourceDescriptor],
    *,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    language: str = "zh",
    max_seconds: int = DEFAULT_MAX_SECONDS,
    output_formats: list[str] | None = None,
    glossary_text: str = "",
    enable_enhancement: bool = True,
    ai_polish: bool = False,
    ai_style: str = "",
    asr_engine: str = "auto",
    asr_model: str = "",
    postprocess_profile: str = "off",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    config: NextEchoConfig | None = None,
) -> dict[str, Any]:
    started_at = now_iso()
    started_monotonic = time.monotonic()
    config_obj = config or load_config()
    descriptors = _coerce_descriptors(sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    auto_qwen_model = asr_model or qwen_model_for_quality("fast" if model_name == "base" else "accurate", config_obj)
    selected_asr_engine = choose_asr_engine(asr_engine, model=auto_qwen_model, config=config_obj)
    _emit_progress(progress_callback, stage="model", message="准备模型", progress=5)
    # Whisper's model and executable are intentionally resolved only when the
    # selected path needs them.  A successful Qwen run must not require a
    # Whisper installation or trigger a model download.
    model_path: Path | None = None
    whisper_bin: str | None = None
    if selected_asr_engine == "whisper-cpp":
        model_path = resolve_whisper_model(model_name)
        _emit_progress(progress_callback, stage="model_ready", message="模型已就绪", progress=15)
        whisper_bin = _which_binary("whisper-cli")
    ffmpeg_bin = _which_binary("ffmpeg")
    if not ffmpeg_bin or (selected_asr_engine == "whisper-cpp" and not whisper_bin):
        missing = [name for name, value in [("ffmpeg", ffmpeg_bin), ("whisper-cli", whisper_bin)] if not value]
        raise FileNotFoundError(f"Missing local dependencies: {', '.join(missing)}")

    formats = dedupe_strings(output_formats or DEFAULT_OUTPUT_FORMATS)
    generated_at = now_iso()
    reported_model_name = (asr_model or auto_qwen_model) if selected_asr_engine == "qwen3-asr" else model_name
    reported_model_path = "" if selected_asr_engine == "qwen3-asr" else str(model_path or "")
    run_config = {
        "started_at": started_at,
        "completed_at": "",
        "elapsed_seconds": 0,
        "generated_at": generated_at,
        "model_name": reported_model_name,
        "model_path": reported_model_path,
        "whisper_model_name": model_name,
        "whisper_model_path": str(model_path or ""),
        "schema_version": 2,
        "asr": {
            "requested_engine": asr_engine,
            "selected_engine": selected_asr_engine,
            "requested_model": asr_model,
            "candidate_model": auto_qwen_model,
        },
        "config": {
            "source": config_obj.source,
            "source_label": config_obj.source_label,
            "source_paths": list(config_obj.source_paths),
        },
        "language": language,
        "max_seconds": max_seconds,
        "output_formats": formats,
        "enhancement": {
            "enabled": enable_enhancement,
            "ai_polish": ai_polish,
            "ai_style": ai_style,
            "postprocess_profile": postprocess_profile,
            "user_glossary_supplied": bool(glossary_text.strip()),
        },
        "postprocess": {"profile": postprocess_profile, "enabled": postprocess_profile != "off"},
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
            glossary_text=glossary_text,
            enable_enhancement=enable_enhancement,
            ai_polish=ai_polish,
            ai_style=ai_style,
            asr_engine=selected_asr_engine,
            asr_model=asr_model,
            postprocess_profile=postprocess_profile,
            config=config_obj,
            whisper_model_name=model_name,
            index=index,
            progress_callback=progress_callback,
            base_progress=base_progress,
            item_count=len(descriptors),
        )
        results.append(result)
    failed_results = [item for item in results if item.get("status") == "failed"]
    completed_at = now_iso()
    elapsed_seconds = max(0, round(time.monotonic() - started_monotonic))
    run_config["completed_at"] = completed_at
    run_config["elapsed_seconds"] = elapsed_seconds
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "generated_at": generated_at,
        "model_name": reported_model_name,
        "model_path": reported_model_path,
        "whisper_model_name": model_name,
        "whisper_model_path": str(model_path or ""),
        "schema_version": 2,
        "asr": run_config["asr"],
        "language": language,
        "max_seconds": max_seconds,
        "output_formats": formats,
        "enhancement": run_config["enhancement"],
        "postprocess": {"profile": postprocess_profile, "enabled": postprocess_profile != "off"},
        "fallbacks": [
            {"source_label": item.get("source_label", ""), **fallback}
            for item in results
            for fallback in (item.get("asr", {}).get("fallbacks") or [])
        ],
        "timings": {"total_elapsed_seconds": elapsed_seconds},
        "artifact_paths": {
            "items_dir": str(items_dir),
            "manifest": str(output_dir / "manifest.json"),
        },
        "run_config_path": str(run_config_path),
        "items_dir": str(items_dir),
        "results": results,
        "error_count": len(failed_results),
        "partial_count": sum(1 for item in results if (item.get("enhancement") or {}).get("status") == "partial"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_path = None
    if failed_results:
        errors_path = output_dir / "errors.json"
        errors_path.write_text(json.dumps({"generated_at": generated_at, "items": failed_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit_progress(progress_callback, stage="complete", message="转写完成", progress=100)
    return {
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "run_config_path": run_config_path,
        "results": results,
        "errors_path": errors_path,
    }


def list_supported_whisper_models() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spec in WHISPER_MODEL_SPECS.items():
        rows.append(
            {
                "name": name,
                "tier": spec["tier"],
                "summary": spec["summary"],
                "min_memory_gb": spec["min_memory_gb"],
            }
        )
    return rows


def recommend_whisper_model(memory_gb: float | None) -> tuple[str, str]:
    if memory_gb is None:
        return "base", "无法读取内存，建议先从 base 开始；如果机器表现稳定，再升级到更大的模型。"
    if memory_gb < 6:
        return "tiny", "内存低于 6GB，建议先用 tiny。"
    if memory_gb < 8:
        return "base", "内存低于 8GB，建议先用 base。"
    if memory_gb < 16:
        return "small", "内存介于 8GB 到 16GB，建议先用 small 或 base。"
    if memory_gb < 24:
        return "medium", "内存介于 16GB 到 24GB，建议先用 medium。"
    return DEFAULT_MODEL_NAME, "内存充足，建议直接使用默认高精度模型。"


def resolve_requested_model(model_name: str | None, quality: str, *, memory_gb: float | None = None) -> str:
    if model_name:
        return model_name
    if quality == "fast":
        return "base"
    if quality == "accurate":
        recommended_model, _ = recommend_whisper_model(memory_gb)
        return DEFAULT_MODEL_NAME if recommended_model in {"medium", DEFAULT_MODEL_NAME} else recommended_model
    return DEFAULT_MODEL_NAME


def transcribe_single_source(
    *,
    descriptor: SourceDescriptor,
    items_dir: Path,
    model_path: Path | None,
    whisper_bin: str | None,
    ffmpeg_bin: str,
    language: str,
    max_seconds: int,
    output_formats: list[str],
    glossary_text: str,
    enable_enhancement: bool,
    ai_polish: bool,
    ai_style: str,
    index: int,
    asr_engine: str = "whisper-cpp",
    asr_model: str = "",
    postprocess_profile: str = "off",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    base_progress: int = 15,
    item_count: int = 1,
    config: NextEchoConfig | None = None,
    whisper_model_name: str = "",
) -> dict[str, Any]:
    source_label = descriptor.source_label or _infer_source_label(descriptor.input, index)
    item_dir = items_dir / source_label
    item_dir.mkdir(parents=True, exist_ok=True)
    source_context: dict[str, Any] = {}
    transcript_path = item_dir / "transcript.txt"
    if transcript_path.exists():
        # A previous run may predate episode-context collection.  Refresh only
        # the lightweight page metadata on cache hits; never rerun media/ASR.
        if descriptor.platform == "xiaoyuzhou" and descriptor.canonical_url:
            context_path = item_dir / "source.context.json"
            if context_path.exists():
                try:
                    cached_context = json.loads(context_path.read_text(encoding="utf-8"))
                    if isinstance(cached_context, dict):
                        source_context = cached_context
                except (OSError, json.JSONDecodeError):
                    pass
            else:
                try:
                    source_context = crawl_episode_context(descriptor.canonical_url, item_dir)
                except Exception:
                    source_context = {}
        cached_metadata = _read_cached_metadata(item_dir)
        if cached_metadata:
            cached_metadata["status"] = "cached"
            cached_metadata["updated_at"] = now_iso()
            if source_context:
                cached_metadata["source_context"] = source_context
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
            audio_path=_find_persisted_audio(item_dir),
            created_at=now_iso(),
            source_context=source_context,
        )
    if descriptor.error and descriptor.source_type != "local_file" and not descriptor.resolved_media_url:
        metadata = _build_metadata(
            descriptor=descriptor,
            source_label=source_label,
            status="failed",
            transcript_text="",
            item_dir=item_dir,
            media_path=_find_persisted_media(item_dir),
            audio_path=_find_persisted_audio(item_dir),
            created_at=now_iso(),
            error=descriptor.error,
        )
        _write_json(item_dir / "metadata.json", metadata)
        _emit_progress(progress_callback, stage="item_failed", message=f"{source_label} 失败：{descriptor.error}", progress=_item_progress(base_progress, 1.0, item_count))
        return metadata

    try:
        if descriptor.platform == "xiaoyuzhou" and descriptor.canonical_url:
            _emit_progress(progress_callback, stage="source_context", message=f"{source_label} 采集节目资料", progress=_item_progress(base_progress, 0.03, item_count))
            try:
                source_context = crawl_episode_context(descriptor.canonical_url, item_dir)
            except Exception as exc:
                # Crawling enriches the run but must never make an otherwise
                # transcribable episode fail.
                source_context = {
                    "schema_version": 1,
                    "status": "failed",
                    "url": descriptor.canonical_url,
                    "provider": "none",
                    "error": str(exc),
                    "speaker_candidates": [],
                    "comments": [],
                }
        with tempfile.TemporaryDirectory(prefix="临时下载-", dir=item_dir) as temp_dir:
            temp_path = Path(temp_dir)
            _emit_progress(progress_callback, stage="media", message=f"{source_label} 获取媒体", progress=_item_progress(base_progress, 0.1, item_count))
            media_path = prepare_media_source(descriptor, temp_path)
            persisted_media_path = _persist_media(media_path, item_dir)
            audio_path = item_dir / PERSISTED_AUDIO_NAME
            _emit_progress(progress_callback, stage="audio", message=f"{source_label} 抽取音频", progress=_item_progress(base_progress, 0.35, item_count))
            extract_audio(
                source_path=persisted_media_path,
                target_path=audio_path,
                ffmpeg_bin=ffmpeg_bin,
                max_seconds=max_seconds,
            )
            _emit_progress(progress_callback, stage="asr", message=f"{source_label} 正在识别", progress=_item_progress(base_progress, 0.55, item_count))
            asr_info = transcribe_audio_with_fallback(
                audio_path=audio_path,
                output_dir=item_dir,
                output_base="transcript",
                whisper_bin=whisper_bin,
                whisper_model_path=model_path,
                whisper_model_name=whisper_model_name or (model_path.name.removeprefix("ggml-").removesuffix(".bin") if model_path else DEFAULT_MODEL_NAME),
                language=language,
                output_formats=output_formats,
                engine=asr_engine,
                model=asr_model or qwen_model_for_quality(
                    "fast" if (model_path and model_path.name == "ggml-base.bin") or whisper_model_name == "base" else "accurate",
                    config,
                ),
                config=config,
            )
        transcript_text = transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else ""
        enhancement = _empty_enhancement_payload()
        if enable_enhancement and transcript_text.strip():
            if postprocess_profile != "off":
                from .postprocess import process_transcript_artifacts

                enhancement = process_transcript_artifacts(
                    item_dir,
                    profile=postprocess_profile,
                    user_glossary_text=glossary_text,
                    style=ai_style,
                    progress_callback=progress_callback,
                )
            else:
                enhancement = enhance_transcript_artifacts(
                    item_dir,
                    user_glossary_text=glossary_text,
                    ai_polish=ai_polish,
                    ai_style=ai_style,
                )
        metadata = _build_metadata(
            descriptor=descriptor,
            source_label=source_label,
            status="ready" if transcript_text.strip() else "empty",
            transcript_text=transcript_text,
            item_dir=item_dir,
            media_path=persisted_media_path,
            audio_path=audio_path,
            created_at=now_iso(),
            enhancement=enhancement,
            asr=asr_info,
            source_context=source_context,
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
            audio_path=_find_persisted_audio(item_dir),
            created_at=now_iso(),
            error=str(exc),
            enhancement=_empty_enhancement_payload(),
            asr={"requested_engine": asr_engine, "actual_engine": "", "fallbacks": []},
            source_context=source_context,
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
        if descriptor.resolver in {"direct", "xiaoyuzhou_page", "apple_podcasts_page", "rss"} and descriptor.resolved_media_url:
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
    target_path = item_dir / f"{PERSISTED_MEDIA_BASENAME}{suffix}"
    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def transcribe_audio_with_fallback(
    *,
    audio_path: Path,
    output_dir: Path,
    output_base: str,
    whisper_bin: str | None,
    whisper_model_path: Path | None,
    whisper_model_name: str,
    language: str,
    output_formats: list[str],
    engine: str,
    model: str,
    config: NextEchoConfig | None = None,
) -> dict[str, Any]:
    """Prefer Qwen on supported Macs, lazily preparing Whisper on fallback."""
    config_obj = config or load_config()
    fallbacks: list[dict[str, str]] = []
    if engine == "qwen3-asr":
        qwen_candidates = [model]
        fast_model = qwen_model_for_quality("fast", config_obj)
        if model != fast_model:
            qwen_candidates.append(fast_model)
        for candidate in qwen_candidates:
            try:
                result = transcribe_qwen_mlx(
                    audio_path=audio_path,
                    item_dir=output_dir,
                    model=candidate,
                    language=language,
                    config=config_obj,
                )
                write_qwen_transcript_result(result, output_dir=output_dir, output_base=output_base, output_formats=output_formats)
                return {
                    "requested_engine": engine,
                    "requested_model": model,
                    "actual_engine": "qwen3-asr",
                    "actual_model": result.model,
                    "version": result.version,
                    "fallbacks": fallbacks,
                    "resources": result.resources,
                }
            except Exception as exc:
                fallbacks.append({"from": candidate, "to": "next", "reason": str(exc), "at": now_iso()})
    # Do not resolve a Whisper model or executable before this point.  This is
    # important for Qwen-only installations and prevents a hidden download or
    # dependency failure from contaminating the successful Qwen path.
    if not whisper_bin:
        whisper_bin = _which_binary("whisper-cli")
    if not whisper_bin:
        raise FileNotFoundError("Missing local dependency: whisper-cli (required only for ASR fallback)")
    if whisper_model_path is None:
        whisper_name = whisper_model_name or str(config_obj.get_path("asr", "whisper_fallback", default=DEFAULT_MODEL_NAME))
        whisper_model_path = resolve_whisper_model(whisper_name)
    try:
        transcribe_audio_file(
            audio_path=audio_path,
            output_dir=output_dir,
            output_base=output_base,
            whisper_bin=whisper_bin,
            model_path=whisper_model_path,
            language=language,
            output_formats=output_formats,
        )
        annotate_whisper_transcript(output_dir / f"{output_base}.json", model=whisper_model_name, language=language)
        return {
            "requested_engine": engine,
            "requested_model": model,
            "actual_engine": "whisper-cpp",
            "actual_model": whisper_model_name,
            "version": "",
            "fallbacks": fallbacks,
        }
    except Exception as exc:
        if whisper_model_name != "base":
            fallbacks.append({"from": whisper_model_name, "to": "base", "reason": str(exc), "at": now_iso()})
            base_path = resolve_whisper_model("base")
            transcribe_audio_file(
                audio_path=audio_path,
                output_dir=output_dir,
                output_base=output_base,
                whisper_bin=whisper_bin,
                model_path=base_path,
                language=language,
                output_formats=output_formats,
            )
            annotate_whisper_transcript(output_dir / f"{output_base}.json", model="base", language=language)
            return {
                "requested_engine": engine,
                "requested_model": model,
                "actual_engine": "whisper-cpp",
                "actual_model": "base",
                "version": "",
                "fallbacks": fallbacks,
            }
        raise


def annotate_whisper_transcript(path: Path, *, model: str, language: str) -> None:
    """Upgrade whisper.cpp JSON in place while retaining its original fields."""
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    payload["engine"] = {
        "backend": "whisper-cpp",
        "model": model,
        "version": "",
    }
    payload.setdefault("result", {})["language"] = payload.get("result", {}).get("language") or language
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_qwen_transcript_result(
    result: TranscriptResult,
    *,
    output_dir: Path,
    output_base: str,
    output_formats: list[str],
) -> None:
    segments = aggregate_qwen_words(result.words, fallback_text=result.text)
    transcript = [
        {
            "id": f"seg_{index:05d}",
            "text": item["text"],
            "offsets": {"from": item["start_ms"], "to": item["end_ms"]},
            "timestamps": {"from": _format_srt_timestamp(item["start_ms"]), "to": _format_srt_timestamp(item["end_ms"])},
        }
        for index, item in enumerate(segments)
    ]
    text = "\n".join(item["text"] for item in segments).strip() or result.text.strip()
    txt_path = output_dir / f"{output_base}.txt"
    txt_path.write_text(format_transcript_attribution(text), encoding="utf-8")
    if "json" in output_formats:
        (output_dir / f"{output_base}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "engine": {
                        "backend": result.backend,
                        "model": result.model,
                        "version": result.version,
                        **({"resources": result.resources} if result.resources else {}),
                        **({"provenance": result.provenance} if result.provenance else {}),
                    },
                    "result": {"language": result.language},
                    "transcription": transcript,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if "srt" in output_formats or "vtt" in output_formats:
        write_subtitles_from_segments(
            transcript,
            srt_path=output_dir / f"{output_base}.srt" if "srt" in output_formats else None,
            vtt_path=output_dir / f"{output_base}.vtt" if "vtt" in output_formats else None,
        )


def aggregate_qwen_words(words: list[dict[str, Any]], *, fallback_text: str) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    current: list[str] = []
    start_ms = 0
    end_ms = 0
    for word in words:
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        word_start = round(float(word.get("start") or 0) * 1000)
        word_end = round(float(word.get("end") or 0) * 1000)
        if not current:
            start_ms = word_start
        current.append(text)
        end_ms = max(word_end, word_start)
        joined = "".join(current)
        if text.endswith(("。", "？", "！", ".", "?", "!")) or end_ms - start_ms >= 15_000 or len(joined) >= 50:
            grouped.append({"text": joined, "start_ms": start_ms, "end_ms": end_ms})
            current = []
    if current:
        grouped.append({"text": "".join(current), "start_ms": start_ms, "end_ms": end_ms})
    if not grouped and fallback_text.strip():
        grouped.append({"text": fallback_text.strip(), "start_ms": 0, "end_ms": 0})
    return grouped


def write_subtitles_from_segments(segments: Iterable[Any], *, srt_path: Path | None, vtt_path: Path | None) -> None:
    rows = list(segments)
    if srt_path is not None:
        srt: list[str] = []
        for index, item in enumerate(rows, start=1):
            text, start_ms, end_ms = _segment_values(item)
            srt.extend([str(index), f"{_format_srt_timestamp(start_ms)} --> {_format_srt_timestamp(max(end_ms, start_ms + 1))}", text, ""])
        srt_path.write_text("\n".join(srt).strip() + "\n", encoding="utf-8")
    if vtt_path is not None:
        vtt: list[str] = ["WEBVTT", ""]
        for item in rows:
            text, start_ms, end_ms = _segment_values(item)
            vtt.extend([f"{_format_vtt_timestamp(start_ms)} --> {_format_vtt_timestamp(max(end_ms, start_ms + 1))}", text, ""])
        vtt_path.write_text("\n".join(vtt).strip() + "\n", encoding="utf-8")


def _segment_values(item: Any) -> tuple[str, int, int]:
    if isinstance(item, dict):
        offsets = item.get("offsets") or {}
        return str(item.get("text") or ""), int(offsets.get("from") or item.get("start_ms") or 0), int(offsets.get("to") or item.get("end_ms") or 0)
    return str(item.text), int(item.start_ms), int(item.end_ms)


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

    with tempfile.TemporaryDirectory(prefix="临时分片-", dir=output_dir) as temp_dir:
        temp_root = Path(temp_dir)
        for chunk_index in range(chunk_count):
            trim_before_ms = chunk_index * LONG_AUDIO_CHUNK_MS
            offset_ms = max(0, trim_before_ms - LONG_AUDIO_CHUNK_OVERLAP_MS)
            chunk_duration_ms = min(
                LONG_AUDIO_CHUNK_MS + (LONG_AUDIO_CHUNK_OVERLAP_MS if chunk_index else 0),
                max(duration_ms - offset_ms, 0),
            )
            if chunk_duration_ms <= 0:
                break
            chunk_audio = temp_root / f"片段-{chunk_index + 1:03d}.wav"
            chunk_output_base = temp_root / f"片段-{chunk_index + 1:03d}" / "转录"
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
            if chunk_json.exists():
                chunk_segments = _load_shifted_transcription_segments(
                    chunk_json,
                    offset_ms,
                    trim_before_ms=trim_before_ms,
                )
                aggregated_segments.extend(chunk_segments)
                chunk_text = "\n".join(str(item.get("text", "")).strip() for item in chunk_segments).strip()
                if chunk_text:
                    aggregated_texts.append(chunk_text)
            elif chunk_txt.exists():
                text = chunk_txt.read_text(encoding="utf-8").strip()
                if text:
                    aggregated_texts.append(text)
            if chunk_srt.exists():
                subtitle_index = _append_shifted_srt(
                    chunk_srt,
                    aggregated_srt,
                    offset_ms,
                    subtitle_index,
                    trim_before_ms=trim_before_ms,
                )
            if chunk_vtt.exists():
                _append_shifted_vtt(chunk_vtt, aggregated_vtt, offset_ms, trim_before_ms=trim_before_ms)

    txt_path = output_dir / f"{output_base}.txt"
    txt_path.write_text("\n".join(aggregated_texts).strip() + "\n", encoding="utf-8")
    _decorate_transcript_text(txt_path)
    if "json" in output_formats:
        json_path = output_dir / f"{output_base}.json"
        json_payload = {
            "systeminfo": "chunked-cpu-fallback",
            "model": {"path": str(model_path)},
            "params": {
                "language": language,
                "chunk_ms": LONG_AUDIO_CHUNK_MS,
                "overlap_ms": LONG_AUDIO_CHUNK_OVERLAP_MS,
                "cpu_fallback": True,
            },
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


def _load_shifted_transcription_segments(
    path: Path,
    offset_ms: int,
    *,
    trim_before_ms: int = 0,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = []
    for item in payload.get("transcription", []):
        if not isinstance(item, dict):
            continue
        timestamps = item.get("timestamps", {})
        offsets = item.get("offsets", {})
        start_ms = int(offsets.get("from", 0)) + offset_ms
        end_ms = int(offsets.get("to", 0)) + offset_ms
        # The previous chunk owns the boundary.  Dropping every segment that
        # starts inside the overlap prevents duplicate sentences while the
        # 30-second look-back still gives Whisper enough context to finish a
        # sentence that happens to straddle the cut.
        if trim_before_ms and start_ms < trim_before_ms:
            continue
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


def _append_shifted_srt(
    path: Path,
    output: list[str],
    offset_ms: int,
    next_index: int,
    *,
    trim_before_ms: int = 0,
) -> int:
    blocks = [block.strip() for block in path.read_text(encoding="utf-8").split("\n\n") if block.strip()]
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        start_raw, end_raw = lines[1].split(" --> ")
        start_ms = _parse_srt_timestamp(start_raw) + offset_ms
        end_ms = _parse_srt_timestamp(end_raw) + offset_ms
        if trim_before_ms and start_ms < trim_before_ms:
            continue
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


def _append_shifted_vtt(
    path: Path,
    output: list[str],
    offset_ms: int,
    *,
    trim_before_ms: int = 0,
) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT":
            continue
        if " --> " in stripped:
            start_raw, end_raw = stripped.split(" --> ")
            start_ms = _parse_vtt_timestamp(start_raw) + offset_ms
            end_ms = _parse_vtt_timestamp(end_raw) + offset_ms
            if trim_before_ms and start_ms < trim_before_ms:
                continue
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
    target = work_dir / f"{PERSISTED_MEDIA_BASENAME}{_infer_extension(source)}"
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
    output_template = str(work_dir / f"{PERSISTED_MEDIA_BASENAME}.%(ext)s")
    platform = detect_platform(source)
    try:
        base_command = build_ytdlp_command(
            source,
            platform=platform,
            command_kind="download",
            output_template=output_template,
        )
    except FileNotFoundError:
        raise FileNotFoundError("yt-dlp")
    commands = [base_command]
    configured_browser = os.environ.get("TRANSCRIBE_YTDLP_BROWSER", "").strip()
    if platform in {"bilibili", "youtube"}:
        if platform == "youtube" and sys.platform == "darwin":
            browsers = ["firefox", "chrome", "brave", "edge"]
        elif platform == "youtube" and os.name == "nt":
            browsers = ["firefox", "chrome", "brave", "edge"]
        elif platform == "youtube":
            browsers = ["firefox", "chrome", "chromium", "brave", "edge"]
        elif sys.platform == "darwin":
            browsers = ["chrome", "edge", "firefox", "brave"]
        elif os.name == "nt":
            browsers = ["edge", "chrome", "firefox", "brave"]
        else:
            browsers = ["firefox", "chrome", "chromium", "edge", "brave"]
        if configured_browser:
            browsers = [configured_browser] + [item for item in browsers if item != configured_browser]
        for browser in browsers:
            commands.append(
                build_ytdlp_command(
                    source,
                    platform=platform,
                    command_kind="download",
                    browser=browser,
                    output_template=output_template,
                )
            )
    last_stderr = ""
    for attempt_index, command in enumerate(commands):
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
            last_stderr = ""
            break
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            last_stderr = stderr or "yt-dlp failed to download this media page."
            lower = last_stderr.lower()
            if attempt_index == 0 and platform in {"bilibili", "youtube"} and any(token in lower for token in ["http error 412", "precondition failed", "http error 403", "cloudflare", "captcha", "sign in to confirm you’re not a bot", "sign in to confirm you're not a bot", "confirm you're not a bot"]) and len(commands) > 1:
                continue
            if attempt_index + 1 < len(commands):
                continue
            raise RuntimeError(last_stderr) from exc
    if last_stderr:
        raise RuntimeError(last_stderr)
    matches = sorted(work_dir.glob(f"{PERSISTED_MEDIA_BASENAME}.*"))
    if not matches:
        raise FileNotFoundError("yt-dlp output not found")
    return matches[0]


def _which_binary(name: str) -> str | None:
    return find_binary(name)


def _infer_source_label(source: str, index: int) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        tail = Path(parsed.path).stem or f"remote-{index:02d}"
        return f"{index:03d}_{slugify(tail)}"
    stem = Path(source).stem
    if stem.lower() in {"source", "audio", "input", "media"}:
        stem = PERSISTED_MEDIA_BASENAME
    return f"{index:03d}_{slugify(stem)}"


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
    for basename in (PERSISTED_MEDIA_BASENAME, "source"):
        matches = sorted(path for path in item_dir.glob(f"{basename}.*") if path.is_file())
        if matches:
            return matches[0]
    return item_dir / PERSISTED_MEDIA_BASENAME


def _find_persisted_audio(item_dir: Path) -> Path:
    current = item_dir / PERSISTED_AUDIO_NAME
    if current.exists():
        return current
    # Keep cached runs created before the naming migration readable.
    legacy = item_dir / "audio.wav"
    return legacy if legacy.exists() else current


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
    enhancement: dict[str, Any] | None = None,
    asr: dict[str, Any] | None = None,
    source_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enhancement = enhancement or _empty_enhancement_payload()
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
        "enhanced_text_path": str(enhancement.get("enhanced_text_path") or ""),
        "enhanced_json_path": str(enhancement.get("enhanced_json_path") or ""),
        "enhanced_srt_path": str(enhancement.get("enhanced_srt_path") or ""),
        "enhanced_vtt_path": str(enhancement.get("enhanced_vtt_path") or ""),
        "summary_json_path": str(enhancement.get("summary_json_path") or ""),
        "knowledge_markdown_path": str(enhancement.get("knowledge_markdown_path") or ""),
        "enhancement": enhancement,
        "asr": asr or {},
        "source_context": source_context or {},
        "output_dir": str(item_dir),
        "excerpt": transcript_text[:400],
        "error": error,
    }
    payload.update(descriptor.to_dict())
    payload["source_label"] = source_label
    payload["error"] = error or descriptor.error
    return payload


def _empty_enhancement_payload() -> dict[str, Any]:
    return {
        "enabled": False,
        "enhanced_text_path": "",
        "enhanced_json_path": "",
        "rules_applied": [],
        "ai": {"requested": False, "applied": False, "reason": ""},
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
