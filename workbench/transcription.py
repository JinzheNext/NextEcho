from __future__ import annotations

from .io_utils import atomic_write, redact, redact_structure

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
from .resources import serialized_task
from .media_commands import run_media_command
from .contracts import PipelineError
from .io_utils import file_hash, value_hash
from .whisper_models import cached_model, find_model, validate_whisper_model


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRANSCRIPTION_ROOT = ROOT / "outputs" / "transcriptions"
DEFAULT_MODEL_ROOT = ROOT / "models" / "whisper.cpp"
KNOWN_MODEL_ROOTS = [
    Path.home() / "Documents" / "jerry_projects" / "AI-hot-topic" / "models" / "whisper.cpp",
]
DEFAULT_MODEL_NAME = "large-v3-turbo-q5_0"
DEFAULT_MAX_SECONDS = 0
DEFAULT_OUTPUT_FORMATS = ["txt", "json", "srt", "vtt"]
LONG_AUDIO_CHUNK_MS = 10 * 60 * 1000
# Prior-only context is owned by the previous chunk. Crossing hypotheses
# retain their complete native range and require review for possible repetition.
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


@serialized_task
def transcribe_media_sources(sources, *, speaker_stage=None, speaker_mode="auto", defer_delivery=False, delivery_job_id="", native_speaker_alignment=False, **kwargs):
    """Compatibility entry routed through the shared ordered pipeline stages."""
    from .pipeline import run_post_asr_stages
    from .publication import invalidate_publication, publish_pipeline, begin_pipeline, mark_pipeline_stage, fail_pipeline
    started = time.monotonic()
    destination = Path(kwargs["output_dir"])
    if not defer_delivery or not (destination / "pipeline.state.json").is_file():
        begin_pipeline(destination, job_id=delivery_job_id or destination.name)
    else:
        invalidate_publication(destination)
    mark_pipeline_stage(destination, "asr")
    try:
        payload = _transcribe_media_sources(sources, _defer_enhancement=True, **kwargs)
        speaker_ran = False
        def run_speaker(current):
            nonlocal speaker_ran
            speaker_ran = True
            mark_pipeline_stage(destination, "speaker")
            return speaker_stage(current)
        def text_stage():
            mark_pipeline_stage(destination, "polish")
            # Speaker compatibility adapters may annotate the persisted manifest.
            # Merge those fields before text processing and final aggregation.
            manifest_path = Path(payload["manifest_path"])
            if speaker_ran and manifest_path.is_file():
                persisted = json.loads(manifest_path.read_text()).get("results", [])
                if len(persisted) != len(payload["results"]):
                    raise PipelineError("PROTOCOL_ERROR", "speaker阶段改变了来源条目数")
                for item, saved in zip(payload["results"],persisted):
                    if item.get("source_label") != saved.get("source_label") or item.get("output_dir") != saved.get("output_dir"):
                        raise PipelineError("PROTOCOL_ERROR", "speaker阶段改变了来源条目身份或顺序")
                    item.update(saved)
            if not kwargs.get("enable_enhancement", True):
                return
            for item in payload["results"]:
                folder = Path(item.get("output_dir") or "")
                if item.get("status") not in {"ready", "cached"} or not (folder / "transcript.json").is_file():
                    continue
                if (item.get("asr") or {}).get("coverage", {}).get("content_state") == "no_speech":
                    continue
                try:
                    from .text_inputs import current_speaker_inputs
                    selected_inputs = current_speaker_inputs(item, enabled=speaker_mode != "off" and speaker_ran)
                    guarded_ai = kwargs.get("ai_polish",False)
                    if kwargs.get("postprocess_profile", "off") != "off" or guarded_ai:
                        from .postprocess import process_transcript_artifacts
                        audit = process_transcript_artifacts(folder, profile=kwargs.get("postprocess_profile", "off") if kwargs.get("postprocess_profile", "off") != "off" else "fast",
                                user_glossary_text=kwargs.get("glossary_text", ""), style=kwargs.get("ai_style", ""),
                                config=kwargs.get("config"), progress_callback=kwargs.get("progress_callback"),
                                input_artifacts=selected_inputs,
                                **({"summarize":False} if guarded_ai and kwargs.get("postprocess_profile", "off") == "off" else {}))
                    else:
                        audit = enhance_transcript_artifacts(folder, user_glossary_text=kwargs.get("glossary_text", ""),
                                ai_polish=kwargs.get("ai_polish", False), ai_style=kwargs.get("ai_style", ""), config=kwargs.get("config"), input_artifacts=selected_inputs)
                    item["enhancement"] = audit
                    item.update(audit.get("paths") or {})
                    for key, value in audit.items():
                        if key.endswith("_path"):
                            item[key] = value
                    _write_json(folder / "metadata.json", item)
                except Exception as exc:
                    if isinstance(exc, PipelineError) and (exc.blocked or exc.code in {"CANCELLED", "RESOURCE_LIMIT", "STAGE_TIMEOUT", "OUTPUT_LIMIT"}):
                        raise
                    item["enhancement"] = {"status": "partial", "error": str(exc)}
                    _write_json(folder / "metadata.json", item)
        speaker = run_post_asr_stages(payload, speaker=run_speaker if speaker_stage else None, polish=text_stage, speaker_mode=speaker_mode, native_speaker_alignment=native_speaker_alignment)
        payload["speaker_transcript"] = speaker
        path = Path(payload["manifest_path"])
        if path.is_file():
            manifest = json.loads(path.read_text())
            manifest.update(results=payload["results"], partial_count=sum(i.get("speaker_status") == "partial" or (i.get("enhancement") or {}).get("status") == "partial" for i in payload["results"]))
            if speaker is not None:
                manifest["speaker_transcript"] = speaker
            _write_json(path, manifest)
        if not defer_delivery:
            elapsed = max(0, round(time.monotonic()-started))
            payload["asr_elapsed_seconds"] = payload.get("elapsed_seconds", 0)
            payload["elapsed_seconds"] = elapsed
            payload["receipt"] = publish_pipeline(destination, job_id=delivery_job_id or destination.name,
                results=payload["results"], stages={"asr":"completed", "speaker":"completed" if speaker_stage and speaker_mode != "off" else "off",
                    "polish":"completed" if kwargs.get("enable_enhancement",True) else "off"},
                elapsed_seconds=elapsed)
        return payload

    except BaseException as exc:
        code = "CANCELLED" if isinstance(exc, KeyboardInterrupt) else getattr(exc, "code", "WORKER_FAILED")
        status = "cancelled" if code == "CANCELLED" else "blocked" if getattr(exc, "blocked", False) else "failed"
        fail_pipeline(destination, code=code, status=status)
        raise


def _transcribe_media_sources(
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
    _defer_enhancement: bool = False,
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
    effective_asr_model = asr_model
    if selected_asr_engine in {"fun-asr-nano", "sensevoice"}:
        profile_id = "fun-asr-nano-pt" if selected_asr_engine == "fun-asr-nano" else "sensevoice-int8"
        auto_qwen_model = asr_model or config_obj.get_path("models", profile_id, "path", default="")
        effective_asr_model = auto_qwen_model
    _emit_progress(progress_callback, stage="model", message="准备模型", progress=5)
    # Whisper's model and executable are intentionally resolved only when the
    # selected path needs them.  A successful Qwen run must not require a
    # Whisper installation or trigger a model download.
    model_path: Path | None = None
    whisper_bin: str | None = None
    if selected_asr_engine == "whisper-cpp":
        if config_obj.get_path("asr", "strict_selection", default=False):
            model_path = cached_model(model_name, [DEFAULT_MODEL_ROOT, *KNOWN_MODEL_ROOTS])
        else:
            model_path = resolve_whisper_model(model_name)
        _emit_progress(progress_callback, stage="model_ready", message="模型检查完成", progress=15)
        whisper_bin = _which_binary("whisper-cli")
    ffmpeg_bin = _which_binary("ffmpeg")
    if not ffmpeg_bin or (selected_asr_engine == "whisper-cpp" and not whisper_bin and not config_obj.get_path("asr", "allow_fallback", default=False)):
        missing = ["ffmpeg"] if not ffmpeg_bin else []
        if selected_asr_engine == "whisper-cpp" and not whisper_bin and not config_obj.get_path("asr", "allow_fallback", default=False):
            missing.append("whisper-cli")
        raise FileNotFoundError(f"Missing local dependencies: {', '.join(missing)}")

    formats = dedupe_strings(output_formats or DEFAULT_OUTPUT_FORMATS)
    generated_at = now_iso()
    reported_model_name = (asr_model or auto_qwen_model) if selected_asr_engine != "whisper-cpp" else model_name
    reported_model_path = "" if selected_asr_engine != "whisper-cpp" else str(model_path or "")
    run_config = {
        "started_at": started_at,
        "completed_at": "",
        "elapsed_seconds": 0,
        "generated_at": generated_at,
        "model_name": reported_model_name,
        "model_path": reported_model_path,
        "whisper_model_name": model_name,
        "whisper_model_path": str(model_path or ""),
        "schema_version": 3,
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
    atomic_write(run_config_path, json.dumps(run_config, ensure_ascii=False, indent=2))
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
            asr_model=effective_asr_model,
            postprocess_profile=postprocess_profile,
            defer_enhancement=_defer_enhancement,
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
    item_asr = [dict(item.get("asr") or {}) for item in results if isinstance(item.get("asr"), dict)]
    actual_engines = list(dict.fromkeys(str(item.get("actual_engine") or "") for item in item_asr if item.get("actual_engine")))
    actual_models = list(dict.fromkeys(str(item.get("actual_model") or "") for item in item_asr if item.get("actual_model")))
    run_config["asr"].update(
        {
            "actual_engine": actual_engines[0] if len(actual_engines) == 1 else "mixed" if actual_engines else "",
            "actual_model": actual_models[0] if len(actual_models) == 1 else "mixed" if actual_models else "",
            "actual_engines": actual_engines,
            "actual_models": actual_models,
            "provenance": [
                {"source_label": result.get("source_label", ""), **dict((result.get("asr") or {}).get("provenance") or {})}
                for result in results
                if (result.get("asr") or {}).get("provenance")
            ],
        }
    )
    atomic_write(run_config_path, json.dumps(run_config, ensure_ascii=False, indent=2))

    manifest = {
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "generated_at": generated_at,
        "model_name": reported_model_name,
        "model_path": reported_model_path,
        "whisper_model_name": model_name,
        "whisper_model_path": str(model_path or ""),
        "schema_version": 3,
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
    atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    errors_path = None
    if failed_results:
        errors_path = output_dir / "errors.json"
        atomic_write(errors_path, json.dumps({"generated_at": generated_at, "items": failed_results}, ensure_ascii=False, indent=2))
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
    defer_enhancement: bool = False,
) -> dict[str, Any]:
    source_label = descriptor.source_label or _infer_source_label(descriptor.input, index)
    item_dir = items_dir / source_label
    item_dir.mkdir(parents=True, exist_ok=True)
    source_context: dict[str, Any] = {}
    frozen_context = bool(config and config.get_path("runtime", "freeze_source_context", default=False))
    if frozen_context:
        context_path = item_dir / "source.context.json"
        expected = config.get_path("runtime", "frozen_source_contexts", default={}).get(
            f"items/{source_label}/source.context.json")
        if context_path.is_file():
            if expected != file_hash(context_path):
                raise PipelineError("CACHE_MISMATCH", "冻结的节目资料hash不符，不能刷新替代", blocked=True)
            try:
                source_context = json.loads(context_path.read_text())
                if not isinstance(source_context, dict):
                    raise ValueError("invalid source context")
            except (OSError, ValueError) as exc:
                raise PipelineError("CACHE_MISMATCH", "冻结的节目资料无效", blocked=True) from exc
        elif expected:
            raise PipelineError("CACHE_MISMATCH", "冻结的节目资料缺失", blocked=True)
        else:
            source_context = {"schema_version": 1, "status": "not_captured", "provider": "resume-frozen",
                              "url": descriptor.canonical_url, "comments": [], "speaker_candidates": []}
    transcript_path = item_dir / "transcript.txt"
    source_file = Path(descriptor.input).expanduser()
    local_source = descriptor.source_type == "local_file" and source_file.is_file()
    identity_files = [model_path] if model_path is not None else []
    candidate_root = Path(asr_model).expanduser() if asr_model else None
    runtime_identity = {}
    if asr_engine in {"fun-asr-nano", "sensevoice"}:
        from .model_profiles import PROFILES
        from .capabilities import NATIVE_REQUIREMENTS, probe_environment
        key = "fun-asr-nano-pt" if asr_engine == "fun-asr-nano" else "sensevoice-int8"
        if candidate_root and candidate_root.is_dir():
            identity_files += [candidate_root / name for name in PROFILES[key].required_files]
        settings = config.get_path("models", key, default={}) if config else {}
        python = str(Path(settings.get("worker_python") or sys.executable).expanduser().absolute())
        runtime_identity = {"python": python, "environment": probe_environment(python, NATIVE_REQUIREMENTS[key])}
        if Path(python).is_file():
            identity_files.append(Path(python))
        identity_files += [Path(__file__).with_name(name) for name in ("local_asr.py", "local_asr_worker.py", "audio_segments.py", "polish_context.py", "chinese_text.py")]
    for binary in (ffmpeg_bin, whisper_bin if asr_engine == "whisper-cpp" else None):
        if binary and Path(binary).is_file():
            identity_files.append(Path(binary))
    vad_path = Path(config.get_path("asr", "vad_model", default="")) if config else None
    if vad_path and vad_path.is_file():
        identity_files.append(vad_path)
    model_identity = {str(p): file_hash(p) for p in identity_files if p.is_file()}
    if asr_engine == "qwen3-asr":
        from .model_profiles import complete_hf_snapshot
        from .asr import qwen_model_provenance
        resolved = complete_hf_snapshot(asr_model)
        model_identity["qwen"] = qwen_model_provenance(asr_model, snapshot_path=resolved) if resolved else "unknown"
    frozen_media = config.get_path("runtime", "frozen_media", default={}).get(f"items/{source_label}") if config else None
    source_identity = frozen_media["sha256"] if descriptor.source_type == "local_file" and frozen_media else file_hash(source_file) if local_source else {
        "input": descriptor.input, "canonical_url": descriptor.canonical_url, "platform": descriptor.platform,
        "source_id": descriptor.source_id}
    verified_resume = bool(config and config.get_path("runtime", "resume_verified", default=False))
    audio_checkpoint_path = item_dir / "audio.checkpoint.json"
    try:
        audio_checkpoint = json.loads(audio_checkpoint_path.read_text())
    except (OSError, ValueError):
        audio_checkpoint = {}
    locator = value_hash(source_identity)
    media_name = audio_checkpoint.get("media_name", "")
    persisted_media = item_dir / media_name
    media_valid = bool(media_name and Path(media_name).name == media_name and
                       audio_checkpoint.get("source_signature") == locator and
                       (local_source or verified_resume) and persisted_media.is_file() and
                       file_hash(persisted_media) == audio_checkpoint.get("media_sha256"))
    if frozen_media and (not media_valid or audio_checkpoint.get("media_sha256") != frozen_media["sha256"]
            or frozen_media["relative_path"] != f"items/{source_label}/{media_name}"):
        raise PipelineError("CACHE_MISMATCH", "冻结媒体副本缺失或改变，不能从原路径重新获取替代内容", blocked=True)
    audio_settings = {"max_seconds": max_seconds, "sample_rate": 16000, "channels": 1,
                      "ffmpeg": model_identity.get(str(Path(ffmpeg_bin)), ffmpeg_bin), "version": 1}
    def audio_signature(media_hash):
        return value_hash({"media_sha256": media_hash, **audio_settings})
    audio_path = item_dir / PERSISTED_AUDIO_NAME
    audio_valid = bool(media_valid and audio_checkpoint.get("input_fingerprint") == audio_signature(audio_checkpoint.get("media_sha256"))
                       and audio_path.is_file() and file_hash(audio_path) == audio_checkpoint.get("audio_sha256"))
    fingerprint_inputs = {"source": source_identity,
                              "asr_engine": asr_engine, "asr_model": asr_model,
                              "model_identity": model_identity,
                              "whisper_model": whisper_model_name if asr_engine == "whisper-cpp" else None, "language": language,
                              "max_seconds": max_seconds, "formats": output_formats,
                              "runtime_identity": runtime_identity,
                              "audio_normalization": audio_settings,
                              "recognition_prompt": {k: config.get_path("asr", k, default=None) if config else None
                                                     for k in ("output_script", "contextual_hotwords", "hotwords")},
                              "version": 6}
    if asr_engine in {"whisper-cpp", "qwen3-asr"}:
        fingerprint_inputs["whisper_timeline_implementation"] = {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("transcription.py", "whisper_chunks.py", "whisper_models.py")}
        fingerprint_inputs["whisper_weight_registry"] = file_hash(Path(__file__).with_name("data") / "whisper-models.json")
    def asr_signature(identity_override=None):
        context_path = item_dir / "source.context.json"
        contextual = config and config.get_path("asr", "contextual_hotwords", default=False)
        context_terms = []
        if contextual and context_path.is_file():
            from .polish_context import asr_hotwords
            try:
                context_terms = asr_hotwords(json.loads(context_path.read_text()))
            except (OSError, ValueError):
                pass
        return value_hash({**fingerprint_inputs, **({"model_identity": identity_override} if identity_override is not None else {}),
                           "media_sha256": audio_checkpoint.get("media_sha256") if media_valid else None,
                           "context_terms": context_terms})
    fingerprint = asr_signature()
    previous = _read_cached_metadata(item_dir) or {}
    hashes = previous.get("cache_artifact_hashes") or {}
    valid_files = bool(hashes) and all(Path(name).name == name and (item_dir / name).is_file() and file_hash(item_dir / name) == digest for name, digest in hashes.items())
    cache_matches = previous.get("cache_fingerprint") == fingerprint and valid_files and (local_source or verified_resume)
    cache_migration = None
    if (not cache_matches and verified_resume and valid_files and audio_valid
            and asr_engine in {"fun-asr-nano", "sensevoice"}):
        from .asr_cache import relocated_native_identity
        migrated = relocated_native_identity(previous, item_dir, model_identity,
                                             runtime_identity["python"], Path(__file__).parent)
        if migrated and previous.get("cache_fingerprint") == asr_signature(migrated[0]):
            cache_matches, cache_migration = True, migrated[1]
    if transcript_path.exists() and not cache_matches:
        if not verified_resume:
            raise PipelineError("CACHE_MISMATCH", "已有转写与本次输入/模型/范围不一致或无法核验，请选择新输出目录", blocked=True)
        # A resume attempt has its own copied directory. Invalidate staged
        # derivatives, never the parent run's original transcript.
        for path in item_dir.glob("transcript.*"):
            if path.is_file():
                path.unlink()
        (item_dir / "postprocess.audit.json").unlink(missing_ok=True)
        if (item_dir / "postprocess").is_dir():
            shutil.rmtree(item_dir / "postprocess")
    if transcript_path.exists():
        # A previous run may predate episode-context collection.  Refresh only
        # the lightweight page metadata on cache hits; never rerun media/ASR.
        if not frozen_context and descriptor.platform == "xiaoyuzhou" and descriptor.canonical_url:
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
            previous_metadata = cached_metadata
            cached_asr = dict(previous_metadata.get("asr") or {})
            if cached_asr.get("actual_engine") == "whisper-cpp":
                cached_asr.update(_whisper_chunk_info(item_dir, "transcript", audio_path=_find_persisted_audio(item_dir)))
            # ASR命中只证明识别产物。新计划必须重新选择后续阶段；不能
            # 沿用旧identity/enhanced/summary路径作为本轮采用结果。
            cached_metadata = _build_metadata(
                descriptor=descriptor, source_label=source_label, status="cached",
                transcript_text=_safe_read_text(transcript_path), item_dir=item_dir,
                media_path=_find_persisted_media(item_dir), audio_path=_find_persisted_audio(item_dir),
                created_at=previous_metadata.get("created_at") or now_iso(),
                asr=cached_asr,
                source_context=source_context or previous_metadata.get("source_context") or {},
            )
            cached_metadata.update({key: value for key, value in previous_metadata.items()
                                    if key.startswith("cache_")})
            cached_metadata["updated_at"] = now_iso()
            cached_metadata["cache_fingerprint"] = fingerprint
            cached_metadata["cache_implementation_root"] = str(Path(__file__).parent)
            if cache_migration:
                cached_metadata["cache_migration"] = cache_migration
            if source_context:
                cached_metadata["source_context"] = source_context
            if enable_enhancement and not defer_enhancement and _safe_read_text(transcript_path).strip():
                if postprocess_profile != "off":
                    from .postprocess import process_transcript_artifacts
                    enhancement = process_transcript_artifacts(item_dir, profile=postprocess_profile,
                        user_glossary_text=glossary_text, style=ai_style, config=config, progress_callback=progress_callback, input_artifacts={})
                else:
                    enhancement = enhance_transcript_artifacts(item_dir, user_glossary_text=glossary_text,
                        ai_polish=ai_polish, ai_style=ai_style, config=config, input_artifacts={})
                cached_metadata["enhancement"] = enhancement
                cached_metadata.update(enhancement.get("paths") or {})
                cached_metadata.update({k: v for k, v in enhancement.items() if k.endswith("_path")})
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
        metadata["error_code"] = "SOURCE_AUTH_REQUIRED" if descriptor.requires_auth else "SOURCE_UNAVAILABLE"
        metadata["blocked"] = descriptor.requires_auth
        _write_json(item_dir / "metadata.json", metadata)
        _emit_progress(progress_callback, stage="item_failed", message=f"{source_label} 失败：{descriptor.error}", progress=_item_progress(base_progress, 1.0, item_count))
        return metadata

    try:
        if not frozen_context and descriptor.platform == "xiaoyuzhou" and descriptor.canonical_url:
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
            if media_valid:
                persisted_media_path = persisted_media
            else:
                media_path = prepare_media_source(descriptor, temp_path)
                persisted_media_path = _persist_media(media_path, item_dir)
                media_valid = True
                audio_checkpoint = {"schema_version": 1, "source_signature": locator,
                                    "media_name": persisted_media_path.name, "media_sha256": file_hash(persisted_media_path)}
            _emit_progress(progress_callback, stage="audio", message=f"{source_label} 抽取音频", progress=_item_progress(base_progress, 0.35, item_count))
            if not audio_valid:
                extract_audio(source_path=persisted_media_path, target_path=audio_path,
                              ffmpeg_bin=ffmpeg_bin, max_seconds=max_seconds)
            audio_checkpoint.update(input_fingerprint=audio_signature(audio_checkpoint["media_sha256"]),
                                    audio_sha256=file_hash(audio_path), status="completed", completed_at=now_iso())
            _write_json(audio_checkpoint_path, audio_checkpoint)
            fingerprint = asr_signature()
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
        if enable_enhancement and not defer_enhancement and transcript_text.strip():
            if postprocess_profile != "off":
                from .postprocess import process_transcript_artifacts

                enhancement = process_transcript_artifacts(
                    item_dir,
                    input_artifacts={},
                    profile=postprocess_profile,
                    user_glossary_text=glossary_text,
                    style=ai_style,
                    config=config,
                    progress_callback=progress_callback,
                )
            else:
                enhancement = enhance_transcript_artifacts(
                    item_dir,
                    input_artifacts={},
                    user_glossary_text=glossary_text,
                    ai_polish=ai_polish,
                    ai_style=ai_style,
                    config=config,
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
        if (asr_info.get("coverage") or {}).get("content_state") == "no_speech":
            metadata.update(status="empty", char_count=0)
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
        if isinstance(exc, PipelineError):
            metadata["error_code"] = exc.code
            metadata["blocked"] = exc.blocked
        _emit_progress(progress_callback, stage="item_failed", message=f"{source_label} 失败：{exc}", progress=_item_progress(base_progress, 1.0, item_count))
    metadata["cache_fingerprint"] = asr_signature()
    metadata["cache_implementation_root"] = str(Path(__file__).parent)
    derived = [Path(v) for k, v in metadata.items() if k.startswith("normalized_") and k.endswith("_path") and v]
    metadata["cache_artifact_hashes"] = {p.name: file_hash(p) for p in [transcript_path, item_dir / "transcript.json", item_dir / PERSISTED_AUDIO_NAME,
                                          item_dir / "transcript.chunks.audit.json", *derived]
                                          if p.is_file() and metadata.get("status") != "failed"}
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
    atomic_write(output_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return output_path


def ensure_whisper_model(model_name: str, model_root: Path = DEFAULT_MODEL_ROOT) -> Path:
    from .whisper_download import prepare_model
    return prepare_model(model_name, model_root, KNOWN_MODEL_ROOTS)


def resolve_whisper_model(model_name_or_path: str) -> Path:
    cached = cached_model(model_name_or_path, [DEFAULT_MODEL_ROOT, *KNOWN_MODEL_ROOTS])
    return cached if cached is not None else ensure_whisper_model(model_name_or_path)


def prepare_media_source(source: str | SourceDescriptor, work_dir: Path) -> Path:
    descriptor = source if isinstance(source, SourceDescriptor) else _coerce_descriptors([source])[0]
    if descriptor.error and not descriptor.resolved_media_url and descriptor.source_type != "local_file":
        raise PipelineError("SOURCE_AUTH_REQUIRED" if descriptor.requires_auth else "SOURCE_UNAVAILABLE",
                            descriptor.error, blocked=descriptor.requires_auth, stage="resolve")
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
    run_media_command(command, timeout=7200, stage='audio-extract')
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
    if config_obj.get_path("asr", "allow_fallback", default=False) and (config_obj.get_path("asr", "strict_selection", default=False) or engine in {"fun-asr-nano", "sensevoice"}):
        from .model_profiles import PROFILES
        data = config_obj.to_dict()
        data.setdefault("asr", {})["allow_fallback"] = False
        primary_config = NextEchoConfig(data)
        arguments = dict(audio_path=audio_path, output_dir=output_dir, output_base=output_base,
                whisper_bin=whisper_bin, whisper_model_path=whisper_model_path, whisper_model_name=whisper_model_name,
                language=language, output_formats=output_formats, config=primary_config)
        try:
            return transcribe_audio_with_fallback(**arguments, engine=engine, model=model)
        except Exception as original:
            if getattr(original, "code", "") in {"CANCELLED", "INVALID_INPUT", "PROVIDER_BUSY_UNKNOWN"}:
                raise
            allowed = config_obj.get_path("asr", "fallback_order", default=[] if engine == "sensevoice" else ["sensevoice-int8"])
            if not isinstance(allowed, list) or len(allowed) > 2 or any(k not in {"sensevoice-int8", "fun-asr-nano-pt"} for k in allowed):
                raise PipelineError("INVALID_INPUT", "回退链必须是最多2个已登记CPU ASR档案") from original
            chain = []
            previous_engine, previous_model, previous_error = engine, model, original
            for key in allowed:
                candidate = PROFILES[key]
                if candidate.engine == engine:
                    continue
                path = config_obj.get_path("models", key, "path", default="")
                attempt = {"from_engine": previous_engine, "from": previous_model, "to_engine": candidate.engine,
                           "to": path, "reason": getattr(previous_error, "code", type(previous_error).__name__), "at": now_iso()}
                chain.append(attempt)
                try:
                    actual = transcribe_audio_with_fallback(**arguments, engine=candidate.engine, model=path)
                    attempt["outcome"] = "selected"
                    actual.update(requested_engine=engine, requested_model=model, fallbacks=chain, degraded=True)
                    actual.setdefault("provenance", {})["fallbacks"] = chain
                    return actual
                except Exception as failed:
                    if getattr(failed, "code", "") in {"CANCELLED", "INVALID_INPUT", "PROVIDER_BUSY_UNKNOWN"}:
                        raise
                    attempt.update(outcome="failed", failure_reason=getattr(failed, "code", type(failed).__name__))
                    previous_engine, previous_model, previous_error = candidate.engine, path, failed
            raise original
    fallbacks: list[dict[str, str]] = []
    if engine in {"fun-asr-nano", "sensevoice"}:
        from .local_asr import transcribe_local_audio
        result = transcribe_local_audio(audio_path, output_dir, engine=engine, model=model, language=language, config=config_obj)
        write_qwen_transcript_result(result, output_dir=output_dir, output_base=output_base, output_formats=output_formats)
        normalized_paths = {}
        if config_obj.get_path("asr", "output_script", default="original") == "simplified":
            from .chinese_text import write_simplified_artifacts
            normalized_paths = write_simplified_artifacts(output_dir, json.loads((output_dir / "transcript.json").read_text()))
        return {"requested_engine": engine, "requested_model": model, "actual_engine": result.backend,
                "actual_model": result.model, "version": result.version, "provenance": result.provenance,
                "timing_precision": result.timing_precision, "coverage": result.coverage,
                "resources": result.resources, "fallbacks": result.provenance.get("fallbacks", []), **normalized_paths}
    if engine == "qwen3-asr":
        qwen_candidates = [model]
        fast_model = qwen_model_for_quality("fast", config_obj)
        if model != fast_model and not config_obj.get_path("asr", "strict_selection", default=False):
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
                    "provenance": dict(result.provenance or {}),
                    "fallbacks": fallbacks,
                    "resources": result.resources,
                }
            except Exception as exc:
                if isinstance(exc, PipelineError):
                    raise
                if config_obj.get_path("asr", "strict_selection", default=False):
                    raise
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
        if config_obj.get_path("asr", "strict_selection", default=False):
            whisper_model_path = cached_model(whisper_name, [DEFAULT_MODEL_ROOT, *KNOWN_MODEL_ROOTS])
            if whisper_model_path is None:
                raise PipelineError("MODEL_MISSING", "Whisper权重未缓存；普通run不会下载", blocked=True)
        else:
            whisper_model_path = resolve_whisper_model(whisper_name)
    try:
        weight_identity = validate_whisper_model(whisper_model_path)
        transcribe_audio_file(
            audio_path=audio_path,
            output_dir=output_dir,
            output_base=output_base,
            whisper_bin=whisper_bin,
            model_path=whisper_model_path,
            language=language,
            output_formats=output_formats,
        )
        annotate_whisper_transcript(output_dir / f"{output_base}.json", model=whisper_model_name, language=language, provenance=weight_identity)
        return {
            "requested_engine": engine,
            "requested_model": model,
            "actual_engine": "whisper-cpp",
            "actual_model": whisper_model_name,
            "version": "",
            "fallbacks": fallbacks,
            "provenance": weight_identity,
            **_whisper_chunk_info(output_dir, output_base, audio_path=audio_path),
        }
    except Exception as exc:
        if isinstance(exc, PipelineError):
            raise
        if config_obj.get_path("asr", "strict_selection", default=False):
            raise
        if whisper_model_name != "base":
            fallbacks.append({"from": whisper_model_name, "to": "base", "reason": str(exc), "at": now_iso()})
            base_path = resolve_whisper_model("base")
            weight_identity = validate_whisper_model(base_path, name="base")
            transcribe_audio_file(
                audio_path=audio_path,
                output_dir=output_dir,
                output_base=output_base,
                whisper_bin=whisper_bin,
                model_path=base_path,
                language=language,
                output_formats=output_formats,
            )
            annotate_whisper_transcript(output_dir / f"{output_base}.json", model="base", language=language, provenance=weight_identity)
            return {
                "requested_engine": engine,
                "requested_model": model,
                "actual_engine": "whisper-cpp",
                "actual_model": "base",
                "version": "",
                "fallbacks": fallbacks,
                "provenance": weight_identity,
                **_whisper_chunk_info(output_dir, output_base, audio_path=audio_path),
            }
        raise


def annotate_whisper_transcript(path: Path, *, model: str, language: str, provenance: dict | None = None) -> None:
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
    if provenance is not None:
        payload["engine"]["provenance"] = provenance
    payload.setdefault("result", {})["language"] = payload.get("result", {}).get("language") or language
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))


def write_qwen_transcript_result(
    result: TranscriptResult,
    *,
    output_dir: Path,
    output_base: str,
    output_formats: list[str],
) -> None:
    segments = result.segments or aggregate_qwen_words(result.words, fallback_text="")
    if result.text.strip() and not segments:
        raise ValueError("Transcript text has no measured time windows")
    transcript = [
        {
            "id": f"seg_{index:05d}",
            "text": item["text"],
            "offsets": {"from": item["start_ms"], "to": item["end_ms"]},
            "timestamps": {"from": _format_srt_timestamp(item["start_ms"]), "to": _format_srt_timestamp(item["end_ms"])},
            "words": item.get("words", []),
        }
        for index, item in enumerate(segments)
    ]
    text = "\n".join(item["text"] for item in segments).strip() or result.text.strip()
    txt_path = output_dir / f"{output_base}.txt"
    atomic_write(txt_path, format_transcript_attribution(text))
    if "json" in output_formats:
        atomic_write(output_dir / f"{output_base}.json", json.dumps(
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
                    "timing_precision": result.timing_precision,
                    "coverage": result.coverage,
                    "transcription": transcript,
                },
                ensure_ascii=False,
                indent=2,
            ))
    if segments and ("srt" in output_formats or "vtt" in output_formats):
        write_subtitles_from_segments(
            transcript,
            srt_path=output_dir / f"{output_base}.srt" if "srt" in output_formats else None,
            vtt_path=output_dir / f"{output_base}.vtt" if "vtt" in output_formats else None,
        )


def aggregate_qwen_words(words: list[dict[str, Any]], *, fallback_text: str) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    current: list[str] = []
    current_words: list[dict[str, Any]] = []
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
        current_words.append({"text": text, "start": word_start / 1000.0, "end": word_end / 1000.0})
        end_ms = max(word_end, word_start)
        joined = "".join(current)
        if text.endswith(("。", "？", "！", ".", "?", "!")) or end_ms - start_ms >= 15_000 or len(joined) >= 50:
            grouped.append({"text": joined, "start_ms": start_ms, "end_ms": end_ms, "words": current_words})
            current = []
            current_words = []
    if current:
        grouped.append({"text": "".join(current), "start_ms": start_ms, "end_ms": end_ms, "words": current_words})
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
        atomic_write(srt_path, "\n".join(srt).strip() + "\n")
    if vtt_path is not None:
        vtt: list[str] = ["WEBVTT", ""]
        for item in rows:
            text, start_ms, end_ms = _segment_values(item)
            vtt.extend([f"{_format_vtt_timestamp(start_ms)} --> {_format_vtt_timestamp(max(end_ms, start_ms + 1))}", text, ""])
        atomic_write(vtt_path, "\n".join(vtt).strip() + "\n")


def _segment_values(item: Any) -> tuple[str, int, int]:
    if isinstance(item, dict):
        offsets = item.get("offsets") or {}
        return str(item.get("text") or ""), int(offsets.get("from") or item.get("start_ms") or 0), int(offsets.get("to") or item.get("end_ms") or 0)
    return str(item.text), int(item.start_ms), int(item.end_ms)


def _bounded_whisper(command: list[str], model_path: Path, output_dir: Path) -> None:
    identity = validate_whisper_model(model_path)
    _write_json(output_dir / "logs/whisper.model.json", identity)
    if os.name != "posix":
        # The new detached protocol rejects unsupported platforms up front.
        # Retain the established Windows CLI until native job ownership lands.
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=7200)
        return
    from .resources import GIB, model_policy, run_owned
    from .job_store import current_cancelled
    estimate = int(identity["bytes"] * 2) + 256 * 1024**2
    try:
        run_owned(command, log_path=output_dir / "logs/whisper.log",
                  policy=model_policy(estimate, max(estimate * 3, GIB), 7200), cancel=current_cancelled)
    except PipelineError as exc:
        if exc.code == "WORKER_FAILED":
            path = output_dir / "logs/whisper.log"
            tail = path.read_text(errors="replace")[-4000:] if path.is_file() else ""
            raise subprocess.CalledProcessError(1, command, stderr=tail) from exc
        raise


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
    _bounded_whisper(command, model_path, output_dir)


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
    (output_dir / f"{output_base}.chunks.audit.json").unlink(missing_ok=True)
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
        language=language,
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
    if type(duration_ms) is not int or duration_ms <= 0:
        raise PipelineError("INVALID_INPUT", "Whisper分片音频时长无效", stage="asr")
    chunk_count = max(1, math.ceil(duration_ms / LONG_AUDIO_CHUNK_MS))
    aggregated_segments: list[dict[str, Any]] = []
    audit_path = output_dir / f"{output_base}.chunks.audit.json"
    audit_path.unlink(missing_ok=True)
    evidence = Path(tempfile.mkdtemp(prefix=f".{output_base}-whisper-chunks-", dir=output_dir))
    chunk_records = []

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
            # 原生输出及所属进程诊断保留；只有临时分片音频会清理。
            chunk_output_base = evidence / f"chunk-{chunk_index + 1:04d}" / "transcript"
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
                output_formats=dedupe_strings([*output_formats, "json"]),
            )
            chunk_json = chunk_output_base.with_suffix(".json")
            if not chunk_json.is_file() or chunk_json.stat().st_size > 32 * 1024**2:
                raise PipelineError("PROTOCOL_ERROR", "Whisper分片未生成合并必需的原生JSON", stage="asr")
            retained = chunk_json
            relative = str(retained.relative_to(output_dir))
            artifacts = []
            for path in sorted(chunk_output_base.parent.rglob("*")):
                if path.is_symlink():
                    raise PipelineError("PROTOCOL_ERROR", "Whisper分片产物不允许符号链接", stage="asr")
                if path.is_file():
                    artifacts.append({"relative_path": str(path.relative_to(output_dir)), "sha256": file_hash(path)})
            chunk_records.append({"chunk_index": chunk_index, "offset_ms": offset_ms,
                "duration_ms": chunk_duration_ms, "retain_start_ms": trim_before_ms,
                "relative_json": relative, "sha256": file_hash(retained), "artifacts": artifacts})
            chunk_segments = _load_shifted_transcription_segments(
                retained, offset_ms, trim_before_ms=trim_before_ms)
            for row in chunk_segments:
                if row["offsets"]["to"] > offset_ms + chunk_duration_ms:
                    raise PipelineError("PROTOCOL_ERROR", "Whisper分片时间超过实际音频范围", stage="asr")
                row.update(source_chunk_index=chunk_index, source_chunk_json=relative)
            aggregated_segments.extend(chunk_segments)

    aggregated_segments.sort(key=lambda row: row["offsets"]["from"])
    review = []
    for index, row in enumerate(aggregated_segments):
        row["id"] = f"seg_{index:05d}"
        if row.get("seam_review"):
            review.append({"segment_id": row["id"], "source_chunk_index": row["source_chunk_index"],
                           "offsets": row["offsets"], **row["seam_review"]})
    txt_path = output_dir / f"{output_base}.txt"
    atomic_write(txt_path, "\n".join(row["text"].strip() for row in aggregated_segments).strip() + "\n")
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
            "timing_precision": "segment",
            "review_required": bool(review),
        }
        atomic_write(json_path, json.dumps(json_payload, ensure_ascii=False, indent=2))
    write_subtitles_from_segments(aggregated_segments,
        srt_path=output_dir / f"{output_base}.srt" if "srt" in output_formats else None,
        vtt_path=output_dir / f"{output_base}.vtt" if "vtt" in output_formats else None)
    atomic_write(audit_path, json.dumps({"schema_version": 2, "protocol": "whisper-measured-seam-v2",
        "audio_sha256": file_hash(audio_path), "processed_duration_ms": duration_ms,
        "chunks": chunk_records, "review": review,
        "policy": "previous chunk owns wholly prior context; crossing segments retained whole; no invented word times"},
        ensure_ascii=False, indent=2))


def _whisper_chunk_info(output_dir: Path, output_base: str, *, audio_path: Path | None = None) -> dict:
    """恢复时重新核对原生分片字节，把artifact路径绑定当前run。"""
    audit = output_dir / f"{output_base}.chunks.audit.json"
    if audit.is_symlink():
        raise PipelineError("CACHE_MISMATCH", "Whisper分片台账不能是符号链接", blocked=True, stage="asr")
    if not audit.is_file():
        return {}
    try:
        if audit.stat().st_size > 4 * 1024**2:
            raise ValueError("audit size limit")
        data = json.loads(audit.read_text())
        if data.get("schema_version") != 2 or data.get("protocol") != "whisper-measured-seam-v2":
            raise ValueError("unknown chunk protocol")
        if audio_path is not None and file_hash(audio_path) != data.get("audio_sha256"):
            raise ValueError("changed normalized source audio")
        sources, artifacts = [], []
        if not isinstance(data.get("chunks"), list) or not data["chunks"]:
            raise ValueError("missing native chunks")
        def checked(relative, digest):
            path = output_dir / relative
            if (Path(relative).is_absolute() or not path.resolve().is_relative_to(output_dir.resolve())
                    or path.is_symlink() or file_hash(path) != digest):
                raise ValueError("changed chunk")
            return path
        for row in data["chunks"]:
            path = checked(row["relative_json"], row["sha256"])
            sources.append({"json_path": str(path.resolve()), "sha256": row["sha256"]})
            for artifact in row.get("artifacts", []):
                path = checked(artifact["relative_path"], artifact["sha256"])
                artifacts.append({"artifact_path": str(path.resolve()), "sha256": artifact["sha256"]})
        return {"timing_precision": "segment", "review_required": bool(data["review"]),
                "chunk_audit_path": str(audit.resolve()), "chunk_sources": sources,
                "chunk_evidence": artifacts,
                "chunk_audio_sha256": data["audio_sha256"],
                "coverage": {"processed_seconds": data["processed_duration_ms"] / 1000}}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PipelineError("CACHE_MISMATCH", "Whisper分片证据缺失或改变，请创建新attempt", blocked=True, stage="asr") from exc


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
    _bounded_whisper(command, model_path, output_dir)


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
    run_media_command(command, timeout=300, stage='audio-segment')
    return target_path


def probe_audio_duration_ms(audio_path: Path) -> int:
    completed = run_media_command(
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
        check=True, timeout=30, stage="audio-probe",
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
    from .whisper_chunks import shifted_segments
    rows = shifted_segments(json.loads(path.read_text(encoding="utf-8")), offset_ms, trim_before_ms)
    for row in rows:
        row["timestamps"] = {"from": _format_srt_timestamp(row["offsets"]["from"]),
                             "to": _format_srt_timestamp(row["offsets"]["to"])}
    return rows


def _append_shifted_srt(path: Path, output: list[str], offset_ms: int, next_index: int,
                        *, trim_before_ms: int = 0) -> int:
    from .whisper_chunks import keep_interval, subtitle_cues
    for cue in subtitle_cues(path.read_text(encoding="utf-8"), vtt=False):
        start, end = cue["start_ms"] + offset_ms, cue["end_ms"] + offset_ms
        if not keep_interval(start, end, trim_before_ms):
            continue
        output.extend([str(next_index), f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}",
                       *cue["lines"], ""])
        next_index += 1
    return next_index


def _append_shifted_vtt(path: Path, output: list[str], offset_ms: int, *, trim_before_ms: int = 0) -> None:
    from .whisper_chunks import keep_interval, subtitle_cues
    for cue in subtitle_cues(path.read_text(encoding="utf-8"), vtt=True):
        start, end = cue["start_ms"] + offset_ms, cue["end_ms"] + offset_ms
        if not keep_interval(start, end, trim_before_ms):
            continue
        if cue["identifier"]:
            output.append(cue["identifier"])
        output.extend([f"{_format_vtt_timestamp(start)} --> {_format_vtt_timestamp(end)}{cue['settings']}",
                       *cue["lines"], ""])


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
    run_media_command(command, timeout=900, stage='media-download')
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
            run_media_command(command, timeout=900, stage='media-download')
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
    atomic_write(path, format_transcript_attribution(path.read_text(encoding="utf-8")))


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
    atomic_write(path, json.dumps(redact_structure(payload), ensure_ascii=False, indent=2))


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
        **{k: v for k, v in (asr or {}).items() if k.startswith("normalized_") and k.endswith("_path")},
        "source_context": source_context or {},
        "output_dir": str(item_dir),
        "excerpt": transcript_text[:400],
        "error": error,
    }
    payload.update(descriptor.to_dict())
    payload["source_label"] = source_label
    payload["error"] = error or descriptor.error
    if descriptor.source_id:
        payload["source_id"] = descriptor.source_id
    if descriptor.selection_evidence:
        payload["source_selection"] = descriptor.selection_evidence
    if (item_dir / "source.context.json").is_file():
        payload["source_context_json_path"] = str((item_dir / "source.context.json").resolve())
    for key in ("text_path", "json_path", "srt_path", "vtt_path"):
        if not Path(payload[key]).is_file():
            payload[key] = ""
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
        callback({"stage": stage, "message": redact(message), "progress": progress})


def _find_existing_model(model_name: str, model_root: Path) -> Path | None:
    return find_model(model_name, [model_root, *KNOWN_MODEL_ROOTS])


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
