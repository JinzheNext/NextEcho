from __future__ import annotations

import json
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from threading import Event, Lock, Thread
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4
import time

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module
    resource = None

from workbench.speaker_transcript import (
    build_speaker_transcript,
    detect_speaker_backend_status,
)
from workbench.config import load_config
from workbench.postprocess import FAST_MODEL, QUALITY_MODEL, ollama_status, process_transcript_artifacts, resolve_postprocess_models
from workbench.asr import (
    QWEN_ACCURATE_MODEL,
    QWEN_FAST_MODEL,
    QWEN_FORCED_ALIGNER_MODEL,
    is_apple_silicon,
    qwen_forced_aligner_cached,
    qwen_forced_aligner_model,
    qwen_memory_safe,
    qwen_mlx_available,
    qwen_accurate_model,
    qwen_fast_model,
    qwen_model_cached,
    qwen_model_provenance,
)
from workbench.sources import resolve_sources
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from workbench.transcription import DEFAULT_MODEL_NAME, transcribe_media_sources
from workbench.task_coordinator import HeavyTaskCoordinator, HeavyTaskLease

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "outputs" / "transcriptions"
ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".aac", ".mov", ".webm"}
TERMINAL_ITEM_STATUSES = {"ready", "empty", "failed", "cached"}
STALE_JOB_SECONDS = 180
HEARTBEAT_SECONDS = 15
TRANSCRIPTION_JOB = "transcription"
SPEAKER_TRANSCRIPT_JOB = "speaker_transcript"
ENHANCEMENT_JOB = "enhancement"
MODEL_DOWNLOAD_JOB = "model_download"
QUALITY_TO_MODEL = {
    "accurate": DEFAULT_MODEL_NAME,
    "fast": "base",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
JOBS: dict[str, dict] = {}
JOBS_LOCK = Lock()
HEAVY_TASK_COORDINATOR = HeavyTaskCoordinator(heartbeat_seconds=HEARTBEAT_SECONDS)
# Kept as a compatibility name for integrations that imported the old
# constant.  New work must use HEAVY_TASK_COORDINATOR so all heavy stages
# share exactly one lease.
HEAVY_MODEL_LOCK = HEAVY_TASK_COORDINATOR
SPEAKER_PROFILES_PATH = ROOT / "outputs" / "speaker_profiles.json"
JOB_STATUS_VALUES = {
    "queued",
    "waiting_for_model",
    "downloading",
    "transcribing",
    "diarizing",
    "aligning",
    "identifying_speakers",
    "polishing",
    "summarizing",
    "packaging",
    "ready",
    "partial",
    "failed",
}
LEGACY_TERMINAL_STATUSES = {"completed", "failed"}
STAGE_TO_STATUS = {
    "queued": "queued",
    "waiting_for_model": "waiting_for_model",
    "model": "transcribing",
    "model_ready": "transcribing",
    "downloading": "downloading",
    "media": "transcribing",
    "audio": "transcribing",
    "asr": "transcribing",
    "source_context": "transcribing",
    "preprocess": "diarizing",
    "diarization": "diarizing",
    "align": "aligning",
    "identifying_speakers": "identifying_speakers",
    "polish": "polishing",
    "summarize": "summarizing",
    "knowledge_complete": "packaging",
    "item_complete": "packaging",
    "item_failed": "partial",
    "write": "packaging",
    "complete": "packaging",
    "failed": "failed",
}
MANIFEST_SCHEMA_VERSION = 3


def _resource_snapshot() -> dict[str, object]:
    """Capture lightweight local resource telemetry without extra deps."""

    snapshot: dict[str, object] = {
        "platform": platform.system().lower(),
        "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "peak_rss_bytes": None,
        "available": False,
    }
    if resource is not None:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # macOS reports bytes; Linux/BSD traditionally report KiB.
            raw = int(getattr(usage, "ru_maxrss", 0) or 0)
            snapshot["peak_rss_bytes"] = raw if platform.system() == "Darwin" else raw * 1024
            snapshot["available"] = True
        except (OSError, ValueError):
            pass
    return snapshot


def _configured_model(getter, fallback: str, config=None) -> str:
    """Read central model selection while keeping the legacy API resilient."""

    try:
        value = str((getter(config) if config is not None else getter()) or "").strip()
    except TypeError:
        # A test/integration may provide the old zero-argument getter.
        try:
            value = str(getter() or "").strip()
        except Exception:
            value = ""
    except Exception:
        value = ""
    return value or fallback


def _effective_config_value(config, section: str, key: str, default=None):
    """Read one config value from either ``NextEchoConfig`` or a plain mapping."""

    try:
        getter = getattr(config, "get_path", None)
        if callable(getter):
            return getter(section, key, default=default)
        if isinstance(config, dict):
            return (config.get(section) or {}).get(key, default)
    except (AttributeError, TypeError, ValueError):
        return default
    return default


def _model_provenance(model: str) -> dict:
    try:
        return qwen_model_provenance(model)
    except Exception as exc:
        return {"source_repo": model, "revision": "", "sha256": "", "error": str(exc)}


def _resource_metrics(start: dict | None = None, end: dict | None = None, *, stages: dict | None = None) -> dict:
    start = start or {}
    end = end or _resource_snapshot()
    start_rss = start.get("peak_rss_bytes")
    end_rss = end.get("peak_rss_bytes")
    delta = None
    if isinstance(start_rss, int) and isinstance(end_rss, int):
        delta = max(0, end_rss - start_rss)
    return {
        "available": bool(end.get("available")),
        "platform": end.get("platform") or platform.system().lower(),
        "peak_rss_bytes": end_rss,
        "peak_rss_delta_bytes": delta,
        "memory_pressure": "unknown",
        "available_memory_bytes": None,
        "swap_delta_bytes": None,
        "stages": stages or {},
        "captured_at": end.get("captured_at") or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def _default_manifest_fields(run_dir: Path, *, run_config: dict | None = None) -> dict:
    config = run_config or {}
    enhancement = config.get("enhancement") or {}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "asr": dict(config.get("asr") or {}),
        "diarization": {
            "status": "not_started",
            "backend": "",
            "model": "",
            "fallbacks": [],
            "coverage": {},
        },
        "speaker_identity": {
            "schema_version": 3,
            "status": "not_started",
            "speakers": [],
            "mapping_path": str(run_dir / "speaker.identity-map.json"),
        },
        "postprocess": dict(config.get("postprocess") or {
            "profile": enhancement.get("postprocess_profile", "off"),
            "enabled": enhancement.get("postprocess_profile", "off") != "off",
        }),
        "fallbacks": [],
        "timings": {"total_elapsed_seconds": int(config.get("elapsed_seconds") or 0), "stages": {}},
        "artifact_paths": {
            "items_dir": str(run_dir / "items"),
            "manifest": str(run_dir / "manifest.json"),
            "run_config": str(run_dir / "run_config.json"),
            "errors": str(run_dir / "errors.json"),
            "speaker_json": str(run_dir / "transcript.speakers.json"),
            "speaker_txt": str(run_dir / "transcript.speakers.txt"),
            "speaker_md": str(run_dir / "transcript.speakers.md"),
        },
        "resource_metrics": _resource_metrics(),
    }


def _upgrade_manifest_v3(run_dir: Path, manifest: dict | None = None, *, job: dict | None = None) -> dict:
    """Add v3 provenance fields while preserving all v2 fields verbatim."""

    manifest = dict(manifest or {})
    config = load_run_config(run_dir) or {}
    defaults = _default_manifest_fields(run_dir, run_config=config)
    for key, value in defaults.items():
        if key not in manifest:
            manifest[key] = value
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest.setdefault("asr", {})
    manifest["asr"] = {**dict(config.get("asr") or {}), **dict(manifest.get("asr") or {})}
    manifest["asr"].setdefault("requested_engine", "")
    manifest["asr"].setdefault("requested_model", "")
    manifest["asr"].setdefault("actual_engine", "")
    manifest["asr"].setdefault("actual_model", manifest.get("model_name") or "")
    manifest["asr"].setdefault("revision", manifest["asr"].get("model_revision") or "")
    manifest["asr"].setdefault("model_revision", manifest["asr"].get("revision") or "")
    manifest["asr"].setdefault("model_digest", "")
    manifest["asr"].setdefault("fallbacks", [])
    manifest["diarization"] = {
        "status": "not_started",
        "backend": "",
        "model": "",
        "model_revision": "",
        "model_digest": "",
        "fallbacks": [],
        "coverage": {},
        **dict(manifest.get("diarization") or {}),
    }
    manifest["speaker_identity"] = {
        "schema_version": 3,
        "status": "not_started",
        "speakers": [],
        "mapping_path": str(run_dir / "speaker.identity-map.json"),
        **dict(manifest.get("speaker_identity") or {}),
    }
    manifest.setdefault("postprocess", {})
    manifest["postprocess"] = {**dict(config.get("postprocess") or {}), **dict(manifest.get("postprocess") or {})}
    manifest["postprocess"].setdefault("profile", (config.get("enhancement") or {}).get("postprocess_profile", "off"))
    manifest["postprocess"].setdefault("enabled", manifest["postprocess"].get("profile") not in {None, "off"})
    manifest["postprocess"].setdefault("actual_models", [])
    manifest["postprocess"].setdefault("fallbacks", [])
    manifest.setdefault("fallbacks", [])
    existing_timings = dict(manifest.get("timings") or {})
    if not existing_timings.get("total_elapsed_seconds") and manifest.get("elapsed_seconds"):
        existing_timings["total_elapsed_seconds"] = int(manifest.get("elapsed_seconds") or 0)
    else:
        existing_timings.setdefault("total_elapsed_seconds", int(manifest.get("elapsed_seconds") or 0))
    existing_timings["stages"] = dict(existing_timings.get("stages") or {})
    manifest["timings"] = existing_timings
    manifest.setdefault("artifact_paths", {})
    manifest["artifact_paths"] = {**dict(manifest.get("artifact_paths") or {}), **defaults["artifact_paths"]}
    # These aliases existed in older manifests.  Keep them stable and rooted
    # in the run directory so temporary worker paths never leak to clients.
    manifest["run_config_path"] = str(run_dir / "run_config.json")
    manifest["items_dir"] = str(run_dir / "items")
    manifest.setdefault("resource_metrics", _resource_metrics())
    if job:
        start_metrics = job.get("resource_metrics_start")
        stage_timings = dict(job.get("stage_timings") or {})
        current_stage = str(job.get("stage") or "")
        current_started = job.get("stage_started_monotonic")
        if current_stage and isinstance(current_started, (int, float)):
            stage_timings[current_stage] = round(max(0.0, time.monotonic() - float(current_started)), 3)
        manifest["resource_metrics"] = _resource_metrics(start_metrics, _resource_snapshot(), stages=stage_timings)
        manifest["timings"].update({"stages": stage_timings})
        manifest["timings"].setdefault("job_elapsed_seconds", int(job.get("elapsed_seconds") or 0))

    stage_values = manifest["timings"].get("stages") or {}
    for name, aliases in {
        "asr_seconds": ("model", "media", "audio", "asr"),
        "diarization_seconds": ("preprocess", "diarization", "align", "identifying_speakers"),
        "postprocess_seconds": ("polish", "summarize"),
    }.items():
        manifest["timings"].setdefault(name, round(sum(float(stage_values.get(key) or 0) for key in aliases), 3))

    # Normalize item-level provenance so clients can render a single shape for
    # old and new runs.  Unknown values remain unknown; no names are inferred.
    all_fallbacks = [item for item in manifest.get("fallbacks") or [] if isinstance(item, dict)]
    result_items = [item for item in manifest.get("results") or [] if isinstance(item, dict)]
    for item in result_items:
        if not isinstance(item, dict):
            continue
        item.setdefault("schema_version", 3)
        item["asr"] = dict(item.get("asr") or {})
        item["asr"].setdefault("requested_engine", manifest["asr"].get("requested_engine") or "")
        item["asr"].setdefault("requested_model", manifest["asr"].get("requested_model") or "")
        item["asr"].setdefault("actual_engine", item["asr"].get("selected_engine") or "")
        item["asr"].setdefault("actual_model", item["asr"].get("requested_model") or manifest.get("model_name") or "")
        item["asr"].setdefault("revision", item["asr"].get("model_revision") or "")
        item["asr"].setdefault("model_revision", item["asr"].get("revision") or "")
        item["asr"].setdefault("model_digest", "")
        item["asr"].setdefault("fallbacks", [])
        item["diarization"] = {"status": item.get("speaker_status") or "not_started", "backend": "", "model": "", "model_revision": "", "model_digest": "", "fallbacks": [], "coverage": {}, **dict(item.get("diarization") or {})}
        item["speaker_identity"] = {"schema_version": 3, "status": item.get("speaker_status") or "not_started", "speakers": [], **dict(item.get("speaker_identity") or {})}
        item.setdefault("postprocess", item.get("enhancement") or {})
        item.setdefault("fallbacks", list(item["asr"].get("fallbacks") or []))
        item.setdefault("timings", {})
        item_dir = None
        try:
            item_dir = _resolve_item_dir(run_dir, item)
        except (NameError, OSError, ValueError):
            item_dir = None
        if item_dir is None:
            label = secure_filename(str(item.get("source_label") or "item")) or "item"
            item_dir = run_dir / "items" / label
        artifact_names = {
            "text_path": "transcript.txt",
            "json_path": "transcript.json",
            "srt_path": "transcript.srt",
            "vtt_path": "transcript.vtt",
            "media_path": "source.media",
            "audio_path": "转录音频.wav",
        }
        artifact_paths = dict(item.get("artifact_paths") or {})
        for key, filename in artifact_names.items():
            if item.get(key):
                artifact_paths[key] = str(item_dir / filename)
        item["artifact_paths"] = artifact_paths
        item.setdefault("resource_metrics", _resource_metrics())
        for fallback in item["asr"].get("fallbacks") or []:
            if isinstance(fallback, dict):
                all_fallbacks.append({"source_label": item.get("source_label", ""), **fallback})
    actual_engines = list(dict.fromkeys(str(item["asr"].get("actual_engine") or "") for item in result_items if item["asr"].get("actual_engine")))
    actual_models = list(dict.fromkeys(str(item["asr"].get("actual_model") or "") for item in result_items if item["asr"].get("actual_model")))
    if not manifest["asr"].get("actual_engine") and actual_engines:
        manifest["asr"]["actual_engine"] = actual_engines[0] if len(actual_engines) == 1 else "mixed"
    if not manifest["asr"].get("actual_model") and actual_models:
        manifest["asr"]["actual_model"] = actual_models[0] if len(actual_models) == 1 else "mixed"
    postprocess_models = list(
        dict.fromkeys(
            str(model)
            for item in result_items
            for model in ((item.get("enhancement") or {}).get("actual_models") or [])
            if model
        )
    )
    if not manifest["postprocess"].get("actual_models") and postprocess_models:
        manifest["postprocess"]["actual_models"] = postprocess_models
    unique_fallbacks = []
    seen_fallbacks = set()
    for fallback in all_fallbacks:
        try:
            key = json.dumps(fallback, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            key = repr(fallback)
        if key in seen_fallbacks:
            continue
        seen_fallbacks.add(key)
        unique_fallbacks.append(fallback)
    manifest["fallbacks"] = unique_fallbacks
    speaker = manifest.get("speaker_transcript") or {}
    if speaker:
        manifest["diarization"] = {
            **dict(manifest.get("diarization") or {}),
            "status": speaker.get("status") or "ready",
            "backend": speaker.get("backend") or "",
            "backends": speaker.get("backends") or [],
            "fallback_used": bool(speaker.get("fallback_used")),
            "coverage": speaker.get("coverage") or {},
            "failures": speaker.get("failures") or [],
        }
        manifest["speaker_identity"] = {
            **dict(manifest.get("speaker_identity") or {}),
            "status": speaker.get("status") or "ready",
            "speakers": speaker.get("speakers") or [],
            "mapping_path": str(run_dir / "speaker.identity-map.json"),
        }
    if speaker.get("status") == "partial" or manifest.get("partial_count") or manifest.get("error_count"):
        manifest["status"] = "partial"
    elif manifest.get("completed_at") or manifest.get("results"):
        manifest["status"] = "ready"
    else:
        manifest.setdefault("status", "queued")
    return manifest


def _persist_manifest_v3(run_dir: Path, *, job: dict | None = None) -> dict | None:
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    upgraded = _upgrade_manifest_v3(run_dir, manifest, job=job)
    path.write_text(json.dumps(upgraded, ensure_ascii=False, indent=2), encoding="utf-8")
    return upgraded


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def parse_urls(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def serialize_result(payload: dict) -> dict:
    output_dir = Path(payload["output_dir"])
    manifest = json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))
    manifest = _upgrade_manifest_v3(output_dir, manifest)
    Path(payload["manifest_path"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_id": output_dir.name,
        "output_dir": str(output_dir),
        "started_at": manifest.get("started_at") or payload.get("started_at") or "",
        "completed_at": manifest.get("completed_at") or payload.get("completed_at") or "",
        "total_elapsed_seconds": int(manifest.get("elapsed_seconds") or payload.get("elapsed_seconds") or 0),
        "manifest": manifest,
        "errors_path": str(payload["errors_path"]) if payload.get("errors_path") else "",
        "speaker_transcript": load_speaker_transcript(output_dir),
    }


def serialize_run(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _upgrade_manifest_v3(run_dir, manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_id": run_dir.name,
        "output_dir": str(run_dir),
        "started_at": manifest.get("started_at") or "",
        "completed_at": manifest.get("completed_at") or "",
        "total_elapsed_seconds": int(manifest.get("elapsed_seconds") or 0),
        "manifest": manifest,
        "errors_path": str(run_dir / "errors.json") if (run_dir / "errors.json").exists() else "",
        "speaker_transcript": load_speaker_transcript(run_dir),
    }


def load_run_config(run_dir: Path) -> dict | None:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return None
    return json.loads(run_config_path.read_text(encoding="utf-8"))


def load_item_metadata(run_dir: Path) -> list[dict]:
    items_dir = run_dir / "items"
    if not items_dir.exists():
        return []
    rows: list[dict] = []
    for metadata_path in sorted(items_dir.glob("*/metadata.json")):
        try:
            rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rows


def synthesize_manifest_if_complete(run_dir: Path) -> Path | None:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    run_config = load_run_config(run_dir)
    if not run_config:
        return None
    results = load_item_metadata(run_dir)
    source_count = int(run_config.get("source_count") or 0)
    if source_count <= 0 or len(results) < source_count:
        return None
    if any(item.get("status") not in TERMINAL_ITEM_STATUSES for item in results):
        return None

    started_at = run_config.get("started_at") or ""
    # Completion is inferred from the persisted item statuses, never from a
    # directory mtime.  Filesystems can touch a run for unrelated reasons and
    # mtime cannot distinguish a live worker from an abandoned one.
    completed_epoch = time.time()
    completed_at = datetime.fromtimestamp(completed_epoch, UTC).isoformat().replace("+00:00", "Z")
    elapsed_seconds = 0
    if started_at:
        try:
            started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            elapsed_seconds = max(0, round(completed_epoch - started_dt.timestamp()))
        except ValueError:
            elapsed_seconds = 0
    run_config["completed_at"] = completed_at
    run_config["elapsed_seconds"] = elapsed_seconds
    run_config_path = run_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    failed_results = [item for item in results if item.get("status") == "failed"]
    manifest = {
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "generated_at": run_config.get("generated_at") or started_at,
        "model_name": run_config.get("model_name") or "",
        "model_path": run_config.get("model_path") or "",
        "language": run_config.get("language") or "",
        "max_seconds": int(run_config.get("max_seconds") or 0),
        "output_formats": run_config.get("output_formats") or [],
        "enhancement": run_config.get("enhancement") or {},
        "postprocess": run_config.get("postprocess") or {
            "profile": (run_config.get("enhancement") or {}).get("postprocess_profile", "off"),
            "enabled": bool((run_config.get("enhancement") or {}).get("postprocess_profile") not in {None, "off"}),
        },
        "fallbacks": [
            {"source_label": item.get("source_label", ""), **fallback}
            for item in results
            for fallback in (item.get("asr", {}).get("fallbacks") or [])
        ],
        "timings": {"total_elapsed_seconds": elapsed_seconds},
        "artifact_paths": {"items_dir": str(run_dir / "items"), "manifest": str(manifest_path)},
        "run_config_path": str(run_config_path),
        "items_dir": str(run_dir / "items"),
        "results": results,
        "error_count": len(failed_results),
        "partial_count": sum(1 for item in results if (item.get("enhancement") or {}).get("status") == "partial"),
    }
    manifest = _upgrade_manifest_v3(run_dir, manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed_results:
        errors_path = run_dir / "errors.json"
        errors_path.write_text(
            json.dumps(
                {
                    "generated_at": manifest["generated_at"],
                    "items": failed_results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return manifest_path


def reconcile_job_state(job: dict) -> dict:
    current_status = str(job.get("status") or "")
    legacy_running = current_status == "running"
    if current_status in JOB_STATUS_VALUES and current_status in {"ready", "partial", "failed"}:
        return job
    job_type = job.get("job_type") or TRANSCRIPTION_JOB
    if job_type == MODEL_DOWNLOAD_JOB:
        return _reconcile_stale_job(job)
    run_id = job.get("run_id") or ""
    run_dir = RUNS_ROOT / run_id
    if job_type == TRANSCRIPTION_JOB:
        manifest_path = synthesize_manifest_if_complete(run_dir)
        if manifest_path and manifest_path.exists():
            result = serialize_run(run_dir)
            partial = bool(result.get("manifest", {}).get("partial_count"))
            job.update(
                status="completed" if legacy_running else ("partial" if partial else "ready"),
                legacy_status="completed",
                result=result,
                progress=100,
                stage="complete",
                message="任务已自动收尾",
                elapsed_seconds=result.get("total_elapsed_seconds", job.get("elapsed_seconds", 0)),
                error="",
            )
            return job
    elif job_type == SPEAKER_TRANSCRIPT_JOB:
        speaker_transcript = load_speaker_transcript(run_dir)
        if speaker_transcript:
            result = serialize_run(run_dir)
            job.update(
                status="completed" if legacy_running else ("partial" if speaker_transcript.get("status") == "partial" else "ready"),
                legacy_status="completed",
                result=result,
                progress=100,
                stage="complete",
                message="访谈逐字稿已完成",
                error="",
            )
            return job
    return _reconcile_stale_job(job)


def _reconcile_stale_job(job: dict) -> dict:
    """Mark abandoned jobs from their heartbeat, not filesystem activity."""

    heartbeat = float(job.get("heartbeat_at") or job.get("updated_at") or 0)
    if not heartbeat:
        started_at = str(job.get("started_at") or "")
        try:
            heartbeat = datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            heartbeat = 0.0
    if heartbeat and (time.time() - heartbeat) >= STALE_JOB_SECONDS:
        detail = "任务心跳已超过 180 秒没有更新，可能是服务重启或本地进程中断。"
        _set_job_terminal(job, "failed", detail, error=detail)
    return job


def summarize_run_title(manifest: dict) -> str:
    results = manifest.get("results") or []
    if not results:
        return manifest.get("generated_at") or "未命名转录"
    titles = [str(item.get("title") or item.get("source_label") or "").strip() for item in results]
    titles = [title for title in titles if title]
    if not titles:
        return manifest.get("generated_at") or "未命名转录"
    if len(titles) == 1:
        return titles[0]
    return f"{titles[0]} 等 {len(titles)} 条"


def save_uploads(files: Iterable, inbox_dir: Path) -> list[str]:
    sources: list[str] = []
    for file in files:
        filename = secure_filename(file.filename or "")
        if not filename:
            continue
        if not allowed_file(filename):
            raise ValueError(f"Unsupported file type: {filename}")
        target = inbox_dir / filename
        file.save(target)
        sources.append(str(target))
    return sources


def load_speaker_transcript(run_dir: Path) -> dict | None:
    json_path = run_dir / "transcript.speakers.json"
    txt_path = run_dir / "transcript.speakers.txt"
    md_path = run_dir / "transcript.speakers.md"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            payload["paths"] = {
                "json": str(json_path),
                "txt": str(txt_path),
                "md": str(md_path),
            }
            return payload

    # Multi-source runs write one immutable speaker result per item and an
    # aggregate at the run root.  Older runs may only have item-level files;
    # assemble a read-only compatibility view here without changing them.
    item_payloads: list[dict] = []
    items_dir = run_dir / "items"
    if items_dir.exists():
        for item_dir in sorted(path for path in items_dir.iterdir() if path.is_dir()):
            item_json = item_dir / "transcript.speakers.json"
            if not item_json.exists():
                continue
            try:
                item_payload = json.loads(item_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(item_payload, dict):
                item_payloads.append(
                    {
                        "source_label": item_dir.name,
                        "payload": item_payload,
                        "paths": {
                            "json": str(item_json),
                            "txt": str(item_dir / "transcript.speakers.txt"),
                            "md": str(item_dir / "transcript.speakers.md"),
                        },
                    }
                )
    if not item_payloads:
        return None
    return _aggregate_speaker_payload(run_dir, item_payloads, write_files=False)


def _status_for_stage(stage: str | None) -> str | None:
    if not stage:
        return None
    return STAGE_TO_STATUS.get(str(stage))


def _set_job_terminal(job: dict, status: str, message: str, *, error: str = "") -> None:
    """Update a job dict in place, retaining the legacy terminal alias."""

    job["status"] = status
    job["legacy_status"] = "completed" if status in {"ready", "partial"} else "failed"
    job["stage"] = "complete" if status in {"ready", "partial"} else "failed"
    job["message"] = message
    job["error"] = error
    job["updated_at"] = time.time()
    job["heartbeat_at"] = job["updated_at"]


def update_job(job_id: str, **fields: object) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        stage = fields.get("stage")
        previous_stage = str(job.get("stage") or "")
        if stage and str(stage) != previous_stage:
            stage_started = job.get("stage_started_monotonic")
            if isinstance(stage_started, (int, float)) and previous_stage:
                timings = dict(job.get("stage_timings") or {})
                timings[previous_stage] = round(max(0.0, time.monotonic() - float(stage_started)), 3)
                fields["stage_timings"] = timings
            fields["stage_started_monotonic"] = time.monotonic()
        if stage and "status" not in fields:
            status = _status_for_stage(str(stage))
            if status:
                fields["status"] = status
        status = str(fields.get("status") or "")
        if status in {"ready", "partial", "completed"}:
            fields.setdefault("legacy_status", "completed")
        elif status == "failed":
            fields.setdefault("legacy_status", "failed")
        timestamp = time.time()
        fields.setdefault("updated_at", timestamp)
        fields.setdefault("heartbeat_at", timestamp)
        job.update(fields)


def _job_snapshot(job_id: str) -> dict:
    with JOBS_LOCK:
        return dict(JOBS.get(job_id) or {})


def _new_job(job_id: str, job_type: str, *, run_id: str = "", message: str = "任务已创建", **fields: object) -> dict:
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "job_id": job_id,
        "job_type": job_type,
        "run_id": run_id,
        "status": "queued",
        "legacy_status": "running",
        "stage": "queued",
        "message": message,
        "progress": 0,
        "elapsed_seconds": 0,
        "started_at": created_at,
        "created_at": created_at,
        "updated_at": time.time(),
        "heartbeat_at": time.time(),
        "heartbeat_count": 0,
        "resource_metrics_start": _resource_snapshot(),
        "stage_timings": {},
        "stage_started_monotonic": time.monotonic(),
        "result": None,
        "error": "",
        **fields,
    }


def _start_coordinated_heartbeat(job_id: str, lease: HeavyTaskLease, message: str) -> Event:
    def on_heartbeat(snapshot) -> None:
        update_job(
            job_id,
            heartbeat_count=int(_job_snapshot(job_id).get("heartbeat_count") or 0) + 1,
            heartbeat_message=message,
            coordinator=snapshot.to_dict(),
        )

    return HEAVY_TASK_COORDINATOR.start_heartbeat(lease, on_heartbeat)


def _acquire_heavy_task(job_id: str, *, stage: str, message: str) -> HeavyTaskLease:
    def on_wait(position: int) -> None:
        update_job(
            job_id,
            status="waiting_for_model",
            stage="waiting_for_model",
            queue_position=position,
            message="正在等待本地模型空闲",
            progress=1,
        )

    lease = HEAVY_TASK_COORDINATOR.acquire(job_id, stage=stage, on_wait=on_wait)
    update_job(
        job_id,
        status=_status_for_stage(stage) or "transcribing",
        stage=stage,
        queue_position=0,
        message=message,
    )
    return lease


def _safe_run_dir(run_id: str) -> Path | None:
    run_dir = (RUNS_ROOT / str(run_id)).resolve()
    try:
        run_dir.relative_to(RUNS_ROOT.resolve())
    except ValueError:
        return None
    return run_dir


def _manifest_results(run_dir: Path) -> list[dict]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [item for item in payload.get("results", []) if isinstance(item, dict)]


def _resolve_item_dir(run_dir: Path, item: dict) -> Path | None:
    value = item.get("output_dir") or item.get("item_dir")
    candidate = Path(str(value)).expanduser() if value else run_dir / "items" / str(item.get("source_label") or "")
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        return None
    return candidate


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def _write_raw_diarization_artifacts(item_dir: Path, payload: dict) -> None:
    """Persist the first diarization output as an immutable evidence layer."""

    raw_json_path = item_dir / "diarization.raw.json"
    raw_rttm_path = item_dir / "diarization.raw.rttm"
    if raw_json_path.exists():
        return
    turns = payload.get("speaker_turns") or []
    audio_path = Path(str(payload.get("source_audio") or ""))
    raw_turns = []
    rttm_lines: list[str] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        start_ms = max(0, round(float(turn.get("start") or 0) * 1000))
        end_ms = max(start_ms, round(float(turn.get("end") or 0) * 1000))
        raw_label = str(turn.get("raw_speaker_label") or turn.get("speaker") or f"SPEAKER_{index:02d}")
        row = {
            "turn_id": f"turn_{index:05d}",
            "raw_speaker_label": raw_label,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": float(turn.get("confidence") or 0.0),
            "overlap": bool(turn.get("overlap", False)),
        }
        raw_turns.append(row)
        duration = max(0.0, (end_ms - start_ms) / 1000)
        rttm_lines.append(f"SPEAKER {item_dir.name} 1 {start_ms / 1000:.3f} {duration:.3f} <NA> <NA> {raw_label} <NA> <NA>")
    raw_payload = {
        "schema_version": 1,
        "backend": str(payload.get("backend") or "unknown"),
        "model": str(payload.get("diarization_model") or payload.get("model") or ""),
        "model_revision": str(payload.get("model_revision") or ""),
        "model_digest": str(payload.get("model_digest") or ""),
        "audio_sha256": _sha256_file(audio_path),
        "turns": raw_turns,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    raw_json_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_rttm_path.write_text("\n".join(rttm_lines) + ("\n" if rttm_lines else ""), encoding="utf-8")


def _copy_immutable_artifact(source: Path, target: Path) -> None:
    """Copy a generated evidence artifact only when no immutable target exists."""

    if target.exists() or not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _rewrite_speaker_paths(value: object, *, staging: Path, item_dir: Path) -> object:
    """Remove transient staging paths from persisted speaker JSON."""

    if isinstance(value, dict):
        return {key: _rewrite_speaker_paths(child, staging=staging, item_dir=item_dir) for key, child in value.items()}
    if isinstance(value, list):
        return [_rewrite_speaker_paths(child, staging=staging, item_dir=item_dir) for child in value]
    if isinstance(value, str):
        prefix = str(staging)
        if value == prefix or value.startswith(prefix + os.sep) or value.startswith(prefix + "/"):
            return str(item_dir) + value[len(prefix):]
    return value


def _speaker_payload_v3(payload: dict, *, source_label: str = "") -> dict:
    turns = [item for item in (payload.get("speaker_turns") or []) if isinstance(item, dict)]
    labels = list(dict.fromkeys(str(item.get("raw_speaker") or item.get("speaker_id") or item.get("speaker") or "Speaker 1") for item in turns))
    name_map = payload.get("speaker_map") or {}
    inferred = payload.get("speaker_name_inference") or {}
    speakers = []
    existing_speakers = [item for item in (payload.get("speakers") or []) if isinstance(item, dict)]
    by_id = {
        str(item.get("speaker_id") or (item.get("raw_labels") or [""])[0] or item.get("display_name") or ""): item
        for item in existing_speakers
    }
    for label in labels:
        existing = dict(by_id.get(label) or {})
        display_name = str(existing.get("display_name") or name_map.get(label) or "").strip()
        identity_status = str(existing.get("identity_status") or ("resolved" if display_name and display_name != label else "unknown"))
        evidence = list(existing.get("identity_evidence") or [])
        speakers.append(
            {
                **existing,
                "speaker_id": str(existing.get("speaker_id") or label),
                "raw_labels": existing.get("raw_labels") or [label],
                "display_name": display_name or label,
                "confidence": float(existing.get("confidence") or (0.9 if identity_status == "resolved" else 0.0)),
                "identity_status": identity_status,
                "identity_evidence": evidence,
            }
        )
    coverage = _coverage_for_turns(turns)
    output = dict(payload)
    output.update(
        {
            "schema_version": 3,
            "source_label": source_label or str(payload.get("source_label") or ""),
            "speakers": speakers,
            "coverage": coverage,
            "identity_status": "resolved" if any(item["identity_status"] == "resolved" for item in speakers) else "unknown",
            "fallback_reason": "backend is best-effort" if payload.get("fallback_used") else "",
            "speaker_name_inference": inferred,
        }
    )
    return output


def _coverage_for_turns(turns: list[dict]) -> dict:
    starts = [float(item.get("start") or 0) for item in turns if isinstance(item, dict)]
    ends = [float(item.get("end") or 0) for item in turns if isinstance(item, dict)]
    if not starts or not ends:
        return {"start_ms": 0, "end_ms": 0, "duration_ms": 0, "known": False}
    start_ms = max(0, round(min(starts) * 1000))
    end_ms = max(start_ms, round(max(ends) * 1000))
    return {"start_ms": start_ms, "end_ms": end_ms, "duration_ms": end_ms - start_ms, "known": True}


def _write_speaker_text_files(run_dir: Path, turns: list[dict]) -> None:
    blocks: list[str] = []
    markdown: list[str] = []
    for turn in turns:
        speaker = str(turn.get("speaker") or "Speaker 1")
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        source = str(turn.get("source_label") or "").strip()
        prefix = f"[{source}] " if source else ""
        blocks.append(f"{prefix}{speaker}：{text}")
        start = float(turn.get("start") or 0)
        end = float(turn.get("end") or 0)
        markdown.append(f"{prefix}{speaker} [{int(start // 3600):02d}:{int(start % 3600 // 60):02d}:{int(start % 60):02d} - {int(end // 3600):02d}:{int(end % 3600 // 60):02d}:{int(end % 60):02d}]：{text}")
    (run_dir / "transcript.speakers.txt").write_text("\n\n".join(blocks), encoding="utf-8")
    (run_dir / "transcript.speakers.md").write_text("\n\n".join(markdown), encoding="utf-8")


def _aggregate_speaker_payload(run_dir: Path, item_payloads: list[dict], *, write_files: bool = True) -> dict:
    item_views = []
    all_turns: list[dict] = []
    all_speakers: list[dict] = []
    backends: list[str] = []
    any_fallback = False
    has_partial = False
    for item in item_payloads:
        source_label = str(item.get("source_label") or "")
        payload = _speaker_payload_v3(dict(item.get("payload") or {}), source_label=source_label)
        item_views.append({"source_label": source_label, "status": payload.get("status", "ready"), "payload": payload, "paths": item.get("paths") or {}})
        for turn in payload.get("speaker_turns") or []:
            if isinstance(turn, dict):
                all_turns.append({**turn, "source_label": source_label})
        for speaker in payload.get("speakers") or []:
            if isinstance(speaker, dict):
                all_speakers.append({**speaker, "source_label": source_label})
        backend = str(payload.get("backend") or "")
        if backend and backend not in backends:
            backends.append(backend)
        any_fallback = any_fallback or bool(payload.get("fallback_used"))
        has_partial = has_partial or payload.get("status") == "partial"
    aggregate = {
        "schema_version": 3,
        "status": "partial" if has_partial else "ready",
        "items": item_views,
        "speaker_turns": all_turns,
        "speakers": all_speakers,
        "speaker_count_detected": len({str(item.get("speaker") or "") for item in all_turns if item.get("speaker")}),
        "backend": backends[0] if len(backends) == 1 else ("mixed" if backends else ""),
        "backends": backends,
        "fallback_used": any_fallback,
        "quality_note": "本地说话人分离结果，建议人工复核" if any_fallback else "说话人分离已完成",
        "coverage": _coverage_for_turns(all_turns),
        "paths": {
            "json": str(run_dir / "transcript.speakers.json"),
            "txt": str(run_dir / "transcript.speakers.txt"),
            "md": str(run_dir / "transcript.speakers.md"),
        },
    }
    if write_files:
        (run_dir / "transcript.speakers.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_speaker_text_files(run_dir, all_turns)
        (run_dir / "speaker_map.json").write_text(
            json.dumps({str(item.get("speaker") or ""): str(item.get("display_name") or "") for item in all_speakers}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return aggregate


def _speaker_item_payload(item_dir: Path, source_label: str) -> dict:
    json_path = item_dir / "transcript.speakers.json"
    if not json_path.exists():
        return {"source_label": source_label, "payload": {}, "paths": {}}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return {
        "source_label": source_label,
        "payload": payload if isinstance(payload, dict) else {},
        "paths": {
            "json": str(json_path),
            "txt": str(item_dir / "transcript.speakers.txt"),
            "md": str(item_dir / "transcript.speakers.md"),
        },
    }


def _run_speaker_for_items(
    job_id: str,
    run_dir: Path,
    *,
    required: bool = False,
    backend: str = "auto",
    min_speakers: int = 2,
    max_speakers: int = 6,
) -> tuple[dict | None, list[dict]]:
    """Run diarization independently per source and retain raw ASR on errors."""

    item_payloads: list[dict] = []
    failures: list[dict] = []
    results = _manifest_results(run_dir)
    for item in results:
        source_label = str(item.get("source_label") or "")
        item_dir = _resolve_item_dir(run_dir, item)
        if item_dir is None or not (item_dir / "transcript.json").exists():
            failures.append({"source_label": source_label, "error": "原始转录或条目目录不存在"})
            continue
        try:
            # build_speaker_transcript historically chooses the first
            # transcript*.json file in a directory, which may be an enhanced
            # artifact.  Isolate the raw transcript and audio in a short-lived
            # staging directory so the immutable ASR evidence is always used.
            stage_dir = item_dir / ".speaker-staging"
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
            stage_dir.mkdir(parents=True, exist_ok=True)
            raw_transcript = stage_dir / "transcript.json"
            shutil.copy2(item_dir / "transcript.json", raw_transcript)
            audio_candidates = [item_dir / "转录音频.wav", item_dir / "audio.wav"]
            audio_path = next((path for path in audio_candidates if path.exists()), None)
            if audio_path is None:
                raise FileNotFoundError("转录音频.wav")
            staged_audio = stage_dir / "转录音频.wav"
            try:
                staged_audio.symlink_to(audio_path)
            except OSError:
                shutil.copy2(audio_path, staged_audio)
            context_path = item_dir / "source.context.json"
            if context_path.exists():
                shutil.copy2(context_path, stage_dir / context_path.name)
            def speaker_progress(event: dict) -> None:
                event_stage = str(event.get("stage") or "")
                if event_stage == "align":
                    update_job(job_id, status="aligning", stage="align", message=event.get("message") or "正在对齐转写内容", progress=event.get("progress", 75))
                else:
                    update_job(job_id, **event)

            speaker_kwargs = {"progress_callback": speaker_progress}
            if backend and backend != "auto":
                speaker_kwargs["backend"] = backend
            if min_speakers != 2:
                speaker_kwargs["min_speakers"] = min_speakers
            if max_speakers != 6:
                speaker_kwargs["max_speakers"] = max_speakers
            payload = build_speaker_transcript(stage_dir, **speaker_kwargs)
            update_job(job_id, status="identifying_speakers", stage="identifying_speakers", message="正在识别说话人身份", progress=86)
            payload = _speaker_payload_v3(payload, source_label=source_label)
            payload["run_dir"] = str(item_dir)
            payload["source_audio"] = str(audio_path)
            payload = _rewrite_speaker_paths(payload, staging=stage_dir, item_dir=item_dir)
            # Keep the app contract testable with lightweight/dynamic
            # backends that return a payload without using the legacy writer.
            if not (stage_dir / "transcript.speakers.json").exists():
                (stage_dir / "transcript.speakers.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if not (stage_dir / "transcript.speakers.txt").exists() or not (stage_dir / "transcript.speakers.md").exists():
                _write_speaker_text_files(stage_dir, payload.get("speaker_turns") or [])
            if not (stage_dir / "speaker_map.json").exists():
                (stage_dir / "speaker_map.json").write_text(json.dumps(payload.get("speaker_map") or {}, ensure_ascii=False, indent=2), encoding="utf-8")
            _copy_immutable_artifact(stage_dir / "diarization.raw.json", item_dir / "diarization.raw.json")
            _copy_immutable_artifact(stage_dir / "diarization.raw.rttm", item_dir / "diarization.raw.rttm")
            staged_voice_reference = stage_dir / "derived" / "audio.voice_reference.wav"
            stable_voice_reference = item_dir / "derived" / "audio.voice_reference.wav"
            _copy_immutable_artifact(staged_voice_reference, stable_voice_reference)
            if not (item_dir / "diarization.raw.json").exists():
                _write_raw_diarization_artifacts(item_dir, payload)
            payload = _rewrite_speaker_paths(payload, staging=stage_dir, item_dir=item_dir)
            if isinstance(payload.get("diarization_raw"), dict):
                payload["diarization_raw"]["json_path"] = str(item_dir / "diarization.raw.json")
                payload["diarization_raw"]["rttm_path"] = str(item_dir / "diarization.raw.rttm")
            if isinstance(payload.get("preprocess"), dict):
                payload["preprocess"]["voice_reference_audio"] = str(stable_voice_reference)
            for filename in ("transcript.speakers.json", "transcript.speakers.txt", "transcript.speakers.md", "speaker_map.json"):
                source_path = stage_dir / filename
                if source_path.exists():
                    target_path = item_dir / filename
                    if filename == "transcript.speakers.json":
                        source_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    source_path.replace(target_path)
            shutil.rmtree(stage_dir, ignore_errors=True)
            item_payloads.append(_speaker_item_payload(item_dir, source_label))
        except Exception as exc:
            failures.append({"source_label": source_label, "error": str(exc)})
            shutil.rmtree(item_dir / ".speaker-staging", ignore_errors=True)
            if required:
                # Keep processing out of the worker; the caller can mark the
                # task failed while the original transcript remains on disk.
                continue
    if not item_payloads:
        return None, failures
    aggregate = _aggregate_speaker_payload(run_dir, item_payloads)
    if failures:
        aggregate["status"] = "partial"
        aggregate["failures"] = failures
        (run_dir / "transcript.speakers.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate, failures


def _annotate_manifest_speaker(run_dir: Path, aggregate: dict | None, failures: list[dict]) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    aggregate = aggregate or {}
    by_source = {str(item.get("source_label") or ""): item for item in aggregate.get("items") or [] if isinstance(item, dict)}
    for item in manifest.get("results") or []:
        source_label = str(item.get("source_label") or "")
        view = by_source.get(source_label)
        if view:
            payload = view.get("payload") or {}
            item["speaker_status"] = "ready"
            item["speaker_transcript"] = view.get("paths") or {}
            item["diarization"] = {
                "backend": payload.get("backend") or "",
                "model": payload.get("model") or (payload.get("diarization") or {}).get("model") or "",
                "model_revision": payload.get("model_revision") or (payload.get("diarization") or {}).get("model_revision") or "",
                "model_digest": payload.get("model_digest") or (payload.get("diarization") or {}).get("model_digest") or "",
                "fallback_used": bool(payload.get("fallback_used")),
                "fallback_reason": payload.get("fallback_reason") or "",
                "coverage": payload.get("coverage") or {},
                "speaker_count_detected": payload.get("speaker_count_detected") or 0,
                "raw": payload.get("diarization_raw") or {},
            }
            item["speaker_identity"] = {
                "schema_version": 3,
                "status": payload.get("identity_status") or "ready",
                "speakers": payload.get("speakers") or [],
                "mapping_path": str(run_dir / "speaker.identity-map.json"),
            }
        elif any(str(error.get("source_label") or "") == source_label for error in failures):
            item["speaker_status"] = "partial"
    manifest["speaker_transcript"] = {
        "status": aggregate.get("status") or ("partial" if failures else "not_started"),
        "backend": aggregate.get("backend") or "",
        "backends": aggregate.get("backends") or [],
        "fallback_used": bool(aggregate.get("fallback_used")),
        "coverage": aggregate.get("coverage") or {},
        "speakers": aggregate.get("speakers") or [],
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_transcription_job(
    job_id: str,
    sources: list[str],
    run_dir: Path,
    model: str,
    language: str,
    max_seconds: int,
    glossary_text: str,
    ai_polish: bool,
    ai_style: str,
    asr_engine: str = "auto",
    asr_model: str = "",
    postprocess_profile: str = "off",
    speaker_mode: str = "auto",
) -> None:
    started_monotonic = time.monotonic()

    def on_progress(event: dict) -> None:
        update_job(
            job_id,
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
            **event,
        )
    lease: HeavyTaskLease | None = None
    stop_heartbeat: Event | None = None
    try:
        lease = _acquire_heavy_task(job_id, stage="model", message="本地模型已占用，任务开始处理")
        stop_heartbeat = _start_coordinated_heartbeat(job_id, lease, "本地模型正在持续处理，请稍候")
        payload = transcribe_media_sources(
            sources,
            output_dir=run_dir,
            model_name=model,
            language=language,
            max_seconds=max_seconds,
            glossary_text=glossary_text,
            enable_enhancement=True,
            ai_polish=ai_polish,
            ai_style=ai_style,
            asr_engine=asr_engine,
            asr_model=asr_model,
            postprocess_profile=postprocess_profile,
            progress_callback=on_progress,
        )
        speaker_payload = None
        speaker_failures: list[dict] = []
        if speaker_mode != "off":
            lease.set_stage("diarization")
            update_job(job_id, status="diarizing", stage="diarization", message="正在逐条区分说话人", progress=96)
            speaker_payload, speaker_failures = _run_speaker_for_items(
                job_id,
                run_dir,
                required=speaker_mode == "required",
            )
            _annotate_manifest_speaker(run_dir, speaker_payload, speaker_failures)
            if speaker_failures:
                detail = "；".join(f"{item.get('source_label')}: {item.get('error')}" for item in speaker_failures)
                update_job(job_id, message=f"部分说话人结果未完成，原始转录仍可使用：{detail}", progress=98)
                if speaker_mode == "required":
                    raise RuntimeError(f"说话人区分未完成：{detail}")
        _persist_manifest_v3(run_dir, job=_job_snapshot(job_id))
        result = serialize_result(payload)
        if speaker_payload is not None:
            result["speaker_transcript"] = speaker_payload
        failed_count = int(result["manifest"].get("error_count") or 0)
        partial_count = int(result["manifest"].get("partial_count") or 0)
        total_count = len(result["manifest"].get("results", []))
        message = "转写完成"
        is_partial = bool(partial_count or speaker_failures)
        if partial_count:
            message = f"转写完成，{partial_count} 条智能加工部分完成，原始稿仍可使用"
        if speaker_failures:
            message = f"转写完成，{len(speaker_failures)} 条说话人结果部分完成，原始稿仍可使用"
        if failed_count and failed_count < total_count:
            message = f"转写完成，{failed_count} 条失败"
        elif failed_count and failed_count == total_count:
            message = "任务完成，但所有条目都失败了"
            is_partial = True
        update_job(job_id, status="packaging", stage="complete", message="正在封装结果", progress=99)
        update_job(
            job_id,
            status="partial" if is_partial else "ready",
            result=result,
            progress=100,
            stage="complete",
            message=message,
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
            speaker_failures=speaker_failures,
        )
    except Exception as exc:  # local operator tool: surface exact failure
        update_job(
            job_id,
            status="failed",
            stage="failed",
            message=str(exc),
            error=str(exc),
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
        )
    finally:
        if stop_heartbeat is not None:
            stop_heartbeat.set()
        if lease is not None:
            lease.release()


def run_speaker_transcript_job(
    job_id: str,
    run_dir: Path,
    backend: str = "auto",
    min_speakers: int = 2,
    max_speakers: int = 6,
) -> None:
    started_monotonic = time.monotonic()
    lease: HeavyTaskLease | None = None
    stop_heartbeat: Event | None = None
    try:
        lease = _acquire_heavy_task(job_id, stage="diarization", message="本地模型已占用，开始说话人分离")
        stop_heartbeat = _start_coordinated_heartbeat(job_id, lease, "说话人模型正在持续处理，请稍候")
        speaker_payload, failures = _run_speaker_for_items(
            job_id,
            run_dir,
            backend=backend,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        _annotate_manifest_speaker(run_dir, speaker_payload, failures)
        _persist_manifest_v3(run_dir, job=_job_snapshot(job_id))
        if speaker_payload is None and failures:
            raise RuntimeError("；".join(str(item.get("error") or "说话人分离失败") for item in failures))
        status = "partial" if failures else "ready"
        update_job(
            job_id,
            status="packaging",
            stage="complete",
            progress=99,
            message="正在封装访谈逐字稿",
        )
        _persist_manifest_v3(run_dir, job=_job_snapshot(job_id))
        update_job(
            job_id,
            status=status,
            result=serialize_run(run_dir),
            progress=100,
            stage="complete",
            message="访谈逐字稿已完成",
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
            speaker_failures=failures,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            stage="failed",
            message=str(exc),
            error=str(exc),
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
        )
    finally:
        if stop_heartbeat is not None:
            stop_heartbeat.set()
        if lease is not None:
            lease.release()


def update_enhancement_metadata(run_dir: Path, item_dir: Path, audit: dict) -> None:
    """Expose new artifacts in existing metadata/manifest without replacing raw output."""
    metadata_path = item_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["enhancement"] = audit
        metadata.update(audit.get("paths") or {})
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("results") or []:
        item_path = item.get("item_dir") or item.get("output_dir") or ""
        if Path(str(item_path)).resolve() == item_dir.resolve():
            item["enhancement"] = audit
            item.update(audit.get("paths") or {})
    manifest.setdefault("postprocess", {})["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_enhancement_job(job_id: str, run_dir: Path, profile: str, glossary_text: str, style: str) -> None:
    started_monotonic = time.monotonic()

    def on_progress(event: dict) -> None:
        update_job(job_id, elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)), **event)
    lease: HeavyTaskLease | None = None
    stop_heartbeat: Event | None = None
    try:
        lease = _acquire_heavy_task(job_id, stage="polishing", message="本地模型已占用，开始智能整理")
        stop_heartbeat = _start_coordinated_heartbeat(job_id, lease, "本地模型正在持续整理，请稍候")
        # Resolve once per historical job so every item uses the same
        # validated central selection (and a config override is not lost).
        effective_config = load_config()
        item_dirs = sorted((run_dir / "items").glob("*"))
        processed = 0
        partial = 0
        for item_dir in item_dirs:
            if not (item_dir / "transcript.json").exists():
                continue
            audit = process_transcript_artifacts(
                item_dir,
                profile=profile,
                user_glossary_text=glossary_text,
                style=style,
                config=effective_config,
                progress_callback=on_progress,
            )
            update_enhancement_metadata(run_dir, item_dir, audit)
            processed += 1
            partial += int(audit.get("status") == "partial")
        if not processed:
            raise FileNotFoundError("这个历史任务没有可供整理的 transcript.json")
        _persist_manifest_v3(run_dir, job=_job_snapshot(job_id))
        update_job(job_id, status="packaging", stage="complete", progress=99, message="正在封装智能整理结果")
        update_job(
            job_id,
            status="partial" if partial else "ready",
            result=serialize_run(run_dir),
            progress=100,
            stage="complete",
            message="智能整理与知识摘要已完成" + ("，部分条目保留原始稿" if partial else ""),
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
            partial_count=partial,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            stage="failed",
            message=str(exc),
            error=str(exc),
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
        )
    finally:
        if stop_heartbeat is not None:
            stop_heartbeat.set()
        if lease is not None:
            lease.release()


def run_model_download_job(job_id: str, kind: str) -> None:
    accurate_model = _configured_model(qwen_accurate_model, QWEN_ACCURATE_MODEL)
    fast_model = _configured_model(qwen_fast_model, QWEN_FAST_MODEL)
    aligner_model = _configured_model(qwen_forced_aligner_model, QWEN_FORCED_ALIGNER_MODEL)
    model_map = {
        "asr-accurate": ("asr", accurate_model),
        "asr-fast": ("asr", fast_model),
        "asr-forced-aligner": ("asr-aligner", aligner_model),
        "llm-fast": ("ollama", FAST_MODEL),
        "llm-quality": ("ollama", QUALITY_MODEL),
    }
    target = model_map.get(kind)
    if target is None:
        update_job(job_id, status="failed", stage="failed", message="未知模型类型", error="未知模型类型")
        return
    provider, model = target
    lease: HeavyTaskLease | None = None
    stop_heartbeat: Event | None = None
    try:
        lease = _acquire_heavy_task(job_id, stage="downloading", message=f"正在下载 {model}")
        stop_heartbeat = _start_coordinated_heartbeat(job_id, lease, "模型下载正在持续进行，请稍候")
        update_job(job_id, status="downloading", stage="downloading", message=f"正在下载 {model}", progress=5)
        if provider in {"asr", "asr-aligner"}:
            if not qwen_mlx_available():
                raise RuntimeError("Qwen ASR 仅可在安装 mlx-qwen3-asr 的 Apple Silicon Mac 上下载")
            repos = [model] if provider == "asr-aligner" else [model, aligner_model]
            command = [
                sys.executable,
                "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    f"[snapshot_download(repo, allow_patterns=['*.json','*.safetensors','*.txt','*.model']) for repo in {repos!r}]"
                ),
            ]
        else:
            command = ["ollama", "pull", model]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError((completed.stderr or completed.stdout or "模型下载失败").strip())
        update_job(job_id, status="ready", stage="complete", message=f"{model} 已准备好", progress=100)
    except Exception as exc:
        update_job(job_id, status="failed", stage="failed", message=str(exc), error=str(exc))
    finally:
        if stop_heartbeat is not None:
            stop_heartbeat.set()
        if lease is not None:
            lease.release()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "NextEcho",
            "runs_root": str(RUNS_ROOT),
        }
    )


@app.post("/api/transcribe")
def transcribe():
    urls = parse_urls(request.form.get("urls", ""))
    quality = request.form.get("quality", "accurate").strip() or "accurate"
    model = QUALITY_TO_MODEL.get(quality, DEFAULT_MODEL_NAME)
    language = "auto"
    max_seconds = 0
    glossary_text = request.form.get("glossary", "").strip()
    ai_polish = request.form.get("ai_polish", "").strip().lower() in {"1", "true", "on", "yes"}
    ai_style = request.form.get("ai_style", "").strip()
    asr_engine = request.form.get("asr_engine", "auto").strip() or "auto"
    asr_model = request.form.get("asr_model", "").strip()
    postprocess_profile = request.form.get("postprocess_profile", "fast").strip() or "fast"
    speaker_mode = request.form.get("speaker_mode", "auto").strip() or "auto"
    if postprocess_profile not in {"off", "balanced", "quality", "fast"}:
        return jsonify({"error": "Unsupported postprocess profile."}), 400
    if asr_engine not in {"auto", "qwen3-asr", "whisper-cpp"}:
        return jsonify({"error": "Unsupported ASR engine."}), 400
    if speaker_mode not in {"off", "auto", "required"}:
        return jsonify({"error": "Unsupported speaker mode."}), 400
    # Compatibility with the original checkbox: it now means the balanced local workflow.
    if ai_polish and request.form.get("postprocess_profile") is None:
        postprocess_profile = "fast"

    run_id = f"run_{now_slug()}_{uuid4().hex[:8]}"
    run_dir = RUNS_ROOT / run_id
    inbox_dir = run_dir / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    try:
        uploaded_sources = save_uploads(request.files.getlist("files"), inbox_dir)
    except ValueError as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400

    sources = [*uploaded_sources, *urls]
    if not sources:
        shutil.rmtree(run_dir, ignore_errors=True)
        return jsonify({"error": "Please add at least one local file or remote URL."}), 400

    job_id = uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = _new_job(
            job_id,
            TRANSCRIPTION_JOB,
            run_id=run_id,
            message="任务已创建",
            asr_engine=asr_engine,
            asr_model=asr_model,
            postprocess_profile=postprocess_profile,
            speaker_mode=speaker_mode,
        )
    Thread(
        target=run_transcription_job,
        args=(
            job_id, sources, run_dir, model, language, max_seconds, glossary_text, ai_polish, ai_style,
            asr_engine, asr_model, postprocess_profile,
            speaker_mode,
        ),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "run_id": run_id}), 202


@app.get("/api/models/status")
def model_status():
    config_error = ""
    try:
        effective_config = load_config()
    except Exception as exc:
        # Keep the status endpoint useful even when an explicitly supplied
        # local override is malformed; callers still see the fallback choice.
        effective_config = None
        config_error = str(exc)
    configured_accurate = str(_effective_config_value(effective_config, "asr", "accurate_model", "") or "").strip() or _configured_model(qwen_accurate_model, QWEN_ACCURATE_MODEL, effective_config)
    configured_fast = str(_effective_config_value(effective_config, "asr", "fast_model", "") or "").strip() or _configured_model(qwen_fast_model, QWEN_FAST_MODEL, effective_config)
    configured_aligner = str(_effective_config_value(effective_config, "asr", "forced_aligner", "") or "").strip() or _configured_model(qwen_forced_aligner_model, QWEN_FORCED_ALIGNER_MODEL, effective_config)
    try:
        speaker_backend = detect_speaker_backend_status(effective_config)
    except TypeError:
        # Compatibility with injected status providers that predate config.
        try:
            speaker_backend = detect_speaker_backend_status()
        except Exception as exc:
            speaker_backend = {"ready": False, "selected_backend": "", "error": str(exc)}
    except Exception as exc:
        speaker_backend = {"ready": False, "selected_backend": "", "error": str(exc)}
    try:
        ollama = ollama_status()
    except Exception as exc:
        ollama = {"ready": False, "models": [], "error": str(exc)}
    postprocess_profile = str(_effective_config_value(effective_config, "postprocess", "profile", "fast") or "fast").strip().lower()
    if postprocess_profile not in {"fast", "balanced", "quality"}:
        postprocess_profile = "fast"
    try:
        postprocess_plan = resolve_postprocess_models(
            postprocess_profile,
            config=effective_config,
            ollama_info=ollama,
        )
    except Exception as exc:
        postprocess_plan = {
            "provider": str(_effective_config_value(effective_config, "postprocess", "provider", "selection") or "selection"),
            "profile": postprocess_profile,
            "selected_4b": str(_effective_config_value(effective_config, "postprocess", "fallback_model", FAST_MODEL) or FAST_MODEL),
            "quality_model": str(_effective_config_value(effective_config, "postprocess", "quality_model", QUALITY_MODEL) or QUALITY_MODEL),
            "error": str(exc),
        }
    configured_llm_fast = str(postprocess_plan.get("selected_4b") or FAST_MODEL)
    configured_llm_quality = str(postprocess_plan.get("quality_model") or QUALITY_MODEL)
    configured_asr_engine = str(_effective_config_value(effective_config, "asr", "engine", "auto") or "auto").strip().lower()
    qwen_runtime_ready = qwen_mlx_available() and qwen_memory_safe()
    if configured_asr_engine == "whisper-cpp" or (configured_asr_engine not in {"qwen3-asr", "auto"} and not qwen_runtime_ready):
        selected_asr_engine = "whisper-cpp"
        selected_asr_model = str(_effective_config_value(effective_config, "asr", "whisper_fallback", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME)
    elif qwen_runtime_ready:
        selected_asr_engine = "qwen3-asr"
        selected_asr_model = configured_accurate
    else:
        selected_asr_engine = "whisper-cpp"
        selected_asr_model = str(_effective_config_value(effective_config, "asr", "whisper_fallback", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME)
    ollama_models = {
        str(item.get("name") or item.get("model") or "")
        for item in (ollama.get("models") or [])
        if isinstance(item, dict)
    }
    ollama_models.update(str(item) for item in (ollama.get("models") or []) if not isinstance(item, dict) and str(item).strip())
    if ollama_models:
        ollama["fast_model_ready"] = configured_llm_fast in ollama_models
        ollama["quality_model_ready"] = configured_llm_quality in ollama_models
    coordinator = HEAVY_TASK_COORDINATOR.snapshot().to_dict()
    return jsonify(
        {
            "apple_silicon": is_apple_silicon(),
            "qwen_mlx_available": qwen_mlx_available(),
            "qwen_memory_safe": qwen_memory_safe(),
            # Keep the original two keys for old clients; the configured IDs
            # and provenance below are the authoritative selection view.
            "asr_models": {"accurate": QWEN_ACCURATE_MODEL, "fast": QWEN_FAST_MODEL, "configured_accurate": configured_accurate, "configured_fast": configured_fast},
            "qwen_asr_model_cache": {
                "accurate": qwen_model_cached(configured_accurate),
                "fast": qwen_model_cached(configured_fast),
            },
            "qwen_forced_aligner": {
                "model": configured_aligner,
                "cached": qwen_forced_aligner_cached(configured_aligner),
                "provenance": _model_provenance(configured_aligner),
            },
            "qwen_asr_provenance": {
                "accurate": _model_provenance(configured_accurate),
                "fast": _model_provenance(configured_fast),
            },
            "ollama": ollama,
            "llm_models": {"fast": configured_llm_fast, "quality": configured_llm_quality},
            "postprocess": {
                "profile": postprocess_profile,
                "provider": postprocess_plan.get("provider") or "selection",
                "selected_4b": configured_llm_fast,
                "quality_model": configured_llm_quality,
                "config_source": postprocess_plan.get("config_source") or getattr(effective_config, "source_label", "fallback"),
                "error": postprocess_plan.get("error") or config_error,
            },
            "diarization": {
                **speaker_backend,
                "requested_backend": _effective_config_value(effective_config, "diarization", "backend", "auto"),
                "fallback_order": _effective_config_value(effective_config, "diarization", "fallback_order", []),
                "model": "pyannote/speaker-diarization-community-1" if speaker_backend.get("selected_backend") == "pyannote" else "",
                "fallback": "resemblyzer" if speaker_backend.get("selected_backend") == "resemblyzer" else "segment-clustering" if speaker_backend.get("selected_backend") == "segment-clustering" else "",
            },
            "selected": {
                "asr": {"engine": selected_asr_engine, "model": selected_asr_model, "requested_engine": configured_asr_engine},
                "diarization": {"backend": speaker_backend.get("selected_backend") or "unavailable"},
                "llm": {"backend": postprocess_plan.get("provider") or "ollama", "model": configured_llm_fast},
            },
            "config": {
                "source": getattr(effective_config, "source", "fallback") if effective_config is not None else "fallback",
                "source_label": getattr(effective_config, "source_label", "fallback") if effective_config is not None else "fallback",
            },
            "coordinator": coordinator,
            "heartbeat_seconds": HEARTBEAT_SECONDS,
        }
    )


@app.post("/api/models/download")
def download_model():
    payload = request.get_json(silent=True) if request.is_json else request.form
    kind = str((payload or {}).get("kind") or "")
    if kind not in {"asr-accurate", "asr-fast", "asr-forced-aligner", "llm-fast", "llm-quality"}:
        return jsonify({"error": "Unsupported model type."}), 400
    job_id = uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = _new_job(job_id, MODEL_DOWNLOAD_JOB, message="模型下载任务已创建", model_kind=kind)
    Thread(target=run_model_download_job, args=(job_id, kind), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.post("/api/resolve")
def resolve():
    payload = request.get_json(silent=True) if request.is_json else None
    raw_urls = payload.get("urls", "") if isinstance(payload, dict) else request.form.get("urls", "")
    urls = parse_urls(raw_urls)
    if not urls:
        return jsonify({"error": "Please add at least one remote URL."}), 400
    return jsonify({"sources": [item.to_dict() for item in resolve_sources(urls)]})


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        reconcile_job_state(job)
        return jsonify(job)


@app.get("/api/runs")
def list_runs():
    runs = []
    for run_dir in sorted(RUNS_ROOT.glob("run_*"), reverse=True):
        manifest_path = synthesize_manifest_if_complete(run_dir) or (run_dir / "manifest.json")
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        display_title = summarize_run_title(manifest)
        runs.append(
            {
                "run_id": manifest_path.parent.name,
                "display_title": display_title,
                "generated_at": manifest.get("generated_at"),
                "item_count": len(manifest.get("results", [])),
                "error_count": int(manifest.get("error_count") or 0),
                "model_name": manifest.get("model_name"),
                "quality": "更快" if manifest.get("model_name") == "base" else "高精度",
            }
        )
    return jsonify({"runs": runs[:20]})


@app.get("/api/runs/<run_id>")
def get_run(run_id: str):
    run_dir = _safe_run_dir(run_id)
    if run_dir is None:
        return jsonify({"error": "run not found"}), 404
    manifest_path = synthesize_manifest_if_complete(run_dir) or (run_dir / "manifest.json")
    if not manifest_path.exists():
        return jsonify({"error": "run not found"}), 404
    return jsonify(serialize_run(run_dir))


@app.post("/api/runs/<run_id>/speaker-transcript")
@app.post("/api/runs/<run_id>/re-diarize")
@app.post("/api/runs/<run_id>/rediarize")
def create_speaker_transcript(run_id: str):
    run_dir = _safe_run_dir(run_id)
    if run_dir is None or not (run_dir / "manifest.json").exists():
        return jsonify({"error": "run not found"}), 404
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload if isinstance(payload, dict) else {}
    backend = str(payload.get("backend") or "auto").strip().lower() or "auto"
    if backend not in {"auto", "pyannote", "pyannote-community-1", "sherpa", "sherpa-onnx", "resemblyzer", "segment-clustering", "fallback"}:
        return jsonify({"error": "Unsupported diarization backend."}), 400
    try:
        min_speakers = max(1, min(32, int(payload.get("min_speakers", 2))))
        max_speakers = max(min_speakers, min(32, int(payload.get("max_speakers", 6))))
    except (TypeError, ValueError):
        return jsonify({"error": "min_speakers and max_speakers must be integers"}), 400
    job_id = uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = _new_job(
            job_id,
            SPEAKER_TRANSCRIPT_JOB,
            run_id=run_id,
            message="访谈逐字稿任务已创建",
            backend=backend,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            re_diarize=request.path.endswith(("re-diarize", "rediarize")),
        )
    Thread(
        target=run_speaker_transcript_job,
        args=(job_id, run_dir, backend, min_speakers, max_speakers),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "run_id": run_id}), 202


@app.get("/api/runs/<run_id>/speaker-status")
def speaker_status(run_id: str):
    run_dir = _safe_run_dir(run_id)
    if run_dir is None or not (run_dir / "manifest.json").exists():
        return jsonify({"error": "run not found"}), 404
    speaker_transcript = load_speaker_transcript(run_dir)
    results = _manifest_results(run_dir)
    item_statuses = []
    for item in results:
        source_label = str(item.get("source_label") or "")
        item_dir = _resolve_item_dir(run_dir, item)
        item_payload = _speaker_item_payload(item_dir, source_label) if item_dir else {"payload": {}}
        item_statuses.append(
            {
                "source_label": source_label,
                "transcript_status": item.get("status") or "missing",
                "speaker_status": "ready" if item_payload.get("payload") else "not_started",
                "paths": item_payload.get("paths") or {},
            }
        )
    active_jobs = []
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.get("run_id") != run_id:
                continue
            if job.get("status") not in {"ready", "partial", "failed", "completed"}:
                active_jobs.append(dict(job))
    status = "not_started"
    if speaker_transcript:
        status = "partial" if speaker_transcript.get("status") == "partial" else "ready"
    elif active_jobs:
        status = str(active_jobs[-1].get("status") or "queued")
    return jsonify(
        {
            "run_id": run_id,
            "status": status,
            "available": bool(speaker_transcript),
            "backend": speaker_transcript.get("backend", "") if speaker_transcript else "",
            "backends": speaker_transcript.get("backends", []) if speaker_transcript else [],
            "fallback_used": bool(speaker_transcript.get("fallback_used")) if speaker_transcript else False,
            "coverage": speaker_transcript.get("coverage", {}) if speaker_transcript else {},
            "speakers": speaker_transcript.get("speakers", []) if speaker_transcript else [],
            "items": item_statuses,
            "job": active_jobs[-1] if active_jobs else None,
            "transcript": speaker_transcript,
        }
    )


def _coerce_speaker_mapping(payload: object) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    candidate = payload.get("mapping", payload.get("speaker_map", payload))
    if not isinstance(candidate, dict):
        return None
    mapping: dict[str, str] = {}
    for raw_key, raw_value in candidate.items():
        key = str(raw_key).strip()
        value = str(raw_value or "").strip()
        if not key or len(key) > 120 or len(value) > 120:
            return None
        mapping[key] = value
    return mapping


def _apply_manual_speaker_map(payload: dict, mapping: dict[str, str]) -> dict:
    updated = dict(payload)
    turns = []
    for raw_turn in payload.get("speaker_turns") or []:
        if not isinstance(raw_turn, dict):
            continue
        turn = dict(raw_turn)
        identity = str(turn.get("speaker_id") or turn.get("raw_speaker_label") or turn.get("speaker") or "Speaker 1")
        turn["speaker_id"] = identity
        turn["speaker"] = mapping.get(identity, str(turn.get("speaker") or identity)) or identity
        turns.append(turn)
    updated["speaker_turns"] = turns
    speakers = []
    for raw_speaker in payload.get("speakers") or []:
        if not isinstance(raw_speaker, dict):
            continue
        speaker = dict(raw_speaker)
        identity = str(speaker.get("speaker_id") or speaker.get("raw_labels", [""])[0] or speaker.get("display_name") or "")
        display_name = mapping.get(identity, str(speaker.get("display_name") or identity)).strip() or identity
        speaker["speaker_id"] = identity
        speaker["display_name"] = display_name
        speaker["identity_status"] = "resolved" if display_name and display_name != identity else "unknown"
        evidence = list(speaker.get("identity_evidence") or [])
        if identity in mapping:
            evidence.append({"type": "manual", "source": "speaker-map-api", "confidence": 1.0})
        speaker["identity_evidence"] = evidence
        speaker["confidence"] = 1.0 if identity in mapping and display_name else float(speaker.get("confidence") or 0.0)
        speakers.append(speaker)
    updated["speakers"] = speakers
    updated["speaker_map"] = {**dict(payload.get("speaker_map") or {}), **mapping}
    updated["mapping_updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    updated["identity_status"] = "resolved" if any(item.get("identity_status") == "resolved" for item in speakers) else "unknown"
    nested_items = []
    for raw_item in payload.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        item_view = dict(raw_item)
        nested_payload = item_view.get("payload")
        if isinstance(nested_payload, dict):
            item_view["payload"] = _apply_manual_speaker_map(nested_payload, mapping)
        nested_items.append(item_view)
    if nested_items:
        updated["items"] = nested_items
    return updated


@app.post("/api/runs/<run_id>/speaker-map")
def save_speaker_map(run_id: str):
    run_dir = _safe_run_dir(run_id)
    if run_dir is None or not (run_dir / "manifest.json").exists():
        return jsonify({"error": "run not found"}), 404
    body = request.get_json(silent=True)
    mapping = _coerce_speaker_mapping(body)
    if mapping is None:
        return jsonify({"error": "mapping must be an object of speaker IDs to display names"}), 400
    speaker_transcript = load_speaker_transcript(run_dir)
    if not speaker_transcript:
        return jsonify({"error": "speaker transcript not found; generate it first"}), 404
    updated = _apply_manual_speaker_map(speaker_transcript, mapping)
    # Keep the raw diarization files immutable.  This separate identity map is
    # the only layer changed by the manual correction endpoint.
    identity_path = run_dir / "speaker.identity-map.json"
    identity_path.write_text(
        json.dumps({"schema_version": 1, "mapping": mapping, "updated_at": updated["mapping_updated_at"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    root_json = run_dir / "transcript.speakers.json"
    root_json.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_speaker_text_files(run_dir, updated.get("speaker_turns") or [])
    (run_dir / "speaker_map.json").write_text(json.dumps(updated.get("speaker_map") or mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in updated.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_payload = item.get("payload") or {}
        paths = item.get("paths") or {}
        json_value = str(paths.get("json") or "").strip()
        if not json_value:
            continue
        item_json = Path(json_value)
        try:
            item_json.write_text(json.dumps(item_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_speaker_text_files(item_json.parent, item_payload.get("speaker_turns") or [])
            (item_json.parent / "speaker_map.json").write_text(json.dumps(item_payload.get("speaker_map") or mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            continue
    return jsonify({"run_id": run_id, "status": updated.get("status", "ready"), "mapping": mapping, "transcript": updated})


@app.post("/api/runs/<run_id>/enhance")
@app.post("/api/runs/<run_id>/re-enhance")
def enhance_run(run_id: str):
    run_dir = _safe_run_dir(run_id)
    if run_dir is None or not (run_dir / "manifest.json").exists():
        return jsonify({"error": "run not found"}), 404
    payload = request.get_json(silent=True) if request.is_json else request.form
    profile = str((payload or {}).get("profile") or "fast")
    if profile not in {"balanced", "quality", "fast"}:
        return jsonify({"error": "Unsupported postprocess profile."}), 400
    glossary_text = str((payload or {}).get("glossary") or "")
    style = str((payload or {}).get("style") or "")
    job_id = uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = _new_job(job_id, ENHANCEMENT_JOB, run_id=run_id, message="智能整理任务已创建", profile=profile)
    Thread(target=run_enhancement_job, args=(job_id, run_dir, profile, glossary_text, style), daemon=True).start()
    return jsonify({"job_id": job_id, "run_id": run_id}), 202


def _read_speaker_profiles() -> list[dict]:
    if not SPEAKER_PROFILES_PATH.exists():
        return []
    try:
        payload = json.loads(SPEAKER_PROFILES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("profiles") if isinstance(payload, dict) else []
    return [item for item in rows if isinstance(item, dict)]


def _public_speaker_profile(profile: dict) -> dict:
    # Embeddings are deliberately never persisted by this API and are also
    # stripped if an older local file happens to contain one.
    return {
        "person_id": str(profile.get("person_id") or ""),
        "display_name": str(profile.get("display_name") or ""),
        "sample_paths": [str(path) for path in profile.get("sample_paths") or [] if isinstance(path, str)],
        "embedding_model": str(profile.get("embedding_model") or ""),
        "embedding_digest": str(profile.get("embedding_digest") or ""),
        "threshold": float(profile["threshold"]) if profile.get("threshold") is not None else 0.78,
        "consent": bool(profile.get("consent")),
        "created_at": str(profile.get("created_at") or ""),
    }


def _write_speaker_profiles(profiles: list[dict]) -> None:
    SPEAKER_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEAKER_PROFILES_PATH.write_text(
        json.dumps({"schema_version": 1, "profiles": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        os.chmod(SPEAKER_PROFILES_PATH, 0o600)
    except OSError:
        pass


@app.get("/api/speaker-profiles")
def list_speaker_profiles():
    with JOBS_LOCK:
        rows = [_public_speaker_profile(item) for item in _read_speaker_profiles()]
    return jsonify({"profiles": rows, "storage": "local-only"})


@app.post("/api/speaker-profiles")
def create_speaker_profile():
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload if isinstance(payload, dict) else {}
    display_name = str(payload.get("display_name") or payload.get("name") or "").strip()
    if not display_name or len(display_name) > 120:
        return jsonify({"error": "display_name is required and must be at most 120 characters"}), 400
    raw_id = str(payload.get("person_id") or "").strip()
    person_id = secure_filename(raw_id).replace("-", "_") if raw_id else ""
    if not person_id:
        person_id = "person_" + uuid4().hex[:12]
    sample_paths = payload.get("sample_paths") or payload.get("samples") or []
    if isinstance(sample_paths, str):
        sample_paths = [sample_paths]
    if not isinstance(sample_paths, list) or len(sample_paths) > 32:
        return jsonify({"error": "sample_paths must be a list with at most 32 local paths"}), 400
    normalized_paths: list[str] = []
    for raw_path in sample_paths:
        path_value = str(raw_path or "").strip()
        if not path_value or path_value.startswith(("http://", "https://")):
            return jsonify({"error": "speaker samples must be local file paths"}), 400
        normalized_paths.append(str(Path(path_value).expanduser().resolve()))
    try:
        threshold = float(payload.get("threshold", 0.78))
    except (TypeError, ValueError):
        return jsonify({"error": "threshold must be a number between 0 and 1"}), 400
    if not 0 <= threshold <= 1:
        return jsonify({"error": "threshold must be a number between 0 and 1"}), 400
    profile = {
        "person_id": person_id,
        "display_name": display_name,
        "sample_paths": normalized_paths,
        "embedding_model": str(payload.get("embedding_model") or ""),
        "embedding_digest": str(payload.get("embedding_digest") or ""),
        "threshold": threshold,
        "consent": bool(payload.get("consent", False)),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    with JOBS_LOCK:
        profiles = _read_speaker_profiles()
        if any(str(item.get("person_id") or "") == person_id for item in profiles):
            return jsonify({"error": "person_id already exists"}), 409
        profiles.append(profile)
        _write_speaker_profiles(profiles)
    return jsonify({"profile": _public_speaker_profile(profile), "storage": "local-only"}), 201


@app.delete("/api/speaker-profiles/<person_id>")
def delete_speaker_profile(person_id: str):
    target = secure_filename(str(person_id)).replace("-", "_")
    with JOBS_LOCK:
        profiles = _read_speaker_profiles()
        kept = [item for item in profiles if str(item.get("person_id") or "") != target]
        if len(kept) == len(profiles):
            return jsonify({"error": "speaker profile not found"}), 404
        _write_speaker_profiles(kept)
    # Enrollment audio is user-owned and is intentionally not deleted by this
    # metadata operation; callers can remove it explicitly if desired.
    return jsonify({"deleted": target, "storage": "local-only"})


@app.get("/artifacts/<run_id>/<path:artifact_path>")
def artifact(run_id: str, artifact_path: str):
    run_dir = _safe_run_dir(run_id)
    if run_dir is None:
        return jsonify({"error": "invalid run"}), 404
    return send_from_directory(
        run_dir,
        artifact_path,
        as_attachment=request.args.get("download") == "1",
    )


if __name__ == "__main__":
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=8765, debug=False)
