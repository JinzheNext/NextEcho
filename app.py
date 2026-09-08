from __future__ import annotations

import json
import shutil
import subprocess
import sys
from threading import Event, Lock, Thread
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4
import time

from workbench.speaker_transcript import build_speaker_transcript
from workbench.postprocess import FAST_MODEL, QUALITY_MODEL, ollama_status, process_transcript_artifacts
from workbench.asr import (
    QWEN_ACCURATE_MODEL,
    QWEN_FAST_MODEL,
    QWEN_FORCED_ALIGNER_MODEL,
    is_apple_silicon,
    qwen_forced_aligner_cached,
    qwen_memory_safe,
    qwen_mlx_available,
    qwen_model_cached,
)
from workbench.sources import resolve_sources
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from workbench.transcription import DEFAULT_MODEL_NAME, transcribe_media_sources

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "outputs" / "transcriptions"
ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".aac", ".mov", ".webm"}
TERMINAL_ITEM_STATUSES = {"ready", "empty", "failed", "cached"}
STALE_JOB_SECONDS = 180
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
HEAVY_MODEL_LOCK = Lock()


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def parse_urls(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def serialize_result(payload: dict) -> dict:
    output_dir = Path(payload["output_dir"])
    manifest = json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))
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


def run_last_updated_epoch(run_dir: Path) -> float:
    newest = run_dir.stat().st_mtime if run_dir.exists() else 0.0
    for path in run_dir.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return newest


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
    completed_epoch = run_last_updated_epoch(run_dir)
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
    if job.get("status") != "running":
        return job
    job_type = job.get("job_type") or TRANSCRIPTION_JOB
    if job_type == MODEL_DOWNLOAD_JOB:
        return job
    run_id = job.get("run_id") or ""
    run_dir = RUNS_ROOT / run_id
    if job_type == TRANSCRIPTION_JOB:
        manifest_path = synthesize_manifest_if_complete(run_dir)
        if manifest_path and manifest_path.exists():
            result = serialize_run(run_dir)
            job.update(
                status="completed",
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
                status="completed",
                result=result,
                progress=100,
                stage="complete",
                message="访谈逐字稿已完成",
                error="",
            )
            return job
    last_updated = run_last_updated_epoch(run_dir) if run_dir.exists() else 0.0
    heartbeat = float(job.get("updated_at") or 0)
    is_stale = max(last_updated, heartbeat) and (time.time() - max(last_updated, heartbeat)) >= STALE_JOB_SECONDS
    if is_stale:
        item_metadata = load_item_metadata(run_dir)
        detail = "运行目录长时间没有更新。"
        if job_type == SPEAKER_TRANSCRIPT_JOB:
            detail = "访谈逐字稿任务长时间没有写出说话人结果，常见原因是依赖缺失或服务中断。"
        elif item_metadata:
            detail = "条目产物已部分写出，但任务未完成收尾。"
        elif any(run_dir.rglob("*")):
            detail = "仅生成了部分中间文件，常见原因是服务重启或本地进程中断。"
        job.update(
            status="failed",
            stage="failed",
            message=f"任务已中断：{detail}",
            error=f"任务已中断：{detail}",
        )
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
    if not json_path.exists():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["paths"] = {
        "json": str(json_path),
        "txt": str(txt_path),
        "md": str(md_path),
    }
    return payload


def update_job(job_id: str, **fields: object) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(fields, updated_at=time.time())


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
    stop_heartbeat = Event()

    def on_progress(event: dict) -> None:
        update_job(
            job_id,
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
            **event,
        )

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(15):
            update_job(job_id, message="本地模型正在持续处理，请稍候")

    Thread(target=heartbeat_loop, daemon=True).start()

    acquired = HEAVY_MODEL_LOCK.acquire(blocking=False)
    if not acquired:
        update_job(job_id, stage="queued", message="正在等待本地模型空闲", progress=1)
        HEAVY_MODEL_LOCK.acquire()
    try:
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
        if speaker_mode != "off":
            update_job(job_id, stage="diarization", message="正在区分说话人", progress=96)
            try:
                speaker_payload = build_speaker_transcript(run_dir, progress_callback=on_progress)
            except Exception as exc:
                if speaker_mode == "required":
                    raise
                update_job(job_id, stage="diarization", message=f"说话人区分未完成，原始转录仍可使用：{exc}", progress=96)
        result = serialize_result(payload)
        if speaker_payload is not None:
            result["speaker_transcript"] = speaker_payload
        failed_count = int(result["manifest"].get("error_count") or 0)
        partial_count = int(result["manifest"].get("partial_count") or 0)
        total_count = len(result["manifest"].get("results", []))
        message = "转写完成"
        if partial_count:
            message = f"转写完成，{partial_count} 条智能加工部分完成，原始稿仍可使用"
        if failed_count and failed_count < total_count:
            message = f"转写完成，{failed_count} 条失败"
        elif failed_count and failed_count == total_count:
            message = "任务完成，但所有条目都失败了"
        update_job(
            job_id,
            status="completed",
            result=result,
            progress=100,
            stage="complete",
            message=message,
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
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
        stop_heartbeat.set()
        HEAVY_MODEL_LOCK.release()


def run_speaker_transcript_job(
    job_id: str,
    run_dir: Path,
) -> None:
    started_monotonic = time.monotonic()

    def on_progress(event: dict) -> None:
        update_job(
            job_id,
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
            **event,
        )

    try:
        build_speaker_transcript(run_dir, progress_callback=on_progress)
        update_job(
            job_id,
            status="completed",
            result=serialize_run(run_dir),
            progress=100,
            stage="complete",
            message="访谈逐字稿已完成",
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
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
    stop_heartbeat = Event()

    def on_progress(event: dict) -> None:
        update_job(job_id, elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)), **event)

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(15):
            update_job(job_id, message="本地模型正在持续整理，请稍候")

    Thread(target=heartbeat_loop, daemon=True).start()

    acquired = HEAVY_MODEL_LOCK.acquire(blocking=False)
    if not acquired:
        update_job(job_id, stage="queued", message="正在等待本地模型空闲", progress=1)
        HEAVY_MODEL_LOCK.acquire()
    try:
        item_dirs = sorted((run_dir / "items").glob("*"))
        processed = 0
        for item_dir in item_dirs:
            if not (item_dir / "transcript.json").exists():
                continue
            audit = process_transcript_artifacts(
                item_dir,
                profile=profile,
                user_glossary_text=glossary_text,
                style=style,
                progress_callback=on_progress,
            )
            update_enhancement_metadata(run_dir, item_dir, audit)
            processed += 1
        if not processed:
            raise FileNotFoundError("这个历史任务没有可供整理的 transcript.json")
        update_job(
            job_id,
            status="completed",
            result=serialize_run(run_dir),
            progress=100,
            stage="complete",
            message="智能整理与知识摘要已完成",
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
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
        stop_heartbeat.set()
        HEAVY_MODEL_LOCK.release()


def run_model_download_job(job_id: str, kind: str) -> None:
    model_map = {
        "asr-accurate": ("asr", QWEN_ACCURATE_MODEL),
        "asr-fast": ("asr", QWEN_FAST_MODEL),
        "llm-fast": ("ollama", FAST_MODEL),
        "llm-quality": ("ollama", QUALITY_MODEL),
    }
    target = model_map.get(kind)
    if target is None:
        update_job(job_id, status="failed", stage="failed", message="未知模型类型", error="未知模型类型")
        return
    provider, model = target
    acquired = HEAVY_MODEL_LOCK.acquire(blocking=False)
    if not acquired:
        update_job(job_id, stage="queued", message="正在等待本地模型空闲", progress=1)
        HEAVY_MODEL_LOCK.acquire()
    try:
        update_job(job_id, stage="model", message=f"正在下载 {model}", progress=5)
        if provider == "asr":
            if not qwen_mlx_available():
                raise RuntimeError("Qwen ASR 仅可在安装 mlx-qwen3-asr 的 Apple Silicon Mac 上下载")
            repos = [model, QWEN_FORCED_ALIGNER_MODEL]
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
        update_job(job_id, status="completed", stage="complete", message=f"{model} 已准备好", progress=100)
    except Exception as exc:
        update_job(job_id, status="failed", stage="failed", message=str(exc), error=str(exc))
    finally:
        HEAVY_MODEL_LOCK.release()


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
        JOBS[job_id] = {
            "job_id": job_id,
            "job_type": TRANSCRIPTION_JOB,
            "run_id": run_id,
            "status": "running",
            "stage": "queued",
            "message": "任务已创建",
            "progress": 0,
            "elapsed_seconds": 0,
            "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "updated_at": time.time(),
            "asr_engine": asr_engine,
            "asr_model": asr_model,
            "postprocess_profile": postprocess_profile,
            "speaker_mode": speaker_mode,
            "result": None,
            "error": "",
        }
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
    return jsonify(
        {
            "apple_silicon": is_apple_silicon(),
            "qwen_mlx_available": qwen_mlx_available(),
            "qwen_memory_safe": qwen_memory_safe(),
            "asr_models": {"accurate": QWEN_ACCURATE_MODEL, "fast": QWEN_FAST_MODEL},
            "qwen_asr_model_cache": {
                "accurate": qwen_model_cached(QWEN_ACCURATE_MODEL),
                "fast": qwen_model_cached(QWEN_FAST_MODEL),
            },
            "qwen_forced_aligner": {
                "model": QWEN_FORCED_ALIGNER_MODEL,
                "cached": qwen_forced_aligner_cached(),
            },
            "ollama": ollama_status(),
            "llm_models": {"fast": FAST_MODEL, "quality": QUALITY_MODEL},
        }
    )


@app.post("/api/models/download")
def download_model():
    payload = request.get_json(silent=True) if request.is_json else request.form
    kind = str((payload or {}).get("kind") or "")
    if kind not in {"asr-accurate", "asr-fast", "llm-fast", "llm-quality"}:
        return jsonify({"error": "Unsupported model type."}), 400
    job_id = uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "job_type": MODEL_DOWNLOAD_JOB,
            "run_id": "",
            "status": "running",
            "stage": "queued",
            "message": "模型下载任务已创建",
            "progress": 0,
            "elapsed_seconds": 0,
            "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "updated_at": time.time(),
            "result": None,
            "error": "",
        }
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
    run_dir = (RUNS_ROOT / run_id).resolve()
    manifest_path = synthesize_manifest_if_complete(run_dir) or (run_dir / "manifest.json")
    if RUNS_ROOT.resolve() not in run_dir.parents or not manifest_path.exists():
        return jsonify({"error": "run not found"}), 404
    return jsonify(serialize_run(run_dir))


@app.post("/api/runs/<run_id>/speaker-transcript")
def create_speaker_transcript(run_id: str):
    run_dir = (RUNS_ROOT / run_id).resolve()
    if RUNS_ROOT.resolve() not in run_dir.parents or not (run_dir / "manifest.json").exists():
        return jsonify({"error": "run not found"}), 404
    job_id = uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "job_type": SPEAKER_TRANSCRIPT_JOB,
            "run_id": run_id,
            "status": "running",
            "stage": "queued",
            "message": "访谈逐字稿任务已创建",
            "progress": 0,
            "elapsed_seconds": 0,
            "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "result": None,
            "error": "",
        }
    Thread(
        target=run_speaker_transcript_job,
        args=(job_id, run_dir),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "run_id": run_id}), 202


@app.post("/api/runs/<run_id>/enhance")
def enhance_run(run_id: str):
    run_dir = (RUNS_ROOT / run_id).resolve()
    if RUNS_ROOT.resolve() not in run_dir.parents or not (run_dir / "manifest.json").exists():
        return jsonify({"error": "run not found"}), 404
    payload = request.get_json(silent=True) if request.is_json else request.form
    profile = str((payload or {}).get("profile") or "fast")
    if profile not in {"balanced", "quality", "fast"}:
        return jsonify({"error": "Unsupported postprocess profile."}), 400
    glossary_text = str((payload or {}).get("glossary") or "")
    style = str((payload or {}).get("style") or "")
    job_id = uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "job_type": ENHANCEMENT_JOB,
            "run_id": run_id,
            "status": "running",
            "stage": "queued",
            "message": "智能整理任务已创建",
            "progress": 0,
            "elapsed_seconds": 0,
            "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "updated_at": time.time(),
            "result": None,
            "error": "",
        }
    Thread(target=run_enhancement_job, args=(job_id, run_dir, profile, glossary_text, style), daemon=True).start()
    return jsonify({"job_id": job_id, "run_id": run_id}), 202


@app.get("/artifacts/<run_id>/<path:artifact_path>")
def artifact(run_id: str, artifact_path: str):
    run_dir = (RUNS_ROOT / run_id).resolve()
    if RUNS_ROOT.resolve() not in run_dir.parents:
        return jsonify({"error": "invalid run"}), 404
    return send_from_directory(
        run_dir,
        artifact_path,
        as_attachment=request.args.get("download") == "1",
    )


if __name__ == "__main__":
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=8765, debug=False)
