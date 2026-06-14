from __future__ import annotations

import json
import shutil
from threading import Lock, Thread
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4
import time

from workbench.speaker_transcript import build_speaker_transcript
from workbench.sources import resolve_sources
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from workbench.transcription import DEFAULT_MODEL_NAME, transcribe_media_sources

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "outputs" / "transcriptions"
ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".aac", ".mov", ".webm"}
TERMINAL_ITEM_STATUSES = {"ready", "empty", "failed", "cached"}
STALE_JOB_SECONDS = 180
QUALITY_TO_MODEL = {
    "accurate": DEFAULT_MODEL_NAME,
    "fast": "base",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
JOBS: dict[str, dict] = {}
JOBS_LOCK = Lock()


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
        "run_config_path": str(run_config_path),
        "items_dir": str(run_dir / "items"),
        "results": results,
        "error_count": len(failed_results),
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
    run_id = job.get("run_id") or ""
    run_dir = RUNS_ROOT / run_id
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

    last_updated = run_last_updated_epoch(run_dir) if run_dir.exists() else 0.0
    is_stale = last_updated and (time.time() - last_updated) >= STALE_JOB_SECONDS
    if is_stale:
        item_metadata = load_item_metadata(run_dir)
        detail = "运行目录长时间没有更新。"
        if item_metadata:
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
        JOBS[job_id].update(fields)


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
) -> None:
    started_monotonic = time.monotonic()

    def on_progress(event: dict) -> None:
        update_job(
            job_id,
            elapsed_seconds=max(0, round(time.monotonic() - started_monotonic)),
            **event,
        )

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
            progress_callback=on_progress,
        )
        result = serialize_result(payload)
        failed_count = int(result["manifest"].get("error_count") or 0)
        total_count = len(result["manifest"].get("results", []))
        message = "转写完成"
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
            "run_id": run_id,
            "status": "running",
            "stage": "queued",
            "message": "任务已创建",
            "progress": 0,
            "elapsed_seconds": 0,
            "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "result": None,
            "error": "",
        }
    Thread(
        target=run_transcription_job,
        args=(job_id, sources, run_dir, model, language, max_seconds, glossary_text, ai_polish, ai_style),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "run_id": run_id}), 202


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
        runs.append(
            {
                "run_id": manifest_path.parent.name,
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
