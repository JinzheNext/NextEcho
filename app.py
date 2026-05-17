from __future__ import annotations

import json
import shutil
from threading import Lock, Thread
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from workbench.transcription import DEFAULT_MODEL_NAME, transcribe_media_sources

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "outputs" / "transcriptions"
ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".aac", ".mov", ".webm"}

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
        "manifest": manifest,
    }


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
) -> None:
    def on_progress(event: dict) -> None:
        update_job(job_id, **event)

    try:
        payload = transcribe_media_sources(
            sources,
            output_dir=run_dir,
            model_name=model,
            language=language,
            max_seconds=max_seconds,
            progress_callback=on_progress,
        )
        update_job(job_id, status="completed", result=serialize_result(payload), progress=100, stage="complete", message="转写完成")
    except Exception as exc:  # local operator tool: surface exact failure
        update_job(job_id, status="failed", stage="failed", message=str(exc), error=str(exc))


@app.get("/")
def index():
    return render_template("index.html", default_model=DEFAULT_MODEL_NAME)


@app.post("/api/transcribe")
def transcribe():
    urls = parse_urls(request.form.get("urls", ""))
    model = request.form.get("model", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    language = request.form.get("language", "zh").strip() or "zh"
    try:
        max_seconds = int(request.form.get("max_seconds", "0"))
    except ValueError:
        return jsonify({"error": "max_seconds must be an integer"}), 400

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
            "result": None,
            "error": "",
        }
    Thread(
        target=run_transcription_job,
        args=(job_id, sources, run_dir, model, language, max_seconds),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "run_id": run_id}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(job)


@app.get("/api/runs")
def list_runs():
    runs = []
    for manifest_path in sorted(RUNS_ROOT.glob("*/manifest.json"), reverse=True):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "run_id": manifest_path.parent.name,
                "generated_at": manifest.get("generated_at"),
                "item_count": len(manifest.get("results", [])),
                "model_name": manifest.get("model_name"),
            }
        )
    return jsonify({"runs": runs[:20]})


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
    app.run(host="127.0.0.1", port=8765, debug=True)
