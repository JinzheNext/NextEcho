"""Resource-bounded, environment-isolated CPU ASR adapter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .asr import TranscriptResult
from .config import NextEchoConfig
from .contracts import PipelineError
from .io_utils import file_hash, write_json
from .model_profiles import PROFILES, validate_model_files
from .resources import model_policy, run_owned
from .job_store import current_cancelled


def transcribe_local_audio(audio: Path, item_dir: Path, *, engine: str, model: str,
                           language: str, config: NextEchoConfig) -> TranscriptResult:
    # Global selection/fallback belongs to the shared transcription service.
    return _transcribe_local_audio(audio, item_dir, engine=engine, model=model, language=language, config=config)


def _transcribe_local_audio(audio: Path, item_dir: Path, *, engine: str, model: str,
                            language: str, config: NextEchoConfig) -> TranscriptResult:
    profile_id = "fun-asr-nano-pt" if engine == "fun-asr-nano" else "sensevoice-int8"
    profile = PROFILES[profile_id]
    settings = config.get_path("models", profile_id, default={})
    root = Path(model).expanduser().resolve()
    status = validate_model_files(root, profile.required_files)
    if not status["complete"]:
        raise PipelineError("MODEL_MISSING", "模型文件未准备完整：" + ", ".join(status["missing_files"]), blocked=True)
    weights = "model.pt" if engine == "fun-asr-nano" else "model.int8.onnx"
    digest = file_hash(root / weights)
    if settings.get("calibration_files") and any(file_hash(root / name) != settings["calibration_files"].get(name) for name in profile.required_files):
        raise PipelineError("RESOURCE_BLOCKED", "模型文件已改变，请重新prepare校准", blocked=True)
    if settings.get("model_sha256") and settings["model_sha256"] != digest:
        raise PipelineError("RESOURCE_BLOCKED", "权重已改变，旧资源校准不可复用，请重新prepare校准", blocked=True)
    if settings.get("runtime_versions"):
        from .capabilities import NATIVE_REQUIREMENTS, probe_environment
        actual = probe_environment(settings.get("worker_python") or sys.executable, NATIVE_REQUIREMENTS[profile_id])
        for package in NATIVE_REQUIREMENTS[profile_id]:
            if actual["versions"].get(package) != settings["runtime_versions"].get(package):
                raise PipelineError("RESOURCE_BLOCKED", "模型运行时已改变，请重新受控校准", blocked=True)
    vad = Path(config.get_path("asr", "vad_model", default="")).expanduser()
    if not vad.is_file():
        raise PipelineError("MODEL_MISSING", "请在asr.vad_model绑定本地Silero VAD ONNX文件", blocked=True)
    if settings.get("vad_sha256") and settings["vad_sha256"] != file_hash(vad):
        raise PipelineError("RESOURCE_BLOCKED", "VAD已改变，请重新prepare校准", blocked=True)
    policy = model_policy(settings.get("estimated_bytes", 0), settings.get("max_rss_bytes", 0),
                          settings.get("timeout_seconds", 0), config)
    request = item_dir / "asr.worker.request.json"
    output = item_dir / "asr.worker.result.json"
    # Never consume a stale worker output if a retry fails before writing.
    output.unlink(missing_ok=True)
    context = {}
    if config.get_path("asr", "contextual_hotwords", default=False):
        try:
            context = json.loads((item_dir / "source.context.json").read_text())
        except (OSError, ValueError):
            pass
    from .polish_context import asr_hotwords
    explicit = config.get_path("asr", "hotwords", default=[])
    script = config.get_path("asr", "output_script", default="original")
    if not isinstance(explicit, list) or not all(isinstance(v, str) for v in explicit) or script not in {"original", "simplified"}:
        raise PipelineError("INVALID_INPUT", "asr.hotwords必须是字符串列表，output_script必须为original或simplified")
    hotwords = asr_hotwords(context, explicit) if engine == "fun-asr-nano" else []
    write_json(request, {"schema_version": 1, "audio": str(audio.resolve()), "model": str(root),
                         "engine": engine, "language": language, "vad_model": str(vad.resolve()),
                         "output_script": script,
                         "hotwords": hotwords, "progress": str((item_dir / "asr.progress.json").resolve()),
                         "output": str(output.resolve())})
    python = str(Path(settings.get("worker_python") or sys.executable).expanduser().absolute())
    # Running the script by absolute path also works when the model venv does
    # not have the application installed; it imports its sibling package.
    worker = Path(__file__).with_name("local_asr_worker.py")
    worker_digest = file_hash(worker)
    segmentation_digest = file_hash(Path(__file__).with_name("audio_segments.py"))
    from .capabilities import NATIVE_REQUIREMENTS, probe_environment
    runtime_identity = probe_environment(python, NATIVE_REQUIREMENTS[profile_id])
    try:
        resources = run_owned([python, str(worker), str(request)], log_path=item_dir / "logs/asr.log", policy=policy,
                              cancel=current_cancelled, interval=float(settings.get("resource_sample_seconds", 2)))
    except PipelineError as exc:
        log = item_dir / "logs/asr.log"
        if exc.code == "WORKER_FAILED" and log.is_file() and "ModuleNotFoundError" in log.read_text(errors="replace")[-8000:]:
            raise PipelineError("DEPENDENCY_MISSING", "所选worker环境缺少ASR依赖，请安装对应extras", blocked=True) from exc
        raise
    if not output.is_file():
        raise PipelineError("PROTOCOL_ERROR", "ASR worker未生成结果文件")
    if output.stat().st_size > 32 * 1024**2:
        raise PipelineError("PROTOCOL_ERROR", "ASR worker结果超过大小上限")
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
        validate_worker_result(payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise PipelineError("PROTOCOL_ERROR", "ASR worker结果不符合协议") from exc
    if payload.get("peak_worker_rss_bytes"):
        resources["peak_worker_rss_bytes"] = max(resources.get("peak_worker_rss_bytes") or 0, payload["peak_worker_rss_bytes"])
        if payload["peak_worker_rss_bytes"] > policy.max_rss_bytes:
            raise PipelineError("RESOURCE_LIMIT", "worker实测峰值超过资源档案，需重新受控校准", stage="asr")
    provenance = {"source_repo": profile.model, "model_path": str(root), "sha256": digest,
                  "quantization": profile.quantization, "runtime_versions": payload.get("runtime_versions", {}),
                  "vad_sha256": file_hash(vad), "release_status": profile.release_status}
    provenance["segmentation"] = {"windows": payload.get("windows", []), "review": payload.get("review", [])}
    provenance["worker_sha256"] = worker_digest
    provenance["segmentation_sha256"] = segmentation_digest
    provenance["environment_versions"] = runtime_identity["versions"]
    provenance["recognition_prompt"] = payload.get("recognition_prompt", {})
    resources["worker_timings"] = payload.get("timings", {})
    return TranscriptResult(text=payload["text"], language=language, words=[], backend=engine,
                            model=str(root), version=payload.get("version", ""), resources=resources,
                            provenance=provenance, segments=payload["segments"],
                            timing_precision="vad-segment", coverage=payload["coverage"])


def validate_worker_result(payload):
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported result version")
    if not isinstance(payload.get("text"), str) or not isinstance(payload.get("segments"), list):
        raise ValueError("Missing text/segments")
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Missing coverage")
    import math
    seconds = coverage.get("covered_seconds")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid duration")
    previous = 0
    for row in payload["segments"]:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("Invalid segment")
        start, end = row.get("start_ms"), row.get("end_ms")
        if type(start) is not int or type(end) is not int or not previous <= start < end <= math.ceil(seconds * 1000):
            raise ValueError("Invalid measured segment timing")
        previous = end
    if bool(payload["text"].strip()) != any(r["text"].strip() for r in payload["segments"]):
        raise ValueError("Text/segments disagree")
