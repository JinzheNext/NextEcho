"""Read-only capability diagnostics; importing it never imports a model runtime."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .binaries import find_binary
from .media_commands import run_media_command
from .contracts import PipelineError, PipelineRequest, plan_request
from .model_profiles import PROFILES, complete_hf_snapshot, native_asr_status, validate_model_files

NATIVE_REQUIREMENTS = {
    "fun-asr-nano-pt": {"funasr": "1.4.16", "torch": "2.14.1", "kaldi-native-fbank": "1.22.3", "sherpa-onnx": "1.13.8", "soundfile": None,
                        "transformers": None, "tokenizers": None, "numpy": None, "huggingface-hub": None},
    "sensevoice-int8": {"sherpa-onnx": "1.13.8", "soundfile": None, "numpy": None},
}


def resolve_worker_python(value):
    value = str(value or sys.executable)
    if re.fullmatch(r"\d+\.\d+", value):
        value = sys.executable if value == f"{sys.version_info.major}.{sys.version_info.minor}" else shutil.which("python" + value)
    return str(Path(value).expanduser().absolute()) if value else ""


def acoustic_speaker_status(config):
    from .diarization import _sherpa_model_paths
    settings = config.get_path("diarization", default={})
    native = settings.get("backend") == "nemotron-metal-candidate"
    paths = [] if native else _sherpa_model_paths(settings.get("sherpa") or {})
    root = Path(str(settings.get("ctc_model_path") or config.get_path("models", "sensevoice-int8", "path", default=""))).expanduser()
    python = resolve_worker_python(settings.get("worker_python"))
    environment = probe_environment(python or "/nonexistent/nextecho-python", NATIVE_REQUIREMENTS["sensevoice-int8"])
    weights = len(paths) == 2 and all(p.is_file() and p.stat().st_size > 0 for p in paths)
    native_info = {}
    if native:
        from .nemotron_speaker import binding, validate_assets
        try:
            ledger, runtime, profile = binding(settings)
            locked, _ = validate_assets(ledger, runtime)
            weights = True
            native_info = {"model": "nvidia/Nemotron-3-Diarization", "model_revision": locked["model"]["revision"],
                           "model_digest": locked["model"]["sha256"], "attribution_profile": profile}
        except PipelineError as exc:
            weights = False
            native_info = {"error_code": exc.code, "error": str(exc)}
    ctc = validate_model_files(root, PROFILES["sensevoice-int8"].required_files)
    ready = weights and ctc["complete"] and environment["available"]
    return {"status": "ready" if ready else "missing", "release_status": "candidate",
            "backend": "nemotron-metal-candidate" if native else "sherpa-global-native-ctc", "environment": environment,
            **native_info,
            "speaker_models_available": weights, "ctc_files": ctc, "word_timestamps_generated": False,
            "next_action": "已具备运行条件；分人和身份仍需回听审核" if ready else
            ("绑定diarization.nemotron的asset_ledger/runtime_root，并准备SenseVoice及独立CTC worker" if native else
             "运行prepare --speaker-model-root绑定segmentation.onnx/embedding.onnx，并准备SenseVoice及同一独立worker环境")}


def probe_environment(python, requirements):
    code = '''import importlib.metadata,json,sys
r={}
for name in json.loads(sys.argv[1]):
 try:r[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:r[name]=None
print(json.dumps(r))'''
    python = str(Path(python).expanduser().absolute())
    try:
        checked = run_media_command([python, "-c", code, json.dumps(list(requirements))], timeout=5, stage="environment-probe", check=False)
        versions = json.loads(checked.stdout) if checked.returncode == 0 else {}
        if not isinstance(versions, dict):
            versions = {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        versions = {}
    missing = [name for name, version in requirements.items() if not versions.get(name) or version and versions[name] != version]
    return {"python": python, "versions": versions, "missing_or_mismatched": missing, "available": not missing}


def inspect_capabilities(config, *, profile="balanced", capability="all"):
    plan = plan_request(PipelineRequest(("diagnostic",), profile=profile), config)
    checks = {}
    if capability in {"all", "asr"}:
        key = plan.asr_profile
        settings = config.get_path("models", key, default={})
        files = validate_model_files(Path(settings.get("path", ".")), PROFILES[key].required_files)
        environment = probe_environment(settings.get("worker_python") or sys.executable, NATIVE_REQUIREMENTS[key])
        vad = Path(config.get_path("asr", "vad_model", default="")).expanduser()
        ffmpeg = find_binary("ffmpeg")
        ready = files["complete"] and environment["available"] and vad.is_file() and bool(ffmpeg)
        release = next(row for row in native_asr_status(config) if row["id"] == key)
        checks["asr"] = {"status": "ready" if ready else "missing", "profile": key, "files": files,
            "environment": environment, "vad_available": vad.is_file(), "ffmpeg_available": bool(ffmpeg),
            "resource_calibrated": all(settings.get(k, 0) > 0 for k in ("estimated_bytes", "max_rss_bytes", "timeout_seconds")),
            "release_status": settings.get("release_status", "candidate"),
            "calibration_identity_valid": release["calibration_identity_valid"],
            "next_action": "资源与质量验收通过后才提升默认" if ready else f"为所选worker安装asr-{'funasr' if key == 'fun-asr-nano-pt' else 'sensevoice'} extras并运行prepare绑定模型及VAD"}
    if capability in {"all", "align"}:
        aligner = config.get_path("asr", "forced_aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
        checks["align"] = {"status": "unsupported", "error_code": "NEEDS_ALIGNER", "model": aligner,
            "model_cached": complete_hf_snapshot(aligner) is not None,
            "reason": "外部文本精确对齐尚未通过50个核对点验收；segment模式可用"}
    if capability in {"all", "polish"}:
        from .llm_providers import request_json
        model = PROFILES[plan.polish_profile]
        endpoint = config.get_path("providers", "splash" if model.engine == "local-chat-compatible" else "ollama", "endpoint",
                default="http://127.0.0.1:8081" if model.engine == "local-chat-compatible" else "http://127.0.0.1:11434")
        try:
            payload = request_json(endpoint, "/v1/models" if model.engine == "local-chat-compatible" else "/api/tags", timeout=3)
            names = [r.get("id") for r in payload.get("data", [])] if model.engine == "local-chat-compatible" else [r.get("name") for r in payload.get("models", [])]
            checks["polish"] = {"status": "ready" if model.model in names else "missing", "model": model.model, "provider": model.engine,
                                "release_status": config.get_path("models", plan.polish_profile, "release_status", default=model.release_status), "next_action": "保持所选本机模型服务就绪；此检查不生成内容"}
        except Exception as exc:
            checks["polish"] = {"status": "missing", "model": model.model, "error_code": getattr(exc, "code", "PROVIDER_UNAVAILABLE"), "next_action": "启动所选本机模型服务；不会自动换模型"}
    if capability in {"all", "speaker"}:
        checks["speaker"] = acoustic_speaker_status(config)
    return {"schema_version": 1, "profile": profile, "capabilities": checks,
            "ok": all(row["status"] == "ready" for row in checks.values()), "production_ready": all(row["status"] == "ready" and row.get("release_status", "ready") == "ready" and row.get("calibration_identity_valid", True) for row in checks.values())}
