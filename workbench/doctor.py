from __future__ import annotations

import json
import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .speaker_transcript import HF_TOKEN_ENV_VARS, detect_speaker_backend_status
from .asr import (
    is_apple_silicon,
    qwen_forced_aligner_cached,
    qwen_memory_status,
    qwen_mlx_available,
    qwen_model_cached,
    qwen_accurate_model,
    qwen_fast_model,
    qwen_forced_aligner_model,
)
from .config import NextEchoConfig, load_config
from .model_profiles import native_asr_status, selection_decision
from .postprocess import FAST_MODEL, QUALITY_MODEL, ollama_status
from .binaries import find_binary
from .transcription import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_ROOT,
    KNOWN_MODEL_ROOTS,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    list_supported_whisper_models,
    recommend_whisper_model,
)

FAST_MODEL_NAME = "base"
# ffmpeg is needed to normalize audio for either backend.  Whisper is a
# fallback-only dependency and therefore must not make a Qwen-only setup fail.
REQUIRED_BINARIES = ["ffmpeg"]
OPTIONAL_BINARIES = ["whisper-cli", "yt-dlp", "curl"]


@dataclass
class BinaryCheck:
    name: str
    path: str | None
    required: bool
    ok: bool


@dataclass
class ModelCheck:
    name: str
    path: str | None
    ok: bool


@dataclass
class DoctorReport:
    ok: bool
    os: str
    machine: str
    python: str
    memory_gb: float | None
    binaries: list[BinaryCheck]
    models: list[ModelCheck]
    recommended_quality: str
    recommendation_reason: str
    recommended_model: str
    recommended_model_reason: str
    supported_models: list[dict[str, Any]]
    speaker_transcript: dict[str, Any]
    local_ai: dict[str, Any]
    notes: list[str]
    config_source: str = ""
    config_sources: list[str] = field(default_factory=list)


def command_path(name: str) -> str | None:
    return find_binary(name)


def detect_memory_gb() -> float | None:
    system = platform.system().lower()
    try:
        if system == "darwin":
            completed = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
            return round(int(completed.stdout.strip()) / (1024**3), 1)
        if system == "windows":
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True,
                text=True,
                check=True,
            )
            return round(int(completed.stdout.strip()) / (1024**3), 1)
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024**2), 1)
    except Exception:
        return None
    return None


def check_model(name: str) -> ModelCheck:
    from .contracts import PipelineError
    from .whisper_models import cached_model
    try:
        path = cached_model(name, [DEFAULT_MODEL_ROOT, *KNOWN_MODEL_ROOTS])
        return ModelCheck(name=name, path=str(path) if path else "", ok=path is not None)
    except PipelineError:
        return ModelCheck(name=name, path=str(Path(name).expanduser()), ok=False)


def recommend_quality(memory_gb: float | None, accurate_model: ModelCheck, fast_model: ModelCheck) -> tuple[str, str]:
    if accurate_model.ok:
        if memory_gb is None:
            return "accurate", "已找到高精度模型；无法读取内存，默认优先质量。"
        if memory_gb >= 16:
            return "accurate", "内存至少 16GB 且高精度模型可用，推荐高精度。"
        if memory_gb >= 8:
            return "accurate", "内存至少 8GB；可先尝试高精度，如速度不理想再切换更快。"
        if fast_model.ok:
            return "fast", "内存低于 8GB 且更快模型可用，推荐更快模式。"
        return "accurate", "只有高精度模型可用；如运行失败请下载 base 模型或释放内存。"
    if fast_model.ok:
        return "fast", "未找到高精度模型，但找到 base 模型，推荐更快模式。"
    return "accurate", "未找到本地模型；请先在模型管理中确认下载，或运行 download-model。"


def package_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def detect_hf_token_env() -> str | None:
    for name in HF_TOKEN_ENV_VARS:
        if os.environ.get(name):
            return name
    return None


def _speaker_candidate_rows(status: dict[str, Any], candidates: list[str]) -> list[dict[str, Any]]:
    """Normalize legacy speaker status into the configured candidate list."""

    pyannote_installed = bool(status.get("pyannote_installed"))
    hf_token = bool(status.get("hf_token_env"))
    resemblyzer_installed = bool(status.get("resemblyzer_installed"))
    sherpa_installed = bool(status.get("sherpa_onnx_installed") or status.get("sherpa_installed"))
    if not sherpa_installed:
        sherpa_installed = package_available("sherpa_onnx")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        normalized = candidate.lower().replace("_", "-")
        if normalized in {"pyannote", "pyannote-community-1", "pyannote-community1"}:
            snapshot_known = "pyannote_snapshot_ready" in status
            snapshot_ready = bool(status.get("pyannote_snapshot_ready"))
            available = pyannote_installed and (snapshot_ready if snapshot_known else hf_token)
            reason = "ready" if available else (
                "requires pyannote.audio and a pinned local snapshot with matching SHA-256"
                if snapshot_known else "requires pyannote.audio and HF token"
            )
            backend = "pyannote-community-1"
        elif normalized in {"sherpa", "sherpa-onnx", "sherpa-onnx-diarization"}:
            available = sherpa_installed
            reason = "ready" if available else "sherpa_onnx is not installed"
            backend = "sherpa-onnx"
        elif normalized in {"resemblyzer", "resemblyzer-fallback"}:
            available = resemblyzer_installed
            reason = "ready" if available else "resemblyzer is not installed"
            backend = "resemblyzer"
        else:
            available = False
            reason = "unknown backend"
            backend = candidate
        rows.append({"backend": backend, "configured_name": candidate, "available": available, "reason": reason})
    return rows


def run_doctor(
    config: NextEchoConfig | None = None,
    *,
    config_path: str | Path | None = None,
) -> DoctorReport:
    config_obj = config or load_config(config_path)
    binary_checks = [
        BinaryCheck(name=name, path=command_path(name), required=True, ok=command_path(name) is not None)
        for name in REQUIRED_BINARIES
    ]
    binary_checks.extend(
        BinaryCheck(name=name, path=command_path(name), required=False, ok=command_path(name) is not None)
        for name in OPTIONAL_BINARIES
    )
    memory_gb = detect_memory_gb()
    accurate_model = check_model(DEFAULT_MODEL_NAME)
    fast_model = check_model(FAST_MODEL_NAME)
    recommended_quality, reason = recommend_quality(memory_gb, accurate_model, fast_model)
    recommended_model, recommended_model_reason = recommend_whisper_model(memory_gb)
    speaker_backend = detect_speaker_backend_status(config=config_obj)
    accurate_qwen_model = qwen_accurate_model(config_obj)
    fast_qwen_model = qwen_fast_model(config_obj)
    forced_aligner_model = qwen_forced_aligner_model(config_obj)
    qwen_gate = qwen_memory_status(config_obj)
    whisper_binary = command_path("whisper-cli")
    whisper_candidate_model = str(config_obj.get_path("asr", "whisper_fallback", default=DEFAULT_MODEL_NAME))
    whisper_candidate_check = check_model(whisper_candidate_model)
    qwen_candidate = {
        "engine": "qwen3-asr",
        "model": accurate_qwen_model,
        "runtime_available": qwen_mlx_available(),
        "model_cached": qwen_model_cached(accurate_qwen_model),
        "forced_aligner": forced_aligner_model,
        "forced_aligner_cached": qwen_forced_aligner_cached(config=config_obj),
        "memory_gate": qwen_gate,
    }
    qwen_candidate["eligible"] = bool(
        qwen_candidate["runtime_available"]
        and qwen_candidate["model_cached"]
        and qwen_candidate["forced_aligner_cached"]
        and qwen_gate["safe"]
    )
    native_profiles = native_asr_status(config_obj)
    asr_selection = selection_decision(config_obj.get_path("asr", "engine", default="auto"),
        qwen_model=accurate_qwen_model, whisper_model=whisper_candidate_model,
        qwen_eligible=qwen_candidate["eligible"], native=native_profiles)
    whisper_candidate = {
        "engine": "whisper-cpp",
        "model": whisper_candidate_model,
        "binary": whisper_binary or "",
        "binary_available": bool(whisper_binary),
        "model_path": whisper_candidate_check.path or "",
        "model_cached": whisper_candidate_check.ok,
    }
    whisper_candidate["eligible"] = bool(whisper_binary and whisper_candidate_check.ok)
    configured_speaker_candidates = config_obj.get_path(
        "diarization", "benchmark_candidates", default=["pyannote-community-1", "sherpa-onnx", "resemblyzer"]
    )
    if not isinstance(configured_speaker_candidates, list):
        configured_speaker_candidates = []
    speaker_candidates = _speaker_candidate_rows(speaker_backend, [str(item) for item in configured_speaker_candidates])
    local_ai = {
        "apple_silicon": is_apple_silicon(),
        "qwen_mlx_available": qwen_mlx_available(),
        "qwen_memory_safe": bool(qwen_gate["safe"]),
        "qwen_memory": qwen_gate,
        "qwen_asr_models": {"accurate": accurate_qwen_model, "fast": fast_qwen_model},
        "qwen_asr_model_cache": {
            "accurate": qwen_model_cached(accurate_qwen_model),
            "fast": qwen_model_cached(fast_qwen_model),
        },
        "qwen_forced_aligner": {
            "model": forced_aligner_model,
            "cached": qwen_forced_aligner_cached(config=config_obj),
        },
        "asr_candidates": [*native_profiles, qwen_candidate, whisper_candidate],
        "asr_selection": asr_selection,
        "ollama": ollama_status(),
        "llm_models": {
            "fast": str(config_obj.get_path("postprocess", "fallback_model", default=FAST_MODEL)),
            "quality": str(config_obj.get_path("postprocess", "preferred_model_id", default=QUALITY_MODEL)),
        },
        "llm_provider": str(config_obj.get_path("postprocess", "provider", default="selection")),
        "postprocess_profile": str(config_obj.get_path("postprocess", "profile", default="fast")),
    }
    pyannote_installed = bool(speaker_backend["pyannote_installed"])
    resemblyzer_installed = bool(speaker_backend.get("resemblyzer_installed"))
    sherpa_installed = bool(speaker_backend.get("sherpa_onnx_installed") or speaker_backend.get("sherpa_installed") or package_available("sherpa_onnx"))
    hf_token_env = speaker_backend["hf_token_env"] or detect_hf_token_env()
    fallback_backend_available = bool(speaker_backend["fallback_backend_available"])
    notes: list[str] = []
    if not any(binary.name == "yt-dlp" and binary.ok for binary in binary_checks):
        notes.append("yt-dlp 未安装：本地文件和直链媒体仍可用，但网页链接解析可能失败。")
    if not any(binary.name == "curl" and binary.ok for binary in binary_checks):
        notes.append("curl 未安装：模型自动下载和直链媒体下载可能失败。")
    missing_required = [binary.name for binary in binary_checks if binary.required and not binary.ok]
    if missing_required:
        notes.append(f"缺少必需依赖：{', '.join(missing_required)}。")
    if not whisper_candidate["eligible"]:
        notes.append("Whisper fallback 当前不可用；只有在 Qwen 候选可用时才不会影响转录。")
    if not pyannote_installed and not resemblyzer_installed and not sherpa_installed:
        notes.append("pyannote.audio 和本地 speaker embedding 均未安装：当前只能使用轻量聚类分离。")
    elif (resemblyzer_installed or sherpa_installed) and not pyannote_installed:
        notes.append("已启用本地 speaker embedding；如需更高精度，可再配置 pyannote 和 HF_TOKEN。")
    if not hf_token_env:
        notes.append("未检测到 Hugging Face 访问令牌：无法启用 pyannote 高精度说话人分离。")
    if fallback_backend_available and not hf_token_env and not resemblyzer_installed and not sherpa_installed:
        notes.append("当前会回退到本地 segment-clustering 方案；它可用，但多人访谈的准确率低于 pyannote。")
    if local_ai["apple_silicon"] and not local_ai["qwen_mlx_available"]:
        notes.append("Apple Silicon Qwen3-ASR 未安装：高精度转录会回退到 whisper.cpp。")
    elif local_ai["apple_silicon"] and not any(local_ai["qwen_asr_model_cache"].values()):
        notes.append("Qwen3-ASR 运行时已安装，但本地权重尚未完成缓存；auto 将继续使用 whisper.cpp。")
    elif local_ai["apple_silicon"] and not local_ai["qwen_forced_aligner"]["cached"]:
        notes.append("Qwen3-ASR 主模型已缓存，但 ForcedAligner 未完成；需要补齐它才能生成可靠时间戳。")
    if local_ai["qwen_mlx_available"] and not local_ai["qwen_memory_safe"]:
        reasons = ", ".join(qwen_gate.get("reasons") or []) or "资源不足"
        notes.append(f"Qwen 当前未通过可用内存/压力门禁（{reasons}）；auto 暂不加载 Qwen。")
    if not local_ai["ollama"].get("ready"):
        notes.append("本机 Ollama 未就绪：知识整理与摘要不可用。")
    ok = not missing_required
    return DoctorReport(
        ok=ok,
        os=platform.system(),
        machine=platform.machine(),
        python=sys.version.split()[0],
        memory_gb=memory_gb,
        binaries=binary_checks,
        models=[accurate_model, fast_model],
        recommended_quality=recommended_quality,
        recommendation_reason=reason,
        recommended_model=recommended_model,
        recommended_model_reason=recommended_model_reason,
        supported_models=list_supported_whisper_models(),
        speaker_transcript={
            "pyannote_installed": pyannote_installed,
            "resemblyzer_installed": resemblyzer_installed,
            "sherpa_onnx_installed": sherpa_installed,
            "hf_token_env": hf_token_env or "",
            "fallback_backend_available": fallback_backend_available,
            "selected_backend": speaker_backend["selected_backend"],
            "ready": bool(speaker_backend["ready"]) and ok,
            "candidates": speaker_candidates,
            "configured_candidates": [str(item) for item in configured_speaker_candidates],
        },
        local_ai=local_ai,
        notes=notes,
        config_source=config_obj.source,
        config_sources=list(config_obj.source_paths),
    )


def report_to_dict(report: DoctorReport) -> dict[str, Any]:
    return asdict(report)


def print_human_report(report: DoctorReport) -> None:
    status = "OK" if report.ok else "NEEDS_ATTENTION"
    print(f"{PROJECT_NAME} Doctor: {status}")
    print(f"Tagline: {PROJECT_TAGLINE}")
    print(f"OS: {report.os} / {report.machine}")
    print(f"Python: {report.python}")
    print(f"Memory: {report.memory_gb if report.memory_gb is not None else 'unknown'} GB")
    print(f"Config: {report.config_source or 'built-in defaults'}")
    print("\nBinaries:")
    for binary in report.binaries:
        mark = "✓" if binary.ok else "✗"
        required = "required" if binary.required else "optional"
        print(f"  {mark} {binary.name} ({required}) {binary.path or ''}")
    print("\nModels:")
    for model in report.models:
        mark = "✓" if model.ok else "✗"
        print(f"  {mark} {model.name} {model.path or ''}")
    print(f"\nRecommended quality: {report.recommended_quality}")
    print(f"Reason: {report.recommendation_reason}")
    print(f"Recommended model: {report.recommended_model}")
    print(f"Model reason: {report.recommended_model_reason}")
    print("\nSupported models:")
    for model in report.supported_models:
        print(f"  - {model['name']}: {model['summary']} (suggested memory >= {model['min_memory_gb']}GB)")
    print("\nSpeaker transcript:")
    print(f"  pyannote.audio installed: {report.speaker_transcript['pyannote_installed']}")
    print(f"  resemblyzer installed: {report.speaker_transcript.get('resemblyzer_installed', False)}")
    print(f"  HF token env: {report.speaker_transcript['hf_token_env'] or 'missing'}")
    print(f"  fallback backend available: {report.speaker_transcript['fallback_backend_available']}")
    print(f"  selected backend: {report.speaker_transcript['selected_backend'] or 'none'}")
    print(f"  ready: {report.speaker_transcript['ready']}")
    print("  configured candidates:")
    for candidate in report.speaker_transcript.get("candidates", []):
        mark = "✓" if candidate.get("available") else "✗"
        print(f"    {mark} {candidate.get('backend')}: {candidate.get('reason')}")
    print("\nLocal AI:")
    print(f"  Apple Silicon: {report.local_ai['apple_silicon']}")
    print(f"  Qwen3-ASR MLX available: {report.local_ai['qwen_mlx_available']}")
    for candidate in report.local_ai.get("asr_candidates", []):
        mark = "✓" if candidate.get("eligible") else "✗"
        print(f"  {mark} ASR candidate: {candidate.get('engine')} / {candidate.get('model')}")
    memory = report.local_ai.get("qwen_memory") or {}
    if memory:
        print(
            f"  Qwen memory gate: {memory.get('safe')} ({memory.get('available_gb', 'unknown')} GB available, "
            f"pressure={memory.get('pressure')}, swap delta={memory.get('swap_delta_gb', 'unknown')} GB)"
        )
    print(f"  Ollama ready: {report.local_ai['ollama'].get('ready')}")
    if report.notes:
        print("\nNotes:")
        for note in report.notes:
            print(f"  - {note}")


if __name__ == "__main__":
    current = run_doctor()
    if "--json" in sys.argv:
        print(json.dumps(report_to_dict(current), ensure_ascii=False, indent=2))
    else:
        print_human_report(current)
    raise SystemExit(0 if current.ok else 1)
