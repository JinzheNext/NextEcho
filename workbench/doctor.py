from __future__ import annotations

import json
import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .speaker_transcript import HF_TOKEN_ENV_VARS, detect_speaker_backend_status
from .asr import (
    QWEN_ACCURATE_MODEL,
    QWEN_FAST_MODEL,
    QWEN_FORCED_ALIGNER_MODEL,
    is_apple_silicon,
    qwen_forced_aligner_cached,
    qwen_memory_safe,
    qwen_mlx_available,
    qwen_model_cached,
)
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
REQUIRED_BINARIES = ["ffmpeg", "whisper-cli"]
OPTIONAL_BINARIES = ["yt-dlp", "curl"]


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
    candidate = Path(name).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [root / f"ggml-{name}.bin" for root in [DEFAULT_MODEL_ROOT, *KNOWN_MODEL_ROOTS]]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return ModelCheck(name=name, path=str(path), ok=True)
    return ModelCheck(name=name, path=str(candidates[0]), ok=False)


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


def run_doctor() -> DoctorReport:
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
    speaker_backend = detect_speaker_backend_status()
    local_ai = {
        "apple_silicon": is_apple_silicon(),
        "qwen_mlx_available": qwen_mlx_available(),
        "qwen_memory_safe": qwen_memory_safe(),
        "qwen_asr_models": {"accurate": QWEN_ACCURATE_MODEL, "fast": QWEN_FAST_MODEL},
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
    pyannote_installed = bool(speaker_backend["pyannote_installed"])
    resemblyzer_installed = bool(speaker_backend.get("resemblyzer_installed"))
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
    if not pyannote_installed and not resemblyzer_installed:
        notes.append("pyannote.audio 和本地 speaker embedding 均未安装：当前只能使用轻量聚类分离。")
    elif resemblyzer_installed and not pyannote_installed:
        notes.append("已启用本地 speaker embedding；如需更高精度，可再配置 pyannote 和 HF_TOKEN。")
    if not hf_token_env:
        notes.append("未检测到 Hugging Face 访问令牌：无法启用 pyannote 高精度说话人分离。")
    if fallback_backend_available and not hf_token_env and not resemblyzer_installed:
        notes.append("当前会回退到本地 segment-clustering 方案；它可用，但多人访谈的准确率低于 pyannote。")
    if local_ai["apple_silicon"] and not local_ai["qwen_mlx_available"]:
        notes.append("Apple Silicon Qwen3-ASR 未安装：高精度转录会回退到 whisper.cpp。")
    elif local_ai["apple_silicon"] and not any(local_ai["qwen_asr_model_cache"].values()):
        notes.append("Qwen3-ASR 运行时已安装，但本地权重尚未完成缓存；auto 将继续使用 whisper.cpp。")
    elif local_ai["apple_silicon"] and not local_ai["qwen_forced_aligner"]["cached"]:
        notes.append("Qwen3-ASR 主模型已缓存，但 ForcedAligner 未完成；需要补齐它才能生成可靠时间戳。")
    if local_ai["qwen_mlx_available"] and not local_ai["qwen_memory_safe"]:
        notes.append("当前 Swap 超过 MLX 安全门禁；auto 暂不加载 Qwen，释放内存或重启后再试。")
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
            "hf_token_env": hf_token_env or "",
            "fallback_backend_available": fallback_backend_available,
            "selected_backend": speaker_backend["selected_backend"],
            "ready": bool(speaker_backend["ready"]) and ok,
        },
        local_ai=local_ai,
        notes=notes,
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
    print("\nLocal AI:")
    print(f"  Apple Silicon: {report.local_ai['apple_silicon']}")
    print(f"  Qwen3-ASR MLX available: {report.local_ai['qwen_mlx_available']}")
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
