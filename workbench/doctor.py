from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .transcription import DEFAULT_MODEL_NAME, DEFAULT_MODEL_ROOT, resolve_whisper_model

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
    notes: list[str]


def command_path(name: str) -> str | None:
    return shutil.which(name)


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
    try:
        path = resolve_whisper_model(name)
        return ModelCheck(name=name, path=str(path), ok=path.exists() and path.stat().st_size > 0)
    except Exception:
        candidate = DEFAULT_MODEL_ROOT / f"ggml-{name}.bin"
        return ModelCheck(name=name, path=str(candidate), ok=False)


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
    return "accurate", "未找到本地模型；首次运行会按高精度默认模型下载。"


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
    notes: list[str] = []
    if not any(binary.name == "yt-dlp" and binary.ok for binary in binary_checks):
        notes.append("yt-dlp 未安装：本地文件和直链媒体仍可用，但网页链接解析可能失败。")
    if not any(binary.name == "curl" and binary.ok for binary in binary_checks):
        notes.append("curl 未安装：模型自动下载和直链媒体下载可能失败。")
    missing_required = [binary.name for binary in binary_checks if binary.required and not binary.ok]
    if missing_required:
        notes.append(f"缺少必需依赖：{', '.join(missing_required)}。")
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
        notes=notes,
    )


def report_to_dict(report: DoctorReport) -> dict[str, Any]:
    return asdict(report)


def print_human_report(report: DoctorReport) -> None:
    status = "OK" if report.ok else "NEEDS_ATTENTION"
    print(f"Local Transcription Workbench Doctor: {status}")
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
