from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


QWEN_ACCURATE_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN_FAST_MODEL = "Qwen/Qwen3-ASR-0.6B"
QWEN_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"


@dataclass
class TranscriptResult:
    text: str
    language: str
    words: list[dict[str, Any]]
    backend: str
    model: str
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AsrBackend(Protocol):
    """Common contract for local ASR engines."""

    name: str

    def transcribe(self, audio_path: Path, *, language: str, model: str) -> TranscriptResult:
        ...


class QwenMlxBackend:
    name = "qwen3-asr"

    def transcribe(self, audio_path: Path, *, language: str, model: str) -> TranscriptResult:
        return transcribe_qwen_mlx(audio_path=audio_path, item_dir=audio_path.parent, model=model, language=language)


class WhisperCppBackend:
    name = "whisper-cpp"

    def __init__(self, whisper_bin: str = "whisper-cli") -> None:
        self.whisper_bin = whisper_bin

    def transcribe(self, audio_path: Path, *, language: str, model: str) -> TranscriptResult:
        """The established writer remains in transcription.py; this adapter gives callers one shape."""
        import tempfile
        from .transcription import transcribe_audio_file

        with tempfile.TemporaryDirectory(dir=audio_path.parent) as temp_dir:
            work_dir = Path(temp_dir)
            transcribe_audio_file(
                audio_path=audio_path,
                output_dir=work_dir,
                output_base="transcript",
                whisper_bin=self.whisper_bin,
                model_path=Path(model),
                language=language,
                output_formats=["txt", "json"],
            )
            payload = json.loads((work_dir / "transcript.json").read_text(encoding="utf-8"))
            segments = payload.get("transcription") or []
            return TranscriptResult(
                text="\n".join(str(item.get("text") or "") for item in segments),
                language=language,
                words=[],
                backend=self.name,
                model=model,
            )


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def qwen_mlx_available() -> bool:
    return is_apple_silicon() and importlib.util.find_spec("mlx_qwen3_asr") is not None


def _hf_model_snapshot_cached(model: str) -> bool:
    """Return true only when a complete local HF snapshot is present.

    Installing the MLX package alone must not make ``auto`` trigger a multi-GB
    network download in the middle of a transcription job.  Explicit
    ``--asr-engine qwen3-asr`` still permits the worker to download on demand.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / f"models--{model.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return False
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir():
            continue
        for candidate in snapshot.rglob("*"):
            if not candidate.is_file() or candidate.suffix not in {".safetensors", ".bin"}:
                continue
            resolved = candidate.resolve()
            if resolved.name.endswith(".incomplete"):
                continue
            if resolved.stat().st_size >= 5 * 1024 * 1024:
                return True
    return False


def qwen_model_cached(model: str) -> bool:
    return _hf_model_snapshot_cached(model)


def qwen_forced_aligner_cached() -> bool:
    """Return whether the timestamp aligner required by our writer is cached."""
    return _hf_model_snapshot_cached(QWEN_FORCED_ALIGNER_MODEL)


def qwen_memory_safe() -> bool:
    """Honor the local 16GB Mac safety gate before auto-loading MLX."""
    if not is_apple_silicon():
        return False
    try:
        completed = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True)
        match = re.search(r"used\s*=\s*([0-9.]+)M", completed.stdout)
        if not match:
            return True
        return float(match.group(1)) * 1024 * 1024 <= 512 * 1024 * 1024
    except Exception:
        return True


def choose_asr_engine(requested: str = "auto", *, model: str = "") -> str:
    if requested not in {"auto", "qwen3-asr", "whisper-cpp"}:
        raise ValueError(f"Unsupported ASR engine: {requested}")
    if requested == "auto":
        return "qwen3-asr" if (
            qwen_mlx_available()
            and qwen_memory_safe()
            and qwen_model_cached(model or QWEN_ACCURATE_MODEL)
            and qwen_forced_aligner_cached()
        ) else "whisper-cpp"
    return requested


def qwen_model_for_quality(quality: str) -> str:
    return QWEN_FAST_MODEL if quality == "fast" else QWEN_ACCURATE_MODEL


def transcribe_qwen_mlx(
    *,
    audio_path: Path,
    item_dir: Path,
    model: str,
    language: str,
    timeout_seconds: int = 60 * 60,
) -> TranscriptResult:
    """Run MLX ASR in a short-lived child process so Metal memory is released."""
    if not qwen_mlx_available():
        raise RuntimeError("Qwen3-ASR MLX is unavailable. Install the Apple Silicon optional runtime first.")
    # A deterministic, human-readable scratch name makes interrupted runs
    # understandable.  The file is always removed in ``finally``.
    result_path = item_dir / "临时-Qwen-ASR.json"
    result_path.unlink(missing_ok=True)
    try:
        command = [
            sys.executable,
            "-m",
            "workbench.qwen_asr_worker",
            "--audio",
            str(audio_path),
            "--model",
            model,
            "--language",
            language,
            "--output",
            str(result_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Qwen3-ASR worker failed.").strip()
            raise RuntimeError(detail)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return TranscriptResult(
            text=str(payload.get("text", "")).strip(),
            language=str(payload.get("language", "")),
            words=list(payload.get("words") or []),
            backend="qwen3-asr",
            model=str(payload.get("model") or model),
            version=str(payload.get("version") or ""),
        )
    finally:
        result_path.unlink(missing_ok=True)


def qwen_runtime_version() -> str:
    try:
        return importlib.metadata.version("mlx-qwen3-asr")
    except importlib.metadata.PackageNotFoundError:
        return ""
