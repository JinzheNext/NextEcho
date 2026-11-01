from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .config import NextEchoConfig, load_config

# Kept as a compatibility alias for the original web API.  New code should
# use ``qwen_accurate_model()`` so the experimental MLX candidate comes from
# central config.  The old model ID is intentionally not used by auto.
QWEN_ACCURATE_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN_EXPERIMENTAL_ACCURATE_MODEL = "aufklarer/Qwen3-ASR-1.7B-MLX-5bit"
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
    resources: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
    timing_precision: str = "native"
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time resource data used by the MLX safety gate."""

    available_bytes: int | None
    pressure: str
    swap_used_bytes: int | None
    captured_at: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["available_gb"] = _bytes_to_gb(self.available_bytes)
        payload["swap_used_gb"] = _bytes_to_gb(self.swap_used_bytes)
        return payload


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
    network download in the middle of a transcription job. Prepare weights
    separately, including for an explicitly selected engine.
    """
    from .model_profiles import complete_hf_snapshot
    return complete_hf_snapshot(model) is not None


def qwen_accurate_model(config: NextEchoConfig | None = None) -> str:
    """Return the configured experimental accurate model candidate."""

    cfg = config or load_config()
    return str(cfg.get_path("asr", "accurate_model", default=QWEN_EXPERIMENTAL_ACCURATE_MODEL))


def qwen_fast_model(config: NextEchoConfig | None = None) -> str:
    cfg = config or load_config()
    return str(cfg.get_path("asr", "fast_model", default=QWEN_FAST_MODEL))


def qwen_forced_aligner_model(config: NextEchoConfig | None = None) -> str:
    cfg = config or load_config()
    return str(cfg.get_path("asr", "forced_aligner", default=QWEN_FORCED_ALIGNER_MODEL))


def qwen_model_cached(model: str) -> bool:
    return _hf_model_snapshot_cached(model)


def qwen_model_provenance(model: str, *, snapshot_path: Path | None = None) -> dict[str, Any]:
    """Return revision and content-addressed hashes for a cached MLX model."""

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots = hf_home / "hub" / f"models--{model.replace('/', '--')}" / "snapshots"
    candidates = (
        sorted(
            (path for path in snapshots.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots.exists()
        else []
    )
    snapshot = snapshot_path or (candidates[0] if candidates else None)
    files: list[dict[str, Any]] = []
    weight_digests: list[str] = []
    total_size = 0
    if snapshot is not None:
        from .io_utils import file_hash
        for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
            resolved = path.resolve()
            size = resolved.stat().st_size
            total_size += size
            digest = resolved.name if re.fullmatch(r"[0-9a-f]{64}", resolved.name) else file_hash(resolved)
            if path.suffix in {".safetensors", ".bin"} and digest:
                weight_digests.append(digest)
            files.append({"path": str(path.relative_to(snapshot)), "size_bytes": size, "blob_sha256": digest})
    manifest_raw = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    quantization_match = re.search(
        r"(?:^|[-_/])((?:\d+(?:\.\d+)?[- ]?bit)|(?:I?Q\d+(?:_[A-Za-z0-9]+)*))(?=$|[-_/])",
        model,
        flags=re.IGNORECASE,
    )
    return {
        "source_repo": model,
        "revision": snapshot.name if snapshot is not None else "",
        "quantization": quantization_match.group(1) if quantization_match else "",
        "file_size_bytes": total_size,
        "sha256": weight_digests[0] if len(weight_digests) == 1 else "",
        "weight_sha256": weight_digests,
        "cache_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "mlx_version": qwen_runtime_version(),
    }


def qwen_forced_aligner_cached(model: str | None = None, *, config: NextEchoConfig | None = None) -> bool:
    """Return whether the timestamp aligner required by our writer is cached."""
    return _hf_model_snapshot_cached(model or qwen_forced_aligner_model(config))


def _bytes_to_gb(value: int | None) -> float | None:
    return round(value / (1024**3), 3) if value is not None else None


def _parse_size_bytes(value: str, unit: str = "") -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    multiplier = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
    }.get(str(unit).lower(), 1)
    return max(0, int(number * multiplier))


def _read_swap_used_bytes() -> int | None:
    """Read current swap usage without treating cumulative swap as a veto."""

    try:
        completed = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # macOS emits e.g. ``total = 4.00G used = 0.98G free = 3.02G``.
    return parse_swap_usage(completed.stdout or "")


def parse_swap_usage(output: str) -> int | None:
    """Parse a macOS ``vm.swapusage`` line into bytes."""

    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)(?:B)?", output or "", re.I)
    return _parse_size_bytes(match.group(1), match.group(2)) if match else None


def swap_used_bytes() -> int | None:
    """Public read-only swap counter used for stage delta reporting."""

    return _read_swap_used_bytes()


def _read_memory_pressure() -> tuple[str, dict[str, Any]]:
    """Return normalized pressure (normal/warn/critical) and raw details."""

    try:
        completed = subprocess.run(
            ["memory_pressure", "-Q"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        raw = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown", {"source": "memory_pressure", "raw": ""}
    lowered = raw.lower()
    if "critical" in lowered:
        level = "critical"
    elif any(token in lowered for token in ("warn", "warning", "pressure level: 1")):
        level = "warn"
    else:
        percent_match = re.search(r"free\s+percentage\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", lowered)
        if percent_match:
            percent = float(percent_match.group(1))
            level = "critical" if percent <= 5 else "warn" if percent <= 15 else "normal"
        else:
            level = "normal" if re.search(r"\bnormal\b|pressure level:\s*0\b", lowered) else "unknown"
    percent_match = re.search(r"free\s+percentage\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", lowered)
    details: dict[str, Any] = {"source": "memory_pressure", "raw": raw.strip(), "free_percent": None}
    if percent_match:
        details["free_percent"] = float(percent_match.group(1))
    return level, details


def _read_available_memory_bytes() -> tuple[int | None, dict[str, Any]]:
    """Estimate immediately reclaimable memory on macOS/Linux.

    macOS does not expose one stable ``MemAvailable`` counter.  We prefer
    vm_stat's free + inactive + speculative + purgeable pages and use
    memory_pressure's free percentage as a fallback.  Linux support makes the
    helper deterministic in CI and is not used to select MLX on non-Macs.
    """

    system = platform.system().lower()
    if system == "darwin":
        page_size = 4096
        try:
            size_result = subprocess.run(["sysctl", "-n", "vm.pagesize"], capture_output=True, text=True, check=True, timeout=3)
            page_size = int((size_result.stdout or "").strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        try:
            completed = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True, timeout=3)
            values: dict[str, int] = {}
            for name, value in re.findall(r"^(Pages [^:]+):\s*([0-9.]+)", completed.stdout or "", re.M):
                try:
                    values[name.lower()] = int(float(value))
                except ValueError:
                    continue
            reclaimable_names = ("pages free", "pages inactive", "pages speculative", "pages purgeable")
            available = sum(values.get(name, 0) for name in reclaimable_names) * page_size
            if available:
                return available, {"source": "vm_stat", "page_size": page_size, "pages": values}
        except (OSError, subprocess.SubprocessError):
            pass
        # memory_pressure output is useful on macOS versions whose vm_stat
        # labels differ, provided hw.memsize is available.
        try:
            total_result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True, timeout=3)
            total = int((total_result.stdout or "").strip())
            _, pressure_details = _read_memory_pressure()
            percent = pressure_details.get("free_percent")
            if percent is not None:
                return int(total * float(percent) / 100), {"source": "memory_pressure", "total_bytes": total, **pressure_details}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return None, {"source": "darwin", "error": "available memory unavailable"}
    proc_meminfo = Path("/proc/meminfo")
    if proc_meminfo.exists():
        try:
            values: dict[str, int] = {}
            for line in proc_meminfo.read_text(encoding="utf-8").splitlines():
                match = re.match(r"^(MemAvailable|MemFree|Buffers|Cached):\s*(\d+)\s*kB", line)
                if match:
                    values[match.group(1)] = int(match.group(2)) * 1024
            available = values.get("MemAvailable") or sum(values.get(key, 0) for key in ("MemFree", "Buffers", "Cached"))
            return (available or None), {"source": "/proc/meminfo", "values": values}
        except (OSError, ValueError):
            pass
    return None, {"source": system or "unknown", "error": "available memory unavailable"}


def memory_snapshot() -> MemorySnapshot:
    """Capture available memory, pressure and cumulative swap once."""

    available, available_details = _read_available_memory_bytes()
    pressure, pressure_details = _read_memory_pressure()
    return MemorySnapshot(
        available_bytes=available,
        pressure=pressure,
        swap_used_bytes=_read_swap_used_bytes(),
        captured_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        details={"available": available_details, "pressure": pressure_details},
    )


# Descriptive aliases make the resource API discoverable without breaking the
# short names used internally.
capture_memory_snapshot = memory_snapshot
get_memory_snapshot = memory_snapshot


def _coerce_snapshot(value: MemorySnapshot | dict[str, Any] | None) -> MemorySnapshot | None:
    if value is None:
        return None
    if isinstance(value, MemorySnapshot):
        return value
    if isinstance(value, dict):
        available_bytes = _int_or_none(value.get("available_bytes"))
        if available_bytes is None and value.get("available_gb") is not None:
            try:
                available_bytes = int(float(value["available_gb"]) * 1024**3)
            except (TypeError, ValueError):
                pass
        swap_bytes = _int_or_none(value.get("swap_used_bytes"))
        if swap_bytes is None and value.get("swap_used_gb") is not None:
            try:
                swap_bytes = int(float(value["swap_used_gb"]) * 1024**3)
            except (TypeError, ValueError):
                pass
        return MemorySnapshot(
            available_bytes=available_bytes,
            pressure=str(value.get("pressure") or "unknown"),
            swap_used_bytes=swap_bytes,
            captured_at=str(value.get("captured_at") or ""),
            details=dict(value.get("details") or {}),
        )
    raise TypeError("memory snapshot must be MemorySnapshot, dict, or None")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def swap_delta_bytes(before: MemorySnapshot | dict[str, Any] | None, after: MemorySnapshot | dict[str, Any] | None) -> int | None:
    before_snapshot = _coerce_snapshot(before)
    after_snapshot = _coerce_snapshot(after)
    if not before_snapshot or not after_snapshot or before_snapshot.swap_used_bytes is None or after_snapshot.swap_used_bytes is None:
        return None
    return after_snapshot.swap_used_bytes - before_snapshot.swap_used_bytes


def qwen_memory_status(
    config: NextEchoConfig | None = None,
    *,
    snapshot: MemorySnapshot | dict[str, Any] | None = None,
    baseline: MemorySnapshot | dict[str, Any] | None = None,
    available_memory_gb: float | None = None,
    available_gb: float | None = None,
    pressure: str | None = None,
    swap_delta_gb: float | None = None,
) -> dict[str, Any]:
    """Evaluate the MLX gate and return explainable diagnostics.

    Cumulative swap is intentionally informational.  Only a measured
    before/after swap delta can veto a run, preventing an old unrelated swap
    allocation from disabling Qwen forever.
    """

    cfg = config or load_config()
    snap = _coerce_snapshot(snapshot) or memory_snapshot()
    if available_memory_gb is not None or available_gb is not None or pressure is not None or swap_delta_gb is not None:
        available_value = snap.available_bytes
        requested_available = available_memory_gb if available_memory_gb is not None else available_gb
        if requested_available is not None:
            available_value = int(float(requested_available) * 1024**3)
        swap_value = snap.swap_used_bytes
        baseline_value = _coerce_snapshot(baseline)
        if swap_delta_gb is not None:
            # Keep the reported delta deterministic for callers that already
            # measured it; the synthetic baseline is never written to disk.
            base_swap = baseline_value.swap_used_bytes if baseline_value and baseline_value.swap_used_bytes is not None else 0
            swap_value = base_swap + int(float(swap_delta_gb) * 1024**3)
            baseline = MemorySnapshot(available_bytes=None, pressure="normal", swap_used_bytes=base_swap, captured_at="")
        snap = MemorySnapshot(
            available_bytes=available_value,
            pressure=pressure or snap.pressure,
            swap_used_bytes=swap_value,
            captured_at=snap.captured_at,
            details=snap.details,
        )
    base = _coerce_snapshot(baseline)
    reserve_gb = float(cfg.get_path("hardware", "reserve_memory_gb", default=3.0) or 3.0)
    max_swap_delta_gb = float(cfg.get_path("hardware", "max_swap_delta_gb", default=1.0) or 1.0)
    max_pressure = str(cfg.get_path("hardware", "memory_pressure_max", default="warn") or "warn").lower()
    reserve_bytes = int(reserve_gb * 1024**3)
    max_swap_delta_bytes = int(max_swap_delta_gb * 1024**3)
    delta = swap_delta_bytes(base, snap)
    reasons: list[str] = []
    safe = True
    if not is_apple_silicon():
        safe = False
        reasons.append("not_apple_silicon")
    if snap.available_bytes is None:
        safe = False
        reasons.append("available_memory_unknown")
    elif snap.available_bytes < reserve_bytes:
        safe = False
        reasons.append("reserve_memory_not_met")
    pressure_rank = {"normal": 0, "warn": 1, "critical": 2, "unknown": 99}
    if pressure_rank.get(snap.pressure, 99) > pressure_rank.get(max_pressure, 1):
        safe = False
        reasons.append(f"memory_pressure_{snap.pressure}")
    if delta is not None and delta > max_swap_delta_bytes:
        safe = False
        reasons.append("swap_delta_exceeded")
    if snap.pressure == "unknown":
        reasons.append("memory_pressure_unknown")
    return {
        "safe": safe,
        "available_bytes": snap.available_bytes,
        "available_gb": _bytes_to_gb(snap.available_bytes),
        "pressure": snap.pressure,
        "reserve_bytes": reserve_bytes,
        "reserve_gb": reserve_gb,
        "swap_used_bytes": snap.swap_used_bytes,
        "swap_used_gb": _bytes_to_gb(snap.swap_used_bytes),
        "swap_delta_bytes": delta,
        "swap_delta_gb": _bytes_to_gb(delta),
        "max_swap_delta_bytes": max_swap_delta_bytes,
        "max_swap_delta_gb": max_swap_delta_gb,
        "memory_pressure_max": max_pressure,
        "captured_at": snap.captured_at,
        "reasons": reasons,
        "details": snap.details,
    }


def qwen_memory_safe(
    config: NextEchoConfig | None = None,
    *,
    snapshot: MemorySnapshot | dict[str, Any] | None = None,
    baseline: MemorySnapshot | dict[str, Any] | None = None,
    available_memory_gb: float | None = None,
    available_gb: float | None = None,
    pressure: str | None = None,
    swap_delta_gb: float | None = None,
) -> bool:
    return bool(
        qwen_memory_status(
            config,
            snapshot=snapshot,
            baseline=baseline,
            available_memory_gb=available_memory_gb,
            available_gb=available_gb,
            pressure=pressure,
            swap_delta_gb=swap_delta_gb,
        )["safe"]
    )


def choose_asr_engine(
    requested: str = "auto",
    *,
    model: str = "",
    config: NextEchoConfig | None = None,
) -> str:
    return choose_asr_plan(requested, model=model, config=config)["engine"]


def choose_asr_plan(requested="auto", *, model="", config=None):
    from .model_profiles import native_asr_status, selection_decision
    cfg = config or load_config()
    candidate = model or qwen_accurate_model(cfg)
    eligible = (qwen_mlx_available() and qwen_memory_safe(cfg) and qwen_model_cached(candidate)
                and qwen_forced_aligner_cached(config=cfg)) if requested == "auto" else False
    return selection_decision(requested, qwen_model=candidate,
               whisper_model=cfg.get_path("asr", "whisper_fallback", default="large-v3-turbo-q5_0"),
               qwen_eligible=eligible, native=native_asr_status(cfg))


def qwen_model_for_quality(quality: str, config: NextEchoConfig | None = None) -> str:
    return qwen_fast_model(config) if quality == "fast" else qwen_accurate_model(config)


def transcribe_qwen_mlx(
    *,
    audio_path: Path,
    item_dir: Path,
    model: str,
    language: str,
    timeout_seconds: int = 60 * 60,
    config: NextEchoConfig | None = None,
) -> TranscriptResult:
    """Run MLX ASR in a short-lived child process so Metal memory is released."""
    if not qwen_mlx_available():
        raise RuntimeError("Qwen3-ASR MLX is unavailable. Install the Apple Silicon optional runtime first.")
    resource_before = memory_snapshot()
    from .model_profiles import complete_hf_snapshot
    from .contracts import PipelineError
    resolved_snapshot = complete_hf_snapshot(model)
    if resolved_snapshot is None:
        raise PipelineError("MODEL_MISSING", "Qwen本地模型未准备完整，请先准备权重", blocked=True)
    if not qwen_memory_safe(config, snapshot=resource_before):
        raise PipelineError("RESOURCE_BLOCKED", "Qwen载入前资源检查未通过", blocked=True)
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
            str(resolved_snapshot),
            "--language",
            language,
            "--output",
            str(result_path),
        ]
        from .resources import GIB, model_policy, run_owned
        from .job_store import current_cancelled
        provenance = qwen_model_provenance(model, snapshot_path=resolved_snapshot)
        estimate = int(provenance["file_size_bytes"] * 1.5) + 512 * 1024**2
        run_owned(command, log_path=item_dir / "logs/qwen-asr.log",
                  policy=model_policy(estimate, max(estimate * 3, GIB), timeout_seconds, config), cancel=current_cancelled)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = TranscriptResult(
            text=str(payload.get("text", "")).strip(),
            language=str(payload.get("language", "")),
            words=list(payload.get("words") or []),
            backend="qwen3-asr",
            model=model,
            version=str(payload.get("version") or ""),
            provenance=qwen_model_provenance(model, snapshot_path=resolved_snapshot),
        )
        resource_after = memory_snapshot()
        result.resources = {
            "memory_before": resource_before.to_dict(),
            "memory_after": resource_after.to_dict(),
            "swap_delta_bytes": swap_delta_bytes(resource_before, resource_after),
            "swap_delta_gb": _bytes_to_gb(swap_delta_bytes(resource_before, resource_after)),
            "gate": qwen_memory_status(config, snapshot=resource_before, baseline=None),
        }
        return result
    finally:
        result_path.unlink(missing_ok=True)


def qwen_runtime_version() -> str:
    try:
        return importlib.metadata.version("mlx-qwen3-asr")
    except importlib.metadata.PackageNotFoundError:
        return ""
