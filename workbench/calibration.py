"""Bounded short-sample resource calibration; never promotes model quality."""
from __future__ import annotations

import time
import sys
import wave
from pathlib import Path

from .config import NextEchoConfig
from .contracts import PipelineError
from .io_utils import file_hash, write_json
from .local_asr import transcribe_local_audio
from .model_profiles import PROFILES
from .resources import GIB, private_dir, sample_resources


def calibrate(config: NextEchoConfig, profile_id: str, audio: Path, destination: Path) -> dict:
    profile = PROFILES[profile_id]
    with wave.open(str(audio), "rb") as f:
        seconds = f.getnframes() / f.getframerate()
        if not 0 < seconds <= 30 or f.getframerate() != 16000 or f.getnchannels() != 1:
            raise PipelineError("INVALID_INPUT", "校准样本必须为≤30秒的16kHz单声道WAV")
    private_dir(destination)
    data = config.to_dict()
    data.setdefault("asr", {}).update(allow_fallback=False, strict_selection=True)
    settings = data.setdefault("models", {}).setdefault(profile_id, {})
    # A new controlled calibration intentionally replaces stale bindings.
    settings.pop("model_sha256", None)
    settings.pop("runtime_versions", None)
    root = Path(settings.get("path", ""))
    size = sum((root / name).stat().st_size for name in profile.required_files if (root / name).is_file())
    # Provisional calibration-only limits are derived and logged, not claimed
    # as measured production limits. The preflight can refuse this probe.
    estimate = int(size * 3) + GIB if profile.engine == "fun-asr-nano" else int(size * 1.5) + 256 * 1024**2
    settings.update(estimated_bytes=estimate, max_rss_bytes=int(estimate * 1.5),
                    timeout_seconds=180, resource_sample_seconds=.1)
    report = {"schema_version": 1, "profile": profile_id, "audio_sha256": file_hash(audio),
              "sample_seconds": seconds, "memory_before": sample_resources(), "provisional_estimate_bytes": estimate,
              "provisional_max_rss_bytes": settings["max_rss_bytes"],
              "created_at": time.time(), "release_status": "candidate"}
    try:
        result = transcribe_local_audio(audio, destination, engine=profile.engine, model=str(root), language="zh", config=NextEchoConfig(data))
        peak = result.resources.get("peak_worker_rss_bytes")
        if not peak:
            raise PipelineError("RESOURCE_BLOCKED", "校准没有取得有效峰值内存，不能放行", blocked=True)
        settings.update(estimated_bytes=int(peak * 1.3), max_rss_bytes=int(peak * 1.8), timeout_seconds=7200,
                        resource_sample_seconds=2, model_sha256=result.provenance["sha256"], release_status="candidate")
        from .capabilities import NATIVE_REQUIREMENTS, probe_environment
        settings["runtime_versions"] = probe_environment(settings.get("worker_python") or sys.executable, NATIVE_REQUIREMENTS[profile_id])["versions"]
        settings["calibration_files"] = {name: file_hash(root / name) for name in profile.required_files}
        settings["vad_sha256"] = result.provenance["vad_sha256"]
        report.update(status="calibrated", measured_peak_bytes=peak, settings=settings, provenance=result.provenance,
                      resources=result.resources)
        return report
    except PipelineError as exc:
        report.update(status="blocked" if exc.blocked else "failed", error_code=exc.code, message=str(exc))
        raise
    finally:
        write_json(destination / "calibration.json", report)
