"""Explicit bounded Metal candidate probes, never a default or fallback."""
from __future__ import annotations

import json
import math
import array
import os
from pathlib import Path
import sys
import wave

from .contracts import PipelineError
from .io_utils import file_hash, write_json
from .resources import ExecutionLease, GIB, model_policy, run_owned
from .nemotron_timeline import validate_segments, verify_long_timeline
from .nemotron_probabilities import verify_full_probability_archive


def validate_candidate(data: dict, folder: Path, *, duration: float, origin_ms: int,
                       long_context: bool = False, retain_full_probabilities: bool = False) -> dict:
    try:
        count, slots, cadence = data["frame_count"], data["speaker_slot_capacity"], data["seconds_per_frame"]
        first = data["retained_frame_start"]
        if (data["schema_version"] != (2 if long_context else 1) or data["artifact_kind"] != "nemotron-native-candidate"
                or type(count) is not int or not 0 < count <= math.ceil(duration * 100) + 16
                or slots != 8 or not math.isclose(cadence, 0.01, abs_tol=1e-8)
                or type(first) is not int or not 0 <= first < count or (not long_context and first != 0)
                or data["origin_ms"] != origin_ms
                or not math.isclose(data["duration_seconds"], duration, abs_tol=1e-6)
                or data["backend_requested"] != "metal" or data["preset"] != "v3-offline"
                or data["full_attention"] is not False):
            raise ValueError("wrong native candidate metadata")
        path = Path(data["probability_path"])
        if not path.is_absolute() or not path.resolve().is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError("external/missing probabilities")
        if path.stat().st_size != (count - first) * slots * 4 or file_hash(path) != data["probability_sha256"]:
            raise ValueError("native probability bytes changed")
        probabilities = array.array("f", path.read_bytes())
        if sys.byteorder != "little":
            probabilities.byteswap()
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities):
            raise ValueError("nonfinite/out-of-range probabilities")
        validate_segments(data["segments"], duration)
        if long_context:
            verify_long_timeline(data, folder, probabilities)
        if bool(data.get("full_probability_archive")) != retain_full_probabilities:
            raise ValueError("requested full probability archive is missing or unexpected")
        if retain_full_probabilities:
            verify_full_probability_archive(data, folder)
        return data
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise PipelineError("PROTOCOL_ERROR", "Nemotron候选帧/区间/概率文件无效", blocked=True) from exc


def probe_nemotron(audio: Path, folder: Path, *, ledger: Path, runtime_root: Path,
                   origin_ms: int, cancel=lambda: False, sample_interval=0.2,
                   long_context: bool = False, restart_from: Path | None = None,
                   retain_full_probabilities: bool = False, config=None) -> dict:
    if type(long_context) is not bool:
        raise PipelineError("INVALID_INPUT", "long_context必须为显式布尔值")
    if type(retain_full_probabilities) is not bool or (retain_full_probabilities and not long_context):
        raise PipelineError("INVALID_INPUT", "完整概率归档须显式长模式和布尔开关")
    with wave.open(str(audio), "rb") as f:
        duration = f.getnframes() / f.getframerate()
        if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype()) != (1, 2, 16000, "NONE") or not 0 < duration <= (14400 if long_context else 300):
            raise PipelineError("INVALID_INPUT", "Nemotron需mono16k PCM16；普通探针≤300秒，显式长上下文实验≤4小时")
    if type(origin_ms) is not int or origin_ms < 0:
        raise PipelineError("INVALID_INPUT", "真实音频origin_ms必须为非负整数")
    locked, library = validate_assets(ledger, runtime_root)
    model = locked["model"]
    if folder.exists():
        raise PipelineError("INVALID_INPUT", "探针目录必须为新目录，保留旧成功/失败证据")
    if restart_from is not None and (folder.resolve().is_relative_to(Path(restart_from).resolve())
                                    or Path(restart_from).resolve().is_relative_to(folder.resolve())):
        raise PipelineError("INVALID_INPUT", "重放目录须与原尝试分开，不得嵌套")
    folder.mkdir(parents=True, mode=0o700)
    request = {"schema_version": 1, "audio_path": str(audio.resolve()), "audio_sha256": file_hash(audio),
        "origin_ms": origin_ms, "model_path": model["path"], "model_sha256": model["sha256"],
        "library": str(library),
        "output": str((folder / "native.json").resolve()), "progress": str((folder / "worker.state.json").resolve()),
        "ledger_sha256": file_hash(ledger), "worker_sha256": file_hash(Path(__file__).with_name("nemotron_worker.py")),
        "timeline_sha256": file_hash(Path(__file__).with_name("nemotron_timeline.py")),
        "probabilities_code_sha256": file_hash(Path(__file__).with_name("nemotron_probabilities.py")),
        "controller_sha256": file_hash(Path(__file__)), "long_context": long_context,
        "retain_full_probabilities": retain_full_probabilities}
    if restart_from is not None:
        try:
            from .resources import process_alive, group_running
            previous_request = Path(restart_from) / "request.json"
            previous = json.loads(previous_request.read_text())
            keys = ("audio_sha256", "origin_ms", "model_sha256", "library", "ledger_sha256", "worker_sha256", "timeline_sha256", "probabilities_code_sha256", "controller_sha256", "long_context", "retain_full_probabilities")
            if any(previous.get(k) != request[k] for k in keys) or (Path(restart_from) / "probe.result.json").exists():
                raise ValueError("restart inputs changed or previous attempt already completed")
            owner = Path(restart_from) / "worker.guard.owner.json"
            if owner.is_file():
                info = json.loads(owner.read_text())
                if process_alive(info["pid"], info["birth"]) or group_running(info["pgid"]):
                    raise ValueError("previous owned process group still alive")
            elif not (Path(restart_from) / "probe.failure.json").is_file():
                raise ValueError("previous state cannot be independently verified")
            request["recovery"] = {"mode": "whole-stage-replay-no-state-import",
                "parent_request_sha256": file_hash(previous_request), "parent": str(Path(restart_from).resolve())}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PipelineError("INVALID_INPUT", "无法核验原尝试已结束且输入相同，不能重放") from exc
    write_json(folder / "request.json", request)
    for name in ("nemotron_worker.py", "nemotron_timeline.py", "nemotron_probabilities.py"):
        (folder / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (folder / "controller.source.py").write_bytes(Path(__file__).read_bytes())
    policy = model_policy(GIB, 4 * GIB, 900 if long_context else 240 if duration <= 60 else 600, config)
    try:
        def check_wait():
            if cancel():
                raise PipelineError("CANCELLED", "Nemotron探针排队已取消")
        check_wait()
        def check_log(_sample):
            log = folder / "worker.log"
            if log.is_file() and log.stat().st_size > 8 * 1024**2:
                raise PipelineError("OUTPUT_LIMIT", "Nemotron诊断超过8MiB，停止所属探针")
        with ExecutionLease(timeout=60, on_wait=check_wait):
            resources = run_owned(["/usr/bin/env", "-u", "DYLD_LIBRARY_PATH", "-u", "DYLD_INSERT_LIBRARIES",
                sys.executable, str(Path(__file__).with_name("nemotron_worker.py")), str((folder / "request.json").resolve())],
                log_path=folder / "worker.log", policy=policy, cancel=cancel, interval=sample_interval, on_sample=check_log)
        data = validate_candidate(json.loads((folder / "native.json").read_text()), folder, duration=duration, origin_ms=origin_ms,
                                  long_context=long_context, retain_full_probabilities=retain_full_probabilities)
        if file_hash(audio) != request["audio_sha256"]:
            raise PipelineError("CACHE_MISMATCH", "探针音频在运行中改变", blocked=True)
        report = {"schema_version": 1, "status": "candidate-probe-complete", "native": data, "resources": resources,
                  "effective_reserve_bytes": policy.reserve_bytes, "worker_rss_ceiling_bytes": policy.max_rss_bytes,
                  "gpu_memory_measurement": "not-independently-measured", "quality_validation": "pending-author-gold",
                  "precise_alignment_release": False, "default_promotion": False,
                  "recovery": request.get("recovery"), "long_context": long_context,
                  "full_probability_archive_validated": retain_full_probabilities}
        write_json(folder / "probe.result.json", report)
        return report
    except Exception as exc:
        write_json(folder / "probe.failure.json", {"error_type": type(exc).__name__, "error_code": getattr(exc, "code", ""),
            "effective_reserve_bytes": policy.reserve_bytes, "message": str(exc), "success": False,
            "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
            "cause_message": str(exc.__cause__)[:512] if exc.__cause__ else None})
        raise


def validate_assets(ledger: Path, runtime_root: Path) -> tuple[dict, Path]:
    """Read-only verification, also required before reusing a completed stage."""
    if not ledger.exists() and not ledger.is_symlink():
        raise PipelineError("MODEL_MISSING", "Nemotron分人资产台账文件不存在，请重新绑定分人模型资产", blocked=True)
    try:
        return _validate_assets(ledger, runtime_root)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PipelineError("MODEL_INVALID", "Nemotron固定资产台账缺失或无效", blocked=True) from exc


def _validate_assets(ledger: Path, runtime_root: Path) -> tuple[dict, Path]:
    locked = json.loads(ledger.read_text())
    model = locked["model"]
    if (model["revision"] != "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
            or locked["code"]["commit"] != "6a3ca369370782acee0dd155e34394ba60b920c4"
            or model["sha256"] != "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"
            or locked["code"]["release_archive_sha256"] != "5cb02ba7c04f0b5585ce5cde9c830be5f0b7dfb4c83083c500109c381b4f2da9"
            or file_hash(Path(model["path"])) != model["sha256"]):
        raise PipelineError("MODEL_INVALID", "Nemotron候选固定代码/权重身份不符", blocked=True)
    for name, row in locked["runtime_files"].items():
        path = runtime_root / name
        if not path.resolve().is_relative_to(runtime_root.resolve()) or not path.is_file() or file_hash(path) != row["sha256"]:
            raise PipelineError("MODEL_INVALID", "Nemotron候选运行时文件改变", blocked=True)
    if "runtime_symlinks" not in locked:
        raise PipelineError("MODEL_INVALID", "候选运行时需要包含官方归档符号链接台账", blocked=True)
    for name, target in locked["runtime_symlinks"].items():
        path = runtime_root / name
        if not path.is_symlink() or os.readlink(path) != target or not path.resolve().is_relative_to(runtime_root.resolve()):
            raise PipelineError("MODEL_INVALID", "Nemotron运行时链接改变或越界", blocked=True)
    libraries = [name for name in locked["runtime_files"] if name.endswith("/lib/libnemo_speech_asr_c.1.dylib")]
    if len(libraries) != 1:
        raise PipelineError("MODEL_INVALID", "固定运行时缺少唯一diar C ABI库", blocked=True)
    return locked, (runtime_root / libraries[0]).resolve()
