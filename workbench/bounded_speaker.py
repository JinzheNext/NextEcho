"""Product entry for the unchanged speaker algorithm in an owned worker."""
from __future__ import annotations

import functools
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from .config import load_config
from .media_commands import run_media_command
from .contracts import PipelineError
from .io_utils import write_json
from .job_store import current_cancelled
from .resources import GIB, model_policy, private_dir, run_owned, runtime_dir, serialized_task
from .speaker_transcript import build_speaker_transcript


@functools.wraps(build_speaker_transcript)
@serialized_task
def bounded_build_speaker_transcript(input_path, *, progress_callback=None, config=None, **kwargs):
    config = config or load_config()
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    if kwargs.get("backend", "auto") != "auto":
        data.setdefault("diarization", {})["backend"] = kwargs["backend"]
    settings = data.get("diarization", {})
    configured_python = str(settings.get("worker_python") or sys.executable)
    if re.fullmatch(r"\d+\.\d+", configured_python):
        configured_python = sys.executable if configured_python == f"{sys.version_info.major}.{sys.version_info.minor}" else shutil.which("python" + configured_python)
    if not configured_python or not Path(configured_python).expanduser().is_file():
        raise PipelineError("DEPENDENCY_MISSING", "说话人独立解释器不可用，请绑定diarization.worker_python", blocked=True)
    python = str(Path(configured_python).expanduser().absolute())
    worker = Path(__file__).with_name("speaker_worker.py")
    with tempfile.TemporaryDirectory(prefix="speaker-", dir=private_dir(runtime_dir() / "requests")) as scratch:
        folder = Path(scratch)
        request, response, progress = folder / "request.json", folder / "response.json", folder / "progress.json"
        request_data = {"input": str(input_path), "config": data, "kwargs": {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
                        "output": str(response), "progress": str(progress)}
        write_json(request, request_data)
        probe = run_media_command([python, str(worker), str(request), "--probe"], timeout=10, stage="speaker-probe", check=False, config=config)
        if probe.returncode or not response.is_file():
            raise PipelineError("DEPENDENCY_MISSING", "说话人环境无法导入运行时，检查所选环境依赖", blocked=True)
        status = json.loads(response.read_text())
        if not status.get("ready"):
            raise PipelineError("DEPENDENCY_MISSING", "所选说话人后端缺依赖、权重或访问授权，请检查doctor --capability speaker", blocked=True)
        backend = status["selected_backend"]
        limits = settings.get("resources", {}).get(backend, settings)
        if backend in {"segment-clustering", "timeline-fallback", "local", "fallback"}:
            source = Path(input_path)
            size = source.stat().st_size if source.is_file() else sum(p.stat().st_size for p in source.rglob("*.wav"))
            estimate = max(256 * 1024**2, size * 5 + 256 * 1024**2)
            maximum = max(GIB, estimate * 2)
        else:
            estimate, maximum = limits.get("estimated_bytes", 0), limits.get("max_rss_bytes", 0)
        policy = model_policy(limits.get("estimated_bytes", estimate), limits.get("max_rss_bytes", maximum),
                              limits.get("timeout_seconds", 7200), config)
        response.unlink()
        last_event = None
        def on_sample(_):
            nonlocal last_event
            if progress_callback and progress.is_file():
                event = json.loads(progress.read_text())
                if event != last_event:
                    progress_callback(event)
                    last_event = event
        log = private_dir(runtime_dir() / "speaker-logs") / f"{folder.name}.log"
        resources = run_owned([python, str(worker), str(request)], log_path=log, policy=policy,
                cancel=current_cancelled, on_sample=on_sample)
        on_sample(None)
        if not response.is_file() or response.stat().st_size > 32 * 1024**2:
            raise PipelineError("PROTOCOL_ERROR", "说话人worker结果缺失或过大")
        payload = json.loads(response.read_text())
        if payload.get("schema_version") != 1:
            raise PipelineError("PROTOCOL_ERROR", "说话人worker版本不兼容")
        if payload.get("error_code"):
            raise PipelineError(payload["error_code"], payload.get("message", "说话人阶段失败"), blocked=payload.get("blocked", False), stage="speaker")
        result = payload["result"]
        result["resources"] = resources
        return result
