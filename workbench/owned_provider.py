"""Opt-in dedicated model service, started only after ASR has released RAM.

An external listener is never adopted or stopped. Startup commands and budgets
come from installation configuration, never from media or model output.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from .contracts import PipelineError
from .io_utils import write_json
from .job_store import JobStore, current_cancelled
from .llm_providers import endpoint_url, request_json
from .resources import ExecutionLease, group_rss, group_running, model_policy, private_dir, process_birth, runtime_dir, sample_resources, stop_owned


def listeners(port):
    if not shutil.which("lsof"):
        raise PipelineError("DEPENDENCY_MISSING", "专用本机服务需要lsof验证端口所有权", blocked=True)
    checked = subprocess.run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=3)
    return [int(line) for line in checked.stdout.splitlines() if line.strip().isdigit()]


def in_group(pid, pgid):
    row = subprocess.run(["ps", "-p", str(pid), "-o", "pgid="], capture_output=True, text=True, timeout=3)
    return row.stdout.strip() == str(pgid)


@contextlib.contextmanager
def owned_provider(settings, *, endpoint, model, config):
    if settings.get("ownership", "external") != "owned":
        yield None
        return
    endpoint = endpoint_url(endpoint)
    from urllib.parse import urlsplit
    port = urlsplit(endpoint).port or 80
    command = settings.get("start_command")
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        raise PipelineError("INVALID_INPUT", "owned服务必须绑定start_command参数数组")
    policy = model_policy(settings.get("estimated_bytes", 0), settings.get("max_rss_bytes", 0), settings.get("timeout_seconds", 7200), config)
    with ExecutionLease() as lease:
        if listeners(port):
            raise PipelineError("OWNERSHIP_CONFLICT", "端口已被其他服务占用；不会接管或停止它", blocked=True)
        baseline = sample_resources()
        policy.preflight(baseline)
        directory = private_dir(runtime_dir() / "owned-services" / uuid4().hex)
        log_path = directory / "service.log"
        guard_result = directory / "service.guard.json"
        guard_owner = directory / "service.guard.owner.json"
        guarded = [sys.executable, str(Path(__file__).with_name("process_guard.py")),
            "--owner", str(os.getpid()), "--birth", process_birth(os.getpid()), "--fd", str(lease.fd),
            "--timeout", str(policy.timeout_seconds), "--max-rss", str(policy.max_rss_bytes), "--result", str(guard_result), "--", *command]
        stopped = threading.Event()
        state = {"error": None, "samples": []}
        started = time.monotonic()
        with log_path.open("ab") as log:
            log_path.chmod(0o600)
            process = subprocess.Popen(guarded, cwd=settings.get("start_cwd"), start_new_session=True,
                    pass_fds=(lease.fd,), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    env={**os.environ, "NEXTECHO_RUNTIME_DIR": str(lease.root), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        watcher = None
        job_id = os.environ.get("NEXTECHO_JOB_ID")
        try:
            if job_id:
                JobStore().register_owned(job_id, process.pid, process_birth(process.pid))
            def watch():
                violations = 0
                while not stopped.is_set():
                    try:
                        if current_cancelled():
                            raise PipelineError("CANCELLED", "任务已取消", stage="polish")
                        if time.monotonic() - started > policy.timeout_seconds:
                            raise PipelineError("STAGE_TIMEOUT", "专用模型服务超过阶段时限", stage="polish")
                        sample = sample_resources()
                        sample["worker_rss_bytes"] = group_rss(process.pid)
                        state["samples"].append(sample)
                        violations = violations + 1 if policy.violated(sample, baseline) else 0
                        if violations >= 2:
                            raise PipelineError("RESOURCE_LIMIT", "专用模型服务超过资源边界", stage="polish")
                    except Exception as exc:
                        state["error"] = exc
                        stop_owned(process, ownership_path=guard_owner)
                        return
                    if stopped.wait(float(settings.get("sample_seconds", 2))):
                        return
            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
            deadline = started + min(float(settings.get("startup_timeout_seconds", 120)), policy.timeout_seconds)
            while True:
                if state["error"]:
                    raise state["error"]
                if process.poll() is not None:
                    code = json.loads(guard_result.read_text()).get("error_code") if guard_result.is_file() else "PROVIDER_UNAVAILABLE"
                    raise PipelineError(code, "专用模型服务未成功启动，见本地service.log", blocked=True, stage="polish")
                bound = listeners(port)
                if bound and not all(in_group(pid, process.pid) for pid in bound):
                    raise PipelineError("OWNERSHIP_CONFLICT", "端口被其他进程取得，已停止自身启动尝试", blocked=True)
                if bound:
                    try:
                        tags = request_json(endpoint, settings.get("probe_path", "/v1/models"), timeout=1)
                        names = [r.get("id") for r in tags.get("data", [])] + [r.get("name") for r in tags.get("models", [])]
                        if model in names:
                            break
                    except PipelineError:
                        pass
                if time.monotonic() >= deadline:
                    raise PipelineError("STAGE_TIMEOUT", "专用服务启动超时", stage="polish")
                time.sleep(.2)
            try:
                yield {"owned_pgid": process.pid, "log_path": str(log_path), "resources_path": str(directory / "resources.json")}
            except Exception as exc:
                # A stopped service can make the next metadata/HTTP read fail.
                # Preserve the protection event as the actionable root cause.
                if state["error"] and getattr(exc, "code", "") != "CANCELLED":
                    raise state["error"] from exc
                raise
            if state["error"]:
                raise state["error"]
        finally:
            stopped.set()
            if watcher:
                watcher.join(timeout=6)
            stop_owned(process, ownership_path=guard_owner)
            # Only this process group was stopped. Its pending requests are
            # now provably ended; external service records remain untouched.
            ended = not group_running(process.pid)
            if ended:
                JobStore().provider_end(endpoint, completed=True)
            write_json(directory / "resources.json", {"baseline": baseline, "samples": state["samples"],
                "elapsed_seconds": time.monotonic() - started, "ownership": "owned", "model": model,
                "cleanup_complete": ended, "error_code": getattr(state["error"], "code", "")})
            if not ended:
                raise PipelineError("RESOURCE_BLOCKED", "所属模型进程组未确认结束；已保留占用和诊断", blocked=True)
