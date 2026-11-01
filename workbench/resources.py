"""Process-wide execution leases and bounded owned subprocesses.

The OS lock, rather than a heartbeat TTL, owns mutual exclusion. Workers may
inherit its descriptor so killing a supervisor cannot admit a second model.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import math
import functools
import os
import signal
import subprocess
import sys
import threading
import time
from time import monotonic as _monotonic
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .contracts import PipelineError
from .io_utils import write_json

_LOCAL = threading.local()
_CONFIG = contextvars.ContextVar("nextecho_resources_config", default=None)
GIB = 1024 ** 3
MINIMUM_RESERVE_BYTES = 3 * GIB


def _counter(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def runtime_dir() -> Path:
    from .config import user_runtime_path
    return user_runtime_path().resolve()


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def process_birth(pid: int) -> str:
    result = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "command="], capture_output=True, text=True, timeout=3)
    return result.stdout.strip() if result.returncode == 0 else ""


def process_alive(pid: int, birth: str) -> bool:
    return bool(birth and process_birth(pid) == birth)


class ExecutionLease:
    def __init__(self, root: Path | None = None, *, timeout: float | None = None, on_wait: Callable | None = None):
        self.root = root or runtime_dir()
        self.timeout, self.on_wait = timeout, on_wait
        self.fd: int | None = None
        self.nested = False
        self.inherited = False

    def __enter__(self):
        if os.name != "posix":
            raise PipelineError("PLATFORM_UNSUPPORTED", "后台资源控制当前支持POSIX运行环境", blocked=True)
        active = getattr(_LOCAL, "lease", None)
        if active is not None and active.root == self.root:
            self.fd, self.nested = active.fd, True
            return self
        import fcntl

        private_dir(self.root)
        inherited = os.environ.get("NEXTECHO_INHERITED_LEASE_FD", "")
        if inherited.isdigit():
            try:
                fd = int(inherited)
                actual, expected = os.fstat(fd), (self.root / "heavy.lock").stat()
                if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.fd, self.nested, self.inherited = fd, True, True
                    _LOCAL.lease = self
                    return self
            except (OSError, ValueError):
                pass
        self.fd = os.open(self.root / "heavy.lock", os.O_CREAT | os.O_RDWR, 0o600)
        started = _monotonic()
        try:
            while True:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if (self.root / "jobs.sqlite3").is_file():
                        from .job_store import JobStore
                        JobStore(self.root).ensure_providers_idle()
                    _LOCAL.lease = self
                    return self
                except BlockingIOError:
                    if self.timeout is not None and _monotonic() - started >= self.timeout:
                        raise TimeoutError("本地模型正忙，等待执行租约超时")
                    if self.on_wait:
                        self.on_wait()
                    time.sleep(0.1)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def __exit__(self, *exc):
        if self.inherited:
            _LOCAL.lease = None
        if not self.nested and self.fd is not None:
            # Do not explicitly LOCK_UN: an inherited descriptor may still
            # represent a live model after supervisor death.
            os.close(self.fd)
            self.fd = None
            _LOCAL.lease = None


def serialized_task(function):
    @functools.wraps(function)
    def invoke(*args, **kwargs):
        # Keep legacy non-POSIX execution behavior until its native lease
        # implementation is validated. New headless jobs reject that platform.
        callback = kwargs.get("progress_callback")
        manager = ExecutionLease(on_wait=(lambda: callback({"stage": "queued", "status": "queued", "message": "等待本地模型执行资格"})) if callback else None) if os.name == "posix" else contextlib.nullcontext()
        with manager:
            token = _CONFIG.set(kwargs.get("config") or _CONFIG.get())
            try:
                return function(*args, **kwargs)
            finally:
                _CONFIG.reset(token)
    return invoke


@dataclass(frozen=True)
class ResourcePolicy:
    estimated_bytes: int
    max_rss_bytes: int
    timeout_seconds: float
    reserve_bytes: int = 3 * GIB
    max_swap_delta_bytes: int = GIB
    max_pressure: str = "warn"

    def preflight(self, sample: dict) -> None:
        if min(self.estimated_bytes, self.max_rss_bytes, self.timeout_seconds) <= 0:
            raise PipelineError("RESOURCE_BLOCKED", "模型资源档案尚未校准", blocked=True)
        available = sample.get("available_bytes")
        pressure = sample.get("pressure", "unknown")
        ranks = {"normal": 0, "warn": 1, "critical": 2}
        if not _counter(available) or pressure not in ranks or not _counter(sample.get("swap_used_bytes")):
            raise PipelineError("RESOURCE_BLOCKED", "无法测量当前可用内存/压力", blocked=True)
        if available < self.estimated_bytes + self.reserve_bytes or ranks[pressure] > ranks[self.max_pressure]:
            raise PipelineError("RESOURCE_BLOCKED", "内存不足以满足新增工作集及预留，请释放本机内存后重试", blocked=True)

    def violated(self, sample, baseline):
        if not all(_counter(value) for value in (sample.get("available_bytes"), sample.get("swap_used_bytes"),
                baseline.get("swap_used_bytes"), sample.get("worker_rss_bytes"))):
            return True
        delta = None if sample.get("swap_used_bytes") is None or baseline.get("swap_used_bytes") is None else sample["swap_used_bytes"] - baseline["swap_used_bytes"]
        return (sample.get("pressure") not in {"normal", "warn"} or
                sample.get("available_bytes") is None or delta is None or
                sample["available_bytes"] < self.reserve_bytes or
                (self.max_pressure == "normal" and sample.get("pressure") != "normal") or
                delta > self.max_swap_delta_bytes or sample.get("worker_rss_bytes", 0) > self.max_rss_bytes)


def model_policy(estimated_bytes, max_rss_bytes, timeout_seconds, config=None):
    from .config import load_config
    cfg = config or _CONFIG.get() or load_config()
    hardware = cfg.get("hardware", {})
    pressure = hardware.get("memory_pressure_max", "warn")
    if pressure not in {"normal", "warn"}:
        raise PipelineError("INVALID_INPUT", "memory_pressure_max必须为normal或warn")
    try:
        values = [float(v) for v in (estimated_bytes, max_rss_bytes, timeout_seconds,
                   hardware.get("reserve_memory_gb", 3), hardware.get("max_swap_delta_gb", 1))]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("nonfinite or negative resource budget")
        estimate, maximum, deadline, reserve, swap = values
        reserve_bytes = max(MINIMUM_RESERVE_BYTES, int(reserve * GIB))
        swap_bytes = int(swap * GIB)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineError("INVALID_INPUT", "资源预算必须为有限非负数") from exc
    return ResourcePolicy(int(estimate), int(maximum), deadline,
             reserve_bytes=reserve_bytes,
             max_swap_delta_bytes=swap_bytes, max_pressure=pressure)


def sample_resources() -> dict:
    from .asr import memory_snapshot
    return memory_snapshot().to_dict()


def group_rss(pgid: int) -> int:
    result = subprocess.run(["ps", "-axo", "pgid=,rss="], capture_output=True, text=True, timeout=3)
    if result.returncode:
        raise PipelineError("RESOURCE_BLOCKED", "无法测量工作进程内存", blocked=True)
    return sum(int(fields[1]) * 1024 for row in result.stdout.splitlines()
               if len(fields := row.split()) == 2 and int(fields[0]) == pgid)


def group_running(pgid: int) -> bool:
    checked = subprocess.run(["ps", "-axo", "pgid=,stat="], capture_output=True, text=True, timeout=3)
    if checked.returncode:
        raise PipelineError("RESOURCE_BLOCKED", "无法核验所属进程组是否已结束", blocked=True)
    return any(int(parts[0]) == pgid and not parts[1].startswith("Z")
               for row in checked.stdout.splitlines() if len(parts := row.split()) == 2)


def stop_owned(process: subprocess.Popen, *, ownership_path: Path | None = None) -> None:
    if process.poll() is not None:
        # A killed guardian cannot run its finally block. Its child identity
        # proves the still-live process group belongs to this invocation.
        if ownership_path and ownership_path.is_file():
            try:
                child = json.loads(ownership_path.read_text())
                if process_alive(child["pid"], child["birth"]):
                    members = subprocess.run(["ps", "-axo", "pid=,pgid="], capture_output=True, text=True, timeout=3)
                    if members.returncode:
                        raise PipelineError("RESOURCE_BLOCKED", "无法核验所属进程组成员", blocked=True)
                    identities = {int(f[0]): process_birth(int(f[0])) for row in members.stdout.splitlines()
                                  if len(f := row.split()) == 2 and int(f[1]) == child["pgid"]}
                    os.killpg(child["pgid"], signal.SIGTERM)
                    until = _monotonic() + 3
                    while group_running(child["pgid"]) and _monotonic() < until:
                        time.sleep(.05)
                    # A surviving identified member anchors the original group
                    # even if its model leader has already exited on TERM.
                    if group_running(child["pgid"]) and any(process_alive(pid, birth) for pid, birth in identities.items()):
                        os.killpg(child["pgid"], signal.SIGKILL)
                    if group_running(child["pgid"]):
                        until = _monotonic() + 2
                        while group_running(child["pgid"]) and _monotonic() < until:
                            time.sleep(.05)
                        if group_running(child["pgid"]):
                            raise PipelineError("RESOURCE_BLOCKED", "所属进程组未确认清理完成，保留执行占用", blocked=True)
            except (OSError, ValueError, KeyError):
                pass
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # The live guardian is the final lease holder. Kill its members, not
        # that guardian, until it can acknowledge complete group cleanup.
        members = subprocess.run(["ps", "-axo", "pid=,pgid="], capture_output=True, text=True, timeout=3)
        if members.returncode:
            raise PipelineError("RESOURCE_BLOCKED", "所属进程组清理无法核验，guardian继续持有租约", blocked=True)
        for row in members.stdout.splitlines():
            fields = row.split()
            if len(fields) == 2 and int(fields[1]) == process.pid and int(fields[0]) != process.pid:
                try:
                    os.kill(int(fields[0]), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise PipelineError("RESOURCE_BLOCKED", "所属进程组尚未确认清理，guardian继续持有租约", blocked=True) from exc
    except ProcessLookupError:
        process.wait()


def run_owned(command: list[str], *, log_path: Path, policy: ResourcePolicy,
              cancel: Callable[[], bool] = lambda: False, sampler: Callable = sample_resources,
              interval: float = 2, on_spawn: Callable | None = None, on_sample: Callable | None = None) -> dict:
    baseline = {}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    trace = []
    with ExecutionLease(on_wait=lambda: _check_cancel(cancel)) as lease, log_path.open("ab") as log:
        log_path.chmod(0o600)
        # Recheck after waiting: memory can change while queued.
        baseline = sampler()
        policy.preflight(baseline)
        guarded = [sys.executable, str(Path(__file__).with_name("process_guard.py")),
                   "--owner", str(os.getpid()), "--birth", process_birth(os.getpid()), "--fd", str(lease.fd),
                   "--timeout", str(policy.timeout_seconds), "--max-rss", str(policy.max_rss_bytes),
                   "--result", str(log_path.with_suffix(".guard.json")), "--", *command]
        log_path.with_suffix(".guard.json").unlink(missing_ok=True)
        ownership_path = log_path.with_suffix(".guard.owner.json")
        ownership_path.unlink(missing_ok=True)
        process = subprocess.Popen(guarded, stdout=log, stderr=log, start_new_session=True,
                                   pass_fds=(lease.fd,), env={**os.environ, "NEXTECHO_SUPERVISOR_PID": str(os.getpid()),
                                   "NEXTECHO_RUNTIME_DIR": str(lease.root)})
        started = _monotonic()
        violations = 0
        try:
            job_id = os.environ.get("NEXTECHO_JOB_ID")
            if job_id:
                from .job_store import JobStore
                JobStore().register_owned(job_id, process.pid, process_birth(process.pid))
            if on_spawn:
                on_spawn(process.pid, process_birth(process.pid))
            while process.poll() is None:
                _check_cancel(cancel)
                if _monotonic() - started > policy.timeout_seconds:
                    raise PipelineError("STAGE_TIMEOUT", "模型阶段超过时间上限")
                sample = sampler()
                sample["worker_rss_bytes"] = group_rss(process.pid)
                trace.append(sample)
                if on_sample:
                    on_sample(sample)
                bad = policy.violated(sample, baseline)
                violations = violations + 1 if bad else 0
                if violations >= 2:
                    raise PipelineError("RESOURCE_LIMIT", "模型阶段超过资源保护边界")
                try:
                    process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                guard_result = log_path.with_suffix(".guard.json")
                if guard_result.is_file():
                    code = json.loads(guard_result.read_text()).get("error_code", "WORKER_FAILED")
                    raise PipelineError(code, "所属工作进程被资源/时间保护中止")
                raise PipelineError("WORKER_FAILED", f"模型worker退出码{process.returncode}，详细信息见本地日志")
            return {"elapsed_seconds": _monotonic() - started, "samples": len(trace),
                    "peak_worker_rss_bytes": max((s["worker_rss_bytes"] for s in trace), default=None)}
        finally:
            try:
                stop_owned(process, ownership_path=ownership_path)
            finally:
                write_json(log_path.with_suffix(".resources.json"), {"baseline": baseline, "samples": trace,
                    "effective_policy": {"estimated_bytes": policy.estimated_bytes, "max_rss_bytes": policy.max_rss_bytes,
                        "reserve_bytes": policy.reserve_bytes, "max_swap_delta_bytes": policy.max_swap_delta_bytes,
                        "timeout_seconds": policy.timeout_seconds}})


def _check_cancel(cancel: Callable[[], bool]) -> None:
    if cancel():
        raise PipelineError("CANCELLED", "任务已取消")
