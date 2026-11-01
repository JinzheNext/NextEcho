"""Pinned atomic model preparation with finite, owner-aware download locking."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from .contracts import PipelineError
from .io_utils import write_json
from .job_store import current_cancelled
from .media_commands import run_media_command
from .resources import ExecutionLease, process_birth
from .whisper_models import model_candidates, model_spec, validate_whisper_model


def _cancel():
    if current_cancelled():
        raise PipelineError("CANCELLED", "模型准备已取消", stage="model-download")


def _inode(path):
    s = path.lstat()
    return s.st_dev, s.st_ino


def _birth_hash(pid):
    birth = process_birth(pid) if os.name == "posix" else ""
    return hashlib.sha256(birth.encode()).hexdigest() if birth else ""


def _owner_dead(owner):
    if os.name != "posix":
        return False  # legacy Windows: finite wait, no unverified stale-lock theft
    actual = _birth_hash(owner["pid"])
    if actual:
        return actual != owner["birth_sha256"]
    try:
        os.kill(owner["pid"], 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass
    return False


class ModelDownloadLock:
    """Persistent OS guard plus old-version O_EXCL sentinel; never unlink guard."""
    def __init__(self, root: Path, name: str, *, timeout: float = 60):
        model_spec(name)
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise PipelineError("INVALID_INPUT", "模型锁时限必须为有限正数")
        self.root, self.name, self.timeout = root, name, timeout
        self.path = root / f".ggml-{name}.lock"
        self.guard = root / f".ggml-{name}.download.guard"
        self.fd = None
        self.owned = None

    def _wait(self, deadline):
        _cancel()
        if time.monotonic() >= deadline:
            raise PipelineError("MODEL_BUSY", "Whisper模型准备正忙；有限等待已结束，请稍后重试", blocked=True, stage="model-download")
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        deadline = time.monotonic() + self.timeout
        try:
            _cancel()
            if self.guard.is_symlink():
                raise PipelineError("MODEL_LOCK_INVALID", "模型锁guard不是常规文件", blocked=True)
            self.fd = os.open(self.guard, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), 0o600)
            if not stat.S_ISREG(os.fstat(self.fd).st_mode):
                raise PipelineError("MODEL_LOCK_INVALID", "模型锁guard不是常规文件", blocked=True)
            if os.name == "posix":
                import fcntl
                def acquire():
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                if os.fstat(self.fd).st_size == 0:
                    os.write(self.fd, b"\0")
                def acquire():
                    os.lseek(self.fd, 0, 0)
                    msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            while True:
                try:
                    acquire()
                    break
                except (BlockingIOError, PermissionError):
                    self._wait(deadline)
            birth = _birth_hash(os.getpid())
            if os.name == "posix" and not birth:
                raise PipelineError("MODEL_LOCK_INVALID", "无法核验下载锁启动身份", blocked=True)
            owner = {"schema_version": 2, "pid": os.getpid(), "birth_sha256": birth,
                     "nonce": uuid4().hex, "name": self.name}
            # Write fully before atomic O_EXCL publication; a crash cannot leave
            # a new empty sentinel which would be indistinguishable from legacy.
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".model-lock-", delete=False) as temporary:
                temp_path = Path(temporary.name)
                temporary.write(json.dumps(owner).encode())
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                while True:
                    _cancel()
                    try:
                        os.link(temp_path, self.path)
                        self.owned = _inode(self.path)
                        return self
                    except FileExistsError:
                        s = self.path.lstat()
                        identity = s.st_dev, s.st_ino
                        try:
                            if not stat.S_ISREG(s.st_mode) or not 0 < s.st_size <= 4096:
                                raise ValueError("legacy/unrecognized model lock")
                            previous = json.loads(self.path.read_text())
                            if (previous.get("schema_version") != 2 or previous.get("name") != self.name
                                    or type(previous.get("pid")) is not int or previous["pid"] <= 0
                                    or not isinstance(previous.get("birth_sha256"), str)
                                    or not re.fullmatch("[0-9a-f]{64}", previous["birth_sha256"])
                                    or not isinstance(previous.get("nonce"), str)
                                    or not re.fullmatch("[0-9a-f]{32}", previous["nonce"])):
                                raise ValueError("unrecognized model lock owner")
                        except (ValueError, TypeError, AttributeError, UnicodeError) as exc:
                            raise PipelineError("MODEL_LOCK_LEGACY", "旧式或未知下载锁无法确认归属；请确认无旧下载进程后手动移走锁文件", blocked=True, stage="model-download") from exc
                        except FileNotFoundError:
                            continue  # an older downloader completed while inspected
                        if _owner_dead(previous):
                            if _inode(self.path) != identity:
                                continue
                            self.path.unlink()
                            continue
                        self._wait(deadline)
            finally:
                temp_path.unlink(missing_ok=True)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        try:
            if self.owned and self.path.exists() and _inode(self.path) == self.owned:
                self.path.unlink()
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


def _download_file(spec, destination):
    with tempfile.TemporaryDirectory(prefix=".model-request-", dir=destination.parent) as folder:
        request = Path(folder) / "request.json"
        write_json(request, {"spec": spec, "destination": str(destination)})
        try:
            run_media_command([sys.executable, str(Path(__file__).with_name("whisper_download_worker.py")), str(request)],
                              timeout=900, stage="model-download")
        except subprocess.CalledProcessError as exc:
            raise PipelineError("MODEL_DOWNLOAD_FAILED", "Whisper固定版本下载失败或大小/hash不符；旧权重保持，请检查网络后重试", blocked=True, stage="model-download") from exc


def prepare_model(name: str, root: Path, other_roots: list[Path], *, lock_timeout=60) -> Path:
    spec = model_spec(name)
    lock = ModelDownloadLock(root.expanduser().resolve(), name, timeout=lock_timeout)
    root = root.expanduser().resolve()
    target = root / spec["filename"]
    for existing in model_candidates(name, [root, *other_roots]):
        if existing.exists() or existing.is_symlink():
            try:
                return Path(validate_whisper_model(existing, name=name)["path"])
            except PipelineError:
                if existing != target or existing.is_symlink() or not existing.is_file():
                    raise  # no silent environment override, symlink or directory repair
                break
    # Resource queue has its own finite budget, separate from transport's 900s.
    lease = ExecutionLease(timeout=lock_timeout, on_wait=_cancel) if os.name == "posix" else contextlib.nullcontext()
    try:
        _cancel()
        with lease, lock:
            if target.exists():
                try:
                    return Path(validate_whisper_model(target, name=name)["path"])
                except PipelineError:
                    if target.is_symlink() or not target.is_file():
                        raise
            fd, raw = tempfile.mkstemp(prefix=f".ggml-{name}.", suffix=".part", dir=root)
            os.close(fd)
            part = Path(raw)
            try:
                _download_file(spec, part)
                identity = validate_whisper_model(part, name=name)
                _cancel()
                if target.exists() or target.is_symlink():
                    if target.is_symlink() or not target.is_file():
                        raise PipelineError("MODEL_INVALID", "下载目标已变为非普通文件，拒绝覆盖", blocked=True)
                    # Preserve invalid old bytes only once a fully valid replacement exists.
                    os.link(target, root / f".ggml-{name}.invalid-{uuid4().hex}.bin")
                os.replace(part, target)
                identity["path"] = str(target)
                write_json(root / f".ggml-{name}.identity.json", identity)
                if os.name == "posix":
                    directory = os.open(root, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                return target
            finally:
                part.unlink(missing_ok=True)
    except TimeoutError as exc:
        raise PipelineError("MODEL_BUSY", "本地执行租约等待超时，未启动模型下载", blocked=True, stage="model-download") from exc
