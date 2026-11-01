"""Persist a single user intent before dispatch; never deduplicate by media URL."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

MAX_INTENT_BYTES = 64 * 1024


def signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def mutates(arguments):
    return arguments[:1] != ["jobs"] or arguments[1:2] == ["resume"]


def validate_binding(binding):
    if (not isinstance(binding, dict) or binding.get("schema_version") != 1
            or not isinstance(binding.get("python"), str) or not binding["python"]
            or not all(isinstance(binding.get(k, ""), str) for k in ("config", "runtime_dir"))):
        raise ValueError("运行时绑定协议不兼容，请重新绑定")


def create_intent(binding, arguments, binding_file: Path, target: Path | None = None):
    if not arguments or not mutates(arguments):
        raise ValueError("请求文件用于转写或jobs resume；查询/等待/取消可直接调用")
    request_id = str(uuid4())
    data = {"schema_version": 1, "artifact_kind": "nextecho-request-intent", "request_id": request_id,
            "binding": binding, "arguments": arguments, "caller_cwd": str(Path.cwd().resolve())}
    data["integrity_sha256"] = signature(data)
    encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
    if len(encoded) > MAX_INTENT_BYTES:
        raise ValueError("请求过长；请减少输入或参数")
    target = (target or binding_file.parent / "requests" / f"{request_id}.json").expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, scratch = tempfile.mkstemp(prefix=".intent-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        # Exclusive publication: another intent must never be overwritten.
        os.link(scratch, target)
        if os.name == "posix":
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(scratch).unlink(missing_ok=True)
    return target.resolve(), data


def load_intent(path: Path):
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INTENT_BYTES:
        raise ValueError("请求文件缺失、不是普通文件或超过大小上限")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("请求文件不是对象")
    original = data.get("integrity_sha256")
    if signature({k: v for k, v in data.items() if k != "integrity_sha256"}) != original:
        raise ValueError("请求文件已改变；需要新意图时另建请求，不改旧文件")
    if (data.get("schema_version") != 1 or data.get("artifact_kind") != "nextecho-request-intent"
            or not isinstance(data.get("request_id"), str)
            or str(UUID(data["request_id"])) != data["request_id"]
            or not isinstance(data.get("arguments"), list) or not data["arguments"]
            or not all(isinstance(a, str) and "\0" not in a for a in data["arguments"])
            or not mutates(data["arguments"]) or not isinstance(data.get("caller_cwd"), str)
            or not Path(data["caller_cwd"]).is_absolute()
            or not isinstance(data.get("binding"), dict)):
        raise ValueError("请求文件协议无效")
    validate_binding(data["binding"])
    return path.resolve(), data


def has_key(arguments):
    return any(a == "--idempotency-key" or a.startswith("--idempotency-key=") for a in arguments)
