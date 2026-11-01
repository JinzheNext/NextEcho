"""Private atomic files shared by jobs and stage checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, default=str))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def redact(value: str) -> str:
    def clean(match):
        try:
            u = urlsplit(match.group())
        except ValueError:
            return "[REDACTED_URL]"
        host = u.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        try:
            if u.port:
                host += f":{u.port}"
        except ValueError:
            pass
        return urlunsplit((u.scheme, host, u.path, "", ""))
    value = re.sub(r"https?://[^\s\"'<>]+", clean, value)
    value = re.sub(r"(?i)(authorization\s*[:=]\s*)(?:Bearer|Basic)\s+[^\s,;\"']+", r"\1[REDACTED]", value)
    value = re.sub(r"(?im)(\bcookie\s*:\s*)[^\r\n]+", r"\1[REDACTED]", value)
    value = re.sub(r'''(?i)(["']?(?:token|cookie|authorization|password|api[_-]?key)["']?\s*[:=]\s*)(["'])(.*?)\2''', r"\1[REDACTED]", value)
    return re.sub(r"(?i)(token|cookie|authorization|password|api[_-]?key)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", value)


def redact_structure(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if re.fullmatch(r"(?i)(?:access_token|refresh_token|hf_token|token|cookie|authorization|password|api[_-]?key)", str(k))
                else v if isinstance(v, str) and Path(v).is_absolute() and (str(k).endswith("_path") or k in {"path", "run_dir", "output_dir"})
                else redact_structure(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_structure(v) for v in value]
    return redact(value) if isinstance(value, str) else value
