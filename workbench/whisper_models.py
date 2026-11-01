"""固定官方权重身份与完整GGML结构检查；不把格式有效当作质量放行。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from pathlib import Path

from .contracts import PipelineError

REGISTRY = Path(__file__).with_name("data") / "whisper-models.json"
MAX_MODEL_BYTES = 16 * 1024**3
# 每块元素数/字节数；不支持的未来GGML类型明确阻断，不猜大小。
TENSOR_TYPES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
                6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
                10: (256, 84), 11: (256, 110), 12: (256, 144),
                13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2)}


def registry() -> dict:
    data = json.loads(REGISTRY.read_text())
    if data.get("schema_version") != 1 or data.get("repo_id") != "ggerganov/whisper.cpp":
        raise PipelineError("PROTOCOL_ERROR", "Whisper固定权重索引无效", blocked=True)
    return data


def model_spec(name: str) -> dict:
    data = registry()
    if name not in data["models"]:
        raise PipelineError("INVALID_INPUT", "未知Whisper模型名称；本地文件请给出实际路径")
    row = data["models"][name]
    return {**row, "name": name, "repo_id": data["repo_id"], "revision": data["revision"],
            "license_declaration": data["license_declaration"],
            "url": f"https://huggingface.co/{data['repo_id']}/resolve/{data['revision']}/{row['filename']}"}


def expected_tensors(h: tuple) -> dict[str, tuple]:
    vocab, ac, a, _, al, tc, t, _, tl, mels, _ = h
    shapes = {"encoder.positional_embedding": (a, ac), "encoder.conv1.weight": (3, mels, a),
        "encoder.conv1.bias": (1, a), "encoder.conv2.weight": (3, a, a), "encoder.conv2.bias": (1, a),
        "encoder.ln_post.weight": (a,), "encoder.ln_post.bias": (a,),
        "decoder.positional_embedding": (t, tc), "decoder.token_embedding.weight": (t, vocab),
        "decoder.ln.weight": (t,), "decoder.ln.bias": (t,)}
    for kind, width, layers in (("encoder", a, al), ("decoder", t, tl)):
        for i in range(layers):
            prefix = f"{kind}.blocks.{i}."
            block = {"attn_ln.weight": (width,), "attn_ln.bias": (width,),
                "mlp_ln.weight": (width,), "mlp_ln.bias": (width,),
                "mlp.0.weight": (width, 4 * width), "mlp.0.bias": (4 * width,),
                "mlp.2.weight": (4 * width, width), "mlp.2.bias": (width,)}
            for attention in (["attn", "cross_attn"] if kind == "decoder" else ["attn"]):
                for projection in ("query", "key", "value", "out"):
                    block[f"{attention}.{projection}.weight"] = (width, width)
                    if projection != "key":
                        block[f"{attention}.{projection}.bias"] = (width,)
                if attention == "cross_attn":
                    block.update({"cross_attn_ln.weight": (width,), "cross_attn_ln.bias": (width,)})
            shapes.update({prefix + name: shape for name, shape in block.items()})
    return shapes


def _structure(f, size: int) -> dict:
    def read(n):
        if n < 0 or f.tell() + n > size:
            raise ValueError("truncated GGML")
        value = f.read(n)
        if len(value) != n:
            raise ValueError("short GGML read")
        return value
    def integer():
        return struct.unpack("<i", read(4))[0]
    def skip(n):
        if n < 0 or f.tell() + n > size:
            raise ValueError("truncated GGML payload")
        f.seek(n, 1)
    if integer() != 0x67676D6C:
        raise ValueError("not Whisper GGML magic")
    h = struct.unpack("<11i", read(44))
    vocab, ac, a, ah, al, tc, t, th, tl, mels, ftype = h
    if (not 1024 <= vocab <= 1000000 or not 1 <= ac <= 1000000 or not 1 <= tc <= 1000000
            or not 1 <= a <= 8192 or a != t or not 1 <= ah <= 128 or not 1 <= th <= 128
            or a % ah or t % th or not 1 <= al <= 128 or not 1 <= tl <= 128
            or mels not in {80, 128} or not 0 <= ftype // 1000 <= 2
            or ftype % 1000 not in {0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 24}):
        raise ValueError("invalid/unsupported Whisper hyperparameters")
    nmel, nfft = integer(), integer()
    if nmel != mels or not 1 <= nfft <= 100000:
        raise ValueError("invalid mel filters")
    skip(nmel * nfft * 4)
    entries = integer()
    if not 1 <= entries <= vocab:
        raise ValueError("invalid vocabulary count")
    for _ in range(entries):
        length = integer()
        if not 0 <= length <= 1024**2:
            raise ValueError("invalid vocabulary entry")
        skip(length)
    expected = expected_tensors(h)
    found = set()
    while f.tell() < size:
        dims, length, dtype = integer(), integer(), integer()
        if not 1 <= dims <= 4 or not 1 <= length <= 1024 or dtype not in TENSOR_TYPES:
            raise ValueError("invalid/unsupported GGML tensor header")
        shape = tuple(integer() for _ in range(dims))
        if any(n <= 0 for n in shape):
            raise ValueError("invalid GGML tensor dimensions")
        name = read(length).decode("utf-8")
        padded = shape + (1,) * (4 - len(shape))
        target = expected.get(name)
        if name in found or target is None or padded != target + (1,) * (4 - len(target)):
            raise ValueError("missing/duplicate/incorrect Whisper tensor shape")
        block, width = TENSOR_TYPES[dtype]
        if shape[0] % block:
            raise ValueError("invalid quantized row size")
        skip(math.prod(shape) // block * width)
        found.add(name)
    if found != expected.keys():
        raise ValueError("incomplete Whisper tensor set")
    return {"tensor_count": len(found), "vocabulary_entries": entries,
            "audio_layers": al, "text_layers": tl, "ftype": ftype, "format": "whisper-ggml"}


def validate_whisper_model(path: Path, *, name: str | None = None) -> dict:
    """命名模型必须匹配官方hash；显式自带文件完整解析并记录自身hash。"""
    expected = model_spec(name) if name is not None else None
    try:
        resolved = path.expanduser().resolve(strict=True)
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError("not a regular GGML model")
        fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as f:
            before = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode) or not 64 <= before.st_size <= MAX_MODEL_BYTES:
                raise ValueError("not a bounded regular GGML model")
            if expected and before.st_size != expected["bytes"]:
                raise ValueError("official model size mismatch")
            structure = _structure(f, before.st_size)
            f.seek(0)
            sha = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024**2), b""):
                sha.update(chunk)
            digest = sha.hexdigest()
            after = os.fstat(f.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("model changed while validating")
        current = resolved.stat()
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("model path replaced while validating")
        if expected and digest != expected["sha256"]:
            raise ValueError("official model SHA256 mismatch")
        if expected is None:
            for known_name, row in registry()["models"].items():
                if before.st_size == row["bytes"] and digest == row["sha256"]:
                    expected = model_spec(known_name)
                    break
        return {"path": str(resolved), "bytes": before.st_size, "sha256": digest, **structure,
                "identity": "official-pinned" if expected else "user-supplied",
                "repo_id": expected["repo_id"] if expected else "", "revision": expected["revision"] if expected else "",
                "license_declaration": expected["license_declaration"] if expected else "unknown",
                "quality_verified": False}
    except (OSError, ValueError, UnicodeError, struct.error) as exc:
        raise PipelineError("MODEL_INVALID", "Whisper权重不是完整有效文件或与固定身份不符", blocked=True, stage="model") from exc


def model_candidates(name: str, roots: list[Path]) -> list[Path]:
    spec = model_spec(name)  # also rejects path traversal / unknown names before I/O
    env_roots = [Path(os.environ[e]).expanduser() for e in ("TRANSCRIBE_MODEL_DIR", "WHISPER_MODEL_DIR") if os.environ.get(e)]
    return list(dict.fromkeys(root.expanduser().resolve() / spec["filename"] for root in [*env_roots, *roots]))


def find_model(name: str, roots: list[Path]) -> Path | None:
    for path in model_candidates(name, roots):
        if path.exists() or path.is_symlink():
            # An invalid higher-priority selection is not silently bypassed.
            return Path(validate_whisper_model(path, name=name)["path"])
    return None


def cached_model(reference: str, roots: list[Path]) -> Path | None:
    candidate = Path(reference).expanduser()
    if candidate.exists() or candidate.is_symlink():
        return Path(validate_whisper_model(candidate)["path"])
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.suffix or reference.startswith("~"):
        raise PipelineError("MODEL_MISSING", "指定的Whisper本地模型文件不存在", blocked=True, stage="model")
    return find_model(reference, roots)
