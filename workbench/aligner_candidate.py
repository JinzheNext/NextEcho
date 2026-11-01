"""本地外部原稿对齐候选；不注册 precise，也不自动下载模型。"""

from __future__ import annotations

import json
import math
import re
import sys
import wave
from pathlib import Path

from .contracts import PipelineError
from .io_utils import file_hash, value_hash, write_json
from .job_store import current_cancelled
from .resources import GIB, model_policy, run_owned

MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
REQUIRED_FILES = {
    "config.json", "model.safetensors", "processor_config.json", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja", "README.md",
}


def verify_local_model(root: Path, ledger: dict) -> dict:
    """读取全部指定字节；文件存在或可变的 main 名称不足以证明模型身份。"""
    if not root.is_dir():
        raise PipelineError("MODEL_MISSING", "本地对齐候选模型目录不存在", blocked=True)
    root = root.resolve(strict=True)
    files = ledger.get("files") if isinstance(ledger, dict) else None
    if (
        not isinstance(ledger, dict) or not isinstance(files, dict)
        or ledger.get("model_id") != MODEL_ID
        or ledger.get("license") != "apache-2.0"
        or not re.fullmatch(r"[0-9a-f]{40}", str(ledger.get("revision", "")))
        or set(files) != REQUIRED_FILES
    ):
        raise PipelineError("MODEL_MISSING", "对齐候选的固定版本/许可/文件台账无效", blocked=True)
    for name, record in files.items():
        path = root / name
        if (not isinstance(record, dict)
                or isinstance(record.get("bytes"), bool) or not isinstance(record.get("bytes"), int)
                or record["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
                or path.is_symlink() or not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or file_hash(path) != record.get("sha256")):
            raise PipelineError("MODEL_MISSING", "对齐候选模型文件不完整或已改变", blocked=True)
    return {"model_id": MODEL_ID, "revision": ledger["revision"],
            "license": ledger["license"], "ledger_sha256": value_hash(ledger)}


def validate_native_alignment(text: str, rows: list, duration_seconds: float,
                              origin_ms: int = 0) -> dict:
    """只保存原生区间；未能精确贴回的 token 保留诊断，不按字长补时间。"""
    if not isinstance(rows, list) or len(rows) > 30000:
        raise PipelineError("PROTOCOL_ERROR", "对齐结果数量无效")
    anchors, unmatched, cursor, previous_start = [], [], 0, -1.0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"]:
            raise PipelineError("PROTOCOL_ERROR", "对齐结果缺少原生 token")
        start, end = row.get("start_time"), row.get("end_time")
        if (any(isinstance(t, bool) or not isinstance(t, (int, float))
                or not math.isfinite(t) for t in (start, end))
                or start < 0 or end < start or end > duration_seconds
                or start < previous_start):
            raise PipelineError("PROTOCOL_ERROR", "对齐结果时间越界或逆序")
        previous_start = start
        token = row["text"]
        at = text.find(token, cursor)
        if at < 0:
            unmatched.append({"index": index, **row, "reason": "no_exact_source_span"})
            continue
        anchors.append({"text": token, "char_start": at, "char_end": at + len(token),
                        "native_start_ms": round(start * 1000) + origin_ms,
                        "native_end_ms": round(end * 1000) + origin_ms,
                        "native_duration_seconds": end - start})
        cursor = at + len(token)
    covered = {i for row in anchors for i in range(row["char_start"], row["char_end"])}
    unanchored = []
    left = None
    for i in range(len(text) + 1):
        missing = i < len(text) and i not in covered
        if missing and left is None:
            left = i
        elif not missing and left is not None:
            unanchored.append({"char_start": left, "char_end": i, "text": text[left:i]})
            left = None
    lexical = {i for i, char in enumerate(text) if char.isalnum()}
    positive, zero = set(), set()
    longest_zero_run = run_chars = 0
    previous_zero_time = None
    for anchor in anchors:
        positions = {i for i in range(anchor["char_start"], anchor["char_end"]) if text[i].isalnum()}
        if anchor["native_duration_seconds"] > 0:
            positive.update(positions)
            previous_zero_time, run_chars = None, 0
        else:
            zero.update(positions)
            if anchor["native_start_ms"] != previous_zero_time:
                run_chars = 0
            run_chars += len(positions)
            previous_zero_time = anchor["native_start_ms"]
            longest_zero_run = max(longest_zero_run, run_chars)
    # Text matching can be perfect while many predicted words share a single
    # zero-width time. Preserve the native predictions, expose the distinction;
    # never manufacture a duration or use coverage as identity/timing approval.
    return {"anchors": anchors, "unmatched_native_tokens": unmatched,
            "unanchored_source_spans": unanchored,
            "lexical_char_coverage": len(lexical & covered) / len(lexical) if lexical else None,
            "positive_duration_lexical_char_coverage": len(positive) / len(lexical) if lexical else None,
            "zero_duration_lexical_chars": len(zero),
            "max_same_time_zero_duration_native_lexical_chars": longest_zero_run,
            "timing_warnings": ["zero_duration_lexical_model_predictions"] if zero else [],
            "coverage_scope": "exact source-text matching only; not timing accuracy",
            "temporal_identity_evidence_verified": False,
            "source_text": text, "source_text_sha256": value_hash(text),
            "timing_precision": "native-forced-alignment-candidate",
            "precise_verified": False, "human_timepoint_gate": "pending"}


def run_qwen_alignment_candidate(*, python: Path, model_dir: Path, ledger: dict,
                                 audio: Path, text: str, destination: Path,
                                 origin_ms: int = 0, timeout: float = 240,
                                 attention: str = "sdpa",
                                 config=None,
                                 cancel=current_cancelled) -> dict:
    """一次冷载、CPU/4线程/batch1；共享所属组、资源采样及取消。"""
    if not isinstance(text, str) or not text.strip() or len(text) > 30000:
        raise PipelineError("INVALID_INPUT", "对齐原稿为空或超过候选预算")
    if isinstance(origin_ms, bool) or not isinstance(origin_ms, int) or origin_ms < 0:
        raise PipelineError("INVALID_INPUT", "音频全局起点无效")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0 or timeout > 900):
        raise PipelineError("INVALID_INPUT", "对齐候选时限必须在900秒内")
    if attention not in {"eager", "sdpa"}:
        raise PipelineError("INVALID_INPUT", "候选只接受显式CPU注意力实现")
    if cancel():
        raise PipelineError("CANCELLED", "对齐候选已取消")
    audio = audio.resolve(strict=True)
    with wave.open(str(audio), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise PipelineError("INVALID_INPUT", "候选只接受mono16k PCM16 WAV")
    if not 0 < duration <= 300:
        raise PipelineError("INVALID_INPUT", "Qwen候选每次音频不得超过5分钟")
    model = verify_local_model(model_dir, ledger)
    # 实验预算，不是已校准生产档案；60秒探针通过后才考虑5分钟。
    policy = model_policy(8 * GIB, 10 * GIB, timeout, config)
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    request = {"schema_version": 1, "audio": str(audio), "audio_sha256": file_hash(audio),
               "text": text, "text_sha256": value_hash(text), "duration_seconds": duration,
               "origin_ms": origin_ms, "model_dir": str(model_dir.resolve()), "model": model,
               "output": str((destination / "native.json").resolve()),
               "threads": 4, "device": "cpu", "dtype": "float32", "attention": attention}
    worker = Path(__file__).with_name("aligner_worker.py")
    request["worker_sha256"] = file_hash(worker)
    controller = Path(__file__)
    request["controller_sha256"] = file_hash(controller)
    # 每个真实尝试保留对应入口字节；后续修改不能抹掉失败实现。
    (destination / "worker.source.py").write_bytes(worker.read_bytes())
    (destination / "controller.source.py").write_bytes(controller.read_bytes())
    request_path = destination / "request.json"
    write_json(request_path, request)
    log = destination / "worker.log"
    resources = run_owned([str(python), str(worker), str(request_path.resolve())],
                          log_path=log, policy=policy, cancel=cancel)
    if file_hash(audio) != request["audio_sha256"]:
        raise PipelineError("CACHE_MISMATCH", "对齐期间源音频发生变化")
    native_path = destination / "native.json"
    if not native_path.is_file() or native_path.stat().st_size > 4 * 1024**2:
        raise PipelineError("PROTOCOL_ERROR", "对齐结果缺失或超过预算")
    native = json.loads(native_path.read_text())
    if (native.get("schema_version") != 1 or native.get("audio_sha256") != request["audio_sha256"]
            or native.get("text_sha256") != request["text_sha256"]
            or native.get("model") != model):
        raise PipelineError("PROTOCOL_ERROR", "对齐结果与冻结输入不符")
    result = validate_native_alignment(text, native.get("timestamps"), duration, origin_ms)
    result.update({"schema_version": 1, "model": model, "resources": resources,
                   "runtime": native.get("runtime"), "timings": native.get("timings"),
                   "audio_sha256": request["audio_sha256"], "origin_ms": origin_ms,
                   "duration_seconds": duration, "request_sha256": file_hash(request_path),
                   "native_sha256": file_hash(native_path), "production_default_changed": False,
                   "controller_python": sys.version.split()[0]})
    write_json(destination / "alignment.json", result)
    return result
