"""Audited compatibility text edits; explicit commands retain owned limits."""

from __future__ import annotations

import copy
import math
import shlex
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from .contracts import PipelineError
from .fidelity import high_risk_edits, text_patch_evidence
from .io_utils import atomic_write, file_hash, value_hash, write_json


def audit_folder(root: Path | None = None) -> Path:
    from .resources import private_dir, runtime_dir

    parent = root or private_dir(runtime_dir() / "legacy-polish-audits")
    return private_dir(parent / uuid4().hex)


def record_text(folder: Path, name: str, text: str) -> dict:
    path = folder / (name + ".txt")
    atomic_write(path, text)
    return {
        "path": str(path.resolve()),
        "artifact_path": str(path.resolve()),
        "sha256": file_hash(path),
        "characters": len(text),
    }


def safe_normalize(text, rules):
    from .enhancement import normalize_text_for_enhancement, apply_glossary
    from .postprocess import ensure_preserved

    cleaned = normalize_text_for_enhancement(text)
    candidate, applied = apply_glossary(cleaned, rules)
    reason = ""
    try:
        ensure_preserved(text, cleaned, rules)
        for edit in applied:
            if set(high_risk_edits(edit["from"], edit["to"])) - {
                "ENGLISH_TOKEN_CHANGED"
            }:
                raise ValueError("高风险词表修改需要独立核对")
    except ValueError as exc:
        reason = str(exc).split(":", 1)[0]
    adopted = text if reason else candidate
    return (
        adopted,
        ([] if reason else applied),
        {
            "candidate_text": candidate,
            "rejected": bool(reason),
            "reason": reason,
            "candidate_patch": text_patch_evidence(
                text, candidate, segment_id="text", layer="normalization-candidate"
            ),
            "adopted_patch": text_patch_evidence(
                text, adopted, segment_id="text", layer="normalization-adopted"
            ),
        },
    )


def partition_candidate(source: list[dict], candidate: str) -> list[dict]:
    """Assign edits to existing text rows; never move a boundary or time."""
    spans, cursor = [], 0
    for row in source:
        spans.append((cursor, cursor + len(row["text"])))
        cursor += len(row["text"]) + 2
    before = "\n\n".join(row["text"] for row in source)
    edits = [[] for _ in source]
    for tag, left, right, target_left, target_right in SequenceMatcher(
        None, before, candidate, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        owners = [
            i
            for i, (start, end) in enumerate(spans)
            if end > start and start <= left <= right <= end
        ]
        if len(owners) != 1:
            raise ValueError("自由文本修改跨越已有段边界或分隔符，无法保留声音/段身份")
        i = owners[0]
        edits[i].append(
            (
                left - spans[i][0],
                right - spans[i][0],
                candidate[target_left:target_right],
            )
        )
    result = copy.deepcopy(source)
    for row, patches in zip(result, edits):
        for left, right, replacement in reversed(patches):
            row["text"] = row["text"][:left] + replacement + row["text"][right:]
    if "\n\n".join(row["text"] for row in result) != candidate:
        raise ValueError("自由文本修改不能无损贴回既有段")
    return result


def run_legacy_polish(text, *, rules, command, style="", audit_dir=None, config=None):
    from .config import load_config
    from .media_commands import run_media_command
    from .postprocess import ensure_preserved, fidelity_implementation
    from .resources import model_policy

    config = config or load_config()
    limits = config.get_path("postprocess", "legacy_command", default={})
    if not isinstance(limits, dict):
        raise PipelineError(
            "CONFIG_ERROR",
            "postprocess.legacy_command必须为资源配置表",
            blocked=True,
            stage="legacy-polish",
        )
    estimate, maximum = limits.get("estimated_bytes"), limits.get("max_rss_bytes")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in (estimate, maximum)
    ):
        raise PipelineError(
            "RESOURCE_BLOCKED",
            "旧模型命令需显式配置postprocess.legacy_command.estimated_bytes与max_rss_bytes",
            blocked=True,
            stage="legacy-polish",
        )
    timeout = limits.get("timeout_seconds", 120)
    output_limit = limits.get("max_output_bytes", 1024**2)
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or timeout <= 0
        or type(output_limit) is not int
        or output_limit <= 0
    ):
        raise PipelineError(
            "CONFIG_ERROR",
            "旧模型命令时限或输出限制无效",
            blocked=True,
            stage="legacy-polish",
        )
    output_limit = min(output_limit, 8 * 1024**2)
    policy = model_policy(estimate, maximum, timeout, config)
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise PipelineError(
            "CONFIG_ERROR", "旧模型命令格式无效", stage="legacy-polish"
        ) from exc
    if not argv:
        raise PipelineError("CONFIG_ERROR", "旧模型命令为空", stage="legacy-polish")
    folder = audit_folder(audit_dir)
    request = {
        "text": text,
        "style": style,
        "glossary": {"replacements": rules.replacements, "protected": rules.protected},
        "allowed_edits": [
            "typo-fix",
            "term-normalization",
            "punctuation-normalization",
            "spoken-language-segmentation",
        ],
    }
    write_json(folder / "request.json", request)
    source = record_text(folder, "input", text)
    import json

    try:
        completed = run_media_command(
            argv,
            input=json.dumps(request, ensure_ascii=False),
            timeout=timeout,
            check=False,
            stage="legacy-polish",
            config=config,
            max_output_bytes=output_limit,
            resource_policy=policy,
        )
    except BaseException as exc:
        write_json(
            folder / "command.audit.json",
            {
                "schema_version": 1,
                "status": "interrupted",
                "input": source,
                "error_code": getattr(exc, "code", "INTERRUPTED"),
                "command_sha256": value_hash(argv),
            },
        )
        raise
    stdout = record_text(folder, "stdout", completed.stdout)
    stderr = record_text(folder, "stderr", completed.stderr)
    candidate = completed.stdout.strip()
    record = record_text(folder, "candidate", candidate)
    reason = ""
    if completed.returncode or not candidate:
        reason = "旧模型命令失败或输出为空；详见私有候选记录"
    else:
        try:
            ensure_preserved(text, candidate, rules)
        except ValueError as exc:
            reason = str(exc).split(":", 1)[0]
    output = text if reason else candidate
    changed = output != text
    audit = {
        "schema_version": 1,
        "requested": True,
        "applied": changed,
        "status": "partial" if reason or changed else "ready",
        "reason": reason,
        "protected_fields_pass": not reason,
        "review_required": changed,
        "semantic_review": "pending",
        "scope": "unstructured command text; caller must preserve row identity separately",
        "input": source,
        "request": {
            "artifact_path": str((folder / "request.json").resolve()),
            "sha256": file_hash(folder / "request.json"),
        },
        "stdout": stdout,
        "stderr": stderr,
        "candidate": record,
        "adopted": record_text(folder, "command-adopted", output),
        "candidate_patch": text_patch_evidence(
            text, candidate, segment_id="text", layer="command-candidate"
        ),
        "adopted_patch": text_patch_evidence(
            text, output, segment_id="text", layer="command-adopted"
        ),
        "returncode": completed.returncode,
        "command_sha256": value_hash(argv),
        "implementation": fidelity_implementation(),
        "resource_budget": {
            "estimated_bytes": estimate,
            "max_rss_bytes": maximum,
            "reserve_bytes": policy.reserve_bytes,
            "timeout_seconds": timeout,
            "max_output_bytes": output_limit,
        },
    }
    path = folder / "command.audit.json"
    write_json(path, audit)
    return output, {
        **audit,
        "audit_path": str(path.resolve()),
        "audit_sha256": file_hash(path),
    }
