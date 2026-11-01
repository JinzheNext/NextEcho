from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_GLOSSARY_PATHS = [
    PROJECT_ROOT / "glossary.json",
    PROJECT_ROOT / "glossary.txt",
]
BUILTIN_REPLACEMENTS = {
    "波客": "播客",
}
BUILTIN_PROTECTED = {
    "NextEcho",
    "@金哲Next",
}


@dataclass
class GlossaryRules:
    replacements: list[tuple[str, str]] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)


def load_glossary_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.expanduser().read_text(encoding="utf-8")


def project_glossary_text() -> str:
    parts: list[str] = []
    for path in PROJECT_GLOSSARY_PATHS:
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts).strip()


def parse_glossary_text(raw: str) -> GlossaryRules:
    text = raw.strip()
    if not text:
        return GlossaryRules()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _parse_glossary_lines(text)
    return _parse_glossary_json(payload)


def merge_glossaries(*rulesets: GlossaryRules) -> GlossaryRules:
    replacements: dict[str, str] = {}
    protected: list[str] = []
    seen_protected: set[str] = set()
    for rules in rulesets:
        for source, target in rules.replacements:
            replacements[source] = target
        for item in rules.protected:
            if item and item not in seen_protected:
                seen_protected.add(item)
                protected.append(item)
    ordered = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
    return GlossaryRules(replacements=ordered, protected=protected)


def default_glossary_rules(user_glossary_text: str = "") -> GlossaryRules:
    builtin = GlossaryRules(
        replacements=sorted(
            BUILTIN_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True
        ),
        protected=sorted(BUILTIN_PROTECTED),
    )
    return merge_glossaries(
        builtin,
        parse_glossary_text(project_glossary_text()),
        parse_glossary_text(user_glossary_text),
    )


def normalize_text_for_enhancement(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    replacements = {
        "， ": "，",
        "。 ": "。",
        "！ ": "！",
        "？ ": "？",
        "； ": "；",
        "： ": "：",
        "（ ": "（",
        " ）": "）",
        "AI ": "AI ",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized.strip()


def strip_filler_words(text: str) -> str:
    """Remove only delimited hesitation 呃 inside a longer utterance.

    Attached characters can be real words (金额、额外、哈佛); 嗯/啊/哈 can
    also be meaningful responses. Standalone interjections are preserved.
    Further disfluency edits belong to an auditable candidate patch.
    """
    if re.fullmatch(r"[\s，。！？、,.!?嗯呃额啊哈唉诶]+", text):
        return text.strip()
    cleaned = re.sub(r"(^|[，,；;\s])呃+[，,；;\s]+", lambda match: match[1], text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def apply_glossary(text: str, rules: GlossaryRules) -> tuple[str, list[dict[str, str]]]:
    protected_placeholders: dict[str, str] = {}
    protected_text = text
    for index, item in enumerate(sorted(rules.protected, key=len, reverse=True)):
        if not item:
            continue
        token = f"__NEXTECHO_PROTECTED_{index}__"
        protected_placeholders[token] = item
        protected_text = protected_text.replace(item, token)

    replaced = protected_text
    applied: list[dict[str, str]] = []
    for source, target in rules.replacements:
        if not source or source == target or source not in replaced:
            continue
        replaced = replaced.replace(source, target)
        applied.append({"from": source, "to": target})

    for token, item in protected_placeholders.items():
        replaced = replaced.replace(token, item)
    return replaced, applied


def maybe_ai_polish(
    text: str,
    *,
    rules: GlossaryRules,
    enabled: bool,
    style: str = "",
    audit_dir: Path | None = None,
    config=None,
) -> tuple[str, dict[str, Any]]:
    if not enabled:
        return text, {
            "requested": False,
            "applied": False,
            "reason": "",
            "status": "ready",
        }
    command = os.environ.get("TRANSCRIBE_AI_POLISH_COMMAND", "").strip()
    if not command:
        return text, {
            "requested": True,
            "applied": False,
            "status": "partial",
            "reason": "AI polishing is not configured. Set TRANSCRIBE_AI_POLISH_COMMAND to enable it.",
            "protected_fields_pass": False,
        }
    from .legacy_polish import run_legacy_polish

    return run_legacy_polish(
        text,
        rules=rules,
        command=command,
        style=style,
        audit_dir=audit_dir,
        config=config,
    )


def enhance_transcript_artifacts(
    item_dir: Path,
    *,
    user_glossary_text: str = "",
    ai_polish: bool = False,
    ai_style: str = "",
    config=None,
    input_artifacts: dict | None = None,
) -> dict[str, Any]:
    import copy
    import math
    from .contracts import PipelineError
    from .fidelity import text_patch_evidence
    from .io_utils import atomic_write, file_hash, write_json
    from .legacy_polish import (
        audit_folder,
        record_text,
        safe_normalize,
        partition_candidate,
    )
    from .postprocess import (
        ensure_preserved,
        fidelity_implementation,
        load_speaker_turns,
    )
    from .transcription import format_transcript_attribution

    txt_path, json_path = item_dir / "transcript.txt", item_dir / "transcript.json"
    if not txt_path.is_file():
        return {
            "enabled": False,
            "status": "partial",
            "enhanced_text_path": "",
            "enhanced_json_path": "",
            "rules_applied": [],
            "ai": {
                "requested": ai_polish,
                "applied": False,
                "reason": "transcript.txt not found",
            },
        }
    original_text = txt_path.read_text(encoding="utf-8")
    raw_payload = {}
    if json_path.is_file():
        try:
            raw_payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(raw_payload, dict):
                raise ValueError("root")
        except (ValueError, TypeError) as exc:
            raise PipelineError(
                "PROTOCOL_ERROR", "原始转录JSON不可读取，整理未执行"
            ) from exc
    rows = None
    speaker_source = None
    from .text_inputs import select_text_input
    input_selection = select_text_input(item_dir, input_artifacts)
    protected = {path: file_hash(path) for path in (txt_path, json_path,
        *((item_dir / "transcript.speakers.json",) if input_selection["layer"] != "raw-asr" else ())) if path.is_file()}
    if input_selection["layer"] == "acoustic-dialogue":
        from .acoustic_dialogue import load_dialogue_segments
        load_dialogue_segments(item_dir, file_hash(json_path))
    if input_selection["layer"] != "raw-asr":
        if not (item_dir / "transcript.speakers.json").is_file():
            raise PipelineError("CACHE_MISMATCH", "所选speaker输入缺少显式兼容导出", blocked=True)
        turns = load_speaker_turns(item_dir, required=True, local_only=True)
        rows = []
        for index, turn in enumerate(turns):
            start, end = turn["start"], turn["end"]
            if (
                any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in (start, end)
                )
                or not 0 <= start <= end
            ):
                raise PipelineError("PROTOCOL_ERROR", "speaker试听范围无效")
            rows.append(
                {
                    **copy.deepcopy(turn),
                    "id": turn.get("id") or f"speaker-review-{index:06d}",
                    "offsets": {"from": round(start * 1000), "to": round(end * 1000)},
                    "annotation_kind": "text-review-envelope",
                    "timing_precision": "review-envelope",
                }
            )
        speaker_source = {
            "path": str(item_dir / "transcript.speakers.json"),
            "sha256": file_hash(item_dir / "transcript.speakers.json"),
            "kind": "speaker-artifacts",
        }
    elif raw_payload.get("transcription"):
        rows = copy.deepcopy(raw_payload["transcription"])
    if rows is not None:
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("text"), str)
            for row in rows
        ):
            raise PipelineError("PROTOCOL_ERROR", "原始逐段文字无效")
        ids = []
        for index, row in enumerate(rows):
            if row.get("id") is None:
                row["id"] = f"legacy-source-{index:06d}"
                row["derived_id_from_source_index"] = index
            if type(row["id"]) not in (str, int):
                raise PipelineError("PROTOCOL_ERROR", "原始段ID无效")
            ids.append((type(row["id"]).__name__, row["id"]))
        if len(set(ids)) != len(ids):
            raise PipelineError("PROTOCOL_ERROR", "原始段ID重复，不能合并整理")
        if not rows:
            rows = None
    rules = default_glossary_rules(user_glossary_text)
    folder = audit_folder(item_dir / "postprocess" / "legacy-enhancement")
    source_text = (
        "\n\n".join(row["text"] for row in rows) if rows is not None else original_text
    )
    record_text(folder, "original-txt", original_text)
    original_record = record_text(folder, "selected-input", source_text)
    normalization, replacements = [], []
    normalized = copy.deepcopy(rows)
    if rows is not None:
        for before, after in zip(rows, normalized):
            after["text"], applied, audit = safe_normalize(before["text"], rules)
            audit["segment_id"] = before["id"]
            normalization.append(audit)
            replacements.extend(applied)
        cleaned = "\n\n".join(row["text"] for row in normalized)
    else:
        cleaned, replacements, audit = safe_normalize(source_text, rules)
        normalization.append(audit)
    write_json(folder / "normalization.json", normalization)
    record_text(folder, "normalized-input", cleaned)
    candidate, ai_info = maybe_ai_polish(
        cleaned,
        rules=rules,
        enabled=ai_polish,
        style=ai_style,
        audit_dir=folder / "command",
        config=config,
    )
    rejected = bool(ai_polish and ai_info.get("reason"))
    mapping_error = ""
    selected = None
    if rows is not None and not rejected:
        try:
            selected = partition_candidate(normalized, candidate)
            if ai_polish:
                for before, after in zip(normalized, selected):
                    ensure_preserved(before["text"], after["text"], rules)
        except ValueError as exc:
            mapping_error = str(exc).split(":", 1)[0]
            rejected = True
    if rejected:
        final_text = source_text
        selected = copy.deepcopy(rows)
        replacements = []
        ai_info = {
            **ai_info,
            "command_stage_applied": ai_info.get("applied", False),
            "applied": False,
            "status": "partial",
            "protected_fields_pass": False,
            "reason": mapping_error or ai_info.get("reason"),
            "fallback_input_layer": "original-selected-input",
        }
    else:
        final_text = candidate
    if selected is not None:
        final_text = "\n\n".join(
            (str(row.get("speaker") or "") + "：" if row.get("speaker") else "")
            + row["text"]
            for row in selected
        )
    fields_pass = not rejected and not any(audit["rejected"] for audit in normalization)
    partial = not fields_pass or ai_info.get("status") == "partial"
    status = "partial" if partial else "ready"
    if any(
        not path.is_file() or file_hash(path) != digest
        for path, digest in protected.items()
    ):
        raise PipelineError(
            "CACHE_MISMATCH", "整理时原始稿或speaker证据发生变化", blocked=True
        )
    adopted_body = (
        "\n\n".join(row["text"] for row in selected)
        if selected is not None
        else final_text
    )
    final_record = record_text(folder, "final-adopted", adopted_body)
    patches = (
        [
            text_patch_evidence(
                before["text"],
                after["text"],
                segment_id=str(before["id"]),
                layer="final-adopted",
            )
            for before, after in zip(rows, selected)
            if before["text"] != after["text"]
        ]
        if rows is not None
        else [
            text_patch_evidence(
                source_text, adopted_body, segment_id="text", layer="final-adopted"
            )
        ]
    )
    audit = {
        "schema_version": 1,
        "status": status,
        "semantic_review": "pending",
        "protected_fields_pass": fields_pass,
        "selected_input": original_record,
        "final_adopted": final_record,
        "normalization_path": str((folder / "normalization.json").resolve()),
        "normalization_sha256": file_hash(folder / "normalization.json"),
        "ai": ai_info,
        "adopted_patches": patches,
        "fallback_input_layer": "original-selected-input" if rejected else "",
        "source_files": {
            str(path.resolve()): digest for path, digest in protected.items()
        },
        "speaker_source": speaker_source,
        "input_selection": input_selection,
        "implementation": fidelity_implementation(),
        "row_identity_and_times_preserved": selected is not None,
        "unstructured_text": selected is None,
    }
    audit_path = folder / "enhancement.audit.json"
    write_json(audit_path, audit)
    enhanced = copy.deepcopy(raw_payload)
    enhanced["enhancement"] = {
        "glossary_applied": bool(replacements),
        "rules_applied": replacements,
        "protected": rules.protected,
        "ai": ai_info,
        "style": ai_style,
        "status": status,
        "protected_fields_pass": fields_pass,
        "semantic_review": "pending",
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": file_hash(audit_path),
        "speaker_source": speaker_source,
    }
    enhanced["enhanced_text"] = final_text
    if selected is not None:
        enhanced["enhanced_transcription"] = selected
    text_path, json_out = (
        item_dir / "transcript.enhanced.txt",
        item_dir / "transcript.enhanced.json",
    )
    atomic_write(text_path, format_transcript_attribution(final_text))
    write_json(json_out, enhanced)
    return {
        "enabled": True,
        "status": status,
        "protected_fields_pass": fields_pass,
        "semantic_review": "pending",
        "enhanced_text_path": str(text_path),
        "enhanced_json_path": str(json_out),
        "rules_applied": replacements,
        "ai": ai_info,
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": file_hash(audit_path),
        "evidence_paths": {
            name.replace("-", "_").replace(".", "_") + "_path": str(
                (folder / name).resolve()
            )
            for name in (
                "original-txt.txt",
                "selected-input.txt",
                "normalized-input.txt",
                "normalization.json",
                "final-adopted.txt",
            )
        },
    }


def _parse_glossary_json(payload: Any) -> GlossaryRules:
    replacements: list[tuple[str, str]] = []
    protected: list[str] = []
    if isinstance(payload, dict):
        raw_replacements = payload.get("replacements", payload.get("rules", []))
        raw_protected = payload.get("protected", payload.get("protect", []))
        if isinstance(raw_replacements, dict):
            replacements.extend(
                (str(key), str(value)) for key, value in raw_replacements.items()
            )
        elif isinstance(raw_replacements, list):
            for item in raw_replacements:
                if isinstance(item, dict) and "from" in item and "to" in item:
                    replacements.append((str(item["from"]), str(item["to"])))
        if isinstance(raw_protected, list):
            protected.extend(str(item) for item in raw_protected if str(item).strip())
    return GlossaryRules(
        replacements=sorted(replacements, key=lambda item: len(item[0]), reverse=True),
        protected=protected,
    )


def _parse_glossary_lines(text: str) -> GlossaryRules:
    replacements: list[tuple[str, str]] = []
    protected: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            source, target = [item.strip() for item in line.split("=>", 1)]
            if source and target:
                replacements.append((source, target))
            continue
        if ":" in line:
            key, value = [item.strip() for item in line.split(":", 1)]
            if key.lower() in {"protect", "protected", "保护词"} and value:
                protected.append(value)
    return GlossaryRules(
        replacements=sorted(replacements, key=lambda item: len(item[0]), reverse=True),
        protected=protected,
    )
