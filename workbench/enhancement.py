from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
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
        replacements=sorted(BUILTIN_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True),
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
    """Remove common speech fillers while leaving the raw transcript untouched.

    ASR often emits standalone hesitation runs (嗯、呃、啊、哈).  These are
    presentation noise in a readable transcript, but the original transcript
    remains the evidence artifact.  We intentionally keep this list narrow so
    meaningful words such as “对” and “吧” are not removed.
    """
    cleaned = re.sub(r"[嗯呃额啊哈唉诶]+", "", text)
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
) -> tuple[str, dict[str, Any]]:
    if not enabled:
        return text, {"requested": False, "applied": False, "reason": ""}

    command_raw = os.environ.get("TRANSCRIBE_AI_POLISH_COMMAND", "").strip()
    if not command_raw:
        return text, {
            "requested": True,
            "applied": False,
            "reason": "AI polishing is not configured. Set TRANSCRIBE_AI_POLISH_COMMAND to enable it.",
        }

    payload = {
        "text": text,
        "style": style,
        "glossary": {
            "replacements": rules.replacements,
            "protected": rules.protected,
        },
        "allowed_edits": [
            "typo-fix",
            "term-normalization",
            "punctuation-normalization",
            "spoken-language-segmentation",
        ],
    }
    completed = subprocess.run(
        shlex.split(command_raw),
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0 or not output:
        return text, {
            "requested": True,
            "applied": False,
            "reason": (completed.stderr or completed.stdout or "AI polishing command failed.").strip(),
        }
    return output, {"requested": True, "applied": True, "reason": ""}


def enhance_transcript_artifacts(
    item_dir: Path,
    *,
    user_glossary_text: str = "",
    ai_polish: bool = False,
    ai_style: str = "",
) -> dict[str, Any]:
    txt_path = item_dir / "transcript.txt"
    json_path = item_dir / "transcript.json"
    if not txt_path.exists():
        return {
            "enabled": False,
            "enhanced_text_path": "",
            "enhanced_json_path": "",
            "rules_applied": [],
            "ai": {"requested": ai_polish, "applied": False, "reason": "transcript.txt not found"},
        }

    original_text = txt_path.read_text(encoding="utf-8")
    rules = default_glossary_rules(user_glossary_text)
    normalized_text = normalize_text_for_enhancement(original_text)
    glossary_text, replacements = apply_glossary(normalized_text, rules)
    final_text, ai_info = maybe_ai_polish(
        glossary_text,
        rules=rules,
        enabled=ai_polish,
        style=ai_style,
    )

    enhanced_text_path = item_dir / "transcript.enhanced.txt"
    enhanced_json_path = item_dir / "transcript.enhanced.json"
    from .transcription import format_transcript_attribution

    enhanced_text_path.write_text(format_transcript_attribution(final_text), encoding="utf-8")

    json_payload: dict[str, Any] = {}
    if json_path.exists():
        try:
            json_payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            json_payload = {}
    json_payload["enhancement"] = {
        "glossary_applied": bool(replacements),
        "rules_applied": replacements,
        "protected": rules.protected,
        "ai": ai_info,
        "style": ai_style,
    }
    json_payload["enhanced_text"] = final_text
    enhanced_json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "enabled": True,
        "enhanced_text_path": str(enhanced_text_path),
        "enhanced_json_path": str(enhanced_json_path),
        "rules_applied": replacements,
        "ai": ai_info,
    }


def _parse_glossary_json(payload: Any) -> GlossaryRules:
    replacements: list[tuple[str, str]] = []
    protected: list[str] = []
    if isinstance(payload, dict):
        raw_replacements = payload.get("replacements", payload.get("rules", []))
        raw_protected = payload.get("protected", payload.get("protect", []))
        if isinstance(raw_replacements, dict):
            replacements.extend((str(key), str(value)) for key, value in raw_replacements.items())
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
