"""Explicit dialogue -> v3 compatibility export, preserving IDs and evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from .contracts import PipelineError
from .io_utils import atomic_write, write_json

DIALOGUE_VERSIONS = {
    "native-ctc-dialogue-v1",
    "native-ctc-dialogue-v2-evidence",
    "native-ctc-dialogue-v3-artifacts",
}


def is_dialogue(payload):
    return (
        payload.get("schema_version") == 1
        and payload.get("version") in DIALOGUE_VERSIONS
    )


def dialogue_to_v3(payload: dict, *, source_label: str = "") -> dict:
    try:
        if not is_dialogue(payload) or payload["speakers"]["schema_version"] != 3:
            raise ValueError("unknown dialogue protocol")
        speakers = copy.deepcopy(payload["speakers"]["speakers"])
        rows = copy.deepcopy(payload["transcription"])
        if not isinstance(speakers, list) or not isinstance(rows, list):
            raise ValueError("invalid lists")
        by_id, by_raw = {}, {}
        for index, person in enumerate(speakers):
            identity = person["speaker_id"]
            if (
                not isinstance(identity, str)
                or not identity
                or identity in by_id
                or not isinstance(person["identity_evidence"], list)
            ):
                raise ValueError("invalid identity")
            if not isinstance(person["raw_labels"], list) or not person["raw_labels"]:
                raise ValueError("missing acoustic label")
            score = person["confidence"]
            if (
                not isinstance(score, (float, int))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("invalid evidence score")
            person["anonymous_label"] = f"说话人 {index + 1}"
            by_id[identity] = person
            for label in person["raw_labels"]:
                if not isinstance(label, str) or not label or label in by_raw:
                    raise ValueError("ambiguous raw identity")
                by_raw[label] = person
        turns = []
        for row in rows:
            start, end = row["start_ms"], row["end_ms"]
            if (
                not all(
                    isinstance(v, (int, float)) and math.isfinite(v)
                    for v in (start, end)
                )
                or start < 0
                or end < start
                or not isinstance(row["text"], str)
            ):
                raise ValueError("invalid review range or text")
            identity = row.get("speaker_id", "")
            person = by_id.get(identity)
            if identity and (
                person is None or by_raw.get(row.get("raw_speaker")) is not person
            ):
                raise ValueError("turn references a different identity")
            if person:
                display = (
                    person["display_name"]
                    if person["identity_status"] == "resolved"
                    else person["anonymous_label"]
                )
                row.update(
                    speaker=display,
                    display_name=person["display_name"],
                    identity_status=person["identity_status"],
                    identity_evidence=copy.deepcopy(person["identity_evidence"]),
                )
            elif row.get("raw_speaker"):
                raise ValueError("unmapped acoustic label")
            row.update(
                start=start / 1000,
                end=end / 1000,
                annotation_kind="text-review-envelope",
                timing_precision="review-envelope",
            )
            turns.append(row)
        mapping = {
            raw: p["display_name"]
            for raw, p in by_raw.items()
            if p["identity_status"] == "resolved"
        }
        starts = [r["start_ms"] for r in turns]
        ends = [r["end_ms"] for r in turns]
        return {
            **{
                k: copy.deepcopy(v)
                for k, v in payload.items()
                if k not in {"transcription", "speaker_turns", "speakers"}
            },
            "schema_name": "nextecho.speakers",
            "schema_version": 3,
            "dialogue_version": payload["version"],
            "source_label": source_label or payload.get("source_label", ""),
            "speakers": speakers,
            "speaker_turns": turns,
            "speaker_map": mapping,
            "speaker_count_detected": len(speakers),
            "identity_status": "resolved" if mapping else "unknown",
            "annotation_kind": "text-review-envelope",
            "scoring_eligible": False,
            "coverage": {
                "start_ms": min(starts, default=0),
                "end_ms": max(ends, default=0),
                "duration_ms": max(ends, default=0) - min(starts, default=0),
                "known": bool(turns),
                "kind": "review-envelope",
            },
            "preview_text": "".join(r["text"] for r in turns)[:400],
        }
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise PipelineError(
            "PROTOCOL_ERROR", "对话到v3导出的身份、时间或证据不一致"
        ) from exc


def write_v3_outputs(root: Path, payload: dict) -> dict:
    from .speaker_transcript import render_speaker_text

    if (
        payload.get("schema_version") != 3
        or not isinstance(payload.get("speakers"), list)
        or not isinstance(payload.get("speaker_turns"), list)
    ):
        raise PipelineError("PROTOCOL_ERROR", "speaker兼容文件必须为v3列表协议")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "speaker_json_path": str(root / "transcript.speakers.json"),
        "speaker_text_path": str(root / "transcript.speakers.txt"),
        "speaker_markdown_path": str(root / "transcript.speakers.md"),
        "speaker_map_path": str(root / "speaker_map.json"),
    }
    write_json(Path(paths["speaker_json_path"]), payload)
    atomic_write(
        Path(paths["speaker_text_path"]),
        render_speaker_text(payload["speaker_turns"], with_timestamps=False),
    )
    atomic_write(
        Path(paths["speaker_markdown_path"]),
        render_speaker_text(payload["speaker_turns"], with_timestamps=True),
    )
    write_json(Path(paths["speaker_map_path"]), payload.get("speaker_map", {}))
    return paths


def read_speaker_file(path: Path) -> dict:
    from .speaker_scope import scope_from_files
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PipelineError("PROTOCOL_ERROR", "speaker文件根对象无效")
    if is_dialogue(payload):
        return scope_from_files(dialogue_to_v3(payload), path.parent)
    if payload.get("schema_version") == 3 and not isinstance(
        payload.get("speakers"), list
    ):
        raise PipelineError("PROTOCOL_ERROR", "v3 speaker文件需要speakers列表")
    if payload.get("schema_name") == "nextecho.speakers":
        # This export is derived from a particular immutable dialogue. Never
        # silently repair an altered export during a read or trust stale names.
        from .io_utils import file_hash

        try:
            canonical = path.parent / "transcript.dialogue.json"
            if payload["canonical_sha256"] != file_hash(canonical):
                raise ValueError("changed canonical dialogue")
            base = json.loads(canonical.read_text(encoding="utf-8"))
            expected = dialogue_to_v3(
                mapped_dialogue(path.parent, base),
                source_label=payload.get("source_label", ""),
            )
            if "recording_identity" in payload:
                expected = scope_from_files(expected, path.parent)
            for key in (
                "dialogue_version",
                "speakers",
                "speaker_turns",
                "speaker_map",
                "speaker_labels",
                "identity_status",
                "identity_mapping_sha256",
                "annotation_kind",
                "scoring_eligible",
                "coverage",
                "acoustic_evidence",
                "attribution_profile",
                "backend",
                "model",
                "model_revision",
                "model_digest",
                "release_status",
                "quality_validation",
                "default_promotion",
                *(["recording_identity"] if "recording_identity" in payload else []),
            ):
                if payload.get(key) != expected.get(key):
                    raise ValueError(f"inconsistent export: {key}")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise PipelineError(
                "CACHE_MISMATCH",
                "speaker导出与对话基底或人工身份映射不一致",
                blocked=True,
            ) from exc
    return scope_from_files(payload, path.parent)


def apply_identity_mapping(payload: dict, mapping: dict[str, str]) -> dict:
    """Only explicit, existing speaker IDs authorize manual identity changes."""
    if not isinstance(mapping, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()
    ):
        raise PipelineError("INVALID_INPUT", "人工映射需要字符串ID与姓名")
    updated = copy.deepcopy(payload)
    people = updated.get("speakers")
    if not isinstance(people, list):
        raise PipelineError("PROTOCOL_ERROR", "人工映射需要v3身份列表")
    by_id = {p["speaker_id"]: p for p in people}
    if any(identity not in by_id for identity in mapping):
        raise PipelineError("INVALID_INPUT", "映射包含当前录音中不存在的speaker ID")
    for identity, name in mapping.items():
        person = by_id[identity]
        person.update(
            display_name=name or "unknown",
            identity_status="resolved" if name else "unknown",
            confidence=1.0 if name else 0.0,
        )
        event = {
            "type": "manual",
            "source": "speaker-map-api",
            "confidence": 1.0,
            "details": {"display_name": name, "action": "assign" if name else "clear"},
        }
        evidence = person.setdefault("identity_evidence", [])
        if not evidence or evidence[-1] != event:
            evidence.append(event)
    for turn in updated.get("speaker_turns", []):
        person = by_id.get(turn.get("speaker_id"))
        if person and person["speaker_id"] in mapping:
            turn.update(
                speaker=person["display_name"]
                if person["identity_status"] == "resolved"
                else person.get("anonymous_label", person["speaker_id"]),
                display_name=person["display_name"],
                identity_status=person["identity_status"],
                identity_evidence=copy.deepcopy(person["identity_evidence"]),
            )
    updated["speaker_map"] = {
        raw: p["display_name"]
        for p in people
        if p["identity_status"] == "resolved"
        for raw in p.get("raw_labels", [])
    }
    updated["speaker_labels"] = {
        raw: (
            p["display_name"]
            if p["identity_status"] == "resolved"
            else p.get("anonymous_label", p["speaker_id"])
        )
        for p in people
        for raw in p.get("raw_labels", [])
    }
    updated["identity_status"] = (
        "resolved"
        if any(p["identity_status"] == "resolved" for p in people)
        else "unknown"
    )
    return updated


def mapping_for_dialogue(root: Path, payload: dict) -> dict:
    from .io_utils import file_hash

    path = root / "speaker.identity-map.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(
            "CACHE_MISMATCH", "人工身份映射不可读取", blocked=True
        ) from exc
    if not isinstance(data, dict):
        raise PipelineError("PROTOCOL_ERROR", "人工身份映射根对象无效")
    if (
        data.get("schema_version") != 1
        or data.get("canonical_sha256") != file_hash(root / "transcript.dialogue.json")
        or not isinstance(data.get("mapping"), dict)
    ):
        raise PipelineError(
            "CACHE_MISMATCH",
            "人工身份映射与当前对话基底不一致，请重新核对speaker ID",
            blocked=True,
        )
    if any(
        not isinstance(k, str) or not isinstance(v, str)
        for k, v in data["mapping"].items()
    ):
        raise PipelineError("PROTOCOL_ERROR", "身份映射需要字符串ID与姓名")
    return data["mapping"]


def mapped_dialogue(root: Path, payload: dict) -> dict:
    mapping = mapping_for_dialogue(root, payload)
    if not mapping:
        return payload
    updated = copy.deepcopy(payload)
    mapped = apply_identity_mapping(dialogue_to_v3(payload), mapping)
    updated["speakers"]["speakers"] = mapped["speakers"]
    by_id = {r["id"]: r for r in mapped["speaker_turns"]}
    keys = ("speaker", "display_name", "identity_status", "identity_evidence")
    for row in updated["transcription"]:
        row.update(
            {key: by_id[row["id"]][key] for key in keys if key in by_id[row["id"]]}
        )
    updated["speaker_turns"] = mapped["speaker_turns"]
    updated["speaker_labels"] = mapped["speaker_labels"]
    from .io_utils import file_hash

    updated["identity_mapping_sha256"] = file_hash(root / "speaker.identity-map.json")
    return updated


def write_identity_reading(root: Path, speaker_payload: dict) -> dict:
    """Apply diarized character spans without changing adopted source text."""
    from .io_utils import file_hash
    from .transcription import format_transcript_attribution

    turns = [r for r in speaker_payload.get("speaker_turns", []) if isinstance(r, dict)]
    by_id = {r["id"]: r for r in turns if r.get("id")}
    parent_turns = {}
    for turn in turns:
        for parent in turn.get("segment_ids", []):
            parent_turns.setdefault(str(parent), []).append(turn)
    original_rows = {}
    raw_path_value = str(speaker_payload.get("transcript_source_path") or root / "transcript.json")
    raw_path = Path(raw_path_value).expanduser()
    if raw_path.is_file():
        try:
            expected_hash = str(speaker_payload.get("transcript_source_sha256") or "")
            if not expected_hash or file_hash(raw_path) == expected_hash:
                raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                original_rows = {str(item.get("id") or item.get("segment_id") or f"seg_{index:05d}"): str(item.get("text") or "").strip()
                                 for index, item in enumerate(raw_payload.get("transcription") or []) if isinstance(item, dict)}
        except (OSError, ValueError, json.JSONDecodeError):
            original_rows = {}
    source = root / "transcript.enhanced.json"
    if source.is_file():
        original = json.loads(source.read_text())
        rows = copy.deepcopy(original.get("enhanced_transcription"))
        if not isinstance(rows, list):
            raise PipelineError(
                "PROTOCOL_ERROR", "增强稿没有明确的enhanced_transcription层"
            )
        rendered_rows, unassigned = [], []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise PipelineError("PROTOCOL_ERROR", "增强稿包含无效字符区间来源")
            source_id, text = str(row.get("id") or ""), row["text"]
            span_text = original_rows.get(source_id, text)
            assignments = [None] * len(span_text)
            has_span_claim = False
            for turn in turns:
                speaker_id = str(turn.get("speaker_id") or "")
                speaker_uuid = str(turn.get("speaker_uuid") or "")
                speaker = str(turn.get("speaker") or "归属待核对")
                for span in turn.get("alignment_spans", []):
                    if not isinstance(span, dict) or str(span.get("segment_id") or "") != source_id:
                        continue
                    has_span_claim = True
                    if span.get("source_text_sha256") != hashlib.sha256(span_text.encode("utf-8")).hexdigest():
                        continue
                    start, end = span.get("text_start"), span.get("text_end")
                    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(span_text):
                        continue
                    native_start = span.get("native_start_ms")
                    native_end = span.get("native_end_ms")
                    reason = str(span.get("assignment_reason") or "")
                    valid_start = isinstance(native_start, (int, float)) and math.isfinite(native_start)
                    valid_end = isinstance(native_end, (int, float)) and math.isfinite(native_end)
                    if native_start is not None or native_end is not None:
                        valid_native_interval = valid_start and valid_end and native_end > native_start
                        valid_native_onset = valid_start and native_end is None and reason == "native_onset"
                        if not (valid_native_interval or valid_native_onset):
                            continue
                    evidence = {"assignment_reason": span.get("assignment_reason", ""),
                                "native_start_ms": native_start,
                                "native_end_ms": native_end,
                                "review_start_ms": span.get("review_start_ms"),
                                "review_end_ms": span.get("review_end_ms")}
                    identity = {"speaker": speaker, "speaker_id": speaker_id,
                                "speaker_uuid": speaker_uuid, "evidence": evidence}
                    for position in range(start, end):
                        if assignments[position] is None:
                            assignments[position] = identity
                        elif assignments[position] != identity:
                            assignments[position] = {"speaker": "归属待核对", "speaker_id": "", "speaker_uuid": "",
                                                     "evidence": {"assignment_reason": "conflicting_character_spans"}}
            if span_text != text:
                projected = [None] * len(text)
                matcher = SequenceMatcher(None, span_text, text, autojunk=False)
                for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
                    if tag == "equal":
                        projected[right_start:right_end] = assignments[left_start:left_end]
                # New punctuation and spacing do not carry lexical identity;
                # inheriting the preceding matched voice is safe for reading.
                for index, char in enumerate(text):
                    if projected[index] is None and (char.isspace() or unicodedata.category(char)[0] == "P") and index:
                        previous = projected[index - 1]
                        if previous is not None:
                            projected[index] = {"speaker": previous["speaker"], "speaker_id": previous["speaker_id"],
                                                "speaker_uuid": previous["speaker_uuid"],
                                                "evidence": {"assignment_reason": "adopted_punctuation_context"}}
                assignments = projected
            # Old speaker files may not contain character spans. Whole-parent
            # fallback is permitted only when the acoustic UUID is unique.
            if text and not any(assignments) and not has_span_claim:
                exact = by_id.get(source_id)
                candidates = [exact] if exact else parent_turns.get(source_id, [])
                unique = {
                    (str(item.get("speaker_uuid") or "") or str(item.get("speaker_id") or ""))
                    for item in candidates if item
                }
                if len(candidates) == 1 and len(unique) == 1 and next(iter(unique)):
                    item = candidates[0]
                    assignments = [{"speaker": str(item.get("speaker") or ""),
                                    "speaker_id": str(item.get("speaker_id") or ""),
                                    "speaker_uuid": str(item.get("speaker_uuid") or ""),
                                    "evidence": {"assignment_reason": "unique_acoustic_parent"}} for _ in text]
            # An absent or zero-width character alignment is never upgraded to
            # a whole-window or nearest-speaker assignment.
            if len(assignments) != len(text):
                assignments = [None] * len(text)
            for index, item in enumerate(assignments):
                if item is None:
                    assignments[index] = {"speaker": "归属待核对", "speaker_id": "", "speaker_uuid": "",
                                          "evidence": {"assignment_reason": "no_valid_character_span"}}
            cursor = 0
            run_index = 0
            while cursor < len(text):
                item = assignments[cursor]
                end = cursor + 1
                identity_key = (item["speaker_uuid"], item["speaker_id"], item["speaker"])
                while end < len(text) and (
                    assignments[end]["speaker_uuid"], assignments[end]["speaker_id"], assignments[end]["speaker"]
                ) == identity_key:
                    end += 1
                run_items = assignments[cursor:end]
                evidence = [part["evidence"] for part in run_items]
                reasons = sorted({str(part.get("assignment_reason") or "") for part in evidence if part.get("assignment_reason")})
                voice_evidence = []
                for offset, item_evidence in enumerate(evidence):
                    native_start = item_evidence.get("native_start_ms")
                    native_end = item_evidence.get("native_end_ms")
                    if isinstance(native_start, (int, float)) and isinstance(native_end, (int, float)):
                        timing_kind = "native_interval" if native_end > native_start else "invalid_zero_width"
                    elif isinstance(native_start, (int, float)):
                        timing_kind = "native_onset"
                    else:
                        timing_kind = "review_only"
                    entry = {"text_start": cursor + offset, "text_end": cursor + offset + 1,
                             "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                             "timing_kind": timing_kind, "native_start_ms": native_start,
                             "native_end_ms": native_end,
                             "assignment_reason": item_evidence.get("assignment_reason", "")}
                    merge_keys = ("source_text_sha256", "timing_kind", "native_start_ms", "native_end_ms", "assignment_reason")
                    if voice_evidence and voice_evidence[-1]["text_end"] == entry["text_start"] and all(
                        voice_evidence[-1].get(key) == entry.get(key) for key in merge_keys
                    ):
                        voice_evidence[-1]["text_end"] = entry["text_end"]
                    else:
                        voice_evidence.append(entry)
                output = copy.deepcopy(row)
                output.update(id=f"{source_id}-voice-{run_index:03d}", parent_segment_id=source_id,
                              text=text[cursor:end], speaker=item["speaker"],
                              speaker_id=item["speaker_id"], speaker_uuid=item["speaker_uuid"],
                              text_start=cursor, text_end=end,
                              source_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                              identity_evidence={"assignment_reasons": reasons, "character_count": end - cursor},
                              voice_evidence=voice_evidence,
                              speaker_assignment_basis=("native" if any(row["timing_kind"] in {"native_interval", "native_onset"} for row in voice_evidence)
                                                       else "review_only" if item["speaker_id"] else "unassigned"))
                starts = [part.get("review_start_ms") for part in evidence if isinstance(part.get("review_start_ms"), (int, float))]
                ends = [part.get("review_end_ms") for part in evidence if isinstance(part.get("review_end_ms"), (int, float))]
                if starts:
                    output["offsets"] = {"from": min(starts), "to": max(ends or starts)}
                rendered_rows.append(output)
                if not item["speaker_id"]:
                    unassigned.append({"segment_id": source_id, "text_start": cursor, "text_end": end})
                run_index += 1
                cursor = end
        rows = rendered_rows
    else:
        source = root / "transcript.speakers.json"
        rows = [
            {
                "id": r.get("id", ""),
                "text": r["text"],
                "speaker": r["speaker"],
                "speaker_id": r.get("speaker_id", ""),
                "offsets": {"from": r["start"] * 1000, "to": r["end"] * 1000},
            }
            for r in speaker_payload["speaker_turns"]
        ]
        unassigned = []
    body = "\n\n".join(
        str(r["speaker"]) + "：" + r["text"] for r in rows if r["text"].strip()
    )
    target, audit = root / "transcript.identity.txt", root / "transcript.identity.json"
    write_json(
        audit,
        {
            "schema_version": 2,
            "annotation_kind": "speaker-attributed-character-spans",
            "recording_id": str(speaker_payload.get("recording_id") or Path(str(speaker_payload.get("source_audio") or "recording")).stem),
            "source_path": str(source),
            "source_sha256": file_hash(source),
            "alignment_source_path": str(raw_path) if raw_path.is_file() else "",
            "alignment_source_sha256": str(speaker_payload.get("transcript_source_sha256") or ""),
            "source_field": "enhanced_transcription" if source.name == "transcript.enhanced.json" else "speaker_turns",
            "source_audio": str(speaker_payload.get("source_audio") or ""),
            "source_audio_sha256": str(speaker_payload.get("source_audio_sha256") or ""),
            "transcription": rows,
            "unmapped_segment_ids": unassigned,
            "text_reconstruction_exact": True,
            "identity_assignment_basis": "speaker_uuid_and_native_character_spans",
            "text_edited": False,
            "status": "partial" if unassigned else "ready",
        },
    )
    atomic_write(target, format_transcript_attribution(body))
    return {
        "identity_text_path": str(target),
        "identity_json_path": str(audit),
        "identity_render_status": "partial" if unassigned else "ready",
    }
