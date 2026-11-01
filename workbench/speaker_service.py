"""Shared speaker artifacts and recording views, without HTTP or model loading."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .contracts import PipelineError
from .io_utils import atomic_write, file_hash, write_json
from .speaker_artifacts import is_dialogue, dialogue_to_v3, read_speaker_file
from .speaker_scope import SCOPE_VERSION, LEGACY_SCOPE


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def _write_raw_diarization_artifacts(item_dir: Path, payload: dict) -> None:
    """Persist the first diarization output as an immutable evidence layer."""

    raw_json_path = item_dir / "diarization.raw.json"
    raw_rttm_path = item_dir / "diarization.raw.rttm"
    if raw_json_path.exists() or raw_rttm_path.exists():
        if raw_json_path.exists() and raw_rttm_path.exists():
            return
        raise RuntimeError("incomplete immutable diarization artifacts")
    diarization = (
        payload.get("diarization")
        if isinstance(payload.get("diarization"), dict)
        else {}
    )
    turns = diarization.get("turns") or payload.get("speaker_turns") or []
    metadata = (
        diarization.get("metadata")
        if isinstance(diarization.get("metadata"), dict)
        else {}
    )
    preprocess = (
        payload.get("preprocess") if isinstance(payload.get("preprocess"), dict) else {}
    )
    audio_path = Path(
        str(
            metadata.get("audio_path")
            or preprocess.get("voice_reference_audio")
            or payload.get("source_audio")
            or ""
        )
    )
    raw_turns = []
    rttm_lines: list[str] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        start_ms = max(0, round(float(turn.get("start") or 0) * 1000))
        end_ms = max(start_ms, round(float(turn.get("end") or 0) * 1000))
        raw_label = str(
            turn.get("raw_speaker_label") or turn.get("raw_speaker") or ""
        ).strip()
        if not raw_label:
            raise RuntimeError(
                "cannot reconstruct immutable diarization without an anonymous raw speaker label"
            )
        row = {
            "turn_id": f"turn_{index:05d}",
            "raw_speaker_label": raw_label,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": float(turn.get("confidence") or 0.0),
            "overlap": bool(turn.get("overlap", False)),
        }
        raw_turns.append(row)
        duration = max(0.0, (end_ms - start_ms) / 1000)
        rttm_lines.append(
            f"SPEAKER {item_dir.name} 1 {start_ms / 1000:.3f} {duration:.3f} <NA> <NA> {raw_label} <NA> <NA>"
        )
    raw_payload = {
        "schema_version": 1,
        "backend": str(
            diarization.get("backend") or payload.get("backend") or "unknown"
        ),
        "model": str(
            diarization.get("model")
            or payload.get("diarization_model")
            or payload.get("model")
            or ""
        ),
        "model_revision": str(
            diarization.get("model_revision") or payload.get("model_revision") or ""
        ),
        "model_digest": str(
            diarization.get("model_digest") or payload.get("model_digest") or ""
        ),
        "audio_sha256": _sha256_file(audio_path),
        "turns": raw_turns,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    atomic_write(raw_json_path, json.dumps(raw_payload, ensure_ascii=False, indent=2))
    atomic_write(raw_rttm_path, "\n".join(rttm_lines) + ("\n" if rttm_lines else ""))


def _copy_immutable_artifact(source: Path, target: Path) -> None:
    """Copy a generated evidence artifact only when no immutable target exists."""

    if target.exists() or not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _rewrite_speaker_paths(value: object, *, staging: Path, item_dir: Path) -> object:
    """Remove transient staging paths from persisted speaker JSON."""

    if isinstance(value, dict):
        return {
            key: _rewrite_speaker_paths(child, staging=staging, item_dir=item_dir)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_speaker_paths(child, staging=staging, item_dir=item_dir)
            for child in value
        ]
    if isinstance(value, str):
        prefix = str(staging)
        if (
            value == prefix
            or value.startswith(prefix + os.sep)
            or value.startswith(prefix + "/")
        ):
            return str(item_dir) + value[len(prefix) :]
    return value


def _speaker_payload_v3(payload: dict, *, source_label: str = "") -> dict:
    if is_dialogue(payload):
        return dialogue_to_v3(payload, source_label=source_label)
    if (
        payload.get("schema_name") == "nextecho.speakers"
        and payload.get("schema_version") == 3
    ):
        # Modern exports already carry validated IDs, anonymous identities and
        # explicit unassigned rows. Inferring headers from display text invents
        # a person for those rows and discards the canonical identity contract.
        if not isinstance(payload.get("speakers"), list) or not isinstance(
            payload.get("speaker_turns"), list
        ):
            raise PipelineError("PROTOCOL_ERROR", "v3 speaker需要人物和对话列表")
        return {
            **payload,
            "source_label": source_label or payload.get("source_label", ""),
        }
    if payload.get("schema_version") == 3 and not isinstance(
        payload.get("speakers"), list
    ):
        raise PipelineError("PROTOCOL_ERROR", "v3 speaker需要speakers列表")
    if payload.get("schema_version") == 3 and isinstance(
        payload.get("speaker_turns"), list
    ):
        return {
            **payload,
            "source_label": source_label or payload.get("source_label", ""),
            "coverage": payload.get("coverage")
            or _coverage_for_turns(payload["speaker_turns"]),
        }
    turns = [
        item for item in (payload.get("speaker_turns") or []) if isinstance(item, dict)
    ]
    labels = list(
        dict.fromkeys(
            str(
                item.get("raw_speaker")
                or item.get("speaker_id")
                or item.get("speaker")
                or "Speaker 1"
            )
            for item in turns
        )
    )
    name_map = payload.get("speaker_map") or {}
    inferred = payload.get("speaker_name_inference") or {}
    speakers = []
    existing_speakers = [
        item for item in (payload.get("speakers") or []) if isinstance(item, dict)
    ]
    by_id: dict[str, dict] = {}
    for item in existing_speakers:
        keys = [
            str(item.get("speaker_id") or ""),
            *(str(label) for label in (item.get("raw_labels") or [])),
        ]
        for key in keys:
            if key:
                by_id[key] = item
    for label in labels:
        existing = dict(by_id.get(label) or {})
        display_name = str(
            existing.get("display_name") or name_map.get(label) or ""
        ).strip()
        identity_status = str(
            existing.get("identity_status")
            or ("resolved" if display_name and display_name != label else "unknown")
        )
        evidence = list(existing.get("identity_evidence") or [])
        speakers.append(
            {
                **existing,
                "speaker_id": str(existing.get("speaker_id") or label),
                "raw_labels": existing.get("raw_labels") or [label],
                "display_name": display_name or label,
                "confidence": float(
                    existing.get("confidence")
                    or (0.9 if identity_status == "resolved" else 0.0)
                ),
                "identity_status": identity_status,
                "identity_evidence": evidence,
            }
        )
    coverage = _coverage_for_turns(turns)
    output = dict(payload)
    output.update(
        {
            "schema_version": 3,
            "source_label": source_label or str(payload.get("source_label") or ""),
            "speakers": speakers,
            "coverage": coverage,
            "identity_status": "resolved"
            if any(item["identity_status"] == "resolved" for item in speakers)
            else "unknown",
            "fallback_reason": "backend is best-effort"
            if payload.get("fallback_used")
            else "",
            "speaker_name_inference": inferred,
        }
    )
    return output


def _coverage_for_turns(turns: list[dict]) -> dict:
    starts = [float(item.get("start") or 0) for item in turns if isinstance(item, dict)]
    ends = [float(item.get("end") or 0) for item in turns if isinstance(item, dict)]
    if not starts or not ends:
        return {"start_ms": 0, "end_ms": 0, "duration_ms": 0, "known": False}
    start_ms = max(0, round(min(starts) * 1000))
    end_ms = max(start_ms, round(max(ends) * 1000))
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "known": True,
    }


def _write_speaker_text_files(run_dir: Path, turns: list[dict]) -> None:
    blocks: list[str] = []
    markdown: list[str] = []
    for turn in turns:
        speaker = str(turn.get("speaker") or "Speaker 1")
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        source = str(turn.get("source_label") or "").strip()
        prefix = f"[{source}] " if source else ""
        blocks.append(f"{prefix}{speaker}：{text}")
        start = float(turn.get("start") or 0)
        end = float(turn.get("end") or 0)
        markdown.append(
            f"{prefix}{speaker} [{int(start // 3600):02d}:{int(start % 3600 // 60):02d}:{int(start % 60):02d} - {int(end // 3600):02d}:{int(end % 3600 // 60):02d}:{int(end % 60):02d}]：{text}"
        )
    atomic_write(run_dir / "transcript.speakers.txt", "\n\n".join(blocks))
    atomic_write(run_dir / "transcript.speakers.md", "\n\n".join(markdown))


def _aggregate_speaker_payload(
    run_dir: Path, item_payloads: list[dict], *, write_files: bool = True, identity_scope_version: str = SCOPE_VERSION
) -> dict:
    if identity_scope_version not in {SCOPE_VERSION, LEGACY_SCOPE}:
        raise PipelineError("PROTOCOL_ERROR", "未知的声音汇总身份版本")
    item_views = []
    all_turns: list[dict] = []
    all_speakers: list[dict] = []
    backends: list[str] = []
    any_fallback = False
    has_partial = False
    for item_index, item in enumerate(item_payloads):
        source_label = str(item.get("source_label") or "")
        payload = _speaker_payload_v3(
            dict(item.get("payload") or {}), source_label=source_label
        )
        if identity_scope_version == LEGACY_SCOPE:
            payload.pop("recording_identity", None)
            for key in ("speakers", "speaker_turns"):
                payload[key] = [{k: v for k, v in row.items() if k != "speaker_uuid"}
                                for row in payload.get(key, [])]
        # IDs in acoustic exports are local to a recording. Display names and
        # local spk_00 must not cause unrelated recordings to share a map entry.
        namespace = ""
        public_ids = None
        scope = payload.get("recording_identity") or {}
        if payload.get("schema_name") == "nextecho.speakers" or (
                identity_scope_version == SCOPE_VERSION and payload.get("speakers")):
            from workbench.io_utils import value_hash

            namespace = (
                "src_"
                + value_hash([source_label, payload.get("raw_sha256"), item_index])[:16]
            )
            if identity_scope_version == SCOPE_VERSION and scope.get("status") == "verified-recording-stage":
                namespace = "rec_" + scope["recording_uuid"]
                public_ids = {p["speaker_id"]: namespace + "/" + p["speaker_uuid"] for p in payload["speakers"]}
        item_views.append(
            {
                "source_label": source_label,
                "speaker_namespace": namespace,
                **({"public_speaker_ids": public_ids, "recording_identity": scope} if public_ids is not None else {}),
                "status": payload.get("status", "ready"),
                "payload": payload,
                "paths": item.get("paths") or {},
                "artifact_paths": item.get("artifact_paths") or {},
            }
        )
        for turn in payload.get("speaker_turns") or []:
            if isinstance(turn, dict):
                scoped = {**turn, "source_label": source_label}
                if namespace and turn.get("speaker_id"):
                    scoped.update(
                        local_speaker_id=turn["speaker_id"],
                        speaker_id=public_ids[turn["speaker_id"]] if public_ids is not None else namespace + "/" + turn["speaker_id"],
                    )
                all_turns.append(scoped)
        for speaker in payload.get("speakers") or []:
            if isinstance(speaker, dict):
                scoped = {**speaker, "source_label": source_label}
                if namespace:
                    scoped.update(
                        local_speaker_id=speaker["speaker_id"],
                        speaker_id=public_ids[speaker["speaker_id"]] if public_ids is not None else namespace + "/" + speaker["speaker_id"],
                    )
                all_speakers.append(scoped)
        backend = str(payload.get("backend") or "")
        if backend and backend not in backends:
            backends.append(backend)
        any_fallback = any_fallback or bool(payload.get("fallback_used"))
        has_partial = has_partial or payload.get("status") == "partial"
    aggregate = {
        "schema_name": "nextecho.speaker-aggregate",
        "schema_version": 3,
        **({"identity_scope_version": identity_scope_version} if identity_scope_version != LEGACY_SCOPE else {}),
        "status": "partial" if has_partial else "ready",
        "items": item_views,
        "speaker_turns": all_turns,
        "speakers": all_speakers,
        "speaker_count_detected": len(all_speakers),
        "speaker_map": {
            person["speaker_id"]: person["display_name"]
            for person in all_speakers
            if person.get("identity_status") == "resolved"
        },
        "backend": backends[0] if len(backends) == 1 else ("mixed" if backends else ""),
        "backends": backends,
        "fallback_used": any_fallback,
        "quality_note": "本地说话人分离结果，建议人工复核"
        if any_fallback
        else "说话人分离已完成",
        "coverage": _coverage_for_turns(all_turns),
        "paths": {
            "json": str(run_dir / "transcript.speakers.json"),
            "txt": str(run_dir / "transcript.speakers.txt"),
            "md": str(run_dir / "transcript.speakers.md"),
        },
    }
    keys = [p["speaker_id"] for p in all_speakers]
    if any(view["payload"].get("release_status") == "candidate" for view in item_views):
        aggregate.update(release_status="candidate", quality_validation="pending-author-gold",
                         quality_note="候选分人已执行，声音归属及姓名仍需作者审核")
    if identity_scope_version == SCOPE_VERSION and len(keys) != len(set(keys)):
        raise PipelineError("PROTOCOL_ERROR", "声音汇总包含重复的同一录音身份，拒绝歧义映射")
    if write_files:
        atomic_write(
            run_dir / "transcript.speakers.json",
            json.dumps(aggregate, ensure_ascii=False, indent=2),
        )
        _write_speaker_text_files(run_dir, all_turns)
        write_json(run_dir / "speaker_map.json", aggregate["speaker_map"])
    return aggregate


def relocated_speaker_view(run_dir: Path, payload: dict) -> dict:
    """Read a copied aggregate through local canonical files, without writes."""
    entries = []
    for entry in payload.get("items") or []:
        old = Path((entry.get("paths") or {}).get("json") or "")
        if not old.is_absolute() or old.name != "transcript.speakers.json":
            raise PipelineError("CACHE_MISMATCH", "说话人副本缺少明确条目路径", blocked=True)
        folder = run_dir / "items" / old.parent.name
        if not folder.resolve().is_relative_to((run_dir / "items").resolve()):
            raise PipelineError("CACHE_MISMATCH", "说话人副本路径越界", blocked=True)
        entries.append(_speaker_item_payload(folder, entry.get("source_label", "")))
    local = _aggregate_speaker_payload(run_dir, entries, write_files=False,
        identity_scope_version=payload.get("identity_scope_version", LEGACY_SCOPE))
    for key in ("speakers", "speaker_turns", "speaker_count_detected"):
        if local.get(key) != payload.get(key):
            raise PipelineError("CACHE_MISMATCH", "说话人副本身份与原汇总不一致", blocked=True)
    return local


def _speaker_item_payload(item_dir: Path, source_label: str) -> dict:
    json_path = item_dir / "transcript.speakers.json"
    if not json_path.exists():
        return {"source_label": source_label, "payload": {}, "paths": {}}
    try:
        payload = read_speaker_file(json_path)
    except (OSError, json.JSONDecodeError):
        payload = {}
    return {
        "source_label": source_label,
        "payload": payload if isinstance(payload, dict) else {},
        "paths": {
            "json": str(json_path),
            "txt": str(item_dir / "transcript.speakers.txt"),
            "md": str(item_dir / "transcript.speakers.md"),
        },
        "artifact_paths": speaker_artifact_paths(item_dir, payload=payload),
    }


def speaker_artifact_paths(item_dir: Path, *, payload=None) -> dict:
    names = {
        "speaker_json_path": "transcript.speakers.json",
        "speaker_text_path": "transcript.speakers.txt",
        "speaker_markdown_path": "transcript.speakers.md",
        "speaker_map_path": "speaker_map.json",
        "dialogue_json_path": "transcript.dialogue.json",
        "dialogue_text_path": "transcript.dialogue.txt",
        "voice_linking_json_path": "speaker-stages/voice-linking.json",
        "diarization_raw_json_path": "diarization.raw.json",
        "diarization_raw_rttm_path": "diarization.raw.rttm",
    }
    result = {
        key: str((item_dir / name).resolve())
        for key, name in names.items()
        if (item_dir / name).is_file()
    }
    evidence = (payload or {}).get("acoustic_evidence")
    if evidence is not None:
        result.pop("voice_linking_json_path", None)
        try:
            paths, hashes = evidence["paths"], evidence["hashes"]
            if not paths or set(paths) != set(hashes):
                raise ValueError("missing evidence hashes")
            for key, raw in paths.items():
                path = Path(raw)
                if not key.endswith("_path") or not path.is_absolute():
                    raise ValueError("invalid evidence path")
                if not path.resolve().is_relative_to(item_dir.resolve()):
                    # Immutable dialogue files retain their creation paths.
                    # A derived run may use only the exact relative copy in
                    # its own item, verified against the canonical digest.
                    # Never follow the external pointer or guess by basename.
                    parts = path.parts
                    offsets = [i + 2 for i in range(len(parts) - 1)
                               if parts[i:i + 2] == ("items", item_dir.name)]
                    if len(offsets) != 1:
                        raise ValueError("unbound external evidence")
                    path = item_dir.joinpath(*parts[offsets[0]:])
                if (not path.resolve().is_relative_to(item_dir.resolve())
                        or not path.is_file() or file_hash(path) != hashes[key]):
                    raise ValueError("changed or external evidence")
                result[key] = str(path.resolve())
        except (TypeError, KeyError, ValueError, OSError, AttributeError) as exc:
            raise PipelineError("CACHE_MISMATCH", "speaker声学证据缺失、改变或越界", blocked=True) from exc
    return result


def fatal_speaker_error(exc: Exception) -> bool:
    return isinstance(exc, PipelineError) and exc.code not in {
        "DEPENDENCY_MISSING",
        "MODEL_MISSING",
        "NEEDS_ALIGNER",
    }


def resolve_item_dir(run_dir: Path, item: dict) -> Path:
    raw = item.get("output_dir") or item.get("item_dir")
    path = (
        Path(raw).expanduser()
        if raw
        else Path("items") / str(item.get("source_label") or "")
    )
    path = (path if path.is_absolute() else run_dir / path).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise PipelineError("PROTOCOL_ERROR", "speaker条目目录越界")
    return path


def build_item_speakers(
    item_dir: Path,
    item: dict,
    *,
    config=None,
    backend="auto",
    min_speakers=None,
    max_speakers=None,
    progress_callback=None,
    legacy_builder=None,
) -> dict:
    """Use native anchors for coarse ASR; isolate raw inputs for old backends."""
    from .config import load_config

    config = config or load_config()
    if backend == "auto":
        backend = config.get_path("diarization", "backend", default="auto")
    if not isinstance(backend, str) or not backend:
        raise PipelineError("CONFIG_ERROR", "diarization.backend必须为有效字符串")
    item_dir = item_dir.resolve()
    protected = {
        path: file_hash(path)
        for name in (
            "transcript.json",
            "transcript.enhanced.json",
            "transcript.enhanced.txt",
        )
        if (path := item_dir / name).is_file()
    }
    asr = item.get("asr") or {}
    coarse = (
        asr.get("actual_engine") in {"fun-asr-nano", "sensevoice"}
        and asr.get("timing_precision") != "precise"
    )
    try:
        if backend == "nemotron-metal-candidate":
            from .acoustic_dialogue import build_acoustic_dialogue

            result = build_acoustic_dialogue(item_dir, config=config,
                progress_callback=progress_callback, backend=backend)
        elif coarse and backend in {"auto", "sherpa-onnx"}:
            from .acoustic_dialogue import build_acoustic_dialogue

            result = build_acoustic_dialogue(
                item_dir, config=config, progress_callback=progress_callback
            )
        else:
            result = _build_legacy_speakers(
                item_dir,
                config=config,
                backend=backend,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                progress_callback=progress_callback,
                legacy_builder=legacy_builder,
            )
        if result.get("status") not in {"ready", "partial"}:
            raise PipelineError(
                "PROTOCOL_ERROR", "speaker后端未返回明确的ready/partial状态"
            )
        view = _speaker_item_payload(
            item_dir, str(item.get("source_label") or item_dir.name)
        )
        if not view["payload"] or view["payload"].get("status") != result["status"]:
            raise PipelineError("PROTOCOL_ERROR", "speaker后端返回值与实际产物不一致")
        required = {
            "speaker_json_path",
            "speaker_text_path",
            "speaker_markdown_path",
            "speaker_map_path",
        }
        if not required.issubset(view["artifact_paths"]):
            raise PipelineError(
                "PROTOCOL_ERROR", "speaker后端缺少必要JSON/TXT/MD/map产物"
            )
        scope = view["payload"].get("recording_identity") or {}
        if scope.get("status") == "verified-recording-stage":
            # Persist only the derived export, never the acoustic/raw dialogue.
            write_json(item_dir / "transcript.speakers.json", view["payload"])
        item["speaker_recording_identity"] = scope
        for key in list(item):
            if (key == "voice_linking_json_path" or key.startswith("native_") and key.endswith("_path")) and key not in view["artifact_paths"]:
                item.pop(key)
        item.update(view["artifact_paths"])
        item["speaker_status"] = result["status"]
        if "alignment_audit" in result:
            item["speaker_alignment"] = result["alignment_audit"]
        write_json(item_dir / "metadata.json", item)
        return view
    finally:
        if any(
            not path.is_file() or file_hash(path) != digest
            for path, digest in protected.items()
        ):
            raise PipelineError(
                "CACHE_MISMATCH", "speaker阶段改变了原始稿或既有采用稿", blocked=True
            )


def _build_legacy_speakers(
    item_dir,
    *,
    config,
    backend,
    min_speakers,
    max_speakers,
    progress_callback,
    legacy_builder,
):
    from .bounded_speaker import bounded_build_speaker_transcript

    builder = legacy_builder or bounded_build_speaker_transcript
    audio = next(
        (p for p in (item_dir / "转录音频.wav", item_dir / "audio.wav") if p.is_file()),
        None,
    )
    if audio is None:
        raise PipelineError("SOURCE_UNAVAILABLE", "speaker阶段缺少原声WAV")
    with tempfile.TemporaryDirectory(
        prefix=".speaker-staging-", dir=item_dir
    ) as temporary:
        stage = Path(temporary)
        shutil.copy2(item_dir / "transcript.json", stage / "transcript.json")
        try:
            (stage / "转录音频.wav").symlink_to(audio)
        except OSError:
            shutil.copy2(audio, stage / "转录音频.wav")
        if (item_dir / "source.context.json").is_file():
            shutil.copy2(
                item_dir / "source.context.json", stage / "source.context.json"
            )
        result = builder(
            stage,
            config=config,
            backend=backend,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            progress_callback=progress_callback,
        )
        payload = _speaker_payload_v3(result, source_label=item_dir.name)
        for index, turn in enumerate(payload.get("speaker_turns") or []):
            turn.setdefault("id", f"legacy-dialogue-{index:06d}")
        payload.update(run_dir=str(item_dir), source_audio=str(audio))
        payload = _rewrite_speaker_paths(payload, staging=stage, item_dir=item_dir)
        # Only actual acoustic artifacts may be registered as raw evidence.
        # Do not reconstruct a raw track from text-aligned reading rows.
        for name in ("diarization.raw.json", "diarization.raw.rttm"):
            if not (stage / name).is_file():
                raise PipelineError(
                    "PROTOCOL_ERROR", "旧speaker后端未保存实际声学原始产物"
                )
            _copy_immutable_artifact(stage / name, item_dir / name)
        reference = Path("derived/audio.voice_reference.wav")
        _copy_immutable_artifact(stage / reference, item_dir / reference)
        write_json(stage / "transcript.speakers.json", payload)
        if not all(
            (stage / name).is_file()
            for name in ("transcript.speakers.txt", "transcript.speakers.md")
        ):
            _write_speaker_text_files(stage, payload.get("speaker_turns") or [])
        if not (stage / "speaker_map.json").is_file():
            write_json(stage / "speaker_map.json", payload.get("speaker_map") or {})
        for name in (
            "transcript.speakers.json",
            "transcript.speakers.txt",
            "transcript.speakers.md",
            "speaker_map.json",
        ):
            (stage / name).replace(item_dir / name)
        return payload


def build_run_speakers(
    run_dir: Path,
    *,
    mode="auto",
    config=None,
    backend="auto",
    min_speakers=None,
    max_speakers=None,
    progress_callback=None,
    legacy_builder=None,
):
    if mode == "off":
        return None
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    views, failures = [], []
    for item in manifest["results"]:
        if item.get("status") not in {"ready", "cached", "partial"} or not item.get(
            "char_count", 1
        ):
            continue
        folder = resolve_item_dir(run_dir, item)
        try:
            views.append(
                build_item_speakers(
                    folder,
                    item,
                    config=config,
                    backend=backend,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    progress_callback=progress_callback,
                    legacy_builder=legacy_builder,
                )
            )
        except Exception as exc:
            if mode == "required" or fatal_speaker_error(exc):
                raise
            item["speaker_status"] = "partial"
            item["speaker_error"] = {
                "code": getattr(exc, "code", "SPEAKER_FAILED"),
                "message": str(exc),
            }
            failures.append(
                {"source_label": item.get("source_label", ""), **item["speaker_error"]}
            )
            write_json(folder / "metadata.json", item)
    aggregate = _aggregate_speaker_payload(run_dir, views) if views else None
    if aggregate is not None:
        if failures:
            aggregate.update(status="partial", failures=failures)
            write_json(run_dir / "transcript.speakers.json", aggregate)
        for item in manifest["results"]:
            if item.get("status") in {"ready", "cached", "partial"}:
                item["aggregate_speaker_json_path"] = str(
                    run_dir / "transcript.speakers.json"
                )
                item["aggregate_speaker_text_path"] = str(
                    run_dir / "transcript.speakers.txt"
                )
                item["aggregate_speaker_markdown_path"] = str(
                    run_dir / "transcript.speakers.md"
                )
                item["aggregate_speaker_map_path"] = str(run_dir / "speaker_map.json")
                break
    manifest["speaker_transcript"] = aggregate or {
        "status": "partial" if failures else "not_started",
        "failures": failures,
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest["speaker_transcript"]
