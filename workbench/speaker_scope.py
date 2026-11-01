"""Recording/stage identifiers, never a cross-recording biometric identity."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
from uuid import UUID, uuid5
import wave

from .contracts import PipelineError
from .io_utils import file_hash, value_hash

SCOPE_VERSION = "recording-stage-uuid-v1"
LEGACY_SCOPE = "legacy-run-local-v1"
_NAMESPACE = UUID("9cf2c21f-6c2c-5f94-83be-ecbd173427bf")
_HASH = re.compile("[0-9a-f]{64}")


def _unverified(payload, reason):
    result = copy.deepcopy(payload)
    result["recording_identity"] = {"version": SCOPE_VERSION, "status": "unverified-legacy", "reason": reason}
    # An unavailable binding cannot leave old IDs looking verified.
    for row in [*result.get("speakers", []), *result.get("speaker_turns", [])]:
        row.pop("speaker_uuid", None)
    return result


def bind_recording_scope(payload: dict, *, source_id: str, audio_sha256: str,
                         audio_frames: int, sample_rate: int) -> dict:
    """Caller verifies the file; deterministic IDs exclude paths/text/names."""
    if not source_id or not isinstance(source_id, str) or not _HASH.fullmatch(audio_sha256):
        raise ValueError("missing source/audio identity")
    activity = payload.get("acoustic_activity") or {}
    if activity.get("annotation_kind") != "acoustic-diarization" or not activity.get("turns"):
        return _unverified(payload, "no-explicit-acoustic-activity")
    rows = []
    for turn in activity["turns"]:
        a, b = turn.get("start"), turn.get("end")
        label = turn.get("raw_speaker_label") or turn.get("raw_speaker") or turn.get("speaker")
        if (type(a) not in (int, float) or type(b) not in (int, float)
                or not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b
                or b > audio_frames / sample_rate + .16 or not isinstance(label, str) or not label):
            raise PipelineError("PROTOCOL_ERROR", "声音UUID需要有限的原生声学区间及真实音频范围")
        rows.append({"raw_label": label, "start": float(a), "end": float(b), "overlap": bool(turn.get("overlap"))})
    rows.sort(key=lambda r: (r["start"], r["end"], r["raw_label"], r["overlap"]))
    provenance = payload.get("provenance") or {}
    models = provenance.get("speaker_models") or {}
    # Only immutable weight IDs; installation directories do not define voices.
    weights = sorted((Path(k).name, v) for k, v in models.items() if isinstance(v, str) and _HASH.fullmatch(v))
    model_identity = {k: payload.get(k) for k in ("backend", "model", "model_revision", "model_digest")}
    model_identity.update(weights=weights, runtime_packages=(provenance.get("runtime") or {}).get("packages", {}))
    acoustic_revision = value_hash([rows, model_identity])
    recording_uuid = str(uuid5(_NAMESPACE, value_hash([source_id, audio_sha256, audio_frames, sample_rate])))
    stage = uuid5(UUID(recording_uuid), acoustic_revision)
    scope = {"version": SCOPE_VERSION, "status": "verified-recording-stage", "source_id": source_id,
             "source_identity_kind": "audio-content-only" if source_id == "audio-sha256:" + audio_sha256 else "source-id-and-audio",
             "model_identity_known": bool(weights or any(payload.get(k) for k in ("backend", "model", "model_revision", "model_digest"))),
             "audio_sha256": audio_sha256, "audio_frames": audio_frames, "sample_rate": sample_rate,
             "timeline": "audio-file-start", "recording_uuid": recording_uuid, "acoustic_revision": acoustic_revision,
             "identity_semantics": "recording-local acoustic cluster; not a registered person"}
    result = copy.deepcopy(payload)
    result["recording_identity"] = scope
    observed = {r["raw_label"] for r in rows}
    by_id = {}
    for person in result.get("speakers", []):
        raw = person.get("raw_labels")
        if not isinstance(raw, list) or not raw or not set(raw).issubset(observed):
            raise PipelineError("PROTOCOL_ERROR", "声音UUID人物引用了未测量的声学簇")
        identity = person["speaker_id"]
        if identity in by_id:
            raise PipelineError("PROTOCOL_ERROR", "声音UUID的局部人物ID重复")
        voice = str(uuid5(stage, value_hash(sorted(raw))))
        person["speaker_uuid"] = voice
        by_id[identity] = voice
    for turn in result.get("speaker_turns", []):
        identity = turn.get("speaker_id")
        if identity:
            if identity not in by_id:
                raise PipelineError("PROTOCOL_ERROR", "声音UUID对话引用未知人物")
            turn["speaker_uuid"] = by_id[identity]
        else:
            turn.pop("speaker_uuid", None)
    return result


def scope_from_files(payload: dict, folder: Path) -> dict:
    if payload.get("schema_name") == "nextecho.speaker-aggregate":
        return payload
    declared = payload.get("recording_identity")
    strict = isinstance(declared, dict) and declared.get("status") == "verified-recording-stage"
    activity = payload.get("acoustic_activity")
    if not isinstance(activity, dict) or activity.get("annotation_kind") != "acoustic-diarization" or not activity.get("turns"):
        if strict:
            raise PipelineError("CACHE_MISMATCH", "声音UUID缺少明确的声学活动来源", blocked=True)
        return _unverified(payload, "no-explicit-acoustic-activity")
    if not isinstance(payload.get("audio_sha256"), str) or not _HASH.fullmatch(payload["audio_sha256"]):
        if strict:
            raise PipelineError("CACHE_MISMATCH", "声音UUID缺少独立音频hash", blocked=True)
        return _unverified(payload, "no-declared-audio-identity")
    audio = next((folder / n for n in ("转录音频.wav", "audio.wav", "audio.full.wav") if (folder / n).is_file()), None)
    if audio is None:
        if strict:
            raise PipelineError("CACHE_MISMATCH", "声音UUID缺少原音频", blocked=True)
        return _unverified(payload, "audio-unavailable")
    try:
        with wave.open(str(audio), "rb") as f:
            frames, rate = f.getnframes(), f.getframerate()
            if not frames or not rate or f.getcomptype() != "NONE":
                raise ValueError("not PCM")
            count = 0
            while raw := f.readframes(65536):
                count += len(raw)
            if count != frames * f.getnchannels() * f.getsampwidth():
                raise ValueError("truncated PCM")
        digest = file_hash(audio)
        if payload.get("audio_sha256") and payload["audio_sha256"] != digest:
            raise PipelineError("CACHE_MISMATCH", "声音UUID的音频与声学来源hash不符", blocked=True)
        metadata = folder / "metadata.json"
        item = json.loads(metadata.read_text()) if metadata.is_file() else {}
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            source_id = "audio-sha256:" + digest
        bound = bind_recording_scope(payload, source_id=source_id, audio_sha256=digest, audio_frames=frames, sample_rate=rate)
    except (OSError, ValueError, TypeError, wave.Error, EOFError) as exc:
        if strict:
            raise PipelineError("CACHE_MISMATCH", "声音UUID来源不可核验", blocked=True) from exc
        return _unverified(payload, "legacy-source-unverifiable")
    if strict:
        if bound["recording_identity"] != declared:
            raise PipelineError("CACHE_MISMATCH", "声音UUID来源或声学版本发生变化", blocked=True)
        for key in ("speakers", "speaker_turns"):
            if [r.get("speaker_uuid") for r in payload.get(key, [])] != [r.get("speaker_uuid") for r in bound.get(key, [])]:
                raise PipelineError("CACHE_MISMATCH", "声音UUID人物/对话绑定改变", blocked=True)
    return bound


def public_speaker_key(entry: dict, local_id: str) -> str:
    keys = entry.get("public_speaker_ids")
    if isinstance(keys, dict):
        if local_id not in keys:
            raise PipelineError("PROTOCOL_ERROR", "当前录音缺少局部到公开声音ID映射")
        return keys[local_id]
    namespace = entry.get("speaker_namespace")
    return namespace + "/" + local_id if namespace else local_id
