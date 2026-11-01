"""Measured acoustic/CTC dialogue with conservative, auditable identity mapping."""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from .contracts import PipelineError
from .media_commands import run_media_command
from .dialogue_alignment import align_dialogue_segments
from .io_utils import atomic_write, file_hash, value_hash, write_json
from .speaker_identity import DefaultSpeakerIdentityResolver

DIALOGUE_VERSION = "native-ctc-dialogue-v3-artifacts"


def _nonspeech_marker(text: str) -> bool:
    return bool(re.fullmatch(r"\s*\[(?:noise|music|silence|applause|laughter|音乐|噪声|静音)\]\s*", text, flags=re.I))


def build_dialogue_payload(raw_segments: list[dict], segments: list[dict], tokens: list[dict],
                           acoustic_turns: list[dict], context: dict, *, uncertain_labels: set | None = None,
                           attribution_turns: list[dict] | None = None) -> dict:
    """All word times are native CTC evidence; none are interpolated."""
    identities_input = []
    for segment in raw_segments:
        words = [t for t in tokens if segment["start_ms"] <= t["start_ms"] < segment["end_ms"]]
        identities_input.append({**segment, "words": words})
    uncertain_labels = uncertain_labels or set()
    identities = DefaultSpeakerIdentityResolver().resolve(
        [t for t in acoustic_turns if str(t.get("raw_speaker_label") or t["speaker"]) not in uncertain_labels],
        identities_input, context=context)
    speakers = identities.by_raw_label()
    labels = {label: (s.display_name if s.identity_status == "resolved" else f"说话人 {int(s.speaker_id.split('_')[-1]) + 1}")
              for label, s in speakers.items()}
    # Cluster labels remain global for the entire audio, not independently
    # numbered in each window. Metadata names never change acoustic labels.
    measured_turns = [{"start": t["start"], "end": t["end"],
                       "overlap": bool(t.get("overlap")),
                       "uncertain": bool(t.get("uncertain")) or str(t.get("raw_speaker_label") or t["speaker"]) in uncertain_labels,
                       "speaker": str(t.get("raw_speaker_label") or t["speaker"])}
                      for t in (attribution_turns if attribution_turns is not None else acoustic_turns)]
    # Name uncertainty must never erase an independently measured voice.
    aligned, audit = align_dialogue_segments(segments, tokens, measured_turns, prefix="dialogue")
    rows = []
    for row in aligned:
        label = row["speaker"]
        person = speakers.get(label)
        display = labels.get(label, label)
        if label.startswith("交叠发言"):
            for raw in sorted(labels, key=len, reverse=True):
                display = display.replace(raw, labels[raw])
        rows.append({**row, "speaker": display, "raw_speaker": label if person else "",
                     "speaker_id": person.speaker_id if person else "",
                     "identity_status": person.identity_status if person else "unassigned",
                     "identity_evidence": [e.to_dict() for e in person.identity_evidence] if person else [],
                     "offsets": {"from": row["start_ms"], "to": row["end_ms"]}})
    assert "".join(r["text"] for r in rows) == "".join(s["text"] for s in segments)
    uncertain = sum(len(r["text"].strip()) for r in rows
                    if r["identity_status"] == "unassigned" and not _nonspeech_marker(r["text"]))
    return {"schema_version": 1, "version": DIALOGUE_VERSION,
            "status": "partial" if uncertain else "ready", "unassigned_text_chars": uncertain,
            "transcription": rows, "speakers": identities.to_dict(), "speaker_labels": labels,
            "speaker_turns": [{**r, "start": r["start_ms"] / 1000, "end": r["end_ms"] / 1000} for r in rows],
            "alignment_audit": audit, "timing_precision": "review-envelope",
            "annotation_kind": "text-review-envelope", "scoring_eligible": False,
            "acoustic_activity": {"annotation_kind": "acoustic-diarization", "turns": acoustic_turns},
            "identity_policy": "Direct acoustic evidence only; unknown names retain stable anonymous labels."}


def write_dialogue_outputs(item_dir: Path, payload: dict) -> dict:
    from .transcription import format_transcript_attribution
    rows, blocks = payload["transcription"], []
    for row in rows:
        if _nonspeech_marker(row["text"]):
            continue
        if blocks and blocks[-1]["speaker"] == row["speaker"] and row["start_ms"] - blocks[-1]["end_ms"] <= 1200:
            blocks[-1]["text"] += row["text"]
            blocks[-1]["end_ms"] = max(blocks[-1]["end_ms"], row["end_ms"])
        else:
            blocks.append(dict(row))
    text = "\n\n".join(r["speaker"] + "：" + r["text"] for r in blocks if r["text"].strip())
    paths = {"dialogue_json_path": str(item_dir / "transcript.dialogue.json"),
             "dialogue_text_path": str(item_dir / "transcript.dialogue.txt"),
             "speaker_json_path": str(item_dir / "transcript.speakers.json")}
    if payload.get("backend") != "nemotron-metal-candidate" and (item_dir / "speaker-stages/voice-linking.json").is_file():
        paths["voice_linking_json_path"] = str(item_dir / "speaker-stages/voice-linking.json")
    write_json(Path(paths["dialogue_json_path"]), payload)
    from .speaker_artifacts import dialogue_to_v3, write_v3_outputs
    exported = dialogue_to_v3(payload)
    exported["canonical_sha256"] = file_hash(Path(paths["dialogue_json_path"]))
    paths.update(write_v3_outputs(item_dir, exported))
    atomic_write(Path(paths["dialogue_text_path"]), format_transcript_attribution(text))
    write_json(item_dir / "dialogue.integrity.json", {
        "sha256": file_hash(Path(paths["dialogue_json_path"])),
        "implementation": _implementation_hashes()})
    return paths


def _implementation_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ("acoustic_dialogue.py", "dialogue_alignment.py", "speaker_identity.py", "acoustic_chunks.py",
             "voice_linking.py", "dialogue_worker.py", "diarization.py", "speaker_artifacts.py",
             "nemotron_speaker.py", "nemotron_candidate.py", "nemotron_worker.py",
             "nemotron_timeline.py", "nemotron_probabilities.py")}


def _validate_stage(data, operation):
    try:
        clips = data["clips"]
        if data["schema_version"] != 1 or len(clips) != 1 or clips[0]["id"] != "full":
            raise ValueError("wrong clip")
        rows = clips[0]["tokens"] if operation == "ctc" else clips[0]["diarization"]["turns"]
        if not isinstance(rows, list):
            raise ValueError("wrong rows")
        previous = -1
        for row in rows:
            start = row["start_ms"] if operation == "ctc" else row["start"]
            end = row.get("end_ms") if operation == "ctc" else row["end"]
            if not math.isfinite(start) or start < 0 or start < previous:
                raise ValueError("invalid start")
            if end is not None and (not math.isfinite(end) or end < start):
                raise ValueError("invalid end")
            previous = start
            if operation == "ctc" and not isinstance(row["text"], str):
                raise ValueError("invalid token")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise PipelineError("PROTOCOL_ERROR", "说话人阶段输出协议或测量时间无效") from exc


def _relink_measured_chunks(folder: Path, acoustic: dict, signature: str, threshold: float) -> dict:
    from .voice_linking import link_acoustic_measurements
    measurements, hashes = [], {}
    frames = round(acoustic["audio_seconds"]*16000)
    try:
        for index in range(acoustic["chunk_count"]):
            path = folder / "acoustic-chunks" / f"{index:04d}.json"
            digest = file_hash(path)
            if path.with_suffix(".sha256").read_text().strip() != digest:
                raise ValueError("chunk checksum")
            data = json.loads(path.read_text())
            start, end = round(data["core_start"]*16000), round(data["core_end"]*16000)
            expected = value_hash([signature, max(0,start-32000), min(frames,end+32000), start,end])
            if data["signature"] != expected or data["index"] != index:
                raise ValueError("chunk identity")
            measurements.append(data)
            hashes[path.name] = digest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PipelineError("CACHE_MISMATCH", "声学块缺失或校验失败，请恢复说话人阶段重新测量坏块", blocked=True) from exc
    result = link_acoustic_measurements(measurements, threshold=threshold)
    result["measurement_hashes"] = hashes
    write_json(folder / "voice-linking.json", result)
    return {**acoustic, **result, "measurement_stage_sha256": file_hash(folder / "diarize-global.json")}


def load_dialogue_segments(item_dir: Path, raw_hash: str) -> tuple[list[dict] | None, dict]:
    path = item_dir / "transcript.dialogue.json"
    if not path.is_file():
        return None, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        integrity = json.loads((item_dir / "dialogue.integrity.json").read_text())
        if (data.get("raw_sha256") != raw_hash or data.get("version") != DIALOGUE_VERSION or
                integrity.get("sha256") != file_hash(path) or integrity.get("implementation") != _implementation_hashes()):
            raise ValueError("derived integrity mismatch")
        if not isinstance(data.get("transcription"), list):
            raise ValueError("invalid rows")
    except (OSError, ValueError, TypeError) as exc:
        raise PipelineError("CACHE_MISMATCH", "对话派生层与原稿、代码或内容校验不匹配，请重新执行说话人阶段", blocked=True) from exc
    from .speaker_artifacts import mapped_dialogue
    data = mapped_dialogue(item_dir, data)
    return data["transcription"], data


def build_acoustic_dialogue(item_dir: Path, *, config, progress_callback=None,
                            audio_path: Path | None = None, adopted_segments: list[dict] | None = None,
                            backend: str = "sherpa-onnx") -> dict:
    """Resource-controlled global diarization then streaming native CTC anchors."""
    from .diarization import _sherpa_model_paths
    from .job_store import current_cancelled
    from .postprocess import load_segments
    from .resources import GIB, model_policy, run_owned, serialized_task

    @serialized_task
    def execute():
        settings = config.get_path("diarization", default={})
        from .nemotron_speaker import BACKEND, binding, run_native_stage
        native_candidate = backend == BACKEND
        if native_candidate:
            binding(settings)
        model_paths = [] if native_candidate else _sherpa_model_paths(settings.get("sherpa") or {})
        ctc_root = Path(str(settings.get("ctc_model_path") or config.get_path("models", "sensevoice-int8", "path", default=""))).expanduser()
        from .capabilities import resolve_worker_python
        python = Path(resolve_worker_python(settings.get("worker_python")) or "/nonexistent/nextecho-python")
        if not python.is_file():
            raise PipelineError("DEPENDENCY_MISSING", "请为说话人阶段绑定diarization.worker_python的完整路径", blocked=True)
        try:
            probe = run_media_command([str(python), "-B", "-c",
                "import importlib.metadata as m,json,sys;print(json.dumps({'python':sys.version,'packages':{n:m.version(n) for n in ['sherpa-onnx','soundfile','numpy']}}))"],
                capture_output=True, text=True, timeout=10, stage="speaker-probe", check=False, config=config)
            if probe.returncode:
                raise ValueError("worker environment unavailable")
            runtime = json.loads(probe.stdout)
            if not isinstance(runtime, dict) or not isinstance(runtime.get("packages"), dict):
                raise ValueError("invalid environment response")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise PipelineError("DEPENDENCY_MISSING", "说话人worker不可执行、探测超时或缺少sherpa-onnx/soundfile/numpy，请检查绑定", blocked=True) from exc
        if not native_candidate and (len(model_paths) != 2 or not all(p.is_file() for p in model_paths)):
            raise PipelineError("MODEL_MISSING", "说话人segmentation/embedding权重未绑定", blocked=True)
        if not all((ctc_root / name).is_file() for name in ("model.int8.onnx", "tokens.txt")):
            raise PipelineError("NEEDS_ALIGNER", "请绑定diarization.ctc_model_path到已准备的SenseVoice模型目录", blocked=True)
        source = item_dir / "transcript.json"
        metadata_path = item_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        audio = audio_path or Path(metadata.get("audio_path") or item_dir / "audio.wav")
        if not audio.is_file():
            raise PipelineError("SOURCE_UNAVAILABLE", "缺少说话人阶段的原声WAV")
        folder = item_dir / "speaker-stages"
        folder.mkdir(parents=True, exist_ok=True)
        worker = Path(__file__).with_name("dialogue_worker.py")
        models = {str(p): file_hash(p) for p in model_paths}
        audio_hash, raw_hash = file_hash(audio), file_hash(source)
        strategy = {"strategy": settings.get("strategy", "bounded-prototypes-v1"),
                    "chunk_seconds": int(settings.get("chunk_seconds", 600)),
                    "num_threads": int(settings.get("num_threads", 4)),
                    "global_voice_threshold": float(settings.get("global_voice_threshold", .8))}
        chunk_code = file_hash(Path(__file__).with_name("acoustic_chunks.py"))
        measurement_signature = value_hash({"audio": audio_hash, "models": models, "runtime": runtime,
            "worker": file_hash(worker), "chunk_code": chunk_code,
            "adapter": file_hash(Path(__file__).with_name("diarization.py")),
            "parameters": {k:v for k,v in strategy.items() if k != "global_voice_threshold"},
            "num_clusters": int(settings.get("num_clusters", -1)), "threshold": float(settings.get("cluster_threshold", .5))})
        clips = [{"id": "full", "audio": str(audio.resolve())}]
        resources = {}

        def stage(operation, model_settings, estimate, maximum, *, force=False):
            output, cache = folder / (operation + ".json"), folder / (operation + ".cache.json")
            fingerprint = value_hash({"audio": audio_hash, "operation": operation, "worker": file_hash(worker),
                "runtime": runtime, "diarization_code": file_hash(Path(__file__).with_name("diarization.py")) if operation == "diarize-global" else None,
                **model_settings})
            if not force and output.is_file() and cache.is_file():
                try:
                    old = json.loads(cache.read_text())
                    if not isinstance(old, dict):
                        old = {}
                except (OSError, ValueError):
                    old = {}
                if old.get("fingerprint") == fingerprint and old.get("sha256") == file_hash(output):
                    cached = json.loads(output.read_text())
                    _validate_stage(cached, operation)
                    resources[operation] = {"cached": True, "generation_resources": old["resources"]}
                    return cached
            request = folder / (operation + ".request.json")
            write_json(request, {"operation": operation, "clips": clips, "output": str(output),
                       "model_paths": [str(p) for p in model_paths], "model": str(ctc_root),
                       **strategy, "measurement_signature": measurement_signature,
                       "num_clusters": int(settings.get("num_clusters", -1)),
                       "cluster_threshold": float(settings.get("cluster_threshold", 0.5))})
            policy = model_policy(estimate, maximum, float(settings.get("timeout_seconds", 7200)), config)
            curve = run_owned([str(python), "-B", str(worker), str(request)], log_path=folder / (operation + ".log"),
                              policy=policy, cancel=current_cancelled,
                              on_sample=(lambda _: progress_callback({"stage": "speaker", "operation": operation})) if progress_callback else None)
            if not output.is_file() or output.stat().st_size > 64 * 1024**2:
                raise PipelineError("PROTOCOL_ERROR", "说话人阶段结果缺失或超出预算")
            data = json.loads(output.read_text())
            _validate_stage(data, operation)
            write_json(cache, {"fingerprint": fingerprint, "sha256": file_hash(output), "resources": curve})
            resources[operation] = {"cached": False, "generation_resources": curve}
            return data

        acoustic_settings = {"models": models, "num_clusters": int(settings.get("num_clusters", -1)),
                           "threshold": float(settings.get("cluster_threshold", 0.5)), **strategy,
                           "chunk_code": chunk_code, "linking_code": file_hash(Path(__file__).with_name("voice_linking.py"))}
        estimate, maximum = max(2 * GIB, audio.stat().st_size * 4 + GIB), max(6 * GIB, audio.stat().st_size * 8)
        native = run_native_stage(audio, folder / "nemotron", settings=settings, config=config,
                                  cancel=current_cancelled, progress_callback=progress_callback) if native_candidate else None
        diarization = None if native_candidate else stage("diarize-global", acoustic_settings, estimate, maximum)
        ctc = stage("ctc", {"model": file_hash(ctc_root / "model.int8.onnx"), "tokens": file_hash(ctc_root / "tokens.txt")}, GIB, 3 * GIB)
        raw = load_segments(json.loads(source.read_text()))
        raw_rows = [{"id": r.id, "text": r.text, "start_ms": r.start_ms, "end_ms": r.end_ms} for r in raw]
        context_path = item_dir / "source.context.json"
        context = json.loads(context_path.read_text()) if context_path.is_file() else {}
        acoustic = native if native_candidate else diarization["clips"][0]["diarization"]
        if acoustic.get("backend") == "sherpa-bounded-acoustic-prototypes":
            try:
                acoustic = _relink_measured_chunks(folder, acoustic, measurement_signature, strategy["global_voice_threshold"])
            except PipelineError as exc:
                if exc.code != "CACHE_MISMATCH":
                    raise
                # A valid aggregate cannot hide a damaged private measurement.
                # Retry the worker once; it independently reuses all good chunks.
                diarization = stage("diarize-global", acoustic_settings, estimate, maximum, force=True)
                acoustic = _relink_measured_chunks(folder, diarization["clips"][0]["diarization"],
                                                    measurement_signature, strategy["global_voice_threshold"])
        weak = {g["label"] for g in (acoustic.get("global_linking") or {}).get("groups", []) if g.get("insufficient_embedding")}
        payload = build_dialogue_payload(raw_rows, adopted_segments if adopted_segments is not None else raw_rows, ctc["clips"][0]["tokens"],
                    acoustic["turns"], context, uncertain_labels=weak, attribution_turns=acoustic.get("attribution_turns"))
        if native_candidate:
            models = acoustic["speaker_models"]
            resources[BACKEND] = acoustic["resources"]
            payload.update(**{k: acoustic[k] for k in ("backend", "model", "model_revision", "model_digest",
                             "release_status", "quality_validation", "default_promotion", "attribution_profile")},
                           acoustic_evidence=acoustic["evidence"])
        payload.update(raw_sha256=raw_hash, audio_sha256=audio_hash, resources=resources,
                       acoustic_audit={**{k:v for k,v in acoustic.items() if k not in {"turns", "attribution_turns", "evidence", "resources"}}, "unverified_global_labels": sorted(weak)},
                       text_layer="enhanced" if adopted_segments is not None else "raw",
                       text_layer_sha256=value_hash(adopted_segments if adopted_segments is not None else raw_rows),
                       provenance={"speaker_models": models, "ctc_model": file_hash(ctc_root / "model.int8.onnx"),
                                   "runtime": runtime,
                                   "alignment_code": file_hash(Path(__file__).with_name("dialogue_alignment.py")),
                                   "identity_code": file_hash(Path(__file__).with_name("speaker_identity.py"))})
        if file_hash(audio) != audio_hash or file_hash(source) != raw_hash:
            raise PipelineError("CACHE_MISMATCH", "speaker阶段原声或原稿发生改变", blocked=True)
        paths = write_dialogue_outputs(item_dir, payload)
        return {**payload, "paths": paths}
    return execute()
