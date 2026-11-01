"""Explicit Native candidate stage; private evidence stays in the item folder."""
from __future__ import annotations

import array
from collections import Counter, defaultdict
import ctypes
import json
from pathlib import Path
import platform
import shutil
import sys
from uuid import uuid4
import wave

from .contracts import PipelineError
from .io_utils import file_hash, value_hash, write_json
from .nemotron_candidate import probe_nemotron, validate_assets, validate_candidate

BACKEND = "nemotron-metal-candidate"
ATTRIBUTION_PROFILES = {"native-padded", "posterior-priority"}
UNCERTAIN_BOUNDARY = "__UNRESOLVED_PADDING_BOUNDARY__"


def binding(settings):
    row = settings.get("nemotron") or {}
    if not isinstance(row, dict):
        raise PipelineError("CONFIG_ERROR", "diarization.nemotron须为配置表")
    profile = row.get("attribution_profile", "native-padded")
    if not isinstance(profile, str) or profile not in ATTRIBUTION_PROFILES:
        raise PipelineError("CONFIG_ERROR", "Nemotron attribution_profile须为native-padded或posterior-priority")
    paths = []
    for key in ("asset_ledger", "runtime_root"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or not Path(value).expanduser().is_absolute():
            raise PipelineError("CONFIG_ERROR", "请显式绑定diarization.nemotron." + key + "的绝对路径")
        paths.append(Path(value).expanduser().resolve())
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise PipelineError("DEPENDENCY_MISSING", "Nemotron固定Metal候选仅支持Apple Silicon macOS", blocked=True)
    return *paths, profile


def posterior_cores(values, cadence):
    """Pinned native hysteresis, without padding/merge/minimum duration."""
    onset, offset = ctypes.c_float(.641).value, ctypes.c_float(.561).value
    turns, frames = [], len(values) // 8
    for channel in range(8):
        active, start = False, 0
        for frame in range(frames):
            value = values[frame * 8 + channel]
            if not active and value > onset:
                active, start = True, frame
            elif active and value < offset:
                active = False
                turns.append({"start": start * cadence, "end": frame * cadence, "speaker": channel + 1})
        if active:
            turns.append({"start": start * cadence, "end": frames * cadence, "speaker": channel + 1})
    return sorted(turns, key=lambda r: (r["start"], r["speaker"]))


def priority_activity(padded, cores):
    """All core voices win over padding; ambiguous padding is not a third voice."""
    events = defaultdict(lambda: [Counter(), Counter()])
    for kind, rows in enumerate((cores, padded)):
        for row in rows:
            events[row["start"]][kind][row["speaker"]] += 1
            events[row["end"]][kind][row["speaker"]] -= 1
    active, grouped, previous = [Counter(), Counter()], defaultdict(list), None
    for time in sorted(events):
        if previous is not None and previous < time:
            core = sorted(s for s, n in active[0].items() if n > 0)
            pad = sorted(s for s, n in active[1].items() if n > 0)
            selected = [(s, False) for s in core] if core else (
                [(pad[0], False)] if len(pad) == 1 else [(UNCERTAIN_BOUNDARY, True)] if pad else [])
            for key in selected:
                ranges = grouped[key]
                if ranges and ranges[-1][1] == previous:
                    ranges[-1][1] = time
                else:
                    ranges.append([previous, time])
        for kind in (0, 1):
            active[kind].update(events[time][kind])
        previous = time
    return sorted([{"start": a, "end": b, "speaker": label, "uncertain": uncertain}
                   for (label, uncertain), ranges in grouped.items() for a, b in ranges],
                  key=lambda r: (r["start"], str(r["speaker"])))


def _label(row):
    label = row["speaker"] if row.get("uncertain") else "native_channel_" + str(row["speaker"])
    return {**row, "speaker": label, "raw_speaker_label": label}


def _relocate_completed_cache(attempt, pointer, previous, *, config, expected, audio, duration):
    """Keep original bytes; rebase only file links in a verified copied view."""
    try:
        runtime = config.get_path("runtime", default={})
        if not isinstance(runtime, dict) or runtime.get("resume_verified") is not True:
            raise ValueError("not a verified resume copy")
        child_run, parent_run = Path(runtime["resume_run_dir"]), Path(runtime["resume_parent_dir"])
        relative = attempt.resolve().relative_to(child_run.resolve())
        source = parent_run.resolve() / relative
        if source.resolve() == attempt.resolve() or not source.resolve().is_relative_to(parent_run.resolve()):
            raise ValueError("invalid parent attempt")
        if file_hash(pointer) != file_hash(source.parent / "attempt.json"):
            raise ValueError("parent pointer changed after copy")
        # Preserve the exact pre-migration native/request/report bytes, not just
        # a claim that the JSON fields are equivalent.
        for name, digest in previous.get("hashes", {}).items():
            if (not (source / name).resolve().is_relative_to(source.resolve())
                    or not (attempt / name).resolve().is_relative_to(attempt.resolve())
                    or file_hash(source / name) != digest or file_hash(attempt / name) != digest):
                raise ValueError("copied evidence differs from parent")
        request = json.loads((source / "request.json").read_text())
        if any(request.get(k) != v for k, v in expected.items()):
            raise ValueError("parent request identity differs")
        native = json.loads((source / "native.json").read_text())
        report = json.loads((source / "probe.result.json").read_text())
        if report["native"] != native or report["status"] != "candidate-probe-complete":
            raise ValueError("parent completion differs")
        # A blocked intermediate resume can contain verified copies whose
        # JSON still points to the grandparent. Resolve only paths under the
        # original declared output root, with hashes from the copied ledger;
        # never read the old external location as evidence for the new view.
        source_view = {}
        origin = Path(request["output"]).parent
        def source_local(value):
            path = Path(value)
            if not path.is_absolute():
                raise ValueError("relative native evidence")
            relative = path.relative_to(origin)
            local_path = source / relative
            digest = previous["hashes"].get(relative.as_posix())
            if (not digest or not local_path.resolve().is_relative_to(source.resolve())
                    or file_hash(local_path) != digest):
                raise ValueError("unverified inherited native path")
            source_view[str(path)] = str(local_path)
            return str(local_path)
        native["probability_path"] = source_local(native["probability_path"])
        native["full_probability_archive"]["path"] = source_local(native["full_probability_archive"]["path"])
        for row in native["checkpoints"]:
            row["path"] = source_local(row["path"])
        validate_candidate(native, source, duration=duration, origin_ms=0,
                           long_context=True, retain_full_probabilities=True)
        copied = attempt
        # Publish the new pointer last. A crash during relocation leaves the
        # copied original untouched and permits a new independent retry.
        attempt = copied.parent / uuid4().hex
        shutil.copytree(copied, attempt)
        originals = {}
        for name in ("native.json", "request.json", "probe.result.json"):
            raw = (source / name).read_bytes()
            if (attempt / name).read_bytes() != raw:
                raise ValueError("copied model JSON differs from parent")
            digest = file_hash(source / name)
            target = attempt / ("source." + name.removesuffix(".json") + "." + digest + ".json")
            if target.exists() and target.read_bytes() != raw:
                raise ValueError("immutable source snapshot differs")
            if not target.exists():
                target.write_bytes(raw)
            originals[name] = {"path": str(target), "sha256": digest}
        moved = {}

        def local(path):
            old = Path(path)
            if not old.is_absolute() or not old.resolve().is_relative_to(source.resolve()):
                raise ValueError("parent data path escapes native attempt")
            new = attempt / old.relative_to(source)
            if file_hash(new) != file_hash(old):
                raise ValueError("copied data bytes differ")
            moved[str(old)] = str(new)
            return str(new)

        native["probability_path"] = local(native["probability_path"])
        native["full_probability_archive"]["path"] = local(native["full_probability_archive"]["path"])
        for row in native["checkpoints"]:
            row["path"] = local(row["path"])
        relocation = {"kind": "verified-copy-path-view-not-new-inference", "source_attempt": str(source),
                      "originals": originals, "data_path_mapping": moved, "verified_source_path_view": source_view}
        native["relocations"] = [*native.get("relocations", []), relocation]
        request.update(audio_path=str(audio.resolve()), output=str(attempt / "native.json"),
                       progress=str(attempt / "worker.state.json"))
        report.update(native=native, relocation=relocation,
                      resource_measurements_scope="original generation; no inference during relocation")
        write_json(attempt / "native.json", native)
        write_json(attempt / "request.json", request)
        write_json(attempt / "probe.result.json", report)
        proof = attempt / ("relocation." + originals["native.json"]["sha256"] + ".json")
        write_json(proof, relocation)
        # Recheck the view using the ordinary probability/timeline validator.
        validate_candidate(native, attempt, duration=duration, origin_ms=0, long_context=True, retain_full_probabilities=True)
        hashes = {name: file_hash(attempt / name) for name in previous.get("hashes", {})}
        for path in [proof, *(Path(r["path"]) for r in originals.values())]:
            hashes[str(path.relative_to(attempt))] = file_hash(path)
        record = {**previous, "directory": attempt.name, "hashes": hashes}
        write_json(pointer, record)
        return record, attempt
    except (OSError, ValueError, KeyError, TypeError, AttributeError, PipelineError) as exc:
        raise PipelineError("CACHE_MISMATCH", "Native恢复副本无法核验或迁移，原尝试保留", blocked=True) from exc


def run_native_stage(audio: Path, folder: Path, *, settings, config, cancel, progress_callback=None):
    ledger, runtime, profile = binding(settings)
    locked, library = validate_assets(ledger, runtime)
    with wave.open(str(audio), "rb") as f:
        duration = f.getnframes() / f.getframerate()
        if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype()) != (1, 2, 16000, "NONE") or not 0 < duration <= 14400:
            raise PipelineError("INVALID_INPUT", "Native候选需要mono16k PCM16，时长须在4小时内")
    code = {name: file_hash(Path(__file__).with_name(name)) for name in
            ("nemotron_candidate.py", "nemotron_worker.py", "nemotron_timeline.py", "nemotron_probabilities.py")}
    fingerprint = value_hash({"audio": file_hash(audio), "ledger": file_hash(ledger), "runtime": str(runtime),
                              "library": str(library), "code": code, "origin_ms": 0, "full_archive": True})
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    pointer = folder / "attempt.json"
    cached, previous = False, None
    if pointer.is_file():
        try:
            previous = json.loads(pointer.read_text())
            attempt = (folder / previous["directory"]).resolve()
            if attempt.parent != folder.resolve():
                raise ValueError("attempt path escapes stage")
            if previous["fingerprint"] != fingerprint:
                previous = None
            elif (attempt / "probe.result.json").is_file():
                cached = True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PipelineError("CACHE_MISMATCH", "Native阶段记录损坏，保留证据并使用新输出目录", blocked=True) from exc
    if cancel():
        raise PipelineError("CANCELLED", "Native阶段已取消")
    if not cached:
        parent = attempt if previous is not None and (attempt / "request.json").is_file() else None
        attempt = folder / uuid4().hex
        write_json(pointer, {"fingerprint": fingerprint, "directory": attempt.name})
        if progress_callback:
            progress_callback({"stage": "speaker", "operation": BACKEND, "cached": False})
        probe_nemotron(audio, attempt, ledger=ledger, runtime_root=runtime, origin_ms=0, cancel=cancel,
                       long_context=True, retain_full_probabilities=True, restart_from=parent, config=config)
    try:
        request = json.loads((attempt / "request.json").read_text())
        expected = {"audio_sha256": file_hash(audio), "model_sha256": locked["model"]["sha256"],
                    "ledger_sha256": file_hash(ledger), "library": str(library), "origin_ms": 0,
                    "worker_sha256": code["nemotron_worker.py"], "timeline_sha256": code["nemotron_timeline.py"],
                    "probabilities_code_sha256": code["nemotron_probabilities.py"],
                    "controller_sha256": code["nemotron_candidate.py"], "long_context": True, "retain_full_probabilities": True}
        native_input = json.loads((attempt / "native.json").read_text())
        if cached and not Path(native_input["probability_path"]).resolve().is_relative_to(attempt.resolve()):
            previous, attempt = _relocate_completed_cache(attempt, pointer, previous, config=config,
                expected=expected, audio=audio, duration=duration)
            request = json.loads((attempt / "request.json").read_text())
            native_input = json.loads((attempt / "native.json").read_text())
        if any(request.get(k) != v for k, v in expected.items()):
            raise ValueError("native request identity changed")
        native = validate_candidate(native_input, attempt,
                                    duration=duration, origin_ms=0, long_context=True, retain_full_probabilities=True)
        report = json.loads((attempt / "probe.result.json").read_text())
        if (report["status"] != "candidate-probe-complete" or report["native"] != native
                or not isinstance(report["resources"], dict)):
            raise ValueError("native completion report changed")
        if previous is not None and cached and "hashes" in previous:
            for name, digest in previous["hashes"].items():
                path = attempt / name
                if not path.resolve().is_relative_to(attempt.resolve()) or file_hash(path) != digest:
                    raise ValueError("cached evidence changed")
    except (OSError, ValueError, KeyError, TypeError, PipelineError) as exc:
        raise PipelineError("CACHE_MISMATCH" if cached else "PROTOCOL_ERROR", "Native完整阶段证据无法复核", blocked=True) from exc
    values = array.array("f", Path(native["full_probability_archive"]["path"]).read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    cores = posterior_cores(values, native["seconds_per_frame"])
    # Native binary cores define recording-local voices independently of which
    # reading projection was selected; an uncertainty sentinel is never a voice.
    acoustic = [_label(row) for row in cores]
    activity = native["segments"] if profile == "native-padded" else priority_activity(native["segments"], cores)
    attribution = [_label(row) for row in activity]
    native_path, raw_path, rttm = attempt / "native.json", attempt / "diarization.raw.json", attempt / "diarization.raw.rttm"
    raw = {"schema_version": 1, "annotation_kind": "acoustic-diarization", "backend": BACKEND,
           "model": "nvidia/Nemotron-3-Diarization", "model_revision": locked["model"]["revision"],
           "model_digest": locked["model"]["sha256"], "audio_sha256": request["audio_sha256"],
           "segmentation_profile": "official-v3-offline-padded", "turns": [_label(r) for r in native["segments"]]}
    rttm_text = "".join(f"SPEAKER audio 1 {r['start']:.9f} {r['end']-r['start']:.9f} <NA> <NA> {r['speaker']} <NA> <NA>\n" for r in raw["turns"])
    try:
        if raw_path.exists() or rttm.exists():
            if not raw_path.is_file() or json.loads(raw_path.read_text()) != raw or not rttm.is_file() or rttm.read_text() != rttm_text:
                raise ValueError("changed or incomplete raw evidence")
        else:
            write_json(raw_path, raw)
            rttm.write_text(rttm_text)
    except (OSError, ValueError) as exc:
        raise PipelineError("CACHE_MISMATCH", "Native不可变声学证据发生改变或未完整写入", blocked=True) from exc
    evidence = {"diarization_raw_json_path": str(raw_path), "diarization_raw_rttm_path": str(rttm),
                "native_candidate_json_path": str(native_path), "native_request_json_path": str(attempt / "request.json"),
                "native_report_json_path": str(attempt / "probe.result.json"),
                "native_probability_path": native["full_probability_archive"]["path"],
                "native_tail_probability_path": native["probability_path"],
                **{f"native_checkpoint_{i:04d}_path": row["path"] for i, row in enumerate(native["checkpoints"])},
                **{f"native_source_{i:04d}_path": str(path) for i, path in enumerate(
                    sorted([*attempt.glob("source.*.json"), *attempt.glob("relocation.*.json")]))}}
    hashes = {str(Path(path).relative_to(attempt)): file_hash(Path(path)) for path in evidence.values()}
    write_json(pointer, {"fingerprint": fingerprint, "directory": attempt.name, "hashes": hashes})
    return {"backend": BACKEND, "turns": acoustic, "attribution_turns": attribution,
            "model": raw["model"], "model_revision": raw["model_revision"], "model_digest": raw["model_digest"],
            "speaker_models": {str(locked["model"]["path"]): raw["model_digest"]},
            "attribution_profile": profile, "voice_evidence": "native-hysteresis-cores-without-padding",
            "release_status": "candidate", "quality_validation": "pending-author-gold", "default_promotion": False,
            "speaker_slot_capacity": 8, "speaker_count_constraint_applied": False, "resume_granularity": "whole-stage",
            "evidence": {"paths": evidence, "hashes": {key: file_hash(Path(path)) for key, path in evidence.items()}},
            "resources": {"cached": cached, "generation_resources": report["resources"]}}
