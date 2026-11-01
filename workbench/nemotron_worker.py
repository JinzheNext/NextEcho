"""Isolated v0.2.0 C-ABI candidate; no ASR, names, downloads or model registry."""
from __future__ import annotations

import array
import ctypes as C
import json
import math
import os
from pathlib import Path
import sys
import time
import wave

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.io_utils import file_hash, write_json
from workbench.nemotron_probabilities import ProbabilityArchive


class ModelConfig(C.Structure):
    _fields_ = [("size", C.c_size_t), ("model_path", C.c_char_p), ("gpu", C.c_int32),
        ("preset", C.c_char_p), ("chunk_frames", C.c_int32), ("right_context_frames", C.c_int32),
        ("left_context_frames", C.c_int32), ("fifo_frames", C.c_int32),
        ("spkcache_frames", C.c_int32), ("update_period_frames", C.c_int32)]


class SegmentationConfig(C.Structure):
    _fields_ = [("size", C.c_size_t), ("onset", C.c_float), ("offset", C.c_float),
        ("pad_onset_sec", C.c_double), ("pad_offset_sec", C.c_double),
        ("min_gap_sec", C.c_double), ("min_duration_sec", C.c_double)]


class NativeSegment(C.Structure):
    _fields_ = [("start_time", C.c_double), ("end_time", C.c_double), ("speaker", C.c_int32)]


def bind(library):
    handle, fptr = C.c_void_p, C.POINTER(C.c_float)
    prototypes = {
        "nemo_speech_diar_create": ([C.POINTER(ModelConfig), C.POINTER(handle)], C.c_int),
        "nemo_speech_diar_destroy": ([handle], None),
        "nemo_speech_diar_num_speakers": ([handle], C.c_int32),
        "nemo_speech_diar_seconds_per_frame": ([handle], C.c_double),
        "nemo_speech_diar_stream_open": ([handle, C.POINTER(handle)], C.c_int),
        "nemo_speech_diar_stream_push_f32": ([handle, fptr, C.c_size_t, C.c_int32], C.c_int),
        "nemo_speech_diar_stream_finish": ([handle], C.c_int),
        "nemo_speech_diar_stream_close": ([handle], None),
        "nemo_speech_diar_frame_count": ([handle], C.c_int64),
        "nemo_speech_diar_frame_probs_start": ([handle], C.c_int64),
        "nemo_speech_diar_frame_probs": ([handle, fptr, C.c_size_t], C.c_int),
        "nemo_speech_diar_segments": ([handle, C.POINTER(SegmentationConfig), C.POINTER(NativeSegment), C.c_size_t, C.POINTER(C.c_size_t)], C.c_int),
        "nemo_speech_asr_last_error": ([], C.c_char_p),
    }
    for name, (args, result) in prototypes.items():
        function = getattr(library, name)
        function.argtypes, function.restype = args, result
    return library


def infer(request):
    start = time.monotonic()
    output = Path(request["output"])
    progress = Path(request["progress"])
    def state(phase, **data):
        write_json(progress, {"phase": phase, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - start, **data})
    state("library")
    if file_hash(Path(__file__)) != request["worker_sha256"]:
        raise ValueError("native worker changed after request freeze")
    if request.get("timeline_sha256") and file_hash(Path(__file__).with_name("nemotron_timeline.py")) != request["timeline_sha256"]:
        raise ValueError("native timeline validator changed after request freeze")
    if request.get("probabilities_code_sha256") and file_hash(Path(__file__).with_name("nemotron_probabilities.py")) != request["probabilities_code_sha256"]:
        raise ValueError("probability archive implementation changed after request freeze")
    if file_hash(Path(request["audio_path"])) != request["audio_sha256"] or file_hash(Path(request["model_path"])) != request["model_sha256"]:
        raise ValueError("audio/model bytes changed before native load")
    lib = bind(C.CDLL(request["library"]))
    def checked(status):
        if status:
            error = lib.nemo_speech_asr_last_error()
            raise RuntimeError("native diarization: " + (error.decode(errors="replace") if error else str(status)))
    model, stream = C.c_void_p(), C.c_void_p()
    long_context = request.get("long_context", False)
    if type(long_context) is not bool:
        raise ValueError("explicit long_context boolean required")
    retain_full = request.get("retain_full_probabilities", False)
    if type(retain_full) is not bool or (retain_full and not long_context):
        raise ValueError("full probability export requires explicit long mode")
    archive = None
    checkpoints = []
    config = ModelConfig(size=C.sizeof(ModelConfig), model_path=os.fsencode(request["model_path"]),
                         gpu=0, preset=b"v3-offline", left_context_frames=-1)
    try:
        state("loading")
        checked(lib.nemo_speech_diar_create(C.byref(config), C.byref(model)))
        loaded = time.monotonic()
        slots = lib.nemo_speech_diar_num_speakers(model)
        cadence = lib.nemo_speech_diar_seconds_per_frame(model)
        if slots != 8 or not math.isclose(cadence, 0.01, abs_tol=1e-8):
            raise ValueError("unexpected native capacity/cadence")
        checked(lib.nemo_speech_diar_stream_open(model, C.byref(stream)))
        if retain_full:
            archive = ProbabilityArchive(output.with_suffix(".all-probabilities.f32"))
        def export_probabilities(*, force=False):
            if archive is None:
                return
            current = lib.nemo_speech_diar_frame_count(stream)
            if current == archive.frames and not force:
                return
            first_retained = lib.nemo_speech_diar_frame_probs_start(stream)
            size = (current - first_retained) * slots
            if size < 0:
                raise ValueError("invalid native probability retention")
            retained = (C.c_float * size)()
            checked(lib.nemo_speech_diar_frame_probs(stream, retained, size))
            archive.update(first_frame=first_retained, frame_count=current,
                retained_bytes=C.string_at(C.addressof(retained), size * C.sizeof(C.c_float)))
        def native_segments():
            size = C.c_size_t()
            checked(lib.nemo_speech_diar_segments(stream, None, None, 0, C.byref(size)))
            if size.value > 100000:
                raise ValueError("too many native segments")
            segments = (NativeSegment * size.value)()
            checked(lib.nemo_speech_diar_segments(stream, None, segments, len(segments), C.byref(size)))
            return [{"start": s.start_time, "end": s.end_time, "speaker": s.speaker} for s in segments[:size.value]]
        def checkpoint(pushed):
            folder = output.parent / "checkpoints"
            folder.mkdir(exist_ok=True)
            path = folder / f"{len(checkpoints):04d}.json"
            if path.exists():
                raise ValueError("refusing to overwrite native checkpoint")
            write_json(path, {"frame_count": lib.nemo_speech_diar_frame_count(stream),
                "retained_frame_start": lib.nemo_speech_diar_frame_probs_start(stream),
                "input_samples_consumed": pushed, "segments": native_segments(),
                "scope": "native output snapshot; not serialized model state"})
            checkpoints.append({"path": str(path.resolve()), "sha256": file_hash(path)})
        with wave.open(request["audio_path"], "rb") as audio:
            rate, frames = audio.getframerate(), audio.getnframes()
            if (audio.getnchannels(), audio.getsampwidth(), rate, audio.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("requires mono16k PCM16")
            duration = frames / rate
            if not 0 < duration <= (14400 if long_context else 300):
                raise ValueError("candidate duration exceeds explicit bounded probe mode")
            pushed = 0
            previous_first, next_checkpoint, next_progress = 0, 300 * rate, 0
            while raw := audio.readframes(8000):
                pcm = array.array("h", raw)
                if sys.byteorder != "little":
                    pcm.byteswap()
                values = array.array("f", (v / 32768 for v in pcm))
                pointer = (C.c_float * len(values)).from_buffer(values)
                if not long_context or pushed >= next_progress:
                    state("inference", pushed_seconds=pushed / rate, frame_count=lib.nemo_speech_diar_frame_count(stream),
                          retained_frame_start=lib.nemo_speech_diar_frame_probs_start(stream))
                    next_progress = pushed + 5 * rate
                checked(lib.nemo_speech_diar_stream_push_f32(stream, pointer, len(values), rate))
                pushed += len(values)
                first_now = lib.nemo_speech_diar_frame_probs_start(stream)
                if first_now < previous_first:
                    raise ValueError("native retained clock went backwards")
                export_probabilities(force=first_now != previous_first)
                if long_context and (first_now != previous_first or pushed >= next_checkpoint):
                    checkpoint(pushed)
                    next_checkpoint = pushed + 300 * rate
                    state("inference", pushed_seconds=pushed / rate, frame_count=lib.nemo_speech_diar_frame_count(stream),
                          retained_frame_start=first_now)
                previous_first = first_now
            if pushed != frames:
                raise ValueError("audio truncated while streaming")
        state("finishing", pushed_seconds=duration)
        checked(lib.nemo_speech_diar_stream_finish(stream))
        inferred = time.monotonic()
        export_probabilities(force=True)
        count = lib.nemo_speech_diar_frame_count(stream)
        first = lib.nemo_speech_diar_frame_probs_start(stream)
        if not 0 <= first < count or (not long_context and first != 0) or not 0 < count <= math.ceil(duration / cadence) + 16:
            raise ValueError("unexpected native frame coverage/compaction")
        n = (count - first) * slots
        probabilities = (C.c_float * n)()
        checked(lib.nemo_speech_diar_frame_probs(stream, probabilities, n))
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities):
            raise ValueError("nonfinite/out-of-range native probabilities")
        # NULL uses exact library defaults, recorded alongside pinned code;
        # no cherry-picked new thresholds are applied to this probe.
        turns = native_segments()
        if long_context:
            checkpoint(pushed)
        probability_path = output.with_suffix(".probabilities.f32")
        with probability_path.open("wb") as f:
            f.write(C.string_at(C.addressof(probabilities), n * C.sizeof(C.c_float)))
            f.flush()
            os.fsync(f.fileno())
        probability_path.chmod(0o600)
        result = {"schema_version": 2 if long_context else 1, "artifact_kind": "nemotron-native-candidate", "backend_requested": "metal",
            "gpu_index_requested": 0, "preset": "v3-offline", "full_attention": False,
            "duration_seconds": duration, "origin_ms": request["origin_ms"], "speaker_slot_capacity": slots,
            "frame_count": count, "retained_frame_start": first, "seconds_per_frame": cadence,
            "probability_path": str(probability_path), "probability_sha256": file_hash(probability_path),
            "probability_format": "little-endian-float32-frame-major-8-channels", "segments": turns,
            "observed_active_channels": sorted({s["speaker"] for s in turns}),
            "segmentation": "pinned-native-library-defaults", "resume_granularity": "whole-stage",
            "postprocessing_source_literals": {"onset": 0.641, "offset": 0.561, "pad_onset_seconds": 0.229,
                "pad_offset_seconds": 0.079, "min_duration_on_seconds": 0.511, "min_duration_off_seconds": 0.296},
            "timings": {"load_seconds": loaded - start, "stream_and_finish_seconds": inferred - loaded,
                        "worker_seconds": time.monotonic() - start}, "quality_validation": "not-measured"}
        if long_context:
            result.update(timeline_protocol="native-retained-tail-v2", probability_scope="retained-tail",
                segments_scope="native-whole-timeline", completed=True, input_samples_consumed=pushed,
                retained_probability_frames=count-first, checkpoints=checkpoints)
        if archive is not None:
            result["full_probability_archive"] = archive.finish(frame_count=count)
        write_json(output, result)
        state("completed", frame_count=count)
    finally:
        if archive is not None:
            archive.close()
        if stream.value:
            lib.nemo_speech_diar_stream_close(stream)
        if model.value:
            lib.nemo_speech_diar_destroy(model)


if __name__ == "__main__":
    infer(json.loads(Path(sys.argv[1]).read_text()))
