"""Acoustic-turn review worker. Always launch through resources.run_owned.

Turns come from audio diarization. ASR is run on each actual voice interval,
so a coarse transcript never needs invented word times or text-based speakers.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.io_utils import write_json


def exclusive_speaker_windows(turns):
    """Sweep true acoustic boundaries; overlapping voices stay explicitly mixed."""
    boundaries = sorted({float(t[key]) for t in turns for key in ("start", "end")})
    output = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        middle = (start + end) / 2
        active = [t for t in turns if t["start"] <= middle < t["end"]]
        labels = sorted({t["speaker"] for t in active})
        if not labels:
            continue
        speaker = labels[0] if len(labels) == 1 else "交叠发言（" + "＋".join(labels) + "）"
        row = {"start": start, "end": end, "speaker": speaker,
               "acoustic_label": "+".join(sorted({t["acoustic_label"] for t in active})),
               "overlap": len(labels) > 1}
        if output and output[-1]["speaker"] == speaker and start - output[-1]["end"] < .15:
            output[-1]["end"] = end
        else:
            output.append(row)
    return output


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import soundfile as sf
    if request["operation"] in {"diarize", "diarize-global"}:
        from workbench.diarization import DiarizationOptions, create_diarization_backend
        results = []
        for clip in request["clips"]:
            if request["operation"] == "diarize-global" and request.get("strategy") == "bounded-prototypes-v1":
                from workbench.acoustic_chunks import bounded_diarization
                started = time.monotonic()
                result = bounded_diarization(request, clip)
                results.append({**clip, "diarization": result, "elapsed_seconds": time.monotonic()-started})
                continue
            legacy = request["operation"] == "diarize"
            options = DiarizationOptions(min_speakers=2 if legacy else 1, max_speakers=2 if legacy else 8,
                backend_config={"model_paths": request["model_paths"],
                    "num_clusters": 2 if legacy else int(request.get("num_clusters", -1)),
                    "cluster_threshold": float(request.get("cluster_threshold", 0.5))})
            started = time.monotonic()
            result = create_diarization_backend("sherpa-onnx").diarize(Path(clip["audio"]), options)
            results.append({**clip, "diarization": result.to_dict(), "elapsed_seconds": time.monotonic() - started})
        write_json(Path(request["output"]), {"schema_version": 1, "clips": results})
        return
    if request["operation"] == "ctc":
        import sherpa_onnx
        root = Path(request["model"])
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(root / "model.int8.onnx"), tokens=str(root / "tokens.txt"),
            num_threads=4, use_itn=True, language="zh", debug=False)
        results = []
        for clip in request["clips"]:
            tokens = []
            with sf.SoundFile(clip["audio"]) as audio:
                if audio.samplerate != 16000 or audio.channels != 1:
                    raise ValueError("CTC anchors require mono 16k audio")
                frames = len(audio)
                for start in range(0, frames, 29 * 16000):
                    left = max(0, start - 8000)
                    audio.seek(left)
                    samples = audio.read(min(frames, start + 29 * 16000) - left, dtype="float32")
                    stream = recognizer.create_stream()
                    stream.accept_waveform(16000, samples)
                    recognizer.decode_stream(stream)
                    result = stream.result
                    durations = list(result.durations)
                    for index, (token, at) in enumerate(zip(result.tokens, result.timestamps)):
                        onset = float(at) + left / 16000
                        if start and onset < start / 16000:
                            continue
                        duration = float(durations[index]) if len(durations) == len(result.tokens) else None
                        if str(token).startswith('<|'):
                            continue
                        tokens.append({"text": str(token).replace('▁', ' '), "start_ms": round(onset * 1000),
                            "end_ms": round((onset + duration) * 1000) if duration is not None else None,
                            "native_duration": duration is not None})
            results.append({"id": clip["id"], "tokens": tokens})
        write_json(Path(request["output"]), {"schema_version": 1, "clips": results,
            "timing_source": "SenseVoice native CTC token timestamps/durations; not interpolated word times"})
        return
    if request["operation"] != "asr":
        raise ValueError("Unknown dialogue operation")
    from funasr import AutoModel
    from workbench.audio_segments import merge_recognized_segments
    started = time.monotonic()
    model = AutoModel(model=request["model"], hub="hf", device="cpu", trust_remote_code=True,
                      disable_update=True, disable_pbar=True, disable_log=True)
    loaded = time.monotonic() - started
    results = []
    scratch = Path(request["output"]).parent / "turn.wav"
    for clip in request["clips"]:
        rows = []
        with sf.SoundFile(clip["audio"]) as audio:
            if audio.samplerate != 16000 or audio.channels != 1:
                raise ValueError("Expected original mono16k audio")
            for index, turn in enumerate(clip["turns"]):
                left, right = round(turn["start"] * 16000), min(len(audio), round(turn["end"] * 16000))
                if right <= left:
                    continue
                windows = []
                for start in range(left, right, 29 * 16000):
                    end = min(right, start + 29 * 16000)
                    # Context stays inside this speaker's acoustic interval.
                    begin = max(left, start - 3200)
                    audio.seek(begin)
                    sf.write(str(scratch), audio.read(end - begin, dtype="float32"), 16000)
                    response = model.generate(input=[str(scratch)], cache={}, batch_size=1, language="中文")
                    text = response[0].get("text", "").strip() if response else ""
                    windows.append({"text": text, "start_ms": round(begin / 16), "end_ms": round(end / 16),
                                    "retain_start_ms": round(start / 16), "retain_end_ms": round(end / 16)})
                segments, review = merge_recognized_segments(windows)
                text = "".join(s["text"] for s in segments)
                if text:
                    rows.append({"id": f"{clip['id']}-turn-{index + 1:03}", "text": text,
                        "start_ms": round(left / 16), "end_ms": round(right / 16), "speaker": turn["speaker"],
                        "acoustic_label": turn["acoustic_label"], "overlap": turn.get("overlap", False),
                        "asr_windows": windows, "review": review})
        results.append({**clip, "segments": rows})
        write_json(Path(request["output"]), {"schema_version": 1, "clips": results,
            "timing_precision": "acoustic-turn-window", "model_load_count": 1,
            "model_initialization_seconds": loaded, "elapsed_seconds": time.monotonic() - started})
    scratch.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
