"""One model load per bounded audio job; complete short clips keep context."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.audio_segments import merge_recognized_segments, split_window
from workbench.io_utils import write_json


def _watch_parent():
    supervisor = int(os.environ.get("NEXTECHO_SUPERVISOR_PID", "0"))
    if not supervisor:
        return
    def watch():
        while True:
            try:
                os.kill(supervisor, 0)
            except ProcessLookupError:
                os._exit(70)
            time.sleep(1)
    threading.Thread(target=watch, daemon=True).start()


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request.get("schema_version") != 1:
        raise ValueError("Unsupported ASR request version")
    _watch_parent()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import soundfile as sf
    import sherpa_onnx

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = request["vad_model"]
    cfg.silero_vad.min_silence_duration = .4
    cfg.silero_vad.min_speech_duration = .15
    cfg.silero_vad.max_speech_duration = 29
    cfg.sample_rate = 16000
    cfg.num_threads = 1
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    root, engine = Path(request["model"]), request["engine"]
    rows, windows, decode_seconds = [], [], []
    model, load_seconds, load_count = None, 0, 0
    with tempfile.TemporaryDirectory(prefix="nextecho-asr-") as scratch, sf.SoundFile(request["audio"]) as audio, sf.SoundFile(request["audio"]) as reader:
        if audio.samplerate != 16000 or audio.channels != 1:
            raise ValueError("Expected 16kHz mono normalized audio")
        frames = len(audio)
        def recognize(left, right):
            nonlocal model, load_seconds, load_count
            if model is None:
                started = time.monotonic()
                if engine == "fun-asr-nano":
                    from funasr import AutoModel
                    model = AutoModel(model=str(root), hub="hf", device="cpu", trust_remote_code=True,
                                      disable_update=True, disable_pbar=True, disable_log=True)
                else:
                    model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                        model=str(root / "model.int8.onnx"), tokens=str(root / "tokens.txt"),
                        num_threads=4, use_itn=True, language=request["language"], debug=False)
                load_seconds = time.monotonic() - started
                load_count += 1
            started = time.monotonic()
            reader.seek(left)
            samples = reader.read(right - left, dtype="float32")
            if engine == "fun-asr-nano":
                wav = Path(request["audio"])
                if left != 0 or right != frames:
                    wav = Path(scratch) / "window.wav"
                    sf.write(str(wav), samples, 16000)
                language = {"zh": "简体中文" if request.get("output_script") == "simplified" else "中文", "en": "英文", "ja": "日文"}.get(request["language"])
                options = {"language": language} if language else {}
                if request.get("hotwords"):
                    options["hotwords"] = request["hotwords"]
                result = model.generate(input=[str(wav)], cache={}, batch_size=1, **options)
                text = (result[0].get("text") or "").strip() if result else ""
            else:
                stream = model.create_stream()
                stream.accept_waveform(16000, samples)
                model.decode_stream(stream)
                text = stream.result.text.strip()
            decode_seconds.append(time.monotonic() - started)
            return text
        previous_core_end = None
        def emit(start, end, *, whole_clip=False):
            nonlocal previous_core_end
            for window in split_window(start, end, max_seconds=30 if whole_clip else 29):
                # Actual sample bounds, with <=30s expansion. Keep whole short
                # utterances intact rather than decoding subsecond VAD pieces.
                context = 8000 if previous_core_end == window.start_sample else 0
                left = window.start_sample if whole_clip else max(0, window.start_sample - 3200 - context)
                right = window.end_sample if whole_clip else min(frames, window.end_sample + 3200)
                bounds = {"raw_start_sample": left, "raw_end_sample": right,
                          "retain_start_sample": window.start_sample, "retain_end_sample": window.end_sample}
                windows.append(bounds)
                rows.append({"text": recognize(left, right), "start_ms": round(left / 16), "end_ms": round(right / 16),
                             "retain_start_ms": window.start_ms, "retain_end_ms": window.end_ms})
                if request.get("progress"):
                    write_json(Path(request["progress"]), {"stage": "asr", "completed_windows": len(rows),
                               "processed_seconds": window.end_sample / 16000, "duration_seconds": frames / 16000,
                               "elapsed_decode_seconds": sum(decode_seconds)})
                previous_core_end = window.end_sample
        pending, found = None, False
        short_clip = frames <= 30 * 16000
        def drain():
            nonlocal pending, found
            while not vad.empty():
                speech = vad.front
                start, end = int(speech.start), min(frames, int(speech.start) + len(speech.samples))
                vad.pop()
                if start >= end:
                    continue
                found = True
                if short_clip:
                    continue
                if pending and start - pending[1] <= 19200 and end - pending[0] <= 29 * 16000:
                    pending = (pending[0], end)
                else:
                    if pending:
                        emit(*pending)
                    pending = (start, end)
        for block in audio.blocks(blocksize=512, dtype="float32", always_2d=False):
            vad.accept_waveform(block)
            drain()
        vad.flush()
        drain()
        if short_clip and found:
            emit(0, frames, whole_clip=True)
        elif pending:
            emit(*pending)
    segments, review = merge_recognized_segments(rows)
    versions = {}
    for name in ("sherpa-onnx", "funasr", "torch", "soundfile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak_rss *= 1024
    write_json(Path(request["output"]), {"schema_version": 1, "text": "\n".join(r["text"] for r in segments),
               "peak_worker_rss_bytes": peak_rss, "segments": segments, "runtime_versions": versions,
               "version": versions.get("funasr" if engine == "fun-asr-nano" else "sherpa-onnx", ""),
               "timings": {"model_load_count": load_count, "model_initialization_seconds": load_seconds,
                           "first_window_seconds": decode_seconds[0] if decode_seconds else None,
                           "warm_window_seconds": decode_seconds[1:]},
               "recognition_prompt": {"output_script": request.get("output_script", "original"),
                                      "hotwords": request.get("hotwords", [])},
               "windows": windows, "review": review, "coverage": {"covered_seconds": frames / 16000, "partial": False,
               "content_state": "speech" if segments else "no_speech"}})


if __name__ == "__main__":
    main()
