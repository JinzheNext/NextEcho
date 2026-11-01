"""Bounded audio measurements with private, hash-checked voice prototypes."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .io_utils import file_hash, value_hash, write_json
from .voice_linking import link_voice_prototypes


def bounded_diarization(request: dict, clip: dict) -> dict:
    import numpy as np
    import sherpa_onnx
    import soundfile as sf
    from .diarization import _build_sherpa_diarizer, _run_sherpa_diarizer

    seconds = int(request.get("chunk_seconds", 600))
    if not 60 <= seconds <= 600:
        raise ValueError("Acoustic core chunks must be between 60 and 600 seconds")
    threads = int(request.get("num_threads", 4))
    if not 1 <= threads <= 4:
        raise ValueError("Acoustic threads must be between 1 and 4")
    model_paths = [Path(p) for p in request["model_paths"]]
    diarizer = _build_sherpa_diarizer(sherpa_onnx, {"num_threads": threads,
        "num_clusters": request.get("num_clusters", -1), "cluster_threshold": request.get("cluster_threshold", .5)},
        model_paths, 1, 8)
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_paths[1]), num_threads=threads))
    folder = Path(request["output"]).parent / "acoustic-chunks"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress = folder.parent / "diarize-global.progress.json"
    measurements, hits = [], []
    with sf.SoundFile(clip["audio"]) as audio:
        rate, frames = audio.samplerate, len(audio)
        if rate != 16000 or audio.channels != 1:
            raise ValueError("Bounded diarization requires mono 16k WAV")
        total = math.ceil(frames / (seconds*rate))
        for index, core_start in enumerate(range(0, frames, seconds*rate)):
            core_end = min(frames, core_start + seconds*rate)
            left, right = max(0, core_start-2*rate), min(frames, core_end+2*rate)
            signature = value_hash([request["measurement_signature"], left, right, core_start, core_end])
            path, checksum = folder / f"{index:04d}.json", folder / f"{index:04d}.sha256"
            cached = None
            try:
                if checksum.read_text().strip() == file_hash(path):
                    data = json.loads(path.read_text())
                    if data.get("signature") == signature:
                        cached = data
            except (OSError, ValueError, TypeError):
                pass
            if cached is not None:
                measurements.append(cached)
                hits.append(index)
            else:
                audio.seek(left)
                samples = audio.read(right-left, dtype="float32")
                scratch = folder / "current.wav"
                sf.write(str(scratch), samples, rate, subtype="PCM_16")
                try:
                    local = _run_sherpa_diarizer(diarizer, scratch)
                finally:
                    scratch.unlink(missing_ok=True)
                turns = []
                for turn in local:
                    start, end = max(core_start/rate, left/rate+turn.start), min(core_end/rate, left/rate+turn.end)
                    if end > start:
                        turns.append({"start": start, "end": end, "speaker": turn.raw_label,
                                      "raw_speaker_label": turn.raw_label, "overlap": turn.overlap})
                prototypes = []
                for label in dict.fromkeys(t.raw_label for t in local):
                    # Sample only exclusive interior audio, never mixed voices.
                    candidates = [t for t in local if t.raw_label == label and t.end-t.start >= 1.5 and not t.overlap
                        and not any(o.raw_label != label and min(t.end,o.end)>max(t.start,o.start) for o in local)]
                    pieces = []
                    for turn in sorted(candidates, key=lambda t: t.end-t.start, reverse=True)[:3]:
                        length = min(4., turn.end-turn.start-.2)
                        start = (turn.start+turn.end-length)/2
                        pieces.append(samples[round(start*rate):round((start+length)*rate)])
                    embedding = []
                    if pieces:
                        stream = extractor.create_stream()
                        stream.accept_waveform(rate, np.concatenate(pieces))
                        stream.input_finished()
                        if extractor.is_ready(stream):
                            vector = np.asarray(extractor.compute(stream), dtype=np.float32)
                            norm = float(np.linalg.norm(vector))
                            if norm > 0 and np.all(np.isfinite(vector)):
                                embedding = (vector/norm).tolist()
                    own = [t for t in turns if t["speaker"] == label]
                    if own:
                        prototypes.append({"key": f"{index:04d}:{label}", "chunk": index,
                            "local_label": label, "first": min(t["start"] for t in own), "embedding": embedding})
                measured = {"signature": signature, "index": index, "core_start": core_start/rate,
                    "core_end": core_end/rate, "turns": turns, "prototypes": prototypes}
                write_json(path, measured)
                checksum.write_text(file_hash(path))
                checksum.chmod(0o600)
                measurements.append(measured)
            write_json(progress, {"stage": "speaker", "completed_chunks": index+1, "total_chunks": total,
                                  "cached_chunks": len(hits), "core_seconds_completed": core_end/rate})
    nodes = [p for measurement in measurements for p in measurement["prototypes"]]
    mapping, audit = link_voice_prototypes(nodes, threshold=float(request.get("global_voice_threshold", .8)))
    turns = []
    for measurement in measurements:
        for turn in measurement["turns"]:
            key = f"{measurement['index']:04d}:{turn['speaker']}"
            label = mapping[key]
            turns.append({**turn, "speaker": label, "raw_speaker_label": label, "source_acoustic_key": key})
    turns.sort(key=lambda t: (t["start"], t["end"], t["speaker"]))
    return {"turns": turns, "backend": "sherpa-bounded-acoustic-prototypes", "partial": True,
            "global_linking": audit, "chunk_count": len(measurements), "cached_chunks": hits,
            "audio_seconds": frames/rate, "quality_status": "human_review_pending"}
