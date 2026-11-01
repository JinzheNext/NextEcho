"""Qwen原稿强制对齐；仅由aligner_candidate受控启动，离线载入。"""
from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.io_utils import file_hash, value_hash, write_json


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    if file_hash(Path(__file__)) != request["worker_sha256"]:
        raise ValueError("Frozen alignment worker changed")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if file_hash(Path(request["audio"])) != request["audio_sha256"] or value_hash(request["text"]) != request["text_sha256"]:
        raise ValueError("Frozen alignment input changed")
    started = time.monotonic()
    import torch
    import soundfile as sf
    from transformers import AutoProcessor, AutoModelForTokenClassification
    from transformers.utils import logging

    logging.disable_progress_bar()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    state = Path(request["output"]).with_name("worker.state.json")
    write_json(state, {"phase": "loading"})
    processor = AutoProcessor.from_pretrained(request["model_dir"], local_files_only=True, trust_remote_code=False)
    model = AutoModelForTokenClassification.from_pretrained(
        request["model_dir"], local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation=request["attention"])
    model.to("cpu").eval()
    loaded = time.monotonic()
    samples, sr = sf.read(request["audio"], dtype="float32")
    if sr != 16000 or samples.ndim != 1 or len(samples) > 300 * sr:
        raise ValueError("Expected bounded mono16k source audio")
    inputs, words = processor.prepare_forced_aligner_inputs(
        audio=samples, transcript=request["text"], language="Chinese", return_tensors="pt")
    inputs = inputs.to(model.device, model.dtype)
    prepared = time.monotonic()
    write_json(state, {"phase": "inference", "prepared_at_monotonic": prepared})
    with torch.inference_mode():
        # 单次分类不生成后续token，不保留生成用的KV缓存。
        outputs = model(**inputs, use_cache=False)
    inferred = time.monotonic()
    timestamps = processor.decode_forced_alignment(
        logits=outputs.logits, input_ids=inputs["input_ids"], word_lists=words,
        timestamp_token_id=model.config.timestamp_token_id)[0]
    write_json(Path(request["output"]), {"schema_version": 1, "audio_sha256": request["audio_sha256"],
        "text_sha256": request["text_sha256"], "model": request["model"], "timestamps": timestamps,
        "runtime": {"python": sys.version.split()[0], "device": "cpu", "dtype": "float32", "threads": 4,
            "attention": request["attention"],
            "use_cache": False,
            "implementation_sha256": {name: file_hash(Path(inspect.getfile(cls)))
                for name, cls in (("processor", type(processor)), ("model", type(model)))},
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "soundfile")}},
        "timings": {"cold_initialization_seconds": loaded - started,
                    "preparation_seconds": prepared - loaded,
                    "inference_seconds": inferred - prepared,
                    "total_worker_seconds": time.monotonic() - started}})
    write_json(state, {"phase": "completed"})


if __name__ == "__main__":
    main()
